//! Pure-Rust AMD RDNA2 (GFX10.3 / Xbox Series Scarlett) -> WebGPU WGSL shader translator
//! with built-in Naga validation.
//!
//! Decodes RDNA2 machine-code instruction streams via [`crate::rdna2`], lowers the
//! register-machine ISA (`VGPR`, `SGPR`, `VCC`, `SCC`, `SMEM`, `VINTRP`, `MIMG`, `EXP`,
//! `VALU`, `SALU`) into a baseline-WebGPU [`naga::Module`], validates it unconditionally
//! with [`naga::valid::Validator`] (`Capabilities::empty()`), and emits clean WGSL text
//! via [`naga::back::wgsl::write_string`].
//!
//! Guarantees:
//! * Zero `unsafe` code.
//! * Panic-free on empty, misaligned, truncated, or arbitrary fuzz inputs.
//! * Strictly sanitized outputs (no host paths, no raw shader byte dumps).
//! * Graceful fallback for unsupported opcodes: emits sanitized WGSL comments and
//!   increments `unsupported_op_count` while keeping the generated WGSL valid under Naga.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::num::NonZeroU32;

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, BinaryOperator, Binding, BuiltIn, EntryPoint, Expression, Function,
    FunctionArgument, FunctionResult, GlobalVariable, Handle, ImageClass, ImageDimension,
    Interpolation, Literal, LocalVariable, MathFunction, MemoryDecorations, Module, Range,
    ResourceBinding, SampleLevel, Sampling, Scalar, ScalarKind, ShaderStage as NagaStage, Span,
    Statement, StorageAccess, StorageFormat, StructMember, SwitchCase, SwitchValue,
    SwizzleComponent, Type, TypeInner, UnaryOperator, VectorSize,
};

use crate::convert::{ShaderStage, format_shader_hash};
use crate::pack::write_json_string;
use crate::rdna2::{
    Rdna2Detail, Rdna2ExportTarget, Rdna2InferredStage, Rdna2Instruction, Rdna2ProgramSummary,
    analyze_instructions, decode_program, decode_vop3_interp_attr, sop2_src_pair_widths,
    sopc_src_pair_widths, vop2_is_co_ci, vop2_is_madk, vop3_is_interp_f16,
};
use crate::rdna2_cfg::{
    BlockId, BranchKind, ControlFlowGraph, DISPATCH_EXIT_PC, ExecMaskOp, ExecMaskWrite,
    STRUCTURED_LOOP_MAX_ITERATIONS, StructuredNode, Terminator, classify_exec_mask_op,
    structure_cfg,
};

/// Default minimum number of `vec4<u32>` slots in a statically-sized uniform CBV (`cb0`, `cb1`, ...).
/// Tightly sized to `1` (`16` bytes) so small Xbox Series CBVs (e.g. 48 B / 96 B) are never
/// over-declared as 256 B uniform blocks in WebGPU.
pub const DEFAULT_MIN_CBV_VEC4_COUNT: u32 = 1;

/// Minimum number of `vec4<u32>` slots (`256` bytes) emitted for a CBV that uses a dynamic/unknown
/// `soffset` where no upper bound is statically known.
pub const DEFAULT_DYNAMIC_CBV_VEC4_COUNT: u32 = 16;

/// Upper bound on `vec4<u32>` slots in a single CBV (1024 * 16B = 16 KiB, within WebGPU 64 KiB baseline).
pub const MAX_CBV_VEC4_COUNT: u32 = 1024;

/// Maximum inter-stage location count under WebGPU baseline limits.
const MAX_WEBGPU_LOCATIONS: u8 = 16;

/// Maximum color attachment count under WebGPU baseline limits.
const MAX_WEBGPU_MRTS: u8 = 8;

/// WebGPU texture format enum code for `rgba8unorm` (`pso::WgpuTextureFormat::Rgba8Unorm`,
/// `docs/gpu_commands.md` / `runtime/gpu/stage_reflection.h`).
pub const WGPU_STORAGE_FORMAT_RGBA8_UNORM: u8 = 22;

/// Binding number offset applied to reflected fragment-stage bindings in `@group(0)` so
/// independently translated vertex (`0..`) and fragment (`32..`) stages never collide on the
/// same `@group(0) @binding(N)` with different descriptor-table sources during `join_pipeline_layout`.
const FRAGMENT_REFLECTED_BINDING_OFFSET: u32 = 32;

/// First extra binding index (`cb1`, `tex1`, `samp1`, `sbuf*`, `uav_tex*`) for unreflected
/// fragment stages, kept disjoint from vertex extra bindings (`3..15`) so `layout:'auto'`
/// never hits a vertex-vs-fragment binding type mismatch (e.g. buffer vs texture at `@binding(4)`).
const FRAGMENT_EXTRA_BINDING_BASE: u32 = 16;

/// Returns `true` iff `f` can be emitted directly as a WGSL `Literal::F32` without
/// risking Naga decimal rounding overflow past `f32::MAX` (which Chrome/Dawn Tint
/// rejects as unrepresentable), losing `-0.0`, or emitting subnormals / NaN / Inf.
///
/// Naga prints `f32::MAX` (`0x7f7fffff`, `3.4028234663852886e38`) rounded up to
/// `340282350000000000000000000000000000000f` (`3.4028235e38 > f32::MAX`), which
/// Tint rejects. Any finite float with `|f| >= 3.4e38` (including `f32::MAX`,
/// `-f32::MAX`, `0x7f7ffffe`, etc.) routes through `pg6_u32_bits_to_float(bits)`
/// for an exact IEEE-754 bitcast that Tint accepts unconditionally.
#[inline]
fn is_safe_wgsl_f32_literal(f: f32) -> bool {
    f.to_bits() == 0 || (f.is_finite() && f.is_normal() && f.abs() < 3.4e38_f32)
}

/// Stage override option for [`translate_rdna2_to_wgsl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Rdna2StageOverride {
    #[default]
    Auto,
    Vertex,
    Pixel,
    Compute,
}

impl Rdna2StageOverride {
    /// Parse a CLI stage flag value (`auto`, `vertex`, `pixel`, `fragment`, `compute`).
    pub fn parse_str(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "vertex" | "vs" => Ok(Self::Vertex),
            "pixel" | "fragment" | "ps" | "fs" => Ok(Self::Pixel),
            "compute" | "cs" => Ok(Self::Compute),
            other => Err(format!(
                "invalid --stage value '{other}' (expected 'auto', 'vertex', 'pixel', or 'compute')"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Vertex => "vertex",
            Self::Pixel => "pixel",
            Self::Compute => "compute",
        }
    }
}

impl fmt::Display for Rdna2StageOverride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default compute workgroup dimensions (`@workgroup_size(64, 1, 1)`).
pub const DEFAULT_COMPUTE_WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];

/// LDS workgroup shared memory size in 32-bit dwords (`4096 * 4 = 16 KiB`).
const LDS_WORKGROUP_DWORDS: u32 = 4096;

/// Mask applied to LDS dword indices (`4095 = 0x0FFF`) so workgroup memory accesses never trap.
const LDS_WORKGROUP_INDEX_MASK: u32 = LDS_WORKGROUP_DWORDS - 1;

/// Options controlling RDNA2 -> WGSL translation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2TranslateOptions {
    /// Pipeline stage selection (`Auto` infers from `EXP`/`VINTRP`/memory instructions).
    pub stage: Rdna2StageOverride,
    /// Optional explicit WGSL entry-point name (defaults to `vs_main`/`fs_main`/`cs_main`).
    pub entry_point: Option<String>,
    /// Minimum number of `vec4<u32>` entries emitted when `cb0` uniform buffer is needed.
    pub min_cbv_vec4_count: u32,
    /// Workgroup size `(X, Y, Z)` for `ShaderStage::Compute` (defaults to `[64, 1, 1]`).
    pub workgroup_size: [u32; 3],
    /// Optional base SGPR index for seeding `(wg_id.x, wg_id.y, wg_id.z)` in compute shaders.
    /// When `None`, defaults to `s0`/`s1`/`s2` for SGPRs read before being written.
    pub workgroup_id_sgpr_base: Option<u8>,
}

impl Default for Rdna2TranslateOptions {
    fn default() -> Self {
        Self {
            stage: Rdna2StageOverride::Auto,
            entry_point: None,
            min_cbv_vec4_count: DEFAULT_MIN_CBV_VEC4_COUNT,
            workgroup_size: DEFAULT_COMPUTE_WORKGROUP_SIZE,
            workgroup_id_sgpr_base: None,
        }
    }
}

/// Origin of a reflected binding in console user-data SGPRs or static samplers
/// (`runtime/gpu/stage_reflection.h` / `tools/stage_reflection_sidecar.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgslBindingSource {
    /// Descriptor loaded from a descriptor table whose 32-bit pointer lives in `user_sgpr`
    /// (s8+ for graphics, s4+ for compute) at `table_byte_offset` bytes into the table.
    Table {
        user_sgpr: u32,
        table_byte_offset: u32,
    },
    /// Root constants starting at `user_sgpr`.
    RootConstants { user_sgpr: u32 },
    /// Root CBV/SRV/UAV descriptor (64-bit GPU VA) starting at `user_sgpr`.
    RootDescriptor { user_sgpr: u32 },
    /// Inline static sampler S# materialized directly in SGPRs (0-based index in stage order;
    /// fallback when any of the 4 S# descriptor words is not statically known).
    StaticSampler { index: u32 },
    /// Inline GFX10 S# sampler whose 4 descriptor dwords are statically known constants (#482 / #950).
    InlineSampler { sdwords: [u32; 4] },
    /// SM 6.6 `SamplerDescriptorHeap[index]` (`SAMPLER_HEAP_DIRECTLY_INDEXED`, 0x800):
    /// 16-byte S# loaded from heap base in `base_user_sgpr` at byte offset derived from `index_user_sgpr`.
    SamplerHeap {
        base_user_sgpr: u32,
        stride: u32,
        index_user_sgpr: u32,
        index_shift: u32,
    },
    /// SM 6.6 `ResourceDescriptorHeap[index]` (`CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`, 0x400):
    /// V#/T# loaded from heap base in `base_user_sgpr` at byte offset derived from `index_user_sgpr`.
    ResourceHeap {
        base_user_sgpr: u32,
        stride: u32,
        index_user_sgpr: u32,
        index_shift: u32,
    },
}

/// Reflected vertex input attribute fetched by a vertex stage (`stage_reflection.h`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgslVertexInputInfo {
    pub location: u32,
    pub vb_slot: u32,
    pub byte_offset: u32,
    pub components: u8,
    pub component_type: &'static str,
    pub fetch_format: Option<&'static str>,
}

/// Explicit WebGPU `@group(group) @binding(binding)` resource declaration emitted in a
/// translated WGSL module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgslBindingInfo {
    pub group: u32,
    pub binding: u32,
    pub kind: &'static str,
    pub name: String,
    /// Texture view dimension (`"2d"`, `"2d-array"`, etc.) for texture/storage-texture bindings.
    pub view_dimension: Option<&'static str>,
    /// WebGPU texture format enum code for storage textures (e.g. `22` = `rgba8unorm`).
    pub storage_format: Option<u8>,
    /// Minimum WebGPU uniform buffer binding size in bytes (`Some(N * 16)` for `"uniform"`
    /// bindings, `None` for non-buffer bindings such as textures and samplers).
    pub min_binding_size_bytes: Option<u32>,
    /// Highest `vec4` index statically read from this uniform CBV (`Some(max_vec4_index)` for
    /// `"uniform"` bindings, `None` for non-buffer bindings).
    pub highest_vec4_offset: Option<u32>,
    /// Descriptor origin in console user-data SGPRs or static samplers when statically known.
    pub source: Option<WgslBindingSource>,
}

/// Write a JSON array of [`WgslBindingInfo`] entries with the given leading indentation.
pub(crate) fn write_wgsl_bindings_json(
    out: &mut String,
    bindings: &[WgslBindingInfo],
    indent: &str,
) {
    if bindings.is_empty() {
        out.push_str("[]");
        return;
    }
    out.push_str("[\n");
    for (i, b) in bindings.iter().enumerate() {
        let _ = write!(
            out,
            "{indent}  {{\"group\": {}, \"binding\": {}, \"kind\": ",
            b.group, b.binding
        );
        write_json_string(out, b.kind);
        out.push_str(", \"name\": ");
        write_json_string(out, &b.name);
        if let Some(dim) = b.view_dimension {
            out.push_str(", \"view_dimension\": ");
            write_json_string(out, dim);
        }
        if let Some(fmt) = b.storage_format {
            let _ = write!(out, ", \"storage_format\": {fmt}");
        }
        if let Some(min_bytes) = b.min_binding_size_bytes {
            let _ = write!(out, ", \"min_binding_size_bytes\": {min_bytes}");
        }
        if let Some(highest_vec4) = b.highest_vec4_offset {
            let _ = write!(out, ", \"highest_vec4_offset\": {highest_vec4}");
        }
        if let Some(src) = b.source {
            out.push_str(", \"source\": ");
            match src {
                WgslBindingSource::Table {
                    user_sgpr,
                    table_byte_offset,
                } => {
                    let _ = write!(
                        out,
                        "{{\"user_sgpr\": {user_sgpr}, \"table_byte_offset\": {table_byte_offset}}}"
                    );
                }
                WgslBindingSource::RootConstants { user_sgpr } => {
                    let _ = write!(
                        out,
                        "{{\"user_sgpr\": {user_sgpr}, \"root_constants\": true}}"
                    );
                }
                WgslBindingSource::RootDescriptor { user_sgpr } => {
                    let _ = write!(
                        out,
                        "{{\"user_sgpr\": {user_sgpr}, \"root_descriptor\": true}}"
                    );
                }
                WgslBindingSource::StaticSampler { index } => {
                    let _ = write!(out, "{{\"static_sampler\": {index}}}");
                }
                WgslBindingSource::InlineSampler { sdwords } => {
                    let _ = write!(
                        out,
                        "{{\"inline_sampler\": {{\"sdwords\": [{}, {}, {}, {}]}}}}",
                        sdwords[0], sdwords[1], sdwords[2], sdwords[3]
                    );
                }
                WgslBindingSource::SamplerHeap {
                    base_user_sgpr,
                    stride,
                    index_user_sgpr,
                    index_shift,
                } => {
                    let _ = write!(
                        out,
                        "{{\"sampler_heap\": {{\"base_user_sgpr\": {base_user_sgpr}, \"stride\": {stride}, \"index\": {{\"user_sgpr\": {index_user_sgpr}, \"shift\": {index_shift}}}}}}}"
                    );
                }
                WgslBindingSource::ResourceHeap {
                    base_user_sgpr,
                    stride,
                    index_user_sgpr,
                    index_shift,
                } => {
                    let _ = write!(
                        out,
                        "{{\"resource_heap\": {{\"base_user_sgpr\": {base_user_sgpr}, \"stride\": {stride}, \"index\": {{\"user_sgpr\": {index_user_sgpr}, \"shift\": {index_shift}}}}}}}"
                    );
                }
            }
        }
        out.push('}');
        if i + 1 < bindings.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
}

/// Write a JSON array of [`WgslVertexInputInfo`] entries with the given leading indentation.
pub(crate) fn write_wgsl_vertex_inputs_json(
    out: &mut String,
    inputs: &[WgslVertexInputInfo],
    indent: &str,
) {
    if inputs.is_empty() {
        out.push_str("[]");
        return;
    }
    out.push_str("[\n");
    for (i, v) in inputs.iter().enumerate() {
        let _ = write!(
            out,
            "{indent}  {{\"location\": {}, \"vb_slot\": {}, \"byte_offset\": {}, \"components\": {}, \"component_type\": ",
            v.location, v.vb_slot, v.byte_offset, v.components
        );
        write_json_string(out, v.component_type);
        if let Some(fmt) = v.fetch_format {
            out.push_str(", \"fetch_format\": ");
            write_json_string(out, fmt);
        }
        out.push('}');
        if i + 1 < inputs.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
}

/// Summary of WebGPU bindings and stage I/O synthesized for a translated RDNA2 shader.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rdna2BindingSummary {
    pub has_cbv: bool,
    pub cbv_vec4_count: u32,
    pub has_texture_2d: bool,
    pub has_sampler: bool,
    pub has_storage_buffer: bool,
    pub has_storage_texture: bool,
    pub has_lds: bool,
    pub vertex_input_mode: String,
    pub fragment_input_locations: Vec<u32>,
    pub output_locations: Vec<u32>,
    pub exports_position: bool,
    pub exports_depth: bool,
}

/// Result of translating an RDNA2 binary shader into validated WebGPU WGSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2WgslOutput {
    pub wgsl: String,
    pub stage: ShaderStage,
    pub entry_point: String,
    pub hash_fnv1a64: u64,
    pub translated_op_count: usize,
    pub unsupported_op_count: usize,
    pub naga_validated: bool,
    pub bindings: Vec<WgslBindingInfo>,
    pub vertex_inputs: Vec<WgslVertexInputInfo>,
    pub binding_summary: Rdna2BindingSummary,
}

impl Rdna2WgslOutput {
    /// Render a strictly sanitized deterministic JSON summary (zero host paths, zero raw bytes).
    pub fn to_json(&self) -> String {
        self.to_json_with_source(None)
    }

    /// Render a strictly sanitized deterministic JSON summary with an optional sanitized source basename.
    pub fn to_json_with_source(&self, source_name: Option<&str>) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"isa\": \"rdna2_gfx10_3\",\n");
        if let Some(src) = source_name {
            out.push_str("  \"source\": ");
            write_json_string(&mut out, src);
            out.push_str(",\n");
        }
        out.push_str("  \"hash_fnv1a64\": ");
        write_json_string(&mut out, &format_shader_hash(self.hash_fnv1a64));
        out.push_str(",\n");
        out.push_str("  \"stage\": ");
        write_json_string(&mut out, self.stage.as_str());
        out.push_str(",\n");
        out.push_str("  \"entry_point\": ");
        write_json_string(&mut out, &self.entry_point);
        out.push_str(",\n");
        let _ = writeln!(
            out,
            "  \"translated_op_count\": {},",
            self.translated_op_count
        );
        let _ = writeln!(
            out,
            "  \"unsupported_op_count\": {},",
            self.unsupported_op_count
        );
        let _ = writeln!(out, "  \"naga_validated\": {},", self.naga_validated);
        let _ = writeln!(out, "  \"wgsl_byte_len\": {},", self.wgsl.len());

        // Explicit WGSL @group/@binding map
        out.push_str("  \"bindings\": ");
        write_wgsl_bindings_json(&mut out, &self.bindings, "  ");
        out.push_str(",\n");

        if !self.vertex_inputs.is_empty() {
            out.push_str("  \"vertex_inputs\": ");
            write_wgsl_vertex_inputs_json(&mut out, &self.vertex_inputs, "  ");
            out.push_str(",\n");
        }

        // binding_summary
        let bs = &self.binding_summary;
        out.push_str("  \"binding_summary\": {\n");
        let _ = writeln!(out, "    \"has_cbv\": {},", bs.has_cbv);
        let _ = writeln!(out, "    \"cbv_vec4_count\": {},", bs.cbv_vec4_count);
        let _ = writeln!(out, "    \"has_texture_2d\": {},", bs.has_texture_2d);
        let _ = writeln!(out, "    \"has_sampler\": {},", bs.has_sampler);
        let _ = writeln!(out, "    \"has_storage_buffer\": {},", bs.has_storage_buffer);
        let _ = writeln!(
            out,
            "    \"has_storage_texture\": {},",
            bs.has_storage_texture
        );
        let _ = writeln!(out, "    \"has_lds\": {},", bs.has_lds);
        out.push_str("    \"vertex_input_mode\": ");
        write_json_string(&mut out, &bs.vertex_input_mode);
        out.push_str(",\n");

        out.push_str("    \"fragment_input_locations\": [");
        for (i, loc) in bs.fragment_input_locations.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{loc}");
        }
        out.push_str("],\n");

        out.push_str("    \"output_locations\": [");
        for (i, loc) in bs.output_locations.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{loc}");
        }
        out.push_str("],\n");

        let _ = writeln!(out, "    \"exports_position\": {},", bs.exports_position);
        let _ = writeln!(out, "    \"exports_depth\": {}", bs.exports_depth);
        out.push_str("  }\n");
        out.push_str("}\n");
        out
    }
}

/// Errors produced during RDNA2 -> WGSL translation or Naga verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rdna2TranslateError {
    NagaValidation(String),
    WgslEmit(String),
    WgslParse(String),
}

impl fmt::Display for Rdna2TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NagaValidation(msg) => write!(f, "Naga WGSL validation failed: {msg}"),
            Self::WgslEmit(msg) => write!(f, "Naga WGSL emission failed: {msg}"),
            Self::WgslParse(msg) => write!(f, "Naga WGSL parse check failed: {msg}"),
        }
    }
}

impl std::error::Error for Rdna2TranslateError {}

// ---------------------------------------------------------------------------
// Pre-pass program planning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertexInputMode {
    None,
    VertexIndex,
    Attributes,
}

impl VertexInputMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::VertexIndex => "vertex_index",
            Self::Attributes => "attributes",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VertexInputAttr {
    /// Vertex-buffer slot index (`vb_slot`, 0 for single-VB shaders).
    vb_slot: u32,
    /// Byte offset inside the vertex buffer for MUBUF/MTBUF fetches (`None` for legacy synthetic attributes).
    byte_offset: Option<u16>,
    location: u32,
    name: String,
    /// Vector component width (`1` -> scalar, `2` -> `vec2`, `3` -> `vec3`, `4` -> `vec4`).
    components: u8,
    /// WGSL / WebGPU attribute scalar type (`"f32"`, `"u32"`, or `"i32"`).
    component_type: &'static str,
    /// Explicit WebGPU vertex format name when decoded from GFX10.3 `MTBUF` `format`.
    fetch_format: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VertexFetchPlan {
    components: u8,
    component_type: &'static str,
    fetch_format: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DynamicHeapOrigin {
    base_user_sgpr: u32,
    stride: u32,
    index_user_sgpr: u32,
    index_shift: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CbvKey {
    /// Base SGPR pair index (raw `sbase` for unreflected direct/indirect SMEM loads;
    /// normalized to `0` when `user_sgpr` or `dynamic_heap` uniquely identifies the CBV).
    base_sbase: u8,
    /// Known console user-data SGPR when sourced from a descriptor table or root descriptor.
    user_sgpr: Option<u32>,
    /// `Some(table_offset)` for a coalesced descriptor-table V# / CBV-ptr entry,
    /// `None` for direct SMEM or dynamic heap CBVs.
    table_offset: Option<i32>,
    /// `Some(origin)` for a coalesced SM 6.6 `ResourceDescriptorHeap` V# CBV entry.
    dynamic_heap: Option<DynamicHeapOrigin>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CbvUsage {
    max_vec4_index: u32,
    has_dynamic_offset: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CbvSlotPlan {
    key: CbvKey,
    vec4_count: u32,
    highest_vec4_offset: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ResourceKey {
    Table {
        user_sgpr: u32,
        table_byte_offset: u32,
    },
    RootDescriptor {
        user_sgpr: u32,
    },
    StaticSampler(u32),
    InlineSampler([u32; 4]),
    SamplerHeap {
        base_user_sgpr: u32,
        stride: u32,
        index_user_sgpr: u32,
        index_shift: u32,
    },
    ResourceHeap {
        base_user_sgpr: u32,
        stride: u32,
        index_user_sgpr: u32,
        index_shift: u32,
    },
    RawSlot(u8),
}

impl ResourceKey {
    fn to_source(self) -> Option<WgslBindingSource> {
        match self {
            Self::Table {
                user_sgpr,
                table_byte_offset,
            } => Some(WgslBindingSource::Table {
                user_sgpr,
                table_byte_offset,
            }),
            Self::RootDescriptor { user_sgpr } => {
                Some(WgslBindingSource::RootDescriptor { user_sgpr })
            }
            Self::StaticSampler(index) => Some(WgslBindingSource::StaticSampler { index }),
            Self::InlineSampler(sdwords) => Some(WgslBindingSource::InlineSampler { sdwords }),
            Self::SamplerHeap {
                base_user_sgpr,
                stride,
                index_user_sgpr,
                index_shift,
            } => Some(WgslBindingSource::SamplerHeap {
                base_user_sgpr,
                stride,
                index_user_sgpr,
                index_shift,
            }),
            Self::ResourceHeap {
                base_user_sgpr,
                stride,
                index_user_sgpr,
                index_shift,
            } => Some(WgslBindingSource::ResourceHeap {
                base_user_sgpr,
                stride,
                index_user_sgpr,
                index_shift,
            }),
            Self::RawSlot(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TexDim {
    D2,
    D2Array,
    D3,
}

impl TexDim {
    fn from_mimg_dim(dim: u8) -> Self {
        match dim {
            5 => Self::D2Array,
            2 => Self::D3,
            _ => Self::D2,
        }
    }

    fn view_dimension_name(self) -> &'static str {
        match self {
            Self::D2 => "2d",
            Self::D2Array => "2d-array",
            Self::D3 => "3d",
        }
    }

    fn extra_coord_vgprs(self) -> u8 {
        match self {
            Self::D2 => 0,
            Self::D2Array | Self::D3 => 1,
        }
    }

    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::D2, o) => o,
            (s, Self::D2) => s,
            (s, _) => s,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StorageBufSlotPlan {
    key: ResourceKey,
    writable: bool,
}

struct ProgramPlan {
    stage: ShaderStage,
    entry_point: String,
    workgroup_size: [u32; 3],
    compute_seed_wg_sgprs: Vec<(u8, u32)>,
    active_inst_count: usize,
    vertex_input_mode: VertexInputMode,
    vertex_attrs: Vec<VertexInputAttr>,
    has_buffer_fetch: bool,
    legacy_uv_attr_index: Option<usize>,
    vsharp_fetch_insts: BTreeSet<usize>,
    indirect_smem_cbv_keys: BTreeMap<usize, CbvKey>,
    cbv_sources: BTreeMap<CbvKey, WgslBindingSource>,
    inst_tex_keys: BTreeMap<usize, ResourceKey>,
    inst_samp_keys: BTreeMap<usize, ResourceKey>,
    inst_storage_buf_keys: BTreeMap<usize, ResourceKey>,
    inst_storage_tex_keys: BTreeMap<usize, ResourceKey>,
    tex_dims: BTreeMap<ResourceKey, TexDim>,
    storage_tex_dims: BTreeMap<ResourceKey, TexDim>,
    vertex_storage_buf_insts: BTreeSet<usize>,
    vertex_fetch_vb_slots: BTreeMap<usize, u32>,
    cbv_slots: Vec<CbvSlotPlan>,
    cbv_vec4_count: u32,
    tex_slots: Vec<ResourceKey>,
    samp_slots: Vec<ResourceKey>,
    storage_buf_slots: Vec<StorageBufSlotPlan>,
    storage_tex_slots: Vec<ResourceKey>,
    uses_lds: bool,
    frag_attrs: Vec<u8>,
    vs_params: Vec<u8>,
    ps_mrts: Vec<u8>,
    ps_exports_depth: bool,
    vgprs: BTreeSet<u8>,
    sgprs: BTreeSet<u8>,
    /// Dedicated `(vgpr, lane)` local spill slots written by constant-lane `v_writelane_b32`.
    lane_spill_slots: BTreeSet<(u8, u8)>,
    needs_vcc: bool,
    needs_scc: bool,
    /// Single-lane `exec_mask` model is required (exec-mask SOP1 ops, `exec` SOP1/SOP2
    /// destinations, or `s_cbranch_execz/execnz`).
    needs_exec: bool,
}

fn resolve_stage(inferred: Rdna2InferredStage, override_stage: Rdna2StageOverride) -> ShaderStage {
    match override_stage {
        Rdna2StageOverride::Vertex => ShaderStage::Vertex,
        Rdna2StageOverride::Pixel => ShaderStage::Fragment,
        Rdna2StageOverride::Compute => ShaderStage::Compute,
        Rdna2StageOverride::Auto => match inferred {
            Rdna2InferredStage::Vertex => ShaderStage::Vertex,
            Rdna2InferredStage::Pixel => ShaderStage::Fragment,
            Rdna2InferredStage::Compute | Rdna2InferredStage::Unknown => ShaderStage::Compute,
        },
    }
}

fn default_entry_point(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vs_main",
        ShaderStage::Fragment => "fs_main",
        ShaderStage::Compute => "cs_main",
    }
}

fn sanitize_wgsl_ident(raw: &str, fallback: &str) -> String {
    let mut out = String::new();
    for (i, ch) in raw.chars().enumerate() {
        let valid = if i == 0 {
            ch.is_ascii_alphabetic() || ch == '_'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_'
        };
        if valid {
            out.push(ch);
        } else if !out.is_empty() {
            out.push('_');
        }
    }
    if out.is_empty() || out == "_" {
        fallback.to_string()
    } else {
        out
    }
}

fn smem_load_dword_count(op: u8) -> Option<u8> {
    match op {
        0x00 | 0x08 => Some(1),
        0x01 | 0x09 => Some(2),
        0x02 | 0x0A => Some(4),
        0x03 | 0x0B => Some(8),
        0x04 | 0x0C => Some(16),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct MimgAddrLayout {
    total_vgprs: u8,
    bias_off: Option<u8>,
    compare_off: Option<u8>,
    grad_off: Option<u8>,
    uv_off: u8,
    lod_off: Option<u8>,
    is_zero_lod: bool,
    is_load: bool,
    is_store: bool,
    is_get_lod: bool,
}

/// Decode GFX10.3 2D MIMG address VGPR layout (`[offset], [bias], [z_compare], [grad], u, v, [lod/clamp]`).
fn mimg_addr_layout(op: u8) -> Option<MimgAddrLayout> {
    let base = MimgAddrLayout {
        total_vgprs: 2,
        bias_off: None,
        compare_off: None,
        grad_off: None,
        uv_off: 0,
        lod_off: None,
        is_zero_lod: false,
        is_load: false,
        is_store: false,
        is_get_lod: false,
    };
    match op {
        // image_load: (x, y)
        0x00 => Some(MimgAddrLayout {
            is_load: true,
            ..base
        }),
        // image_load_mip: (x, y, mip)
        0x01 => Some(MimgAddrLayout {
            total_vgprs: 3,
            lod_off: Some(2),
            is_load: true,
            ..base
        }),
        // image_store / image_store_pck: (x, y)
        0x08 | 0x0A => Some(MimgAddrLayout {
            is_store: true,
            ..base
        }),
        // image_store_mip / image_store_mip_pck: (x, y, mip)
        0x09 | 0x0B => Some(MimgAddrLayout {
            total_vgprs: 3,
            lod_off: Some(2),
            is_store: true,
            ..base
        }),
        // image_sample: (u, v)
        0x20 => Some(base),
        // image_sample_cl: (u, v, clamp)
        0x21 => Some(MimgAddrLayout {
            total_vgprs: 3,
            ..base
        }),
        // image_sample_d: (dudx, dvdx, dudy, dvdy, u, v)
        0x22 => Some(MimgAddrLayout {
            total_vgprs: 6,
            grad_off: Some(0),
            uv_off: 4,
            ..base
        }),
        // image_sample_l: (u, v, lod)
        0x24 => Some(MimgAddrLayout {
            total_vgprs: 3,
            lod_off: Some(2),
            ..base
        }),
        // image_sample_b: (bias, u, v)
        0x25 => Some(MimgAddrLayout {
            total_vgprs: 3,
            bias_off: Some(0),
            uv_off: 1,
            ..base
        }),
        // image_sample_lz: (u, v)
        0x27 => Some(MimgAddrLayout {
            is_zero_lod: true,
            ..base
        }),
        // image_sample_c: (z_ref, u, v)
        0x28 => Some(MimgAddrLayout {
            total_vgprs: 3,
            compare_off: Some(0),
            uv_off: 1,
            ..base
        }),
        // image_sample_c_l: (z_ref, u, v, lod)
        0x2C => Some(MimgAddrLayout {
            total_vgprs: 4,
            compare_off: Some(0),
            uv_off: 1,
            lod_off: Some(3),
            ..base
        }),
        // image_sample_c_b: (bias, z_ref, u, v)
        0x2D => Some(MimgAddrLayout {
            total_vgprs: 4,
            bias_off: Some(0),
            compare_off: Some(1),
            uv_off: 2,
            ..base
        }),
        // image_sample_c_lz: (z_ref, u, v)
        0x2F => Some(MimgAddrLayout {
            total_vgprs: 3,
            compare_off: Some(0),
            uv_off: 1,
            is_zero_lod: true,
            ..base
        }),
        // image_sample_o: (offset, u, v)
        0x30 => Some(MimgAddrLayout {
            total_vgprs: 3,
            uv_off: 1,
            ..base
        }),
        // image_sample_l_o: (offset, u, v, lod)
        0x34 => Some(MimgAddrLayout {
            total_vgprs: 4,
            uv_off: 1,
            lod_off: Some(3),
            ..base
        }),
        // image_sample_b_o: (offset, bias, u, v)
        0x35 => Some(MimgAddrLayout {
            total_vgprs: 4,
            bias_off: Some(1),
            uv_off: 2,
            ..base
        }),
        // image_sample_lz_o: (offset, u, v)
        0x37 => Some(MimgAddrLayout {
            total_vgprs: 3,
            uv_off: 1,
            is_zero_lod: true,
            ..base
        }),
        // image_sample_c_o: (offset, z_ref, u, v)
        0x38 => Some(MimgAddrLayout {
            total_vgprs: 4,
            compare_off: Some(1),
            uv_off: 2,
            ..base
        }),
        // image_sample_c_lz_o: (offset, z_ref, u, v)
        0x3F => Some(MimgAddrLayout {
            total_vgprs: 4,
            compare_off: Some(1),
            uv_off: 2,
            is_zero_lod: true,
            ..base
        }),
        // image_get_lod: (u, v) -> mip levels (safe WGSL mip 0.0 fallback)
        0x60 => Some(MimgAddrLayout {
            is_get_lod: true,
            ..base
        }),
        _ => None,
    }
}

fn vop3p_is_ternary(op: u8) -> bool {
    !matches!(op, 0x01..=0x08 | 0x0A..=0x0D | 0x0F..=0x12)
}

fn mubuf_load_components(op: u8) -> Option<u8> {
    match op {
        0x00 | 0x0C => Some(1),
        0x01 | 0x0D => Some(2),
        0x02 | 0x0F => Some(3),
        0x03 | 0x0E => Some(4),
        _ => None,
    }
}

fn mtbuf_load_components(op: u8) -> Option<u8> {
    match op {
        0x00 => Some(1),
        0x01 => Some(2),
        0x02 => Some(3),
        0x03 => Some(4),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageBufOpKind {
    LoadUbyte,
    LoadSbyte,
    LoadUshort,
    LoadSshort,
    LoadDwords(u8),
    StoreDwords(u8),
}

impl StorageBufOpKind {
    const fn is_store(self) -> bool {
        matches!(self, Self::StoreDwords(_))
    }

    const fn vdata_count(self) -> u8 {
        match self {
            Self::LoadUbyte | Self::LoadSbyte | Self::LoadUshort | Self::LoadSshort => 1,
            Self::LoadDwords(n) | Self::StoreDwords(n) => n,
        }
    }

    const fn elem_stride_bytes(self) -> u32 {
        match self {
            Self::LoadUbyte | Self::LoadSbyte => 1,
            Self::LoadUshort | Self::LoadSshort => 2,
            Self::LoadDwords(n) | Self::StoreDwords(n) => (n as u32) * 4,
        }
    }
}

fn mubuf_storage_op(op: u8) -> Option<StorageBufOpKind> {
    match op {
        0x00 | 0x0C => Some(StorageBufOpKind::LoadDwords(1)),
        0x01 | 0x0D => Some(StorageBufOpKind::LoadDwords(2)),
        0x02 | 0x0F => Some(StorageBufOpKind::LoadDwords(3)),
        0x03 | 0x0E => Some(StorageBufOpKind::LoadDwords(4)),
        0x08 => Some(StorageBufOpKind::LoadUbyte),
        0x09 => Some(StorageBufOpKind::LoadSbyte),
        0x0A => Some(StorageBufOpKind::LoadUshort),
        0x0B => Some(StorageBufOpKind::LoadSshort),
        0x04 | 0x1C => Some(StorageBufOpKind::StoreDwords(1)),
        0x05 | 0x1D => Some(StorageBufOpKind::StoreDwords(2)),
        0x06 | 0x1F => Some(StorageBufOpKind::StoreDwords(3)),
        0x07 | 0x1E => Some(StorageBufOpKind::StoreDwords(4)),
        _ => None,
    }
}

const fn mubuf_is_atomic(op: u8) -> bool {
    matches!(op, 0x32..=0x3A)
}

fn mtbuf_storage_op(op: u8) -> Option<StorageBufOpKind> {
    match op {
        0x00 => Some(StorageBufOpKind::LoadDwords(1)),
        0x01 => Some(StorageBufOpKind::LoadDwords(2)),
        0x02 => Some(StorageBufOpKind::LoadDwords(3)),
        0x03 => Some(StorageBufOpKind::LoadDwords(4)),
        0x04 => Some(StorageBufOpKind::StoreDwords(1)),
        0x05 => Some(StorageBufOpKind::StoreDwords(2)),
        0x06 => Some(StorageBufOpKind::StoreDwords(3)),
        0x07 => Some(StorageBufOpKind::StoreDwords(4)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DsLdsOp {
    ReadContiguous(u8),
    WriteContiguous(u8),
    Read2 { stride_bytes: u32 },
    Write2 { stride_bytes: u32 },
}

fn ds_lds_op(op: u8) -> Option<DsLdsOp> {
    match op {
        0x0D => Some(DsLdsOp::WriteContiguous(1)),
        0x0E => Some(DsLdsOp::Write2 { stride_bytes: 4 }),
        0x0F => Some(DsLdsOp::Write2 { stride_bytes: 256 }),
        0x36 => Some(DsLdsOp::ReadContiguous(1)),
        0x37 => Some(DsLdsOp::Read2 { stride_bytes: 4 }),
        0x38 => Some(DsLdsOp::Read2 { stride_bytes: 256 }),
        0x4D => Some(DsLdsOp::WriteContiguous(2)),
        0x76 => Some(DsLdsOp::ReadContiguous(2)),
        0xDE => Some(DsLdsOp::WriteContiguous(3)),
        0xDF => Some(DsLdsOp::WriteContiguous(4)),
        0xFE => Some(DsLdsOp::ReadContiguous(3)),
        0xFF => Some(DsLdsOp::ReadContiguous(4)),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
/// 64-bit shift flavour emitted on `(lo, hi)` u32 register pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shift64Kind {
    Left,
    LogicalRight,
    ArithmeticRight,
}

fn record_src_usage(
    src: u16,
    vgprs: &mut BTreeSet<u8>,
    sgprs: &mut BTreeSet<u8>,
    written_vgprs: &BTreeSet<u8>,
    read_before_write_vgprs: &mut BTreeSet<u8>,
    written_sgprs: &BTreeSet<u8>,
    read_before_write_sgprs: &mut BTreeSet<u8>,
    needs_vcc: &mut bool,
    needs_scc: &mut bool,
) {
    match src {
        0..=105 => {
            let s = src as u8;
            sgprs.insert(s);
            if !written_sgprs.contains(&s) {
                read_before_write_sgprs.insert(s);
            }
        }
        106 => {
            *needs_vcc = true;
        }
        251 => {
            *needs_vcc = true;
        }
        253 => {
            *needs_scc = true;
        }
        256..=511 => {
            let v = (src - 256) as u8;
            vgprs.insert(v);
            if !written_vgprs.contains(&v) {
                read_before_write_vgprs.insert(v);
            }
        }
        _ => {}
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SgprOrigin {
    /// Lattice bottom: scratch SGPR/VGPR not yet written on this path.
    Unwritten,
    /// Lattice top: overwritten with a non-tracked/dynamic value or conflicting origins.
    Unknown,
    /// Hardware vertex-buffer table pointer (`s[0:1]` in vertex stages), optionally offset by bytes.
    VbTablePtr { byte_offset: i32 },
    /// Console user-data SGPR (`s8+` for graphics, `s4+` for compute), optionally offset by bytes.
    UserTable { user_sgpr: u32, byte_offset: i32 },
    /// Shifted/scaled console user-data SGPR (`s_lshl_b32`, `s_mul_i32`, `s_mulk_i32`,
    /// `s_lshl{1..4}_add_u32`), used as a dynamic descriptor-heap byte offset.
    ScaledUserSgpr {
        user_sgpr: u32,
        stride: u32,
        shift: u32,
    },
    /// 32-bit GPUVA pointer formed by adding a heap base user SGPR and a scaled index user SGPR
    /// (`s_add_u32` / `s_add_i32` / `s_lshl{1..4}_add_u32`).
    DynamicHeapPtr(DynamicHeapOrigin),
    /// Descriptor dword (or nested table / CBV pointer) loaded from a user-data descriptor table
    /// via `s_load_dword*`.
    TableDesc {
        def_inst_idx: usize,
        user_sgpr: u32,
        table_byte_offset: u32,
        /// Additional constant byte offset applied to this loaded pointer via SALU.
        addr_offset: i32,
    },
    /// Descriptor dword loaded from an SM 6.6 dynamic sampler/resource heap via `s_load_dword*`.
    DynamicHeapDesc {
        def_inst_idx: usize,
        origin: DynamicHeapOrigin,
    },
    /// Descriptor dword loaded from an unknown/legacy table `sbase` (e.g. `s[0:1]` in pixel/compute),
    /// kept so unreflected indirect V# coalescing and spurious-CBV suppression survive copies/branches.
    RawTableDesc {
        def_inst_idx: usize,
        table_sbase: u8,
        table_offset: i32,
        is_dynamic: bool,
    },
    /// Vertex-buffer V# descriptor loaded from the hardware VB table (`s[0:1]`).
    VbDesc {
        def_inst_idx: usize,
        vb_slot: u32,
    },
    /// Materialized from immediate/literal constants (tracks exact `u32` value when statically known,
    /// and `def_inst_idx` to detect inline static sampler S#).
    ConstImmediate {
        def_inst_idx: usize,
        value: Option<u32>,
    },
}

#[derive(Debug, Default)]
struct DescriptorAnalysis {
    vsharp_fetch_insts: BTreeSet<usize>,
    indirect_smem_cbv_keys: BTreeMap<usize, CbvKey>,
    cbv_sources: BTreeMap<CbvKey, WgslBindingSource>,
    inst_tex_keys: BTreeMap<usize, ResourceKey>,
    inst_samp_keys: BTreeMap<usize, ResourceKey>,
    inst_storage_buf_keys: BTreeMap<usize, ResourceKey>,
    inst_storage_tex_keys: BTreeMap<usize, ResourceKey>,
    vertex_storage_buf_insts: BTreeSet<usize>,
    vertex_fetch_vb_slots: BTreeMap<usize, u32>,
}

#[derive(Clone, PartialEq, Eq)]
struct OriginState {
    origins: [SgprOrigin; 106],
    lane_origins: BTreeMap<(u8, u8), SgprOrigin>,
    vgpr_uniform_origins: [SgprOrigin; 256],
}

impl OriginState {
    fn initial(stage: ShaderStage) -> Self {
        let mut origins = [SgprOrigin::Unwritten; 106];
        // Seed console ABI initial SGPR origins:
        //   Graphics: system SGPRs in s0..s7 (vertex hardware VB table ptr in s[0:1]),
        //             user data starts at s8..s31; scratch SGPRs s32..s105 start Unwritten (bottom).
        //   Compute:  system SGPRs in s0..s3, user data starts at s4..s31; s32..s105 Unwritten.
        let user_base: usize = if stage == ShaderStage::Compute { 4 } else { 8 };
        for slot in origins.iter_mut().take(user_base) {
            *slot = SgprOrigin::Unknown;
        }
        if stage == ShaderStage::Vertex {
            origins[0] = SgprOrigin::VbTablePtr { byte_offset: 0 };
        }
        for (sgpr, slot) in origins.iter_mut().enumerate().take(32).skip(user_base) {
            *slot = SgprOrigin::UserTable {
                user_sgpr: sgpr as u32,
                byte_offset: 0,
            };
        }
        Self {
            origins,
            lane_origins: BTreeMap::new(),
            vgpr_uniform_origins: [SgprOrigin::Unwritten; 256],
        }
    }

    /// Monotonic CFG dataflow join. Returns `true` if `self` changed.
    fn join_from(
        &mut self,
        other: &Self,
        def_aliases: &mut BTreeMap<usize, BTreeSet<usize>>,
    ) -> bool {
        let mut changed = false;
        for (dst, &src) in self.origins.iter_mut().zip(other.origins.iter()) {
            let joined = join_cfg_origin(*dst, src, Some(def_aliases));
            if joined != *dst {
                *dst = joined;
                changed = true;
            }
        }
        for (dst, &src) in self
            .vgpr_uniform_origins
            .iter_mut()
            .zip(other.vgpr_uniform_origins.iter())
        {
            let joined = join_cfg_origin(*dst, src, Some(def_aliases));
            if joined != *dst {
                *dst = joined;
                changed = true;
            }
        }
        if !self.lane_origins.is_empty() {
            let mut to_remove = Vec::new();
            for (&k, dst) in &mut self.lane_origins {
                match other.lane_origins.get(&k).copied() {
                    Some(src) => {
                        let joined = join_cfg_origin(*dst, src, Some(def_aliases));
                        if matches!(joined, SgprOrigin::Unknown | SgprOrigin::Unwritten) {
                            to_remove.push(k);
                        } else if joined != *dst {
                            *dst = joined;
                            changed = true;
                        }
                    }
                    None => {
                        to_remove.push(k);
                    }
                }
            }
            if !to_remove.is_empty() {
                for k in to_remove {
                    self.lane_origins.remove(&k);
                }
                changed = true;
            }
        }
        changed
    }
}

struct BufferLoadUse {
    inst_idx: usize,
    def_inst_idx: usize,
    cbv_key: Option<CbvKey>,
    source: Option<WgslBindingSource>,
}

struct CbvPtrLoad {
    inst_idx: usize,
    ptr_def: usize,
    cbv_key: CbvKey,
    source: WgslBindingSource,
}

#[derive(Default)]
struct AnalysisCollector {
    out: DescriptorAnalysis,
    tainted_defs: BTreeSet<usize>,
    def_parents: BTreeMap<usize, usize>,
    def_aliases: BTreeMap<usize, BTreeSet<usize>>,
    direct_descriptor_defs: BTreeSet<usize>,
    raw_descriptor_defs: BTreeSet<usize>,
    buffer_load_uses: Vec<BufferLoadUse>,
    cbv_ptr_loads: Vec<CbvPtrLoad>,
    direct_root_cbvs: Vec<(usize, u32)>,
    next_static_sampler_index: u32,
    static_sampler_by_def: BTreeMap<usize, u32>,
}

fn decode_inline_u32_src(src: u16, literal: Option<u32>) -> Option<i32> {
    if src == 255 {
        literal.map(|v| v as i32)
    } else if (128..=192).contains(&src) {
        Some((src - 128) as i32)
    } else if (193..=208).contains(&src) {
        Some(-(src as i32 - 192))
    } else {
        None
    }
}

/// Exact IEEE-754 `f32` bit pattern for GFX10.3 inline float constants (`240..=248`).
fn decode_inline_float_u32(src: u16) -> Option<u32> {
    match src {
        240 => Some(0.5f32.to_bits()),
        241 => Some((-0.5f32).to_bits()),
        242 => Some(1.0f32.to_bits()),
        243 => Some((-1.0f32).to_bits()),
        244 => Some(2.0f32.to_bits()),
        245 => Some((-2.0f32).to_bits()),
        246 => Some(4.0f32.to_bits()),
        247 => Some((-4.0f32).to_bits()),
        248 => Some(0x3E22_F983), // 1 / (2 * pi)
        _ => None,
    }
}

/// Exact IEEE-754 `f64` bit pattern for GFX10.3 64-bit inline float constants (`240..=248`).
fn decode_inline_float_u64(src: u16) -> Option<u64> {
    match src {
        240 => Some(0.5f64.to_bits()),
        241 => Some((-0.5f64).to_bits()),
        242 => Some(1.0f64.to_bits()),
        243 => Some((-1.0f64).to_bits()),
        244 => Some(2.0f64.to_bits()),
        245 => Some((-2.0f64).to_bits()),
        246 => Some(4.0f64.to_bits()),
        247 => Some((-4.0f64).to_bits()),
        248 => Some(0x3FC4_5F30_6DC9_C883), // 1 / (2 * pi) in IEEE-754 binary64
        _ => None,
    }
}

/// Decode a static wave lane index (`0..=63`) from `v_readlane_b32` / `v_writelane_b32` `src1`.
/// Returns `None` when `src1` is a dynamic register (SGPR, `m0`, etc.).
fn decode_const_lane_index(src1: u16, literal: Option<u32>) -> Option<u8> {
    decode_inline_u32_src(src1, literal).map(|v| (v as u32 & 63) as u8)
}

fn resolve_const_u32(src: u8, literal: Option<u32>, origins: &[SgprOrigin; 106]) -> Option<u32> {
    if let Some(v) = decode_inline_u32_src(u16::from(src), literal) {
        Some(v as u32)
    } else if let Some(bits) = decode_inline_float_u32(u16::from(src)) {
        Some(bits)
    } else if src <= 105
        && let SgprOrigin::ConstImmediate { value: Some(v), .. } = origins[src as usize]
    {
        Some(v)
    } else {
        None
    }
}

fn resolve_const_lane_index(
    src1: u16,
    literal: Option<u32>,
    origins: &[SgprOrigin; 106],
) -> Option<u8> {
    if let Some(lane) = decode_const_lane_index(src1, literal) {
        Some(lane)
    } else if src1 <= 105
        && let SgprOrigin::ConstImmediate { value: Some(v), .. } = origins[src1 as usize]
    {
        Some((v & 63) as u8)
    } else {
        None
    }
}

fn is_const_src(src: u8, origins: &[SgprOrigin; 106]) -> bool {
    src >= 128
        || (src <= 105 && matches!(origins[src as usize], SgprOrigin::ConstImmediate { .. }))
}

fn normalize_read_origin(o: SgprOrigin) -> SgprOrigin {
    if o == SgprOrigin::Unwritten {
        SgprOrigin::Unknown
    } else {
        o
    }
}

fn record_def_alias(
    def_aliases: Option<&mut BTreeMap<usize, BTreeSet<usize>>>,
    d1: usize,
    d2: usize,
) {
    if d1 != d2
        && let Some(aliases) = def_aliases
    {
        aliases.entry(d1.min(d2)).or_default().insert(d1.max(d2));
    }
}

fn resolve_src_origin_u8(
    src: u8,
    literal: Option<u32>,
    inst_idx: usize,
    origins: &[SgprOrigin; 106],
) -> SgprOrigin {
    if src <= 105 {
        normalize_read_origin(origins[src as usize])
    } else if let Some(v) = decode_inline_u32_src(u16::from(src), literal) {
        SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: Some(v as u32),
        }
    } else if let Some(bits) = decode_inline_float_u32(u16::from(src)) {
        SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: Some(bits),
        }
    } else if src >= 128 {
        SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: None,
        }
    } else {
        SgprOrigin::Unknown
    }
}

fn resolve_src_pair_origins(
    src: u8,
    literal: Option<u32>,
    inst_idx: usize,
    origins: &[SgprOrigin; 106],
) -> (SgprOrigin, SgprOrigin) {
    if src <= 105 {
        let lo = normalize_read_origin(origins[src as usize]);
        let hi = if src <= 104 {
            normalize_read_origin(origins[(src + 1) as usize])
        } else {
            SgprOrigin::Unknown
        };
        (lo, hi)
    } else if let Some(v) = decode_inline_u32_src(u16::from(src), literal) {
        let hi_val = if src == 0xFF {
            0
        } else if v < 0 {
            u32::MAX
        } else {
            0
        };
        (
            SgprOrigin::ConstImmediate {
                def_inst_idx: inst_idx,
                value: Some(v as u32),
            },
            SgprOrigin::ConstImmediate {
                def_inst_idx: inst_idx,
                value: Some(hi_val),
            },
        )
    } else if let Some(bits) = decode_inline_float_u64(u16::from(src)) {
        (
            SgprOrigin::ConstImmediate {
                def_inst_idx: inst_idx,
                value: Some(bits as u32),
            },
            SgprOrigin::ConstImmediate {
                def_inst_idx: inst_idx,
                value: Some((bits >> 32) as u32),
            },
        )
    } else if src >= 128 {
        let c = SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: None,
        };
        (c, c)
    } else {
        (SgprOrigin::Unknown, SgprOrigin::Unknown)
    }
}

/// Extract all 4 statically known GFX10 S# descriptor dwords from `s[samp_sgpr .. samp_sgpr+3]`
/// when every register in the quad holds `ConstImmediate { value: Some(d), .. }`.
fn resolve_inline_sampler_sdwords(origins: &[SgprOrigin; 106], samp_sgpr: u8) -> Option<[u32; 4]> {
    let base = samp_sgpr as usize;
    if base + 3 >= origins.len() {
        return None;
    }
    let mut sdwords = [0u32; 4];
    for (i, slot) in sdwords.iter_mut().enumerate() {
        match origins[base + i] {
            SgprOrigin::ConstImmediate { value: Some(v), .. } => *slot = v,
            _ => return None,
        }
    }
    Some(sdwords)
}

fn is_known_descriptor_or_table(o: SgprOrigin) -> bool {
    matches!(
        o,
        SgprOrigin::VbTablePtr { .. }
            | SgprOrigin::UserTable { .. }
            | SgprOrigin::ScaledUserSgpr { .. }
            | SgprOrigin::DynamicHeapPtr(_)
            | SgprOrigin::TableDesc { .. }
            | SgprOrigin::DynamicHeapDesc { .. }
            | SgprOrigin::VbDesc { .. }
            | SgprOrigin::RawTableDesc { .. }
    )
}

fn is_const_zero(o: SgprOrigin) -> bool {
    matches!(o, SgprOrigin::ConstImmediate { value: Some(0), .. })
}

/// Monotonic 4-level lattice join for CFG block merges:
/// `Unwritten (Bottom) < ConstZero < { NonZeroConst | KnownTable/Desc } < Unknown (Top)`.
fn join_cfg_origin(
    a: SgprOrigin,
    b: SgprOrigin,
    def_aliases: Option<&mut BTreeMap<usize, BTreeSet<usize>>>,
) -> SgprOrigin {
    if a == b {
        return a;
    }
    match (a, b) {
        (SgprOrigin::Unwritten, other) | (other, SgprOrigin::Unwritten) => other,
        (SgprOrigin::Unknown, _) | (_, SgprOrigin::Unknown) => SgprOrigin::Unknown,
        (
            SgprOrigin::TableDesc {
                def_inst_idx: d1,
                user_sgpr: u1,
                table_byte_offset: o1,
                addr_offset: ao1,
            },
            SgprOrigin::TableDesc {
                def_inst_idx: d2,
                user_sgpr: u2,
                table_byte_offset: o2,
                addr_offset: ao2,
            },
        ) if u1 == u2 && o1 == o2 && ao1 == ao2 => {
            record_def_alias(def_aliases, d1, d2);
            SgprOrigin::TableDesc {
                def_inst_idx: d1.min(d2),
                user_sgpr: u1,
                table_byte_offset: o1,
                addr_offset: ao1,
            }
        }
        (
            SgprOrigin::DynamicHeapDesc {
                def_inst_idx: d1,
                origin: h1,
            },
            SgprOrigin::DynamicHeapDesc {
                def_inst_idx: d2,
                origin: h2,
            },
        ) if h1 == h2 => {
            record_def_alias(def_aliases, d1, d2);
            SgprOrigin::DynamicHeapDesc {
                def_inst_idx: d1.min(d2),
                origin: h1,
            }
        }
        (
            SgprOrigin::VbDesc {
                def_inst_idx: d1,
                vb_slot: v1,
            },
            SgprOrigin::VbDesc {
                def_inst_idx: d2,
                vb_slot: v2,
            },
        ) if v1 == v2 => {
            record_def_alias(def_aliases, d1, d2);
            SgprOrigin::VbDesc {
                def_inst_idx: d1.min(d2),
                vb_slot: v1,
            }
        }
        (
            SgprOrigin::RawTableDesc {
                def_inst_idx: d1,
                table_sbase: b1,
                table_offset: o1,
                is_dynamic: dyn1,
            },
            SgprOrigin::RawTableDesc {
                def_inst_idx: d2,
                table_sbase: b2,
                table_offset: o2,
                is_dynamic: dyn2,
            },
        ) if b1 == b2 && o1 == o2 && dyn1 == dyn2 => {
            record_def_alias(def_aliases, d1, d2);
            SgprOrigin::RawTableDesc {
                def_inst_idx: d1.min(d2),
                table_sbase: b1,
                table_offset: o1,
                is_dynamic: dyn1,
            }
        }
        (
            SgprOrigin::ConstImmediate {
                def_inst_idx: d1,
                value: v1,
            },
            SgprOrigin::ConstImmediate {
                def_inst_idx: d2,
                value: v2,
            },
        ) => SgprOrigin::ConstImmediate {
            def_inst_idx: d1.min(d2),
            value: if v1 == v2 { v1 } else { None },
        },
        // Zero-init / null-pointer fallback (`0`) promotes to the non-null table/descriptor origin.
        (z, known) | (known, z) if is_const_zero(z) && is_known_descriptor_or_table(known) => known,
        _ => SgprOrigin::Unknown,
    }
}

/// Merge two branch origins for `s_cmov_b32/b64`, `s_cmovk_i32`, `s_cselect_b32/b64`.
/// Preserves the origin when both branches agree or when one branch is a known table/descriptor
/// origin and the other is `Unknown`, `Unwritten`, or a constant fallback (e.g. null-pointer guard).
fn merge_select_origin(
    a: SgprOrigin,
    b: SgprOrigin,
    inst_idx: usize,
    def_aliases: Option<&mut BTreeMap<usize, BTreeSet<usize>>>,
) -> SgprOrigin {
    if a == b {
        return a;
    }
    match (a, b) {
        (SgprOrigin::Unwritten, other) | (other, SgprOrigin::Unwritten) => other,
        (
            SgprOrigin::TableDesc {
                def_inst_idx: d1,
                user_sgpr: u1,
                table_byte_offset: o1,
                addr_offset: ao1,
            },
            SgprOrigin::TableDesc {
                def_inst_idx: d2,
                user_sgpr: u2,
                table_byte_offset: o2,
                addr_offset: ao2,
            },
        ) if u1 == u2 && o1 == o2 && ao1 == ao2 => {
            record_def_alias(def_aliases, d1, d2);
            SgprOrigin::TableDesc {
                def_inst_idx: d1.min(d2),
                user_sgpr: u1,
                table_byte_offset: o1,
                addr_offset: ao1,
            }
        }
        (
            SgprOrigin::DynamicHeapDesc {
                def_inst_idx: d1,
                origin: h1,
            },
            SgprOrigin::DynamicHeapDesc {
                def_inst_idx: d2,
                origin: h2,
            },
        ) if h1 == h2 => {
            record_def_alias(def_aliases, d1, d2);
            SgprOrigin::DynamicHeapDesc {
                def_inst_idx: d1.min(d2),
                origin: h1,
            }
        }
        (
            SgprOrigin::VbDesc {
                def_inst_idx: d1,
                vb_slot: v1,
            },
            SgprOrigin::VbDesc {
                def_inst_idx: d2,
                vb_slot: v2,
            },
        ) if v1 == v2 => {
            record_def_alias(def_aliases, d1, d2);
            SgprOrigin::VbDesc {
                def_inst_idx: d1.min(d2),
                vb_slot: v1,
            }
        }
        (
            SgprOrigin::RawTableDesc {
                def_inst_idx: d1,
                table_sbase: b1,
                table_offset: o1,
                is_dynamic: dyn1,
            },
            SgprOrigin::RawTableDesc {
                def_inst_idx: d2,
                table_sbase: b2,
                table_offset: o2,
                is_dynamic: dyn2,
            },
        ) if b1 == b2 && o1 == o2 && dyn1 == dyn2 => {
            record_def_alias(def_aliases, d1, d2);
            SgprOrigin::RawTableDesc {
                def_inst_idx: d1.min(d2),
                table_sbase: b1,
                table_offset: o1,
                is_dynamic: dyn1,
            }
        }
        (
            SgprOrigin::ConstImmediate { value: v1, .. },
            SgprOrigin::ConstImmediate { value: v2, .. },
        ) => SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: if v1 == v2 { v1 } else { None },
        },
        (known, SgprOrigin::Unknown | SgprOrigin::ConstImmediate { .. })
        | (SgprOrigin::Unknown | SgprOrigin::ConstImmediate { .. }, known)
            if is_known_descriptor_or_table(known) =>
        {
            known
        }
        _ => SgprOrigin::Unknown,
    }
}

/// Returns `true` for low-bit pointer alignment masks `& ~((1 << k) - 1)` for `k in 0..=6`
/// (e.g., `0xFFFFFFFF`, `0xFFFFFFFC`, `0xFFFFFFF8`, `0xFFFFFFF0`, `0xFFFFFFE0`, `0xFFFFFFC0`).
fn is_pointer_align_mask(mask: u32) -> bool {
    let cleared = !mask;
    cleared <= 63 && cleared.wrapping_add(1).is_power_of_two()
}

fn apply_const_offset(origin: SgprOrigin, delta: i32) -> SgprOrigin {
    if delta.unsigned_abs() > 0x0100_0000 {
        return SgprOrigin::Unknown;
    }
    match origin {
        SgprOrigin::UserTable {
            user_sgpr,
            byte_offset,
        } => SgprOrigin::UserTable {
            user_sgpr,
            byte_offset: byte_offset.saturating_add(delta),
        },
        SgprOrigin::VbTablePtr { byte_offset } => SgprOrigin::VbTablePtr {
            byte_offset: byte_offset.saturating_add(delta),
        },
        SgprOrigin::TableDesc {
            def_inst_idx,
            user_sgpr,
            table_byte_offset,
            addr_offset,
        } => SgprOrigin::TableDesc {
            def_inst_idx,
            user_sgpr,
            table_byte_offset,
            addr_offset: addr_offset.saturating_add(delta),
        },
        SgprOrigin::ScaledUserSgpr { .. } if delta == 0 => origin,
        SgprOrigin::DynamicHeapPtr(_) if delta == 0 => origin,
        SgprOrigin::DynamicHeapDesc { .. } if delta == 0 => origin,
        SgprOrigin::VbDesc { .. } if delta == 0 => origin,
        SgprOrigin::RawTableDesc { .. } if delta == 0 => origin,
        _ => SgprOrigin::Unknown,
    }
}

fn apply_or_const(origin: SgprOrigin, imm: u32, inst_idx: usize) -> SgprOrigin {
    if imm == 0 {
        return origin;
    }
    if imm == u32::MAX {
        return SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: Some(u32::MAX),
        };
    }
    match origin {
        SgprOrigin::ConstImmediate { value, .. } => SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: value.map(|v| v | imm),
        },
        // LLVM `or disjoint` optimization on aligned descriptor-table pointers:
        // `(table_base + off) | imm == table_base + (off + imm)` when `off & imm == 0`.
        SgprOrigin::UserTable {
            user_sgpr,
            byte_offset,
        } if imm < 0x0010_0000 && byte_offset >= 0 && ((byte_offset as u32) & imm) == 0 => {
            SgprOrigin::UserTable {
                user_sgpr,
                byte_offset: byte_offset.saturating_add(imm as i32),
            }
        }
        SgprOrigin::VbTablePtr { byte_offset }
            if imm < 0x0010_0000 && byte_offset >= 0 && ((byte_offset as u32) & imm) == 0 =>
        {
            SgprOrigin::VbTablePtr {
                byte_offset: byte_offset.saturating_add(imm as i32),
            }
        }
        SgprOrigin::TableDesc {
            def_inst_idx,
            user_sgpr,
            table_byte_offset,
            addr_offset,
        } if imm < 0x0010_0000 && addr_offset >= 0 && ((addr_offset as u32) & imm) == 0 => {
            SgprOrigin::TableDesc {
                def_inst_idx,
                user_sgpr,
                table_byte_offset,
                addr_offset: addr_offset.saturating_add(imm as i32),
            }
        }
        _ => SgprOrigin::Unknown,
    }
}

fn apply_and_const(origin: SgprOrigin, mask: u32, inst_idx: usize) -> SgprOrigin {
    if mask == u32::MAX {
        return origin;
    }
    if mask == 0 {
        return SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: Some(0),
        };
    }
    match origin {
        SgprOrigin::ConstImmediate { value, .. } => SgprOrigin::ConstImmediate {
            def_inst_idx: inst_idx,
            value: value.map(|v| v & mask),
        },
        SgprOrigin::UserTable {
            user_sgpr,
            byte_offset,
        } if is_pointer_align_mask(mask) => SgprOrigin::UserTable {
            user_sgpr,
            byte_offset: if byte_offset >= 0 {
                ((byte_offset as u32) & mask) as i32
            } else {
                byte_offset
            },
        },
        SgprOrigin::VbTablePtr { byte_offset } if is_pointer_align_mask(mask) => {
            SgprOrigin::VbTablePtr {
                byte_offset: if byte_offset >= 0 {
                    ((byte_offset as u32) & mask) as i32
                } else {
                    byte_offset
                },
            }
        }
        SgprOrigin::DynamicHeapPtr(origin) if is_pointer_align_mask(mask) => {
            SgprOrigin::DynamicHeapPtr(origin)
        }
        SgprOrigin::ScaledUserSgpr {
            user_sgpr,
            stride,
            shift,
        } if is_pointer_align_mask(mask) && (stride & !mask) == 0 => SgprOrigin::ScaledUserSgpr {
            user_sgpr,
            stride,
            shift,
        },
        SgprOrigin::TableDesc {
            def_inst_idx,
            user_sgpr,
            table_byte_offset,
            addr_offset,
        } if is_pointer_align_mask(mask) => SgprOrigin::TableDesc {
            def_inst_idx,
            user_sgpr,
            table_byte_offset,
            addr_offset: if addr_offset >= 0 {
                ((addr_offset as u32) & mask) as i32
            } else {
                addr_offset
            },
        },
        _ => SgprOrigin::Unknown,
    }
}

fn scaled_user_sgpr_stride(origin: SgprOrigin) -> Option<(u32, u32)> {
    match origin {
        SgprOrigin::UserTable {
            user_sgpr,
            byte_offset: 0,
        } => Some((user_sgpr, 1)),
        SgprOrigin::ScaledUserSgpr {
            user_sgpr, stride, ..
        } => Some((user_sgpr, stride)),
        _ => None,
    }
}

fn make_scaled_user_sgpr(user_sgpr: u32, stride: u32) -> SgprOrigin {
    if stride == 0 {
        return SgprOrigin::Unknown;
    }
    SgprOrigin::ScaledUserSgpr {
        user_sgpr,
        stride,
        shift: if stride.is_power_of_two() {
            stride.trailing_zeros()
        } else {
            0
        },
    }
}

fn scale_user_sgpr_by_shift(origin: SgprOrigin, shift: u32) -> SgprOrigin {
    if shift >= 32 {
        return SgprOrigin::Unknown;
    }
    let mult = 1u32 << shift;
    match origin {
        SgprOrigin::UserTable {
            user_sgpr,
            byte_offset: 0,
        } => SgprOrigin::ScaledUserSgpr {
            user_sgpr,
            stride: mult,
            shift,
        },
        SgprOrigin::ScaledUserSgpr {
            user_sgpr, stride, ..
        } => match stride.checked_mul(mult) {
            Some(new_stride) if new_stride > 0 => make_scaled_user_sgpr(user_sgpr, new_stride),
            _ => SgprOrigin::Unknown,
        },
        _ => SgprOrigin::Unknown,
    }
}

fn scale_user_sgpr_by_mul(origin: SgprOrigin, factor: i32) -> SgprOrigin {
    if factor <= 0 {
        return SgprOrigin::Unknown;
    }
    let f = factor as u32;
    match origin {
        SgprOrigin::UserTable {
            user_sgpr,
            byte_offset: 0,
        } => make_scaled_user_sgpr(user_sgpr, f),
        SgprOrigin::ScaledUserSgpr {
            user_sgpr, stride, ..
        } => match stride.checked_mul(f) {
            Some(new_stride) if new_stride > 0 => make_scaled_user_sgpr(user_sgpr, new_stride),
            _ => SgprOrigin::Unknown,
        },
        _ => SgprOrigin::Unknown,
    }
}

fn sop1_writes_pair(op: u8) -> bool {
    matches!(
        op,
        0x04 | 0x06
            | 0x08
            | 0x0A
            | 0x0C
            | 0x1C
            | 0x1E
            | 0x1F
            | 0x21
            | 0x24..=0x2B
            | 0x2D
            | 0x2F
            | 0x31
            | 0x37..=0x3B
    )
}

fn sop1_reads_pair(op: u8) -> bool {
    matches!(
        op,
        0x04 | 0x06
            | 0x08
            | 0x0A
            | 0x0C
            | 0x0E
            | 0x10
            | 0x12
            | 0x14
            | 0x16
            | 0x18
            | 0x20
            | 0x22
            | 0x24..=0x2B
            | 0x2D
            | 0x2F
            | 0x31
            | 0x37..=0x3A
    )
}

fn taint_sgpr_if_loaded(
    collector: &mut Option<&mut AnalysisCollector>,
    origins: &[SgprOrigin; 106],
    src: u16,
) {
    if let Some(col) = collector.as_deref_mut()
        && src <= 105
    {
        match origins[src as usize] {
            SgprOrigin::TableDesc { def_inst_idx, .. }
            | SgprOrigin::DynamicHeapDesc { def_inst_idx, .. }
            | SgprOrigin::RawTableDesc { def_inst_idx, .. }
            | SgprOrigin::VbDesc { def_inst_idx, .. } => {
                col.tainted_defs.insert(def_inst_idx);
            }
            _ => {}
        }
    }
}

fn clear_vgpr_origins(state: &mut OriginState, vdst: u8) {
    state.vgpr_uniform_origins[vdst as usize] = SgprOrigin::Unknown;
    if !state.lane_origins.is_empty() {
        state.lane_origins.retain(|&(v, _), _| v != vdst);
    }
}

fn copy_vgpr_origins(state: &mut OriginState, vdst: u8, vsrc: u8) {
    if vdst == vsrc {
        return;
    }
    state.vgpr_uniform_origins[vdst as usize] = state.vgpr_uniform_origins[vsrc as usize];
    if !state.lane_origins.is_empty() {
        state.lane_origins.retain(|&(v, _), _| v != vdst);
        let copied: Vec<(u8, SgprOrigin)> = state
            .lane_origins
            .range((vsrc, 0)..=(vsrc, u8::MAX))
            .map(|(&(_, lane), &orig)| (lane, orig))
            .collect();
        for (lane, orig) in copied {
            state.lane_origins.insert((vdst, lane), orig);
        }
    }
}

fn swap_vgpr_origins(state: &mut OriginState, va: u8, vb: u8) {
    if va == vb {
        return;
    }
    state.vgpr_uniform_origins.swap(va as usize, vb as usize);
    if !state.lane_origins.is_empty() {
        let a_lanes: Vec<(u8, SgprOrigin)> = state
            .lane_origins
            .range((va, 0)..=(va, u8::MAX))
            .map(|(&(_, lane), &orig)| (lane, orig))
            .collect();
        let b_lanes: Vec<(u8, SgprOrigin)> = state
            .lane_origins
            .range((vb, 0)..=(vb, u8::MAX))
            .map(|(&(_, lane), &orig)| (lane, orig))
            .collect();
        state.lane_origins.retain(|&(v, _), _| v != va && v != vb);
        for (lane, orig) in a_lanes {
            state.lane_origins.insert((vb, lane), orig);
        }
        for (lane, orig) in b_lanes {
            state.lane_origins.insert((va, lane), orig);
        }
    }
}

fn is_synthesized_root_vsharp(origins: &[SgprOrigin; 106], rsrc_sgpr: u8) -> bool {
    let idx = rsrc_sgpr as usize;
    idx + 3 < origins.len()
        && (matches!(origins[idx + 2], SgprOrigin::ConstImmediate { .. })
            || matches!(origins[idx + 3], SgprOrigin::ConstImmediate { .. }))
}

/// Resolve the 4-SGPR V# descriptor origin at `s[rsrc_sgpr .. rsrc_sgpr+3]`.
///
/// Checks word 0 (`origins[rsrc_sgpr]`) first; when word 0 is `Unknown`/`Unwritten` (e.g.
/// after dynamic base-address arithmetic or partial self-overwrite across a CFG merge), falls
/// back to matching consecutive `TableDesc` / `DynamicHeapDesc` / `VbDesc` origins in words 2 & 3
/// (`num_records` and `rsrc_word3`) from the same defining instruction.
fn resolve_vsharp_origin(origins: &[SgprOrigin; 106], rsrc_sgpr: u8) -> SgprOrigin {
    let idx = rsrc_sgpr as usize;
    if idx >= origins.len() {
        return SgprOrigin::Unknown;
    }
    let w0 = origins[idx];
    if !matches!(w0, SgprOrigin::Unknown | SgprOrigin::Unwritten) {
        return w0;
    }
    if idx + 3 < origins.len() {
        match (origins[idx + 2], origins[idx + 3]) {
            (
                SgprOrigin::TableDesc {
                    def_inst_idx: d2,
                    user_sgpr: u2,
                    table_byte_offset: o2,
                    ..
                },
                SgprOrigin::TableDesc {
                    def_inst_idx: d3,
                    user_sgpr: u3,
                    table_byte_offset: o3,
                    ..
                },
            ) if d2 == d3 && u2 == u3 && o2 >= 8 && o3 == o2 + 4 => {
                return SgprOrigin::TableDesc {
                    def_inst_idx: d2,
                    user_sgpr: u2,
                    table_byte_offset: o2 - 8,
                    addr_offset: 0,
                };
            }
            (
                SgprOrigin::DynamicHeapDesc {
                    def_inst_idx: d2,
                    origin: h2,
                },
                SgprOrigin::DynamicHeapDesc {
                    def_inst_idx: d3,
                    origin: h3,
                },
            ) if d2 == d3 && h2 == h3 => {
                return SgprOrigin::DynamicHeapDesc {
                    def_inst_idx: d2,
                    origin: h2,
                };
            }
            (
                SgprOrigin::VbDesc {
                    def_inst_idx: d2,
                    vb_slot: v2,
                },
                SgprOrigin::VbDesc {
                    def_inst_idx: d3,
                    vb_slot: v3,
                },
            ) if d2 == d3 && v2 == v3 => {
                return SgprOrigin::VbDesc {
                    def_inst_idx: d2,
                    vb_slot: v2,
                };
            }
            _ => {}
        }
    }
    w0
}

fn step_origin_state(
    state: &mut OriginState,
    inst_idx: usize,
    inst: &Rdna2Instruction,
    stage: ShaderStage,
    mut collector: Option<&mut AnalysisCollector>,
) {
    if inst.truncated {
        return;
    }
    match &inst.detail {
        Rdna2Detail::Sopc {
            op, ssrc0, ssrc1, ..
        } => {
            taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc0));
            taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc1));
            let (pair0, pair1) = sopc_src_pair_widths(*op);
            if pair0 && *ssrc0 <= 104 {
                taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc0 + 1));
            }
            if pair1 && *ssrc1 <= 104 {
                taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc1 + 1));
            }
        }
        Rdna2Detail::Sop1 {
            op,
            sdst,
            ssrc0,
            literal,
        } => {
            // Pure register copies & conditional moves never taint descriptor provenance.
            let is_copy_or_cmov = matches!(*op, 0x03 | 0x04 | 0x05 | 0x06);
            if !is_copy_or_cmov {
                taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc0));
                if sop1_reads_pair(*op) && *ssrc0 <= 104 {
                    taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc0 + 1));
                }
                if matches!(*op, 0x1B..=0x1E) {
                    taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*sdst));
                    if matches!(*op, 0x1C | 0x1E) && *sdst <= 104 {
                        taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*sdst + 1));
                    }
                }
            }
            if sop1_writes_pair(*op) {
                if *sdst <= 104 {
                    let (new_lo, new_hi) = match *op {
                        0x04 => {
                            // s_mov_b64
                            resolve_src_pair_origins(*ssrc0, *literal, inst_idx, &state.origins)
                        }
                        0x06 => {
                            // s_cmov_b64
                            let (src_lo, src_hi) =
                                resolve_src_pair_origins(*ssrc0, *literal, inst_idx, &state.origins);
                            let dst_lo = state.origins[*sdst as usize];
                            let dst_hi = state.origins[(*sdst + 1) as usize];
                            let lo = merge_select_origin(
                                dst_lo,
                                src_lo,
                                inst_idx,
                                collector.as_deref_mut().map(|c| &mut c.def_aliases),
                            );
                            let hi = merge_select_origin(
                                dst_hi,
                                src_hi,
                                inst_idx,
                                collector.as_deref_mut().map(|c| &mut c.def_aliases),
                            );
                            (lo, hi)
                        }
                        0x08 => {
                            // s_not_b64
                            let (src_lo, src_hi) =
                                resolve_src_pair_origins(*ssrc0, *literal, inst_idx, &state.origins);
                            match (src_lo, src_hi) {
                                (
                                    SgprOrigin::ConstImmediate { value: lo, .. },
                                    SgprOrigin::ConstImmediate { value: hi, .. },
                                ) => (
                                    SgprOrigin::ConstImmediate {
                                        def_inst_idx: inst_idx,
                                        value: lo.map(|v| !v),
                                    },
                                    SgprOrigin::ConstImmediate {
                                        def_inst_idx: inst_idx,
                                        value: hi.map(|v| !v),
                                    },
                                ),
                                _ => (SgprOrigin::Unknown, SgprOrigin::Unknown),
                            }
                        }
                        0x0C => {
                            // s_brev_b64
                            let (src_lo, src_hi) =
                                resolve_src_pair_origins(*ssrc0, *literal, inst_idx, &state.origins);
                            match (src_lo, src_hi) {
                                (
                                    SgprOrigin::ConstImmediate { value: lo, .. },
                                    SgprOrigin::ConstImmediate { value: hi, .. },
                                ) => {
                                    let rev = lo.zip(hi).map(|(l, h)| {
                                        ((u64::from(h) << 32) | u64::from(l)).reverse_bits()
                                    });
                                    (
                                        SgprOrigin::ConstImmediate {
                                            def_inst_idx: inst_idx,
                                            value: rev.map(|v| v as u32),
                                        },
                                        SgprOrigin::ConstImmediate {
                                            def_inst_idx: inst_idx,
                                            value: rev.map(|v| (v >> 32) as u32),
                                        },
                                    )
                                }
                                _ => (SgprOrigin::Unknown, SgprOrigin::Unknown),
                            }
                        }
                        0x1C | 0x1E => {
                            // s_bitset0_b64 / s_bitset1_b64
                            let dst_lo = state.origins[*sdst as usize];
                            let dst_hi = state.origins[(*sdst + 1) as usize];
                            if let (
                                SgprOrigin::ConstImmediate { value: lo_v, .. },
                                SgprOrigin::ConstImmediate { value: hi_v, .. },
                            ) = (dst_lo, dst_hi)
                                && is_const_src(*ssrc0, &state.origins)
                            {
                                let bit = resolve_const_u32(*ssrc0, *literal, &state.origins);
                                let val64 = lo_v.zip(hi_v).zip(bit).map(|((l, h), b)| {
                                    let cur = (u64::from(h) << 32) | u64::from(l);
                                    let mask = 1u64 << (b & 63);
                                    if *op == 0x1C { cur & !mask } else { cur | mask }
                                });
                                (
                                    SgprOrigin::ConstImmediate {
                                        def_inst_idx: inst_idx,
                                        value: val64.map(|v| v as u32),
                                    },
                                    SgprOrigin::ConstImmediate {
                                        def_inst_idx: inst_idx,
                                        value: val64.map(|v| (v >> 32) as u32),
                                    },
                                )
                            } else {
                                (SgprOrigin::Unknown, SgprOrigin::Unknown)
                            }
                        }
                        _ => (SgprOrigin::Unknown, SgprOrigin::Unknown),
                    };
                    state.origins[*sdst as usize] = new_lo;
                    state.origins[(*sdst + 1) as usize] = new_hi;
                } else if *sdst == 105 {
                    state.origins[105] = SgprOrigin::Unknown;
                }
            } else if *sdst <= 105 {
                let prev = state.origins[*sdst as usize];
                let next = match *op {
                    0x03 => {
                        // s_mov_b32
                        resolve_src_origin_u8(*ssrc0, *literal, inst_idx, &state.origins)
                    }
                    0x05 => {
                        // s_cmov_b32
                        let src = resolve_src_origin_u8(*ssrc0, *literal, inst_idx, &state.origins);
                        merge_select_origin(
                            prev,
                            src,
                            inst_idx,
                            collector.as_deref_mut().map(|c| &mut c.def_aliases),
                        )
                    }
                    0x07 => {
                        // s_not_b32
                        if let Some(v) = resolve_const_u32(*ssrc0, *literal, &state.origins) {
                            SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(!v),
                            }
                        } else if is_const_src(*ssrc0, &state.origins) {
                            SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            }
                        } else {
                            SgprOrigin::Unknown
                        }
                    }
                    0x0B => {
                        // s_brev_b32
                        resolve_const_u32(*ssrc0, *literal, &state.origins).map_or(
                            SgprOrigin::Unknown,
                            |v| SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v.reverse_bits()),
                            },
                        )
                    }
                    0x19 => {
                        // s_sext_i32_i8
                        resolve_const_u32(*ssrc0, *literal, &state.origins).map_or(
                            SgprOrigin::Unknown,
                            |v| SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(i32::from(v as i8) as u32),
                            },
                        )
                    }
                    0x1A => {
                        // s_sext_i32_i16
                        resolve_const_u32(*ssrc0, *literal, &state.origins).map_or(
                            SgprOrigin::Unknown,
                            |v| SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(i32::from(v as i16) as u32),
                            },
                        )
                    }
                    0x1B => {
                        // s_bitset0_b32
                        if let SgprOrigin::ConstImmediate { value, .. } = prev
                            && is_const_src(*ssrc0, &state.origins)
                        {
                            let bit = resolve_const_u32(*ssrc0, *literal, &state.origins);
                            SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: value
                                    .zip(bit)
                                    .map(|(v, b)| v & !(1u32 << (b & 31))),
                            }
                        } else {
                            SgprOrigin::Unknown
                        }
                    }
                    0x1D => {
                        // s_bitset1_b32
                        if let SgprOrigin::ConstImmediate { value, .. } = prev
                            && is_const_src(*ssrc0, &state.origins)
                        {
                            let bit = resolve_const_u32(*ssrc0, *literal, &state.origins);
                            SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: value
                                    .zip(bit)
                                    .map(|(v, b)| v | (1u32 << (b & 31))),
                            }
                        } else {
                            SgprOrigin::Unknown
                        }
                    }
                    0x34 => {
                        // s_abs_i32
                        resolve_const_u32(*ssrc0, *literal, &state.origins).map_or(
                            SgprOrigin::Unknown,
                            |v| SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some((v as i32).wrapping_abs() as u32),
                            },
                        )
                    }
                    _ => SgprOrigin::Unknown,
                };
                state.origins[*sdst as usize] = next;
            }
        }
        Rdna2Detail::Sopk {
            op, sdst, simm16, ..
        } => {
            if matches!(*op, 0x03..=0x0E | 0x10) {
                taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*sdst));
            }
            if !matches!(*op, 0x03..=0x0E | 0x17..=0x1A) && *sdst <= 105 {
                let prev_origin = state.origins[*sdst as usize];
                let imm = i32::from(*simm16 as i16);
                state.origins[*sdst as usize] = match *op {
                    0x00 => {
                        // s_movk_i32
                        SgprOrigin::ConstImmediate {
                            def_inst_idx: inst_idx,
                            value: Some(imm as u32),
                        }
                    }
                    0x02 => {
                        // s_cmovk_i32
                        merge_select_origin(
                            prev_origin,
                            SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(imm as u32),
                            },
                            inst_idx,
                            collector.as_deref_mut().map(|c| &mut c.def_aliases),
                        )
                    }
                    0x0F => {
                        // s_addk_i32
                        if let SgprOrigin::ConstImmediate { value, .. } = prev_origin {
                            SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: value.map(|v| v.wrapping_add(imm as u32)),
                            }
                        } else {
                            apply_const_offset(prev_origin, imm)
                        }
                    }
                    0x10 => {
                        // s_mulk_i32
                        if let SgprOrigin::ConstImmediate { value, .. } = prev_origin {
                            SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: value.map(|v| v.wrapping_mul(imm as u32)),
                            }
                        } else {
                            scale_user_sgpr_by_mul(prev_origin, imm)
                        }
                    }
                    _ => SgprOrigin::Unknown,
                };
            }
        }
        Rdna2Detail::Sop2 {
            op,
            sdst,
            ssrc0,
            ssrc1,
            literal,
        } => {
            let (pair0, pair1) = sop2_src_pair_widths(*op);
            // Address arithmetic, selects, and alignment masks on pointers do not taint loaded
            // descriptors (if the result is later consumed by math/compare/VALU, that taints it).
            let is_addr_or_select_op = matches!(
                *op,
                0x00..=0x05
                    | 0x0A
                    | 0x0B
                    | 0x0E
                    | 0x0F
                    | 0x10
                    | 0x11
                    | 0x14
                    | 0x15
                    | 0x2E..=0x31
            );
            if !is_addr_or_select_op {
                taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc0));
                taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc1));
                if pair0 && *ssrc0 <= 104 {
                    taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc0 + 1));
                }
                if pair1 && *ssrc1 <= 104 {
                    taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*ssrc1 + 1));
                }
            }
            if *op == 0x25 && *sdst <= 104 {
                // s_bfm_b64 writes a 64-bit SGPR pair from two 32-bit operands (frequently
                // used to materialize inline static sampler S# descriptors).
                let (lo_orig, hi_orig) = if is_const_src(*ssrc0, &state.origins)
                    && is_const_src(*ssrc1, &state.origins)
                {
                    let val64 = resolve_const_u32(*ssrc0, *literal, &state.origins)
                        .zip(resolve_const_u32(*ssrc1, *literal, &state.origins))
                        .map(|(w, o)| {
                            let width = w & 63;
                            let off = o & 63;
                            let mask = if width == 0 {
                                0u64
                            } else {
                                (1u64 << width).wrapping_sub(1)
                            };
                            mask.wrapping_shl(off)
                        });
                    (
                        SgprOrigin::ConstImmediate {
                            def_inst_idx: inst_idx,
                            value: val64.map(|v| v as u32),
                        },
                        SgprOrigin::ConstImmediate {
                            def_inst_idx: inst_idx,
                            value: val64.map(|v| (v >> 32) as u32),
                        },
                    )
                } else {
                    (SgprOrigin::Unknown, SgprOrigin::Unknown)
                };
                state.origins[*sdst as usize] = lo_orig;
                state.origins[(*sdst + 1) as usize] = hi_orig;
            } else if pair0 && *sdst <= 104 {
                let (s0_lo, s0_hi) =
                    resolve_src_pair_origins(*ssrc0, *literal, inst_idx, &state.origins);
                let (s1_lo, s1_hi) = if pair1 {
                    resolve_src_pair_origins(*ssrc1, *literal, inst_idx, &state.origins)
                } else {
                    let s1 = resolve_src_origin_u8(*ssrc1, *literal, inst_idx, &state.origins);
                    (s1, SgprOrigin::Unknown)
                };
                let (new_lo, new_hi) = match *op {
                    0x0B => {
                        // s_cselect_b64
                        let lo = merge_select_origin(
                            s0_lo,
                            s1_lo,
                            inst_idx,
                            collector.as_deref_mut().map(|c| &mut c.def_aliases),
                        );
                        let hi = merge_select_origin(
                            s0_hi,
                            s1_hi,
                            inst_idx,
                            collector.as_deref_mut().map(|c| &mut c.def_aliases),
                        );
                        (lo, hi)
                    }
                    0x0F => {
                        // s_and_b64
                        let lo = match (s0_lo, s1_lo) {
                            (SgprOrigin::ConstImmediate { value: Some(m), .. }, other)
                            | (other, SgprOrigin::ConstImmediate { value: Some(m), .. }) => {
                                apply_and_const(other, m, inst_idx)
                            }
                            (
                                SgprOrigin::ConstImmediate { .. },
                                SgprOrigin::ConstImmediate { .. },
                            ) => SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            },
                            _ => SgprOrigin::Unknown,
                        };
                        let hi = match (s0_hi, s1_hi) {
                            (SgprOrigin::ConstImmediate { value: Some(m), .. }, other)
                            | (other, SgprOrigin::ConstImmediate { value: Some(m), .. }) => {
                                apply_and_const(other, m, inst_idx)
                            }
                            (
                                SgprOrigin::ConstImmediate { .. },
                                SgprOrigin::ConstImmediate { .. },
                            ) => SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            },
                            _ => SgprOrigin::Unknown,
                        };
                        (lo, hi)
                    }
                    0x11 => {
                        // s_or_b64
                        let lo = match (s0_lo, s1_lo) {
                            (SgprOrigin::ConstImmediate { value: Some(m), .. }, other)
                            | (other, SgprOrigin::ConstImmediate { value: Some(m), .. }) => {
                                apply_or_const(other, m, inst_idx)
                            }
                            (
                                SgprOrigin::ConstImmediate { .. },
                                SgprOrigin::ConstImmediate { .. },
                            ) => SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            },
                            _ => SgprOrigin::Unknown,
                        };
                        let hi = match (s0_hi, s1_hi) {
                            (SgprOrigin::ConstImmediate { value: Some(m), .. }, other)
                            | (other, SgprOrigin::ConstImmediate { value: Some(m), .. }) => {
                                apply_or_const(other, m, inst_idx)
                            }
                            (
                                SgprOrigin::ConstImmediate { .. },
                                SgprOrigin::ConstImmediate { .. },
                            ) => SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            },
                            _ => SgprOrigin::Unknown,
                        };
                        (lo, hi)
                    }
                    0x15 => {
                        // s_andn2_b64: ssrc0 & !ssrc1
                        let lo = match s1_lo {
                            SgprOrigin::ConstImmediate { value: Some(m), .. } => {
                                apply_and_const(s0_lo, !m, inst_idx)
                            }
                            SgprOrigin::ConstImmediate { .. }
                                if matches!(s0_lo, SgprOrigin::ConstImmediate { .. }) =>
                            {
                                SgprOrigin::ConstImmediate {
                                    def_inst_idx: inst_idx,
                                    value: None,
                                }
                            }
                            _ => SgprOrigin::Unknown,
                        };
                        let hi = match s1_hi {
                            SgprOrigin::ConstImmediate { value: Some(m), .. } => {
                                apply_and_const(s0_hi, !m, inst_idx)
                            }
                            SgprOrigin::ConstImmediate { .. }
                                if matches!(s0_hi, SgprOrigin::ConstImmediate { .. }) =>
                            {
                                SgprOrigin::ConstImmediate {
                                    def_inst_idx: inst_idx,
                                    value: None,
                                }
                            }
                            _ => SgprOrigin::Unknown,
                        };
                        (lo, hi)
                    }
                    0x1F => {
                        // s_lshl_b64: 64-bit heap-index shift `(u64(idx) << shift)`.
                        if let Some(raw_sh) = resolve_const_u32(*ssrc1, *literal, &state.origins) {
                            let sh = raw_sh & 63;
                            if let (
                                SgprOrigin::ConstImmediate { value: lo_v, .. },
                                SgprOrigin::ConstImmediate { value: hi_v, .. },
                            ) = (s0_lo, s0_hi)
                            {
                                let val64 = lo_v.zip(hi_v).map(|(l, h)| {
                                    ((u64::from(h) << 32) | u64::from(l)).wrapping_shl(sh)
                                });
                                (
                                    SgprOrigin::ConstImmediate {
                                        def_inst_idx: inst_idx,
                                        value: val64.map(|v| v as u32),
                                    },
                                    SgprOrigin::ConstImmediate {
                                        def_inst_idx: inst_idx,
                                        value: val64.map(|v| (v >> 32) as u32),
                                    },
                                )
                            } else if sh < 32 {
                                (
                                    scale_user_sgpr_by_shift(s0_lo, sh),
                                    SgprOrigin::Unknown,
                                )
                            } else {
                                (SgprOrigin::Unknown, SgprOrigin::Unknown)
                            }
                        } else {
                            (SgprOrigin::Unknown, SgprOrigin::Unknown)
                        }
                    }
                    0x13 | 0x17 | 0x19 | 0x1B | 0x1D => {
                        // 64-bit bitwise ops: s_xor_b64, s_orn2_b64, s_nand_b64, s_nor_b64, s_xnor_b64
                        if *op == 0x13 && *ssrc0 == *ssrc1 && *ssrc0 <= 104 {
                            let z = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(0),
                            };
                            (z, z)
                        } else if let (
                            SgprOrigin::ConstImmediate { value: lo0, .. },
                            SgprOrigin::ConstImmediate { value: hi0, .. },
                            SgprOrigin::ConstImmediate { value: lo1, .. },
                            SgprOrigin::ConstImmediate { value: hi1, .. },
                        ) = (s0_lo, s0_hi, s1_lo, s1_hi)
                        {
                            let fold32 = |a: u32, b: u32| match *op {
                                0x13 => a ^ b,
                                0x17 => a | !b,
                                0x19 => !(a & b),
                                0x1B => !(a | b),
                                _ => !(a ^ b),
                            };
                            (
                                SgprOrigin::ConstImmediate {
                                    def_inst_idx: inst_idx,
                                    value: lo0.zip(lo1).map(|(a, b)| fold32(a, b)),
                                },
                                SgprOrigin::ConstImmediate {
                                    def_inst_idx: inst_idx,
                                    value: hi0.zip(hi1).map(|(a, b)| fold32(a, b)),
                                },
                            )
                        } else {
                            (SgprOrigin::Unknown, SgprOrigin::Unknown)
                        }
                    }
                    0x21 | 0x23 | 0x29 | 0x2A => {
                        // s_lshr_b64, s_ashr_i64, s_bfe_u64, s_bfe_i64
                        if let (
                            SgprOrigin::ConstImmediate { value: lo0, .. },
                            SgprOrigin::ConstImmediate { value: hi0, .. },
                            SgprOrigin::ConstImmediate { value: s1_v, .. },
                        ) = (s0_lo, s0_hi, s1_lo)
                        {
                            let val64 = lo0.zip(hi0).zip(s1_v).map(|((l, h), s1)| {
                                let x = (u64::from(h) << 32) | u64::from(l);
                                match *op {
                                    0x21 => x.wrapping_shr(s1 & 63),
                                    0x23 => ((x as i64).wrapping_shr(s1 & 63)) as u64,
                                    0x29 => {
                                        let off = s1 & 63;
                                        let width = (s1 >> 16) & 0x7F;
                                        if width == 0 {
                                            0
                                        } else if width >= 64 {
                                            x.wrapping_shr(off)
                                        } else {
                                            x.wrapping_shr(off) & ((1u64 << width) - 1)
                                        }
                                    }
                                    _ => {
                                        // 0x2A: s_bfe_i64
                                        let off = s1 & 63;
                                        let width = (s1 >> 16) & 0x7F;
                                        if width == 0 {
                                            0
                                        } else if width >= 64 {
                                            ((x as i64).wrapping_shr(off)) as u64
                                        } else {
                                            let sh = 64 - width;
                                            let ext = x.wrapping_shr(off);
                                            (((ext.wrapping_shl(sh)) as i64).wrapping_shr(sh)) as u64
                                        }
                                    }
                                }
                            });
                            (
                                SgprOrigin::ConstImmediate {
                                    def_inst_idx: inst_idx,
                                    value: val64.map(|v| v as u32),
                                },
                                SgprOrigin::ConstImmediate {
                                    def_inst_idx: inst_idx,
                                    value: val64.map(|v| (v >> 32) as u32),
                                },
                            )
                        } else {
                            (SgprOrigin::Unknown, SgprOrigin::Unknown)
                        }
                    }
                    _ => (SgprOrigin::Unknown, SgprOrigin::Unknown),
                };
                state.origins[*sdst as usize] = new_lo;
                state.origins[(*sdst + 1) as usize] = new_hi;
            } else if *sdst <= 105 {
                let src0_origin = if *ssrc0 <= 105 {
                    state.origins[*ssrc0 as usize]
                } else {
                    SgprOrigin::Unknown
                };
                let src1_origin = if *ssrc1 <= 105 {
                    state.origins[*ssrc1 as usize]
                } else {
                    SgprOrigin::Unknown
                };
                let c0 = resolve_const_u32(*ssrc0, *literal, &state.origins);
                let c1 = resolve_const_u32(*ssrc1, *literal, &state.origins);
                let mut next_origin = SgprOrigin::Unknown;
                match *op {
                    // s_add_u32 / s_add_i32: constant fold, table pointer + offset,
                    // heap base + scaled index, or index doubling (`idx + idx`).
                    0x00 | 0x02 => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0.wrapping_add(v1)),
                            };
                        } else if let Some(imm) = c1 {
                            next_origin = apply_const_offset(src0_origin, imm as i32);
                        } else if let Some(imm) = c0 {
                            next_origin = apply_const_offset(src1_origin, imm as i32);
                        } else if let (
                            SgprOrigin::UserTable {
                                user_sgpr: base_user_sgpr,
                                byte_offset: 0,
                            },
                            SgprOrigin::ScaledUserSgpr {
                                user_sgpr: index_user_sgpr,
                                stride,
                                shift: index_shift,
                            },
                        ) = (src0_origin, src1_origin)
                        {
                            if base_user_sgpr == index_user_sgpr {
                                if let Some(new_stride) = stride.checked_add(1) {
                                    next_origin = make_scaled_user_sgpr(index_user_sgpr, new_stride);
                                }
                            } else {
                                next_origin = SgprOrigin::DynamicHeapPtr(DynamicHeapOrigin {
                                    base_user_sgpr,
                                    stride,
                                    index_user_sgpr,
                                    index_shift,
                                });
                            }
                        } else if let (
                            SgprOrigin::ScaledUserSgpr {
                                user_sgpr: index_user_sgpr,
                                stride,
                                shift: index_shift,
                            },
                            SgprOrigin::UserTable {
                                user_sgpr: base_user_sgpr,
                                byte_offset: 0,
                            },
                        ) = (src0_origin, src1_origin)
                        {
                            if base_user_sgpr == index_user_sgpr {
                                if let Some(new_stride) = stride.checked_add(1) {
                                    next_origin = make_scaled_user_sgpr(index_user_sgpr, new_stride);
                                }
                            } else {
                                next_origin = SgprOrigin::DynamicHeapPtr(DynamicHeapOrigin {
                                    base_user_sgpr,
                                    stride,
                                    index_user_sgpr,
                                    index_shift,
                                });
                            }
                        } else if let (Some((u0, s0)), Some((u1, s1))) = (
                            scaled_user_sgpr_stride(src0_origin),
                            scaled_user_sgpr_stride(src1_origin),
                        ) && u0 == u1
                            && let Some(new_stride) = s0.checked_add(s1)
                        {
                            next_origin = make_scaled_user_sgpr(u0, new_stride);
                        }
                    }
                    // s_sub_u32 / s_sub_i32: constant fold, self-sub zero, or table pointer - constant offset.
                    0x01 | 0x03 => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0.wrapping_sub(v1)),
                            };
                        } else if *ssrc0 == *ssrc1 && *ssrc0 <= 105 {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(0),
                            };
                        } else if let Some(imm) = c1 {
                            next_origin = apply_const_offset(src0_origin, (imm as i32).wrapping_neg());
                        }
                    }
                    // s_min_i32 / s_min_u32 / s_max_i32 / s_max_u32
                    0x06..=0x09 => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            let folded = match *op {
                                0x06 => (v0 as i32).min(v1 as i32) as u32,
                                0x07 => v0.min(v1),
                                0x08 => (v0 as i32).max(v1 as i32) as u32,
                                _ => v0.max(v1),
                            };
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(folded),
                            };
                        }
                    }
                    // s_cselect_b32
                    0x0A => {
                        let s0 = resolve_src_origin_u8(*ssrc0, *literal, inst_idx, &state.origins);
                        let s1 = resolve_src_origin_u8(*ssrc1, *literal, inst_idx, &state.origins);
                        next_origin = merge_select_origin(
                            s0,
                            s1,
                            inst_idx,
                            collector.as_deref_mut().map(|c| &mut c.def_aliases),
                        );
                    }
                    // s_and_b32
                    0x0E => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0 & v1),
                            };
                        } else if let Some(m) = c1 {
                            next_origin = apply_and_const(src0_origin, m, inst_idx);
                        } else if let Some(m) = c0 {
                            next_origin = apply_and_const(src1_origin, m, inst_idx);
                        } else if is_const_src(*ssrc0, &state.origins)
                            && is_const_src(*ssrc1, &state.origins)
                        {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            };
                        }
                    }
                    // s_andn2_b32: `ssrc0 & !ssrc1` (e.g. `s_andn2_b32 s12, s8, 3` == `s8 & ~3`).
                    0x14 => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0 & !v1),
                            };
                        } else if let Some(m) = c1 {
                            next_origin = apply_and_const(src0_origin, !m, inst_idx);
                        } else if is_const_src(*ssrc0, &state.origins)
                            && is_const_src(*ssrc1, &state.origins)
                        {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            };
                        }
                    }
                    // s_or_b32: constant fold or `or disjoint` table-offset addition.
                    0x10 => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0 | v1),
                            };
                        } else if let Some(imm) = c1 {
                            next_origin = apply_or_const(src0_origin, imm, inst_idx);
                        } else if let Some(imm) = c0 {
                            next_origin = apply_or_const(src1_origin, imm, inst_idx);
                        } else if is_const_src(*ssrc0, &state.origins)
                            && is_const_src(*ssrc1, &state.origins)
                        {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            };
                        }
                    }
                    // Other bitwise ops on constants (inline static sampler construction).
                    0x12 | 0x16 | 0x18 | 0x1A | 0x1C => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            let folded = match *op {
                                0x12 => v0 ^ v1,
                                0x16 => v0 | !v1,
                                0x18 => !(v0 & v1),
                                0x1A => !(v0 | v1),
                                _ => !(v0 ^ v1),
                            };
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(folded),
                            };
                        } else if *op == 0x12 && *ssrc0 == *ssrc1 && *ssrc0 <= 105 {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(0),
                            };
                        } else if is_const_src(*ssrc0, &state.origins)
                            && is_const_src(*ssrc1, &state.origins)
                        {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: None,
                            };
                        }
                    }
                    // s_lshl_b32: constant fold or scale user SGPR index by `1 << shift`.
                    0x1E => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0.wrapping_shl(v1 & 31)),
                            };
                        } else if let Some(raw_shift) = c1 {
                            next_origin = scale_user_sgpr_by_shift(src0_origin, raw_shift & 31);
                        }
                    }
                    // s_lshr_b32 / s_ashr_i32 on constants
                    0x20 => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0.wrapping_shr(v1 & 31)),
                            };
                        }
                    }
                    0x22 => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(((v0 as i32).wrapping_shr(v1 & 31)) as u32),
                            };
                        }
                    }
                    // s_bfm_b32 with constant operands (common in inline static sampler S#).
                    0x24 if is_const_src(*ssrc0, &state.origins)
                        && is_const_src(*ssrc1, &state.origins) =>
                    {
                        let folded = c0.zip(c1).map(|(w, o)| {
                            let width = w & 31;
                            let off = o & 31;
                            let mask = if width == 0 {
                                0u32
                            } else {
                                (1u32 << width).wrapping_sub(1)
                            };
                            mask.wrapping_shl(off)
                        });
                        next_origin = SgprOrigin::ConstImmediate {
                            def_inst_idx: inst_idx,
                            value: folded,
                        };
                    }
                    // s_mul_i32: constant fold or scale user SGPR index by constant stride.
                    0x26 => {
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0.wrapping_mul(v1)),
                            };
                        } else if let Some(imm) = c1 {
                            next_origin = scale_user_sgpr_by_mul(src0_origin, imm as i32);
                        } else if let Some(imm) = c0 {
                            next_origin = scale_user_sgpr_by_mul(src1_origin, imm as i32);
                        }
                    }
                    // s_lshl{1,2,3,4}_add_u32: `(ssrc0 << k) + ssrc1`.
                    // Covers both SM 6.6 heap addressing (`(idx << k) + base`), shift-add stride
                    // scaling (`(idx << k) + idx`), and compact table-offset adds (`(imm << k) + base`,
                    // e.g. `s_lshl4_add_u32 s12, 6, s8` for `s8 + 96` or `8, s8` for `s8 + 128`).
                    0x2E..=0x31 => {
                        let shift = u32::from(*op - 0x2D);
                        if let (Some(v0), Some(v1)) = (c0, c1) {
                            next_origin = SgprOrigin::ConstImmediate {
                                def_inst_idx: inst_idx,
                                value: Some(v0.wrapping_shl(shift).wrapping_add(v1)),
                            };
                        } else if let Some(v0) = c0 {
                            let delta = (v0 as i32).wrapping_shl(shift);
                            next_origin = apply_const_offset(src1_origin, delta);
                        } else if let SgprOrigin::ScaledUserSgpr {
                            user_sgpr: index_user_sgpr,
                            stride,
                            shift: index_shift,
                        } = scale_user_sgpr_by_shift(src0_origin, shift)
                        {
                            if let Some((u1, s1)) = scaled_user_sgpr_stride(src1_origin)
                                && u1 == index_user_sgpr
                                && let Some(new_stride) = stride.checked_add(s1)
                            {
                                next_origin = make_scaled_user_sgpr(index_user_sgpr, new_stride);
                            } else if let SgprOrigin::UserTable {
                                user_sgpr: base_user_sgpr,
                                byte_offset: 0,
                            } = src1_origin
                            {
                                next_origin = SgprOrigin::DynamicHeapPtr(DynamicHeapOrigin {
                                    base_user_sgpr,
                                    stride,
                                    index_user_sgpr,
                                    index_shift,
                                });
                            } else if c1 == Some(0) {
                                next_origin = SgprOrigin::ScaledUserSgpr {
                                    user_sgpr: index_user_sgpr,
                                    stride,
                                    shift: index_shift,
                                };
                            }
                        }
                    }
                    // Bitfield extract / pack / absdiff / mul_hi on constants (inline S# sampler).
                    0x27 | 0x28 | 0x2C | 0x32..=0x36
                        if is_const_src(*ssrc0, &state.origins)
                            && is_const_src(*ssrc1, &state.origins) =>
                    {
                        let folded = c0.zip(c1).map(|(v0, v1)| match *op {
                            0x27 => {
                                // s_bfe_u32: offset = s1[4:0], width = s1[22:16]
                                let off = v1 & 31;
                                let width = (v1 >> 16) & 0x7F;
                                if width == 0 {
                                    0
                                } else if width >= 32 {
                                    v0.wrapping_shr(off)
                                } else {
                                    v0.wrapping_shr(off) & ((1u32 << width) - 1)
                                }
                            }
                            0x28 => {
                                // s_bfe_i32: signed bitfield extract
                                let off = v1 & 31;
                                let width = (v1 >> 16) & 0x7F;
                                if width == 0 {
                                    0
                                } else if width >= 32 {
                                    ((v0 as i32).wrapping_shr(off)) as u32
                                } else {
                                    let sh = 32 - width;
                                    let ext = v0.wrapping_shr(off);
                                    (((ext.wrapping_shl(sh)) as i32).wrapping_shr(sh)) as u32
                                }
                            }
                            0x2C => {
                                // s_absdiff_i32
                                (v0 as i32).wrapping_sub(v1 as i32).wrapping_abs() as u32
                            }
                            0x32 => {
                                // s_pack_ll_b32_b16: (v0[15:0]) | (v1[15:0] << 16)
                                (v0 & 0x0000_FFFF) | ((v1 & 0x0000_FFFF) << 16)
                            }
                            0x33 => {
                                // s_pack_lh_b32_b16: (v0[15:0]) | (v1[31:16] << 16)
                                (v0 & 0x0000_FFFF) | (v1 & 0xFFFF_0000)
                            }
                            0x34 => {
                                // s_pack_hh_b32_b16: (v0[31:16]) | (v1[31:16] << 16)
                                (v0 >> 16) | (v1 & 0xFFFF_0000)
                            }
                            0x35 => {
                                // s_mul_hi_u32
                                ((u64::from(v0) * u64::from(v1)) >> 32) as u32
                            }
                            _ => {
                                // 0x36: s_mul_hi_i32
                                ((i64::from(v0 as i32) * i64::from(v1 as i32)) >> 32) as u32
                            }
                        });
                        next_origin = SgprOrigin::ConstImmediate {
                            def_inst_idx: inst_idx,
                            value: folded,
                        };
                    }
                    _ => {}
                }
                state.origins[*sdst as usize] = next_origin;
            }
        }
        Rdna2Detail::Smem {
            op,
            sdata,
            sbase,
            soffset,
            offset,
            ..
        } => {
            let const_soffset: Option<i32> = if *soffset == 0x7D {
                Some(0)
            } else if let Some(v) = decode_inline_u32_src(u16::from(*soffset), None) {
                Some(v)
            } else if *soffset <= 105
                && let SgprOrigin::ConstImmediate { value: Some(v), .. } =
                    state.origins[*soffset as usize]
            {
                Some(v as i32)
            } else {
                None
            };
            let is_dynamic = const_soffset.is_none();
            let sbase_sgpr = sbase.saturating_mul(2);

            // Snapshot base & dynamic-offset origins BEFORE overwriting sdata (since sdata can
            // alias sbase_sgpr and/or soffset, e.g. `s_load_dwordx4 s[8:11], s[8:9], s10`).
            let base_origin = if matches!(*op, 0x08..=0x0C) {
                resolve_vsharp_origin(&state.origins, sbase_sgpr)
            } else if (sbase_sgpr as usize) < state.origins.len() {
                state.origins[sbase_sgpr as usize]
            } else {
                SgprOrigin::Unknown
            };
            let soffset_origin = if is_dynamic && (*soffset as usize) < state.origins.len() {
                state.origins[*soffset as usize]
            } else {
                SgprOrigin::Unknown
            };

            if matches!(*op, 0x08..=0x0C) {
                // s_buffer_load_* consuming a V# in s[sbase_sgpr .. sbase_sgpr+3]
                if is_dynamic {
                    taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*soffset));
                }
                if let Some(col) = collector.as_deref_mut() {
                    match base_origin {
                        SgprOrigin::TableDesc {
                            def_inst_idx,
                            user_sgpr,
                            table_byte_offset,
                            ..
                        } => {
                            let key = CbvKey {
                                base_sbase: 0,
                                user_sgpr: Some(user_sgpr),
                                table_offset: Some(table_byte_offset as i32),
                                dynamic_heap: None,
                            };
                            col.buffer_load_uses.push(BufferLoadUse {
                                inst_idx,
                                def_inst_idx,
                                cbv_key: Some(key),
                                source: Some(WgslBindingSource::Table {
                                    user_sgpr,
                                    table_byte_offset,
                                }),
                            });
                        }
                        SgprOrigin::DynamicHeapDesc {
                            def_inst_idx,
                            origin: heap,
                        } => {
                            let key = CbvKey {
                                base_sbase: 0,
                                user_sgpr: None,
                                table_offset: None,
                                dynamic_heap: Some(heap),
                            };
                            col.buffer_load_uses.push(BufferLoadUse {
                                inst_idx,
                                def_inst_idx,
                                cbv_key: Some(key),
                                source: Some(WgslBindingSource::ResourceHeap {
                                    base_user_sgpr: heap.base_user_sgpr,
                                    stride: heap.stride,
                                    index_user_sgpr: heap.index_user_sgpr,
                                    index_shift: heap.index_shift,
                                }),
                            });
                        }
                        SgprOrigin::RawTableDesc {
                            def_inst_idx,
                            table_sbase,
                            table_offset,
                            is_dynamic: raw_dyn,
                        } => {
                            let cbv_key = if raw_dyn {
                                None
                            } else {
                                Some(CbvKey {
                                    base_sbase: table_sbase,
                                    user_sgpr: None,
                                    table_offset: Some(table_offset),
                                    dynamic_heap: None,
                                })
                            };
                            col.buffer_load_uses.push(BufferLoadUse {
                                inst_idx,
                                def_inst_idx,
                                cbv_key,
                                source: None,
                            });
                        }
                        SgprOrigin::UserTable { user_sgpr, .. } => {
                            col.direct_root_cbvs.push((inst_idx, user_sgpr));
                        }
                        _ => {}
                    }
                }
                if let Some(count) = smem_load_dword_count(*op) {
                    for k in 0..count {
                        let reg = sdata.saturating_add(k);
                        if (reg as usize) < state.origins.len() {
                            state.origins[reg as usize] = SgprOrigin::Unknown;
                        }
                    }
                }
            } else if let Some(count) = smem_load_dword_count(*op) {
                // s_load_dword / x2 / x4 / x8 / x16
                let dynamic_heap_from_soffset = if is_dynamic {
                    match (base_origin, soffset_origin) {
                        (
                            SgprOrigin::UserTable {
                                user_sgpr: base_user_sgpr,
                                byte_offset: 0,
                            },
                            SgprOrigin::ScaledUserSgpr {
                                user_sgpr: index_user_sgpr,
                                stride,
                                shift: index_shift,
                            },
                        ) => Some(DynamicHeapOrigin {
                            base_user_sgpr,
                            stride,
                            index_user_sgpr,
                            index_shift,
                        }),
                        (
                            SgprOrigin::UserTable {
                                user_sgpr: base_user_sgpr,
                                byte_offset: 0,
                            },
                            SgprOrigin::UserTable {
                                user_sgpr: index_user_sgpr,
                                byte_offset: 0,
                            },
                        ) => Some(DynamicHeapOrigin {
                            base_user_sgpr,
                            stride: 1,
                            index_user_sgpr,
                            index_shift: 0,
                        }),
                        _ => None,
                    }
                } else {
                    None
                };
                if is_dynamic && dynamic_heap_from_soffset.is_none() {
                    taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*soffset));
                }

                if let Some(col) = collector.as_deref_mut() {
                    match base_origin {
                        SgprOrigin::UserTable { user_sgpr, .. } => {
                            col.direct_root_cbvs.push((inst_idx, user_sgpr));
                        }
                        SgprOrigin::TableDesc {
                            def_inst_idx: ptr_def,
                            user_sgpr,
                            table_byte_offset,
                            ..
                        } => {
                            col.def_parents.insert(inst_idx, ptr_def);
                            let key = CbvKey {
                                base_sbase: 0,
                                user_sgpr: Some(user_sgpr),
                                table_offset: Some(table_byte_offset as i32),
                                dynamic_heap: None,
                            };
                            col.cbv_ptr_loads.push(CbvPtrLoad {
                                inst_idx,
                                ptr_def,
                                cbv_key: key,
                                source: WgslBindingSource::Table {
                                    user_sgpr,
                                    table_byte_offset,
                                },
                            });
                        }
                        SgprOrigin::DynamicHeapDesc {
                            def_inst_idx: ptr_def,
                            origin: heap,
                        } => {
                            col.def_parents.insert(inst_idx, ptr_def);
                            let key = CbvKey {
                                base_sbase: 0,
                                user_sgpr: None,
                                table_offset: None,
                                dynamic_heap: Some(heap),
                            };
                            col.cbv_ptr_loads.push(CbvPtrLoad {
                                inst_idx,
                                ptr_def,
                                cbv_key: key,
                                source: WgslBindingSource::ResourceHeap {
                                    base_user_sgpr: heap.base_user_sgpr,
                                    stride: heap.stride,
                                    index_user_sgpr: heap.index_user_sgpr,
                                    index_shift: heap.index_shift,
                                },
                            });
                        }
                        SgprOrigin::RawTableDesc {
                            def_inst_idx: ptr_def,
                            ..
                        } => {
                            col.def_parents.insert(inst_idx, ptr_def);
                        }
                        _ => {}
                    }
                }

                // Clear destination origins before populating derived descriptor/pointer origins.
                for k in 0..count {
                    let reg = sdata.saturating_add(k);
                    if (reg as usize) < state.origins.len() {
                        state.origins[reg as usize] = SgprOrigin::Unknown;
                    }
                }

                // Track descriptor & nested-table-pointer loads (`s_load_dword` / `x2` / `x4` /
                // `x8` / `x16`), including SM 6.6 dynamic heap fetches and 2-level table chains.
                if sdata.saturating_add(count) <= 106 {
                    if let Some(s_off) = const_soffset {
                        let raw_off = offset.saturating_add(s_off);
                        match base_origin {
                            SgprOrigin::DynamicHeapPtr(heap)
                            | SgprOrigin::DynamicHeapDesc { origin: heap, .. } => {
                                for k in 0..count {
                                    let reg = sdata.saturating_add(k);
                                    state.origins[reg as usize] = SgprOrigin::DynamicHeapDesc {
                                        def_inst_idx: inst_idx,
                                        origin: heap,
                                    };
                                }
                            }
                            SgprOrigin::VbTablePtr { byte_offset } => {
                                let eff = byte_offset.saturating_add(raw_off);
                                for k in 0..count {
                                    let reg = sdata.saturating_add(k);
                                    let vb_slot = (eff.saturating_add(i32::from(k) * 4)).max(0)
                                        as u32
                                        / 16;
                                    state.origins[reg as usize] = SgprOrigin::VbDesc {
                                        def_inst_idx: inst_idx,
                                        vb_slot,
                                    };
                                }
                            }
                            SgprOrigin::UserTable {
                                user_sgpr,
                                byte_offset,
                            } => {
                                let eff = byte_offset.saturating_add(raw_off);
                                for k in 0..count {
                                    let reg = sdata.saturating_add(k);
                                    let byte_off =
                                        (eff.saturating_add(i32::from(k) * 4)).max(0) as u32;
                                    state.origins[reg as usize] = SgprOrigin::TableDesc {
                                        def_inst_idx: inst_idx,
                                        user_sgpr,
                                        table_byte_offset: byte_off,
                                        addr_offset: 0,
                                    };
                                }
                            }
                            SgprOrigin::TableDesc {
                                user_sgpr,
                                table_byte_offset,
                                addr_offset,
                                ..
                            } => {
                                // 2-level nested descriptor-table load: accumulate outer table
                                // offset + pointer SALU offset + inner SMEM offset.
                                let eff = (table_byte_offset as i32)
                                    .saturating_add(addr_offset)
                                    .saturating_add(raw_off);
                                for k in 0..count {
                                    let reg = sdata.saturating_add(k);
                                    let byte_off =
                                        (eff.saturating_add(i32::from(k) * 4)).max(0) as u32;
                                    state.origins[reg as usize] = SgprOrigin::TableDesc {
                                        def_inst_idx: inst_idx,
                                        user_sgpr,
                                        table_byte_offset: byte_off,
                                        addr_offset: 0,
                                    };
                                }
                            }
                            _ => {
                                if matches!(*op, 0x02 | 0x03 | 0x04) && (*sdata % 2 == 0) {
                                    for k in 0..count {
                                        let reg = sdata.saturating_add(k);
                                        state.origins[reg as usize] = SgprOrigin::RawTableDesc {
                                            def_inst_idx: inst_idx,
                                            table_sbase: *sbase,
                                            table_offset: raw_off
                                                .saturating_add(i32::from(k) * 4),
                                            is_dynamic: false,
                                        };
                                    }
                                }
                            }
                        }
                    } else if let Some(heap) = dynamic_heap_from_soffset {
                        for k in 0..count {
                            let reg = sdata.saturating_add(k);
                            state.origins[reg as usize] = SgprOrigin::DynamicHeapDesc {
                                def_inst_idx: inst_idx,
                                origin: heap,
                            };
                        }
                    } else if matches!(*op, 0x02 | 0x03 | 0x04) && (*sdata % 2 == 0) {
                        for k in 0..count {
                            let reg = sdata.saturating_add(k);
                            state.origins[reg as usize] = SgprOrigin::RawTableDesc {
                                def_inst_idx: inst_idx,
                                table_sbase: *sbase,
                                table_offset: 0,
                                is_dynamic: true,
                            };
                        }
                    }
                }
            }
        }
        Rdna2Detail::Vop1 {
            op,
            vdst,
            src0,
            literal,
        } => {
            if !matches!(op, 0x00 | 0x1B | 0x41) {
                match *op {
                    0x01 => {
                        // v_mov_b32: preserve uniform SGPR/VGPR origin (e.g. uniform SGPR spill).
                        if *src0 <= 105 {
                            clear_vgpr_origins(state, *vdst);
                            state.vgpr_uniform_origins[*vdst as usize] =
                                state.origins[*src0 as usize];
                        } else if (256..=511).contains(src0) {
                            copy_vgpr_origins(state, *vdst, (*src0 - 256) as u8);
                        } else if let Some(v) = decode_inline_u32_src(*src0, *literal) {
                            clear_vgpr_origins(state, *vdst);
                            state.vgpr_uniform_origins[*vdst as usize] =
                                SgprOrigin::ConstImmediate {
                                    def_inst_idx: inst_idx,
                                    value: Some(v as u32),
                                };
                        } else if let Some(bits) = decode_inline_float_u32(*src0) {
                            clear_vgpr_origins(state, *vdst);
                            state.vgpr_uniform_origins[*vdst as usize] =
                                SgprOrigin::ConstImmediate {
                                    def_inst_idx: inst_idx,
                                    value: Some(bits),
                                };
                        } else {
                            clear_vgpr_origins(state, *vdst);
                        }
                    }
                    0x02 => {
                        // v_readfirstlane_b32 sdst, src0: restore lane 0 or uniform VGPR origin.
                        let sd = *vdst & 0x7F;
                        if sd <= 105 {
                            let restored = if (256..=511).contains(src0) {
                                let vsrc = (*src0 - 256) as u8;
                                normalize_read_origin(
                                    state
                                        .lane_origins
                                        .get(&(vsrc, 0))
                                        .copied()
                                        .unwrap_or(state.vgpr_uniform_origins[vsrc as usize]),
                                )
                            } else if *src0 <= 105 {
                                normalize_read_origin(state.origins[*src0 as usize])
                            } else {
                                SgprOrigin::Unknown
                            };
                            state.origins[sd as usize] = restored;
                        }
                    }
                    0x65 => {
                        // v_swap_b32 vdst, vsrc0: preserve & swap both uniform and per-lane origins.
                        if (256..=511).contains(src0) {
                            swap_vgpr_origins(state, *vdst, (*src0 - 256) as u8);
                        } else {
                            clear_vgpr_origins(state, *vdst);
                        }
                    }
                    _ => {
                        taint_sgpr_if_loaded(&mut collector, &state.origins, *src0);
                        clear_vgpr_origins(state, *vdst);
                    }
                }
            }
        }
        Rdna2Detail::Vopc { src0, .. } => {
            taint_sgpr_if_loaded(&mut collector, &state.origins, *src0);
        }
        Rdna2Detail::Vop2 { vdst, src0, .. } => {
            taint_sgpr_if_loaded(&mut collector, &state.origins, *src0);
            clear_vgpr_origins(state, *vdst);
        }
        Rdna2Detail::Vop3 {
            op,
            vdst,
            sdst,
            src0,
            src1,
            src2,
            literal,
            ..
        } => {
            let is_interp_f16 = vop3_is_interp_f16(*op);
            let is_spill_or_move = matches!(*op, 0x181 | 0x182 | 0x1E5 | 0x360 | 0x361);
            if !is_interp_f16 && !is_spill_or_move {
                taint_sgpr_if_loaded(&mut collector, &state.origins, *src0);
            }
            let is_unary = (0x180..=0x1FF).contains(op);
            if !is_unary && !is_spill_or_move {
                taint_sgpr_if_loaded(&mut collector, &state.origins, *src1);
            }
            if matches!(*op, 0x2FF..=0x301) && *src1 <= 104 {
                taint_sgpr_if_loaded(&mut collector, &state.origins, *src1 + 1);
            }
            if *op == 0x364 {
                if *src0 <= 104 {
                    taint_sgpr_if_loaded(&mut collector, &state.origins, *src0 + 1);
                }
                if *src1 <= 104 {
                    taint_sgpr_if_loaded(&mut collector, &state.origins, *src1 + 1);
                }
            }
            let is_ternary = *op != 0x342
                && matches!(
                    *op,
                    0x100
                        | 0x101
                        | 0x128..=0x12A
                        | 0x140..=0x165
                        | 0x16D..=0x171
                        | 0x178
                        | 0x340..=0x35F
                        | 0x36D..=0x372
                        | 0x377
                        | 0x378
                );
            if is_ternary {
                taint_sgpr_if_loaded(&mut collector, &state.origins, *src2);
            }
            if let Some(sd) = sdst {
                let sd7 = *sd & 0x7F;
                if sd7 <= 105 {
                    state.origins[sd7 as usize] = SgprOrigin::Unknown;
                    if sd7 <= 104 {
                        state.origins[(sd7 + 1) as usize] = SgprOrigin::Unknown;
                    }
                }
            }
            if *op <= 0xFF {
                let sd = *vdst & 0x7F;
                if sd <= 105 {
                    state.origins[sd as usize] = SgprOrigin::Unknown;
                    if sd <= 104 {
                        state.origins[(sd + 1) as usize] = SgprOrigin::Unknown;
                    }
                }
            } else if *op == 0x181 {
                // v_mov_b32_e64
                if *src0 <= 105 {
                    clear_vgpr_origins(state, *vdst);
                    state.vgpr_uniform_origins[*vdst as usize] = state.origins[*src0 as usize];
                } else if (256..=511).contains(src0) {
                    copy_vgpr_origins(state, *vdst, (*src0 - 256) as u8);
                } else if let Some(v) = decode_inline_u32_src(*src0, *literal) {
                    clear_vgpr_origins(state, *vdst);
                    state.vgpr_uniform_origins[*vdst as usize] = SgprOrigin::ConstImmediate {
                        def_inst_idx: inst_idx,
                        value: Some(v as u32),
                    };
                } else if let Some(bits) = decode_inline_float_u32(*src0) {
                    clear_vgpr_origins(state, *vdst);
                    state.vgpr_uniform_origins[*vdst as usize] = SgprOrigin::ConstImmediate {
                        def_inst_idx: inst_idx,
                        value: Some(bits),
                    };
                } else {
                    clear_vgpr_origins(state, *vdst);
                }
            } else if *op == 0x182 {
                // v_readfirstlane_b32_e64
                let sd = *vdst & 0x7F;
                if sd <= 105 {
                    let restored = if (256..=511).contains(src0) {
                        let vsrc = (*src0 - 256) as u8;
                        normalize_read_origin(
                            state
                                .lane_origins
                                .get(&(vsrc, 0))
                                .copied()
                                .unwrap_or(state.vgpr_uniform_origins[vsrc as usize]),
                        )
                    } else if *src0 <= 105 {
                        normalize_read_origin(state.origins[*src0 as usize])
                    } else {
                        SgprOrigin::Unknown
                    };
                    state.origins[sd as usize] = restored;
                }
            } else if *op == 0x1E5 {
                // v_swap_b32_e64
                if (256..=511).contains(src0) {
                    swap_vgpr_origins(state, *vdst, (*src0 - 256) as u8);
                } else {
                    clear_vgpr_origins(state, *vdst);
                }
            } else if *op == 0x361 {
                // v_writelane_b32 vdst, ssrc0, lane: preserve spilled SGPR origin per (vgpr, lane)
                if let Some(lane) = resolve_const_lane_index(*src1, *literal, &state.origins) {
                    let val_origin = if *src0 <= 105 {
                        state.origins[*src0 as usize]
                    } else if let Some(v) = decode_inline_u32_src(*src0, *literal) {
                        SgprOrigin::ConstImmediate {
                            def_inst_idx: inst_idx,
                            value: Some(v as u32),
                        }
                    } else if let Some(bits) = decode_inline_float_u32(*src0) {
                        SgprOrigin::ConstImmediate {
                            def_inst_idx: inst_idx,
                            value: Some(bits),
                        }
                    } else if *src0 >= 128 && *src0 < 256 {
                        SgprOrigin::ConstImmediate {
                            def_inst_idx: inst_idx,
                            value: None,
                        }
                    } else {
                        SgprOrigin::Unknown
                    };
                    state.vgpr_uniform_origins[*vdst as usize] = SgprOrigin::Unknown;
                    state.lane_origins.insert((*vdst, lane), val_origin);
                } else {
                    clear_vgpr_origins(state, *vdst);
                }
            } else if *op == 0x360 {
                // v_readlane_b32 sdst, vsrc0, lane: restore spilled SGPR origin from (vsrc0, lane)
                // or from uniform VGPR spill.
                let sd = *vdst & 0x7F;
                if sd <= 105 {
                    let restored = if (256..=511).contains(src0) {
                        let vsrc = (*src0 - 256) as u8;
                        let raw = if let Some(lane) =
                            resolve_const_lane_index(*src1, *literal, &state.origins)
                        {
                            state
                                .lane_origins
                                .get(&(vsrc, lane))
                                .copied()
                                .unwrap_or(state.vgpr_uniform_origins[vsrc as usize])
                        } else {
                            state.vgpr_uniform_origins[vsrc as usize]
                        };
                        normalize_read_origin(raw)
                    } else {
                        SgprOrigin::Unknown
                    };
                    state.origins[sd as usize] = restored;
                }
            } else {
                clear_vgpr_origins(state, *vdst);
                if (matches!(*op, 0x2FF..=0x301) || *op == 0x364) && *vdst < 255 {
                    clear_vgpr_origins(state, *vdst + 1);
                }
            }
        }
        Rdna2Detail::Vop3P {
            op,
            vdst,
            src0,
            src1,
            src2,
            ..
        } => {
            taint_sgpr_if_loaded(&mut collector, &state.origins, *src0);
            taint_sgpr_if_loaded(&mut collector, &state.origins, *src1);
            if vop3p_is_ternary(*op) {
                taint_sgpr_if_loaded(&mut collector, &state.origins, *src2);
            }
            clear_vgpr_origins(state, *vdst);
        }
        Rdna2Detail::Mimg {
            op,
            srsrc,
            ssamp,
            vdata,
            dmask,
            ..
        } => {
            if let Some(layout) = mimg_addr_layout(*op) {
                if !layout.is_store {
                    let eff_mask = if *dmask == 0 { 0xF } else { *dmask };
                    let out_cnt = eff_mask.count_ones() as u8;
                    for d in 0..out_cnt {
                        clear_vgpr_origins(state, vdata.saturating_add(d));
                    }
                }
                if let Some(col) = collector.as_deref_mut() {
                    let rsrc_sgpr = srsrc.saturating_mul(4);
                    if (rsrc_sgpr as usize) < state.origins.len() {
                        match state.origins[rsrc_sgpr as usize] {
                            SgprOrigin::TableDesc {
                                def_inst_idx,
                                user_sgpr,
                                table_byte_offset,
                                ..
                            } => {
                                col.direct_descriptor_defs.insert(def_inst_idx);
                                let key = ResourceKey::Table {
                                    user_sgpr,
                                    table_byte_offset,
                                };
                                if layout.is_store {
                                    col.out.inst_storage_tex_keys.insert(inst_idx, key);
                                } else if !layout.is_get_lod {
                                    col.out.inst_tex_keys.insert(inst_idx, key);
                                }
                            }
                            SgprOrigin::DynamicHeapDesc {
                                def_inst_idx,
                                origin,
                            } => {
                                col.direct_descriptor_defs.insert(def_inst_idx);
                                let key = ResourceKey::ResourceHeap {
                                    base_user_sgpr: origin.base_user_sgpr,
                                    stride: origin.stride,
                                    index_user_sgpr: origin.index_user_sgpr,
                                    index_shift: origin.index_shift,
                                };
                                if layout.is_store {
                                    col.out.inst_storage_tex_keys.insert(inst_idx, key);
                                } else if !layout.is_get_lod {
                                    col.out.inst_tex_keys.insert(inst_idx, key);
                                }
                            }
                            SgprOrigin::RawTableDesc {
                                def_inst_idx,
                                is_dynamic: true,
                                ..
                            } => {
                                col.raw_descriptor_defs.insert(def_inst_idx);
                            }
                            _ => {}
                        }
                    }
                    if !layout.is_load && !layout.is_store {
                        let samp_sgpr = ssamp.saturating_mul(4);
                        if (samp_sgpr as usize) < state.origins.len() {
                            match state.origins[samp_sgpr as usize] {
                                SgprOrigin::TableDesc {
                                    def_inst_idx,
                                    user_sgpr,
                                    table_byte_offset,
                                    ..
                                } => {
                                    col.direct_descriptor_defs.insert(def_inst_idx);
                                    if !layout.is_get_lod {
                                        col.out.inst_samp_keys.insert(
                                            inst_idx,
                                            ResourceKey::Table {
                                                user_sgpr,
                                                table_byte_offset,
                                            },
                                        );
                                    }
                                }
                                SgprOrigin::DynamicHeapDesc {
                                    def_inst_idx,
                                    origin,
                                } => {
                                    col.direct_descriptor_defs.insert(def_inst_idx);
                                    if !layout.is_get_lod {
                                        col.out.inst_samp_keys.insert(
                                            inst_idx,
                                            ResourceKey::SamplerHeap {
                                                base_user_sgpr: origin.base_user_sgpr,
                                                stride: origin.stride,
                                                index_user_sgpr: origin.index_user_sgpr,
                                                index_shift: origin.index_shift,
                                            },
                                        );
                                    }
                                }
                                SgprOrigin::ConstImmediate { def_inst_idx, .. } => {
                                    if !layout.is_get_lod {
                                        let key = if let Some(sdwords) =
                                            resolve_inline_sampler_sdwords(
                                                &state.origins,
                                                samp_sgpr,
                                            )
                                        {
                                            ResourceKey::InlineSampler(sdwords)
                                        } else {
                                            let idx = *col
                                                .static_sampler_by_def
                                                .entry(def_inst_idx)
                                                .or_insert_with(|| {
                                                    let i = col.next_static_sampler_index;
                                                    col.next_static_sampler_index += 1;
                                                    i
                                                });
                                            ResourceKey::StaticSampler(idx)
                                        };
                                        col.out.inst_samp_keys.insert(inst_idx, key);
                                    }
                                }
                                SgprOrigin::RawTableDesc {
                                    def_inst_idx,
                                    is_dynamic: true,
                                    ..
                                } => {
                                    col.raw_descriptor_defs.insert(def_inst_idx);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        Rdna2Detail::Mubuf {
            op,
            idxen,
            srsrc,
            soffset,
            vdata,
            ..
        } => {
            taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*soffset));
            if let Some(buf_op) = mubuf_storage_op(*op)
                && !buf_op.is_store()
            {
                for d in 0..buf_op.vdata_count() {
                    clear_vgpr_origins(state, vdata.saturating_add(d));
                }
            }
            if let Some(col) = collector.as_deref_mut() {
                let rsrc_sgpr = srsrc.saturating_mul(4);
                let origin = resolve_vsharp_origin(&state.origins, rsrc_sgpr);
                match origin {
                    SgprOrigin::VbDesc {
                        def_inst_idx,
                        vb_slot,
                    } if stage == ShaderStage::Vertex && mubuf_load_components(*op).is_some() => {
                        col.direct_descriptor_defs.insert(def_inst_idx);
                        col.out.vertex_fetch_vb_slots.insert(inst_idx, vb_slot);
                    }
                    SgprOrigin::TableDesc {
                        def_inst_idx,
                        user_sgpr,
                        table_byte_offset,
                        ..
                    } => {
                        col.direct_descriptor_defs.insert(def_inst_idx);
                        col.out.inst_storage_buf_keys.insert(
                            inst_idx,
                            ResourceKey::Table {
                                user_sgpr,
                                table_byte_offset,
                            },
                        );
                        if stage == ShaderStage::Vertex && mubuf_storage_op(*op).is_some() {
                            col.out.vertex_storage_buf_insts.insert(inst_idx);
                        }
                    }
                    SgprOrigin::DynamicHeapDesc {
                        def_inst_idx,
                        origin,
                    } => {
                        col.direct_descriptor_defs.insert(def_inst_idx);
                        col.out.inst_storage_buf_keys.insert(
                            inst_idx,
                            ResourceKey::ResourceHeap {
                                base_user_sgpr: origin.base_user_sgpr,
                                stride: origin.stride,
                                index_user_sgpr: origin.index_user_sgpr,
                                index_shift: origin.index_shift,
                            },
                        );
                        if stage == ShaderStage::Vertex && mubuf_storage_op(*op).is_some() {
                            col.out.vertex_storage_buf_insts.insert(inst_idx);
                        }
                    }
                    SgprOrigin::UserTable { user_sgpr, .. }
                        if (stage != ShaderStage::Vertex || !*idxen)
                            && is_synthesized_root_vsharp(&state.origins, rsrc_sgpr) =>
                    {
                        col.out
                            .inst_storage_buf_keys
                            .insert(inst_idx, ResourceKey::RootDescriptor { user_sgpr });
                        if stage == ShaderStage::Vertex && mubuf_storage_op(*op).is_some() {
                            col.out.vertex_storage_buf_insts.insert(inst_idx);
                        }
                    }
                    SgprOrigin::RawTableDesc { def_inst_idx, .. } => {
                        col.raw_descriptor_defs.insert(def_inst_idx);
                    }
                    _ => {}
                }
            }
        }
        Rdna2Detail::Mtbuf {
            op,
            idxen,
            srsrc,
            soffset,
            vdata,
            ..
        } => {
            taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*soffset));
            if let Some(buf_op) = mtbuf_storage_op(*op)
                && !buf_op.is_store()
            {
                for d in 0..buf_op.vdata_count() {
                    clear_vgpr_origins(state, vdata.saturating_add(d));
                }
            }
            if let Some(col) = collector.as_deref_mut() {
                let rsrc_sgpr = srsrc.saturating_mul(4);
                let origin = resolve_vsharp_origin(&state.origins, rsrc_sgpr);
                match origin {
                    SgprOrigin::VbDesc {
                        def_inst_idx,
                        vb_slot,
                    } if stage == ShaderStage::Vertex && mtbuf_load_components(*op).is_some() => {
                        col.direct_descriptor_defs.insert(def_inst_idx);
                        col.out.vertex_fetch_vb_slots.insert(inst_idx, vb_slot);
                    }
                    SgprOrigin::TableDesc {
                        def_inst_idx,
                        user_sgpr,
                        table_byte_offset,
                        ..
                    } => {
                        col.direct_descriptor_defs.insert(def_inst_idx);
                        col.out.inst_storage_buf_keys.insert(
                            inst_idx,
                            ResourceKey::Table {
                                user_sgpr,
                                table_byte_offset,
                            },
                        );
                        if stage == ShaderStage::Vertex && mtbuf_storage_op(*op).is_some() {
                            col.out.vertex_storage_buf_insts.insert(inst_idx);
                        }
                    }
                    SgprOrigin::DynamicHeapDesc {
                        def_inst_idx,
                        origin,
                    } => {
                        col.direct_descriptor_defs.insert(def_inst_idx);
                        col.out.inst_storage_buf_keys.insert(
                            inst_idx,
                            ResourceKey::ResourceHeap {
                                base_user_sgpr: origin.base_user_sgpr,
                                stride: origin.stride,
                                index_user_sgpr: origin.index_user_sgpr,
                                index_shift: origin.index_shift,
                            },
                        );
                        if stage == ShaderStage::Vertex && mtbuf_storage_op(*op).is_some() {
                            col.out.vertex_storage_buf_insts.insert(inst_idx);
                        }
                    }
                    SgprOrigin::UserTable { user_sgpr, .. }
                        if (stage != ShaderStage::Vertex || !*idxen)
                            && is_synthesized_root_vsharp(&state.origins, rsrc_sgpr) =>
                    {
                        col.out
                            .inst_storage_buf_keys
                            .insert(inst_idx, ResourceKey::RootDescriptor { user_sgpr });
                        if stage == ShaderStage::Vertex && mtbuf_storage_op(*op).is_some() {
                            col.out.vertex_storage_buf_insts.insert(inst_idx);
                        }
                    }
                    SgprOrigin::RawTableDesc { def_inst_idx, .. } => {
                        col.raw_descriptor_defs.insert(def_inst_idx);
                    }
                    _ => {}
                }
            }
        }
        Rdna2Detail::Flat { saddr, vdst, .. } => {
            taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*saddr));
            if *saddr <= 104 {
                taint_sgpr_if_loaded(&mut collector, &state.origins, u16::from(*saddr + 1));
            }
            clear_vgpr_origins(state, *vdst);
        }
        Rdna2Detail::Vintrp { vdst, .. } => {
            clear_vgpr_origins(state, *vdst);
        }
        Rdna2Detail::Ds { op, vdst, .. } => {
            if let Some(DsLdsOp::ReadContiguous(n)) = ds_lds_op(*op) {
                for k in 0..n {
                    clear_vgpr_origins(state, vdst.saturating_add(k));
                }
            } else if let Some(DsLdsOp::Read2 { .. }) = ds_lds_op(*op) {
                clear_vgpr_origins(state, *vdst);
                clear_vgpr_origins(state, vdst.saturating_add(1));
            }
        }
        Rdna2Detail::Sopp { .. } | Rdna2Detail::Exp { .. } | Rdna2Detail::Unknown { .. } => {}
    }
}

fn mark_untainted_def_and_parents(
    start_def: usize,
    force_start: bool,
    def_parents: &BTreeMap<usize, usize>,
    def_aliases: &BTreeMap<usize, BTreeSet<usize>>,
    tainted_defs: &BTreeSet<usize>,
    vsharp_fetch_insts: &mut BTreeSet<usize>,
) {
    let mut visited: BTreeSet<usize> = BTreeSet::new();
    let mut stack: Vec<(usize, bool)> = vec![(start_def, force_start)];
    while let Some((cur, forced)) = stack.pop() {
        if visited.len() >= 32 || !visited.insert(cur) {
            continue;
        }
        if forced || !tainted_defs.contains(&cur) {
            vsharp_fetch_insts.insert(cur);
        }
        if let Some(aliases) = def_aliases.get(&cur) {
            for &alias in aliases {
                if !visited.contains(&alias) {
                    stack.push((alias, forced));
                }
            }
        }
        if let Some(&parent) = def_parents.get(&cur)
            && !visited.contains(&parent)
        {
            stack.push((parent, false));
        }
    }
}

/// Detect descriptor-table V#/T#/S# indirections (`s_load_dword*`) and track console user-data
/// SGPR provenance (`s8+` graphics, `s4+` compute) across CFG basic blocks, SALU address
/// derivation, 2-level nested table pointers, VGPR lane/uniform spills, hardware VB-table
/// (`s[0:1]`), SM 6.6 dynamic descriptor heaps, and inline static-sampler S# origins
/// for `<pack>.reflect.json` (S31 / #662, S34 / #733, S40 / #890, S41 / #917).
fn analyze_vsharp_indirections(
    active: &[Rdna2Instruction],
    stage: ShaderStage,
) -> DescriptorAnalysis {
    if active.is_empty() {
        return DescriptorAnalysis::default();
    }

    let cfg = ControlFlowGraph::from_instructions(active);
    let num_blocks = cfg.blocks.len();

    // Pass 1: Monotonic forward dataflow fixpoint over CFG basic blocks so clobbers on one
    // branch (e.g. before `s_branch` / early `s_endpgm`) never pollute sibling blocks.
    // Use an ordered `BTreeSet` worklist (lowest block ID first = near-RPO, deduplicated) so
    // chained `if/else` diamonds converge linearly without burning the per-block visit budget.
    let mut block_in: Vec<Option<OriginState>> = vec![None; num_blocks];
    let mut visits: Vec<u8> = vec![0; num_blocks];
    let mut worklist: BTreeSet<BlockId> = BTreeSet::new();
    let mut pass1_def_aliases: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let entry_state = OriginState::initial(stage);
    let Some(entry) = cfg.entry() else {
        return DescriptorAnalysis::default();
    };
    block_in[entry] = Some(entry_state.clone());
    worklist.insert(entry);

    while let Some(bid) = worklist.pop_first() {
        if visits[bid] >= 32 {
            continue;
        }
        visits[bid] += 1;
        let block = &cfg.blocks[bid];
        if block.synthetic_exit {
            continue;
        }
        let Some(mut state) = block_in[bid].clone() else {
            continue;
        };
        for inst_idx in block.instr_range.clone() {
            step_origin_state(&mut state, inst_idx, &active[inst_idx], stage, None);
        }
        for succ in block.terminator.successors() {
            if cfg.blocks[succ].synthetic_exit {
                continue;
            }
            match &mut block_in[succ] {
                None => {
                    block_in[succ] = Some(state.clone());
                    worklist.insert(succ);
                }
                Some(existing) => {
                    if existing.join_from(&state, &mut pass1_def_aliases) {
                        worklist.insert(succ);
                    }
                }
            }
        }
    }

    // Pass 2: Single pass over every non-synthetic block using converged incoming origins to
    // record descriptor uses, tainting, nested pointer chains, and CBV sources.
    let mut collector = AnalysisCollector {
        def_aliases: pass1_def_aliases,
        ..AnalysisCollector::default()
    };
    for (bid, block) in cfg.blocks.iter().enumerate() {
        if block.synthetic_exit || block.instr_range.is_empty() {
            continue;
        }
        let mut state = block_in[bid].clone().unwrap_or_else(|| entry_state.clone());
        for inst_idx in block.instr_range.clone() {
            step_origin_state(
                &mut state,
                inst_idx,
                &active[inst_idx],
                stage,
                Some(&mut collector),
            );
        }
    }

    // Finalize descriptor fetch suppression & CBV source attribution.
    let AnalysisCollector {
        mut out,
        tainted_defs,
        def_parents,
        def_aliases,
        direct_descriptor_defs,
        raw_descriptor_defs,
        buffer_load_uses,
        cbv_ptr_loads,
        direct_root_cbvs,
        ..
    } = collector;

    for def_idx in direct_descriptor_defs {
        mark_untainted_def_and_parents(
            def_idx,
            true,
            &def_parents,
            &def_aliases,
            &tainted_defs,
            &mut out.vsharp_fetch_insts,
        );
    }

    for def_idx in raw_descriptor_defs {
        if !tainted_defs.contains(&def_idx) {
            mark_untainted_def_and_parents(
                def_idx,
                false,
                &def_parents,
                &def_aliases,
                &tainted_defs,
                &mut out.vsharp_fetch_insts,
            );
        }
    }

    for buf_use in buffer_load_uses {
        if buf_use.source.is_some() {
            // `s_buffer_load_*` consumes a 128-bit hardware V# descriptor from a known
            // descriptor table or dynamic heap. Always suppress its V# load(s) and attribute
            // the CBV read even if a V# word (e.g. `num_records` / stride) or a copied SGPR
            // pair was also inspected by scalar/vector ALU or comparisons (S41 / #917).
            mark_untainted_def_and_parents(
                buf_use.def_inst_idx,
                true,
                &def_parents,
                &def_aliases,
                &tainted_defs,
                &mut out.vsharp_fetch_insts,
            );
            if let Some(key) = buf_use.cbv_key {
                out.indirect_smem_cbv_keys.insert(buf_use.inst_idx, key);
                if let Some(src) = buf_use.source {
                    out.cbv_sources.entry(key).or_insert(src);
                }
            }
        } else if !tainted_defs.contains(&buf_use.def_inst_idx)
            || out.vsharp_fetch_insts.contains(&buf_use.def_inst_idx)
        {
            mark_untainted_def_and_parents(
                buf_use.def_inst_idx,
                false,
                &def_parents,
                &def_aliases,
                &tainted_defs,
                &mut out.vsharp_fetch_insts,
            );
            if let Some(key) = buf_use.cbv_key {
                out.indirect_smem_cbv_keys.insert(buf_use.inst_idx, key);
            }
        }
    }

    // 2-level CBV pointer loads (`s_load_dwordx2` from table -> `s_load_dword*` reading CBV data):
    // any inner load that was NOT itself a descriptor fetch reads CBV data from its pointer's table slot.
    for ptr_load in cbv_ptr_loads {
        if !out.vsharp_fetch_insts.contains(&ptr_load.inst_idx) {
            if !tainted_defs.contains(&ptr_load.ptr_def) {
                mark_untainted_def_and_parents(
                    ptr_load.ptr_def,
                    false,
                    &def_parents,
                    &def_aliases,
                    &tainted_defs,
                    &mut out.vsharp_fetch_insts,
                );
            }
            out.indirect_smem_cbv_keys
                .insert(ptr_load.inst_idx, ptr_load.cbv_key);
            out.cbv_sources
                .entry(ptr_load.cbv_key)
                .or_insert(ptr_load.source);
        }
    }

    // Direct root-descriptor CBV loads (`s_load_dword*` / `s_buffer_load_*` on a `UserTable` SGPR)
    // that were not suppressed as descriptor/pointer fetches.
    for (inst_idx, user_sgpr) in direct_root_cbvs {
        if !out.vsharp_fetch_insts.contains(&inst_idx)
            && !out.indirect_smem_cbv_keys.contains_key(&inst_idx)
        {
            let key = CbvKey {
                base_sbase: 0,
                user_sgpr: Some(user_sgpr),
                table_offset: None,
                dynamic_heap: None,
            };
            out.indirect_smem_cbv_keys.insert(inst_idx, key);
            out.cbv_sources
                .entry(key)
                .or_insert(WgslBindingSource::RootDescriptor { user_sgpr });
        }
    }

    out
}

/// Map a vertex-shader MUBUF/MTBUF attribute fetch to `(attribute_byte_offset, component_shift)`.
///
/// A 1-component 8-bit `tbuffer_load_format_x` (`BUF_FMT_8_*`, formats 1..=6) at a byte offset
/// that is not dword aligned reads a single channel of a packed 4x8-bit attribute (e.g.
/// `offset:7` on an `R8G8B8A8_UNORM` element at `+4` reads `.w`). Such fetches are folded into
/// the dword-aligned `vec4<f32>` attribute (matching WebGPU `unorm8x4`) with a component shift.
fn vertex_fetch_slot(is_mtbuf: bool, op: u8, format: u8, byte_offset: u16) -> (u16, u8) {
    let sub_dword = (byte_offset % 4) as u8;
    if is_mtbuf && op == 0x00 && (1..=6).contains(&format) && sub_dword != 0 {
        (byte_offset - u16::from(sub_dword), sub_dword)
    } else {
        (byte_offset, 0)
    }
}

/// Decode a GFX10.3 unified `MTBUF` `format` code into `(component_type, fetch_format)` for
/// stage reflection (`runtime/gpu/stage_reflection.cpp::vertex_format_from_name`).
fn mtbuf_vertex_format_info(format: u8, folded_4x8: bool) -> (&'static str, Option<&'static str>) {
    if folded_4x8 {
        return match format {
            1 | 3 => ("f32", Some("unorm8x4")),
            2 | 4 => ("f32", Some("snorm8x4")),
            5 => ("u32", Some("uint8x4")),
            6 => ("i32", Some("sint8x4")),
            _ => ("f32", Some("unorm8x4")),
        };
    }
    match format {
        1 | 3 => ("f32", Some("unorm8")),
        2 | 4 => ("f32", Some("snorm8")),
        5 => ("u32", Some("uint8")),
        6 => ("i32", Some("sint8")),
        7 | 9 => ("f32", Some("unorm16")),
        8 | 10 => ("f32", Some("snorm16")),
        11 => ("u32", Some("uint16")),
        12 => ("i32", Some("sint16")),
        13 => ("f32", Some("float16")),
        14 | 16 => ("f32", Some("unorm8x2")),
        15 | 17 => ("f32", Some("snorm8x2")),
        18 => ("u32", Some("uint8x2")),
        19 => ("i32", Some("sint8x2")),
        20 => ("u32", Some("uint32")),
        21 => ("i32", Some("sint32")),
        22 => ("f32", Some("float32")),
        23 | 25 => ("f32", Some("unorm16x2")),
        24 | 26 => ("f32", Some("snorm16x2")),
        27 => ("u32", Some("uint16x2")),
        28 => ("i32", Some("sint16x2")),
        29 => ("f32", Some("float16x2")),
        32 | 38 => ("f32", Some("unorm10-10-10-2")),
        44 | 46 | 56 => ("f32", Some("unorm8x4")),
        45 | 47 | 57 => ("f32", Some("snorm8x4")),
        48 | 60 => ("u32", Some("uint8x4")),
        49 | 61 => ("i32", Some("sint8x4")),
        50 | 64 => ("u32", Some("uint32x2")),
        51 | 65 => ("i32", Some("sint32x2")),
        52 | 66 => ("f32", Some("float32x2")),
        53 | 67 => ("f32", Some("unorm16x4")),
        54 | 68 => ("f32", Some("snorm16x4")),
        58 | 72 => ("f32", Some("float16x4")),
        73 => ("u32", Some("uint32x3")),
        74 => ("i32", Some("sint32x3")),
        75 => ("f32", Some("float32x3")),
        76 => ("u32", Some("uint32x4")),
        77 => ("i32", Some("sint32x4")),
        78 => ("f32", Some("float32x4")),
        _ => ("f32", None),
    }
}

fn plan_program(
    summary: &Rdna2ProgramSummary,
    instructions: &[Rdna2Instruction],
    options: &Rdna2TranslateOptions,
) -> ProgramPlan {
    let stage = resolve_stage(summary.inferred_stage, options.stage);
    let entry_point = match &options.entry_point {
        Some(custom) => sanitize_wgsl_ident(custom, default_entry_point(stage)),
        None => default_entry_point(stage).to_string(),
    };

    // Determine effective instruction slice (stop at first s_endpgm if no branch targets past it).
    let max_branch_target = summary
        .branches
        .iter()
        .map(|b| b.target_offset)
        .max()
        .unwrap_or(-1);
    let mut active_inst_count = instructions.len();
    for (idx, inst) in instructions.iter().enumerate() {
        if let Rdna2Detail::Sopp { op, .. } = inst.detail
            && matches!(op, 0x01 | 0x1B | 0x1E)
            && max_branch_target <= inst.offset as i64
        {
            active_inst_count = idx + 1;
            break;
        }
    }

    let active = &instructions[..active_inst_count];
    let desc_analysis = analyze_vsharp_indirections(active, stage);
    let vsharp_fetch_insts = desc_analysis.vsharp_fetch_insts;
    let indirect_smem_cbv_keys = desc_analysis.indirect_smem_cbv_keys;
    let vertex_storage_buf_insts = desc_analysis.vertex_storage_buf_insts;
    let vertex_fetch_vb_slots = desc_analysis.vertex_fetch_vb_slots;

    let mut cbv_usage: BTreeMap<CbvKey, CbvUsage> = BTreeMap::new();
    let mut tex_dims: BTreeMap<ResourceKey, TexDim> = BTreeMap::new();
    let mut samp_rsrc_set: BTreeSet<ResourceKey> = BTreeSet::new();
    let mut storage_buf_usage: BTreeMap<ResourceKey, bool> = BTreeMap::new();
    let mut storage_tex_dims: BTreeMap<ResourceKey, TexDim> = BTreeMap::new();
    let mut uses_lds = false;
    let mut has_buffer_fetch = false;
    let mut vertex_buf_fetches: BTreeMap<(u32, u16), VertexFetchPlan> = BTreeMap::new();
    let mut frag_attr_set: BTreeSet<u8> = BTreeSet::new();
    let mut vs_param_set: BTreeSet<u8> = BTreeSet::new();
    let mut ps_mrt_set: BTreeSet<u8> = BTreeSet::new();
    let mut ps_exports_depth = false;

    let mut vgprs: BTreeSet<u8> = BTreeSet::new();
    let mut sgprs: BTreeSet<u8> = BTreeSet::new();
    let mut lane_spill_slots: BTreeSet<(u8, u8)> = BTreeSet::new();
    let mut written_vgprs: BTreeSet<u8> = BTreeSet::new();
    let mut read_before_write_vgprs: BTreeSet<u8> = BTreeSet::new();
    let mut written_sgprs: BTreeSet<u8> = BTreeSet::new();
    let mut read_before_write_sgprs: BTreeSet<u8> = BTreeSet::new();
    let mut needs_vcc = false;
    let mut needs_scc = false;
    let mut needs_exec = false;

    // Record a source operand read (register usage, read-before-write, VCC/SCC needs).
    macro_rules! record_src {
        ($src:expr) => {
            record_src_usage(
                $src,
                &mut vgprs,
                &mut sgprs,
                &written_vgprs,
                &mut read_before_write_vgprs,
                &written_sgprs,
                &mut read_before_write_sgprs,
                &mut needs_vcc,
                &mut needs_scc,
            )
        };
    }
    // Record the high half of a 64-bit SGPR (`s[n:n+1]`) or VGPR (`v[n:n+1]`) pair source.
    macro_rules! record_src_pair_hi {
        ($src:expr) => {{
            let src: u16 = $src;
            if src <= 104 || (256..=510).contains(&src) {
                record_src!(src + 1);
            }
        }};
    }

    for (inst_idx, inst) in active.iter().enumerate() {
        if inst.truncated {
            continue;
        }
        match &inst.detail {
            Rdna2Detail::Sopc {
                op, ssrc0, ssrc1, ..
            } => {
                record_src_usage(
                    u16::from(*ssrc0),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    u16::from(*ssrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                let (pair0, pair1) = sopc_src_pair_widths(*op);
                if pair0 {
                    record_src_pair_hi!(u16::from(*ssrc0));
                }
                if pair1 {
                    record_src_pair_hi!(u16::from(*ssrc1));
                }
                needs_scc = true;
            }
            Rdna2Detail::Sop1 {
                op, sdst, ssrc0, ..
            } => {
                record_src_usage(
                    u16::from(*ssrc0),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if *sdst == 126 {
                    // `exec_lo` / `exec` destination (e.g. `s_mov_b64 exec, s[0:1]`).
                    needs_exec = true;
                }
                if let Some(write) = classify_exec_mask_op(inst) {
                    // `s_*_saveexec_*` / `s_andn{1,2}_wrexec_*`: read `exec`, write `exec`,
                    // `sdst` (an SGPR pair for `_b64`) and `scc`.
                    needs_exec = true;
                    needs_scc = true;
                    if write.wave64 {
                        record_src_pair_hi!(u16::from(*ssrc0));
                    }
                    if *sdst <= 105 {
                        sgprs.insert(*sdst);
                        written_sgprs.insert(*sdst);
                        if write.wave64 && *sdst <= 104 {
                            sgprs.insert(*sdst + 1);
                            written_sgprs.insert(*sdst + 1);
                        }
                    } else if *sdst == 106 {
                        needs_vcc = true;
                    }
                } else if matches!(*op, 0x04 | 0x08 | 0x0A) {
                    // s_mov_b64 / s_not_b64 / s_wqm_b64: 64-bit SGPR pair source and destination
                    if *op == 0x08 {
                        needs_scc = true;
                    }
                    if *ssrc0 <= 104 {
                        let hi = *ssrc0 + 1;
                        sgprs.insert(hi);
                        if !written_sgprs.contains(&hi) {
                            read_before_write_sgprs.insert(hi);
                        }
                    }
                    if *sdst <= 104 {
                        sgprs.insert(*sdst);
                        sgprs.insert(*sdst + 1);
                        written_sgprs.insert(*sdst);
                        written_sgprs.insert(*sdst + 1);
                    } else if *sdst == 105 {
                        sgprs.insert(105);
                        written_sgprs.insert(105);
                    } else if *sdst == 106 {
                        needs_vcc = true;
                    }
                } else {
                    if matches!(*op, 0x05 | 0x1B | 0x1D) {
                        if *op == 0x05 {
                            needs_scc = true;
                        }
                        if *sdst <= 105 {
                            sgprs.insert(*sdst);
                            if !written_sgprs.contains(sdst) {
                                read_before_write_sgprs.insert(*sdst);
                            }
                        }
                    }
                    if *sdst <= 105 {
                        sgprs.insert(*sdst);
                        written_sgprs.insert(*sdst);
                    }
                }
            }
            Rdna2Detail::Sopk { op, sdst, .. } => {
                if matches!(*op, 0x03..=0x0E) {
                    needs_scc = true;
                    record_src!(u16::from(*sdst));
                } else if *sdst <= 105 {
                    if matches!(*op, 0x0F | 0x10) && !written_sgprs.contains(sdst) {
                        read_before_write_sgprs.insert(*sdst);
                    }
                    sgprs.insert(*sdst);
                    written_sgprs.insert(*sdst);
                }
            }
            Rdna2Detail::Sop2 {
                op,
                sdst,
                ssrc0,
                ssrc1,
                ..
            } => {
                record_src_usage(
                    u16::from(*ssrc0),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    u16::from(*ssrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                let (pair0, pair1) = sop2_src_pair_widths(*op);
                if pair0 {
                    record_src_pair_hi!(u16::from(*ssrc0));
                }
                if pair1 {
                    record_src_pair_hi!(u16::from(*ssrc1));
                }
                if matches!(*op, 0x04 | 0x05 | 0x0A | 0x0B) {
                    // s_addc_u32 / s_subb_u32 / s_cselect_b32 / s_cselect_b64 read SCC
                    needs_scc = true;
                }
                if *sdst == 126 {
                    // `exec_lo` / `exec` destination (e.g. `s_or_b32 exec_lo, exec_lo, s0`).
                    needs_exec = true;
                }
                if (pair0 || *op == 0x25) && *sdst <= 104 {
                    // 64-bit SGPR pair destination
                    sgprs.insert(*sdst);
                    sgprs.insert(*sdst + 1);
                    written_sgprs.insert(*sdst);
                    written_sgprs.insert(*sdst + 1);
                } else if *sdst <= 105 {
                    sgprs.insert(*sdst);
                    written_sgprs.insert(*sdst);
                } else if pair0 && *sdst == 106 {
                    needs_vcc = true;
                }
            }
            Rdna2Detail::Smem {
                op,
                sdata,
                sbase,
                soffset,
                offset,
                ..
            } => {
                if vsharp_fetch_insts.contains(&inst_idx) {
                    continue;
                }
                if let Some(count) = smem_load_dword_count(*op) {
                    let is_dynamic = *soffset != 0x7D && *soffset <= 105;
                    if is_dynamic {
                        sgprs.insert(*soffset);
                        if !written_sgprs.contains(soffset) {
                            read_before_write_sgprs.insert(*soffset);
                        }
                    }
                    let base_word = if *offset > 0 {
                        (*offset as u32) / 4
                    } else {
                        0
                    };
                    let loaded_dwords = if *sdata <= 105 {
                        u32::from(count).min(u32::from(106 - *sdata))
                    } else {
                        u32::from(count)
                    }
                    .max(1);
                    let highest_dword = base_word.saturating_add(loaded_dwords - 1);
                    let highest_vec4 = highest_dword / 4;
                    let cbv_key = indirect_smem_cbv_keys
                        .get(&inst_idx)
                        .copied()
                        .unwrap_or(CbvKey {
                            base_sbase: *sbase,
                            user_sgpr: None,
                            table_offset: None,
                            dynamic_heap: None,
                        });
                    cbv_usage
                        .entry(cbv_key)
                        .and_modify(|usage| {
                            usage.max_vec4_index = usage.max_vec4_index.max(highest_vec4);
                            usage.has_dynamic_offset |= is_dynamic;
                        })
                        .or_insert(CbvUsage {
                            max_vec4_index: highest_vec4,
                            has_dynamic_offset: is_dynamic,
                        });
                    for k in 0..count {
                        let reg = sdata.saturating_add(k);
                        if reg <= 105 {
                            sgprs.insert(reg);
                            written_sgprs.insert(reg);
                        }
                    }
                }
            }
            Rdna2Detail::Vop1 { op, vdst, src0, .. } => {
                if !matches!(op, 0x00 | 0x1B | 0x41) {
                    record_src_usage(
                        *src0,
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &written_sgprs,
                        &mut read_before_write_sgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                    if *op == 0x02 {
                        let sd = *vdst & 0x7F;
                        if sd <= 105 {
                            sgprs.insert(sd);
                            written_sgprs.insert(sd);
                        }
                    } else if *op == 0x65 {
                        // v_swap_b32 vdst, vsrc0: reads AND writes both VGPRs
                        record_src!(256 + u16::from(*vdst));
                        vgprs.insert(*vdst);
                        written_vgprs.insert(*vdst);
                        if (256..=511).contains(src0) {
                            let vsrc = (*src0 - 256) as u8;
                            vgprs.insert(vsrc);
                            written_vgprs.insert(vsrc);
                        }
                    } else {
                        vgprs.insert(*vdst);
                        written_vgprs.insert(*vdst);
                    }
                }
            }
            Rdna2Detail::Vopc {
                op, vsrc1, src0, ..
            } => {
                record_src_usage(
                    *src0,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    256 + u16::from(*vsrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if (*op & 0x10) != 0 {
                    needs_exec = true;
                } else {
                    needs_vcc = true;
                }
            }
            Rdna2Detail::Vop2 {
                op,
                vdst,
                vsrc1,
                src0,
                ..
            } => {
                record_src_usage(
                    *src0,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    256 + u16::from(*vsrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if matches!(op, 0x02 | 0x06 | 0x1F | 0x2B | 0x36 | 0x3C) {
                    // v_dot2c_f32_f16 / v_fmac* / v_mac_f32 / v_pk_fmac_f16 read vdst as accumulator
                    record_src_usage(
                        256 + u16::from(*vdst),
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &written_sgprs,
                        &mut read_before_write_sgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
                if matches!(op, 0x00 | 0x01) || vop2_is_co_ci(*op) {
                    // v_cndmask_b32 reads VCC; v_*_co_ci_u32 read carry-in from and write carry-out to VCC
                    needs_vcc = true;
                }
                vgprs.insert(*vdst);
                written_vgprs.insert(*vdst);
            }
            Rdna2Detail::Vop3 {
                op,
                vdst,
                sdst,
                src0,
                src1,
                src2,
                literal,
                ..
            } => {
                if vop3_is_interp_f16(*op) {
                    let (attr, _, _) = decode_vop3_interp_attr(*src0);
                    if stage == ShaderStage::Fragment {
                        frag_attr_set.insert(attr % MAX_WEBGPU_LOCATIONS);
                    }
                    vgprs.insert(*vdst);
                    written_vgprs.insert(*vdst);
                    continue;
                }
                record_src_usage(
                    *src0,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                let is_unary = (0x180..=0x1FF).contains(op);
                if !is_unary {
                    record_src_usage(
                        *src1,
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &written_sgprs,
                        &mut read_before_write_sgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
                let is_ternary = matches!(
                    *op,
                    0x100
                        | 0x101
                        | 0x128..=0x12A
                        | 0x140..=0x165
                        | 0x16D..=0x171
                        | 0x178
                        | 0x340..=0x35F
                        | 0x36D..=0x372
                        | 0x377
                        | 0x378
                );
                if is_ternary {
                    record_src_usage(
                        *src2,
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &written_sgprs,
                        &mut read_before_write_sgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
                if matches!(*op, 0x102 | 0x106 | 0x11F | 0x12B | 0x136 | 0x13C | 0x1E5) {
                    // VOP3-promoted accumulating VOP2 ops & v_swap_b32 read vdst
                    record_src!(256 + u16::from(*vdst));
                }
                let is_shift64 = matches!(*op, 0x2FF..=0x301);
                if is_shift64 {
                    // v_lshlrev_b64 / v_lshrrev_b64 / v_ashrrev_i64 read a 64-bit src1 pair
                    record_src_pair_hi!(*src1);
                }
                let is_pair_dst = is_shift64 || *op == 0x364;
                // Lane-mask SGPR destinations are written as canonical `(lo = lane bit, hi = 0)` pairs.
                let mut record_lane_mask_dst = |sd7: u8| {
                    if sd7 <= 105 {
                        sgprs.insert(sd7);
                        written_sgprs.insert(sd7);
                        if sd7 <= 104 {
                            sgprs.insert(sd7 + 1);
                            written_sgprs.insert(sd7 + 1);
                        }
                    } else if sd7 == 106 {
                        needs_vcc = true;
                    }
                };
                if let Some(sd) = sdst {
                    record_lane_mask_dst(*sd & 0x7F);
                }
                if *op <= 0xFF {
                    if ((*op as u8) & 0x10) != 0 {
                        needs_exec = true;
                    }
                    record_lane_mask_dst(*vdst & 0x7F);
                } else if *op == 0x360 {
                    // v_readlane_b32 sdst, vsrc0, ssrc1: writes 32-bit scalar destination sdst
                    let sd = *vdst & 0x7F;
                    if sd <= 105 {
                        sgprs.insert(sd);
                        written_sgprs.insert(sd);
                    } else if sd == 106 {
                        needs_vcc = true;
                    } else if sd == 126 {
                        needs_exec = true;
                    }
                } else if *op == 0x361 {
                    // v_writelane_b32 vdst, ssrc0, ssrc1: reserve per-(vgpr, lane) slot for const lane
                    if let Some(lane) = decode_const_lane_index(*src1, *literal) {
                        lane_spill_slots.insert((*vdst, lane));
                    }
                    vgprs.insert(*vdst);
                    written_vgprs.insert(*vdst);
                } else {
                    vgprs.insert(*vdst);
                    written_vgprs.insert(*vdst);
                    if *op == 0x1E5 && (256..=511).contains(src0) {
                        let vsrc = (*src0 - 256) as u8;
                        vgprs.insert(vsrc);
                        written_vgprs.insert(vsrc);
                    }
                    if is_pair_dst && *vdst < 255 {
                        vgprs.insert(*vdst + 1);
                        written_vgprs.insert(*vdst + 1);
                    }
                }
            }
            Rdna2Detail::Vop3P {
                op,
                vdst,
                src0,
                src1,
                src2,
                ..
            } => {
                record_src_usage(
                    *src0,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    *src1,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if vop3p_is_ternary(*op) {
                    record_src_usage(
                        *src2,
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &written_sgprs,
                        &mut read_before_write_sgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
                vgprs.insert(*vdst);
                written_vgprs.insert(*vdst);
            }
            Rdna2Detail::Vintrp { vdst, attr, .. } => {
                if stage == ShaderStage::Fragment {
                    frag_attr_set.insert(*attr % MAX_WEBGPU_LOCATIONS);
                }
                vgprs.insert(*vdst);
                written_vgprs.insert(*vdst);
            }
            Rdna2Detail::Mimg {
                op,
                dmask,
                dim,
                vaddr,
                vdata,
                srsrc,
                ssamp,
                ..
            } => {
                if let Some(layout) = mimg_addr_layout(*op) {
                    if layout.is_get_lod {
                        let eff_mask = if *dmask == 0 { 0xF } else { *dmask };
                        let out_cnt = eff_mask.count_ones() as u8;
                        for d in 0..out_cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                        continue;
                    }
                    let tex_dim = TexDim::from_mimg_dim(*dim);
                    let addr_vgprs = layout.total_vgprs.saturating_add(tex_dim.extra_coord_vgprs());
                    if layout.is_store {
                        if stage != ShaderStage::Vertex {
                            let key = desc_analysis
                                .inst_storage_tex_keys
                                .get(&inst_idx)
                                .copied()
                                .unwrap_or(ResourceKey::RawSlot(*srsrc));
                            storage_tex_dims
                                .entry(key)
                                .and_modify(|d| *d = d.merge(tex_dim))
                                .or_insert(tex_dim);
                            for a in 0..addr_vgprs {
                                let reg = vaddr.saturating_add(a);
                                vgprs.insert(reg);
                                if !written_vgprs.contains(&reg) {
                                    read_before_write_vgprs.insert(reg);
                                }
                            }
                            let eff_mask = if *dmask == 0 { 0xF } else { *dmask };
                            let in_cnt = eff_mask.count_ones() as u8;
                            for d in 0..in_cnt {
                                let reg = vdata.saturating_add(d);
                                vgprs.insert(reg);
                                if !written_vgprs.contains(&reg) {
                                    read_before_write_vgprs.insert(reg);
                                }
                            }
                        }
                    } else {
                        let key = desc_analysis
                            .inst_tex_keys
                            .get(&inst_idx)
                            .copied()
                            .unwrap_or(ResourceKey::RawSlot(*srsrc));
                        tex_dims
                            .entry(key)
                            .and_modify(|d| *d = d.merge(tex_dim))
                            .or_insert(tex_dim);
                        if !layout.is_load {
                            let skey = desc_analysis
                                .inst_samp_keys
                                .get(&inst_idx)
                                .copied()
                                .unwrap_or(ResourceKey::RawSlot(*ssamp));
                            samp_rsrc_set.insert(skey);
                        }
                        for a in 0..addr_vgprs {
                            let reg = vaddr.saturating_add(a);
                            vgprs.insert(reg);
                            if !written_vgprs.contains(&reg) {
                                read_before_write_vgprs.insert(reg);
                            }
                        }
                        let eff_mask = if *dmask == 0 { 0xF } else { *dmask };
                        let out_cnt = eff_mask.count_ones() as u8;
                        for d in 0..out_cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                    }
                }
            }
            Rdna2Detail::Mubuf {
                op,
                offset,
                offen,
                idxen,
                vaddr,
                vdata,
                srsrc,
                soffset,
                ..
            } => {
                if stage == ShaderStage::Vertex && !vertex_storage_buf_insts.contains(&inst_idx) {
                    if let Some(comps) = mubuf_load_components(*op) {
                        has_buffer_fetch = true;
                        let vb_slot = vertex_fetch_vb_slots.get(&inst_idx).copied().unwrap_or(0);
                        vertex_buf_fetches
                            .entry((vb_slot, *offset))
                            .and_modify(|p| p.components = p.components.max(comps))
                            .or_insert(VertexFetchPlan {
                                components: comps,
                                component_type: "f32",
                                fetch_format: None,
                            });
                        for d in 0..comps {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                    }
                } else if let Some(buf_op) = mubuf_storage_op(*op) {
                    let writable = buf_op.is_store() && stage != ShaderStage::Vertex;
                    let key = desc_analysis
                        .inst_storage_buf_keys
                        .get(&inst_idx)
                        .copied()
                        .unwrap_or(ResourceKey::RawSlot(*srsrc));
                    storage_buf_usage
                        .entry(key)
                        .and_modify(|w| *w |= writable)
                        .or_insert(writable);
                    if *offen || *idxen {
                        vgprs.insert(*vaddr);
                        if !written_vgprs.contains(vaddr) {
                            read_before_write_vgprs.insert(*vaddr);
                        }
                    }
                    if *offen && *idxen {
                        let v1 = vaddr.saturating_add(1);
                        vgprs.insert(v1);
                        if !written_vgprs.contains(&v1) {
                            read_before_write_vgprs.insert(v1);
                        }
                    }
                    if *soffset <= 105 {
                        sgprs.insert(*soffset);
                        if !written_sgprs.contains(soffset) {
                            read_before_write_sgprs.insert(*soffset);
                        }
                    }
                    let cnt = buf_op.vdata_count();
                    if buf_op.is_store() {
                        for d in 0..cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            if !written_vgprs.contains(&reg) {
                                read_before_write_vgprs.insert(reg);
                            }
                        }
                    } else {
                        for d in 0..cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                    }
                } else if mubuf_is_atomic(*op) && stage != ShaderStage::Vertex {
                    let key = desc_analysis
                        .inst_storage_buf_keys
                        .get(&inst_idx)
                        .copied()
                        .unwrap_or(ResourceKey::RawSlot(*srsrc));
                    storage_buf_usage.insert(key, true);
                }
            }
            Rdna2Detail::Mtbuf {
                op,
                format,
                offset,
                offen,
                idxen,
                vaddr,
                vdata,
                srsrc,
                soffset,
                ..
            } => {
                if stage == ShaderStage::Vertex && !vertex_storage_buf_insts.contains(&inst_idx) {
                    if let Some(comps) = mtbuf_load_components(*op) {
                        has_buffer_fetch = true;
                        let vb_slot = vertex_fetch_vb_slots.get(&inst_idx).copied().unwrap_or(0);
                        let (attr_offset, shift) = vertex_fetch_slot(true, *op, *format, *offset);
                        let attr_comps = if shift != 0 { 4 } else { comps };
                        let (comp_ty, fetch_fmt) = mtbuf_vertex_format_info(*format, shift != 0);
                        vertex_buf_fetches
                            .entry((vb_slot, attr_offset))
                            .and_modify(|p| {
                                p.components = p.components.max(attr_comps);
                                if p.fetch_format.is_none() && fetch_fmt.is_some() {
                                    p.component_type = comp_ty;
                                    p.fetch_format = fetch_fmt;
                                }
                            })
                            .or_insert(VertexFetchPlan {
                                components: attr_comps,
                                component_type: comp_ty,
                                fetch_format: fetch_fmt,
                            });
                        for d in 0..comps {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                    }
                } else if let Some(buf_op) = mtbuf_storage_op(*op) {
                    let writable = buf_op.is_store() && stage != ShaderStage::Vertex;
                    let key = desc_analysis
                        .inst_storage_buf_keys
                        .get(&inst_idx)
                        .copied()
                        .unwrap_or(ResourceKey::RawSlot(*srsrc));
                    storage_buf_usage
                        .entry(key)
                        .and_modify(|w| *w |= writable)
                        .or_insert(writable);
                    if *offen || *idxen {
                        vgprs.insert(*vaddr);
                        if !written_vgprs.contains(vaddr) {
                            read_before_write_vgprs.insert(*vaddr);
                        }
                    }
                    if *offen && *idxen {
                        let v1 = vaddr.saturating_add(1);
                        vgprs.insert(v1);
                        if !written_vgprs.contains(&v1) {
                            read_before_write_vgprs.insert(v1);
                        }
                    }
                    if *soffset <= 105 {
                        sgprs.insert(*soffset);
                        if !written_sgprs.contains(soffset) {
                            read_before_write_sgprs.insert(*soffset);
                        }
                    }
                    let cnt = buf_op.vdata_count();
                    if buf_op.is_store() {
                        for d in 0..cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            if !written_vgprs.contains(&reg) {
                                read_before_write_vgprs.insert(reg);
                            }
                        }
                    } else {
                        for d in 0..cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                    }
                }
            }
            Rdna2Detail::Ds {
                op,
                addr,
                data0,
                data1,
                vdst,
                ..
            } => {
                if stage == ShaderStage::Compute
                    && let Some(lds_op) = ds_lds_op(*op)
                {
                    uses_lds = true;
                    vgprs.insert(*addr);
                    if !written_vgprs.contains(addr) {
                        read_before_write_vgprs.insert(*addr);
                    }
                    match lds_op {
                        DsLdsOp::ReadContiguous(n) => {
                            for k in 0..n {
                                let reg = vdst.saturating_add(k);
                                vgprs.insert(reg);
                                written_vgprs.insert(reg);
                            }
                        }
                        DsLdsOp::Read2 { .. } => {
                            for k in 0..2u8 {
                                let reg = vdst.saturating_add(k);
                                vgprs.insert(reg);
                                written_vgprs.insert(reg);
                            }
                        }
                        DsLdsOp::WriteContiguous(n) => {
                            for k in 0..n {
                                let reg = data0.saturating_add(k);
                                vgprs.insert(reg);
                                if !written_vgprs.contains(&reg) {
                                    read_before_write_vgprs.insert(reg);
                                }
                            }
                        }
                        DsLdsOp::Write2 { .. } => {
                            for &reg in &[data0, data1] {
                                vgprs.insert(*reg);
                                if !written_vgprs.contains(reg) {
                                    read_before_write_vgprs.insert(*reg);
                                }
                            }
                        }
                    }
                }
            }
            Rdna2Detail::Exp {
                target,
                en,
                compr,
                vsrc,
                ..
            } => {
                if *compr {
                    if (*en & 0b0011) != 0 {
                        vgprs.insert(vsrc[0]);
                        if !written_vgprs.contains(&vsrc[0]) {
                            read_before_write_vgprs.insert(vsrc[0]);
                        }
                    }
                    if (*en & 0b1100) != 0 {
                        vgprs.insert(vsrc[1]);
                        if !written_vgprs.contains(&vsrc[1]) {
                            read_before_write_vgprs.insert(vsrc[1]);
                        }
                    }
                } else {
                    for (idx, &reg) in vsrc.iter().enumerate() {
                        if (*en & (1 << idx)) != 0 {
                            vgprs.insert(reg);
                            if !written_vgprs.contains(&reg) {
                                read_before_write_vgprs.insert(reg);
                            }
                        }
                    }
                }
                match target {
                    Rdna2ExportTarget::Param(n) if stage == ShaderStage::Vertex => {
                        // Reserve @builtin(position) + up to 15 param locations.
                        vs_param_set.insert(*n % (MAX_WEBGPU_LOCATIONS - 1));
                    }
                    Rdna2ExportTarget::Mrt(n) if stage == ShaderStage::Fragment => {
                        if *n < MAX_WEBGPU_MRTS {
                            ps_mrt_set.insert(*n);
                        }
                    }
                    Rdna2ExportTarget::MrtZ if stage == ShaderStage::Fragment => {
                        ps_exports_depth = true;
                    }
                    _ => {}
                }
            }
            Rdna2Detail::Sopp { op, .. } => match BranchKind::from_sopp_op(*op) {
                // Branch conditions lifted by `rdna2_cfg` read SCC / VCC / EXEC.
                Some(BranchKind::Scc0 | BranchKind::Scc1) => needs_scc = true,
                Some(BranchKind::Vccz | BranchKind::Vccnz) => needs_vcc = true,
                Some(BranchKind::Execz | BranchKind::Execnz) => needs_exec = true,
                Some(BranchKind::Unconditional) | None => {}
            },
            Rdna2Detail::Flat { .. } | Rdna2Detail::Unknown { .. } => {}
        }
    }

    let mut vertex_attrs: Vec<VertexInputAttr> = Vec::new();
    let mut legacy_uv_attr_index: Option<usize> = None;
    let vertex_input_mode = if stage == ShaderStage::Vertex {
        if has_buffer_fetch && !vertex_buf_fetches.is_empty() {
            // Map MUBUF/MTBUF vertex fetches deterministically by ascending (vb_slot, byte_offset):
            //   (0, 0)  -> @location(0) position: vec3 (3-comp) or vec4
            //   (0, 12) -> @location(1+) color: vec4<f32> (R8G8B8A8_UNORM) or exact integer width
            //   (0, 16) -> @location(2+) uv: vec2<f32> (R32G32_FLOAT at +16B)
            //   any other slot/offset -> deterministic sequential @location(N) attr_<offset>
            let mut next_loc: u32 = 0;
            for (&(vb_slot, byte_offset), &fetch) in &vertex_buf_fetches {
                let max_comps = fetch.components;
                let (location, name, components) = match (vb_slot, byte_offset) {
                    (0, 0) => {
                        let loc = next_loc;
                        next_loc = loc + 1;
                        let comps = if fetch.component_type != "f32" {
                            max_comps.clamp(1, 4)
                        } else if max_comps == 3 {
                            3
                        } else {
                            4
                        };
                        (loc, "position".to_string(), comps)
                    }
                    (0, 12) => {
                        let loc = next_loc.max(1);
                        next_loc = loc + 1;
                        let comps = if fetch.component_type != "f32" {
                            max_comps.clamp(1, 4)
                        } else {
                            4
                        };
                        (loc, "color".to_string(), comps)
                    }
                    (0, 16) => {
                        let loc = next_loc.max(2);
                        next_loc = loc + 1;
                        let comps = if fetch.component_type != "f32" {
                            max_comps.clamp(1, 4)
                        } else {
                            2
                        };
                        (loc, "uv".to_string(), comps)
                    }
                    (0, other) => {
                        let loc = next_loc;
                        next_loc = loc + 1;
                        (loc, format!("attr_{other}"), max_comps.clamp(1, 4))
                    }
                    (slot, other) => {
                        let loc = next_loc;
                        next_loc = loc + 1;
                        (loc, format!("vb{slot}_attr_{other}"), max_comps.clamp(1, 4))
                    }
                };
                if location < u32::from(MAX_WEBGPU_LOCATIONS) {
                    vertex_attrs.push(VertexInputAttr {
                        vb_slot,
                        byte_offset: Some(byte_offset),
                        location,
                        name,
                        components,
                        component_type: fetch.component_type,
                        fetch_format: fetch.fetch_format,
                    });
                }
            }
            // Preserve legacy multi-VGPR v4/v5 UV input when the shader reads v4 or v5
            // before writing and has no explicit offset-16 UV buffer fetch.
            let reads_legacy_uv =
                read_before_write_vgprs.contains(&4) || read_before_write_vgprs.contains(&5);
            if reads_legacy_uv
                && !vertex_buf_fetches.contains_key(&(0, 16))
                && next_loc < u32::from(MAX_WEBGPU_LOCATIONS)
            {
                legacy_uv_attr_index = Some(vertex_attrs.len());
                vertex_attrs.push(VertexInputAttr {
                    vb_slot: 0,
                    byte_offset: None,
                    location: next_loc,
                    name: "uv".to_string(),
                    components: 2,
                    component_type: "f32",
                    fetch_format: None,
                });
            }
            if vertex_attrs.iter().any(|a| a.byte_offset.is_some()) {
                VertexInputMode::Attributes
            } else {
                vertex_attrs.clear();
                legacy_uv_attr_index = None;
                vgprs.insert(0);
                VertexInputMode::VertexIndex
            }
        } else {
            // Buffer-pulling / procedural vertex shaders with zero IA VbDesc vertex fetches
            // use `@builtin(vertex_index)` only and emit `vertex_input_mode: "vertex_index"`
            // with no `@location(...)` inputs (S42 / #949).
            vgprs.insert(0);
            VertexInputMode::VertexIndex
        }
    } else {
        VertexInputMode::None
    };

    let workgroup_size = [
        options.workgroup_size[0].max(1),
        options.workgroup_size[1].max(1),
        options.workgroup_size[2].max(1),
    ];
    let mut compute_seed_wg_sgprs: Vec<(u8, u32)> = Vec::new();
    if stage == ShaderStage::Compute {
        vgprs.insert(0);
        let wg_base = options.workgroup_id_sgpr_base.unwrap_or(0);
        for c in 0..3u8 {
            let reg = wg_base.saturating_add(c);
            if reg <= 105
                && sgprs.contains(&reg)
                && (options.workgroup_id_sgpr_base.is_some()
                    || read_before_write_sgprs.contains(&reg))
            {
                compute_seed_wg_sgprs.push((reg, u32::from(c)));
            }
        }
    }

    let mut ps_mrts: Vec<u8> = ps_mrt_set.into_iter().collect();
    if stage == ShaderStage::Fragment && ps_mrts.is_empty() {
        ps_mrts.push(0);
    }

    let min_vec4s = options.min_cbv_vec4_count.clamp(1, MAX_CBV_VEC4_COUNT);
    let cbv_slots: Vec<CbvSlotPlan> = cbv_usage
        .into_iter()
        .map(|(key, usage)| {
            let static_needed = usage.max_vec4_index.saturating_add(1);
            let floor = if usage.has_dynamic_offset {
                min_vec4s.max(DEFAULT_DYNAMIC_CBV_VEC4_COUNT)
            } else {
                min_vec4s
            };
            let vec4_count = static_needed.max(floor).clamp(1, MAX_CBV_VEC4_COUNT);
            let highest_vec4_offset = usage.max_vec4_index.min(vec4_count - 1);
            CbvSlotPlan {
                key,
                vec4_count,
                highest_vec4_offset,
            }
        })
        .collect();
    let tex_slots: Vec<ResourceKey> = tex_dims.keys().copied().collect();
    let samp_slots: Vec<ResourceKey> = samp_rsrc_set.into_iter().collect();
    let storage_buf_slots: Vec<StorageBufSlotPlan> = storage_buf_usage
        .into_iter()
        .map(|(key, writable)| StorageBufSlotPlan { key, writable })
        .collect();
    let storage_tex_slots: Vec<ResourceKey> = storage_tex_dims.keys().copied().collect();

    let cbv_vec4_count = cbv_slots
        .iter()
        .map(|slot| slot.vec4_count)
        .max()
        .unwrap_or(0);

    ProgramPlan {
        stage,
        entry_point,
        workgroup_size,
        compute_seed_wg_sgprs,
        active_inst_count,
        vertex_input_mode,
        vertex_attrs,
        has_buffer_fetch,
        legacy_uv_attr_index,
        vsharp_fetch_insts,
        indirect_smem_cbv_keys,
        cbv_sources: desc_analysis.cbv_sources,
        inst_tex_keys: desc_analysis.inst_tex_keys,
        inst_samp_keys: desc_analysis.inst_samp_keys,
        inst_storage_buf_keys: desc_analysis.inst_storage_buf_keys,
        inst_storage_tex_keys: desc_analysis.inst_storage_tex_keys,
        tex_dims,
        storage_tex_dims,
        vertex_storage_buf_insts,
        vertex_fetch_vb_slots,
        cbv_slots,
        cbv_vec4_count,
        tex_slots,
        samp_slots,
        storage_buf_slots,
        storage_tex_slots,
        uses_lds,
        frag_attrs: frag_attr_set.into_iter().collect(),
        vs_params: vs_param_set.into_iter().collect(),
        ps_mrts,
        ps_exports_depth,
        vgprs,
        sgprs,
        lane_spill_slots,
        needs_vcc,
        needs_scc,
        needs_exec,
    }
}

// ---------------------------------------------------------------------------
// Naga IR builder
// ---------------------------------------------------------------------------

struct CommonTypes {
    f32_ty: Handle<Type>,
    u32_ty: Handle<Type>,
    i32_ty: Handle<Type>,
    vec2_f32_ty: Handle<Type>,
    vec2_i32_ty: Handle<Type>,
    vec3_f32_ty: Handle<Type>,
    vec3_i32_ty: Handle<Type>,
    vec4_f32_ty: Handle<Type>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstExprType {
    F32,
    U32,
    I32,
    Vec2F32,
}

struct IrEmitter {
    module: Module,
    types: CommonTypes,
    vsharp_fetch_insts: BTreeSet<usize>,
    indirect_smem_cbv_keys: BTreeMap<usize, CbvKey>,
    inst_tex_keys: BTreeMap<usize, ResourceKey>,
    inst_samp_keys: BTreeMap<usize, ResourceKey>,
    inst_storage_buf_keys: BTreeMap<usize, ResourceKey>,
    inst_storage_tex_keys: BTreeMap<usize, ResourceKey>,
    vertex_storage_buf_insts: BTreeSet<usize>,
    vertex_fetch_vb_slots: BTreeMap<usize, u32>,
    cb_globals: BTreeMap<CbvKey, (Handle<GlobalVariable>, u32)>,
    tex_globals: BTreeMap<ResourceKey, (Handle<GlobalVariable>, TexDim)>,
    samp_globals: BTreeMap<ResourceKey, Handle<GlobalVariable>>,
    storage_buf_globals: BTreeMap<ResourceKey, (Handle<GlobalVariable>, bool)>,
    storage_tex_globals: BTreeMap<ResourceKey, (Handle<GlobalVariable>, TexDim)>,
    lds_global: Option<Handle<GlobalVariable>>,
    emitted_bindings: Vec<WgslBindingInfo>,
    stage: ShaderStage,
    workgroup_size: [u32; 3],
    compute_seed_wg_sgprs: Vec<(u8, u32)>,
    func: Function,
    helper_u32_to_f32: Option<Handle<Function>>,
    helper_unpack2x16float: Option<Handle<Function>>,
    helper_pack2x16float: Option<Handle<Function>>,
    helper_rt_f32: Option<Handle<Function>>,
    helper_rt_u32: Option<Handle<Function>>,
    helper_rt_i32: Option<Handle<Function>>,
    helper_rt_vec2_f32: Option<Handle<Function>>,
    vgpr_locals: BTreeMap<u8, Handle<LocalVariable>>,
    sgpr_locals: BTreeMap<u8, Handle<LocalVariable>>,
    /// Dedicated per-`(vgpr, lane)` `u32` locals for lossless `v_writelane_b32`/`v_readlane_b32` SGPR spills.
    lane_spill_locals: BTreeMap<(u8, u8), Handle<LocalVariable>>,
    /// Reverse index `vgpr -> [(lane, local)]` for syncing VALU overwrites & dynamic-lane reads.
    vgpr_lane_slots: BTreeMap<u8, Vec<(u8, Handle<LocalVariable>)>>,
    vcc_local: Option<Handle<LocalVariable>>,
    scc_local: Option<Handle<LocalVariable>>,
    /// Single-lane `exec_mask: u32` (`1u` = this invocation's lane is active).
    exec_local: Option<Handle<LocalVariable>>,
    /// `true` when the body is lowered through `rdna2_cfg` structured control flow.
    structured_cf: bool,
    /// Nesting depth of structured `if` / `loop` / `switch` bodies currently being lowered.
    cf_depth: usize,
    /// Set once a `return` was emitted inside nested control flow: everything after it at the
    /// top level is non-uniform for WGSL uniformity analysis.
    cf_nonuniform_return: bool,
    /// Counter for unique structured-loop / dispatcher local names.
    cf_local_counter: usize,
    vs_out_local: Option<Handle<LocalVariable>>,
    vs_param_member_index: BTreeMap<u8, u32>,
    ps_out_local: Option<Handle<LocalVariable>>,
    ps_mrt_member_index: BTreeMap<u8, u32>,
    ps_depth_member_index: Option<u32>,
    frag_in_arg: Option<u32>,
    frag_attr_member_index: BTreeMap<u8, u32>,
    vs_in_arg: Option<u32>,
    /// Map from `(vb_slot, vertex buffer byte offset)` -> `(VsIn member_index, attr_components, scalar_kind)`.
    vs_attr_by_offset: BTreeMap<(u32, u16), (u32, u8, ScalarKind)>,
    has_buffer_fetch: bool,
    legacy_uv_member_index: Option<u32>,
    vertex_input_mode: VertexInputMode,
    translated_op_count: usize,
    unsupported_op_count: usize,
    unsupported_comments: Vec<String>,
    /// Sanitized informational comments for ops translated as deliberate no-ops.
    note_comments: Vec<String>,
}

fn align_up(value: u32, align: u32) -> u32 {
    if align <= 1 {
        return value;
    }
    (value + (align - 1)) & !(align - 1)
}

fn insert_type(module: &mut Module, name: Option<String>, inner: TypeInner) -> Handle<Type> {
    module
        .types
        .insert(Type { name, inner }, Span::UNDEFINED)
}

impl IrEmitter {
    fn new(plan: &ProgramPlan) -> Self {
        let mut module = Module::default();
        let f32_ty = insert_type(&mut module, None, TypeInner::Scalar(Scalar::F32));
        let u32_ty = insert_type(&mut module, None, TypeInner::Scalar(Scalar::U32));
        let i32_ty = insert_type(&mut module, None, TypeInner::Scalar(Scalar::I32));
        let bool_ty = insert_type(&mut module, None, TypeInner::Scalar(Scalar::BOOL));
        let vec2_f32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::F32,
            },
        );
        let vec2_i32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::I32,
            },
        );
        let vec3_f32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::F32,
            },
        );
        let vec3_i32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::I32,
            },
        );
        let vec3_u32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::U32,
            },
        );
        let vec4_f32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::F32,
            },
        );
        let vec4_u32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::U32,
            },
        );

        let types = CommonTypes {
            f32_ty,
            u32_ty,
            i32_ty,
            vec2_f32_ty,
            vec2_i32_ty,
            vec3_f32_ty,
            vec3_i32_ty,
            vec4_f32_ty,
        };

        // Uniform constant buffers:
        //   cb0 -> @group(0) @binding(0)
        //   tex0 -> @group(0) @binding(1)
        //   samp0 -> @group(0) @binding(2)
        //   Additional cb1../tex1../samp1.. -> @group(0) @binding(3+)
        let mut cb_globals: BTreeMap<CbvKey, (Handle<GlobalVariable>, u32)> = BTreeMap::new();
        let mut tex_globals: BTreeMap<ResourceKey, (Handle<GlobalVariable>, TexDim)> =
            BTreeMap::new();
        let mut samp_globals: BTreeMap<ResourceKey, Handle<GlobalVariable>> = BTreeMap::new();
        let mut storage_buf_globals: BTreeMap<ResourceKey, (Handle<GlobalVariable>, bool)> =
            BTreeMap::new();
        let mut storage_tex_globals: BTreeMap<ResourceKey, (Handle<GlobalVariable>, TexDim)> =
            BTreeMap::new();
        let mut emitted_bindings: Vec<WgslBindingInfo> = Vec::new();
        let has_any_binding = !plan.cbv_slots.is_empty()
            || !plan.tex_slots.is_empty()
            || !plan.samp_slots.is_empty()
            || !plan.storage_buf_slots.is_empty()
            || !plan.storage_tex_slots.is_empty();
        let all_bindings_reflected = has_any_binding
            && plan
                .cbv_slots
                .iter()
                .all(|s| plan.cbv_sources.contains_key(&s.key))
            && plan.tex_slots.iter().all(|s| s.to_source().is_some())
            && plan.samp_slots.iter().all(|s| s.to_source().is_some())
            && plan
                .storage_buf_slots
                .iter()
                .all(|s| s.key.to_source().is_some())
            && plan
                .storage_tex_slots
                .iter()
                .all(|s| s.to_source().is_some());
        let binding_base: u32 =
            if plan.stage == ShaderStage::Fragment && all_bindings_reflected {
                FRAGMENT_REFLECTED_BINDING_OFFSET
            } else {
                0
            };
        let mut next_extra_binding: u32 =
            if plan.stage == ShaderStage::Fragment && !all_bindings_reflected {
                FRAGMENT_EXTRA_BINDING_BASE
            } else {
                3
            };

        if !plan.cbv_slots.is_empty() {
            let mut cbv_struct_by_count: BTreeMap<u32, Handle<Type>> = BTreeMap::new();
            for (idx, slot) in plan.cbv_slots.iter().enumerate() {
                let count = slot.vec4_count.max(1);
                let cbv_struct_ty = *cbv_struct_by_count.entry(count).or_insert_with(|| {
                    let arr_ty = insert_type(
                        &mut module,
                        None,
                        TypeInner::Array {
                            base: vec4_u32_ty,
                            size: ArraySize::Constant(
                                NonZeroU32::new(count).unwrap_or(NonZeroU32::MIN),
                            ),
                            stride: 16,
                        },
                    );
                    let struct_name = if idx == 0 {
                        "Rdna2Cbv".to_string()
                    } else {
                        format!("Rdna2Cbv{idx}")
                    };
                    insert_type(
                        &mut module,
                        Some(struct_name),
                        TypeInner::Struct {
                            members: vec![StructMember {
                                name: Some("data".to_string()),
                                ty: arr_ty,
                                binding: None,
                                offset: 0,
                            }],
                            span: count.saturating_mul(16),
                        },
                    )
                });
                let binding_num = binding_base
                    + if idx == 0 {
                        0
                    } else {
                        let b = next_extra_binding;
                        next_extra_binding += 1;
                        b
                    };
                let name = format!("cb{idx}");
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Uniform,
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: cbv_struct_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                cb_globals.insert(slot.key, (g, count));
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind: "uniform",
                    name,
                    view_dimension: None,
                    storage_format: None,
                    min_binding_size_bytes: Some(count.saturating_mul(16)),
                    highest_vec4_offset: Some(slot.highest_vec4_offset),
                    source: plan.cbv_sources.get(&slot.key).copied(),
                });
            }
        }

        if !plan.tex_slots.is_empty() {
            let mut tex_2d_ty: Option<Handle<Type>> = None;
            let mut tex_2d_array_ty: Option<Handle<Type>> = None;
            let mut tex_3d_ty: Option<Handle<Type>> = None;
            let samp_ty = insert_type(
                &mut module,
                None,
                TypeInner::Sampler { comparison: false },
            );
            for (idx, &key) in plan.tex_slots.iter().enumerate() {
                let tex_dim = plan.tex_dims.get(&key).copied().unwrap_or(TexDim::D2);
                let tex_ty = match tex_dim {
                    TexDim::D2Array => *tex_2d_array_ty.get_or_insert_with(|| {
                        insert_type(
                            &mut module,
                            None,
                            TypeInner::Image {
                                dim: ImageDimension::D2,
                                arrayed: true,
                                class: ImageClass::Sampled {
                                    kind: ScalarKind::Float,
                                    multi: false,
                                },
                            },
                        )
                    }),
                    TexDim::D3 => *tex_3d_ty.get_or_insert_with(|| {
                        insert_type(
                            &mut module,
                            None,
                            TypeInner::Image {
                                dim: ImageDimension::D3,
                                arrayed: false,
                                class: ImageClass::Sampled {
                                    kind: ScalarKind::Float,
                                    multi: false,
                                },
                            },
                        )
                    }),
                    TexDim::D2 => *tex_2d_ty.get_or_insert_with(|| {
                        insert_type(
                            &mut module,
                            None,
                            TypeInner::Image {
                                dim: ImageDimension::D2,
                                arrayed: false,
                                class: ImageClass::Sampled {
                                    kind: ScalarKind::Float,
                                    multi: false,
                                },
                            },
                        )
                    }),
                };
                let binding_num = binding_base
                    + if idx == 0 {
                        1
                    } else {
                        let b = next_extra_binding;
                        next_extra_binding += 1;
                        b
                    };
                let name = format!("tex{idx}");
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Handle,
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: tex_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                tex_globals.insert(key, (g, tex_dim));
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind: "texture_2d",
                    name,
                    view_dimension: Some(tex_dim.view_dimension_name()),
                    storage_format: None,
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    source: key.to_source(),
                });
            }
            for (idx, &key) in plan.samp_slots.iter().enumerate() {
                let binding_num = binding_base
                    + if idx == 0 {
                        2
                    } else {
                        let b = next_extra_binding;
                        next_extra_binding += 1;
                        b
                    };
                let name = format!("samp{idx}");
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Handle,
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: samp_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                samp_globals.insert(key, g);
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind: "sampler",
                    name,
                    view_dimension: None,
                    storage_format: None,
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    source: key.to_source(),
                });
            }
        }

        if !plan.storage_buf_slots.is_empty() {
            let u32_runtime_arr_ty = insert_type(
                &mut module,
                None,
                TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            );
            let mut srv_idx = 0u32;
            let mut uav_idx = 0u32;
            for slot in &plan.storage_buf_slots {
                let binding_num = binding_base + next_extra_binding;
                next_extra_binding += 1;
                let (name, access, kind) = if slot.writable {
                    let n = format!("uav_buf{uav_idx}");
                    uav_idx += 1;
                    (
                        n,
                        StorageAccess::LOAD | StorageAccess::STORE,
                        "storage",
                    )
                } else {
                    let n = format!("sbuf{srv_idx}");
                    srv_idx += 1;
                    (n, StorageAccess::LOAD, "read-only-storage")
                };
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Storage { access },
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: u32_runtime_arr_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                storage_buf_globals.insert(slot.key, (g, slot.writable));
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind,
                    name,
                    view_dimension: None,
                    storage_format: None,
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    source: slot.key.to_source(),
                });
            }
        }

        if !plan.storage_tex_slots.is_empty() {
            let mut storage_tex_2d_ty: Option<Handle<Type>> = None;
            let mut storage_tex_2d_array_ty: Option<Handle<Type>> = None;
            let mut storage_tex_3d_ty: Option<Handle<Type>> = None;
            for (idx, &key) in plan.storage_tex_slots.iter().enumerate() {
                let tex_dim = plan.storage_tex_dims.get(&key).copied().unwrap_or(TexDim::D2);
                let storage_tex_ty = match tex_dim {
                    TexDim::D2Array => *storage_tex_2d_array_ty.get_or_insert_with(|| {
                        insert_type(
                            &mut module,
                            None,
                            TypeInner::Image {
                                dim: ImageDimension::D2,
                                arrayed: true,
                                class: ImageClass::Storage {
                                    format: StorageFormat::Rgba8Unorm,
                                    access: StorageAccess::STORE,
                                },
                            },
                        )
                    }),
                    TexDim::D3 => *storage_tex_3d_ty.get_or_insert_with(|| {
                        insert_type(
                            &mut module,
                            None,
                            TypeInner::Image {
                                dim: ImageDimension::D3,
                                arrayed: false,
                                class: ImageClass::Storage {
                                    format: StorageFormat::Rgba8Unorm,
                                    access: StorageAccess::STORE,
                                },
                            },
                        )
                    }),
                    TexDim::D2 => *storage_tex_2d_ty.get_or_insert_with(|| {
                        insert_type(
                            &mut module,
                            None,
                            TypeInner::Image {
                                dim: ImageDimension::D2,
                                arrayed: false,
                                class: ImageClass::Storage {
                                    format: StorageFormat::Rgba8Unorm,
                                    access: StorageAccess::STORE,
                                },
                            },
                        )
                    }),
                };
                let binding_num = binding_base + next_extra_binding;
                next_extra_binding += 1;
                let name = format!("uav_tex{idx}");
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Handle,
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: storage_tex_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                storage_tex_globals.insert(key, (g, tex_dim));
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind: "storage-texture",
                    name,
                    view_dimension: Some(tex_dim.view_dimension_name()),
                    storage_format: Some(WGPU_STORAGE_FORMAT_RGBA8_UNORM),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    source: key.to_source(),
                });
            }
        }

        let lds_global = if plan.uses_lds && plan.stage == ShaderStage::Compute {
            let lds_arr_ty = insert_type(
                &mut module,
                None,
                TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(
                        NonZeroU32::new(LDS_WORKGROUP_DWORDS).unwrap_or(NonZeroU32::MIN),
                    ),
                    stride: 4,
                },
            );
            Some(module.global_variables.append(
                GlobalVariable {
                    name: Some("lds".to_string()),
                    space: AddressSpace::WorkGroup,
                    binding: None,
                    ty: lds_arr_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            ))
        } else {
            None
        };

        emitted_bindings.sort_by_key(|b| (b.group, b.binding));

        let mut func = Function {
            name: Some(plan.entry_point.clone()),
            ..Function::default()
        };

        let mut vs_in_arg = None;
        let mut vs_attr_by_offset: BTreeMap<(u32, u16), (u32, u8, ScalarKind)> = BTreeMap::new();
        let mut legacy_uv_member_index: Option<u32> = None;
        let mut frag_in_arg = None;
        let mut frag_attr_member_index = BTreeMap::new();
        let mut vs_out_local = None;
        let mut vs_param_member_index = BTreeMap::new();
        let mut ps_out_local = None;
        let mut ps_mrt_member_index = BTreeMap::new();
        let mut ps_depth_member_index = None;

        // Stage inputs and outputs
        match plan.stage {
            ShaderStage::Vertex => {
                match plan.vertex_input_mode {
                    VertexInputMode::VertexIndex => {
                        func.arguments.push(FunctionArgument {
                            name: Some("vertex_index".to_string()),
                            ty: u32_ty,
                            binding: Some(Binding::BuiltIn(BuiltIn::VertexIndex)),
                        });
                        vs_in_arg = Some(0);
                    }
                    VertexInputMode::Attributes => {
                        let mut members = vec![StructMember {
                            name: Some("vertex_index".to_string()),
                            ty: u32_ty,
                            binding: Some(Binding::BuiltIn(BuiltIn::VertexIndex)),
                            offset: 0,
                        }];
                        let mut cur_offset = 4u32;
                        let mut max_align = 4u32;
                        let mut vec2_u32_ty: Option<Handle<Type>> = None;
                        let mut vec3_i32_ty: Option<Handle<Type>> = None;
                        let mut vec4_i32_ty: Option<Handle<Type>> = None;
                        for (attr_idx, attr) in plan.vertex_attrs.iter().enumerate() {
                            let scalar_kind = match attr.component_type {
                                "u32" => ScalarKind::Uint,
                                "i32" => ScalarKind::Sint,
                                _ => ScalarKind::Float,
                            };
                            let (ty, align, size) = match (scalar_kind, attr.components) {
                                (ScalarKind::Uint, 1) => (u32_ty, 4u32, 4u32),
                                (ScalarKind::Uint, 2) => (
                                    *vec2_u32_ty.get_or_insert_with(|| {
                                        insert_type(
                                            &mut module,
                                            None,
                                            TypeInner::Vector {
                                                size: VectorSize::Bi,
                                                scalar: Scalar::U32,
                                            },
                                        )
                                    }),
                                    8u32,
                                    8u32,
                                ),
                                (ScalarKind::Uint, 3) => (vec3_u32_ty, 16u32, 12u32),
                                (ScalarKind::Uint, _) => (vec4_u32_ty, 16u32, 16u32),
                                (ScalarKind::Sint, 1) => (i32_ty, 4u32, 4u32),
                                (ScalarKind::Sint, 2) => (vec2_i32_ty, 8u32, 8u32),
                                (ScalarKind::Sint, 3) => (
                                    *vec3_i32_ty.get_or_insert_with(|| {
                                        insert_type(
                                            &mut module,
                                            None,
                                            TypeInner::Vector {
                                                size: VectorSize::Tri,
                                                scalar: Scalar::I32,
                                            },
                                        )
                                    }),
                                    16u32,
                                    12u32,
                                ),
                                (ScalarKind::Sint, _) => (
                                    *vec4_i32_ty.get_or_insert_with(|| {
                                        insert_type(
                                            &mut module,
                                            None,
                                            TypeInner::Vector {
                                                size: VectorSize::Quad,
                                                scalar: Scalar::I32,
                                            },
                                        )
                                    }),
                                    16u32,
                                    16u32,
                                ),
                                (_, 1) => (f32_ty, 4u32, 4u32),
                                (_, 2) => (vec2_f32_ty, 8u32, 8u32),
                                (_, 3) => (vec3_f32_ty, 16u32, 12u32),
                                (_, _) => (vec4_f32_ty, 16u32, 16u32),
                            };
                            cur_offset = align_up(cur_offset, align);
                            max_align = max_align.max(align);
                            let member_idx = members.len() as u32;
                            if let Some(byte_off) = attr.byte_offset {
                                vs_attr_by_offset.insert(
                                    (attr.vb_slot, byte_off),
                                    (member_idx, attr.components, scalar_kind),
                                );
                            }
                            if plan.legacy_uv_attr_index == Some(attr_idx) {
                                legacy_uv_member_index = Some(member_idx);
                            }
                            members.push(StructMember {
                                name: Some(attr.name.clone()),
                                ty,
                                binding: Some(Binding::Location {
                                    location: attr.location,
                                    interpolation: None,
                                    sampling: None,
                                    blend_src: None,
                                    per_primitive: false,
                                }),
                                offset: cur_offset,
                            });
                            cur_offset += size;
                        }
                        let span = align_up(cur_offset, max_align);
                        let vs_in_ty = insert_type(
                            &mut module,
                            Some("VsIn".to_string()),
                            TypeInner::Struct { members, span },
                        );
                        func.arguments.push(FunctionArgument {
                            name: Some("vs_in".to_string()),
                            ty: vs_in_ty,
                            binding: None,
                        });
                        vs_in_arg = Some(0);
                    }
                    VertexInputMode::None => {}
                }

                let mut out_members = vec![StructMember {
                    name: Some("position".to_string()),
                    ty: vec4_f32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::Position { invariant: false })),
                    offset: 0,
                }];
                let mut offset = 16u32;
                for &param_idx in &plan.vs_params {
                    let member_idx = out_members.len() as u32;
                    vs_param_member_index.insert(param_idx, member_idx);
                    out_members.push(StructMember {
                        name: Some(format!("param_{param_idx}")),
                        ty: vec4_f32_ty,
                        binding: Some(Binding::Location {
                            location: u32::from(param_idx),
                            interpolation: Some(Interpolation::Perspective),
                            sampling: Some(Sampling::Center),
                            blend_src: None,
                            per_primitive: false,
                        }),
                        offset,
                    });
                    offset += 16;
                }
                let vs_out_ty = insert_type(
                    &mut module,
                    Some("VsOut".to_string()),
                    TypeInner::Struct {
                        members: out_members,
                        span: offset,
                    },
                );
                func.result = Some(FunctionResult {
                    ty: vs_out_ty,
                    binding: None,
                });
                let out_local = func.local_variables.append(
                    LocalVariable {
                        name: Some("vs_out".to_string()),
                        ty: vs_out_ty,
                        init: None,
                    },
                    Span::UNDEFINED,
                );
                vs_out_local = Some(out_local);
            }
            ShaderStage::Fragment => {
                if !plan.frag_attrs.is_empty() {
                    let mut in_members = Vec::with_capacity(plan.frag_attrs.len());
                    let mut offset = 0u32;
                    for &attr in &plan.frag_attrs {
                        let member_idx = in_members.len() as u32;
                        frag_attr_member_index.insert(attr, member_idx);
                        in_members.push(StructMember {
                            name: Some(format!("attr_{attr}")),
                            ty: vec4_f32_ty,
                            binding: Some(Binding::Location {
                                location: u32::from(attr),
                                interpolation: Some(Interpolation::Perspective),
                                sampling: Some(Sampling::Center),
                                blend_src: None,
                                per_primitive: false,
                            }),
                            offset,
                        });
                        offset += 16;
                    }
                    let frag_in_ty = insert_type(
                        &mut module,
                        Some("FragIn".to_string()),
                        TypeInner::Struct {
                            members: in_members,
                            span: offset,
                        },
                    );
                    func.arguments.push(FunctionArgument {
                        name: Some("frag_in".to_string()),
                        ty: frag_in_ty,
                        binding: None,
                    });
                    frag_in_arg = Some(0);
                }

                let mut out_members = Vec::new();
                let mut offset = 0u32;
                for &mrt in &plan.ps_mrts {
                    let member_idx = out_members.len() as u32;
                    ps_mrt_member_index.insert(mrt, member_idx);
                    out_members.push(StructMember {
                        name: Some(format!("mrt_{mrt}")),
                        ty: vec4_f32_ty,
                        binding: Some(Binding::Location {
                            location: u32::from(mrt),
                            interpolation: None,
                            sampling: None,
                            blend_src: None,
                            per_primitive: false,
                        }),
                        offset,
                    });
                    offset += 16;
                }
                if plan.ps_exports_depth {
                    let member_idx = out_members.len() as u32;
                    ps_depth_member_index = Some(member_idx);
                    out_members.push(StructMember {
                        name: Some("frag_depth".to_string()),
                        ty: f32_ty,
                        binding: Some(Binding::BuiltIn(BuiltIn::FragDepth)),
                        offset,
                    });
                    offset += 16;
                }
                let ps_out_ty = insert_type(
                    &mut module,
                    Some("PsOut".to_string()),
                    TypeInner::Struct {
                        members: out_members,
                        span: offset,
                    },
                );
                func.result = Some(FunctionResult {
                    ty: ps_out_ty,
                    binding: None,
                });
                let out_local = func.local_variables.append(
                    LocalVariable {
                        name: Some("ps_out".to_string()),
                        ty: ps_out_ty,
                        init: None,
                    },
                    Span::UNDEFINED,
                );
                ps_out_local = Some(out_local);
            }
            ShaderStage::Compute => {
                func.arguments.push(FunctionArgument {
                    name: Some("local_id".to_string()),
                    ty: vec3_u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationId)),
                });
                func.arguments.push(FunctionArgument {
                    name: Some("wg_id".to_string()),
                    ty: vec3_u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
                });
                func.arguments.push(FunctionArgument {
                    name: Some("global_id".to_string()),
                    ty: vec3_u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::GlobalInvocationId)),
                });
            }
        }

        // Allocate VGPR locals (`var v_i: f32`) and SGPR locals (`var s_i: u32`).
        let mut vgpr_locals = BTreeMap::new();
        for &v in &plan.vgprs {
            let local = func.local_variables.append(
                LocalVariable {
                    name: Some(format!("v_{v}")),
                    ty: f32_ty,
                    init: None,
                },
                Span::UNDEFINED,
            );
            vgpr_locals.insert(v, local);
        }

        let mut sgpr_locals = BTreeMap::new();
        for &s in &plan.sgprs {
            let local = func.local_variables.append(
                LocalVariable {
                    name: Some(format!("s_{s}")),
                    ty: u32_ty,
                    init: None,
                },
                Span::UNDEFINED,
            );
            sgpr_locals.insert(s, local);
        }

        let mut lane_spill_locals = BTreeMap::new();
        let mut vgpr_lane_slots: BTreeMap<u8, Vec<(u8, Handle<LocalVariable>)>> = BTreeMap::new();
        for &(v, lane) in &plan.lane_spill_slots {
            let local = func.local_variables.append(
                LocalVariable {
                    name: Some(format!("lane_v{v}_{lane}")),
                    ty: u32_ty,
                    init: None,
                },
                Span::UNDEFINED,
            );
            lane_spill_locals.insert((v, lane), local);
            vgpr_lane_slots.entry(v).or_default().push((lane, local));
        }

        let vcc_local = if plan.needs_vcc {
            Some(func.local_variables.append(
                LocalVariable {
                    name: Some("vcc".to_string()),
                    ty: bool_ty,
                    init: None,
                },
                Span::UNDEFINED,
            ))
        } else {
            None
        };

        let scc_local = if plan.needs_scc {
            Some(func.local_variables.append(
                LocalVariable {
                    name: Some("scc".to_string()),
                    ty: bool_ty,
                    init: None,
                },
                Span::UNDEFINED,
            ))
        } else {
            None
        };

        let exec_local = if plan.needs_exec {
            Some(func.local_variables.append(
                LocalVariable {
                    name: Some("exec_mask".to_string()),
                    ty: u32_ty,
                    init: None,
                },
                Span::UNDEFINED,
            ))
        } else {
            None
        };

        let mut emitter = Self {
            module,
            types,
            vsharp_fetch_insts: plan.vsharp_fetch_insts.clone(),
            indirect_smem_cbv_keys: plan.indirect_smem_cbv_keys.clone(),
            inst_tex_keys: plan.inst_tex_keys.clone(),
            inst_samp_keys: plan.inst_samp_keys.clone(),
            inst_storage_buf_keys: plan.inst_storage_buf_keys.clone(),
            inst_storage_tex_keys: plan.inst_storage_tex_keys.clone(),
            vertex_storage_buf_insts: plan.vertex_storage_buf_insts.clone(),
            vertex_fetch_vb_slots: plan.vertex_fetch_vb_slots.clone(),
            cb_globals,
            tex_globals,
            samp_globals,
            storage_buf_globals,
            storage_tex_globals,
            lds_global,
            emitted_bindings,
            stage: plan.stage,
            workgroup_size: plan.workgroup_size,
            compute_seed_wg_sgprs: plan.compute_seed_wg_sgprs.clone(),
            func,
            helper_u32_to_f32: None,
            helper_unpack2x16float: None,
            helper_pack2x16float: None,
            helper_rt_f32: None,
            helper_rt_u32: None,
            helper_rt_i32: None,
            helper_rt_vec2_f32: None,
            vgpr_locals,
            sgpr_locals,
            lane_spill_locals,
            vgpr_lane_slots,
            vcc_local,
            scc_local,
            exec_local,
            structured_cf: false,
            cf_depth: 0,
            cf_nonuniform_return: false,
            cf_local_counter: 0,
            vs_out_local,
            vs_param_member_index,
            ps_out_local,
            ps_mrt_member_index,
            ps_depth_member_index,
            frag_in_arg,
            frag_attr_member_index,
            vs_in_arg,
            vs_attr_by_offset,
            has_buffer_fetch: plan.has_buffer_fetch,
            legacy_uv_member_index,
            vertex_input_mode: plan.vertex_input_mode,
            translated_op_count: 0,
            unsupported_op_count: 0,
            unsupported_comments: Vec::new(),
            note_comments: Vec::new(),
        };

        emitter.emit_prologue();
        emitter
    }

    // -- Expression & statement helpers --

    fn raw_expr(&mut self, expr: Expression) -> Handle<Expression> {
        self.func.expressions.append(expr, Span::UNDEFINED)
    }

    fn emit_expr_raw(&mut self, expr: Expression) -> Handle<Expression> {
        let h = self.func.expressions.append(expr, Span::UNDEFINED);
        self.func
            .body
            .push(Statement::Emit(Range::new_from_bounds(h, h)), Span::UNDEFINED);
        h
    }

    fn is_const_expr(&self, h: Handle<Expression>) -> bool {
        match self.func.expressions[h] {
            Expression::Literal(_) | Expression::ZeroValue(_) => true,
            Expression::Unary { expr, .. } | Expression::As { expr, .. } => {
                self.is_const_expr(expr)
            }
            Expression::Binary { left, right, .. } => {
                self.is_const_expr(left) && self.is_const_expr(right)
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                self.is_const_expr(condition)
                    && self.is_const_expr(accept)
                    && self.is_const_expr(reject)
            }
            Expression::Splat { value, .. } => self.is_const_expr(value),
            Expression::Swizzle { vector, .. } => self.is_const_expr(vector),
            Expression::AccessIndex { base, .. } => self.is_const_expr(base),
            Expression::Compose { ref components, .. } => {
                components.iter().all(|&c| self.is_const_expr(c))
            }
            Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                self.is_const_expr(arg)
                    && arg1.is_none_or(|a| self.is_const_expr(a))
                    && arg2.is_none_or(|a| self.is_const_expr(a))
                    && arg3.is_none_or(|a| self.is_const_expr(a))
            }
            _ => false,
        }
    }

    fn const_expr_type(&self, h: Handle<Expression>) -> Option<ConstExprType> {
        match self.func.expressions[h] {
            Expression::Literal(Literal::F32(_)) => Some(ConstExprType::F32),
            Expression::Literal(Literal::U32(_)) => Some(ConstExprType::U32),
            Expression::Literal(Literal::I32(_)) => Some(ConstExprType::I32),
            Expression::As { kind, .. } => match kind {
                ScalarKind::Float => Some(ConstExprType::F32),
                ScalarKind::Uint => Some(ConstExprType::U32),
                ScalarKind::Sint => Some(ConstExprType::I32),
                _ => None,
            },
            Expression::Compose { ty, .. } if ty == self.types.vec2_f32_ty => {
                Some(ConstExprType::Vec2F32)
            }
            Expression::Splat {
                size: VectorSize::Bi,
                value,
            } if self.const_expr_type(value) == Some(ConstExprType::F32) => {
                Some(ConstExprType::Vec2F32)
            }
            Expression::Swizzle {
                size: VectorSize::Bi,
                vector,
                ..
            } if self.const_expr_type(vector) == Some(ConstExprType::Vec2F32) => {
                Some(ConstExprType::Vec2F32)
            }
            Expression::AccessIndex { base, .. }
                if self.const_expr_type(base) == Some(ConstExprType::Vec2F32) =>
            {
                Some(ConstExprType::F32)
            }
            Expression::Unary { expr, .. } => self.const_expr_type(expr),
            Expression::Binary { op, left, .. } => match op {
                BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterEqual
                | BinaryOperator::LogicalAnd
                | BinaryOperator::LogicalOr => None,
                _ => self.const_expr_type(left),
            },
            Expression::Select { accept, .. } => self.const_expr_type(accept),
            Expression::Math { fun, arg, .. } => match fun {
                MathFunction::Unpack2x16float => Some(ConstExprType::Vec2F32),
                MathFunction::Pack2x16float => Some(ConstExprType::U32),
                MathFunction::Dot => Some(ConstExprType::F32),
                _ => self.const_expr_type(arg),
            },
            _ => None,
        }
    }

    fn call_helper_1(
        &mut self,
        func_handle: Handle<Function>,
        arg: Handle<Expression>,
    ) -> Handle<Expression> {
        let res = self
            .func
            .expressions
            .append(Expression::CallResult(func_handle), Span::UNDEFINED);
        self.func.body.push(
            Statement::Call {
                function: func_handle,
                arguments: vec![arg],
                result: Some(res),
            },
            Span::UNDEFINED,
        );
        res
    }

    fn ensure_helper_u32_to_f32(&mut self) -> Handle<Function> {
        if let Some(h) = self.helper_u32_to_f32 {
            return h;
        }
        let mut f = Function {
            name: Some("pg6_u32_bits_to_float".to_string()),
            arguments: vec![FunctionArgument {
                name: Some("x".to_string()),
                ty: self.types.u32_ty,
                binding: None,
            }],
            result: Some(FunctionResult {
                ty: self.types.f32_ty,
                binding: None,
            }),
            ..Function::default()
        };
        let arg = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        let cast = f.expressions.append(
            Expression::As {
                expr: arg,
                kind: ScalarKind::Float,
                convert: None,
            },
            Span::UNDEFINED,
        );
        f.body
            .push(Statement::Emit(Range::new_from_bounds(cast, cast)), Span::UNDEFINED);
        f.body
            .push(Statement::Return { value: Some(cast) }, Span::UNDEFINED);
        let h = self.module.functions.append(f, Span::UNDEFINED);
        self.helper_u32_to_f32 = Some(h);
        h
    }

    fn call_pg6_u32_to_f32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = self.ensure_helper_u32_to_f32();
        self.call_helper_1(f, arg)
    }

    fn ensure_helper_unpack2x16float(&mut self) -> Handle<Function> {
        if let Some(h) = self.helper_unpack2x16float {
            return h;
        }
        let mut f = Function {
            name: Some("pg6_unpack_half_pair".to_string()),
            arguments: vec![FunctionArgument {
                name: Some("x".to_string()),
                ty: self.types.u32_ty,
                binding: None,
            }],
            result: Some(FunctionResult {
                ty: self.types.vec2_f32_ty,
                binding: None,
            }),
            ..Function::default()
        };
        let arg = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        let unpacked = f.expressions.append(
            Expression::Math {
                fun: MathFunction::Unpack2x16float,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Emit(Range::new_from_bounds(unpacked, unpacked)),
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Return {
                value: Some(unpacked),
            },
            Span::UNDEFINED,
        );
        let h = self.module.functions.append(f, Span::UNDEFINED);
        self.helper_unpack2x16float = Some(h);
        h
    }

    fn call_pg6_unpack2x16float(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = self.ensure_helper_unpack2x16float();
        self.call_helper_1(f, arg)
    }

    fn ensure_helper_pack2x16float(&mut self) -> Handle<Function> {
        if let Some(h) = self.helper_pack2x16float {
            return h;
        }
        let mut f = Function {
            name: Some("pg6_pack_half_pair".to_string()),
            arguments: vec![FunctionArgument {
                name: Some("v".to_string()),
                ty: self.types.vec2_f32_ty,
                binding: None,
            }],
            result: Some(FunctionResult {
                ty: self.types.u32_ty,
                binding: None,
            }),
            ..Function::default()
        };
        let arg = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        let packed = f.expressions.append(
            Expression::Math {
                fun: MathFunction::Pack2x16float,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Emit(Range::new_from_bounds(packed, packed)),
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Return {
                value: Some(packed),
            },
            Span::UNDEFINED,
        );
        let h = self.module.functions.append(f, Span::UNDEFINED);
        self.helper_pack2x16float = Some(h);
        h
    }

    fn call_pg6_pack2x16float(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = self.ensure_helper_pack2x16float();
        self.call_helper_1(f, arg)
    }

    fn ensure_identity_helper(
        module: &mut Module,
        slot: &mut Option<Handle<Function>>,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<Function> {
        if let Some(h) = *slot {
            return h;
        }
        let mut f = Function {
            name: Some(name.to_string()),
            arguments: vec![FunctionArgument {
                name: Some("x".to_string()),
                ty,
                binding: None,
            }],
            result: Some(FunctionResult { ty, binding: None }),
            ..Function::default()
        };
        let arg = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        f.body
            .push(Statement::Return { value: Some(arg) }, Span::UNDEFINED);
        let h = module.functions.append(f, Span::UNDEFINED);
        *slot = Some(h);
        h
    }

    fn call_pg6_rt_f32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = Self::ensure_identity_helper(
            &mut self.module,
            &mut self.helper_rt_f32,
            "pg6_runtime_float",
            self.types.f32_ty,
        );
        self.call_helper_1(f, arg)
    }

    fn call_pg6_rt_u32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = Self::ensure_identity_helper(
            &mut self.module,
            &mut self.helper_rt_u32,
            "pg6_runtime_uint",
            self.types.u32_ty,
        );
        self.call_helper_1(f, arg)
    }

    fn call_pg6_rt_i32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = Self::ensure_identity_helper(
            &mut self.module,
            &mut self.helper_rt_i32,
            "pg6_runtime_sint",
            self.types.i32_ty,
        );
        self.call_helper_1(f, arg)
    }

    fn call_pg6_rt_vec2_f32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = Self::ensure_identity_helper(
            &mut self.module,
            &mut self.helper_rt_vec2_f32,
            "pg6_runtime_vec2f",
            self.types.vec2_f32_ty,
        );
        self.call_helper_1(f, arg)
    }

    fn wrap_const_runtime(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        match self.const_expr_type(arg) {
            Some(ConstExprType::F32) => self.call_pg6_rt_f32(arg),
            Some(ConstExprType::U32) => self.call_pg6_rt_u32(arg),
            Some(ConstExprType::I32) => self.call_pg6_rt_i32(arg),
            Some(ConstExprType::Vec2F32) => self.call_pg6_rt_vec2_f32(arg),
            None => arg,
        }
    }

    fn emit_expr(&mut self, expr: Expression) -> Handle<Expression> {
        match expr {
            // Fold integer/boolean constant binary ops in Rust (wrapping semantics) so Tint
            // WGSL const-eval never rejects shift/add/sub/mul overflow on literal operands,
            // and guard non-finite float const-eval operations through runtime helpers.
            Expression::Binary { op, left, right } => {
                match (&self.func.expressions[left], &self.func.expressions[right]) {
                    (
                        Expression::Literal(Literal::U32(a)),
                        Expression::Literal(Literal::U32(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinaryOperator::Add => return self.lit_u32(a.wrapping_add(b)),
                            BinaryOperator::Subtract => return self.lit_u32(a.wrapping_sub(b)),
                            BinaryOperator::Multiply => return self.lit_u32(a.wrapping_mul(b)),
                            BinaryOperator::Divide => {
                                return self.lit_u32(if b == 0 { 0 } else { a / b });
                            }
                            BinaryOperator::Modulo => {
                                return self.lit_u32(if b == 0 { 0 } else { a % b });
                            }
                            BinaryOperator::And => return self.lit_u32(a & b),
                            BinaryOperator::InclusiveOr => return self.lit_u32(a | b),
                            BinaryOperator::ExclusiveOr => return self.lit_u32(a ^ b),
                            BinaryOperator::ShiftLeft => {
                                return self.lit_u32(a.wrapping_shl(b & 31));
                            }
                            BinaryOperator::ShiftRight => {
                                return self.lit_u32(a.wrapping_shr(b & 31));
                            }
                            BinaryOperator::Equal => return self.lit_bool(a == b),
                            BinaryOperator::NotEqual => return self.lit_bool(a != b),
                            BinaryOperator::Less => return self.lit_bool(a < b),
                            BinaryOperator::LessEqual => return self.lit_bool(a <= b),
                            BinaryOperator::Greater => return self.lit_bool(a > b),
                            BinaryOperator::GreaterEqual => return self.lit_bool(a >= b),
                            _ => {}
                        }
                    }
                    (
                        Expression::Literal(Literal::I32(a)),
                        Expression::Literal(Literal::U32(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinaryOperator::ShiftLeft => {
                                return self.lit_i32(a.wrapping_shl(b & 31));
                            }
                            BinaryOperator::ShiftRight => {
                                return self.lit_i32(a.wrapping_shr(b & 31));
                            }
                            _ => {}
                        }
                    }
                    (
                        Expression::Literal(Literal::I32(a)),
                        Expression::Literal(Literal::I32(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinaryOperator::Add => return self.lit_i32(a.wrapping_add(b)),
                            BinaryOperator::Subtract => return self.lit_i32(a.wrapping_sub(b)),
                            BinaryOperator::Multiply => return self.lit_i32(a.wrapping_mul(b)),
                            BinaryOperator::Divide => {
                                return self.lit_i32(if b == 0 { 0 } else { a.wrapping_div(b) });
                            }
                            BinaryOperator::Modulo => {
                                return self.lit_i32(if b == 0 { 0 } else { a.wrapping_rem(b) });
                            }
                            BinaryOperator::And => return self.lit_i32(a & b),
                            BinaryOperator::InclusiveOr => return self.lit_i32(a | b),
                            BinaryOperator::ExclusiveOr => return self.lit_i32(a ^ b),
                            BinaryOperator::Equal => return self.lit_bool(a == b),
                            BinaryOperator::NotEqual => return self.lit_bool(a != b),
                            BinaryOperator::Less => return self.lit_bool(a < b),
                            BinaryOperator::LessEqual => return self.lit_bool(a <= b),
                            BinaryOperator::Greater => return self.lit_bool(a > b),
                            BinaryOperator::GreaterEqual => return self.lit_bool(a >= b),
                            _ => {}
                        }
                    }
                    (
                        Expression::Literal(Literal::Bool(a)),
                        Expression::Literal(Literal::Bool(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinaryOperator::LogicalAnd => return self.lit_bool(a && b),
                            BinaryOperator::LogicalOr => return self.lit_bool(a || b),
                            BinaryOperator::Equal => return self.lit_bool(a == b),
                            BinaryOperator::NotEqual => return self.lit_bool(a != b),
                            _ => {}
                        }
                    }
                    (
                        Expression::Literal(Literal::F32(a)),
                        Expression::Literal(Literal::F32(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        let unsafe_const = match op {
                            BinaryOperator::Add => !(a + b).is_finite(),
                            BinaryOperator::Subtract => !(a - b).is_finite(),
                            BinaryOperator::Multiply => !(a * b).is_finite(),
                            BinaryOperator::Divide => b == 0.0 || !(a / b).is_finite(),
                            _ => false,
                        };
                        if unsafe_const {
                            let rt_left = self.call_pg6_rt_f32(left);
                            return self.emit_expr_raw(Expression::Binary {
                                op,
                                left: rt_left,
                                right,
                            });
                        }
                    }
                    _ => {
                        if matches!(
                            op,
                            BinaryOperator::Add
                                | BinaryOperator::Subtract
                                | BinaryOperator::Multiply
                                | BinaryOperator::Divide
                                | BinaryOperator::Modulo
                                | BinaryOperator::ShiftLeft
                                | BinaryOperator::ShiftRight
                        ) && self.is_const_expr(left)
                            && self.is_const_expr(right)
                        {
                            let rt_left = self.wrap_const_runtime(left);
                            return self.emit_expr_raw(Expression::Binary {
                                op,
                                left: rt_left,
                                right,
                            });
                        }
                    }
                }
                self.emit_expr_raw(Expression::Binary { op, left, right })
            }
            Expression::Unary { op, expr: inner } => {
                match (op, &self.func.expressions[inner]) {
                    (UnaryOperator::BitwiseNot, Expression::Literal(Literal::U32(a))) => {
                        return self.lit_u32(!*a);
                    }
                    (UnaryOperator::BitwiseNot, Expression::Literal(Literal::I32(a))) => {
                        return self.lit_i32(!*a);
                    }
                    (UnaryOperator::Negate, Expression::Literal(Literal::I32(a))) => {
                        return self.lit_i32((*a).wrapping_neg());
                    }
                    (UnaryOperator::LogicalNot, Expression::Literal(Literal::Bool(a))) => {
                        return self.lit_bool(!*a);
                    }
                    _ => {}
                }
                self.emit_expr_raw(Expression::Unary { op, expr: inner })
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                if let Expression::Literal(Literal::Bool(c)) = self.func.expressions[condition] {
                    return if c { accept } else { reject };
                }
                self.emit_expr_raw(Expression::Select {
                    condition,
                    accept,
                    reject,
                })
            }
            Expression::As {
                expr: inner,
                kind,
                convert,
            } => {
                match convert {
                    None => {
                        match (&self.func.expressions[inner], kind) {
                            (Expression::Literal(Literal::U32(bits)), ScalarKind::Float) => {
                                let bits = *bits;
                                let f = f32::from_bits(bits);
                                if is_safe_wgsl_f32_literal(f) {
                                    return self.raw_expr(Expression::Literal(Literal::F32(f)));
                                }
                                return self.call_pg6_u32_to_f32(inner);
                            }
                            (Expression::Literal(Literal::I32(val)), ScalarKind::Float) => {
                                let bits = *val as u32;
                                let f = f32::from_bits(bits);
                                if is_safe_wgsl_f32_literal(f) {
                                    return self.raw_expr(Expression::Literal(Literal::F32(f)));
                                }
                                let u = self.lit_u32(bits);
                                return self.call_pg6_u32_to_f32(u);
                            }
                            (Expression::Literal(Literal::U32(val)), ScalarKind::Sint) => {
                                return self.lit_i32(*val as i32);
                            }
                            (Expression::Literal(Literal::I32(val)), ScalarKind::Uint) => {
                                return self.lit_u32(*val as u32);
                            }
                            (Expression::Literal(Literal::F32(f)), ScalarKind::Uint) => {
                                return self.lit_u32(f.to_bits());
                            }
                            (Expression::Literal(Literal::F32(f)), ScalarKind::Sint) => {
                                return self.lit_i32(f.to_bits() as i32);
                            }
                            _ if kind == ScalarKind::Float && self.is_const_expr(inner) => {
                                let u_arg =
                                    if self.const_expr_type(inner) == Some(ConstExprType::I32) {
                                        self.emit_expr_raw(Expression::As {
                                            expr: inner,
                                            kind: ScalarKind::Uint,
                                            convert: None,
                                        })
                                    } else {
                                        inner
                                    };
                                return self.call_pg6_u32_to_f32(u_arg);
                            }
                            _ => {}
                        }
                    }
                    Some(4) => match (&self.func.expressions[inner], kind) {
                        (Expression::Literal(Literal::F32(f)), ScalarKind::Uint)
                            if !(0.0..4_294_967_296.0).contains(f) =>
                        {
                            let rt = self.call_pg6_rt_f32(inner);
                            return self.emit_expr_raw(Expression::As {
                                expr: rt,
                                kind,
                                convert,
                            });
                        }
                        (Expression::Literal(Literal::F32(f)), ScalarKind::Sint)
                            if !(-2_147_483_648.0..2_147_483_648.0).contains(f) =>
                        {
                            let rt = self.call_pg6_rt_f32(inner);
                            return self.emit_expr_raw(Expression::As {
                                expr: rt,
                                kind,
                                convert,
                            });
                        }
                        _ if matches!(kind, ScalarKind::Uint | ScalarKind::Sint)
                            && self.is_const_expr(inner)
                            && self.const_expr_type(inner) == Some(ConstExprType::F32) =>
                        {
                            let rt = self.call_pg6_rt_f32(inner);
                            return self.emit_expr_raw(Expression::As {
                                expr: rt,
                                kind,
                                convert,
                            });
                        }
                        _ => {}
                    },
                    _ => {}
                }
                self.emit_expr_raw(Expression::As {
                    expr: inner,
                    kind,
                    convert,
                })
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                match fun {
                    MathFunction::Unpack2x16float => {
                        if let Expression::Literal(Literal::U32(bits)) =
                            self.func.expressions[arg]
                        {
                            let lo_non_finite = (bits & 0x7C00) == 0x7C00;
                            let hi_non_finite = ((bits >> 16) & 0x7C00) == 0x7C00;
                            if lo_non_finite || hi_non_finite {
                                return self.call_pg6_unpack2x16float(arg);
                            }
                        } else if self.is_const_expr(arg) {
                            return self.call_pg6_unpack2x16float(arg);
                        }
                    }
                    MathFunction::Pack2x16float => {
                        if self.is_const_expr(arg) {
                            return self.call_pg6_pack2x16float(arg);
                        }
                    }
                    MathFunction::ReverseBits => {
                        if let Expression::Literal(Literal::U32(a)) = self.func.expressions[arg] {
                            return self.lit_u32(a.reverse_bits());
                        }
                    }
                    MathFunction::CountOneBits => {
                        if let Expression::Literal(Literal::U32(a)) = self.func.expressions[arg] {
                            return self.lit_u32(a.count_ones());
                        }
                    }
                    MathFunction::CountLeadingZeros => {
                        if let Expression::Literal(Literal::U32(a)) = self.func.expressions[arg] {
                            return self.lit_u32(a.leading_zeros());
                        }
                    }
                    MathFunction::CountTrailingZeros => {
                        if let Expression::Literal(Literal::U32(a)) = self.func.expressions[arg] {
                            return self.lit_u32(a.trailing_zeros());
                        }
                    }
                    MathFunction::Abs => {
                        if let Expression::Literal(Literal::I32(a)) = self.func.expressions[arg] {
                            return self.lit_i32(a.wrapping_abs());
                        }
                    }
                    MathFunction::Min | MathFunction::Max => {
                        if let Some(a1) = arg1 {
                            match (&self.func.expressions[arg], &self.func.expressions[a1]) {
                                (
                                    Expression::Literal(Literal::U32(a)),
                                    Expression::Literal(Literal::U32(b)),
                                ) => {
                                    let res = if fun == MathFunction::Min {
                                        (*a).min(*b)
                                    } else {
                                        (*a).max(*b)
                                    };
                                    return self.lit_u32(res);
                                }
                                (
                                    Expression::Literal(Literal::I32(a)),
                                    Expression::Literal(Literal::I32(b)),
                                ) => {
                                    let res = if fun == MathFunction::Min {
                                        (*a).min(*b)
                                    } else {
                                        (*a).max(*b)
                                    };
                                    return self.lit_i32(res);
                                }
                                _ => {}
                            }
                        }
                    }
                    MathFunction::Clamp => {
                        if let (Some(lo_h), Some(hi_h)) = (arg1, arg2) {
                            match (
                                &self.func.expressions[arg],
                                &self.func.expressions[lo_h],
                                &self.func.expressions[hi_h],
                            ) {
                                (
                                    Expression::Literal(Literal::U32(a)),
                                    Expression::Literal(Literal::U32(lo)),
                                    Expression::Literal(Literal::U32(hi)),
                                ) => {
                                    return self.lit_u32((*a).max(*lo).min(*hi));
                                }
                                (
                                    Expression::Literal(Literal::I32(a)),
                                    Expression::Literal(Literal::I32(lo)),
                                    Expression::Literal(Literal::I32(hi)),
                                ) => {
                                    return self.lit_i32((*a).max(*lo).min(*hi));
                                }
                                (
                                    Expression::Literal(Literal::F32(_)),
                                    Expression::Literal(Literal::F32(lo)),
                                    Expression::Literal(Literal::F32(hi)),
                                ) => {
                                    if *lo > *hi || !lo.is_finite() || !hi.is_finite() {
                                        let rt = self.call_pg6_rt_f32(arg);
                                        return self.emit_expr_raw(Expression::Math {
                                            fun,
                                            arg: rt,
                                            arg1,
                                            arg2,
                                            arg3,
                                        });
                                    }
                                }
                                _ if self.is_const_expr(arg)
                                    && self.is_const_expr(lo_h)
                                    && self.is_const_expr(hi_h) =>
                                {
                                    let rt = self.wrap_const_runtime(arg);
                                    return self.emit_expr_raw(Expression::Math {
                                        fun,
                                        arg: rt,
                                        arg1,
                                        arg2,
                                        arg3,
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                    MathFunction::InverseSqrt | MathFunction::Log2 => {
                        let needs_rt = match self.func.expressions[arg] {
                            Expression::Literal(Literal::F32(x)) => x <= 0.0 || !x.is_finite(),
                            _ => self.is_const_expr(arg),
                        };
                        if needs_rt {
                            let rt = self.wrap_const_runtime(arg);
                            return self.emit_expr_raw(Expression::Math {
                                fun,
                                arg: rt,
                                arg1,
                                arg2,
                                arg3,
                            });
                        }
                    }
                    MathFunction::Sqrt => {
                        let needs_rt = match self.func.expressions[arg] {
                            Expression::Literal(Literal::F32(x)) => x < 0.0 || !x.is_finite(),
                            _ => self.is_const_expr(arg),
                        };
                        if needs_rt {
                            let rt = self.wrap_const_runtime(arg);
                            return self.emit_expr_raw(Expression::Math {
                                fun,
                                arg: rt,
                                arg1,
                                arg2,
                                arg3,
                            });
                        }
                    }
                    MathFunction::Exp2 => {
                        let needs_rt = match self.func.expressions[arg] {
                            Expression::Literal(Literal::F32(x)) => x >= 128.0 || !x.is_finite(),
                            _ => self.is_const_expr(arg),
                        };
                        if needs_rt {
                            let rt = self.wrap_const_runtime(arg);
                            return self.emit_expr_raw(Expression::Math {
                                fun,
                                arg: rt,
                                arg1,
                                arg2,
                                arg3,
                            });
                        }
                    }
                    MathFunction::Fma => {
                        if let (Some(a1), Some(a2)) = (arg1, arg2)
                            && self.is_const_expr(arg)
                            && self.is_const_expr(a1)
                            && self.is_const_expr(a2)
                        {
                            let needs_rt = match (
                                &self.func.expressions[arg],
                                &self.func.expressions[a1],
                                &self.func.expressions[a2],
                            ) {
                                (
                                    Expression::Literal(Literal::F32(a)),
                                    Expression::Literal(Literal::F32(b)),
                                    Expression::Literal(Literal::F32(c)),
                                ) => !a.mul_add(*b, *c).is_finite(),
                                _ => true,
                            };
                            if needs_rt {
                                let rt = self.wrap_const_runtime(arg);
                                return self.emit_expr_raw(Expression::Math {
                                    fun,
                                    arg: rt,
                                    arg1,
                                    arg2,
                                    arg3,
                                });
                            }
                        }
                    }
                    _ => {}
                }
                self.emit_expr_raw(Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2,
                    arg3,
                })
            }
            other => self.emit_expr_raw(other),
        }
    }

    fn lit_f32(&mut self, value: f32) -> Handle<Expression> {
        if is_safe_wgsl_f32_literal(value) {
            self.raw_expr(Expression::Literal(Literal::F32(value)))
        } else {
            let bits = self.lit_u32(value.to_bits());
            self.call_pg6_u32_to_f32(bits)
        }
    }

    fn lit_f32_from_bits(&mut self, bits: u32) -> Handle<Expression> {
        let f = f32::from_bits(bits);
        if is_safe_wgsl_f32_literal(f) {
            self.raw_expr(Expression::Literal(Literal::F32(f)))
        } else {
            let u = self.lit_u32(bits);
            self.call_pg6_u32_to_f32(u)
        }
    }

    fn lit_u32(&mut self, value: u32) -> Handle<Expression> {
        self.raw_expr(Expression::Literal(Literal::U32(value)))
    }

    fn lit_i32(&mut self, value: i32) -> Handle<Expression> {
        self.raw_expr(Expression::Literal(Literal::I32(value)))
    }

    fn lit_bool(&mut self, value: bool) -> Handle<Expression> {
        self.raw_expr(Expression::Literal(Literal::Bool(value)))
    }

    fn bitcast_f32_to_u32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Uint,
            convert: None,
        })
    }

    fn bitcast_f32_to_i32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Sint,
            convert: None,
        })
    }

    fn bitcast_u32_to_f32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Float,
            convert: None,
        })
    }

    fn bitcast_u32_to_i32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Sint,
            convert: None,
        })
    }

    fn bitcast_i32_to_u32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Uint,
            convert: None,
        })
    }

    fn bitcast_i32_to_f32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Float,
            convert: None,
        })
    }

    fn store_local(&mut self, local: Handle<LocalVariable>, value: Handle<Expression>) {
        let ptr = self.raw_expr(Expression::LocalVariable(local));
        self.func
            .body
            .push(Statement::Store { pointer: ptr, value }, Span::UNDEFINED);
    }

    fn load_local(&mut self, local: Handle<LocalVariable>) -> Handle<Expression> {
        let ptr = self.raw_expr(Expression::LocalVariable(local));
        self.emit_expr(Expression::Load { pointer: ptr })
    }

    fn load_vgpr_f32(&mut self, reg: u8) -> Handle<Expression> {
        if let Some(&local) = self.vgpr_locals.get(&reg) {
            self.load_local(local)
        } else {
            self.lit_f32(0.0)
        }
    }

    fn store_vgpr_f32(&mut self, reg: u8, val: Handle<Expression>) {
        if let Some(&local) = self.vgpr_locals.get(&reg) {
            self.store_local(local, val);
        }
        if let Some(slots) = self.vgpr_lane_slots.get(&reg).cloned() {
            let u = self.bitcast_f32_to_u32(val);
            for (_, slot_local) in slots {
                self.store_local(slot_local, u);
            }
        }
    }

    fn store_vgpr_u32(&mut self, reg: u8, val: Handle<Expression>) {
        let f = self.bitcast_u32_to_f32(val);
        self.store_vgpr_f32(reg, f);
    }

    fn store_vgpr_i32(&mut self, reg: u8, val: Handle<Expression>) {
        let f = self.bitcast_i32_to_f32(val);
        self.store_vgpr_f32(reg, f);
    }

    /// Pack an `f32` value into the low 16 bits (`[15:0]`) of `v[reg]` as IEEE-754 `f16`
    /// using core WGSL `pack2x16float`, preserving the upper 16 bits (`[31:16]`).
    fn store_vgpr_f16_lo(&mut self, reg: u8, val_f32: Handle<Expression>) {
        let zero = self.lit_f32(0.0);
        let pair = self.emit_expr(Expression::Compose {
            ty: self.types.vec2_f32_ty,
            components: vec![val_f32, zero],
        });
        let packed = self.emit_expr(Expression::Math {
            fun: MathFunction::Pack2x16float,
            arg: pair,
            arg1: None,
            arg2: None,
            arg3: None,
        });
        let mask_lo = self.lit_u32(0x0000_FFFF);
        let new_lo = self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: packed,
            right: mask_lo,
        });
        let old_f = self.load_vgpr_f32(reg);
        let old_u = self.bitcast_f32_to_u32(old_f);
        let mask_hi = self.lit_u32(0xFFFF_0000);
        let preserved_hi = self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: old_u,
            right: mask_hi,
        });
        let combined = self.emit_expr(Expression::Binary {
            op: BinaryOperator::InclusiveOr,
            left: preserved_hi,
            right: new_lo,
        });
        self.store_vgpr_u32(reg, combined);
    }

    /// Pack an `f32` value into the high 16 bits (`[31:16]`) of `v[reg]` as IEEE-754 `f16`
    /// using core WGSL `pack2x16float`, preserving the lower 16 bits (`[15:0]`).
    fn store_vgpr_f16_hi(&mut self, reg: u8, val_f32: Handle<Expression>) {
        let zero = self.lit_f32(0.0);
        let pair = self.emit_expr(Expression::Compose {
            ty: self.types.vec2_f32_ty,
            components: vec![zero, val_f32],
        });
        let packed = self.emit_expr(Expression::Math {
            fun: MathFunction::Pack2x16float,
            arg: pair,
            arg1: None,
            arg2: None,
            arg3: None,
        });
        let mask_hi = self.lit_u32(0xFFFF_0000);
        let new_hi = self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: packed,
            right: mask_hi,
        });
        let old_f = self.load_vgpr_f32(reg);
        let old_u = self.bitcast_f32_to_u32(old_f);
        let mask_lo = self.lit_u32(0x0000_FFFF);
        let preserved_lo = self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: old_u,
            right: mask_lo,
        });
        let combined = self.emit_expr(Expression::Binary {
            op: BinaryOperator::InclusiveOr,
            left: preserved_lo,
            right: new_hi,
        });
        self.store_vgpr_u32(reg, combined);
    }

    fn load_sgpr_u32(&mut self, reg: u8) -> Handle<Expression> {
        if let Some(&local) = self.sgpr_locals.get(&reg) {
            self.load_local(local)
        } else {
            self.lit_u32(0)
        }
    }

    fn store_sgpr_u32(&mut self, reg: u8, val: Handle<Expression>) {
        if let Some(&local) = self.sgpr_locals.get(&reg) {
            self.store_local(local, val);
        } else if reg == 126 {
            // `exec_lo` destination: keep this lane's bit.
            self.store_exec_lane_bit(val);
        }
    }

    /// Load the single-lane `exec_mask` (`1u` when the model is inactive: all lanes on).
    fn load_exec(&mut self) -> Handle<Expression> {
        if let Some(local) = self.exec_local {
            self.load_local(local)
        } else {
            self.lit_u32(1)
        }
    }

    /// `exec_mask = (val & 1u)` (no-op when the shader never needs `exec_mask`).
    fn store_exec_lane_bit(&mut self, val: Handle<Expression>) {
        if let Some(local) = self.exec_local {
            let one = self.lit_u32(1);
            let bit = self.emit_bin(BinaryOperator::And, val, one);
            self.store_local(local, bit);
        }
    }

    fn load_vcc(&mut self) -> Handle<Expression> {
        if let Some(local) = self.vcc_local {
            self.load_local(local)
        } else {
            self.lit_bool(false)
        }
    }

    fn store_vcc(&mut self, val: Handle<Expression>) {
        if let Some(local) = self.vcc_local {
            self.store_local(local, val);
        }
    }

    fn load_scc(&mut self) -> Handle<Expression> {
        if let Some(local) = self.scc_local {
            self.load_local(local)
        } else {
            self.lit_bool(false)
        }
    }

    fn store_scc(&mut self, val: Handle<Expression>) {
        if let Some(local) = self.scc_local {
            self.store_local(local, val);
        }
    }

    fn emit_prologue(&mut self) {
        // Zero-initialize all declared VGPRs, SGPRs, VCC, SCC, and stage output structs.
        let vgpr_list: Vec<(u8, Handle<LocalVariable>)> =
            self.vgpr_locals.iter().map(|(&k, &v)| (k, v)).collect();
        for (_, local) in &vgpr_list {
            let z = self.lit_f32(0.0);
            self.store_local(*local, z);
        }
        let sgpr_list: Vec<(u8, Handle<LocalVariable>)> =
            self.sgpr_locals.iter().map(|(&k, &v)| (k, v)).collect();
        for (_, local) in &sgpr_list {
            let z = self.lit_u32(0);
            self.store_local(*local, z);
        }
        let lane_list: Vec<Handle<LocalVariable>> =
            self.lane_spill_locals.values().copied().collect();
        for local in lane_list {
            let z = self.lit_u32(0);
            self.store_local(local, z);
        }
        if let Some(vcc) = self.vcc_local {
            let f = self.lit_bool(false);
            self.store_local(vcc, f);
        }
        if let Some(scc) = self.scc_local {
            let f = self.lit_bool(false);
            self.store_local(scc, f);
        }
        if let Some(exec) = self.exec_local {
            // One WGSL invocation models one active wave lane.
            let one = self.lit_u32(1);
            self.store_local(exec, one);
        }

        // Stage-specific prologue bindings into VGPRs & default output struct values.
        match self.stage {
            ShaderStage::Vertex => {
                if let Some(out_local) = self.vs_out_local {
                    let out_ty = self.func.local_variables[out_local].ty;
                    let zero_struct = self.raw_expr(Expression::ZeroValue(out_ty));
                    self.store_local(out_local, zero_struct);
                    // Default position = vec4<f32>(0.0, 0.0, 0.0, 1.0)
                    let c0 = self.lit_f32(0.0);
                    let c1 = self.lit_f32(0.0);
                    let c2 = self.lit_f32(0.0);
                    let c3 = self.lit_f32(1.0);
                    let default_pos = self.emit_expr(Expression::Compose {
                        ty: self.types.vec4_f32_ty,
                        components: vec![c0, c1, c2, c3],
                    });
                    let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                    let pos_ptr = self.emit_expr(Expression::AccessIndex {
                        base: base_ptr,
                        index: 0,
                    });
                    self.func.body.push(
                        Statement::Store {
                            pointer: pos_ptr,
                            value: default_pos,
                        },
                        Span::UNDEFINED,
                    );
                }

                if let Some(arg_idx) = self.vs_in_arg {
                    match self.vertex_input_mode {
                        VertexInputMode::VertexIndex => {
                            let vi = self.raw_expr(Expression::FunctionArgument(arg_idx));
                            self.store_vgpr_u32(0, vi);
                        }
                        VertexInputMode::Attributes => {
                            let vs_in = self.raw_expr(Expression::FunctionArgument(arg_idx));
                            if self.has_buffer_fetch {
                                // Provide SV_VertexID in v0 prior to buffer attribute fetches.
                                if self.vgpr_locals.contains_key(&0) {
                                    let vi = self.emit_expr(Expression::AccessIndex {
                                        base: vs_in,
                                        index: 0,
                                    });
                                    self.store_vgpr_u32(0, vi);
                                }
                            } else {
                                // Legacy multi-VGPR attribute input without MUBUF/MTBUF fetches.
                                let pos = self.emit_expr(Expression::AccessIndex {
                                    base: vs_in,
                                    index: 1,
                                });
                                for c in 0..4u32 {
                                    let reg = c as u8;
                                    if self.vgpr_locals.contains_key(&reg) {
                                        let comp = self.emit_expr(Expression::AccessIndex {
                                            base: pos,
                                            index: c,
                                        });
                                        self.store_vgpr_f32(reg, comp);
                                    }
                                }
                            }
                            if let Some(uv_member_idx) = self.legacy_uv_member_index {
                                let uv = self.emit_expr(Expression::AccessIndex {
                                    base: vs_in,
                                    index: uv_member_idx,
                                });
                                for c in 0..2u32 {
                                    let reg = 4 + c as u8;
                                    if self.vgpr_locals.contains_key(&reg) {
                                        let comp = self.emit_expr(Expression::AccessIndex {
                                            base: uv,
                                            index: c,
                                        });
                                        self.store_vgpr_f32(reg, comp);
                                    }
                                }
                            }
                        }
                        VertexInputMode::None => {}
                    }
                }
            }
            ShaderStage::Fragment => {
                if let Some(out_local) = self.ps_out_local {
                    let out_ty = self.func.local_variables[out_local].ty;
                    let zero_struct = self.raw_expr(Expression::ZeroValue(out_ty));
                    self.store_local(out_local, zero_struct);
                }
            }
            ShaderStage::Compute => {
                // Seed GFX10.3 compute thread ID VGPRs: v0 = local_id.x, v1 = local_id.y, v2 = local_id.z.
                // (For 1D workgroups `(X, 1, 1)`, `local_id.x` is identical to the packed 1D thread ID.)
                let local_id = self.raw_expr(Expression::FunctionArgument(0));
                for c in 0..3u32 {
                    let reg = c as u8;
                    if self.vgpr_locals.contains_key(&reg) {
                        let comp = self.emit_expr(Expression::AccessIndex {
                            base: local_id,
                            index: c,
                        });
                        self.store_vgpr_u32(reg, comp);
                    }
                }
                // Seed compute workgroup ID SGPRs (`s0`/`s1`/`s2` or configured base) when not overwritten.
                if !self.compute_seed_wg_sgprs.is_empty() {
                    let wg_id = self.raw_expr(Expression::FunctionArgument(1));
                    let seed_list = self.compute_seed_wg_sgprs.clone();
                    for (reg, comp_idx) in seed_list {
                        if self.sgpr_locals.contains_key(&reg) {
                            let comp = self.emit_expr(Expression::AccessIndex {
                                base: wg_id,
                                index: comp_idx,
                            });
                            self.store_sgpr_u32(reg, comp);
                        }
                    }
                }
            }
        }
    }

    // -- Operand readers --

    fn inline_float_value(src: u16) -> Option<f32> {
        match src {
            240 => Some(0.5),
            241 => Some(-0.5),
            242 => Some(1.0),
            243 => Some(-1.0),
            244 => Some(2.0),
            245 => Some(-2.0),
            246 => Some(4.0),
            247 => Some(-4.0),
            248 => Some(0.159_154_94),
            _ => None,
        }
    }

    /// Exact IEEE-754 binary16 bit pattern for GFX10.3 inline float constants (`240..=248`).
    fn inline_f16_bits(src: u16) -> Option<u32> {
        match src {
            240 => Some(0x3800), // 0.5
            241 => Some(0xB800), // -0.5
            242 => Some(0x3C00), // 1.0
            243 => Some(0xBC00), // -1.0
            244 => Some(0x4000), // 2.0
            245 => Some(0xC000), // -2.0
            246 => Some(0x4400), // 4.0
            247 => Some(0xC400), // -4.0
            248 => Some(0x3118), // 1/(2*pi)
            _ => None,
        }
    }

    fn read_src_u32(&mut self, src: u16, literal: Option<u32>) -> Handle<Expression> {
        match src {
            0..=105 => self.load_sgpr_u32(src as u8),
            106 => {
                let cond = self.load_vcc();
                let t = self.lit_u32(1);
                let f = self.lit_u32(0);
                self.emit_expr(Expression::Select {
                    condition: cond,
                    accept: t,
                    reject: f,
                })
            }
            126 if self.exec_local.is_some() => self.load_exec(),
            127 if self.exec_local.is_some() => self.lit_u32(0),
            126 | 127 => self.lit_u32(0xFFFF_FFFF),
            128..=192 => self.lit_u32(u32::from(src - 128)),
            193..=208 => {
                let neg = -(i32::from(src) - 192);
                self.lit_u32(neg as u32)
            }
            240..=248 => {
                let f = Self::inline_float_value(src).unwrap_or(0.0);
                self.lit_u32(f.to_bits())
            }
            251 => {
                let cond = self.load_vcc();
                let t = self.lit_u32(0);
                let f = self.lit_u32(1);
                self.emit_expr(Expression::Select {
                    condition: cond,
                    accept: t,
                    reject: f,
                })
            }
            253 => {
                let cond = self.load_scc();
                let t = self.lit_u32(1);
                let f = self.lit_u32(0);
                self.emit_expr(Expression::Select {
                    condition: cond,
                    accept: t,
                    reject: f,
                })
            }
            255 => self.lit_u32(literal.unwrap_or(0)),
            256..=511 => {
                let vf = self.load_vgpr_f32((src - 256) as u8);
                self.bitcast_f32_to_u32(vf)
            }
            _ => self.lit_u32(0),
        }
    }

    fn read_src_i32(&mut self, src: u16, literal: Option<u32>) -> Handle<Expression> {
        match src {
            128..=192 => self.lit_i32(i32::from(src - 128)),
            193..=208 => self.lit_i32(-(i32::from(src) - 192)),
            255 => self.lit_i32(literal.unwrap_or(0) as i32),
            _ => {
                let u = self.read_src_u32(src, literal);
                self.bitcast_u32_to_i32(u)
            }
        }
    }

    fn read_src_f32(
        &mut self,
        src: u16,
        literal: Option<u32>,
        abs_mod: bool,
        neg_mod: bool,
    ) -> Handle<Expression> {
        let mut val = match src {
            0..=105 => {
                let u = self.load_sgpr_u32(src as u8);
                self.bitcast_u32_to_f32(u)
            }
            128 => self.lit_f32(0.0),
            129..=208 => {
                let u = self.read_src_u32(src, literal);
                self.bitcast_u32_to_f32(u)
            }
            240..=248 => self.lit_f32(Self::inline_float_value(src).unwrap_or(0.0)),
            255 => self.lit_f32_from_bits(literal.unwrap_or(0)),
            256..=511 => self.load_vgpr_f32((src - 256) as u8),
            _ => {
                let u = self.read_src_u32(src, literal);
                self.bitcast_u32_to_f32(u)
            }
        };

        if abs_mod {
            val = self.emit_expr(Expression::Math {
                fun: MathFunction::Abs,
                arg: val,
                arg1: None,
                arg2: None,
                arg3: None,
            });
        }
        if neg_mod {
            val = self.emit_expr(Expression::Unary {
                op: UnaryOperator::Negate,
                expr: val,
            });
        }
        val
    }

    /// Read an `f16` (low or high 16-bit lane via `unpack2x16float`) or `f32` operand as `f32`,
    /// applying optional `abs` and `neg` modifiers (used by `v_fma_mix*` and scalar `*_f16` ops).
    fn read_src_mixed_f16_or_f32(
        &mut self,
        src: u16,
        literal: Option<u32>,
        is_f16: bool,
        hi_half: bool,
        abs_mod: bool,
        neg_mod: bool,
    ) -> Handle<Expression> {
        let mut val = if !is_f16 {
            self.read_src_f32(src, literal, false, false)
        } else {
            match src {
                128 => self.lit_f32(0.0),
                240..=248 => self.lit_f32(Self::inline_float_value(src).unwrap_or(0.0)),
                _ => {
                    let u = self.read_src_u32(src, literal);
                    let pair = self.emit_expr(Expression::Math {
                        fun: MathFunction::Unpack2x16float,
                        arg: u,
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    });
                    self.emit_expr(Expression::AccessIndex {
                        base: pair,
                        index: if hi_half { 1 } else { 0 },
                    })
                }
            }
        };
        if abs_mod {
            val = self.emit_expr(Expression::Math {
                fun: MathFunction::Abs,
                arg: val,
                arg1: None,
                arg2: None,
                arg3: None,
            });
        }
        if neg_mod {
            val = self.emit_expr(Expression::Unary {
                op: UnaryOperator::Negate,
                expr: val,
            });
        }
        val
    }

    /// Unpack a 16-bit IEEE-754 `f16` literal stored in the low 16 bits of a 32-bit word into `f32`.
    fn lit_f16_from_u32_lo(&mut self, bits: u32) -> Handle<Expression> {
        let u = self.lit_u32(bits & 0xFFFF);
        let pair = self.emit_expr(Expression::Math {
            fun: MathFunction::Unpack2x16float,
            arg: u,
            arg1: None,
            arg2: None,
            arg3: None,
        });
        self.emit_expr(Expression::AccessIndex {
            base: pair,
            index: 0,
        })
    }

    /// Read a packed 2x16-bit `f16` VOP3P operand as `vec2<f32>` via core WGSL `unpack2x16float`
    /// (zero `enable f16;` required), applying exact GFX10.3 `op_sel`, `op_sel_hi`, `neg_lo`,
    /// and `neg_hi` modifiers.
    fn read_src_pk_f16(
        &mut self,
        src: u16,
        literal: Option<u32>,
        op_sel_lo: bool,
        op_sel_hi: bool,
        neg_lo: bool,
        neg_hi: bool,
    ) -> Handle<Expression> {
        let raw = match src {
            128 => {
                let z = self.lit_f32(0.0);
                self.emit_expr(Expression::Splat {
                    size: VectorSize::Bi,
                    value: z,
                })
            }
            240..=248 => {
                // GFX10.3 inline float constants populate the low 16-bit f16 half (0.0 in hi).
                let lo = self.lit_f32(Self::inline_float_value(src).unwrap_or(0.0));
                let hi = self.lit_f32(0.0);
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![lo, hi],
                })
            }
            _ => {
                let u = self.read_src_u32(src, literal);
                self.emit_expr(Expression::Math {
                    fun: MathFunction::Unpack2x16float,
                    arg: u,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            }
        };

        let selected = match (op_sel_lo, op_sel_hi) {
            (false, true) => raw,
            (false, false) => self.emit_expr(Expression::Swizzle {
                size: VectorSize::Bi,
                vector: raw,
                pattern: [
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                ],
            }),
            (true, true) => self.emit_expr(Expression::Swizzle {
                size: VectorSize::Bi,
                vector: raw,
                pattern: [
                    SwizzleComponent::Y,
                    SwizzleComponent::Y,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                ],
            }),
            (true, false) => self.emit_expr(Expression::Swizzle {
                size: VectorSize::Bi,
                vector: raw,
                pattern: [
                    SwizzleComponent::Y,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                ],
            }),
        };

        match (neg_lo, neg_hi) {
            (false, false) => selected,
            (true, true) => self.emit_expr(Expression::Unary {
                op: UnaryOperator::Negate,
                expr: selected,
            }),
            (true, false) => {
                let x = self.emit_expr(Expression::AccessIndex {
                    base: selected,
                    index: 0,
                });
                let nx = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: x,
                });
                let y = self.emit_expr(Expression::AccessIndex {
                    base: selected,
                    index: 1,
                });
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![nx, y],
                })
            }
            (false, true) => {
                let x = self.emit_expr(Expression::AccessIndex {
                    base: selected,
                    index: 0,
                });
                let y = self.emit_expr(Expression::AccessIndex {
                    base: selected,
                    index: 1,
                });
                let ny = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: y,
                });
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![x, ny],
                })
            }
        }
    }

    fn update_scc_nonzero(&mut self, val_u32: Handle<Expression>) {
        if self.scc_local.is_some() {
            let zero = self.lit_u32(0);
            let nonzero = self.emit_expr(Expression::Binary {
                op: BinaryOperator::NotEqual,
                left: val_u32,
                right: zero,
            });
            self.store_scc(nonzero);
        }
    }

    fn emit_bin(
        &mut self,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit_expr(Expression::Binary { op, left, right })
    }

    fn emit_select(
        &mut self,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit_expr(Expression::Select {
            condition,
            accept,
            reject,
        })
    }

    fn emit_bit_not(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::Unary {
            op: UnaryOperator::BitwiseNot,
            expr,
        })
    }

    /// Canonical lane-mask SGPR value for a per-lane boolean (`1u` / `0u`).
    fn bool_to_u32(&mut self, cond: Handle<Expression>) -> Handle<Expression> {
        let t = self.lit_u32(1);
        let f = self.lit_u32(0);
        self.emit_select(cond, t, f)
    }

    /// Store a per-lane boolean into a 64-bit lane-mask destination (`vcc` or `s[n:n+1]`).
    ///
    /// One WGSL invocation models one wave lane, so SGPR lane masks are canonically
    /// `(lo = lane bit, hi = 0)`; `exec` / `null` destinations are no-ops.
    fn store_lane_mask(&mut self, sdst: u8, cond: Handle<Expression>) {
        let sd7 = sdst & 0x7F;
        if sd7 == 106 {
            self.store_vcc(cond);
        } else if sd7 <= 105 {
            let mask_u = self.bool_to_u32(cond);
            self.store_sgpr_u32(sd7, mask_u);
            if sd7 <= 104 {
                let zero = self.lit_u32(0);
                self.store_sgpr_u32(sd7 + 1, zero);
            }
        }
    }

    /// Read this lane's bit (bit 0 of the low dword) of a 64-bit lane-mask source.
    fn read_lane_mask_bit(&mut self, src: u16, literal: Option<u32>) -> Handle<Expression> {
        match src {
            106 => self.load_vcc(),
            126 if self.exec_local.is_none() => self.lit_bool(true),
            _ => {
                let v = self.read_src_u32(src, literal);
                let one = self.lit_u32(1);
                let bit = self.emit_bin(BinaryOperator::And, v, one);
                let zero = self.lit_u32(0);
                self.emit_bin(BinaryOperator::NotEqual, bit, zero)
            }
        }
    }

    /// Read a 64-bit scalar source operand as `(lo, hi)` u32 halves.
    ///
    /// SGPR pairs read both registers, `vcc` yields `(lane bit, 0)`, `exec` is all ones, integer
    /// inline constants are sign-extended, float inline constants use their `f64` bit pattern,
    /// and 32-bit literals are zero-extended.
    fn read_ssrc_pair(
        &mut self,
        src: u8,
        literal: Option<u32>,
    ) -> (Handle<Expression>, Handle<Expression>) {
        match src {
            0..=104 => (self.load_sgpr_u32(src), self.load_sgpr_u32(src + 1)),
            105 => (self.load_sgpr_u32(105), self.lit_u32(0)),
            106 => (self.read_src_u32(106, None), self.lit_u32(0)),
            126 if self.exec_local.is_some() => (self.load_exec(), self.lit_u32(0)),
            126 => (self.lit_u32(0xFFFF_FFFF), self.lit_u32(0xFFFF_FFFF)),
            128..=192 => (self.lit_u32(u32::from(src - 128)), self.lit_u32(0)),
            193..=208 => {
                let neg = -(i32::from(src) - 192);
                (self.lit_u32(neg as u32), self.lit_u32(0xFFFF_FFFF))
            }
            240..=248 => {
                let f = f64::from(Self::inline_float_value(u16::from(src)).unwrap_or(0.0));
                let bits = f.to_bits();
                (
                    self.lit_u32((bits & 0xFFFF_FFFF) as u32),
                    self.lit_u32((bits >> 32) as u32),
                )
            }
            255 => (self.lit_u32(literal.unwrap_or(0)), self.lit_u32(0)),
            _ => (self.read_src_u32(u16::from(src), literal), self.lit_u32(0)),
        }
    }

    /// Read a 64-bit vector-ALU source (`v[n:n+1]`, SGPR pair, or constant) as `(lo, hi)`.
    fn read_vsrc_pair(
        &mut self,
        src: u16,
        literal: Option<u32>,
    ) -> (Handle<Expression>, Handle<Expression>) {
        match src {
            0..=255 => self.read_ssrc_pair(src as u8, literal),
            256..=510 => {
                let lo = self.read_src_u32(src, None);
                let hi = self.read_src_u32(src + 1, None);
                (lo, hi)
            }
            _ => (self.read_src_u32(src, literal), self.lit_u32(0)),
        }
    }

    /// Store a 64-bit `(lo, hi)` scalar result into an SGPR pair destination.
    ///
    /// `vcc` receives this lane's bit (bit 0 of `lo`); `exec` / `null` are no-ops.
    fn store_sdst_pair(&mut self, sdst: u8, lo: Handle<Expression>, hi: Handle<Expression>) {
        match sdst {
            0..=104 => {
                self.store_sgpr_u32(sdst, lo);
                self.store_sgpr_u32(sdst + 1, hi);
            }
            105 => self.store_sgpr_u32(105, lo),
            106 => {
                let one = self.lit_u32(1);
                let bit = self.emit_bin(BinaryOperator::And, lo, one);
                let zero = self.lit_u32(0);
                let cond = self.emit_bin(BinaryOperator::NotEqual, bit, zero);
                self.store_vcc(cond);
            }
            126 => self.store_exec_lane_bit(lo),
            _ => {}
        }
    }

    /// `scc = ((lo | hi) != 0u)` for 64-bit scalar results.
    fn update_scc_nonzero_pair(&mut self, lo: Handle<Expression>, hi: Handle<Expression>) {
        if self.scc_local.is_some() {
            let any = self.emit_bin(BinaryOperator::InclusiveOr, lo, hi);
            self.update_scc_nonzero(any);
        }
    }

    /// Exact unsigned `32 x 32 -> hi32` multiply using only `u32` 16-bit half-word products.
    fn emit_mul_hi_u32(
        &mut self,
        a: Handle<Expression>,
        b: Handle<Expression>,
    ) -> Handle<Expression> {
        let m16 = self.lit_u32(0xFFFF);
        let s16 = self.lit_u32(16);
        let a_lo = self.emit_bin(BinaryOperator::And, a, m16);
        let a_hi = self.emit_bin(BinaryOperator::ShiftRight, a, s16);
        let b_lo = self.emit_bin(BinaryOperator::And, b, m16);
        let b_hi = self.emit_bin(BinaryOperator::ShiftRight, b, s16);
        let ll = self.emit_bin(BinaryOperator::Multiply, a_lo, b_lo);
        let lh = self.emit_bin(BinaryOperator::Multiply, a_lo, b_hi);
        let hl = self.emit_bin(BinaryOperator::Multiply, a_hi, b_lo);
        let hh = self.emit_bin(BinaryOperator::Multiply, a_hi, b_hi);
        // mid = (ll >> 16) + lo16(lh) + lo16(hl)  (<= 3 * 0xFFFF, never overflows)
        let ll_hi = self.emit_bin(BinaryOperator::ShiftRight, ll, s16);
        let lh_lo = self.emit_bin(BinaryOperator::And, lh, m16);
        let hl_lo = self.emit_bin(BinaryOperator::And, hl, m16);
        let mid0 = self.emit_bin(BinaryOperator::Add, ll_hi, lh_lo);
        let mid = self.emit_bin(BinaryOperator::Add, mid0, hl_lo);
        // hi = hh + (lh >> 16) + (hl >> 16) + (mid >> 16)
        let lh_hi = self.emit_bin(BinaryOperator::ShiftRight, lh, s16);
        let hl_hi = self.emit_bin(BinaryOperator::ShiftRight, hl, s16);
        let mid_hi = self.emit_bin(BinaryOperator::ShiftRight, mid, s16);
        let r0 = self.emit_bin(BinaryOperator::Add, hh, lh_hi);
        let r1 = self.emit_bin(BinaryOperator::Add, r0, hl_hi);
        self.emit_bin(BinaryOperator::Add, r1, mid_hi)
    }

    /// Exact signed `32 x 32 -> hi32` multiply on `u32` bit patterns:
    /// `mulhi_s(a, b) = mulhi_u(a, b) - (a < 0 ? b : 0) - (b < 0 ? a : 0)` (mod 2^32).
    fn emit_mul_hi_i32(
        &mut self,
        a: Handle<Expression>,
        b: Handle<Expression>,
    ) -> Handle<Expression> {
        let hu = self.emit_mul_hi_u32(a, b);
        let sign = self.lit_u32(0x8000_0000);
        let zero = self.lit_u32(0);
        let a_neg = self.emit_bin(BinaryOperator::GreaterEqual, a, sign);
        let b_neg = self.emit_bin(BinaryOperator::GreaterEqual, b, sign);
        let corr_a = self.emit_select(a_neg, b, zero);
        let corr_b = self.emit_select(b_neg, a, zero);
        let r0 = self.emit_bin(BinaryOperator::Subtract, hu, corr_a);
        self.emit_bin(BinaryOperator::Subtract, r0, corr_b)
    }

    /// 64-bit shift of a `(lo, hi)` u32 pair by `amount & 63`, using only masked (`< 32`)
    /// WGSL shift amounts and `select` for the `>= 32` half-swap.
    fn emit_shift64(
        &mut self,
        kind: Shift64Kind,
        lo: Handle<Expression>,
        hi: Handle<Expression>,
        amount: Handle<Expression>,
    ) -> (Handle<Expression>, Handle<Expression>) {
        let mask63 = self.lit_u32(63);
        let mask31 = self.lit_u32(31);
        let c32 = self.lit_u32(32);
        let zero = self.lit_u32(0);
        let amt = self.emit_bin(BinaryOperator::And, amount, mask63);
        let s5 = self.emit_bin(BinaryOperator::And, amt, mask31);
        let big = self.emit_bin(BinaryOperator::GreaterEqual, amt, c32);
        let s5_nonzero = self.emit_bin(BinaryOperator::NotEqual, s5, zero);
        let inv_raw = self.emit_bin(BinaryOperator::Subtract, c32, s5);
        let inv = self.emit_bin(BinaryOperator::And, inv_raw, mask31);
        let (lo_small, hi_small, lo_big, hi_big) = match kind {
            Shift64Kind::Left => {
                let lo_sh = self.emit_bin(BinaryOperator::ShiftLeft, lo, s5);
                let hi_sh = self.emit_bin(BinaryOperator::ShiftLeft, hi, s5);
                let spill_raw = self.emit_bin(BinaryOperator::ShiftRight, lo, inv);
                let spill = self.emit_select(s5_nonzero, spill_raw, zero);
                let hi_small = self.emit_bin(BinaryOperator::InclusiveOr, hi_sh, spill);
                (lo_sh, hi_small, zero, lo_sh)
            }
            Shift64Kind::LogicalRight | Shift64Kind::ArithmeticRight => {
                let lo_sh = self.emit_bin(BinaryOperator::ShiftRight, lo, s5);
                let spill_raw = self.emit_bin(BinaryOperator::ShiftLeft, hi, inv);
                let spill = self.emit_select(s5_nonzero, spill_raw, zero);
                let lo_small = self.emit_bin(BinaryOperator::InclusiveOr, lo_sh, spill);
                if kind == Shift64Kind::LogicalRight {
                    let hi_sh = self.emit_bin(BinaryOperator::ShiftRight, hi, s5);
                    (lo_small, hi_sh, hi_sh, zero)
                } else {
                    let hi_i = self.bitcast_u32_to_i32(hi);
                    let hi_sar_i = self.emit_bin(BinaryOperator::ShiftRight, hi_i, s5);
                    let hi_sar = self.bitcast_i32_to_u32(hi_sar_i);
                    let c31 = self.lit_u32(31);
                    let sign_i = self.emit_bin(BinaryOperator::ShiftRight, hi_i, c31);
                    let sign = self.bitcast_i32_to_u32(sign_i);
                    (lo_small, hi_sar, hi_sar, sign)
                }
            }
        };
        let out_lo = self.emit_select(big, lo_big, lo_small);
        let out_hi = self.emit_select(big, hi_big, hi_small);
        (out_lo, out_hi)
    }

    /// One 32-bit half of a SOP2 `*_b64` bitwise op (`and`/`or`/`xor`/`andn2`/`orn2`/`nand`/`nor`/`xnor`).
    fn emit_sop2_bitwise_half(
        &mut self,
        op: u8,
        l: Handle<Expression>,
        r: Handle<Expression>,
    ) -> Handle<Expression> {
        let (bin, invert_rhs, invert_result) = match op {
            0x0E | 0x0F => (BinaryOperator::And, false, false),
            0x10 | 0x11 => (BinaryOperator::InclusiveOr, false, false),
            0x12 | 0x13 => (BinaryOperator::ExclusiveOr, false, false),
            0x14 | 0x15 => (BinaryOperator::And, true, false),
            0x16 | 0x17 => (BinaryOperator::InclusiveOr, true, false),
            0x18 | 0x19 => (BinaryOperator::And, false, true),
            0x1A | 0x1B => (BinaryOperator::InclusiveOr, false, true),
            _ => (BinaryOperator::ExclusiveOr, false, true),
        };
        let rhs = if invert_rhs { self.emit_bit_not(r) } else { r };
        let res = self.emit_bin(bin, l, rhs);
        if invert_result {
            self.emit_bit_not(res)
        } else {
            res
        }
    }

    /// IEEE-754 binary32 class test for `v_cmp_class_f32`: returns `true` when the class of
    /// `bits` has its bit set in `mask` (0 sNaN, 1 qNaN, 2 -Inf, 3 -Normal, 4 -Subnormal,
    /// 5 -0, 6 +0, 7 +Subnormal, 8 +Normal, 9 +Inf).
    fn emit_f32_class_test(
        &mut self,
        bits: Handle<Expression>,
        mask: Handle<Expression>,
    ) -> Handle<Expression> {
        let s23 = self.lit_u32(23);
        let s31 = self.lit_u32(31);
        let exp_mask = self.lit_u32(0xFF);
        let mant_mask = self.lit_u32(0x007F_FFFF);
        let quiet_bit = self.lit_u32(0x0040_0000);
        let zero = self.lit_u32(0);
        let one = self.lit_u32(1);
        let exp_sh = self.emit_bin(BinaryOperator::ShiftRight, bits, s23);
        let exp = self.emit_bin(BinaryOperator::And, exp_sh, exp_mask);
        let mant = self.emit_bin(BinaryOperator::And, bits, mant_mask);
        let sign = self.emit_bin(BinaryOperator::ShiftRight, bits, s31);
        let neg = self.emit_bin(BinaryOperator::NotEqual, sign, zero);
        let exp_max = self.emit_bin(BinaryOperator::Equal, exp, exp_mask);
        let exp_zero = self.emit_bin(BinaryOperator::Equal, exp, zero);
        let mant_zero = self.emit_bin(BinaryOperator::Equal, mant, zero);
        let quiet_raw = self.emit_bin(BinaryOperator::And, bits, quiet_bit);
        let quiet = self.emit_bin(BinaryOperator::NotEqual, quiet_raw, zero);

        let idx_snan = self.lit_u32(0);
        let idx_qnan = self.lit_u32(1);
        let idx_ninf = self.lit_u32(2);
        let idx_nnorm = self.lit_u32(3);
        let idx_nsub = self.lit_u32(4);
        let idx_nzero = self.lit_u32(5);
        let idx_pzero = self.lit_u32(6);
        let idx_psub = self.lit_u32(7);
        let idx_pnorm = self.lit_u32(8);
        let idx_pinf = self.lit_u32(9);

        let nan_idx = self.emit_select(quiet, idx_qnan, idx_snan);
        let inf_idx = self.emit_select(neg, idx_ninf, idx_pinf);
        let zero_idx = self.emit_select(neg, idx_nzero, idx_pzero);
        let sub_idx = self.emit_select(neg, idx_nsub, idx_psub);
        let norm_idx = self.emit_select(neg, idx_nnorm, idx_pnorm);
        let special_idx = self.emit_select(mant_zero, inf_idx, nan_idx);
        let tiny_idx = self.emit_select(mant_zero, zero_idx, sub_idx);
        let finite_idx = self.emit_select(exp_zero, tiny_idx, norm_idx);
        let idx = self.emit_select(exp_max, special_idx, finite_idx);

        let shifted = self.emit_bin(BinaryOperator::ShiftRight, mask, idx);
        let bit = self.emit_bin(BinaryOperator::And, shifted, one);
        self.emit_bin(BinaryOperator::NotEqual, bit, zero)
    }

    /// IEEE-754 binary16 class test for `v_cmp_class_f16`: returns `true` when the class of
    /// `bits16` (low 16 bits) has its bit set in `mask` (0 sNaN, 1 qNaN, 2 -Inf, 3 -Normal,
    /// 4 -Subnormal, 5 -0, 6 +0, 7 +Subnormal, 8 +Normal, 9 +Inf).
    fn emit_f16_class_test(
        &mut self,
        bits16: Handle<Expression>,
        mask: Handle<Expression>,
    ) -> Handle<Expression> {
        let s10 = self.lit_u32(10);
        let s15 = self.lit_u32(15);
        let exp_mask = self.lit_u32(0x1F);
        let mant_mask = self.lit_u32(0x03FF);
        let quiet_bit = self.lit_u32(0x0200);
        let zero = self.lit_u32(0);
        let one = self.lit_u32(1);
        let exp_sh = self.emit_bin(BinaryOperator::ShiftRight, bits16, s10);
        let exp = self.emit_bin(BinaryOperator::And, exp_sh, exp_mask);
        let mant = self.emit_bin(BinaryOperator::And, bits16, mant_mask);
        let sign_sh = self.emit_bin(BinaryOperator::ShiftRight, bits16, s15);
        let sign = self.emit_bin(BinaryOperator::And, sign_sh, one);
        let neg = self.emit_bin(BinaryOperator::NotEqual, sign, zero);
        let exp_max = self.emit_bin(BinaryOperator::Equal, exp, exp_mask);
        let exp_zero = self.emit_bin(BinaryOperator::Equal, exp, zero);
        let mant_zero = self.emit_bin(BinaryOperator::Equal, mant, zero);
        let quiet_raw = self.emit_bin(BinaryOperator::And, bits16, quiet_bit);
        let quiet = self.emit_bin(BinaryOperator::NotEqual, quiet_raw, zero);

        let idx_snan = self.lit_u32(0);
        let idx_qnan = self.lit_u32(1);
        let idx_ninf = self.lit_u32(2);
        let idx_nnorm = self.lit_u32(3);
        let idx_nsub = self.lit_u32(4);
        let idx_nzero = self.lit_u32(5);
        let idx_pzero = self.lit_u32(6);
        let idx_psub = self.lit_u32(7);
        let idx_pnorm = self.lit_u32(8);
        let idx_pinf = self.lit_u32(9);

        let nan_idx = self.emit_select(quiet, idx_qnan, idx_snan);
        let inf_idx = self.emit_select(neg, idx_ninf, idx_pinf);
        let zero_idx = self.emit_select(neg, idx_nzero, idx_pzero);
        let sub_idx = self.emit_select(neg, idx_nsub, idx_psub);
        let norm_idx = self.emit_select(neg, idx_nnorm, idx_pnorm);
        let special_idx = self.emit_select(mant_zero, inf_idx, nan_idx);
        let tiny_idx = self.emit_select(mant_zero, zero_idx, sub_idx);
        let finite_idx = self.emit_select(exp_zero, tiny_idx, norm_idx);
        let idx = self.emit_select(exp_max, special_idx, finite_idx);

        let shifted = self.emit_bin(BinaryOperator::ShiftRight, mask, idx);
        let bit = self.emit_bin(BinaryOperator::And, shifted, one);
        self.emit_bin(BinaryOperator::NotEqual, bit, zero)
    }

    fn apply_float_out_mods(
        &mut self,
        mut val: Handle<Expression>,
        omod: u8,
        clamp: bool,
    ) -> Handle<Expression> {
        let scale = match omod {
            1 => Some(2.0f32),
            2 => Some(4.0f32),
            3 => Some(0.5f32),
            _ => None,
        };
        if let Some(s) = scale {
            let factor = self.lit_f32(s);
            val = self.emit_expr(Expression::Binary {
                op: BinaryOperator::Multiply,
                left: val,
                right: factor,
            });
        }
        if clamp {
            let lo = self.lit_f32(0.0);
            let hi = self.lit_f32(1.0);
            val = self.emit_expr(Expression::Math {
                fun: MathFunction::Clamp,
                arg: val,
                arg1: Some(lo),
                arg2: Some(hi),
                arg3: None,
            });
        }
        val
    }

    fn record_unsupported(&mut self, inst: &Rdna2Instruction) {
        self.unsupported_op_count += 1;
        // Strictly sanitized: instruction offset, encoding family, and mnemonic only.
        let suffix = if inst.truncated { " (truncated)" } else { "" };
        self.unsupported_comments.push(format!(
            "// unsupported_op [0x{:04x}] {}: {}{}",
            inst.offset,
            inst.encoding.as_str(),
            inst.mnemonic,
            suffix
        ));
    }

    // -- Instruction dispatch --

    fn translate_instruction(&mut self, inst_idx: usize, inst: &Rdna2Instruction) {
        if inst.truncated {
            self.record_unsupported(inst);
            return;
        }

        let exec_write = classify_exec_mask_op(inst);
        let handled = match &inst.detail {
            Rdna2Detail::Sop1 {
                sdst,
                ssrc0,
                literal,
                ..
            } if exec_write.is_some() => {
                self.translate_exec_mask_write(exec_write.unwrap(), *sdst, *ssrc0, *literal)
            }
            Rdna2Detail::Sopp { op, .. } => self.translate_sopp(*op, inst.offset),
            Rdna2Detail::Sopc {
                op,
                ssrc0,
                ssrc1,
                literal,
            } => self.translate_sopc(*op, *ssrc0, *ssrc1, *literal),
            Rdna2Detail::Sop1 {
                op,
                sdst,
                ssrc0,
                literal,
            } => self.translate_sop1(*op, *sdst, *ssrc0, *literal),
            Rdna2Detail::Sopk {
                op, sdst, simm16, ..
            } => self.translate_sopk(*op, *sdst, *simm16),
            Rdna2Detail::Sop2 {
                op,
                sdst,
                ssrc0,
                ssrc1,
                literal,
            } => self.translate_sop2(*op, *sdst, *ssrc0, *ssrc1, *literal),
            Rdna2Detail::Smem {
                op,
                sdata,
                sbase,
                soffset,
                offset,
                ..
            } => self.translate_smem(inst_idx, *op, *sdata, *sbase, *soffset, *offset),
            Rdna2Detail::Vop1 {
                op,
                vdst,
                src0,
                literal,
            } => self.translate_vop1(*op, *vdst, *src0, *literal, 0, false, false, false),
            Rdna2Detail::Vopc {
                op,
                vsrc1,
                src0,
                literal,
            } => self.translate_vopc(*op, *src0, 256 + u16::from(*vsrc1), *literal, 0, 0, 0, None),
            Rdna2Detail::Vop2 {
                op,
                vdst,
                vsrc1,
                src0,
                literal,
            } => self.translate_vop2(
                *op,
                *vdst,
                *src0,
                256 + u16::from(*vsrc1),
                None,
                *literal,
                0,
                false,
                0,
                0,
                None,
            ),
            Rdna2Detail::Vop3 {
                op,
                vdst,
                sdst,
                src0,
                src1,
                src2,
                clamp,
                omod,
                abs,
                neg,
                op_sel,
                literal,
            } => self.translate_vop3(
                *op, *vdst, *sdst, *src0, *src1, *src2, *clamp, *omod, *abs, *neg, *op_sel, *literal,
            ),
            Rdna2Detail::Vop3P {
                op,
                vdst,
                src0,
                src1,
                src2,
                clamp,
                op_sel,
                op_sel_hi,
                neg_lo,
                neg_hi,
                literal,
            } => self.translate_vop3p(
                *op,
                *vdst,
                *src0,
                *src1,
                *src2,
                *clamp,
                *op_sel,
                *op_sel_hi,
                *neg_lo,
                *neg_hi,
                *literal,
            ),
            Rdna2Detail::Vintrp {
                op,
                vdst,
                attr,
                attrchan,
                ..
            } => self.translate_vintrp(*op, *vdst, *attr, *attrchan),
            Rdna2Detail::Mimg {
                op,
                dmask,
                dim,
                vaddr,
                vdata,
                srsrc,
                ssamp,
                ..
            } => self.translate_mimg(inst_idx, *op, *dmask, *dim, *vaddr, *vdata, *srsrc, *ssamp),
            Rdna2Detail::Mubuf {
                op,
                offset,
                offen,
                idxen,
                vaddr,
                vdata,
                srsrc,
                soffset,
                ..
            } => self.translate_mubuf_or_mtbuf(
                inst_idx, false, *op, 0, *offset, *offen, *idxen, *vaddr, *vdata, *srsrc, *soffset,
            ),
            Rdna2Detail::Mtbuf {
                op,
                format,
                offset,
                offen,
                idxen,
                vaddr,
                vdata,
                srsrc,
                soffset,
                ..
            } => self.translate_mubuf_or_mtbuf(
                inst_idx, true, *op, *format, *offset, *offen, *idxen, *vaddr, *vdata, *srsrc, *soffset,
            ),
            Rdna2Detail::Ds {
                op,
                offset0,
                offset1,
                addr,
                data0,
                data1,
                vdst,
                ..
            } => self.translate_ds(*op, *offset0, *offset1, *addr, *data0, *data1, *vdst),
            Rdna2Detail::Exp {
                target,
                en,
                compr,
                vsrc,
                ..
            } => self.translate_exp(*target, *en, *compr, *vsrc),
            Rdna2Detail::Flat { .. } | Rdna2Detail::Unknown { .. } => false,
        };

        if handled {
            self.translated_op_count += 1;
        } else {
            self.record_unsupported(inst);
        }
    }

    // -- SOPP / SOPC / SOP1 / SOPK / SOP2 / SMEM --

    fn translate_sopp(&mut self, op: u8, offset: usize) -> bool {
        match op {
            // s_nop, s_endpgm, s_waitcnt, s_setprio (scheduling priority hint),
            // s_endpgm_saved, s_endpgm_ordered_ps_done, s_code_end,
            // s_inst_prefetch, s_clause, s_wait_idle, s_waitcnt_depctr, s_round_mode
            0x00 | 0x01 | 0x0C | 0x0F | 0x1B | 0x1E | 0x1F | 0x20 | 0x21 | 0x22 | 0x23 | 0x24 => {
                true
            }
            // s_branch / s_cbranch_{scc0,scc1,vccz,vccnz,execz,execnz}: lifted into structured
            // WGSL control flow by `rdna2_cfg` (the branch itself emits no statement).
            0x02 | 0x04..=0x09 => self.structured_cf,
            // s_barrier: `workgroupBarrier()` in compute; WGSL/WebGPU forbids it in vertex and
            // fragment stages (no workgroups there), so it lowers to a documented no-op.
            // WGSL also requires barriers in uniform control flow, which lifted RDNA2 branches
            // (conditions on per-invocation `scc` / `vcc` / `exec_mask`) never are.
            0x0A => {
                if self.stage == ShaderStage::Compute && self.in_nonuniform_cf() {
                    self.note_comments.push(format!(
                        "// note [0x{offset:04x}] s_barrier skipped inside structured control flow for WGSL uniformity"
                    ));
                } else if self.stage == ShaderStage::Compute {
                    self.func.body.push(
                        Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
                        Span::UNDEFINED,
                    );
                } else {
                    self.note_comments.push(format!(
                        "// note [0x{offset:04x}] s_barrier: no-op in {} stage (workgroupBarrier is compute-only)",
                        self.stage.as_str()
                    ));
                }
                true
            }
            _ => false,
        }
    }

    fn translate_sopc(&mut self, op: u8, ssrc0: u8, ssrc1: u8, literal: Option<u32>) -> bool {
        match op {
            // s_bitcmp0_b32 / s_bitcmp1_b32: scc = (src0 & (1u << (src1 & 31u))) ==/!= 0u
            0x0C | 0x0D => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let idx_raw = self.read_src_u32(u16::from(ssrc1), literal);
                let mask31 = self.lit_u32(31);
                let idx = self.emit_bin(BinaryOperator::And, idx_raw, mask31);
                let one = self.lit_u32(1);
                let bit_mask = self.emit_bin(BinaryOperator::ShiftLeft, one, idx);
                let bit = self.emit_bin(BinaryOperator::And, v, bit_mask);
                let zero = self.lit_u32(0);
                let cmp_op = if op == 0x0C {
                    BinaryOperator::Equal
                } else {
                    BinaryOperator::NotEqual
                };
                let cmp = self.emit_bin(cmp_op, bit, zero);
                self.store_scc(cmp);
                return true;
            }
            // s_bitcmp0_b64 / s_bitcmp1_b64: test bit (src1 & 63) of the 64-bit src0 pair
            0x0E | 0x0F => {
                let (lo, hi) = self.read_ssrc_pair(ssrc0, literal);
                let idx_raw = self.read_src_u32(u16::from(ssrc1), literal);
                let mask63 = self.lit_u32(63);
                let mask31 = self.lit_u32(31);
                let c32 = self.lit_u32(32);
                let idx = self.emit_bin(BinaryOperator::And, idx_raw, mask63);
                let in_hi = self.emit_bin(BinaryOperator::GreaterEqual, idx, c32);
                let word = self.emit_select(in_hi, hi, lo);
                let sh = self.emit_bin(BinaryOperator::And, idx, mask31);
                let one = self.lit_u32(1);
                let bit_mask = self.emit_bin(BinaryOperator::ShiftLeft, one, sh);
                let bit = self.emit_bin(BinaryOperator::And, word, bit_mask);
                let zero = self.lit_u32(0);
                let cmp_op = if op == 0x0E {
                    BinaryOperator::Equal
                } else {
                    BinaryOperator::NotEqual
                };
                let cmp = self.emit_bin(cmp_op, bit, zero);
                self.store_scc(cmp);
                return true;
            }
            // s_cmp_eq_u64 / s_cmp_lg_u64 on SGPR pairs
            0x12 | 0x13 => {
                let (a_lo, a_hi) = self.read_ssrc_pair(ssrc0, literal);
                let (b_lo, b_hi) = self.read_ssrc_pair(ssrc1, literal);
                let cmp = if op == 0x12 {
                    let eq_lo = self.emit_bin(BinaryOperator::Equal, a_lo, b_lo);
                    let eq_hi = self.emit_bin(BinaryOperator::Equal, a_hi, b_hi);
                    self.emit_bin(BinaryOperator::LogicalAnd, eq_lo, eq_hi)
                } else {
                    let ne_lo = self.emit_bin(BinaryOperator::NotEqual, a_lo, b_lo);
                    let ne_hi = self.emit_bin(BinaryOperator::NotEqual, a_hi, b_hi);
                    self.emit_bin(BinaryOperator::LogicalOr, ne_lo, ne_hi)
                };
                self.store_scc(cmp);
                return true;
            }
            _ => {}
        }
        let bin_op = match op {
            0x00 | 0x06 => BinaryOperator::Equal,
            0x01 | 0x07 => BinaryOperator::NotEqual,
            0x02 | 0x08 => BinaryOperator::Greater,
            0x03 | 0x09 => BinaryOperator::GreaterEqual,
            0x04 | 0x0A => BinaryOperator::Less,
            0x05 | 0x0B => BinaryOperator::LessEqual,
            _ => return false,
        };
        let is_signed = op <= 0x05;
        let (left, right) = if is_signed {
            (
                self.read_src_i32(u16::from(ssrc0), literal),
                self.read_src_i32(u16::from(ssrc1), literal),
            )
        } else {
            (
                self.read_src_u32(u16::from(ssrc0), literal),
                self.read_src_u32(u16::from(ssrc1), literal),
            )
        };
        let cmp = self.emit_expr(Expression::Binary {
            op: bin_op,
            left,
            right,
        });
        self.store_scc(cmp);
        true
    }

    fn translate_sop1(&mut self, op: u8, sdst: u8, ssrc0: u8, literal: Option<u32>) -> bool {
        match op {
            // s_mov_b32 / s_wqm_b32 (whole-quad-mode exec mask setup is identity in WGSL)
            0x03 | 0x09 => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                self.store_sgpr_u32(sdst, v);
                if op == 0x09 {
                    self.update_scc_nonzero(v);
                }
                true
            }
            // s_mov_b64 / s_wqm_b64: 64-bit scalar register-pair move (evaluate both halves before storing)
            0x04 | 0x0A => {
                let (lo, hi) = self.read_ssrc_pair(ssrc0, literal);
                match sdst {
                    0..=104 => {
                        self.store_sgpr_u32(sdst, lo);
                        self.store_sgpr_u32(sdst + 1, hi);
                    }
                    105 => {
                        self.store_sgpr_u32(105, lo);
                    }
                    106 => {
                        let combined = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::InclusiveOr,
                            left: lo,
                            right: hi,
                        });
                        let zero = self.lit_u32(0);
                        let cond = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::NotEqual,
                            left: combined,
                            right: zero,
                        });
                        self.store_vcc(cond);
                    }
                    126 => self.store_exec_lane_bit(lo),
                    _ => {}
                }
                true
            }
            // s_cmov_b32
            0x05 => {
                let cond = self.load_scc();
                let accept = self.read_src_u32(u16::from(ssrc0), literal);
                let reject = self.load_sgpr_u32(sdst);
                let res = self.emit_expr(Expression::Select {
                    condition: cond,
                    accept,
                    reject,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_not_b32
            0x07 => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let res = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: v,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
                true
            }
            // s_not_b64: 64-bit bitwise NOT on SGPR pair; SCC = ((lo | hi) != 0)
            0x08 => {
                let (lo, hi) = self.read_ssrc_pair(ssrc0, literal);
                let nlo = self.emit_bit_not(lo);
                let nhi = self.emit_bit_not(hi);
                self.store_sdst_pair(sdst, nlo, nhi);
                self.update_scc_nonzero_pair(nlo, nhi);
                true
            }
            // s_brev_b32
            0x0B => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::ReverseBits,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_bcnt1_i32_b32
            0x0F => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::CountOneBits,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
                true
            }
            // s_ff1_i32_b32: find first 1 bit from LSB; returns -1 (0xFFFFFFFF) when src0 == 0
            0x13 => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let ctz = self.emit_expr(Expression::Math {
                    fun: MathFunction::CountTrailingZeros,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let zero = self.lit_u32(0);
                let is_zero = self.emit_bin(BinaryOperator::Equal, v, zero);
                let neg_one = self.lit_u32(0xFFFF_FFFF);
                let res = self.emit_select(is_zero, neg_one, ctz);
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_flbit_i32_b32: count leading zeros from MSB; returns -1 (0xFFFFFFFF) when src0 == 0
            0x15 => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let clz = self.emit_expr(Expression::Math {
                    fun: MathFunction::CountLeadingZeros,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let zero = self.lit_u32(0);
                let is_zero = self.emit_bin(BinaryOperator::Equal, v, zero);
                let neg_one = self.lit_u32(0xFFFF_FFFF);
                let res = self.emit_select(is_zero, neg_one, clz);
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_sext_i32_i8 / s_sext_i32_i16
            0x19 | 0x1A => {
                let v = self.read_src_i32(u16::from(ssrc0), literal);
                let shift_bits = if op == 0x19 { 24u32 } else { 16u32 };
                let sh = self.lit_u32(shift_bits);
                let left = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: v,
                    right: sh,
                });
                let sext = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left,
                    right: sh,
                });
                let u = self.bitcast_i32_to_u32(sext);
                self.store_sgpr_u32(sdst, u);
                true
            }
            // s_bitset0_b32 (0x1B) / s_bitset1_b32 (0x1D): clear/set bit (src0 & 31) in sdst
            0x1B | 0x1D => {
                let idx_raw = self.read_src_u32(u16::from(ssrc0), literal);
                let mask31 = self.lit_u32(31);
                let idx = self.emit_bin(BinaryOperator::And, idx_raw, mask31);
                let one = self.lit_u32(1);
                let bit = self.emit_bin(BinaryOperator::ShiftLeft, one, idx);
                let cur = self.load_sgpr_u32(sdst);
                let res = if op == 0x1D {
                    self.emit_bin(BinaryOperator::InclusiveOr, cur, bit)
                } else {
                    let nbit = self.emit_bit_not(bit);
                    self.emit_bin(BinaryOperator::And, cur, nbit)
                };
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_abs_i32
            0x34 => {
                let v = self.read_src_i32(u16::from(ssrc0), literal);
                let a = self.emit_expr(Expression::Math {
                    fun: MathFunction::Abs,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let u = self.bitcast_i32_to_u32(a);
                self.store_sgpr_u32(sdst, u);
                true
            }
            _ => false,
        }
    }

    fn translate_sopk(&mut self, op: u8, sdst: u8, simm16: u16) -> bool {
        let imm_u32 = (simm16 as i16 as i32) as u32;
        match op {
            // s_movk_i32
            0x00 => {
                let v = self.lit_u32(imm_u32);
                self.store_sgpr_u32(sdst, v);
                true
            }
            // s_cmpk_{eq,lg,gt,ge,lt,le}_{i32,u32}: compare sdst against sign/zero-extended simm16 -> SCC
            0x03..=0x0E => {
                let is_signed = op <= 0x08;
                let cc = if is_signed { op - 0x03 } else { op - 0x09 };
                let bin_op = match cc {
                    0 => BinaryOperator::Equal,
                    1 => BinaryOperator::NotEqual,
                    2 => BinaryOperator::Greater,
                    3 => BinaryOperator::GreaterEqual,
                    4 => BinaryOperator::Less,
                    _ => BinaryOperator::LessEqual,
                };
                let (lhs, rhs) = if is_signed {
                    (
                        self.read_src_i32(u16::from(sdst), None),
                        self.lit_i32(i32::from(simm16 as i16)),
                    )
                } else {
                    (
                        self.read_src_u32(u16::from(sdst), None),
                        self.lit_u32(u32::from(simm16)),
                    )
                };
                let cmp = self.emit_bin(bin_op, lhs, rhs);
                self.store_scc(cmp);
                true
            }
            // s_addk_i32
            0x0F => {
                let lhs = self.load_sgpr_u32(sdst);
                let rhs = self.lit_u32(imm_u32);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: lhs,
                    right: rhs,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_mulk_i32
            0x10 => {
                let lhs = self.load_sgpr_u32(sdst);
                let rhs = self.lit_u32(imm_u32);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: lhs,
                    right: rhs,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_waitcnt_vscnt / vmcnt / expcnt / lgkmcnt
            0x17..=0x1A => true,
            _ => false,
        }
    }

    fn translate_sop2(
        &mut self,
        op: u8,
        sdst: u8,
        ssrc0: u8,
        ssrc1: u8,
        literal: Option<u32>,
    ) -> bool {
        let s0_u = || u16::from(ssrc0);
        let s1_u = || u16::from(ssrc1);
        match op {
            // s_add_u32 / s_add_i32
            0x00 | 0x02 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: l,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                if op == 0x00 && self.scc_local.is_some() {
                    let carry = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Less,
                        left: res,
                        right: l,
                    });
                    self.store_scc(carry);
                }
                true
            }
            // s_sub_u32 / s_sub_i32
            0x01 | 0x03 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: l,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                if op == 0x01 && self.scc_local.is_some() {
                    let borrow = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Less,
                        left: l,
                        right: r,
                    });
                    self.store_scc(borrow);
                }
                true
            }
            // s_addc_u32: ssrc0 + ssrc1 + scc (sets scc = carry-out)
            0x04 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let scc_in = self.load_scc();
                let one = self.lit_u32(1);
                let zero = self.lit_u32(0);
                let cin = self.emit_select(scc_in, one, zero);
                let sum1 = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: l,
                    right: r,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: sum1,
                    right: cin,
                });
                self.store_sgpr_u32(sdst, res);
                if self.scc_local.is_some() {
                    let c1 = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Less,
                        left: sum1,
                        right: l,
                    });
                    let c2 = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Less,
                        left: res,
                        right: sum1,
                    });
                    let carry = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::LogicalOr,
                        left: c1,
                        right: c2,
                    });
                    self.store_scc(carry);
                }
                true
            }
            // s_subb_u32: ssrc0 - ssrc1 - scc (sets scc = borrow-out)
            0x05 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let scc_in = self.load_scc();
                let one = self.lit_u32(1);
                let zero = self.lit_u32(0);
                let bin = self.emit_select(scc_in, one, zero);
                let diff1 = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: l,
                    right: r,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: diff1,
                    right: bin,
                });
                self.store_sgpr_u32(sdst, res);
                if self.scc_local.is_some() {
                    let b1 = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Less,
                        left: l,
                        right: r,
                    });
                    let b2 = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Less,
                        left: diff1,
                        right: bin,
                    });
                    let borrow = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::LogicalOr,
                        left: b1,
                        right: b2,
                    });
                    self.store_scc(borrow);
                }
                true
            }
            // s_min_i32 / s_max_i32
            0x06 | 0x08 => {
                let l = self.read_src_i32(s0_u(), literal);
                let r = self.read_src_i32(s1_u(), literal);
                let fun = if op == 0x06 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                let u = self.bitcast_i32_to_u32(res);
                self.store_sgpr_u32(sdst, u);
                true
            }
            // s_min_u32 / s_max_u32
            0x07 | 0x09 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let fun = if op == 0x07 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_cselect_b32: scc ? ssrc0 : ssrc1
            0x0A => {
                let cond = self.load_scc();
                let accept = self.read_src_u32(s0_u(), literal);
                let reject = self.read_src_u32(s1_u(), literal);
                let res = self.emit_expr(Expression::Select {
                    condition: cond,
                    accept,
                    reject,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_cselect_b64: scc ? src0[0:1] : src1[0:1]
            0x0B => {
                let cond = self.load_scc();
                let (a_lo, a_hi) = self.read_ssrc_pair(ssrc0, literal);
                let (b_lo, b_hi) = self.read_ssrc_pair(ssrc1, literal);
                let lo = self.emit_select(cond, a_lo, b_lo);
                let hi = self.emit_select(cond, a_hi, b_hi);
                self.store_sdst_pair(sdst, lo, hi);
                true
            }
            // s_{and,or,xor,andn2,orn2,nand,nor,xnor}_b64 on SGPR pairs; scc = ((lo | hi) != 0u)
            0x0F | 0x11 | 0x13 | 0x15 | 0x17 | 0x19 | 0x1B | 0x1D => {
                let (l_lo, l_hi) = self.read_ssrc_pair(ssrc0, literal);
                let (r_lo, r_hi) = self.read_ssrc_pair(ssrc1, literal);
                let lo = self.emit_sop2_bitwise_half(op, l_lo, r_lo);
                let hi = self.emit_sop2_bitwise_half(op, l_hi, r_hi);
                self.store_sdst_pair(sdst, lo, hi);
                self.update_scc_nonzero_pair(lo, hi);
                true
            }
            // s_lshl_b64 / s_lshr_b64 / s_ashr_i64: 64-bit pair shift by (src1 & 63); scc = result != 0
            0x1F | 0x21 | 0x23 => {
                let (lo, hi) = self.read_ssrc_pair(ssrc0, literal);
                let amount = self.read_src_u32(s1_u(), literal);
                let kind = match op {
                    0x1F => Shift64Kind::Left,
                    0x21 => Shift64Kind::LogicalRight,
                    _ => Shift64Kind::ArithmeticRight,
                };
                let (rlo, rhi) = self.emit_shift64(kind, lo, hi, amount);
                self.store_sdst_pair(sdst, rlo, rhi);
                self.update_scc_nonzero_pair(rlo, rhi);
                true
            }
            // s_mul_hi_u32 / s_mul_hi_i32: exact high 32 bits of the 64-bit product
            0x35 | 0x36 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let res = if op == 0x35 {
                    self.emit_mul_hi_u32(l, r)
                } else {
                    self.emit_mul_hi_i32(l, r)
                };
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_and_b32 / s_or_b32 / s_xor_b32
            0x0E | 0x10 | 0x12 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let bin = match op {
                    0x0E => BinaryOperator::And,
                    0x10 => BinaryOperator::InclusiveOr,
                    _ => BinaryOperator::ExclusiveOr,
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
                true
            }
            // s_andn2_b32 / s_orn2_b32: s0 & ~s1 / s0 | ~s1
            0x14 | 0x16 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let nr = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: r,
                });
                let bin = if op == 0x14 {
                    BinaryOperator::And
                } else {
                    BinaryOperator::InclusiveOr
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: nr,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
                true
            }
            // s_nand_b32 / s_nor_b32 / s_xnor_b32: ~(s0 op s1)
            0x18 | 0x1A | 0x1C => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let bin = match op {
                    0x18 => BinaryOperator::And,
                    0x1A => BinaryOperator::InclusiveOr,
                    _ => BinaryOperator::ExclusiveOr,
                };
                let inner = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: r,
                });
                let res = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: inner,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
                true
            }
            // s_lshl_b32 / s_lshr_b32
            0x1E | 0x20 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let mask = self.lit_u32(31);
                let shift = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: r,
                    right: mask,
                });
                let bin = if op == 0x1E {
                    BinaryOperator::ShiftLeft
                } else {
                    BinaryOperator::ShiftRight
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: shift,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
                true
            }
            // s_ashr_i32
            0x22 => {
                let l = self.read_src_i32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let mask = self.lit_u32(31);
                let shift = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: r,
                    right: mask,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: l,
                    right: shift,
                });
                let u = self.bitcast_i32_to_u32(res);
                self.store_sgpr_u32(sdst, u);
                self.update_scc_nonzero(u);
                true
            }
            // s_bfm_b32: ((1u << (s0 & 31)) - 1u) << (s1 & 31)
            0x24 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let mask31 = self.lit_u32(31);
                let width = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: l,
                    right: mask31,
                });
                let offset = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: r,
                    right: mask31,
                });
                let one = self.lit_u32(1);
                let sh_w = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: one,
                    right: width,
                });
                let base_mask = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: sh_w,
                    right: one,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: base_mask,
                    right: offset,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_bfm_b64: ((1u64 << (s0 & 63)) - 1u64) << (s1 & 63) -> s[sdst:sdst+1]
            0x25 => {
                if let (Some(w_i), Some(o_i)) = (
                    decode_inline_u32_src(s0_u(), literal),
                    decode_inline_u32_src(s1_u(), literal),
                ) {
                    let w = (w_i as u32) & 63;
                    let o = (o_i as u32) & 63;
                    let base: u64 = if w == 0 {
                        0
                    } else if w >= 64 {
                        u64::MAX
                    } else {
                        (1u64 << w) - 1
                    };
                    let val = base.wrapping_shl(o);
                    let lo = self.lit_u32(val as u32);
                    let hi = self.lit_u32((val >> 32) as u32);
                    self.store_sgpr_u32(sdst, lo);
                    if sdst <= 104 {
                        self.store_sgpr_u32(sdst + 1, hi);
                    }
                } else {
                    // Dynamic width/offset: compute 32-bit low half and zero-extended high half.
                    let l = self.read_src_u32(s0_u(), literal);
                    let r = self.read_src_u32(s1_u(), literal);
                    let mask31 = self.lit_u32(31);
                    let width = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: l,
                        right: mask31,
                    });
                    let offset = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: r,
                        right: mask31,
                    });
                    let one = self.lit_u32(1);
                    let sh_w = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: one,
                        right: width,
                    });
                    let base_mask = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Subtract,
                        left: sh_w,
                        right: one,
                    });
                    let lo = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: base_mask,
                        right: offset,
                    });
                    let zero = self.lit_u32(0);
                    self.store_sgpr_u32(sdst, lo);
                    if sdst <= 104 {
                        self.store_sgpr_u32(sdst + 1, zero);
                    }
                }
                true
            }
            // s_mul_i32
            0x26 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: l,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_bfe_u32: unsigned bitfield extract
            //   offset = src1[4:0], width = src1[22:16]
            //   sdst = (width == 0) ? 0 : ((src0 >> offset) & mask(width))
            0x27 => {
                let s0 = self.read_src_u32(s0_u(), literal);
                let s1 = self.read_src_u32(s1_u(), literal);
                let mask31 = self.lit_u32(31);
                let offset = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: s1,
                    right: mask31,
                });
                let shift16 = self.lit_u32(16);
                let s1_hi = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: s1,
                    right: shift16,
                });
                let mask7f = self.lit_u32(0x7F);
                let width = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: s1_hi,
                    right: mask7f,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: s0,
                    right: offset,
                });
                let max_shift = self.lit_u32(31);
                let safe_w = self.emit_expr(Expression::Math {
                    fun: MathFunction::Min,
                    arg: width,
                    arg1: Some(max_shift),
                    arg2: None,
                    arg3: None,
                });
                let one = self.lit_u32(1);
                let one_shl = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: one,
                    right: safe_w,
                });
                let narrow_mask = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: one_shl,
                    right: one,
                });
                let lit32 = self.lit_u32(32);
                let wide = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::GreaterEqual,
                    left: width,
                    right: lit32,
                });
                let all_ones = self.lit_u32(0xFFFF_FFFF);
                let eff_mask = self.emit_expr(Expression::Select {
                    condition: wide,
                    accept: all_ones,
                    reject: narrow_mask,
                });
                let extracted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: shifted,
                    right: eff_mask,
                });
                let zero = self.lit_u32(0);
                let is_zero_w = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Equal,
                    left: width,
                    right: zero,
                });
                let res = self.emit_expr(Expression::Select {
                    condition: is_zero_w,
                    accept: zero,
                    reject: extracted,
                });
                self.store_sgpr_u32(sdst, res);
                let nonzero = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::NotEqual,
                    left: res,
                    right: zero,
                });
                self.store_scc(nonzero);
                true
            }
            // s_lshl{1,2,3,4}_add_u32
            0x2E..=0x31 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let sh = self.lit_u32(u32::from(op - 0x2D));
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: l,
                    right: sh,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: shifted,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_pack_ll_b32_b16 / s_pack_lh_b32_b16 / s_pack_hh_b32_b16
            0x32..=0x34 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let mask_lo = self.lit_u32(0x0000_FFFF);
                let mask_hi = self.lit_u32(0xFFFF_0000);
                let sh16 = self.lit_u32(16);
                let lo_half = if op == 0x34 {
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftRight,
                        left: l,
                        right: sh16,
                    })
                } else {
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: l,
                        right: mask_lo,
                    })
                };
                let hi_half = if op == 0x32 {
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: r,
                        right: sh16,
                    })
                } else {
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: r,
                        right: mask_hi,
                    })
                };
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: lo_half,
                    right: hi_half,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            _ => false,
        }
    }

    fn translate_smem(
        &mut self,
        inst_idx: usize,
        op: u8,
        sdata: u8,
        sbase: u8,
        soffset: u8,
        byte_offset: i32,
    ) -> bool {
        if self.vsharp_fetch_insts.contains(&inst_idx) {
            return true;
        }
        let Some(count) = smem_load_dword_count(op) else {
            return false;
        };
        let cbv_key = self
            .indirect_smem_cbv_keys
            .get(&inst_idx)
            .copied()
            .unwrap_or(CbvKey {
                base_sbase: sbase,
                user_sgpr: None,
                table_offset: None,
                dynamic_heap: None,
            });
        let Some(&(cb_global, cb_vec4_count)) = self
            .cb_globals
            .get(&cbv_key)
            .or_else(|| self.cb_globals.values().next())
        else {
            return false;
        };
        let total_dwords = cb_vec4_count.max(1).saturating_mul(4);

        let cb_ptr = self.raw_expr(Expression::GlobalVariable(cb_global));
        let data_ptr = self.emit_expr(Expression::AccessIndex {
            base: cb_ptr,
            index: 0,
        });

        let has_dynamic_offset = soffset != 0x7D && soffset <= 105;
        let base_word_dyn = if has_dynamic_offset {
            let s_off = self.load_sgpr_u32(soffset);
            let imm_off = self.lit_u32(byte_offset as u32);
            let byte_sum = self.emit_expr(Expression::Binary {
                op: BinaryOperator::Add,
                left: s_off,
                right: imm_off,
            });
            let two = self.lit_u32(2);
            Some(self.emit_expr(Expression::Binary {
                op: BinaryOperator::ShiftRight,
                left: byte_sum,
                right: two,
            }))
        } else {
            None
        };

        let static_base_word = if byte_offset > 0 {
            (byte_offset as u32) / 4
        } else {
            0
        };
        let mut cached_static_vec: Option<(u32, Handle<Expression>)> = None;

        for k in 0..count {
            let dst_reg = sdata.saturating_add(k);
            if dst_reg > 105 {
                break;
            }
            let elem_val = if let Some(base_w) = base_word_dyn {
                let k_lit = self.lit_u32(u32::from(k));
                let raw_word = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: base_w,
                    right: k_lit,
                });
                let total_lit = self.lit_u32(total_dwords);
                let bounded_word = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Modulo,
                    left: raw_word,
                    right: total_lit,
                });
                let two = self.lit_u32(2);
                let vec_idx = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: bounded_word,
                    right: two,
                });
                let three = self.lit_u32(3);
                let comp_idx = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: bounded_word,
                    right: three,
                });
                let vec_ptr = self.emit_expr(Expression::Access {
                    base: data_ptr,
                    index: vec_idx,
                });
                let vec_val = self.emit_expr(Expression::Load { pointer: vec_ptr });
                self.emit_expr(Expression::Access {
                    base: vec_val,
                    index: comp_idx,
                })
            } else {
                let word_idx = (static_base_word.saturating_add(u32::from(k))) % total_dwords;
                let vec_idx = word_idx / 4;
                let comp_idx = word_idx % 4;
                let vec_val = match cached_static_vec {
                    Some((cached_idx, handle)) if cached_idx == vec_idx => handle,
                    _ => {
                        let vec_ptr = self.emit_expr(Expression::AccessIndex {
                            base: data_ptr,
                            index: vec_idx,
                        });
                        let loaded = self.emit_expr(Expression::Load { pointer: vec_ptr });
                        cached_static_vec = Some((vec_idx, loaded));
                        loaded
                    }
                };
                self.emit_expr(Expression::AccessIndex {
                    base: vec_val,
                    index: comp_idx,
                })
            };

            self.store_sgpr_u32(dst_reg, elem_val);
        }

        true
    }

    // -- VALU: VOP1 / VOPC / VOP2 / VOP3 / VOP3P --

    #[allow(clippy::too_many_arguments)]
    fn translate_vop1(
        &mut self,
        op: u8,
        vdst: u8,
        src0: u16,
        literal: Option<u32>,
        omod: u8,
        clamp: bool,
        abs0: bool,
        neg0: bool,
    ) -> bool {
        match op {
            // v_nop / v_pipeflush / v_clrexcp
            0x00 | 0x1B | 0x41 => true,
            // v_mov_b32
            0x01 => {
                let val = if abs0 || neg0 || omod != 0 || clamp {
                    let f = self.read_src_f32(src0, literal, abs0, neg0);
                    self.apply_float_out_mods(f, omod, clamp)
                } else {
                    self.read_src_f32(src0, literal, false, false)
                };
                self.store_vgpr_f32(vdst, val);
                true
            }
            // v_readfirstlane_b32 sdst, vsrc0
            0x02 => {
                let u = self.read_src_u32(src0, literal);
                self.store_sgpr_u32(vdst & 0x7F, u);
                true
            }
            // v_cvt_f32_i32
            0x05 => {
                let s = self.read_src_i32(src0, literal);
                let f = self.emit_expr(Expression::As {
                    expr: s,
                    kind: ScalarKind::Float,
                    convert: Some(4),
                });
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_cvt_f32_u32
            0x06 => {
                let u = self.read_src_u32(src0, literal);
                let f = self.emit_expr(Expression::As {
                    expr: u,
                    kind: ScalarKind::Float,
                    convert: Some(4),
                });
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_cvt_u32_f32
            0x07 => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let u = self.emit_expr(Expression::As {
                    expr: f,
                    kind: ScalarKind::Uint,
                    convert: Some(4),
                });
                self.store_vgpr_u32(vdst, u);
                true
            }
            // v_cvt_i32_f32
            0x08 => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let i = self.emit_expr(Expression::As {
                    expr: f,
                    kind: ScalarKind::Sint,
                    convert: Some(4),
                });
                self.store_vgpr_i32(vdst, i);
                true
            }
            // v_cvt_f16_f32: convert f32 -> f16 stored in low 16 bits of vdst
            0x0A => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_cvt_f32_f16: convert low 16 bits f16 -> f32
            0x0B => {
                let f = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_cvt_rpi_i32_f32 / v_cvt_flr_i32_f32
            0x0C | 0x0D => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let shifted = if op == 0x0C {
                    let half = self.lit_f32(0.5);
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Add,
                        left: f,
                        right: half,
                    })
                } else {
                    f
                };
                let floored = self.emit_expr(Expression::Math {
                    fun: MathFunction::Floor,
                    arg: shifted,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let i = self.emit_expr(Expression::As {
                    expr: floored,
                    kind: ScalarKind::Sint,
                    convert: Some(4),
                });
                self.store_vgpr_i32(vdst, i);
                true
            }
            // v_cvt_f32_ubyte0..3: extract byte k as f32
            0x11..=0x14 => {
                let u = self.read_src_u32(src0, literal);
                let shift_bits = u32::from(op - 0x11) * 8;
                let shifted = if shift_bits > 0 {
                    let sh = self.lit_u32(shift_bits);
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftRight,
                        left: u,
                        right: sh,
                    })
                } else {
                    u
                };
                let mask = self.lit_u32(0xFF);
                let byte_u = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: shifted,
                    right: mask,
                });
                let f = self.emit_expr(Expression::As {
                    expr: byte_u,
                    kind: ScalarKind::Float,
                    convert: Some(4),
                });
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_fract_f32 / v_trunc_f32 / v_ceil_f32 / v_rndne_f32 / v_floor_f32 /
            // v_exp_f32 / v_log_f32 / v_rsq_f32 / v_sqrt_f32
            0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x25 | 0x27 | 0x2E | 0x33 => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let fun = match op {
                    0x20 => MathFunction::Fract,
                    0x21 => MathFunction::Trunc,
                    0x22 => MathFunction::Ceil,
                    0x23 => MathFunction::Round,
                    0x24 => MathFunction::Floor,
                    0x25 => MathFunction::Exp2,
                    0x27 => MathFunction::Log2,
                    0x2E => MathFunction::InverseSqrt,
                    _ => MathFunction::Sqrt,
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: f,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_rcp_f32 / v_rcp_iflag_f32
            0x2A | 0x2B => {
                let one = self.lit_f32(1.0);
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: one,
                    right: f,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_sin_f32 / v_cos_f32 (input is in turns: angle = x * 2*pi)
            0x35 | 0x36 => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let tau = self.lit_f32(std::f32::consts::TAU);
                let rad = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: f,
                    right: tau,
                });
                let fun = if op == 0x35 {
                    MathFunction::Sin
                } else {
                    MathFunction::Cos
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: rad,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_not_b32
            0x37 => {
                let u = self.read_src_u32(src0, literal);
                let res = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: u,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_bfrev_b32
            0x38 => {
                let u = self.read_src_u32(src0, literal);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::ReverseBits,
                    arg: u,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_rcp_f16 (emulated via f32 + pack2x16float, zero `enable f16;`)
            0x54 => {
                let one = self.lit_f32(1.0);
                let f = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: one,
                    right: f,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_sqrt_f16 / v_rsq_f16 / v_log_f16 / v_exp_f16 / v_floor_f16 / v_ceil_f16 /
            // v_trunc_f16 / v_rndne_f16 / v_fract_f16
            0x55 | 0x56 | 0x57 | 0x58 | 0x5B | 0x5C | 0x5D | 0x5E | 0x5F => {
                let f = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let fun = match op {
                    0x55 => MathFunction::Sqrt,
                    0x56 => MathFunction::InverseSqrt,
                    0x57 => MathFunction::Log2,
                    0x58 => MathFunction::Exp2,
                    0x5B => MathFunction::Floor,
                    0x5C => MathFunction::Ceil,
                    0x5D => MathFunction::Trunc,
                    0x5E => MathFunction::Round,
                    _ => MathFunction::Fract,
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: f,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_swap_b32: atomically swap v[vdst] and v[vsrc0]
            0x65 => {
                if (256..=511).contains(&src0) {
                    let vsrc = (src0 - 256) as u8;
                    if vsrc != vdst {
                        let old_dst = self.load_vgpr_f32(vdst);
                        let old_src = self.load_vgpr_f32(vsrc);
                        self.store_vgpr_f32(vdst, old_src);
                        self.store_vgpr_f32(vsrc, old_dst);
                    }
                } else {
                    let val = self.read_src_f32(src0, literal, false, false);
                    self.store_vgpr_f32(vdst, val);
                }
                true
            }
            _ => false,
        }
    }

    /// Evaluate any of the 16 IEEE-754 floating-point comparison condition codes (NaN-aware):
    ///   0=f, 1=lt, 2=eq, 3=le, 4=gt, 5=lg, 6=ge, 7=o,
    ///   8=u, 9=nge, 10=nlg, 11=ngt, 12=nle, 13=neq, 14=nlt, 15=tru
    fn emit_float_cmp(
        &mut self,
        cc: u8,
        l: Handle<Expression>,
        r: Handle<Expression>,
    ) -> Handle<Expression> {
        match cc {
            0 => self.lit_bool(false),
            1 => self.emit_bin(BinaryOperator::Less, l, r),
            2 => self.emit_bin(BinaryOperator::Equal, l, r),
            3 => self.emit_bin(BinaryOperator::LessEqual, l, r),
            4 => self.emit_bin(BinaryOperator::Greater, l, r),
            5 => {
                let lt = self.emit_bin(BinaryOperator::Less, l, r);
                let gt = self.emit_bin(BinaryOperator::Greater, l, r);
                self.emit_bin(BinaryOperator::LogicalOr, lt, gt)
            }
            6 => self.emit_bin(BinaryOperator::GreaterEqual, l, r),
            7 => {
                let lo = self.emit_bin(BinaryOperator::Equal, l, l);
                let ro = self.emit_bin(BinaryOperator::Equal, r, r);
                self.emit_bin(BinaryOperator::LogicalAnd, lo, ro)
            }
            8 => {
                let lu = self.emit_bin(BinaryOperator::NotEqual, l, l);
                let ru = self.emit_bin(BinaryOperator::NotEqual, r, r);
                self.emit_bin(BinaryOperator::LogicalOr, lu, ru)
            }
            9 => {
                // nge: !(l >= r) — true when l < r or either operand is NaN
                let ge = self.emit_bin(BinaryOperator::GreaterEqual, l, r);
                self.emit_expr(Expression::Unary {
                    op: UnaryOperator::LogicalNot,
                    expr: ge,
                })
            }
            10 => {
                // nlg: !(l < r || l > r) — true when l == r or either operand is NaN
                let lt = self.emit_bin(BinaryOperator::Less, l, r);
                let gt = self.emit_bin(BinaryOperator::Greater, l, r);
                let lg = self.emit_bin(BinaryOperator::LogicalOr, lt, gt);
                self.emit_expr(Expression::Unary {
                    op: UnaryOperator::LogicalNot,
                    expr: lg,
                })
            }
            11 => {
                // ngt: !(l > r) — true when l <= r or either operand is NaN
                let gt = self.emit_bin(BinaryOperator::Greater, l, r);
                self.emit_expr(Expression::Unary {
                    op: UnaryOperator::LogicalNot,
                    expr: gt,
                })
            }
            12 => {
                // nle: !(l <= r) — true when l > r or either operand is NaN
                let le = self.emit_bin(BinaryOperator::LessEqual, l, r);
                self.emit_expr(Expression::Unary {
                    op: UnaryOperator::LogicalNot,
                    expr: le,
                })
            }
            13 => self.emit_bin(BinaryOperator::NotEqual, l, r),
            14 => {
                // nlt: !(l < r) — true when l >= r or either operand is NaN
                let lt = self.emit_bin(BinaryOperator::Less, l, r);
                self.emit_expr(Expression::Unary {
                    op: UnaryOperator::LogicalNot,
                    expr: lt,
                })
            }
            _ => self.lit_bool(true),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_vopc(
        &mut self,
        op: u8,
        src0: u16,
        src1: u16,
        literal: Option<u32>,
        abs: u8,
        neg: u8,
        op_sel: u8,
        sdst_override: Option<u8>,
    ) -> bool {
        let cmp = match op {
            // f32 comparisons: 0x00..=0x0F (v_cmp_*_f32) and 0x10..=0x1F (v_cmpx_*_f32)
            0x00..=0x1F => {
                let cc = op & 0xF;
                let l = self.read_src_f32(src0, literal, (abs & 1) != 0, (neg & 1) != 0);
                let r = self.read_src_f32(src1, literal, (abs & 2) != 0, (neg & 2) != 0);
                self.emit_float_cmp(cc, l, r)
            }
            // f16 comparisons: GFX10.3 (0xC8..=0xCF, 0xD8..=0xDF, 0xE8..=0xEF, 0xF8..=0xFF)
            // plus legacy/compat 0x40..=0x5F
            0x40..=0x5F | 0xC8..=0xCF | 0xD8..=0xDF | 0xE8..=0xEF | 0xF8..=0xFF => {
                let cc = if op <= 0x5F {
                    op & 0xF
                } else if op >= 0xE8 {
                    8 + (op & 0x7)
                } else {
                    op & 0x7
                };
                let l = self.read_src_mixed_f16_or_f32(
                    src0,
                    literal,
                    true,
                    (op_sel & 1) != 0,
                    (abs & 1) != 0,
                    (neg & 1) != 0,
                );
                let r = self.read_src_mixed_f16_or_f32(
                    src1,
                    literal,
                    true,
                    (op_sel & 2) != 0,
                    (abs & 2) != 0,
                    (neg & 2) != 0,
                );
                self.emit_float_cmp(cc, l, r)
            }
            // i16 comparisons: GFX10.3 (0x89..=0x8E, 0x99..=0x9E) and compat (0x60..=0x67, 0x70..=0x77)
            0x60..=0x67 | 0x70..=0x77 | 0x89..=0x8E | 0x99..=0x9E => {
                let cc = op & 0x7;
                let s16 = self.lit_u32(16);
                let l_raw = self.read_src_i32(src0, literal);
                let l = if (op_sel & 1) != 0 {
                    self.emit_bin(BinaryOperator::ShiftRight, l_raw, s16)
                } else {
                    let shl = self.emit_bin(BinaryOperator::ShiftLeft, l_raw, s16);
                    self.emit_bin(BinaryOperator::ShiftRight, shl, s16)
                };
                let r_raw = self.read_src_i32(src1, literal);
                let r = if (op_sel & 2) != 0 {
                    self.emit_bin(BinaryOperator::ShiftRight, r_raw, s16)
                } else {
                    let shl = self.emit_bin(BinaryOperator::ShiftLeft, r_raw, s16);
                    self.emit_bin(BinaryOperator::ShiftRight, shl, s16)
                };
                self.emit_int_cmp(cc, l, r)
            }
            // u16 comparisons: GFX10.3 (0xA9..=0xAE, 0xB9..=0xBE) and compat (0x68..=0x6F, 0x78..=0x7F)
            0x68..=0x6F | 0x78..=0x7F | 0xA9..=0xAE | 0xB9..=0xBE => {
                let cc = op & 0x7;
                let s16 = self.lit_u32(16);
                let mask16 = self.lit_u32(0xFFFF);
                let l_raw = self.read_src_u32(src0, literal);
                let l = if (op_sel & 1) != 0 {
                    self.emit_bin(BinaryOperator::ShiftRight, l_raw, s16)
                } else {
                    self.emit_bin(BinaryOperator::And, l_raw, mask16)
                };
                let r_raw = self.read_src_u32(src1, literal);
                let r = if (op_sel & 2) != 0 {
                    self.emit_bin(BinaryOperator::ShiftRight, r_raw, s16)
                } else {
                    self.emit_bin(BinaryOperator::And, r_raw, mask16)
                };
                self.emit_int_cmp(cc, l, r)
            }
            // i32 comparisons: 0x80..=0x87 / 0x90..=0x97
            0x80..=0x87 | 0x90..=0x97 => {
                let cc = op & 0x7;
                let l = self.read_src_i32(src0, literal);
                let r = self.read_src_i32(src1, literal);
                self.emit_int_cmp(cc, l, r)
            }
            // u32 comparisons: 0xC0..=0xC7 / 0xD0..=0xD7
            0xC0..=0xC7 | 0xD0..=0xD7 => {
                let cc = op & 0x7;
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                self.emit_int_cmp(cc, l, r)
            }
            // v_cmp_class_f32 / v_cmpx_class_f32: IEEE-754 class of src0 tested against the
            // src1 class mask (bits 0..9). Source modifiers act on the raw sign bit.
            0x88 | 0x98 => {
                let mut bits = self.read_src_u32(src0, literal);
                if (abs & 1) != 0 {
                    let clear_sign = self.lit_u32(0x7FFF_FFFF);
                    bits = self.emit_bin(BinaryOperator::And, bits, clear_sign);
                }
                if (neg & 1) != 0 {
                    let sign = self.lit_u32(0x8000_0000);
                    bits = self.emit_bin(BinaryOperator::ExclusiveOr, bits, sign);
                }
                let mask = self.read_src_u32(src1, literal);
                self.emit_f32_class_test(bits, mask)
            }
            // v_cmp_class_f16 / v_cmpx_class_f16: IEEE-754 binary16 class of src0 (respecting
            // op_sel[0], abs[0], neg[0]) tested against the 10-bit class mask in src1.
            0x8F | 0x9F => {
                let mut bits16 = if let Some(hbits) = Self::inline_f16_bits(src0) {
                    self.lit_u32(if (op_sel & 1) != 0 { 0 } else { hbits })
                } else {
                    let raw = self.read_src_u32(src0, literal);
                    if (op_sel & 1) != 0 {
                        let s16 = self.lit_u32(16);
                        self.emit_bin(BinaryOperator::ShiftRight, raw, s16)
                    } else {
                        let mask16 = self.lit_u32(0xFFFF);
                        self.emit_bin(BinaryOperator::And, raw, mask16)
                    }
                };
                if (abs & 1) != 0 {
                    let clear_sign = self.lit_u32(0x7FFF);
                    bits16 = self.emit_bin(BinaryOperator::And, bits16, clear_sign);
                }
                if (neg & 1) != 0 {
                    let sign = self.lit_u32(0x8000);
                    bits16 = self.emit_bin(BinaryOperator::ExclusiveOr, bits16, sign);
                }
                let mask = self.read_src_u32(src1, literal);
                self.emit_f16_class_test(bits16, mask)
            }
            _ => return false,
        };

        let is_cmpx = (op & 0x10) != 0;
        if is_cmpx {
            // GFX10.3 v_cmpx_* updates EXEC (single-lane model: exec_mask &= cmp_bit).
            if let Some(local) = self.exec_local {
                let old_exec = self.load_local(local);
                let cmp_bit = self.bool_to_u32(cmp);
                let new_exec = self.emit_bin(BinaryOperator::And, old_exec, cmp_bit);
                self.store_local(local, new_exec);
            }
            if let Some(sd) = sdst_override
                && (sd & 0x7F) <= 106
            {
                self.store_lane_mask(sd, cmp);
            }
        } else {
            match sdst_override {
                Some(sd) => self.store_lane_mask(sd, cmp),
                None => self.store_vcc(cmp),
            }
        }
        true
    }

    fn emit_int_cmp(
        &mut self,
        cc: u8,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        match cc {
            0 => self.lit_bool(false),
            1 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::Less,
                left,
                right,
            }),
            2 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::Equal,
                left,
                right,
            }),
            3 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::LessEqual,
                left,
                right,
            }),
            4 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::Greater,
                left,
                right,
            }),
            5 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::NotEqual,
                left,
                right,
            }),
            6 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left,
                right,
            }),
            _ => self.lit_bool(true),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_vop2(
        &mut self,
        op: u8,
        vdst: u8,
        src0: u16,
        src1: u16,
        cond_src: Option<u16>,
        literal: Option<u32>,
        omod: u8,
        clamp: bool,
        abs: u8,
        neg: u8,
        carry_sdst: Option<u8>,
    ) -> bool {
        let abs0 = (abs & 1) != 0;
        let neg0 = (neg & 1) != 0;
        let abs1 = (abs & 2) != 0;
        let neg1 = (neg & 2) != 0;

        match op {
            // v_cndmask_b32 vdst, src0, src1, cond (cond ? src1 : src0)
            // GFX10.3 VOP2 opcode 0x00 and 0x01 (and VOP3 0x100 / 0x101).
            0x00 | 0x01 => {
                let reject = self.read_src_f32(src0, literal, abs0, neg0);
                let accept = self.read_src_f32(src1, literal, abs1, neg1);
                let cond = match cond_src {
                    Some(106) | None => self.load_vcc(),
                    Some(csrc) => {
                        let cu = self.read_src_u32(csrc, literal);
                        let zero = self.lit_u32(0);
                        self.emit_expr(Expression::Binary {
                            op: BinaryOperator::NotEqual,
                            left: cu,
                            right: zero,
                        })
                    }
                };
                let res = self.emit_expr(Expression::Select {
                    condition: cond,
                    accept,
                    reject,
                });
                self.store_vgpr_f32(vdst, res);
                true
            }
            // v_dot2c_f32_f16: vdst = dot(unpack(src0), unpack(src1)) + vdst
            0x02 => {
                let a = self.read_src_pk_f16(src0, literal, false, true, neg0, neg0);
                let b = self.read_src_pk_f16(src1, literal, false, true, neg1, neg1);
                let acc = self.load_vgpr_f32(vdst);
                let d = self.emit_expr(Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                let sum = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: d,
                    right: acc,
                });
                let out = self.apply_float_out_mods(sum, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_add_f32
            0x03 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: l,
                    right: r,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_sub_f32
            0x04 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: l,
                    right: r,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_subrev_f32: src1 - src0
            0x05 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: r,
                    right: l,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_fmac_legacy_f32 / v_fmac_f32: vdst = fma(src0, src1, vdst)
            0x06 | 0x2B => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.load_vgpr_f32(vdst);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_mul_legacy_f32 / v_mul_f32
            0x07 | 0x08 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let prod = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: l,
                    right: r,
                });
                let res = if op == 0x07 {
                    let zero = self.lit_f32(0.0);
                    let l_zero = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Equal,
                        left: l,
                        right: zero,
                    });
                    let r_zero = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Equal,
                        left: r,
                        right: zero,
                    });
                    let either_zero = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::LogicalOr,
                        left: l_zero,
                        right: r_zero,
                    });
                    self.emit_expr(Expression::Select {
                        condition: either_zero,
                        accept: zero,
                        reject: prod,
                    })
                } else {
                    prod
                };
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_mul_i32_i24: sign-extend low 24 bits and multiply
            0x09 => {
                let l_raw = self.read_src_i32(src0, literal);
                let r_raw = self.read_src_i32(src1, literal);
                let sh8 = self.lit_u32(8);
                let l_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: l_raw,
                    right: sh8,
                });
                let l = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: l_sh,
                    right: sh8,
                });
                let r_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: r_raw,
                    right: sh8,
                });
                let r = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: r_sh,
                    right: sh8,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: l,
                    right: r,
                });
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_mul_u32_u24: mask low 24 bits and multiply
            0x0B => {
                let l_raw = self.read_src_u32(src0, literal);
                let r_raw = self.read_src_u32(src1, literal);
                let mask24 = self.lit_u32(0x00FF_FFFF);
                let l = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: l_raw,
                    right: mask24,
                });
                let r = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: r_raw,
                    right: mask24,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: l,
                    right: r,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_min_f32 / v_max_f32
            0x0F | 0x10 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let fun = if op == 0x0F {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_min_i32 / v_max_i32
            0x11 | 0x12 => {
                let l = self.read_src_i32(src0, literal);
                let r = self.read_src_i32(src1, literal);
                let fun = if op == 0x11 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_min_u32 / v_max_u32
            0x13 | 0x14 => {
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                let fun = if op == 0x13 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_mul_hi_i32_i24 / v_mul_hi_u32_u24: bits [47:32] of the 24x24-bit product
            0x0A | 0x0C => {
                let l_raw = self.read_src_u32(src0, literal);
                let r_raw = self.read_src_u32(src1, literal);
                let res = if op == 0x0C {
                    let mask24 = self.lit_u32(0x00FF_FFFF);
                    let l = self.emit_bin(BinaryOperator::And, l_raw, mask24);
                    let r = self.emit_bin(BinaryOperator::And, r_raw, mask24);
                    self.emit_mul_hi_u32(l, r)
                } else {
                    let sh8 = self.lit_u32(8);
                    let l_i = self.bitcast_u32_to_i32(l_raw);
                    let r_i = self.bitcast_u32_to_i32(r_raw);
                    let l_shl = self.emit_bin(BinaryOperator::ShiftLeft, l_i, sh8);
                    let r_shl = self.emit_bin(BinaryOperator::ShiftLeft, r_i, sh8);
                    let l_sext = self.emit_bin(BinaryOperator::ShiftRight, l_shl, sh8);
                    let r_sext = self.emit_bin(BinaryOperator::ShiftRight, r_shl, sh8);
                    let l = self.bitcast_i32_to_u32(l_sext);
                    let r = self.bitcast_i32_to_u32(r_sext);
                    self.emit_mul_hi_i32(l, r)
                };
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_lshrrev_b32 / v_lshlrev_b32: src1 >>/<< (src0 & 31)
            0x16 | 0x1A => {
                let sh_raw = self.read_src_u32(src0, literal);
                let val = self.read_src_u32(src1, literal);
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: sh_raw,
                    right: mask,
                });
                let bin = if op == 0x16 {
                    BinaryOperator::ShiftRight
                } else {
                    BinaryOperator::ShiftLeft
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: val,
                    right: sh,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_ashrrev_i32: i32(src1) >> (src0 & 31)
            0x18 => {
                let sh_raw = self.read_src_u32(src0, literal);
                let val = self.read_src_i32(src1, literal);
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: sh_raw,
                    right: mask,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: val,
                    right: sh,
                });
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_and_b32 / v_or_b32 / v_xor_b32
            0x1B..=0x1D => {
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                let bin = match op {
                    0x1B => BinaryOperator::And,
                    0x1C => BinaryOperator::InclusiveOr,
                    _ => BinaryOperator::ExclusiveOr,
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: r,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_xnor_b32: ~(src0 ^ src1)
            0x1E => {
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                let x = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ExclusiveOr,
                    left: l,
                    right: r,
                });
                let res = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: x,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_mac_f32 (GFX10.1): vdst = src0 * src1 + vdst (unfused)
            0x1F => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.load_vgpr_f32(vdst);
                let prod = self.emit_bin(BinaryOperator::Multiply, a, b);
                let res = self.emit_bin(BinaryOperator::Add, prod, c);
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_madmk_f32 (GFX10.1): vdst = src0 * K + src1 (unfused, mandatory K literal)
            // v_madak_f32 (GFX10.1): vdst = src0 * src1 + K (unfused, mandatory K literal)
            0x20 | 0x21 => {
                let a = self.read_src_f32(src0, None, abs0, neg0);
                let b = self.read_src_f32(src1, None, abs1, neg1);
                let k = self.lit_f32_from_bits(literal.unwrap_or(0));
                let res = if op == 0x20 {
                    let prod = self.emit_bin(BinaryOperator::Multiply, a, k);
                    self.emit_bin(BinaryOperator::Add, prod, b)
                } else {
                    let prod = self.emit_bin(BinaryOperator::Multiply, a, b);
                    self.emit_bin(BinaryOperator::Add, prod, k)
                };
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_add_co_ci_u32 / v_sub_co_ci_u32 / v_subrev_co_ci_u32:
            //   carry-in from VCC (VOP2) or src2 (VOP3B); carry/borrow-out to VCC or sdst.
            0x28..=0x2A => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let cin_bool = match cond_src {
                    Some(csrc) => self.read_lane_mask_bit(csrc, literal),
                    None => self.load_vcc(),
                };
                let cin = self.bool_to_u32(cin_bool);
                let (res, carry) = if op == 0x28 {
                    let s1 = self.emit_bin(BinaryOperator::Add, a, b);
                    let c1 = self.emit_bin(BinaryOperator::Less, s1, a);
                    let s2 = self.emit_bin(BinaryOperator::Add, s1, cin);
                    let c2 = self.emit_bin(BinaryOperator::Less, s2, s1);
                    let c = self.emit_bin(BinaryOperator::LogicalOr, c1, c2);
                    (s2, c)
                } else {
                    let (x, y) = if op == 0x29 { (a, b) } else { (b, a) };
                    let d1 = self.emit_bin(BinaryOperator::Subtract, x, y);
                    let b1 = self.emit_bin(BinaryOperator::Less, x, y);
                    let d2 = self.emit_bin(BinaryOperator::Subtract, d1, cin);
                    let b2 = self.emit_bin(BinaryOperator::Less, d1, cin);
                    let c = self.emit_bin(BinaryOperator::LogicalOr, b1, b2);
                    (d2, c)
                };
                self.store_vgpr_u32(vdst, res);
                self.store_lane_mask(carry_sdst.unwrap_or(106), carry);
                true
            }
            // v_add_nc_u32 / v_sub_nc_u32 / v_subrev_nc_u32
            0x25 | 0x26 | 0x27 => {
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                let (left, right, bin) = match op {
                    0x25 => (l, r, BinaryOperator::Add),
                    0x26 => (l, r, BinaryOperator::Subtract),
                    _ => (r, l, BinaryOperator::Subtract),
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left,
                    right,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_fmamk_f32: vdst = fma(src0, K, src1)
            0x2C => {
                let a = self.read_src_f32(src0, None, abs0, neg0);
                let k = self.lit_f32_from_bits(literal.unwrap_or(0));
                let c = self.read_src_f32(src1, None, abs1, neg1);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(k),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_fmaak_f32: vdst = fma(src0, src1, K)
            0x2D => {
                let a = self.read_src_f32(src0, None, abs0, neg0);
                let b = self.read_src_f32(src1, None, abs1, neg1);
                let k = self.lit_f32_from_bits(literal.unwrap_or(0));
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(k),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_cvt_pkrtz_f16_f32: pack2x16float(vec2<f32>(src0, src1))
            0x2F => {
                let a_raw = self.read_src_f32(src0, literal, abs0, neg0);
                let b_raw = self.read_src_f32(src1, literal, abs1, neg1);
                let a = self.apply_float_out_mods(a_raw, omod, clamp);
                let b = self.apply_float_out_mods(b_raw, omod, clamp);
                let pair = self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![a, b],
                });
                let packed = self.emit_expr(Expression::Math {
                    fun: MathFunction::Pack2x16float,
                    arg: pair,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_u32(vdst, packed);
                true
            }
            // v_add_f16 / v_sub_f16 / v_subrev_f16 / v_mul_f16 (f32 math + pack2x16float, no `enable f16;`)
            0x32..=0x35 => {
                let a = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, literal, true, false, abs1, neg1);
                let (left, right, bin) = match op {
                    0x32 => (a, b, BinaryOperator::Add),
                    0x33 => (a, b, BinaryOperator::Subtract),
                    0x34 => (b, a, BinaryOperator::Subtract),
                    _ => (a, b, BinaryOperator::Multiply),
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left,
                    right,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_fmac_f16: vdst.lo = fma(src0.lo, src1.lo, vdst.lo)
            0x36 => {
                let a = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, literal, true, false, abs1, neg1);
                let c = self.read_src_mixed_f16_or_f32(
                    256 + u16::from(vdst),
                    None,
                    true,
                    false,
                    false,
                    false,
                );
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_fmamk_f16: vdst.lo = fma(src0.lo, K_f16, src1.lo)
            0x37 => {
                let a = self.read_src_mixed_f16_or_f32(src0, None, true, false, abs0, neg0);
                let k = self.lit_f16_from_u32_lo(literal.unwrap_or(0));
                let c = self.read_src_mixed_f16_or_f32(src1, None, true, false, abs1, neg1);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(k),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_fmaak_f16: vdst.lo = fma(src0.lo, src1.lo, K_f16)
            0x38 => {
                let a = self.read_src_mixed_f16_or_f32(src0, None, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, None, true, false, abs1, neg1);
                let k = self.lit_f16_from_u32_lo(literal.unwrap_or(0));
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(k),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_max_f16 / v_min_f16
            0x39 | 0x3A => {
                let a = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, literal, true, false, abs1, neg1);
                let fun = if op == 0x39 {
                    MathFunction::Max
                } else {
                    MathFunction::Min
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_pk_fmac_f16: packed 2xf16 fma(src0, src1, vdst) via unpack2x16float/pack2x16float
            0x3C => {
                let a = self.read_src_pk_f16(src0, literal, false, true, neg0, neg0);
                let b = self.read_src_pk_f16(src1, literal, false, true, neg1, neg1);
                let c = self.read_src_pk_f16(
                    256 + u16::from(vdst),
                    None,
                    false,
                    true,
                    false,
                    false,
                );
                let mut res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                if clamp {
                    let z = self.lit_f32(0.0);
                    let o = self.lit_f32(1.0);
                    let lo = self.emit_expr(Expression::Splat {
                        size: VectorSize::Bi,
                        value: z,
                    });
                    let hi = self.emit_expr(Expression::Splat {
                        size: VectorSize::Bi,
                        value: o,
                    });
                    res = self.emit_expr(Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: res,
                        arg1: Some(lo),
                        arg2: Some(hi),
                        arg3: None,
                    });
                }
                let packed = self.emit_expr(Expression::Math {
                    fun: MathFunction::Pack2x16float,
                    arg: res,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_u32(vdst, packed);
                true
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_vop3(
        &mut self,
        op: u16,
        vdst: u8,
        sdst: Option<u8>,
        src0: u16,
        src1: u16,
        src2: u16,
        clamp: bool,
        omod: u8,
        abs: u8,
        neg: u8,
        op_sel: u8,
        literal: Option<u32>,
    ) -> bool {
        // VOPC promoted to VOP3 (0x000..=0x0FF)
        if op <= 0x0FF {
            return self.translate_vopc(
                op as u8,
                src0,
                src1,
                literal,
                abs,
                neg,
                op_sel,
                Some(vdst),
            );
        }
        // VOP2 promoted to VOP3 (0x100..=0x13F)
        if (0x100..=0x13F).contains(&op) {
            let v2_op = (op - 0x100) as u8;
            if vop2_is_madk(v2_op) {
                // *mk*/*ak* forms are VOP2-only (inline K literal); no VOP3 encoding exists.
                return false;
            }
            let cond = if matches!(v2_op, 0x00 | 0x01) || vop2_is_co_ci(v2_op) {
                Some(src2)
            } else {
                None
            };
            return self.translate_vop2(
                v2_op, vdst, src0, src1, cond, literal, omod, clamp, abs, neg, sdst,
            );
        }
        // VOP1 promoted to VOP3 (0x180..=0x1FF)
        if (0x180..=0x1FF).contains(&op) {
            let v1_op = (op - 0x180) as u8;
            return self.translate_vop1(
                v1_op,
                vdst,
                src0,
                literal,
                omod,
                clamp,
                (abs & 1) != 0,
                (neg & 1) != 0,
            );
        }

        let abs0 = (abs & 1) != 0;
        let neg0 = (neg & 1) != 0;
        let abs1 = (abs & 2) != 0;
        let neg1 = (neg & 2) != 0;
        let abs2 = (abs & 4) != 0;
        let neg2 = (neg & 4) != 0;

        match op {
            // v_mad_legacy_f32 / v_mad_f32 / v_fma_f32
            0x140 | 0x141 | 0x14B => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.read_src_f32(src2, literal, abs2, neg2);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_mad_i32_i24: sext24(src0) * sext24(src1) + src2
            0x142 => {
                let a_raw = self.read_src_i32(src0, literal);
                let b_raw = self.read_src_i32(src1, literal);
                let c = self.read_src_i32(src2, literal);
                let sh8 = self.lit_u32(8);
                let a_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: a_raw,
                    right: sh8,
                });
                let a = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: a_sh,
                    right: sh8,
                });
                let b_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: b_raw,
                    right: sh8,
                });
                let b = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: b_sh,
                    right: sh8,
                });
                let prod = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: prod,
                    right: c,
                });
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_mad_u32_u24: (src0 & 0xFFFFFF) * (src1 & 0xFFFFFF) + src2
            0x143 => {
                let a_raw = self.read_src_u32(src0, literal);
                let b_raw = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let mask24 = self.lit_u32(0x00FF_FFFF);
                let a = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: a_raw,
                    right: mask24,
                });
                let b = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: b_raw,
                    right: mask24,
                });
                let prod = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: prod,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_cubeid_f32 (0x144), v_cubesc_f32 (0x145), v_cubetc_f32 (0x146), v_cubema_f32 (0x147):
            // Standard AMD GFX10.3 cubemap face/coordinate math on (x = src0, y = src1, z = src2).
            0x144..=0x147 => {
                let x = self.read_src_f32(src0, literal, abs0, neg0);
                let y = self.read_src_f32(src1, literal, abs1, neg1);
                let z = self.read_src_f32(src2, literal, abs2, neg2);
                let ax = self.emit_expr(Expression::Math {
                    fun: MathFunction::Abs,
                    arg: x,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let ay = self.emit_expr(Expression::Math {
                    fun: MathFunction::Abs,
                    arg: y,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let az = self.emit_expr(Expression::Math {
                    fun: MathFunction::Abs,
                    arg: z,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let az_ge_ax = self.emit_bin(BinaryOperator::GreaterEqual, az, ax);
                let az_ge_ay = self.emit_bin(BinaryOperator::GreaterEqual, az, ay);
                let z_major = self.emit_bin(BinaryOperator::LogicalAnd, az_ge_ax, az_ge_ay);
                let y_major = self.emit_bin(BinaryOperator::GreaterEqual, ay, ax);
                let zero = self.lit_f32(0.0);
                let res = match op {
                    // v_cubeid_f32: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5
                    0x144 => {
                        let x_neg = self.emit_bin(BinaryOperator::Less, x, zero);
                        let y_neg = self.emit_bin(BinaryOperator::Less, y, zero);
                        let z_neg = self.emit_bin(BinaryOperator::Less, z, zero);
                        let f0 = self.lit_f32(0.0);
                        let f1 = self.lit_f32(1.0);
                        let f2 = self.lit_f32(2.0);
                        let f3 = self.lit_f32(3.0);
                        let f4 = self.lit_f32(4.0);
                        let f5 = self.lit_f32(5.0);
                        let x_id = self.emit_select(x_neg, f1, f0);
                        let y_id = self.emit_select(y_neg, f3, f2);
                        let z_id = self.emit_select(z_neg, f5, f4);
                        let xy_id = self.emit_select(y_major, y_id, x_id);
                        self.emit_select(z_major, z_id, xy_id)
                    }
                    // v_cubesc_f32: z_major -> (z<0 ? -x : x), y_major -> x, x_major -> (x<0 ? z : -z)
                    0x145 => {
                        let x_neg = self.emit_bin(BinaryOperator::Less, x, zero);
                        let z_neg = self.emit_bin(BinaryOperator::Less, z, zero);
                        let neg_x = self.emit_expr(Expression::Unary {
                            op: UnaryOperator::Negate,
                            expr: x,
                        });
                        let neg_z = self.emit_expr(Expression::Unary {
                            op: UnaryOperator::Negate,
                            expr: z,
                        });
                        let x_sc = self.emit_select(x_neg, z, neg_z);
                        let z_sc = self.emit_select(z_neg, neg_x, x);
                        let xy_sc = self.emit_select(y_major, x, x_sc);
                        self.emit_select(z_major, z_sc, xy_sc)
                    }
                    // v_cubetc_f32: z_major -> -y, y_major -> (y<0 ? -z : z), x_major -> -y
                    0x146 => {
                        let y_neg = self.emit_bin(BinaryOperator::Less, y, zero);
                        let neg_y = self.emit_expr(Expression::Unary {
                            op: UnaryOperator::Negate,
                            expr: y,
                        });
                        let neg_z = self.emit_expr(Expression::Unary {
                            op: UnaryOperator::Negate,
                            expr: z,
                        });
                        let y_tc = self.emit_select(y_neg, neg_z, z);
                        let xy_tc = self.emit_select(y_major, y_tc, neg_y);
                        self.emit_select(z_major, neg_y, xy_tc)
                    }
                    // v_cubema_f32: 2.0 * signed major axis
                    _ => {
                        let two = self.lit_f32(2.0);
                        let xy_ma = self.emit_select(y_major, y, x);
                        let ma = self.emit_select(z_major, z, xy_ma);
                        self.emit_bin(BinaryOperator::Multiply, two, ma)
                    }
                };
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_bfe_u32: unsigned bitfield extract (offset = src1[4:0], width = src2[4:0])
            0x148 => {
                let s0 = self.read_src_u32(src0, literal);
                let s1 = self.read_src_u32(src1, literal);
                let s2 = self.read_src_u32(src2, literal);
                let mask31 = self.lit_u32(31);
                let offset = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: s1,
                    right: mask31,
                });
                let width = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: s2,
                    right: mask31,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: s0,
                    right: offset,
                });
                let one = self.lit_u32(1);
                let one_shl = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: one,
                    right: width,
                });
                let bit_mask = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: one_shl,
                    right: one,
                });
                let extracted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: shifted,
                    right: bit_mask,
                });
                let zero = self.lit_u32(0);
                let is_zero = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Equal,
                    left: width,
                    right: zero,
                });
                let res = self.emit_expr(Expression::Select {
                    condition: is_zero,
                    accept: zero,
                    reject: extracted,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_bfi_b32: (src0 & src1) | (~src0 & src2)
            0x14A => {
                let mask = self.read_src_u32(src0, literal);
                let a = self.read_src_u32(src1, literal);
                let b = self.read_src_u32(src2, literal);
                let nmask = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: mask,
                });
                let lhs = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: mask,
                    right: a,
                });
                let rhs = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: nmask,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: lhs,
                    right: rhs,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_alignbit_b32: funnel shift right of (src0:src1) by (src2 & 31)
            0x14E => {
                let hi = self.read_src_u32(src0, literal);
                let lo = self.read_src_u32(src1, literal);
                let sh_raw = self.read_src_u32(src2, literal);
                let mask31 = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: sh_raw,
                    right: mask31,
                });
                let zero = self.lit_u32(0);
                let is_zero = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Equal,
                    left: sh,
                    right: zero,
                });
                let c32 = self.lit_u32(32);
                let inv_sh_raw = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: c32,
                    right: sh,
                });
                let inv_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: inv_sh_raw,
                    right: mask31,
                });
                let hi_part = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: hi,
                    right: inv_sh,
                });
                let lo_part = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: lo,
                    right: sh,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: hi_part,
                    right: lo_part,
                });
                let res = self.emit_expr(Expression::Select {
                    condition: is_zero,
                    accept: lo,
                    reject: shifted,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_min3_f32 / v_max3_f32
            0x151 | 0x154 => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.read_src_f32(src2, literal, abs2, neg2);
                let fun = if op == 0x151 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let ab = self.emit_expr(Expression::Math {
                    fun,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: ab,
                    arg1: Some(c),
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_med3_f32: clamp(a, min(b, c), max(b, c))
            0x157 => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.read_src_f32(src2, literal, abs2, neg2);
                let lo = self.emit_expr(Expression::Math {
                    fun: MathFunction::Min,
                    arg: b,
                    arg1: Some(c),
                    arg2: None,
                    arg3: None,
                });
                let hi = self.emit_expr(Expression::Math {
                    fun: MathFunction::Max,
                    arg: b,
                    arg1: Some(c),
                    arg2: None,
                    arg3: None,
                });
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Clamp,
                    arg: a,
                    arg1: Some(lo),
                    arg2: Some(hi),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_min3_i32 / v_max3_i32 / v_med3_i32
            0x152 | 0x155 | 0x158 => {
                let a = self.read_src_i32(src0, literal);
                let b = self.read_src_i32(src1, literal);
                let c = self.read_src_i32(src2, literal);
                let res = if op == 0x158 {
                    let lo = self.emit_expr(Expression::Math {
                        fun: MathFunction::Min,
                        arg: b,
                        arg1: Some(c),
                        arg2: None,
                        arg3: None,
                    });
                    let hi = self.emit_expr(Expression::Math {
                        fun: MathFunction::Max,
                        arg: b,
                        arg1: Some(c),
                        arg2: None,
                        arg3: None,
                    });
                    self.emit_expr(Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: a,
                        arg1: Some(lo),
                        arg2: Some(hi),
                        arg3: None,
                    })
                } else {
                    let fun = if op == 0x152 {
                        MathFunction::Min
                    } else {
                        MathFunction::Max
                    };
                    let ab = self.emit_expr(Expression::Math {
                        fun,
                        arg: a,
                        arg1: Some(b),
                        arg2: None,
                        arg3: None,
                    });
                    self.emit_expr(Expression::Math {
                        fun,
                        arg: ab,
                        arg1: Some(c),
                        arg2: None,
                        arg3: None,
                    })
                };
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_min3_u32 / v_max3_u32 / v_med3_u32
            0x153 | 0x156 | 0x159 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let res = if op == 0x159 {
                    let lo = self.emit_expr(Expression::Math {
                        fun: MathFunction::Min,
                        arg: b,
                        arg1: Some(c),
                        arg2: None,
                        arg3: None,
                    });
                    let hi = self.emit_expr(Expression::Math {
                        fun: MathFunction::Max,
                        arg: b,
                        arg1: Some(c),
                        arg2: None,
                        arg3: None,
                    });
                    self.emit_expr(Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: a,
                        arg1: Some(lo),
                        arg2: Some(hi),
                        arg3: None,
                    })
                } else {
                    let fun = if op == 0x153 {
                        MathFunction::Min
                    } else {
                        MathFunction::Max
                    };
                    let ab = self.emit_expr(Expression::Math {
                        fun,
                        arg: a,
                        arg1: Some(b),
                        arg2: None,
                        arg3: None,
                    });
                    self.emit_expr(Expression::Math {
                        fun,
                        arg: ab,
                        arg1: Some(c),
                        arg2: None,
                        arg3: None,
                    })
                };
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_cvt_pk_u8_f32: clamp f32 src0 to [0, 255], convert to u8, pack into byte (src1 & 3) of src2
            0x15E => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let lo = self.lit_f32(0.0);
                let hi = self.lit_f32(255.0);
                let clamped = self.emit_expr(Expression::Math {
                    fun: MathFunction::Clamp,
                    arg: f,
                    arg1: Some(lo),
                    arg2: Some(hi),
                    arg3: None,
                });
                let u_raw = self.emit_expr(Expression::As {
                    expr: clamped,
                    kind: ScalarKind::Uint,
                    convert: Some(4),
                });
                let mask_ff = self.lit_u32(0xFF);
                let byte_val = self.emit_bin(BinaryOperator::And, u_raw, mask_ff);
                let s1 = self.read_src_u32(src1, literal);
                let mask3 = self.lit_u32(3);
                let byte_idx = self.emit_bin(BinaryOperator::And, s1, mask3);
                let sh3 = self.lit_u32(3);
                let shift = self.emit_bin(BinaryOperator::ShiftLeft, byte_idx, sh3);
                let shifted_byte = self.emit_bin(BinaryOperator::ShiftLeft, byte_val, shift);
                let byte_mask = self.emit_bin(BinaryOperator::ShiftLeft, mask_ff, shift);
                let inv_mask = self.emit_bit_not(byte_mask);
                let s2 = self.read_src_u32(src2, literal);
                let cleared = self.emit_bin(BinaryOperator::And, s2, inv_mask);
                let res = self.emit_bin(BinaryOperator::InclusiveOr, cleared, shifted_byte);
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_mul_lo_u32
            0x169 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: a,
                    right: b,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_add_co_u32 / v_sub_co_u32 / v_subrev_co_u32 (VOP3B with carry-out in sdst)
            0x30F | 0x310 | 0x319 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let (res, carry) = match op {
                    0x30F => {
                        let sum = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Add,
                            left: a,
                            right: b,
                        });
                        let c = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Less,
                            left: sum,
                            right: a,
                        });
                        (sum, c)
                    }
                    0x310 => {
                        let diff = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Subtract,
                            left: a,
                            right: b,
                        });
                        let c = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Less,
                            left: a,
                            right: b,
                        });
                        (diff, c)
                    }
                    _ => {
                        let diff = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Subtract,
                            left: b,
                            right: a,
                        });
                        let c = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Less,
                            left: b,
                            right: a,
                        });
                        (diff, c)
                    }
                };
                self.store_vgpr_u32(vdst, res);
                if let Some(sd) = sdst {
                    self.store_lane_mask(sd, carry);
                }
                true
            }
            // v_mul_hi_u32 / v_mul_hi_i32: exact high 32 bits of the 32x32 product (u32-only WGSL)
            0x16A | 0x16C => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let res = if op == 0x16A {
                    self.emit_mul_hi_u32(a, b)
                } else {
                    self.emit_mul_hi_i32(a, b)
                };
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_xor3_b32: src0 ^ src1 ^ src2 (GFX10.3 VOP3 0x178)
            0x178 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let ab = self.emit_bin(BinaryOperator::ExclusiveOr, a, b);
                let res = self.emit_bin(BinaryOperator::ExclusiveOr, ab, c);
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_lshlrev_b64 / v_lshrrev_b64 / v_ashrrev_i64:
            //   vdst[0:1] = src1[0:1] <</>>/>>> (src0 & 63) on u32 register pairs
            0x2FF..=0x301 => {
                let amount = self.read_src_u32(src0, literal);
                let (lo, hi) = self.read_vsrc_pair(src1, literal);
                let kind = match op {
                    0x2FF => Shift64Kind::Left,
                    0x300 => Shift64Kind::LogicalRight,
                    _ => Shift64Kind::ArithmeticRight,
                };
                let (rlo, rhi) = self.emit_shift64(kind, lo, hi, amount);
                self.store_vgpr_u32(vdst, rlo);
                if vdst < 255 {
                    self.store_vgpr_u32(vdst + 1, rhi);
                }
                true
            }
            // v_interp_p1ll_f16 (0x342), v_interp_p1lv_f16 (0x343), v_interp_p2_f16 (0x35A):
            // GFX10.3 VOP3 16-bit fragment attribute interpolation.
            // src0 packs {high[8], attrchan[7:6], attr[5:0]}; op_sel[3] selects dst lo/hi f16 half on p2.
            0x342 | 0x343 | 0x35A => {
                if self.stage != ShaderStage::Fragment {
                    return false;
                }
                let Some(arg_idx) = self.frag_in_arg else {
                    return false;
                };
                let (attr, attrchan, _high) = decode_vop3_interp_attr(src0);
                let clamped_attr = attr % MAX_WEBGPU_LOCATIONS;
                let Some(&member_idx) = self.frag_attr_member_index.get(&clamped_attr) else {
                    return false;
                };
                let frag_in = self.raw_expr(Expression::FunctionArgument(arg_idx));
                let attr_vec = self.emit_expr(Expression::AccessIndex {
                    base: frag_in,
                    index: member_idx,
                });
                let comp = self.emit_expr(Expression::AccessIndex {
                    base: attr_vec,
                    index: u32::from(attrchan & 0x3),
                });
                let out = self.apply_float_out_mods(comp, omod, clamp);
                if op == 0x35A {
                    if (op_sel & 0x8) != 0 {
                        self.store_vgpr_f16_hi(vdst, out);
                    } else {
                        self.store_vgpr_f16_lo(vdst, out);
                    }
                } else {
                    self.store_vgpr_f32(vdst, out);
                }
                true
            }
            // v_xad_u32: (src0 ^ src1) + src2
            0x345 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let x = self.emit_bin(BinaryOperator::ExclusiveOr, a, b);
                let res = self.emit_bin(BinaryOperator::Add, x, c);
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_lshl_add_u32: (src0 << (src1 & 31)) + src2
            0x346 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: b,
                    right: mask,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: a,
                    right: sh,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: shifted,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_add_lshl_u32: (src0 + src1) << (src2 & 31)
            0x347 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let sum = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: a,
                    right: b,
                });
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: c,
                    right: mask,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: sum,
                    right: sh,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_fma_f16: vdst.lo = fma(src0.lo, src1.lo, src2.lo)
            0x34B => {
                let a = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, literal, true, false, abs1, neg1);
                let c = self.read_src_mixed_f16_or_f32(src2, literal, true, false, abs2, neg2);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_readlane_b32 sdst, vsrc0, ssrc1: read a lane of vsrc0 into scalar sdst.
            // Constant lanes read per-(vgpr, lane) spill locals when present (falling back to
            // v[vsrc0]); dynamic lanes select across known constant lane slots before fallback.
            0x360 => {
                let val = if (256..=511).contains(&src0) {
                    let vsrc = (src0 - 256) as u8;
                    if let Some(lane) = decode_const_lane_index(src1, literal) {
                        if let Some(&slot_local) = self.lane_spill_locals.get(&(vsrc, lane)) {
                            self.load_local(slot_local)
                        } else {
                            self.read_src_u32(src0, literal)
                        }
                    } else {
                        let mut acc = self.read_src_u32(src0, literal);
                        if let Some(slots) = self.vgpr_lane_slots.get(&vsrc).cloned() {
                            let raw_lane = self.read_src_u32(src1, literal);
                            let mask63 = self.lit_u32(63);
                            let lane_idx = self.emit_bin(BinaryOperator::And, raw_lane, mask63);
                            for (slot_lane, slot_local) in slots {
                                let target = self.lit_u32(u32::from(slot_lane));
                                let is_match =
                                    self.emit_bin(BinaryOperator::Equal, lane_idx, target);
                                let slot_val = self.load_local(slot_local);
                                acc = self.emit_select(is_match, slot_val, acc);
                            }
                        }
                        acc
                    }
                } else {
                    self.read_src_u32(src0, literal)
                };
                let sd = vdst & 0x7F;
                if sd == 106 {
                    let one = self.lit_u32(1);
                    let bit = self.emit_bin(BinaryOperator::And, val, one);
                    let zero = self.lit_u32(0);
                    let cond = self.emit_bin(BinaryOperator::NotEqual, bit, zero);
                    self.store_vcc(cond);
                } else {
                    self.store_sgpr_u32(sd, val);
                }
                true
            }
            // v_writelane_b32 vdst, ssrc0, ssrc1: write scalar data into a lane of vdst.
            // Constant lanes write dedicated per-(vgpr, lane) spill locals so multi-lane SGPR
            // spills into the same VGPR never clobber one another; lane 0 also mirrors to v[vdst].
            0x361 => {
                let val = self.read_src_u32(src0, literal);
                if let Some(lane) = decode_const_lane_index(src1, literal) {
                    if let Some(&slot_local) = self.lane_spill_locals.get(&(vdst, lane)) {
                        self.store_local(slot_local, val);
                    }
                    if lane == 0
                        && let Some(&v_local) = self.vgpr_locals.get(&vdst)
                    {
                        let f = self.bitcast_u32_to_f32(val);
                        self.store_local(v_local, f);
                    }
                } else if let Some(&v_local) = self.vgpr_locals.get(&vdst) {
                    let f = self.bitcast_u32_to_f32(val);
                    self.store_local(v_local, f);
                }
                true
            }
            // v_add_f64: f32-emulated addition into dst VGPR pair (low = f32 sum, high = 0)
            0x364 => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let sum = self.emit_bin(BinaryOperator::Add, a, b);
                let out = self.apply_float_out_mods(sum, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                if vdst < 255 {
                    let zero = self.lit_u32(0);
                    self.store_vgpr_u32(vdst + 1, zero);
                }
                true
            }
            // v_add3_u32
            0x36D => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let ab = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: ab,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_lshl_or_b32: (src0 << (src1 & 31)) | src2
            0x36E => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: b,
                    right: mask,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: a,
                    right: sh,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: shifted,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_and_or_b32: (src0 & src1) | src2 (GFX10.3 VOP3 0x371; 0x36F retained for compat)
            0x36F | 0x371 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let ab = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: ab,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_or3_b32: src0 | src1 | src2 (GFX10.3 VOP3 0x372; 0x370 retained for compat)
            0x370 | 0x372 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let ab = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: ab,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_vop3p(
        &mut self,
        op: u8,
        vdst: u8,
        src0: u16,
        src1: u16,
        src2: u16,
        clamp: bool,
        op_sel: u8,
        op_sel_hi: u8,
        neg_lo: u8,
        neg_hi: u8,
        literal: Option<u32>,
    ) -> bool {
        let read_pk = |this: &mut Self, s: u16, idx: u8| {
            let bit = 1u8 << idx;
            this.read_src_pk_f16(
                s,
                literal,
                (op_sel & bit) != 0,
                (op_sel_hi & bit) != 0,
                (neg_lo & bit) != 0,
                (neg_hi & bit) != 0,
            )
        };
        let pack_and_store = |this: &mut Self, mut vec2_val: Handle<Expression>| {
            if clamp {
                let z = this.lit_f32(0.0);
                let o = this.lit_f32(1.0);
                let lo = this.emit_expr(Expression::Splat {
                    size: VectorSize::Bi,
                    value: z,
                });
                let hi = this.emit_expr(Expression::Splat {
                    size: VectorSize::Bi,
                    value: o,
                });
                vec2_val = this.emit_expr(Expression::Math {
                    fun: MathFunction::Clamp,
                    arg: vec2_val,
                    arg1: Some(lo),
                    arg2: Some(hi),
                    arg3: None,
                });
            }
            let packed = this.emit_expr(Expression::Math {
                fun: MathFunction::Pack2x16float,
                arg: vec2_val,
                arg1: None,
                arg2: None,
                arg3: None,
            });
            this.store_vgpr_u32(vdst, packed);
        };

        match op {
            // v_pk_fma_f16
            0x0E => {
                let a = read_pk(self, src0, 0);
                let b = read_pk(self, src1, 1);
                let c = read_pk(self, src2, 2);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                pack_and_store(self, res);
                true
            }
            // v_pk_add_f16 / v_pk_mul_f16
            0x0F | 0x10 => {
                let a = read_pk(self, src0, 0);
                let b = read_pk(self, src1, 1);
                let bin = if op == 0x0F {
                    BinaryOperator::Add
                } else {
                    BinaryOperator::Multiply
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: a,
                    right: b,
                });
                pack_and_store(self, res);
                true
            }
            // v_pk_min_f16 / v_pk_max_f16
            0x11 | 0x12 => {
                let a = read_pk(self, src0, 0);
                let b = read_pk(self, src1, 1);
                let fun = if op == 0x11 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                pack_and_store(self, res);
                true
            }
            // v_dot2_f32_f16: dot(unpack(src0), unpack(src1)) + src2
            0x13 => {
                let a = read_pk(self, src0, 0);
                let b = read_pk(self, src1, 1);
                let c = self.read_src_f32(src2, literal, false, (neg_lo & 4) != 0);
                let d = self.emit_expr(Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                let sum = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: d,
                    right: c,
                });
                let out = self.apply_float_out_mods(sum, 0, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_fma_mix_f32 (0x20), v_fma_mixlo_f16 (0x21), v_fma_mixhi_f16 (0x22):
            //   op_sel_hi[i]: 0 = f32 source, 1 = f16 source
            //   op_sel[i]:    when f16 source, 0 = lo 16 bits, 1 = hi 16 bits
            //   neg_hi[i]:    abs modifier
            //   neg_lo[i]:    neg modifier
            0x20..=0x22 => {
                let a = self.read_src_mixed_f16_or_f32(
                    src0,
                    literal,
                    (op_sel_hi & 1) != 0,
                    (op_sel & 1) != 0,
                    (neg_hi & 1) != 0,
                    (neg_lo & 1) != 0,
                );
                let b = self.read_src_mixed_f16_or_f32(
                    src1,
                    literal,
                    (op_sel_hi & 2) != 0,
                    (op_sel & 2) != 0,
                    (neg_hi & 2) != 0,
                    (neg_lo & 2) != 0,
                );
                let c = self.read_src_mixed_f16_or_f32(
                    src2,
                    literal,
                    (op_sel_hi & 4) != 0,
                    (op_sel & 4) != 0,
                    (neg_hi & 4) != 0,
                    (neg_lo & 4) != 0,
                );
                let fma_res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(fma_res, 0, clamp);
                match op {
                    0x20 => self.store_vgpr_f32(vdst, out),
                    0x21 => self.store_vgpr_f16_lo(vdst, out),
                    _ => self.store_vgpr_f16_hi(vdst, out),
                }
                true
            }
            _ => false,
        }
    }

    // -- VINTRP (Fragment attribute interpolation) --

    fn translate_vintrp(&mut self, op: u8, vdst: u8, attr: u8, attrchan: u8) -> bool {
        if !matches!(op, 0..=2) || self.stage != ShaderStage::Fragment {
            return false;
        }
        let Some(arg_idx) = self.frag_in_arg else {
            return false;
        };
        let clamped_attr = attr % MAX_WEBGPU_LOCATIONS;
        let Some(&member_idx) = self.frag_attr_member_index.get(&clamped_attr) else {
            return false;
        };
        let frag_in = self.raw_expr(Expression::FunctionArgument(arg_idx));
        let attr_vec = self.emit_expr(Expression::AccessIndex {
            base: frag_in,
            index: member_idx,
        });
        let comp = self.emit_expr(Expression::AccessIndex {
            base: attr_vec,
            index: u32::from(attrchan & 0x3),
        });
        self.store_vgpr_f32(vdst, comp);
        true
    }

    // -- MIMG (2D / 2D-Array / 3D Texture sampling, loading & storing) --

    #[allow(clippy::too_many_arguments)]
    fn translate_mimg(
        &mut self,
        inst_idx: usize,
        op: u8,
        dmask: u8,
        dim: u8,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        ssamp: u8,
    ) -> bool {
        let Some(layout) = mimg_addr_layout(op) else {
            return false;
        };
        if layout.is_get_lod {
            // WGSL has no textureQueryLod builtin; emit safe mip 0.0 fallback into dst VGPRs per dmask.
            let eff_mask = if dmask == 0 { 0xF } else { dmask };
            let mut dst_offset = 0u8;
            for ch in 0..4u32 {
                if (eff_mask & (1 << ch)) != 0 {
                    let zero = self.lit_f32(0.0);
                    self.store_vgpr_f32(vdata.saturating_add(dst_offset), zero);
                    dst_offset = dst_offset.saturating_add(1);
                }
            }
            return true;
        }
        let inst_dim = TexDim::from_mimg_dim(dim);

        if layout.is_store {
            if self.stage == ShaderStage::Vertex {
                return false;
            }
            let key = self
                .inst_storage_tex_keys
                .get(&inst_idx)
                .copied()
                .unwrap_or(ResourceKey::RawSlot(srsrc));
            let Some(&(uav_tex_global, tex_dim)) = self
                .storage_tex_globals
                .get(&key)
                .or_else(|| self.storage_tex_globals.values().next())
            else {
                return false;
            };
            let image = self.raw_expr(Expression::GlobalVariable(uav_tex_global));
            let ux = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off));
            let uy = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 1));
            let ix = self.bitcast_f32_to_i32(ux);
            let iy = self.bitcast_f32_to_i32(uy);
            let (coord, array_index) = match tex_dim {
                TexDim::D3 => {
                    let iz = if inst_dim == TexDim::D3 {
                        let uz = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 2));
                        self.bitcast_f32_to_i32(uz)
                    } else {
                        self.lit_i32(0)
                    };
                    (
                        self.emit_expr(Expression::Compose {
                            ty: self.types.vec3_i32_ty,
                            components: vec![ix, iy, iz],
                        }),
                        None,
                    )
                }
                TexDim::D2Array => {
                    let arr = if inst_dim == TexDim::D2Array {
                        let s = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 2));
                        self.bitcast_f32_to_i32(s)
                    } else {
                        self.lit_i32(0)
                    };
                    (
                        self.emit_expr(Expression::Compose {
                            ty: self.types.vec2_i32_ty,
                            components: vec![ix, iy],
                        }),
                        Some(arr),
                    )
                }
                TexDim::D2 => (
                    self.emit_expr(Expression::Compose {
                        ty: self.types.vec2_i32_ty,
                        components: vec![ix, iy],
                    }),
                    None,
                ),
            };

            let eff_mask = if dmask == 0 { 0xF } else { dmask };
            let mut src_offset = 0u8;
            let mut comps = Vec::with_capacity(4);
            for ch in 0..4u32 {
                if (eff_mask & (1 << ch)) != 0 {
                    comps.push(self.load_vgpr_f32(vdata.saturating_add(src_offset)));
                    src_offset = src_offset.saturating_add(1);
                } else {
                    comps.push(self.lit_f32(if ch == 3 { 1.0 } else { 0.0 }));
                }
            }
            let val = self.emit_expr(Expression::Compose {
                ty: self.types.vec4_f32_ty,
                components: comps,
            });
            self.func.body.push(
                Statement::ImageStore {
                    image,
                    coordinate: coord,
                    array_index,
                    value: val,
                },
                Span::UNDEFINED,
            );
            return true;
        }

        let key = self
            .inst_tex_keys
            .get(&inst_idx)
            .copied()
            .unwrap_or(ResourceKey::RawSlot(srsrc));
        let Some(&(tex_global, tex_dim)) = self
            .tex_globals
            .get(&key)
            .or_else(|| self.tex_globals.values().next())
        else {
            return false;
        };
        let image = self.raw_expr(Expression::GlobalVariable(tex_global));
        let lod_shift = inst_dim.extra_coord_vgprs();

        let sampled = if layout.is_load {
            // image_load / image_load_mip
            let ux = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off));
            let uy = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 1));
            let ix = self.bitcast_f32_to_i32(ux);
            let iy = self.bitcast_f32_to_i32(uy);
            let (coord, array_index) = match tex_dim {
                TexDim::D3 => {
                    let iz = if inst_dim == TexDim::D3 {
                        let uz = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 2));
                        self.bitcast_f32_to_i32(uz)
                    } else {
                        self.lit_i32(0)
                    };
                    (
                        self.emit_expr(Expression::Compose {
                            ty: self.types.vec3_i32_ty,
                            components: vec![ix, iy, iz],
                        }),
                        None,
                    )
                }
                TexDim::D2Array => {
                    let arr = if inst_dim == TexDim::D2Array {
                        let s = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 2));
                        self.bitcast_f32_to_i32(s)
                    } else {
                        self.lit_i32(0)
                    };
                    (
                        self.emit_expr(Expression::Compose {
                            ty: self.types.vec2_i32_ty,
                            components: vec![ix, iy],
                        }),
                        Some(arr),
                    )
                }
                TexDim::D2 => (
                    self.emit_expr(Expression::Compose {
                        ty: self.types.vec2_i32_ty,
                        components: vec![ix, iy],
                    }),
                    None,
                ),
            };
            let level = if let Some(lod_off) = layout.lod_off {
                let ml = self.load_vgpr_f32(vaddr.saturating_add(lod_off.saturating_add(lod_shift)));
                Some(self.bitcast_f32_to_i32(ml))
            } else {
                Some(self.lit_i32(0))
            };
            self.emit_expr(Expression::ImageLoad {
                image,
                coordinate: coord,
                array_index,
                sample: None,
                level,
            })
        } else {
            // image_sample* (sampled 2D, 2D-array, or 3D texture)
            let skey = self
                .inst_samp_keys
                .get(&inst_idx)
                .copied()
                .unwrap_or(ResourceKey::RawSlot(ssamp));
            let Some(&samp_global) = self
                .samp_globals
                .get(&skey)
                .or_else(|| self.samp_globals.values().next())
            else {
                return false;
            };
            let sampler = self.raw_expr(Expression::GlobalVariable(samp_global));
            let u = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off));
            let v = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 1));
            let (coord, array_index) = match tex_dim {
                TexDim::D3 => {
                    let w = if inst_dim == TexDim::D3 {
                        self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 2))
                    } else {
                        self.lit_f32(0.0)
                    };
                    (
                        self.emit_expr(Expression::Compose {
                            ty: self.types.vec3_f32_ty,
                            components: vec![u, v, w],
                        }),
                        None,
                    )
                }
                TexDim::D2Array => {
                    let arr = if inst_dim == TexDim::D2Array {
                        let s = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 2));
                        self.emit_expr(Expression::As {
                            expr: s,
                            kind: ScalarKind::Sint,
                            convert: Some(4),
                        })
                    } else {
                        self.lit_i32(0)
                    };
                    (
                        self.emit_expr(Expression::Compose {
                            ty: self.types.vec2_f32_ty,
                            components: vec![u, v],
                        }),
                        Some(arr),
                    )
                }
                TexDim::D2 => (
                    self.emit_expr(Expression::Compose {
                        ty: self.types.vec2_f32_ty,
                        components: vec![u, v],
                    }),
                    None,
                ),
            };

            // WGSL only permits implicit-derivative `textureSample` / `textureSampleBias`
            // in Fragment shaders; Vertex/Compute stages must use explicit `textureSampleLevel`
            // or `textureSampleGrad`.
            let level = if layout.is_zero_lod {
                SampleLevel::Exact(self.lit_f32(0.0))
            } else if let Some(lod_off) = layout.lod_off {
                let lod =
                    self.load_vgpr_f32(vaddr.saturating_add(lod_off.saturating_add(lod_shift)));
                SampleLevel::Exact(lod)
            } else if let Some(grad_off) = layout.grad_off {
                if tex_dim == TexDim::D3 {
                    let dudx = self.load_vgpr_f32(vaddr.saturating_add(grad_off));
                    let dvdx = self.load_vgpr_f32(vaddr.saturating_add(grad_off + 1));
                    let dwdx = if inst_dim == TexDim::D3 {
                        self.load_vgpr_f32(vaddr.saturating_add(grad_off + 2))
                    } else {
                        self.lit_f32(0.0)
                    };
                    let dy_base = if inst_dim == TexDim::D3 { 3 } else { 2 };
                    let dudy = self.load_vgpr_f32(vaddr.saturating_add(grad_off + dy_base));
                    let dvdy = self.load_vgpr_f32(vaddr.saturating_add(grad_off + dy_base + 1));
                    let dwdy = if inst_dim == TexDim::D3 {
                        self.load_vgpr_f32(vaddr.saturating_add(grad_off + dy_base + 2))
                    } else {
                        self.lit_f32(0.0)
                    };
                    let x = self.emit_expr(Expression::Compose {
                        ty: self.types.vec3_f32_ty,
                        components: vec![dudx, dvdx, dwdx],
                    });
                    let y = self.emit_expr(Expression::Compose {
                        ty: self.types.vec3_f32_ty,
                        components: vec![dudy, dvdy, dwdy],
                    });
                    SampleLevel::Gradient { x, y }
                } else {
                    let dudx = self.load_vgpr_f32(vaddr.saturating_add(grad_off));
                    let dvdx = self.load_vgpr_f32(vaddr.saturating_add(grad_off + 1));
                    let dudy = self.load_vgpr_f32(vaddr.saturating_add(grad_off + 2));
                    let dvdy = self.load_vgpr_f32(vaddr.saturating_add(grad_off + 3));
                    let x = self.emit_expr(Expression::Compose {
                        ty: self.types.vec2_f32_ty,
                        components: vec![dudx, dvdx],
                    });
                    let y = self.emit_expr(Expression::Compose {
                        ty: self.types.vec2_f32_ty,
                        components: vec![dudy, dvdy],
                    });
                    SampleLevel::Gradient { x, y }
                }
            } else if let Some(bias_off) = layout.bias_off {
                // Implicit-derivative sampling requires uniform control flow in WGSL.
                if self.stage == ShaderStage::Fragment && !self.in_nonuniform_cf() {
                    let bias = self.load_vgpr_f32(vaddr.saturating_add(bias_off));
                    SampleLevel::Bias(bias)
                } else {
                    SampleLevel::Exact(self.lit_f32(0.0))
                }
            } else if self.stage == ShaderStage::Fragment && !self.in_nonuniform_cf() {
                SampleLevel::Auto
            } else {
                SampleLevel::Exact(self.lit_f32(0.0))
            };

            let raw_sample = self.emit_expr(Expression::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate: coord,
                array_index,
                offset: None,
                level,
                depth_ref: None,
                clamp_to_edge: false,
            });

            if let Some(cmp_off) = layout.compare_off {
                // Depth/shadow comparison on baseline `texture_2d<f32>` + `sampler`:
                // returns 1.0 when `z_ref <= tex.x`, else 0.0, replicated across channels.
                let z_ref = self.load_vgpr_f32(vaddr.saturating_add(cmp_off));
                let depth_val = self.emit_expr(Expression::AccessIndex {
                    base: raw_sample,
                    index: 0,
                });
                let passed = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::LessEqual,
                    left: z_ref,
                    right: depth_val,
                });
                let one = self.lit_f32(1.0);
                let zero = self.lit_f32(0.0);
                let cmp_f = self.emit_expr(Expression::Select {
                    condition: passed,
                    accept: one,
                    reject: zero,
                });
                self.emit_expr(Expression::Splat {
                    size: VectorSize::Quad,
                    value: cmp_f,
                })
            } else {
                raw_sample
            }
        };

        let eff_mask = if dmask == 0 { 0xF } else { dmask };
        let mut dst_offset = 0u8;
        for ch in 0..4u32 {
            if (eff_mask & (1 << ch)) != 0 {
                let comp = self.emit_expr(Expression::AccessIndex {
                    base: sampled,
                    index: ch,
                });
                self.store_vgpr_f32(vdata.saturating_add(dst_offset), comp);
                dst_offset = dst_offset.saturating_add(1);
            }
        }
        true
    }

    // -- MUBUF / MTBUF vertex attribute fetch & storage buffer load/store helpers --

    #[allow(clippy::too_many_arguments)]
    fn translate_mubuf_or_mtbuf(
        &mut self,
        inst_idx: usize,
        is_mtbuf: bool,
        op: u8,
        format: u8,
        byte_offset: u16,
        offen: bool,
        idxen: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
    ) -> bool {
        if self.stage == ShaderStage::Vertex && !self.vertex_storage_buf_insts.contains(&inst_idx) {
            let vb_slot = self.vertex_fetch_vb_slots.get(&inst_idx).copied().unwrap_or(0);
            return self.translate_vertex_buf_fetch(
                vb_slot,
                is_mtbuf,
                op,
                format,
                byte_offset,
                vdata,
            );
        }
        self.translate_storage_buf_op(
            inst_idx,
            is_mtbuf,
            op,
            byte_offset,
            offen,
            idxen,
            vaddr,
            vdata,
            srsrc,
            soffset,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_storage_buf_op(
        &mut self,
        inst_idx: usize,
        is_mtbuf: bool,
        op: u8,
        byte_offset: u16,
        offen: bool,
        idxen: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
    ) -> bool {
        let Some(buf_op) = (if is_mtbuf {
            mtbuf_storage_op(op)
        } else {
            mubuf_storage_op(op)
        }) else {
            return false;
        };
        let key = self
            .inst_storage_buf_keys
            .get(&inst_idx)
            .copied()
            .unwrap_or(ResourceKey::RawSlot(srsrc));
        let Some(&(buf_global, writable)) = self
            .storage_buf_globals
            .get(&key)
            .or_else(|| self.storage_buf_globals.values().next())
        else {
            return false;
        };
        if buf_op.is_store() && !writable {
            return false;
        }

        // Compute effective byte address: vaddr_contrib + soffset + imm_offset
        let mut byte_addr: Option<Handle<Expression>> = None;
        let stride = buf_op.elem_stride_bytes();
        if offen && idxen {
            let idx_f = self.load_vgpr_f32(vaddr);
            let idx_u = self.bitcast_f32_to_u32(idx_f);
            let stride_lit = self.lit_u32(stride);
            let idx_bytes = self.emit_expr(Expression::Binary {
                op: BinaryOperator::Multiply,
                left: idx_u,
                right: stride_lit,
            });
            let off_f = self.load_vgpr_f32(vaddr.saturating_add(1));
            let off_u = self.bitcast_f32_to_u32(off_f);
            let sum = self.emit_expr(Expression::Binary {
                op: BinaryOperator::Add,
                left: idx_bytes,
                right: off_u,
            });
            byte_addr = Some(sum);
        } else if offen {
            let off_f = self.load_vgpr_f32(vaddr);
            byte_addr = Some(self.bitcast_f32_to_u32(off_f));
        } else if idxen {
            let idx_f = self.load_vgpr_f32(vaddr);
            let idx_u = self.bitcast_f32_to_u32(idx_f);
            if stride == 1 {
                byte_addr = Some(idx_u);
            } else {
                let stride_lit = self.lit_u32(stride);
                byte_addr = Some(self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: idx_u,
                    right: stride_lit,
                }));
            }
        }

        let soff_expr = if soffset <= 105 {
            Some(self.load_sgpr_u32(soffset))
        } else if (129..=208).contains(&soffset) {
            Some(self.read_src_u32(u16::from(soffset), None))
        } else {
            None
        };
        if let Some(se) = soff_expr {
            byte_addr = Some(match byte_addr {
                Some(cur) => self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: cur,
                    right: se,
                }),
                None => se,
            });
        }

        if byte_offset != 0 {
            let imm = self.lit_u32(u32::from(byte_offset));
            byte_addr = Some(match byte_addr {
                Some(cur) => self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: cur,
                    right: imm,
                }),
                None => imm,
            });
        }

        let eff_byte_addr = byte_addr.unwrap_or_else(|| self.lit_u32(0));
        let two = self.lit_u32(2);
        let base_dword = self.emit_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: eff_byte_addr,
            right: two,
        });

        let buf_ptr = self.raw_expr(Expression::GlobalVariable(buf_global));
        let buf_len = self.emit_expr(Expression::ArrayLength(buf_ptr));
        let one = self.lit_u32(1);
        let clamped_len = self.emit_expr(Expression::Math {
            fun: MathFunction::Max,
            arg: buf_len,
            arg1: Some(one),
            arg2: None,
            arg3: None,
        });
        let max_idx = self.emit_expr(Expression::Binary {
            op: BinaryOperator::Subtract,
            left: clamped_len,
            right: one,
        });

        match buf_op {
            StorageBufOpKind::LoadUbyte | StorageBufOpKind::LoadSbyte => {
                let safe_idx = self.emit_expr(Expression::Math {
                    fun: MathFunction::Min,
                    arg: base_dword,
                    arg1: Some(max_idx),
                    arg2: None,
                    arg3: None,
                });
                let elem_ptr = self.emit_expr(Expression::Access {
                    base: buf_ptr,
                    index: safe_idx,
                });
                let word = self.emit_expr(Expression::Load { pointer: elem_ptr });
                let mask3 = self.lit_u32(3);
                let byte_lane = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: eff_byte_addr,
                    right: mask3,
                });
                let three = self.lit_u32(3);
                let bit_shift = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: byte_lane,
                    right: three,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: word,
                    right: bit_shift,
                });
                if buf_op == StorageBufOpKind::LoadUbyte {
                    let mask_ff = self.lit_u32(0xFF);
                    let val = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: shifted,
                        right: mask_ff,
                    });
                    self.store_vgpr_u32(vdata, val);
                } else {
                    let s_raw = self.bitcast_u32_to_i32(shifted);
                    let sh24 = self.lit_u32(24);
                    let shl = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: s_raw,
                        right: sh24,
                    });
                    let sext = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftRight,
                        left: shl,
                        right: sh24,
                    });
                    self.store_vgpr_i32(vdata, sext);
                }
                true
            }
            StorageBufOpKind::LoadUshort | StorageBufOpKind::LoadSshort => {
                let safe_idx = self.emit_expr(Expression::Math {
                    fun: MathFunction::Min,
                    arg: base_dword,
                    arg1: Some(max_idx),
                    arg2: None,
                    arg3: None,
                });
                let elem_ptr = self.emit_expr(Expression::Access {
                    base: buf_ptr,
                    index: safe_idx,
                });
                let word = self.emit_expr(Expression::Load { pointer: elem_ptr });
                let mask2 = self.lit_u32(2);
                let half_lane = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: eff_byte_addr,
                    right: mask2,
                });
                let three = self.lit_u32(3);
                let bit_shift = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: half_lane,
                    right: three,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: word,
                    right: bit_shift,
                });
                if buf_op == StorageBufOpKind::LoadUshort {
                    let mask_ffff = self.lit_u32(0xFFFF);
                    let val = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: shifted,
                        right: mask_ffff,
                    });
                    self.store_vgpr_u32(vdata, val);
                } else {
                    let s_raw = self.bitcast_u32_to_i32(shifted);
                    let sh16 = self.lit_u32(16);
                    let shl = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: s_raw,
                        right: sh16,
                    });
                    let sext = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftRight,
                        left: shl,
                        right: sh16,
                    });
                    self.store_vgpr_i32(vdata, sext);
                }
                true
            }
            StorageBufOpKind::LoadDwords(comps) => {
                let mut loaded = Vec::with_capacity(usize::from(comps));
                for k in 0..comps {
                    let idx_k = if k == 0 {
                        base_dword
                    } else {
                        let k_lit = self.lit_u32(u32::from(k));
                        self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Add,
                            left: base_dword,
                            right: k_lit,
                        })
                    };
                    let safe_idx = self.emit_expr(Expression::Math {
                        fun: MathFunction::Min,
                        arg: idx_k,
                        arg1: Some(max_idx),
                        arg2: None,
                        arg3: None,
                    });
                    let elem_ptr = self.emit_expr(Expression::Access {
                        base: buf_ptr,
                        index: safe_idx,
                    });
                    loaded.push(self.emit_expr(Expression::Load { pointer: elem_ptr }));
                }
                for (k, word) in loaded.into_iter().enumerate() {
                    self.store_vgpr_u32(vdata.saturating_add(k as u8), word);
                }
                true
            }
            StorageBufOpKind::StoreDwords(comps) => {
                for k in 0..comps {
                    let idx_k = if k == 0 {
                        base_dword
                    } else {
                        let k_lit = self.lit_u32(u32::from(k));
                        self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Add,
                            left: base_dword,
                            right: k_lit,
                        })
                    };
                    let safe_idx = self.emit_expr(Expression::Math {
                        fun: MathFunction::Min,
                        arg: idx_k,
                        arg1: Some(max_idx),
                        arg2: None,
                        arg3: None,
                    });
                    let elem_ptr = self.emit_expr(Expression::Access {
                        base: buf_ptr,
                        index: safe_idx,
                    });
                    let vf = self.load_vgpr_f32(vdata.saturating_add(k));
                    let vu = self.bitcast_f32_to_u32(vf);
                    self.func.body.push(
                        Statement::Store {
                            pointer: elem_ptr,
                            value: vu,
                        },
                        Span::UNDEFINED,
                    );
                }
                true
            }
        }
    }

    // -- DS (LDS Workgroup Shared Memory) --

    fn lds_masked_dword_index(
        &mut self,
        base_addr_u32: Handle<Expression>,
        byte_offset: u32,
    ) -> Handle<Expression> {
        let eff_addr = if byte_offset == 0 {
            base_addr_u32
        } else {
            let off_lit = self.lit_u32(byte_offset);
            self.emit_expr(Expression::Binary {
                op: BinaryOperator::Add,
                left: base_addr_u32,
                right: off_lit,
            })
        };
        let two = self.lit_u32(2);
        let dword_idx = self.emit_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: eff_addr,
            right: two,
        });
        let mask = self.lit_u32(LDS_WORKGROUP_INDEX_MASK);
        self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: dword_idx,
            right: mask,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_ds(
        &mut self,
        op: u8,
        offset0: u8,
        offset1: u8,
        addr: u8,
        data0: u8,
        data1: u8,
        vdst: u8,
    ) -> bool {
        if self.stage != ShaderStage::Compute {
            return false;
        }
        let Some(lds_global) = self.lds_global else {
            return false;
        };
        let Some(lds_op) = ds_lds_op(op) else {
            return false;
        };

        let lds_ptr = self.raw_expr(Expression::GlobalVariable(lds_global));
        let addr_f = self.load_vgpr_f32(addr);
        let addr_u = self.bitcast_f32_to_u32(addr_f);

        match lds_op {
            DsLdsOp::ReadContiguous(n) => {
                let imm16 = u32::from(offset0) | (u32::from(offset1) << 8);
                let mut loaded = Vec::with_capacity(usize::from(n));
                for k in 0..n {
                    let idx = self.lds_masked_dword_index(addr_u, imm16.saturating_add(u32::from(k) * 4));
                    let elem_ptr = self.emit_expr(Expression::Access {
                        base: lds_ptr,
                        index: idx,
                    });
                    loaded.push(self.emit_expr(Expression::Load { pointer: elem_ptr }));
                }
                for (k, word) in loaded.into_iter().enumerate() {
                    self.store_vgpr_u32(vdst.saturating_add(k as u8), word);
                }
                true
            }
            DsLdsOp::WriteContiguous(n) => {
                let imm16 = u32::from(offset0) | (u32::from(offset1) << 8);
                for k in 0..n {
                    let idx = self.lds_masked_dword_index(addr_u, imm16.saturating_add(u32::from(k) * 4));
                    let elem_ptr = self.emit_expr(Expression::Access {
                        base: lds_ptr,
                        index: idx,
                    });
                    let vf = self.load_vgpr_f32(data0.saturating_add(k));
                    let vu = self.bitcast_f32_to_u32(vf);
                    self.func.body.push(
                        Statement::Store {
                            pointer: elem_ptr,
                            value: vu,
                        },
                        Span::UNDEFINED,
                    );
                }
                true
            }
            DsLdsOp::Read2 { stride_bytes } => {
                let idx0 = self.lds_masked_dword_index(addr_u, u32::from(offset0) * stride_bytes);
                let ptr0 = self.emit_expr(Expression::Access {
                    base: lds_ptr,
                    index: idx0,
                });
                let w0 = self.emit_expr(Expression::Load { pointer: ptr0 });

                let idx1 = self.lds_masked_dword_index(addr_u, u32::from(offset1) * stride_bytes);
                let ptr1 = self.emit_expr(Expression::Access {
                    base: lds_ptr,
                    index: idx1,
                });
                let w1 = self.emit_expr(Expression::Load { pointer: ptr1 });

                self.store_vgpr_u32(vdst, w0);
                self.store_vgpr_u32(vdst.saturating_add(1), w1);
                true
            }
            DsLdsOp::Write2 { stride_bytes } => {
                let idx0 = self.lds_masked_dword_index(addr_u, u32::from(offset0) * stride_bytes);
                let ptr0 = self.emit_expr(Expression::Access {
                    base: lds_ptr,
                    index: idx0,
                });
                let vf0 = self.load_vgpr_f32(data0);
                let vu0 = self.bitcast_f32_to_u32(vf0);
                self.func.body.push(
                    Statement::Store {
                        pointer: ptr0,
                        value: vu0,
                    },
                    Span::UNDEFINED,
                );

                let idx1 = self.lds_masked_dword_index(addr_u, u32::from(offset1) * stride_bytes);
                let ptr1 = self.emit_expr(Expression::Access {
                    base: lds_ptr,
                    index: idx1,
                });
                let vf1 = self.load_vgpr_f32(data1);
                let vu1 = self.bitcast_f32_to_u32(vf1);
                self.func.body.push(
                    Statement::Store {
                        pointer: ptr1,
                        value: vu1,
                    },
                    Span::UNDEFINED,
                );
                true
            }
        }
    }

    fn translate_vertex_buf_fetch(
        &mut self,
        vb_slot: u32,
        is_mtbuf: bool,
        op: u8,
        format: u8,
        byte_offset: u16,
        vdata: u8,
    ) -> bool {
        if self.stage != ShaderStage::Vertex
            || self.vertex_input_mode != VertexInputMode::Attributes
        {
            return false;
        }
        let Some(arg_idx) = self.vs_in_arg else {
            return false;
        };
        let Some(comps) = (if is_mtbuf {
            mtbuf_load_components(op)
        } else {
            mubuf_load_components(op)
        }) else {
            return false;
        };
        let (attr_offset, shift) = vertex_fetch_slot(is_mtbuf, op, format, byte_offset);
        let Some(&(member_idx, attr_comps, scalar_kind)) =
            self.vs_attr_by_offset.get(&(vb_slot, attr_offset))
        else {
            return false;
        };
        let vs_in = self.raw_expr(Expression::FunctionArgument(arg_idx));
        let attr_val = self.emit_expr(Expression::AccessIndex {
            base: vs_in,
            index: member_idx,
        });
        for c in 0..comps {
            let src_c = c.saturating_add(shift);
            if src_c < attr_comps {
                let elem = if attr_comps == 1 {
                    attr_val
                } else {
                    self.emit_expr(Expression::AccessIndex {
                        base: attr_val,
                        index: u32::from(src_c),
                    })
                };
                match scalar_kind {
                    ScalarKind::Uint => self.store_vgpr_u32(vdata.saturating_add(c), elem),
                    ScalarKind::Sint => self.store_vgpr_i32(vdata.saturating_add(c), elem),
                    _ => self.store_vgpr_f32(vdata.saturating_add(c), elem),
                }
            } else {
                let fallback = match scalar_kind {
                    ScalarKind::Uint => self.lit_u32(if c == 3 { 1 } else { 0 }),
                    ScalarKind::Sint => self.lit_i32(if c == 3 { 1 } else { 0 }),
                    _ => self.lit_f32(if c == 3 { 1.0 } else { 0.0 }),
                };
                match scalar_kind {
                    ScalarKind::Uint => self.store_vgpr_u32(vdata.saturating_add(c), fallback),
                    ScalarKind::Sint => self.store_vgpr_i32(vdata.saturating_add(c), fallback),
                    _ => self.store_vgpr_f32(vdata.saturating_add(c), fallback),
                }
            }
        }
        true
    }

    // -- EXP (Vertex position/param & Pixel MRT/depth exports) --

    fn build_export_vec4(
        &mut self,
        en: u8,
        compr: bool,
        vsrc: [u8; 4],
        default_w: f32,
    ) -> Handle<Expression> {
        if compr {
            let lo_vec = if (en & 0b0011) != 0 {
                let vf = self.load_vgpr_f32(vsrc[0]);
                let vu = self.bitcast_f32_to_u32(vf);
                self.emit_expr(Expression::Math {
                    fun: MathFunction::Unpack2x16float,
                    arg: vu,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            } else {
                let z0 = self.lit_f32(0.0);
                let z1 = self.lit_f32(0.0);
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![z0, z1],
                })
            };
            let hi_vec = if (en & 0b1100) != 0 {
                let vf = self.load_vgpr_f32(vsrc[1]);
                let vu = self.bitcast_f32_to_u32(vf);
                self.emit_expr(Expression::Math {
                    fun: MathFunction::Unpack2x16float,
                    arg: vu,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            } else {
                let z2 = self.lit_f32(0.0);
                let w3 = self.lit_f32(default_w);
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![z2, w3],
                })
            };

            let c0 = if (en & 0b0001) != 0 {
                self.emit_expr(Expression::AccessIndex {
                    base: lo_vec,
                    index: 0,
                })
            } else {
                self.lit_f32(0.0)
            };
            let c1 = if (en & 0b0010) != 0 {
                self.emit_expr(Expression::AccessIndex {
                    base: lo_vec,
                    index: 1,
                })
            } else {
                self.lit_f32(0.0)
            };
            let c2 = if (en & 0b0100) != 0 {
                self.emit_expr(Expression::AccessIndex {
                    base: hi_vec,
                    index: 0,
                })
            } else {
                self.lit_f32(0.0)
            };
            let c3 = if (en & 0b1000) != 0 {
                self.emit_expr(Expression::AccessIndex {
                    base: hi_vec,
                    index: 1,
                })
            } else {
                self.lit_f32(default_w)
            };
            self.emit_expr(Expression::Compose {
                ty: self.types.vec4_f32_ty,
                components: vec![c0, c1, c2, c3],
            })
        } else {
            let mut comps = Vec::with_capacity(4);
            for i in 0..4usize {
                if (en & (1 << i)) != 0 {
                    comps.push(self.load_vgpr_f32(vsrc[i]));
                } else {
                    comps.push(self.lit_f32(if i == 3 { default_w } else { 0.0 }));
                }
            }
            self.emit_expr(Expression::Compose {
                ty: self.types.vec4_f32_ty,
                components: comps,
            })
        }
    }

    fn translate_exp(
        &mut self,
        target: Rdna2ExportTarget,
        en: u8,
        compr: bool,
        vsrc: [u8; 4],
    ) -> bool {
        match (self.stage, target) {
            (_, Rdna2ExportTarget::Null) => true,
            (ShaderStage::Vertex, Rdna2ExportTarget::Pos(0)) => {
                let Some(out_local) = self.vs_out_local else {
                    return false;
                };
                let val = self.build_export_vec4(en, compr, vsrc, 1.0);
                let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                let field_ptr = self.emit_expr(Expression::AccessIndex {
                    base: base_ptr,
                    index: 0,
                });
                self.func.body.push(
                    Statement::Store {
                        pointer: field_ptr,
                        value: val,
                    },
                    Span::UNDEFINED,
                );
                true
            }
            (ShaderStage::Vertex, Rdna2ExportTarget::Param(n)) => {
                let Some(out_local) = self.vs_out_local else {
                    return false;
                };
                let clamped = n % (MAX_WEBGPU_LOCATIONS - 1);
                let Some(&member_idx) = self.vs_param_member_index.get(&clamped) else {
                    return false;
                };
                let val = self.build_export_vec4(en, compr, vsrc, 0.0);
                let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                let field_ptr = self.emit_expr(Expression::AccessIndex {
                    base: base_ptr,
                    index: member_idx,
                });
                self.func.body.push(
                    Statement::Store {
                        pointer: field_ptr,
                        value: val,
                    },
                    Span::UNDEFINED,
                );
                true
            }
            (ShaderStage::Fragment, Rdna2ExportTarget::Mrt(n)) => {
                let Some(out_local) = self.ps_out_local else {
                    return false;
                };
                let Some(&member_idx) = self.ps_mrt_member_index.get(&n) else {
                    return false;
                };
                let val = self.build_export_vec4(en, compr, vsrc, 1.0);
                let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                let field_ptr = self.emit_expr(Expression::AccessIndex {
                    base: base_ptr,
                    index: member_idx,
                });
                self.func.body.push(
                    Statement::Store {
                        pointer: field_ptr,
                        value: val,
                    },
                    Span::UNDEFINED,
                );
                true
            }
            (ShaderStage::Fragment, Rdna2ExportTarget::MrtZ) => {
                let Some(out_local) = self.ps_out_local else {
                    return false;
                };
                let Some(member_idx) = self.ps_depth_member_index else {
                    return false;
                };
                let depth_val = if (en & 1) != 0 {
                    self.load_vgpr_f32(vsrc[0])
                } else {
                    self.lit_f32(0.0)
                };
                let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                let field_ptr = self.emit_expr(Expression::AccessIndex {
                    base: base_ptr,
                    index: member_idx,
                });
                self.func.body.push(
                    Statement::Store {
                        pointer: field_ptr,
                        value: depth_val,
                    },
                    Span::UNDEFINED,
                );
                true
            }
            _ => false,
        }
    }

    // -- Exec mask (single-lane model) --

    /// `s_*_saveexec_b32/b64` and `s_andn{1,2}_wrexec_b32/b64` on the single-lane
    /// `exec_mask` (`0u` / `1u`): `exec_mask = op(src0 & 1u, exec_mask) & 1u`,
    /// `scc = (exec_mask != 0u)`; `sdst` receives the old mask (saveexec) or the new mask
    /// (wrexec), with a zero high dword for `_b64`.
    fn translate_exec_mask_write(
        &mut self,
        write: ExecMaskWrite,
        sdst: u8,
        ssrc0: u8,
        literal: Option<u32>,
    ) -> bool {
        let src = if write.wave64 {
            self.read_ssrc_pair(ssrc0, literal).0
        } else {
            self.read_src_u32(u16::from(ssrc0), literal)
        };
        let one = self.lit_u32(1);
        let s = self.emit_bin(BinaryOperator::And, src, one);
        let old = self.load_exec();
        let (bin, not_s, not_e, not_result) = match write.op {
            ExecMaskOp::AndSaveExec => (BinaryOperator::And, false, false, false),
            ExecMaskOp::OrSaveExec => (BinaryOperator::InclusiveOr, false, false, false),
            ExecMaskOp::XorSaveExec => (BinaryOperator::ExclusiveOr, false, false, false),
            ExecMaskOp::AndN2SaveExec | ExecMaskOp::AndN2WrExec => {
                (BinaryOperator::And, false, true, false)
            }
            ExecMaskOp::OrN2SaveExec => (BinaryOperator::InclusiveOr, false, true, false),
            ExecMaskOp::NandSaveExec => (BinaryOperator::And, false, false, true),
            ExecMaskOp::NorSaveExec => (BinaryOperator::InclusiveOr, false, false, true),
            ExecMaskOp::XnorSaveExec => (BinaryOperator::ExclusiveOr, false, false, true),
            ExecMaskOp::AndN1SaveExec | ExecMaskOp::AndN1WrExec => {
                (BinaryOperator::And, true, false, false)
            }
            ExecMaskOp::OrN1SaveExec => (BinaryOperator::InclusiveOr, true, false, false),
        };
        let lhs = if not_s { self.emit_bit_not(s) } else { s };
        let rhs = if not_e { self.emit_bit_not(old) } else { old };
        let raw = self.emit_bin(bin, lhs, rhs);
        let raw = if not_result {
            self.emit_bit_not(raw)
        } else {
            raw
        };
        let one = self.lit_u32(1);
        let new = self.emit_bin(BinaryOperator::And, raw, one);

        let saved = if matches!(write.op, ExecMaskOp::AndN1WrExec | ExecMaskOp::AndN2WrExec) {
            new
        } else {
            old
        };
        if write.wave64 {
            let zero = self.lit_u32(0);
            self.store_sdst_pair(sdst, saved, zero);
        } else if sdst == 106 {
            let zero = self.lit_u32(0);
            let cond = self.emit_bin(BinaryOperator::NotEqual, saved, zero);
            self.store_vcc(cond);
        } else {
            self.store_sgpr_u32(sdst, saved);
        }
        if let Some(local) = self.exec_local {
            self.store_local(local, new);
        }
        self.update_scc_nonzero(new);
        true
    }

    // -- Structured control flow (rdna2_cfg) --

    /// `true` while lowering code that WGSL uniformity analysis treats as non-uniform:
    /// any nested structured body, or top-level code after a nested `return`.
    fn in_nonuniform_cf(&self) -> bool {
        self.structured_cf && (self.cf_depth > 0 || self.cf_nonuniform_return)
    }

    /// `true` when the current block already ends in a control transfer, so further
    /// statements would be unreachable.
    fn body_terminated(&self) -> bool {
        matches!(
            self.func.body.last(),
            Some(
                Statement::Return { .. } | Statement::Break | Statement::Continue | Statement::Kill
            )
        )
    }

    fn push_stmt(&mut self, stmt: Statement) {
        self.func.body.push(stmt, Span::UNDEFINED);
    }

    /// Emit the stage-correct `return` (`return out;` for vertex/fragment, `return;` for
    /// compute).
    fn emit_stage_return(&mut self) {
        let out_local = match self.stage {
            ShaderStage::Vertex => self.vs_out_local,
            ShaderStage::Fragment => self.ps_out_local,
            ShaderStage::Compute => None,
        };
        let value = out_local.map(|local| self.load_local(local));
        self.push_stmt(Statement::Return { value });
        if self.cf_depth > 0 {
            self.cf_nonuniform_return = true;
        }
    }

    /// Lower `f` into a fresh nested block (one structured nesting level deeper).
    fn lower_nested(&mut self, f: impl FnOnce(&mut Self)) -> naga::Block {
        let saved = std::mem::replace(&mut self.func.body, naga::Block::new());
        self.cf_depth += 1;
        f(self);
        self.cf_depth -= 1;
        std::mem::replace(&mut self.func.body, saved)
    }

    fn new_cf_local(&mut self, prefix: &str) -> Handle<LocalVariable> {
        let name = format!("{prefix}{}", self.cf_local_counter);
        self.cf_local_counter += 1;
        self.func.local_variables.append(
            LocalVariable {
                name: Some(name),
                ty: self.types.u32_ty,
                init: None,
            },
            Span::UNDEFINED,
        )
    }

    /// WGSL condition for "branch of `kind` is taken" (`scc`/`vcc` are `bool`,
    /// `exec_mask` is `u32`).
    fn branch_condition(&mut self, kind: BranchKind) -> Handle<Expression> {
        match kind {
            BranchKind::Unconditional => self.lit_bool(true),
            BranchKind::Scc1 => self.load_scc(),
            BranchKind::Vccnz => self.load_vcc(),
            BranchKind::Scc0 | BranchKind::Vccz => {
                let v = if kind == BranchKind::Scc0 {
                    self.load_scc()
                } else {
                    self.load_vcc()
                };
                self.emit_expr(Expression::Unary {
                    op: UnaryOperator::LogicalNot,
                    expr: v,
                })
            }
            BranchKind::Execz | BranchKind::Execnz => {
                let e = self.load_exec();
                let zero = self.lit_u32(0);
                let op = if kind == BranchKind::Execz {
                    BinaryOperator::Equal
                } else {
                    BinaryOperator::NotEqual
                };
                self.emit_bin(op, e, zero)
            }
        }
    }

    /// Translate every instruction of a basic block, including its trailing control
    /// instruction (branches count as translated: their control flow is structured).
    ///
    /// Register values never carry over between blocks as folded constants: every read
    /// re-loads the function-scope `var`, so values assigned on one path or loop iteration
    /// can never leak into a merge block or loop header.
    fn lower_basic_block(
        &mut self,
        id: BlockId,
        cfg: &ControlFlowGraph,
        insts: &[Rdna2Instruction],
    ) {
        let Some(block) = cfg.blocks.get(id) else {
            return;
        };
        for idx in block.instr_range.clone() {
            if self.body_terminated() {
                break;
            }
            if let Some(inst) = insts.get(idx) {
                self.translate_instruction(idx, inst);
            }
        }
    }

    /// Lower a structured control-flow tree from [`structure_cfg`] into naga IR.
    fn lower_structured(
        &mut self,
        node: &StructuredNode,
        cfg: &ControlFlowGraph,
        insts: &[Rdna2Instruction],
    ) {
        if self.body_terminated() {
            return;
        }
        match node {
            StructuredNode::Block(id) => self.lower_basic_block(*id, cfg, insts),
            StructuredNode::Seq(nodes) => {
                for n in nodes {
                    if self.body_terminated() {
                        break;
                    }
                    self.lower_structured(n, cfg, insts);
                }
            }
            StructuredNode::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                let condition = self.branch_condition(*cond);
                let accept = self.lower_nested(|s| s.lower_structured(then_branch, cfg, insts));
                let reject = match else_branch {
                    Some(e) => self.lower_nested(|s| s.lower_structured(e, cfg, insts)),
                    None => naga::Block::new(),
                };
                self.push_stmt(Statement::If {
                    condition,
                    accept,
                    reject,
                });
            }
            StructuredNode::Loop { body } => {
                let guard = self.new_cf_local("_pg6_loop");
                let zero = self.lit_u32(0);
                self.store_local(guard, zero);
                let body = self.lower_nested(|s| {
                    s.emit_iteration_guard(guard, STRUCTURED_LOOP_MAX_ITERATIONS);
                    s.lower_structured(body, cfg, insts);
                });
                self.push_stmt(Statement::Loop {
                    body,
                    continuing: naga::Block::new(),
                    break_if: None,
                });
            }
            StructuredNode::Break => self.push_stmt(Statement::Break),
            StructuredNode::Continue => self.push_stmt(Statement::Continue),
            StructuredNode::Return => self.emit_stage_return(),
            StructuredNode::Dispatch {
                entry,
                blocks,
                max_iterations,
            } => self.lower_dispatch(*entry, blocks, *max_iterations, cfg, insts),
        }
    }

    /// `if (guard >= max) { break; } guard = guard + 1u;`
    fn emit_iteration_guard(&mut self, guard: Handle<LocalVariable>, max: u32) {
        let g = self.load_local(guard);
        let limit = self.lit_u32(max);
        let done = self.emit_bin(BinaryOperator::GreaterEqual, g, limit);
        let mut brk = naga::Block::new();
        brk.push(Statement::Break, Span::UNDEFINED);
        self.push_stmt(Statement::If {
            condition: done,
            accept: brk,
            reject: naga::Block::new(),
        });
        let one = self.lit_u32(1);
        let next = self.emit_bin(BinaryOperator::Add, g, one);
        self.store_local(guard, next);
    }

    /// Edge to `target` inside a dispatcher: program exit returns, member blocks set the
    /// program counter, anything else leaves the dispatcher.
    fn dispatch_transfer(
        &mut self,
        target: BlockId,
        members: &BTreeSet<BlockId>,
        pc: Handle<LocalVariable>,
        cfg: &ControlFlowGraph,
    ) {
        if Some(target) == cfg.exit_block {
            self.emit_stage_return();
        } else {
            let next = if members.contains(&target) {
                u32::try_from(target).unwrap_or(DISPATCH_EXIT_PC)
            } else {
                DISPATCH_EXIT_PC
            };
            let v = self.lit_u32(next);
            self.store_local(pc, v);
        }
    }

    /// Bounded `loop { switch (pc) { ... } }` fallback for irreducible regions.
    fn lower_dispatch(
        &mut self,
        entry: BlockId,
        blocks: &[BlockId],
        max_iterations: u32,
        cfg: &ControlFlowGraph,
        insts: &[Rdna2Instruction],
    ) {
        let members: BTreeSet<BlockId> = blocks
            .iter()
            .copied()
            .filter(|&b| {
                b < cfg.blocks.len() && u32::try_from(b).is_ok_and(|v| v != DISPATCH_EXIT_PC)
            })
            .collect();
        let pc = self.new_cf_local("_pg6_pc");
        let iter = self.new_cf_local("_pg6_iter");
        let entry_pc = if members.contains(&entry) {
            u32::try_from(entry).unwrap_or(DISPATCH_EXIT_PC)
        } else {
            DISPATCH_EXIT_PC
        };
        let v = self.lit_u32(entry_pc);
        self.store_local(pc, v);
        let zero = self.lit_u32(0);
        self.store_local(iter, zero);

        let body = self.lower_nested(|s| {
            s.emit_iteration_guard(iter, max_iterations);
            // if (pc == EXIT) { break; }
            let cur = s.load_local(pc);
            let exit = s.lit_u32(DISPATCH_EXIT_PC);
            let leave = s.emit_bin(BinaryOperator::Equal, cur, exit);
            let mut brk = naga::Block::new();
            brk.push(Statement::Break, Span::UNDEFINED);
            s.push_stmt(Statement::If {
                condition: leave,
                accept: brk,
                reject: naga::Block::new(),
            });

            let selector = s.load_local(pc);
            let mut cases = Vec::with_capacity(members.len() + 1);
            for &id in &members {
                let terminator = cfg.blocks[id].terminator;
                let case_body = s.lower_nested(|s| {
                    s.lower_basic_block(id, cfg, insts);
                    if s.body_terminated() {
                        return;
                    }
                    match terminator {
                        Terminator::End => s.emit_stage_return(),
                        Terminator::Fallthrough(t) | Terminator::Branch(t) => {
                            s.dispatch_transfer(t, &members, pc, cfg);
                        }
                        Terminator::CondBranch {
                            kind,
                            target,
                            fallthrough,
                        } => {
                            let condition = s.branch_condition(kind);
                            let accept =
                                s.lower_nested(|s| s.dispatch_transfer(target, &members, pc, cfg));
                            let reject = s.lower_nested(|s| {
                                s.dispatch_transfer(fallthrough, &members, pc, cfg);
                            });
                            s.push_stmt(Statement::If {
                                condition,
                                accept,
                                reject,
                            });
                        }
                    }
                });
                cases.push(SwitchCase {
                    value: SwitchValue::U32(u32::try_from(id).unwrap_or(DISPATCH_EXIT_PC)),
                    body: case_body,
                    fall_through: false,
                });
            }
            let mut default_body = naga::Block::new();
            let exit = s.lit_u32(DISPATCH_EXIT_PC);
            let ptr = s.raw_expr(Expression::LocalVariable(pc));
            default_body.push(
                Statement::Store {
                    pointer: ptr,
                    value: exit,
                },
                Span::UNDEFINED,
            );
            cases.push(SwitchCase {
                value: SwitchValue::Default,
                body: default_body,
                fall_through: false,
            });
            s.push_stmt(Statement::Switch { selector, cases });
        });
        self.push_stmt(Statement::Loop {
            body,
            continuing: naga::Block::new(),
            break_if: None,
        });
    }

    // -- Finalize function & module --

    fn finish(
        mut self,
        entry_name: String,
    ) -> (
        Module,
        usize,
        usize,
        Vec<String>,
        Vec<String>,
        Vec<WgslBindingInfo>,
    ) {
        // Emit stage return statement (unless structured lowering already ended the body
        // with one).
        if !self.body_terminated() {
            self.emit_stage_return();
        }

        let (naga_stage, workgroup_size) = match self.stage {
            ShaderStage::Vertex => (NagaStage::Vertex, [0, 0, 0]),
            ShaderStage::Fragment => (NagaStage::Fragment, [0, 0, 0]),
            ShaderStage::Compute => (NagaStage::Compute, self.workgroup_size),
        };

        self.module.entry_points.push(EntryPoint {
            name: entry_name,
            stage: naga_stage,
            early_depth_test: None,
            workgroup_size,
            workgroup_size_overrides: None,
            function: self.func,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        (
            self.module,
            self.translated_op_count,
            self.unsupported_op_count,
            self.unsupported_comments,
            self.note_comments,
            self.emitted_bindings,
        )
    }
}

/// Baseline WebGPU capabilities permitted during Naga validation.
/// `SHADER_FLOAT16_IN_FLOAT32` enables core WGSL `pack2x16float`/`unpack2x16float`
/// built-ins (no `enable f16;` extension required in WebGPU).
const BASELINE_WEBGPU_CAPABILITIES: Capabilities = Capabilities::SHADER_FLOAT16_IN_FLOAT32;

/// Translate a raw little-endian AMD RDNA2 (GFX10.3 / Scarlett) shader binary into
/// a complete, Naga-validated WebGPU WGSL module.
///
/// Never panics on empty, truncated, misaligned, or arbitrary input byte slices.
pub fn translate_rdna2_to_wgsl(
    bytes: &[u8],
    options: &Rdna2TranslateOptions,
) -> Result<Rdna2WgslOutput, Rdna2TranslateError> {
    let instructions = decode_program(bytes);
    let summary = analyze_instructions(bytes, &instructions);
    let plan = plan_program(&summary, &instructions, options);

    let mut emitter = IrEmitter::new(&plan);
    let active = &instructions[..plan.active_inst_count];
    let cfg = ControlFlowGraph::from_instructions(active);
    let real_block_count = cfg.blocks.iter().filter(|b| !b.synthetic_exit).count();
    let has_branch = active.iter().any(|inst| {
        matches!(inst.detail, Rdna2Detail::Sopp { op, .. } if BranchKind::from_sopp_op(op).is_some())
    });
    if real_block_count <= 1 && !has_branch {
        // Fast path: straight-line program.
        for (inst_idx, inst) in active.iter().enumerate() {
            emitter.translate_instruction(inst_idx, inst);
        }
    } else {
        // Lift SOPP branches into structured WGSL control flow (if / loop / bounded
        // dispatcher). Every register is a function-scope `var` declared up front.
        emitter.structured_cf = true;
        let tree = structure_cfg(&cfg);
        emitter.lower_structured(&tree, &cfg, active);
    }

    let entry_point = plan.entry_point.clone();
    let (
        module,
        translated_op_count,
        unsupported_op_count,
        unsupported_comments,
        note_comments,
        bindings,
    ) = emitter.finish(entry_point.clone());

    // Unconditional Naga validation against baseline WebGPU capabilities.
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    let module_info = validator
        .validate(&module)
        .map_err(|err| Rdna2TranslateError::NagaValidation(err.emit_to_string("")))?;

    let wgsl_body = naga::back::wgsl::write_string(
        &module,
        &module_info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .map_err(|err| Rdna2TranslateError::WgslEmit(err.to_string()))?;

    // Prepend strictly sanitized WGSL header & graceful fallback opcode comments.
    let mut wgsl = String::with_capacity(wgsl_body.len() + 256);
    let _ = writeln!(
        wgsl,
        "// Translated from AMD RDNA2 (GFX10.3) by playgta6-shaders"
    );
    let _ = writeln!(
        wgsl,
        "// hash: {} | stage: {} | translated_ops: {} | unsupported_ops: {}",
        format_shader_hash(summary.fnv1a_hash),
        plan.stage.as_str(),
        translated_op_count,
        unsupported_op_count
    );
    for comment in unsupported_comments.iter().chain(note_comments.iter()) {
        let _ = writeln!(wgsl, "{comment}");
    }
    wgsl.push('\n');
    wgsl.push_str(&wgsl_body);

    // When `naga/wgsl-in` is active (unit tests), additionally round-trip parse & validate the
    // final WGSL text (including comments) via `naga::front::wgsl::parse_str`.
    #[cfg(test)]
    {
        let reparsed = naga::front::wgsl::parse_str(&wgsl)
            .map_err(|err| Rdna2TranslateError::WgslParse(err.emit_to_string(&wgsl)))?;
        let mut revalidator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
        revalidator
            .validate(&reparsed)
            .map_err(|err| Rdna2TranslateError::NagaValidation(err.emit_to_string(&wgsl)))?;
    }

    let output_locations: Vec<u32> = match plan.stage {
        ShaderStage::Vertex => plan.vs_params.iter().map(|&p| u32::from(p)).collect(),
        ShaderStage::Fragment => plan.ps_mrts.iter().map(|&m| u32::from(m)).collect(),
        ShaderStage::Compute => Vec::new(),
    };

    let binding_summary = Rdna2BindingSummary {
        has_cbv: !plan.cbv_slots.is_empty(),
        cbv_vec4_count: plan.cbv_vec4_count,
        has_texture_2d: !plan.tex_slots.is_empty(),
        has_sampler: !plan.samp_slots.is_empty(),
        has_storage_buffer: !plan.storage_buf_slots.is_empty(),
        has_storage_texture: !plan.storage_tex_slots.is_empty(),
        has_lds: plan.uses_lds,
        vertex_input_mode: plan.vertex_input_mode.as_str().to_string(),
        fragment_input_locations: plan.frag_attrs.iter().map(|&a| u32::from(a)).collect(),
        output_locations,
        exports_position: plan.stage == ShaderStage::Vertex,
        exports_depth: plan.stage == ShaderStage::Fragment && plan.ps_exports_depth,
    };

    let vertex_inputs: Vec<WgslVertexInputInfo> = if plan.stage == ShaderStage::Vertex
        && plan.vertex_input_mode == VertexInputMode::Attributes
    {
        plan.vertex_attrs
            .iter()
            .filter_map(|attr| {
                attr.byte_offset.map(|byte_offset| WgslVertexInputInfo {
                    location: attr.location,
                    vb_slot: attr.vb_slot,
                    byte_offset: u32::from(byte_offset),
                    components: attr.components,
                    component_type: attr.component_type,
                    fetch_format: attr.fetch_format,
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(Rdna2WgslOutput {
        wgsl,
        stage: plan.stage,
        entry_point,
        hash_fnv1a64: summary.fnv1a_hash,
        translated_op_count,
        unsupported_op_count,
        naga_validated: true,
        bindings,
        vertex_inputs,
        binding_summary,
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::fnv1a_64;

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(words.len() * 4);
        for &w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn enc_sopp(op: u8, simm16: u16) -> u32 {
        (0b1011_1111_1u32 << 23) | (u32::from(op & 0x7F) << 16) | u32::from(simm16)
    }

    fn enc_sop1(op: u8, sdst: u8, ssrc0: u8) -> u32 {
        (0b1011_1110_1u32 << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | (u32::from(op) << 8)
            | u32::from(ssrc0)
    }

    fn enc_sop2(op: u8, sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b10u32 << 30)
            | (u32::from(op & 0x7F) << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | (u32::from(ssrc1) << 8)
            | u32::from(ssrc0)
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

    fn enc_vop1(op: u8, vdst: u8, src0: u16) -> u32 {
        (0b0111_111u32 << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(op) << 9)
            | u32::from(src0 & 0x1FF)
    }

    fn enc_vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
        (0b0111_110u32 << 25)
            | (u32::from(op) << 17)
            | (u32::from(vsrc1) << 9)
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

    #[allow(clippy::too_many_arguments)]
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

    fn assert_naga_wgsl_valid(wgsl: &str) {
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\nWGSL:\n{wgsl}", e.emit_to_string(wgsl)));
        let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("WGSL validation failed:\n{}\nWGSL:\n{wgsl}", e.emit_to_string(wgsl)));
    }

    #[test]
    fn translates_fullscreen_triangle_vertex_shader_and_validates_with_naga() {
        // Real procedural fullscreen-triangle RDNA2 VS:
        //   v_lshlrev_b32 v1, 1, v0           ; (vid << 1)
        //   v_and_b32     v1, 2, v1           ; u_int = (vid << 1) & 2
        //   v_and_b32     v2, 2, v0           ; v_int = vid & 2
        //   v_cvt_f32_u32 v1, v1              ; u = f32(u_int)
        //   v_cvt_f32_u32 v2, v2              ; v = f32(v_int)
        //   v_fma_f32     v3, v1, 2.0, -1.0   ; pos.x = u * 2.0 - 1.0
        //   v_fma_f32     v4, -v2, 2.0, 1.0   ; pos.y = 1.0 - v * 2.0
        //   v_mov_b32     v5, 0.0             ; pos.z = 0.0
        //   v_mov_b32     v6, 1.0             ; pos.w = 1.0
        //   exp pos0 v3, v4, v5, v6
        //   exp param0 v1, v2, off, off done
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vop2(0x1A, 1, 128 + 1, 0)); // v_lshlrev_b32 v1, 1, v0
        words.push(enc_vop2(0x1B, 1, 128 + 2, 1)); // v_and_b32 v1, 2, v1
        words.push(enc_vop2(0x1B, 2, 128 + 2, 0)); // v_and_b32 v2, 2, v0
        words.push(enc_vop1(0x06, 1, 256 + 1)); // v_cvt_f32_u32 v1, v1
        words.push(enc_vop1(0x06, 2, 256 + 2)); // v_cvt_f32_u32 v2, v2
        words.extend_from_slice(&enc_vop3(0x14B, 3, 256 + 1, 244, 243, false, 0, 0, 0)); // fma(v1, 2.0, -1.0)
        words.extend_from_slice(&enc_vop3(0x14B, 4, 256 + 2, 244, 242, false, 0, 0, 0b001)); // fma(-v2, 2.0, 1.0)
        words.push(enc_vop1(0x01, 5, 128)); // v_mov_b32 v5, 0
        words.push(enc_vop1(0x01, 6, 242)); // v_mov_b32 v6, 1.0
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [3, 4, 5, 6])); // pos0
        words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [1, 2, 0, 0])); // param0 done
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("fullscreen-triangle VS translation should succeed");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.entry_point, "vs_main");
        assert_eq!(out.hash_fnv1a64, fnv1a_64(&bytes));
        assert_eq!(out.translated_op_count, 12);
        assert_eq!(out.unsupported_op_count, 0);
        assert_eq!(out.binding_summary.vertex_input_mode, "vertex_index");
        assert!(out.binding_summary.exports_position);
        assert_eq!(out.binding_summary.output_locations, vec![0]);

        assert!(out.wgsl.contains("@vertex"));
        assert!(out.wgsl.contains("@builtin(vertex_index)"));
        assert!(out.wgsl.contains("@builtin(position)"));
        assert!(out.wgsl.contains("@location(0)"));
        assert!(out.wgsl.contains("struct VsOut"));
        assert!(out.wgsl.contains("fma("));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_textured_splash_blit_pixel_shader_with_vintrp_mimg_exp() {
        // Textured splash-blit PS:
        //   v_interp_p1_f32 v2, v0, attr0.x
        //   v_interp_p2_f32 v2, v1, attr0.x
        //   v_interp_p1_f32 v3, v0, attr0.y
        //   v_interp_p2_f32 v3, v1, attr0.y
        //   image_sample v4, v2, s[8:15], s[4:7] dmask:0xf dim:1
        //   s_waitcnt vmcnt(0)
        //   exp mrt0 v4, v5, v6, v7 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 2, 1));
        words.push(enc_sopp(0x0C, 0x3F70)); // s_waitcnt vmcnt(0)
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("textured splash-blit PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.entry_point, "fs_main");
        assert_eq!(out.translated_op_count, 8);
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.binding_summary.has_sampler);
        assert!(!out.binding_summary.has_cbv);
        assert_eq!(out.binding_summary.fragment_input_locations, vec![0]);
        assert_eq!(out.binding_summary.output_locations, vec![0]);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 1,
                    kind: "texture_2d",
                    name: "tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: Some("2d"),
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
            ]
        );

        assert!(out.wgsl.contains("@fragment"));
        assert!(out.wgsl.contains("fn fs_main("));
        assert!(!out.wgsl.contains("fn ps_main("));
        assert!(out.wgsl.contains("struct FragIn"));
        assert!(out.wgsl.contains("struct PsOut"));
        assert!(out.wgsl.contains("texture_2d<f32>"));
        assert!(out.wgsl.contains("sampler"));
        assert!(
            out.wgsl.contains("textureSample("),
            "expected textureSample in WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("tex0") && out.wgsl.contains("samp0"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_smem_cbv_modulated_pixel_shader() {
        // CBV-modulated PS:
        //   s_load_dwordx4 s[4:7], s[0:1], 0x10 glc
        //   s_waitcnt lgkmcnt(0)
        //   v_interp_p1_f32 v2, v0, attr0.x
        //   v_interp_p2_f32 v2, v1, attr0.x
        //   v_interp_p1_f32 v3, v0, attr0.y
        //   v_interp_p2_f32 v3, v1, attr0.y
        //   image_sample_lz v4, v2, s[8:15], s[12:15] dmask:0xf dim:1
        //   v_mul_f32 v4, s4, v4
        //   v_mul_f32 v5, s5, v5
        //   v_mul_f32 v6, s6, v6
        //   v_mul_f32 v7, s7, v7
        //   exp mrt0 v4, v5, v6, v7 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x02, 4, 0, 0x7D, 0x10, true, false));
        words.push(enc_sopp(0x0C, 0xC07F));
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 4, 2, 3)); // image_sample_lz
        words.push(enc_vop2(0x08, 4, 4, 4)); // v_mul_f32 v4, s4, v4
        words.push(enc_vop2(0x08, 5, 5, 5)); // v_mul_f32 v5, s5, v5
        words.push(enc_vop2(0x08, 6, 6, 6)); // v_mul_f32 v6, s6, v6
        words.push(enc_vop2(0x08, 7, 7, 7)); // v_mul_f32 v7, s7, v7
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("CBV-modulated PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert!(out.binding_summary.has_cbv);
        // s_load_dwordx4 at offset 0x10 reads dwords 4..=7 -> vec4 index 1 -> 2 vec4s (32 bytes).
        assert_eq!(out.binding_summary.cbv_vec4_count, 2);
        assert_eq!(out.bindings[0].min_binding_size_bytes, Some(32));
        assert_eq!(out.bindings[0].highest_vec4_offset, Some(1));
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.binding_summary.has_sampler);
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.wgsl.contains("struct Rdna2Cbv"));
        assert!(out.wgsl.contains("array<vec4<u32>, 2>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@group(0) @binding(0)"));
        assert!(
            out.wgsl.contains("var<uniform>") && out.wgsl.contains("cb0") && out.wgsl.contains("Rdna2Cbv"),
            "expected uniform cb0 binding in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("textureSampleLevel("),
            "expected textureSampleLevel in WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"has_cbv\": true"));
        assert!(json.contains("\"naga_validated\": true"));
    }

    #[test]
    fn translates_compressed_f16_export_pixel_shader_with_unpack2x16float() {
        // Packed f16 output PS:
        //   v_mov_b32 v0, 1.0
        //   v_mov_b32 v1, 0.5
        //   v_cvt_pkrtz_f16_f32 v2, v0, v1
        //   v_cvt_pkrtz_f16_f32 v3, v1, v0
        //   exp mrt0 v2, v3, off, off done compr vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vop1(0x01, 0, 242)); // 1.0
        words.push(enc_vop1(0x01, 1, 240)); // 0.5
        words.push(enc_vop2(0x2F, 2, 256 + 0, 1)); // v_cvt_pkrtz_f16_f32 v2, v0, v1
        words.push(enc_vop2(0x2F, 3, 256 + 1, 0)); // v_cvt_pkrtz_f16_f32 v3, v1, v0
        words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [2, 3, 0, 0])); // compr
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("compressed f16 export PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.wgsl.contains("pack2x16float"));
        assert!(out.wgsl.contains("unpack2x16float"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_valu_salu_modifiers_and_gracefully_handles_unsupported_ops() {
        // Exercise VOP3 abs/neg/clamp/omod, VOPC + v_cndmask_b32, SALU, math builtins,
        // and an unsupported DS instruction to verify graceful comment fallback.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 2, 128 + 7)); // s_mov_b32 s2, 7
        words.push(enc_sop2(0x00, 3, 2, 128 + 3)); // s_add_u32 s3, s2, 3
        words.push(enc_vop1(0x06, 0, 3)); // v_cvt_f32_u32 v0, s3
        words.push(enc_vop1(0x2A, 1, 256 + 0)); // v_rcp_f32 v1, v0
        words.push(enc_vop1(0x2E, 2, 256 + 1)); // v_rsq_f32 v2, v1
        words.push(enc_vop1(0x33, 3, 256 + 2)); // v_sqrt_f32 v3, v2
        words.push(enc_vop1(0x24, 4, 256 + 3)); // v_floor_f32 v4, v3
        words.push(enc_vop1(0x20, 5, 256 + 3)); // v_fract_f32 v5, v3
        words.push(enc_vopc(0x01, 256 + 4, 5)); // v_cmp_lt_f32 vcc, v4, v5
        words.push(enc_vop2(0x01, 6, 256 + 4, 5)); // v_cndmask_b32 v6, v4, v5, vcc
        words.extend_from_slice(&enc_vop3(
            0x141,
            7,
            256 + 6,
            242,
            240,
            true,  // clamp
            1,     // mul:2
            0b001, // |v6|
            0b010, // -1.0
        ));
        // Unsupported DS instruction: ds_swizzle_b32 v8, v0, v1
        let ds_w0 = (0b110110u32 << 26) | (0x3Du32 << 18);
        let ds_w1 = (8u32 << 24) | (1u32 << 8);
        words.extend_from_slice(&[ds_w0, ds_w1]);
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [6, 7, 4, 5]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("program with unsupported DS op must still produce valid WGSL");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 1);
        assert!(out.translated_op_count >= 12);
        assert!(out.wgsl.contains("// unsupported_op"));
        assert!(out.wgsl.contains("ds_swizzle_b32"));
        assert!(out.wgsl.contains("inverseSqrt("));
        assert!(out.wgsl.contains("sqrt("));
        assert!(out.wgsl.contains("floor("));
        assert!(out.wgsl.contains("fract("));
        assert!(out.wgsl.contains("abs("));
        assert!(out.wgsl.contains("clamp("));
        assert!(out.wgsl.contains("select("));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn handles_empty_truncated_misaligned_and_fuzz_inputs_without_panicking() {
        // Empty and sub-word buffers across all stage overrides
        for len in 0..=3 {
            let buf = vec![0x5Au8; len];
            for stage in [
                Rdna2StageOverride::Auto,
                Rdna2StageOverride::Vertex,
                Rdna2StageOverride::Pixel,
                Rdna2StageOverride::Compute,
            ] {
                let opts = Rdna2TranslateOptions {
                    stage,
                    ..Rdna2TranslateOptions::default()
                };
                let out = translate_rdna2_to_wgsl(&buf, &opts)
                    .expect("empty/sub-word buffer must produce valid WGSL");
                assert!(out.naga_validated);
                assert_naga_wgsl_valid(&out.wgsl);
            }
        }

        // Truncated 64-bit instructions (missing second word)
        let truncated_words = [
            enc_smem(0x02, 0, 0, 0x7D, 0, false, false)[0],
            enc_exp(12, 0xF, false, true, false, [0, 1, 2, 3])[0],
            enc_vop3(0x14B, 0, 0, 0, 0, false, 0, 0, 0)[0],
            enc_mimg(0x20, 0xF, 1, 0, 0, 0, 0)[0],
            enc_vop2(0x2C, 0, 256, 1),
        ];
        for w0 in truncated_words {
            let mut buf = w0.to_le_bytes().to_vec();
            buf.push(0xCC);
            let out = translate_rdna2_to_wgsl(&buf, &Rdna2TranslateOptions::default())
                .expect("truncated instruction must degrade gracefully to valid WGSL");
            assert!(out.naga_validated);
            assert!(out.unsupported_op_count >= 1);
            assert!(out.wgsl.contains("(truncated)"));
            assert_naga_wgsl_valid(&out.wgsl);
        }

        // Deterministic pseudo-random fuzz buffers across all stage overrides
        let mut state: u64 = 0xCAFE_BABE_1337_4242;
        for sample_len in [16usize, 64, 131, 256] {
            let mut fuzz = Vec::with_capacity(sample_len);
            for _ in 0..sample_len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                fuzz.push((state & 0xFF) as u8);
            }
            for stage in [
                Rdna2StageOverride::Auto,
                Rdna2StageOverride::Vertex,
                Rdna2StageOverride::Pixel,
                Rdna2StageOverride::Compute,
            ] {
                let opts = Rdna2TranslateOptions {
                    stage,
                    ..Rdna2TranslateOptions::default()
                };
                let out = translate_rdna2_to_wgsl(&fuzz, &opts)
                    .expect("arbitrary fuzz bytes must still produce Naga-valid WGSL");
                assert!(out.naga_validated);
                assert_naga_wgsl_valid(&out.wgsl);
            }
        }
    }

    #[test]
    fn translates_attribute_driven_vertex_shader_with_mubuf_and_multi_vgpr_inputs() {
        // Attribute-driven VS using MUBUF buffer_load_format_xyzw + multi-VGPR uv inputs:
        //   buffer_load_format_xyzw v0, v0, s[4:7], 0 idxen
        //   exp pos0 v0, v1, v2, v3
        //   exp param0 v4, v5, off, off done
        //   s_endpgm
        let mubuf_w0 = (0b111000u32 << 26) | (0x03u32 << 18) | (1 << 13);
        let mubuf_w1 = (128u32 << 24) | (1u32 << 16);
        let mut words = Vec::new();
        words.extend_from_slice(&[mubuf_w0, mubuf_w1]);
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [0, 1, 2, 3]));
        words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [4, 5, 0, 0]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("attribute-driven VS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.binding_summary.vertex_input_mode, "attributes");
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.wgsl.contains("struct VsIn"));
        assert!(out.wgsl.contains("@location(0)"));
        assert!(out.wgsl.contains("@location(1)"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_compute_shader_with_barrier_image_load_dynamic_smem_and_vop3p() {
        // Compute shader exercising:
        //   s_mov_b32 s2, 16
        //   s_load_dwordx2 s[4:5], s[0:1], s2, 0x4
        //   image_load v4, v0, s[8:15] dmask:0xf dim:1
        //   v_pk_add_f16 v6, v4, v5, v0
        //   v_dot2_f32_f16 v7, v4, v5, v6 clamp
        //   s_barrier
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 2, 128 + 16)); // s_mov_b32 s2, 16
        words.extend_from_slice(&enc_smem(0x01, 4, 0, 2, 4, false, false)); // dynamic soffset=s2
        words.extend_from_slice(&enc_mimg(0x00, 0xF, 1, 0, 4, 2, 0)); // image_load
        // VOP3P: prefix 0b110011, op 0x0F (v_pk_add_f16), vdst=6, src0=v4, src1=v5, src2=v0
        let vop3p_add_w0 = (0b110011u32 << 26) | (0x0Fu32 << 16) | 6;
        let vop3p_add_w1 = ((256u32 + 0) << 18) | ((256u32 + 5) << 9) | (256u32 + 4);
        words.extend_from_slice(&[vop3p_add_w0, vop3p_add_w1]);
        // VOP3P: op 0x13 (v_dot2_f32_f16), clamp=1, vdst=7, src0=v4, src1=v5, src2=v6
        let vop3p_dot_w0 = (0b110011u32 << 26) | (0x13u32 << 16) | (1 << 15) | 7;
        let vop3p_dot_w1 = ((256u32 + 6) << 18) | ((256u32 + 5) << 9) | (256u32 + 4);
        words.extend_from_slice(&[vop3p_dot_w0, vop3p_dot_w1]);
        words.push(enc_sopp(0x0A, 0)); // s_barrier
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("compute shader should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.entry_point, "cs_main");
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.binding_summary.has_cbv);
        // Dynamic soffset falls back cleanly to at least 16 vec4s (256 bytes).
        assert_eq!(out.binding_summary.cbv_vec4_count, DEFAULT_DYNAMIC_CBV_VEC4_COUNT);
        assert_eq!(out.bindings[0].min_binding_size_bytes, Some(256));
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.wgsl.contains("@compute"));
        assert!(out.wgsl.contains("workgroupBarrier()"));
        assert!(out.wgsl.contains("textureLoad("));
        assert!(out.wgsl.contains("dot("));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_multi_mrt_and_depth_export_pixel_shader_with_explicit_lod() {
        // Multi-MRT + MRTZ PS with image_sample_l:
        //   v_interp_mov_f32 v2, p0, attr0.x
        //   v_interp_mov_f32 v3, p0, attr0.y
        //   v_mov_b32 v4, 2.0
        //   image_sample_l v5, v2, s[8:15], s[4:7] dmask:0xf dim:1
        //   exp mrt0 v5, v6, v7, v8
        //   exp mrt1 v8, v7, v6, v5
        //   exp mrtz v4, off, off, off done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vintrp(2, 2, 0, 0, 2)); // v_interp_mov_f32 v2, p0, attr0.x
        words.push(enc_vintrp(2, 3, 0, 1, 2)); // v_interp_mov_f32 v3, p0, attr0.y
        words.push(enc_vop1(0x01, 4, 244)); // v_mov_b32 v4, 2.0
        words.extend_from_slice(&enc_mimg(0x24, 0xF, 1, 2, 5, 2, 1)); // image_sample_l
        words.extend_from_slice(&enc_exp(0, 0xF, false, false, false, [5, 6, 7, 8])); // mrt0
        words.extend_from_slice(&enc_exp(1, 0xF, false, false, false, [8, 7, 6, 5])); // mrt1
        words.extend_from_slice(&enc_exp(8, 0x1, false, true, true, [4, 0, 0, 0])); // mrtz
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("multi-MRT + MRTZ PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0);
        assert_eq!(out.binding_summary.output_locations, vec![0, 1]);
        assert!(out.binding_summary.exports_depth);
        assert!(out.wgsl.contains("@builtin(frag_depth)"));
        assert!(out.wgsl.contains("textureSampleLevel("));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_vop3p(
        op: u8,
        vdst: u8,
        src0: u16,
        src1: u16,
        src2: u16,
        clamp: bool,
        op_sel: u8,
        op_sel_hi: u8,
        neg_lo: u8,
        neg_hi: u8,
    ) -> [u32; 2] {
        let w0 = (0b110011u32 << 26)
            | (u32::from(op & 0x7F) << 16)
            | (u32::from(clamp) << 15)
            | (u32::from((op_sel_hi >> 2) & 0x1) << 14)
            | (u32::from(op_sel & 0x7) << 11)
            | (u32::from(neg_hi & 0x7) << 8)
            | u32::from(vdst);
        let w1 = (u32::from(neg_lo & 0x7) << 29)
            | (u32::from(op_sel_hi & 0x3) << 27)
            | (u32::from(src2 & 0x1FF) << 18)
            | (u32::from(src1 & 0x1FF) << 9)
            | u32::from(src0 & 0x1FF);
        [w0, w1]
    }

    fn enc_mubuf(op: u8, offset: u16, vdata: u8, vaddr: u8, srsrc: u8, idxen: bool) -> [u32; 2] {
        let w0 = (0b111000u32 << 26)
            | (u32::from(op & 0x7F) << 18)
            | (u32::from(idxen) << 13)
            | u32::from(offset & 0xFFF);
        let w1 = (128u32 << 24)
            | (u32::from(srsrc & 0x1F) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    fn enc_mtbuf(
        op: u8,
        format: u8,
        offset: u16,
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        idxen: bool,
    ) -> [u32; 2] {
        let w0 = (0b111010u32 << 26)
            | (u32::from(format & 0x7F) << 19)
            | (u32::from(op & 0x7) << 16)
            | (u32::from(idxen) << 13)
            | u32::from(offset & 0xFFF);
        let w1 = (128u32 << 24)
            | (u32::from((op >> 3) & 0x1) << 21)
            | (u32::from(srsrc & 0x1F) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    #[test]
    fn translates_frame1_vertex_shader_with_24b_offsets_and_all_new_opcodes() {
        // Synthetic Frame-1 vertex shader exercising:
        //   - s_setprio 0
        //   - s_mov_b64 s[6:7], s[2:3]
        //   - s_bfe_u32 s8, s6, 0x00080004 (offset=4, width=8)
        //   - s_load_dwordx4 s[12:15], s[0:1], 0x0
        //   - buffer_load_dwordx3 v[1:3], v0, s[4:7], 0 idxen offset:0       -> @location(0) position: vec3<f32>
        //   - tbuffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen offset:12 -> @location(1) color: vec4<f32>
        //   - buffer_load_dwordx2 v[8:9], v0, s[4:7], 0 idxen offset:16      -> @location(2) uv: vec2<f32>
        //   - v_fmaak_f16 v10, v8, v9, 0x00003c00 (f16 1.0)
        //   - v_cmp_gt_f32 vcc, v1, 0.0
        //   - v_cndmask_b32 v11, v2, v3, vcc (VOP2 opcode 0x00!)
        //   - v_cndmask_b32 v12, v4, v5, s8  (VOP3 opcode 0x100!)
        //   - exp pos0 v1, v11, v3, v7
        //   - exp param0 v12, v5, v6, v7
        //   - exp param1 v8, v9, v10, off done
        //   - s_endpgm
        let mut words = vec![
            enc_sopp(0x0F, 0),            // s_setprio 0
            enc_sop1(0x04, 6, 2),         // s_mov_b64 s[6:7], s[2:3]
            enc_sop2(0x27, 8, 6, 0xFF),   // s_bfe_u32 s8, s6, 0x00080004
            0x0008_0004,
        ];
        words.extend_from_slice(&enc_smem(0x02, 12, 0, 0x7D, 0, false, false)); // s_load_dwordx4 -> cb0
        words.extend_from_slice(&enc_mubuf(0x0F, 0, 1, 0, 1, true)); // buffer_load_dwordx3 @ offset 0
        words.extend_from_slice(&enc_mtbuf(0x03, 4, 12, 4, 0, 1, true)); // tbuffer_load_format_xyzw @ offset 12
        words.extend_from_slice(&enc_mubuf(0x0D, 16, 8, 0, 1, true)); // buffer_load_dwordx2 @ offset 16
        words.push(enc_vop2(0x38, 10, 256 + 8, 9)); // v_fmaak_f16 v10, v8, v9, 0x3c00
        words.push(0x0000_3C00); // f16 1.0
        words.push(enc_vopc(0x04, 256 + 1, 1)); // v_cmp_gt_f32 vcc, v1, v1
        words.push(enc_vop2(0x00, 11, 256 + 2, 3)); // v_cndmask_b32 (VOP2 0x00)
        words.extend_from_slice(&enc_vop3(
            0x100,
            12,
            256 + 4,
            256 + 5,
            8,
            false,
            0,
            0,
            0,
        )); // v_cndmask_b32 (VOP3 0x100)
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [1, 11, 3, 7]));
        words.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [12, 5, 6, 7]));
        words.extend_from_slice(&enc_exp(33, 0x7, false, true, false, [8, 9, 10, 0]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("Frame-1 synthetic VS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.entry_point, "vs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.vertex_input_mode, "attributes");
        assert!(out.binding_summary.has_cbv);
        assert_eq!(
            out.bindings,
            vec![WgslBindingInfo {
                group: 0,
                binding: 0,
                kind: "uniform",
                name: "cb0".to_string(),
                min_binding_size_bytes: Some(16),
                highest_vec4_offset: Some(0),
                view_dimension: None,
                source: None,
                storage_format: None,
            }]
        );

        // Distinct @location(0), @location(1), @location(2) with exact types & no `enable f16;`.
        assert!(out.wgsl.contains("@location(0) position: vec3<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(1) color: vec4<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(2) uv: vec2<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("vs_in.position"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("vs_in.color"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("vs_in.uv"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("select("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"), "must not require shader-f16");
        assert!(!out.wgsl.contains("<f16>"), "must not emit f16 type");
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"bindings\": ["), "JSON:\n{json}");
        assert!(
            json.contains("{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 16, \"highest_vec4_offset\": 0}"),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn maps_arbitrary_vertex_fetch_offsets_deterministically_without_collisions() {
        // Fetches at arbitrary non-canonical offsets 8 (vec2), 24 (vec4), 40 (f32).
        let mut words = Vec::new();
        words.extend_from_slice(&enc_mubuf(0x0D, 24, 4, 0, 1, true)); // offset 24 (2 comps)
        words.extend_from_slice(&enc_mubuf(0x0E, 40, 8, 0, 1, true)); // offset 40 (4 comps)
        words.extend_from_slice(&enc_mtbuf(0x00, 4, 8, 1, 0, 1, true)); // offset 8 (1 comp)
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [1, 4, 8, 11]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("arbitrary-offset VS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0);
        // Sorted by ascending byte offset: 8 -> loc 0 (f32), 24 -> loc 1 (vec2), 40 -> loc 2 (vec4).
        // (Naga's WGSL namer appends `_` to identifiers ending in digits.)
        assert!(out.wgsl.contains("@location(0) attr_8_: f32"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(1) attr_24_: vec2<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(2) attr_40_: vec4<f32>"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_frame1_pixel_shader_with_v_cndmask_and_v_fma_mix_feeding_compr_mrt0() {
        // Synthetic Frame-1 pixel shader exercising:
        //   - v_interp_p1/p2 attr0.xyzw (color/uv input)
        //   - v_cmp_gt_f32 vcc, v2, 0.5
        //   - v_cndmask_b32 v6, v2, 1.0, vcc (VOP2 opcode 0x00!)
        //   - v_fma_mixlo_f16 v8, v6, 0.5, 0.25 clamp  (low 16 bits -> R)
        //   - v_fma_mixhi_f16 v8, v3, 1.0, 0.0         (high 16 bits -> G, preserving R!)
        //   - v_fma_mixlo_f16 v9, v4, 1.0, 0.0         (low 16 bits -> B)
        //   - v_fma_mixhi_f16 v9, 1.0, 1.0, 0.0        (high 16 bits -> A, preserving B!)
        //   - exp mrt0 v8, v9, off, off done compr vm
        //   - s_endpgm
        let mut words = vec![
            enc_vintrp(0, 2, 0, 0, 0),
            enc_vintrp(1, 2, 0, 0, 1),
            enc_vintrp(0, 3, 0, 1, 0),
            enc_vintrp(1, 3, 0, 1, 1),
            enc_vintrp(0, 4, 0, 2, 0),
            enc_vintrp(1, 4, 0, 2, 1),
            enc_vopc(0x04, 240, 2), // v_cmp_gt_f32 vcc, 0.5, v2
            enc_vop2(0x00, 6, 242, 2), // v_cndmask_b32 v6, 1.0, v2, vcc (VOP2 0x00)
        ];
        // v_fma_mixlo_f16 v8, v6, 0.5, 0x3e800000 (0.25f) clamp (f32 sources: op_sel_hi = 0)
        words.extend_from_slice(&enc_vop3p(
            0x21,
            8,
            256 + 6,
            240,
            0xFF,
            true,
            0,
            0,
            0,
            0,
        ));
        words.push(0.25f32.to_bits());
        // v_fma_mixhi_f16 v8, v3, 1.0, 0.0 (op_sel_hi = 0, neg_hi = 0b001 abs on src0)
        words.extend_from_slice(&enc_vop3p(
            0x22,
            8,
            256 + 3,
            242,
            128,
            false,
            0,
            0,
            0,
            0b001,
        ));
        // v_fma_mixlo_f16 v9, v8.lo (f16 source! op_sel_hi = 0b001, op_sel = 0b000), v4, 0.0
        words.extend_from_slice(&enc_vop3p(
            0x21,
            9,
            256 + 8,
            256 + 4,
            128,
            false,
            0b000,
            0b001,
            0,
            0,
        ));
        // v_fma_mixhi_f16 v9, 1.0, 1.0, 0.0
        words.extend_from_slice(&enc_vop3p(
            0x22,
            9,
            242,
            242,
            128,
            false,
            0,
            0,
            0,
            0,
        ));
        words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [8, 9, 0, 0])); // compr vm done
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("Frame-1 synthetic PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.entry_point, "fs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.bindings.is_empty());

        // Entry point must be fs_main, packing + unpacking v8/v9 with real fma computations.
        assert!(out.wgsl.contains("fn fs_main("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("fma("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("select("), "WGSL:\n{}", out.wgsl);
        // Verify bit-preserving masks for mixlo (0xFFFFu / 0xFFFF0000u) and mixhi.
        assert!(
            out.wgsl.contains("65535u") || out.wgsl.contains("0xffffu"),
            "expected low-16 mask in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("4294901760u") || out.wgsl.contains("0xffff0000u"),
            "expected high-16 mask in WGSL:\n{}",
            out.wgsl
        );
        // Must NOT hardcode zero mrt_0 export and must NOT emit f16 type.
        assert!(!out.wgsl.contains("enable f16"), "must not require shader-f16");
        assert!(!out.wgsl.contains("<f16>"), "must not emit f16 type");
        assert!(out.wgsl.contains("ps_out.mrt_0_ = vec4<f32>("), "WGSL:\n{}", out.wgsl);
        assert!(
            !out.wgsl.contains("ps_out.mrt_0_ = vec4<f32>(0f, 0f, 0f, 0f)"),
            "compressed mrt0 export must unpack computed v_fma_mix* halves:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"entry_point\": \"fs_main\""), "JSON:\n{json}");
        assert!(json.contains("\"bindings\": []"), "JSON:\n{json}");
    }

    #[test]
    fn bindings_json_accurately_lists_all_emitted_group_binding_declarations() {
        // CBV + MIMG shader -> group 0 binding 0 (cb0), binding 1 (tex0), binding 2 (samp0).
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x02, 4, 0, 0x7D, 0, false, false));
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 2, 1));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default()).unwrap();
        assert_eq!(out.bindings.len(), 3);
        assert_eq!(out.bindings[0].group, 0);
        assert_eq!(out.bindings[0].binding, 0);
        assert_eq!(out.bindings[0].kind, "uniform");
        assert_eq!(out.bindings[0].name, "cb0");
        assert_eq!(out.bindings[0].min_binding_size_bytes, Some(16));
        assert_eq!(out.bindings[0].highest_vec4_offset, Some(0));
        assert_eq!(out.bindings[1].group, 0);
        assert_eq!(out.bindings[1].binding, 1);
        assert_eq!(out.bindings[1].kind, "texture_2d");
        assert_eq!(out.bindings[1].name, "tex0");
        assert_eq!(out.bindings[1].min_binding_size_bytes, None);
        assert_eq!(out.bindings[1].highest_vec4_offset, None);
        assert_eq!(out.bindings[2].group, 0);
        assert_eq!(out.bindings[2].binding, 2);
        assert_eq!(out.bindings[2].kind, "sampler");
        assert_eq!(out.bindings[2].name, "samp0");
        assert_eq!(out.bindings[2].min_binding_size_bytes, None);
        assert_eq!(out.bindings[2].highest_vec4_offset, None);

        // Every declared binding in `out.bindings` must match a real `@group(g) @binding(b)` in WGSL.
        for b in &out.bindings {
            let needle = format!("@group({}) @binding({})", b.group, b.binding);
            assert!(out.wgsl.contains(&needle), "missing {needle} in WGSL:\n{}", out.wgsl);
        }

        let json = out.to_json();
        assert!(
            json.contains("{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 16, \"highest_vec4_offset\": 0}"),
            "JSON:\n{json}"
        );
        assert!(
            json.contains("{\"group\": 0, \"binding\": 1, \"kind\": \"texture_2d\", \"name\": \"tex0\", \"view_dimension\": \"2d\"}"),
            "JSON:\n{json}"
        );
        assert!(
            json.contains("{\"group\": 0, \"binding\": 2, \"kind\": \"sampler\", \"name\": \"samp0\"}"),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn translates_textured_splash_ui_ps_with_cbv_image_sample_lz_v_pk_fma_f16_and_pkrtz_compr() {
        // Synthetic textured splash/UI pixel shader exercising:
        //   s_wqm_b64 exec, exec
        //   v_interp_p1_f32 / v_interp_p2_f32 attr0.xy (UV)
        //   s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0 (tint/bias CBV)
        //   s_waitcnt lgkmcnt(0)
        //   image_sample_lz v4, v2, s[8:15], s[12:15] dmask:0xf dim:1
        //   s_waitcnt vmcnt(0)
        //   v_cvt_pkrtz_f16_f32 v8, v4, v5
        //   v_cvt_pkrtz_f16_f32 v9, v6, v7
        //   v_pk_fma_f16 v8, v8, s4, s5 op_sel:[0,1,0] op_sel_hi:[1,0,1] neg_lo:[0,0,1] neg_hi:[1,0,0] clamp
        //   v_pk_fma_f16 v9, v9, s6, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0] clamp
        //   exp mrt0 v8, v9, off, off done compr vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_sop1(0x0A, 126, 126)); // s_wqm_b64 exec, exec
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x0, false, false)); // s_buffer_load_dwordx4
        words.push(enc_sopp(0x0C, 0xC07F)); // s_waitcnt lgkmcnt(0)
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 4, 2, 3)); // image_sample_lz
        words.push(enc_sopp(0x0C, 0x3F70)); // s_waitcnt vmcnt(0)
        words.push(enc_vop2(0x2F, 8, 256 + 4, 5)); // v_cvt_pkrtz_f16_f32 v8, v4, v5
        words.push(enc_vop2(0x2F, 9, 256 + 6, 7)); // v_cvt_pkrtz_f16_f32 v9, v6, v7
        words.extend_from_slice(&enc_vop3p(
            0x0E,
            8,
            256 + 8,
            4, // s4
            5, // s5
            true,
            0b010, // op_sel: src1 uses hi for lo lane
            0b101, // op_sel_hi: src1 uses lo for hi lane (.yx swap)
            0b100, // neg_lo on src2
            0b001, // neg_hi on src0
        ));
        words.extend_from_slice(&enc_vop3p(
            0x0E,
            9,
            256 + 9,
            6,   // s6
            242, // inline 1.0 broadcast via op_sel_hi bit 2 = 0
            true,
            0b000,
            0b011,
            0b000,
            0b000,
        ));
        words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [8, 9, 0, 0])); // compr done vm
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("textured splash/UI PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.entry_point, "fs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.translated_op_count, 15);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 0,
                    kind: "uniform",
                    name: "cb0".to_string(),
                    min_binding_size_bytes: Some(16),
                    highest_vec4_offset: Some(0),
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 1,
                    kind: "texture_2d",
                    name: "tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: Some("2d"),
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
            ]
        );
        assert!(out.wgsl.contains("textureSampleLevel("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("fma("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("clamp("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"), "must not require shader-f16");
        assert!(!out.wgsl.contains("<f16>"), "must not emit f16 type");
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_vs_and_fs_with_v_pk_ops_v_med3_f32_image_sample_l_and_s_buffer_load_x8_x16() {
        // 1) Synthetic Vertex Shader exercising s_buffer_load_dwordx8 + x16 across two CBVs (cb0, cb1),
        //    v_pk_mul_f16, v_pk_add_f16, v_pk_min_f16, v_pk_max_f16, v_pk_fmac_f16, v_med3_f32,
        //    and image_sample_l.
        let mut vs_words = Vec::new();
        // s_buffer_load_dwordx8 s[4:11], s[0:3], 0x20 (cb0 words 8..15)
        vs_words.extend_from_slice(&enc_smem(0x0B, 4, 0, 0x7D, 0x20, false, false));
        // s_buffer_load_dwordx16 s[16:31], s[4:7], 0x40 (cb1 words 16..31)
        vs_words.extend_from_slice(&enc_smem(0x0C, 16, 2, 0x7D, 0x40, false, false));
        vs_words.push(enc_sopp(0x0C, 0xC07F));
        vs_words.push(enc_vop1(0x06, 1, 256 + 0)); // v_cvt_f32_u32 v1, v0
        vs_words.push(enc_vop1(0x01, 2, 240)); // v_mov_b32 v2, 0.5
        // v_med3_f32 v2, v1, s4, s5
        vs_words.extend_from_slice(&enc_vop3(0x157, 2, 256 + 1, 4, 5, true, 0, 0, 0));
        // v_pk_mul_f16 v3, v1, s6 op_sel:[0,0] op_sel_hi:[1,1]
        vs_words.extend_from_slice(&enc_vop3p(0x10, 3, 256 + 1, 6, 0, false, 0, 0b011, 0, 0));
        // v_pk_add_f16 v3, v3, s7 op_sel:[0,1] op_sel_hi:[1,0] neg_lo:[0,1] clamp
        vs_words.extend_from_slice(&enc_vop3p(
            0x0F,
            3,
            256 + 3,
            7,
            0,
            true,
            0b010,
            0b001,
            0b010,
            0,
        ));
        // v_pk_min_f16 v4, v3, s8
        vs_words.extend_from_slice(&enc_vop3p(0x11, 4, 256 + 3, 8, 0, false, 0, 0b011, 0, 0));
        // v_pk_max_f16 v4, v4, s16
        vs_words.extend_from_slice(&enc_vop3p(0x12, 4, 256 + 4, 16, 0, false, 0, 0b011, 0, 0));
        // v_pk_fmac_f16 v4, v3, s31 (VOP2 0x3C)
        vs_words.push(enc_vop2(0x3C, 4, 31, 3));
        // image_sample_l v5, v0, s[8:15], s[12:15] dmask:0xf dim:1
        vs_words.extend_from_slice(&enc_mimg(0x24, 0xF, 1, 0, 5, 2, 3));
        vs_words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [5, 6, 7, 8])); // pos0
        vs_words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [2, 4, 0, 0])); // param0 done
        vs_words.push(enc_sopp(0x01, 0));

        let vs_bytes = words_to_bytes(&vs_words);
        let vs_out = translate_rdna2_to_wgsl(&vs_bytes, &Rdna2TranslateOptions::default())
            .expect("VS with v_pk_*_f16, v_med3_f32, image_sample_l, and s_buffer_load_x8/x16 must succeed");
        assert!(vs_out.naga_validated);
        assert_eq!(vs_out.stage, ShaderStage::Vertex);
        assert_eq!(vs_out.unsupported_op_count, 0, "WGSL:\n{}", vs_out.wgsl);
        assert_eq!(vs_out.bindings.len(), 4); // cb0@0, tex0@1, samp0@2, cb1@3
        assert_eq!(vs_out.bindings[0].name, "cb0");
        assert_eq!(vs_out.bindings[0].min_binding_size_bytes, Some(64));
        assert_eq!(vs_out.bindings[0].highest_vec4_offset, Some(3));
        assert_eq!(vs_out.bindings[1].name, "tex0");
        assert_eq!(vs_out.bindings[2].name, "samp0");
        assert_eq!(vs_out.bindings[3].name, "cb1");
        assert_eq!(vs_out.bindings[3].binding, 3);
        assert_eq!(vs_out.bindings[3].min_binding_size_bytes, Some(128));
        assert_eq!(vs_out.bindings[3].highest_vec4_offset, Some(7));
        assert!(vs_out.wgsl.contains("textureSampleLevel("));
        assert!(
            vs_out.wgsl.contains("cb0_.data[2]") && vs_out.wgsl.contains("cb0_.data[3]"),
            "expected cb0_.data[2..4] in WGSL:\n{}",
            vs_out.wgsl
        );
        assert!(
            vs_out.wgsl.contains("cb1_.data[4]") && vs_out.wgsl.contains("cb1_.data[7]"),
            "expected cb1_.data[4..8] in WGSL:\n{}",
            vs_out.wgsl
        );
        assert!(!vs_out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&vs_out.wgsl);

        // 2) Synthetic Fragment Shader exercising s_buffer_load_dwordx8, image_sample_l,
        //    v_med3_f32, v_pk_mul_f16, v_pk_add_f16, and VOP3 v_cvt_pkrtz_f16_f32 clamp.
        let mut fs_words = Vec::new();
        fs_words.extend_from_slice(&enc_smem(0x0B, 4, 0, 0x7D, 0x0, false, false)); // s_buffer_load_dwordx8
        fs_words.push(enc_vintrp(0, 2, 0, 0, 0));
        fs_words.push(enc_vintrp(1, 2, 0, 0, 1));
        fs_words.push(enc_vintrp(0, 3, 0, 1, 0));
        fs_words.push(enc_vintrp(1, 3, 0, 1, 1));
        fs_words.push(enc_vop1(0x01, 4, 242)); // lod = 1.0 in v4
        fs_words.extend_from_slice(&enc_mimg(0x24, 0xF, 1, 2, 5, 2, 1)); // image_sample_l v[5:8], v[2:4]
        fs_words.extend_from_slice(&enc_vop3(0x157, 5, 256 + 5, 4, 5, false, 0, 0, 0)); // v_med3_f32
        // VOP3 v_cvt_pkrtz_f16_f32 (0x12E) with clamp
        fs_words.extend_from_slice(&enc_vop3(0x12F, 9, 256 + 5, 256 + 6, 0, true, 0, 0, 0));
        fs_words.extend_from_slice(&enc_vop3(0x12F, 10, 256 + 7, 256 + 8, 0, true, 0, 0, 0));
        fs_words.extend_from_slice(&enc_vop3p(0x10, 9, 256 + 9, 6, 0, false, 0, 0b011, 0, 0)); // v_pk_mul_f16
        fs_words.extend_from_slice(&enc_vop3p(0x0F, 9, 256 + 9, 7, 0, true, 0, 0b011, 0, 0)); // v_pk_add_f16
        fs_words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [9, 10, 0, 0]));
        fs_words.push(enc_sopp(0x01, 0));

        let fs_bytes = words_to_bytes(&fs_words);
        let fs_out = translate_rdna2_to_wgsl(&fs_bytes, &Rdna2TranslateOptions::default())
            .expect("FS with v_pk_mul/add_f16, v_med3_f32, image_sample_l, and s_buffer_load_x8 must succeed");
        assert!(fs_out.naga_validated);
        assert_eq!(fs_out.stage, ShaderStage::Fragment);
        assert_eq!(fs_out.unsupported_op_count, 0, "WGSL:\n{}", fs_out.wgsl);
        assert!(fs_out.wgsl.contains("textureSampleLevel("));
        assert!(fs_out.wgsl.contains("pack2x16float("));
        assert!(fs_out.wgsl.contains("unpack2x16float("));
        assert!(!fs_out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&fs_out.wgsl);
    }

    #[test]
    fn translates_all_mimg_sampling_and_load_variants_with_dmask_and_multi_texture_bindings() {
        // Fragment shader exercising all 7 required MIMG variants + partial dmasks + 2 textures:
        //   image_sample      (0x20) dmask:0xf on tex0 (srsrc=2)
        //   image_sample_lz   (0x27) dmask:0x3 (RG compacted to v10..v11) on tex0
        //   image_sample_l    (0x24) dmask:0x9 (RA compacted to v12..v13) on tex1 (srsrc=4)
        //   image_sample_b    (0x25) dmask:0xf (bias in v1, uv in v2..v3) on tex0
        //   image_sample_c    (0x28) dmask:0x1 (z_ref in v1, uv in v2..v3 -> shadow factor in v18) on tex1
        //   image_load        (0x00) dmask:0x3 on tex0
        //   image_load_mip    (0x01) dmask:0x5 (RB compacted to v21..v22) on tex1
        let mut words = Vec::new();
        words.push(enc_vop1(0x01, 1, 240)); // v1 = 0.5 (bias / z_ref)
        words.push(enc_vintrp(0, 2, 0, 0, 0)); // v2 = uv.x
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0)); // v3 = uv.y
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.push(enc_vop1(0x01, 4, 244)); // v4 = 2.0 (lod / mip)
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 6, 2, 1)); // image_sample -> v6..v9
        words.extend_from_slice(&enc_mimg(0x27, 0x3, 1, 2, 10, 2, 1)); // image_sample_lz (RG) -> v10..v11
        words.extend_from_slice(&enc_mimg(0x24, 0x9, 1, 2, 12, 4, 1)); // image_sample_l (RA) on tex1 -> v12..v13
        words.extend_from_slice(&enc_mimg(0x25, 0xF, 1, 1, 14, 2, 1)); // image_sample_b -> v14..v17
        words.extend_from_slice(&enc_mimg(0x28, 0x1, 1, 1, 18, 4, 1)); // image_sample_c on tex1 -> v18
        words.extend_from_slice(&enc_mimg(0x00, 0x3, 1, 2, 19, 2, 0)); // image_load -> v19..v20
        words.extend_from_slice(&enc_mimg(0x01, 0x5, 1, 2, 21, 4, 0)); // image_load_mip on tex1 -> v21..v22
        // Combine into mrt0
        words.push(enc_vop2(0x03, 6, 256 + 10, 6));
        words.push(enc_vop2(0x03, 7, 256 + 12, 14));
        words.push(enc_vop2(0x08, 8, 256 + 18, 19));
        words.push(enc_vop2(0x03, 9, 256 + 21, 9));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [6, 7, 8, 9]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("all MIMG sampling & load variants must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 1,
                    kind: "texture_2d",
                    name: "tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: Some("2d"),
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: FRAGMENT_EXTRA_BINDING_BASE,
                    kind: "texture_2d",
                    name: "tex1".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: Some("2d"),
                    source: None,
                    storage_format: None,
                },
            ]
        );
        assert!(out.wgsl.contains("textureSample("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("textureSampleLevel("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("textureSampleBias("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("textureLoad("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("tex0") && out.wgsl.contains("tex1"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_common_gfx10_3_scalar_and_vector_alu_ops_with_zero_unsupported() {
        // Exercises all common UI/splash scalar & vector ALU ops:
        //   s_lshl_b32, s_lshr_b32, s_and_b32, s_or_b32, s_xor_b32, s_pack_ll/lh/hh_b32_b16,
        //   v_med3_f32, v_floor_f32, v_fract_f32, v_rcp_f32, v_rsq_f32, v_sqrt_f32,
        //   v_exp_f32, v_log_f32, v_cvt_f32_ubyte0..3, v_bfi_b32, v_lshl_or_b32, v_or3_b32,
        //   v_med3_i32, v_med3_u32, v_rcp_f16, v_sqrt_f16
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 2, 128 + 15)); // s_mov_b32 s2, 15
        words.push(enc_sop2(0x1E, 3, 2, 128 + 2)); // s_lshl_b32 s3, s2, 2
        words.push(enc_sop2(0x20, 4, 3, 128 + 1)); // s_lshr_b32 s4, s3, 1
        words.push(enc_sop2(0x0E, 5, 4, 128 + 7)); // s_and_b32 s5, s4, 7
        words.push(enc_sop2(0x10, 6, 5, 128 + 8)); // s_or_b32 s6, s5, 8
        words.push(enc_sop2(0x12, 7, 6, 128 + 3)); // s_xor_b32 s7, s6, 3
        words.push(enc_sop2(0x32, 8, 5, 6)); // s_pack_ll_b32_b16 s8, s5, s6
        words.push(enc_sop2(0x33, 9, 8, 7)); // s_pack_lh_b32_b16 s9, s8, s7
        words.push(enc_sop2(0x34, 10, 9, 8)); // s_pack_hh_b32_b16 s10, s9, s8
        words.push(enc_vop1(0x06, 0, 7)); // v_cvt_f32_u32 v0, s7
        words.push(enc_vop1(0x24, 1, 256 + 0)); // v_floor_f32 v1, v0
        words.push(enc_vop1(0x20, 2, 256 + 0)); // v_fract_f32 v2, v0
        words.push(enc_vop1(0x2A, 3, 256 + 0)); // v_rcp_f32 v3, v0
        words.push(enc_vop1(0x2E, 4, 256 + 3)); // v_rsq_f32 v4, v3
        words.push(enc_vop1(0x33, 5, 256 + 4)); // v_sqrt_f32 v5, v4
        words.push(enc_vop1(0x25, 6, 256 + 5)); // v_exp_f32 v6, v5
        words.push(enc_vop1(0x27, 7, 256 + 6)); // v_log_f32 v7, v6
        words.extend_from_slice(&enc_vop3(0x157, 8, 256 + 7, 256 + 1, 256 + 2, true, 0, 0, 0)); // v_med3_f32
        words.push(enc_vop1(0x11, 9, 10)); // v_cvt_f32_ubyte0 v9, s10
        words.push(enc_vop1(0x12, 10, 10)); // v_cvt_f32_ubyte1 v10, s10
        words.push(enc_vop1(0x13, 11, 10)); // v_cvt_f32_ubyte2 v11, s10
        words.push(enc_vop1(0x14, 12, 10)); // v_cvt_f32_ubyte3 v12, s10
        words.extend_from_slice(&enc_vop3(0x14A, 13, 10, 256 + 9, 256 + 10, false, 0, 0, 0)); // v_bfi_b32
        words.extend_from_slice(&enc_vop3(0x36E, 14, 256 + 13, 128 + 2, 256 + 11, false, 0, 0, 0)); // v_lshl_or_b32
        words.extend_from_slice(&enc_vop3(0x370, 15, 256 + 14, 256 + 12, 7, false, 0, 0, 0)); // v_or3_b32
        words.extend_from_slice(&enc_vop3(0x158, 16, 256 + 13, 256 + 14, 256 + 15, false, 0, 0, 0)); // v_med3_i32
        words.extend_from_slice(&enc_vop3(0x159, 17, 256 + 13, 256 + 14, 256 + 15, false, 0, 0, 0)); // v_med3_u32
        words.push(enc_vop1(0x54, 18, 256 + 8)); // v_rcp_f16
        words.push(enc_vop1(0x55, 18, 256 + 18)); // v_sqrt_f16
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [8, 9, 17, 18]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("common GFX10.3 ALU ops must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("floor("));
        assert!(out.wgsl.contains("fract("));
        assert!(out.wgsl.contains("inverseSqrt("));
        assert!(out.wgsl.contains("sqrt("));
        assert!(out.wgsl.contains("exp2("));
        assert!(out.wgsl.contains("log2("));
        assert!(out.wgsl.contains("clamp("));
        assert!(!out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn tight_cbv_sizing_48_byte_frame1_cbv_avoids_256_byte_overdeclaration() {
        // Synthetic shader reading only cb[0]..cb[2] (byte offsets 0..44):
        //   s_buffer_load_dwordx4 s[4:7],  s[0:3], 0x00  (cb[0]: bytes 0..15)
        //   s_buffer_load_dwordx4 s[8:11], s[0:3], 0x10  (cb[1]: bytes 16..31)
        //   s_buffer_load_dword   s12,     s[0:3], 0x2C  (cb[2].w: bytes 44..47)
        //   v_mov_b32 v0, s4
        //   v_add_f32 v1, s8, s12
        //   exp mrt0 v0, v1, v0, v1 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x00, false, false));
        words.extend_from_slice(&enc_smem(0x0A, 8, 0, 0x7D, 0x10, false, false));
        words.extend_from_slice(&enc_smem(0x08, 12, 0, 0x7D, 0x2C, false, false));
        words.push(enc_vop1(0x01, 0, 4));
        words.push(enc_vop2(0x03, 1, 8, 0));
        words.push(enc_vop2(0x03, 1, 12, 1));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("small 48-byte CBV shader should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.cbv_vec4_count, 3);
        assert_eq!(
            out.bindings,
            vec![WgslBindingInfo {
                group: 0,
                binding: 0,
                kind: "uniform",
                name: "cb0".to_string(),
                min_binding_size_bytes: Some(48),
                highest_vec4_offset: Some(2),
                view_dimension: None,
                source: None,
                storage_format: None,
            }]
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 3>"),
            "expected tightly sized 3-vec4 (48 B) CBV array in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("array<vec4<u32>, 16>"),
            "must NOT over-declare 256-byte CBV for 48-byte read:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("cb0_.data[2]"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 48, \"highest_vec4_offset\": 2}"
            ),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn tight_cbv_sizing_large_cbv_reads_vec4_20_without_clamping() {
        // Synthetic shader reading cb[20] (byte offset 20 * 16 = 320 = 0x140):
        //   s_buffer_load_dwordx4 s[4:7], s[0:3], 0x140
        //   v_mov_b32 v0, s4
        //   v_mov_b32 v1, s7
        //   exp mrt0 v0, v1, v0, v1 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x140, false, false));
        words.push(enc_vop1(0x01, 0, 4));
        words.push(enc_vop1(0x01, 1, 7));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("large 336-byte CBV shader should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.cbv_vec4_count, 21);
        assert_eq!(
            out.bindings,
            vec![WgslBindingInfo {
                group: 0,
                binding: 0,
                kind: "uniform",
                name: "cb0".to_string(),
                min_binding_size_bytes: Some(336),
                highest_vec4_offset: Some(20),
                view_dimension: None,
                source: None,
                storage_format: None,
            }]
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 21>"),
            "expected 21-vec4 (336 B) CBV array in WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("cb0_.data[20]"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 336, \"highest_vec4_offset\": 20}"
            ),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn tight_cbv_sizing_multiple_cbvs_sized_independently() {
        // Synthetic shader with two CBVs:
        //   cb0 (sbase=0): reads vec4 1 (offset 0x10) -> N=2 (32 bytes), highest_vec4_offset=1
        //   cb1 (sbase=2): reads vec4 5 (offset 0x50) -> N=6 (96 bytes), highest_vec4_offset=5
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x10, false, false));
        words.extend_from_slice(&enc_smem(0x0A, 8, 2, 0x7D, 0x50, false, false));
        words.push(enc_vop1(0x01, 0, 4));
        words.push(enc_vop1(0x01, 1, 8));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("multi-CBV shader should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.cbv_vec4_count, 6);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 0,
                    kind: "uniform",
                    name: "cb0".to_string(),
                    min_binding_size_bytes: Some(32),
                    highest_vec4_offset: Some(1),
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: FRAGMENT_EXTRA_BINDING_BASE,
                    kind: "uniform",
                    name: "cb1".to_string(),
                    min_binding_size_bytes: Some(96),
                    highest_vec4_offset: Some(5),
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
            ]
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 2>"),
            "expected cb0 sized to 2 vec4s (32 B):\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 6>"),
            "expected cb1 sized to 6 vec4s (96 B):\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("cb0_.data[1]"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("cb1_.data[5]"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
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
    fn tight_cbv_sizing_dynamic_soffset_falls_back_to_max_static_or_16() {
        // Case A: Dynamic soffset with small static offset (0x04) -> falls back to 16 vec4s (256 B).
        let mut words_small = Vec::new();
        words_small.push(enc_sop1(0x03, 2, 128 + 16)); // s_mov_b32 s2, 16
        words_small.extend_from_slice(&enc_smem(0x09, 4, 0, 2, 0x04, false, false)); // dynamic soffset=s2
        words_small.push(enc_vop1(0x01, 0, 4));
        words_small.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 0, 0, 0]));
        words_small.push(enc_sopp(0x01, 0));

        let out_small = translate_rdna2_to_wgsl(
            &words_to_bytes(&words_small),
            &Rdna2TranslateOptions::default(),
        )
        .expect("dynamic CBV shader should translate cleanly");
        assert!(out_small.naga_validated);
        assert_eq!(out_small.bindings[0].min_binding_size_bytes, Some(256));
        assert_eq!(out_small.bindings[0].highest_vec4_offset, Some(0));
        assert!(
            out_small.wgsl.contains("array<vec4<u32>, 16>"),
            "WGSL:\n{}",
            out_small.wgsl
        );
        assert_naga_wgsl_valid(&out_small.wgsl);

        // Case B: Dynamic soffset + static read at cb[20] -> max(20 + 1, 16) = 21 vec4s (336 B).
        let mut words_large = Vec::new();
        words_large.push(enc_sop1(0x03, 2, 128 + 16));
        words_large.extend_from_slice(&enc_smem(0x09, 4, 0, 2, 0x04, false, false));
        words_large.extend_from_slice(&enc_smem(0x0A, 8, 0, 0x7D, 0x140, false, false));
        words_large.push(enc_vop1(0x01, 0, 4));
        words_large.push(enc_vop1(0x01, 1, 8));
        words_large.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        words_large.push(enc_sopp(0x01, 0));

        let out_large = translate_rdna2_to_wgsl(
            &words_to_bytes(&words_large),
            &Rdna2TranslateOptions::default(),
        )
        .expect("dynamic + large static CBV shader should translate cleanly");
        assert!(out_large.naga_validated);
        assert_eq!(out_large.bindings[0].min_binding_size_bytes, Some(336));
        assert_eq!(out_large.bindings[0].highest_vec4_offset, Some(20));
        assert!(
            out_large.wgsl.contains("array<vec4<u32>, 21>"),
            "WGSL:\n{}",
            out_large.wgsl
        );
        assert_naga_wgsl_valid(&out_large.wgsl);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_mubuf_full(
        op: u8,
        offset: u16,
        offen: bool,
        idxen: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
    ) -> [u32; 2] {
        let w0 = (0b111000u32 << 26)
            | (u32::from(op & 0x7F) << 18)
            | (u32::from(idxen) << 13)
            | (u32::from(offen) << 12)
            | u32::from(offset & 0xFFF);
        let w1 = (u32::from(soffset) << 24)
            | (u32::from(srsrc & 0x1F) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    fn enc_ds(op: u8, offset0: u8, offset1: u8, addr: u8, data0: u8, data1: u8, vdst: u8) -> [u32; 2] {
        let w0 = (0b110110u32 << 26)
            | (u32::from(op) << 18)
            | (u32::from(offset1) << 8)
            | u32::from(offset0);
        let w1 = (u32::from(vdst) << 24)
            | (u32::from(data1) << 16)
            | (u32::from(data0) << 8)
            | u32::from(addr);
        [w0, w1]
    }

    #[test]
    fn translates_compute_buffer_copy_transform_with_storage_bindings_and_bounds_clamping() {
        // Compute shader: global_idx = (wg_id.x << 6) + local_id.x, byte_off = global_idx << 4
        //   v_lshl_add_u32 v3, s0, 6, v0          ; s0 seeded from wg_id.x, v0 from local_id.x
        //   v_lshlrev_b32  v3, 4, v3              ; byte offset = global_idx * 16
        //   buffer_load_dwordx4 v[4:7], v3, s[4:7], 0 offen          (srsrc=1 -> read-only-storage)
        //   buffer_load_ubyte   v8,     v3, s[4:7], 0 offen offset:2 (srsrc=1 -> read-only-storage)
        //   buffer_load_ushort  v9,     v3, s[4:7], 0 offen offset:4 (srsrc=1 -> read-only-storage)
        //   v_add_f32 v4, v4, 1.0
        //   v_mul_f32 v5, v5, 2.0
        //   v_add_nc_u32 v6, v6, v8
        //   v_add_nc_u32 v7, v7, v9
        //   buffer_store_dwordx4 v[4:7], v3, s[8:11], 0 offen        (srsrc=2 -> storage)
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_vop3(0x346, 3, 0, 128 + 6, 256 + 0, false, 0, 0, 0)); // v_lshl_add_u32 v3, s0, 6, v0
        words.push(enc_vop2(0x1A, 3, 128 + 4, 3)); // v_lshlrev_b32 v3, 4, v3
        words.extend_from_slice(&enc_mubuf_full(0x0E, 0, true, false, 3, 4, 1, 128)); // buffer_load_dwordx4
        words.extend_from_slice(&enc_mubuf_full(0x08, 2, true, false, 3, 8, 1, 128)); // buffer_load_ubyte
        words.extend_from_slice(&enc_mubuf_full(0x0A, 4, true, false, 3, 9, 1, 128)); // buffer_load_ushort
        words.push(enc_vop2(0x03, 4, 242, 4)); // v_add_f32 v4, 1.0, v4
        words.push(enc_vop2(0x08, 5, 244, 5)); // v_mul_f32 v5, 2.0, v5
        words.push(enc_vop2(0x25, 6, 256 + 8, 6)); // v_add_nc_u32 v6, v8, v6
        words.push(enc_vop2(0x25, 7, 256 + 9, 7)); // v_add_nc_u32 v7, v9, v7
        words.extend_from_slice(&enc_mubuf_full(0x1E, 0, true, false, 3, 4, 2, 128)); // buffer_store_dwordx4
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("compute buffer copy/transform shader must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.entry_point, "cs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.binding_summary.has_storage_buffer);
        assert!(!out.binding_summary.has_storage_texture);
        assert!(!out.binding_summary.has_lds);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 3,
                    kind: "read-only-storage",
                    name: "sbuf0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 4,
                    kind: "storage",
                    name: "uav_buf0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
            ]
        );

        assert!(out.wgsl.contains("@compute @workgroup_size(64, 1, 1)"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@builtin(local_invocation_id) local_id: vec3<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@builtin(workgroup_id) wg_id: vec3<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@builtin(global_invocation_id) global_id: vec3<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var<storage> sbuf0_: array<u32>") || out.wgsl.contains("var<storage, read> sbuf0_: array<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var<storage, read_write> uav_buf0_: array<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("arrayLength("), "expected bounds clamping via arrayLength:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("var<workgroup>"), "must NOT emit LDS when no DS ops exist:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"), "must not emit enable f16");
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"kind\": \"read-only-storage\""), "JSON:\n{json}");
        assert!(json.contains("\"kind\": \"storage\""), "JSON:\n{json}");
        assert!(json.contains("\"has_storage_buffer\": true"), "JSON:\n{json}");
    }

    #[test]
    fn translates_compute_texture_filter_with_image_sample_lz_and_image_store() {
        // Compute 2D texture filter (workgroup_size 8x8x1):
        //   v_cvt_f32_u32 v4, v0   ; u = f32(local_id.x)
        //   v_cvt_f32_u32 v5, v1   ; v = f32(local_id.y)
        //   image_sample_lz v[6:9], v[4:5], s[8:15], s[4:7] dmask:0xf dim:1
        //   s_waitcnt vmcnt(0)
        //   v_mul_f32 v6, v6, 0.5
        //   image_store v[6:9], v[0:1], s[16:23] dmask:0xf dim:1
        //   image_store_mip v[6:9], v[0:2], s[16:23] dmask:0xf dim:1
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vop1(0x06, 4, 256 + 0));
        words.push(enc_vop1(0x06, 5, 256 + 1));
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 4, 6, 2, 1)); // image_sample_lz -> tex0@1, samp0@2
        words.push(enc_sopp(0x0C, 0x3F70));
        words.push(enc_vop2(0x08, 6, 240, 6));
        words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 6, 4, 0)); // image_store -> uav_tex0@3
        words.extend_from_slice(&enc_mimg(0x09, 0xF, 1, 0, 6, 4, 0)); // image_store_mip -> uav_tex0@3
        words.push(enc_sopp(0x01, 0));

        let opts = Rdna2TranslateOptions {
            workgroup_size: [8, 8, 1],
            ..Rdna2TranslateOptions::default()
        };
        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &opts)
            .expect("compute texture filter shader must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.entry_point, "cs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.binding_summary.has_sampler);
        assert!(out.binding_summary.has_storage_texture);
        assert!(!out.binding_summary.has_lds);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 1,
                    kind: "texture_2d",
                    name: "tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: Some("2d"),
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: None,
                    source: None,
                    storage_format: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 3,
                    kind: "storage-texture",
                    name: "uav_tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                    view_dimension: Some("2d"),
                    source: None,
                    storage_format: Some(WGPU_STORAGE_FORMAT_RGBA8_UNORM),
                },
            ]
        );

        assert!(out.wgsl.contains("@compute @workgroup_size(8, 8, 1)"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("texture_2d<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("texture_storage_2d<rgba8unorm,write>")
                || out.wgsl.contains("texture_storage_2d<rgba8unorm, write>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("textureSampleLevel("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("textureStore("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"kind\": \"storage-texture\""), "JSON:\n{json}");
        assert!(json.contains("\"has_storage_texture\": true"), "JSON:\n{json}");
    }

    #[test]
    fn translates_compute_lds_reduction_exchange_with_barrier_and_buffer_store() {
        // Compute LDS reduction/exchange covering ds_write_b32, ds_write2_b32, ds_write_b64,
        // ds_write_b128, s_barrier, ds_read_b32, ds_read2_b32, ds_read_b64, ds_read_b128,
        // and buffer_store_dword{,x2,x3,x4}:
        let mut words = Vec::new();
        words.push(enc_vop2(0x1A, 3, 128 + 2, 0)); // v_lshlrev_b32 v3, 2, v0 (byte addr = local_id.x * 4)
        words.push(enc_vop1(0x01, 4, 242)); // v4 = 1.0
        words.push(enc_vop1(0x01, 5, 244)); // v5 = 2.0
        words.push(enc_vop1(0x01, 6, 240)); // v6 = 0.5
        words.push(enc_vop1(0x01, 7, 246)); // v7 = 4.0
        words.extend_from_slice(&enc_ds(0x0D, 0, 0, 3, 4, 0, 0)); // ds_write_b32
        words.extend_from_slice(&enc_ds(0x0E, 1, 2, 3, 4, 5, 0)); // ds_write2_b32
        words.extend_from_slice(&enc_ds(0x4D, 16, 0, 3, 4, 0, 0)); // ds_write_b64
        words.extend_from_slice(&enc_ds(0xDF, 32, 0, 3, 4, 0, 0)); // ds_write_b128
        words.push(enc_sopp(0x0A, 0)); // s_barrier
        words.extend_from_slice(&enc_ds(0x36, 0, 0, 3, 0, 0, 8)); // ds_read_b32 -> v8
        words.extend_from_slice(&enc_ds(0x37, 1, 2, 3, 0, 0, 9)); // ds_read2_b32 -> v9..v10
        words.extend_from_slice(&enc_ds(0x76, 16, 0, 3, 0, 0, 11)); // ds_read_b64 -> v11..v12
        words.extend_from_slice(&enc_ds(0xFF, 32, 0, 3, 0, 0, 13)); // ds_read_b128 -> v13..v16
        words.push(enc_vop2(0x03, 8, 256 + 9, 8)); // v_add_f32 v8, v9, v8
        words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 3, 8, 1, 128)); // buffer_store_dword
        words.extend_from_slice(&enc_mubuf_full(0x1D, 16, true, false, 3, 11, 1, 128)); // buffer_store_dwordx2
        words.extend_from_slice(&enc_mubuf_full(0x1F, 32, true, false, 3, 13, 1, 128)); // buffer_store_dwordx3
        words.extend_from_slice(&enc_mubuf_full(0x0C, 0, true, false, 3, 17, 1, 128)); // buffer_load_dword on same UAV
        words.extend_from_slice(&enc_mubuf_full(0x0D, 8, true, false, 3, 18, 2, 128)); // buffer_load_dwordx2 on SRV
        words.extend_from_slice(&enc_mubuf_full(0x0F, 16, true, false, 3, 20, 2, 128)); // buffer_load_dwordx3 on SRV
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("compute LDS reduction shader must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.entry_point, "cs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.binding_summary.has_lds);
        assert!(out.binding_summary.has_storage_buffer);
        // srsrc=1 has both loads and stores -> merged into a single read_write `storage` binding.
        // srsrc=2 has only loads -> `read-only-storage` binding.
        assert_eq!(out.bindings.len(), 2);
        assert_eq!(out.bindings[0].kind, "storage");
        assert_eq!(out.bindings[1].kind, "read-only-storage");

        assert!(out.wgsl.contains("var<workgroup> lds: array<u32, 4096>;"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("workgroupBarrier();"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("4095u"), "expected safe 4095u LDS index mask:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn compute_workgroup_id_sgprs_respect_user_data_overwrite_and_custom_base() {
        // Case A: s0 overwritten by s_mov_b32 before read -> must NOT seed s_0 = wg_id.x,
        // while s1 is read before write -> MUST seed s_1 = wg_id.y.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 0, 128 + 42)); // s_mov_b32 s0, 42 (overwrites s0)
        words.push(enc_sop2(0x00, 3, 0, 1)); // s_add_u32 s3, s0, s1 (reads s1 before write)
        words.push(enc_vop1(0x01, 4, 3)); // v_mov_b32 v4, s3
        words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 4, 1, 128));
        words.push(enc_sopp(0x01, 0));

        let out_a = translate_rdna2_to_wgsl(&words_to_bytes(&words), &Rdna2TranslateOptions::default())
            .expect("compute shader should translate cleanly");
        assert!(out_a.naga_validated);
        assert!(!out_a.wgsl.contains("s_0_ = wg_id.x"), "WGSL:\n{}", out_a.wgsl);
        assert!(out_a.wgsl.contains("s_1_ = wg_id.y"), "WGSL:\n{}", out_a.wgsl);
        assert_naga_wgsl_valid(&out_a.wgsl);

        // Case B: Custom workgroup_id_sgpr_base = Some(4) -> seeds s4 = wg_id.x.
        let mut words_b = Vec::new();
        words_b.push(enc_vop2(0x25, 4, 4, 0)); // v_add_nc_u32 v4, s4, v0
        words_b.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 4, 1, 128));
        words_b.push(enc_sopp(0x01, 0));

        let opts_b = Rdna2TranslateOptions {
            workgroup_id_sgpr_base: Some(4),
            ..Rdna2TranslateOptions::default()
        };
        let out_b = translate_rdna2_to_wgsl(&words_to_bytes(&words_b), &opts_b)
            .expect("custom wg_id sgpr base compute shader should translate cleanly");
        assert!(out_b.naga_validated);
        assert!(out_b.wgsl.contains("s_4_ = wg_id.x"), "WGSL:\n{}", out_b.wgsl);
        assert_naga_wgsl_valid(&out_b.wgsl);
    }

    #[test]
    fn tint_const_eval_safe_nan_and_shift_overflow() {
        // Exercises all live Chrome/Dawn Tint const-eval rejection patterns from Test #38:
        //   1. bitcast<f32>(4294967295u) / bitcast<f32>(4294967288u) (inline -1 / -8 as f32)
        //   2. unpack2x16float(32505u) / unpack2x16float(4294967295u) (f16 NaN in v_fmaak_f16 & v_pk_add_f16)
        //   3. 4294967295u << 16u (s_pack_ll_b32_b16), 3204448256u << 3u (s_lshl3_add_u32),
        //      4294967295u << 1u (s_lshl1_add_u32), and u32 const add/sub/mul overflow.
        let mut words = vec![
            // s_mov_b32 s2, 7
            enc_sop1(0x03, 2, 128 + 7),
            // s_pack_ll_b32_b16 s3, s2, -1  (src1 = 193 -> 4294967295u << 16u)
            enc_sop2(0x32, 3, 2, 193),
            // s_lshl3_add_u32 s4, -0.5, s2  (src0 = 241 -> 3204448256u << 3u)
            enc_sop2(0x30, 4, 241, 2),
            // s_lshl1_add_u32 s5, -1, s2    (src0 = 193 -> 4294967295u << 1u)
            enc_sop2(0x2E, 5, 193, 2),
            // s_add_u32 s6, -1, 16          (4294967295u + 16u const overflow -> folds to 15u)
            enc_sop2(0x00, 6, 193, 128 + 16),
            // v_mov_b32 v0, -1              (src0 = 193 -> bitcast<f32>(4294967295u))
            enc_vop1(0x01, 0, 193),
            // v_add_f32 v1, -8, v0          (src0 = 200 -> bitcast<f32>(4294967288u))
            enc_vop2(0x03, 1, 200, 0),
            // v_fmaak_f16 v2, v0, v1, 0x00007ef9 (32505u -> f16 NaN in unpack2x16float)
            enc_vop2(0x38, 2, 256, 1),
            0x0000_7EF9,
        ];
        // v_pk_add_f16 v3, -1, v2       (src0 = 193 -> unpack2x16float(4294967295u))
        words.extend_from_slice(&enc_vop3p(0x0F, 3, 193, 256 + 2, 0, false, 0, 0b011, 0, 0));
        // v_rcp_f32 v4, 0.0             (1.0 / 0.0 const-eval guard)
        words.push(enc_vop1(0x2A, 4, 128));
        // v_add_f32 v4, s3, v4
        words.push(enc_vop2(0x03, 4, 3, 4));
        // v_add_f32 v4, s4, v4
        words.push(enc_vop2(0x03, 4, 4, 4));
        // v_add_f32 v4, s5, v4
        words.push(enc_vop2(0x03, 4, 5, 4));
        // v_add_f32 v4, s6, v4
        words.push(enc_vop2(0x03, 4, 6, 4));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 4]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("shader with NaN bitcasts, f16 NaN unpacks, and shift overflows must validate");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // Must NEVER emit raw const-eval NaN bitcast<f32>, unpack2x16float, or overflowing shifts.
        assert!(
            !out.wgsl.contains("bitcast<f32>(4294967295u)"),
            "raw NaN bitcast<f32>(4294967295u) must not appear:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("bitcast<f32>(4294967288u)"),
            "raw NaN bitcast<f32>(4294967288u) must not appear:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("unpack2x16float(4294967295u)"),
            "raw NaN unpack2x16float(4294967295u) must not appear:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("unpack2x16float(32505u)"),
            "raw NaN unpack2x16float(32505u) must not appear:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("4294967295u << 16u"),
            "overflowing const shift 4294967295u << 16u must be folded:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("3204448256u << 3u"),
            "overflowing const shift 3204448256u << 3u must be folded:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("4294967295u << 1u"),
            "overflowing const shift 4294967295u << 1u must be folded:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("fn pg6_u32_bits_to_float("),
            "expected pg6_u32_bits_to_float runtime helper in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("fn pg6_unpack_half_pair("),
            "expected pg6_unpack_half_pair runtime helper in WGSL:\n{}",
            out.wgsl
        );
    }

    #[test]
    fn coalesces_descriptor_table_s_load_vsharp_into_s_buffer_load_cb0() {
        // Real PSO #50450 pattern:
        //   s_load_dwordx4        s[12:15], s[0:1], 0x0   ; fetch V# descriptor from root table
        //   s_waitcnt             lgkmcnt(0)
        //   s_buffer_load_dwordx4 s[8:11], s[12:15], 0x20 ; read actual CBV data[2] via V#
        //   s_waitcnt             lgkmcnt(0)
        //   v_mov_b32             v0, s8
        //   v_mov_b32             v1, s9
        //   v_mov_b32             v2, s10
        //   v_mov_b32             v3, s11
        //   exp mrt0              v0, v1, v2, v3 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x02, 12, 0, 0x7D, 0x0, false, false)); // s_load_dwordx4 s[12:15], s[0:1], 0x0
        words.push(enc_sopp(0x0C, 0xC07F)); // s_waitcnt lgkmcnt(0)
        words.extend_from_slice(&enc_smem(0x0A, 8, 6, 0x7D, 0x20, false, false)); // s_buffer_load_dwordx4 s[8:11], s[12:15], 0x20 (sbase=6 -> s12)
        words.push(enc_sopp(0x0C, 0xC07F));
        words.push(enc_vop1(0x01, 0, 8));
        words.push(enc_vop1(0x01, 1, 9));
        words.push(enc_vop1(0x01, 2, 10));
        words.push(enc_vop1(0x01, 3, 11));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("descriptor-table V# -> s_buffer_load PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(
            out.bindings,
            vec![WgslBindingInfo {
                group: 0,
                binding: 0,
                kind: "uniform",
                name: "cb0".to_string(),
                min_binding_size_bytes: Some(48),
                highest_vec4_offset: Some(2),
                view_dimension: None,
                source: None,
                storage_format: None,
            }]
        );
        assert_eq!(out.binding_summary.cbv_vec4_count, 3);
        assert!(
            out.wgsl.contains("@group(0) @binding(0)"),
            "expected @group(0) @binding(0) in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("@binding(3)"),
            "must NOT emit spurious @binding(3) for coalesced descriptor-table CBV:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 3>"),
            "expected tightly sized array<vec4<u32>, 3> (48 B) in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("cb0_.data[2]"),
            "expected direct cb0_.data[2] read in WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        // Also verify in-place overwrite: s_buffer_load_dwordx4 s[12:15], s[12:15], 0x20
        // followed by VALU consuming s12..s15 still coalesces to a single cb0 @binding(0)!
        let mut inplace_words = Vec::new();
        inplace_words.extend_from_slice(&enc_smem(0x02, 12, 0, 0x7D, 0x0, false, false));
        inplace_words.push(enc_sopp(0x0C, 0xC07F));
        inplace_words.extend_from_slice(&enc_smem(0x0A, 12, 6, 0x7D, 0x20, false, false));
        inplace_words.push(enc_sopp(0x0C, 0xC07F));
        inplace_words.push(enc_vop1(0x01, 0, 12));
        inplace_words.push(enc_vop1(0x01, 1, 13));
        inplace_words.push(enc_vop1(0x01, 2, 14));
        inplace_words.push(enc_vop1(0x01, 3, 15));
        inplace_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        inplace_words.push(enc_sopp(0x01, 0));

        let inplace_out = translate_rdna2_to_wgsl(
            &words_to_bytes(&inplace_words),
            &Rdna2TranslateOptions::default(),
        )
        .expect("in-place V# overwrite s_buffer_load PS should translate cleanly");
        assert!(inplace_out.naga_validated);
        assert_eq!(inplace_out.bindings.len(), 1);
        assert_eq!(inplace_out.bindings[0].binding, 0);
        assert_eq!(inplace_out.bindings[0].min_binding_size_bytes, Some(48));
        assert!(!inplace_out.wgsl.contains("@binding(3)"));
        assert!(inplace_out.wgsl.contains("cb0_.data[2]"));
        assert_naga_wgsl_valid(&inplace_out.wgsl);
    }

    #[test]
    fn is_safe_wgsl_f32_literal_predicate_bounds() {
        // Safe normal floats and +0.0 emit directly as WGSL f32 literals.
        assert!(is_safe_wgsl_f32_literal(0.0_f32));
        assert!(is_safe_wgsl_f32_literal(1.0_f32));
        assert!(is_safe_wgsl_f32_literal(-1.0_f32));
        assert!(is_safe_wgsl_f32_literal(0.5_f32));
        assert!(is_safe_wgsl_f32_literal(f32::MIN_POSITIVE));
        assert!(is_safe_wgsl_f32_literal(-f32::MIN_POSITIVE));
        assert!(is_safe_wgsl_f32_literal(3.39e38_f32));
        assert!(is_safe_wgsl_f32_literal(-3.39e38_f32));

        // Near-f32::MAX values (where Naga decimal printing can round up past f32::MAX)
        // must route through pg6_u32_bits_to_float(bits).
        assert!(!is_safe_wgsl_f32_literal(f32::MAX));
        assert!(!is_safe_wgsl_f32_literal(-f32::MAX));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x7F7F_FFFF)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0xFF7F_FFFF)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x7F7F_FFFE)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0xFF7F_FFFE)));
        assert!(!is_safe_wgsl_f32_literal(3.4e38_f32));
        assert!(!is_safe_wgsl_f32_literal(-3.4e38_f32));

        // -0.0, subnormals, infinities, and NaNs also route through pg6_u32_bits_to_float.
        assert!(!is_safe_wgsl_f32_literal(-0.0_f32));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x0000_0001)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x007F_FFFF)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x8000_0001)));
        assert!(!is_safe_wgsl_f32_literal(f32::INFINITY));
        assert!(!is_safe_wgsl_f32_literal(f32::NEG_INFINITY));
        assert!(!is_safe_wgsl_f32_literal(f32::NAN));
    }

    #[test]
    fn f32_max_and_neg_f32_max_literals_route_through_bitcast_helper() {
        // Reproduces Chrome/Dawn Tint parse rejection from Test #42:
        //   v_min3_f32 / v_max3_f32 / v_min_f32 / v_max_f32 / v_mov_b32 with
        //   0x7f7fffff (f32::MAX), 0xff7fffff (-f32::MAX), and 0x7f7ffffe.
        let mut words = Vec::new();
        // v_min3_f32 v0, 0x7f7fffff, v1, v2
        words.extend_from_slice(&enc_vop3(0x151, 0, 255, 256 + 1, 256 + 2, false, 0, 0, 0));
        words.push(0x7F7F_FFFF);
        // v_max3_f32 v1, 0xff7fffff, v0, v2
        words.extend_from_slice(&enc_vop3(0x154, 1, 255, 256, 256 + 2, false, 0, 0, 0));
        words.push(0xFF7F_FFFF);
        // v_min_f32 v2, 0x7f7fffff, v0
        words.push(enc_vop2(0x0F, 2, 255, 0));
        words.push(0x7F7F_FFFF);
        // v_max_f32 v3, 0xff7fffff, v1
        words.push(enc_vop2(0x10, 3, 255, 1));
        words.push(0xFF7F_FFFF);
        // v_mov_b32 v4, 0x7f7ffffe (near-max, also routed via bitcast helper)
        words.push(enc_vop1(0x01, 4, 255));
        words.push(0x7F7F_FFFE);
        // v_add_f32 v3, v3, v4
        words.push(enc_vop2(0x03, 3, 256 + 3, 4));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("shader with f32::MAX / -f32::MAX literals must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        assert!(
            !out.wgsl.contains("340282350000000000000000000000000000000f"),
            "Naga rounded-up f32::MAX decimal literal must NOT appear in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("340282330000000000000000000000000000000f"),
            "near-f32::MAX literal 0x7f7ffffe must NOT appear as raw decimal in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("fn pg6_u32_bits_to_float("),
            "expected pg6_u32_bits_to_float runtime helper in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("pg6_u32_bits_to_float(2139095039u)"),
            "expected exact 0x7f7fffff (2139095039u) bitcast call in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("pg6_u32_bits_to_float(4286578687u)"),
            "expected exact 0xff7fffff (4286578687u) bitcast call in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("pg6_u32_bits_to_float(2139095038u)"),
            "expected exact 0x7f7ffffe (2139095038u) bitcast call in WGSL:\n{}",
            out.wgsl
        );
    }

    /// Synthetic Xbox Series preamble (36 bytes): 6 arbitrary header dwords + the fixed tail
    /// `[0xd5800000, 0x03fe0100, 0x00000ac0]` (the last decodes as `v_cndmask_b32 v0, 64, v5, vcc`).
    fn xbox_preamble_words() -> Vec<u32> {
        vec![
            0x0000_0001,
            0x0000_0024,
            0x1234_5678,
            0x0000_0000,
            0xCAFE_F00D,
            0x0000_0100,
            0xd580_0000,
            0x03fe_0100,
            0x0000_0ac0,
        ]
    }

    /// PSO #12114-shaped UI vertex shader body (starting at the real entry point):
    ///   s_inst_prefetch 0x3
    ///   s_mov_b32 s9, 0
    ///   s_mov_b64 s[12:13], s[0:1]
    ///   s_bfe_u32 s13, s13, 0x00080000
    ///   s_load_dwordx4 s[8:11], s[8:9], 0x0      ; CBV V# table
    ///   s_load_dwordx4 s[12:15], s[12:13], 0x0   ; vertex-buffer V# table
    ///   v_add_nc_u32 v0, s2, v0
    ///   tbuffer_load_format_xyzw v[4:7], v0, s[12:15], 0 format:56 idxen          ; COLOR0
    ///   tbuffer_load_format_x    v1,     v0, s[12:15], 0 format:1  idxen offset:7 ; COLOR1.w
    ///   buffer_load_dwordx2      v[2:3], v0, s[12:15], 0 idxen offset:8           ; POSITION0
    ///   s_buffer_load_dwordx16 s[8:23], s[8:11], 0x0
    ///   v_mul_f32 v8, s8, v2 ; v_mul_f32 v9, s9, v3
    ///   exp pos0 v8, v9, v2, v3 done ; exp param0 v4, v5, v6, v1 ; s_endpgm
    fn pso_12114_vs_body_words() -> Vec<u32> {
        let mut words = vec![
            0xbfa0_0003,                  // s_inst_prefetch 0x3
            enc_sop1(0x03, 9, 128),       // s_mov_b32 s9, 0
            enc_sop1(0x04, 12, 0),        // s_mov_b64 s[12:13], s[0:1]
            enc_sop2(0x27, 13, 13, 0xFF), // s_bfe_u32 s13, s13, 0x00080000
            0x0008_0000,
        ];
        words.extend_from_slice(&enc_smem(0x02, 8, 4, 0x7D, 0, false, false));
        words.extend_from_slice(&enc_smem(0x02, 12, 6, 0x7D, 0, false, false));
        words.push(enc_vop2(0x25, 0, 2, 0)); // v_add_nc_u32 v0, s2, v0
        words.extend_from_slice(&enc_mtbuf(0x03, 56, 0, 4, 0, 3, true));
        words.extend_from_slice(&enc_mtbuf(0x00, 1, 7, 1, 0, 3, true));
        words.extend_from_slice(&enc_mubuf(0x0D, 8, 2, 0, 3, true));
        words.extend_from_slice(&enc_smem(0x0C, 8, 4, 0x7D, 0, false, false));
        words.push(enc_vop2(0x08, 8, 8, 2)); // v_mul_f32 v8, s8, v2
        words.push(enc_vop2(0x08, 9, 9, 3)); // v_mul_f32 v9, s9, v3
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [8, 9, 2, 3]));
        words.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [4, 5, 6, 1]));
        words.push(enc_sopp(0x01, 0));
        words
    }

    /// WGSL text without the leading `// Translated ...` / `// hash: ...` header lines.
    fn wgsl_without_hash_header(wgsl: &str) -> String {
        wgsl.lines()
            .filter(|l| !l.starts_with("// hash:"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn skips_xbox_preamble_preserves_full_blob_hash_and_never_clobbers_v0() {
        let body = pso_12114_vs_body_words();
        let mut words = xbox_preamble_words();
        words.extend_from_slice(&body);
        let full = words_to_bytes(&words);
        let bare = words_to_bytes(&body);
        assert_eq!(&full[0x24..], &bare[..]);

        let decoded = decode_program(&full);
        assert_eq!(decoded[0].offset, 0x24);
        assert_eq!(decoded[0].mnemonic, "s_inst_prefetch");
        assert!(decoded.iter().all(|i| i.mnemonic != "v_cndmask_b32"));

        let out = translate_rdna2_to_wgsl(&full, &Rdna2TranslateOptions::default())
            .expect("preamble-prefixed VS should translate");
        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        // Shader identity key: FNV-1a 64 over the FULL blob (matches d3d12_xs native_code_identity).
        assert_eq!(out.hash_fnv1a64, fnv1a_64(&full));
        assert_ne!(out.hash_fnv1a64, fnv1a_64(&bare));
        assert!(
            out.wgsl.contains(&format_shader_hash(fnv1a_64(&full))),
            "WGSL:\n{}",
            out.wgsl
        );
        // No bogus `v_cndmask_b32 v0, 64, v5, vcc` (v0 clobber / v5 pre-read -> legacy uv).
        assert!(!out.wgsl.contains("select("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("64f"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains(" uv:"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("@location(3)"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("unknown_enc"), "WGSL:\n{}", out.wgsl);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // The stripped body translates identically apart from the hash header line.
        let bare_out = translate_rdna2_to_wgsl(&bare, &Rdna2TranslateOptions::default())
            .expect("bare VS body should translate");
        assert_eq!(bare_out.hash_fnv1a64, fnv1a_64(&bare));
        assert_eq!(
            wgsl_without_hash_header(&out.wgsl),
            wgsl_without_hash_header(&bare_out.wgsl)
        );
        assert_eq!(out.bindings, bare_out.bindings);
        assert_eq!(out.translated_op_count, bare_out.translated_op_count);
    }

    #[test]
    fn pso_12114_vertex_buffer_vsharp_emits_no_bogus_cbv_and_three_vs_inputs() {
        let mut words = xbox_preamble_words();
        words.extend_from_slice(&pso_12114_vs_body_words());
        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("PSO #12114-shaped VS should translate");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.binding_summary.vertex_input_mode, "attributes");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);

        // Only the CBV descriptor-table slot `cb0` at @binding(0); no `cb1` for the VB V#.
        assert_eq!(out.bindings.len(), 1, "bindings: {:?}", out.bindings);
        assert_eq!(out.bindings[0].group, 0);
        assert_eq!(out.bindings[0].binding, 0);
        assert_eq!(out.bindings[0].kind, "uniform");
        assert_eq!(out.bindings[0].name, "cb0");
        assert_eq!(out.bindings[0].highest_vec4_offset, Some(3));
        assert!(
            out.wgsl.contains("var<uniform> cb0_"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(!out.wgsl.contains("cb1"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("@binding(3)"), "WGSL:\n{}", out.wgsl);

        // VsIn: exactly locations 0 (COLOR0 unorm8x4), 1 (COLOR1 unorm8x4, `.w` read by the
        // offset:7 8-bit fetch) and 2 (POSITION0 float2).
        assert!(out.wgsl.contains("struct VsIn"), "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("@location(0) position: vec4<f32>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("@location(1) attr_4_: vec4<f32>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("@location(2) attr_8_: vec2<f32>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(!out.wgsl.contains("@location(3)"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("attr_7"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("vs_in.attr_4_.w"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("vs_in.attr_4_.x"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn vertex_buffer_vsharp_also_read_by_s_buffer_load_still_binds_cbv() {
        // A V# consumed by both an idxen vertex fetch and `s_buffer_load` is a real CBV.
        let mut words = vec![];
        words.extend_from_slice(&enc_smem(0x02, 12, 0, 0x7D, 0, false, false)); // s_load_dwordx4 s[12:15], s[0:1]
        words.extend_from_slice(&enc_mubuf(0x0D, 0, 2, 0, 3, true)); // buffer_load_dwordx2 v[2:3] idxen
        words.extend_from_slice(&enc_smem(0x08, 16, 6, 0x7D, 0, false, false)); // s_buffer_load_dword s16, s[12:15]
        words.push(enc_vop2(0x08, 8, 16, 2)); // v_mul_f32 v8, s16, v2
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [8, 3, 2, 3]));
        words.push(enc_sopp(0x01, 0));
        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("VS should translate");
        assert_eq!(out.bindings.len(), 1, "bindings: {:?}", out.bindings);
        assert_eq!(out.bindings[0].name, "cb0");
        assert_naga_wgsl_valid(&out.wgsl);
    }

    // -----------------------------------------------------------------------
    // S28: GFX10 VOP2 slots, 64-bit SOP2/SOPC, mul_hi, 64-bit shifts, class, s_barrier
    // -----------------------------------------------------------------------

    fn enc_sopc(op: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b1011_1111_0u32 << 23)
            | (u32::from(op & 0x7F) << 16)
            | (u32::from(ssrc1) << 8)
            | u32::from(ssrc0)
    }

    /// VOP3B encoding (`sdst` in bits [14:8]) for carry-out ops such as `v_add_co_ci_u32_e64`.
    fn enc_vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        let w0 = (0b110101u32 << 26)
            | (u32::from(op & 0x3FF) << 16)
            | (u32::from(sdst & 0x7F) << 8)
            | u32::from(vdst);
        let w1 = (u32::from(src2 & 0x1FF) << 18)
            | (u32::from(src1 & 0x1FF) << 9)
            | u32::from(src0 & 0x1FF);
        [w0, w1]
    }

    fn translate_as(words: &[u32], stage: Rdna2StageOverride) -> Rdna2WgslOutput {
        let bytes = words_to_bytes(words);
        let options = Rdna2TranslateOptions {
            stage,
            ..Rdna2TranslateOptions::default()
        };
        translate_rdna2_to_wgsl(&bytes, &options)
            .unwrap_or_else(|e| panic!("translation must succeed: {e}"))
    }

    /// Right-hand side of the last `name = <rhs>;` assignment in the WGSL text.
    fn last_assignment(wgsl: &str, name: &str) -> Option<String> {
        let prefix = format!("{name} = ");
        wgsl.lines()
            .map(str::trim)
            .rfind(|line| line.starts_with(&prefix) && line.ends_with(';'))
            .map(|line| line[prefix.len()..line.len() - 1].to_string())
    }

    /// Reference multiply-high values computed with native 64-bit integers.
    fn ref_mul_hi_u32(a: u32, b: u32) -> u32 {
        ((u64::from(a) * u64::from(b)) >> 32) as u32
    }

    fn ref_mul_hi_i32(a: u32, b: u32) -> u32 {
        ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32
    }

    /// Reference `v_cmp_class_f32` class index computed with Rust's `f32::classify`.
    fn ref_f32_class(bits: u32) -> u32 {
        let f = f32::from_bits(bits);
        let neg = (bits >> 31) != 0;
        match f.classify() {
            std::num::FpCategory::Nan => u32::from((bits & 0x0040_0000) != 0),
            std::num::FpCategory::Infinite => {
                if neg {
                    2
                } else {
                    9
                }
            }
            std::num::FpCategory::Normal => {
                if neg {
                    3
                } else {
                    8
                }
            }
            std::num::FpCategory::Subnormal => {
                if neg {
                    4
                } else {
                    7
                }
            }
            std::num::FpCategory::Zero => {
                if neg {
                    5
                } else {
                    6
                }
            }
        }
    }

    #[test]
    fn translates_gfx10_vop2_alu_fmac_madk_carry_and_pkrtz_ops_with_zero_unsupported() {
        let mut words = Vec::new();
        words.push(enc_vop1(0x01, 1, 242)); // v_mov_b32 v1, 1.0
        words.push(enc_vop1(0x01, 2, 244)); // v_mov_b32 v2, 2.0
        words.push(enc_vop2(0x2B, 3, 256 + 1, 2)); // v_fmac_f32 v3, v1, v2
        words.push(enc_vop2(0x2C, 4, 256 + 1, 2)); // v_fmamk_f32 v4, v1, K, v2
        words.push(0x4040_0000); //   K = 3.0
        words.push(enc_vop2(0x2D, 5, 256 + 1, 2)); // v_fmaak_f32 v5, v1, v2, K
        words.push(0x3F00_0000); //   K = 0.5
        words.push(enc_vop2(0x1F, 3, 256 + 4, 5)); // v_mac_f32 v3, v4, v5
        words.push(enc_vop2(0x20, 6, 256 + 1, 2)); // v_madmk_f32 v6, v1, K, v2
        words.push(0x4000_0000); //   K = 2.0
        words.push(enc_vop2(0x21, 7, 128 + 3, 2)); // v_madak_f32 v7, 3, v2, K (inline src0; K still present)
        words.push(0x4000_0000); //   K = 2.0
        words.push(enc_vop1(0x01, 8, 128 + 7)); // v_mov_b32 v8, 7
        words.push(enc_vop1(0x01, 9, 0xFF)); // v_mov_b32 v9, 0x89abcdef
        words.push(0x89AB_CDEF);
        words.push(enc_vop2(0x25, 10, 256 + 8, 9)); // v_add_nc_u32 v10, v8, v9
        words.push(enc_vop2(0x26, 11, 256 + 10, 8)); // v_sub_nc_u32 v11, v10, v8
        words.push(enc_vop2(0x27, 12, 256 + 8, 11)); // v_subrev_nc_u32 v12, v8, v11
        words.push(enc_vop2(0x1E, 13, 256 + 11, 12)); // v_xnor_b32 v13, v11, v12
        words.push(enc_vop2(0x1B, 14, 128 + 63, 13)); // v_and_b32 v14, 63, v13
        words.push(enc_vop2(0x1C, 14, 256 + 14, 8)); // v_or_b32 v14, v14, v8
        words.push(enc_vop2(0x1D, 14, 256 + 14, 9)); // v_xor_b32 v14, v14, v9
        words.push(enc_vop2(0x1A, 15, 128 + 2, 14)); // v_lshlrev_b32 v15, 2, v14
        words.push(enc_vop2(0x16, 16, 128 + 1, 15)); // v_lshrrev_b32 v16, 1, v15
        words.push(enc_vop2(0x18, 17, 128 + 1, 15)); // v_ashrrev_i32 v17, 1, v15
        words.push(enc_vop2(0x0C, 18, 256 + 9, 15)); // v_mul_hi_u32_u24 v18, v9, v15
        words.push(enc_vop2(0x0A, 19, 256 + 9, 17)); // v_mul_hi_i32_i24 v19, v9, v17
        words.extend_from_slice(&enc_vop3b(0x30F, 20, 106, 256 + 9, 256 + 9, 0)); // v_add_co_u32 v20, vcc, v9, v9
        words.push(enc_vop2(0x28, 21, 256 + 9, 9)); // v_add_co_ci_u32 v21, vcc, v9, v9, vcc
        words.push(enc_vop2(0x29, 22, 256 + 8, 21)); // v_sub_co_ci_u32 v22, vcc, v8, v21, vcc
        words.push(enc_vop2(0x2A, 23, 256 + 8, 22)); // v_subrev_co_ci_u32 v23, vcc, v8, v22, vcc
        words.extend_from_slice(&enc_vop3b(0x128, 24, 4, 256 + 22, 256 + 23, 106)); // v_add_co_ci_u32_e64 v24, s[4:5], v22, v23, vcc
        words.extend_from_slice(&enc_vop3b(0x129, 25, 106, 256 + 24, 256 + 23, 4)); // v_sub_co_ci_u32_e64 v25, vcc, v24, v23, s[4:5]
        words.push(enc_vop2(0x2F, 26, 256 + 3, 4)); // v_cvt_pkrtz_f16_f32 v26, v3, v4
        words.extend_from_slice(&enc_vop3(0x12F, 27, 256 + 6, 256 + 7, 0, true, 0, 0, 0)); // VOP3 pkrtz clamp
        words.push(enc_vop1(0x06, 28, 256 + 25)); // v_cvt_f32_u32 v28, v25
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [3, 5, 26, 28]));
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let out = translate_as(&words, Rdna2StageOverride::Auto);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("fma("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);
        // Carry chains flow through the VCC lane bit and the SGPR lane mask s[4:5].
        assert!(out.wgsl.contains("vcc"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("s_4_"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("s_5_"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn vop2_only_madk_forms_are_rejected_in_vop3_encoding() {
        let mut words = Vec::new();
        words.extend_from_slice(&enc_vop3(0x12C, 1, 256, 256, 0, false, 0, 0, 0)); // invalid v_fmamk_f32_e64
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [1, 1, 1, 1]));
        words.push(enc_sopp(0x01, 0));
        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 1, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("v_fmamk_f32"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_64bit_sop2_bitwise_cselect_shift_and_sopc_ops_on_runtime_sgpr_pairs() {
        let mut words = Vec::new();
        // s[0:3] are runtime values (read before written).
        words.push(enc_sop2(0x0F, 8, 0, 2)); // s_and_b64 s[8:9], s[0:1], s[2:3]
        words.push(enc_sop2(0x11, 10, 0, 2)); // s_or_b64 s[10:11], s[0:1], s[2:3]
        words.push(enc_sop2(0x13, 12, 0, 2)); // s_xor_b64 s[12:13], s[0:1], s[2:3]
        words.push(enc_sop2(0x15, 14, 0, 2)); // s_andn2_b64 s[14:15], s[0:1], s[2:3]
        words.push(enc_sop2(0x17, 16, 0, 8)); // s_orn2_b64 s[16:17], s[0:1], s[8:9]
        words.push(enc_sop2(0x19, 18, 0, 8)); // s_nand_b64 s[18:19], s[0:1], s[8:9]
        words.push(enc_sop2(0x1B, 20, 0, 8)); // s_nor_b64 s[20:21], s[0:1], s[8:9]
        words.push(enc_sop2(0x1D, 22, 0, 8)); // s_xnor_b64 s[22:23], s[0:1], s[8:9]
        words.push(enc_sopc(0x13, 8, 10)); // s_cmp_lg_u64 s[8:9], s[10:11]
        words.push(enc_sop2(0x0B, 24, 0, 2)); // s_cselect_b64 s[24:25], s[0:1], s[2:3]
        words.push(enc_sopc(0x0D, 0, 128 + 3)); // s_bitcmp1_b32 s0, 3
        words.push(enc_sop2(0x0A, 26, 0, 1)); // s_cselect_b32 s26, s0, s1
        words.push(enc_sopc(0x0C, 0, 2)); // s_bitcmp0_b32 s0, s2
        words.push(enc_sopc(0x0E, 0, 2)); // s_bitcmp0_b64 s[0:1], s2
        words.push(enc_sopc(0x0F, 0, 128 + 1)); // s_bitcmp1_b64 s[0:1], 1
        words.push(enc_sopc(0x12, 0, 2)); // s_cmp_eq_u64 s[0:1], s[2:3]
        words.push(enc_sop2(0x1F, 28, 0, 2)); // s_lshl_b64 s[28:29], s[0:1], s2
        words.push(enc_sop2(0x21, 30, 0, 2)); // s_lshr_b64 s[30:31], s[0:1], s2
        words.push(enc_sop2(0x23, 32, 0, 2)); // s_ashr_i64 s[32:33], s[0:1], s2
        words.push(enc_sop2(0x35, 34, 0, 1)); // s_mul_hi_u32 s34, s0, s1
        words.push(enc_sop2(0x36, 35, 0, 1)); // s_mul_hi_i32 s35, s0, s1
        words.push(enc_sop2(0x15, 106, 126, 8)); // s_andn2_b64 vcc, exec, s[8:9]
        words.push(enc_sop2(0x13, 106, 126, 106)); // s_xor_b64 vcc, exec, vcc
        words.push(enc_sop2(0x0F, 126, 126, 106)); // s_and_b64 exec, exec, vcc (exec write is a no-op)
        words.push(enc_vop2(0x01, 1, 128, 0)); // v_cndmask_b32 v1, 0, v0, vcc
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        for reg in [
            8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35,
        ] {
            assert!(
                last_assignment(&out.wgsl, &format!("s_{reg}_")).is_some(),
                "missing store to s{reg}:\n{}",
                out.wgsl
            );
        }
        assert!(
            last_assignment(&out.wgsl, "scc").is_some(),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("select("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("i64") && !out.wgsl.contains("u64"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn scalar_mul_hi_shift64_bitwise64_and_sopc_constant_fold_to_exact_reference_values() {
        // With literal / inline-constant operands every emitted u32 op folds in Rust, so the
        // stored constants prove the half-word mul-hi, pair-shift, and 64-bit bitwise trees.
        let a = 0xDEAD_BEEFu32;
        let b = 0x9E37_79B9u32;
        // (program body, register name, expected RHS)
        let cases: Vec<(Vec<u32>, &str, String)> = vec![
            (
                vec![enc_sop2(0x35, 4, 0xFF, 0xFF), a],
                "s_4_",
                format!("{}u", ref_mul_hi_u32(a, a)),
            ),
            (
                vec![enc_sop2(0x35, 4, 0xFF, 0xFF), b],
                "s_4_",
                format!("{}u", ref_mul_hi_u32(b, b)),
            ),
            (
                vec![enc_sop2(0x35, 4, 0xFF, 128 + 64), b],
                "s_4_",
                format!("{}u", ref_mul_hi_u32(b, 64)),
            ),
            (
                vec![enc_sop2(0x36, 4, 0xFF, 0xFF), a],
                "s_4_",
                format!("{}u", ref_mul_hi_i32(a, a)),
            ),
            (
                vec![enc_sop2(0x36, 4, 0xFF, 192 + 7), b],
                "s_4_",
                format!("{}u", ref_mul_hi_i32(b, (-7i32) as u32)),
            ),
            (
                vec![enc_sop2(0x36, 4, 0xFF, 128 + 5), 0x7FFF_FFFF],
                "s_4_",
                format!("{}u", ref_mul_hi_i32(0x7FFF_FFFF, 5)),
            ),
            // s_lshr_b64 s[4:5], -1, 36 -> (0x0fffffff, 0)
            (
                vec![enc_sop2(0x21, 4, 193, 128 + 36)],
                "s_4_",
                format!("{}u", 0x0FFF_FFFFu32),
            ),
            (
                vec![enc_sop2(0x21, 4, 193, 128 + 36)],
                "s_5_",
                "0u".to_string(),
            ),
            // s_lshr_b64 s[4:5], -1, 4 -> (0xffffffff, 0x0fffffff)
            (
                vec![enc_sop2(0x21, 4, 193, 128 + 4)],
                "s_4_",
                format!("{}u", u32::MAX),
            ),
            (
                vec![enc_sop2(0x21, 4, 193, 128 + 4)],
                "s_5_",
                format!("{}u", 0x0FFF_FFFFu32),
            ),
            // s_lshl_b64 s[4:5], 0x80000001 (zero-extended), 31 -> (0x80000000, 0x40000000)
            (
                vec![enc_sop2(0x1F, 4, 0xFF, 128 + 31), 0x8000_0001],
                "s_4_",
                format!("{}u", 0x8000_0000u32),
            ),
            (
                vec![enc_sop2(0x1F, 4, 0xFF, 128 + 31), 0x8000_0001],
                "s_5_",
                format!("{}u", 0x4000_0000u32),
            ),
            // s_lshl_b64 s[4:5], 0x12345678, 44 -> (0, 0x45678000)
            (
                vec![enc_sop2(0x1F, 4, 0xFF, 128 + 44), 0x1234_5678],
                "s_4_",
                "0u".to_string(),
            ),
            (
                vec![enc_sop2(0x1F, 4, 0xFF, 128 + 44), 0x1234_5678],
                "s_5_",
                format!("{}u", 0x4567_8000u32),
            ),
            // shift amount is masked to 6 bits: s_lshr_b64 s[4:5], 0x12345678, 64 -> unchanged
            (
                vec![enc_sop2(0x21, 4, 0xFF, 128 + 64), 0x1234_5678],
                "s_4_",
                format!("{}u", 0x1234_5678u32),
            ),
            // s_ashr_i64 s[4:5], -16, 2 -> (0xfffffffc, 0xffffffff)
            (
                vec![enc_sop2(0x23, 4, 208, 128 + 2)],
                "s_4_",
                format!("{}u", 0xFFFF_FFFCu32),
            ),
            (
                vec![enc_sop2(0x23, 4, 208, 128 + 2)],
                "s_5_",
                format!("{}u", u32::MAX),
            ),
            // s_ashr_i64 s[4:5], -16, 40 -> (0xffffffff, 0xffffffff)
            (
                vec![enc_sop2(0x23, 4, 208, 128 + 40)],
                "s_4_",
                format!("{}u", u32::MAX),
            ),
            // 64-bit bitwise: s_andn2_b64 s[4:5], 0xf0f0f0f1, -2 -> (0xf0f0f0f1 & ~0xfffffffe, 0 & ~0xffffffff)
            (
                vec![enc_sop2(0x15, 4, 0xFF, 194), 0xF0F0_F0F1],
                "s_4_",
                "1u".to_string(),
            ),
            (
                vec![enc_sop2(0x1B, 4, 128, 128)],
                "s_5_",
                format!("{}u", u32::MAX),
            ), // s_nor_b64 0, 0
            (
                vec![enc_sop2(0x1D, 4, 0xFF, 193), 0xF0F0_F0F0],
                "s_4_",
                format!("{}u", 0xF0F0_F0F0u32),
            ), // s_xnor_b64 x, -1
            (
                vec![enc_sop2(0x1D, 4, 0xFF, 193), 0xF0F0_F0F0],
                "s_5_",
                "0u".to_string(),
            ),
            // scc = ((lo | hi) != 0) for 64-bit bitwise ops (read back by s_cselect_b32)
            (
                vec![enc_sop2(0x13, 4, 193, 193), enc_sop2(0x0A, 6, 128, 128)],
                "scc",
                "false".to_string(),
            ),
            (
                vec![
                    enc_sop2(0x11, 4, 128, 0xFF),
                    0x0000_0100,
                    enc_sop2(0x0A, 6, 128, 128),
                ],
                "scc",
                "true".to_string(),
            ),
            // SOPC bit tests and 64-bit compares
            (
                vec![enc_sopc(0x0D, 0xFF, 128 + 3), 0x8],
                "scc",
                "true".to_string(),
            ),
            (
                vec![enc_sopc(0x0D, 0xFF, 128 + 35), 0x8],
                "scc",
                "true".to_string(),
            ), // 35 & 31 == 3
            (
                vec![enc_sopc(0x0C, 0xFF, 128 + 3), 0x8],
                "scc",
                "false".to_string(),
            ),
            (
                vec![enc_sopc(0x0C, 0xFF, 128 + 2), 0x8],
                "scc",
                "true".to_string(),
            ),
            (
                vec![enc_sopc(0x0F, 193, 128 + 63)],
                "scc",
                "true".to_string(),
            ), // bit 63 of -1
            (
                vec![enc_sopc(0x0F, 0xFF, 128 + 33), 0x2],
                "scc",
                "false".to_string(),
            ), // hi = 0
            (
                vec![enc_sopc(0x0E, 0xFF, 128 + 1), 0x2],
                "scc",
                "false".to_string(),
            ),
            (vec![enc_sopc(0x12, 193, 193)], "scc", "true".to_string()),
            (
                vec![enc_sopc(0x12, 193, 0xFF), u32::MAX],
                "scc",
                "false".to_string(),
            ), // hi differs
            (
                vec![enc_sopc(0x13, 128 + 1, 128 + 1)],
                "scc",
                "false".to_string(),
            ),
            (
                vec![enc_sopc(0x13, 128 + 1, 128 + 2)],
                "scc",
                "true".to_string(),
            ),
            // VCC destinations receive this lane's bit (bit 0 of lo)
            (
                vec![enc_sop2(0x13, 106, 126, 128 + 1)],
                "vcc",
                "false".to_string(),
            ), // s_xor_b64 vcc, exec, 1
            (
                vec![enc_sop2(0x13, 106, 126, 128 + 2)],
                "vcc",
                "true".to_string(),
            ), // s_xor_b64 vcc, exec, 2
        ];
        for (body, reg, expect) in cases {
            let mut words = body.clone();
            words.push(enc_sopp(0x01, 0));
            let out = translate_as(&words, Rdna2StageOverride::Compute);
            assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
            assert_eq!(
                last_assignment(&out.wgsl, reg).as_deref(),
                Some(expect.as_str()),
                "program {body:08x?} -> {reg}:\n{}",
                out.wgsl
            );
        }
    }

    #[test]
    fn vector_mul_hi_and_shift64_ops_translate_with_runtime_operands_and_validate() {
        let mut words = Vec::new();
        words.extend_from_slice(&enc_vop3(0x16A, 2, 256, 6, 0, false, 0, 0, 0)); // v_mul_hi_u32 v2, v0, s6
        words.extend_from_slice(&enc_vop3(0x16C, 3, 256, 256 + 2, 0, false, 0, 0, 0)); // v_mul_hi_i32 v3, v0, v2
        words.extend_from_slice(&enc_vop3(0x16A, 10, 0xFF, 256 + 3, 0, false, 0, 0, 0)); // v_mul_hi_u32 v10, lit, v3
        words.push(0x0001_0001);
        words.extend_from_slice(&enc_vop3(0x300, 4, 256, 256 + 2, 0, false, 0, 0, 0)); // v_lshrrev_b64 v[4:5], v0, v[2:3]
        words.extend_from_slice(&enc_vop3(0x2FF, 6, 6, 0, 0, false, 0, 0, 0)); // v_lshlrev_b64 v[6:7], s6, s[0:1]
        words.extend_from_slice(&enc_vop3(0x301, 8, 7, 256 + 4, 0, false, 0, 0, 0)); // v_ashrrev_i64 v[8:9], s7, v[4:5]
        words.extend_from_slice(&enc_vop3(0x300, 11, 128 + 40, 193, 0, false, 0, 0, 0)); // v_lshrrev_b64 v[11:12], 40, -1
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        // 16-bit half-word products and 6-bit masked shift amounts; no 64-bit integer types.
        assert!(out.wgsl.contains("65535u"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("63u"), "WGSL:\n{}", out.wgsl);
        assert!(
            !out.wgsl.contains("u64") && !out.wgsl.contains("i64"),
            "WGSL:\n{}",
            out.wgsl
        );
        for reg in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
            assert!(
                last_assignment(&out.wgsl, &format!("v_{reg}_")).is_some(),
                "missing store to v{reg}:\n{}",
                out.wgsl
            );
        }
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn v_cmp_class_f32_constant_folds_to_reference_ieee754_classes() {
        let values: [u32; 14] = [
            0x7F80_0001, // signaling NaN
            0x7FC0_0000, // quiet NaN
            0xFF80_0000, // -Inf
            0xBF80_0000, // -1.0 (normal)
            0x8000_0001, // -subnormal
            0x8000_0000, // -0
            0x0000_0000, // +0
            0x0000_0001, // +subnormal
            0x3F80_0000, // +1.0 (normal)
            0x7F80_0000, // +Inf
            0x0000_0080, // +subnormal with mantissa bit 7 set
            0x0000_0100, // +subnormal with mantissa bit 8 set (bit 7 clear)
            0x3F80_0100, // +normal with bit 8 set
            0x3F80_0080, // +normal with bit 8 clear
        ];
        // (src1 operand encoding, mask value); `None` reuses the shared VOP3 literal (mask == bits).
        let masks: [(Option<u16>, u32); 11] = [
            (Some(128), 0),
            (Some(128 + 1), 1),
            (Some(128 + 2), 2),
            (Some(128 + 4), 4),
            (Some(128 + 8), 8),
            (Some(128 + 16), 16),
            (Some(128 + 32), 32),
            (Some(128 + 63), 63),
            (Some(128 + 64), 64),
            (Some(193), u32::MAX),
            (None, 0),
        ];
        for bits in values {
            let class = ref_f32_class(bits);
            for (mask_src, mask_val) in masks {
                let (src1, mask) = match mask_src {
                    Some(src) => (src, mask_val),
                    None => (0xFF, bits),
                };
                let mut words = Vec::new();
                // v_cmp_class_f32_e64 vcc, lit(bits), mask
                words.extend_from_slice(&enc_vop3(0x088, 106, 0xFF, src1, 0, false, 0, 0, 0));
                words.push(bits);
                words.push(enc_sopp(0x01, 0));
                let out = translate_as(&words, Rdna2StageOverride::Compute);
                assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
                let expect = ((mask >> class) & 1) != 0;
                assert_eq!(
                    last_assignment(&out.wgsl, "vcc").as_deref(),
                    Some(if expect { "true" } else { "false" }),
                    "bits {bits:#010x} (class {class}) mask {mask:#x}:\n{}",
                    out.wgsl
                );
            }
        }

        // |x| and -x source modifiers act on the raw sign bit: class(-|-1.0|) == -normal.
        let mut words = Vec::new();
        words.extend_from_slice(&enc_vop3(
            0x088,
            106,
            0xFF,
            128 + 8,
            0,
            false,
            0,
            0b01,
            0b01,
        ));
        words.push(0xBF80_0000);
        words.push(enc_sopp(0x01, 0));
        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(
            last_assignment(&out.wgsl, "vcc").as_deref(),
            Some("true"),
            "WGSL:\n{}",
            out.wgsl
        );
    }

    #[test]
    fn v_cmp_class_f32_runtime_vopc_and_vop3_sgpr_forms_validate() {
        let mut words = Vec::new();
        words.push(enc_vopc(0x88, 256, 1)); // v_cmp_class_f32 vcc, v0, v1
        words.push(enc_vop2(0x01, 2, 128, 0)); // v_cndmask_b32 v2, 0, v0, vcc
        words.extend_from_slice(&enc_vop3(0x088, 10, 256 + 2, 0xFF, 0, false, 0, 0b01, 0)); // v_cmp_class_f32 s[10:11], |v2|, 0x3ff
        words.push(0x3FF);
        words.push(enc_sop2(0x0F, 106, 10, 106)); // s_and_b64 vcc, s[10:11], vcc
        words.push(enc_vop2(0x01, 3, 128, 2)); // v_cndmask_b32 v3, 0, v2, vcc
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [2, 3, 2, 3]));
        words.push(enc_sopp(0x01, 0));
        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        // Lane-mask SGPR destinations are canonical (lo = lane bit, hi = 0).
        assert_eq!(
            last_assignment(&out.wgsl, "s_11_").as_deref(),
            Some("0u"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s_barrier_emits_workgroup_barrier_only_in_compute_and_a_note_elsewhere() {
        let body = |words: &mut Vec<u32>| {
            words.push(enc_vop1(0x01, 1, 242)); // v_mov_b32 v1, 1.0
            words.push(enc_sopp(0x0A, 0)); // s_barrier
        };

        // Compute: real workgroup barrier.
        let mut cs = Vec::new();
        body(&mut cs);
        cs.push(enc_sopp(0x01, 0));
        let out = translate_as(&cs, Rdna2StageOverride::Compute);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("workgroupBarrier()"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(!out.wgsl.contains("// note"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // Vertex: barrier must not be emitted (WGSL forbids it outside compute).
        let mut vs = Vec::new();
        body(&mut vs);
        vs.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [1, 1, 1, 1])); // exp pos0
        vs.push(enc_sopp(0x01, 0));
        let out = translate_as(&vs, Rdna2StageOverride::Vertex);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            !out.wgsl.contains("workgroupBarrier()"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl
                .contains("// note [0x0004] s_barrier: no-op in vertex stage (workgroupBarrier is compute-only)"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        // Fragment: same benign no-op.
        let mut ps = Vec::new();
        body(&mut ps);
        ps.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [1, 1, 1, 1])); // exp mrt0
        ps.push(enc_sopp(0x01, 0));
        let out = translate_as(&ps, Rdna2StageOverride::Pixel);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            !out.wgsl.contains("workgroupBarrier()"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("s_barrier: no-op in "),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    // -- S30: structured control flow & exec_mask (rdna2_cfg wiring) --

    /// SOPP `simm16` for a branch at byte `from` to byte `to` (`to = from + 4 + simm * 4`).
    fn branch_simm(from: usize, to: usize) -> u16 {
        (((to as i64 - from as i64 - 4) / 4) as i16) as u16
    }

    fn structured_tree(words: &[u32]) -> StructuredNode {
        let insts = decode_program(&words_to_bytes(words));
        structure_cfg(&ControlFlowGraph::from_instructions(&insts))
    }

    fn tree_has_else(node: &StructuredNode) -> bool {
        match node {
            StructuredNode::If {
                then_branch,
                else_branch,
                ..
            } => else_branch.is_some() || tree_has_else(then_branch),
            StructuredNode::Seq(nodes) => nodes.iter().any(tree_has_else),
            StructuredNode::Loop { body } => tree_has_else(body),
            _ => false,
        }
    }

    fn tree_has_loop(node: &StructuredNode) -> bool {
        match node {
            StructuredNode::Loop { .. } => true,
            StructuredNode::If {
                then_branch,
                else_branch,
                ..
            } => tree_has_loop(then_branch) || else_branch.as_deref().is_some_and(tree_has_loop),
            StructuredNode::Seq(nodes) => nodes.iter().any(tree_has_loop),
            _ => false,
        }
    }

    #[test]
    fn s_cbranch_scc0_if_skip_and_s_branch_diamond_lift_to_structured_wgsl() {
        // If-skip (compute):
        //   s_mov_b32 s0, 5 ; s_cmp_eq_u32 s0, 3 ; s_cbranch_scc0 L
        //   v_mov_b32 v1, 1.0
        // L: v_mov_b32 v2, 2.0 ; s_endpgm
        let words = vec![
            enc_sop1(0x03, 0, 133),
            enc_sopc(0x06, 0, 131),
            enc_sopp(0x04, branch_simm(8, 16)),
            enc_vop1(0x01, 1, 242),
            enc_vop1(0x01, 2, 244),
            enc_sopp(0x01, 0),
        ];
        let tree = structured_tree(&words);
        assert!(tree.is_fully_structured(), "{tree:?}");
        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.translated_op_count, 6, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("if "), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("switch"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("exec_mask"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // If/else diamond with an `s_branch` merge (pixel):
        //   s_mov_b32 s0, 5 ; s_cmp_eq_u32 s0, 3 ; s_cbranch_scc1 ELSE
        //   v_mov_b32 v1, 1.0 ; s_branch MERGE
        // ELSE: v_mov_b32 v1, 2.0
        // MERGE: v_add_f32 v2, v1, v1 ; exp mrt0 v2 x4 ; s_endpgm
        let mut words = vec![
            enc_sop1(0x03, 0, 133),
            enc_sopc(0x06, 0, 131),
            enc_sopp(0x05, branch_simm(8, 20)),
            enc_vop1(0x01, 1, 242),
            enc_sopp(0x02, branch_simm(16, 24)),
            enc_vop1(0x01, 1, 244),
            enc_vop2(0x03, 2, 257, 1),
        ];
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [2, 2, 2, 2]));
        words.push(enc_sopp(0x01, 0));
        let tree = structured_tree(&words);
        assert!(tree.is_fully_structured(), "{tree:?}");
        assert!(tree_has_else(&tree), "{tree:?}");
        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.translated_op_count, 9, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("} else {"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("= ps_out;"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("return;"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s_and_saveexec_b32_execz_if_else_diamond_models_single_lane_exec_mask() {
        // LLVM-style divergent if/else:
        //   v_mov_b32 v0, 1.0 ; v_cmp_lt_f32 vcc, 0.5, v0
        //   s_and_saveexec_b32 s2, vcc_lo ; s_xor_b32 s2, exec_lo, s2
        //   s_cbranch_execz ELSE
        //   v_mov_b32 v1, 2.0
        // ELSE: s_or_saveexec_b32 s2, s2 ; s_xor_b32 exec_lo, exec_lo, s2
        //   s_cbranch_execz END
        //   v_mov_b32 v1, 4.0
        // END: s_or_b32 exec_lo, exec_lo, s2 ; exp mrt0 v1 x4 ; s_endpgm
        let mut words = vec![
            enc_vop1(0x01, 0, 242),
            enc_vopc(0x01, 240, 0),
            enc_sop1(0x3C, 2, 106),
            enc_sop2(0x12, 2, 126, 2),
            enc_sopp(0x08, branch_simm(16, 24)),
            enc_vop1(0x01, 1, 244),
            enc_sop1(0x3D, 2, 2),
            enc_sop2(0x12, 126, 126, 2),
            enc_sopp(0x08, branch_simm(32, 40)),
            enc_vop1(0x01, 1, 246),
            enc_sop2(0x10, 126, 126, 2),
        ];
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [1, 1, 1, 1]));
        words.push(enc_sopp(0x01, 0));

        let tree = structured_tree(&words);
        assert!(tree.is_fully_structured(), "{tree:?}");
        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.translated_op_count, 13, "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("var exec_mask: u32;"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("exec_mask = 1u;"), "WGSL:\n{}", out.wgsl);
        // Both `s_cbranch_execz` skips become `if (exec_mask != 0u) { ... }`.
        assert_eq!(
            out.wgsl.matches("!= 0u) {").count(),
            2,
            "WGSL:\n{}",
            out.wgsl
        );
        // `s_or_b32 exec_lo, exec_lo, s2` keeps only this lane's bit.
        assert!(
            last_assignment(&out.wgsl, "exec_mask").is_some_and(|rhs| rhs.contains("& 1u")),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn backward_s_cbranch_scc1_loop_reloads_accumulators_every_iteration() {
        //   v_mov_b32 v1, 0 ; s_mov_b32 s0, 0
        // LOOP: v_add_f32 v1, 1.0, v1 ; s_add_u32 s0, s0, 1
        //   s_cmp_lt_u32 s0, 4 ; s_cbranch_scc1 LOOP
        //   exp mrt0 v1 x4 ; s_endpgm
        let mut words = vec![
            enc_vop1(0x01, 1, 128),
            enc_sop1(0x03, 0, 128),
            enc_vop2(0x03, 1, 242, 1),
            enc_sop2(0x00, 0, 0, 129),
            enc_sopc(0x0A, 0, 132),
            enc_sopp(0x05, branch_simm(20, 8)),
        ];
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [1, 1, 1, 1]));
        words.push(enc_sopp(0x01, 0));

        let tree = structured_tree(&words);
        assert!(tree.is_fully_structured(), "{tree:?}");
        assert!(tree_has_loop(&tree), "{tree:?}");
        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.translated_op_count, 8, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("loop {"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("_pg6_loop"), "WGSL:\n{}", out.wgsl);
        // Constant-folding must never carry the pre-loop `0` into the loop body: the
        // accumulators are re-loaded from their `var`s on every iteration.
        assert!(!out.wgsl.contains("v_1_ = 1f;"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("s_0_ = 1u;"), "WGSL:\n{}", out.wgsl);
        let loop_body = &out.wgsl[out.wgsl.find("loop {").unwrap()..];
        assert!(loop_body.contains("= v_1_;"), "WGSL:\n{}", out.wgsl);
        assert!(loop_body.contains("= s_0_;"), "WGSL:\n{}", out.wgsl);
        assert!(
            last_assignment(&out.wgsl, "v_1_").is_some_and(|rhs| rhs.starts_with("(1f + ")),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn irreducible_cfg_falls_back_to_dispatch_with_stage_correct_returns() {
        //   v_mov_b32 v1, 0 ; s_cmp_eq_u32 s0, 0 ; s_cbranch_scc1 B
        // A: v_add_f32 v1, 1.0, v1
        // B: v_add_f32 v1, 2.0, v1 ; s_add_u32 s1, s1, 1 ; s_cmp_lt_u32 s1, 4
        //    s_cbranch_scc1 A          (loop A<->B entered at both A and B)
        //    [exp] ; s_endpgm
        let body = vec![
            enc_vop1(0x01, 1, 128),
            enc_sopc(0x06, 0, 128),
            enc_sopp(0x05, branch_simm(8, 16)),
            enc_vop2(0x03, 1, 242, 1),
            enc_vop2(0x03, 1, 244, 1),
            enc_sop2(0x00, 1, 1, 129),
            enc_sopc(0x0A, 1, 132),
            enc_sopp(0x05, branch_simm(28, 12)),
        ];

        let mut vs = body.clone();
        vs.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [1, 1, 1, 1]));
        vs.push(enc_sopp(0x01, 0));
        let mut ps = body.clone();
        ps.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [1, 1, 1, 1]));
        ps.push(enc_sopp(0x01, 0));
        let mut cs = body;
        cs.push(enc_sopp(0x01, 0));

        for (words, stage, ret) in [
            (&vs, Rdna2StageOverride::Vertex, "= vs_out;"),
            (&ps, Rdna2StageOverride::Pixel, "= ps_out;"),
            (&cs, Rdna2StageOverride::Compute, "return;"),
        ] {
            let tree = structured_tree(words);
            assert!(tree.dispatch_count() >= 1, "{stage}: {tree:?}");
            let out = translate_as(words, stage);
            assert_eq!(out.unsupported_op_count, 0, "{stage} WGSL:\n{}", out.wgsl);
            assert!(out.wgsl.contains("switch"), "{stage} WGSL:\n{}", out.wgsl);
            assert!(out.wgsl.contains("_pg6_pc"), "{stage} WGSL:\n{}", out.wgsl);
            // One return inside the dispatcher's `s_endpgm` case, one at function end.
            assert_eq!(
                out.wgsl.matches(ret).count(),
                2,
                "{stage} WGSL:\n{}",
                out.wgsl
            );
            if stage == Rdna2StageOverride::Compute {
                assert!(!out.wgsl.contains("= vs_out;"), "WGSL:\n{}", out.wgsl);
                assert!(!out.wgsl.contains("= ps_out;"), "WGSL:\n{}", out.wgsl);
            } else {
                // Value-returning entry points must never emit a bare `return;`.
                assert!(!out.wgsl.contains("return;"), "{stage} WGSL:\n{}", out.wgsl);
            }
            assert_naga_wgsl_valid(&out.wgsl);
        }
    }

    #[test]
    fn s_barrier_and_implicit_lod_sampling_inside_branches_pass_uniformity_validation() {
        // Compute: barrier inside a branch is skipped; the one after the merge stays.
        //   s_cmp_eq_u32 s0, 0 ; s_cbranch_scc1 L ; s_barrier
        // L: s_barrier ; s_endpgm
        let words = vec![
            enc_sopc(0x06, 0, 128),
            enc_sopp(0x05, branch_simm(4, 12)),
            enc_sopp(0x0A, 0),
            enc_sopp(0x0A, 0),
            enc_sopp(0x01, 0),
        ];
        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains(
                "// note [0x0008] s_barrier skipped inside structured control flow for WGSL uniformity"
            ),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_eq!(
            out.wgsl.matches("workgroupBarrier()").count(),
            1,
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        // Fragment: `image_sample` inside a branch must use an explicit LOD.
        let mut words = vec![
            enc_vintrp(0, 2, 0, 0, 0),
            enc_vintrp(1, 2, 0, 0, 1),
            enc_vintrp(0, 3, 0, 1, 0),
            enc_vintrp(1, 3, 0, 1, 1),
            enc_sopc(0x06, 0, 128),
            enc_sopp(0x05, branch_simm(20, 32)),
        ];
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 2, 1));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));
        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("textureSampleLevel("),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(!out.wgsl.contains("textureSample("), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s_andn2_wrexec_b32_and_saveexec_b64_with_execz_translate_without_unsupported_ops() {
        //   v_mov_b32 v0, 1.0 ; v_cmp_lt_f32 vcc, 0.5, v0
        //   s_andn2_wrexec_b32 s3, vcc_lo ; s_cbranch_execz L
        //   v_mov_b32 v1, 2.0
        // L: s_mov_b32 exec_lo, -1 ; s_endpgm
        let words = vec![
            enc_vop1(0x01, 0, 242),
            enc_vopc(0x01, 240, 0),
            enc_sop1(0x47, 3, 106),
            enc_sopp(0x08, branch_simm(12, 20)),
            enc_vop1(0x01, 1, 244),
            enc_sop1(0x03, 126, 193),
            enc_sopp(0x01, 0),
        ];
        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.translated_op_count, 7, "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("var exec_mask: u32;"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("if ("), "WGSL:\n{}", out.wgsl);
        // wrexec writes the *new* mask to sdst; `s_mov_b32 exec_lo, -1` re-enables the lane.
        assert!(
            last_assignment(&out.wgsl, "s_3_").is_some(),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_eq!(
            last_assignment(&out.wgsl, "exec_mask").as_deref(),
            Some("1u"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(
            last_assignment(&out.wgsl, "scc").is_some(),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        // Wave64 form: s_and_saveexec_b64 s[4:5], vcc ; s_cbranch_execnz T ; s_branch E
        // T: v_mov_b32 v1, 2.0
        // E: s_mov_b64 exec, s[4:5] ; s_endpgm
        let words = vec![
            enc_vop1(0x01, 0, 242),
            enc_vopc(0x01, 240, 0),
            enc_sop1(0x24, 4, 106),
            enc_sopp(0x09, branch_simm(12, 20)),
            enc_sopp(0x02, branch_simm(16, 24)),
            enc_vop1(0x01, 1, 244),
            enc_sop1(0x04, 126, 4),
            enc_sopp(0x01, 0),
        ];
        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("if ("), "WGSL:\n{}", out.wgsl);
        assert_eq!(
            last_assignment(&out.wgsl, "s_5_").as_deref(),
            Some("0u"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn every_saveexec_and_wrexec_opcode_translates_and_validates() {
        let ops: Vec<u8> = (0x24..=0x2B)
            .chain(0x37..=0x3A)
            .chain(0x3C..=0x47)
            .collect();
        assert_eq!(ops.len(), 24);
        for op in ops {
            let words = vec![
                enc_vopc(0x01, 240, 0),
                enc_sop1(op, 2, 106),
                enc_sopp(0x08, branch_simm(8, 16)),
                enc_vop1(0x01, 1, 244),
                enc_sopp(0x01, 0),
            ];
            let insts = decode_program(&words_to_bytes(&words));
            assert!(classify_exec_mask_op(&insts[1]).is_some(), "op {op:#x}");
            let out = translate_as(&words, Rdna2StageOverride::Compute);
            assert_eq!(
                out.unsupported_op_count, 0,
                "op {op:#x} WGSL:\n{}",
                out.wgsl
            );
            assert!(
                out.wgsl.contains("exec_mask"),
                "op {op:#x} WGSL:\n{}",
                out.wgsl
            );
            assert_naga_wgsl_valid(&out.wgsl);
        }
    }

    #[test]
    fn straight_line_programs_keep_the_linear_fast_path() {
        // No branches and no exec writes: no exec_mask, no structured constructs, and the
        // compute barrier stays a real `workgroupBarrier()`.
        let words = vec![enc_vop1(0x01, 1, 242), enc_sopp(0x0A, 0), enc_sopp(0x01, 0)];
        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("exec_mask"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("loop"), "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("workgroupBarrier()"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s31_compute_descriptor_table_sources_2d_array_and_tight_16b_cbv() {
        // Compute stage copying user SGPR s4 -> s8, zero-extending s9=0,
        // loading CBV V# from s[8:9]+0x40 and two T# descriptors via s_load_dwordx16 at +0x0,
        // reading 16B from CBV, materializing an inline static sampler in s[12:15],
        // and sampling a dim:5 (2d-array) texture with image_sample_l.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 8, 4)); // s_mov_b32 s8, s4
        words.push(enc_sop1(0x03, 9, 0x80)); // s_mov_b32 s9, 0
        words.extend_from_slice(&enc_smem(0x02, 12, 4, 0x7D, 0x40, false, false)); // s_load_dwordx4 s[12:15], s[8:9], 0x40
        words.extend_from_slice(&enc_smem(0x04, 24, 4, 0x7D, 0x00, false, false)); // s_load_dwordx16 s[24:39], s[8:9], 0x0
        words.extend_from_slice(&enc_smem(0x0A, 8, 6, 0x7D, 0x00, false, false)); // s_buffer_load_dwordx4 s[8:11], s[12:15], 0x0
        words.push(enc_sop1(0x03, 12, 0x81)); // inline static sampler in s[12:15]
        words.push(enc_sop1(0x03, 13, 0x82));
        words.push(enc_sop1(0x03, 14, 0x83));
        words.push(enc_sop1(0x03, 15, 0x84));
        // image_sample_l v4, v8, s[24:31] (srsrc=6), s[12:15] (ssamp=3) dmask:0x1 dim:5 (2d_array)
        words.extend_from_slice(&enc_mimg(0x24, 0x1, 5, 8, 4, 6, 3));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("texture_2d_array<f32>"), "WGSL:\n{}", out.wgsl);
        // Only the real 16-byte CBV should exist (NOT a bogus 64-byte CBV from s_load_dwordx16).
        let cbvs: Vec<_> = out.bindings.iter().filter(|b| b.kind == "uniform").collect();
        assert_eq!(cbvs.len(), 1, "bindings: {:?}", out.bindings);
        assert_eq!(cbvs[0].min_binding_size_bytes, Some(16));
        assert_eq!(
            cbvs[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0x40,
            })
        );
        let tex = out.bindings.iter().find(|b| b.kind == "texture_2d").unwrap();
        assert_eq!(tex.view_dimension, Some("2d-array"));
        assert_eq!(
            tex.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0,
            })
        );
        let samp = out.bindings.iter().find(|b| b.kind == "sampler").unwrap();
        assert_eq!(
            samp.source,
            Some(WgslBindingSource::InlineSampler {
                sdwords: [1, 2, 3, 4],
            })
        );
        let json = out.to_json();
        assert!(json.contains("\"user_sgpr\": 4, \"table_byte_offset\": 64"));
        assert!(json.contains("\"view_dimension\": \"2d-array\""));
        assert!(json.contains("\"inline_sampler\": {\"sdwords\": [1, 2, 3, 4]}"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s31_vertex_inputs_and_user_data_storage_buffer_separation() {
        // Vertex stage:
        //   s[10:11] <- s[0:1] (hardware VB table ptr) -> s_load_dwordx4 s[12:15], s[10:11], 0x0 (vb_slot 0)
        //   s[8:9]   <- user data table s8 (s9=0)      -> s_load_dwordx4 s[16:19], s[8:9], 0x20 (storage SRV)
        //   buffer_load_dwordx3 v0, v0, s[12:15], 0 idxen offset:0   (VsIn attr @location(0))
        //   buffer_load_dwordx2 v4, v0, s[12:15], 0 idxen offset:16  (VsIn attr @location(2))
        //   buffer_load_dwordx4 v8, v0, s[16:19], 0 offen            (read-only-storage sbuf0)
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 9, 0x80)); // s_mov_b32 s9, 0
        words.push(enc_sop1(0x04, 10, 0)); // s_mov_b64 s[10:11], s[0:1]
        words.extend_from_slice(&enc_smem(0x02, 12, 5, 0x7D, 0x00, false, false)); // VB slot 0
        words.extend_from_slice(&enc_smem(0x02, 16, 4, 0x7D, 0x20, false, false)); // user table slot @ +0x20
        // MUBUF buffer_load_dwordx3 (op 0x0F) idxen=1, offset=0, srsrc=3 (s[12:15])
        let mubuf_pos_w0 = (0b111000u32 << 26) | (0x0Fu32 << 18) | (1u32 << 13);
        let mubuf_pos_w1 = (0x80u32 << 24) | (3u32 << 16);
        words.extend_from_slice(&[mubuf_pos_w0, mubuf_pos_w1]);
        // MUBUF buffer_load_dwordx2 (op 0x0D) idxen=1, offset=16, srsrc=3 (s[12:15])
        let mubuf_uv_w0 = (0b111000u32 << 26) | (0x0Du32 << 18) | (1u32 << 13) | 16;
        let mubuf_uv_w1 = (0x80u32 << 24) | (3u32 << 16) | (4u32 << 8);
        words.extend_from_slice(&[mubuf_uv_w0, mubuf_uv_w1]);
        // MUBUF buffer_load_dwordx4 (op 0x0E) offen=1, offset=0, srsrc=4 (s[16:19])
        let mubuf_sbuf_w0 = (0b111000u32 << 26) | (0x0Eu32 << 18) | (1u32 << 12);
        let mubuf_sbuf_w1 = (0x80u32 << 24) | (4u32 << 16) | (8u32 << 8);
        words.extend_from_slice(&[mubuf_sbuf_w0, mubuf_sbuf_w1]);
        words.extend_from_slice(&enc_exp(12, 0xF, true, false, false, [0, 1, 2, 8]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Vertex);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.vertex_inputs.len(), 2, "inputs: {:?}", out.vertex_inputs);
        assert_eq!(out.vertex_inputs[0].vb_slot, 0);
        assert_eq!(out.vertex_inputs[0].byte_offset, 0);
        assert_eq!(out.vertex_inputs[0].components, 3);
        assert_eq!(out.vertex_inputs[1].vb_slot, 0);
        assert_eq!(out.vertex_inputs[1].byte_offset, 16);
        assert_eq!(out.vertex_inputs[1].components, 2);

        let sbuf = out
            .bindings
            .iter()
            .find(|b| b.kind == "read-only-storage")
            .expect("expected read-only-storage binding for user-data vertex buffer");
        assert_eq!(
            sbuf.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0x20,
            })
        );
        let json = out.to_json();
        assert!(json.contains("\"vertex_inputs\""));
        assert!(json.contains("\"vb_slot\": 0, \"byte_offset\": 16"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s31_vop3_0x371_v_and_or_b32_translates() {
        let mut words = Vec::new();
        words.push(enc_vop1(0x01, 8, 242)); // v_mov_b32 v8, 1.0
        // v_and_or_b32 (0x371) v12, 0x80000000 (literal), v8, 1.0
        words.extend_from_slice(&enc_vop3(0x371, 12, 255, 256 + 8, 242, false, 0, 0, 0));
        words.push(0x8000_0000);
        words.push(enc_sopp(0x01, 0));
        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s32_storage_texture_exports_rgba8unorm_storage_format_22() {
        // Compute shader writing a 2D UAV via image_store -> storage_format: 22 (Rgba8Unorm).
        let mut words = Vec::new();
        words.push(enc_vop1(0x01, 6, 242)); // v6 = 1.0
        words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 6, 4, 0)); // image_store -> uav_tex0
        words.push(enc_sopp(0x01, 0));
        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        let uav = out
            .bindings
            .iter()
            .find(|b| b.kind == "storage-texture")
            .expect("expected storage-texture binding");
        assert_eq!(uav.storage_format, Some(WGPU_STORAGE_FORMAT_RGBA8_UNORM));
        let json = out.to_json();
        assert!(json.contains("\"storage_format\": 22"), "JSON:\n{json}");
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s32_reflected_fragment_bindings_use_disjoint_offset_32() {
        // Fragment shader loading CBV V# and T# descriptors from graphics user SGPR table s8
        // (zero-extended s9=0) plus an inline static sampler -> all bindings must land at 32+ so
        // VS (0..31) and PS (32+) never collide in join_pipeline_layout.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0 (zero-extend graphics user table ptr s8)
        words.extend_from_slice(&enc_smem(0x02, 12, 4, 0x7D, 0x00, false, false)); // CBV0 V# @ +0x00
        words.extend_from_slice(&enc_smem(0x02, 16, 4, 0x7D, 0x20, false, false)); // CBV1 V# @ +0x20
        words.extend_from_slice(&enc_smem(0x03, 24, 4, 0x7D, 0x40, false, false)); // T# @ +0x40
        words.extend_from_slice(&enc_smem(0x0A, 8, 6, 0x7D, 0x00, false, false)); // cb0 load
        words.extend_from_slice(&enc_smem(0x0A, 4, 8, 0x7D, 0x00, false, false)); // cb1 load
        words.push(enc_sop1(0x03, 20, 0x81)); // inline static sampler in s[20:23] (ssamp=5)
        words.push(enc_sop1(0x03, 21, 0x82));
        words.push(enc_sop1(0x03, 22, 0x83));
        words.push(enc_sop1(0x03, 23, 0x84));
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 0, 4, 6, 5)); // image_sample_lz tex0/samp0
        words.push(enc_vop2(0x03, 4, 8, 4)); // v4 = s8 + v4
        words.push(enc_vop2(0x03, 5, 4, 5)); // v5 = s4 + v5
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.binding >= FRAGMENT_REFLECTED_BINDING_OFFSET),
            "all reflected PS bindings must be >= 32: {:?}",
            out.bindings
        );
        let cb0 = out.bindings.iter().find(|b| b.name == "cb0").unwrap();
        let tex0 = out.bindings.iter().find(|b| b.name == "tex0").unwrap();
        let samp0 = out.bindings.iter().find(|b| b.name == "samp0").unwrap();
        let cb1 = out.bindings.iter().find(|b| b.name == "cb1").unwrap();
        assert_eq!(cb0.binding, 32);
        assert_eq!(tex0.binding, 33);
        assert_eq!(samp0.binding, 34);
        assert_eq!(cb1.binding, 35);
        assert!(out.wgsl.contains("@group(0) @binding(32)"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@group(0) @binding(33)"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@group(0) @binding(34)"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@group(0) @binding(35)"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s32_unreflected_fragment_extra_bindings_start_at_16_avoiding_vs_collision() {
        // Unreflected VS with extra cb1 (binding 3) + unreflected PS with extra tex1 (binding 16)
        // must never share a binding slot with mismatched resource types.
        let mut vs_words = Vec::new();
        vs_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x00, false, false)); // cb0 @0
        vs_words.extend_from_slice(&enc_smem(0x0A, 8, 2, 0x7D, 0x00, false, false)); // cb1 @3
        vs_words.push(enc_vop1(0x01, 4, 4));
        vs_words.push(enc_vop1(0x01, 5, 8));
        vs_words.extend_from_slice(&enc_exp(12, 0xF, true, false, false, [4, 5, 4, 5]));
        vs_words.push(enc_sopp(0x01, 0));
        let vs_out = translate_as(&vs_words, Rdna2StageOverride::Vertex);
        assert_eq!(vs_out.unsupported_op_count, 0, "VS WGSL:\n{}", vs_out.wgsl);
        let vs_cb1 = vs_out.bindings.iter().find(|b| b.name == "cb1").unwrap();
        assert_eq!(vs_cb1.binding, 3);

        let mut ps_words = Vec::new();
        ps_words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 0, 4, 2, 1)); // tex0 @1, samp0 @2
        ps_words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 0, 8, 4, 1)); // tex1 @16
        ps_words.push(enc_vop2(0x03, 4, 256 + 4, 8));
        ps_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        ps_words.push(enc_sopp(0x01, 0));
        let ps_out = translate_as(&ps_words, Rdna2StageOverride::Pixel);
        assert_eq!(ps_out.unsupported_op_count, 0, "PS WGSL:\n{}", ps_out.wgsl);
        let ps_tex1 = ps_out.bindings.iter().find(|b| b.name == "tex1").unwrap();
        assert_eq!(ps_tex1.binding, FRAGMENT_EXTRA_BINDING_BASE);
        assert_ne!(vs_cb1.binding, ps_tex1.binding);
        assert_naga_wgsl_valid(&vs_out.wgsl);
        assert_naga_wgsl_valid(&ps_out.wgsl);
    }

    #[test]
    fn s32_mtbuf_integer_and_sub_element_vertex_inputs_emit_typed_vs_in_and_metadata() {
        // Vertex shader fetching:
        //   offset 0:  MTBUF format:56 (unorm8x4) -> vec4<f32>, component_type="f32", fetch_format="unorm8x4"
        //   offset 8:  MTBUF format:14 (unorm8x2) -> vec2<f32>, component_type="f32", fetch_format="unorm8x2"
        //   offset 12: MTBUF format:5  (uint8)    -> u32,       component_type="u32", fetch_format="uint8"
        let mut words = Vec::new();
        words.push(enc_sop1(0x04, 12, 0)); // s[12:13] = s[0:1] (VB table)
        words.extend_from_slice(&enc_smem(0x02, 12, 6, 0x7D, 0, false, false)); // VB slot 0 V#
        words.extend_from_slice(&enc_mtbuf(0x03, 56, 0, 4, 0, 3, true)); // xyzw unorm8x4 @ 0
        words.extend_from_slice(&enc_mtbuf(0x01, 14, 8, 8, 0, 3, true)); // xy   unorm8x2 @ 8
        words.extend_from_slice(&enc_mtbuf(0x00, 5, 12, 10, 0, 3, true)); // x    uint8    @ 12
        words.push(enc_vop1(0x06, 11, 256 + 10)); // v_cvt_f32_u32 v11, v10
        words.extend_from_slice(&enc_exp(12, 0xF, true, false, false, [4, 5, 8, 11]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Vertex);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.vertex_inputs.len(), 3, "vertex_inputs: {:?}", out.vertex_inputs);

        assert_eq!(out.vertex_inputs[0].byte_offset, 0);
        assert_eq!(out.vertex_inputs[0].components, 4);
        assert_eq!(out.vertex_inputs[0].component_type, "f32");
        assert_eq!(out.vertex_inputs[0].fetch_format, Some("unorm8x4"));

        assert_eq!(out.vertex_inputs[1].byte_offset, 8);
        assert_eq!(out.vertex_inputs[1].components, 2);
        assert_eq!(out.vertex_inputs[1].component_type, "f32");
        assert_eq!(out.vertex_inputs[1].fetch_format, Some("unorm8x2"));

        assert_eq!(out.vertex_inputs[2].byte_offset, 12);
        assert_eq!(out.vertex_inputs[2].components, 1);
        assert_eq!(out.vertex_inputs[2].component_type, "u32");
        assert_eq!(out.vertex_inputs[2].fetch_format, Some("uint8"));

        assert!(out.wgsl.contains("@location(1) attr_8_: vec2<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(2) color: u32"), "WGSL:\n{}", out.wgsl);
        let json = out.to_json();
        assert!(json.contains("\"component_type\": \"u32\""), "JSON:\n{json}");
        assert!(json.contains("\"fetch_format\": \"uint8\""), "JSON:\n{json}");
        assert!(json.contains("\"fetch_format\": \"unorm8x2\""), "JSON:\n{json}");
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s33_dynamic_heap_sampler_load_emits_no_bogus_cbv() {
        // Reproduces RS #3 SM 6.6 SamplerDescriptorHeap pixel shader:
        //   s_mov_b32 s10, s11
        //   s_mov_b32 s11, 0
        //   s_lshl_b32 s9, s9, 4
        //   s_load_dwordx4 s[16:19], s[10:11], s9   ; dynamic sampler descriptor load
        //   s_mov_b32 s9, 0
        //   s_load_dwordx8 s[24:31], s[8:9], 0x0    ; static table T# @ s8 + 0
        //   image_sample v0, v2, s[24:31], s[16:19] dmask:0xf dim:1
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 10, 11)); // s10 = s11
        words.push(enc_sop1(0x03, 11, 0x80)); // s11 = 0
        words.push(enc_sop2(0x0E, 9, 9, 128 + 4)); // s_lshl_b32 s9, s9, 4
        words.extend_from_slice(&enc_smem(0x02, 16, 5, 9, 0x0, false, false)); // dynamic s_load_dwordx4
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0
        words.extend_from_slice(&enc_smem(0x03, 24, 4, 0x7D, 0x0, false, false)); // T# @ s8 + 0
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 0, 6, 4)); // image_sample srsrc=6, ssamp=4
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.kind != "uniform"),
            "dynamic sampler descriptor load must NOT emit bogus cb0: {:?}",
            out.bindings
        );
        assert_eq!(out.bindings.len(), 2, "bindings: {:?}", out.bindings);
        assert_eq!(out.bindings[0].kind, "texture_2d");
        assert_eq!(
            out.bindings[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0,
            })
        );
        assert_eq!(out.bindings[1].kind, "sampler");
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s33_resource_key_deduplicates_multi_sgpr_loads_and_separates_reused_sgprs() {
        // Compute shader:
        //   - Loads T# from s5 + 32 into BOTH s[32:39] and s[40:47] and image_load's both -> 1 tex0!
        //   - Loads V# from s5 + 64 into s[16:19] -> buffer_store, then reloads s[16:19] from
        //     s5 + 96 -> buffer_store -> 2 distinct uav_buf0 (s5+64) and uav_buf1 (s5+96)!
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 8, 5)); // s8 = s5
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0
        words.extend_from_slice(&enc_smem(0x03, 32, 4, 0x7D, 0x20, false, false)); // T# s5+32 -> s[32:39]
        words.extend_from_slice(&enc_mimg(0x00, 0xF, 1, 0, 4, 8, 0)); // image_load srsrc=8
        words.extend_from_slice(&enc_smem(0x03, 40, 4, 0x7D, 0x20, false, false)); // SAME T# s5+32 -> s[40:47]
        words.extend_from_slice(&enc_mimg(0x00, 0xF, 1, 0, 8, 10, 0)); // image_load srsrc=10
        // V# s5+64 -> s[16:19], buffer_store_dwordx4 (MUBUF op 0x1E) offen=1 srsrc=4
        words.extend_from_slice(&enc_smem(0x02, 16, 4, 0x7D, 0x40, false, false));
        let st_w0 = (0b111000u32 << 26) | (0x1Eu32 << 18) | (1u32 << 12);
        let st_w1 = (0x80u32 << 24) | (4u32 << 16) | (4u32 << 8);
        words.extend_from_slice(&[st_w0, st_w1]);
        // Reload SAME s[16:19] from s5+96 -> buffer_store_dwordx4
        words.extend_from_slice(&enc_smem(0x02, 16, 4, 0x7D, 0x60, false, false));
        let st2_w1 = (0x80u32 << 24) | (4u32 << 16) | (8u32 << 8);
        words.extend_from_slice(&[st_w0, st2_w1]);
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        let texs: Vec<_> = out.bindings.iter().filter(|b| b.kind == "texture_2d").collect();
        assert_eq!(texs.len(), 1, "duplicate T# loads must collapse to 1 binding: {:?}", out.bindings);
        assert_eq!(
            texs[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 5,
                table_byte_offset: 32,
            })
        );
        let uavs: Vec<_> = out.bindings.iter().filter(|b| b.kind == "storage").collect();
        assert_eq!(uavs.len(), 2, "reused SGPR with 2 V# offsets must emit 2 UAVs: {:?}", out.bindings);
        assert_eq!(
            uavs[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 5,
                table_byte_offset: 64,
            })
        );
        assert_eq!(
            uavs[1].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 5,
                table_byte_offset: 96,
            })
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s33_s_bfm_b64_inline_static_sampler_emits_single_static_sampler_source() {
        // Reproduces RS #5 PS inline static sampler built with s_bfm_b64 + s_mov_b64,
        // reused across two image_sample calls.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0
        words.extend_from_slice(&enc_smem(0x03, 24, 4, 0x7D, 0x0, false, false)); // T# @ s8 + 0
        words.push(enc_sop2(0x25, 20, 128 + 2, 128 + 54)); // s_bfm_b64 s[20:21], 2, 54
        words.push(enc_sop1(0x04, 22, 0xFF)); // s_mov_b64 s[22:23], 0x14500000
        words.push(0x1450_0000);
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 0, 4, 6, 5)); // image_sample #1
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 0, 8, 6, 5)); // image_sample #2 (same sampler)
        words.push(enc_vop2(0x03, 4, 256 + 4, 8));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        let samps: Vec<_> = out.bindings.iter().filter(|b| b.kind == "sampler").collect();
        assert_eq!(samps.len(), 1, "expected 1 deduplicated inline sampler: {:?}", out.bindings);
        assert_eq!(
            samps[0].source,
            Some(WgslBindingSource::InlineSampler {
                sdwords: [0, 0x00C0_0000, 0x1450_0000, 0],
            })
        );
        // All bindings are reflected -> fragment offset +32 applies!
        assert_eq!(samps[0].binding, FRAGMENT_REFLECTED_BINDING_OFFSET + 2);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s33_mimg_dim2_emits_3d_sampled_and_storage_textures() {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 8, 5)); // s8 = s5
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0
        words.extend_from_slice(&enc_smem(0x03, 16, 4, 0x7D, 0x0, false, false)); // 3D SRV @ s5+0
        words.extend_from_slice(&enc_smem(0x03, 24, 4, 0x7D, 0x20, false, false)); // 3D UAV @ s5+32
        words.extend_from_slice(&enc_mimg(0x00, 0xF, 2, 0, 4, 4, 0)); // image_load dim:2 (3d)
        words.extend_from_slice(&enc_mimg(0x08, 0xF, 2, 0, 4, 6, 0)); // image_store dim:2 (3d)
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("texture_3d<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("texture_storage_3d<rgba8unorm,write>")
                || out.wgsl.contains("texture_storage_3d<rgba8unorm, write>"),
            "WGSL:\n{}",
            out.wgsl
        );
        let tex = out.bindings.iter().find(|b| b.kind == "texture_2d").unwrap();
        let uav = out.bindings.iter().find(|b| b.kind == "storage-texture").unwrap();
        assert_eq!(tex.view_dimension, Some("3d"));
        assert_eq!(uav.view_dimension, Some("3d"));
        assert_eq!(uav.storage_format, Some(WGPU_STORAGE_FORMAT_RGBA8_UNORM));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s34_sampler_heap_pattern_a_emits_source_json_and_no_bogus_cbv() {
        // SM 6.6 SAMPLER_HEAP_DIRECTLY_INDEXED (0x800) Pattern A (#482 Comment #21):
        //   s_mov_b32 s10, s11
        //   s_lshl_b32 s9, s9, 4
        //   s_load_dwordx4 s[16:19], s[10:11], s9
        //   s_mov_b32 s9, 0
        //   s_load_dwordx8 s[24:31], s[8:9], 0x0
        //   image_sample v0, v2, s[24:31], s[16:19] dmask:0xf dim:1
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 10, 11)); // s_mov_b32 s10, s11
        words.push(enc_sop2(0x1E, 9, 9, 128 + 4)); // s_lshl_b32 s9, s9, 4
        words.extend_from_slice(&enc_smem(0x02, 16, 5, 9, 0x0, false, false)); // s_load_dwordx4 s[16:19], s[10:11], s9
        words.push(enc_sop1(0x03, 9, 0x80)); // s_mov_b32 s9, 0
        words.extend_from_slice(&enc_smem(0x03, 24, 4, 0x7D, 0x0, false, false)); // s_load_dwordx8 s[24:31], s[8:9], 0x0
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 0, 6, 4)); // image_sample srsrc=6, ssamp=4
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.kind != "uniform"),
            "dynamic sampler heap load must NOT fabricate a spurious CBV: {:?}",
            out.bindings
        );
        assert_eq!(out.bindings.len(), 2, "bindings: {:?}", out.bindings);
        assert_eq!(out.bindings[0].kind, "texture_2d");
        assert_eq!(
            out.bindings[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0,
            })
        );
        assert_eq!(out.bindings[1].kind, "sampler");
        assert_eq!(
            out.bindings[1].source,
            Some(WgslBindingSource::SamplerHeap {
                base_user_sgpr: 11,
                stride: 16,
                index_user_sgpr: 9,
                index_shift: 4,
            })
        );
        // All bindings are reflected -> fragment offset +32 applies.
        assert_eq!(out.bindings[0].binding, FRAGMENT_REFLECTED_BINDING_OFFSET + 1);
        assert_eq!(out.bindings[1].binding, FRAGMENT_REFLECTED_BINDING_OFFSET + 2);
        let json = out.to_json();
        assert!(
            json.contains(
                "\"source\": {\"sampler_heap\": {\"base_user_sgpr\": 11, \"stride\": 16, \"index\": {\"user_sgpr\": 9, \"shift\": 4}}}"
            ),
            "JSON:\n{json}"
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s34_sampler_heap_pattern_b_in_place_overwrite_emits_source() {
        // SM 6.6 SAMPLER_HEAP_DIRECTLY_INDEXED (0x800) Pattern B (#482 Comment #21):
        //   s_mov_b32 s10, s9
        //   s_mov_b32 s9, 0
        //   s_load_dwordx8 s[16:23], s[8:9], 0x0
        //   s_mov_b32 s8, s11
        //   s_lshl_b32 s10, s10, 4
        //   s_load_dwordx4 s[8:11], s[8:9], s10    ; overwrites BOTH base (s8:9) and index (s10)!
        //   image_sample v0, v2, s[16:23], s[8:11] dmask:0xf dim:1
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 10, 9)); // s_mov_b32 s10, s9
        words.push(enc_sop1(0x03, 9, 0x80)); // s_mov_b32 s9, 0
        words.extend_from_slice(&enc_smem(0x03, 16, 4, 0x7D, 0x0, false, false)); // T# @ s8 + 0 -> s[16:23]
        words.push(enc_sop1(0x03, 8, 11)); // s_mov_b32 s8, s11
        words.push(enc_sop2(0x1E, 10, 10, 128 + 4)); // s_lshl_b32 s10, s10, 4
        words.extend_from_slice(&enc_smem(0x02, 8, 4, 10, 0x0, false, false)); // s_load_dwordx4 s[8:11], s[8:9], s10
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 0, 4, 2)); // image_sample srsrc=4, ssamp=2
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.kind != "uniform"),
            "in-place dynamic sampler heap load must NOT fabricate a spurious CBV: {:?}",
            out.bindings
        );
        assert_eq!(out.bindings.len(), 2, "bindings: {:?}", out.bindings);
        assert_eq!(
            out.bindings[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0,
            })
        );
        assert_eq!(
            out.bindings[1].source,
            Some(WgslBindingSource::SamplerHeap {
                base_user_sgpr: 11,
                stride: 16,
                index_user_sgpr: 9,
                index_shift: 4,
            })
        );
        assert_eq!(out.bindings[0].binding, FRAGMENT_REFLECTED_BINDING_OFFSET + 1);
        assert_eq!(out.bindings[1].binding, FRAGMENT_REFLECTED_BINDING_OFFSET + 2);
        let json = out.to_json();
        assert!(
            json.contains(
                "\"source\": {\"sampler_heap\": {\"base_user_sgpr\": 11, \"stride\": 16, \"index\": {\"user_sgpr\": 9, \"shift\": 4}}}"
            ),
            "JSON:\n{json}"
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s34_resource_heap_texture_and_buffers_emit_source_and_key_by_origin() {
        // Compute shader using SM 6.6 ResourceDescriptorHeap (0x400) + SamplerDescriptorHeap (0x800):
        //   - s4 = resource heap base, s5 = tex index, s6 = buf0 index, s7 = buf1 index,
        //     s8 = sampler heap base, s9 = sampler index
        //   - T# loaded via s_lshl_b32 (shift 5 -> stride 32) into BOTH s[24:31] and s[32:39]
        //     (must deduplicate into 1 texture_2d binding!).
        //   - S# loaded via s_lshl4_add_u32 + static-offset s_load_dwordx4 -> SamplerHeap!
        //   - V#0 loaded via s_mul_i32 (pow2 stride 16 -> shift 4) into s[16:19] -> buffer_load.
        //   - V#1 loaded via s_mul_i32 (non-pow2 stride 24 -> shift 0) into SAME s[16:19] -> buffer_load
        //     (must separate into 2 distinct read-only-storage bindings despite SGPR reuse!).
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 10, 4)); // s10 = s4 (resource heap base)
        words.push(enc_sop1(0x03, 11, 0x80)); // s11 = 0
        words.push(enc_sop2(0x1E, 12, 5, 128 + 5)); // s_lshl_b32 s12, s5, 5 (stride 32)
        words.extend_from_slice(&enc_smem(0x03, 24, 5, 12, 0x0, false, false)); // T# -> s[24:31]
        words.extend_from_slice(&enc_smem(0x03, 32, 5, 12, 0x0, false, false)); // SAME T# -> s[32:39]
        // S# via s_lshl4_add_u32 s14, s9, s8 ; s_mov_b32 s15, 0 ; s_load_dwordx4 s[20:23], s[14:15], 0x0
        words.push(enc_sop2(0x31, 14, 9, 8)); // s_lshl4_add_u32 s14, s9, s8
        words.push(enc_sop1(0x03, 15, 0x80)); // s15 = 0
        words.extend_from_slice(&enc_smem(0x02, 20, 7, 0x7D, 0x0, false, false)); // S# -> s[20:23]
        // Sample both T# copies with the dynamic heap sampler
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 0, 4, 6, 5)); // image_sample_lz srsrc=6 (s24), ssamp=5 (s20)
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 0, 8, 8, 5)); // image_sample_lz srsrc=8 (s32), ssamp=5 (s20)
        // V#0: s_mul_i32 s13, s6, 16 -> s_load_dwordx4 s[16:19], s[10:11], s13 -> buffer_load_dwordx4
        words.push(enc_sop2(0x26, 13, 6, 128 + 16)); // s_mul_i32 s13, s6, 16
        words.extend_from_slice(&enc_smem(0x02, 16, 5, 13, 0x0, false, false)); // V#0 -> s[16:19]
        let ld_w0 = (0b111000u32 << 26) | (0x0Eu32 << 18) | (1u32 << 12);
        let ld1_w1 = (0x80u32 << 24) | (4u32 << 16) | (12u32 << 8);
        words.extend_from_slice(&[ld_w0, ld1_w1]);
        // V#1: s_mul_i32 s13, s7, 24 -> reload SAME s[16:19] -> buffer_load_dwordx4
        words.push(enc_sop2(0x26, 13, 7, 128 + 24)); // s_mul_i32 s13, s7, 24
        words.extend_from_slice(&enc_smem(0x02, 16, 5, 13, 0x0, false, false)); // V#1 -> s[16:19]
        let ld2_w1 = (0x80u32 << 24) | (4u32 << 16) | (16u32 << 8);
        words.extend_from_slice(&[ld_w0, ld2_w1]);
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.kind != "uniform"),
            "dynamic heap T#/S#/V# loads must NOT fabricate spurious CBVs: {:?}",
            out.bindings
        );

        let texs: Vec<_> = out.bindings.iter().filter(|b| b.kind == "texture_2d").collect();
        assert_eq!(texs.len(), 1, "duplicate ResourceHeap T# must deduplicate: {:?}", out.bindings);
        assert_eq!(
            texs[0].source,
            Some(WgslBindingSource::ResourceHeap {
                base_user_sgpr: 4,
                stride: 32,
                index_user_sgpr: 5,
                index_shift: 5,
            })
        );

        let samps: Vec<_> = out.bindings.iter().filter(|b| b.kind == "sampler").collect();
        assert_eq!(samps.len(), 1, "bindings: {:?}", out.bindings);
        assert_eq!(
            samps[0].source,
            Some(WgslBindingSource::SamplerHeap {
                base_user_sgpr: 8,
                stride: 16,
                index_user_sgpr: 9,
                index_shift: 4,
            })
        );

        let sbufs: Vec<_> = out
            .bindings
            .iter()
            .filter(|b| b.kind == "read-only-storage")
            .collect();
        assert_eq!(sbufs.len(), 2, "reused SGPR across 2 heap indices must emit 2 sbufs: {:?}", out.bindings);
        assert_eq!(
            sbufs[0].source,
            Some(WgslBindingSource::ResourceHeap {
                base_user_sgpr: 4,
                stride: 16,
                index_user_sgpr: 6,
                index_shift: 4,
            })
        );
        assert_eq!(
            sbufs[1].source,
            Some(WgslBindingSource::ResourceHeap {
                base_user_sgpr: 4,
                stride: 24,
                index_user_sgpr: 7,
                index_shift: 0,
            })
        );

        let json = out.to_json();
        assert!(
            json.contains(
                "\"source\": {\"resource_heap\": {\"base_user_sgpr\": 4, \"stride\": 32, \"index\": {\"user_sgpr\": 5, \"shift\": 5}}}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "\"source\": {\"resource_heap\": {\"base_user_sgpr\": 4, \"stride\": 16, \"index\": {\"user_sgpr\": 6, \"shift\": 4}}}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "\"source\": {\"resource_heap\": {\"base_user_sgpr\": 4, \"stride\": 24, \"index\": {\"user_sgpr\": 7, \"shift\": 0}}}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "\"source\": {\"sampler_heap\": {\"base_user_sgpr\": 8, \"stride\": 16, \"index\": {\"user_sgpr\": 9, \"shift\": 4}}}"
            ),
            "JSON:\n{json}"
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    fn enc_sopk(op: u8, sdst: u8, simm16: u16) -> u32 {
        (0b1011u32 << 28)
            | (u32::from(op & 0x1F) << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | u32::from(simm16)
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_vop3_opsel(
        op: u16,
        vdst: u8,
        src0: u16,
        src1: u16,
        src2: u16,
        op_sel: u8,
        clamp: bool,
        omod: u8,
    ) -> [u32; 2] {
        let w0 = (0b110101u32 << 26)
            | (u32::from(op & 0x3FF) << 16)
            | (u32::from(clamp) << 15)
            | (u32::from(op_sel & 0xF) << 11)
            | u32::from(vdst);
        let w1 = (u32::from(omod & 0x3) << 27)
            | (u32::from(src2 & 0x1FF) << 18)
            | (u32::from(src1 & 0x1FF) << 9)
            | u32::from(src0 & 0x1FF);
        [w0, w1]
    }

    #[test]
    fn s38_translates_v_interp_f16_p1ll_p2_with_op_sel_packing_into_fma_mix_and_compr_mrt0() {
        // Real UI/menu fragment shader pattern using VOP3 f16 interpolation:
        //   v_interp_p1ll_f16 v15, v0, attr0.x
        //   v_interp_p2_f16   v16, v1, attr0.x, v15               (op_sel=0 -> v16.lo)
        //   v_interp_p1ll_f16 v15, v0, attr0.y
        //   v_interp_p2_f16   v16, v1, attr0.y, v15 op_sel:[0,0,0,1] (op_sel[3]=1 -> v16.hi)
        //   v_interp_p1ll_f16 v17, v0, attr1.x high
        //   v_interp_p2_f16   v18, v1, attr1.x, v17 high          (op_sel=0 -> v18.lo)
        //   v_interp_p1lv_f16 v17, v0, attr1.w, v18
        //   v_interp_p2_f16   v18, v1, attr1.w, v17 op_sel:[0,0,0,1] clamp
        //   v_fma_mixlo_f16   v19, v16, v18, 0.5 op_sel_hi:[1,1,0]
        //   v_fma_mixhi_f16   v19, v16, v18, 0.5 op_sel:[1,1,0] op_sel_hi:[1,1,0]
        //   exp mrt0 v19, v18, off, off done compr vm
        //   s_endpgm
        let attr0_x: u16 = (0 << 6) | 0;
        let attr0_y: u16 = (1 << 6) | 0;
        let attr1_x_hi: u16 = (1 << 8) | (0 << 6) | 1;
        let attr1_w: u16 = (3 << 6) | 1;

        let mut words = Vec::new();
        words.extend_from_slice(&enc_vop3_opsel(0x342, 15, attr0_x, 256 + 0, 0, 0, false, 0));
        words.extend_from_slice(&enc_vop3_opsel(0x35A, 16, attr0_x, 256 + 1, 256 + 15, 0, false, 0));
        words.extend_from_slice(&enc_vop3_opsel(0x342, 15, attr0_y, 256 + 0, 0, 0, false, 0));
        words.extend_from_slice(&enc_vop3_opsel(
            0x35A,
            16,
            attr0_y,
            256 + 1,
            256 + 15,
            0b1000,
            false,
            0,
        ));
        words.extend_from_slice(&enc_vop3_opsel(0x342, 17, attr1_x_hi, 256 + 0, 0, 0, false, 0));
        words.extend_from_slice(&enc_vop3_opsel(
            0x35A,
            18,
            attr1_x_hi,
            256 + 1,
            256 + 17,
            0,
            false,
            0,
        ));
        words.extend_from_slice(&enc_vop3_opsel(
            0x343,
            17,
            attr1_w,
            256 + 0,
            256 + 18,
            0,
            false,
            0,
        ));
        words.extend_from_slice(&enc_vop3_opsel(
            0x35A,
            18,
            attr1_w,
            256 + 1,
            256 + 17,
            0b1000,
            true,
            0,
        ));
        words.extend_from_slice(&enc_vop3p(
            0x21,
            19,
            256 + 16,
            256 + 18,
            240,
            false,
            0b000,
            0b011,
            0,
            0,
        ));
        words.extend_from_slice(&enc_vop3p(
            0x22,
            19,
            256 + 16,
            256 + 18,
            240,
            false,
            0b011,
            0b011,
            0,
            0,
        ));
        words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [19, 18, 0, 0]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_rdna2_to_wgsl(&words_to_bytes(&words), &Rdna2TranslateOptions::default())
            .expect("v_interp_*_f16 fragment shader must translate and validate");
        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.fragment_input_locations, vec![0, 1]);
        assert!(
            out.wgsl.contains("@location(0) attr_0_: vec4<f32>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("@location(1) attr_1_: vec4<f32>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(
            !out.wgsl.contains("var s_0_:") && !out.wgsl.contains("var s_1_:"),
            "attr bitfield in src0 must not allocate bogus SGPRs:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s38_translates_ieee_f32_cmp_and_cmpx_family_with_nan_aware_logic_and_exec_updates() {
        let mut words = Vec::new();
        // Seed v0, v1 from fragment attr0.xy so compares stay runtime expressions.
        words.push(enc_vintrp(0, 0, 0, 0, 0));
        words.push(enc_vintrp(0, 1, 0, 1, 0));
        // VOPC v_cmp_*_f32: o (0x07), u (0x08), nge (0x09), nlg (0x0A), ngt (0x0B), nle (0x0C), nlt (0x0E)
        words.push(enc_vopc(0x07, 256 + 0, 1));
        words.push(enc_vopc(0x08, 256 + 0, 1));
        words.push(enc_vopc(0x09, 256 + 0, 1));
        words.push(enc_vopc(0x0A, 256 + 0, 1));
        words.push(enc_vopc(0x0B, 256 + 0, 1));
        words.push(enc_vopc(0x0C, 256 + 0, 1));
        words.push(enc_vopc(0x0E, 256 + 0, 1));
        // VOP3 v_cmp_*_f32 into SGPR pair s[4:5]
        words.extend_from_slice(&enc_vop3(0x00E, 4, 256 + 0, 256 + 1, 0, false, 0, 0, 0)); // v_cmp_nlt_f32
        words.extend_from_slice(&enc_vop3(0x009, 6, 256 + 0, 256 + 1, 0, false, 0, 0, 0)); // v_cmp_nge_f32
        words.extend_from_slice(&enc_vop3(0x00B, 8, 256 + 0, 256 + 1, 0, false, 0, 0, 0)); // v_cmp_ngt_f32
        words.extend_from_slice(&enc_vop3(0x00A, 10, 256 + 0, 256 + 1, 0, false, 0, 0, 0)); // v_cmp_nlg_f32
        // VOPC v_cmpx_*_f32: nge (0x19), nlg (0x1A), ngt (0x1B), nlt (0x1E)
        words.push(enc_vopc(0x19, 256 + 0, 1));
        words.push(enc_vopc(0x1A, 256 + 0, 1));
        words.push(enc_vopc(0x1B, 256 + 0, 1));
        words.push(enc_vopc(0x1E, 256 + 0, 1));
        // VOP3 v_cmpx_*_f32 writing exec_lo (0x7E)
        words.extend_from_slice(&enc_vop3(0x01E, 126, 256 + 0, 256 + 1, 0, false, 0, 0b001, 0));
        words.extend_from_slice(&enc_vop3(0x019, 126, 256 + 0, 256 + 1, 0, false, 0, 0, 0b010));
        words.extend_from_slice(&enc_vop3(0x01B, 126, 256 + 0, 256 + 1, 0, false, 0, 0, 0));
        words.extend_from_slice(&enc_vop3(0x01A, 126, 256 + 0, 256 + 1, 0, false, 0, 0, 0));
        // Structured s_cbranch_execz +1 (skips 1 dword v_mov_b32 v2, 1.0)
        words.push(enc_sopp(0x08, 1));
        words.push(enc_vop1(0x01, 2, 242));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 2]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var exec_mask: u32"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("exec_mask ="), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("!("), "NaN-aware negated compare expected:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s38_translates_s_round_mode_and_image_get_lod_without_breaking_texture_bindings() {
        // Case A: s_round_mode + image_get_lod alongside image_sample (bindings stay intact)
        let mut words = Vec::new();
        words.push(enc_sopp(0x24, 0x0000)); // s_round_mode
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        // Load T# from user SGPR s8 table (sbase=4 -> s[8:9]) into s[12:19], S# into s[20:23]
        words.extend_from_slice(&enc_smem(0x03, 12, 4, 0x7D, 0x20, false, false)); // T# -> s[12:19]
        words.extend_from_slice(&enc_smem(0x02, 20, 4, 0x7D, 0x40, false, false)); // S# -> s[20:23]
        words.extend_from_slice(&enc_mimg(0x60, 0x3, 1, 2, 4, 3, 5)); // image_get_lod v[4:5], v[2:3], s[12:19], s[20:23] dmask:0x3
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 6, 3, 5)); // image_sample v[6:9], v[2:3], s[12:19], s[20:23]
        words.push(enc_vop2(0x03, 6, 256 + 6, 4)); // v6 = v6 + v4 (uses lod.x)
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [6, 7, 8, 9]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.translated_op_count >= 9);
        assert_eq!(
            out.bindings.iter().filter(|b| b.kind == "texture_2d").count(),
            1,
            "bindings: {:?}",
            out.bindings
        );
        assert_eq!(
            out.bindings.iter().filter(|b| b.kind == "sampler").count(),
            1,
            "bindings: {:?}",
            out.bindings
        );
        assert!(
            out.bindings.iter().all(|b| b.kind != "uniform"),
            "T#/S# descriptor loads must not become CBVs: {:?}",
            out.bindings
        );
        assert!(out.wgsl.contains("v_4_ = 0f;"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("v_5_ = 0f;"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // Case B: standalone image_get_lod (no image_sample) emits zero dead texture/CBV bindings
        let mut solo = Vec::new();
        solo.push(enc_sopp(0x24, 0x0001)); // s_round_mode
        solo.extend_from_slice(&enc_smem(0x03, 12, 4, 0x7D, 0x20, false, false)); // T#
        solo.extend_from_slice(&enc_smem(0x02, 20, 4, 0x7D, 0x40, false, false)); // S#
        solo.extend_from_slice(&enc_mimg(0x60, 0xF, 1, 2, 4, 3, 5)); // image_get_lod dmask:0xf
        solo.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        solo.push(enc_sopp(0x01, 0));

        let out_solo = translate_as(&solo, Rdna2StageOverride::Pixel);
        assert!(out_solo.naga_validated);
        assert_eq!(out_solo.unsupported_op_count, 0, "WGSL:\n{}", out_solo.wgsl);
        assert!(
            out_solo.bindings.is_empty(),
            "standalone image_get_lod must emit no dead bindings or bogus CBVs: {:?}",
            out_solo.bindings
        );
        assert_naga_wgsl_valid(&out_solo.wgsl);
    }

    #[test]
    fn s38_translates_v_add_f64_s_flbit_s_bitset1_s_cmpk_and_v_cvt_pk_u8_f32() {
        // Part 1: runtime forms (with fragment attribute + CBV inputs)
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 0, 0, 0, 0)); // v0 = attr0.x
        words.push(enc_vintrp(0, 1, 0, 1, 0)); // v1 = attr0.y
        words.extend_from_slice(&enc_smem(0x00, 2, 0, 0x7D, 0x00, false, false)); // s2 = cb0[0].x
        // s_flbit_i32_b32 s3, s2
        words.push(enc_sop1(0x15, 3, 2));
        // s_bitset1_b32 s3, 5 (read-modify-write s3)
        words.push(enc_sop1(0x1D, 3, 128 + 5));
        // s_bitset0_b32 s3, 2
        words.push(enc_sop1(0x1B, 3, 128 + 2));
        // s_cmpk_eq_i32 s3, 32 -> SCC, then s_cselect_b32 s4, s3, 0
        words.push(enc_sopk(0x03, 3, 32));
        words.push(enc_sop2(0x0A, 4, 3, 128));
        // Also cover signed/unsigned s_cmpk_* variants
        words.push(enc_sopk(0x04, 3, (-1i16) as u16)); // s_cmpk_lg_i32
        words.push(enc_sopk(0x09, 3, 0xFFFF)); // s_cmpk_eq_u32
        words.push(enc_sopk(0x0D, 3, 100)); // s_cmpk_lt_u32
        // v_add_f64 v[4:5], v0, v1
        words.extend_from_slice(&enc_vop3(0x364, 4, 256 + 0, 256 + 1, 0, false, 0, 0, 0));
        // v_cvt_pk_u8_f32 v6, v0, 0, 0
        words.extend_from_slice(&enc_vop3(0x15E, 6, 256 + 0, 128 + 0, 128 + 0, false, 0, 0, 0));
        // v_cvt_pk_u8_f32 v6, v1, 2, v6
        words.extend_from_slice(&enc_vop3(0x15E, 6, 256 + 1, 128 + 2, 256 + 6, false, 0, 0, 0));
        // v_cvt_pk_u8_f32 v7, v4, s4, v6 (runtime byte index + constant f32 variant too)
        words.extend_from_slice(&enc_vop3(0x15E, 7, 256 + 4, 4, 256 + 6, false, 0, 0, 0));
        words.extend_from_slice(&enc_vop3(0x15E, 8, 242, 128 + 3, 256 + 7, false, 0, 0, 0));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 8]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("countLeadingZeros("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("v_5_ = 0f;"), "v_add_f64 hi word should be 0:\n{}", out.wgsl);
        assert!(out.wgsl.contains("clamp("), "v_cvt_pk_u8_f32 should clamp f32:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // Part 2: constant-folding verification for s_flbit_i32_b32 exact edge cases
        let mut cwords = Vec::new();
        cwords.push(enc_sop1(0x15, 2, 128 + 0)); // s_flbit_i32_b32 s2, 0 -> 0xFFFFFFFF (-1)
        cwords.push(enc_sop1(0x15, 3, 0xFF)); // s_flbit_i32_b32 s3, 0x80000000 -> 0
        cwords.push(0x8000_0000);
        cwords.push(enc_sop1(0x15, 4, 0xFF)); // s_flbit_i32_b32 s4, 0x0000CCCC -> 16
        cwords.push(0x0000_CCCC);
        cwords.push(enc_sop1(0x15, 5, 128 + 1)); // s_flbit_i32_b32 s5, 1 -> 31
        cwords.push(enc_sopp(0x01, 0));

        let cout = translate_as(&cwords, Rdna2StageOverride::Compute);
        assert_eq!(cout.unsupported_op_count, 0, "WGSL:\n{}", cout.wgsl);
        assert_eq!(last_assignment(&cout.wgsl, "s_2_").as_deref(), Some("4294967295u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_3_").as_deref(), Some("0u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_4_").as_deref(), Some("16u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_5_").as_deref(), Some("31u"));
        assert_naga_wgsl_valid(&cout.wgsl);
    }

    #[test]
    fn s39_translates_v_swap_b32_with_exact_read_write_tracking_and_temp_swap() {
        // Fragment shader swapping v0 and v1 via VOP1 v_swap_b32, swapping v2 and v3 via VOP3
        // v_swap_b32 (0x1E5), plus self-swap v_swap_b32 v2, v2 (no-op).
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 0, 0, 0, 0)); // v0 = attr0.x
        words.push(enc_vintrp(0, 1, 0, 1, 0)); // v1 = attr0.y
        words.push(enc_vintrp(0, 2, 0, 2, 0)); // v2 = attr0.z
        words.push(enc_vintrp(0, 3, 0, 3, 0)); // v3 = attr0.w
        words.push(enc_vop1(0x65, 0, 256 + 1)); // v_swap_b32 v0, v1
        words.extend_from_slice(&enc_vop3(0x1E5, 2, 256 + 3, 0, 0, false, 0, 0, 0)); // v_swap_b32_e64 v2, v3
        words.push(enc_vop1(0x65, 2, 256 + 2)); // self-swap no-op
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        // Verify swap captures pre-store values in `let` temps before writing both VGPRs
        // (so v_0_ and v_1_ do not clobber each other).
        let swap_idx = out
            .wgsl
            .find("v_0_ = _e")
            .expect("v_swap_b32 must assign captured v_1_ temp into v_0_");
        let after_swap = &out.wgsl[swap_idx..];
        assert!(
            after_swap.contains("v_1_ = _e"),
            "v_swap_b32 must assign captured v_0_ temp into v_1_:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        // Verify vertex read-before-write tracking: swapping unwritten v4 with v0 marks v4 as read!
        // Without IA MUBUF/MTBUF fetches, vertex_input_mode stays "vertex_index" (S42 / #949),
        // while with an IA MUBUF fetch, reading unwritten v4 triggers the legacy uv input.
        let mut vs_words = Vec::new();
        vs_words.push(enc_vop1(0x65, 2, 256 + 0)); // v_swap_b32 v2, v0 (no IA fetch -> vertex_index)
        vs_words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [0, 1, 2, 3]));
        vs_words.push(enc_sopp(0x01, 0));
        let vs_out = translate_as(&vs_words, Rdna2StageOverride::Vertex);
        assert_eq!(vs_out.unsupported_op_count, 0, "WGSL:\n{}", vs_out.wgsl);
        assert_eq!(
            vs_out.binding_summary.vertex_input_mode, "vertex_index",
            "VS without IA buffer fetches must emit vertex_index mode (S42 / #949)"
        );
        assert!(vs_out.wgsl.contains("v_2_ = _e"), "WGSL:\n{}", vs_out.wgsl);
        assert_naga_wgsl_valid(&vs_out.wgsl);

        let mut vs_attr_words = Vec::new();
        vs_attr_words.extend_from_slice(&enc_mubuf(0x03, 0, 0, 0, 1, true)); // buffer_load_format_xyzw
        vs_attr_words.push(enc_vop1(0x65, 4, 256 + 0)); // v_swap_b32 v4, v0 (v4 read before write!)
        vs_attr_words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [0, 1, 2, 3]));
        vs_attr_words.push(enc_sopp(0x01, 0));
        let vs_attr_out = translate_as(&vs_attr_words, Rdna2StageOverride::Vertex);
        assert_eq!(vs_attr_out.unsupported_op_count, 0, "WGSL:\n{}", vs_attr_out.wgsl);
        assert_eq!(vs_attr_out.binding_summary.vertex_input_mode, "attributes");
        assert!(
            vs_attr_out.wgsl.contains("@location(1) uv: vec2<f32>"),
            "v_swap_b32 with unwritten v4 must record read-before-write VGPR:\n{}",
            vs_attr_out.wgsl
        );
        assert_naga_wgsl_valid(&vs_attr_out.wgsl);
    }

    #[test]
    fn s39_translates_v_writelane_and_v_readlane_b32_multi_lane_sgpr_spill_and_descriptor_origin() {
        // Real GTA VI compiler pattern:
        //   1. Spill descriptor table pointer s8 -> (v40, lane 0) and scalar constants/loop vars -> (v40, lane 1 & 2)
        //   2. Clobber s8 with another value
        //   3. Reload s12 <- (v40, lane 0) [restoring s8's UserTable origin!], s14 <- (v40, lane 1), s15 <- (v40, lane 2)
        //   4. Load T# from s[12:13] + 0x10 and sample it (verifying descriptor reflection survives the spill!)
        //   5. Also test dynamic-lane v_readlane_b32 (selecting across lane slots) & v_readlane_b32 on a normal VALU VGPR.
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0)); // v2 = attr0.x
        words.push(enc_vintrp(0, 3, 0, 1, 0)); // v3 = attr0.y
        // Spill s8 (UserTable 8) to v40 lane 0, 0x11112222 to v40 lane 1, 0x33334444 to v40 lane 2
        words.extend_from_slice(&enc_vop3(0x361, 40, 8, 128 + 0, 0, false, 0, 0, 0)); // v_writelane_b32 v40, s8, 0
        words.extend_from_slice(&enc_vop3(0x361, 40, 0xFF, 128 + 1, 0, false, 0, 0, 0)); // v_writelane_b32 v40, lit1, 1
        words.push(0x1111_2222);
        words.extend_from_slice(&enc_vop3(0x361, 40, 0xFF, 128 + 2, 0, false, 0, 0, 0)); // v_writelane_b32 v40, lit2, 2
        words.push(0x3333_4444);
        // Clobber s8
        words.push(enc_sop1(0x03, 8, 128 + 0)); // s_mov_b32 s8, 0
        // Reload spilled values from DISTINCT lanes of v40
        words.extend_from_slice(&enc_vop3(0x360, 12, 256 + 40, 128 + 0, 0, false, 0, 0, 0)); // s12 <- v40[0] (s8!)
        words.push(enc_sop1(0x03, 13, 0x80)); // s13 = 0 (zero-extend 32-bit console table ptr)
        words.extend_from_slice(&enc_vop3(0x360, 14, 256 + 40, 128 + 1, 0, false, 0, 0, 0)); // s14 <- v40[1]
        words.extend_from_slice(&enc_vop3(0x360, 15, 256 + 40, 128 + 2, 0, false, 0, 0, 0)); // s15 <- v40[2]
        // Dynamic-lane read from v40 using s14 as lane index
        words.extend_from_slice(&enc_vop3(0x360, 16, 256 + 40, 14, 0, false, 0, 0, 0)); // s16 <- v40[s14 & 63]
        // Readlane from normal VALU VGPR v2 (no writelane on v2 -> reads v_2_)
        words.extend_from_slice(&enc_vop3(0x360, 17, 256 + 2, 128 + 0, 0, false, 0, 0, 0)); // s17 <- v2[0]
        // Load T# from reloaded s[12:13] + 0x10 -> s[24:31], S# from s[12:13] + 0x30 -> s[20:23]
        words.extend_from_slice(&enc_smem(0x03, 24, 6, 0x7D, 0x10, false, false)); // sbase=6 -> s[12:13]
        words.extend_from_slice(&enc_smem(0x02, 20, 6, 0x7D, 0x30, false, false));
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 6, 5)); // image_sample srsrc=6 (s24), ssamp=5 (s20)
        // Use reloaded scalars s14, s15, s16, s17
        words.push(enc_vop2(0x1D, 4, 14, 4)); // v4 ^= s14
        words.push(enc_vop2(0x1D, 5, 15, 5)); // v5 ^= s15
        words.push(enc_vop2(0x1D, 6, 16, 6)); // v6 ^= s16
        words.push(enc_vop2(0x1D, 7, 17, 7)); // v7 ^= s17
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        // Must allocate distinct per-(vgpr, lane) spill locals so lane 1 & 2 never clobber lane 0!
        assert!(out.wgsl.contains("var lane_v40_0_: u32;"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var lane_v40_1_: u32;"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var lane_v40_2_: u32;"), "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("lane_v40_1_ = 286335522u;"),
            "0x11112222 must be stored in lane_v40_1_:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("lane_v40_2_ = 858997828u;"),
            "0x33334444 must be stored in lane_v40_2_:\n{}",
            out.wgsl
        );
        // v_readlane_b32 writes SGPRs s12, s14..s17 and must NOT fabricate bogus VGPRs v_12_/v_14_/v_15_
        assert!(!out.wgsl.contains("var v_12_:"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("var v_14_:"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("var v_15_:"), "WGSL:\n{}", out.wgsl);
        // Descriptor table reflection MUST survive the spill/reload through v40 lane 0!
        assert!(
            out.bindings.iter().all(|b| b.kind != "uniform"),
            "spilled descriptor table T#/S# loads must not fabricate bogus CBVs: {:?}",
            out.bindings
        );
        let tex = out
            .bindings
            .iter()
            .find(|b| b.kind == "texture_2d")
            .expect("texture_2d binding expected");
        assert_eq!(
            tex.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0x10,
            })
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s39_translates_s_not_b64_and_s_ff1_i32_b32_runtime_and_const_fold() {
        // Part 1: Runtime shader with s_not_b64 (updating SGPR pair + SCC), s_not_b64 vcc,
        // and s_ff1_i32_b32
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x01, 2, 0, 0x7D, 0x00, false, false)); // s[2:3] = cb0[0].xy
        words.push(enc_sop1(0x08, 4, 2)); // s_not_b64 s[4:5], s[2:3] (sets scc)
        words.push(enc_sop2(0x0A, 6, 4, 5)); // s_cselect_b32 s6, s4, s5 (reads scc)
        words.push(enc_sop1(0x08, 106, 106)); // s_not_b64 vcc, vcc
        words.push(enc_sop1(0x13, 7, 6)); // s_ff1_i32_b32 s7, s6
        words.push(enc_vop1(0x01, 0, 4));
        words.push(enc_vop1(0x01, 1, 5));
        words.push(enc_vop1(0x01, 2, 6));
        words.push(enc_vop1(0x01, 3, 7));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var s_4_: u32;"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var s_5_: u32;"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var scc: bool;"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("countTrailingZeros("), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // Part 2: Exact constant folding for s_not_b64 (lo/hi/scc) and s_ff1_i32_b32 edge cases
        let mut cwords = Vec::new();
        cwords.push(enc_sop1(0x08, 2, 128 + 0)); // s_not_b64 s[2:3], 0 -> (0xFFFFFFFF, 0xFFFFFFFF), scc=true
        cwords.push(enc_sop2(0x0A, 4, 128 + 1, 128 + 0)); // s4 = scc ? 1 : 0 -> 1
        cwords.push(enc_sop1(0x08, 6, 193)); // s_not_b64 s[6:7], -1 -> (0, 0), scc=false
        cwords.push(enc_sop2(0x0A, 8, 128 + 1, 128 + 0)); // s8 = scc ? 1 : 0 -> 0
        cwords.push(enc_sop1(0x13, 10, 128 + 0)); // s_ff1_i32_b32 s10, 0 -> 0xFFFFFFFF
        cwords.push(enc_sop1(0x13, 11, 128 + 1)); // s_ff1_i32_b32 s11, 1 -> 0
        cwords.push(enc_sop1(0x13, 12, 0xFF)); // s_ff1_i32_b32 s12, 0x80000000 -> 31
        cwords.push(0x8000_0000);
        cwords.push(enc_sop1(0x13, 13, 0xFF)); // s_ff1_i32_b32 s13, 0x00010000 -> 16
        cwords.push(0x0001_0000);
        cwords.push(enc_sop1(0x13, 14, 0xFF)); // s_ff1_i32_b32 s14, 0x000000F0 -> 4
        cwords.push(0x0000_00F0);
        cwords.push(enc_sopp(0x01, 0));

        let cout = translate_as(&cwords, Rdna2StageOverride::Compute);
        assert_eq!(cout.unsupported_op_count, 0, "WGSL:\n{}", cout.wgsl);
        assert_eq!(last_assignment(&cout.wgsl, "s_2_").as_deref(), Some("4294967295u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_3_").as_deref(), Some("4294967295u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_6_").as_deref(), Some("0u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_7_").as_deref(), Some("0u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_10_").as_deref(), Some("4294967295u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_11_").as_deref(), Some("0u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_12_").as_deref(), Some("31u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_13_").as_deref(), Some("16u"));
        assert_eq!(last_assignment(&cout.wgsl, "s_14_").as_deref(), Some("4u"));
        assert_naga_wgsl_valid(&cout.wgsl);
    }

    #[test]
    fn s39_translates_v_cmp_class_f16_all_10_ieee754_classes_op_sel_and_16bit_compares() {
        // Part 1: Verify exact constant-folded IEEE-754 binary16 classification for all 10 classes:
        //   0 sNaN, 1 qNaN, 2 -Inf, 3 -Normal, 4 -Subnormal, 5 -0, 6 +0, 7 +Subnormal, 8 +Normal, 9 +Inf
        let cases: &[(u32, u32)] = &[
            (0x7C01, 0), // sNaN
            (0x7E00, 1), // qNaN
            (0xFC00, 2), // -Inf
            (0xBC00, 3), // -1.0 (-Normal)
            (0x8001, 4), // -Subnormal
            (0x8000, 5), // -0.0
            (0x0000, 6), // +0.0
            (0x0001, 7), // +Subnormal
            (0x3C00, 8), // +1.0 (+Normal)
            (0x7C00, 9), // +Inf
        ];
        for &(hbits, expected_class) in cases {
            let match_mask = 1u32 << expected_class;
            let miss_mask = 0x3FFu32 ^ match_mask;
            // Test low half (op_sel = 0) with garbage in upper 16 bits, and high half (op_sel = 1)
            let packed_lo = 0xFFFF_0000 | hbits;
            let packed_hi = (hbits << 16) | 0x3C00; // low is +1.0, high is target
            let mut words = Vec::new();
            // v_cmp_class_f16 s[2:3], lit(packed_lo), match_mask (inline when <= 64; runtime lit mask otherwise)
            if match_mask <= 64 {
                words.extend_from_slice(&enc_vop3_opsel(
                    0x08F,
                    2,
                    0xFF,
                    128 + match_mask as u16,
                    0,
                    0,
                    false,
                    0,
                ));
                words.push(packed_lo);
                // op_sel[0]=1 selects high half
                words.extend_from_slice(&enc_vop3_opsel(
                    0x08F,
                    4,
                    0xFF,
                    128 + match_mask as u16,
                    0,
                    1,
                    false,
                    0,
                ));
                words.push(packed_hi);
                words.push(enc_sopp(0x01, 0));
                let out = translate_as(&words, Rdna2StageOverride::Compute);
                assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
                assert_eq!(
                    last_assignment(&out.wgsl, "s_2_").as_deref(),
                    Some("1u"),
                    "f16 0x{hbits:04x} expected class {expected_class}:\n{}",
                    out.wgsl
                );
                assert_eq!(
                    last_assignment(&out.wgsl, "s_4_").as_deref(),
                    Some("1u"),
                    "f16 hi-half 0x{hbits:04x} expected class {expected_class}:\n{}",
                    out.wgsl
                );
                assert_naga_wgsl_valid(&out.wgsl);
            } else {
                // Verify runtime lowering with literal mask (`match_mask` and `miss_mask`)
                words.push(enc_sop1(0x03, 10, 0xFF));
                words.push(packed_lo);
                words.extend_from_slice(&enc_vop3_opsel(0x08F, 2, 10, 0xFF, 0, 0, false, 0));
                words.push(match_mask);
                words.extend_from_slice(&enc_vop3_opsel(0x08F, 4, 10, 0xFF, 0, 0, false, 0));
                words.push(miss_mask);
                words.push(enc_sopp(0x01, 0));
                let out = translate_as(&words, Rdna2StageOverride::Compute);
                assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
                assert_naga_wgsl_valid(&out.wgsl);
            }
        }

        // Verify abs/neg modifiers on v_cmp_class_f16: |-1.0| -> +Normal (bit 8), -(1.0) -> -Normal (bit 3)
        let mut mod_words = Vec::new();
        // v_cmp_class_f16 s[2:3], -1.0 (inline 243) with abs=1, mask=0x100 (+Normal) -> 1u
        mod_words.extend_from_slice(&enc_vop3(0x08F, 2, 243, 0xFF, 0, false, 0, 0b01, 0));
        mod_words.push(1u32 << 8);
        // v_cmp_class_f16 s[4:5], 1.0 (inline 242) with neg=1, mask=0x008 (-Normal) -> 1u
        mod_words.extend_from_slice(&enc_vop3(0x08F, 4, 242, 128 + 8, 0, false, 0, 0, 0b01));
        mod_words.push(enc_sopp(0x01, 0));
        let mod_out = translate_as(&mod_words, Rdna2StageOverride::Compute);
        assert_eq!(mod_out.unsupported_op_count, 0, "WGSL:\n{}", mod_out.wgsl);
        assert_eq!(last_assignment(&mod_out.wgsl, "s_2_").as_deref(), Some("1u"));
        assert_eq!(last_assignment(&mod_out.wgsl, "s_4_").as_deref(), Some("1u"));
        assert_naga_wgsl_valid(&mod_out.wgsl);

        // Part 2: Runtime fragment shader exercising VOPC/VOP3 v_cmp_class_f16, v_cmpx_class_f16,
        // and GFX10.3 16-bit VOPC compares (0xC9, 0xCA, 0xCC, 0xCE, 0xD9, 0xE8, 0x89, 0xA9)
        let mut rt = Vec::new();
        rt.push(enc_vintrp(0, 0, 0, 0, 0));
        rt.push(enc_vintrp(0, 1, 0, 1, 0));
        rt.push(enc_vopc(0x8F, 256 + 0, 1)); // v_cmp_class_f16 vcc, v0, v1
        rt.push(enc_vopc(0x9F, 256 + 0, 1)); // v_cmpx_class_f16 exec, v0, v1
        rt.extend_from_slice(&enc_vop3_opsel(0x08F, 4, 256 + 0, 128 + 3, 0, 1, false, 0)); // op_sel hi
        rt.push(enc_vopc(0xC9, 256 + 0, 1)); // v_cmp_lt_f16
        rt.push(enc_vopc(0xCA, 256 + 0, 1)); // v_cmp_eq_f16
        rt.push(enc_vopc(0xCC, 256 + 0, 1)); // v_cmp_gt_f16
        rt.push(enc_vopc(0xCE, 256 + 0, 1)); // v_cmp_ge_f16
        rt.push(enc_vopc(0xD9, 256 + 0, 1)); // v_cmpx_lt_f16
        rt.push(enc_vopc(0xE8, 256 + 0, 1)); // v_cmp_u_f16
        rt.push(enc_vopc(0x89, 256 + 0, 1)); // v_cmp_lt_i16
        rt.push(enc_vopc(0xA9, 256 + 0, 1)); // v_cmp_lt_u16
        rt.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        rt.push(enc_sopp(0x01, 0));

        let rt_out = translate_as(&rt, Rdna2StageOverride::Pixel);
        assert!(rt_out.naga_validated);
        assert_eq!(rt_out.unsupported_op_count, 0, "WGSL:\n{}", rt_out.wgsl);
        assert!(rt_out.wgsl.contains("var vcc: bool;"), "WGSL:\n{}", rt_out.wgsl);
        assert!(
            rt_out.wgsl.contains("var exec_mask: u32;"),
            "WGSL:\n{}",
            rt_out.wgsl
        );
        assert!(rt_out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", rt_out.wgsl);
        assert_naga_wgsl_valid(&rt_out.wgsl);
    }

    #[test]
    fn s39_translates_v_xad_u32_v_or3_b32_0x372_v_xor3_b32_and_cubemap_vop3_ops() {
        // Part 1: Constant-folded exact verification of v_xad_u32, v_or3_b32 (0x372), v_xor3_b32 (0x178)
        let mut cwords = Vec::new();
        // v_xad_u32 v1, 15, 6, 10 => (15 ^ 6) + 10 = 9 + 10 = 19
        cwords.extend_from_slice(&enc_vop3(
            0x345,
            1,
            128 + 15,
            128 + 6,
            128 + 10,
            false,
            0,
            0,
            0,
        ));
        // v_or3_b32 (0x372) v2, 1, 4, 16 => 1 | 4 | 16 = 21
        cwords.extend_from_slice(&enc_vop3(
            0x372,
            2,
            128 + 1,
            128 + 4,
            128 + 16,
            false,
            0,
            0,
            0,
        ));
        // v_xor3_b32 (0x178) v3, 7, 12, 5 => 7 ^ 12 ^ 5 = 14
        cwords.extend_from_slice(&enc_vop3(
            0x178,
            3,
            128 + 7,
            128 + 12,
            128 + 5,
            false,
            0,
            0,
            0,
        ));
        cwords.push(enc_sopp(0x01, 0));

        let cout = translate_as(&cwords, Rdna2StageOverride::Compute);
        assert_eq!(cout.unsupported_op_count, 0, "WGSL:\n{}", cout.wgsl);
        assert!(
            cout.wgsl.contains("19u") && cout.wgsl.contains("21u") && cout.wgsl.contains("14u"),
            "expected folded results 19u, 21u, 14u:\n{}",
            cout.wgsl
        );
        assert_naga_wgsl_valid(&cout.wgsl);

        // Part 2: Runtime fragment shader exercising all 4 cubemap VOP3 ops + bitwise ternary ops
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 0, 0, 0, 0)); // v0 = dir.x
        words.push(enc_vintrp(0, 1, 0, 1, 0)); // v1 = dir.y
        words.push(enc_vintrp(0, 2, 0, 2, 0)); // v2 = dir.z
        words.extend_from_slice(&enc_vop3(0x144, 4, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0)); // v_cubeid_f32
        words.extend_from_slice(&enc_vop3(0x145, 5, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0)); // v_cubesc_f32
        words.extend_from_slice(&enc_vop3(0x146, 6, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0)); // v_cubetc_f32
        words.extend_from_slice(&enc_vop3(0x147, 7, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0)); // v_cubema_f32
        // Also exercise modifiers (abs/neg/clamp) on v_cubema_f32 and runtime v_xad_u32/v_or3_b32/v_xor3_b32
        words.extend_from_slice(&enc_vop3(
            0x147,
            8,
            256 + 0,
            256 + 1,
            256 + 2,
            true,
            0,
            0b001,
            0b010,
        ));
        words.extend_from_slice(&enc_vop3(0x345, 9, 256 + 4, 256 + 5, 256 + 6, false, 0, 0, 0)); // v_xad_u32
        words.extend_from_slice(&enc_vop3(0x372, 10, 256 + 4, 256 + 5, 256 + 6, false, 0, 0, 0)); // v_or3_b32
        words.extend_from_slice(&enc_vop3(0x178, 11, 256 + 4, 256 + 5, 256 + 6, false, 0, 0, 0)); // v_xor3_b32
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.extend_from_slice(&enc_exp(1, 0xF, false, true, true, [8, 9, 10, 11]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("abs("), "cubemap ops must compute abs(x/y/z):\n{}", out.wgsl);
        assert!(out.wgsl.contains("select("), "cubemap ops must select face/axes:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    // -- S40 (#890): SGPR descriptor-source propagation fixes --

    #[test]
    fn s40_propagates_64bit_add_addc_or_lshl_add_and_const_sgpr_soffset_table_offsets() {
        // Fragment shader deriving multiple T#/S#/CBV descriptors from root table s8 via:
        //   - s_add_u32 + s_addc_u32 (accumulating +32, then +32 -> +64)
        //   - s_or_b32 (LLVM `or disjoint`: s8 | 32 -> +32)
        //   - s_lshl4_add_u32 (compact `(6 << 4) + s8` -> +96, `(8 << 4) + s8` -> +128)
        //   - s_movk_i32 + s_load_dwordx4 with constant-SGPR `soffset` (+160)
        //   - s_mov_b64 copy of derived 64-bit pointer pair
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0)); // v2 = attr0.x
        words.push(enc_vintrp(0, 3, 0, 1, 0)); // v3 = attr0.y
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0

        // 1. s_or_b32 s10, s8, 32 -> s8 + 32; s_mov_b32 s11, 0 -> T#0 @ s8 + 32
        words.push(enc_sop2(0x10, 10, 8, 128 + 32)); // s_or_b32 s10, s8, 32
        words.push(enc_sop1(0x03, 11, 0x80)); // s11 = 0
        words.extend_from_slice(&enc_smem(0x03, 24, 5, 0x7D, 0x0, false, false)); // T#0 @ s8+32 -> s[24:31]

        // 2. s_add_u32 + s_addc_u32 twice: s[10:11] + 32 -> s[12:13] (+64), then s_mov_b64 -> s[14:15]
        words.push(enc_sop2(0x00, 12, 10, 128 + 32)); // s_add_u32 s12, s10, 32 -> s8 + 64
        words.push(enc_sop2(0x04, 13, 11, 128 + 0)); // s_addc_u32 s13, s11, 0
        words.push(enc_sop1(0x04, 14, 12)); // s_mov_b64 s[14:15], s[12:13]
        words.extend_from_slice(&enc_smem(0x03, 32, 7, 0x7D, 0x0, false, false)); // T#1 @ s8+64 -> s[32:39]

        // 3. s_lshl4_add_u32 s10, 6, s8 -> (6 << 4) + s8 = s8 + 96 -> T#2 @ s8 + 96
        words.push(enc_sop2(0x31, 10, 128 + 6, 8)); // s_lshl4_add_u32 s10, 6, s8
        words.push(enc_sop1(0x03, 11, 0x80));
        words.extend_from_slice(&enc_smem(0x03, 40, 5, 0x7D, 0x0, false, false)); // T#2 @ s8+96 -> s[40:47]

        // 4. s_lshl4_add_u32 s10, 8, s8 -> (8 << 4) + s8 = s8 + 128 -> CBV V# @ s8 + 128
        words.push(enc_sop2(0x31, 10, 128 + 8, 8)); // s_lshl4_add_u32 s10, 8, s8
        words.push(enc_sop1(0x03, 11, 0x80));
        words.extend_from_slice(&enc_smem(0x02, 16, 5, 0x7D, 0x0, false, false)); // CBV V# @ s8+128 -> s[16:19]
        words.extend_from_slice(&enc_smem(0x0A, 4, 8, 0x7D, 0x0, false, false)); // s_buffer_load_dwordx4 s[4:7], s[16:19], 0

        // 5. Constant-SGPR soffset via s_movk_i32 s10, 160 -> S# @ s8 + 160
        words.push(enc_sopk(0x00, 10, 160)); // s_movk_i32 s10, 160
        words.extend_from_slice(&enc_smem(0x02, 20, 4, 10, 0x0, false, false)); // S# @ s8+160 -> s[20:23]

        // Sample all 3 textures with S# @ s8+160 and combine with CBV s4
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 6, 5)); // srsrc=6 (s24), ssamp=5 (s20)
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 8, 8, 5)); // srsrc=8 (s32), ssamp=5 (s20)
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 12, 10, 5)); // srsrc=10 (s40), ssamp=5 (s20)
        words.push(enc_vop2(0x03, 4, 256 + 4, 8));
        words.push(enc_vop2(0x03, 4, 256 + 4, 12));
        words.push(enc_vop2(0x03, 4, 4, 4)); // v4 += s4
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.source.is_some()),
            "every binding must have a reflected source: {:?}",
            out.bindings
        );
        let cbv = out.bindings.iter().find(|b| b.kind == "uniform").unwrap();
        assert_eq!(
            cbv.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 128,
            })
        );
        let tex_offsets: Vec<u32> = out
            .bindings
            .iter()
            .filter(|b| b.kind == "texture_2d")
            .map(|b| match b.source {
                Some(WgslBindingSource::Table {
                    user_sgpr: 8,
                    table_byte_offset,
                }) => table_byte_offset,
                other => panic!("unexpected tex source: {other:?}"),
            })
            .collect();
        assert_eq!(tex_offsets, vec![32, 64, 96]);
        let samp = out.bindings.iter().find(|b| b.kind == "sampler").unwrap();
        assert_eq!(
            samp.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 160,
            })
        );
        // All bindings reflected -> fragment offset +32 applies.
        assert!(out.bindings.iter().all(|b| b.binding >= FRAGMENT_REFLECTED_BINDING_OFFSET));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s40_propagates_nested_2level_table_pointers_and_cbv_pointers_without_spurious_cbv() {
        // Vertex shader exercising 2-level `s_load_dwordx2` indirections from user table s8:
        //   1. `s_load_dwordx2 s[12:13], s[8:9], 0x20` -> nested table ptr -> `s_load_dwordx8 s[16:23], s[12:13], 0x40` (T# @ s8 + 0x60)
        //   2. `s_load_dwordx2 s[14:15], s[8:9], 0x80` -> 64-bit CBV ptr -> two `s_load_dwordx4` CBV reads (must coalesce into 1 CBV @ s8 + 0x80, 32B!)
        //   3. `s_load_dwordx4 s[24:27], s[8:9], 0xA0` -> copied via `s_mov_b64` to s[28:31] -> `s_buffer_load_dwordx4` (CBV @ s8 + 0xA0)
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0

        // Nested table ptr at s8 + 0x20 -> T# at +0x40 => effective table_byte_offset = 0x60
        words.extend_from_slice(&enc_smem(0x01, 12, 4, 0x7D, 0x20, false, false)); // s_load_dwordx2 s[12:13], s[8:9], 0x20
        words.extend_from_slice(&enc_smem(0x03, 16, 6, 0x7D, 0x40, false, false)); // s_load_dwordx8 s[16:23], s[12:13], 0x40

        // 64-bit CBV pointer at s8 + 0x80 -> two CBV data loads at +0x00 and +0x10
        words.extend_from_slice(&enc_smem(0x01, 14, 4, 0x7D, 0x80, false, false)); // s_load_dwordx2 s[14:15], s[8:9], 0x80
        words.extend_from_slice(&enc_smem(0x02, 4, 7, 0x7D, 0x00, false, false)); // cb0[0]
        words.extend_from_slice(&enc_smem(0x02, 32, 7, 0x7D, 0x10, false, false)); // cb0[1]

        // V# loaded at s8 + 0xA0 into s[24:27], copied via s_mov_b64 into s[28:31], consumed by s_buffer_load_dwordx4
        words.extend_from_slice(&enc_smem(0x02, 24, 4, 0x7D, 0xA0, false, false)); // V# @ s8+0xA0
        words.push(enc_sop1(0x04, 28, 24)); // s_mov_b64 s[28:29], s[24:25]
        words.push(enc_sop1(0x04, 30, 26)); // s_mov_b64 s[30:31], s[26:27]
        words.extend_from_slice(&enc_smem(0x0A, 36, 14, 0x7D, 0x00, false, false)); // s_buffer_load_dwordx4 s[36:39], s[28:31], 0

        // image_load from T# s[16:23] + combine with CBV values
        words.extend_from_slice(&enc_mimg(0x00, 0xF, 1, 0, 4, 4, 0)); // image_load srsrc=4 (s16)
        words.push(enc_vop2(0x03, 4, 4, 4)); // v4 += s4
        words.push(enc_vop2(0x03, 5, 32, 5)); // v5 += s32
        words.push(enc_vop2(0x03, 6, 36, 6)); // v6 += s36
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Vertex);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.source.is_some()),
            "every binding must have a reflected source: {:?}",
            out.bindings
        );

        let cbvs: Vec<_> = out.bindings.iter().filter(|b| b.kind == "uniform").collect();
        assert_eq!(
            cbvs.len(),
            2,
            "nested pointer/V# loads must NOT fabricate extra CBVs: {:?}",
            out.bindings
        );
        assert_eq!(
            cbvs[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0x80,
            })
        );
        assert_eq!(cbvs[0].min_binding_size_bytes, Some(32));
        assert_eq!(
            cbvs[1].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0xA0,
            })
        );
        assert_eq!(cbvs[1].min_binding_size_bytes, Some(16));

        let tex = out.bindings.iter().find(|b| b.kind == "texture_2d").unwrap();
        assert_eq!(
            tex.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0x60,
            })
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s40_propagates_and_masking_movk_mul_heap_indexing_and_lshl64_heap_sources() {
        // Compute shader exercising:
        //   1. Table address masked with `s_and_b32 ..., 0xFFFFFFFC` & `s_andn2_b32 ..., 3` -> T#0 @ s4 + 32
        //   2. SM 6.6 ResourceHeap via `s_movk_i32` stride 32 + `s_mul_i32` + `s_add_u32` + `s_addc_u32` -> T#1
        //   3. SM 6.6 SamplerHeap via 64-bit `s_lshl_b64` (shift 4) + `s_add_u32` + `s_addc_u32` -> S#0
        //   4. SM 6.6 ResourceHeap via shift-add non-pow2 stride (`s_lshl1_add_u32` = 3x, `s_lshl_b32` << 3 = 24x) -> sbuf0
        let mut words = Vec::new();

        // 1. Masking s4 with `s_and_b32 s12, s4, 0xFFFFFFFC` then `s_andn2_b32 s12, s12, 3`
        words.push(enc_sop2(0x0E, 12, 4, 0xFF)); // s_and_b32 s12, s4, 0xFFFFFFFC
        words.push(0xFFFF_FFFC);
        words.push(enc_sop2(0x14, 12, 12, 128 + 3)); // s_andn2_b32 s12, s12, 3
        words.push(enc_sop1(0x03, 13, 0x80)); // s13 = 0
        words.extend_from_slice(&enc_smem(0x03, 24, 6, 0x7D, 0x20, false, false)); // T#0 @ s4 + 32 -> s[24:31]

        // 2. ResourceHeap via s_movk_i32 s14, 32; s_mul_i32 s14, s6, s14; s_add_u32 s16, s5, s14; s_addc_u32 s17, 0, 0
        words.push(enc_sopk(0x00, 14, 32)); // s_movk_i32 s14, 32
        words.push(enc_sop2(0x26, 14, 6, 14)); // s_mul_i32 s14, s6, s14 (stride 32, shift 5)
        words.push(enc_sop2(0x00, 16, 5, 14)); // s_add_u32 s16, s5, s14
        words.push(enc_sop2(0x04, 17, 128 + 0, 128 + 0)); // s_addc_u32 s17, 0, 0
        words.extend_from_slice(&enc_smem(0x03, 32, 8, 0x7D, 0x0, false, false)); // T#1 -> s[32:39]

        // 3. SamplerHeap via 64-bit shift: s[14:15] = (s8, 0); s_lshl_b64 s[14:15], s[14:15], 4; s_add_u32 s16, s7, s14; s_addc_u32 s17, 0, s15
        words.push(enc_sop1(0x03, 14, 8)); // s14 = s8 (sampler index)
        words.push(enc_sop1(0x03, 15, 0x80)); // s15 = 0
        words.push(enc_sop2(0x1F, 14, 14, 128 + 4)); // s_lshl_b64 s[14:15], s[14:15], 4
        words.push(enc_sop2(0x00, 16, 7, 14)); // s_add_u32 s16, s7, s14 (base s7)
        words.push(enc_sop2(0x04, 17, 128 + 0, 15)); // s_addc_u32 s17, 0, s15
        words.extend_from_slice(&enc_smem(0x02, 20, 8, 0x7D, 0x0, false, false)); // S#0 -> s[20:23]

        // 4. Non-pow2 stride 24 via shift-add: s_lshl1_add_u32 s14, s9, s9 (3*s9); s_lshl_b32 s14, s14, 3 (24*s9)
        words.push(enc_sop2(0x2E, 14, 9, 9)); // s_lshl1_add_u32 s14, s9, s9
        words.push(enc_sop2(0x1E, 14, 14, 128 + 3)); // s_lshl_b32 s14, s14, 3 (stride 24, shift 0)
        words.push(enc_sop1(0x03, 16, 5)); // s16 = s5 (resource heap base)
        words.push(enc_sop1(0x03, 17, 0x80)); // s17 = 0
        words.extend_from_slice(&enc_smem(0x02, 40, 8, 14, 0x0, false, false)); // V# -> s[40:43]

        // Consume all descriptors
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 0, 4, 6, 5)); // T#0 + S#0
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 0, 8, 8, 5)); // T#1 + S#0
        let ld_w0 = (0b111000u32 << 26) | (0x0Eu32 << 18) | (1u32 << 12);
        let ld_w1 = (0x80u32 << 24) | (10u32 << 16) | (12u32 << 8); // srsrc=10 (s[40:43])
        words.extend_from_slice(&[ld_w0, ld_w1]);
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.kind != "uniform"),
            "no spurious CBVs expected: {:?}",
            out.bindings
        );
        assert!(
            out.bindings.iter().all(|b| b.source.is_some()),
            "every binding must have a source: {:?}",
            out.bindings
        );

        let texs: Vec<_> = out.bindings.iter().filter(|b| b.kind == "texture_2d").collect();
        assert_eq!(texs.len(), 2, "bindings: {:?}", out.bindings);
        assert_eq!(
            texs[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 32,
            })
        );
        assert_eq!(
            texs[1].source,
            Some(WgslBindingSource::ResourceHeap {
                base_user_sgpr: 5,
                stride: 32,
                index_user_sgpr: 6,
                index_shift: 5,
            })
        );

        let samp = out.bindings.iter().find(|b| b.kind == "sampler").unwrap();
        assert_eq!(
            samp.source,
            Some(WgslBindingSource::SamplerHeap {
                base_user_sgpr: 7,
                stride: 16,
                index_user_sgpr: 8,
                index_shift: 4,
            })
        );

        let sbuf = out
            .bindings
            .iter()
            .find(|b| b.kind == "read-only-storage")
            .unwrap();
        assert_eq!(
            sbuf.source,
            Some(WgslBindingSource::ResourceHeap {
                base_user_sgpr: 5,
                stride: 24,
                index_user_sgpr: 9,
                index_shift: 0,
            })
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s40_propagates_cmov_cselect_multi_lane_descriptor_spills_and_branch_isolation() {
        // Fragment shader testing:
        //   1. `s_cselect_b32` with null fallback (`s8` vs `0`) & `s_cmov_b32` with same origin
        //   2. Full 256-bit T# (8 lanes) & 128-bit V# CBV (4 lanes) spilled via `v_writelane_b32`,
        //      original SGPRs clobbered, then reloaded via `v_readlane_b32`
        //   3. Uniform VGPR spill (`v_mov_b32` + `v_readfirstlane_b32`) for a sampler table pointer
        //   4. If/else branch isolation: `then` branch clobbers `s8` before `s_branch MERGE`,
        //      while `else` branch still sees intact `s8` (UserTable 8) and loads T#1!
        //   5. Inline static sampler built via `s_mov_b32` + `s_or_b32` + `s_bitset1_b32`
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0)); // v2 = attr0.x
        words.push(enc_vintrp(0, 3, 0, 1, 0)); // v3 = attr0.y
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0

        // 1. s_cselect_b32 s10, s8, 0 (known table origin + null fallback -> UserTable(8, 0))
        words.push(enc_sopc(0x06, 8, 128 + 0)); // sets scc
        words.push(enc_sop2(0x0A, 10, 8, 128 + 0)); // s_cselect_b32 s10, s8, 0
        words.push(enc_sop1(0x05, 10, 8)); // s_cmov_b32 s10, s8 (both branches UserTable(8, 0))
        words.push(enc_sop1(0x03, 11, 0x80));

        // 2. Load T#0 @ s8+0x20 into s[16:23] and CBV V# @ s8+0x40 into s[12:15], spill all 12 dwords to v40[0..11]
        words.extend_from_slice(&enc_smem(0x03, 16, 5, 0x7D, 0x20, false, false)); // T#0 @ s8+0x20
        words.extend_from_slice(&enc_smem(0x02, 12, 5, 0x7D, 0x40, false, false)); // V# @ s8+0x40
        for lane in 0..8u16 {
            words.extend_from_slice(&enc_vop3(
                0x361,
                40,
                16 + lane,
                128 + lane,
                0,
                false,
                0,
                0,
                0,
            ));
        }
        for lane in 0..4u16 {
            words.extend_from_slice(&enc_vop3(
                0x361,
                40,
                12 + lane,
                128 + 8 + lane,
                0,
                false,
                0,
                0,
                0,
            ));
        }
        // Uniform VGPR spill of table ptr s10 via v_mov_b32 v41, s10 -> v_readfirstlane_b32 s10, v41
        words.push(enc_vop1(0x01, 41, 10)); // v_mov_b32 v41, s10
        // Clobber s10, s12..s23
        words.push(enc_sop1(0x03, 10, 128 + 0));
        for r in 12..=23u8 {
            words.push(enc_sop1(0x03, r, 128 + 0));
        }
        // Reload T#0 into s[24:31] and CBV V# into s[32:35] from v40[0..11]
        for lane in 0..8u16 {
            words.extend_from_slice(&enc_vop3(
                0x360,
                24 + lane as u8,
                256 + 40,
                128 + lane,
                0,
                false,
                0,
                0,
                0,
            ));
        }
        for lane in 0..4u16 {
            words.extend_from_slice(&enc_vop3(
                0x360,
                32 + lane as u8,
                256 + 40,
                128 + 8 + lane,
                0,
                false,
                0,
                0,
                0,
            ));
        }
        // Reload table ptr from v41 via v_readfirstlane_b32 and load S#0 @ s8+0x60 into s[36:39]
        words.push(enc_vop1(0x02, 10, 256 + 41)); // v_readfirstlane_b32 s10, v41
        words.push(enc_sop1(0x03, 11, 0x80));
        words.extend_from_slice(&enc_smem(0x02, 36, 5, 0x7D, 0x60, false, false)); // S#0 @ s8+0x60

        // Consume reloaded CBV V# s[32:35] and T#0 s[24:31] + S#0 s[36:39]
        words.extend_from_slice(&enc_smem(0x0A, 4, 16, 0x7D, 0x0, false, false)); // s_buffer_load_dwordx4 s[4:7], s[32:35], 0
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 4, 6, 9)); // srsrc=6 (s24), ssamp=9 (s36)

        // 5. Inline static sampler built with s_mov_b32 + s_or_b32 + s_bitset1_b32 in s[40:43]
        words.push(enc_sop1(0x03, 40, 128 + 1));
        words.push(enc_sop2(0x10, 40, 40, 128 + 4)); // s_or_b32 s40, s40, 4
        words.push(enc_sop1(0x1D, 40, 128 + 8)); // s_bitset1_b32 s40, 8
        words.push(enc_sop1(0x03, 41, 128 + 0));
        words.push(enc_sop1(0x03, 42, 128 + 0));
        words.push(enc_sop1(0x03, 43, 128 + 0));

        // 4. Branch diamond: `then` clobbers s8 then jumps to MERGE; `else` loads T#1 @ s8+0x80!
        let branch_pos = words.len() * 4;
        // Placeholder for s_cbranch_scc0 ELSE (patched below)
        let cbranch_idx = words.len();
        words.push(0);
        // THEN block: clobber s8 -> 0, branch to MERGE
        words.push(enc_sop1(0x03, 8, 128 + 0)); // clobbers s8 on THEN path only!
        let sbranch_idx = words.len();
        words.push(0); // placeholder for s_branch MERGE
        // ELSE block: s8 MUST still be UserTable(8, 0)!
        let else_off = words.len() * 4;
        words.extend_from_slice(&enc_smem(0x03, 48, 4, 0x7D, 0x80, false, false)); // T#1 @ s8+0x80 -> s[48:55]
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 8, 12, 10)); // srsrc=12 (s48), ssamp=10 (s40 static)
        words.push(enc_vop2(0x03, 4, 256 + 4, 8));
        // MERGE block:
        let merge_off = words.len() * 4;
        words[cbranch_idx] = enc_sopp(0x04, branch_simm(branch_pos, else_off));
        words[sbranch_idx] = enc_sopp(0x02, branch_simm(sbranch_idx * 4, merge_off));

        words.push(enc_vop2(0x03, 4, 4, 4)); // v4 += s4
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.source.is_some()),
            "every binding must have a reflected source: {:?}",
            out.bindings
        );

        let cbvs: Vec<_> = out.bindings.iter().filter(|b| b.kind == "uniform").collect();
        assert_eq!(cbvs.len(), 1, "spilled V# must not create extra CBV: {:?}", out.bindings);
        assert_eq!(
            cbvs[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 8,
                table_byte_offset: 0x40,
            })
        );

        let tex_offsets: Vec<u32> = out
            .bindings
            .iter()
            .filter(|b| b.kind == "texture_2d")
            .map(|b| match b.source {
                Some(WgslBindingSource::Table {
                    user_sgpr: 8,
                    table_byte_offset,
                }) => table_byte_offset,
                other => panic!("unexpected tex source: {other:?}"),
            })
            .collect();
        assert_eq!(tex_offsets, vec![0x20, 0x80]);

        let samps: Vec<_> = out.bindings.iter().filter(|b| b.kind == "sampler").collect();
        assert_eq!(samps.len(), 2, "bindings: {:?}", out.bindings);
        assert!(samps.iter().any(|s| {
            s.source
                == Some(WgslBindingSource::Table {
                    user_sgpr: 8,
                    table_byte_offset: 0x60,
                })
        }));
        assert!(samps.iter().any(|s| {
            s.source
                == Some(WgslBindingSource::InlineSampler {
                    sdwords: [0x105, 0, 0, 0],
                })
        }));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s40_never_fabricates_source_on_conflicting_select_or_loop_varying_pointer() {
        // Conflicting `s_cselect_b32` between two DIFFERENT table offsets (s8+32 vs s8+64) must
        // stay `source: None` (never arbitrarily pick one!).
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_sop2(0x00, 10, 8, 128 + 32)); // s10 = s8 + 32
        words.push(enc_sop2(0x00, 12, 8, 128 + 64)); // s12 = s8 + 64
        words.push(enc_sopc(0x06, 8, 128 + 0));
        words.push(enc_sop2(0x0A, 14, 10, 12)); // s_cselect_b32 s14, s10, s12 -> CONFLICT -> Unknown!
        words.push(enc_sop1(0x03, 15, 0x80));
        words.extend_from_slice(&enc_smem(0x03, 24, 7, 0x7D, 0x0, false, false)); // T# from ambiguous s[14:15]
        words.extend_from_slice(&enc_mimg(0x00, 0xF, 1, 2, 4, 6, 0)); // image_load
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        let tex = out.bindings.iter().find(|b| b.kind == "texture_2d").unwrap();
        assert_eq!(
            tex.source, None,
            "conflicting s_cselect_b32 between two different offsets must NOT fabricate a source"
        );
        assert_naga_wgsl_valid(&out.wgsl);

        // Loop-varying table pointer (`s12 = s12 + 32` inside loop) must converge to `source: None`!
        let mut lwords = Vec::new();
        lwords.push(enc_sop1(0x03, 12, 8)); // s12 = s8 + 0
        lwords.push(enc_sop1(0x03, 13, 0x80));
        let loop_off = lwords.len() * 4;
        lwords.extend_from_slice(&enc_smem(0x03, 24, 6, 0x7D, 0x0, false, false)); // T# from loop-varying s[12:13]
        lwords.extend_from_slice(&enc_mimg(0x00, 0xF, 1, 0, 4, 6, 0)); // image_load
        lwords.push(enc_sop2(0x00, 12, 12, 128 + 32)); // s12 += 32 each iteration!
        lwords.push(enc_sopc(0x0A, 12, 128 + 64));
        let back_pos = lwords.len() * 4;
        lwords.push(enc_sopp(0x05, branch_simm(back_pos, loop_off))); // s_cbranch_scc1 LOOP
        lwords.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        lwords.push(enc_sopp(0x01, 0));

        let lout = translate_as(&lwords, Rdna2StageOverride::Pixel);
        assert!(lout.naga_validated);
        assert_eq!(lout.unsupported_op_count, 0, "WGSL:\n{}", lout.wgsl);
        let ltex = lout.bindings.iter().find(|b| b.kind == "texture_2d").unwrap();
        assert_eq!(
            ltex.source, None,
            "loop-varying table pointer must NOT fabricate a fixed static offset"
        );
        assert_naga_wgsl_valid(&lout.wgsl);
    }

    #[test]
    fn s40_synthesized_root_descriptor_vsharp_for_mubuf_emits_root_descriptor_source() {
        // Compute shader synthesizing a 4-dword V# from 64-bit Root SRV GPUVA in s[4:5]
        // (`s12 = s4; s13 = s5; s14 = 0xFFFFFFFF; s15 = 0x31014000`) and loading via MUBUF.
        let mut words = Vec::new();
        words.push(enc_sop1(0x04, 12, 4)); // s_mov_b64 s[12:13], s[4:5]
        words.push(enc_sop1(0x03, 14, 193)); // s_mov_b32 s14, -1 (num_records)
        words.push(enc_sop1(0x03, 15, 0xFF)); // s_mov_b32 s15, 0x31014000 (V# word 3)
        words.push(0x3101_4000);
        // buffer_load_dwordx4 v[4:7], v0, s[12:15], 0 offen (srsrc=3)
        let ld_w0 = (0b111000u32 << 26) | (0x0Eu32 << 18) | (1u32 << 12);
        let ld_w1 = (0x80u32 << 24) | (3u32 << 16) | (4u32 << 8);
        words.extend_from_slice(&[ld_w0, ld_w1]);
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        let sbuf = out
            .bindings
            .iter()
            .find(|b| b.kind == "read-only-storage")
            .expect("expected read-only-storage binding");
        assert_eq!(
            sbuf.source,
            Some(WgslBindingSource::RootDescriptor { user_sgpr: 4 })
        );
        let json = out.to_json();
        assert!(
            json.contains("\"source\": {\"user_sgpr\": 4, \"root_descriptor\": true}"),
            "JSON:\n{json}"
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    // -- S41 (#917): Fix S40 vertex stage reflection regression --

    #[test]
    fn s41_vertex_split_s_mov_b64_self_overwrite_tainted_vsharp_and_root_descriptor_no_none_source()
    {
        // Reproduces the exact GFX10.3 vertex shader pattern from Issue #917:
        //   - V# loaded from s17 + 0x40 into s[12:15]
        //   - s_buffer_load_dwordx2 s[22:23], s[12:15], 0x0
        //   - s_mov_b64 s[24:25], s12 (copies low 64 bits of V#)
        //   - s_buffer_load_dwordx2 s[12:13], s[12:15], s18 (reads s[12:15] AND overwrites s[12:13])
        //   - s_mov_b64 s[26:27], s14 (copies high 64 bits of V#)
        //   - scalar compare on V# word 2 (s26 / num_records) taints the V# load def
        //   - s_mov_b32 s12, s24 restores s12 before buffer_load_dword v2, v2, s[12:15], 0 offen
        //   - Synthesized RootDescriptor V# in s[12:15] from s[19:20] with s_buffer_load_* overwriting
        //     s17 and s19 while continuing to read s[12:15]
        //   - Subsequent table V# reloads from s16 + 0x40 and s17 + {0x0, 0x60, 0x80}
        //   - Final buffer_load_dword using the split-copied V# in s[24:27]
        //
        // Previously in S40, tainting def(s[12:15]) dropped indirect_smem_cbv_keys for the earlier
        // s_buffer_load_dwordx2 reads on s[12:15], emitting a `cb` binding with `source: None`.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 10, 17)); // s_mov_b32 s10, s17
        words.push(enc_sop1(0x03, 11, 0x80)); // s_mov_b32 s11, 0
        words.extend_from_slice(&enc_smem(0x02, 12, 5, 0x7D, 0x40, false, false)); // s_load_dwordx4 s[12:15], s[10:11], 0x40
        words.extend_from_slice(&enc_smem(0x09, 22, 6, 0x7D, 0x0, false, false)); // s_buffer_load_dwordx2 s[22:23], s[12:15], 0x0
        words.push(enc_sop1(0x04, 24, 12)); // s_mov_b64 s[24:25], s[12:13]
        words.extend_from_slice(&enc_smem(0x09, 12, 6, 18, 0x0, false, false)); // s_buffer_load_dwordx2 s[12:13], s[12:15], s18
        words.push(enc_sop1(0x04, 26, 14)); // s_mov_b64 s[26:27], s[14:15]
        // Inspect V# word 2 (num_records) in s26 -> taints the V# load def!
        words.push(enc_sopc(0x06, 26, 128 + 0)); // s_cmp_eq_u32 s26, 0
        // Restore s12 from s24 and read storage buffer via s[12:15] (offen)
        words.push(enc_sop1(0x03, 12, 24)); // s_mov_b32 s12, s24
        let mubuf_s12_w0 = (0b111000u32 << 26) | (0x0Cu32 << 18) | (1u32 << 12); // buffer_load_dword offen
        let mubuf_s12_w1 = (0x80u32 << 24) | (3u32 << 16) | (2u32 << 8) | 2u32; // vdata=v2, vaddr=v2, srsrc=3 (s[12:15])
        words.extend_from_slice(&[mubuf_s12_w0, mubuf_s12_w1]);
        // Synthesize RootDescriptor V# in s[12:15] from s[19:20]
        words.push(enc_sop1(0x03, 12, 19)); // s_mov_b32 s12, s19
        words.push(enc_sop1(0x03, 13, 20)); // s_mov_b32 s13, s20
        words.push(enc_sop1(0x03, 14, 193)); // s_mov_b32 s14, -1
        words.push(enc_sop1(0x03, 15, 0xFF)); // s_mov_b32 s15, 0x31016fac
        words.push(0x3101_6FAC);
        words.extend_from_slice(&enc_smem(0x0B, 40, 6, 0x7D, 0x60, false, false)); // s_buffer_load_dwordx8 s[40:47], s[12:15], 0x60
        words.extend_from_slice(&enc_smem(0x08, 17, 6, 0x7D, 0x0, false, false)); // s_buffer_load_dword s17, s[12:15], 0x0
        words.extend_from_slice(&enc_smem(0x08, 19, 6, 0x7D, 0xC, false, false)); // s_buffer_load_dword s19, s[12:15], 0xc
        words.extend_from_slice(&enc_smem(0x08, 17, 6, 0x7D, 0x18, false, false)); // s_buffer_load_dword s17, s[12:15], 0x18
        // Reload table V# descriptors from s16 + 0x40 and s17 + {0x0, 0x60, 0x80}
        words.extend_from_slice(&enc_smem(0x02, 12, 8, 0x7D, 0x40, false, false)); // s_load_dwordx4 s[12:15], s[16:17], 0x40
        words.extend_from_slice(&enc_smem(0x08, 30, 6, 0x7D, 0x0, false, false)); // s_buffer_load_dword s30, s[12:15], 0x0
        words.extend_from_slice(&enc_smem(0x02, 16, 5, 0x7D, 0x0, false, false)); // s_load_dwordx4 s[16:19], s[10:11], 0x0
        words.extend_from_slice(&enc_smem(0x08, 31, 8, 0x7D, 0x0, false, false)); // s_buffer_load_dword s31, s[16:19], 0x0
        words.extend_from_slice(&enc_smem(0x02, 40, 5, 0x7D, 0x60, false, false)); // s_load_dwordx4 s[40:43], s[10:11], 0x60
        words.extend_from_slice(&enc_smem(0x08, 32, 20, 0x7D, 0x0, false, false)); // s_buffer_load_dword s32, s[40:43], 0x0
        words.extend_from_slice(&enc_smem(0x02, 12, 5, 0x7D, 0x80, false, false)); // s_load_dwordx4 s[12:15], s[10:11], 0x80
        words.extend_from_slice(&enc_smem(0x08, 33, 6, 0x7D, 0x0, false, false)); // s_buffer_load_dword s33, s[12:15], 0x0
        // Final buffer_load_dword using the split-copied V# in s[24:27] (srsrc=6)
        let mubuf_s24_w0 = (0b111000u32 << 26) | (0x0Cu32 << 18) | (1u32 << 12);
        let mubuf_s24_w1 = (0x80u32 << 24) | (6u32 << 16) | (3u32 << 8) | 2u32; // vdata=v3, vaddr=v2, srsrc=6 (s[24:27])
        words.extend_from_slice(&[mubuf_s24_w0, mubuf_s24_w1]);
        // Combine results and export pos0
        words.push(enc_vop2(0x03, 2, 22, 2)); // v2 += s22
        words.push(enc_vop2(0x03, 3, 30, 3)); // v3 += s30
        words.push(enc_vop2(0x03, 4, 31, 2)); // v4 = s31 + v2
        words.push(enc_vop2(0x03, 5, 32, 3)); // v5 = s32 + v3
        words.push(enc_vop2(0x03, 5, 33, 5)); // v5 += s33
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [2, 3, 4, 5]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Vertex);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.source.is_some()),
            "every binding must have a non-None reflected source (zero source: None): {:?}",
            out.bindings
        );

        let cbv_sources: Vec<WgslBindingSource> = out
            .bindings
            .iter()
            .filter(|b| b.kind == "uniform")
            .map(|b| b.source.expect("uniform binding must have source"))
            .collect();
        assert_eq!(
            cbv_sources,
            vec![
                WgslBindingSource::Table {
                    user_sgpr: 16,
                    table_byte_offset: 0x40,
                },
                WgslBindingSource::Table {
                    user_sgpr: 17,
                    table_byte_offset: 0x0,
                },
                WgslBindingSource::Table {
                    user_sgpr: 17,
                    table_byte_offset: 0x40,
                },
                WgslBindingSource::Table {
                    user_sgpr: 17,
                    table_byte_offset: 0x60,
                },
                WgslBindingSource::Table {
                    user_sgpr: 17,
                    table_byte_offset: 0x80,
                },
                WgslBindingSource::RootDescriptor { user_sgpr: 19 },
            ],
            "bindings: {:?}",
            out.bindings
        );

        let sbufs: Vec<_> = out
            .bindings
            .iter()
            .filter(|b| b.kind == "read-only-storage")
            .collect();
        assert_eq!(sbufs.len(), 1, "bindings: {:?}", out.bindings);
        assert_eq!(
            sbufs[0].source,
            Some(WgslBindingSource::Table {
                user_sgpr: 17,
                table_byte_offset: 0x40,
            })
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s41_vsharp_word23_recovery_cfg_unwritten_scratch_and_merged_branch_defs() {
        // Exercises three additional S41 reflection fixes in one vertex shader:
        //   1. Dynamic base-address offset on V# words 0..1 (`s_add_u32 s12, s24, s22;
        //      s_addc_u32 s13, s25, 0`) while words 2..3 (`s14..s15`) still carry intact
        //      `TableDesc(s17, 0x40 + 8/12)` -> recovers `Table(s17, 0x40)` for both
        //      `s_buffer_load_dword` and `buffer_load_dword`.
        //   2. High scratch SGPR `s[40:43]` (`>= s32`, starts `Unwritten`) initialized inside
        //      an `if` branch (`s17 + 0x60`) and consumed after the merge -> keeps `Table(s17, 0x60)`
        //      instead of being wiped to `Unknown` by the unwritten skip edge.
        //   3. Two branches of an `if/else` diamond both load `s[44:47]` from `s17 + 0x80`
        //      at distinct instructions (`d1` and `d2`) and merge before `s_buffer_load_dword` ->
        //      BOTH `d1` and `d2` are suppressed (no leaked `RootDescriptor(s17)` CBV).
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 10, 17)); // s10 = s17
        words.push(enc_sop1(0x03, 11, 0x80)); // s11 = 0
        words.extend_from_slice(&enc_smem(0x02, 12, 5, 0x7D, 0x40, false, false)); // V# @ s17+0x40
        words.extend_from_slice(&enc_smem(0x09, 22, 6, 0x7D, 0x0, false, false)); // s[22:23] = dynamic data
        words.push(enc_sop1(0x04, 24, 12)); // s[24:25] = s[12:13]
        // Dynamic offset added to base_addr_lo/hi of s[12:15]; s[14:15] remain TableDesc(s17, 72/76)
        words.push(enc_sop2(0x00, 12, 24, 22)); // s_add_u32 s12, s24, s22 (Unknown!)
        words.push(enc_sop2(0x04, 13, 25, 128 + 0)); // s_addc_u32 s13, s25, 0 (Unknown!)
        words.extend_from_slice(&enc_smem(0x08, 28, 6, 0x7D, 0x0, false, false)); // s_buffer_load_dword s28, s[12:15], 0
        let mubuf_w0 = (0b111000u32 << 26) | (0x0Cu32 << 18) | (1u32 << 12);
        let mubuf_w1 = (0x80u32 << 24) | (3u32 << 16) | (2u32 << 8) | 2u32;
        words.extend_from_slice(&[mubuf_w0, mubuf_w1]); // buffer_load_dword v2, v2, s[12:15], 0 offen

        // 2. Skip-if initializing high scratch SGPR s[40:43] (`Unwritten` on skip path)
        words.push(enc_sopc(0x06, 22, 128 + 0));
        let skip_branch_idx = words.len();
        words.push(0); // s_cbranch_scc0 AFTER_INIT
        words.extend_from_slice(&enc_smem(0x02, 40, 5, 0x7D, 0x60, false, false)); // s[40:43] = V# @ s17+0x60
        let after_init_off = words.len() * 4;
        words[skip_branch_idx] = enc_sopp(0x04, branch_simm(skip_branch_idx * 4, after_init_off));
        words.extend_from_slice(&enc_smem(0x08, 29, 20, 0x7D, 0x0, false, false)); // s_buffer_load_dword s29, s[40:43], 0

        // 3. If/else diamond loading s[44:47] from s17+0x80 on BOTH branches (d1 and d2)
        words.push(enc_sopc(0x06, 23, 128 + 0));
        let diamond_cbranch_idx = words.len();
        words.push(0); // s_cbranch_scc0 ELSE
        words.extend_from_slice(&enc_smem(0x02, 44, 5, 0x7D, 0x80, false, false)); // d1: s[44:47] @ s17+0x80
        let diamond_sbranch_idx = words.len();
        words.push(0); // s_branch MERGE
        let else_off = words.len() * 4;
        words.extend_from_slice(&enc_smem(0x02, 44, 5, 0x7D, 0x80, false, false)); // d2: s[44:47] @ s17+0x80
        let merge_off = words.len() * 4;
        words[diamond_cbranch_idx] = enc_sopp(0x04, branch_simm(diamond_cbranch_idx * 4, else_off));
        words[diamond_sbranch_idx] = enc_sopp(0x02, branch_simm(diamond_sbranch_idx * 4, merge_off));
        words.extend_from_slice(&enc_smem(0x08, 30, 22, 0x7D, 0x0, false, false)); // s_buffer_load_dword s30, s[44:47], 0

        words.push(enc_vop2(0x03, 2, 28, 2));
        words.push(enc_vop2(0x03, 3, 29, 2));
        words.push(enc_vop2(0x03, 4, 30, 3));
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [2, 3, 4, 4]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Vertex);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(
            out.bindings.iter().all(|b| b.source.is_some()),
            "every binding must have a non-None reflected source: {:?}",
            out.bindings
        );

        let cbv_sources: Vec<WgslBindingSource> = out
            .bindings
            .iter()
            .filter(|b| b.kind == "uniform")
            .map(|b| b.source.expect("uniform binding must have source"))
            .collect();
        assert_eq!(
            cbv_sources,
            vec![
                WgslBindingSource::Table {
                    user_sgpr: 17,
                    table_byte_offset: 0x40,
                },
                WgslBindingSource::Table {
                    user_sgpr: 17,
                    table_byte_offset: 0x60,
                },
                WgslBindingSource::Table {
                    user_sgpr: 17,
                    table_byte_offset: 0x80,
                },
            ],
            "merged branch V# defs must both be suppressed without leaking RootDescriptor(17): {:?}",
            out.bindings
        );
        let sbuf = out
            .bindings
            .iter()
            .find(|b| b.kind == "read-only-storage")
            .expect("expected read-only-storage binding");
        assert_eq!(
            sbuf.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 17,
                table_byte_offset: 0x40,
            })
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s42_buffer_pulling_vertex_stage_emits_vertex_index_mode_and_no_bogus_locations() {
        // Reproduces #949 (1,687 buffer-pulling vertex stages):
        //   - Loads V# from user-data descriptor table `s8 + 0x20` (`TableDesc`)
        //   - Synthesizes a root descriptor V# from `s10` (`RootDescriptor`)
        //   - Pulls vertex data via MUBUF `buffer_load_dwordx4` using `v1` (SV_InstanceID /
        //     read-before-write VGPR >= 1) and `v0` (SV_VertexID)
        //   - Zero IA `VbDesc` vertex fetches!
        // Must emit `vertex_input_mode == "vertex_index"`, empty `vertex_inputs` (omitted in JSON),
        // `@builtin(vertex_index)` with NO unused `@location(...)` inputs on `vs_main`, and 100%
        // reflected binding sources.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0 (zero-extend table ptr s8)
        words.extend_from_slice(&enc_smem(0x02, 12, 4, 0x7D, 0x20, false, false)); // TableDesc V# @ s8+0x20 -> s[12:15]
        // Synthesized root descriptor V# in s[16:19] from user SGPR s10
        words.push(enc_sop1(0x03, 16, 10)); // s16 = s10 (UserTable 10)
        words.push(enc_sop1(0x03, 17, 0x80)); // s17 = 0
        words.push(enc_sop1(0x03, 18, 0xFF)); // s18 = 0xFFFFFFFF (num_records)
        words.push(0xFFFF_FFFF);
        words.push(enc_sop1(0x03, 19, 0xFF)); // s19 = rsrc_word3
        words.push(0x3101_4000);
        // MUBUF buffer_load_dwordx4 (0x0E) offen=1 using vaddr=v1 (reads v1 before write!) from s[12:15]
        let mubuf1_w0 = (0b111000u32 << 26) | (0x0Eu32 << 18) | (1u32 << 12);
        let mubuf1_w1 = (0x80u32 << 24) | (3u32 << 16) | (4u32 << 8) | 1u32;
        words.extend_from_slice(&[mubuf1_w0, mubuf1_w1]);
        // MUBUF buffer_load_dwordx4 (0x0E) offen=1 using vaddr=v0 from root descriptor s[16:19]
        let mubuf2_w0 = (0b111000u32 << 26) | (0x0Eu32 << 18) | (1u32 << 12);
        let mubuf2_w1 = (0x80u32 << 24) | (4u32 << 16) | (8u32 << 8) | 0u32;
        words.extend_from_slice(&[mubuf2_w0, mubuf2_w1]);
        words.push(enc_vop2(0x03, 4, 256 + 4, 8)); // v4 += v8
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Vertex);
        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(
            out.binding_summary.vertex_input_mode, "vertex_index",
            "buffer-pulling VS with zero IA VbDesc fetches must NOT report vertex_input_mode=attributes"
        );
        assert!(out.vertex_inputs.is_empty(), "vertex_inputs: {:?}", out.vertex_inputs);
        assert!(
            out.wgsl.contains("@builtin(vertex_index)"),
            "WGSL must declare @builtin(vertex_index):\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("struct VsIn"),
            "buffer-pulling VS must not declare unused VsIn @location inputs:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("@location(0) position"),
            "buffer-pulling VS must not declare synthetic @location(0) position:\n{}",
            out.wgsl
        );
        assert_eq!(out.bindings.len(), 2, "bindings: {:?}", out.bindings);
        assert!(
            out.bindings.iter().all(|b| b.source.is_some()),
            "every binding must have a reflected source: {:?}",
            out.bindings
        );
        let json = out.to_json();
        assert!(
            json.contains("\"vertex_input_mode\": \"vertex_index\""),
            "JSON:\n{json}"
        );
        assert!(!json.contains("\"vertex_inputs\""), "JSON:\n{json}");
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s43_inline_sampler_sdwords_from_salu_const_ops_and_copies() {
        // Tests Timon's exact #950 inline S# sampler JSON contract:
        //   {"inline_sampler": {"sdwords": [d0, d1, d2, d3]}}
        // Constructs TWO real GFX10 S# descriptors using a broad mix of SALU constant ops:
        //   Sampler A = [0x92, 0x00F0_0000, 0x0850_0000, 0] (clamp-to-edge bilinear, max LOD 15)
        //     - d0 via s_movk_i32 + s_or_b32
        //     - d1 via s_bfm_b32
        //     - d2 via s_movk_i32 + s_pack_ll_b32_b16
        //     - d3 via self s_xor_b32 s23, s8, s8 -> 0
        //     - copied via s_mov_b64 into s[28:31] and reused (must deduplicate to 1 binding!)
        //   Sampler B = [0x3000, 0x00F0_0000, 0x0400_0000, 0] (comparison sampler)
        //     - d0/d1 seeded via s_bfm_b64 s[32:33], 4, 52 -> [0, 0x00F0_0000]
        //     - d0 overwritten via sign-extended negative s_movk_i32 (-16) -> s_not_b32 (0xF) ->
        //       s_lshl_b32 (, 12) (0xF000) -> s_sub_u32 (, 0xC000) -> 0x3000
        //     - d2 via inline float constant 244 (2.0f32 = 0x4000_0000) -> s_lshr_b32 (, 4) -> 0x0400_0000
        //     - d3 via s_bfe_u32 + s_add_u32 + s_and_b32 -> 0
        //   Also verifies s_pack_lh_b32_b16 and s_pack_hh_b32_b16 constant folding in Sampler C.
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0
        words.extend_from_slice(&enc_smem(0x03, 40, 4, 0x7D, 0x00, false, false)); // T# @ s8+0 -> s[40:47]

        // Sampler A in s[20:23]: [0x92, 0x00F0_0000, 0x0850_0000, 0]
        words.push(enc_sopk(0x00, 20, 0x0090)); // s_movk_i32 s20, 0x0090
        words.push(enc_sop2(0x10, 20, 20, 128 + 2)); // s_or_b32 s20, s20, 2 -> 0x92
        words.push(enc_sop2(0x24, 21, 128 + 4, 128 + 20)); // s_bfm_b32 s21, 4, 20 -> 0x00F0_0000
        words.push(enc_sopk(0x00, 26, 0x0850)); // s_movk_i32 s26, 0x0850
        words.push(enc_sop2(0x32, 22, 128 + 0, 26)); // s_pack_ll_b32_b16 s22, 0, s26 -> 0x0850_0000
        words.push(enc_sop2(0x12, 23, 8, 8)); // s_xor_b32 s23, s8, s8 -> 0

        // Copy Sampler A into s[28:31] via s_mov_b64
        words.push(enc_sop1(0x04, 28, 20)); // s_mov_b64 s[28:29], s[20:21]
        words.push(enc_sop1(0x04, 30, 22)); // s_mov_b64 s[30:31], s[22:23]

        // Sampler B in s[32:35]: [0x3000, 0x00F0_0000, 0x0400_0000, 0]
        words.push(enc_sop2(0x25, 32, 128 + 4, 128 + 52)); // s_bfm_b64 s[32:33], 4, 52 -> [0, 0x00F0_0000]
        words.push(enc_sopk(0x00, 32, (-16i16) as u16)); // s_movk_i32 s32, -16 (0xFFFF_FFF0)
        words.push(enc_sop1(0x07, 32, 32)); // s_not_b32 s32, s32 -> 0xF
        words.push(enc_sop2(0x1E, 32, 32, 128 + 12)); // s_lshl_b32 s32, s32, 12 -> 0xF000
        words.push(enc_sop2(0x01, 32, 32, 0xFF)); // s_sub_u32 s32, s32, 0xC000 -> 0x3000
        words.push(0x0000_C000);
        words.push(enc_sop1(0x03, 34, 244)); // s_mov_b32 s34, 2.0 (0x4000_0000)
        words.push(enc_sop2(0x20, 34, 34, 128 + 4)); // s_lshr_b32 s34, s34, 4 -> 0x0400_0000
        words.push(enc_sop1(0x03, 35, 0xFF)); // s_mov_b32 s35, 0x1234_5678
        words.push(0x1234_5678);
        words.push(enc_sop2(0x27, 35, 35, 0xFF)); // s_bfe_u32 s35, s35, off=8/w=4 -> 0x6
        words.push(0x0004_0008);
        words.push(enc_sop2(0x00, 35, 35, 128 + 10)); // s_add_u32 s35, s35, 10 -> 0x10
        words.push(enc_sop2(0x0E, 35, 35, 128 + 0)); // s_and_b32 s35, s35, 0 -> 0

        // Sampler C in s[36:39]: exercises s_pack_lh_b32_b16 and s_pack_hh_b32_b16
        words.push(enc_sop1(0x03, 36, 0xFF));
        words.push(0x1111_0092);
        words.push(enc_sop1(0x03, 37, 0xFF));
        words.push(0x00F0_2222);
        words.push(enc_sop2(0x33, 36, 36, 37)); // s_pack_lh_b32_b16 -> 0x00F0_0092
        words.push(enc_sop2(0x34, 37, 36, 37)); // s_pack_hh_b32_b16 -> 0x00F0_00F0
        words.push(enc_sop1(0x03, 38, 128 + 0));
        words.push(enc_sop1(0x03, 39, 128 + 0));

        // Sample with Sampler A (s[20:23]), copied Sampler A (s[28:31]), Sampler B (s[32:35]), Sampler C (s[36:39])
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 4, 10, 5)); // ssamp=5 -> s[20:23]
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 8, 10, 7)); // ssamp=7 -> s[28:31] (dup of A!)
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 12, 10, 8)); // ssamp=8 -> s[32:35]
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 16, 10, 9)); // ssamp=9 -> s[36:39]
        words.push(enc_vop2(0x03, 4, 256 + 4, 8));
        words.push(enc_vop2(0x03, 4, 256 + 4, 12));
        words.push(enc_vop2(0x03, 4, 256 + 4, 16));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);

        let samps: Vec<_> = out.bindings.iter().filter(|b| b.kind == "sampler").collect();
        assert_eq!(
            samps.len(),
            3,
            "copied Sampler A quad must deduplicate; expected 3 distinct inline samplers: {:?}",
            samps
        );
        let sources: Vec<WgslBindingSource> = samps
            .iter()
            .map(|s| s.source.expect("sampler must have reflected source"))
            .collect();
        assert!(sources.contains(&WgslBindingSource::InlineSampler {
            sdwords: [0x92, 0x00F0_0000, 0x0850_0000, 0],
        }));
        assert!(sources.contains(&WgslBindingSource::InlineSampler {
            sdwords: [0x3000, 0x00F0_0000, 0x0400_0000, 0],
        }));
        assert!(sources.contains(&WgslBindingSource::InlineSampler {
            sdwords: [0x00F0_0092, 0x00F0_00F0, 0, 0],
        }));

        let json = out.to_json();
        assert!(
            json.contains("\"inline_sampler\": {\"sdwords\": [146, 15728640, 139460608, 0]}"),
            "JSON:\n{json}"
        );
        assert!(
            json.contains("\"inline_sampler\": {\"sdwords\": [12288, 15728640, 67108864, 0]}"),
            "JSON:\n{json}"
        );
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn s43_inline_sampler_with_unknown_word_falls_back_to_static_sampler() {
        // When word 0 is ConstImmediate but another word in s[20:23] is dynamic/ambiguous
        // (e.g. s_cselect_b32 between two different constants -> value: None), we must NOT
        // fabricate sdwords — fall back cleanly to StaticSampler { index: 0 }.
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_sop1(0x03, 9, 0x80)); // s9 = 0
        words.extend_from_slice(&enc_smem(0x03, 24, 4, 0x7D, 0x00, false, false)); // T# @ s8+0
        words.push(enc_sop1(0x03, 20, 128 + 1)); // s20 = 1 (known const)
        words.push(enc_sopc(0x06, 8, 128 + 0)); // s_cmp_eq_u32 s8, 0
        words.push(enc_sop2(0x0A, 21, 128 + 2, 128 + 3)); // s_cselect_b32 s21, 2, 3 -> value: None!
        words.push(enc_sop1(0x03, 22, 128 + 0));
        words.push(enc_sop1(0x03, 23, 128 + 0));
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 4, 6, 5)); // ssamp=5 -> s[20:23]
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Pixel);
        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        let samp = out.bindings.iter().find(|b| b.kind == "sampler").unwrap();
        assert_eq!(
            samp.source,
            Some(WgslBindingSource::StaticSampler { index: 0 })
        );
        let json = out.to_json();
        assert!(json.contains("\"static_sampler\": 0"), "JSON:\n{json}");
        assert!(!json.contains("\"inline_sampler\""), "JSON:\n{json}");
        assert_naga_wgsl_valid(&out.wgsl);
    }
}
