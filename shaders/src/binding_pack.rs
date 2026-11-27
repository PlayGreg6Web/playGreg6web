#![forbid(unsafe_code)]

//! WebGPU/D3D12 per-stage binding-limit auditing and semantics-preserving WGSL binding
//! compaction/packing (`#2110`, Refs `#663`, `#482`).
//!
//! D3D12/Dawn hardware per-stage resource limits cannot be raised past:
//! * `maxUniformBuffersPerShaderStage`  = 12
//! * `maxStorageBuffersPerShaderStage`  = 16
//! * `maxStorageTexturesPerShaderStage` = 8
//! * `maxSampledTexturesPerShaderStage` = 16
//! * `maxSamplersPerShaderStage`        = 16
//!
//! This module provides:
//! 1. Per-stage binding counting & overflow auditing across `WgslBindingInfo`, `SidecarBinding`,
//!    and `naga::Module` IR (including runtime-owned group-3 `rt_arena` and `ms_out` bindings).
//! 2. Canonical descriptor-source normalization so duplicate/aliased descriptor bindings collapse
//!    to a single `@group/@binding` global while keeping reflection metadata 100% in sync.
//! 3. Post-lowering Naga IR dead/unreferenced binding pruning and duplicate binding deduplication
//!    with full `Handle<GlobalVariable>` remapping across entry points and helper functions.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use naga::{
    AddressSpace, Arena, Block, Expression, Function, GlobalVariable, Handle, ImageClass, Module,
    Span, Statement, StorageAccess, TypeInner,
};

use crate::rdna2_to_wgsl::{
    CbvIndexedHeapSource, LaneIndexedHeapSource, WgslBindingInfo, WgslBindingSource,
};
use crate::reflect::{SidecarBinding, SidecarBindingKind};

/// WebGPU/D3D12 hardware ceiling for `maxUniformBuffersPerShaderStage`.
pub const WEBGPU_D3D12_MAX_UNIFORM_BUFFERS_PER_STAGE: usize = 12;

/// WebGPU/D3D12 hardware ceiling for `maxStorageBuffersPerShaderStage`.
pub const WEBGPU_D3D12_MAX_STORAGE_BUFFERS_PER_STAGE: usize = 16;

/// WebGPU/D3D12 hardware ceiling for `maxStorageTexturesPerShaderStage`.
pub const WEBGPU_D3D12_MAX_STORAGE_TEXTURES_PER_STAGE: usize = 8;

/// WebGPU/D3D12 hardware ceiling for `maxSampledTexturesPerShaderStage`.
pub const WEBGPU_D3D12_MAX_SAMPLED_TEXTURES_PER_STAGE: usize = 16;

/// WebGPU/D3D12 hardware ceiling for `maxSamplersPerShaderStage`.
pub const WEBGPU_D3D12_MAX_SAMPLERS_PER_STAGE: usize = 16;

/// Runtime-owned RT arena prelude storage buffers (`@group(3) @binding(16..21)`):
/// `pg6_arena0..3`, `pg6_arena_dir`, and `pg6_rt_stats`.
pub const RT_ARENA_STORAGE_BUFFER_COUNT: usize = 6;

/// Runtime-owned mesh compute output buffer (`@group(3) @binding(0)` `ms_out`).
pub const MESH_OUTPUT_STORAGE_BUFFER_COUNT: usize = 1;

/// Group index reserved for runtime-owned bindings (`ms_out` and `rt_arena`).
const RUNTIME_OWNED_GROUP: u32 = 3;

/// Per-stage WebGPU resource binding limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageBindingLimits {
    pub max_uniform_buffers: usize,
    pub max_storage_buffers: usize,
    pub max_storage_textures: usize,
    pub max_sampled_textures: usize,
    pub max_samplers: usize,
}

impl StageBindingLimits {
    /// D3D12 / Dawn hardware per-stage binding ceilings (12 / 16 / 8 / 16 / 16).
    pub const fn webgpu_d3d12() -> Self {
        Self {
            max_uniform_buffers: WEBGPU_D3D12_MAX_UNIFORM_BUFFERS_PER_STAGE,
            max_storage_buffers: WEBGPU_D3D12_MAX_STORAGE_BUFFERS_PER_STAGE,
            max_storage_textures: WEBGPU_D3D12_MAX_STORAGE_TEXTURES_PER_STAGE,
            max_sampled_textures: WEBGPU_D3D12_MAX_SAMPLED_TEXTURES_PER_STAGE,
            max_samplers: WEBGPU_D3D12_MAX_SAMPLERS_PER_STAGE,
        }
    }

    /// Strict WebGPU 1.0 baseline default per-stage binding limits (12 / 8 / 4 / 16 / 16).
    pub const fn webgpu_baseline() -> Self {
        Self {
            max_uniform_buffers: 12,
            max_storage_buffers: 8,
            max_storage_textures: 4,
            max_sampled_textures: 16,
            max_samplers: 16,
        }
    }
}

impl Default for StageBindingLimits {
    fn default() -> Self {
        Self::webgpu_d3d12()
    }
}

/// Resource kind for per-stage WebGPU binding-limit diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageBindingLimitKind {
    UniformBuffers,
    StorageBuffers,
    StorageTextures,
    SampledTextures,
    Samplers,
}

impl StageBindingLimitKind {
    /// Exact WebGPU `GPUSupportedLimits` limit name.
    pub const fn as_limit_name(self) -> &'static str {
        match self {
            Self::UniformBuffers => "maxUniformBuffersPerShaderStage",
            Self::StorageBuffers => "maxStorageBuffersPerShaderStage",
            Self::StorageTextures => "maxStorageTexturesPerShaderStage",
            Self::SampledTextures => "maxSampledTexturesPerShaderStage",
            Self::Samplers => "maxSamplersPerShaderStage",
        }
    }

    /// Short sanitized JSON key name for this resource kind.
    pub const fn as_kind_key(self) -> &'static str {
        match self {
            Self::UniformBuffers => "uniform_buffers",
            Self::StorageBuffers => "storage_buffers",
            Self::StorageTextures => "storage_textures",
            Self::SampledTextures => "sampled_textures",
            Self::Samplers => "samplers",
        }
    }
}

/// Per-stage binding counts broken down by WebGPU resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StageBindingCounts {
    pub uniform_buffers: usize,
    pub storage_buffers: usize,
    pub storage_textures: usize,
    pub sampled_textures: usize,
    pub samplers: usize,
}

impl StageBindingCounts {
    /// Total bindings across all five per-stage resource classes.
    pub const fn total(&self) -> usize {
        self.uniform_buffers
            + self.storage_buffers
            + self.storage_textures
            + self.sampled_textures
            + self.samplers
    }

    /// Count per-stage resource bindings from translated [`WgslBindingInfo`] metadata,
    /// including runtime-owned group-3 `rt_arena` (6 storage buffers) and `ms_out` (1 storage buffer)
    /// bindings that count against the hardware stage limit.
    pub fn from_wgsl_bindings(
        bindings: &[WgslBindingInfo],
        rt_arena: bool,
        has_mesh_output: bool,
    ) -> Self {
        let mut counts = Self::default();
        for b in bindings {
            match b.kind {
                "uniform" => counts.uniform_buffers += 1,
                "storage" | "read-only-storage" => counts.storage_buffers += 1,
                "storage-texture" => counts.storage_textures += 1,
                "texture_2d" | "texture_2d_array" | "texture_3d" | "texture_cube"
                | "texture_cube_array" | "texture_depth_2d" | "texture_depth_2d_array"
                | "texture_depth_cube" | "texture_depth_cube_array" | "texture" => {
                    counts.sampled_textures += 1;
                }
                "sampler" | "sampler_comparison" => counts.samplers += 1,
                _ => {}
            }
        }
        if rt_arena {
            counts.storage_buffers += RT_ARENA_STORAGE_BUFFER_COUNT;
        }
        if has_mesh_output {
            counts.storage_buffers += MESH_OUTPUT_STORAGE_BUFFER_COUNT;
        }
        counts
    }

    /// Count per-stage resource bindings from [`SidecarBinding`] entries (including runtime-owned
    /// `rt_arena` and `mesh_output` storage buffers).
    pub fn from_sidecar_bindings(
        bindings: &[SidecarBinding],
        rt_arena: bool,
        has_mesh_output: bool,
    ) -> Self {
        let mut counts = Self::default();
        for b in bindings {
            match b.kind {
                SidecarBindingKind::Uniform => counts.uniform_buffers += 1,
                SidecarBindingKind::Storage | SidecarBindingKind::ReadOnlyStorage => {
                    counts.storage_buffers += 1;
                }
                SidecarBindingKind::StorageTexture => counts.storage_textures += 1,
                SidecarBindingKind::Texture => counts.sampled_textures += 1,
                SidecarBindingKind::Sampler | SidecarBindingKind::ComparisonSampler => {
                    counts.samplers += 1;
                }
            }
        }
        if rt_arena {
            counts.storage_buffers += RT_ARENA_STORAGE_BUFFER_COUNT;
        }
        if has_mesh_output {
            counts.storage_buffers += MESH_OUTPUT_STORAGE_BUFFER_COUNT;
        }
        counts
    }

    /// Count all `@group/@binding` global resource declarations in a [`naga::Module`].
    pub fn from_naga_module(module: &Module) -> Self {
        let mut counts = Self::default();
        for (_, var) in module.global_variables.iter() {
            if var.binding.is_none() {
                continue;
            }
            match var.space {
                AddressSpace::Uniform => counts.uniform_buffers += 1,
                AddressSpace::Storage { .. } => counts.storage_buffers += 1,
                AddressSpace::Handle => match module.types[var.ty].inner {
                    TypeInner::Image {
                        class: ImageClass::Storage { .. },
                        ..
                    } => counts.storage_textures += 1,
                    TypeInner::Image { .. } => counts.sampled_textures += 1,
                    TypeInner::Sampler { .. } => counts.samplers += 1,
                    _ => {}
                },
                _ => {}
            }
        }
        counts
    }
}

/// One per-stage binding-limit overflow finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageBindingLimitOverflow {
    pub kind: StageBindingLimitKind,
    pub count: usize,
    pub limit: usize,
}

/// Per-stage binding-limit audit result against [`StageBindingLimits`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageBindingLimitAudit {
    pub counts: StageBindingCounts,
    pub limits: StageBindingLimits,
    pub overflows: Vec<StageBindingLimitOverflow>,
}

impl StageBindingLimitAudit {
    /// Evaluate `counts` against `limits`.
    pub fn from_counts(counts: StageBindingCounts, limits: StageBindingLimits) -> Self {
        let mut overflows = Vec::new();
        let checks = [
            (
                StageBindingLimitKind::UniformBuffers,
                counts.uniform_buffers,
                limits.max_uniform_buffers,
            ),
            (
                StageBindingLimitKind::StorageBuffers,
                counts.storage_buffers,
                limits.max_storage_buffers,
            ),
            (
                StageBindingLimitKind::StorageTextures,
                counts.storage_textures,
                limits.max_storage_textures,
            ),
            (
                StageBindingLimitKind::SampledTextures,
                counts.sampled_textures,
                limits.max_sampled_textures,
            ),
            (
                StageBindingLimitKind::Samplers,
                counts.samplers,
                limits.max_samplers,
            ),
        ];
        for (kind, count, limit) in checks {
            if count > limit {
                overflows.push(StageBindingLimitOverflow { kind, count, limit });
            }
        }
        Self {
            counts,
            limits,
            overflows,
        }
    }

    /// Audit translated [`WgslBindingInfo`] declarations against `limits`.
    pub fn audit_wgsl_bindings(
        bindings: &[WgslBindingInfo],
        rt_arena: bool,
        has_mesh_output: bool,
        limits: StageBindingLimits,
    ) -> Self {
        let counts = StageBindingCounts::from_wgsl_bindings(bindings, rt_arena, has_mesh_output);
        Self::from_counts(counts, limits)
    }

    /// Audit [`SidecarBinding`] declarations against `limits`.
    pub fn audit_sidecar_bindings(
        bindings: &[SidecarBinding],
        rt_arena: bool,
        has_mesh_output: bool,
        limits: StageBindingLimits,
    ) -> Self {
        let counts = StageBindingCounts::from_sidecar_bindings(bindings, rt_arena, has_mesh_output);
        Self::from_counts(counts, limits)
    }

    /// Audit all `@group/@binding` declarations in a [`naga::Module`] against `limits`.
    pub fn audit_naga_module(module: &Module, limits: StageBindingLimits) -> Self {
        let counts = StageBindingCounts::from_naga_module(module);
        Self::from_counts(counts, limits)
    }

    /// `true` when every resource class fits within `self.limits`.
    pub fn within_limits(&self) -> bool {
        self.overflows.is_empty()
    }
}

/// Sanitized per-stage binding-compaction counters produced during RDNA2 -> WGSL lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BindingPackStats {
    /// Unreferenced `@group/@binding` globals pruned from the module & reflection.
    pub dead_bindings_pruned: usize,
    /// Duplicate/aliased bindings folded onto a single `@group/@binding` global.
    pub duplicate_bindings_folded: usize,
    /// CBV uniform bindings merged into an existing storage buffer with the exact same source.
    pub cbv_merged_into_storage: usize,
    /// Excess non-root-constant uniform buffers (`> 12`) promoted to `var<storage, read>`.
    pub uniform_promoted_to_storage: usize,
}

impl BindingPackStats {
    /// Total bindings eliminated or re-packed by compaction passes.
    pub const fn total_reduced(&self) -> usize {
        self.dead_bindings_pruned
            + self.duplicate_bindings_folded
            + self.cbv_merged_into_storage
    }

    /// `true` when no compaction action fired for this stage.
    pub const fn is_zero(&self) -> bool {
        self.dead_bindings_pruned == 0
            && self.duplicate_bindings_folded == 0
            && self.cbv_merged_into_storage == 0
            && self.uniform_promoted_to_storage == 0
    }

    /// Accumulate another [`BindingPackStats`] into `self`.
    pub fn merge_from(&mut self, other: Self) {
        self.dead_bindings_pruned += other.dead_bindings_pruned;
        self.duplicate_bindings_folded += other.duplicate_bindings_folded;
        self.cbv_merged_into_storage += other.cbv_merged_into_storage;
        self.uniform_promoted_to_storage += other.uniform_promoted_to_storage;
    }
}

/// Sanitized aggregate binding-limit & compaction summary across a batch of shader stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackBindingLimitSummary {
    pub stages_within_limits: usize,
    pub stages_over_limit: usize,
    pub over_uniform_buffers: usize,
    pub over_storage_buffers: usize,
    pub over_storage_textures: usize,
    pub over_sampled_textures: usize,
    pub over_samplers: usize,
    pub max_uniform_buffers: usize,
    pub max_storage_buffers: usize,
    pub max_storage_textures: usize,
    pub max_sampled_textures: usize,
    pub max_samplers: usize,
    pub total_dead_bindings_pruned: usize,
    pub total_duplicate_bindings_folded: usize,
    pub total_cbv_merged_into_storage: usize,
    pub total_uniform_promoted_to_storage: usize,
}

impl PackBindingLimitSummary {
    /// Record one stage's audit and compaction stats into the aggregate summary.
    pub fn record_audit(&mut self, audit: &StageBindingLimitAudit, stats: BindingPackStats) {
        if audit.within_limits() {
            self.stages_within_limits += 1;
        } else {
            self.stages_over_limit += 1;
            for ov in &audit.overflows {
                match ov.kind {
                    StageBindingLimitKind::UniformBuffers => self.over_uniform_buffers += 1,
                    StageBindingLimitKind::StorageBuffers => self.over_storage_buffers += 1,
                    StageBindingLimitKind::StorageTextures => self.over_storage_textures += 1,
                    StageBindingLimitKind::SampledTextures => self.over_sampled_textures += 1,
                    StageBindingLimitKind::Samplers => self.over_samplers += 1,
                }
            }
        }
        self.max_uniform_buffers = self.max_uniform_buffers.max(audit.counts.uniform_buffers);
        self.max_storage_buffers = self.max_storage_buffers.max(audit.counts.storage_buffers);
        self.max_storage_textures = self.max_storage_textures.max(audit.counts.storage_textures);
        self.max_sampled_textures = self.max_sampled_textures.max(audit.counts.sampled_textures);
        self.max_samplers = self.max_samplers.max(audit.counts.samplers);
        self.total_dead_bindings_pruned += stats.dead_bindings_pruned;
        self.total_duplicate_bindings_folded += stats.duplicate_bindings_folded;
        self.total_cbv_merged_into_storage += stats.cbv_merged_into_storage;
        self.total_uniform_promoted_to_storage += stats.uniform_promoted_to_storage;
    }

    /// Record one translated stage directly from its [`WgslBindingInfo`] slice.
    pub fn record_wgsl_stage(
        &mut self,
        bindings: &[WgslBindingInfo],
        rt_arena: bool,
        has_mesh_output: bool,
        stats: BindingPackStats,
        limits: StageBindingLimits,
    ) {
        let audit =
            StageBindingLimitAudit::audit_wgsl_bindings(bindings, rt_arena, has_mesh_output, limits);
        self.record_audit(&audit, stats);
    }

    /// Render a strictly sanitized deterministic JSON object (no paths, no raw bytes).
    pub fn write_json(&self, out: &mut String, indent: &str) {
        let _ = writeln!(out, "{{");
        let _ = writeln!(
            out,
            "{indent}  \"stages_within_limits\": {},",
            self.stages_within_limits
        );
        let _ = writeln!(
            out,
            "{indent}  \"stages_over_limit\": {},",
            self.stages_over_limit
        );
        let _ = writeln!(
            out,
            "{indent}  \"over_uniform_buffers\": {},",
            self.over_uniform_buffers
        );
        let _ = writeln!(
            out,
            "{indent}  \"over_storage_buffers\": {},",
            self.over_storage_buffers
        );
        let _ = writeln!(
            out,
            "{indent}  \"over_storage_textures\": {},",
            self.over_storage_textures
        );
        let _ = writeln!(
            out,
            "{indent}  \"over_sampled_textures\": {},",
            self.over_sampled_textures
        );
        let _ = writeln!(out, "{indent}  \"over_samplers\": {},", self.over_samplers);
        let _ = writeln!(
            out,
            "{indent}  \"max_uniform_buffers\": {},",
            self.max_uniform_buffers
        );
        let _ = writeln!(
            out,
            "{indent}  \"max_storage_buffers\": {},",
            self.max_storage_buffers
        );
        let _ = writeln!(
            out,
            "{indent}  \"max_storage_textures\": {},",
            self.max_storage_textures
        );
        let _ = writeln!(
            out,
            "{indent}  \"max_sampled_textures\": {},",
            self.max_sampled_textures
        );
        let _ = writeln!(out, "{indent}  \"max_samplers\": {},", self.max_samplers);
        let _ = writeln!(
            out,
            "{indent}  \"dead_bindings_pruned\": {},",
            self.total_dead_bindings_pruned
        );
        let _ = writeln!(
            out,
            "{indent}  \"duplicate_bindings_folded\": {},",
            self.total_duplicate_bindings_folded
        );
        let _ = writeln!(
            out,
            "{indent}  \"cbv_merged_into_storage\": {},",
            self.total_cbv_merged_into_storage
        );
        let _ = writeln!(
            out,
            "{indent}  \"uniform_promoted_to_storage\": {}",
            self.total_uniform_promoted_to_storage
        );
        let _ = write!(out, "{indent}}}");
    }

    /// Append strictly sanitized plain-text binding-limit diagnostics to `out`.
    pub fn write_text(&self, out: &mut String) {
        let _ = writeln!(
            out,
            "  d3d12_limits_ok:   {} (over: {}, ub: {}, sb: {}, st: {}, tex: {}, smp: {})",
            self.stages_within_limits,
            self.stages_over_limit,
            self.over_uniform_buffers,
            self.over_storage_buffers,
            self.over_storage_textures,
            self.over_sampled_textures,
            self.over_samplers
        );
        let _ = writeln!(
            out,
            "  binding_pack:      pruned={}, deduped={}, cbv_merged={}, ub_promoted={}",
            self.total_dead_bindings_pruned,
            self.total_duplicate_bindings_folded,
            self.total_cbv_merged_into_storage,
            self.total_uniform_promoted_to_storage
        );
    }
}

// ---------------------------------------------------------------------------
// Canonical binding-source normalization
// ---------------------------------------------------------------------------

/// Canonicalize a [`CbvIndexedHeapSource`] for alias comparison.
///
/// Matches `runtime/gpu/stage_reflection.cpp`: when `stride == (1 << shift)` (with `shift < 32`),
/// `stride` is equivalent to `0` (shift-only power-of-two scaling).
pub const fn canonicalize_cbv_indexed_heap(mut h: CbvIndexedHeapSource) -> CbvIndexedHeapSource {
    if h.shift < 32 && h.stride == (1u32 << h.shift) {
        h.stride = 0;
    }
    h
}

/// Canonicalize a [`LaneIndexedHeapSource`] for alias comparison.
///
/// Matches `runtime/gpu/stage_reflection.cpp`: when `stride == (1 << shift)` (with `shift < 32`),
/// `stride` is equivalent to `0` (shift-only power-of-two scaling).
pub const fn canonicalize_lane_indexed_heap(mut h: LaneIndexedHeapSource) -> LaneIndexedHeapSource {
    if h.shift < 32 && h.stride == (1u32 << h.shift) {
        h.stride = 0;
    }
    h
}

/// Canonicalize a [`WgslBindingSource`] so semantically identical console descriptor origins
/// compare equal for deduplication and cross-kind CBV/storage folding:
/// * Power-of-two `stride == (1 << shift)` in `CbvIndexed*Heap` / `LaneIndexed*Heap` normalizes
///   to `stride = 0` (exact runtime equivalence in `stage_reflection.cpp`).
/// * `RootDescriptor { user_sgpr, table_byte_offset: Some(0) }` normalizes to
///   `RootDescriptor { user_sgpr, table_byte_offset: None }`.
/// * When `user_sgpr` is proven to be a descriptor table pointer in the same stage (present in
///   `known_table_sgprs`), `RootDescriptor { user_sgpr, table_byte_offset }` normalizes to
///   `Table { user_sgpr, table_byte_offset: table_byte_offset.unwrap_or(0) }`.
pub fn canonicalize_binding_source(
    src: WgslBindingSource,
    known_table_sgprs: &BTreeSet<u32>,
) -> WgslBindingSource {
    match src {
        WgslBindingSource::RootDescriptor {
            user_sgpr,
            table_byte_offset,
        } => {
            if known_table_sgprs.contains(&user_sgpr) {
                WgslBindingSource::Table {
                    user_sgpr,
                    table_byte_offset: table_byte_offset.unwrap_or(0),
                }
            } else {
                let norm_off = table_byte_offset.filter(|&off| off != 0);
                WgslBindingSource::RootDescriptor {
                    user_sgpr,
                    table_byte_offset: norm_off,
                }
            }
        }
        WgslBindingSource::CbvIndexedResourceHeap(h) => {
            WgslBindingSource::CbvIndexedResourceHeap(canonicalize_cbv_indexed_heap(h))
        }
        WgslBindingSource::CbvIndexedSamplerHeap(h) => {
            WgslBindingSource::CbvIndexedSamplerHeap(canonicalize_cbv_indexed_heap(h))
        }
        WgslBindingSource::LaneIndexedResourceHeap(h) => {
            WgslBindingSource::LaneIndexedResourceHeap(canonicalize_lane_indexed_heap(h))
        }
        WgslBindingSource::LaneIndexedSamplerHeap(h) => {
            WgslBindingSource::LaneIndexedSamplerHeap(canonicalize_lane_indexed_heap(h))
        }
        other => other,
    }
}

/// Collect all user-data SGPRs proven to hold descriptor-table pointers from a slice of bindings.
pub fn collect_known_table_sgprs(bindings: &[WgslBindingInfo]) -> BTreeSet<u32> {
    let mut set = BTreeSet::new();
    for b in bindings {
        if let Some(WgslBindingSource::Table { user_sgpr, .. }) = b.source {
            set.insert(user_sgpr);
        }
    }
    set
}

// ---------------------------------------------------------------------------
// Naga IR reachability, dead-binding pruning & duplicate-binding folding
// ---------------------------------------------------------------------------

fn mark_expr_globals(
    expr_handle: Handle<Expression>,
    func: &Function,
    visited_exprs: &mut HashSet<Handle<Expression>>,
    referenced_globals: &mut HashSet<Handle<GlobalVariable>>,
) {
    let mut stack = vec![expr_handle];
    while let Some(h) = stack.pop() {
        if !visited_exprs.insert(h) {
            continue;
        }
        let Some(expr) = func.expressions.try_get(h).ok() else {
            continue;
        };
        match expr {
            Expression::GlobalVariable(g) => {
                referenced_globals.insert(*g);
            }
            Expression::Access { base, index } => {
                stack.push(*base);
                stack.push(*index);
            }
            Expression::AccessIndex { base, .. }
            | Expression::Splat { value: base, .. }
            | Expression::Swizzle { vector: base, .. }
            | Expression::Load { pointer: base }
            | Expression::Unary { expr: base, .. }
            | Expression::As { expr: base, .. }
            | Expression::Derivative { expr: base, .. }
            | Expression::ArrayLength(base)
            | Expression::RayQueryGetIntersection { query: base, .. }
            | Expression::RayQueryVertexPositions { query: base, .. } => {
                stack.push(*base);
            }
            Expression::Compose { components, .. } => {
                stack.extend(components.iter().copied());
            }
            Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                ..
            } => {
                stack.push(*image);
                stack.push(*sampler);
                stack.push(*coordinate);
                if let Some(a) = array_index {
                    stack.push(*a);
                }
                if let Some(o) = offset {
                    stack.push(*o);
                }
                match level {
                    naga::SampleLevel::Exact(e) | naga::SampleLevel::Bias(e) => stack.push(*e),
                    naga::SampleLevel::Gradient { x, y } => {
                        stack.push(*x);
                        stack.push(*y);
                    }
                    naga::SampleLevel::Auto | naga::SampleLevel::Zero => {}
                }
                if let Some(d) = depth_ref {
                    stack.push(*d);
                }
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                stack.push(*image);
                stack.push(*coordinate);
                if let Some(a) = array_index {
                    stack.push(*a);
                }
                if let Some(s) = sample {
                    stack.push(*s);
                }
                if let Some(l) = level {
                    stack.push(*l);
                }
            }
            Expression::ImageQuery { image, query } => {
                stack.push(*image);
                if let naga::ImageQuery::Size { level: Some(l) } = query {
                    stack.push(*l);
                }
            }
            Expression::Binary { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                stack.push(*condition);
                stack.push(*accept);
                stack.push(*reject);
            }
            Expression::Relational { argument, .. } => {
                stack.push(*argument);
            }
            Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                stack.push(*arg);
                if let Some(a1) = arg1 {
                    stack.push(*a1);
                }
                if let Some(a2) = arg2 {
                    stack.push(*a2);
                }
                if let Some(a3) = arg3 {
                    stack.push(*a3);
                }
            }
            Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::Override(_)
            | Expression::ZeroValue(_)
            | Expression::FunctionArgument(_)
            | Expression::LocalVariable(_)
            | Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. }
            | Expression::RayQueryProceedResult
            | Expression::CooperativeLoad { .. }
            | Expression::CooperativeMultiplyAdd { .. } => {}
        }
    }
}

fn mark_block_globals(
    block: &Block,
    func: &Function,
    visited_exprs: &mut HashSet<Handle<Expression>>,
    referenced_globals: &mut HashSet<Handle<GlobalVariable>>,
    called_functions: &mut Vec<Handle<Function>>,
) {
    for stmt in block.iter() {
        match stmt {
            Statement::Emit(range) => {
                for h in range.clone() {
                    mark_expr_globals(h, func, visited_exprs, referenced_globals);
                }
            }
            Statement::Block(inner) => {
                mark_block_globals(
                    inner,
                    func,
                    visited_exprs,
                    referenced_globals,
                    called_functions,
                );
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                mark_expr_globals(*condition, func, visited_exprs, referenced_globals);
                mark_block_globals(
                    accept,
                    func,
                    visited_exprs,
                    referenced_globals,
                    called_functions,
                );
                mark_block_globals(
                    reject,
                    func,
                    visited_exprs,
                    referenced_globals,
                    called_functions,
                );
            }
            Statement::Switch { selector, cases } => {
                mark_expr_globals(*selector, func, visited_exprs, referenced_globals);
                for case in cases {
                    mark_block_globals(
                        &case.body,
                        func,
                        visited_exprs,
                        referenced_globals,
                        called_functions,
                    );
                }
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                mark_block_globals(
                    body,
                    func,
                    visited_exprs,
                    referenced_globals,
                    called_functions,
                );
                mark_block_globals(
                    continuing,
                    func,
                    visited_exprs,
                    referenced_globals,
                    called_functions,
                );
                if let Some(cond) = break_if {
                    mark_expr_globals(*cond, func, visited_exprs, referenced_globals);
                }
            }
            Statement::Return { value } => {
                if let Some(v) = value {
                    mark_expr_globals(*v, func, visited_exprs, referenced_globals);
                }
            }
            Statement::Store { pointer, value } => {
                mark_expr_globals(*pointer, func, visited_exprs, referenced_globals);
                mark_expr_globals(*value, func, visited_exprs, referenced_globals);
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                mark_expr_globals(*image, func, visited_exprs, referenced_globals);
                mark_expr_globals(*coordinate, func, visited_exprs, referenced_globals);
                if let Some(a) = array_index {
                    mark_expr_globals(*a, func, visited_exprs, referenced_globals);
                }
                mark_expr_globals(*value, func, visited_exprs, referenced_globals);
            }
            Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                value,
                ..
            } => {
                mark_expr_globals(*image, func, visited_exprs, referenced_globals);
                mark_expr_globals(*coordinate, func, visited_exprs, referenced_globals);
                if let Some(a) = array_index {
                    mark_expr_globals(*a, func, visited_exprs, referenced_globals);
                }
                mark_expr_globals(*value, func, visited_exprs, referenced_globals);
            }
            Statement::Atomic {
                pointer,
                fun,
                value,
                ..
            } => {
                mark_expr_globals(*pointer, func, visited_exprs, referenced_globals);
                mark_expr_globals(*value, func, visited_exprs, referenced_globals);
                if let naga::AtomicFunction::Exchange {
                    compare: Some(cmp),
                } = fun
                {
                    mark_expr_globals(*cmp, func, visited_exprs, referenced_globals);
                }
            }
            Statement::WorkGroupUniformLoad { pointer, .. } => {
                mark_expr_globals(*pointer, func, visited_exprs, referenced_globals);
            }
            Statement::Call {
                function,
                arguments,
                ..
            } => {
                called_functions.push(*function);
                for arg in arguments {
                    mark_expr_globals(*arg, func, visited_exprs, referenced_globals);
                }
            }
            Statement::RayQuery { query, fun } => {
                mark_expr_globals(*query, func, visited_exprs, referenced_globals);
                if let naga::RayQueryFunction::Initialize {
                    acceleration_structure,
                    descriptor,
                } = fun
                {
                    mark_expr_globals(
                        *acceleration_structure,
                        func,
                        visited_exprs,
                        referenced_globals,
                    );
                    mark_expr_globals(*descriptor, func, visited_exprs, referenced_globals);
                }
            }
            Statement::SubgroupBallot { predicate, .. } => {
                if let Some(p) = predicate {
                    mark_expr_globals(*p, func, visited_exprs, referenced_globals);
                }
            }
            Statement::SubgroupCollectiveOperation { argument, .. } => {
                mark_expr_globals(*argument, func, visited_exprs, referenced_globals);
            }
            Statement::SubgroupGather {
                mode, argument, ..
            } => {
                mark_expr_globals(*argument, func, visited_exprs, referenced_globals);
                match mode {
                    naga::GatherMode::Broadcast(idx)
                    | naga::GatherMode::Shuffle(idx)
                    | naga::GatherMode::ShuffleDown(idx)
                    | naga::GatherMode::ShuffleUp(idx)
                    | naga::GatherMode::ShuffleXor(idx)
                    | naga::GatherMode::QuadBroadcast(idx) => {
                        mark_expr_globals(*idx, func, visited_exprs, referenced_globals);
                    }
                    naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
                }
            }
            Statement::CooperativeStore { target, data } => {
                mark_expr_globals(*target, func, visited_exprs, referenced_globals);
                mark_expr_globals(data.pointer, func, visited_exprs, referenced_globals);
                mark_expr_globals(data.stride, func, visited_exprs, referenced_globals);
            }
            Statement::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
                acceleration_structure,
                descriptor,
                payload,
            }) => {
                mark_expr_globals(
                    *acceleration_structure,
                    func,
                    visited_exprs,
                    referenced_globals,
                );
                mark_expr_globals(*descriptor, func, visited_exprs, referenced_globals);
                mark_expr_globals(*payload, func, visited_exprs, referenced_globals);
            }
            Statement::Break
            | Statement::Continue
            | Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_) => {}
        }
    }
}

/// Collect the exact set of [`GlobalVariable`] handles transitively referenced by all entry points
/// and their reachable callee functions in `module`.
pub fn collect_referenced_globals(module: &Module) -> HashSet<Handle<GlobalVariable>> {
    let mut referenced = HashSet::new();
    let mut visited_funcs: HashSet<Handle<Function>> = HashSet::new();
    let mut worklist: Vec<Handle<Function>> = Vec::new();

    for ep in &module.entry_points {
        let mut visited_exprs = HashSet::new();
        mark_block_globals(
            &ep.function.body,
            &ep.function,
            &mut visited_exprs,
            &mut referenced,
            &mut worklist,
        );
    }

    while let Some(fh) = worklist.pop() {
        if !visited_funcs.insert(fh) {
            continue;
        }
        if let Ok(func) = module.functions.try_get(fh) {
            let mut visited_exprs = HashSet::new();
            mark_block_globals(
                &func.body,
                func,
                &mut visited_exprs,
                &mut referenced,
                &mut worklist,
            );
        }
    }

    referenced
}

/// Rebuild `module.global_variables`, retaining only globals mapped to `Some(target_handle)` (or
/// redirecting aliases to another surviving handle) and remapping every
/// `Expression::GlobalVariable` across all entry points and functions.
fn rebuild_globals_with_remap(
    module: &mut Module,
    keep_or_redirect: &HashMap<Handle<GlobalVariable>, Option<Handle<GlobalVariable>>>,
) {
    let mut new_globals: Arena<GlobalVariable> = Arena::new();
    let mut direct_map: HashMap<Handle<GlobalVariable>, Handle<GlobalVariable>> = HashMap::new();

    // Pass 1: append every kept (non-redirected, non-removed) global variable.
    for (old_handle, var) in module.global_variables.iter() {
        match keep_or_redirect.get(&old_handle) {
            Some(Some(target)) if *target != old_handle => {
                // Redirected to another surviving handle; resolved in Pass 2.
            }
            Some(None) => {
                // Pruned.
            }
            _ => {
                let new_h = new_globals.append(var.clone(), Span::UNDEFINED);
                direct_map.insert(old_handle, new_h);
            }
        }
    }

    // Pass 2: build full old_handle -> new_handle map including aliases.
    let mut full_map = direct_map.clone();
    for (&old_handle, mapping) in keep_or_redirect {
        if let Some(target) = mapping {
            if let Some(&new_target) = direct_map.get(target) {
                full_map.insert(old_handle, new_target);
            }
        }
    }

    let remap_expr_arena = |exprs: &mut Arena<Expression>| {
        for (_, expr) in exprs.iter_mut() {
            if let Expression::GlobalVariable(ref mut h) = *expr {
                if let Some(&mapped) = full_map.get(h) {
                    *h = mapped;
                }
            }
        }
    };

    for ep in &mut module.entry_points {
        remap_expr_arena(&mut ep.function.expressions);
    }
    for (_, func) in module.functions.iter_mut() {
        remap_expr_arena(&mut func.expressions);
    }
    remap_expr_arena(&mut module.global_expressions);

    module.global_variables = new_globals;
}

/// Synchronize `bindings` to contain only `(group, binding)` pairs still declared in `module`.
fn sync_bindings_with_module(module: &Module, bindings: &mut Vec<WgslBindingInfo>) {
    let live_slots: BTreeSet<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, var)| var.binding.map(|rb| (rb.group, rb.binding)))
        .collect();
    bindings.retain(|b| live_slots.contains(&(b.group, b.binding)));
}

/// Prune unreferenced `@group/@binding` resource declarations from `module` and `bindings`.
///
/// Never removes:
/// * Non-binding globals (e.g. `var<workgroup> lds`).
/// * Runtime-owned group-3 bindings (`@group(3)` such as `ms_out` or `rt_arena`).
/// * Slots explicitly listed in `preserve_slots`.
///
/// Returns the number of pruned resource bindings.
pub fn prune_unused_module_bindings(
    module: &mut Module,
    bindings: &mut Vec<WgslBindingInfo>,
    preserve_slots: &BTreeSet<(u32, u32)>,
) -> usize {
    let referenced = collect_referenced_globals(module);
    let mut keep_or_redirect: HashMap<Handle<GlobalVariable>, Option<Handle<GlobalVariable>>> =
        HashMap::new();
    let mut pruned = 0usize;

    for (h, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else {
            continue;
        };
        if rb.group == RUNTIME_OWNED_GROUP || preserve_slots.contains(&(rb.group, rb.binding)) {
            continue;
        }
        if !referenced.contains(&h) {
            keep_or_redirect.insert(h, None);
            pruned += 1;
        }
    }

    if pruned > 0 {
        rebuild_globals_with_remap(module, &keep_or_redirect);
        sync_bindings_with_module(module, bindings);
    }
    pruned
}

/// Deduplication key for compatible bound [`GlobalVariable`] declarations in a [`naga::Module`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ModuleDedupKey {
    Uniform {
        source: WgslBindingSource,
        elem_is_vec4_u32_struct: bool,
    },
    Storage {
        source: WgslBindingSource,
        elem_type: Handle<naga::Type>,
    },
    SampledImage {
        source: WgslBindingSource,
        dim: naga::ImageDimension,
        arrayed: bool,
        class: ImageClass,
    },
    StorageImage {
        source: WgslBindingSource,
        dim: naga::ImageDimension,
        arrayed: bool,
        format: naga::StorageFormat,
    },
    Sampler {
        source: WgslBindingSource,
        comparison: bool,
    },
}

fn is_rdna2_cbv_vec4_u32_struct(module: &Module, ty: Handle<naga::Type>) -> Option<u32> {
    let TypeInner::Struct { members, .. } = &module.types[ty].inner else {
        return None;
    };
    if members.len() != 1 || members[0].offset != 0 {
        return None;
    }
    let TypeInner::Array {
        base,
        size: naga::ArraySize::Constant(count),
        stride: 16,
    } = module.types[members[0].ty].inner
    else {
        return None;
    };
    match module.types[base].inner {
        TypeInner::Vector {
            size: naga::VectorSize::Quad,
            scalar: naga::Scalar::U32,
        } => Some(count.get()),
        _ => None,
    }
}

/// Deduplicate bound `@group/@binding` globals in `module` (and matching `bindings`) that refer
/// to the exact same canonical [`WgslBindingSource`] and have compatible WGSL resource types.
///
/// * Never merges unresolved (`source: None`) bindings.
/// * Never merges runtime-owned group-3 bindings.
/// * For duplicate storage buffers with the same source and element type, upgrades the surviving
///   global to `LOAD | STORE` (`kind: "storage"`) if either copy is writable.
/// * For duplicate `Rdna2Cbv` uniform structs with the same source, keeps the larger `vec4`
///   array type and max `min_binding_size_bytes` / `highest_vec4_offset`.
///
/// Returns the number of folded duplicate bindings.
pub fn deduplicate_module_bindings(
    module: &mut Module,
    bindings: &mut Vec<WgslBindingInfo>,
) -> usize {
    let known_table_sgprs = collect_known_table_sgprs(bindings);
    let mut binding_idx_by_slot: HashMap<(u32, u32), usize> = HashMap::new();
    for (idx, b) in bindings.iter().enumerate() {
        binding_idx_by_slot.insert((b.group, b.binding), idx);
    }

    // Canonicalize order by ascending `(group, binding)` so primary selection is deterministic.
    let mut bound_globals: Vec<((u32, u32), Handle<GlobalVariable>)> = module
        .global_variables
        .iter()
        .filter_map(|(h, var)| {
            let rb = var.binding?;
            if rb.group == RUNTIME_OWNED_GROUP {
                return None;
            }
            Some(((rb.group, rb.binding), h))
        })
        .collect();
    bound_globals.sort_by_key(|&(slot, _)| slot);

    let mut primary_by_key: BTreeMap<ModuleDedupKey, (Handle<GlobalVariable>, usize)> =
        BTreeMap::new();
    let mut keep_or_redirect: HashMap<Handle<GlobalVariable>, Option<Handle<GlobalVariable>>> =
        HashMap::new();
    let mut upgrade_storage_access: HashMap<Handle<GlobalVariable>, StorageAccess> = HashMap::new();
    let mut upgrade_uniform_type: HashMap<Handle<GlobalVariable>, Handle<naga::Type>> =
        HashMap::new();
    let mut folded = 0usize;

    for (slot, handle) in bound_globals {
        let Some(&b_idx) = binding_idx_by_slot.get(&slot) else {
            continue;
        };
        let Some(raw_src) = bindings[b_idx].source else {
            continue;
        };
        let canon_src = canonicalize_binding_source(raw_src, &known_table_sgprs);
        let var = &module.global_variables[handle];
        let dedup_key = match var.space {
            AddressSpace::Uniform => {
                if is_rdna2_cbv_vec4_u32_struct(module, var.ty).is_some() {
                    Some(ModuleDedupKey::Uniform {
                        source: canon_src,
                        elem_is_vec4_u32_struct: true,
                    })
                } else {
                    None
                }
            }
            AddressSpace::Storage { .. } => Some(ModuleDedupKey::Storage {
                source: canon_src,
                elem_type: var.ty,
            }),
            AddressSpace::Handle => match module.types[var.ty].inner {
                TypeInner::Image {
                    dim,
                    arrayed,
                    class: ImageClass::Storage { format, .. },
                } => Some(ModuleDedupKey::StorageImage {
                    source: canon_src,
                    dim,
                    arrayed,
                    format,
                }),
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => Some(ModuleDedupKey::SampledImage {
                    source: canon_src,
                    dim,
                    arrayed,
                    class,
                }),
                TypeInner::Sampler { comparison } => Some(ModuleDedupKey::Sampler {
                    source: canon_src,
                    comparison,
                }),
                _ => None,
            },
            _ => None,
        };

        let Some(key) = dedup_key else {
            continue;
        };

        if let Some(&(primary_h, primary_b_idx)) = primary_by_key.get(&key) {
            keep_or_redirect.insert(handle, Some(primary_h));
            folded += 1;

            // Merge metadata & type/access requirements onto the surviving primary binding.
            match key {
                ModuleDedupKey::Storage { .. } => {
                    let dup_access = match module.global_variables[handle].space {
                        AddressSpace::Storage { access } => access,
                        _ => StorageAccess::LOAD,
                    };
                    let cur_access = *upgrade_storage_access.entry(primary_h).or_insert_with(|| {
                        match module.global_variables[primary_h].space {
                            AddressSpace::Storage { access } => access,
                            _ => StorageAccess::LOAD,
                        }
                    });
                    let merged = cur_access | dup_access;
                    upgrade_storage_access.insert(primary_h, merged);
                    if merged.contains(StorageAccess::STORE) {
                        bindings[primary_b_idx].kind = "storage";
                    }
                }
                ModuleDedupKey::Uniform { .. } => {
                    let cur_ty = *upgrade_uniform_type
                        .get(&primary_h)
                        .unwrap_or(&module.global_variables[primary_h].ty);
                    let cur_cnt = is_rdna2_cbv_vec4_u32_struct(module, cur_ty).unwrap_or(0);
                    let dup_ty = module.global_variables[handle].ty;
                    let dup_cnt = is_rdna2_cbv_vec4_u32_struct(module, dup_ty).unwrap_or(0);
                    if dup_cnt > cur_cnt {
                        upgrade_uniform_type.insert(primary_h, dup_ty);
                    }
                    let dup_min = bindings[b_idx].min_binding_size_bytes;
                    let dup_high = bindings[b_idx].highest_vec4_offset;
                    if let Some(d_min) = dup_min {
                        let p_min = bindings[primary_b_idx].min_binding_size_bytes.unwrap_or(0);
                        bindings[primary_b_idx].min_binding_size_bytes = Some(p_min.max(d_min));
                    }
                    if let Some(d_hi) = dup_high {
                        let p_hi = bindings[primary_b_idx].highest_vec4_offset.unwrap_or(0);
                        bindings[primary_b_idx].highest_vec4_offset = Some(p_hi.max(d_hi));
                    }
                }
                _ => {}
            }
        } else {
            primary_by_key.insert(key, (handle, b_idx));
        }
    }

    if folded > 0 {
        for (h, access) in upgrade_storage_access {
            if let AddressSpace::Storage {
                access: ref mut cur,
            } = module.global_variables[h].space
            {
                *cur = access;
            }
        }
        for (h, ty) in upgrade_uniform_type {
            module.global_variables[h].ty = ty;
        }
        rebuild_globals_with_remap(module, &keep_or_redirect);
        sync_bindings_with_module(module, bindings);
    }

    folded
}

/// Run the full post-lowering binding compaction pipeline on `module` and `bindings`:
/// 1. Deduplicate any bound globals sharing an identical canonical descriptor source and type.
/// 2. Prune unreferenced bound globals (excluding `preserve_when_within_limits` while within limits).
/// 3. If the stage still exceeds `limits.max_storage_buffers` and has unreferenced bindings in
///    `preserve_when_within_limits` (e.g., unlowered `buffer_atomic_*`-only UAV placeholders),
///    prune those unreferenced placeholders from highest binding downward until the stage fits
///    within `limits.max_storage_buffers` (or none remain).
pub fn compact_module_bindings(
    module: &mut Module,
    bindings: &mut Vec<WgslBindingInfo>,
    rt_arena: bool,
    has_mesh_output: bool,
    preserve_when_within_limits: &BTreeSet<(u32, u32)>,
    limits: StageBindingLimits,
) -> BindingPackStats {
    let mut stats = BindingPackStats::default();
    stats.duplicate_bindings_folded += deduplicate_module_bindings(module, bindings);
    stats.dead_bindings_pruned +=
        prune_unused_module_bindings(module, bindings, preserve_when_within_limits);

    if !preserve_when_within_limits.is_empty() {
        let counts = StageBindingCounts::from_wgsl_bindings(bindings, rt_arena, has_mesh_output);
        if counts.storage_buffers > limits.max_storage_buffers {
            let excess = counts.storage_buffers - limits.max_storage_buffers;
            let referenced = collect_referenced_globals(module);
            let mut candidates: Vec<((u32, u32), Handle<GlobalVariable>)> = module
                .global_variables
                .iter()
                .filter_map(|(h, var)| {
                    let rb = var.binding?;
                    let slot = (rb.group, rb.binding);
                    if preserve_when_within_limits.contains(&slot) && !referenced.contains(&h) {
                        Some((slot, h))
                    } else {
                        None
                    }
                })
                .collect();
            // Prune highest binding indices first for determinism.
            candidates.sort_by(|a, b| b.0.cmp(&a.0));
            let mut keep_or_redirect: HashMap<
                Handle<GlobalVariable>,
                Option<Handle<GlobalVariable>>,
            > = HashMap::new();
            for (_, h) in candidates.into_iter().take(excess) {
                keep_or_redirect.insert(h, None);
                stats.dead_bindings_pruned += 1;
            }
            if !keep_or_redirect.is_empty() {
                rebuild_globals_with_remap(module, &keep_or_redirect);
                sync_bindings_with_module(module, bindings);
            }
        }
    }

    stats
}
