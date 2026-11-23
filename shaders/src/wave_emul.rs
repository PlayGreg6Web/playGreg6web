//! Wave64 / subgroup emulation and mesh-compute WGSL lowering library (task S5).
//!
//! Xbox Series shaders are compiled with `--wave64-emulate` (HLSL wave intrinsics assume a fixed
//! 64-lane wave) and `--mesh-compute` (mesh/amplification shaders lowered to compute). After the
//! SPIR-V translation they show up as `OpGroupNonUniform*` instructions, the `SubgroupSize` /
//! `SubgroupLocalInvocationId` built-ins, 4 x `u32` ballot masks, and mesh/task execution models.
//! Baseline WebGPU has none of that: `subgroups` is an optional feature whose subgroup size is
//! not guaranteed (and usually below 64), WGSL has no 64-bit integers, and there are no mesh
//! stages. This module provides the lowering building blocks. It never produces a silently wrong
//! shader: every wave use is either lowered under a named [`WaveStrategy`] with its
//! [`LoweringCondition`] recorded in the report, or reported as a [`WaveBlocker`] (in which case
//! the module is left untouched).
//!
//! # Levels and why
//!
//! The work is split across the three representations the converter (`convert.rs`) has:
//!
//! 1. **SPIR-V words, before `naga` parses the module** (same place and style as S3's
//!    [`crate::compat::sanitize_spirv_words`]):
//!    - [`scan_wave_usage_spirv`] classifies every wave/subgroup and mesh/task construct per entry
//!      point (attributed through the `OpFunctionCall` graph). This level is the only one that
//!      sees instructions naga 30's SPIR-V front end rejects (`OpGroupNonUniformElect`,
//!      `...BallotBitCount`, `...AllEqual`, lane masks, mesh execution models, ...), so it is
//!      where the usage report is built.
//!    - [`canonicalize_wave_spirv`] rewrites the two common naga-unsupported intrinsics into exact
//!      equivalents naga can parse: `WaveIsFirstLane` (`Elect` -> `lane == BroadcastFirst(lane)`)
//!      and `WaveActiveCountBits` / `WavePrefixCountBits` (`BallotBitCount` -> `countOneBits` of
//!      the 4 mask words, masked with a lane mask built from `SubgroupLocalInvocationId` for the
//!      scans). Both rewrites are strategy-independent identities, so the strategy is chosen once,
//!      later, on the IR.
//! 2. **naga IR, after parsing** (same place as S4's `lower_module_for_webgpu`):
//!    [`lower_wave_ops`] rewrites `Statement::Subgroup*` and the subgroup built-in entry-point
//!    arguments. Each rewritten statement becomes a `Statement::Call` to a generated helper
//!    function whose `CallResult` takes over the original result handle, so no expression handle
//!    of the input module changes meaning. This level has types (from the validator's
//!    `ModuleInfo`), the call graph and structured control flow, which the soundness checks need,
//!    and its output goes through the converter's existing validator and WGSL writer unchanged.
//! 3. **WGSL source text** for standalone, injectable libraries: the `vec2<u32>` 64-bit mask
//!    helpers ([`u64_helpers_wgsl`]) and the mesh-compute output declarations / allocation /
//!    indirect-draw helpers ([`mesh_output_wgsl`]). They are prefixed (see [`unique_wgsl_prefix`])
//!    so they can be appended to a converted module without name collisions, and only the
//!    helpers actually requested are emitted.
//!
//! # Strategies and conditions
//!
//! [`LoweringCondition`] names *why* a lowering is correct:
//!
//! - `Exact`: identical to the Wave64 result for every input.
//! - `UniformOperand`: the operation is idempotent over identical lane values (`min`, `max`,
//!   `and`, `or`, `all`, `any`, every broadcast/shuffle/read-lane) and the operand was proven
//!   uniform by this module's conservative syntactic analysis (literals, constants, uniform
//!   buffers, read-only storage, `workgroup_id`/`num_workgroups`, and pure arithmetic over them),
//!   so the result is the operand itself. Valid under every strategy and every wave size.
//! - `WorkgroupUniformControlFlow`: workgroup-shared-memory emulation (see below) whose
//!   preconditions were proven structurally.
//! - `RuntimeSubgroupSize64`: native WGSL `subgroups`; exact only if the adapter reports
//!   `subgroupMinSize == subgroupMaxSize == 64`, otherwise the shader computes over the real,
//!   smaller subgroup (approximation; the runtime must check and fall back).
//! - `Wave1Model`: approximation that models every invocation as a wave of one lane. It is
//!   self-consistent (lane count 1, lane index 0, ballot = own bit 0, reductions = own value,
//!   exclusive scans = identity) and correct only for shaders that are correct for *any* wave
//!   size; it is used only when [`WaveLoweringOptions::allow_wave1_model`] is set.
//!
//! Per-intrinsic behaviour ("blocker" = reported, module left unchanged):
//!
//! | HLSL intrinsic | naga IR | `SingleLane` | `NativeSubgroups` | `WorkgroupEmulation` (compute only) |
//! | :--- | :--- | :--- | :--- | :--- |
//! | `WaveGetLaneCount` | `BuiltIn::SubgroupSize` | `1` (Wave1Model) | `subgroup_size` (RuntimeSubgroupSize64) | `64` (Exact, any stage) |
//! | `WaveGetLaneIndex` | `BuiltIn::SubgroupInvocationId` | `0` (Wave1Model) | native (RuntimeSubgroupSize64) | `local_invocation_index & 63` (Exact) |
//! | `WaveIsFirstLane` | canonicalized to lane index + `BroadcastFirst` | `true` (Wave1Model) | native | Exact under uniform CF |
//! | `WaveActiveAllTrue/AnyTrue/Min/Max/BitAnd/BitOr` | `SubgroupCollectiveOperation` | operand if uniform (UniformOperand), else operand (Wave1Model) | native | uniform operand: UniformOperand; else WorkgroupUniformControlFlow |
//! | `WaveActiveSum/Product/BitXor` | `SubgroupCollectiveOperation` | operand (Wave1Model only) | native | WorkgroupUniformControlFlow |
//! | `WavePrefixSum/Product` (exclusive scans) | `SubgroupCollectiveOperation` | identity element `0`/`1` (Wave1Model) | native | WorkgroupUniformControlFlow; scans other than add/mul (legal SPIR-V) are rejected by naga's validator: module-level blocker |
//! | `WaveActiveBallot` | `SubgroupBallot` | `vec4(u32(p), 0, 0, 0)` (Wave1Model) | native | WorkgroupUniformControlFlow |
//! | `WaveActiveCountBits`, `WavePrefixCountBits` | canonicalized to `countOneBits` (+ lane mask) | follows ballot/lane index | follows | follows |
//! | `WaveReadLaneFirst`, `WaveReadLaneAt`, shuffles | `SubgroupGather` | operand (UniformOperand or Wave1Model) | native | WorkgroupUniformControlFlow; out-of-range/inactive source lanes return the caller's own value (SPIR-V leaves them undefined) |
//! | `QuadReadAcross*`, `QuadReadLaneAt` | `SubgroupGather` (quad) | uniform operand only, else blocker (a 1-lane quad would silently zero screen-space differences) | fragment: Exact; compute: RuntimeSubgroupSize64; vertex: blocker | blocker (D3D compute quad layout is not tied to `local_invocation_index`) |
//! | `WaveActiveAllEqual`, `WaveMatch`, `WaveMultiPrefix*`, clustered/rotate/quad-vote, lane masks | rejected by naga 30's SPIR-V front end | blocker (reported by [`scan_wave_usage_spirv`]) | blocker | blocker |
//!
//! Quad *derivatives* (`ddx`/`ddy`) are not wave operations and are never touched.
//!
//! ## Workgroup emulation soundness
//!
//! A wave is mapped to 64 consecutive `local_invocation_index` values (wave `w` = invocations
//! `64w .. 64w+63`; the last wave of a workgroup whose size is not a multiple of 64 is partial,
//! exactly like hardware where those lanes do not exist). D3D does not specify how threads are
//! packed into waves, so any program correct under D3D semantics is correct under this mapping
//! provided every lane of each wave is *active* at the operation. The helpers exchange values
//! through a `var<workgroup>` scratch array between two `workgroupBarrier()`s, which is only
//! valid when **every invocation of the workgroup reaches the call the same number of times**.
//! naga 30's validator does not check barrier uniformity (it only checks derivatives), so this
//! module proves it itself, conservatively:
//!
//! - the op's statement is not inside any `loop`;
//! - every enclosing `if`/`switch` branches on a value proven uniform by the analysis above;
//! - no `return`/`discard`/`break`/`continue` in non-uniform control flow precedes it;
//! - every call site on every call path from the entry point satisfies the same rules;
//! - every entry point reaching it is a compute entry point with a fixed workgroup size.
//!
//! These rules are meant to be no more permissive than the WGSL uniformity analysis browsers run
//! (workgroup-memory loads, which naga treats as uniform but WGSL does not, are non-uniform here),
//! so the barriers should also pass Tint/Dawn; that has not been checked against a browser
//! compiler. The helper loops always run a constant 64 iterations (data-dependent work is
//! predicated), so the helpers themselves contain no non-uniform control flow before a
//! barrier. Floating-point reductions combine lanes in ascending lane order for every lane, so all
//! lanes of a wave see bit-identical results; the order may differ from AMD hardware, which D3D
//! leaves unspecified.
//!
//! # 64-bit masks
//!
//! WGSL has no `u64`; [`u64_helpers_wgsl`] emits `vec2<u32>` (`x` = low word, `y` = high word)
//! helpers for ballot/mask code: and/or/xor/not, shifts by any amount (`0`, `>= 32`, `>= 64`
//! yields zero), `countOneBits`, `firstTrailingBit`/`firstLeadingBit` (`0xffffffff` for zero, like
//! WGSL), equality, unsigned less-than, add with carry, the lane mask `(1 << n) - 1` (including
//! `n == 64`), and bit test.
//!
//! # Mesh-compute output
//!
//! A `--mesh-compute` mesh shader runs as one compute workgroup per meshlet. [`mesh_output_layout`]
//! and [`mesh_output_wgsl`] describe and declare its outputs from a [`MeshOutputSignature`]:
//!
//! - **vertices**: `array<Vertex>` storage buffer, meshlet `m` owns slots
//!   `m * max_vertices .. (m + 1) * max_vertices`; `Vertex` follows WGSL host-shareable layout
//!   rules (vec3 aligned to 16), which also satisfies `GPUVertexBufferLayout` rules, so the buffer
//!   can be bound as a vertex buffer ([`MeshStructLayout::vertex_formats`]) or pulled in a vertex
//!   shader;
//! - **indices**: `array<u32>` storage buffer holding *absolute* vertex slots, compacted across
//!   meshlets: `SetMeshOutputCounts` becomes an `atomicAdd` on the `indexCount` word of a
//!   `drawIndexedIndirect` argument buffer, so culled meshlets (count 0) cost nothing and the
//!   whole dispatch is drawn with one indexed indirect draw (`[indexCount, 1, 0, 0, 0]`, reset to
//!   [`DRAW_INDEXED_INDIRECT_RESET`] before the dispatch);
//! - **primitives** (optional): per-primitive attributes, indexed by the same compacted primitive
//!   index;
//! - `SV_CullPrimitive` on triangles is lowered to a degenerate triangle (all three indices equal),
//!   which the rasterizer discards;
//! - amplification (`DispatchMesh`) appends child mesh groups to a task queue whose count is
//!   clamped into a `dispatchWorkgroupsIndirect` argument buffer by a one-invocation finalize pass.
//!
//! ## Open questions (mesh)
//!
//! - Per-primitive attributes have no WebGPU vertex-input equivalent: consuming them needs the
//!   optional `primitive-index` feature or vertex unsharing; this module only lays them out.
//! - Atomic allocation does not preserve D3D's primitive order across meshlets; order-dependent
//!   blending would need a prefix-sum allocation pass.
//! - Culled lines/points have no degenerate-primitive equivalent that is guaranteed not to
//!   rasterize, so culling is only lowered for triangles.
//! - Task payloads: the queue records the parent task group per mesh group; how payload structs
//!   are stored and indexed is left to the converted shader.
//! - Amplification above 65535 mesh groups per frame is clamped (dropped), not split.
//!
//! # Converter integration (not wired yet)
//!
//! `bytes -> sanitize_spirv_bytes (S3) -> canonicalize_wave_spirv -> spv::parse ->
//! lower_module_for_webgpu (S4) -> lower_wave_ops -> audit -> WaveLoweringReport::validator ->
//! wgsl::write_string -> prepend WaveLoweringReport::wgsl_preamble`. Blockers convert into the
//! existing [`CompatAudit`] report via `to_compat_audit`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{self, Write as _};

use naga::proc::Layouter;
use naga::valid::{
    Capabilities, ModuleInfo, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator,
};
use naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn,
    CollectiveOperation, Expression, Function, FunctionArgument, FunctionResult, GatherMode,
    GlobalVariable, Handle, Literal, LocalVariable, MathFunction, MemoryDecorations, Module, Range,
    Scalar, ScalarKind, ShaderStage, Span, Statement, SubgroupOperation, Type, TypeInner,
    UniqueArena, VectorSize,
};

use crate::compat::{CompatAudit, CompatIssue, CompatIssueKind, CompatSeverity, SpirvStreamError};
use crate::convert::SPIRV_MAGIC;

/// Lane count assumed by `--wave64-emulate` (RDNA2 Wave64).
pub const WAVE64_LANE_COUNT: u32 = 64;

/// Default identifier prefix for generated helper functions, globals, and WGSL declarations.
pub const DEFAULT_HELPER_PREFIX: &str = "pg6w_";

/// WebGPU baseline default `maxComputeWorkgroupStorageSize`.
pub const WEBGPU_MAX_COMPUTE_WORKGROUP_STORAGE_SIZE: u32 = 16384;

/// WebGPU baseline default `maxComputeWorkgroupsPerDimension`.
pub const WEBGPU_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65535;

/// WGSL directive required in front of a module lowered with [`WaveStrategy::NativeSubgroups`].
pub const WGSL_SUBGROUPS_ENABLE: &str = "enable subgroups;\n";

// ---------------------------------------------------------------------------
// SPIR-V constants (Khronos SPIR-V Unified Registry)
// ---------------------------------------------------------------------------

const SPIRV_HEADER_WORDS: usize = 5;

const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_FUNCTION_CALL: u16 = 57;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_DECORATION_GROUP: u16 = 73;
const OP_GROUP_DECORATE: u16 = 74;
const OP_GROUP_MEMBER_DECORATE: u16 = 75;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_IADD: u16 = 128;
const OP_ISUB: u16 = 130;
const OP_SELECT: u16 = 169;
const OP_IEQUAL: u16 = 170;
const OP_UGREATER_THAN_EQUAL: u16 = 174;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_AND: u16 = 199;
const OP_BIT_COUNT: u16 = 205;
const OP_DECORATE_ID: u16 = 332;
const OP_GROUP_NON_UNIFORM_ELECT: u16 = 333;
const OP_GROUP_NON_UNIFORM_BROADCAST_FIRST: u16 = 338;
const OP_GROUP_NON_UNIFORM_BALLOT: u16 = 339;
const OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT: u16 = 342;
const OP_SUBGROUP_BALLOT_KHR: u16 = 4421;
const OP_EMIT_MESH_TASKS_EXT: u16 = 5294;
const OP_SET_MESH_OUTPUTS_EXT: u16 = 5295;
const OP_DECORATE_STRING: u16 = 5632;
const OP_MEMBER_DECORATE_STRING: u16 = 5633;

const DECORATION_BUILTIN: u32 = 11;
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_TASK_PAYLOAD_WORKGROUP_EXT: u32 = 5402;

const BUILTIN_SUBGROUP_SIZE: u32 = 36;
const BUILTIN_NUM_SUBGROUPS: u32 = 38;
const BUILTIN_SUBGROUP_ID: u32 = 40;
const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
const BUILTIN_SUBGROUP_EQ_MASK: u32 = 4416;
const BUILTIN_SUBGROUP_LT_MASK: u32 = 4420;

const EXEC_MODEL_VERTEX: u32 = 0;
const EXEC_MODEL_FRAGMENT: u32 = 4;
const EXEC_MODEL_GL_COMPUTE: u32 = 5;
const EXEC_MODEL_TASK_NV: u32 = 5267;
const EXEC_MODEL_MESH_NV: u32 = 5268;
const EXEC_MODEL_TASK_EXT: u32 = 5364;
const EXEC_MODEL_MESH_EXT: u32 = 5365;

const EXEC_MODE_LOCAL_SIZE: u32 = 17;
const EXEC_MODE_OUTPUT_VERTICES: u32 = 26;
const EXEC_MODE_OUTPUT_POINTS: u32 = 27;
const EXEC_MODE_SUBGROUP_SIZE: u32 = 35;
const EXEC_MODE_OUTPUT_LINES_EXT: u32 = 5269;
const EXEC_MODE_OUTPUT_PRIMITIVES_EXT: u32 = 5270;
const EXEC_MODE_OUTPUT_TRIANGLES_EXT: u32 = 5298;

const GROUP_OP_REDUCE: u32 = 0;
const GROUP_OP_INCLUSIVE_SCAN: u32 = 1;
const GROUP_OP_EXCLUSIVE_SCAN: u32 = 2;
const GROUP_OP_CLUSTERED_REDUCE: u32 = 3;

// ---------------------------------------------------------------------------
// Wave intrinsic model
// ---------------------------------------------------------------------------

/// Combining operation of a wave reduction or scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WaveOp {
    All,
    Any,
    Add,
    Mul,
    Min,
    Max,
    And,
    Or,
    Xor,
}

impl WaveOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Any => "any",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Min => "min",
            Self::Max => "max",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
        }
    }

    const fn from_naga(op: SubgroupOperation) -> Self {
        match op {
            SubgroupOperation::All => Self::All,
            SubgroupOperation::Any => Self::Any,
            SubgroupOperation::Add => Self::Add,
            SubgroupOperation::Mul => Self::Mul,
            SubgroupOperation::Min => Self::Min,
            SubgroupOperation::Max => Self::Max,
            SubgroupOperation::And => Self::And,
            SubgroupOperation::Or => Self::Or,
            SubgroupOperation::Xor => Self::Xor,
        }
    }

    /// `true` if combining identical lane values yields that value (`x op x == x`).
    pub const fn is_idempotent(self) -> bool {
        matches!(
            self,
            Self::All | Self::Any | Self::Min | Self::Max | Self::And | Self::Or
        )
    }
}

impl fmt::Display for WaveOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// SPIR-V `GroupOperation` of a wave reduction, scan, or ballot bit count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WaveGroupOp {
    Reduce,
    InclusiveScan,
    ExclusiveScan,
    ClusteredReduce,
    /// `PartitionedReduceNV` / `PartitionedInclusiveScanNV` / `PartitionedExclusiveScanNV`.
    Partitioned,
    Other(u32),
}

impl WaveGroupOp {
    const fn from_spirv(value: u32) -> Self {
        match value {
            GROUP_OP_REDUCE => Self::Reduce,
            GROUP_OP_INCLUSIVE_SCAN => Self::InclusiveScan,
            GROUP_OP_EXCLUSIVE_SCAN => Self::ExclusiveScan,
            GROUP_OP_CLUSTERED_REDUCE => Self::ClusteredReduce,
            6..=8 => Self::Partitioned,
            other => Self::Other(other),
        }
    }

    const fn from_naga(op: CollectiveOperation) -> Self {
        match op {
            CollectiveOperation::Reduce => Self::Reduce,
            CollectiveOperation::InclusiveScan => Self::InclusiveScan,
            CollectiveOperation::ExclusiveScan => Self::ExclusiveScan,
        }
    }

    const fn is_basic(self) -> bool {
        matches!(
            self,
            Self::Reduce | Self::InclusiveScan | Self::ExclusiveScan
        )
    }
}

impl fmt::Display for WaveGroupOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reduce => f.write_str("Reduce"),
            Self::InclusiveScan => f.write_str("InclusiveScan"),
            Self::ExclusiveScan => f.write_str("ExclusiveScan"),
            Self::ClusteredReduce => f.write_str("ClusteredReduce"),
            Self::Partitioned => f.write_str("Partitioned"),
            Self::Other(value) => write!(f, "GroupOperation({value})"),
        }
    }
}

/// Whether an intrinsic's result depends only on the calling lane or on other lanes' data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LaneDependency {
    /// Computed from the calling lane's own inputs (lane index, wave size, a mask value).
    LaneLocal,
    /// Reads data from other lanes of the wave.
    CrossLane,
}

impl LaneDependency {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LaneLocal => "lane-local",
            Self::CrossLane => "cross-lane",
        }
    }
}

/// A wave/subgroup intrinsic, independent of its SPIR-V or naga IR spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WaveIntrinsic {
    /// `SubgroupSize` built-in (`WaveGetLaneCount`).
    LaneCount,
    /// `SubgroupLocalInvocationId` built-in (`WaveGetLaneIndex`).
    LaneIndex,
    /// `Subgroup{Eq,Ge,Gt,Le,Lt}Mask` built-ins.
    LaneMask,
    /// `NumSubgroups` built-in.
    NumWaves,
    /// `SubgroupId` built-in.
    WaveIndex,
    IsFirstLane,
    AllEqual,
    Ballot,
    InverseBallot,
    BallotBitExtract,
    BallotBitCount(WaveGroupOp),
    BallotFindLsb,
    BallotFindMsb,
    Collective(WaveOp, WaveGroupOp),
    ReadFirstLane,
    ReadLaneAt,
    Shuffle,
    ShuffleXor,
    ShuffleUp,
    ShuffleDown,
    QuadBroadcast,
    QuadSwap,
    QuadVote,
    Rotate,
    Partition,
}

impl WaveIntrinsic {
    /// The HLSL Shader Model 6 intrinsic this construct is generated from, when there is one.
    pub const fn hlsl_name(self) -> &'static str {
        match self {
            Self::LaneCount => "WaveGetLaneCount",
            Self::LaneIndex => "WaveGetLaneIndex",
            Self::IsFirstLane => "WaveIsFirstLane",
            Self::AllEqual => "WaveActiveAllEqual",
            Self::Ballot => "WaveActiveBallot",
            Self::BallotBitCount(WaveGroupOp::Reduce) => "WaveActiveCountBits",
            Self::BallotBitCount(WaveGroupOp::ExclusiveScan) => "WavePrefixCountBits",
            Self::Collective(WaveOp::All, WaveGroupOp::Reduce) => "WaveActiveAllTrue",
            Self::Collective(WaveOp::Any, WaveGroupOp::Reduce) => "WaveActiveAnyTrue",
            Self::Collective(WaveOp::Add, WaveGroupOp::Reduce) => "WaveActiveSum",
            Self::Collective(WaveOp::Mul, WaveGroupOp::Reduce) => "WaveActiveProduct",
            Self::Collective(WaveOp::Min, WaveGroupOp::Reduce) => "WaveActiveMin",
            Self::Collective(WaveOp::Max, WaveGroupOp::Reduce) => "WaveActiveMax",
            Self::Collective(WaveOp::And, WaveGroupOp::Reduce) => "WaveActiveBitAnd",
            Self::Collective(WaveOp::Or, WaveGroupOp::Reduce) => "WaveActiveBitOr",
            Self::Collective(WaveOp::Xor, WaveGroupOp::Reduce) => "WaveActiveBitXor",
            Self::Collective(WaveOp::Add, WaveGroupOp::ExclusiveScan) => "WavePrefixSum",
            Self::Collective(WaveOp::Mul, WaveGroupOp::ExclusiveScan) => "WavePrefixProduct",
            Self::Collective(_, WaveGroupOp::Partitioned) => "WaveMultiPrefix*",
            Self::ReadFirstLane => "WaveReadLaneFirst",
            Self::ReadLaneAt | Self::Shuffle => "WaveReadLaneAt",
            Self::QuadBroadcast => "QuadReadLaneAt",
            Self::QuadSwap => "QuadReadAcrossX/Y/Diagonal",
            Self::QuadVote => "QuadAny/QuadAll",
            Self::Partition => "WaveMatch",
            _ => "(no direct HLSL intrinsic)",
        }
    }

    /// Whether the result reads other lanes' data.
    pub const fn lane_dependency(self) -> LaneDependency {
        match self {
            Self::LaneCount
            | Self::LaneIndex
            | Self::LaneMask
            | Self::NumWaves
            | Self::WaveIndex
            | Self::InverseBallot
            | Self::BallotBitExtract
            | Self::BallotBitCount(_)
            | Self::BallotFindLsb
            | Self::BallotFindMsb => LaneDependency::LaneLocal,
            _ => LaneDependency::CrossLane,
        }
    }

    const fn is_quad(self) -> bool {
        matches!(self, Self::QuadBroadcast | Self::QuadSwap | Self::QuadVote)
    }
}

impl fmt::Display for WaveIntrinsic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BallotBitCount(group) => write!(f, "BallotBitCount({group})"),
            Self::Collective(op, group) => write!(f, "Collective({op}, {group})"),
            other => write!(f, "{other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// SPIR-V framing helpers
// ---------------------------------------------------------------------------

fn spirv_instructions(words: &[u32]) -> Result<Vec<&[u32]>, SpirvStreamError> {
    if words.len() < SPIRV_HEADER_WORDS {
        return Err(SpirvStreamError::TooShort {
            actual_words: words.len(),
            min_words: SPIRV_HEADER_WORDS,
        });
    }
    if words[0] != SPIRV_MAGIC {
        return Err(SpirvStreamError::InvalidMagic {
            magic: words[0],
            is_big_endian: words[0] == SPIRV_MAGIC.swap_bytes(),
        });
    }

    let mut instructions = Vec::new();
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
        let end = offset + word_count;
        if end > words.len() {
            return Err(SpirvStreamError::TruncatedInstruction {
                word_offset: offset,
                opcode,
                declared_words: word_count,
                available_words: words.len() - offset,
            });
        }
        instructions.push(&words[offset..end]);
        offset = end;
    }
    Ok(instructions)
}

fn inst_opcode(inst: &[u32]) -> u16 {
    (inst[0] & 0xffff) as u16
}

fn make_inst(opcode: u16, operands: &[u32]) -> Vec<u32> {
    let mut inst = Vec::with_capacity(operands.len() + 1);
    inst.push((((operands.len() + 1) as u32) << 16) | u32::from(opcode));
    inst.extend_from_slice(operands);
    inst
}

/// Decode a literal string operand; returns the string and the number of words it occupies.
fn decode_literal_string(operands: &[u32]) -> (String, usize) {
    let mut bytes = Vec::new();
    for (index, word) in operands.iter().enumerate() {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                return (String::from_utf8_lossy(&bytes).into_owned(), index + 1);
            }
            bytes.push(byte);
        }
    }
    (String::from_utf8_lossy(&bytes).into_owned(), operands.len())
}

struct SpirvEntryPoint {
    inst_index: usize,
    model: u32,
    function: u32,
    name: String,
    interface: Vec<u32>,
}

/// Entry points and the `OpFunctionCall` graph of a SPIR-V module.
struct SpirvCallGraph {
    entry_points: Vec<SpirvEntryPoint>,
    calls: HashMap<u32, BTreeSet<u32>>,
}

impl SpirvCallGraph {
    fn build(instructions: &[&[u32]]) -> Self {
        let mut entry_points = Vec::new();
        let mut calls: HashMap<u32, BTreeSet<u32>> = HashMap::new();
        let mut current_function = None;
        for (inst_index, inst) in instructions.iter().enumerate() {
            match inst_opcode(inst) {
                OP_ENTRY_POINT if inst.len() >= 4 => {
                    let (name, consumed) = decode_literal_string(&inst[3..]);
                    entry_points.push(SpirvEntryPoint {
                        inst_index,
                        model: inst[1],
                        function: inst[2],
                        name,
                        interface: inst[(3 + consumed).min(inst.len())..].to_vec(),
                    });
                }
                OP_FUNCTION if inst.len() >= 3 => current_function = Some(inst[2]),
                OP_FUNCTION_END => current_function = None,
                OP_FUNCTION_CALL if inst.len() >= 4 => {
                    if let Some(caller) = current_function {
                        calls.entry(caller).or_default().insert(inst[3]);
                    }
                }
                _ => {}
            }
        }
        Self {
            entry_points,
            calls,
        }
    }

    fn reachable_from(&self, root: u32) -> BTreeSet<u32> {
        let mut seen = BTreeSet::new();
        let mut pending = vec![root];
        while let Some(function) = pending.pop() {
            if seen.insert(function)
                && let Some(callees) = self.calls.get(&function)
            {
                pending.extend(callees.iter().copied());
            }
        }
        seen
    }
}

// ---------------------------------------------------------------------------
// Detection & classification (SPIR-V level)
// ---------------------------------------------------------------------------

/// Shader stage of a SPIR-V entry point, as seen by the wave/mesh scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WaveShaderStage {
    Vertex,
    Fragment,
    Compute,
    Task,
    Mesh,
    /// Any other SPIR-V execution model (raw value).
    Other(u32),
}

impl WaveShaderStage {
    const fn from_execution_model(model: u32) -> Self {
        match model {
            EXEC_MODEL_VERTEX => Self::Vertex,
            EXEC_MODEL_FRAGMENT => Self::Fragment,
            EXEC_MODEL_GL_COMPUTE => Self::Compute,
            EXEC_MODEL_TASK_NV | EXEC_MODEL_TASK_EXT => Self::Task,
            EXEC_MODEL_MESH_NV | EXEC_MODEL_MESH_EXT => Self::Mesh,
            other => Self::Other(other),
        }
    }
}

impl fmt::Display for WaveShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vertex => f.write_str("vertex"),
            Self::Fragment => f.write_str("fragment"),
            Self::Compute => f.write_str("compute"),
            Self::Task => f.write_str("task"),
            Self::Mesh => f.write_str("mesh"),
            Self::Other(model) => write!(f, "ExecutionModel({model})"),
        }
    }
}

/// Aggregated uses of one wave intrinsic (one SPIR-V spelling) in a module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveUse {
    pub intrinsic: WaveIntrinsic,
    /// SPIR-V opcode of the instruction, or `None` for a built-in variable load.
    pub spirv_opcode: Option<u16>,
    pub occurrences: usize,
    /// Entry points whose static call graph reaches a use (declaration order).
    pub entry_points: Vec<String>,
    pub stages: Vec<WaveShaderStage>,
    /// naga 30's SPIR-V front end parses this spelling into naga IR.
    pub naga_front_end: bool,
    /// [`canonicalize_wave_spirv`] rewrites this spelling into naga-parseable SPIR-V.
    pub canonicalizable: bool,
}

impl WaveUse {
    pub const fn lane_dependency(&self) -> LaneDependency {
        self.intrinsic.lane_dependency()
    }

    /// `true` if the use can reach [`lower_wave_ops`] (directly or after canonicalization).
    pub const fn reaches_ir_lowering(&self) -> bool {
        self.naga_front_end || self.canonicalizable
    }
}

/// Mesh / amplification (task) entry point found by [`scan_wave_usage_spirv`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshUsage {
    pub entry_point: String,
    pub stage: WaveShaderStage,
    pub max_vertices: Option<u32>,
    pub max_primitives: Option<u32>,
    pub topology: Option<MeshTopology>,
    pub workgroup_size: Option<[u32; 3]>,
    /// The entry point's call graph contains `OpSetMeshOutputsEXT`.
    pub sets_mesh_outputs: bool,
    /// The entry point's call graph contains `OpEmitMeshTasksEXT`.
    pub emits_mesh_tasks: bool,
}

/// Structured wave/subgroup and mesh usage report for one SPIR-V module.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WaveUsageReport {
    pub uses: Vec<WaveUse>,
    pub mesh: Vec<MeshUsage>,
    /// `OpExecutionMode SubgroupSize` request, if any.
    pub requested_subgroup_size: Option<u32>,
    /// The module declares a `TaskPayloadWorkgroupEXT` variable.
    pub uses_task_payload: bool,
}

impl WaveUsageReport {
    pub fn is_empty(&self) -> bool {
        self.uses.is_empty()
            && self.mesh.is_empty()
            && self.requested_subgroup_size.is_none()
            && !self.uses_task_payload
    }

    pub fn has_cross_lane(&self) -> bool {
        self.uses
            .iter()
            .any(|u| u.lane_dependency() == LaneDependency::CrossLane)
    }

    /// Render the usage as compatibility findings in the existing [`CompatAudit`] style:
    /// a `Warning` for wave uses that reach [`lower_wave_ops`], a `Blocker` for uses no strategy
    /// can lower and for mesh/task entry points.
    pub fn to_compat_audit(&self) -> CompatAudit {
        let mut audit = CompatAudit::default();
        for wave_use in &self.uses {
            let spelling = match wave_use.spirv_opcode {
                Some(opcode) => format!("opcode {opcode}"),
                None => "built-in".to_string(),
            };
            let scope = if wave_use.entry_points.is_empty() {
                "no entry point".to_string()
            } else {
                wave_use.entry_points.join(", ")
            };
            let headline = format!(
                "{} [{}] ({spelling}, {}) x{} in {scope}",
                wave_use.intrinsic,
                wave_use.intrinsic.hlsl_name(),
                wave_use.lane_dependency().as_str(),
                wave_use.occurrences,
            );
            if wave_use.reaches_ir_lowering() {
                audit.push_unique(CompatIssue {
                    severity: CompatSeverity::Warning,
                    kind: CompatIssueKind::SubgroupWave64,
                    message: format!("{headline} requires wave lowering"),
                    hint: Some(
                        "lower with wave_emul::lower_wave_ops (SingleLane / NativeSubgroups / WorkgroupEmulation)"
                            .to_string(),
                    ),
                });
            } else {
                audit.push_unique(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::SubgroupWave64,
                    message: format!(
                        "{headline} has no WebGPU lowering: naga's SPIR-V front end rejects it and wave_emul does not canonicalize it"
                    ),
                    hint: Some(
                        "rewrite upstream to supported wave intrinsics or extend wave_emul canonicalization"
                            .to_string(),
                    ),
                });
            }
        }
        if let Some(size) = self.requested_subgroup_size {
            audit.push_unique(CompatIssue {
                severity: CompatSeverity::Warning,
                kind: CompatIssueKind::SubgroupWave64,
                message: format!("module requests OpExecutionMode SubgroupSize({size})"),
                hint: Some(
                    "WebGPU cannot request a subgroup size; WorkgroupEmulation fixes the wave at 64 lanes"
                        .to_string(),
                ),
            });
        }
        for mesh in &self.mesh {
            let mut detail = Vec::new();
            if let Some(v) = mesh.max_vertices {
                detail.push(format!("max_vertices={v}"));
            }
            if let Some(p) = mesh.max_primitives {
                detail.push(format!("max_primitives={p}"));
            }
            if let Some(t) = mesh.topology {
                detail.push(format!("topology={}", t.webgpu_topology()));
            }
            audit.push_unique(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::MeshShader,
                message: format!(
                    "entry point '{}' is a {} shader ({})",
                    mesh.entry_point,
                    mesh.stage,
                    if detail.is_empty() {
                        "no output limits declared".to_string()
                    } else {
                        detail.join(", ")
                    }
                ),
                hint: Some(
                    "lower to compute (--mesh-compute) and declare outputs with wave_emul::mesh_output_wgsl"
                        .to_string(),
                ),
            });
        }
        audit
    }
}

fn builtin_intrinsic(builtin: u32) -> Option<WaveIntrinsic> {
    match builtin {
        BUILTIN_SUBGROUP_SIZE => Some(WaveIntrinsic::LaneCount),
        BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID => Some(WaveIntrinsic::LaneIndex),
        BUILTIN_NUM_SUBGROUPS => Some(WaveIntrinsic::NumWaves),
        BUILTIN_SUBGROUP_ID => Some(WaveIntrinsic::WaveIndex),
        BUILTIN_SUBGROUP_EQ_MASK..=BUILTIN_SUBGROUP_LT_MASK => Some(WaveIntrinsic::LaneMask),
        _ => None,
    }
}

/// Classify a SPIR-V instruction as a wave intrinsic.
fn classify_wave_instruction(inst: &[u32]) -> Option<WaveIntrinsic> {
    let group = || WaveGroupOp::from_spirv(inst.get(4).copied().unwrap_or(GROUP_OP_REDUCE));
    let intrinsic = match inst_opcode(inst) {
        OP_GROUP_NON_UNIFORM_ELECT => WaveIntrinsic::IsFirstLane,
        334 | 4428 => WaveIntrinsic::Collective(WaveOp::All, WaveGroupOp::Reduce),
        335 | 4429 => WaveIntrinsic::Collective(WaveOp::Any, WaveGroupOp::Reduce),
        336 | 4430 => WaveIntrinsic::AllEqual,
        337 | 4432 => WaveIntrinsic::ReadLaneAt,
        OP_GROUP_NON_UNIFORM_BROADCAST_FIRST | 4422 => WaveIntrinsic::ReadFirstLane,
        OP_GROUP_NON_UNIFORM_BALLOT | OP_SUBGROUP_BALLOT_KHR => WaveIntrinsic::Ballot,
        340 => WaveIntrinsic::InverseBallot,
        341 => WaveIntrinsic::BallotBitExtract,
        OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT => WaveIntrinsic::BallotBitCount(group()),
        343 => WaveIntrinsic::BallotFindLsb,
        344 => WaveIntrinsic::BallotFindMsb,
        345 => WaveIntrinsic::Shuffle,
        346 => WaveIntrinsic::ShuffleXor,
        347 => WaveIntrinsic::ShuffleUp,
        348 => WaveIntrinsic::ShuffleDown,
        349 | 350 => WaveIntrinsic::Collective(WaveOp::Add, group()),
        351 | 352 => WaveIntrinsic::Collective(WaveOp::Mul, group()),
        353..=355 => WaveIntrinsic::Collective(WaveOp::Min, group()),
        356..=358 => WaveIntrinsic::Collective(WaveOp::Max, group()),
        359 | 362 => WaveIntrinsic::Collective(WaveOp::And, group()),
        360 | 363 => WaveIntrinsic::Collective(WaveOp::Or, group()),
        361 | 364 => WaveIntrinsic::Collective(WaveOp::Xor, group()),
        365 => WaveIntrinsic::QuadBroadcast,
        366 => WaveIntrinsic::QuadSwap,
        4431 => WaveIntrinsic::Rotate,
        5110 | 5111 => WaveIntrinsic::QuadVote,
        5296 => WaveIntrinsic::Partition,
        _ => return None,
    };
    Some(intrinsic)
}

/// Whether naga 30's SPIR-V front end (`next_block.rs` / `convert.rs::map_builtin`) parses it.
const fn naga_spv_front_end_supports(intrinsic: WaveIntrinsic, opcode: Option<u16>) -> bool {
    match opcode {
        None => matches!(
            intrinsic,
            WaveIntrinsic::LaneCount
                | WaveIntrinsic::LaneIndex
                | WaveIntrinsic::NumWaves
                | WaveIntrinsic::WaveIndex
        ),
        Some(334 | 335 | 337 | 338 | 339 | 345..=348 | 365 | 366) => true,
        Some(349..=364) => match intrinsic {
            WaveIntrinsic::Collective(_, group) => group.is_basic(),
            _ => false,
        },
        Some(_) => false,
    }
}

const fn canonicalization_supports(intrinsic: WaveIntrinsic, opcode: Option<u16>) -> bool {
    match (intrinsic, opcode) {
        (WaveIntrinsic::IsFirstLane, Some(OP_GROUP_NON_UNIFORM_ELECT)) => true,
        (WaveIntrinsic::BallotBitCount(group), Some(OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT)) => {
            group.is_basic()
        }
        _ => false,
    }
}

/// Scan a SPIR-V word stream for wave/subgroup intrinsics and mesh/task constructs.
///
/// Uses are attributed to entry points through the static `OpFunctionCall` graph; built-ins are
/// counted per `OpLoad` of the decorated variable.
pub fn scan_wave_usage_spirv(words: &[u32]) -> Result<WaveUsageReport, SpirvStreamError> {
    let instructions = spirv_instructions(words)?;
    let graph = SpirvCallGraph::build(&instructions);

    let mut builtin_vars: HashMap<u32, u32> = HashMap::new();
    let mut exec_modes: HashMap<u32, Vec<&[u32]>> = HashMap::new();
    let mut raw_uses: Vec<(u32, WaveIntrinsic, Option<u16>)> = Vec::new();
    let mut mesh_ops: HashMap<u32, (bool, bool)> = HashMap::new();
    let mut report = WaveUsageReport::default();
    let mut current_function = None;

    for inst in &instructions {
        let opcode = inst_opcode(inst);
        match opcode {
            OP_EXECUTION_MODE if inst.len() >= 3 => {
                exec_modes.entry(inst[1]).or_default().push(inst);
                if inst[2] == EXEC_MODE_SUBGROUP_SIZE {
                    report.requested_subgroup_size = inst.get(3).copied();
                }
            }
            OP_DECORATE if inst.len() >= 4 && inst[2] == DECORATION_BUILTIN => {
                builtin_vars.insert(inst[1], inst[3]);
            }
            OP_VARIABLE
                if inst.len() >= 4 && inst[3] == STORAGE_CLASS_TASK_PAYLOAD_WORKGROUP_EXT =>
            {
                report.uses_task_payload = true;
            }
            OP_FUNCTION if inst.len() >= 3 => current_function = Some(inst[2]),
            OP_FUNCTION_END => current_function = None,
            _ => {}
        }
        let Some(function) = current_function else {
            continue;
        };
        if opcode == OP_LOAD && inst.len() >= 4 {
            if let Some(intrinsic) = builtin_vars
                .get(&inst[3])
                .copied()
                .and_then(builtin_intrinsic)
            {
                raw_uses.push((function, intrinsic, None));
            }
        } else if opcode == OP_SET_MESH_OUTPUTS_EXT {
            mesh_ops.entry(function).or_default().0 = true;
        } else if opcode == OP_EMIT_MESH_TASKS_EXT {
            mesh_ops.entry(function).or_default().1 = true;
        } else if let Some(intrinsic) = classify_wave_instruction(inst) {
            raw_uses.push((function, intrinsic, Some(opcode)));
        }
    }

    let reach: Vec<BTreeSet<u32>> = graph
        .entry_points
        .iter()
        .map(|ep| graph.reachable_from(ep.function))
        .collect();

    let mut aggregated: BTreeMap<(WaveIntrinsic, Option<u16>), (usize, BTreeSet<usize>)> =
        BTreeMap::new();
    for (function, intrinsic, opcode) in raw_uses {
        let entry = aggregated.entry((intrinsic, opcode)).or_default();
        entry.0 += 1;
        for (index, reached) in reach.iter().enumerate() {
            if reached.contains(&function) {
                entry.1.insert(index);
            }
        }
    }
    for ((intrinsic, opcode), (occurrences, eps)) in aggregated {
        let stages: BTreeSet<WaveShaderStage> = eps
            .iter()
            .map(|&i| WaveShaderStage::from_execution_model(graph.entry_points[i].model))
            .collect();
        report.uses.push(WaveUse {
            intrinsic,
            spirv_opcode: opcode,
            occurrences,
            entry_points: eps
                .iter()
                .map(|&i| graph.entry_points[i].name.clone())
                .collect(),
            stages: stages.into_iter().collect(),
            naga_front_end: naga_spv_front_end_supports(intrinsic, opcode),
            canonicalizable: canonicalization_supports(intrinsic, opcode),
        });
    }

    for (index, ep) in graph.entry_points.iter().enumerate() {
        let stage = WaveShaderStage::from_execution_model(ep.model);
        if !matches!(stage, WaveShaderStage::Mesh | WaveShaderStage::Task) {
            continue;
        }
        let mut usage = MeshUsage {
            entry_point: ep.name.clone(),
            stage,
            max_vertices: None,
            max_primitives: None,
            topology: None,
            workgroup_size: None,
            sets_mesh_outputs: false,
            emits_mesh_tasks: false,
        };
        for mode in exec_modes.get(&ep.function).into_iter().flatten() {
            match mode[2] {
                EXEC_MODE_OUTPUT_VERTICES => usage.max_vertices = mode.get(3).copied(),
                EXEC_MODE_OUTPUT_PRIMITIVES_EXT => usage.max_primitives = mode.get(3).copied(),
                EXEC_MODE_OUTPUT_POINTS => usage.topology = Some(MeshTopology::Points),
                EXEC_MODE_OUTPUT_LINES_EXT => usage.topology = Some(MeshTopology::Lines),
                EXEC_MODE_OUTPUT_TRIANGLES_EXT => usage.topology = Some(MeshTopology::Triangles),
                EXEC_MODE_LOCAL_SIZE if mode.len() >= 6 => {
                    usage.workgroup_size = Some([mode[3], mode[4], mode[5]]);
                }
                _ => {}
            }
        }
        for function in &reach[index] {
            if let Some(&(sets, emits)) = mesh_ops.get(function) {
                usage.sets_mesh_outputs |= sets;
                usage.emits_mesh_tasks |= emits;
            }
        }
        report.mesh.push(usage);
    }

    Ok(report)
}

// ---------------------------------------------------------------------------
// Canonicalization (SPIR-V level)
// ---------------------------------------------------------------------------

/// Output of [`canonicalize_wave_spirv`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CanonicalizedWaveSpirv {
    pub words: Vec<u32>,
    /// Rewritten instruction counts per intrinsic.
    pub rewritten: BTreeMap<WaveIntrinsic, usize>,
    /// Instructions left untouched, with the reason (they stay naga parse blockers).
    pub skipped: Vec<String>,
}

struct IdAllocator {
    next: u32,
}

impl IdAllocator {
    fn fresh(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }
}

/// Lazily created declarations needed by the canonicalized instruction sequences.
struct CanonicalDecls {
    ids: IdAllocator,
    uint_ty: Option<u32>,
    bool_ty: Option<u32>,
    input_uint_ptr: Option<u32>,
    uint_consts: HashMap<u32, u32>,
    lane_var: Option<u32>,
    new_decls: Vec<Vec<u32>>,
    new_annotations: Vec<Vec<u32>>,
}

impl CanonicalDecls {
    fn uint(&mut self) -> u32 {
        if let Some(id) = self.uint_ty {
            return id;
        }
        let id = self.ids.fresh();
        self.new_decls.push(make_inst(OP_TYPE_INT, &[id, 32, 0]));
        self.uint_ty = Some(id);
        id
    }

    fn boolean(&mut self) -> u32 {
        if let Some(id) = self.bool_ty {
            return id;
        }
        let id = self.ids.fresh();
        self.new_decls.push(make_inst(OP_TYPE_BOOL, &[id]));
        self.bool_ty = Some(id);
        id
    }

    fn uint_const(&mut self, value: u32) -> u32 {
        if let Some(&id) = self.uint_consts.get(&value) {
            return id;
        }
        let ty = self.uint();
        let id = self.ids.fresh();
        self.new_decls
            .push(make_inst(OP_CONSTANT, &[ty, id, value]));
        self.uint_consts.insert(value, id);
        id
    }

    fn lane_index_var(&mut self) -> u32 {
        if let Some(id) = self.lane_var {
            return id;
        }
        let uint = self.uint();
        let ptr = match self.input_uint_ptr {
            Some(ptr) => ptr,
            None => {
                let ptr = self.ids.fresh();
                self.new_decls.push(make_inst(
                    OP_TYPE_POINTER,
                    &[ptr, STORAGE_CLASS_INPUT, uint],
                ));
                self.input_uint_ptr = Some(ptr);
                ptr
            }
        };
        let var = self.ids.fresh();
        self.new_decls
            .push(make_inst(OP_VARIABLE, &[ptr, var, STORAGE_CLASS_INPUT]));
        self.new_annotations.push(make_inst(
            OP_DECORATE,
            &[
                var,
                DECORATION_BUILTIN,
                BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID,
            ],
        ));
        self.lane_var = Some(var);
        var
    }
}

const fn is_annotation(opcode: u16) -> bool {
    matches!(
        opcode,
        OP_DECORATE
            | OP_MEMBER_DECORATE
            | OP_DECORATION_GROUP
            | OP_GROUP_DECORATE
            | OP_GROUP_MEMBER_DECORATE
            | OP_DECORATE_ID
            | OP_DECORATE_STRING
            | OP_MEMBER_DECORATE_STRING
    )
}

/// `OpType*`, `OpConstant*`, `OpSpecConstant*`, `OpVariable`, `OpUndef` (declaration section).
const fn is_global_declaration(opcode: u16) -> bool {
    matches!(opcode, 1 | 19..=39 | 41..=52 | OP_VARIABLE | 322 | 5341)
}

/// Rewrite the naga-unsupported wave intrinsics that have exact equivalents into
/// naga-parseable SPIR-V:
///
/// - `OpGroupNonUniformElect` -> `lane == OpGroupNonUniformBroadcastFirst(lane)` with `lane`
///   loaded from a `SubgroupLocalInvocationId` input (added, decorated, and listed on every
///   entry point that reaches the rewrite if not already present).
/// - `OpGroupNonUniformBallotBitCount` (Reduce / InclusiveScan / ExclusiveScan) -> sum of
///   `OpBitCount` over the four mask words, ANDed with the lane mask
///   `(1 << min(max(k - 32c, 0), 32)) - 1` per word `c` for scans (`k` = lane index, `+1` for
///   inclusive). Only applied when the operand is an `OpGroupNonUniformBallot` result: SPIR-V
///   counts only bits below the subgroup size, which every lowering's ballot guarantees to be the
///   only bits set.
///
/// Returns the input words unchanged when nothing needs rewriting.
pub fn canonicalize_wave_spirv(words: &[u32]) -> Result<CanonicalizedWaveSpirv, SpirvStreamError> {
    let instructions = spirv_instructions(words)?;
    let mut result = CanonicalizedWaveSpirv {
        words: Vec::new(),
        rewritten: BTreeMap::new(),
        skipped: Vec::new(),
    };

    let needs_rewrite = instructions.iter().any(|inst| {
        matches!(
            inst_opcode(inst),
            OP_GROUP_NON_UNIFORM_ELECT | OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT
        )
    });
    if !needs_rewrite {
        result.words = words.to_vec();
        return Ok(result);
    }

    let graph = SpirvCallGraph::build(&instructions);
    let mut decls = CanonicalDecls {
        ids: IdAllocator { next: words[3] },
        uint_ty: None,
        bool_ty: None,
        input_uint_ptr: None,
        uint_consts: HashMap::new(),
        lane_var: None,
        new_decls: Vec::new(),
        new_annotations: Vec::new(),
    };
    let mut ballot_ids = HashSet::new();
    let mut first_function = instructions.len();
    let mut last_annotation = None;
    let mut first_declaration = None;

    for (index, inst) in instructions.iter().enumerate() {
        let opcode = inst_opcode(inst);
        if first_function == instructions.len() {
            if opcode == OP_FUNCTION {
                first_function = index;
            } else if is_annotation(opcode) {
                last_annotation = Some(index);
            } else if is_global_declaration(opcode) && first_declaration.is_none() {
                first_declaration = Some(index);
            }
        }
        match opcode {
            OP_TYPE_INT if inst.len() >= 4 && inst[2] == 32 && inst[3] == 0 => {
                decls.uint_ty.get_or_insert(inst[1]);
            }
            OP_TYPE_BOOL if inst.len() >= 2 => {
                decls.bool_ty.get_or_insert(inst[1]);
            }
            OP_TYPE_POINTER
                if inst.len() >= 4
                    && inst[2] == STORAGE_CLASS_INPUT
                    && Some(inst[3]) == decls.uint_ty =>
            {
                decls.input_uint_ptr.get_or_insert(inst[1]);
            }
            OP_CONSTANT if inst.len() >= 4 && Some(inst[1]) == decls.uint_ty => {
                decls.uint_consts.entry(inst[3]).or_insert(inst[2]);
            }
            OP_DECORATE
                if inst.len() >= 4
                    && inst[2] == DECORATION_BUILTIN
                    && inst[3] == BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID =>
            {
                decls.lane_var.get_or_insert(inst[1]);
            }
            OP_GROUP_NON_UNIFORM_BALLOT | OP_SUBGROUP_BALLOT_KHR if inst.len() >= 3 => {
                ballot_ids.insert(inst[2]);
            }
            _ => {}
        }
    }

    let mut replacements: HashMap<usize, Vec<Vec<u32>>> = HashMap::new();
    let mut lane_functions: BTreeSet<u32> = BTreeSet::new();
    let mut current_function = None;
    for (index, inst) in instructions.iter().enumerate() {
        match inst_opcode(inst) {
            OP_FUNCTION if inst.len() >= 3 => current_function = Some(inst[2]),
            OP_FUNCTION_END => current_function = None,
            OP_GROUP_NON_UNIFORM_ELECT if inst.len() >= 4 => {
                // [header, bool_ty, result, scope]
                let (bool_ty, result_id, scope) = (inst[1], inst[2], inst[3]);
                let uint = decls.uint();
                let lane_var = decls.lane_index_var();
                let lane = decls.ids.fresh();
                let first = decls.ids.fresh();
                replacements.insert(
                    index,
                    vec![
                        make_inst(OP_LOAD, &[uint, lane, lane_var]),
                        make_inst(
                            OP_GROUP_NON_UNIFORM_BROADCAST_FIRST,
                            &[uint, first, scope, lane],
                        ),
                        make_inst(OP_IEQUAL, &[bool_ty, result_id, lane, first]),
                    ],
                );
                lane_functions.extend(current_function);
                *result
                    .rewritten
                    .entry(WaveIntrinsic::IsFirstLane)
                    .or_default() += 1;
            }
            OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT if inst.len() >= 6 => {
                // [header, uint_ty, result, scope, group_op, ballot]
                let (result_ty, result_id, group, value) = (inst[1], inst[2], inst[4], inst[5]);
                let intrinsic = WaveIntrinsic::BallotBitCount(WaveGroupOp::from_spirv(group));
                if !ballot_ids.contains(&value) {
                    result.skipped.push(format!(
                        "{intrinsic} %{result_id}: operand %{value} is not an OpGroupNonUniformBallot result, so bits above the subgroup size cannot be ruled out"
                    ));
                    continue;
                }
                if !matches!(
                    group,
                    GROUP_OP_REDUCE | GROUP_OP_INCLUSIVE_SCAN | GROUP_OP_EXCLUSIVE_SCAN
                ) {
                    result.skipped.push(format!(
                        "{intrinsic} %{result_id}: group operation not canonicalized"
                    ));
                    continue;
                }
                if decls.uint() != result_ty {
                    result.skipped.push(format!(
                        "{intrinsic} %{result_id}: result type %{result_ty} is not the module's 32-bit unsigned integer type"
                    ));
                    continue;
                }
                let sequence = if group == GROUP_OP_REDUCE {
                    ballot_count_reduce(&mut decls, result_ty, result_id, value)
                } else {
                    lane_functions.extend(current_function);
                    ballot_count_scan(
                        &mut decls,
                        result_ty,
                        result_id,
                        value,
                        group == GROUP_OP_INCLUSIVE_SCAN,
                    )
                };
                replacements.insert(index, sequence);
                *result.rewritten.entry(intrinsic).or_default() += 1;
            }
            _ => {}
        }
    }

    // Entry points that reach a lane-index use must list the input variable in their interface.
    let mut interface_additions: HashMap<usize, u32> = HashMap::new();
    if let Some(lane_var) = decls.lane_var
        && !lane_functions.is_empty()
    {
        for ep in &graph.entry_points {
            let reaches = graph
                .reachable_from(ep.function)
                .iter()
                .any(|f| lane_functions.contains(f));
            if reaches && !ep.interface.contains(&lane_var) {
                interface_additions.insert(ep.inst_index, lane_var);
            }
        }
    }

    let annotation_index = last_annotation
        .map(|i| i + 1)
        .or(first_declaration)
        .unwrap_or(first_function);

    let mut out = Vec::with_capacity(words.len() + 64);
    out.extend_from_slice(&words[..SPIRV_HEADER_WORDS]);
    for (index, inst) in instructions.iter().enumerate() {
        if index == annotation_index {
            for annotation in &decls.new_annotations {
                out.extend_from_slice(annotation);
            }
        }
        if index == first_function {
            for decl in &decls.new_decls {
                out.extend_from_slice(decl);
            }
        }
        if let Some(&var) = interface_additions.get(&index) {
            let mut operands = inst[1..].to_vec();
            operands.push(var);
            out.extend_from_slice(&make_inst(OP_ENTRY_POINT, &operands));
        } else if let Some(sequence) = replacements.get(&index) {
            for replacement in sequence {
                out.extend_from_slice(replacement);
            }
        } else {
            out.extend_from_slice(inst);
        }
    }
    out[3] = decls.ids.next;
    result.words = out;
    Ok(result)
}

fn ballot_count_reduce(
    decls: &mut CanonicalDecls,
    uint: u32,
    result_id: u32,
    ballot: u32,
) -> Vec<Vec<u32>> {
    let mut sequence = Vec::new();
    let mut counts = Vec::with_capacity(4);
    for component in 0..4 {
        let word = decls.ids.fresh();
        let count = decls.ids.fresh();
        sequence.push(make_inst(
            OP_COMPOSITE_EXTRACT,
            &[uint, word, ballot, component],
        ));
        sequence.push(make_inst(OP_BIT_COUNT, &[uint, count, word]));
        counts.push(count);
    }
    sum_counts(decls, uint, result_id, &counts, &mut sequence);
    sequence
}

fn ballot_count_scan(
    decls: &mut CanonicalDecls,
    uint: u32,
    result_id: u32,
    ballot: u32,
    inclusive: bool,
) -> Vec<Vec<u32>> {
    let bool_ty = decls.boolean();
    let lane_var = decls.lane_index_var();
    let zero = decls.uint_const(0);
    let one = decls.uint_const(1);
    let thirty_two = decls.uint_const(32);
    let all_ones = decls.uint_const(u32::MAX);

    let mut sequence = Vec::new();
    let lane = decls.ids.fresh();
    sequence.push(make_inst(OP_LOAD, &[uint, lane, lane_var]));
    // k = number of low lanes counted: lane (exclusive) or lane + 1 (inclusive).
    let k = if inclusive {
        let k = decls.ids.fresh();
        sequence.push(make_inst(OP_IADD, &[uint, k, lane, one]));
        k
    } else {
        lane
    };

    let mut counts = Vec::with_capacity(4);
    for component in 0..4u32 {
        let low_lane = decls.uint_const(component * 32);
        let ge = decls.ids.fresh();
        let diff = decls.ids.fresh();
        let bits = decls.ids.fresh();
        let full = decls.ids.fresh();
        let shifted = decls.ids.fresh();
        let partial = decls.ids.fresh();
        let mask = decls.ids.fresh();
        let word = decls.ids.fresh();
        let masked = decls.ids.fresh();
        let count = decls.ids.fresh();
        // bits = clamp(k - 32c, 0, ..); mask = bits >= 32 ? ~0 : (1 << bits) - 1
        sequence.push(make_inst(
            OP_UGREATER_THAN_EQUAL,
            &[bool_ty, ge, k, low_lane],
        ));
        sequence.push(make_inst(OP_ISUB, &[uint, diff, k, low_lane]));
        sequence.push(make_inst(OP_SELECT, &[uint, bits, ge, diff, zero]));
        sequence.push(make_inst(
            OP_UGREATER_THAN_EQUAL,
            &[bool_ty, full, bits, thirty_two],
        ));
        sequence.push(make_inst(
            OP_SHIFT_LEFT_LOGICAL,
            &[uint, shifted, one, bits],
        ));
        sequence.push(make_inst(OP_ISUB, &[uint, partial, shifted, one]));
        sequence.push(make_inst(OP_SELECT, &[uint, mask, full, all_ones, partial]));
        sequence.push(make_inst(
            OP_COMPOSITE_EXTRACT,
            &[uint, word, ballot, component],
        ));
        sequence.push(make_inst(OP_BITWISE_AND, &[uint, masked, word, mask]));
        sequence.push(make_inst(OP_BIT_COUNT, &[uint, count, masked]));
        counts.push(count);
    }
    sum_counts(decls, uint, result_id, &counts, &mut sequence);
    sequence
}

fn sum_counts(
    decls: &mut CanonicalDecls,
    uint: u32,
    result_id: u32,
    counts: &[u32],
    sequence: &mut Vec<Vec<u32>>,
) {
    let mut acc = counts[0];
    for (index, &count) in counts.iter().enumerate().skip(1) {
        let id = if index + 1 == counts.len() {
            result_id
        } else {
            decls.ids.fresh()
        };
        sequence.push(make_inst(OP_IADD, &[uint, id, acc, count]));
        acc = id;
    }
}

// ---------------------------------------------------------------------------
// naga IR lowering: public API
// ---------------------------------------------------------------------------

/// Wave64 lowering strategy applied by [`lower_wave_ops`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WaveStrategy {
    /// Lane-free lowering: uniform-operand identities, plus the wave-of-1 model when allowed.
    SingleLane,
    /// Keep the operations as WGSL `subgroups` built-ins (optional WebGPU feature).
    NativeSubgroups,
    /// Emulate Wave64 inside compute workgroups with `var<workgroup>` scratch and barriers.
    WorkgroupEmulation,
}

impl WaveStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SingleLane => "SingleLane",
            Self::NativeSubgroups => "NativeSubgroups",
            Self::WorkgroupEmulation => "WorkgroupEmulation",
        }
    }
}

impl fmt::Display for WaveStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a lowered wave operation is correct (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoweringCondition {
    Exact,
    UniformOperand,
    WorkgroupUniformControlFlow,
    RuntimeSubgroupSize64,
    Wave1Model,
}

impl LoweringCondition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "Exact",
            Self::UniformOperand => "UniformOperand",
            Self::WorkgroupUniformControlFlow => "WorkgroupUniformControlFlow",
            Self::RuntimeSubgroupSize64 => "RuntimeSubgroupSize64",
            Self::Wave1Model => "Wave1Model",
        }
    }

    /// `true` when the result can differ from Wave64 hardware (`Wave1Model` always,
    /// `RuntimeSubgroupSize64` unless the runtime verified a fixed 64-lane subgroup size).
    pub const fn may_differ_from_wave64(self) -> bool {
        matches!(self, Self::Wave1Model | Self::RuntimeSubgroupSize64)
    }
}

impl fmt::Display for LoweringCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Options for [`lower_wave_ops`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveLoweringOptions {
    pub strategy: WaveStrategy,
    /// Accept [`LoweringCondition::Wave1Model`] approximations under [`WaveStrategy::SingleLane`].
    pub allow_wave1_model: bool,
    /// Prefix for generated identifiers; made unique against the module's names.
    pub helper_prefix: String,
    /// Workgroup storage budget for [`WaveStrategy::WorkgroupEmulation`] scratch arrays
    /// (including the module's own `var<workgroup>` variables).
    pub max_workgroup_storage_bytes: u32,
}

impl WaveLoweringOptions {
    pub fn new(strategy: WaveStrategy) -> Self {
        Self {
            strategy,
            allow_wave1_model: false,
            helper_prefix: DEFAULT_HELPER_PREFIX.to_string(),
            max_workgroup_storage_bytes: WEBGPU_MAX_COMPUTE_WORKGROUP_STORAGE_SIZE,
        }
    }
}

/// One wave operation lowered by [`lower_wave_ops`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredWaveOp {
    /// `entry point 'name'`, `function 'name'`, optionally with the argument index.
    pub location: String,
    pub intrinsic: WaveIntrinsic,
    pub strategy: WaveStrategy,
    pub condition: LoweringCondition,
    /// Generated helper function called in place of the operation, if any.
    pub helper: Option<String>,
}

/// A wave operation [`lower_wave_ops`] refused to lower, with the precise reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveBlocker {
    pub location: String,
    pub intrinsic: Option<WaveIntrinsic>,
    pub reason: String,
}

/// Result of [`lower_wave_ops`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WaveLoweringReport {
    /// The module was rewritten (no blockers, post-lowering validation passed).
    pub applied: bool,
    pub lowered: Vec<LoweredWaveOp>,
    pub blockers: Vec<WaveBlocker>,
    /// Generated helper functions, in module order.
    pub helper_functions: Vec<String>,
    /// The lowered module still contains subgroup operations (native strategy).
    pub requires_subgroups_feature: bool,
    /// Bytes of `var<workgroup>` scratch added by the emulation helpers.
    pub workgroup_scratch_bytes: u32,
}

impl WaveLoweringReport {
    /// No wave operation was found.
    pub fn is_empty(&self) -> bool {
        self.lowered.is_empty() && self.blockers.is_empty()
    }

    pub fn has_blockers(&self) -> bool {
        !self.blockers.is_empty()
    }

    /// Directives that must precede the WGSL written from the lowered module.
    pub const fn wgsl_preamble(&self) -> &'static str {
        if self.requires_subgroups_feature {
            WGSL_SUBGROUPS_ENABLE
        } else {
            ""
        }
    }

    /// Validator configured for the lowered module: WebGPU baseline, plus the `subgroups`
    /// capability (compute and fragment stages) when the native strategy kept operations.
    pub fn validator(&self) -> Validator {
        if self.requires_subgroups_feature {
            let mut validator = Validator::new(ValidationFlags::all(), Capabilities::SUBGROUP);
            validator
                .subgroup_stages(ShaderStages::COMPUTE | ShaderStages::FRAGMENT)
                .subgroup_operations(SubgroupOperationSet::all());
            validator
        } else {
            Validator::new(ValidationFlags::all(), Capabilities::empty())
        }
    }

    /// Render blockers (and approximations) as [`CompatAudit`] findings.
    pub fn to_compat_audit(&self) -> CompatAudit {
        let mut audit = CompatAudit::default();
        for blocker in &self.blockers {
            let what = blocker
                .intrinsic
                .map(|i| format!("{i} [{}]", i.hlsl_name()))
                .unwrap_or_else(|| "wave lowering".to_string());
            audit.push_unique(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::SubgroupWave64,
                message: format!("{what} in {}: {}", blocker.location, blocker.reason),
                hint: Some(
                    "choose another WaveStrategy or restructure the wave use upstream".to_string(),
                ),
            });
        }
        for op in &self.lowered {
            let severity = if op.condition.may_differ_from_wave64() {
                CompatSeverity::Warning
            } else {
                CompatSeverity::Rewritten
            };
            audit.push_unique(CompatIssue {
                severity,
                kind: CompatIssueKind::SubgroupWave64,
                message: format!(
                    "{} [{}] in {} lowered by {} ({})",
                    op.intrinsic,
                    op.intrinsic.hlsl_name(),
                    op.location,
                    op.strategy,
                    op.condition
                ),
                hint: None,
            });
        }
        audit
    }
}

/// Lower every naga IR subgroup operation and subgroup built-in of `module` under
/// `options.strategy` (see the module documentation for the per-intrinsic table).
///
/// The rewrite is transactional: if any use cannot be lowered, every blocker is reported and the
/// module is left unchanged. On success the module validates with [`WaveLoweringReport::validator`]
/// capabilities plus whatever the input already needed.
pub fn lower_wave_ops(module: &mut Module, options: &WaveLoweringOptions) -> WaveLoweringReport {
    if !module_has_wave_ops(module) {
        return WaveLoweringReport::default();
    }

    let mut pre_validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    pre_validator
        .subgroup_stages(ShaderStages::all())
        .subgroup_operations(SubgroupOperationSet::all());
    let info = match pre_validator.validate(module) {
        Ok(info) => info,
        Err(err) => {
            return WaveLoweringReport {
                blockers: vec![WaveBlocker {
                    location: "module".to_string(),
                    intrinsic: None,
                    reason: format!(
                        "module does not validate even with subgroup capabilities, so operand types and control flow cannot be trusted: {}",
                        error_chain(err.as_inner())
                    ),
                }],
                ..WaveLoweringReport::default()
            };
        }
    };

    let plan = plan_lowering(module, &info, options);
    if !plan.report.blockers.is_empty() {
        return plan.report;
    }

    let backup = module.clone();
    let mut report = materialize(module, plan, options);
    let mut post_capabilities = Capabilities::all()
        - Capabilities::SUBGROUP
        - Capabilities::SUBGROUP_BARRIER
        - Capabilities::SUBGROUP_VERTEX_STAGE;
    if report.requires_subgroups_feature {
        post_capabilities |= Capabilities::SUBGROUP;
    }
    let mut post_validator = Validator::new(ValidationFlags::all(), post_capabilities);
    post_validator
        .subgroup_stages(ShaderStages::COMPUTE | ShaderStages::FRAGMENT)
        .subgroup_operations(SubgroupOperationSet::all());
    if let Err(err) = post_validator.validate(module) {
        *module = backup;
        report.applied = false;
        report.blockers.push(WaveBlocker {
            location: "module".to_string(),
            intrinsic: None,
            reason: format!(
                "internal: lowered module failed validation, lowering rolled back: {}",
                error_chain(err.as_inner())
            ),
        });
        return report;
    }
    report.applied = true;
    report
}

/// `error: source: source ...` on one line.
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(inner) = source {
        let _ = write!(text, ": {inner}");
        source = inner.source();
    }
    text
}

fn module_has_wave_ops(module: &Module) -> bool {
    let has_subgroup_builtin = module.entry_points.iter().any(|ep| {
        ep.function
            .arguments
            .iter()
            .any(|arg| subgroup_builtin_of(arg, &module.types).is_some())
    });
    has_subgroup_builtin
        || module
            .functions
            .iter()
            .map(|(_, f)| f)
            .chain(module.entry_points.iter().map(|ep| &ep.function))
            .any(|f| block_has_subgroup_statement(&f.body))
}

fn block_has_subgroup_statement(block: &Block) -> bool {
    block.iter().any(|stmt| match stmt {
        Statement::SubgroupBallot { .. }
        | Statement::SubgroupCollectiveOperation { .. }
        | Statement::SubgroupGather { .. } => true,
        Statement::Block(inner) => block_has_subgroup_statement(inner),
        Statement::If { accept, reject, .. } => {
            block_has_subgroup_statement(accept) || block_has_subgroup_statement(reject)
        }
        Statement::Switch { cases, .. } => {
            cases.iter().any(|c| block_has_subgroup_statement(&c.body))
        }
        Statement::Loop {
            body, continuing, ..
        } => block_has_subgroup_statement(body) || block_has_subgroup_statement(continuing),
        _ => false,
    })
}

/// The subgroup built-in carried by an entry-point argument (top level or struct member).
fn subgroup_builtin_of(arg: &FunctionArgument, types: &UniqueArena<Type>) -> Option<BuiltIn> {
    let is_subgroup = |b: &Binding| match b {
        Binding::BuiltIn(
            builtin @ (BuiltIn::SubgroupSize
            | BuiltIn::SubgroupInvocationId
            | BuiltIn::NumSubgroups
            | BuiltIn::SubgroupId),
        ) => Some(*builtin),
        _ => None,
    };
    if let Some(binding) = &arg.binding {
        return is_subgroup(binding);
    }
    if let TypeInner::Struct { members, .. } = &types[arg.ty].inner {
        return members
            .iter()
            .find_map(|m| m.binding.as_ref().and_then(is_subgroup));
    }
    None
}

// ---------------------------------------------------------------------------
// naga IR lowering: analysis
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FnRef {
    Function(Handle<Function>),
    Entry(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum GatherKind {
    BroadcastFirst,
    Broadcast,
    Shuffle,
    ShuffleXor,
    ShuffleUp,
    ShuffleDown,
}

impl GatherKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::BroadcastFirst => "read_first_lane",
            Self::Broadcast => "read_lane_at",
            Self::Shuffle => "shuffle",
            Self::ShuffleXor => "shuffle_xor",
            Self::ShuffleUp => "shuffle_up",
            Self::ShuffleDown => "shuffle_down",
        }
    }
}

#[derive(Clone, Copy)]
enum SiteKind {
    Ballot {
        predicate: Option<Handle<Expression>>,
    },
    Collective {
        op: SubgroupOperation,
        collective: CollectiveOperation,
        argument: Handle<Expression>,
    },
    Gather {
        mode: GatherMode,
        argument: Handle<Expression>,
    },
}

#[derive(Clone, Copy)]
struct Site {
    result: Handle<Expression>,
    kind: SiteKind,
    /// Uniform control flow relative to the enclosing function's entry.
    uniform_position: bool,
}

impl Site {
    fn intrinsic(&self) -> WaveIntrinsic {
        match self.kind {
            SiteKind::Ballot { .. } => WaveIntrinsic::Ballot,
            SiteKind::Collective { op, collective, .. } => {
                WaveIntrinsic::Collective(WaveOp::from_naga(op), WaveGroupOp::from_naga(collective))
            }
            SiteKind::Gather { mode, .. } => match mode {
                GatherMode::BroadcastFirst => WaveIntrinsic::ReadFirstLane,
                GatherMode::Broadcast(_) => WaveIntrinsic::ReadLaneAt,
                GatherMode::Shuffle(_) => WaveIntrinsic::Shuffle,
                GatherMode::ShuffleXor(_) => WaveIntrinsic::ShuffleXor,
                GatherMode::ShuffleUp(_) => WaveIntrinsic::ShuffleUp,
                GatherMode::ShuffleDown(_) => WaveIntrinsic::ShuffleDown,
                GatherMode::QuadBroadcast(_) => WaveIntrinsic::QuadBroadcast,
                GatherMode::QuadSwap(_) => WaveIntrinsic::QuadSwap,
            },
        }
    }
}

/// Conservative syntactic value-uniformity analysis (see module docs, `UniformOperand`).
struct UniformityOracle<'a> {
    module: &'a Module,
    function: &'a Function,
    is_entry: bool,
    memo: HashMap<Handle<Expression>, bool>,
}

impl UniformityOracle<'_> {
    fn is_uniform(&mut self, handle: Handle<Expression>) -> bool {
        if let Some(&known) = self.memo.get(&handle) {
            return known;
        }
        let uniform = match self.function.expressions[handle] {
            Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::Override(_)
            | Expression::ZeroValue(_) => true,
            Expression::Splat { value, .. } => self.is_uniform(value),
            Expression::Swizzle { vector, .. } => self.is_uniform(vector),
            Expression::Unary { expr, .. }
            | Expression::As { expr, .. }
            | Expression::Relational { argument: expr, .. } => self.is_uniform(expr),
            Expression::AccessIndex { base, .. } => self.is_uniform(base),
            Expression::Access { base, index } => self.is_uniform(base) && self.is_uniform(index),
            Expression::Binary { left, right, .. } => {
                self.is_uniform(left) && self.is_uniform(right)
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => self.is_uniform(condition) && self.is_uniform(accept) && self.is_uniform(reject),
            Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                self.is_uniform(arg)
                    && [arg1, arg2, arg3]
                        .into_iter()
                        .flatten()
                        .all(|h| self.is_uniform(h))
            }
            Expression::Compose { ref components, .. } => {
                let components = components.clone();
                components.into_iter().all(|h| self.is_uniform(h))
            }
            Expression::Load { pointer } => self.is_uniform_pointer(pointer),
            Expression::FunctionArgument(index) => {
                self.is_entry
                    && matches!(
                        self.function
                            .arguments
                            .get(index as usize)
                            .and_then(|a| a.binding.as_ref()),
                        Some(Binding::BuiltIn(
                            BuiltIn::WorkGroupId | BuiltIn::NumWorkGroups | BuiltIn::WorkGroupSize
                        ))
                    )
            }
            _ => false,
        };
        self.memo.insert(handle, uniform);
        uniform
    }

    fn is_uniform_pointer(&mut self, pointer: Handle<Expression>) -> bool {
        match self.function.expressions[pointer] {
            Expression::GlobalVariable(global) => {
                match self.module.global_variables[global].space {
                    AddressSpace::Uniform | AddressSpace::Immediate => true,
                    AddressSpace::Storage { access } => {
                        !access.contains(naga::StorageAccess::STORE)
                    }
                    _ => false,
                }
            }
            Expression::Access { base, index } => {
                self.is_uniform_pointer(base) && self.is_uniform(index)
            }
            Expression::AccessIndex { base, .. } => self.is_uniform_pointer(base),
            _ => false,
        }
    }
}

struct FunctionScan {
    sites: Vec<Site>,
    calls: Vec<(Handle<Function>, bool)>,
}

/// Collects subgroup sites and call sites with their structural control-flow uniformity.
struct ControlFlowWalker<'a> {
    oracle: UniformityOracle<'a>,
    scan: FunctionScan,
}

impl ControlFlowWalker<'_> {
    /// Returns `(left_function_nonuniformly, left_construct_nonuniformly)`.
    fn walk(&mut self, block: &Block, mut uniform: bool) -> (bool, bool) {
        let mut left_function = false;
        let mut left_construct = false;
        for statement in block.iter() {
            match *statement {
                Statement::SubgroupBallot { result, predicate } => self.scan.sites.push(Site {
                    result,
                    kind: SiteKind::Ballot { predicate },
                    uniform_position: uniform,
                }),
                Statement::SubgroupCollectiveOperation {
                    op,
                    collective_op,
                    argument,
                    result,
                } => self.scan.sites.push(Site {
                    result,
                    kind: SiteKind::Collective {
                        op,
                        collective: collective_op,
                        argument,
                    },
                    uniform_position: uniform,
                }),
                Statement::SubgroupGather {
                    mode,
                    argument,
                    result,
                } => self.scan.sites.push(Site {
                    result,
                    kind: SiteKind::Gather { mode, argument },
                    uniform_position: uniform,
                }),
                Statement::Call { function, .. } => self.scan.calls.push((function, uniform)),
                Statement::Block(ref inner) => {
                    let (f, c) = self.walk(inner, uniform);
                    left_function |= f;
                    left_construct |= c;
                }
                Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    let branch_uniform = uniform && self.oracle.is_uniform(condition);
                    let (f1, c1) = self.walk(accept, branch_uniform);
                    let (f2, c2) = self.walk(reject, branch_uniform);
                    left_function |= f1 | f2;
                    left_construct |= c1 | c2;
                }
                Statement::Switch {
                    selector,
                    ref cases,
                } => {
                    let branch_uniform = uniform && self.oracle.is_uniform(selector);
                    for case in cases {
                        let (f, c) = self.walk(&case.body, branch_uniform);
                        left_function |= f;
                        left_construct |= c;
                    }
                }
                Statement::Loop {
                    ref body,
                    ref continuing,
                    ..
                } => {
                    // Loop bodies are conservatively non-uniform; their break/continue stay local.
                    let (f1, _) = self.walk(body, false);
                    let (f2, _) = self.walk(continuing, false);
                    left_function |= f1 | f2;
                }
                Statement::Return { .. } | Statement::Kill => left_function |= !uniform,
                Statement::Break | Statement::Continue => left_construct |= !uniform,
                _ => {}
            }
            if left_function || left_construct {
                uniform = false;
            }
        }
        (left_function, left_construct)
    }
}

fn scan_function(module: &Module, function: &Function, is_entry: bool) -> FunctionScan {
    let mut walker = ControlFlowWalker {
        oracle: UniformityOracle {
            module,
            function,
            is_entry,
            memo: HashMap::new(),
        },
        scan: FunctionScan {
            sites: Vec::new(),
            calls: Vec::new(),
        },
    };
    walker.walk(&function.body, true);
    walker.scan
}

/// Per-function call context: reaching entry points and uniform entry.
struct CallContext {
    entries: BTreeSet<usize>,
    uniformly_entered: bool,
}

fn compute_call_contexts(
    module: &Module,
    scans: &BTreeMap<FnRef, FunctionScan>,
) -> BTreeMap<FnRef, CallContext> {
    let mut contexts: BTreeMap<FnRef, CallContext> = BTreeMap::new();
    for index in 0..module.entry_points.len() {
        contexts.insert(
            FnRef::Entry(index),
            CallContext {
                entries: BTreeSet::from([index]),
                uniformly_entered: true,
            },
        );
    }
    // Incoming call sites per callee: (caller, uniform position).
    let mut incoming: HashMap<Handle<Function>, Vec<(FnRef, bool)>> = HashMap::new();
    for (&caller, scan) in scans {
        for &(callee, uniform) in &scan.calls {
            incoming.entry(callee).or_default().push((caller, uniform));
        }
    }
    // naga requires callees to precede callers, so descending handle order visits every caller
    // before its callees.
    let handles: Vec<Handle<Function>> = module.functions.iter().map(|(h, _)| h).collect();
    for &handle in handles.iter().rev() {
        let mut context = CallContext {
            entries: BTreeSet::new(),
            uniformly_entered: true,
        };
        let mut reached = false;
        for &(caller, uniform) in incoming.get(&handle).into_iter().flatten() {
            if let Some(caller_context) = contexts.get(&caller)
                && !caller_context.entries.is_empty()
            {
                reached = true;
                context
                    .entries
                    .extend(caller_context.entries.iter().copied());
                context.uniformly_entered &= uniform && caller_context.uniformly_entered;
            }
        }
        if !reached {
            context.uniformly_entered = false;
        }
        contexts.insert(FnRef::Function(handle), context);
    }
    contexts
}

// ---------------------------------------------------------------------------
// naga IR lowering: planning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum HelperKey {
    Identity {
        ty: Handle<Type>,
    },
    Wave1Exclusive {
        op: SubgroupOperation,
        ty: Handle<Type>,
    },
    Wave1Ballot,
    EmuCollective {
        op: SubgroupOperation,
        collective: CollectiveOperation,
        ty: Handle<Type>,
    },
    EmuGather {
        kind: GatherKind,
        ty: Handle<Type>,
        signed_index: bool,
    },
    EmuBallot,
    EmuLaneIndex,
    EmuWaveIndex,
}

impl HelperKey {
    /// Element type of the `var<workgroup>` scratch array this helper exchanges values through.
    fn scratch_element(self) -> Option<ScratchElement> {
        match self {
            Self::EmuCollective { ty, .. } | Self::EmuGather { ty, .. } => {
                Some(ScratchElement::Of(ty))
            }
            Self::EmuBallot => Some(ScratchElement::U32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ScratchElement {
    U32,
    Of(Handle<Type>),
}

#[derive(Clone, Copy)]
enum ArgSource {
    Expr(Handle<Expression>),
    True,
}

enum SiteAction {
    Keep,
    Call {
        helper: HelperKey,
        args: Vec<ArgSource>,
    },
}

enum BuiltinAction {
    Keep,
    Literal(u32),
    Helper(HelperKey),
}

struct Plan {
    sites: HashMap<(FnRef, Handle<Expression>), SiteAction>,
    builtin_args: BTreeMap<usize, Vec<(usize, BuiltinAction)>>,
    prologue_entries: BTreeSet<usize>,
    helpers: BTreeSet<HelperKey>,
    report: WaveLoweringReport,
}

fn fn_label(module: &Module, fn_ref: FnRef) -> String {
    match fn_ref {
        FnRef::Entry(index) => format!("entry point '{}'", module.entry_points[index].name),
        FnRef::Function(handle) => match &module.functions[handle].name {
            Some(name) => format!("function '{name}'"),
            None => format!("function #{}", handle.index()),
        },
    }
}

fn function_of(module: &Module, fn_ref: FnRef) -> &Function {
    match fn_ref {
        FnRef::Entry(index) => &module.entry_points[index].function,
        FnRef::Function(handle) => &module.functions[handle],
    }
}

fn workgroup_invocations(size: [u32; 3]) -> u32 {
    size[0].saturating_mul(size[1]).saturating_mul(size[2])
}

/// Scalar description of a type usable by the emulation helpers.
#[derive(Clone, Copy)]
struct ValueShape {
    scalar: Scalar,
    size: Option<VectorSize>,
}

fn value_shape(types: &UniqueArena<Type>, ty: Handle<Type>) -> Option<ValueShape> {
    match types[ty].inner {
        TypeInner::Scalar(scalar) => Some(ValueShape { scalar, size: None }),
        TypeInner::Vector { size, scalar } => Some(ValueShape {
            scalar,
            size: Some(size),
        }),
        _ => None,
    }
}

/// 32-bit numeric scalars/vectors and scalar `bool` (stored as `u32` in scratch).
fn emulation_supports(shape: ValueShape) -> bool {
    match shape.scalar.kind {
        ScalarKind::Bool => shape.size.is_none(),
        ScalarKind::Float | ScalarKind::Sint | ScalarKind::Uint => shape.scalar.width == 4,
        _ => false,
    }
}

/// The exclusive scan's identity element (`0` for add, `1` for mul) is representable. naga's
/// validator only accepts add/mul scans, so other operations never reach the planner.
fn exclusive_identity_exists(op: SubgroupOperation, shape: ValueShape) -> bool {
    matches!(op, SubgroupOperation::Add | SubgroupOperation::Mul)
        && shape.scalar.width == 4
        && matches!(
            shape.scalar.kind,
            ScalarKind::Float | ScalarKind::Sint | ScalarKind::Uint
        )
}

fn index_scalar_kind(
    module: &Module,
    info: &ModuleInfo,
    fn_ref: FnRef,
    expr: Handle<Expression>,
) -> Option<ScalarKind> {
    let function_info = match fn_ref {
        FnRef::Entry(index) => info.get_entry_point(index),
        FnRef::Function(handle) => &info[handle],
    };
    match *function_info[expr].ty.inner_with(&module.types) {
        TypeInner::Scalar(Scalar {
            kind: kind @ (ScalarKind::Uint | ScalarKind::Sint),
            width: 4,
        }) => Some(kind),
        _ => None,
    }
}

/// A subgroup statement being planned.
struct SiteTarget {
    fn_ref: FnRef,
    site: Site,
    location: String,
    intrinsic: WaveIntrinsic,
    result_ty: Option<Handle<Type>>,
}

struct Planner<'a> {
    module: &'a Module,
    info: &'a ModuleInfo,
    options: &'a WaveLoweringOptions,
    plan: Plan,
}

fn plan_lowering(module: &Module, info: &ModuleInfo, options: &WaveLoweringOptions) -> Plan {
    let mut scans: BTreeMap<FnRef, FunctionScan> = BTreeMap::new();
    for (handle, function) in module.functions.iter() {
        scans.insert(
            FnRef::Function(handle),
            scan_function(module, function, false),
        );
    }
    for (index, ep) in module.entry_points.iter().enumerate() {
        scans.insert(
            FnRef::Entry(index),
            scan_function(module, &ep.function, true),
        );
    }
    let contexts = compute_call_contexts(module, &scans);

    let mut planner = Planner {
        module,
        info,
        options,
        plan: Plan {
            sites: HashMap::new(),
            builtin_args: BTreeMap::new(),
            prologue_entries: BTreeSet::new(),
            helpers: BTreeSet::new(),
            report: WaveLoweringReport::default(),
        },
    };

    for (&fn_ref, scan) in &scans {
        let context = &contexts[&fn_ref];
        for site in &scan.sites {
            planner.plan_site(fn_ref, context, site);
        }
    }
    for index in 0..module.entry_points.len() {
        planner.plan_builtins(index);
    }
    planner.check_emulation_budget();
    planner.plan
}

impl Planner<'_> {
    fn lowered(
        &mut self,
        location: String,
        intrinsic: WaveIntrinsic,
        condition: LoweringCondition,
        helper: Option<HelperKey>,
    ) {
        let helper_name =
            helper.map(|key| helper_name(&self.options.helper_prefix, key, &self.module.types));
        if let Some(key) = helper {
            self.plan.helpers.insert(key);
        }
        self.plan.report.lowered.push(LoweredWaveOp {
            location,
            intrinsic,
            strategy: self.options.strategy,
            condition,
            helper: helper_name,
        });
    }

    fn blocked(&mut self, location: String, intrinsic: WaveIntrinsic, reason: impl Into<String>) {
        self.plan.report.blockers.push(WaveBlocker {
            location,
            intrinsic: Some(intrinsic),
            reason: reason.into(),
        });
    }

    fn plan_site(&mut self, fn_ref: FnRef, context: &CallContext, site: &Site) {
        let module = self.module;
        let function = function_of(module, fn_ref);
        let (result_ty, argument) = match site.kind {
            SiteKind::Ballot { .. } => (None, None),
            SiteKind::Collective { argument, .. } | SiteKind::Gather { argument, .. } => {
                let ty = match function.expressions[site.result] {
                    Expression::SubgroupOperationResult { ty } => Some(ty),
                    _ => None,
                };
                (ty, Some(argument))
            }
        };
        let target = SiteTarget {
            fn_ref,
            site: *site,
            location: fn_label(module, fn_ref),
            intrinsic: site.intrinsic(),
            result_ty,
        };

        if context.entries.is_empty() {
            self.plan_unreachable(target);
            return;
        }

        // Uniform-operand shortcut: exact under every strategy.
        let idempotent = match site.kind {
            SiteKind::Collective { op, collective, .. } => {
                WaveOp::from_naga(op).is_idempotent()
                    && collective != CollectiveOperation::ExclusiveScan
            }
            SiteKind::Gather { .. } => true,
            SiteKind::Ballot { .. } => false,
        };
        if let (true, Some(ty), Some(argument)) = (idempotent, result_ty, argument) {
            let mut oracle = UniformityOracle {
                module,
                function,
                is_entry: matches!(fn_ref, FnRef::Entry(_)),
                memo: HashMap::new(),
            };
            if oracle.is_uniform(argument) {
                let helper = HelperKey::Identity { ty };
                self.plan.sites.insert(
                    (fn_ref, site.result),
                    SiteAction::Call {
                        helper,
                        args: vec![ArgSource::Expr(argument)],
                    },
                );
                self.lowered(
                    target.location,
                    target.intrinsic,
                    LoweringCondition::UniformOperand,
                    Some(helper),
                );
                return;
            }
        }

        let stages: BTreeSet<ShaderStage> = context
            .entries
            .iter()
            .map(|&i| module.entry_points[i].stage)
            .collect();
        match self.options.strategy {
            WaveStrategy::SingleLane => self.plan_single_lane(target),
            WaveStrategy::NativeSubgroups => {
                if stages
                    .iter()
                    .any(|s| !matches!(s, ShaderStage::Compute | ShaderStage::Fragment))
                {
                    self.blocked(
                        target.location,
                        target.intrinsic,
                        "WGSL subgroup operations are only available in compute and fragment shaders",
                    );
                    return;
                }
                let condition = if target.intrinsic.is_quad()
                    && stages.iter().all(|s| *s == ShaderStage::Fragment)
                {
                    LoweringCondition::Exact
                } else {
                    LoweringCondition::RuntimeSubgroupSize64
                };
                self.plan
                    .sites
                    .insert((fn_ref, site.result), SiteAction::Keep);
                self.plan.report.requires_subgroups_feature = true;
                self.lowered(target.location, target.intrinsic, condition, None);
            }
            WaveStrategy::WorkgroupEmulation => self.plan_emulation(target, context, &stages),
        }
    }

    /// Code no entry point reaches never executes, so any valid replacement is exact.
    fn plan_unreachable(&mut self, target: SiteTarget) {
        let SiteTarget {
            fn_ref,
            site,
            location,
            intrinsic,
            result_ty,
        } = target;
        let (helper, args) = match (site.kind, result_ty) {
            (SiteKind::Ballot { predicate }, _) => (
                HelperKey::Wave1Ballot,
                vec![predicate.map_or(ArgSource::True, ArgSource::Expr)],
            ),
            (
                SiteKind::Collective {
                    op,
                    collective: CollectiveOperation::ExclusiveScan,
                    argument,
                },
                Some(ty),
            ) => (
                HelperKey::Wave1Exclusive { op, ty },
                vec![ArgSource::Expr(argument)],
            ),
            (
                SiteKind::Collective { argument, .. } | SiteKind::Gather { argument, .. },
                Some(ty),
            ) => (HelperKey::Identity { ty }, vec![ArgSource::Expr(argument)]),
            _ => {
                self.blocked(location, intrinsic, "operation result has no type");
                return;
            }
        };
        self.plan
            .sites
            .insert((fn_ref, site.result), SiteAction::Call { helper, args });
        self.lowered(
            format!("{location} (unreachable from every entry point)"),
            intrinsic,
            LoweringCondition::Exact,
            Some(helper),
        );
    }

    fn plan_single_lane(&mut self, target: SiteTarget) {
        let SiteTarget {
            fn_ref,
            site,
            location,
            intrinsic,
            result_ty,
        } = target;
        if intrinsic.is_quad() {
            self.blocked(
                location,
                intrinsic,
                "quad operation with a non-uniform operand: a one-lane quad would silently zero screen-space differences",
            );
            return;
        }
        if !self.options.allow_wave1_model {
            self.blocked(
                location,
                intrinsic,
                "cross-lane operation with an operand not proven uniform; SingleLane is exact only for uniform operands of idempotent operations (set allow_wave1_model to accept the wave-of-1 approximation)",
            );
            return;
        }
        let (helper, args) = match (site.kind, result_ty) {
            (SiteKind::Ballot { predicate }, _) => (
                HelperKey::Wave1Ballot,
                vec![predicate.map_or(ArgSource::True, ArgSource::Expr)],
            ),
            (
                SiteKind::Collective {
                    op,
                    collective: CollectiveOperation::ExclusiveScan,
                    argument,
                },
                Some(ty),
            ) => {
                let supported = value_shape(&self.module.types, ty)
                    .is_some_and(|shape| exclusive_identity_exists(op, shape));
                if !supported {
                    self.blocked(
                        location,
                        intrinsic,
                        "exclusive scan without a representable identity element (only add/mul scans of 32-bit numbers are supported)",
                    );
                    return;
                }
                (
                    HelperKey::Wave1Exclusive { op, ty },
                    vec![ArgSource::Expr(argument)],
                )
            }
            (
                SiteKind::Collective { argument, .. } | SiteKind::Gather { argument, .. },
                Some(ty),
            ) => (HelperKey::Identity { ty }, vec![ArgSource::Expr(argument)]),
            _ => {
                self.blocked(location, intrinsic, "operation result has no type");
                return;
            }
        };
        self.plan
            .sites
            .insert((fn_ref, site.result), SiteAction::Call { helper, args });
        self.lowered(
            location,
            intrinsic,
            LoweringCondition::Wave1Model,
            Some(helper),
        );
    }

    fn plan_emulation(
        &mut self,
        target: SiteTarget,
        context: &CallContext,
        stages: &BTreeSet<ShaderStage>,
    ) {
        let SiteTarget {
            fn_ref,
            site,
            location,
            intrinsic,
            result_ty,
        } = target;
        let module = self.module;
        if stages.iter().any(|s| *s != ShaderStage::Compute) {
            self.blocked(
                location,
                intrinsic,
                "workgroup emulation needs a compute stage (workgroup memory and barriers); reached from a non-compute entry point",
            );
            return;
        }
        if let Some(&index) = context
            .entries
            .iter()
            .find(|&&i| module.entry_points[i].workgroup_size_overrides.is_some())
        {
            self.blocked(
                location,
                intrinsic,
                format!(
                    "entry point '{}' has a pipeline-overridable workgroup size; the wave partition must be known when lowering",
                    module.entry_points[index].name
                ),
            );
            return;
        }
        if !(site.uniform_position && context.uniformly_entered) {
            self.blocked(
                location,
                intrinsic,
                "reached in potentially non-uniform control flow (inside a loop, under a branch on a value not proven uniform, after a non-uniform return/break, or through such a call site); the emulation's workgroup barriers would be unsound",
            );
            return;
        }
        if intrinsic.is_quad() {
            self.blocked(
                location,
                intrinsic,
                "quad operations are not emulated: D3D's compute-shader quad layout is not defined in terms of local_invocation_index",
            );
            return;
        }

        let (helper, args) = match site.kind {
            SiteKind::Ballot { predicate } => (
                HelperKey::EmuBallot,
                vec![predicate.map_or(ArgSource::True, ArgSource::Expr)],
            ),
            SiteKind::Collective {
                op,
                collective,
                argument,
            } => {
                let Some(ty) = result_ty else {
                    self.blocked(location, intrinsic, "operation result has no type");
                    return;
                };
                let Some(shape) = value_shape(&module.types, ty).filter(|s| emulation_supports(*s))
                else {
                    self.blocked(
                        location,
                        intrinsic,
                        "operand type is not a 32-bit scalar/vector or scalar bool",
                    );
                    return;
                };
                if collective == CollectiveOperation::ExclusiveScan
                    && !exclusive_identity_exists(op, shape)
                {
                    self.blocked(
                        location,
                        intrinsic,
                        "exclusive scan without a representable identity element (only add/mul scans of 32-bit numbers are supported)",
                    );
                    return;
                }
                (
                    HelperKey::EmuCollective { op, collective, ty },
                    vec![ArgSource::Expr(argument)],
                )
            }
            SiteKind::Gather { mode, argument } => {
                let Some(ty) = result_ty else {
                    self.blocked(location, intrinsic, "operation result has no type");
                    return;
                };
                if !value_shape(&module.types, ty).is_some_and(emulation_supports) {
                    self.blocked(
                        location,
                        intrinsic,
                        "operand type is not a 32-bit scalar/vector or scalar bool",
                    );
                    return;
                }
                let (kind, index) = match mode {
                    GatherMode::BroadcastFirst => (GatherKind::BroadcastFirst, None),
                    GatherMode::Broadcast(i) => (GatherKind::Broadcast, Some(i)),
                    GatherMode::Shuffle(i) => (GatherKind::Shuffle, Some(i)),
                    GatherMode::ShuffleXor(i) => (GatherKind::ShuffleXor, Some(i)),
                    GatherMode::ShuffleUp(i) => (GatherKind::ShuffleUp, Some(i)),
                    GatherMode::ShuffleDown(i) => (GatherKind::ShuffleDown, Some(i)),
                    GatherMode::QuadBroadcast(_) | GatherMode::QuadSwap(_) => return,
                };
                let mut args = vec![ArgSource::Expr(argument)];
                let mut signed_index = false;
                if let Some(index) = index {
                    match index_scalar_kind(module, self.info, fn_ref, index) {
                        Some(kind) => signed_index = kind == ScalarKind::Sint,
                        None => {
                            self.blocked(
                                location,
                                intrinsic,
                                "lane index operand is not a 32-bit integer scalar",
                            );
                            return;
                        }
                    }
                    args.push(ArgSource::Expr(index));
                }
                (
                    HelperKey::EmuGather {
                        kind,
                        ty,
                        signed_index,
                    },
                    args,
                )
            }
        };
        self.plan
            .prologue_entries
            .extend(context.entries.iter().copied());
        self.plan
            .sites
            .insert((fn_ref, site.result), SiteAction::Call { helper, args });
        self.lowered(
            location,
            intrinsic,
            LoweringCondition::WorkgroupUniformControlFlow,
            Some(helper),
        );
    }

    fn plan_builtins(&mut self, index: usize) {
        let module = self.module;
        let ep = &module.entry_points[index];
        for (arg_index, arg) in ep.function.arguments.iter().enumerate() {
            let Some(builtin) = subgroup_builtin_of(arg, &module.types) else {
                continue;
            };
            let intrinsic = match builtin {
                BuiltIn::SubgroupSize => WaveIntrinsic::LaneCount,
                BuiltIn::SubgroupInvocationId => WaveIntrinsic::LaneIndex,
                BuiltIn::NumSubgroups => WaveIntrinsic::NumWaves,
                _ => WaveIntrinsic::WaveIndex,
            };
            let location = format!("entry point '{}' argument #{arg_index}", ep.name);
            if arg.binding.is_none() {
                self.blocked(
                    location,
                    intrinsic,
                    "subgroup built-in inside an entry-point struct argument is not supported",
                );
                continue;
            }
            if module.types[arg.ty].inner != TypeInner::Scalar(Scalar::U32) {
                self.blocked(
                    location,
                    intrinsic,
                    "subgroup built-in argument is not a u32 scalar",
                );
                continue;
            }
            let compute = ep.stage == ShaderStage::Compute;
            let (action, condition) = match self.options.strategy {
                WaveStrategy::SingleLane => match intrinsic {
                    WaveIntrinsic::LaneCount | WaveIntrinsic::LaneIndex
                        if self.options.allow_wave1_model =>
                    {
                        let value = u32::from(intrinsic == WaveIntrinsic::LaneCount);
                        (BuiltinAction::Literal(value), LoweringCondition::Wave1Model)
                    }
                    WaveIntrinsic::LaneCount | WaveIntrinsic::LaneIndex => {
                        self.blocked(
                            location,
                            intrinsic,
                            "SingleLane maps this built-in only under the wave-of-1 model (set allow_wave1_model); WorkgroupEmulation lowers it exactly",
                        );
                        continue;
                    }
                    _ => {
                        self.blocked(
                            location,
                            intrinsic,
                            "no wave-of-1 mapping for NumSubgroups/SubgroupId (not produced by HLSL wave intrinsics)",
                        );
                        continue;
                    }
                },
                WaveStrategy::NativeSubgroups => {
                    let stage_ok = compute
                        || (ep.stage == ShaderStage::Fragment
                            && matches!(
                                intrinsic,
                                WaveIntrinsic::LaneCount | WaveIntrinsic::LaneIndex
                            ));
                    if !stage_ok {
                        self.blocked(
                            location,
                            intrinsic,
                            "WGSL subgroup built-ins are not available in this stage",
                        );
                        continue;
                    }
                    self.plan.report.requires_subgroups_feature = true;
                    (
                        BuiltinAction::Keep,
                        LoweringCondition::RuntimeSubgroupSize64,
                    )
                }
                WaveStrategy::WorkgroupEmulation => match intrinsic {
                    WaveIntrinsic::LaneCount => (
                        BuiltinAction::Literal(WAVE64_LANE_COUNT),
                        LoweringCondition::Exact,
                    ),
                    _ if !compute => {
                        self.blocked(
                            location,
                            intrinsic,
                            "lane/wave index emulation needs a compute stage (derived from local_invocation_index)",
                        );
                        continue;
                    }
                    _ if ep.workgroup_size_overrides.is_some() => {
                        self.blocked(
                            location,
                            intrinsic,
                            "pipeline-overridable workgroup size: the wave partition must be known when lowering",
                        );
                        continue;
                    }
                    WaveIntrinsic::NumWaves => (
                        BuiltinAction::Literal(
                            workgroup_invocations(ep.workgroup_size).div_ceil(WAVE64_LANE_COUNT),
                        ),
                        LoweringCondition::Exact,
                    ),
                    WaveIntrinsic::LaneIndex => {
                        self.plan.prologue_entries.insert(index);
                        (
                            BuiltinAction::Helper(HelperKey::EmuLaneIndex),
                            LoweringCondition::Exact,
                        )
                    }
                    _ => {
                        self.plan.prologue_entries.insert(index);
                        (
                            BuiltinAction::Helper(HelperKey::EmuWaveIndex),
                            LoweringCondition::Exact,
                        )
                    }
                },
            };
            let helper = match action {
                BuiltinAction::Helper(key) => Some(key),
                _ => None,
            };
            self.lowered(location, intrinsic, condition, helper);
            self.plan
                .builtin_args
                .entry(index)
                .or_default()
                .push((arg_index, action));
        }
    }

    fn check_emulation_budget(&mut self) {
        let scratch: BTreeSet<ScratchElement> = self
            .plan
            .helpers
            .iter()
            .filter_map(|key| key.scratch_element())
            .collect();
        if scratch.is_empty() {
            return;
        }
        let module = self.module;
        let lanes = self.scratch_len();
        let mut layouter = Layouter::default();
        if layouter.update(module.to_ctx()).is_err() {
            self.plan.report.blockers.push(WaveBlocker {
                location: "module".to_string(),
                intrinsic: None,
                reason: "type layout failed; workgroup storage cannot be budgeted".to_string(),
            });
            return;
        }
        let existing: u32 = module
            .global_variables
            .iter()
            .filter(|(_, var)| var.space == AddressSpace::WorkGroup)
            .map(|(_, var)| round_up(16, layouter[var.ty].size))
            .sum();
        // `bool`, `u32` and ballot scratch share one `array<u32>` (see `materialize`).
        let arrays: BTreeSet<Option<Handle<Type>>> = scratch
            .iter()
            .map(|element| match *element {
                ScratchElement::Of(ty)
                    if !is_bool_type(&module.types, ty)
                        && module.types[ty].inner != TypeInner::Scalar(Scalar::U32) =>
                {
                    Some(ty)
                }
                _ => None,
            })
            .collect();
        let added: u32 = arrays
            .iter()
            .map(|array| {
                let stride = array.map_or(4, |ty| scratch_stride(&module.types, ty));
                round_up(16, stride * lanes)
            })
            .sum();
        self.plan.report.workgroup_scratch_bytes = added;
        if existing + added > self.options.max_workgroup_storage_bytes {
            self.plan.report.blockers.push(WaveBlocker {
                location: "module".to_string(),
                intrinsic: None,
                reason: format!(
                    "workgroup emulation needs {added} bytes of scratch on top of {existing} bytes already declared, exceeding the {} byte workgroup storage budget",
                    self.options.max_workgroup_storage_bytes
                ),
            });
        }
    }

    fn scratch_len(&self) -> u32 {
        scratch_lanes(self.module, &self.plan.prologue_entries)
    }
}

/// Scratch array length: the largest workgroup among the entry points using emulation.
fn scratch_lanes(module: &Module, entries: &BTreeSet<usize>) -> u32 {
    entries
        .iter()
        .map(|&i| workgroup_invocations(module.entry_points[i].workgroup_size))
        .max()
        .unwrap_or(1)
        .max(1)
}

const fn round_up(align: u32, value: u32) -> u32 {
    value.div_ceil(align) * align
}

/// Array stride of a scratch element (`bool` is stored as `u32`).
fn scratch_stride(types: &UniqueArena<Type>, ty: Handle<Type>) -> u32 {
    match value_shape(types, ty) {
        Some(ValueShape { size: None, .. }) => 4,
        Some(ValueShape {
            size: Some(VectorSize::Bi),
            ..
        }) => 8,
        _ => 16,
    }
}

fn type_tag(types: &UniqueArena<Type>, ty: Handle<Type>) -> String {
    fn scalar_tag(scalar: Scalar) -> String {
        match scalar.kind {
            ScalarKind::Bool => "bool".to_string(),
            ScalarKind::Float => format!("f{}", scalar.width * 8),
            ScalarKind::Sint => format!("i{}", scalar.width * 8),
            ScalarKind::Uint => format!("u{}", scalar.width * 8),
            _ => format!("abstract{}", scalar.width * 8),
        }
    }
    match types[ty].inner {
        TypeInner::Scalar(scalar) => scalar_tag(scalar),
        TypeInner::Vector { size, scalar } => format!("vec{}{}", size as u8, scalar_tag(scalar)),
        _ => format!("ty{}", ty.index()),
    }
}

const fn op_tag(op: SubgroupOperation) -> &'static str {
    WaveOp::from_naga(op).as_str()
}

const fn collective_tag(collective: CollectiveOperation) -> &'static str {
    match collective {
        CollectiveOperation::Reduce => "reduce",
        CollectiveOperation::InclusiveScan => "inclusive_scan",
        CollectiveOperation::ExclusiveScan => "exclusive_scan",
    }
}

fn helper_name(prefix: &str, key: HelperKey, types: &UniqueArena<Type>) -> String {
    match key {
        HelperKey::Identity { ty } => format!("{prefix}identity_{}", type_tag(types, ty)),
        HelperKey::Wave1Exclusive { op, ty } => format!(
            "{prefix}wave1_exclusive_{}_{}",
            op_tag(op),
            type_tag(types, ty)
        ),
        HelperKey::Wave1Ballot => format!("{prefix}wave1_ballot"),
        HelperKey::EmuCollective { op, collective, ty } => format!(
            "{prefix}wave_{}_{}_{}",
            op_tag(op),
            collective_tag(collective),
            type_tag(types, ty)
        ),
        HelperKey::EmuGather {
            kind,
            ty,
            signed_index,
        } => format!(
            "{prefix}wave_{}_{}{}",
            kind.as_str(),
            type_tag(types, ty),
            if signed_index { "_i32_lane" } else { "" }
        ),
        HelperKey::EmuBallot => format!("{prefix}wave_ballot"),
        HelperKey::EmuLaneIndex => format!("{prefix}wave_lane_index"),
        HelperKey::EmuWaveIndex => format!("{prefix}wave_index"),
    }
}

/// Make `base` unique against every module-scope name of `module`.
fn unique_ir_prefix(module: &Module, base: &str) -> String {
    let mut names: Vec<&str> = Vec::new();
    names.extend(
        module
            .functions
            .iter()
            .filter_map(|(_, f)| f.name.as_deref()),
    );
    names.extend(
        module
            .global_variables
            .iter()
            .filter_map(|(_, g)| g.name.as_deref()),
    );
    names.extend(module.types.iter().filter_map(|(_, t)| t.name.as_deref()));
    names.extend(
        module
            .constants
            .iter()
            .filter_map(|(_, c)| c.name.as_deref()),
    );
    names.extend(
        module
            .overrides
            .iter()
            .filter_map(|(_, o)| o.name.as_deref()),
    );
    for ep in &module.entry_points {
        names.push(&ep.name);
        names.extend(ep.function.name.as_deref());
    }
    unique_prefix_among(names.into_iter(), base)
}

fn unique_prefix_among<'a>(names: impl Iterator<Item = &'a str> + Clone, base: &str) -> String {
    let stem = base.trim_end_matches('_');
    let mut candidate = base.to_string();
    let mut counter = 1u32;
    while names.clone().any(|name| name.starts_with(&candidate)) {
        candidate = format!("{stem}{counter}_");
        counter += 1;
    }
    candidate
}

/// Return a prefix derived from `base` that no identifier in `existing_wgsl` starts with, so
/// generated helpers can be appended to that source without collisions.
pub fn unique_wgsl_prefix(existing_wgsl: &str, base: &str) -> String {
    let identifiers = existing_wgsl.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
    unique_prefix_among(identifiers.filter(|t| !t.is_empty()), base)
}

// ---------------------------------------------------------------------------
// naga IR lowering: materialization
// ---------------------------------------------------------------------------

struct LoweringGlobals {
    prefix: String,
    u32_ty: Handle<Type>,
    i32_ty: Handle<Type>,
    bool_ty: Handle<Type>,
    vec4_u32_ty: Handle<Type>,
    local_index: Option<Handle<GlobalVariable>>,
    workgroup_invocations: Option<Handle<GlobalVariable>>,
    scratch: BTreeMap<ScratchElement, Handle<GlobalVariable>>,
}

fn insert_type(module: &mut Module, inner: TypeInner) -> Handle<Type> {
    module
        .types
        .insert(Type { name: None, inner }, Span::UNDEFINED)
}

fn append_global(
    module: &mut Module,
    name: String,
    space: AddressSpace,
    ty: Handle<Type>,
) -> Handle<GlobalVariable> {
    module.global_variables.append(
        GlobalVariable {
            name: Some(name),
            space,
            binding: None,
            ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    )
}

fn materialize(
    module: &mut Module,
    plan: Plan,
    options: &WaveLoweringOptions,
) -> WaveLoweringReport {
    let Plan {
        sites,
        builtin_args,
        prologue_entries,
        helpers,
        mut report,
    } = plan;

    let prefix = unique_ir_prefix(module, &options.helper_prefix);
    if prefix != options.helper_prefix {
        // Names recorded during planning used the requested prefix; re-derive them.
        for op in &mut report.lowered {
            if let Some(name) = &mut op.helper
                && let Some(rest) = name.strip_prefix(&options.helper_prefix)
            {
                *name = format!("{prefix}{rest}");
            }
        }
    }

    let u32_ty = insert_type(module, TypeInner::Scalar(Scalar::U32));
    let i32_ty = insert_type(module, TypeInner::Scalar(Scalar::I32));
    let bool_ty = insert_type(module, TypeInner::Scalar(Scalar::BOOL));
    let vec4_u32_ty = insert_type(
        module,
        TypeInner::Vector {
            size: VectorSize::Quad,
            scalar: Scalar::U32,
        },
    );

    let lanes = scratch_lanes(module, &prologue_entries);
    let (local_index, workgroup_invocations_g) = if prologue_entries.is_empty() {
        (None, None)
    } else {
        (
            Some(append_global(
                module,
                format!("{prefix}local_invocation_index"),
                AddressSpace::Private,
                u32_ty,
            )),
            Some(append_global(
                module,
                format!("{prefix}workgroup_invocations"),
                AddressSpace::Private,
                u32_ty,
            )),
        )
    };

    // One `var<workgroup>` array per storage element type; `bool` values and ballot bits share
    // the `u32` array.
    let scratch_elements: BTreeSet<ScratchElement> = helpers
        .iter()
        .filter_map(|key| key.scratch_element())
        .collect();
    let mut arrays: BTreeMap<Handle<Type>, Handle<GlobalVariable>> = BTreeMap::new();
    let mut scratch = BTreeMap::new();
    for element in scratch_elements {
        let (base, stride) = match element {
            ScratchElement::Of(ty) if !is_bool_type(&module.types, ty) => {
                (ty, scratch_stride(&module.types, ty))
            }
            _ => (u32_ty, 4),
        };
        let global = match arrays.get(&base) {
            Some(&global) => global,
            None => {
                let array_ty = insert_type(
                    module,
                    TypeInner::Array {
                        base,
                        size: ArraySize::Constant(
                            std::num::NonZeroU32::new(lanes).unwrap_or(std::num::NonZeroU32::MIN),
                        ),
                        stride,
                    },
                );
                let name = format!("{prefix}scratch_{}", type_tag(&module.types, base));
                let global = append_global(module, name, AddressSpace::WorkGroup, array_ty);
                arrays.insert(base, global);
                global
            }
        };
        scratch.insert(element, global);
    }

    let globals = LoweringGlobals {
        prefix: prefix.clone(),
        u32_ty,
        i32_ty,
        bool_ty,
        vec4_u32_ty,
        local_index,
        workgroup_invocations: workgroup_invocations_g,
        scratch,
    };

    // Build helpers, then rebuild the function arena with helpers first (naga requires callees
    // to precede callers).
    let mut built: Vec<(HelperKey, Function)> = Vec::new();
    for &key in &helpers {
        built.push((key, build_helper(key, &globals, &module.types)));
    }
    let old_functions = std::mem::take(&mut module.functions);
    let mut new_functions = Arena::new();
    let mut helper_handles: HashMap<HelperKey, Handle<Function>> = HashMap::new();
    for (key, function) in built {
        report
            .helper_functions
            .push(function.name.clone().unwrap_or_default());
        helper_handles.insert(key, new_functions.append(function, Span::UNDEFINED));
    }
    let mut remap: Vec<Handle<Function>> = Vec::with_capacity(old_functions.len());
    let mut old_handles: Vec<Handle<Function>> = Vec::with_capacity(old_functions.len());
    let mut old_functions = old_functions;
    for (old_handle, function, span) in old_functions.drain() {
        old_handles.push(old_handle);
        remap.push(new_functions.append(function, span));
    }
    module.functions = new_functions;
    if let Some(doc_comments) = module.doc_comments.as_mut() {
        let functions = std::mem::take(&mut doc_comments.functions);
        doc_comments.functions = functions
            .into_iter()
            .map(|(handle, docs)| (remap[handle.index()], docs))
            .collect();
    }

    let rewriter = Rewriter {
        sites: &sites,
        helper_handles: &helper_handles,
        remap: &remap,
    };
    for (old_handle, &new_handle) in old_handles.iter().zip(remap.iter()) {
        let function = &mut module.functions[new_handle];
        rewriter.rewrite_function(function, FnRef::Function(*old_handle));
    }
    for index in 0..module.entry_points.len() {
        let mut function = std::mem::take(&mut module.entry_points[index].function);
        rewriter.rewrite_function(&mut function, FnRef::Entry(index));
        let actions = builtin_args.get(&index).map(Vec::as_slice).unwrap_or(&[]);
        let needs_prologue = prologue_entries.contains(&index);
        let total = workgroup_invocations(module.entry_points[index].workgroup_size);
        rewrite_entry_arguments(
            &mut function,
            &module.types,
            actions,
            needs_prologue,
            total,
            &globals,
            &helper_handles,
        );
        module.entry_points[index].function = function;
    }

    report
}

struct Rewriter<'a> {
    sites: &'a HashMap<(FnRef, Handle<Expression>), SiteAction>,
    helper_handles: &'a HashMap<HelperKey, Handle<Function>>,
    remap: &'a [Handle<Function>],
}

impl Rewriter<'_> {
    fn rewrite_function(&self, function: &mut Function, fn_ref: FnRef) {
        for (_, expr) in function.expressions.iter_mut() {
            if let Expression::CallResult(callee) = expr {
                *callee = self.remap[callee.index()];
            }
        }
        let mut body = std::mem::take(&mut function.body);
        self.rewrite_block(&mut body, &mut function.expressions, fn_ref);
        function.body = body;
    }

    fn rewrite_block(&self, block: &mut Block, expressions: &mut Arena<Expression>, fn_ref: FnRef) {
        for statement in block.iter_mut() {
            let result = match statement {
                Statement::Call { function, .. } => {
                    *function = self.remap[function.index()];
                    None
                }
                Statement::Block(inner) => {
                    self.rewrite_block(inner, expressions, fn_ref);
                    None
                }
                Statement::If { accept, reject, .. } => {
                    self.rewrite_block(accept, expressions, fn_ref);
                    self.rewrite_block(reject, expressions, fn_ref);
                    None
                }
                Statement::Switch { cases, .. } => {
                    for case in cases.iter_mut() {
                        self.rewrite_block(&mut case.body, expressions, fn_ref);
                    }
                    None
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    self.rewrite_block(body, expressions, fn_ref);
                    self.rewrite_block(continuing, expressions, fn_ref);
                    None
                }
                Statement::SubgroupBallot { result, .. }
                | Statement::SubgroupCollectiveOperation { result, .. }
                | Statement::SubgroupGather { result, .. } => Some(*result),
                _ => None,
            };
            let Some(result) = result else {
                continue;
            };
            let Some(SiteAction::Call { helper, args }) = self.sites.get(&(fn_ref, result)) else {
                continue;
            };
            let helper_handle = self.helper_handles[helper];
            let arguments = args
                .iter()
                .map(|arg| match *arg {
                    ArgSource::Expr(handle) => handle,
                    ArgSource::True => expressions
                        .append(Expression::Literal(Literal::Bool(true)), Span::UNDEFINED),
                })
                .collect();
            expressions[result] = Expression::CallResult(helper_handle);
            *statement = Statement::Call {
                function: helper_handle,
                arguments,
                result: Some(result),
            };
        }
    }
}

fn rewrite_entry_arguments(
    function: &mut Function,
    types: &UniqueArena<Type>,
    actions: &[(usize, BuiltinAction)],
    needs_prologue: bool,
    workgroup_total: u32,
    globals: &LoweringGlobals,
    helper_handles: &HashMap<HelperKey, Handle<Function>>,
) {
    let mut lane_calls: Vec<(Handle<Function>, Handle<Expression>)> = Vec::new();
    let mut removed: Vec<usize> = Vec::new();
    for (arg_index, action) in actions {
        let replacement = match action {
            BuiltinAction::Keep => continue,
            BuiltinAction::Literal(value) => Expression::Literal(Literal::U32(*value)),
            BuiltinAction::Helper(key) => Expression::CallResult(helper_handles[key]),
        };
        let uses: Vec<Handle<Expression>> = function
            .expressions
            .iter()
            .filter(
                |(_, e)| matches!(e, Expression::FunctionArgument(i) if *i as usize == *arg_index),
            )
            .map(|(h, _)| h)
            .collect();
        for handle in uses {
            if let (BuiltinAction::Helper(key), Expression::CallResult(_)) = (action, &replacement)
            {
                lane_calls.push((helper_handles[key], handle));
            }
            function.expressions[handle] = replacement.clone();
        }
        removed.push(*arg_index);
    }
    removed.sort_unstable();
    for &arg_index in removed.iter().rev() {
        function.arguments.remove(arg_index);
        for (_, expr) in function.expressions.iter_mut() {
            if let Expression::FunctionArgument(i) = expr
                && *i as usize > arg_index
            {
                *i -= 1;
            }
        }
    }

    let mut prologue = Block::new();
    if needs_prologue
        && let (Some(local_index_g), Some(total_g)) =
            (globals.local_index, globals.workgroup_invocations)
    {
        let local_index = local_invocation_index_expr(function, types, globals, &mut prologue);
        let pointer = function
            .expressions
            .append(Expression::GlobalVariable(local_index_g), Span::UNDEFINED);
        prologue.push(
            Statement::Store {
                pointer,
                value: local_index,
            },
            Span::UNDEFINED,
        );
        let pointer = function
            .expressions
            .append(Expression::GlobalVariable(total_g), Span::UNDEFINED);
        let value = function.expressions.append(
            Expression::Literal(Literal::U32(workgroup_total)),
            Span::UNDEFINED,
        );
        prologue.push(Statement::Store { pointer, value }, Span::UNDEFINED);
    }
    for (helper, result) in lane_calls {
        prologue.push(
            Statement::Call {
                function: helper,
                arguments: Vec::new(),
                result: Some(result),
            },
            Span::UNDEFINED,
        );
    }
    if !prologue.is_empty() {
        function.body.splice(0..0, prologue);
    }
}

/// Expression yielding `local_invocation_index` in an entry point, adding the argument if absent.
fn local_invocation_index_expr(
    function: &mut Function,
    types: &UniqueArena<Type>,
    globals: &LoweringGlobals,
    prologue: &mut Block,
) -> Handle<Expression> {
    let is_local_index =
        |b: &Option<Binding>| matches!(b, Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)));
    for (arg_index, arg) in function.arguments.iter().enumerate() {
        if is_local_index(&arg.binding) {
            return function.expressions.append(
                Expression::FunctionArgument(arg_index as u32),
                Span::UNDEFINED,
            );
        }
        if let TypeInner::Struct { members, .. } = &types[arg.ty].inner
            && let Some(member) = members.iter().position(|m| is_local_index(&m.binding))
        {
            let base = function.expressions.append(
                Expression::FunctionArgument(arg_index as u32),
                Span::UNDEFINED,
            );
            let access = function.expressions.append(
                Expression::AccessIndex {
                    base,
                    index: member as u32,
                },
                Span::UNDEFINED,
            );
            prologue.push(
                Statement::Emit(Range::new_from_bounds(access, access)),
                Span::UNDEFINED,
            );
            return access;
        }
    }
    let arg_index = function.arguments.len() as u32;
    function.arguments.push(FunctionArgument {
        name: Some(format!("{}local_invocation_index_in", globals.prefix)),
        ty: globals.u32_ty,
        binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
    });
    function
        .expressions
        .append(Expression::FunctionArgument(arg_index), Span::UNDEFINED)
}

// ---------------------------------------------------------------------------
// naga IR helper construction
// ---------------------------------------------------------------------------

struct FnBuilder {
    function: Function,
    blocks: Vec<Block>,
    u32_ty: Handle<Type>,
}

impl FnBuilder {
    fn new(name: String, result: Handle<Type>, u32_ty: Handle<Type>) -> Self {
        Self {
            function: Function {
                name: Some(name),
                result: Some(FunctionResult {
                    ty: result,
                    binding: None,
                }),
                ..Function::default()
            },
            blocks: vec![Block::new()],
            u32_ty,
        }
    }

    fn arg(&mut self, name: &str, ty: Handle<Type>) -> Handle<Expression> {
        let index = self.function.arguments.len() as u32;
        self.function.arguments.push(FunctionArgument {
            name: Some(name.to_string()),
            ty,
            binding: None,
        });
        self.expr(Expression::FunctionArgument(index))
    }

    fn push(&mut self, statement: Statement) {
        if let Some(block) = self.blocks.last_mut() {
            block.push(statement, Span::UNDEFINED);
        }
    }

    fn expr(&mut self, expression: Expression) -> Handle<Expression> {
        let pre_emitted = expression.needs_pre_emit();
        let handle = self
            .function
            .expressions
            .append(expression, Span::UNDEFINED);
        if !pre_emitted {
            self.push(Statement::Emit(Range::new_from_bounds(handle, handle)));
        }
        handle
    }

    fn named(&mut self, name: &str, handle: Handle<Expression>) -> Handle<Expression> {
        self.function
            .named_expressions
            .insert(handle, name.to_string());
        handle
    }

    fn lit(&mut self, literal: Literal) -> Handle<Expression> {
        self.expr(Expression::Literal(literal))
    }

    fn u32(&mut self, value: u32) -> Handle<Expression> {
        self.lit(Literal::U32(value))
    }

    fn binary(
        &mut self,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(Expression::Binary { op, left, right })
    }

    fn math2(
        &mut self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(Expression::Math {
            fun,
            arg,
            arg1: Some(arg1),
            arg2: None,
            arg3: None,
        })
    }

    fn select(
        &mut self,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        self.expr(Expression::Select {
            condition,
            accept,
            reject,
        })
    }

    fn global(&mut self, global: Handle<GlobalVariable>) -> Handle<Expression> {
        self.expr(Expression::GlobalVariable(global))
    }

    fn local(&mut self, name: &str, ty: Handle<Type>) -> Handle<Expression> {
        let local = self.function.local_variables.append(
            LocalVariable {
                name: Some(name.to_string()),
                ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        self.expr(Expression::LocalVariable(local))
    }

    fn load(&mut self, pointer: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::Load { pointer })
    }

    fn store(&mut self, pointer: Handle<Expression>, value: Handle<Expression>) {
        self.push(Statement::Store { pointer, value });
    }

    fn index(&mut self, base: Handle<Expression>, index: Handle<Expression>) -> Handle<Expression> {
        self.expr(Expression::Access { base, index })
    }

    fn barrier(&mut self) {
        self.push(Statement::ControlBarrier(Barrier::WORK_GROUP));
    }

    fn open(&mut self) {
        self.blocks.push(Block::new());
    }

    fn close(&mut self) -> Block {
        self.blocks.pop().unwrap_or_default()
    }

    fn if_then(&mut self, condition: Handle<Expression>, accept: Block) {
        self.push(Statement::If {
            condition,
            accept,
            reject: Block::new(),
        });
    }

    fn ret(&mut self, value: Handle<Expression>) {
        self.push(Statement::Return { value: Some(value) });
    }

    /// `for (var j = start; j < 64u; j++) { body(j) }` â€” a uniform, constant trip count.
    fn lane_loop(&mut self, start: u32, body: impl FnOnce(&mut Self, Handle<Expression>)) {
        let u32_ty = self.u32_ty;
        let counter = self.local("j", u32_ty);
        let first = self.u32(start);
        self.store(counter, first);
        self.open();
        let j = self.load(counter);
        body(self, j);
        let body_block = self.close();
        self.open();
        let current = self.load(counter);
        let one = self.u32(1);
        let next = self.binary(BinaryOperator::Add, current, one);
        self.store(counter, next);
        let limit = self.u32(WAVE64_LANE_COUNT);
        let done = self.binary(BinaryOperator::GreaterEqual, next, limit);
        let continuing = self.close();
        self.push(Statement::Loop {
            body: body_block,
            continuing,
            break_if: Some(done),
        });
    }

    fn finish(mut self) -> Function {
        self.function.body = self.blocks.pop().unwrap_or_default();
        self.function
    }
}

/// Values every emulation helper derives from the private lane globals.
struct WaveCoords {
    local_index: Handle<Expression>,
    count: Handle<Expression>,
    base: Handle<Expression>,
    lane: Handle<Expression>,
}

fn wave_coords(b: &mut FnBuilder, globals: &LoweringGlobals) -> WaveCoords {
    // Emulation helpers are only planned together with a prologue entry point, which is what
    // creates these globals.
    let local_index_g = globals
        .local_index
        .expect("emulation helpers imply the local_invocation_index global");
    let total_g = globals
        .workgroup_invocations
        .expect("emulation helpers imply the workgroup_invocations global");
    let pointer = b.global(local_index_g);
    let local_index = b.load(pointer);
    let local_index = b.named("local_index", local_index);
    let pointer = b.global(total_g);
    let count = b.load(pointer);
    let count = b.named("invocations", count);
    let wave_mask = b.u32(!(WAVE64_LANE_COUNT - 1));
    let base = b.binary(BinaryOperator::And, local_index, wave_mask);
    let base = b.named("wave_base", base);
    let lane_mask = b.u32(WAVE64_LANE_COUNT - 1);
    let lane = b.binary(BinaryOperator::And, local_index, lane_mask);
    let lane = b.named("lane", lane);
    WaveCoords {
        local_index,
        count,
        base,
        lane,
    }
}

fn is_bool_type(types: &UniqueArena<Type>, ty: Handle<Type>) -> bool {
    matches!(
        types[ty].inner,
        TypeInner::Scalar(Scalar {
            kind: ScalarKind::Bool,
            ..
        })
    )
}

/// Encode a value for the scratch array (`bool` -> `u32`).
fn encode(b: &mut FnBuilder, value: Handle<Expression>, is_bool: bool) -> Handle<Expression> {
    if is_bool {
        let one = b.u32(1);
        let zero = b.u32(0);
        b.select(value, one, zero)
    } else {
        value
    }
}

fn decode(b: &mut FnBuilder, value: Handle<Expression>, is_bool: bool) -> Handle<Expression> {
    if is_bool {
        let zero = b.u32(0);
        b.binary(BinaryOperator::NotEqual, value, zero)
    } else {
        value
    }
}

/// Write the caller's value into scratch between two barriers; returns the scratch pointer.
fn publish(
    b: &mut FnBuilder,
    coords: &WaveCoords,
    scratch: Handle<GlobalVariable>,
    value: Handle<Expression>,
) -> Handle<Expression> {
    b.barrier();
    let array = b.global(scratch);
    let slot = b.index(array, coords.local_index);
    b.store(slot, value);
    b.barrier();
    array
}

fn combine(
    b: &mut FnBuilder,
    op: SubgroupOperation,
    is_bool: bool,
    left: Handle<Expression>,
    right: Handle<Expression>,
) -> Handle<Expression> {
    match (op, is_bool) {
        (SubgroupOperation::Add, _) => b.binary(BinaryOperator::Add, left, right),
        (SubgroupOperation::Mul, _) => b.binary(BinaryOperator::Multiply, left, right),
        (SubgroupOperation::Min, _) => b.math2(MathFunction::Min, left, right),
        (SubgroupOperation::Max, _) => b.math2(MathFunction::Max, left, right),
        (SubgroupOperation::All | SubgroupOperation::And, true) => {
            b.binary(BinaryOperator::LogicalAnd, left, right)
        }
        (SubgroupOperation::Any | SubgroupOperation::Or, true) => {
            b.binary(BinaryOperator::LogicalOr, left, right)
        }
        (SubgroupOperation::Xor, true) => b.binary(BinaryOperator::NotEqual, left, right),
        (SubgroupOperation::And | SubgroupOperation::All, false) => {
            b.binary(BinaryOperator::And, left, right)
        }
        (SubgroupOperation::Or | SubgroupOperation::Any, false) => {
            b.binary(BinaryOperator::InclusiveOr, left, right)
        }
        (SubgroupOperation::Xor, false) => b.binary(BinaryOperator::ExclusiveOr, left, right),
    }
}

/// Identity element of an exclusive scan: `1` for mul, zero otherwise (add; and the value
/// returned by never-executed unreachable code).
fn identity_value(
    b: &mut FnBuilder,
    types: &UniqueArena<Type>,
    op: SubgroupOperation,
    ty: Handle<Type>,
) -> Handle<Expression> {
    let one = match value_shape(types, ty) {
        Some(shape) if op == SubgroupOperation::Mul => match shape.scalar.kind {
            ScalarKind::Float => Some((Literal::F32(1.0), shape.size)),
            ScalarKind::Sint => Some((Literal::I32(1), shape.size)),
            ScalarKind::Uint => Some((Literal::U32(1), shape.size)),
            _ => None,
        },
        _ => None,
    };
    match one {
        Some((literal, None)) => b.lit(literal),
        Some((literal, Some(size))) => {
            let value = b.lit(literal);
            b.expr(Expression::Splat { size, value })
        }
        None => b.expr(Expression::ZeroValue(ty)),
    }
}

fn build_helper(key: HelperKey, globals: &LoweringGlobals, types: &UniqueArena<Type>) -> Function {
    let name = helper_name(&globals.prefix, key, types);
    let u32_ty = globals.u32_ty;
    match key {
        HelperKey::Identity { ty } => {
            let mut b = FnBuilder::new(name, ty, u32_ty);
            let value = b.arg("value", ty);
            b.ret(value);
            b.finish()
        }
        HelperKey::Wave1Exclusive { op, ty } => {
            let mut b = FnBuilder::new(name, ty, u32_ty);
            let _value = b.arg("value", ty);
            let identity = identity_value(&mut b, types, op, ty);
            b.ret(identity);
            b.finish()
        }
        HelperKey::Wave1Ballot => {
            let mut b = FnBuilder::new(name, globals.vec4_u32_ty, u32_ty);
            let predicate = b.arg("predicate", globals.bool_ty);
            let bit = encode(&mut b, predicate, true);
            let zero = b.u32(0);
            let mask = b.expr(Expression::Compose {
                ty: globals.vec4_u32_ty,
                components: vec![bit, zero, zero, zero],
            });
            b.ret(mask);
            b.finish()
        }
        HelperKey::EmuLaneIndex | HelperKey::EmuWaveIndex => {
            let mut b = FnBuilder::new(name, u32_ty, u32_ty);
            let coords = wave_coords(&mut b, globals);
            let value = if key == HelperKey::EmuLaneIndex {
                coords.lane
            } else {
                let shift = b.u32(6);
                b.binary(BinaryOperator::ShiftRight, coords.local_index, shift)
            };
            b.ret(value);
            b.finish()
        }
        HelperKey::EmuBallot => build_emulated_ballot(name, globals),
        HelperKey::EmuCollective { op, collective, ty } => {
            build_emulated_collective(name, globals, types, op, collective, ty)
        }
        HelperKey::EmuGather {
            kind,
            ty,
            signed_index,
        } => build_emulated_gather(name, globals, types, kind, ty, signed_index),
    }
}

fn build_emulated_ballot(name: String, globals: &LoweringGlobals) -> Function {
    let u32_ty = globals.u32_ty;
    let mut b = FnBuilder::new(name, globals.vec4_u32_ty, u32_ty);
    let predicate = b.arg("predicate", globals.bool_ty);
    let coords = wave_coords(&mut b, globals);
    let encoded = encode(&mut b, predicate, true);
    let scratch = globals.scratch[&ScratchElement::U32];
    let array = publish(&mut b, &coords, scratch, encoded);
    let low = b.local("low", u32_ty);
    let high = b.local("high", u32_ty);
    let zero = b.u32(0);
    b.store(low, zero);
    b.store(high, zero);
    b.lane_loop(0, |b, j| {
        let slot_index = b.binary(BinaryOperator::Add, coords.base, j);
        let in_range = b.binary(BinaryOperator::Less, slot_index, coords.count);
        b.open();
        let slot = b.index(array, slot_index);
        let value = b.load(slot);
        let zero = b.u32(0);
        let set = b.binary(BinaryOperator::NotEqual, value, zero);
        b.open();
        let mask31 = b.u32(31);
        let shift = b.binary(BinaryOperator::And, j, mask31);
        let one = b.u32(1);
        let bit = b.binary(BinaryOperator::ShiftLeft, one, shift);
        let thirty_two = b.u32(32);
        let is_low = b.binary(BinaryOperator::Less, j, thirty_two);
        let zero = b.u32(0);
        let low_bit = b.select(is_low, bit, zero);
        let high_bit = b.select(is_low, zero, bit);
        let current = b.load(low);
        let updated = b.binary(BinaryOperator::InclusiveOr, current, low_bit);
        b.store(low, updated);
        let current = b.load(high);
        let updated = b.binary(BinaryOperator::InclusiveOr, current, high_bit);
        b.store(high, updated);
        let set_block = b.close();
        b.if_then(set, set_block);
        let in_range_block = b.close();
        b.if_then(in_range, in_range_block);
    });
    let low_value = b.load(low);
    let high_value = b.load(high);
    let zero = b.u32(0);
    let mask = b.expr(Expression::Compose {
        ty: globals.vec4_u32_ty,
        components: vec![low_value, high_value, zero, zero],
    });
    b.ret(mask);
    b.finish()
}

fn build_emulated_collective(
    name: String,
    globals: &LoweringGlobals,
    types: &UniqueArena<Type>,
    op: SubgroupOperation,
    collective: CollectiveOperation,
    ty: Handle<Type>,
) -> Function {
    let is_bool = is_bool_type(types, ty);
    let mut b = FnBuilder::new(name, ty, globals.u32_ty);
    let value = b.arg("value", ty);
    let coords = wave_coords(&mut b, globals);
    let encoded = encode(&mut b, value, is_bool);
    let scratch = globals.scratch[&ScratchElement::Of(ty)];
    let array = publish(&mut b, &coords, scratch, encoded);

    // Every lane combines in ascending lane order starting at lane 0, so all lanes of a wave
    // compute bit-identical reductions.
    let acc = b.local("acc", ty);
    let first_slot = b.index(array, coords.base);
    let first = b.load(first_slot);
    let first = decode(&mut b, first, is_bool);
    b.store(acc, first);
    b.lane_loop(1, |b, j| {
        let slot_index = b.binary(BinaryOperator::Add, coords.base, j);
        let include = match collective {
            CollectiveOperation::Reduce => b.binary(BinaryOperator::Less, slot_index, coords.count),
            CollectiveOperation::InclusiveScan => {
                b.binary(BinaryOperator::LessEqual, j, coords.lane)
            }
            CollectiveOperation::ExclusiveScan => b.binary(BinaryOperator::Less, j, coords.lane),
        };
        b.open();
        let slot = b.index(array, slot_index);
        let lane_value = b.load(slot);
        let lane_value = decode(b, lane_value, is_bool);
        let current = b.load(acc);
        let combined = combine(b, op, is_bool, current, lane_value);
        b.store(acc, combined);
        let block = b.close();
        b.if_then(include, block);
    });
    let result = b.load(acc);
    if collective == CollectiveOperation::ExclusiveScan {
        let identity = identity_value(&mut b, types, op, ty);
        let zero = b.u32(0);
        let first_lane = b.binary(BinaryOperator::Equal, coords.lane, zero);
        let out = b.select(first_lane, identity, result);
        b.ret(out);
    } else {
        b.ret(result);
    }
    b.finish()
}

fn build_emulated_gather(
    name: String,
    globals: &LoweringGlobals,
    types: &UniqueArena<Type>,
    kind: GatherKind,
    ty: Handle<Type>,
    signed_index: bool,
) -> Function {
    let is_bool = is_bool_type(types, ty);
    let mut b = FnBuilder::new(name, ty, globals.u32_ty);
    let value = b.arg("value", ty);
    let index = if kind == GatherKind::BroadcastFirst {
        None
    } else {
        let index_ty = if signed_index {
            globals.i32_ty
        } else {
            globals.u32_ty
        };
        let raw = b.arg("lane_operand", index_ty);
        Some(if signed_index {
            b.expr(Expression::As {
                expr: raw,
                kind: ScalarKind::Uint,
                convert: Some(4),
            })
        } else {
            raw
        })
    };
    let coords = wave_coords(&mut b, globals);
    let encoded = encode(&mut b, value, is_bool);
    let scratch = globals.scratch[&ScratchElement::Of(ty)];
    let array = publish(&mut b, &coords, scratch, encoded);

    let lanes = b.u32(WAVE64_LANE_COUNT);
    let (source, valid) = match (kind, index) {
        (GatherKind::Broadcast | GatherKind::Shuffle, Some(i)) => {
            let valid = b.binary(BinaryOperator::Less, i, lanes);
            (i, Some(valid))
        }
        (GatherKind::ShuffleXor, Some(i)) => {
            let source = b.binary(BinaryOperator::ExclusiveOr, coords.lane, i);
            let valid = b.binary(BinaryOperator::Less, source, lanes);
            (source, Some(valid))
        }
        (GatherKind::ShuffleDown, Some(i)) => {
            let room = b.binary(BinaryOperator::Subtract, lanes, coords.lane);
            let valid = b.binary(BinaryOperator::Less, i, room);
            let source = b.binary(BinaryOperator::Add, coords.lane, i);
            (source, Some(valid))
        }
        (GatherKind::ShuffleUp, Some(i)) => {
            let valid = b.binary(BinaryOperator::LessEqual, i, coords.lane);
            let source = b.binary(BinaryOperator::Subtract, coords.lane, i);
            (source, Some(valid))
        }
        // BroadcastFirst: under uniform control flow every lane is active, so lane 0 is first.
        _ => (b.u32(0), None),
    };
    let lane_mask = b.u32(WAVE64_LANE_COUNT - 1);
    let wrapped = b.binary(BinaryOperator::And, source, lane_mask);
    let slot_index = b.binary(BinaryOperator::Add, coords.base, wrapped);
    let one = b.u32(1);
    let last = b.binary(BinaryOperator::Subtract, coords.count, one);
    let clamped = b.math2(MathFunction::Min, slot_index, last);
    let slot = b.index(array, clamped);
    let loaded = b.load(slot);
    let loaded = decode(&mut b, loaded, is_bool);
    let exists = b.binary(BinaryOperator::Less, slot_index, coords.count);
    let valid = match valid {
        Some(valid) => b.binary(BinaryOperator::LogicalAnd, valid, exists),
        None => exists,
    };
    // Out-of-range or nonexistent source lanes (undefined in SPIR-V) yield the caller's value.
    let out = b.select(valid, loaded, value);
    b.ret(out);
    b.finish()
}

// ---------------------------------------------------------------------------
// 64-bit wave mask helpers (WGSL source)
// ---------------------------------------------------------------------------

/// A `vec2<u32>` (`x` = low word, `y` = high word) 64-bit helper emitted by [`u64_helpers_wgsl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum U64Helper {
    And,
    Or,
    Xor,
    Not,
    /// Shift left by any `u32` amount; amounts `>= 64` yield zero.
    Shl,
    /// Logical shift right by any `u32` amount; amounts `>= 64` yield zero.
    Shr,
    CountOneBits,
    /// Index of the lowest set bit, `0xffffffff` for zero.
    FirstTrailingBit,
    /// Index of the highest set bit, `0xffffffff` for zero.
    FirstLeadingBit,
    Equal,
    /// Unsigned `a < b`.
    LessThan,
    /// `a + b` modulo 2^64 (carry from the low into the high word).
    Add,
    /// `(1 << n) - 1`; `n >= 64` yields all ones.
    LaneMask,
    /// Bit `i` set; `i >= 64` yields `false`.
    TestBit,
}

impl U64Helper {
    pub const ALL: [Self; 14] = [
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Not,
        Self::Shl,
        Self::Shr,
        Self::CountOneBits,
        Self::FirstTrailingBit,
        Self::FirstLeadingBit,
        Self::Equal,
        Self::LessThan,
        Self::Add,
        Self::LaneMask,
        Self::TestBit,
    ];

    /// Function name without the prefix.
    pub const fn function_suffix(self) -> &'static str {
        match self {
            Self::And => "u64_and",
            Self::Or => "u64_or",
            Self::Xor => "u64_xor",
            Self::Not => "u64_not",
            Self::Shl => "u64_shl",
            Self::Shr => "u64_shr",
            Self::CountOneBits => "u64_count_one_bits",
            Self::FirstTrailingBit => "u64_first_trailing_bit",
            Self::FirstLeadingBit => "u64_first_leading_bit",
            Self::Equal => "u64_equal",
            Self::LessThan => "u64_less_than",
            Self::Add => "u64_add",
            Self::LaneMask => "u64_lane_mask",
            Self::TestBit => "u64_test_bit",
        }
    }

    pub fn function_name(self, prefix: &str) -> String {
        format!("{prefix}{}", self.function_suffix())
    }

    fn source(self, prefix: &str) -> String {
        let name = self.function_name(prefix);
        let body = match self {
            Self::And => "(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {\n    return a & b;\n}",
            Self::Or => "(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {\n    return a | b;\n}",
            Self::Xor => "(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {\n    return a ^ b;\n}",
            Self::Not => "(a: vec2<u32>) -> vec2<u32> {\n    return ~a;\n}",
            Self::Shl => {
                "(a: vec2<u32>, n: u32) -> vec2<u32> {\n    let s = n & 31u;\n    let carry = (a.x >> 1u) >> (31u - s);\n    let small = vec2<u32>(a.x << s, (a.y << s) | carry);\n    let large = vec2<u32>(0u, a.x << s);\n    return select(select(small, large, n >= 32u), vec2<u32>(0u, 0u), n >= 64u);\n}"
            }
            Self::Shr => {
                "(a: vec2<u32>, n: u32) -> vec2<u32> {\n    let s = n & 31u;\n    let carry = (a.y << 1u) << (31u - s);\n    let small = vec2<u32>((a.x >> s) | carry, a.y >> s);\n    let large = vec2<u32>(a.y >> s, 0u);\n    return select(select(small, large, n >= 32u), vec2<u32>(0u, 0u), n >= 64u);\n}"
            }
            Self::CountOneBits => {
                "(a: vec2<u32>) -> u32 {\n    return countOneBits(a.x) + countOneBits(a.y);\n}"
            }
            Self::FirstTrailingBit => {
                "(a: vec2<u32>) -> u32 {\n    let high = select(0xffffffffu, 32u + firstTrailingBit(a.y), a.y != 0u);\n    return select(high, firstTrailingBit(a.x), a.x != 0u);\n}"
            }
            Self::FirstLeadingBit => {
                "(a: vec2<u32>) -> u32 {\n    let low = select(0xffffffffu, firstLeadingBit(a.x), a.x != 0u);\n    return select(low, 32u + firstLeadingBit(a.y), a.y != 0u);\n}"
            }
            Self::Equal => "(a: vec2<u32>, b: vec2<u32>) -> bool {\n    return all(a == b);\n}",
            Self::LessThan => {
                "(a: vec2<u32>, b: vec2<u32>) -> bool {\n    return (a.y < b.y) || ((a.y == b.y) && (a.x < b.x));\n}"
            }
            Self::Add => {
                "(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {\n    let low = a.x + b.x;\n    let carry = select(0u, 1u, low < a.x);\n    return vec2<u32>(low, a.y + b.y + carry);\n}"
            }
            Self::LaneMask => {
                "(n: u32) -> vec2<u32> {\n    let partial = (1u << (n & 31u)) - 1u;\n    let low = select(partial, 0xffffffffu, n >= 32u);\n    let high = select(select(0u, partial, n >= 32u), 0xffffffffu, n >= 64u);\n    return vec2<u32>(low, high);\n}"
            }
            Self::TestBit => {
                "(a: vec2<u32>, bit: u32) -> bool {\n    let word = select(a.y, a.x, bit < 32u);\n    return (bit < 64u) && (((word >> (bit & 31u)) & 1u) != 0u);\n}"
            }
        };
        format!("fn {name}{body}\n")
    }
}

/// WGSL source of the requested 64-bit mask helpers (deduplicated, canonical order), each named
/// `{prefix}u64_*`. Helpers are independent, so any subset is self-contained.
pub fn u64_helpers_wgsl(prefix: &str, helpers: &[U64Helper]) -> String {
    let selected: BTreeSet<U64Helper> = helpers.iter().copied().collect();
    let mut out = String::new();
    for (index, helper) in selected.into_iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        out.push_str(&helper.source(prefix));
    }
    out
}

// ---------------------------------------------------------------------------
// Mesh-compute lowering helpers
// ---------------------------------------------------------------------------

/// D3D12 mesh shader limit on `max_vertices`.
pub const D3D_MAX_MESH_OUTPUT_VERTICES: u32 = 256;
/// D3D12 mesh shader limit on `max_primitives`.
pub const D3D_MAX_MESH_OUTPUT_PRIMITIVES: u32 = 256;
/// D3D12 limit on mesh/amplification threads per group.
pub const D3D_MAX_MESH_THREADS: u32 = 128;

// Indirect-argument layouts follow the WebGPU spec and match `web/workers/indirect_draw.js`
// (`wgslStructSource` / `wgslArrayStride`): same field order, snake_case names and strides
// (16 / 20 / 12 bytes, `instance_count` at byte offset 4). The only difference is that
// `{prefix}DrawIndexedIndirectArgs.index_count` is declared `atomic<u32>` because the mesh
// lowering allocates indices with `atomicAdd`; `atomic<u32>` has the same size, alignment and
// offset as `u32`, so the buffer bytes are identical. The lowering emits a single record (not an
// array), at buffer offset 0.

/// `drawIndirect` arguments: `vertexCount, instanceCount, firstVertex, firstInstance`.
pub const DRAW_INDIRECT_ARGS_BYTES: u32 = 16;
/// `drawIndexedIndirect` arguments: `indexCount, instanceCount, firstIndex, baseVertex (i32),
/// firstInstance`.
pub const DRAW_INDEXED_INDIRECT_ARGS_BYTES: u32 = 20;
/// `dispatchWorkgroupsIndirect` arguments: `x, y, z`.
pub const DISPATCH_INDIRECT_ARGS_BYTES: u32 = 12;
/// Words to write into the draw-argument buffer before each mesh-compute dispatch.
pub const DRAW_INDEXED_INDIRECT_RESET: [u32; 5] = [0, 1, 0, 0, 0];
/// Byte offset of `indexCount` (the atomic allocation counter) in the draw-argument buffer.
pub const DRAW_INDEXED_INDEX_COUNT_OFFSET: u32 = 0;

/// Output primitive topology of a mesh shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MeshTopology {
    Points,
    Lines,
    Triangles,
}

impl MeshTopology {
    pub const fn indices_per_primitive(self) -> u32 {
        match self {
            Self::Points => 1,
            Self::Lines => 2,
            Self::Triangles => 3,
        }
    }

    /// WebGPU `GPUPrimitiveTopology` used to draw the lowered output.
    pub const fn webgpu_topology(self) -> &'static str {
        match self {
            Self::Points => "point-list",
            Self::Lines => "line-list",
            Self::Triangles => "triangle-list",
        }
    }
}

/// Format of one mesh output attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshAttributeFormat {
    F32,
    Vec2F32,
    Vec3F32,
    Vec4F32,
    U32,
    Vec2U32,
    Vec3U32,
    Vec4U32,
    I32,
    Vec2I32,
    Vec3I32,
    Vec4I32,
}

impl MeshAttributeFormat {
    const fn components(self) -> u32 {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 1,
            Self::Vec2F32 | Self::Vec2U32 | Self::Vec2I32 => 2,
            Self::Vec3F32 | Self::Vec3U32 | Self::Vec3I32 => 3,
            Self::Vec4F32 | Self::Vec4U32 | Self::Vec4I32 => 4,
        }
    }

    const fn scalar(self) -> &'static str {
        match self {
            Self::F32 | Self::Vec2F32 | Self::Vec3F32 | Self::Vec4F32 => "f32",
            Self::U32 | Self::Vec2U32 | Self::Vec3U32 | Self::Vec4U32 => "u32",
            Self::I32 | Self::Vec2I32 | Self::Vec3I32 | Self::Vec4I32 => "i32",
        }
    }

    /// WGSL type spelling.
    pub fn wgsl_type(self) -> String {
        match self.components() {
            1 => self.scalar().to_string(),
            n => format!("vec{n}<{}>", self.scalar()),
        }
    }

    /// Byte size (WGSL `SizeOf`).
    pub const fn size(self) -> u32 {
        4 * self.components()
    }

    /// Byte alignment (WGSL `AlignOf`: vec3 aligns to 16).
    pub const fn align(self) -> u32 {
        match self.components() {
            1 => 4,
            2 => 8,
            _ => 16,
        }
    }

    /// WebGPU `GPUVertexFormat` for binding the vertex buffer directly.
    pub const fn vertex_format(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::Vec2F32 => "float32x2",
            Self::Vec3F32 => "float32x3",
            Self::Vec4F32 => "float32x4",
            Self::U32 => "uint32",
            Self::Vec2U32 => "uint32x2",
            Self::Vec3U32 => "uint32x3",
            Self::Vec4U32 => "uint32x4",
            Self::I32 => "sint32",
            Self::Vec2I32 => "sint32x2",
            Self::Vec3I32 => "sint32x3",
            Self::Vec4I32 => "sint32x4",
        }
    }
}

/// A named per-vertex or per-primitive mesh output attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshAttribute {
    pub name: String,
    pub format: MeshAttributeFormat,
}

impl MeshAttribute {
    pub fn new(name: &str, format: MeshAttributeFormat) -> Self {
        Self {
            name: name.to_string(),
            format,
        }
    }
}

/// Declared output signature of a mesh shader (`[outputtopology]`, `[numthreads]`, `out
/// vertices/indices/primitives` array sizes and element structs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshOutputSignature {
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub topology: MeshTopology,
    pub vertex_attributes: Vec<MeshAttribute>,
    pub primitive_attributes: Vec<MeshAttribute>,
    /// Mesh threadgroup size; maps 1:1 to `@workgroup_size`.
    pub threads_per_meshlet: [u32; 3],
}

/// One field of a laid-out output struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshFieldLayout {
    pub name: String,
    pub format: MeshAttributeFormat,
    pub offset: u32,
}

/// WGSL host-shareable layout of an output struct (also its array stride).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshStructLayout {
    pub fields: Vec<MeshFieldLayout>,
    pub size: u32,
    pub align: u32,
}

impl MeshStructLayout {
    fn compute(attributes: &[MeshAttribute]) -> Self {
        let mut fields = Vec::with_capacity(attributes.len());
        let mut end = 0u32;
        let mut align = 4u32;
        for attribute in attributes {
            let offset = round_up(attribute.format.align(), end);
            fields.push(MeshFieldLayout {
                name: attribute.name.clone(),
                format: attribute.format,
                offset,
            });
            end = offset + attribute.format.size();
            align = align.max(attribute.format.align());
        }
        Self {
            fields,
            size: round_up(align, end),
            align,
        }
    }

    /// `(GPUVertexFormat, offset, shaderLocation)` per field for a `GPUVertexBufferLayout` with
    /// `arrayStride = size`.
    pub fn vertex_formats(&self) -> Vec<(&'static str, u32, u32)> {
        self.fields
            .iter()
            .enumerate()
            .map(|(location, f)| (f.format.vertex_format(), f.offset, location as u32))
            .collect()
    }
}

/// Buffer layout of a lowered mesh shader's outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshOutputLayout {
    pub vertex: MeshStructLayout,
    pub primitive: Option<MeshStructLayout>,
    pub indices_per_primitive: u32,
    /// Bytes per index (`u32` indices, `GPUIndexFormat "uint32"`).
    pub index_stride: u32,
    pub vertices_per_meshlet: u32,
    pub primitives_per_meshlet: u32,
    pub workgroup_size: [u32; 3],
    pub topology: MeshTopology,
}

impl MeshOutputLayout {
    /// Vertex buffer size for `meshlets` mesh workgroups (fixed per-meshlet slots).
    pub fn vertex_buffer_bytes(&self, meshlets: u32) -> u64 {
        u64::from(meshlets) * u64::from(self.vertices_per_meshlet) * u64::from(self.vertex.size)
    }

    /// Worst-case compacted index buffer size.
    pub fn index_buffer_bytes(&self, meshlets: u32) -> u64 {
        u64::from(meshlets)
            * u64::from(self.primitives_per_meshlet)
            * u64::from(self.indices_per_primitive)
            * u64::from(self.index_stride)
    }

    /// Worst-case compacted per-primitive buffer size (0 without primitive attributes).
    pub fn primitive_buffer_bytes(&self, meshlets: u32) -> u64 {
        self.primitive.as_ref().map_or(0, |p| {
            u64::from(meshlets) * u64::from(self.primitives_per_meshlet) * u64::from(p.size)
        })
    }

    /// Absolute vertex slot (and value stored in the index buffer) of a meshlet-local vertex.
    pub fn vertex_slot(&self, meshlet: u32, local_vertex: u32) -> u64 {
        u64::from(meshlet) * u64::from(self.vertices_per_meshlet) + u64::from(local_vertex)
    }
}

/// Bind group slots of the lowered mesh outputs (all `var<storage, read_write>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshBindings {
    pub group: u32,
    pub vertices: u32,
    pub indices: u32,
    /// Used only when the signature has primitive attributes.
    pub primitives: u32,
    pub draw_args: u32,
    /// `(dispatch_args, task_queue)` bindings for amplification (task) emission, if lowered.
    pub amplification: Option<(u32, u32)>,
}

impl Default for MeshBindings {
    fn default() -> Self {
        Self {
            group: 0,
            vertices: 0,
            indices: 1,
            primitives: 2,
            draw_args: 3,
            amplification: None,
        }
    }
}

/// Invalid mesh output signature or dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshLoweringError {
    OutOfRange {
        what: &'static str,
        value: u64,
        max: u64,
    },
    NoVertexAttributes,
    InvalidAttributeName(String),
    DuplicateAttribute(String),
    DuplicateBinding(u32),
    InvalidPrefix(String),
}

impl fmt::Display for MeshLoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { what, value, max } => {
                write!(f, "{what} = {value} is outside 1..={max}")
            }
            Self::NoVertexAttributes => f.write_str("mesh output declares no vertex attributes"),
            Self::InvalidAttributeName(name) => {
                write!(f, "'{name}' is not a valid WGSL identifier")
            }
            Self::DuplicateAttribute(name) => write!(f, "attribute '{name}' is declared twice"),
            Self::DuplicateBinding(binding) => {
                write!(
                    f,
                    "binding {binding} is used by more than one mesh output buffer"
                )
            }
            Self::InvalidPrefix(prefix) => write!(f, "'{prefix}' is not a valid identifier prefix"),
        }
    }
}

impl std::error::Error for MeshLoweringError {}

const WGSL_RESERVED: &[&str] = &[
    "alias",
    "break",
    "case",
    "const",
    "const_assert",
    "continue",
    "continuing",
    "default",
    "diagnostic",
    "discard",
    "else",
    "enable",
    "false",
    "fn",
    "for",
    "if",
    "let",
    "loop",
    "override",
    "requires",
    "return",
    "struct",
    "switch",
    "true",
    "var",
    "while",
    "array",
    "atomic",
    "bool",
    "f16",
    "f32",
    "i32",
    "mat2x2",
    "mat3x3",
    "mat4x4",
    "ptr",
    "sampler",
    "texture_2d",
    "u32",
    "vec2",
    "vec3",
    "vec4",
];

fn is_wgsl_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.starts_with("__")
        && name != "_"
        && !WGSL_RESERVED.contains(&name)
}

fn check_range(what: &'static str, value: u64, max: u64) -> Result<(), MeshLoweringError> {
    if value == 0 || value > max {
        Err(MeshLoweringError::OutOfRange { what, value, max })
    } else {
        Ok(())
    }
}

/// Validate a signature against D3D12 mesh limits and compute its output buffer layout.
pub fn mesh_output_layout(
    signature: &MeshOutputSignature,
) -> Result<MeshOutputLayout, MeshLoweringError> {
    check_range(
        "max_vertices",
        u64::from(signature.max_vertices),
        u64::from(D3D_MAX_MESH_OUTPUT_VERTICES),
    )?;
    check_range(
        "max_primitives",
        u64::from(signature.max_primitives),
        u64::from(D3D_MAX_MESH_OUTPUT_PRIMITIVES),
    )?;
    let [x, y, z] = signature.threads_per_meshlet;
    for (what, value) in [("threads_x", x), ("threads_y", y), ("threads_z", z)] {
        check_range(what, u64::from(value), u64::from(D3D_MAX_MESH_THREADS))?;
    }
    check_range(
        "threads_per_meshlet",
        u64::from(x) * u64::from(y) * u64::from(z),
        u64::from(D3D_MAX_MESH_THREADS),
    )?;
    if signature.vertex_attributes.is_empty() {
        return Err(MeshLoweringError::NoVertexAttributes);
    }
    for attributes in [
        &signature.vertex_attributes,
        &signature.primitive_attributes,
    ] {
        let mut seen = HashSet::new();
        for attribute in attributes {
            if !is_wgsl_identifier(&attribute.name) {
                return Err(MeshLoweringError::InvalidAttributeName(
                    attribute.name.clone(),
                ));
            }
            if !seen.insert(attribute.name.as_str()) {
                return Err(MeshLoweringError::DuplicateAttribute(
                    attribute.name.clone(),
                ));
            }
        }
    }
    Ok(MeshOutputLayout {
        vertex: MeshStructLayout::compute(&signature.vertex_attributes),
        primitive: (!signature.primitive_attributes.is_empty())
            .then(|| MeshStructLayout::compute(&signature.primitive_attributes)),
        indices_per_primitive: signature.topology.indices_per_primitive(),
        index_stride: 4,
        vertices_per_meshlet: signature.max_vertices,
        primitives_per_meshlet: signature.max_primitives,
        workgroup_size: signature.threads_per_meshlet,
        topology: signature.topology,
    })
}

/// Split `meshlets` mesh workgroups into a dispatch grid that respects
/// `maxComputeWorkgroupsPerDimension` (65535). Padding groups are rejected in the shader by
/// `{prefix}meshlet_index`.
pub fn mesh_dispatch_grid(meshlets: u64) -> Result<[u32; 3], MeshLoweringError> {
    let max = u64::from(WEBGPU_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION);
    if meshlets == 0 {
        return Ok([0, 1, 1]);
    }
    if meshlets > max * max * max {
        return Err(MeshLoweringError::OutOfRange {
            what: "meshlets",
            value: meshlets,
            max: max * max * max,
        });
    }
    let x = meshlets.min(max);
    let rows = meshlets.div_ceil(x);
    let y = rows.min(max);
    let z = rows.div_ceil(y);
    Ok([x as u32, y as u32, z as u32])
}

fn write_struct(out: &mut String, name: &str, layout: &MeshStructLayout) {
    let _ = writeln!(out, "struct {name} {{");
    for field in &layout.fields {
        let _ = writeln!(
            out,
            "    {}: {}, // offset {}",
            field.name,
            field.format.wgsl_type(),
            field.offset
        );
    }
    let _ = writeln!(out, "}}");
}

/// WGSL declarations and helpers for a `--mesh-compute` mesh shader's outputs (see the module
/// documentation for the buffer contract). All identifiers start with `prefix`.
///
/// Emitted: `{p}Vertex` / `{p}Primitive` structs, the four storage bindings, the
/// `drawIndexedIndirect` argument struct, `{p}meshlet_index`, `{p}vertex_slot`,
/// `{p}allocate_primitives` (`SetMeshOutputCounts`; must be called by every invocation in
/// workgroup-uniform control flow), `{p}write_vertex`, the topology's primitive writer
/// (`{p}write_triangle` with `SV_CullPrimitive`, `{p}write_line`, or `{p}write_point`),
/// `{p}write_primitive`, the frustum test `{p}sphere_in_frustum`, and, with
/// [`MeshBindings::amplification`], `{p}emit_mesh_tasks` / `{p}mesh_task_parent` /
/// `{p}finalize_mesh_dispatch`.
pub fn mesh_output_wgsl(
    signature: &MeshOutputSignature,
    bindings: &MeshBindings,
    prefix: &str,
) -> Result<String, MeshLoweringError> {
    let layout = mesh_output_layout(signature)?;
    if !prefix.is_empty() && !is_wgsl_identifier(prefix.trim_end_matches('_')) {
        return Err(MeshLoweringError::InvalidPrefix(prefix.to_string()));
    }
    let mut used = BTreeSet::new();
    let mut slots = vec![bindings.vertices, bindings.indices, bindings.draw_args];
    if layout.primitive.is_some() {
        slots.push(bindings.primitives);
    }
    if let Some((dispatch, queue)) = bindings.amplification {
        slots.extend([dispatch, queue]);
    }
    for slot in slots {
        if !used.insert(slot) {
            return Err(MeshLoweringError::DuplicateBinding(slot));
        }
    }

    let p = prefix;
    let g = bindings.group;
    let ipp = layout.indices_per_primitive;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// mesh-compute outputs: one workgroup per meshlet, {} x {} x {} threads",
        layout.workgroup_size[0], layout.workgroup_size[1], layout.workgroup_size[2]
    );
    let _ = writeln!(
        out,
        "const {p}MAX_VERTICES: u32 = {}u;",
        layout.vertices_per_meshlet
    );
    let _ = writeln!(
        out,
        "const {p}MAX_PRIMITIVES: u32 = {}u;",
        layout.primitives_per_meshlet
    );
    let _ = writeln!(out, "const {p}INDICES_PER_PRIMITIVE: u32 = {ipp}u;");
    out.push('\n');
    write_struct(&mut out, &format!("{p}Vertex"), &layout.vertex);
    if let Some(primitive) = &layout.primitive {
        write_struct(&mut out, &format!("{p}Primitive"), primitive);
    }
    let _ = writeln!(
        out,
        "struct {p}DrawIndexedIndirectArgs {{\n    index_count: atomic<u32>,\n    instance_count: u32,\n    first_index: u32,\n    base_vertex: i32,\n    first_instance: u32,\n}}\n"
    );
    let _ = writeln!(
        out,
        "@group({g}) @binding({}) var<storage, read_write> {p}vertices: array<{p}Vertex>;",
        bindings.vertices
    );
    let _ = writeln!(
        out,
        "@group({g}) @binding({}) var<storage, read_write> {p}indices: array<u32>;",
        bindings.indices
    );
    if layout.primitive.is_some() {
        let _ = writeln!(
            out,
            "@group({g}) @binding({}) var<storage, read_write> {p}primitives: array<{p}Primitive>;",
            bindings.primitives
        );
    }
    let _ = writeln!(
        out,
        "@group({g}) @binding({}) var<storage, read_write> {p}draw_args: {p}DrawIndexedIndirectArgs;",
        bindings.draw_args
    );
    let _ = writeln!(out, "var<workgroup> {p}primitive_base: u32;\n");

    let _ = writeln!(
        out,
        "// Flattened meshlet index of a (possibly 2D/3D-split) dispatch; 0xffffffffu marks padding groups.
fn {p}meshlet_index(workgroup_id: vec3<u32>, num_workgroups: vec3<u32>, meshlet_count: u32) -> u32 {{
    let flat = workgroup_id.x + num_workgroups.x * (workgroup_id.y + num_workgroups.y * workgroup_id.z);
    return select(0xffffffffu, flat, flat < meshlet_count);
}}

fn {p}vertex_slot(meshlet: u32, local_vertex: u32) -> u32 {{
    return meshlet * {p}MAX_VERTICES + min(local_vertex, {p}MAX_VERTICES - 1u);
}}

// SetMeshOutputCounts: call from every invocation in workgroup-uniform control flow.
// Returns the meshlet's first primitive in the compacted index stream.
fn {p}allocate_primitives(local_index: u32, primitive_count: u32) -> u32 {{
    if (local_index == 0u) {{
        let count = min(primitive_count, {p}MAX_PRIMITIVES);
        {p}primitive_base = atomicAdd(&{p}draw_args.index_count, count * {p}INDICES_PER_PRIMITIVE) / {p}INDICES_PER_PRIMITIVE;
    }}
    return workgroupUniformLoad(&{p}primitive_base);
}}

fn {p}write_vertex(meshlet: u32, local_vertex: u32, vertex: {p}Vertex) {{
    if (local_vertex < {p}MAX_VERTICES) {{
        {p}vertices[meshlet * {p}MAX_VERTICES + local_vertex] = vertex;
    }}
}}
"
    );
    match layout.topology {
        MeshTopology::Triangles => {
            let _ = writeln!(
                out,
                "// SV_CullPrimitive is lowered to a degenerate (zero-area) triangle.
fn {p}write_triangle(meshlet: u32, primitive_base: u32, primitive: u32, primitive_count: u32, local_indices: vec3<u32>, culled: bool) {{
    if (primitive < min(primitive_count, {p}MAX_PRIMITIVES)) {{
        let first = (primitive_base + primitive) * 3u;
        let slots = vec3<u32>({p}vertex_slot(meshlet, local_indices.x), {p}vertex_slot(meshlet, local_indices.y), {p}vertex_slot(meshlet, local_indices.z));
        let written = select(slots, vec3<u32>(slots.x), culled);
        {p}indices[first] = written.x;
        {p}indices[first + 1u] = written.y;
        {p}indices[first + 2u] = written.z;
    }}
}}
"
            );
        }
        MeshTopology::Lines => {
            let _ = writeln!(
                out,
                "fn {p}write_line(meshlet: u32, primitive_base: u32, primitive: u32, primitive_count: u32, local_indices: vec2<u32>) {{
    if (primitive < min(primitive_count, {p}MAX_PRIMITIVES)) {{
        let first = (primitive_base + primitive) * 2u;
        {p}indices[first] = {p}vertex_slot(meshlet, local_indices.x);
        {p}indices[first + 1u] = {p}vertex_slot(meshlet, local_indices.y);
    }}
}}
"
            );
        }
        MeshTopology::Points => {
            let _ = writeln!(
                out,
                "fn {p}write_point(meshlet: u32, primitive_base: u32, primitive: u32, primitive_count: u32, local_index_value: u32) {{
    if (primitive < min(primitive_count, {p}MAX_PRIMITIVES)) {{
        {p}indices[primitive_base + primitive] = {p}vertex_slot(meshlet, local_index_value);
    }}
}}
"
            );
        }
    }
    if layout.primitive.is_some() {
        let _ = writeln!(
            out,
            "fn {p}write_primitive(primitive_base: u32, primitive: u32, primitive_count: u32, value: {p}Primitive) {{
    if (primitive < min(primitive_count, {p}MAX_PRIMITIVES)) {{
        {p}primitives[primitive_base + primitive] = value;
    }}
}}
"
        );
    }
    let _ = writeln!(
        out,
        "// Meshlet culling: bounding sphere (xyz = center, w = radius) against 6 inward-facing planes (n.p + d >= 0 inside).
fn {p}sphere_in_frustum(sphere: vec4<f32>, planes: array<vec4<f32>, 6>) -> bool {{
    let c = sphere.xyz;
    let r = -sphere.w;
    return dot(planes[0].xyz, c) + planes[0].w >= r
        && dot(planes[1].xyz, c) + planes[1].w >= r
        && dot(planes[2].xyz, c) + planes[2].w >= r
        && dot(planes[3].xyz, c) + planes[3].w >= r
        && dot(planes[4].xyz, c) + planes[4].w >= r
        && dot(planes[5].xyz, c) + planes[5].w >= r;
}}"
    );
    if let Some((dispatch, queue)) = bindings.amplification {
        let _ = writeln!(
            out,
            "
struct {p}DispatchIndirectArgs {{
    x: u32,
    y: u32,
    z: u32,
}}

struct {p}MeshTaskQueue {{
    count: atomic<u32>,
    parents: array<u32>,
}}

const {p}MAX_MESH_GROUPS: u32 = {WEBGPU_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION}u;

@group({g}) @binding({dispatch}) var<storage, read_write> {p}mesh_dispatch: {p}DispatchIndirectArgs;
@group({g}) @binding({queue}) var<storage, read_write> {p}mesh_tasks: {p}MeshTaskQueue;

// DispatchMesh from an amplification workgroup: call from one invocation; appends `count` mesh
// groups whose parent is `task_group` and returns the first mesh group index.
fn {p}emit_mesh_tasks(task_group: u32, count: u32) -> u32 {{
    let base = atomicAdd(&{p}mesh_tasks.count, count);
    for (var i = 0u; i < count; i = i + 1u) {{
        if (base + i < {p}MAX_MESH_GROUPS) {{
            {p}mesh_tasks.parents[base + i] = task_group;
        }}
    }}
    return base;
}}

fn {p}mesh_task_parent(mesh_group: u32) -> u32 {{
    return {p}mesh_tasks.parents[mesh_group];
}}

// Run in a single invocation between the amplification and mesh passes; mesh groups beyond
// {p}MAX_MESH_GROUPS are dropped.
fn {p}finalize_mesh_dispatch() {{
    let count = min(atomicLoad(&{p}mesh_tasks.count), {p}MAX_MESH_GROUPS);
    {p}mesh_dispatch = {p}DispatchIndirectArgs(count, 1u, 1u);
}}"
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod interp {
    //! Test-only reference interpreter for the naga IR subset emitted by this module and by
    //! naga's WGSL / SPIR-V front ends for the synthetic test shaders. One OS thread per
    //! invocation, workgroup/storage memory behind a mutex, and real barriers, so the
    //! workgroup-shared emulation helpers execute with the synchronization they rely on.

    use std::collections::HashMap;
    use std::sync::{Condvar, Mutex};

    use naga::{
        AddressSpace, ArraySize, AtomicFunction, BinaryOperator, Binding, Block, BuiltIn,
        Expression, Function, Handle, Literal, MathFunction, Module, RelationalFunction,
        ScalarKind, Statement, Type, TypeInner, UnaryOperator,
    };

    /// Length used for runtime-sized arrays.
    pub const RUNTIME_ARRAY_LEN: usize = 64;

    #[derive(Clone, Debug, PartialEq)]
    pub enum Val {
        U(u32),
        I(i32),
        F(f32),
        B(bool),
        C(Vec<Val>),
        P(Ptr),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct Ptr {
        global: bool,
        root: usize,
        path: Vec<usize>,
    }

    impl Val {
        pub fn u(&self) -> u32 {
            match self {
                Val::U(v) => *v,
                Val::I(v) => *v as u32,
                Val::F(v) => v.to_bits(),
                Val::B(v) => u32::from(*v),
                other => panic!("expected a scalar, got {other:?}"),
            }
        }

        pub fn b(&self) -> bool {
            match self {
                Val::B(v) => *v,
                other => panic!("expected bool, got {other:?}"),
            }
        }

        pub fn comps(&self) -> &[Val] {
            match self {
                Val::C(c) => c,
                other => panic!("expected a composite, got {other:?}"),
            }
        }

        pub fn u64_pair(value: u64) -> Val {
            Val::C(vec![Val::U(value as u32), Val::U((value >> 32) as u32)])
        }

        pub fn to_u64(&self) -> u64 {
            let c = self.comps();
            u64::from(c[0].u()) | (u64::from(c[1].u()) << 32)
        }

        fn index(&self) -> usize {
            match self {
                Val::U(v) => *v as usize,
                Val::I(v) => *v as usize,
                other => panic!("expected an index, got {other:?}"),
            }
        }

        fn ptr(self) -> Ptr {
            match self {
                Val::P(p) => p,
                other => panic!("expected a pointer, got {other:?}"),
            }
        }
    }

    pub fn zero(module: &Module, ty: Handle<Type>) -> Val {
        match &module.types[ty].inner {
            TypeInner::Scalar(s) | TypeInner::Atomic(s) => scalar_zero(s.kind),
            TypeInner::Vector { size, scalar } => {
                Val::C(vec![scalar_zero(scalar.kind); *size as usize])
            }
            TypeInner::Array { base, size, .. } => {
                let len = match size {
                    ArraySize::Constant(n) => n.get() as usize,
                    _ => RUNTIME_ARRAY_LEN,
                };
                Val::C(vec![zero(module, *base); len])
            }
            TypeInner::Struct { members, .. } => {
                Val::C(members.iter().map(|m| zero(module, m.ty)).collect())
            }
            other => panic!("unsupported type {other:?}"),
        }
    }

    fn scalar_zero(kind: ScalarKind) -> Val {
        match kind {
            ScalarKind::Uint => Val::U(0),
            ScalarKind::Sint => Val::I(0),
            ScalarKind::Float => Val::F(0.0),
            ScalarKind::Bool => Val::B(false),
            other => panic!("unsupported scalar {other:?}"),
        }
    }

    fn literal(l: Literal) -> Val {
        match l {
            Literal::U32(v) => Val::U(v),
            Literal::I32(v) => Val::I(v),
            Literal::F32(v) => Val::F(v),
            Literal::Bool(v) => Val::B(v),
            other => panic!("unsupported literal {other:?}"),
        }
    }

    fn eval_const(module: &Module, handle: Handle<Expression>) -> Val {
        match &module.global_expressions[handle] {
            Expression::Literal(l) => literal(*l),
            Expression::ZeroValue(ty) => zero(module, *ty),
            Expression::Constant(c) => eval_const(module, module.constants[*c].init),
            Expression::Compose { components, .. } => {
                Val::C(components.iter().map(|&h| eval_const(module, h)).collect())
            }
            Expression::Splat { size, value } => {
                Val::C(vec![eval_const(module, *value); *size as usize])
            }
            other => panic!("unsupported constant expression {other:?}"),
        }
    }

    fn get<'v>(mut value: &'v Val, path: &[usize]) -> &'v Val {
        for &i in path {
            value = &value.comps()[i];
        }
        value
    }

    fn get_mut<'v>(mut value: &'v mut Val, path: &[usize]) -> &'v mut Val {
        for &i in path {
            value = match value {
                Val::C(c) => &mut c[i],
                other => panic!("cannot index {other:?}"),
            };
        }
        value
    }

    /// Barrier that fails every waiter once a participant panicked (no hung tests).
    pub struct TestBarrier {
        state: Mutex<(usize, usize, bool)>,
        condvar: Condvar,
        participants: usize,
    }

    impl TestBarrier {
        pub fn new(participants: usize) -> Self {
            Self {
                state: Mutex::new((0, 0, false)),
                condvar: Condvar::new(),
                participants,
            }
        }

        fn wait(&self) {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            assert!(!state.2, "another invocation panicked");
            let generation = state.1;
            state.0 += 1;
            if state.0 == self.participants {
                state.0 = 0;
                state.1 += 1;
                self.condvar.notify_all();
                return;
            }
            while state.1 == generation && !state.2 {
                state = self.condvar.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            assert!(!state.2, "another invocation panicked");
        }

        fn poison(&self) {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.2 = true;
            self.condvar.notify_all();
        }
    }

    struct PoisonOnPanic<'a>(&'a TestBarrier);

    impl Drop for PoisonOnPanic<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.0.poison();
            }
        }
    }

    /// Non-private module memory (workgroup, storage, uniform), keyed by global index.
    pub struct Memory {
        globals: Mutex<HashMap<usize, Val>>,
    }

    impl Memory {
        pub fn new(module: &Module) -> Self {
            let globals = module
                .global_variables
                .iter()
                .filter(|(_, g)| g.space != AddressSpace::Private)
                .map(|(h, g)| (h.index(), zero(module, g.ty)))
                .collect();
            Self {
                globals: Mutex::new(globals),
            }
        }

        fn index_of(module: &Module, name: &str) -> usize {
            module
                .global_variables
                .iter()
                .find(|(_, g)| g.name.as_deref() == Some(name))
                .map(|(h, _)| h.index())
                .unwrap_or_else(|| panic!("no global named {name}"))
        }

        pub fn set(&self, module: &Module, name: &str, value: Val) {
            let index = Self::index_of(module, name);
            self.lock().insert(index, value);
        }

        pub fn get(&self, module: &Module, name: &str) -> Val {
            let index = Self::index_of(module, name);
            self.lock()[&index].clone()
        }

        fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<usize, Val>> {
            self.globals.lock().unwrap_or_else(|e| e.into_inner())
        }

        fn reset_workgroup(&self, module: &Module) {
            let mut globals = self.lock();
            for (h, g) in module.global_variables.iter() {
                if g.space == AddressSpace::WorkGroup {
                    globals.insert(h.index(), zero(module, g.ty));
                }
            }
        }
    }

    enum Flow {
        Next,
        Break,
        Continue,
        Return(Option<Val>),
    }

    struct Frame<'f> {
        function: &'f Function,
        args: Vec<Val>,
        locals: Vec<Val>,
        values: HashMap<Handle<Expression>, Val>,
    }

    struct Invocation<'a> {
        module: &'a Module,
        memory: &'a Memory,
        private: HashMap<usize, Val>,
        barrier: &'a TestBarrier,
    }

    impl<'a> Invocation<'a> {
        fn new(module: &'a Module, memory: &'a Memory, barrier: &'a TestBarrier) -> Self {
            let private = module
                .global_variables
                .iter()
                .filter(|(_, g)| g.space == AddressSpace::Private)
                .map(|(h, g)| {
                    let value = match g.init {
                        Some(init) => eval_const(module, init),
                        None => zero(module, g.ty),
                    };
                    (h.index(), value)
                })
                .collect();
            Self {
                module,
                memory,
                private,
                barrier,
            }
        }

        fn call(&mut self, function: &Function, args: Vec<Val>) -> Option<Val> {
            let mut frame = Frame {
                function,
                args,
                locals: function
                    .local_variables
                    .iter()
                    .map(|(_, lv)| zero(self.module, lv.ty))
                    .collect(),
                values: HashMap::new(),
            };
            for (index, (_, local)) in function.local_variables.iter().enumerate() {
                if let Some(init) = local.init {
                    frame.locals[index] = self.eval(&mut frame, init);
                }
            }
            match self.block(&mut frame, &function.body) {
                Flow::Return(value) => value,
                _ => None,
            }
        }

        fn is_private(&self, ptr: &Ptr) -> bool {
            ptr.global && self.private.contains_key(&ptr.root)
        }

        fn read(&self, frame: &Frame, ptr: &Ptr) -> Val {
            if !ptr.global {
                get(&frame.locals[ptr.root], &ptr.path).clone()
            } else if self.is_private(ptr) {
                get(&self.private[&ptr.root], &ptr.path).clone()
            } else {
                let globals = self.memory.lock();
                get(&globals[&ptr.root], &ptr.path).clone()
            }
        }

        fn modify(&mut self, frame: &mut Frame, ptr: &Ptr, f: impl FnOnce(&mut Val) -> Val) -> Val {
            if !ptr.global {
                f(get_mut(&mut frame.locals[ptr.root], &ptr.path))
            } else if self.is_private(ptr) {
                let root = self.private.get_mut(&ptr.root).expect("private global");
                f(get_mut(root, &ptr.path))
            } else {
                let mut globals = self.memory.lock();
                let root = globals.get_mut(&ptr.root).expect("module global");
                f(get_mut(root, &ptr.path))
            }
        }

        fn block(&mut self, frame: &mut Frame, block: &Block) -> Flow {
            for statement in block.iter() {
                let flow = self.statement(frame, statement);
                if !matches!(flow, Flow::Next) {
                    return flow;
                }
            }
            Flow::Next
        }

        fn statement(&mut self, frame: &mut Frame, statement: &Statement) -> Flow {
            match statement {
                Statement::Emit(range) => {
                    for handle in range.clone() {
                        let value = self.compute(frame, handle);
                        frame.values.insert(handle, value);
                    }
                }
                Statement::Block(inner) => return self.block(frame, inner),
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let taken = if self.eval(frame, *condition).b() {
                        accept
                    } else {
                        reject
                    };
                    return self.block(frame, taken);
                }
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => loop {
                    match self.block(frame, body) {
                        Flow::Break => break,
                        Flow::Return(v) => return Flow::Return(v),
                        Flow::Next | Flow::Continue => {}
                    }
                    match self.block(frame, continuing) {
                        Flow::Return(v) => return Flow::Return(v),
                        Flow::Break => break,
                        Flow::Next | Flow::Continue => {}
                    }
                    if let Some(condition) = break_if
                        && self.eval(frame, *condition).b()
                    {
                        break;
                    }
                },
                Statement::Break => return Flow::Break,
                Statement::Continue => return Flow::Continue,
                Statement::Return { value } => {
                    return Flow::Return(value.map(|h| self.eval(frame, h)));
                }
                Statement::ControlBarrier(_) => self.barrier.wait(),
                Statement::MemoryBarrier(_) => {}
                Statement::Store { pointer, value } => {
                    let ptr = self.eval(frame, *pointer).ptr();
                    let value = self.eval(frame, *value);
                    self.modify(frame, &ptr, |slot| std::mem::replace(slot, value));
                }
                Statement::Call {
                    function,
                    arguments,
                    result,
                } => {
                    let args = arguments.iter().map(|&a| self.eval(frame, a)).collect();
                    let module = self.module;
                    let value = self.call(&module.functions[*function], args);
                    if let Some(result) = result {
                        let value = value.expect("call without a return value");
                        frame.values.insert(*result, value);
                    }
                }
                Statement::Atomic {
                    pointer,
                    fun,
                    value,
                    result,
                } => {
                    let ptr = self.eval(frame, *pointer).ptr();
                    let operand = self.eval(frame, *value).u();
                    let fun = *fun;
                    let old = self.modify(frame, &ptr, |slot| {
                        let old = slot.u();
                        let new = match fun {
                            AtomicFunction::Add => old.wrapping_add(operand),
                            AtomicFunction::Subtract => old.wrapping_sub(operand),
                            AtomicFunction::InclusiveOr => old | operand,
                            AtomicFunction::And => old & operand,
                            AtomicFunction::Max => old.max(operand),
                            AtomicFunction::Min => old.min(operand),
                            other => panic!("unsupported atomic {other:?}"),
                        };
                        std::mem::replace(slot, Val::U(new))
                    });
                    if let Some(result) = result {
                        frame.values.insert(*result, old);
                    }
                }
                Statement::WorkGroupUniformLoad { pointer, result } => {
                    let ptr = self.eval(frame, *pointer).ptr();
                    self.barrier.wait();
                    let value = self.read(frame, &ptr);
                    self.barrier.wait();
                    frame.values.insert(*result, value);
                }
                other => panic!("unsupported statement {other:?}"),
            }
            Flow::Next
        }

        fn eval(&mut self, frame: &mut Frame, handle: Handle<Expression>) -> Val {
            if let Some(value) = frame.values.get(&handle) {
                return value.clone();
            }
            self.compute(frame, handle)
        }

        fn compute(&mut self, frame: &mut Frame, handle: Handle<Expression>) -> Val {
            let function = frame.function;
            match &function.expressions[handle] {
                Expression::Literal(l) => literal(*l),
                Expression::Constant(c) => eval_const(self.module, self.module.constants[*c].init),
                Expression::ZeroValue(ty) => zero(self.module, *ty),
                Expression::Compose { components, .. } => {
                    Val::C(components.iter().map(|&h| self.eval(frame, h)).collect())
                }
                Expression::Splat { size, value } => {
                    Val::C(vec![self.eval(frame, *value); *size as usize])
                }
                Expression::Swizzle {
                    size,
                    vector,
                    pattern,
                } => {
                    let v = self.eval(frame, *vector);
                    Val::C(
                        pattern[..*size as usize]
                            .iter()
                            .map(|c| v.comps()[*c as usize].clone())
                            .collect(),
                    )
                }
                Expression::AccessIndex { base, index } => {
                    let base = self.eval(frame, *base);
                    access(base, *index as usize)
                }
                Expression::Access { base, index } => {
                    let base = self.eval(frame, *base);
                    let index = self.eval(frame, *index).index();
                    access(base, index)
                }
                Expression::Load { pointer } => {
                    let ptr = self.eval(frame, *pointer).ptr();
                    self.read(frame, &ptr)
                }
                Expression::Unary { op, expr } => {
                    let v = self.eval(frame, *expr);
                    map1(&v, &|x| unary(*op, x))
                }
                Expression::Binary { op, left, right } => {
                    let l = self.eval(frame, *left);
                    let r = self.eval(frame, *right);
                    map2(&l, &r, &|a, b| binary(*op, a, b))
                }
                Expression::Select {
                    condition,
                    accept,
                    reject,
                } => {
                    let c = self.eval(frame, *condition);
                    let a = self.eval(frame, *accept);
                    let r = self.eval(frame, *reject);
                    match c {
                        Val::B(true) => a,
                        Val::B(false) => r,
                        Val::C(cs) => Val::C(
                            cs.iter()
                                .enumerate()
                                .map(|(i, c)| {
                                    if c.b() {
                                        a.comps()[i].clone()
                                    } else {
                                        r.comps()[i].clone()
                                    }
                                })
                                .collect(),
                        ),
                        other => panic!("bad select condition {other:?}"),
                    }
                }
                Expression::Math { fun, arg, arg1, .. } => {
                    let a = self.eval(frame, *arg);
                    let b = arg1.map(|h| self.eval(frame, h));
                    math(*fun, &a, b.as_ref())
                }
                Expression::As {
                    expr,
                    kind,
                    convert,
                } => {
                    let v = self.eval(frame, *expr);
                    map1(&v, &|x| cast(x, *kind, convert.is_some()))
                }
                Expression::Relational { fun, argument } => {
                    let v = self.eval(frame, *argument);
                    let mut bools = v.comps().iter().map(Val::b);
                    Val::B(match fun {
                        RelationalFunction::All => bools.all(|b| b),
                        RelationalFunction::Any => bools.any(|b| b),
                        other => panic!("unsupported relational {other:?}"),
                    })
                }
                Expression::FunctionArgument(i) => frame.args[*i as usize].clone(),
                Expression::GlobalVariable(g) => Val::P(Ptr {
                    global: true,
                    root: g.index(),
                    path: Vec::new(),
                }),
                Expression::LocalVariable(l) => Val::P(Ptr {
                    global: false,
                    root: l.index(),
                    path: Vec::new(),
                }),
                Expression::ArrayLength(_) => Val::U(RUNTIME_ARRAY_LEN as u32),
                other => panic!("unsupported or not-yet-produced expression {other:?}"),
            }
        }
    }

    fn access(base: Val, index: usize) -> Val {
        match base {
            Val::P(mut p) => {
                p.path.push(index);
                Val::P(p)
            }
            Val::C(c) => c[index].clone(),
            other => panic!("cannot index {other:?}"),
        }
    }

    fn map1(v: &Val, f: &dyn Fn(&Val) -> Val) -> Val {
        match v {
            Val::C(c) => Val::C(c.iter().map(f).collect()),
            scalar => f(scalar),
        }
    }

    fn map2(a: &Val, b: &Val, f: &dyn Fn(&Val, &Val) -> Val) -> Val {
        match (a, b) {
            (Val::C(x), Val::C(y)) => Val::C(x.iter().zip(y).map(|(p, q)| f(p, q)).collect()),
            (Val::C(x), s) => Val::C(x.iter().map(|p| f(p, s)).collect()),
            (s, Val::C(y)) => Val::C(y.iter().map(|q| f(s, q)).collect()),
            (p, q) => f(p, q),
        }
    }

    fn unary(op: UnaryOperator, v: &Val) -> Val {
        match (op, v) {
            (UnaryOperator::Negate, Val::I(x)) => Val::I(x.wrapping_neg()),
            (UnaryOperator::Negate, Val::F(x)) => Val::F(-x),
            (UnaryOperator::LogicalNot, Val::B(x)) => Val::B(!x),
            (UnaryOperator::BitwiseNot, Val::U(x)) => Val::U(!x),
            (UnaryOperator::BitwiseNot, Val::I(x)) => Val::I(!x),
            other => panic!("unsupported unary {other:?}"),
        }
    }

    /// WGSL scalar semantics: wrapping integer arithmetic, shift amounts modulo 32.
    fn binary(op: BinaryOperator, a: &Val, b: &Val) -> Val {
        use BinaryOperator as Op;
        match (a, b) {
            (Val::U(x), Val::U(y)) => {
                let (x, y) = (*x, *y);
                match op {
                    Op::Add => Val::U(x.wrapping_add(y)),
                    Op::Subtract => Val::U(x.wrapping_sub(y)),
                    Op::Multiply => Val::U(x.wrapping_mul(y)),
                    Op::Divide => Val::U(x.checked_div(y).unwrap_or(x)),
                    Op::Modulo => Val::U(if y == 0 { 0 } else { x % y }),
                    Op::And => Val::U(x & y),
                    Op::InclusiveOr => Val::U(x | y),
                    Op::ExclusiveOr => Val::U(x ^ y),
                    Op::ShiftLeft => Val::U(x.wrapping_shl(y)),
                    Op::ShiftRight => Val::U(x.wrapping_shr(y)),
                    Op::Equal => Val::B(x == y),
                    Op::NotEqual => Val::B(x != y),
                    Op::Less => Val::B(x < y),
                    Op::LessEqual => Val::B(x <= y),
                    Op::Greater => Val::B(x > y),
                    Op::GreaterEqual => Val::B(x >= y),
                    other => panic!("unsupported u32 op {other:?}"),
                }
            }
            (Val::I(x), Val::U(y)) if matches!(op, Op::ShiftLeft | Op::ShiftRight) => {
                if op == Op::ShiftLeft {
                    Val::I(x.wrapping_shl(*y))
                } else {
                    Val::I(x.wrapping_shr(*y))
                }
            }
            (Val::I(x), Val::I(y)) => {
                let (x, y) = (*x, *y);
                match op {
                    Op::Add => Val::I(x.wrapping_add(y)),
                    Op::Subtract => Val::I(x.wrapping_sub(y)),
                    Op::Multiply => Val::I(x.wrapping_mul(y)),
                    Op::Divide => Val::I(if y == 0 { x } else { x.wrapping_div(y) }),
                    Op::Modulo => Val::I(if y == 0 { 0 } else { x.wrapping_rem(y) }),
                    Op::And => Val::I(x & y),
                    Op::InclusiveOr => Val::I(x | y),
                    Op::ExclusiveOr => Val::I(x ^ y),
                    Op::Equal => Val::B(x == y),
                    Op::NotEqual => Val::B(x != y),
                    Op::Less => Val::B(x < y),
                    Op::LessEqual => Val::B(x <= y),
                    Op::Greater => Val::B(x > y),
                    Op::GreaterEqual => Val::B(x >= y),
                    other => panic!("unsupported i32 op {other:?}"),
                }
            }
            (Val::F(x), Val::F(y)) => {
                let (x, y) = (*x, *y);
                match op {
                    Op::Add => Val::F(x + y),
                    Op::Subtract => Val::F(x - y),
                    Op::Multiply => Val::F(x * y),
                    Op::Divide => Val::F(x / y),
                    Op::Equal => Val::B(x == y),
                    Op::NotEqual => Val::B(x != y),
                    Op::Less => Val::B(x < y),
                    Op::LessEqual => Val::B(x <= y),
                    Op::Greater => Val::B(x > y),
                    Op::GreaterEqual => Val::B(x >= y),
                    other => panic!("unsupported f32 op {other:?}"),
                }
            }
            (Val::B(x), Val::B(y)) => {
                let (x, y) = (*x, *y);
                match op {
                    Op::LogicalAnd | Op::And => Val::B(x && y),
                    Op::LogicalOr | Op::InclusiveOr => Val::B(x || y),
                    Op::Equal => Val::B(x == y),
                    Op::NotEqual | Op::ExclusiveOr => Val::B(x != y),
                    other => panic!("unsupported bool op {other:?}"),
                }
            }
            other => panic!("unsupported operands {other:?}"),
        }
    }

    fn math(fun: MathFunction, a: &Val, b: Option<&Val>) -> Val {
        match fun {
            MathFunction::Min | MathFunction::Max => {
                let min = fun == MathFunction::Min;
                map2(a, b.expect("binary math"), &|x, y| match (x, y) {
                    (Val::U(p), Val::U(q)) => Val::U(if min { *p.min(q) } else { *p.max(q) }),
                    (Val::I(p), Val::I(q)) => Val::I(if min { *p.min(q) } else { *p.max(q) }),
                    (Val::F(p), Val::F(q)) => Val::F(if min { p.min(*q) } else { p.max(*q) }),
                    other => panic!("unsupported min/max {other:?}"),
                })
            }
            MathFunction::CountOneBits => map1(a, &|x| Val::U(x.u().count_ones())),
            MathFunction::FirstTrailingBit => map1(a, &|x| {
                let v = x.u();
                Val::U(if v == 0 { u32::MAX } else { v.trailing_zeros() })
            }),
            MathFunction::FirstLeadingBit => map1(a, &|x| {
                let v = x.u();
                Val::U(if v == 0 {
                    u32::MAX
                } else {
                    31 - v.leading_zeros()
                })
            }),
            MathFunction::Dot => {
                let b = b.expect("dot");
                Val::F(
                    a.comps()
                        .iter()
                        .zip(b.comps())
                        .map(|(x, y)| match (x, y) {
                            (Val::F(p), Val::F(q)) => p * q,
                            other => panic!("unsupported dot {other:?}"),
                        })
                        .sum(),
                )
            }
            other => panic!("unsupported math {other:?}"),
        }
    }

    fn cast(v: &Val, kind: ScalarKind, convert: bool) -> Val {
        match (v, kind, convert) {
            (Val::U(x), ScalarKind::Uint, _) => Val::U(*x),
            (Val::I(x), ScalarKind::Sint, _) => Val::I(*x),
            (Val::F(x), ScalarKind::Float, _) => Val::F(*x),
            (Val::B(x), ScalarKind::Bool, _) => Val::B(*x),
            (Val::U(x), ScalarKind::Sint, _) => Val::I(*x as i32),
            (Val::I(x), ScalarKind::Uint, _) => Val::U(*x as u32),
            (Val::U(x), ScalarKind::Float, true) => Val::F(*x as f32),
            (Val::I(x), ScalarKind::Float, true) => Val::F(*x as f32),
            (Val::F(x), ScalarKind::Uint, true) => Val::U(*x as u32),
            (Val::F(x), ScalarKind::Sint, true) => Val::I(*x as i32),
            (Val::U(x), ScalarKind::Float, false) => Val::F(f32::from_bits(*x)),
            (Val::I(x), ScalarKind::Float, false) => Val::F(f32::from_bits(*x as u32)),
            (Val::F(x), ScalarKind::Uint, false) => Val::U(x.to_bits()),
            (Val::F(x), ScalarKind::Sint, false) => Val::I(x.to_bits() as i32),
            (Val::B(x), ScalarKind::Uint, true) => Val::U(u32::from(*x)),
            (Val::U(x), ScalarKind::Bool, true) => Val::B(*x != 0),
            other => panic!("unsupported cast {other:?}"),
        }
    }

    struct InvocationIds {
        local: [u32; 3],
        local_index: u32,
        workgroup: [u32; 3],
        size: [u32; 3],
        count: [u32; 3],
    }

    fn builtin_value(builtin: BuiltIn, ids: &InvocationIds) -> Val {
        let vec3 = |v: [u32; 3]| Val::C(v.iter().map(|&x| Val::U(x)).collect());
        match builtin {
            BuiltIn::LocalInvocationIndex => Val::U(ids.local_index),
            BuiltIn::LocalInvocationId => vec3(ids.local),
            BuiltIn::WorkGroupId => vec3(ids.workgroup),
            BuiltIn::NumWorkGroups => vec3(ids.count),
            BuiltIn::GlobalInvocationId => vec3([
                ids.workgroup[0] * ids.size[0] + ids.local[0],
                ids.workgroup[1] * ids.size[1] + ids.local[1],
                ids.workgroup[2] * ids.size[2] + ids.local[2],
            ]),
            other => panic!("unsupported entry built-in {other:?}"),
        }
    }

    /// Run compute entry point `entry` over `workgroups` (workgroups sequentially, one thread
    /// per invocation, workgroup memory zeroed per workgroup).
    pub fn run_compute(module: &Module, memory: &Memory, entry: &str, workgroups: [u32; 3]) {
        let ep = module
            .entry_points
            .iter()
            .find(|ep| ep.name == entry)
            .unwrap_or_else(|| panic!("no entry point {entry}"));
        let size = ep.workgroup_size;
        let total = size[0] * size[1] * size[2];
        for wz in 0..workgroups[2] {
            for wy in 0..workgroups[1] {
                for wx in 0..workgroups[0] {
                    memory.reset_workgroup(module);
                    let barrier = TestBarrier::new(total as usize);
                    std::thread::scope(|scope| {
                        for local_index in 0..total {
                            let barrier = &barrier;
                            scope.spawn(move || {
                                let _guard = PoisonOnPanic(barrier);
                                let ids = InvocationIds {
                                    local: [
                                        local_index % size[0],
                                        (local_index / size[0]) % size[1],
                                        local_index / (size[0] * size[1]),
                                    ],
                                    local_index,
                                    workgroup: [wx, wy, wz],
                                    size,
                                    count: workgroups,
                                };
                                let args = ep
                                    .function
                                    .arguments
                                    .iter()
                                    .map(|arg| match &arg.binding {
                                        Some(Binding::BuiltIn(b)) => builtin_value(*b, &ids),
                                        _ => match &module.types[arg.ty].inner {
                                            TypeInner::Struct { members, .. } => Val::C(
                                                members
                                                    .iter()
                                                    .map(|m| match &m.binding {
                                                        Some(Binding::BuiltIn(b)) => {
                                                            builtin_value(*b, &ids)
                                                        }
                                                        other => panic!("binding {other:?}"),
                                                    })
                                                    .collect(),
                                            ),
                                            other => panic!("unsupported argument {other:?}"),
                                        },
                                    })
                                    .collect();
                                let mut invocation = Invocation::new(module, memory, barrier);
                                invocation.call(&ep.function, args);
                            });
                        }
                    });
                }
            }
        }
    }

    /// Call a non-entry function by name in a single invocation.
    pub fn call_function(module: &Module, name: &str, args: Vec<Val>) -> Val {
        let function = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some(name))
            .map(|(_, f)| f)
            .unwrap_or_else(|| panic!("no function {name}"));
        let memory = Memory::new(module);
        let barrier = TestBarrier::new(1);
        let mut invocation = Invocation::new(module, &memory, &barrier);
        invocation
            .call(function, args)
            .expect("function returned no value")
    }

    /// Value of the first global in `space` (for modules whose globals are unnamed).
    pub fn first_global_in(memory: &Memory, module: &Module, space: AddressSpace) -> Val {
        let index = module
            .global_variables
            .iter()
            .find(|(_, g)| g.space == space)
            .map(|(h, _)| h.index())
            .expect("no global in that address space");
        memory.lock()[&index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::interp::{Memory, Val, call_function, first_global_in, run_compute};
    use super::*;
    use crate::convert::test_spirv::{SpirvBuilder, build_minimal_compute_spv};
    use naga::back::wgsl;

    // ----- shared helpers ---------------------------------------------------------------------

    fn parse_wgsl(source: &str) -> Module {
        naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!("WGSL parse failed:\n{}\n{source}", e.emit_to_string(source))
        })
    }

    /// Parse and validate WGSL against the WebGPU baseline (no capabilities).
    fn validate_wgsl_baseline(source: &str) -> Module {
        let module = parse_wgsl(source);
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
        if let Err(e) = validator.validate(&module) {
            panic!(
                "baseline validation failed: {}\n{source}",
                e.emit_to_string(source)
            );
        }
        module
    }

    /// Validate a lowered module with the report's validator, write WGSL, re-parse and
    /// re-validate it; returns the WGSL and the re-parsed module.
    fn emit_and_reparse(module: &Module, report: &WaveLoweringReport) -> (String, Module) {
        let info = report
            .validator()
            .validate(module)
            .unwrap_or_else(|e| panic!("lowered module invalid: {}", e.emit_to_string("")));
        let source = wgsl::write_string(module, &info, wgsl::WriterFlags::empty())
            .expect("WGSL writer failed");
        let reparsed = parse_wgsl(&source);
        if let Err(e) = report.validator().validate(&reparsed) {
            panic!(
                "re-parsed WGSL invalid: {}\n{source}",
                e.emit_to_string(&source)
            );
        }
        (source, reparsed)
    }

    fn lower(source: &str, options: &WaveLoweringOptions) -> (Module, WaveLoweringReport) {
        let mut module = parse_wgsl(source);
        let report = lower_wave_ops(&mut module, options);
        (module, report)
    }

    fn emulation() -> WaveLoweringOptions {
        WaveLoweringOptions::new(WaveStrategy::WorkgroupEmulation)
    }

    fn single_lane(allow_wave1_model: bool) -> WaveLoweringOptions {
        WaveLoweringOptions {
            allow_wave1_model,
            ..WaveLoweringOptions::new(WaveStrategy::SingleLane)
        }
    }

    fn words_of(memory: &Memory, module: &Module, name: &str) -> Vec<u32> {
        memory
            .get(module, name)
            .comps()
            .iter()
            .map(Val::u)
            .collect()
    }

    /// Lanes of the Wave64 wave containing invocation `index` of a `total`-invocation group.
    fn wave_lanes(index: u32, total: u32) -> std::ops::Range<u32> {
        let base = index & !63;
        base..(base + 64).min(total)
    }

    fn assert_unchanged(before: &Module, after: &Module) {
        assert_eq!(
            format!("{before:?}"),
            format!("{after:?}"),
            "a blocked lowering must leave the module untouched"
        );
    }

    // ----- SPIR-V fixtures --------------------------------------------------------------------

    const OP_CAPABILITY: u16 = 17;
    const OP_MEMORY_MODEL: u16 = 14;
    const OP_TYPE_VOID: u16 = 19;
    const OP_TYPE_VECTOR: u16 = 23;
    const OP_TYPE_ARRAY: u16 = 28;
    const OP_TYPE_STRUCT: u16 = 30;
    const OP_TYPE_FUNCTION: u16 = 33;
    const OP_CONSTANT_TRUE: u16 = 41;
    const OP_CONSTANT_COMPOSITE: u16 = 44;
    const OP_STORE: u16 = 62;
    const OP_ACCESS_CHAIN: u16 = 65;
    const OP_IMUL: u16 = 132;
    const OP_UMOD: u16 = 137;
    const OP_LABEL: u16 = 248;
    const OP_RETURN: u16 = 253;
    const CAP_SHADER: u32 = 1;
    const CAP_GROUP_NON_UNIFORM: u32 = 61;
    const CAP_GROUP_NON_UNIFORM_VOTE: u32 = 62;
    const CAP_GROUP_NON_UNIFORM_BALLOT: u32 = 64;
    const CAP_MESH_SHADING_EXT: u32 = 5283;
    const EXEC_MODE_ORIGIN_UPPER_LEFT: u32 = 7;
    const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
    const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
    const DECORATION_BLOCK: u32 = 2;
    const DECORATION_ARRAY_STRIDE: u32 = 6;
    const DECORATION_BINDING: u32 = 33;
    const DECORATION_DESCRIPTOR_SET: u32 = 34;
    const DECORATION_OFFSET: u32 = 35;
    const SCOPE_SUBGROUP: u32 = 3;
    const OP_GROUP_NON_UNIFORM_ALL_EQUAL: u16 = 336;

    fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
        crate::compat::spirv_bytes_to_words(bytes).expect("aligned SPIR-V")
    }

    fn parse_spv(words: &[u32]) -> Result<Module, naga::front::spv::Error> {
        let bytes = crate::compat::spirv_words_to_bytes(words);
        naga::front::spv::parse_u8_slice(
            &bytes,
            &naga::front::spv::Options {
                adjust_coordinate_space: true,
                strict_capabilities: true,
                block_ctx_dump_prefix: None,
            },
        )
    }

    /// Compute shader storing, for invocation `i`, four words at `4 * i`:
    /// `WaveIsFirstLane()`, `WaveActiveCountBits(i % 3 == 0)`, `WavePrefixCountBits(i % 3 == 0)`
    /// and the inclusive count -- spelled with `OpGroupNonUniformElect` / `...BallotBitCount`,
    /// which naga 30 cannot parse. `predeclare_lane_index` adds a `SubgroupLocalInvocationId`
    /// input up front (listed in the entry point interface).
    fn build_ballot_count_spv(workgroup: u32, predeclare_lane_index: bool) -> Vec<u32> {
        let mut b = SpirvBuilder::new();
        let void = b.id();
        let fn_ty = b.id();
        let uint = b.id();
        let boolean = b.id();
        let v4uint = b.id();
        let consts: Vec<u32> = (0..5).map(|_| b.id()).collect();
        let c_len = b.id();
        let arr = b.id();
        let block = b.id();
        let ptr_block = b.id();
        let ptr_uint_sb = b.id();
        let ptr_in_uint = b.id();
        let buffer = b.id();
        let lid = b.id();
        let lane_var = b.id();
        let main = b.id();
        let label = b.id();
        let (c0, c1, c2, c3, c4) = (consts[0], consts[1], consts[2], consts[3], consts[4]);

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_BALLOT]);
        b.inst(OP_MEMORY_MODEL, &[0, 1]);
        let mut interface = vec![lid];
        if predeclare_lane_index {
            interface.push(lane_var);
        }
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_GL_COMPUTE, main],
            "main",
            &interface,
        );
        b.inst(
            OP_EXECUTION_MODE,
            &[main, EXEC_MODE_LOCAL_SIZE, workgroup, 1, 1],
        );
        b.inst(
            OP_DECORATE,
            &[lid, DECORATION_BUILTIN, BUILTIN_LOCAL_INVOCATION_INDEX],
        );
        if predeclare_lane_index {
            b.inst(
                OP_DECORATE,
                &[
                    lane_var,
                    DECORATION_BUILTIN,
                    BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID,
                ],
            );
        }
        b.inst(OP_DECORATE, &[arr, DECORATION_ARRAY_STRIDE, 4]);
        b.inst(OP_DECORATE, &[block, DECORATION_BLOCK]);
        b.inst(OP_MEMBER_DECORATE, &[block, 0, DECORATION_OFFSET, 0]);
        b.inst(OP_DECORATE, &[buffer, DECORATION_DESCRIPTOR_SET, 0]);
        b.inst(OP_DECORATE, &[buffer, DECORATION_BINDING, 0]);
        b.inst(OP_TYPE_VOID, &[void]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void]);
        b.inst(OP_TYPE_INT, &[uint, 32, 0]);
        b.inst(OP_TYPE_BOOL, &[boolean]);
        b.inst(OP_TYPE_VECTOR, &[v4uint, uint, 4]);
        for (value, &id) in consts.iter().enumerate() {
            b.inst(OP_CONSTANT, &[uint, id, value as u32]);
        }
        b.inst(OP_CONSTANT, &[uint, c_len, 4 * 256]);
        b.inst(OP_TYPE_ARRAY, &[arr, uint, c_len]);
        b.inst(OP_TYPE_STRUCT, &[block, arr]);
        b.inst(
            OP_TYPE_POINTER,
            &[ptr_block, STORAGE_CLASS_STORAGE_BUFFER, block],
        );
        b.inst(
            OP_TYPE_POINTER,
            &[ptr_uint_sb, STORAGE_CLASS_STORAGE_BUFFER, uint],
        );
        b.inst(OP_TYPE_POINTER, &[ptr_in_uint, STORAGE_CLASS_INPUT, uint]);
        b.inst(
            OP_VARIABLE,
            &[ptr_block, buffer, STORAGE_CLASS_STORAGE_BUFFER],
        );
        b.inst(OP_VARIABLE, &[ptr_in_uint, lid, STORAGE_CLASS_INPUT]);
        if predeclare_lane_index {
            b.inst(OP_VARIABLE, &[ptr_in_uint, lane_var, STORAGE_CLASS_INPUT]);
        }
        b.inst(OP_FUNCTION, &[void, main, 0, fn_ty]);
        b.inst(OP_LABEL, &[label]);
        let li = b.id();
        b.inst(OP_LOAD, &[uint, li, lid]);
        let base = b.id();
        b.inst(OP_IMUL, &[uint, base, li, c4]);
        let modulo = b.id();
        b.inst(OP_UMOD, &[uint, modulo, li, c3]);
        let predicate = b.id();
        b.inst(OP_IEQUAL, &[boolean, predicate, modulo, c0]);
        let elect = b.id();
        b.inst(OP_GROUP_NON_UNIFORM_ELECT, &[boolean, elect, c3]);
        let ballot = b.id();
        b.inst(
            OP_GROUP_NON_UNIFORM_BALLOT,
            &[v4uint, ballot, c3, predicate],
        );
        let elect_word = b.id();
        b.inst(OP_SELECT, &[uint, elect_word, elect, c1, c0]);
        let mut results = vec![elect_word];
        for group in [
            GROUP_OP_REDUCE,
            GROUP_OP_EXCLUSIVE_SCAN,
            GROUP_OP_INCLUSIVE_SCAN,
        ] {
            let count = b.id();
            b.inst(
                OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT,
                &[uint, count, c3, group, ballot],
            );
            results.push(count);
        }
        for (k, value) in results.into_iter().enumerate() {
            let index = b.id();
            b.inst(OP_IADD, &[uint, index, base, consts[k]]);
            let ptr = b.id();
            b.inst(OP_ACCESS_CHAIN, &[ptr_uint_sb, ptr, buffer, c0, index]);
            b.inst(OP_STORE, &[ptr, value]);
        }
        let _ = c2;
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        bytes_to_words(&b.finish_bytes())
    }

    // ----- detection --------------------------------------------------------------------------

    #[test]
    fn scan_attributes_wave_ops_to_entry_points_through_the_call_graph() {
        let mut b = SpirvBuilder::new();
        let void = b.id();
        let fn_ty = b.id();
        let uint = b.id();
        let boolean = b.id();
        let v4uint = b.id();
        let c3 = b.id();
        let c_true = b.id();
        let ptr_in_uint = b.id();
        let size_var = b.id();
        let helper = b.id();
        let cs = b.id();
        let fs = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_VOTE]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_BALLOT]);
        b.inst(OP_MEMORY_MODEL, &[0, 1]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_GL_COMPUTE, cs], "cs", &[]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_FRAGMENT, fs],
            "fs",
            &[size_var],
        );
        b.inst(OP_EXECUTION_MODE, &[cs, EXEC_MODE_LOCAL_SIZE, 64, 1, 1]);
        b.inst(OP_EXECUTION_MODE, &[fs, EXEC_MODE_ORIGIN_UPPER_LEFT]);
        b.inst(
            OP_DECORATE,
            &[size_var, DECORATION_BUILTIN, BUILTIN_SUBGROUP_SIZE],
        );
        b.inst(OP_TYPE_VOID, &[void]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void]);
        b.inst(OP_TYPE_INT, &[uint, 32, 0]);
        b.inst(OP_TYPE_BOOL, &[boolean]);
        b.inst(OP_TYPE_VECTOR, &[v4uint, uint, 4]);
        b.inst(OP_CONSTANT, &[uint, c3, SCOPE_SUBGROUP]);
        b.inst(OP_CONSTANT_TRUE, &[boolean, c_true]);
        b.inst(OP_TYPE_POINTER, &[ptr_in_uint, STORAGE_CLASS_INPUT, uint]);
        b.inst(OP_VARIABLE, &[ptr_in_uint, size_var, STORAGE_CLASS_INPUT]);
        for (function, body) in [(helper, 0), (cs, 1), (fs, 2)] {
            b.inst(OP_FUNCTION, &[void, function, 0, fn_ty]);
            let label = b.id();
            b.inst(OP_LABEL, &[label]);
            match body {
                0 => {
                    let ballot = b.id();
                    b.inst(OP_GROUP_NON_UNIFORM_BALLOT, &[v4uint, ballot, c3, c_true]);
                    let equal = b.id();
                    b.inst(OP_GROUP_NON_UNIFORM_ALL_EQUAL, &[boolean, equal, c3, c3]);
                }
                1 => {
                    let call = b.id();
                    b.inst(OP_FUNCTION_CALL, &[void, call, helper]);
                }
                _ => {
                    for _ in 0..2 {
                        let size = b.id();
                        b.inst(OP_LOAD, &[uint, size, size_var]);
                    }
                }
            }
            b.inst(OP_RETURN, &[]);
            b.inst(OP_FUNCTION_END, &[]);
        }
        let words = bytes_to_words(&b.finish_bytes());
        let report = scan_wave_usage_spirv(&words).expect("well-formed SPIR-V");

        let find = |intrinsic: WaveIntrinsic| {
            report
                .uses
                .iter()
                .find(|u| u.intrinsic == intrinsic)
                .unwrap_or_else(|| panic!("missing {intrinsic}: {report:?}"))
        };
        let lane_count = find(WaveIntrinsic::LaneCount);
        assert_eq!(lane_count.occurrences, 2);
        assert_eq!(lane_count.entry_points, vec!["fs".to_string()]);
        assert_eq!(lane_count.stages, vec![WaveShaderStage::Fragment]);
        assert_eq!(lane_count.spirv_opcode, None);
        assert_eq!(lane_count.lane_dependency(), LaneDependency::LaneLocal);
        assert!(lane_count.naga_front_end);

        let ballot = find(WaveIntrinsic::Ballot);
        assert_eq!(ballot.entry_points, vec!["cs".to_string()]);
        assert_eq!(ballot.stages, vec![WaveShaderStage::Compute]);
        assert_eq!(ballot.lane_dependency(), LaneDependency::CrossLane);
        assert!(ballot.naga_front_end && ballot.reaches_ir_lowering());
        assert_eq!(ballot.intrinsic.hlsl_name(), "WaveActiveBallot");

        let all_equal = find(WaveIntrinsic::AllEqual);
        assert!(!all_equal.naga_front_end && !all_equal.canonicalizable);
        assert!(report.has_cross_lane());

        let audit = report.to_compat_audit();
        assert_eq!(audit.blocker_count(), 1, "{audit:?}");
        assert_eq!(audit.warning_count(), 2, "{audit:?}");
        let blocker = audit.blockers().next().expect("blocker");
        assert_eq!(blocker.kind, CompatIssueKind::SubgroupWave64);
        assert!(
            blocker.message.contains("WaveActiveAllEqual"),
            "{blocker:?}"
        );
    }

    #[test]
    fn scan_reports_mesh_and_task_entry_points_with_output_limits() {
        let mut b = SpirvBuilder::new();
        let void = b.id();
        let fn_ty = b.id();
        let uint = b.id();
        let c1 = b.id();
        let ptr_payload = b.id();
        let payload = b.id();
        let ms = b.id();
        let ts = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_CAPABILITY, &[CAP_MESH_SHADING_EXT]);
        b.inst(OP_MEMORY_MODEL, &[0, 1]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_MESH_EXT, ms], "ms", &[]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_TASK_EXT, ts], "ts", &[payload]);
        b.inst(OP_EXECUTION_MODE, &[ms, EXEC_MODE_OUTPUT_VERTICES, 64]);
        b.inst(
            OP_EXECUTION_MODE,
            &[ms, EXEC_MODE_OUTPUT_PRIMITIVES_EXT, 126],
        );
        b.inst(OP_EXECUTION_MODE, &[ms, EXEC_MODE_OUTPUT_TRIANGLES_EXT]);
        b.inst(OP_EXECUTION_MODE, &[ms, EXEC_MODE_LOCAL_SIZE, 32, 1, 1]);
        b.inst(OP_EXECUTION_MODE, &[ts, EXEC_MODE_LOCAL_SIZE, 32, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void]);
        b.inst(OP_TYPE_INT, &[uint, 32, 0]);
        b.inst(OP_CONSTANT, &[uint, c1, 1]);
        b.inst(
            OP_TYPE_POINTER,
            &[ptr_payload, STORAGE_CLASS_TASK_PAYLOAD_WORKGROUP_EXT, uint],
        );
        b.inst(
            OP_VARIABLE,
            &[
                ptr_payload,
                payload,
                STORAGE_CLASS_TASK_PAYLOAD_WORKGROUP_EXT,
            ],
        );
        b.inst(OP_FUNCTION, &[void, ms, 0, fn_ty]);
        let label = b.id();
        b.inst(OP_LABEL, &[label]);
        b.inst(OP_SET_MESH_OUTPUTS_EXT, &[c1, c1]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.inst(OP_FUNCTION, &[void, ts, 0, fn_ty]);
        let label = b.id();
        b.inst(OP_LABEL, &[label]);
        b.inst(OP_EMIT_MESH_TASKS_EXT, &[c1, c1, c1, payload]);
        b.inst(OP_FUNCTION_END, &[]);

        let words = bytes_to_words(&b.finish_bytes());
        let report = scan_wave_usage_spirv(&words).expect("well-formed SPIR-V");
        assert!(report.uses.is_empty());
        assert!(report.uses_task_payload);
        assert_eq!(
            report.mesh,
            vec![
                MeshUsage {
                    entry_point: "ms".to_string(),
                    stage: WaveShaderStage::Mesh,
                    max_vertices: Some(64),
                    max_primitives: Some(126),
                    topology: Some(MeshTopology::Triangles),
                    workgroup_size: Some([32, 1, 1]),
                    sets_mesh_outputs: true,
                    emits_mesh_tasks: false,
                },
                MeshUsage {
                    entry_point: "ts".to_string(),
                    stage: WaveShaderStage::Task,
                    max_vertices: None,
                    max_primitives: None,
                    topology: None,
                    workgroup_size: Some([32, 1, 1]),
                    sets_mesh_outputs: false,
                    emits_mesh_tasks: true,
                },
            ]
        );
        let audit = report.to_compat_audit();
        assert_eq!(audit.blocker_count(), 2);
        assert!(
            audit
                .blockers()
                .all(|i| i.kind == CompatIssueKind::MeshShader)
        );
        assert!(
            audit
                .blockers()
                .any(|i| i.message.contains("max_vertices=64")
                    && i.message.contains("triangle-list"))
        );
    }

    #[test]
    fn scan_of_wave_free_shader_is_empty_and_malformed_streams_are_errors() {
        let words = bytes_to_words(&build_minimal_compute_spv());
        let report = scan_wave_usage_spirv(&words).expect("well-formed SPIR-V");
        assert!(report.is_empty() && !report.has_cross_lane());
        assert!(report.to_compat_audit().issues.is_empty());

        assert!(matches!(
            scan_wave_usage_spirv(&[0x1234_5678, 0, 0, 0, 0]),
            Err(SpirvStreamError::InvalidMagic { .. })
        ));
        // An instruction header announcing 5 words with none following.
        let mut truncated = words.clone();
        truncated.push((5 << 16) | u32::from(OP_CAPABILITY));
        assert!(scan_wave_usage_spirv(&truncated).is_err());
        assert!(canonicalize_wave_spirv(&truncated).is_err());
    }

    // ----- canonicalization -------------------------------------------------------------------

    #[test]
    fn canonicalization_is_the_identity_without_target_instructions() {
        let words = bytes_to_words(&build_minimal_compute_spv());
        let result = canonicalize_wave_spirv(&words).expect("well-formed SPIR-V");
        assert_eq!(result.words, words);
        assert!(result.rewritten.is_empty() && result.skipped.is_empty());
    }

    /// Reference Wave64 results of `build_ballot_count_spv` for invocation `i`.
    fn ballot_count_reference(i: u32, total: u32) -> [u32; 4] {
        let lanes = wave_lanes(i, total);
        let predicate = |l: &u32| l.is_multiple_of(3);
        [
            u32::from(i == lanes.start),
            lanes.clone().filter(predicate).count() as u32,
            (lanes.start..i).filter(predicate).count() as u32,
            (lanes.start..=i).filter(predicate).count() as u32,
        ]
    }

    #[test]
    fn canonicalized_elect_and_ballot_counts_parse_lower_and_execute_exactly() {
        const TOTAL: u32 = 100;
        let words = build_ballot_count_spv(TOTAL, false);
        assert!(
            parse_spv(&words).is_err(),
            "naga 30 is expected to reject OpGroupNonUniformElect / BallotBitCount"
        );
        let canonical = canonicalize_wave_spirv(&words).expect("well-formed SPIR-V");
        assert_eq!(
            canonical.rewritten,
            BTreeMap::from([
                (WaveIntrinsic::IsFirstLane, 1),
                (WaveIntrinsic::BallotBitCount(WaveGroupOp::Reduce), 1),
                (WaveIntrinsic::BallotBitCount(WaveGroupOp::InclusiveScan), 1),
                (WaveIntrinsic::BallotBitCount(WaveGroupOp::ExclusiveScan), 1),
            ])
        );
        assert!(canonical.skipped.is_empty());
        let usage = scan_wave_usage_spirv(&canonical.words).expect("well-formed SPIR-V");
        assert!(usage.uses.iter().all(|u| u.naga_front_end), "{usage:?}");
        assert!(
            usage
                .uses
                .iter()
                .any(|u| u.intrinsic == WaveIntrinsic::LaneIndex)
        );

        // Exact under workgroup emulation (wave = 64 consecutive invocations, partial last wave).
        let mut module = parse_spv(&canonical.words).expect("canonical SPIR-V parses");
        let report = lower_wave_ops(&mut module, &emulation());
        assert!(report.applied, "{:?}", report.blockers);
        assert!(
            report
                .lowered
                .iter()
                .all(|op| !op.condition.may_differ_from_wave64()),
            "{:?}",
            report.lowered
        );
        let (_, module) = emit_and_reparse(&module, &report);
        let memory = Memory::new(&module);
        run_compute(&module, &memory, "main", [1, 1, 1]);
        let out: Vec<u32> = first_global_in(
            &memory,
            &module,
            AddressSpace::Storage {
                access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
            },
        )
        .comps()[0]
            .comps()
            .iter()
            .map(Val::u)
            .collect();
        for i in 0..TOTAL {
            let got = &out[(4 * i) as usize..(4 * i + 4) as usize];
            assert_eq!(got, ballot_count_reference(i, TOTAL), "invocation {i}");
        }

        // The wave-of-1 model gives the self-consistent 1-lane answers.
        let mut module = parse_spv(&canonical.words).expect("canonical SPIR-V parses");
        let report = lower_wave_ops(&mut module, &single_lane(true));
        assert!(report.applied, "{:?}", report.blockers);
        let (_, module) = emit_and_reparse(&module, &report);
        let memory = Memory::new(&module);
        run_compute(&module, &memory, "main", [1, 1, 1]);
        let out: Vec<u32> = first_global_in(
            &memory,
            &module,
            AddressSpace::Storage {
                access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
            },
        )
        .comps()[0]
            .comps()
            .iter()
            .map(Val::u)
            .collect();
        for i in 0..TOTAL {
            let own = u32::from(i.is_multiple_of(3));
            assert_eq!(
                &out[(4 * i) as usize..(4 * i + 4) as usize],
                [1, own, 0, own]
            );
        }
    }

    #[test]
    fn canonicalization_reuses_a_declared_lane_index_and_is_deterministic() {
        let words = build_ballot_count_spv(64, true);
        let first = canonicalize_wave_spirv(&words).expect("well-formed SPIR-V");
        let second = canonicalize_wave_spirv(&words).expect("well-formed SPIR-V");
        assert_eq!(first, second);
        let instructions = spirv_instructions(&first.words).expect("well-formed output");
        let lane_decorations = instructions
            .iter()
            .filter(|inst| {
                inst_opcode(inst) == OP_DECORATE
                    && inst.get(2) == Some(&DECORATION_BUILTIN)
                    && inst.get(3) == Some(&BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID)
            })
            .count();
        assert_eq!(lane_decorations, 1, "the declared input must be reused");
        let entry = instructions
            .iter()
            .find(|inst| inst_opcode(inst) == OP_ENTRY_POINT)
            .expect("entry point");
        let original_entry = spirv_instructions(&words)
            .expect("well-formed input")
            .into_iter()
            .find(|inst| inst_opcode(inst) == OP_ENTRY_POINT)
            .expect("entry point");
        assert_eq!(*entry, original_entry, "interface already lists the input");
        parse_spv(&first.words).expect("canonical SPIR-V parses");
    }

    #[test]
    fn canonicalization_skips_bit_counts_of_values_that_are_not_ballots() {
        let mut b = SpirvBuilder::new();
        let void = b.id();
        let fn_ty = b.id();
        let uint = b.id();
        let v4uint = b.id();
        let c3 = b.id();
        let mask = b.id();
        let main = b.id();
        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_BALLOT]);
        b.inst(OP_MEMORY_MODEL, &[0, 1]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_GL_COMPUTE, main], "main", &[]);
        b.inst(OP_EXECUTION_MODE, &[main, EXEC_MODE_LOCAL_SIZE, 64, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void]);
        b.inst(OP_TYPE_INT, &[uint, 32, 0]);
        b.inst(OP_TYPE_VECTOR, &[v4uint, uint, 4]);
        b.inst(OP_CONSTANT, &[uint, c3, SCOPE_SUBGROUP]);
        b.inst(OP_CONSTANT_COMPOSITE, &[v4uint, mask, c3, c3, c3, c3]);
        b.inst(OP_FUNCTION, &[void, main, 0, fn_ty]);
        let label = b.id();
        b.inst(OP_LABEL, &[label]);
        let count = b.id();
        b.inst(
            OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT,
            &[uint, count, c3, GROUP_OP_REDUCE, mask],
        );
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        let words = bytes_to_words(&b.finish_bytes());

        let result = canonicalize_wave_spirv(&words).expect("well-formed SPIR-V");
        assert!(result.rewritten.is_empty());
        assert_eq!(result.skipped.len(), 1);
        assert!(result.skipped[0].contains("not an OpGroupNonUniformBallot result"));
        assert!(
            parse_spv(&result.words).is_err(),
            "the instruction must stay a blocker"
        );
    }

    // ----- naga IR lowering -------------------------------------------------------------------

    const PROBE_SLOTS: u32 = 26;

    /// Compute shader exercising every emulated wave operation; `WG` is the workgroup size.
    fn emulation_probe(workgroup: u32) -> String {
        r#"
struct Params {
    value: u32,
    flag: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<u32, OUT_LEN>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(WG)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) size: u32,
) {
    let x = (li * 2654435761u) >> 20u;
    let o = li * SLOTSu;
    out[o + 0u] = subgroupAdd(x);
    out[o + 1u] = subgroupExclusiveAdd(x);
    out[o + 2u] = subgroupInclusiveAdd(x);
    out[o + 3u] = subgroupMul(x | 1u);
    out[o + 4u] = subgroupMin(x);
    out[o + 5u] = subgroupMax(x);
    out[o + 6u] = subgroupAnd(x | 0xf0u);
    out[o + 7u] = subgroupOr(x);
    out[o + 8u] = subgroupXor(x);
    let ballot = subgroupBallot((x & 3u) == 0u);
    out[o + 9u] = ballot.x;
    out[o + 10u] = ballot.y;
    out[o + 11u] = select(0u, 1u, subgroupAll(x < 3000u));
    out[o + 12u] = select(0u, 1u, subgroupAny(x == 7u));
    out[o + 13u] = subgroupBroadcastFirst(x);
    out[o + 14u] = subgroupBroadcast(x, 37u);
    out[o + 15u] = subgroupShuffle(x, (lane * 5u + 3u) & 63u);
    out[o + 16u] = subgroupShuffleXor(x, 5u);
    out[o + 17u] = subgroupShuffleUp(x, 3u);
    out[o + 18u] = subgroupShuffleDown(x, 2u);
    out[o + 19u] = lane;
    out[o + 20u] = size;
    out[o + 21u] = bitcast<u32>(subgroupAdd(f32(x) * 0.5));
    out[o + 22u] = bitcast<u32>(subgroupMin(i32(x) - 2000));
    let v = subgroupInclusiveAdd(vec2<u32>(x, 1u));
    out[o + 23u] = v.x ^ (v.y << 16u);
    out[o + 24u] = subgroupMax(params.value);
    if (params.flag != 0u) {
        out[o + 25u] = subgroupOr(x);
    }
}
"#
        .replace("OUT_LEN", &(workgroup * PROBE_SLOTS).to_string())
        .replace("SLOTS", &PROBE_SLOTS.to_string())
        .replace("WG", &workgroup.to_string())
    }

    fn probe_value(index: u32) -> u32 {
        index.wrapping_mul(2_654_435_761) >> 20
    }

    /// Reference Wave64 result of every probe slot for invocation `i`.
    fn probe_reference(i: u32, total: u32, uniform_value: u32) -> Vec<u32> {
        let lanes = wave_lanes(i, total);
        let lane = i - lanes.start;
        let xs: Vec<u32> = lanes.clone().map(probe_value).collect();
        let own = xs[lane as usize];
        let read = |source: u32| -> Option<u32> {
            (source < 64 && lanes.start + source < lanes.end).then(|| xs[source as usize])
        };
        let sum = |values: &[u32]| values.iter().fold(0u32, |a, v| a.wrapping_add(*v));
        let ballot: u64 = xs
            .iter()
            .enumerate()
            .filter(|(_, v)| *v & 3 == 0)
            .fold(0, |mask, (l, _)| mask | (1u64 << l));
        let mut float_sum = xs[0] as f32 * 0.5;
        for v in &xs[1..] {
            float_sum += *v as f32 * 0.5;
        }
        let inclusive = sum(&xs[..=lane as usize]);
        vec![
            sum(&xs),
            sum(&xs[..lane as usize]),
            inclusive,
            xs.iter().fold(1u32, |a, v| a.wrapping_mul(v | 1)),
            *xs.iter().min().expect("non-empty wave"),
            *xs.iter().max().expect("non-empty wave"),
            xs.iter().fold(u32::MAX, |a, v| a & (v | 0xf0)),
            xs.iter().fold(0, |a, v| a | v),
            xs.iter().fold(0, |a, v| a ^ v),
            ballot as u32,
            (ballot >> 32) as u32,
            u32::from(xs.iter().all(|v| *v < 3000)),
            u32::from(xs.contains(&7)),
            xs[0],
            read(37).unwrap_or(own),
            read((lane * 5 + 3) & 63).unwrap_or(own),
            read(lane ^ 5).unwrap_or(own),
            lane.checked_sub(3).and_then(read).unwrap_or(own),
            read(lane + 2).unwrap_or(own),
            lane,
            WAVE64_LANE_COUNT,
            float_sum.to_bits(),
            xs.iter()
                .map(|v| *v as i32 - 2000)
                .min()
                .expect("non-empty") as u32,
            inclusive ^ ((lane + 1) << 16),
            uniform_value,
            xs.iter().fold(0, |a, v| a | v),
        ]
    }

    #[test]
    fn workgroup_emulation_executes_every_wave_op_exactly_for_full_and_partial_waves() {
        for total in [128u32, 100] {
            let (module, report) = lower(&emulation_probe(total), &emulation());
            assert!(report.applied, "{:?}", report.blockers);
            assert!(!report.requires_subgroups_feature);
            assert_eq!(report.wgsl_preamble(), "");
            let conditions: BTreeMap<LoweringCondition, usize> =
                report.lowered.iter().fold(BTreeMap::new(), |mut m, op| {
                    *m.entry(op.condition).or_default() += 1;
                    m
                });
            assert_eq!(conditions[&LoweringCondition::UniformOperand], 1);
            assert_eq!(
                conditions[&LoweringCondition::Exact],
                2,
                "lane index + lane count"
            );
            assert_eq!(
                conditions[&LoweringCondition::WorkgroupUniformControlFlow],
                22,
                "{:?}",
                report.lowered
            );
            assert!(
                report
                    .helper_functions
                    .iter()
                    .all(|name| name.starts_with(DEFAULT_HELPER_PREFIX))
            );
            // u32/bool share one scratch array; f32, i32 and vec2<u32> get their own.
            assert_eq!(report.workgroup_scratch_bytes, total * (4 + 4 + 4 + 8));

            let (source, module) = emit_and_reparse(&module, &report);
            assert!(!source.contains("subgroup"), "{source}");
            assert!(source.contains("workgroupBarrier()"));

            let memory = Memory::new(&module);
            memory.set(&module, "params", Val::C(vec![Val::U(0xabcd), Val::U(1)]));
            run_compute(&module, &memory, "main", [1, 1, 1]);
            let out = words_of(&memory, &module, "out");
            for i in 0..total {
                let start = (i * PROBE_SLOTS) as usize;
                let got = &out[start..start + PROBE_SLOTS as usize];
                let expected = probe_reference(i, total, 0xabcd);
                for (slot, (g, e)) in got.iter().zip(&expected).enumerate() {
                    assert_eq!(g, e, "workgroup {total}, invocation {i}, slot {slot}");
                }
            }
        }
    }

    #[test]
    fn single_lane_is_exact_for_uniform_operands_and_otherwise_needs_the_wave1_model() {
        let source = r#"
struct Params {
    value: u32,
    flag: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<u32, 40>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(4)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) size: u32,
) {
    let x = li + 1u;
    let o = li * 10u;
    out[o + 0u] = subgroupAdd(x);
    out[o + 1u] = subgroupExclusiveAdd(x);
    out[o + 2u] = subgroupExclusiveMul(x);
    let ballot = subgroupBallot(li == 2u);
    out[o + 3u] = ballot.x;
    out[o + 4u] = ballot.y;
    out[o + 5u] = lane;
    out[o + 6u] = size;
    out[o + 7u] = subgroupBroadcastFirst(x);
    out[o + 8u] = subgroupMax(params.value);
    out[o + 9u] = select(0u, 1u, subgroupAll(li < 2u));
}
"#;
        // Without the approximation only the uniform-operand use could be lowered.
        let original = parse_wgsl(source);
        let (module, report) = lower(source, &single_lane(false));
        assert!(!report.applied);
        assert_unchanged(&original, &module);
        assert_eq!(report.blockers.len(), 8, "{:?}", report.blockers);
        assert!(
            report
                .blockers
                .iter()
                .any(|b| b.reason.contains("allow_wave1_model"))
        );
        assert_eq!(
            report.lowered.len(),
            1,
            "the uniform-operand use is lowerable"
        );
        assert_eq!(
            report.lowered[0].condition,
            LoweringCondition::UniformOperand
        );
        let audit = report.to_compat_audit();
        assert_eq!(audit.blocker_count(), 8);

        let (module, report) = lower(source, &single_lane(true));
        assert!(report.applied, "{:?}", report.blockers);
        assert!(
            report
                .lowered
                .iter()
                .filter(|op| op.condition != LoweringCondition::UniformOperand)
                .all(|op| op.condition == LoweringCondition::Wave1Model)
        );
        assert!(report.to_compat_audit().warning_count() > 0);
        let (source, module) = emit_and_reparse(&module, &report);
        assert!(!source.contains("subgroup") && !source.contains("workgroupBarrier"));
        let memory = Memory::new(&module);
        memory.set(&module, "params", Val::C(vec![Val::U(77), Val::U(0)]));
        run_compute(&module, &memory, "main", [1, 1, 1]);
        let out = words_of(&memory, &module, "out");
        for li in 0..4u32 {
            let x = li + 1;
            let start = (li * 10) as usize;
            assert_eq!(
                &out[start..start + 10],
                [
                    x,
                    0,
                    1,
                    u32::from(li == 2),
                    0,
                    0,
                    1,
                    x,
                    77,
                    u32::from(li < 2)
                ],
                "invocation {li}"
            );
        }
    }

    #[test]
    fn native_strategy_keeps_subgroup_ops_and_requires_the_feature() {
        let (module, report) = lower(
            &emulation_probe(64),
            &WaveLoweringOptions::new(WaveStrategy::NativeSubgroups),
        );
        assert!(report.applied, "{:?}", report.blockers);
        assert!(report.requires_subgroups_feature);
        assert_eq!(report.wgsl_preamble(), WGSL_SUBGROUPS_ENABLE);
        assert!(
            report
                .lowered
                .iter()
                .filter(|op| op.condition != LoweringCondition::UniformOperand)
                .all(|op| op.condition == LoweringCondition::RuntimeSubgroupSize64)
        );
        // naga's WGSL front end parses subgroup built-ins but rejects `enable subgroups;`, so
        // the body is re-validated without the directive.
        let (source, _) = emit_and_reparse(&module, &report);
        assert!(source.contains("subgroupAdd(") && source.contains("subgroupBallot("));
        assert!(report.validator().validate(&module).is_ok());
        assert!(
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .is_err(),
            "the native module must not pass baseline validation"
        );

        let vertex = r#"
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let s = subgroupAdd(f32(vi));
    return vec4<f32>(s, 0.0, 0.0, 1.0);
}
"#;
        let original = parse_wgsl(vertex);
        let (module, report) = lower(
            vertex,
            &WaveLoweringOptions::new(WaveStrategy::NativeSubgroups),
        );
        assert!(!report.applied);
        assert!(report.blockers[0].reason.contains("compute and fragment"));
        assert_unchanged(&original, &module);
    }

    #[test]
    fn quad_ops_are_exact_natively_in_fragment_shaders_and_blocked_elsewhere() {
        let fragment = r#"
@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let q = quadSwapX(p.x);
    return vec4<f32>(q, 0.0, 0.0, 1.0);
}
"#;
        let (module, report) = lower(
            fragment,
            &WaveLoweringOptions::new(WaveStrategy::NativeSubgroups),
        );
        assert!(report.applied);
        assert_eq!(report.lowered[0].intrinsic, WaveIntrinsic::QuadSwap);
        assert_eq!(report.lowered[0].condition, LoweringCondition::Exact);
        emit_and_reparse(&module, &report);

        let (_, report) = lower(fragment, &emulation());
        assert!(report.blockers[0].reason.contains("compute stage"));
        let (_, report) = lower(fragment, &single_lane(true));
        assert!(report.blockers[0].reason.contains("quad"));
    }

    fn compute_with_body(body: &str) -> String {
        format!(
            r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32, 64>;

fn wave_sum(v: u32) -> u32 {{
    return subgroupAdd(v);
}}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {{
{body}
}}
"#
        )
    }

    #[test]
    fn workgroup_emulation_blocks_potentially_non_uniform_control_flow() {
        let cases = [
            "    if (li < 10u) { out[li] = subgroupAdd(li); }",
            "    for (var i = 0u; i < 2u; i = i + 1u) { out[li] = subgroupAdd(li + i); }",
            "    if (li == 3u) { return; }\n    out[li] = subgroupAdd(li);",
            "    for (var i = 0u; i < 2u; i = i + 1u) { out[li] = wave_sum(li + i); }",
        ];
        for body in cases {
            let source = compute_with_body(body);
            let original = parse_wgsl(&source);
            let (module, report) = lower(&source, &emulation());
            assert!(!report.applied, "{body}");
            assert!(
                report
                    .blockers
                    .iter()
                    .any(|b| b.reason.contains("non-uniform control flow")),
                "{body}: {:?}",
                report.blockers
            );
            assert_unchanged(&original, &module);
        }
    }

    #[test]
    fn workgroup_emulation_lowers_calls_from_uniform_positions_with_helpers_first() {
        let source = compute_with_body("    out[li] = wave_sum(li);");
        let (module, report) = lower(&source, &emulation());
        assert!(report.applied, "{:?}", report.blockers);
        let first = module.functions.iter().next().map(|(_, f)| f.name.clone());
        assert_eq!(
            first.flatten().as_deref(),
            Some("pg6w_wave_add_reduce_u32"),
            "helpers must precede their callers"
        );
        let (_, module) = emit_and_reparse(&module, &report);
        let memory = Memory::new(&module);
        run_compute(&module, &memory, "main", [1, 1, 1]);
        assert!(words_of(&memory, &module, "out").iter().all(|&v| v == 2016));
    }

    #[test]
    fn workgroup_emulation_reports_overrides_budget_and_missing_identities() {
        let overridden = r#"
override wg: u32 = 64u;
@group(0) @binding(0) var<storage, read_write> out: array<u32, 64>;

@compute @workgroup_size(wg)
fn main(@builtin(local_invocation_index) li: u32) {
    out[li] = subgroupAdd(li);
}
"#;
        let (_, report) = lower(overridden, &emulation());
        assert!(
            report.blockers[0].reason.contains("pipeline-overridable"),
            "{:?}",
            report.blockers
        );

        let source = compute_with_body("    out[li] = subgroupAdd(li);");
        let tight = WaveLoweringOptions {
            max_workgroup_storage_bytes: 64,
            ..emulation()
        };
        let original = parse_wgsl(&source);
        let (module, report) = lower(&source, &tight);
        assert!(report.blockers.iter().any(|b| b.reason.contains("budget")));
        assert_unchanged(&original, &module);

        // A min exclusive scan (legal SPIR-V, not expressible in WGSL) is rejected by naga's
        // validator before planning; the module-level blocker keeps the module untouched.
        let source =
            compute_with_body("    out[li] = bitcast<u32>(subgroupExclusiveAdd(f32(li)));");
        let mut module = parse_wgsl(&source);
        for statement in module.entry_points[0].function.body.iter_mut() {
            if let Statement::SubgroupCollectiveOperation { op, .. } = statement {
                *op = SubgroupOperation::Min;
            }
        }
        let original = module.clone();
        let report = lower_wave_ops(&mut module, &emulation());
        assert!(
            report.blockers[0].reason.contains("does not validate"),
            "{:?}",
            report.blockers
        );
        assert_unchanged(&original, &module);

        // 16-bit operands: no emulation scratch type and no representable wave-of-1 identity.
        let half = format!(
            "enable f16;\n{}",
            compute_with_body("    out[li] = u32(subgroupExclusiveAdd(f16(li)));")
        );
        let (_, report) = lower(&half, &emulation());
        assert!(report.blockers.iter().any(|b| b.reason.contains("32-bit")));
        let (_, report) = lower(&half, &single_lane(true));
        assert!(
            report
                .blockers
                .iter()
                .any(|b| b.reason.contains("identity")),
            "{:?}",
            report.blockers
        );
    }

    #[test]
    fn subgroup_builtins_follow_the_strategy_and_stage_rules() {
        let fragment = r#"
@fragment
fn fs(@builtin(subgroup_size) size: u32) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(size), 0.0, 0.0, 1.0);
}
"#;
        let (module, report) = lower(fragment, &emulation());
        assert!(report.applied, "{:?}", report.blockers);
        assert_eq!(report.lowered[0].intrinsic, WaveIntrinsic::LaneCount);
        assert_eq!(report.lowered[0].condition, LoweringCondition::Exact);
        let (source, _) = emit_and_reparse(&module, &report);
        assert!(!source.contains("subgroup_size") && source.contains("64"));

        let lane_index = r#"
@fragment
fn fs(@builtin(subgroup_invocation_id) lane: u32) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(lane), 0.0, 0.0, 1.0);
}
"#;
        let (_, report) = lower(lane_index, &emulation());
        assert!(report.blockers[0].reason.contains("compute stage"));
        let (module, report) = lower(lane_index, &single_lane(true));
        assert!(report.applied);
        emit_and_reparse(&module, &report);

        let waves = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32, 200>;

@compute @workgroup_size(100)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(num_subgroups) waves: u32,
    @builtin(subgroup_id) wave: u32,
) {
    out[li * 2u] = waves;
    out[li * 2u + 1u] = wave;
}
"#;
        let (module, report) = lower(waves, &emulation());
        assert!(report.applied, "{:?}", report.blockers);
        let (_, module) = emit_and_reparse(&module, &report);
        let memory = Memory::new(&module);
        run_compute(&module, &memory, "main", [1, 1, 1]);
        let out = words_of(&memory, &module, "out");
        for li in 0..100u32 {
            assert_eq!(out[(li * 2) as usize], 2);
            assert_eq!(out[(li * 2 + 1) as usize], li / 64);
        }
        let (_, report) = lower(waves, &single_lane(true));
        assert_eq!(report.blockers.len(), 2);
    }

    #[test]
    fn unreachable_wave_code_is_lowered_exactly_and_wave_free_modules_are_untouched() {
        let source = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32, 4>;

fn unused(v: u32) -> u32 {
    return subgroupAdd(v);
}

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) li: u32) {
    out[li] = li;
}
"#;
        let (module, report) = lower(source, &single_lane(false));
        assert!(report.applied, "{:?}", report.blockers);
        assert_eq!(report.lowered[0].condition, LoweringCondition::Exact);
        assert!(report.lowered[0].location.contains("unreachable"));
        emit_and_reparse(&module, &report);

        let plain = "@compute @workgroup_size(1)\nfn main() {}\n";
        let original = parse_wgsl(plain);
        let (module, report) = lower(plain, &emulation());
        assert!(report.is_empty() && !report.applied && !report.has_blockers());
        assert_unchanged(&original, &module);
    }

    #[test]
    fn lowering_is_deterministic_and_avoids_prefix_collisions() {
        let probe = emulation_probe(64);
        let emit = |source: &str| {
            let (module, report) = lower(source, &emulation());
            emit_and_reparse(&module, &report).0
        };
        assert_eq!(emit(&probe), emit(&probe));

        let colliding = compute_with_body("    out[li] = pg6w_user(li) + subgroupAdd(li);")
            .replace(
                "fn wave_sum",
                "fn pg6w_user(v: u32) -> u32 {\n    return v;\n}\n\nfn wave_sum",
            );
        let (module, report) = lower(&colliding, &emulation());
        assert!(report.applied, "{:?}", report.blockers);
        assert!(
            report
                .helper_functions
                .iter()
                .all(|name| name.starts_with("pg6w1_")),
            "{:?}",
            report.helper_functions
        );
        assert!(
            report
                .lowered
                .iter()
                .filter_map(|op| op.helper.as_deref())
                .all(|name| name.starts_with("pg6w1_"))
        );
        let (source, _) = emit_and_reparse(&module, &report);
        assert!(source.contains("fn pg6w_user("));
    }

    // ----- 64-bit mask helpers ----------------------------------------------------------------

    #[test]
    fn u64_helpers_validate_in_every_combination_and_only_requested_ones_are_emitted() {
        for mask in 0u32..(1 << U64Helper::ALL.len()) {
            let selected: Vec<U64Helper> = U64Helper::ALL
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, h)| *h)
                .collect();
            let source = u64_helpers_wgsl("pg6w_", &selected);
            assert_eq!(source.matches("fn ").count(), selected.len());
            for helper in U64Helper::ALL {
                let declared = source.contains(&format!("fn {}(", helper.function_name("pg6w_")));
                assert_eq!(declared, selected.contains(&helper), "mask {mask:#x}");
            }
            validate_wgsl_baseline(&source);
        }
        use U64Helper::{And, Shl};
        assert_eq!(
            u64_helpers_wgsl("p_", &[Shl, And, Shl]),
            u64_helpers_wgsl("p_", &[And, Shl])
        );
    }

    fn u64_samples() -> Vec<u64> {
        let mut values = vec![
            0,
            1,
            2,
            0x8000_0000,
            0xffff_ffff,
            0x1_0000_0000,
            0x1_ffff_ffff,
            0x8000_0000_0000_0000,
            0x7fff_ffff_ffff_ffff,
            u64::MAX,
            0x0123_4567_89ab_cdef,
            0xfedc_ba98_7654_3210,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..24 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            values.push(state);
        }
        values
    }

    #[test]
    fn u64_helpers_execute_like_native_u64_on_edge_cases() {
        let module = validate_wgsl_baseline(&u64_helpers_wgsl("t_", &U64Helper::ALL));
        let call = |helper: U64Helper, args: Vec<Val>| {
            call_function(&module, &helper.function_name("t_"), args)
        };
        let pair = Val::u64_pair;
        let values = u64_samples();
        let amounts: Vec<u32> = (0..=70)
            .chain([95, 96, 127, 128, 1000, u32::MAX - 1, u32::MAX])
            .collect();

        for &a in &values {
            assert_eq!(call(U64Helper::Not, vec![pair(a)]).to_u64(), !a);
            assert_eq!(
                call(U64Helper::CountOneBits, vec![pair(a)]).u(),
                a.count_ones()
            );
            let trailing = if a == 0 { u32::MAX } else { a.trailing_zeros() };
            assert_eq!(
                call(U64Helper::FirstTrailingBit, vec![pair(a)]).u(),
                trailing
            );
            let leading = if a == 0 {
                u32::MAX
            } else {
                63 - a.leading_zeros()
            };
            assert_eq!(call(U64Helper::FirstLeadingBit, vec![pair(a)]).u(), leading);
            for &n in &amounts {
                let shl = if n >= 64 { 0 } else { a << n };
                let shr = if n >= 64 { 0 } else { a >> n };
                assert_eq!(
                    call(U64Helper::Shl, vec![pair(a), Val::U(n)]).to_u64(),
                    shl,
                    "{a:#x} << {n}"
                );
                assert_eq!(
                    call(U64Helper::Shr, vec![pair(a), Val::U(n)]).to_u64(),
                    shr,
                    "{a:#x} >> {n}"
                );
                assert_eq!(
                    call(U64Helper::TestBit, vec![pair(a), Val::U(n)]).b(),
                    n < 64 && (a >> n) & 1 == 1,
                    "bit {n} of {a:#x}"
                );
            }
            for &b in &values {
                let ab = || vec![pair(a), pair(b)];
                assert_eq!(call(U64Helper::And, ab()).to_u64(), a & b);
                assert_eq!(call(U64Helper::Or, ab()).to_u64(), a | b);
                assert_eq!(call(U64Helper::Xor, ab()).to_u64(), a ^ b);
                assert_eq!(call(U64Helper::Equal, ab()).b(), a == b);
                assert_eq!(
                    call(U64Helper::LessThan, ab()).b(),
                    a < b,
                    "{a:#x} < {b:#x}"
                );
                assert_eq!(
                    call(U64Helper::Add, ab()).to_u64(),
                    a.wrapping_add(b),
                    "{a:#x} + {b:#x}"
                );
            }
        }
        for &n in &amounts {
            let expected = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
            assert_eq!(
                call(U64Helper::LaneMask, vec![Val::U(n)]).to_u64(),
                expected,
                "lane mask {n}"
            );
        }
    }

    #[test]
    fn u64_helpers_inject_without_collisions_using_a_unique_prefix() {
        let existing = "fn pg6w_u64_and(a: u32) -> u32 {\n    return a;\n}\n\n@compute @workgroup_size(1)\nfn main() {\n    _ = pg6w_u64_and(1u);\n}\n";
        let prefix = unique_wgsl_prefix(existing, DEFAULT_HELPER_PREFIX);
        assert_eq!(prefix, "pg6w1_");
        let combined = format!("{existing}\n{}", u64_helpers_wgsl(&prefix, &U64Helper::ALL));
        validate_wgsl_baseline(&combined);
        let clash = format!(
            "{existing}\n{}",
            u64_helpers_wgsl(DEFAULT_HELPER_PREFIX, &[U64Helper::And])
        );
        assert!(naga::front::wgsl::parse_str(&clash).is_err());
        assert_eq!(unique_wgsl_prefix("fn main() {}", "pg6w_"), "pg6w_");
    }

    // ----- mesh-compute helpers ---------------------------------------------------------------

    fn sample_signature(topology: MeshTopology, primitives: bool) -> MeshOutputSignature {
        use MeshAttributeFormat as F;
        MeshOutputSignature {
            max_vertices: 4,
            max_primitives: 2,
            topology,
            vertex_attributes: vec![
                MeshAttribute::new("position", F::Vec4F32),
                MeshAttribute::new("normal", F::Vec3F32),
                MeshAttribute::new("uv", F::Vec2F32),
                MeshAttribute::new("id", F::U32),
            ],
            primitive_attributes: if primitives {
                vec![
                    MeshAttribute::new("material", F::U32),
                    MeshAttribute::new("flags", F::Vec2U32),
                ]
            } else {
                Vec::new()
            },
            threads_per_meshlet: [4, 1, 1],
        }
    }

    fn struct_layout(module: &Module, name: &str) -> (Vec<u32>, u32) {
        let (_, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no struct {name}"));
        match &ty.inner {
            TypeInner::Struct { members, span } => {
                (members.iter().map(|m| m.offset).collect(), *span)
            }
            other => panic!("{name} is {other:?}"),
        }
    }

    #[test]
    fn mesh_output_layout_matches_naga_wgsl_layout_rules() {
        let signature = sample_signature(MeshTopology::Triangles, true);
        let layout = mesh_output_layout(&signature).expect("valid signature");
        let offsets = |l: &MeshStructLayout| l.fields.iter().map(|f| f.offset).collect::<Vec<_>>();
        assert_eq!(offsets(&layout.vertex), vec![0, 16, 32, 40]);
        assert_eq!((layout.vertex.size, layout.vertex.align), (48, 16));
        let primitive = layout.primitive.as_ref().expect("primitive layout");
        assert_eq!(offsets(primitive), vec![0, 8]);
        assert_eq!(primitive.size, 16);

        let source = mesh_output_wgsl(&signature, &MeshBindings::default(), "pg6m_")
            .expect("valid signature");
        let module = validate_wgsl_baseline(&source);
        assert_eq!(
            struct_layout(&module, "pg6m_Vertex"),
            (offsets(&layout.vertex), layout.vertex.size)
        );
        assert_eq!(
            struct_layout(&module, "pg6m_Primitive"),
            (offsets(primitive), primitive.size)
        );
        assert_eq!(
            struct_layout(&module, "pg6m_DrawIndexedIndirectArgs"),
            (vec![0, 4, 8, 12, 16], DRAW_INDEXED_INDIRECT_ARGS_BYTES)
        );
        // Field names/order shared with web/workers/indirect_draw.js.
        assert!(source.contains(
            "index_count: atomic<u32>,\n    instance_count: u32,\n    first_index: u32,\n    base_vertex: i32,\n    first_instance: u32,"
        ));

        assert_eq!(
            layout.vertex.vertex_formats(),
            vec![
                ("float32x4", 0, 0),
                ("float32x3", 16, 1),
                ("float32x2", 32, 2),
                ("uint32", 40, 3)
            ]
        );
        assert_eq!(layout.vertex_buffer_bytes(3), 3 * 4 * 48);
        assert_eq!(layout.index_buffer_bytes(3), 3 * 2 * 3 * 4);
        assert_eq!(layout.primitive_buffer_bytes(3), 3 * 2 * 16);
        assert_eq!(layout.vertex_slot(2, 1), 9);
        assert_eq!(layout.topology.webgpu_topology(), "triangle-list");
    }

    #[test]
    fn mesh_output_wgsl_validates_for_every_topology_and_option() {
        for topology in [
            MeshTopology::Points,
            MeshTopology::Lines,
            MeshTopology::Triangles,
        ] {
            for primitives in [false, true] {
                for amplification in [None, Some((4, 5))] {
                    let signature = sample_signature(topology, primitives);
                    let bindings = MeshBindings {
                        group: 2,
                        amplification,
                        ..MeshBindings::default()
                    };
                    let source =
                        mesh_output_wgsl(&signature, &bindings, "m_").expect("valid signature");
                    validate_wgsl_baseline(&source);
                    let writer = match topology {
                        MeshTopology::Points => "fn m_write_point(",
                        MeshTopology::Lines => "fn m_write_line(",
                        MeshTopology::Triangles => "fn m_write_triangle(",
                    };
                    assert!(source.contains(writer));
                    assert_eq!(source.contains("fn m_write_primitive("), primitives);
                    assert_eq!(
                        source.contains("fn m_emit_mesh_tasks("),
                        amplification.is_some()
                    );
                    assert!(source.contains("@group(2) @binding(0)"));
                }
            }
        }
    }

    #[test]
    fn mesh_signature_and_binding_errors_are_reported() {
        let valid = sample_signature(MeshTopology::Triangles, true);
        let with = |f: &dyn Fn(&mut MeshOutputSignature)| {
            let mut s = valid.clone();
            f(&mut s);
            mesh_output_layout(&s)
        };
        assert!(matches!(
            with(&|s| s.max_vertices = 0),
            Err(MeshLoweringError::OutOfRange {
                what: "max_vertices",
                ..
            })
        ));
        assert!(matches!(
            with(&|s| s.max_primitives = 257),
            Err(MeshLoweringError::OutOfRange {
                what: "max_primitives",
                ..
            })
        ));
        assert!(matches!(
            with(&|s| s.threads_per_meshlet = [16, 16, 1]),
            Err(MeshLoweringError::OutOfRange {
                what: "threads_per_meshlet",
                ..
            })
        ));
        assert_eq!(
            with(&|s| s.vertex_attributes.clear()),
            Err(MeshLoweringError::NoVertexAttributes)
        );
        assert_eq!(
            with(&|s| s.vertex_attributes[1].name = "1st".to_string()),
            Err(MeshLoweringError::InvalidAttributeName("1st".to_string()))
        );
        assert_eq!(
            with(&|s| s.primitive_attributes[1].name = "material".to_string()),
            Err(MeshLoweringError::DuplicateAttribute(
                "material".to_string()
            ))
        );
        let clash = MeshBindings {
            draw_args: 0,
            ..MeshBindings::default()
        };
        assert_eq!(
            mesh_output_wgsl(&valid, &clash, "m_"),
            Err(MeshLoweringError::DuplicateBinding(0))
        );
        assert_eq!(
            mesh_output_wgsl(&valid, &MeshBindings::default(), "9x_"),
            Err(MeshLoweringError::InvalidPrefix("9x_".to_string()))
        );
    }

    #[test]
    fn mesh_dispatch_grid_respects_webgpu_dimension_limits() {
        let max = u64::from(WEBGPU_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION);
        assert_eq!(mesh_dispatch_grid(0), Ok([0, 1, 1]));
        assert_eq!(mesh_dispatch_grid(7), Ok([7, 1, 1]));
        assert_eq!(mesh_dispatch_grid(max), Ok([65535, 1, 1]));
        assert_eq!(mesh_dispatch_grid(max + 1), Ok([65535, 2, 1]));
        assert_eq!(mesh_dispatch_grid(max * max + 1), Ok([65535, 65535, 2]));
        for meshlets in [1, 1000, max + 1, 4_194_303, max * max + 1] {
            let [x, y, z] = mesh_dispatch_grid(meshlets).expect("within limits");
            let groups = u64::from(x) * u64::from(y) * u64::from(z);
            assert!(groups >= meshlets && groups - meshlets < u64::from(x) * u64::from(y));
        }
        assert!(mesh_dispatch_grid(max * max * max + 1).is_err());
    }

    #[test]
    fn mesh_compute_helpers_compact_cull_and_emit_draw_arguments_when_executed() {
        let signature = sample_signature(MeshTopology::Triangles, true);
        let mut source = mesh_output_wgsl(&signature, &MeshBindings::default(), "pg6m_")
            .expect("valid signature");
        source.push_str(
            r#"
@compute @workgroup_size(4)
fn mesh_main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let meshlet = pg6m_meshlet_index(wg, nwg, 3u);
    if (meshlet == 0xffffffffu) {
        return;
    }
    let primitive_count = select(2u, 0u, meshlet == 1u);
    let base = pg6m_allocate_primitives(li, primitive_count);
    pg6m_write_vertex(meshlet, li, pg6m_Vertex(vec4<f32>(f32(meshlet), f32(li), 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), vec2<f32>(0.5, 0.5), meshlet * 4u + li));
    if (li < 2u) {
        pg6m_write_triangle(meshlet, base, li, primitive_count, vec3<u32>(0u, li + 1u, li + 2u), meshlet == 2u && li == 1u);
        pg6m_write_primitive(base, li, primitive_count, pg6m_Primitive(meshlet * 10u + li, vec2<u32>(li, meshlet)));
    }
}
"#,
        );
        let module = validate_wgsl_baseline(&source);
        let memory = Memory::new(&module);
        let reset: Vec<Val> = DRAW_INDEXED_INDIRECT_RESET
            .iter()
            .enumerate()
            .map(|(i, &w)| if i == 3 { Val::I(w as i32) } else { Val::U(w) })
            .collect();
        memory.set(&module, "pg6m_draw_args", Val::C(reset));
        // 4 groups for 3 meshlets: the padding group must not allocate or write anything.
        run_compute(&module, &memory, "mesh_main", [4, 1, 1]);

        let draw = memory.get(&module, "pg6m_draw_args");
        let draw: Vec<u32> = draw.comps().iter().map(Val::u).collect();
        assert_eq!(draw, vec![12, 1, 0, 0, 0]);
        let indices = words_of(&memory, &module, "pg6m_indices");
        assert_eq!(&indices[..12], [0, 1, 2, 0, 2, 3, 8, 9, 10, 8, 8, 8]);
        assert!(indices[12..].iter().all(|&i| i == 0));
        let primitives = memory.get(&module, "pg6m_primitives");
        let materials: Vec<u32> = primitives.comps()[..4]
            .iter()
            .map(|p| p.comps()[0].u())
            .collect();
        assert_eq!(materials, vec![0, 1, 20, 21]);
        let vertices = memory.get(&module, "pg6m_vertices");
        for (slot, vertex) in vertices.comps()[..12].iter().enumerate() {
            assert_eq!(vertex.comps()[3].u(), slot as u32, "vertex slot {slot}");
        }
    }

    #[test]
    fn amplification_queue_records_parents_and_clamps_the_mesh_dispatch() {
        let signature = sample_signature(MeshTopology::Triangles, false);
        let bindings = MeshBindings {
            amplification: Some((4, 5)),
            ..MeshBindings::default()
        };
        let mut source = mesh_output_wgsl(&signature, &bindings, "pg6m_").expect("valid");
        source.push_str(
            r#"
@compute @workgroup_size(1)
fn task_main(@builtin(workgroup_id) wg: vec3<u32>) {
    _ = pg6m_emit_mesh_tasks(wg.x, wg.x + 1u);
}

@compute @workgroup_size(1)
fn finalize() {
    pg6m_finalize_mesh_dispatch();
}
"#,
        );
        let module = validate_wgsl_baseline(&source);
        assert_eq!(
            struct_layout(&module, "pg6m_DispatchIndirectArgs"),
            (vec![0, 4, 8], DISPATCH_INDIRECT_ARGS_BYTES)
        );
        assert!(source.contains("x: u32,\n    y: u32,\n    z: u32,"));
        let memory = Memory::new(&module);
        run_compute(&module, &memory, "task_main", [3, 1, 1]);
        run_compute(&module, &memory, "finalize", [1, 1, 1]);
        let dispatch: Vec<u32> = memory
            .get(&module, "pg6m_mesh_dispatch")
            .comps()
            .iter()
            .map(Val::u)
            .collect();
        assert_eq!(dispatch, vec![6, 1, 1]);
        let queue = memory.get(&module, "pg6m_mesh_tasks");
        let parents: Vec<u32> = queue.comps()[1].comps()[..6].iter().map(Val::u).collect();
        assert_eq!(parents, vec![0, 1, 1, 2, 2, 2]);
    }
}
