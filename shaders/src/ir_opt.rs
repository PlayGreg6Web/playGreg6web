#![forbid(unsafe_code)]

//! Post-lowering semantics-preserving Naga IR optimization pass (`#2137`, Refs `#663`, `#482`).
//!
//! Runs after `compact_module_bindings` and before `naga::valid::Validator` / WGSL emission to
//! shrink emitted WGSL size, reduce Dawn/Tint compilation latency, and lower GPU VGPR register
//! pressure while preserving 100% of observable shader semantics (including IEEE-754 `NaN` /
//! `-0.0` float edge cases, `@group/@binding` globals, entry-point interfaces, and reflection
//! metadata).
//!
//! Passes applied to each entry point and helper function (iterated to a bounded fixpoint):
//! 1. **Identity & round-trip `As` / bitcast cancellation:**
//!    * `bitcast<T>(x)` where `type(x) == T` -> `x`
//!    * `T(x)` (numeric identity conversion with matching scalar kind & width) -> `x`
//!    * `bitcast<T>(bitcast<U>(x))` where `type(x) == T` and bitwidths match -> `x`
//! 2. **Exact constant folding on pure expressions:**
//!    * `Unary` / `Binary` on scalar `u32`, `i32`, `bool` literals (WGSL wrapping integer
//!      arithmetic, safe `Div`/`Mod` skipping division by zero and `i32::MIN / -1`, shifts
//!      masking shift count `< 32`).
//!    * `Select { condition: Literal(Bool(b)), accept, reject }` -> `accept` or `reject`.
//!    * `AccessIndex { base: Splat { value, .. }, .. }` -> `value`.
//!    * `AccessIndex { base: Compose { components, .. }, index }` -> `components[index]`
//!      (strictly when components map 1-to-1 to elements, never mis-indexing flattened vectors).
//!    * Pure scalar literal bitcasts (`u32 <-> i32`, `f32 -> u32/i32`, and safe finite normal
//!      `u32/i32 -> f32` matching Tint const-eval rules).
//! 3. **Safe integer/boolean algebraic identities (strictly preserving IEEE-754 float semantics):**
//!    * `x + 0`, `0 + x`, `x - 0`, `x * 1`, `1 * x`, `x | 0`, `0 | x`, `x ^ 0`, `0 ^ x`,
//!      `x & ~0`, `~0 & x`, `x << 0`, `x >> 0`, `x & x`, `x | x` -> `x` (integer only).
//!    * `x ^ x` -> `0` (scalar integer, pure `x`).
//!    * `select(a, a, c)` -> `a`.
//!    * `!!b` -> `b`.
//!    * Never folds float `x * 0.0` or `x + 0.0` (`NaN` / `-0.0` preserved).
//! 4. **Dead local-variable pruning & unreferenced pure `Emit` trimming + arena compaction:**
//!    * Removes `LocalVariable`s that are never loaded and never escaped by pointer (`Call`,
//!      `Atomic`, `WorkGroupUniformLoad`, `RayQuery`, etc.) and whose `Store`s have transitively
//!      pure values/indices.
//!    * Prunes unreferenced pure expressions from `Statement::Emit` ranges while strictly
//!      preserving side-effecting / memory / derivative expressions (`Load`, `ImageSample`,
//!      `ImageLoad`, `ImageQuery`, `Derivative`, `CallResult`, `AtomicResult`,
//!      `WorkGroupUniformLoadResult`, `Subgroup*`, `RayQuery*`, `Cooperative*`).
//!    * Compacts the function expression arena in strictly order-preserving topological order so
//!      no dead expressions or dangling `LocalVariable` handles survive into validation or WGSL.

use std::collections::{HashMap, HashSet};

use naga::{
    AddressSpace, Arena, AtomicFunction, BinaryOperator, Block, Constant, Expression, Function,
    GatherMode, GlobalVariable, Handle, ImageClass, ImageQuery, Literal, LocalVariable,
    MathFunction, Module, Override, Range, RayPipelineFunction, RayQueryFunction,
    RelationalFunction, SampleLevel, Scalar, ScalarKind, Span, Statement, StorageFormat,
    SwitchCase, Type, TypeInner, UnaryOperator, UniqueArena, VectorSize,
};

/// Maximum fixed-point optimization iterations per function.
const MAX_OPT_PASSES: usize = 4;

/// Sanitized counters summarizing Naga IR optimizations applied to a shader module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IrOptStats {
    /// Identity (`bitcast<T>(x) -> x`) and round-trip (`bitcast<T>(bitcast<U>(x)) -> x`) casts cancelled.
    pub bitcasts_cancelled: usize,
    /// Pure constant expressions folded (`Binary`, `Unary`, `Select`, `Splat`/`Compose` + `AccessIndex`, `As`).
    pub constants_folded: usize,
    /// Safe integer/boolean algebraic identities simplified (`x + 0`, `x * 1`, `x | 0`, `x ^ 0`, `x & ~0`, `x << 0`, `x >> 0`, `x ^ x`, `x & x`, `x | x`, `select(a, a, c)`, `!!b`).
    pub identities_simplified: usize,
    /// Dead `LocalVariable` declarations removed along with their pure stores.
    pub dead_locals_removed: usize,
    /// Unreferenced pure expressions pruned from `Statement::Emit` ranges.
    pub dead_emits_trimmed: usize,
}

impl IrOptStats {
    /// Total optimization actions applied across all categories.
    pub const fn total_changes(&self) -> usize {
        self.bitcasts_cancelled
            + self.constants_folded
            + self.identities_simplified
            + self.dead_locals_removed
            + self.dead_emits_trimmed
    }

    /// `true` when no optimization fired.
    pub const fn is_zero(&self) -> bool {
        self.bitcasts_cancelled == 0
            && self.constants_folded == 0
            && self.identities_simplified == 0
            && self.dead_locals_removed == 0
            && self.dead_emits_trimmed == 0
    }

    /// Accumulate counters from `other` into `self`.
    pub fn merge_from(&mut self, other: Self) {
        self.bitcasts_cancelled += other.bitcasts_cancelled;
        self.constants_folded += other.constants_folded;
        self.identities_simplified += other.identities_simplified;
        self.dead_locals_removed += other.dead_locals_removed;
        self.dead_emits_trimmed += other.dead_emits_trimmed;
    }
}

/// Read-only module-level tables needed to infer types inside any [`Function`].
struct ModuleContext<'a> {
    types: &'a UniqueArena<Type>,
    constants: &'a Arena<Constant>,
    overrides: &'a Arena<Override>,
    global_variables: &'a Arena<GlobalVariable>,
    func_ret_types: &'a [Option<Handle<Type>>],
}

/// Optimize every entry point and helper function in `module` in place.
///
/// Never alters `@group/@binding` global declarations, entry point signatures/workgroup sizes,
/// or reflection metadata.
pub fn optimize_module(module: &mut Module) -> IrOptStats {
    let func_ret_types: Vec<Option<Handle<Type>>> = module
        .functions
        .iter()
        .map(|(_, f)| f.result.as_ref().map(|r| r.ty))
        .collect();

    let mut total = IrOptStats::default();

    for ep in &mut module.entry_points {
        let ctx = ModuleContext {
            types: &module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            global_variables: &module.global_variables,
            func_ret_types: &func_ret_types,
        };
        total.merge_from(optimize_function_with_ctx(&ctx, &mut ep.function));
    }

    let fn_handles: Vec<Handle<Function>> = module.functions.iter().map(|(h, _)| h).collect();
    for fh in fn_handles {
        let ctx = ModuleContext {
            types: &module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            global_variables: &module.global_variables,
            func_ret_types: &func_ret_types,
        };
        let func = &mut module.functions[fh];
        total.merge_from(optimize_function_with_ctx(&ctx, func));
    }

    total
}

/// Optimize a single [`Function`] against module-level type/global/function tables.
pub fn optimize_function(module: &Module, func: &mut Function) -> IrOptStats {
    let func_ret_types: Vec<Option<Handle<Type>>> = module
        .functions
        .iter()
        .map(|(_, f)| f.result.as_ref().map(|r| r.ty))
        .collect();
    let ctx = ModuleContext {
        types: &module.types,
        constants: &module.constants,
        overrides: &module.overrides,
        global_variables: &module.global_variables,
        func_ret_types: &func_ret_types,
    };
    optimize_function_with_ctx(&ctx, func)
}

fn optimize_function_with_ctx(ctx: &ModuleContext<'_>, func: &mut Function) -> IrOptStats {
    let mut total = IrOptStats::default();

    for _ in 0..MAX_OPT_PASSES {
        let mut pass = IrOptStats::default();

        // 1. Algebraic simplification, bitcast cancellation, and constant folding.
        simplify_expressions(ctx, func, &mut pass);

        // 2. Dead local-variable pruning (locals never loaded or escaped by pointer, whose
        //    stores have transitively pure values and indices).
        prune_dead_locals(func, &mut pass);

        // 3. Dead pure-expression Emit trimming and order-preserving expression arena compaction.
        compact_live_expressions_and_emits(func, &mut pass);

        total.merge_from(pass);
        if pass.is_zero() {
            break;
        }
    }

    total
}

// ---------------------------------------------------------------------------
// WGSL / Tint float literal safety & expression classification helpers
// ---------------------------------------------------------------------------

/// Matches `rdna2_to_wgsl::is_safe_wgsl_f32_literal`: returns `true` iff `f` can be emitted
/// directly as a WGSL `Literal::F32` without risking Naga decimal rounding past `f32::MAX`,
/// losing `-0.0`, or emitting subnormals / `NaN` / `Inf`.
#[inline]
fn is_safe_wgsl_f32_literal(f: f32) -> bool {
    f.to_bits() == 0 || (f.is_finite() && f.is_normal() && f.abs() < 3.4e38_f32)
}

/// Returns `true` if `expr` requires a `Statement::Emit` in Naga IR when live.
///
/// Pre-emitted constants/handles (`Literal`, `Constant`, `Override`, `ZeroValue`,
/// `FunctionArgument`, `GlobalVariable`, `LocalVariable`) and statement-result expressions
/// (`CallResult`, `AtomicResult`, `WorkGroupUniformLoadResult`, `SubgroupBallotResult`,
/// `SubgroupOperationResult`, `RayQueryProceedResult`) must NEVER appear in `Statement::Emit`.
#[inline]
fn is_emittable_expr(expr: &Expression) -> bool {
    !matches!(
        expr,
        Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::Override(_)
            | Expression::ZeroValue(_)
            | Expression::FunctionArgument(_)
            | Expression::GlobalVariable(_)
            | Expression::LocalVariable(_)
            | Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. }
            | Expression::RayQueryProceedResult
    )
}

/// Returns `true` if a single expression node is strictly pure (no memory loads, no texture
/// sampling/loads, no derivatives, no function/atomic/subgroup/ray-query/cooperative results).
#[inline]
fn is_pure_expr_node(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::Override(_)
            | Expression::ZeroValue(_)
            | Expression::FunctionArgument(_)
            | Expression::GlobalVariable(_)
            | Expression::LocalVariable(_)
            | Expression::Access { .. }
            | Expression::AccessIndex { .. }
            | Expression::Splat { .. }
            | Expression::Swizzle { .. }
            | Expression::Compose { .. }
            | Expression::Unary { .. }
            | Expression::Binary { .. }
            | Expression::Select { .. }
            | Expression::Relational { .. }
            | Expression::Math { .. }
            | Expression::As { .. }
            | Expression::ArrayLength(_)
    )
}

/// Visit every direct child `Handle<Expression>` referenced by `expr`.
fn visit_expr_children(expr: &Expression, mut f: impl FnMut(Handle<Expression>)) {
    match expr {
        Expression::Access { base, index } => {
            f(*base);
            f(*index);
        }
        Expression::AccessIndex { base, .. }
        | Expression::Splat { value: base, .. }
        | Expression::Swizzle { vector: base, .. }
        | Expression::Load { pointer: base }
        | Expression::Unary { expr: base, .. }
        | Expression::As { expr: base, .. }
        | Expression::Derivative { expr: base, .. }
        | Expression::Relational { argument: base, .. }
        | Expression::ArrayLength(base)
        | Expression::RayQueryGetIntersection { query: base, .. }
        | Expression::RayQueryVertexPositions { query: base, .. } => {
            f(*base);
        }
        Expression::Compose { components, .. } => {
            for &c in components {
                f(c);
            }
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
            f(*image);
            f(*sampler);
            f(*coordinate);
            if let Some(a) = *array_index {
                f(a);
            }
            if let Some(o) = *offset {
                f(o);
            }
            match *level {
                SampleLevel::Exact(e) | SampleLevel::Bias(e) => f(e),
                SampleLevel::Gradient { x, y } => {
                    f(x);
                    f(y);
                }
                SampleLevel::Auto | SampleLevel::Zero => {}
            }
            if let Some(d) = *depth_ref {
                f(d);
            }
        }
        Expression::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            f(*image);
            f(*coordinate);
            if let Some(a) = *array_index {
                f(a);
            }
            if let Some(s) = *sample {
                f(s);
            }
            if let Some(l) = *level {
                f(l);
            }
        }
        Expression::ImageQuery { image, query } => {
            f(*image);
            if let ImageQuery::Size { level: Some(l) } = *query {
                f(l);
            }
        }
        Expression::Binary { left, right, .. } => {
            f(*left);
            f(*right);
        }
        Expression::Select {
            condition,
            accept,
            reject,
        } => {
            f(*condition);
            f(*accept);
            f(*reject);
        }
        Expression::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            f(*arg);
            if let Some(a1) = *arg1 {
                f(a1);
            }
            if let Some(a2) = *arg2 {
                f(a2);
            }
            if let Some(a3) = *arg3 {
                f(a3);
            }
        }
        Expression::CooperativeLoad { data, .. } => {
            f(data.pointer);
            f(data.stride);
        }
        Expression::CooperativeMultiplyAdd { a, b, c } => {
            f(*a);
            f(*b);
            f(*c);
        }
        Expression::Literal(_)
        | Expression::Constant(_)
        | Expression::Override(_)
        | Expression::ZeroValue(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_)
        | Expression::CallResult(_)
        | Expression::AtomicResult { .. }
        | Expression::WorkGroupUniformLoadResult { .. }
        | Expression::SubgroupBallotResult
        | Expression::SubgroupOperationResult { .. }
        | Expression::RayQueryProceedResult => {}
    }
}

/// Rewrite every direct child `Handle<Expression>` inside `expr` using `map`.
fn remap_expr_children(expr: &mut Expression, mut map: impl FnMut(Handle<Expression>) -> Handle<Expression>) {
    match expr {
        Expression::Access { base, index } => {
            *base = map(*base);
            *index = map(*index);
        }
        Expression::AccessIndex { base, .. }
        | Expression::Splat { value: base, .. }
        | Expression::Swizzle { vector: base, .. }
        | Expression::Load { pointer: base }
        | Expression::Unary { expr: base, .. }
        | Expression::As { expr: base, .. }
        | Expression::Derivative { expr: base, .. }
        | Expression::Relational { argument: base, .. }
        | Expression::ArrayLength(base)
        | Expression::RayQueryGetIntersection { query: base, .. }
        | Expression::RayQueryVertexPositions { query: base, .. } => {
            *base = map(*base);
        }
        Expression::Compose { components, .. } => {
            for c in components.iter_mut() {
                *c = map(*c);
            }
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
            *image = map(*image);
            *sampler = map(*sampler);
            *coordinate = map(*coordinate);
            if let Some(a) = array_index {
                *a = map(*a);
            }
            if let Some(o) = offset {
                *o = map(*o);
            }
            match level {
                SampleLevel::Exact(e) | SampleLevel::Bias(e) => *e = map(*e),
                SampleLevel::Gradient { x, y } => {
                    *x = map(*x);
                    *y = map(*y);
                }
                SampleLevel::Auto | SampleLevel::Zero => {}
            }
            if let Some(d) = depth_ref {
                *d = map(*d);
            }
        }
        Expression::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            *image = map(*image);
            *coordinate = map(*coordinate);
            if let Some(a) = array_index {
                *a = map(*a);
            }
            if let Some(s) = sample {
                *s = map(*s);
            }
            if let Some(l) = level {
                *l = map(*l);
            }
        }
        Expression::ImageQuery { image, query } => {
            *image = map(*image);
            if let ImageQuery::Size { level: Some(l) } = query {
                *l = map(*l);
            }
        }
        Expression::Binary { left, right, .. } => {
            *left = map(*left);
            *right = map(*right);
        }
        Expression::Select {
            condition,
            accept,
            reject,
        } => {
            *condition = map(*condition);
            *accept = map(*accept);
            *reject = map(*reject);
        }
        Expression::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            *arg = map(*arg);
            if let Some(a1) = arg1 {
                *a1 = map(*a1);
            }
            if let Some(a2) = arg2 {
                *a2 = map(*a2);
            }
            if let Some(a3) = arg3 {
                *a3 = map(*a3);
            }
        }
        Expression::CooperativeLoad { data, .. } => {
            data.pointer = map(data.pointer);
            data.stride = map(data.stride);
        }
        Expression::CooperativeMultiplyAdd { a, b, c } => {
            *a = map(*a);
            *b = map(*b);
            *c = map(*c);
        }
        Expression::Literal(_)
        | Expression::Constant(_)
        | Expression::Override(_)
        | Expression::ZeroValue(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_)
        | Expression::CallResult(_)
        | Expression::AtomicResult { .. }
        | Expression::WorkGroupUniformLoadResult { .. }
        | Expression::SubgroupBallotResult
        | Expression::SubgroupOperationResult { .. }
        | Expression::RayQueryProceedResult => {}
    }
}

/// Remap every input expression handle in `block` (and optionally statement `result` handles when
/// `remap_results` is `true`, used during arena compaction). Does not modify `Statement::Emit`.
fn remap_block_exprs(
    block: &mut Block,
    remap_results: bool,
    map: &impl Fn(Handle<Expression>) -> Handle<Expression>,
) {
    for stmt in block.iter_mut() {
        match stmt {
            Statement::Emit(_)
            | Statement::Break
            | Statement::Continue
            | Statement::Kill
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_) => {}
            Statement::Block(inner) => remap_block_exprs(inner, remap_results, map),
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                *condition = map(*condition);
                remap_block_exprs(accept, remap_results, map);
                remap_block_exprs(reject, remap_results, map);
            }
            Statement::Switch { selector, cases } => {
                *selector = map(*selector);
                for case in cases.iter_mut() {
                    remap_block_exprs(&mut case.body, remap_results, map);
                }
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                remap_block_exprs(body, remap_results, map);
                remap_block_exprs(continuing, remap_results, map);
                if let Some(cond) = break_if {
                    *cond = map(*cond);
                }
            }
            Statement::Return { value } => {
                if let Some(v) = value {
                    *v = map(*v);
                }
            }
            Statement::Store { pointer, value } => {
                *pointer = map(*pointer);
                *value = map(*value);
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                *image = map(*image);
                *coordinate = map(*coordinate);
                if let Some(a) = array_index {
                    *a = map(*a);
                }
                *value = map(*value);
            }
            Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun: _,
                value,
            } => {
                *image = map(*image);
                *coordinate = map(*coordinate);
                if let Some(a) = array_index {
                    *a = map(*a);
                }
                *value = map(*value);
            }
            Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                *pointer = map(*pointer);
                if let AtomicFunction::Exchange {
                    compare: Some(cmp),
                } = fun
                {
                    *cmp = map(*cmp);
                }
                *value = map(*value);
                if remap_results
                    && let Some(r) = result
                {
                    *r = map(*r);
                }
            }
            Statement::WorkGroupUniformLoad { pointer, result } => {
                *pointer = map(*pointer);
                if remap_results {
                    *result = map(*result);
                }
            }
            Statement::Call {
                function: _,
                arguments,
                result,
            } => {
                for arg in arguments.iter_mut() {
                    *arg = map(*arg);
                }
                if remap_results
                    && let Some(r) = result
                {
                    *r = map(*r);
                }
            }
            Statement::RayQuery { query, fun } => {
                *query = map(*query);
                match fun {
                    RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        *acceleration_structure = map(*acceleration_structure);
                        *descriptor = map(*descriptor);
                    }
                    RayQueryFunction::Proceed { result } => {
                        if remap_results {
                            *result = map(*result);
                        }
                    }
                    RayQueryFunction::GenerateIntersection { hit_t } => {
                        *hit_t = map(*hit_t);
                    }
                    RayQueryFunction::ConfirmIntersection | RayQueryFunction::Terminate => {}
                }
            }
            Statement::SubgroupBallot { result, predicate } => {
                if let Some(p) = predicate {
                    *p = map(*p);
                }
                if remap_results {
                    *result = map(*result);
                }
            }
            Statement::SubgroupCollectiveOperation {
                op: _,
                collective_op: _,
                argument,
                result,
            } => {
                *argument = map(*argument);
                if remap_results {
                    *result = map(*result);
                }
            }
            Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                *argument = map(*argument);
                match mode {
                    GatherMode::Broadcast(idx)
                    | GatherMode::Shuffle(idx)
                    | GatherMode::ShuffleDown(idx)
                    | GatherMode::ShuffleUp(idx)
                    | GatherMode::ShuffleXor(idx)
                    | GatherMode::QuadBroadcast(idx) => {
                        *idx = map(*idx);
                    }
                    GatherMode::BroadcastFirst | GatherMode::QuadSwap(_) => {}
                }
                if remap_results {
                    *result = map(*result);
                }
            }
            Statement::CooperativeStore { target, data } => {
                *target = map(*target);
                data.pointer = map(data.pointer);
                data.stride = map(data.stride);
            }
            Statement::RayPipelineFunction(RayPipelineFunction::TraceRay {
                acceleration_structure,
                descriptor,
                payload,
            }) => {
                *acceleration_structure = map(*acceleration_structure);
                *descriptor = map(*descriptor);
                *payload = map(*payload);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Type inference
// ---------------------------------------------------------------------------

fn storage_format_scalar(format: StorageFormat) -> Scalar {
    match format {
        StorageFormat::R8Uint
        | StorageFormat::R16Uint
        | StorageFormat::Rg8Uint
        | StorageFormat::R32Uint
        | StorageFormat::Rg16Uint
        | StorageFormat::Rgba8Uint
        | StorageFormat::Rg32Uint
        | StorageFormat::Rgba16Uint
        | StorageFormat::Rgba32Uint
        | StorageFormat::Rgb10a2Uint => Scalar::U32,
        StorageFormat::R8Sint
        | StorageFormat::R16Sint
        | StorageFormat::Rg8Sint
        | StorageFormat::R32Sint
        | StorageFormat::Rg16Sint
        | StorageFormat::Rgba8Sint
        | StorageFormat::Rg32Sint
        | StorageFormat::Rgba16Sint
        | StorageFormat::Rgba32Sint => Scalar::I32,
        StorageFormat::R64Uint => Scalar::U64,
        _ => Scalar::F32,
    }
}

fn scalar_of_type_inner(ty: &TypeInner) -> Option<Scalar> {
    match *ty {
        TypeInner::Scalar(s)
        | TypeInner::Vector { scalar: s, .. }
        | TypeInner::Matrix { scalar: s, .. } => Some(s),
        _ => None,
    }
}

/// Extract `(optional_vector_size, scalar)` for bitcast/conversion compatibility checks.
/// Returns `None` for non-scalar/non-vector types (matrices, structs, arrays, pointers).
fn value_shape(ty: &TypeInner) -> Option<(Option<VectorSize>, Scalar)> {
    match *ty {
        TypeInner::Scalar(s) => Some((None, s)),
        TypeInner::Vector { size, scalar } => Some((Some(size), scalar)),
        _ => None,
    }
}

fn infer_single_expr_type(
    ctx: &ModuleContext<'_>,
    func: &Function,
    resolved: &[Option<TypeInner>],
    expr: &Expression,
) -> Option<TypeInner> {
    let get = |h: Handle<Expression>| resolved.get(h.index()).and_then(|t| t.as_ref());

    match expr {
        Expression::Literal(lit) => Some(TypeInner::Scalar(match lit {
            Literal::F64(_) => Scalar::F64,
            Literal::F32(_) => Scalar::F32,
            Literal::F16(_) => Scalar::F16,
            Literal::U32(_) => Scalar::U32,
            Literal::I32(_) => Scalar::I32,
            Literal::U16(_) => Scalar {
                kind: ScalarKind::Uint,
                width: 2,
            },
            Literal::I16(_) => Scalar {
                kind: ScalarKind::Sint,
                width: 2,
            },
            Literal::U64(_) => Scalar::U64,
            Literal::I64(_) => Scalar::I64,
            Literal::Bool(_) => Scalar::BOOL,
            Literal::AbstractInt(_) => Scalar::ABSTRACT_INT,
            Literal::AbstractFloat(_) => Scalar::ABSTRACT_FLOAT,
        })),
        Expression::Constant(c) => Some(ctx.types[ctx.constants[*c].ty].inner.clone()),
        Expression::Override(o) => Some(ctx.types[ctx.overrides[*o].ty].inner.clone()),
        Expression::ZeroValue(ty) => Some(ctx.types[*ty].inner.clone()),
        Expression::Compose { ty, .. } => Some(ctx.types[*ty].inner.clone()),
        Expression::FunctionArgument(idx) => func
            .arguments
            .get(*idx as usize)
            .map(|a| ctx.types[a.ty].inner.clone()),
        Expression::GlobalVariable(g) => {
            let var = &ctx.global_variables[*g];
            match var.space {
                AddressSpace::Handle => Some(ctx.types[var.ty].inner.clone()),
                space => Some(TypeInner::Pointer {
                    base: var.ty,
                    space,
                }),
            }
        }
        Expression::LocalVariable(lv) => Some(TypeInner::Pointer {
            base: func.local_variables[*lv].ty,
            space: AddressSpace::Function,
        }),
        Expression::Load { pointer } => match get(*pointer)? {
            TypeInner::Pointer { base, .. } => Some(ctx.types[*base].inner.clone()),
            TypeInner::ValuePointer {
                size: Some(size),
                scalar,
                ..
            } => Some(TypeInner::Vector {
                size: *size,
                scalar: *scalar,
            }),
            TypeInner::ValuePointer {
                size: None, scalar, ..
            } => Some(TypeInner::Scalar(*scalar)),
            _ => None,
        },
        Expression::Access { base, .. } => infer_access_type(ctx, get(*base)?, None),
        Expression::AccessIndex { base, index } => {
            infer_access_type(ctx, get(*base)?, Some(*index))
        }
        Expression::Splat { size, value } => match get(*value)? {
            TypeInner::Scalar(scalar) => Some(TypeInner::Vector {
                size: *size,
                scalar: *scalar,
            }),
            _ => None,
        },
        Expression::Swizzle { size, vector, .. } => match get(*vector)? {
            TypeInner::Vector { scalar, .. } => Some(TypeInner::Vector {
                size: *size,
                scalar: *scalar,
            }),
            _ => None,
        },
        Expression::Unary { expr, .. } | Expression::Derivative { expr, .. } => get(*expr).cloned(),
        Expression::Binary { op, left, right } => {
            let lt = get(*left)?;
            let rt = get(*right)?;
            match op {
                BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterEqual
                | BinaryOperator::LogicalAnd
                | BinaryOperator::LogicalOr => match lt {
                    TypeInner::Scalar(_) => Some(TypeInner::Scalar(Scalar::BOOL)),
                    TypeInner::Vector { size, .. } => Some(TypeInner::Vector {
                        size: *size,
                        scalar: Scalar::BOOL,
                    }),
                    _ => None,
                },
                BinaryOperator::Multiply => match (lt, rt) {
                    (
                        TypeInner::Matrix {
                            columns: _,
                            rows,
                            scalar,
                        },
                        TypeInner::Vector { .. },
                    ) => Some(TypeInner::Vector {
                        size: *rows,
                        scalar: *scalar,
                    }),
                    (
                        TypeInner::Vector { .. },
                        TypeInner::Matrix {
                            columns,
                            rows: _,
                            scalar,
                        },
                    ) => Some(TypeInner::Vector {
                        size: *columns,
                        scalar: *scalar,
                    }),
                    (
                        TypeInner::Matrix { rows, scalar, .. },
                        TypeInner::Matrix { columns, .. },
                    ) => Some(TypeInner::Matrix {
                        columns: *columns,
                        rows: *rows,
                        scalar: *scalar,
                    }),
                    (TypeInner::Scalar(_), TypeInner::Vector { .. })
                    | (TypeInner::Scalar(_), TypeInner::Matrix { .. }) => Some(rt.clone()),
                    _ => Some(lt.clone()),
                },
                _ => match (lt, rt) {
                    (TypeInner::Scalar(_), TypeInner::Vector { .. }) => Some(rt.clone()),
                    _ => Some(lt.clone()),
                },
            }
        }
        Expression::Select { accept, .. } => get(*accept).cloned(),
        Expression::Relational { fun, argument } => match fun {
            RelationalFunction::All | RelationalFunction::Any => {
                Some(TypeInner::Scalar(Scalar::BOOL))
            }
            RelationalFunction::IsNan | RelationalFunction::IsInf => match get(*argument)? {
                TypeInner::Scalar(_) => Some(TypeInner::Scalar(Scalar::BOOL)),
                TypeInner::Vector { size, .. } => Some(TypeInner::Vector {
                    size: *size,
                    scalar: Scalar::BOOL,
                }),
                _ => None,
            },
        },
        Expression::As {
            expr,
            kind,
            convert,
        } => match get(*expr)? {
            TypeInner::Scalar(s) => Some(TypeInner::Scalar(Scalar {
                kind: *kind,
                width: convert.unwrap_or(s.width),
            })),
            TypeInner::Vector { size, scalar } => Some(TypeInner::Vector {
                size: *size,
                scalar: Scalar {
                    kind: *kind,
                    width: convert.unwrap_or(scalar.width),
                },
            }),
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => Some(TypeInner::Matrix {
                columns: *columns,
                rows: *rows,
                scalar: Scalar {
                    kind: *kind,
                    width: convert.unwrap_or(scalar.width),
                },
            }),
            _ => None,
        },
        Expression::Math { fun, arg, .. } => match fun {
            MathFunction::Dot
            | MathFunction::Length
            | MathFunction::Distance
            | MathFunction::Determinant => Some(TypeInner::Scalar(scalar_of_type_inner(get(*arg)?)?)),
            MathFunction::Pack4x8snorm
            | MathFunction::Pack4x8unorm
            | MathFunction::Pack2x16snorm
            | MathFunction::Pack2x16unorm
            | MathFunction::Pack2x16float
            | MathFunction::Pack4xI8
            | MathFunction::Pack4xU8
            | MathFunction::Pack4xI8Clamp
            | MathFunction::Pack4xU8Clamp => Some(TypeInner::Scalar(Scalar::U32)),
            MathFunction::Unpack4x8snorm | MathFunction::Unpack4x8unorm => {
                Some(TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                })
            }
            MathFunction::Unpack2x16snorm
            | MathFunction::Unpack2x16unorm
            | MathFunction::Unpack2x16float => Some(TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::F32,
            }),
            MathFunction::Unpack4xI8 => Some(TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::I32,
            }),
            MathFunction::Unpack4xU8 => Some(TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::U32,
            }),
            MathFunction::Modf
            | MathFunction::Frexp
            | MathFunction::Outer
            | MathFunction::Transpose
            | MathFunction::Dot4I8Packed
            | MathFunction::Dot4U8Packed => None,
            _ => get(*arg).cloned(),
        },
        Expression::ImageSample {
            image, depth_ref, ..
        } => {
            if depth_ref.is_some() {
                return Some(TypeInner::Scalar(Scalar::F32));
            }
            match get(*image)? {
                TypeInner::Image { class, .. } => match *class {
                    ImageClass::Sampled { kind, .. } => Some(TypeInner::Vector {
                        size: VectorSize::Quad,
                        scalar: Scalar { kind, width: 4 },
                    }),
                    ImageClass::Depth { .. } => Some(TypeInner::Scalar(Scalar::F32)),
                    ImageClass::Storage { format, .. } => Some(TypeInner::Vector {
                        size: VectorSize::Quad,
                        scalar: storage_format_scalar(format),
                    }),
                    ImageClass::External => Some(TypeInner::Vector {
                        size: VectorSize::Quad,
                        scalar: Scalar::F32,
                    }),
                },
                _ => None,
            }
        }
        Expression::ImageLoad { image, .. } => match get(*image)? {
            TypeInner::Image { class, .. } => match *class {
                ImageClass::Sampled { kind, .. } => Some(TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar { kind, width: 4 },
                }),
                ImageClass::Depth { .. } => Some(TypeInner::Scalar(Scalar::F32)),
                ImageClass::Storage { format, .. } => Some(TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: storage_format_scalar(format),
                }),
                ImageClass::External => Some(TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                }),
            },
            _ => None,
        },
        Expression::CallResult(callee) => ctx
            .func_ret_types
            .get(callee.index())
            .and_then(|opt| *opt)
            .map(|ty| ctx.types[ty].inner.clone()),
        Expression::AtomicResult { ty, .. }
        | Expression::WorkGroupUniformLoadResult { ty }
        | Expression::SubgroupOperationResult { ty } => Some(ctx.types[*ty].inner.clone()),
        Expression::SubgroupBallotResult => Some(TypeInner::Vector {
            size: VectorSize::Quad,
            scalar: Scalar::U32,
        }),
        Expression::ArrayLength(_) => Some(TypeInner::Scalar(Scalar::U32)),
        Expression::RayQueryProceedResult => Some(TypeInner::Scalar(Scalar::BOOL)),
        Expression::ImageQuery { .. }
        | Expression::RayQueryGetIntersection { .. }
        | Expression::RayQueryVertexPositions { .. }
        | Expression::CooperativeLoad { .. }
        | Expression::CooperativeMultiplyAdd { .. } => None,
    }
}

fn infer_access_type(
    ctx: &ModuleContext<'_>,
    base_ty: &TypeInner,
    const_index: Option<u32>,
) -> Option<TypeInner> {
    match base_ty {
        TypeInner::Vector { scalar, .. } => Some(TypeInner::Scalar(*scalar)),
        TypeInner::Matrix { rows, scalar, .. } => Some(TypeInner::Vector {
            size: *rows,
            scalar: *scalar,
        }),
        TypeInner::Array { base, .. } => Some(ctx.types[*base].inner.clone()),
        TypeInner::Struct { members, .. } => {
            let idx = const_index? as usize;
            members.get(idx).map(|m| ctx.types[m.ty].inner.clone())
        }
        TypeInner::Pointer { base, space } => match &ctx.types[*base].inner {
            TypeInner::Vector { scalar, .. } => Some(TypeInner::ValuePointer {
                size: None,
                scalar: *scalar,
                space: *space,
            }),
            TypeInner::Matrix { rows, scalar, .. } => Some(TypeInner::ValuePointer {
                size: Some(*rows),
                scalar: *scalar,
                space: *space,
            }),
            TypeInner::Array { base: elem, .. } => Some(TypeInner::Pointer {
                base: *elem,
                space: *space,
            }),
            TypeInner::Struct { members, .. } => {
                let idx = const_index? as usize;
                members.get(idx).map(|m| TypeInner::Pointer {
                    base: m.ty,
                    space: *space,
                })
            }
            _ => None,
        },
        TypeInner::ValuePointer {
            size: Some(_),
            scalar,
            space,
        } => Some(TypeInner::ValuePointer {
            size: None,
            scalar: *scalar,
            space: *space,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Pass 1: Expression simplification (bitcast cancellation, constant folding, identities)
// ---------------------------------------------------------------------------

fn simplify_expressions(ctx: &ModuleContext<'_>, func: &mut Function, stats: &mut IrOptStats) {
    let n = func.expressions.len();
    if n == 0 {
        return;
    }

    // Track handles directly consumed by `As { kind: Float, convert: None }` or float `Binary` ops
    // so folding an expression to a bare `Literal` never creates a Tint-rejected const-eval
    // expression (e.g., `bitcast<f32>(0x7f800000u)` or `1e38f * 10.0f`).
    let mut used_by_float_bitcast = vec![false; n];
    let mut used_by_float_binary = vec![false; n];
    for (_, expr) in func.expressions.iter() {
        match expr {
            Expression::As {
                expr: inner,
                kind: ScalarKind::Float,
                convert: None,
            } => {
                used_by_float_bitcast[inner.index()] = true;
            }
            Expression::Binary {
                op:
                    BinaryOperator::Add
                    | BinaryOperator::Subtract
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Modulo,
                left,
                right,
            } => {
                used_by_float_binary[left.index()] = true;
                used_by_float_binary[right.index()] = true;
            }
            _ => {}
        }
    }

    let handles: Vec<Handle<Expression>> = func.expressions.iter().map(|(h, _)| h).collect();
    let mut remap: Vec<Handle<Expression>> = handles.clone();
    let mut resolved: Vec<Option<TypeInner>> = vec![None; n];
    let mut any_remapped = false;

    for h in handles {
        let idx = h.index();

        // Rewrite all child handles to their already-simplified canonical handles.
        {
            let expr = &mut func.expressions[h];
            remap_expr_children(expr, |c| remap[c.index()]);
        }

        // Infer type of `h` from its canonicalized operands.
        let cur_ty = infer_single_expr_type(ctx, func, &resolved, &func.expressions[h]);
        resolved[idx] = cur_ty.clone();

        // Guard helper: check whether replacing `h` with a bare `Literal` could trigger an unsafe
        // Tint const-eval in a direct consumer.
        let safe_to_become_literal = |lit: Literal| -> bool {
            if used_by_float_bitcast[idx] {
                let bits = match lit {
                    Literal::U32(u) => Some(u),
                    Literal::I32(i) => Some(i as u32),
                    _ => None,
                };
                if let Some(b) = bits
                    && !is_safe_wgsl_f32_literal(f32::from_bits(b))
                {
                    return false;
                }
            }
            if used_by_float_binary[idx]
                && let Literal::F32(f) = lit
                && !is_safe_wgsl_f32_literal(f)
            {
                return false;
            }
            true
        };

        // Guard helper: check whether redirecting `h` to `target` (an existing handle) is safe
        // against turning a float bitcast or float binary op into an unsafe Tint const-eval.
        let safe_to_redirect_to = |target: Handle<Expression>| -> bool {
            if let Expression::Literal(lit) = func.expressions[target] {
                if !safe_to_become_literal(lit) {
                    return false;
                }
                if used_by_float_binary[idx] && matches!(lit, Literal::F32(_)) {
                    // Avoid turning a runtime float binary op into an unchecked two-literal
                    // float const-eval that could overflow to Inf/NaN in Tint.
                    return false;
                }
            }
            true
        };

        enum Rewrite {
            Redirect(Handle<Expression>, RewriteKind),
            Literal(Literal, RewriteKind),
        }
        enum RewriteKind {
            Bitcast,
            ConstFold,
            Identity,
        }

        let rewrite = match &func.expressions[h] {
            // 1. Identity & round-trip `As` / bitcast cancellation + literal bitcast folding.
            Expression::As {
                expr: inner,
                kind,
                convert,
            } => {
                let inner = *inner;
                let kind = *kind;
                let convert = *convert;
                let inner_shape = resolved[inner.index()].as_ref().and_then(value_shape);

                // (a) Identity bitcast (`bitcast<T>(x)` where `x: T`) or identity numeric cast
                //     (`T(x)` where `x: T` with matching scalar kind and width).
                let is_identity = match (inner_shape, convert) {
                    (Some((_, s)), None) => s.kind == kind,
                    (Some((_, s)), Some(w)) => s.kind == kind && s.width == w,
                    _ => false,
                };

                if is_identity && safe_to_redirect_to(inner) {
                    Some(Rewrite::Redirect(inner, RewriteKind::Bitcast))
                } else if convert.is_none() {
                    // (b) Round-trip bitcast: `bitcast<T>(bitcast<U>(orig))` where
                    //     `type(orig) == T` and bitwidths match.
                    if let Expression::As {
                        expr: orig,
                        kind: _mid_kind,
                        convert: None,
                    } = func.expressions[inner]
                    {
                        let orig_shape = resolved[orig.index()].as_ref().and_then(value_shape);
                        match (orig_shape, inner_shape) {
                            (Some((orig_vec, orig_s)), Some((mid_vec, mid_s)))
                                if orig_vec == mid_vec
                                    && orig_s.kind == kind
                                    && orig_s.width == mid_s.width
                                    && safe_to_redirect_to(orig) =>
                            {
                                Some(Rewrite::Redirect(orig, RewriteKind::Bitcast))
                            }
                            _ => None,
                        }
                    } else if let Expression::Literal(lit) = func.expressions[inner] {
                        // (c) Constant scalar bitcast folding (keeping Tint float literal safety).
                        fold_literal_bitcast(lit, kind)
                            .filter(|&out_lit| safe_to_become_literal(out_lit))
                            .map(|out_lit| Rewrite::Literal(out_lit, RewriteKind::ConstFold))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }

            // 2. Unary constant folding & `!!b -> b`.
            Expression::Unary { op, expr: inner } => {
                let op = *op;
                let inner = *inner;
                match (op, &func.expressions[inner]) {
                    (UnaryOperator::LogicalNot, Expression::Literal(Literal::Bool(b))) => {
                        Some(Rewrite::Literal(Literal::Bool(!*b), RewriteKind::ConstFold))
                    }
                    (UnaryOperator::BitwiseNot, Expression::Literal(Literal::U32(v))) => {
                        let out = Literal::U32(!*v);
                        safe_to_become_literal(out)
                            .then_some(Rewrite::Literal(out, RewriteKind::ConstFold))
                    }
                    (UnaryOperator::BitwiseNot, Expression::Literal(Literal::I32(v))) => {
                        let out = Literal::I32(!*v);
                        safe_to_become_literal(out)
                            .then_some(Rewrite::Literal(out, RewriteKind::ConstFold))
                    }
                    (UnaryOperator::Negate, Expression::Literal(Literal::I32(v))) => {
                        let out = Literal::I32((*v).wrapping_neg());
                        safe_to_become_literal(out)
                            .then_some(Rewrite::Literal(out, RewriteKind::ConstFold))
                    }
                    (
                        UnaryOperator::LogicalNot,
                        Expression::Unary {
                            op: UnaryOperator::LogicalNot,
                            expr: orig,
                        },
                    ) if safe_to_redirect_to(*orig) => {
                        Some(Rewrite::Redirect(*orig, RewriteKind::Identity))
                    }
                    _ => None,
                }
            }

            // 3. Binary constant folding & safe integer/boolean algebraic identities.
            Expression::Binary { op, left, right } => {
                let op = *op;
                let left = *left;
                let right = *right;

                // (a) Constant folding on literal scalar operands (`u32`, `i32`, `bool`).
                if let Some(folded) =
                    fold_literal_binary(op, &func.expressions[left], &func.expressions[right])
                {
                    safe_to_become_literal(folded)
                        .then_some(Rewrite::Literal(folded, RewriteKind::ConstFold))
                } else {
                    // (b) Safe algebraic identities (strictly integer `u32`/`i32` only — NEVER float!).
                    simplify_integer_binary_identity(
                        op,
                        left,
                        right,
                        cur_ty.as_ref(),
                        &resolved,
                        func,
                    )
                    .and_then(|res| match res {
                        IntIdentity::Redirect(target) if safe_to_redirect_to(target) => {
                            Some(Rewrite::Redirect(target, RewriteKind::Identity))
                        }
                        IntIdentity::Literal(lit) if safe_to_become_literal(lit) => {
                            Some(Rewrite::Literal(lit, RewriteKind::Identity))
                        }
                        _ => None,
                    })
                }
            }

            // 4. Select constant folding (`select(reject, accept, const_bool)`) & `select(a, a, c) -> a`.
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let condition = *condition;
                let accept = *accept;
                let reject = *reject;
                if let Expression::Literal(Literal::Bool(b)) = func.expressions[condition] {
                    let target = if b { accept } else { reject };
                    safe_to_redirect_to(target)
                        .then_some(Rewrite::Redirect(target, RewriteKind::ConstFold))
                } else if accept == reject && safe_to_redirect_to(accept) {
                    Some(Rewrite::Redirect(accept, RewriteKind::Identity))
                } else {
                    None
                }
            }

            // 5. `AccessIndex` folding through `Splat` and 1-to-1 `Compose`.
            Expression::AccessIndex { base, index } => {
                let base = *base;
                let index = *index;
                match &func.expressions[base] {
                    Expression::Splat { size, value } => {
                        let target = *value;
                        if (index as usize) < (*size as usize) && safe_to_redirect_to(target) {
                            Some(Rewrite::Redirect(target, RewriteKind::ConstFold))
                        } else {
                            None
                        }
                    }
                    Expression::Compose { ty, components } => {
                        let ty = *ty;
                        let is_one_to_one = match &ctx.types[ty].inner {
                            TypeInner::Vector { size, .. } => components.len() == *size as usize,
                            TypeInner::Matrix { columns, .. } => {
                                components.len() == *columns as usize
                            }
                            TypeInner::Struct { members, .. } => components.len() == members.len(),
                            TypeInner::Array { .. } => true,
                            _ => false,
                        };
                        if is_one_to_one
                            && let Some(&target) = components.get(index as usize)
                            && safe_to_redirect_to(target)
                        {
                            Some(Rewrite::Redirect(target, RewriteKind::ConstFold))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }

            _ => None,
        };

        if let Some(rew) = rewrite {
            match rew {
                Rewrite::Redirect(target, kind) => {
                    remap[idx] = target;
                    resolved[idx] = resolved[target.index()].clone();
                    any_remapped = true;
                    match kind {
                        RewriteKind::Bitcast => stats.bitcasts_cancelled += 1,
                        RewriteKind::ConstFold => stats.constants_folded += 1,
                        RewriteKind::Identity => stats.identities_simplified += 1,
                    }
                }
                Rewrite::Literal(lit, kind) => {
                    func.expressions[h] = Expression::Literal(lit);
                    resolved[idx] =
                        infer_single_expr_type(ctx, func, &resolved, &func.expressions[h]);
                    func.named_expressions.shift_remove(&h);
                    match kind {
                        RewriteKind::Bitcast => stats.bitcasts_cancelled += 1,
                        RewriteKind::ConstFold => stats.constants_folded += 1,
                        RewriteKind::Identity => stats.identities_simplified += 1,
                    }
                }
            }
        }
    }

    if any_remapped {
        remap_block_exprs(&mut func.body, false, &|h| remap[h.index()]);
        for (_, local) in func.local_variables.iter_mut() {
            if let Some(ref mut init_h) = local.init {
                *init_h = remap[init_h.index()];
            }
        }
    }
}

fn fold_literal_bitcast(lit: Literal, target_kind: ScalarKind) -> Option<Literal> {
    match (lit, target_kind) {
        (Literal::U32(u), ScalarKind::Sint) => Some(Literal::I32(u as i32)),
        (Literal::I32(i), ScalarKind::Uint) => Some(Literal::U32(i as u32)),
        (Literal::F32(f), ScalarKind::Uint) => Some(Literal::U32(f.to_bits())),
        (Literal::F32(f), ScalarKind::Sint) => Some(Literal::I32(f.to_bits() as i32)),
        (Literal::U32(u), ScalarKind::Float) => {
            let f = f32::from_bits(u);
            is_safe_wgsl_f32_literal(f).then_some(Literal::F32(f))
        }
        (Literal::I32(i), ScalarKind::Float) => {
            let f = f32::from_bits(i as u32);
            is_safe_wgsl_f32_literal(f).then_some(Literal::F32(f))
        }
        _ => None,
    }
}

fn fold_literal_binary(
    op: BinaryOperator,
    left: &Expression,
    right: &Expression,
) -> Option<Literal> {
    match (left, right) {
        (Expression::Literal(Literal::U32(a)), Expression::Literal(Literal::U32(b))) => {
            let (a, b) = (*a, *b);
            match op {
                BinaryOperator::Add => Some(Literal::U32(a.wrapping_add(b))),
                BinaryOperator::Subtract => Some(Literal::U32(a.wrapping_sub(b))),
                BinaryOperator::Multiply => Some(Literal::U32(a.wrapping_mul(b))),
                BinaryOperator::Divide => (b != 0).then(|| Literal::U32(a / b)),
                BinaryOperator::Modulo => (b != 0).then(|| Literal::U32(a % b)),
                BinaryOperator::And => Some(Literal::U32(a & b)),
                BinaryOperator::InclusiveOr => Some(Literal::U32(a | b)),
                BinaryOperator::ExclusiveOr => Some(Literal::U32(a ^ b)),
                BinaryOperator::ShiftLeft => Some(Literal::U32(a.wrapping_shl(b & 31))),
                BinaryOperator::ShiftRight => Some(Literal::U32(a.wrapping_shr(b & 31))),
                BinaryOperator::Equal => Some(Literal::Bool(a == b)),
                BinaryOperator::NotEqual => Some(Literal::Bool(a != b)),
                BinaryOperator::Less => Some(Literal::Bool(a < b)),
                BinaryOperator::LessEqual => Some(Literal::Bool(a <= b)),
                BinaryOperator::Greater => Some(Literal::Bool(a > b)),
                BinaryOperator::GreaterEqual => Some(Literal::Bool(a >= b)),
                _ => None,
            }
        }
        (Expression::Literal(Literal::I32(a)), Expression::Literal(Literal::I32(b))) => {
            let (a, b) = (*a, *b);
            match op {
                BinaryOperator::Add => Some(Literal::I32(a.wrapping_add(b))),
                BinaryOperator::Subtract => Some(Literal::I32(a.wrapping_sub(b))),
                BinaryOperator::Multiply => Some(Literal::I32(a.wrapping_mul(b))),
                BinaryOperator::Divide => {
                    (b != 0 && !(a == i32::MIN && b == -1)).then(|| Literal::I32(a / b))
                }
                BinaryOperator::Modulo => {
                    (b != 0 && !(a == i32::MIN && b == -1)).then(|| Literal::I32(a % b))
                }
                BinaryOperator::And => Some(Literal::I32(a & b)),
                BinaryOperator::InclusiveOr => Some(Literal::I32(a | b)),
                BinaryOperator::ExclusiveOr => Some(Literal::I32(a ^ b)),
                BinaryOperator::Equal => Some(Literal::Bool(a == b)),
                BinaryOperator::NotEqual => Some(Literal::Bool(a != b)),
                BinaryOperator::Less => Some(Literal::Bool(a < b)),
                BinaryOperator::LessEqual => Some(Literal::Bool(a <= b)),
                BinaryOperator::Greater => Some(Literal::Bool(a > b)),
                BinaryOperator::GreaterEqual => Some(Literal::Bool(a >= b)),
                _ => None,
            }
        }
        (Expression::Literal(Literal::I32(a)), Expression::Literal(Literal::U32(b))) => {
            let (a, b) = (*a, *b);
            match op {
                BinaryOperator::ShiftLeft => Some(Literal::I32(a.wrapping_shl(b & 31))),
                BinaryOperator::ShiftRight => Some(Literal::I32(a.wrapping_shr(b & 31))),
                _ => None,
            }
        }
        (Expression::Literal(Literal::Bool(a)), Expression::Literal(Literal::Bool(b))) => {
            let (a, b) = (*a, *b);
            match op {
                BinaryOperator::LogicalAnd => Some(Literal::Bool(a && b)),
                BinaryOperator::LogicalOr => Some(Literal::Bool(a || b)),
                BinaryOperator::Equal => Some(Literal::Bool(a == b)),
                BinaryOperator::NotEqual => Some(Literal::Bool(a != b)),
                _ => None,
            }
        }
        _ => None,
    }
}

enum IntIdentity {
    Redirect(Handle<Expression>),
    Literal(Literal),
}

fn is_int_zero(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::Literal(Literal::U32(0) | Literal::I32(0))
    )
}

fn is_int_one(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::Literal(Literal::U32(1) | Literal::I32(1))
    )
}

fn is_int_all_ones(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::Literal(Literal::U32(u32::MAX) | Literal::I32(-1))
    )
}

fn simplify_integer_binary_identity(
    op: BinaryOperator,
    left: Handle<Expression>,
    right: Handle<Expression>,
    result_ty: Option<&TypeInner>,
    resolved: &[Option<TypeInner>],
    func: &Function,
) -> Option<IntIdentity> {
    let res_ty = result_ty?;
    let res_scalar = scalar_of_type_inner(res_ty)?;
    // Strictly integer (`u32`/`i32`) only — NEVER simplify floats (`NaN`, `-0.0`, etc.)!
    if !matches!(res_scalar, Scalar::U32 | Scalar::I32) {
        return None;
    }

    let left_expr = &func.expressions[left];
    let right_expr = &func.expressions[right];
    let left_matches_ty = resolved[left.index()].as_ref() == Some(res_ty);
    let right_matches_ty = resolved[right.index()].as_ref() == Some(res_ty);

    match op {
        BinaryOperator::Add => {
            if is_int_zero(right_expr) && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
            if is_int_zero(left_expr) && right_matches_ty {
                return Some(IntIdentity::Redirect(right));
            }
        }
        BinaryOperator::Subtract => {
            if is_int_zero(right_expr) && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
        }
        BinaryOperator::Multiply => {
            if is_int_one(right_expr) && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
            if is_int_one(left_expr) && right_matches_ty {
                return Some(IntIdentity::Redirect(right));
            }
        }
        BinaryOperator::InclusiveOr => {
            if is_int_zero(right_expr) && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
            if is_int_zero(left_expr) && right_matches_ty {
                return Some(IntIdentity::Redirect(right));
            }
            if left == right && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
        }
        BinaryOperator::ExclusiveOr => {
            if is_int_zero(right_expr) && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
            if is_int_zero(left_expr) && right_matches_ty {
                return Some(IntIdentity::Redirect(right));
            }
            if left == right && is_pure_expr_node(left_expr) {
                match *res_ty {
                    TypeInner::Scalar(Scalar::U32) => {
                        return Some(IntIdentity::Literal(Literal::U32(0)));
                    }
                    TypeInner::Scalar(Scalar::I32) => {
                        return Some(IntIdentity::Literal(Literal::I32(0)));
                    }
                    _ => {}
                }
            }
        }
        BinaryOperator::And => {
            if is_int_all_ones(right_expr) && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
            if is_int_all_ones(left_expr) && right_matches_ty {
                return Some(IntIdentity::Redirect(right));
            }
            if left == right && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
        }
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
            if matches!(right_expr, Expression::Literal(Literal::U32(0))) && left_matches_ty {
                return Some(IntIdentity::Redirect(left));
            }
        }
        _ => {}
    }

    None
}

// ---------------------------------------------------------------------------
// Pass 2: Dead local-variable pruning
// ---------------------------------------------------------------------------

/// Compute, for every expression handle `h`, whether it is a pointer derived strictly from a
/// [`LocalVariable`] (directly via `Expression::LocalVariable` or via `Access`/`AccessIndex`).
fn compute_local_ptr_origins(func: &Function) -> Vec<Option<Handle<LocalVariable>>> {
    let mut origins: Vec<Option<Handle<LocalVariable>>> = vec![None; func.expressions.len()];
    for (h, expr) in func.expressions.iter() {
        origins[h.index()] = match expr {
            Expression::LocalVariable(lv) => Some(*lv),
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                origins[base.index()]
            }
            _ => None,
        };
    }
    origins
}

/// Compute `transitively_pure[h]`: `true` iff `h` and all its transitive operands are pure.
fn compute_transitively_pure(func: &Function) -> Vec<bool> {
    let mut pure = vec![false; func.expressions.len()];
    for (h, expr) in func.expressions.iter() {
        if !is_pure_expr_node(expr) {
            continue;
        }
        let mut all_children_pure = true;
        visit_expr_children(expr, |c| {
            if !pure[c.index()] {
                all_children_pure = false;
            }
        });
        pure[h.index()] = all_children_pure;
    }
    pure
}

/// Mark every `LocalVariable` that MUST be kept alive:
/// * Any local whose pointer (or derived sub-pointer) is loaded, passed as a `Call` argument,
///   used in `Atomic`, `WorkGroupUniformLoad`, `RayQuery`, `RayPipelineFunction`,
///   `CooperativeStore`, or used in ANY expression position other than `Access.base` /
///   `AccessIndex.base` / `Store.pointer`.
/// * Any local written by a `Statement::Store` whose value (or pointer index chain) is NOT
///   transitively pure.
fn collect_live_locals(
    func: &Function,
    local_origins: &[Option<Handle<LocalVariable>>],
    transitively_pure: &[bool],
) -> HashSet<Handle<LocalVariable>> {
    let mut live_locals: HashSet<Handle<LocalVariable>> = HashSet::new();
    let mut visited_exprs: HashSet<Handle<Expression>> = HashSet::new();

    // Walk an expression used in a non-store-target position: any local pointer reached here
    // escapes or is read -> mark its origin local live!
    fn mark_expr_value_use(
        root: Handle<Expression>,
        func: &Function,
        local_origins: &[Option<Handle<LocalVariable>>],
        visited: &mut HashSet<Handle<Expression>>,
        live_locals: &mut HashSet<Handle<LocalVariable>>,
    ) {
        let mut stack = vec![root];
        while let Some(h) = stack.pop() {
            if let Some(lv) = local_origins[h.index()] {
                live_locals.insert(lv);
            }
            if !visited.insert(h) {
                continue;
            }
            if let Ok(expr) = func.expressions.try_get(h) {
                visit_expr_children(expr, |c| stack.push(c));
            }
        }
    }

    // Walk a `Statement::Store { pointer, value }` target pointer chain (`Access`/`AccessIndex`
    // rooted at `LocalVariable(lv)`): mark any index expressions as value uses and check whether
    // all indices are transitively pure.
    fn check_store_pointer_chain(
        mut ptr: Handle<Expression>,
        func: &Function,
        local_origins: &[Option<Handle<LocalVariable>>],
        transitively_pure: &[bool],
        visited: &mut HashSet<Handle<Expression>>,
        live_locals: &mut HashSet<Handle<LocalVariable>>,
    ) -> bool {
        let mut all_indices_pure = true;
        loop {
            match func.expressions[ptr] {
                Expression::LocalVariable(_) => return all_indices_pure,
                Expression::Access { base, index } => {
                    if !transitively_pure[index.index()] {
                        all_indices_pure = false;
                    }
                    mark_expr_value_use(index, func, local_origins, visited, live_locals);
                    ptr = base;
                }
                Expression::AccessIndex { base, .. } => {
                    ptr = base;
                }
                _ => {
                    mark_expr_value_use(ptr, func, local_origins, visited, live_locals);
                    return false;
                }
            }
        }
    }

    fn walk_block(
        block: &Block,
        func: &Function,
        local_origins: &[Option<Handle<LocalVariable>>],
        transitively_pure: &[bool],
        visited: &mut HashSet<Handle<Expression>>,
        live_locals: &mut HashSet<Handle<LocalVariable>>,
    ) {
        for stmt in block.iter() {
            match stmt {
                Statement::Emit(range) => {
                    // Emitted `Load` / non-pure expressions will stay emitted, so any local
                    // pointer they use MUST remain alive. Pure `Access`/`AccessIndex` on a local
                    // pointer does not by itself read the local unless consumed elsewhere.
                    for h in range.clone() {
                        if let Ok(expr) = func.expressions.try_get(h)
                            && !is_pure_expr_node(expr)
                        {
                            mark_expr_value_use(h, func, local_origins, visited, live_locals);
                        }
                    }
                }
                Statement::Block(inner) => {
                    walk_block(
                        inner,
                        func,
                        local_origins,
                        transitively_pure,
                        visited,
                        live_locals,
                    );
                }
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    mark_expr_value_use(*condition, func, local_origins, visited, live_locals);
                    walk_block(
                        accept,
                        func,
                        local_origins,
                        transitively_pure,
                        visited,
                        live_locals,
                    );
                    walk_block(
                        reject,
                        func,
                        local_origins,
                        transitively_pure,
                        visited,
                        live_locals,
                    );
                }
                Statement::Switch { selector, cases } => {
                    mark_expr_value_use(*selector, func, local_origins, visited, live_locals);
                    for case in cases {
                        walk_block(
                            &case.body,
                            func,
                            local_origins,
                            transitively_pure,
                            visited,
                            live_locals,
                        );
                    }
                }
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    walk_block(
                        body,
                        func,
                        local_origins,
                        transitively_pure,
                        visited,
                        live_locals,
                    );
                    walk_block(
                        continuing,
                        func,
                        local_origins,
                        transitively_pure,
                        visited,
                        live_locals,
                    );
                    if let Some(cond) = break_if {
                        mark_expr_value_use(*cond, func, local_origins, visited, live_locals);
                    }
                }
                Statement::Return { value } => {
                    if let Some(v) = value {
                        mark_expr_value_use(*v, func, local_origins, visited, live_locals);
                    }
                }
                Statement::Store { pointer, value } => {
                    if let Some(lv) = local_origins[pointer.index()] {
                        let ptr_indices_pure = check_store_pointer_chain(
                            *pointer,
                            func,
                            local_origins,
                            transitively_pure,
                            visited,
                            live_locals,
                        );
                        mark_expr_value_use(*value, func, local_origins, visited, live_locals);
                        if !ptr_indices_pure || !transitively_pure[value.index()] {
                            live_locals.insert(lv);
                        }
                    } else {
                        mark_expr_value_use(*pointer, func, local_origins, visited, live_locals);
                        mark_expr_value_use(*value, func, local_origins, visited, live_locals);
                    }
                }
                Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    mark_expr_value_use(*image, func, local_origins, visited, live_locals);
                    mark_expr_value_use(*coordinate, func, local_origins, visited, live_locals);
                    if let Some(a) = array_index {
                        mark_expr_value_use(*a, func, local_origins, visited, live_locals);
                    }
                    mark_expr_value_use(*value, func, local_origins, visited, live_locals);
                }
                Statement::ImageAtomic {
                    image,
                    coordinate,
                    array_index,
                    value,
                    ..
                } => {
                    mark_expr_value_use(*image, func, local_origins, visited, live_locals);
                    mark_expr_value_use(*coordinate, func, local_origins, visited, live_locals);
                    if let Some(a) = array_index {
                        mark_expr_value_use(*a, func, local_origins, visited, live_locals);
                    }
                    mark_expr_value_use(*value, func, local_origins, visited, live_locals);
                }
                Statement::Atomic {
                    pointer,
                    fun,
                    value,
                    ..
                } => {
                    mark_expr_value_use(*pointer, func, local_origins, visited, live_locals);
                    if let AtomicFunction::Exchange {
                        compare: Some(cmp),
                    } = fun
                    {
                        mark_expr_value_use(*cmp, func, local_origins, visited, live_locals);
                    }
                    mark_expr_value_use(*value, func, local_origins, visited, live_locals);
                }
                Statement::WorkGroupUniformLoad { pointer, .. } => {
                    mark_expr_value_use(*pointer, func, local_origins, visited, live_locals);
                }
                Statement::Call { arguments, .. } => {
                    for arg in arguments {
                        mark_expr_value_use(*arg, func, local_origins, visited, live_locals);
                    }
                }
                Statement::RayQuery { query, fun } => {
                    mark_expr_value_use(*query, func, local_origins, visited, live_locals);
                    match fun {
                        RayQueryFunction::Initialize {
                            acceleration_structure,
                            descriptor,
                        } => {
                            mark_expr_value_use(
                                *acceleration_structure,
                                func,
                                local_origins,
                                visited,
                                live_locals,
                            );
                            mark_expr_value_use(
                                *descriptor,
                                func,
                                local_origins,
                                visited,
                                live_locals,
                            );
                        }
                        RayQueryFunction::GenerateIntersection { hit_t } => {
                            mark_expr_value_use(*hit_t, func, local_origins, visited, live_locals);
                        }
                        RayQueryFunction::Proceed { .. }
                        | RayQueryFunction::ConfirmIntersection
                        | RayQueryFunction::Terminate => {}
                    }
                }
                Statement::SubgroupBallot { predicate, .. } => {
                    if let Some(p) = predicate {
                        mark_expr_value_use(*p, func, local_origins, visited, live_locals);
                    }
                }
                Statement::SubgroupCollectiveOperation { argument, .. } => {
                    mark_expr_value_use(*argument, func, local_origins, visited, live_locals);
                }
                Statement::SubgroupGather { mode, argument, .. } => {
                    mark_expr_value_use(*argument, func, local_origins, visited, live_locals);
                    match mode {
                        GatherMode::Broadcast(idx)
                        | GatherMode::Shuffle(idx)
                        | GatherMode::ShuffleDown(idx)
                        | GatherMode::ShuffleUp(idx)
                        | GatherMode::ShuffleXor(idx)
                        | GatherMode::QuadBroadcast(idx) => {
                            mark_expr_value_use(*idx, func, local_origins, visited, live_locals);
                        }
                        GatherMode::BroadcastFirst | GatherMode::QuadSwap(_) => {}
                    }
                }
                Statement::CooperativeStore { target, data } => {
                    mark_expr_value_use(*target, func, local_origins, visited, live_locals);
                    mark_expr_value_use(data.pointer, func, local_origins, visited, live_locals);
                    mark_expr_value_use(data.stride, func, local_origins, visited, live_locals);
                }
                Statement::RayPipelineFunction(RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                }) => {
                    mark_expr_value_use(
                        *acceleration_structure,
                        func,
                        local_origins,
                        visited,
                        live_locals,
                    );
                    mark_expr_value_use(*descriptor, func, local_origins, visited, live_locals);
                    mark_expr_value_use(*payload, func, local_origins, visited, live_locals);
                }
                Statement::Break
                | Statement::Continue
                | Statement::Kill
                | Statement::ControlBarrier(_)
                | Statement::MemoryBarrier(_) => {}
            }
        }
    }

    walk_block(
        &func.body,
        func,
        local_origins,
        transitively_pure,
        &mut visited_exprs,
        &mut live_locals,
    );
    for (lv, local) in func.local_variables.iter() {
        if let Some(init_h) = local.init {
            if !transitively_pure[init_h.index()] {
                live_locals.insert(lv);
            }
            mark_expr_value_use(init_h, func, local_origins, &mut visited_exprs, &mut live_locals);
        }
    }
    live_locals
}

fn strip_dead_local_stores(
    block: &mut Block,
    local_origins: &[Option<Handle<LocalVariable>>],
    dead_locals: &HashSet<Handle<LocalVariable>>,
) {
    let old_block = std::mem::replace(block, Block::new());
    let mut new_block = Block::with_capacity(old_block.len());
    for (stmt, span) in old_block.span_into_iter() {
        match stmt {
            Statement::Store { pointer, .. }
                if local_origins[pointer.index()]
                    .is_some_and(|lv| dead_locals.contains(&lv)) =>
            {
                // Drop store to dead local variable.
            }
            Statement::Block(mut inner) => {
                strip_dead_local_stores(&mut inner, local_origins, dead_locals);
                new_block.push(Statement::Block(inner), span);
            }
            Statement::If {
                condition,
                mut accept,
                mut reject,
            } => {
                strip_dead_local_stores(&mut accept, local_origins, dead_locals);
                strip_dead_local_stores(&mut reject, local_origins, dead_locals);
                new_block.push(
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    span,
                );
            }
            Statement::Switch {
                selector,
                mut cases,
            } => {
                for case in &mut cases {
                    strip_dead_local_stores(&mut case.body, local_origins, dead_locals);
                }
                new_block.push(Statement::Switch { selector, cases }, span);
            }
            Statement::Loop {
                mut body,
                mut continuing,
                break_if,
            } => {
                strip_dead_local_stores(&mut body, local_origins, dead_locals);
                strip_dead_local_stores(&mut continuing, local_origins, dead_locals);
                new_block.push(
                    Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    },
                    span,
                );
            }
            other => new_block.push(other, span),
        }
    }
    *block = new_block;
}

fn prune_dead_locals(func: &mut Function, stats: &mut IrOptStats) {
    if func.local_variables.is_empty() {
        return;
    }

    let local_origins = compute_local_ptr_origins(func);
    let transitively_pure = compute_transitively_pure(func);
    let live_locals = collect_live_locals(func, &local_origins, &transitively_pure);

    if live_locals.len() == func.local_variables.len() {
        return;
    }

    let mut dead_locals: HashSet<Handle<LocalVariable>> = HashSet::new();
    for (lv, _) in func.local_variables.iter() {
        if !live_locals.contains(&lv) {
            dead_locals.insert(lv);
        }
    }
    if dead_locals.is_empty() {
        return;
    }

    // Remove all `Store` statements targeting dead locals first.
    strip_dead_local_stores(&mut func.body, &local_origins, &dead_locals);

    // Rebuild `func.local_variables` with only live locals and remap surviving
    // `Expression::LocalVariable` handles. (Any dead `Expression::LocalVariable` /
    // `Access`/`AccessIndex` expressions are unreachable and will be dropped immediately in
    // `compact_live_expressions_and_emits`, after which no dangling local handles exist.)
    let mut new_locals: Arena<LocalVariable> = Arena::new();
    let mut local_remap: HashMap<Handle<LocalVariable>, Handle<LocalVariable>> = HashMap::new();
    for (old_lv, var) in func.local_variables.iter() {
        if live_locals.contains(&old_lv) {
            let new_lv = new_locals.append(var.clone(), Span::UNDEFINED);
            local_remap.insert(old_lv, new_lv);
        }
    }

    for (_, expr) in func.expressions.iter_mut() {
        if let Expression::LocalVariable(ref mut lv) = *expr
            && let Some(&mapped) = local_remap.get(lv)
        {
            *lv = mapped;
        }
    }

    stats.dead_locals_removed += dead_locals.len();
    func.local_variables = new_locals;
}

// ---------------------------------------------------------------------------
// Pass 3: Expression liveness, Emit trimming, and arena compaction
// ---------------------------------------------------------------------------

fn mark_live_expr(
    h: Handle<Expression>,
    live: &mut [bool],
    worklist: &mut Vec<Handle<Expression>>,
) {
    if !live[h.index()] {
        live[h.index()] = true;
        worklist.push(h);
    }
}

fn collect_live_expressions(func: &Function) -> Vec<bool> {
    let n = func.expressions.len();
    let mut live = vec![false; n];
    let mut worklist: Vec<Handle<Expression>> = Vec::new();

    fn collect_block_roots(
        block: &Block,
        func: &Function,
        live: &mut [bool],
        worklist: &mut Vec<Handle<Expression>>,
    ) {
        for stmt in block.iter() {
            match stmt {
                Statement::Emit(range) => {
                    // Keep every non-pure expression emitted in `range` (`Load`, `ImageSample`,
                    // `ImageLoad`, `ImageQuery`, `Derivative`, etc.). Pure expressions in `Emit`
                    // are kept ONLY if transitively referenced by a live root.
                    for h in range.clone() {
                        if let Ok(expr) = func.expressions.try_get(h)
                            && !is_pure_expr_node(expr)
                        {
                            mark_live_expr(h, live, worklist);
                        }
                    }
                }
                Statement::Block(inner) => collect_block_roots(inner, func, live, worklist),
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    mark_live_expr(*condition, live, worklist);
                    collect_block_roots(accept, func, live, worklist);
                    collect_block_roots(reject, func, live, worklist);
                }
                Statement::Switch { selector, cases } => {
                    mark_live_expr(*selector, live, worklist);
                    for case in cases {
                        collect_block_roots(&case.body, func, live, worklist);
                    }
                }
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    collect_block_roots(body, func, live, worklist);
                    collect_block_roots(continuing, func, live, worklist);
                    if let Some(cond) = break_if {
                        mark_live_expr(*cond, live, worklist);
                    }
                }
                Statement::Return { value } => {
                    if let Some(v) = value {
                        mark_live_expr(*v, live, worklist);
                    }
                }
                Statement::Store { pointer, value } => {
                    mark_live_expr(*pointer, live, worklist);
                    mark_live_expr(*value, live, worklist);
                }
                Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    mark_live_expr(*image, live, worklist);
                    mark_live_expr(*coordinate, live, worklist);
                    if let Some(a) = array_index {
                        mark_live_expr(*a, live, worklist);
                    }
                    mark_live_expr(*value, live, worklist);
                }
                Statement::ImageAtomic {
                    image,
                    coordinate,
                    array_index,
                    value,
                    ..
                } => {
                    mark_live_expr(*image, live, worklist);
                    mark_live_expr(*coordinate, live, worklist);
                    if let Some(a) = array_index {
                        mark_live_expr(*a, live, worklist);
                    }
                    mark_live_expr(*value, live, worklist);
                }
                Statement::Atomic {
                    pointer,
                    fun,
                    value,
                    result,
                } => {
                    mark_live_expr(*pointer, live, worklist);
                    if let AtomicFunction::Exchange {
                        compare: Some(cmp),
                    } = fun
                    {
                        mark_live_expr(*cmp, live, worklist);
                    }
                    mark_live_expr(*value, live, worklist);
                    if let Some(r) = result {
                        mark_live_expr(*r, live, worklist);
                    }
                }
                Statement::WorkGroupUniformLoad { pointer, result } => {
                    mark_live_expr(*pointer, live, worklist);
                    mark_live_expr(*result, live, worklist);
                }
                Statement::Call {
                    arguments, result, ..
                } => {
                    for arg in arguments {
                        mark_live_expr(*arg, live, worklist);
                    }
                    if let Some(r) = result {
                        mark_live_expr(*r, live, worklist);
                    }
                }
                Statement::RayQuery { query, fun } => {
                    mark_live_expr(*query, live, worklist);
                    match fun {
                        RayQueryFunction::Initialize {
                            acceleration_structure,
                            descriptor,
                        } => {
                            mark_live_expr(*acceleration_structure, live, worklist);
                            mark_live_expr(*descriptor, live, worklist);
                        }
                        RayQueryFunction::Proceed { result } => {
                            mark_live_expr(*result, live, worklist);
                        }
                        RayQueryFunction::GenerateIntersection { hit_t } => {
                            mark_live_expr(*hit_t, live, worklist);
                        }
                        RayQueryFunction::ConfirmIntersection | RayQueryFunction::Terminate => {}
                    }
                }
                Statement::SubgroupBallot { result, predicate } => {
                    mark_live_expr(*result, live, worklist);
                    if let Some(p) = predicate {
                        mark_live_expr(*p, live, worklist);
                    }
                }
                Statement::SubgroupCollectiveOperation {
                    argument, result, ..
                } => {
                    mark_live_expr(*argument, live, worklist);
                    mark_live_expr(*result, live, worklist);
                }
                Statement::SubgroupGather {
                    mode,
                    argument,
                    result,
                } => {
                    mark_live_expr(*argument, live, worklist);
                    mark_live_expr(*result, live, worklist);
                    match mode {
                        GatherMode::Broadcast(idx)
                        | GatherMode::Shuffle(idx)
                        | GatherMode::ShuffleDown(idx)
                        | GatherMode::ShuffleUp(idx)
                        | GatherMode::ShuffleXor(idx)
                        | GatherMode::QuadBroadcast(idx) => mark_live_expr(*idx, live, worklist),
                        GatherMode::BroadcastFirst | GatherMode::QuadSwap(_) => {}
                    }
                }
                Statement::CooperativeStore { target, data } => {
                    mark_live_expr(*target, live, worklist);
                    mark_live_expr(data.pointer, live, worklist);
                    mark_live_expr(data.stride, live, worklist);
                }
                Statement::RayPipelineFunction(RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                }) => {
                    mark_live_expr(*acceleration_structure, live, worklist);
                    mark_live_expr(*descriptor, live, worklist);
                    mark_live_expr(*payload, live, worklist);
                }
                Statement::Break
                | Statement::Continue
                | Statement::Kill
                | Statement::ControlBarrier(_)
                | Statement::MemoryBarrier(_) => {}
            }
        }
    }

    collect_block_roots(&func.body, func, &mut live, &mut worklist);
    for (_, local) in func.local_variables.iter() {
        if let Some(init_h) = local.init {
            mark_live_expr(init_h, &mut live, &mut worklist);
        }
    }

    // Transitively mark all operands of live expressions.
    while let Some(h) = worklist.pop() {
        if let Ok(expr) = func.expressions.try_get(h) {
            visit_expr_children(expr, |c| mark_live_expr(c, &mut live, &mut worklist));
        }
    }

    live
}

fn rewrite_block_emits_and_handles(
    block: &mut Block,
    old_to_new: &[Option<Handle<Expression>>],
    new_exprs: &Arena<Expression>,
    dead_emits_trimmed: &mut usize,
) {
    let old_block = std::mem::replace(block, Block::new());
    let mut new_block = Block::with_capacity(old_block.len());
    let map = |h: Handle<Expression>| {
        old_to_new[h.index()].expect("live statement expression must have a compacted handle")
    };

    for (stmt, span) in old_block.span_into_iter() {
        match stmt {
            Statement::Emit(range) => {
                let mut surviving: Vec<Handle<Expression>> = Vec::new();
                let mut total_in_range = 0usize;
                for old_h in range {
                    total_in_range += 1;
                    if let Some(new_h) = old_to_new[old_h.index()]
                        && is_emittable_expr(&new_exprs[new_h])
                    {
                        surviving.push(new_h);
                    }
                }
                let trimmed = total_in_range.saturating_sub(surviving.len());
                *dead_emits_trimmed += trimmed;

                // Group surviving emittable handles into maximal contiguous ranges in `new_exprs`.
                let mut i = 0;
                while i < surviving.len() {
                    let first = surviving[i];
                    let mut last = first;
                    i += 1;
                    while i < surviving.len() && surviving[i].index() == last.index() + 1 {
                        last = surviving[i];
                        i += 1;
                    }
                    new_block.push(Statement::Emit(Range::new_from_bounds(first, last)), span);
                }
            }
            Statement::Block(mut inner) => {
                rewrite_block_emits_and_handles(
                    &mut inner,
                    old_to_new,
                    new_exprs,
                    dead_emits_trimmed,
                );
                new_block.push(Statement::Block(inner), span);
            }
            Statement::If {
                condition,
                mut accept,
                mut reject,
            } => {
                rewrite_block_emits_and_handles(
                    &mut accept,
                    old_to_new,
                    new_exprs,
                    dead_emits_trimmed,
                );
                rewrite_block_emits_and_handles(
                    &mut reject,
                    old_to_new,
                    new_exprs,
                    dead_emits_trimmed,
                );
                new_block.push(
                    Statement::If {
                        condition: map(condition),
                        accept,
                        reject,
                    },
                    span,
                );
            }
            Statement::Switch { selector, cases } => {
                let new_cases: Vec<SwitchCase> = cases
                    .into_iter()
                    .map(|mut c| {
                        rewrite_block_emits_and_handles(
                            &mut c.body,
                            old_to_new,
                            new_exprs,
                            dead_emits_trimmed,
                        );
                        c
                    })
                    .collect();
                new_block.push(
                    Statement::Switch {
                        selector: map(selector),
                        cases: new_cases,
                    },
                    span,
                );
            }
            Statement::Loop {
                mut body,
                mut continuing,
                break_if,
            } => {
                rewrite_block_emits_and_handles(
                    &mut body,
                    old_to_new,
                    new_exprs,
                    dead_emits_trimmed,
                );
                rewrite_block_emits_and_handles(
                    &mut continuing,
                    old_to_new,
                    new_exprs,
                    dead_emits_trimmed,
                );
                new_block.push(
                    Statement::Loop {
                        body,
                        continuing,
                        break_if: break_if.map(map),
                    },
                    span,
                );
            }
            mut other => {
                // Single-statement remap via helper block (zero duplication across Statement arms).
                let mut tmp = Block::with_capacity(1);
                tmp.push(other, span);
                remap_block_exprs(&mut tmp, true, &map);
                if let Some((remapped, remapped_span)) = tmp.span_into_iter().next() {
                    other = remapped;
                    new_block.push(other, remapped_span);
                }
            }
        }
    }

    *block = new_block;
}

fn compact_live_expressions_and_emits(func: &mut Function, stats: &mut IrOptStats) {
    let n = func.expressions.len();
    if n == 0 {
        return;
    }

    let live = collect_live_expressions(func);

    // Check whether compaction or emit trimming is needed:
    //  * any dead expression exists, OR
    //  * any `Statement::Emit` contains a non-emittable handle (e.g. an expression folded in-place
    //    to `Literal`) or an empty range.
    let has_dead_expr = live.iter().any(|&is_live| !is_live);
    let has_dirty_emit = block_has_dirty_emit(&func.body, &func.expressions);
    if !has_dead_expr && !has_dirty_emit {
        return;
    }

    // Rebuild `func.expressions` in ascending handle order so topological order is strictly kept.
    let mut new_exprs: Arena<Expression> = Arena::new();
    let mut old_to_new: Vec<Option<Handle<Expression>>> = vec![None; n];

    for (old_h, expr) in func.expressions.iter() {
        if !live[old_h.index()] {
            continue;
        }
        let mut cloned = expr.clone();
        remap_expr_children(&mut cloned, |c| {
            old_to_new[c.index()].expect("operand of live expression must already be compacted")
        });
        let span = func.expressions.get_span(old_h);
        let new_h = new_exprs.append(cloned, span);
        old_to_new[old_h.index()] = Some(new_h);
    }

    // Rewrite `Statement::Emit` ranges and remap all statement expression handles (including results).
    rewrite_block_emits_and_handles(
        &mut func.body,
        &old_to_new,
        &new_exprs,
        &mut stats.dead_emits_trimmed,
    );

    // Remap surviving `LocalVariable.init` expression handles.
    for (_, local) in func.local_variables.iter_mut() {
        if let Some(ref mut init_h) = local.init {
            *init_h = old_to_new[init_h.index()]
                .expect("live local variable initializer expression must have a compacted handle");
        }
    }

    // Remap `func.named_expressions` (dropping dead or newly non-emittable expressions).
    let old_named = std::mem::take(&mut func.named_expressions);
    for (old_h, name) in old_named {
        if let Some(new_h) = old_to_new[old_h.index()]
            && is_emittable_expr(&new_exprs[new_h])
        {
            func.named_expressions.insert(new_h, name);
        }
    }

    func.expressions = new_exprs;
}

fn block_has_dirty_emit(block: &Block, exprs: &Arena<Expression>) -> bool {
    for stmt in block.iter() {
        match stmt {
            Statement::Emit(range) => {
                if range.first_and_last().is_none() {
                    return true;
                }
                for h in range.clone() {
                    if let Ok(e) = exprs.try_get(h)
                        && !is_emittable_expr(e)
                    {
                        return true;
                    }
                }
            }
            Statement::Block(inner) => {
                if block_has_dirty_emit(inner, exprs) {
                    return true;
                }
            }
            Statement::If { accept, reject, .. } => {
                if block_has_dirty_emit(accept, exprs) || block_has_dirty_emit(reject, exprs) {
                    return true;
                }
            }
            Statement::Switch { cases, .. } => {
                if cases.iter().any(|c| block_has_dirty_emit(&c.body, exprs)) {
                    return true;
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                if block_has_dirty_emit(body, exprs) || block_has_dirty_emit(continuing, exprs) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}
