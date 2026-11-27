#![forbid(unsafe_code)]

//! Post-lowering semantics-preserving Naga IR optimization pass (`#2137`, `#2164`, Refs `#663`, `#482`).
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
//! 4. **Intra-block Store-to-Load local variable forwarding (`#2164`):**
//!    * Tracks non-escaped `LocalVariable`s across straight-line `Block`s (including pre-emitted
//!      `LocalVariable.init` values at function entry), resetting state inside `If`/`Loop`/`Switch`
//!      sub-blocks and invalidating written locals and load-dependent values across intervening
//!      stores, calls, atomics, barriers, and control flow.
//!    * Forwards dominating SSA values directly into `Expression::Load { pointer: LocalVariable }`,
//!      enabling dead-local pruning and further bitcast/constant simplification on the next pass.
//! 5. **Intra-block Common Subexpression Elimination (CSE) for pure expressions (`#2164`):**
//!    * Scoped dominance-respecting value numbering over pure value expressions (`Binary`, `Unary`,
//!      `As`, `Splat`, `Compose`, `Swizzle`, value `Access`/`AccessIndex`, `Select`, `Math`,
//!      `Relational`), normalizing commutative integer/boolean binary operators while never CSEing
//!      memory loads, texture samples/loads, derivatives, or statement-result expressions.
//! 6. **Dead local-variable pruning & unreferenced pure `Emit` trimming + arena compaction:**
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
    RelationalFunction, SampleLevel, Scalar, ScalarKind, Span, Statement, StorageAccess,
    StorageFormat, SwitchCase, SwizzleComponent, Type, TypeInner, UnaryOperator, UniqueArena,
    VectorSize,
};

/// Maximum fixed-point optimization iterations per function.
const MAX_OPT_PASSES: usize = 8;

/// Sanitized counters summarizing Naga IR optimizations applied to a shader module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IrOptStats {
    /// Identity (`bitcast<T>(x) -> x`) and round-trip (`bitcast<T>(bitcast<U>(x)) -> x`) casts cancelled.
    pub bitcasts_cancelled: usize,
    /// Pure constant expressions folded (`Binary`, `Unary`, `Select`, `Splat`/`Compose` + `AccessIndex`, `As`).
    pub constants_folded: usize,
    /// Safe integer/boolean algebraic identities simplified (`x + 0`, `x * 1`, `x * 0`, `x & 0`, `x | 0`, `x ^ 0`, `x & ~0`, `x << 0`, `x >> 0`, `x ^ x`, `x & x`, `x | x`, `select(a, a, c)`, `!!b`, `!(a == b)`).
    pub identities_simplified: usize,
    /// Constant-condition `If` branches folded, empty pure `If`s pruned, and unreachable tails trimmed after `Return`/`Break`/`Continue`/`Kill`.
    pub dead_branches_folded: usize,
    /// Local variable `Load` expressions forwarded directly from dominating intra-block stores or initializers.
    pub locals_forwarded: usize,
    /// Duplicate pure expressions deduplicated by intra-block Common Subexpression Elimination (CSE).
    pub cse_eliminated: usize,
    /// Duplicate immutable `Load` expressions from read-only `@group/@binding` globals (`Uniform` / `Storage(LOAD)`) deduplicated by intra-block CSE.
    pub readonly_loads_cse: usize,
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
            + self.dead_branches_folded
            + self.locals_forwarded
            + self.cse_eliminated
            + self.readonly_loads_cse
            + self.dead_locals_removed
            + self.dead_emits_trimmed
    }

    /// `true` when no optimization fired.
    pub const fn is_zero(&self) -> bool {
        self.bitcasts_cancelled == 0
            && self.constants_folded == 0
            && self.identities_simplified == 0
            && self.dead_branches_folded == 0
            && self.locals_forwarded == 0
            && self.cse_eliminated == 0
            && self.readonly_loads_cse == 0
            && self.dead_locals_removed == 0
            && self.dead_emits_trimmed == 0
    }

    /// Accumulate counters from `other` into `self`.
    pub fn merge_from(&mut self, other: Self) {
        self.bitcasts_cancelled += other.bitcasts_cancelled;
        self.constants_folded += other.constants_folded;
        self.identities_simplified += other.identities_simplified;
        self.dead_branches_folded += other.dead_branches_folded;
        self.locals_forwarded += other.locals_forwarded;
        self.cse_eliminated += other.cse_eliminated;
        self.readonly_loads_cse += other.readonly_loads_cse;
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

        // 2. Constant-condition dead-branch folding & unreachable statement trimming (#2178).
        fold_dead_branches_and_trim_unreachable(func, &mut pass);

        // 3. Intra-block Store-to-Load local variable forwarding (#2164).
        forward_local_loads(ctx, func, &mut pass);

        // 4. Intra-block Common Subexpression Elimination (CSE) for pure expressions and
        //    immutable read-only global loads (#2164, #2178).
        eliminate_common_subexpressions(ctx, func, &mut pass);

        // 5. Dead local-variable pruning (locals never loaded or escaped by pointer, whose
        //    stores have transitively pure values and indices).
        prune_dead_locals(func, &mut pass);

        // 6. Dead pure-expression Emit trimming and order-preserving expression arena compaction.
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

    // Track handles directly consumed by `As { kind: Float, convert: None }`, float `Binary` ops,
    // or integer `Divide`/`Modulo` divisors so folding an expression to a bare `Literal` never
    // creates a Tint-rejected const-eval expression (e.g., `bitcast<f32>(0x7f800000u)`,
    // `1e38f * 10.0f`, or `10u / 0u`).
    let mut used_by_float_bitcast = vec![false; n];
    let mut used_by_float_binary = vec![false; n];
    let mut used_as_divisor = vec![false; n];
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
                    op @ (BinaryOperator::Add
                    | BinaryOperator::Subtract
                    | BinaryOperator::Multiply
                    | BinaryOperator::Divide
                    | BinaryOperator::Modulo),
                left,
                right,
            } => {
                used_by_float_binary[left.index()] = true;
                used_by_float_binary[right.index()] = true;
                if matches!(op, BinaryOperator::Divide | BinaryOperator::Modulo) {
                    used_as_divisor[right.index()] = true;
                }
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
            if used_as_divisor[idx] && matches!(lit, Literal::U32(0) | Literal::I32(0)) {
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
            ReplaceExpr(Expression, RewriteKind),
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
                    // Negated comparison identities strictly on integer/boolean operands (NEVER
                    // floats, where `!(NaN < b)` != `NaN >= b`).
                    (
                        UnaryOperator::LogicalNot,
                        Expression::Binary {
                            op: cmp_op,
                            left,
                            right,
                        },
                    ) => {
                        let cmp_op = *cmp_op;
                        let left = *left;
                        let right = *right;
                        let l_kind = resolved[left.index()]
                            .as_ref()
                            .and_then(value_shape)
                            .map(|(_, s)| s.kind);
                        let r_kind = resolved[right.index()]
                            .as_ref()
                            .and_then(value_shape)
                            .map(|(_, s)| s.kind);
                        let inverted = match (cmp_op, l_kind, r_kind) {
                            (
                                BinaryOperator::Equal,
                                Some(ScalarKind::Uint | ScalarKind::Sint | ScalarKind::Bool),
                                Some(ScalarKind::Uint | ScalarKind::Sint | ScalarKind::Bool),
                            ) if l_kind == r_kind => Some(BinaryOperator::NotEqual),
                            (
                                BinaryOperator::NotEqual,
                                Some(ScalarKind::Uint | ScalarKind::Sint | ScalarKind::Bool),
                                Some(ScalarKind::Uint | ScalarKind::Sint | ScalarKind::Bool),
                            ) if l_kind == r_kind => Some(BinaryOperator::Equal),
                            (
                                BinaryOperator::Less,
                                Some(ScalarKind::Uint | ScalarKind::Sint),
                                Some(ScalarKind::Uint | ScalarKind::Sint),
                            ) if l_kind == r_kind => Some(BinaryOperator::GreaterEqual),
                            (
                                BinaryOperator::LessEqual,
                                Some(ScalarKind::Uint | ScalarKind::Sint),
                                Some(ScalarKind::Uint | ScalarKind::Sint),
                            ) if l_kind == r_kind => Some(BinaryOperator::Greater),
                            (
                                BinaryOperator::Greater,
                                Some(ScalarKind::Uint | ScalarKind::Sint),
                                Some(ScalarKind::Uint | ScalarKind::Sint),
                            ) if l_kind == r_kind => Some(BinaryOperator::LessEqual),
                            (
                                BinaryOperator::GreaterEqual,
                                Some(ScalarKind::Uint | ScalarKind::Sint),
                                Some(ScalarKind::Uint | ScalarKind::Sint),
                            ) if l_kind == r_kind => Some(BinaryOperator::Less),
                            _ => None,
                        };
                        inverted.map(|inv_op| {
                            Rewrite::ReplaceExpr(
                                Expression::Binary {
                                    op: inv_op,
                                    left,
                                    right,
                                },
                                RewriteKind::Identity,
                            )
                        })
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
                Rewrite::ReplaceExpr(new_expr, kind) => {
                    func.expressions[h] = new_expr;
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
            if (is_int_zero(right_expr) && is_pure_expr_node(left_expr))
                || (is_int_zero(left_expr) && is_pure_expr_node(right_expr))
            {
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
            if (is_int_zero(right_expr) && is_pure_expr_node(left_expr))
                || (is_int_zero(left_expr) && is_pure_expr_node(right_expr))
            {
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
// Pass 2: Constant-condition dead-branch folding & unreachable trimming (#2178)
// ---------------------------------------------------------------------------

/// Compute `transitively_pure[h]`: `true` iff `h` and all of its transitive expression operands
/// satisfy [`is_pure_expr_node`].
fn compute_transitively_pure_exprs(func: &Function) -> Vec<bool> {
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

fn stmt_contains_return(stmt: &Statement) -> bool {
    match stmt {
        Statement::Return { .. } => true,
        Statement::Block(inner) => inner.iter().any(stmt_contains_return),
        Statement::If { accept, reject, .. } => {
            accept.iter().any(stmt_contains_return) || reject.iter().any(stmt_contains_return)
        }
        Statement::Switch { cases, .. } => cases
            .iter()
            .any(|c| c.body.iter().any(stmt_contains_return)),
        Statement::Loop {
            body, continuing, ..
        } => body.iter().any(stmt_contains_return) || continuing.iter().any(stmt_contains_return),
        _ => false,
    }
}

fn fold_and_trim_block(
    block: &mut Block,
    exprs: &Arena<Expression>,
    transitively_pure: &[bool],
    func_is_void: bool,
    stats: &mut IrOptStats,
) {
    let old_block = std::mem::replace(block, Block::new());
    let mut folded_stmts: Vec<(Statement, Span)> = Vec::with_capacity(old_block.len());

    for (stmt, span) in old_block.span_into_iter() {
        match stmt {
            Statement::Block(mut inner) => {
                fold_and_trim_block(&mut inner, exprs, transitively_pure, func_is_void, stats);
                if !inner.is_empty() {
                    folded_stmts.push((Statement::Block(inner), span));
                }
            }
            Statement::If {
                condition,
                mut accept,
                mut reject,
            } => {
                fold_and_trim_block(&mut accept, exprs, transitively_pure, func_is_void, stats);
                fold_and_trim_block(&mut reject, exprs, transitively_pure, func_is_void, stats);
                match exprs[condition] {
                    Expression::Literal(Literal::Bool(true)) => {
                        stats.dead_branches_folded += 1;
                        folded_stmts.extend(accept.span_into_iter());
                    }
                    Expression::Literal(Literal::Bool(false)) => {
                        stats.dead_branches_folded += 1;
                        folded_stmts.extend(reject.span_into_iter());
                    }
                    _ if accept.is_empty()
                        && reject.is_empty()
                        && transitively_pure[condition.index()] =>
                    {
                        stats.dead_branches_folded += 1;
                    }
                    _ => {
                        folded_stmts.push((
                            Statement::If {
                                condition,
                                accept,
                                reject,
                            },
                            span,
                        ));
                    }
                }
            }
            Statement::Switch {
                selector,
                mut cases,
            } => {
                for case in &mut cases {
                    fold_and_trim_block(
                        &mut case.body,
                        exprs,
                        transitively_pure,
                        func_is_void,
                        stats,
                    );
                }
                folded_stmts.push((Statement::Switch { selector, cases }, span));
            }
            Statement::Loop {
                mut body,
                mut continuing,
                break_if,
            } => {
                fold_and_trim_block(&mut body, exprs, transitively_pure, func_is_void, stats);
                fold_and_trim_block(
                    &mut continuing,
                    exprs,
                    transitively_pure,
                    func_is_void,
                    stats,
                );
                folded_stmts.push((
                    Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    },
                    span,
                ));
            }
            other => folded_stmts.push((other, span)),
        }
    }

    // Trim unreachable statements following an unconditional terminator (`Return`, `Break`,
    // `Continue`, or `Kill` when dropping the tail cannot remove a `Return` required by WGSL
    // control-flow analysis on non-void functions).
    let mut new_block = Block::with_capacity(folded_stmts.len());
    let total_len = folded_stmts.len();
    for (idx, (stmt, span)) in folded_stmts.iter().enumerate() {
        let is_hard_term = matches!(
            stmt,
            Statement::Return { .. } | Statement::Break | Statement::Continue
        );
        let is_trim_kill = matches!(stmt, Statement::Kill)
            && (func_is_void
                || !folded_stmts[idx + 1..]
                    .iter()
                    .any(|(s, _)| stmt_contains_return(s)));
        new_block.push(stmt.clone(), *span);
        if (is_hard_term || is_trim_kill) && idx + 1 < total_len {
            stats.dead_branches_folded += 1;
            break;
        }
    }

    *block = new_block;
}

fn fold_dead_branches_and_trim_unreachable(func: &mut Function, stats: &mut IrOptStats) {
    if func.body.is_empty() {
        return;
    }
    let transitively_pure = compute_transitively_pure_exprs(func);
    let func_is_void = func.result.is_none();
    fold_and_trim_block(
        &mut func.body,
        &func.expressions,
        &transitively_pure,
        func_is_void,
        stats,
    );
}

// ---------------------------------------------------------------------------
// Pass 3: Intra-block Store-to-Load local variable forwarding (#2164)
// ---------------------------------------------------------------------------

/// Maximum distinct non-escaped locals tracked precisely per expression before falling back to
/// the conservative `reads_any_local` flag (keeps per-expression dependency propagation O(1)).
const MAX_TRACKED_LOCALS: usize = 16;

/// Follow a remapping table to its canonical root handle (bounded to prevent cycles).
#[inline]
fn resolve_remap(remap: &[Handle<Expression>], mut h: Handle<Expression>) -> Handle<Expression> {
    for _ in 0..remap.len() {
        let next = remap[h.index()];
        if next == h {
            return h;
        }
        h = next;
    }
    h
}

/// Infer the [`TypeInner`] of every expression in `func` in ascending arena order.
fn infer_all_expr_types(ctx: &ModuleContext<'_>, func: &Function) -> Vec<Option<TypeInner>> {
    let mut resolved: Vec<Option<TypeInner>> = vec![None; func.expressions.len()];
    for (h, expr) in func.expressions.iter() {
        resolved[h.index()] = infer_single_expr_type(ctx, func, &resolved, expr);
    }
    resolved
}

/// Returns `true` iff `var` is strictly immutable for the entire shader invocation
/// (`AddressSpace::Uniform` or read-only `AddressSpace::Storage { access: StorageAccess::LOAD }`).
#[inline]
fn is_immutable_readonly_global(var: &GlobalVariable) -> bool {
    match var.space {
        AddressSpace::Uniform => true,
        AddressSpace::Storage { access } => access == StorageAccess::LOAD,
        _ => false,
    }
}

/// Compute whether each expression handle is a pointer rooted in an immutable read-only global
/// (`AddressSpace::Uniform` or `AddressSpace::Storage { access: StorageAccess::LOAD }`).
fn compute_uniform_ptr_origins(ctx: &ModuleContext<'_>, func: &Function) -> Vec<bool> {
    let mut is_readonly = vec![false; func.expressions.len()];
    for (h, expr) in func.expressions.iter() {
        is_readonly[h.index()] = match expr {
            Expression::GlobalVariable(g) => {
                is_immutable_readonly_global(&ctx.global_variables[*g])
            }
            Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                is_readonly[base.index()]
            }
            _ => false,
        };
    }
    is_readonly
}

/// Identify all `LocalVariable`s whose address escapes into ANY non-`Load`/`Store`/`Access`
/// position (e.g., `Call` arguments, `Atomic`, `WorkGroupUniformLoad`, `RayQuery`,
/// `RayPipelineFunction`, `CooperativeStore`, `CooperativeLoad`, `ArrayLength`, etc.).
///
/// Escaped locals may be mutated indirectly and are NEVER tracked or forwarded.
fn compute_escaped_locals(
    func: &Function,
    local_origins: &[Option<Handle<LocalVariable>>],
) -> HashSet<Handle<LocalVariable>> {
    let mut escaped: HashSet<Handle<LocalVariable>> = HashSet::new();

    let mut mark_if_local = |h: Handle<Expression>| {
        if let Some(lv) = local_origins[h.index()] {
            escaped.insert(lv);
        }
    };

    // 1. Expression operands: only `Access.base`, `AccessIndex.base`, and `Load.pointer` are
    //    non-escaping local pointer positions.
    for (_, expr) in func.expressions.iter() {
        match expr {
            Expression::LocalVariable(_) | Expression::Load { .. } => {}
            Expression::Access { base: _, index } => {
                mark_if_local(*index);
            }
            Expression::AccessIndex { base: _, .. } => {}
            other => {
                visit_expr_children(other, |c| mark_if_local(c));
            }
        }
    }

    // 2. Statement operands: only `Store.pointer` is a non-escaping local pointer position.
    fn mark_escaped(
        h: Handle<Expression>,
        local_origins: &[Option<Handle<LocalVariable>>],
        escaped: &mut HashSet<Handle<LocalVariable>>,
    ) {
        if let Some(lv) = local_origins[h.index()] {
            escaped.insert(lv);
        }
    }

    fn walk_stmt_escapes(
        block: &Block,
        local_origins: &[Option<Handle<LocalVariable>>],
        escaped: &mut HashSet<Handle<LocalVariable>>,
    ) {
        for stmt in block.iter() {
            match stmt {
                Statement::Emit(_)
                | Statement::Break
                | Statement::Continue
                | Statement::Kill
                | Statement::ControlBarrier(_)
                | Statement::MemoryBarrier(_) => {}
                Statement::Block(inner) => walk_stmt_escapes(inner, local_origins, escaped),
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    mark_escaped(*condition, local_origins, escaped);
                    walk_stmt_escapes(accept, local_origins, escaped);
                    walk_stmt_escapes(reject, local_origins, escaped);
                }
                Statement::Switch { selector, cases } => {
                    mark_escaped(*selector, local_origins, escaped);
                    for case in cases {
                        walk_stmt_escapes(&case.body, local_origins, escaped);
                    }
                }
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    walk_stmt_escapes(body, local_origins, escaped);
                    walk_stmt_escapes(continuing, local_origins, escaped);
                    if let Some(cond) = break_if {
                        mark_escaped(*cond, local_origins, escaped);
                    }
                }
                Statement::Return { value } => {
                    if let Some(v) = value {
                        mark_escaped(*v, local_origins, escaped);
                    }
                }
                Statement::Store { pointer: _, value } => {
                    mark_escaped(*value, local_origins, escaped);
                }
                Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    mark_escaped(*image, local_origins, escaped);
                    mark_escaped(*coordinate, local_origins, escaped);
                    if let Some(a) = array_index {
                        mark_escaped(*a, local_origins, escaped);
                    }
                    mark_escaped(*value, local_origins, escaped);
                }
                Statement::ImageAtomic {
                    image,
                    coordinate,
                    array_index,
                    value,
                    ..
                } => {
                    mark_escaped(*image, local_origins, escaped);
                    mark_escaped(*coordinate, local_origins, escaped);
                    if let Some(a) = array_index {
                        mark_escaped(*a, local_origins, escaped);
                    }
                    mark_escaped(*value, local_origins, escaped);
                }
                Statement::Atomic {
                    pointer,
                    fun,
                    value,
                    ..
                } => {
                    mark_escaped(*pointer, local_origins, escaped);
                    if let AtomicFunction::Exchange {
                        compare: Some(cmp),
                    } = fun
                    {
                        mark_escaped(*cmp, local_origins, escaped);
                    }
                    mark_escaped(*value, local_origins, escaped);
                }
                Statement::WorkGroupUniformLoad { pointer, .. } => {
                    mark_escaped(*pointer, local_origins, escaped);
                }
                Statement::Call { arguments, .. } => {
                    for arg in arguments {
                        mark_escaped(*arg, local_origins, escaped);
                    }
                }
                Statement::RayQuery { query, fun } => {
                    mark_escaped(*query, local_origins, escaped);
                    match fun {
                        RayQueryFunction::Initialize {
                            acceleration_structure,
                            descriptor,
                        } => {
                            mark_escaped(*acceleration_structure, local_origins, escaped);
                            mark_escaped(*descriptor, local_origins, escaped);
                        }
                        RayQueryFunction::GenerateIntersection { hit_t } => {
                            mark_escaped(*hit_t, local_origins, escaped);
                        }
                        RayQueryFunction::Proceed { .. }
                        | RayQueryFunction::ConfirmIntersection
                        | RayQueryFunction::Terminate => {}
                    }
                }
                Statement::SubgroupBallot { predicate, .. } => {
                    if let Some(p) = predicate {
                        mark_escaped(*p, local_origins, escaped);
                    }
                }
                Statement::SubgroupCollectiveOperation { argument, .. } => {
                    mark_escaped(*argument, local_origins, escaped);
                }
                Statement::SubgroupGather { mode, argument, .. } => {
                    mark_escaped(*argument, local_origins, escaped);
                    match mode {
                        GatherMode::Broadcast(idx)
                        | GatherMode::Shuffle(idx)
                        | GatherMode::ShuffleDown(idx)
                        | GatherMode::ShuffleUp(idx)
                        | GatherMode::ShuffleXor(idx)
                        | GatherMode::QuadBroadcast(idx) => {
                            mark_escaped(*idx, local_origins, escaped);
                        }
                        GatherMode::BroadcastFirst | GatherMode::QuadSwap(_) => {}
                    }
                }
                Statement::CooperativeStore { target, data } => {
                    mark_escaped(*target, local_origins, escaped);
                    mark_escaped(data.pointer, local_origins, escaped);
                    mark_escaped(data.stride, local_origins, escaped);
                }
                Statement::RayPipelineFunction(RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                }) => {
                    mark_escaped(*acceleration_structure, local_origins, escaped);
                    mark_escaped(*descriptor, local_origins, escaped);
                    mark_escaped(*payload, local_origins, escaped);
                }
            }
        }
    }

    walk_stmt_escapes(&func.body, local_origins, &mut escaped);

    for (_, local) in func.local_variables.iter() {
        if let Some(init_h) = local.init
            && let Some(lv) = local_origins[init_h.index()]
        {
            escaped.insert(lv);
        }
    }

    escaped
}

/// Transitive memory & uniformity dependency summary for a single SSA expression handle.
#[derive(Debug, Clone, Default)]
struct ExprDeps {
    /// Sorted, deduplicated non-escaped locals loaded transitively (when `!reads_any_local`).
    locals: Vec<Handle<LocalVariable>>,
    /// Conservative overflow flag when > `MAX_TRACKED_LOCALS` distinct locals are read.
    reads_any_local: bool,
    /// Reads mutable non-local memory (storage/workgroup/private/escaped-local `Load`,
    /// `ImageLoad`, `CooperativeLoad`, `WorkGroupUniformLoadResult`, ray query results).
    reads_mutable_mem: bool,
    /// Contains a uniformity- or control-flow-sensitive op (`Derivative`, `ImageSample`,
    /// `Subgroup*`, `CallResult`, `AtomicResult`, etc.).
    has_cf_sensitive_op: bool,
    /// WGSL const-expression (pure tree composed solely of `Literal`/`Constant`/`ZeroValue`).
    is_const_expr: bool,
}

impl ExprDeps {
    fn add_local(&mut self, lv: Handle<LocalVariable>) {
        if self.reads_any_local {
            return;
        }
        if let Err(pos) = self.locals.binary_search(&lv) {
            if self.locals.len() >= MAX_TRACKED_LOCALS {
                self.reads_any_local = true;
                self.locals.clear();
            } else {
                self.locals.insert(pos, lv);
            }
        }
    }

    fn merge_child(&mut self, child: &ExprDeps) {
        self.reads_mutable_mem |= child.reads_mutable_mem;
        self.has_cf_sensitive_op |= child.has_cf_sensitive_op;
        self.is_const_expr &= child.is_const_expr;
        if child.reads_any_local {
            self.reads_any_local = true;
            self.locals.clear();
        } else if !self.reads_any_local {
            for &lv in &child.locals {
                self.add_local(lv);
            }
        }
    }

    #[inline]
    fn reads_local(&self, lv: Handle<LocalVariable>) -> bool {
        self.reads_any_local || self.locals.binary_search(&lv).is_ok()
    }

    #[inline]
    fn reads_any_in(&self, locals: &HashSet<Handle<LocalVariable>>) -> bool {
        if locals.is_empty() {
            return false;
        }
        self.reads_any_local || self.locals.iter().any(|lv| locals.contains(lv))
    }
}

fn compute_single_expr_deps(
    expr: &Expression,
    deps: &[ExprDeps],
    remap: &[Handle<Expression>],
    local_origins: &[Option<Handle<LocalVariable>>],
    escaped_locals: &HashSet<Handle<LocalVariable>>,
    uniform_ptr_origins: &[bool],
) -> ExprDeps {
    match expr {
        Expression::Literal(_) | Expression::Constant(_) | Expression::ZeroValue(_) => ExprDeps {
            is_const_expr: true,
            ..Default::default()
        },
        Expression::Override(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_) => ExprDeps::default(),
        Expression::Load { pointer } => {
            let ptr = resolve_remap(remap, *pointer);
            let mut out = ExprDeps::default();
            out.merge_child(&deps[ptr.index()]);
            out.is_const_expr = false;
            match local_origins[ptr.index()] {
                Some(lv) if !escaped_locals.contains(&lv) => out.add_local(lv),
                Some(_) => out.reads_mutable_mem = true,
                None => {
                    if !uniform_ptr_origins[ptr.index()] {
                        out.reads_mutable_mem = true;
                    }
                }
            }
            out
        }
        Expression::ImageLoad { .. }
        | Expression::CooperativeLoad { .. }
        | Expression::WorkGroupUniformLoadResult { .. }
        | Expression::RayQueryGetIntersection { .. }
        | Expression::RayQueryVertexPositions { .. }
        | Expression::RayQueryProceedResult => {
            let mut out = ExprDeps::default();
            visit_expr_children(expr, |c| {
                let rc = resolve_remap(remap, c);
                out.merge_child(&deps[rc.index()]);
            });
            out.is_const_expr = false;
            out.reads_mutable_mem = true;
            out.has_cf_sensitive_op = true;
            out
        }
        Expression::Derivative { .. }
        | Expression::ImageSample { .. }
        | Expression::SubgroupBallotResult
        | Expression::SubgroupOperationResult { .. }
        | Expression::CallResult(_)
        | Expression::AtomicResult { .. } => {
            let mut out = ExprDeps::default();
            visit_expr_children(expr, |c| {
                let rc = resolve_remap(remap, c);
                out.merge_child(&deps[rc.index()]);
            });
            out.is_const_expr = false;
            out.has_cf_sensitive_op = true;
            out
        }
        Expression::ImageQuery { .. }
        | Expression::CooperativeMultiplyAdd { .. }
        | Expression::ArrayLength(_) => {
            let mut out = ExprDeps::default();
            visit_expr_children(expr, |c| {
                let rc = resolve_remap(remap, c);
                out.merge_child(&deps[rc.index()]);
            });
            out.is_const_expr = false;
            out
        }
        // Pure value/pointer constructors and arithmetic expressions.
        Expression::Access { .. }
        | Expression::AccessIndex { .. }
        | Expression::Splat { .. }
        | Expression::Swizzle { .. }
        | Expression::Compose { .. }
        | Expression::Unary { .. }
        | Expression::Binary { .. }
        | Expression::Select { .. }
        | Expression::Relational { .. }
        | Expression::Math { .. }
        | Expression::As { .. } => {
            let mut out = ExprDeps {
                is_const_expr: true,
                ..Default::default()
            };
            visit_expr_children(expr, |c| {
                let rc = resolve_remap(remap, c);
                out.merge_child(&deps[rc.index()]);
            });
            out
        }
    }
}

/// Precompute conservative WGSL/Tint const-eval guards for forwarding constant expressions into
/// `Load` sites:
/// * `used_by_float_bitcast[h]`: `h` feeds `bitcast<f32>(...)` (must reject NaN/Inf/subnormal/-0.0).
/// * `unsafe_for_const_target[h]`: `h` feeds an op where substituting a constant expression could
///   trigger WGSL const-eval rejection (`Divide`/`Modulo`/`Shift*`, float arithmetic, `Math`,
///   `Relational`, `Access.index`, numeric `As { convert: Some(_) }`, or signed `Unary::Negate`).
fn compute_const_target_guards(
    func: &Function,
    resolved: &[Option<TypeInner>],
) -> (Vec<bool>, Vec<bool>) {
    let n = func.expressions.len();
    let mut used_by_float_bitcast = vec![false; n];
    let mut unsafe_for_const_target = vec![false; n];

    for (h, expr) in func.expressions.iter() {
        match expr {
            Expression::Access { index, .. } => {
                unsafe_for_const_target[index.index()] = true;
            }
            Expression::As {
                expr: inner,
                kind,
                convert,
            } => {
                if convert.is_some() {
                    unsafe_for_const_target[inner.index()] = true;
                } else if *kind == ScalarKind::Float {
                    used_by_float_bitcast[inner.index()] = true;
                }
            }
            Expression::Binary { op, left, right } => {
                let is_dangerous_op = matches!(
                    op,
                    BinaryOperator::Divide
                        | BinaryOperator::Modulo
                        | BinaryOperator::ShiftLeft
                        | BinaryOperator::ShiftRight
                );
                let scalar_kind = resolved[h.index()]
                    .as_ref()
                    .and_then(scalar_of_type_inner)
                    .map(|s| s.kind);
                let is_float_or_unknown_arith = matches!(
                    op,
                    BinaryOperator::Add | BinaryOperator::Subtract | BinaryOperator::Multiply
                ) && !matches!(
                    scalar_kind,
                    Some(ScalarKind::Uint)
                );
                if is_dangerous_op || is_float_or_unknown_arith {
                    unsafe_for_const_target[left.index()] = true;
                    unsafe_for_const_target[right.index()] = true;
                }
            }
            Expression::Unary {
                op: UnaryOperator::Negate,
                expr: inner,
            } => {
                unsafe_for_const_target[inner.index()] = true;
            }
            Expression::Math { .. } | Expression::Relational { .. } => {
                visit_expr_children(expr, |c| {
                    unsafe_for_const_target[c.index()] = true;
                });
            }
            _ => {}
        }
    }

    // Propagate `unsafe_for_const_target` backward through transparent value wrappers so a
    // constant forwarded into e.g. `vec2(x, 1.0)` feeding `sqrt(...)` is also protected.
    let handles: Vec<Handle<Expression>> = func.expressions.iter().map(|(h, _)| h).collect();
    for h in handles.into_iter().rev() {
        if !unsafe_for_const_target[h.index()] {
            continue;
        }
        match &func.expressions[h] {
            Expression::Compose { .. }
            | Expression::Splat { .. }
            | Expression::Swizzle { .. }
            | Expression::AccessIndex { .. }
            | Expression::Select { .. }
            | Expression::As { .. } => {
                visit_expr_children(&func.expressions[h], |c| {
                    unsafe_for_const_target[c.index()] = true;
                });
            }
            _ => {}
        }
    }

    (used_by_float_bitcast, unsafe_for_const_target)
}

#[inline]
fn can_forward_to_load(
    h_load: Handle<Expression>,
    target: Handle<Expression>,
    func: &Function,
    deps: &[ExprDeps],
    used_by_float_bitcast: &[bool],
    unsafe_for_const_target: &[bool],
) -> bool {
    if !deps[target.index()].is_const_expr {
        return true;
    }
    let idx = h_load.index();
    if unsafe_for_const_target[idx] {
        return false;
    }
    if used_by_float_bitcast[idx] {
        match func.expressions[target] {
            Expression::Literal(Literal::U32(u)) => is_safe_wgsl_f32_literal(f32::from_bits(u)),
            Expression::Literal(Literal::I32(i)) => {
                is_safe_wgsl_f32_literal(f32::from_bits(i as u32))
            }
            _ => false,
        }
    } else {
        true
    }
}

/// Summary of writes and side effects inside a nested control-flow block (`Block`/`If`/`Loop`/`Switch`).
#[derive(Debug, Default)]
struct BlockEffects {
    written_locals: HashSet<Handle<LocalVariable>>,
    mutates_mem: bool,
}

fn collect_block_effects(
    block: &Block,
    local_origins: &[Option<Handle<LocalVariable>>],
    escaped_locals: &HashSet<Handle<LocalVariable>>,
    out: &mut BlockEffects,
) {
    for stmt in block.iter() {
        match stmt {
            Statement::Emit(_)
            | Statement::Break
            | Statement::Continue
            | Statement::Return { .. }
            | Statement::Kill
            | Statement::SubgroupBallot { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::SubgroupGather { .. } => {}
            Statement::Block(inner) => {
                collect_block_effects(inner, local_origins, escaped_locals, out);
            }
            Statement::If { accept, reject, .. } => {
                collect_block_effects(accept, local_origins, escaped_locals, out);
                collect_block_effects(reject, local_origins, escaped_locals, out);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_block_effects(&case.body, local_origins, escaped_locals, out);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                collect_block_effects(body, local_origins, escaped_locals, out);
                collect_block_effects(continuing, local_origins, escaped_locals, out);
            }
            Statement::Store { pointer, .. } => match local_origins[pointer.index()] {
                Some(lv) if !escaped_locals.contains(&lv) => {
                    out.written_locals.insert(lv);
                }
                _ => {
                    out.mutates_mem = true;
                }
            },
            Statement::ImageStore { .. }
            | Statement::ImageAtomic { .. }
            | Statement::Atomic { .. }
            | Statement::WorkGroupUniformLoad { .. }
            | Statement::Call { .. }
            | Statement::RayQuery { .. }
            | Statement::RayPipelineFunction(_)
            | Statement::CooperativeStore { .. }
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_) => {
                out.mutates_mem = true;
            }
        }
    }
}

fn invalidate_known_for_effects(
    known_local: &mut HashMap<Handle<LocalVariable>, Handle<Expression>>,
    effects: &BlockEffects,
    remap: &[Handle<Expression>],
    deps: &[ExprDeps],
) {
    known_local.retain(|&lv, &mut val| {
        if effects.written_locals.contains(&lv) {
            return false;
        }
        let d = &deps[resolve_remap(remap, val).index()];
        if d.has_cf_sensitive_op {
            return false;
        }
        if effects.mutates_mem && d.reads_mutable_mem {
            return false;
        }
        if d.reads_any_in(&effects.written_locals) {
            return false;
        }
        true
    });
}

struct ForwardContext<'a> {
    func: &'a Function,
    local_origins: &'a [Option<Handle<LocalVariable>>],
    escaped_locals: &'a HashSet<Handle<LocalVariable>>,
    uniform_ptr_origins: &'a [bool],
    used_by_float_bitcast: &'a [bool],
    unsafe_for_const_target: &'a [bool],
    deps: Vec<ExprDeps>,
    remap: Vec<Handle<Expression>>,
    forwarded_loads: Vec<Handle<Expression>>,
}

fn walk_forward_block(
    block: &Block,
    known_local: &mut HashMap<Handle<LocalVariable>, Handle<Expression>>,
    fctx: &mut ForwardContext<'_>,
    stats: &mut IrOptStats,
) {
    for stmt in block.iter() {
        match stmt {
            Statement::Emit(range) => {
                for h in range.clone() {
                    let expr = &fctx.func.expressions[h];
                    // Dynamically recompute deps using already-forwarded children.
                    let updated_deps = compute_single_expr_deps(
                        expr,
                        &fctx.deps,
                        &fctx.remap,
                        fctx.local_origins,
                        fctx.escaped_locals,
                        fctx.uniform_ptr_origins,
                    );
                    fctx.deps[h.index()] = updated_deps;

                    if let Expression::Load { pointer } = *expr {
                        let ptr = resolve_remap(&fctx.remap, pointer);
                        if let Expression::LocalVariable(lv) = fctx.func.expressions[ptr]
                            && !fctx.escaped_locals.contains(&lv)
                        {
                            if let Some(&candidate) = known_local.get(&lv) {
                                let target = resolve_remap(&fctx.remap, candidate);
                                if target != h
                                    && target.index() < h.index()
                                    && can_forward_to_load(
                                        h,
                                        target,
                                        fctx.func,
                                        &fctx.deps,
                                        fctx.used_by_float_bitcast,
                                        fctx.unsafe_for_const_target,
                                    )
                                {
                                    fctx.remap[h.index()] = target;
                                    fctx.deps[h.index()] = fctx.deps[target.index()].clone();
                                    fctx.forwarded_loads.push(h);
                                    stats.locals_forwarded += 1;
                                }
                            } else {
                                // Record this load for straight-line load-to-load reuse until `lv` is written.
                                known_local.insert(lv, h);
                            }
                        }
                    }
                }
            }
            Statement::Store { pointer, value } => {
                let ptr = resolve_remap(&fctx.remap, *pointer);
                let val = resolve_remap(&fctx.remap, *value);
                match fctx.local_origins[ptr.index()] {
                    Some(lv) if !fctx.escaped_locals.contains(&lv) => {
                        // Invalidate `lv` and any tracked entry whose expression reads `lv`.
                        known_local.retain(|&k, &mut v| {
                            k != lv
                                && !fctx.deps[resolve_remap(&fctx.remap, v).index()].reads_local(lv)
                        });
                        // Only record whole-local stores whose value does not self-read `lv`.
                        if matches!(fctx.func.expressions[ptr], Expression::LocalVariable(_))
                            && !fctx.deps[val.index()].reads_local(lv)
                        {
                            known_local.insert(lv, val);
                        }
                    }
                    _ => {
                        // Non-local or escaped-local store: invalidate all mutable-memory-dependent entries.
                        known_local.retain(|_, &mut v| {
                            !fctx.deps[resolve_remap(&fctx.remap, v).index()].reads_mutable_mem
                        });
                    }
                }
            }
            Statement::Call { .. }
            | Statement::Atomic { .. }
            | Statement::ImageStore { .. }
            | Statement::ImageAtomic { .. }
            | Statement::RayQuery { .. }
            | Statement::RayPipelineFunction(_)
            | Statement::CooperativeStore { .. } => {
                known_local.retain(|_, &mut v| {
                    !fctx.deps[resolve_remap(&fctx.remap, v).index()].reads_mutable_mem
                });
            }
            Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_)
            | Statement::WorkGroupUniformLoad { .. } => {
                known_local.retain(|_, &mut v| {
                    let d = &fctx.deps[resolve_remap(&fctx.remap, v).index()];
                    !d.reads_mutable_mem && !d.has_cf_sensitive_op
                });
            }
            Statement::Kill => {
                known_local.retain(|_, &mut v| {
                    !fctx.deps[resolve_remap(&fctx.remap, v).index()].has_cf_sensitive_op
                });
            }
            Statement::Break | Statement::Continue | Statement::Return { .. } => {
                known_local.clear();
            }
            Statement::Block(inner) => {
                let mut inner_known = HashMap::new();
                walk_forward_block(inner, &mut inner_known, fctx, stats);
                let mut effects = BlockEffects::default();
                collect_block_effects(inner, fctx.local_origins, fctx.escaped_locals, &mut effects);
                invalidate_known_for_effects(known_local, &effects, &fctx.remap, &fctx.deps);
            }
            Statement::If { accept, reject, .. } => {
                let mut accept_known = HashMap::new();
                walk_forward_block(accept, &mut accept_known, fctx, stats);
                let mut reject_known = HashMap::new();
                walk_forward_block(reject, &mut reject_known, fctx, stats);
                let mut effects = BlockEffects::default();
                collect_block_effects(
                    accept,
                    fctx.local_origins,
                    fctx.escaped_locals,
                    &mut effects,
                );
                collect_block_effects(
                    reject,
                    fctx.local_origins,
                    fctx.escaped_locals,
                    &mut effects,
                );
                invalidate_known_for_effects(known_local, &effects, &fctx.remap, &fctx.deps);
            }
            Statement::Switch { cases, .. } => {
                let mut effects = BlockEffects::default();
                for case in cases {
                    let mut case_known = HashMap::new();
                    walk_forward_block(&case.body, &mut case_known, fctx, stats);
                    collect_block_effects(
                        &case.body,
                        fctx.local_origins,
                        fctx.escaped_locals,
                        &mut effects,
                    );
                }
                invalidate_known_for_effects(known_local, &effects, &fctx.remap, &fctx.deps);
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                let mut body_known = HashMap::new();
                walk_forward_block(body, &mut body_known, fctx, stats);
                let mut cont_known = HashMap::new();
                walk_forward_block(continuing, &mut cont_known, fctx, stats);
                let mut effects = BlockEffects::default();
                collect_block_effects(body, fctx.local_origins, fctx.escaped_locals, &mut effects);
                collect_block_effects(
                    continuing,
                    fctx.local_origins,
                    fctx.escaped_locals,
                    &mut effects,
                );
                invalidate_known_for_effects(known_local, &effects, &fctx.remap, &fctx.deps);
            }
            Statement::SubgroupBallot { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::SubgroupGather { .. } => {}
        }
    }
}

fn forward_local_loads(ctx: &ModuleContext<'_>, func: &mut Function, stats: &mut IrOptStats) {
    let n = func.expressions.len();
    if n == 0 || func.local_variables.is_empty() {
        return;
    }

    let local_origins = compute_local_ptr_origins(func);
    let escaped_locals = compute_escaped_locals(func, &local_origins);
    let uniform_ptr_origins = compute_uniform_ptr_origins(ctx, func);
    let resolved = infer_all_expr_types(ctx, func);
    let (used_by_float_bitcast, unsafe_for_const_target) =
        compute_const_target_guards(func, &resolved);

    let handles: Vec<Handle<Expression>> = func.expressions.iter().map(|(h, _)| h).collect();
    let remap_init = handles.clone();
    let mut deps: Vec<ExprDeps> = vec![ExprDeps::default(); n];
    for (h, expr) in func.expressions.iter() {
        deps[h.index()] = compute_single_expr_deps(
            expr,
            &deps,
            &remap_init,
            &local_origins,
            &escaped_locals,
            &uniform_ptr_origins,
        );
    }

    // Seed known values from non-escaped `LocalVariable.init` (strictly pre-emitted handles so
    // forwarding never references an unemitted expression).
    let mut known_local: HashMap<Handle<LocalVariable>, Handle<Expression>> = HashMap::new();
    for (lv, local) in func.local_variables.iter() {
        if !escaped_locals.contains(&lv)
            && let Some(init_h) = local.init
            && !is_emittable_expr(&func.expressions[init_h])
        {
            known_local.insert(lv, init_h);
        }
    }

    let (remap, forwarded_loads) = {
        let mut fctx = ForwardContext {
            func,
            local_origins: &local_origins,
            escaped_locals: &escaped_locals,
            uniform_ptr_origins: &uniform_ptr_origins,
            used_by_float_bitcast: &used_by_float_bitcast,
            unsafe_for_const_target: &unsafe_for_const_target,
            deps,
            remap: remap_init,
            forwarded_loads: Vec::new(),
        };
        walk_forward_block(&func.body, &mut known_local, &mut fctx, stats);
        (fctx.remap, fctx.forwarded_loads)
    };

    if forwarded_loads.is_empty() {
        return;
    }

    // Flatten remap chains to their final root handles.
    let flat_remap: Vec<Handle<Expression>> = handles
        .iter()
        .map(|&h| resolve_remap(&remap, h))
        .collect();

    for (_, expr) in func.expressions.iter_mut() {
        remap_expr_children(expr, |c| flat_remap[c.index()]);
    }
    remap_block_exprs(&mut func.body, false, &|h| flat_remap[h.index()]);
    for (_, local) in func.local_variables.iter_mut() {
        if let Some(ref mut init_h) = local.init {
            *init_h = flat_remap[init_h.index()];
        }
    }

    // Neutralize forwarded `Load` handles so `prune_dead_locals` and
    // `compact_live_expressions_and_emits` no longer see them as live loads.
    for h in forwarded_loads {
        func.expressions[h] = Expression::Literal(Literal::Bool(false));
        func.named_expressions.shift_remove(&h);
    }
}

// ---------------------------------------------------------------------------
// Pass 3: Intra-block Common Subexpression Elimination (CSE) (#2164)
// ---------------------------------------------------------------------------

/// Canonical operand identity for CSE keys: merges duplicate pre-emitted handles (`Literal`,
/// `Constant`, `Override`, `ZeroValue`, `FunctionArgument`, `GlobalVariable`, `LocalVariable`)
/// while preserving bit-exact IEEE-754 float representations (`-0.0` != `+0.0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CseOperand {
    Handle(Handle<Expression>),
    LitF64(u64),
    LitF32(u32),
    LitF16(u16),
    LitU32(u32),
    LitI32(i32),
    LitU16(u16),
    LitI16(i16),
    LitU64(u64),
    LitI64(i64),
    LitBool(bool),
    LitAbstractInt(i64),
    LitAbstractFloat(u64),
    Constant(Handle<Constant>),
    Override(Handle<Override>),
    ZeroValue(Handle<Type>),
    FunctionArgument(u32),
    GlobalVariable(Handle<GlobalVariable>),
    LocalVariable(Handle<LocalVariable>),
}

/// Canonical step along a read-only global pointer chain (`AccessIndex` / `Access`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PtrStep {
    Index(u32),
    Dynamic(CseOperand),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CseKey {
    Access {
        base: CseOperand,
        index: CseOperand,
    },
    AccessIndex {
        base: CseOperand,
        index: u32,
    },
    Splat {
        size: VectorSize,
        value: CseOperand,
    },
    Swizzle {
        size: VectorSize,
        vector: CseOperand,
        pattern: [u8; 4],
    },
    Compose {
        ty: Handle<Type>,
        components: Vec<CseOperand>,
    },
    Unary {
        op: UnaryOperator,
        expr: CseOperand,
    },
    Binary {
        op: BinaryOperator,
        left: CseOperand,
        right: CseOperand,
    },
    Select {
        condition: CseOperand,
        accept: CseOperand,
        reject: CseOperand,
    },
    Relational {
        fun: RelationalFunction,
        argument: CseOperand,
    },
    Math {
        fun: MathFunction,
        arg: CseOperand,
        arg1: Option<CseOperand>,
        arg2: Option<CseOperand>,
        arg3: Option<CseOperand>,
    },
    As {
        expr: CseOperand,
        kind: ScalarKind,
        convert: Option<u8>,
    },
    /// Immutable `Load` from a read-only `@group/@binding` global (`Uniform` or `Storage(LOAD)`)
    /// along a canonical pure pointer chain (`GlobalVariable` + `AccessIndex`/`Access` steps).
    ReadOnlyLoad {
        global: Handle<GlobalVariable>,
        steps: Vec<PtrStep>,
    },
}

fn canonicalize_cse_operand(func: &Function, h: Handle<Expression>) -> CseOperand {
    match func.expressions[h] {
        Expression::Literal(lit) => match lit {
            Literal::F64(v) => CseOperand::LitF64(v.to_bits()),
            Literal::F32(v) => CseOperand::LitF32(v.to_bits()),
            Literal::F16(v) => CseOperand::LitF16(v.to_bits()),
            Literal::U32(v) => CseOperand::LitU32(v),
            Literal::I32(v) => CseOperand::LitI32(v),
            Literal::U16(v) => CseOperand::LitU16(v),
            Literal::I16(v) => CseOperand::LitI16(v),
            Literal::U64(v) => CseOperand::LitU64(v),
            Literal::I64(v) => CseOperand::LitI64(v),
            Literal::Bool(v) => CseOperand::LitBool(v),
            Literal::AbstractInt(v) => CseOperand::LitAbstractInt(v),
            Literal::AbstractFloat(v) => CseOperand::LitAbstractFloat(v.to_bits()),
        },
        Expression::Constant(c) => CseOperand::Constant(c),
        Expression::Override(o) => CseOperand::Override(o),
        Expression::ZeroValue(ty) => CseOperand::ZeroValue(ty),
        Expression::FunctionArgument(idx) => CseOperand::FunctionArgument(idx),
        Expression::GlobalVariable(g) => CseOperand::GlobalVariable(g),
        Expression::LocalVariable(lv) => CseOperand::LocalVariable(lv),
        _ => CseOperand::Handle(h),
    }
}

/// Returns `true` iff `(left op right) == (right op left)` is provably bit-exact and type-preserving
/// (strictly matching integer or boolean scalar/vector operands; never floats or matrices).
fn is_safe_commutative_cse_op(
    op: BinaryOperator,
    left: Handle<Expression>,
    right: Handle<Expression>,
    resolved: &[Option<TypeInner>],
) -> bool {
    let (Some(lt), Some(rt)) = (resolved[left.index()].as_ref(), resolved[right.index()].as_ref())
    else {
        return false;
    };
    if lt != rt {
        return false;
    }
    let Some((_, scalar)) = value_shape(lt) else {
        return false;
    };
    match op {
        BinaryOperator::Add
        | BinaryOperator::Multiply
        | BinaryOperator::And
        | BinaryOperator::InclusiveOr
        | BinaryOperator::ExclusiveOr => matches!(scalar.kind, ScalarKind::Uint | ScalarKind::Sint),
        BinaryOperator::Equal | BinaryOperator::NotEqual => {
            matches!(
                scalar.kind,
                ScalarKind::Uint | ScalarKind::Sint | ScalarKind::Bool
            )
        }
        _ => false,
    }
}

/// Canonicalize a pointer chain rooted in an immutable read-only global (`AddressSpace::Uniform`
/// or `AddressSpace::Storage { access: StorageAccess::LOAD }`) with pure `AccessIndex`/`Access`
/// index steps. Returns `None` for any mutable global (`Storage` with `STORE`, `WorkGroup`,
/// `Private`), local variable, or non-chain pointer.
fn canonicalize_readonly_ptr_chain(
    ctx: &ModuleContext<'_>,
    func: &Function,
    ptr: Handle<Expression>,
    remap: &[Handle<Expression>],
) -> Option<(Handle<GlobalVariable>, Vec<PtrStep>)> {
    const MAX_CHAIN_DEPTH: usize = 64;
    let mut cur = resolve_remap(remap, ptr);
    let mut steps: Vec<PtrStep> = Vec::new();
    for _ in 0..MAX_CHAIN_DEPTH {
        match func.expressions[cur] {
            Expression::GlobalVariable(g) => {
                if is_immutable_readonly_global(&ctx.global_variables[g]) {
                    steps.reverse();
                    return Some((g, steps));
                }
                return None;
            }
            Expression::AccessIndex { base, index } => {
                steps.push(PtrStep::Index(index));
                cur = resolve_remap(remap, base);
            }
            Expression::Access { base, index } => {
                let r_idx = resolve_remap(remap, index);
                let step = match func.expressions[r_idx] {
                    Expression::Literal(Literal::U32(k)) => PtrStep::Index(k),
                    _ => PtrStep::Dynamic(canonicalize_cse_operand(func, r_idx)),
                };
                steps.push(step);
                cur = resolve_remap(remap, base);
            }
            _ => return None,
        }
    }
    None
}

fn build_cse_key(
    ctx: &ModuleContext<'_>,
    h: Handle<Expression>,
    func: &Function,
    resolved: &[Option<TypeInner>],
    remap: &[Handle<Expression>],
) -> Option<CseKey> {
    let op_of = |c: Handle<Expression>| canonicalize_cse_operand(func, resolve_remap(remap, c));
    match &func.expressions[h] {
        // Value-typed `Access`/`AccessIndex` only — never CSE pointer / `ValuePointer` accesses,
        // because baking a `ValuePointer` (e.g. `&vec.x`) into a WGSL `let` is illegal in WGSL.
        Expression::Access { base, index } => {
            match resolved[h.index()].as_ref()? {
                TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => return None,
                _ => {}
            }
            Some(CseKey::Access {
                base: op_of(*base),
                index: op_of(*index),
            })
        }
        Expression::AccessIndex { base, index } => {
            match resolved[h.index()].as_ref()? {
                TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => return None,
                _ => {}
            }
            Some(CseKey::AccessIndex {
                base: op_of(*base),
                index: *index,
            })
        }
        Expression::Splat { size, value } => Some(CseKey::Splat {
            size: *size,
            value: op_of(*value),
        }),
        Expression::Swizzle {
            size,
            vector,
            pattern,
        } => {
            // Canonicalize unused trailing pattern lanes past `size`.
            let mut norm_pattern = [0u8; 4];
            let count = *size as usize;
            for i in 0..count {
                norm_pattern[i] = match pattern[i] {
                    SwizzleComponent::X => 0,
                    SwizzleComponent::Y => 1,
                    SwizzleComponent::Z => 2,
                    SwizzleComponent::W => 3,
                };
            }
            Some(CseKey::Swizzle {
                size: *size,
                vector: op_of(*vector),
                pattern: norm_pattern,
            })
        }
        Expression::Compose { ty, components } => Some(CseKey::Compose {
            ty: *ty,
            components: components.iter().map(|&c| op_of(c)).collect(),
        }),
        Expression::Unary { op, expr } => Some(CseKey::Unary {
            op: *op,
            expr: op_of(*expr),
        }),
        Expression::Binary { op, left, right } => {
            let rl = resolve_remap(remap, *left);
            let rr = resolve_remap(remap, *right);
            let lop = canonicalize_cse_operand(func, rl);
            let rop = canonicalize_cse_operand(func, rr);
            let (a, b) = if is_safe_commutative_cse_op(*op, rl, rr, resolved) && rop < lop {
                (rop, lop)
            } else {
                (lop, rop)
            };
            Some(CseKey::Binary {
                op: *op,
                left: a,
                right: b,
            })
        }
        Expression::Select {
            condition,
            accept,
            reject,
        } => Some(CseKey::Select {
            condition: op_of(*condition),
            accept: op_of(*accept),
            reject: op_of(*reject),
        }),
        Expression::Relational { fun, argument } => Some(CseKey::Relational {
            fun: *fun,
            argument: op_of(*argument),
        }),
        Expression::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        } => Some(CseKey::Math {
            fun: *fun,
            arg: op_of(*arg),
            arg1: arg1.map(op_of),
            arg2: arg2.map(op_of),
            arg3: arg3.map(op_of),
        }),
        Expression::As {
            expr,
            kind,
            convert,
        } => Some(CseKey::As {
            expr: op_of(*expr),
            kind: *kind,
            convert: *convert,
        }),
        // Immutable read-only `@group/@binding` global loads (`Uniform` / `Storage(LOAD)` only!).
        Expression::Load { pointer } => {
            let (global, steps) = canonicalize_readonly_ptr_chain(ctx, func, *pointer, remap)?;
            Some(CseKey::ReadOnlyLoad { global, steps })
        }
        // Never CSE mutable loads, texture ops, derivatives, statement results, or pre-emitted handles.
        _ => None,
    }
}

struct CseContext<'a> {
    ctx: &'a ModuleContext<'a>,
    func: &'a Function,
    resolved: &'a [Option<TypeInner>],
    table: HashMap<CseKey, Handle<Expression>>,
    undo_stack: Vec<CseKey>,
    remap: Vec<Handle<Expression>>,
    cse_dropped: Vec<Handle<Expression>>,
    readonly_loads_dropped: Vec<Handle<Expression>>,
}

impl CseContext<'_> {
    fn walk_scoped(&mut self, block: &Block, stats: &mut IrOptStats) {
        let mark = self.undo_stack.len();
        walk_cse_block(block, self, stats);
        while self.undo_stack.len() > mark {
            if let Some(key) = self.undo_stack.pop() {
                self.table.remove(&key);
            }
        }
    }
}

fn walk_cse_block(block: &Block, cctx: &mut CseContext<'_>, stats: &mut IrOptStats) {
    for stmt in block.iter() {
        match stmt {
            Statement::Emit(range) => {
                for h in range.clone() {
                    if let Some(key) =
                        build_cse_key(cctx.ctx, h, cctx.func, cctx.resolved, &cctx.remap)
                    {
                        if let Some(&canon) = cctx.table.get(&key) {
                            if canon != h && canon.index() < h.index() {
                                cctx.remap[h.index()] = canon;
                                if matches!(key, CseKey::ReadOnlyLoad { .. }) {
                                    cctx.readonly_loads_dropped.push(h);
                                    stats.readonly_loads_cse += 1;
                                } else {
                                    cctx.cse_dropped.push(h);
                                    stats.cse_eliminated += 1;
                                }
                            }
                        } else {
                            cctx.table.insert(key.clone(), h);
                            cctx.undo_stack.push(key);
                        }
                    }
                }
            }
            Statement::Block(inner) => {
                cctx.walk_scoped(inner, stats);
            }
            Statement::If { accept, reject, .. } => {
                cctx.walk_scoped(accept, stats);
                cctx.walk_scoped(reject, stats);
            }
            Statement::Switch { cases, .. } => {
                for case in cases {
                    cctx.walk_scoped(&case.body, stats);
                }
            }
            Statement::Loop {
                body, continuing, ..
            } => {
                cctx.walk_scoped(body, stats);
                cctx.walk_scoped(continuing, stats);
            }
            _ => {}
        }
    }
}

fn eliminate_common_subexpressions(
    ctx: &ModuleContext<'_>,
    func: &mut Function,
    stats: &mut IrOptStats,
) {
    let n = func.expressions.len();
    if n == 0 {
        return;
    }

    let resolved = infer_all_expr_types(ctx, func);
    let handles: Vec<Handle<Expression>> = func.expressions.iter().map(|(h, _)| h).collect();

    let (remap, cse_dropped, readonly_loads_dropped) = {
        let mut cctx = CseContext {
            ctx,
            func,
            resolved: &resolved,
            table: HashMap::new(),
            undo_stack: Vec::new(),
            remap: handles.clone(),
            cse_dropped: Vec::new(),
            readonly_loads_dropped: Vec::new(),
        };
        walk_cse_block(&func.body, &mut cctx, stats);
        (cctx.remap, cctx.cse_dropped, cctx.readonly_loads_dropped)
    };

    if cse_dropped.is_empty() && readonly_loads_dropped.is_empty() {
        return;
    }

    let flat_remap: Vec<Handle<Expression>> = handles
        .iter()
        .map(|&h| resolve_remap(&remap, h))
        .collect();

    for (_, expr) in func.expressions.iter_mut() {
        remap_expr_children(expr, |c| flat_remap[c.index()]);
    }
    remap_block_exprs(&mut func.body, false, &|h| flat_remap[h.index()]);
    for (_, local) in func.local_variables.iter_mut() {
        if let Some(ref mut init_h) = local.init {
            *init_h = flat_remap[init_h.index()];
        }
    }
    for h in cse_dropped {
        func.named_expressions.shift_remove(&h);
    }
    // Neutralize deduplicated read-only `Load` handles so `collect_live_expressions` and
    // `compact_live_expressions_and_emits` drop both the duplicate `Load` and its dead pointer
    // chain cleanly from `Statement::Emit` and the expression arena.
    for h in readonly_loads_dropped {
        func.expressions[h] = Expression::Literal(Literal::Bool(false));
        func.named_expressions.shift_remove(&h);
    }
}

// ---------------------------------------------------------------------------
// Pass 4: Dead local-variable pruning
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

/// Compute `drop_safe[h]`: `true` iff `h` is composed exclusively of pure expression nodes and
/// `Load`s of non-escaped `LocalVariable`s (100% side-effect-free, non-trapping, non-aliased
/// function-scope reads), so a `Store` to an otherwise dead local can be safely dropped.
fn compute_drop_safe_exprs(
    func: &Function,
    local_origins: &[Option<Handle<LocalVariable>>],
    escaped_locals: &HashSet<Handle<LocalVariable>>,
) -> Vec<bool> {
    let mut safe = vec![false; func.expressions.len()];
    for (h, expr) in func.expressions.iter() {
        let node_safe = match expr {
            Expression::Load { pointer } => {
                local_origins[pointer.index()].is_some_and(|lv| !escaped_locals.contains(&lv))
            }
            other => is_pure_expr_node(other),
        };
        if !node_safe {
            continue;
        }
        let mut all_children_safe = true;
        visit_expr_children(expr, |c| {
            if !safe[c.index()] {
                all_children_safe = false;
            }
        });
        safe[h.index()] = all_children_safe;
    }
    safe
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
    let escaped_locals = compute_escaped_locals(func, &local_origins);
    let drop_safe = compute_drop_safe_exprs(func, &local_origins, &escaped_locals);
    let live_locals = collect_live_locals(func, &local_origins, &drop_safe);

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
