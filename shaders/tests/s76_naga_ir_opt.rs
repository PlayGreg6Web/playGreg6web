//! S76 (#2137, Refs #663, #482): Post-lowering Naga IR algebraic simplification, bitcast
//! cancellation, constant folding, dead-local / dead-emit pruning, IEEE-754 float edge safety,
//! and end-to-end RDNA2 -> WGSL byte/expression count reduction tests.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, BinaryOperator, Expression, Function, FunctionArgument, FunctionResult,
    GlobalVariable, Handle, Literal, LocalVariable, Module, Range, ResourceBinding, Scalar,
    ScalarKind, ShaderStage as NagaStage, Span, Statement, StorageAccess, Type, TypeInner,
    UnaryOperator,
};
use playgta6_shaders::ir_opt::{IrOptStats, optimize_function, optimize_module};
use playgta6_shaders::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, translate_rdna2_to_wgsl,
};

const BASELINE_WEBGPU_CAPABILITIES: Capabilities =
    Capabilities::SHADER_FLOAT16_IN_FLOAT32.union(Capabilities::CUBE_ARRAY_TEXTURES);

fn validate_and_emit_wgsl(module: &Module) -> String {
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    let info = validator
        .validate(module)
        .unwrap_or_else(|e| panic!("naga validation failed:\n{}", e.emit_to_string("")));
    let wgsl = naga::back::wgsl::write_string(
        module,
        &info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .expect("naga WGSL emission must succeed");
    let reparsed = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|e| panic!("WGSL round-trip parse failed:\n{}\n\nWGSL:\n{wgsl}", e.emit_to_string(&wgsl)));
    let mut revalidator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    revalidator
        .validate(&reparsed)
        .unwrap_or_else(|e| panic!("WGSL round-trip validation failed:\n{}\n\nWGSL:\n{wgsl}", e.emit_to_string(&wgsl)));
    wgsl
}

fn parse_wgsl(src: &str) -> Module {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\n\nSRC:\n{src}", e.emit_to_string(src)));
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("input WGSL validation failed:\n{}\n\nSRC:\n{src}", e.emit_to_string(src)));
    module
}

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

fn enc_vop1(op: u8, vdst: u8, src0: u16) -> u32 {
    (0b0111_111u32 << 25)
        | (u32::from(vdst) << 17)
        | (u32::from(op) << 9)
        | u32::from(src0 & 0x1FF)
}

fn enc_vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (u32::from(op & 0x3F) << 25)
        | (u32::from(vdst) << 17)
        | (u32::from(vsrc1) << 9)
        | u32::from(src0 & 0x1FF)
}

fn enc_vintrp(op: u8, vdst: u8, attr: u8, attrchan: u8, vsrc: u8) -> u32 {
    (0b1100_10u32 << 26)
        | (u32::from(vdst) << 18)
        | (u32::from(op & 0x3) << 16)
        | (u32::from(attr & 0x3F) << 10)
        | (u32::from(attrchan & 0x3) << 8)
        | u32::from(vsrc)
}

fn enc_exp(target: u8, en: u8, compr: bool, done: bool, vm: bool, vsrc: [u8; 4]) -> [u32; 2] {
    let w0 = (0b1111_10u32 << 26)
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

fn count_module_exprs(module: &Module) -> usize {
    let ep_count: usize = module
        .entry_points
        .iter()
        .map(|ep| ep.function.expressions.len())
        .sum();
    let fn_count: usize = module
        .functions
        .iter()
        .map(|(_, f)| f.expressions.len())
        .sum();
    ep_count + fn_count
}

#[test]
fn cancels_identity_and_round_trip_bitcasts_while_preserving_mismatched_bitcasts() {
    let src = r#"
@group(0) @binding(0) var<storage, read> in_f: array<f32>;
@group(0) @binding(1) var<storage, read> in_u: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_f: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_u: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_i: array<i32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let f = in_f[idx];
    let u = in_u[idx];

    // Identity bitcasts: bitcast<f32>(f) and bitcast<u32>(u).
    let id_f = bitcast<f32>(f);
    let id_u = bitcast<u32>(u);

    // Round-trip bitcasts: bitcast<f32>(bitcast<u32>(f)) and bitcast<u32>(bitcast<f32>(u)).
    let rt_f = bitcast<f32>(bitcast<u32>(id_f));
    let rt_u = bitcast<u32>(bitcast<f32>(id_u));

    // Nested double round-trip: bitcast<f32>(bitcast<i32>(bitcast<f32>(bitcast<u32>(f)))).
    let rt_nested = bitcast<f32>(bitcast<i32>(bitcast<f32>(bitcast<u32>(f))));

    // Mismatched-type bitcasts MUST be preserved:
    //  * bitcast<f32>(u) -> u32 != f32
    //  * bitcast<i32>(bitcast<f32>(u)) -> orig is u32, outer is i32 (mismatched!)
    let mismatch_1 = bitcast<f32>(u);
    let mismatch_2 = bitcast<i32>(bitcast<f32>(u));

    out_f[idx] = rt_f + rt_nested + mismatch_1;
    out_u[idx] = rt_u;
    out_i[idx] = mismatch_2;
}
"#;

    let mut module = parse_wgsl(src);
    let before_exprs = count_module_exprs(&module);
    let stats = optimize_module(&mut module);
    let after_exprs = count_module_exprs(&module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.bitcasts_cancelled >= 6,
        "expected >= 6 identity/round-trip bitcasts cancelled, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(
        after_exprs < before_exprs,
        "expected expression arena compaction ({before_exprs} -> {after_exprs})"
    );

    // Count surviving bitcast<...> calls in the emitted WGSL:
    // Only mismatch_1 (`bitcast<f32>(...)`) and mismatch_2 (`bitcast<i32>(bitcast<f32>(...))`)
    // should remain (3 bitcasts total instead of 11).
    let bitcast_count = wgsl.matches("bitcast<").count();
    assert_eq!(
        bitcast_count, 3,
        "only the mismatched bitcasts should survive (got {bitcast_count}):\n{wgsl}"
    );
    assert!(
        wgsl.contains("bitcast<i32>(bitcast<f32>("),
        "mismatched u32->f32->i32 chain must be preserved:\n{wgsl}"
    );
    // All 5 @group/@binding globals must remain intact.
    for b in 0..5 {
        assert!(
            wgsl.contains(&format!("@group(0) @binding({b})")),
            "missing @binding({b}) in:\n{wgsl}"
        );
    }
}

#[test]
fn folds_integer_and_boolean_constants_and_identities_while_preserving_ieee754_floats() {
    // Construct a Naga IR module directly so we exercise exact unfolded Constant/Binary/Unary/Select
    // nodes without WGSL front-end pre-folding them away first.
    let mut module = Module::default();
    let u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let i32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::I32),
        },
        Span::UNDEFINED,
    );
    let f32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        },
        Span::UNDEFINED,
    );
    let bool_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::BOOL),
        },
        Span::UNDEFINED,
    );
    let arr_u32 = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base: u32_ty,
                size: naga::ArraySize::Dynamic,
                stride: 4,
            },
        },
        Span::UNDEFINED,
    );
    let arr_i32 = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base: i32_ty,
                size: naga::ArraySize::Dynamic,
                stride: 4,
            },
        },
        Span::UNDEFINED,
    );
    let arr_f32 = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base: f32_ty,
                size: naga::ArraySize::Dynamic,
                stride: 4,
            },
        },
        Span::UNDEFINED,
    );

    let g_u = module.global_variables.append(
        GlobalVariable {
            name: Some("buf_u".to_string()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_u32,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let g_i = module.global_variables.append(
        GlobalVariable {
            name: Some("buf_i".to_string()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: arr_i32,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let g_f = module.global_variables.append(
        GlobalVariable {
            name: Some("buf_f".to_string()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 2,
            }),
            ty: arr_f32,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );

    let mut func = Function {
        name: Some("test_ops".to_string()),
        arguments: vec![
            FunctionArgument {
                name: Some("x_u".to_string()),
                ty: u32_ty,
                binding: None,
            },
            FunctionArgument {
                name: Some("x_i".to_string()),
                ty: i32_ty,
                binding: None,
            },
            FunctionArgument {
                name: Some("x_f".to_string()),
                ty: f32_ty,
                binding: None,
            },
            FunctionArgument {
                name: Some("cond".to_string()),
                ty: bool_ty,
                binding: None,
            },
        ],
        ..Function::default()
    };

    let emit = |f: &mut Function, e: Expression| -> Handle<Expression> {
        let h = f.expressions.append(e, Span::UNDEFINED);
        f.body
            .push(Statement::Emit(Range::new_from_bounds(h, h)), Span::UNDEFINED);
        h
    };
    let raw = |f: &mut Function, e: Expression| -> Handle<Expression> {
        f.expressions.append(e, Span::UNDEFINED)
    };
    let store_slot = |f: &mut Function, g: Handle<GlobalVariable>, idx: u32, val: Handle<Expression>| {
        let g_expr = f.expressions.append(Expression::GlobalVariable(g), Span::UNDEFINED);
        let ptr = f.expressions.append(
            Expression::AccessIndex {
                base: g_expr,
                index: idx,
            },
            Span::UNDEFINED,
        );
        f.body
            .push(Statement::Emit(Range::new_from_bounds(ptr, ptr)), Span::UNDEFINED);
        f.body
            .push(Statement::Store { pointer: ptr, value: val }, Span::UNDEFINED);
    };

    let x_u = raw(&mut func, Expression::FunctionArgument(0));
    let x_i = raw(&mut func, Expression::FunctionArgument(1));
    let x_f = raw(&mut func, Expression::FunctionArgument(2));
    let cond = raw(&mut func, Expression::FunctionArgument(3));

    // --- Constant folding tests ---
    // Wrapping u32 add: 0xFFFF_FFFFu + 5u -> 4u
    let u_max = raw(&mut func, Expression::Literal(Literal::U32(u32::MAX)));
    let u_5 = raw(&mut func, Expression::Literal(Literal::U32(5)));
    let c_wrap_add = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Add,
            left: u_max,
            right: u_5,
        },
    );
    store_slot(&mut func, g_u, 0, c_wrap_add);

    // Shift masking: 1u << 35u -> 1u << 3 -> 8u
    let u_1 = raw(&mut func, Expression::Literal(Literal::U32(1)));
    let u_35 = raw(&mut func, Expression::Literal(Literal::U32(35)));
    let c_shl = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::ShiftLeft,
            left: u_1,
            right: u_35,
        },
    );
    store_slot(&mut func, g_u, 1, c_shl);

    // Div by zero MUST NOT be folded (skip cleanly): 10u / 0u -> stays binary op
    let u_10 = raw(&mut func, Expression::Literal(Literal::U32(10)));
    let u_0 = raw(&mut func, Expression::Literal(Literal::U32(0)));
    let c_div_zero = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Divide,
            left: u_10,
            right: u_0,
        },
    );
    // Keep c_div_zero in a runtime expression (not const-eval'd by WGSL parser if combined with x_u?
    // Note: Naga WGSL frontend rejects literal `10u / 0u` as const-eval error on reparse, so test
    // div-by-zero preservation directly on the Naga IR before WGSL emit, and test safe `10u / 2u` here!)
    let u_2 = raw(&mut func, Expression::Literal(Literal::U32(2)));
    let c_div_ok = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Divide,
            left: u_10,
            right: u_2,
        },
    );
    store_slot(&mut func, g_u, 2, c_div_ok);
    let _ = c_div_zero; // tested separately in dedicated IR test below

    // Select on constant bool: select(x_u, 99u, true) -> 99u
    let b_true = raw(&mut func, Expression::Literal(Literal::Bool(true)));
    let u_99 = raw(&mut func, Expression::Literal(Literal::U32(99)));
    let c_sel = emit(
        &mut func,
        Expression::Select {
            condition: b_true,
            accept: u_99,
            reject: x_u,
        },
    );
    store_slot(&mut func, g_u, 3, c_sel);

    // --- Integer & boolean algebraic identities ---
    // Chain: ((((((((x_u + 0u) - 0u) * 1u) | 0u) ^ 0u) & 0xFFFFFFFFu) << 0u) >> 0u) -> x_u
    let t1 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Add,
            left: x_u,
            right: u_0,
        },
    );
    let t2 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Subtract,
            left: t1,
            right: u_0,
        },
    );
    let t3 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Multiply,
            left: t2,
            right: u_1,
        },
    );
    let t4 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::InclusiveOr,
            left: t3,
            right: u_0,
        },
    );
    let t5 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::ExclusiveOr,
            left: t4,
            right: u_0,
        },
    );
    let t6 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::And,
            left: t5,
            right: u_max,
        },
    );
    let t7 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::ShiftLeft,
            left: t6,
            right: u_0,
        },
    );
    let t8 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: t7,
            right: u_0,
        },
    );
    // x_u & x_u -> x_u, x_u | x_u -> x_u, select(x_u, x_u, !!cond) -> x_u
    let t9 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::And,
            left: t8,
            right: x_u,
        },
    );
    let t10 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::InclusiveOr,
            left: t9,
            right: x_u,
        },
    );
    let not1 = emit(
        &mut func,
        Expression::Unary {
            op: UnaryOperator::LogicalNot,
            expr: cond,
        },
    );
    let not2 = emit(
        &mut func,
        Expression::Unary {
            op: UnaryOperator::LogicalNot,
            expr: not1,
        },
    );
    let t11 = emit(
        &mut func,
        Expression::Select {
            condition: not2,
            accept: t10,
            reject: x_u,
        },
    );
    store_slot(&mut func, g_u, 4, t11);

    // Pure `x_u ^ x_u -> 0u`
    let xor_self = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::ExclusiveOr,
            left: x_u,
            right: x_u,
        },
    );
    store_slot(&mut func, g_u, 5, xor_self);

    // Signed i32 identities: (0i + x_i) & -1i -> x_i, and x_i ^ x_i -> 0i
    let i_0 = raw(&mut func, Expression::Literal(Literal::I32(0)));
    let i_neg1 = raw(&mut func, Expression::Literal(Literal::I32(-1)));
    let i_add0 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Add,
            left: i_0,
            right: x_i,
        },
    );
    let i_and_all = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::And,
            left: i_add0,
            right: i_neg1,
        },
    );
    store_slot(&mut func, g_i, 0, i_and_all);
    let i_xor_self = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::ExclusiveOr,
            left: x_i,
            right: x_i,
        },
    );
    store_slot(&mut func, g_i, 1, i_xor_self);

    // --- IEEE-754 Float NON-folding safety checks! ---
    // x_f + 0.0f MUST NOT fold (because -0.0 + +0.0 == +0.0, changing sign of -0.0!)
    // x_f * 0.0f MUST NOT fold (because NaN * 0.0 == NaN and Inf * 0.0 == NaN!)
    // x_f * 1.0f MUST NOT fold (sNaN quietening / float fidelity)
    let f_0 = raw(&mut func, Expression::Literal(Literal::F32(0.0)));
    let f_1 = raw(&mut func, Expression::Literal(Literal::F32(1.0)));
    let f_add0 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Add,
            left: x_f,
            right: f_0,
        },
    );
    let f_mul0 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Multiply,
            left: x_f,
            right: f_0,
        },
    );
    let f_mul1 = emit(
        &mut func,
        Expression::Binary {
            op: BinaryOperator::Multiply,
            left: x_f,
            right: f_1,
        },
    );
    store_slot(&mut func, g_f, 0, f_add0);
    store_slot(&mut func, g_f, 1, f_mul0);
    store_slot(&mut func, g_f, 2, f_mul1);

    let fn_handle = module.functions.append(func, Span::UNDEFINED);

    // Add minimal compute entry point calling `test_ops`.
    let mut ep_fn = Function {
        name: Some("main".to_string()),
        ..Function::default()
    };
    let a0 = ep_fn
        .expressions
        .append(Expression::Literal(Literal::U32(7)), Span::UNDEFINED);
    let a1 = ep_fn
        .expressions
        .append(Expression::Literal(Literal::I32(-3)), Span::UNDEFINED);
    let a2 = ep_fn
        .expressions
        .append(Expression::Literal(Literal::F32(2.5)), Span::UNDEFINED);
    let a3 = ep_fn
        .expressions
        .append(Expression::Literal(Literal::Bool(false)), Span::UNDEFINED);
    ep_fn.body.push(
        Statement::Call {
            function: fn_handle,
            arguments: vec![a0, a1, a2, a3],
            result: None,
        },
        Span::UNDEFINED,
    );
    module.entry_points.push(naga::EntryPoint {
        name: "main".to_string(),
        stage: NagaStage::Compute,
        early_depth_test: None,
        workgroup_size: [1, 1, 1],
        workgroup_size_overrides: None,
        function: ep_fn,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.constants_folded >= 4,
        "expected >= 4 constants folded, got {stats:?}"
    );
    assert!(
        stats.identities_simplified >= 14,
        "expected >= 14 integer/bool identities simplified, got {stats:?}"
    );
    assert!(
        stats.dead_emits_trimmed >= 14,
        "expected dead pure emits trimmed, got {stats:?}"
    );

    // Verify exact folded results in WGSL:
    assert!(wgsl.contains("buf_u[0] = 4u;"), "wrapping add:\n{wgsl}");
    assert!(wgsl.contains("buf_u[1] = 8u;"), "masked shift:\n{wgsl}");
    assert!(wgsl.contains("buf_u[2] = 5u;"), "safe div:\n{wgsl}");
    assert!(wgsl.contains("buf_u[3] = 99u;"), "const select:\n{wgsl}");
    assert!(wgsl.contains("buf_u[4] = x_u;"), "u32 identity chain:\n{wgsl}");
    assert!(wgsl.contains("buf_u[5] = 0u;"), "pure x_u ^ x_u:\n{wgsl}");
    assert!(wgsl.contains("buf_i[0] = x_i;"), "i32 identity chain:\n{wgsl}");
    assert!(wgsl.contains("buf_i[1] = 0i;"), "pure x_i ^ x_i:\n{wgsl}");

    // Verify IEEE-754 float ops were NOT folded!
    assert!(
        wgsl.contains("buf_f[0] = (x_f + 0f);"),
        "float `x + 0.0` must NOT be folded:\n{wgsl}"
    );
    assert!(
        wgsl.contains("buf_f[1] = (x_f * 0f);"),
        "float `x * 0.0` must NOT be folded:\n{wgsl}"
    );
    assert!(
        wgsl.contains("buf_f[2] = (x_f * 1f);"),
        "float `x * 1.0` must NOT be folded:\n{wgsl}"
    );
}

#[test]
fn skips_division_and_modulo_by_zero_and_i32_min_overflow_safely() {
    let mut module = Module::default();
    let u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let i32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::I32),
        },
        Span::UNDEFINED,
    );

    let mut func = Function {
        name: Some("div_guard".to_string()),
        result: Some(FunctionResult {
            ty: i32_ty,
            binding: None,
        }),
        ..Function::default()
    };
    let u10 = func
        .expressions
        .append(Expression::Literal(Literal::U32(10)), Span::UNDEFINED);
    let u0 = func
        .expressions
        .append(Expression::Literal(Literal::U32(0)), Span::UNDEFINED);
    let div0 = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Divide,
            left: u10,
            right: u0,
        },
        Span::UNDEFINED,
    );
    let mod0 = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Modulo,
            left: u10,
            right: u0,
        },
        Span::UNDEFINED,
    );
    let imin = func
        .expressions
        .append(Expression::Literal(Literal::I32(i32::MIN)), Span::UNDEFINED);
    let ineg1 = func
        .expressions
        .append(Expression::Literal(Literal::I32(-1)), Span::UNDEFINED);
    let idiv_ovf = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Divide,
            left: imin,
            right: ineg1,
        },
        Span::UNDEFINED,
    );
    let div0_i = func.expressions.append(
        Expression::As {
            expr: div0,
            kind: ScalarKind::Sint,
            convert: None,
        },
        Span::UNDEFINED,
    );
    let mod0_i = func.expressions.append(
        Expression::As {
            expr: mod0,
            kind: ScalarKind::Sint,
            convert: None,
        },
        Span::UNDEFINED,
    );
    let sum1 = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Add,
            left: div0_i,
            right: mod0_i,
        },
        Span::UNDEFINED,
    );
    let sum2 = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Add,
            left: sum1,
            right: idiv_ovf,
        },
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Emit(Range::new_from_bounds(div0, sum2)),
        Span::UNDEFINED,
    );
    func.body
        .push(Statement::Return { value: Some(sum2) }, Span::UNDEFINED);

    let _ = u32_ty;
    let stats = optimize_function(&module, &mut func);
    assert_eq!(
        stats.constants_folded, 0,
        "div/mod by zero and i32::MIN / -1 must NOT fold (got {stats:?})"
    );
}

#[test]
fn folds_splat_and_compose_access_index_without_misindexing_flattened_vectors() {
    let src = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = src[i];
    let b = src[i + 1u];
    let c = src[i + 2u];
    let d = src[i + 3u];

    // Splat + AccessIndex -> scalar value `a`
    let splat_v = vec4<f32>(a);
    let from_splat = splat_v.z;

    // 1-to-1 scalar Compose + AccessIndex -> exact component `c`
    let comp_v = vec4<f32>(a, b, c, d);
    let from_comp = comp_v.z;

    // Flattened vector Compose (`vec4<f32>(vec2<f32>(a, b), c, d)`) has 3 sub-components,
    // NOT 4 — index 2 is `c` (component 1 of Compose!), NOT `components[2]` (`d`).
    // Pass must NOT mis-index this!
    let flat_v = vec4<f32>(vec2<f32>(a, b), c, d);
    let from_flat = flat_v.w;

    dst[i] = from_splat + from_comp + from_flat;
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.constants_folded >= 2,
        "expected Splat + Compose AccessIndex to fold, got {stats:?}\nWGSL:\n{wgsl}"
    );
    // `splat_v` and `comp_v` should be completely eliminated; only `flat_v` keeps a vec4 compose.
    assert_eq!(
        wgsl.matches("vec4<f32>(").count(),
        1,
        "only the flattened vec4 compose should remain:\n{wgsl}"
    );
    assert!(
        wgsl.contains(".w"),
        "flattened vec4 .w access must remain intact (never mis-indexed):\n{wgsl}"
    );
}

#[test]
fn prunes_dead_locals_while_preserving_loaded_pointer_passed_and_atomic_vars() {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> out_buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> atom_buf: array<atomic<u32>>;

fn bump_ptr(p: ptr<function, u32>, delta: u32) {
    *p = *p + delta;
}

fn side_effect_write(idx: u32) -> u32 {
    return atomicAdd(&atom_buf[idx], 1u);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // 1. Dead unused local (never loaded, never stored) -> PRUNED
    var dead_unused: u32;

    // 2. Dead written-only local with pure stores -> PRUNED
    var dead_pure: u32 = 0u;
    dead_pure = (idx + 10u) * 2u;

    // 3. Dead local written with impure call result -> PRESERVED (call side effect intact)
    var dead_impure: u32 = 0u;
    dead_impure = side_effect_write(idx);

    // 4. Local passed by pointer to helper function -> MUST BE PRESERVED!
    var ptr_local: u32 = 5u;
    bump_ptr(&ptr_local, idx);

    // 5. Normal live local (loaded) -> MUST BE PRESERVED!
    var live_local: u32 = 1u;
    live_local = ptr_local + 3u;

    out_buf[idx] = live_local;
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.dead_locals_removed >= 2,
        "expected dead_unused and dead_pure locals pruned, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("dead_unused"),
        "dead_unused must be removed:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("dead_pure"),
        "dead_pure must be removed:\n{wgsl}"
    );
    assert!(
        wgsl.contains("ptr_local"),
        "pointer-passed local must be preserved:\n{wgsl}"
    );
    assert!(
        wgsl.contains("bump_ptr(") && wgsl.contains("&ptr_local"),
        "call with pointer to local must be preserved:\n{wgsl}"
    );
    assert!(
        wgsl.contains("side_effect_write("),
        "side-effecting call must be preserved:\n{wgsl}"
    );
    assert!(
        wgsl.contains("atomicAdd("),
        "atomic operation must be preserved:\n{wgsl}"
    );
    assert!(
        wgsl.contains("live_local"),
        "loaded local must be preserved:\n{wgsl}"
    );
}

#[test]
fn preserves_local_used_directly_in_atomic_statement() {
    // Construct a Naga IR function where a LocalVariable handle appears as an Atomic pointer
    // operand (verifying `collect_live_locals` never prunes any local referenced by `Statement::Atomic`).
    let mut module = Module::default();
    let u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let atomic_u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Atomic(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let mut func = Function {
        name: Some("atomic_guard".to_string()),
        ..Function::default()
    };
    let dead_lv = func.local_variables.append(
        LocalVariable {
            name: Some("dead_lv".to_string()),
            ty: u32_ty,
            init: None,
        },
        Span::UNDEFINED,
    );
    let atomic_lv = func.local_variables.append(
        LocalVariable {
            name: Some("atomic_lv".to_string()),
            ty: atomic_u32_ty,
            init: None,
        },
        Span::UNDEFINED,
    );
    let dead_ptr = func
        .expressions
        .append(Expression::LocalVariable(dead_lv), Span::UNDEFINED);
    let one = func
        .expressions
        .append(Expression::Literal(Literal::U32(1)), Span::UNDEFINED);
    func.body.push(
        Statement::Store {
            pointer: dead_ptr,
            value: one,
        },
        Span::UNDEFINED,
    );
    let atomic_ptr = func
        .expressions
        .append(Expression::LocalVariable(atomic_lv), Span::UNDEFINED);
    let atomic_res = func.expressions.append(
        Expression::AtomicResult {
            ty: u32_ty,
            comparison: false,
        },
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Atomic {
            pointer: atomic_ptr,
            fun: naga::AtomicFunction::Add,
            value: one,
            result: Some(atomic_res),
        },
        Span::UNDEFINED,
    );

    let stats = optimize_function(&module, &mut func);
    assert_eq!(stats.dead_locals_removed, 1, "only dead_lv should be pruned");
    assert_eq!(func.local_variables.len(), 1);
    let (_, surviving) = func.local_variables.iter().next().unwrap();
    assert_eq!(surviving.name.as_deref(), Some("atomic_lv"));
}

#[test]
fn trims_unreferenced_pure_emits_while_keeping_loads_and_texture_samples() {
    let src = r#"
@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var samp0: sampler;
@group(0) @binding(2) var<storage, read> src_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Side-effecting / memory expressions whose results are unused must remain emitted.
    let unused_sample = textureSample(tex0, samp0, uv);
    let unused_load = src_buf[0u];

    // Pure dead expressions built from pure inputs must be trimmed.
    let dead_pure_1 = uv.x + uv.y;
    let dead_pure_2 = dead_pure_1 * 2.0;

    out_buf[0u] = uv.x;
    return vec4<f32>(uv.x, uv.y, 0.0, 1.0);
}
"#;

    let mut module = parse_wgsl(src);
    let before_exprs = count_module_exprs(&module);
    let stats = optimize_module(&mut module);
    let after_exprs = count_module_exprs(&module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.dead_emits_trimmed >= 2,
        "expected unreferenced pure emits trimmed, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(
        after_exprs < before_exprs,
        "expected expression count drop ({before_exprs} -> {after_exprs})"
    );
    assert!(
        wgsl.contains("textureSample("),
        "ImageSample must not be dropped:\n{wgsl}"
    );
    assert!(
        wgsl.contains("src_buf[0]"),
        "Load must not be dropped:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("dead_pure_1") && !wgsl.contains("dead_pure_2"),
        "unreferenced pure expressions must be trimmed:\n{wgsl}"
    );
}

#[test]
fn end_to_end_rdna2_to_wgsl_reduces_size_and_passes_validation_and_roundtrip() {
    // Build a realistic RDNA2 fragment shader with:
    //  * Multiple VGPR integer/bitwise ops (which emit `bitcast<f32>(bitcast<u32>(...))` and
    //    `bitcast<u32>(bitcast<f32>(...))` chains during lowering).
    //  * Identity ops (`v_add_nc_u32` + 0, `v_or_b32` | 0, `v_xor_b32` ^ 0, `v_and_b32` & -1,
    //    `v_lshlrev_b32` << 0, `s_add_u32` + 0, `s_mul_i32` * 1).
    //  * Dead registers written with pure constants (`v15 = 1.0`, `s12 = 42`) and never read.
    let mut words = Vec::new();
    // Interpolate attr0.xy into v0, v1
    words.push(enc_vintrp(0, 0, 0, 0, 0));
    words.push(enc_vintrp(0, 1, 0, 1, 0));
    // Dead pure register writes (never loaded)
    words.push(enc_vop1(0x01, 15, 242)); // v_mov_b32 v15, 1.0 (dead!)
    words.push(enc_sop1(0x03, 12, 128 + 42)); // s_mov_b32 s12, 42 (dead!)
    // Bitwise / integer VGPR chain with identity operands
    words.push(enc_vop2(0x25, 2, 128 + 0, 0)); // v_add_nc_u32 v2, 0, v0
    words.push(enc_vop2(0x1C, 3, 128 + 0, 2)); // v_or_b32 v3, 0, v2
    words.push(enc_vop2(0x1D, 4, 128 + 0, 3)); // v_xor_b32 v4, 0, v3
    words.push(enc_vop2(0x1B, 5, 193, 4)); // v_and_b32 v5, -1, v4
    words.push(enc_vop2(0x1A, 6, 128 + 0, 5)); // v_lshlrev_b32 v6, 0, v5
    words.push(enc_vop2(0x16, 7, 128 + 0, 6)); // v_lshrrev_b32 v7, 0, v6
    // Self-idempotent ops
    words.push(enc_vop2(0x1B, 8, 256 + 7, 7)); // v_and_b32 v8, v7, v7
    words.push(enc_vop2(0x1C, 9, 256 + 8, 8)); // v_or_b32 v9, v8, v8
    // Scalar identity ops feeding a live VGPR
    words.push(enc_sop2(0x00, 4, 2, 128 + 0)); // s_add_u32 s4, s2, 0
    words.push(enc_sop2(0x2C, 5, 4, 128 + 1)); // s_absdiff_i32 / or s_mul_i32 below
    words.push(enc_sop2(0x24, 6, 4, 128 + 1)); // s_mul_i32 s6, s4, 1
    words.push(enc_vop2(0x25, 10, 6, 9)); // v_add_nc_u32 v10, s6, v9
    // Live float arithmetic (must keep IEEE-754 float ops intact)
    words.push(enc_vop2(0x03, 11, 256 + 1, 10)); // v_add_f32 v11, v1, v10
    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [10, 11, 8, 9]));
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let out1 = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Pixel,
            ..Default::default()
        },
    )
    .expect("RDNA2 translation must succeed");
    let out2 = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Pixel,
            ..Default::default()
        },
    )
    .expect("second RDNA2 translation must succeed");

    // Determinism check
    assert_eq!(out1.wgsl, out2.wgsl, "optimized WGSL must be deterministic");
    assert_eq!(out1.ir_opt, out2.ir_opt, "IrOptStats must be deterministic");
    assert!(out1.naga_validated);

    // Verify tangible IR optimization fired across multiple categories:
    assert!(
        out1.ir_opt.bitcasts_cancelled > 0,
        "expected bitcast cancellations in RDNA2 VGPR chain, got {:?}",
        out1.ir_opt
    );
    assert!(
        out1.ir_opt.identities_simplified > 0,
        "expected integer identity simplifications, got {:?}",
        out1.ir_opt
    );
    assert!(
        out1.ir_opt.dead_locals_removed >= 2,
        "expected dead locals (v_15_, s_12_, etc.) removed, got {:?}",
        out1.ir_opt
    );
    assert!(
        out1.ir_opt.dead_emits_trimmed > 0,
        "expected dead pure emits trimmed, got {:?}",
        out1.ir_opt
    );

    // Dead v15 and s12 must be gone from emitted WGSL.
    assert!(
        !out1.wgsl.contains("v_15_") && !out1.wgsl.contains("s_12_"),
        "dead registers should be pruned:\n{}",
        out1.wgsl
    );

    // Measure concrete WGSL byte & expression reduction on a synthetic unoptimized Naga module
    // mirroring the exact RDNA2 VGPR bitcast/identity pattern before vs after `optimize_module`.
    let unopt_wgsl = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var dead_v: f32 = 0.0;
    var dead_s: u32 = 0u;
    dead_v = 1.0;
    dead_s = (40u + 2u) * 1u;
    let x = src[gid.x];
    let a = bitcast<f32>((bitcast<u32>(x) + 0u));
    let b = bitcast<f32>((bitcast<u32>(a) | 0u));
    let c = bitcast<f32>((bitcast<u32>(b) ^ 0u));
    let d = bitcast<f32>((bitcast<u32>(c) & 0xffffffffu));
    let e = bitcast<f32>((bitcast<u32>(d) << 0u));
    let f = bitcast<f32>((bitcast<u32>(e) >> 0u));
    let g = bitcast<f32>(bitcast<u32>(f));
    dst[gid.x] = g;
}
"#;
    let mut mod_unopt = parse_wgsl(unopt_wgsl);
    let wgsl_before = validate_and_emit_wgsl(&mod_unopt);
    let exprs_before = count_module_exprs(&mod_unopt);
    let opt_stats = optimize_module(&mut mod_unopt);
    let wgsl_after = validate_and_emit_wgsl(&mod_unopt);
    let exprs_after = count_module_exprs(&mod_unopt);

    assert!(
        wgsl_after.len() < wgsl_before.len(),
        "expected tangible WGSL byte reduction ({} -> {} bytes)\nBEFORE:\n{wgsl_before}\nAFTER:\n{wgsl_after}",
        wgsl_before.len(),
        wgsl_after.len()
    );
    assert!(
        exprs_after < exprs_before,
        "expected tangible Naga expression count reduction ({exprs_before} -> {exprs_after})"
    );
    assert!(
        opt_stats.bitcasts_cancelled >= 7 && opt_stats.identities_simplified >= 6,
        "expected full bitcast+identity chain collapse, got {opt_stats:?}\nAFTER:\n{wgsl_after}"
    );
    assert!(
        !wgsl_after.contains("bitcast<"),
        "all redundant round-trip bitcasts in chain must collapse to zero:\n{wgsl_after}"
    );

    // Verify JSON summary surfaces `ir_opt` cleanly without host paths.
    let json = out1.to_json();
    assert!(json.contains("\"ir_opt\""), "JSON:\n{json}");
    assert!(json.contains("\"bitcasts_cancelled\""), "JSON:\n{json}");
    assert!(json.contains("\"dead_locals_removed\""), "JSON:\n{json}");

    // Round-trip parse & validate the final RDNA2-lowered WGSL text.
    let reparsed = parse_wgsl(&out1.wgsl);
    let _ = validate_and_emit_wgsl(&reparsed);
}

#[test]
fn ir_opt_stats_helpers_merge_and_zero_check() {
    let mut a = IrOptStats::default();
    assert!(a.is_zero());
    assert_eq!(a.total_changes(), 0);

    let b = IrOptStats {
        bitcasts_cancelled: 3,
        constants_folded: 2,
        identities_simplified: 5,
        dead_locals_removed: 1,
        dead_emits_trimmed: 4,
    };
    assert!(!b.is_zero());
    assert_eq!(b.total_changes(), 15);
    a.merge_from(b);
    assert_eq!(a, b);
}
