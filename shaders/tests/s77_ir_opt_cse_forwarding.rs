//! S77 (#2164, Refs #663, #482, #2137): Intra-block Store-to-Load local variable forwarding
//! and intra-block Common Subexpression Elimination (CSE) soundness, validation, and end-to-end
//! RDNA2 -> WGSL reduction tests.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, AtomicFunction, BinaryOperator, Expression, Function, FunctionArgument,
    FunctionResult, GlobalVariable, Literal, LocalVariable, Module, Range, ResourceBinding,
    Scalar, ShaderStage as NagaStage, Span, Statement, StorageAccess, Type, TypeInner,
};
use playgta6_shaders::ir_opt::{optimize_function, optimize_module};
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
    let reparsed = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| {
        panic!(
            "WGSL round-trip parse failed:\n{}\n\nWGSL:\n{wgsl}",
            e.emit_to_string(&wgsl)
        )
    });
    let mut revalidator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    revalidator.validate(&reparsed).unwrap_or_else(|e| {
        panic!(
            "WGSL round-trip validation failed:\n{}\n\nWGSL:\n{wgsl}",
            e.emit_to_string(&wgsl)
        )
    });
    wgsl
}

fn parse_wgsl(src: &str) -> Module {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
        panic!(
            "WGSL parse failed:\n{}\n\nSRC:\n{src}",
            e.emit_to_string(src)
        )
    });
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    validator.validate(&module).unwrap_or_else(|e| {
        panic!(
            "input WGSL validation failed:\n{}\n\nSRC:\n{src}",
            e.emit_to_string(src)
        )
    });
    module
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

fn count_module_locals(module: &Module) -> usize {
    let ep_count: usize = module
        .entry_points
        .iter()
        .map(|ep| ep.function.local_variables.len())
        .sum();
    let fn_count: usize = module
        .functions
        .iter()
        .map(|(_, f)| f.local_variables.len())
        .sum();
    ep_count + fn_count
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

fn enc_sop2(op: u8, sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
    (0b10u32 << 30)
        | (u32::from(op & 0x7F) << 23)
        | (u32::from(sdst & 0x7F) << 16)
        | (u32::from(ssrc1) << 8)
        | u32::from(ssrc0)
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

#[test]
fn forwards_straight_line_stores_and_initializers_and_prunes_resulting_dead_locals() {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> out_u: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_f: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;

    // Initializer-only local forwarded into arithmetic + dead-local pruned.
    var init_const: u32 = 42u;
    let from_init = x + init_const;

    // Multi-hop VGPR-like store->load->bitcast->store->load chain.
    var t0: f32 = 0.0;
    var t1: f32 = 0.0;
    var t2: f32 = 0.0;
    t0 = bitcast<f32>(from_init);
    t1 = bitcast<f32>((bitcast<u32>(t0) + 0u) ^ 7u);
    t2 = bitcast<f32>((bitcast<u32>(t1) | 0u) * 1u);

    out_u[x] = bitcast<u32>(t2);
    out_f[x] = t2;
}
"#;

    let mut module = parse_wgsl(src);
    let locals_before = count_module_locals(&module);
    let exprs_before = count_module_exprs(&module);
    let wgsl_before = validate_and_emit_wgsl(&module);

    let stats = optimize_module(&mut module);
    let locals_after = count_module_locals(&module);
    let exprs_after = count_module_exprs(&module);
    let wgsl_after = validate_and_emit_wgsl(&module);

    assert_eq!(locals_before, 4);
    assert_eq!(
        locals_after, 0,
        "all 4 straight-line locals should forward and prune cleanly:\n{wgsl_after}"
    );
    assert!(
        stats.locals_forwarded >= 4,
        "expected >= 4 forwarded local loads, got {stats:?}"
    );
    assert!(
        stats.dead_locals_removed == 4,
        "expected all 4 dead locals removed after forwarding, got {stats:?}"
    );
    assert!(
        stats.bitcasts_cancelled >= 3,
        "forwarding across locals should expose round-trip bitcasts, got {stats:?}"
    );
    assert!(exprs_after < exprs_before);
    assert!(wgsl_after.len() < wgsl_before.len());
    assert!(
        !wgsl_after.contains("var init_const")
            && !wgsl_after.contains("var t0")
            && !wgsl_after.contains("var t1")
            && !wgsl_after.contains("var t2"),
        "eliminated locals must not appear in WGSL:\n{wgsl_after}"
    );
    // Exact final computation `((gid.x + 42u) ^ 7u)` must be preserved!
    assert!(wgsl_after.contains("42u"), "WGSL:\n{wgsl_after}");
    assert!(wgsl_after.contains("7u"), "WGSL:\n{wgsl_after}");
}

#[test]
fn soundness_across_if_else_loop_backedge_and_switch_never_misforwards() {
    let src = r#"
@group(0) @binding(0) var<storage, read> cond_buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let sel = cond_buf[idx];

    // 1. If/Else: `if_var` modified in accept MUST NOT forward pre-if value (`10u`) after If,
    //    whereas `untouched` (never written inside If) CAN still forward `77u` after If!
    var if_var: u32 = 10u;
    var untouched: u32 = 77u;
    if (sel > 0u) {
        // Intra-block forwarding inside accept branch still works cleanly.
        var branch_tmp: u32 = 0u;
        branch_tmp = idx + 5u;
        if_var = branch_tmp;
    } else {
        if_var = idx + 9u;
    }

    // 2. Loop back-edge: `i` and `acc` are loop-carried! Must NEVER forward `0u` into the loop
    //    body or after the loop!
    var i: u32 = 0u;
    var acc: u32 = 1u;
    loop {
        if (i >= sel) {
            break;
        }
        var loop_tmp: u32 = 0u;
        loop_tmp = i + 2u;
        acc = acc + loop_tmp;
        i = i + 1u;
    }

    // 3. Switch: `sw_var` written in a case arm MUST NOT forward pre-switch value (`3u`) after Switch.
    var sw_var: u32 = 3u;
    switch (sel) {
        case 0u: {
            sw_var = idx + 100u;
        }
        case 1u: {
            sw_var = idx + 200u;
        }
        default: {
            sw_var = idx + 300u;
        }
    }

    out_buf[0] = if_var;
    out_buf[1] = untouched;
    out_buf[2] = acc;
    out_buf[3] = sw_var;
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    // `untouched`, `branch_tmp`, and `loop_tmp` are all forwarded and eliminated!
    assert!(
        stats.locals_forwarded >= 3,
        "expected untouched + branch_tmp + loop_tmp forwarded, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("var untouched")
            && !wgsl.contains("var branch_tmp")
            && !wgsl.contains("var loop_tmp"),
        "straight-line locals inside/around CF should be pruned:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_buf[1] = 77u;"),
        "untouched pure local across If/Loop/Switch should forward to 77u:\n{wgsl}"
    );

    // `if_var`, `i`, `acc`, and `sw_var` MUST remain live variables!
    assert!(wgsl.contains("var if_var"), "if_var must remain live:\n{wgsl}");
    assert!(wgsl.contains("var i"), "loop counter i must remain live:\n{wgsl}");
    assert!(wgsl.contains("var acc"), "loop accumulator acc must remain live:\n{wgsl}");
    assert!(wgsl.contains("var sw_var"), "sw_var must remain live:\n{wgsl}");
    assert!(
        !wgsl.contains("out_buf[0] = 10u;")
            && !wgsl.contains("out_buf[2] = 1u;")
            && !wgsl.contains("out_buf[3] = 3u;"),
        "pre-CF values must NEVER be misforwarded past modifying If/Loop/Switch:\n{wgsl}"
    );
}

#[test]
fn soundness_invalidates_load_dependent_entries_across_intervening_stores_and_escaped_locals() {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> gbuf: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<u32>;

fn mutate_ptr(p: ptr<function, u32>, v: u32) {
    *p = *p + v;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // 1. Escaped local passed to `mutate_ptr`: MUST NOT forward `5u` past the call!
    var escaped: u32 = 5u;
    mutate_ptr(&escaped, idx);
    out_buf[0] = escaped;

    // 2. Local `b` holds a value loaded from mutable local `a` (whose initial value is unknown
    //    because `a` was written in a conditional branch), then `a` is overwritten before `b` is read!
    var a: u32 = 1u;
    if (idx > 0u) {
        a = idx + 10u;
    }
    var b: u32 = 0u;
    b = a;
    a = 999u;
    out_buf[1] = b; // MUST read OLD `a`, NOT `999u`!
    out_buf[2] = a; // CAN forward `999u`!

    // 3. Local `c` holds a value loaded from mutable storage `gbuf[0]`, then `gbuf[0]` is stored
    //    before `c` is read!
    var c: u32 = 0u;
    c = gbuf[0];
    gbuf[0] = 1234u;
    out_buf[3] = c; // MUST read OLD `gbuf[0]`, NOT `1234u`!

    // 4. Partial vector component write (`v.x = ...`) must invalidate whole-vector `v`!
    var v: vec4<u32> = vec4<u32>(1u, 2u, 3u, 4u);
    v.x = idx + 50u;
    out_buf[4] = v.x;
}
"#;

    let mut module = parse_wgsl(src);
    let _stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    // 1. Escaped local must NOT forward to 5u.
    assert!(wgsl.contains("var escaped"), "escaped local must stay:\n{wgsl}");
    assert!(
        !wgsl.contains("out_buf[0] = 5u;"),
        "escaped local must NOT forward pre-call value:\n{wgsl}"
    );
    // 2. `out_buf[1]` MUST NOT be `999u`, while `out_buf[2]` safely forwards `999u`!
    assert!(
        !wgsl.contains("out_buf[1] = 999u;"),
        "intervening store to `a` must invalidate `b`:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_buf[2] = 999u;"),
        "store `a = 999u` should safely forward to `out_buf[2]`:\n{wgsl}"
    );
    // 3. `out_buf[3]` MUST NOT be `1234u`!
    assert!(
        !wgsl.contains("out_buf[3] = 1234u;"),
        "intervening storage store must invalidate load-dependent `c`:\n{wgsl}"
    );
    // 4. Partial vector write `v.x = idx + 50u` must remain intact (not replaced by `1u`).
    assert!(
        !wgsl.contains("out_buf[4] = 1u;"),
        "partial component write must invalidate whole-vector known value:\n{wgsl}"
    );
}

#[test]
fn soundness_preserves_atomic_escaped_local_and_self_updating_local() {
    // Directly build Naga IR with an Atomic on a LocalVariable (ensuring escape analysis skips it)
    // and a self-updating local `x = x + 1u; return x;` after an unknown branch.
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
        name: Some("atomic_and_self_update".to_string()),
        result: Some(FunctionResult {
            ty: u32_ty,
            binding: None,
        }),
        ..Function::default()
    };
    let atom_lv = func.local_variables.append(
        LocalVariable {
            name: Some("atom_lv".to_string()),
            ty: atomic_u32_ty,
            init: None,
        },
        Span::UNDEFINED,
    );
    let one = func
        .expressions
        .append(Expression::Literal(Literal::U32(1)), Span::UNDEFINED);
    let atom_ptr = func
        .expressions
        .append(Expression::LocalVariable(atom_lv), Span::UNDEFINED);
    let atom_res = func.expressions.append(
        Expression::AtomicResult {
            ty: u32_ty,
            comparison: false,
        },
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Atomic {
            pointer: atom_ptr,
            fun: AtomicFunction::Add,
            value: one,
            result: Some(atom_res),
        },
        Span::UNDEFINED,
    );
    func.body
        .push(Statement::Return { value: Some(atom_res) }, Span::UNDEFINED);

    let stats = optimize_function(&module, &mut func);
    assert_eq!(stats.locals_forwarded, 0);
    assert_eq!(stats.dead_locals_removed, 0);
    assert_eq!(func.local_variables.len(), 1);
}

#[test]
fn soundness_guards_wgsl_const_eval_traps_on_forwarded_constants() {
    // Verify that forwarding constants NEVER turns runtime-valid expressions into WGSL/Tint
    // const-eval errors: bitcast<f32>(+Inf), 10u / 0u, sqrt(-1.0), arr[99u], 1e38f * 10.0f.
    let src = r#"
@group(0) @binding(0) var<storage, read_write> out_f: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_u: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var inf_bits: u32 = 0x7f800000u;
    var zero_u: u32 = 0u;
    var neg_one_f: f32 = -1.0;
    var big_f: f32 = 1e38;
    var ten_f: f32 = 10.0;
    var oob_idx: u32 = 99u;
    var arr: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 4u);

    let f_inf = bitcast<f32>(inf_bits);
    let q = 10u / zero_u;
    let s = sqrt(neg_one_f);
    let ovf = big_f * ten_f;
    let elem = arr[oob_idx];

    out_f[gid.x] = f_inf + s + ovf;
    out_u[gid.x] = q + elem;
}
"#;

    let mut module = parse_wgsl(src);
    let _stats = optimize_module(&mut module);
    // Must pass BOTH Naga validation AND strict WGSL round-trip parsing/const-eval!
    let wgsl = validate_and_emit_wgsl(&module);
    assert!(!wgsl.contains("bitcast<f32>(2139095040u)"), "WGSL:\n{wgsl}");
    assert!(!wgsl.contains("10u / 0u"), "WGSL:\n{wgsl}");
    assert!(!wgsl.contains("sqrt(-1"), "WGSL:\n{wgsl}");
}

#[test]
fn cse_deduplicates_pure_expressions_and_commutative_integer_ops_within_block_and_dominating_scope() {
    let src = r#"
@group(0) @binding(0) var<storage, read> in_u: array<u32>;
@group(0) @binding(1) var<storage, read> in_f: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_u: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_f: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = in_u[i];
    let b = in_u[i + 1u];
    let fx = in_f[i];
    let fy = in_f[i + 1u];

    // Repeated address/bitfield calculations + commutative integer ops (`a + b` vs `b + a`,
    // `a * b` vs `b * a`, `a & b` vs `b & a`, `a | b` vs `b | a`, `a ^ b` vs `b ^ a`).
    let addr1 = ((i * 4u) + 16u) & 255u;
    let addr2 = (255u & (16u + (4u * i)));

    let sum1 = a + b;
    let sum2 = b + a;
    let xor1 = a ^ b;
    let xor2 = b ^ a;
    let eq1 = (a == b);
    let eq2 = (b == a);

    // Pure math CSE (`clamp`, `fma`).
    let m1 = fma(fx, fy, 0.5);
    let m2 = fma(fx, fy, 0.5);

    // Dominating outer expression reused inside `if` branch!
    var branch_res: u32 = 0u;
    if (eq1 && eq2) {
        let addr_in_if = ((i * 4u) + 16u) & 255u;
        branch_res = addr_in_if + sum2;
    }

    // IEEE-754 float order preservation: `fx + fy` and `fy + fx` MUST NOT commute!
    let fsum_xy = fx + fy;
    let fsum_yx = fy + fx;

    out_u[0] = addr1 + addr2 + sum1 + xor1 + xor2 + branch_res;
    out_f[0] = m1 + m2 + fsum_xy + fsum_yx;
}
"#;

    let mut module = parse_wgsl(src);
    let exprs_before = count_module_exprs(&module);
    let wgsl_before = validate_and_emit_wgsl(&module);

    let stats = optimize_module(&mut module);
    let exprs_after = count_module_exprs(&module);
    let wgsl_after = validate_and_emit_wgsl(&module);

    assert!(
        stats.cse_eliminated >= 8,
        "expected >= 8 CSE eliminations across address/commutative/math/dominating-scope ops, got {stats:?}\nWGSL:\n{wgsl_after}"
    );
    assert!(
        exprs_after < exprs_before,
        "expected expression count drop ({exprs_before} -> {exprs_after})"
    );
    assert!(
        wgsl_after.len() < wgsl_before.len(),
        "expected WGSL byte shrink ({} -> {})",
        wgsl_before.len(),
        wgsl_after.len()
    );
    // `fma(` should appear only ONCE after CSE!
    assert_eq!(
        wgsl_after.matches("fma(").count(),
        1,
        "duplicate pure fma() must be CSE'd to 1 call:\n{wgsl_after}"
    );
    // Float `fx + fy` and `fy + fx` must NOT be commuted into a single add!
    assert!(
        wgsl_after.contains("(fx + fy)") && wgsl_after.contains("(fy + fx)"),
        "float addition order must be preserved (no float commutativity CSE):\n{wgsl_after}"
    );
}

#[test]
fn cse_never_merges_across_sibling_branches_or_side_effecting_and_derivative_ops() {
    let src = r#"
@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var samp0: sampler;
@group(0) @binding(2) var<storage, read_write> sbuf: array<f32>;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Side-effecting / memory / control-flow-sensitive ops MUST NEVER be CSE'd!
    let s1 = textureSample(tex0, samp0, uv);
    let s2 = textureSample(tex0, samp0, uv);
    let d1 = dpdx(uv.x);
    let d2 = dpdx(uv.x);
    let l1 = sbuf[0];
    sbuf[0] = uv.x;
    let l2 = sbuf[0];

    // Identical pure computations in SIBLING If/Else branches and Switch arms MUST NOT reuse
    // handles across non-dominating sibling scopes (would fail Naga dominance validation!).
    var r: f32 = 0.0;
    if (uv.x > 0.5) {
        let k = (uv.x * 3.0) + 1.0;
        r = k + (uv.x * 3.0) + 1.0;
    } else {
        let k = (uv.x * 3.0) + 1.0;
        r = k * 2.0;
    }
    let after_if = (uv.x * 3.0) + 1.0;

    let sel = u32(uv.y > 0.5);
    switch (sel) {
        case 0u: {
            r = r + (uv.y * 5.0);
        }
        default: {
            r = r + (uv.y * 5.0);
        }
    }

    return vec4<f32>(s1.x + s2.x, d1 + d2, l1 + l2, r + after_if);
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    // Must pass strict Naga dominance validation and WGSL round-trip!
    let wgsl = validate_and_emit_wgsl(&module);

    assert_eq!(
        wgsl.matches("textureSample(").count(),
        2,
        "ImageSample must NEVER be CSE'd:\n{wgsl}"
    );
    assert_eq!(
        wgsl.matches("dpdx(").count(),
        2,
        "Derivative dpdx() must NEVER be CSE'd:\n{wgsl}"
    );
    // Both storage loads `sbuf[0]` before and after `sbuf[0] = uv.x;` must stay intact!
    assert!(
        wgsl.matches("sbuf[0]").count() >= 3,
        "both Load(sbuf[0]) and Store(sbuf[0]) must stay intact:\n{wgsl}"
    );
    // Duplicate `(uv.x * 3.0)` inside the `if` branch WAS CSE'd!
    assert!(
        stats.cse_eliminated >= 1,
        "expected intra-branch CSE inside `if`, got {stats:?}\nWGSL:\n{wgsl}"
    );
}

#[test]
fn cse_handles_distinct_duplicate_literal_handles_in_raw_naga_ir() {
    // Directly construct a Naga IR function where two identical `x + 4u` expressions reference
    // TWO DIFFERENT `Expression::Literal(Literal::U32(4))` handles (mirroring RDNA2 lowering!).
    let mut module = Module::default();
    let u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
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
    let g_out = module.global_variables.append(
        GlobalVariable {
            name: Some("out_buf".to_string()),
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

    let mut func = Function {
        name: Some("dup_lit_cse".to_string()),
        arguments: vec![FunctionArgument {
            name: Some("x".to_string()),
            ty: u32_ty,
            binding: None,
        }],
        ..Function::default()
    };

    let arg_a = func
        .expressions
        .append(Expression::FunctionArgument(0), Span::UNDEFINED);
    let lit_a = func
        .expressions
        .append(Expression::Literal(Literal::U32(4)), Span::UNDEFINED);
    let add1 = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Add,
            left: arg_a,
            right: lit_a,
        },
        Span::UNDEFINED,
    );
    let arg_b = func
        .expressions
        .append(Expression::FunctionArgument(0), Span::UNDEFINED);
    let lit_b = func
        .expressions
        .append(Expression::Literal(Literal::U32(4)), Span::UNDEFINED);
    let add2 = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Add,
            left: lit_b,
            right: arg_b,
        },
        Span::UNDEFINED,
    );
    let g_expr = func
        .expressions
        .append(Expression::GlobalVariable(g_out), Span::UNDEFINED);
    let p0 = func.expressions.append(
        Expression::AccessIndex {
            base: g_expr,
            index: 0,
        },
        Span::UNDEFINED,
    );
    let p1 = func.expressions.append(
        Expression::AccessIndex {
            base: g_expr,
            index: 1,
        },
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Emit(Range::new_from_bounds(add1, add1)),
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Emit(Range::new_from_bounds(add2, add2)),
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Emit(Range::new_from_bounds(p0, p1)),
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Store {
            pointer: p0,
            value: add1,
        },
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Store {
            pointer: p1,
            value: add2,
        },
        Span::UNDEFINED,
    );

    let fh = module.functions.append(func, Span::UNDEFINED);
    let mut ep = Function {
        name: Some("main".to_string()),
        ..Function::default()
    };
    let c0 = ep
        .expressions
        .append(Expression::Literal(Literal::U32(9)), Span::UNDEFINED);
    ep.body.push(
        Statement::Call {
            function: fh,
            arguments: vec![c0],
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
        function: ep,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert_eq!(
        stats.cse_eliminated, 1,
        "expected duplicate `x + 4u` / `4u + x` with distinct literal handles to CSE, got {stats:?}"
    );
    assert_eq!(
        wgsl.matches(" + 4u").count(),
        1,
        "only one `+ 4u` computation should survive:\n{wgsl}"
    );
}

#[test]
fn end_to_end_rdna2_to_wgsl_demonstrates_cse_and_forwarding_reductions() {
    // Realistic RDNA2 pixel shader with:
    //  * Interpolated attr0.xy into v0, v1
    //  * Multi-hop VGPR chains (v2 -> v3 -> v4 -> v5 -> v6) carrying repeated bitfield/offset math
    //  * Identical address/offset computations across two parallel VGPR paths (`v7`, `v8`)
    //  * Scalar chain (s2 -> s3 -> s4) feeding live VGPRs
    let mut words = Vec::new();
    words.push(enc_vintrp(0, 0, 0, 0, 0)); // v0 = attr0.x
    words.push(enc_vintrp(0, 1, 0, 1, 0)); // v1 = attr0.y

    // Path A: v2 = u32(v0) + 16; v3 = v2 & 0x3F; v4 = v3 << 2
    words.push(enc_vop2(0x25, 2, 128 + 16, 0)); // v_add_nc_u32 v2, 16, v0
    words.push(enc_vop2(0x1B, 3, 128 + 63, 2)); // v_and_b32 v3, 63, v2
    words.push(enc_vop2(0x1A, 4, 128 + 2, 3)); // v_lshlrev_b32 v4, 2, v3

    // Path B (exact same bitfield/address computation repeated in different VGPRs!):
    // v5 = u32(v0) + 16; v6 = v5 & 0x3F; v7 = v6 << 2
    words.push(enc_vop2(0x25, 5, 128 + 16, 0)); // v_add_nc_u32 v5, 16, v0
    words.push(enc_vop2(0x1B, 6, 128 + 63, 5)); // v_and_b32 v6, 63, v5
    words.push(enc_vop2(0x1A, 7, 128 + 2, 6)); // v_lshlrev_b32 v7, 2, v6

    // Combine with v1 and scalar chain s2 -> s3
    words.push(enc_sop2(0x00, 2, 128 + 7, 128 + 5)); // s_add_u32 s2, 7, 5 -> 12
    words.push(enc_sop2(0x00, 3, 2, 128 + 3)); // s_add_u32 s3, s2, 3 -> 15
    words.push(enc_vop2(0x25, 8, 3, 4)); // v_add_nc_u32 v8, s3, v4
    words.push(enc_vop2(0x25, 9, 3, 7)); // v_add_nc_u32 v9, s3, v7
    words.push(enc_vop2(0x03, 10, 256 + 1, 8)); // v_add_f32 v10, v1, v8
    words.push(enc_vop2(0x03, 11, 256 + 1, 9)); // v_add_f32 v11, v1, v9

    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [8, 9, 10, 11]));
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

    // Determinism & validation
    assert_eq!(out1.wgsl, out2.wgsl, "optimized WGSL must be deterministic");
    assert_eq!(out1.ir_opt, out2.ir_opt, "IrOptStats must be deterministic");
    assert!(out1.naga_validated);

    // Verify all new passes fired significantly on the real RDNA2 shader!
    assert!(
        out1.ir_opt.locals_forwarded >= 10,
        "expected >= 10 forwarded VGPR/SGPR loads, got {:?}\nWGSL:\n{}",
        out1.ir_opt,
        out1.wgsl
    );
    assert!(
        out1.ir_opt.cse_eliminated >= 4,
        "expected >= 4 CSE eliminations on duplicated path A/B math, got {:?}\nWGSL:\n{}",
        out1.ir_opt,
        out1.wgsl
    );
    assert!(
        out1.ir_opt.dead_locals_removed >= 10,
        "expected all temporary VGPR/SGPR locals pruned after forwarding, got {:?}\nWGSL:\n{}",
        out1.ir_opt,
        out1.wgsl
    );

    // Every single temporary VGPR (`var v_0_..v_11_`) and SGPR (`var s_2_`, `var s_3_`) should be gone!
    for v in 0..=11 {
        assert!(
            !out1.wgsl.contains(&format!("var v_{v}_:")),
            "temporary var v_{v}_ should be eliminated:\n{}",
            out1.wgsl
        );
    }
    assert!(
        !out1.wgsl.contains("var s_2_:") && !out1.wgsl.contains("var s_3_:"),
        "temporary scalar locals s_2_/s_3_ should be eliminated:\n{}",
        out1.wgsl
    );
    // Duplicate `63u & ...` and `<< 2u` from Path A & Path B should CSE to ONE single copy!
    assert_eq!(
        out1.wgsl.matches("63u").count(),
        1,
        "duplicate `63u` mask across path A & B must CSE to 1:\n{}",
        out1.wgsl
    );
    assert_eq!(
        out1.wgsl.matches("<< 2u").count(),
        1,
        "duplicate `<< 2u` across path A & B must CSE to 1:\n{}",
        out1.wgsl
    );

    // Verify JSON summary surfaces the new sanitized counters cleanly.
    let json = out1.to_json();
    assert!(json.contains("\"locals_forwarded\""), "JSON:\n{json}");
    assert!(json.contains("\"cse_eliminated\""), "JSON:\n{json}");

    // Round-trip parse & validate final WGSL text.
    let reparsed = parse_wgsl(&out1.wgsl);
    let _ = validate_and_emit_wgsl(&reparsed);
}
