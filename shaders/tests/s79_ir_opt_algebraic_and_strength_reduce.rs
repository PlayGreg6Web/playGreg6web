#![forbid(unsafe_code)]

//! S79 (#2193): Naga IR optimization pass tests for:
//! 1. Unsigned power-of-two strength reduction (`u32` / `vecN<u32>` `/`, `%`, `*`)
//!    with strict negative guards for signed `i32`, floats, and non-powers-of-two.
//! 2. Pure algebraic, bitwise, double-negation, `Select`, and nested-shift identities
//!    with strict IEEE-754 float, shift-overflow, and impure-arm safety guards.
//! 3. Constant-index `AccessIndex` / `Swizzle` folding through `Splat` and 1-to-1 `Compose`
//!    (vectors & arrays), with strict negative guard for sub-vector packed `vec4(vec2, vec2)`.
//! 4. End-to-end RDNA2 -> WGSL validation, JSON stats surface, and determinism.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, BinaryOperator, Expression, Function, FunctionArgument, GlobalVariable, Literal,
    Module, Range, ResourceBinding, Scalar, ShaderStage as NagaStage, Span, Statement,
    StorageAccess, Type, TypeInner, VectorSize,
};
use playgta6_shaders::ir_opt::{IrOptStats, optimize_module};
use playgta6_shaders::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, translate_rdna2_to_wgsl,
};

const BASELINE_WEBGPU_CAPABILITIES: Capabilities =
    Capabilities::SHADER_FLOAT16_IN_FLOAT32.union(Capabilities::CUBE_ARRAY_TEXTURES);

fn parse_wgsl(src: &str) -> Module {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\n\nSRC:\n{src}", e.emit_to_string(src)));
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    validator.validate(&module).unwrap_or_else(|e| {
        panic!(
            "input WGSL validation failed:\n{}\n\nSRC:\n{src}",
            e.emit_to_string(src)
        )
    });
    module
}

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
fn unsigned_pow2_strength_reduction_scalar_and_vector_with_signed_and_non_pow2_guards() {
    let src = r#"
@group(0) @binding(0) var<storage, read> in_u: array<u32>;
@group(0) @binding(1) var<storage, read> in_i: array<i32>;
@group(0) @binding(2) var<storage, read> in_v: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> in_iv: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read_write> out_u: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_i: array<i32>;
@group(0) @binding(6) var<storage, read_write> out_v: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> out_iv: array<vec4<i32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let x = in_u[idx];
    let si = in_i[idx];
    let vx = in_v[idx];
    let vsi = in_iv[idx];

    // Scalar u32 power-of-two strength reductions (must rewrite to >>, &, <<)
    out_u[0] = x / 8u;
    out_u[1] = x % 16u;
    out_u[2] = x * 32u;
    out_u[3] = 64u * x;
    out_u[4] = x / 0x80000000u;
    out_u[5] = x % 0x80000000u;
    out_u[6] = x * 0x80000000u;

    // Scalar u32 non-power-of-two MUST NOT strength reduce
    out_u[7] = x / 3u;
    out_u[8] = x % 6u;
    out_u[9] = x * 5u;

    // Scalar u32 / 1u and % 1u fold via identity
    out_u[10] = x / 1u;
    out_u[11] = gid.x % 1u;

    // Vector vec4<u32> power-of-two strength reductions (both splat/vector and scalar rhs/lhs)
    out_v[0] = vx / vec4<u32>(4u);
    out_v[1] = vx % vec4<u32>(8u);
    out_v[2] = vx * vec4<u32>(16u);
    out_v[3] = vx / 32u;
    out_v[4] = vx % 64u;
    out_v[5] = vx * 128u;
    out_v[6] = 2u * vx;
    out_v[7] = vx * vec4<u32>(4u, 4u, 4u, 4u);

    // Non-uniform vector pow2 compose MUST NOT be mis-reduced
    out_v[8] = vx * vec4<u32>(2u, 4u, 8u, 16u);

    // NEGATIVE SAFETY GUARDS: signed i32 and vec4<i32> MUST NEVER be strength-reduced!
    out_i[0] = si / 4;
    out_i[1] = si % 4;
    out_i[2] = si * 4;
    out_iv[0] = vsi / vec4<i32>(8);
    out_iv[1] = vsi % vec4<i32>(8);
    out_iv[2] = vsi * vec4<i32>(8);
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    // 7 scalar + 8 vector = 15 unsigned pow2 strength reductions
    assert_eq!(
        stats.strength_reductions, 15,
        "expected exactly 15 unsigned pow2 strength reductions, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(
        stats.identities_simplified >= 2,
        "expected x / 1u and gid.x % 1u identities, got {stats:?}\nWGSL:\n{wgsl}"
    );

    // Verify exact scalar u32 strength reductions in emitted WGSL:
    assert!(wgsl.contains("out_u[0] = (x >> 3u);"), "x / 8u -> x >> 3u:\n{wgsl}");
    assert!(wgsl.contains("out_u[1] = (x & 15u);"), "x % 16u -> x & 15u:\n{wgsl}");
    assert!(wgsl.contains("out_u[2] = (x << 5u);"), "x * 32u -> x << 5u:\n{wgsl}");
    assert!(wgsl.contains("out_u[3] = (x << 6u);"), "64u * x -> x << 6u:\n{wgsl}");
    assert!(wgsl.contains("out_u[4] = (x >> 31u);"), "x / 2^31 -> x >> 31u:\n{wgsl}");
    assert!(
        wgsl.contains("out_u[5] = (x & 2147483647u);"),
        "x % 2^31 -> x & 0x7FFFFFFFu:\n{wgsl}"
    );
    assert!(wgsl.contains("out_u[6] = (x << 31u);"), "x * 2^31 -> x << 31u:\n{wgsl}");

    // Verify non-power-of-two u32 ops remained untouched:
    assert!(wgsl.contains("out_u[7] = (x / 3u);"), "non-pow2 div must stay:\n{wgsl}");
    assert!(wgsl.contains("out_u[8] = (x % 6u);"), "non-pow2 mod must stay:\n{wgsl}");
    assert!(wgsl.contains("out_u[9] = (x * 5u);"), "non-pow2 mul must stay:\n{wgsl}");

    // Verify identities x / 1u -> x and gid.x % 1u -> 0u:
    assert!(wgsl.contains("out_u[10] = x;"), "x / 1u -> x:\n{wgsl}");
    assert!(wgsl.contains("out_u[11] = 0u;"), "gid.x % 1u -> 0u:\n{wgsl}");

    // Verify vector vec4<u32> strength reductions use valid vec4<u32> shift/mask operands
    // (note: CSE may share identical `vec4(2u)` / `vec4(7u)` splats into a `let _eN = vec4(...)`):
    assert!(
        wgsl.contains("out_v[0] = (vx >> vec4(2u));")
            || (wgsl.contains("= vec4(2u);") && wgsl.contains("out_v[0] = (vx >> ")),
        "vec4 / 4u:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_v[1] = (vx & vec4(7u));")
            || (wgsl.contains("= vec4(7u);") && wgsl.contains("out_v[1] = (vx & ")),
        "vec4 % 8u:\n{wgsl}"
    );
    assert!(wgsl.contains("out_v[2] = (vx << vec4(4u));"), "vec4 * 16u:\n{wgsl}");
    assert!(wgsl.contains("out_v[3] = (vx >> vec4(5u));"), "vec4 / scalar 32u:\n{wgsl}");
    assert!(wgsl.contains("out_v[4] = (vx & vec4(63u));"), "vec4 % scalar 64u:\n{wgsl}");
    assert!(
        wgsl.contains("out_v[5] = (vx << vec4(7u));")
            || (wgsl.contains("= vec4(7u);") && wgsl.contains("out_v[5] = (vx << ")),
        "vec4 * scalar 128u:\n{wgsl}"
    );
    assert!(wgsl.contains("out_v[6] = (vx << vec4(1u));"), "scalar 2u * vec4:\n{wgsl}");
    assert!(
        wgsl.contains("out_v[7] = (vx << vec4(2u));")
            || (wgsl.contains("= vec4(2u);") && wgsl.contains("out_v[7] = (vx << ")),
        "vec4 * uniform compose:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_v[8] = (vx * vec4<u32>(2u, 4u, 8u, 16u));"),
        "non-uniform vec4 pow2 compose must stay untouched:\n{wgsl}"
    );

    // Verify signed i32 / vec4<i32> ops were NEVER strength-reduced!
    assert!(wgsl.contains("out_i[0] = (si / 4i);"), "signed i32 / 4 MUST NOT become shift:\n{wgsl}");
    assert!(wgsl.contains("out_i[1] = (si % 4i);"), "signed i32 % 4 MUST NOT become mask:\n{wgsl}");
    assert!(wgsl.contains("out_i[2] = (si * 4i);"), "signed i32 * 4 MUST NOT become shift:\n{wgsl}");
    assert!(
        wgsl.contains("out_iv[0] = (vsi / "),
        "signed vec4<i32> / 8 MUST NOT become shift:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_iv[1] = (vsi % "),
        "signed vec4<i32> % 8 MUST NOT become mask:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_iv[2] = (vsi * "),
        "signed vec4<i32> * 8 MUST NOT become shift:\n{wgsl}"
    );
}

#[test]
fn pure_algebraic_bitwise_double_negation_and_nested_shift_identities_with_safety_guards() {
    let src = r#"
@group(0) @binding(0) var<storage, read> in_f: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_u: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_i: array<i32>;
@group(0) @binding(3) var<storage, read_write> out_vu: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> out_vi: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read_write> out_f: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let si = i32(x);
    let vx = vec4<u32>(gid.x, gid.y, gid.z, 7u);
    let vsi = vec4<i32>(si, si + 1, si + 2, si + 3);
    let fx = in_f[x];
    let b = (x == 123u);

    // 1. Same-handle pure integer identities (scalar & vector)
    out_u[0] = x - x;
    out_u[1] = x ^ x;
    out_u[2] = x | x;
    out_u[3] = x & x;
    out_i[0] = si - si;
    out_i[1] = si ^ si;
    out_vu[0] = vx - vx;
    out_vu[1] = vx ^ vx;
    out_vi[0] = vsi - vsi;
    out_vi[1] = vsi ^ vsi;

    // 2. Absorbing bitwise-OR with all-ones (`0xFFFFFFFFu` / `-1i`)
    out_u[4] = x | 0xffffffffu;
    out_u[5] = 0xffffffffu | (x + 1u);
    out_i[2] = si | -1;
    out_i[3] = -1 | (si + 2);

    // 3. Double logical / bitwise / signed-integer negation (`!!b`, `~~x`, `--x`)
    out_u[6] = u32(!!b);
    out_u[7] = ~~x;
    out_i[4] = ~~si;
    out_i[5] = -(-si);
    out_vu[2] = ~~vx;
    out_vi[2] = -(-vsi);

    // 4. Nested constant shift merging (`a + b < 32`)
    out_u[8] = (x << 3u) << 4u;
    out_u[9] = (x >> 5u) >> 6u;
    out_i[6] = (si << 2u) << 7u;
    out_i[7] = (si >> 4u) >> 8u;
    // Chained with strength reduction: `(x * 4u) << 3u` -> `(x << 2u) << 3u` -> `x << 5u`!
    out_u[10] = (x * 4u) << 3u;

    // 5. NEGATIVE SAFETY GUARDS:
    // (a) Shift overflow `a + b >= 32` MUST NOT merge!
    out_u[11] = (x << 20u) << 16u;
    out_u[12] = (x >> 16u) >> 16u;
    out_i[8] = (si >> 20u) >> 15u;

    // (b) Float `fx - fx` and `-(-fx)` MUST NOT fold!
    out_f[0] = fx - fx;
    out_f[1] = -(-fx);

    // (c) Bitcast-to-f32 NaN const-eval guard: `bitcast<f32>(x | 0xFFFFFFFFu)` MUST NOT fold
    //     to `bitcast<f32>(0xFFFFFFFFu)` (Tint rejects NaN const-eval)!
    out_f[2] = bitcast<f32>(x | 0xffffffffu);
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.identities_simplified >= 20,
        "expected >= 20 algebraic/bitwise/negation/shift identities, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(
        stats.strength_reductions >= 1,
        "expected `x * 4u` strength reduction, got {stats:?}\nWGSL:\n{wgsl}"
    );

    // Same-handle integer identities:
    assert!(wgsl.contains("out_u[0] = 0u;"), "u32 x - x -> 0u:\n{wgsl}");
    assert!(wgsl.contains("out_u[1] = 0u;"), "u32 x ^ x -> 0u:\n{wgsl}");
    assert!(wgsl.contains("out_u[2] = gid.x;") || wgsl.contains("out_u[2] = x;"), "x | x -> x:\n{wgsl}");
    assert!(wgsl.contains("out_u[3] = gid.x;") || wgsl.contains("out_u[3] = x;"), "x & x -> x:\n{wgsl}");
    assert!(wgsl.contains("out_i[0] = 0i;"), "i32 si - si -> 0i:\n{wgsl}");
    assert!(wgsl.contains("out_i[1] = 0i;"), "i32 si ^ si -> 0i:\n{wgsl}");
    assert!(wgsl.contains("out_vu[0] = vec4<u32>();"), "vec4<u32> vx - vx -> ZeroValue:\n{wgsl}");
    assert!(wgsl.contains("out_vu[1] = vec4<u32>();"), "vec4<u32> vx ^ vx -> ZeroValue:\n{wgsl}");
    assert!(wgsl.contains("out_vi[0] = vec4<i32>();"), "vec4<i32> vsi - vsi -> ZeroValue:\n{wgsl}");
    assert!(wgsl.contains("out_vi[1] = vec4<i32>();"), "vec4<i32> vsi ^ vsi -> ZeroValue:\n{wgsl}");

    // Absorbing bitwise-OR with all-ones:
    assert!(wgsl.contains("out_u[4] = 4294967295u;"), "x | 0xFFFFFFFFu -> 0xFFFFFFFFu:\n{wgsl}");
    assert!(wgsl.contains("out_u[5] = 4294967295u;"), "0xFFFFFFFFu | (x+1) -> 0xFFFFFFFFu:\n{wgsl}");
    assert!(wgsl.contains("out_i[2] = -1i;"), "si | -1i -> -1i:\n{wgsl}");
    assert!(wgsl.contains("out_i[3] = -1i;"), "-1i | (si+2) -> -1i:\n{wgsl}");

    // Double negation:
    assert!(!wgsl.contains("!!"), "!!b must cancel:\n{wgsl}");
    assert!(!wgsl.contains("~~"), "~~x must cancel:\n{wgsl}");
    assert!(wgsl.contains("out_i[5] = si;"), "--si must cancel to si:\n{wgsl}");
    assert!(wgsl.contains("out_vi[2] = vsi;"), "--vsi must cancel to vsi:\n{wgsl}");

    // Merged constant shifts:
    assert!(wgsl.contains("<< 7u"), "(x << 3u) << 4u -> x << 7u:\n{wgsl}");
    assert!(wgsl.contains(">> 11u"), "(x >> 5u) >> 6u -> x >> 11u:\n{wgsl}");
    assert!(wgsl.contains("<< 9u"), "(si << 2u) << 7u -> si << 9u:\n{wgsl}");
    assert!(wgsl.contains(">> 12u"), "(si >> 4u) >> 8u -> si >> 12u:\n{wgsl}");
    assert!(wgsl.contains("<< 5u"), "(x * 4u) << 3u -> x << 5u:\n{wgsl}");

    // Negative safety guards:
    assert!(
        wgsl.contains("<< 20u") && wgsl.contains("<< 16u"),
        "shift overflow 20 + 16 >= 32 MUST NOT merge:\n{wgsl}"
    );
    assert!(
        wgsl.contains(">> 16u"),
        "shift overflow 16 + 16 >= 32 MUST NOT merge:\n{wgsl}"
    );
    assert!(
        wgsl.contains(">> 20u") && wgsl.contains(">> 15u"),
        "signed shift overflow 20 + 15 >= 32 MUST NOT merge:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_f[0] = (fx - fx);"),
        "float `fx - fx` MUST NOT fold to 0.0 (NaN/Inf!):\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_f[1] = -(-(fx));"),
        "float `--fx` MUST NOT be touched:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("bitcast<f32>(4294967295u)"),
        "bitcast<f32>(x | 0xFFFFFFFFu) MUST NOT const-eval to NaN:\n{wgsl}"
    );
}

#[test]
fn select_identities_and_impure_arm_or_condition_guards() {
    let src = r#"
@group(0) @binding(0) var<storage, read> in_u: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_u: array<u32>;
@group(0) @binding(2) var<storage, read_write> atom_buf: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> out_v: array<vec4<u32>>;

fn side_effect(idx: u32) -> u32 {
    return atomicAdd(&atom_buf[idx], 1u);
}

fn side_effect_cond(idx: u32) -> bool {
    return atomicAdd(&atom_buf[idx], 1u) != 0u;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = (x == 1u);
    let vc = vec4<bool>(x == 1u, y == 2u, x == 3u, y == 4u);
    let va = vec4<u32>(x, y, 1u, 2u);
    let vb = vec4<u32>(3u, 4u, x, y);

    // 1. `select(a, a, pure_cond)` -> `a`
    out_u[0] = select(x, x, c);

    // 2. `select(reject, accept, !cond)` -> `select(accept, reject, cond)` (scalar & vector)
    // Use a boolean conjunction so `!cond` does not simplify via single-comparison inversion first:
    let b_load = (in_u[x] != 0u);
    out_u[1] = select(x, y, !(b_load && c));
    out_v[0] = select(va, vb, !vc);

    // 3. Constant bool condition with PURE discarded arm -> folds cleanly
    out_u[2] = select(x + 10u, y + 20u, true);
    out_u[3] = select(x + 30u, y + 40u, false);

    // 4. NEGATIVE SAFETY GUARDS:
    // (a) Constant bool condition with IMPURE discarded arm (`side_effect` / `Load`) MUST NOT drop arm!
    let imp_arm = side_effect(x) + 1u;
    out_u[4] = select(imp_arm, y, true);
    let imp_load = in_u[x + 1u];
    out_u[5] = select(x, imp_load, false);

    // (b) `select(a, a, impure_cond)` MUST NOT fold when `condition` is impure!
    let imp_c = side_effect_cond(x);
    out_u[6] = select(x, x, imp_c);
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.identities_simplified >= 3,
        "expected select(a,a,c) + scalar/vector select(!c) identities, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(
        stats.constants_folded >= 2,
        "expected pure const-cond selects to fold, got {stats:?}\nWGSL:\n{wgsl}"
    );

    // Pure selects folded cleanly:
    assert!(wgsl.contains("out_u[0] = gid.x;") || wgsl.contains("out_u[0] = x;"), "select(x, x, c) -> x:\n{wgsl}");
    assert!(wgsl.contains("out_u[2] = (gid.y + 20u);") || wgsl.contains("out_u[2] = (y + 20u);"), "const true select:\n{wgsl}");
    assert!(wgsl.contains("out_u[3] = (gid.x + 30u);") || wgsl.contains("out_u[3] = (x + 30u);"), "const false select:\n{wgsl}");

    // Negated-condition selects inverted cleanly without `!`:
    assert!(
        wgsl.contains("select(y, x,") || wgsl.contains("select(gid.y, gid.x,"),
        "select(x, y, !(b && c)) should swap to select(y, x, b && c):\n{wgsl}"
    );
    assert!(
        wgsl.contains("select(vb, va, vc)"),
        "vector select(va, vb, !vc) should swap to select(vb, va, vc):\n{wgsl}"
    );

    // Impure discarded arms and impure condition MUST remain intact!
    assert!(
        wgsl.contains("side_effect(") && wgsl.contains("side_effect_cond("),
        "impure side effects in select MUST be preserved:\n{wgsl}"
    );
    assert!(
        wgsl.contains("select(imp_arm") || wgsl.contains("true)"),
        "select with impure discarded call arm must NOT fold:\n{wgsl}"
    );
    assert!(
        wgsl.contains("select(x, imp_load, false)") || wgsl.contains("select(gid.x, imp_load, false)"),
        "select with impure discarded Load arm must NOT fold:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_u[6] = select(x, x, ") || wgsl.contains("out_u[6] = select(gid.x, gid.x, "),
        "select(x, x, imp_c) with impure condition must NOT fold:\n{wgsl}"
    );
}

#[test]
fn access_index_and_swizzle_folding_through_splat_compose_array_with_subvector_guard() {
    let src = r#"
@group(0) @binding(0) var<storage, read> in_f: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_f: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_v2: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> out_v4: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = in_f[i];
    let b = in_f[i + 1u];
    let c = in_f[i + 2u];
    let d = in_f[i + 3u];

    // 1. Splat extraction -> scalar
    let sp = vec4<f32>(a);
    out_f[0] = sp.w;

    // 2. 1-to-1 vector Compose extraction -> exact component
    let v = vec4<f32>(a, b, c, d);
    out_f[1] = v.y;

    // 3. 1-to-1 Array Compose extraction -> exact element
    let arr = array<f32, 3>(a, b, c);
    out_f[2] = arr[2];

    // 4. AccessIndex through Swizzle through Compose -> exact scalar `b` (`v.zyx.y == v.y == b`)
    let swiz = v.zyx;
    out_f[3] = swiz.y;

    // 5. Swizzle on Splat -> smaller Splat (`vec4(a).xy` -> `vec2(a)`)
    out_v2[0] = sp.xy;

    // 6. Identity Swizzle (`.xyzw` on `vec4`) -> original vector
    out_v4[0] = v.xyzw;

    // 7. NEGATIVE SAFETY GUARD: Sub-vector packed Compose `vec4(vec2(a, b), vec2(c, d))` has
    //    `components.len() == 2 != 4`! Accessing `.z` MUST NOT mis-index `components[2]`!
    let packed = vec4<f32>(vec2<f32>(a, b), vec2<f32>(c, d));
    out_f[4] = packed.z;
    out_f[5] = packed.w;
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.vector_extracts_folded >= 6,
        "expected >= 6 vector/array/swizzle extract folds, got {stats:?}\nWGSL:\n{wgsl}"
    );

    assert!(wgsl.contains("out_f[0] = a;"), "Splat.w -> a:\n{wgsl}");
    assert!(wgsl.contains("out_f[1] = b;"), "Compose.y -> b:\n{wgsl}");
    assert!(wgsl.contains("out_f[2] = c;"), "Array Compose[2] -> c:\n{wgsl}");
    assert!(wgsl.contains("out_f[3] = b;"), "Compose.zyx.y -> b:\n{wgsl}");
    assert!(wgsl.contains("[0] = vec2(a);"), "Splat(4).xy -> Splat(2):\n{wgsl}");
    assert!(!wgsl.contains(".xyzw"), "identity .xyzw swizzle should eliminate:\n{wgsl}");
    // Sub-vector packed Compose `.z` and `.w` MUST remain intact and never panic or mis-index!
    assert!(
        wgsl.contains("packed.z") && wgsl.contains("packed.w"),
        "sub-vector packed vec4(vec2, vec2).z/.w MUST remain intact:\n{wgsl}"
    );
}

#[test]
fn ir_opt_direct_naga_ir_edge_cases_pass_validator() {
    // Directly build Naga IR for cases WGSL front-end might pre-fold or emit differently:
    // - `vec4<u32>` `x - x` when `vec4<u32>` type is NOT pre-registered in `module.types`
    //   (forces `SplatLiteral` helper splice path!)
    // - `Select` with `Splat(true)` condition and pure discarded arm
    // - `AccessIndex` on `Splat` inside a conditional `If` block
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

    // Note: `vec4<u32>` is intentionally NOT inserted into `module.types`!
    let mut func = Function {
        name: Some("main".to_string()),
        arguments: vec![FunctionArgument {
            name: Some("gid".to_string()),
            ty: u32_ty,
            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::LocalInvocationIndex)),
        }],
        ..Function::default()
    };

    let gid = func
        .expressions
        .append(Expression::FunctionArgument(0), Span::UNDEFINED);
    let u8_lit = func
        .expressions
        .append(Expression::Literal(Literal::U32(8)), Span::UNDEFINED);
    let splat_x = func.expressions.append(
        Expression::Splat {
            size: VectorSize::Quad,
            value: gid,
        },
        Span::UNDEFINED,
    );
    let splat_mul = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Multiply,
            left: splat_x,
            right: u8_lit,
        },
        Span::UNDEFINED,
    );
    let sub_self = func.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Subtract,
            left: splat_mul,
            right: splat_mul,
        },
        Span::UNDEFINED,
    );
    let elem0 = func.expressions.append(
        Expression::AccessIndex {
            base: sub_self,
            index: 0,
        },
        Span::UNDEFINED,
    );
    let g_expr = func
        .expressions
        .append(Expression::GlobalVariable(g_out), Span::UNDEFINED);
    let ptr0 = func.expressions.append(
        Expression::AccessIndex {
            base: g_expr,
            index: 0,
        },
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Emit(Range::new_from_bounds(splat_x, elem0)),
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Emit(Range::new_from_bounds(ptr0, ptr0)),
        Span::UNDEFINED,
    );
    func.body.push(
        Statement::Store {
            pointer: ptr0,
            value: elem0,
        },
        Span::UNDEFINED,
    );

    module.entry_points.push(naga::EntryPoint {
        name: "main".to_string(),
        stage: NagaStage::Compute,
        early_depth_test: None,
        workgroup_size: [64, 1, 1],
        workgroup_size_overrides: None,
        function: func,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.strength_reductions >= 1
            && stats.identities_simplified >= 1
            && stats.vector_extracts_folded >= 1,
        "expected strength reduction + vector x-x via SplatLiteral + Splat extract fold, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_buf[0] = 0u;"),
        "full chain should collapse all the way to 0u:\n{wgsl}"
    );
}

#[test]
fn end_to_end_rdna2_to_wgsl_preserves_validation_determinism_and_exposes_new_json_stats() {
    // Build a small RDNA2 pixel shader exercising shifts, bitwise ops, and exports.
    let mut words = Vec::new();
    // s_lshl_b32 s2, s0, 2 ; s_lshl_b32 s3, s2, 3  -> nested shift!
    words.push(enc_sop2(0x0E, 2, 0, 128 + 2));
    words.push(enc_sop2(0x0E, 3, 2, 128 + 3));
    // s_or_b32 s4, s3, -1  -> absorbing OR with -1
    words.push(enc_sop2(0x0A, 4, 3, 193)); // inline const -1 (193)
    // v_sub_nc_u32 v5, v1, v1  -> x - x -> 0
    words.push(enc_vop2(0x26, 5, 256 + 1, 1));
    // v_lshlrev_b32 v6, 2, v0 ; v_lshlrev_b32 v7, 3, v6
    words.push(enc_vop2(0x18, 6, 128 + 2, 0));
    words.push(enc_vop2(0x18, 7, 128 + 3, 6));
    // Export MRT0 from v4/v5/v6/v7
    words.push(enc_vop2(0x25, 4, 4, 7)); // v_add_nc_u32 v4, s4, v7
    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
    words.push(enc_sopp(0x01, 0)); // s_endpgm

    let bytes = words_to_bytes(&words);
    let opts = Rdna2TranslateOptions {
        stage: Rdna2StageOverride::Pixel,
        ..Default::default()
    };
    let out1 = translate_rdna2_to_wgsl(&bytes, &opts).expect("RDNA2 translation must succeed");
    let out2 = translate_rdna2_to_wgsl(&bytes, &opts).expect("second translation must succeed");

    assert!(out1.naga_validated);
    assert_eq!(out1.wgsl, out2.wgsl, "WGSL output must be deterministic");
    assert_eq!(out1.ir_opt, out2.ir_opt, "IrOptStats must be deterministic");

    // Verify new counters appear in sanitized JSON output:
    let json = out1.to_json();
    assert!(
        json.contains("\"strength_reductions\""),
        "JSON must include strength_reductions:\n{json}"
    );
    assert!(
        json.contains("\"vector_extracts_folded\""),
        "JSON must include vector_extracts_folded:\n{json}"
    );

    // Round-trip validate emitted WGSL:
    let reparsed = parse_wgsl(&out1.wgsl);
    let _ = validate_and_emit_wgsl(&reparsed);

    // Sanity check IrOptStats helper methods with the new fields:
    let mut s = IrOptStats::default();
    assert!(s.is_zero());
    s.strength_reductions = 2;
    assert!(!s.is_zero());
    assert_eq!(s.total_changes(), 2);
    s.vector_extracts_folded = 3;
    assert_eq!(s.total_changes(), 5);
}
