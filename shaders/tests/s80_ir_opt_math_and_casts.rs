#![forbid(unsafe_code)]
#![allow(unused_mut)]

//! S80 (#2207): Naga IR optimization pass tests for:
//! 1. Bit-exact `Expression::As` redundant identity cast elimination & literal/splat conversion
//!    folding (`i32 <-> u32`, `bool <-> u32/i32/f32`, `u32/i32 -> f32`, finite in-range
//!    `f32 -> u32/i32`) with strict negative guards for NaN/Inf/out-of-range/width-changing casts.
//! 2. Bit-exact `Expression::Math` integer (`u32`/`i32`) literal constant folding (`Abs` including
//!    `i32::MIN`, `Min`, `Max`, `Clamp`, `CountOneBits`, `CountLeadingZeros`, `CountTrailingZeros`,
//!    `ReverseBits`, `FirstLeadingBit`, `FirstTrailingBit`, `ExtractBits`, `InsertBits`) and pure
//!    integer algebraic identities (`min(x,x)`, `max(x,x)`, `abs(abs(x))`, `abs(u32_x)`, `u32`
//!    bounds identities, `clamp(x,c,c)`) with impure-argument negative guards.
//! 3. Bit-exact finite `f32` `Expression::Math` literal constant folding (`Abs`, `Floor`, `Ceil`,
//!    `Trunc`, `Saturate`, `Min`, `Max`, `Clamp`, `Fma`) and IEEE-754-safe idempotent identities
//!    (`abs(abs(x))`, `saturate(saturate(x))`, `floor(floor(x))`, `ceil(ceil(x))`, `trunc(trunc(x))`)
//!    with explicit negative safety guards (`fma(a, 0.0, c)`, `mix(x, x, t)`, non-constant float
//!    `min`/`max`/`clamp`, NaN/Inf/overflow/`-0.0`).
//! 4. End-to-end RDNA2 -> WGSL validation, determinism, and `casts_folded`/`math_ops_folded`
//!    JSON stats surface.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal,
    MathFunction, Module, Range, ResourceBinding, Scalar, ScalarKind, ShaderStage as NagaStage,
    Span, Statement, StorageAccess, Type, TypeInner, VectorSize,
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

/// Helper to build a storage-buffer compute module with typed output arrays (`out_u`, `out_i`,
/// `out_f`, `out_vu`, `out_vi`, `out_vf`).
struct DirectIrHarness {
    module: Module,
    func: std::cell::RefCell<Function>,
    out_u: Handle<Expression>,
    out_i: Handle<Expression>,
    out_f: Handle<Expression>,
    out_vu: Handle<Expression>,
    out_vi: Handle<Expression>,
    out_vf: Handle<Expression>,
    gid_arg: Handle<Expression>,
}

impl DirectIrHarness {
    fn new() -> Self {
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
        let vec3_u32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::UNDEFINED,
        );
        let vec4_u32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::U32,
                },
            },
            Span::UNDEFINED,
        );
        let vec4_i32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::I32,
                },
            },
            Span::UNDEFINED,
        );
        let vec4_f32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                },
            },
            Span::UNDEFINED,
        );

        let mk_arr = |m: &mut Module, elem: Handle<Type>, stride: u32| {
            m.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Array {
                        base: elem,
                        size: naga::ArraySize::Dynamic,
                        stride,
                    },
                },
                Span::UNDEFINED,
            )
        };
        let arr_u = mk_arr(&mut module, u32_ty, 4);
        let arr_i = mk_arr(&mut module, i32_ty, 4);
        let arr_f = mk_arr(&mut module, f32_ty, 4);
        let arr_vu = mk_arr(&mut module, vec4_u32_ty, 16);
        let arr_vi = mk_arr(&mut module, vec4_i32_ty, 16);
        let arr_vf = mk_arr(&mut module, vec4_f32_ty, 16);

        let mk_buf = |m: &mut Module, name: &str, binding: u32, ty: Handle<Type>| {
            m.global_variables.append(
                GlobalVariable {
                    name: Some(name.to_string()),
                    space: AddressSpace::Storage {
                        access: StorageAccess::LOAD | StorageAccess::STORE,
                    },
                    binding: Some(ResourceBinding { group: 0, binding }),
                    ty,
                    init: None,
                    memory_decorations: naga::MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            )
        };

        let g_out_u = mk_buf(&mut module, "out_u", 0, arr_u);
        let g_out_i = mk_buf(&mut module, "out_i", 1, arr_i);
        let g_out_f = mk_buf(&mut module, "out_f", 2, arr_f);
        let g_out_vu = mk_buf(&mut module, "out_vu", 3, arr_vu);
        let g_out_vi = mk_buf(&mut module, "out_vi", 4, arr_vi);
        let g_out_vf = mk_buf(&mut module, "out_vf", 5, arr_vf);

        let mut func = Function {
            name: Some("main".to_string()),
            arguments: vec![FunctionArgument {
                name: Some("gid".to_string()),
                ty: vec3_u32_ty,
                binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
            }],
            ..Default::default()
        };

        let out_u = func
            .expressions
            .append(Expression::GlobalVariable(g_out_u), Span::UNDEFINED);
        let out_i = func
            .expressions
            .append(Expression::GlobalVariable(g_out_i), Span::UNDEFINED);
        let out_f = func
            .expressions
            .append(Expression::GlobalVariable(g_out_f), Span::UNDEFINED);
        let out_vu = func
            .expressions
            .append(Expression::GlobalVariable(g_out_vu), Span::UNDEFINED);
        let out_vi = func
            .expressions
            .append(Expression::GlobalVariable(g_out_vi), Span::UNDEFINED);
        let out_vf = func
            .expressions
            .append(Expression::GlobalVariable(g_out_vf), Span::UNDEFINED);
        let gid_arg = func
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);

        Self {
            module,
            func: std::cell::RefCell::new(func),
            out_u,
            out_i,
            out_f,
            out_vu,
            out_vi,
            out_vf,
            gid_arg,
        }
    }

    fn lit(&self, l: Literal) -> Handle<Expression> {
        self.func
            .borrow_mut()
            .expressions
            .append(Expression::Literal(l), Span::UNDEFINED)
    }

    fn emit_expr(&self, e: Expression) -> Handle<Expression> {
        let mut func = self.func.borrow_mut();
        let h = func.expressions.append(e, Span::UNDEFINED);
        func.body
            .push(Statement::Emit(Range::new_from_bounds(h, h)), Span::UNDEFINED);
        h
    }

    fn store_idx(&self, buf: Handle<Expression>, idx: u32, val: Handle<Expression>) {
        let ptr = self.emit_expr(Expression::AccessIndex {
            base: buf,
            index: idx,
        });
        self.func.borrow_mut().body.push(
            Statement::Store {
                pointer: ptr,
                value: val,
            },
            Span::UNDEFINED,
        );
    }

    fn finish(mut self) -> Module {
        self.module.entry_points.push(naga::EntryPoint {
            name: "main".to_string(),
            stage: NagaStage::Compute,
            early_depth_test: None,
            workgroup_size: [64, 1, 1],
            workgroup_size_overrides: None,
            function: self.func.into_inner(),
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        self.module
    }
}

#[test]
fn as_cast_identity_and_literal_splat_folding_with_boundary_and_safety_guards() {
    let mut h = DirectIrHarness::new();
    let gid_x = h.emit_expr(Expression::AccessIndex {
        base: h.gid_arg,
        index: 0,
    });
    let gid_vec = h.emit_expr(Expression::Splat {
        size: VectorSize::Quad,
        value: gid_x,
    });

    // 1. Redundant identity numeric casts (scalar & vector) -> must redirect to inner
    let id_u = h.emit_expr(Expression::As {
        expr: gid_x,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 0, id_u);

    let id_vu = h.emit_expr(Expression::As {
        expr: gid_vec,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_vu, 0, id_vu);

    let f_val = h.emit_expr(Expression::As {
        expr: gid_x,
        kind: ScalarKind::Float,
        convert: Some(4),
    });
    let id_f = h.emit_expr(Expression::As {
        expr: f_val,
        kind: ScalarKind::Float,
        convert: Some(4),
    });
    h.store_idx(h.out_f, 0, id_f);

    // 2. Literal `As` conversions:
    // i32 <-> u32 exact two's-complement wrap
    let i_neg1 = h.lit(Literal::I32(-1));
    let u_from_neg1 = h.emit_expr(Expression::As {
        expr: i_neg1,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 1, u_from_neg1);

    let i_min = h.lit(Literal::I32(i32::MIN));
    let u_from_imin = h.emit_expr(Expression::As {
        expr: i_min,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 2, u_from_imin);

    let u_max = h.lit(Literal::U32(u32::MAX));
    let i_from_umax = h.emit_expr(Expression::As {
        expr: u_max,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    h.store_idx(h.out_i, 0, i_from_umax);

    // bool -> u32 / i32 / f32
    let b_true = h.lit(Literal::Bool(true));
    let b_false = h.lit(Literal::Bool(false));
    let u_from_true = h.emit_expr(Expression::As {
        expr: b_true,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    let u_from_false = h.emit_expr(Expression::As {
        expr: b_false,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    let i_from_true = h.emit_expr(Expression::As {
        expr: b_true,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    let f_from_true = h.emit_expr(Expression::As {
        expr: b_true,
        kind: ScalarKind::Float,
        convert: Some(4),
    });
    let f_from_false = h.emit_expr(Expression::As {
        expr: b_false,
        kind: ScalarKind::Float,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 3, u_from_true);
    h.store_idx(h.out_u, 4, u_from_false);
    h.store_idx(h.out_i, 1, i_from_true);
    h.store_idx(h.out_f, 1, f_from_true);
    h.store_idx(h.out_f, 2, f_from_false);

    // u32 / i32 -> bool -> u32 (chained fold across passes!)
    let u_42 = h.lit(Literal::U32(42));
    let b_from_u42 = h.emit_expr(Expression::As {
        expr: u_42,
        kind: ScalarKind::Bool,
        convert: Some(Scalar::BOOL.width),
    });
    let u_from_b42 = h.emit_expr(Expression::As {
        expr: b_from_u42,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    let i_0 = h.lit(Literal::I32(0));
    let b_from_i0 = h.emit_expr(Expression::As {
        expr: i_0,
        kind: ScalarKind::Bool,
        convert: Some(Scalar::BOOL.width),
    });
    let u_from_bi0 = h.emit_expr(Expression::As {
        expr: b_from_i0,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 5, u_from_b42);
    h.store_idx(h.out_u, 6, u_from_bi0);

    // u32 / i32 -> f32 (exact IEEE-754 round-to-nearest-even)
    let u_7 = h.lit(Literal::U32(7));
    let f_from_u7 = h.emit_expr(Expression::As {
        expr: u_7,
        kind: ScalarKind::Float,
        convert: Some(4),
    });
    let i_neg13 = h.lit(Literal::I32(-13));
    let f_from_ineg13 = h.emit_expr(Expression::As {
        expr: i_neg13,
        kind: ScalarKind::Float,
        convert: Some(4),
    });
    h.store_idx(h.out_f, 3, f_from_u7);
    h.store_idx(h.out_f, 4, f_from_ineg13);

    // Finite in-range f32 -> u32 / i32 (including exact representable boundaries!)
    let f_3_75 = h.lit(Literal::F32(3.75));
    let u_from_f375 = h.emit_expr(Expression::As {
        expr: f_3_75,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    let f_u32_max_safe = h.lit(Literal::F32(4294967040.0));
    let u_from_max_safe = h.emit_expr(Expression::As {
        expr: f_u32_max_safe,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    let f_neg9_9 = h.lit(Literal::F32(-9.9));
    let i_from_neg99 = h.emit_expr(Expression::As {
        expr: f_neg9_9,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    let f_i32_min_safe = h.lit(Literal::F32(-2147483648.0));
    let i_from_imin_safe = h.emit_expr(Expression::As {
        expr: f_i32_min_safe,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    let f_i32_max_safe = h.lit(Literal::F32(2147483520.0));
    let i_from_imax_safe = h.emit_expr(Expression::As {
        expr: f_i32_max_safe,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 7, u_from_f375);
    h.store_idx(h.out_u, 8, u_from_max_safe);
    h.store_idx(h.out_i, 2, i_from_neg99);
    h.store_idx(h.out_i, 3, i_from_imin_safe);
    h.store_idx(h.out_i, 4, i_from_imax_safe);

    // 3. Constant `Splat(Literal)` `As` conversion
    let splat_i5 = h.emit_expr(Expression::Splat {
        size: VectorSize::Quad,
        value: h.lit(Literal::I32(-2)),
    });
    let splat_f = h.emit_expr(Expression::As {
        expr: splat_i5,
        kind: ScalarKind::Float,
        convert: Some(4),
    });
    h.store_idx(h.out_vf, 0, splat_f);

    let splat_u9 = h.emit_expr(Expression::Splat {
        size: VectorSize::Quad,
        value: h.lit(Literal::U32(9)),
    });
    let splat_i9 = h.emit_expr(Expression::As {
        expr: splat_u9,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    h.store_idx(h.out_vi, 0, splat_i9);

    let mut module = h.finish();
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.casts_folded >= 20,
        "expected >= 20 casts folded, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(wgsl.contains("out_u[0] = gid.x;"), "identity u32(gid.x):\n{wgsl}");
    assert!(wgsl.contains("out_u[1] = 4294967295u;"), "u32(-1i):\n{wgsl}");
    assert!(wgsl.contains("out_u[2] = 2147483648u;"), "u32(i32::MIN):\n{wgsl}");
    assert!(wgsl.contains("out_i[0] = -1i;"), "i32(u32::MAX):\n{wgsl}");
    assert!(wgsl.contains("out_u[3] = 1u;"), "u32(true):\n{wgsl}");
    assert!(wgsl.contains("out_u[4] = 0u;"), "u32(false):\n{wgsl}");
    assert!(wgsl.contains("out_i[1] = 1i;"), "i32(true):\n{wgsl}");
    assert!(wgsl.contains("out_f[1] = 1f;"), "f32(true):\n{wgsl}");
    assert!(wgsl.contains("out_f[2] = 0f;"), "f32(false):\n{wgsl}");
    assert!(wgsl.contains("out_u[5] = 1u;"), "u32(bool(42u)):\n{wgsl}");
    assert!(wgsl.contains("out_u[6] = 0u;"), "u32(bool(0i)):\n{wgsl}");
    assert!(wgsl.contains("out_f[3] = 7f;"), "f32(7u):\n{wgsl}");
    assert!(wgsl.contains("out_f[4] = -13f;"), "f32(-13i):\n{wgsl}");
    assert!(wgsl.contains("out_u[7] = 3u;"), "u32(3.75f):\n{wgsl}");
    assert!(wgsl.contains("out_u[8] = 4294967040u;"), "u32(4294967040.0f):\n{wgsl}");
    assert!(wgsl.contains("out_i[2] = -9i;"), "i32(-9.9f):\n{wgsl}");
    assert!(
        wgsl.contains("out_i[3] = -2147483648i;")
            || wgsl.contains("out_i[3] = (-2147483647i - 1i);")
            || wgsl.contains("out_i[3] = i32(-2147483648);"),
        "i32(-2147483648.0f):\n{wgsl}"
    );
    assert!(wgsl.contains("out_i[4] = 2147483520i;"), "i32(2147483520.0f):\n{wgsl}");
    assert!(wgsl.contains("out_vf[0] = vec4(-2f);"), "vec4<f32>(vec4<i32>(-2)):\n{wgsl}");
    assert!(wgsl.contains("out_vi[0] = vec4(9i);"), "vec4<i32>(vec4<u32>(9u)):\n{wgsl}");
    // Identity `f32(f32(gid.x))` must collapse to a single `f32(gid.x)`.
    assert!(
        !wgsl.contains("f32(f32("),
        "redundant nested f32(f32(...)) must collapse:\n{wgsl}"
    );
}

#[test]
fn as_cast_negative_guards_never_fold_nan_inf_or_out_of_range_float_casts() {
    let mut h = DirectIrHarness::new();

    // Construct out-of-range / NaN / Inf f32 literals directly in Naga IR (never round-tripped
    // through WGSL const-eval, validating directly via Naga's IR validator).
    let f_neg = h.lit(Literal::F32(-0.5));
    let u_from_neg = h.emit_expr(Expression::As {
        expr: f_neg,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 0, u_from_neg);

    let f_u32_ovf = h.lit(Literal::F32(4294967296.0)); // 2^32 (out of u32 range)
    let u_from_ovf = h.emit_expr(Expression::As {
        expr: f_u32_ovf,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 1, u_from_ovf);

    let f_i32_pos_ovf = h.lit(Literal::F32(2147483648.0)); // 2^31 (out of i32 range)
    let i_from_pos_ovf = h.emit_expr(Expression::As {
        expr: f_i32_pos_ovf,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    h.store_idx(h.out_i, 0, i_from_pos_ovf);

    let f_i32_neg_ovf = h.lit(Literal::F32(-2147483904.0)); // < i32::MIN
    let i_from_neg_ovf = h.emit_expr(Expression::As {
        expr: f_i32_neg_ovf,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    h.store_idx(h.out_i, 1, i_from_neg_ovf);

    let f_nan = h.lit(Literal::F32(f32::NAN));
    let u_from_nan = h.emit_expr(Expression::As {
        expr: f_nan,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 2, u_from_nan);

    let f_inf = h.lit(Literal::F32(f32::INFINITY));
    let i_from_inf = h.emit_expr(Expression::As {
        expr: f_inf,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    h.store_idx(h.out_i, 2, i_from_inf);

    let f_neg_inf = h.lit(Literal::F32(f32::NEG_INFINITY));
    let i_from_neg_inf = h.emit_expr(Expression::As {
        expr: f_neg_inf,
        kind: ScalarKind::Sint,
        convert: Some(4),
    });
    h.store_idx(h.out_i, 3, i_from_neg_inf);

    // Chained safety guard: `u32(f32(-1i))` must NOT fold `f32(-1i)` into bare `-1.0f` feeding
    // `u32(...)` because Tint/WGSL rejects `u32(-1.0)` at const-eval!
    let i_neg1 = h.lit(Literal::I32(-1));
    let f_neg1 = h.emit_expr(Expression::As {
        expr: i_neg1,
        kind: ScalarKind::Float,
        convert: Some(4),
    });
    let u_from_fneg1 = h.emit_expr(Expression::As {
        expr: f_neg1,
        kind: ScalarKind::Uint,
        convert: Some(4),
    });
    h.store_idx(h.out_u, 3, u_from_fneg1);

    let mut module = h.finish();
    let stats = optimize_module(&mut module);

    assert_eq!(
        stats.casts_folded, 0,
        "NaN, ±Inf, out-of-range float casts, and const-eval-guarded casts MUST NOT fold, got {stats:?}"
    );
    // All 8 `As` expressions (plus inner `f_neg1`) must survive intact in the IR.
    let ep_fn = &module.entry_points[0].function;
    let surviving_as = ep_fn
        .expressions
        .iter()
        .filter(|(_, e)| matches!(e, Expression::As { convert: Some(4), .. }))
        .count();
    assert_eq!(
        surviving_as, 9,
        "all 9 guarded Expression::As nodes must remain unfolded"
    );
}

#[test]
fn integer_math_literal_constant_folding_exact_wgsl_semantics_and_guards() {
    let mut h = DirectIrHarness::new();

    // 1. `Abs`: `u32` identity, `i32` positive/negative, and `i32::MIN` exact wrapping!
    let abs_u = h.emit_expr(Expression::Math {
        fun: MathFunction::Abs,
        arg: h.lit(Literal::U32(99)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let abs_i_neg = h.emit_expr(Expression::Math {
        fun: MathFunction::Abs,
        arg: h.lit(Literal::I32(-42)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let abs_i_min = h.emit_expr(Expression::Math {
        fun: MathFunction::Abs,
        arg: h.lit(Literal::I32(i32::MIN)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    h.store_idx(h.out_u, 0, abs_u);
    h.store_idx(h.out_i, 0, abs_i_neg);
    h.store_idx(h.out_i, 1, abs_i_min);

    // 2. `Min`, `Max`, `Clamp` (valid `lo <= hi`)
    let min_u = h.emit_expr(Expression::Math {
        fun: MathFunction::Min,
        arg: h.lit(Literal::U32(10)),
        arg1: Some(h.lit(Literal::U32(3))),
        arg2: None,
        arg3: None,
    });
    let max_i = h.emit_expr(Expression::Math {
        fun: MathFunction::Max,
        arg: h.lit(Literal::I32(-10)),
        arg1: Some(h.lit(Literal::I32(-3))),
        arg2: None,
        arg3: None,
    });
    let clamp_u = h.emit_expr(Expression::Math {
        fun: MathFunction::Clamp,
        arg: h.lit(Literal::U32(100)),
        arg1: Some(h.lit(Literal::U32(5))),
        arg2: Some(h.lit(Literal::U32(20))),
        arg3: None,
    });
    let clamp_i = h.emit_expr(Expression::Math {
        fun: MathFunction::Clamp,
        arg: h.lit(Literal::I32(-50)),
        arg1: Some(h.lit(Literal::I32(-12))),
        arg2: Some(h.lit(Literal::I32(12))),
        arg3: None,
    });
    h.store_idx(h.out_u, 1, min_u);
    h.store_idx(h.out_i, 2, max_i);
    h.store_idx(h.out_u, 2, clamp_u);
    h.store_idx(h.out_i, 3, clamp_i);

    // 3. Bit-counting & bit-reversal ops: `CountOneBits`, `CountLeadingZeros`,
    //    `CountTrailingZeros`, `ReverseBits`
    let popcnt = h.emit_expr(Expression::Math {
        fun: MathFunction::CountOneBits,
        arg: h.lit(Literal::U32(0xF0F0_0A05)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let clz_u = h.emit_expr(Expression::Math {
        fun: MathFunction::CountLeadingZeros,
        arg: h.lit(Literal::U32(0x0010_0000)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let clz_i_neg1 = h.emit_expr(Expression::Math {
        fun: MathFunction::CountLeadingZeros,
        arg: h.lit(Literal::I32(-1)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let ctz_u = h.emit_expr(Expression::Math {
        fun: MathFunction::CountTrailingZeros,
        arg: h.lit(Literal::U32(0x0000_0040)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let rev_u = h.emit_expr(Expression::Math {
        fun: MathFunction::ReverseBits,
        arg: h.lit(Literal::U32(0x8000_0001)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    h.store_idx(h.out_u, 3, popcnt);
    h.store_idx(h.out_u, 4, clz_u);
    h.store_idx(h.out_i, 4, clz_i_neg1);
    h.store_idx(h.out_u, 5, ctz_u);
    h.store_idx(h.out_u, 6, rev_u);

    // 4. `FirstLeadingBit` & `FirstTrailingBit` exact WGSL spec edge cases!
    let flb_u0 = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstLeadingBit,
        arg: h.lit(Literal::U32(0)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let flb_u_msb = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstLeadingBit,
        arg: h.lit(Literal::U32(0x8000_0000)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let flb_i0 = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstLeadingBit,
        arg: h.lit(Literal::I32(0)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let flb_ineg1 = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstLeadingBit,
        arg: h.lit(Literal::I32(-1)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let flb_ineg2 = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstLeadingBit,
        arg: h.lit(Literal::I32(-2)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let flb_imin = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstLeadingBit,
        arg: h.lit(Literal::I32(i32::MIN)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let ftb_u0 = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstTrailingBit,
        arg: h.lit(Literal::U32(0)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let ftb_i0 = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstTrailingBit,
        arg: h.lit(Literal::I32(0)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let ftb_imin = h.emit_expr(Expression::Math {
        fun: MathFunction::FirstTrailingBit,
        arg: h.lit(Literal::I32(i32::MIN)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    h.store_idx(h.out_u, 7, flb_u0);
    h.store_idx(h.out_u, 8, flb_u_msb);
    h.store_idx(h.out_i, 5, flb_i0);
    h.store_idx(h.out_i, 6, flb_ineg1);
    h.store_idx(h.out_i, 7, flb_ineg2);
    h.store_idx(h.out_i, 8, flb_imin);
    h.store_idx(h.out_u, 9, ftb_u0);
    h.store_idx(h.out_i, 9, ftb_i0);
    h.store_idx(h.out_i, 10, ftb_imin);

    // 5. `ExtractBits` & `InsertBits` (unsigned & sign-extending signed!)
    let ext_u = h.emit_expr(Expression::Math {
        fun: MathFunction::ExtractBits,
        arg: h.lit(Literal::U32(0x1234_5678)),
        arg1: Some(h.lit(Literal::U32(8))),
        arg2: Some(h.lit(Literal::U32(8))),
        arg3: None,
    });
    let ext_i = h.emit_expr(Expression::Math {
        fun: MathFunction::ExtractBits,
        arg: h.lit(Literal::I32(0x0000_00F0)),
        arg1: Some(h.lit(Literal::U32(4))),
        arg2: Some(h.lit(Literal::U32(4))),
        arg3: None,
    });
    let ins_u = h.emit_expr(Expression::Math {
        fun: MathFunction::InsertBits,
        arg: h.lit(Literal::U32(0x1234_0078)),
        arg1: Some(h.lit(Literal::U32(0xAB))),
        arg2: Some(h.lit(Literal::U32(8))),
        arg3: Some(h.lit(Literal::U32(8))),
    });
    h.store_idx(h.out_u, 10, ext_u);
    h.store_idx(h.out_i, 11, ext_i);
    h.store_idx(h.out_u, 11, ins_u);

    // 6. NEGATIVE SAFETY GUARDS: invalid `clamp(x, lo > hi)` and `extractBits(offset + count > 32)`
    //    Wait: we validate those in a separate IR-only test below so WGSL re-parse doesn't reject
    //    intentionally invalid const-eval bounds. Here all ops are valid WGSL!

    let mut module = h.finish();
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert_eq!(
        stats.math_ops_folded, 24,
        "expected all 24 integer math literal ops to fold, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(wgsl.contains("out_u[0] = 99u;"), "abs(99u):\n{wgsl}");
    assert!(wgsl.contains("out_i[0] = 42i;"), "abs(-42i):\n{wgsl}");
    assert!(
        wgsl.contains("out_i[1] = -2147483648i;")
            || wgsl.contains("out_i[1] = (-2147483647i - 1i);")
            || wgsl.contains("out_i[1] = i32(-2147483648);"),
        "abs(i32::MIN) MUST wrap to i32::MIN:\n{wgsl}"
    );
    assert!(wgsl.contains("out_u[1] = 3u;"), "min(10u, 3u):\n{wgsl}");
    assert!(wgsl.contains("out_i[2] = -3i;"), "max(-10i, -3i):\n{wgsl}");
    assert!(wgsl.contains("out_u[2] = 20u;"), "clamp(100u, 5u, 20u):\n{wgsl}");
    assert!(wgsl.contains("out_i[3] = -12i;"), "clamp(-50i, -12i, 12i):\n{wgsl}");
    assert!(wgsl.contains("out_u[3] = 12u;"), "countOneBits(0xF0F00A05u) == 12:\n{wgsl}");
    assert!(wgsl.contains("out_u[4] = 11u;"), "countLeadingZeros(0x00100000u) == 11:\n{wgsl}");
    assert!(wgsl.contains("out_i[4] = 0i;"), "countLeadingZeros(-1i) == 0:\n{wgsl}");
    assert!(wgsl.contains("out_u[5] = 6u;"), "countTrailingZeros(0x40u) == 6:\n{wgsl}");
    assert!(
        wgsl.contains("out_u[6] = 2147483649u;"),
        "reverseBits(0x80000001u) == 0x80000001u:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_u[7] = 4294967295u;"),
        "firstLeadingBit(0u) == 0xFFFFFFFFu:\n{wgsl}"
    );
    assert!(wgsl.contains("out_u[8] = 31u;"), "firstLeadingBit(0x80000000u) == 31u:\n{wgsl}");
    assert!(wgsl.contains("out_i[5] = -1i;"), "firstLeadingBit(0i) == -1i:\n{wgsl}");
    assert!(wgsl.contains("out_i[6] = -1i;"), "firstLeadingBit(-1i) == -1i:\n{wgsl}");
    assert!(wgsl.contains("out_i[7] = 0i;"), "firstLeadingBit(-2i) == 0i:\n{wgsl}");
    assert!(wgsl.contains("out_i[8] = 30i;"), "firstLeadingBit(i32::MIN) == 30i:\n{wgsl}");
    assert!(
        wgsl.contains("out_u[9] = 4294967295u;"),
        "firstTrailingBit(0u) == 0xFFFFFFFFu:\n{wgsl}"
    );
    assert!(wgsl.contains("out_i[9] = -1i;"), "firstTrailingBit(0i) == -1i:\n{wgsl}");
    assert!(wgsl.contains("out_i[10] = 31i;"), "firstTrailingBit(i32::MIN) == 31i:\n{wgsl}");
    assert!(wgsl.contains("out_u[10] = 86u;"), "extractBits(0x12345678u, 8, 8) == 0x56:\n{wgsl}");
    assert!(wgsl.contains("out_i[11] = -1i;"), "signed extractBits(0xF0i, 4, 4) == -1i:\n{wgsl}");
    assert!(
        wgsl.contains("out_u[11] = 305441656u;"),
        "insertBits(0x12340078u, 0xAB, 8, 8) == 0x1234AB78:\n{wgsl}"
    );
}

#[test]
fn integer_math_identities_and_impure_argument_safety_guards() {
    let src = r#"
@group(0) @binding(0) var<storage, read> in_u: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_u: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_i: array<i32>;
@group(0) @binding(3) var<storage, read_write> out_vu: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> atom_buf: array<atomic<u32>>;

fn side_effect(idx: u32) -> u32 {
    return atomicAdd(&atom_buf[idx], 1u);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let si = i32(x);
    let vx = vec4<u32>(gid.x, gid.y, gid.z, 3u);

    // Pure integer identities (must fold):
    out_u[0] = min(x, x);
    out_u[1] = max(x, x);
    out_i[0] = min(si, si);
    out_i[1] = max(si, si);
    out_vu[0] = min(vx, vx);
    out_vu[1] = abs(vx);

    // Unsigned abs(u32) -> u32, signed abs(abs(si)) -> abs(si)
    out_u[2] = abs(x);
    out_i[2] = abs(abs(si));

    // Scalar u32 bounds identities (both operand orders)
    out_u[3] = min(x, 0u);
    out_u[4] = min(0u, x + 1u);
    out_u[5] = max(x, 0u);
    out_u[6] = max(0u, x);
    out_u[7] = min(x, 0xffffffffu);
    out_u[8] = min(0xffffffffu, x);
    out_u[9] = max(x, 0xffffffffu);
    out_u[10] = max(0xffffffffu, x + 2u);

    // Equal constant clamp bounds on pure scalar u32 / i32
    out_u[11] = clamp(x, 7u, 7u);
    out_i[3] = clamp(si, -5i, -5i);

    // NEGATIVE SAFETY GUARDS: impure arguments MUST NOT be discarded!
    let imp_call = side_effect(x);
    out_u[12] = min(imp_call, 0u);
    let imp_call2 = side_effect(x + 1u);
    out_u[13] = max(imp_call2, 0xffffffffu);
    let imp_call3 = side_effect(x + 2u);
    out_u[14] = clamp(imp_call3, 9u, 9u);
    let imp_load = in_u[x];
    out_u[15] = min(imp_load, 0u);
    out_u[16] = clamp(imp_load, 4u, 4u);
}
"#;

    let mut module = parse_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module);

    assert!(
        stats.math_ops_folded >= 18,
        "expected >= 18 integer math identities folded, got {stats:?}\nWGSL:\n{wgsl}"
    );

    assert!(wgsl.contains("out_u[0] = gid.x;") || wgsl.contains("out_u[0] = x;"), "min(x,x):\n{wgsl}");
    assert!(wgsl.contains("out_u[1] = gid.x;") || wgsl.contains("out_u[1] = x;"), "max(x,x):\n{wgsl}");
    assert!(wgsl.contains("out_i[0] = si;"), "min(si,si):\n{wgsl}");
    assert!(wgsl.contains("out_i[1] = si;"), "max(si,si):\n{wgsl}");
    assert!(wgsl.contains("out_vu[0] = vx;"), "min(vx,vx):\n{wgsl}");
    assert!(wgsl.contains("out_vu[1] = vx;"), "abs(vec4<u32>):\n{wgsl}");
    assert!(wgsl.contains("out_u[2] = gid.x;") || wgsl.contains("out_u[2] = x;"), "abs(u32):\n{wgsl}");
    assert!(!wgsl.contains("abs(abs("), "abs(abs(si)) must collapse to abs(si):\n{wgsl}");
    assert!(wgsl.contains("out_i[2] = abs(si);"), "abs(abs(si)) -> abs(si):\n{wgsl}");
    assert!(wgsl.contains("out_u[3] = 0u;"), "min(x, 0u) -> 0u:\n{wgsl}");
    assert!(wgsl.contains("out_u[4] = 0u;"), "min(0u, x+1) -> 0u:\n{wgsl}");
    assert!(wgsl.contains("out_u[5] = gid.x;") || wgsl.contains("out_u[5] = x;"), "max(x, 0u):\n{wgsl}");
    assert!(wgsl.contains("out_u[6] = gid.x;") || wgsl.contains("out_u[6] = x;"), "max(0u, x):\n{wgsl}");
    assert!(wgsl.contains("out_u[7] = gid.x;") || wgsl.contains("out_u[7] = x;"), "min(x, MAX):\n{wgsl}");
    assert!(wgsl.contains("out_u[8] = gid.x;") || wgsl.contains("out_u[8] = x;"), "min(MAX, x):\n{wgsl}");
    assert!(wgsl.contains("out_u[9] = 4294967295u;"), "max(x, MAX) -> MAX:\n{wgsl}");
    assert!(wgsl.contains("out_u[10] = 4294967295u;"), "max(MAX, x+2) -> MAX:\n{wgsl}");
    assert!(wgsl.contains("out_u[11] = 7u;"), "clamp(x, 7u, 7u) -> 7u:\n{wgsl}");
    assert!(wgsl.contains("out_i[3] = -5i;"), "clamp(si, -5i, -5i) -> -5i:\n{wgsl}");

    // Negative safety guards: all impure calls and loads MUST remain intact!
    assert_eq!(
        wgsl.matches("= side_effect(").count(),
        3,
        "all 3 impure side_effect calls must survive:\n{wgsl}"
    );
    assert!(wgsl.contains("out_u[12] = min(") && wgsl.contains(", 0u);"), "min(impure, 0u) must stay:\n{wgsl}");
    assert!(
        wgsl.contains("out_u[13] = max(") && wgsl.contains(", 4294967295u);"),
        "max(impure, MAX) must stay:\n{wgsl}"
    );
    assert!(
        wgsl.contains("out_u[14] = clamp(") && wgsl.contains(", 9u, 9u);"),
        "clamp(impure, 9u, 9u) must stay:\n{wgsl}"
    );
    assert!(wgsl.contains("min(imp_load, 0u)"), "min(load, 0u) must stay:\n{wgsl}");
    assert!(wgsl.contains("clamp(imp_load, 4u, 4u)"), "clamp(load, 4u, 4u) must stay:\n{wgsl}");
}

#[test]
fn float_math_literal_folding_idempotent_identities_and_ieee754_negative_guards() {
    // Part A: Direct Naga IR float literal constant folding (so WGSL parser doesn't pre-fold).
    let mut h = DirectIrHarness::new();

    let f_abs = h.emit_expr(Expression::Math {
        fun: MathFunction::Abs,
        arg: h.lit(Literal::F32(-3.5)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let f_abs_neg0 = h.emit_expr(Expression::Math {
        fun: MathFunction::Abs,
        arg: h.lit(Literal::F32(-0.0)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let f_floor = h.emit_expr(Expression::Math {
        fun: MathFunction::Floor,
        arg: h.lit(Literal::F32(-2.25)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let f_ceil = h.emit_expr(Expression::Math {
        fun: MathFunction::Ceil,
        arg: h.lit(Literal::F32(2.25)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let f_trunc = h.emit_expr(Expression::Math {
        fun: MathFunction::Trunc,
        arg: h.lit(Literal::F32(-2.75)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let f_sat_lo = h.emit_expr(Expression::Math {
        fun: MathFunction::Saturate,
        arg: h.lit(Literal::F32(-4.0)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let f_sat_hi = h.emit_expr(Expression::Math {
        fun: MathFunction::Saturate,
        arg: h.lit(Literal::F32(2.5)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let f_sat_mid = h.emit_expr(Expression::Math {
        fun: MathFunction::Saturate,
        arg: h.lit(Literal::F32(0.25)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let f_min = h.emit_expr(Expression::Math {
        fun: MathFunction::Min,
        arg: h.lit(Literal::F32(1.5)),
        arg1: Some(h.lit(Literal::F32(-2.5))),
        arg2: None,
        arg3: None,
    });
    let f_max = h.emit_expr(Expression::Math {
        fun: MathFunction::Max,
        arg: h.lit(Literal::F32(1.5)),
        arg1: Some(h.lit(Literal::F32(4.5))),
        arg2: None,
        arg3: None,
    });
    let f_clamp = h.emit_expr(Expression::Math {
        fun: MathFunction::Clamp,
        arg: h.lit(Literal::F32(5.0)),
        arg1: Some(h.lit(Literal::F32(-1.0))),
        arg2: Some(h.lit(Literal::F32(2.5))),
        arg3: None,
    });
    let f_fma = h.emit_expr(Expression::Math {
        fun: MathFunction::Fma,
        arg: h.lit(Literal::F32(2.0)),
        arg1: Some(h.lit(Literal::F32(3.5))),
        arg2: Some(h.lit(Literal::F32(1.25))),
        arg3: None,
    });

    h.store_idx(h.out_f, 0, f_abs);
    h.store_idx(h.out_f, 1, f_abs_neg0);
    h.store_idx(h.out_f, 2, f_floor);
    h.store_idx(h.out_f, 3, f_ceil);
    h.store_idx(h.out_f, 4, f_trunc);
    h.store_idx(h.out_f, 5, f_sat_lo);
    h.store_idx(h.out_f, 6, f_sat_hi);
    h.store_idx(h.out_f, 7, f_sat_mid);
    h.store_idx(h.out_f, 8, f_min);
    h.store_idx(h.out_f, 9, f_max);
    h.store_idx(h.out_f, 10, f_clamp);
    h.store_idx(h.out_f, 11, f_fma);

    // Uniform `Splat(Literal::F32)` math fold: `abs(vec4(-2.5f)) -> vec4(2.5f)`
    let splat_neg = h.emit_expr(Expression::Splat {
        size: VectorSize::Quad,
        value: h.lit(Literal::F32(-2.5)),
    });
    let splat_abs = h.emit_expr(Expression::Math {
        fun: MathFunction::Abs,
        arg: splat_neg,
        arg1: None,
        arg2: None,
        arg3: None,
    });
    h.store_idx(h.out_vf, 0, splat_abs);

    let mut module_a = h.finish();
    let stats_a = optimize_module(&mut module_a);
    let wgsl_a = validate_and_emit_wgsl(&module_a);

    assert_eq!(
        stats_a.math_ops_folded, 13,
        "expected all 13 finite f32 math literal ops to fold, got {stats_a:?}\nWGSL:\n{wgsl_a}"
    );
    assert!(wgsl_a.contains("out_f[0] = 3.5f;"), "abs(-3.5):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[1] = 0f;"), "abs(-0.0) -> +0.0:\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[2] = -3f;"), "floor(-2.25):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[3] = 3f;"), "ceil(2.25):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[4] = -2f;"), "trunc(-2.75):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[5] = 0f;"), "saturate(-4.0):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[6] = 1f;"), "saturate(2.5):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[7] = 0.25f;"), "saturate(0.25):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[8] = -2.5f;"), "min(1.5, -2.5):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[9] = 4.5f;"), "max(1.5, 4.5):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[10] = 2.5f;"), "clamp(5.0, -1.0, 2.5):\n{wgsl_a}");
    assert!(wgsl_a.contains("out_f[11] = 8.25f;"), "fma(2.0, 3.5, 1.25) == 8.25:\n{wgsl_a}");
    assert!(wgsl_a.contains("out_vf[0] = vec4(2.5f);"), "abs(vec4(-2.5)):\n{wgsl_a}");

    // Part B: WGSL idempotent float identities AND explicit negative IEEE-754 safety guards!
    let src_b = r#"
@group(0) @binding(0) var<storage, read> in_f: array<f32>;
@group(0) @binding(1) var<storage, read> in_vf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out_f: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_vf: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = in_f[i];
    let c = in_f[i + 1u];
    let t = in_f[i + 2u];
    let va = in_vf[i];

    // Idempotent identities (100% bit-exact for ALL IEEE-754 floats, scalar & vector):
    out_f[0] = abs(abs(a));
    out_f[1] = saturate(saturate(a));
    out_f[2] = floor(floor(a));
    out_f[3] = ceil(ceil(a));
    out_f[4] = trunc(trunc(a));
    out_vf[0] = abs(abs(va));
    out_vf[1] = saturate(saturate(va));
    out_vf[2] = floor(floor(va));
    out_vf[3] = ceil(ceil(va));
    out_vf[4] = trunc(trunc(va));

    // EXPLICIT NEGATIVE SAFETY GUARDS (MUST NOT FOLD!):
    // 1. `fma(a, 0.0, c)` is unsafe when `a = Inf/NaN` or `c = -0.0`.
    out_f[5] = fma(a, 0.0, c);
    // 2. `mix(a, a, t)` is unsafe when `t = NaN`.
    out_f[6] = mix(a, a, t);
    // 3. Non-constant float `min(a, a)`, `max(a, a)`, `clamp(a, 0.5, 0.5)` stay conservative!
    out_f[7] = min(a, a);
    out_f[8] = max(a, a);
    out_f[9] = clamp(a, 0.5, 0.5);
}
"#;

    let mut module_b = parse_wgsl(src_b);
    let stats_b = optimize_module(&mut module_b);
    let wgsl_b = validate_and_emit_wgsl(&module_b);

    assert_eq!(
        stats_b.math_ops_folded, 10,
        "expected exactly the 10 idempotent float identities to fold, got {stats_b:?}\nWGSL:\n{wgsl_b}"
    );
    assert!(!wgsl_b.contains("abs(abs("), "abs(abs(f)) must collapse:\n{wgsl_b}");
    assert!(!wgsl_b.contains("saturate(saturate("), "saturate(saturate(f)) must collapse:\n{wgsl_b}");
    assert!(!wgsl_b.contains("floor(floor("), "floor(floor(f)) must collapse:\n{wgsl_b}");
    assert!(!wgsl_b.contains("ceil(ceil("), "ceil(ceil(f)) must collapse:\n{wgsl_b}");
    assert!(!wgsl_b.contains("trunc(trunc("), "trunc(trunc(f)) must collapse:\n{wgsl_b}");

    // Negative safety guards MUST all survive!
    assert!(wgsl_b.contains("fma(a, 0f, c)"), "fma(a, 0.0, c) MUST NOT fold:\n{wgsl_b}");
    assert!(wgsl_b.contains("mix(a, a, t)"), "mix(x, x, t) MUST NOT fold:\n{wgsl_b}");
    assert!(wgsl_b.contains("min(a, a)"), "float min(a, a) MUST NOT fold:\n{wgsl_b}");
    assert!(wgsl_b.contains("max(a, a)"), "float max(a, a) MUST NOT fold:\n{wgsl_b}");
    assert!(wgsl_b.contains("clamp(a, 0.5f, 0.5f)"), "float clamp(a, 0.5, 0.5) MUST NOT fold:\n{wgsl_b}");
}

#[test]
fn math_negative_guards_for_nan_inf_overflow_neg_zero_and_invalid_bounds() {
    let mut h = DirectIrHarness::new();

    // Overflowing FMA (`1e30 * 1e30 + 0.0 == Inf`) must NOT fold
    let fma_ovf = h.emit_expr(Expression::Math {
        fun: MathFunction::Fma,
        arg: h.lit(Literal::F32(1e30)),
        arg1: Some(h.lit(Literal::F32(1e30))),
        arg2: Some(h.lit(Literal::F32(0.0))),
        arg3: None,
    });
    h.store_idx(h.out_f, 0, fma_ovf);

    // `-0.0` producing ops (`ceil(-0.5) == -0.0`, `trunc(-0.5) == -0.0`) must NOT fold
    let ceil_neg0 = h.emit_expr(Expression::Math {
        fun: MathFunction::Ceil,
        arg: h.lit(Literal::F32(-0.5)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let trunc_neg0 = h.emit_expr(Expression::Math {
        fun: MathFunction::Trunc,
        arg: h.lit(Literal::F32(-0.5)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    h.store_idx(h.out_f, 1, ceil_neg0);
    h.store_idx(h.out_f, 2, trunc_neg0);

    // NaN / Inf inputs to float math ops must NOT fold
    let abs_nan = h.emit_expr(Expression::Math {
        fun: MathFunction::Abs,
        arg: h.lit(Literal::F32(f32::NAN)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    let floor_inf = h.emit_expr(Expression::Math {
        fun: MathFunction::Floor,
        arg: h.lit(Literal::F32(f32::INFINITY)),
        arg1: None,
        arg2: None,
        arg3: None,
    });
    h.store_idx(h.out_f, 3, abs_nan);
    h.store_idx(h.out_f, 4, floor_inf);

    // Invalid `clamp` bounds (`lo > hi`) for u32, i32, f32 must NOT fold
    let bad_clamp_u = h.emit_expr(Expression::Math {
        fun: MathFunction::Clamp,
        arg: h.lit(Literal::U32(5)),
        arg1: Some(h.lit(Literal::U32(10))),
        arg2: Some(h.lit(Literal::U32(2))),
        arg3: None,
    });
    let bad_clamp_i = h.emit_expr(Expression::Math {
        fun: MathFunction::Clamp,
        arg: h.lit(Literal::I32(0)),
        arg1: Some(h.lit(Literal::I32(10))),
        arg2: Some(h.lit(Literal::I32(-2))),
        arg3: None,
    });
    let bad_clamp_f = h.emit_expr(Expression::Math {
        fun: MathFunction::Clamp,
        arg: h.lit(Literal::F32(5.0)),
        arg1: Some(h.lit(Literal::F32(10.0))),
        arg2: Some(h.lit(Literal::F32(2.0))),
        arg3: None,
    });
    h.store_idx(h.out_u, 0, bad_clamp_u);
    h.store_idx(h.out_i, 0, bad_clamp_i);
    h.store_idx(h.out_f, 5, bad_clamp_f);

    // Invalid `extractBits` / `insertBits` (`offset + count > 32`) must NOT fold
    let bad_ext = h.emit_expr(Expression::Math {
        fun: MathFunction::ExtractBits,
        arg: h.lit(Literal::U32(0x1234_5678)),
        arg1: Some(h.lit(Literal::U32(24))),
        arg2: Some(h.lit(Literal::U32(16))),
        arg3: None,
    });
    let bad_ins = h.emit_expr(Expression::Math {
        fun: MathFunction::InsertBits,
        arg: h.lit(Literal::U32(0)),
        arg1: Some(h.lit(Literal::U32(1))),
        arg2: Some(h.lit(Literal::U32(20))),
        arg3: Some(h.lit(Literal::U32(20))),
        });
    h.store_idx(h.out_u, 1, bad_ext);
    h.store_idx(h.out_u, 2, bad_ins);

    let mut module = h.finish();
    let stats = optimize_module(&mut module);

    assert_eq!(
        stats.math_ops_folded, 0,
        "overflowing FMA, -0.0 output, NaN/Inf, and invalid clamp/bit-range bounds MUST NOT fold, got {stats:?}"
    );
    let ep_fn = &module.entry_points[0].function;
    let surviving_math = ep_fn
        .expressions
        .iter()
        .filter(|(_, e)| matches!(e, Expression::Math { .. }))
        .count();
    assert_eq!(
        surviving_math, 10,
        "all 10 guarded Expression::Math nodes must remain unfolded"
    );
}

#[test]
fn end_to_end_rdna2_to_wgsl_validation_determinism_and_json_stats() {
    // Build a small RDNA2 pixel shader exercising conversions (v_cvt_f32_u32 / v_cvt_u32_f32),
    // float floor/trunc/min/max, and MRT0 export.
    let mut words = Vec::new();
    // v_cvt_f32_u32 v2, 128+5  (VOP1 op 0x06, inline const 5 -> f32(5u))
    words.push(enc_vop1(0x06, 2, 128 + 5));
    // v_floor_f32 v3, v2       (VOP1 op 0x24 -> floor(5.0))
    words.push(enc_vop1(0x24, 3, 256 + 2));
    // v_trunc_f32 v4, v3       (VOP1 op 0x21 -> trunc(5.0))
    words.push(enc_vop1(0x21, 4, 256 + 3));
    // v_min_u32 v5, 128+0, v0  (VOP2 op 0x10 -> min(0u, v0))
    words.push(enc_vop2(0x10, 5, 128, 0));
    // v_max_u32 v6, 128+0, v1  (VOP2 op 0x11 -> max(0u, v1))
    words.push(enc_vop2(0x11, 6, 128, 1));
    // Export MRT0 from v3/v4/v5/v6
    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [3, 4, 5, 6]));
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

    // Verify new counters are serialized in sanitized JSON output:
    let json = out1.to_json();
    assert!(
        json.contains("\"casts_folded\""),
        "JSON must include casts_folded:\n{json}"
    );
    assert!(
        json.contains("\"math_ops_folded\""),
        "JSON must include math_ops_folded:\n{json}"
    );

    // Round-trip parse & validate emitted WGSL:
    let reparsed = parse_wgsl(&out1.wgsl);
    let _ = validate_and_emit_wgsl(&reparsed);

    // Verify IrOptStats helper methods account for the new counters:
    let mut s = IrOptStats::default();
    assert!(s.is_zero());
    s.casts_folded = 4;
    assert!(!s.is_zero());
    assert_eq!(s.total_changes(), 4);
    s.math_ops_folded = 7;
    assert_eq!(s.total_changes(), 11);
    let mut merged = IrOptStats::default();
    merged.merge_from(s);
    assert_eq!(merged, s);
}
