//! Differential testing against the host CPU (x86-64 hosts only): every supported ALU, shift,
//! rotate, multiply and divide form, at all four operand widths, is executed natively (inline asm)
//! and as lifted wasm64 code, and results plus all architecturally defined flags are compared.

use iced_x86::{Code, Instruction, Register};

use super::exec::{
    CODE_BASE, Case, RAX, RBX, RCX, RDX, RSI, RSP, STACK_TOP, build, exec_or_skip, run,
};
use super::ir::Lcg;
use super::{Program, encode};

/// Reference implementations that execute the instruction on the host CPU (x86-64 hosts only).
/// Each sets CF to the requested carry-in first and returns the result plus RFLAGS.
mod native {
    use std::arch::asm;

    pub type Bin = fn(u64, u64, bool) -> (u64, u64);
    pub type Un = fn(u64, bool) -> (u64, u64);
    pub type Shift = fn(u64, u8, bool) -> (u64, u64);
    pub type Wide = fn(u64, u64, u64, bool) -> (u64, u64, u64);

    macro_rules! bin {
        ($($name:ident: $ins:literal,)*) => {$(
            pub fn $name(a: u64, b: u64, cf: bool) -> (u64, u64) {
                let mut a = a;
                let flags: u64;
                // SAFETY: register-only arithmetic; pushfq/pop leave the stack balanced.
                unsafe {
                    asm!("bt {c}, 0", $ins, "pushfq", "pop {f}", a = inout(reg) a, b = in(reg) b,
                        c = in(reg) u64::from(cf), f = out(reg) flags);
                }
                (a, flags)
            }
        )*};
    }

    macro_rules! un {
        ($($name:ident: $ins:literal,)*) => {$(
            pub fn $name(a: u64, cf: bool) -> (u64, u64) {
                let mut a = a;
                let flags: u64;
                // SAFETY: as above.
                unsafe {
                    asm!("bt {c}, 0", $ins, "pushfq", "pop {f}", a = inout(reg) a,
                        c = in(reg) u64::from(cf), f = out(reg) flags);
                }
                (a, flags)
            }
        )*};
    }

    macro_rules! shift {
        ($($name:ident: $ins:literal,)*) => {$(
            pub fn $name(a: u64, count: u8, cf: bool) -> (u64, u64) {
                let mut a = a;
                let flags: u64;
                // SAFETY: as above.
                unsafe {
                    asm!("bt {c}, 0", $ins, "pushfq", "pop {f}", a = inout(reg) a, in("cl") count,
                        c = in(reg) u64::from(cf), f = out(reg) flags);
                }
                (a, flags)
            }
        )*};
    }

    macro_rules! wide {
        ($($name:ident: $ins:literal,)*) => {$(
            pub fn $name(rax: u64, rdx: u64, src: u64, cf: bool) -> (u64, u64, u64) {
                let (mut rax, mut rdx) = (rax, rdx);
                let flags: u64;
                // SAFETY: as above; callers never pass inputs that raise #DE.
                unsafe {
                    asm!("bt {c}, 0", $ins, "pushfq", "pop {f}", inout("rax") rax, inout("rdx") rdx,
                        s = in(reg) src, c = in(reg) u64::from(cf), f = out(reg) flags);
                }
                (rax, rdx, flags)
            }
        )*};
    }

    bin! {
        add8: "add {a:l}, {b:l}", add16: "add {a:x}, {b:x}", add32: "add {a:e}, {b:e}", add64: "add {a}, {b}",
        adc8: "adc {a:l}, {b:l}", adc16: "adc {a:x}, {b:x}", adc32: "adc {a:e}, {b:e}", adc64: "adc {a}, {b}",
        sub8: "sub {a:l}, {b:l}", sub16: "sub {a:x}, {b:x}", sub32: "sub {a:e}, {b:e}", sub64: "sub {a}, {b}",
        sbb8: "sbb {a:l}, {b:l}", sbb16: "sbb {a:x}, {b:x}", sbb32: "sbb {a:e}, {b:e}", sbb64: "sbb {a}, {b}",
        cmp8: "cmp {a:l}, {b:l}", cmp16: "cmp {a:x}, {b:x}", cmp32: "cmp {a:e}, {b:e}", cmp64: "cmp {a}, {b}",
        and8: "and {a:l}, {b:l}", and16: "and {a:x}, {b:x}", and32: "and {a:e}, {b:e}", and64: "and {a}, {b}",
        or8: "or {a:l}, {b:l}", or16: "or {a:x}, {b:x}", or32: "or {a:e}, {b:e}", or64: "or {a}, {b}",
        xor8: "xor {a:l}, {b:l}", xor16: "xor {a:x}, {b:x}", xor32: "xor {a:e}, {b:e}", xor64: "xor {a}, {b}",
        test8: "test {a:l}, {b:l}", test16: "test {a:x}, {b:x}", test32: "test {a:e}, {b:e}", test64: "test {a}, {b}",
        imul16: "imul {a:x}, {b:x}", imul32: "imul {a:e}, {b:e}", imul64: "imul {a}, {b}",
    }

    un! {
        inc8: "inc {a:l}", inc16: "inc {a:x}", inc32: "inc {a:e}", inc64: "inc {a}",
        dec8: "dec {a:l}", dec16: "dec {a:x}", dec32: "dec {a:e}", dec64: "dec {a}",
        neg8: "neg {a:l}", neg16: "neg {a:x}", neg32: "neg {a:e}", neg64: "neg {a}",
        not8: "not {a:l}", not16: "not {a:x}", not32: "not {a:e}", not64: "not {a}",
    }

    shift! {
        shl8: "shl {a:l}, cl", shl16: "shl {a:x}, cl", shl32: "shl {a:e}, cl", shl64: "shl {a}, cl",
        shr8: "shr {a:l}, cl", shr16: "shr {a:x}, cl", shr32: "shr {a:e}, cl", shr64: "shr {a}, cl",
        sar8: "sar {a:l}, cl", sar16: "sar {a:x}, cl", sar32: "sar {a:e}, cl", sar64: "sar {a}, cl",
        rol8: "rol {a:l}, cl", rol16: "rol {a:x}, cl", rol32: "rol {a:e}, cl", rol64: "rol {a}, cl",
        ror8: "ror {a:l}, cl", ror16: "ror {a:x}, cl", ror32: "ror {a:e}, cl", ror64: "ror {a}, cl",
    }

    wide! {
        mul8: "mul {s:l}", mul16: "mul {s:x}", mul32: "mul {s:e}", mul64: "mul {s}",
        imul8w: "imul {s:l}", imul16w: "imul {s:x}", imul32w: "imul {s:e}", imul64w: "imul {s}",
        div8: "div {s:l}", div16: "div {s:x}", div32: "div {s:e}", div64: "div {s}",
        idiv8: "idiv {s:l}", idiv16: "idiv {s:x}", idiv32: "idiv {s:e}", idiv64: "idiv {s}",
    }
}

/// Which flags are architecturally defined after an instruction (and therefore compared).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Defined {
    /// add/adc/sub/sbb/cmp/neg/inc/dec: all six.
    All,
    /// and/or/xor/test: AF undefined.
    Logic,
    /// not: no flag changes (only the carry-in is known on both sides).
    CarryOnly,
    Shl,
    Shr,
    Sar,
    Rotate,
    /// imul/mul: CF and OF.
    Overflow,
    /// div/idiv: all undefined.
    Nothing,
}

impl Defined {
    /// Mask in `[CF, PF, AF, ZF, SF, OF]` order.
    fn mask(self, bits: u32, count: u64) -> [bool; 6] {
        let (t, f) = (true, false);
        let masked = count & if bits == 64 { 63 } else { 31 };
        match self {
            Defined::All => [t; 6],
            Defined::Logic => [t, t, f, t, t, t],
            Defined::CarryOnly => [t, f, f, f, f, f],
            Defined::Overflow => [t, f, f, f, f, t],
            Defined::Nothing => [f; 6],
            _ if masked == 0 => [t, f, f, f, f, f],
            Defined::Rotate => [t, f, f, f, f, masked == 1],
            // SHL/SHR leave CF undefined once the count reaches the operand width.
            kind => [
                kind == Defined::Sar || masked < u64::from(bits),
                t,
                f,
                t,
                t,
                masked == 1,
            ],
        }
    }
}

/// Whether `div`/`idiv` with these inputs raises #DE (checked before running it natively).
fn divide_faults(bits: u32, signed: bool, rax: u64, rdx: u64, src: u64) -> bool {
    let m = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let (dividend, divisor) = if bits == 8 {
        (u128::from(rax & 0xFFFF), u128::from(src & 0xFF))
    } else {
        (
            (u128::from(rdx & m) << bits) | u128::from(rax & m),
            u128::from(src & m),
        )
    };
    if divisor == 0 {
        return true;
    }
    if !signed {
        return dividend / divisor > u128::from(m);
    }
    let sext = |v: u128, width: u32| ((v << (128 - width)) as i128) >> (128 - width);
    let q = sext(dividend, 2 * bits) / sext(divisor, bits);
    let limit = 1i128 << (bits - 1);
    q < -limit || q >= limit
}

#[test]
fn integer_semantics_match_the_host_cpu() {
    use native::*;

    if !exec_or_skip("integer_semantics_match_the_host_cpu") {
        return;
    }

    #[derive(Clone, Copy)]
    enum Native {
        Bin(Bin),
        Un(Un),
        Shift(Shift),
        Wide(Wide),
    }
    struct Op {
        name: String,
        bits: u32,
        va: u64,
        native: Native,
        defined: Defined,
        signed_div: Option<bool>,
    }

    const WIDTHS: [u32; 4] = [8, 16, 32, 64];
    let dst = [Register::BL, Register::BX, Register::EBX, Register::RBX];
    let src = [Register::SIL, Register::SI, Register::ESI, Register::RSI];
    let mut ops: Vec<Op> = Vec::new();
    let mut pieces = Vec::new();
    let mut add = |name: &str, w: usize, instr: Instruction, native, defined, signed_div| {
        let va = CODE_BASE + ops.len() as u64 * 0x20;
        let bytes = encode(va, &[instr, Instruction::with(Code::Retnq)]);
        pieces.push((va, bytes));
        ops.push(Op {
            name: format!("{name}{}", WIDTHS[w]),
            bits: WIDTHS[w],
            va,
            native,
            defined,
            signed_div,
        });
    };

    use Code as C;
    let bins: [(&str, [Code; 4], [Bin; 4], Defined); 9] = [
        (
            "add",
            [
                C::Add_rm8_r8,
                C::Add_rm16_r16,
                C::Add_rm32_r32,
                C::Add_rm64_r64,
            ],
            [add8, add16, add32, add64],
            Defined::All,
        ),
        (
            "adc",
            [
                C::Adc_rm8_r8,
                C::Adc_rm16_r16,
                C::Adc_rm32_r32,
                C::Adc_rm64_r64,
            ],
            [adc8, adc16, adc32, adc64],
            Defined::All,
        ),
        (
            "sub",
            [
                C::Sub_rm8_r8,
                C::Sub_rm16_r16,
                C::Sub_rm32_r32,
                C::Sub_rm64_r64,
            ],
            [sub8, sub16, sub32, sub64],
            Defined::All,
        ),
        (
            "sbb",
            [
                C::Sbb_rm8_r8,
                C::Sbb_rm16_r16,
                C::Sbb_rm32_r32,
                C::Sbb_rm64_r64,
            ],
            [sbb8, sbb16, sbb32, sbb64],
            Defined::All,
        ),
        (
            "cmp",
            [
                C::Cmp_rm8_r8,
                C::Cmp_rm16_r16,
                C::Cmp_rm32_r32,
                C::Cmp_rm64_r64,
            ],
            [cmp8, cmp16, cmp32, cmp64],
            Defined::All,
        ),
        (
            "and",
            [
                C::And_rm8_r8,
                C::And_rm16_r16,
                C::And_rm32_r32,
                C::And_rm64_r64,
            ],
            [and8, and16, and32, and64],
            Defined::Logic,
        ),
        (
            "or",
            [C::Or_rm8_r8, C::Or_rm16_r16, C::Or_rm32_r32, C::Or_rm64_r64],
            [or8, or16, or32, or64],
            Defined::Logic,
        ),
        (
            "xor",
            [
                C::Xor_rm8_r8,
                C::Xor_rm16_r16,
                C::Xor_rm32_r32,
                C::Xor_rm64_r64,
            ],
            [xor8, xor16, xor32, xor64],
            Defined::Logic,
        ),
        (
            "test",
            [
                C::Test_rm8_r8,
                C::Test_rm16_r16,
                C::Test_rm32_r32,
                C::Test_rm64_r64,
            ],
            [test8, test16, test32, test64],
            Defined::Logic,
        ),
    ];
    for (name, codes, natives, defined) in bins {
        for w in 0..4 {
            let instr = Instruction::with2(codes[w], dst[w], src[w]).expect("binop");
            add(name, w, instr, Native::Bin(natives[w]), defined, None);
        }
    }
    let imuls: [(Code, Bin); 3] = [
        (C::Imul_r16_rm16, imul16),
        (C::Imul_r32_rm32, imul32),
        (C::Imul_r64_rm64, imul64),
    ];
    for (i, (code, native)) in imuls.into_iter().enumerate() {
        let instr = Instruction::with2(code, dst[i + 1], src[i + 1]).expect("imul");
        add(
            "imul2_",
            i + 1,
            instr,
            Native::Bin(native),
            Defined::Overflow,
            None,
        );
    }
    let uns: [(&str, [Code; 4], [Un; 4], Defined); 4] = [
        (
            "inc",
            [C::Inc_rm8, C::Inc_rm16, C::Inc_rm32, C::Inc_rm64],
            [inc8, inc16, inc32, inc64],
            Defined::All,
        ),
        (
            "dec",
            [C::Dec_rm8, C::Dec_rm16, C::Dec_rm32, C::Dec_rm64],
            [dec8, dec16, dec32, dec64],
            Defined::All,
        ),
        (
            "neg",
            [C::Neg_rm8, C::Neg_rm16, C::Neg_rm32, C::Neg_rm64],
            [neg8, neg16, neg32, neg64],
            Defined::All,
        ),
        (
            "not",
            [C::Not_rm8, C::Not_rm16, C::Not_rm32, C::Not_rm64],
            [not8, not16, not32, not64],
            Defined::CarryOnly,
        ),
    ];
    for (name, codes, natives, defined) in uns {
        for w in 0..4 {
            let instr = Instruction::with1(codes[w], dst[w]).expect("unop");
            add(name, w, instr, Native::Un(natives[w]), defined, None);
        }
    }
    let shifts: [(&str, [Code; 4], [Shift; 4], Defined); 5] = [
        (
            "shl",
            [
                C::Shl_rm8_CL,
                C::Shl_rm16_CL,
                C::Shl_rm32_CL,
                C::Shl_rm64_CL,
            ],
            [shl8, shl16, shl32, shl64],
            Defined::Shl,
        ),
        (
            "shr",
            [
                C::Shr_rm8_CL,
                C::Shr_rm16_CL,
                C::Shr_rm32_CL,
                C::Shr_rm64_CL,
            ],
            [shr8, shr16, shr32, shr64],
            Defined::Shr,
        ),
        (
            "sar",
            [
                C::Sar_rm8_CL,
                C::Sar_rm16_CL,
                C::Sar_rm32_CL,
                C::Sar_rm64_CL,
            ],
            [sar8, sar16, sar32, sar64],
            Defined::Sar,
        ),
        (
            "rol",
            [
                C::Rol_rm8_CL,
                C::Rol_rm16_CL,
                C::Rol_rm32_CL,
                C::Rol_rm64_CL,
            ],
            [rol8, rol16, rol32, rol64],
            Defined::Rotate,
        ),
        (
            "ror",
            [
                C::Ror_rm8_CL,
                C::Ror_rm16_CL,
                C::Ror_rm32_CL,
                C::Ror_rm64_CL,
            ],
            [ror8, ror16, ror32, ror64],
            Defined::Rotate,
        ),
    ];
    for (name, codes, natives, defined) in shifts {
        for w in 0..4 {
            let instr = Instruction::with2(codes[w], dst[w], Register::CL).expect("shift");
            add(name, w, instr, Native::Shift(natives[w]), defined, None);
        }
    }
    /// name, encodings, host implementations, defined flags, Some(signed) for divisions
    type WideRow = (&'static str, [Code; 4], [Wide; 4], Defined, Option<bool>);
    let wides: [WideRow; 4] = [
        (
            "mul",
            [C::Mul_rm8, C::Mul_rm16, C::Mul_rm32, C::Mul_rm64],
            [mul8, mul16, mul32, mul64],
            Defined::Overflow,
            None,
        ),
        (
            "imul1_",
            [C::Imul_rm8, C::Imul_rm16, C::Imul_rm32, C::Imul_rm64],
            [imul8w, imul16w, imul32w, imul64w],
            Defined::Overflow,
            None,
        ),
        (
            "div",
            [C::Div_rm8, C::Div_rm16, C::Div_rm32, C::Div_rm64],
            [div8, div16, div32, div64],
            Defined::Nothing,
            Some(false),
        ),
        (
            "idiv",
            [C::Idiv_rm8, C::Idiv_rm16, C::Idiv_rm32, C::Idiv_rm64],
            [idiv8, idiv16, idiv32, idiv64],
            Defined::Nothing,
            Some(true),
        ),
    ];
    for (name, codes, natives, defined, signed_div) in wides {
        for w in 0..4 {
            let instr = Instruction::with1(codes[w], dst[w]).expect("wide");
            add(
                name,
                w,
                instr,
                Native::Wide(natives[w]),
                defined,
                signed_div,
            );
        }
    }
    let program = pieces
        .iter()
        .fold(Program::new(), |p, (va, bytes)| p.bytes(*va, bytes));
    let functions = program.discover();
    assert_eq!(functions.len(), ops.len());
    let built = build("hostdiff", functions.values());

    // Width boundaries plus pseudo-random values (upper bits exercise sub-register merging).
    let mut rng = Lcg(0xD1FF);
    let mut values: Vec<u64> = vec![
        0,
        1,
        0x7F,
        0x80,
        0xFF,
        0x7FFF,
        0x8000,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFF,
        i64::MAX as u64,
        i64::MIN as u64,
        u64::MAX,
    ];
    values.extend((0..5).map(|_| rng.next()));
    let counts: [u64; 14] = [0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 255];

    // (op index, case, inputs a/b-or-count/rdx, carry-in)
    let mut cases = Vec::new();
    let mut meta = Vec::new();
    let base_case = |va: u64| {
        let mut c = Case::new(va);
        c.regs = [0; 16];
        c.regs[RSP] = STACK_TOP;
        c.out = vec![RAX, RDX, RBX, RSP];
        c
    };
    for (i, op) in ops.iter().enumerate() {
        for cin in [false, true] {
            let mut push = |c: Case, inputs: (u64, u64, u64)| {
                let mut c = c;
                c.flags[0] = cin;
                cases.push(c);
                meta.push((i, inputs, cin));
            };
            match op.native {
                Native::Bin(_) => {
                    for &a in &values {
                        for &b in &values {
                            push(base_case(op.va).reg(RBX, a).reg(RSI, b), (a, b, 0));
                        }
                    }
                }
                Native::Un(_) => {
                    for &a in &values {
                        push(base_case(op.va).reg(RBX, a), (a, 0, 0));
                    }
                }
                Native::Shift(_) => {
                    for &a in &values {
                        for &n in &counts {
                            push(base_case(op.va).reg(RBX, a).reg(RCX, n), (a, n, 0));
                        }
                    }
                }
                Native::Wide(_) => {
                    for &a in &values {
                        for &s in &values {
                            let sign = if (a as i64) < 0 { u64::MAX } else { 0 };
                            for rdx in [0, sign, 1, rng.next()] {
                                push(
                                    base_case(op.va).reg(RAX, a).reg(RBX, s).reg(RDX, rdx),
                                    (a, s, rdx),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    let outcomes = run(&built, &cases);

    const FLAG_BITS: [u32; 6] = [0, 2, 4, 6, 7, 11];
    const FLAG_NAMES: [&str; 6] = ["CF", "PF", "AF", "ZF", "SF", "OF"];
    let mut mismatches = Vec::new();
    for (o, &(i, (x, y, z), cin)) in outcomes.iter().zip(&meta) {
        let op = &ops[i];
        let label = format!(
            "{} a={x:#x} b={y:#x} rdx={z:#x} cf={}",
            op.name,
            u8::from(cin)
        );
        let (expected_regs, flags, count): (Vec<(usize, u64)>, u64, u64) = match op.native {
            Native::Bin(f) => {
                let (r, fl) = f(x, y, cin);
                (vec![(RBX, r)], fl, 0)
            }
            Native::Un(f) => {
                let (r, fl) = f(x, cin);
                (vec![(RBX, r)], fl, 0)
            }
            Native::Shift(f) => {
                let (r, fl) = f(x, y as u8, cin);
                (vec![(RBX, r)], fl, y)
            }
            Native::Wide(f) => {
                if let Some(signed) = op.signed_div
                    && divide_faults(op.bits, signed, x, z, y)
                {
                    if !o.error.starts_with("divide error") {
                        mismatches.push(format!("{label}: expected #DE, got {:?}", o.error));
                    }
                    continue;
                }
                let (rax, rdx, fl) = f(x, z, y, cin);
                (vec![(RAX, rax), (RDX, rdx)], fl, 0)
            }
        };
        if !o.error.is_empty() {
            mismatches.push(format!("{label}: trapped: {}", o.error));
            continue;
        }
        for (reg, value) in expected_regs {
            if o.regs[reg] != value {
                mismatches.push(format!(
                    "{label}: reg {reg} = {:#x}, host {value:#x}",
                    o.regs[reg]
                ));
            }
        }
        let defined = op.defined.mask(op.bits, count);
        for k in 0..6 {
            let host = flags >> FLAG_BITS[k] & 1 == 1;
            if defined[k] && o.flags[k] != host {
                mismatches.push(format!(
                    "{label}: {} = {}, host {}",
                    FLAG_NAMES[k], o.flags[k], host
                ));
            }
        }
        if o.regs[RSP] != STACK_TOP {
            mismatches.push(format!("{label}: rsp not restored"));
        }
    }
    println!(
        "host differential test: {} ops, {} cases",
        ops.len(),
        cases.len()
    );
    assert!(
        mismatches.is_empty(),
        "{} mismatches, first ones:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(25)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
