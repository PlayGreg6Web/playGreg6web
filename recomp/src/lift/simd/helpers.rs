//! Module-level helpers of the SIMD lifter: double-precision FMA in integer arithmetic,
//! carry-less multiplication and the AES S-boxes.

use std::fmt::Write as _;

/// Straight-line IR text builder for the helper definitions.
struct Ir {
    out: String,
    n: u32,
}

impl Ir {
    fn new(header: &str) -> Self {
        Ir {
            out: format!("{header}\n"),
            n: 0,
        }
    }

    fn v(&mut self, rhs: &str) -> String {
        let name = format!("%v{}", self.n);
        self.n += 1;
        let _ = writeln!(self.out, "  {name} = {rhs}");
        name
    }

    fn line(&mut self, text: &str) {
        let _ = writeln!(self.out, "{text}");
    }

    fn sel(&mut self, c: &str, a: &str, b: &str) -> String {
        self.v(&format!("select i1 {c}, i64 {a}, i64 {b}"))
    }

    /// `x << n` for `0 <= n < 64` given as a value that may be anything; amounts are masked.
    fn shl(&mut self, x: &str, n: &str) -> String {
        let m = self.v(&format!("and i64 {n}, 63"));
        self.v(&format!("shl i64 {x}, {m}"))
    }

    fn lshr(&mut self, x: &str, n: &str) -> String {
        let m = self.v(&format!("and i64 {n}, 63"));
        self.v(&format!("lshr i64 {x}, {m}"))
    }

    fn clz64(&mut self, x: &str) -> String {
        self.v(&format!("call i64 @llvm.ctlz.i64(i64 {x}, i1 false)"))
    }

    /// Leading zeros of the 128-bit value `h:l`.
    fn clz128(&mut self, h: &str, l: &str) -> String {
        let zh = self.clz64(h);
        let zl = self.clz64(l);
        let zl = self.v(&format!("add i64 {zl}, 64"));
        let hz = self.v(&format!("icmp eq i64 {h}, 0"));
        self.sel(&hz, &zl, &zh)
    }

    /// `h:l << n` for `0 <= n < 128`.
    fn shl128(&mut self, h: &str, l: &str, n: &str) -> (String, String) {
        let big = self.v(&format!("icmp uge i64 {n}, 64"));
        let n0 = self.v(&format!("icmp eq i64 {n}, 0"));
        let lo_s = self.shl(l, n);
        let hi_s = self.shl(h, n);
        let back = self.v(&format!("sub i64 64, {n}"));
        let carry = self.lshr(l, &back);
        let carry = self.sel(&n0, "0", &carry);
        let hi_small = self.v(&format!("or i64 {hi_s}, {carry}"));
        let hi = self.sel(&big, &lo_s, &hi_small);
        let lo = self.sel(&big, "0", &lo_s);
        (hi, lo)
    }

    /// `h:l >> n` for any `n >= 0` (signed i64 known non-negative), plus whether set bits were
    /// shifted out.
    fn lshr128(&mut self, h: &str, l: &str, n: &str) -> (String, String, String) {
        let huge = self.v(&format!("icmp uge i64 {n}, 128"));
        let big = self.v(&format!("icmp uge i64 {n}, 64"));
        let n0 = self.v(&format!("icmp eq i64 {n}, 0"));
        let one = self.shl("1", n);
        let mask = self.v(&format!("sub i64 {one}, 1"));
        // n < 64
        let lo_a = self.lshr(l, n);
        let back = self.v(&format!("sub i64 64, {n}"));
        let up = self.shl(h, &back);
        let up = self.sel(&n0, "0", &up);
        let lo_small = self.v(&format!("or i64 {lo_a}, {up}"));
        let hi_small = self.lshr(h, n);
        let lost_small = self.v(&format!("and i64 {l}, {mask}"));
        // 64 <= n < 128 (the masked amount is n - 64)
        let lo_big = self.lshr(h, n);
        let hm = self.v(&format!("and i64 {h}, {mask}"));
        let lost_big = self.v(&format!("or i64 {l}, {hm}"));
        let lo = self.sel(&big, &lo_big, &lo_small);
        let hi = self.sel(&big, "0", &hi_small);
        let lost = self.sel(&big, &lost_big, &lost_small);
        let all = self.v(&format!("or i64 {h}, {l}"));
        let lo = self.sel(&huge, "0", &lo);
        let hi = self.sel(&huge, "0", &hi);
        let lost = self.sel(&huge, &all, &lost);
        let lost = self.v(&format!("icmp ne i64 {lost}, 0"));
        (hi, lo, lost)
    }

    fn add128(&mut self, ah: &str, al: &str, bh: &str, bl: &str) -> (String, String) {
        let lo = self.v(&format!("add i64 {al}, {bl}"));
        let c = self.v(&format!("icmp ult i64 {lo}, {al}"));
        let c = self.v(&format!("zext i1 {c} to i64"));
        let hi = self.v(&format!("add i64 {ah}, {bh}"));
        let hi = self.v(&format!("add i64 {hi}, {c}"));
        (hi, lo)
    }

    fn sub128(&mut self, ah: &str, al: &str, bh: &str, bl: &str) -> (String, String) {
        let lo = self.v(&format!("sub i64 {al}, {bl}"));
        let b = self.v(&format!("icmp ult i64 {al}, {bl}"));
        let b = self.v(&format!("zext i1 {b} to i64"));
        let hi = self.v(&format!("sub i64 {ah}, {bh}"));
        let hi = self.v(&format!("sub i64 {hi}, {b}"));
        (hi, lo)
    }

    /// `a < b` for 128-bit unsigned values.
    fn ult128(&mut self, ah: &str, al: &str, bh: &str, bl: &str) -> String {
        let hl = self.v(&format!("icmp ult i64 {ah}, {bh}"));
        let he = self.v(&format!("icmp eq i64 {ah}, {bh}"));
        let ll = self.v(&format!("icmp ult i64 {al}, {bl}"));
        let el = self.v(&format!("and i1 {he}, {ll}"));
        self.v(&format!("or i1 {hl}, {el}"))
    }
}

/// `double __playgta6_fma_f64(double a, double b, double c)`: `a * b + c` with a single
/// round-to-nearest-even rounding, in integer arithmetic (wasm has no FMA instruction). Needs
/// `@llvm.ctlz.i64` and `@__playgta6_umul128`.
pub(super) fn fma_f64_ir() -> String {
    let mut f = Ir::new(
        "define internal double @__playgta6_fma_f64(double %a, double %b, double %c) unnamed_addr #0 {\nentry:",
    );
    let mut parts = Vec::new();
    for x in ["%a", "%b", "%c"] {
        let bits = f.v(&format!("bitcast double {x} to i64"));
        let sign = f.v(&format!("lshr i64 {bits}, 63"));
        let e = f.v(&format!("lshr i64 {bits}, 52"));
        let e = f.v(&format!("and i64 {e}, 2047"));
        let m = f.v(&format!("and i64 {bits}, 4503599627370495"));
        let mag = f.v(&format!("and i64 {bits}, 9223372036854775807"));
        let finite = f.v(&format!("icmp ne i64 {e}, 2047"));
        let zero = f.v(&format!("icmp eq i64 {mag}, 0"));
        let denormal = f.v(&format!("icmp eq i64 {e}, 0"));
        let mi = f.v(&format!("or i64 {m}, 4503599627370496"));
        let m = f.sel(&denormal, &m, &mi);
        let e = f.sel(&denormal, "1", &e);
        parts.push((sign, e, m, finite, zero));
    }
    let (sa, ea, ma, fa, za) = parts[0].clone();
    let (sb, eb, mb, fb, zb) = parts[1].clone();
    let (sc, ec, mc, fc, zc) = parts[2].clone();

    // Exact 106-bit product, normalised so that its top bit is bit 125.
    let p = f.v(&format!(
        "call {{ i64, i64 }} @__playgta6_umul128(i64 {ma}, i64 {mb})"
    ));
    let pl = f.v(&format!("extractvalue {{ i64, i64 }} {p}, 0"));
    let ph = f.v(&format!("extractvalue {{ i64, i64 }} {p}, 1"));
    let sp = f.v(&format!("xor i64 {sa}, {sb}"));
    let lzp = f.clz128(&ph, &pl);
    let shp = f.v(&format!("sub i64 {lzp}, 2"));
    let (ph, pl) = f.shl128(&ph, &pl, &shp);
    let ep = f.v(&format!("add i64 {ea}, {eb}"));
    let ep = f.v(&format!("sub i64 {ep}, 2150"));
    let ep = f.v(&format!("sub i64 {ep}, {shp}"));
    // The addend, also with its top bit at bit 125.
    let lzc = f.clz64(&mc);
    let shc = f.v(&format!("add i64 {lzc}, 62")); // 125 - (63 - lzc)
    let (ch, cl) = f.shl128("0", &mc, &shc);
    let ecx = f.v(&format!("sub i64 {ec}, 1075"));
    let ecx = f.v(&format!("sub i64 {ecx}, {shc}"));
    // Align the operand with the smaller exponent (sticky bit jammed into bit 0).
    let pbig = f.v(&format!("icmp sge i64 {ep}, {ecx}"));
    let bh = f.sel(&pbig, &ph, &ch);
    let bl = f.sel(&pbig, &pl, &cl);
    let be = f.sel(&pbig, &ep, &ecx);
    let bs = f.sel(&pbig, &sp, &sc);
    let xh = f.sel(&pbig, &ch, &ph);
    let xl = f.sel(&pbig, &cl, &pl);
    let xe = f.sel(&pbig, &ecx, &ep);
    let xs = f.sel(&pbig, &sc, &sp);
    let d = f.v(&format!("sub i64 {be}, {xe}"));
    let (xh, xl, lost) = f.lshr128(&xh, &xl, &d);
    let jam = f.v(&format!("zext i1 {lost} to i64"));
    let xl = f.v(&format!("or i64 {xl}, {jam}"));
    // Signed sum.
    let same = f.v(&format!("icmp eq i64 {bs}, {xs}"));
    let (sh, sl) = f.add128(&bh, &bl, &xh, &xl);
    let (dh, dl) = f.sub128(&bh, &bl, &xh, &xl);
    let (nh, nl) = f.sub128(&xh, &xl, &bh, &bl);
    let borrow = f.ult128(&bh, &bl, &xh, &xl);
    let rh = f.sel(&borrow, &nh, &dh);
    let rl = f.sel(&borrow, &nl, &dl);
    let rh = f.sel(&same, &sh, &rh);
    let rl = f.sel(&same, &sl, &rl);
    let rs = f.sel(&borrow, &xs, &bs);
    let rs = f.sel(&same, &bs, &rs);
    let rall = f.v(&format!("or i64 {rh}, {rl}"));
    let rzero = f.v(&format!("icmp eq i64 {rall}, 0"));
    // Round to 53 bits (or to the subnormal grid): rshift = max(k - 52, -1074 - E).
    let lz = f.clz128(&rh, &rl);
    let k = f.v(&format!("sub i64 127, {lz}"));
    let r1 = f.v(&format!("sub i64 {k}, 52"));
    let r2 = f.v(&format!("sub i64 -1074, {be}"));
    let r1_ge = f.v(&format!("icmp sge i64 {r1}, {r2}"));
    let rshift = f.sel(&r1_ge, &r1, &r2);
    let right = f.v(&format!("icmp sgt i64 {rshift}, 0"));
    // rshift > 0: t = R >> (rshift - 1); round bit = t & 1; q = t >> 1.
    let rm1 = f.v(&format!("sub i64 {rshift}, 1"));
    let rm1 = f.sel(&right, &rm1, "0");
    let (th, tl, sticky) = f.lshr128(&rh, &rl, &rm1);
    let round = f.v(&format!("and i64 {tl}, 1"));
    let (_, ql, _) = f.lshr128(&th, &tl, "1");
    let odd = f.v(&format!("and i64 {ql}, 1"));
    let st = f.v(&format!("zext i1 {sticky} to i64"));
    let so = f.v(&format!("or i64 {st}, {odd}"));
    let inc = f.v(&format!("and i64 {round}, {so}"));
    let qr = f.v(&format!("add i64 {ql}, {inc}"));
    // rshift <= 0: exact, R < 2^53 sits in rl.
    let left = f.v(&format!("sub i64 0, {rshift}"));
    let ql_exact = f.shl(&rl, &left);
    let q = f.sel(&right, &qr, &ql_exact);
    let scale = f.v(&format!("add i64 {be}, {rshift}"));
    let ebits = f.v(&format!("add i64 {scale}, 1074"));
    let ebits = f.v(&format!("shl i64 {ebits}, 52"));
    let bits = f.v(&format!("add i64 {ebits}, {q}"));
    let top = f.v(&format!("add i64 {k}, {be}"));
    let over1 = f.v(&format!("icmp sge i64 {top}, 1024"));
    let over2 = f.v(&format!("icmp uge i64 {bits}, 9218868437227405312"));
    let over = f.v(&format!("or i1 {over1}, {over2}"));
    let mag = f.sel(&over, "9218868437227405312", &bits);
    let mag = f.sel(&rzero, "0", &mag);
    let sbit = f.v(&format!("shl i64 {rs}, 63"));
    let sbit = f.sel(&rzero, "0", &sbit);
    let res = f.v(&format!("or i64 {mag}, {sbit}"));
    let general = f.v(&format!("bitcast i64 {res} to double"));
    // Special operands: zero or non-finite factors are exact through the unfused operations; a
    // non-finite addend with finite factors is the result; a zero addend leaves the product.
    let prod = f.v("fmul double %a, %b");
    let simple = f.v(&format!("fadd double {prod}, %c"));
    let r = f.v(&format!("select i1 {zc}, double {prod}, double {general}"));
    let cnf = f.v(&format!("xor i1 {fc}, true"));
    let r = f.v(&format!("select i1 {cnf}, double %c, double {r}"));
    let ab_fin = f.v(&format!("and i1 {fa}, {fb}"));
    let ab_nz = f.v(&format!("or i1 {za}, {zb}"));
    let ab_nz = f.v(&format!("xor i1 {ab_nz}, true"));
    let general_ok = f.v(&format!("and i1 {ab_fin}, {ab_nz}"));
    let r = f.v(&format!(
        "select i1 {general_ok}, double {r}, double {simple}"
    ));
    f.line(&format!("  ret double {r}\n}}"));
    f.out
}

/// `{lo, hi} = __playgta6_clmul(a, b)`: carry-less 64 x 64 -> 128-bit product.
pub(super) const CLMUL_IR: &str = "\
define internal { i64, i64 } @__playgta6_clmul(i64 %a, i64 %b) unnamed_addr #0 {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %lo = phi i64 [ 0, %entry ], [ %lo.next, %loop ]
  %hi = phi i64 [ 0, %entry ], [ %hi.next, %loop ]
  %bit = lshr i64 %b, %i
  %bit1 = and i64 %bit, 1
  %set = icmp ne i64 %bit1, 0
  %sl = shl i64 %a, %i
  %back = sub i64 64, %i
  %back.m = and i64 %back, 63
  %sh0 = lshr i64 %a, %back.m
  %i.zero = icmp eq i64 %i, 0
  %sh = select i1 %i.zero, i64 0, i64 %sh0
  %xl = select i1 %set, i64 %sl, i64 0
  %xh = select i1 %set, i64 %sh, i64 0
  %lo.next = xor i64 %lo, %xl
  %hi.next = xor i64 %hi, %xh
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, 64
  br i1 %done, label %exit, label %loop
exit:
  %r0 = insertvalue { i64, i64 } poison, i64 %lo.next, 0
  %r1 = insertvalue { i64, i64 } %r0, i64 %hi.next, 1
  ret { i64, i64 } %r1
}
";

/// The AES S-box (FIPS-197): the multiplicative inverse in GF(2^8) followed by the affine map.
pub(super) fn aes_sbox() -> [u8; 256] {
    let mul = |mut a: u8, mut b: u8| {
        let mut p = 0u8;
        while b != 0 {
            if b & 1 != 0 {
                p ^= a;
            }
            let hi = a & 0x80;
            a <<= 1;
            if hi != 0 {
                a ^= 0x1B;
            }
            b >>= 1;
        }
        p
    };
    let mut sbox = [0u8; 256];
    for (x, s) in sbox.iter_mut().enumerate() {
        let x = x as u8;
        let inv = if x == 0 {
            0
        } else {
            (1..=255u8).find(|&y| mul(x, y) == 1).unwrap_or(0)
        };
        *s = inv
            ^ inv.rotate_left(1)
            ^ inv.rotate_left(2)
            ^ inv.rotate_left(3)
            ^ inv.rotate_left(4)
            ^ 0x63;
    }
    sbox
}

/// `@__playgta6_aes_sbox` and `@__playgta6_aes_inv_sbox` (`[256 x i8]` constants).
pub(super) fn aes_tables_ir() -> String {
    let sbox = aes_sbox();
    let mut inv = [0u8; 256];
    for (i, &s) in sbox.iter().enumerate() {
        inv[s as usize] = i as u8;
    }
    let text = |t: &[u8; 256]| t.iter().map(|b| format!("\\{b:02X}")).collect::<String>();
    format!(
        "@__playgta6_aes_sbox = internal unnamed_addr constant [256 x i8] c\"{}\"\n\
         @__playgta6_aes_inv_sbox = internal unnamed_addr constant [256 x i8] c\"{}\"\n",
        text(&sbox),
        text(&inv)
    )
}

/// `{ result, flags } __playgta6_pcmpstr(imm8, a, b, len_a, len_b, implicit)`: the SSE4.2 string
/// compare core shared by `pcmp{i,e}str{i,m}` (Intel SDM "Packed Compare Implicit/Explicit Length
/// Strings"). `a` is the first operand, `b` the second (the string). With `implicit` the lengths
/// are the positions of the first zero element (or the element count); otherwise `len_a`/`len_b`
/// are the signed register values, taken as absolute values and clamped to the element count.
/// `result` is IntRes2 (one bit per element), `flags` holds CF (bit 0), ZF (1), SF (2), OF (3).
/// Needs `@llvm.cttz.i32`.
pub(super) const PCMPSTR_IR: &str = "\
define internal { i32, i32 } @__playgta6_pcmpstr(i32 %imm, <2 x i64> %a, <2 x i64> %b, i64 %la0, i64 %lb0, i1 %implicit) unnamed_addr #0 {
entry:
  %A = alloca [16 x i32], align 4
  %B = alloca [16 x i32], align 4
  store [16 x i32] zeroinitializer, ptr %A, align 4
  store [16 x i32] zeroinitializer, ptr %B, align 4
  %w = and i32 %imm, 1
  %isword = icmp ne i32 %w, 0
  %sg = and i32 %imm, 2
  %issigned = icmp ne i32 %sg, 0
  %n = select i1 %isword, i32 8, i32 16
  %n64 = zext i32 %n to i64
  %aggs = lshr i32 %imm, 2
  %agg = and i32 %aggs, 3
  %pols = lshr i32 %imm, 4
  %pol = and i32 %pols, 3
  %acc0 = icmp eq i32 %agg, 3
  %a8 = bitcast <2 x i64> %a to <16 x i8>
  %a16 = bitcast <2 x i64> %a to <8 x i16>
  %b8 = bitcast <2 x i64> %b to <16 x i8>
  %b16 = bitcast <2 x i64> %b to <8 x i16>
  %nbit = shl i32 1, %n
  %za = icmp eq <16 x i8> %a8, zeroinitializer
  %zab = bitcast <16 x i1> %za to i16
  %zab32 = zext i16 %zab to i32
  %zaw = icmp eq <8 x i16> %a16, zeroinitializer
  %zawb = bitcast <8 x i1> %zaw to i8
  %zawb32 = zext i8 %zawb to i32
  %zam = select i1 %isword, i32 %zawb32, i32 %zab32
  %zal = or i32 %zam, %nbit
  %lai = call i32 @llvm.cttz.i32(i32 %zal, i1 true)
  %zb = icmp eq <16 x i8> %b8, zeroinitializer
  %zbb = bitcast <16 x i1> %zb to i16
  %zbb32 = zext i16 %zbb to i32
  %zbw = icmp eq <8 x i16> %b16, zeroinitializer
  %zbwb = bitcast <8 x i1> %zbw to i8
  %zbwb32 = zext i8 %zbwb to i32
  %zbm = select i1 %isword, i32 %zbwb32, i32 %zbb32
  %zbl = or i32 %zbm, %nbit
  %lbi = call i32 @llvm.cttz.i32(i32 %zbl, i1 true)
  %lan = icmp slt i64 %la0, 0
  %laneg = sub i64 0, %la0
  %laabs = select i1 %lan, i64 %laneg, i64 %la0
  %labig = icmp ugt i64 %laabs, %n64
  %lac = select i1 %labig, i64 %n64, i64 %laabs
  %lae = trunc i64 %lac to i32
  %la = select i1 %implicit, i32 %lai, i32 %lae
  %lbn = icmp slt i64 %lb0, 0
  %lbneg = sub i64 0, %lb0
  %lbabs = select i1 %lbn, i64 %lbneg, i64 %lb0
  %lbbig = icmp ugt i64 %lbabs, %n64
  %lbc = select i1 %lbbig, i64 %n64, i64 %lbabs
  %lbe = trunc i64 %lbc to i32
  %lb = select i1 %implicit, i32 %lbi, i32 %lbe
  br label %fill
fill:
  %fi = phi i32 [ 0, %entry ], [ %fi.n, %fill ]
  %fi7 = and i32 %fi, 7
  %fi15 = and i32 %fi, 15
  %xa8 = extractelement <16 x i8> %a8, i32 %fi15
  %xa16 = extractelement <8 x i16> %a16, i32 %fi7
  %xa8z = zext i8 %xa8 to i32
  %xa8s = sext i8 %xa8 to i32
  %xa16z = zext i16 %xa16 to i32
  %xa16s = sext i16 %xa16 to i32
  %xa8v = select i1 %issigned, i32 %xa8s, i32 %xa8z
  %xa16v = select i1 %issigned, i32 %xa16s, i32 %xa16z
  %xav = select i1 %isword, i32 %xa16v, i32 %xa8v
  %pxa = getelementptr [16 x i32], ptr %A, i32 0, i32 %fi15
  store i32 %xav, ptr %pxa, align 4
  %xb8 = extractelement <16 x i8> %b8, i32 %fi15
  %xb16 = extractelement <8 x i16> %b16, i32 %fi7
  %xb8z = zext i8 %xb8 to i32
  %xb8s = sext i8 %xb8 to i32
  %xb16z = zext i16 %xb16 to i32
  %xb16s = sext i16 %xb16 to i32
  %xb8v = select i1 %issigned, i32 %xb8s, i32 %xb8z
  %xb16v = select i1 %issigned, i32 %xb16s, i32 %xb16z
  %xbv = select i1 %isword, i32 %xb16v, i32 %xb8v
  %pxb = getelementptr [16 x i32], ptr %B, i32 0, i32 %fi15
  store i32 %xbv, ptr %pxb, align 4
  %fi.n = add i32 %fi, 1
  %fdone = icmp uge i32 %fi.n, %n
  br i1 %fdone, label %outer, label %fill
outer:
  %j = phi i32 [ 0, %fill ], [ %j.n, %jend ]
  %r1 = phi i32 [ 0, %fill ], [ %r1.n, %jend ]
  %skip = icmp eq i32 %agg, 2
  br i1 %skip, label %jend, label %inner
inner:
  %i = phi i32 [ 0, %outer ], [ %i.n, %inner ]
  %acc = phi i1 [ %acc0, %outer ], [ %acc.n, %inner ]
  %i15 = and i32 %i, 15
  %i1 = add i32 %i, 1
  %i1m = and i32 %i1, 15
  %pai = getelementptr [16 x i32], ptr %A, i32 0, i32 %i15
  %ai = load i32, ptr %pai, align 4
  %pai1 = getelementptr [16 x i32], ptr %A, i32 0, i32 %i1m
  %ai1 = load i32, ptr %pai1, align 4
  %j15 = and i32 %j, 15
  %pbj = getelementptr [16 x i32], ptr %B, i32 0, i32 %j15
  %bj = load i32, ptr %pbj, align 4
  %k = add i32 %j, %i
  %kin = icmp ult i32 %k, %n
  %k15 = and i32 %k, 15
  %pbk = getelementptr [16 x i32], ptr %B, i32 0, i32 %k15
  %bk = load i32, ptr %pbk, align 4
  %iv = icmp ult i32 %i, %la
  %i1v = icmp ult i32 %i1, %la
  %jv = icmp ult i32 %j, %lb
  %kv = icmp ult i32 %k, %lb
  %eqij = icmp eq i32 %ai, %bj
  %t0 = and i1 %iv, %jv
  %anyt = and i1 %t0, %eqij
  %par = and i32 %i, 1
  %even = icmp eq i32 %par, 0
  %ge = icmp sge i32 %bj, %ai
  %le = icmp sle i32 %bj, %ai1
  %rg0 = and i1 %t0, %i1v
  %rg1 = and i1 %rg0, %even
  %rg2 = and i1 %ge, %le
  %rngt = and i1 %rg1, %rg2
  %eqik = icmp eq i32 %ai, %bk
  %o0 = and i1 %kv, %eqik
  %niv = xor i1 %iv, true
  %ordt = or i1 %niv, %o0
  %ordacc = and i1 %acc, %ordt
  %ordnext = select i1 %kin, i1 %ordacc, i1 %acc
  %anyacc = or i1 %acc, %anyt
  %rngacc = or i1 %acc, %rngt
  %is0 = icmp eq i32 %agg, 0
  %is1 = icmp eq i32 %agg, 1
  %s1 = select i1 %is1, i1 %rngacc, i1 %ordnext
  %acc.n = select i1 %is0, i1 %anyacc, i1 %s1
  %i.n = add i32 %i, 1
  %idone = icmp uge i32 %i.n, %n
  br i1 %idone, label %jend, label %inner
jend:
  %accx = phi i1 [ false, %outer ], [ %acc.n, %inner ]
  %jj15 = and i32 %j, 15
  %pej = getelementptr [16 x i32], ptr %A, i32 0, i32 %jj15
  %aj = load i32, ptr %pej, align 4
  %pbj2 = getelementptr [16 x i32], ptr %B, i32 0, i32 %jj15
  %bj2 = load i32, ptr %pbj2, align 4
  %jva = icmp ult i32 %j, %la
  %jvb = icmp ult i32 %j, %lb
  %jeq = icmp eq i32 %aj, %bj2
  %bothv = and i1 %jva, %jvb
  %eqv = and i1 %bothv, %jeq
  %nova = xor i1 %jva, true
  %novb = xor i1 %jvb, true
  %bothn = and i1 %nova, %novb
  %each = or i1 %eqv, %bothn
  %bit = select i1 %skip, i1 %each, i1 %accx
  %bit32 = zext i1 %bit to i32
  %bsh = shl i32 %bit32, %j
  %r1.n = or i32 %r1, %bsh
  %j.n = add i32 %j, 1
  %jdone = icmp uge i32 %j.n, %n
  br i1 %jdone, label %fin, label %outer
fin:
  %nm1 = sub i32 %nbit, 1
  %vb = shl i32 1, %lb
  %vm = sub i32 %vb, 1
  %p1 = icmp eq i32 %pol, 1
  %p3 = icmp eq i32 %pol, 3
  %x3 = select i1 %p3, i32 %vm, i32 0
  %x = select i1 %p1, i32 %nm1, i32 %x3
  %res = xor i32 %r1.n, %x
  %cfb = icmp ne i32 %res, 0
  %cf = zext i1 %cfb to i32
  %zfb = icmp ult i32 %lb, %n
  %zf0 = zext i1 %zfb to i32
  %zf = shl i32 %zf0, 1
  %sfb = icmp ult i32 %la, %n
  %sf0 = zext i1 %sfb to i32
  %sf = shl i32 %sf0, 2
  %of0 = and i32 %res, 1
  %of = shl i32 %of0, 3
  %f1 = or i32 %cf, %zf
  %f2 = or i32 %f1, %sf
  %flags = or i32 %f2, %of
  %r.0 = insertvalue { i32, i32 } poison, i32 %res, 0
  %r.1 = insertvalue { i32, i32 } %r.0, i32 %flags, 1
  ret { i32, i32 } %r.1
}
";
