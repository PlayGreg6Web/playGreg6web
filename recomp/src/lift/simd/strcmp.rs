//! SSE4.2 string compares `pcmpistri/pcmpistrm/pcmpestri/pcmpestrm` (also VEX): all sixteen
//! format/aggregation/polarity combinations of the immediate through one helper
//! (`__playgta6_pcmpstr`, see [`helpers::PCMPSTR_IR`](super::helpers::PCMPSTR_IR)) that follows the
//! Intel SDM pseudo code, including the invalid-element rules and the flags (CF = result is
//! non-zero, ZF = second operand shorter than 16/8 elements, SF = first operand shorter, OF =
//! result bit 0; AF and PF are cleared).

use iced_x86::{Code, Instruction, Mnemonic};

use super::super::integer::{RAX, RCX, RDX};
use super::super::{Flag, FunctionLifter, R};
use super::{Elem, Intr, Ty};
use crate::lift::{BitIntrinsic, Support};

/// Whether [`FunctionLifter::lift_strcmp`] handles `m`.
pub(super) fn is_strcmp_supported(m: Mnemonic) -> bool {
    use Mnemonic as M;
    matches!(
        m,
        M::Pcmpistri
            | M::Vpcmpistri
            | M::Pcmpistrm
            | M::Vpcmpistrm
            | M::Pcmpestri
            | M::Pcmpestri64
            | M::Vpcmpestri
            | M::Vpcmpestri64
            | M::Pcmpestrm
            | M::Pcmpestrm64
            | M::Vpcmpestrm
            | M::Vpcmpestrm64
    )
}

impl FunctionLifter<'_> {
    pub(super) fn lift_strcmp(&mut self, ins: &Instruction, vex: bool) -> R<bool> {
        use Mnemonic as M;
        let m = ins.mnemonic();
        if !is_strcmp_supported(m) {
            return Ok(false);
        }
        let explicit = matches!(
            m,
            M::Pcmpestri
                | M::Pcmpestri64
                | M::Vpcmpestri
                | M::Vpcmpestri64
                | M::Pcmpestrm
                | M::Pcmpestrm64
                | M::Vpcmpestrm
                | M::Vpcmpestrm64
        );
        let index_form = matches!(
            m,
            M::Pcmpistri
                | M::Vpcmpistri
                | M::Pcmpestri
                | M::Pcmpestri64
                | M::Vpcmpestri
                | M::Vpcmpestri64
        );
        let imm = Self::imm8(ins)?;
        let a = self.v128(ins, 0, Elem::I64)?;
        let b = self.v128(ins, 1, Elem::I64)?;
        let (la, lb) = if explicit {
            let wide = matches!(
                ins.code(),
                Code::Pcmpestri64_xmm_xmmm128_imm8
                    | Code::Pcmpestrm64_xmm_xmmm128_imm8
                    | Code::VEX_Vpcmpestri64_xmm_xmmm128_imm8
                    | Code::VEX_Vpcmpestrm64_xmm_xmmm128_imm8
            );
            let mut lens = Vec::new();
            for reg in [RAX, RDX] {
                let v = self.load_gpr(reg)?;
                lens.push(if wide {
                    v
                } else {
                    let t = self.val(&format!("trunc i64 {v} to i32"));
                    self.val(&format!("sext i32 {t} to i64"))
                });
            }
            (lens[0].clone(), lens[1].clone())
        } else {
            ("0".to_string(), "0".to_string())
        };
        self.use_helper(Intr::PcmpStr);
        let r = self.val(&format!(
            "call {{ i32, i32 }} @__playgta6_pcmpstr(i32 {imm}, <2 x i64> {a}, <2 x i64> {b}, \
             i64 {la}, i64 {lb}, i1 {})",
            !explicit
        ));
        let res = self.val(&format!("extractvalue {{ i32, i32 }} {r}, 0"));
        let flags = self.val(&format!("extractvalue {{ i32, i32 }} {r}, 1"));
        for (flag, bit) in [(Flag::Cf, 0), (Flag::Zf, 1), (Flag::Sf, 2), (Flag::Of, 3)] {
            let sh = self.val(&format!("lshr i32 {flags}, {bit}"));
            let t = self.val(&format!("trunc i32 {sh} to i1"));
            self.set_flag(flag, &t);
        }
        self.set_flag(Flag::Af, "false");
        self.set_flag(Flag::Pf, "false");
        let n: u32 = if imm & 1 != 0 { 8 } else { 16 };
        if index_form {
            // Index of the least (or most) significant set bit of IntRes2, `n` when it is zero.
            let idx = if imm & 0x40 == 0 {
                let with_sentinel = self.val(&format!("or i32 {res}, {}", 1u32 << n));
                self.support
                    .insert(Support::BitIntrinsic(BitIntrinsic::Cttz, 32));
                self.val(&format!(
                    "call i32 @llvm.cttz.i32(i32 {with_sentinel}, i1 true)"
                ))
            } else {
                self.support
                    .insert(Support::BitIntrinsic(BitIntrinsic::Ctlz, 32));
                let lz = self.val(&format!("call i32 @llvm.ctlz.i32(i32 {res}, i1 false)"));
                let top = self.val(&format!("sub i32 31, {lz}"));
                let zero = self.val(&format!("icmp eq i32 {res}, 0"));
                self.val(&format!("select i1 {zero}, i32 {n}, i32 {top}"))
            };
            let wide = self.val(&format!("zext i32 {idx} to i64"));
            self.store_gpr(RCX, &wide)?;
        } else {
            let mask = if imm & 0x40 == 0 {
                let lo = self.val(&format!("zext i32 {res} to i64"));
                self.insert("zeroinitializer", Elem::I64, &lo, 0)
            } else if n == 16 {
                let bits = self.val(&format!("trunc i32 {res} to i16"));
                let v = self.val(&format!("bitcast i16 {bits} to <16 x i1>"));
                let e = self.val(&format!("sext <16 x i1> {v} to {}", Ty::v(Elem::I8).ir()));
                self.cast(&e, Ty::v(Elem::I8), Ty::v(Elem::I64))
            } else {
                let bits = self.val(&format!("trunc i32 {res} to i8"));
                let v = self.val(&format!("bitcast i8 {bits} to <8 x i1>"));
                let e = self.val(&format!("sext <8 x i1> {v} to {}", Ty::v(Elem::I16).ir()));
                self.cast(&e, Ty::v(Elem::I16), Ty::v(Elem::I64))
            };
            self.xset_half(iced_x86::Register::XMM0, 0, &mask, Elem::I64)?;
            if vex {
                self.xset_half(iced_x86::Register::YMM0, 1, "zeroinitializer", Elem::I64)?;
            }
        }
        Ok(true)
    }
}
