//! Host-CPU differential tests for the SSE4.2 string compares `pcmp{i,e}str{i,m}` (see
//! [`simd_diff`](super::simd_diff) for the harness): every combination of the data format,
//! aggregation, polarity and output-select bits of the immediate, with the result index or mask,
//! all flags and the implicit `rax`/`rdx`/`rcx`/`xmm0` operands compared.

use iced_x86::code_asm::*;

use super::simd_diff::{Check, Feature, Kind, Op, check_ops, op};

const EXACT: Check = Check::Exact;
const S42: Feature = Feature::Sse42;

/// Every valid immediate (bit 7 is reserved).
fn all_imms() -> impl Iterator<Item = i32> {
    0..0x80
}

#[test]
fn string_compare_index_forms_match_the_host_cpu() {
    let mut ops: Vec<Op> = Vec::new();
    for imm in all_imms() {
        ops.push(op(S42, Kind::I8, EXACT, |a| a.pcmpistri(xmm1, xmm2, imm)).cases(24));
        ops.push(
            op(S42, Kind::I8, EXACT, |a| a.pcmpestri(xmm1, xmm2, imm))
                .lengths()
                .cases(24),
        );
    }
    for imm in [0x00, 0x0C, 0x0D, 0x15, 0x41, 0x4A, 0x7E] {
        ops.push(op(S42, Kind::I8, EXACT, |a| {
            a.pcmpistri(xmm1, xmmword_ptr(rbx), imm)
        }));
        ops.push(
            op(S42, Kind::I8, EXACT, |a| {
                a.pcmpestri(xmm1, xmmword_ptr(rbx), imm)
            })
            .lengths(),
        );
        ops.push(
            op(S42, Kind::I8, EXACT, |a| a.pcmpestri64(xmm1, xmm2, imm))
                .lengths()
                .gpr(Kind::I64),
        );
        ops.push(op(Feature::Avx, Kind::I8, EXACT, |a| {
            a.vpcmpistri(xmm1, xmm2, imm)
        }));
        ops.push(
            op(Feature::Avx, Kind::I8, EXACT, |a| {
                a.vpcmpestri(xmm1, xmm2, imm)
            })
            .lengths(),
        );
    }
    check_ops("string_compare_index_forms_match_the_host_cpu", ops);
}

#[test]
fn string_compare_mask_forms_match_the_host_cpu() {
    let mut ops: Vec<Op> = Vec::new();
    for imm in all_imms() {
        ops.push(op(S42, Kind::I8, EXACT, |a| a.pcmpistrm(xmm1, xmm2, imm)).cases(16));
        ops.push(
            op(S42, Kind::I8, EXACT, |a| a.pcmpestrm(xmm1, xmm2, imm))
                .lengths()
                .cases(16),
        );
    }
    for imm in [0x00, 0x0C, 0x0D, 0x44, 0x4D, 0x7A] {
        ops.push(op(S42, Kind::I8, EXACT, |a| {
            a.pcmpistrm(xmm1, xmmword_ptr(rbx), imm)
        }));
        ops.push(op(S42, Kind::I8, EXACT, |a| a.pcmpestrm64(xmm1, xmm2, imm)).lengths());
        ops.push(op(Feature::Avx, Kind::I8, EXACT, |a| {
            a.vpcmpistrm(xmm1, xmm2, imm)
        }));
        ops.push(
            op(Feature::Avx, Kind::I8, EXACT, |a| {
                a.vpcmpestrm(xmm1, xmm2, imm)
            })
            .lengths(),
        );
    }
    check_ops("string_compare_mask_forms_match_the_host_cpu", ops);
}
