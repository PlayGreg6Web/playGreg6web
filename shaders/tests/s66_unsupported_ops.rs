//! S66 integration tests: verify clean, deterministic, Naga-validated WGSL lowering
//! for all newly lowered RDNA2 opcodes across VOP3P, VOP1, VOP2, VOP3, SOP*, SMEM, and DS.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use playgta6_shaders::convert::ShaderStage;
use playgta6_shaders::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, Rdna2WgslOutput, translate_rdna2_to_wgsl,
};

const BASELINE_CAPABILITIES: Capabilities = Capabilities::SHADER_FLOAT16_IN_FLOAT32;

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

fn enc_sopc(op: u8, ssrc0: u8, ssrc1: u8) -> u32 {
    (0b1011_1111_0u32 << 23)
        | (u32::from(op & 0x7F) << 16)
        | (u32::from(ssrc1) << 8)
        | u32::from(ssrc0)
}

fn enc_sop1(op: u8, sdst: u8, ssrc0: u8) -> u32 {
    (0b1011_1110_1u32 << 23)
        | (u32::from(sdst & 0x7F) << 16)
        | (u32::from(op) << 8)
        | u32::from(ssrc0)
}

fn enc_sopk(op: u8, sdst: u8, simm16: u16) -> u32 {
    (0b1011u32 << 28)
        | (u32::from(op & 0x1F) << 23)
        | (u32::from(sdst & 0x7F) << 16)
        | u32::from(simm16)
}

fn enc_sop2(op: u8, sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
    (0b10u32 << 30)
        | (u32::from(op & 0x3F) << 23)
        | (u32::from(sdst & 0x7F) << 16)
        | (u32::from(ssrc1) << 8)
        | u32::from(ssrc0)
}

fn enc_smem(op: u8, sdata: u8, sbase: u8, soffset: u8, offset: i32) -> [u32; 2] {
    let w0 = (0b111101u32 << 26)
        | (u32::from(op) << 18)
        | (u32::from(sdata & 0x7F) << 6)
        | u32::from(sbase & 0x3F);
    let w1 = (u32::from(soffset & 0x7F) << 25) | ((offset as u32) & 0x001F_FFFF);
    [w0, w1]
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

#[allow(clippy::too_many_arguments)]
fn enc_vop3(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    clamp: bool,
    op_sel: u8,
    abs: u8,
    neg: u8,
) -> [u32; 2] {
    let w0 = (0b110101u32 << 26)
        | (u32::from(op & 0x3FF) << 16)
        | (u32::from(clamp) << 15)
        | (u32::from(op_sel & 0xF) << 11)
        | (u32::from(abs & 0x7) << 8)
        | u32::from(vdst);
    let w1 = (u32::from(neg & 0x7) << 29)
        | (u32::from(src2 & 0x1FF) << 18)
        | (u32::from(src1 & 0x1FF) << 9)
        | u32::from(src0 & 0x1FF);
    [w0, w1]
}

#[allow(clippy::too_many_arguments)]
fn enc_vop3b(
    op: u16,
    vdst: u8,
    sdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    clamp: bool,
    neg: u8,
) -> [u32; 2] {
    let w0 = (0b110101u32 << 26)
        | (u32::from(op & 0x3FF) << 16)
        | (u32::from(clamp) << 15)
        | (u32::from(sdst & 0x7F) << 8)
        | u32::from(vdst);
    let w1 = (u32::from(neg & 0x7) << 29)
        | (u32::from(src2 & 0x1FF) << 18)
        | (u32::from(src1 & 0x1FF) << 9)
        | u32::from(src0 & 0x1FF);
    [w0, w1]
}

#[allow(clippy::too_many_arguments)]
fn enc_vop3p(
    op: u8,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    clamp: bool,
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
) -> [u32; 2] {
    let w0 = (0b110011u32 << 26)
        | (u32::from(op & 0x7F) << 16)
        | (u32::from(clamp) << 15)
        | (u32::from((op_sel_hi >> 2) & 1) << 14)
        | (u32::from(op_sel & 0x7) << 11)
        | (u32::from(neg_hi & 0x7) << 8)
        | u32::from(vdst);
    let w1 = (u32::from(neg_lo & 0x7) << 29)
        | (u32::from(op_sel_hi & 0x3) << 27)
        | (u32::from(src2 & 0x1FF) << 18)
        | (u32::from(src1 & 0x1FF) << 9)
        | u32::from(src0 & 0x1FF);
    [w0, w1]
}

fn enc_ds(op: u8, offset0: u8, offset1: u8, addr: u8, data0: u8, data1: u8, vdst: u8) -> [u32; 2] {
    let w0 = (0b110110u32 << 26)
        | (u32::from(op) << 18)
        | (u32::from(offset1) << 8)
        | u32::from(offset0);
    let w1 = (u32::from(vdst) << 24)
        | (u32::from(data1) << 16)
        | (u32::from(data0) << 8)
        | u32::from(addr);
    [w0, w1]
}

fn enc_exp(target: u8, en: u8, compr: bool, done: bool, vm: bool, vsrc: [u8; 4]) -> [u32; 2] {
    let w0 = (0b111110u32 << 26)
        | (u32::from(vm) << 12)
        | (u32::from(done) << 11)
        | (u32::from(compr) << 10)
        | (u32::from(target & 0x3F) << 4)
        | u32::from(en & 0xF);
    let w1 = (u32::from(vsrc[3]) << 24)
        | (u32::from(vsrc[2]) << 16)
        | (u32::from(vsrc[1]) << 8)
        | u32::from(vsrc[0]);
    [w0, w1]
}

fn translate_and_validate(words: &[u32], stage: Rdna2StageOverride) -> Rdna2WgslOutput {
    let bytes = words_to_bytes(words);
    let opts = Rdna2TranslateOptions {
        stage,
        ..Rdna2TranslateOptions::default()
    };
    let out = translate_rdna2_to_wgsl(&bytes, &opts)
        .expect("translation must succeed and pass internal Naga validation");
    assert!(out.naga_validated, "expected naga_validated=true");
    assert!(
        out.translated_op_count > 0,
        "expected translated_op_count > 0"
    );
    assert_eq!(
        out.unsupported_op_count, 0,
        "expected zero unsupported ops; WGSL:\n{}",
        out.wgsl
    );

    // Explicitly round-trip parse and validate emitted WGSL text with Naga.
    let module = naga::front::wgsl::parse_str(&out.wgsl)
        .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\nWGSL:\n{}", e.emit_to_string(&out.wgsl), out.wgsl));
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_CAPABILITIES);
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("Naga validation failed:\n{}\nWGSL:\n{}", e.emit_to_string(&out.wgsl), out.wgsl));

    out
}

#[test]
fn s66_vop3p_all_packed_i16_u16_and_dot_ops_unclamped_and_clamped() {
    let mut words = Vec::new();
    // Seed runtime VGPR inputs v0..v3 via compute local_id + arithmetic so expressions stay live.
    words.push(enc_vop2(0x25, 3, 256 + 0, 1)); // v_add_nc_u32 v3, v0, v1

    // All 14 packed 16-bit integer VOP3P ops (0x00..=0x0D), exercising op_sel/op_sel_hi & clamp:
    // 0x00 v_pk_mad_i16 (unclamped & clamped + swizzled op_sel/op_sel_hi)
    words.extend_from_slice(&enc_vop3p(0x00, 4, 256, 257, 258, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x00, 5, 256, 257, 258, true, 0b010, 0b101, 0, 0));
    // 0x01 v_pk_mul_lo_u16
    words.extend_from_slice(&enc_vop3p(0x01, 6, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    // 0x02 v_pk_add_i16 (unclamped & clamped)
    words.extend_from_slice(&enc_vop3p(0x02, 7, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x02, 8, 256, 257, 0, true, 0b001, 0b010, 0, 0));
    // 0x03 v_pk_sub_i16 (unclamped & clamped)
    words.extend_from_slice(&enc_vop3p(0x03, 9, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x03, 10, 256, 257, 0, true, 0b000, 0b111, 0, 0));
    // 0x04 v_pk_lshlrev_b16, 0x05 v_pk_lshrrev_b16, 0x06 v_pk_ashrrev_i16
    words.extend_from_slice(&enc_vop3p(0x04, 11, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x05, 12, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x06, 13, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    // 0x07 v_pk_max_i16, 0x08 v_pk_min_i16
    words.extend_from_slice(&enc_vop3p(0x07, 14, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x08, 15, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    // 0x09 v_pk_mad_u16 (unclamped & clamped)
    words.extend_from_slice(&enc_vop3p(0x09, 16, 256, 257, 258, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x09, 17, 256, 257, 258, true, 0b000, 0b111, 0, 0));
    // 0x0A v_pk_add_u16 (unclamped & clamped), 0x0B v_pk_sub_u16 (unclamped & clamped)
    words.extend_from_slice(&enc_vop3p(0x0A, 18, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x0A, 19, 256, 257, 0, true, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x0B, 20, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x0B, 21, 256, 257, 0, true, 0b000, 0b111, 0, 0));
    // 0x0C v_pk_max_u16, 0x0D v_pk_min_u16
    words.extend_from_slice(&enc_vop3p(0x0C, 22, 256, 257, 0, false, 0b000, 0b111, 0, 0));
    words.extend_from_slice(&enc_vop3p(0x0D, 23, 256, 257, 0, false, 0b000, 0b111, 0, 0));

    // All 6 integer dot-product VOP3P ops (0x14..=0x19), both unclamped & clamped:
    for (idx, op) in (0x14u8..=0x19u8).enumerate() {
        let dst_u = 24 + (idx as u8) * 2;
        let dst_c = dst_u + 1;
        words.extend_from_slice(&enc_vop3p(op, dst_u, 256, 257, 258, false, 0, 0b111, 0, 0));
        words.extend_from_slice(&enc_vop3p(op, dst_c, 256, 257, 258, true, 0, 0b111, 0, 0));
    }

    // Also test literal-constant operands on packed/dot VOP3P ops to verify const-eval safety.
    words.extend_from_slice(&enc_vop3p(0x00, 36, 0xFF, 128 + 3, 193, true, 0, 0b111, 0, 0));
    words.push(0x8000_7FFF);
    words.extend_from_slice(&enc_vop3p(0x14, 37, 0xFF, 193, 0xFF, true, 0, 0b111, 0, 0));
    words.push(0x8000_8000);

    words.push(enc_sopp(0x01, 0)); // s_endpgm

    let out = translate_and_validate(&words, Rdna2StageOverride::Compute);
    assert_eq!(out.stage, ShaderStage::Compute);
    assert!(out.translated_op_count >= 35);
}

#[test]
fn s66_vop1_f16_norm_frexp_ffbh_and_vop3_promoted_ops() {
    let mut words = Vec::new();
    // Runtime VGPRs v0..v3 from vertex attributes / compute local_id.
    // 0x0E v_cvt_off_f32_i4
    words.push(enc_vop1(0x0E, 4, 256 + 0));
    // 0x39 v_ffbh_u32, 0x3A v_ffbl_b32, 0x3B v_ffbh_i32
    words.push(enc_vop1(0x39, 5, 256 + 0));
    words.push(enc_vop1(0x3A, 6, 256 + 1));
    words.push(enc_vop1(0x3B, 7, 256 + 2));
    // 0x3F v_frexp_exp_i32_f32, 0x40 v_frexp_mant_f32
    words.push(enc_vop1(0x3F, 8, 256 + 0));
    words.push(enc_vop1(0x40, 9, 256 + 1));
    // 0x50..=0x53 v_cvt_f16_u16, v_cvt_f16_i16, v_cvt_u16_f16, v_cvt_i16_f16
    words.push(enc_vop1(0x50, 10, 256 + 0));
    words.push(enc_vop1(0x51, 11, 256 + 1));
    words.push(enc_vop1(0x52, 12, 256 + 10));
    words.push(enc_vop1(0x53, 13, 256 + 11));
    // 0x59 v_frexp_mant_f16, 0x5A v_frexp_exp_i16_f16
    words.push(enc_vop1(0x59, 14, 256 + 10));
    words.push(enc_vop1(0x5A, 15, 256 + 11));
    // 0x60 v_sin_f16, 0x61 v_cos_f16
    words.push(enc_vop1(0x60, 16, 256 + 10));
    words.push(enc_vop1(0x61, 17, 256 + 11));
    // 0x62 v_sat_pk_u8_i16, 0x63 v_cvt_norm_i16_f16, 0x64 v_cvt_norm_u16_f16
    words.push(enc_vop1(0x62, 18, 256 + 0));
    words.push(enc_vop1(0x63, 19, 256 + 16));
    words.push(enc_vop1(0x64, 20, 256 + 17));

    // VOP3-promoted forms with clamp/abs/neg + inline float constants (const-eval safety):
    words.extend_from_slice(&enc_vop3(0x180 + 0x3F, 21, 242, 0, 0, false, 0, 0, 1)); // v_frexp_exp_i32_f32(-1.0)
    words.extend_from_slice(&enc_vop3(0x180 + 0x40, 22, 244, 0, 0, true, 0, 1, 0)); // v_frexp_mant_f32(|2.0|) clamp
    words.extend_from_slice(&enc_vop3(0x180 + 0x61, 23, 240, 0, 0, true, 0, 0, 1)); // v_cos_f16(-0.5) clamp
    words.extend_from_slice(&enc_vop3(0x180 + 0x3B, 24, 193, 0, 0, false, 0, 0, 0)); // v_ffbh_i32(-1)
    words.extend_from_slice(&enc_vop3(0x180 + 0x59, 25, 242, 0, 0, false, 0, 0, 0)); // v_frexp_mant_f16(1.0)
    words.extend_from_slice(&enc_vop3(0x180 + 0x5A, 26, 242, 0, 0, false, 0, 0, 0)); // v_frexp_exp_i16_f16(1.0)

    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [9, 17, 19, 20]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_and_validate(&words, Rdna2StageOverride::Pixel);
    assert_eq!(out.stage, ShaderStage::Fragment);
    assert!(out.wgsl.contains("countLeadingZeros("));
    assert!(out.wgsl.contains("countTrailingZeros("));
    assert!(out.wgsl.contains("cos("));
    assert!(out.wgsl.contains("sin("));
}

#[test]
fn s66_vop2_and_vop3_remaining_ops_and_div_sequence() {
    let mut words = Vec::new();
    // VOP2: 0x0D v_dot4c_i32_i8, 0x0E v_dot8c_i32_i4, 0x3B v_ldexp_f16
    words.push(enc_vop2(0x0D, 3, 256 + 0, 1));
    words.push(enc_vop2(0x0E, 4, 256 + 0, 1));
    words.push(enc_vop2(0x3B, 5, 256 + 0, 1));
    // VOP3-promoted v_ldexp_f16 with constant inputs (tests const-eval safety)
    words.extend_from_slice(&enc_vop3(0x13B, 6, 242, 128 + 4, 0, false, 0, 0, 0));
    // VOP3-promoted v_dot4c_i32_i8 with clamp
    words.extend_from_slice(&enc_vop3(0x10D, 7, 256 + 0, 256 + 1, 0, true, 0, 0, 0));

    // VOP3: 0x149 v_bfe_i32, 0x14D v_lerp_u8, 0x14F v_alignbyte_b32
    words.extend_from_slice(&enc_vop3(0x149, 8, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0));
    words.extend_from_slice(&enc_vop3(0x14D, 9, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0));
    words.extend_from_slice(&enc_vop3(0x14F, 10, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0));

    // VOP3 SAD / MSAD: 0x15A, 0x15B, 0x15C, 0x15D, 0x165 (unclamped + clamped)
    words.extend_from_slice(&enc_vop3(0x15A, 11, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0));
    words.extend_from_slice(&enc_vop3(0x15B, 12, 256 + 0, 256 + 1, 256 + 2, true, 0, 0, 0));
    words.extend_from_slice(&enc_vop3(0x15C, 13, 256 + 0, 256 + 1, 256 + 2, true, 0, 0, 0));
    words.extend_from_slice(&enc_vop3(0x15D, 14, 256 + 0, 256 + 1, 256 + 2, true, 0, 0, 0));
    words.extend_from_slice(&enc_vop3(0x165, 15, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0));

    // Full AMD RDNA2 float division sequence: v_div_scale_f32 -> v_rcp_f32 -> v_div_fmas_f32 -> v_div_fixup_f32
    words.extend_from_slice(&enc_vop3b(0x16D, 16, 106, 256 + 0, 256 + 1, 256 + 0, false, 0));
    words.extend_from_slice(&enc_vop3b(0x16D, 17, 106, 256 + 1, 256 + 1, 256 + 0, false, 0));
    words.push(enc_vop1(0x2A, 18, 256 + 17)); // v_rcp_f32 v18, v17
    words.extend_from_slice(&enc_vop3(0x163, 19, 256 + 16, 256 + 18, 256 + 2, false, 0, 0, 0));
    words.extend_from_slice(&enc_vop3(0x15F, 20, 256 + 19, 256 + 1, 256 + 0, true, 0, 0, 0));

    // VOP3 16-bit & byte permute: 0x340 v_mad_u16, 0x344 v_perm_b32, 0x34B v_fma_f16 (hi op_sel), 0x35E v_mad_i16
    words.extend_from_slice(&enc_vop3(0x340, 21, 256 + 0, 256 + 1, 256 + 2, true, 0b1011, 0, 0));
    words.extend_from_slice(&enc_vop3(0x344, 22, 256 + 0, 256 + 1, 256 + 2, false, 0, 0, 0));
    words.extend_from_slice(&enc_vop3(0x34B, 23, 256 + 0, 256 + 1, 256 + 2, false, 0b1101, 0, 0));
    words.extend_from_slice(&enc_vop3(0x35E, 24, 256 + 0, 256 + 1, 256 + 2, true, 0b1110, 0, 0));

    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [20, 21, 22, 24]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_and_validate(&words, Rdna2StageOverride::Pixel);
    assert_eq!(out.stage, ShaderStage::Fragment);
    assert!(out.wgsl.contains("ldexp("));
}

#[test]
fn s66_sop_and_smem_ops() {
    let mut words = Vec::new();
    // SOPP hints: s_wakeup, s_sethalt, s_sleep, s_icache_inv, s_incperflevel, s_decperflevel,
    // s_ttracedata, s_set_gpr_idx_off, s_denorm_mode, s_ttracedata_imm
    for op in [0x03u8, 0x0D, 0x0E, 0x13, 0x14, 0x15, 0x16, 0x1C, 0x25, 0x28] {
        words.push(enc_sopp(op, 0));
    }

    // SOP1: s_bcnt0_i32_b32 (0x0D), s_ff0_i32_b32 (0x11), s_flbit_i32 (0x17), s_quadmask_b32 (0x2C)
    words.push(enc_sop1(0x0D, 4, 0));
    words.push(enc_sop1(0x11, 5, 1));
    words.push(enc_sop1(0x17, 6, 2));
    words.push(enc_sop1(0x2C, 7, 0));

    // SOPK: s_version (0x01), s_cmovk_i32 (0x02), s_setreg_b32 (0x12), s_setreg_imm32_b32 (0x15 + lit)
    words.push(enc_sopk(0x01, 0, 0));
    words.push(enc_sopc(0x00, 4, 5)); // sets SCC for s_cmovk_i32
    words.push(enc_sopk(0x02, 8, 0x1234));
    words.push(enc_sopk(0x12, 8, 0x0001));
    words.push(enc_sopk(0x15, 0, 0x0001));
    words.push(0x0000_0007); // imm32 for s_setreg_imm32_b32

    // SOP2: s_bfe_i32 (0x28), s_bfe_u64 (0x29), s_bfe_i64 (0x2A), s_absdiff_i32 (0x2C)
    words.push(enc_sop2(0x28, 9, 0, 1));
    words.push(enc_sop2(0x29, 10, 0, 2)); // s[10:11], s[0:1], s2
    words.push(enc_sop2(0x2A, 12, 0, 2)); // s[12:13], s[0:1], s2
    words.push(enc_sop2(0x2C, 14, 0, 1));

    // SMEM cache/probe hints: s_gl1_inv (0x20), s_dcache_inv (0x21), s_atc_probe (0x26), s_atc_probe_buffer (0x27)
    for op in [0x20u8, 0x21, 0x26, 0x27] {
        words.extend_from_slice(&enc_smem(op, 0, 0, 0x7D, 0));
    }

    // Move results into v0 and end
    words.push(enc_vop1(0x01, 0, 14));
    words.push(enc_sopp(0x01, 0));

    let out = translate_and_validate(&words, Rdna2StageOverride::Compute);
    assert_eq!(out.stage, ShaderStage::Compute);
    assert!(out.translated_op_count >= 24);
}

#[test]
fn s66_ds_subdword_byte_and_short_lds_reads_and_writes() {
    let mut words = Vec::new();
    // v0 is local_id.x (addr), v1 is local_id.y (data)
    // ds_write_b8 (0x1E) with offset 0 and offset 3
    words.extend_from_slice(&enc_ds(0x1E, 0, 0, 0, 1, 0, 0));
    words.extend_from_slice(&enc_ds(0x1E, 3, 0, 0, 1, 0, 0));
    // ds_write_b16 (0x1F) with offset 2 and 16-bit offset 0x0102
    words.extend_from_slice(&enc_ds(0x1F, 2, 0, 0, 1, 0, 0));
    words.extend_from_slice(&enc_ds(0x1F, 2, 1, 0, 1, 0, 0));
    // s_barrier between writes and reads
    words.push(enc_sopp(0x0A, 0));
    // ds_read_i8 (0x39), ds_read_u8 (0x3A), ds_read_i16 (0x3B), ds_read_u16 (0x3C)
    words.extend_from_slice(&enc_ds(0x39, 1, 0, 0, 0, 0, 4));
    words.extend_from_slice(&enc_ds(0x3A, 3, 0, 0, 0, 0, 5));
    words.extend_from_slice(&enc_ds(0x3B, 2, 0, 0, 0, 0, 6));
    words.extend_from_slice(&enc_ds(0x3C, 2, 1, 0, 0, 0, 7));
    words.push(enc_sopp(0x01, 0));

    let out = translate_and_validate(&words, Rdna2StageOverride::Compute);
    assert_eq!(out.stage, ShaderStage::Compute);
    assert!(out.binding_summary.has_lds, "expected has_lds=true");
    assert!(out.wgsl.contains("workgroupBarrier()"));
    assert!(out.wgsl.contains("lds["), "WGSL:\n{}", out.wgsl);
    assert_eq!(out.translated_op_count, 10);
}

#[test]
fn s66_constant_fold_semantics_and_determinism() {
    let mut words = Vec::new();
    // 1. v_ffbh_i32(0x0000_0001) -> 31u (31 leading sign bits matching 0)
    words.push(enc_vop1(0x3B, 4, 128 + 1));
    // 2. v_ffbh_i32(-1) -> 0xFFFF_FFFFu
    words.push(enc_vop1(0x3B, 5, 193));
    // 3. v_frexp_exp_i32_f32(4.0) -> 3 (0.5 * 2^3 = 4.0)
    words.push(enc_vop1(0x3F, 6, 246));
    // 4. v_frexp_mant_f32(4.0) -> 0.5f
    words.push(enc_vop1(0x40, 7, 246));
    // 5. v_pk_sub_u16 clamped: (2, 10) - (5, 3) -> (0, 7) = 0x0007_0000u
    words.extend_from_slice(&enc_vop3p(0x0B, 8, 0xFF, 0xFF, 0, true, 0, 0b11, 0, 0));
    words.push(0x000A_0002); // literal shared by src0 and src1 -> (2-2=0, 10-10=0)
    // 6. v_dot4_u32_u8: (0x01020304 . 0x01020304) + 10 = (1+4+9+16) + 10 = 40u
    words.extend_from_slice(&enc_vop3p(0x17, 9, 0xFF, 0xFF, 128 + 10, false, 0, 0b111, 0, 0));
    words.push(0x0102_0304);
    // 7. s_absdiff_i32(-5, 3) -> 8u
    words.push(enc_sop2(0x2C, 4, 192 + 5, 128 + 3));
    // 8. v_perm_b32(src0=0x44332211, src1=0x44332211, sel=0x0C_0D_03_00) -> 0x00_FF_44_11
    words.extend_from_slice(&enc_vop3(0x344, 10, 0xFF, 0xFF, 128 + 12, false, 0, 0, 0));
    words.push(0x4433_2211);
    // 9. Boundary VGPR v255 with v_pk_add_i16 and v_cvt_f16_u16
    words.extend_from_slice(&enc_vop3p(0x02, 255, 256 + 0, 128 + 7, 0, true, 0, 0b11, 0, 0));
    words.push(enc_vop1(0x50, 255, 256 + 255));
    // Observe v7 and s4 via LDS so dead-local pruning retains their folded assignments.
    words.extend_from_slice(&enc_ds(0x0D, 0, 0, 0, 7, 0, 0));
    words.push(enc_vop1(0x01, 250, 4));
    words.extend_from_slice(&enc_ds(0x0D, 4, 0, 0, 250, 0, 0));
    words.push(enc_sopp(0x01, 0));

    let out1 = translate_and_validate(&words, Rdna2StageOverride::Compute);
    let out2 = translate_and_validate(&words, Rdna2StageOverride::Compute);
    assert_eq!(out1.wgsl, out2.wgsl, "translation must be deterministic");

    // Verify key folded constants appear verbatim in emitted WGSL.
    assert!(
        out1.wgsl.contains("pg6_u32_bits_to_float(31u)"),
        "expected v_ffbh_i32(1) == 31u; WGSL:\n{}",
        out1.wgsl
    );
    assert!(
        out1.wgsl.contains("pg6_u32_bits_to_float(4294967295u)"),
        "expected v_ffbh_i32(-1) == 0xFFFF_FFFFu; WGSL:\n{}",
        out1.wgsl
    );
    assert!(
        out1.wgsl.contains("pg6_u32_bits_to_float(3u)"),
        "expected v_frexp_exp_i32_f32(4.0) == 3; WGSL:\n{}",
        out1.wgsl
    );
    assert!(
        out1.wgsl.contains("v_7_ = 0.5f;"),
        "expected v_frexp_mant_f32(4.0) == 0.5f; WGSL:\n{}",
        out1.wgsl
    );
    assert!(
        out1.wgsl.contains("pg6_u32_bits_to_float(40u)"),
        "expected v_dot4_u32_u8 == 40u; WGSL:\n{}",
        out1.wgsl
    );
    assert!(
        out1.wgsl.contains("s_4_ = 8u;"),
        "expected s_absdiff_i32(-5, 3) == 8u; WGSL:\n{}",
        out1.wgsl
    );
}

