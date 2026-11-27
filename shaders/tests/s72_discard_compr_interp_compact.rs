//! S72 (Shaders Batch 2 / Refs #663, #482): Synthetic end-to-end tests for:
//! 1. Fragment `Statement::Kill` (`discard;`) emission on `EXP NULL vm`, `EXP MRT*/MRTZ done vm`,
//!    multi-MRT `vm` deduplication, pending-`vm` flush on return, and post-discard WGSL §16.4
//!    uniformity lowering (`textureSampleLevel(..., 0f)` instead of implicit-derivative `textureSample`).
//! 2. `EXP compr=true` pair enable mask (`en[0]` enables R+G from `vsrc0`, `en[2]` enables B+A from
//!    `vsrc1`, matching RDNA2 ISA `en = 0x5` / `0x1` / `0x4`).
//! 3. `v_interp_*_f16` `high` bit handling (extracting high half `.y` and paired low half `.x` via
//!    `unpack2x16float(bitcast<u32>(...))`).
//! 4. VS / Mesh / PS inter-stage `@location` wrap parity modulo 15 (`MAX_INTER_STAGE_LOCATIONS`).
//! 5. Lazy Naga IR helper `fn pg6_byte_perm(s0: u32, s1: u32, s2: u32) -> u32` for runtime
//!    `v_perm_b32` + zero-helper constant folding for literal `v_perm_b32`.
//! 6. Post-Naga WGSL whitespace/indentation compaction (`compact_wgsl_whitespace`) preserving all
//!    downstream parser contracts (`@group(0) @binding(N) var<...>`, `@compute @workgroup_size(...) fn`,
//!    comments, LF-only, trailing LF).

use naga::valid::{Capabilities, ValidationFlags, Validator};
use playgta6_shaders::convert::ShaderStage;
use playgta6_shaders::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, Rdna2WgslOutput, compact_wgsl_whitespace,
    translate_rdna2_to_wgsl,
};

const BASELINE_WEBGPU_CAPABILITIES: Capabilities =
    Capabilities::SHADER_FLOAT16_IN_FLOAT32.union(Capabilities::CUBE_ARRAY_TEXTURES);

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

fn enc_vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    (0b0111_110u32 << 25)
        | (u32::from(op) << 17)
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
    op_sel: u8,
    clamp: bool,
    omod: u8,
) -> [u32; 2] {
    let w0 = (0b110101u32 << 26)
        | (u32::from(op & 0x3FF) << 16)
        | (u32::from(clamp) << 15)
        | (u32::from(op_sel & 0xF) << 11)
        | u32::from(vdst);
    let w1 = (u32::from(omod & 0x3) << 27)
        | (u32::from(src2 & 0x1FF) << 18)
        | (u32::from(src1 & 0x1FF) << 9)
        | u32::from(src0 & 0x1FF);
    [w0, w1]
}

fn enc_vintrp(op: u8, vdst: u8, attr: u8, attrchan: u8, vsrc: u8) -> u32 {
    (0b110010u32 << 26)
        | (u32::from(vdst) << 18)
        | (u32::from(op & 0x3) << 16)
        | (u32::from(attr & 0x3F) << 10)
        | (u32::from(attrchan & 0x3) << 8)
        | u32::from(vsrc)
}

fn enc_exp(
    target: u8,
    en: u8,
    compr: bool,
    done: bool,
    vm: bool,
    vsrc: [u8; 4],
) -> [u32; 2] {
    let w0 = (0b111110u32 << 26)
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

#[allow(clippy::too_many_arguments)]
fn enc_mimg(
    op: u8,
    dmask: u8,
    dim: u8,
    vaddr: u8,
    vdata: u8,
    srsrc: u8,
    ssamp: u8,
) -> [u32; 2] {
    let w0 = (0b111100u32 << 26)
        | (u32::from(op & 0x7F) << 18)
        | (u32::from(dmask & 0xF) << 8)
        | (u32::from(dim & 0x7) << 3);
    let w1 = u32::from(vaddr)
        | (u32::from(vdata) << 8)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(ssamp & 0x1F) << 21);
    [w0, w1]
}

fn translate_and_validate(words: &[u32], stage: Rdna2StageOverride) -> Rdna2WgslOutput {
    let out = translate_rdna2_to_wgsl(
        &words_to_bytes(words),
        &Rdna2TranslateOptions {
            stage,
            ..Rdna2TranslateOptions::default()
        },
    )
    .expect("synthetic shader must translate");
    assert!(out.naga_validated, "expected naga_validated:\n{}", out.wgsl);
    assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
    assert!(!out.wgsl.contains('\r'), "WGSL must be LF-only");
    assert!(out.wgsl.ends_with('\n'), "WGSL must end with LF");

    let module = naga::front::wgsl::parse_str(&out.wgsl)
        .unwrap_or_else(|err| panic!("naga WGSL parse failed:\n{err}\n\nWGSL:\n{}", out.wgsl));
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    validator
        .validate(&module)
        .unwrap_or_else(|err| panic!("naga WGSL validate failed:\n{err}\n\nWGSL:\n{}", out.wgsl));
    out
}

#[test]
fn s72_fragment_exp_null_vm_and_branchless_mrt_vm_emit_discard() {
    // 1. Branching early-discard pattern:
    //    v_interp_p1_f32 v2, v0, attr0.x
    //    v_cmpx_gt_f32 exec, 0.5, v2   (kills lanes where v2 <= 0.5)
    //    s_cbranch_execz +2            (jumps to L_kill if exec_mask == 0)
    //    exp mrt0 v2, v2, v2, v2 done vm
    //    s_endpgm
    //  L_kill:
    //    exp null off, off, off, off done vm
    //    s_endpgm
    let mut branching = Vec::new();
    branching.push(enc_vintrp(0, 2, 0, 0, 0));
    branching.push(enc_vopc(0x14, 240, 2)); // v_cmpx_gt_f32 0.5, v2
    branching.push(enc_sopp(0x08, 3)); // s_cbranch_execz -> skip exp mrt0 (2 words) + s_endpgm (1 word)
    branching.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [2, 2, 2, 2]));
    branching.push(enc_sopp(0x01, 0));
    branching.extend_from_slice(&enc_exp(9, 0x0, false, true, true, [0, 0, 0, 0])); // EXP NULL done vm
    branching.push(enc_sopp(0x01, 0));

    let out_branch = translate_and_validate(&branching, Rdna2StageOverride::Pixel);
    assert_eq!(out_branch.stage, ShaderStage::Fragment);
    assert!(
        out_branch.wgsl.contains("discard;"),
        "EXP NULL vm + EXP MRT0 done vm must emit discard:\n{}",
        out_branch.wgsl
    );
    assert!(
        out_branch.wgsl.contains("return "),
        "fragment shader with discard must still emit return ps_out:\n{}",
        out_branch.wgsl
    );

    // 2. Branchless alpha-test pattern: v_cmpx_* followed directly by EXP MRT0 done vm.
    let mut branchless = Vec::new();
    branchless.push(enc_vintrp(0, 2, 0, 0, 0));
    branchless.push(enc_vopc(0x14, 240, 2)); // v_cmpx_gt_f32
    branchless.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [2, 2, 2, 2]));
    branchless.push(enc_sopp(0x01, 0));

    let out_branchless = translate_and_validate(&branchless, Rdna2StageOverride::Pixel);
    assert_eq!(
        out_branchless.wgsl.matches("discard;").count(),
        1,
        "branchless v_cmpx + EXP MRT0 done vm must emit exactly 1 discard:\n{}",
        out_branchless.wgsl
    );

    // 3. Multi-MRT with vm=1 on both mrt0 (done=0) and mrt1 (done=1): emit discard ONCE.
    let mut multi_mrt = Vec::new();
    multi_mrt.push(enc_vintrp(0, 2, 0, 0, 0));
    multi_mrt.push(enc_vopc(0x14, 240, 2));
    multi_mrt.extend_from_slice(&enc_exp(0, 0xF, false, false, true, [2, 2, 2, 2]));
    multi_mrt.extend_from_slice(&enc_exp(1, 0xF, false, true, true, [2, 2, 2, 2]));
    multi_mrt.push(enc_sopp(0x01, 0));

    let out_multi = translate_and_validate(&multi_mrt, Rdna2StageOverride::Pixel);
    assert_eq!(
        out_multi.wgsl.matches("discard;").count(),
        1,
        "multi-MRT vm exports must deduplicate to a single discard:\n{}",
        out_multi.wgsl
    );

    // 4. Fragment shader with EXP MRT0 vm=1, done=0 followed by s_endpgm: flushes discard on return.
    let mut pending_vm = Vec::new();
    pending_vm.push(enc_vintrp(0, 2, 0, 0, 0));
    pending_vm.push(enc_vopc(0x14, 240, 2));
    pending_vm.extend_from_slice(&enc_exp(0, 0xF, false, false, true, [2, 2, 2, 2]));
    pending_vm.push(enc_sopp(0x01, 0));

    let out_pending = translate_and_validate(&pending_vm, Rdna2StageOverride::Pixel);
    assert_eq!(
        out_pending.wgsl.matches("discard;").count(),
        1,
        "vm=1 done=0 followed by s_endpgm must flush discard before return:\n{}",
        out_pending.wgsl
    );

    // 5. Straight-line fragment shader with mid-shader EXP NULL vm BEFORE image_sample:
    //    must mark subsequent CF non-uniform so image_sample lowers to explicit textureSampleLevel
    //    for WGSL §16.4 uniformity (Dawn/Tint compliance).
    let mut mid_discard_sample = Vec::new();
    mid_discard_sample.push(enc_vintrp(0, 2, 0, 0, 0));
    mid_discard_sample.push(enc_vintrp(0, 3, 0, 1, 0));
    mid_discard_sample.push(enc_vopc(0x14, 240, 2));
    mid_discard_sample.extend_from_slice(&enc_exp(9, 0x0, false, false, true, [0, 0, 0, 0])); // EXP NULL vm
    mid_discard_sample.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 0, 0)); // image_sample 2D
    mid_discard_sample.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
    mid_discard_sample.push(enc_sopp(0x01, 0));

    let out_mid = translate_and_validate(&mid_discard_sample, Rdna2StageOverride::Pixel);
    assert!(out_mid.wgsl.contains("discard;"), "WGSL:\n{}", out_mid.wgsl);
    assert!(
        out_mid.wgsl.contains("textureSampleLevel("),
        "post-discard straight-line image_sample must use textureSampleLevel for WGSL §16.4 uniformity:\n{}",
        out_mid.wgsl
    );
    assert!(
        !out_mid.wgsl.contains("textureSample("),
        "post-discard image_sample must not emit implicit-derivative textureSample:\n{}",
        out_mid.wgsl
    );

    // 6. Fragment shader with EXP NULL done vm and NO exec-mask ops must NOT emit discard.
    let mut no_exec = Vec::new();
    no_exec.push(enc_vintrp(0, 2, 0, 0, 0));
    no_exec.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [2, 2, 2, 2]));
    no_exec.push(enc_sopp(0x01, 0));
    let out_no_exec = translate_and_validate(&no_exec, Rdna2StageOverride::Pixel);
    assert!(
        !out_no_exec.wgsl.contains("discard;"),
        "shader without exec_mask mutations must not emit spurious discard:\n{}",
        out_no_exec.wgsl
    );
}

#[test]
fn s72_exp_compr_pair_enable_mask_en0_rg_and_en2_ba() {
    // Standard RDNA2 compressed RGBA export uses en = 0x5 (0b0101):
    //   en[0] -> unpack2x16float(vsrc[0]).xy for R and G
    //   en[2] -> unpack2x16float(vsrc[1]).xy for B and A
    let mut en5 = Vec::new();
    en5.push(enc_vintrp(0, 2, 0, 0, 0));
    en5.push(enc_vintrp(0, 3, 0, 1, 0));
    en5.extend_from_slice(&enc_exp(0, 0x5, true, true, true, [2, 3, 0, 0]));
    en5.push(enc_sopp(0x01, 0));

    let out_en5 = translate_and_validate(&en5, Rdna2StageOverride::Pixel);
    assert_eq!(
        out_en5.wgsl.matches("unpack2x16float(").count(),
        2,
        "en=0x5 compr export must unpack both vsrc0 and vsrc1:\n{}",
        out_en5.wgsl
    );
    assert!(
        out_en5.wgsl.contains(".x") && out_en5.wgsl.contains(".y"),
        "en=0x5 compr export must read both .x and .y of each unpacked fp16x2 pair:\n{}",
        out_en5.wgsl
    );
    // Must NOT hardcode G = 0f or A = 1f when en = 0x5!
    let mrt_line = out_en5
        .wgsl
        .lines()
        .find(|l| l.contains("ps_out.mrt_0_ = vec4<f32>("))
        .unwrap_or_else(|| panic!("missing mrt_0 assignment:\n{}", out_en5.wgsl));
    assert!(
        !mrt_line.contains("0f") && !mrt_line.contains("1f"),
        "en=0x5 compr export must not drop G (0f) or A (1f): {mrt_line}"
    );

    // RG-only compressed export: en = 0x1 -> unpacks vsrc0 (.x, .y) and defaults B=0f, A=1f.
    let mut en1 = Vec::new();
    en1.push(enc_vintrp(0, 2, 0, 0, 0));
    en1.extend_from_slice(&enc_exp(0, 0x1, true, true, true, [2, 3, 0, 0]));
    en1.push(enc_sopp(0x01, 0));
    let out_en1 = translate_and_validate(&en1, Rdna2StageOverride::Pixel);
    assert_eq!(
        out_en1.wgsl.matches("unpack2x16float(").count(),
        1,
        "en=0x1 compr export must unpack only vsrc0:\n{}",
        out_en1.wgsl
    );
    let mrt1_line = out_en1
        .wgsl
        .lines()
        .find(|l| l.contains("ps_out.mrt_0_ = vec4<f32>("))
        .unwrap_or_else(|| panic!("missing mrt_0 assignment:\n{}", out_en1.wgsl));
    assert!(
        mrt1_line.ends_with(", 0f, 1f);"),
        "en=0x1 compr export must keep R+G from vsrc0 and default B=0f, A=1f: {mrt1_line}"
    );

    // BA-only compressed export: en = 0x4 -> defaults R=0f, G=0f and unpacks vsrc1 (.x, .y).
    let mut en4 = Vec::new();
    en4.push(enc_vintrp(0, 3, 0, 1, 0));
    en4.extend_from_slice(&enc_exp(0, 0x4, true, true, true, [2, 3, 0, 0]));
    en4.push(enc_sopp(0x01, 0));
    let out_en4 = translate_and_validate(&en4, Rdna2StageOverride::Pixel);
    assert_eq!(
        out_en4.wgsl.matches("unpack2x16float(").count(),
        1,
        "en=0x4 compr export must unpack only vsrc1:\n{}",
        out_en4.wgsl
    );
    let mrt4_line = out_en4
        .wgsl
        .lines()
        .find(|l| l.contains("ps_out.mrt_0_ = vec4<f32>("))
        .unwrap_or_else(|| panic!("missing mrt_0 assignment:\n{}", out_en4.wgsl));
    assert!(
        mrt4_line.contains("vec4<f32>(0f, 0f, "),
        "en=0x4 compr export must default R=0f, G=0f and unpack B+A from vsrc1: {mrt4_line}"
    );
}

#[test]
fn s72_v_interp_f16_high_bit_unpacks_upper_and_paired_lower_halves() {
    // Fragment shader interpolating both low (high=0) and high (high=1) f16 halves of attr0.x:
    //   v_interp_p1ll_f16 v4, v0, attr0.x        (high=0 -> .x of unpack2x16float(bitcast<u32>(attr_0.x)))
    //   v_interp_p1ll_f16 v5, v0, attr0.x high   (high=1 -> .y of unpack2x16float(bitcast<u32>(attr_0.x)))
    //   v_interp_p2_f16   v6, v1, attr0.x, v4    (op_sel=0 -> v6.lo)
    //   v_interp_p2_f16   v6, v1, attr0.x, v5 high op_sel:[0,0,0,1] (v6.hi)
    //   exp mrt0 v6, off, off, off done compr vm (en=0x1)
    let attr0_x_lo: u16 = 0;
    let attr0_x_hi: u16 = 1 << 8;

    let mut words = Vec::new();
    words.extend_from_slice(&enc_vop3(0x342, 4, attr0_x_lo, 256 + 0, 0, 0, false, 0));
    words.extend_from_slice(&enc_vop3(0x342, 5, attr0_x_hi, 256 + 0, 0, 0, false, 0));
    words.extend_from_slice(&enc_vop3(0x35A, 6, attr0_x_lo, 256 + 1, 256 + 4, 0, false, 0));
    words.extend_from_slice(&enc_vop3(
        0x35A,
        6,
        attr0_x_hi,
        256 + 1,
        256 + 5,
        0b1000,
        false,
        0,
    ));
    // Export v6 plus v4/v5 so p1ll results remain live (not pruned by dead-local elimination).
    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [6, 4, 5, 0]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_and_validate(&words, Rdna2StageOverride::Pixel);
    assert!(
        out.wgsl.contains("unpack2x16float(bitcast<u32>("),
        "v_interp_*_f16 with high bit must unpack fp16x2 via unpack2x16float(bitcast<u32>(...)):\n{}",
        out.wgsl
    );
    // Both .x (low half) and .y (high half) must be extracted from the unpacked pair;
    // intra-block CSE deduplicates identical `unpack2x16float(bitcast<u32>(attr0.x))` calls.
    let unpack_count = out.wgsl.matches("unpack2x16float(bitcast<u32>(").count();
    assert!(
        unpack_count >= 1,
        "expected unpack2x16float(bitcast<u32>(...)) for p1ll/p2 (got {unpack_count}):\n{}",
        out.wgsl
    );
    assert!(
        out.wgsl.contains(".x") && out.wgsl.contains(".y"),
        "expected both low (.x) and high (.y) unpacked f16 halves:\n{}",
        out.wgsl
    );
    assert!(
        out.wgsl.matches("pack2x16float(").count()
            .saturating_sub(out.wgsl.matches("unpack2x16float(").count())
            >= 2,
        "expected both low and high f16 halves re-packed into v6:\n{}",
        out.wgsl
    );
}

#[test]
fn s72_interstage_location_wrap_parity_mod_15_between_vs_and_ps() {
    // VS exporting param15 and param16 -> wraps modulo 15 to @location(0) and @location(1).
    let mut vs_words = Vec::new();
    vs_words.push(enc_vop1(0x01, 0, 242)); // v0 = 1.0
    vs_words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [0, 0, 0, 0])); // pos0
    vs_words.extend_from_slice(&enc_exp(32 + 15, 0xF, false, false, false, [0, 0, 0, 0])); // param15 -> loc 0
    vs_words.extend_from_slice(&enc_exp(32 + 16, 0xF, false, true, false, [0, 0, 0, 0])); // param16 -> loc 1
    vs_words.push(enc_sopp(0x01, 0));

    let vs_out = translate_and_validate(&vs_words, Rdna2StageOverride::Vertex);
    assert_eq!(vs_out.binding_summary.output_locations, vec![0, 1]);

    // PS reading attr15 (Vintrp f32) and attr16 (v_interp_p1ll_f16) -> MUST ALSO wrap modulo 15
    // to @location(0) and @location(1) so VS outputs and PS inputs match identically!
    let mut ps_words = Vec::new();
    ps_words.push(enc_vintrp(0, 2, 15, 0, 0)); // attr15.x -> loc 0 (15 % 15)
    ps_words.extend_from_slice(&enc_vop3(0x342, 3, 16, 256 + 0, 0, 0, false, 0)); // attr16.x -> loc 1 (16 % 15)
    ps_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [2, 3, 2, 3]));
    ps_words.push(enc_sopp(0x01, 0));

    let ps_out = translate_and_validate(&ps_words, Rdna2StageOverride::Pixel);
    assert_eq!(
        ps_out.binding_summary.fragment_input_locations,
        vs_out.binding_summary.output_locations,
        "VS and PS inter-stage @location wrapping must match modulo 15:\nVS:\n{}\nPS:\n{}",
        vs_out.wgsl,
        ps_out.wgsl
    );
    assert!(
        !ps_out.wgsl.contains("@location(15)"),
        "PS must never wrap to @location(15):\n{}",
        ps_out.wgsl
    );
}

#[test]
fn s72_v_perm_b32_uses_lazy_helper_for_runtime_and_const_folds_literals() {
    // 1. Runtime v_perm_b32 invoked 4 times: emits `fn pg6_byte_perm(` ONCE and calls it 4 times.
    let mut rt_words = Vec::new();
    rt_words.push(enc_vintrp(0, 2, 0, 0, 0));
    rt_words.push(enc_vintrp(0, 3, 0, 1, 0));
    rt_words.push(enc_vintrp(0, 4, 0, 2, 0));
    for dst in 5..9u8 {
        rt_words.extend_from_slice(&enc_vop3(
            0x344,
            dst,
            256 + 2,
            256 + 3,
            256 + 4,
            0,
            false,
            0,
        ));
    }
    rt_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [5, 6, 7, 8]));
    rt_words.push(enc_sopp(0x01, 0));

    let rt_out = translate_and_validate(&rt_words, Rdna2StageOverride::Pixel);
    assert_eq!(
        rt_out.wgsl.matches("fn pg6_byte_perm(").count(),
        1,
        "runtime v_perm_b32 must emit pg6_byte_perm helper exactly once:\n{}",
        rt_out.wgsl
    );
    assert_eq!(
        rt_out.wgsl.matches("pg6_byte_perm(").count(),
        5, // 1 definition + 4 call sites
        "expected 1 def + 4 call sites for pg6_byte_perm:\n{}",
        rt_out.wgsl
    );
    assert!(
        rt_out.wgsl.len() < 3000,
        "4x v_perm_b32 with helper must be compact (was ~9 KB inline, got {} bytes)",
        rt_out.wgsl.len()
    );

    // 2. All-literal v_perm_b32: folds directly in Rust without emitting pg6_byte_perm.
    //    src0 = 1 (0x00000001 -> inline const 129), src1 = 2 (0x00000002 -> inline const 130),
    //    src2 = 4 (0x00000004 -> inline const 132: byte0=src0[0]=1, byte1..3=src1[0]=2 -> 0x02020201).
    let mut const_words = Vec::new();
    const_words.extend_from_slice(&enc_vop3(0x344, 5, 129, 130, 132, 0, false, 0));
    const_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [5, 5, 5, 5]));
    const_words.push(enc_sopp(0x01, 0));

    let const_out = translate_and_validate(&const_words, Rdna2StageOverride::Pixel);
    assert!(
        !const_out.wgsl.contains("pg6_byte_perm"),
        "all-literal v_perm_b32 must constant-fold without emitting helper:\n{}",
        const_out.wgsl
    );
}

#[test]
fn s72_post_naga_wgsl_compaction_strips_indent_and_preserves_parser_contracts() {
    let raw = "// header comment\r\n\r\n@group(0) @binding(0) var<uniform> cb0: Cb0;\r\n\r\n@compute @workgroup_size(64, 1, 1)\r\nfn cs_main() {\r\n    var x: u32 = 0u;\r\n        if (x == 0u) {\r\n            x = 1u;   \t\r\n        }\r\n\r\n    return;\r\n}\r\n";
    let compacted = compact_wgsl_whitespace(raw);
    assert!(!compacted.contains('\r'));
    assert!(compacted.ends_with('\n'));
    assert!(
        !compacted.contains("\n\n"),
        "compacted WGSL must have no blank lines:\n{compacted}"
    );
    for line in compacted.lines() {
        assert!(!line.is_empty());
        assert!(
            !line.starts_with(' ') && !line.starts_with('\t'),
            "line still has leading indent: {line:?}"
        );
        assert!(
            !line.ends_with(' ') && !line.ends_with('\t'),
            "line still has trailing whitespace: {line:?}"
        );
    }
    // Downstream C++/JS/Python parser invariants preserved verbatim:
    assert!(compacted.contains("// header comment\n"));
    assert!(compacted.contains("@group(0) @binding(0) var<uniform> cb0: Cb0;\n"));
    assert!(compacted.contains("@compute @workgroup_size(64, 1, 1)\n"));
    assert!(compacted.contains("fn cs_main() {\n"));
    assert!(compacted.len() < raw.len());

    // End-to-end check on a real translated shader: zero leading spaces or blank lines.
    let mut words = Vec::new();
    words.push(enc_vintrp(0, 2, 0, 0, 0));
    words.push(enc_vopc(0x14, 240, 2));
    words.extend_from_slice(&enc_exp(0, 0x5, true, true, true, [2, 2, 0, 0]));
    words.push(enc_sopp(0x01, 0));
    let out = translate_and_validate(&words, Rdna2StageOverride::Pixel);
    assert!(!out.wgsl.contains("\n\n"));
    for line in out.wgsl.lines() {
        assert!(
            !line.starts_with(' ') && !line.starts_with('\t'),
            "emitted WGSL line has leading indentation: {line:?}"
        );
    }
}
