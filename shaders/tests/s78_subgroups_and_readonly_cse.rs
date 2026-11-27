//! S78 (#2178, Refs #663, #482, #2164): Hardware WebGPU `subgroups` wave lowering mode
//! (`WaveEmulMode::Subgroups` vs `WaveEmulMode::WorkgroupLds`) and deeper `ir_opt` passes
//! (constant-condition dead-branch folding, unreachable trimming, pure integer zero/negated-cmp
//! identities, and intra-block immutable read-only `Load` CSE).

use naga::valid::{Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};
use naga::{
    AddressSpace, BinaryOperator, Expression, Function, FunctionArgument, GlobalVariable, Literal,
    Module, Range, ResourceBinding, Scalar, ShaderStage as NagaStage, Span, Statement,
    StorageAccess, Type, TypeInner,
};
use playgta6_shaders::ir_opt::optimize_module;
use playgta6_shaders::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, WaveEmulMode, translate_rdna2_to_wgsl,
};
use playgta6_shaders::wave_emul::{
    RdnaComputeWaveEmulConfig, WGSL_SUBGROUPS_ENABLE, WaveLoweringOptions, WaveSize,
    lower_wave_ops, rdna_wave_emul_wgsl,
};

const BASELINE_WEBGPU_CAPABILITIES: Capabilities =
    Capabilities::SHADER_FLOAT16_IN_FLOAT32.union(Capabilities::CUBE_ARRAY_TEXTURES);

fn validate_wgsl_with_caps(wgsl: &str, caps: Capabilities) -> Module {
    let parse_src = if caps.contains(Capabilities::SUBGROUP) {
        wgsl.replacen(WGSL_SUBGROUPS_ENABLE, "", 1)
    } else {
        wgsl.to_string()
    };
    let module = naga::front::wgsl::parse_str(&parse_src).unwrap_or_else(|e| {
        panic!(
            "WGSL parse failed:\n{}\n\nWGSL:\n{wgsl}",
            e.emit_to_string(&parse_src)
        )
    });
    let mut validator = Validator::new(ValidationFlags::all(), caps);
    if caps.contains(Capabilities::SUBGROUP) {
        validator
            .subgroup_stages(ShaderStages::COMPUTE | ShaderStages::FRAGMENT)
            .subgroup_operations(SubgroupOperationSet::all());
    }
    validator.validate(&module).unwrap_or_else(|e| {
        panic!(
            "Naga validation failed (caps={caps:?}):\n{}\n\nWGSL:\n{wgsl}",
            e.emit_to_string(&parse_src)
        )
    });
    module
}

fn validate_and_emit_wgsl(module: &Module, caps: Capabilities) -> String {
    let mut validator = Validator::new(ValidationFlags::all(), caps);
    if caps.contains(Capabilities::SUBGROUP) {
        validator
            .subgroup_stages(ShaderStages::COMPUTE | ShaderStages::FRAGMENT)
            .subgroup_operations(SubgroupOperationSet::all());
    }
    let info = validator
        .validate(module)
        .unwrap_or_else(|e| panic!("naga validation failed:\n{}", e.emit_to_string("")));
    let wgsl = naga::back::wgsl::write_string(
        module,
        &info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .expect("naga WGSL emission must succeed");
    let _ = validate_wgsl_with_caps(&wgsl, caps);
    wgsl
}

fn parse_baseline_wgsl(src: &str) -> Module {
    validate_wgsl_with_caps(src, BASELINE_WEBGPU_CAPABILITIES)
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

fn enc_vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    let w0 = (0b110101u32 << 26) | (u32::from(op & 0x3FF) << 16) | u32::from(vdst);
    let w1 = (u32::from(src2 & 0x1FF) << 18)
        | (u32::from(src1 & 0x1FF) << 9)
        | u32::from(src0 & 0x1FF);
    [w0, w1]
}

fn enc_ds(
    op: u8,
    offset0: u8,
    offset1: u8,
    addr: u8,
    data0: u8,
    data1: u8,
    vdst: u8,
) -> [u32; 2] {
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

#[allow(clippy::too_many_arguments)]
fn enc_mubuf(
    op: u8,
    offset: u16,
    offen: bool,
    idxen: bool,
    vaddr: u8,
    vdata: u8,
    srsrc: u8,
    soffset: u8,
) -> [u32; 2] {
    let w0 = (0b111000u32 << 26)
        | (u32::from(op & 0x7F) << 18)
        | (u32::from(idxen) << 13)
        | (u32::from(offen) << 12)
        | u32::from(offset & 0xFFF);
    let w1 = (u32::from(soffset) << 24)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
    [w0, w1]
}

fn enc_smem_buffer_load(op: u8, sdata: u8, sbase_pair: u8, offset: u32) -> [u32; 2] {
    let w0 = (0b111101u32 << 26)
        | (u32::from(op & 0xFF) << 18)
        | (u32::from(sdata & 0x7F) << 6)
        | u32::from(sbase_pair & 0x3F);
    let w1 = (125u32 << 25) | (offset & 0x1F_FFFF);
    [w0, w1]
}

fn enc_exp(target: u8, en: u8, done: bool, vsrc: [u8; 4]) -> [u32; 2] {
    let w0 = (0b1111_10u32 << 26)
        | (u32::from(done) << 11)
        | (u32::from(target & 0x3F) << 4)
        | u32::from(en & 0xF);
    let w1 = u32::from(vsrc[0])
        | (u32::from(vsrc[1]) << 8)
        | (u32::from(vsrc[2]) << 16)
        | (u32::from(vsrc[3]) << 24);
    [w0, w1]
}

// ===========================================================================
// Part 1: Hardware WebGPU `subgroups` wave lowering mode (`WaveEmulMode`)
// ===========================================================================

#[test]
fn wave_emul_mode_defaults_to_workgroup_lds_and_parses_cli_values() {
    assert_eq!(WaveEmulMode::default(), WaveEmulMode::WorkgroupLds);
    assert_eq!(Rdna2TranslateOptions::default().wave_emul_mode, WaveEmulMode::WorkgroupLds);
    assert_eq!(WaveEmulMode::parse_str("lds").unwrap(), WaveEmulMode::WorkgroupLds);
    assert_eq!(
        WaveEmulMode::parse_str("workgroup-lds").unwrap(),
        WaveEmulMode::WorkgroupLds
    );
    assert_eq!(WaveEmulMode::parse_str("subgroups").unwrap(), WaveEmulMode::Subgroups);
    assert_eq!(WaveEmulMode::parse_str("SUBGROUP").unwrap(), WaveEmulMode::Subgroups);
    assert!(WaveEmulMode::parse_str("invalid").is_err());
    assert_eq!(WaveEmulMode::WorkgroupLds.wgsl_preamble(), "");
    assert_eq!(WaveEmulMode::Subgroups.wgsl_preamble(), WGSL_SUBGROUPS_ENABLE);
}

#[test]
fn subgroups_mode_lowers_readfirstlane_ballot_swizzle_bpermute_permlane_without_lds_or_barriers() {
    // RDNA2 compute shader exercising:
    //  * v_readfirstlane_b32 (VOP1 0x02)
    //  * v_cmp_gt_u32 -> s[10:11] ballot + s_bcnt1_i32_b64 + v_mbcnt_lo/hi + v_cndmask_b32
    //  * ds_swizzle_b32 XOR butterfly (0x101F -> XOR 4), quad reverse (0x801B -> XOR 3),
    //    and general bitmode broadcast (0x0000 -> lane & 32)
    //  * ds_bpermute_b32 (0xB3)
    //  * v_permlane16_b32 (0x377) & v_permlanex16_b32 (0x378)
    //  * buffer_store_dword to keep all computed VGPRs live
    let mut words = Vec::new();
    // v1 = v0 + 7
    words.push(enc_vop2(0x25, 1, 128 + 7, 0));
    // v_readfirstlane_b32 s8, v1
    words.push(enc_vop1(0x02, 8, 256 + 1));
    // v_cmp_gt_u32 s[10:11], v0, 3 (VOP3 0xC4: Greater u32)
    words.extend_from_slice(&enc_vop3(0xC4, 10, 256 + 0, 128 + 3, 0));
    // s_bcnt1_i32_b64 s12, s[10:11]
    words.push(enc_sop1(0x10, 12, 10));
    // v_mbcnt_lo_u32_b32 v2, s10, 0 ; v_mbcnt_hi_u32_b32 v2, s11, v2
    words.extend_from_slice(&enc_vop3(0x365, 2, 10, 128 + 0, 0));
    words.extend_from_slice(&enc_vop3(0x366, 2, 11, 256 + 2, 0));
    // v_cndmask_b32 v3, v1, v2, s10 (VOP3 0x100)
    words.extend_from_slice(&enc_vop3(0x100, 3, 256 + 1, 256 + 2, 10));
    // ds_swizzle_b32 v4, v3 offset:0x101F (XOR 4 -> subgroupShuffleXor)
    words.extend_from_slice(&enc_ds(0x35, 0x1F, 0x10, 3, 0, 0, 4));
    // ds_swizzle_b32 v5, v4 offset:0x801B (quad reverse -> subgroupShuffleXor 3)
    words.extend_from_slice(&enc_ds(0x35, 0x1B, 0x80, 4, 0, 0, 5));
    // ds_swizzle_b32 v6, v5 offset:0x0000 (broadcast lane 0/32 -> subgroupShuffle)
    words.extend_from_slice(&enc_ds(0x35, 0x00, 0x00, 5, 0, 0, 6));
    // ds_bpermute_b32 v7, v0, v6 (pull by byte_addr >> 2 -> subgroupShuffle)
    words.extend_from_slice(&enc_ds(0xB3, 0, 0, 0, 6, 0, 7));
    // v_permlane16_b32 v8, v7, s8, s12 (0x377 -> subgroupShuffle)
    words.extend_from_slice(&enc_vop3(0x377, 8, 256 + 7, 8, 12));
    // v_permlanex16_b32 v9, v8, s8, s12 (0x378 -> subgroupShuffle)
    words.extend_from_slice(&enc_vop3(0x378, 9, 256 + 8, 8, 12));
    // Combine s8 + s12 + v9 into v10 and store to buffer via V# s[4:7]
    words.push(enc_vop2(0x25, 10, 8, 9));
    words.push(enc_vop2(0x25, 10, 12, 10));
    words.extend_from_slice(&enc_mubuf(0x1C, 0, true, false, 0, 10, 1, 128));
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);

    // 1. Default WorkgroupLds mode: baseline WebGPU, uses pg6_wave_lds + workgroupBarrier().
    let lds_out = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Compute,
            wave_emul_mode: WaveEmulMode::WorkgroupLds,
            ..Default::default()
        },
    )
    .expect("WorkgroupLds translation must succeed");
    assert!(!lds_out.uses_subgroups);
    assert!(lds_out.naga_validated);
    assert!(
        !lds_out.wgsl.contains("enable subgroups;"),
        "WorkgroupLds must never emit `enable subgroups;`:\n{}",
        lds_out.wgsl
    );
    assert!(
        lds_out.wgsl.contains("pg6_wave_lds"),
        "WorkgroupLds should declare pg6_wave_lds:\n{}",
        lds_out.wgsl
    );
    assert!(
        lds_out.wgsl.contains("workgroupBarrier()"),
        "WorkgroupLds should emit workgroupBarrier():\n{}",
        lds_out.wgsl
    );
    let _ = validate_wgsl_with_caps(&lds_out.wgsl, BASELINE_WEBGPU_CAPABILITIES);

    // 2. Subgroups mode: `enable subgroups;`, ZERO pg6_wave_lds, ZERO workgroupBarrier()!
    let sg_out = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Compute,
            wave_emul_mode: WaveEmulMode::Subgroups,
            ..Default::default()
        },
    )
    .expect("Subgroups translation must succeed");
    assert!(sg_out.uses_subgroups);
    assert!(sg_out.naga_validated);
    assert_eq!(
        sg_out.wgsl.matches("enable subgroups;").count(),
        1,
        "Subgroups mode must emit exactly one `enable subgroups;`:\n{}",
        sg_out.wgsl
    );
    assert!(
        !sg_out.wgsl.contains("pg6_wave_lds"),
        "Subgroups mode must NOT allocate pg6_wave_lds for hardware-lowered ops:\n{}",
        sg_out.wgsl
    );
    assert!(
        !sg_out.wgsl.contains("workgroupBarrier()"),
        "Subgroups mode must emit ZERO workgroupBarrier() for hardware-lowered ops:\n{}",
        sg_out.wgsl
    );
    assert!(
        sg_out.wgsl.contains("subgroupBroadcastFirst("),
        "v_readfirstlane_b32 must lower to subgroupBroadcastFirst:\n{}",
        sg_out.wgsl
    );
    assert!(
        sg_out.wgsl.contains("subgroupBallot("),
        "v_cmp_* lane mask must lower to subgroupBallot:\n{}",
        sg_out.wgsl
    );
    assert!(
        sg_out.wgsl.contains("subgroupShuffleXor("),
        "ds_swizzle_b32 XOR/quad-reverse must lower to subgroupShuffleXor:\n{}",
        sg_out.wgsl
    );
    assert!(
        sg_out.wgsl.contains("subgroupShuffle("),
        "ds_bpermute_b32 / general swizzle / v_permlane must lower to subgroupShuffle:\n{}",
        sg_out.wgsl
    );
    assert!(
        sg_out.wgsl.contains("@builtin(subgroup_invocation_id)"),
        "expected @builtin(subgroup_invocation_id) entry-point arg:\n{}",
        sg_out.wgsl
    );
    assert!(
        sg_out.to_json().contains("\"uses_subgroups\": true"),
        "JSON summary should surface uses_subgroups:\n{}",
        sg_out.to_json()
    );

    // Must validate cleanly under Naga `Capabilities::SUBGROUP`...
    let sg_module = validate_wgsl_with_caps(
        &sg_out.wgsl,
        BASELINE_WEBGPU_CAPABILITIES | Capabilities::SUBGROUP,
    );
    // ...and MUST fail baseline validation without `SUBGROUP`!
    let mut baseline_validator =
        Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    assert!(
        baseline_validator.validate(&sg_module).is_err(),
        "subgroup module must require Capabilities::SUBGROUP"
    );
}

#[test]
fn subgroups_mode_falls_back_to_wave_lds_only_for_forward_scatter_ds_permute() {
    // Mix ds_swizzle_b32 (hardware subgroupShuffleXor) with ds_permute_b32 (forward scatter with
    // highest-lane-wins semantics -> falls back to pg6_wave_lds).
    let mut words = Vec::new();
    words.extend_from_slice(&enc_ds(0x35, 0x1F, 0x04, 0, 0, 0, 1)); // ds_swizzle XOR 1
    words.extend_from_slice(&enc_ds(0xB2, 0, 0, 0, 1, 0, 2)); // ds_permute_b32 (scatter)
    words.extend_from_slice(&enc_mubuf(0x1C, 0, true, false, 0, 2, 1, 128));
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let out = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Compute,
            wave_emul_mode: WaveEmulMode::Subgroups,
            ..Default::default()
        },
    )
    .expect("mixed subgroups + ds_permute fallback must succeed");

    assert!(out.uses_subgroups);
    assert!(out.wgsl.contains("enable subgroups;"), "WGSL:\n{}", out.wgsl);
    assert!(
        out.wgsl.contains("subgroupShuffleXor("),
        "ds_swizzle_b32 should still use hardware subgroupShuffleXor:\n{}",
        out.wgsl
    );
    assert!(
        out.wgsl.contains("pg6_wave_lds") && out.wgsl.contains("workgroupBarrier()"),
        "ds_permute_b32 should fall back cleanly to pg6_wave_lds:\n{}",
        out.wgsl
    );
    let _ = validate_wgsl_with_caps(
        &out.wgsl,
        BASELINE_WEBGPU_CAPABILITIES | Capabilities::SUBGROUP,
    );
}

#[test]
fn subgroups_mode_vertex_stays_baseline_while_fragment_emits_subgroups() {
    // 1. Vertex shader with v_readfirstlane_b32: WebGPU subgroups excludes vertex stage!
    let mut vs_words = Vec::new();
    vs_words.push(enc_vop1(0x02, 4, 256 + 0)); // v_readfirstlane_b32 s4, v0
    vs_words.push(enc_vop2(0x25, 1, 4, 0)); // v_add_nc_u32 v1, s4, v0
    vs_words.extend_from_slice(&enc_exp(12, 0xF, true, [1, 1, 1, 1])); // EXP POS0
    vs_words.push(enc_sopp(0x01, 0));

    let vs_out = translate_rdna2_to_wgsl(
        &words_to_bytes(&vs_words),
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Vertex,
            wave_emul_mode: WaveEmulMode::Subgroups,
            ..Default::default()
        },
    )
    .expect("vertex translation must stay baseline-valid under WaveEmulMode::Subgroups");
    assert!(!vs_out.uses_subgroups);
    assert!(!vs_out.wgsl.contains("enable subgroups;"));
    let _ = validate_wgsl_with_caps(&vs_out.wgsl, BASELINE_WEBGPU_CAPABILITIES);

    // 2. Fragment shader with v_readfirstlane_b32 + ds_swizzle_b32: WebGPU subgroups supports fragment!
    let mut fs_words = Vec::new();
    fs_words.push(enc_vop1(0x02, 4, 256 + 0)); // v_readfirstlane_b32 s4, v0
    fs_words.extend_from_slice(&enc_ds(0x35, 0x1F, 0x04, 0, 0, 0, 1)); // ds_swizzle XOR 1
    fs_words.push(enc_vop2(0x25, 2, 4, 1));
    fs_words.extend_from_slice(&enc_exp(0, 0xF, true, [2, 2, 2, 2])); // EXP MRT0
    fs_words.push(enc_sopp(0x01, 0));

    let fs_out = translate_rdna2_to_wgsl(
        &words_to_bytes(&fs_words),
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Pixel,
            wave_emul_mode: WaveEmulMode::Subgroups,
            ..Default::default()
        },
    )
    .expect("fragment translation with Subgroups must succeed");
    assert!(fs_out.uses_subgroups);
    assert!(fs_out.wgsl.contains("enable subgroups;"), "WGSL:\n{}", fs_out.wgsl);
    assert!(fs_out.wgsl.contains("subgroupBroadcastFirst("), "WGSL:\n{}", fs_out.wgsl);
    assert!(fs_out.wgsl.contains("subgroupShuffleXor("), "WGSL:\n{}", fs_out.wgsl);
    let _ = validate_wgsl_with_caps(
        &fs_out.wgsl,
        BASELINE_WEBGPU_CAPABILITIES | Capabilities::SUBGROUP,
    );
}

#[test]
fn wave_emul_reductions_and_exclusive_add_lower_cleanly_in_both_modes() {
    // 1. Standalone WGSL helper library comparison via `rdna_wave_emul_wgsl`.
    let cfg = RdnaComputeWaveEmulConfig::new(WaveSize::Wave64, 64).with_prefix("s78_");
    let lds_lib = rdna_wave_emul_wgsl(WaveEmulMode::WorkgroupLds, &cfg);
    let sg_lib = rdna_wave_emul_wgsl(WaveEmulMode::Subgroups, &cfg);

    assert!(!lds_lib.contains("enable subgroups;"));
    assert!(lds_lib.contains("workgroupBarrier()"));
    assert!(lds_lib.contains("fn s78_wave_reduce_add_u32("));
    assert!(lds_lib.contains("fn s78_wave_exclusive_add_u32("));
    let _ = validate_wgsl_with_caps(&lds_lib, Capabilities::empty());

    assert!(sg_lib.starts_with(WGSL_SUBGROUPS_ENABLE));
    assert!(!sg_lib.contains("workgroupBarrier()"));
    for builtin in [
        "subgroupBroadcastFirst(",
        "subgroupBallot(",
        "subgroupShuffle(",
        "subgroupShuffleXor(",
        "subgroupAdd(",
        "subgroupMin(",
        "subgroupMax(",
        "subgroupAnd(",
        "subgroupOr(",
        "subgroupXor(",
        "subgroupExclusiveAdd(",
    ] {
        assert!(sg_lib.contains(builtin), "missing {builtin} in:\n{sg_lib}");
    }
    let _ = validate_wgsl_with_caps(&sg_lib, Capabilities::SUBGROUP);

    // 2. Naga IR `lower_wave_ops` driven via `WaveLoweringOptions::from_emul_mode`.
    let probe = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32, 640>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
    let v = li + 1u;
    let b = subgroupBallot((v & 1u) == 0u);
    let o = li * 10u;
    out[o + 0u] = subgroupBroadcastFirst(v);
    out[o + 1u] = b.x ^ b.y;
    out[o + 2u] = subgroupShuffleXor(v, 1u);
    out[o + 3u] = subgroupAdd(v);
    out[o + 4u] = subgroupMin(v);
    out[o + 5u] = subgroupMax(v);
    out[o + 6u] = subgroupAnd(v);
    out[o + 7u] = subgroupOr(v);
    out[o + 8u] = subgroupXor(v);
    out[o + 9u] = subgroupExclusiveAdd(v);
}
"#;

    let mut sg_mod = validate_wgsl_with_caps(probe, Capabilities::SUBGROUP);
    let sg_report = lower_wave_ops(&mut sg_mod, &WaveLoweringOptions::from_emul_mode(WaveEmulMode::Subgroups));
    assert!(sg_report.applied, "{:?}", sg_report.blockers);
    assert!(sg_report.requires_subgroups_feature);
    assert_eq!(sg_report.wgsl_preamble(), WGSL_SUBGROUPS_ENABLE);
    let sg_wgsl = validate_and_emit_wgsl(&sg_mod, Capabilities::SUBGROUP);
    assert!(!sg_wgsl.contains("workgroupBarrier()"));
    assert!(sg_wgsl.contains("subgroupAdd(") && sg_wgsl.contains("subgroupExclusiveAdd("));

    let mut lds_mod = validate_wgsl_with_caps(probe, Capabilities::SUBGROUP);
    let lds_report = lower_wave_ops(&mut lds_mod, &WaveLoweringOptions::from_emul_mode(WaveEmulMode::WorkgroupLds));
    assert!(lds_report.applied, "{:?}", lds_report.blockers);
    assert!(!lds_report.requires_subgroups_feature);
    assert_eq!(lds_report.wgsl_preamble(), "");
    let lds_wgsl = validate_and_emit_wgsl(&lds_mod, Capabilities::empty());
    assert!(lds_wgsl.contains("workgroupBarrier()"));
    assert!(!lds_wgsl.contains("subgroupAdd("));
}

// ===========================================================================
// Part 2: Deeper `ir_opt.rs` passes
// ===========================================================================

#[test]
fn dead_branch_folding_splices_const_if_and_prunes_empty_pure_if() {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> out_buf: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;

    // 1. Constant true `if`: splices accept, drops reject (`9999u`).
    if (true) {
        out_buf[0] = x + 1u;
    } else {
        out_buf[0] = 9999u;
    }

    // 2. Constant false `if`: splices reject, drops accept (`8888u`).
    if (false) {
        out_buf[1] = 8888u;
    } else {
        out_buf[1] = x + 2u;
    }

    // 3. Forwarded constant local feeding `if`: folded on subsequent fixpoint iteration!
    var flag: bool = true;
    if (flag) {
        out_buf[2] = x + 3u;
    } else {
        out_buf[2] = 7777u;
    }

    // 4. Empty `if` with pure condition: pruned completely!
    if ((x & 7u) == 3u) {
    } else {
    }

    // 5. Dynamic non-empty `if`: MUST be preserved!
    if (x > 10u) {
        out_buf[3] = x + 4u;
    }
}
"#;

    let mut module = parse_baseline_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module, BASELINE_WEBGPU_CAPABILITIES);

    assert!(
        stats.dead_branches_folded >= 4,
        "expected >= 4 folded/pruned branches, got {stats:?}\nWGSL:\n{wgsl}"
    );
    assert!(!wgsl.contains("9999u"), "dead else branch must be dropped:\n{wgsl}");
    assert!(!wgsl.contains("8888u"), "dead then branch must be dropped:\n{wgsl}");
    assert!(!wgsl.contains("7777u"), "dead forwarded-flag else must be dropped:\n{wgsl}");
    assert!(!wgsl.contains("var flag"), "forwarded constant flag local should prune:\n{wgsl}");
    assert_eq!(
        wgsl.matches("if ").count(),
        1,
        "only the dynamic `if (x > 10u)` should survive:\n{wgsl}"
    );
    assert!(wgsl.contains("out_buf[3] ="), "live dynamic branch must stay:\n{wgsl}");
}

#[test]
fn unreachable_statement_trimming_drops_code_after_return_break_continue_and_kill() {
    // 1. Compute shader: unreachable tails after `return`, `break`, and `continue`.
    let compute_src = r#"
@group(0) @binding(0) var<storage, read_write> out_buf: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    var acc: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= 4u) {
            break;
            out_buf[1] = 1111u;
        }
        i = i + 1u;
        if ((i & 1u) == 0u) {
            continue;
            out_buf[2] = 2222u;
        }
        acc = acc + x;
    }
    out_buf[0] = acc;
    return;
    out_buf[3] = 3333u;
}
"#;
    let mut cs_mod = parse_baseline_wgsl(compute_src);
    let cs_stats = optimize_module(&mut cs_mod);
    let cs_wgsl = validate_and_emit_wgsl(&cs_mod, BASELINE_WEBGPU_CAPABILITIES);
    assert!(
        cs_stats.dead_branches_folded >= 3,
        "expected unreachable tails trimmed after break/continue/return, got {cs_stats:?}\n{cs_wgsl}"
    );
    assert!(!cs_wgsl.contains("1111u"), "code after break must be trimmed:\n{cs_wgsl}");
    assert!(!cs_wgsl.contains("2222u"), "code after continue must be trimmed:\n{cs_wgsl}");
    assert!(!cs_wgsl.contains("3333u"), "code after return must be trimmed:\n{cs_wgsl}");

    // 2. Fragment shader: unreachable tail after `discard;` inside branch (without dropping
    //    required `return` of non-void function) AND in void fragment entry point!
    let frag_src = r#"
@group(0) @binding(0) var<storage, read_write> dbg: array<u32>;

@fragment
fn fs_nonvoid(@location(0) x: f32) -> @location(0) vec4<f32> {
    if (x < 0.0) {
        discard;
        dbg[0] = 4444u;
    }
    if (x > 10.0) {
        discard;
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        return vec4<f32>(x, 0.0, 0.0, 1.0);
    }
}
"#;
    let mut fs_mod = parse_baseline_wgsl(frag_src);
    let fs_stats = optimize_module(&mut fs_mod);
    let fs_wgsl = validate_and_emit_wgsl(&fs_mod, BASELINE_WEBGPU_CAPABILITIES);
    assert!(
        fs_stats.dead_branches_folded >= 1,
        "expected unreachable tail after discard trimmed, got {fs_stats:?}\n{fs_wgsl}"
    );
    assert!(!fs_wgsl.contains("4444u"), "code after discard in branch must be trimmed:\n{fs_wgsl}");
    assert!(fs_wgsl.contains("discard;"), "discard statement itself must stay:\n{fs_wgsl}");

    // 3. Void fragment shader: `discard;` at top level trims trailing statements.
    let void_frag_src = r#"
@group(0) @binding(0) var<storage, read_write> dbg: array<u32>;

@fragment
fn fs_void() {
    discard;
    dbg[0] = 5555u;
}
"#;
    let mut vf_mod = parse_baseline_wgsl(void_frag_src);
    let vf_stats = optimize_module(&mut vf_mod);
    let vf_wgsl = validate_and_emit_wgsl(&vf_mod, BASELINE_WEBGPU_CAPABILITIES);
    assert!(vf_stats.dead_branches_folded >= 1);
    assert!(!vf_wgsl.contains("5555u"), "code after discard in void fs must be trimmed:\n{vf_wgsl}");
}

#[test]
fn pure_integer_zero_and_negated_cmp_identities_preserve_float_nan_and_div_zero_safety() {
    let src = r#"
@group(0) @binding(0) var<storage, read> in_u: array<u32>;
@group(0) @binding(1) var<storage, read> in_f: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_u: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_f: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = in_u[x];
    let si = i32(x);
    let fx = in_f[x];
    let fy = in_f[x + 1u];

    // Pure scalar integer zero identities (`x * 0`, `0 * x`, `x & 0`, `0 & x`).
    let z0 = (x + 5u) * 0u;
    let z1 = 0u * (x + 6u);
    let z2 = (x + 7u) & 0u;
    let z3 = 0u & (x + 8u);
    let z4 = u32((si + 9) * 0) + u32(0 & (si + 10));

    // Negated integer & bool comparisons (`!(a == b)` -> `a != b`, `!(a != b)` -> `a == b`,
    // `!(a < b)` -> `a >= b`, etc.).
    let c0 = !(x == y);
    let c1 = !(x != y);
    let c2 = !(x < y);
    let c3 = !(c0 == c1);

    // Div-by-zero WGSL const-eval guard: `(x * 0u)` feeding divisor MUST NOT fold to `10u / 0u`!
    let denom = x * 0u;
    let safe_div = 10u / denom;

    // IEEE-754 float safety: `fx * 0.0` MUST NOT fold to `0.0` (NaN / -0.0!), and
    // `!(fx < fy)` / `!(fx == fy)` MUST NOT rewrite float comparisons!
    let fz = fx * 0.0;
    let f_not_lt = !(fx < fy);
    let f_not_eq = !(fx == fy);

    out_u[0] = z0 + z1 + z2 + z3 + z4 + u32(c0) + u32(c1) + u32(c2) + u32(c3) + safe_div;
    out_f[0] = fz + select(0.0, 1.0, f_not_lt) + select(0.0, 2.0, f_not_eq);
}
"#;

    let mut module = parse_baseline_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module, BASELINE_WEBGPU_CAPABILITIES);

    assert!(
        stats.identities_simplified >= 9,
        "expected >= 9 integer zero & negated-cmp identities simplified, got {stats:?}\nWGSL:\n{wgsl}"
    );
    // `z0..z4` folded to 0 and then folded together via constant folding!
    assert!(!wgsl.contains("+ 5u"), "dead `(x + 5u) * 0u` should fold away:\n{wgsl}");
    assert!(!wgsl.contains("+ 6u"), "dead `0u * (x + 6u)` should fold away:\n{wgsl}");
    assert!(!wgsl.contains("+ 7u"), "dead `(x + 7u) & 0u` should fold away:\n{wgsl}");
    assert!(!wgsl.contains("+ 8u"), "dead `0u & (x + 8u)` should fold away:\n{wgsl}");
    // Integer negated comparisons became `!=`, `==`, `>=` cleanly!
    assert!(wgsl.contains("x != y"), "!(x == y) should simplify to x != y:\n{wgsl}");
    assert!(wgsl.contains("x >= y"), "!(x < y) should simplify to x >= y:\n{wgsl}");
    // Div-by-zero const-eval trap was prevented!
    assert!(!wgsl.contains("10u / 0u"), "must never create const 10u / 0u:\n{wgsl}");
    // Float `fx * 0.0` and `!(fx < fy)` / `!(fx == fy)` MUST stay intact for NaN / -0.0!
    assert!(wgsl.contains("fx * 0.0") || wgsl.contains("fx * 0f"), "float * 0.0 must stay:\n{wgsl}");
    assert!(wgsl.contains("!((") || wgsl.contains("!(fx"), "float negated cmp must stay:\n{wgsl}");
}

#[test]
fn readonly_load_cse_deduplicates_uniform_and_readonly_storage_and_never_mutable() {
    let src = r#"
struct Params {
    scale: vec4<f32>,
    counts: array<vec4<u32>, 4>,
}

@group(0) @binding(0) var<uniform> ubo: Params;
@group(0) @binding(1) var<storage, read> ro_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> rw_buf: array<u32>;
var<workgroup> wg_arr: array<u32, 64>;
var<private> priv_val: u32;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let idx = gid.x;

    // 1. Read-only `var<uniform>` loads (struct member + constant array index + dynamic index):
    //    MUST CSE even across intervening writable storage writes and workgroup barriers!
    let u0 = ubo.scale.x;
    let u1 = ubo.counts[2].y;
    let r0 = ro_buf[idx];
    let r1 = ro_buf[3];

    // Mutable writes + barrier in between:
    rw_buf[idx] = r0;
    wg_arr[li] = r1;
    priv_val = r0 + 1u;
    workgroupBarrier();

    // Duplicate loads from immutable `var<uniform>` and `var<storage, read>`:
    let u0_dup = ubo.scale.x;
    let u1_dup = ubo.counts[2].y;
    let r0_dup = ro_buf[idx];
    let r1_dup = ro_buf[3];

    // Dominating read-only load reused inside an `if` branch:
    var branch_acc: u32 = 0u;
    if (idx > 0u) {
        let r0_in_if = ro_buf[idx];
        let u1_in_if = ubo.counts[2].y;
        // Sibling-only read-only load (MUST NOT be reused in `else`!).
        let sib = ro_buf[7];
        branch_acc = r0_in_if + u1_in_if + sib;
    } else {
        let sib_else = ro_buf[7];
        branch_acc = sib_else * 2u;
    }

    // 2. Mutable globals (`var<storage, read_write>`, `var<workgroup>`, `var<private>`):
    //    MUST NEVER be CSE'd!
    let rw_a = rw_buf[0];
    rw_buf[0] = rw_a + 1u;
    let rw_b = rw_buf[0];

    let wg_a = wg_arr[0];
    wg_arr[0] = wg_a + 1u;
    let wg_b = wg_arr[0];

    let pv_a = priv_val;
    priv_val = pv_a + 1u;
    let pv_b = priv_val;

    rw_buf[idx] = u32(u0 + u0_dup) + u1 + u1_dup + r0_dup + r1_dup + branch_acc + rw_a + rw_b + wg_a + wg_b + pv_a + pv_b;
}
"#;

    let mut module = parse_baseline_wgsl(src);
    let stats = optimize_module(&mut module);
    let wgsl = validate_and_emit_wgsl(&module, BASELINE_WEBGPU_CAPABILITIES);

    assert!(
        stats.readonly_loads_cse >= 6,
        "expected >= 6 read-only uniform/storage Load CSE hits (4 straight-line + 2 dominating into if), got {stats:?}\nWGSL:\n{wgsl}"
    );
    // `ubo.scale.x`, `ubo.counts[2].y`, `ro_buf[idx]`, and `ro_buf[3]` should each be loaded ONLY ONCE!
    assert_eq!(
        wgsl.matches("ubo.scale.x").count(),
        1,
        "ubo.scale.x must be CSE'd to 1 load:\n{wgsl}"
    );
    assert_eq!(
        wgsl.matches("ubo.counts[2].y").count(),
        1,
        "ubo.counts[2].y must be CSE'd to 1 load:\n{wgsl}"
    );
    assert_eq!(
        wgsl.matches("ro_buf[idx]").count(),
        1,
        "ro_buf[idx] must be CSE'd to 1 load:\n{wgsl}"
    );
    assert_eq!(
        wgsl.matches("ro_buf[3]").count(),
        1,
        "ro_buf[3] must be CSE'd to 1 load:\n{wgsl}"
    );
    // Sibling branch `ro_buf[7]` MUST stay separate in both branches (2 loads)!
    assert_eq!(
        wgsl.matches("ro_buf[7]").count(),
        2,
        "sibling if/else loads of ro_buf[7] must NOT merge across branches:\n{wgsl}"
    );
    // Mutable globals MUST NEVER be CSE'd (each has 2 reads + 1 write = 3 occurrences)!
    assert!(
        wgsl.matches("rw_buf[0]").count() >= 3,
        "read_write storage loads must NEVER be CSE'd:\n{wgsl}"
    );
    assert!(
        wgsl.matches("wg_arr[0]").count() >= 3,
        "workgroup loads must NEVER be CSE'd:\n{wgsl}"
    );
    assert!(
        wgsl.matches("priv_val").count() >= 4,
        "private loads must NEVER be CSE'd:\n{wgsl}"
    );
}

#[test]
fn readonly_load_cse_works_on_raw_naga_ir_with_distinct_pointer_chains() {
    // Construct raw Naga IR where two loads read `ro_buf[0]` via two distinct pointer chains
    // (`Access { base: GlobalVariable(ro), index: Literal(0u) }` vs `AccessIndex { base: GlobalVariable(ro), index: 0 }`).
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
    let g_ro = module.global_variables.append(
        GlobalVariable {
            name: Some("ro_data".to_string()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
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
    let g_out = module.global_variables.append(
        GlobalVariable {
            name: Some("out_data".to_string()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: arr_u32,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );

    let mut ep = Function {
        name: Some("main".to_string()),
        arguments: vec![FunctionArgument {
            name: Some("gid".to_string()),
            ty: module.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Vector {
                        size: naga::VectorSize::Tri,
                        scalar: Scalar::U32,
                    },
                },
                Span::UNDEFINED,
            ),
            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
        }],
        ..Function::default()
    };

    let ro1 = ep
        .expressions
        .append(Expression::GlobalVariable(g_ro), Span::UNDEFINED);
    let idx0 = ep
        .expressions
        .append(Expression::Literal(Literal::U32(0)), Span::UNDEFINED);
    let ptr1 = ep.expressions.append(
        Expression::Access {
            base: ro1,
            index: idx0,
        },
        Span::UNDEFINED,
    );
    let load1 = ep
        .expressions
        .append(Expression::Load { pointer: ptr1 }, Span::UNDEFINED);

    let ro2 = ep
        .expressions
        .append(Expression::GlobalVariable(g_ro), Span::UNDEFINED);
    let ptr2 = ep.expressions.append(
        Expression::AccessIndex {
            base: ro2,
            index: 0,
        },
        Span::UNDEFINED,
    );
    let load2 = ep
        .expressions
        .append(Expression::Load { pointer: ptr2 }, Span::UNDEFINED);

    let sum = ep.expressions.append(
        Expression::Binary {
            op: BinaryOperator::Add,
            left: load1,
            right: load2,
        },
        Span::UNDEFINED,
    );
    let out_g = ep
        .expressions
        .append(Expression::GlobalVariable(g_out), Span::UNDEFINED);
    let out_ptr = ep.expressions.append(
        Expression::AccessIndex {
            base: out_g,
            index: 0,
        },
        Span::UNDEFINED,
    );
    ep.body.push(
        Statement::Emit(Range::new_from_bounds(ptr1, load1)),
        Span::UNDEFINED,
    );
    ep.body.push(
        Statement::Emit(Range::new_from_bounds(ptr2, sum)),
        Span::UNDEFINED,
    );
    ep.body.push(
        Statement::Emit(Range::new_from_bounds(out_ptr, out_ptr)),
        Span::UNDEFINED,
    );
    ep.body.push(
        Statement::Store {
            pointer: out_ptr,
            value: sum,
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
    let wgsl = validate_and_emit_wgsl(&module, BASELINE_WEBGPU_CAPABILITIES);
    assert_eq!(stats.readonly_loads_cse, 1, "expected 1 read-only load CSE hit: {stats:?}");
    assert_eq!(
        wgsl.matches("ro_data[").count(),
        1,
        "duplicate read-only load across Access/AccessIndex chains must collapse to 1:\n{wgsl}"
    );
}

#[test]
fn end_to_end_rdna2_cbv_readonly_load_cse_and_subgroups_json_metrics() {
    // RDNA2 compute shader that loads the same CBV dword twice via `s_buffer_load_dword` and
    // broadcasts across the wave via `v_readfirstlane_b32` + `ds_swizzle_b32`.
    let mut words = Vec::new();
    // s_buffer_load_dword s10, s[0:3], 0x0 (cb0[0].x)
    words.extend_from_slice(&enc_smem_buffer_load(0x08, 10, 0, 0));
    // s_buffer_load_dword s11, s[0:3], 0x0 (duplicate load of cb0[0].x!)
    words.extend_from_slice(&enc_smem_buffer_load(0x08, 11, 0, 0));
    // v1 = v0 + s10 ; v2 = v1 + s11
    words.push(enc_vop2(0x25, 1, 10, 0));
    words.push(enc_vop2(0x25, 2, 11, 1));
    // ds_swizzle_b32 v3, v2 offset:0x041F (XOR 1)
    words.extend_from_slice(&enc_ds(0x35, 0x1F, 0x04, 2, 0, 0, 3));
    // v_readfirstlane_b32 s12, v3
    words.push(enc_vop1(0x02, 12, 256 + 3));
    // v4 = v3 + s12 ; buffer_store_dword v4
    words.push(enc_vop2(0x25, 4, 12, 3));
    words.extend_from_slice(&enc_mubuf(0x1C, 0, true, false, 0, 4, 1, 128));
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let out = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Compute,
            wave_emul_mode: WaveEmulMode::Subgroups,
            ..Default::default()
        },
    )
    .expect("RDNA2 Subgroups translation must succeed");

    assert!(out.uses_subgroups);
    assert!(out.wgsl.contains("enable subgroups;"));
    assert!(!out.wgsl.contains("pg6_wave_lds"));
    assert!(!out.wgsl.contains("workgroupBarrier()"));
    assert!(
        out.ir_opt.readonly_loads_cse >= 1,
        "duplicate CBV uniform load should be CSE'd, got {:?}\nWGSL:\n{}",
        out.ir_opt,
        out.wgsl
    );
    let json = out.to_json();
    assert!(json.contains("\"uses_subgroups\": true"), "JSON:\n{json}");
    assert!(json.contains("\"dead_branches_folded\""), "JSON:\n{json}");
    assert!(json.contains("\"readonly_loads_cse\""), "JSON:\n{json}");
}
