//! Synthetic end-to-end tests for Xbox Series `FSSM` `s_swappc_b64` vertex fetch synthesis,
//! `v_nop_e64` (`VOP3 op=0x180`) 3-dword literal consumption, and optional `FSSM` inter-stage
//! `vs_outputs` / `ps_inputs` semantic reflection metadata.

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use naga::valid::{Capabilities, ValidationFlags, Validator};
use playgta6_shaders::convert::{ShaderStage, fnv1a_64, format_shader_hash};
use playgta6_shaders::rdna2::{
    Rdna2Detail, Rdna2Encoding, XBOX_SHADER_SHORT_PREAMBLE_LEN, analyze_program, decode_program,
    rdna2_code_start_offset,
};
use playgta6_shaders::rdna2_pack::{Rdna2PackOptions, pack_rdna2_manifest};
use playgta6_shaders::rdna2_to_wgsl::{
    FssmVertexInput, Rdna2StageOverride, Rdna2TranslateOptions, WgslStageParamInfo,
    parse_fssm_stage_metadata, translate_rdna2_to_wgsl,
};
use playgta6_shaders::reflect::SidecarBuildOptions;

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

fn assert_naga_valid(wgsl: &str) {
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|err| panic!("naga WGSL parse failed:\n{err}\n\nWGSL:\n{wgsl}"));
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    validator
        .validate(&module)
        .unwrap_or_else(|err| panic!("naga WGSL validate failed:\n{err}\n\nWGSL:\n{wgsl}"));
}

/// Build a synthetic Xbox Series `FSSM` v0x101 metadata blob with VS inputs/outputs and/or PS inputs.
fn build_synthetic_fssm(
    vs_inputs: &[(u8, u8, u8, u8)],  // (semantic_reg, mask, base_vgpr, max_comp)
    vs_outputs: &[(u8, u8)],         // (semantic_reg, param_idx)
    ps_inputs: &[(u8, u8, u8)],      // (semantic_reg, mask, attr_idx)
) -> Vec<u8> {
    const ABSENT: u64 = 0xDEAD_BEEF_DEAD_BEEF;
    let mut pl = vec![0u8; 0x600];
    pl[0..8].copy_from_slice(&6u64.to_le_bytes());
    for slot_off in [0x08usize, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
        pl[slot_off..slot_off + 8].copy_from_slice(&ABSENT.to_le_bytes());
    }

    let mut cursor: usize = 0xb0;
    if !vs_inputs.is_empty() || !vs_outputs.is_empty() {
        let vs_rec = cursor;
        pl[0x28..0x30].copy_from_slice(&(vs_rec as u64).to_le_bytes());
        let ext = vs_rec + 0x54;
        for (i, &(sem, mask, vgpr, max_c)) in vs_inputs.iter().enumerate() {
            let off = ext + 0x24 + i * 5;
            pl[off..off + 5].copy_from_slice(&[6, sem, mask, vgpr, max_c]);
        }
        pl[ext + 0xc4..ext + 0xc8].copy_from_slice(&(vs_inputs.len() as u32).to_le_bytes());
        for (i, &(sem, param)) in vs_outputs.iter().enumerate() {
            let off = ext + 0xc8 + i * 6;
            pl[off..off + 6].copy_from_slice(&[6, sem, param, 0, 0, 0]);
        }
        pl[ext + 0x188..ext + 0x18c].copy_from_slice(&(vs_outputs.len() as u32).to_le_bytes());
        cursor = ext + 0x1a0;
    }

    if !ps_inputs.is_empty() {
        let ps_rec = cursor;
        pl[0x30..0x38].copy_from_slice(&(ps_rec as u64).to_le_bytes());
        let ext = ps_rec + 0x54;
        for (i, &(sem, mask, attr)) in ps_inputs.iter().enumerate() {
            let off = ext + 0x24 + i * 8;
            pl[off..off + 8].copy_from_slice(&[6, sem, mask, attr, 0, 0, sem, 2]);
        }
        pl[ext + 0x124..ext + 0x128].copy_from_slice(&(ps_inputs.len() as u32).to_le_bytes());
    }

    let mut out = Vec::with_capacity(16 + pl.len());
    out.extend_from_slice(b"FSSM");
    out.extend_from_slice(&0x101u32.to_le_bytes());
    out.extend_from_slice(&(pl.len() as u64).to_le_bytes());
    out.extend_from_slice(&pl);
    out
}

#[test]
fn v_nop_e64_with_src2_literal_consumes_third_dword_without_unknown_enc() {
    // Two standalone 12-byte `v_nop_e64` metadata carrier records (without `s_inst_prefetch 0x3`
    // so `rdna2_code_start_offset` returns 0 and the decoder must decode both records directly),
    // where the 3rd literal dword of record 1 has high bits `0b110001` (`unknown_enc_0x31` if not
    // consumed!) and the 3rd literal dword of record 0 has `0x2327_4cd5`.
    let words = vec![
        0xd580_dd9e, // v_nop_e64 (VOP3 op=0x180)
        0x03fe_bd64, // src2 = 0xFF (literal)
        0x2327_4cd5, // 3rd dword literal payload
        0xd580_5b0c, // v_nop_e64 (VOP3 op=0x180)
        0x03ff_f3b7, // src2 = 0xFF (literal)
        0xc7c6_ac36, // 3rd dword literal payload (would decode as unknown_enc_0x31 if unconsumed!)
        enc_sopp(0x01, 0), // s_endpgm at 0x18
    ];
    let bytes = words_to_bytes(&words);
    assert_eq!(rdna2_code_start_offset(&bytes), 0);

    let insts = decode_program(&bytes);
    assert_eq!(insts.len(), 3, "decoded: {insts:?}");
    assert_eq!(insts[0].offset, 0x00);
    assert_eq!(insts[0].size, 12);
    assert_eq!(insts[0].encoding, Rdna2Encoding::Vop3);
    assert_eq!(insts[0].mnemonic, "v_nop");
    assert!(matches!(
        insts[0].detail,
        Rdna2Detail::Vop3 {
            op: 0x180,
            literal: Some(0x2327_4cd5),
            ..
        }
    ));

    assert_eq!(insts[1].offset, 0x0c);
    assert_eq!(insts[1].size, 12);
    assert_eq!(insts[1].mnemonic, "v_nop");
    assert!(matches!(
        insts[1].detail,
        Rdna2Detail::Vop3 {
            op: 0x180,
            literal: Some(0xc7c6_ac36),
            ..
        }
    ));

    assert_eq!(insts[2].offset, 0x18);
    assert_eq!(insts[2].mnemonic, "s_endpgm");

    let summary = analyze_program(&bytes);
    assert_eq!(summary.unknown_instructions, 0);
    assert_eq!(
        summary.max_vgpr, None,
        "v_nop_e64 metadata carrier must not pollute max_vgpr"
    );
    assert_eq!(
        summary.max_sgpr, None,
        "v_nop_e64 metadata carrier must not pollute max_sgpr"
    );

    let out = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Vertex,
            ..Rdna2TranslateOptions::default()
        },
    )
    .expect("v_nop_e64 shader must translate cleanly");
    assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
    assert!(!out.wgsl.contains("unknown_enc"));
    assert_naga_valid(&out.wgsl);
}

#[test]
fn fssm_swappc_vs_synthesizes_vertex_inputs_and_loads_vgprs_both_with_and_without_meta() {
    // Synthetic unlinked FSSM Blit VS matching the exact instruction structure of `VS_Blit`:
    //   +0x00..+0x17: 2-record 24B Xbox v_nop_e64 preamble
    //   +0x18: s_swappc_b64 s[4:5], s[4:5]  (fetch shader populates v4..v6, v8..v11, v12..v13)
    //   +0x1c: s_inst_prefetch 0x3
    //   +0x20: v_mov_b32 v7, 1.0
    //   +0x24: v_mov_b32 v2, 0.0
    //   +0x28: s_waitcnt vmcnt(0)
    //   +0x2c: exp pos0, v4, v5, v6, v7 done
    //   +0x34: exp param0, v8, v9, v10, v11
    //   +0x3c: exp param1, v12, v13, v2, v7
    //   +0x44: s_endpgm
    let mut words = vec![
        0xd580_dd9e,
        0x03fe_bd64,
        0x2327_4cd5,
        0xd580_5b0c,
        0x03ff_f3b7,
        0xc7c6_ac36,
        enc_sop1(0x21, 4, 4), // s_swappc_b64 s[4:5], s[4:5]
        0xbfa0_0003,          // s_inst_prefetch 0x3
        enc_vop1(0x01, 7, 242), // v_mov_b32 v7, 1.0
        enc_vop1(0x01, 2, 128), // v_mov_b32 v2, 0
        enc_sopp(0x0C, 0x0F70), // s_waitcnt vmcnt(0)
    ];
    words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [4, 5, 6, 7]));
    words.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [8, 9, 10, 11]));
    words.extend_from_slice(&enc_exp(33, 0xF, false, false, false, [12, 13, 2, 7]));
    words.push(enc_sopp(0x01, 0)); // s_endpgm
    let bytes = words_to_bytes(&words);

    // 24-byte preamble followed by s_swappc_b64 + s_inst_prefetch 0x3 is cleanly skipped.
    assert_eq!(
        rdna2_code_start_offset(&bytes),
        XBOX_SHADER_SHORT_PREAMBLE_LEN
    );

    // Case A: Raw RDNA2 translation WITHOUT `.meta` (fallback heuristic).
    let out_raw = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Vertex,
            ..Rdna2TranslateOptions::default()
        },
    )
    .expect("raw s_swappc_b64 VS must translate");
    assert!(out_raw.naga_validated);
    assert_eq!(out_raw.unsupported_op_count, 0, "WGSL:\n{}", out_raw.wgsl);
    assert_eq!(out_raw.binding_summary.vertex_input_mode, "attributes");
    assert_eq!(out_raw.vertex_inputs.len(), 3, "inputs: {:?}", out_raw.vertex_inputs);
    assert_eq!(out_raw.vertex_inputs[0].location, 0);
    assert_eq!(out_raw.vertex_inputs[0].vb_slot, 0);
    assert_eq!(out_raw.vertex_inputs[0].byte_offset, 0);
    assert_eq!(out_raw.vertex_inputs[0].components, 3);
    assert_eq!(out_raw.vertex_inputs[0].component_type, "f32");
    assert_eq!(out_raw.vertex_inputs[1].location, 1);
    assert_eq!(out_raw.vertex_inputs[1].byte_offset, 12);
    assert_eq!(out_raw.vertex_inputs[1].components, 4);
    assert_eq!(out_raw.vertex_inputs[2].location, 2);
    assert_eq!(out_raw.vertex_inputs[2].byte_offset, 16);
    assert_eq!(out_raw.vertex_inputs[2].components, 2);
    assert!(out_raw.wgsl.contains("@location(0)"));
    assert!(out_raw.wgsl.contains("@location(1)"));
    assert!(out_raw.wgsl.contains("@location(2)"));
    assert_naga_valid(&out_raw.wgsl);

    // Case B: Translation WITH parsed FSSM `.meta` (exact VGPR table + `vs_outputs` map).
    let fssm_bytes = build_synthetic_fssm(
        &[(0, 0x7, 4, 2), (1, 0xF, 8, 3), (2, 0x3, 12, 1)],
        &[(1, 0), (2, 1)],
        &[],
    );
    let parsed_meta = parse_fssm_stage_metadata(&fssm_bytes).expect("valid FSSM must parse");
    assert_eq!(
        parsed_meta.vs_inputs,
        vec![
            FssmVertexInput {
                semantic_reg: 0,
                mask: 0x7,
                base_vgpr: 4,
                max_comp: 2
            },
            FssmVertexInput {
                semantic_reg: 1,
                mask: 0xF,
                base_vgpr: 8,
                max_comp: 3
            },
            FssmVertexInput {
                semantic_reg: 2,
                mask: 0x3,
                base_vgpr: 12,
                max_comp: 1
            },
        ]
    );
    assert_eq!(
        parsed_meta.vs_outputs,
        vec![
            WgslStageParamInfo {
                semantic_reg: 1,
                location: 0,
                mask: None
            },
            WgslStageParamInfo {
                semantic_reg: 2,
                location: 1,
                mask: None
            },
        ]
    );

    let out_meta = translate_rdna2_to_wgsl(
        &bytes,
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Vertex,
            fssm_meta: Some(parsed_meta),
            ..Rdna2TranslateOptions::default()
        },
    )
    .expect("FSSM-backed s_swappc_b64 VS must translate");
    assert_eq!(out_meta.unsupported_op_count, 0);
    assert_eq!(out_meta.vertex_inputs, out_raw.vertex_inputs);
    assert_eq!(out_meta.vs_outputs.len(), 2);
    let json = out_meta.to_json();
    assert!(json.contains("\"vertex_inputs\""));
    assert!(json.contains("\"byte_offset\": 12"));
    assert!(json.contains("\"byte_offset\": 16"));
    assert!(json.contains("\"vs_outputs\""));
    assert!(json.contains("\"semantic_reg\": 2, \"location\": 1"));
    assert_naga_valid(&out_meta.wgsl);
}

#[test]
fn fssm_swappc_vs_infers_u32_component_type_for_integer_batch_index_attribute() {
    // Synthetic Scaleform-style FSSM VS:
    //   v4..v5 = float2 POSITION0 (used in v_add_f32 / exp pos0)
    //   v8     = uint  COLOR1 batch index (used only in v_lshlrev_b32)
    let mut words = vec![
        enc_sop1(0x21, 4, 4),   // s_swappc_b64 s[4:5], s[4:5]
        enc_vop2(0x1A, 0, 130, 8), // v_lshlrev_b32 v0, 2, v8 (integer shift on v8)
        enc_vop1(0x06, 6, 256), // v_cvt_f32_u32 v6, v0
        enc_vop1(0x01, 7, 242), // v_mov_b32 v7, 1.0
    ];
    words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [4, 5, 6, 7]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_rdna2_to_wgsl(
        &words_to_bytes(&words),
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Vertex,
            ..Rdna2TranslateOptions::default()
        },
    )
    .expect("integer-attribute FSSM VS must translate");
    assert_eq!(out.unsupported_op_count, 0);
    assert_eq!(out.vertex_inputs.len(), 2);
    assert_eq!(out.vertex_inputs[0].components, 2);
    assert_eq!(out.vertex_inputs[0].component_type, "f32");
    assert_eq!(out.vertex_inputs[0].byte_offset, 0);
    assert_eq!(out.vertex_inputs[1].components, 1);
    assert_eq!(out.vertex_inputs[1].component_type, "u32");
    assert_eq!(out.vertex_inputs[1].byte_offset, 8);
    assert!(
        out.wgsl.contains("@location(1) attr_1") && out.wgsl.contains(": u32,"),
        "WGSL:\n{}",
        out.wgsl
    );
    assert_naga_valid(&out.wgsl);
}

#[test]
fn fssm_pack_manifest_loads_sibling_meta_for_vs_and_ps_and_emits_sidecar_semantics() {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("pg6_fssm_pack_{}_{nanos}", std::process::id()));
    let native_dir = dir.join("native");
    fs::create_dir_all(&native_dir).unwrap();

    // Build VS (s_swappc_b64) + PS (v_interp attr0) + matching FSSM `.meta` files where
    // VS exports COLOR0 (sem=1 -> loc=0) and TEXCOORD0 (sem=2 -> loc=1), while PS reads
    // only TEXCOORD0 (sem=2 -> loc=0, mask=0x3).
    let mut vs_words = vec![
        enc_sop1(0x21, 4, 4),
        enc_vop1(0x01, 7, 242),
        enc_vop1(0x01, 2, 128),
    ];
    vs_words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [4, 5, 6, 7]));
    vs_words.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [8, 9, 10, 11]));
    vs_words.extend_from_slice(&enc_exp(33, 0xF, false, false, false, [12, 13, 2, 7]));
    vs_words.push(enc_sopp(0x01, 0));
    let vs_bytes = words_to_bytes(&vs_words);
    let vs_hash = fnv1a_64(&vs_bytes);

    let mut ps_words = vec![
        enc_vintrp(0, 2, 0, 0, 0),
        enc_vintrp(0, 3, 0, 1, 0),
    ];
    ps_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [2, 3, 2, 3]));
    ps_words.push(enc_sopp(0x01, 0));
    let ps_bytes = words_to_bytes(&ps_words);
    let ps_hash = fnv1a_64(&ps_bytes);

    let vs_meta = build_synthetic_fssm(
        &[(0, 0x7, 4, 2), (1, 0xF, 8, 3), (2, 0x3, 12, 1)],
        &[(1, 0), (2, 1)],
        &[],
    );
    let ps_meta = build_synthetic_fssm(&[], &[], &[(2, 0x3, 0)]);

    let vs_id = format_shader_hash(vs_hash);
    let ps_id = format_shader_hash(ps_hash);
    fs::write(native_dir.join(format!("{vs_id}.bin")), &vs_bytes).unwrap();
    fs::write(native_dir.join(format!("{vs_id}.meta")), &vs_meta).unwrap();
    fs::write(native_dir.join(format!("{ps_id}.bin")), &ps_bytes).unwrap();
    fs::write(native_dir.join(format!("{ps_id}.meta")), &ps_meta).unwrap();

    let manifest = format!(
        r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{vs_id}", "stage": "vertex", "code_file": "native/{vs_id}.bin", "metadata_file": "native/{vs_id}.meta"}},
    {{"code_identity": "{ps_id}", "stage": "pixel", "code_file": "native/{ps_id}.bin", "metadata_file": "native/{ps_id}.meta"}}
  ]
}}"#
    );
    let manifest_path = dir.join("manifest.json");
    fs::write(&manifest_path, manifest).unwrap();

    let packed = pack_rdna2_manifest(&manifest_path, &Rdna2PackOptions::default())
        .expect("manifest with FSSM .meta files must pack cleanly");
    assert_eq!(packed.summary.translated_ok, 2);
    assert_eq!(packed.summary.failed, 0);

    let sidecar = packed
        .summary
        .to_stage_reflection_sidecar(&SidecarBuildOptions::default());
    assert!(sidecar.validate().is_empty());

    let vs_entry = sidecar.stages.get(&vs_hash).expect("VS in sidecar");
    assert_eq!(vs_entry.stage, ShaderStage::Vertex);
    assert_eq!(vs_entry.vertex_inputs.len(), 3);
    assert_eq!(
        vs_entry.vs_outputs,
        vec![
            WgslStageParamInfo {
                semantic_reg: 1,
                location: 0,
                mask: None
            },
            WgslStageParamInfo {
                semantic_reg: 2,
                location: 1,
                mask: None
            },
        ]
    );

    let ps_entry = sidecar.stages.get(&ps_hash).expect("PS in sidecar");
    assert_eq!(ps_entry.stage, ShaderStage::Fragment);
    assert_eq!(
        ps_entry.ps_inputs,
        vec![WgslStageParamInfo {
            semantic_reg: 2,
            location: 0,
            mask: Some(3)
        }]
    );

    let sidecar_json = sidecar.render_json();
    assert!(sidecar_json.contains("\"vertex_inputs\""));
    assert!(sidecar_json.contains("\"vs_outputs\""));
    assert!(sidecar_json.contains("\"ps_inputs\""));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn parse_fssm_stage_metadata_rejects_corrupt_or_truncated_inputs_safely() {
    assert_eq!(parse_fssm_stage_metadata(&[]), None);
    assert_eq!(parse_fssm_stage_metadata(b"NOT_FSSM_AT_ALL"), None);
    let mut bad_ver = build_synthetic_fssm(&[(0, 0x7, 4, 2)], &[], &[]);
    bad_ver[4] = 0xFF;
    assert_eq!(parse_fssm_stage_metadata(&bad_ver), None);
    let truncated = &build_synthetic_fssm(&[(0, 0x7, 4, 2)], &[], &[])[..32];
    assert_eq!(parse_fssm_stage_metadata(truncated), None);
}
