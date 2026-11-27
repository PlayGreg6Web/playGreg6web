//! Integration tests for S75 (#2120, Refs #663, #482):
//! RDNA2 mesh-to-compute lowering coverage, `s16`..`s47` mesh user-data SGPR join reflection,
//! auto-detected mesh packing under `mesh_compute_pack_key`, CFG-aware `GS_ALLOC_REQ` count
//! inference across culled early exits, LDS atomics, wave cross-lane DS ops, `v_mbcnt_*`, and
//! 64-bit `SOP1` bitmask ops.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use naga::valid::{Capabilities, ValidationFlags, Validator};
use playgta6_shaders::binding_pack::{
    MESH_OUTPUT_STORAGE_BUFFER_COUNT, StageBindingCounts, StageBindingLimitAudit,
    StageBindingLimits,
};
use playgta6_shaders::convert::ShaderStage;
use playgta6_shaders::pack::read_pack;
use playgta6_shaders::rdna2::{
    Rdna2InferredStage, SENDMSG_ID_GS_ALLOC_REQ, SENDMSG_ID_GS_DONE, analyze_program,
};
use playgta6_shaders::rdna2_pack::{
    Rdna2BlobInput, Rdna2PackOptions, Rdna2StreamingOptions, pack_rdna2_blobs,
    pack_rdna2_blobs_streaming_to_file, pack_rdna2_manifest,
};
use playgta6_shaders::rdna2_to_wgsl::{
    MESH_OUTPUT_GROUP, MeshOutputLayout, Rdna2StageOverride, Rdna2TranslateOptions,
    WgslBindingSource, mesh_compute_pack_key, translate_rdna2_to_wgsl,
};
use playgta6_shaders::reflect::{
    SidecarBuildOptions, SidecarStageInput, StageReflectionSidecar,
};

// ---------------------------------------------------------------------------
// Synthetic GFX10.3 / RDNA2 instruction encoders (zero game data)
// ---------------------------------------------------------------------------

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

fn branch_simm(branch_byte_off: usize, target_byte_off: usize) -> u16 {
    let next_pc = (branch_byte_off + 4) as isize;
    let delta_dwords = (target_byte_off as isize - next_pc) / 4;
    (delta_dwords as i16) as u16
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

fn enc_sop2(op: u8, sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
    (0b10u32 << 30)
        | (u32::from(op & 0x7F) << 23)
        | (u32::from(sdst & 0x7F) << 16)
        | (u32::from(ssrc1) << 8)
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

fn enc_smem(
    op: u8,
    sdata: u8,
    sbase: u8,
    soffset: u8,
    offset: i32,
    glc: bool,
    dlc: bool,
) -> [u32; 2] {
    let w0 = (0b111101u32 << 26)
        | (u32::from(op) << 18)
        | (u32::from(glc) << 16)
        | (u32::from(dlc) << 14)
        | (u32::from(sdata & 0x7F) << 6)
        | u32::from(sbase & 0x3F);
    let w1 = (u32::from(soffset & 0x7F) << 25) | ((offset as u32) & 0x1F_FFFF);
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

fn enc_mimg(op: u8, dmask: u8, dim: u8, vaddr: u8, vdata: u8, srsrc: u8, ssamp: u8) -> [u32; 2] {
    let op_lo = u32::from(op & 0x7F);
    let op_hi = u32::from((op >> 7) & 0x1);
    let w0 = (0b111100u32 << 26)
        | (op_lo << 18)
        | (u32::from(dmask & 0xF) << 8)
        | (u32::from(dim & 0x7) << 3)
        | op_hi;
    let w1 = (u32::from(ssamp & 0x1F) << 21)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
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

fn assert_naga_wgsl_valid(wgsl: &str) {
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|e| panic!("WGSL parse failed:\n{e}\n--- WGSL ---\n{wgsl}"));
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("Naga validation failed:\n{e:?}\n--- WGSL ---\n{wgsl}"));
}

fn make_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("{prefix}_{}_{nanos}", std::process::id()));
    fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

/// Build a minimal valid mesh shader seeded from `s16` (table ptr -> V# buffer load) +
/// `s_sendmsg(GS_ALLOC_REQ)` + `exp pos0` + `exp param0` + `exp prim`.
fn build_s16_seeded_mesh_blob() -> Vec<u8> {
    let mut w = Vec::new();
    // 32-bit descriptor table pointer in s16 -> s[48:49]
    w.push(enc_sop1(0x03, 48, 16)); // s_mov_b32 s48, s16
    w.push(enc_sop1(0x03, 49, 128)); // s_mov_b32 s49, 0
    // Load V# at s16 + 0x00 into s[52:55] (srsrc = 13)
    w.extend_from_slice(&enc_smem(0x02, 52, 24, 0x7D, 0x00, false, false));
    // buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen (GFX10.3 MUBUF op 0x0F)
    w.extend_from_slice(&enc_mubuf(0x0F, 0, true, false, 0, 4, 13, 0x80));
    // SetMeshOutputCounts(verts=3, prims=1)
    w.push(enc_sop1(0x03, 124, 0xFF));
    w.push((1u32 << 12) | 3u32);
    w.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_ALLOC_REQ)));
    // Primitive indices (0, 1, 2) packed in v8
    w.push(enc_vop1(0x01, 8, 0xFF));
    w.push((2u32 << 20) | (1u32 << 10));
    w.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [4, 5, 6, 7])); // pos0
    w.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [4, 5, 6, 7])); // param0
    w.extend_from_slice(&enc_exp(20, 0x1, false, true, false, [8, 0, 0, 0])); // prim
    w.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_DONE)));
    w.push(enc_sopp(0x01, 0));
    words_to_bytes(&w)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn s75_mesh_s16_table_indirect_chain_and_extended_user_sgprs_join_cleanly() {
    let mut w = Vec::new();

    // Touch NGG/mesh system SGPRs s0, s8, s15 via SALU to verify they NEVER leak as user_sgpr < 16.
    w.push(enc_sop2(0x00, 60, 0, 8)); // s60 = s0 + s8
    w.push(enc_sop2(0x00, 61, 60, 15)); // s61 = s60 + s15

    // 1) Root table pointer in s16 -> V# storage buffer at s16 + 0x00
    w.push(enc_sop1(0x03, 48, 16)); // s48 = s16
    w.push(enc_sop1(0x03, 49, 128)); // s49 = 0
    w.extend_from_slice(&enc_smem(0x02, 52, 24, 0x7D, 0x00, false, false)); // s[52:55] = V# @ s16+0x00
    w.extend_from_slice(&enc_mubuf(0x0F, 0, true, false, 0, 4, 13, 0x80)); // buffer_load_dwordx4 v[4:7]

    // 2) Indirect 2-level `s_load_dwordx2` CBV pointer chain seeded from s16 (+0x20 -> CBV data)
    w.extend_from_slice(&enc_smem(0x01, 56, 24, 0x7D, 0x20, false, false)); // s[56:57] = ptr @ s16+0x20
    w.extend_from_slice(&enc_smem(0x02, 64, 28, 0x7D, 0x00, false, false)); // s[64:67] = CBV floats
    w.push(enc_vop2(0x03, 4, 64, 4)); // v4 = s64 + v4

    // 3) Extended mesh user-data SGPRs past s31 (`s32` resource table & `s34` sampler table,
    //    covering GFX10.3 SPI_SHADER_USER_DATA_GS_0..31 = s16..s47).
    w.push(enc_sop1(0x03, 70, 32)); // s70 = s32 (extended user SGPR!)
    w.push(enc_sop1(0x03, 71, 128));
    w.extend_from_slice(&enc_smem(0x03, 72, 35, 0x7D, 0x40, false, false)); // s[72:79] = T# @ s32+0x40 (srsrc=18)

    w.push(enc_sop1(0x03, 80, 34)); // s80 = s34 (extended user SGPR!)
    w.push(enc_sop1(0x03, 81, 128));
    w.extend_from_slice(&enc_smem(0x02, 84, 40, 0x7D, 0x10, false, false)); // s[84:87] = S# @ s34+0x10 (ssamp=21)

    // image_sample_lz v[10:13], v[4:5], s[72:79], s[84:87] dmask:0xf dim:SQ_RSRC_IMG_2D
    w.extend_from_slice(&enc_mimg(0x25, 0xF, 1, 4, 10, 18, 21));
    w.push(enc_vop2(0x03, 4, 256 + 10, 4));

    // Use s61 in VALU so system SGPR math stays live
    w.push(enc_vop2(0x03, 5, 61, 5));

    // Mesh outputs: SetMeshOutputCounts(verts=64, prims=32)
    w.push(enc_sop1(0x03, 124, 0xFF));
    w.push((32u32 << 12) | 64u32);
    w.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_ALLOC_REQ)));
    w.push(enc_vop1(0x01, 16, 0xFF));
    w.push((2u32 << 20) | (1u32 << 10));
    w.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [4, 5, 6, 7]));
    w.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [10, 11, 12, 13]));
    w.extend_from_slice(&enc_exp(20, 0x1, false, true, false, [16, 0, 0, 0]));
    w.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_DONE)));
    w.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&w);
    let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
        .expect("s16 + extended SGPR mesh shader must translate under Auto stage");

    assert_eq!(out.stage, ShaderStage::Compute);
    assert_eq!(out.entry_point, "cs_main");
    assert!(out.naga_validated);
    assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
    assert_eq!(
        out.mesh_output,
        Some(MeshOutputLayout {
            max_vertices: 64,
            max_primitives: 32,
            vertex_words: 8,
        })
    );

    // Every emitted binding must be in groups 0..2, have a resolved source, and NEVER reference s0..s15.
    assert!(out.bindings.iter().all(|b| b.group < MESH_OUTPUT_GROUP));
    assert!(
        out.bindings.iter().all(|b| b.source.is_some()),
        "all mesh bindings must resolve cleanly: {:?}",
        out.bindings
    );
    for b in &out.bindings {
        if let Some(
            WgslBindingSource::Table { user_sgpr, .. }
            | WgslBindingSource::RootDescriptor { user_sgpr, .. }
            | WgslBindingSource::RootConstants { user_sgpr },
        ) = b.source
        {
            assert!(
                user_sgpr >= 16,
                "mesh user_sgpr must be >= 16 (never s0..s15), got {b:?}"
            );
        }
    }

    // Verify all 4 reflected bindings from s16 (+0x00, +0x20), s32 (+0x40), and s34 (+0x10).
    assert!(
        out.bindings.iter().any(|b| {
            b.kind == "read-only-storage"
                && b.source
                    == Some(WgslBindingSource::Table {
                        user_sgpr: 16,
                        table_byte_offset: 0x00,
                    })
        }),
        "missing s16+0x00 storage buffer binding: {:?}",
        out.bindings
    );
    assert!(
        out.bindings.iter().any(|b| {
            b.kind == "uniform"
                && b.source
                    == Some(WgslBindingSource::Table {
                        user_sgpr: 16,
                        table_byte_offset: 0x20,
                    })
        }),
        "missing s16+0x20 2-level CBV table binding: {:?}",
        out.bindings
    );
    assert!(
        out.bindings.iter().any(|b| {
            b.kind == "texture_2d"
                && b.source
                    == Some(WgslBindingSource::Table {
                        user_sgpr: 32,
                        table_byte_offset: 0x40,
                    })
        }),
        "missing extended s32+0x40 texture binding: {:?}",
        out.bindings
    );
    assert!(
        out.bindings.iter().any(|b| {
            b.kind == "sampler"
                && b.source
                    == Some(WgslBindingSource::Table {
                        user_sgpr: 34,
                        table_byte_offset: 0x10,
                    })
        }),
        "missing extended s34+0x10 sampler binding: {:?}",
        out.bindings
    );

    // Sidecar v3 join check + binding_pack D3D12 limit accounting with ms_out.
    let sidecar = StageReflectionSidecar::build(
        &[SidecarStageInput {
            hash: mesh_compute_pack_key(0x55AA_1122_3344_5566),
            stage: out.stage,
            vertex_input_mode: &out.binding_summary.vertex_input_mode,
            bindings: &out.bindings,
            vertex_inputs: &out.vertex_inputs,
            vs_outputs: &out.vs_outputs,
            ps_inputs: &out.ps_inputs,
            mesh_output: out.mesh_output,
            rt_arena: out.rt_arena,
        }],
        &SidecarBuildOptions::default(),
    );
    assert_eq!(sidecar.census.mesh_compute_stages, 1);
    assert_eq!(sidecar.census.unresolved_bindings, 0);
    assert_eq!(sidecar.census.skipped_stages, 0);

    let counts =
        StageBindingCounts::from_wgsl_bindings(&out.bindings, out.rt_arena, out.mesh_output.is_some());
    assert_eq!(
        counts.storage_buffers,
        1 + MESH_OUTPUT_STORAGE_BUFFER_COUNT,
        "must count both reflected s16 storage buffer and runtime group-3 ms_out"
    );
    let audit = StageBindingLimitAudit::audit_wgsl_bindings(
        &out.bindings,
        out.rt_arena,
        out.mesh_output.is_some(),
        StageBindingLimits::webgpu_d3d12(),
    );
    assert!(audit.within_limits(), "D3D12 limit violations: {audit:?}");
    assert_naga_wgsl_valid(&out.wgsl);
}

#[test]
fn s75_mesh_auto_stage_detection_vs_explicit_vertex_override() {
    let bytes = build_s16_seeded_mesh_blob();

    // 1) Decoder / analyzer infers Mesh stage from GS_ALLOC_REQ + exp prim.
    let summary = analyze_program(&bytes);
    assert_eq!(summary.inferred_stage, Rdna2InferredStage::Mesh);
    assert_eq!(summary.inferred_stage.as_str(), "mesh");

    // 2) Auto stage override automatically lowers to mesh-compute with ms_out & s16 user base.
    let auto_out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
        .expect("Auto stage mesh translation must succeed");
    assert_eq!(auto_out.stage, ShaderStage::Compute);
    assert_eq!(auto_out.entry_point, "cs_main");
    assert!(auto_out.mesh_output.is_some());
    assert!(auto_out.wgsl.contains("@group(3) @binding(0)"));
    assert!(auto_out.wgsl.contains("var<storage, read_write> ms_out: array<u32>;"));
    assert_eq!(
        auto_out.bindings[0].source,
        Some(WgslBindingSource::Table {
            user_sgpr: 16,
            table_byte_offset: 0,
        })
    );
    assert_naga_wgsl_valid(&auto_out.wgsl);

    // 3) Explicit Vertex override preserves Vertex stage (NGG vertex shaders stay vs_main).
    let vs_opts = Rdna2TranslateOptions {
        stage: Rdna2StageOverride::Vertex,
        ..Rdna2TranslateOptions::default()
    };
    let vs_out =
        translate_rdna2_to_wgsl(&bytes, &vs_opts).expect("explicit Vertex override must succeed");
    assert_eq!(vs_out.stage, ShaderStage::Vertex);
    assert_eq!(vs_out.entry_point, "vs_main");
    assert_eq!(vs_out.mesh_output, None);
    assert!(!vs_out.wgsl.contains("ms_out"));
    assert_naga_wgsl_valid(&vs_out.wgsl);
}

#[test]
fn s75_mesh_culled_zero_prim_early_exit_and_cfg_const_alloc_counts() {
    // Mesh shader with a zero-primitive culled early exit AND a multi-block SALU-computed
    // non-zero `GS_ALLOC_REQ` (`128` verts, `126` prims) exceeding default `workgroup_size = 64`:
    //   if (s16 == 0) { m0 = 0; s_sendmsg(GS_ALLOC_REQ); s_endpgm; }
    //   s20 = 126; s20 = s20 << 12; m0 = s20 | 128; s_sendmsg(GS_ALLOC_REQ);
    //   s_and_saveexec_b32 s22, vcc_lo;  // enable exec_mask tracking!
    //   exp pos0, pos1 (clip dist), param0..param2, prim; s_endpgm;
    let mut w = Vec::new();

    // s_cmp_eq_u32 s0, 0 (runtime-dynamic `wg_id.x`) -> branch to active path if scc == 0
    w.push(enc_sopc(0x06, 0, 128));
    let br_to_active = w.len();
    w.push(0); // placeholder for s_cbranch_scc0

    // Culled early-exit path: m0 = 0; s_sendmsg(GS_ALLOC_REQ); s_endpgm
    w.push(enc_sop1(0x03, 124, 128)); // s_mov_b32 m0, 0
    w.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_ALLOC_REQ)));
    w.push(enc_sopp(0x01, 0));

    // Active path: compute m0 = (126 << 12) | 128 via SALU
    let active_byte_off = w.len() * 4;
    w[br_to_active] = enc_sopp(0x04, branch_simm(br_to_active * 4, active_byte_off));

    w.push(enc_sop1(0x03, 20, 0xFF)); // s_mov_b32 s20, 126
    w.push(126u32);
    w.push(enc_sop2(0x1E, 20, 20, 128 + 12)); // s_lshl_b32 s20, s20, 12
    w.push(enc_sop2(0x10, 124, 20, 0xFF)); // s_or_b32 m0, s20, 128
    w.push(128u32);
    w.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_ALLOC_REQ)));

    // Exercise runtime-dynamic exec_mask tracking (`v_cmp_lt_f32` + `s_and_saveexec_b32`) to
    // confirm GS_ALLOC_REQ & exp respect exec_mask.
    w.push((0b0111_110u32 << 25) | (0x01u32 << 17) | (0u32 << 9) | 240u32); // v_cmp_lt_f32 vcc, 0.5, v0
    w.push(enc_sop1(0x3C, 22, 106)); // s_and_saveexec_b32 s22, vcc_lo

    // Export pos0, pos1 (auxiliary clip distance), param0, param1, param2, and prim.
    w.push(enc_vop1(0x01, 4, 0xF2)); // v4 = 1.0
    w.push(enc_vop1(0x01, 5, 0xF2));
    w.push(enc_vop1(0x01, 6, 0xF2));
    w.push(enc_vop1(0x01, 7, 0xF2));
    w.push(enc_vop1(0x01, 8, 0xFF));
    w.push((2u32 << 20) | (1u32 << 10));
    w.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [4, 5, 6, 7])); // pos0
    w.extend_from_slice(&enc_exp(13, 0xF, false, false, false, [4, 5, 6, 7])); // pos1
    w.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [4, 5, 6, 7])); // param0
    w.extend_from_slice(&enc_exp(33, 0xF, false, false, false, [4, 5, 6, 7])); // param1
    w.extend_from_slice(&enc_exp(34, 0xF, false, false, false, [4, 5, 6, 7])); // param2
    w.extend_from_slice(&enc_exp(20, 0x1, false, true, false, [8, 0, 0, 0])); // prim
    w.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&w);
    let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
        .expect("culled early-exit mesh shader must translate");

    assert_eq!(out.stage, ShaderStage::Compute);
    assert!(out.naga_validated);
    assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
    assert_eq!(
        out.mesh_output,
        Some(MeshOutputLayout {
            max_vertices: 128,
            max_primitives: 126,
            vertex_words: 16, // 4 (pos) + 3 * 4 (param0..2)
        }),
        "zero-prim cull early exit must not erase static (128, 126) alloc counts"
    );
    assert!(
        out.wgsl.contains("exec_mask"),
        "expected exec_mask guard in lowered mesh WGSL:\n{}",
        out.wgsl
    );
    assert_naga_wgsl_valid(&out.wgsl);
}

#[test]
fn s75_mesh_lds_atomics_wave_cross_lane_mbcnt_and_64bit_salu_lowering() {
    let mut w = Vec::new();

    // 1) Seed v2 = v0 << 2 (byte address into LDS), v3 = 1
    w.push(enc_vop2(0x18, 2, 128 + 2, 0)); // v_lshlrev_b32 v2, 2, v0
    w.push(enc_vop1(0x01, 3, 128 + 1)); // v_mov_b32 v3, 1

    // 2) Mixed LDS regular store/load + barrier + LDS atomics (`ds_add_rtn_u32`, `ds_umax_u32`,
    //    `ds_inc_rtn_u32`, `ds_wrxchg_rtn_b32`) on shared `array<atomic<u32>, 4096>`.
    w.extend_from_slice(&enc_ds(0x0D, 0, 0, 2, 3, 0, 0)); // ds_write_b32 v2, v3
    w.push(enc_sopp(0x0A, 0)); // s_barrier
    w.extend_from_slice(&enc_ds(0x36, 0, 0, 2, 0, 0, 4)); // ds_read_b32 v4, v2
    w.extend_from_slice(&enc_ds(0x20, 4, 0, 2, 3, 0, 5)); // ds_add_rtn_u32 v5, v2, v3 offset:4
    w.extend_from_slice(&enc_ds(0x08, 4, 0, 2, 4, 0, 0)); // ds_max_u32 v2, v4 offset:4
    w.extend_from_slice(&enc_ds(0x23, 8, 0, 2, 3, 0, 6)); // ds_inc_rtn_u32 v6, v2, v3 offset:8
    w.extend_from_slice(&enc_ds(0x2D, 8, 0, 2, 5, 0, 7)); // ds_wrxchg_rtn_b32 v7, v2, v5 offset:8

    // 3) Cross-lane wave DS ops in uniform CF:
    //    - ds_swizzle_b32 QuadPerm(3,2,1,0) -> offset16 = 0x801B
    //    - ds_swizzle_b32 BitMode(and=0x1F, or=0, xor=1) -> offset16 = 0x041F
    //    - ds_bpermute_b32 v9, v2, v6
    //    - ds_permute_b32 v10, v2, v7
    w.extend_from_slice(&enc_ds(0x35, 0x1B, 0x80, 5, 0, 0, 8));
    w.extend_from_slice(&enc_ds(0x35, 0x1F, 0x04, 8, 0, 0, 8));
    w.extend_from_slice(&enc_ds(0xB3, 0, 0, 2, 6, 0, 9));
    w.extend_from_slice(&enc_ds(0xB2, 0, 0, 2, 7, 0, 10));

    // 4) Wave DS op inside non-uniform branch (must remain WGSL uniformity valid!)
    w.push(enc_sopc(0x06, 16, 128));
    let br_skip_swz = w.len();
    w.push(0);
    w.extend_from_slice(&enc_ds(0x35, 0xE4, 0x80, 9, 0, 0, 9));
    let after_swz = w.len() * 4;
    w[br_skip_swz] = enc_sopp(0x04, branch_simm(br_skip_swz * 4, after_swz));

    // 5) Wave prefix sum & lane read: v_mbcnt_lo_u32_b32, v_mbcnt_hi_u32_b32, v_readfirstlane_b32_e64
    w.extend_from_slice(&enc_vop3(0x365, 11, 0xFF, 128, 0)); // v_mbcnt_lo_u32_b32 v11, 0xFFFF_FFFF, 0
    w.push(0xFFFF_FFFFu32);
    w.extend_from_slice(&enc_vop3(0x366, 12, 0xFF, 256 + 11, 0)); // v_mbcnt_hi_u32_b32 v12, 0xFFFF_FFFF, v11
    w.push(0xFFFF_FFFFu32);
    w.extend_from_slice(&enc_vop3(0x182, 30, 256 + 12, 0, 0)); // v_readfirstlane_b32_e64 s30, v12

    // 6) 64-bit SOP1 bitmask ops commonly used in mesh wave compaction:
    w.push(enc_sop1(0x04, 32, 0xFF)); // s_mov_b64 s[32:33], 0x55AA_00FF
    w.push(0x55AA_00FFu32);
    w.push(enc_sop1(0x06, 34, 32)); // s_cmov_b64 s[34:35], s[32:33]
    w.push(enc_sop1(0x0C, 36, 34)); // s_brev_b64 s[36:37], s[34:35]
    w.push(enc_sop1(0x0E, 38, 36)); // s_bcnt0_i32_b64 s38, s[36:37]
    w.push(enc_sop1(0x10, 39, 36)); // s_bcnt1_i32_b64 s39, s[36:37]
    w.push(enc_sop1(0x12, 40, 36)); // s_ff0_i32_b64 s40, s[36:37]
    w.push(enc_sop1(0x14, 41, 36)); // s_ff1_i32_b64 s41, s[36:37]
    w.push(enc_sop1(0x16, 42, 36)); // s_flbit_i32_b64 s42, s[36:37]
    w.push(enc_sop1(0x18, 43, 36)); // s_flbit_i32_i64 s43, s[36:37]
    w.push(enc_sop1(0x1C, 34, 128 + 5)); // s_bitset0_b64 s[34:35], 5
    w.push(enc_sop1(0x1E, 34, 128 + 37)); // s_bitset1_b64 s[34:35], 37
    w.push(enc_sop1(0x2D, 44, 34)); // s_quadmask_b64 s[44:45], s[34:35]

    // Fold results into exported vertex & primitive words (including s40/s42 so
    // `countTrailingZeros`/`countLeadingZeros` remain live after dead-local pruning).
    w.push(enc_sop2(0x00, 46, 30, 39));
    w.push(enc_sop2(0x00, 46, 46, 40));
    w.push(enc_sop2(0x00, 46, 46, 42));
    w.push(enc_sop2(0x00, 46, 46, 44));
    w.push(enc_vop2(0x03, 8, 46, 8));
    w.push(enc_vop2(0x03, 9, 256 + 10, 9));

    // Mesh allocation & exports
    w.push(enc_sop1(0x03, 124, 0xFF));
    w.push((16u32 << 12) | 32u32);
    w.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_ALLOC_REQ)));
    w.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [8, 9, 11, 12]));
    w.extend_from_slice(&enc_exp(20, 0x1, false, true, false, [12, 0, 0, 0]));
    w.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&w);
    let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
        .expect("full mesh wave/LDS-atomic/64-bit SALU shader must translate");

    assert_eq!(out.stage, ShaderStage::Compute);
    assert!(out.naga_validated);
    assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
    assert!(
        out.wgsl.contains("array<atomic<u32>, 4096>"),
        "expected atomic LDS array declaration:\n{}",
        out.wgsl
    );
    assert!(
        out.wgsl.contains("atomicAdd(")
            && out.wgsl.contains("atomicMax(")
            && out.wgsl.contains("atomicExchange(")
            && out.wgsl.contains("atomicLoad(")
            && out.wgsl.contains("atomicStore("),
        "expected mixed atomic LDS ops in WGSL:\n{}",
        out.wgsl
    );
    assert!(
        out.wgsl.contains("var<workgroup> pg6_wave_lds: array<u32, 256>;"),
        "expected wave cross-lane LDS scratch buffer:\n{}",
        out.wgsl
    );
    assert!(
        out.wgsl.contains("countOneBits(")
            && out.wgsl.contains("countTrailingZeros(")
            && out.wgsl.contains("countLeadingZeros(")
            && out.wgsl.contains("reverseBits("),
        "expected lowered 64-bit SOP1 & mbcnt bit-ops in WGSL:\n{}",
        out.wgsl
    );
    assert_naga_wgsl_valid(&out.wgsl);
}

#[test]
fn s75_pack_rdna2_auto_mesh_blobs_streaming_and_manifest_roundtrip() {
    let mesh_bytes = build_s16_seeded_mesh_blob();
    let raw_hash: u64 = 0x0000_1122_3344_5566;
    let expected_salted_hash = mesh_compute_pack_key(raw_hash);

    // 1) Raw unnamed hex blob (`0x0000112233445566.bin`, NO `ms_` prefix, StageOverride::Auto).
    let blobs = vec![Rdna2BlobInput {
        name: format!("0x{raw_hash:016x}.bin"),
        bytes: mesh_bytes.clone(),
        stage_override: Rdna2StageOverride::Auto,
        hash_override: None,
        fssm_meta: None,
    }];
    let pack_opts = Rdna2PackOptions::default();

    let in_mem = pack_rdna2_blobs(&blobs, &pack_opts).expect("in-memory pack must succeed");
    assert_eq!(in_mem.summary.translated_ok, 1);
    assert_eq!(in_mem.summary.compute_packed, 1);
    assert_eq!(in_mem.summary.vertex_packed, 0);
    assert_eq!(in_mem.summary.failed, 0);
    assert_eq!(in_mem.entries.len(), 1);
    assert_eq!(
        in_mem.entries[0].hash, expected_salted_hash,
        "auto-detected mesh blob must be packed under mesh_compute_pack_key(raw_hash)"
    );
    assert_eq!(in_mem.entries[0].stage, ShaderStage::Compute);
    assert_eq!(
        in_mem.summary.entries[0].mesh_output,
        Some(MeshOutputLayout {
            max_vertices: 3,
            max_primitives: 1,
            vertex_words: 8,
        })
    );
    assert_eq!(
        in_mem.summary.entries[0].bindings[0].source,
        Some(WgslBindingSource::Table {
            user_sgpr: 16,
            table_byte_offset: 0,
        })
    );
    assert_naga_wgsl_valid(&in_mem.entries[0].wgsl);

    // Verify parsed WGPK binary round-trips cleanly.
    let parsed = read_pack(&in_mem.pack_bytes).expect("read_pack must succeed");
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].hash, expected_salted_hash);
    assert_eq!(parsed.entries[0].stage, ShaderStage::Compute);
    assert!(parsed.entries[0].wgsl.contains("@group(3) @binding(0)"));
    assert_naga_wgsl_valid(&parsed.entries[0].wgsl);

    // 2) Streaming packer produces byte-identical WGPK output for the same auto-detected mesh blob.
    let temp_dir = make_temp_dir("pg6_s75_mesh_pack");
    let stream_out_path = temp_dir.join("stream.wgsl.pack");
    let stream_res = pack_rdna2_blobs_streaming_to_file(
        &blobs,
        &stream_out_path,
        &Rdna2StreamingOptions::default(),
    )
    .expect("streaming mesh pack must succeed");
    let streamed_bytes = fs::read(&stream_out_path).expect("read streamed pack");
    assert_eq!(
        streamed_bytes, in_mem.pack_bytes,
        "streaming and in-memory packs must be byte-identical for auto-detected mesh blobs"
    );
    assert_eq!(stream_res.toc_entries[0].hash, expected_salted_hash);

    // 3) Manifest packing accepts `"stage": "mesh_shader"` (in addition to `"mesh"` / `"ms"`).
    let code_id = playgta6_shaders::convert::fnv1a_64(&mesh_bytes);
    let blob_file = temp_dir.join("stage_0.bin");
    fs::write(&blob_file, &mesh_bytes).expect("write manifest blob");
    let manifest_path = temp_dir.join("manifest.json");
    let manifest_json = format!(
        r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{
      "code_identity": "0x{code_id:016x}",
      "stage": "mesh_shader",
      "code_file": "stage_0.bin",
      "code_bytes": {}
    }}
  ]
}}"#,
        mesh_bytes.len()
    );
    fs::write(&manifest_path, manifest_json).expect("write manifest.json");
    let manifest_res =
        pack_rdna2_manifest(&manifest_path, &pack_opts).expect("manifest mesh_shader pack must succeed");
    assert_eq!(manifest_res.summary.translated_ok, 1);
    assert_eq!(manifest_res.entries[0].hash, mesh_compute_pack_key(code_id));
    assert_eq!(manifest_res.entries[0].stage, ShaderStage::Compute);
    assert!(manifest_res.summary.entries[0].mesh_output.is_some());
    assert_naga_wgsl_valid(&manifest_res.entries[0].wgsl);

    let _ = fs::remove_dir_all(&temp_dir);
}
