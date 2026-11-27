//! S74 (#2110, Refs #663, #482): Per-stage WGSL binding-limit packing, dead-binding pruning,
//! duplicate/aliased descriptor deduplication, excess CBV -> `read-only-storage` spillover,
//! reflection/sidecar synchronization, and WebGPU/D3D12 per-stage limit auditing (12/16/8/16/16).

use std::collections::BTreeSet;

use naga::valid::{Capabilities, ValidationFlags, Validator};
use playgta6_shaders::binding_pack::{
    BindingPackStats, MESH_OUTPUT_STORAGE_BUFFER_COUNT, RT_ARENA_STORAGE_BUFFER_COUNT,
    StageBindingCounts, StageBindingLimitAudit, StageBindingLimitKind, StageBindingLimits,
    WEBGPU_D3D12_MAX_SAMPLED_TEXTURES_PER_STAGE, WEBGPU_D3D12_MAX_SAMPLERS_PER_STAGE,
    WEBGPU_D3D12_MAX_STORAGE_BUFFERS_PER_STAGE, WEBGPU_D3D12_MAX_STORAGE_TEXTURES_PER_STAGE,
    WEBGPU_D3D12_MAX_UNIFORM_BUFFERS_PER_STAGE, canonicalize_binding_source,
    deduplicate_module_bindings, prune_unused_module_bindings,
};
use playgta6_shaders::compat::{CompatIssueKind, WebGpuLimits, audit_naga_module_with_limits};
use playgta6_shaders::convert::ShaderStage;
use playgta6_shaders::rdna2_pack::{Rdna2BlobInput, Rdna2PackOptions, pack_rdna2_blobs};
use playgta6_shaders::rdna2_to_wgsl::{
    CbvIndexedHeapSource, CbvRefSource, LaneIndexedHeapSource, Rdna2StageOverride,
    Rdna2TranslateOptions, Rdna2WgslOutput, WgslBindingInfo, WgslBindingSource,
    translate_rdna2_to_wgsl,
};
use playgta6_shaders::reflect::{SidecarBuildOptions, SidecarStageInput, StageReflectionSidecar};

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

fn enc_smem(
    op: u8,
    sdata: u8,
    sbase_pair: u8,
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
        | u32::from(sbase_pair & 0x3F);
    let w1 = (u32::from(soffset & 0x7F) << 25) | ((offset as u32) & 0x001F_FFFF);
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
        | u32::from(offset & 0x0FFF);
    let w1 = (u32::from(soffset) << 24)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
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

fn branch_simm(branch_byte_off: usize, target_byte_off: usize) -> u16 {
    let rel_bytes = target_byte_off as isize - (branch_byte_off as isize + 4);
    assert_eq!(rel_bytes % 4, 0);
    ((rel_bytes / 4) as i16) as u16
}

fn parse_and_validate_wgsl(wgsl: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|err| panic!("naga WGSL parse failed:\n{err}\n\nWGSL:\n{wgsl}"));
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    validator
        .validate(&module)
        .unwrap_or_else(|err| panic!("naga WGSL validate failed:\n{err}\n\nWGSL:\n{wgsl}"));
    module
}

/// Assert 1-to-1 synchronization between emitted WGSL `@group/@binding` globals (excluding
/// runtime-owned `@group(3)`) and `out.bindings`, plus valid sidecar reflection.
fn assert_wgsl_and_reflection_in_sync(out: &Rdna2WgslOutput) -> naga::Module {
    assert!(out.naga_validated, "expected naga_validated:\n{}", out.wgsl);
    let module = parse_and_validate_wgsl(&out.wgsl);

    let wgsl_slots: BTreeSet<(u32, u32)> = module
        .global_variables
        .iter()
        .filter_map(|(_, var)| {
            let rb = var.binding?;
            if rb.group == 3 {
                None
            } else {
                Some((rb.group, rb.binding))
            }
        })
        .collect();
    let refl_slots: BTreeSet<(u32, u32)> = out
        .bindings
        .iter()
        .map(|b| (b.group, b.binding))
        .collect();
    assert_eq!(
        wgsl_slots, refl_slots,
        "WGSL @group/@binding slots must match out.bindings 1-to-1.\nWGSL:\n{}\nbindings: {:?}",
        out.wgsl, out.bindings
    );

    let sidecar = StageReflectionSidecar::build(
        &[SidecarStageInput {
            hash: out.hash_fnv1a64,
            stage: out.stage,
            bindings: &out.bindings,
            vertex_inputs: &out.vertex_inputs,
            vertex_input_mode: &out.binding_summary.vertex_input_mode,
            vs_outputs: &out.vs_outputs,
            ps_inputs: &out.ps_inputs,
            mesh_output: out.mesh_output,
            rt_arena: out.rt_arena,
        }],
        &SidecarBuildOptions::default(),
    );
    assert!(
        sidecar.validate().is_empty(),
        "sidecar validation errors: {:?}",
        sidecar.validate()
    );
    if let Some(entry) = sidecar.stages.get(&out.hash_fnv1a64) {
        let sidecar_slots: BTreeSet<(u32, u32)> = entry
            .bindings
            .iter()
            .map(|b| (b.group, b.binding))
            .collect();
        assert_eq!(sidecar_slots, refl_slots);
    }

    module
}

fn translate_compute(words: &[u32]) -> Rdna2WgslOutput {
    translate_rdna2_to_wgsl(
        &words_to_bytes(words),
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Compute,
            ..Rdna2TranslateOptions::default()
        },
    )
    .expect("compute shader must translate")
}

// ===========================================================================
// 1. Dead / unreferenced binding pruning (unreachable CFG block)
// ===========================================================================

#[test]
fn s74_dead_unreachable_branch_bindings_are_pruned_across_all_resource_kinds() {
    // Compute shader (user table pointer at s4):
    //   s20 = s4, s21 = 0
    //   s_branch L_LIVE           (jumps directly over dead block!)
    // L_DEAD (unreachable):
    //   s_load_dwordx4 s[24:27], s[20:21], 0x20 -> s_buffer_load_dword (dead CBV)
    //   s_load_dwordx8 s[32:39], s[20:21], 0x40 -> image_sample (dead tex + dead samp)
    //   s_load_dwordx4 s[40:43], s[20:21], 0x60 -> buffer_load_dword (dead sbuf)
    //   s_load_dwordx8 s[48:55], s[20:21], 0x80 -> image_store (dead uav_tex)
    // L_LIVE:
    //   s_load_dwordx4 s[60:63], s[20:21], 0x00 -> s_buffer_load_dword (live CBV)
    //   s_load_dwordx4 s[64:67], s[20:21], 0xA0 -> buffer_store_dword (live uav_buf)
    //   s_endpgm
    let mut words = Vec::new();
    words.push(enc_sop1(0x03, 20, 4)); // s20 = s4
    words.push(enc_sop1(0x03, 21, 0x80)); // s21 = 0

    let branch_idx = words.len();
    words.push(0); // placeholder for s_branch L_LIVE

    // Dead unreachable block:
    words.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, 0x20, false, false)); // V# -> s[24:27]
    words.extend_from_slice(&enc_smem(0x08, 28, 12, 0x7D, 0x00, false, false)); // dead s_buffer_load_dword
    words.extend_from_slice(&enc_smem(0x03, 32, 10, 0x7D, 0x40, false, false)); // T# -> s[32:39]
    words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 0, 4, 8, 2)); // dead image_sample (srsrc=8, ssamp=2)
    words.extend_from_slice(&enc_smem(0x02, 40, 10, 0x7D, 0x60, false, false)); // V# -> s[40:43]
    words.extend_from_slice(&enc_mubuf(0x0C, 0, true, false, 0, 8, 10, 0x7D)); // dead buffer_load_dword
    words.extend_from_slice(&enc_smem(0x03, 48, 10, 0x7D, 0x80, false, false)); // T# -> s[48:55]
    words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 8, 12, 0)); // dead image_store

    let live_byte_off = words.len() * 4;
    words[branch_idx] = enc_sopp(0x02, branch_simm(branch_idx * 4, live_byte_off));

    // Live block:
    words.extend_from_slice(&enc_smem(0x02, 60, 10, 0x7D, 0x00, false, false)); // live CBV V# @ s4+0x00
    words.extend_from_slice(&enc_smem(0x08, 29, 30, 0x7D, 0x00, false, false)); // live s_buffer_load_dword
    words.extend_from_slice(&enc_smem(0x02, 64, 10, 0x7D, 0xA0, false, false)); // live UAV V# @ s4+0xA0
    words.extend_from_slice(&enc_mubuf(0x1C, 0, true, false, 0, 1, 16, 0x7D)); // live buffer_store_dword
    words.push(enc_sopp(0x01, 0)); // s_endpgm

    let out = translate_compute(&words);
    assert_wgsl_and_reflection_in_sync(&out);

    assert!(
        out.binding_pack.dead_bindings_pruned >= 5,
        "expected >= 5 dead bindings pruned (cb, tex, samp, sbuf, uav_tex), got {:?}",
        out.binding_pack
    );
    assert_eq!(
        out.bindings.len(),
        2,
        "only live CBV and live UAV buffer should survive: {:?}",
        out.bindings
    );
    assert!(out.bindings.iter().any(|b| b.kind == "uniform"
        && b.source == Some(WgslBindingSource::Table { user_sgpr: 4, table_byte_offset: 0x00 })));
    assert!(out.bindings.iter().any(|b| b.kind == "storage"
        && b.source == Some(WgslBindingSource::Table { user_sgpr: 4, table_byte_offset: 0xA0 })));
    assert!(out.binding_summary.has_cbv);
    assert!(out.binding_summary.has_storage_buffer);
    assert!(!out.binding_summary.has_texture_2d);
    assert!(!out.binding_summary.has_sampler);
    assert!(!out.binding_summary.has_storage_texture);
}

// ===========================================================================
// 2. Atomic-only UAV placeholders: preserved within limit, pruned over limit
// ===========================================================================

#[test]
fn s74_atomic_only_uav_placeholders_preserved_within_limit_and_pruned_over_limit() {
    // Case A: 2 distinct buffer_atomic_add UAVs (well within 16 limit) -> both preserved.
    let mut small = Vec::new();
    small.push(enc_sop1(0x03, 20, 4));
    small.push(enc_sop1(0x03, 21, 0x80));
    for i in 0u32..2 {
        small.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, (i * 32) as i32, false, false));
        small.extend_from_slice(&enc_mubuf(0x34, 0, true, false, 0, 1, 6, 0x7D)); // buffer_atomic_add
    }
    small.push(enc_sopp(0x01, 0));

    let out_small = translate_compute(&small);
    assert_wgsl_and_reflection_in_sync(&out_small);
    assert_eq!(
        out_small.bindings.iter().filter(|b| b.kind == "storage").count(),
        2,
        "within-limit atomic UAV bindings must stay preserved: {:?}",
        out_small.bindings
    );

    // Case B: 1 live buffer_store_dword UAV + 18 distinct buffer_atomic_add UAVs (= 19 > 16) ->
    // 3 unreferenced atomic-only UAV placeholders are pruned to fit the 16-storage-buffer D3D12 cap.
    let mut over = Vec::new();
    over.push(enc_sop1(0x03, 20, 4));
    over.push(enc_sop1(0x03, 21, 0x80));
    // Live store at offset 0x00:
    over.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, 0, false, false));
    over.extend_from_slice(&enc_mubuf(0x1C, 0, true, false, 0, 1, 6, 0x7D));
    // 18 atomic-only slots at offsets 0x20..0x240:
    for i in 1u32..=18 {
        over.extend_from_slice(&enc_smem(0x02, 28, 10, 0x7D, (i * 32) as i32, false, false));
        over.extend_from_slice(&enc_mubuf(0x34, 0, true, false, 0, 1, 7, 0x7D));
    }
    over.push(enc_sopp(0x01, 0));

    let out_over = translate_compute(&over);
    assert_wgsl_and_reflection_in_sync(&out_over);
    let counts = StageBindingCounts::from_wgsl_bindings(&out_over.bindings, out_over.rt_arena, false);
    assert_eq!(
        counts.storage_buffers, WEBGPU_D3D12_MAX_STORAGE_BUFFERS_PER_STAGE,
        "over-limit atomic-only placeholders must prune down to 16 storage buffers: {:?}",
        out_over.bindings
    );
    assert_eq!(out_over.binding_pack.dead_bindings_pruned, 3);
    assert!(
        out_over.bindings.iter().any(|b| b.source
            == Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0
            })),
        "live buffer_store_dword UAV must remain intact"
    );
}

// ===========================================================================
// 3. Duplicate / aliased binding deduplication & source canonicalization
// ===========================================================================

#[test]
fn s74_duplicate_aliased_descriptors_fold_via_sgpr_copies_and_canonical_sources() {
    // 1) Unit checks on canonicalize_binding_source:
    let mut table_sgprs = BTreeSet::new();
    table_sgprs.insert(8);

    // RootDescriptor on a known table SGPR canonicalizes to Table.
    assert_eq!(
        canonicalize_binding_source(
            WgslBindingSource::RootDescriptor {
                user_sgpr: 8,
                table_byte_offset: None,
            },
            &table_sgprs
        ),
        WgslBindingSource::Table {
            user_sgpr: 8,
            table_byte_offset: 0,
        }
    );
    assert_eq!(
        canonicalize_binding_source(
            WgslBindingSource::RootDescriptor {
                user_sgpr: 8,
                table_byte_offset: Some(64),
            },
            &table_sgprs
        ),
        WgslBindingSource::Table {
            user_sgpr: 8,
            table_byte_offset: 64,
        }
    );
    // RootDescriptor on a non-table SGPR normalizes Some(0) -> None.
    assert_eq!(
        canonicalize_binding_source(
            WgslBindingSource::RootDescriptor {
                user_sgpr: 10,
                table_byte_offset: Some(0),
            },
            &table_sgprs
        ),
        WgslBindingSource::RootDescriptor {
            user_sgpr: 10,
            table_byte_offset: None,
        }
    );
    // Power-of-two stride == (1 << shift) canonicalizes identically to stride == 0.
    let cbv_heap_zero = CbvIndexedHeapSource {
        cbv: CbvRefSource::RootDescriptor { user_sgpr: 10 },
        byte_offset: 16,
        shift: 5,
        bit_offset: 0,
        mask: u32::MAX,
        stride: 0,
    };
    let cbv_heap_pow2 = CbvIndexedHeapSource {
        stride: 32,
        ..cbv_heap_zero
    };
    assert_eq!(
        canonicalize_binding_source(WgslBindingSource::CbvIndexedResourceHeap(cbv_heap_zero), &table_sgprs),
        canonicalize_binding_source(WgslBindingSource::CbvIndexedResourceHeap(cbv_heap_pow2), &table_sgprs),
    );
    let lane_heap_zero = LaneIndexedHeapSource {
        max_count: 16,
        shift: 5,
        stride: 0,
        window_base: None,
    };
    let lane_heap_pow2 = LaneIndexedHeapSource {
        stride: 32,
        ..lane_heap_zero
    };
    assert_eq!(
        canonicalize_binding_source(WgslBindingSource::LaneIndexedResourceHeap(lane_heap_zero), &table_sgprs),
        canonicalize_binding_source(WgslBindingSource::LaneIndexedResourceHeap(lane_heap_pow2), &table_sgprs),
    );

    // 2) End-to-end RDNA2 compute shader:
    //    Load the EXACT same CBV V# (s4+0x20) into two different SGPR quads (s[24:27] and s[28:31])
    //    AND copy s[24:27] -> s[32:35] via s_mov_b64; load the exact same UAV V# (s4+0x40) twice
    //    (once for load, once for store); load the exact same T# (s4+0x60) twice for image_store.
    let mut words = Vec::new();
    words.push(enc_sop1(0x03, 20, 4));
    words.push(enc_sop1(0x03, 21, 0x80));

    // CBV @ s4+0x20 loaded into s[24:27], reloaded into s[28:31], and copied into s[32:35]:
    words.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, 0x20, false, false));
    words.extend_from_slice(&enc_smem(0x02, 28, 10, 0x7D, 0x20, false, false));
    words.push(enc_sop1(0x04, 32, 24)); // s_mov_b64 s[32:33], s[24:25]
    words.push(enc_sop1(0x04, 34, 26)); // s_mov_b64 s[34:35], s[26:27]
    words.extend_from_slice(&enc_smem(0x08, 40, 12, 0x7D, 0x00, false, false)); // from s[24:27]
    words.extend_from_slice(&enc_smem(0x08, 41, 14, 0x7D, 0x10, false, false)); // from s[28:31]
    words.extend_from_slice(&enc_smem(0x08, 42, 16, 0x7D, 0x20, false, false)); // from s[32:35]

    // Storage buffer @ s4+0x40 loaded into s[44:47] (read) and s[48:51] (write):
    words.extend_from_slice(&enc_smem(0x02, 44, 10, 0x7D, 0x40, false, false));
    words.extend_from_slice(&enc_smem(0x02, 48, 10, 0x7D, 0x40, false, false));
    words.extend_from_slice(&enc_mubuf(0x0C, 0, true, false, 0, 2, 11, 0x7D)); // buffer_load_dword
    words.extend_from_slice(&enc_mubuf(0x1C, 0, true, false, 0, 2, 12, 0x7D)); // buffer_store_dword

    // Storage texture @ s4+0x60 loaded into s[56:63] and s[64:71]:
    words.extend_from_slice(&enc_smem(0x03, 56, 10, 0x7D, 0x60, false, false));
    words.extend_from_slice(&enc_smem(0x03, 64, 10, 0x7D, 0x60, false, false));
    words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 2, 14, 0));
    words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 2, 16, 0));

    words.push(enc_sopp(0x01, 0));

    let out = translate_compute(&words);
    assert_wgsl_and_reflection_in_sync(&out);
    assert_eq!(
        out.bindings.len(),
        3,
        "all duplicate SGPR copies must collapse to exactly 1 CBV, 1 UAV buf, 1 UAV tex: {:?}",
        out.bindings
    );
    let cb = out.bindings.iter().find(|b| b.kind == "uniform").unwrap();
    assert_eq!(cb.min_binding_size_bytes, Some(48)); // highest offset 0x20 -> vec4 index 2 -> 48 B
    assert!(out.bindings.iter().any(|b| b.kind == "storage"));
    assert!(out.bindings.iter().any(|b| b.kind == "storage-texture"));
}

// ===========================================================================
// 4. Direct Naga IR pruning & deduplication (module + reflection 100% synced)
// ===========================================================================

#[test]
fn s74_direct_naga_module_pruning_and_deduplication_keeps_ir_and_reflection_in_sync() {
    let src = r#"
struct Rdna2Cbv {
    data: array<vec4<u32>, 4>,
}
struct Rdna2Cbv1 {
    data: array<vec4<u32>, 8>,
}

@group(0) @binding(0) var<uniform> cb0: Rdna2Cbv;
@group(0) @binding(1) var<uniform> cb1: Rdna2Cbv1;
@group(0) @binding(2) var<uniform> cb_dead: Rdna2Cbv;
@group(0) @binding(3) var<storage, read> sbuf0: array<u32>;
@group(0) @binding(4) var<storage, read_write> uav0: array<u32>;
@group(0) @binding(5) var<storage, read> unres0: array<u32>;
@group(0) @binding(6) var<storage, read> unres1: array<u32>;
@group(0) @binding(7) var tex0: texture_2d<f32>;
@group(0) @binding(8) var tex1: texture_2d<f32>;
@group(0) @binding(9) var samp0: sampler;
@group(0) @binding(10) var samp1: sampler;
@group(0) @binding(11) var uav_tex_dead: texture_storage_2d<rgba8unorm, write>;
@group(3) @binding(0) var<storage, read_write> ms_out: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let a = cb0.data[0].x + cb1.data[7].x;
    let b = sbuf0[0] + unres0[0] + unres1[0];
    let c = textureSampleLevel(tex0, samp0, vec2<f32>(0.5, 0.5), 0.0);
    let d = textureSampleLevel(tex1, samp1, vec2<f32>(0.5, 0.5), 0.0);
    uav0[0] = a + b + bitcast<u32>(c.x + d.x);
}
"#;

    let mut module = parse_and_validate_wgsl(src);
    let heap_zero = CbvIndexedHeapSource {
        cbv: CbvRefSource::RootDescriptor { user_sgpr: 6 },
        byte_offset: 8,
        shift: 5,
        bit_offset: 0,
        mask: u32::MAX,
        stride: 0,
    };
    let heap_pow2 = CbvIndexedHeapSource {
        stride: 32,
        ..heap_zero
    };

    let mut bindings = vec![
        WgslBindingInfo {
            group: 0,
            binding: 0,
            kind: "uniform",
            name: "cb0".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: Some(64),
            highest_vec4_offset: Some(3),
            source: Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0,
            }),
        },
        WgslBindingInfo {
            group: 0,
            binding: 1,
            kind: "uniform",
            name: "cb1".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: Some(128),
            highest_vec4_offset: Some(7),
            // RootDescriptor on known table SGPR 4 aliases Table(4, 0)!
            source: Some(WgslBindingSource::RootDescriptor {
                user_sgpr: 4,
                table_byte_offset: None,
            }),
        },
        WgslBindingInfo {
            group: 0,
            binding: 2,
            kind: "uniform",
            name: "cb_dead".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: Some(64),
            highest_vec4_offset: Some(0),
            source: Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 64,
            }),
        },
        WgslBindingInfo {
            group: 0,
            binding: 3,
            kind: "read-only-storage",
            name: "sbuf0".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::CbvIndexedResourceHeap(heap_zero)),
        },
        WgslBindingInfo {
            group: 0,
            binding: 4,
            kind: "storage",
            name: "uav0".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::CbvIndexedResourceHeap(heap_pow2)),
        },
        WgslBindingInfo {
            group: 0,
            binding: 5,
            kind: "read-only-storage",
            name: "unres0".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: None,
        },
        WgslBindingInfo {
            group: 0,
            binding: 6,
            kind: "read-only-storage",
            name: "unres1".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: None,
        },
        WgslBindingInfo {
            group: 0,
            binding: 7,
            kind: "texture_2d",
            name: "tex0".to_string(),
            view_dimension: Some("2d"),
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 128,
            }),
        },
        WgslBindingInfo {
            group: 0,
            binding: 8,
            kind: "texture_2d",
            name: "tex1".to_string(),
            view_dimension: Some("2d"),
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 128,
            }),
        },
        WgslBindingInfo {
            group: 0,
            binding: 9,
            kind: "sampler",
            name: "samp0".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::InlineSampler {
                sdwords: [1, 2, 3, 4],
            }),
        },
        WgslBindingInfo {
            group: 0,
            binding: 10,
            kind: "sampler",
            name: "samp1".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::InlineSampler {
                sdwords: [1, 2, 3, 4],
            }),
        },
        WgslBindingInfo {
            group: 0,
            binding: 11,
            kind: "storage-texture",
            name: "uav_tex_dead".to_string(),
            view_dimension: Some("2d"),
            storage_format: Some(22),
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 256,
            }),
        },
    ];

    let folded = deduplicate_module_bindings(&mut module, &mut bindings);
    assert_eq!(
        folded, 4,
        "must fold cb1->cb0, uav0->sbuf0, tex1->tex0, samp1->samp0 (never unresolved)"
    );

    let pruned = prune_unused_module_bindings(&mut module, &mut bindings, &BTreeSet::new());
    assert_eq!(
        pruned, 2,
        "must prune cb_dead and uav_tex_dead while preserving @group(3) ms_out"
    );

    // Re-validate mutated IR and re-emit WGSL, then re-parse with Naga!
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    let info = validator
        .validate(&module)
        .expect("compacted Naga module must validate cleanly");
    let emitted = naga::back::wgsl::write_string(
        &module,
        &info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .expect("WGSL emission must succeed");
    let reparsed = parse_and_validate_wgsl(&emitted);

    // Surviving bindings: cb0 (upgraded to 128 B), sbuf0 (upgraded to writable "storage"),
    // unres0, unres1, tex0, samp0 = 6 group-0 bindings (+ ms_out in group 3).
    assert_eq!(bindings.len(), 6, "bindings: {bindings:?}");
    let cb0 = bindings.iter().find(|b| b.binding == 0).unwrap();
    assert_eq!(cb0.min_binding_size_bytes, Some(128));
    assert_eq!(cb0.highest_vec4_offset, Some(7));
    let merged_sbuf = bindings.iter().find(|b| b.binding == 3).unwrap();
    assert_eq!(
        merged_sbuf.kind, "storage",
        "folding read_write uav0 into read-only sbuf0 must upgrade kind to writable storage"
    );
    assert!(bindings.iter().any(|b| b.binding == 5 && b.source.is_none()));
    assert!(bindings.iter().any(|b| b.binding == 6 && b.source.is_none()));
    assert!(
        reparsed
            .global_variables
            .iter()
            .any(|(_, v)| v.binding == Some(naga::ResourceBinding { group: 3, binding: 0 })),
        "runtime-owned @group(3) ms_out must never be pruned"
    );
    let reparsed_g0: BTreeSet<(u32, u32)> = reparsed
        .global_variables
        .iter()
        .filter_map(|(_, v)| {
            let rb = v.binding?;
            (rb.group != 3).then_some((rb.group, rb.binding))
        })
        .collect();
    let refl_g0: BTreeSet<(u32, u32)> = bindings.iter().map(|b| (b.group, b.binding)).collect();
    assert_eq!(reparsed_g0, refl_g0);
}

// ===========================================================================
// 5. Excess uniform buffer (> 12) spillover & cross-kind CBV->storage folding
// ===========================================================================

#[test]
fn s74_excess_uniform_buffers_promote_to_read_only_storage_and_fit_d3d12_limits() {
    // Compute shader reading 15 distinct table CBVs (s4 + 0x00..0x1C0) + 2 storage buffers:
    //   12 CBVs stay var<uniform> (cb0 stays at @binding(0)), 3 excess CBVs promote to
    //   var<storage, read> -> 12 uniforms + 5 storage buffers, 100% within D3D12 (12 / 16)!
    let mut words = Vec::new();
    words.push(enc_sop1(0x03, 20, 4));
    words.push(enc_sop1(0x03, 21, 0x80));
    for i in 0u32..15 {
        words.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, (i * 32) as i32, false, false));
        words.extend_from_slice(&enc_smem(0x08, 30, 12, 0x7D, 0, false, false));
    }
    for i in 15u32..17 {
        words.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, (i * 32) as i32, false, false));
        words.extend_from_slice(&enc_mubuf(0x0C, 0, true, false, 0, 1, 6, 0x7D));
    }
    words.push(enc_sopp(0x01, 0));

    let out = translate_compute(&words);
    assert_wgsl_and_reflection_in_sync(&out);

    let audit = StageBindingLimitAudit::audit_wgsl_bindings(
        &out.bindings,
        out.rt_arena,
        out.mesh_output.is_some(),
        StageBindingLimits::webgpu_d3d12(),
    );
    assert!(
        audit.within_limits(),
        "promoted shader must fit within D3D12 limits: {audit:?}"
    );
    assert_eq!(audit.counts.uniform_buffers, 12);
    assert_eq!(audit.counts.storage_buffers, 5);
    assert_eq!(out.binding_pack.uniform_promoted_to_storage, 3);
    assert!(
        out.bindings
            .iter()
            .any(|b| b.binding == 0 && b.kind == "uniform" && b.name == "cb0"),
        "cb0 must stay uniform at @binding(0)"
    );
}

#[test]
fn s74_excess_uniform_buffers_merge_into_same_source_storage_buffers_without_spending_slots() {
    // Compute shader reading 14 distinct table CBVs (offsets 0..13 * 32), where slots 0 and 1
    // are ALSO read via buffer_load_dword -> cross-kind merge folds those 2 CBVs directly into
    // the 2 existing storage buffers (0 extra storage slots consumed!), bringing uniform buffers
    // from 14 -> 12 and keeping storage buffers at 2!
    let mut words = Vec::new();
    words.push(enc_sop1(0x03, 20, 4));
    words.push(enc_sop1(0x03, 21, 0x80));
    for i in 0u32..14 {
        words.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, (i * 32) as i32, false, false));
        words.extend_from_slice(&enc_smem(0x08, 30, 12, 0x7D, 0, false, false));
        if i < 2 {
            words.extend_from_slice(&enc_mubuf(0x0C, 0, true, false, 0, 1, 6, 0x7D));
        }
    }
    words.push(enc_sopp(0x01, 0));

    let out = translate_compute(&words);
    assert_wgsl_and_reflection_in_sync(&out);

    let counts = StageBindingCounts::from_wgsl_bindings(&out.bindings, out.rt_arena, false);
    assert_eq!(counts.uniform_buffers, 12);
    assert_eq!(counts.storage_buffers, 2);
    assert_eq!(out.binding_pack.cbv_merged_into_storage, 2);
    assert_eq!(out.binding_pack.uniform_promoted_to_storage, 0);
}

#[test]
fn s74_uniform_spillover_respects_storage_buffer_ceiling_of_16() {
    // Compute shader with 15 distinct CBVs + 14 distinct storage buffers:
    // Only 2 CBVs can promote into storage (14 + 2 = 16), never pushing storage > 16!
    let mut words = Vec::new();
    words.push(enc_sop1(0x03, 20, 4));
    words.push(enc_sop1(0x03, 21, 0x80));
    for i in 0u32..15 {
        words.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, (i * 32) as i32, false, false));
        words.extend_from_slice(&enc_smem(0x08, 30, 12, 0x7D, 0, false, false));
    }
    for i in 15u32..29 {
        words.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, (i * 32) as i32, false, false));
        words.extend_from_slice(&enc_mubuf(0x0C, 0, true, false, 0, 1, 6, 0x7D));
    }
    words.push(enc_sopp(0x01, 0));

    let out = translate_compute(&words);
    assert_wgsl_and_reflection_in_sync(&out);

    let audit = StageBindingLimitAudit::audit_wgsl_bindings(
        &out.bindings,
        out.rt_arena,
        out.mesh_output.is_some(),
        StageBindingLimits::webgpu_d3d12(),
    );
    assert_eq!(out.binding_pack.uniform_promoted_to_storage, 2);
    assert_eq!(audit.counts.storage_buffers, 16);
    assert_eq!(audit.counts.uniform_buffers, 13);
    assert!(!audit.within_limits());
    assert_eq!(audit.overflows.len(), 1);
    assert_eq!(audit.overflows[0].kind, StageBindingLimitKind::UniformBuffers);
    assert_eq!(audit.overflows[0].count, 13);
    assert_eq!(audit.overflows[0].limit, 12);
}

// ===========================================================================
// 6. Per-stage WebGPU/D3D12 limit audit, pack summary, sidecar & compat checks
// ===========================================================================

#[test]
fn s74_per_stage_d3d12_limit_audit_covers_all_five_resource_kinds_and_runtime_group3() {
    let d3d12 = StageBindingLimits::webgpu_d3d12();
    assert_eq!(d3d12.max_uniform_buffers, 12);
    assert_eq!(d3d12.max_storage_buffers, 16);
    assert_eq!(d3d12.max_storage_textures, 8);
    assert_eq!(d3d12.max_sampled_textures, 16);
    assert_eq!(d3d12.max_samplers, 16);

    let exact_boundary = StageBindingCounts {
        uniform_buffers: WEBGPU_D3D12_MAX_UNIFORM_BUFFERS_PER_STAGE,
        storage_buffers: WEBGPU_D3D12_MAX_STORAGE_BUFFERS_PER_STAGE,
        storage_textures: WEBGPU_D3D12_MAX_STORAGE_TEXTURES_PER_STAGE,
        sampled_textures: WEBGPU_D3D12_MAX_SAMPLED_TEXTURES_PER_STAGE,
        samplers: WEBGPU_D3D12_MAX_SAMPLERS_PER_STAGE,
    };
    let ok_audit = StageBindingLimitAudit::from_counts(exact_boundary, d3d12);
    assert!(ok_audit.within_limits());
    assert!(ok_audit.overflows.is_empty());

    let over_all_five = StageBindingCounts {
        uniform_buffers: 13,
        storage_buffers: 17,
        storage_textures: 9,
        sampled_textures: 17,
        samplers: 17,
    };
    let bad_audit = StageBindingLimitAudit::from_counts(over_all_five, d3d12);
    assert!(!bad_audit.within_limits());
    assert_eq!(bad_audit.overflows.len(), 5);
    assert_eq!(
        bad_audit
            .overflows
            .iter()
            .map(|o| (o.kind, o.kind.as_limit_name(), o.count, o.limit))
            .collect::<Vec<_>>(),
        vec![
            (
                StageBindingLimitKind::UniformBuffers,
                "maxUniformBuffersPerShaderStage",
                13,
                12
            ),
            (
                StageBindingLimitKind::StorageBuffers,
                "maxStorageBuffersPerShaderStage",
                17,
                16
            ),
            (
                StageBindingLimitKind::StorageTextures,
                "maxStorageTexturesPerShaderStage",
                9,
                8
            ),
            (
                StageBindingLimitKind::SampledTextures,
                "maxSampledTexturesPerShaderStage",
                17,
                16
            ),
            (
                StageBindingLimitKind::Samplers,
                "maxSamplersPerShaderStage",
                17,
                16
            ),
        ]
    );

    // Verify rt_arena (+6) and mesh_output (+1) storage buffer accounting:
    let ten_sbufs: Vec<WgslBindingInfo> = (0..10u32)
        .map(|i| WgslBindingInfo {
            group: 0,
            binding: i,
            kind: "read-only-storage",
            name: format!("sbuf{i}"),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: i * 32,
            }),
        })
        .collect();
    let rt_ok = StageBindingLimitAudit::audit_wgsl_bindings(&ten_sbufs, true, false, d3d12);
    assert_eq!(
        rt_ok.counts.storage_buffers,
        10 + RT_ARENA_STORAGE_BUFFER_COUNT
    );
    assert!(rt_ok.within_limits());

    let rt_and_mesh_over = StageBindingLimitAudit::audit_wgsl_bindings(&ten_sbufs, true, true, d3d12);
    assert_eq!(
        rt_and_mesh_over.counts.storage_buffers,
        10 + RT_ARENA_STORAGE_BUFFER_COUNT + MESH_OUTPUT_STORAGE_BUFFER_COUNT
    );
    assert!(!rt_and_mesh_over.within_limits());
    assert_eq!(
        rt_and_mesh_over.overflows[0].kind,
        StageBindingLimitKind::StorageBuffers
    );
}

#[test]
fn s74_pack_summary_sidecar_and_compat_audit_report_d3d12_limits_cleanly() {
    // Shader 1: 15 CBVs -> compacted via 3 uniform->storage promotions (within D3D12 limits!)
    let mut cs_promoted = Vec::new();
    cs_promoted.push(enc_sop1(0x03, 20, 4));
    cs_promoted.push(enc_sop1(0x03, 21, 0x80));
    for i in 0u32..15 {
        cs_promoted.extend_from_slice(&enc_smem(0x02, 24, 10, 0x7D, (i * 32) as i32, false, false));
        cs_promoted.extend_from_slice(&enc_smem(0x08, 30, 12, 0x7D, 0, false, false));
    }
    cs_promoted.push(enc_sopp(0x01, 0));

    // Shader 2: 10 distinct storage textures (> 8 D3D12 limit -> legitimately over-limit)
    let mut cs_over_stex = Vec::new();
    cs_over_stex.push(enc_sop1(0x03, 20, 4));
    cs_over_stex.push(enc_sop1(0x03, 21, 0x80));
    for i in 0u32..10 {
        cs_over_stex.extend_from_slice(&enc_smem(0x03, 24, 10, 0x7D, (i * 32) as i32, false, false));
        cs_over_stex.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 4, 6, 0));
    }
    cs_over_stex.push(enc_sopp(0x01, 0));

    let blobs = vec![
        Rdna2BlobInput::new("cs_0000000000000001.bin", words_to_bytes(&cs_promoted))
            .with_stage(Rdna2StageOverride::Compute),
        Rdna2BlobInput::new("cs_0000000000000002.bin", words_to_bytes(&cs_over_stex))
            .with_stage(Rdna2StageOverride::Compute),
    ];

    let packed = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
        .expect("pack_rdna2_blobs must succeed");
    let bl = packed.summary.binding_limits;
    assert_eq!(bl.stages_within_limits, 1);
    assert_eq!(bl.stages_over_limit, 1);
    assert_eq!(bl.over_uniform_buffers, 0);
    assert_eq!(bl.over_storage_buffers, 0);
    assert_eq!(bl.over_storage_textures, 1);
    assert_eq!(bl.over_sampled_textures, 0);
    assert_eq!(bl.over_samplers, 0);
    assert_eq!(bl.max_uniform_buffers, 12);
    assert_eq!(bl.max_storage_textures, 10);
    assert_eq!(bl.total_uniform_promoted_to_storage, 3);

    let json = packed.summary.to_json();
    assert!(json.contains("\"binding_limits\":"), "JSON:\n{json}");
    assert!(json.contains("\"stages_within_limits\": 1"), "JSON:\n{json}");
    assert!(json.contains("\"stages_over_limit\": 1"), "JSON:\n{json}");
    assert!(json.contains("\"over_storage_textures\": 1"), "JSON:\n{json}");
    assert!(
        json.contains("\"uniform_promoted_to_storage\": 3"),
        "JSON:\n{json}"
    );
    let text = packed.summary.to_text();
    assert!(text.contains("d3d12_limits_ok:   1 (over: 1,"), "TEXT:\n{text}");
    assert!(text.contains("ub_promoted=3"), "TEXT:\n{text}");

    // Sidecar audit matches pack summary 1-to-1:
    let sidecar = packed
        .summary
        .to_stage_reflection_sidecar(&SidecarBuildOptions::default());
    assert!(sidecar.validate().is_empty());
    let sidecar_audit = sidecar.audit_stage_binding_limits(StageBindingLimits::webgpu_d3d12());
    assert!(!sidecar_audit.all_within_limits());
    assert_eq!(sidecar_audit.per_stage_overflows.len(), 1);
    assert_eq!(sidecar_audit.summary.stages_within_limits, 1);
    assert_eq!(sidecar_audit.summary.over_storage_textures, 1);
    let over_stage = sidecar_audit
        .per_stage_overflows
        .get(&2u64)
        .expect("stage 2 must be flagged over storage textures");
    assert_eq!(
        over_stage.overflows[0].kind,
        StageBindingLimitKind::StorageTextures
    );
    assert_eq!(over_stage.overflows[0].count, 10);
    assert_eq!(over_stage.overflows[0].limit, 8);

    // Compat Naga module audit against WebGpuLimits::webgpu_d3d12():
    let over_entry = packed
        .entries
        .iter()
        .find(|e| e.hash == 2u64)
        .expect("packed entry 2");
    assert_eq!(over_entry.stage, ShaderStage::Compute);
    let over_mod = parse_and_validate_wgsl(&over_entry.wgsl);
    let compat_audit = audit_naga_module_with_limits(&over_mod, &WebGpuLimits::webgpu_d3d12());
    assert!(
        compat_audit
            .issues
            .iter()
            .any(|i| i.kind == CompatIssueKind::StorageTextureLimitExceeded),
        "compat D3D12 audit must flag StorageTextureLimitExceeded: {:?}",
        compat_audit.issues
    );

    let ok_entry = packed
        .entries
        .iter()
        .find(|e| e.hash == 1u64)
        .expect("packed entry 1");
    let ok_mod = parse_and_validate_wgsl(&ok_entry.wgsl);
    let ok_compat_audit = audit_naga_module_with_limits(&ok_mod, &WebGpuLimits::webgpu_d3d12());
    assert!(
        !ok_compat_audit.issues.iter().any(|i| matches!(
            i.kind,
            CompatIssueKind::UniformBufferLimitExceeded
                | CompatIssueKind::StorageBufferLimitExceeded
                | CompatIssueKind::StorageTextureLimitExceeded
                | CompatIssueKind::SampledTextureLimitExceeded
                | CompatIssueKind::SamplerLimitExceeded
        )),
        "promoted stage 1 must have zero D3D12 binding-limit warnings: {:?}",
        ok_compat_audit.issues
    );

    // Sanity check BindingPackStats helper methods:
    let mut stats = BindingPackStats {
        dead_bindings_pruned: 2,
        duplicate_bindings_folded: 3,
        cbv_merged_into_storage: 1,
        uniform_promoted_to_storage: 4,
    };
    assert_eq!(stats.total_reduced(), 6);
    assert!(!stats.is_zero());
    stats.merge_from(BindingPackStats {
        dead_bindings_pruned: 1,
        duplicate_bindings_folded: 0,
        cbv_merged_into_storage: 2,
        uniform_promoted_to_storage: 1,
    });
    assert_eq!(stats.dead_bindings_pruned, 3);
    assert_eq!(stats.cbv_merged_into_storage, 3);
    assert_eq!(stats.uniform_promoted_to_storage, 5);
}
