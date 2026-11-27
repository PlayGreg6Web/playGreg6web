//! S67 (`#1676`, Refs `#482`, `#651`) — Provenance tests for writable compute UAVs
//! (`uav_buf*`, `uav_tex*` 2D / 2D-array / 3D) and bindless `texture_2d` / `sampler`
//! descriptor sources, plus negative soundness guards verifying zero false positives.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use playgta6_shaders::rdna2_to_wgsl::{
    CbvIndexedHeapSource, CbvRefSource, LaneIndexedHeapSource, MAX_LANE_INDEXED_WINDOW,
    Rdna2StageOverride, Rdna2TranslateOptions, Rdna2WgslOutput, WgslBindingInfo, WgslBindingSource,
    translate_rdna2_to_wgsl,
};

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn translate_as(words: &[u32], stage: Rdna2StageOverride) -> Rdna2WgslOutput {
    let bytes = words_to_bytes(words);
    let opts = Rdna2TranslateOptions {
        stage,
        ..Rdna2TranslateOptions::default()
    };
    translate_rdna2_to_wgsl(&bytes, &opts)
        .unwrap_or_else(|e| panic!("translation must succeed: {e}"))
}

fn assert_naga_wgsl_valid(wgsl: &str) {
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\nWGSL:\n{wgsl}", e.emit_to_string(wgsl)));
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    validator.validate(&module).unwrap_or_else(|e| {
        panic!(
            "WGSL validation failed:\n{}\nWGSL:\n{wgsl}",
            e.emit_to_string(wgsl)
        )
    });
}

/// Validate every emitted `WgslBindingSource` against the `tools/stage_reflection_sidecar.py`
/// v1/v2/v3 contract (`table_byte_offset % 32 == 0`, `byte_offset % 4 == 0`, `shift < 32`,
/// `bit_offset < 32`, `mask != 0`, `1 <= max_count <= 256`).
fn assert_sidecar_schema_valid(bindings: &[WgslBindingInfo]) {
    let check_cbv = |cbv: CbvRefSource| {
        if let CbvRefSource::Table {
            table_byte_offset, ..
        } = cbv
        {
            assert_eq!(
                table_byte_offset % 32,
                0,
                "CbvRefSource::Table table_byte_offset must be multiple of 32: {cbv:?}"
            );
        }
    };
    for b in bindings {
        let Some(src) = b.source else { continue };
        match src {
            WgslBindingSource::RootDescriptor {
                table_byte_offset: Some(tbo),
                ..
            } => {
                assert_eq!(
                    tbo % 32,
                    0,
                    "v2 RootDescriptor table_byte_offset must be multiple of 32: {src:?}"
                );
            }
            WgslBindingSource::SamplerHeap { index_shift, .. }
            | WgslBindingSource::ResourceHeap { index_shift, .. } => {
                assert!(index_shift < 32, "heap index_shift must be < 32: {src:?}");
            }
            WgslBindingSource::CbvIndexedResourceHeap(h)
            | WgslBindingSource::CbvIndexedSamplerHeap(h) => {
                check_cbv(h.cbv);
                assert_eq!(h.byte_offset % 4, 0, "byte_offset must be % 4 == 0: {h:?}");
                assert!(h.shift < 32, "shift must be < 32: {h:?}");
                assert!(h.bit_offset < 32, "bit_offset must be < 32: {h:?}");
                assert_ne!(h.mask, 0, "mask must be non-zero: {h:?}");
            }
            WgslBindingSource::LaneIndexedResourceHeap(h)
            | WgslBindingSource::LaneIndexedSamplerHeap(h) => {
                assert!(
                    (1..=MAX_LANE_INDEXED_WINDOW).contains(&h.max_count),
                    "max_count must be in 1..=256: {h:?}"
                );
                assert!(h.shift < 32, "shift must be < 32: {h:?}");
                if let Some(wb) = h.window_base {
                    check_cbv(wb.cbv);
                    assert_eq!(
                        wb.byte_offset % 4,
                        0,
                        "window_base byte_offset must be % 4 == 0: {wb:?}"
                    );
                    assert!(wb.bit_offset < 32, "window_base bit_offset < 32: {wb:?}");
                    assert_ne!(wb.mask, 0, "window_base mask != 0: {wb:?}");
                }
            }
            _ => {}
        }
    }
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
        | (u32::from(op & 0x7F) << 23)
        | (u32::from(sdst & 0x7F) << 16)
        | (u32::from(ssrc1) << 8)
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
    omod: u8,
    abs: u8,
    neg: u8,
) -> [u32; 2] {
    let w0 = (0b110101u32 << 26)
        | (u32::from(op & 0x3FF) << 16)
        | (u32::from(clamp) << 15)
        | (u32::from(abs & 0x7) << 8)
        | u32::from(vdst);
    let w1 = (u32::from(neg & 0x7) << 29)
        | (u32::from(omod & 0x3) << 27)
        | (u32::from(src2 & 0x1FF) << 18)
        | (u32::from(src1 & 0x1FF) << 9)
        | u32::from(src0 & 0x1FF);
    [w0, w1]
}

fn enc_vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    let w0 = (0b110101u32 << 26)
        | (u32::from(op & 0x3FF) << 16)
        | (u32::from(sdst & 0x7F) << 8)
        | u32::from(vdst);
    let w1 = (u32::from(src2 & 0x1FF) << 18)
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
fn enc_mimg_full(
    op: u8,
    dmask: u8,
    dim: u8,
    r128: bool,
    vaddr: u8,
    vdata: u8,
    srsrc: u8,
    ssamp: u8,
) -> [u32; 2] {
    let w0 = (0b111100u32 << 26)
        | (u32::from(op & 0x7F) << 18)
        | (u32::from(r128) << 15)
        | (u32::from(dmask & 0xF) << 8)
        | (u32::from(dim & 0x7) << 3)
        | u32::from((op >> 7) & 0x1);
    let w1 = (u32::from(ssamp & 0x1F) << 21)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
    [w0, w1]
}

#[allow(clippy::too_many_arguments)]
fn enc_mubuf_full(
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

#[allow(clippy::too_many_arguments)]
fn enc_mtbuf_full(
    op: u8,
    format: u8,
    offset: u16,
    offen: bool,
    idxen: bool,
    vaddr: u8,
    vdata: u8,
    srsrc: u8,
    soffset: u8,
) -> [u32; 2] {
    let w0 = (0b111010u32 << 26)
        | (u32::from(format & 0x7F) << 19)
        | (u32::from(op & 0x7) << 16)
        | (u32::from(idxen) << 13)
        | (u32::from(offen) << 12)
        | u32::from(offset & 0xFFF);
    let w1 = (u32::from(soffset) << 24)
        | (u32::from((op >> 3) & 0x1) << 21)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
    [w0, w1]
}

fn branch_simm(branch_byte_off: usize, target_byte_off: usize) -> u16 {
    let rel_dwords = ((target_byte_off as isize) - ((branch_byte_off + 4) as isize)) / 4;
    (rel_dwords as i16) as u16
}

// ===========================================================================
// Target 1: Compute UAV Buffers (`uav_buf*`: MUBUF store/atomic & MTBUF store)
// ===========================================================================

#[test]
fn s67_compute_uav_buf_table_patched_64bit_offset_root_synthesized_and_heaps() {
    // Compute shader (user data starts at s4) exercising all writable UAV buffer shapes:
    //   1. Table UAV V# @ s4+0x40 with 64-bit dynamic base offset (s_add_u32 + s_addc_u32 w/
    //      dynamic high word s1) and w2 num_records shrunk (s_sub_u32) -> `buffer_store_dword`
    //      recovers `Table { user_sgpr: 4, table_byte_offset: 0x40 }`.
    //   2. Root UAV V# from s[6:7] with dynamic offset (`s_add_u32 s24, s6, s0` + `s_addc_u32 s25, s7, 0`
    //      + `s_and_b32 0xffff` + `s_or_b32 0x00040000` stride + `s_mov_b32 w2/w3`) -> `buffer_atomic_add`
    //      recovers `RootDescriptor { user_sgpr: 6, table_byte_offset: None }`.
    //   3. Root UAV V# from s[8:9] -> `tbuffer_store_format_xyzw` (MTBUF)
    //      recovers `RootDescriptor { user_sgpr: 8, table_byte_offset: None }`.
    //   4. CBV-indexed heap UAV V# via spilled heap base (`s_load_dwordx2` from s4+0x20 -> TableDesc)
    //      + root CBV dword `s[10:11]+0x10` shifted << 5 -> `buffer_store_dword`
    //      recovers `CbvIndexedResourceHeap`.
    //   5. Lane-indexed heap UAV V# via `v_med3_u32(v0, 0, 15)` (max_count=16) + `v_mad_u32_u24`
    //      (v * 32 + s12 heap base) + `v_readfirstlane_b32` -> `buffer_store_dword`
    //      recovers `LaneIndexedResourceHeap { max_count: 16, shift: 5, stride: 32 }`.
    let mut words = Vec::new();
    // Table ptr s4 in s[48:49] (s49 = 0)
    words.push(enc_sop1(0x03, 48, 4));
    words.push(enc_sop1(0x03, 49, 0x80));

    // 1) Table UAV V# @ s4+0x40 -> s[32:35], dynamically offset 64-bit VA + shrunk num_records
    words.extend_from_slice(&enc_smem(0x02, 32, 24, 0x7D, 0x40, false, false));
    words.push(enc_sop2(0x00, 32, 32, 0)); // s_add_u32 s32, s32, s0 (dynamic lo!)
    words.push(enc_sop2(0x04, 33, 33, 1)); // s_addc_u32 s33, s33, s1 (dynamic hi!)
    words.push(enc_sop2(0x01, 34, 34, 0)); // s_sub_u32 s34, s34, s0 (shrink num_records)
    words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 1, 8, 0x7D)); // buffer_store_dword s[32:35]

    // 2) Dynamically offset Root UAV V# from s[6:7] -> s[36:39] -> buffer_atomic_add (0x34)
    words.push(enc_sop2(0x00, 36, 6, 0)); // s_add_u32 s36, s6, s0
    words.push(enc_sop2(0x04, 37, 7, 128)); // s_addc_u32 s37, s7, 0
    words.push(enc_sop2(0x0E, 37, 37, 0xFF)); // s_and_b32 s37, s37, 0x0000ffff
    words.push(0x0000_FFFF);
    words.push(enc_sop2(0x10, 37, 37, 0xFF)); // s_or_b32 s37, s37, 0x00040000 (stride=4)
    words.push(0x0004_0000);
    words.push(enc_sop1(0x03, 38, 0xFF)); // num_records = 0xFFFFFFFF
    words.push(0xFFFF_FFFF);
    words.push(enc_sop1(0x03, 39, 0xFF)); // rsrc3 = 0x00020000
    words.push(0x0002_0000);
    words.extend_from_slice(&enc_mubuf_full(0x34, 0, true, false, 0, 1, 9, 0x7D)); // buffer_atomic_add s[36:39]

    // 3) Synthesized Root UAV V# from s[8:9] -> s[40:43] -> tbuffer_store_format_xyzw (MTBUF 0x07)
    words.push(enc_sop2(0x00, 40, 8, 0)); // s_add_u32 s40, s8, s0
    words.push(enc_sop2(0x04, 41, 9, 1)); // s_addc_u32 s41, s9, s1 (dynamic hi carry!)
    words.push(enc_sopk(0x00, 42, 1024)); // s_movk_i32 s42, 1024 (ConstImmediate!)
    words.push(enc_sop1(0x03, 43, 0xFF));
    words.push(0x3101_4000);
    words.extend_from_slice(&enc_mtbuf_full(0x07, 4, 0, true, false, 0, 1, 10, 0x7D)); // tbuffer_store_format_xyzw s[40:43]

    // 4) CBV-indexed heap UAV via spilled heap base (`s_load_dwordx2` from s4+0x20 -> TableDesc)
    words.extend_from_slice(&enc_smem(0x01, 50, 24, 0x7D, 0x20, false, false)); // s[50:51] = spilled heap base
    words.extend_from_slice(&enc_smem(0x00, 52, 5, 0x7D, 0x10, false, false)); // s52 = root CBV s[10:11]+0x10
    words.push(enc_sop2(0x1E, 52, 52, 128 + 5)); // s_lshl_b32 s52, s52, 5
    words.push(enc_sop2(0x00, 50, 50, 52)); // s_add_u32 s50, s50 (TableDesc!), s52 (ScaledCbvDword)
    words.push(enc_sop2(0x04, 51, 51, 128)); // s_addc_u32 s51, s51, 0
    words.extend_from_slice(&enc_smem(0x02, 56, 25, 0x7D, 0, false, false)); // V# -> s[56:59]
    words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 1, 14, 0x7D)); // buffer_store_dword s[56:59]

    // 5) Lane-indexed heap UAV via v_med3_u32(v0, 0, 15) + v_mad_u32_u24(v, 32, s12) + v_readfirstlane_b32
    words.extend_from_slice(&enc_vop3(0x159, 4, 256 + 0, 128, 128 + 15, false, 0, 0, 0)); // v_med3_u32 v4, v0, 0, 15
    words.extend_from_slice(&enc_vop3(0x143, 5, 256 + 4, 128 + 32, 12, false, 0, 0, 0)); // v_mad_u32_u24 v5, v4, 32, s12
    words.push(enc_vop1(0x02, 60, 256 + 5)); // v_readfirstlane_b32 s60, v5
    words.push(enc_sop1(0x03, 61, 0x80)); // s61 = 0
    words.extend_from_slice(&enc_smem(0x02, 64, 30, 0x7D, 0, false, false)); // V# -> s[64:67]
    words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 1, 16, 0x7D)); // buffer_store_dword s[64:67]

    words.push(enc_sopp(0x01, 0));

    let out = translate_as(&words, Rdna2StageOverride::Compute);
    assert!(out.naga_validated);
    assert_naga_wgsl_valid(&out.wgsl);
    assert_sidecar_schema_valid(&out.bindings);

    let storage_bindings: Vec<_> = out
        .bindings
        .iter()
        .filter(|b| b.kind == "storage")
        .collect();
    assert_eq!(
        storage_bindings.len(),
        5,
        "expected 5 writable UAV storage buffers, got: {:?}",
        out.bindings
    );
    assert!(
        storage_bindings.iter().all(|b| b.source.is_some()),
        "every writable compute UAV buffer must have a reflected source: {:?}",
        storage_bindings
    );

    let sources: Vec<_> = storage_bindings.iter().filter_map(|b| b.source).collect();
    assert!(
        sources.contains(&WgslBindingSource::Table {
            user_sgpr: 4,
            table_byte_offset: 0x40,
        }),
        "missing Table(4, 0x40) UAV: {sources:?}"
    );
    assert!(
        sources.contains(&WgslBindingSource::RootDescriptor {
            user_sgpr: 6,
            table_byte_offset: None,
        }),
        "missing RootDescriptor(6) atomic UAV: {sources:?}"
    );
    assert!(
        sources.contains(&WgslBindingSource::RootDescriptor {
            user_sgpr: 8,
            table_byte_offset: None,
        }),
        "missing RootDescriptor(8) MTBUF UAV: {sources:?}"
    );
    assert!(
        sources.contains(&WgslBindingSource::CbvIndexedResourceHeap(
            CbvIndexedHeapSource {
                cbv: CbvRefSource::RootDescriptor { user_sgpr: 10 },
                byte_offset: 0x10,
                shift: 5,
                bit_offset: 0,
                mask: u32::MAX,
                stride: 32,
            }
        )),
        "missing spilled-base CbvIndexedResourceHeap UAV: {sources:?}"
    );
    assert!(
        sources.contains(&WgslBindingSource::LaneIndexedResourceHeap(
            LaneIndexedHeapSource {
                max_count: 16,
                shift: 5,
                stride: 32,
                window_base: None,
            }
        )),
        "missing v_med3_u32 + v_mad_u32_u24 LaneIndexedResourceHeap UAV: {sources:?}"
    );
}

// ===========================================================================
// Target 1: Compute UAV Storage Textures (`uav_tex*` 2D, 2D-array, 3D)
// ===========================================================================

#[test]
fn s67_compute_uav_tex_2d_2darray_3d_patched_r128_two_level_and_heaps() {
    // Compute shader covering writable `image_store` UAVs:
    //   1. 2D `image_store` (dim=1) from table s4+0x40 with w0 patched (`s_or_b32`), w3 patched
    //      (`s_and_b32`), w6 DCC bit cleared (`s_bitset0_b32`) -> `Table(4, 0x40)`.
    //   2. 2D-array `image_store` (dim=5) using `r128=true` (4-dword T#) from table s4+0x80
    //      with w0 clobbered and w3 patched (`s_bitset1_b32`) -> `Table(4, 0x80)`.
    //   3. 3D `image_store` (dim=2) via 2-level nested table pointer (`s_load_dwordx2` s4+0x20 ->
    //      `s_load_dwordx8` +0x40) -> `Table(4, 0x60)`.
    //   4. 2D-array `image_store` (dim=5) from `CbvIndexedResourceHeap` (table CBV `s4+0x00`
    //      V# -> `s_buffer_load_dwordx4` dword 2 (+0x08) -> `v_bfe_u32` -> `v_lshlrev_b32` << 5 ->
    //      `v_readfirstlane_b32` -> `s_load_dwordx8`).
    //   5. 3D `image_store` (dim=2) from `LaneIndexedResourceHeap` (spilled heap base `TableDesc`
    //      + `v_readfirstlane_b32` & 7 -> << 5 -> `s_add_u32` -> `s_load_dwordx8`).
    let mut words = Vec::new();
    words.push(enc_sop1(0x03, 32, 4)); // s32 = s4
    words.push(enc_sop1(0x03, 33, 0x80)); // s33 = 0

    // 1) Patched 8-dword T# 2D UAV @ s4+0x40 -> s[40:47]
    words.extend_from_slice(&enc_smem(0x03, 40, 16, 0x7D, 0x40, false, false));
    words.push(enc_sop2(0x10, 40, 40, 0xFF)); // s_or_b32 s40, s40, 0x80000000 (w0 patched!)
    words.push(0x8000_0000);
    words.push(enc_sop1(0x1B, 46, 128 + 28)); // s_bitset0_b32 s46, 28 (clear DCC bit on w6!)
    words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 1, false, 0, 4, 10, 0)); // image_store 2D s[40:47]

    // 2) r128=true 4-dword T# 2D-array UAV @ s4+0x80 -> s[48:51]
    words.extend_from_slice(&enc_smem(0x02, 48, 16, 0x7D, 0x80, false, false));
    words.push(enc_sop1(0x03, 48, 0)); // clobber w0 with dynamic s0!
    words.push(enc_sop1(0x1D, 51, 128 + 29)); // s_bitset1_b32 s51, 29 (patch w3!)
    words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 5, true, 0, 4, 12, 0)); // image_store 2D-array r128 s[48:51]

    // 3) 2-level nested pointer -> 3D UAV @ s4+0x20 (+0x40 -> 0x60) -> s[56:63]
    words.extend_from_slice(&enc_smem(0x01, 34, 16, 0x7D, 0x20, false, false));
    words.extend_from_slice(&enc_smem(0x03, 56, 17, 0x7D, 0x40, false, false));
    words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 2, false, 0, 4, 14, 0)); // image_store 3D s[56:63]

    // 4) CBV-indexed 2D-array UAV via multi-dword CBV load + VALU bfe/shift + v_readfirstlane_b32
    words.extend_from_slice(&enc_smem(0x02, 64, 16, 0x7D, 0x00, false, false)); // CBV V# @ s4+0x00 -> s[64:67]
    words.extend_from_slice(&enc_smem(0x0A, 68, 32, 0x7D, 0x00, false, false)); // s_buffer_load_dwordx4 s[68:71] (s70 = +0x08)
    words.extend_from_slice(&enc_vop3(0x148, 10, 70, 128 + 8, 128 + 12, false, 0, 0, 0)); // v_bfe_u32 v10, s70, 8, 12
    words.push(enc_vop2(0x1A, 10, 128 + 5, 10)); // v_lshlrev_b32 v10, 5, v10
    words.push(enc_vop1(0x02, 72, 256 + 10)); // v_readfirstlane_b32 s72, v10
    words.push(enc_sop1(0x03, 36, 5)); // s36 = s5 (resource heap base)
    words.push(enc_sop1(0x03, 37, 0x80)); // s37 = 0
    words.extend_from_slice(&enc_smem(0x03, 80, 18, 72, 0, false, false)); // T# -> s[80:87]
    words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 5, false, 0, 4, 20, 0)); // image_store 2D-array s[80:87]

    // 5) Lane-indexed 3D UAV via spilled heap base (`s[34:35]` TableDesc!) + masked lane index
    words.push(enc_vop1(0x02, 73, 256 + 0)); // v_readfirstlane_b32 s73, v0
    words.push(enc_sop2(0x0E, 73, 73, 128 + 7)); // s_and_b32 s73, s73, 7 -> max_count=8
    words.push(enc_sop2(0x1E, 73, 73, 128 + 5)); // s_lshl_b32 s73, s73, 5
    words.push(enc_sop2(0x00, 74, 34, 73)); // s_add_u32 s74, s34 (TableDesc!), s73 (ScaledLaneIndex)
    words.push(enc_sop2(0x04, 75, 35, 128)); // s_addc_u32 s75, s35, 0
    words.extend_from_slice(&enc_smem(0x03, 88, 37, 0x7D, 0, false, false)); // T# -> s[88:95]
    words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 2, false, 0, 4, 22, 0)); // image_store 3D s[88:95]

    words.push(enc_sopp(0x01, 0));

    let out = translate_as(&words, Rdna2StageOverride::Compute);
    assert!(out.naga_validated);
    assert_naga_wgsl_valid(&out.wgsl);
    assert_sidecar_schema_valid(&out.bindings);

    let stex: Vec<_> = out
        .bindings
        .iter()
        .filter(|b| b.kind == "storage-texture")
        .collect();
    assert_eq!(stex.len(), 5, "expected 5 storage-texture UAVs: {:?}", out.bindings);
    assert!(
        stex.iter().all(|b| b.source.is_some()),
        "every storage-texture UAV must have a source: {:?}",
        stex
    );

    let has = |dim: &str, expected: WgslBindingSource| {
        stex.iter()
            .any(|b| b.view_dimension == Some(dim) && b.source == Some(expected))
    };
    assert!(
        has(
            "2d",
            WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0x40
            }
        ),
        "missing 2D patched UAV: {stex:?}"
    );
    assert!(
        has(
            "2d-array",
            WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0x80
            }
        ),
        "missing 2D-array r128 UAV: {stex:?}"
    );
    assert!(
        has(
            "3d",
            WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0x60
            }
        ),
        "missing 3D 2-level UAV: {stex:?}"
    );
    assert!(
        has(
            "2d-array",
            WgslBindingSource::CbvIndexedResourceHeap(CbvIndexedHeapSource {
                cbv: CbvRefSource::Table {
                    user_sgpr: 4,
                    table_byte_offset: 0,
                },
                byte_offset: 8,
                shift: 5,
                bit_offset: 8,
                mask: 0xFFF,
                stride: 32,
            })
        ),
        "missing 2D-array CBV-indexed UAV: {stex:?}"
    );
    assert!(
        has(
            "3d",
            WgslBindingSource::LaneIndexedResourceHeap(LaneIndexedHeapSource {
                max_count: 8,
                shift: 5,
                stride: 32,
                window_base: None,
            })
        ),
        "missing 3D lane-indexed UAV: {stex:?}"
    );
}

#[test]
fn s67_guarded_low_scratch_sgpr_uav_and_lane_spill_across_cfg_merge() {
    // Exercises two CFG-join fixes:
    //   1. Guarded conditional load of UAV T# into LOW SGPRs `s[8:15]` (which are scratch,
    //      never read as user SGPRs) across `s_cbranch_scc1 skip` -> survives merge!
    //   2. Guarded conditional load of UAV V# into `s[16:19]` across `s_cbranch_scc1 skip`
    //      -> survives merge!
    //   3. One-sided `v_writelane_b32` spill of UAV T# words into an otherwise untouched VGPR
    //      `v20` inside a guarded branch -> `v_readlane_b32` after merge restores `Table(4, 0x80)`!
    let mut words = Vec::new();
    words.push(enc_sop1(0x03, 32, 4)); // s32 = s4 (table ptr)
    words.push(enc_sop1(0x03, 33, 0x80)); // s33 = 0

    // Guarded branch loading s[8:15] (T# @ s4+0x20), s[16:19] (V# @ s4+0x40),
    // and spilling T# @ s4+0x80 into v20 lanes 0..1 via v_writelane_b32.
    words.push(enc_sopc(0x06, 0, 128)); // s_cmp_eq_u32 s0, 0
    let cbranch_idx = words.len();
    words.push(0); // placeholder: s_cbranch_scc1 skip
    words.extend_from_slice(&enc_smem(0x03, 8, 16, 0x7D, 0x20, false, false)); // s[8:15] = T# @ s4+0x20
    words.extend_from_slice(&enc_smem(0x02, 16, 16, 0x7D, 0x40, false, false)); // s[16:19] = V# @ s4+0x40
    words.extend_from_slice(&enc_smem(0x03, 40, 16, 0x7D, 0x80, false, false)); // s[40:47] = T# @ s4+0x80
    words.extend_from_slice(&enc_vop3(0x361, 20, 40, 128 + 0, 0, false, 0, 0, 0)); // v_writelane_b32 v20, s40, 0
    words.extend_from_slice(&enc_vop3(0x361, 20, 41, 128 + 1, 0, false, 0, 0, 0)); // v_writelane_b32 v20, s41, 1
    let skip_off = words.len() * 4;
    words[cbranch_idx] = enc_sopp(0x05, branch_simm(cbranch_idx * 4, skip_off));

    // After CFG merge: consume s[8:15], s[16:19], and reload s[48:49] from v20 lanes 0..1
    words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 1, false, 0, 4, 2, 0)); // image_store s[8:15]
    words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 4, 4, 0x7D)); // buffer_store s[16:19]
    words.extend_from_slice(&enc_vop3(0x360, 48, 256 + 20, 128 + 0, 0, false, 0, 0, 0)); // v_readlane_b32 s48, v20, 0
    words.extend_from_slice(&enc_vop3(0x360, 49, 256 + 20, 128 + 1, 0, false, 0, 0, 0)); // v_readlane_b32 s49, v20, 1
    words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 1, false, 0, 4, 12, 0)); // image_store s[48:55]
    words.push(enc_sopp(0x01, 0));

    let out = translate_as(&words, Rdna2StageOverride::Compute);
    assert!(out.naga_validated);
    assert_naga_wgsl_valid(&out.wgsl);
    assert_sidecar_schema_valid(&out.bindings);
    assert!(
        out.bindings.iter().all(|b| b.source.is_some()),
        "guarded low-SGPR & lane-spilled UAVs must all resolve: {:?}",
        out.bindings
    );
    let sources: Vec<_> = out.bindings.iter().filter_map(|b| b.source).collect();
    assert!(sources.contains(&WgslBindingSource::Table {
        user_sgpr: 4,
        table_byte_offset: 0x20,
    }));
    assert!(sources.contains(&WgslBindingSource::Table {
        user_sgpr: 4,
        table_byte_offset: 0x40,
    }));
    assert!(sources.contains(&WgslBindingSource::Table {
        user_sgpr: 4,
        table_byte_offset: 0x80,
    }));
}

// ===========================================================================
// Target 2: Bindless `texture_2d` & `sampler` + Inline Samplers + Waterfalls
// ===========================================================================

#[test]
fn s67_bindless_texture_and_sampler_valu_roundtrip_64bit_shift_and_spilled_heap() {
    // Fragment shader (graphics user data starts at s8) testing bindless texture & sampler chains:
    //   1. Multi-dword CBV load (`s_buffer_load_dwordx4` from Table CBV s8+0x40, reading dword 3
    //      at +0x0C) -> `v_mul_u32_u24 v4, 32, s_dw3` -> `v_readfirstlane_b32` -> `s_load_dwordx8` T#
    //      -> `CbvIndexedResourceHeap { Table(8, 0x40), byte_offset: 12, shift: 5, stride: 32 }`.
    //   2. Root CBV `s[10:11]` dword at +0x14 -> `v_and_or_b32` (& 0xFF | 0) -> `v_lshlrev_b64` << 4
    //      -> `v_readfirstlane_b32` -> `s_load_dwordx4` S#
    //      -> `CbvIndexedSamplerHeap { RootDescriptor(10), byte_offset: 20, shift: 4, mask: 0xFF, stride: 16 }`.
    //   3. Root constant index `s14` -> `s_lshl_b64 s[40:41], s[14:15], 5` -> `s_add_u32` + `s_addc_u32`
    //      with spilled heap base (`s_load_dwordx2` from s8+0x20 -> `TableDesc`) -> `s_load_dwordx8` T#
    //      -> `CbvIndexedResourceHeap { RootConstants(14), byte_offset: 0, shift: 5, stride: 32 }`.
    //   4. `s_lshl4_add_u32` + `s_addc_u32` carry propagation on table pointer `s[8:9]` ->
    //      `s_load_dwordx4` S# at `s8 + 96` (`6 << 4`) -> `Table(8, 96)`.
    let mut words = Vec::new();
    // Interpolate UV v2, v3
    words.push(enc_vintrp(0, 2, 0, 0, 0));
    words.push(enc_vintrp(1, 2, 0, 0, 1));
    words.push(enc_vintrp(0, 3, 0, 1, 0));
    words.push(enc_vintrp(1, 3, 0, 1, 1));

    // Table ptr s8 in s[32:33], resource heap base s16 in s[34:35], sampler heap base s18 in s[36:37]
    words.push(enc_sop1(0x03, 32, 8));
    words.push(enc_sop1(0x03, 33, 0x80));
    words.push(enc_sop1(0x03, 34, 16));
    words.push(enc_sop1(0x03, 35, 0x80));
    words.push(enc_sop1(0x03, 36, 18));
    words.push(enc_sop1(0x03, 37, 0x80));

    // 1) Multi-dword CBV load (s8+0x40) -> dword 3 (s43, +0x0C) -> v_mul_u32_u24 * 32 -> v_readfirstlane_b32 -> T#
    words.extend_from_slice(&enc_smem(0x02, 44, 16, 0x7D, 0x40, false, false)); // CBV V# -> s[44:47]
    words.extend_from_slice(&enc_smem(0x0A, 40, 22, 0x7D, 0x00, false, false)); // s_buffer_load_dwordx4 s[40:43]
    words.push(enc_vop1(0x01, 4, 43)); // v_mov_b32 v4, s43
    words.push(enc_vop2(0x0B, 4, 128 + 32, 4)); // v_mul_u32_u24 v4, 32, v4
    words.push(enc_vop1(0x02, 48, 256 + 4)); // v_readfirstlane_b32 s48, v4
    words.extend_from_slice(&enc_smem(0x03, 56, 17, 48, 0, false, false)); // T#1 -> s[56:63]

    // 2) Root CBV s[10:11]+0x14 -> v_and_or_b32(& 0xFF | 0) -> v_lshlrev_b64 << 4 ->
    //    v_add_co_u32 (heap base s18) -> v_readfirstlane_b32 -> S#
    words.extend_from_slice(&enc_smem(0x00, 49, 5, 0x7D, 0x14, false, false)); // s49 = RootDescriptor(10) + 0x14
    words.extend_from_slice(&enc_vop3(0x36F, 6, 49, 0xFF, 128, false, 0, 0, 0)); // v_and_or_b32 v6, s49, 0xFF, 0
    words.push(0x0000_00FF);
    words.push(enc_vop1(0x01, 7, 128)); // v7 = 0
    words.extend_from_slice(&enc_vop3(0x2FF, 6, 128 + 4, 256 + 6, 0, false, 0, 0, 0)); // v_lshlrev_b64 v[6:7], 4, v[6:7]
    words.extend_from_slice(&enc_vop3b(0x30F, 6, 106, 18, 256 + 6, 0)); // v_add_co_u32 v6, vcc, s18, v6
    words.push(enc_vop1(0x02, 50, 256 + 6)); // v_readfirstlane_b32 s50, v6
    words.push(enc_sop1(0x03, 51, 0x80)); // s51 = 0
    words.extend_from_slice(&enc_smem(0x02, 64, 25, 0x7D, 0, false, false)); // S#1 -> s[64:67]

    // 3) Root constant s14 -> s_lshl_b64 << 5 + spilled heap base (s8+0x20 -> TableDesc) -> T#2
    words.extend_from_slice(&enc_smem(0x01, 52, 16, 0x7D, 0x20, false, false)); // s[52:53] = spilled heap base
    words.push(enc_sop1(0x03, 54, 14)); // s54 = s14 (RootConstants 14)
    words.push(enc_sop1(0x03, 55, 0x80)); // s55 = 0
    words.push(enc_sop2(0x1F, 54, 54, 128 + 5)); // s_lshl_b64 s[54:55], s[54:55], 5
    words.push(enc_sop2(0x00, 52, 52, 54)); // s_add_u32 s52, s52 (TableDesc!), s54 (ScaledUserSgpr)
    words.push(enc_sop2(0x04, 53, 53, 55)); // s_addc_u32 s53, s53, s55
    words.extend_from_slice(&enc_smem(0x03, 72, 26, 0x7D, 0, false, false)); // T#2 -> s[72:79]

    // 4) s_lshl4_add_u32 + s_addc_u32 on table pointer s[8:9] -> S#2 @ s8+96
    words.push(enc_sop2(0x31, 68, 128 + 6, 32)); // s_lshl4_add_u32 s68, 6, s32 (= s8 + 96)
    words.push(enc_sop2(0x04, 69, 33, 128)); // s_addc_u32 s69, s33, 0
    words.extend_from_slice(&enc_smem(0x02, 80, 34, 0x7D, 0, false, false)); // S#2 -> s[80:83]

    // Sample both texture/sampler pairs and export
    words.extend_from_slice(&enc_mimg_full(0x20, 0xF, 1, false, 2, 8, 14, 16)); // T#1 s[56:63], S#1 s[64:67]
    words.extend_from_slice(&enc_mimg_full(0x20, 0xF, 1, false, 2, 12, 18, 20)); // T#2 s[72:79], S#2 s[80:83]
    words.push(enc_vop2(0x03, 8, 256 + 8, 12));
    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [8, 9, 10, 11]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_as(&words, Rdna2StageOverride::Pixel);
    assert!(out.naga_validated);
    assert_naga_wgsl_valid(&out.wgsl);
    assert_sidecar_schema_valid(&out.bindings);
    assert!(
        out.bindings.iter().all(|b| b.source.is_some()),
        "every bindless texture & sampler must resolve: {:?}",
        out.bindings
    );

    let sources: Vec<_> = out.bindings.iter().filter_map(|b| b.source).collect();
    assert!(
        sources.contains(&WgslBindingSource::CbvIndexedResourceHeap(
            CbvIndexedHeapSource {
                cbv: CbvRefSource::Table {
                    user_sgpr: 8,
                    table_byte_offset: 0x40,
                },
                byte_offset: 12,
                shift: 5,
                bit_offset: 0,
                mask: u32::MAX,
                stride: 32,
            }
        )),
        "missing multi-dword VALU v_mul_u32_u24 CbvIndexedResourceHeap: {sources:?}"
    );
    assert!(
        sources.contains(&WgslBindingSource::CbvIndexedSamplerHeap(
            CbvIndexedHeapSource {
                cbv: CbvRefSource::RootDescriptor { user_sgpr: 10 },
                byte_offset: 0x14,
                shift: 4,
                bit_offset: 0,
                mask: 0xFF,
                stride: 16,
            }
        )),
        "missing v_and_or_b32 + v_lshlrev_b64 CbvIndexedSamplerHeap: {sources:?}"
    );
    assert!(
        sources.contains(&WgslBindingSource::CbvIndexedResourceHeap(
            CbvIndexedHeapSource {
                cbv: CbvRefSource::RootConstants { user_sgpr: 14 },
                byte_offset: 0,
                shift: 5,
                bit_offset: 0,
                mask: u32::MAX,
                stride: 32,
            }
        )),
        "missing s_lshl_b64 + spilled-base RootConstants CbvIndexedResourceHeap: {sources:?}"
    );
    assert!(
        sources.contains(&WgslBindingSource::Table {
            user_sgpr: 8,
            table_byte_offset: 96,
        }),
        "missing s_lshl4_add_u32 + s_addc_u32 Table(8, 96) sampler: {sources:?}"
    );
}

#[test]
fn s67_inline_sampler_const_fold_bitreplicate_valu_and_guarded_low_sgprs() {
    // Fragment shader building inline samplers using:
    //   1. `s_movk_i32` + `s_or_b32` + `s_lshl_b32` + `s_bitreplicate_b64_b32` + VALU const-fold
    //      (`v_or_b32` -> `v_readfirstlane_b32`) into guarded low scratch SGPRs `s[16:19]`!
    //   2. Waterfall loop (`v_readfirstlane_b32` + `v_max_i32(v, 0)` + `v_min_i32(v, 11)` -> max_count=12,
    //      `v_mul_u32_u24 * 24` -> stride=24 + `s_cbranch_execnz` backedge) loading T#.
    let mut words = Vec::new();
    words.push(enc_vintrp(0, 2, 0, 0, 0));
    words.push(enc_vintrp(1, 2, 0, 0, 1));
    words.push(enc_vintrp(0, 3, 0, 1, 0));
    words.push(enc_vintrp(1, 3, 0, 1, 1));

    // Resource heap base in s8 -> s[32:33]
    words.push(enc_sop1(0x03, 32, 8));
    words.push(enc_sop1(0x03, 33, 0x80));

    // Guarded conditional construction of inline sampler in low scratch SGPRs s[16:19]:
    //   s16 = (0x0012 << 4) | 0x2 = 0x122
    //   s17 = v_readfirstlane_b32(v_or_b32(0x10, 0x05)) = 0x15
    //   s[18:19] = s_bitreplicate_b64_b32(0x0001_0001) = (lo=0x0000_0003, hi=0x0000_0003)
    words.push(enc_sopc(0x06, 0, 128));
    let cbranch_idx = words.len();
    words.push(0); // s_cbranch_scc1 skip
    words.push(enc_sopk(0x00, 36, 0x0012)); // s36 = 0x12
    words.push(enc_sop2(0x1E, 36, 36, 128 + 4)); // s36 = 0x120
    words.push(enc_sop2(0x10, 16, 36, 128 + 2)); // s16 = 0x122
    words.push(enc_vop1(0x01, 10, 128 + 0x10)); // v10 = 0x10
    words.push(enc_vop2(0x1C, 10, 128 + 5, 10)); // v_or_b32 v10, 5, v10 -> 0x15
    words.push(enc_vop1(0x02, 17, 256 + 10)); // v_readfirstlane_b32 s17, v10 -> 0x15
    words.push(enc_sop1(0x3B, 18, 0xFF)); // s_bitreplicate_b64_b32 s[18:19], 0x0001_0001
    words.push(0x0001_0001);
    let skip_off = words.len() * 4;
    words[cbranch_idx] = enc_sopp(0x05, branch_simm(cbranch_idx * 4, skip_off));

    // Waterfall loop over clamped per-lane VGPR index (max(0, v2) then min(11, v) -> max_count=12)
    // with non-pow2 stride 24 via `v_mul_u32_u24`
    words.push(enc_sop1(0x03, 30, 126)); // s30 = exec_lo
    let loop_start = words.len();
    words.push(enc_vop2(0x12, 11, 128 + 0, 2)); // v_max_i32 v11, 0, v2
    words.push(enc_vop2(0x11, 11, 128 + 11, 11)); // v_min_i32 v11, 11, v11 -> max_count = 12
    words.push(enc_vop2(0x0B, 12, 128 + 24, 11)); // v_mul_u32_u24 v12, 24, v11 -> stride = 24
    words.push(enc_vop1(0x02, 38, 256 + 12)); // v_readfirstlane_b32 s38, v12
    words.push(enc_vopc(0x02, 38, 12)); // v_cmp_eq_u32 vcc_lo, s38, v12
    words.push(enc_sop1(0x3C, 31, 106)); // s_and_saveexec_b32 s31, vcc_lo
    words.extend_from_slice(&enc_smem(0x03, 40, 16, 38, 0, false, false)); // T# -> s[40:47]
    words.extend_from_slice(&enc_mimg_full(0x20, 0xF, 1, false, 2, 4, 10, 4)); // image_sample s[40:47], s[16:19]
    words.push(enc_sop2(0x12, 126, 126, 31)); // s_xor_b32 exec_lo, exec_lo, s31
    let branch_idx = words.len();
    let rel = (loop_start as i32) - ((branch_idx + 1) as i32);
    words.push(enc_sopp(0x09, (rel as i16) as u16)); // s_cbranch_execnz loop_start
    words.push(enc_sop1(0x03, 126, 30)); // restore exec_lo

    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_as(&words, Rdna2StageOverride::Pixel);
    assert!(out.naga_validated);
    assert_naga_wgsl_valid(&out.wgsl);
    assert_sidecar_schema_valid(&out.bindings);
    assert!(
        out.bindings.iter().all(|b| b.source.is_some()),
        "bindings must all resolve: {:?}",
        out.bindings
    );

    let samp = out.bindings.iter().find(|b| b.kind == "sampler").unwrap();
    assert_eq!(
        samp.source,
        Some(WgslBindingSource::InlineSampler {
            sdwords: [0x122, 0x15, 3, 3],
        }),
        "expected constant-folded inline sampler: {:?}",
        samp
    );
    let tex = out
        .bindings
        .iter()
        .find(|b| b.kind == "texture_2d")
        .unwrap();
    assert_eq!(
        tex.source,
        Some(WgslBindingSource::LaneIndexedResourceHeap(
            LaneIndexedHeapSource {
                max_count: 12,
                shift: 0,
                stride: 24,
                window_base: None,
            }
        )),
        "expected waterfall lane-indexed texture with max_count=12 stride=24: {:?}",
        tex
    );
}

// ===========================================================================
// Negative / Soundness Guards (ZERO false positives!)
// ===========================================================================

#[test]
fn s67_negative_untraceable_dynamic_table_r128_isolation_and_conflicting_branches() {
    // 1. Dynamically offset `UserTable` (`s_add_u32 s32, s4, s0`) dereferenced as `sbase` of
    //    `s_load_dwordx4` MUST NOT emit `Table(4, 0)` -> `source` must be `None`!
    {
        let mut words = Vec::new();
        words.push(enc_sop2(0x00, 32, 4, 0)); // s_add_u32 s32, s4, s0 (dynamic table offset!)
        words.push(enc_sop1(0x03, 33, 0x80)); // s33 = 0
        words.extend_from_slice(&enc_smem(0x02, 40, 16, 0x7D, 0, false, false)); // s_load_dwordx4 s[40:43], s[32:33], 0
        words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 1, 10, 0x7D)); // buffer_store_dword s[40:43]
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        let uav = out.bindings.iter().find(|b| b.kind == "storage").unwrap();
        assert_eq!(
            uav.source, None,
            "dynamically offset UserTable dereferenced as table sbase must stay source=None: {:?}",
            out.bindings
        );
    }

    // 2. Dynamically offset `UserTable(4)` in w0 of V# whose w1 is an unrelated dynamic `s1`
    //    (NOT `s5` and NOT `ConstImmediate`) MUST NOT emit `RootDescriptor(4)` -> `source == None`!
    {
        let mut words = Vec::new();
        words.push(enc_sop2(0x00, 32, 4, 0)); // w0 = s4 + s0
        words.push(enc_sop1(0x03, 33, 1)); // w1 = s1 (Unknown, NOT s5!)
        words.push(enc_sop1(0x03, 34, 0xFF));
        words.push(0xFFFF_FFFF);
        words.push(enc_sop1(0x03, 35, 0xFF));
        words.push(0x0002_0000);
        words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 1, 8, 0x7D));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        let uav = out.bindings.iter().find(|b| b.kind == "storage").unwrap();
        assert_eq!(
            uav.source, None,
            "synthesized V# with mismatched w1 must stay source=None: {:?}",
            out.bindings
        );
    }

    // 3. `r128=true` (4-dword T#) with clobbered `s[40:43]` and an UNRELATED valid descriptor in
    //    `s[44:47]` MUST NOT read words 4..7 -> `source` must be `None`!
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 32, 4));
        words.push(enc_sop1(0x03, 33, 0x80));
        // Put an unrelated descriptor in s[44:47] (words 4..7 of s[40:47])
        words.extend_from_slice(&enc_smem(0x02, 44, 16, 0x7D, 0x80, false, false));
        // Overwrite s[40:43] with Unknown system SGPR s0
        words.push(enc_sop1(0x03, 40, 0));
        words.push(enc_sop1(0x03, 41, 0));
        words.push(enc_sop1(0x03, 42, 0));
        words.push(enc_sop1(0x03, 43, 0));
        // r128=true only reads s[40:43]!
        words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 1, true, 0, 4, 10, 0));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        let stex = out
            .bindings
            .iter()
            .find(|b| b.kind == "storage-texture")
            .unwrap();
        assert_eq!(
            stex.source, None,
            "r128=true 4-dword T# must never bleed into adjacent s[44:47]: {:?}",
            out.bindings
        );
    }

    // 4a. 2-way if/else loading two static table slots from the SAME table (`s4+0x20` vs `s4+0x40`)
    //     into `s[40:47]` must lower to a runtime-predicated 2-way store declaring BOTH bindings
    //     with exact `Table` provenance (never collapsing to one arbitrary branch!).
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 32, 4));
        words.push(enc_sop1(0x03, 33, 0x80));
        words.push(enc_sopc(0x06, 0, 128));
        let cb_idx = words.len();
        words.push(0); // s_cbranch_scc0 ELSE
        words.extend_from_slice(&enc_smem(0x03, 40, 16, 0x7D, 0x20, false, false));
        let sb_idx = words.len();
        words.push(0); // s_branch MERGE
        let else_off = words.len() * 4;
        words.extend_from_slice(&enc_smem(0x03, 40, 16, 0x7D, 0x40, false, false));
        let merge_off = words.len() * 4;
        words[cb_idx] = enc_sopp(0x04, branch_simm(cb_idx * 4, else_off));
        words[sb_idx] = enc_sopp(0x02, branch_simm(sb_idx * 4, merge_off));
        words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 1, false, 0, 4, 10, 0));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        assert_sidecar_schema_valid(&out.bindings);
        let stexes: Vec<_> = out
            .bindings
            .iter()
            .filter(|b| b.kind == "storage-texture")
            .collect();
        assert_eq!(
            stexes.len(),
            2,
            "2-way static table diamond must declare both storage-texture bindings: {:?}",
            out.bindings
        );
        let sources: Vec<_> = stexes.iter().map(|b| b.source).collect();
        assert!(
            sources.contains(&Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0x20,
            })) && sources.contains(&Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0x40,
            })),
            "both Table(s4, 0x20) and Table(s4, 0x40) must be preserved: {sources:?}"
        );
        assert!(
            out.wgsl.contains("_pg6_desc_sel0"),
            "predicated 2-way store must emit runtime descriptor selector:\n{}",
            out.wgsl
        );
    }

    // 4b. Cross-table conflicting if/else (`s4+0x20` vs `s5+0x40`) MUST stay `source == None`
    //     (no arbitrary branch picking!).
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 32, 4));
        words.push(enc_sop1(0x03, 33, 0x80));
        words.push(enc_sop1(0x03, 34, 5));
        words.push(enc_sop1(0x03, 35, 0x80));
        words.push(enc_sopc(0x06, 0, 128));
        let cb_idx = words.len();
        words.push(0); // s_cbranch_scc0 ELSE
        words.extend_from_slice(&enc_smem(0x03, 40, 16, 0x7D, 0x20, false, false)); // s4+0x20
        let sb_idx = words.len();
        words.push(0); // s_branch MERGE
        let else_off = words.len() * 4;
        words.extend_from_slice(&enc_smem(0x03, 40, 17, 0x7D, 0x40, false, false)); // s5+0x40
        let merge_off = words.len() * 4;
        words[cb_idx] = enc_sopp(0x04, branch_simm(cb_idx * 4, else_off));
        words[sb_idx] = enc_sopp(0x02, branch_simm(sb_idx * 4, merge_off));
        words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 1, false, 0, 4, 10, 0));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        let stex = out
            .bindings
            .iter()
            .find(|b| b.kind == "storage-texture")
            .unwrap();
        assert_eq!(
            stex.source, None,
            "cross-table conflicting if/else must remain source=None: {:?}",
            out.bindings
        );
    }
}

#[test]
fn s67_vopc_sdwab_vcc_preservation_lane_add_va_pack_spack_and_37_compute_uav_patterns() {
    // 1. VOPC VOP_SDWAB (`src0 == 0xF9`, `w1[15] == 1`, `sdst = s64`) must NOT clobber `vcc`
    //    holding a 64-bit descriptor table pointer!
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 106, 4)); // s_mov_b32 vcc_lo, s4 (UserTable s4)
        words.push(enc_sop1(0x03, 107, 0x80)); // s_mov_b32 vcc_hi, 0
        // VOPC v_cmp_class_f32 (op=0x88, bit4=0) with src0=0xF9 (SDWA) + w1(sd=1, sdst=64)
        words.push(enc_vopc(0x88, 0xF9, 1));
        words.push((1u32 << 15) | (64u32 << 8)); // sd=1, sdst=s64
        // Subsequent s_load_dwordx4 using sbase=vcc (pair 53 = s[106:107]) at offset 0x400
        words.extend_from_slice(&enc_smem(0x02, 40, 53, 0x7D, 0x400, false, false));
        words.extend_from_slice(&enc_mubuf_full(0x0C, 0, true, false, 0, 1, 10, 0x7D));
        words.push(enc_sopp(0x01, 0));

        // Verify disassembler also prints `s64`, not `vcc`.
        let decoded = playgta6_shaders::rdna2::decode_program(&words_to_bytes(&words));
        assert!(
            decoded[2].operands.starts_with("s64,"),
            "VOP_SDWAB must decode sdst=s64: {:?}",
            decoded[2]
        );

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        assert_sidecar_schema_valid(&out.bindings);
        let sbuf = out
            .bindings
            .iter()
            .find(|b| b.kind == "read-only-storage")
            .unwrap();
        assert_eq!(
            sbuf.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0x400,
            }),
            "VOP_SDWAB targeting s64 must preserve vcc table provenance: {:?}",
            out.bindings
        );
    }

    // 2. `s_add_u32` combining `UserTable(s4, 0) + LaneIndex(window_base: None)` from
    //    `v_readfirstlane_b32` -> `LaneIndexedResourceHeap` (parity with SMEM soffset).
    {
        let mut words = Vec::new();
        // v_readfirstlane_b32 (VOP1 op=0x02) s30, v0
        words.push(enc_vop1(0x02, 30, 256));
        // s_add_u32 s32, s4, s30; s_mov_b32 s33, 0
        words.push(enc_sop2(0x00, 32, 4, 30));
        words.push(enc_sop1(0x03, 33, 0x80));
        // s_load_dwordx4 s[40:43], s[32:33], 0
        words.extend_from_slice(&enc_smem(0x02, 40, 16, 0x7D, 0, false, false));
        words.extend_from_slice(&enc_mubuf_full(0x0C, 0, true, false, 0, 1, 10, 0x7D));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        assert_sidecar_schema_valid(&out.bindings);
        let sbuf = out
            .bindings
            .iter()
            .find(|b| b.kind == "read-only-storage")
            .unwrap();
        assert!(
            matches!(sbuf.source, Some(WgslBindingSource::LaneIndexedResourceHeap(_))),
            "UserTable + LaneIndex(window_base: None) via s_add_u32 must resolve to LaneIndexedResourceHeap: {:?}",
            out.bindings
        );
    }

    // 3. 48-bit aligned CBV VA pack/unpack (`s_lshr_b64 8` -> `v_mov_b32` -> `v_readlane_b32` ->
    //    `s_lshl_b64 8` -> `s_or_b32 48` -> `s_load_dword`) preserves `Table` CBV source.
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 32, 4)); // s32 = UserTable s4
        words.push(enc_sop1(0x03, 33, 0x80)); // s33 = 0
        // s_load_dwordx4 s[40:43], s[32:33], 0x60 -> TableDesc(s4, 0x60)
        words.extend_from_slice(&enc_smem(0x02, 40, 16, 0x7D, 0x60, false, false));
        // s_mov_b32 s44, s40; s_and_b32 s45, s41, 0x0000ffff
        words.push(enc_sop1(0x03, 44, 40));
        words.push(enc_sop2(0x0E, 45, 41, 0xFF));
        words.push(0x0000_FFFF);
        // s_lshr_b64 (SOP2 0x21) s[46:47], s[44:45], 8 (inline 136)
        words.push(enc_sop2(0x21, 46, 44, 136));
        // Uniform VGPR spill round-trip: v_mov_b32 v5, s46; v_readlane_b32 (VOP3 0x360) s48, v5, s0
        words.push(enc_vop1(0x01, 5, 46));
        words.extend_from_slice(&enc_vop3(0x360, 48, 256 + 5, 0, 0, false, 0, 0, 0));
        // s_mov_b32 s49, 0; s_lshl_b64 (SOP2 0x1F) s[50:51], s[48:49], 8
        words.push(enc_sop1(0x03, 49, 0x80));
        words.push(enc_sop2(0x1F, 50, 48, 136));
        // s_or_b32 s50, s50, 48 (inline 176)
        words.push(enc_sop2(0x10, 50, 50, 176));
        // s_load_dword s52, s[50:51], 0x0 (scalar CBV read!)
        words.extend_from_slice(&enc_smem(0x00, 52, 25, 0x7D, 0, false, false));
        words.push(enc_vop1(0x01, 2, 52));
        words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 2, 10, 0x7D));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        assert_sidecar_schema_valid(&out.bindings);
        let cbv = out.bindings.iter().find(|b| b.kind == "uniform").unwrap();
        assert_eq!(
            cbv.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 4,
                table_byte_offset: 0x60,
            }),
            "48-bit VA s_lshr_b64/s_lshl_b64 round-trip must preserve Table CBV source: {:?}",
            out.bindings
        );
    }

    // 4. Non-constant `s_pack_ll_b32_b16` / `s_pack_hh_b32_b16` with zero upper half extracts
    //    16-bit CBV heap index cleanly (`CbvIndexedResourceHeap`).
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 32, 4)); // Table s4
        words.push(enc_sop1(0x03, 33, 0x80));
        words.extend_from_slice(&enc_smem(0x02, 40, 16, 0x7D, 0x20, false, false)); // CBV V#
        words.extend_from_slice(&enc_smem(0x08, 44, 20, 0x7D, 0x10, false, false)); // s_buffer_load_dword s44
        // s_pack_hh_b32_b16 (SOP2 0x34) s45, s44, 0 -> (s44 >> 16)
        words.push(enc_sop2(0x34, 45, 44, 0x80));
        // s_lshl_b32 s45, s45, 5 (* 32B T#)
        words.push(enc_sop2(0x1E, 45, 45, 133));
        words.push(enc_sop1(0x03, 34, 6)); // ResourceHeap UserTable s6
        words.push(enc_sop1(0x03, 35, 0x80));
        words.extend_from_slice(&enc_smem(0x03, 48, 17, 45, 0, false, false)); // s_load_dwordx8 T#
        words.extend_from_slice(&enc_mimg_full(0x00, 0xF, 1, false, 0, 4, 12, 0)); // image_load
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        assert_sidecar_schema_valid(&out.bindings);
        let tex = out.bindings.iter().find(|b| b.kind == "texture_2d").unwrap();
        assert_eq!(
            tex.source,
            Some(WgslBindingSource::CbvIndexedResourceHeap(
                CbvIndexedHeapSource {
                    cbv: CbvRefSource::Table {
                        user_sgpr: 4,
                        table_byte_offset: 0x20,
                    },
                    byte_offset: 0x10,
                    shift: 5,
                    bit_offset: 16,
                    mask: 0xFFFF,
                    stride: 32,
                }
            )),
            "s_pack_hh_b32_b16 dst, cbv_dword, 0 must resolve CbvIndexedResourceHeap: {:?}",
            out.bindings
        );
    }

    // 5. 2-way CFG diamond selecting between 2 static `TableDesc` slots for writable `buffer_store_dword`
    //    (`s4+0x300` vs `s4+0x320`) declares BOTH `storage` bindings with runtime selector dispatch.
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 32, 4));
        words.push(enc_sop1(0x03, 33, 0x80));
        words.push(enc_sopc(0x06, 0, 128));
        let cb_idx = words.len();
        words.push(0); // s_cbranch_scc0 ELSE
        words.extend_from_slice(&enc_smem(0x02, 40, 16, 0x7D, 0x300, false, false));
        let sb_idx = words.len();
        words.push(0); // s_branch MERGE
        let else_off = words.len() * 4;
        words.extend_from_slice(&enc_smem(0x02, 40, 16, 0x7D, 0x320, false, false));
        let merge_off = words.len() * 4;
        words[cb_idx] = enc_sopp(0x04, branch_simm(cb_idx * 4, else_off));
        words[sb_idx] = enc_sopp(0x02, branch_simm(sb_idx * 4, merge_off));
        words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 1, 10, 0x7D));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        assert_sidecar_schema_valid(&out.bindings);
        let uavs: Vec<_> = out.bindings.iter().filter(|b| b.kind == "storage").collect();
        assert_eq!(uavs.len(), 2, "expected 2 storage UAV bindings: {:?}", out.bindings);
        assert!(uavs.iter().all(|b| b.source.is_some()));
        assert!(out.wgsl.contains("_pg6_desc_sel0"), "WGSL:\n{}", out.wgsl);
    }

    // 6. Bounded dynamic-`soffset` CBV index array (`s_cmp_lt_u32 s0, 6` -> `s_lshl_b32 s_off, s0, 2`
    //    -> `s_buffer_load_dword` -> `s_lshl_b32 5` -> `s_load_dwordx8` -> `image_store`) expands
    //    into 6 `CbvIndexedResourceHeap` storage-texture bindings + `switch (_pg6_desc_sel0)`.
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 32, 4)); // Table s4
        words.push(enc_sop1(0x03, 33, 0x80));
        words.push(enc_sop1(0x03, 34, 6)); // ResourceHeap s6
        words.push(enc_sop1(0x03, 35, 0x80));
        words.extend_from_slice(&enc_smem(0x02, 40, 16, 0x7D, 0x20, false, false)); // CBV V#
        // s_cmp_lt_u32 (SOPC 0x0A) s0, 6 (inline 134)
        words.push(enc_sopc(0x0A, 0, 134));
        let guard_idx = words.len();
        words.push(0); // s_cbranch_scc0 EXIT
        // In-bounds path (s0 < 6):
        words.push(enc_sop2(0x1E, 44, 0, 130)); // s_lshl_b32 s44, s0, 2
        words.extend_from_slice(&enc_smem(0x08, 45, 20, 44, 0, false, false)); // s_buffer_load_dword s45, s[40:43], s44
        words.push(enc_sop2(0x1E, 45, 45, 133)); // s_lshl_b32 s45, s45, 5
        words.extend_from_slice(&enc_smem(0x03, 48, 17, 45, 0, false, false)); // s_load_dwordx8 s[48:55], s[34:35], s45
        words.extend_from_slice(&enc_mimg_full(0x08, 0xF, 1, false, 0, 4, 12, 0)); // image_store
        let exit_off = words.len() * 4;
        words[guard_idx] = enc_sopp(0x04, branch_simm(guard_idx * 4, exit_off));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        assert_sidecar_schema_valid(&out.bindings);
        let stexes: Vec<_> = out
            .bindings
            .iter()
            .filter(|b| b.kind == "storage-texture")
            .collect();
        assert_eq!(
            stexes.len(),
            6,
            "bounded idx < 6 CBV index array must emit 6 CbvIndexedResourceHeap UAVs: {:?}",
            out.bindings
        );
        for (k, b) in stexes.iter().enumerate() {
            assert_eq!(
                b.source,
                Some(WgslBindingSource::CbvIndexedResourceHeap(
                    CbvIndexedHeapSource {
                        cbv: CbvRefSource::Table {
                            user_sgpr: 4,
                            table_byte_offset: 0x20,
                        },
                        byte_offset: (k as u32) * 4,
                        shift: 5,
                        bit_offset: 0,
                        mask: u32::MAX,
                        stride: 32,
                    }
                ))
            );
        }
        assert!(
            out.wgsl.contains("switch ") && out.wgsl.contains("textureStore(uav_tex5_"),
            "WGSL:\n{}",
            out.wgsl
        );
    }

    // 7. Guarded `s_load_dwordx4` merged with stale `Unknown` scratch across `s_cbranch_execz`
    //    bypass resolves `Table(s5, 0x80)` for `buffer_atomic_add` cleanly.
    {
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 32, 0)); // s32 = s0 (Unknown stale scratch)
        words.push(enc_sop1(0x03, 33, 0)); // s33 = s0 (Unknown stale scratch)
        words.push(enc_sop1(0x03, 76, 5)); // s76 = UserTable s5
        words.push(enc_sop1(0x03, 77, 0x80)); // s77 = 0
        // s_cbranch_execz (SOPP 0x08) MERGE
        let cb_idx = words.len();
        words.push(0);
        // Active path: s_load_dwordx4 s[32:35], s[76:77], 0x80
        words.extend_from_slice(&enc_smem(0x02, 32, 38, 0x7D, 0x80, false, false));
        let merge_off = words.len() * 4;
        words[cb_idx] = enc_sopp(0x08, branch_simm(cb_idx * 4, merge_off));
        // buffer_atomic_add (op=0x32, glc=true) on s[32:35] (srsrc=8)
        words.extend_from_slice(&enc_mubuf_full(0x32, 0, true, false, 0, 4, 8, 0x7D));
        words.push(enc_sopp(0x01, 0));

        let out = translate_as(&words, Rdna2StageOverride::Compute);
        assert!(out.naga_validated);
        assert_naga_wgsl_valid(&out.wgsl);
        assert_sidecar_schema_valid(&out.bindings);
        let uav = out.bindings.iter().find(|b| b.kind == "storage").unwrap();
        assert_eq!(
            uav.source,
            Some(WgslBindingSource::Table {
                user_sgpr: 5,
                table_byte_offset: 0x80,
            }),
            "s_cbranch_execz bypass carrying stale Unknown scratch must not erase TableDesc: {:?}",
            out.bindings
        );
    }
}
