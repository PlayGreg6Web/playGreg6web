//! Integration tests for S65 — RT BVH Prelude Emission (#1674, Refs #482, #1653).

use naga::valid::{Capabilities, ValidationFlags, Validator};
use playgta6_shaders::rdna2::{
    Rdna2Detail, Rdna2Encoding, Rdna2InferredStage, analyze_program, decode_program, mimg_addr_vgpr,
};
use playgta6_shaders::rdna2_pack::{Rdna2BlobInput, Rdna2PackOptions, pack_rdna2_blobs};
use playgta6_shaders::rdna2_to_wgsl::{
    RT_PRELUDE_VERSION, Rdna2StageOverride, Rdna2TranslateOptions, translate_rdna2_to_wgsl,
};

/// Exact v1 runtime-owned RT WGSL prelude from `runtime/gpu/rt_bvh.cpp::prelude_wgsl()`.
const RUNTIME_PRELUDE_V1_WGSL: &str = r#"// pg6 runtime prelude, version 1 (rt_bvh.h kPreludeVersion). Group 3, bindings 16-21: contract C3.
@group(3) @binding(16) var<storage, read_write> pg6_arena0: array<u32>;
@group(3) @binding(17) var<storage, read_write> pg6_arena1: array<u32>;
@group(3) @binding(18) var<storage, read_write> pg6_arena2: array<u32>;
@group(3) @binding(19) var<storage, read_write> pg6_arena3: array<u32>;
@group(3) @binding(20) var<storage, read> pg6_arena_dir: array<u32>;
@group(3) @binding(21) var<storage, read_write> pg6_rt_stats: array<atomic<u32>>;

// Dword at a 40-bit address (x = low 32 bits, y = bits 39:32). Unmapped: counts a fault, reads 0.
fn pg6_load32(addr: vec2<u32>) -> u32 {
    let page = (addr.x >> 16u) | ((addr.y & 0xffu) << 16u);
    let t = pg6_arena_dir[page >> 12u];
    if (t == 0u) {
        atomicAdd(&pg6_rt_stats[0], 1u);
        return 0u;
    }
    let e = pg6_arena_dir[4096u + (t - 1u) * 4096u + (page & 0xfffu)];
    if ((e & 0x80000000u) == 0u) {
        atomicAdd(&pg6_rt_stats[0], 1u);
        return 0u;
    }
    let w = (e & 0x1fffffffu) * 16384u + ((addr.x & 0xffffu) >> 2u);
    switch ((e >> 29u) & 3u) {
        case 0u: { return pg6_arena0[w]; }
        case 1u: { return pg6_arena1[w]; }
        case 2u: { return pg6_arena2[w]; }
        default: { return pg6_arena3[w]; }
    }
}

fn pg6_add40(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var carry = 0u;
    if (lo < a.x) { carry = 1u; }
    return vec2<u32>(lo, (a.y + b.y + carry) & 0xffu);
}

fn pg6_addr_off(a: vec2<u32>, off: u32) -> vec2<u32> {
    return pg6_add40(a, vec2<u32>(off, 0u));
}

fn pg6_f(a: vec2<u32>, off: u32) -> f32 {
    return bitcast<f32>(pg6_load32(pg6_addr_off(a, off)));
}

// Slab test of one box: .x = 1.0 when entered, .y = the entry distance.
fn pg6_slab(mn: vec3<f32>, mx: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, extent: f32) -> vec2<f32> {
    var lo = 0.0;
    var hi = extent;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if (mn[a] > mx[a]) {
            return vec2<f32>(0.0, 0.0);
        }
        let t0 = (mn[a] - o[a]) * inv[a];
        let t1 = (mx[a] - o[a]) * inv[a];
        if (t0 != t0 || t1 != t1) {
            continue;
        }
        var a0 = t1;
        var a1 = t0;
        if (t0 < t1) {
            a0 = t0;
            a1 = t1;
        }
        if (a0 > lo) { lo = a0; }
        if (a1 < hi) { hi = a1; }
    }
    if (lo <= hi) {
        return vec2<f32>(1.0, lo);
    }
    return vec2<f32>(0.0, 0.0);
}

fn pg6_tri_miss() -> vec4<u32> {
    return vec4<u32>(0x7f800000u, 0x3f800000u, 0u, 0u);
}

fn pg6_vec_nan(v: vec3<f32>) -> bool {
    return v.x != v.x || v.y != v.y || v.z != v.z;
}

fn pg6_triangle(addr: vec2<u32>, kind: u32, o: vec3<f32>, d: vec3<f32>) -> vec4<u32> {
    var vi = vec3<u32>(0u, 1u, 2u);
    if (kind == 1u) { vi = vec3<u32>(1u, 3u, 2u); }
    if (kind == 2u) { vi = vec3<u32>(2u, 3u, 4u); }
    if (kind == 3u) { vi = vec3<u32>(2u, 4u, 0u); }
    var v: array<vec3<f32>, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let b = vi[i] * 12u;
        v[i] = vec3<f32>(pg6_f(addr, b), pg6_f(addr, b + 4u), pg6_f(addr, b + 8u)) - o;
    }
    var kz = 0u;
    var az = abs(d.x);
    if (abs(d.y) > az) { kz = 1u; az = abs(d.y); }
    if (abs(d.z) > az) { kz = 2u; }
    let dz = d[kz];
    if (dz == 0.0) {
        return pg6_tri_miss();
    }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    if (dz < 0.0) {
        let t = kx;
        kx = ky;
        ky = t;
    }
    let sx = d[kx] / dz;
    let sy = d[ky] / dz;
    let sz = 1.0 / dz;
    var px: array<f32, 3>;
    var py: array<f32, 3>;
    var pz: array<f32, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        px[i] = v[i][kx] - sx * v[i][kz];
        py[i] = v[i][ky] - sy * v[i][kz];
        pz[i] = sz * v[i][kz];
    }
    let u = px[2] * py[1] - py[2] * px[1];
    let vv = px[0] * py[2] - py[0] * px[2];
    let w = px[1] * py[0] - py[1] * px[0];
    if ((u < 0.0 || vv < 0.0 || w < 0.0) && (u > 0.0 || vv > 0.0 || w > 0.0)) {
        return pg6_tri_miss();
    }
    let det = u + vv + w;
    let t_num = u * pz[0] + vv * pz[1] + w * pz[2];
    if (det == 0.0 || det != det || t_num != t_num || u != u || vv != vv || w != w) {
        return pg6_tri_miss();
    }
    let sel = (pg6_load32(pg6_addr_off(addr, 60u)) >> (8u * kind)) & 0xffu;
    var wt = array<f32, 4>(u, vv, w, 0.0);
    return vec4<u32>(bitcast<u32>(t_num), bitcast<u32>(det), bitcast<u32>(wt[sel & 3u]),
                     bitcast<u32>(wt[(sel >> 2u) & 3u]));
}

// C4: the console's image_bvh(64)_intersect_ray node test. desc = the 128-bit BVH descriptor,
// node = the node pointer (x low, y high; high 0 for the 32-bit form).
fn pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>, dir: vec3<f32>,
                     inv_dir: vec3<f32>) -> vec4<u32> {
    let kind = node.x & 7u;
    if (kind >= 6u) {
        return pg6_tri_miss();
    }
    let nan_ray = extent != extent || pg6_vec_nan(origin) || pg6_vec_nan(dir) || pg6_vec_nan(inv_dir);
    // address = (base256 << 8) + ((ptr >> 3) << 6), 40 bits
    let base = vec2<u32>(desc.x << 8u, ((desc.y & 0xffu) << 8u) | (desc.x >> 24u));
    let idx = vec2<u32>((node.x >> 3u) | (node.y << 29u), node.y >> 3u);
    let off = vec2<u32>(idx.x << 6u, (idx.y << 6u) | (idx.x >> 26u));
    let addr = pg6_add40(base, off);
    if (kind < 4u) {
        if (nan_ray) {
            return pg6_tri_miss();
        }
        return pg6_triangle(addr, kind, origin, dir);
    }
    if (nan_ray) {
        return vec4<u32>(0xffffffffu);
    }
    var ptr: array<u32, 4>;
    var key: array<f32, 4>;
    var hit: array<bool, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        ptr[i] = pg6_load32(pg6_addr_off(addr, 4u * i));
        var mn: vec3<f32>;
        var mx: vec3<f32>;
        if (kind == 5u) {
            let b = 16u + 24u * i;
            mn = vec3<f32>(pg6_f(addr, b), pg6_f(addr, b + 4u), pg6_f(addr, b + 8u));
            mx = vec3<f32>(pg6_f(addr, b + 12u), pg6_f(addr, b + 16u), pg6_f(addr, b + 20u));
        } else {
            let b = 16u + 12u * i;
            let a = unpack2x16float(pg6_load32(pg6_addr_off(addr, b)));
            let bb = unpack2x16float(pg6_load32(pg6_addr_off(addr, b + 4u)));
            let c = unpack2x16float(pg6_load32(pg6_addr_off(addr, b + 8u)));
            mn = vec3<f32>(a.x, a.y, bb.x);
            mx = vec3<f32>(bb.y, c.x, c.y);
        }
        key[i] = 0.0;
        hit[i] = false;
        if (ptr[i] != 0xffffffffu) {
            let s = pg6_slab(mn, mx, origin, inv_dir, extent);
            hit[i] = s.x > 0.5;
            key[i] = s.y;
        }
    }
    var out = vec4<u32>(0xffffffffu);
    let sort = (desc.y >> 31u) != 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (!hit[i]) {
            continue;
        }
        var rank = i;
        if (sort) {
            rank = 0u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                if (hit[j] && (key[j] < key[i] || (key[j] == key[i] && j < i))) {
                    rank = rank + 1u;
                }
            }
        }
        out[rank] = ptr[i];
    }
    return out;
}
"#;

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

fn enc_mimg_full(
    op: u8,
    dmask: u8,
    dim: u8,
    r128: bool,
    a16: bool,
    nsize: u8,
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
        | (u32::from(nsize & 0x3) << 1)
        | u32::from((op >> 7) & 0x1);
    let w1 = (u32::from(a16) << 30)
        | (u32::from(ssamp & 0x1F) << 21)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
    [w0, w1]
}

fn enc_flat(seg: u8, op: u8, offset: i16, lds: bool, addr: u8, data: u8, saddr: u8, vdst: u8) -> [u32; 2] {
    let raw_off = if seg == 0 {
        (offset as u16) & 0x0FFF
    } else {
        (offset as u16) & 0x0FFF
    };
    let w0 = (0b110111u32 << 26)
        | (u32::from(op & 0x7F) << 18)
        | (u32::from(seg & 0x3) << 14)
        | (u32::from(lds) << 13)
        | u32::from(raw_off);
    let w1 = (u32::from(vdst) << 24)
        | (u32::from(saddr & 0x7F) << 16)
        | (u32::from(data) << 8)
        | u32::from(addr);
    [w0, w1]
}

fn assert_valid_with_runtime_prelude(wgsl: &str) {
    // Emitted WGSL MUST NOT declare the runtime-owned prelude bindings or helpers.
    for forbidden in [
        "pg6_arena0",
        "pg6_arena1",
        "pg6_arena2",
        "pg6_arena3",
        "pg6_arena_dir",
        "pg6_rt_stats",
        "fn pg6_load32(",
        "fn pg6_bvh_intersect(",
        "@group(3)",
    ] {
        assert!(
            !wgsl.contains(forbidden),
            "emitted RT WGSL must not declare runtime-owned `{forbidden}`:\n{wgsl}"
        );
    }

    let combined = format!("{RUNTIME_PRELUDE_V1_WGSL}\n{wgsl}");
    let module = naga::front::wgsl::parse_str(&combined)
        .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\n---\n{combined}", e.emit_to_string(&combined)));
    let mut validator = Validator::new(
        ValidationFlags::all(),
        Capabilities::SHADER_FLOAT16_IN_FLOAT32,
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("Naga validation failed:\n{}\n---\n{combined}", e.emit_to_string(&combined)));
}

#[test]
fn c1_decoder_maps_0xe6_and_0xe7_to_bvh_not_0xe0_or_0xe1() {
    let mut words = Vec::new();
    words.extend_from_slice(&enc_mimg_full(0xE6, 0xF, 0, true, false, 0, 2, 16, 2, 0));
    words.extend_from_slice(&enc_mimg_full(0xE7, 0xF, 0, true, true, 0, 4, 20, 3, 0));
    words.extend_from_slice(&enc_mimg_full(0xE0, 0xF, 0, true, false, 0, 0, 0, 0, 0));
    words.extend_from_slice(&enc_mimg_full(0xE1, 0xF, 0, true, false, 0, 0, 0, 0, 0));
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let insts = decode_program(&bytes);
    assert_eq!(insts.len(), 5);

    assert_eq!(insts[0].encoding, Rdna2Encoding::Mimg);
    assert_eq!(insts[0].mnemonic, "image_bvh_intersect_ray");
    assert_eq!(insts[1].encoding, Rdna2Encoding::Mimg);
    assert_eq!(insts[1].mnemonic, "image_bvh64_intersect_ray");
    // 0xE0 and 0xE1 are NOT BVH ops on GFX10.3.
    assert_eq!(insts[2].mnemonic, "mimg_op_0xe0");
    assert_eq!(insts[3].mnemonic, "mimg_op_0xe1");

    // Analyze single BVH instruction: compute stage, 4-dword srsrc, 4 vdata VGPRs, 11 addr VGPRs.
    let mut bvh_only = Vec::new();
    bvh_only.extend_from_slice(&enc_mimg_full(0xE6, 0xF, 0, false, false, 0, 2, 20, 3, 0));
    bvh_only.push(enc_sopp(0x01, 0));
    let summary = analyze_program(&words_to_bytes(&bvh_only));
    assert_eq!(summary.inferred_stage, Rdna2InferredStage::Compute);
    assert_eq!(summary.max_sgpr, Some(15)); // srsrc=3 -> s[12:15] (4 dwords even with r128=false)
    assert_eq!(summary.max_vgpr, Some(23)); // vdata=20..23, vaddr=2..12
    assert_eq!(summary.resource_refs.len(), 1);
    assert_eq!(summary.resource_refs[0].ssamp_sgpr, None);
}

#[test]
fn c1_decoder_captures_mimg_nsa_address_vgprs() {
    let mut words = Vec::new();
    // image_bvh_intersect_ray (0xE6) with nsize=3 (12 extra NSA address bytes)
    let mimg = enc_mimg_full(0xE6, 0xF, 0, true, false, 3, 10, 30, 1, 0);
    words.extend_from_slice(&mimg);
    words.push(u32::from_le_bytes([11, 12, 13, 14]));
    words.push(u32::from_le_bytes([15, 16, 17, 18]));
    words.push(u32::from_le_bytes([19, 20, 21, 0]));
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let insts = decode_program(&bytes);
    assert_eq!(insts.len(), 2);
    assert_eq!(insts[0].size, 20);
    match &insts[0].detail {
        Rdna2Detail::Mimg {
            op,
            nsize,
            vaddr,
            nsa_vaddrs,
            ..
        } => {
            assert_eq!(*op, 0xE6);
            assert_eq!(*nsize, 3);
            assert_eq!(*vaddr, 10);
            for i in 0..11u8 {
                assert_eq!(mimg_addr_vgpr(*vaddr, *nsize, nsa_vaddrs, i), Some(10 + i));
            }
        }
        other => panic!("expected Mimg, got {other:?}"),
    }
}

#[test]
fn lowers_bvh32_and_bvh64_both_full_and_a16_with_runtime_prelude() {
    for (op, a16, label) in [
        (0xE6u8, false, "bvh32_f32"),
        (0xE6u8, true, "bvh32_a16"),
        (0xE7u8, false, "bvh64_f32"),
        (0xE7u8, true, "bvh64_a16"),
    ] {
        let mut words = Vec::new();
        words.extend_from_slice(&enc_mimg_full(op, 0xF, 0, true, a16, 0, 0, 16, 2, 0));
        words.push(enc_sopp(0x01, 0));

        let out = translate_rdna2_to_wgsl(
            &words_to_bytes(&words),
            &Rdna2TranslateOptions {
                stage: Rdna2StageOverride::Compute,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{label} failed: {e}"));

        assert_eq!(out.unsupported_op_count, 0, "{label} had unsupported ops");
        assert!(out.rt_arena, "{label} must set rt_arena");
        assert_eq!(out.rt_prelude_version, Some(RT_PRELUDE_VERSION));
        assert_eq!(out.bvh_ops, 1, "{label} bvh_ops");
        assert!(!out.raw_address, "{label} has no FLAT/GLOBAL ops");
        assert!(!out.atomic64, "{label} has no atomic64");
        assert!(
            out.wgsl.contains("pg6_bvh_intersect("),
            "{label} must call pg6_bvh_intersect:\n{}",
            out.wgsl
        );
        if a16 {
            assert!(
                out.wgsl.contains("unpack2x16float("),
                "{label} A16 must unpack f16 halves:\n{}",
                out.wgsl
            );
        }

        let json = out.to_json();
        assert!(json.contains("\"rt_arena\": true"), "{label} json:\n{json}");
        assert!(
            json.contains("\"rt_prelude_version\": 1"),
            "{label} json:\n{json}"
        );
        assert!(json.contains("\"bvh_ops\": 1"), "{label} json:\n{json}");

        assert_valid_with_runtime_prelude(&out.wgsl);
    }
}

#[test]
fn lowers_bvh_with_nsa_non_contiguous_vaddrs() {
    let mut words = Vec::new();
    // Seed non-contiguous NSA operand v23 with runtime v0 (`local_id.x`) so store-to-load
    // forwarding propagates it directly into `pg6_bvh_intersect(...)` even after dead-local pruning.
    words.push((0b0111_111u32 << 25) | (23u32 << 17) | (0x01u32 << 9) | 256u32); // v_mov_b32 v23, v0
    // image_bvh64_intersect_ray (0xE7), a16=false (12 VGPRs), nsize=3 (vaddr + 12 extra bytes)
    let mimg = enc_mimg_full(0xE7, 0xF, 0, true, false, 3, 2, 24, 1, 0);
    words.extend_from_slice(&mimg);
    words.push(u32::from_le_bytes([3, 5, 7, 9]));
    words.push(u32::from_le_bytes([11, 13, 15, 17]));
    words.push(u32::from_le_bytes([19, 21, 23, 0]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_rdna2_to_wgsl(
        &words_to_bytes(&words),
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Compute,
            ..Default::default()
        },
    )
    .expect("NSA BVH64 should translate cleanly");

    assert_eq!(out.unsupported_op_count, 0);
    assert!(out.rt_arena);
    assert_eq!(out.bvh_ops, 1);
    assert!(out.wgsl.contains("pg6_bvh_intersect("));
    // Verify non-contiguous NSA VGPR v23 (`bitcast<f32>(local_id.x)`) was wired into `pg6_bvh_intersect`.
    assert!(
        out.wgsl.contains("local_id.x"),
        "non-contiguous NSA operand v23 must flow into pg6_bvh_intersect:\n{}",
        out.wgsl
    );
    assert_valid_with_runtime_prelude(&out.wgsl);
}

#[test]
fn lowers_flat_and_global_loads_to_pg6_load32_and_detects_atomic64() {
    let mut words = Vec::new();
    // global_load_dword v4, v[0:1], off offset:16
    words.extend_from_slice(&enc_flat(2, 0x0C, 16, false, 0, 0, 0x7F, 4));
    // global_load_dwordx4 v[8:11], v2, s[4:5] offset:-8
    words.extend_from_slice(&enc_flat(2, 0x0E, -8, false, 2, 0, 4, 8));
    // flat_load_dwordx2 v[12:13], v[0:1] offset:32
    words.extend_from_slice(&enc_flat(0, 0x0D, 32, false, 0, 0, 0x7F, 12));
    // global_load_ubyte v14, v[0:1], off offset:1
    words.extend_from_slice(&enc_flat(2, 0x08, 1, false, 0, 0, 0x7F, 14));
    // global_load_sbyte v15, v[0:1], off offset:3
    words.extend_from_slice(&enc_flat(2, 0x09, 3, false, 0, 0, 0x7F, 15));
    // global_load_ushort v16, v[0:1], off offset:2
    words.extend_from_slice(&enc_flat(2, 0x0A, 2, false, 0, 0, 0x7F, 16));
    // global_load_sshort v17, v[0:1], off offset:2
    words.extend_from_slice(&enc_flat(2, 0x0B, 2, false, 0, 0, 0x7F, 17));
    // global_load_dwordx3 v[20:22], v[0:1], off offset:0
    words.extend_from_slice(&enc_flat(2, 0x0F, 0, false, 0, 0, 0x7F, 20));
    words.push(enc_sopp(0x01, 0));

    let out = translate_rdna2_to_wgsl(
        &words_to_bytes(&words),
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Compute,
            ..Default::default()
        },
    )
    .expect("FLAT/GLOBAL loads should translate cleanly");

    assert_eq!(out.unsupported_op_count, 0);
    assert!(out.rt_arena);
    assert_eq!(out.rt_prelude_version, Some(RT_PRELUDE_VERSION));
    assert_eq!(out.bvh_ops, 0);
    assert!(out.raw_address);
    assert!(!out.atomic64);
    assert!(out.wgsl.contains("pg6_load32("), "wgsl:\n{}", out.wgsl);
    assert_valid_with_runtime_prelude(&out.wgsl);

    // 64-bit global atomic (0x52 = global_atomic_add_x2) remains unsupported and flags atomic64.
    let mut a64_words = Vec::new();
    a64_words.extend_from_slice(&enc_flat(2, 0x52, 0, false, 0, 2, 0x7F, 4));
    a64_words.push(enc_sopp(0x01, 0));
    let a64_out = translate_rdna2_to_wgsl(
        &words_to_bytes(&a64_words),
        &Rdna2TranslateOptions {
            stage: Rdna2StageOverride::Compute,
            ..Default::default()
        },
    )
    .expect("atomic64 fallback should still produce valid WGSL");
    assert_eq!(a64_out.unsupported_op_count, 1);
    assert!(a64_out.raw_address);
    assert!(a64_out.atomic64);
    assert!(!a64_out.rt_arena);
    assert_eq!(a64_out.rt_prelude_version, None);
    let a64_json = a64_out.to_json();
    assert!(a64_json.contains("\"raw_address\": true"));
    assert!(a64_json.contains("\"atomic64\": true"));
    assert!(!a64_json.contains("\"rt_arena\""));
}

#[test]
fn rdna2_pack_emits_rt_arena_metadata_only_for_rt_shaders() {
    let mut rt_words = Vec::new();
    rt_words.extend_from_slice(&enc_mimg_full(0xE6, 0xF, 0, true, false, 0, 0, 16, 1, 0));
    rt_words.extend_from_slice(&enc_flat(2, 0x0C, 0, false, 0, 0, 0x7F, 20));
    rt_words.push(enc_sopp(0x01, 0));

    let plain_words = vec![enc_sopp(0x00, 0), enc_sopp(0x01, 0)];

    let blobs = vec![
        Rdna2BlobInput::new("cs_rt.bin", words_to_bytes(&rt_words))
            .with_stage(Rdna2StageOverride::Compute)
            .with_hash(0x0000_0000_0000_0001),
        Rdna2BlobInput::new("cs_plain.bin", words_to_bytes(&plain_words))
            .with_stage(Rdna2StageOverride::Compute)
            .with_hash(0x0000_0000_0000_0002),
    ];

    let packed = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
        .expect("packing RT and plain shaders should succeed");
    assert_eq!(packed.summary.entries.len(), 2);
    assert_valid_with_runtime_prelude(&packed.entries[0].wgsl);
    assert!(packed.entries[0].wgsl.contains("pg6_bvh_intersect("));
    assert!(packed.entries[0].wgsl.contains("pg6_load32("));

    let rt_entry = &packed.summary.entries[0];
    assert!(rt_entry.rt_arena);
    assert_eq!(rt_entry.rt_prelude_version, Some(1));
    assert_eq!(rt_entry.bvh_ops, 1);
    assert!(rt_entry.raw_address);
    assert!(!rt_entry.atomic64);

    let plain_entry = &packed.summary.entries[1];
    assert!(!plain_entry.rt_arena);
    assert_eq!(plain_entry.rt_prelude_version, None);
    assert_eq!(plain_entry.bvh_ops, 0);
    assert!(!plain_entry.raw_address);
    assert!(!plain_entry.atomic64);

    let json = packed.summary.to_json();
    // RT entry has rt_arena + rt_prelude_version + bvh_ops + raw_address
    assert!(json.contains("\"rt_arena\": true"), "json:\n{json}");
    assert!(
        json.contains("\"rt_prelude_version\": 1"),
        "json:\n{json}"
    );
    assert!(json.contains("\"bvh_ops\": 1"), "json:\n{json}");
    assert!(json.contains("\"raw_address\": true"), "json:\n{json}");
    // Plain entry does NOT get rt_arena emitted (only 1 occurrence in the whole JSON).
    assert_eq!(json.matches("\"rt_arena\"").count(), 1, "json:\n{json}");
}
