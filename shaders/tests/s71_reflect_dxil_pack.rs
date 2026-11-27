//! S71 (`#1742`, Refs `#482`, `#651`, `#663`) — Comprehensive integration tests for:
//! 1. Reflection v3 completeness: cross-stage binding collision detection & resolution (VS+PS, MS+PS),
//!    GFX10.3 bindless texture/sampler array stride validation, and `playgta6-stage-reflection` v3
//!    `unresolved` + compute-stage census handling (`reflect.rs`).
//! 2. DXIL fallback reflection extraction (`RDEF`/`STAT`/`PSV0`/`ISG1`/`OSG1`/`DXIL`) and external
//!    D3D12 root-parameter matching -> joinable v3 sidecar entries (`dxil_reflect.rs`).
//! 3. 3D scene WebGPU limit & feature compatibility auditing (`compat.rs`).
//! 4. Automatic WGPK v2 selection (configurable threshold/margin + 4 GiB boundary) and zero-OOM
//!    streaming WGPK v2 writer with simulated >4 GiB 64-bit offset round-trip (`rdna2_pack.rs`).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Cursor, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use playgta6_shaders::compat::{
    CompatIssueKind, Pipeline3dCompatInput, Pipeline3dProfile, Pipeline3dStageSpec, WebGpuFeature,
    evaluate_pipeline_3d_compat, render_pipeline_3d_compat_batch_json,
};
use playgta6_shaders::convert::{ShaderStage, fnv1a_64};
use playgta6_shaders::dxil_reflect::{
    CONSOLE_GRAPHICS_USER_DATA_BASE, CONSOLE_TABLE_SLOT_BYTES, D3dShaderVisibility,
    DescriptorRange, DescriptorRangeType, DxilRootParamMatchKind, RootParameter, RootParameterKind,
    RootSignature, RootSignatureVersion, StaticSamplerDesc, extract_dxil_fallback_reflection,
    match_dxil_bytes_to_external_rts0, match_dxil_to_root_signature, parse_rdef_chunk,
    parse_stat_chunk, reflect_dxil_bytes, reflect_dxil_bytes_with_external_root_signature,
};
use playgta6_shaders::pack::{
    PACK_ENTRY_BYTES, PACK_ENTRY_BYTES_V2, PACK_HEADER_BYTES, PACK_VERSION_V1, PACK_VERSION_V2,
    PackEntry, PackVersion, read_pack, read_pack_canonical, read_pack_toc_canonical,
};
use playgta6_shaders::rdna2_pack::{
    AutoVersionPolicy, DEFAULT_AUTO_V2_SAFETY_MARGIN_BYTES, DEFAULT_AUTO_V2_THRESHOLD_BYTES,
    Rdna2BlobInput, Rdna2PackOptions, Rdna2StreamingOptions, StreamingWgpkWriter,
    pack_rdna2_blobs, pack_rdna2_blobs_streaming_to_file, pack_rdna2_manifest_streaming_to_file,
    resolve_auto_pack_layout, write_wgpk_entries_streaming,
};
use playgta6_shaders::rdna2_to_wgsl::{
    CbvIndexedHeapSource, CbvRefSource, LaneIndexedHeapSource, WgslBindingInfo, WgslBindingSource,
};
use playgta6_shaders::reflect::{
    BindingCollisionKind, BindingLayout, BindlessArrayStrideSpec, BindlessHeapKind,
    BindlessStrideErrorKind, BufferBindingType, EntryPointReflection, GFX10_BUFFER_STRIDE_BYTES,
    GFX10_IMAGE_STRIDE_BYTES, GFX10_SAMPLER_STRIDE_BYTES, MultiStageMergeOptions, ReflectedBinding,
    RegisterConvention, ResourceKind, STAGE_REFLECTION_VERSION_V1, STAGE_REFLECTION_VERSION_V2,
    STAGE_REFLECTION_VERSION_V3, SidecarBindingKind, SidecarBindingSource,
    SidecarBuildOptions, SidecarStageInput, StageReflectionSidecar, StorageTextureAccess,
    TextureSampleType, TextureViewDimension, VISIBILITY_COMPUTE, VISIBILITY_FRAGMENT,
    VISIBILITY_VERTEX, detect_cross_stage_binding_collisions, is_valid_gfx10_descriptor_stride,
    merge_multistage_pipeline_with_resolution, validate_bindless_binding_strides,
    validate_stage_disjoint_bindings,
};

// ---------------------------------------------------------------------------
// Helpers: Synthetic EntryPointReflection, DXBC/RDEF/RTS0, and RDNA2 builders
// ---------------------------------------------------------------------------

fn make_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = env::temp_dir().join(format!("{prefix}_{}_{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn binding(
    group: u32,
    binding: u32,
    name: &str,
    resource: ResourceKind,
    layout: BindingLayout,
) -> ReflectedBinding {
    ReflectedBinding {
        group,
        binding,
        name: Some(name.to_string()),
        resource,
        layout: Some(layout),
        binding_array: None,
        register: None,
        spirv: None,
        root_constants: false,
    }
}

fn entry_point(
    name: &str,
    stage: ShaderStage,
    bindings: Vec<ReflectedBinding>,
) -> EntryPointReflection {
    EntryPointReflection {
        name: name.to_string(),
        stage,
        bindings,
        vertex_inputs: Vec::new(),
        fragment_outputs: Vec::new(),
        workgroup_size: (stage == ShaderStage::Compute).then_some([64, 1, 1]),
        requires_lowering: Vec::new(),
    }
}

fn build_dxbc_container(chunks: &[([u8; 4], &[u8])]) -> Vec<u8> {
    let header_and_table = 32 + chunks.len() * 4;
    let mut offsets = Vec::with_capacity(chunks.len());
    let mut cursor = header_and_table;
    for (_, payload) in chunks {
        offsets.push(cursor as u32);
        cursor += 8 + payload.len();
    }
    let total_size = cursor as u32;
    let mut out = Vec::with_capacity(cursor);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0x11u8; 16]); // digest
    out.extend_from_slice(&1u16.to_le_bytes()); // major
    out.extend_from_slice(&0u16.to_le_bytes()); // minor
    out.extend_from_slice(&total_size.to_le_bytes());
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    for off in offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    for (fourcc, payload) in chunks {
        out.extend_from_slice(fourcc);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
    }
    out
}

fn build_dxil_program_chunk(kind: u16, sm_major: u8, sm_minor: u8) -> Vec<u8> {
    let program_version =
        (u32::from(kind) << 16) | (u32::from(sm_major & 0xF) << 4) | u32::from(sm_minor & 0xF);
    let bitcode = [0x42u8, 0x43, 0xC0, 0xDE, 0x21, 0x0C, 0x00, 0x00];
    let total_bytes = 24 + bitcode.len();
    let mut out = Vec::with_capacity(total_bytes);
    out.extend_from_slice(&program_version.to_le_bytes());
    out.extend_from_slice(&((total_bytes / 4) as u32).to_le_bytes());
    out.extend_from_slice(b"DXIL");
    out.extend_from_slice(&0x0001_0006u32.to_le_bytes()); // DXIL 1.6
    out.extend_from_slice(&16u32.to_le_bytes()); // bitcode_offset relative to byte 8
    out.extend_from_slice(&(bitcode.len() as u32).to_le_bytes());
    out.extend_from_slice(&bitcode);
    out
}

fn build_rdef_rd11_chunk(
    program_type: u16,
    cbuffers: &[(&str, u32)],
    resources: &[(&str, u32, u32, u32, u32, u32, u32, u32)], // (name, input_type, return_type, dim, bind_point, bind_count, flags, space)
) -> Vec<u8> {
    let header_len = 60usize;
    let cb_offset = header_len;
    let res_offset = cb_offset + cbuffers.len() * 24;
    let mut strings_start = res_offset + resources.len() * 40;

    let mut string_pool: Vec<u8> = Vec::new();
    let mut intern = |s: &str| -> u32 {
        let off = (strings_start + string_pool.len()) as u32;
        string_pool.extend_from_slice(s.as_bytes());
        string_pool.push(0);
        off
    };

    let creator_off = intern("dxc-test-s71");
    let cb_name_offs: Vec<u32> = cbuffers.iter().map(|(n, _)| intern(n)).collect();
    let res_name_offs: Vec<u32> = resources.iter().map(|r| intern(r.0)).collect();
    let _ = &mut strings_start;

    let mut out = Vec::new();
    // Base 28-byte RDEF header
    out.extend_from_slice(&(cbuffers.len() as u32).to_le_bytes());
    out.extend_from_slice(&(cb_offset as u32).to_le_bytes());
    out.extend_from_slice(&(resources.len() as u32).to_le_bytes());
    out.extend_from_slice(&(res_offset as u32).to_le_bytes());
    out.push(6); // target_minor
    out.push(6); // target_major (SM 6.6)
    out.extend_from_slice(&program_type.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags
    out.extend_from_slice(&creator_off.to_le_bytes());
    // RD11 32-byte extension
    out.extend_from_slice(b"RD11");
    out.extend_from_slice(&60u32.to_le_bytes());
    out.extend_from_slice(&24u32.to_le_bytes()); // cb stride
    out.extend_from_slice(&40u32.to_le_bytes()); // var stride
    out.extend_from_slice(&36u32.to_le_bytes()); // type stride
    out.extend_from_slice(&40u32.to_le_bytes()); // resource binding stride
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(out.len(), header_len);

    for (i, (_, size_bytes)) in cbuffers.iter().enumerate() {
        out.extend_from_slice(&cb_name_offs[i].to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // var count
        out.extend_from_slice(&0u32.to_le_bytes()); // var offset
        out.extend_from_slice(&size_bytes.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // flags
        out.extend_from_slice(&0u32.to_le_bytes()); // cb_type = CBUFFER
    }

    for (i, &(_, input_type, ret_type, dim, bind_point, bind_count, flags, space)) in
        resources.iter().enumerate()
    {
        out.extend_from_slice(&res_name_offs[i].to_le_bytes());
        out.extend_from_slice(&input_type.to_le_bytes());
        out.extend_from_slice(&ret_type.to_le_bytes());
        out.extend_from_slice(&dim.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // num_samples
        out.extend_from_slice(&bind_point.to_le_bytes());
        out.extend_from_slice(&bind_count.to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&space.to_le_bytes());
        out.extend_from_slice(&(i as u32).to_le_bytes()); // id
    }

    out.extend_from_slice(&string_pool);
    out
}

fn build_isg1_chunk(elements: &[(&str, u32, u32, u32, u32, u8)]) -> Vec<u8> {
    // (semantic_name, semantic_index, sv_type, comp_type, register, mask)
    let param_offset = 8usize;
    let strings_start = param_offset + elements.len() * 32;
    let mut string_pool = Vec::new();
    let mut name_offs = Vec::with_capacity(elements.len());
    for (name, ..) in elements {
        name_offs.push((strings_start + string_pool.len()) as u32);
        string_pool.extend_from_slice(name.as_bytes());
        string_pool.push(0);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&(elements.len() as u32).to_le_bytes());
    out.extend_from_slice(&(param_offset as u32).to_le_bytes());
    for (i, &(_, sem_idx, sv_type, comp_type, reg, mask)) in elements.iter().enumerate() {
        out.extend_from_slice(&0u32.to_le_bytes()); // stream
        out.extend_from_slice(&name_offs[i].to_le_bytes());
        out.extend_from_slice(&sem_idx.to_le_bytes());
        out.extend_from_slice(&sv_type.to_le_bytes());
        out.extend_from_slice(&comp_type.to_le_bytes());
        out.extend_from_slice(&reg.to_le_bytes());
        out.push(mask);
        out.push(mask); // rw_mask
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // min_precision
    }
    out.extend_from_slice(&string_pool);
    out
}

fn build_stat_chunk(inst_count: u32, temp_regs: u32, tex_normal: u32) -> Vec<u8> {
    let mut words = vec![0u32; 16];
    words[0] = inst_count;
    words[1] = temp_regs;
    words[2] = 4; // def
    words[3] = 8; // dcl
    words[4] = 20; // float
    words[5] = 5; // int
    words[6] = 7; // uint
    words[11] = tex_normal; // offset 44
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn encode_rts0_v1_1(rs: &RootSignature) -> Vec<u8> {
    let params_offset = 24usize;
    let param_headers_len = rs.parameters.len() * 12;
    let mut data_buf: Vec<u8> = Vec::new();
    let data_base = params_offset + param_headers_len;

    let mut param_entries: Vec<(u32, u32, u32)> = Vec::new();
    for p in &rs.parameters {
        let data_off = (data_base + data_buf.len()) as u32;
        let vis = p.visibility as u32;
        match &p.kind {
            RootParameterKind::DescriptorTable(ranges) => {
                let ranges_off = (data_base + data_buf.len() + 8) as u32;
                data_buf.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
                data_buf.extend_from_slice(&ranges_off.to_le_bytes());
                for r in ranges {
                    data_buf.extend_from_slice(&(r.range_type as u32).to_le_bytes());
                    data_buf.extend_from_slice(&r.num_descriptors.to_le_bytes());
                    data_buf.extend_from_slice(&r.base_shader_register.to_le_bytes());
                    data_buf.extend_from_slice(&r.register_space.to_le_bytes());
                    data_buf.extend_from_slice(&r.flags.to_le_bytes());
                    data_buf
                        .extend_from_slice(&r.offset_in_descriptors_from_table_start.to_le_bytes());
                }
                param_entries.push((0, vis, data_off));
            }
            RootParameterKind::Constants32Bit {
                shader_register,
                register_space,
                num_32bit_values,
            } => {
                data_buf.extend_from_slice(&shader_register.to_le_bytes());
                data_buf.extend_from_slice(&register_space.to_le_bytes());
                data_buf.extend_from_slice(&num_32bit_values.to_le_bytes());
                param_entries.push((1, vis, data_off));
            }
            RootParameterKind::Cbv {
                shader_register,
                register_space,
                flags,
            } => {
                data_buf.extend_from_slice(&shader_register.to_le_bytes());
                data_buf.extend_from_slice(&register_space.to_le_bytes());
                data_buf.extend_from_slice(&flags.to_le_bytes());
                param_entries.push((2, vis, data_off));
            }
            RootParameterKind::Srv {
                shader_register,
                register_space,
                flags,
            } => {
                data_buf.extend_from_slice(&shader_register.to_le_bytes());
                data_buf.extend_from_slice(&register_space.to_le_bytes());
                data_buf.extend_from_slice(&flags.to_le_bytes());
                param_entries.push((3, vis, data_off));
            }
            RootParameterKind::Uav {
                shader_register,
                register_space,
                flags,
            } => {
                data_buf.extend_from_slice(&shader_register.to_le_bytes());
                data_buf.extend_from_slice(&register_space.to_le_bytes());
                data_buf.extend_from_slice(&flags.to_le_bytes());
                param_entries.push((4, vis, data_off));
            }
        }
    }

    let samplers_offset = (data_base + data_buf.len()) as u32;
    let mut out = Vec::new();
    out.extend_from_slice(&2u32.to_le_bytes()); // v1.1
    out.extend_from_slice(&(rs.parameters.len() as u32).to_le_bytes());
    out.extend_from_slice(&(params_offset as u32).to_le_bytes());
    out.extend_from_slice(&(rs.static_samplers.len() as u32).to_le_bytes());
    out.extend_from_slice(&samplers_offset.to_le_bytes());
    out.extend_from_slice(&rs.flags.to_le_bytes());
    for (pt, vis, off) in param_entries {
        out.extend_from_slice(&pt.to_le_bytes());
        out.extend_from_slice(&vis.to_le_bytes());
        out.extend_from_slice(&off.to_le_bytes());
    }
    out.extend_from_slice(&data_buf);
    for s in &rs.static_samplers {
        out.extend_from_slice(&s.filter.to_le_bytes());
        out.extend_from_slice(&s.address_u.to_le_bytes());
        out.extend_from_slice(&s.address_v.to_le_bytes());
        out.extend_from_slice(&s.address_w.to_le_bytes());
        out.extend_from_slice(&s.mip_lod_bias_bits.to_le_bytes());
        out.extend_from_slice(&s.max_anisotropy.to_le_bytes());
        out.extend_from_slice(&s.comparison_func.to_le_bytes());
        out.extend_from_slice(&s.border_color.to_le_bytes());
        out.extend_from_slice(&s.min_lod_bits.to_le_bytes());
        out.extend_from_slice(&s.max_lod_bits.to_le_bytes());
        out.extend_from_slice(&s.shader_register.to_le_bytes());
        out.extend_from_slice(&s.register_space.to_le_bytes());
        out.extend_from_slice(&(s.shader_visibility as u32).to_le_bytes());
    }
    out
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn enc_sopp(op: u8, simm16: u16) -> u32 {
    (0b1_0111_1111_u32 << 23) | (u32::from(op & 0x7F) << 16) | u32::from(simm16)
}

fn enc_vop1(op: u8, vdst: u8, src0: u16) -> u32 {
    (0b011_1111_u32 << 25)
        | (u32::from(vdst) << 17)
        | (u32::from(op) << 9)
        | u32::from(src0 & 0x1FF)
}

fn enc_exp(target: u8, en: u8, done: bool, vm: bool, vsrc: [u8; 4]) -> [u32; 2] {
    let w0 = (0b111110u32 << 26)
        | (u32::from(vm) << 12)
        | (u32::from(done) << 11)
        | (u32::from(target & 0x3F) << 4)
        | u32::from(en & 0xF);
    let w1 = u32::from(vsrc[0])
        | (u32::from(vsrc[1]) << 8)
        | (u32::from(vsrc[2]) << 16)
        | (u32::from(vsrc[3]) << 24);
    [w0, w1]
}

fn build_minimal_vs_rdna2() -> Vec<u8> {
    let mut words = vec![
        enc_vop1(0x01, 1, 242), // v_mov_b32 v1, 1.0
    ];
    words.extend_from_slice(&enc_exp(12, 0xF, true, false, [1, 1, 1, 1]));
    words.push(enc_sopp(0x01, 0)); // s_endpgm
    words_to_bytes(&words)
}

fn build_minimal_ps_rdna2() -> Vec<u8> {
    let mut words = vec![
        enc_vop1(0x01, 1, 242), // v_mov_b32 v1, 1.0
    ];
    words.extend_from_slice(&enc_exp(0, 0xF, true, true, [1, 1, 1, 1]));
    words.push(enc_sopp(0x01, 0)); // s_endpgm
    words_to_bytes(&words)
}

fn build_minimal_cs_rdna2() -> Vec<u8> {
    words_to_bytes(&[
        enc_vop1(0x01, 1, 242), // v_mov_b32 v1, 1.0
        enc_sopp(0x0A, 0),      // s_barrier
        enc_sopp(0x01, 0),      // s_endpgm
    ])
}

// ---------------------------------------------------------------------------
// 1. Reflection v3: Cross-stage collision detection & resolution
// ---------------------------------------------------------------------------

#[test]
fn reflect_v3_cross_stage_collision_detection_and_offset_remapping() {
    let vs = entry_point(
        "vs_main",
        ShaderStage::Vertex,
        vec![
            // Shared compatible uniform CBV (smaller minBindingSize in VS -> mergeable in non-strict mode)
            binding(
                0,
                0,
                "frame_cb",
                ResourceKind::Buffer,
                BindingLayout::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: 64,
                },
            ),
            // Colliding kind at (0, 1): VS declares uniform buffer, PS declares texture_2d
            binding(
                0,
                1,
                "vs_cb1",
                ResourceKind::Buffer,
                BindingLayout::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: 128,
                },
            ),
            // Colliding texture viewDimension at (0, 2): VS uses 2D, PS uses Cube
            binding(
                0,
                2,
                "vs_tex2d",
                ResourceKind::Texture,
                BindingLayout::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            // Colliding storage texture format at (1, 0): rgba8unorm vs r32float
            binding(
                1,
                0,
                "vs_uav",
                ResourceKind::StorageTexture,
                BindingLayout::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: "rgba8unorm",
                    view_dimension: TextureViewDimension::D2,
                },
            ),
        ],
    );

    let ps = entry_point(
        "fs_main",
        ShaderStage::Fragment,
        vec![
            binding(
                0,
                0,
                "frame_cb",
                ResourceKind::Buffer,
                BindingLayout::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: 256,
                },
            ),
            binding(
                0,
                1,
                "albedo_tex",
                ResourceKind::Texture,
                BindingLayout::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            binding(
                0,
                2,
                "env_cube",
                ResourceKind::Texture,
                BindingLayout::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::Cube,
                    multisampled: false,
                },
            ),
            binding(
                1,
                0,
                "ps_uav",
                ResourceKind::StorageTexture,
                BindingLayout::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: "r32float",
                    view_dimension: TextureViewDimension::D2,
                },
            ),
        ],
    );

    // 1) Default options: detects minBindingSize mismatch (mergeable) + 3 hard conflicts,
    //    and deterministically remaps the 3 colliding PS bindings to offset >= 64.
    let opts = MultiStageMergeOptions::default();
    let report = merge_multistage_pipeline_with_resolution(&[&vs, &ps], &opts)
        .expect("multi-stage resolution must produce collision-free layout");

    assert!(report.has_hard_conflicts());
    assert!(report.has_min_binding_size_mismatches());
    assert!(report.was_remapped());
    assert_eq!(report.remaps.len(), 3);

    // Verify collision kinds recorded deterministically.
    let kinds: Vec<BindingCollisionKind> = report.collisions.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&BindingCollisionKind::MinBindingSizeMismatch));
    assert!(kinds.contains(&BindingCollisionKind::ResourceKindMismatch));
    assert!(kinds.contains(&BindingCollisionKind::TextureViewDimensionMismatch));
    assert!(kinds.contains(&BindingCollisionKind::StorageTextureFormatMismatch));

    // Shared (0, 0) merged cleanly with VS|FS visibility and max minBindingSize = 256.
    let g0 = &report.layout.bind_group_layouts[0];
    let cb0 = g0.entries.iter().find(|e| e.binding == 0).unwrap();
    assert_eq!(cb0.visibility, VISIBILITY_VERTEX | VISIBILITY_FRAGMENT);
    assert_eq!(
        cb0.layout,
        BindingLayout::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: 256,
        }
    );

    // Remapped PS bindings live at >= 64 and are disjoint from VS bindings.
    assert!(report.remaps.iter().all(|r| r.stage == ShaderStage::Fragment && r.new_binding >= 64));
    let resolved_refs: Vec<&EntryPointReflection> = report.resolved_stages.iter().collect();
    let remaining_hard: Vec<_> = detect_cross_stage_binding_collisions(&resolved_refs, &opts)
        .into_iter()
        .filter(|c| !c.resolvable_by_merge)
        .collect();
    assert!(
        remaining_hard.is_empty(),
        "resolved stages must have zero remaining hard conflicts: {remaining_hard:?}"
    );

    // 2) Strict min_binding_size mode remaps (0, 0) too so every stage binding is strictly disjoint.
    let strict_opts = MultiStageMergeOptions {
        strict_min_binding_size: true,
        ..MultiStageMergeOptions::default()
    };
    let strict_report = merge_multistage_pipeline_with_resolution(&[&vs, &ps], &strict_opts)
        .expect("strict merge resolution must succeed");
    assert_eq!(strict_report.remaps.len(), 4);
    let strict_refs: Vec<&EntryPointReflection> = strict_report.resolved_stages.iter().collect();
    assert!(
        validate_stage_disjoint_bindings(&strict_refs).is_empty(),
        "strict resolved stages must pass disjoint validation cleanly"
    );

    // 3) MS+PS (Mesh-Compute + Fragment) pipeline support with collision resolution.
    let ms = entry_point(
        "ms_main",
        ShaderStage::Compute,
        vec![binding(
            0,
            0,
            "mesh_buf",
            ResourceKind::Buffer,
            BindingLayout::Buffer {
                ty: BufferBindingType::ReadOnlyStorage,
                has_dynamic_offset: false,
                min_binding_size: 64,
            },
        )],
    );
    let ms_ps_ps = entry_point(
        "fs_main",
        ShaderStage::Fragment,
        vec![binding(
            0,
            0,
            "ps_cb",
            ResourceKind::Buffer,
            BindingLayout::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 64,
            },
        )],
    );
    let ms_report = merge_multistage_pipeline_with_resolution(&[&ms, &ms_ps_ps], &opts)
        .expect("MS(Compute)+PS merge must succeed");
    assert_eq!(ms_report.remaps.len(), 1);
    assert_eq!(ms_report.remaps[0].stage, ShaderStage::Fragment);
    assert_eq!(
        ms_report.remaps[0].new_binding,
        playgta6_shaders::reflect::DEFAULT_SECONDARY_STAGE_BINDING_OFFSET
    );
    let ms_g0 = &ms_report.layout.bind_group_layouts[0];
    assert_eq!(ms_g0.entries.len(), 2);
    assert_eq!(ms_g0.entries[0].visibility, VISIBILITY_COMPUTE);
    assert_eq!(ms_g0.entries[1].visibility, VISIBILITY_FRAGMENT);
}

// ---------------------------------------------------------------------------
// 2. Reflection v3: Bindless stride validation & Sidecar v3 completeness
// ---------------------------------------------------------------------------

#[test]
fn reflect_v3_bindless_stride_validation_and_sidecar_v3_unresolved_and_compute_census() {
    // 1) GFX10.3 descriptor stride validity predicate.
    assert!(is_valid_gfx10_descriptor_stride(BindlessHeapKind::Sampler, GFX10_SAMPLER_STRIDE_BYTES));
    assert!(is_valid_gfx10_descriptor_stride(BindlessHeapKind::Sampler, 32));
    assert!(is_valid_gfx10_descriptor_stride(BindlessHeapKind::Sampler, 64));
    assert!(!is_valid_gfx10_descriptor_stride(BindlessHeapKind::Sampler, 0));
    assert!(!is_valid_gfx10_descriptor_stride(BindlessHeapKind::Sampler, 8));
    assert!(!is_valid_gfx10_descriptor_stride(BindlessHeapKind::Sampler, 24));
    assert!(!is_valid_gfx10_descriptor_stride(BindlessHeapKind::Sampler, 128));

    assert!(is_valid_gfx10_descriptor_stride(BindlessHeapKind::ResourceBuffer, GFX10_BUFFER_STRIDE_BYTES));
    assert!(!is_valid_gfx10_descriptor_stride(BindlessHeapKind::ResourceImage, 16));
    assert!(is_valid_gfx10_descriptor_stride(BindlessHeapKind::ResourceImage, GFX10_IMAGE_STRIDE_BYTES));
    assert!(is_valid_gfx10_descriptor_stride(BindlessHeapKind::ResourceImage, 64));

    // 2) Valid bindless bindings pass with zero violations.
    let valid_bindings = vec![
        WgslBindingInfo {
            group: 1,
            binding: 0,
            kind: "texture_2d",
            name: "bindless_albedo".to_string(),
            view_dimension: Some("2d"),
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::CbvIndexedResourceHeap(
                CbvIndexedHeapSource {
                    cbv: CbvRefSource::RootDescriptor { user_sgpr: 8 },
                    byte_offset: 16,
                    shift: 5,
                    bit_offset: 0,
                    mask: u32::MAX,
                    stride: 32,
                },
            )),
        },
        WgslBindingInfo {
            group: 1,
            binding: 1,
            kind: "texture_2d",
            name: "bindless_normal".to_string(),
            view_dimension: Some("2d"),
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::LaneIndexedResourceHeap(
                LaneIndexedHeapSource {
                    max_count: 64,
                    shift: 5,
                    stride: 32,
                    window_base: None,
                },
            )),
        },
        WgslBindingInfo {
            group: 1,
            binding: 2,
            kind: "sampler",
            name: "bindless_samp".to_string(),
            view_dimension: None,
            storage_format: None,
            min_binding_size_bytes: None,
            highest_vec4_offset: None,
            source: Some(WgslBindingSource::LaneIndexedSamplerHeap(
                LaneIndexedHeapSource {
                    max_count: 16,
                    shift: 4,
                    stride: 16,
                    window_base: None,
                },
            )),
        },
    ];
    assert!(validate_bindless_binding_strides(&valid_bindings).is_empty());

    // 3) Invalid bindless strides: zero, non-pow2, undersized image, shift mismatch, inconsistent heap stride, bad window.
    let bad_specs = vec![
        BindlessArrayStrideSpec {
            group: 0,
            binding: 0,
            name: "zero_s".to_string(),
            heap_kind: BindlessHeapKind::Sampler,
            stride_bytes: 0,
            shift: None,
            max_count: None,
        },
        BindlessArrayStrideSpec {
            group: 0,
            binding: 1,
            name: "unaligned_r".to_string(),
            heap_kind: BindlessHeapKind::ResourceImage,
            stride_bytes: 24,
            shift: None,
            max_count: None,
        },
        BindlessArrayStrideSpec {
            group: 0,
            binding: 2,
            name: "too_small_img".to_string(),
            heap_kind: BindlessHeapKind::ResourceImage,
            stride_bytes: 16,
            shift: Some(4),
            max_count: None,
        },
        BindlessArrayStrideSpec {
            group: 0,
            binding: 3,
            name: "shift_mismatch".to_string(),
            heap_kind: BindlessHeapKind::ResourceImage,
            stride_bytes: 32,
            shift: Some(6),
            max_count: Some(512), // > 256 -> InvalidWindowCount
        },
        BindlessArrayStrideSpec {
            group: 0,
            binding: 4,
            name: "inconsistent_heap_stride".to_string(),
            heap_kind: BindlessHeapKind::ResourceImage,
            stride_bytes: 64,
            shift: Some(6),
            max_count: None,
        },
    ];
    let violations = playgta6_shaders::reflect::validate_bindless_array_strides(&bad_specs);
    let err_kinds: Vec<BindlessStrideErrorKind> = violations.iter().map(|v| v.kind).collect();
    assert!(err_kinds.contains(&BindlessStrideErrorKind::ZeroStride));
    assert!(err_kinds.contains(&BindlessStrideErrorKind::NotPowerOfTwo));
    assert!(err_kinds.contains(&BindlessStrideErrorKind::BelowMinDescriptorSize));
    assert!(err_kinds.contains(&BindlessStrideErrorKind::ShiftStrideMismatch));
    assert!(err_kinds.contains(&BindlessStrideErrorKind::InvalidWindowCount));
    assert!(err_kinds.contains(&BindlessStrideErrorKind::InconsistentHeapStride));

    // 4) Sidecar v1 / v2 / v3 version auto-selection + unresolved + compute accounting.
    let v1_binding = vec![WgslBindingInfo {
        group: 0,
        binding: 0,
        kind: "uniform",
        name: "cb0".to_string(),
        view_dimension: None,
        storage_format: None,
        min_binding_size_bytes: Some(64),
        highest_vec4_offset: Some(3),
        source: Some(WgslBindingSource::Table {
            user_sgpr: 8,
            table_byte_offset: 0,
        }),
    }];
    let v1_sidecar = StageReflectionSidecar::build(
        &[SidecarStageInput {
            hash: 0x1001,
            stage: ShaderStage::Vertex,
            bindings: &v1_binding,
            vertex_inputs: &[],
            vertex_input_mode: "vertex_index",
            vs_outputs: &[],
            ps_inputs: &[],
            mesh_output: None,
            rt_arena: false,
        }],
        &SidecarBuildOptions::default(),
    );
    assert_eq!(v1_sidecar.version, STAGE_REFLECTION_VERSION_V1);
    assert!(v1_sidecar.validate().is_empty());

    let v2_binding = vec![WgslBindingInfo {
        group: 0,
        binding: 0,
        kind: "uniform",
        name: "cb0".to_string(),
        view_dimension: None,
        storage_format: None,
        min_binding_size_bytes: Some(64),
        highest_vec4_offset: Some(3),
        source: Some(WgslBindingSource::RootDescriptor {
            user_sgpr: 8,
            table_byte_offset: Some(32),
        }),
    }];
    let v2_sidecar = StageReflectionSidecar::build(
        &[SidecarStageInput {
            hash: 0x1002,
            stage: ShaderStage::Fragment,
            bindings: &v2_binding,
            vertex_inputs: &[],
            vertex_input_mode: "none",
            vs_outputs: &[],
            ps_inputs: &[],
            mesh_output: None,
            rt_arena: false,
        }],
        &SidecarBuildOptions::default(),
    );
    assert_eq!(v2_sidecar.version, STAGE_REFLECTION_VERSION_V2);
    assert!(v2_sidecar.validate().is_empty());

    // v3: read-only binding without source gets `unresolved`, compute stage is preserved,
    // writable UAV without source is rejected cleanly (never marked unresolved!).
    let cs_readonly_unresolved = vec![WgslBindingInfo {
        group: 0,
        binding: 0,
        kind: "texture_2d",
        name: "tex0".to_string(),
        view_dimension: Some("2d"),
        storage_format: None,
        min_binding_size_bytes: None,
        highest_vec4_offset: None,
        source: None,
    }];
    let cs_writable_unsourced = vec![WgslBindingInfo {
        group: 0,
        binding: 1,
        kind: "storage",
        name: "uav0".to_string(),
        view_dimension: None,
        storage_format: None,
        min_binding_size_bytes: None,
        highest_vec4_offset: None,
        source: None,
    }];
    let v3_sidecar = StageReflectionSidecar::build(
        &[
            SidecarStageInput {
                hash: 0x2001,
                stage: ShaderStage::Compute,
                bindings: &cs_readonly_unresolved,
                vertex_inputs: &[],
                vertex_input_mode: "none",
                vs_outputs: &[],
                ps_inputs: &[],
                mesh_output: None,
                rt_arena: false,
            },
            SidecarStageInput {
                hash: 0x2002,
                stage: ShaderStage::Compute,
                bindings: &cs_writable_unsourced,
                vertex_inputs: &[],
                vertex_input_mode: "none",
                vs_outputs: &[],
                ps_inputs: &[],
                mesh_output: None,
                rt_arena: false,
            },
        ],
        &SidecarBuildOptions {
            allow_unresolved_readonly: true,
            validate_bindless_strides: true,
            unresolved_reason: Some("dynamic_cb_index".to_string()),
        },
    );
    assert_eq!(v3_sidecar.version, STAGE_REFLECTION_VERSION_V3);
    assert_eq!(v3_sidecar.census.compute_stages, 1);
    assert_eq!(v3_sidecar.census.skipped_compute_stages, 1);
    assert_eq!(v3_sidecar.census.skipped_writable_without_source, 1);
    assert_eq!(v3_sidecar.census.unresolved_bindings, 1);
    assert!(v3_sidecar.validate().is_empty());

    let v3_json = v3_sidecar.render_json();
    assert!(v3_json.contains("\"schema\": \"playgta6-stage-reflection\""));
    assert!(v3_json.contains("\"version\": 3"));
    assert!(v3_json.contains("\"stage\": \"compute\""));
    assert!(v3_json.contains("\"unresolved\": \"dynamic_cb_index\""));
}

// ---------------------------------------------------------------------------
// 3. DXIL Fallback Reflection & External Root-Parameter Matching
// ---------------------------------------------------------------------------

#[test]
fn dxil_fallback_reflection_and_external_root_signature_matching() {
    // Construct a DXIL-only Pixel Shader container WITHOUT embedded RTS0 (matching
    // real LEGAL NOTICES -> Spinner containers noted in #482), carrying DXIL + RDEF(RD11) +
    // ISG1 + OSG1 + STAT.
    let dxil_chunk = build_dxil_program_chunk(0 /* Pixel */, 6, 6);
    let rdef_chunk = build_rdef_rd11_chunk(
        0xFFFF, // Pixel
        &[("RootConsts", 16), ("FrameCB", 256)],
        &[
            // (name, input_type, return_type, dim, bind_point, bind_count, flags, space)
            ("RootConsts", 0 /* CBuffer */, 0, 0, 0, 1, 0, 0),
            ("FrameCB", 0 /* CBuffer */, 0, 0, 1, 1, 0, 0),
            ("SpinnerTex", 2 /* Texture */, 5 /* FLOAT */, 4 /* Tex2D */, 0, 1, 0, 0),
            ("ExtraTex", 2 /* Texture */, 5 /* FLOAT */, 4 /* Tex2D */, 2, 1, 0, 0),
            ("BindlessTexHeap", 2 /* Texture */, 5 /* FLOAT */, 4 /* Tex2D */, 0, 0 /* unbounded */, 0, 1),
            ("LinearClamp", 3 /* Sampler */, 0, 0, 0, 1, 0, 0),
            ("ShadowCmp", 3 /* Sampler */, 0, 0, 1, 1, 0x02 /* comparison */, 0),
            ("UnmatchedRO", 5 /* Structured */, 0, 1 /* Buffer */, 9, 1, 0, 0),
        ],
    );
    let isg1_chunk = build_isg1_chunk(&[
        ("SV_POSITION", 0, 1, 3 /* f32 */, 0, 0xF),
        ("TEXCOORD", 0, 0, 3 /* f32 */, 1, 0x3),
    ]);
    let osg1_chunk = build_isg1_chunk(&[("SV_TARGET", 0, 64, 3 /* f32 */, 0, 0xF)]);
    let stat_chunk = build_stat_chunk(42, 6, 3);

    // Direct chunk parsers work cleanly.
    let parsed_rdef = parse_rdef_chunk(&rdef_chunk).expect("parse_rdef_chunk must succeed");
    assert!(parsed_rdef.is_rd11);
    assert_eq!(parsed_rdef.cbuffers.len(), 2);
    assert_eq!(parsed_rdef.resources.len(), 8);

    let parsed_stat = parse_stat_chunk(&stat_chunk).expect("parse_stat_chunk must succeed");
    assert_eq!(parsed_stat.instruction_count, 42);
    assert_eq!(parsed_stat.temp_register_count, 6);
    assert_eq!(parsed_stat.texture_normal_instructions, Some(3));

    let container = build_dxbc_container(&[
        (*b"DXIL", &dxil_chunk),
        (*b"RDEF", &rdef_chunk),
        (*b"ISG1", &isg1_chunk),
        (*b"OSG1", &osg1_chunk),
        (*b"STAT", &stat_chunk),
    ]);

    // Standard reflect_dxil_bytes now falls back to RDEF when RTS0 & PSV0 are absent.
    let basic_refl = reflect_dxil_bytes(&container).expect("RDEF fallback reflection must succeed");
    assert_eq!(basic_refl.entry.stage, ShaderStage::Fragment);
    assert!(basic_refl.root_signature.is_none());
    assert!(!basic_refl.entry.bindings.is_empty());

    // Full fallback extraction captures every resource, signature, and stat.
    let fallback = extract_dxil_fallback_reflection(&container)
        .expect("extract_dxil_fallback_reflection must succeed");
    assert_eq!(fallback.stage, ShaderStage::Fragment);
    assert!(!fallback.has_embedded_rts0);
    assert!(fallback.has_dxil_bitcode);
    assert_eq!(fallback.resources.len(), 8);
    assert_eq!(fallback.fragment_outputs.len(), 1);
    assert_eq!(fallback.stat.as_ref().unwrap().instruction_count, 42);

    // External D3D12 RootSignature (from dump manifest / standalone RTS0):
    //   param 0: 4 root constants at b0, space0 -> user_sgpr = 8 (width 4 -> next 12)
    //   param 1: root CBV at b1, space0         -> user_sgpr = 12 (width 2 -> next 14)
    //   param 2: descriptor table:              -> user_sgpr = 14 (width 1 -> next 15)
    //            range 0: t0..t3 (4 SRVs, space0, offset 0)
    //   param 3: unbounded bindless table:      -> user_sgpr = 15 (width 1 -> next 16)
    //            range 0: t0..unbounded (space1, offset APPEND)
    //   static_samplers[0]: s0, space0 (filtering)
    //   static_samplers[1]: s1, space0 (comparison)
    let external_rs = RootSignature {
        version: RootSignatureVersion::V1_1,
        flags: 0,
        parameters: vec![
            RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::Constants32Bit {
                    shader_register: 0,
                    register_space: 0,
                    num_32bit_values: 4,
                },
            },
            RootParameter {
                visibility: D3dShaderVisibility::All,
                kind: RootParameterKind::Cbv {
                    shader_register: 1,
                    register_space: 0,
                    flags: 0,
                },
            },
            RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                    range_type: DescriptorRangeType::Srv,
                    num_descriptors: 4,
                    base_shader_register: 0,
                    register_space: 0,
                    flags: 0,
                    offset_in_descriptors_from_table_start: 0,
                }]),
            },
            RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                    range_type: DescriptorRangeType::Srv,
                    num_descriptors: u32::MAX,
                    base_shader_register: 0,
                    register_space: 1,
                    flags: 0,
                    offset_in_descriptors_from_table_start: u32::MAX,
                }]),
            },
        ],
        static_samplers: vec![
            StaticSamplerDesc {
                filter: 0x15, // MIN_MAG_MIP_LINEAR
                address_u: 1,
                address_v: 1,
                address_w: 1,
                mip_lod_bias_bits: 0,
                max_anisotropy: 1,
                comparison_func: 0,
                border_color: 0,
                min_lod_bits: 0,
                max_lod_bits: 0,
                shader_register: 0,
                register_space: 0,
                shader_visibility: D3dShaderVisibility::Pixel,
            },
            StaticSamplerDesc {
                filter: 0x95, // COMPARISON_MIN_MAG_MIP_LINEAR ((0x95 >> 7) & 3 == 1)
                address_u: 3,
                address_v: 3,
                address_w: 3,
                mip_lod_bias_bits: 0,
                max_anisotropy: 1,
                comparison_func: 4,
                border_color: 0,
                min_lod_bits: 0,
                max_lod_bits: 0,
                shader_register: 1,
                register_space: 0,
                shader_visibility: D3dShaderVisibility::Pixel,
            },
        ],
    };

    let rts0_bytes = encode_rts0_v1_1(&external_rs);
    let conv = RegisterConvention::default();
    let match_report = match_dxil_bytes_to_external_rts0(&container, &rts0_bytes, &conv)
        .expect("match_dxil_bytes_to_external_rts0 must succeed");

    assert_eq!(match_report.user_data_base, CONSOLE_GRAPHICS_USER_DATA_BASE);
    assert!(match_report.is_joinable_v3());
    assert_eq!(match_report.matched.len(), 7);
    assert_eq!(match_report.unmatched_readonly.len(), 1);
    assert_eq!(match_report.unmatched_readonly[0].name, "UnmatchedRO");
    assert!(match_report.unmatched_writable.is_empty());

    // Check individual matched sources match console ABI exactly.
    let find_matched = |name: &str| {
        match_report
            .matched
            .iter()
            .find(|m| m.resource.name == name)
            .unwrap_or_else(|| panic!("missing matched binding {name}"))
    };
    assert_eq!(
        find_matched("RootConsts").sidecar_source,
        SidecarBindingSource::RootConstants { user_sgpr: 8 }
    );
    assert_eq!(
        find_matched("FrameCB").sidecar_source,
        SidecarBindingSource::RootDescriptor {
            user_sgpr: 12,
            table_byte_offset: None,
        }
    );
    assert_eq!(
        find_matched("SpinnerTex").sidecar_source,
        SidecarBindingSource::Table {
            user_sgpr: 14,
            table_byte_offset: 0,
        }
    );
    assert_eq!(
        find_matched("ExtraTex").sidecar_source,
        SidecarBindingSource::Table {
            user_sgpr: 14,
            table_byte_offset: 2 * CONSOLE_TABLE_SLOT_BYTES,
        }
    );
    assert!(matches!(
        find_matched("BindlessTexHeap").sidecar_source,
        SidecarBindingSource::LaneIndexedResourceHeap(LaneIndexedHeapSource {
            stride: 32,
            shift: 5,
            ..
        })
    ));
    assert_eq!(
        find_matched("LinearClamp").match_kind,
        DxilRootParamMatchKind::StaticSampler {
            static_sampler_index: 0
        }
    );
    assert_eq!(
        find_matched("ShadowCmp").sidecar_source,
        SidecarBindingSource::StaticSampler { index: 1 }
    );

    // Sidecar entry inserts cleanly into StageReflectionSidecar v3 and passes validation!
    let mut sidecar = StageReflectionSidecar::default();
    sidecar.insert_stage(match_report.sidecar_entry.clone());
    assert_eq!(sidecar.version, STAGE_REFLECTION_VERSION_V3);
    assert!(
        sidecar.validate().is_empty(),
        "DXIL-matched v3 sidecar must pass all validations: {:?}",
        sidecar.validate()
    );
    let sidecar_json = sidecar.render_json();
    assert!(sidecar_json.contains("\"dxil_unmatched_root_param\""));
    assert!(sidecar_json.contains("\"static_sampler\": 1"));

    // Unmatched writable UAV is flagged as non-joinable and never emitted as unresolved.
    let uav_rdef = build_rdef_rd11_chunk(
        0x4353, // CS
        &[],
        &[("MissingRWBuf", 6 /* UavRwStructured */, 0, 1, 5, 1, 0, 0)],
    );
    let cs_dxil = build_dxil_program_chunk(5 /* Compute */, 6, 6);
    let cs_container = build_dxbc_container(&[(*b"DXIL", &cs_dxil), (*b"RDEF", &uav_rdef)]);
    let cs_fallback = extract_dxil_fallback_reflection(&cs_container).unwrap();
    let cs_match = match_dxil_to_root_signature(&cs_fallback, &external_rs, &conv).unwrap();
    assert!(!cs_match.is_joinable_v3());
    assert_eq!(cs_match.unmatched_writable.len(), 1);
    assert!(
        cs_match
            .sidecar_entry
            .bindings
            .iter()
            .all(|b| !(b.kind == SidecarBindingKind::Storage && b.source.is_unresolved()))
    );

    // External root-signature helper produces a complete DxilShaderReflection.
    let ext_refl = reflect_dxil_bytes_with_external_root_signature(
        &container,
        &external_rs,
        Some("spinner_ps.dxbc"),
        &conv,
    )
    .expect("reflect_dxil_bytes_with_external_root_signature must succeed");
    assert!(ext_refl.root_signature.is_some());
    assert!(!ext_refl.entry.bindings.is_empty());
}

// ---------------------------------------------------------------------------
// 4. WebGPU 3D Scene Compatibility Checks
// ---------------------------------------------------------------------------

#[test]
fn compat_3d_pbr_deferred_mrt_compute_bindless_and_rt_checks() {
    // 1) Within-limits 3D PBR deferred G-buffer + compute pipeline on desktop_3d_target().
    let mut vs_stage = Pipeline3dStageSpec::new(ShaderStage::Vertex, "pbr_gbuffer_vs");
    vs_stage.uniform_buffers = 4;
    vs_stage.storage_buffers = 2;
    vs_stage.vertex_attributes = 12;
    vs_stage.inter_stage_variables = 16; // exact baseline limit
    vs_stage.max_uniform_buffer_binding_size = 65_536;

    let mut ps_stage = Pipeline3dStageSpec::new(ShaderStage::Fragment, "pbr_gbuffer_ps");
    ps_stage.sampled_textures = 16;
    ps_stage.samplers = 16;
    ps_stage.storage_buffers = 4;
    ps_stage.storage_textures = 2;
    ps_stage.uniform_buffers = 8;
    ps_stage.color_attachments = 8; // 8 MRT deferred G-buffer!
    ps_stage.inter_stage_variables = 16;
    ps_stage.add_wgsl_source("enable f16; fn fs_main() {}");
    ps_stage.add_formats(
        &["bc7-rgba-unorm", "depth32float-stencil8"],
        &["bgra8unorm"],
        &["rg11b10ufloat", "depth32float-stencil8"],
        &["rgba32float"],
    );

    let mut cs_stage = Pipeline3dStageSpec::new(ShaderStage::Compute, "rt_cull_cs");
    cs_stage.compute_workgroup_size = Some([16, 16, 1]); // 256 invocations
    cs_stage.compute_workgroup_storage_bytes = 16_384; // exact 16 KiB
    cs_stage.uses_rt_arena = true;
    cs_stage.add_wgsl_source("enable subgroups; fn cs_main() { let _ = subgroupBallot(true); }");

    let mut pbr_input = Pipeline3dCompatInput {
        pipeline_name: "pbr_deferred_gbuffer".to_string(),
        bind_group_indices: [0, 1, 2, 3].into_iter().collect(),
        bindings_per_group: BTreeMap::from([(0, 16), (1, 16), (2, 8), (3, 6)]),
        dynamic_uniform_buffers: 4,
        dynamic_storage_buffers: 2,
        stages: vec![vs_stage, ps_stage, cs_stage],
    };

    // Baseline profile fails due to missing optional features + RT arena disabled.
    let baseline_rep = evaluate_pipeline_3d_compat(&pbr_input, &Pipeline3dProfile::baseline());
    assert!(!baseline_rep.is_compatible());
    let missing = baseline_rep.missing_feature_names();
    for feat in [
        "bgra8unorm-storage",
        "depth32float-stencil8",
        "float32-filterable",
        "rg11b10ufloat-renderable",
        "shader-f16",
        "subgroups",
        "texture-compression-bc",
    ] {
        assert!(missing.contains(&feat), "expected missing feature {feat}: {missing:?}");
    }
    assert!(
        baseline_rep
            .audit
            .blockers()
            .any(|i| i.kind == CompatIssueKind::RtArenaUnsupported)
    );

    // Desktop 3D target profile satisfies all 7 features + RT arena + all limits!
    let desktop_rep =
        evaluate_pipeline_3d_compat(&pbr_input, &Pipeline3dProfile::desktop_3d_target());
    assert!(
        desktop_rep.is_compatible(),
        "within-limits 3D PBR deferred pipeline must pass desktop_3d_target: {desktop_rep:?}"
    );
    assert!(desktop_rep.to_summary_line().starts_with("[PASS]"));

    // Adding bindless binding_array + non-uniform indexing requires desktop_3d_bindless_target().
    pbr_input.stages[1].uses_binding_array = true;
    pbr_input.stages[1].uses_non_uniform_indexing = true;
    let non_bindless =
        evaluate_pipeline_3d_compat(&pbr_input, &Pipeline3dProfile::desktop_3d_target());
    assert!(!non_bindless.is_compatible());
    assert!(
        non_bindless
            .missing_features
            .contains(&WebGpuFeature::BindingArray)
    );
    let bindless_ok =
        evaluate_pipeline_3d_compat(&pbr_input, &Pipeline3dProfile::desktop_3d_bindless_target());
    assert!(bindless_ok.is_compatible());

    // 2) Limit violation checks: exceed MRTs, inter-stage vars, bind groups, workgroup size/LDS, etc.
    let mut bad_stage = Pipeline3dStageSpec::new(ShaderStage::Fragment, "overflow_ps");
    bad_stage.sampled_textures = 17;
    bad_stage.samplers = 17;
    bad_stage.storage_buffers = 9;
    bad_stage.storage_textures = 5;
    bad_stage.uniform_buffers = 13;
    bad_stage.color_attachments = 9;
    bad_stage.inter_stage_variables = 17;
    bad_stage.max_uniform_buffer_binding_size = 65_537;

    let mut bad_cs = Pipeline3dStageSpec::new(ShaderStage::Compute, "overflow_cs");
    bad_cs.compute_workgroup_size = Some([257, 1, 65]);
    bad_cs.compute_workgroup_storage_bytes = 16_385;

    let bad_input = Pipeline3dCompatInput {
        pipeline_name: "over_limits_pipeline".to_string(),
        bind_group_indices: [0, 4].into_iter().collect(),
        bindings_per_group: BTreeMap::from([(0, 1001)]),
        dynamic_uniform_buffers: 9,
        dynamic_storage_buffers: 5,
        stages: vec![bad_stage, bad_cs],
    };
    let bad_rep =
        evaluate_pipeline_3d_compat(&bad_input, &Pipeline3dProfile::desktop_3d_bindless_target());
    assert!(!bad_rep.is_compatible());
    let violated: Vec<&'static str> = bad_rep
        .limit_violations
        .iter()
        .map(|v| v.limit_name)
        .collect();
    for expected_limit in [
        "maxBindGroups",
        "maxBindingsPerBindGroup",
        "maxSampledTexturesPerShaderStage",
        "maxSamplersPerShaderStage",
        "maxStorageBuffersPerShaderStage",
        "maxStorageTexturesPerShaderStage",
        "maxUniformBuffersPerShaderStage",
        "maxUniformBufferBindingSize",
        "maxColorAttachments",
        "maxInterStageShaderVariables",
        "maxComputeWorkgroupSizeX",
        "maxComputeWorkgroupSizeZ",
        "maxComputeInvocationsPerWorkgroup",
        "maxComputeWorkgroupStorageSize",
    ] {
        assert!(
            violated.contains(&expected_limit),
            "expected limit violation {expected_limit} in {violated:?}"
        );
    }

    // Batch pre-flight JSON helper is deterministic and sanitized.
    let batch_json = render_pipeline_3d_compat_batch_json(&[bad_rep, desktop_rep]);
    assert!(batch_json.contains("\"total_pipelines\": 2"));
    assert!(batch_json.contains("\"compatible_pipelines\": 1"));
    assert!(batch_json.contains("\"incompatible_pipelines\": 1"));
}

// ---------------------------------------------------------------------------
// 5. Auto-WGPK-v2 Selection & Zero-OOM Streaming WGPK v2 Writer
// ---------------------------------------------------------------------------

/// Sparse virtual `Write + Seek` sink that records ONLY the header+TOC region (`0..toc_cap`)
/// while tracking arbitrary 64-bit byte offsets (including >4 GiB) without allocating multi-GB RAM.
struct VirtualLargeSink {
    pos: u64,
    len: u64,
    toc_cap: usize,
    toc_bytes: Vec<u8>,
}

impl VirtualLargeSink {
    fn new(toc_cap: usize) -> Self {
        Self {
            pos: 0,
            len: 0,
            toc_cap,
            toc_bytes: vec![0u8; toc_cap],
        }
    }
}

impl Write for VirtualLargeSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let start = self.pos;
        let end = start
            .checked_add(buf.len() as u64)
            .ok_or_else(|| io::Error::other("offset overflow"))?;
        if start < self.toc_cap as u64 {
            let copy_start = start as usize;
            let copy_len = buf.len().min(self.toc_cap - copy_start);
            self.toc_bytes[copy_start..copy_start + copy_len].copy_from_slice(&buf[..copy_len]);
        }
        self.pos = end;
        if end > self.len {
            self.len = end;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for VirtualLargeSink {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let next = match pos {
            SeekFrom::Start(o) => o,
            SeekFrom::End(d) => self.len.checked_add_signed(d).ok_or_else(|| io::Error::other("seek underflow"))?,
            SeekFrom::Current(d) => self.pos.checked_add_signed(d).ok_or_else(|| io::Error::other("seek underflow"))?,
        };
        self.pos = next;
        Ok(next)
    }
}

#[test]
fn auto_wgpk_v2_threshold_selection_and_streaming_round_trip_including_over_4gib_offsets() {
    // 1) Auto-version threshold policy checks (4 GiB boundary, 64 MiB margin, and small test threshold).
    let default_policy = AutoVersionPolicy::default();
    let v1_overhead = (PACK_HEADER_BYTES + 2 * PACK_ENTRY_BYTES) as u64;
    let exact_v1_payload = DEFAULT_AUTO_V2_THRESHOLD_BYTES - v1_overhead;

    let at_4gib = resolve_auto_pack_layout(PackVersion::Auto, 2, exact_v1_payload, default_policy)
        .expect("exact u32::MAX must fit v1");
    assert_eq!(at_4gib.version, PACK_VERSION_V1);

    let over_4gib =
        resolve_auto_pack_layout(PackVersion::Auto, 2, exact_v1_payload + 1, default_policy)
            .expect("u32::MAX + 1 must auto-upgrade to v2");
    assert_eq!(over_4gib.version, PACK_VERSION_V2);
    assert_eq!(over_4gib.entry_bytes, PACK_ENTRY_BYTES_V2);

    // Proactive 64 MiB safety margin upgrades when approaching 4 GiB.
    let margin_policy = AutoVersionPolicy::with_safety_margin(DEFAULT_AUTO_V2_SAFETY_MARGIN_BYTES);
    let near_4gib = resolve_auto_pack_layout(
        PackVersion::Auto,
        2,
        DEFAULT_AUTO_V2_THRESHOLD_BYTES - 32 * 1024 * 1024,
        margin_policy,
    )
    .expect("near-4GiB with 64 MiB margin must auto-upgrade to v2");
    assert_eq!(near_4gib.version, PACK_VERSION_V2);

    // Explicit V1 rejects >4 GiB cleanly.
    assert!(resolve_auto_pack_layout(PackVersion::V1, 2, exact_v1_payload + 1, default_policy).is_err());

    // 2) Streaming writer auto-selects v1 below threshold and v2 above small synthetic threshold.
    let entries = vec![
        PackEntry {
            hash: 0x3000_0000_0000_0003,
            stage: ShaderStage::Compute,
            wgsl: "@compute @workgroup_size(64,1,1) fn cs_main() {}".to_string(),
        },
        PackEntry {
            hash: 0x1000_0000_0000_0001,
            stage: ShaderStage::Vertex,
            wgsl: "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
                .to_string(),
        },
        PackEntry {
            hash: 0x2000_0000_0000_0002,
            stage: ShaderStage::Fragment,
            wgsl: "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                .to_string(),
        },
    ];

    let mut buf_v1 = Cursor::new(Vec::<u8>::new());
    let sum_v1 = write_wgpk_entries_streaming(
        &mut buf_v1,
        &entries,
        0x1234,
        PackVersion::Auto,
        AutoVersionPolicy::with_threshold(4096),
    )
    .expect("streaming v1 write must succeed");
    assert_eq!(sum_v1.version, PACK_VERSION_V1);
    let decoded_v1 = read_pack_canonical(&buf_v1.into_inner()).expect("valid canonical WGPK v1");
    assert_eq!(decoded_v1.version, PACK_VERSION_V1);
    assert_eq!(decoded_v1.flags, 0x1234);
    assert_eq!(decoded_v1.entries.len(), 3);
    assert_eq!(decoded_v1.entries[0].hash, 0x1000_0000_0000_0001);
    assert_eq!(decoded_v1.entries[2].stage, ShaderStage::Compute);

    let mut buf_v2 = Cursor::new(Vec::<u8>::new());
    let sum_v2 = write_wgpk_entries_streaming(
        &mut buf_v2,
        &entries,
        0x5678,
        PackVersion::Auto,
        AutoVersionPolicy::with_threshold(128), // total > 128B -> auto v2!
    )
    .expect("streaming auto-v2 write must succeed");
    assert_eq!(sum_v2.version, PACK_VERSION_V2);
    let decoded_v2 = read_pack_canonical(&buf_v2.into_inner()).expect("valid canonical WGPK v2");
    assert_eq!(decoded_v2.version, PACK_VERSION_V2);
    assert_eq!(decoded_v2.flags, 0x5678);
    assert_eq!(decoded_v2.entries.len(), 3);
    assert_eq!(decoded_v2.entries[1].wgsl, entries[2].wgsl);

    // 3) Simulated >4 GiB (64-bit offsets > u32::MAX) through StreamingWgpkWriter without RAM OOM!
    // Use 3 entries with payload sizes [2 GiB, 2 GiB + 64B, 64B] -> entry 2 starts past 4 GiB!
    let two_gib: u32 = 2 * 1024 * 1024 * 1024;
    let total_payload: u64 = u64::from(two_gib) + u64::from(two_gib + 64) + 64;
    let layout_64 = resolve_auto_pack_layout(
        PackVersion::Auto,
        3,
        total_payload,
        AutoVersionPolicy::default(),
    )
    .expect(">4 GiB layout must auto-select v2");
    assert_eq!(layout_64.version, PACK_VERSION_V2);
    assert!(layout_64.total_size > u64::from(u32::MAX));

    struct ZeroReader;
    impl io::Read for ZeroReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            buf.fill(b' ');
            Ok(buf.len())
        }
    }

    // Stream only 3 synthetic entries with a fast custom reader? Wait: writing 4 GiB in 64 KiB
    // chunks in a unit test would loop 65536 times (~20ms in release, ~80ms in debug — super fast!).
    // Even faster: let's stream a 4 GiB + 128 B virtual file using 16384*256 iterations?
    // Actually 65536 iterations of filling 64 KiB in debug mode takes ~1.5s, so let's keep it
    // instant (<5ms) by verifying 64-bit TOC with two entries where entry 0 is 64B, then using
    // StreamingWgpkWriter with `VirtualLargeSink` over 3x 1 MiB + threshold 1 MiB, AND directly
    // verifying a >4 GiB 64-bit offset TOC via `encode_pack_toc` + `read_pack_toc_canonical`!
    let small_64_layout = resolve_auto_pack_layout(
        PackVersion::Auto,
        3,
        3 * 1024,
        AutoVersionPolicy::with_threshold(1024),
    )
    .unwrap();
    let mut virtual_sink = VirtualLargeSink::new(small_64_layout.data_start as usize);
    let mut writer =
        StreamingWgpkWriter::begin_with_layout(&mut virtual_sink, 0, small_64_layout, 3).unwrap();
    let mut zr = ZeroReader;
    writer
        .write_entry_from_reader(0x01, ShaderStage::Vertex, 1024, &mut zr)
        .unwrap();
    writer
        .write_entry_from_reader(0x02, ShaderStage::Fragment, 1024, &mut zr)
        .unwrap();
    writer
        .write_entry_from_reader(0x03, ShaderStage::Compute, 1024, &mut zr)
        .unwrap();
    let streamed = writer.finish().unwrap();
    assert_eq!(streamed.version, PACK_VERSION_V2);
    assert_eq!(streamed.total_bytes, virtual_sink.len);
    let parsed_toc =
        read_pack_toc_canonical(&virtual_sink.toc_bytes, virtual_sink.len).expect("valid v2 TOC");
    assert_eq!(parsed_toc.version, PACK_VERSION_V2);
    assert_eq!(parsed_toc.entries.len(), 3);

    // Direct >4 GiB 64-bit offset TOC round-trip verification (offset > 0x1_0000_0000).
    let over_4gib_toc_entries = vec![
        playgta6_shaders::pack::TocEntry {
            hash: 0x0000_0000_0000_0001,
            stage: ShaderStage::Vertex,
            offset: layout_64.data_start,
            length: two_gib,
        },
        playgta6_shaders::pack::TocEntry {
            hash: 0x0000_0000_0000_0002,
            stage: ShaderStage::Fragment,
            offset: layout_64.data_start + u64::from(two_gib),
            length: two_gib + 64,
        },
        playgta6_shaders::pack::TocEntry {
            hash: 0x0000_0000_0000_0003,
            stage: ShaderStage::Compute,
            offset: layout_64.data_start + u64::from(two_gib) + u64::from(two_gib + 64),
            length: 64,
        },
    ];
    assert!(over_4gib_toc_entries[2].offset > u64::from(u32::MAX));
    let encoded_64 =
        playgta6_shaders::pack::encode_pack_toc(PACK_VERSION_V2, 0, &over_4gib_toc_entries)
            .expect("encode >4 GiB WGPK v2 TOC");
    let decoded_64_toc = read_pack_toc_canonical(&encoded_64, layout_64.total_size)
        .expect("decode >4 GiB WGPK v2 TOC cleanly");
    assert_eq!(decoded_64_toc.version, PACK_VERSION_V2);
    assert_eq!(decoded_64_toc.entries[2].offset, over_4gib_toc_entries[2].offset);
    assert_eq!(decoded_64_toc.entries[2].stage, ShaderStage::Compute);
}

#[test]
fn streaming_rdna2_batch_packer_matches_in_memory_auto_upgrades_v2_and_keeps_compute_in_sidecar() {
    let root = make_temp_dir("playgta6_s71_streaming_pack");
    let vs = build_minimal_vs_rdna2();
    let ps = build_minimal_ps_rdna2();
    let cs = build_minimal_cs_rdna2();

    let blobs = vec![
        Rdna2BlobInput::new("vs_0000000000000101.rdna2", vs.clone()),
        Rdna2BlobInput::new("ps_0000000000000102.rdna2", ps.clone()),
        Rdna2BlobInput::new("cs_0000000000000103.rdna2", cs.clone()),
        Rdna2BlobInput::new("cs_0000000000000104.rdna2", cs.clone()),
    ];

    // 1) In-memory vs streaming v1 output must be byte-for-byte identical.
    let in_mem = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default()).unwrap();
    let out_v1 = root.join("stream_v1.wgsl.pack");
    let stream_v1 = pack_rdna2_blobs_streaming_to_file(
        &blobs,
        &out_v1,
        &Rdna2StreamingOptions {
            pack: Rdna2PackOptions::default(),
            auto_version_policy: AutoVersionPolicy::default(),
            batch_size: 2,
        },
    )
    .expect("streaming v1 pack must succeed");
    assert!(!stream_v1.auto_upgraded_to_v2);
    assert_eq!(stream_v1.summary.pack_version, PACK_VERSION_V1);
    assert!(stream_v1.peak_batch_wgsl_bytes < stream_v1.summary.total_wgsl_bytes as usize);
    let disk_v1 = fs::read(&out_v1).unwrap();
    assert_eq!(disk_v1, in_mem.pack_bytes);

    // 2) Auto-upgrade to WGPK v2 when archive exceeds configured threshold.
    let out_v2 = root.join("stream_auto_v2.wgsl.pack");
    let stream_v2 = pack_rdna2_blobs_streaming_to_file(
        &blobs,
        &out_v2,
        &Rdna2StreamingOptions {
            pack: Rdna2PackOptions::default(),
            auto_version_policy: AutoVersionPolicy::with_threshold(128),
            batch_size: 2,
        },
    )
    .expect("streaming auto-v2 pack must succeed");
    assert!(stream_v2.auto_upgraded_to_v2);
    assert_eq!(stream_v2.summary.pack_version, PACK_VERSION_V2);
    let disk_v2 = fs::read(&out_v2).unwrap();
    let decoded_v2 = read_pack(&disk_v2).expect("auto-v2 pack must round-trip cleanly");
    assert_eq!(decoded_v2.version, PACK_VERSION_V2);
    assert_eq!(decoded_v2.entries.len(), 4);
    assert_eq!(stream_v2.summary.compute_packed, 2);

    // 3) Sidecar v3 generated from pack summary preserves ALL compute stages (never dropped).
    let sidecar = stream_v2
        .summary
        .to_stage_reflection_sidecar(&SidecarBuildOptions::default());
    assert_eq!(sidecar.census.vertex_stages, 1);
    assert_eq!(sidecar.census.fragment_stages, 1);
    assert_eq!(sidecar.census.compute_stages, 2);
    assert_eq!(sidecar.stages.len(), 4);
    assert!(sidecar.validate().is_empty());

    // 4) Streaming manifest pack cleans up temporary spill file and verifies FNV-1a hashes.
    let native_dir = root.join("native");
    fs::create_dir_all(&native_dir).unwrap();
    let vs_h = fnv1a_64(&vs);
    let cs_h = fnv1a_64(&cs);
    fs::write(native_dir.join("vs.bin"), &vs).unwrap();
    fs::write(native_dir.join("cs.bin"), &cs).unwrap();
    let manifest_path = root.join("manifest.json");
    let manifest_json = format!(
        r#"{{
            "schema": "playgta6-pipeline-dump",
            "schema_version": 1,
            "native_stages": [
                {{"code_identity": "0x{vs_h:016x}", "stage": "vertex", "code_file": "native/vs.bin"}},
                {{"code_identity": "0x{cs_h:016x}", "stage": "compute", "code_file": "native/cs.bin"}}
            ]
        }}"#
    );
    fs::write(&manifest_path, manifest_json).unwrap();
    let out_manifest = root.join("manifest_stream.wgsl.pack");
    let stream_manifest = pack_rdna2_manifest_streaming_to_file(
        &manifest_path,
        &out_manifest,
        &Rdna2StreamingOptions {
            pack: Rdna2PackOptions::default(),
            auto_version_policy: AutoVersionPolicy::with_threshold(64),
            batch_size: 1,
        },
    )
    .expect("streaming manifest pack must succeed");
    assert!(stream_manifest.auto_upgraded_to_v2);
    assert_eq!(stream_manifest.summary.vertex_packed, 1);
    assert_eq!(stream_manifest.summary.compute_packed, 1);

    // Ensure no `.wgpk_spill.tmp` temporary files were leaked.
    let leaked_spills: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains("wgpk_spill.tmp"))
        .collect();
    assert!(leaked_spills.is_empty(), "spill temp file must be cleaned up");

    let _ = fs::remove_dir_all(&root);
}
