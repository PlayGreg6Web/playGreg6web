#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gpu/pso_cache.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_pso_cache.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu::pso;
using D = DxgiFormat;
using TF = WgpuTextureFormat;
using VF = WgpuVertexFormat;

bool has(uint64_t diagnostics, uint64_t bits) {
    return (diagnostics & bits) == bits;
}

const D3D12InputElementDesc kPositionOnly[] = {
    {"POSITION", 0, D::R32G32B32_FLOAT, 0, kD3D12AppendAlignedElement, D3D12InputClassification::PerVertexData, 0},
};

// A plain opaque-geometry PSO: VS+PS, one RGBA8 target, D32 depth, float3 positions.
D3D12GraphicsPsoDesc basic_desc() {
    D3D12GraphicsPsoDesc d{};
    d.root_signature_id = 7;
    d.vs = 0x1111;
    d.ps = 0x2222;
    d.NumRenderTargets = 1;
    d.RTVFormats[0] = D::R8G8B8A8_UNORM;
    d.DSVFormat = D::D32_FLOAT;
    d.pInputElementDescs = kPositionOnly;
    d.NumInputElements = 1;
    return d;
}

GraphicsPipelineKey key_of(const D3D12GraphicsPsoDesc& d, const PipelineVariantState& v = {},
                           const PipelineKeyOptions& o = {}) {
    return make_graphics_pipeline_key(d, v, o).key;
}

uint64_t diag_of(const D3D12GraphicsPsoDesc& d, const PipelineVariantState& v = {},
                 const PipelineKeyOptions& o = {}) {
    return make_graphics_pipeline_key(d, v, o).diagnostics;
}

// ---------------------------------------------------------------------------
// 1. Enum lowering tables (exhaustive over the raw D3D12 value ranges)
// ---------------------------------------------------------------------------

void test_enum_tables() {
    // D3D12_BLEND: 1..11 and 14..21 are valid.
    const WgpuBlendFactor expected_blend[22] = {
        WgpuBlendFactor::Count, WgpuBlendFactor::Zero, WgpuBlendFactor::One, WgpuBlendFactor::Src,
        WgpuBlendFactor::OneMinusSrc, WgpuBlendFactor::SrcAlpha, WgpuBlendFactor::OneMinusSrcAlpha,
        WgpuBlendFactor::DstAlpha, WgpuBlendFactor::OneMinusDstAlpha, WgpuBlendFactor::Dst,
        WgpuBlendFactor::OneMinusDst, WgpuBlendFactor::SrcAlphaSaturated, WgpuBlendFactor::Count,
        WgpuBlendFactor::Count, WgpuBlendFactor::Constant, WgpuBlendFactor::OneMinusConstant,
        WgpuBlendFactor::Src1, WgpuBlendFactor::OneMinusSrc1, WgpuBlendFactor::Src1Alpha,
        WgpuBlendFactor::OneMinusSrc1Alpha, WgpuBlendFactor::Constant, WgpuBlendFactor::OneMinusConstant,
    };
    for (uint32_t raw = 0; raw < 64; ++raw) {
        const auto m = map_blend_factor(static_cast<D3D12Blend>(raw));
        const WgpuBlendFactor want = raw < 22 ? expected_blend[raw] : WgpuBlendFactor::Count;
        CHECK(m.has_value() == (want != WgpuBlendFactor::Count));
        if (m) {
            CHECK(*m == want);
        }
    }

    const WgpuBlendOperation expected_op[6] = {WgpuBlendOperation::Count, WgpuBlendOperation::Add,
                                               WgpuBlendOperation::Subtract, WgpuBlendOperation::ReverseSubtract,
                                               WgpuBlendOperation::Min, WgpuBlendOperation::Max};
    for (uint32_t raw = 0; raw < 16; ++raw) {
        const auto m = map_blend_op(static_cast<D3D12BlendOp>(raw));
        const WgpuBlendOperation want = raw < 6 ? expected_op[raw] : WgpuBlendOperation::Count;
        CHECK(m.has_value() == (want != WgpuBlendOperation::Count));
        if (m) {
            CHECK(*m == want);
        }
    }

    const WgpuCompareFunction expected_cmp[9] = {
        WgpuCompareFunction::Undefined, WgpuCompareFunction::Never,   WgpuCompareFunction::Less,
        WgpuCompareFunction::Equal,     WgpuCompareFunction::LessEqual, WgpuCompareFunction::Greater,
        WgpuCompareFunction::NotEqual,  WgpuCompareFunction::GreaterEqual, WgpuCompareFunction::Always};
    for (uint32_t raw = 0; raw < 16; ++raw) {
        const auto m = map_compare_function(static_cast<D3D12ComparisonFunc>(raw));
        const bool valid = raw >= 1 && raw <= 8;
        CHECK(m.has_value() == valid);
        if (m) {
            CHECK(*m == expected_cmp[raw]);
        }
    }

    const WgpuStencilOperation expected_sop[9] = {
        WgpuStencilOperation::Count,          WgpuStencilOperation::Keep,           WgpuStencilOperation::Zero,
        WgpuStencilOperation::Replace,        WgpuStencilOperation::IncrementClamp, WgpuStencilOperation::DecrementClamp,
        WgpuStencilOperation::Invert,         WgpuStencilOperation::IncrementWrap,  WgpuStencilOperation::DecrementWrap};
    for (uint32_t raw = 0; raw < 16; ++raw) {
        const auto m = map_stencil_op(static_cast<D3D12StencilOp>(raw));
        const bool valid = raw >= 1 && raw <= 8;
        CHECK(m.has_value() == valid);
        if (m) {
            CHECK(*m == expected_sop[raw]);
        }
    }

    for (uint32_t raw = 0; raw < 8; ++raw) {
        const auto m = map_cull_mode(static_cast<D3D12CullMode>(raw));
        CHECK(m.has_value() == (raw >= 1 && raw <= 3));
    }
    CHECK(map_cull_mode(D3D12CullMode::None) == WgpuCullMode::None);
    CHECK(map_cull_mode(D3D12CullMode::Front) == WgpuCullMode::Front);
    CHECK(map_cull_mode(D3D12CullMode::Back) == WgpuCullMode::Back);
    // No Y flip: D3D12 clockwise-front stays clockwise-front.
    CHECK(map_front_face(false) == WgpuFrontFace::Cw);
    CHECK(map_front_face(true) == WgpuFrontFace::Ccw);

    for (uint32_t raw = 0; raw < 128; ++raw) {
        const TopologyMapping t = map_primitive_topology(static_cast<D3DPrimitiveTopology>(raw));
        if (raw >= 33 && raw <= 64) {
            CHECK(t.type == D3D12PrimitiveTopologyType::Patch);
            CHECK(has(t.diagnostics, PipelineDiag::TessellationUnsupported));
        } else if ((raw >= 1 && raw <= 6) || (raw >= 10 && raw <= 13)) {
            CHECK(t.type != D3D12PrimitiveTopologyType::Undefined);
            CHECK(!has(t.diagnostics, PipelineDiag::InvalidEnumValue));
        } else {
            CHECK(has(t.diagnostics, PipelineDiag::InvalidEnumValue));
        }
    }
    CHECK(map_primitive_topology(D3DPrimitiveTopology::PointList).topology == WgpuPrimitiveTopology::PointList);
    CHECK(map_primitive_topology(D3DPrimitiveTopology::LineStrip).topology == WgpuPrimitiveTopology::LineStrip);
    CHECK(map_primitive_topology(D3DPrimitiveTopology::TriangleStrip).topology == WgpuPrimitiveTopology::TriangleStrip);
    CHECK(map_primitive_topology(D3DPrimitiveTopology::TriangleFan).diagnostics ==
          PipelineDiag::TriangleFanNeedsConversion);
    CHECK(map_primitive_topology(D3DPrimitiveTopology::TriangleStripAdj).topology == WgpuPrimitiveTopology::TriangleStrip);
    CHECK(has(map_primitive_topology(D3DPrimitiveTopology::LineListAdj).diagnostics,
              PipelineDiag::AdjacencyTopologyUnsupported));
    // WgpuPrimitiveTopology shares the frozen packet numbering.
    CHECK(static_cast<uint32_t>(WgpuPrimitiveTopology::PointList) ==
          static_cast<uint32_t>(playgta6::GpuPrimitiveTopology::PointList));
    CHECK(std::string(primitive_topology_name(WgpuPrimitiveTopology::TriangleStrip)) == "triangle-strip");

    for (uint32_t s = 0; s <= 32; ++s) {
        CHECK(map_sample_count(s) == (s <= 1 ? 1u : 4u));
    }

    // Every WebGPU enum value has a unique GPU* string; out-of-range values have none.
    auto unique_names = [](auto name_fn, uint32_t count) {
        std::set<std::string> names;
        bool all = true;
        for (uint32_t i = 0; i < count; ++i) {
            const char* n = name_fn(i);
            all = all && n != nullptr && *n != '\0';
            if (n != nullptr) {
                names.insert(n);
            }
        }
        return all && names.size() == count && name_fn(count) == nullptr;
    };
    CHECK(unique_names([](uint32_t i) { return texture_format_name(static_cast<TF>(i)); },
                       static_cast<uint32_t>(TF::Count)));
    CHECK(unique_names([](uint32_t i) { return vertex_format_name(static_cast<VF>(i)); },
                       static_cast<uint32_t>(VF::Count)));
    CHECK(unique_names([](uint32_t i) { return blend_factor_name(static_cast<WgpuBlendFactor>(i)); },
                       static_cast<uint32_t>(WgpuBlendFactor::Count)));
    CHECK(unique_names([](uint32_t i) { return blend_operation_name(static_cast<WgpuBlendOperation>(i)); },
                       static_cast<uint32_t>(WgpuBlendOperation::Count)));
    CHECK(unique_names([](uint32_t i) { return compare_function_name(static_cast<WgpuCompareFunction>(i)); },
                       static_cast<uint32_t>(WgpuCompareFunction::Count)));
    CHECK(unique_names([](uint32_t i) { return stencil_operation_name(static_cast<WgpuStencilOperation>(i)); },
                       static_cast<uint32_t>(WgpuStencilOperation::Count)));
    CHECK(unique_names([](uint32_t i) { return cull_mode_name(static_cast<WgpuCullMode>(i)); },
                       static_cast<uint32_t>(WgpuCullMode::Count)));
    CHECK(unique_names([](uint32_t i) { return front_face_name(static_cast<WgpuFrontFace>(i)); },
                       static_cast<uint32_t>(WgpuFrontFace::Count)));
    CHECK(unique_names([](uint32_t i) { return primitive_topology_name(static_cast<WgpuPrimitiveTopology>(i)); },
                       static_cast<uint32_t>(WgpuPrimitiveTopology::Count)));
    CHECK(unique_names([](uint32_t i) { return index_format_name(static_cast<WgpuIndexFormat>(i)); },
                       static_cast<uint32_t>(WgpuIndexFormat::Count)));
    CHECK(unique_names([](uint32_t i) { return load_op_name(static_cast<WgpuLoadOp>(i)); },
                       static_cast<uint32_t>(WgpuLoadOp::Count)));
    CHECK(unique_names([](uint32_t i) { return store_op_name(static_cast<WgpuStoreOp>(i)); },
                       static_cast<uint32_t>(WgpuStoreOp::Count)));
    std::set<std::string> features;
    for (uint32_t bit = 1; bit <= WgpuFeature::All; bit <<= 1) {
        CHECK(feature_name(bit) != nullptr);
        if (feature_name(bit) != nullptr) {
            features.insert(feature_name(bit));
        }
    }
    CHECK(features.size() == 9);
    CHECK(feature_name(WgpuFeature::All + 1) == nullptr);
    CHECK(std::string(feature_name(WgpuFeature::Float32Filterable)) == "float32-filterable");
    CHECK(std::string(feature_name(WgpuFeature::Bgra8UnormStorage)) == "bgra8unorm-storage");
    CHECK((WgpuFeature::All & WgpuFeature::Bgra8UnormStorage) == WgpuFeature::Bgra8UnormStorage);
    CHECK(WgpuFeature::Bgra8UnormStorage == (WgpuFeature::TextureFormatsTier1 << 1));
    CHECK(std::string(texture_format_name(TF::Depth24PlusStencil8)) == "depth24plus-stencil8");
    CHECK(std::string(vertex_format_name(VF::Unorm10_10_10_2)) == "unorm10-10-10-2");
    CHECK(std::string(blend_factor_name(WgpuBlendFactor::SrcAlphaSaturated)) == "src-alpha-saturated");
}

// ---------------------------------------------------------------------------
// 2. DXGI -> texture format mapping (exhaustive invariants + spot checks)
// ---------------------------------------------------------------------------

void test_texture_format_mapping() {
    const FormatUsage usages[] = {FormatUsage::RenderTarget, FormatUsage::DepthStencil, FormatUsage::Sampled};
    const uint32_t feature_sets[] = {0u, WgpuFeature::All};
    uint32_t rt_supported_core = 0;
    for (uint32_t raw = 0; raw <= 140; ++raw) {
        for (FormatUsage usage : usages) {
            for (uint32_t features : feature_sets) {
                const TextureFormatMapping m = map_texture_format(static_cast<D>(raw), usage, features);
                CHECK(m.supported() == ((m.diagnostics & FormatDiag::Unsupported) == 0));
                CHECK(static_cast<uint32_t>(m.format) < static_cast<uint32_t>(TF::Count));
                CHECK((m.required_features & ~features) == 0);
                if (!m.supported()) {
                    continue;
                }
                const auto& info = texture_format_info(m.format);
                CHECK((info.required_feature & ~features) == 0);
                if (usage == FormatUsage::RenderTarget) {
                    CHECK(is_color_renderable(m.format, features));
                    CHECK((info.caps & TextureFormatCaps::Compressed) == 0);
                    if (features == 0) {
                        ++rt_supported_core;
                    }
                }
                if (usage == FormatUsage::DepthStencil) {
                    CHECK(has_depth_aspect(m.format) || has_stencil_aspect(m.format));
                }
            }
        }
    }
    CHECK(rt_supported_core > 40);

    struct Expect {
        D dxgi;
        FormatUsage usage;
        uint32_t features;
        TF format;
        uint32_t diag;  // bits that must be present (exact match when supported)
    };
    const Expect expects[] = {
        {D::R8G8B8A8_UNORM, FormatUsage::RenderTarget, 0, TF::Rgba8Unorm, 0},
        {D::R8G8B8A8_UNORM_SRGB, FormatUsage::RenderTarget, 0, TF::Rgba8UnormSrgb, 0},
        {D::B8G8R8A8_UNORM_SRGB, FormatUsage::RenderTarget, 0, TF::Bgra8UnormSrgb, 0},
        {D::B8G8R8X8_UNORM, FormatUsage::RenderTarget, 0, TF::Bgra8Unorm, FormatDiag::AlphaIgnored},
        {D::R16G16B16A16_FLOAT, FormatUsage::RenderTarget, 0, TF::Rgba16Float, 0},
        {D::R10G10B10A2_UNORM, FormatUsage::RenderTarget, 0, TF::Rgb10a2Unorm, 0},
        {D::R32G32B32A32_FLOAT, FormatUsage::RenderTarget, 0, TF::Rgba32Float, 0},
        {D::R8G8B8A8_TYPELESS, FormatUsage::RenderTarget, 0, TF::Rgba8Unorm,
         FormatDiag::TypelessResolved | FormatDiag::NeedsViewReinterpretation},
        {D::R11G11B10_FLOAT, FormatUsage::RenderTarget, 0, TF::Rgba16Float, FormatDiag::FeatureFallback},
        {D::R11G11B10_FLOAT, FormatUsage::RenderTarget, WgpuFeature::Rg11b10UfloatRenderable, TF::Rg11b10Ufloat, 0},
        {D::R11G11B10_FLOAT, FormatUsage::Sampled, 0, TF::Rg11b10Ufloat, 0},
        {D::R8G8B8A8_SNORM, FormatUsage::RenderTarget, 0, TF::Rgba16Float, FormatDiag::FeatureFallback},
        {D::R8G8B8A8_SNORM, FormatUsage::RenderTarget, WgpuFeature::TextureFormatsTier1, TF::Rgba8Snorm, 0},
        {D::R8G8B8A8_SNORM, FormatUsage::Sampled, 0, TF::Rgba8Snorm, 0},
        {D::R16G16_UNORM, FormatUsage::Sampled, 0, TF::Rg32Float,
         FormatDiag::FeatureFallback | FormatDiag::NeedsCpuConversion},
        {D::R16G16_UNORM, FormatUsage::RenderTarget, WgpuFeature::TextureFormatsTier1, TF::Rg16Unorm, 0},
        {D::D32_FLOAT, FormatUsage::DepthStencil, 0, TF::Depth32Float, 0},
        {D::D24_UNORM_S8_UINT, FormatUsage::DepthStencil, 0, TF::Depth24PlusStencil8, 0},
        {D::D16_UNORM, FormatUsage::DepthStencil, 0, TF::Depth16Unorm, 0},
        {D::D32_FLOAT_S8X24_UINT, FormatUsage::DepthStencil, 0, TF::Depth24PlusStencil8, FormatDiag::FeatureFallback},
        {D::D32_FLOAT_S8X24_UINT, FormatUsage::DepthStencil, WgpuFeature::Depth32FloatStencil8,
         TF::Depth32FloatStencil8, 0},
        {D::R32_TYPELESS, FormatUsage::DepthStencil, 0, TF::Depth32Float,
         FormatDiag::TypelessResolved | FormatDiag::NeedsViewReinterpretation},
        {D::R24G8_TYPELESS, FormatUsage::DepthStencil, 0, TF::Depth24PlusStencil8,
         FormatDiag::TypelessResolved | FormatDiag::NeedsViewReinterpretation},
        {D::R16_TYPELESS, FormatUsage::DepthStencil, 0, TF::Depth16Unorm,
         FormatDiag::TypelessResolved | FormatDiag::NeedsViewReinterpretation},
        {D::BC1_UNORM, FormatUsage::Sampled, WgpuFeature::TextureCompressionBc, TF::Bc1RgbaUnorm, 0},
        {D::BC1_UNORM_SRGB, FormatUsage::Sampled, 0, TF::Rgba8UnormSrgb,
         FormatDiag::FeatureFallback | FormatDiag::NeedsCpuConversion},
        {D::BC5_SNORM, FormatUsage::Sampled, 0, TF::Rg8Snorm,
         FormatDiag::FeatureFallback | FormatDiag::NeedsCpuConversion},
        {D::BC6H_UF16, FormatUsage::Sampled, 0, TF::Rgba16Float,
         FormatDiag::FeatureFallback | FormatDiag::NeedsCpuConversion},
        {D::BC7_UNORM_SRGB, FormatUsage::Sampled, WgpuFeature::TextureCompressionBc, TF::Bc7RgbaUnormSrgb, 0},
        {D::R9G9B9E5_SHAREDEXP, FormatUsage::Sampled, 0, TF::Rgb9e5Ufloat, 0},
        {D::R9G9B9E5_SHAREDEXP, FormatUsage::RenderTarget, 0, TF::Rgba16Float, FormatDiag::FeatureFallback},
        {D::R9G9B9E5_SHAREDEXP, FormatUsage::RenderTarget, WgpuFeature::All, TF::Rgba16Float,
         FormatDiag::FeatureFallback},
        {D::R8_UNORM, FormatUsage::Sampled, 0, TF::R8Unorm, 0},
        {D::R8_UNORM, FormatUsage::RenderTarget, 0, TF::R8Unorm, 0},
        // A8_UNORM: rgba8unorm holding the value in alpha (uploads widen, targets write alpha only).
        {D::A8_UNORM, FormatUsage::Sampled, 0, TF::Rgba8Unorm, FormatDiag::FeatureFallback | FormatDiag::NeedsCpuConversion},
        {D::A8_UNORM, FormatUsage::RenderTarget, 0, TF::Rgba8Unorm,
         FormatDiag::FeatureFallback | FormatDiag::NeedsCpuConversion},
    };
    for (const Expect& e : expects) {
        const TextureFormatMapping m = map_texture_format(e.dxgi, e.usage, e.features);
        CHECK(m.format == e.format);
        CHECK(m.diagnostics == e.diag);
    }
    // R9G9B9E5_SHAREDEXP is sampled-only in WebGPU: Sampled uses rgb9e5ufloat directly, while
    // RenderTarget falls back to core rgba16float with no extra device features required.
    {
        const TextureFormatMapping sampled = map_texture_format(D::R9G9B9E5_SHAREDEXP, FormatUsage::Sampled, 0);
        CHECK(sampled.supported());
        CHECK(sampled.format == TF::Rgb9e5Ufloat);
        CHECK((sampled.diagnostics & FormatDiag::FeatureFallback) == 0);
        CHECK(sampled.required_features == 0);

        const TextureFormatMapping rt = map_texture_format(D::R9G9B9E5_SHAREDEXP, FormatUsage::RenderTarget, 0);
        CHECK(rt.supported());
        CHECK(rt.format == TF::Rgba16Float);
        CHECK(has(rt.diagnostics, FormatDiag::FeatureFallback));
        CHECK(rt.required_features == 0);

        const TextureFormatMapping ds = map_texture_format(D::R9G9B9E5_SHAREDEXP, FormatUsage::DepthStencil, 0);
        CHECK(!ds.supported());
        CHECK(has(ds.diagnostics, FormatDiag::Unsupported));
    }
    // G59: packed 16-bit B5G5R5A1_UNORM (86) / B4G4R4A4_UNORM (115) promote to core rgba8unorm for
    // Sampled and RenderTarget (no device features needed); there is no depth-stencil form.
    {
        CHECK(static_cast<uint32_t>(D::B5G5R5A1_UNORM) == 86);
        CHECK(static_cast<uint32_t>(D::B4G4R4A4_UNORM) == 115);
        for (D packed : {D::B5G5R5A1_UNORM, D::B4G4R4A4_UNORM}) {
            for (uint32_t features : {0u, WgpuFeature::All}) {
                for (FormatUsage usage : {FormatUsage::Sampled, FormatUsage::RenderTarget}) {
                    const TextureFormatMapping m = map_texture_format(packed, usage, features);
                    CHECK(m.supported());
                    CHECK(m.format == TF::Rgba8Unorm);
                    CHECK(m.aspect == WgpuTextureAspect::All);
                    CHECK(m.required_features == 0);
                    CHECK(m.diagnostics == (FormatDiag::FeatureFallback | FormatDiag::NeedsCpuConversion));
                }
                const TextureFormatMapping ds = map_texture_format(packed, FormatUsage::DepthStencil, features);
                CHECK(!ds.supported());
                CHECK(ds.diagnostics == FormatDiag::Unsupported);
            }
        }
    }
    // G59: Xbox D16_UNORM_S8_UINT (118) promotes to depth24plus-stencil8 for DepthStencil and Sampled with
    // no device features; it is never a color target. Its R16_UNORM_X8 (119) / X16_G8 (120) views sample
    // the depth / stencil aspect of the same promoted format.
    {
        CHECK(static_cast<uint32_t>(D::D16_UNORM_S8_UINT) == 118);
        CHECK(static_cast<uint32_t>(D::R16_UNORM_X8_TYPELESS) == 119);
        CHECK(static_cast<uint32_t>(D::X16_TYPELESS_G8_UINT) == 120);
        for (uint32_t features : {0u, WgpuFeature::All}) {
            const TextureFormatMapping ds = map_texture_format(D::D16_UNORM_S8_UINT, FormatUsage::DepthStencil, features);
            CHECK(ds.supported());
            CHECK(ds.format == TF::Depth24PlusStencil8);
            CHECK(ds.aspect == WgpuTextureAspect::All);
            CHECK(ds.required_features == 0);
            CHECK(ds.diagnostics == FormatDiag::FeatureFallback);
            CHECK(has_depth_aspect(ds.format) && has_stencil_aspect(ds.format));

            const TextureFormatMapping sampled = map_texture_format(D::D16_UNORM_S8_UINT, FormatUsage::Sampled, features);
            CHECK(sampled.supported());
            CHECK(sampled.format == TF::Depth24PlusStencil8);
            CHECK(sampled.aspect == WgpuTextureAspect::All);
            CHECK(sampled.required_features == 0);
            CHECK(sampled.diagnostics == FormatDiag::FeatureFallback);

            const TextureFormatMapping rt = map_texture_format(D::D16_UNORM_S8_UINT, FormatUsage::RenderTarget, features);
            CHECK(!rt.supported());
            CHECK(rt.diagnostics == FormatDiag::Unsupported);

            const TextureFormatMapping depth_view =
                map_texture_format(D::R16_UNORM_X8_TYPELESS, FormatUsage::Sampled, features);
            CHECK(depth_view.supported());
            CHECK(depth_view.format == TF::Depth24PlusStencil8);
            CHECK(depth_view.aspect == WgpuTextureAspect::DepthOnly);
            CHECK(depth_view.required_features == 0);
            CHECK(depth_view.diagnostics == FormatDiag::FeatureFallback);

            const TextureFormatMapping stencil_view =
                map_texture_format(D::X16_TYPELESS_G8_UINT, FormatUsage::Sampled, features);
            CHECK(stencil_view.supported());
            CHECK(stencil_view.format == TF::Depth24PlusStencil8);
            CHECK(stencil_view.aspect == WgpuTextureAspect::StencilOnly);
            CHECK(stencil_view.required_features == 0);
            CHECK(stencil_view.diagnostics == FormatDiag::FeatureFallback);
        }
        // The 16-bit D16_UNORM (no stencil) keeps its exact depth16unorm mapping.
        const TextureFormatMapping d16 = map_texture_format(D::D16_UNORM, FormatUsage::DepthStencil, 0);
        CHECK(d16.format == TF::Depth16Unorm);
        CHECK(d16.diagnostics == 0);
    }
    // A8_UNORM maps to core rgba8unorm with no extra device features required.
    CHECK(map_texture_format(D::A8_UNORM, FormatUsage::Sampled, 0).required_features == 0);
    CHECK(map_texture_format(D::A8_UNORM, FormatUsage::RenderTarget, 0).required_features == 0);
    // Depth-sampling views carry the aspect.
    CHECK(map_texture_format(D::R24_UNORM_X8_TYPELESS, FormatUsage::Sampled, 0).aspect ==
          WgpuTextureAspect::DepthOnly);
    CHECK(map_texture_format(D::X24_TYPELESS_G8_UINT, FormatUsage::Sampled, 0).aspect ==
          WgpuTextureAspect::StencilOnly);
    CHECK(map_texture_format(D::R11G11B10_FLOAT, FormatUsage::RenderTarget, WgpuFeature::Rg11b10UfloatRenderable)
              .required_features == WgpuFeature::Rg11b10UfloatRenderable);
    CHECK(map_texture_format(D::BC3_UNORM, FormatUsage::Sampled, WgpuFeature::TextureCompressionBc).required_features ==
          WgpuFeature::TextureCompressionBc);

    // Unsupported for the requested usage.
    const std::pair<D, FormatUsage> unsupported[] = {
        {D::UNKNOWN, FormatUsage::RenderTarget},       {D::R32G32B32_FLOAT, FormatUsage::RenderTarget},
        {D::R32G32B32_FLOAT, FormatUsage::Sampled},    {D::B5G6R5_UNORM, FormatUsage::Sampled},
        {D::B5G5R5A1_UNORM, FormatUsage::DepthStencil},{D::A8_UNORM, FormatUsage::DepthStencil},
        {D::R1_UNORM, FormatUsage::Sampled},           {D::B4G4R4A4_UNORM, FormatUsage::DepthStencil},
        {D::D16_UNORM_S8_UINT, FormatUsage::RenderTarget},
        {D::R16_UNORM_X8_TYPELESS, FormatUsage::RenderTarget},
        {D::X16_TYPELESS_G8_UINT, FormatUsage::RenderTarget},
        {D::BC1_UNORM, FormatUsage::RenderTarget},     {D::BC7_UNORM, FormatUsage::DepthStencil},
        {D::D32_FLOAT, FormatUsage::RenderTarget},     {D::R32_FLOAT, FormatUsage::DepthStencil},
        {D::R9G9B9E5_SHAREDEXP, FormatUsage::DepthStencil}, {static_cast<D>(103), FormatUsage::Sampled},
        {D::R10G10B10_XR_BIAS_A2_UNORM, FormatUsage::RenderTarget},
    };
    for (const auto& [dxgi, usage] : unsupported) {
        const TextureFormatMapping m = map_texture_format(dxgi, usage, WgpuFeature::All);
        CHECK(!m.supported());
        CHECK(has(m.diagnostics, FormatDiag::Unsupported));
    }

    // Capability queries.
    CHECK(is_blendable(TF::Rgba8Unorm, 0));
    CHECK(!is_blendable(TF::Rgba8Uint, WgpuFeature::All));
    CHECK(!is_blendable(TF::R32Float, 0));
    CHECK(is_blendable(TF::R32Float, WgpuFeature::Float32Blendable));
    CHECK(!is_color_renderable(TF::Bc1RgbaUnorm, WgpuFeature::All));
    CHECK(!is_color_renderable(TF::R16Unorm, 0));
    CHECK(is_color_renderable(TF::R16Unorm, WgpuFeature::TextureFormatsTier1));
    CHECK(has_depth_aspect(TF::Depth24PlusStencil8) && has_stencil_aspect(TF::Depth24PlusStencil8));
    CHECK(!has_stencil_aspect(TF::Depth32Float));
    CHECK(has_alpha_channel(TF::Bgra8Unorm) && !has_alpha_channel(TF::Rg16Float));

    // WebGPU color attachment bytes per sample (byte cost + component alignment).
    const TF eight_rgba8[8] = {TF::Rgba8Unorm, TF::Rgba8Unorm, TF::Rgba8Unorm, TF::Rgba8Unorm,
                               TF::Rgba8Unorm, TF::Rgba8Unorm, TF::Rgba8Unorm, TF::Rgba8Unorm};
    CHECK(color_attachment_bytes_per_sample(std::span<const TF>(eight_rgba8, 4)) == 32);
    CHECK(color_attachment_bytes_per_sample(std::span<const TF>(eight_rgba8, 8)) == 64);
    const TF mixed[3] = {TF::R8Unorm, TF::Undefined, TF::R32Float};
    CHECK(color_attachment_bytes_per_sample(std::span<const TF>(mixed, 3)) == 8);  // 1 -> align 4 -> 8
    const TF uints[2] = {TF::Rgba8Uint, TF::R16Float};
    CHECK(color_attachment_bytes_per_sample(std::span<const TF>(uints, 2)) == 6);
}

// ---------------------------------------------------------------------------
// 3. DXGI -> vertex format mapping
// ---------------------------------------------------------------------------

// G66: 16-bit normalized formats are only unfilterable-float in WebGPU. A SampledOnly texture gets a
// filterable float representation: float32 (exact) with float32-filterable, else float16 (lossy).
void test_sampled_only_norm16() {
    using D = DxgiFormat;
    const uint32_t tier1 = WgpuFeature::TextureFormatsTier1;
    const uint32_t with_f32 = tier1 | WgpuFeature::Float32Filterable;
    struct Case {
        D dxgi;
        TF f32, f16;
    };
    for (const Case& c : {Case{D::R16_UNORM, TF::R32Float, TF::R16Float}, Case{D::R16_SNORM, TF::R32Float, TF::R16Float},
                          Case{D::R16G16_UNORM, TF::Rg32Float, TF::Rg16Float},
                          Case{D::R16G16B16A16_SNORM, TF::Rgba32Float, TF::Rgba16Float}}) {
        // Sampled (render targets and their views) keeps the native format.
        CHECK(map_texture_format(c.dxgi, FormatUsage::Sampled, tier1).format != c.f32);
        const TextureFormatMapping exact = map_texture_format(c.dxgi, FormatUsage::SampledOnly, with_f32);
        CHECK(exact.format == c.f32 && exact.required_features == WgpuFeature::Float32Filterable);
        CHECK((exact.diagnostics & FormatDiag::NeedsCpuConversion) != 0);
        CHECK((exact.diagnostics & FormatDiag::LossyConversion) == 0);
        const TextureFormatMapping lossy = map_texture_format(c.dxgi, FormatUsage::SampledOnly, tier1);
        CHECK(lossy.format == c.f16 && lossy.required_features == 0);
        CHECK((lossy.diagnostics & (FormatDiag::NeedsCpuConversion | FormatDiag::LossyConversion)) ==
              (FormatDiag::NeedsCpuConversion | FormatDiag::LossyConversion));
        CHECK(map_texture_format(c.dxgi, FormatUsage::SampledOnly, 0).format == c.f16);
    }
    // Other formats: SampledOnly is Sampled.
    for (D d : {D::R8G8B8A8_UNORM, D::R16_FLOAT, D::R32_FLOAT, D::BC7_UNORM}) {
        const TextureFormatMapping a = map_texture_format(d, FormatUsage::Sampled, WgpuFeature::All);
        const TextureFormatMapping b = map_texture_format(d, FormatUsage::SampledOnly, WgpuFeature::All);
        CHECK(a.format == b.format && a.diagnostics == b.diagnostics && a.required_features == b.required_features);
    }

    // Texel conversion.
    const uint16_t unorm[4] = {0, 65535, 32768, 1};
    float f[4] = {};
    CHECK(convert_norm16_texels(D::R16_UNORM, TF::R32Float, reinterpret_cast<const uint8_t*>(unorm), 4,
                                reinterpret_cast<uint8_t*>(f)));
    CHECK(f[0] == 0.0f && f[1] == 1.0f && f[2] == 32768.0f / 65535.0f && f[3] == 1.0f / 65535.0f);
    const int16_t snorm[4] = {-32768, -32767, 32767, 0};
    CHECK(convert_norm16_texels(D::R16_SNORM, TF::R32Float, reinterpret_cast<const uint8_t*>(snorm), 4,
                                reinterpret_cast<uint8_t*>(f)));
    CHECK(f[0] == -1.0f && f[1] == -1.0f && f[2] == 1.0f && f[3] == 0.0f);
    uint16_t h[4] = {};
    CHECK(convert_norm16_texels(D::R16_UNORM, TF::R16Float, reinterpret_cast<const uint8_t*>(unorm), 4,
                                reinterpret_cast<uint8_t*>(h)));
    // 0 -> 0x0000, 1.0 -> 0x3C00, 0.500007 -> 0x3800 (rounded), 1/65535 -> subnormal 0x0100 (256 x 2^-24).
    CHECK(h[0] == 0x0000 && h[1] == 0x3C00 && h[2] == 0x3800 && h[3] == 0x0100);
    // Mismatched pairs are refused.
    CHECK(!convert_norm16_texels(D::R16_UNORM, TF::Rg32Float, reinterpret_cast<const uint8_t*>(unorm), 4,
                                 reinterpret_cast<uint8_t*>(f)));
    CHECK(!convert_norm16_texels(D::R8G8B8A8_UNORM, TF::R32Float, reinterpret_cast<const uint8_t*>(unorm), 4,
                                 reinterpret_cast<uint8_t*>(f)));
    // A8_UNORM texels widen to (0, 0, 0, a); other pairs are refused.
    const uint8_t a8[3] = {0x00, 0x7f, 0xff};
    uint8_t wide[12] = {};
    CHECK(widen_a8_texels(D::A8_UNORM, TF::Rgba8Unorm, a8, 3, wide));
    CHECK(wide[3] == 0x00 && wide[7] == 0x7f && wide[11] == 0xff && wide[0] == 0 && wide[4] == 0 && wide[10] == 0);
    CHECK(!widen_a8_texels(D::R8_UNORM, TF::Rgba8Unorm, a8, 3, wide));
    CHECK(!widen_a8_texels(D::A8_UNORM, TF::R8Unorm, a8, 3, wide));
    // An A8 render target writes alpha only (its rgba8unorm stand-in keeps rgb at 0).
    D3D12GraphicsPsoDesc a8_rt = basic_desc();
    a8_rt.RTVFormats[0] = D::A8_UNORM;
    CHECK(translate_graphics_pso(a8_rt, PipelineKeyOptions{}).base.color_targets[0].write_mask ==
          D3D12ColorWriteEnable::Alpha);
    CHECK(translate_graphics_pso(basic_desc(), PipelineKeyOptions{}).base.color_targets[0].write_mask ==
          D3D12ColorWriteEnable::All);
}

void test_vertex_format_mapping() {
    std::set<uint32_t> covered;
    uint32_t supported = 0;
    for (uint32_t raw = 0; raw <= 140; ++raw) {
        const VertexFormatMapping m = map_vertex_format(static_cast<D>(raw));
        CHECK(m.byte_size == vertex_format_byte_size(m.format));
        if (m.supported()) {
            ++supported;
            CHECK(m.byte_size > 0);
            covered.insert(static_cast<uint32_t>(m.format));
        }
    }
    // Every DXGI IA format with a WebGPU equivalent maps to a distinct GPUVertexFormat, and every
    // GPUVertexFormat is reachable from DXGI.
    CHECK(supported == static_cast<uint32_t>(VF::Count) - 1u);
    CHECK(covered.size() == static_cast<size_t>(VF::Count) - 1u);

    CHECK(map_vertex_format(D::R32G32B32_FLOAT).format == VF::Float32x3);
    CHECK(map_vertex_format(D::R32G32B32_FLOAT).byte_size == 12);
    CHECK(map_vertex_format(D::R8G8B8A8_UNORM).format == VF::Unorm8x4);
    CHECK(map_vertex_format(D::B8G8R8A8_UNORM).format == VF::Unorm8x4Bgra);
    CHECK(map_vertex_format(D::R10G10B10A2_UNORM).format == VF::Unorm10_10_10_2);
    CHECK(map_vertex_format(D::R16G16_SNORM).format == VF::Snorm16x2);
    CHECK(map_vertex_format(D::R16_FLOAT).format == VF::Float16);
    CHECK(map_vertex_format(D::R8_UINT).format == VF::Uint8);
    CHECK(!map_vertex_format(D::R11G11B10_FLOAT).supported());
    CHECK(!map_vertex_format(D::R10G10B10A2_UINT).supported());
    CHECK(!map_vertex_format(D::R8G8B8A8_UNORM_SRGB).supported());
    CHECK(!map_vertex_format(D::B5G6R5_UNORM).supported());
    CHECK(!map_vertex_format(D::UNKNOWN).supported());
}

// ---------------------------------------------------------------------------
// 4. Input layout -> vertex buffer layouts
// ---------------------------------------------------------------------------

void test_input_layout() {
    using IC = D3D12InputClassification;
    constexpr uint32_t A = kD3D12AppendAlignedElement;

    // Append-aligned offsets: each element follows the previous one, aligned to min(4, size).
    const D3D12InputElementDesc packed[] = {
        {"POSITION", 0, D::R32G32B32_FLOAT, 0, A, IC::PerVertexData, 0},  // 0
        {"NORMAL", 0, D::R8G8B8A8_SNORM, 0, A, IC::PerVertexData, 0},     // 12
        {"TEXCOORD", 0, D::R16G16_FLOAT, 0, A, IC::PerVertexData, 0},     // 16
        {"COLOR", 0, D::R8_UNORM, 0, A, IC::PerVertexData, 0},            // 20
        {"TANGENT", 0, D::R16_FLOAT, 0, A, IC::PerVertexData, 0},         // align(21, 2) = 22
        {"BLENDWEIGHT", 0, D::R32_FLOAT, 0, 32, IC::PerVertexData, 0},    // explicit 32
        {"BLENDINDICES", 0, D::R8G8B8A8_UINT, 0, A, IC::PerVertexData, 0},// 36
    };
    D3D12GraphicsPsoDesc d = basic_desc();
    d.pInputElementDescs = packed;
    d.NumInputElements = 7;
    GraphicsKeyResult r = make_graphics_pipeline_key(d);
    CHECK(r.diagnostics == 0);
    CHECK(r.key.vertex_buffer_count == 1);
    CHECK(r.key.vertex_attribute_count == 7);
    CHECK(r.key.vertex_buffers[0].array_stride == 40);
    CHECK(r.key.vertex_buffers[0].step_mode == WgpuVertexStepMode::Vertex);
    CHECK(r.key.vertex_buffers[0].attribute_count == 7);
    const uint32_t expected_offsets[7] = {0, 12, 16, 20, 22, 32, 36};
    const VF expected_formats[7] = {VF::Float32x3, VF::Snorm8x4, VF::Float16x2, VF::Unorm8,
                                    VF::Float16,   VF::Float32,  VF::Uint8x4};
    for (uint32_t i = 0; i < 7; ++i) {
        CHECK(r.key.vertex_attributes[i].shader_location == i);
        CHECK(r.key.vertex_attributes[i].offset == expected_offsets[i]);
        CHECK(r.key.vertex_attributes[i].format == expected_formats[i]);
        CHECK(r.key.vertex_attributes[i].buffer_index == 0);
    }

    // Instancing: slot 3 per-instance matrix rows, slot 5 per-instance constant (step rate 0), slots 1/2/4
    // are holes. Attributes are sorted by (slot, location).
    const D3D12InputElementDesc instanced[] = {
        {"WORLD", 0, D::R32G32B32A32_FLOAT, 3, A, IC::PerInstanceData, 1},
        {"WORLD", 1, D::R32G32B32A32_FLOAT, 3, A, IC::PerInstanceData, 1},
        {"POSITION", 0, D::R32G32B32_FLOAT, 0, A, IC::PerVertexData, 0},
        {"TINT", 0, D::R8G8B8A8_UNORM, 5, 0, IC::PerInstanceData, 0},
    };
    d.pInputElementDescs = instanced;
    d.NumInputElements = 4;
    r = make_graphics_pipeline_key(d);
    CHECK(r.diagnostics == 0);
    CHECK(r.key.vertex_buffer_count == 6);
    CHECK(r.key.vertex_buffers[0].step_mode == WgpuVertexStepMode::Vertex);
    CHECK(r.key.vertex_buffers[0].array_stride == 12);
    CHECK(r.key.vertex_buffers[1].step_mode == WgpuVertexStepMode::Unused);
    CHECK(r.key.vertex_buffers[2].step_mode == WgpuVertexStepMode::Unused);
    CHECK(r.key.vertex_buffers[3].step_mode == WgpuVertexStepMode::Instance);
    CHECK(r.key.vertex_buffers[3].array_stride == 32);
    CHECK(r.key.vertex_buffers[3].attribute_count == 2);
    CHECK(r.key.vertex_buffers[4].step_mode == WgpuVertexStepMode::Unused);
    CHECK(r.key.vertex_buffers[5].step_mode == WgpuVertexStepMode::Instance);
    CHECK(r.key.vertex_buffers[5].array_stride == 0);  // step rate 0: every instance reads element 0
    CHECK(r.key.vertex_attributes[0].buffer_index == 0 && r.key.vertex_attributes[0].shader_location == 2);
    CHECK(r.key.vertex_attributes[1].buffer_index == 3 && r.key.vertex_attributes[1].shader_location == 0);
    CHECK(r.key.vertex_attributes[2].buffer_index == 3 && r.key.vertex_attributes[2].offset == 16);
    CHECK(r.key.vertex_attributes[3].buffer_index == 5);

    // Bound strides (IASetVertexBuffers) override the tight stride; step-rate-0 slots stay 0.
    PipelineVariantState v{};
    v.bound_vertex_slots = (1u << 0) | (1u << 3) | (1u << 5);
    v.vertex_strides[0] = 48;
    v.vertex_strides[3] = 64;
    v.vertex_strides[5] = 4;
    r = make_graphics_pipeline_key(d, v);
    CHECK(r.diagnostics == 0);
    CHECK(r.key.vertex_buffers[0].array_stride == 48);
    CHECK(r.key.vertex_buffers[3].array_stride == 64);
    CHECK(r.key.vertex_buffers[5].array_stride == 0);
    v.vertex_strides[0] = 14;  // not a multiple of 4
    CHECK(has(diag_of(d, v), PipelineDiag::VertexLayoutInvalid));
    v.vertex_strides[0] = 8;  // float3 at offset 0 overruns an 8-byte stride
    CHECK(has(diag_of(d, v), PipelineDiag::VertexLayoutInvalid));
    v.vertex_strides[0] = 4096;  // > maxVertexBufferArrayStride
    CHECK(has(diag_of(d, v), PipelineDiag::VertexLayoutInvalid));

    // Instance step rate > 1 has no WebGPU equivalent.
    const D3D12InputElementDesc stepped[] = {{"POSITION", 0, D::R32G32_FLOAT, 1, A, IC::PerInstanceData, 2}};
    d.pInputElementDescs = stepped;
    d.NumInputElements = 1;
    CHECK(has(diag_of(d), PipelineDiag::InstanceStepRateUnsupported));

    // Shader input signature: case-insensitive semantic matching, unused elements dropped, missing inputs
    // diagnosed.
    const D3D12InputElementDesc layout[] = {
        {"position", 0, D::R32G32B32_FLOAT, 0, A, IC::PerVertexData, 0},
        {"TEXCOORD", 0, D::R32G32_FLOAT, 0, A, IC::PerVertexData, 0},
        {"TEXCOORD", 1, D::R32G32_FLOAT, 0, A, IC::PerVertexData, 0},
    };
    const VertexInputSignatureElement sig[] = {{"TexCoord", 1, 0}, {"POSITION", 0, 3}};
    d.pInputElementDescs = layout;
    d.NumInputElements = 3;
    d.vs_input_signature = sig;
    d.vs_input_signature_count = 2;
    r = make_graphics_pipeline_key(d);
    CHECK(r.diagnostics == 0);
    CHECK(r.key.vertex_attribute_count == 2);
    CHECK(r.key.vertex_attributes[0].shader_location == 0 && r.key.vertex_attributes[0].offset == 20);
    CHECK(r.key.vertex_attributes[1].shader_location == 3 && r.key.vertex_attributes[1].offset == 0);
    CHECK(r.key.vertex_buffers[0].array_stride == 28);  // stride still covers the dropped element
    const VertexInputSignatureElement sig_missing[] = {{"POSITION", 0, 0}, {"NORMAL", 0, 1}};
    d.vs_input_signature = sig_missing;
    CHECK(has(diag_of(d), PipelineDiag::MissingVertexInput));
    d.vs_input_signature = nullptr;
    d.vs_input_signature_count = 0;

    // Slot out of range, unsupported format, and too many locations.
    const D3D12InputElementDesc bad[] = {
        {"A", 0, D::R32_FLOAT, 9, A, IC::PerVertexData, 0},
        {"B", 0, D::R11G11B10_FLOAT, 0, A, IC::PerVertexData, 0},
        {"C", 0, D::R32_FLOAT, 0, A, IC::PerVertexData, 0},
    };
    d.pInputElementDescs = bad;
    d.NumInputElements = 3;
    r = make_graphics_pipeline_key(d);
    CHECK(has(r.diagnostics, PipelineDiag::VertexSlotOutOfRange | PipelineDiag::VertexFormatUnsupported));
    CHECK(r.key.vertex_attribute_count == 1);
    CHECK(r.key.vertex_attributes[0].shader_location == 2);
    std::vector<D3D12InputElementDesc> many(18, D3D12InputElementDesc{"T", 0, D::R32_FLOAT, 0, A, IC::PerVertexData, 0});
    for (uint32_t i = 0; i < many.size(); ++i) {
        many[i].SemanticIndex = i;
    }
    d.pInputElementDescs = many.data();
    d.NumInputElements = static_cast<uint32_t>(many.size());
    r = make_graphics_pipeline_key(d);
    CHECK(has(r.diagnostics, PipelineDiag::TooManyVertexAttributes));
    CHECK(r.key.vertex_attribute_count == kMaxVertexAttributes);
    CHECK(r.key.vertex_buffers[0].array_stride == 72);
}

// ---------------------------------------------------------------------------
// 5. Key canonicalisation and hashing
// ---------------------------------------------------------------------------

void test_key_canonicalisation() {
    const D3D12GraphicsPsoDesc base = basic_desc();
    const GraphicsPipelineKey k0 = key_of(base);
    const uint64_t h0 = hash_key(k0);
    CHECK(k0.color_target_count == 1);
    CHECK(k0.color_targets[0].format == TF::Rgba8Unorm);
    CHECK(k0.color_targets[0].write_mask == 0xF);
    CHECK(k0.color_targets[0].blend_enabled == 0);
    CHECK(k0.depth_stencil.format == TF::Depth32Float);
    CHECK(k0.depth_stencil.depth_compare == WgpuCompareFunction::Less);
    CHECK(k0.depth_stencil.depth_write_enabled == 1);
    CHECK(k0.primitive.topology == WgpuPrimitiveTopology::TriangleList);
    CHECK(k0.primitive.cull_mode == WgpuCullMode::Back);
    CHECK(k0.primitive.front_face == WgpuFrontFace::Cw);
    CHECK(k0.multisample.count == 1 && k0.multisample.mask == 1);
    CHECK(k0.vs_shader == 0x1111 && k0.ps_shader == 0x2222 && k0.root_signature_id == 7);

    // Stable hash: pinned values guard against accidental algorithm changes (persistent caches).
    CHECK(stable_hash64("", 0) == stable_hash64("", 0));
    CHECK(stable_hash64("abc", 3) != stable_hash64("abd", 3));
    CHECK(stable_hash64("abcdefgh", 8) != stable_hash64("abcdefgh\0", 9));
    CHECK(stable_hash64("", 0) == 0x9ca066f1a4ab2eeaull);
    CHECK(stable_hash64("abc", 3) == 0xceb3551ae13032e9ull);

    // Descriptions that differ only in irrelevant fields produce identical keys.
    std::vector<D3D12GraphicsPsoDesc> equivalent;
    {
        D3D12GraphicsPsoDesc d = base;  // disabled blend with junk factors
        d.BlendState.RenderTarget[0].SrcBlend = D3D12Blend::SrcAlpha;
        d.BlendState.RenderTarget[0].DestBlend = D3D12Blend::InvSrcAlpha;
        d.BlendState.RenderTarget[0].BlendOp = D3D12BlendOp::Max;
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // enabled blend that is a pass-through
        d.BlendState.RenderTarget[0].BlendEnable = true;
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // RT[1..7] ignored without IndependentBlendEnable
        for (uint32_t i = 1; i < 8; ++i) {
            d.BlendState.RenderTarget[i].BlendEnable = true;
            d.BlendState.RenderTarget[i].SrcBlend = D3D12Blend::DestColor;
            d.BlendState.RenderTarget[i].RenderTargetWriteMask = 1;
        }
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // formats beyond NumRenderTargets are ignored
        d.RTVFormats[3] = D::R16G16B16A16_FLOAT;
        d.RTVFormats[7] = D::R32_UINT;
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // stencil disabled with junk ops/masks
        d.DepthStencilState.StencilReadMask = 0x0F;
        d.DepthStencilState.StencilWriteMask = 0x3C;
        d.DepthStencilState.FrontFace.StencilPassOp = D3D12StencilOp::Invert;
        d.DepthStencilState.BackFace.StencilFunc = D3D12ComparisonFunc::Equal;
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // stencil enabled on a format without stencil
        d.DepthStencilState.StencilEnable = true;
        d.DepthStencilState.FrontFace.StencilPassOp = D3D12StencilOp::Replace;
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // -0.0 bias equals +0.0; NaN treated as 0
        d.RasterizerState.SlopeScaledDepthBias = -0.0f;
        d.RasterizerState.DepthBiasClamp = std::numeric_limits<float>::quiet_NaN();
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // quality ignored at 1 sample; strip cut ignored for lists
        d.SampleDesc.Quality = 3;
        d.IBStripCutValue = D3D12IndexBufferStripCutValue::Value0xFFFF;
        d.SampleMask = 0xFFFFFFF1u;  // only bit 0 matters at 1 sample
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // logic op COPY == no blending
        d.BlendState.RenderTarget[0].LogicOpEnable = true;
        d.BlendState.RenderTarget[0].LogicOp = D3D12LogicOp::Copy;
        equivalent.push_back(d);
    }
    {
        D3D12GraphicsPsoDesc d = base;  // semantic names are case-insensitive
        static const D3D12InputElementDesc lower[] = {
            {"position", 0, D::R32G32B32_FLOAT, 0, 0, D3D12InputClassification::PerVertexData, 0}};
        d.pInputElementDescs = lower;
        equivalent.push_back(d);
    }
    for (size_t i = 0; i < equivalent.size(); ++i) {
        const GraphicsPipelineKey k = key_of(equivalent[i]);
        if (!(k == k0)) {
            std::fprintf(stderr, "equivalent desc #%zu produced a different key\n", i);
        }
        CHECK(k == k0);
        CHECK(hash_key(k) == h0);
    }

    // Depth disabled: compare/write/bias collapse regardless of junk.
    D3D12GraphicsPsoDesc nd = base;
    nd.DepthStencilState.DepthEnable = false;
    D3D12GraphicsPsoDesc nd2 = nd;
    nd2.DepthStencilState.DepthFunc = D3D12ComparisonFunc::Greater;
    nd2.DepthStencilState.DepthWriteMask = D3D12DepthWriteMask::Zero;
    nd2.RasterizerState.DepthBias = 100;
    CHECK(key_of(nd) == key_of(nd2));
    CHECK(key_of(nd).depth_stencil.depth_compare == WgpuCompareFunction::Always);

    // Blend canonicalisation: alpha colour factors == alpha factors, MIN/MAX ignore factors.
    D3D12GraphicsPsoDesc b1 = base;
    b1.BlendState.RenderTarget[0].BlendEnable = true;
    b1.BlendState.RenderTarget[0].SrcBlend = D3D12Blend::SrcAlpha;
    b1.BlendState.RenderTarget[0].DestBlend = D3D12Blend::InvSrcAlpha;
    b1.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12Blend::SrcColor;
    b1.BlendState.RenderTarget[0].DestBlendAlpha = D3D12Blend::InvDestColor;
    D3D12GraphicsPsoDesc b2 = b1;
    b2.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12Blend::SrcAlpha;
    b2.BlendState.RenderTarget[0].DestBlendAlpha = D3D12Blend::InvDestAlpha;
    CHECK(key_of(b1) == key_of(b2));
    const GraphicsPipelineKey kb = key_of(b1);
    CHECK(kb.color_targets[0].blend_enabled == 1);
    CHECK(kb.color_targets[0].color.src_factor == WgpuBlendFactor::SrcAlpha);
    CHECK(kb.color_targets[0].alpha.dst_factor == WgpuBlendFactor::OneMinusDstAlpha);
    D3D12GraphicsPsoDesc m1 = base;
    m1.BlendState.RenderTarget[0].BlendEnable = true;
    m1.BlendState.RenderTarget[0].BlendOp = D3D12BlendOp::Max;
    m1.BlendState.RenderTarget[0].SrcBlend = D3D12Blend::SrcColor;
    D3D12GraphicsPsoDesc m2 = m1;
    m2.BlendState.RenderTarget[0].SrcBlend = D3D12Blend::DestAlpha;
    m2.BlendState.RenderTarget[0].DestBlend = D3D12Blend::BlendFactor;
    CHECK(key_of(m1) == key_of(m2));
    CHECK(key_of(m1).color_targets[0].color.src_factor == WgpuBlendFactor::One);
    CHECK(key_of(m1).color_targets[0].color.dst_factor == WgpuBlendFactor::One);

    // Independent blend: RT[0] replicated to every bound target when disabled.
    D3D12GraphicsPsoDesc ib = b1;
    ib.NumRenderTargets = 3;
    ib.RTVFormats[1] = D::R8G8B8A8_UNORM;
    ib.RTVFormats[2] = D::R16G16B16A16_FLOAT;
    const GraphicsPipelineKey kib = key_of(ib);
    CHECK(kib.color_target_count == 3);
    CHECK(kib.color_targets[2].blend_enabled == 1);
    CHECK(kib.color_targets[2].color.dst_factor == WgpuBlendFactor::OneMinusSrcAlpha);
    ib.BlendState.IndependentBlendEnable = true;  // RT[1], RT[2] default: no blending
    CHECK(key_of(ib).color_targets[1].blend_enabled == 0);
    CHECK(key_of(ib).color_targets[0].blend_enabled == 1);

    // Every relevant change produces a different key and hash.
    std::vector<D3D12GraphicsPsoDesc> different;
    auto add = [&](auto mutate) {
        D3D12GraphicsPsoDesc d = base;
        mutate(d);
        different.push_back(d);
    };
    static const D3D12InputElementDesc offset4[] = {
        {"POSITION", 0, D::R32G32B32_FLOAT, 0, 4, D3D12InputClassification::PerVertexData, 0}};
    static const D3D12InputElementDesc half4[] = {
        {"POSITION", 0, D::R16G16B16A16_FLOAT, 0, kD3D12AppendAlignedElement, D3D12InputClassification::PerVertexData, 0}};
    static const D3D12InputElementDesc per_instance[] = {
        {"POSITION", 0, D::R32G32B32_FLOAT, 0, 0, D3D12InputClassification::PerInstanceData, 1}};
    static const D3D12InputElementDesc slot1[] = {
        {"POSITION", 0, D::R32G32B32_FLOAT, 1, 0, D3D12InputClassification::PerVertexData, 0}};
    add([](D3D12GraphicsPsoDesc& d) { d.vs = 0x1112; });
    add([](D3D12GraphicsPsoDesc& d) { d.ps = 0x2223; });
    add([](D3D12GraphicsPsoDesc& d) { d.root_signature_id = 8; });
    add([](D3D12GraphicsPsoDesc& d) {
        d.BlendState.RenderTarget[0].BlendEnable = true;
        d.BlendState.RenderTarget[0].SrcBlend = D3D12Blend::SrcAlpha;
    });
    add([](D3D12GraphicsPsoDesc& d) {
        d.BlendState.RenderTarget[0].BlendEnable = true;
        d.BlendState.RenderTarget[0].BlendOpAlpha = D3D12BlendOp::RevSubtract;
    });
    add([](D3D12GraphicsPsoDesc& d) { d.BlendState.RenderTarget[0].RenderTargetWriteMask = 7; });
    add([](D3D12GraphicsPsoDesc& d) { d.RasterizerState.CullMode = D3D12CullMode::None; });
    add([](D3D12GraphicsPsoDesc& d) { d.RasterizerState.FrontCounterClockwise = true; });
    add([](D3D12GraphicsPsoDesc& d) { d.RasterizerState.DepthBias = 1; });
    add([](D3D12GraphicsPsoDesc& d) { d.RasterizerState.SlopeScaledDepthBias = 1.5f; });
    add([](D3D12GraphicsPsoDesc& d) { d.RasterizerState.DepthBiasClamp = 0.25f; });
    add([](D3D12GraphicsPsoDesc& d) { d.DepthStencilState.DepthFunc = D3D12ComparisonFunc::GreaterEqual; });
    add([](D3D12GraphicsPsoDesc& d) { d.DepthStencilState.DepthWriteMask = D3D12DepthWriteMask::Zero; });
    add([](D3D12GraphicsPsoDesc& d) {
        d.DSVFormat = D::D24_UNORM_S8_UINT;
        d.DepthStencilState.StencilEnable = true;
        d.DepthStencilState.FrontFace.StencilPassOp = D3D12StencilOp::Replace;
    });
    add([](D3D12GraphicsPsoDesc& d) { d.DSVFormat = D::D24_UNORM_S8_UINT; });
    add([](D3D12GraphicsPsoDesc& d) { d.DSVFormat = D::UNKNOWN; });
    add([](D3D12GraphicsPsoDesc& d) { d.RTVFormats[0] = D::R8G8B8A8_UNORM_SRGB; });
    add([](D3D12GraphicsPsoDesc& d) {
        d.NumRenderTargets = 2;
        d.RTVFormats[1] = D::R8G8B8A8_UNORM;
    });
    add([](D3D12GraphicsPsoDesc& d) { d.SampleDesc.Count = 4; });
    add([](D3D12GraphicsPsoDesc& d) { d.SampleMask = 0; });
    add([](D3D12GraphicsPsoDesc& d) { d.PrimitiveTopologyType = D3D12PrimitiveTopologyType::Line; });
    add([](D3D12GraphicsPsoDesc& d) { d.pInputElementDescs = offset4; });
    add([](D3D12GraphicsPsoDesc& d) { d.pInputElementDescs = half4; });
    add([](D3D12GraphicsPsoDesc& d) { d.pInputElementDescs = per_instance; });
    add([](D3D12GraphicsPsoDesc& d) { d.pInputElementDescs = slot1; });
    add([](D3D12GraphicsPsoDesc& d) { d.NumInputElements = 0; });
    add([](D3D12GraphicsPsoDesc& d) { d.ps_output_mask = 0; });
    std::set<uint64_t> hashes{h0};
    for (size_t i = 0; i < different.size(); ++i) {
        const GraphicsPipelineKey k = key_of(different[i]);
        if (k == k0) {
            std::fprintf(stderr, "different desc #%zu produced the base key\n", i);
        }
        CHECK(!(k == k0));
        hashes.insert(hash_key(k));
    }
    CHECK(hashes.size() == different.size() + 1);

    // Compute keys.
    const ComputePipelineKey c0 = make_compute_pipeline_key({3, 0xABC});
    CHECK(c0 == make_compute_pipeline_key({3, 0xABC}));
    CHECK(!(c0 == make_compute_pipeline_key({4, 0xABC})));
    CHECK(!(c0 == make_compute_pipeline_key({3, 0xABD})));
    CHECK(hash_key(c0) == hash_key(make_compute_pipeline_key({3, 0xABC})));
}

// ---------------------------------------------------------------------------
// 6. Pipeline variants (dynamic state baked into WebGPU pipelines)
// ---------------------------------------------------------------------------

void test_variants() {
    const D3D12GraphicsPsoDesc d = basic_desc();
    PipelineVariantState v{};
    const GraphicsPipelineKey list = key_of(d, v);

    v.topology = D3DPrimitiveTopology::TriangleList;
    CHECK(key_of(d, v) == list);
    v.index_format = WgpuIndexFormat::Uint32;  // irrelevant for list topologies
    CHECK(key_of(d, v) == list);

    v.topology = D3DPrimitiveTopology::TriangleStrip;
    GraphicsKeyResult strip = make_graphics_pipeline_key(d, v);
    CHECK(strip.key.primitive.topology == WgpuPrimitiveTopology::TriangleStrip);
    CHECK(strip.key.primitive.strip_index_format == WgpuIndexFormat::Uint32);
    CHECK(has(strip.diagnostics, PipelineDiag::StripCutValueMismatch));  // D3D12 cut disabled
    D3D12GraphicsPsoDesc cut32 = d;
    cut32.IBStripCutValue = D3D12IndexBufferStripCutValue::Value0xFFFFFFFF;
    CHECK(!has(diag_of(cut32, v), PipelineDiag::StripCutValueMismatch));
    v.index_format = WgpuIndexFormat::Uint16;
    CHECK(has(diag_of(cut32, v), PipelineDiag::StripCutValueMismatch));
    CHECK(!(key_of(d, v) == strip.key));
    v.index_format = WgpuIndexFormat::Undefined;  // non-indexed: format from the PSO cut value
    CHECK(key_of(cut32, v).primitive.strip_index_format == WgpuIndexFormat::Uint32);
    CHECK(key_of(d, v).primitive.strip_index_format == WgpuIndexFormat::Undefined);

    v.topology = D3DPrimitiveTopology::LineList;  // PSO says triangles
    CHECK(has(diag_of(d, v), PipelineDiag::TopologyTypeMismatch));
    CHECK(key_of(d, v).primitive.topology == WgpuPrimitiveTopology::LineList);

    // Bound render targets are authoritative.
    PipelineVariantState b{};
    b.flags = VariantFlags::BoundTargets;
    b.num_render_targets = 1;
    b.rtv_formats[0] = D::R8G8B8A8_UNORM;
    b.dsv_format = D::D32_FLOAT;
    b.sample_count = 1;
    CHECK(key_of(d, b) == list);
    CHECK(diag_of(d, b) == 0);
    b.rtv_formats[0] = D::R8G8B8A8_UNORM_SRGB;
    GraphicsKeyResult srgb = make_graphics_pipeline_key(d, b);
    CHECK(srgb.key.color_targets[0].format == TF::Rgba8UnormSrgb);
    CHECK(has(srgb.diagnostics, PipelineDiag::RenderTargetFormatMismatch));
    b.rtv_formats[0] = D::R8G8B8A8_UNORM;
    b.num_render_targets = 2;  // RTV bound where the PSO declares none: masked
    b.rtv_formats[1] = D::R16G16B16A16_FLOAT;
    GraphicsKeyResult extra = make_graphics_pipeline_key(d, b);
    CHECK(extra.key.color_target_count == 2);
    CHECK(extra.key.color_targets[1].format == TF::Rgba16Float);
    CHECK(extra.key.color_targets[1].write_mask == 0);
    CHECK(!has(extra.diagnostics, PipelineDiag::RenderTargetFormatMismatch));
    b.num_render_targets = 1;
    b.dsv_format = D::UNKNOWN;  // no DSV bound: no depth state at all
    CHECK(key_of(d, b).depth_stencil.format == TF::Undefined);
    CHECK(key_of(d, b).depth_stencil.depth_write_enabled == 0);
    b.dsv_format = D::D24_UNORM_S8_UINT;
    CHECK(has(diag_of(d, b), PipelineDiag::DepthFormatMismatch));
    b.dsv_format = D::D32_FLOAT;
    b.sample_count = 8;
    GraphicsKeyResult ms = make_graphics_pipeline_key(d, b);
    CHECK(ms.key.multisample.count == 4);
    CHECK(ms.key.multisample.mask == 0xF);
    CHECK(has(ms.diagnostics, PipelineDiag::SampleCountRounded | PipelineDiag::SampleCountMismatch));

    // GraphicsPso resolves variants through the shared cache.
    PipelineCache cache;
    GraphicsPso pso(d);
    PipelineVariantState s{};
    s.flags = VariantFlags::BoundTargets;
    s.num_render_targets = 1;
    s.rtv_formats[0] = D::R8G8B8A8_UNORM;
    s.dsv_format = D::D32_FLOAT;
    s.sample_count = 1;
    s.topology = D3DPrimitiveTopology::TriangleList;
    const PipelineResolution r1 = pso.resolve(cache, s);
    CHECK(r1.usable() && r1.created && r1.key != nullptr);
    CHECK(*r1.key == list);
    const PipelineResolution r2 = pso.resolve(cache, s);
    CHECK(r2.pipeline_id == r1.pipeline_id && !r2.created);
    CHECK(cache.stats().lookups == 1);  // second resolve never touched the shared cache
    s.index_format = WgpuIndexFormat::Uint16;  // canonicalised away for list topology
    CHECK(pso.resolve(cache, s).pipeline_id == r1.pipeline_id);
    CHECK(pso.variant_count() == 1);
    s.topology = D3DPrimitiveTopology::TriangleStrip;
    const PipelineResolution r3 = pso.resolve(cache, s);
    CHECK(r3.created && r3.pipeline_id != r1.pipeline_id);
    CHECK(r3.key->primitive.strip_index_format == WgpuIndexFormat::Uint16);
    CHECK(pso.variant_count() == 2);
    CHECK(cache.stats().variants == 1);
    s.topology = D3DPrimitiveTopology::TriangleList;
    CHECK(pso.resolve(cache, s).pipeline_id == r1.pipeline_id);
    CHECK(cache.stats().misses == 2);

    // A second PSO with the same description shares pipelines.
    GraphicsPso twin(d);
    const PipelineResolution t1 = twin.resolve(cache, s);
    CHECK(t1.pipeline_id == r1.pipeline_id && !t1.created);
    CHECK(cache.ref_count(list) == 2);
}

// ---------------------------------------------------------------------------
// 7. Unsupported-feature diagnostics and fallbacks
// ---------------------------------------------------------------------------

void test_diagnostics() {
    const D3D12GraphicsPsoDesc base = basic_desc();
    CHECK(diag_of(base) == 0);

    auto with = [&](auto mutate) {
        D3D12GraphicsPsoDesc d = base;
        mutate(d);
        return make_graphics_pipeline_key(d);
    };

    // Shader stages.
    CHECK(is_fatal(with([](auto& d) { d.gs = 1; }).diagnostics));
    CHECK(has(with([](auto& d) { d.hs = 1; d.ds = 2; }).diagnostics, PipelineDiag::TessellationUnsupported));
    CHECK(is_fatal(with([](auto& d) { d.PrimitiveTopologyType = D3D12PrimitiveTopologyType::Patch; }).diagnostics));
    GraphicsKeyResult mesh = with([](auto& d) { d.vs = 0; d.ms = 5; });
    CHECK(has(mesh.diagnostics, PipelineDiag::MeshShaderUnsupported));
    CHECK(!has(mesh.diagnostics, PipelineDiag::MissingVertexShader));
    CHECK(is_fatal(with([](auto& d) { d.vs = 0; }).diagnostics));
    GraphicsKeyResult so = with([](auto& d) { d.stream_output_enabled = true; });
    CHECK(so.diagnostics == PipelineDiag::StreamOutputUnsupported && !is_fatal(so.diagnostics));

    // Wireframe falls back to solid.
    GraphicsKeyResult wire = with([](auto& d) { d.RasterizerState.FillMode = D3D12FillMode::Wireframe; });
    CHECK(wire.diagnostics == PipelineDiag::WireframeFillUnsupported);
    CHECK(wire.key == key_of(base));

    // Logic ops.
    GraphicsKeyResult noop = with([](auto& d) {
        d.BlendState.RenderTarget[0].LogicOpEnable = true;
        d.BlendState.RenderTarget[0].LogicOp = D3D12LogicOp::Noop;
    });
    CHECK(noop.diagnostics == 0 && noop.key.color_targets[0].write_mask == 0);
    GraphicsKeyResult xor_op = with([](auto& d) {
        d.BlendState.RenderTarget[0].LogicOpEnable = true;
        d.BlendState.RenderTarget[0].LogicOp = D3D12LogicOp::Xor;
    });
    CHECK(xor_op.diagnostics == PipelineDiag::LogicOpUnsupported);
    CHECK(xor_op.key == key_of(base));

    // Dual-source blending.
    auto dual = [](D3D12GraphicsPsoDesc& d) {
        d.BlendState.RenderTarget[0].BlendEnable = true;
        d.BlendState.RenderTarget[0].SrcBlend = D3D12Blend::One;
        d.BlendState.RenderTarget[0].DestBlend = D3D12Blend::InvSrc1Color;
    };
    GraphicsKeyResult no_dual = with(dual);
    CHECK(no_dual.diagnostics == PipelineDiag::DualSourceBlendUnavailable);
    CHECK(no_dual.key.color_targets[0].color.dst_factor == WgpuBlendFactor::OneMinusSrc);
    D3D12GraphicsPsoDesc dd = base;
    dual(dd);
    PipelineKeyOptions dual_opts{};
    dual_opts.features = WgpuFeature::DualSourceBlending;
    GraphicsKeyResult yes_dual = make_graphics_pipeline_key(dd, {}, dual_opts);
    CHECK(yes_dual.diagnostics == 0);
    CHECK(yes_dual.key.color_targets[0].color.dst_factor == WgpuBlendFactor::OneMinusSrc1);
    dd.NumRenderTargets = 2;
    dd.RTVFormats[1] = D::R8G8B8A8_UNORM;
    CHECK(is_fatal(make_graphics_pipeline_key(dd, {}, dual_opts).diagnostics));

    // Blending on formats WebGPU cannot blend.
    auto alpha_blend = [](D3D12GraphicsPsoDesc& d) {
        d.BlendState.RenderTarget[0].BlendEnable = true;
        d.BlendState.RenderTarget[0].SrcBlend = D3D12Blend::SrcAlpha;
        d.BlendState.RenderTarget[0].DestBlend = D3D12Blend::InvSrcAlpha;
    };
    GraphicsKeyResult uint_blend = with([&](auto& d) {
        alpha_blend(d);
        d.RTVFormats[0] = D::R32_UINT;
    });
    CHECK(uint_blend.diagnostics == PipelineDiag::BlendOnNonBlendableFormat);
    CHECK(uint_blend.key.color_targets[0].blend_enabled == 0);
    D3D12GraphicsPsoDesc fb = base;
    alpha_blend(fb);
    fb.RTVFormats[0] = D::R32_FLOAT;
    CHECK(diag_of(fb) == PipelineDiag::BlendOnNonBlendableFormat);
    PipelineKeyOptions f32{};
    f32.features = WgpuFeature::Float32Blendable;
    CHECK(make_graphics_pipeline_key(fb, {}, f32).diagnostics == 0);
    CHECK(make_graphics_pipeline_key(fb, {}, f32).key.color_targets[0].blend_enabled == 1);

    // ALPHA_FACTOR approximated only in the colour component.
    CHECK(with([](auto& d) {
              d.BlendState.RenderTarget[0].BlendEnable = true;
              d.BlendState.RenderTarget[0].SrcBlend = D3D12Blend::AlphaFactor;
          }).diagnostics == PipelineDiag::AlphaFactorApproximated);
    CHECK(with([](auto& d) {
              d.BlendState.RenderTarget[0].BlendEnable = true;
              d.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12Blend::AlphaFactor;
          }).diagnostics == 0);

    // Alpha-to-coverage needs 4 samples and a blendable RT0 with alpha.
    GraphicsKeyResult a2c1 = with([](auto& d) { d.BlendState.AlphaToCoverageEnable = true; });
    CHECK(a2c1.diagnostics == PipelineDiag::AlphaToCoverageDropped);
    CHECK(a2c1.key == key_of(base));
    GraphicsKeyResult a2c4 = with([](auto& d) {
        d.BlendState.AlphaToCoverageEnable = true;
        d.SampleDesc.Count = 4;
    });
    CHECK(a2c4.diagnostics == 0 && a2c4.key.multisample.alpha_to_coverage_enabled == 1);
    CHECK(has(with([](auto& d) {
                  d.BlendState.AlphaToCoverageEnable = true;
                  d.SampleDesc.Count = 4;
                  d.RTVFormats[0] = D::R16G16_FLOAT;
              }).diagnostics,
              PipelineDiag::AlphaToCoverageDropped));

    // Rasterizer extras.
    CHECK(with([](auto& d) { d.RasterizerState.ConservativeRaster = D3D12ConservativeRasterizationMode::On; })
              .diagnostics == PipelineDiag::ConservativeRasterUnsupported);
    CHECK(with([](auto& d) { d.DepthStencilState.DepthBoundsTestEnable = true; }).diagnostics ==
          PipelineDiag::DepthBoundsUnsupported);
    CHECK(with([](auto& d) { d.RasterizerState.ForcedSampleCount = 4; }).diagnostics ==
          PipelineDiag::ForcedSampleCountUnsupported);
    GraphicsKeyResult clip = with([](auto& d) { d.RasterizerState.DepthClipEnable = false; });
    CHECK(clip.diagnostics == PipelineDiag::DepthClipControlUnavailable && clip.key.primitive.unclipped_depth == 0);
    D3D12GraphicsPsoDesc dc = base;
    dc.RasterizerState.DepthClipEnable = false;
    PipelineKeyOptions dcc{};
    dcc.features = WgpuFeature::DepthClipControl;
    CHECK(make_graphics_pipeline_key(dc, {}, dcc).key.primitive.unclipped_depth == 1);
    CHECK(make_graphics_pipeline_key(dc, {}, dcc).diagnostics == 0);
    GraphicsKeyResult lines = with([](auto& d) {
        d.PrimitiveTopologyType = D3D12PrimitiveTopologyType::Line;
        d.RasterizerState.DepthBias = 4;
        d.RasterizerState.AntialiasedLineEnable = true;
    });
    CHECK(has(lines.diagnostics, PipelineDiag::DepthBiasDroppedForNonTriangle | PipelineDiag::LineRasterModeIgnored));
    CHECK(lines.key.depth_stencil.depth_bias == 0);
    CHECK(with([](auto& d) { d.RasterizerState.DepthBias = 4; }).key.depth_stencil.depth_bias == 4);

    // Topologies without WebGPU equivalents.
    PipelineVariantState v{};
    v.topology = D3DPrimitiveTopology::TriangleFan;
    CHECK(diag_of(base, v) == PipelineDiag::TriangleFanNeedsConversion);
    v.topology = D3DPrimitiveTopology::TriangleListAdj;
    CHECK(diag_of(base, v) == PipelineDiag::AdjacencyTopologyUnsupported);
    v.topology = static_cast<D3DPrimitiveTopology>(36);  // 4-control-point patch list
    CHECK(is_fatal(diag_of(base, v)));

    // Target formats.
    CHECK(is_fatal(with([](auto& d) { d.RTVFormats[0] = D::R32G32B32_FLOAT; }).diagnostics));
    GraphicsKeyResult r11 = with([](auto& d) { d.RTVFormats[0] = D::R11G11B10_FLOAT; });
    CHECK(r11.diagnostics == PipelineDiag::TargetFormatFallback);
    CHECK(r11.key.color_targets[0].format == TF::Rgba16Float);
    GraphicsKeyResult r9 = with([](auto& d) { d.RTVFormats[0] = D::R9G9B9E5_SHAREDEXP; });
    CHECK(r9.diagnostics == PipelineDiag::TargetFormatFallback);
    CHECK(r9.key.color_targets[0].format == TF::Rgba16Float);
    GraphicsKeyResult d32s8 = with([](auto& d) { d.DSVFormat = D::D32_FLOAT_S8X24_UINT; });
    CHECK(d32s8.diagnostics == PipelineDiag::TargetFormatFallback);
    CHECK(d32s8.key.depth_stencil.format == TF::Depth24PlusStencil8);
    GraphicsKeyResult b5g5r5a1 = with([](auto& d) { d.RTVFormats[0] = D::B5G5R5A1_UNORM; });
    CHECK(b5g5r5a1.diagnostics == PipelineDiag::TargetFormatFallback);
    CHECK(b5g5r5a1.key.color_targets[0].format == TF::Rgba8Unorm);
    GraphicsKeyResult b4g4r4a4 = with([](auto& d) { d.RTVFormats[0] = D::B4G4R4A4_UNORM; });
    CHECK(b4g4r4a4.diagnostics == PipelineDiag::TargetFormatFallback);
    CHECK(b4g4r4a4.key.color_targets[0].format == TF::Rgba8Unorm);
    GraphicsKeyResult d16s8 = with([](auto& d) { d.DSVFormat = D::D16_UNORM_S8_UINT; });
    CHECK(d16s8.diagnostics == PipelineDiag::TargetFormatFallback);
    CHECK(d16s8.key.depth_stencil.format == TF::Depth24PlusStencil8);
    CHECK(is_fatal(with([](auto& d) {
                       d.RTVFormats[0] = D::R32G32B32A32_UINT;
                       d.SampleDesc.Count = 4;
                   }).diagnostics));  // 32-bit-per-channel formats cannot be multisampled
    auto wide = [](D3D12GraphicsPsoDesc& d) {
        d.NumRenderTargets = 8;
        for (auto& f : d.RTVFormats) {
            f = D::R16G16B16A16_FLOAT;
        }
    };
    GraphicsKeyResult wide_r = with(wide);
    CHECK(wide_r.diagnostics == PipelineDiag::ColorAttachmentBytesExceeded);
    D3D12GraphicsPsoDesc wd = base;
    wide(wd);
    PipelineKeyOptions big{};
    big.max_color_attachment_bytes_per_sample = 64;
    CHECK(make_graphics_pipeline_key(wd, {}, big).diagnostics == 0);

    // Null pixel shader.
    GraphicsKeyResult nps = with([](auto& d) { d.ps = 0; });
    CHECK(nps.diagnostics == PipelineDiag::NullPixelShaderWithColorTargets);
    CHECK(nps.key.color_targets[0].write_mask == 0);
    CHECK(with([](auto& d) {
              d.ps = 0;
              d.NumRenderTargets = 0;
              d.RTVFormats[0] = D::UNKNOWN;
          }).diagnostics == 0);  // depth-only pass
    CHECK(is_fatal(with([](auto& d) {
                       d.NumRenderTargets = 0;
                       d.DSVFormat = D::UNKNOWN;
                   }).diagnostics));
    // PS that does not write SV_Target1.
    GraphicsKeyResult partial_ps = with([](auto& d) {
        d.NumRenderTargets = 2;
        d.RTVFormats[1] = D::R8G8B8A8_UNORM;
        d.ps_output_mask = 1;
    });
    CHECK(partial_ps.key.color_targets[0].write_mask == 0xF && partial_ps.key.color_targets[1].write_mask == 0);

    // Stencil with D24S8 is kept; sample quality ignored; invalid enums diagnosed.
    GraphicsKeyResult st = with([](auto& d) {
        d.DSVFormat = D::D24_UNORM_S8_UINT;
        d.DepthStencilState.StencilEnable = true;
        d.DepthStencilState.StencilReadMask = 0x0F;
        d.DepthStencilState.FrontFace.StencilFunc = D3D12ComparisonFunc::Equal;
        d.DepthStencilState.BackFace.StencilFailOp = D3D12StencilOp::Zero;
    });
    CHECK(st.diagnostics == 0);
    CHECK(st.key.depth_stencil.stencil_front.compare == WgpuCompareFunction::Equal);
    CHECK(st.key.depth_stencil.stencil_back.fail_op == WgpuStencilOperation::Zero);
    CHECK(st.key.depth_stencil.stencil_read_mask == 0x0F && st.key.depth_stencil.stencil_write_mask == 0xFF);
    CHECK(with([](auto& d) {
              d.SampleDesc.Count = 4;
              d.SampleDesc.Quality = 0xFFFFFFFFu;
          }).diagnostics == PipelineDiag::SampleQualityIgnored);
    CHECK(has(with([](auto& d) { d.DepthStencilState.DepthFunc = static_cast<D3D12ComparisonFunc>(42); }).diagnostics,
              PipelineDiag::InvalidEnumValue));
    CHECK(has(with([](auto& d) {
                  d.BlendState.RenderTarget[0].BlendEnable = true;
                  d.BlendState.RenderTarget[0].SrcBlend = static_cast<D3D12Blend>(12);
              }).diagnostics,
              PipelineDiag::InvalidEnumValue));
}

// ---------------------------------------------------------------------------
// 8. Pipeline cache: hits/misses/stats, eviction, collisions, concurrency
// ---------------------------------------------------------------------------

uint64_t constant_hash(const void*, size_t) {
    return 42;
}

std::vector<GraphicsPipelineKey> distinct_keys(uint32_t count) {
    std::vector<GraphicsPipelineKey> keys;
    for (uint32_t i = 0; i < count; ++i) {
        D3D12GraphicsPsoDesc d = basic_desc();
        d.vs = 0x1000 + i;
        keys.push_back(key_of(d));
    }
    return keys;
}

void test_cache_basics() {
    const auto keys = distinct_keys(4);
    PipelineCache cache;
    const CacheLookup a = cache.acquire(keys[0]);
    CHECK(a.created && a.pipeline_id == 1);
    const CacheLookup b = cache.acquire(keys[1]);
    CHECK(b.created && b.pipeline_id == 2);
    const CacheLookup a2 = cache.acquire(keys[0]);
    CHECK(!a2.created && a2.pipeline_id == 1);
    CHECK(cache.find(keys[0]) == 1u);
    CHECK(!cache.find(keys[2]).has_value());
    CHECK(cache.ref_count(keys[0]) == 2);
    const ComputePipelineKey ck = make_compute_pipeline_key({1, 0xC5});
    const CacheLookup c = cache.acquire(ck);
    CHECK(c.created && c.pipeline_id == 3);
    CHECK(cache.acquire(ck).pipeline_id == 3);
    CHECK(cache.find(ck) == 3u);

    PipelineCacheStats s = cache.stats();
    CHECK(s.lookups == 5 && s.hits == 2 && s.misses == 3);
    CHECK(s.live_graphics == 2 && s.live_compute == 1);
    CHECK(s.evictions == 0 && s.hash_collisions == 0);

    // Retain policy keeps unreferenced pipelines.
    CHECK(!cache.release(keys[1]).has_value());
    CHECK(cache.find(keys[1]) == 2u);
    CHECK(cache.ref_count(keys[1]) == 0);
    CHECK(!cache.release(keys[3]).has_value());  // unknown key

    // Eviction policy + custom id allocator.
    std::atomic<uint32_t> next{100};
    PipelineCacheOptions opts{};
    opts.eviction = EvictionPolicy::EvictUnreferenced;
    opts.id_allocator = [&next] { return next.fetch_add(10); };
    PipelineCache evicting(opts);
    CHECK(evicting.acquire(keys[0]).pipeline_id == 100);
    CHECK(evicting.acquire(keys[0]).pipeline_id == 100);
    CHECK(evicting.acquire(keys[1]).pipeline_id == 110);
    CHECK(!evicting.release(keys[0]).has_value());
    CHECK(evicting.release(keys[0]) == 100u);
    CHECK(!evicting.find(keys[0]).has_value());
    CHECK(evicting.stats().evictions == 1 && evicting.stats().live_graphics == 1);
    const CacheLookup again = evicting.acquire(keys[0]);
    CHECK(again.created && again.pipeline_id == 120);
    CHECK(evicting.release(ck) == std::nullopt);

    // GraphicsPso::release drops every variant reference.
    GraphicsPso pso(basic_desc());
    PipelineVariantState v{};
    const uint32_t list_id = pso.resolve(evicting, v).pipeline_id;
    v.topology = D3DPrimitiveTopology::LineList;
    const uint32_t line_id = pso.resolve(evicting, v).pipeline_id;
    v.topology = D3DPrimitiveTopology::PointList;
    v.topology = D3DPrimitiveTopology::Undefined;
    std::vector<uint32_t> evicted = pso.release(evicting);
    std::sort(evicted.begin(), evicted.end());
    CHECK(evicted.size() == 2);
    CHECK(std::find(evicted.begin(), evicted.end(), list_id) != evicted.end());
    CHECK(std::find(evicted.begin(), evicted.end(), line_id) != evicted.end());
    CHECK(pso.variant_count() == 0);

    // Fatal keys never reach the cache.
    D3D12GraphicsPsoDesc gs = basic_desc();
    gs.gs = 9;
    GraphicsPso fatal(gs);
    const uint64_t before = evicting.stats().lookups;
    const PipelineResolution fr = fatal.resolve(evicting, {});
    CHECK(fr.pipeline_id == 0 && !fr.usable() && !fr.created);
    CHECK(evicting.stats().lookups == before);
    CHECK(fatal.release(evicting).empty());
}

void test_cache_collisions() {
    PipelineCacheOptions opts{};
    opts.hash_override = &constant_hash;
    PipelineCache cache(opts);
    const auto keys = distinct_keys(5);
    std::set<uint32_t> ids;
    for (const auto& k : keys) {
        const CacheLookup l = cache.acquire(k);
        CHECK(l.created);
        ids.insert(l.pipeline_id);
    }
    CHECK(ids.size() == keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        CHECK(cache.find(keys[i]).has_value());
        CHECK(!cache.acquire(keys[i]).created);
    }
    CHECK(cache.stats().hash_collisions == keys.size() - 1);
    CHECK(cache.stats().misses == keys.size());
}

void test_cache_concurrency() {
    const auto keys = distinct_keys(64);
    PipelineCache cache;
    constexpr int kThreads = 8;
    constexpr int kRounds = 40;
    std::vector<std::vector<uint32_t>> seen(kThreads, std::vector<uint32_t>(keys.size(), 0));
    std::atomic<int> created{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int round = 0; round < kRounds; ++round) {
                for (size_t j = 0; j < keys.size(); ++j) {
                    const size_t i = (j * 7 + static_cast<size_t>(t) * 13 + static_cast<size_t>(round)) % keys.size();
                    const CacheLookup l = cache.acquire(keys[i]);
                    if (l.created) {
                        created.fetch_add(1);
                    }
                    if (seen[t][i] == 0) {
                        seen[t][i] = l.pipeline_id;
                    } else if (seen[t][i] != l.pipeline_id) {
                        seen[t][i] = 0xFFFFFFFFu;  // inconsistent id
                    }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    CHECK(created.load() == static_cast<int>(keys.size()));
    std::set<uint32_t> ids;
    for (size_t i = 0; i < keys.size(); ++i) {
        for (int t = 1; t < kThreads; ++t) {
            CHECK(seen[t][i] == seen[0][i]);
        }
        CHECK(seen[0][i] != 0 && seen[0][i] != 0xFFFFFFFFu);
        ids.insert(seen[0][i]);
    }
    CHECK(ids.size() == keys.size());
    const PipelineCacheStats s = cache.stats();
    const uint64_t total = static_cast<uint64_t>(kThreads) * kRounds * keys.size();
    CHECK(s.lookups == total && s.misses == keys.size() && s.hits == total - keys.size());
    CHECK(cache.ref_count(keys[0]) == static_cast<uint32_t>(kThreads * kRounds));

    // Concurrent resolves on one shared PSO object from several recording threads.
    PipelineCache shared;
    GraphicsPso pso(basic_desc());
    std::vector<std::thread> recorders;
    std::array<std::array<uint32_t, 3>, kThreads> ids_by_thread{};
    for (int t = 0; t < kThreads; ++t) {
        recorders.emplace_back([&, t] {
            const D3DPrimitiveTopology topologies[3] = {D3DPrimitiveTopology::TriangleList,
                                                        D3DPrimitiveTopology::TriangleStrip,
                                                        D3DPrimitiveTopology::LineList};
            for (int round = 0; round < 200; ++round) {
                const int which = (round + t) % 3;
                PipelineVariantState v{};
                v.topology = topologies[which];
                ids_by_thread[t][static_cast<size_t>(which)] = pso.resolve(shared, v).pipeline_id;
            }
        });
    }
    for (auto& th : recorders) {
        th.join();
    }
    for (int t = 1; t < kThreads; ++t) {
        CHECK(ids_by_thread[t] == ids_by_thread[0]);
    }
    CHECK(pso.variant_count() == 3);
    CHECK(shared.stats().misses == 3);
}

// ---------------------------------------------------------------------------
// 9. Output-merger tracker: viewport/scissor, dirty filtering, clears, pass inference
// ---------------------------------------------------------------------------

std::string trace(const CommandListStateTracker& t) {
    std::string s;
    for (const StateEvent& e : t.events()) {
        switch (e.type) {
            case StateEventType::BeginRenderPass: s += 'B'; break;
            case StateEventType::EndRenderPass: s += 'E'; break;
            case StateEventType::SetPipeline: s += 'P'; break;
            case StateEventType::SetViewport: s += 'V'; break;
            case StateEventType::SetScissorRect: s += 'S'; break;
            case StateEventType::SetBlendConstant: s += 'K'; break;
            case StateEventType::SetStencilReference: s += 'R'; break;
            case StateEventType::ClearFallback: s += 'C'; break;
        }
    }
    return s;
}

RenderTargetViewInfo rtv(uint32_t id, uint32_t w = 1280, uint32_t h = 720, D fmt = D::R8G8B8A8_UNORM) {
    return RenderTargetViewInfo{id, fmt, w, h, 1};
}

DepthStencilViewInfo dsv(uint32_t id, uint32_t w = 1280, uint32_t h = 720, D fmt = D::D24_UNORM_S8_UINT,
                         uint32_t flags = 0) {
    return DepthStencilViewInfo{id, fmt, w, h, 1, flags};
}

void set_full_viewport(CommandListStateTracker& t, float w = 1280.0f, float h = 720.0f) {
    const D3D12Viewport vp{0.0f, 0.0f, w, h, 0.0f, 1.0f};
    const D3D12Rect sc{0, 0, static_cast<int32_t>(w), static_cast<int32_t>(h)};
    t.RSSetViewports(1, &vp);
    t.RSSetScissorRects(1, &sc);
}

void test_viewport_scissor_conversion() {
    uint64_t diag = 0;
    WgpuViewport v = CommandListStateTracker::convert_viewport({10.0f, 20.0f, 640.0f, 360.0f, 0.1f, 0.9f}, 8192, &diag);
    CHECK(diag == 0);
    CHECK(v == (WgpuViewport{10.0f, 20.0f, 640.0f, 360.0f, 0.1f, 0.9f}));
    // Larger than the render target is legal in WebGPU (guard band); only the device limits clamp.
    v = CommandListStateTracker::convert_viewport({-100.0f, -50.0f, 4000.0f, 3000.0f, 0.0f, 1.0f}, 8192, &diag);
    CHECK(diag == 0 && v.x == -100.0f && v.width == 4000.0f);
    v = CommandListStateTracker::convert_viewport({-20000.0f, 0.0f, 9000.0f, 100.0f, 0.0f, 1.0f}, 8192, &diag);
    CHECK(has(diag, StateDiag::ViewportClamped));
    CHECK(v.x == -16384.0f && v.width == 8192.0f);
    diag = 0;
    v = CommandListStateTracker::convert_viewport({16000.0f, 0.0f, 1000.0f, 100.0f, 0.0f, 1.0f}, 8192, &diag);
    CHECK(has(diag, StateDiag::ViewportClamped));
    CHECK(v.x + v.width <= 16383.0f);
    diag = 0;
    v = CommandListStateTracker::convert_viewport({0.0f, 0.0f, 10.0f, 10.0f, 1.0f, 0.0f}, 8192, &diag);
    CHECK(diag == StateDiag::ViewportDepthInverted);
    CHECK(v.min_depth == 0.0f && v.max_depth == 1.0f);
    diag = 0;
    v = CommandListStateTracker::convert_viewport(
        {std::numeric_limits<float>::quiet_NaN(), 0.0f, -5.0f, 10.0f, -1.0f, 2.0f}, 8192, &diag);
    CHECK(has(diag, StateDiag::ViewportClamped));
    CHECK(v.x == 0.0f && v.width == 0.0f && v.min_depth == 0.0f && v.max_depth == 1.0f);

    CHECK(CommandListStateTracker::convert_scissor({0, 0, 1280, 720}, 1280, 720) == (WgpuScissorRect{0, 0, 1280, 720}));
    CHECK(CommandListStateTracker::convert_scissor({-10, -10, 5000, 5000}, 1280, 720) ==
          (WgpuScissorRect{0, 0, 1280, 720}));
    CHECK(CommandListStateTracker::convert_scissor({100, 50, 300, 250}, 1280, 720) ==
          (WgpuScissorRect{100, 50, 200, 200}));
    CHECK(CommandListStateTracker::convert_scissor({500, 500, 100, 100}, 1280, 720).width == 0);
    CHECK(CommandListStateTracker::convert_scissor({2000, 10, 3000, 20}, 1280, 720) ==
          (WgpuScissorRect{1280, 10, 0, 10}));
}

void test_tracker_dirty_filtering() {
    PipelineCache cache;
    GraphicsPso pso(basic_desc());
    CommandListStateTracker t;
    const RenderTargetViewInfo rt = rtv(1);
    const DepthStencilViewInfo ds = dsv(2, 1280, 720, D::D32_FLOAT);
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&rt, 1), &ds);
    t.SetPipelineState(&pso);
    set_full_viewport(t);

    CHECK(t.prepare_draw(&cache));
    // Full viewport/scissor equal the WebGPU per-pass defaults: nothing to emit besides the pipeline.
    CHECK(trace(t) == "BP");
    CHECK(t.events()[1].pipeline_key != nullptr);  // first use: creation packet needed
    CHECK(t.prepare_draw(&cache));
    CHECK(trace(t) == "BP");  // nothing changed

    // Redundant D3D12 calls are filtered.
    const uint64_t filtered = t.stats().redundant_calls_filtered;
    set_full_viewport(t);
    t.SetPipelineState(&pso);
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&rt, 1), &ds);
    t.OMSetStencilRef(0);
    t.IASetPrimitiveTopology(D3DPrimitiveTopology::Undefined);
    CHECK(t.stats().redundant_calls_filtered == filtered + 6);

    // Changes emit exactly the changed state.
    const D3D12Viewport half{0.0f, 0.0f, 640.0f, 360.0f, 0.0f, 1.0f};
    t.RSSetViewports(1, &half);
    const float bf[4] = {0.5f, 0.5f, 0.5f, 1.0f};
    t.OMSetBlendFactor(bf);
    t.OMSetStencilRef(3);
    CHECK(t.prepare_draw(&cache));
    CHECK(trace(t) == "BPVKR");
    CHECK(t.events()[2].viewport.width == 640.0f);
    CHECK(t.events()[3].blend_constant[0] == 0.5f);
    CHECK(t.events()[4].stencil_reference == 3);
    t.OMSetBlendFactor(nullptr);  // D3D12: null means {1,1,1,1}
    CHECK(t.prepare_draw(&cache));
    CHECK(trace(t) == "BPVKRK");
    CHECK(t.events()[5].blend_constant[3] == 1.0f);

    // Topology change -> new pipeline variant, created once.
    t.IASetPrimitiveTopology(D3DPrimitiveTopology::LineList);
    CHECK(t.prepare_draw(&cache));
    CHECK(trace(t) == "BPVKRKP");
    CHECK(t.events()[6].pipeline_key != nullptr);
    CHECK(t.events()[6].pipeline_key->primitive.topology == WgpuPrimitiveTopology::LineList);
    t.IASetPrimitiveTopology(D3DPrimitiveTopology::TriangleList);
    CHECK(t.prepare_draw(&cache));
    CHECK(t.events().back().type == StateEventType::SetPipeline);
    CHECK(t.events().back().pipeline_key == nullptr);  // already created
    CHECK(t.events().back().pipeline_id == t.events()[1].pipeline_id);

    // Strip topology + index format + bound vertex stride feed the variant key.
    t.IASetPrimitiveTopology(D3DPrimitiveTopology::TriangleStrip);
    t.set_index_format(WgpuIndexFormat::Uint16);
    const uint32_t stride = 16;
    t.set_vertex_strides(0, 1, &stride);
    CHECK(t.prepare_draw(&cache));
    const StateEvent& last = t.events().back();
    CHECK(last.type == StateEventType::SetPipeline && last.pipeline_key != nullptr);
    CHECK(last.pipeline_key->primitive.strip_index_format == WgpuIndexFormat::Uint16);
    CHECK(last.pipeline_key->vertex_buffers[0].array_stride == 16);
    CHECK(t.stats().passes_begun == 1);
    CHECK(t.stats().draws == 7);

    // Unset viewport/scissor falls back to the full attachment with a diagnostic.
    CommandListStateTracker u;
    u.set_render_targets(std::span<const RenderTargetViewInfo>(&rt, 1), nullptr);
    CHECK(u.prepare_draw(nullptr));
    CHECK(has(u.diagnostics(), StateDiag::ViewportNotSet | StateDiag::ScissorNotSet));
    CHECK(trace(u) == "B");

    // Scissor larger than the target is clamped to the attachment.
    const D3D12Rect big{-5, -5, 4000, 4000};
    const D3D12Rect small{10, 10, 20, 20};
    u.RSSetScissorRects(1, &small);
    CHECK(u.prepare_draw(nullptr));
    CHECK(trace(u) == "BS");
    u.RSSetScissorRects(1, &big);
    CHECK(u.prepare_draw(nullptr));
    CHECK(trace(u) == "BSS");
    CHECK(u.events().back().scissor == (WgpuScissorRect{0, 0, 1280, 720}));
    const D3D12Viewport vps[2] = {{0, 0, 1280, 720, 0, 1}, {0, 0, 10, 10, 0, 1}};
    u.RSSetViewports(2, vps);
    CHECK(has(u.diagnostics(), StateDiag::MultipleViewportsIgnored));

    // reset() returns to the initial state.
    t.reset();
    CHECK(t.events().empty() && !t.in_render_pass() && t.stats().draws == 0);
    CHECK(!t.prepare_draw(&cache));
    CHECK(has(t.diagnostics(), StateDiag::DrawWithoutAttachments));
}

void test_pass_inference_clear_then_draw() {
    PipelineCache cache;
    GraphicsPso pso(basic_desc());
    CommandListStateTracker t;
    const RenderTargetViewInfo rt = rtv(10);
    const DepthStencilViewInfo ds = dsv(11, 1280, 720, D::D32_FLOAT);
    const float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};

    // Clear before binding (common pattern) and after binding both fold into load ops.
    t.ClearRenderTargetView(rt, red);
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&rt, 1), &ds);
    t.ClearDepthStencilView(ds, D3D12ClearFlags::Depth, 0.0f, 0);
    t.SetPipelineState(&pso);
    set_full_viewport(t);
    CHECK(trace(t).empty());
    CHECK(!t.in_render_pass());
    CHECK(t.prepare_draw(&cache));
    CHECK(trace(t) == "BP");
    const RenderPassDesc& p = t.passes()[0];
    CHECK(p.color_count == 1 && p.width == 1280 && p.height == 720 && p.sample_count == 1);
    CHECK(p.color[0].view_id == 10 && p.color[0].format == TF::Rgba8Unorm);
    CHECK(p.color[0].load_op == WgpuLoadOp::Clear && p.color[0].store_op == WgpuStoreOp::Store);
    CHECK(p.color[0].clear_value[0] == 1.0f && p.color[0].clear_value[1] == 0.0f);
    CHECK(p.has_depth_stencil && p.depth_stencil.view_id == 11);
    CHECK(p.depth_stencil.depth_load_op == WgpuLoadOp::Clear && p.depth_stencil.depth_clear_value == 0.0f);
    CHECK(p.depth_stencil.stencil_load_op == WgpuLoadOp::Undefined);  // D32 has no stencil aspect
    CHECK(t.stats().clears_folded == 2);
    t.close();
    CHECK(trace(t) == "BPE");
    CHECK(!t.in_render_pass());
}

void test_pass_inference_rt_switch() {
    CommandListStateTracker t;
    const RenderTargetViewInfo a = rtv(1);
    const RenderTargetViewInfo b = rtv(2);
    const RenderTargetViewInfo gbuffer[3] = {rtv(3), rtv(4, 1280, 720, D::R16G16B16A16_FLOAT), rtv(5)};
    const DepthStencilViewInfo ds = dsv(9);
    set_full_viewport(t);

    t.set_render_targets(std::span<const RenderTargetViewInfo>(&a, 1), &ds);
    CHECK(t.prepare_draw(nullptr));
    // A -> B -> A without draws in between costs nothing.
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&b, 1), &ds);
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&a, 1), &ds);
    CHECK(t.prepare_draw(nullptr));
    CHECK(trace(t) == "B");
    // Real switch: end + begin.
    t.set_render_targets(std::span<const RenderTargetViewInfo>(gbuffer, 3), &ds);
    CHECK(t.prepare_draw(nullptr));
    CHECK(trace(t) == "BEB");
    CHECK(t.passes()[1].color_count == 3);
    CHECK(t.passes()[1].color[1].format == TF::Rgba16Float);
    CHECK(t.passes()[1].color[2].load_op == WgpuLoadOp::Load);
    CHECK(t.variant_state().num_render_targets == 3);
    CHECK(t.variant_state().rtv_formats[1] == D::R16G16B16A16_FLOAT);
    CHECK(t.variant_state().dsv_format == D::D24_UNORM_S8_UINT);
    // Same targets but the DSV becomes read-only: a new pass is needed (different attachment usage).
    const DepthStencilViewInfo ro = dsv(9, 1280, 720, D::D24_UNORM_S8_UINT, D3D12DsvFlags::ReadOnlyDepth);
    t.set_render_targets(std::span<const RenderTargetViewInfo>(gbuffer, 3), &ro);
    CHECK(t.prepare_draw(nullptr));
    CHECK(trace(t) == "BEBEB");
    CHECK(t.passes()[2].depth_stencil.depth_read_only);
    CHECK(t.passes()[2].depth_stencil.depth_load_op == WgpuLoadOp::Undefined);
    CHECK(t.passes()[2].depth_stencil.stencil_load_op == WgpuLoadOp::Load);
    // Holes in the RTV array are preserved.
    const RenderTargetViewInfo holes[3] = {rtv(1), RenderTargetViewInfo{}, rtv(2)};
    t.set_render_targets(std::span<const RenderTargetViewInfo>(holes, 3), nullptr);
    CHECK(t.prepare_draw(nullptr));
    CHECK(t.passes()[3].color_count == 3 && t.passes()[3].color[1].view_id == 0 && !t.passes()[3].has_depth_stencil);
    // Mismatched sizes are diagnosed; the pass uses the smallest extent.
    const RenderTargetViewInfo mixed[2] = {rtv(1), rtv(7, 640, 360)};
    t.set_render_targets(std::span<const RenderTargetViewInfo>(mixed, 2), nullptr);
    CHECK(t.prepare_draw(nullptr));
    CHECK(has(t.diagnostics(), StateDiag::AttachmentSizeMismatch));
    CHECK(t.passes()[4].width == 640);
    t.clear_events();
    CHECK(t.events().empty() && t.passes().empty());
    CHECK(t.in_render_pass());
}

void test_pass_inference_depth_only() {
    PipelineCache cache;
    D3D12GraphicsPsoDesc shadow = basic_desc();
    shadow.ps = 0;
    shadow.NumRenderTargets = 0;
    shadow.RTVFormats[0] = D::UNKNOWN;
    shadow.RasterizerState.DepthBias = 16;
    shadow.RasterizerState.SlopeScaledDepthBias = 2.0f;
    GraphicsPso pso(shadow);
    CommandListStateTracker t;
    const DepthStencilViewInfo sm = dsv(20, 2048, 2048, D::D32_FLOAT);
    t.set_render_targets({}, &sm);
    t.ClearDepthStencilView(sm, D3D12ClearFlags::Depth | D3D12ClearFlags::Stencil, 1.0f, 0);
    t.SetPipelineState(&pso);
    set_full_viewport(t, 2048.0f, 2048.0f);
    CHECK(t.prepare_draw(&cache));
    CHECK(trace(t) == "BP");
    const RenderPassDesc& p = t.passes()[0];
    CHECK(p.color_count == 0 && p.has_depth_stencil);
    CHECK(p.depth_stencil.depth_load_op == WgpuLoadOp::Clear && p.depth_stencil.depth_clear_value == 1.0f);
    CHECK(p.width == 2048);
    const GraphicsPipelineKey* key = t.events()[1].pipeline_key;
    CHECK(key != nullptr);
    CHECK(key->color_target_count == 0 && key->ps_shader == 0);
    CHECK(key->depth_stencil.format == TF::Depth32Float && key->depth_stencil.depth_bias == 16);
    CHECK(t.pipeline_diagnostics() == 0);
}

void test_pass_inference_mid_pass_clear() {
    const RenderTargetViewInfo rt = rtv(1);
    const DepthStencilViewInfo ds = dsv(2);
    const float blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    PipelineCache cache;
    GraphicsPso pso(basic_desc());

    // SplitPass (default): end, then fold into the next pass's load op.
    CommandListStateTracker split;
    split.set_render_targets(std::span<const RenderTargetViewInfo>(&rt, 1), &ds);
    split.SetPipelineState(&pso);
    set_full_viewport(split);
    CHECK(split.prepare_draw(&cache));
    split.ClearRenderTargetView(rt, blue);
    CHECK(!split.in_render_pass());
    CHECK(split.prepare_draw(&cache));
    CHECK(trace(split) == "BPEBP");
    CHECK(split.passes()[1].color[0].load_op == WgpuLoadOp::Clear);
    CHECK(split.passes()[1].color[0].clear_value[2] == 1.0f);
    CHECK(split.passes()[1].depth_stencil.depth_load_op == WgpuLoadOp::Load);

    // DrawFallback: stay in the pass, emit a full-rect clear, then re-emit pass state.
    TrackerOptions opts{};
    opts.mid_pass_clear = MidPassClearPolicy::DrawFallback;
    CommandListStateTracker fb(opts);
    fb.set_render_targets(std::span<const RenderTargetViewInfo>(&rt, 1), &ds);
    fb.SetPipelineState(&pso);
    set_full_viewport(fb);
    CHECK(fb.prepare_draw(&cache));
    fb.ClearDepthStencilView(ds, D3D12ClearFlags::Stencil, 1.0f, 0x80);
    CHECK(fb.in_render_pass());
    CHECK(fb.prepare_draw(&cache));
    CHECK(trace(fb) == "BPCPVSKR");
    const ClearFallback& c = fb.clear_fallbacks()[0];
    CHECK(c.depth_stencil && c.view_id == 2 && c.clear_flags == D3D12ClearFlags::Stencil && c.stencil == 0x80);
    CHECK(c.rects.size() == 1 && c.rects[0] == (WgpuScissorRect{0, 0, 1280, 720}));
    CHECK(fb.stats().clear_fallbacks == 1 && fb.stats().passes_begun == 1);
}

void test_pass_inference_partial_clears() {
    const RenderTargetViewInfo rt = rtv(1);
    const RenderTargetViewInfo other = rtv(5, 256, 256);
    const float green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    CommandListStateTracker t;
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&rt, 1), nullptr);
    set_full_viewport(t);

    // Partial clear of a bound view before any draw: begins the bound pass, which the draw then reuses.
    const D3D12Rect rects[3] = {{-10, -10, 100, 100}, {2000, 2000, 2100, 2100}, {200, 200, 300, 260}};
    t.ClearRenderTargetView(rt, green, 3, rects);
    CHECK(trace(t) == "BC");
    const ClearFallback& c = t.clear_fallbacks()[0];
    CHECK(!c.depth_stencil && c.attachment_index == 0 && c.color[1] == 1.0f);
    CHECK(c.rects.size() == 2);  // the off-target rect was dropped
    CHECK(c.rects[0] == (WgpuScissorRect{0, 0, 100, 100}));
    CHECK(c.rects[1] == (WgpuScissorRect{200, 200, 100, 60}));
    CHECK(t.prepare_draw(nullptr));
    CHECK(trace(t) == "BCVSKR");  // fallback clobbered encoder state: re-emitted
    CHECK(t.stats().passes_begun == 1);

    // A rect covering the whole view is a full clear.
    const D3D12Rect cover{0, 0, 1280, 720};
    t.ClearRenderTargetView(rt, green, 1, &cover);
    CHECK(trace(t) == "BCVSKRE");  // mid-pass full clear (SplitPass)
    // Fully clipped rects are a no-op.
    const D3D12Rect outside{5000, 5000, 6000, 6000};
    t.ClearRenderTargetView(rt, green, 1, &outside);
    CHECK(trace(t) == "BCVSKRE");

    // Partial clear of an unbound view: temporary pass around the fallback.
    t.ClearRenderTargetView(other, green, 1, &rects[2]);
    CHECK(trace(t) == "BCVSKREBCE");
    CHECK(t.passes().back().color[0].view_id == 5);
    CHECK(t.passes().back().width == 256);
    CHECK(t.clear_fallbacks().back().rects[0] == (WgpuScissorRect{200, 200, 56, 56}));
    // The pending full clear of the bound view folds into the next draw's pass.
    CHECK(t.prepare_draw(nullptr));
    CHECK(t.passes().back().color[0].view_id == 1);
    CHECK(t.passes().back().color[0].load_op == WgpuLoadOp::Clear);
}

void test_pending_clear_flush_and_merge() {
    CommandListStateTracker t;
    const RenderTargetViewInfo bound = rtv(1);
    const RenderTargetViewInfo unbound = rtv(2, 64, 64, D::R16G16B16A16_FLOAT);
    const DepthStencilViewInfo ds = dsv(3);
    const float c1[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float c2[4] = {0.5f, 0.6f, 0.7f, 0.8f};
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&bound, 1), &ds);
    set_full_viewport(t);
    CHECK(t.prepare_draw(nullptr));

    // Clears of views outside the open pass stay pending; the last colour wins, depth+stencil merge.
    t.ClearRenderTargetView(unbound, c1);
    t.ClearRenderTargetView(unbound, c2);
    const DepthStencilViewInfo other_ds = dsv(4, 64, 64);
    t.ClearDepthStencilView(other_ds, D3D12ClearFlags::Depth, 0.25f, 0);
    t.ClearDepthStencilView(other_ds, D3D12ClearFlags::Stencil, 0.0f, 7);
    CHECK(t.in_render_pass());
    CHECK(trace(t) == "B");

    // Read-only aspects cannot be cleared.
    const DepthStencilViewInfo ro = dsv(6, 64, 64, D::D24_UNORM_S8_UINT, D3D12DsvFlags::ReadOnlyDepth);
    t.ClearDepthStencilView(ro, D3D12ClearFlags::Depth, 0.0f, 0);
    CHECK(has(t.diagnostics(), StateDiag::ClearOnReadOnlyAspect));

    t.close();
    CHECK(trace(t) == "BEBEBE");
    CHECK(t.stats().clear_only_passes == 2);
    const RenderPassDesc& color_pass = t.passes()[1];
    CHECK(color_pass.color[0].view_id == 2 && color_pass.color[0].load_op == WgpuLoadOp::Clear);
    CHECK(color_pass.color[0].clear_value[0] == 0.5f);
    CHECK(color_pass.color[0].format == TF::Rgba16Float);
    const RenderPassDesc& ds_pass = t.passes()[2];
    CHECK(ds_pass.color_count == 0 && ds_pass.depth_stencil.view_id == 4);
    CHECK(ds_pass.depth_stencil.depth_load_op == WgpuLoadOp::Clear && ds_pass.depth_stencil.depth_clear_value == 0.25f);
    CHECK(ds_pass.depth_stencil.stencil_load_op == WgpuLoadOp::Clear && ds_pass.depth_stencil.stencil_clear_value == 7);
    t.close();
    CHECK(trace(t) == "BEBEBE");  // idempotent
}

class FakeResolver final : public DescriptorViewResolver {
public:
    uint64_t rtv_descriptor_increment() const override { return 32; }
    bool resolve_rtv(uint64_t handle, RenderTargetViewInfo* out) const override {
        if (handle < 0x1000 || handle >= 0x1000 + 32 * 8 || (handle - 0x1000) % 32 != 0) {
            return false;
        }
        *out = rtv(static_cast<uint32_t>(100 + (handle - 0x1000) / 32));
        return true;
    }
    bool resolve_dsv(uint64_t handle, DepthStencilViewInfo* out) const override {
        if (handle != 0x9000) {
            return false;
        }
        *out = dsv(200);
        return true;
    }
};

void test_om_set_render_targets_descriptors() {
    FakeResolver resolver;
    CommandListStateTracker t;
    set_full_viewport(t);

    // Single handle to a descriptor range: consecutive descriptors.
    const uint64_t first = 0x1000;
    const uint64_t ds = 0x9000;
    t.OMSetRenderTargets(3, &first, true, &ds, resolver);
    CHECK(t.prepare_draw(nullptr));
    const RenderPassDesc& p = t.passes()[0];
    CHECK(p.color_count == 3);
    CHECK(p.color[0].view_id == 100 && p.color[1].view_id == 101 && p.color[2].view_id == 102);
    CHECK(p.depth_stencil.view_id == 200);

    // Array form, same views: redundant.
    const uint64_t handles[3] = {0x1000, 0x1020, 0x1040};
    const uint64_t filtered = t.stats().redundant_calls_filtered;
    t.OMSetRenderTargets(3, handles, false, &ds, resolver);
    CHECK(t.stats().redundant_calls_filtered == filtered + 1);

    // Unresolvable handle becomes a hole with a diagnostic.
    const uint64_t bad[2] = {0x1000, 0x1234};
    t.OMSetRenderTargets(2, bad, false, nullptr, resolver);
    CHECK(has(t.diagnostics(), StateDiag::UnresolvedDescriptor));
    CHECK(t.variant_state().num_render_targets == 1);
    CHECK(t.variant_state().dsv_format == D::UNKNOWN);

    // No attachments at all: the draw is skipped.
    t.OMSetRenderTargets(0, nullptr, false, nullptr, resolver);
    CHECK(!t.prepare_draw(nullptr));
    CHECK(t.stats().draws_skipped == 1);
}

void test_tracker_pipeline_failures() {
    PipelineCache cache;
    CommandListStateTracker t;
    const RenderTargetViewInfo rt = rtv(1);
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&rt, 1), nullptr);
    set_full_viewport(t);
    CHECK(!t.prepare_draw(&cache));  // no PSO
    CHECK(has(t.diagnostics(), StateDiag::DrawWithoutPipeline));
    D3D12GraphicsPsoDesc gs = basic_desc();
    gs.gs = 1;
    GraphicsPso fatal(gs);
    t.SetPipelineState(&fatal);
    CHECK(!t.prepare_draw(&cache));
    CHECK(has(t.diagnostics(), StateDiag::PipelineUnusable));
    CHECK(has(t.pipeline_diagnostics(), PipelineDiag::GeometryShaderUnsupported));
    CHECK(trace(t).empty());  // no pass was begun for skipped draws
    CHECK(t.stats().draws_skipped == 2);

    // Bound sRGB view vs PSO UNORM format: variant created with the bound format, diagnosed.
    GraphicsPso pso(basic_desc());
    const RenderTargetViewInfo srgb = rtv(2, 1280, 720, D::R8G8B8A8_UNORM_SRGB);
    t.set_render_targets(std::span<const RenderTargetViewInfo>(&srgb, 1), nullptr);
    t.SetPipelineState(&pso);
    CHECK(t.prepare_draw(&cache));
    CHECK(has(t.pipeline_diagnostics(), PipelineDiag::RenderTargetFormatMismatch));
    CHECK(t.events().back().pipeline_key->color_targets[0].format == TF::Rgba8UnormSrgb);
    CHECK(t.events().back().pipeline_key->depth_stencil.format == TF::Undefined);
}

}  // namespace

int main() {
    test_sampled_only_norm16();
    test_enum_tables();
    test_texture_format_mapping();
    test_vertex_format_mapping();
    test_input_layout();
    test_key_canonicalisation();
    test_variants();
    test_diagnostics();
    test_cache_basics();
    test_cache_collisions();
    test_cache_concurrency();
    test_viewport_scissor_conversion();
    test_tracker_dirty_filtering();
    test_pass_inference_clear_then_draw();
    test_pass_inference_rt_switch();
    test_pass_inference_depth_only();
    test_pass_inference_mid_pass_clear();
    test_pass_inference_partial_clears();
    test_pending_clear_flush_and_merge();
    test_om_set_render_targets_descriptors();
    test_tracker_pipeline_failures();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_pso_cache failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_pso_cache_tests passed.\n");
    return 0;
}
