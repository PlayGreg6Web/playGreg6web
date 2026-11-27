#include "gpu/state_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "gpu/stage_reflection.h"
#include "gpu/pipeline_prewarm.h"
#include "gpu/prewarm_pipeline_desc.h"

namespace playgta6::gpu::bridge {

// =====================================================================================================
// Seam: the three type families must agree wherever values cross without conversion.
// =====================================================================================================

static_assert(static_cast<uint32_t>(d3d::BindResourceKind::Buffer) == static_cast<uint32_t>(GpuBindResourceType::Buffer));
static_assert(static_cast<uint32_t>(d3d::BindResourceKind::TextureView) ==
              static_cast<uint32_t>(GpuBindResourceType::TextureView));
static_assert(static_cast<uint32_t>(d3d::BindResourceKind::Sampler) == static_cast<uint32_t>(GpuBindResourceType::Sampler));
static_assert(d3d::DsvFlags::ReadOnlyDepth == pso::D3D12DsvFlags::ReadOnlyDepth);
static_assert(d3d::DsvFlags::ReadOnlyStencil == pso::D3D12DsvFlags::ReadOnlyStencil);
static_assert(d3d::WebGpuStage::Vertex == GpuShaderStageBits::Vertex);
static_assert(d3d::WebGpuStage::Fragment == GpuShaderStageBits::Fragment);
static_assert(d3d::WebGpuStage::Compute == GpuShaderStageBits::Compute);
static_assert(static_cast<uint8_t>(pso::WgpuLoadOp::Load) == static_cast<uint8_t>(GpuLoadOp::Load));
static_assert(static_cast<uint8_t>(pso::WgpuLoadOp::Clear) == static_cast<uint8_t>(GpuLoadOp::Clear));
static_assert(static_cast<uint8_t>(pso::WgpuStoreOp::Store) == static_cast<uint8_t>(GpuStoreOp::Store));
static_assert(static_cast<uint8_t>(pso::WgpuStoreOp::Discard) == static_cast<uint8_t>(GpuStoreOp::Discard));
static_assert(static_cast<uint8_t>(pso::WgpuTextureAspect::DepthOnly) == static_cast<uint8_t>(GpuTextureAspect::DepthOnly));
static_assert(static_cast<uint8_t>(pso::WgpuTextureAspect::StencilOnly) ==
              static_cast<uint8_t>(GpuTextureAspect::StencilOnly));
static_assert(sizeof(GpuRenderPipelineKey) == sizeof(pso::GraphicsPipelineKey));
static_assert(sizeof(GpuComputePipelineKey) == sizeof(pso::ComputePipelineKey));
static_assert(pso::kD3D12SimultaneousRenderTargetCount == kGpuMaxColorAttachments);
static_assert(static_cast<uint32_t>(pso::D3D12ComparisonFunc::Always) == 8);  // D3D12_COMPARISON_FUNC

namespace {

constexpr uint32_t kRootConstantAlignment = 256;  // minUniformBufferOffsetAlignment (default limit)
constexpr uint64_t kFnv1aOffsetBasis64 = 0xcbf29ce484222325ull;
constexpr uint64_t kFnv1aPrime64 = 0x100000001b3ull;

uint32_t align_up(uint32_t value, uint32_t alignment) noexcept {
    return (value + alignment - 1u) / alignment * alignment;
}

float canonical_finite_float(float value) noexcept {
    if (!std::isfinite(value) || value == 0.0f) {
        return 0.0f;
    }
    return value;
}

uint32_t canonical_finite_bits(float value) noexcept {
    const float clean = canonical_finite_float(value);
    uint32_t bits = 0;
    std::memcpy(&bits, &clean, sizeof(bits));
    return bits;
}

bool is_dual_source_factor_code(uint8_t f) noexcept {
    using W = pso::WgpuBlendFactor;
    return f == static_cast<uint8_t>(W::Src1) || f == static_cast<uint8_t>(W::OneMinusSrc1) ||
           f == static_cast<uint8_t>(W::Src1Alpha) || f == static_cast<uint8_t>(W::OneMinusSrc1Alpha);
}

uint8_t downgrade_dual_source_factor_code(uint8_t f) noexcept {
    using W = pso::WgpuBlendFactor;
    switch (static_cast<W>(f)) {
    case W::Src1: return static_cast<uint8_t>(W::Src);
    case W::OneMinusSrc1: return static_cast<uint8_t>(W::OneMinusSrc);
    case W::Src1Alpha: return static_cast<uint8_t>(W::SrcAlpha);
    case W::OneMinusSrc1Alpha: return static_cast<uint8_t>(W::OneMinusSrcAlpha);
    default: return f;
    }
}

uint32_t mip_extent(uint32_t extent, uint32_t mip) noexcept {
    return mip >= 32 ? 1u : std::max(1u, extent >> mip);
}

uint8_t texture_format_code(pso::DxgiFormat format, pso::FormatUsage usage, uint32_t features,
                            pso::WgpuTextureAspect* aspect) {
    const pso::TextureFormatMapping m = pso::map_texture_format(format, usage, features);
    if (aspect != nullptr) {
        *aspect = m.aspect;
    }
    return static_cast<uint8_t>(m.format);
}

GpuAddressMode to_address_mode(uint32_t d3d12_mode, bool* exact) noexcept {
    switch (d3d12_mode) {
    case 1: return GpuAddressMode::Repeat;        // WRAP
    case 2: return GpuAddressMode::MirrorRepeat;  // MIRROR
    case 3: return GpuAddressMode::ClampToEdge;   // CLAMP
    case 4:                                       // BORDER: no border colors in WebGPU
        *exact = false;
        return GpuAddressMode::ClampToEdge;
    case 5:  // MIRROR_ONCE
        *exact = false;
        return GpuAddressMode::MirrorRepeat;
    default:
        *exact = false;
        return GpuAddressMode::ClampToEdge;
    }
}

uint64_t hash_shader_binding(uint64_t seed, const d3d::ShaderBinding& b) noexcept {
    uint64_t h = hash_combine64(seed, b.binding);
    h = hash_combine64(h, (static_cast<uint64_t>(b.register_class) << 32) | b.shader_register);
    h = hash_combine64(h, (uint64_t{b.register_space} << 32) | (b.root_constants ? 1u : 0u) |
                              (b.inline_sampler ? 2u : 0u) | (static_cast<uint64_t>(b.heap) << 8) |
                              (static_cast<uint64_t>(b.heap_index_source) << 16));
    h = hash_combine64(h, (uint64_t{b.heap_index_parameter} << 32) | b.heap_index_dword);
    h = hash_combine64(h, (uint64_t{b.heap_shift} << 32) | b.cb_table_slot);
    h = hash_combine64(h, (uint64_t{b.cb_byte_offset} << 32) | b.cb_bit_offset);
    h = hash_combine64(h, (uint64_t{b.cb_mask} << 32) | b.heap_stride);
    h = hash_combine64(h, (uint64_t{b.array_size} << 32) | b.array_index);
    h = hash_combine64(h, (uint64_t{b.table_parameter} << 32) | b.table_slot);
    if (b.inline_sampler) {
        h = hash_combine64(h, fnv1a_hash64(&b.sampler, sizeof(b.sampler)));
    }
    return h;
}

}  // namespace

uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

uint64_t fnv1a_hash64(const void* data, size_t size, uint64_t seed) noexcept {
    uint64_t h = seed;
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= kFnv1aPrime64;
    }
    return h;
}

uint64_t hash_combine64(uint64_t seed, uint64_t value) noexcept {
    return splitmix64(seed ^ (splitmix64(value) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2)));
}

uint64_t hash_sampler_payload(const GpuCreateSampler2Payload& payload) noexcept {
    const StateBridgeDevice::SamplerKey key = StateBridgeDevice::make_sampler_key(payload);
    return splitmix64(fnv1a_hash64(&key, sizeof(key)));
}

uint64_t hash_view_record(const ViewRecord& record) noexcept {
    const StateBridgeDevice::ViewKey key = StateBridgeDevice::make_view_key(record);
    return splitmix64(fnv1a_hash64(&key, sizeof(key)));
}

uint64_t hash_render_pipeline_key(const GpuRenderPipelineKey& key) noexcept {
    return splitmix64(fnv1a_hash64(&key, sizeof(key)));
}

pso::WgpuIndexFormat to_wgpu_index_format(GpuIndexFormat format) noexcept {
    return format == GpuIndexFormat::Uint32 ? pso::WgpuIndexFormat::Uint32 : pso::WgpuIndexFormat::Uint16;
}

GpuPrimitiveTopology to_gpu_primitive_topology(pso::D3DPrimitiveTopology topology, bool* exact) noexcept {
    const pso::TopologyMapping tm = pso::map_primitive_topology(topology);
    if (exact != nullptr) {
        *exact = (tm.diagnostics == 0);
    }
    return static_cast<GpuPrimitiveTopology>(tm.topology);
}

bool is_strip_topology(GpuPrimitiveTopology topology) noexcept {
    return topology == GpuPrimitiveTopology::TriangleStrip || topology == GpuPrimitiveTopology::LineStrip;
}

bool is_strip_topology(pso::WgpuPrimitiveTopology topology) noexcept {
    return topology == pso::WgpuPrimitiveTopology::TriangleStrip ||
           topology == pso::WgpuPrimitiveTopology::LineStrip;
}

pso::WgpuIndexFormat resolve_strip_index_format(pso::WgpuPrimitiveTopology topology,
                                                pso::WgpuIndexFormat bound_index_format,
                                                pso::D3D12IndexBufferStripCutValue strip_cut) noexcept {
    if (!is_strip_topology(topology)) {
        return pso::WgpuIndexFormat::Undefined;
    }
    if (bound_index_format == pso::WgpuIndexFormat::Uint16 || bound_index_format == pso::WgpuIndexFormat::Uint32) {
        return bound_index_format;
    }
    switch (strip_cut) {
    case pso::D3D12IndexBufferStripCutValue::Value0xFFFF: return pso::WgpuIndexFormat::Uint16;
    case pso::D3D12IndexBufferStripCutValue::Value0xFFFFFFFF: return pso::WgpuIndexFormat::Uint32;
    default: return pso::WgpuIndexFormat::Undefined;
    }
}

float sanitize_depth_bias_float(float value) noexcept {
    return canonical_finite_float(value);
}

uint32_t sanitize_depth_bias_bits(float value) noexcept {
    return canonical_finite_bits(value);
}

bool sanitize_render_pipeline_key(GpuRenderPipelineKey* key, uint32_t features) noexcept {
    if (key == nullptr) {
        return false;
    }
    const GpuRenderPipelineKey before = *key;

    if (key->color_target_count > kGpuMaxColorAttachments) {
        key->color_target_count = kGpuMaxColorAttachments;
    }
    constexpr GpuBlendComponentKey kDefaultBlend{
        static_cast<uint8_t>(pso::WgpuBlendOperation::Add),
        static_cast<uint8_t>(pso::WgpuBlendFactor::One),
        static_cast<uint8_t>(pso::WgpuBlendFactor::Zero),
        0,
    };
    const bool dual_source_ok = (features & pso::WgpuFeature::DualSourceBlending) != 0;
    for (uint32_t i = 0; i < kGpuMaxColorAttachments; ++i) {
        GpuColorTargetKey& ct = key->color_targets[i];
        if (i >= key->color_target_count || ct.format == 0 ||
            ct.format >= static_cast<uint8_t>(pso::WgpuTextureFormat::Count)) {
            ct = GpuColorTargetKey{};
            continue;
        }
        ct.write_mask &= 0xFu;
        ct.reserved = 0;
        const auto fmt = static_cast<pso::WgpuTextureFormat>(ct.format);
        if (ct.blend_enabled != 0) {
            if (!pso::is_blendable(fmt, features)) {
                ct.blend_enabled = 0;
            } else {
                ct.blend_enabled = 1;
                auto clean_comp = [&](GpuBlendComponentKey& c) {
                    c.reserved = 0;
                    if (!dual_source_ok) {
                        if (is_dual_source_factor_code(c.src_factor)) {
                            c.src_factor = downgrade_dual_source_factor_code(c.src_factor);
                        }
                        if (is_dual_source_factor_code(c.dst_factor)) {
                            c.dst_factor = downgrade_dual_source_factor_code(c.dst_factor);
                        }
                    }
                    if (c.operation == static_cast<uint8_t>(pso::WgpuBlendOperation::Min) ||
                        c.operation == static_cast<uint8_t>(pso::WgpuBlendOperation::Max)) {
                        c.src_factor = static_cast<uint8_t>(pso::WgpuBlendFactor::One);
                        c.dst_factor = static_cast<uint8_t>(pso::WgpuBlendFactor::One);
                    }
                };
                clean_comp(ct.color);
                clean_comp(ct.alpha);
            }
        }
        if (ct.blend_enabled == 0) {
            ct.color = kDefaultBlend;
            ct.alpha = kDefaultBlend;
        }
    }

    // Primitive topology & stripIndexFormat rules.
    const auto topo = static_cast<pso::WgpuPrimitiveTopology>(key->primitive.topology);
    const bool strip = is_strip_topology(topo);
    const bool triangle = topo == pso::WgpuPrimitiveTopology::TriangleList ||
                          topo == pso::WgpuPrimitiveTopology::TriangleStrip;
    if (!strip) {
        key->primitive.strip_index_format = static_cast<uint8_t>(pso::WgpuIndexFormat::Undefined);
    } else if (key->primitive.strip_index_format > static_cast<uint8_t>(pso::WgpuIndexFormat::Uint32)) {
        key->primitive.strip_index_format = static_cast<uint8_t>(pso::WgpuIndexFormat::Undefined);
    }
    key->primitive.reserved[0] = 0;
    key->primitive.reserved[1] = 0;
    key->primitive.reserved[2] = 0;

    // Depth-stencil & depth-bias rules.
    GpuDepthStencilKey& ds = key->depth_stencil;
    if (ds.format == 0 || ds.format >= static_cast<uint8_t>(pso::WgpuTextureFormat::Count)) {
        ds = GpuDepthStencilKey{};
    } else {
        const auto ds_fmt = static_cast<pso::WgpuTextureFormat>(ds.format);
        const bool has_depth = pso::has_depth_aspect(ds_fmt);
        const bool has_stencil = pso::has_stencil_aspect(ds_fmt);
        ds.reserved = 0;
        if (!has_depth) {
            ds.depth_write_enabled = 0;
            ds.depth_compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
        } else {
            ds.depth_write_enabled = ds.depth_write_enabled != 0 ? 1 : 0;
            if (ds.depth_compare == 0 || ds.depth_compare >= static_cast<uint8_t>(pso::WgpuCompareFunction::Count)) {
                ds.depth_compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
            }
        }
        constexpr GpuStencilFaceKey kDefaultStencil{
            static_cast<uint8_t>(pso::WgpuCompareFunction::Always),
            static_cast<uint8_t>(pso::WgpuStencilOperation::Keep),
            static_cast<uint8_t>(pso::WgpuStencilOperation::Keep),
            static_cast<uint8_t>(pso::WgpuStencilOperation::Keep),
        };
        if (!has_stencil) {
            ds.stencil_front = kDefaultStencil;
            ds.stencil_back = kDefaultStencil;
            ds.stencil_read_mask = 0;
            ds.stencil_write_mask = 0;
        } else {
            if (ds.stencil_front.compare == 0 ||
                ds.stencil_front.compare >= static_cast<uint8_t>(pso::WgpuCompareFunction::Count)) {
                ds.stencil_front.compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
            }
            if (ds.stencil_back.compare == 0 ||
                ds.stencil_back.compare >= static_cast<uint8_t>(pso::WgpuCompareFunction::Count)) {
                ds.stencil_back.compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
            }
        }
        float slope = 0.0f;
        float clamp = 0.0f;
        std::memcpy(&slope, &ds.depth_bias_slope_scale_bits, sizeof(slope));
        std::memcpy(&clamp, &ds.depth_bias_clamp_bits, sizeof(clamp));
        ds.depth_bias_slope_scale_bits = sanitize_depth_bias_bits(slope);
        ds.depth_bias_clamp_bits = sanitize_depth_bias_bits(clamp);

        const bool depth_used =
            has_depth && (ds.depth_write_enabled != 0 ||
                          ds.depth_compare != static_cast<uint8_t>(pso::WgpuCompareFunction::Always));
        if (!depth_used || !triangle) {
            ds.depth_bias = 0;
            ds.depth_bias_slope_scale_bits = 0;
            ds.depth_bias_clamp_bits = 0;
        }
    }
    key->reserved = 0;
    return std::memcmp(&before, key, sizeof(GpuRenderPipelineKey)) != 0;
}

CompatibleViewFormats compatible_view_formats(pso::WgpuTextureFormat format) noexcept {
    CompatibleViewFormats out{};
    const pso::WgpuTextureFormat sibling = pso::srgb_sibling(format);
    if (sibling != pso::WgpuTextureFormat::Undefined) {
        out.formats[0] = sibling;
        out.count = 1;
    }
    return out;
}

CompatibleViewFormats compatible_view_formats_for_dxgi(pso::DxgiFormat dxgi_format,
                                                       pso::FormatUsage usage,
                                                       uint32_t features) noexcept {
    const pso::TextureFormatMapping m = pso::map_texture_format(dxgi_format, usage, features);
    if (!m.supported()) {
        return CompatibleViewFormats{};
    }
    return compatible_view_formats(m.format);
}

bool is_compatible_view_format(pso::WgpuTextureFormat texture_format, pso::WgpuTextureFormat view_format) noexcept {
    if (view_format == pso::WgpuTextureFormat::Undefined) {
        return true;
    }
    if (texture_format == pso::WgpuTextureFormat::Undefined) {
        return false;
    }
    if (texture_format == view_format) {
        return true;
    }
    return pso::srgb_sibling(texture_format) == view_format;
}

bool is_srgb_format(pso::WgpuTextureFormat format) noexcept {
    return (pso::texture_format_info(format).caps & pso::TextureFormatCaps::Srgb) != 0;
}

bool is_hdr_color_format(pso::DxgiFormat format) noexcept {
    using D = pso::DxgiFormat;
    switch (format) {
    case D::R16G16B16A16_FLOAT:
    case D::R16G16B16A16_TYPELESS:
    case D::R11G11B10_FLOAT:
    case D::R10G10B10A2_UNORM:
    case D::R10G10B10A2_TYPELESS:
    case D::R32G32B32A32_FLOAT:
    case D::R32G32B32A32_TYPELESS:
    case D::R32G32_FLOAT:
    case D::R32_FLOAT:
    case D::R16G16_FLOAT:
    case D::R16_FLOAT:
    case D::R9G9B9E5_SHAREDEXP:
    case D::R10G10B10_7E3_A2_FLOAT:
    case D::R10G10B10_6E4_A2_FLOAT:
    case D::BC6H_UF16:
    case D::BC6H_SF16:
    case D::BC6H_TYPELESS:
        return true;
    default:
        return false;
    }
}

bool is_depth_stencil_dxgi_format(pso::DxgiFormat format) noexcept {
    using D = pso::DxgiFormat;
    switch (format) {
    case D::D32_FLOAT_S8X24_UINT:
    case D::R32G8X24_TYPELESS:
    case D::R32_FLOAT_X8X24_TYPELESS:
    case D::X32_TYPELESS_G8X24_UINT:
    case D::D32_FLOAT:
    case D::D24_UNORM_S8_UINT:
    case D::R24G8_TYPELESS:
    case D::R24_UNORM_X8_TYPELESS:
    case D::X24_TYPELESS_G8_UINT:
    case D::D16_UNORM:
    case D::D16_UNORM_S8_UINT:
    case D::R16_UNORM_X8_TYPELESS:
    case D::X16_TYPELESS_G8_UINT:
        return true;
    default:
        return false;
    }
}

ResolvedViewFormat resolve_sampled_view_format(pso::DxgiFormat texture_dxgi,
                                               pso::DxgiFormat view_dxgi,
                                               bool sampled_only,
                                               uint32_t features,
                                               bool is_depth_resource) noexcept {
    using D = pso::DxgiFormat;
    using F = pso::WgpuTextureFormat;
    using A = pso::WgpuTextureAspect;
    ResolvedViewFormat out{};

    const bool explicit_depth_tex = is_depth_stencil_dxgi_format(texture_dxgi);
    const bool typeless_depth_tex =
        is_depth_resource && (texture_dxgi == D::R32_TYPELESS || texture_dxgi == D::R16_TYPELESS);
    const bool depth_tex = explicit_depth_tex || typeless_depth_tex ||
                           ((texture_dxgi == D::R32_TYPELESS && view_dxgi == D::D32_FLOAT) ||
                            (texture_dxgi == D::R16_TYPELESS && view_dxgi == D::D16_UNORM));

    if (depth_tex) {
        const pso::TextureFormatMapping dm =
            pso::map_texture_format(texture_dxgi, pso::FormatUsage::DepthStencil, features);
        if (dm.supported()) {
            out.format = dm.format;
            out.depth_sibling = true;
            const bool stencil_req = view_dxgi == D::X24_TYPELESS_G8_UINT ||
                                     view_dxgi == D::X32_TYPELESS_G8X24_UINT ||
                                     view_dxgi == D::X16_TYPELESS_G8_UINT;
            if (stencil_req && pso::has_stencil_aspect(dm.format)) {
                out.aspect = A::StencilOnly;
            } else if (pso::has_depth_aspect(dm.format)) {
                out.aspect = A::DepthOnly;
            } else if (pso::has_stencil_aspect(dm.format)) {
                out.aspect = A::StencilOnly;
            }
            return out;
        }
    }

    const pso::FormatUsage usage = sampled_only ? pso::FormatUsage::SampledOnly : pso::FormatUsage::Sampled;
    const pso::TextureFormatMapping tex_map =
        texture_dxgi != D::UNKNOWN ? pso::map_texture_format(texture_dxgi, usage, features)
                                   : pso::TextureFormatMapping{};

    if (view_dxgi == D::UNKNOWN) {
        out.format = F::Undefined;  // default view uses the texture's own format
        out.aspect = tex_map.aspect;
        return out;
    }

    if (texture_dxgi == D::A8_UNORM && view_dxgi != D::A8_UNORM) {
        // A8_UNORM is promoted to rgba8unorm (0, 0, 0, a) and cannot be sRGB-aliased or reinterpreted.
        out.format = tex_map.format;
        out.aspect = tex_map.aspect;
        out.fallback_to_base = true;
        return out;
    }
    const pso::TextureFormatMapping view_map = pso::map_texture_format(view_dxgi, usage, features);
    if (!tex_map.supported()) {
        out.format = view_map.format;
        out.aspect = view_map.aspect;
        return out;
    }
    if (!view_map.supported()) {
        out.format = tex_map.format;
        out.aspect = tex_map.aspect;
        out.fallback_to_base = true;
        return out;
    }
    if (view_map.format == tex_map.format) {
        out.format = view_map.format;
        out.aspect = view_map.aspect;
        return out;
    }
    if (is_compatible_view_format(tex_map.format, view_map.format)) {
        out.format = view_map.format;
        out.aspect = A::All;
        out.srgb_aliased = true;
        return out;
    }
    // WebGPU rejects cross-type view format reinterpretation (e.g. rgba8unorm texture viewed as
    // rgba8uint): fall back cleanly to the texture's base format.
    out.format = tex_map.format;
    out.aspect = tex_map.aspect;
    out.fallback_to_base = true;
    return out;
}

GpuTextureViewDimension to_view_dimension(uint32_t dim, bool* supported) noexcept {
    namespace V = d3d::ViewDimension;
    *supported = true;
    switch (dim) {
    case V::Unknown: return GpuTextureViewDimension::Undefined;
    case V::Texture1D: return GpuTextureViewDimension::Dim1D;
    case V::Texture2D:
    case V::Texture2DMs: return GpuTextureViewDimension::Dim2D;
    case V::Texture2DArray: return GpuTextureViewDimension::Dim2DArray;
    case V::Texture3D: return GpuTextureViewDimension::Dim3D;
    case V::TextureCube: return GpuTextureViewDimension::Cube;
    case V::TextureCubeArray: return GpuTextureViewDimension::CubeArray;
    default:  // 1D arrays and multisampled arrays have no WebGPU view dimension
        *supported = false;
        return GpuTextureViewDimension::Undefined;
    }
}

GpuCreateSampler2Payload to_sampler_payload(const d3d::SamplerDesc& desc, bool* exact) noexcept {
    bool ok = true;
    GpuCreateSampler2Payload p{};
    p.address_u = static_cast<uint8_t>(to_address_mode(desc.address_u, &ok));
    p.address_v = static_cast<uint8_t>(to_address_mode(desc.address_v, &ok));
    p.address_w = static_cast<uint8_t>(to_address_mode(desc.address_w, &ok));
    // D3D12_FILTER: mip = bits 0..1, mag = bits 2..3, min = bits 4..5, anisotropic = 0x40,
    // reduction = bits 7..8 (0 standard, 1 comparison, 2 minimum, 3 maximum).
    const uint32_t f = desc.filter;
    const bool anisotropic = (f & 0x40u) != 0;
    const uint32_t reduction = (f >> 7) & 0x3u;
    p.mipmap_filter = static_cast<uint8_t>(anisotropic ? 1u : (f & 0x1u));
    p.mag_filter = static_cast<uint8_t>(anisotropic ? 1u : ((f >> 2) & 0x1u));
    p.min_filter = static_cast<uint8_t>(anisotropic ? 1u : ((f >> 4) & 0x1u));
    p.max_anisotropy = static_cast<uint16_t>(anisotropic ? std::clamp<uint32_t>(desc.max_anisotropy, 1u, 16u) : 1u);
    if (reduction == 1) {
        const auto cmp = pso::map_compare_function(static_cast<pso::D3D12ComparisonFunc>(desc.comparison_func));
        p.compare = static_cast<uint8_t>(cmp ? *cmp : pso::WgpuCompareFunction::Never);
        ok = ok && cmp.has_value();
    } else if (reduction != 0) {
        ok = false;  // min/max reduction filters have no WebGPU equivalent
    }
    if (desc.mip_lod_bias != 0.0f) {
        ok = false;
    }
    const float lod_min = std::isfinite(desc.min_lod) ? std::clamp(desc.min_lod, 0.0f, 32.0f) : 0.0f;
    const float lod_max = std::isfinite(desc.max_lod) ? std::clamp(desc.max_lod, lod_min, 32.0f) : 32.0f;
    p.lod_min_clamp = lod_min == 0.0f ? 0.0f : lod_min;
    const float lod_max_out = std::max(p.lod_min_clamp, lod_max);
    p.lod_max_clamp = lod_max_out == 0.0f ? 0.0f : lod_max_out;
    if (exact != nullptr) {
        *exact = ok;
    }
    return p;
}

d3d::SamplerDesc sampler_for_binding(const d3d::SamplerDesc& desc, uint8_t layout_sampler_type, bool* changed) noexcept {
    d3d::SamplerDesc out = desc;
    const bool comparison = ((desc.filter >> 7) & 0x3u) == 1u;
    const bool drop = comparison && layout_sampler_type != static_cast<uint8_t>(GpuSamplerBindingType::Comparison);
    if (drop) {
        out.filter &= ~(0x3u << 7);
    }
    if (changed != nullptr) {
        *changed = drop;
    }
    return out;
}

// =====================================================================================================
// StateBridgeDevice
// =====================================================================================================

StateBridgeDevice::StateBridgeDevice(d3d::DescriptorHeapRegistry& heaps,
                                     BridgeOptions options,
                                     const d3d::ResourceStateRegistry* resource_states)
    : heaps_(heaps), options_(std::move(options)), resource_states_(resource_states) {
    if (!options_.id_allocator) {
        auto counter = std::make_shared<std::atomic<uint32_t>>(1u);
        options_.id_allocator = [counter] { return counter->fetch_add(1u, std::memory_order_relaxed); };
    }
    options_.root_constant_ring_bytes =
        std::max(align_up(options_.root_constant_ring_bytes, kRootConstantAlignment), kRootConstantAlignment);
    pso::PipelineCacheOptions cache_options;
    cache_options.id_allocator = options_.id_allocator;
    pipeline_cache_ = std::make_unique<pso::PipelineCache>(std::move(cache_options));
}

namespace {
std::atomic<uint64_t> g_live_views{0};
std::atomic<uint64_t> g_live_samplers{0};
std::atomic<uint64_t> g_live_command_lists{0};

void adjust_gauge(std::atomic<uint64_t>& gauge, size_t& tracked, size_t now) noexcept {
    if (now >= tracked) {
        gauge.fetch_add(now - tracked, std::memory_order_relaxed);
    } else {
        gauge.fetch_sub(tracked - now, std::memory_order_relaxed);
    }
    tracked = now;
}
}  // namespace

BridgeLiveCounts bridge_live_counts() noexcept {
    BridgeLiveCounts c;
    c.interned_views = g_live_views.load(std::memory_order_relaxed);
    c.samplers = g_live_samplers.load(std::memory_order_relaxed);
    c.cached_bind_groups = d3d::descriptor_live_counts().cached_bind_groups;
    c.command_lists = g_live_command_lists.load(std::memory_order_relaxed);
    return c;
}

StateBridgeDevice::~StateBridgeDevice() {
    rt_arena::clear_report_provider(this);
    adjust_gauge(g_live_views, views_tracked_, 0);
    adjust_gauge(g_live_samplers, samplers_tracked_, 0);
}

void StateBridgeDevice::register_texture(uint32_t resource_id, const TextureInfo& info) {
    bool replaced = false;
    uint32_t kept_floor = 0;
    uint32_t kept_offset = 0;
    {
        std::unique_lock lock(resources_mutex_);
        if (auto it = textures_.find(resource_id); it != textures_.end()) {
            const TextureInfo& old = it->second;
            replaced = old.format != info.format || old.width != info.width || old.height != info.height ||
                       old.depth_or_array_size != info.depth_or_array_size || old.mip_levels != info.mip_levels ||
                       old.dimension != info.dimension;
            kept_floor = replaced ? 0u : old.resident_mip_floor;
            kept_offset = replaced ? 0u : old.host_mip_offset;
            if (auto v = textures_by_va_.find(it->second.gpu_va);
                it->second.gpu_va != 0 && v != textures_by_va_.end() && v->second == resource_id) {
                textures_by_va_.erase(v);
            }
        }
        TextureInfo& stored = textures_[resource_id];
        stored = info;
        if (stored.resident_mip_floor == 0) {
            stored.resident_mip_floor = kept_floor;  // the same texture registered again keeps its held levels
        }
        if (stored.host_mip_offset == 0) {
            stored.host_mip_offset = kept_offset;
        }
        retired_resources_.erase(resource_id);
        if (info.gpu_va != 0) {
            textures_by_va_[info.gpu_va] = resource_id;
        }
    }
    if (replaced) {
        {
            std::lock_guard vlock(views_mutex_);
            depth_textures_.erase(resource_id);
        }
        const std::vector<uint32_t> retired_intermediates =
            retire_intermediates(resource_id);  // a new description: intermediates of the old one are stale
        if (!retired_intermediates.empty()) {
            notify_bridges_invalidated(retired_intermediates, {});
        }
    }
}

bool StateBridgeDevice::retired(uint32_t resource_id) const {
    std::shared_lock lock(resources_mutex_);
    return retired_resources_.count(resource_id) != 0;
}

uint32_t StateBridgeDevice::texture_at(uint64_t gpu_va) const {
    std::shared_lock lock(resources_mutex_);
    auto it = textures_by_va_.find(gpu_va);
    return it != textures_by_va_.end() ? it->second : 0u;
}

void StateBridgeDevice::register_buffer(uint32_t resource_id, uint64_t gpu_va, uint64_t byte_size,
                                        uint64_t allocated_size) {
    std::unique_lock lock(resources_mutex_);
    if (auto it = buffer_va_.find(resource_id); it != buffer_va_.end()) {
        buffers_by_va_.erase(it->second);
    }
    buffers_by_va_[gpu_va] = {resource_id, byte_size};
    retired_resources_.erase(resource_id);
    buffer_va_[resource_id] = gpu_va;
    buffer_allocated_[resource_id] = std::max(byte_size, allocated_size);
}

uint64_t StateBridgeDevice::buffer_allocated_size(uint32_t resource_id) const {
    std::shared_lock lock(resources_mutex_);
    auto it = buffer_allocated_.find(resource_id);
    return it != buffer_allocated_.end() ? it->second : 0;
}

uint64_t StateBridgeDevice::buffer_gpu_va(uint32_t resource_id) const {
    std::shared_lock lock(resources_mutex_);
    auto it = buffer_va_.find(resource_id);
    return it != buffer_va_.end() ? it->second : 0;
}

void StateBridgeDevice::unregister_resource(uint32_t resource_id) {
    unregister_resources(std::span<const uint32_t>(&resource_id, 1));
}

void StateBridgeDevice::unregister_resources(std::span<const uint32_t> resource_ids) {
    std::vector<uint32_t> invalidated_resources;
    std::vector<uint32_t> invalidated_views;
    invalidated_resources.reserve(resource_ids.size());
    for (const uint32_t id : resource_ids) {
        unregister_resource_quiet(id, &invalidated_resources, &invalidated_views);
    }
    notify_bridges_invalidated(invalidated_resources, invalidated_views);
}

void StateBridgeDevice::unregister_resource_quiet(uint32_t resource_id, std::vector<uint32_t>* invalidated_resources,
                                                  std::vector<uint32_t>* invalidated_views) {
    if (resource_id == 0) {
        return;
    }
    {
        std::unique_lock lock(resources_mutex_);
        bool known = false;
        if (auto it = textures_.find(resource_id); it != textures_.end()) {
            if (auto v = textures_by_va_.find(it->second.gpu_va);
                v != textures_by_va_.end() && v->second == resource_id) {
                textures_by_va_.erase(v);
            }
            textures_.erase(it);
            known = true;
        }
        if (auto it = buffer_va_.find(resource_id); it != buffer_va_.end()) {
            buffers_by_va_.erase(it->second);
            buffer_va_.erase(it);
            known = true;
        }
        buffer_allocated_.erase(resource_id);
        if (known && resource_id != 0) {
            retired_resources_.insert(resource_id);
            retire_epoch_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    {
        std::lock_guard lock(retire_mutex_);
        resource_last_use_.erase(resource_id);
    }
    const std::vector<uint32_t> retired_views = retire_views(resource_id);
    const std::vector<uint32_t> retired_intermediates = retire_intermediates(resource_id);
    {
        // G66: ExecuteIndirect clamp bind groups over the buffer.
        std::vector<uint32_t> stale;
        {
            std::lock_guard lock(objects_mutex_);
            for (auto it = indirect_groups_.begin(); it != indirect_groups_.end();) {
                if (it->first.first == resource_id || it->first.second == resource_id) {
                    stale.push_back(it->second);
                    it = indirect_groups_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (const uint32_t id : stale) {
            defer_destroy_resource(id);
        }
    }
    invalidated_resources->push_back(resource_id);
    invalidated_resources->insert(invalidated_resources->end(), retired_intermediates.begin(),
                                  retired_intermediates.end());
    invalidated_views->insert(invalidated_views->end(), retired_views.begin(), retired_views.end());
}

void StateBridgeDevice::attach_bridge(CommandListBridge* bridge) {
    if (bridge == nullptr) {
        return;
    }
    std::lock_guard lock(bridges_mutex_);
    bridges_.push_back(bridge);
}

void StateBridgeDevice::detach_bridge(CommandListBridge* bridge) {
    std::lock_guard lock(bridges_mutex_);
    bridges_.erase(std::remove(bridges_.begin(), bridges_.end(), bridge), bridges_.end());
}

void StateBridgeDevice::notify_bridges_invalidated(std::span<const uint32_t> resource_ids,
                                                   std::span<const uint32_t> view_ids) {
    if (resource_ids.empty() && view_ids.empty()) {
        return;
    }
    std::vector<uint32_t> retired_bgs;
    {
        std::lock_guard lock(bridges_mutex_);
        for (CommandListBridge* b : bridges_) {
            if (!b->try_invalidate_immediately(resource_ids, view_ids, &retired_bgs)) {
                b->queue_invalidation(resource_ids, view_ids);
            }
        }
    }
    for (const uint32_t bg_id : retired_bgs) {
        defer_destroy_resource(bg_id);
    }
}

void StateBridgeDevice::note_resource_used(uint32_t resource_id) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard lock(retire_mutex_);
    frame_used_resources_.insert(resource_id);
    resource_last_use_[resource_id] = ++resource_use_tick_;
}

void StateBridgeDevice::note_encoder_resource_used(const CommandEncoder& encoder, uint32_t resource_id) {
    if (resource_id == 0) {
        return;
    }
    {
        std::lock_guard lock(bridges_mutex_);
        for (CommandListBridge* b : bridges_) {
            if (&b->encoder() == &encoder) {
                b->note_recorded_resource(resource_id);
                return;
            }
        }
    }
    note_resource_used(resource_id);
}

bool StateBridgeDevice::is_resource_used_in_frame(uint32_t resource_id) const {
    if (resource_id == 0) {
        return false;
    }
    std::lock_guard lock(retire_mutex_);
    return frame_used_resources_.count(resource_id) != 0;
}

bool StateBridgeDevice::is_resource_recorded(uint32_t resource_id) const {
    if (resource_id == 0) {
        return false;
    }
    std::lock_guard lock(bridges_mutex_);
    for (const CommandListBridge* b : bridges_) {
        if (b->has_recorded_resource(resource_id)) {
            return true;
        }
    }
    return false;
}

uint64_t StateBridgeDevice::resource_last_use_tick(uint32_t resource_id) const {
    if (resource_id == 0) {
        return 0;
    }
    std::lock_guard lock(retire_mutex_);
    const auto it = resource_last_use_.find(resource_id);
    return it != resource_last_use_.end() ? it->second : 0u;
}

bool StateBridgeDevice::destroy_or_defer_resource(uint32_t resource_id, CommandEncoder& encoder) {
    if (resource_id == 0) {
        return false;
    }
    {
        std::lock_guard lock(retire_mutex_);
        if (frame_used_resources_.count(resource_id) != 0) {
            if (deferred_destroy_set_.insert(resource_id).second) {
                deferred_destroys_.push_back(resource_id);
            }
            return true;
        }
    }
    encoder.destroy_resource(resource_id);
    return false;
}

void StateBridgeDevice::defer_destroy_resource(uint32_t resource_id) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard lock(retire_mutex_);
    if (deferred_destroy_set_.insert(resource_id).second) {
        deferred_destroys_.push_back(resource_id);
    }
}

bool StateBridgeDevice::take_deferred_destroy(uint32_t resource_id, CommandEncoder& encoder) {
    if (resource_id == 0) {
        return false;
    }
    bool found = false;
    {
        std::lock_guard lock(retire_mutex_);
        if (deferred_destroy_set_.erase(resource_id) != 0) {
            deferred_destroys_.erase(
                std::remove(deferred_destroys_.begin(), deferred_destroys_.end(), resource_id),
                deferred_destroys_.end());
            found = true;
        }
    }
    if (found) {
        encoder.destroy_resource(resource_id);
    }
    return found;
}

size_t StateBridgeDevice::flush_deferred_destroys_impl(CommandEncoder& encoder, bool respect_unexecuted) {
    std::unordered_set<uint32_t> pinned;
    if (respect_unexecuted) {
        std::lock_guard lock(bridges_mutex_);
        for (const CommandListBridge* b : bridges_) {
            b->collect_unexecuted_resources(&pinned);
        }
    }
    std::vector<uint32_t> to_destroy;
    {
        std::lock_guard lock(objects_mutex_);
        if (!retired_ids_.empty()) {
            std::lock_guard rlock(retire_mutex_);
            for (const uint32_t id : retired_ids_) {
                if (id != 0 && deferred_destroy_set_.insert(id).second) {
                    deferred_destroys_.push_back(id);
                }
            }
            retired_ids_.clear();
        }
    }
    {
        std::lock_guard lock(retire_mutex_);
        if (pinned.empty()) {
            to_destroy.swap(deferred_destroys_);
            deferred_destroy_set_.clear();
            frame_used_resources_.clear();
        } else {
            std::vector<uint32_t> still_deferred;
            still_deferred.reserve(deferred_destroys_.size());
            for (const uint32_t id : deferred_destroys_) {
                if (pinned.count(id) != 0) {
                    still_deferred.push_back(id);
                } else {
                    deferred_destroy_set_.erase(id);
                    to_destroy.push_back(id);
                }
            }
            deferred_destroys_.swap(still_deferred);
            frame_used_resources_ = std::move(pinned);
        }
    }
    for (const uint32_t id : to_destroy) {
        encoder.destroy_resource(id);
    }
    return to_destroy.size();
}

size_t StateBridgeDevice::flush_deferred_destroys(CommandEncoder& encoder) {
    return flush_deferred_destroys_impl(encoder, false);
}

void StateBridgeDevice::present(CommandEncoder& encoder) {
    encoder.present();
    flush_deferred_destroys_impl(encoder, false);
}

bool StateBridgeDevice::advance_present_frame(uint64_t present_count, CommandEncoder& encoder) {
    bool advanced = false;
    {
        std::lock_guard lock(retire_mutex_);
        if (present_count > synced_present_count_) {
            synced_present_count_ = present_count;
            advanced = true;
        }
    }
    if (!advanced) {
        return false;
    }
    flush_deferred_destroys_impl(encoder, true);
    return true;
}

size_t StateBridgeDevice::deferred_destroy_count() const {
    std::lock_guard lock(retire_mutex_);
    return deferred_destroys_.size();
}

void StateBridgeDevice::register_shader_module(uint64_t shader_identity, uint32_t module_id) {
    std::unique_lock lock(resources_mutex_);
    shader_modules_[shader_identity] = module_id;
}

bool StateBridgeDevice::set_resident_mip_floor(uint32_t resource_id, uint32_t first_resident_mip) {
    std::unique_lock lock(resources_mutex_);
    const auto it = textures_.find(resource_id);
    if (it == textures_.end()) {
        return false;
    }
    const uint32_t floor = std::min(first_resident_mip, std::max(it->second.mip_levels, 1u) - 1u);
    if (it->second.resident_mip_floor != floor) {
        it->second.resident_mip_floor = floor;
        residency_epoch_.fetch_add(1, std::memory_order_acq_rel);
    }
    return true;
}

bool StateBridgeDevice::set_host_mip_offset(uint32_t resource_id, uint32_t offset) {
    {
        std::unique_lock lock(resources_mutex_);
        const auto it = textures_.find(resource_id);
        if (it == textures_.end()) {
            return false;
        }
        const uint32_t off = std::min(offset, std::max(it->second.mip_levels, 1u) - 1u);
        if (it->second.host_mip_offset == off) {
            return true;
        }
        it->second.host_mip_offset = off;
        residency_epoch_.fetch_add(1, std::memory_order_acq_rel);
    }
    const std::vector<uint32_t> retired_views = retire_views(resource_id);
    const uint32_t ids[1] = {resource_id};
    notify_bridges_invalidated(ids, retired_views);
    return true;
}

bool StateBridgeDevice::translate_host_mip(uint32_t resource_id, uint32_t* mip) const {
    std::shared_lock lock(resources_mutex_);
    const auto it = textures_.find(resource_id);
    if (it == textures_.end() || it->second.host_mip_offset == 0) {
        return true;
    }
    if (*mip < it->second.host_mip_offset) {
        return false;
    }
    *mip -= it->second.host_mip_offset;
    return true;
}

void shift_view_to_host_mips(uint32_t levels, uint32_t offset, uint16_t* base, uint16_t* count) noexcept {
    levels = std::max(levels, 1u);
    if (offset == 0) {
        return;
    }
    offset = std::min(offset, levels - 1u);
    const uint32_t b = *base;
    const uint32_t end = *count == 0 ? levels : std::min(levels, b + *count);
    const uint32_t nb = std::min(std::max(b, offset), levels - 1u);
    *base = static_cast<uint16_t>(nb - offset);
    if (*count != 0) {
        *count = static_cast<uint16_t>(std::max(end, nb + 1u) - nb);
    }
}

bool clamp_view_to_resident_mips(uint32_t mip_levels, uint32_t resident_floor, float min_lod_clamp, uint16_t* base,
                                 uint16_t* count) noexcept {
    const uint32_t levels = std::max(mip_levels, 1u);
    uint32_t floor = resident_floor;
    if (min_lod_clamp > 0.0f) {  // false for NaN
        floor = std::max(floor, static_cast<uint32_t>(std::floor(std::min(min_lod_clamp, 65535.0f))));
    }
    const uint32_t b = *base;
    if (floor <= b || b >= levels) {
        return false;
    }
    const uint32_t end = *count == 0 ? levels : std::min(levels, b + *count);  // one past the view's last level
    const uint32_t raised = std::min(floor, end - 1u);
    if (raised <= b) {
        return false;
    }
    *base = static_cast<uint16_t>(raised);
    if (*count != 0) {
        *count = static_cast<uint16_t>(end - raised);
    }
    return true;
}

uint32_t held_mip_floor(std::span<const uint64_t> subresource_marks, uint32_t mips, uint32_t layers,
                        bool streamable) noexcept {
    const auto held = [&](uint32_t mip) {
        for (uint32_t slice = 0; slice < layers; ++slice) {
            const size_t index = mip + size_t{slice} * mips;
            if (index < subresource_marks.size() && subresource_marks[index] != 0) {
                return true;
            }
        }
        return false;
    };
    if (!streamable || mips <= 1u || layers == 0u) {
        return 0;
    }
    uint32_t floor = mips;
    while (floor > 0u && held(floor - 1u)) {
        --floor;
    }
    return floor == mips ? 0u : floor;
}

bool StateBridgeDevice::texture_info(uint32_t resource_id, TextureInfo* out) const {
    std::shared_lock lock(resources_mutex_);
    const auto it = textures_.find(resource_id);
    if (it == textures_.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

uint32_t StateBridgeDevice::module_id(uint64_t shader_identity, bool* known) const {
    if (shader_identity == 0) {
        *known = true;
        return 0;
    }
    std::shared_lock lock(resources_mutex_);
    const auto it = shader_modules_.find(shader_identity);
    *known = it != shader_modules_.end();
    return *known ? it->second : 0u;
}

void StateBridgeDevice::register_pipeline_layout(uint64_t first_shader, uint64_t second_shader, PipelineLayoutDesc desc) {
    std::lock_guard lock(objects_mutex_);
    PipelineLayoutRecord* layout = intern_pipeline_layout_locked(desc);
    if (layout != nullptr) {
        layouts_by_shaders_[ShaderPairKey{first_shader, second_shader}] = layout;
    }
}

StateBridgeDevice::PipelineLayoutRecord* StateBridgeDevice::intern_pipeline_layout_locked(PipelineLayoutDesc& desc) {
    // Binding arrays (G62) only for a consumer that binds them: otherwise the layout is refused
    // (no layout registered: the pipeline falls back like an unknown one).
    for (const GroupLayoutDesc& g : desc.groups) {
        for (const LayoutBinding& b : g.bindings) {
            if (b.entry.binding_array_size != 0 &&
                (!options_.binding_arrays || b.entry.binding_array_size > kGpuMaxBindingArraySize)) {
                return nullptr;
            }
        }
    }
    std::vector<GroupLayoutRecord*> groups;
    groups.reserve(desc.groups.size() + (desc.rt_arena ? 4u : 0u));
    for (GroupLayoutDesc& g : desc.groups) {
        std::stable_sort(g.bindings.begin(), g.bindings.end(), [](const LayoutBinding& a, const LayoutBinding& b) {
            return a.entry.binding < b.entry.binding;
        });
        auto record = std::make_unique<GroupLayoutRecord>();
        record->layout_entries.reserve(g.bindings.size());
        record->entries.reserve(g.bindings.size());
        record->joins.reserve(g.bindings.size());
        bool modified = false;
        for (LayoutBinding b : g.bindings) {
            b.join.binding = b.entry.binding;
            // Root constants are uploaded to the shared ring and bound with a dynamic offset.
            if (b.join.root_constants && b.entry.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
                (b.entry.flags & GpuBindGroupLayoutEntryFlags::HasDynamicOffset) == 0) {
                b.entry.flags |= GpuBindGroupLayoutEntryFlags::HasDynamicOffset;
                modified = true;
            }
            record->layout_entries.push_back(b.entry);
            if (b.entry.binding_array_size == 0) {
                b.join.array_size = 0;
                b.join.array_index = 0;
                record->entries.push_back(b.entry);
                record->joins.push_back(b.join);
                continue;
            }
            // Binding array: one bind group entry (and join) per element.
            record->header.flags |= GpuLayoutFlags::BindingArrays;
            b.join.array_size = b.entry.binding_array_size;
            for (uint32_t k = 0; k < b.entry.binding_array_size; ++k) {
                b.join.array_index = k;
                record->entries.push_back(b.entry);
                record->joins.push_back(b.join);
            }
        }
        // A layout that differs from the manifest's no longer carries the manifest id.
        record->header.content_id = modified ? 0 : g.content_id;
        record->header.entry_count = static_cast<uint32_t>(record->layout_entries.size());

        uint64_t ghash = hash_combine64(splitmix64(record->header.content_id), record->header.flags);
        ghash = hash_combine64(ghash, record->layout_entries.size());
        if (!record->layout_entries.empty()) {
            ghash = hash_combine64(
                ghash, fnv1a_hash64(record->layout_entries.data(),
                                    record->layout_entries.size() * sizeof(GpuBindGroupLayoutEntry)));
        }
        for (const d3d::ShaderBinding& j : record->joins) {
            ghash = hash_shader_binding(ghash, j);
        }
        record->content_hash = ghash;

        GroupLayoutRecord* found = nullptr;
        if (const auto bit = group_layouts_by_hash_.find(ghash); bit != group_layouts_by_hash_.end()) {
            for (GroupLayoutRecord* existing : bit->second) {
                if (!existing->runtime_owned && existing->header.content_id == record->header.content_id &&
                    existing->header.flags == record->header.flags &&
                    existing->entries.size() == record->entries.size() &&
                    existing->layout_entries.size() == record->layout_entries.size() &&
                    // memcmp with null pointers is undefined even for size 0 (both vectors empty).
                    (record->layout_entries.empty() ||
                     std::memcmp(existing->layout_entries.data(), record->layout_entries.data(),
                                 record->layout_entries.size() * sizeof(GpuBindGroupLayoutEntry)) == 0)) {
                    const bool same_joins = std::equal(
                        existing->joins.begin(), existing->joins.end(), record->joins.begin(),
                        [](const d3d::ShaderBinding& a, const d3d::ShaderBinding& b) { return d3d::same_join(a, b); });
                    if (same_joins) {
                        found = existing;
                        break;
                    }
                }
            }
        }
        if (found == nullptr) {
            record->id = allocate_id();
            record->header.id = record->id;
            found = record.get();
            group_layouts_by_hash_[ghash].push_back(found);
            group_layouts_.push_back(std::move(record));
        }
        groups.push_back(found);
    }

    if (desc.rt_arena) {
        if (groups.size() > rt_arena::kBindGroup) {
            return nullptr;  // the runtime owns group 3
        }
        while (groups.size() < rt_arena::kBindGroup) {
            if (empty_group_layout_ == nullptr) {
                auto record = std::make_unique<GroupLayoutRecord>();
                record->id = allocate_id();
                record->header.id = record->id;
                record->runtime_owned = false;
                record->content_hash = hash_combine64(splitmix64(0), 0);
                record->content_hash = hash_combine64(record->content_hash, 0);
                empty_group_layout_ = record.get();
                group_layouts_by_hash_[record->content_hash].push_back(empty_group_layout_);
                group_layouts_.push_back(std::move(record));
            }
            groups.push_back(empty_group_layout_);
        }
        if (rt_group_layout_ == nullptr) {
            auto record = std::make_unique<GroupLayoutRecord>();
            record->id = allocate_id();
            record->header.id = record->id;
            for (uint32_t b = rt_arena::kBindArena0; b <= rt_arena::kBindStats; ++b) {
                GpuBindGroupLayoutEntry entry{};
                entry.binding = b;
                entry.visibility = GpuShaderStageBits::Compute;
                entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
                entry.binding_type = static_cast<uint8_t>(b == rt_arena::kBindDirectory
                                                              ? GpuBufferBindingType::ReadOnlyStorage
                                                              : GpuBufferBindingType::Storage);
                record->entries.push_back(entry);
                record->layout_entries.push_back(entry);
                record->joins.push_back(d3d::ShaderBinding{});
            }
            record->header.entry_count = static_cast<uint32_t>(record->layout_entries.size());
            record->runtime_owned = true;
            rt_group_layout_ = record.get();
            group_layouts_.push_back(std::move(record));
        }
        groups.push_back(rt_group_layout_);
    }

    uint64_t pl_hash = hash_combine64(splitmix64(desc.content_id), desc.rt_arena ? 1ull : 0ull);
    for (const GroupLayoutRecord* grp : groups) {
        pl_hash = hash_combine64(pl_hash, grp != nullptr ? grp->id : 0u);
    }

    PipelineLayoutRecord* layout = nullptr;
    if (const auto pit = pipeline_layouts_by_hash_.find(pl_hash); pit != pipeline_layouts_by_hash_.end()) {
        for (PipelineLayoutRecord* existing : pit->second) {
            if (existing->groups == groups && existing->content_id == desc.content_id &&
                existing->rt_arena == desc.rt_arena) {
                layout = existing;
                break;
            }
        }
    }
    if (layout == nullptr) {
        auto record = std::make_unique<PipelineLayoutRecord>();
        record->id = allocate_id();
        record->content_id = desc.content_id;
        record->content_hash = pl_hash;
        record->groups = std::move(groups);
        record->rt_arena = desc.rt_arena;
        record->rt_address_sources = desc.rt_address_sources;
        layout = record.get();
        pipeline_layouts_by_hash_[pl_hash].push_back(layout);
        pipeline_layouts_.push_back(std::move(record));
    }
    return layout;
}

StateBridgeDevice::PipelineLayoutRecord* StateBridgeDevice::find_layout(uint64_t first_shader, uint64_t second_shader) {
    std::lock_guard lock(objects_mutex_);
    const auto it = layouts_by_shaders_.find(ShaderPairKey{first_shader, second_shader});
    return it == layouts_by_shaders_.end() ? nullptr : it->second;
}

void StateBridgeDevice::ensure_layout_emitted(PipelineLayoutRecord* layout, CommandEncoder& encoder) {
    if (layout == nullptr) {
        return;
    }
    prewarm_barrier();  // a layout the prewarm is emitting right now is complete (and submitted) once this returns
    ensure_layout_emitted_impl(layout, encoder, false);
}

void StateBridgeDevice::ensure_layout_emitted_impl(PipelineLayoutRecord* layout, CommandEncoder& encoder,
                                                   bool by_prewarm) {
    std::lock_guard lock(objects_mutex_);
    if (layout->emitted) {
        return;
    }
    std::array<uint32_t, 8> stack_ids{};
    std::vector<uint32_t> heap_ids;
    uint32_t* group_ids = stack_ids.data();
    const size_t n = layout->groups.size();
    if (n > stack_ids.size()) {
        heap_ids.resize(n);
        group_ids = heap_ids.data();
    }
    for (size_t i = 0; i < n; ++i) {
        GroupLayoutRecord* group = layout->groups[i];
        if (!group->emitted) {
            auto emit_group = [&](CommandEncoder& dst) {
                dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &group->header, sizeof(group->header),
                                group->layout_entries.data(),
                                static_cast<uint32_t>(group->layout_entries.size() * sizeof(GpuBindGroupLayoutEntry)));
            };
            if (by_prewarm) {
                emit_group(encoder);
            } else {
                emit_device_object(group->id, encoder, emit_group);
            }
            group->emitted = true;
            group->emitted_by_prewarm = by_prewarm;
        }
        group_ids[i] = group->id;
    }
    GpuCreatePipelineLayoutPayload header{};
    header.id = layout->id;
    header.bind_group_layout_count = static_cast<uint32_t>(n);
    header.content_id = layout->content_id;
    auto emit_pl = [&](CommandEncoder& dst) {
        dst.emit_packet(GpuOpcode::CreatePipelineLayout, &header, sizeof(header), group_ids,
                        static_cast<uint32_t>(n * sizeof(uint32_t)));
    };
    if (by_prewarm) {
        emit_pl(encoder);
    } else {
        emit_device_object(layout->id, encoder, emit_pl);
    }
    layout->emitted = true;
    layout->emitted_by_prewarm = by_prewarm;
}

StateBridgeDevice::SamplerKey StateBridgeDevice::make_sampler_key(const GpuCreateSampler2Payload& payload) noexcept {
    SamplerKey key{};
    key.address_u = payload.address_u;
    key.address_v = payload.address_v;
    key.address_w = payload.address_w;
    key.mag_filter = payload.mag_filter;
    key.min_filter = payload.min_filter;
    key.mipmap_filter = payload.mipmap_filter;
    key.compare = payload.compare;
    key.max_anisotropy = payload.max_anisotropy;
    key.lod_min_clamp_bits = canonical_finite_bits(payload.lod_min_clamp);
    key.lod_max_clamp_bits = canonical_finite_bits(payload.lod_max_clamp);
    return key;
}

uint32_t StateBridgeDevice::intern_sampler(const d3d::SamplerDesc& desc, CommandEncoder& encoder, uint64_t* diagnostics) {
    bool exact = true;
    GpuCreateSampler2Payload payload = to_sampler_payload(desc, &exact);
    if (!exact && diagnostics != nullptr) {
        *diagnostics |= BridgeDiag::SamplerApproximated;
    }
    const SamplerKey key = make_sampler_key(payload);
    std::lock_guard lock(objects_mutex_);
    if (const auto it = samplers_.find(key); it != samplers_.end()) {
        return it->second;
    }
    payload.id = allocate_id();
    emit_device_object(payload.id, encoder, [&](CommandEncoder& dst) {
        dst.emit_packet(GpuOpcode::CreateSampler2, &payload, sizeof(payload));
    });
    samplers_.emplace(key, payload.id);
    adjust_gauge(g_live_samplers, samplers_tracked_, samplers_.size());
    return payload.id;
}

uint32_t StateBridgeDevice::ensure_root_constant_ring(CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    if (ring_buffer_id_ == 0) {
        ring_buffer_id_ = allocate_id();
        emit_device_object(ring_buffer_id_, encoder, [&](CommandEncoder& dst) {
            // Storage: the ExecuteIndirect root-constant pass writes per-command slots (indirect_constants_wgsl).
            dst.create_buffer(ring_buffer_id_,
                              GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst | GpuBufferUsage::Storage,
                              options_.root_constant_ring_bytes);
        });
    }
    return ring_buffer_id_;
}

uint32_t StateBridgeDevice::rt_prepare_dispatch(const GroupLayoutRecord& group,
                                                const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                                                CommandEncoder& encoder,
                                                const std::vector<rt_residency::AddressSource>* sources,
                                                const rt_residency::DispatchInputs* inputs) {
    std::lock_guard lock(objects_mutex_);
    if (rt_arena_ == nullptr) {
        // Bookkeeping only (no GPU buffers yet): the counters work with RT off.
        rt_arena_ = std::make_unique<rt_arena::Arena>(options_.rt_arenas, options_.rt_pages_per_arena);
        rt_arena::set_report_provider(this, [this] { return rt_report_json(); });
    }
    rt_arena::Arena& arena = *rt_arena_;
    if (!options_.rt_enabled) {
        arena.note_skip(rt_arena::SkipReason::RtOff);
        return 0;
    }
    arena.begin_dispatch();
    if (sources != nullptr && !sources->empty() && inputs != nullptr) {
        // 1. Faults the GPU logged for earlier dispatches become resident now (uploaded below).
        if (rt_fault_issued_ != rt_fault_consumed_ && rt_fault_done_.load(std::memory_order_acquire) >= rt_fault_issued_) {
            rt_fault_consumed_ = rt_fault_issued_;
            if (rt_faults_.consume(rt_fault_words_.get(), rt_residency::kStatsWords, arena)) {
                rt_fault_zero_pending_ = true;  // words 1..31 are zeroed on the GPU before this dispatch
            }
        }
        // 2. Plan from the stage's address sources and ensure (a counted skip leaves nothing recorded).
        rt_residency::Plan plan;
        const rt_residency::Outcome out = rt_residency::prepare_dispatch(arena, *sources, *inputs, &plan);
        ++rt_plans_;
        rt_unresolved_ += plan.unresolved;
        for (size_t i = 0; i < static_cast<size_t>(rt_residency::Unresolved::Count); ++i) {
            rt_unresolved_by_reason_[i] += plan.unresolved_by_reason[i];
        }
        rt_tables_followed_ += plan.tables_followed;
        rt_table_buffers_ += plan.table_buffers;
        for (size_t i = 0; i < static_cast<size_t>(rt_residency::TableSkip::Count); ++i) {
            rt_table_skips_[i] += plan.table_skips[i];
        }
        rt_structures_ += plan.structures;
        rt_bad_headers_ += plan.bad_headers;
        if (!out.ok) {
            return 0;
        }
    }
    for (const auto& [addr, size] : ranges) {
        const rt_arena::Ensure e = arena.ensure_range(addr, size);
        if (e == rt_arena::Ensure::ArenaFull || e == rt_arena::Ensure::TableFull) {
            arena.note_skip(rt_arena::SkipReason::ArenaFull);
            return 0;
        }
        if (e == rt_arena::Ensure::OutOfRange) {
            arena.note_skip(rt_arena::SkipReason::NotResident);
            return 0;
        }
    }
    if (rt_dir_buffer_ == 0) {
        const uint64_t arena_bytes = static_cast<uint64_t>(options_.rt_pages_per_arena) * rt_arena::kPageBytes;
        const uint32_t arenas = std::min(std::max(options_.rt_arenas, 1u), rt_arena::kArenaCount);
        for (uint32_t i = 0; i < arenas; ++i) {  // unused arenas stay unbound (null bindings)
            rt_buffers_[i] = allocate_id();
            encoder.create_buffer(rt_buffers_[i], GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, arena_bytes);
        }
        rt_dir_buffer_ = allocate_id();
        encoder.create_buffer(rt_dir_buffer_, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst,
                              (static_cast<uint64_t>(rt_arena::kTopEntries) +
                               static_cast<uint64_t>(rt_arena::kMaxL2Tables) * rt_arena::kL2Entries) * 4u);
        rt_stats_buffer_ = allocate_id();
        encoder.create_buffer(rt_stats_buffer_,
                              GpuBufferUsage::Storage | GpuBufferUsage::CopyDst | GpuBufferUsage::CopySrc, 256);
    }
    if (rt_fault_words_ == nullptr) {
        rt_fault_words_ = std::make_unique<uint32_t[]>(rt_residency::kStatsWords);
    }
    if (rt_fault_zero_pending_) {
        static const uint32_t zeros[rt_residency::kStatsWords - 1] = {};
        encoder.write_buffer(rt_stats_buffer_, 4u, zeros, sizeof(zeros));
        rt_fault_zero_pending_ = false;
    }
    // Pages first, then the directory that points at them.
    static thread_local std::vector<uint8_t> page;
    page.resize(rt_arena::kPageBytes);
    for (const rt_arena::Upload& u : arena.take_uploads()) {
        const bool ok = options_.rt_page_reader &&
                        options_.rt_page_reader(u.page << rt_arena::kPageShift, page.data(), page.size());
        if (!ok) {
            std::fill(page.begin(), page.end(), uint8_t{0});
            arena.note_unreadable();
        }
        encoder.write_buffer(rt_buffers_[u.arena], static_cast<uint32_t>(rt_arena::slot_byte_offset(u.slot)),
                             page.data(), static_cast<uint32_t>(page.size()));
    }
    uint32_t first = 0, last = 0;
    if (arena.take_dir_dirty(&first, &last)) {
        encoder.write_buffer(rt_dir_buffer_, first * 4u, arena.directory().data() + first, (last - first + 1u) * 4u);
    }
    uint32_t& id = rt_groups_[group.id];
    if (id == 0) {
        id = allocate_id();
        GpuBindGroupEntry2 e[6]{};
        for (uint32_t i = 0; i < 6; ++i) {
            e[i].binding = rt_arena::kBindArena0 + i;
            e[i].resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
            e[i].resource_id = i < rt_arena::kArenaCount ? rt_buffers_[i]
                               : i == 4                   ? rt_dir_buffer_
                                                          : rt_stats_buffer_;
        }
        GpuCreateBindGroup2Payload g{};
        g.id = id;
        g.layout_id = group.id;
        g.entry_count = 6;
        encoder.emit_packet(GpuOpcode::CreateBindGroup2, &g, sizeof(g), e, sizeof(e));
    }
    arena.note_dispatch();
    return id;
}

void StateBridgeDevice::rt_mark_cpu_dirty(uint64_t addr, uint64_t size) {
    std::lock_guard lock(objects_mutex_);
    if (rt_arena_ == nullptr || size == 0) {
        return;
    }
    for (uint64_t p = rt_arena::page_of(addr); p <= rt_arena::page_of(addr + size - 1); ++p) {
        rt_arena_->mark_cpu_dirty(p);
    }
}

void StateBridgeDevice::rt_mark_cpu_remapped(uint64_t addr, uint64_t size) {
    std::lock_guard lock(objects_mutex_);
    if (rt_arena_ != nullptr) {
        rt_arena_->mark_cpu_remapped(addr, size);
    }
}

bool StateBridgeDevice::rt_fault_readback_begin(GpuReadbackBufferPayload* payload, uint32_t* stats_buffer_id) {
    std::lock_guard lock(objects_mutex_);
    if (!options_.rt_enabled || rt_stats_buffer_ == 0 || rt_fault_words_ == nullptr) {
        return false;
    }
    if (rt_fault_issued_ != rt_fault_consumed_) {
        return false;  // one in flight or landed but not consumed yet (the next prepare does): never queue
    }
    ++rt_fault_issued_;
    payload->seq = rt_fault_issued_;
    payload->buffer_id = rt_stats_buffer_;
    payload->src_offset = 0;
    payload->size = rt_residency::kStatsWords * 4u;
    payload->dst_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rt_fault_words_.get()));
    payload->done_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&rt_fault_done_));
    *stats_buffer_id = rt_stats_buffer_;
    return true;
}

bool StateBridgeDevice::rt_read_guest(uint64_t addr, void* dst, size_t bytes) const {
    return options_.rt_page_reader && options_.rt_page_reader(addr, dst, bytes);
}

std::string StateBridgeDevice::rt_report_json() {
    std::lock_guard lock(objects_mutex_);
    if (rt_arena_ == nullptr) {
        return "{}";
    }
    std::string out = rt_arena_->json();
    if (!out.empty() && out.back() == '}') {
        char buf[1024];
        std::snprintf(buf, sizeof buf,
                      ", \"residency\": {\"plans\": %llu, \"unresolved_sources\": %llu, "
                      "\"unresolved_by_reason\": {\"%s\": %llu, \"%s\": %llu, \"%s\": %llu, \"%s\": %llu, \"%s\": %llu}, "
                      "\"structures\": %llu, "
                      "\"tables_followed\": %llu, \"table_buffers\": %llu, "
                      "\"table_skips\": {\"%s\": %llu, \"%s\": %llu, \"%s\": %llu, \"%s\": %llu, \"%s\": %llu}, "
                      "\"bad_headers\": %llu, \"fault_readbacks_issued\": %u, \"fault\": ",
                      (unsigned long long)rt_plans_, (unsigned long long)rt_unresolved_,
                      rt_residency::unresolved_name(rt_residency::Unresolved::NoInput),
                      (unsigned long long)rt_unresolved_by_reason_[0],
                      rt_residency::unresolved_name(rt_residency::Unresolved::Unbound),
                      (unsigned long long)rt_unresolved_by_reason_[1],
                      rt_residency::unresolved_name(rt_residency::Unresolved::Null),
                      (unsigned long long)rt_unresolved_by_reason_[2],
                      rt_residency::unresolved_name(rt_residency::Unresolved::DescriptorTable),
                      (unsigned long long)rt_unresolved_by_reason_[3],
                      rt_residency::unresolved_name(rt_residency::Unresolved::OutOfRange),
                      (unsigned long long)rt_unresolved_by_reason_[4],
                      (unsigned long long)rt_structures_, (unsigned long long)rt_tables_followed_,
                      (unsigned long long)rt_table_buffers_,
                      rt_residency::table_skip_name(rt_residency::TableSkip::Zero), (unsigned long long)rt_table_skips_[0],
                      rt_residency::table_skip_name(rt_residency::TableSkip::NotBuffer),
                      (unsigned long long)rt_table_skips_[1],
                      rt_residency::table_skip_name(rt_residency::TableSkip::Unreadable),
                      (unsigned long long)rt_table_skips_[2],
                      rt_residency::table_skip_name(rt_residency::TableSkip::NullBase),
                      (unsigned long long)rt_table_skips_[3],
                      rt_residency::table_skip_name(rt_residency::TableSkip::Truncated),
                      (unsigned long long)rt_table_skips_[4],
                      (unsigned long long)rt_bad_headers_, rt_fault_issued_);
        out.pop_back();
        out += buf;
        out += rt_faults_.json();
        out += "}}";
    }
    return out;
}

uint32_t StateBridgeDevice::ensure_default_vertex_buffer(CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    if (default_vertex_buffer_id_ == 0) {
        default_vertex_buffer_id_ = allocate_id();
        emit_device_object(default_vertex_buffer_id_, encoder, [&](CommandEncoder& dst) {
            dst.create_buffer(default_vertex_buffer_id_, GpuBufferUsage::Vertex | GpuBufferUsage::CopyDst,
                              pso::kDefaultVertexBufferBytes);
            const float f[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            const uint32_t u[4] = {0, 0, 0, 1};
            dst.write_buffer(default_vertex_buffer_id_, 0, f, sizeof(f));
            dst.write_buffer(default_vertex_buffer_id_, pso::kDefaultVertexIntOffset, u, sizeof(u));
        });
    }
    return default_vertex_buffer_id_;
}

uint32_t StateBridgeDevice::allocate_root_constants(uint32_t bytes, bool* wrapped) {
    std::lock_guard lock(objects_mutex_);
    const uint32_t size = align_up(std::max(bytes, 1u), kRootConstantAlignment);
    *wrapped = false;
    if (ring_cursor_ + size > options_.root_constant_ring_bytes) {
        ring_cursor_ = 0;
        *wrapped = true;
    }
    const uint32_t offset = ring_cursor_;
    ring_cursor_ += size;
    return offset;
}

bool StateBridgeDevice::claim_compute_creation(uint32_t pipeline_id) {
    prewarm_barrier();
    std::lock_guard lock(objects_mutex_);
    return emitted_compute_.insert(pipeline_id).second;
}

// ---- Native pipeline prewarm ----

void StateBridgeDevice::set_module_pack_key_resolver(std::function<uint64_t(uint32_t)> resolver) {
    module_pack_key_ = std::move(resolver);
}

void StateBridgeDevice::prewarm_barrier() {
    if (prewarm_sync_.load(std::memory_order_acquire)) {
        std::lock_guard lock(prewarm_mutex_);
    }
}

void StateBridgeDevice::note_pipeline_reached(uint32_t pipeline_id) {
    if (!prewarm_sync_.load(std::memory_order_acquire)) {
        return;
    }
    uint32_t task = 0;
    {
        std::lock_guard lock(prewarm_mutex_);  // also waits for a build in progress
        const auto it = prewarm_pipelines_.find(pipeline_id);
        if (it == prewarm_pipelines_.end()) {
            return;
        }
        task = it->second;
        prewarm_pipelines_.erase(it);
    }
    pipeline_prewarm().note_task_used(task);
}

bool StateBridgeDevice::describe_layout(const PipelineLayoutRecord* layout, PrewarmPipelineDesc* desc) {
    if (layout == nullptr) {
        desc->has_layout = false;
        return true;
    }
    std::lock_guard lock(objects_mutex_);
    desc->has_layout = true;
    desc->layout.content_id = layout->content_id;
    for (const GroupLayoutRecord* g : layout->groups) {
        if (g->runtime_owned) {
            return false;  // bound by the runtime itself (mesh output): not replayable from a warm list
        }
        GroupLayoutDesc gd;
        gd.content_id = g->header.content_id;
        size_t join = 0;
        for (const GpuBindGroupLayoutEntry& e : g->layout_entries) {
            if (join >= g->joins.size()) {
                return false;
            }
            gd.bindings.push_back(LayoutBinding{e, g->joins[join]});
            join += e.binding_array_size != 0 ? e.binding_array_size : 1;  // arrays hold one join per element
        }
        desc->layout.groups.push_back(std::move(gd));
    }
    return true;
}

void StateBridgeDevice::record_graphics_pipeline(const pso::GraphicsPipelineKey& key, uint32_t vs_module,
                                                 uint32_t fs_module, const PipelineLayoutRecord* layout) {
    if (!module_pack_key_) {
        return;
    }
    PrewarmPipelineDesc d;
    d.kind = kPrewarmKindGraphics;
    d.pack_key0 = module_pack_key_(vs_module);
    d.pack_key1 = fs_module != 0 ? module_pack_key_(fs_module) : 0;
    if (d.pack_key0 == 0 || (fs_module != 0 && d.pack_key1 == 0) || !describe_layout(layout, &d)) {
        return;  // a stage that is not a pack module cannot be rebuilt from the warm list
    }
    d.graphics = key;
    pipeline_prewarm().note_title_pipeline_full(d.kind, d.pack_key0, d.pack_key1, encode_prewarm_pipeline_desc(d));
}

void StateBridgeDevice::record_compute_pipeline(const pso::ComputePipelineKey& key, uint32_t cs_module,
                                                const PipelineLayoutRecord* layout) {
    if (!module_pack_key_) {
        return;
    }
    PrewarmPipelineDesc d;
    d.kind = kPrewarmKindCompute;
    d.pack_key0 = module_pack_key_(cs_module);
    if (d.pack_key0 == 0 || !describe_layout(layout, &d)) {
        return;
    }
    d.compute = key;
    pipeline_prewarm().note_title_pipeline_full(d.kind, d.pack_key0, 0, encode_prewarm_pipeline_desc(d));
}

PrewarmPipelineResult StateBridgeDevice::prewarm_pipeline(const PrewarmPipelineDesc& desc, uint32_t vs_module_id,
                                                          uint32_t fs_module_id, uint32_t task_id,
                                                          const std::function<void(CommandEncoder&)>& submit) {
    const bool compute = desc.kind == kPrewarmKindCompute;
    if (vs_module_id == 0 || (!compute && desc.pack_key1 != 0 && fs_module_id == 0)) {
        return PrewarmPipelineResult::Failed;
    }
    // Held from the pipeline cache lookup to the submit of the creation packets: a title request that finds
    // the pipeline (or its layout) already there waits here first (prewarm_barrier / note_pipeline_reached).
    std::lock_guard prewarm_lock(prewarm_mutex_);
    PipelineLayoutRecord* layout = nullptr;
    if (desc.has_layout) {
        PipelineLayoutDesc copy = desc.layout;
        std::lock_guard lock(objects_mutex_);
        layout = intern_pipeline_layout_locked(copy);
        if (layout == nullptr) {
            return PrewarmPipelineResult::Failed;
        }
        // A layout (or one of its groups) a title list already emitted may still sit unsubmitted in that
        // list: a pipeline creation submitted now could reach the worker before it. The title got there
        // first anyway.
        if ((layout->emitted && !layout->emitted_by_prewarm) ||
            std::any_of(layout->groups.begin(), layout->groups.end(),
                        [](const GroupLayoutRecord* g) { return g->emitted && !g->emitted_by_prewarm; })) {
            return PrewarmPipelineResult::Lost;
        }
    }
    uint32_t pipeline_id = 0;
    if (compute) {
        pipeline_id = pipeline_cache_->acquire(desc.compute).pipeline_id;
        std::lock_guard lock(objects_mutex_);
        if (pipeline_id == 0 || !emitted_compute_.insert(pipeline_id).second) {
            return pipeline_id == 0 ? PrewarmPipelineResult::Failed : PrewarmPipelineResult::Lost;
        }
    } else {
        const pso::CacheLookup lookup = pipeline_cache_->acquire(desc.graphics);
        if (!lookup.created) {
            pipeline_cache_->release(desc.graphics);  // the title (or an earlier build) created it
            return PrewarmPipelineResult::Lost;
        }
        pipeline_id = lookup.pipeline_id;
    }
    CommandEncoder enc;
    if (layout != nullptr) {
        ensure_layout_emitted_impl(layout, enc, true);
    }
    const uint32_t layout_id = layout != nullptr ? layout->id : 0;
    if (compute) {
        GpuCreateComputePipeline2Payload payload{};
        payload.id = pipeline_id;
        payload.cs_module_id = vs_module_id;
        payload.pipeline_layout_id = layout_id;
        std::memcpy(&payload.key, &desc.compute, sizeof(payload.key));
        enc.emit_packet(GpuOpcode::CreateComputePipeline2, &payload, sizeof(payload));
    } else {
        GpuCreateRenderPipeline2Payload payload{};
        payload.id = pipeline_id;
        payload.vs_module_id = vs_module_id;
        payload.fs_module_id = fs_module_id;
        payload.pipeline_layout_id = layout_id;
        std::memcpy(&payload.key, &desc.graphics, sizeof(payload.key));
        sanitize_render_pipeline_key(&payload.key, options_.features);
        enc.emit_packet(GpuOpcode::CreateRenderPipeline2, &payload, sizeof(payload));
    }
    submit(enc);
    prewarm_pipelines_[pipeline_id] = task_id;
    return PrewarmPipelineResult::Built;
}

// ---- #482: storage views through an intermediate texture of the stage's storage format ----

namespace {

// Float/normalized formats WGSL can declare as a write-only storage texture under the enabled
// WebGPU device features: the conversion pass loads one as unfilterable-float and stores the other.
bool convertible_storage_format(uint8_t code, uint32_t features) noexcept {
    using F = pso::WgpuTextureFormat;
    switch (static_cast<F>(code)) {
    case F::Rgba8Unorm:
    case F::Rgba8Snorm:
    case F::Rgba16Float:
    case F::R32Float:
    case F::Rg32Float:
    case F::Rgba32Float:
        return true;
    case F::Bgra8Unorm:
        return (features & pso::WgpuFeature::Bgra8UnormStorage) != 0;
    case F::R8Unorm:
    case F::R8Snorm:
    case F::Rg8Unorm:
    case F::Rg8Snorm:
    case F::R16Unorm:
    case F::R16Snorm:
    case F::R16Float:
    case F::Rg16Unorm:
    case F::Rg16Snorm:
    case F::Rg16Float:
    case F::Rgba16Unorm:
    case F::Rgba16Snorm:
    case F::Rgb10a2Unorm:
    case F::Rg11b10Ufloat:
        return (features & pso::WgpuFeature::TextureFormatsTier1) != 0;
    default:
        return false;
    }
}

// Renderable float/color formats that can receive UAV copy-out via a fullscreen render pass when
// the host WebGPU device lacks STORAGE_BINDING for the format (e.g. without texture-formats-tier1
// or bgra8unorm-storage).
bool renderable_color_storage_fallback(uint8_t code, uint32_t features) noexcept {
    using F = pso::WgpuTextureFormat;
    switch (static_cast<F>(code)) {
    case F::Rgba8Unorm:
    case F::Rgba8UnormSrgb:
    case F::Bgra8Unorm:
    case F::Bgra8UnormSrgb:
    case F::R8Unorm:
    case F::Rg8Unorm:
    case F::R16Float:
    case F::Rg16Float:
    case F::Rgba16Float:
    case F::R32Float:
    case F::Rg32Float:
    case F::Rgba32Float:
    case F::Rgb10a2Unorm:
        return true;
    case F::Rg11b10Ufloat:
        return (features & pso::WgpuFeature::Rg11b10UfloatRenderable) != 0;
    default:
        return false;
    }
}

// Non-quantizing core WebGPU storage format chosen for a redirected intermediate when the layout
// does not pin a specific storage format (preserves HDR/float range instead of clamping to 8-bit).
uint8_t preferred_storage_intermediate_format(uint8_t own_format) noexcept {
    using F = pso::WgpuTextureFormat;
    switch (static_cast<F>(own_format)) {
    case F::R32Float:
    case F::Rg32Float:
    case F::Rgba32Float:
        return static_cast<uint8_t>(F::Rgba32Float);
    case F::Rgba16Float:
    case F::R16Float:
    case F::Rg16Float:
    case F::Rg11b10Ufloat:
    case F::Rgb10a2Unorm:
    case F::R16Unorm:
    case F::R16Snorm:
    case F::Rg16Unorm:
    case F::Rg16Snorm:
    case F::Rgba16Unorm:
    case F::Rgba16Snorm:
        return static_cast<uint8_t>(F::Rgba16Float);
    case F::Rgba8Snorm:
    case F::R8Snorm:
    case F::Rg8Snorm:
        return static_cast<uint8_t>(F::Rgba8Snorm);
    default:
        return static_cast<uint8_t>(F::Rgba8Unorm);
    }
}

std::string storage_render_out_wgsl() {
    return "@group(0) @binding(0) var src: texture_2d<f32>;\n"
           "struct VsOut { @builtin(position) pos: vec4<f32> };\n"
           "@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {\n"
           "  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\n"
           "  return VsOut(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0));\n"
           "}\n"
           "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n"
           "  return textureLoad(src, vec2<i32>(in.pos.xy), 0);\n"
           "}\n";
}

std::string storage_conversion_wgsl(uint8_t dst_format, uint8_t dimension) {
    const char* format = pso::texture_format_name(static_cast<pso::WgpuTextureFormat>(dst_format));
    std::string s;
    switch (static_cast<GpuTextureViewDimension>(dimension)) {
    case GpuTextureViewDimension::Dim3D:
        s = "@group(0) @binding(0) var src: texture_3d<f32>;\n"
            "@group(0) @binding(1) var dst: texture_storage_3d<FMT, write>;\n"
            "@compute @workgroup_size(4, 4, 4)\n"
            "fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
            "  if (any(id >= textureDimensions(dst))) { return; }\n"
            "  textureStore(dst, id, textureLoad(src, id, 0));\n"
            "}\n";
        break;
    case GpuTextureViewDimension::Dim2DArray:
        s = "@group(0) @binding(0) var src: texture_2d_array<f32>;\n"
            "@group(0) @binding(1) var dst: texture_storage_2d_array<FMT, write>;\n"
            "@compute @workgroup_size(8, 8, 1)\n"
            "fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
            "  if (any(id.xy >= textureDimensions(dst)) || id.z >= textureNumLayers(dst)) { return; }\n"
            "  textureStore(dst, id.xy, id.z, textureLoad(src, id.xy, id.z, 0));\n"
            "}\n";
        break;
    default:
        s = "@group(0) @binding(0) var src: texture_2d<f32>;\n"
            "@group(0) @binding(1) var dst: texture_storage_2d<FMT, write>;\n"
            "@compute @workgroup_size(8, 8, 1)\n"
            "fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
            "  if (any(id.xy >= textureDimensions(dst))) { return; }\n"
            "  textureStore(dst, id.xy, textureLoad(src, id.xy, 0));\n"
            "}\n";
        break;
    }
    const size_t at = s.find("FMT");
    s.replace(at, 3, format != nullptr ? format : "rgba8unorm");
    return s;
}

}  // namespace

void StateBridgeDevice::flush_retired_locked(CommandEncoder& encoder) {
    for (const uint32_t id : retired_ids_) {
        destroy_or_defer_resource(id, encoder);
    }
    retired_ids_.clear();
}

std::vector<uint32_t> StateBridgeDevice::retire_intermediates(uint32_t resource_id) {
    std::vector<uint32_t> retired_intermediates;
    std::vector<uint32_t> to_defer;
    {
        std::lock_guard lock(objects_mutex_);
        std::vector<uint32_t> gone{resource_id};
        for (auto it = intermediates_.begin(); it != intermediates_.end();) {
            if (it->first.resource_id == resource_id) {
                gone.push_back(it->second);
                retired_intermediates.push_back(it->second);
                to_defer.push_back(it->second);
                it = intermediates_.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = conversion_groups_.begin(); it != conversion_groups_.end();) {
            const bool uses = std::find(gone.begin(), gone.end(), it->first.src) != gone.end() ||
                              std::find(gone.begin(), gone.end(), it->first.dst) != gone.end();
            if (uses) {
                to_defer.push_back(it->second);
                it = conversion_groups_.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const uint32_t id : to_defer) {
        defer_destroy_resource(id);
    }
    return retired_intermediates;
}

uint32_t StateBridgeDevice::intermediate_storage_texture(uint32_t resource_id, const TextureInfo& info,
                                                         uint8_t format, CommandEncoder& encoder,
                                                         bool stencil_plane) {
    std::lock_guard lock(objects_mutex_);
    flush_retired_locked(encoder);
    uint32_t& id =
        intermediates_[IntermediateKey{resource_id, format, static_cast<uint8_t>(stencil_plane ? 1 : 0), {}}];
    if (id == 0) {
        id = allocate_id();
        GpuCreateTexture2Payload p{};
        p.id = id;
        p.width = std::max(info.width, 1u);
        p.height = std::max(info.height, 1u);
        p.depth_or_layers = std::max(info.depth_or_array_size, 1u);
        p.mip_levels = std::max(info.mip_levels, 1u);
        p.sample_count = 1;
        p.usage_flags = GpuTextureUsage::TextureBinding | GpuTextureUsage::StorageBinding;
        p.dimension = static_cast<uint8_t>(info.dimension);
        p.format = format;
        emit_device_object(id, encoder, [&](CommandEncoder& dst) {
            dst.emit_packet(GpuOpcode::CreateTexture2, &p, sizeof(p));
        });
    }
    return id;
}

StateBridgeDevice::StorageConversionPipeline StateBridgeDevice::storage_conversion_pipeline(
    uint8_t dst_format, uint8_t dimension, CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    StorageConversionPipeline& c = conversion_pipelines_[uint32_t{dst_format} << 8 | dimension];
    if (c.pipeline == 0) {
        GpuBindGroupLayoutEntry entries[2] = {};
        entries[0].binding = 0;
        entries[0].visibility = GpuShaderStageBits::Compute;
        entries[0].type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
        entries[0].binding_type = static_cast<uint8_t>(GpuTextureSampleType::UnfilterableFloat);
        entries[0].view_dimension = dimension;
        entries[1].binding = 1;
        entries[1].visibility = GpuShaderStageBits::Compute;
        entries[1].type = static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture);
        entries[1].binding_type = static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly);
        entries[1].view_dimension = dimension;
        entries[1].storage_format = dst_format;
        c.group_layout = allocate_id();
        emit_device_object(c.group_layout, encoder, [&](CommandEncoder& dst) {
            GpuCreateBindGroupLayoutPayload layout{};
            layout.id = c.group_layout;
            layout.entry_count = 2;
            dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &layout, sizeof(layout), entries, sizeof(entries));
        });
        c.pipeline = allocate_id();
        emit_device_object(c.pipeline, encoder, [&](CommandEncoder& dst) {
            const uint32_t module = allocate_id();
            dst.create_shader_module(module, GpuShaderStage::Compute, storage_conversion_wgsl(dst_format, dimension));
            GpuCreatePipelineLayoutPayload pl{};
            pl.id = allocate_id();
            pl.bind_group_layout_count = 1;
            dst.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &c.group_layout, sizeof(uint32_t));
            GpuCreateComputePipeline2Payload p{};
            p.id = c.pipeline;
            p.cs_module_id = module;
            p.pipeline_layout_id = pl.id;
            dst.emit_packet(GpuOpcode::CreateComputePipeline2, &p, sizeof(p));
        });
    }
    return c;
}

uint32_t StateBridgeDevice::storage_conversion_group(const StorageConversionPipeline& pipeline, uint8_t dimension,
                                                     uint32_t src, uint8_t src_format, uint32_t dst,
                                                     uint8_t dst_format, uint32_t mip, uint32_t base_layer,
                                                     uint32_t layer_count, CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    uint32_t& id = conversion_groups_[ConversionGroupKey{pipeline.group_layout, src, dst, mip, base_layer, layer_count}];
    if (id == 0) {
        id = allocate_id();
        GpuBindGroupEntry2 e[2] = {};
        const uint32_t ids[2] = {src, dst};
        const uint8_t formats[2] = {src_format, dst_format};
        for (uint32_t i = 0; i < 2; ++i) {
            e[i].binding = i;
            e[i].resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
            e[i].resource_id = ids[i];
            e[i].view.format = formats[i];
            e[i].view.dimension = dimension;
            e[i].view.base_mip_level = static_cast<uint16_t>(mip);
            e[i].view.mip_level_count = 1;
            e[i].view.base_array_layer = base_layer;
            e[i].view.array_layer_count = layer_count;
        }
        GpuCreateBindGroup2Payload g{};
        g.id = id;
        g.layout_id = pipeline.group_layout;
        g.entry_count = 2;
        emit_device_object(id, encoder, [&](CommandEncoder& out) {
            out.emit_packet(GpuOpcode::CreateBindGroup2, &g, sizeof(g), e, sizeof(e));
        });
    }
    return id;
}

StateBridgeDevice::StorageConversionPipeline StateBridgeDevice::storage_render_out_pipeline(
    uint8_t target_format, CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    StorageConversionPipeline& c = render_out_pipelines_[target_format];
    if (c.pipeline == 0) {
        GpuBindGroupLayoutEntry entry{};
        entry.binding = 0;
        entry.visibility = GpuShaderStageBits::Fragment;
        entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
        entry.binding_type = static_cast<uint8_t>(GpuTextureSampleType::UnfilterableFloat);
        entry.view_dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        c.group_layout = allocate_id();
        emit_device_object(c.group_layout, encoder, [&](CommandEncoder& dst) {
            GpuCreateBindGroupLayoutPayload layout{};
            layout.id = c.group_layout;
            layout.entry_count = 1;
            dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &layout, sizeof(layout), &entry, sizeof(entry));
        });
        c.pipeline = allocate_id();
        emit_device_object(c.pipeline, encoder, [&](CommandEncoder& dst) {
            const std::string wgsl = storage_render_out_wgsl();
            const uint32_t vs = allocate_id();
            const uint32_t fs = allocate_id();
            dst.create_shader_module(vs, GpuShaderStage::Vertex, wgsl);
            dst.create_shader_module(fs, GpuShaderStage::Fragment, wgsl);
            GpuCreatePipelineLayoutPayload pl{};
            pl.id = allocate_id();
            pl.bind_group_layout_count = 1;
            dst.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &c.group_layout, sizeof(uint32_t));
            GpuCreateRenderPipeline2Payload p{};
            p.id = c.pipeline;
            p.vs_module_id = vs;
            p.fs_module_id = fs;
            p.pipeline_layout_id = pl.id;
            GpuRenderPipelineKey& k = p.key;
            k.ps_shader = 1;
            k.color_target_count = 1;
            k.color_targets[0].format = target_format;
            k.color_targets[0].write_mask = 0xFu;
            k.primitive.topology = static_cast<uint8_t>(GpuPrimitiveTopology::TriangleList);
            k.multisample.count = 1;
            k.multisample.mask = 0xFFFFFFFFu;
            sanitize_render_pipeline_key(&k, options_.features);
            dst.emit_packet(GpuOpcode::CreateRenderPipeline2, &p, sizeof(p));
        });
    }
    return c;
}

uint32_t StateBridgeDevice::storage_render_out_group(uint32_t layout, uint32_t intermediate,
                                                     uint8_t intermediate_format, uint32_t target_texture,
                                                     uint32_t mip, uint32_t layer, CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    uint32_t& id = conversion_groups_[ConversionGroupKey{layout, intermediate, target_texture, mip, layer, 1u}];
    if (id == 0) {
        id = allocate_id();
        GpuBindGroupEntry2 e{};
        e.binding = 0;
        e.resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
        e.resource_id = intermediate;
        e.view.format = intermediate_format;
        e.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        e.view.base_mip_level = static_cast<uint16_t>(mip);
        e.view.mip_level_count = 1;
        e.view.base_array_layer = layer;
        e.view.array_layer_count = 1;
        GpuCreateBindGroup2Payload g{};
        g.id = id;
        g.layout_id = layout;
        g.entry_count = 1;
        emit_device_object(id, encoder, [&](CommandEncoder& dst) {
            dst.emit_packet(GpuOpcode::CreateBindGroup2, &g, sizeof(g), &e, sizeof(e));
        });
    }
    return id;
}

// ---- Backlog 35: colour twins of depth planes written by compute ----

namespace {

bool twin_storage_format_ok(uint8_t code) noexcept {
    return code == static_cast<uint8_t>(pso::WgpuTextureFormat::R32Float) ||
           code == static_cast<uint8_t>(pso::WgpuTextureFormat::R32Uint);
}

// What the twin holds for a depth value d: the value itself (r32float), or the unsigned integer of
// the same memory (r32uint: a 16-bit unorm plane is the integer n of d = n / 65535; a 32-bit float
// plane is the float's bits).
std::string depth_twin_encode(uint8_t depth_format, uint8_t twin_format) {
    if (twin_format == static_cast<uint8_t>(pso::WgpuTextureFormat::R32Float)) {
        return "d";
    }
    return depth_format == static_cast<uint8_t>(pso::WgpuTextureFormat::Depth16Unorm)
               ? "u32(round(clamp(d, 0.0, 1.0) * 65535.0))"
               : "bitcast<u32>(d)";
}

std::string depth_twin_in_wgsl(uint8_t depth_format, uint8_t twin_format) {
    const bool uint_twin = twin_format == static_cast<uint8_t>(pso::WgpuTextureFormat::R32Uint);
    std::string s =
        "@group(0) @binding(0) var src: texture_depth_2d;\n"
        "@group(0) @binding(1) var dst: texture_storage_2d<FMT, write>;\n"
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        "  if (any(id.xy >= textureDimensions(dst))) { return; }\n"
        "  let d = textureLoad(src, id.xy, 0);\n"
        "  textureStore(dst, id.xy, VEC(ENC, 0, 0, 0));\n"
        "}\n";
    const auto replace = [&](const char* what, const std::string& with) {
        s.replace(s.find(what), std::string(what).size(), with);
    };
    replace("FMT", uint_twin ? "r32uint" : "r32float");
    replace("VEC", uint_twin ? "vec4<u32>" : "vec4<f32>");
    replace("ENC", depth_twin_encode(depth_format, twin_format));
    return s;
}

std::string depth_twin_out_wgsl(uint8_t depth_format, uint8_t twin_format) {
    const bool uint_twin = twin_format == static_cast<uint8_t>(pso::WgpuTextureFormat::R32Uint);
    std::string decode;
    if (!uint_twin) {
        decode = "v.x";
    } else if (depth_format == static_cast<uint8_t>(pso::WgpuTextureFormat::Depth16Unorm)) {
        decode = "f32(v.x) / 65535.0";
    } else {
        decode = "bitcast<f32>(v.x)";
    }
    std::string s =
        "@group(0) @binding(0) var src: texture_2d<TY>;\n"
        "struct VsOut { @builtin(position) pos: vec4<f32> };\n"
        "struct FsOut { @builtin(frag_depth) depth: f32 };\n"
        "@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {\n"
        "  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\n"
        "  return VsOut(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0));\n"
        "}\n"
        "@fragment fn fs_main(in: VsOut) -> FsOut {\n"
        "  let v = textureLoad(src, vec2<i32>(in.pos.xy), 0);\n"
        "  return FsOut(clamp(DEC, 0.0, 1.0));\n"
        "}\n";
    s.replace(s.find("TY"), 2, uint_twin ? "u32" : "f32");
    s.replace(s.find("DEC"), 3, decode);
    return s;
}

}  // namespace

StateBridgeDevice::DepthTwinPipelines StateBridgeDevice::depth_twin_pipelines(uint8_t depth_format,
                                                                              uint8_t twin_format,
                                                                              CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    DepthTwinPipelines& c = depth_twin_pipelines_[uint32_t{depth_format} << 8 | twin_format];
    if (c.in_pipeline != 0) {
        return c;
    }
    const bool uint_twin = twin_format == static_cast<uint8_t>(pso::WgpuTextureFormat::R32Uint);
    // depth -> twin: a compute pass sampling the depth aspect, storing into the twin.
    {
        GpuBindGroupLayoutEntry entries[2] = {};
        entries[0].binding = 0;
        entries[0].visibility = GpuShaderStageBits::Compute;
        entries[0].type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
        entries[0].binding_type = static_cast<uint8_t>(GpuTextureSampleType::Depth);
        entries[0].view_dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        entries[1].binding = 1;
        entries[1].visibility = GpuShaderStageBits::Compute;
        entries[1].type = static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture);
        entries[1].binding_type = static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly);
        entries[1].view_dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        entries[1].storage_format = twin_format;
        c.in_layout = allocate_id();
        emit_device_object(c.in_layout, encoder, [&](CommandEncoder& dst) {
            GpuCreateBindGroupLayoutPayload layout{};
            layout.id = c.in_layout;
            layout.entry_count = 2;
            dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &layout, sizeof(layout), entries, sizeof(entries));
        });
        c.in_pipeline = allocate_id();
        emit_device_object(c.in_pipeline, encoder, [&](CommandEncoder& dst) {
            const uint32_t module = allocate_id();
            dst.create_shader_module(module, GpuShaderStage::Compute, depth_twin_in_wgsl(depth_format, twin_format));
            GpuCreatePipelineLayoutPayload pl{};
            pl.id = allocate_id();
            pl.bind_group_layout_count = 1;
            dst.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &c.in_layout, sizeof(uint32_t));
            GpuCreateComputePipeline2Payload p{};
            p.id = c.in_pipeline;
            p.cs_module_id = module;
            p.pipeline_layout_id = pl.id;
            dst.emit_packet(GpuOpcode::CreateComputePipeline2, &p, sizeof(p));
        });
    }
    // twin -> depth: a fullscreen triangle whose fragment writes frag_depth from the twin.
    {
        GpuBindGroupLayoutEntry entry{};
        entry.binding = 0;
        entry.visibility = GpuShaderStageBits::Fragment;
        entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
        entry.binding_type = static_cast<uint8_t>(uint_twin ? GpuTextureSampleType::Uint
                                                            : GpuTextureSampleType::UnfilterableFloat);
        entry.view_dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        c.out_layout = allocate_id();
        emit_device_object(c.out_layout, encoder, [&](CommandEncoder& dst) {
            GpuCreateBindGroupLayoutPayload layout{};
            layout.id = c.out_layout;
            layout.entry_count = 1;
            dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &layout, sizeof(layout), &entry, sizeof(entry));
        });
        c.out_pipeline = allocate_id();
        emit_device_object(c.out_pipeline, encoder, [&](CommandEncoder& dst) {
            const std::string wgsl = depth_twin_out_wgsl(depth_format, twin_format);
            const uint32_t vs = allocate_id();
            const uint32_t fs = allocate_id();
            dst.create_shader_module(vs, GpuShaderStage::Vertex, wgsl);
            dst.create_shader_module(fs, GpuShaderStage::Fragment, wgsl);
            GpuCreatePipelineLayoutPayload pl{};
            pl.id = allocate_id();
            pl.bind_group_layout_count = 1;
            dst.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &c.out_layout, sizeof(uint32_t));
            GpuCreateRenderPipeline2Payload p{};
            p.id = c.out_pipeline;
            p.vs_module_id = vs;
            p.fs_module_id = fs;
            p.pipeline_layout_id = pl.id;
            GpuRenderPipelineKey& k = p.key;
            k.ps_shader = 1;
            k.color_target_count = 0;
            k.depth_stencil.format = depth_format;
            k.depth_stencil.depth_write_enabled = 1;
            k.depth_stencil.depth_compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
            k.depth_stencil.stencil_front.compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
            k.depth_stencil.stencil_back.compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
            k.primitive.topology = static_cast<uint8_t>(GpuPrimitiveTopology::TriangleList);
            k.multisample.count = 1;
            k.multisample.mask = 0xFFFFFFFFu;
            dst.emit_packet(GpuOpcode::CreateRenderPipeline2, &p, sizeof(p));
        });
    }
    return c;
}

uint32_t StateBridgeDevice::depth_twin_group(uint32_t layout, uint32_t depth_texture, uint32_t twin,
                                             bool into_twin, uint8_t twin_format, uint32_t mip, uint32_t layer,
                                             CommandEncoder& encoder, bool stencil_plane) {
    std::lock_guard lock(objects_mutex_);
    // Shares the conversion groups' map (so retiring either texture drops it): the layout id tells
    // the two directions apart, the key's mip / layer slots name the subresource.
    uint32_t& id = conversion_groups_[ConversionGroupKey{
        layout, into_twin ? depth_texture : twin, into_twin ? twin : depth_texture, mip, layer, 1}];
    if (id == 0) {
        id = allocate_id();
        GpuBindGroupEntry2 e[2] = {};
        uint32_t count = 1;
        e[0].binding = 0;
        e[0].resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
        e[0].view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        e[0].view.base_mip_level = static_cast<uint16_t>(mip);
        e[0].view.mip_level_count = 1;
        e[0].view.base_array_layer = layer;
        e[0].view.array_layer_count = 1;
        if (into_twin) {
            e[0].resource_id = depth_texture;
            e[0].view.aspect = static_cast<uint8_t>(stencil_plane ? GpuTextureAspect::StencilOnly
                                                                  : GpuTextureAspect::DepthOnly);
            e[1] = e[0];
            e[1].binding = 1;
            e[1].resource_id = twin;
            e[1].view.format = twin_format;
            e[1].view.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
            count = 2;
        } else {
            e[0].resource_id = twin;
            e[0].view.format = twin_format;
        }
        GpuCreateBindGroup2Payload g{};
        g.id = id;
        g.layout_id = layout;
        g.entry_count = count;
        emit_device_object(id, encoder, [&](CommandEncoder& dst) {
            dst.emit_packet(GpuOpcode::CreateBindGroup2, &g, sizeof(g), e, count * sizeof(GpuBindGroupEntry2));
        });
    }
    return id;
}

// Backlog 35, stencil plane. The twin is an unsigned-integer storage texture (r32uint, or r8uint when the
// stage declares it) holding the 8-bit stencil value.
namespace {

std::string stencil_twin_in_wgsl(uint8_t twin_format) {
    const bool r8 = twin_format == static_cast<uint8_t>(pso::WgpuTextureFormat::R8Uint);
    std::string s =
        "@group(0) @binding(0) var src: texture_2d<u32>;\n"
        "@group(0) @binding(1) var dst: texture_storage_2d<FMT, write>;\n"
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        "  if (any(id.xy >= textureDimensions(dst))) { return; }\n"
        "  let s = textureLoad(src, id.xy, 0).x;\n"
        "  textureStore(dst, id.xy, vec4<u32>(s, 0u, 0u, 0u));\n"
        "}\n";
    s.replace(s.find("FMT"), 3, r8 ? "r8uint" : "r32uint");
    return s;
}

// One shader per stencil bit: discards the texels whose twin value has the bit clear, so the masked
// Replace draw (reference = write mask = the bit) sets exactly the texels that hold it.
std::string stencil_twin_out_wgsl(uint32_t bit) {
    std::string s =
        "@group(0) @binding(0) var src: texture_2d<u32>;\n"
        "struct VsOut { @builtin(position) pos: vec4<f32> };\n"
        "@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {\n"
        "  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\n"
        "  return VsOut(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0));\n"
        "}\n"
        "@fragment fn fs_main(in: VsOut) {\n"
        "  let v = textureLoad(src, vec2<i32>(in.pos.xy), 0).x;\n"
        "  if ((v & BITu) == 0u) { discard; }\n"
        "}\n";
    s.replace(s.find("BIT"), 3, std::to_string(1u << bit));
    return s;
}

}  // namespace

StateBridgeDevice::StencilTwinPipelines StateBridgeDevice::stencil_twin_pipelines(uint8_t depth_format,
                                                                                  uint8_t twin_format,
                                                                                  CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    StencilTwinPipelines& c = stencil_twin_pipelines_[uint32_t{depth_format} << 8 | twin_format];
    if (c.in_pipeline != 0) {
        return c;
    }
    {
        GpuBindGroupLayoutEntry entries[2] = {};
        entries[0].binding = 0;
        entries[0].visibility = GpuShaderStageBits::Compute;
        entries[0].type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
        entries[0].binding_type = static_cast<uint8_t>(GpuTextureSampleType::Uint);
        entries[0].view_dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        entries[1].binding = 1;
        entries[1].visibility = GpuShaderStageBits::Compute;
        entries[1].type = static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture);
        entries[1].binding_type = static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly);
        entries[1].view_dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        entries[1].storage_format = twin_format;
        c.in_layout = allocate_id();
        emit_device_object(c.in_layout, encoder, [&](CommandEncoder& dst) {
            GpuCreateBindGroupLayoutPayload layout{};
            layout.id = c.in_layout;
            layout.entry_count = 2;
            dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &layout, sizeof(layout), entries, sizeof(entries));
        });
        c.in_pipeline = allocate_id();
        emit_device_object(c.in_pipeline, encoder, [&](CommandEncoder& dst) {
            const uint32_t module = allocate_id();
            dst.create_shader_module(module, GpuShaderStage::Compute, stencil_twin_in_wgsl(twin_format));
            GpuCreatePipelineLayoutPayload pl{};
            pl.id = allocate_id();
            pl.bind_group_layout_count = 1;
            dst.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &c.in_layout, sizeof(uint32_t));
            GpuCreateComputePipeline2Payload p{};
            p.id = c.in_pipeline;
            p.cs_module_id = module;
            p.pipeline_layout_id = pl.id;
            dst.emit_packet(GpuOpcode::CreateComputePipeline2, &p, sizeof(p));
        });
    }
    {
        GpuBindGroupLayoutEntry entry{};
        entry.binding = 0;
        entry.visibility = GpuShaderStageBits::Fragment;
        entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
        entry.binding_type = static_cast<uint8_t>(GpuTextureSampleType::Uint);
        entry.view_dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        c.out_layout = allocate_id();
        emit_device_object(c.out_layout, encoder, [&](CommandEncoder& dst) {
            GpuCreateBindGroupLayoutPayload layout{};
            layout.id = c.out_layout;
            layout.entry_count = 1;
            dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &layout, sizeof(layout), &entry, sizeof(entry));
        });
        for (uint32_t bit = 0; bit < 8; ++bit) {
            c.out_pipeline[bit] = allocate_id();
            emit_device_object(c.out_pipeline[bit], encoder, [&](CommandEncoder& dst) {
                const std::string wgsl = stencil_twin_out_wgsl(bit);
                const uint32_t vs = allocate_id();
                const uint32_t fs = allocate_id();
                dst.create_shader_module(vs, GpuShaderStage::Vertex, wgsl);
                dst.create_shader_module(fs, GpuShaderStage::Fragment, wgsl);
                GpuCreatePipelineLayoutPayload pl{};
                pl.id = allocate_id();
                pl.bind_group_layout_count = 1;
                dst.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &c.out_layout, sizeof(uint32_t));
                GpuCreateRenderPipeline2Payload p{};
                p.id = c.out_pipeline[bit];
                p.vs_module_id = vs;
                p.fs_module_id = fs;
                p.pipeline_layout_id = pl.id;
                GpuRenderPipelineKey& k = p.key;
                k.ps_shader = 1;
                k.color_target_count = 0;
                k.depth_stencil.format = depth_format;
                k.depth_stencil.depth_write_enabled = 0;  // the depth plane is read-only in this pass
                k.depth_stencil.depth_compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
                GpuStencilFaceKey face{};
                face.compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
                face.fail_op = static_cast<uint8_t>(pso::WgpuStencilOperation::Keep);
                face.depth_fail_op = static_cast<uint8_t>(pso::WgpuStencilOperation::Keep);
                face.pass_op = static_cast<uint8_t>(pso::WgpuStencilOperation::Replace);
                k.depth_stencil.stencil_front = face;
                k.depth_stencil.stencil_back = face;
                k.depth_stencil.stencil_read_mask = 0xFFu;
                k.depth_stencil.stencil_write_mask = 1u << bit;
                k.primitive.topology = static_cast<uint8_t>(GpuPrimitiveTopology::TriangleList);
                k.multisample.count = 1;
                k.multisample.mask = 0xFFFFFFFFu;
                dst.emit_packet(GpuOpcode::CreateRenderPipeline2, &p, sizeof(p));
            });
        }
    }
    return c;
}

// ---- G66: ExecuteIndirect clamp pass ----

const char* indirect_clamp_wgsl() noexcept {
    // One invocation per command. Params: the source record (word index and stride), the command
    // count, the count buffer word (when has_count), the record kind and the index buffer's element
    // count. Commands at or past the count buffer's value, or whose record does not fit in the
    // source buffer, are written as all zeros (no work). Indexed draws are clamped to the index
    // buffer, dispatches to the per-dimension workgroup limit. Every draw is cut to MAX_DRAW_WORK
    // vertices (indices) x instances and every dispatch to MAX_GROUPS workgroups in all: a garbage
    // record (GPU-written arguments) would otherwise keep the GPU busy for seconds (a TDR, device lost).
    return R"(const MAX_DRAW_WORK: u32 = 67108864u;
const MAX_GROUPS: u32 = 4194304u;
fn cut_instances(per_instance: u32, instances: u32) -> u32 {
    if (per_instance == 0u || instances <= MAX_DRAW_WORK / per_instance) {
        return instances;
    }
    return max(MAX_DRAW_WORK / per_instance, 1u);
}
struct Params {
    src_word: u32,
    stride_words: u32,
    count: u32,
    count_word: u32,
    has_count: u32,
    kind: u32,
    index_limit: u32,
    group_limit: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> cnt: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.count) {
        return;
    }
    var words = 4u;
    var out_words = 4u;
    if (p.kind == 1u) {
        words = 5u;
        out_words = 5u;
    } else if (p.kind == 2u) {
        words = 3u;
        out_words = 3u;
    } else if (p.kind == 3u) {
        words = 3u;
        out_words = 7u;
    }
    var n = p.count;
    if (p.has_count != 0u) {
        n = 0u;
        if (p.count_word < arrayLength(&cnt)) {
            n = min(cnt[p.count_word], p.count);
        }
    }
    let base = p.src_word + i * p.stride_words;
    let live = i < n && base < arrayLength(&src) && words <= arrayLength(&src) - base;
    let out = i * out_words;
    for (var k = 0u; k < out_words; k = k + 1u) {
        var v = 0u;
        if (live && k < words) {
            v = src[base + k];
        }
        dst[out + k] = v;
    }
    if (!live) {
        return;
    }
    if (p.kind == 0u) {
        let vertices = min(dst[out], MAX_DRAW_WORK);
        dst[out] = vertices;
        dst[out + 1u] = cut_instances(vertices, dst[out + 1u]);
    } else if (p.kind == 1u) {
        let first = dst[out + 2u];
        var indices = 0u;
        if (first < p.index_limit) {
            indices = min(dst[out], p.index_limit - first);
        }
        indices = min(indices, MAX_DRAW_WORK);
        dst[out] = indices;
        dst[out + 1u] = cut_instances(indices, dst[out + 1u]);
    } else if (p.kind == 2u) {
        let x = min(dst[out], p.group_limit);
        var y = min(dst[out + 1u], p.group_limit);
        var z = min(dst[out + 2u], p.group_limit);
        if (x != 0u && y > MAX_GROUPS / x) {
            y = max(MAX_GROUPS / x, 1u);
        }
        if (x != 0u && y != 0u && z > MAX_GROUPS / (x * y)) {
            z = max(MAX_GROUPS / (x * y), 1u);
        }
        dst[out] = x;
        dst[out + 1u] = y;
        dst[out + 2u] = z;
    } else if (p.kind == 3u) {
        // Mesh dispatch: x-dimension groups only (y = z = 1), at most group_limit; the draw that
        // follows covers index_limit (max primitives * 3) vertices per group.
        var groups = 0u;
        if (dst[out + 1u] == 1u && dst[out + 2u] == 1u) {
            groups = min(dst[out], p.group_limit);
        }
        dst[out] = groups;
        dst[out + 1u] = 1u;
        dst[out + 2u] = 1u;
        dst[out + 3u] = groups * p.index_limit;
        dst[out + 4u] = 1u;
    }
}
)";
}

uint32_t indirect_output_bytes(IndirectKind kind) noexcept {
    switch (kind) {
    case IndirectKind::Draw: return 16u;
    case IndirectKind::DrawIndexed: return 20u;
    case IndirectKind::Dispatch: return 12u;
    case IndirectKind::DispatchMesh: return 28u;
    }
    return 16u;
}

const char* indirect_constants_wgsl() noexcept {
    // One invocation per command. Slots of dead commands (past the count buffer's value, or whose
    // record does not fit in the source buffer) keep the parameter's current values.
    return R"(@group(0) @binding(0) var<storage, read_write> chunk: array<u32>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> cnt: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let count = chunk[2];
    if (i >= count) {
        return;
    }
    let nargs = chunk[5];
    var n = count;
    if (chunk[4] != 0u) {
        n = 0u;
        if (chunk[3] < arrayLength(&cnt)) {
            n = min(cnt[chunk[3]], count);
        }
    }
    let base = chunk[0] + i * chunk[1];
    let live = i < n && base < arrayLength(&src) && chunk[7] <= arrayLength(&src) - base;
    for (var j = 0u; j < nargs; j = j + 1u) {
        let a = 8u + 4u * j;
        let slot = chunk[6] + (i * nargs + j) * 64u;
        let basew = 64u + 64u * j;
        for (var k = 0u; k < 64u; k = k + 1u) {
            chunk[slot + k] = chunk[basew + k];
        }
        if (live) {
            for (var k = 0u; k < chunk[a + 2u]; k = k + 1u) {
                let d = chunk[a + 1u] + k;
                if (d < 64u) {
                    chunk[slot + d] = src[base + chunk[a] + k];
                }
            }
        }
    }
}
)";
}

// ---- G66: mesh pipelines ----

bool valid_mesh_output_layout(const MeshOutputLayout& l) noexcept {
    return l.max_vertices >= 1 && l.max_vertices <= 256 && l.max_primitives >= 1 && l.max_primitives <= 256 &&
           l.vertex_words >= 4 && l.vertex_words % 4 == 0 && l.vertex_words <= 4 + 4 * kMaxMeshAttributes;
}

uint32_t mesh_group_record_words(const MeshOutputLayout& l) noexcept {
    return 4u + l.max_vertices * l.vertex_words + l.max_primitives * 3u;
}

uint32_t mesh_max_groups(const MeshOutputLayout& l) noexcept {
    if (!valid_mesh_output_layout(l)) {
        return 0;
    }
    const uint64_t fit = kMeshOutputBytes / (uint64_t{mesh_group_record_words(l)} * 4u);
    return static_cast<uint32_t>(std::min<uint64_t>(fit, kMaxIndirectWorkgroups));
}

std::string mesh_draw_vertex_wgsl(const MeshOutputLayout& l) {
    const uint32_t attributes = l.vertex_words >= 4 ? (l.vertex_words - 4u) / 4u : 0u;
    std::string w;
    w += "struct VsOut {\n    @builtin(position) position: vec4<f32>,\n";
    for (uint32_t a = 0; a < attributes; ++a) {
        w += "    @location(" + std::to_string(a) + ") a" + std::to_string(a) + ": vec4<f32>,\n";
    }
    w += "}\n";
    w += "@group(" + std::to_string(kMeshOutputGroup) + ") @binding(0) var<storage, read> ms_out: array<u32>;\n";
    w += "const RECORD_WORDS: u32 = " + std::to_string(mesh_group_record_words(l)) + "u;\n";
    w += "const MAX_VERTICES: u32 = " + std::to_string(l.max_vertices) + "u;\n";
    w += "const MAX_PRIMITIVES: u32 = " + std::to_string(l.max_primitives) + "u;\n";
    w += "const VERTEX_WORDS: u32 = " + std::to_string(l.vertex_words) + "u;\n";
    w += "const INDEX_BASE: u32 = " + std::to_string(4u + l.max_vertices * l.vertex_words) + "u;\n";
    w += R"(fn load4(w: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(vec4<u32>(ms_out[w], ms_out[w + 1u], ms_out[w + 2u], ms_out[w + 3u]));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var o: VsOut;
    // Unused primitives: all three corners at one point outside the depth range (no fragments).
    o.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    let per_group = MAX_PRIMITIVES * 3u;
    let g = vertex_index / per_group;
    let k = vertex_index % per_group;
    let prim = k / 3u;
    let rec = g * RECORD_WORDS;
    if (rec + RECORD_WORDS > arrayLength(&ms_out)) {
        return o;
    }
    let vertices = min(ms_out[rec], MAX_VERTICES);
    let primitives = min(ms_out[rec + 1u], MAX_PRIMITIVES);
    if (prim >= primitives) {
        return o;
    }
    let tri = rec + INDEX_BASE + prim * 3u;
    if (max(ms_out[tri], max(ms_out[tri + 1u], ms_out[tri + 2u])) >= vertices) {
        return o;
    }
    let v = rec + 4u + ms_out[tri + k % 3u] * VERTEX_WORDS;
    o.position = load4(v);
)";
    for (uint32_t a = 0; a < attributes; ++a) {
        w += "    o.a" + std::to_string(a) + " = load4(v + " + std::to_string(4u + 4u * a) + "u);\n";
    }
    w += "    return o;\n}\n";
    return w;
}

uint64_t mesh_draw_vertex_identity(const MeshOutputLayout& l) noexcept {
    uint64_t h = 0xcbf29ce484222325ull;
    const uint32_t words[4] = {0x4d534456u /* "MSDV" */, l.max_vertices, l.max_primitives, l.vertex_words};
    for (uint32_t word : words) {
        for (int b = 0; b < 4; ++b) {
            h ^= (word >> (8 * b)) & 0xFFu;
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

StateBridgeDevice::PipelineLayoutRecord* StateBridgeDevice::mesh_layout(PipelineLayoutRecord* base, uint32_t visibility,
                                                                        uint8_t binding_type) {
    std::lock_guard lock(objects_mutex_);
    if (base == nullptr || base->groups.size() > kMeshOutputGroup) {
        return nullptr;
    }
    PipelineLayoutRecord*& found = mesh_layouts_[MeshLayoutKey{base, visibility << 8 | binding_type}];
    if (found != nullptr) {
        return found;
    }
    std::vector<GroupLayoutRecord*> groups = base->groups;
    while (groups.size() < kMeshOutputGroup) {
        if (empty_group_layout_ == nullptr) {
            auto record = std::make_unique<GroupLayoutRecord>();
            record->id = allocate_id();
            record->header.id = record->id;
            record->runtime_owned = false;
            empty_group_layout_ = record.get();
            group_layouts_.push_back(std::move(record));
        }
        groups.push_back(empty_group_layout_);
    }
    GpuBindGroupLayoutEntry entry{};
    entry.binding = 0;
    entry.visibility = visibility;
    entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
    entry.binding_type = binding_type;
    GroupLayoutRecord* output = nullptr;
    for (const auto& existing : group_layouts_) {
        if (existing->runtime_owned && existing->layout_entries.size() == 1 &&
            std::memcmp(&existing->layout_entries[0], &entry, sizeof(entry)) == 0) {
            output = existing.get();
            break;
        }
    }
    if (output == nullptr) {
        auto record = std::make_unique<GroupLayoutRecord>();
        record->id = allocate_id();
        record->header.id = record->id;
        record->header.entry_count = 1;
        record->entries.push_back(entry);
        record->layout_entries.push_back(entry);
        record->joins.push_back(d3d::ShaderBinding{});
        record->runtime_owned = true;
        output = record.get();
        group_layouts_.push_back(std::move(record));
    }
    groups.push_back(output);
    auto layout = std::make_unique<PipelineLayoutRecord>();
    layout->id = allocate_id();
    layout->groups = std::move(groups);
    found = layout.get();
    pipeline_layouts_.push_back(std::move(layout));
    return found;
}

uint32_t StateBridgeDevice::mesh_output_group(const GroupLayoutRecord& group, CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    if (mesh_output_buffer_ == 0) {
        mesh_output_buffer_ = allocate_id();
        emit_device_object(mesh_output_buffer_, encoder, [&](CommandEncoder& dst) {
            dst.create_buffer(mesh_output_buffer_, GpuBufferUsage::Storage, kMeshOutputBytes);
        });
    }
    uint32_t& id = mesh_output_groups_[group.id];
    if (id == 0) {
        id = allocate_id();
        GpuBindGroupEntry2 e{};
        e.binding = 0;
        e.resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
        e.resource_id = mesh_output_buffer_;
        GpuCreateBindGroup2Payload g{};
        g.id = id;
        g.layout_id = group.id;
        g.entry_count = 1;
        emit_device_object(id, encoder, [&](CommandEncoder& dst) {
            dst.emit_packet(GpuOpcode::CreateBindGroup2, &g, sizeof(g), &e, sizeof(e));
        });
    }
    return id;
}

std::unique_ptr<MeshPipelineState> StateBridgeDevice::create_mesh_pipeline_state(
    const pso::D3D12GraphicsPsoDesc& draw_desc, uint64_t mesh_shader, const MeshOutputLayout& output,
    const d3d::RootSignature* root_signature) {
    std::unique_ptr<MeshPipelineState> state(new MeshPipelineState());
    state->output_ = output;
    if (!valid_mesh_output_layout(output)) {
        state->diagnostics_ |= BridgeDiag::MissingPipelineLayout;
    }
    pso::D3D12ComputePsoDesc stage_desc{};
    stage_desc.cs = mesh_shader;
    stage_desc.root_signature_id = draw_desc.root_signature_id;
    state->stage_ = create_compute_pipeline_state(stage_desc, root_signature);
    state->draw_ = create_graphics_pipeline_state(draw_desc, root_signature);
    // Both pipelines get the runtime's output group after their own (bound from the graphics root
    // arguments: the mesh stage is a graphics stage).
    const auto extend = [&](PipelineLayoutRecord** layout, std::vector<d3d::BindGroupPlan>* plans, uint32_t visibility,
                            GpuBufferBindingType type, uint64_t* diagnostics) {
        if (*layout == nullptr) {
            return;
        }
        PipelineLayoutRecord* extended = mesh_layout(*layout, visibility, static_cast<uint8_t>(type));
        if (extended == nullptr) {
            *diagnostics |= BridgeDiag::MissingPipelineLayout;
            *layout = nullptr;
            plans->clear();
            return;
        }
        *layout = extended;
        if (!plans->empty()) {
            plans->resize(extended->groups.size());  // padding and output groups have no plan
        }
    };
    extend(&state->stage_->layout_, &state->stage_->plans_, GpuShaderStageBits::Compute, GpuBufferBindingType::Storage,
           &state->stage_->diagnostics_);
    extend(&state->draw_->layout_, &state->draw_->plans_, GpuShaderStageBits::Vertex,
           GpuBufferBindingType::ReadOnlyStorage, &state->draw_->diagnostics_);
    state->diagnostics_ |= state->stage_->diagnostics_ | state->draw_->diagnostics_;
    return state;
}

void StateBridgeDevice::release_mesh_pipeline_state(std::unique_ptr<MeshPipelineState> state, CommandEncoder& encoder) {
    if (state != nullptr) {
        release_pipeline_state(std::move(state->draw_), encoder);
    }
}

StateBridgeDevice::IndirectClamp StateBridgeDevice::indirect_clamp(CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    IndirectClamp& c = indirect_clamp_;
    if (c.pipeline == 0) {
        GpuBindGroupLayoutEntry entries[4] = {};
        const uint8_t types[4] = {static_cast<uint8_t>(GpuBufferBindingType::Uniform),
                                  static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage),
                                  static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage),
                                  static_cast<uint8_t>(GpuBufferBindingType::Storage)};
        for (uint32_t i = 0; i < 4; ++i) {
            entries[i].binding = i;
            entries[i].visibility = GpuShaderStageBits::Compute;
            entries[i].type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
            entries[i].binding_type = types[i];
        }
        entries[0].flags = GpuBindGroupLayoutEntryFlags::HasDynamicOffset;  // params slot in the root constant ring
        entries[0].min_binding_size = 32;
        entries[3].flags = GpuBindGroupLayoutEntryFlags::HasDynamicOffset;  // output window in the indirect ring
        c.group_layout = allocate_id();
        emit_device_object(c.group_layout, encoder, [&](CommandEncoder& dst) {
            GpuCreateBindGroupLayoutPayload layout{};
            layout.id = c.group_layout;
            layout.entry_count = 4;
            dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &layout, sizeof(layout), entries, sizeof(entries));
        });
        c.pipeline = allocate_id();
        emit_device_object(c.pipeline, encoder, [&](CommandEncoder& dst) {
            const uint32_t module = allocate_id();
            dst.create_shader_module(module, GpuShaderStage::Compute, indirect_clamp_wgsl());
            GpuCreatePipelineLayoutPayload pl{};
            pl.id = allocate_id();
            pl.bind_group_layout_count = 1;
            dst.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &c.group_layout, sizeof(uint32_t));
            GpuCreateComputePipeline2Payload cp{};
            cp.id = c.pipeline;
            cp.cs_module_id = module;
            cp.pipeline_layout_id = pl.id;
            dst.emit_packet(GpuOpcode::CreateComputePipeline2, &cp, sizeof(cp));
        });
        c.ring = allocate_id();
        emit_device_object(c.ring, encoder, [&](CommandEncoder& dst) {
            dst.create_buffer(c.ring, GpuBufferUsage::Storage | GpuBufferUsage::Indirect,
                              kIndirectRingBytes + kIndirectWindowBytes);
        });
    }
    return c;
}

uint32_t StateBridgeDevice::indirect_clamp_group(const IndirectClamp& clamp, uint32_t params_ring, uint32_t args,
                                                 uint32_t count, CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    uint32_t& id = indirect_groups_[ResourcePairKey{args, count}];
    if (id == 0) {
        id = allocate_id();
        GpuBindGroupEntry2 e[4] = {};
        const uint32_t ids[4] = {params_ring, args, count != 0 ? count : args, clamp.ring};
        const uint64_t sizes[4] = {32, 0, 0, kIndirectWindowBytes};
        for (uint32_t i = 0; i < 4; ++i) {
            e[i].binding = i;
            e[i].resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
            e[i].resource_id = ids[i];
            e[i].size = sizes[i];
        }
        GpuCreateBindGroup2Payload g{};
        g.id = id;
        g.layout_id = clamp.group_layout;
        g.entry_count = 4;
        emit_device_object(id, encoder, [&](CommandEncoder& dst) {
            dst.emit_packet(GpuOpcode::CreateBindGroup2, &g, sizeof(g), e, sizeof(e));
        });
    }
    return id;
}

StateBridgeDevice::IndirectConstants StateBridgeDevice::indirect_constants(CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    IndirectConstants& c = indirect_constants_;
    if (c.pipeline == 0) {
        GpuBindGroupLayoutEntry entries[3] = {};
        const uint8_t types[3] = {static_cast<uint8_t>(GpuBufferBindingType::Storage),
                                  static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage),
                                  static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage)};
        for (uint32_t i = 0; i < 3; ++i) {
            entries[i].binding = i;
            entries[i].visibility = GpuShaderStageBits::Compute;
            entries[i].type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
            entries[i].binding_type = types[i];
        }
        entries[0].flags = GpuBindGroupLayoutEntryFlags::HasDynamicOffset;  // the chunk in the root-constant ring
        c.group_layout = allocate_id();
        emit_device_object(c.group_layout, encoder, [&](CommandEncoder& dst) {
            GpuCreateBindGroupLayoutPayload layout{};
            layout.id = c.group_layout;
            layout.entry_count = 3;
            dst.emit_packet(GpuOpcode::CreateBindGroupLayout, &layout, sizeof(layout), entries, sizeof(entries));
        });
        c.pipeline = allocate_id();
        emit_device_object(c.pipeline, encoder, [&](CommandEncoder& dst) {
            const uint32_t module = allocate_id();
            dst.create_shader_module(module, GpuShaderStage::Compute, indirect_constants_wgsl());
            GpuCreatePipelineLayoutPayload pl{};
            pl.id = allocate_id();
            pl.bind_group_layout_count = 1;
            dst.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &c.group_layout, sizeof(uint32_t));
            GpuCreateComputePipeline2Payload cp{};
            cp.id = c.pipeline;
            cp.cs_module_id = module;
            cp.pipeline_layout_id = pl.id;
            dst.emit_packet(GpuOpcode::CreateComputePipeline2, &cp, sizeof(cp));
        });
    }
    return c;
}

uint32_t StateBridgeDevice::indirect_constants_group(const IndirectConstants& pass, uint32_t ring, uint32_t args,
                                                     uint32_t count, uint32_t window_bytes, CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    const uint64_t key = (uint64_t{args} << 40) ^ (uint64_t{count} << 16) ^ uint64_t{window_bytes >> 10};
    uint32_t& id = indirect_constants_groups_[key];
    if (id == 0) {
        id = allocate_id();
        GpuBindGroupEntry2 e[3] = {};
        const uint32_t ids[3] = {ring, args, count != 0 ? count : args};
        const uint64_t sizes[3] = {window_bytes, 0, 0};
        for (uint32_t i = 0; i < 3; ++i) {
            e[i].binding = i;
            e[i].resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
            e[i].resource_id = ids[i];
            e[i].size = sizes[i];
        }
        GpuCreateBindGroup2Payload g{};
        g.id = id;
        g.layout_id = pass.group_layout;
        g.entry_count = 3;
        emit_device_object(id, encoder, [&](CommandEncoder& dst) {
            dst.emit_packet(GpuOpcode::CreateBindGroup2, &g, sizeof(g), e, sizeof(e));
        });
    }
    return id;
}

bool StateBridgeDevice::indirect_clamp_group_known(uint32_t args, uint32_t count) {
    std::lock_guard lock(objects_mutex_);
    const auto it = indirect_groups_.find(ResourcePairKey{args, count});
    return it != indirect_groups_.end() && it->second != 0;
}

uint32_t StateBridgeDevice::allocate_indirect_window() {
    std::lock_guard lock(objects_mutex_);
    // GPU-ordered: the clamp pass that overwrites a reused window runs after the work that read it.
    if (indirect_cursor_ + kIndirectWindowBytes > kIndirectRingBytes) {
        indirect_cursor_ = 0;
    }
    const uint32_t offset = indirect_cursor_;
    indirect_cursor_ += (kIndirectWindowBytes + 255u) & ~255u;
    return offset;
}

StateBridgeDevice::ViewKey StateBridgeDevice::make_view_key(const ViewRecord& record) noexcept {
    ViewKey key{};
    key.texture_id = record.texture_id;
    key.resource_id = record.resource_id;
    key.view = record.view;
    key.view.reserved = 0;
    key.range = record.range;
    key.usage = static_cast<uint8_t>(record.usage);
    return key;
}

static pso::WgpuTextureFormat host_texture_format(const TextureInfo& tex, uint32_t features) noexcept {
    if (tex.created_format != pso::WgpuTextureFormat::Undefined) {
        return tex.created_format;
    }
    const pso::FormatUsage usage = tex.depth_stencil  ? pso::FormatUsage::DepthStencil
                                   : tex.render_target ? pso::FormatUsage::RenderTarget
                                   : tex.sampled_only  ? pso::FormatUsage::SampledOnly
                                                       : pso::FormatUsage::Sampled;
    return pso::map_texture_format(tex.format, usage, features).format;
}

bool StateBridgeDevice::is_depth_resource(uint32_t resource_id) const {
    if (resource_id == 0) {
        return false;
    }
    {
        std::shared_lock lock(resources_mutex_);
        if (const auto it = textures_.find(resource_id); it != textures_.end() && it->second.depth_stencil) {
            return true;
        }
    }
    std::lock_guard lock(views_mutex_);
    return depth_textures_.count(resource_id) != 0;
}

uint32_t StateBridgeDevice::intern_view(const ViewRecord& record) const {
    const ViewKey key = make_view_key(record);
    std::lock_guard lock(views_mutex_);
    if (const auto it = view_ids_.find(key); it != view_ids_.end()) {
        return it->second;
    }
    uint32_t id = 0;
    if (!free_views_.empty()) {
        id = free_views_.back();
        free_views_.pop_back();
        views_[id - 1] = record;
    } else {
        views_.push_back(record);
        view_live_.push_back(0);
        id = static_cast<uint32_t>(views_.size());
    }
    view_live_[id - 1] = 1;
    view_ids_.emplace(key, id);
    views_by_texture_[record.texture_id].push_back(id);
    if (record.resource_id != 0 && record.resource_id != record.texture_id) {
        views_by_texture_[record.resource_id].push_back(id);
    }
    adjust_gauge(g_live_views, views_tracked_, view_ids_.size());
    return id;
}

std::vector<uint32_t> StateBridgeDevice::retire_views(uint32_t texture_id) {
    std::lock_guard lock(views_mutex_);
    depth_textures_.erase(texture_id);
    std::vector<uint32_t> retired;
    const auto it = views_by_texture_.find(texture_id);
    if (it == views_by_texture_.end()) {
        return retired;
    }
    const std::vector<uint32_t> ids = std::move(it->second);
    views_by_texture_.erase(it);
    retired.reserve(ids.size());
    for (const uint32_t id : ids) {
        if (id == 0 || id > view_live_.size() || view_live_[id - 1] == 0) {
            continue;  // already retired through its other texture id
        }
        view_ids_.erase(make_view_key(views_[id - 1]));
        view_live_[id - 1] = 0;
        views_[id - 1] = ViewRecord{};
        free_views_.push_back(id);
        retired.push_back(id);
    }
    adjust_gauge(g_live_views, views_tracked_, view_ids_.size());
    return retired;
}

size_t StateBridgeDevice::live_view_count() const {
    std::lock_guard lock(views_mutex_);
    return view_ids_.size();
}

const ViewRecord* StateBridgeDevice::view(uint32_t view_id) const {
    std::lock_guard lock(views_mutex_);
    if (view_id == 0 || view_id > views_.size() || view_live_[view_id - 1] == 0) {
        return nullptr;
    }
    return &views_[view_id - 1];
}

bool StateBridgeDevice::resolve_view_descriptor(uint64_t cpu_handle, d3d::DescriptorKind kind,
                                                d3d::ViewDescriptor* out) const {
    d3d::ResolvedDescriptor resolved{};
    if (heaps_.resolve_cpu(d3d::CpuDescriptorHandle{cpu_handle}, &resolved) != d3d::Status::Ok ||
        resolved.heap == nullptr) {
        return false;
    }
    const d3d::ViewDescriptor* v = resolved.heap->view(resolved.index);
    if (v == nullptr || v->kind != kind) {
        return false;
    }
    *out = *v;
    return true;
}

uint64_t StateBridgeDevice::rtv_descriptor_increment() const {
    return d3d::descriptor_handle_increment_size(d3d::DescriptorHeapType::Rtv);
}

bool StateBridgeDevice::resolve_rtv(uint64_t cpu_handle, pso::RenderTargetViewInfo* out) const {
    d3d::ViewDescriptor v{};
    if (!resolve_view_descriptor(cpu_handle, d3d::DescriptorKind::Rtv, &v)) {
        return false;
    }
    *out = pso::RenderTargetViewInfo{};
    if (v.resource_id == 0) {
        return true;  // null descriptor: unbound slot
    }
    TextureInfo tex{};
    if (!texture_info(v.resource_id, &tex)) {
        return false;
    }
    const pso::DxgiFormat format = v.format != 0 ? to_dxgi(v.format) : tex.format;
    const uint32_t mip = v.first_mip;
    uint32_t layer = 0;
    switch (v.dimension) {
    case d3d::ViewDimension::Texture1DArray:
    case d3d::ViewDimension::Texture2DArray:
    case d3d::ViewDimension::Texture2DMsArray:
        layer = v.first_array_slice;
        break;
    default:
        break;
    }
    const pso::WgpuTextureFormat tex_wgpu =
        tex.created_format != pso::WgpuTextureFormat::Undefined
            ? tex.created_format
            : pso::map_texture_format(tex.format, pso::FormatUsage::RenderTarget, options_.features).format;
    pso::WgpuTextureFormat view_wgpu =
        pso::map_texture_format(format, pso::FormatUsage::RenderTarget, options_.features).format;
    pso::DxgiFormat resolved_dxgi = format;
    if (tex.format == pso::DxgiFormat::A8_UNORM) {
        // A8_UNORM is held as rgba8unorm with (0, 0, 0, a): keep A8_UNORM identity so
        // pipeline specialization and clears preserve the alpha-only write mask and zero RGB.
        view_wgpu = tex_wgpu;
        resolved_dxgi = pso::DxgiFormat::A8_UNORM;
    } else if (tex_wgpu != pso::WgpuTextureFormat::Undefined && view_wgpu != pso::WgpuTextureFormat::Undefined &&
               !is_compatible_view_format(tex_wgpu, view_wgpu)) {
        view_wgpu = tex_wgpu;
        resolved_dxgi = tex.format;
    }
    ViewRecord record{};
    record.resource_id = v.resource_id;
    record.texture_id = tex.swapchain ? kGpuSwapchainTextureId : v.resource_id;
    record.view.format = static_cast<uint8_t>(view_wgpu);
    record.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
    record.view.base_mip_level = static_cast<uint16_t>(mip);
    record.view.mip_level_count = 1;
    record.view.base_array_layer = layer;
    record.view.array_layer_count = 1;
    record.range = d3d::SubresourceRange{mip, 1, layer, 1, 0, d3d::kAllMips};
    record.usage = d3d::ResourceUsage::RenderTarget;
    out->view_id = intern_view(record);
    out->format = resolved_dxgi;
    out->width = mip_extent(tex.width, mip);
    out->height = mip_extent(tex.height, mip);
    out->sample_count = tex.sample_count;
    return true;
}

bool StateBridgeDevice::resolve_dsv(uint64_t cpu_handle, pso::DepthStencilViewInfo* out) const {
    d3d::ViewDescriptor v{};
    if (!resolve_view_descriptor(cpu_handle, d3d::DescriptorKind::Dsv, &v)) {
        return false;
    }
    *out = pso::DepthStencilViewInfo{};
    if (v.resource_id == 0) {
        return true;
    }
    TextureInfo tex{};
    if (!texture_info(v.resource_id, &tex)) {
        return false;
    }
    const pso::DxgiFormat format = v.format != 0 ? to_dxgi(v.format) : tex.format;
    const uint32_t mip = v.first_mip;
    uint32_t layer = 0;
    switch (v.dimension) {
    case d3d::DsvDimension::Texture1DArray:
    case d3d::DsvDimension::Texture2DArray:
    case d3d::DsvDimension::Texture2DMsArray:
        layer = v.first_array_slice;
        break;
    default:
        break;
    }
    const uint32_t flags = v.flags & (d3d::DsvFlags::ReadOnlyDepth | d3d::DsvFlags::ReadOnlyStencil);
    const pso::WgpuTextureFormat wgpu_fmt =
        pso::map_texture_format(format, pso::FormatUsage::DepthStencil, options_.features).format;
    const bool has_depth = pso::has_depth_aspect(wgpu_fmt);
    const bool has_stencil = pso::has_stencil_aspect(wgpu_fmt);
    const bool depth_ro = (flags & d3d::DsvFlags::ReadOnlyDepth) != 0;
    const bool stencil_ro = (flags & d3d::DsvFlags::ReadOnlyStencil) != 0;
    const bool all_ro = (!has_depth || depth_ro) && (!has_stencil || stencil_ro);
    {
        std::lock_guard lock(views_mutex_);
        depth_textures_.insert(v.resource_id);
    }
    ViewRecord record{};
    record.resource_id = v.resource_id;
    record.texture_id = tex.swapchain ? kGpuSwapchainTextureId : v.resource_id;
    record.view.format = static_cast<uint8_t>(wgpu_fmt);
    record.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
    record.view.base_mip_level = static_cast<uint16_t>(mip);
    record.view.mip_level_count = 1;
    record.view.base_array_layer = layer;
    record.view.array_layer_count = 1;
    record.range = d3d::SubresourceRange{mip, 1, layer, 1, 0, d3d::kAllMips};
    record.usage = all_ro ? d3d::ResourceUsage::DepthStencilRead : d3d::ResourceUsage::DepthStencilWrite;
    out->view_id = intern_view(record);
    out->format = format;
    out->width = mip_extent(tex.width, mip);
    out->height = mip_extent(tex.height, mip);
    out->sample_count = tex.sample_count;
    out->flags = flags;
    return true;
}

bool StateBridgeDevice::resolve_gpu_va(uint64_t gpu_address, d3d::GpuVaResolution* out) const {
    std::shared_lock lock(resources_mutex_);
    auto it = buffers_by_va_.upper_bound(gpu_address);
    if (it == buffers_by_va_.begin()) {
        return false;
    }
    --it;
    const uint64_t base = it->first;
    const auto [resource_id, size] = it->second;
    if (gpu_address - base >= size) {
        return false;
    }
    out->resource_id = resource_id;
    out->offset = gpu_address - base;
    out->resource_size = size;
    return true;
}

bool StateBridgeDevice::gpu_va_trampoline(void* user, uint64_t gpu_address, d3d::GpuVaResolution* out) {
    return static_cast<const StateBridgeDevice*>(user)->resolve_gpu_va(gpu_address, out);
}

namespace {

uint32_t group_stage_mask(const std::vector<GpuBindGroupLayoutEntry>& entries, uint32_t pipeline_stages) {
    uint32_t mask = 0;
    for (const GpuBindGroupLayoutEntry& e : entries) {
        mask |= e.visibility;
    }
    mask &= pipeline_stages;
    return mask != 0 ? mask : pipeline_stages;
}

}  // namespace

std::unique_ptr<GraphicsPipelineState> StateBridgeDevice::create_graphics_pipeline_state(
    const pso::D3D12GraphicsPsoDesc& desc,
    const d3d::RootSignature* root_signature) {
    pso::PipelineKeyOptions key_options;
    key_options.features = options_.features;
    key_options.max_color_attachment_bytes_per_sample = options_.max_color_attachment_bytes_per_sample;
    // Sanitize depth-bias float fields (NaN, +/-Inf, -0.0f -> +0.0f) before PSO translation so
    // non-finite values neither poison WebGPU validation nor fragment the PipelineCache.
    pso::D3D12GraphicsPsoDesc sanitized = desc;
    sanitized.RasterizerState.SlopeScaledDepthBias =
        sanitize_depth_bias_float(desc.RasterizerState.SlopeScaledDepthBias);
    sanitized.RasterizerState.DepthBiasClamp = sanitize_depth_bias_float(desc.RasterizerState.DepthBiasClamp);
    std::unique_ptr<GraphicsPipelineState> state(new GraphicsPipelineState(sanitized, key_options));
    state->root_signature_ = root_signature;
    bool vs_known = true;
    bool ps_known = true;
    state->vs_module_id_ = module_id(desc.vs, &vs_known);
    state->fs_module_id_ = module_id(desc.ps, &ps_known);
    if (!vs_known || !ps_known) {
        state->diagnostics_ |= BridgeDiag::UnknownShaderModule;
    }
    state->layout_ = find_layout(desc.vs, desc.ps);
    if (state->layout_ == nullptr) {
        state->diagnostics_ |= BridgeDiag::MissingPipelineLayout;
    } else if (root_signature == nullptr) {
        state->diagnostics_ |= BridgeDiag::BindGroupBuildFailed;
    } else {
        for (const GroupLayoutRecord* group : state->layout_->groups) {
            const uint32_t stages = group_stage_mask(group->entries, d3d::WebGpuStage::Vertex | d3d::WebGpuStage::Fragment);
            state->plans_.push_back(d3d::BindGroupPlan::build(*root_signature, group->joins, stages));
        }
    }
    return state;
}

std::unique_ptr<ComputePipelineState> StateBridgeDevice::create_compute_pipeline_state(
    const pso::D3D12ComputePsoDesc& desc,
    const d3d::RootSignature* root_signature) {
    std::unique_ptr<ComputePipelineState> state(new ComputePipelineState());
    state->key_ = pso::make_compute_pipeline_key(desc);
    state->root_signature_ = root_signature;
    bool known = true;
    state->cs_module_id_ = module_id(desc.cs, &known);
    if (!known || state->cs_module_id_ == 0) {
        state->diagnostics_ |= BridgeDiag::UnknownShaderModule;
    }
    state->layout_ = find_layout(desc.cs, 0);
    if (state->layout_ == nullptr) {
        state->diagnostics_ |= BridgeDiag::MissingPipelineLayout;
    } else if (root_signature == nullptr) {
        state->diagnostics_ |= BridgeDiag::BindGroupBuildFailed;
    } else {
        for (const GroupLayoutRecord* group : state->layout_->groups) {
            if (group->runtime_owned) {
                state->plans_.emplace_back();  // bound by the runtime (RT arena group)
                continue;
            }
            state->plans_.push_back(d3d::BindGroupPlan::build(*root_signature, group->joins, d3d::WebGpuStage::Compute));
        }
    }
    if (state->cs_module_id_ != 0) {
        state->pipeline_id_ = pipeline_cache_->acquire(state->key_).pipeline_id;
    }
    return state;
}

void StateBridgeDevice::release_pipeline_state(std::unique_ptr<GraphicsPipelineState> state, CommandEncoder& encoder) {
    if (state == nullptr) {
        return;
    }
    for (uint32_t id : state->pso_.release(*pipeline_cache_)) {
        encoder.destroy_resource(id);
    }
}

// =====================================================================================================
// CommandListBridge
// =====================================================================================================

namespace {

// Tracks which CommandListBridge is currently executing BindGroupCache::get_or_build /
// refresh_table_entries on this thread so a re-entrant unregister_resource (e.g. from
// Device::ensure_address_texture LRU eviction during MemorySlotSource::view) queues its
// invalidation instead of mutating BindGroupCache mid-lookup.
thread_local CommandListBridge* t_building_bridge = nullptr;

struct BuildingBridgeGuard {
    CommandListBridge* prev = nullptr;
    explicit BuildingBridgeGuard(CommandListBridge* self) noexcept : prev(t_building_bridge) {
        t_building_bridge = self;
    }
    ~BuildingBridgeGuard() noexcept { t_building_bridge = prev; }
};

pso::TrackerOptions tracker_options(const BridgeOptions& options) {
    pso::TrackerOptions t;
    t.mid_pass_clear = options.mid_pass_clear;
    t.features = options.features;
    t.max_texture_dimension_2d = options.max_texture_dimension_2d;
    return t;
}

}  // namespace

CommandListBridge::CommandListBridge(StateBridgeDevice& device, CommandEncoder& encoder)
    : device_(device),
      encoder_(encoder),
      tracker_(tracker_options(device.options())),
      resource_states_(device.resource_states_) {
    bind_groups_.set_slot_source(device.options().slot_source);
    g_live_command_lists.fetch_add(1, std::memory_order_relaxed);
    device_.attach_bridge(this);
    Reset();
}

CommandListBridge::~CommandListBridge() {
    // A released list takes its cached WebGPU bind groups with it: without this every list the
    // title creates and releases left its bind groups alive on the host for the rest of the run.
    // Deferred, because a group may still be referenced by a recording submitted this frame.
    {
        std::lock_guard lock(cache_mutex_);
        std::vector<uint32_t> ids;
        bind_groups_.clear(&ids);
        for (const auto& kv : empty_bind_groups_) {
            ids.push_back(kv.second);
        }
        empty_bind_groups_.clear();
        for (const uint32_t id : ids) {
            device_.defer_destroy_resource(id);
        }
    }
    device_.detach_bridge(this);
    g_live_command_lists.fetch_sub(1, std::memory_order_relaxed);
}

void CommandListBridge::invalidate_resources_locked(std::span<const uint32_t> resource_ids,
                                                    std::span<const uint32_t> view_ids,
                                                    std::vector<uint32_t>* retired_bgs) {
    if (resource_ids.empty() && view_ids.empty()) {
        return;
    }
    std::vector<uint32_t> local_retired;
    std::vector<uint64_t> evicted_hashes;
    bind_groups_.evict_resources_and_views(resource_ids, view_ids, &local_retired, &evicted_hashes);
    for (const uint64_t h : evicted_hashes) {
        invalidated_key_hashes_.insert(h);
    }
    auto contains = [](std::span<const uint32_t> ids, uint32_t id) noexcept -> bool {
        if (id == 0) {
            return false;
        }
        for (const uint32_t candidate : ids) {
            if (candidate == id) {
                return true;
            }
        }
        return false;
    };
    // The filters skip the scans when no id can match (see InvalidationIdFilter); a full scan rebuilds
    // its filter from what remains.
    if (!redirects_.empty() && redirect_filter_.may_hold_any(resource_ids, {})) {
        redirect_filter_.clear();
        for (auto it = redirects_.begin(); it != redirects_.end();) {
            const bool match = std::any_of(it->second.begin(), it->second.end(), [&](const StorageRedirect& r) {
                return contains(resource_ids, r.texture) || contains(resource_ids, r.intermediate);
            });
            if (match) {
                if (it->first != 0 &&
                    std::find(local_retired.begin(), local_retired.end(), it->first) == local_retired.end()) {
                    local_retired.push_back(it->first);
                }
                it = redirects_.erase(it);
            } else {
                note_redirect_ids(it->second);
                ++it;
            }
        }
    }
    for (const uint32_t bg_id : local_retired) {
        redirects_.erase(bg_id);
        scoped_bind_groups_.erase(bg_id);
        for (BoundGroup& b : render_groups_) {
            if (b.bind_group_id == bg_id) {
                b.bind_group_id = 0;
                b.dynamic_offsets.clear();
            }
        }
        for (BoundGroup& b : compute_groups_) {
            if (b.bind_group_id == bg_id) {
                b.bind_group_id = 0;
                b.dynamic_offsets.clear();
            }
        }
        for (MemoryBindGroup& m : memory_groups_) {
            if (m.built.gpu_id == bg_id) {
                m.invalidated = true;
            }
        }
        ++stats_.bind_groups_invalidated;
        invalidated_bgs_to_retire_.push_back(bg_id);
        if (retired_bgs != nullptr) {
            retired_bgs->push_back(bg_id);
        }
    }
    if (memory_groups_.empty() || !memory_filter_.may_hold_any(resource_ids, view_ids)) {
        return;
    }
    memory_filter_.clear();
    for (MemoryBindGroup& m : memory_groups_) {
        note_memory_group_ids(m.built);  // an invalidated group is still refreshed (and kept) later
        if (m.invalidated) {
            continue;
        }
        bool stale = false;
        for (const d3d::BindGroupEntry& e : m.built.entries) {
            if (contains(resource_ids, e.resource_id) || contains(resource_ids, e.view.resource_id) ||
                contains(resource_ids, e.view.counter_resource_id)) {
                stale = true;
                break;
            }
        }
        if (!stale && !view_ids.empty()) {
            for (const uint32_t vid : m.built.view_ids) {
                if (contains(view_ids, vid)) {
                    stale = true;
                    break;
                }
            }
        }
        if (stale) {
            m.invalidated = true;
        }
    }
}

void CommandListBridge::note_memory_group_ids(const d3d::CachedBindGroup& built) noexcept {
    for (const d3d::BindGroupEntry& e : built.entries) {
        memory_filter_.set(e.resource_id, false);
        memory_filter_.set(e.view.resource_id, false);
        memory_filter_.set(e.view.counter_resource_id, false);
    }
    for (const uint32_t vid : built.view_ids) {
        memory_filter_.set(vid, true);
    }
}

void CommandListBridge::note_redirect_ids(const std::vector<StorageRedirect>& redirects) noexcept {
    for (const StorageRedirect& r : redirects) {
        redirect_filter_.set(r.texture, false);
        redirect_filter_.set(r.intermediate, false);
    }
}

bool CommandListBridge::try_invalidate_immediately(std::span<const uint32_t> resource_ids,
                                                   std::span<const uint32_t> view_ids,
                                                   std::vector<uint32_t>* retired_bgs) {
    if (t_building_bridge == this) {
        return false;
    }
    std::unique_lock lock(cache_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    invalidate_resources_locked(resource_ids, view_ids, retired_bgs);
    return true;
}

void CommandListBridge::queue_invalidation(std::span<const uint32_t> resource_ids,
                                           std::span<const uint32_t> view_ids) {
    std::lock_guard lock(pending_mutex_);
    pending_invalidated_resources_.insert(pending_invalidated_resources_.end(), resource_ids.begin(),
                                          resource_ids.end());
    pending_invalidated_views_.insert(pending_invalidated_views_.end(), view_ids.begin(), view_ids.end());
    pending_invalidation_epoch_.fetch_add(1, std::memory_order_release);
}

void CommandListBridge::drain_invalidations_locked() {
    const uint64_t epoch = pending_invalidation_epoch_.load(std::memory_order_acquire);
    if (epoch == seen_invalidation_epoch_) {
        return;
    }
    std::vector<uint32_t> resources;
    std::vector<uint32_t> views;
    {
        std::lock_guard lock(pending_mutex_);
        resources.swap(pending_invalidated_resources_);
        views.swap(pending_invalidated_views_);
        seen_invalidation_epoch_ = pending_invalidation_epoch_.load(std::memory_order_relaxed);
    }
    std::vector<uint32_t> retired_bgs;
    invalidate_resources_locked(resources, views, &retired_bgs);
    for (const uint32_t bg_id : retired_bgs) {
        device_.defer_destroy_resource(bg_id);
    }
}

void CommandListBridge::flush_invalidations() {
    std::lock_guard lock(cache_mutex_);
    drain_invalidations_locked();
}

size_t CommandListBridge::cached_bind_group_count() const {
    std::lock_guard lock(cache_mutex_);
    return bind_groups_.size();
}

void CommandListBridge::note_recorded_resource(uint32_t resource_id) {
    if (resource_id == 0) {
        return;
    }
    {
        std::lock_guard lock(recorded_mutex_);
        recorded_resources_.insert(resource_id);
    }
    device_.note_resource_used(resource_id);
}

bool CommandListBridge::has_recorded_resource(uint32_t resource_id) const {
    if (resource_id == 0) {
        return false;
    }
    std::lock_guard lock(recorded_mutex_);
    return !executed_ && recorded_resources_.count(resource_id) != 0;
}

void CommandListBridge::clear_recorded_resources(bool executed) {
    std::lock_guard lock(recorded_mutex_);
    recorded_resources_.clear();
    executed_ = executed;
}

void CommandListBridge::collect_unexecuted_resources(std::unordered_set<uint32_t>* out) const {
    std::lock_guard lock(recorded_mutex_);
    if (executed_) {
        return;
    }
    out->insert(recorded_resources_.begin(), recorded_resources_.end());
}

void CommandListBridge::mark_executed_and_pin_in_frame() {
    std::vector<uint32_t> snapshot;
    {
        std::lock_guard lock(recorded_mutex_);
        executed_ = true;
        snapshot.assign(recorded_resources_.begin(), recorded_resources_.end());
    }
    for (const uint32_t id : snapshot) {
        device_.note_resource_used(id);
    }
}

void CommandListBridge::note_bind_group_used_locked(const d3d::CachedBindGroup& cached, uint32_t bind_group_id) {
    note_recorded_resource(bind_group_id);
    for (const d3d::BindGroupEntry& e : cached.entries) {
        if (e.resource_id != 0) {
            note_recorded_resource(e.resource_id);
        }
        if (e.view.resource_id != 0 && e.view.resource_id != e.resource_id) {
            note_recorded_resource(e.view.resource_id);
        }
        if (e.view.counter_resource_id != 0) {
            note_recorded_resource(e.view.counter_resource_id);
        }
    }
    for (const d3d::CachedBindGroup::Relocation& r : cached.relocations) {
        if (r.buffer_id != 0) {
            note_recorded_resource(r.buffer_id);
        }
    }
    if (const auto it = redirects_.find(bind_group_id); it != redirects_.end()) {
        for (const StorageRedirect& r : it->second) {
            note_recorded_resource(r.texture);
            note_recorded_resource(r.intermediate);
        }
    }
}

void CommandListBridge::Reset(GraphicsPipelineState* initial_state) {
    dispatch_written_.clear();
    rt_ranges_.clear();
    clamp_batch_.close();
    // O36: bound the per-list bind group cache. Its keys change per draw (ring-allocated root CBVs,
    // descriptor table bases), so it grew for the whole run, each entry a live WebGPU bind group,
    // until the browser renderer process ran out of memory (~215 s, #726). The previous recording
    // is finished here, so dropped bind groups are destroyed at the head of the new one (or
    // deferred past Present if still referenced in the current frame).
    clear_recorded_resources(false);
    {
        std::lock_guard lock(cache_mutex_);
        drain_invalidations_locked();
        std::vector<uint32_t> retired_bgs;
        retired_bgs.swap(invalidated_bgs_to_retire_);
        for (const uint32_t bg_id : retired_bgs) {
            device_.take_deferred_destroy(bg_id, encoder_);
        }
        std::vector<uint32_t> evicted;
        bind_groups_.end_epoch(kMaxCachedBindGroups, &evicted);
        for (const uint32_t id : evicted) {
            redirects_.erase(id);
            device_.destroy_or_defer_resource(id, encoder_);
        }
        render_groups_.clear();
        compute_groups_.clear();
        scoped_bind_groups_.clear();
        memory_groups_.clear();
        memory_filter_.clear();
    }
    tracker_.reset();
    root_args_.reset();
    resource_states_.reset();
    usage_scope_.clear();
    usage_conflicts_.clear();
    graphics_pso_ = nullptr;
    compute_pso_ = nullptr;
    vertex_buffers_ = {};
    vertex_dirty_ = 0;
    index_buffer_ = {};
    index_dirty_ = false;
    render_pass_open_ = false;
    compute_pass_valid_ = false;
    compute_pipeline_bound_ = 0;
    for (auto& point : root_uploads_) {
        for (RootConstantUpload& upload : point) {
            upload.valid = false;
        }
    }
    diagnostics_ = 0;
    stats_ = BridgeStats{};
    open_ = true;
    if (initial_state != nullptr) {
        SetPipelineState(initial_state);
    }
}

bool CommandListBridge::writable() noexcept {
    if (!open_) {
        diagnostics_ |= BridgeDiag::CommandListClosed;
        return false;
    }
    return true;
}

void CommandListBridge::Close() {
    clamp_batch_.close();
    if (!writable()) {
        return;
    }
    tracker_.close();
    drain_events();
    {
        std::lock_guard lock(cache_mutex_);
        drain_invalidations_locked();
    }
    open_ = false;
}

void CommandListBridge::Present() {
    if (open_) {
        tracker_.flush_pending_clears();
        tracker_.end_render_pass();
        drain_events();
    }
    {
        std::lock_guard lock(cache_mutex_);
        drain_invalidations_locked();
        invalidated_bgs_to_retire_.clear();
    }
    clear_recorded_resources(true);
    device_.present(encoder_);
}

void CommandListBridge::SetPipelineState(GraphicsPipelineState* state) {
    if (!writable()) {
        return;
    }
    graphics_pso_ = state;
    mesh_pso_ = nullptr;
    tracker_.SetPipelineState(state != nullptr ? &state->pso_ : nullptr);
}

void CommandListBridge::SetMeshPipelineState(MeshPipelineState* state) {
    if (!writable()) {
        return;
    }
    graphics_pso_ = state != nullptr ? state->draw_.get() : nullptr;
    mesh_pso_ = state;
    tracker_.SetPipelineState(graphics_pso_ != nullptr ? &graphics_pso_->pso_ : nullptr);
}

void CommandListBridge::SetComputePipelineState(ComputePipelineState* state) {
    if (!writable()) {
        return;
    }
    compute_pso_ = state;
}

void CommandListBridge::OMSetRenderTargets(uint32_t num_render_target_descriptors,
                                           const d3d::CpuDescriptorHandle* render_target_descriptors,
                                           bool rts_single_handle_to_descriptor_range,
                                           const d3d::CpuDescriptorHandle* depth_stencil_descriptor) {
    if (!writable()) {
        return;
    }
    std::array<uint64_t, pso::kMaxColorTargets> handles{};
    const uint32_t n = std::min(num_render_target_descriptors, pso::kMaxColorTargets);
    if (render_target_descriptors != nullptr) {
        const uint32_t copy = rts_single_handle_to_descriptor_range ? std::min(n, 1u) : n;
        for (uint32_t i = 0; i < copy; ++i) {
            handles[i] = render_target_descriptors[i].ptr;
        }
    }
    const uint64_t dsv = depth_stencil_descriptor != nullptr ? depth_stencil_descriptor->ptr : 0;
    const uint64_t rtv_inc = device_.rtv_descriptor_increment();
    om_attachments_.fill(0);
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t h = rts_single_handle_to_descriptor_range
                               ? (handles[0] != 0 ? handles[0] + i * rtv_inc : 0u)
                               : handles[i];
        om_attachments_[i] = device_.attachment_texture(h, d3d::DescriptorKind::Rtv);
    }
    om_attachments_[pso::kMaxColorTargets] = device_.attachment_texture(dsv, d3d::DescriptorKind::Dsv);
    tracker_.OMSetRenderTargets(n, render_target_descriptors != nullptr ? handles.data() : nullptr,
                                rts_single_handle_to_descriptor_range,
                                depth_stencil_descriptor != nullptr ? &dsv : nullptr, device_);
}

void CommandListBridge::ClearRenderTargetView(d3d::CpuDescriptorHandle rtv,
                                              const float color_rgba[4],
                                              uint32_t num_rects,
                                              const pso::D3D12Rect* rects) {
    if (!writable()) {
        return;
    }
    pso::RenderTargetViewInfo info{};
    if (!device_.resolve_rtv(rtv.ptr, &info) || info.view_id == 0) {
        diagnostics_ |= BridgeDiag::UnknownTexture;
        return;
    }
    tracker_.ClearRenderTargetView(info, color_rgba, num_rects, rects);
    drain_events();
}

void CommandListBridge::ClearDepthStencilView(d3d::CpuDescriptorHandle dsv,
                                              uint32_t clear_flags,
                                              float depth,
                                              uint8_t stencil,
                                              uint32_t num_rects,
                                              const pso::D3D12Rect* rects) {
    if (!writable()) {
        return;
    }
    pso::DepthStencilViewInfo info{};
    if (!device_.resolve_dsv(dsv.ptr, &info) || info.view_id == 0) {
        diagnostics_ |= BridgeDiag::UnknownTexture;
        return;
    }
    tracker_.ClearDepthStencilView(info, clear_flags, depth, stencil, num_rects, rects);
    drain_events();
}

void CommandListBridge::OMSetBlendFactor(const float blend_factor[4]) {
    if (!writable()) {
        return;
    }
    if (blend_factor == nullptr) {
        tracker_.OMSetBlendFactor(nullptr);
        return;
    }
    const float clean[4] = {
        canonical_finite_float(blend_factor[0]),
        canonical_finite_float(blend_factor[1]),
        canonical_finite_float(blend_factor[2]),
        canonical_finite_float(blend_factor[3]),
    };
    tracker_.OMSetBlendFactor(clean);
}

void CommandListBridge::OMSetStencilRef(uint32_t stencil_ref) {
    if (writable()) {
        tracker_.OMSetStencilRef(stencil_ref);
    }
}

void CommandListBridge::RSSetViewports(uint32_t num_viewports, const pso::D3D12Viewport* viewports) {
    if (writable()) {
        tracker_.RSSetViewports(num_viewports, viewports);
    }
}

void CommandListBridge::RSSetScissorRects(uint32_t num_rects, const pso::D3D12Rect* rects) {
    if (writable()) {
        tracker_.RSSetScissorRects(num_rects, rects);
    }
}

void CommandListBridge::IASetPrimitiveTopology(pso::D3DPrimitiveTopology topology) {
    if (writable()) {
        tracker_.IASetPrimitiveTopology(topology);
    }
}

void CommandListBridge::IASetVertexBuffers(uint32_t start_slot,
                                           uint32_t num_views,
                                           const d3d12_xs::D3D12VertexBufferView* views) {
    if (!writable() || start_slot >= pso::kMaxVertexBuffers) {
        return;
    }
    const uint32_t count = std::min(num_views, pso::kMaxVertexBuffers - start_slot);
    std::array<uint32_t, pso::kMaxVertexBuffers> strides{};
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t slot = start_slot + i;
        vertex_buffers_[slot] = views != nullptr ? VertexBinding{views[i].buffer_id, views[i].offset} : VertexBinding{};
        strides[i] = views != nullptr ? views[i].stride_in_bytes : 0u;
        vertex_dirty_ |= 1u << slot;
    }
    tracker_.set_vertex_strides(start_slot, count, views != nullptr ? strides.data() : nullptr);
}

void CommandListBridge::IASetIndexBuffer(const d3d12_xs::D3D12IndexBufferView* view) {
    if (!writable()) {
        return;
    }
    index_buffer_ = view != nullptr ? *view : d3d12_xs::D3D12IndexBufferView{};
    index_dirty_ = true;
    if (view != nullptr && view->buffer_id != 0) {
        tracker_.set_index_format(to_wgpu_index_format(view->format));
    } else {
        tracker_.set_index_format(pso::WgpuIndexFormat::Undefined);
    }
}

d3d::Status CommandListBridge::SetDescriptorHeaps(uint32_t count, d3d::DescriptorHeap* const* heaps) {
    return writable() ? root_args_.set_descriptor_heaps(count, heaps) : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootSignature(const d3d::RootSignature* signature) {
    return writable() ? root_args_.set_root_signature(d3d::BindPoint::Graphics, signature) : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetComputeRootSignature(const d3d::RootSignature* signature) {
    return writable() ? root_args_.set_root_signature(d3d::BindPoint::Compute, signature) : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootDescriptorTable(uint32_t parameter_index, d3d::GpuDescriptorHandle base) {
    return writable() ? root_args_.set_descriptor_table(d3d::BindPoint::Graphics, parameter_index, base)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetComputeRootDescriptorTable(uint32_t parameter_index, d3d::GpuDescriptorHandle base) {
    return writable() ? root_args_.set_descriptor_table(d3d::BindPoint::Compute, parameter_index, base)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRoot32BitConstants(uint32_t parameter_index, uint32_t num_values,
                                                            const uint32_t* values, uint32_t dest_offset) {
    return writable() ? root_args_.set_root_constants(d3d::BindPoint::Graphics, parameter_index, num_values, values,
                                                      dest_offset)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetComputeRoot32BitConstants(uint32_t parameter_index, uint32_t num_values,
                                                           const uint32_t* values, uint32_t dest_offset) {
    return writable() ? root_args_.set_root_constants(d3d::BindPoint::Compute, parameter_index, num_values, values,
                                                      dest_offset)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootConstantBufferView(uint32_t parameter_index, uint64_t gpu_address) {
    return writable() ? root_args_.set_root_view(d3d::BindPoint::Graphics, parameter_index, d3d::RootParameterType::Cbv,
                                                 gpu_address)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetComputeRootConstantBufferView(uint32_t parameter_index, uint64_t gpu_address) {
    return writable() ? root_args_.set_root_view(d3d::BindPoint::Compute, parameter_index, d3d::RootParameterType::Cbv,
                                                 gpu_address)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootShaderResourceView(uint32_t parameter_index, uint64_t gpu_address) {
    return writable() ? root_args_.set_root_view(d3d::BindPoint::Graphics, parameter_index, d3d::RootParameterType::Srv,
                                                 gpu_address)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootUnorderedAccessView(uint32_t parameter_index, uint64_t gpu_address) {
    return writable() ? root_args_.set_root_view(d3d::BindPoint::Graphics, parameter_index, d3d::RootParameterType::Uav,
                                                 gpu_address)
                      : d3d::Status::InvalidArgument;
}

// ---- Event translation ----

void CommandListBridge::drain_events() {
    const std::vector<pso::StateEvent>& events = tracker_.events();
    if (events.empty()) {
        return;
    }
    for (const pso::StateEvent& e : events) {
        switch (e.type) {
        case pso::StateEventType::BeginRenderPass:
            emit_begin_render_pass(tracker_.passes()[e.index]);
            break;
        case pso::StateEventType::EndRenderPass:
            encoder_.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
            end_usage_scope();
            render_pass_open_ = false;
            // Fragment storage bindings redirected to intermediates: what the pass wrote goes back.
            for (const StorageRedirect& r : pass_redirects_) {
                if (r.access != static_cast<uint8_t>(GpuStorageTextureAccess::ReadOnly)) {
                    emit_storage_conversion(r, false);
                }
            }
            if (!pass_redirects_.empty()) {
                pass_redirects_.clear();
                compute_pipeline_bound_ = 0;
                compute_groups_.clear();
                compute_pass_valid_ = false;
            }
            break;
        case pso::StateEventType::SetPipeline:
            if (e.pipeline_key != nullptr && graphics_pso_ != nullptr) {
                GpuCreateRenderPipeline2Payload payload{};
                payload.id = e.pipeline_id;
                payload.vs_module_id = graphics_pso_->vs_module_id_;
                payload.fs_module_id = graphics_pso_->fs_module_id_;
                if (graphics_pso_->layout_ != nullptr) {
                    device_.ensure_layout_emitted(graphics_pso_->layout_, encoder_);
                    payload.pipeline_layout_id = graphics_pso_->layout_->id;
                }
                std::memcpy(&payload.key, e.pipeline_key, sizeof(payload.key));
                sanitize_render_pipeline_key(&payload.key, device_.options().features);
                device_.emit_device_object(payload.id, encoder_, [&](CommandEncoder& dst) {
                    dst.emit_packet(GpuOpcode::CreateRenderPipeline2, &payload, sizeof(payload));
                });
                ++stats_.pipelines_created;
                diagnostics_ |= graphics_pso_->diagnostics_;
                device_.record_graphics_pipeline(*e.pipeline_key, graphics_pso_->vs_module_id_,
                                                 graphics_pso_->fs_module_id_, graphics_pso_->layout_);
            } else if (e.pipeline_key == nullptr) {
                device_.note_pipeline_reached(e.pipeline_id);  // may be a pipeline the prewarm built
            }
            encoder_.set_pipeline(e.pipeline_id);
            break;
        case pso::StateEventType::SetViewport: {
            const GpuSetViewportPayload p{e.viewport.x, e.viewport.y, e.viewport.width,
                                          e.viewport.height, e.viewport.min_depth, e.viewport.max_depth};
            encoder_.emit_packet(GpuOpcode::SetViewport, &p, sizeof(p));
            break;
        }
        case pso::StateEventType::SetScissorRect: {
            const GpuSetScissorRectPayload p{e.scissor.x, e.scissor.y, e.scissor.width, e.scissor.height};
            encoder_.emit_packet(GpuOpcode::SetScissorRect, &p, sizeof(p));
            break;
        }
        case pso::StateEventType::SetBlendConstant: {
            const GpuSetBlendConstantPayload p{
                canonical_finite_float(e.blend_constant[0]),
                canonical_finite_float(e.blend_constant[1]),
                canonical_finite_float(e.blend_constant[2]),
                canonical_finite_float(e.blend_constant[3]),
            };
            encoder_.emit_packet(GpuOpcode::SetBlendConstant, &p, sizeof(p));
            break;
        }
        case pso::StateEventType::SetStencilReference: {
            const GpuSetStencilReferencePayload p{e.stencil_reference};
            encoder_.emit_packet(GpuOpcode::SetStencilReference, &p, sizeof(p));
            break;
        }
        case pso::StateEventType::ClearFallback:
            // Partial-rect clears need a worker-side clear draw that ABI v2 does not have yet.
            ++stats_.clear_fallbacks_dropped;
            diagnostics_ |= BridgeDiag::ClearFallbackDropped;
            break;
        }
    }
    tracker_.clear_events();
}

void CommandListBridge::emit_begin_render_pass(const pso::RenderPassDesc& pass) {
    GpuBeginRenderPassPayload p{};
    p.color_count = std::min(pass.color_count, kGpuMaxColorAttachments);
    usage_scope_.clear();
    scoped_bind_groups_.clear();
    for (uint32_t i = 0; i < p.color_count; ++i) {
        const pso::ColorAttachmentDesc& c = pass.color[i];
        const ViewRecord* v = c.view_id != 0 ? device_.view(c.view_id) : nullptr;
        if (v == nullptr || v->texture_id == 0) {
            continue;
        }
        GpuRenderPassColorAttachment& a = p.color[i];
        a.texture_id = v->texture_id;
        a.load_op = static_cast<uint8_t>(c.load_op);
        a.store_op = static_cast<uint8_t>(c.store_op);
        a.view = v->view;
        for (size_t ch = 0; ch < 4; ++ch) {
            a.clear_value[ch] = canonical_finite_float(c.clear_value[ch]);
        }
        usage_scope_.add(v->resource_id, v->range, v->usage);
        note_recorded_resource(v->texture_id);
        note_recorded_resource(v->resource_id);
    }
    if (pass.has_depth_stencil) {
        const pso::DepthStencilAttachmentDesc& d = pass.depth_stencil;
        if (const ViewRecord* v = device_.view(d.view_id); v != nullptr && v->texture_id != 0) {
            GpuRenderPassDepthStencilAttachment& a = p.depth_stencil;
            const bool has_depth = pso::has_depth_aspect(d.format);
            const bool has_stencil = pso::has_stencil_aspect(d.format);
            a.texture_id = v->texture_id;
            a.depth_load_op = static_cast<uint8_t>(d.depth_load_op);
            a.depth_store_op = static_cast<uint8_t>(d.depth_store_op);
            a.stencil_load_op = static_cast<uint8_t>(d.stencil_load_op);
            a.stencil_store_op = static_cast<uint8_t>(d.stencil_store_op);
            a.view = v->view;
            a.depth_clear_value = std::isfinite(d.depth_clear_value)
                                      ? std::clamp(d.depth_clear_value, 0.0f, 1.0f)
                                      : 1.0f;
            a.stencil_clear_value = d.stencil_clear_value & 0xFFu;
            a.depth_read_only = (has_depth && d.depth_read_only) ? 1 : 0;
            a.stencil_read_only = (has_stencil && d.stencil_read_only) ? 1 : 0;
            if (!d.depth_read_only) {
                // A colour twin of the written subresources is behind now.
                depth_twin_.note_depth_written(v->resource_id, v->view.base_mip_level, v->view.mip_level_count,
                                               v->view.base_array_layer, v->view.array_layer_count);
            }
            if (has_stencil && !d.stencil_read_only) {
                stencil_twin_.note_depth_written(v->resource_id, v->view.base_mip_level, v->view.mip_level_count,
                                                 v->view.base_array_layer, v->view.array_layer_count);
            }
            usage_scope_.add(v->resource_id, v->range, v->usage);
            note_recorded_resource(v->texture_id);
            note_recorded_resource(v->resource_id);
        }
    }
    encoder_.emit_packet(GpuOpcode::BeginRenderPass, &p, sizeof(p));
    render_pass_open_ = true;
    ++stats_.render_passes;
    invalidate_pass_bindings();
    compute_pass_valid_ = false;
}

void CommandListBridge::invalidate_pass_bindings() {
    render_groups_.clear();
    vertex_dirty_ = 0;
    for (uint32_t slot = 0; slot < pso::kMaxVertexBuffers; ++slot) {
        if (vertex_buffers_[slot].buffer_id != 0) {
            vertex_dirty_ |= 1u << slot;
        }
    }
    index_dirty_ = index_buffer_.buffer_id != 0;
}

void CommandListBridge::end_usage_scope() {
    const size_t before = usage_conflicts_.size();
    usage_scope_.find_conflicts(&usage_conflicts_);
    if (usage_conflicts_.size() != before) {
        diagnostics_ |= BridgeDiag::UsageConflict;
        stats_.usage_conflicts += usage_conflicts_.size() - before;
    }
    usage_scope_.clear();
    scoped_bind_groups_.clear();
}

// ---- Bind groups ----

GpuTextureViewDesc CommandListBridge::texture_view_desc(const d3d::ViewDescriptor& v) {
    namespace V = d3d::ViewDimension;
    GpuTextureViewDesc d{};
    const uint32_t features = device_.options().features;
    pso::WgpuTextureAspect aspect = pso::WgpuTextureAspect::All;
    TextureInfo tex{};
    const bool known = device_.texture_info(v.resource_id, &tex);
    // Views of a SampledOnly texture map the same way, so they name the format it was created with.
    const pso::FormatUsage usage = known && tex.sampled_only ? pso::FormatUsage::SampledOnly : pso::FormatUsage::Sampled;
    const bool uav = v.kind == d3d::DescriptorKind::Uav;
    if (!uav && known) {
        const ResolvedViewFormat resolved = resolve_sampled_view_format(
            tex.format, to_dxgi(v.format), tex.sampled_only, features, device_.is_depth_resource(v.resource_id));
        pso::WgpuTextureFormat fmt = resolved.format;
        // Reconcile against the host texture's actual created format (e.g. R11G11B10_FLOAT or
        // RGBA8_SNORM render targets promoted to rgba16float cannot take a rg11b10ufloat view;
        // sRGB textures lowered to their linear sibling for STORAGE_BINDING must emit an
        // explicit *-srgb view format when the guest SRV leaves format as UNKNOWN).
        if (!resolved.depth_sibling) {
            const pso::WgpuTextureFormat host_fmt = host_texture_format(tex, features);
            if (fmt == pso::WgpuTextureFormat::Undefined) {
                const pso::WgpuTextureFormat guest_fmt =
                    pso::map_texture_format(tex.format, usage, features).format;
                if (host_fmt != pso::WgpuTextureFormat::Undefined &&
                    guest_fmt != pso::WgpuTextureFormat::Undefined &&
                    guest_fmt != host_fmt &&
                    is_compatible_view_format(host_fmt, guest_fmt)) {
                    fmt = guest_fmt;
                }
            } else if (host_fmt != pso::WgpuTextureFormat::Undefined &&
                       !is_compatible_view_format(host_fmt, fmt)) {
                fmt = host_fmt;
            }
        }
        d.format = static_cast<uint8_t>(fmt);
        aspect = resolved.aspect;
    } else if (v.format != 0) {
        d.format = texture_format_code(to_dxgi(v.format), usage, features, &aspect);
    } else if (known) {
        texture_format_code(tex.format, usage, features, &aspect);  // aspect of default views
    } else {
        diagnostics_ |= BridgeDiag::UnknownTexture;
    }
    d.aspect = static_cast<uint8_t>(aspect);
    bool supported = true;
    d.dimension = static_cast<uint8_t>(to_view_dimension(v.dimension, &supported));
    if (!supported) {
        diagnostics_ |= BridgeDiag::UnsupportedViewDimension;
    }
    d.base_mip_level = v.first_mip;
    d.mip_level_count = uav ? 1 : (v.mip_levels == 0xffffu ? 0 : v.mip_levels);
    // G66 partly-held mip chains: a sampled view starts at the first held level (and at the view's
    // min-LOD clamp), so the shader never samples a level the title did not stream in.
    if (v.kind == d3d::DescriptorKind::Srv && known && (tex.resident_mip_floor != 0 || v.min_lod_clamp > 0.0f) &&
        clamp_view_to_resident_mips(tex.mip_levels, tex.resident_mip_floor, v.min_lod_clamp, &d.base_mip_level,
                                    &d.mip_level_count)) {
        ++stats_.mip_clamped_binds;
    }
    if (v.kind == d3d::DescriptorKind::Srv && known && tex.host_mip_offset != 0) {
        // The host texture was re-created at the capped size: guest level N is host level N - offset.
        shift_view_to_host_mips(tex.mip_levels, tex.host_mip_offset, &d.base_mip_level, &d.mip_level_count);
    }
    switch (v.dimension) {
    case V::Texture1D:
    case V::Texture2D:
    case V::Texture2DMs:
        d.array_layer_count = 1;
        break;
    case V::Texture1DArray:
    case V::Texture2DArray:
    case V::Texture2DMsArray:
        d.base_array_layer = v.first_array_slice;
        d.array_layer_count = v.array_size == 0xFFFFFFFFu ? 0u : v.array_size;
        break;
    case V::TextureCube:
        d.array_layer_count = 6;
        break;
    case V::TextureCubeArray:
        d.base_array_layer = v.first_array_slice;  // First2DArrayFace
        d.array_layer_count =
            (v.array_size == 0xFFFFFFFFu || v.array_size > 0xFFFFFFFFu / 6u) ? 0u : v.array_size * 6u;  // NumCubes
        break;
    default:  // default view / 3D: whole texture
        break;
    }
    if (v.kind == d3d::DescriptorKind::Srv && v.component_mapping != d3d::kDefaultShader4ComponentMapping) {
        diagnostics_ |= BridgeDiag::ComponentMappingIgnored;
    }
    return d;
}

uint32_t CommandListBridge::upload_root_constants(d3d::BindPoint bind_point, uint32_t parameter_index) {
    if (parameter_index >= d3d::kMaxRootParameters) {
        return 0;
    }
    if (const RootConstantOverride& o = root_override_[static_cast<uint32_t>(bind_point) & 1u][parameter_index];
        o.valid) {
        return o.offset;  // the current command's slot of an ExecuteIndirect with constant arguments
    }
    const std::span<const uint32_t> values = root_args_.root_constants(bind_point, parameter_index);
    RootConstantUpload& upload = root_uploads_[static_cast<uint32_t>(bind_point) & 1u][parameter_index];
    if (upload.valid && std::equal(values.begin(), values.end(), upload.values.begin(), upload.values.end())) {
        return upload.offset;
    }
    const uint32_t buffer_id = device_.ensure_root_constant_ring(encoder_);
    // The whole 256-byte ring slot is written (zero past the values): it is what the binding
    // covers (see the RootConstants case of to_bind_entry).
    const uint32_t bytes = align_up(std::max<uint32_t>(static_cast<uint32_t>(values.size()) * 4u, 1u),
                                    kRootConstantAlignment);
    bool wrapped = false;
    const uint32_t offset = device_.allocate_root_constants(bytes, &wrapped);
    if (wrapped) {
        diagnostics_ |= BridgeDiag::RootConstantRingWrapped;
    }
    const uint32_t words = bytes / 4u;
    if (words <= 64u) {
        uint32_t stack_data[64] = {};
        std::copy(values.begin(), values.end(), stack_data);
        encoder_.write_buffer(buffer_id, offset, stack_data, bytes);
    } else {
        std::vector<uint32_t> data(words, 0u);
        std::copy(values.begin(), values.end(), data.begin());
        encoder_.write_buffer(buffer_id, offset, data.data(), bytes);
    }
    upload.values.assign(values.begin(), values.end());
    upload.offset = offset;
    upload.valid = true;
    ++stats_.root_constant_uploads;
    return offset;
}

bool CommandListBridge::emit_bind_group(d3d::BindPoint /*bind_point*/,
                                        const StateBridgeDevice::GroupLayoutRecord& group,
                                        d3d::CachedBindGroup* cached,
                                        const d3d::BindGroupPlan* plan) {
    if (cached->entries.size() != group.entries.size()) {
        diagnostics_ |= BridgeDiag::BindGroupBuildFailed;
        return false;
    }
    std::vector<GpuBindGroupEntry2>& entries = scratch_gpu_entries_;
    gpu_entries(group, *cached, encoder_, &entries);
    cached->relocations = pending_relocations_;
    cached->relocation_epoch = device_.options().relocation_epoch ? device_.options().relocation_epoch() : 0;
    cached->retire_epoch = device_.retire_epoch();
    cached->residency_epoch = device_.residency_epoch();
    cached->residency_key = residency_key(*cached);
    cached->view_ids = pending_view_ids_;
    cached->null_write_entries = pending_null_writes_;
    bind_groups_.note_group_ids(*cached);
    GpuCreateBindGroup2Payload header{};
    header.id = device_.allocate_id();
    header.layout_id = group.id;
    header.entry_count = static_cast<uint32_t>(entries.size());
    header.flags = (group.header.flags & GpuLayoutFlags::BindingArrays) != 0 ? GpuBindGroupFlags::BindingArrays : 0u;
    const uint32_t entry_bytes = static_cast<uint32_t>(entries.size() * sizeof(GpuBindGroupEntry2));
    encoder_.emit_packet(GpuOpcode::CreateBindGroup2, &header, sizeof(header), entries.data(), entry_bytes);
    cached->gpu_id = header.id;
    ++stats_.bind_groups_created;
    if (!pending_redirects_.empty()) {
        stats_.storage_views_converted += pending_redirects_.size();
        note_redirect_ids(pending_redirects_);
        redirects_[header.id] = std::move(pending_redirects_);
        pending_redirects_.clear();
    }
    if (cached->from_memory && plan != nullptr) {
        note_memory_group(group, *plan, *cached, encoder_.size() - entry_bytes);
    }
    return true;
}

bool CommandListBridge::binds_retired(d3d::CachedBindGroup* cached) const {
    const uint64_t epoch = device_.retire_epoch();
    if (epoch == cached->retire_epoch) {
        return false;
    }
    for (const d3d::BindGroupEntry& e : cached->entries) {
        if ((e.kind == d3d::BindResourceKind::TextureView || e.kind == d3d::BindResourceKind::Buffer) &&
            e.resource_id != 0 && device_.retired(e.resource_id)) {
            return true;
        }
    }
    cached->retire_epoch = epoch;
    return false;
}

uint64_t CommandListBridge::residency_key(const d3d::CachedBindGroup& cached) const {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const d3d::BindGroupEntry& e : cached.entries) {
        if (e.kind != d3d::BindResourceKind::TextureView || e.resource_id == 0 ||
            e.view.kind != d3d::DescriptorKind::Srv) {
            continue;
        }
        TextureInfo tex{};
        if (!device_.texture_info(e.resource_id, &tex) || (tex.resident_mip_floor == 0 && tex.host_mip_offset == 0)) {
            continue;
        }
        h = (h ^ e.resource_id) * 0x100000001b3ull;
        h = (h ^ tex.resident_mip_floor) * 0x100000001b3ull;
        h = (h ^ (uint64_t{tex.host_mip_offset} + 0x9e3779b9ull)) * 0x100000001b3ull;
    }
    return h;
}

bool CommandListBridge::residency_changed(d3d::CachedBindGroup* cached) const {
    const uint64_t epoch = device_.residency_epoch();
    if (epoch == cached->residency_epoch) {
        return false;
    }
    if (residency_key(*cached) != cached->residency_key) {
        return true;  // emit_bind_group records the new epoch and key
    }
    cached->residency_epoch = epoch;
    return false;
}

bool CommandListBridge::relocations_moved(d3d::CachedBindGroup* cached) {
    const BridgeOptions& options = device_.options();
    if (cached->relocations.empty() || !options.relocation_epoch) {
        return false;
    }
    const uint64_t epoch = options.relocation_epoch();
    if (epoch == cached->relocation_epoch) {
        return false;
    }
    // Resolve each relocation again (a hit also marks the home used, so the LRU keeps it).
    bool moved = false;
    for (const d3d::CachedBindGroup::Relocation& r : cached->relocations) {
        uint32_t id = 0;
        uint64_t offset = 0;
        const bool ok = r.storage ? options.relocate_storage && options.relocate_storage(r.va, r.size, &id, &offset)
                                  : options.relocate_uniform &&
                                        options.relocate_uniform(r.va, r.size, r.binding_size, &id, &offset);
        if (!ok || id != r.buffer_id || offset != r.offset) {
            moved = true;
            break;
        }
    }
    if (!moved) {
        cached->relocation_epoch = options.relocation_epoch();
    }
    return moved;
}

void CommandListBridge::note_memory_group(const StateBridgeDevice::GroupLayoutRecord& group,
                                          const d3d::BindGroupPlan& plan,
                                          const d3d::CachedBindGroup& cached,
                                          size_t entries_offset) {
    for (MemoryBindGroup& m : memory_groups_) {
        if (m.built.gpu_id == cached.gpu_id) {
            if (entries_offset != kNoPacket) {
                m.entries_offset = entries_offset;
                m.built = cached;
                note_memory_group_ids(m.built);
                // Keep the home list in step with the entries (binds_home stays as it was).
                std::vector<std::pair<uint32_t, uint64_t>> homes;
                m.binds_home = binds_relocation_home(m.built, &homes) || m.binds_home;
                m.homes = std::move(homes);
            }
            return;
        }
    }
    note_memory_group_ids(cached);
    MemoryBindGroup m{entries_offset, &group, plan, cached};
    m.binds_home = binds_relocation_home(cached, &m.homes);
    m.relocation_epoch = cached.relocation_epoch;
    memory_groups_.push_back(std::move(m));
    ++stats_.memory_bind_groups;
}

bool CommandListBridge::binds_relocation_home(const d3d::CachedBindGroup& cached,
                                              std::vector<std::pair<uint32_t, uint64_t>>* homes) const {
    const BridgeOptions& options = device_.options();
    if (homes != nullptr) {
        homes->clear();
    }
    if (!options.relocation_home) {
        return false;
    }
    bool any = false;
    for (const d3d::BindGroupEntry& e : cached.entries) {
        if (e.kind == d3d::BindResourceKind::Buffer && e.resource_id != 0 && options.relocation_home(e.resource_id)) {
            any = true;
            if (homes == nullptr) {
                break;
            }
            homes->emplace_back(e.resource_id, e.offset);
        }
    }
    return any;
}

uint32_t CommandListBridge::refresh_memory_bind_groups(CommandEncoder& prelude) {
    mark_executed_and_pin_in_frame();
    const d3d::DescriptorSlotSource* source = device_.options().slot_source;
    if (source == nullptr) {
        return 0;
    }
    std::lock_guard lock(cache_mutex_);
    drain_invalidations_locked();
    const BridgeOptions& options = device_.options();
    // Read before any slot is decoded: a home handed to another range meanwhile is seen next time.
    const uint64_t relocation_epoch = options.relocation_epoch ? options.relocation_epoch() : 0;
    uint32_t refreshed = 0;
    for (MemoryBindGroup& m : memory_groups_) {
        // #663: a home bound straight from descriptor memory may have been handed to another range
        // since these entries were resolved (unchanged slot words do not show it): resolve again.
        bool force_rebuild = m.invalidated;
        if (!force_rebuild && m.binds_home && m.relocation_epoch != relocation_epoch) {
            // Only a home this group binds being handed on (or evicted) since its epoch changes what
            // its unchanged slot words resolve to; other ranges' evictions leave it as built.
            force_rebuild = !options.storage_homes_unmoved ||
                            !options.storage_homes_unmoved(m.homes.data(), m.homes.size(), m.relocation_epoch);
        }
        if (!force_rebuild) {
            TextureInfo tmp{};
            for (const d3d::BindGroupEntry& e : m.built.entries) {
                if (e.kind == d3d::BindResourceKind::TextureView && e.resource_id != 0 &&
                    !device_.texture_info(e.resource_id, &tmp)) {
                    force_rebuild = true;
                    break;
                }
            }
        }
        bool changed = false;
        {
            BuildingBridgeGuard guard(this);
            changed = d3d::refresh_table_entries(device_.heaps(), *source, m.plan,
                                                 &StateBridgeDevice::gpu_va_trampoline, &device_, &m.built,
                                                 force_rebuild);
        }
        m.relocation_epoch = relocation_epoch;
        if (changed) {
            m.invalidated = false;
            m.binds_home = binds_relocation_home(m.built, &m.homes);
            std::vector<GpuBindGroupEntry2>& entries = scratch_gpu_entries_;
            gpu_entries(*m.group, m.built, prelude, &entries);
            m.built.view_ids = pending_view_ids_;
            m.built.null_write_entries = pending_null_writes_;
            note_memory_group_ids(m.built);
            // The conversions were recorded with the dispatches: a redirect the recording did not
            // make has none, so that binding is bound null as before (#482).
            const auto recorded = redirects_.find(m.built.gpu_id);
            for (const StorageRedirect& r : pending_redirects_) {
                const bool known =
                    recorded != redirects_.end() &&
                    std::any_of(recorded->second.begin(), recorded->second.end(),
                                [&](const StorageRedirect& o) { return o.same_target(r); });
                if (!known && r.entry < entries.size()) {
                    entries[r.entry].resource_id = 0;
                    entries[r.entry].view = GpuTextureViewDesc{};
                    diagnostics_ |= BridgeDiag::StorageViewNoUsage;
                    ++stats_.storage_views_format_mismatch;
                }
            }
            pending_redirects_.clear();
            const size_t bytes = entries.size() * sizeof(GpuBindGroupEntry2);
            if (m.entries_offset != kNoPacket) {
                encoder_.patch(m.entries_offset, entries.data(), bytes);
            } else {
                // Created by an earlier recording of this list: create it again under the same id.
                GpuCreateBindGroup2Payload header{};
                header.id = m.built.gpu_id;
                header.layout_id = m.group->id;
                header.entry_count = static_cast<uint32_t>(entries.size());
                header.flags = (m.group->header.flags & GpuLayoutFlags::BindingArrays) != 0 ? GpuBindGroupFlags::BindingArrays : 0u;
                prelude.emit_packet(GpuOpcode::CreateBindGroup2, &header, sizeof(header), entries.data(),
                                    static_cast<uint32_t>(bytes));
            }
            ++refreshed;
            ++stats_.memory_bind_groups_refreshed;
        }
        note_bind_group_used_locked(m.built, m.built.gpu_id);
        sync_bound_textures(*m.group, m.built, prelude);
    }
    drain_invalidations_locked();
    return refreshed;
}

void CommandListBridge::sync_bound_textures(const StateBridgeDevice::GroupLayoutRecord& group,
                                            const d3d::CachedBindGroup& cached,
                                            CommandEncoder& target) {
    if (!device_.options().sync_texture_memory) {
        return;
    }
    for (size_t i = 0; i < cached.entries.size() && i < group.joins.size(); ++i) {
        const d3d::BindGroupEntry& e = cached.entries[i];
        if (e.kind == d3d::BindResourceKind::TextureView && e.resource_id != 0 &&
            (e.view.kind == d3d::DescriptorKind::Srv ||
             group.joins[i].register_class == d3d::DescriptorRangeType::Srv)) {
            if (device_.options().sync_texture_memory(e.resource_id, target)) {
                ++stats_.texture_memory_syncs;
            }
        }
    }
}

// Depth and depth-stencil formats: WebGPU cannot bind them as storage textures.
static bool is_depth_stencil_format(pso::DxgiFormat f) {
    return is_depth_stencil_dxgi_format(f);
}

void CommandListBridge::gpu_entries(const StateBridgeDevice::GroupLayoutRecord& group,
                                    const d3d::CachedBindGroup& cached,
                                    CommandEncoder& side,
                                    std::vector<GpuBindGroupEntry2>* result) {
    std::vector<GpuBindGroupEntry2>& entries = *result;
    entries.assign(cached.entries.size(), GpuBindGroupEntry2{});
    pending_redirects_.clear();
    pending_relocations_.clear();
    pending_view_ids_.clear();
    const auto relocated = [this](size_t entry, bool storage, uint64_t va, uint64_t size, uint64_t binding_size,
                                  uint32_t id, uint64_t offset) {
        pending_relocations_.push_back({static_cast<uint32_t>(entry), storage, va, size, binding_size, id, offset});
    };
    for (size_t i = 0; i < entries.size() && i < group.entries.size(); ++i) {
        const d3d::BindGroupEntry& e = cached.entries[i];
        const GpuBindGroupLayoutEntry& layout = group.entries[i];
        GpuBindGroupEntry2& out = entries[i];
        out.binding = layout.binding;
        if (i < group.joins.size()) {
            out.array_index = static_cast<uint16_t>(group.joins[i].array_index);  // 0 outside binding arrays
        }
        switch (e.kind) {
        case d3d::BindResourceKind::Buffer: {
            out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
            if (e.resource_id != 0) {
                // Pin before the retired() check so a concurrent Resource::~Resource either sees
                // the pin (and defers destroy / preserves a deferred create) or we see retired().
                note_recorded_resource(e.resource_id);
                if (device_.retired(e.resource_id)) {
                    diagnostics_ |= BridgeDiag::NullBinding;  // G66 (#691): destroyed buffer, see TextureView
                    ++stats_.retired_resources_nulled;
                    break;
                }
            }
            out.resource_id = e.resource_id;
            out.offset = e.offset;
            const bool uniform_layout =
                layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
                layout.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Uniform);
            // G66 (#691): a constant-buffer view with no size (a console descriptor of zero
            // records) would bind "to the end of the buffer": past WebGPU's 64 KiB uniform limit,
            // and at the view's 16-byte offset it skipped the relocation below and invalidated the
            // bind group (every draw using it). Bind the stage's block (or the 64 KiB a uniform
            // binding may cover) from the view's start, within the buffer.
            uint64_t view_size = e.size;
            if (uniform_layout && view_size == 0) {
                const uint64_t allocated = device_.buffer_allocated_size(e.resource_id);
                const uint64_t room = allocated > e.offset ? allocated - e.offset : 0;
                const uint64_t block = layout.min_binding_size != 0 ? align_up(layout.min_binding_size, 16u)
                                                                    : kMaxUniformBindingSize;
                view_size = std::min<uint64_t>({room, block, kMaxUniformBindingSize});
            }
            out.size = view_size;
            // WebGPU binds uniform buffers only at 256-byte aligned offsets; the title's constant
            // buffers are 16-byte aligned (Xbox). Relocate to a copy at an aligned offset.
            if (uniform_layout && e.offset % kUniformOffsetAlignment != 0) {
                const uint64_t base = device_.buffer_gpu_va(e.resource_id);
                uint32_t id = 0;
                uint64_t offset = 0;
                // A reflected stage (G62) gives the binding's minimum size: the home covers it,
                // zero past the view's bytes.
                const uint64_t home = std::max<uint64_t>(view_size, layout.min_binding_size);
                if (base != 0 && view_size != 0 && device_.options().relocate_uniform &&
                    device_.options().relocate_uniform(base + e.offset, view_size, home, &id, &offset)) {
                    relocated(i, false, base + e.offset, view_size, home, id, offset);
                    out.resource_id = id;
                    out.offset = offset;
                    // The relocated home is a whole multiple of 256 bytes (zero past the view).
                    out.size = align_up(home, kUniformOffsetAlignment);
                } else {
                    // No copy (GPU-only memory, no bytes): an unaligned uniform offset would
                    // invalidate the bind group and every draw using it. Bound null instead.
                    diagnostics_ |= BridgeDiag::UnalignedUniform | BridgeDiag::NullBinding;
                    out.resource_id = 0;
                    out.offset = 0;
                    out.size = 0;
                }
            } else if (layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
                       layout.binding_type == static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage) &&
                       e.offset % kUniformOffsetAlignment != 0) {
                // O36: storage bindings need a 256-byte aligned offset too (WebGPU's
                // minStorageBufferOffsetAlignment); a read-only view at any other offset is bound
                // as a relocated copy whose first byte is the view's (see relocate_storage).
                const uint64_t base = device_.buffer_gpu_va(e.resource_id);
                uint32_t id = 0;
                uint64_t offset = 0;
                // G66: a root SRV/UAV has no view size (D3D12 bounds it by the end of its
                // resource). The title's UI vertex stage reads the same root SRV both as a uniform
                // header and as read-only storage (vertex pulling, LEGAL NOTICES alpha mask): the
                // uniform was relocated, this binding was left at the unaligned offset and nulled.
                // Copy the memory that follows, up to the end of the buffer and at most 64 KiB
                // (the stage clamps its reads to arrayLength).
                uint64_t copy_size = view_size;
                if (copy_size == 0) {
                    const uint64_t allocated = device_.buffer_allocated_size(e.resource_id);
                    const uint64_t room = allocated > e.offset ? allocated - e.offset : 0;
                    copy_size = std::min<uint64_t>(room, kMaxUniformBindingSize);
                }
                if (base != 0 && copy_size != 0 && device_.options().relocate_storage &&
                    device_.options().relocate_storage(base + e.offset, copy_size, &id, &offset)) {
                    relocated(i, true, base + e.offset, copy_size, 0, id, offset);
                    out.resource_id = id;
                    out.offset = offset;
                    out.size = align_up(copy_size, 4u);
                    ++stats_.storage_relocated;
                } else {
                    diagnostics_ |= BridgeDiag::UnalignedStorage;
                }
            } else if (layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) && view_size != 0 &&
                       (layout.min_binding_size != 0
                            ? layout.min_binding_size > view_size
                            : layout.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Uniform) &&
                                  view_size % kUniformOffsetAlignment != 0)) {
                // Xbox constant-buffer views are 16-byte multiples, and a stage may declare a larger
                // uniform block than the view (on the console it reads whatever memory follows).
                // WebGPU needs the binding to cover the block: bind up to the stage's reflected
                // minimum (G62). Without one the block size is unknown here: bind the view through
                // to the rest of its buffer, at most the 64 KiB uniform binding limit (the memory
                // that follows, as on the console), at least the 256-byte minimum of PC D3D12.
                // Extend the range when the buffer has the room; otherwise bind a relocated copy
                // whose bytes past the view are zero (the zero-extension WebGPU needs, not data).
                const uint64_t target = layout.min_binding_size != 0 ? layout.min_binding_size
                                                                     : align_up(view_size, kUniformOffsetAlignment);
                // Room: the WebGPU buffer behind the view, including the zero padding a buffer
                // smaller than a uniform block is allocated with (Resource::initialize).
                const uint64_t base = device_.buffer_gpu_va(e.resource_id);
                const uint64_t allocated = device_.buffer_allocated_size(e.resource_id);
                const uint64_t room = allocated > e.offset ? allocated - e.offset : 0;
                if (base != 0 && room != 0) {
                    out.size = std::min<uint64_t>(target, room);
                    if (layout.min_binding_size == 0 && room >= target) {
                        out.size = std::min<uint64_t>(room, kMaxUniformBindingSize);
                    }
                }
                const bool uniform = layout.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Uniform);
                uint32_t id = 0;
                uint64_t offset = 0;
                if (uniform && room < target && base != 0 && device_.options().relocate_uniform &&
                    device_.options().relocate_uniform(base + e.offset, view_size, target, &id, &offset)) {
                    relocated(i, false, base + e.offset, view_size, target, id, offset);
                    out.resource_id = id;
                    out.offset = offset;
                    out.size = align_up(target, kUniformOffsetAlignment);
                    ++stats_.uniforms_zero_extended;
                }
            } else if (layout.type != static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
                       (e.view.kind == d3d::DescriptorKind::Cbv || e.view.kind == d3d::DescriptorKind::Srv) &&
                       e.size != 0 && e.size % kUniformOffsetAlignment != 0) {
                // A constant-buffer or read-only buffer view in a slot the conventional (root-
                // signature) layout does not describe as a buffer: the stage's translated code may
                // still read it as a uniform block whose size this layer does not know. Bind it
                // through to the rest of its buffer (at most 64 KiB, the memory that follows, as on
                // the console; the zero padding of a small buffer), at least 256 bytes; when the
                // buffer has less room, as a relocated copy zero past the view's bytes (what D3D12's
                // bounds-checked reads past a buffer view return).
                const uint64_t target = align_up(e.size, kUniformOffsetAlignment);
                const uint64_t base = device_.buffer_gpu_va(e.resource_id);
                const uint64_t allocated = device_.buffer_allocated_size(e.resource_id);
                const uint64_t room = allocated > e.offset ? allocated - e.offset : 0;
                uint32_t id = 0;
                uint64_t offset = 0;
                if (room >= target && e.offset % kUniformOffsetAlignment == 0) {
                    out.size = std::min<uint64_t>(room, kMaxUniformBindingSize);
                } else if (base != 0 && device_.options().relocate_uniform &&
                           device_.options().relocate_uniform(base + e.offset, e.size, target, &id, &offset)) {
                    relocated(i, false, base + e.offset, e.size, target, id, offset);
                    out.resource_id = id;
                    out.offset = offset;
                    out.size = target;
                    ++stats_.uniforms_zero_extended;
                }
            }
            // G66 (#691): WebGPU binds uniform and storage buffers only at 256-byte aligned
            // offsets; any other offset invalidates the bind group and every draw using it (a
            // zero-record console view at a 16-byte offset turned a whole capture black). What is
            // still unaligned here (no relocated copy) is bound null: a zero resource, as the
            // console reads a view without records.
            if (layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) && out.resource_id != 0 &&
                out.offset % kUniformOffsetAlignment != 0) {
                diagnostics_ |= (uniform_layout ? BridgeDiag::UnalignedUniform : BridgeDiag::UnalignedStorage) |
                                BridgeDiag::NullBinding;
                out.resource_id = 0;
                out.offset = 0;
                out.size = 0;
            }
            break;
        }
        case d3d::BindResourceKind::TextureView: {
            out.resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
            if (e.resource_id != 0) {
                note_recorded_resource(e.resource_id);
                if (device_.retired(e.resource_id)) {
                    // G66 / G70 (#691): a stale descriptor of a destroyed texture (the title released it
                    // and left the slot): binding it would invalidate the whole submit. Bound null.
                    diagnostics_ |= BridgeDiag::UnknownTexture | BridgeDiag::NullBinding;
                    ++stats_.retired_resources_nulled;
                    break;
                }
            }
            out.resource_id = e.resource_id;
            out.view = texture_view_desc(e.view);
            TextureInfo info{};
            const bool known = e.resource_id != 0 && device_.texture_info(e.resource_id, &info);
            if (layout.type == static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture)) {
                // A storage view needs a texture created with STORAGE_BINDING; binding one of a
                // texture without it would invalidate the whole command buffer. Bound null instead.
                // The texture's own format (or its sRGB sibling) must be the stage's storage format.
                const uint8_t own =
                    known ? static_cast<uint8_t>(host_texture_format(info, device_.options().features)) : 0u;
                const bool format_ok =
                    layout.storage_format == 0 || own == layout.storage_format ||
                    own == static_cast<uint8_t>(
                               pso::srgb_sibling(static_cast<pso::WgpuTextureFormat>(layout.storage_format)));
                const bool depth_texture = e.resource_id != 0 && known &&
                                           (is_depth_stencil_format(info.format) ||
                                            device_.is_depth_resource(e.resource_id));
                StorageRedirect twin;
                if (depth_texture && depth_twin_redirect(layout, e.resource_id, info, out.view, side, &twin)) {
                    // Backlog 35: WebGPU has no storage access on depth formats. The stage's
                    // writes go to a colour twin, copied into the depth plane after the dispatch.
                    twin.entry = i;
                    out.resource_id = twin.intermediate;
                    note_recorded_resource(twin.intermediate);
                    out.view = GpuTextureViewDesc{};
                    out.view.format = twin.storage_format;
                    out.view.dimension = twin.dimension;
                    out.view.base_mip_level = static_cast<uint16_t>(twin.mip);
                    out.view.mip_level_count = 1;
                    out.view.base_array_layer = twin.base_layer;
                    out.view.array_layer_count = twin.layer_count;
                    pending_redirects_.push_back(twin);
                    ++(twin.stencil_twin ? stats_.stencil_twin_bindings : stats_.depth_twin_bindings);
                } else if (depth_texture) {
                    ++stats_.storage_views_depth_untwinned;  // no twin possible: the writes are lost
                    out.resource_id = 0;
                    out.view = GpuTextureViewDesc{};
                    diagnostics_ |= BridgeDiag::StorageViewNoUsage;
                    ++stats_.storage_views_without_usage;
                } else if (StorageRedirect r;
                           e.resource_id != 0 && known && !info.storage && info.render_target &&
                           storage_redirect(layout, e.resource_id, info, own, out.view, side, true, &r)) {
                    // Renderable color/HDR texture whose host format lacks STORAGE_BINDING (e.g.
                    // Rg11b10Ufloat/Rgb10a2Unorm/R16Float/Rg16Float/Bgra8Unorm without tier1 or
                    // bgra8unorm-storage): write into a float/HDR storage intermediate and copy
                    // back via a fullscreen render pass instead of dropping the compute dispatch.
                    r.entry = i;
                    out.resource_id = r.intermediate;
                    note_recorded_resource(r.intermediate);
                    out.view.format = r.storage_format;
                    pending_redirects_.push_back(r);
                } else if (e.resource_id != 0 && (!known || !info.storage)) {
                    out.resource_id = 0;
                    out.view = GpuTextureViewDesc{};
                    diagnostics_ |= BridgeDiag::StorageViewNoUsage;
                    ++stats_.storage_views_without_usage;
                } else if (StorageRedirect r;
                           e.resource_id != 0 && !format_ok &&
                           storage_redirect(layout, e.resource_id, info, own, out.view, side, false, &r)) {
                    // #482: a storage binding whose format differs from the stage's declared
                    // storage format is bound to an intermediate texture in the declared format,
                    // converted into the real texture after the dispatch (and from it before, for
                    // read/read-write access).
                    r.entry = i;
                    out.resource_id = r.intermediate;
                    note_recorded_resource(r.intermediate);
                    out.view.format = r.storage_format;
                    pending_redirects_.push_back(r);
                } else if (e.resource_id != 0 && !format_ok) {
                    // WebGPU requires the storage format to be the texture's: another format
                    // invalidates the bind group and the whole command buffer. Bound null instead.
                    out.resource_id = 0;
                    out.view = GpuTextureViewDesc{};
                    diagnostics_ |= BridgeDiag::StorageViewNoUsage;
                    ++stats_.storage_views_format_mismatch;
                } else if (e.resource_id != 0 && layout.storage_format != 0 &&
                           out.view.format != layout.storage_format) {
                    // Storage views must match the stage's storage format (never sRGB or a
                    // reinterpreted typeless view format that createView would reject).
                    out.view.format = layout.storage_format;
                } else if (e.resource_id != 0 && layout.storage_format == 0 && own != 0 &&
                           out.view.format != 0 && out.view.format != own) {
                    // Conventional layout without an explicit storage format: keep the texture's
                    // own format so WebGPU createView does not fail on sRGB or typeless aliases.
                    out.view.format = own;
                }
            } else if (e.resource_id != 0 && !known) {
                // G70 (#691): texture was unregistered / destroyed; bind null instead of emitting
                // a bind group referencing a destroyed GPUTexture that fails queue.submit().
                out.resource_id = 0;
                out.view = GpuTextureViewDesc{};
                diagnostics_ |= BridgeDiag::UnknownTexture | BridgeDiag::NullBinding;
            }
            if (out.resource_id != 0) {
                ViewRecord vr{};
                vr.texture_id = out.resource_id;
                vr.resource_id = e.resource_id;
                vr.view = out.view;
                vr.range = d3d::view_subresource_range(e.view);
                vr.usage = d3d::view_usage(e.view);
                pending_view_ids_.push_back(device_.intern_view(vr));
            }
            break;
        }
        case d3d::BindResourceKind::Sampler: {
            out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Sampler);
            bool dropped = false;
            const d3d::SamplerDesc s =
                layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Sampler)
                    ? sampler_for_binding(e.sampler, layout.binding_type, &dropped)
                    : e.sampler;
            if (dropped) {
                diagnostics_ |= BridgeDiag::SamplerApproximated;
            }
            out.resource_id = device_.intern_sampler(s, side, &diagnostics_);
            break;
        }
        case d3d::BindResourceKind::RootConstants:
            out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
            out.resource_id = device_.ensure_root_constant_ring(side);
            // Bind the whole ring slot (256 bytes, zero past the values), like a constant-buffer
            // view padded to the PC D3D12 minimum: stages may declare the constants as a larger
            // uniform block, and WebGPU rejects a binding smaller than the shader's block.
            out.size = align_up(std::max<uint32_t>(static_cast<uint32_t>(e.size), 1u), kRootConstantAlignment);
            break;
        case d3d::BindResourceKind::Null:
            switch (static_cast<GpuBindingLayoutType>(layout.type)) {
            case GpuBindingLayoutType::Buffer:
                out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
                break;
            case GpuBindingLayoutType::Sampler:
                out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Sampler);
                break;
            default:
                out.resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
                break;
            }
            diagnostics_ |= BridgeDiag::NullBinding;
            break;
        }
    }
    // Writable storage bindings left null: the dispatch using this group is dropped (backlog 39).
    // Only for translator-manifest layouts (content_id != 0), which list the bindings the shader
    // uses: a conventional root-signature layout declares every slot, so a null there says nothing.
    pending_null_writes_ = 0;
    for (size_t i = 0; group.header.content_id != 0 && i < entries.size() && i < group.entries.size(); ++i) {
        if (entries[i].resource_id != 0) {
            continue;
        }
        const GpuBindGroupLayoutEntry& layout = group.entries[i];
        const bool tex = layout.type == static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture) &&
                         layout.binding_type != static_cast<uint8_t>(GpuStorageTextureAccess::ReadOnly);
        const bool buf = layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
                         layout.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Storage);
        if (tex || buf) {
            ++pending_null_writes_;
        }
    }
}

bool CommandListBridge::storage_redirect(const GpuBindGroupLayoutEntry& layout, uint32_t resource_id,
                                         const TextureInfo& info, uint8_t own_format, const GpuTextureViewDesc& view,
                                         CommandEncoder& side, bool allow_render_copy_out, StorageRedirect* out) {
    using D = GpuTextureViewDimension;
    const uint32_t features = device_.options().features;
    // Compute or fragment only: the conversions run as dispatches next to the stage's own (compute),
    // or around the render pass (fragment).
    // Vertex and mixed vertex/fragment bindings are converted the same way as fragment ones.
    const bool compute_only = (layout.visibility & ~GpuShaderStageBits::Compute) == 0;
    const bool render_only = (layout.visibility & GpuShaderStageBits::Compute) == 0;
    if (layout.visibility == 0 || (!compute_only && !render_only) || info.sample_count > 1) {
        return false;
    }
    const D dim = layout.view_dimension == static_cast<uint8_t>(D::Undefined) ? D::Dim2D
                                                                             : static_cast<D>(layout.view_dimension);
    const bool volume = info.dimension == GpuTextureDimension::Dim3D;
    uint8_t target_format = own_format;
    uint8_t inter_format = layout.storage_format;
    if (allow_render_copy_out) {
        if (!compute_only || volume || info.dimension != GpuTextureDimension::Dim2D ||
            (dim != D::Dim2D && dim != D::Dim2DArray)) {
            return false;
        }
        // D3D12 UAV writes raw linear bits even on *_SRGB resources: render back through the
        // linear sibling view format when available so gamma encoding is not applied twice.
        if (is_srgb_format(static_cast<pso::WgpuTextureFormat>(target_format))) {
            const pso::WgpuTextureFormat lin = pso::srgb_sibling(static_cast<pso::WgpuTextureFormat>(target_format));
            if (lin != pso::WgpuTextureFormat::Undefined) {
                target_format = static_cast<uint8_t>(lin);
            }
        }
        if (!renderable_color_storage_fallback(target_format, features)) {
            return false;
        }
        if (inter_format == 0) {
            inter_format = preferred_storage_intermediate_format(target_format);
        }
        if (!convertible_storage_format(inter_format, features)) {
            return false;
        }
    } else {
        if (!convertible_storage_format(own_format, features) ||
            !convertible_storage_format(layout.storage_format, features)) {
            return false;
        }
    }
    if ((dim == D::Dim3D) != volume || (dim != D::Dim3D && dim != D::Dim2D && dim != D::Dim2DArray) ||
        view.base_mip_level >= std::max(info.mip_levels, 1u)) {
        return false;
    }
    StorageRedirect r;
    r.texture = resource_id;
    r.format = target_format;
    r.storage_format = inter_format;
    r.dimension = static_cast<uint8_t>(dim);
    r.access = layout.binding_type;
    r.mip = view.base_mip_level;
    const uint32_t layers = volume ? 1u : std::max(info.depth_or_array_size, 1u);
    r.base_layer = volume ? 0u : view.base_array_layer;
    r.layer_count = dim == D::Dim2DArray ? (view.array_layer_count != 0 ? view.array_layer_count
                                                                        : layers - std::min(r.base_layer, layers))
                                         : 1u;
    if (r.layer_count == 0 || r.base_layer + r.layer_count > layers) {
        return false;
    }
    r.extent[0] = mip_extent(info.width, r.mip);
    r.extent[1] = mip_extent(info.height, r.mip);
    r.extent[2] = volume ? mip_extent(info.depth_or_array_size, r.mip) : r.layer_count;
    r.render_copy_out = allow_render_copy_out;
    r.intermediate = device_.intermediate_storage_texture(resource_id, info, inter_format, side);
    *out = r;
    return true;
}

bool CommandListBridge::depth_twin_redirect(const GpuBindGroupLayoutEntry& layout, uint32_t resource_id,
                                            const TextureInfo& info, const GpuTextureViewDesc& view,
                                            CommandEncoder& side, StorageRedirect* out) {
    using D = GpuTextureViewDimension;
    // Compute-only, a single-sample 2D texture (any mip level and array layers: tracked and copied
    // per subresource), stored as r32float or r32uint. The stencil plane has no twin (counted as
    // untwinned); volume and cube views have none either.
    const D dim = layout.view_dimension == static_cast<uint8_t>(D::Undefined) ? D::Dim2D
                                                                             : static_cast<D>(layout.view_dimension);
    if (layout.visibility == 0 || (layout.visibility & ~GpuShaderStageBits::Compute) != 0 ||
        info.sample_count > 1 || info.dimension != GpuTextureDimension::Dim2D ||
        !twin_storage_format_ok(layout.storage_format) || (dim != D::Dim2D && dim != D::Dim2DArray) ||
        view.base_mip_level >= std::max(info.mip_levels, 1u)) {
        return false;
    }
    const bool stencil = view.aspect == static_cast<uint8_t>(GpuTextureAspect::StencilOnly);
    // The stencil plane twins as an unsigned-integer storage texture only.
    if (stencil && layout.storage_format != static_cast<uint8_t>(pso::WgpuTextureFormat::R32Uint) &&
        layout.storage_format != static_cast<uint8_t>(pso::WgpuTextureFormat::R8Uint)) {
        return false;
    }
    const uint8_t depth_format =
        texture_format_code(info.format, pso::FormatUsage::DepthStencil, device_.options().features, nullptr);
    if (stencil ? !pso::has_stencil_aspect(static_cast<pso::WgpuTextureFormat>(depth_format))
                : !pso::has_depth_aspect(static_cast<pso::WgpuTextureFormat>(depth_format))) {
        return false;
    }
    StorageRedirect r;
    r.texture = resource_id;
    r.format = depth_format;
    r.storage_format = layout.storage_format;
    r.dimension = static_cast<uint8_t>(dim);
    r.access = layout.binding_type;
    r.mip = view.base_mip_level;
    const uint32_t layers = std::max(info.depth_or_array_size, 1u);
    r.base_layer = view.base_array_layer;
    r.layer_count = dim == D::Dim2DArray ? (view.array_layer_count != 0 ? view.array_layer_count
                                                                        : layers - std::min(r.base_layer, layers))
                                         : 1u;
    if (r.layer_count == 0 || r.base_layer + r.layer_count > layers) {
        return false;
    }
    r.extent[0] = mip_extent(info.width, r.mip);
    r.extent[1] = mip_extent(info.height, r.mip);
    r.extent[2] = r.layer_count;
    r.depth_twin = true;
    r.stencil_twin = stencil;
    r.depth_format = depth_format;
    r.intermediate = device_.intermediate_storage_texture(resource_id, info, layout.storage_format, side, stencil);
    *out = r;
    return true;
}

bool CommandListBridge::emit_stencil_twin_in(const StorageRedirect& r) {
    bool copied = false;
    for (uint32_t layer = r.base_layer; layer < r.base_layer + r.layer_count; ++layer) {
        if (!stencil_twin_.needs_copy_in(r.texture, r.mip, layer)) {
            continue;
        }
        const StateBridgeDevice::StencilTwinPipelines p =
            device_.stencil_twin_pipelines(r.depth_format, r.storage_format, encoder_);
        const uint32_t group = device_.depth_twin_group(p.in_layout, r.texture, r.intermediate, true,
                                                        r.storage_format, r.mip, layer, encoder_, true);
        encoder_.set_compute_pipeline(p.in_pipeline);
        const GpuSetBindGroup2Payload set{0u, group, 0u, 0u};
        encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
        note_recorded_resource(r.texture);
        note_recorded_resource(r.intermediate);
        note_recorded_resource(group);
        encoder_.dispatch_compute((r.extent[0] + 7u) / 8u, (r.extent[1] + 7u) / 8u, 1u);
        stencil_twin_.note_copied_in(r.texture, r.mip, layer);
        ++stats_.stencil_twin_copies_in;
        copied = true;
    }
    return copied;
}

void CommandListBridge::emit_stencil_twin_out(const StorageRedirect& r) {
    for (uint32_t layer = r.base_layer; layer < r.base_layer + r.layer_count; ++layer) {
        emit_stencil_twin_out_sub(r, r.mip, layer);
    }
}

void CommandListBridge::emit_stencil_twin_out_sub(const StorageRedirect& r, uint32_t mip, uint32_t layer) {
    if (!stencil_twin_.needs_copy_out(r.texture, mip, layer)) {
        return;
    }
    const StateBridgeDevice::StencilTwinPipelines p =
        device_.stencil_twin_pipelines(r.depth_format, r.storage_format, encoder_);
    const uint32_t group = device_.depth_twin_group(p.out_layout, r.texture, r.intermediate, false, r.storage_format,
                                                    mip, layer, encoder_, true);
    // The twin holds the whole subresource (it was copied in before the stage ran), so the pass
    // clears the stencil plane to 0 and the eight masked draws set every bit the twin holds.
    GpuBeginRenderPassPayload pass{};
    pass.color_count = 0;
    GpuRenderPassDepthStencilAttachment& a = pass.depth_stencil;
    a.texture_id = r.texture;
    a.depth_read_only = pso::has_depth_aspect(static_cast<pso::WgpuTextureFormat>(r.depth_format)) ? 1 : 0;
    a.stencil_load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
    a.stencil_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
    a.stencil_clear_value = 0;
    a.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
    a.view.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
    a.view.base_mip_level = static_cast<uint16_t>(mip);
    a.view.mip_level_count = 1;
    a.view.base_array_layer = layer;
    a.view.array_layer_count = 1;
    encoder_.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof(pass));
    ++stats_.render_passes;
    const GpuSetBindGroup2Payload set{0u, group, 0u, 0u};
    for (uint32_t bit = 0; bit < 8; ++bit) {
        encoder_.set_pipeline(p.out_pipeline[bit]);
        encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
        const GpuSetStencilReferencePayload ref{1u << bit};
        encoder_.emit_packet(GpuOpcode::SetStencilReference, &ref, sizeof(ref));
        encoder_.draw(3, 1, 0, 0);
        ++stats_.stencil_twin_draws;
    }
    note_recorded_resource(r.texture);
    note_recorded_resource(r.intermediate);
    note_recorded_resource(group);
    encoder_.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    stencil_twin_.note_copied_out(r.texture, mip, layer);
    ++stats_.stencil_twin_copies_out;
    invalidate_pass_bindings();
    compute_pass_valid_ = false;
    compute_pipeline_bound_ = 0;
    compute_groups_.clear();
}

void CommandListBridge::flush_stencil_twin(uint32_t texture) {
    const auto it = stencil_twin_flush_.find(texture);
    if (it == stencil_twin_flush_.end()) {
        return;
    }
    for (const DepthTwinTracker::Sub& sub : stencil_twin_.pending_out(texture)) {
        emit_stencil_twin_out_sub(it->second, sub.mip, sub.layer);
    }
}

bool CommandListBridge::emit_depth_twin_in(const StorageRedirect& r) {
    bool copied = false;
    for (uint32_t layer = r.base_layer; layer < r.base_layer + r.layer_count; ++layer) {
        if (!depth_twin_.needs_copy_in(r.texture, r.mip, layer)) {
            continue;
        }
        const StateBridgeDevice::DepthTwinPipelines p =
            device_.depth_twin_pipelines(r.depth_format, r.storage_format, encoder_);
        const uint32_t group = device_.depth_twin_group(p.in_layout, r.texture, r.intermediate, true,
                                                        r.storage_format, r.mip, layer, encoder_);
        encoder_.set_compute_pipeline(p.in_pipeline);
        const GpuSetBindGroup2Payload set{0u, group, 0u, 0u};
        encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
        note_recorded_resource(r.texture);
        note_recorded_resource(r.intermediate);
        note_recorded_resource(group);
        encoder_.dispatch_compute((r.extent[0] + 7u) / 8u, (r.extent[1] + 7u) / 8u, 1u);
        depth_twin_.note_copied_in(r.texture, r.mip, layer);
        ++stats_.depth_twin_copies_in;
        copied = true;
    }
    return copied;
}

void CommandListBridge::emit_depth_twin_out(const StorageRedirect& r) {
    for (uint32_t layer = r.base_layer; layer < r.base_layer + r.layer_count; ++layer) {
        emit_depth_twin_out_sub(r, r.mip, layer);
    }
}

void CommandListBridge::emit_depth_twin_out_sub(const StorageRedirect& r, uint32_t mip, uint32_t layer) {
    if (!depth_twin_.needs_copy_out(r.texture, mip, layer)) {
        return;
    }
    const StateBridgeDevice::DepthTwinPipelines p = device_.depth_twin_pipelines(r.depth_format, r.storage_format, encoder_);
    const uint32_t group = device_.depth_twin_group(p.out_layout, r.texture, r.intermediate, false, r.storage_format,
                                                    mip, layer, encoder_);
    GpuBeginRenderPassPayload pass{};
    pass.color_count = 0;
    GpuRenderPassDepthStencilAttachment& a = pass.depth_stencil;
    a.texture_id = r.texture;
    a.depth_load_op = static_cast<uint8_t>(GpuLoadOp::Load);
    a.depth_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
    if (pso::has_stencil_aspect(static_cast<pso::WgpuTextureFormat>(r.depth_format))) {
        a.stencil_read_only = 1;  // the stencil plane is untouched
    }
    a.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
    a.view.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
    a.view.base_mip_level = static_cast<uint16_t>(mip);
    a.view.mip_level_count = 1;
    a.view.base_array_layer = layer;
    a.view.array_layer_count = 1;
    encoder_.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof(pass));
    ++stats_.render_passes;
    encoder_.set_pipeline(p.out_pipeline);
    const GpuSetBindGroup2Payload set{0u, group, 0u, 0u};
    encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
    note_recorded_resource(r.texture);
    note_recorded_resource(r.intermediate);
    note_recorded_resource(group);
    encoder_.draw(3, 1, 0, 0);
    encoder_.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    depth_twin_.note_copied_out(r.texture, mip, layer);
    ++stats_.depth_twin_copies_out;
    invalidate_pass_bindings();
    compute_pass_valid_ = false;
    compute_pipeline_bound_ = 0;
    compute_groups_.clear();
}

void CommandListBridge::flush_depth_twin(uint32_t texture) {
    // A copy out of a depth texture (read-back) sees the twin's pending writes first.
    const auto it = depth_twin_flush_.find(texture);
    if (it == depth_twin_flush_.end()) {
        return;
    }
    for (const DepthTwinTracker::Sub& sub : depth_twin_.pending_out(texture)) {
        emit_depth_twin_out_sub(it->second, sub.mip, sub.layer);
    }
}

void CommandListBridge::emit_storage_conversion(const StorageRedirect& r, bool into_intermediate) {
    if (!into_intermediate && r.render_copy_out) {
        emit_storage_render_out(r);
        return;
    }
    const uint8_t dst_format = into_intermediate ? r.storage_format : r.format;
    const StateBridgeDevice::StorageConversionPipeline pipeline =
        device_.storage_conversion_pipeline(dst_format, r.dimension, encoder_);
    const uint32_t group = into_intermediate
                               ? device_.storage_conversion_group(pipeline, r.dimension, r.texture, r.format,
                                                                  r.intermediate, r.storage_format, r.mip,
                                                                  r.base_layer, r.layer_count, encoder_)
                               : device_.storage_conversion_group(pipeline, r.dimension, r.intermediate,
                                                                  r.storage_format, r.texture, r.format, r.mip,
                                                                  r.base_layer, r.layer_count, encoder_);
    encoder_.set_compute_pipeline(pipeline.pipeline);
    const GpuSetBindGroup2Payload set{0u, group, 0u, 0u};
    encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
    note_recorded_resource(r.texture);
    note_recorded_resource(r.intermediate);
    note_recorded_resource(group);
    const bool volume = r.dimension == static_cast<uint8_t>(GpuTextureViewDimension::Dim3D);
    const uint32_t xy = volume ? 4u : 8u;
    encoder_.dispatch_compute((r.extent[0] + xy - 1) / xy, (r.extent[1] + xy - 1) / xy,
                              volume ? (r.extent[2] + 3u) / 4u : r.extent[2]);
    ++stats_.storage_conversions;
}

void CommandListBridge::emit_storage_render_out(const StorageRedirect& r) {
    const StateBridgeDevice::StorageConversionPipeline p =
        device_.storage_render_out_pipeline(r.format, encoder_);
    for (uint32_t layer = r.base_layer; layer < r.base_layer + r.layer_count; ++layer) {
        const uint32_t group = device_.storage_render_out_group(p.group_layout, r.intermediate, r.storage_format,
                                                                r.texture, r.mip, layer, encoder_);
        GpuBeginRenderPassPayload pass{};
        pass.color_count = 1;
        GpuRenderPassColorAttachment& a = pass.color[0];
        a.texture_id = r.texture;
        a.load_op = static_cast<uint8_t>(GpuLoadOp::Load);
        a.store_op = static_cast<uint8_t>(GpuStoreOp::Store);
        a.view.format = r.format;
        a.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        a.view.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
        a.view.base_mip_level = static_cast<uint16_t>(r.mip);
        a.view.mip_level_count = 1;
        a.view.base_array_layer = layer;
        a.view.array_layer_count = 1;
        encoder_.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof(pass));
        ++stats_.render_passes;
        encoder_.set_pipeline(p.pipeline);
        const GpuSetBindGroup2Payload set{0u, group, 0u, 0u};
        encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
        note_recorded_resource(r.texture);
        note_recorded_resource(r.intermediate);
        note_recorded_resource(group);
        encoder_.draw(3, 1, 0, 0);
        encoder_.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
        ++stats_.storage_conversions;
        invalidate_pass_bindings();
        compute_pass_valid_ = false;
        compute_pipeline_bound_ = 0;
        compute_groups_.clear();
    }
}

void CommandListBridge::add_usage(const d3d::BindGroupEntry& entry, const d3d::ShaderBinding& join) {
    if (entry.resource_id == 0) {
        return;
    }
    if (entry.kind == d3d::BindResourceKind::TextureView) {
        usage_scope_.add(entry.resource_id, d3d::view_subresource_range(entry.view), d3d::view_usage(entry.view));
    } else if (entry.kind == d3d::BindResourceKind::Buffer) {
        d3d::ResourceUsage usage = d3d::ResourceUsage::Sampled;
        if (join.register_class == d3d::DescriptorRangeType::Uav) {
            usage = d3d::ResourceUsage::StorageWrite;
        } else if (join.register_class == d3d::DescriptorRangeType::Srv) {
            usage = d3d::ResourceUsage::StorageRead;
        }
        usage_scope_.add(entry.resource_id, d3d::SubresourceRange{}, usage);
    }
}

bool CommandListBridge::apply_bind_groups(d3d::BindPoint bind_point,
                                          StateBridgeDevice::PipelineLayoutRecord* layout,
                                          const std::vector<d3d::BindGroupPlan>& plans,
                                          std::vector<BoundGroup>* bound,
                                          bool track_usage) {
    std::lock_guard cache_lock(cache_mutex_);
    drain_invalidations_locked();
    bound->resize(layout->groups.size());
    bool null_unresolved = false, null_empty = false;  // fallback path accounting, see BridgeStats
    for (size_t g = 0; g < layout->groups.size(); ++g) {
        const StateBridgeDevice::GroupLayoutRecord& group = *layout->groups[g];
        if (group.runtime_owned) {
            continue;  // G66 mesh output group: bound by the caller
        }
        uint32_t bind_group_id = 0;
        std::array<uint32_t, 8> stack_offsets{};
        uint32_t offset_count = 0;
        std::vector<uint32_t> heap_offsets;
        d3d::CachedBindGroup* cached = nullptr;
        if (group.entries.empty()) {
            auto it = empty_bind_groups_.find(group.id);
            if (it == empty_bind_groups_.end()) {
                device_.ensure_layout_emitted(layout, encoder_);
                GpuCreateBindGroup2Payload header{};
                header.id = device_.allocate_id();
                header.layout_id = group.id;
                encoder_.emit_packet(GpuOpcode::CreateBindGroup2, &header, sizeof(header));
                ++stats_.bind_groups_created;
                it = empty_bind_groups_.emplace(group.id, header.id).first;
            }
            bind_group_id = it->second;
            note_recorded_resource(bind_group_id);
        } else {
            const uint64_t heap_misses_before = bind_groups_.heap_index_misses();
            const uint64_t cb_misses_before = bind_groups_.cb_index_misses();
            const uint64_t hits_before = bind_groups_.hits();
            const uint64_t misses_before = bind_groups_.misses();
            d3d::BindGroupLookup lookup;
            {
                BuildingBridgeGuard guard(this);
                lookup = bind_groups_.get_or_build(
                    root_args_, bind_point, plans[g], group.id, &StateBridgeDevice::gpu_va_trampoline, &device_);
                if (lookup.status == d3d::Status::Ok && lookup.bind_group != nullptr && !lookup.rebuilt &&
                    lookup.bind_group->gpu_id != 0 && binds_retired(lookup.bind_group)) {
                    // G66 (#691): defense-in-depth fallback if a cached entry still names a retired
                    // resource: rebuild it from the descriptors and retire the old bind group.
                    const uint32_t old_id = bind_groups_.erase(lookup.bind_group);
                    lookup = bind_groups_.get_or_build(root_args_, bind_point, plans[g], group.id,
                                                       &StateBridgeDevice::gpu_va_trampoline, &device_);
                    if (lookup.retired_gpu_id == 0) {
                        lookup.retired_gpu_id = old_id;
                    } else if (old_id != 0) {
                        redirects_.erase(old_id);
                        encoder_.destroy_resource(old_id);
                    }
                    ++stats_.bind_groups_resource_retired;
                }
            }
            if (lookup.status == d3d::Status::Ok && lookup.bind_group != nullptr && lookup.rebuilt &&
                invalidated_key_hashes_.erase(lookup.bind_group->key_hash) != 0) {
                ++stats_.bind_groups_resource_retired;
            }
            stats_.bind_group_cache_hits += bind_groups_.hits() - hits_before;
            stats_.bind_group_cache_misses += bind_groups_.misses() - misses_before;
            stats_.heap_index_misses += bind_groups_.heap_index_misses() - heap_misses_before;
            stats_.cb_index_misses += bind_groups_.cb_index_misses() - cb_misses_before;
            if (lookup.status != d3d::Status::Ok || lookup.bind_group == nullptr) {
                drain_invalidations_locked();
                diagnostics_ |= lookup.status == d3d::Status::RootArgumentNotSet ? BridgeDiag::RootArgumentNotSet
                                                                                  : BridgeDiag::BindGroupBuildFailed;
                return false;
            }
            cached = lookup.bind_group;
            if (lookup.rebuilt || cached->gpu_id == 0) {
                device_.ensure_layout_emitted(layout, encoder_);
                if (!emit_bind_group(bind_point, group, cached, &plans[g])) {
                    drain_invalidations_locked();
                    return false;
                }
                if (lookup.retired_gpu_id != 0) {
                    redirects_.erase(lookup.retired_gpu_id);
                    encoder_.destroy_resource(lookup.retired_gpu_id);
                    ++stats_.bind_groups_retired;
                }
            } else if (relocations_moved(cached)) {
                // G66 (#691): a relocated home of this cached group now holds another range (the
                // LRU evicted it while the group sat in the cache): bind a group over the new homes.
                const uint32_t old_id = cached->gpu_id;
                device_.ensure_layout_emitted(layout, encoder_);
                if (!emit_bind_group(bind_point, group, cached, &plans[g])) {
                    drain_invalidations_locked();
                    return false;
                }
                redirects_.erase(old_id);
                encoder_.destroy_resource(old_id);
                ++stats_.bind_groups_relocation_moved;
            } else if (residency_changed(cached)) {
                // G66 partly-held mip chains: a bound texture's held levels changed since this group
                // was created: bind a group whose views start at the new first held level.
                const uint32_t old_id = cached->gpu_id;
                device_.ensure_layout_emitted(layout, encoder_);
                if (!emit_bind_group(bind_point, group, cached, &plans[g])) {
                    drain_invalidations_locked();
                    return false;
                }
                redirects_.erase(old_id);
                encoder_.destroy_resource(old_id);
                ++stats_.bind_groups_residency_changed;
            } else if (cached->from_memory) {
                note_memory_group(group, plans[g], *cached, kNoPacket);  // created by an earlier recording
            }
            sync_bound_textures(group, *cached, encoder_);
            bind_group_id = cached->gpu_id;
            note_bind_group_used_locked(*cached, bind_group_id);
            for (size_t i = 0; i < group.entries.size(); ++i) {
                if ((group.entries[i].flags & GpuBindGroupLayoutEntryFlags::HasDynamicOffset) == 0) {
                    continue;
                }
                const d3d::BindGroupEntry& e = cached->entries[i];
                const uint32_t dyn_off = e.kind == d3d::BindResourceKind::RootConstants
                                             ? upload_root_constants(bind_point, e.root_parameter_index)
                                             : 0u;
                if (heap_offsets.empty() && offset_count < stack_offsets.size()) {
                    stack_offsets[offset_count++] = dyn_off;
                } else {
                    if (heap_offsets.empty()) {
                        heap_offsets.assign(stack_offsets.begin(), stack_offsets.begin() + offset_count);
                    }
                    heap_offsets.push_back(dyn_off);
                    ++offset_count;
                }
            }
        }
        if (cached != nullptr && cached->null_entries != 0) {
            (plans[g].unresolved_count() != 0 ? null_unresolved : null_empty) = true;
        }
        BoundGroup& b = (*bound)[g];
        b.null_write_entries = cached != nullptr ? cached->null_write_entries : 0u;
        b.written.clear();
        if (bind_point == d3d::BindPoint::Compute && cached != nullptr) {
            for (size_t i = 0; i < group.entries.size() && i < cached->entries.size(); ++i) {
                const GpuBindGroupLayoutEntry& le = group.entries[i];
                const d3d::BindGroupEntry& ce = cached->entries[i];
                if (le.type == static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture) &&
                    le.binding_type != static_cast<uint8_t>(GpuStorageTextureAccess::ReadOnly) &&
                    ce.kind == d3d::BindResourceKind::TextureView && ce.resource_id != 0) {
                    const WrittenTexture w{ce.resource_id, d3d::view_subresource_range(ce.view).first_mip};
                    if (std::find(b.written.begin(), b.written.end(), w) == b.written.end()) {
                        b.written.push_back(w);
                    }
                }
            }
        }
        const uint32_t* offsets_ptr = heap_offsets.empty() ? stack_offsets.data() : heap_offsets.data();
        const bool same_offsets = b.dynamic_offsets.size() == offset_count &&
                                  (offset_count == 0 ||
                                   std::memcmp(b.dynamic_offsets.data(), offsets_ptr,
                                               offset_count * sizeof(uint32_t)) == 0);
        if (b.bind_group_id != bind_group_id || !same_offsets) {
            const GpuSetBindGroup2Payload header{static_cast<uint32_t>(g), bind_group_id, offset_count, 0u};
            encoder_.emit_packet(GpuOpcode::SetBindGroup2, &header, sizeof(header), offsets_ptr,
                                 static_cast<uint32_t>(offset_count * sizeof(uint32_t)));
            b.bind_group_id = bind_group_id;
            b.dynamic_offsets.assign(offsets_ptr, offsets_ptr + offset_count);
        }
        if (track_usage && cached != nullptr && scoped_bind_groups_.insert(bind_group_id).second) {
            for (size_t i = 0; i < cached->entries.size(); ++i) {
                add_usage(cached->entries[i], group.joins[i]);
            }
        }
        // Drain any re-entrant invalidation that arrived while building this group.
        drain_invalidations_locked();
    }
    stats_.null_bound_unresolved += null_unresolved ? 1u : 0u;
    stats_.null_bound_empty += null_empty ? 1u : 0u;
    return true;
}

// ---- Work ----

bool CommandListBridge::prepare_render_draw() {
    bool ok = false;
    // Attempt 1 may find fragment storage bindings redirected to intermediates that the pass has not
    // copied in yet: the pass ends, the conversions run, and attempt 2 reopens it (loadOp load).
    for (int attempt = 0; attempt < 2; ++attempt) {
        ok = tracker_.prepare_draw(&device_.pipeline_cache());
        drain_events();
        if (ok && graphics_pso_ != nullptr && graphics_pso_->layout_ != nullptr &&
            graphics_pso_->plans_.size() == graphics_pso_->layout_->groups.size() &&
            !apply_bind_groups(d3d::BindPoint::Graphics, graphics_pso_->layout_, graphics_pso_->plans_,
                               &render_groups_, true)) {
            ++stats_.draws_skipped;
            diagnostics_ |= BridgeDiag::DrawSkipped;
            return false;
        }
        if (!ok) {
            break;
        }
        if (redirects_.empty()) {
            break;
        }
        std::vector<StorageRedirect> active;
        for (const BoundGroup& b : render_groups_) {
            if (const auto it = redirects_.find(b.bind_group_id); it != redirects_.end()) {
                active.insert(active.end(), it->second.begin(), it->second.end());
            }
        }
        // The same texture as a render attachment of this pass: detected and counted, not converted.
        active.erase(std::remove_if(active.begin(), active.end(),
                                    [&](const StorageRedirect& r) {
                                        const bool alias = std::find(om_attachments_.begin(), om_attachments_.end(),
                                                                     r.texture) != om_attachments_.end();
                                        stats_.storage_views_attachment_alias += alias ? 1u : 0u;
                                        return alias;
                                    }),
                     active.end());
        if (active.empty()) {
            break;
        }
        const auto tracked = [&](const StorageRedirect& r) {
            return std::any_of(pass_redirects_.begin(), pass_redirects_.end(),
                               [&](const StorageRedirect& o) { return o.same_target(r); });
        };
        bool need_in = false;
        for (const StorageRedirect& r : active) {
            need_in |= !tracked(r) && r.access != static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly);
        }
        if (!need_in || attempt == 1) {
            for (const StorageRedirect& r : active) {
                if (!tracked(r)) {
                    pass_redirects_.push_back(r);
                }
            }
            break;
        }
        tracker_.flush_pending_clears();
        tracker_.end_render_pass();
        drain_events();  // converts out what the pass wrote so far
        for (const StorageRedirect& r : active) {
            if (r.access != static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly)) {
                emit_storage_conversion(r, true);  // real texture -> intermediate
            }
        }
        compute_pipeline_bound_ = 0;
        compute_groups_.clear();
        compute_pass_valid_ = false;
        pass_redirects_ = std::move(active);
    }
    if (!ok) {
        ++stats_.draws_skipped;
        diagnostics_ |= BridgeDiag::DrawSkipped;
    }
    return ok;
}

void CommandListBridge::apply_input_assembler(bool indexed) {
    if (graphics_pso_ != nullptr) {
        const uint32_t mask = graphics_pso_->pso().translation().default_vertex_slot_mask;
        for (uint32_t slot = 0; mask != 0 && slot < pso::kMaxVertexBuffers; ++slot) {
            if ((mask & (1u << slot)) != 0) {
                encoder_.set_vertex_buffer(slot, device_.ensure_default_vertex_buffer(encoder_), 0);
            }
        }
    }
    for (uint32_t slot = 0; slot < pso::kMaxVertexBuffers; ++slot) {
        if (vertex_buffers_[slot].buffer_id != 0) {
            note_recorded_resource(vertex_buffers_[slot].buffer_id);
        }
        if ((vertex_dirty_ & (1u << slot)) != 0 && vertex_buffers_[slot].buffer_id != 0) {
            encoder_.set_vertex_buffer(slot, vertex_buffers_[slot].buffer_id, vertex_buffers_[slot].offset);
        }
    }
    vertex_dirty_ = 0;
    if (indexed && index_buffer_.buffer_id != 0) {
        note_recorded_resource(index_buffer_.buffer_id);
        if (index_dirty_) {
            encoder_.set_index_buffer(index_buffer_.buffer_id, index_buffer_.format, index_buffer_.offset);
            index_dirty_ = false;
        }
    }
}

void CommandListBridge::DrawInstanced(uint32_t vertex_count_per_instance,
                                      uint32_t instance_count,
                                      uint32_t start_vertex_location,
                                      uint32_t start_instance_location) {
    if (!writable() || !prepare_render_draw()) {
        return;
    }
    apply_input_assembler(false);
    encoder_.draw(vertex_count_per_instance, instance_count, start_vertex_location, start_instance_location);
    ++stats_.draws;
}

void CommandListBridge::DrawIndexedInstanced(uint32_t index_count_per_instance,
                                             uint32_t instance_count,
                                             uint32_t start_index_location,
                                             int32_t base_vertex_location,
                                             uint32_t start_instance_location) {
    if (!writable()) {
        return;
    }
    if (index_buffer_.buffer_id == 0) {
        ++stats_.draws_skipped;
        diagnostics_ |= BridgeDiag::DrawSkipped;
        return;
    }
    if (!prepare_render_draw()) {
        return;
    }
    apply_input_assembler(true);
    encoder_.draw_indexed(index_count_per_instance, instance_count, start_index_location, base_vertex_location,
                          start_instance_location);
    ++stats_.draws;
}

ComputePipelineState* CommandListBridge::prepare_compute_dispatch() {
    return prepare_compute_dispatch(compute_pso_, d3d::BindPoint::Compute);
}

ComputePipelineState* CommandListBridge::prepare_compute_dispatch(ComputePipelineState* state,
                                                                  d3d::BindPoint bind_point) {
    // G7: pending clears must land and the render pass must end before compute work.
    tracker_.flush_pending_clears();
    tracker_.end_render_pass();
    drain_events();
    if (state == nullptr || state->pipeline_id_ == 0) {
        diagnostics_ |= BridgeDiag::NoComputePipeline;
        return nullptr;
    }
    if (device_.claim_compute_creation(state->pipeline_id_)) {
        GpuCreateComputePipeline2Payload payload{};
        payload.id = state->pipeline_id_;
        payload.cs_module_id = state->cs_module_id_;
        if (state->layout_ != nullptr) {
            device_.ensure_layout_emitted(state->layout_, encoder_);
            payload.pipeline_layout_id = state->layout_->id;
        }
        std::memcpy(&payload.key, &state->key_, sizeof(payload.key));
        device_.emit_device_object(payload.id, encoder_, [&](CommandEncoder& dst) {
            dst.emit_packet(GpuOpcode::CreateComputePipeline2, &payload, sizeof(payload));
        });
        ++stats_.pipelines_created;
        diagnostics_ |= state->diagnostics_;
        device_.record_compute_pipeline(state->key_, state->cs_module_id_, state->layout_);
    } else {
        device_.note_pipeline_reached(state->pipeline_id_);  // may be a pipeline the prewarm built
    }
    if (!compute_pass_valid_ || compute_pipeline_bound_ != state->pipeline_id_) {
        encoder_.set_compute_pipeline(state->pipeline_id_);
        compute_groups_.clear();
        compute_pass_valid_ = true;
        compute_pipeline_bound_ = state->pipeline_id_;
    }
    if (state->layout_ != nullptr && state->plans_.size() == state->layout_->groups.size() &&
        !apply_bind_groups(bind_point, state->layout_, state->plans_, &compute_groups_, false)) {
        return nullptr;
    }
    if (bind_point == d3d::BindPoint::Compute && state->layout_ != nullptr && state->layout_->rt_arena &&
        !prepare_rt_dispatch(state)) {
        return nullptr;
    }
    return state;
}

bool CommandListBridge::prepare_rt_dispatch(ComputePipelineState* state) {
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.swap(rt_ranges_);
    const std::vector<rt_residency::AddressSource>& sources = state->layout_->rt_address_sources;
    rt_residency::DispatchInputs in;
    if (!sources.empty()) {
        constexpr d3d::BindPoint kBp = d3d::BindPoint::Compute;
        const d3d::RootSignature* rs = root_args_.root_signature(kBp);
        in.root_param = [this, rs](uint32_t index, uint64_t* va) {
            if (rs == nullptr || index >= rs->parameter_count()) {
                return false;
            }
            const d3d::RootArgument& a = root_args_.argument(kBp, index);
            if (!a.is_set || a.gpu_address == 0) {
                return false;
            }
            *va = a.gpu_address;
            return true;
        };
        in.root_constant = [this, rs](uint32_t index, uint32_t dword, uint32_t* value) {
            if (rs == nullptr || index >= rs->parameter_count()) {
                return false;
            }
            const auto c = root_args_.root_constants(kBp, index);
            if (dword >= c.size()) {
                return false;
            }
            *value = c[dword];
            return true;
        };
        // User SGPR dword -> root signature slot (compute user data starts at kComputeUserDataBase).
        in.user_data = [this, rs](uint32_t sgpr, uint32_t* value) {
            uint32_t dword = 0;
            if (rs == nullptr) return false;
            const int pi = root_parameter_containing_user_sgpr(rs->desc(), true, sgpr, &dword, 0);
            if (pi < 0) return false;
            const d3d::RootParameterDesc& p = rs->desc().parameters[static_cast<size_t>(pi)];
            if (p.parameter_type == d3d::RootParameterType::Constants32Bit) {
                const auto c = root_args_.root_constants(kBp, static_cast<uint32_t>(pi));
                if (dword >= c.size()) return false;
                *value = c[dword];
                return true;
            }
            if (p.parameter_type == d3d::RootParameterType::DescriptorTable) return false;
            const d3d::RootArgument& a = root_args_.argument(kBp, static_cast<uint32_t>(pi));
            if (!a.is_set || a.gpu_address == 0 || dword > 1) return false;
            *value = dword == 0 ? static_cast<uint32_t>(a.gpu_address) : static_cast<uint32_t>(a.gpu_address >> 32);
            return true;
        };
        in.user_data_failure = [rs](uint32_t sgpr) {
            if (rs == nullptr) return rt_residency::Unresolved::NoInput;
            uint32_t dword = 0;
            const int pi = root_parameter_containing_user_sgpr(rs->desc(), true, sgpr, &dword, 0);
            if (pi < 0) return rt_residency::Unresolved::OutOfRange;
            if (rs->desc().parameters[static_cast<size_t>(pi)].parameter_type == d3d::RootParameterType::DescriptorTable) {
                return rt_residency::Unresolved::DescriptorTable;
            }
            return rt_residency::Unresolved::Unbound;
        };
        // A user_data dword holding a descriptor table: its slots sit in guest descriptor memory
        // (heap cpu_address + index * increment), 32-byte GFX10 descriptors.
        in.descriptor_table = [this, rs](uint32_t sgpr, rt_residency::DescriptorTableRef* t) {
            uint32_t dword = 0;
            if (rs == nullptr) return false;
            const int pi = root_parameter_containing_user_sgpr(rs->desc(), true, sgpr, &dword, 0);
            if (pi < 0) return false;
            const d3d::RootParameterDesc& p = rs->desc().parameters[static_cast<size_t>(pi)];
            if (p.parameter_type != d3d::RootParameterType::DescriptorTable) return false;
            const d3d::RootArgument& a = root_args_.argument(kBp, static_cast<uint32_t>(pi));
            if (!a.is_set || a.heap == nullptr || a.heap->desc().cpu_address == 0 || a.base_index >= a.heap->capacity() ||
                a.heap->increment() < 32) {
                return false;
            }
            uint64_t slots = 0;
            for (const d3d::DescriptorRangeDesc& r : p.ranges) slots += r.num_descriptors;
            const uint64_t avail = a.heap->capacity() - a.base_index;
            t->truncated = slots > avail;
            t->count = static_cast<uint32_t>(std::min<uint64_t>(std::min(slots, avail), rt_residency::kMaxTableDescriptors + 1));
            t->stride = a.heap->increment();
            t->address = a.heap->desc().cpu_address + uint64_t{a.base_index} * t->stride;
            return true;
        };
        in.read_guest = [this](uint64_t addr, void* dst, size_t bytes) {
            return device_.rt_read_guest(addr, dst, bytes);
        };
    }
    const uint32_t group = device_.rt_prepare_dispatch(*state->layout_->groups[rt_arena::kBindGroup], ranges, encoder_,
                                                       sources.empty() ? nullptr : &sources,
                                                       sources.empty() ? nullptr : &in);
    if (group == 0) {
        ++stats_.rt_skipped;
        return false;
    }
    const GpuSetBindGroup2Payload set{rt_arena::kBindGroup, group, 0u, 0u};
    encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
    if (compute_groups_.size() > rt_arena::kBindGroup) {
        compute_groups_[rt_arena::kBindGroup].bind_group_id = group;
    }
    ++stats_.rt_dispatches;
    return true;
}

void CommandListBridge::RtRequireRange(uint64_t addr, uint64_t size) {
    rt_ranges_.emplace_back(addr, size);
}

bool CommandListBridge::prepare_mesh_stage() {
    ComputePipelineState* stage = mesh_pso_->stage_.get();
    if (stage->layout_ == nullptr || stage->plans_.size() != stage->layout_->groups.size() ||
        prepare_compute_dispatch(stage, d3d::BindPoint::Graphics) == nullptr) {
        return false;
    }
    const uint32_t group = device_.mesh_output_group(*stage->layout_->groups[kMeshOutputGroup], encoder_);
    const GpuSetBindGroup2Payload set{kMeshOutputGroup, group, 0u, 0u};
    encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
    compute_groups_[kMeshOutputGroup].bind_group_id = group;
    return true;
}

bool CommandListBridge::prepare_mesh_draw() {
    GraphicsPipelineState* draw = mesh_pso_->draw_.get();
    if (draw->layout_ == nullptr || draw->plans_.size() != draw->layout_->groups.size()) {
        return false;
    }
    // The emulated mesh output is a triangle list whatever topology the list has set.
    const pso::D3DPrimitiveTopology topology = tracker_.variant_state().topology;
    tracker_.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    const bool ok = prepare_render_draw();
    tracker_.IASetPrimitiveTopology(topology);
    if (!ok) {
        return false;
    }
    const uint32_t group = device_.mesh_output_group(*draw->layout_->groups[kMeshOutputGroup], encoder_);
    const GpuSetBindGroup2Payload set{kMeshOutputGroup, group, 0u, 0u};
    encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
    if (render_groups_.size() > kMeshOutputGroup) {
        render_groups_[kMeshOutputGroup].bind_group_id = group;
    }
    return true;
}

bool CommandListBridge::DispatchMesh(uint32_t x, uint32_t y, uint32_t z) {
    if (!writable()) {
        return false;
    }
    if (mesh_pso_ == nullptr || y != 1 || z != 1 || x > mesh_max_groups(mesh_pso_->output_)) {
        ++stats_.mesh_skipped;
        diagnostics_ |= BridgeDiag::DrawSkipped;
        return false;
    }
    if (x == 0) {
        return true;  // no groups
    }
    if (!prepare_mesh_stage()) {
        ++stats_.mesh_skipped;
        diagnostics_ |= BridgeDiag::DrawSkipped;
        return false;
    }
    encoder_.dispatch_compute(x, 1, 1);
    ++stats_.dispatches;
    if (!prepare_mesh_draw()) {
        ++stats_.mesh_skipped;
        return false;
    }
    encoder_.draw(x * mesh_pso_->output_.max_primitives * 3u, 1, 0, 0);
    ++stats_.draws;
    ++stats_.mesh_dispatches;
    return true;
}

bool CommandListBridge::drop_null_storage_dispatch(uint32_t pipeline_id) {
    if (!device_.options().drop_null_storage_dispatch) {
        return false;
    }
    bool null_write = false;
    for (const BoundGroup& b : compute_groups_) {
        null_write = null_write || b.null_write_entries != 0;
    }
    if (!null_write) {
        return false;
    }
    ++stats_.dispatches_dropped_null_storage;
    diagnostics_ |= BridgeDiag::NullStorageDispatchDropped;
    if (null_storage_pipelines_.insert(pipeline_id).second) {
        ++stats_.null_storage_dispatch_pipelines;
    }
    return true;
}

void CommandListBridge::Dispatch(uint32_t thread_group_count_x,
                                 uint32_t thread_group_count_y,
                                 uint32_t thread_group_count_z) {
    if (!writable()) {
        return;
    }
    ComputePipelineState* state = prepare_compute_dispatch();
    if (state == nullptr) {
        return;
    }
    if (drop_null_storage_dispatch(state->pipeline_id_)) {
        return;
    }
    // #482: storage bindings redirected to an intermediate texture. Dispatches in one compute pass
    // are separate usage scopes, so the conversions run as dispatches around the stage's.
    std::vector<StorageRedirect> active;
    for (const BoundGroup& b : compute_groups_) {
        if (const auto it = redirects_.find(b.bind_group_id); it != redirects_.end()) {
            active.insert(active.end(), it->second.begin(), it->second.end());
        }
    }
    bool copied_in = false;
    for (const StorageRedirect& r : active) {
        if (r.depth_twin) {
            // Backlog 35: the twin must hold the depth plane (unwritten texels survive the copy
            // back), whatever the stage's access; only while it may differ.
            copied_in |= r.stencil_twin ? emit_stencil_twin_in(r) : emit_depth_twin_in(r);
        } else if (r.access != static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly)) {
            emit_storage_conversion(r, true);  // the stage reads it: real texture -> intermediate
            copied_in = true;
        }
    }
    if (copied_in) {  // the stage's pipeline and bind groups again
        encoder_.set_compute_pipeline(state->pipeline_id_);
        for (size_t g = 0; g < compute_groups_.size(); ++g) {
            const BoundGroup& b = compute_groups_[g];
            const GpuSetBindGroup2Payload header{static_cast<uint32_t>(g), b.bind_group_id,
                                                 static_cast<uint32_t>(b.dynamic_offsets.size()), 0u};
            encoder_.emit_packet(GpuOpcode::SetBindGroup2, &header, sizeof(header), b.dynamic_offsets.data(),
                                 static_cast<uint32_t>(b.dynamic_offsets.size() * sizeof(uint32_t)));
        }
    }
    encoder_.dispatch_compute(thread_group_count_x, thread_group_count_y, thread_group_count_z);
    ++stats_.dispatches;
    for (const BoundGroup& b : compute_groups_) {
        for (const WrittenTexture& w : b.written) {
            if (dispatch_written_.size() < 64 &&
                std::find(dispatch_written_.begin(), dispatch_written_.end(), w) == dispatch_written_.end()) {
                dispatch_written_.push_back(w);
            }
        }
    }
    if (state->layout_ != nullptr && state->layout_->rt_arena && !state->layout_->rt_address_sources.empty()) {
        GpuReadbackBufferPayload rb{};
        uint32_t stats_buffer = 0;
        if (device_.rt_fault_readback_begin(&rb, &stats_buffer)) {
            Readback(GpuOpcode::ReadbackBuffer, &rb, sizeof(rb), stats_buffer);
        }
    }
    for (const StorageRedirect& r : active) {
        if (r.access != static_cast<uint8_t>(GpuStorageTextureAccess::ReadOnly)) {
            if (r.depth_twin && r.stencil_twin) {
                for (uint32_t layer = r.base_layer; layer < r.base_layer + r.layer_count; ++layer) {
                    stencil_twin_.note_twin_written(r.texture, r.mip, layer);
                }
                stencil_twin_flush_[r.texture] = r;
                emit_stencil_twin_out(r);  // the twin -> the stencil plane (clear + eight masked draws)
            } else if (r.depth_twin) {
                for (uint32_t layer = r.base_layer; layer < r.base_layer + r.layer_count; ++layer) {
                    depth_twin_.note_twin_written(r.texture, r.mip, layer);
                }
                depth_twin_flush_[r.texture] = r;
                emit_depth_twin_out(r);  // the twin -> the depth plane (a draw writing frag_depth)
            } else {
                emit_storage_conversion(r, false);  // what the stage wrote: intermediate -> real texture
            }
        }
    }
    if (!active.empty()) {  // the next dispatch sets the stage's pipeline and bind groups again
        compute_pipeline_bound_ = 0;
        compute_groups_.clear();
    }
}

namespace {
// One clamp pass: params upload, pipeline, bind group, dispatch. The encoded size does not depend
// on the values, so a run's slots can be reserved ahead of time and patched.
void encode_clamp_pass(CommandEncoder& enc, uint32_t pipeline, uint32_t params_ring, uint32_t params_offset,
                       const uint32_t (&params)[8], uint32_t group, uint32_t window, uint32_t count) {
    enc.write_buffer(params_ring, params_offset, params, sizeof(params));
    enc.set_compute_pipeline(pipeline);
    const GpuSetBindGroup2Payload set{0u, group, 2u, 0u};
    const uint32_t offsets[2] = {params_offset, window};
    enc.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set), offsets, sizeof(offsets));
    enc.dispatch_compute((count + 63u) / 64u, 1, 1);
}
}  // namespace

bool CommandListBridge::emit_indirect_clamp(const IndirectExecution& e, uint32_t count, uint32_t* ring,
                                            uint32_t* out_offset) {
    // Backlog 41: clamps of plain draws are batched. The first of a run is recorded inline and
    // reserves patchable slots behind it; later executions whose argument buffers were not written
    // in between put their clamp into one of those slots, ahead of the draws between, instead of
    // breaking the render pass for a one-group compute pass each.
    const bool batchable = e.kind == IndirectKind::Draw || e.kind == IndirectKind::DrawIndexed;
    if (batchable) {
        clamp_batch_.scan(encoder_.data(), encoder_.size());
    } else {
        clamp_batch_.close();
    }
    const bool candidate =
        batchable && clamp_batch_.open() && device_.indirect_clamp_group_known(e.args_buffer, e.count_buffer);
    const auto end_pass = [&] {
        // The clamp pass is compute work: the render pass ends first.
        tracker_.flush_pending_clears();
        tracker_.end_render_pass();
        drain_events();
    };
    if (!candidate) {
        end_pass();
    }
    const size_t size_before = encoder_.size();
    const StateBridgeDevice::IndirectClamp clamp = device_.indirect_clamp(encoder_);
    const uint32_t params_ring = device_.ensure_root_constant_ring(encoder_);
    uint32_t index_limit = 0xFFFFFFFFu;
    uint32_t group_limit = kMaxIndirectWorkgroups;
    if (e.kind == IndirectKind::DrawIndexed && index_buffer_.size_in_bytes != 0) {
        index_limit = index_buffer_.size_in_bytes / (index_buffer_.format == GpuIndexFormat::Uint32 ? 4u : 2u);
    } else if (e.kind == IndirectKind::DispatchMesh && mesh_pso_ != nullptr) {
        index_limit = mesh_pso_->output_.max_primitives * 3u;  // draw vertices per group
        group_limit = mesh_max_groups(mesh_pso_->output_);
    }
    const uint32_t params[8] = {static_cast<uint32_t>(e.args_offset / 4u) + e.work_word,
                                e.stride / 4u,
                                count,
                                static_cast<uint32_t>(e.count_offset / 4u),
                                e.count_buffer != 0 ? 1u : 0u,
                                static_cast<uint32_t>(e.kind),
                                index_limit,
                                group_limit};
    bool wrapped = false;
    const uint32_t params_offset = device_.allocate_root_constants(sizeof(params), &wrapped);
    if (wrapped) {
        diagnostics_ |= BridgeDiag::RootConstantRingWrapped;
    }
    const uint32_t group = device_.indirect_clamp_group(clamp, params_ring, e.args_buffer, e.count_buffer, encoder_);
    const uint32_t window = device_.allocate_indirect_window();
    note_recorded_resource(e.args_buffer);
    note_recorded_resource(e.count_buffer);
    *ring = clamp.ring;
    *out_offset = window;

    // A wrapped parameter slot may still be read by work recorded after the run's start, so a
    // wrapped execution is never moved ahead.
    if (candidate && !wrapped && encoder_.size() == size_before) {
        const size_t slot = clamp_batch_.try_join(e.args_buffer, e.count_buffer, window);
        if (slot != IndirectClampBatch::kNoSlot) {
            CommandEncoder tmp;
            encode_clamp_pass(tmp, clamp.pipeline, params_ring, params_offset, params, group, window, count);
            if (tmp.size() == kIndirectClampSlotBytes && encoder_.patch(slot, tmp.data(), tmp.size())) {
                ++stats_.indirect_clamps_batched;
                return true;
            }
            // Unreachable while the slot size is fixed; keep the stream correct regardless.
            clamp_batch_.close();
        }
    }
    if (candidate) {
        end_pass();
    }
    encode_clamp_pass(encoder_, clamp.pipeline, params_ring, params_offset, params, group, window, count);
    // The stage's pipeline and bind groups are set again by the next dispatch.
    compute_pipeline_bound_ = 0;
    compute_groups_.clear();
    if (batchable) {
        const size_t reserve_at = encoder_.size();
        for (uint32_t i = 1; i < kIndirectClampBatchSize; ++i) {
            encoder_.nop_padded(kIndirectClampSlotBytes - static_cast<uint32_t>(sizeof(GpuPacketHeader)));
        }
        clamp_batch_.begin(reserve_at, kIndirectClampSlotBytes, encoder_.size(), window);
    }
    return true;
}

bool CommandListBridge::ExecuteIndirect(const IndirectExecution& e) {
    if (!writable()) {
        return false;
    }
    const uint32_t record = e.kind == IndirectKind::DrawIndexed                                        ? 20u
                            : e.kind == IndirectKind::Dispatch || e.kind == IndirectKind::DispatchMesh ? 12u
                                                                                                       : 16u;
    const uint32_t out_record = indirect_output_bytes(e.kind);
    const auto skip = [&] {
        ++stats_.indirect_skipped;
        diagnostics_ |= BridgeDiag::DrawSkipped;
        return false;
    };
    const bool constants = e.constant_count != 0;
    if (constants && (e.constant_count > kMaxIndirectRootConstants || e.kind == IndirectKind::DispatchMesh)) {
        return skip();  // mesh dispatches keep their single-argument form
    }
    for (uint32_t j = 0; j < e.constant_count && j < kMaxIndirectRootConstants; ++j) {
        const IndirectRootConstant& c = e.constants[j];
        if (c.root_parameter >= d3d::kMaxRootParameters || c.num_words == 0 || c.dest_word >= 64u ||
            c.num_words > 64u - c.dest_word || c.record_word + c.num_words > e.work_word) {
            return skip();
        }
    }
    if (e.args_buffer == 0 || e.stride < e.work_word * 4u + record || e.stride % 4u != 0 || e.args_offset % 4u != 0 ||
        e.count_offset % 4u != 0 || e.args_offset / 4u > 0xFFFFFFFFull || e.count_offset / 4u > 0xFFFFFFFFull) {
        return skip();
    }
    if (e.max_count == 0) {
        ++stats_.indirect_executions;
        return true;  // no commands
    }
    uint32_t count = std::min(e.max_count, kMaxIndirectCommands);
    if (constants) {
        count = std::min(count, indirect_constants_max_commands(e.constant_count));
        if (indirect_constants_chunk_bytes(e.constant_count, count) > device_.options().root_constant_ring_bytes) {
            return skip();
        }
    }
    if (e.kind == IndirectKind::DispatchMesh) {
        if (mesh_pso_ == nullptr || mesh_pso_->stage_->pipeline_id_ == 0 || mesh_max_groups(mesh_pso_->output_) == 0) {
            ++stats_.mesh_skipped;
            return skip();
        }
    } else if (e.kind == IndirectKind::Dispatch) {
        if (compute_pso_ == nullptr || compute_pso_->pipeline_id_ == 0) {
            diagnostics_ |= BridgeDiag::NoComputePipeline;
            return skip();
        }
    } else if (graphics_pso_ == nullptr || (e.kind == IndirectKind::DrawIndexed && index_buffer_.buffer_id == 0)) {
        return skip();
    }
    stats_.indirect_truncated += e.max_count - count;
    uint32_t ring = 0;
    uint32_t base = 0;
    uint32_t constants_chunk = 0;
    uint32_t constants_window = 0;
    if (constants && !emit_indirect_constants(e, count, &constants_chunk, &constants_window)) {
        return skip();
    }
    emit_indirect_clamp(e, count, &ring, &base);
    if (e.kind == IndirectKind::DispatchMesh) {
        // Per command: the mesh stage as compute (dispatch record), then the draw over its output.
        for (uint32_t i = 0; i < count; ++i) {
            const uint64_t at = uint64_t{base} + uint64_t{i} * out_record;
            if (!prepare_mesh_stage()) {
                ++stats_.mesh_skipped;
                return skip();
            }
            encoder_.indirect(GpuOpcode::DispatchIndirect, ring, at);
            if (!prepare_mesh_draw()) {
                ++stats_.mesh_skipped;
                return skip();
            }
            encoder_.indirect(GpuOpcode::DrawIndirect, ring, at + 12u);
            ++stats_.mesh_dispatches;
        }
        stats_.indirect_commands += count;
        ++stats_.indirect_executions;
        return true;
    }
    GpuOpcode opcode = GpuOpcode::DispatchIndirect;
    const d3d::BindPoint bind_point = e.kind == IndirectKind::Dispatch ? d3d::BindPoint::Compute
                                                                       : d3d::BindPoint::Graphics;
    // Readies the pipeline and bindings for the next command (once for the whole execution unless
    // the commands bind different root constants).
    const auto prepare = [&]() -> bool {
        if (e.kind == IndirectKind::Dispatch) {
            ComputePipelineState* indirect_state = prepare_compute_dispatch();
            if (indirect_state == nullptr) {
                return false;
            }
            if (drop_null_storage_dispatch(indirect_state->pipeline_id_)) {
                return false;
            }
            // Storage bindings redirected to an intermediate texture (#482) need conversions around the
            // stage, which only direct dispatches run: such an execution is skipped.
            for (const BoundGroup& b : compute_groups_) {
                if (redirects_.count(b.bind_group_id) != 0) {
                    return false;
                }
            }
            return true;
        }
        if (!prepare_render_draw()) {
            return false;
        }
        apply_input_assembler(e.kind == IndirectKind::DrawIndexed);
        return true;
    };
    if (e.kind != IndirectKind::Dispatch) {
        opcode = e.kind == IndirectKind::DrawIndexed ? GpuOpcode::DrawIndexedIndirect : GpuOpcode::DrawIndirect;
    }
    if (!constants) {
        if (!prepare()) {
            return skip();
        }
        for (uint32_t i = 0; i < count; ++i) {
            encoder_.indirect(opcode, ring, uint64_t{base} + uint64_t{i} * out_record);
        }
    } else {
        auto& overrides = root_override_[static_cast<uint32_t>(bind_point) & 1u];
        const uint32_t first_slot = 4u * indirect_constants_first_slot_word(e.constant_count);
        bool ok = true;
        for (uint32_t i = 0; i < count && ok; ++i) {
            for (uint32_t j = 0; j < e.constant_count; ++j) {
                overrides[e.constants[j].root_parameter] = {
                    constants_chunk + first_slot + (i * e.constant_count + j) * 256u, true};
            }
            ok = prepare();
            if (ok) {
                encoder_.indirect(opcode, ring, uint64_t{base} + uint64_t{i} * out_record);
            }
        }
        for (uint32_t j = 0; j < e.constant_count; ++j) {
            overrides[e.constants[j].root_parameter] = {};
        }
        if (!ok) {
            return skip();
        }
        ++stats_.indirect_constant_executions;
        stats_.indirect_constant_commands += count;
    }
    stats_.indirect_commands += count;
    ++stats_.indirect_executions;
    return true;
}

bool CommandListBridge::emit_indirect_constants(const IndirectExecution& e, uint32_t count, uint32_t* chunk_offset,
                                                uint32_t* window_bytes) {
    // The compute pass must not share a render pass.
    clamp_batch_.close();
    tracker_.flush_pending_clears();
    tracker_.end_render_pass();
    drain_events();
    const d3d::BindPoint bind_point = e.kind == IndirectKind::Dispatch ? d3d::BindPoint::Compute
                                                                       : d3d::BindPoint::Graphics;
    const uint32_t nargs = e.constant_count;
    const uint32_t bytes = indirect_constants_chunk_bytes(nargs, count);
    uint32_t window = 4096;
    while (window < bytes) {
        window <<= 1;
    }
    window = std::min(window, kIndirectConstantsMaxBytes);
    const uint32_t ring = device_.ensure_root_constant_ring(encoder_);
    bool wrapped = false;
    const uint32_t offset = device_.allocate_root_constants(window, &wrapped);
    if (wrapped) {
        diagnostics_ |= BridgeDiag::RootConstantRingWrapped;
    }
    // Header and base values (the parameters' current values), uploaded with the chunk start.
    std::vector<uint32_t> head(indirect_constants_first_slot_word(nargs), 0u);
    head[0] = static_cast<uint32_t>(e.args_offset / 4u);
    head[1] = e.stride / 4u;
    head[2] = count;
    head[3] = static_cast<uint32_t>(e.count_offset / 4u);
    head[4] = e.count_buffer != 0 ? 1u : 0u;
    head[5] = nargs;
    head[6] = indirect_constants_first_slot_word(nargs);
    head[7] = e.work_word + (e.kind == IndirectKind::DrawIndexed ? 5u : e.kind == IndirectKind::Draw ? 4u : 3u);
    for (uint32_t j = 0; j < nargs; ++j) {
        const IndirectRootConstant& c = e.constants[j];
        head[8 + 4 * j] = c.record_word;
        head[9 + 4 * j] = c.dest_word;
        head[10 + 4 * j] = c.num_words;
        const std::span<const uint32_t> values = root_args_.root_constants(bind_point, c.root_parameter);
        std::copy_n(values.begin(), std::min<size_t>(values.size(), 64u), head.begin() + 64 + 64 * j);
    }
    encoder_.write_buffer(ring, offset, head.data(), static_cast<uint32_t>(head.size() * 4u));
    const StateBridgeDevice::IndirectConstants pass = device_.indirect_constants(encoder_);
    const uint32_t group =
        device_.indirect_constants_group(pass, ring, e.args_buffer, e.count_buffer, window, encoder_);
    note_recorded_resource(e.args_buffer);
    note_recorded_resource(e.count_buffer);
    encoder_.set_compute_pipeline(pass.pipeline);
    const GpuSetBindGroup2Payload set{0u, group, 1u, 0u};
    encoder_.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set), &offset, sizeof(offset));
    encoder_.dispatch_compute((count + 63u) / 64u, 1, 1);
    // The stage's pipeline and bind groups are set again by the next dispatch.
    compute_pipeline_bound_ = 0;
    compute_groups_.clear();
    *chunk_offset = offset;
    *window_bytes = window;
    return true;
}

void CommandListBridge::ResourceBarrier(uint32_t num_barriers, const d3d::ResourceBarrier* barriers) {
    if (!writable()) {
        return;
    }
    // G7: anything that reads or copies a render target must see pending clears, and WebGPU usage
    // scopes end with the pass.
    tracker_.flush_pending_clears();
    tracker_.end_render_pass();
    drain_events();
    if (num_barriers != 0 && barriers != nullptr) {
        resource_states_.resource_barrier(num_barriers, barriers);
    }
}

size_t CommandListBridge::Readback(GpuOpcode opcode, const void* payload, uint32_t payload_bytes,
                                  uint32_t resource_id) {
    if (!writable()) {
        return SIZE_MAX;
    }
    tracker_.flush_pending_clears();
    tracker_.end_render_pass();
    drain_events();
    flush_depth_twin(resource_id);  // the read sees what a compute stage wrote into a depth twin
    flush_stencil_twin(resource_id);
    note_recorded_resource(resource_id);
    const size_t at = encoder_.size() + sizeof(GpuPacketHeader);
    encoder_.emit_packet(opcode, payload, payload_bytes);
    return at;
}

void CommandListBridge::QueryPacket(GpuOpcode opcode, const void* payload, uint32_t payload_bytes,
                                    uint32_t resource_id, bool ends_pass) {
    if (!writable()) {
        return;
    }
    if (ends_pass) {
        tracker_.flush_pending_clears();
        tracker_.end_render_pass();
    }
    drain_events();
    if (resource_id != 0) {
        note_recorded_resource(resource_id);
    }
    encoder_.emit_packet(opcode, payload, payload_bytes);
}

void CommandListBridge::WriteTexture(uint32_t texture_id, uint32_t mip_level, uint32_t origin_x, uint32_t origin_y,
                                     uint32_t origin_z, uint32_t width, uint32_t height, uint32_t depth_or_layers,
                                     uint32_t bytes_per_row, uint32_t rows_per_image, const void* data,
                                     uint32_t data_len) {
    if (!writable()) {
        return;
    }
    note_recorded_resource(texture_id);
    // A texture write into a depth plane leaves a colour twin of it behind.
    depth_twin_.note_depth_written(texture_id, mip_level, 1, origin_z, std::max(depth_or_layers, 1u));
    stencil_twin_.note_depth_written(texture_id, mip_level, 1, origin_z, std::max(depth_or_layers, 1u));
    if (!device_.translate_host_mip(texture_id, &mip_level)) {
        return;  // a level the capped host texture lacks (gpu/texture_budget.h)
    }
    encoder_.write_texture(texture_id, mip_level, origin_x, origin_y, origin_z, width, height, depth_or_layers,
                           bytes_per_row, rows_per_image, data, data_len);
}

}  // namespace playgta6::gpu::bridge
