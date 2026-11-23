#include "gpu/pso_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <tuple>
#include <utility>

namespace playgta6::gpu::pso {

namespace {

template <typename E>
constexpr size_t idx(E e) noexcept {
    return static_cast<size_t>(e);
}

constexpr uint32_t align_up(uint32_t value, uint32_t alignment) noexcept {
    return alignment == 0 ? value : (value + alignment - 1u) / alignment * alignment;
}

uint32_t canonical_float_bits(float value) noexcept {
    if (std::isnan(value) || value == 0.0f) {
        return 0u;  // NaN and -0.0 collapse onto +0.0
    }
    return std::bit_cast<uint32_t>(value);
}

char ascii_lower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool semantic_equal(const char* a, const char* b) noexcept {
    if (a == nullptr || b == nullptr) {
        return a == b;
    }
    while (*a != '\0' && *b != '\0') {
        if (ascii_lower(*a) != ascii_lower(*b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == *b;
}

// ---------------------------------------------------------------------------------------------------
// WebGPU format capability table (WebGPU spec, "Texture Format Capabilities").
// ---------------------------------------------------------------------------------------------------

using F = WgpuTextureFormat;
namespace C = TextureFormatCaps;
namespace Feat = WgpuFeature;

constexpr uint16_t kR = C::ColorRenderable;
constexpr uint16_t kB = C::Blendable;
constexpr uint16_t kM = C::Multisample;
constexpr uint16_t kD = C::Depth;
constexpr uint16_t kS = C::Stencil;
constexpr uint16_t kA = C::HasAlpha;
constexpr uint16_t kG = C::Srgb;
constexpr uint16_t kI = C::Integer;
constexpr uint16_t kC = C::Compressed;

constexpr WgpuTextureFormatInfo kTextureFormatInfo[] = {
    {"undefined", 0, 0, 0, 0, 0, 0, F::Undefined},
    {"r8unorm", kR | kB | kM, 0, 0, 0, 1, 1, F::Undefined},
    {"r8snorm", 0, 0, Feat::TextureFormatsTier1, Feat::TextureFormatsTier1, 1, 1, F::R16Float},
    {"r8uint", kR | kM | kI, 0, 0, 0, 1, 1, F::Undefined},
    {"r8sint", kR | kM | kI, 0, 0, 0, 1, 1, F::Undefined},
    {"r16unorm", kR | kB | kM, Feat::TextureFormatsTier1, 0, 0, 2, 2, F::R32Float},
    {"r16snorm", kR | kB | kM, Feat::TextureFormatsTier1, 0, 0, 2, 2, F::R32Float},
    {"r16uint", kR | kM | kI, 0, 0, 0, 2, 2, F::Undefined},
    {"r16sint", kR | kM | kI, 0, 0, 0, 2, 2, F::Undefined},
    {"r16float", kR | kB | kM, 0, 0, 0, 2, 2, F::Undefined},
    {"rg8unorm", kR | kB | kM, 0, 0, 0, 2, 1, F::Undefined},
    {"rg8snorm", 0, 0, Feat::TextureFormatsTier1, Feat::TextureFormatsTier1, 2, 1, F::Rg16Float},
    {"rg8uint", kR | kM | kI, 0, 0, 0, 2, 1, F::Undefined},
    {"rg8sint", kR | kM | kI, 0, 0, 0, 2, 1, F::Undefined},
    {"r32uint", kR | kI, 0, 0, 0, 4, 4, F::Undefined},
    {"r32sint", kR | kI, 0, 0, 0, 4, 4, F::Undefined},
    {"r32float", kR | kM, 0, 0, Feat::Float32Blendable, 4, 4, F::Undefined},
    {"rg16unorm", kR | kB | kM, Feat::TextureFormatsTier1, 0, 0, 4, 2, F::Rg32Float},
    {"rg16snorm", kR | kB | kM, Feat::TextureFormatsTier1, 0, 0, 4, 2, F::Rg32Float},
    {"rg16uint", kR | kM | kI, 0, 0, 0, 4, 2, F::Undefined},
    {"rg16sint", kR | kM | kI, 0, 0, 0, 4, 2, F::Undefined},
    {"rg16float", kR | kB | kM, 0, 0, 0, 4, 2, F::Undefined},
    {"rgba8unorm", kR | kB | kM | kA, 0, 0, 0, 8, 1, F::Undefined},
    {"rgba8unorm-srgb", kR | kB | kM | kA | kG, 0, 0, 0, 8, 1, F::Undefined},
    {"rgba8snorm", kA, 0, Feat::TextureFormatsTier1, Feat::TextureFormatsTier1, 8, 1, F::Rgba16Float},
    {"rgba8uint", kR | kM | kA | kI, 0, 0, 0, 4, 1, F::Undefined},
    {"rgba8sint", kR | kM | kA | kI, 0, 0, 0, 4, 1, F::Undefined},
    {"bgra8unorm", kR | kB | kM | kA, 0, 0, 0, 8, 1, F::Undefined},
    {"bgra8unorm-srgb", kR | kB | kM | kA | kG, 0, 0, 0, 8, 1, F::Undefined},
    {"rgb9e5ufloat", 0, 0, 0, 0, 0, 0, F::Undefined},
    {"rgb10a2uint", kR | kM | kA | kI, 0, 0, 0, 8, 4, F::Undefined},
    {"rgb10a2unorm", kR | kB | kM | kA, 0, 0, 0, 8, 4, F::Undefined},
    {"rg11b10ufloat", 0, 0, Feat::Rg11b10UfloatRenderable, Feat::Rg11b10UfloatRenderable, 8, 4,
     F::Rgba16Float},
    {"rg32uint", kR | kI, 0, 0, 0, 8, 4, F::Undefined},
    {"rg32sint", kR | kI, 0, 0, 0, 8, 4, F::Undefined},
    {"rg32float", kR, 0, 0, Feat::Float32Blendable, 8, 4, F::Undefined},
    {"rgba16unorm", kR | kB | kM | kA, Feat::TextureFormatsTier1, 0, 0, 8, 2, F::Rgba32Float},
    {"rgba16snorm", kR | kB | kM | kA, Feat::TextureFormatsTier1, 0, 0, 8, 2, F::Rgba32Float},
    {"rgba16uint", kR | kM | kA | kI, 0, 0, 0, 8, 2, F::Undefined},
    {"rgba16sint", kR | kM | kA | kI, 0, 0, 0, 8, 2, F::Undefined},
    {"rgba16float", kR | kB | kM | kA, 0, 0, 0, 8, 2, F::Undefined},
    {"rgba32uint", kR | kA | kI, 0, 0, 0, 16, 4, F::Undefined},
    {"rgba32sint", kR | kA | kI, 0, 0, 0, 16, 4, F::Undefined},
    {"rgba32float", kR | kA, 0, 0, Feat::Float32Blendable, 16, 4, F::Undefined},
    {"stencil8", kS | kM, 0, 0, 0, 0, 0, F::Undefined},
    {"depth16unorm", kD | kM, 0, 0, 0, 0, 0, F::Undefined},
    {"depth24plus", kD | kM, 0, 0, 0, 0, 0, F::Undefined},
    {"depth24plus-stencil8", kD | kS | kM, 0, 0, 0, 0, 0, F::Undefined},
    {"depth32float", kD | kM, 0, 0, 0, 0, 0, F::Undefined},
    {"depth32float-stencil8", kD | kS | kM, Feat::Depth32FloatStencil8, 0, 0, 0, 0,
     F::Depth24PlusStencil8},
    {"bc1-rgba-unorm", kC | kA, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba8Unorm},
    {"bc1-rgba-unorm-srgb", kC | kA | kG, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba8UnormSrgb},
    {"bc2-rgba-unorm", kC | kA, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba8Unorm},
    {"bc2-rgba-unorm-srgb", kC | kA | kG, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba8UnormSrgb},
    {"bc3-rgba-unorm", kC | kA, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba8Unorm},
    {"bc3-rgba-unorm-srgb", kC | kA | kG, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba8UnormSrgb},
    {"bc4-r-unorm", kC, Feat::TextureCompressionBc, 0, 0, 0, 0, F::R8Unorm},
    {"bc4-r-snorm", kC, Feat::TextureCompressionBc, 0, 0, 0, 0, F::R8Snorm},
    {"bc5-rg-unorm", kC, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rg8Unorm},
    {"bc5-rg-snorm", kC, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rg8Snorm},
    {"bc6h-rgb-ufloat", kC, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba16Float},
    {"bc6h-rgb-float", kC, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba16Float},
    {"bc7-rgba-unorm", kC | kA, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba8Unorm},
    {"bc7-rgba-unorm-srgb", kC | kA | kG, Feat::TextureCompressionBc, 0, 0, 0, 0, F::Rgba8UnormSrgb},
};
static_assert(std::size(kTextureFormatInfo) == idx(F::Count), "texture format table out of sync");

struct VertexFormatInfo {
    const char* name;
    uint8_t byte_size;
};

constexpr VertexFormatInfo kVertexFormatInfo[] = {
    {"undefined", 0},  {"uint8", 1},     {"uint8x2", 2},   {"uint8x4", 4},   {"sint8", 1},
    {"sint8x2", 2},    {"sint8x4", 4},   {"unorm8", 1},    {"unorm8x2", 2},  {"unorm8x4", 4},
    {"snorm8", 1},     {"snorm8x2", 2},  {"snorm8x4", 4},  {"uint16", 2},    {"uint16x2", 4},
    {"uint16x4", 8},   {"sint16", 2},    {"sint16x2", 4},  {"sint16x4", 8},  {"unorm16", 2},
    {"unorm16x2", 4},  {"unorm16x4", 8}, {"snorm16", 2},   {"snorm16x2", 4}, {"snorm16x4", 8},
    {"float16", 2},    {"float16x2", 4}, {"float16x4", 8}, {"float32", 4},   {"float32x2", 8},
    {"float32x3", 12}, {"float32x4", 16},{"uint32", 4},    {"uint32x2", 8},  {"uint32x3", 12},
    {"uint32x4", 16},  {"sint32", 4},    {"sint32x2", 8},  {"sint32x3", 12}, {"sint32x4", 16},
    {"unorm10-10-10-2", 4}, {"unorm8x4-bgra", 4},
};
static_assert(std::size(kVertexFormatInfo) == idx(WgpuVertexFormat::Count), "vertex format table out of sync");

constexpr const char* kBlendFactorNames[] = {
    "zero",           "one",
    "src",            "one-minus-src",
    "src-alpha",      "one-minus-src-alpha",
    "dst",            "one-minus-dst",
    "dst-alpha",      "one-minus-dst-alpha",
    "src-alpha-saturated", "constant",
    "one-minus-constant",  "src1",
    "one-minus-src1", "src1-alpha",
    "one-minus-src1-alpha",
};
static_assert(std::size(kBlendFactorNames) == idx(WgpuBlendFactor::Count));

constexpr const char* kBlendOperationNames[] = {"add", "subtract", "reverse-subtract", "min", "max"};
static_assert(std::size(kBlendOperationNames) == idx(WgpuBlendOperation::Count));

constexpr const char* kCompareNames[] = {"undefined", "never",   "less",      "equal",         "less-equal",
                                          "greater",   "not-equal", "greater-equal", "always"};
static_assert(std::size(kCompareNames) == idx(WgpuCompareFunction::Count));

constexpr const char* kStencilOpNames[] = {"keep",          "zero",           "replace",       "invert",
                                           "increment-clamp", "decrement-clamp", "increment-wrap",
                                           "decrement-wrap"};
static_assert(std::size(kStencilOpNames) == idx(WgpuStencilOperation::Count));

constexpr const char* kCullNames[] = {"none", "front", "back"};
constexpr const char* kFrontFaceNames[] = {"ccw", "cw"};
constexpr const char* kTopologyNames[] = {"triangle-list", "triangle-strip", "line-list", "line-strip",
                                          "point-list"};
static_assert(std::size(kTopologyNames) == idx(WgpuPrimitiveTopology::Count));
constexpr const char* kIndexFormatNames[] = {"undefined", "uint16", "uint32"};
constexpr const char* kLoadOpNames[] = {"undefined", "load", "clear"};
constexpr const char* kStoreOpNames[] = {"undefined", "store", "discard"};

template <typename E, size_t N>
const char* table_name(const char* const (&table)[N], E value) noexcept {
    const size_t i = idx(value);
    return i < N ? table[i] : nullptr;
}

// ---------------------------------------------------------------------------------------------------
// DXGI description used by the mapping functions.
// ---------------------------------------------------------------------------------------------------

struct DxgiDescription {
    WgpuTextureFormat color = F::Undefined;  // RenderTarget / Sampled
    WgpuTextureFormat depth = F::Undefined;  // DepthStencil (and depth-sampling views)
    WgpuTextureAspect depth_view_aspect = WgpuTextureAspect::All;
    bool typeless = false;
    bool alpha_ignored = false;
};

DxgiDescription describe_dxgi(DxgiFormat format) noexcept {
    using D = DxgiFormat;
    DxgiDescription r{};
    auto color = [&](F f) {
        r.color = f;
        return r;
    };
    auto typeless_color = [&](F f) {
        r.color = f;
        r.typeless = true;
        return r;
    };
    switch (format) {
        case D::R32G32B32A32_TYPELESS: return typeless_color(F::Rgba32Float);
        case D::R32G32B32A32_FLOAT: return color(F::Rgba32Float);
        case D::R32G32B32A32_UINT: return color(F::Rgba32Uint);
        case D::R32G32B32A32_SINT: return color(F::Rgba32Sint);
        case D::R16G16B16A16_TYPELESS: return typeless_color(F::Rgba16Float);
        case D::R16G16B16A16_FLOAT: return color(F::Rgba16Float);
        case D::R16G16B16A16_UNORM: return color(F::Rgba16Unorm);
        case D::R16G16B16A16_UINT: return color(F::Rgba16Uint);
        case D::R16G16B16A16_SNORM: return color(F::Rgba16Snorm);
        case D::R16G16B16A16_SINT: return color(F::Rgba16Sint);
        case D::R32G32_TYPELESS: return typeless_color(F::Rg32Float);
        case D::R32G32_FLOAT: return color(F::Rg32Float);
        case D::R32G32_UINT: return color(F::Rg32Uint);
        case D::R32G32_SINT: return color(F::Rg32Sint);
        case D::R32G8X24_TYPELESS:
            r.depth = F::Depth32FloatStencil8;
            r.typeless = true;
            return r;
        case D::D32_FLOAT_S8X24_UINT:
            r.depth = F::Depth32FloatStencil8;
            return r;
        case D::R32_FLOAT_X8X24_TYPELESS:
            r.depth = F::Depth32FloatStencil8;
            r.depth_view_aspect = WgpuTextureAspect::DepthOnly;
            return r;
        case D::X32_TYPELESS_G8X24_UINT:
            r.depth = F::Depth32FloatStencil8;
            r.depth_view_aspect = WgpuTextureAspect::StencilOnly;
            return r;
        case D::R10G10B10A2_TYPELESS: return typeless_color(F::Rgb10a2Unorm);
        case D::R10G10B10A2_UNORM: return color(F::Rgb10a2Unorm);
        case D::R10G10B10A2_UINT: return color(F::Rgb10a2Uint);
        case D::R11G11B10_FLOAT: return color(F::Rg11b10Ufloat);
        case D::R8G8B8A8_TYPELESS: return typeless_color(F::Rgba8Unorm);
        case D::R8G8B8A8_UNORM: return color(F::Rgba8Unorm);
        case D::R8G8B8A8_UNORM_SRGB: return color(F::Rgba8UnormSrgb);
        case D::R8G8B8A8_UINT: return color(F::Rgba8Uint);
        case D::R8G8B8A8_SNORM: return color(F::Rgba8Snorm);
        case D::R8G8B8A8_SINT: return color(F::Rgba8Sint);
        case D::R16G16_TYPELESS: return typeless_color(F::Rg16Float);
        case D::R16G16_FLOAT: return color(F::Rg16Float);
        case D::R16G16_UNORM: return color(F::Rg16Unorm);
        case D::R16G16_UINT: return color(F::Rg16Uint);
        case D::R16G16_SNORM: return color(F::Rg16Snorm);
        case D::R16G16_SINT: return color(F::Rg16Sint);
        case D::R32_TYPELESS:
            r.color = F::R32Float;
            r.depth = F::Depth32Float;
            r.typeless = true;
            return r;
        case D::D32_FLOAT:
            r.depth = F::Depth32Float;
            return r;
        case D::R32_FLOAT: return color(F::R32Float);
        case D::R32_UINT: return color(F::R32Uint);
        case D::R32_SINT: return color(F::R32Sint);
        case D::R24G8_TYPELESS:
            r.depth = F::Depth24PlusStencil8;
            r.typeless = true;
            return r;
        case D::D24_UNORM_S8_UINT:
            r.depth = F::Depth24PlusStencil8;
            return r;
        case D::R24_UNORM_X8_TYPELESS:
            r.depth = F::Depth24PlusStencil8;
            r.depth_view_aspect = WgpuTextureAspect::DepthOnly;
            return r;
        case D::X24_TYPELESS_G8_UINT:
            r.depth = F::Depth24PlusStencil8;
            r.depth_view_aspect = WgpuTextureAspect::StencilOnly;
            return r;
        case D::R8G8_TYPELESS: return typeless_color(F::Rg8Unorm);
        case D::R8G8_UNORM: return color(F::Rg8Unorm);
        case D::R8G8_UINT: return color(F::Rg8Uint);
        case D::R8G8_SNORM: return color(F::Rg8Snorm);
        case D::R8G8_SINT: return color(F::Rg8Sint);
        case D::R16_TYPELESS:
            r.color = F::R16Float;
            r.depth = F::Depth16Unorm;
            r.typeless = true;
            return r;
        case D::R16_FLOAT: return color(F::R16Float);
        case D::D16_UNORM:
            r.depth = F::Depth16Unorm;
            return r;
        case D::R16_UNORM: return color(F::R16Unorm);
        case D::R16_UINT: return color(F::R16Uint);
        case D::R16_SNORM: return color(F::R16Snorm);
        case D::R16_SINT: return color(F::R16Sint);
        case D::R8_TYPELESS: return typeless_color(F::R8Unorm);
        case D::R8_UNORM: return color(F::R8Unorm);
        case D::R8_UINT: return color(F::R8Uint);
        case D::R8_SNORM: return color(F::R8Snorm);
        case D::R8_SINT: return color(F::R8Sint);
        case D::R9G9B9E5_SHAREDEXP: return color(F::Rgb9e5Ufloat);
        case D::BC1_TYPELESS: return typeless_color(F::Bc1RgbaUnorm);
        case D::BC1_UNORM: return color(F::Bc1RgbaUnorm);
        case D::BC1_UNORM_SRGB: return color(F::Bc1RgbaUnormSrgb);
        case D::BC2_TYPELESS: return typeless_color(F::Bc2RgbaUnorm);
        case D::BC2_UNORM: return color(F::Bc2RgbaUnorm);
        case D::BC2_UNORM_SRGB: return color(F::Bc2RgbaUnormSrgb);
        case D::BC3_TYPELESS: return typeless_color(F::Bc3RgbaUnorm);
        case D::BC3_UNORM: return color(F::Bc3RgbaUnorm);
        case D::BC3_UNORM_SRGB: return color(F::Bc3RgbaUnormSrgb);
        case D::BC4_TYPELESS: return typeless_color(F::Bc4RUnorm);
        case D::BC4_UNORM: return color(F::Bc4RUnorm);
        case D::BC4_SNORM: return color(F::Bc4RSnorm);
        case D::BC5_TYPELESS: return typeless_color(F::Bc5RgUnorm);
        case D::BC5_UNORM: return color(F::Bc5RgUnorm);
        case D::BC5_SNORM: return color(F::Bc5RgSnorm);
        case D::B8G8R8A8_UNORM: return color(F::Bgra8Unorm);
        case D::B8G8R8X8_UNORM:
            r.alpha_ignored = true;
            return color(F::Bgra8Unorm);
        case D::B8G8R8A8_TYPELESS: return typeless_color(F::Bgra8Unorm);
        case D::B8G8R8A8_UNORM_SRGB: return color(F::Bgra8UnormSrgb);
        case D::B8G8R8X8_TYPELESS:
            r.alpha_ignored = true;
            return typeless_color(F::Bgra8Unorm);
        case D::B8G8R8X8_UNORM_SRGB:
            r.alpha_ignored = true;
            return color(F::Bgra8UnormSrgb);
        case D::BC6H_TYPELESS: return typeless_color(F::Bc6hRgbUfloat);
        case D::BC6H_UF16: return color(F::Bc6hRgbUfloat);
        case D::BC6H_SF16: return color(F::Bc6hRgbFloat);
        case D::BC7_TYPELESS: return typeless_color(F::Bc7RgbaUnorm);
        case D::BC7_UNORM: return color(F::Bc7RgbaUnorm);
        case D::BC7_UNORM_SRGB: return color(F::Bc7RgbaUnormSrgb);
        default:
            // UNKNOWN, 3-component 32-bit formats, A8, R1, packed 4:2:2, B5G6R5/B5G5R5A1/B4G4R4A4,
            // XR_BIAS and video formats: no WebGPU equivalent.
            return r;
    }
}

bool is_multisample_capable(WgpuTextureFormat format, uint32_t features) noexcept {
    const auto& info = texture_format_info(format);
    if ((info.caps & C::Multisample) != 0) {
        return true;
    }
    return info.renderable_feature != 0 && (features & info.renderable_feature) != 0;
}

}  // namespace

// =====================================================================================================
// Names and format queries
// =====================================================================================================

const WgpuTextureFormatInfo& texture_format_info(WgpuTextureFormat format) noexcept {
    const size_t i = idx(format);
    return i < std::size(kTextureFormatInfo) ? kTextureFormatInfo[i] : kTextureFormatInfo[0];
}

const char* texture_format_name(WgpuTextureFormat format) noexcept {
    const size_t i = idx(format);
    return i < std::size(kTextureFormatInfo) ? kTextureFormatInfo[i].name : nullptr;
}

const char* vertex_format_name(WgpuVertexFormat format) noexcept {
    const size_t i = idx(format);
    return i < std::size(kVertexFormatInfo) ? kVertexFormatInfo[i].name : nullptr;
}

const char* blend_factor_name(WgpuBlendFactor factor) noexcept { return table_name(kBlendFactorNames, factor); }
const char* blend_operation_name(WgpuBlendOperation op) noexcept { return table_name(kBlendOperationNames, op); }
const char* compare_function_name(WgpuCompareFunction func) noexcept { return table_name(kCompareNames, func); }
const char* stencil_operation_name(WgpuStencilOperation op) noexcept { return table_name(kStencilOpNames, op); }
const char* cull_mode_name(WgpuCullMode mode) noexcept { return table_name(kCullNames, mode); }
const char* front_face_name(WgpuFrontFace face) noexcept { return table_name(kFrontFaceNames, face); }
const char* primitive_topology_name(WgpuPrimitiveTopology topology) noexcept {
    return table_name(kTopologyNames, topology);
}
const char* index_format_name(WgpuIndexFormat format) noexcept { return table_name(kIndexFormatNames, format); }
const char* load_op_name(WgpuLoadOp op) noexcept { return table_name(kLoadOpNames, op); }
const char* store_op_name(WgpuStoreOp op) noexcept { return table_name(kStoreOpNames, op); }

const char* feature_name(uint32_t single_feature_bit) noexcept {
    switch (single_feature_bit) {
        case WgpuFeature::DepthClipControl: return "depth-clip-control";
        case WgpuFeature::Depth32FloatStencil8: return "depth32float-stencil8";
        case WgpuFeature::TextureCompressionBc: return "texture-compression-bc";
        case WgpuFeature::Float32Blendable: return "float32-blendable";
        case WgpuFeature::Rg11b10UfloatRenderable: return "rg11b10ufloat-renderable";
        case WgpuFeature::DualSourceBlending: return "dual-source-blending";
        case WgpuFeature::TextureFormatsTier1: return "texture-formats-tier1";
        default: return nullptr;
    }
}

bool is_color_renderable(WgpuTextureFormat format, uint32_t features) noexcept {
    const auto& info = texture_format_info(format);
    if (format == F::Undefined || (info.required_feature & ~features) != 0) {
        return false;
    }
    if ((info.caps & C::ColorRenderable) != 0) {
        return true;
    }
    return info.renderable_feature != 0 && (features & info.renderable_feature) != 0;
}

bool is_blendable(WgpuTextureFormat format, uint32_t features) noexcept {
    if (!is_color_renderable(format, features)) {
        return false;
    }
    const auto& info = texture_format_info(format);
    if ((info.caps & C::Blendable) != 0) {
        return true;
    }
    return info.blendable_feature != 0 && (features & info.blendable_feature) != 0;
}

bool has_depth_aspect(WgpuTextureFormat format) noexcept {
    return (texture_format_info(format).caps & C::Depth) != 0;
}

bool has_stencil_aspect(WgpuTextureFormat format) noexcept {
    return (texture_format_info(format).caps & C::Stencil) != 0;
}

bool has_alpha_channel(WgpuTextureFormat format) noexcept {
    return (texture_format_info(format).caps & C::HasAlpha) != 0;
}

uint32_t color_attachment_bytes_per_sample(std::span<const WgpuTextureFormat> formats) noexcept {
    uint32_t total = 0;
    for (WgpuTextureFormat format : formats) {
        if (format == F::Undefined) {
            continue;
        }
        const auto& info = texture_format_info(format);
        total = align_up(total, info.rt_alignment);
        total += info.rt_byte_cost;
    }
    return total;
}

uint32_t vertex_format_byte_size(WgpuVertexFormat format) noexcept {
    const size_t i = idx(format);
    return i < std::size(kVertexFormatInfo) ? kVertexFormatInfo[i].byte_size : 0u;
}

TextureFormatMapping map_texture_format(DxgiFormat format, FormatUsage usage, uint32_t features) noexcept {
    TextureFormatMapping m{};
    const DxgiDescription d = describe_dxgi(format);

    WgpuTextureFormat chosen = F::Undefined;
    switch (usage) {
        case FormatUsage::RenderTarget:
            chosen = d.color;
            break;
        case FormatUsage::DepthStencil:
            chosen = d.depth;
            break;
        case FormatUsage::Sampled:
            if (d.color != F::Undefined) {
                chosen = d.color;
            } else {
                chosen = d.depth;
                m.aspect = d.depth_view_aspect;
            }
            break;
    }
    if (chosen == F::Undefined) {
        m.diagnostics |= FormatDiag::Unsupported;
        m.aspect = WgpuTextureAspect::All;
        return m;
    }
    if (d.typeless) {
        m.diagnostics |= FormatDiag::TypelessResolved | FormatDiag::NeedsViewReinterpretation;
    }
    if (d.alpha_ignored) {
        m.diagnostics |= FormatDiag::AlphaIgnored;
    }

    const auto& info = texture_format_info(chosen);
    const bool compressed = (info.caps & C::Compressed) != 0;
    if (compressed && usage != FormatUsage::Sampled) {
        m.diagnostics |= FormatDiag::Unsupported;  // D3D12 cannot render to BC formats either
        return m;
    }

    // Existence: the format itself needs a feature.
    if (info.required_feature != 0 && (features & info.required_feature) == 0) {
        chosen = info.fallback;
        m.diagnostics |= FormatDiag::FeatureFallback;
        if (!has_depth_aspect(chosen)) {
            m.diagnostics |= FormatDiag::NeedsCpuConversion;  // decompress / widen uploaded texels
        }
    } else {
        m.required_features |= info.required_feature;
    }

    if (usage == FormatUsage::RenderTarget && !is_color_renderable(chosen, features)) {
        const auto& cur = texture_format_info(chosen);
        if (cur.renderable_feature != 0 && (features & cur.renderable_feature) == 0 &&
            cur.fallback != F::Undefined && is_color_renderable(cur.fallback, features)) {
            chosen = cur.fallback;
            m.diagnostics |= FormatDiag::FeatureFallback;
        } else {
            m.diagnostics |= FormatDiag::Unsupported;
            return m;
        }
    }
    if (usage == FormatUsage::RenderTarget) {
        const auto& cur = texture_format_info(chosen);
        if ((cur.caps & C::ColorRenderable) == 0) {
            m.required_features |= cur.renderable_feature;
        }
    }
    if (usage == FormatUsage::DepthStencil && !has_depth_aspect(chosen) && !has_stencil_aspect(chosen)) {
        m.diagnostics |= FormatDiag::Unsupported;
        return m;
    }

    m.format = chosen;
    return m;
}

VertexFormatMapping map_vertex_format(DxgiFormat format) noexcept {
    using D = DxgiFormat;
    using V = WgpuVertexFormat;
    V v = V::Undefined;
    switch (format) {
        case D::R32G32B32A32_FLOAT: v = V::Float32x4; break;
        case D::R32G32B32A32_UINT: v = V::Uint32x4; break;
        case D::R32G32B32A32_SINT: v = V::Sint32x4; break;
        case D::R32G32B32_FLOAT: v = V::Float32x3; break;
        case D::R32G32B32_UINT: v = V::Uint32x3; break;
        case D::R32G32B32_SINT: v = V::Sint32x3; break;
        case D::R16G16B16A16_FLOAT: v = V::Float16x4; break;
        case D::R16G16B16A16_UNORM: v = V::Unorm16x4; break;
        case D::R16G16B16A16_UINT: v = V::Uint16x4; break;
        case D::R16G16B16A16_SNORM: v = V::Snorm16x4; break;
        case D::R16G16B16A16_SINT: v = V::Sint16x4; break;
        case D::R32G32_FLOAT: v = V::Float32x2; break;
        case D::R32G32_UINT: v = V::Uint32x2; break;
        case D::R32G32_SINT: v = V::Sint32x2; break;
        case D::R10G10B10A2_UNORM: v = V::Unorm10_10_10_2; break;
        case D::R8G8B8A8_UNORM: v = V::Unorm8x4; break;
        case D::R8G8B8A8_UINT: v = V::Uint8x4; break;
        case D::R8G8B8A8_SNORM: v = V::Snorm8x4; break;
        case D::R8G8B8A8_SINT: v = V::Sint8x4; break;
        case D::R16G16_FLOAT: v = V::Float16x2; break;
        case D::R16G16_UNORM: v = V::Unorm16x2; break;
        case D::R16G16_UINT: v = V::Uint16x2; break;
        case D::R16G16_SNORM: v = V::Snorm16x2; break;
        case D::R16G16_SINT: v = V::Sint16x2; break;
        case D::R32_FLOAT: v = V::Float32; break;
        case D::R32_UINT: v = V::Uint32; break;
        case D::R32_SINT: v = V::Sint32; break;
        case D::R8G8_UNORM: v = V::Unorm8x2; break;
        case D::R8G8_UINT: v = V::Uint8x2; break;
        case D::R8G8_SNORM: v = V::Snorm8x2; break;
        case D::R8G8_SINT: v = V::Sint8x2; break;
        case D::R16_FLOAT: v = V::Float16; break;
        case D::R16_UNORM: v = V::Unorm16; break;
        case D::R16_UINT: v = V::Uint16; break;
        case D::R16_SNORM: v = V::Snorm16; break;
        case D::R16_SINT: v = V::Sint16; break;
        case D::R8_UNORM: v = V::Unorm8; break;
        case D::R8_UINT: v = V::Uint8; break;
        case D::R8_SNORM: v = V::Snorm8; break;
        case D::R8_SINT: v = V::Sint8; break;
        case D::B8G8R8A8_UNORM: v = V::Unorm8x4Bgra; break;
        default: break;
    }
    return VertexFormatMapping{v, vertex_format_byte_size(v)};
}

std::optional<WgpuBlendFactor> map_blend_factor(D3D12Blend blend) noexcept {
    using B = D3D12Blend;
    using W = WgpuBlendFactor;
    switch (blend) {
        case B::Zero: return W::Zero;
        case B::One: return W::One;
        case B::SrcColor: return W::Src;
        case B::InvSrcColor: return W::OneMinusSrc;
        case B::SrcAlpha: return W::SrcAlpha;
        case B::InvSrcAlpha: return W::OneMinusSrcAlpha;
        case B::DestAlpha: return W::DstAlpha;
        case B::InvDestAlpha: return W::OneMinusDstAlpha;
        case B::DestColor: return W::Dst;
        case B::InvDestColor: return W::OneMinusDst;
        case B::SrcAlphaSat: return W::SrcAlphaSaturated;
        case B::BlendFactor: return W::Constant;
        case B::InvBlendFactor: return W::OneMinusConstant;
        case B::Src1Color: return W::Src1;
        case B::InvSrc1Color: return W::OneMinusSrc1;
        case B::Src1Alpha: return W::Src1Alpha;
        case B::InvSrc1Alpha: return W::OneMinusSrc1Alpha;
        case B::AlphaFactor: return W::Constant;
        case B::InvAlphaFactor: return W::OneMinusConstant;
    }
    return std::nullopt;
}

std::optional<WgpuBlendOperation> map_blend_op(D3D12BlendOp op) noexcept {
    switch (op) {
        case D3D12BlendOp::Add: return WgpuBlendOperation::Add;
        case D3D12BlendOp::Subtract: return WgpuBlendOperation::Subtract;
        case D3D12BlendOp::RevSubtract: return WgpuBlendOperation::ReverseSubtract;
        case D3D12BlendOp::Min: return WgpuBlendOperation::Min;
        case D3D12BlendOp::Max: return WgpuBlendOperation::Max;
    }
    return std::nullopt;
}

std::optional<WgpuCompareFunction> map_compare_function(D3D12ComparisonFunc func) noexcept {
    using D = D3D12ComparisonFunc;
    using W = WgpuCompareFunction;
    switch (func) {
        case D::Never: return W::Never;
        case D::Less: return W::Less;
        case D::Equal: return W::Equal;
        case D::LessEqual: return W::LessEqual;
        case D::Greater: return W::Greater;
        case D::NotEqual: return W::NotEqual;
        case D::GreaterEqual: return W::GreaterEqual;
        case D::Always: return W::Always;
        case D::None: break;
    }
    return std::nullopt;
}

std::optional<WgpuStencilOperation> map_stencil_op(D3D12StencilOp op) noexcept {
    using D = D3D12StencilOp;
    using W = WgpuStencilOperation;
    switch (op) {
        case D::Keep: return W::Keep;
        case D::Zero: return W::Zero;
        case D::Replace: return W::Replace;
        case D::IncrSat: return W::IncrementClamp;
        case D::DecrSat: return W::DecrementClamp;
        case D::Invert: return W::Invert;
        case D::Incr: return W::IncrementWrap;
        case D::Decr: return W::DecrementWrap;
    }
    return std::nullopt;
}

std::optional<WgpuCullMode> map_cull_mode(D3D12CullMode mode) noexcept {
    switch (mode) {
        case D3D12CullMode::None: return WgpuCullMode::None;
        case D3D12CullMode::Front: return WgpuCullMode::Front;
        case D3D12CullMode::Back: return WgpuCullMode::Back;
    }
    return std::nullopt;
}

WgpuFrontFace map_front_face(bool front_counter_clockwise) noexcept {
    // Same framebuffer-space orientation in both APIs; see the header comment.
    return front_counter_clockwise ? WgpuFrontFace::Ccw : WgpuFrontFace::Cw;
}

uint32_t map_sample_count(uint32_t d3d12_count) noexcept {
    return d3d12_count <= 1 ? 1u : 4u;
}

TopologyMapping map_primitive_topology(D3DPrimitiveTopology topology) noexcept {
    using T = D3DPrimitiveTopology;
    using W = WgpuPrimitiveTopology;
    using Type = D3D12PrimitiveTopologyType;
    const uint32_t raw = static_cast<uint32_t>(topology);
    switch (topology) {
        case T::PointList: return {W::PointList, Type::Point, 0};
        case T::LineList: return {W::LineList, Type::Line, 0};
        case T::LineStrip: return {W::LineStrip, Type::Line, 0};
        case T::TriangleList: return {W::TriangleList, Type::Triangle, 0};
        case T::TriangleStrip: return {W::TriangleStrip, Type::Triangle, 0};
        case T::TriangleFan:
            return {W::TriangleList, Type::Triangle, PipelineDiag::TriangleFanNeedsConversion};
        case T::LineListAdj: return {W::LineList, Type::Line, PipelineDiag::AdjacencyTopologyUnsupported};
        case T::LineStripAdj: return {W::LineStrip, Type::Line, PipelineDiag::AdjacencyTopologyUnsupported};
        case T::TriangleListAdj:
            return {W::TriangleList, Type::Triangle, PipelineDiag::AdjacencyTopologyUnsupported};
        case T::TriangleStripAdj:
            return {W::TriangleStrip, Type::Triangle, PipelineDiag::AdjacencyTopologyUnsupported};
        default: break;
    }
    if (raw >= static_cast<uint32_t>(T::ControlPointPatchList1) &&
        raw <= static_cast<uint32_t>(T::ControlPointPatchList32)) {
        return {W::TriangleList, Type::Patch, PipelineDiag::TessellationUnsupported};
    }
    return {W::TriangleList, Type::Undefined, PipelineDiag::InvalidEnumValue};
}

// =====================================================================================================
// Keys and hashing
// =====================================================================================================

bool operator==(const GraphicsPipelineKey& a, const GraphicsPipelineKey& b) noexcept {
    return std::memcmp(&a, &b, sizeof(GraphicsPipelineKey)) == 0;
}

bool operator==(const ComputePipelineKey& a, const ComputePipelineKey& b) noexcept {
    return std::memcmp(&a, &b, sizeof(ComputePipelineKey)) == 0;
}

uint64_t stable_hash64(const void* data, size_t size) noexcept {
    // MurmurHash3-style 64-bit mixing over explicit little-endian words (independent of host
    // alignment, compiler and run).
    constexpr uint64_t c1 = 0x87c37b91114253d5ull;
    constexpr uint64_t c2 = 0x4cf5ad432745937full;
    const auto* bytes = static_cast<const unsigned char*>(data);
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(size) * c1);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        uint64_t k = 0;
        for (int b = 0; b < 8; ++b) {
            k |= static_cast<uint64_t>(bytes[i + static_cast<size_t>(b)]) << (8 * b);
        }
        k *= c1;
        k = std::rotl(k, 31);
        k *= c2;
        h ^= k;
        h = std::rotl(h, 27) * 5u + 0x52dce729u;
    }
    if (i < size) {
        uint64_t k = 0;
        for (int b = 0; i + static_cast<size_t>(b) < size; ++b) {
            k |= static_cast<uint64_t>(bytes[i + static_cast<size_t>(b)]) << (8 * b);
        }
        k *= c1;
        k = std::rotl(k, 31);
        k *= c2;
        h ^= k;
    }
    h ^= static_cast<uint64_t>(size);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

uint64_t hash_key(const GraphicsPipelineKey& key) noexcept {
    return stable_hash64(&key, sizeof(key));
}

uint64_t hash_key(const ComputePipelineKey& key) noexcept {
    return stable_hash64(&key, sizeof(key));
}

// =====================================================================================================
// PSO translation
// =====================================================================================================

namespace {

constexpr BlendComponentKey kDefaultBlendComponent{WgpuBlendOperation::Add, WgpuBlendFactor::One,
                                                   WgpuBlendFactor::Zero, 0};
constexpr StencilFaceKey kDefaultStencilFace{WgpuCompareFunction::Always, WgpuStencilOperation::Keep,
                                             WgpuStencilOperation::Keep, WgpuStencilOperation::Keep};

bool same_component(const BlendComponentKey& a, const BlendComponentKey& b) noexcept {
    return a.operation == b.operation && a.src_factor == b.src_factor && a.dst_factor == b.dst_factor;
}

bool is_dual_source(WgpuBlendFactor f) noexcept {
    return f == WgpuBlendFactor::Src1 || f == WgpuBlendFactor::OneMinusSrc1 ||
           f == WgpuBlendFactor::Src1Alpha || f == WgpuBlendFactor::OneMinusSrc1Alpha;
}

WgpuBlendFactor drop_dual_source(WgpuBlendFactor f) noexcept {
    switch (f) {
        case WgpuBlendFactor::Src1: return WgpuBlendFactor::Src;
        case WgpuBlendFactor::OneMinusSrc1: return WgpuBlendFactor::OneMinusSrc;
        case WgpuBlendFactor::Src1Alpha: return WgpuBlendFactor::SrcAlpha;
        case WgpuBlendFactor::OneMinusSrc1Alpha: return WgpuBlendFactor::OneMinusSrcAlpha;
        default: return f;
    }
}

// In the alpha component only the factor's alpha matters, so colour factors collapse onto their
// alpha counterparts (exact) and src-alpha-saturated is 1.
WgpuBlendFactor alpha_canonical(WgpuBlendFactor f) noexcept {
    switch (f) {
        case WgpuBlendFactor::Src: return WgpuBlendFactor::SrcAlpha;
        case WgpuBlendFactor::OneMinusSrc: return WgpuBlendFactor::OneMinusSrcAlpha;
        case WgpuBlendFactor::Dst: return WgpuBlendFactor::DstAlpha;
        case WgpuBlendFactor::OneMinusDst: return WgpuBlendFactor::OneMinusDstAlpha;
        case WgpuBlendFactor::Src1: return WgpuBlendFactor::Src1Alpha;
        case WgpuBlendFactor::OneMinusSrc1: return WgpuBlendFactor::OneMinusSrc1Alpha;
        case WgpuBlendFactor::SrcAlphaSaturated: return WgpuBlendFactor::One;
        default: return f;
    }
}

BlendComponentKey lower_blend_component(D3D12Blend src, D3D12Blend dst, D3D12BlendOp op, bool alpha,
                                        uint32_t features, uint64_t* diag) {
    BlendComponentKey c = kDefaultBlendComponent;
    const auto mop = map_blend_op(op);
    const auto msrc = map_blend_factor(src);
    const auto mdst = map_blend_factor(dst);
    if (!mop || !msrc || !mdst) {
        *diag |= PipelineDiag::InvalidEnumValue;
    }
    c.operation = mop.value_or(WgpuBlendOperation::Add);
    c.src_factor = msrc.value_or(WgpuBlendFactor::One);
    c.dst_factor = mdst.value_or(WgpuBlendFactor::Zero);
    if (!alpha && (src == D3D12Blend::AlphaFactor || src == D3D12Blend::InvAlphaFactor ||
                   dst == D3D12Blend::AlphaFactor || dst == D3D12Blend::InvAlphaFactor)) {
        *diag |= PipelineDiag::AlphaFactorApproximated;
    }
    if ((is_dual_source(c.src_factor) || is_dual_source(c.dst_factor)) &&
        (features & WgpuFeature::DualSourceBlending) == 0) {
        *diag |= PipelineDiag::DualSourceBlendUnavailable;
        c.src_factor = drop_dual_source(c.src_factor);
        c.dst_factor = drop_dual_source(c.dst_factor);
    }
    if (alpha) {
        c.src_factor = alpha_canonical(c.src_factor);
        c.dst_factor = alpha_canonical(c.dst_factor);
    }
    if (c.operation == WgpuBlendOperation::Min || c.operation == WgpuBlendOperation::Max) {
        // D3D12 ignores factors for MIN/MAX; WebGPU requires them to be "one".
        c.src_factor = WgpuBlendFactor::One;
        c.dst_factor = WgpuBlendFactor::One;
    }
    return c;
}

ColorTargetKey lower_render_target_blend(const D3D12RenderTargetBlendDesc& rt, uint32_t features,
                                         uint64_t* diag) {
    ColorTargetKey ct{};
    ct.write_mask = static_cast<uint8_t>(rt.RenderTargetWriteMask & D3D12ColorWriteEnable::All);
    ct.color = kDefaultBlendComponent;
    ct.alpha = kDefaultBlendComponent;
    if (rt.LogicOpEnable) {
        switch (rt.LogicOp) {
            case D3D12LogicOp::Copy:
                break;  // identical to no blending
            case D3D12LogicOp::Noop:
                ct.write_mask = 0;  // identical to masking every channel
                break;
            default:
                *diag |= PipelineDiag::LogicOpUnsupported;
                if (static_cast<uint32_t>(rt.LogicOp) > static_cast<uint32_t>(D3D12LogicOp::OrInverted)) {
                    *diag |= PipelineDiag::InvalidEnumValue;
                }
                break;
        }
        return ct;
    }
    if (!rt.BlendEnable) {
        return ct;
    }
    ct.color = lower_blend_component(rt.SrcBlend, rt.DestBlend, rt.BlendOp, false, features, diag);
    ct.alpha = lower_blend_component(rt.SrcBlendAlpha, rt.DestBlendAlpha, rt.BlendOpAlpha, true, features, diag);
    const bool passthrough =
        same_component(ct.color, kDefaultBlendComponent) && same_component(ct.alpha, kDefaultBlendComponent);
    ct.blend_enabled = passthrough ? 0 : 1;
    return ct;
}

WgpuPrimitiveTopology default_topology(D3D12PrimitiveTopologyType type) noexcept {
    switch (type) {
        case D3D12PrimitiveTopologyType::Point: return WgpuPrimitiveTopology::PointList;
        case D3D12PrimitiveTopologyType::Line: return WgpuPrimitiveTopology::LineList;
        default: return WgpuPrimitiveTopology::TriangleList;
    }
}

bool is_strip(WgpuPrimitiveTopology t) noexcept {
    return t == WgpuPrimitiveTopology::TriangleStrip || t == WgpuPrimitiveTopology::LineStrip;
}

bool is_triangle(WgpuPrimitiveTopology t) noexcept {
    return t == WgpuPrimitiveTopology::TriangleList || t == WgpuPrimitiveTopology::TriangleStrip;
}

StencilFaceKey lower_stencil_face(const D3D12DepthStencilOpDesc& face, uint64_t* diag) {
    const auto cmp = map_compare_function(face.StencilFunc);
    const auto fail = map_stencil_op(face.StencilFailOp);
    const auto dfail = map_stencil_op(face.StencilDepthFailOp);
    const auto pass = map_stencil_op(face.StencilPassOp);
    if (!cmp || !fail || !dfail || !pass) {
        *diag |= PipelineDiag::InvalidEnumValue;
    }
    return StencilFaceKey{cmp.value_or(WgpuCompareFunction::Always), fail.value_or(WgpuStencilOperation::Keep),
                          dfail.value_or(WgpuStencilOperation::Keep), pass.value_or(WgpuStencilOperation::Keep)};
}

void canonicalize_stencil_masks(DepthStencilKey& ds) noexcept {
    const auto compare_irrelevant = [](WgpuCompareFunction c) {
        return c == WgpuCompareFunction::Always || c == WgpuCompareFunction::Never;
    };
    const auto writes = [](const StencilFaceKey& f) {
        return f.fail_op != WgpuStencilOperation::Keep || f.depth_fail_op != WgpuStencilOperation::Keep ||
               f.pass_op != WgpuStencilOperation::Keep;
    };
    if (compare_irrelevant(ds.stencil_front.compare) && compare_irrelevant(ds.stencil_back.compare)) {
        ds.stencil_read_mask = 0;
    }
    if (!writes(ds.stencil_front) && !writes(ds.stencil_back)) {
        ds.stencil_write_mask = 0;
    }
}

void lower_input_layout(const D3D12GraphicsPsoDesc& desc, GraphicsPsoTranslation& t) {
    struct Attr {
        uint32_t slot;
        uint32_t offset;
        uint32_t location;
        WgpuVertexFormat format;
    };
    std::vector<Attr> attrs;
    uint32_t slot_end[kD3D12IaVertexInputSlotCount] = {};
    WgpuVertexStepMode slot_step[kMaxVertexBuffers] = {};
    uint32_t location_mask = 0;
    std::vector<bool> signature_matched(desc.vs_input_signature != nullptr ? desc.vs_input_signature_count : 0u,
                                        false);
    uint64_t& diag = t.diagnostics;

    const uint32_t count = desc.pInputElementDescs != nullptr ? desc.NumInputElements : 0u;
    for (uint32_t i = 0; i < count; ++i) {
        const D3D12InputElementDesc& e = desc.pInputElementDescs[i];
        const VertexFormatMapping vf = map_vertex_format(e.Format);
        if (!vf.supported()) {
            diag |= PipelineDiag::VertexFormatUnsupported;
            continue;
        }
        if (e.InputSlot >= kD3D12IaVertexInputSlotCount) {
            diag |= PipelineDiag::VertexSlotOutOfRange;
            continue;
        }
        uint32_t offset = e.AlignedByteOffset;
        if (offset == kD3D12AppendAlignedElement) {
            offset = align_up(slot_end[e.InputSlot], std::min(4u, vf.byte_size));
        }
        slot_end[e.InputSlot] = std::max(slot_end[e.InputSlot], offset + vf.byte_size);
        if (e.InputSlot >= kMaxVertexBuffers) {
            diag |= PipelineDiag::VertexSlotOutOfRange;
            continue;
        }

        uint32_t location = i;
        if (desc.vs_input_signature != nullptr) {
            bool found = false;
            for (uint32_t s = 0; s < desc.vs_input_signature_count; ++s) {
                const auto& sig = desc.vs_input_signature[s];
                if (sig.semantic_index == e.SemanticIndex && semantic_equal(sig.semantic_name, e.SemanticName)) {
                    location = sig.location;
                    signature_matched[s] = true;
                    found = true;
                    break;
                }
            }
            if (!found) {
                continue;  // element not consumed by the vertex shader: legal in D3D12, dropped
            }
        }
        if (location >= kMaxVertexAttributes) {
            diag |= PipelineDiag::TooManyVertexAttributes;
            continue;
        }
        if ((location_mask & (1u << location)) != 0) {
            diag |= PipelineDiag::VertexLayoutInvalid;  // two elements feeding one location
            continue;
        }

        WgpuVertexStepMode step = WgpuVertexStepMode::Vertex;
        if (e.InputSlotClass == D3D12InputClassification::PerInstanceData) {
            step = WgpuVertexStepMode::Instance;
            if (e.InstanceDataStepRate == 0) {
                t.instance_stride_zero_slots |= 1u << e.InputSlot;  // every instance reads element 0
            } else if (e.InstanceDataStepRate > 1) {
                diag |= PipelineDiag::InstanceStepRateUnsupported;
            }
        } else if (e.InputSlotClass != D3D12InputClassification::PerVertexData) {
            diag |= PipelineDiag::InvalidEnumValue;
        }
        if (slot_step[e.InputSlot] == WgpuVertexStepMode::Unused) {
            slot_step[e.InputSlot] = step;
        } else if (slot_step[e.InputSlot] != step) {
            diag |= PipelineDiag::VertexLayoutInvalid;  // mixed classification in one slot
        }

        location_mask |= 1u << location;
        attrs.push_back(Attr{e.InputSlot, offset, location, vf.format});
    }

    for (size_t s = 0; s < signature_matched.size(); ++s) {
        if (!signature_matched[s]) {
            diag |= PipelineDiag::MissingVertexInput;
        }
    }

    std::sort(attrs.begin(), attrs.end(), [](const Attr& a, const Attr& b) {
        return a.slot != b.slot ? a.slot < b.slot : a.location < b.location;
    });

    GraphicsPipelineKey& k = t.base;
    for (size_t i = 0; i < attrs.size(); ++i) {
        VertexAttributeKey& a = k.vertex_attributes[i];
        a.offset = attrs[i].offset;
        a.format = attrs[i].format;
        a.shader_location = static_cast<uint8_t>(attrs[i].location);
        a.buffer_index = static_cast<uint8_t>(attrs[i].slot);
        VertexBufferKey& b = k.vertex_buffers[attrs[i].slot];
        b.step_mode = slot_step[attrs[i].slot];
        ++b.attribute_count;
        t.used_vertex_slots |= 1u << attrs[i].slot;
        k.vertex_buffer_count = std::max(k.vertex_buffer_count, attrs[i].slot + 1u);
    }
    k.vertex_attribute_count = static_cast<uint32_t>(attrs.size());
    t.instance_stride_zero_slots &= t.used_vertex_slots;
    for (uint32_t s = 0; s < kMaxVertexBuffers; ++s) {
        if ((t.used_vertex_slots & (1u << s)) != 0) {
            t.tight_strides[s] = align_up(slot_end[s], 4u);
            k.vertex_buffers[s].array_stride =
                (t.instance_stride_zero_slots & (1u << s)) != 0 ? 0u : t.tight_strides[s];
        }
    }
}

}  // namespace

GraphicsPsoTranslation translate_graphics_pso(const D3D12GraphicsPsoDesc& desc, const PipelineKeyOptions& options) {
    GraphicsPsoTranslation t{};
    GraphicsPipelineKey& k = t.base;
    std::memset(&k, 0, sizeof(k));
    uint64_t& diag = t.diagnostics;
    const uint32_t features = options.features;

    // Shader stages.
    k.vs_shader = desc.vs;
    k.ps_shader = desc.ps;
    k.root_signature_id = desc.root_signature_id;
    if (desc.gs != 0) {
        diag |= PipelineDiag::GeometryShaderUnsupported;
    }
    if (desc.hs != 0 || desc.ds != 0) {
        diag |= PipelineDiag::TessellationUnsupported;
    }
    if (desc.as != 0 || desc.ms != 0) {
        diag |= PipelineDiag::MeshShaderUnsupported;
    } else if (desc.vs == 0) {
        diag |= PipelineDiag::MissingVertexShader;
    }
    if (desc.stream_output_enabled) {
        diag |= PipelineDiag::StreamOutputUnsupported;
    }

    // Primitive.
    t.topology_type = desc.PrimitiveTopologyType;
    switch (desc.PrimitiveTopologyType) {
        case D3D12PrimitiveTopologyType::Point:
        case D3D12PrimitiveTopologyType::Line:
        case D3D12PrimitiveTopologyType::Triangle:
            break;
        case D3D12PrimitiveTopologyType::Patch:
            diag |= PipelineDiag::TessellationUnsupported;
            break;
        default:
            diag |= PipelineDiag::InvalidEnumValue;
            t.topology_type = D3D12PrimitiveTopologyType::Triangle;
            break;
    }
    k.primitive.topology = default_topology(t.topology_type);
    k.primitive.strip_index_format = WgpuIndexFormat::Undefined;
    t.strip_cut_value = desc.IBStripCutValue;
    if (static_cast<uint32_t>(desc.IBStripCutValue) > 2u) {
        diag |= PipelineDiag::InvalidEnumValue;
        t.strip_cut_value = D3D12IndexBufferStripCutValue::Disabled;
    }

    const D3D12RasterizerDesc& rs = desc.RasterizerState;
    if (rs.FillMode == D3D12FillMode::Wireframe) {
        diag |= PipelineDiag::WireframeFillUnsupported;
    } else if (rs.FillMode != D3D12FillMode::Solid) {
        diag |= PipelineDiag::InvalidEnumValue;
    }
    const auto cull = map_cull_mode(rs.CullMode);
    if (!cull) {
        diag |= PipelineDiag::InvalidEnumValue;
    }
    k.primitive.cull_mode = cull.value_or(WgpuCullMode::Back);
    k.primitive.front_face = map_front_face(rs.FrontCounterClockwise);
    if (!rs.DepthClipEnable) {
        if ((features & WgpuFeature::DepthClipControl) != 0) {
            k.primitive.unclipped_depth = 1;
        } else {
            diag |= PipelineDiag::DepthClipControlUnavailable;
        }
    }
    if (rs.ConservativeRaster != D3D12ConservativeRasterizationMode::Off) {
        diag |= PipelineDiag::ConservativeRasterUnsupported;
    }
    if (rs.ForcedSampleCount != 0) {
        diag |= PipelineDiag::ForcedSampleCountUnsupported;
    }
    if (t.topology_type == D3D12PrimitiveTopologyType::Line && (rs.AntialiasedLineEnable || rs.MultisampleEnable)) {
        diag |= PipelineDiag::LineRasterModeIgnored;
    }

    // Depth-stencil (format-independent part).
    const D3D12DepthStencilDesc& dss = desc.DepthStencilState;
    DepthStencilKey& ds = k.depth_stencil;
    if (dss.DepthEnable) {
        const auto cmp = map_compare_function(dss.DepthFunc);
        if (!cmp) {
            diag |= PipelineDiag::InvalidEnumValue;
        }
        ds.depth_compare = cmp.value_or(WgpuCompareFunction::Always);
        if (dss.DepthWriteMask != D3D12DepthWriteMask::Zero && dss.DepthWriteMask != D3D12DepthWriteMask::All) {
            diag |= PipelineDiag::InvalidEnumValue;
        }
        ds.depth_write_enabled = dss.DepthWriteMask == D3D12DepthWriteMask::All ? 1 : 0;
    } else {
        ds.depth_compare = WgpuCompareFunction::Always;
        ds.depth_write_enabled = 0;
    }
    if (dss.StencilEnable) {
        ds.stencil_front = lower_stencil_face(dss.FrontFace, &diag);
        ds.stencil_back = lower_stencil_face(dss.BackFace, &diag);
        ds.stencil_read_mask = dss.StencilReadMask;
        ds.stencil_write_mask = dss.StencilWriteMask;
    } else {
        ds.stencil_front = kDefaultStencilFace;
        ds.stencil_back = kDefaultStencilFace;
    }
    canonicalize_stencil_masks(ds);
    ds.depth_bias = rs.DepthBias;
    ds.depth_bias_slope_scale_bits = canonical_float_bits(rs.SlopeScaledDepthBias);
    ds.depth_bias_clamp_bits = canonical_float_bits(rs.DepthBiasClamp);
    if (dss.DepthBoundsTestEnable) {
        diag |= PipelineDiag::DepthBoundsUnsupported;
    }

    // Blend (format-independent part).
    const D3D12BlendDesc& bs = desc.BlendState;
    for (uint32_t i = 0; i < kMaxColorTargets; ++i) {
        const D3D12RenderTargetBlendDesc& rt = bs.IndependentBlendEnable ? bs.RenderTarget[i] : bs.RenderTarget[0];
        k.color_targets[i] = lower_render_target_blend(rt, features, &diag);
    }
    k.multisample.alpha_to_coverage_enabled = bs.AlphaToCoverageEnable ? 1 : 0;

    // Formats and sampling as declared by the PSO.
    t.num_render_targets = std::min(desc.NumRenderTargets, kMaxColorTargets);
    for (uint32_t i = 0; i < t.num_render_targets; ++i) {
        t.rtv_formats[i] = desc.RTVFormats[i];
    }
    t.dsv_format = desc.DSVFormat;
    t.sample_count = desc.SampleDesc.Count == 0 ? 1u : desc.SampleDesc.Count;
    if (t.sample_count > 1 && desc.SampleDesc.Quality != 0) {
        diag |= PipelineDiag::SampleQualityIgnored;
    }
    t.sample_mask = desc.SampleMask;
    t.ps_output_mask = desc.ps != 0 ? (desc.ps_output_mask & 0xFFu) : 0u;

    lower_input_layout(desc, t);
    return t;
}

uint64_t specialize_graphics_pipeline_key(const GraphicsPsoTranslation& t,
                                          const PipelineVariantState& v,
                                          const PipelineKeyOptions& options,
                                          GraphicsPipelineKey* out_key) {
    GraphicsPipelineKey k = t.base;
    uint64_t diag = 0;
    const uint32_t features = options.features;
    const bool bound = (v.flags & VariantFlags::BoundTargets) != 0;

    // --- Target formats: the bound views are authoritative (WebGPU validates pass/pipeline equality).
    DxgiFormat rtv[kMaxColorTargets] = {};
    DxgiFormat dsv = t.dsv_format;
    uint32_t samples = t.sample_count;
    for (uint32_t i = 0; i < t.num_render_targets; ++i) {
        rtv[i] = t.rtv_formats[i];
    }
    if (bound) {
        const uint32_t n = std::min(v.num_render_targets, kMaxColorTargets);
        for (uint32_t i = 0; i < kMaxColorTargets; ++i) {
            const DxgiFormat b = i < n ? v.rtv_formats[i] : DxgiFormat::UNKNOWN;
            if (b != DxgiFormat::UNKNOWN && rtv[i] != DxgiFormat::UNKNOWN && b != rtv[i]) {
                diag |= PipelineDiag::RenderTargetFormatMismatch;
            }
            rtv[i] = b;
        }
        if (v.dsv_format != DxgiFormat::UNKNOWN && dsv != DxgiFormat::UNKNOWN && v.dsv_format != dsv) {
            diag |= PipelineDiag::DepthFormatMismatch;
        }
        dsv = v.dsv_format;
        const uint32_t bound_samples = v.sample_count == 0 ? 1u : v.sample_count;
        if (bound_samples != samples) {
            diag |= PipelineDiag::SampleCountMismatch;
        }
        samples = bound_samples;
    }

    const uint32_t count = map_sample_count(samples);
    if (samples > 1 && samples != 4) {
        diag |= PipelineDiag::SampleCountRounded;
    }

    // --- Color targets.
    uint32_t highest = 0;
    bool any_dual_source = false;
    WgpuTextureFormat formats[kMaxColorTargets] = {};
    for (uint32_t i = 0; i < kMaxColorTargets; ++i) {
        ColorTargetKey& ct = k.color_targets[i];
        if (rtv[i] == DxgiFormat::UNKNOWN) {
            ct = ColorTargetKey{};
            continue;
        }
        const TextureFormatMapping m = map_texture_format(rtv[i], FormatUsage::RenderTarget, features);
        if (!m.supported()) {
            diag |= PipelineDiag::UnsupportedTargetFormat;
            ct = ColorTargetKey{};
            continue;
        }
        if ((m.diagnostics & FormatDiag::FeatureFallback) != 0) {
            diag |= PipelineDiag::TargetFormatFallback;
        }
        ct.format = m.format;
        formats[i] = m.format;
        const bool pso_declares = i < t.num_render_targets && t.rtv_formats[i] != DxgiFormat::UNKNOWN;
        const bool ps_writes = pso_declares && (t.ps_output_mask & (1u << i)) != 0;
        if (!ps_writes) {
            ct.write_mask = 0;
            ct.blend_enabled = 0;
        }
        if (ct.blend_enabled != 0 && !is_blendable(ct.format, features)) {
            diag |= PipelineDiag::BlendOnNonBlendableFormat;
            ct.blend_enabled = 0;
        }
        if (ct.blend_enabled == 0) {
            ct.color = kDefaultBlendComponent;
            ct.alpha = kDefaultBlendComponent;
        } else if (is_dual_source(ct.color.src_factor) || is_dual_source(ct.color.dst_factor) ||
                   is_dual_source(ct.alpha.src_factor) || is_dual_source(ct.alpha.dst_factor)) {
            any_dual_source = true;
        }
        if (count > 1 && !is_multisample_capable(ct.format, features)) {
            diag |= PipelineDiag::UnsupportedTargetFormat;
        }
        highest = i + 1;
    }
    k.color_target_count = highest;
    if (any_dual_source && highest > 1) {
        diag |= PipelineDiag::DualSourceMultipleTargets;
    }
    if (k.ps_shader == 0 && highest > 0) {
        diag |= PipelineDiag::NullPixelShaderWithColorTargets;
    }
    if (color_attachment_bytes_per_sample(std::span<const WgpuTextureFormat>(formats, highest)) >
        options.max_color_attachment_bytes_per_sample) {
        diag |= PipelineDiag::ColorAttachmentBytesExceeded;
    }

    // --- Depth-stencil.
    DepthStencilKey& ds = k.depth_stencil;
    if (dsv == DxgiFormat::UNKNOWN) {
        ds = DepthStencilKey{};
    } else {
        const TextureFormatMapping m = map_texture_format(dsv, FormatUsage::DepthStencil, features);
        if (!m.supported()) {
            diag |= PipelineDiag::UnsupportedTargetFormat;
            ds = DepthStencilKey{};
        } else {
            if ((m.diagnostics & FormatDiag::FeatureFallback) != 0) {
                diag |= PipelineDiag::TargetFormatFallback;
            }
            ds.format = m.format;
            if (!has_depth_aspect(ds.format)) {
                ds.depth_compare = WgpuCompareFunction::Always;
                ds.depth_write_enabled = 0;
            }
            if (!has_stencil_aspect(ds.format)) {
                ds.stencil_front = kDefaultStencilFace;
                ds.stencil_back = kDefaultStencilFace;
                ds.stencil_read_mask = 0;
                ds.stencil_write_mask = 0;
            }
        }
    }
    if (highest == 0 && ds.format == WgpuTextureFormat::Undefined) {
        diag |= PipelineDiag::NoAttachments;
    }

    // --- Topology (dynamic in D3D12, pipeline state in WebGPU).
    if (v.topology != D3DPrimitiveTopology::Undefined) {
        const TopologyMapping tm = map_primitive_topology(v.topology);
        diag |= tm.diagnostics;
        k.primitive.topology = tm.topology;
        if (tm.type != D3D12PrimitiveTopologyType::Undefined && tm.type != t.topology_type) {
            diag |= PipelineDiag::TopologyTypeMismatch;
        }
    }
    if (is_strip(k.primitive.topology)) {
        WgpuIndexFormat cut = WgpuIndexFormat::Undefined;
        if (t.strip_cut_value == D3D12IndexBufferStripCutValue::Value0xFFFF) {
            cut = WgpuIndexFormat::Uint16;
        } else if (t.strip_cut_value == D3D12IndexBufferStripCutValue::Value0xFFFFFFFF) {
            cut = WgpuIndexFormat::Uint32;
        }
        if (v.index_format != WgpuIndexFormat::Undefined) {
            // WebGPU always restarts strips at the all-ones index of the bound index format.
            if (cut != v.index_format) {
                diag |= PipelineDiag::StripCutValueMismatch;
            }
            k.primitive.strip_index_format = v.index_format;
        } else {
            k.primitive.strip_index_format = cut;
        }
    } else {
        k.primitive.strip_index_format = WgpuIndexFormat::Undefined;
    }

    // --- Depth bias.
    const bool bias_set = ds.depth_bias != 0 || ds.depth_bias_slope_scale_bits != 0 || ds.depth_bias_clamp_bits != 0;
    const bool depth_used = ds.format != WgpuTextureFormat::Undefined && has_depth_aspect(ds.format) &&
                            (ds.depth_write_enabled != 0 || ds.depth_compare != WgpuCompareFunction::Always);
    if (!depth_used) {
        ds.depth_bias = 0;
        ds.depth_bias_slope_scale_bits = 0;
        ds.depth_bias_clamp_bits = 0;
    } else if (bias_set && !is_triangle(k.primitive.topology)) {
        diag |= PipelineDiag::DepthBiasDroppedForNonTriangle;
        ds.depth_bias = 0;
        ds.depth_bias_slope_scale_bits = 0;
        ds.depth_bias_clamp_bits = 0;
    }

    // --- Multisample.
    k.multisample.count = count;
    k.multisample.mask = t.sample_mask & ((1u << count) - 1u);
    if (k.multisample.alpha_to_coverage_enabled != 0) {
        const ColorTargetKey& rt0 = k.color_targets[0];
        const bool ok = count > 1 && rt0.format != WgpuTextureFormat::Undefined &&
                        is_blendable(rt0.format, features) && has_alpha_channel(rt0.format);
        if (!ok) {
            diag |= PipelineDiag::AlphaToCoverageDropped;
            k.multisample.alpha_to_coverage_enabled = 0;
        }
    }

    // --- Vertex buffer strides (dynamic in D3D12 via IASetVertexBuffers).
    for (uint32_t s = 0; s < kMaxVertexBuffers; ++s) {
        VertexBufferKey& b = k.vertex_buffers[s];
        if ((t.used_vertex_slots & (1u << s)) == 0) {
            continue;
        }
        uint32_t stride = t.tight_strides[s];
        if ((t.instance_stride_zero_slots & (1u << s)) != 0) {
            stride = 0;
        } else if ((v.bound_vertex_slots & (1u << s)) != 0) {
            stride = v.vertex_strides[s];
        }
        b.array_stride = stride;
        if (stride % 4u != 0 || stride > kMaxVertexBufferArrayStride) {
            diag |= PipelineDiag::VertexLayoutInvalid;
        }
    }
    for (uint32_t i = 0; i < k.vertex_attribute_count; ++i) {
        const VertexAttributeKey& a = k.vertex_attributes[i];
        const uint32_t size = vertex_format_byte_size(a.format);
        const uint32_t stride = k.vertex_buffers[a.buffer_index].array_stride;
        const uint32_t limit = stride == 0 ? kMaxVertexBufferArrayStride : stride;
        if (a.offset + size > limit || a.offset % std::min(4u, size) != 0) {
            diag |= PipelineDiag::VertexLayoutInvalid;
        }
    }

    *out_key = k;
    return diag;
}

GraphicsKeyResult make_graphics_pipeline_key(const D3D12GraphicsPsoDesc& desc,
                                             const PipelineVariantState& variant,
                                             const PipelineKeyOptions& options) {
    const GraphicsPsoTranslation t = translate_graphics_pso(desc, options);
    GraphicsKeyResult r{};
    r.diagnostics = t.diagnostics | specialize_graphics_pipeline_key(t, variant, options, &r.key);
    return r;
}

ComputePipelineKey make_compute_pipeline_key(const D3D12ComputePsoDesc& desc) noexcept {
    ComputePipelineKey k{};
    k.cs_shader = desc.cs;
    k.root_signature_id = desc.root_signature_id;
    k.reserved = 0;
    return k;
}

// =====================================================================================================
// PipelineCache
// =====================================================================================================

namespace {

constexpr size_t kShardCount = 16;

template <typename Key>
class ShardedPipelineMap {
public:
    struct Node {
        Key key;
        uint32_t id;
        std::atomic<uint32_t> refs;
        Node(const Key& k, uint32_t i) : key(k), id(i), refs(1) {}
    };

    struct FindResult {
        uint32_t id = 0;
        bool found = false;
    };

    // Returns {id, created, collided}.
    template <typename Alloc>
    std::tuple<uint32_t, bool, bool> acquire(uint64_t hash, const Key& key, Alloc&& alloc) {
        Shard& shard = shards_[hash % kShardCount];
        {
            std::shared_lock lock(shard.mutex);
            if (Node* n = find_in(shard, hash, key)) {
                n->refs.fetch_add(1, std::memory_order_relaxed);
                return {n->id, false, false};
            }
        }
        std::unique_lock lock(shard.mutex);
        if (Node* n = find_in(shard, hash, key)) {
            n->refs.fetch_add(1, std::memory_order_relaxed);
            return {n->id, false, false};
        }
        auto& bucket = shard.buckets[hash];
        const bool collided = !bucket.empty();
        const uint32_t id = alloc();
        bucket.push_back(std::make_unique<Node>(key, id));
        return {id, true, collided};
    }

    // Returns {id, evicted}; id 0 when the key is unknown.
    std::pair<uint32_t, bool> release(uint64_t hash, const Key& key, bool evict_unreferenced) {
        Shard& shard = shards_[hash % kShardCount];
        std::unique_lock lock(shard.mutex);
        auto it = shard.buckets.find(hash);
        if (it == shard.buckets.end()) {
            return {0, false};
        }
        auto& bucket = it->second;
        for (size_t i = 0; i < bucket.size(); ++i) {
            Node& n = *bucket[i];
            if (!(n.key == key)) {
                continue;
            }
            uint32_t refs = n.refs.load(std::memory_order_relaxed);
            if (refs > 0) {
                n.refs.store(--refs, std::memory_order_relaxed);
            }
            if (refs == 0 && evict_unreferenced) {
                const uint32_t id = n.id;
                bucket.erase(bucket.begin() + static_cast<std::ptrdiff_t>(i));
                if (bucket.empty()) {
                    shard.buckets.erase(it);
                }
                return {id, true};
            }
            return {n.id, false};
        }
        return {0, false};
    }

    std::optional<std::pair<uint32_t, uint32_t>> find(uint64_t hash, const Key& key) const {
        const Shard& shard = shards_[hash % kShardCount];
        std::shared_lock lock(shard.mutex);
        if (const Node* n = find_in(shard, hash, key)) {
            return std::make_pair(n->id, n->refs.load(std::memory_order_relaxed));
        }
        return std::nullopt;
    }

private:
    struct Shard {
        mutable std::shared_mutex mutex;
        std::unordered_map<uint64_t, std::vector<std::unique_ptr<Node>>> buckets;
    };

    template <typename S>
    static auto find_in(S& shard, uint64_t hash, const Key& key) -> decltype(shard.buckets.begin()->second[0].get()) {
        auto it = shard.buckets.find(hash);
        if (it == shard.buckets.end()) {
            return nullptr;
        }
        for (const auto& node : it->second) {
            if (node->key == key) {  // full-key equality guards against 64-bit hash collisions
                return node.get();
            }
        }
        return nullptr;
    }

    std::array<Shard, kShardCount> shards_;
};

}  // namespace

struct PipelineCache::Impl {
    PipelineCacheOptions options;
    ShardedPipelineMap<GraphicsPipelineKey> graphics;
    ShardedPipelineMap<ComputePipelineKey> compute;
    std::atomic<uint32_t> next_id{1};
    std::atomic<uint64_t> lookups{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> variants{0};
    std::atomic<uint64_t> collisions{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> live_graphics{0};
    std::atomic<uint64_t> live_compute{0};

    uint64_t hash(const void* data, size_t size) const noexcept {
        return options.hash_override != nullptr ? options.hash_override(data, size) : stable_hash64(data, size);
    }

    uint32_t allocate_id() {
        if (options.id_allocator) {
            return options.id_allocator();
        }
        return next_id.fetch_add(1, std::memory_order_relaxed);
    }

    template <typename Map, typename Key>
    CacheLookup acquire(Map& map, const Key& key, std::atomic<uint64_t>& live) {
        lookups.fetch_add(1, std::memory_order_relaxed);
        const auto [id, created, collided] = map.acquire(hash(&key, sizeof(key)), key, [this] { return allocate_id(); });
        if (created) {
            misses.fetch_add(1, std::memory_order_relaxed);
            live.fetch_add(1, std::memory_order_relaxed);
            if (collided) {
                collisions.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            hits.fetch_add(1, std::memory_order_relaxed);
        }
        return CacheLookup{id, created};
    }

    template <typename Map, typename Key>
    std::optional<uint32_t> release(Map& map, const Key& key, std::atomic<uint64_t>& live) {
        const auto [id, evicted] =
            map.release(hash(&key, sizeof(key)), key, options.eviction == EvictionPolicy::EvictUnreferenced);
        if (!evicted) {
            return std::nullopt;
        }
        evictions.fetch_add(1, std::memory_order_relaxed);
        live.fetch_sub(1, std::memory_order_relaxed);
        return id;
    }
};

PipelineCache::PipelineCache(PipelineCacheOptions options) : impl_(std::make_unique<Impl>()) {
    impl_->options = std::move(options);
}

PipelineCache::~PipelineCache() = default;

CacheLookup PipelineCache::acquire(const GraphicsPipelineKey& key) {
    return impl_->acquire(impl_->graphics, key, impl_->live_graphics);
}

CacheLookup PipelineCache::acquire(const ComputePipelineKey& key) {
    return impl_->acquire(impl_->compute, key, impl_->live_compute);
}

std::optional<uint32_t> PipelineCache::release(const GraphicsPipelineKey& key) {
    return impl_->release(impl_->graphics, key, impl_->live_graphics);
}

std::optional<uint32_t> PipelineCache::release(const ComputePipelineKey& key) {
    return impl_->release(impl_->compute, key, impl_->live_compute);
}

std::optional<uint32_t> PipelineCache::find(const GraphicsPipelineKey& key) const {
    const auto r = impl_->graphics.find(impl_->hash(&key, sizeof(key)), key);
    return r ? std::optional<uint32_t>(r->first) : std::nullopt;
}

std::optional<uint32_t> PipelineCache::find(const ComputePipelineKey& key) const {
    const auto r = impl_->compute.find(impl_->hash(&key, sizeof(key)), key);
    return r ? std::optional<uint32_t>(r->first) : std::nullopt;
}

uint32_t PipelineCache::ref_count(const GraphicsPipelineKey& key) const {
    const auto r = impl_->graphics.find(impl_->hash(&key, sizeof(key)), key);
    return r ? r->second : 0u;
}

PipelineCacheStats PipelineCache::stats() const {
    PipelineCacheStats s{};
    s.lookups = impl_->lookups.load(std::memory_order_relaxed);
    s.hits = impl_->hits.load(std::memory_order_relaxed);
    s.misses = impl_->misses.load(std::memory_order_relaxed);
    s.variants = impl_->variants.load(std::memory_order_relaxed);
    s.hash_collisions = impl_->collisions.load(std::memory_order_relaxed);
    s.evictions = impl_->evictions.load(std::memory_order_relaxed);
    s.live_graphics = impl_->live_graphics.load(std::memory_order_relaxed);
    s.live_compute = impl_->live_compute.load(std::memory_order_relaxed);
    return s;
}

void PipelineCache::note_variant() noexcept {
    impl_->variants.fetch_add(1, std::memory_order_relaxed);
}

// =====================================================================================================
// GraphicsPso
// =====================================================================================================

GraphicsPso::GraphicsPso(const D3D12GraphicsPsoDesc& desc, const PipelineKeyOptions& options)
    : translation_(translate_graphics_pso(desc, options)), options_(options) {}

GraphicsPso::~GraphicsPso() = default;

PipelineVariantState GraphicsPso::canonical_variant(const PipelineVariantState& v) const noexcept {
    PipelineVariantState c{};
    c.topology = v.topology;
    c.flags = v.flags & VariantFlags::BoundTargets;
    if (c.flags != 0) {
        c.num_render_targets = std::min(v.num_render_targets, kMaxColorTargets);
        for (uint32_t i = 0; i < c.num_render_targets; ++i) {
            c.rtv_formats[i] = v.rtv_formats[i];
        }
        c.dsv_format = v.dsv_format;
        c.sample_count = v.sample_count;
    }
    const WgpuPrimitiveTopology topo = v.topology == D3DPrimitiveTopology::Undefined
                                           ? translation_.base.primitive.topology
                                           : map_primitive_topology(v.topology).topology;
    c.index_format = is_strip(topo) ? v.index_format : WgpuIndexFormat::Undefined;
    const uint32_t relevant = translation_.used_vertex_slots & ~translation_.instance_stride_zero_slots;
    c.bound_vertex_slots = v.bound_vertex_slots & relevant;
    for (uint32_t s = 0; s < kMaxVertexBuffers; ++s) {
        if ((c.bound_vertex_slots & (1u << s)) != 0) {
            c.vertex_strides[s] = v.vertex_strides[s];
        }
    }
    return c;
}

PipelineResolution GraphicsPso::resolve(PipelineCache& cache, const PipelineVariantState& variant) {
    const PipelineVariantState state = canonical_variant(variant);
    std::lock_guard lock(mutex_);

    auto matches = [&](const Variant& v) { return std::memcmp(&v.state, &state, sizeof(state)) == 0; };
    auto hit = [&](size_t i) {
        last_hit_ = i;
        const Variant& v = *variants_[i];
        return PipelineResolution{v.pipeline_id, false, &v.key, v.diagnostics};
    };
    if (last_hit_ < variants_.size() && matches(*variants_[last_hit_])) {
        return hit(last_hit_);
    }
    for (size_t i = 0; i < variants_.size(); ++i) {
        if (matches(*variants_[i])) {
            return hit(i);
        }
    }

    auto v = std::make_unique<Variant>();
    v->state = state;
    v->diagnostics = translation_.diagnostics |
                     specialize_graphics_pipeline_key(translation_, state, options_, &v->key);
    v->pipeline_id = 0;
    bool created = false;
    if (!is_fatal(v->diagnostics)) {
        const CacheLookup lookup = cache.acquire(v->key);
        v->pipeline_id = lookup.pipeline_id;
        created = lookup.created;
        bool had_pipeline = false;
        bool same_pipeline = false;
        for (const auto& existing : variants_) {
            had_pipeline = had_pipeline || existing->pipeline_id != 0;
            same_pipeline = same_pipeline || existing->pipeline_id == v->pipeline_id;
        }
        if (had_pipeline && !same_pipeline) {
            cache.note_variant();
        }
    }
    variants_.push_back(std::move(v));
    last_hit_ = variants_.size() - 1;
    const Variant& stored = *variants_.back();
    return PipelineResolution{stored.pipeline_id, created, &stored.key, stored.diagnostics};
}

std::vector<uint32_t> GraphicsPso::release(PipelineCache& cache) {
    std::lock_guard lock(mutex_);
    std::vector<uint32_t> evicted;
    for (const auto& v : variants_) {
        if (v->pipeline_id == 0) {
            continue;
        }
        if (const auto id = cache.release(v->key)) {
            evicted.push_back(*id);
        }
    }
    variants_.clear();
    last_hit_ = 0;
    return evicted;
}

size_t GraphicsPso::variant_count() const {
    std::lock_guard lock(mutex_);
    return variants_.size();
}

// =====================================================================================================
// CommandListStateTracker
// =====================================================================================================

namespace {

bool same_rtv(const RenderTargetViewInfo& a, const RenderTargetViewInfo& b) noexcept {
    return a.view_id == b.view_id && a.format == b.format && a.width == b.width && a.height == b.height &&
           a.sample_count == b.sample_count;
}

bool same_dsv(const DepthStencilViewInfo& a, const DepthStencilViewInfo& b) noexcept {
    return a.view_id == b.view_id && a.format == b.format && a.width == b.width && a.height == b.height &&
           a.sample_count == b.sample_count && a.flags == b.flags;
}

bool same_viewport(const D3D12Viewport& a, const D3D12Viewport& b) noexcept {
    return a.TopLeftX == b.TopLeftX && a.TopLeftY == b.TopLeftY && a.Width == b.Width && a.Height == b.Height &&
           a.MinDepth == b.MinDepth && a.MaxDepth == b.MaxDepth;
}

bool same_rect(const D3D12Rect& a, const D3D12Rect& b) noexcept {
    return a.left == b.left && a.top == b.top && a.right == b.right && a.bottom == b.bottom;
}

float finite_or_zero(float v) noexcept {
    return std::isfinite(v) ? v : 0.0f;
}

}  // namespace

bool CommandListStateTracker::Attachments::operator==(const Attachments& other) const noexcept {
    if (num_rtvs != other.num_rtvs || !same_dsv(dsv, other.dsv)) {
        return false;
    }
    for (uint32_t i = 0; i < num_rtvs; ++i) {
        if (!same_rtv(rtvs[i], other.rtvs[i])) {
            return false;
        }
    }
    return true;
}

bool CommandListStateTracker::Attachments::empty() const noexcept {
    return num_rtvs == 0 && dsv.view_id == 0;
}

bool CommandListStateTracker::Attachments::contains(uint32_t view_id) const noexcept {
    if (view_id == 0) {
        return false;
    }
    if (dsv.view_id == view_id) {
        return true;
    }
    for (uint32_t i = 0; i < num_rtvs; ++i) {
        if (rtvs[i].view_id == view_id) {
            return true;
        }
    }
    return false;
}

CommandListStateTracker::CommandListStateTracker(TrackerOptions options) : options_(options) {}

void CommandListStateTracker::reset() {
    const TrackerOptions options = options_;
    *this = CommandListStateTracker(options);
}

void CommandListStateTracker::clear_events() {
    events_.clear();
    passes_.clear();
    clear_fallbacks_.clear();
}

void CommandListStateTracker::update_variant_targets() {
    variant_.flags |= VariantFlags::BoundTargets;
    variant_.num_render_targets = bound_.num_rtvs;
    uint32_t samples = 0;
    for (uint32_t i = 0; i < kMaxColorTargets; ++i) {
        const RenderTargetViewInfo& r = bound_.rtvs[i];
        variant_.rtv_formats[i] = (i < bound_.num_rtvs && r.view_id != 0) ? r.format : DxgiFormat::UNKNOWN;
        if (i < bound_.num_rtvs && r.view_id != 0 && samples == 0) {
            samples = r.sample_count;
        }
    }
    variant_.dsv_format = bound_.dsv.view_id != 0 ? bound_.dsv.format : DxgiFormat::UNKNOWN;
    if (samples == 0 && bound_.dsv.view_id != 0) {
        samples = bound_.dsv.sample_count;
    }
    variant_.sample_count = samples == 0 ? 1u : samples;
}

void CommandListStateTracker::set_render_targets(std::span<const RenderTargetViewInfo> rtvs,
                                                 const DepthStencilViewInfo* dsv) {
    Attachments a{};
    const size_t n = std::min<size_t>(rtvs.size(), kMaxColorTargets);
    for (size_t i = 0; i < n; ++i) {
        if (rtvs[i].view_id != 0) {
            a.rtvs[i] = rtvs[i];
            a.num_rtvs = static_cast<uint32_t>(i + 1);
        }
    }
    if (dsv != nullptr && dsv->view_id != 0) {
        a.dsv = *dsv;
    }
    if (a == bound_) {
        ++stats_.redundant_calls_filtered;
        return;
    }
    bound_ = a;
    update_variant_targets();
}

void CommandListStateTracker::OMSetRenderTargets(uint32_t num_render_target_descriptors,
                                                 const uint64_t* render_target_descriptors,
                                                 bool rts_single_handle_to_descriptor_range,
                                                 const uint64_t* depth_stencil_descriptor,
                                                 const DescriptorViewResolver& resolver) {
    std::array<RenderTargetViewInfo, kMaxColorTargets> rtvs{};
    const uint32_t n = render_target_descriptors != nullptr
                           ? std::min(num_render_target_descriptors, kMaxColorTargets)
                           : 0u;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t handle = rts_single_handle_to_descriptor_range
                                    ? render_target_descriptors[0] + i * resolver.rtv_descriptor_increment()
                                    : render_target_descriptors[i];
        if (!resolver.resolve_rtv(handle, &rtvs[i])) {
            diagnostics_ |= StateDiag::UnresolvedDescriptor;
            rtvs[i] = RenderTargetViewInfo{};
        }
    }
    DepthStencilViewInfo dsv{};
    if (depth_stencil_descriptor != nullptr && !resolver.resolve_dsv(*depth_stencil_descriptor, &dsv)) {
        diagnostics_ |= StateDiag::UnresolvedDescriptor;
        dsv = DepthStencilViewInfo{};
    }
    set_render_targets(std::span<const RenderTargetViewInfo>(rtvs.data(), n), &dsv);
}

CommandListStateTracker::PendingClear* CommandListStateTracker::find_pending(uint32_t view_id) {
    for (PendingClear& p : pending_clears_) {
        if (p.view_id == view_id) {
            return &p;
        }
    }
    return nullptr;
}

void CommandListStateTracker::record_pending_clear(const PendingClear& clear) {
    PendingClear* existing = find_pending(clear.view_id);
    if (existing == nullptr) {
        pending_clears_.push_back(clear);
        return;
    }
    if (!clear.depth_stencil) {
        existing->color = clear.color;
        existing->rtv = clear.rtv;
        return;
    }
    existing->dsv = clear.dsv;
    if ((clear.clear_flags & D3D12ClearFlags::Depth) != 0) {
        existing->depth = clear.depth;
    }
    if ((clear.clear_flags & D3D12ClearFlags::Stencil) != 0) {
        existing->stencil = clear.stencil;
    }
    existing->clear_flags |= clear.clear_flags;
}

void CommandListStateTracker::begin_pass(const Attachments& a) {
    RenderPassDesc p{};
    uint32_t width = std::numeric_limits<uint32_t>::max();
    uint32_t height = std::numeric_limits<uint32_t>::max();
    uint32_t samples = 0;
    auto account = [&](uint32_t w, uint32_t h, uint32_t s) {
        if (width != std::numeric_limits<uint32_t>::max() && (w != width || h != height)) {
            diagnostics_ |= StateDiag::AttachmentSizeMismatch;
        }
        width = std::min(width, w);
        height = std::min(height, h);
        const uint32_t ws = map_sample_count(s);
        if (samples != 0 && ws != samples) {
            diagnostics_ |= StateDiag::AttachmentSampleCountMismatch;
        }
        if (samples == 0) {
            samples = ws;
        }
    };

    for (uint32_t i = 0; i < a.num_rtvs; ++i) {
        const RenderTargetViewInfo& r = a.rtvs[i];
        if (r.view_id == 0) {
            continue;
        }
        account(r.width, r.height, r.sample_count);
        ColorAttachmentDesc& c = p.color[i];
        const TextureFormatMapping m = map_texture_format(r.format, FormatUsage::RenderTarget, options_.features);
        if (!m.supported()) {
            diagnostics_ |= StateDiag::UnsupportedAttachmentFormat;
        }
        c.view_id = r.view_id;
        c.format = m.format;
        c.load_op = WgpuLoadOp::Load;
        c.store_op = WgpuStoreOp::Store;
        if (PendingClear* pc = find_pending(r.view_id); pc != nullptr && !pc->depth_stencil) {
            c.load_op = WgpuLoadOp::Clear;
            c.clear_value = pc->color;
            ++stats_.clears_folded;
            pending_clears_.erase(pending_clears_.begin() + (pc - pending_clears_.data()));
        }
    }
    p.color_count = a.num_rtvs;

    if (a.dsv.view_id != 0) {
        const DepthStencilViewInfo& d = a.dsv;
        account(d.width, d.height, d.sample_count);
        DepthStencilAttachmentDesc& ds = p.depth_stencil;
        p.has_depth_stencil = true;
        const TextureFormatMapping m = map_texture_format(d.format, FormatUsage::DepthStencil, options_.features);
        if (!m.supported()) {
            diagnostics_ |= StateDiag::UnsupportedAttachmentFormat;
        }
        ds.view_id = d.view_id;
        ds.format = m.format;
        ds.depth_read_only = (d.flags & D3D12DsvFlags::ReadOnlyDepth) != 0;
        ds.stencil_read_only = (d.flags & D3D12DsvFlags::ReadOnlyStencil) != 0;
        const bool depth_aspect = has_depth_aspect(m.format);
        const bool stencil_aspect = has_stencil_aspect(m.format);
        if (depth_aspect && !ds.depth_read_only) {
            ds.depth_load_op = WgpuLoadOp::Load;
            ds.depth_store_op = WgpuStoreOp::Store;
        }
        if (stencil_aspect && !ds.stencil_read_only) {
            ds.stencil_load_op = WgpuLoadOp::Load;
            ds.stencil_store_op = WgpuStoreOp::Store;
        }
        if (PendingClear* pc = find_pending(d.view_id); pc != nullptr && pc->depth_stencil) {
            if ((pc->clear_flags & D3D12ClearFlags::Depth) != 0 && ds.depth_load_op != WgpuLoadOp::Undefined) {
                ds.depth_load_op = WgpuLoadOp::Clear;
                ds.depth_clear_value = pc->depth;
            }
            if ((pc->clear_flags & D3D12ClearFlags::Stencil) != 0 && ds.stencil_load_op != WgpuLoadOp::Undefined) {
                ds.stencil_load_op = WgpuLoadOp::Clear;
                ds.stencil_clear_value = pc->stencil;
            }
            ++stats_.clears_folded;
            pending_clears_.erase(pending_clears_.begin() + (pc - pending_clears_.data()));
        }
    }

    p.width = width == std::numeric_limits<uint32_t>::max() ? 0u : width;
    p.height = height == std::numeric_limits<uint32_t>::max() ? 0u : height;
    p.sample_count = samples == 0 ? 1u : samples;

    StateEvent e{};
    e.type = StateEventType::BeginRenderPass;
    e.index = static_cast<uint32_t>(passes_.size());
    passes_.push_back(p);
    events_.push_back(e);
    ++stats_.passes_begun;

    pass_open_ = true;
    pass_attachments_ = a;
    pass_width_ = p.width;
    pass_height_ = p.height;
    pass_state_known_ = true;
    pass_viewport_ = WgpuViewport{0.0f, 0.0f, static_cast<float>(p.width), static_cast<float>(p.height), 0.0f, 1.0f};
    pass_scissor_ = WgpuScissorRect{0, 0, p.width, p.height};
    pass_blend_constant_ = {0.0f, 0.0f, 0.0f, 0.0f};
    pass_stencil_ref_ = 0;
    pass_pipeline_id_ = 0;
}

void CommandListStateTracker::end_render_pass() {
    if (!pass_open_) {
        return;
    }
    StateEvent e{};
    e.type = StateEventType::EndRenderPass;
    events_.push_back(e);
    pass_open_ = false;
}

void CommandListStateTracker::emit_clear_fallback(ClearFallback fallback, const Attachments& temp_pass) {
    bool close_after = false;
    if (!(pass_open_ && pass_attachments_.contains(fallback.view_id))) {
        end_render_pass();
        if (bound_.contains(fallback.view_id)) {
            begin_pass(bound_);  // later draws reuse this pass
        } else {
            begin_pass(temp_pass);
            close_after = true;
        }
    }
    if (!fallback.depth_stencil) {
        for (uint32_t i = 0; i < pass_attachments_.num_rtvs; ++i) {
            if (pass_attachments_.rtvs[i].view_id == fallback.view_id) {
                fallback.attachment_index = i;
                break;
            }
        }
    }
    StateEvent e{};
    e.type = StateEventType::ClearFallback;
    e.index = static_cast<uint32_t>(clear_fallbacks_.size());
    clear_fallbacks_.push_back(std::move(fallback));
    events_.push_back(e);
    ++stats_.clear_fallbacks;
    pass_state_known_ = false;
    if (close_after) {
        end_render_pass();
    }
}

void CommandListStateTracker::ClearRenderTargetView(const RenderTargetViewInfo& rtv,
                                                    const float color_rgba[4],
                                                    uint32_t num_rects,
                                                    const D3D12Rect* rects) {
    if (rtv.view_id == 0 || color_rgba == nullptr) {
        return;
    }
    const WgpuScissorRect full{0, 0, rtv.width, rtv.height};
    std::vector<WgpuScissorRect> clipped;
    bool covers_all = num_rects == 0 || rects == nullptr;
    for (uint32_t i = 0; !covers_all && i < num_rects; ++i) {
        const WgpuScissorRect r = convert_scissor(rects[i], rtv.width, rtv.height);
        if (r == full) {
            covers_all = true;
        } else if (r.width != 0 && r.height != 0) {
            clipped.push_back(r);
        }
    }
    if (!covers_all && clipped.empty()) {
        ++stats_.redundant_calls_filtered;
        return;
    }

    PendingClear pc{};
    pc.view_id = rtv.view_id;
    pc.rtv = rtv;
    pc.color = {color_rgba[0], color_rgba[1], color_rgba[2], color_rgba[3]};

    ClearFallback fb{};
    fb.view_id = rtv.view_id;
    fb.color = pc.color;
    Attachments temp{};
    temp.rtvs[0] = rtv;
    temp.num_rtvs = 1;

    if (!covers_all) {
        fb.rects = std::move(clipped);
        emit_clear_fallback(std::move(fb), temp);
        return;
    }
    if (pass_open_ && pass_attachments_.contains(rtv.view_id)) {
        if (options_.mid_pass_clear == MidPassClearPolicy::DrawFallback) {
            fb.rects.push_back(full);
            emit_clear_fallback(std::move(fb), temp);
            return;
        }
        end_render_pass();
    }
    record_pending_clear(pc);
}

void CommandListStateTracker::ClearDepthStencilView(const DepthStencilViewInfo& dsv,
                                                    uint32_t clear_flags,
                                                    float depth,
                                                    uint8_t stencil,
                                                    uint32_t num_rects,
                                                    const D3D12Rect* rects) {
    if (dsv.view_id == 0) {
        return;
    }
    uint32_t flags = clear_flags & (D3D12ClearFlags::Depth | D3D12ClearFlags::Stencil);
    if ((flags & D3D12ClearFlags::Depth) != 0 && (dsv.flags & D3D12DsvFlags::ReadOnlyDepth) != 0) {
        diagnostics_ |= StateDiag::ClearOnReadOnlyAspect;
        flags &= ~D3D12ClearFlags::Depth;
    }
    if ((flags & D3D12ClearFlags::Stencil) != 0 && (dsv.flags & D3D12DsvFlags::ReadOnlyStencil) != 0) {
        diagnostics_ |= StateDiag::ClearOnReadOnlyAspect;
        flags &= ~D3D12ClearFlags::Stencil;
    }
    if (flags == 0) {
        return;
    }
    const WgpuScissorRect full{0, 0, dsv.width, dsv.height};
    std::vector<WgpuScissorRect> clipped;
    bool covers_all = num_rects == 0 || rects == nullptr;
    for (uint32_t i = 0; !covers_all && i < num_rects; ++i) {
        const WgpuScissorRect r = convert_scissor(rects[i], dsv.width, dsv.height);
        if (r == full) {
            covers_all = true;
        } else if (r.width != 0 && r.height != 0) {
            clipped.push_back(r);
        }
    }
    if (!covers_all && clipped.empty()) {
        ++stats_.redundant_calls_filtered;
        return;
    }

    PendingClear pc{};
    pc.view_id = dsv.view_id;
    pc.depth_stencil = true;
    pc.dsv = dsv;
    pc.clear_flags = flags;
    pc.depth = depth;
    pc.stencil = stencil;

    ClearFallback fb{};
    fb.view_id = dsv.view_id;
    fb.depth_stencil = true;
    fb.clear_flags = flags;
    fb.depth = depth;
    fb.stencil = stencil;
    Attachments temp{};
    temp.dsv = dsv;

    if (!covers_all) {
        fb.rects = std::move(clipped);
        emit_clear_fallback(std::move(fb), temp);
        return;
    }
    if (pass_open_ && pass_attachments_.contains(dsv.view_id)) {
        if (options_.mid_pass_clear == MidPassClearPolicy::DrawFallback) {
            fb.rects.push_back(full);
            emit_clear_fallback(std::move(fb), temp);
            return;
        }
        end_render_pass();
    }
    record_pending_clear(pc);
}

void CommandListStateTracker::RSSetViewports(uint32_t num_viewports, const D3D12Viewport* viewports) {
    if (num_viewports == 0 || viewports == nullptr) {
        viewport_set_ = false;
        return;
    }
    if (num_viewports > 1) {
        diagnostics_ |= StateDiag::MultipleViewportsIgnored;
    }
    if (viewport_set_ && same_viewport(viewport_, viewports[0])) {
        ++stats_.redundant_calls_filtered;
        return;
    }
    viewport_ = viewports[0];
    viewport_set_ = true;
}

void CommandListStateTracker::RSSetScissorRects(uint32_t num_rects, const D3D12Rect* rects) {
    if (num_rects == 0 || rects == nullptr) {
        scissor_set_ = false;
        return;
    }
    if (num_rects > 1) {
        diagnostics_ |= StateDiag::MultipleViewportsIgnored;
    }
    if (scissor_set_ && same_rect(scissor_, rects[0])) {
        ++stats_.redundant_calls_filtered;
        return;
    }
    scissor_ = rects[0];
    scissor_set_ = true;
}

void CommandListStateTracker::OMSetBlendFactor(const float blend_factor[4]) {
    const std::array<float, 4> value = blend_factor != nullptr
                                           ? std::array<float, 4>{blend_factor[0], blend_factor[1], blend_factor[2],
                                                                  blend_factor[3]}
                                           : std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    if (value == blend_factor_) {
        ++stats_.redundant_calls_filtered;
        return;
    }
    blend_factor_ = value;
}

void CommandListStateTracker::OMSetStencilRef(uint32_t stencil_ref) {
    if (stencil_ref == stencil_ref_) {
        ++stats_.redundant_calls_filtered;
        return;
    }
    stencil_ref_ = stencil_ref;
}

void CommandListStateTracker::IASetPrimitiveTopology(D3DPrimitiveTopology topology) {
    if (topology == variant_.topology) {
        ++stats_.redundant_calls_filtered;
        return;
    }
    variant_.topology = topology;
}

void CommandListStateTracker::set_index_format(WgpuIndexFormat format) {
    if (format == variant_.index_format) {
        ++stats_.redundant_calls_filtered;
        return;
    }
    variant_.index_format = format;
}

void CommandListStateTracker::set_vertex_strides(uint32_t start_slot, uint32_t num_views, const uint32_t* strides) {
    for (uint32_t i = 0; i < num_views; ++i) {
        const uint32_t slot = start_slot + i;
        if (slot >= kMaxVertexBuffers) {
            break;
        }
        if (strides == nullptr) {
            variant_.bound_vertex_slots &= ~(1u << slot);
            variant_.vertex_strides[slot] = 0;
        } else {
            variant_.bound_vertex_slots |= 1u << slot;
            variant_.vertex_strides[slot] = strides[i];
        }
    }
}

void CommandListStateTracker::SetPipelineState(GraphicsPso* pso) {
    if (pso == pso_) {
        ++stats_.redundant_calls_filtered;
        return;
    }
    pso_ = pso;
}

WgpuViewport CommandListStateTracker::convert_viewport(const D3D12Viewport& vp,
                                                       uint32_t max_texture_dimension_2d,
                                                       uint64_t* diagnostics) {
    uint64_t diag = 0;
    const float max_dim = static_cast<float>(max_texture_dimension_2d);
    const float range = 2.0f * max_dim;  // WebGPU: maxViewportRange = 2 * maxTextureDimension2D
    WgpuViewport out{};
    out.x = finite_or_zero(vp.TopLeftX);
    out.y = finite_or_zero(vp.TopLeftY);
    out.width = finite_or_zero(vp.Width);
    out.height = finite_or_zero(vp.Height);
    auto clamp_extent = [&](float& origin, float& extent) {
        const float e = std::clamp(extent, 0.0f, max_dim);
        float o = std::max(origin, -range);
        o = std::min(o, range - 1.0f - e);
        if (e != extent || o != origin) {
            diag |= StateDiag::ViewportClamped;
        }
        origin = o;
        extent = e;
    };
    clamp_extent(out.x, out.width);
    clamp_extent(out.y, out.height);
    const float min_d = finite_or_zero(vp.MinDepth);
    const float max_d = std::isfinite(vp.MaxDepth) ? vp.MaxDepth : 1.0f;
    out.min_depth = std::clamp(min_d, 0.0f, 1.0f);
    out.max_depth = std::clamp(max_d, 0.0f, 1.0f);
    if (out.min_depth != vp.MinDepth || out.max_depth != vp.MaxDepth) {
        diag |= StateDiag::ViewportClamped;
    }
    if (out.min_depth > out.max_depth) {
        std::swap(out.min_depth, out.max_depth);
        diag |= StateDiag::ViewportDepthInverted;
    }
    if (diagnostics != nullptr) {
        *diagnostics |= diag;
    }
    return out;
}

WgpuScissorRect CommandListStateTracker::convert_scissor(const D3D12Rect& rect, uint32_t width, uint32_t height) {
    const int64_t w = width;
    const int64_t h = height;
    const int64_t x0 = std::clamp<int64_t>(rect.left, 0, w);
    const int64_t y0 = std::clamp<int64_t>(rect.top, 0, h);
    const int64_t x1 = std::clamp<int64_t>(rect.right, x0, w);
    const int64_t y1 = std::clamp<int64_t>(rect.bottom, y0, h);
    return WgpuScissorRect{static_cast<uint32_t>(x0), static_cast<uint32_t>(y0), static_cast<uint32_t>(x1 - x0),
                           static_cast<uint32_t>(y1 - y0)};
}

bool CommandListStateTracker::prepare_draw(PipelineCache* cache) {
    if (bound_.empty()) {
        diagnostics_ |= StateDiag::DrawWithoutAttachments;
        ++stats_.draws_skipped;
        return false;
    }
    PipelineResolution resolution{};
    if (cache != nullptr) {
        if (pso_ == nullptr) {
            diagnostics_ |= StateDiag::DrawWithoutPipeline;
            ++stats_.draws_skipped;
            return false;
        }
        resolution = pso_->resolve(*cache, variant_);
        pipeline_diagnostics_ |= resolution.diagnostics;
        if (!resolution.usable()) {
            diagnostics_ |= StateDiag::PipelineUnusable;
            ++stats_.draws_skipped;
            return false;
        }
    }

    if (!pass_open_ || !(pass_attachments_ == bound_)) {
        end_render_pass();
        begin_pass(bound_);
    }

    if (cache != nullptr && (!pass_state_known_ || resolution.pipeline_id != pass_pipeline_id_)) {
        StateEvent e{};
        e.type = StateEventType::SetPipeline;
        e.pipeline_id = resolution.pipeline_id;
        e.pipeline_key = resolution.created ? resolution.key : nullptr;
        events_.push_back(e);
        pass_pipeline_id_ = resolution.pipeline_id;
    }

    WgpuViewport vp{0.0f, 0.0f, static_cast<float>(pass_width_), static_cast<float>(pass_height_), 0.0f, 1.0f};
    if (viewport_set_) {
        vp = convert_viewport(viewport_, options_.max_texture_dimension_2d, &diagnostics_);
    } else {
        diagnostics_ |= StateDiag::ViewportNotSet;
    }
    if (!pass_state_known_ || !(vp == pass_viewport_)) {
        StateEvent e{};
        e.type = StateEventType::SetViewport;
        e.viewport = vp;
        events_.push_back(e);
        pass_viewport_ = vp;
    }

    WgpuScissorRect sc{0, 0, pass_width_, pass_height_};
    if (scissor_set_) {
        sc = convert_scissor(scissor_, pass_width_, pass_height_);
    } else {
        diagnostics_ |= StateDiag::ScissorNotSet;
    }
    if (!pass_state_known_ || !(sc == pass_scissor_)) {
        StateEvent e{};
        e.type = StateEventType::SetScissorRect;
        e.scissor = sc;
        events_.push_back(e);
        pass_scissor_ = sc;
    }

    if (!pass_state_known_ || blend_factor_ != pass_blend_constant_) {
        StateEvent e{};
        e.type = StateEventType::SetBlendConstant;
        e.blend_constant = blend_factor_;
        events_.push_back(e);
        pass_blend_constant_ = blend_factor_;
    }
    if (!pass_state_known_ || stencil_ref_ != pass_stencil_ref_) {
        StateEvent e{};
        e.type = StateEventType::SetStencilReference;
        e.stencil_reference = stencil_ref_;
        events_.push_back(e);
        pass_stencil_ref_ = stencil_ref_;
    }

    pass_state_known_ = true;
    ++stats_.draws;
    return true;
}

void CommandListStateTracker::flush_pending_clears() {
    if (pending_clears_.empty()) {
        return;
    }
    end_render_pass();
    while (!pending_clears_.empty()) {
        const PendingClear pc = pending_clears_.front();
        Attachments temp{};
        if (pc.depth_stencil) {
            temp.dsv = pc.dsv;
        } else {
            temp.rtvs[0] = pc.rtv;
            temp.num_rtvs = 1;
        }
        begin_pass(temp);  // consumes the pending clear as loadOp "clear"
        end_render_pass();
        ++stats_.clear_only_passes;
        if (!pending_clears_.empty() && pending_clears_.front().view_id == pc.view_id) {
            pending_clears_.erase(pending_clears_.begin());  // defensive: never loop forever
        }
    }
}

void CommandListStateTracker::close() {
    flush_pending_clears();
    end_render_pass();
}

}  // namespace playgta6::gpu::pso
