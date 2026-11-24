#include "gpu/gfx10_descriptors.h"

#include <algorithm>
#include <cmath>

namespace playgta6::gpu::gfx10 {

namespace {

constexpr uint32_t bits(uint32_t word, uint32_t shift, uint32_t width) noexcept {
    return (word >> shift) & ((width >= 32 ? 0u : (1u << width)) - 1u);
}

constexpr uint32_t field(uint32_t value, uint32_t shift, uint32_t width) noexcept {
    return (value & ((width >= 32 ? 0u : (1u << width)) - 1u)) << shift;
}

constexpr uint32_t sel_field(const Swizzle& s) noexcept {
    return field(s[0], 0, 3) | field(s[1], 3, 3) | field(s[2], 6, 3) | field(s[3], 9, 3);
}

Swizzle sel_from(uint32_t word3) noexcept {
    return Swizzle{static_cast<uint8_t>(bits(word3, 0, 3)), static_cast<uint8_t>(bits(word3, 3, 3)),
                   static_cast<uint8_t>(bits(word3, 6, 3)), static_cast<uint8_t>(bits(word3, 9, 3))};
}

bool valid_sel(uint32_t s) noexcept {
    return s == Sel::Zero || s == Sel::One || (s >= Sel::X && s <= Sel::W);
}

bool valid_sels(uint32_t word3) noexcept {
    return valid_sel(bits(word3, 0, 3)) && valid_sel(bits(word3, 3, 3)) && valid_sel(bits(word3, 6, 3)) &&
           valid_sel(bits(word3, 9, 3));
}

// IMG_FMT values that are reserved on gfx10 (gfx9_plus_merged_enum.h): 78..127, 155, 285..511
// (285..287 are VRS-only formats).
bool valid_image_format(uint32_t f) noexcept {
    if (f == 0 || f >= 285) {
        return false;
    }
    if (f >= 78 && f <= 127) {
        return false;
    }
    return f != 155;
}

bool valid_buffer_format(uint32_t f) noexcept {
    return f != 0 && f < 78;
}

// ---- DXGI <-> hardware format table ----
struct FormatEntry {
    uint32_t dxgi;
    uint32_t hw;
    Swizzle sel;
};

constexpr Swizzle kXYZW = {Sel::X, Sel::Y, Sel::Z, Sel::W};
constexpr Swizzle kXYZ1 = {Sel::X, Sel::Y, Sel::Z, Sel::One};
constexpr Swizzle kXY01 = {Sel::X, Sel::Y, Sel::Zero, Sel::One};
constexpr Swizzle kX001 = {Sel::X, Sel::Zero, Sel::Zero, Sel::One};
constexpr Swizzle k000X = {Sel::Zero, Sel::Zero, Sel::Zero, Sel::X};
constexpr Swizzle kZYXW = {Sel::Z, Sel::Y, Sel::X, Sel::W};
constexpr Swizzle kZYX1 = {Sel::Z, Sel::Y, Sel::X, Sel::One};

// Hardware format names list components from the most significant bits down (2_10_10_10 holds
// R in bits 0..9 and A in bits 30..31, i.e. DXGI R10G10B10A2); one-component reads use X.
// The first entry of a hardware format is the one dxgi_format() picks when selects do not match.
constexpr FormatEntry kFormats[] = {
    {2, 77, kXYZW},    // R32G32B32A32_FLOAT -> 32_32_32_32_FLOAT
    {3, 75, kXYZW},    // R32G32B32A32_UINT
    {4, 76, kXYZW},    // R32G32B32A32_SINT
    {1, 75, kXYZW},    // R32G32B32A32_TYPELESS (as UINT)
    {6, 74, kXYZ1},    // R32G32B32_FLOAT
    {7, 72, kXYZ1},    // R32G32B32_UINT
    {8, 73, kXYZ1},    // R32G32B32_SINT
    {10, 71, kXYZW},   // R16G16B16A16_FLOAT
    {11, 65, kXYZW},   // R16G16B16A16_UNORM
    {12, 69, kXYZW},   // R16G16B16A16_UINT
    {13, 66, kXYZW},   // R16G16B16A16_SNORM
    {14, 70, kXYZW},   // R16G16B16A16_SINT
    {9, 69, kXYZW},    // R16G16B16A16_TYPELESS (as UINT)
    {16, 64, kXY01},   // R32G32_FLOAT
    {17, 62, kXY01},   // R32G32_UINT
    {18, 63, kXY01},   // R32G32_SINT
    {15, 62, kXY01},   // R32G32_TYPELESS (as UINT)
    {24, 50, kXYZW},   // R10G10B10A2_UNORM -> 2_10_10_10_UNORM
    {25, 54, kXYZW},   // R10G10B10A2_UINT
    {23, 54, kXYZW},   // R10G10B10A2_TYPELESS (as UINT)
    {26, 36, kXYZ1},   // R11G11B10_FLOAT -> 10_11_11_FLOAT
    {28, 56, kXYZW},   // R8G8B8A8_UNORM -> 8_8_8_8_UNORM
    {29, 130, kXYZW},  // R8G8B8A8_UNORM_SRGB -> 8_8_8_8_SRGB
    {30, 60, kXYZW},   // R8G8B8A8_UINT
    {31, 57, kXYZW},   // R8G8B8A8_SNORM
    {32, 61, kXYZW},   // R8G8B8A8_SINT
    {27, 60, kXYZW},   // R8G8B8A8_TYPELESS (as UINT)
    {87, 56, kZYXW},   // B8G8R8A8_UNORM -> 8_8_8_8_UNORM read Z, Y, X, W
    {88, 56, kZYX1},   // B8G8R8X8_UNORM
    {91, 130, kZYXW},  // B8G8R8A8_UNORM_SRGB
    {93, 130, kZYX1},  // B8G8R8X8_UNORM_SRGB
    {34, 29, kXY01},   // R16G16_FLOAT
    {35, 23, kXY01},   // R16G16_UNORM
    {36, 27, kXY01},   // R16G16_UINT
    {37, 24, kXY01},   // R16G16_SNORM
    {38, 28, kXY01},   // R16G16_SINT
    {33, 27, kXY01},   // R16G16_TYPELESS (as UINT)
    {41, 22, kX001},   // R32_FLOAT
    {40, 22, kX001},   // D32_FLOAT
    {42, 20, kX001},   // R32_UINT
    {43, 21, kX001},   // R32_SINT
    {39, 20, kX001},   // R32_TYPELESS (as UINT)
    {21, 22, kX001},   // R32_FLOAT_X8X24_TYPELESS (depth plane)
    {20, 22, kX001},   // D32_FLOAT_S8X24_UINT (depth plane)
    {45, 141, kX001},  // D24_UNORM_S8_UINT -> 8_24_UNORM (depth)
    {46, 141, kX001},  // R24_UNORM_X8_TYPELESS
    {49, 14, kXY01},   // R8G8_UNORM
    {50, 18, kXY01},   // R8G8_UINT
    {51, 15, kXY01},   // R8G8_SNORM
    {52, 19, kXY01},   // R8G8_SINT
    {48, 18, kXY01},   // R8G8_TYPELESS (as UINT)
    {54, 13, kX001},   // R16_FLOAT
    {56, 7, kX001},    // R16_UNORM
    {55, 7, kX001},    // D16_UNORM
    {57, 11, kX001},   // R16_UINT
    {58, 8, kX001},    // R16_SNORM
    {59, 12, kX001},   // R16_SINT
    {53, 11, kX001},   // R16_TYPELESS (as UINT)
    {61, 1, kX001},    // R8_UNORM
    {62, 5, kX001},    // R8_UINT
    {63, 2, kX001},    // R8_SNORM
    {64, 6, kX001},    // R8_SINT
    {60, 5, kX001},    // R8_TYPELESS (as UINT)
    {65, 1, k000X},    // A8_UNORM -> 8_UNORM read 0, 0, 0, X
    {67, 132, kXYZ1},  // R9G9B9E5_SHAREDEXP -> 5_9_9_9_FLOAT
    {85, 133, kZYX1},  // B5G6R5_UNORM -> 5_6_5_UNORM (X = bits 0..4 = B)
    {86, 134, kZYXW},  // B5G5R5A1_UNORM -> 1_5_5_5_UNORM
    {71, 169, kXYZW},  // BC1_UNORM
    {72, 170, kXYZW},  // BC1_UNORM_SRGB
    {70, 169, kXYZW},  // BC1_TYPELESS
    {74, 171, kXYZW},  // BC2_UNORM
    {75, 172, kXYZW},  // BC2_UNORM_SRGB
    {73, 171, kXYZW},  // BC2_TYPELESS
    {77, 173, kXYZW},  // BC3_UNORM
    {78, 174, kXYZW},  // BC3_UNORM_SRGB
    {76, 173, kXYZW},  // BC3_TYPELESS
    {80, 175, kX001},  // BC4_UNORM
    {81, 176, kX001},  // BC4_SNORM
    {79, 175, kX001},  // BC4_TYPELESS
    {83, 177, kXY01},  // BC5_UNORM
    {84, 178, kXY01},  // BC5_SNORM
    {82, 177, kXY01},  // BC5_TYPELESS
    {95, 179, kXYZ1},  // BC6H_UF16
    {96, 180, kXYZ1},  // BC6H_SF16
    {94, 179, kXYZ1},  // BC6H_TYPELESS
    {98, 181, kXYZW},  // BC7_UNORM
    {99, 182, kXYZW},  // BC7_UNORM_SRGB
    {97, 181, kXYZW},  // BC7_TYPELESS
};

int32_t to_fixed(float v, float lo, float hi, float scale) noexcept {
    if (std::isnan(v)) {
        v = lo;
    }
    v = std::clamp(v, lo, hi);
    return static_cast<int32_t>(std::lround(v * scale));
}

}  // namespace

// ---- V# ----

void encode(const BufferResource& d, uint32_t out[kBufferDwords]) noexcept {
    out[0] = static_cast<uint32_t>(d.base);
    out[1] = field(static_cast<uint32_t>(d.base >> 32), 0, 16) | field(d.stride, 16, 14) |
             field(d.cache_swizzle ? 1u : 0u, 30, 1) | field(d.swizzle_enable ? 1u : 0u, 31, 1);
    out[2] = d.num_records;
    out[3] = sel_field(d.dst_sel) | field(d.format, 12, 7) | field(d.index_stride, 21, 2) |
             field(d.add_tid_enable ? 1u : 0u, 23, 1) | field(d.resource_level ? 1u : 0u, 24, 1) |
             field(d.oob_select, 28, 2) | field(d.type, 30, 2);
}

BufferResource decode_buffer(const uint32_t in[kBufferDwords]) noexcept {
    BufferResource d;
    d.base = uint64_t{in[0]} | (uint64_t{bits(in[1], 0, 16)} << 32);
    d.stride = bits(in[1], 16, 14);
    d.cache_swizzle = bits(in[1], 30, 1) != 0;
    d.swizzle_enable = bits(in[1], 31, 1) != 0;
    d.num_records = in[2];
    d.dst_sel = sel_from(in[3]);
    d.format = bits(in[3], 12, 7);
    d.index_stride = bits(in[3], 21, 2);
    d.add_tid_enable = bits(in[3], 23, 1) != 0;
    d.resource_level = bits(in[3], 24, 1) != 0;
    d.oob_select = bits(in[3], 28, 2);
    d.type = bits(in[3], 30, 2);
    return d;
}

bool plausible_buffer(const uint32_t w[kBufferDwords]) noexcept {
    // Type 0, valid selects and format, unused bits 19..20 and 25..27 zero. resource_level is not
    // checked: the open-source drivers set it on gfx10, the console's own constant-buffer words leave
    // it 0 (they are read with scalar loads, which ignore format, selects and that bit).
    return bits(w[3], 30, 2) == 0 && valid_sels(w[3]) && valid_buffer_format(bits(w[3], 12, 7)) &&
           bits(w[3], 19, 2) == 0 && bits(w[3], 25, 3) == 0;
}

// ---- T# ----

void encode(const ImageResource& d, uint32_t out[kImageDwords]) noexcept {
    const uint64_t base = d.base >> 8;
    const uint32_t w = d.width == 0 ? 0u : d.width - 1u;
    const uint32_t h = d.height == 0 ? 0u : d.height - 1u;
    const uint32_t depth = d.depth == 0 ? 0u : d.depth - 1u;
    const uint64_t meta = d.meta_data_address >> 8;
    out[0] = static_cast<uint32_t>(base);
    out[1] = field(static_cast<uint32_t>(base >> 32), 0, 8) | field(d.min_lod, 8, 12) | field(d.format, 20, 9) |
             field(w, 30, 2);
    out[2] = field(w >> 2, 0, 14) | field(h, 14, 16) | field(d.resource_level ? 1u : 0u, 31, 1);
    out[3] = sel_field(d.dst_sel) | field(d.base_level, 12, 4) | field(d.last_level, 16, 4) | field(d.sw_mode, 20, 5) |
             field(d.bc_swizzle, 25, 3) | field(d.type, 28, 4);
    out[4] = field(depth, 0, 16) | field(d.base_array, 16, 16);
    out[5] = field(d.array_pitch, 0, 4) | field(d.max_mip, 4, 4) | field(d.min_lod_warn, 8, 12) |
             field(d.perf_mod, 20, 3) | field(d.corner_samples ? 1u : 0u, 23, 1) |
             field(d.lod_hdw_cnt_en ? 1u : 0u, 25, 1) | field(d.prt_default ? 1u : 0u, 26, 1) |
             field(d.big_page ? 1u : 0u, 31, 1);
    out[6] = field(d.counter_bank_id, 0, 8) | field(d.iterate_256 ? 1u : 0u, 10, 1) |
             field(d.max_uncompressed_block_size, 15, 2) | field(d.max_compressed_block_size, 17, 2) |
             field(d.meta_pipe_aligned ? 1u : 0u, 19, 1) | field(d.write_compress_enable ? 1u : 0u, 20, 1) |
             field(d.compression_en ? 1u : 0u, 21, 1) | field(d.alpha_is_on_msb ? 1u : 0u, 22, 1) |
             field(d.color_transform ? 1u : 0u, 23, 1) | field(static_cast<uint32_t>(meta), 24, 8);
    out[7] = static_cast<uint32_t>(meta >> 8);
}

ImageResource decode_image(const uint32_t in[kImageDwords]) noexcept {
    ImageResource d;
    d.base = (uint64_t{in[0]} | (uint64_t{bits(in[1], 0, 8)} << 32)) << 8;
    d.min_lod = bits(in[1], 8, 12);
    d.format = bits(in[1], 20, 9);
    d.width = (bits(in[1], 30, 2) | (bits(in[2], 0, 14) << 2)) + 1u;
    d.height = bits(in[2], 14, 16) + 1u;
    d.resource_level = bits(in[2], 31, 1) != 0;
    d.dst_sel = sel_from(in[3]);
    d.base_level = bits(in[3], 12, 4);
    d.last_level = bits(in[3], 16, 4);
    d.sw_mode = bits(in[3], 20, 5);
    d.bc_swizzle = bits(in[3], 25, 3);
    d.type = bits(in[3], 28, 4);
    d.depth = bits(in[4], 0, 16) + 1u;
    d.base_array = bits(in[4], 16, 16);
    d.array_pitch = bits(in[5], 0, 4);
    d.max_mip = bits(in[5], 4, 4);
    d.min_lod_warn = bits(in[5], 8, 12);
    d.perf_mod = bits(in[5], 20, 3);
    d.corner_samples = bits(in[5], 23, 1) != 0;
    d.lod_hdw_cnt_en = bits(in[5], 25, 1) != 0;
    d.prt_default = bits(in[5], 26, 1) != 0;
    d.big_page = bits(in[5], 31, 1) != 0;
    d.counter_bank_id = bits(in[6], 0, 8);
    d.iterate_256 = bits(in[6], 10, 1) != 0;
    d.max_uncompressed_block_size = bits(in[6], 15, 2);
    d.max_compressed_block_size = bits(in[6], 17, 2);
    d.meta_pipe_aligned = bits(in[6], 19, 1) != 0;
    d.write_compress_enable = bits(in[6], 20, 1) != 0;
    d.compression_en = bits(in[6], 21, 1) != 0;
    d.alpha_is_on_msb = bits(in[6], 22, 1) != 0;
    d.color_transform = bits(in[6], 23, 1) != 0;
    d.meta_data_address = ((uint64_t{in[7]} << 8) | bits(in[6], 24, 8)) << 8;
    return d;
}

bool plausible_image(const uint32_t w[kImageDwords]) noexcept {
    const uint32_t type = bits(w[3], 28, 4);
    const uint32_t base_level = bits(w[3], 12, 4);
    const uint32_t last_level = bits(w[3], 16, 4);
    return type >= ImageType::Tex1D && bits(w[2], 31, 1) == 1 && valid_sels(w[3]) &&
           valid_image_format(bits(w[1], 20, 9)) && base_level <= last_level && bits(w[1], 29, 1) == 0 &&
           bits(w[2], 30, 1) == 0;
}

SlotContent classify_slot(const uint32_t w[8]) noexcept {
    bool zero = true;
    for (int i = 0; i < 8; ++i) {
        zero = zero && w[i] == 0;
    }
    if (zero) {
        return SlotContent::Zero;
    }
    if ((w[3] >> 31) != 0) {
        return plausible_image(w) ? SlotContent::Image : SlotContent::Unknown;
    }
    return plausible_buffer(w) ? SlotContent::Buffer : SlotContent::Unknown;
}

// ---- S# ----

void encode(const SamplerState& d, uint32_t out[kSamplerDwords]) noexcept {
    out[0] = field(d.clamp_x, 0, 3) | field(d.clamp_y, 3, 3) | field(d.clamp_z, 6, 3) |
             field(d.max_aniso_ratio, 9, 3) | field(d.depth_compare_func, 12, 3) |
             field(d.force_unnormalized ? 1u : 0u, 15, 1) | field(d.aniso_threshold, 16, 3) |
             field(d.mc_coord_trunc ? 1u : 0u, 19, 1) | field(d.force_degamma ? 1u : 0u, 20, 1) |
             field(d.aniso_bias, 21, 6) | field(d.trunc_coord ? 1u : 0u, 27, 1) |
             field(d.disable_cube_wrap ? 1u : 0u, 28, 1) | field(d.filter_mode, 29, 2) |
             field(d.skip_degamma ? 1u : 0u, 31, 1);
    out[1] = field(d.min_lod, 0, 12) | field(d.max_lod, 12, 12) | field(d.perf_mip, 24, 4) | field(d.perf_z, 28, 4);
    out[2] = field(d.lod_bias, 0, 14) | field(d.lod_bias_sec, 14, 6) | field(d.xy_mag_filter, 20, 2) |
             field(d.xy_min_filter, 22, 2) | field(d.z_filter, 24, 2) | field(d.mip_filter, 26, 2) |
             field(d.mip_point_preclamp ? 1u : 0u, 28, 1) | field(d.aniso_override ? 1u : 0u, 29, 1) |
             field(d.blend_prt ? 1u : 0u, 30, 1) | field(d.deriv_adjust_en ? 1u : 0u, 31, 1);
    out[3] = field(d.border_color_ptr, 0, 12) | field(d.border_color_type, 30, 2);
}

SamplerState decode_sampler(const uint32_t in[kSamplerDwords]) noexcept {
    SamplerState d;
    d.clamp_x = bits(in[0], 0, 3);
    d.clamp_y = bits(in[0], 3, 3);
    d.clamp_z = bits(in[0], 6, 3);
    d.max_aniso_ratio = bits(in[0], 9, 3);
    d.depth_compare_func = bits(in[0], 12, 3);
    d.force_unnormalized = bits(in[0], 15, 1) != 0;
    d.aniso_threshold = bits(in[0], 16, 3);
    d.mc_coord_trunc = bits(in[0], 19, 1) != 0;
    d.force_degamma = bits(in[0], 20, 1) != 0;
    d.aniso_bias = bits(in[0], 21, 6);
    d.trunc_coord = bits(in[0], 27, 1) != 0;
    d.disable_cube_wrap = bits(in[0], 28, 1) != 0;
    d.filter_mode = bits(in[0], 29, 2);
    d.skip_degamma = bits(in[0], 31, 1) != 0;
    d.min_lod = bits(in[1], 0, 12);
    d.max_lod = bits(in[1], 12, 12);
    d.perf_mip = bits(in[1], 24, 4);
    d.perf_z = bits(in[1], 28, 4);
    d.lod_bias = bits(in[2], 0, 14);
    d.lod_bias_sec = bits(in[2], 14, 6);
    d.xy_mag_filter = bits(in[2], 20, 2);
    d.xy_min_filter = bits(in[2], 22, 2);
    d.z_filter = bits(in[2], 24, 2);
    d.mip_filter = bits(in[2], 26, 2);
    d.mip_point_preclamp = bits(in[2], 28, 1) != 0;
    d.aniso_override = bits(in[2], 29, 1) != 0;
    d.blend_prt = bits(in[2], 30, 1) != 0;
    d.deriv_adjust_en = bits(in[2], 31, 1) != 0;
    d.border_color_ptr = bits(in[3], 0, 12);
    d.border_color_type = bits(in[3], 30, 2);
    return d;
}

// ---- Formats ----

bool hardware_format(uint32_t dxgi, HardwareFormat* out) noexcept {
    for (const FormatEntry& e : kFormats) {
        if (e.dxgi == dxgi) {
            if (out != nullptr) {
                out->format = e.hw;
                out->swizzle = e.sel;
            }
            return true;
        }
    }
    return false;
}

uint32_t dxgi_format(uint32_t hw, const Swizzle& dst_sel, bool* exact) noexcept {
    const FormatEntry* first = nullptr;
    for (const FormatEntry& e : kFormats) {
        if (e.hw != hw) {
            continue;
        }
        if (e.sel == dst_sel) {
            if (exact != nullptr) {
                *exact = true;
            }
            return e.dxgi;
        }
        if (first == nullptr) {
            first = &e;
        }
    }
    if (exact != nullptr) {
        *exact = false;
    }
    return first != nullptr ? first->dxgi : 0u;
}

uint32_t buffer_format_bytes(uint32_t f) noexcept {
    switch (f) {
    case 1: case 2: case 3: case 4: case 5: case 6:
        return 1;
    case 7: case 8: case 9: case 10: case 11: case 12: case 13: case 14: case 15: case 16: case 17: case 18: case 19:
        return 2;
    case 20: case 21: case 22: case 23: case 24: case 25: case 26: case 27: case 28: case 29:
        return 4;
    default:
        break;
    }
    if (f >= 30 && f <= 61) {
        return 4;  // 10_11_11, 11_11_10, 10_10_10_2, 2_10_10_10, 8_8_8_8
    }
    if (f >= 62 && f <= 71) {
        return 8;  // 32_32, 16_16_16_16
    }
    if (f >= 72 && f <= 74) {
        return 12;
    }
    if (f >= 75 && f <= 77) {
        return 16;
    }
    return 0;
}

Swizzle compose_component_mapping(const Swizzle& format_swizzle, uint32_t mapping) noexcept {
    Swizzle out{};
    for (uint32_t c = 0; c < 4; ++c) {
        const uint32_t src = (mapping >> (3 * c)) & 7u;
        if (src <= 3) {
            out[c] = format_swizzle[src];
        } else if (src == 4) {
            out[c] = Sel::Zero;
        } else {
            out[c] = Sel::One;
        }
    }
    return out;
}

bool component_mapping_from(const Swizzle& format_swizzle, const Swizzle& dst_sel, uint32_t* mapping) noexcept {
    uint32_t m = 1u << 12;  // D3D12_SHADER_COMPONENT_MAPPING_ALWAYS_SET_BIT_AVOIDING_ZEROMEM_MISTAKES
    for (uint32_t c = 0; c < 4; ++c) {
        uint32_t src = 0xffu;
        if (dst_sel[c] == Sel::Zero) {
            src = 4;
        } else if (dst_sel[c] == Sel::One) {
            src = 5;
        } else {
            for (uint32_t k = 0; k < 4; ++k) {
                if (format_swizzle[k] == dst_sel[c]) {
                    src = k;
                    break;
                }
            }
        }
        if (src == 0xffu) {
            return false;
        }
        m |= src << (3 * c);
    }
    if (mapping != nullptr) {
        *mapping = m;
    }
    return true;
}

// ---- Samplers ----

uint32_t lod_to_u4_8(float lod) noexcept {
    return static_cast<uint32_t>(to_fixed(lod, 0.0f, 4095.0f / 256.0f, 256.0f));
}

float u4_8_to_lod(uint32_t v) noexcept {
    return static_cast<float>(v & 0xfffu) / 256.0f;
}

uint32_t bias_to_s6_8(float bias) noexcept {
    return static_cast<uint32_t>(to_fixed(bias, -32.0f, 8191.0f / 256.0f, 256.0f)) & 0x3fffu;
}

float s6_8_to_bias(uint32_t v) noexcept {
    int32_t s = static_cast<int32_t>(v & 0x3fffu);
    if ((s & 0x2000) != 0) {
        s -= 0x4000;
    }
    return static_cast<float>(s) / 256.0f;
}

namespace {

// D3D12_TEXTURE_ADDRESS_MODE (1 wrap, 2 mirror, 3 clamp, 4 border, 5 mirror once) <-> SQ_TEX_CLAMP.
uint32_t clamp_from_d3d12(uint32_t mode) noexcept {
    switch (mode) {
    case 2: return 1;  // MIRROR
    case 3: return 2;  // CLAMP_LAST_TEXEL
    case 4: return 6;  // CLAMP_BORDER
    case 5: return 3;  // MIRROR_ONCE_LAST_TEXEL
    default: return 0; // WRAP
    }
}

uint32_t d3d12_from_clamp(uint32_t clamp) noexcept {
    switch (clamp) {
    case 1: return 2;
    case 2: return 3;
    case 3: return 5;
    case 4: case 6: return 4;
    case 5: case 7: return 5;
    default: return 1;
    }
}

uint32_t aniso_ratio(uint32_t max_anisotropy) noexcept {
    if (max_anisotropy >= 16) return 4;
    if (max_anisotropy >= 8) return 3;
    if (max_anisotropy >= 4) return 2;
    if (max_anisotropy >= 2) return 1;
    return 0;
}

}  // namespace

SamplerState sampler_from_d3d12(const d3d12_state::SamplerDesc& d, bool* exact) noexcept {
    SamplerState s;
    bool ok = true;
    // D3D12_FILTER: bits 0..1 mip, 2..3 mag, 4..5 min (0 point, 1 linear), bit 6 anisotropic,
    // bits 7..8 reduction (0 standard, 1 comparison, 2 minimum, 3 maximum).
    const uint32_t f = d.filter;
    const bool aniso = (f & 0x40u) != 0;
    const uint32_t reduction = (f >> 7) & 3u;
    const uint32_t mip = f & 3u;
    const uint32_t mag = (f >> 2) & 3u;
    const uint32_t min = (f >> 4) & 3u;
    s.xy_mag_filter = aniso ? 3u : (mag != 0 ? 1u : 0u);
    s.xy_min_filter = aniso ? 3u : (min != 0 ? 1u : 0u);
    s.mip_filter = mip != 0 ? 2u : 1u;
    s.z_filter = mag != 0 ? 2u : 1u;
    s.max_aniso_ratio = aniso ? aniso_ratio(d.max_anisotropy) : 0u;
    s.filter_mode = reduction == 2 ? 1u : (reduction == 3 ? 2u : 0u);
    s.depth_compare_func = reduction == 1 && d.comparison_func >= 1 && d.comparison_func <= 8 ? d.comparison_func - 1u : 0u;
    s.clamp_x = clamp_from_d3d12(d.address_u);
    s.clamp_y = clamp_from_d3d12(d.address_v);
    s.clamp_z = clamp_from_d3d12(d.address_w);
    s.min_lod = lod_to_u4_8(d.min_lod);
    s.max_lod = lod_to_u4_8(d.max_lod);
    s.lod_bias = bias_to_s6_8(d.mip_lod_bias);
    const float* c = d.border_color;
    if (c[0] == 0.0f && c[1] == 0.0f && c[2] == 0.0f) {
        s.border_color_type = c[3] == 0.0f ? 0u : 1u;
        ok = ok && (c[3] == 0.0f || c[3] == 1.0f);
    } else if (c[0] == 1.0f && c[1] == 1.0f && c[2] == 1.0f && c[3] == 1.0f) {
        s.border_color_type = 2;
    } else {
        s.border_color_type = 0;
        ok = false;
    }
    if (exact != nullptr) {
        *exact = ok;
    }
    return s;
}

d3d12_state::SamplerDesc d3d12_from_sampler(const SamplerState& s) noexcept {
    d3d12_state::SamplerDesc d;
    const bool aniso = s.xy_mag_filter >= 2 || s.xy_min_filter >= 2;
    uint32_t f = 0;
    if (aniso) {
        f = 0x55u;
    } else {
        f = (s.mip_filter == 2 ? 1u : 0u) | ((s.xy_mag_filter & 1u) << 2) | ((s.xy_min_filter & 1u) << 4);
    }
    if (s.filter_mode == 1) {
        f |= 2u << 7;
    } else if (s.filter_mode == 2) {
        f |= 3u << 7;
    } else if (s.depth_compare_func != 0) {
        f |= 1u << 7;
    }
    d.filter = f;
    d.comparison_func = s.depth_compare_func != 0 ? s.depth_compare_func + 1u : 0u;
    d.address_u = d3d12_from_clamp(s.clamp_x);
    d.address_v = d3d12_from_clamp(s.clamp_y);
    d.address_w = d3d12_from_clamp(s.clamp_z);
    d.max_anisotropy = aniso ? (1u << std::min<uint32_t>(s.max_aniso_ratio, 4u)) : 1u;
    d.min_lod = u4_8_to_lod(s.min_lod);
    d.max_lod = u4_8_to_lod(s.max_lod);
    d.mip_lod_bias = s6_8_to_bias(s.lod_bias);
    const float colors[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}};
    const uint32_t t = s.border_color_type < 3 ? s.border_color_type : 0u;
    for (int i = 0; i < 4; ++i) {
        d.border_color[i] = colors[t][i];
    }
    return d;
}

}  // namespace playgta6::gpu::gfx10
