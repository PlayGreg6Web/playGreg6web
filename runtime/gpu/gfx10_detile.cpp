#include "gpu/gfx10_detile.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

namespace playgta6::gpu::gfx10 {

namespace {

enum class SwizzleFamily : uint8_t {
    Linear = 0,
    Standard,  // _S (D3D12 Standard Swizzle)
    Display,   // _D (Displayable / X-first Morton)
    Rotated,   // _R (Rotated / Y-first Morton)
};

struct SwizzleTraits {
    SwizzleFamily family = SwizzleFamily::Linear;
    uint32_t blk_size_log2 = 0;  // 0 for Linear, 8 for 256B, 12 for 4KB, 16 for 64KB/Var
    bool xor_mode = false;
    bool valid = false;
};

constexpr SwizzleTraits classify_mode(SwizzleMode mode) noexcept {
    switch (mode) {
        case SwizzleMode::Linear:
            return {SwizzleFamily::Linear, 0, false, true};
        case SwizzleMode::Sw256BS:
            return {SwizzleFamily::Standard, 8, false, true};
        case SwizzleMode::Sw256BD:
            return {SwizzleFamily::Display, 8, false, true};
        case SwizzleMode::Sw256BR:
            return {SwizzleFamily::Rotated, 8, false, true};
        case SwizzleMode::Sw4KBS:
            return {SwizzleFamily::Standard, 12, false, true};
        case SwizzleMode::Sw4KBD:
            return {SwizzleFamily::Display, 12, false, true};
        case SwizzleMode::Sw4KBR:
            return {SwizzleFamily::Rotated, 12, false, true};
        case SwizzleMode::Sw64KBS:
        case SwizzleMode::SwVarS:
            return {SwizzleFamily::Standard, 16, false, true};
        case SwizzleMode::Sw64KBD:
        case SwizzleMode::SwVarD:
            return {SwizzleFamily::Display, 16, false, true};
        case SwizzleMode::Sw64KBR:
        case SwizzleMode::SwVarR:
            return {SwizzleFamily::Rotated, 16, false, true};
        case SwizzleMode::Sw64KBSX:
        case SwizzleMode::SwVarSX:
            return {SwizzleFamily::Standard, 16, true, true};
        case SwizzleMode::Sw64KBDX:
        case SwizzleMode::SwVarDX:
            return {SwizzleFamily::Display, 16, true, true};
        case SwizzleMode::Sw64KBRX:
        case SwizzleMode::SwVarRX:
            return {SwizzleFamily::Rotated, 16, true, true};
        default:
            return {SwizzleFamily::Linear, 0, false, false};
    }
}

// Convert bytes_per_block in {1, 2, 4, 8, 16} to log2 in [0..4], or 0xFF if invalid.
constexpr uint32_t bpp_to_log2(uint32_t bytes_per_block) noexcept {
    switch (bytes_per_block) {
        case 1:  return 0;
        case 2:  return 1;
        case 4:  return 2;
        case 8:  return 3;
        case 16: return 4;
        default: return 0xFFu;
    }
}

// Portable software PDEP: deposits contiguous low bits of `value` into the bit positions set in `mask`.
constexpr uint32_t deposit_bits(uint32_t value, uint32_t mask) noexcept {
    uint32_t result = 0;
    uint32_t src_bit = 0;
    while (mask != 0) {
        const uint32_t lsb = mask & (~mask + 1u);
        if ((value >> src_bit) & 1u) {
            result |= lsb;
        }
        mask &= mask - 1u;
        ++src_bit;
    }
    return result;
}

// Per-address-bit swizzle equations of the non-XOR GFX10 modes, taken from AMD addrlib's public
// pattern tables (src/core/imported/addrlib/src/gfx10/gfx10SwizzlePattern.h, RB+ variants, which
// RDNA2 uses): GFX10_SW_{256,4K,64K}_{S,D}_RBPLUS_PATINFO and GFX10_SW_{4K,64K}_S3_RBPLUS_PATINFO,
// indexed by log2(bytes per element). Entry i is the source of address bit i: 0 = byte within the
// element (always 0 for an element's first byte), else axis | coordinate bit, with x/y/z counted in
// elements (texels or 4x4 blocks). In each family the 256B/4KB equation is the low 8/12 entries
// of the 64KB one, so one 16-entry table per family covers all three block sizes.
using BitEq = uint8_t;
constexpr BitEq kX = 0x10u;
constexpr BitEq kY = 0x20u;
constexpr BitEq kZ = 0x30u;
constexpr BitEq X(uint32_t b) { return static_cast<BitEq>(kX | b); }
constexpr BitEq Y(uint32_t b) { return static_cast<BitEq>(kY | b); }
constexpr BitEq Z(uint32_t b) { return static_cast<BitEq>(kZ | b); }

// SW_256B_S / SW_4KB_S / SW_64KB_S (2D; thin).
constexpr BitEq kEqStandard[5][16] = {
    {X(0), X(1), X(2), X(3), Y(0), Y(1), Y(2), Y(3), Y(4), X(4), Y(5), X(5), Y(6), X(6), Y(7), X(7)},
    {0, X(0), X(1), X(2), Y(0), Y(1), Y(2), X(3), Y(3), X(4), Y(4), X(5), Y(5), X(6), Y(6), X(7)},
    {0, 0, X(0), X(1), Y(0), Y(1), Y(2), X(2), Y(3), X(3), Y(4), X(4), Y(5), X(5), Y(6), X(6)},
    {0, 0, 0, X(0), Y(0), Y(1), X(1), X(2), Y(2), X(3), Y(3), X(4), Y(4), X(5), Y(5), X(6)},
    {0, 0, 0, 0, Y(0), Y(1), X(0), X(1), Y(2), X(2), Y(3), X(3), Y(4), X(4), Y(5), X(5)},
};

// SW_256B_D / SW_4KB_D / SW_64KB_D (2D; also used per slice for 3D surfaces).
constexpr BitEq kEqDisplay[5][16] = {
    {X(0), X(1), X(2), Y(1), Y(0), Y(2), X(3), Y(3), Y(4), X(4), Y(5), X(5), Y(6), X(6), Y(7), X(7)},
    {0, X(0), X(1), X(2), Y(0), Y(1), Y(2), X(3), Y(3), X(4), Y(4), X(5), Y(5), X(6), Y(6), X(7)},
    {0, 0, X(0), X(1), Y(0), Y(1), X(2), Y(2), Y(3), X(3), Y(4), X(4), Y(5), X(5), Y(6), X(6)},
    {0, 0, 0, X(0), Y(0), X(1), X(2), Y(1), Y(2), X(3), Y(3), X(4), Y(4), X(5), Y(5), X(6)},
    {0, 0, 0, 0, X(0), Y(0), X(1), Y(1), Y(2), X(2), Y(3), X(3), Y(4), X(4), Y(5), X(5)},
};

// SW_4KB_S / SW_64KB_S on 3D surfaces (thick, "S3").
constexpr BitEq kEqStandard3D[5][16] = {
    {X(0), X(1), Z(0), Y(0), Z(1), Y(1), X(2), Z(2), Y(2), X(3), Z(3), Y(3), X(4), Z(4), Y(4), X(5)},
    {0, X(0), Z(0), Y(0), Z(1), Y(1), X(1), Z(2), Y(2), X(2), Z(3), Y(3), X(3), Z(4), Y(4), X(4)},
    {0, 0, X(0), Y(0), Z(0), Y(1), X(1), Z(1), Y(2), X(2), Z(2), Y(3), X(3), Z(3), Y(4), X(4)},
    {0, 0, 0, X(0), Z(0), Y(0), X(1), Z(1), Y(1), X(2), Z(2), Y(2), X(3), Z(3), Y(3), X(4)},
    {0, 0, 0, 0, Z(0), Y(0), X(0), Z(1), Y(1), X(1), Z(2), Y(2), X(2), Z(3), Y(3), X(3)},
};

// 16-bit intra-tile coordinate interleave masks (x is in bytes, y is in blocks), used for the _R
// modes only. GFX10 addrlib defines no pattern for 256B_R / 4KB_R / 64KB_R / VAR_R; these masks
// are not taken from it.
struct Masks2D {
    uint16_t x_bytes_mask;
    uint16_t y_mask;
};

// Y-first Morton masks for the _R modes, indexed by bpp_log2 (0..4).
constexpr Masks2D kMasks2DRotated[5] = {
    {0xAAA3u, 0x555Cu},  // 1B  (8bpp):   256B=16x16, 4KB=64x64,  64KB=256x256
    {0xAAD3u, 0x552Cu},  // 2B  (16bpp):  256B=16x8,  4KB=64x32,  64KB=256x128
    {0xAAABu, 0x5554u},  // 4B  (32bpp):  256B=8x8,   4KB=32x32,  64KB=128x128
    {0xAAD7u, 0x5528u},  // 8B  (64bpp):  256B=8x4,   4KB=32x16,  64KB=128x64
    {0xAAAFu, 0x5550u},  // 16B (128bpp): 256B=4x4,   4KB=16x16,  64KB=64x64
};

// Compile-time verification: in every equation the low bpp_log2 entries are the byte within the
// element, and each axis uses coordinate bits 0..n-1 exactly once within 256B, 4KB and 64KB.
constexpr bool verify_equation(const BitEq* eq, uint32_t b, bool thick) noexcept {
    for (uint32_t i = 0; i < b; ++i) {
        if (eq[i] != 0) return false;
    }
    for (uint32_t blk : {8u, 12u, 16u}) {
        uint32_t seen[4] = {0, 0, 0, 0};
        for (uint32_t i = b; i < blk; ++i) {
            const uint32_t axis = eq[i] >> 4;
            const uint32_t bit = eq[i] & 0xFu;
            if (axis == 0 || axis > 3 || (axis == 3 && !thick)) return false;
            if (seen[axis] & (1u << bit)) return false;
            seen[axis] |= 1u << bit;
        }
        for (uint32_t axis = 1; axis <= 3; ++axis) {
            if ((seen[axis] & (seen[axis] + 1u)) != 0) return false;  // bits 0..n-1, no gaps
        }
    }
    return true;
}

constexpr bool verify_mask_tables() noexcept {
    constexpr uint32_t kExpectedW256[5] = {4, 4, 3, 3, 2};
    constexpr uint32_t kExpectedH256[5] = {4, 3, 3, 2, 2};
    for (uint32_t b = 0; b < 5; ++b) {
        if (!verify_equation(kEqStandard[b], b, false) || !verify_equation(kEqDisplay[b], b, false) ||
            !verify_equation(kEqStandard3D[b], b, true)) {
            return false;
        }
        const uint16_t bpp_bits = static_cast<uint16_t>((1u << b) - 1u);
        const Masks2D sets[1] = {kMasks2DRotated[b]};
        for (const auto& m : sets) {
            if ((m.x_bytes_mask & m.y_mask) != 0u) return false;
            if ((m.x_bytes_mask | m.y_mask) != 0xFFFFu) return false;
            if ((m.x_bytes_mask & bpp_bits) != bpp_bits) return false;
            const uint32_t x256 = static_cast<uint32_t>(std::popcount(static_cast<uint16_t>(m.x_bytes_mask & 0xFFu)));
            const uint32_t y256 = static_cast<uint32_t>(std::popcount(static_cast<uint16_t>(m.y_mask & 0xFFu)));
            if (x256 - b != kExpectedW256[b] || y256 != kExpectedH256[b]) return false;
            const uint32_t x4k = static_cast<uint32_t>(std::popcount(static_cast<uint16_t>(m.x_bytes_mask & 0xFFFu)));
            const uint32_t y4k = static_cast<uint32_t>(std::popcount(static_cast<uint16_t>(m.y_mask & 0xFFFu)));
            if (x4k - b != kExpectedW256[b] + 2u || y4k != kExpectedH256[b] + 2u) return false;
            const uint32_t x64k = static_cast<uint32_t>(std::popcount(m.x_bytes_mask));
            const uint32_t y64k = static_cast<uint32_t>(std::popcount(m.y_mask));
            if (x64k - b != kExpectedW256[b] + 4u || y64k != kExpectedH256[b] + 4u) return false;
        }
    }
    return true;
}
static_assert(verify_mask_tables(), "GFX10 swizzle mask table invariant failed");

// GFX10 64KB/Var _X pipe-bank XOR: leaves bits [0..7] (256B pipe interleave) untouched and XORs
// pipe/bank bits [8..11] with reverse-ordered upper intra-tile bits [15..12]. Bijective involution.
constexpr uint32_t apply_gfx10_xor_64k(uint32_t intra) noexcept {
    const uint32_t hi = (intra >> 12) & 0xFu;
    const uint32_t rev = ((hi & 1u) << 3) | ((hi & 2u) << 1) | ((hi & 4u) >> 1) | ((hi & 8u) >> 3);
    return intra ^ (rev << 8);
}

struct TileShape {
    uint32_t w_log2 = 0;
    uint32_t h_log2 = 0;
    uint32_t d_log2 = 0;
    uint32_t blk_size_log2 = 0;
    const BitEq* eq = nullptr;  // _S / _D / S3: addrlib equation (first blk_size_log2 entries)
    uint16_t x_mask = 0;        // _R only (eq == nullptr)
    uint16_t y_mask = 0;
    bool xor_mode = false;
    bool thick_3d = false;
};

constexpr TileShape compute_tile_shape(SwizzleTraits traits, uint32_t bpp_log2, bool is_3d) noexcept {
    TileShape s{};
    s.blk_size_log2 = traits.blk_size_log2;
    s.xor_mode = traits.xor_mode;
    const uint16_t blk_mask = static_cast<uint16_t>((1u << traits.blk_size_log2) - 1u);

    // addrlib (Gfx10Lib::GetSwizzlePatternInfo): 3D surfaces in 4KB_S / 64KB_S use the thick S3
    // patterns; 256B modes and _D / _R modes stack 2D tiled slices along Z.
    const bool thick = is_3d && traits.family == SwizzleFamily::Standard && traits.blk_size_log2 >= 12u;
    if (traits.family == SwizzleFamily::Standard || traits.family == SwizzleFamily::Display) {
        s.eq = thick ? kEqStandard3D[bpp_log2]
                     : (traits.family == SwizzleFamily::Standard ? kEqStandard[bpp_log2] : kEqDisplay[bpp_log2]);
        uint32_t extent[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < traits.blk_size_log2; ++i) {
            const uint32_t axis = s.eq[i] >> 4;
            if (axis != 0) {
                extent[axis] = std::max(extent[axis], (s.eq[i] & 0xFu) + 1u);
            }
        }
        s.w_log2 = extent[1];
        s.h_log2 = extent[2];
        s.d_log2 = extent[3];
        s.thick_3d = thick;
    } else {
        const Masks2D m = kMasks2DRotated[bpp_log2];
        s.x_mask = static_cast<uint16_t>(m.x_bytes_mask & blk_mask);
        s.y_mask = static_cast<uint16_t>(m.y_mask & blk_mask);
        s.w_log2 = static_cast<uint32_t>(std::popcount(s.x_mask)) - bpp_log2;
        s.h_log2 = static_cast<uint32_t>(std::popcount(s.y_mask));
        s.d_log2 = 0;
        s.thick_3d = false;
    }
    return s;
}

constexpr uint32_t apply_equation(const BitEq* eq, uint32_t blk_size_log2, uint32_t x, uint32_t y,
                                  uint32_t z) noexcept {
    uint32_t intra = 0;
    for (uint32_t i = 0; i < blk_size_log2; ++i) {
        const uint32_t axis = eq[i] >> 4;
        const uint32_t bit = eq[i] & 0xFu;
        const uint32_t coord = axis == 1 ? x : (axis == 2 ? y : z);
        if (axis != 0) {
            intra |= ((coord >> bit) & 1u) << i;
        }
    }
    return intra;
}

constexpr size_t tiled_byte_offset_shape(const TileShape& shape, uint32_t bpp_log2, uint32_t bx, uint32_t by,
                                         uint32_t bz, uint32_t tiles_per_row, uint32_t tiles_per_col) noexcept {
    const uint32_t tile_w_mask = (1u << shape.w_log2) - 1u;
    const uint32_t tile_h_mask = (1u << shape.h_log2) - 1u;
    const uint32_t tile_d_mask = (1u << shape.d_log2) - 1u;

    const uint32_t tile_x = bx >> shape.w_log2;
    const uint32_t tile_y = by >> shape.h_log2;
    const uint32_t tile_z = bz >> shape.d_log2;

    const uint32_t local_y = by & tile_h_mask;
    const uint32_t local_z = bz & tile_d_mask;

    uint32_t intra = 0;
    if (shape.eq != nullptr) {
        intra = apply_equation(shape.eq, shape.blk_size_log2, bx & tile_w_mask, local_y, local_z);
    } else {
        const uint32_t local_x_bytes = (bx & tile_w_mask) << bpp_log2;
        intra = deposit_bits(local_x_bytes, shape.x_mask) | deposit_bits(local_y, shape.y_mask);
    }
    if (shape.xor_mode) {
        intra = apply_gfx10_xor_64k(intra);
    }

    const size_t tiles_per_slice = static_cast<size_t>(tiles_per_row) * static_cast<size_t>(tiles_per_col);
    const size_t tile_index = static_cast<size_t>(tile_z) * tiles_per_slice +
                              static_cast<size_t>(tile_y) * static_cast<size_t>(tiles_per_row) +
                              static_cast<size_t>(tile_x);
    return (tile_index << shape.blk_size_log2) + static_cast<size_t>(intra);
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out) noexcept {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    *out = a * b;
    return true;
}

struct SubresourcePlan {
    bool ok = false;
    const char* error = nullptr;
    SwizzleMode mode = SwizzleMode::Unsupported;
    SwizzleTraits traits{};
    FormatGeometry fmt{};
    uint32_t bpp_log2 = 0;
    uint32_t blocks_x = 0;
    uint32_t blocks_y = 0;
    uint32_t slices = 0;
    uint32_t linear_row_pitch = 0;
    uint32_t src_row_pitch = 0;  // Linear mode only
    TileShape tile_shape{};      // Tiled modes only
    uint32_t tiles_x = 0;        // Tiled modes only
    uint32_t tiles_y = 0;        // Tiled modes only
    uint32_t tiles_z = 0;        // Tiled modes only
    size_t linear_total_bytes = 0;
    size_t tiled_total_bytes = 0;
};

SubresourcePlan plan_subresource(const TextureLayoutDesc& desc, uint32_t mip) noexcept {
    SubresourcePlan p{};

    p.fmt = format_geometry(desc.dxgi_format);
    if (!p.fmt.supported) {
        p.error = "unsupported DXGI format";
        return p;
    }
    p.bpp_log2 = bpp_to_log2(p.fmt.bytes_per_block);
    if (p.bpp_log2 == 0xFFu) {
        p.error = "unsupported format block byte size";
        return p;
    }

    p.mode = decode_d3d12_layout(desc.d3d12_layout);
    p.traits = classify_mode(p.mode);
    if (!p.traits.valid) {
        p.error = "unsupported D3D12 texture layout / swizzle mode";
        return p;
    }

    if (desc.width == 0 || desc.height == 0 || desc.depth == 0) {
        p.error = "invalid texture dimensions (width, height, and depth must be non-zero)";
        return p;
    }

    const uint32_t effective_mips = desc.mip_levels == 0 ? 1u : desc.mip_levels;
    if (mip >= effective_mips) {
        p.error = "mip level out of range";
        return p;
    }

    const uint32_t mip_w = mip >= 32u ? 1u : std::max(1u, desc.width >> mip);
    const uint32_t mip_h = mip >= 32u ? 1u : std::max(1u, desc.height >> mip);
    const uint32_t mip_d = mip >= 32u ? 1u : std::max(1u, desc.depth >> mip);

    const uint64_t bx64 = (static_cast<uint64_t>(mip_w) + p.fmt.block_width - 1u) / p.fmt.block_width;
    const uint64_t by64 = (static_cast<uint64_t>(mip_h) + p.fmt.block_height - 1u) / p.fmt.block_height;
    if (bx64 == 0 || by64 == 0 || bx64 > std::numeric_limits<uint32_t>::max() ||
        by64 > std::numeric_limits<uint32_t>::max()) {
        p.error = "subresource block dimensions overflow";
        return p;
    }
    p.blocks_x = static_cast<uint32_t>(bx64);
    p.blocks_y = static_cast<uint32_t>(by64);
    p.slices = mip_d;

    uint64_t linear_row64 = 0;
    if (!checked_mul_u64(p.blocks_x, p.fmt.bytes_per_block, &linear_row64) ||
        linear_row64 > std::numeric_limits<uint32_t>::max()) {
        p.error = "linear row pitch overflow";
        return p;
    }
    p.linear_row_pitch = static_cast<uint32_t>(linear_row64);

    uint64_t linear_slice64 = 0;
    uint64_t linear_total64 = 0;
    if (!checked_mul_u64(linear_row64, p.blocks_y, &linear_slice64) ||
        !checked_mul_u64(linear_slice64, p.slices, &linear_total64) ||
        linear_total64 > std::numeric_limits<size_t>::max()) {
        p.error = "linear output size overflow";
        return p;
    }
    p.linear_total_bytes = static_cast<size_t>(linear_total64);

    if (p.traits.family == SwizzleFamily::Linear) {
        if (desc.pitch_bytes != 0 && desc.pitch_bytes < p.linear_row_pitch) {
            p.error = "linear pitch_bytes is smaller than tight row width";
            return p;
        }
        p.src_row_pitch = desc.pitch_bytes != 0 ? desc.pitch_bytes : p.linear_row_pitch;

        uint64_t src_slice64 = 0;
        uint64_t src_total64 = 0;
        if (!checked_mul_u64(p.src_row_pitch, p.blocks_y, &src_slice64) ||
            !checked_mul_u64(src_slice64, p.slices, &src_total64) ||
            src_total64 > std::numeric_limits<size_t>::max()) {
            p.error = "linear source footprint overflow";
            return p;
        }
        p.tiled_total_bytes = static_cast<size_t>(src_total64);
    } else {
        uint32_t pitch_in_blocks = p.blocks_x;
        if (desc.pitch_bytes != 0) {
            if (desc.pitch_bytes < p.linear_row_pitch) {
                p.error = "tiled pitch_bytes is smaller than tight row width";
                return p;
            }
            if ((desc.pitch_bytes % p.fmt.bytes_per_block) != 0u) {
                p.error = "tiled pitch_bytes is not a multiple of bytes_per_block";
                return p;
            }
            pitch_in_blocks = desc.pitch_bytes / p.fmt.bytes_per_block;
        }

        const bool is_3d = desc.depth > 1u;
        p.tile_shape = compute_tile_shape(p.traits, p.bpp_log2, is_3d);

        const uint64_t tile_w = uint64_t{1} << p.tile_shape.w_log2;
        const uint64_t tile_h = uint64_t{1} << p.tile_shape.h_log2;
        const uint64_t tile_d = uint64_t{1} << p.tile_shape.d_log2;
        const uint64_t tx64 = (static_cast<uint64_t>(pitch_in_blocks) + tile_w - 1u) >> p.tile_shape.w_log2;
        const uint64_t ty64 = (static_cast<uint64_t>(p.blocks_y) + tile_h - 1u) >> p.tile_shape.h_log2;
        const uint64_t tz64 = (static_cast<uint64_t>(p.slices) + tile_d - 1u) >> p.tile_shape.d_log2;
        if (tx64 == 0 || ty64 == 0 || tz64 == 0 || tx64 > std::numeric_limits<uint32_t>::max() ||
            ty64 > std::numeric_limits<uint32_t>::max() || tz64 > std::numeric_limits<uint32_t>::max()) {
            p.error = "tiled grid dimensions overflow";
            return p;
        }
        p.tiles_x = static_cast<uint32_t>(tx64);
        p.tiles_y = static_cast<uint32_t>(ty64);
        p.tiles_z = static_cast<uint32_t>(tz64);

        uint64_t tiles_xy64 = 0;
        uint64_t total_tiles64 = 0;
        uint64_t total_bytes64 = 0;
        const uint64_t tile_bytes = uint64_t{1} << p.tile_shape.blk_size_log2;
        if (!checked_mul_u64(tx64, ty64, &tiles_xy64) ||
            !checked_mul_u64(tiles_xy64, tz64, &total_tiles64) ||
            !checked_mul_u64(total_tiles64, tile_bytes, &total_bytes64) ||
            total_bytes64 > std::numeric_limits<size_t>::max()) {
            p.error = "tiled footprint byte size overflow";
            return p;
        }
        p.tiled_total_bytes = static_cast<size_t>(total_bytes64);
    }

    p.ok = true;
    return p;
}

}  // namespace

SwizzleMode decode_d3d12_layout(uint32_t d3d12_layout) noexcept {
    if (d3d12_layout == 0u || d3d12_layout == 1u) {
        return SwizzleMode::Linear;
    }
    if (d3d12_layout >= 0x100u && d3d12_layout <= 0x11Fu) {
        switch (d3d12_layout & 0x1Fu) {
            case 0:  return SwizzleMode::Linear;
            case 1:  return SwizzleMode::Sw256BS;
            case 2:  return SwizzleMode::Sw256BD;
            case 3:  return SwizzleMode::Sw256BR;
            case 5:  return SwizzleMode::Sw4KBS;
            case 6:  return SwizzleMode::Sw4KBD;
            case 7:  return SwizzleMode::Sw4KBR;
            case 9:  return SwizzleMode::Sw64KBS;
            case 10: return SwizzleMode::Sw64KBD;
            case 11: return SwizzleMode::Sw64KBR;
            case 13: return SwizzleMode::SwVarS;
            case 14: return SwizzleMode::SwVarD;
            case 15: return SwizzleMode::SwVarR;
            case 25: return SwizzleMode::Sw64KBSX;
            case 26: return SwizzleMode::Sw64KBDX;
            case 27: return SwizzleMode::Sw64KBRX;
            case 29: return SwizzleMode::SwVarSX;
            case 30: return SwizzleMode::SwVarDX;
            case 31: return SwizzleMode::SwVarRX;
            default: return SwizzleMode::Unsupported;
        }
    }
    return SwizzleMode::Unsupported;
}

FormatGeometry format_geometry(uint32_t dxgi_format) noexcept {
    switch (dxgi_format) {
        // 16 bytes per pixel (128-bit)
        case 1:   // DXGI_FORMAT_R32G32B32A32_TYPELESS
        case 2:   // DXGI_FORMAT_R32G32B32A32_FLOAT
        case 3:   // DXGI_FORMAT_R32G32B32A32_UINT
        case 4:   // DXGI_FORMAT_R32G32B32A32_SINT
            return {1, 1, 16, false, true};

        // 8 bytes per pixel (64-bit)
        case 9:   // DXGI_FORMAT_R16G16B16A16_TYPELESS
        case 10:  // DXGI_FORMAT_R16G16B16A16_FLOAT
        case 11:  // DXGI_FORMAT_R16G16B16A16_UNORM
        case 12:  // DXGI_FORMAT_R16G16B16A16_UINT
        case 13:  // DXGI_FORMAT_R16G16B16A16_SNORM
        case 14:  // DXGI_FORMAT_R16G16B16A16_SINT
        case 15:  // DXGI_FORMAT_R32G32_TYPELESS
        case 16:  // DXGI_FORMAT_R32G32_FLOAT
        case 17:  // DXGI_FORMAT_R32G32_UINT
        case 18:  // DXGI_FORMAT_R32G32_SINT
            return {1, 1, 8, false, true};

        // 4 bytes per pixel (32-bit)
        case 23:  // DXGI_FORMAT_R10G10B10A2_TYPELESS
        case 24:  // DXGI_FORMAT_R10G10B10A2_UNORM
        case 25:  // DXGI_FORMAT_R10G10B10A2_UINT
        case 26:  // DXGI_FORMAT_R11G11B10_FLOAT
        case 27:  // DXGI_FORMAT_R8G8B8A8_TYPELESS
        case 28:  // DXGI_FORMAT_R8G8B8A8_UNORM
        case 29:  // DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        case 30:  // DXGI_FORMAT_R8G8B8A8_UINT
        case 31:  // DXGI_FORMAT_R8G8B8A8_SNORM
        case 32:  // DXGI_FORMAT_R8G8B8A8_SINT
        case 33:  // DXGI_FORMAT_R16G16_TYPELESS
        case 34:  // DXGI_FORMAT_R16G16_FLOAT
        case 35:  // DXGI_FORMAT_R16G16_UNORM
        case 36:  // DXGI_FORMAT_R16G16_UINT
        case 37:  // DXGI_FORMAT_R16G16_SNORM
        case 38:  // DXGI_FORMAT_R16G16_SINT
        case 39:  // DXGI_FORMAT_R32_TYPELESS
        case 40:  // DXGI_FORMAT_D32_FLOAT
        case 41:  // DXGI_FORMAT_R32_FLOAT
        case 42:  // DXGI_FORMAT_R32_UINT
        case 43:  // DXGI_FORMAT_R32_SINT
        case 67:  // DXGI_FORMAT_R9G9B9E5_SHAREDEXP
        case 87:  // DXGI_FORMAT_B8G8R8A8_UNORM
        case 88:  // DXGI_FORMAT_B8G8R8X8_UNORM
        case 90:  // DXGI_FORMAT_B8G8R8A8_TYPELESS
        case 91:  // DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        case 92:  // DXGI_FORMAT_B8G8R8X8_TYPELESS
        case 93:  // DXGI_FORMAT_B8G8R8X8_UNORM_SRGB
            return {1, 1, 4, false, true};

        // 2 bytes per pixel (16-bit)
        case 48:  // DXGI_FORMAT_R8G8_TYPELESS
        case 49:  // DXGI_FORMAT_R8G8_UNORM
        case 50:  // DXGI_FORMAT_R8G8_UINT
        case 51:  // DXGI_FORMAT_R8G8_SNORM
        case 52:  // DXGI_FORMAT_R8G8_SINT
        case 53:  // DXGI_FORMAT_R16_TYPELESS
        case 54:  // DXGI_FORMAT_R16_FLOAT
        case 55:  // DXGI_FORMAT_D16_UNORM
        case 56:  // DXGI_FORMAT_R16_UNORM
        case 57:  // DXGI_FORMAT_R16_UINT
        case 58:  // DXGI_FORMAT_R16_SNORM
        case 59:  // DXGI_FORMAT_R16_SINT
            return {1, 1, 2, false, true};

        // 1 byte per pixel (8-bit)
        case 60:  // DXGI_FORMAT_R8_TYPELESS
        case 61:  // DXGI_FORMAT_R8_UNORM
        case 62:  // DXGI_FORMAT_R8_UINT
        case 63:  // DXGI_FORMAT_R8_SNORM
        case 64:  // DXGI_FORMAT_R8_SINT
        case 65:  // DXGI_FORMAT_A8_UNORM
            return {1, 1, 1, false, true};

        // Block-compressed 4x4, 8 bytes per block (BC1, BC4)
        case 70:  // DXGI_FORMAT_BC1_TYPELESS
        case 71:  // DXGI_FORMAT_BC1_UNORM
        case 72:  // DXGI_FORMAT_BC1_UNORM_SRGB
        case 79:  // DXGI_FORMAT_BC4_TYPELESS
        case 80:  // DXGI_FORMAT_BC4_UNORM
        case 81:  // DXGI_FORMAT_BC4_SNORM
            return {4, 4, 8, true, true};

        // Block-compressed 4x4, 16 bytes per block (BC2, BC3, BC5, BC6H, BC7)
        case 73:  // DXGI_FORMAT_BC2_TYPELESS
        case 74:  // DXGI_FORMAT_BC2_UNORM
        case 75:  // DXGI_FORMAT_BC2_UNORM_SRGB
        case 76:  // DXGI_FORMAT_BC3_TYPELESS
        case 77:  // DXGI_FORMAT_BC3_UNORM
        case 78:  // DXGI_FORMAT_BC3_UNORM_SRGB
        case 82:  // DXGI_FORMAT_BC5_TYPELESS
        case 83:  // DXGI_FORMAT_BC5_UNORM
        case 84:  // DXGI_FORMAT_BC5_SNORM
        case 94:  // DXGI_FORMAT_BC6H_TYPELESS
        case 95:  // DXGI_FORMAT_BC6H_UF16
        case 96:  // DXGI_FORMAT_BC6H_SF16
        case 97:  // DXGI_FORMAT_BC7_TYPELESS
        case 98:  // DXGI_FORMAT_BC7_UNORM
        case 99:  // DXGI_FORMAT_BC7_UNORM_SRGB
            return {4, 4, 16, true, true};

        default:
            return {0, 0, 0, false, false};
    }
}

bool is_swizzle_supported(SwizzleMode mode) noexcept {
    return classify_mode(mode).valid;
}

size_t tiled_byte_offset_2d(SwizzleMode mode, uint32_t bx, uint32_t by, uint32_t bytes_per_block,
                            uint32_t pitch_in_blocks) noexcept {
    const SwizzleTraits traits = classify_mode(mode);
    const uint32_t bpp_log2 = bpp_to_log2(bytes_per_block);
    if (!traits.valid || bpp_log2 == 0xFFu) {
        return std::numeric_limits<size_t>::max();
    }

    if (traits.family == SwizzleFamily::Linear) {
        return (static_cast<size_t>(by) * static_cast<size_t>(pitch_in_blocks) + static_cast<size_t>(bx)) *
               static_cast<size_t>(bytes_per_block);
    }

    const TileShape shape = compute_tile_shape(traits, bpp_log2, /*is_3d=*/false);
    const uint32_t tile_w = 1u << shape.w_log2;
    const uint32_t eff_pitch = pitch_in_blocks > 0u ? pitch_in_blocks : tile_w;
    const uint32_t tiles_per_row = static_cast<uint32_t>(
        (static_cast<uint64_t>(eff_pitch) + tile_w - 1u) >> shape.w_log2);
    return tiled_byte_offset_shape(shape, bpp_log2, bx, by, /*bz=*/0u, tiles_per_row, /*tiles_per_col=*/1u);
}

size_t tiled_footprint_bytes(const TextureLayoutDesc& desc, uint32_t mip) noexcept {
    const SubresourcePlan plan = plan_subresource(desc, mip);
    return plan.ok ? plan.tiled_total_bytes : 0u;
}

DetileResult detile_subresource(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* src,
                                size_t src_bytes, std::vector<uint8_t>& linear_out) {
    DetileResult res{};
    linear_out.clear();

    const SubresourcePlan plan = plan_subresource(desc, mip);
    if (!plan.ok) {
        res.error = plan.error;
        return res;
    }

    if (src == nullptr && plan.tiled_total_bytes > 0) {
        res.error = "null source pointer";
        return res;
    }

    if (src_bytes < plan.tiled_total_bytes) {
        res.error = "source buffer truncated for subresource footprint";
        return res;
    }

    linear_out.resize(plan.linear_total_bytes);

    if (plan.traits.family == SwizzleFamily::Linear) {
        const size_t src_slice_bytes = static_cast<size_t>(plan.src_row_pitch) * plan.blocks_y;
        const size_t dst_slice_bytes = static_cast<size_t>(plan.linear_row_pitch) * plan.blocks_y;
        for (uint32_t bz = 0; bz < plan.slices; ++bz) {
            const uint8_t* src_slice = src + static_cast<size_t>(bz) * src_slice_bytes;
            uint8_t* dst_slice = linear_out.data() + static_cast<size_t>(bz) * dst_slice_bytes;
            if (plan.src_row_pitch == plan.linear_row_pitch) {
                std::memcpy(dst_slice, src_slice, dst_slice_bytes);
            } else {
                for (uint32_t by = 0; by < plan.blocks_y; ++by) {
                    std::memcpy(dst_slice + static_cast<size_t>(by) * plan.linear_row_pitch,
                                src_slice + static_cast<size_t>(by) * plan.src_row_pitch,
                                plan.linear_row_pitch);
                }
            }
        }
    } else {
        const uint32_t bpp = plan.fmt.bytes_per_block;
        const size_t dst_slice_bytes = static_cast<size_t>(plan.linear_row_pitch) * plan.blocks_y;
        for (uint32_t bz = 0; bz < plan.slices; ++bz) {
            uint8_t* dst_slice = linear_out.data() + static_cast<size_t>(bz) * dst_slice_bytes;
            for (uint32_t by = 0; by < plan.blocks_y; ++by) {
                uint8_t* dst_row = dst_slice + static_cast<size_t>(by) * plan.linear_row_pitch;
                for (uint32_t bx = 0; bx < plan.blocks_x; ++bx) {
                    const size_t src_off = tiled_byte_offset_shape(plan.tile_shape, plan.bpp_log2, bx, by, bz,
                                                                   plan.tiles_x, plan.tiles_y);
                    if (src_off > src_bytes || bpp > src_bytes - src_off) {
                        linear_out.clear();
                        res.error = "tiled source offset out of bounds";
                        return res;
                    }
                    std::memcpy(dst_row + static_cast<size_t>(bx) * bpp, src + src_off, bpp);
                }
            }
        }
    }

    res.ok = true;
    res.row_pitch_bytes = plan.linear_row_pitch;
    res.rows = plan.blocks_y;
    res.slices = plan.slices;
    res.tiled_bytes_consumed = plan.tiled_total_bytes;
    res.error = nullptr;
    return res;
}

bool tile_subresource_for_test(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* linear_src,
                               size_t linear_src_bytes, std::vector<uint8_t>& tiled_out) {
    tiled_out.clear();

    const SubresourcePlan plan = plan_subresource(desc, mip);
    if (!plan.ok) {
        return false;
    }
    if (linear_src == nullptr && plan.linear_total_bytes > 0) {
        return false;
    }
    if (linear_src_bytes < plan.linear_total_bytes) {
        return false;
    }

    tiled_out.assign(plan.tiled_total_bytes, 0u);

    if (plan.traits.family == SwizzleFamily::Linear) {
        const size_t src_slice_bytes = static_cast<size_t>(plan.linear_row_pitch) * plan.blocks_y;
        const size_t dst_slice_bytes = static_cast<size_t>(plan.src_row_pitch) * plan.blocks_y;
        for (uint32_t bz = 0; bz < plan.slices; ++bz) {
            const uint8_t* src_slice = linear_src + static_cast<size_t>(bz) * src_slice_bytes;
            uint8_t* dst_slice = tiled_out.data() + static_cast<size_t>(bz) * dst_slice_bytes;
            for (uint32_t by = 0; by < plan.blocks_y; ++by) {
                std::memcpy(dst_slice + static_cast<size_t>(by) * plan.src_row_pitch,
                            src_slice + static_cast<size_t>(by) * plan.linear_row_pitch,
                            plan.linear_row_pitch);
            }
        }
    } else {
        const uint32_t bpp = plan.fmt.bytes_per_block;
        const size_t src_slice_bytes = static_cast<size_t>(plan.linear_row_pitch) * plan.blocks_y;
        for (uint32_t bz = 0; bz < plan.slices; ++bz) {
            const uint8_t* src_slice = linear_src + static_cast<size_t>(bz) * src_slice_bytes;
            for (uint32_t by = 0; by < plan.blocks_y; ++by) {
                const uint8_t* src_row = src_slice + static_cast<size_t>(by) * plan.linear_row_pitch;
                for (uint32_t bx = 0; bx < plan.blocks_x; ++bx) {
                    const size_t dst_off = tiled_byte_offset_shape(plan.tile_shape, plan.bpp_log2, bx, by, bz,
                                                                   plan.tiles_x, plan.tiles_y);
                    if (dst_off > tiled_out.size() || bpp > tiled_out.size() - dst_off) {
                        tiled_out.clear();
                        return false;
                    }
                    std::memcpy(tiled_out.data() + dst_off, src_row + static_cast<size_t>(bx) * bpp, bpp);
                }
            }
        }
    }

    return true;
}

namespace {

constexpr uint32_t kMaxMips = 16;

uint32_t shift_ceil(uint32_t v, uint32_t s) noexcept {
    return s >= 32u ? 1u : std::max(1u, static_cast<uint32_t>((uint64_t{v} + (uint64_t{1} << s) - 1u) >> s));
}

uint64_t align_pow2(uint64_t v, uint64_t a) noexcept {
    return (v + a - 1u) & ~(a - 1u);
}

// The whole-surface layout of a tiled resource (see MipPlacement in the header).
struct ChainLayout {
    bool ok = false;
    const char* error = nullptr;
    TileShape shape{};
    FormatGeometry fmt{};
    uint32_t bpp_log2 = 0;
    uint32_t mips = 1;
    uint32_t bw = 1, bh = 1, bd = 1;  // block dimensions in elements
    uint64_t slice_stride = 0;        // bytes per slice (thin) or per block of slices (thick)
    uint64_t total_bytes = 0;
    uint32_t pitch[kMaxMips] = {};    // padded mip pitch in elements
    uint32_t height[kMaxMips] = {};   // padded mip height in elements
    uint64_t slice_bytes[kMaxMips] = {};
    uint64_t size_bytes[kMaxMips] = {};
    uint64_t offset[kMaxMips] = {};   // addrlib ADDR2_MIP_INFO::offset
    uint64_t macro_offset[kMaxMips] = {};
    bool in_tail[kMaxMips] = {};
    uint32_t tail_x[kMaxMips] = {};
    uint32_t tail_y[kMaxMips] = {};
    bool linear = false;  // console linear (ADDR_SW_LINEAR): pitched rows, no blocks
};

// The console linear layout (0x100 | SW_LINEAR) without an explicit pitch: placed by the same
// chain code as the tiled modes. Layouts 0 / 1 (D3D12 UNKNOWN / ROW_MAJOR) and explicit pitches
// keep the single-subresource path (mip 0 at the base, rows at pitch_bytes or tight).
bool is_console_linear(const TextureLayoutDesc& desc) noexcept {
    return desc.d3d12_layout == kConsoleLinearLayout && desc.pitch_bytes == 0u;
}

ChainLayout build_chain(const TextureLayoutDesc& desc) noexcept {
    ChainLayout c{};
    c.fmt = format_geometry(desc.dxgi_format);
    if (!c.fmt.supported) {
        c.error = "unsupported DXGI format";
        return c;
    }
    c.bpp_log2 = bpp_to_log2(c.fmt.bytes_per_block);
    const SwizzleTraits traits = classify_mode(decode_d3d12_layout(desc.d3d12_layout));
    c.linear = is_console_linear(desc);
    if (!traits.valid || (traits.family == SwizzleFamily::Linear && !c.linear)) {
        c.error = "not a tiled layout";
        return c;
    }
    if (desc.width == 0 || desc.height == 0 || desc.depth == 0) {
        c.error = "invalid texture dimensions";
        return c;
    }
    if (desc.depth > 1u && desc.array_size > 1u) {
        c.error = "3D textures have no array slices";
        return c;
    }
    c.mips = std::min(std::max(desc.mip_levels, 1u), kMaxMips);
    const uint32_t ew = (desc.width + c.fmt.block_width - 1u) / c.fmt.block_width;
    const uint32_t eh = (desc.height + c.fmt.block_height - 1u) / c.fmt.block_height;
    const uint32_t depth = desc.depth;
    if (c.linear) {
        // AMD addrlib Gfx10Lib::HwlComputeSurfaceInfoLinear, ADDR_SW_LINEAR: the pitch of every
        // mip is aligned to 256 bytes (256 / bytes-per-element elements), mip sizes are rounded up
        // (Gfx10Lib::GetMipSize: ShiftCeil of the element size), heights are not padded,
        // mips are stored smallest-first (mip 0 last), and each slice (array slice or 3D depth
        // slice) holds the whole chain: slice stride = chain size, surface = chain x slices.
        const uint64_t bpe = c.fmt.bytes_per_block;
        const uint32_t pitch_align = 256u / c.fmt.bytes_per_block;
        uint64_t chain = 0;
        for (int32_t i = static_cast<int32_t>(c.mips) - 1; i >= 0; --i) {
            const uint32_t m = static_cast<uint32_t>(i);
            c.pitch[m] = static_cast<uint32_t>(align_pow2(shift_ceil(ew, m), pitch_align));
            c.height[m] = shift_ceil(eh, m);
            c.slice_bytes[m] = uint64_t{c.pitch[m]} * c.height[m] * bpe;
            c.size_bytes[m] = c.slice_bytes[m] * (depth > 1u ? shift_ceil(depth, m) : 1u);
            c.offset[m] = c.macro_offset[m] = chain;
            chain += c.slice_bytes[m];
        }
        c.slice_stride = chain;
        c.total_bytes = chain * depth * std::max(desc.array_size, 1u);
        c.ok = true;
        return c;
    }
    c.shape = compute_tile_shape(traits, c.bpp_log2, depth > 1u);
    const bool thick = c.shape.thick_3d;
    c.bw = 1u << c.shape.w_log2;
    c.bh = 1u << c.shape.h_log2;
    c.bd = thick ? (1u << c.shape.d_log2) : 1u;
    const uint32_t blk = c.shape.blk_size_log2;
    const uint64_t block_bytes = uint64_t{1} << blk;
    const uint64_t bpe = c.fmt.bytes_per_block;
    const uint64_t num_slices = thick ? align_pow2(depth, c.bd) : uint64_t{depth} * std::max(desc.array_size, 1u);

    uint64_t chain = 0;
    if (blk == 8u) {
        // Micro-tiled (256B): mips smallest-first, each padded to whole blocks, no tail.
        for (int32_t i = static_cast<int32_t>(c.mips) - 1; i >= 0; --i) {
            const uint32_t w = shift_ceil(ew, static_cast<uint32_t>(i));
            const uint32_t h = shift_ceil(eh, static_cast<uint32_t>(i));
            c.pitch[i] = static_cast<uint32_t>(align_pow2(w, c.bw));
            c.height[i] = static_cast<uint32_t>(align_pow2(h, c.bh));
            c.macro_offset[i] = c.offset[i] = chain;
            c.slice_bytes[i] = c.size_bytes[i] = uint64_t{c.pitch[i]} * c.height[i] * bpe;
            chain += c.slice_bytes[i];
        }
    } else if (c.mips == 1u) {
        c.pitch[0] = static_cast<uint32_t>(align_pow2(ew, c.bw));
        c.height[0] = static_cast<uint32_t>(align_pow2(eh, c.bh));
        c.slice_bytes[0] = uint64_t{c.pitch[0]} * c.height[0] * bpe;
        chain = c.slice_bytes[0];
        c.size_bytes[0] = chain * num_slices;
    } else {
        // Mip tail: the largest region that a mip may occupy in the tail block (addrlib
        // Lib::GetMipTailDim), and how many mips fit (Gfx10Lib::GetMaxNumMipsInTail).
        uint32_t tw = c.bw, th = c.bh;
        if (thick) {
            const uint32_t dim = blk % 3u;
            if (dim == 0u) {
                th >>= 1;
            } else if (dim == 1u) {
                tw >>= 1;
            }  // dim == 2 halves the depth, which IsInMipTail does not test
        } else {
            tw >>= 1;
        }
        const uint32_t eff = thick ? blk - (blk - 8u) / 3u : blk;
        const uint32_t max_mips_in_tail = eff <= 11u ? 1u + (1u << (eff - 9u)) : eff - 4u;
        uint32_t first_tail = c.mips;
        uint64_t mip_slice_size[kMaxMips] = {};
        for (uint32_t i = 0; i < c.mips; ++i) {
            const uint32_t w = shift_ceil(ew, i);
            const uint32_t h = shift_ceil(eh, i);
            if (w <= tw && h <= th && c.mips - i <= max_mips_in_tail) {
                first_tail = i;
                chain += block_bytes / c.bd;
                break;
            }
            c.pitch[i] = static_cast<uint32_t>(align_pow2(w, c.bw));
            c.height[i] = static_cast<uint32_t>(align_pow2(h, c.bh));
            const uint64_t slice = uint64_t{c.pitch[i]} * c.height[i] * bpe;
            const uint32_t d = thick ? shift_ceil(depth, i) : 1u;
            c.slice_bytes[i] = slice;
            c.size_bytes[i] = slice * align_pow2(d, c.bd);
            mip_slice_size[i] = slice * c.bd;
            chain += slice;
        }
        uint32_t b256d = 0;
        for (uint32_t i = 0; i < 8u && c.shape.eq != nullptr; ++i) {
            if ((c.shape.eq[i] >> 4) == 3u) b256d = std::max(b256d, (c.shape.eq[i] & 0xFu) + 1u);
        }
        uint32_t tail_depth_blocks = 1;
        uint64_t macro = 0;
        uint64_t offset = 0;
        if (first_tail != c.mips) {
            const uint32_t tail_depth = thick ? shift_ceil(depth, first_tail) : 1u;
            offset = block_bytes * align_pow2(tail_depth, c.bd) / c.bd;
            macro = block_bytes;
            if (thick) {
                tail_depth_blocks = static_cast<uint32_t>(align_pow2(tail_depth, 1u << b256d) >> b256d);
            }
        }
        for (int32_t i = static_cast<int32_t>(first_tail) - 1; i >= 0; --i) {
            c.macro_offset[i] = macro;
            c.offset[i] = offset;
            macro += mip_slice_size[i];
            offset += c.size_bytes[i];
        }
        // Tail mips: placed by coordinate inside the block (Gfx10Lib::ComputeSurfaceInfoMacroTiled).
        uint32_t b256w = 0, b256h = 0;
        for (uint32_t i = 0; i < 8u && c.shape.eq != nullptr; ++i) {
            const uint32_t axis = c.shape.eq[i] >> 4;
            const uint32_t bit = (c.shape.eq[i] & 0xFu) + 1u;
            if (axis == 1u) b256w = std::max(b256w, bit);
            if (axis == 2u) b256h = std::max(b256h, bit);
        }
        if (c.shape.eq == nullptr) {
            constexpr uint32_t kW256[5] = {4, 4, 3, 3, 2};
            constexpr uint32_t kH256[5] = {4, 3, 3, 2, 2};
            b256w = kW256[c.bpp_log2];
            b256h = kH256[c.bpp_log2];
        }
        for (uint32_t i = first_tail; i < c.mips; ++i) {
            const uint32_t m = max_mips_in_tail - 1u - (i - first_tail);
            const uint32_t off = m > 6u ? (16u << m) : (m << 8);
            uint32_t mx = ((off >> 9) & 1u) | ((off >> 10) & 2u) | ((off >> 11) & 4u) | ((off >> 12) & 8u) |
                          ((off >> 13) & 16u) | ((off >> 14) & 32u);
            uint32_t my = ((off >> 8) & 1u) | ((off >> 9) & 2u) | ((off >> 10) & 4u) | ((off >> 11) & 8u) |
                          ((off >> 12) & 16u) | ((off >> 13) & 32u);
            if (blk & 1u) {
                std::swap(mx, my);
                if (c.bpp_log2 & 1u) {
                    my = (my << 1) | (mx & 1u);
                    mx >>= 1;
                }
            }
            c.in_tail[i] = true;
            c.tail_x[i] = mx << b256w;
            c.tail_y[i] = my << b256h;
            c.macro_offset[i] = 0;
            c.offset[i] = uint64_t{off} * tail_depth_blocks;
            // Tail mip pitch/height: the tail region, halved per level down to one 256 B block.
            const uint32_t level = i - first_tail;
            c.pitch[i] = std::max(tw >> level, 1u << b256w);
            c.height[i] = std::max(th >> level, 1u << b256h);
            c.slice_bytes[i] = uint64_t{c.pitch[i]} * c.height[i] * bpe;
            c.size_bytes[i] = c.slice_bytes[i] * (thick ? (uint64_t{tail_depth_blocks} << b256d) : 1u);
        }
    }
    c.slice_stride = thick ? chain * c.bd : chain;
    c.total_bytes = chain * num_slices;
    c.ok = true;
    return c;
}

uint64_t chain_address(const ChainLayout& c, uint32_t mip, uint32_t x, uint32_t y, uint32_t z) noexcept {
    if (c.linear) {
        return c.slice_stride * z + c.offset[mip] +
               (uint64_t{y} * c.pitch[mip] + x) * c.fmt.bytes_per_block;
    }
    const TileShape& s = c.shape;
    const uint32_t tx = x + c.tail_x[mip];
    const uint32_t ty = y + c.tail_y[mip];
    uint32_t intra = 0;
    if (s.eq != nullptr) {
        intra = apply_equation(s.eq, s.blk_size_log2, tx, ty, s.thick_3d ? z : 0u);
    } else {
        intra = deposit_bits((tx & (c.bw - 1u)) << c.bpp_log2, s.x_mask) | deposit_bits(ty & (c.bh - 1u), s.y_mask);
    }
    if (s.xor_mode) {
        intra = apply_gfx10_xor_64k(intra);
    }
    const uint64_t slice = s.thick_3d ? z / c.bd : z;
    const uint64_t blk_index = c.in_tail[mip] ? 0u : uint64_t{y / c.bh} * (c.pitch[mip] / c.bw) + x / c.bw;
    return c.slice_stride * slice + c.macro_offset[mip] + (blk_index << s.blk_size_log2) + intra;
}

// Linear layouts outside the chain code (D3D12 UNKNOWN / ROW_MAJOR, or an explicit pitch).
bool is_linear_layout(const TextureLayoutDesc& desc) noexcept {
    return classify_mode(decode_d3d12_layout(desc.d3d12_layout)).family == SwizzleFamily::Linear &&
           !is_console_linear(desc);
}

}  // namespace

MipPlacement mip_placement(const TextureLayoutDesc& desc, uint32_t mip) noexcept {
    MipPlacement p{};
    if (is_linear_layout(desc)) {
        if (mip != 0u) {
            p.error = "linear mip chains are not modelled (mip 0 only)";
            return p;
        }
        p.surface_bytes = uint64_t{tiled_footprint_bytes(desc, 0u)} * std::max(desc.array_size, 1u);
        p.ok = p.surface_bytes != 0u;
        p.error = p.ok ? nullptr : "unsupported linear subresource";
        return p;
    }
    const ChainLayout c = build_chain(desc);
    if (!c.ok || mip >= c.mips) {
        p.error = c.ok ? "mip level out of range" : c.error;
        return p;
    }
    p.surface_bytes = c.total_bytes;
    p.macro_block_offset = c.macro_offset[mip];
    p.in_tail = c.in_tail[mip];
    p.tail_x = c.tail_x[mip];
    p.tail_y = c.tail_y[mip];
    p.pitch_elements = c.pitch[mip];
    p.height_elements = c.height[mip];
    p.slice_bytes = c.slice_bytes[mip];
    p.size_bytes = c.size_bytes[mip];
    p.offset_bytes = c.offset[mip];
    p.block_bytes = c.linear ? kConsoleLinearAlignment : uint64_t{1} << c.shape.blk_size_log2;
    p.ok = true;
    return p;
}

TileMap tile_map(const TextureLayoutDesc& desc) noexcept {
    TileMap t{};
    if (is_linear_layout(desc)) {
        return t;
    }
    const ChainLayout c = build_chain(desc);
    if (!c.ok) {
        return t;
    }
    t.mips = std::min(c.mips, kMaxTileMapMips);
    t.block_bytes = c.linear ? kConsoleLinearAlignment : uint64_t{1} << c.shape.blk_size_log2;
    t.standard_mips = 0;
    while (t.standard_mips < t.mips && !c.in_tail[t.standard_mips]) {
        t.start_tile[t.standard_mips] = c.macro_offset[t.standard_mips] / t.block_bytes;
        ++t.standard_mips;
    }
    t.packed_mips = t.mips - t.standard_mips;
    t.tail_tiles = t.packed_mips != 0u ? 1u : 0u;
    t.tail_start_tile = t.packed_mips != 0u ? c.macro_offset[t.standard_mips] / t.block_bytes : 0u;
    t.total_tiles = c.total_bytes / t.block_bytes;
    t.slice_tiles = (c.shape.thick_3d || desc.array_size <= 1u) ? t.total_tiles : t.total_tiles / desc.array_size;
    t.ok = true;
    return t;
}

uint64_t surface_element_offset(const TextureLayoutDesc& desc, uint32_t mip, uint32_t x, uint32_t y,
                                uint32_t z) noexcept {
    if (is_linear_layout(desc)) {
        return UINT64_MAX;
    }
    const ChainLayout c = build_chain(desc);
    if (!c.ok || mip >= c.mips) {
        return UINT64_MAX;
    }
    const uint32_t mip_w = std::max(1u, desc.width >> std::min(mip, 31u));
    const uint32_t mip_h = std::max(1u, desc.height >> std::min(mip, 31u));
    const uint32_t mip_d = std::max(1u, desc.depth >> std::min(mip, 31u));
    const uint32_t bx = (mip_w + c.fmt.block_width - 1u) / c.fmt.block_width;
    const uint32_t by = (mip_h + c.fmt.block_height - 1u) / c.fmt.block_height;
    const uint32_t nz = desc.depth > 1u ? mip_d : std::max(desc.array_size, 1u);
    if (x >= bx || y >= by || z >= nz) {
        return UINT64_MAX;
    }
    return chain_address(c, mip, x, y, z);
}

size_t surface_bytes(const TextureLayoutDesc& desc) noexcept {
    const MipPlacement p = mip_placement(desc, 0u);
    return p.ok ? static_cast<size_t>(p.surface_bytes) : 0u;
}

namespace {

constexpr uint32_t kAllSlices = 0xFFFFFFFFu;

struct PreparedSurfaceMip {
    ChainLayout c{};
    uint32_t z0 = 0;
    uint32_t nz = 0;
    uint32_t bx = 0;
    uint32_t by = 0;
    uint32_t bpe = 0;
    uint32_t row_pitch = 0;
    size_t linear_size = 0;
};

const char* prepare_surface_mip(const TextureLayoutDesc& desc, uint32_t mip, uint32_t array_slice,
                                size_t surface_size, PreparedSurfaceMip* out) noexcept {
    out->c = build_chain(desc);
    if (!out->c.ok) {
        return out->c.error;
    }
    if (mip >= out->c.mips) {
        return "mip level out of range";
    }
    if (surface_size < out->c.total_bytes) {
        return "source buffer truncated for surface footprint";
    }
    const uint32_t mip_w = std::max(1u, desc.width >> std::min(mip, 31u));
    const uint32_t mip_h = std::max(1u, desc.height >> std::min(mip, 31u));
    const uint32_t mip_d = std::max(1u, desc.depth >> std::min(mip, 31u));
    const uint32_t layers = std::max(desc.array_size, 1u);
    uint32_t z0 = 0;
    uint32_t nz = 0;
    if (desc.depth > 1u) {
        if (array_slice != 0u && array_slice != kAllSlices) {
            return "3D textures have no array slices";
        }
        nz = mip_d;
    } else if (array_slice == kAllSlices) {
        nz = layers;
    } else {
        if (array_slice >= layers) {
            return "array slice out of range";
        }
        z0 = array_slice;
        nz = 1;
    }
    const uint32_t bx = (mip_w + out->c.fmt.block_width - 1u) / out->c.fmt.block_width;
    const uint32_t by = (mip_h + out->c.fmt.block_height - 1u) / out->c.fmt.block_height;
    const uint32_t bpe = out->c.fmt.bytes_per_block;
    out->z0 = z0;
    out->nz = nz;
    out->bx = bx;
    out->by = by;
    out->bpe = bpe;
    out->row_pitch = bx * bpe;
    out->linear_size = size_t{out->row_pitch} * by * nz;

    if (out->c.linear && by > 0u && out->row_pitch > 0u) {
        const uint64_t src_row_pitch = uint64_t{out->c.pitch[mip]} * bpe;
        uint64_t last_row_off = 0;
        if (!checked_mul_u64(uint64_t{by - 1u}, src_row_pitch, &last_row_off)) {
            return "tiled source offset out of bounds";
        }
        for (uint32_t z = 0; z < nz; ++z) {
            uint64_t slice_off = 0;
            if (!checked_mul_u64(out->c.slice_stride, uint64_t{z0} + z, &slice_off) ||
                out->c.offset[mip] > std::numeric_limits<uint64_t>::max() - slice_off) {
                return "tiled source offset out of bounds";
            }
            const uint64_t slice_base = slice_off + out->c.offset[mip];
            if (slice_base > surface_size || last_row_off > surface_size - slice_base ||
                out->row_pitch > surface_size - slice_base - last_row_off) {
                return "tiled source offset out of bounds";
            }
        }
    }
    return nullptr;
}

// Shared walk for the surface detile/tile functions: calls fn(linear_offset, surface_offset,
// bytes) for mip `mip` of a prepared surface walk. Console-linear surfaces (layout 0x100) copy
// whole slices or rows instead of walking per element; tiled surfaces use the per-axis split of
// chain_address so column/row contributions are computed once per column/row.
template <typename Fn>
const char* walk_surface_mip(const PreparedSurfaceMip& p, uint32_t mip, size_t surface_size, Fn&& fn) {
    const ChainLayout& c = p.c;
    if (c.linear) {
        const uint64_t src_row_pitch = uint64_t{c.pitch[mip]} * p.bpe;
        const size_t copy_row_bytes = p.row_pitch;
        if (p.by == 0u || copy_row_bytes == 0u) {
            return nullptr;
        }
        const size_t slice_linear_bytes = size_t{p.by} * copy_row_bytes;
        for (uint32_t z = 0; z < p.nz; ++z) {
            const uint64_t slice_base = c.slice_stride * (uint64_t{p.z0} + z) + c.offset[mip];
            if (src_row_pitch == copy_row_bytes) {
                fn(size_t{z} * slice_linear_bytes, static_cast<size_t>(slice_base), slice_linear_bytes);
            } else {
                for (uint32_t y = 0; y < p.by; ++y) {
                    fn((size_t{z} * p.by + y) * copy_row_bytes,
                       static_cast<size_t>(slice_base + uint64_t{y} * src_row_pitch), copy_row_bytes);
                }
            }
        }
        return nullptr;
    }
    // chain_address, split by axis: every bit of the block equation (or _R mask) takes one
    // coordinate bit, so the x part of the block-local address and of the block index is computed
    // once per column and the y / z part once per row; the XOR swizzle is applied to their union
    // per element as chain_address applies it. Same addresses, without the per-element equation.
    const TileShape& s = c.shape;
    std::vector<uint32_t> col_intra(p.bx);
    std::vector<uint64_t> col_block(p.bx);
    for (uint32_t x = 0; x < p.bx; ++x) {
        const uint32_t tx = x + c.tail_x[mip];
        col_intra[x] = s.eq != nullptr ? apply_equation(s.eq, s.blk_size_log2, tx, 0u, 0u)
                                       : deposit_bits((tx & (c.bw - 1u)) << c.bpp_log2, s.x_mask);
        col_block[x] = c.in_tail[mip] ? 0u : uint64_t{x / c.bw} << s.blk_size_log2;
    }
    for (uint32_t z = 0; z < p.nz; ++z) {
        const uint32_t zz = p.z0 + z;
        for (uint32_t y = 0; y < p.by; ++y) {
            const size_t lin_row = (size_t{z} * p.by + y) * p.row_pitch;
            const uint32_t ty = y + c.tail_y[mip];
            const uint32_t row_intra = s.eq != nullptr
                                           ? apply_equation(s.eq, s.blk_size_log2, 0u, ty, s.thick_3d ? zz : 0u)
                                           : deposit_bits(ty & (c.bh - 1u), s.y_mask);
            const uint64_t slice = s.thick_3d ? zz / c.bd : zz;
            const uint64_t row_block =
                c.in_tail[mip] ? 0u : (uint64_t{y / c.bh} * (c.pitch[mip] / c.bw)) << s.blk_size_log2;
            const uint64_t row = c.slice_stride * slice + c.macro_offset[mip] + row_block;
            for (uint32_t x = 0; x < p.bx; ++x) {
                uint32_t intra = col_intra[x] | row_intra;
                if (s.xor_mode) {
                    intra = apply_gfx10_xor_64k(intra);
                }
                const uint64_t addr = row + col_block[x] + intra;
                if (addr > surface_size || p.bpe > surface_size - addr) {
                    return "tiled source offset out of bounds";
                }
                fn(lin_row + size_t{x} * p.bpe, static_cast<size_t>(addr), size_t{p.bpe});
            }
        }
    }
    return nullptr;
}

// One element (or row/slice) copy; the element sizes of the supported formats get a fixed-size
// copy the compiler inlines instead of a memcpy call per element.
inline void copy_element(uint8_t* dst, const uint8_t* src, size_t n) noexcept {
    switch (n) {
        case 1: std::memcpy(dst, src, 1); break;
        case 2: std::memcpy(dst, src, 2); break;
        case 4: std::memcpy(dst, src, 4); break;
        case 8: std::memcpy(dst, src, 8); break;
        case 16: std::memcpy(dst, src, 16); break;
        default: std::memcpy(dst, src, n); break;
    }
}

// Linear layouts: mip 0 at the base. Array slices only where every candidate linear rule gives
// the same slice stride (one mip; rows a multiple of 256 B, or an explicit pitch).
DetileResult detile_linear(const TextureLayoutDesc& desc, uint32_t mip, uint32_t array_slice, const uint8_t* surface,
                           size_t surface_size, std::vector<uint8_t>& linear_out) {
    DetileResult res{};
    linear_out.clear();
    if (mip != 0u) {
        res.error = "linear mip chains are not modelled (mip 0 only)";
        return res;
    }
    const uint32_t layers = std::max(desc.array_size, 1u);
    const bool all = array_slice == kAllSlices;
    if (!all && array_slice >= layers) {
        res.error = "array slice out of range";
        return res;
    }
    if (layers == 1u || (!all && array_slice == 0u)) {
        return detile_subresource(desc, 0u, surface, surface_size, linear_out);
    }
    const SubresourcePlan plan = plan_subresource(desc, 0u);
    if (!plan.ok) {
        res.error = plan.error;
        return res;
    }
    if (desc.depth > 1u || std::max(desc.mip_levels, 1u) > 1u ||
        (desc.pitch_bytes == 0u && (plan.linear_row_pitch % 256u) != 0u)) {
        res.error = "linear array slice stride is not modelled";
        return res;
    }
    const size_t stride = plan.tiled_total_bytes;
    const uint32_t first = all ? 0u : array_slice;
    const uint32_t count = all ? layers : 1u;
    std::vector<uint8_t> one;
    for (uint32_t s = first; s < first + count; ++s) {
        const size_t off = stride * s;
        if (surface == nullptr || off > surface_size) {
            linear_out.clear();
            res = DetileResult{};
            res.error = "source buffer truncated for surface footprint";
            return res;
        }
        res = detile_subresource(desc, 0u, surface + off, surface_size - off, one);
        if (!res.ok) {
            linear_out.clear();
            return res;
        }
        linear_out.insert(linear_out.end(), one.begin(), one.end());
    }
    res.slices = count;
    res.tiled_bytes_consumed = stride * (first + count);
    return res;
}

DetileResult detile_surface_impl(const TextureLayoutDesc& desc, uint32_t mip, uint32_t array_slice,
                                 const uint8_t* surface, size_t surface_size, std::vector<uint8_t>& linear_out) {
    if (is_linear_layout(desc)) {
        return detile_linear(desc, mip, array_slice, surface, surface_size, linear_out);
    }
    DetileResult res{};
    linear_out.clear();
    if (surface == nullptr) {
        res.error = "null source pointer";
        return res;
    }
    PreparedSurfaceMip p{};
    const char* err = prepare_surface_mip(desc, mip, array_slice, surface_size, &p);
    if (err != nullptr) {
        linear_out.clear();
        res.error = err;
        return res;
    }
    // One walk validates and copies (console-linear bounds are already checked per slice in
    // prepare_surface_mip; on a tiled out-of-bounds element, partial output is dropped).
    linear_out.resize(p.linear_size);
    uint8_t* const out = linear_out.data();
    err = walk_surface_mip(p, mip, surface_size,
                           [&](size_t lin, size_t src, size_t n) { copy_element(out + lin, surface + src, n); });
    if (err != nullptr) {
        linear_out.clear();
        res.error = err;
        return res;
    }
    res.ok = true;
    res.row_pitch_bytes = p.row_pitch;
    res.rows = p.by;
    res.slices = p.nz;
    res.tiled_bytes_consumed = static_cast<size_t>(p.c.total_bytes);
    return res;
}

bool tile_surface_impl(const TextureLayoutDesc& desc, uint32_t mip, uint32_t array_slice, const uint8_t* linear_src,
                       size_t linear_src_bytes, uint8_t* surface, size_t surface_size) {
    if (linear_src == nullptr || surface == nullptr || is_linear_layout(desc)) {
        return false;
    }
    PreparedSurfaceMip p{};
    const char* err = prepare_surface_mip(desc, mip, array_slice, surface_size, &p);
    if (err != nullptr || linear_src_bytes < p.linear_size) {
        return false;
    }
    if (!p.c.linear) {
        err = walk_surface_mip(p, mip, surface_size, [](size_t, size_t, size_t) {});
        if (err != nullptr) {
            return false;
        }
    }
    walk_surface_mip(p, mip, surface_size,
                     [&](size_t lin, size_t dst, size_t n) { copy_element(surface + dst, linear_src + lin, n); });
    return true;
}

}  // namespace

DetileResult detile_from_surface(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* surface,
                                 size_t surface_size, std::vector<uint8_t>& linear_out) {
    return detile_surface_impl(desc, mip, kAllSlices, surface, surface_size, linear_out);
}

DetileResult detile_surface_subresource(const TextureLayoutDesc& desc, uint32_t mip, uint32_t array_slice,
                                        const uint8_t* surface, size_t surface_size,
                                        std::vector<uint8_t>& linear_out) {
    if (array_slice == kAllSlices) {
        linear_out.clear();
        DetileResult r{};
        r.error = "array slice out of range";
        return r;
    }
    return detile_surface_impl(desc, mip, array_slice, surface, surface_size, linear_out);
}

bool tile_into_surface(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* linear_src,
                       size_t linear_src_bytes, uint8_t* surface, size_t surface_size) {
    return tile_surface_impl(desc, mip, kAllSlices, linear_src, linear_src_bytes, surface, surface_size);
}

bool tile_surface_subresource(const TextureLayoutDesc& desc, uint32_t mip, uint32_t array_slice,
                              const uint8_t* linear_src, size_t linear_src_bytes, uint8_t* surface,
                              size_t surface_size) {
    return array_slice != kAllSlices &&
           tile_surface_impl(desc, mip, array_slice, linear_src, linear_src_bytes, surface, surface_size);
}

bool tile_into_surface_for_test(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* linear_src,
                                size_t linear_src_bytes, std::vector<uint8_t>& surface) {
    const size_t need = surface_bytes(desc);
    if (need == 0u || linear_src == nullptr) {
        return false;
    }
    if (surface.size() < need) {
        surface.resize(need, 0u);
    }
    if (is_linear_layout(desc)) {
        std::vector<uint8_t> tiled;
        if (mip != 0u || !tile_subresource_for_test(desc, 0u, linear_src, linear_src_bytes, tiled)) {
            return false;
        }
        std::memcpy(surface.data(), tiled.data(), tiled.size());
        return true;
    }
    return tile_surface_impl(desc, mip, kAllSlices, linear_src, linear_src_bytes, surface.data(), surface.size());
}

}  // namespace playgta6::gpu::gfx10

