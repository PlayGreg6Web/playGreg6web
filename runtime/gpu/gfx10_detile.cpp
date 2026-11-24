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

}  // namespace playgta6::gpu::gfx10
