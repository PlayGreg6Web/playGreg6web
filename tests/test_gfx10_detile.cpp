// Synthetic unit tests for the AMD GFX10 / RDNA2 texture detiler (runtime/gpu/gfx10_detile.h).
// Zero copyrighted game data: all inputs are deterministically generated synthetic patterns.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "gpu/gfx10_detile.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_gfx10_detile.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu::gfx10;

// Deterministic synthetic byte generator (distinct per block/byte position).
std::vector<uint8_t> make_synthetic_linear(size_t size, uint32_t seed) {
    std::vector<uint8_t> data(size);
    uint32_t x = seed ^ 0x9E3779B9u;
    for (size_t i = 0; i < size; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        data[i] = static_cast<uint8_t>((x ^ static_cast<uint32_t>(i * 131u)) & 0xFFu);
    }
    return data;
}

void verify_round_trip(const TextureLayoutDesc& desc, uint32_t mip, uint32_t seed, bool expect_scrambled = true) {
    const FormatGeometry fmt = format_geometry(desc.dxgi_format);
    CHECK(fmt.supported);

    const uint32_t mip_w = mip >= 32u ? 1u : (desc.width >> mip ? desc.width >> mip : 1u);
    const uint32_t mip_h = mip >= 32u ? 1u : (desc.height >> mip ? desc.height >> mip : 1u);
    const uint32_t mip_d = mip >= 32u ? 1u : (desc.depth >> mip ? desc.depth >> mip : 1u);
    const uint32_t bx = (mip_w + fmt.block_width - 1u) / fmt.block_width;
    const uint32_t by = (mip_h + fmt.block_height - 1u) / fmt.block_height;
    const uint32_t expected_row_pitch = bx * fmt.bytes_per_block;
    const size_t linear_size = static_cast<size_t>(expected_row_pitch) * by * mip_d;

    const std::vector<uint8_t> linear_in = make_synthetic_linear(linear_size, seed);
    std::vector<uint8_t> tiled;
    CHECK(tile_subresource_for_test(desc, mip, linear_in.data(), linear_in.size(), tiled));
    CHECK(!tiled.empty());

    if (expect_scrambled && linear_size >= static_cast<size_t>(fmt.bytes_per_block) * 4u) {
        CHECK(tiled != linear_in);
    }

    std::vector<uint8_t> linear_out;
    const DetileResult res = detile_subresource(desc, mip, tiled.data(), tiled.size(), linear_out);
    CHECK(res.ok);
    CHECK(res.error == nullptr);
    CHECK(res.row_pitch_bytes == expected_row_pitch);
    CHECK(res.rows == by);
    CHECK(res.slices == mip_d);
    CHECK(res.tiled_bytes_consumed == tiled.size());
    CHECK(linear_out.size() == linear_in.size());
    CHECK(linear_out == linear_in);

    // Extra trailing bytes in src buffer must be safely ignored.
    std::vector<uint8_t> padded_tiled = tiled;
    padded_tiled.resize(tiled.size() + 37u, 0xCCu);
    std::vector<uint8_t> linear_out2;
    const DetileResult res2 = detile_subresource(desc, mip, padded_tiled.data(), padded_tiled.size(), linear_out2);
    CHECK(res2.ok);
    CHECK(res2.tiled_bytes_consumed == tiled.size());
    CHECK(linear_out2 == linear_in);
}

void test_decode_d3d12_layout() {
    CHECK(decode_d3d12_layout(0u) == SwizzleMode::Linear);
    CHECK(decode_d3d12_layout(1u) == SwizzleMode::Linear);
    CHECK(decode_d3d12_layout(0x100u) == SwizzleMode::Linear);
    CHECK(decode_d3d12_layout(0x101u) == SwizzleMode::Sw256BS);
    CHECK(decode_d3d12_layout(0x102u) == SwizzleMode::Sw256BD);
    CHECK(decode_d3d12_layout(0x103u) == SwizzleMode::Sw256BR);
    CHECK(decode_d3d12_layout(0x105u) == SwizzleMode::Sw4KBS);
    CHECK(decode_d3d12_layout(0x106u) == SwizzleMode::Sw4KBD);
    CHECK(decode_d3d12_layout(0x107u) == SwizzleMode::Sw4KBR);
    CHECK(decode_d3d12_layout(0x109u) == SwizzleMode::Sw64KBS);
    CHECK(decode_d3d12_layout(0x10Au) == SwizzleMode::Sw64KBD);
    CHECK(decode_d3d12_layout(0x10Bu) == SwizzleMode::Sw64KBR);
    CHECK(decode_d3d12_layout(0x10Du) == SwizzleMode::SwVarS);
    CHECK(decode_d3d12_layout(0x10Eu) == SwizzleMode::SwVarD);
    CHECK(decode_d3d12_layout(0x10Fu) == SwizzleMode::SwVarR);
    CHECK(decode_d3d12_layout(0x119u) == SwizzleMode::Sw64KBSX);
    CHECK(decode_d3d12_layout(0x11Au) == SwizzleMode::Sw64KBDX);
    CHECK(decode_d3d12_layout(0x11Bu) == SwizzleMode::Sw64KBRX);
    CHECK(decode_d3d12_layout(0x11Du) == SwizzleMode::SwVarSX);
    CHECK(decode_d3d12_layout(0x11Eu) == SwizzleMode::SwVarDX);
    CHECK(decode_d3d12_layout(0x11Fu) == SwizzleMode::SwVarRX);

    // Unrecognized / unsupported layouts must never guess.
    CHECK(decode_d3d12_layout(2u) == SwizzleMode::Unsupported);
    CHECK(decode_d3d12_layout(3u) == SwizzleMode::Unsupported);
    CHECK(decode_d3d12_layout(0x104u) == SwizzleMode::Unsupported);
    CHECK(decode_d3d12_layout(0x108u) == SwizzleMode::Unsupported);
    CHECK(decode_d3d12_layout(0x110u) == SwizzleMode::Unsupported);
    CHECK(decode_d3d12_layout(0x118u) == SwizzleMode::Unsupported);
    CHECK(decode_d3d12_layout(0x120u) == SwizzleMode::Unsupported);
    CHECK(decode_d3d12_layout(0x200u) == SwizzleMode::Unsupported);
    CHECK(decode_d3d12_layout(0xFFFFFFFFu) == SwizzleMode::Unsupported);

    CHECK(is_swizzle_supported(SwizzleMode::Linear));
    CHECK(is_swizzle_supported(SwizzleMode::Sw256BD));
    CHECK(is_swizzle_supported(SwizzleMode::Sw64KBDX));
    CHECK(is_swizzle_supported(SwizzleMode::SwVarRX));
    CHECK(!is_swizzle_supported(SwizzleMode::Unsupported));
}

void test_format_geometry() {
    auto check_fmt = [](uint32_t dxgi, uint32_t bw, uint32_t bh, uint32_t bpb, bool comp) {
        const FormatGeometry g = format_geometry(dxgi);
        CHECK(g.supported);
        CHECK(g.block_width == bw);
        CHECK(g.block_height == bh);
        CHECK(g.bytes_per_block == bpb);
        CHECK(g.compressed == comp);
    };

    check_fmt(2u, 1, 1, 16, false);   // R32G32B32A32_FLOAT
    check_fmt(3u, 1, 1, 16, false);   // R32G32B32A32_UINT
    check_fmt(10u, 1, 1, 8, false);   // R16G16B16A16_FLOAT
    check_fmt(11u, 1, 1, 8, false);   // R16G16B16A16_UNORM
    check_fmt(16u, 1, 1, 8, false);   // R32G32_FLOAT
    check_fmt(24u, 1, 1, 4, false);   // R10G10B10A2_UNORM
    check_fmt(26u, 1, 1, 4, false);   // R11G11B10_FLOAT
    check_fmt(27u, 1, 1, 4, false);   // R8G8B8A8_TYPELESS
    check_fmt(28u, 1, 1, 4, false);   // R8G8B8A8_UNORM
    check_fmt(29u, 1, 1, 4, false);   // R8G8B8A8_UNORM_SRGB
    check_fmt(34u, 1, 1, 4, false);   // R16G16_FLOAT
    check_fmt(41u, 1, 1, 4, false);   // R32_FLOAT
    check_fmt(42u, 1, 1, 4, false);   // R32_UINT
    check_fmt(61u, 1, 1, 1, false);   // R8_UNORM
    check_fmt(65u, 1, 1, 1, false);   // A8_UNORM
    check_fmt(87u, 1, 1, 4, false);   // B8G8R8A8_UNORM
    check_fmt(91u, 1, 1, 4, false);   // B8G8R8A8_UNORM_SRGB

    check_fmt(70u, 4, 4, 8, true);    // BC1_TYPELESS
    check_fmt(71u, 4, 4, 8, true);    // BC1_UNORM
    check_fmt(72u, 4, 4, 8, true);    // BC1_UNORM_SRGB
    check_fmt(74u, 4, 4, 16, true);   // BC2_UNORM
    check_fmt(77u, 4, 4, 16, true);   // BC3_UNORM
    check_fmt(80u, 4, 4, 8, true);    // BC4_UNORM
    check_fmt(83u, 4, 4, 16, true);   // BC5_UNORM
    check_fmt(95u, 4, 4, 16, true);   // BC6H_UF16
    check_fmt(96u, 4, 4, 16, true);   // BC6H_SF16
    check_fmt(97u, 4, 4, 16, true);   // BC7_TYPELESS
    check_fmt(98u, 4, 4, 16, true);   // BC7_UNORM
    check_fmt(99u, 4, 4, 16, true);   // BC7_UNORM_SRGB

    // Unsupported formats must report supported = false.
    CHECK(!format_geometry(0u).supported);     // UNKNOWN
    CHECK(!format_geometry(6u).supported);     // R32G32B32_FLOAT (12B non-POT)
    CHECK(!format_geometry(103u).supported);   // NV12 planar video
    CHECK(!format_geometry(9999u).supported);  // Bogus
}

// Verify bijectivity of tiled_byte_offset_2d over a complete single tile across all supported modes & bpps.
void test_tile_bijectivity_2d() {
    const SwizzleMode modes[] = {
        SwizzleMode::Sw256BS,  SwizzleMode::Sw256BD,  SwizzleMode::Sw256BR,
        SwizzleMode::Sw4KBS,   SwizzleMode::Sw4KBD,   SwizzleMode::Sw4KBR,
        SwizzleMode::Sw64KBS,  SwizzleMode::Sw64KBD,  SwizzleMode::Sw64KBR,
        SwizzleMode::SwVarS,   SwizzleMode::SwVarD,   SwizzleMode::SwVarR,
        SwizzleMode::Sw64KBSX, SwizzleMode::Sw64KBDX, SwizzleMode::Sw64KBRX,
        SwizzleMode::SwVarSX,  SwizzleMode::SwVarDX,  SwizzleMode::SwVarRX,
    };
    const uint32_t bpps[5] = {1u, 2u, 4u, 8u, 16u};
    const uint32_t w256[5] = {16u, 16u, 8u, 8u, 4u};
    const uint32_t h256[5] = {16u, 8u, 8u, 4u, 4u};

    for (SwizzleMode m : modes) {
        const uint8_t raw = static_cast<uint8_t>(m);
        const uint32_t scale = (raw <= 3u) ? 1u : (raw <= 7u ? 4u : 16u);
        const size_t tile_bytes = (raw <= 3u) ? 256u : (raw <= 7u ? 4096u : 65536u);
        for (uint32_t bi = 0; bi < 5u; ++bi) {
            const uint32_t bpp = bpps[bi];
            const uint32_t tw = w256[bi] * scale;
            const uint32_t th = h256[bi] * scale;
            const size_t elem_count = static_cast<size_t>(tw) * th;
            std::vector<uint8_t> seen(elem_count, 0u);
            for (uint32_t y = 0; y < th; ++y) {
                for (uint32_t x = 0; x < tw; ++x) {
                    const size_t off = tiled_byte_offset_2d(m, x, y, bpp, tw);
                    CHECK(off < tile_bytes);
                    CHECK((off % bpp) == 0u);
                    const size_t slot = off / bpp;
                    CHECK(seen[slot] == 0u);
                    seen[slot] = 1u;
                }
            }
            for (uint8_t s : seen) {
                CHECK(s == 1u);
            }
        }
    }

    // Invalid inputs return SIZE_MAX.
    CHECK(tiled_byte_offset_2d(SwizzleMode::Unsupported, 0, 0, 4, 16) == std::numeric_limits<size_t>::max());
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 0, 3, 16) == std::numeric_limits<size_t>::max());
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 0, 0, 16) == std::numeric_limits<size_t>::max());
}

void test_r32g32b32a32_2x2_sw256bd() {
    // Observed Xbox Series small LUT / constant texture: R32G32B32A32_FLOAT 2x2 with layout 0x102 (Sw256BD).
    const TextureLayoutDesc desc{
        /*dxgi_format=*/2u,
        /*width=*/2u,
        /*height=*/2u,
        /*depth=*/1u,
        /*mip_levels=*/1u,
        /*d3d12_layout=*/0x102u,
        /*pitch_bytes=*/0u,
    };
    // Exact intra-tile offsets for 16B Sw256BD (Morton 2x2 in 64B cache line):
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 0, 16, 2) == 0u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 1, 0, 16, 2) == 16u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 1, 16, 2) == 32u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 1, 1, 16, 2) == 48u);

    // For a 2x2 16B texture, (0,0)->0, (1,0)->16, (0,1)->32, (1,1)->48 matches linear order in first 64B,
    // while the tiled buffer is padded to the full 256B tile.
    verify_round_trip(desc, 0u, 1001u, /*expect_scrambled=*/true);
}

void test_required_formats_and_modes() {
    // R16G16B16A16_FLOAT 64x64 across 256B/4KB/64KB/Var and XOR modes.
    const uint32_t layouts[] = {
        0x101u, 0x102u, 0x103u, 0x105u, 0x106u, 0x107u,
        0x109u, 0x10Au, 0x10Bu, 0x10Du, 0x10Eu, 0x10Fu,
        0x119u, 0x11Au, 0x11Bu, 0x11Du, 0x11Eu, 0x11Fu,
    };
    uint32_t seed = 2000u;
    for (uint32_t layout : layouts) {
        verify_round_trip({10u, 64u, 64u, 1u, 1u, layout, 0u}, 0u, seed++);
    }

    // R16G16_FLOAT 32x16 across representative modes.
    verify_round_trip({34u, 32u, 16u, 1u, 1u, 0x102u, 0u}, 0u, 3001u);
    verify_round_trip({34u, 32u, 16u, 1u, 1u, 0x106u, 0u}, 0u, 3002u);
    verify_round_trip({34u, 32u, 16u, 1u, 1u, 0x10Au, 0u}, 0u, 3003u);
    verify_round_trip({34u, 32u, 16u, 1u, 1u, 0x11Au, 0u}, 0u, 3004u);

    // R8G8B8A8_UNORM 128x72 (NPOT height).
    verify_round_trip({28u, 128u, 72u, 1u, 1u, 0x102u, 0u}, 0u, 4001u);
    verify_round_trip({28u, 128u, 72u, 1u, 1u, 0x105u, 0u}, 0u, 4002u);
    verify_round_trip({28u, 128u, 72u, 1u, 1u, 0x10Au, 0u}, 0u, 4003u);
    verify_round_trip({28u, 128u, 72u, 1u, 1u, 0x11Bu, 0u}, 0u, 4004u);

    // BC7_UNORM 64x64 and 100x60 (NPOT block-compressed).
    verify_round_trip({98u, 64u, 64u, 1u, 1u, 0x102u, 0u}, 0u, 5001u);
    verify_round_trip({98u, 64u, 64u, 1u, 1u, 0x10Au, 0u}, 0u, 5002u);
    verify_round_trip({98u, 64u, 64u, 1u, 1u, 0x11Au, 0u}, 0u, 5003u);
    verify_round_trip({98u, 100u, 60u, 1u, 1u, 0x102u, 0u}, 0u, 5004u);
    verify_round_trip({98u, 100u, 60u, 1u, 1u, 0x106u, 0u}, 0u, 5005u);
    verify_round_trip({98u, 100u, 60u, 1u, 1u, 0x109u, 0u}, 0u, 5006u);
    verify_round_trip({98u, 100u, 60u, 1u, 1u, 0x119u, 0u}, 0u, 5007u);

    // BC1_UNORM (8B block) and R8_UNORM / A8_UNORM (1B) + mip level > 0.
    verify_round_trip({71u, 96u, 48u, 1u, 4u, 0x10Au, 0u}, 2u, 5100u);
    verify_round_trip({61u, 50u, 30u, 1u, 3u, 0x10Eu, 0u}, 1u, 5101u);
    verify_round_trip({65u, 64u, 64u, 1u, 1u, 0u, 0u}, 0u, 5200u, /*expect_scrambled=*/false);
    verify_round_trip({65u, 64u, 48u, 1u, 1u, 0x102u, 0u}, 0u, 5201u);
    verify_round_trip({65u, 64u, 48u, 1u, 1u, 0x106u, 0u}, 0u, 5202u);
    verify_round_trip({65u, 128u, 96u, 1u, 1u, 0x10Au, 0u}, 0u, 5203u);
    verify_round_trip({65u, 50u, 30u, 1u, 3u, 0x11Au, 0u}, 1u, 5204u);
}

void test_small_3d_16x16x4() {
    // Small 3D 16x16x4 in linear, 2D-sliced tiled modes (Sw256BD, Sw64KBD, Sw64KBDX),
    // and thick 3D Standard modes (Sw4KBS, Sw64KBS, Sw64KBSX).
    verify_round_trip({28u, 16u, 16u, 4u, 1u, 0u, 0u}, 0u, 6000u, /*expect_scrambled=*/false);
    verify_round_trip({28u, 16u, 16u, 4u, 1u, 0x102u, 0u}, 0u, 6001u);
    verify_round_trip({28u, 16u, 16u, 4u, 1u, 0x105u, 0u}, 0u, 6002u);
    verify_round_trip({28u, 16u, 16u, 4u, 1u, 0x109u, 0u}, 0u, 6003u);
    verify_round_trip({28u, 16u, 16u, 4u, 1u, 0x10Au, 0u}, 0u, 6004u);
    verify_round_trip({28u, 16u, 16u, 4u, 1u, 0x119u, 0u}, 0u, 6005u);
    verify_round_trip({10u, 16u, 16u, 4u, 3u, 0x109u, 0u}, 1u, 6006u);  // 8x8x2 mip 1
    verify_round_trip({61u, 16u, 16u, 4u, 1u, 0x105u, 0u}, 0u, 6007u);  // 1B thick 4KB
}

// Golden addresses from AMD addrlib's public gfx10 RB+ pattern tables (gfx10SwizzlePattern.h).
void test_addrlib_equations_2d() {
    // SW_256B_D, 4 B elements: 0 0 X0 X1 Y0 Y1 X2 Y2 (SW_256B_S ends Y2 X2 instead).
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 1, 1, 4, 8) == 4u + 16u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 2, 0, 4, 8) == 8u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 2, 4, 8) == 32u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 4, 0, 4, 8) == 64u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 4, 4, 8) == 128u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BS, 4, 0, 4, 8) == 128u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BS, 0, 4, 4, 8) == 64u);
    // SW_256B_D, 1 B: X0 X1 X2 Y1 Y0 Y2 X3 Y3.
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 1, 1, 16) == 16u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 2, 1, 16) == 8u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 8, 0, 1, 16) == 64u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 8, 1, 16) == 128u);
    // SW_256B_D, 8 B: 0 0 0 X0 Y0 X1 X2 Y1.
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 1, 0, 8, 8) == 8u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 1, 8, 8) == 16u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 2, 0, 8, 8) == 32u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 4, 0, 8, 8) == 64u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw256BD, 0, 2, 8, 8) == 128u);
    // SW_4KB_D, 16 B: 0 0 0 0 X0 Y0 X1 Y1 Y2 X2 Y3 X3.
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw4KBD, 0, 4, 16, 16) == 256u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw4KBD, 4, 0, 16, 16) == 512u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw4KBD, 0, 8, 16, 16) == 1024u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw4KBD, 8, 0, 16, 16) == 2048u);
    // SW_64KB_D, 4 B: 0 0 X0 X1 Y0 Y1 X2 Y2 Y3 X3 Y4 X4 Y5 X5 Y6 X6.
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw64KBD, 4, 4, 4, 128) == 64u + 128u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw64KBD, 0, 8, 4, 128) == 256u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw64KBD, 8, 0, 4, 128) == 512u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw64KBD, 0, 64, 4, 128) == 16384u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw64KBD, 64, 0, 4, 128) == 32768u);
    // SW_4KB_S, 4 B (unchanged): 0 0 X0 X1 Y0 Y1 Y2 X2 Y3 X3 Y4 X4.
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw4KBS, 0, 8, 4, 32) == 256u);
    CHECK(tiled_byte_offset_2d(SwizzleMode::Sw4KBS, 8, 0, 4, 32) == 512u);
}

// Thick 3D SW_4KB_S (addrlib GFX10_SW_4K_S3): tile a texture whose elements carry their own
// coordinates and check where they land.
void check_thick_position(uint32_t dxgi_format, uint32_t bpe, uint32_t w, uint32_t h, uint32_t d, uint32_t x,
                          uint32_t y, uint32_t z, size_t expected_offset) {
    const TextureLayoutDesc desc{dxgi_format, w, h, d, 1u, 0x105u, 0u};
    std::vector<uint8_t> linear(static_cast<size_t>(w) * h * d * bpe, 0u);
    const size_t index = (static_cast<size_t>(z) * h + y) * w + x;
    linear[index * bpe] = 0xA5u;
    std::vector<uint8_t> tiled;
    CHECK(tile_subresource_for_test(desc, 0u, linear.data(), linear.size(), tiled));
    CHECK(expected_offset < tiled.size());
    if (expected_offset < tiled.size()) {
        CHECK(tiled[expected_offset] == 0xA5u);
    }
}

void test_addrlib_equations_3d() {
    // 8 B elements, 4KB_S3: 0 0 0 X0 Z0 Y0 X1 Z1 Y1 X2 Z2 Y2 (8x8x8 block).
    check_thick_position(10u, 8u, 8u, 8u, 8u, 1u, 0u, 0u, 8u);
    check_thick_position(10u, 8u, 8u, 8u, 8u, 0u, 0u, 1u, 16u);
    check_thick_position(10u, 8u, 8u, 8u, 8u, 0u, 1u, 0u, 32u);
    check_thick_position(10u, 8u, 8u, 8u, 8u, 2u, 0u, 0u, 64u);
    check_thick_position(10u, 8u, 8u, 8u, 8u, 0u, 0u, 2u, 128u);
    check_thick_position(10u, 8u, 8u, 8u, 8u, 0u, 2u, 0u, 256u);
    check_thick_position(10u, 8u, 8u, 8u, 8u, 4u, 0u, 0u, 512u);
    check_thick_position(10u, 8u, 8u, 8u, 8u, 0u, 0u, 4u, 1024u);
    check_thick_position(10u, 8u, 8u, 8u, 8u, 0u, 4u, 0u, 2048u);
    // 4 B elements: 0 0 X0 Y0 Z0 Y1 X1 Z1 Y2 X2 Z2 Y3 (8x16x8 block).
    check_thick_position(28u, 4u, 8u, 16u, 8u, 0u, 8u, 0u, 2048u);
    check_thick_position(28u, 4u, 8u, 16u, 8u, 4u, 0u, 0u, 512u);
    check_thick_position(28u, 4u, 8u, 16u, 8u, 0u, 0u, 4u, 1024u);
    // 1 B elements: X0 X1 Z0 Y0 Z1 Y1 X2 Z2 Y2 X3 Z3 Y3 (16x16x16 block).
    check_thick_position(61u, 1u, 16u, 16u, 16u, 0u, 0u, 1u, 4u);
    check_thick_position(61u, 1u, 16u, 16u, 16u, 0u, 1u, 0u, 8u);
    check_thick_position(61u, 1u, 16u, 16u, 16u, 8u, 0u, 0u, 512u);

    // Footprints: whole thick blocks.
    CHECK(tiled_footprint_bytes({28u, 8u, 16u, 8u, 1u, 0x105u, 0u}, 0u) == 4096u);
    CHECK(tiled_footprint_bytes({28u, 16u, 8u, 8u, 1u, 0x105u, 0u}, 0u) == 8192u);
    CHECK(tiled_footprint_bytes({10u, 17u, 17u, 17u, 1u, 0x105u, 0u}, 0u) == 27u * 4096u);
    CHECK(tiled_footprint_bytes({10u, 33u, 33u, 33u, 1u, 0x105u, 0u}, 0u) == 125u * 4096u);
    CHECK(tiled_footprint_bytes({2u, 4u, 8u, 8u, 1u, 0x105u, 0u}, 0u) == 4096u);  // 16 B: 4x8x8
    CHECK(tiled_footprint_bytes({2u, 8u, 8u, 4u, 1u, 0x105u, 0u}, 0u) == 8192u);
    // 64KB_S3, 4 B: 32x32x16 block.
    CHECK(tiled_footprint_bytes({28u, 32u, 32u, 16u, 1u, 0x109u, 0u}, 0u) == 65536u);
    // 3D in _D modes stays 2D-sliced: one 256B_D block (8x8 at 4 B) per slice.
    CHECK(tiled_footprint_bytes({28u, 8u, 8u, 3u, 1u, 0x102u, 0u}, 0u) == 3u * 256u);
    CHECK(tiled_footprint_bytes({28u, 8u, 8u, 1u, 1u, 0x0FFu, 0u}, 0u) == 0u);
}

void test_linear_custom_pitch() {
    // R8G8B8A8_UNORM 20x10: tight row = 80 bytes, custom D3D12 row pitch = 256 bytes.
    const TextureLayoutDesc desc{
        /*dxgi_format=*/28u,
        /*width=*/20u,
        /*height=*/10u,
        /*depth=*/1u,
        /*mip_levels=*/1u,
        /*d3d12_layout=*/1u,  // D3D12_TEXTURE_LAYOUT_ROW_MAJOR
        /*pitch_bytes=*/256u,
    };
    verify_round_trip(desc, 0u, 7001u, /*expect_scrambled=*/true);

    // Verify padding bytes in custom pitch rows are ignored and do not leak into linear_out.
    const std::vector<uint8_t> linear_in = make_synthetic_linear(80u * 10u, 7002u);
    std::vector<uint8_t> pitched(256u * 10u, 0xEEu);
    for (uint32_t y = 0; y < 10u; ++y) {
        std::memcpy(pitched.data() + y * 256u, linear_in.data() + y * 80u, 80u);
    }
    std::vector<uint8_t> linear_out;
    const DetileResult res = detile_subresource(desc, 0u, pitched.data(), pitched.size(), linear_out);
    CHECK(res.ok);
    CHECK(res.row_pitch_bytes == 80u);
    CHECK(res.rows == 10u);
    CHECK(res.slices == 1u);
    CHECK(res.tiled_bytes_consumed == 2560u);
    CHECK(linear_out == linear_in);

    // Pitch smaller than tight row width must be rejected.
    TextureLayoutDesc bad_pitch = desc;
    bad_pitch.pitch_bytes = 64u;
    const DetileResult bad_res = detile_subresource(bad_pitch, 0u, pitched.data(), pitched.size(), linear_out);
    CHECK(!bad_res.ok);
    CHECK(bad_res.error != nullptr);
    CHECK(linear_out.empty());
}

void test_bounds_and_error_rejection() {
    // 1. Truncated source buffer rejection (even 1 byte short).
    const TextureLayoutDesc desc{28u, 32u, 32u, 1u, 1u, 0x102u, 0u};
    const std::vector<uint8_t> linear_in = make_synthetic_linear(32u * 32u * 4u, 8001u);
    std::vector<uint8_t> tiled;
    CHECK(tile_subresource_for_test(desc, 0u, linear_in.data(), linear_in.size(), tiled));

    std::vector<uint8_t> out;
    const DetileResult trunc1 = detile_subresource(desc, 0u, tiled.data(), tiled.size() - 1u, out);
    CHECK(!trunc1.ok);
    CHECK(trunc1.error != nullptr);
    CHECK(out.empty());

    const DetileResult trunc0 = detile_subresource(desc, 0u, tiled.data(), 0u, out);
    CHECK(!trunc0.ok);
    CHECK(trunc0.error != nullptr);

    const DetileResult null_src = detile_subresource(desc, 0u, nullptr, tiled.size(), out);
    CHECK(!null_src.ok);
    CHECK(null_src.error != nullptr);

    // 2. Unsupported layout rejection.
    const uint32_t bad_layouts[] = {2u, 3u, 0x104u, 0x110u, 0x120u, 0xDEADBEEFu};
    for (uint32_t bad_layout : bad_layouts) {
        TextureLayoutDesc bad = desc;
        bad.d3d12_layout = bad_layout;
        const DetileResult r = detile_subresource(bad, 0u, tiled.data(), tiled.size(), out);
        CHECK(!r.ok);
        CHECK(r.error != nullptr);
        CHECK(out.empty());
        CHECK(!tile_subresource_for_test(bad, 0u, linear_in.data(), linear_in.size(), tiled));
    }

    // 3. Unsupported format rejection.
    TextureLayoutDesc bad_fmt = desc;
    bad_fmt.dxgi_format = 6u;  // R32G32B32_FLOAT (12B)
    const DetileResult r_fmt = detile_subresource(bad_fmt, 0u, tiled.data(), tiled.size(), out);
    CHECK(!r_fmt.ok);
    CHECK(r_fmt.error != nullptr);

    // 4. Out-of-range mip & zero dimensions rejection.
    const DetileResult r_mip = detile_subresource(desc, 1u, tiled.data(), tiled.size(), out);
    CHECK(!r_mip.ok);
    CHECK(r_mip.error != nullptr);

    TextureLayoutDesc zero_w = desc;
    zero_w.width = 0u;
    const DetileResult r_zw = detile_subresource(zero_w, 0u, tiled.data(), tiled.size(), out);
    CHECK(!r_zw.ok);
    CHECK(r_zw.error != nullptr);
}

}  // namespace

int main() {
    test_decode_d3d12_layout();
    test_format_geometry();
    test_tile_bijectivity_2d();
    test_r32g32b32a32_2x2_sw256bd();
    test_required_formats_and_modes();
    test_small_3d_16x16x4();
    test_addrlib_equations_2d();
    test_addrlib_equations_3d();
    test_linear_custom_pitch();
    test_bounds_and_error_rejection();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_gfx10_detile: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_gfx10_detile: all checks passed\n");
    return 0;
}
