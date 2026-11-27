// Synthetic tests: D3D12 resource sizes for textures with a console layout value (#551).
// Verified GFX10 modes are sized by the whole-surface placement in gpu/gfx10_detile.h; other
// layouts keep the linear estimate.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_guest_resources.h"
#include "gpu/gfx10_detile.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_console_layout_sizes.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace dg = playgta6::gpu::d3d12_guest;
namespace gfx10 = playgta6::gpu::gfx10;

dg::GuestResourceDesc texture(uint32_t dimension, uint64_t w, uint32_t h, uint16_t d, uint16_t mips, uint32_t format,
                              uint32_t layout) {
    dg::GuestResourceDesc t{};
    t.Dimension = dimension;
    t.Width = w;
    t.Height = h;
    t.DepthOrArraySize = d;
    t.MipLevels = mips;
    t.Format = format;
    t.SampleCount = 1;
    t.Layout = layout;
    return t;
}

void test_sizes() {
    // D3D12 layout 0: the tight estimate (256x256 RGBA8, 9 mips: 349,524 bytes).
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 9, 28, 0)) == 349524u);
    // Console linear 0x100 (#678): addrlib ADDR_SW_LINEAR, 256 B-aligned pitch per mip, the same
    // placement XG and the upload use: 360,192 bytes for 256x256 RGBA8 with 9 mips.
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 9, 28, 0x100)) == 360192u);
    const gfx10::TextureLayoutDesc r32{41u, 128u, 72u, 1u, 7u, 0x100u, 0u};
    CHECK(dg::resource_size_estimate(texture(3, 128, 72, 1, 7, 41, 0x100)) == 55552u);
    CHECK(gfx10::surface_bytes(r32) == 55552u);
    CHECK(dg::resource_size_estimate(texture(3, 128, 72, 4, 7, 41, 0x100)) == 4u * 55552u);    // array
    CHECK(dg::resource_size_estimate(texture(4, 32, 32, 36, 1, 42, 0x100)) == 36u * 32u * 256u);  // 3D
    CHECK(dg::resource_size_estimate(texture(2, 4096, 1, 1, 1, 41, 0x100)) == 16384u);       // 1D
    CHECK(dg::resource_size_estimate(texture(3, 560, 16, 1, 1, 29, 0x100)) == 2304u * 16u);
    // Committed resources use the same size (no 64 KiB-block model for 0x100).
    CHECK(dg::committed_memory_size(texture(3, 128, 72, 1, 7, 41, 0x100)) == 55552u);
    CHECK(dg::committed_memory_size(texture(3, 560, 16, 1, 1, 29, 0x100)) == 2304u * 16u);

    // 4KB_S: mip tail first, mip 0 last (addrlib): 352,256 bytes.
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 9, 28, 0x105)) == 352256u);
    // MipLevels 0 means the full chain.
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 0, 28, 0x105)) == 352256u);
    // Arrays repeat the chain per slice.
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 3, 9, 28, 0x105)) == 3u * 352256u);
    // Thick 3D 4KB_S: 17^3 R16G16B16A16_FLOAT in 8x8x8 blocks of 4 KiB.
    CHECK(dg::resource_size_estimate(texture(4, 17, 17, 17, 1, 10, 0x105)) == 27u * 4096u);
    // 64KB_D, BC1 1024x1024 with 9 mips: the same placement as the detiler.
    const gfx10::TextureLayoutDesc g{71u, 1024u, 1024u, 1u, 9u, 0x10Au, 0u};
    CHECK(dg::resource_size_estimate(texture(3, 1024, 1024, 1, 9, 71, 0x10A)) == gfx10::surface_bytes(g));
    // 256B_D, single mip: whole 8x8 blocks at 4 B (1280x720 -> 160x90 blocks).
    CHECK(dg::resource_size_estimate(texture(3, 1280, 720, 1, 1, 28, 0x102)) == 160u * 90u * 256u);

    // Not verified or not applicable: linear estimate.
    const uint64_t lin_rx = dg::resource_size_estimate(texture(3, 256, 256, 1, 1, 28, 0));
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 1, 28, 0x11B)) == lin_rx);  // 64KB_R_X
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 1, 28, 0x103)) == lin_rx);  // 256B_R
    dg::GuestResourceDesc msaa = texture(3, 256, 256, 1, 1, 28, 0x105);
    msaa.SampleCount = 4;
    CHECK(dg::resource_size_estimate(msaa) == lin_rx * 4u);
    // Depth/stencil (planar) under a verified mode: two tiled surfaces (depth, stencil), each
    // rounded up to whole swizzle blocks. 1000x600 D16_S8: 4 KiB_S blocks are 64x32 (2 B depth)
    // and 64x64 (1 B stencil).
    {
        const gfx10::TextureLayoutDesc dp{56u, 1000u, 600u, 1u, 1u, 0x105u, 0u};
        const gfx10::TextureLayoutDesc st{62u, 1000u, 600u, 1u, 1u, 0x105u, 0u};
        const uint64_t want = gfx10::surface_bytes(dp) + gfx10::surface_bytes(st);
        CHECK(gfx10::surface_bytes(dp) > 0 && gfx10::surface_bytes(st) > 0);
        CHECK(dg::resource_size_estimate(texture(3, 1000, 600, 1, 1, 118, 0x105)) == want);
        CHECK(want > 1000ull * 600ull * 3ull);  // more than the tight estimate (partial blocks)
        CHECK(dg::resource_size_estimate(texture(3, 1000, 600, 1, 1, 118, 0)) == 1000ull * 600ull * 3ull);
    }
    CHECK(dg::resource_size_estimate(texture(3, 1024, 1024, 1, 1, 118, 0x105)) == 1024ull * 1024ull * 3ull);
    // Buffers are their width.
    dg::GuestResourceDesc buffer{};
    buffer.Dimension = 1;
    buffer.Width = 1000;
    buffer.Layout = 0x105;
    CHECK(dg::resource_size_estimate(buffer) == 1000u);
}

dg::GuestResourceDesc small(uint64_t w, uint32_t h, uint16_t mips, uint32_t format, uint64_t alignment) {
    dg::GuestResourceDesc t = texture(3, w, h, 1, mips, format, 0);
    t.Alignment = alignment;
    return t;
}

// D3D12 small-resource placement (4 KiB alignment, layout UNKNOWN): 4 KiB swizzle blocks while
// the surface fits in 64 KiB, else the 64 KiB default.
void test_small_placement() {
    // 128x4 R16G16B16A16_UNORM (8 B texels): 4 KiB blocks of 32x16 texels -> 4 blocks = 16 KiB,
    // though the texels alone are 4 KiB.
    dg::ResourceAllocation a = dg::resource_allocation(small(128, 4, 1, 11, 4096));
    CHECK(a.size == 0x4000u && a.alignment == 0x1000u);
    CHECK(dg::resource_size_estimate(small(128, 4, 1, 11, 4096)) == 0x4000u);
    const gfx10::TextureLayoutDesc sw4{11u, 128u, 4u, 1u, 1u, 0x105u, 0u};
    CHECK(gfx10::surface_bytes(sw4) == 0x4000u);
    // 32x16 RGBA8: one 32x32 block of 4 KiB.
    a = dg::resource_allocation(small(32, 16, 1, 28, 4096));
    CHECK(a.size == 0x1000u && a.alignment == 0x1000u);
    // 64x64 RGBA8 with mips: the SW_4KB_S chain, still within 64 KiB.
    const gfx10::TextureLayoutDesc chain{28u, 64u, 64u, 1u, 7u, 0x105u, 0u};
    a = dg::resource_allocation(small(64, 64, 7, 28, 4096));
    CHECK(a.alignment == 0x1000u && a.size == gfx10::surface_bytes(chain) && a.size <= 0x10000u);
    // 256x256 RGBA8 does not fit in 64 KiB: the 64 KiB default (linear estimate, aligned).
    a = dg::resource_allocation(small(256, 256, 1, 28, 4096));
    CHECK(a.alignment == 0x10000u && a.size == 0x40000u);
    // Alignment 0 is unchanged: 64 KiB, linear estimate.
    a = dg::resource_allocation(small(128, 4, 1, 11, 0));
    CHECK(a.alignment == 0x10000u && a.size == 0x10000u);
    CHECK(dg::resource_size_estimate(small(128, 4, 1, 11, 0)) == 4096u);
    // MSAA never uses small placement.
    dg::GuestResourceDesc msaa = small(32, 16, 1, 28, 4096);
    msaa.SampleCount = 4;
    CHECK(dg::resource_allocation(msaa).alignment != 0x1000u);
    dg::GuestResourceDesc rt = small(32, 16, 1, 28, 4096);
    rt.Flags = 0x1;  // ALLOW_RENDER_TARGET
    CHECK(dg::resource_allocation(rt).alignment == 0x10000u);
    // Explicit console layouts with a block of at most 4 KiB keep their own size.
    dg::GuestResourceDesc linear = small(128, 4, 1, 11, 4096);
    linear.Layout = 0x100;
    a = dg::resource_allocation(linear);
    CHECK(a.alignment == 0x1000u && a.size == ((dg::resource_size_estimate(linear) + 0xFFFu) & ~0xFFFull));
}

// #2071: the title asks for 4 KiB first for every buffer and every texture it estimates under
// 512 KiB, sub-allocates 4 KiB answers inside shared 64 KiB pages, and tile-maps the resident mips
// of larger resources in whole 64 KiB pages from the base rounded down. Only textures whose whole
// surface fits in 64 KiB in a layout with blocks of at most 4 KiB may get 4 KiB.
void test_small_placement_title_requests() {
    // Buffers: never small, whatever their size.
    dg::GuestResourceDesc buffer{};
    buffer.Dimension = 1;
    buffer.Width = 0x3000;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleCount = 1;
    buffer.Alignment = 4096;
    dg::ResourceAllocation a = dg::resource_allocation(buffer);
    CHECK(a.alignment == 0x10000u && a.size == 0x10000u);
    buffer.Width = 0x14000;  // the size that used to straddle two 64 KiB pages at a 4 KiB offset
    a = dg::resource_allocation(buffer);
    CHECK(a.alignment == 0x10000u && a.size == 0x20000u);

    // Texture3D the way the title asks (layout set to 0x105 = SW_4KB_S): 4 KiB only within 64 KiB.
    dg::GuestResourceDesc vol = texture(4, 32, 32, 4, 1, 28, 0x105);
    vol.Alignment = 4096;
    a = dg::resource_allocation(vol);
    const gfx10::TextureLayoutDesc vol_small{28u, 32u, 32u, 4u, 1u, 0x105u, 0u};
    CHECK(a.alignment == 0x1000u && a.size == gfx10::surface_bytes(vol_small) && a.size <= 0x10000u);
    vol = texture(4, 64, 64, 16, 1, 28, 0x105);  // 256 KiB of texels
    vol.Alignment = 4096;
    a = dg::resource_allocation(vol);
    CHECK(a.alignment == 0x10000u && a.size % 0x10000u == 0 && a.size > 0x10000u);
    // Texture3D with layout UNKNOWN is sized like the title's SW_4KB_S request.
    vol = texture(4, 32, 32, 4, 1, 28, 0);
    vol.Alignment = 4096;
    CHECK(dg::resource_allocation(vol).alignment == 0x1000u &&
          dg::resource_allocation(vol).size == gfx10::surface_bytes(vol_small));

    // 2D in a 64 KiB swizzle (0x109 = SW_64KB_S): never 4 KiB, even when small.
    dg::GuestResourceDesc t64 = small(32, 16, 1, 28, 4096);
    t64.Layout = 0x109;
    a = dg::resource_allocation(t64);
    CHECK(a.alignment == 0x10000u && a.size == 0x10000u);
    // 2D with an explicit 4 KiB swizzle over 64 KiB: the default.
    dg::GuestResourceDesc t4 = small(256, 256, 1, 28, 4096);
    t4.Layout = 0x105;
    a = dg::resource_allocation(t4);
    CHECK(a.alignment == 0x10000u && a.size >= 0x40000u && a.size % 0x10000u == 0);
    // ...and within 64 KiB it keeps 4 KiB.
    t4 = small(64, 64, 1, 28, 4096);
    t4.Layout = 0x105;
    a = dg::resource_allocation(t4);
    CHECK(a.alignment == 0x1000u && a.size == 0x4000u);

    // Every 4 KiB answer fits in 64 KiB: a 4 KiB-aligned resource never reaches past the next page.
    for (uint32_t layout : {0u, 0x100u, 0x101u, 0x105u, 0x106u, 0x109u, 0x10Au}) {
        for (uint32_t dim : {2u, 3u, 4u}) {
            for (uint32_t w : {16u, 64u, 128u, 256u}) {
                dg::GuestResourceDesc t = texture(dim, w, dim == 2 ? 1 : w, dim == 4 ? 8 : 1, 1, 28, layout);
                t.Alignment = 4096;
                a = dg::resource_allocation(t);
                CHECK(a.alignment == 0x1000u || a.alignment == 0x10000u);
                if (a.alignment == 0x1000u) {
                    CHECK(a.size <= 0x10000u);
                    CHECK(layout != 0x109u && layout != 0x10Au);
                }
            }
        }
    }
}

// G82 (#1747): 3D volume, cubemap, and HDR mip-chain console layout sizes + incremental
// dirty-band / dirty-slice hashing & run coalescing.
void test_3d_cubemap_hdr_sizes_and_dirty_bands() {
    // 6-face cubemap HDR (R11G11B10_FLOAT = 26, RGBA16F = 10) in 64KB_D and 4KB_S:
    const gfx10::TextureLayoutDesc hdr_cube{26u, 256u, 256u, 1u, 9u, 0x10Au, 0u, 6u};
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 6, 9, 26, 0x10A)) == gfx10::surface_bytes(hdr_cube));
    const gfx10::TextureLayoutDesc f16_cube{10u, 128u, 128u, 1u, 8u, 0x105u, 0u, 6u};
    CHECK(dg::resource_size_estimate(texture(3, 128, 128, 6, 8, 10, 0x105)) == gfx10::surface_bytes(f16_cube));
    // Thick 3D volume mip chain in 64KB_S:
    const gfx10::TextureLayoutDesc vol64{26u, 64u, 64u, 32u, 7u, 0x109u, 0u, 1u};
    CHECK(dg::resource_size_estimate(texture(4, 64, 64, 32, 7, 26, 0x109)) == gfx10::surface_bytes(vol64));

    // Incremental 2D dirty-band hashing & coalescing (4 bands of 64 rows, 256B per row).
    constexpr uint32_t kRows = 256u;
    constexpr size_t kRowStride = 256u;
    std::vector<uint8_t> img(kRows * kRowStride, 0u);
    std::vector<uint64_t> bands0, bands1;
    // Initial sparse image: only band 1 (rows 64..127) non-zero.
    std::memset(img.data() + 64u * kRowStride, 0x42, 16u * kRowStride);
    const uint64_t h0 = dg::hash_dirty_bands(img.data(), kRows, dg::kUploadBandRows, kRowStride, kRowStride, &bands0);
    CHECK(h0 != 0u && bands0.size() == 4u);
    std::vector<uint8_t> zmask = {1u, 0u, 1u, 1u};
    std::vector<dg::DirtyBandRun> runs;
    const uint32_t init_changed =
        dg::compute_dirty_band_runs({}, bands0, kRows, dg::kUploadBandRows, &zmask, &runs);
    CHECK(init_changed == 1u && runs.size() == 1u && runs[0].first == 64u && runs[0].count == 64u);

    // Mutate bands 2 and 3 (adjacent -> coalesced into a single 128-row run); band 0 & 1 unchanged.
    std::memset(img.data() + 130u * kRowStride, 0x77, kRowStride);
    std::memset(img.data() + 200u * kRowStride, 0x88, kRowStride);
    const uint64_t h1 = dg::hash_dirty_bands(img.data(), kRows, dg::kUploadBandRows, kRowStride, kRowStride, &bands1);
    CHECK(h1 != h0);
    const uint32_t diff_changed =
        dg::compute_dirty_band_runs(bands0, bands1, kRows, dg::kUploadBandRows, nullptr, &runs);
    CHECK(diff_changed == 2u && runs.size() == 1u && runs[0].first == 128u && runs[0].count == 128u);

    // Incremental 3D dirty-slice hashing (8 depth slices, 512B per slice): mutate slices 1, 2 and 6.
    constexpr uint32_t kSlices = 8u;
    constexpr size_t kSliceBytes = 512u;
    std::vector<uint8_t> vol(kSlices * kSliceBytes, 0x11u);
    std::vector<uint64_t> z0, z1;
    dg::hash_dirty_bands(vol.data(), kSlices, 1u, kSliceBytes, kSliceBytes, &z0);
    vol[1u * kSliceBytes] = 0xAAu;
    vol[2u * kSliceBytes] = 0xBBu;
    vol[6u * kSliceBytes] = 0xCCu;
    dg::hash_dirty_bands(vol.data(), kSlices, 1u, kSliceBytes, kSliceBytes, &z1);
    const uint32_t z_changed = dg::compute_dirty_band_runs(z0, z1, kSlices, 1u, nullptr, &runs);
    CHECK(z_changed == 3u && runs.size() == 2u);
    CHECK(runs[0].first == 1u && runs[0].count == 2u);
    CHECK(runs[1].first == 6u && runs[1].count == 1u);
}

}  // namespace

int main() {
    test_sizes();
    test_small_placement();
    test_small_placement_title_requests();
    test_3d_cubemap_hdr_sizes_and_dirty_bands();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_console_layout_sizes: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_console_layout_sizes: all checks passed\n");
    return 0;
}
