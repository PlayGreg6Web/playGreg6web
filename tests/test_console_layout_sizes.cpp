// Synthetic tests: D3D12 resource sizes for textures with a console layout value (#551).
// Verified GFX10 modes are sized by the whole-surface placement in gpu/gfx10_detile.h; other
// layouts keep the linear estimate.
#include <cstdint>
#include <cstdio>

#include "gpu/d3d12_guest_objects.h"
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
    // MSAA and explicit console layouts keep their own answers.
    dg::GuestResourceDesc msaa = small(32, 16, 1, 28, 4096);
    msaa.SampleCount = 4;
    CHECK(dg::resource_allocation(msaa).alignment == 0x1000u);
    dg::GuestResourceDesc linear = small(128, 4, 1, 11, 4096);
    linear.Layout = 0x100;
    a = dg::resource_allocation(linear);
    CHECK(a.alignment == 0x1000u && a.size == ((dg::resource_size_estimate(linear) + 0xFFFu) & ~0xFFFull));
}

}  // namespace

int main() {
    test_sizes();
    test_small_placement();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_console_layout_sizes: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_console_layout_sizes: all checks passed\n");
    return 0;
}
