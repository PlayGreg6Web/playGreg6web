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
    // Linear: the estimate is unchanged (256x256 RGBA8, 9 mips: 349,524 bytes).
    const uint64_t linear = dg::resource_size_estimate(texture(3, 256, 256, 1, 9, 28, 0x100));
    CHECK(linear == 349524u);
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 9, 28, 0)) == linear);

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
    const uint64_t lin_rx = dg::resource_size_estimate(texture(3, 256, 256, 1, 1, 28, 0x100));
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 1, 28, 0x11B)) == lin_rx);  // 64KB_R_X
    CHECK(dg::resource_size_estimate(texture(3, 256, 256, 1, 1, 28, 0x103)) == lin_rx);  // 256B_R
    dg::GuestResourceDesc msaa = texture(3, 256, 256, 1, 1, 28, 0x105);
    msaa.SampleCount = 4;
    CHECK(dg::resource_size_estimate(msaa) == lin_rx * 4u);
    // Depth/stencil (planar) keeps its per-plane estimate.
    CHECK(dg::resource_size_estimate(texture(3, 1024, 1024, 1, 1, 118, 0x105)) == 1024ull * 1024ull * 3ull);
    // Buffers are their width.
    dg::GuestResourceDesc buffer{};
    buffer.Dimension = 1;
    buffer.Width = 1000;
    buffer.Layout = 0x105;
    CHECK(dg::resource_size_estimate(buffer) == 1000u);
}

}  // namespace

int main() {
    test_sizes();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_console_layout_sizes: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_console_layout_sizes: all checks passed\n");
    return 0;
}
