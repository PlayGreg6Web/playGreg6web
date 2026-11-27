// Unit tests for the compute-written texture read-back re-tile (runtime/gpu/compute_texture_readback.h)
// and its round trip against gfx10_detile. Synthetic data only.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu/compute_texture_readback.h"
#include "gpu/d3d12_readback.h"
#include "gpu/gfx10_detile.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_compute_texture_readback.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu;
namespace cr = playgta6::gpu::compute_readback;

std::vector<uint8_t> synth(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t x = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v[i] = static_cast<uint8_t>(x & 0x7Fu);  // never all-0xFF
    }
    return v;
}

// Every swizzle mode the detiler supports (console layout 0x100 | mode) x the float formats:
// re-tile -> detile is the identity, and the bytes outside the mip are untouched.
void test_round_trip_all_modes() {
    const uint32_t modes[] = {0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 25, 26, 27, 29, 30, 31};
    const uint32_t formats[] = {41, 54, 16, 34, 10, 2};
    const uint32_t sizes[][2] = {{256, 256}, {64, 32}, {100, 60}, {17, 5}};
    uint32_t seed = 1;
    uint32_t planned = 0;
    for (uint32_t mode : modes) {
        if (!gfx10::is_swizzle_supported(static_cast<gfx10::SwizzleMode>(mode))) continue;
        for (uint32_t f : formats) {
            for (const auto& sz : sizes) {
                for (uint32_t mips : {1u, 4u}) {
                    cr::Target t;
                    t.dxgi_format = f;
                    t.width = sz[0];
                    t.height = sz[1];
                    t.mip_levels = mips;
                    t.d3d12_layout = 0x100u | mode;
                    t.is_2d = true;
                    for (uint32_t mip = 0; mip < mips; ++mip) {
                        const cr::Plan p = cr::plan_readback(t, mip);
                        if (!p.ok) continue;  // unplaceable combination: nothing to read back
                        ++planned;
                        const std::vector<uint8_t> lin = synth(p.linear_bytes, ++seed);
                        std::vector<uint8_t> surface(p.surface_bytes, 0xA5);
                        CHECK(cr::retile_into_guest(p, lin.data(), lin.size(), surface.data(), surface.size()));
                        std::vector<uint8_t> back;
                        const gfx10::DetileResult r =
                            gfx10::detile_surface_subresource(p.desc, mip, 0u, surface.data(), surface.size(), back);
                        CHECK(r.ok);
                        CHECK(back == lin);
                        // a second mip written afterwards must not disturb this one
                        if (mip + 1 < mips) {
                            const cr::Plan q = cr::plan_readback(t, mip + 1);
                            if (q.ok) {
                                const std::vector<uint8_t> lin2 = synth(q.linear_bytes, ++seed);
                                CHECK(cr::retile_into_guest(q, lin2.data(), lin2.size(), surface.data(),
                                                            surface.size()));
                                gfx10::detile_surface_subresource(p.desc, mip, 0u, surface.data(), surface.size(), back);
                                CHECK(back == lin);
                            }
                        }
                    }
                }
            }
        }
    }
    CHECK(planned > 100);
}

void test_plan_rejects() {
    cr::Target t;
    t.dxgi_format = 41;
    t.width = t.height = 256;
    t.d3d12_layout = 0x100u | 9u;
    t.is_2d = true;
    CHECK(cr::plan_readback(t, 0).ok);
    CHECK(cr::plan_readback(t, 0).row_bytes == 1024u);
    CHECK(!cr::plan_readback(t, 1).ok);  // one mip level only
    cr::Target u = t;
    u.is_2d = false;
    CHECK(!cr::plan_readback(u, 0).ok);
    u = t;
    u.dxgi_format = 28;  // R8G8B8A8_UNORM is not a float format
    CHECK(!cr::plan_readback(u, 0).ok);
    u = t;
    u.d3d12_layout = 1;  // D3D12 ROW_MAJOR: not a console layout
    CHECK(!cr::plan_readback(u, 0).ok);
    u = t;
    u.width = u.height = 1024;  // 4 MiB tight: over the cap
    CHECK(!cr::plan_readback(u, 0).ok);
    u = t;
    u.d3d12_layout = 0x100u | 4u;  // undocumented swizzle mode
    CHECK(!cr::plan_readback(u, 0).ok);
}

// Only real GPU results: an untouched (sentinel) scratch is never written to the guest.
void test_never_fabricated() {
    cr::Target t;
    t.dxgi_format = 41;
    t.width = t.height = 64;
    t.d3d12_layout = 0x100u | 10u;
    t.is_2d = true;
    const cr::Plan p = cr::plan_readback(t, 0);
    CHECK(p.ok);
    std::vector<uint8_t> scratch(p.linear_bytes);
    cr::fill_sentinel(scratch.data(), scratch.size());
    CHECK(!cr::was_delivered(scratch.data(), scratch.size()));
    std::vector<uint8_t> surface(p.surface_bytes, 0x5A);
    const std::vector<uint8_t> before = surface;
    CHECK(!cr::retile_into_guest(p, scratch.data(), scratch.size(), surface.data(), surface.size()));
    CHECK(surface == before);
    scratch[scratch.size() / 2] = 0x00;  // the consumer wrote something
    CHECK(cr::was_delivered(scratch.data(), scratch.size()));
    CHECK(cr::retile_into_guest(p, scratch.data(), scratch.size(), surface.data(), surface.size()));
    // short buffers are rejected
    CHECK(!cr::retile_into_guest(p, scratch.data(), scratch.size() - 4, surface.data(), surface.size()));
    CHECK(!cr::retile_into_guest(p, scratch.data(), scratch.size(), surface.data(), surface.size() - 1));
}

// The gate runs a read-back's completion callback once, in order, only after the consumer reported
// it done (the consumer adds 1 to the u32 at done_address), and marks no guest range for va 0.
void test_gate_completion_callback() {
    d3d12_guest::ReadbackGate gate(/*hold=*/false, std::chrono::milliseconds(5000));
    std::vector<int> order;
    const uint32_t s1 = gate.issue_with_completion(0, 0, d3d12_guest::ReadbackKind::Texture, 0, [&] { order.push_back(1); });
    const uint32_t s2 = gate.issue_with_completion(0, 0, d3d12_guest::ReadbackKind::Texture, 0, [&] { order.push_back(2); });
    CHECK(s1 == 1 && s2 == 2);
    CHECK(gate.pending_count() == 2);
    gate.poll();
    CHECK(order.empty());  // nothing completed yet: no callback
    auto* done = reinterpret_cast<std::atomic<uint32_t>*>(gate.done_address());
    done->fetch_add(1);
    CHECK(gate.try_poll_slot(1));
    CHECK(order.size() == 1 && order[0] == 1);
    done->fetch_add(1);
    CHECK(gate.try_poll_slot(2));
    CHECK(order.size() == 2 && order[1] == 2);
    gate.poll();
    CHECK(order.size() == 2);  // never run twice
}
}  // namespace

int main() {
    test_round_trip_all_modes();
    test_plan_rejects();
    test_never_fabricated();
    test_gate_completion_callback();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_compute_texture_readback: OK");
    return 0;
}
