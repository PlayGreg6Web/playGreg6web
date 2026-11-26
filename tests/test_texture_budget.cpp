#include <cstdint>
#include <cstdio>

#include "gpu/texture_budget.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_texture_budget.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu;

void test_cap_floor() {
    CHECK(cap_floor_for(4096, 4096, 13, 0) == 0);      // no cap
    CHECK(cap_floor_for(4096, 2048, 13, 1024) == 2);   // 4096 -> 1024 drops two levels
    CHECK(cap_floor_for(4096, 4096, 13, 256) == 4);
    CHECK(cap_floor_for(512, 512, 10, 1024) == 0);     // already fits
    CHECK(cap_floor_for(300, 100, 9, 256) == 1);       // the larger side decides
    CHECK(cap_floor_for(4096, 4096, 2, 256) == 1);     // never drops the last level
    CHECK(cap_floor_for(4096, 4096, 1, 256) == 0);
}

void test_chain_bytes() {
    // BC1 (4x4 blocks, 8 B): 256x256 mip 0 = 64*64*8, mip 1 = 32*32*8, ...
    CHECK(texture_chain_bytes(256, 256, 1, 1, 4, 4, 8) == 64u * 64u * 8u);
    CHECK(texture_chain_bytes(256, 256, 1, 2, 4, 4, 8, 1) == 32u * 32u * 8u);
    CHECK(texture_chain_bytes(8, 8, 1, 4, 0, 4, 8) == 0);
}

void test_pressure_tiers() {
    constexpr uint64_t b = 1000;
    CHECK(cap_for_pressure(0, 0, 0) == 0);              // no budget, no base cap
    CHECK(cap_for_pressure(999, 0, 1024) == 1024);      // base cap only
    CHECK(cap_for_pressure(690, b, 0) == 0);            // below 70 percent
    CHECK(cap_for_pressure(700, b, 0) == kTextureCapMedium);
    CHECK(cap_for_pressure(840, b, 0) == kTextureCapMedium);
    CHECK(cap_for_pressure(850, b, 0) == kTextureCapHigh);
    CHECK(cap_for_pressure(700, b, 256) == 256);        // the tighter of base and tier wins
    CHECK(cap_for_pressure(850, b, 512) == kTextureCapHigh);
    CHECK(cap_for_pressure(100, b, 1024) == 1024);
    CHECK(cap_for_pressure(UINT64_MAX / 10 * 8, UINT64_MAX, 0) == kTextureCapMedium);  // no overflow
}

void test_tracker() {
    TextureBudget t;
    t.set_budget_bytes(1000);
    CHECK(t.current_cap() == 0);
    t.note_created(720);
    CHECK(t.current_cap() == kTextureCapMedium);
    t.note_created(200);
    CHECK(t.current_cap() == kTextureCapHigh);
    CHECK(t.stats().used_bytes == 920 && t.stats().peak_used_bytes == 920);
    t.note_destroyed(500);  // memory frees: the cap loosens
    CHECK(t.current_cap() == 0);
    CHECK(t.stats().cap_changes == 3 && t.stats().peak_used_bytes == 920);
    t.note_destroyed(100000);  // over-release clamps at zero
    CHECK(t.stats().used_bytes == 0);
    t.note_capped(2, 4096);
    t.note_capped(1, 100);
    t.note_restored();
    const TextureBudgetStats s = t.stats();
    CHECK(s.textures_capped == 2 && s.mips_dropped == 3 && s.bytes_not_uploaded == 4196 && s.textures_restored == 1);
    t.set_base_cap(1024);
    CHECK(t.current_cap() == 1024 && t.stats().base_cap == 1024);
    t.reset();
    CHECK(t.stats().cap_changes == 0 && t.stats().budget_bytes == 0);
}
}  // namespace

int main() {
    test_cap_floor();
    test_chain_bytes();
    test_pressure_tiers();
    test_tracker();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_texture_budget: ok");
    return 0;
}
