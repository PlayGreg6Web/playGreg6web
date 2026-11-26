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

void test_host_realloc() {
    TextureBudget t;
    t.set_budget_bytes(1000);
    t.note_created(900);  // 90%: cap 256
    CHECK(t.current_cap() == kTextureCapHigh);
    // A 1024x1024 BC1-like texture (4x4 blocks of 8 bytes): shrinking it by 2 levels saves its top two.
    const uint64_t full = texture_chain_bytes(1024, 1024, 1, 11, 4, 4, 8);
    const uint64_t capped = texture_chain_bytes(1024, 1024, 1, 11, 4, 4, 8, 2);
    CHECK(full - capped == texture_chain_bytes(1024, 1024, 1, 2, 4, 4, 8));
    // The capped host texture has the chain of a 256x256 texture with 9 levels.
    CHECK(capped == texture_chain_bytes(256, 256, 1, 9, 4, 4, 8));
    t.note_host_realloc(static_cast<int64_t>(300));
    CHECK(t.stats().used_bytes == 600 && t.stats().host_bytes_saved == 300 && t.stats().host_reallocs == 1);
    CHECK(t.stats().peak_used_bytes == 900);
    CHECK(t.current_cap() == 0);
    // Restore hysteresis: growing back by 300 would return to 90%, so it does not fit a 0-level floor.
    CHECK(!t.restore_fits(1024, 1024, 11, 0, 300));
    CHECK(t.cap_with_extra(300) == kTextureCapHigh);
    CHECK(t.restore_fits(1024, 1024, 11, 0, 50));  // 65% stays below the medium tier
    CHECK(t.restore_fits(1024, 1024, 11, 2, 300));  // the capped size itself is allowed
    t.note_host_realloc(-300);  // restored
    CHECK(t.stats().used_bytes == 900 && t.stats().host_bytes_saved == 0 && t.stats().host_reallocs == 2);
    // A shrunk texture destroyed: released bytes are counted back, then its full size leaves.
    t.note_host_realloc(300);
    t.note_shrunk_released(300);
    t.note_destroyed(300 + 0);
    CHECK(t.stats().used_bytes == 600 && t.stats().host_bytes_saved == 0);
    t.reset();
    CHECK(t.stats().host_reallocs == 0 && t.stats().host_bytes_saved == 0);
}
}  // namespace

int main() {
    test_cap_floor();
    test_chain_bytes();
    test_pressure_tiers();
    test_tracker();
    test_host_realloc();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_texture_budget: ok");
    return 0;
}
