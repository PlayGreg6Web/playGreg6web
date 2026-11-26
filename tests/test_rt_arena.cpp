// Unit tests for the paged RT address arena (gpu/rt_arena.h).
#include <cstdio>
#include <string>

#include "gpu/rt_arena.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_rt_arena.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using namespace playgta6::gpu::rt_arena;
}  // namespace

int main() {
    // Address -> page mapping and entry packing.
    CHECK(page_of(0x12345678ull) == 0x1234);
    CHECK(page_of(0xFFFFFFFFFFull) == 0xFFFFFFull);
    CHECK(page_of(0x1FFFFFFFFFFull) == 0xFFFFFFull);  // bits above 40 ignored
    CHECK(top_index(0xFFFFFFull) == 4095 && l2_index(0xFFFFFFull) == 4095);
    CHECK(top_index(0x1000) == 1 && l2_index(0x1000) == 0);
    CHECK(kPagesPerArena == 8192);
    for (uint32_t a = 0; a < 4; ++a) {
        const uint32_t e = pack_entry(a, 8191);
        CHECK(entry_valid(e) && entry_arena(e) == a && entry_slot(e) == 8191);
    }
    CHECK(pack_entry(0, 0) != 0);
    CHECK(kBindArena0 + 3 < kBindDirectory && kBindStats == 21);

    // Residency, directory packing, mapping.
    Arena a(2, 4);  // 8 pages
    CHECK(a.capacity_pages() == 8);
    CHECK(a.lookup_entry(0x500000) == 0);
    CHECK(a.ensure_page(0x50) == Ensure::Uploaded);
    CHECK(a.ensure_page(0x50) == Ensure::Resident);
    const uint32_t e = a.lookup_entry((0x50ull << 16) | 0x123);
    CHECK(entry_valid(e) && entry_arena(e) == 0 && entry_slot(e) == 0);
    CHECK(a.directory()[0] == 1);                   // top[0] -> table 1
    CHECK(a.directory()[kTopEntries + 0x50] == e);  // L2 word
    uint32_t f = 0, l = 0;
    CHECK(a.take_dir_dirty(&f, &l) && f == 0 && l == kTopEntries + 0x50);
    CHECK(!a.take_dir_dirty(&f, &l));
    CHECK(a.ensure_page(0xABCDEF) == Ensure::Uploaded);  // second table
    CHECK(a.directory().size() == kTopEntries + 2u * kL2Entries);
    CHECK(a.take_uploads().size() == 2);
    CHECK(a.take_uploads().empty());

    // Range, LRU eviction, epoch protection.
    Arena b(1, 4);
    b.begin_dispatch();
    CHECK(b.ensure_range(0, 4 * kPageBytes) == Ensure::Uploaded);
    CHECK(b.ensure_page(100) == Ensure::ArenaFull);  // all four protected by this dispatch
    b.begin_dispatch();
    b.begin_frame();
    CHECK(b.ensure_page(1) == Ensure::Resident);  // refresh page 1
    b.begin_dispatch();
    CHECK(b.ensure_page(100) == Ensure::Uploaded);  // evicts LRU victim
    CHECK(b.is_resident(1) && b.is_resident(100));
    CHECK(b.stats().evictions == 1);
    CHECK(b.lookup_entry(100 * kPageBytes) != 0);

    // Pinning keeps structure pages; arena full when pins fill it; release frees them.
    Arena c(1, 4);
    CHECK(c.pin_structure(7, 0, 3 * kPageBytes) == Ensure::Uploaded);
    c.begin_dispatch();
    CHECK(c.ensure_page(50) == Ensure::Uploaded);
    c.begin_dispatch();
    CHECK(c.ensure_page(51) == Ensure::Uploaded);  // evicts 50, not pinned 0..2
    CHECK(c.is_resident(0) && c.is_resident(2) && !c.is_resident(50));
    CHECK(c.pin_structure(8, 200 * kPageBytes, 2 * kPageBytes) == Ensure::ArenaFull);
    CHECK(c.stats().pinned == 3);  // failed pin rolled back
    c.release_structure(7);
    c.begin_dispatch();
    CHECK(c.ensure_page(60) == Ensure::Uploaded);
    CHECK(c.stats().pinned == 0);

    // GPU ownership: dirty marks do not re-upload until remapped.
    Arena d(1, 4);
    d.ensure_page(5);
    d.take_uploads();
    d.mark_gpu_written(5);
    d.mark_cpu_dirty(5);
    CHECK(d.ensure_page(5) == Ensure::Resident);
    CHECK(d.take_uploads().empty() && d.stats().skipped_gpu_owned == 1);
    d.mark_cpu_remapped(5 * kPageBytes, kPageBytes);
    CHECK(d.ensure_page(5) == Ensure::Uploaded && d.take_uploads().size() == 1);
    d.mark_cpu_dirty(5);
    CHECK(d.ensure_page(5) == Ensure::Uploaded);

    // Limits and counters.
    Arena g(4, 8192);
    CHECK(g.capacity_bytes() == (2048ull << 20));
    CHECK(g.ensure_page(1ull << 24) == Ensure::OutOfRange);
    Arena t(1, 100000);
    uint32_t ok = 0;
    for (uint32_t i = 0; i < kMaxL2Tables + 2; ++i) ok += t.ensure_page(static_cast<uint64_t>(i) << 12) == Ensure::Uploaded;
    CHECK(ok == kMaxL2Tables && t.stats().table_full == 2);
    CHECK((kTopEntries + kMaxL2Tables * kL2Entries) * 4ull < (1ull << 20));
    a.note_skip(SkipReason::RtOff);
    a.set_fault_count(9);
    const std::string j = a.json();
    CHECK(j.find("\"rt_off\": 1") != std::string::npos && j.find("\"rt_fault\": 9") != std::string::npos);

    if (g_failures == 0) std::printf("test_rt_arena: OK\n");
    return g_failures ? 1 : 0;
}
