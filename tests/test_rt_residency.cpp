// Unit tests for gpu/rt_residency.h: address_sources parsing, dispatch residency planning (root
// params / root constants / user data, BVH roots through a TLAS), the structures_unavailable gate and
// the fault tracker.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gpu/rt_arena.h"
#include "gpu/rt_bvh.h"
#include "gpu/rt_residency.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_rt_residency.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using namespace playgta6::gpu;
using namespace playgta6::gpu::rt_residency;

struct Guest {
    std::map<uint64_t, uint32_t> dw;
    bool read(uint64_t a, void* dst, size_t n) const {
        auto* p = static_cast<uint32_t*>(dst);
        for (size_t i = 0; i < n / 4; ++i) {
            const auto it = dw.find(a + 4 * i);
            p[i] = it == dw.end() ? 0u : it->second;
        }
        return true;
    }
};

constexpr uint64_t kBlasA = 0x100000, kBlasB = 0x300000, kTlas = 0x200000;

Guest make_guest() {
    Guest g;
    const auto blas = [&](uint64_t at, uint32_t slots) {
        g.dw[at] = rt_bvh::kHeaderVersion | rt_bvh::kHeaderFlagBottomLevel;
        g.dw[at + 4] = slots;
    };
    blas(kBlasA, 1024);  // 64 KiB -> exactly one page
    blas(kBlasB, 2048);  // 128 KiB -> two pages
    g.dw[kTlas] = rt_bvh::kHeaderVersion;
    g.dw[kTlas + 4] = 16;      // 1 KiB
    g.dw[kTlas + 12] = 3;      // 3 instances (two share a BLAS)
    g.dw[kTlas + 13 * 4] = 64; // instances start at +64
    const uint64_t blases[3] = {kBlasA, kBlasB, kBlasA};
    for (int i = 0; i < 3; ++i) {
        const uint64_t at = kTlas + 64 + 64 * i;
        g.dw[at + 14 * 4] = static_cast<uint32_t>(blases[i] >> 8);
        g.dw[at + 15 * 4] = 5;  // root node
    }
    return g;
}

DispatchInputs inputs(const Guest& g) {
    DispatchInputs in;
    in.root_param = [](uint32_t i, uint64_t* va) {
        if (i == 1) { *va = 0x500000; return true; }
        if (i == 2) { *va = 0; return true; }  // null
        return false;
    };
    in.root_constant = [](uint32_t i, uint32_t d, uint32_t* v) {
        if (i != 0) return false;
        const uint64_t tlas = kTlas;
        if (d == 4) { *v = static_cast<uint32_t>(tlas); return true; }
        if (d == 5) { *v = static_cast<uint32_t>(tlas >> 32); return true; }
        return false;
    };
    in.user_data = [](uint32_t i, uint32_t* v) {
        if (i == 8) { *v = 0x00700000; return true; }
        if (i == 9) { *v = 0; return true; }
        return false;
    };
    in.read_guest = [&g](uint64_t a, void* d, size_t n) { return g.read(a, d, n); };
    return in;
}

void test_parse() {
    std::vector<AddressSource> s;
    std::string err;
    CHECK(parse_address_sources(
        R"([{"kind":"root_param","index":3,"extent":65536},
            {"kind":"root_constants","index":0,"dword":4,"bvh_root":true},
            {"kind":"user_data","index":8}])", &s, &err));
    CHECK(s.size() == 3);
    CHECK(s[0].kind == SourceKind::RootParam && s[0].index == 3 && s[0].extent == 65536 && !s[0].bvh_root);
    CHECK(s[1].kind == SourceKind::RootConstants && s[1].dword == 4 && s[1].bvh_root && s[1].extent == 1);
    CHECK(s[2].kind == SourceKind::UserData && s[2].index == 8);
    CHECK(!parse_address_sources(R"([{"kind":"nope","index":1}])", &s, &err));
    CHECK(!parse_address_sources(R"([{"kind":"root_param"}])", &s, &err));
    CHECK(!parse_address_sources("{", &s, &err));
    CHECK(parse_address_sources("[]", &s, &err) && s.empty());
}

void test_plan_and_prepare() {
    const Guest g = make_guest();
    std::vector<AddressSource> s;
    std::string err;
    CHECK(parse_address_sources(
        R"([{"kind":"root_param","index":1,"extent":70000},
            {"kind":"root_param","index":2},
            {"kind":"root_param","index":9},
            {"kind":"root_constants","index":0,"dword":4,"bvh_root":true},
            {"kind":"user_data","index":8,"extent":4}])", &s, &err));
    rt_arena::Arena arena(1, 64);
    Plan plan;
    const Outcome o = prepare_dispatch(arena, s, inputs(g), &plan);
    CHECK(o.ok);
    CHECK(plan.unresolved == 2);  // the null param and the unbound one
    CHECK(plan.structures == 3);  // TLAS + two distinct BLAS (visited once)
    // Resident: root_param 0x500000 + 70000 B (2 pages), TLAS 1 page, BLAS A 1 page, BLAS B 2 pages.
    CHECK(arena.is_resident(rt_arena::page_of(0x500000)) && arena.is_resident(rt_arena::page_of(0x500000) + 1));
    CHECK(arena.is_resident(rt_arena::page_of(kTlas)));
    CHECK(arena.is_resident(rt_arena::page_of(kBlasA)));
    CHECK(arena.is_resident(rt_arena::page_of(kBlasB)) && arena.is_resident(rt_arena::page_of(kBlasB) + 1));
    CHECK(arena.is_resident(rt_arena::page_of(0x700000)));
}

void test_unresolved_reasons() {
    std::vector<AddressSource> s;
    std::string err;
    CHECK(parse_address_sources(
        R"([{"kind":"root_param","index":2},{"kind":"root_param","index":9},
            {"kind":"user_data","index":20},{"kind":"user_data","index":30},{"kind":"user_data","index":8}])", &s, &err));
    const Guest g = make_guest();
    DispatchInputs in = inputs(g);
    in.user_data_failure = [](uint32_t i) { return i == 30 ? Unresolved::DescriptorTable : Unresolved::OutOfRange; };
    const Plan plan = plan_dispatch(s, in, nullptr);
    CHECK(plan.unresolved == 4);
    CHECK(plan.unresolved_by_reason[static_cast<size_t>(Unresolved::Null)] == 1);
    CHECK(plan.unresolved_by_reason[static_cast<size_t>(Unresolved::Unbound)] == 1);
    CHECK(plan.unresolved_by_reason[static_cast<size_t>(Unresolved::OutOfRange)] == 1);
    CHECK(plan.unresolved_by_reason[static_cast<size_t>(Unresolved::DescriptorTable)] == 1);
    CHECK(plan.ranges.size() == 1 && plan.ranges[0].first == 0x700000);
    const Plan none = plan_dispatch(s, DispatchInputs{}, nullptr);
    CHECK(none.unresolved_by_reason[static_cast<size_t>(Unresolved::NoInput)] == 5);
}

void test_unavailable_and_bad() {
    Guest g = make_guest();
    std::vector<AddressSource> s;
    std::string err;
    CHECK(parse_address_sources(R"([{"kind":"root_constants","index":0,"dword":4,"bvh_root":true}])", &s, &err));
    // A GPU-built TLAS cannot be walked.
    rt_arena::Arena arena(1, 64);
    arena.ensure_page(rt_arena::page_of(kTlas));
    arena.mark_gpu_written(rt_arena::page_of(kTlas));
    Plan plan;
    Outcome o = prepare_dispatch(arena, s, inputs(g), &plan);
    CHECK(!o.ok && o.reason == rt_arena::SkipReason::StructuresUnavailable);
    CHECK(arena.stats().skips[static_cast<size_t>(rt_arena::SkipReason::StructuresUnavailable)] == 1);
    CHECK(!arena.is_resident(rt_arena::page_of(kBlasA)));  // nothing was made resident
    // Unreadable guest memory is the same skip.
    rt_arena::Arena a2(1, 64);
    DispatchInputs in = inputs(g);
    in.read_guest = nullptr;
    o = prepare_dispatch(a2, s, in);
    CHECK(!o.ok && o.reason == rt_arena::SkipReason::StructuresUnavailable);
    // A bvh_root that is not a structure: counted, only its header page is made resident.
    g.dw[kTlas] = 0;
    rt_arena::Arena a3(1, 64);
    Plan p3;
    o = prepare_dispatch(a3, s, inputs(g), &p3);
    CHECK(o.ok && p3.bad_headers == 1 && p3.structures == 0);
    CHECK(a3.is_resident(rt_arena::page_of(kTlas)) && !a3.is_resident(rt_arena::page_of(kBlasA)));
}

void test_arena_full() {
    const Guest g = make_guest();
    std::vector<AddressSource> s;
    std::string err;
    CHECK(parse_address_sources(R"([{"kind":"root_constants","index":0,"dword":4,"bvh_root":true}])", &s, &err));
    rt_arena::Arena tiny(1, 2);  // the structure set needs 4 pages
    const Outcome o = prepare_dispatch(tiny, s, inputs(g));
    CHECK(!o.ok && o.reason == rt_arena::SkipReason::ArenaFull);
}

void test_faults() {
    rt_arena::Arena arena(1, 16);
    FaultTracker t;
    uint32_t w[kStatsWords] = {};
    CHECK(!t.consume(w, kStatsWords, arena));  // quiet frame
    w[0] = 5;
    w[1] = 5;
    w[2] = 100 + 1;
    w[3] = 100 + 1;  // duplicate
    w[4] = 200 + 1;
    w[5] = 300 + 1;
    arena.ensure_page(300);  // already resident
    arena.take_uploads();
    CHECK(t.consume(w, kStatsWords, arena));
    CHECK(t.stats().faults == 5);
    CHECK(t.stats().pages_serviced == 2 && t.stats().pages_already_resident == 1);
    CHECK(arena.is_resident(100) && arena.is_resident(200));
    CHECK(arena.take_uploads().size() == 2);  // they upload with the next dispatch
    CHECK(arena.stats().faults == 5);
    // The caller cleared words 1..31; word 0 keeps counting.
    std::memset(w + 1, 0, sizeof(w) - 4);
    w[0] = 7;
    w[1] = 40;  // log overflow: 30 recorded + 10 unlogged
    for (uint32_t i = 0; i < kFaultLogEntries; ++i) w[2 + i] = 1000 + i + 1;
    CHECK(t.consume(w, kStatsWords, arena));
    CHECK(t.stats().faults == 7);
    CHECK(t.stats().unlogged == 10);
    CHECK(t.stats().pages_serviced + t.stats().pages_unserviceable == 2 + 30);
    CHECK(t.stats().pages_unserviceable > 0);  // the 16-page arena cannot hold them
    CHECK(t.json().find("\"pages_serviced\"") != std::string::npos);
}
}  // namespace

int main() {
    test_parse();
    test_plan_and_prepare();
    test_unresolved_reasons();
    test_unavailable_and_bad();
    test_arena_full();
    test_faults();
    if (g_failures == 0) std::printf("test_rt_residency: OK\n");
    return g_failures == 0 ? 0 : 1;
}
