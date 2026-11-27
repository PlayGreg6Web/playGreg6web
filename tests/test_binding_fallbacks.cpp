// Synthetic tests for the binding fallback table and the per-phase tables (gpu/binding_fallbacks.h,
// gpu/drop_table.h, gpu/phase_hash_counts.h).
#include <cstdio>
#include <string>

#include "gpu/binding_fallbacks.h"
#include "gpu/drop_table.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_binding_fallbacks.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using namespace playgta6::gpu::d3d12_guest;
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    CHECK(gpu_phase_for_draws(0) == 0 && gpu_phase_for_draws(9) == 1 && gpu_phase_for_draws(99) == 2 &&
          gpu_phase_for_draws(999) == 3 && gpu_phase_for_draws(5000) == 4);
    set_current_gpu_phase(0);
    BindingFallbackTable t;
    t.record(0xabc, BindingFallbackTable::Reason::Unresolved, false, 3);
    set_current_gpu_phase(3);
    t.record(0xabc, BindingFallbackTable::Reason::EmptySlot, false);
    t.record(0xdef, BindingFallbackTable::Reason::EmptySlot, true, 2);
    CHECK(t.tracked() == 2);
    const std::string d = t.detail_json();
    CHECK(has(d, "\"hash\": \"0x0000000000000abc\", \"draws\": 4, \"dispatches\": 0"));
    CHECK(has(d, "\"unresolved_entry\":3"));
    CHECK(has(d, "\"empty_slot\":1"));
    CHECK(has(d, "\"phases\": ["));
    CHECK(has(d, "{\"name\": \"draws_0\", \"tracked\": 1, \"draws\": 3, \"dispatches\": 0"));
    CHECK(has(d, "{\"name\": \"draws_100_999\", \"tracked\": 2, \"draws\": 1, \"dispatches\": 2"));
    CHECK(!has(d, "draws_1_9"));  // empty phases are omitted

    // Bounded per phase and overall: hashes past the cap land in "other".
    BindingFallbackTable b;
    set_current_gpu_phase(1);
    for (uint64_t i = 1; i <= BindingFallbackTable::kMaxHashes + 10; ++i) {
        b.record(i, BindingFallbackTable::Reason::EmptySlot, false);
    }
    CHECK(b.tracked() == BindingFallbackTable::kMaxHashes);
    const std::string bd = b.detail_json();
    CHECK(has(bd, "\"other\": {\"draws\": 10, \"dispatches\": 0"));
    CHECK(has(bd, "{\"name\": \"draws_1_9\", \"tracked\": 512, \"draws\": 522"));
    b.reset();
    CHECK(b.tracked() == 0 && has(b.detail_json(), "\"phases\": []"));

    // The drop table carries the same phase split.
    DropTable dt;
    set_current_gpu_phase(2);
    dt.record(0x1, DropTable::Reason::RectList, false, 4);
    set_current_gpu_phase(4);
    dt.record(0x1, DropTable::Reason::Bridge, true);
    const std::string dd = dt.detail_json();
    CHECK(has(dd, "{\"name\": \"draws_10_99\", \"tracked\": 1, \"draws\": 4, \"dispatches\": 0"));
    CHECK(has(dd, "{\"name\": \"draws_1000_plus\", \"tracked\": 1, \"draws\": 0, \"dispatches\": 1"));
    CHECK(has(dd, "\"rect_list\":4"));
    dt.reset();
    CHECK(has(dt.detail_json(), "\"phases\": []"));

    set_current_gpu_phase(0);
    if (g_failures == 0) {
        std::printf("test_binding_fallbacks: ok\n");
    }
    return g_failures == 0 ? 0 : 1;
}
