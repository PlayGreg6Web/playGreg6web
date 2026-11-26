// Backlog 29: the memory budget map (os/memory_budget.h), XMem's per-kind summary and the
// ranged /userdata overlay read that replaced a whole-file copy per DirectStorage read.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "os/memory_budget.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_memory_budget.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;

void test_totals_and_json() {
    MemoryBudgetInput in;
    in.arena_committed = uint64_t{3} << 30;
    in.shadow_resident = uint64_t{1} << 30;
    in.shadow_covered = uint64_t{5} << 30;
    in.gpu_texture_bytes = 700;
    in.gpu_parked_bytes = 100;
    in.xmem_pages_nominal = 4 << 20;
    in.xmem_pages_resident = 1 << 20;
    const MemoryBudgetTotals t = memory_budget_totals(in);
    CHECK(t.host_private == (uint64_t{4} << 30));  // covered-but-unallocated shadow costs nothing
    CHECK(t.gpu_visible == 700);                   // parked textures are a subset
    CHECK(t.wasm_headroom == static_cast<int64_t>(uint64_t{12} << 30));
    CHECK(t.xmem_pages_saved == (3u << 20));
    in.arena_committed = uint64_t{17} << 30;
    CHECK(memory_budget_totals(in).wasm_headroom < 0);  // over the browser limit shows negative
    const std::string j = memory_budget_json(in);
    CHECK(j.front() == '{' && j.back() == '}');
    CHECK(j.find("\"pages_not_backed_bytes\": 3145728") != std::string::npos);
    CHECK(j.find("\"headroom\": -") != std::string::npos);
}

void test_xmem_summary() {
    scarlett::xmem_reset_for_testing();
    const scarlett::XMemMemorySummary before = scarlett::xmem_memory_summary();
    CHECK(before.pages == 0 && before.resident_page_bytes == 0);
    size_t count = 4;
    uint64_t pfns[4] = {};
    CHECK(pg6_xbox_XMemAllocatePhysicalPages(0, &count, pfns) == 1);
    scarlett::XMemMemorySummary m = scarlett::xmem_memory_summary();
    CHECK(m.pages == 4);
    CHECK(m.physical_pages_bytes == 4 * (64u << 10));
    CHECK(m.pages_unbacked == 4 && m.resident_page_bytes == 0);  // reserved, nothing committed
    void* mapped = pg6_xbox_XMemMapPhysicalPages(nullptr, 2, pfns);
    CHECK(mapped != nullptr);
    m = scarlett::xmem_memory_summary();
    CHECK(m.pages_unbacked == 2);  // the two mapped pages got bytes, the others stay free
    CHECK(m.resident_page_bytes == 2 * (64u << 10));
    CHECK(m.owned_block_bytes == 2 * (64u << 10));
    CHECK(pg6_xbox_XMemFreePhysicalPages(4, pfns) == 1);
    CHECK(scarlett::xmem_memory_summary().pages == 0);
    scarlett::xmem_reset_for_testing();
}

void test_userdata_range() {
    win32::clear_userdata_overlay();
    std::vector<uint8_t> data(1000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i * 7);
    }
    win32::write_userdata_file("/userdata/mb.bin", data);
    uint64_t size = 0;
    CHECK(win32::userdata_file_size("/userdata/mb.bin", &size) && size == 1000);
    uint8_t buf[16] = {};
    CHECK(win32::read_userdata_range("/userdata/mb.bin", 100, buf, sizeof(buf)));
    CHECK(std::memcmp(buf, data.data() + 100, sizeof(buf)) == 0);
    CHECK(win32::read_userdata_range("/userdata/mb.bin", 984, buf, sizeof(buf)));  // exactly to the end
    CHECK(!win32::read_userdata_range("/userdata/mb.bin", 985, buf, sizeof(buf)));
    CHECK(!win32::read_userdata_range("/userdata/mb.bin", 1001, buf, 0));
    CHECK(!win32::userdata_file_size("/userdata/none.bin", &size));
    CHECK(!win32::read_userdata_range("/userdata/none.bin", 0, buf, 1));
    win32::clear_userdata_overlay();
}
}  // namespace

int main() {
    test_totals_and_json();
    test_xmem_summary();
    test_userdata_range();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_memory_budget: ok");
    return 0;
}
