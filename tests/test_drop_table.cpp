// Synthetic tests for the per-shader-hash drop table (gpu/drop_table.h).
#include <cstdio>
#include <string>

#include "gpu/drop_table.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_drop_table.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using playgta6::gpu::d3d12_guest::DropTable;
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    DropTable t;
    t.record(0xabc, DropTable::Reason::NoModule, false, 3);
    t.record(0xabc, DropTable::Reason::Bridge, true);
    t.record(0x1, DropTable::Reason::MeshRefused, false);
    CHECK(t.tracked() == 2);
    const std::string by = t.by_program_json();
    CHECK(by == "{\"0x0000000000000abc\": 4, \"0x0000000000000001\": 1}");
    const std::string d = t.detail_json();
    CHECK(has(d, "\"hash\": \"0x0000000000000abc\", \"draws\": 3, \"dispatches\": 1"));
    CHECK(has(d, "\"no_shader_module\":3"));
    CHECK(has(d, "\"mesh_refused\":1"));

    DropTable pp;  // a draw skipped while its pipeline is still compiling, keyed by the shader hash
    pp.record(0xfeed, DropTable::Reason::PipelinePending, false, 2);
    CHECK(has(pp.detail_json(), "\"pipeline_pending\":2"));

    // Bounded: hashes past the cap land in "other", tracked ones keep counting.
    DropTable b;
    for (uint64_t i = 1; i <= DropTable::kMaxHashes + 10; ++i) {
        b.record(i, DropTable::Reason::NoModule, false);
    }
    b.record(1, DropTable::Reason::NoModule, false, 5);
    CHECK(b.tracked() == DropTable::kMaxHashes);
    CHECK(has(b.detail_json(), "\"other\": {\"draws\": 10, \"dispatches\": 0"));
    CHECK(has(b.by_program_json(), "\"0x0000000000000001\": 6"));
    CHECK(!has(b.by_program_json(), "other"));

    b.reset();
    CHECK(b.tracked() == 0 && b.by_program_json() == "{}");
    if (g_failures == 0) {
        std::printf("test_drop_table: ok\n");
    }
    return g_failures == 0 ? 0 : 1;
}
