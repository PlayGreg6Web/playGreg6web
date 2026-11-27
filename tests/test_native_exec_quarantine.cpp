// Shader quarantine parsing and drop-table accounting (gpu/native_exec/shader_quarantine.h).
#include <cstdio>
#include <string>

#include "gpu/drop_table.h"
#include "gpu/native_exec/shader_quarantine.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_native_exec_quarantine.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
using playgta6::gpu::d3d12_guest::DropTable;
using playgta6::gpu::native_exec::ShaderQuarantine;
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    uint64_t h = 0;
    CHECK(ShaderQuarantine::parse_hash("0xDEADbeef", &h) && h == 0xdeadbeefull);
    CHECK(ShaderQuarantine::parse_hash("ff", &h) && h == 0xff);
    CHECK(!ShaderQuarantine::parse_hash("", &h));
    CHECK(!ShaderQuarantine::parse_hash("0x", &h));
    CHECK(!ShaderQuarantine::parse_hash("xyz", &h));
    CHECK(!ShaderQuarantine::parse_hash("11111111111111111", &h));  // 17 digits

    ShaderQuarantine q;
    std::string err;
    CHECK(q.add_list("0x1, abc,,0x00000000000000ff", &err));
    CHECK(q.size() == 3 && q.contains(1) && q.contains(0xabc) && q.contains(0xff) && !q.contains(2));
    CHECK(!q.contains(0));
    ShaderQuarantine bad;
    CHECK(!bad.add_list("0x1,nothex", &err) && has(err, "nothex") && bad.empty());  // all or nothing

    // Per-run file: bare hashes and comments allowed.
    ShaderQuarantine f;
    CHECK(f.add_text("# c\n\n0x10\r\n0x20 hangs here\n", false, "run.txt", &err));
    CHECK(f.size() == 2 && f.contains(0x10) && f.contains(0x20));

    // Repo list: reason + issue link required; '#123' mid-line is a reference, not a comment.
    ShaderQuarantine r;
    CHECK(r.add_text("# header only\n", true, "list", &err) && r.empty());  // empty by default
    CHECK(r.add_text("0xaa hair pass hang https://example.org/issues/1\n0xbb crashes DXC #42\n", true, "list", &err));
    CHECK(r.size() == 2 && r.entries().at(0xbb) == "crashes DXC #42");
    ShaderQuarantine r2;
    CHECK(!r2.add_text("0xaa\n", true, "list", &err) && has(err, "list line 1") && r2.empty());
    CHECK(!r2.add_text("0xaa only a reason\n", true, "list", &err));
    CHECK(!r2.add_text("0xcc ok #1\nzz bad #2\n", true, "list", &err) && has(err, "line 2") && r2.empty());

    // Accounting: skips land in the drop table as "quarantined" with the hash, split draws / dispatches.
    DropTable t;
    t.record(0xabc, DropTable::Reason::Quarantined, false, 2);
    t.record(0xabc, DropTable::Reason::Quarantined, true);
    const std::string d = t.detail_json();
    CHECK(has(d, "\"hash\": \"0x0000000000000abc\", \"draws\": 2, \"dispatches\": 1"));
    CHECK(has(d, "\"quarantined\":3"));
    CHECK(!has(d, "no_shader_module\":1"));

    if (g_failures) return 1;
    std::printf("test_native_exec_quarantine: ok\n");
    return 0;
}
