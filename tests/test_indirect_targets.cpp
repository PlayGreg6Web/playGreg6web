// Recorded indirect-call targets (guest/indirect_targets.h): per-thread dedupe, merge into the
// shared log, append-only file with no duplicates across runs, and tolerance of a missing file.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "guest/indirect_targets.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_indirect_targets.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using playgta6::guest::IndirectTargetCollector;
using playgta6::guest::IndirectTargetLog;
using playgta6::guest::IndirectTargetPair;

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

size_t count_lines(const std::string& text, const std::string& needle) {
    size_t n = 0;
    for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + 1)) {
        ++n;
    }
    return n;
}

}  // namespace

int main() {
    const std::string path = "test_indirect_targets.tmp.txt";
    std::remove(path.c_str());

    IndirectTargetCollector c;
    c.note(0x1000, 0x2000);
    c.note(0x1000, 0x2000);  // duplicate
    c.note(0x1000, 0x3000);
    c.note(0x1008, 0x2000);
    auto pending = c.take_pending();
    CHECK(pending.size() == 3);
    CHECK(!c.has_pending());
    c.note(0x1000, 0x2000);  // still remembered after the hand-over
    CHECK(!c.has_pending());

    {
        IndirectTargetLog log(path);  // no file yet
        CHECK(log.flush() == 0);      // nothing recorded: no file is created
        CHECK(slurp(path).empty());
        log.merge(pending);
        std::thread t([&] {  // another thread's collector merging concurrently
            IndirectTargetCollector c2;
            c2.note(0x1000, 0x2000);  // same pair as the first thread
            c2.note(0x5000, 0x6000);
            log.merge(c2.take_pending());
        });
        t.join();
        CHECK(log.flush() == 4);
        CHECK(log.flush() == 0);
    }
    std::string text = slurp(path);
    CHECK(count_lines(text, "0x1000 0x2000\n") == 1);
    CHECK(count_lines(text, "0x5000 0x6000\n") == 1);

    {  // a later run appends only what is new
        IndirectTargetLog log(path);
        CHECK(log.known() == 4);
        log.merge({IndirectTargetPair{0x1000, 0x2000}, IndirectTargetPair{0x7000, 0x8000}});
        CHECK(log.flush() == 1);
    }
    text = slurp(path);
    CHECK(count_lines(text, "0x1000 0x2000\n") == 1);
    CHECK(count_lines(text, "0x7000 0x8000\n") == 1);
    CHECK(count_lines(text, "#") == 1);  // one header, written once

    {  // an unwritable path reports failure and keeps the pairs
        IndirectTargetLog log("no_such_dir_pg6/never/targets.txt");
        log.merge({IndirectTargetPair{1, 2}});
        CHECK(log.flush() == -1);
    }

    std::remove(path.c_str());
    if (g_failures != 0) {
        std::fprintf(stderr, "test_indirect_targets failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_indirect_targets ok");
    return 0;
}
