#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "guest/guest_probe.h"

namespace {
int g_failures = 0;
#define CHECK(expr)                                                                        \
    do {                                                                                   \
        if (!(expr)) {                                                                     \
            std::fprintf(stderr, "FAIL [test_guest_probe.cpp:%d]: %s\n", __LINE__, #expr); \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

using namespace playgta6::guest;

// Fake guest memory: [lo, hi) readable, filled with the low byte of the address; counts calls.
struct FakeMem {
    uint64_t lo = 0x100000, hi = 0x102000;
    int calls = 0;
    ProbeReadFn fn() {
        return [this](uint64_t va, void* dst, size_t len) {
            ++calls;
            if (va < lo || va + len > hi) return false;
            for (size_t i = 0; i < len; ++i) static_cast<uint8_t*>(dst)[i] = static_cast<uint8_t>(va + i);
            return true;
        };
    }
};

void test_parse() {
    ProbeSpec s;
    std::string e;
    CHECK(parse_probe_spec("0x140123456", &s, &e) && s.addr == 0x140123456ull && s.name.empty() && s.args == 0);
    CHECK(parse_probe_spec("140123456:my_fn:args=6:ret:mem=rcx+10,32:mem=rdx-8,16:max=5", &s, &e));
    CHECK(s.addr == 0x140123456ull && s.name == "my_fn" && s.args == 6 && s.ret && s.max_lines == 5);
    CHECK(s.mem.size() == 2 && s.mem[0].reg == 1 && s.mem[0].off == 0x10 && s.mem[0].len == 32);
    CHECK(s.mem[1].reg == 2 && s.mem[1].off == -8 && s.mem[1].len == 16);
    CHECK(parse_probe_spec("0x1000:name=a.b-c:mem=rsp,8", &s, &e) && s.name == "a.b-c" && s.mem[0].off == 0);
    CHECK(parse_probe_spec("game.dll+0x1f40:x", &s, &e) && s.module == "game.dll" && s.addr == 0x1f40);
    CHECK(!parse_probe_spec("", &s, &e));
    CHECK(!parse_probe_spec("0", &s, &e));
    CHECK(!parse_probe_spec("zz", &s, &e));
    CHECK(!parse_probe_spec("0x1000:args=9", &s, &e));
    CHECK(!parse_probe_spec("0x1000:args=", &s, &e));
    CHECK(!parse_probe_spec("0x1000:mem=rcx+10", &s, &e));
    CHECK(!parse_probe_spec("0x1000:mem=rcx+10,0", &s, &e));
    CHECK(!parse_probe_spec("0x1000:mem=rcx+10,257", &s, &e));
    CHECK(!parse_probe_spec("0x1000:mem=foo+10,8", &s, &e));
    CHECK(!parse_probe_spec("0x1000:max=0", &s, &e));
    CHECK(!parse_probe_spec("0x1000:a:b", &s, &e));  // a second bare name
    CHECK(!parse_probe_spec("0x1000::x", &s, &e));
    CHECK(!parse_probe_spec("0x1000:bad name", &s, &e));
    CHECK(!parse_probe_spec("0x1000:mem=rax,8:mem=rax,8:mem=rax,8:mem=rax,8:mem=rax,8", &s, &e));
    CHECK(probe_reg_index("r15") == 15 && probe_reg_index("rsp") == 4 && probe_reg_index("eax") == -1);
}

void test_safe_read() {
    FakeMem m;
    uint8_t buf[64];
    SafeReadResult r = safe_read(m.fn(), 0x100010, buf, 16);
    CHECK(r.status == ReadStatus::Ok && r.bytes_ok == 16 && buf[0] == 0x10 && buf[15] == 0x1f);
    // Raw small values and the null page never reach the reader.
    m.calls = 0;
    CHECK(safe_read(m.fn(), 0x21, buf, 8).status == ReadStatus::NullPage);
    CHECK(safe_read(m.fn(), 0, buf, 8).status == ReadStatus::NullPage);
    CHECK(safe_read(m.fn(), 0xffff800000000000ull, buf, 8).status == ReadStatus::OutOfRange);
    CHECK(safe_read(m.fn(), 0x00007fffffffffffull, buf, 8).status == ReadStatus::Overflow);
    CHECK(safe_read(m.fn(), 0xffffffffffffffffull, buf, 8).status == ReadStatus::OutOfRange);
    CHECK(m.calls == 0);
    // Unmapped.
    CHECK(safe_read(m.fn(), 0x200000, buf, 8).status == ReadStatus::Unmapped);
    // A range that runs off the end of the mapping: readable prefix reported, page-wise.
    m.lo = 0x100000;
    m.hi = 0x101000;
    std::memset(buf, 0, sizeof(buf));
    r = safe_read(m.fn(), 0x100ff0, buf, 64);
    CHECK(r.status == ReadStatus::Partial && r.bytes_ok == 16 && buf[0] == 0xf0 && buf[15] == 0xff);
    // Zero length and a missing reader are harmless.
    CHECK(safe_read(m.fn(), 0x21, buf, 0).status == ReadStatus::Ok);
    CHECK(safe_read(ProbeReadFn(), 0x100000, buf, 8).status == ReadStatus::Unmapped);
}

void test_format_line() {
    FakeMem m;
    ProbeSpec s;
    std::string e;
    CHECK(parse_probe_spec("0x140001000:fn:args=6:ret:mem=rcx+4,4:mem=rdx,8:mem=r8,4", &s, &e));
    ProbeHit h;
    h.seq = 7;
    h.t_ms = 1234;
    h.thread = 3;
    h.source = "entry";
    h.regs[1] = 0x100100;  // rcx: mapped
    h.regs[2] = 0x21;      // rdx: a raw value, not a pointer
    h.regs[8] = 0x1fffe;   // r8: null page neighbour... below 0x100000 but above 0x10000, unmapped
    h.regs[4] = 0x101000 - 0x40;  // rsp: caller slot and stack args readable
    const std::string line = format_probe_line(s, 0x140001000ull, h, m.fn());
    CHECK(line.find("\"seq\":7") != std::string::npos && line.find("\"thread\":3") != std::string::npos);
    CHECK(line.find("\"src\":\"entry\"") != std::string::npos && line.find("\"name\":\"fn\"") != std::string::npos);
    CHECK(line.find("\"caller\":\"0x") != std::string::npos);
    CHECK(line.find("\"args\":[\"0x100100\",\"0x21\"") != std::string::npos);
    CHECK(line.find("\"ret\":\"pending\"") != std::string::npos);
    CHECK(line.find("\"reg\":\"rcx\",\"off\":4,\"addr\":\"0x100104\",\"len\":4,\"status\":\"ok\",\"hex\":\"04050607\"") !=
          std::string::npos);
    CHECK(line.find("\"reg\":\"rdx\",\"off\":0,\"addr\":\"0x21\",\"len\":8,\"status\":\"null_page\"") != std::string::npos);
    CHECK(line.find("\"reg\":\"r8\"") != std::string::npos && line.find("\"status\":\"unmapped\"") != std::string::npos);
    CHECK(line.front() == '{' && line.back() == '}' && line.find('\n') == std::string::npos);
    // A bogus stack pointer: no caller, null stack args, still a well-formed line.
    h.regs[4] = 0x21;
    const std::string bad = format_probe_line(s, 0x140001000ull, h, m.fn());
    CHECK(bad.find("\"caller\":null") != std::string::npos && bad.back() == '}');
}

void test_probe_set() {
    ProbeSet set;
    CHECK(!set.active() && set.find(0x1000) == -1);
    std::vector<ProbeSpec> specs(2);
    std::string e;
    CHECK(parse_probe_spec("0x2000:b:max=2:args=1", &specs[0], &e));
    CHECK(parse_probe_spec("0x1000:a:args=2", &specs[1], &e));
    const std::string path = "test_guest_probe_out.jsonl";
    set.configure(specs, path, 100);
    CHECK(set.active() && set.size() == 2);
    CHECK(set.find(0x1000) == 0 && set.find(0x2000) == 1 && set.find(0x1800) == -1 && set.find(0x3000) == -1 &&
          set.find(0x10) == -1);
    FakeMem m;
    ProbeHit h;
    h.regs[4] = 0x100100;
    for (int i = 0; i < 5; ++i) {
        if (set.count_hit(1, i % 2 == 0)) set.log_hit(1, h, m.fn());
    }
    CHECK(set.calls(1) == 5 && set.logged(1) == 2 && set.entry_calls(1) == 3);  // max=2: rate limited
    const std::string json = set.to_json();
    CHECK(json.find("\"calls\": 5") != std::string::npos && json.find("\"rate_limited\": 3") != std::string::npos);
    set.clear();
    std::ifstream in(path);
    std::string l1, l2, l3;
    CHECK(std::getline(in, l1) && std::getline(in, l2) && !std::getline(in, l3));
    CHECK(l1.find("\"seq\":1") != std::string::npos && l2.find("\"seq\":2") != std::string::npos);
    in.close();
    std::remove(path.c_str());
    // The global budget limits all probes together.
    std::vector<ProbeSpec> one(1);
    CHECK(parse_probe_spec("0x1000:a", &one[0], &e));
    set.configure(one, "", 3);
    int approved = 0;
    for (int i = 0; i < 10; ++i) approved += set.count_hit(0, false) ? 1 : 0;
    CHECK(approved == 3 && set.calls(0) == 10);
}

void test_frames_and_exit_lines() {
    ProbeFrames f;
    ProbeFrames::Frame out;
    // A nested call chain: A (rsp 0x1000) calls B (0xff0) which calls A again (0xfe0, recursion).
    f.push(0, 0xA, 0x1000, 1);
    f.push(1, 0xB, 0xff0, 2);
    f.push(0, 0xA, 0xfe0, 3);
    CHECK(f.depth() == 3);
    CHECK(f.pop(0xA, 0xfe8, &out) && out.entry_seq == 3 && f.depth() == 2);  // the inner A, not the outer
    CHECK(f.pop(0xB, 0xff8, &out) && out.entry_seq == 2 && out.idx == 1);
    CHECK(f.pop(0xA, 0x1008, &out) && out.entry_seq == 1 && f.depth() == 0);
    // An exit nobody pushed (the entry ran before the probe was armed) stays unpaired.
    CHECK(!f.pop(0xA, 0x1008, &out));
    // An unwind past C and D: they never exit. The caller's exit drops them.
    f.push(0, 0xA, 0x2000, 10);
    f.push(2, 0xC, 0x1ff0, 11);
    f.push(3, 0xD, 0x1fe0, 12);
    CHECK(f.pop(0xA, 0x2008, &out) && out.entry_seq == 10 && f.depth() == 0 && f.dropped() == 2);
    // A stale frame (its owner was unwound) is dropped when a new activation starts above it, so a
    // later exit of the same function at the same depth pairs with the new entry.
    f.push(1, 0xB, 0x3000, 20);
    f.push(1, 0xB, 0x3000, 21);  // same slot again: the first one was unwound
    CHECK(f.depth() == 1 && f.pop(0xB, 0x3008, &out) && out.entry_seq == 21);
    // Wrong rsp: no frame matches, the stack is untouched.
    f.push(0, 0xA, 0x4000, 30);
    CHECK(!f.pop(0xA, 0x4010, &out) && f.depth() == 1);
    // The depth is bounded.
    ProbeFrames deep;
    for (uint64_t i = 0; i < ProbeFrames::kMaxDepth + 10; ++i) deep.push(0, 0xA, 0x100000 - i * 16, i + 1);
    CHECK(deep.depth() == ProbeFrames::kMaxDepth);

    ProbeSpec s;
    std::string e;
    CHECK(parse_probe_spec("0x1000:fn:ret", &s, &e));
    const uint64_t xmm0[2] = {0x3ff0000000000000ull, 0};
    const std::string line = format_probe_exit_line(s, 0x1000, 9, 55, 2, 4, true, 0xdead, xmm0);
    CHECK(line == "{\"seq\":9,\"t_ms\":55,\"thread\":2,\"src\":\"exit\",\"name\":\"fn\",\"addr\":\"0x1000\","
                  "\"entry_seq\":4,\"paired\":true,\"rax\":\"0xdead\",\"xmm0\":[\"0x3ff0000000000000\",\"0x0\"]}");
    const std::string loose = format_probe_exit_line(s, 0x1000, 10, 56, 2, 0, false, 1, xmm0);
    CHECK(loose.find("\"entry_seq\":null,\"paired\":false") != std::string::npos);

    // Through the probe set: the exit line names the entry line's seq.
    ProbeSet set;
    std::vector<ProbeSpec> specs(1);
    CHECK(parse_probe_spec("0x1000:fn:ret", &specs[0], &e));
    const std::string path = "test_guest_probe_exit.jsonl";
    set.configure(specs, path, 100);
    FakeMem m;
    ProbeHit h;
    h.regs[4] = 0x100100;
    CHECK(set.count_hit(0, true));
    const uint64_t entry_seq = set.log_hit(0, h, m.fn());
    CHECK(entry_seq == 1);
    set.log_exit(0, 1, 5, entry_seq, true, 0x2a, xmm0);
    set.log_exit(0, 1, 6, 0, false, 0x2b, xmm0);
    CHECK(set.exit_calls(0) == 2 && set.exit_unpaired(0) == 1);
    CHECK(set.to_json().find("\"exit_calls\": 2, \"exit_unpaired\": 1") != std::string::npos);
    set.clear();
    std::ifstream in(path);
    std::string l1, l2, l3, l4;
    CHECK(std::getline(in, l1) && std::getline(in, l2) && std::getline(in, l3) && !std::getline(in, l4));
    CHECK(l2.find("\"src\":\"exit\"") != std::string::npos && l2.find("\"entry_seq\":1") != std::string::npos &&
          l2.find("\"rax\":\"0x2a\"") != std::string::npos);
    CHECK(l3.find("\"paired\":false") != std::string::npos);
    in.close();
    std::remove(path.c_str());
}

}  // namespace

int main() {
    test_parse();
    test_safe_read();
    test_format_line();
    test_probe_set();
    test_frames_and_exit_lines();
    if (g_failures == 0) {
        std::puts("test_guest_probe: ok");
    }
    return g_failures == 0 ? 0 : 1;
}
