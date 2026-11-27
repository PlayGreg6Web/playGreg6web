#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "guest/audio_probes.h"
#include "guest/hash_registry_scan.h"
#include "os/audio_file_stats.h"
#include "os/audio_stream_log.h"

namespace {
int g_failures = 0;
#define CHECK(expr)                                                                        \
    do {                                                                                   \
        if (!(expr)) {                                                                     \
            std::fprintf(stderr, "FAIL [test_audio_probes.cpp:%d]: %s\n", __LINE__, #expr); \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

using namespace playgta6::guest;

void test_stream_log() {
    namespace asl = playgta6::os::audio_stream_log;
    asl::Log l;
    l.set_sources([]() -> uint64_t { return 1500; }, []() -> uint64_t { return 7; });
    for (int i = 0; i < 100; ++i) {
        l.record("xapu_voice_start", 1, static_cast<uint64_t>(i), 1);
    }
    l.record("probe_entry_stream_decoder_create", 0x10);
    CHECK(l.count("xapu_voice_start") == 100);
    CHECK(l.count("missing") == 0);
    const std::string j = l.to_json();
    CHECK(j.find("\"count\": 100, \"first_ms\": 1500, \"first_present\": 7") != std::string::npos);
    CHECK(j.find("probe_entry_stream_decoder_create") != std::string::npos);
    l.reset();
    CHECK(l.count("xapu_voice_start") == 0);
}

void test_probes() {
    AudioProbes p;
    CHECK(!p.active());
    const uint64_t none[4] = {0, 0, 0, 0};
    CHECK(p.hit(0x1000, none) == -1);
    p.configure({{"factory", 0x5000}, {"init", 0x2000}, {"dup", 0x2000}, {"zero", 0}});
    CHECK(p.active() && p.size() == 2);
    CHECK(p.find("factory") >= 0 && p.find("zero") == -1);
    const uint64_t opus[4] = {0x7ff00000, 11, 2, 0x100000000ull};
    const uint64_t xma[4] = {0x7ff00000, 5, 2, 0};
    bool first = false;
    const long f = p.hit(0x5000, opus, &first);
    CHECK(f == p.find("factory") && first);
    CHECK(p.hit(0x5000, opus, &first) == f && !first);
    p.hit(0x5000, xma);
    CHECK(p.hit(0x5001, opus) == -1 && p.hit(0x1fff, opus) == -1 && p.hit(0x9000, opus) == -1);
    const size_t fi = static_cast<size_t>(f);
    CHECK(p.calls(fi) == 3);
    CHECK(p.small_value(fi, 1, 11) == 2 && p.small_value(fi, 1, 5) == 1);
    CHECK(p.small_value(fi, 2, 2) == 3);
    CHECK(p.small_value(fi, 3, 0) == 1);  // xma call; the 1<<32 value is not tallied
    CHECK(p.calls(static_cast<size_t>(p.find("init"))) == 0);
    const std::string j = p.to_json();
    CHECK(j.find("\"name\": \"factory\"") != std::string::npos);
    CHECK(j.find("\"calls\": 3") != std::string::npos);
    CHECK(j.find("\"rdx\": {\"5\": 1, \"11\": 2}") != std::string::npos);
    CHECK(j.find("\"indirect_calls_only\": true") != std::string::npos);

    // Entry counters (lifter-emitted probes) are separate from the dispatch counts.
    const uint64_t init_args[4] = {1, 0x7ff00000, 3, 0};
    bool efirst = false;
    const long e = p.entry_hit(0x2000, init_args, &efirst);
    CHECK(e == p.find("init") && efirst);
    CHECK(p.entry_hit(0x2000, init_args, &efirst) == e && !efirst);
    CHECK(p.entry_hit(0x2001, init_args) == -1);
    const size_t ei = static_cast<size_t>(e);
    CHECK(p.entry_calls(ei) == 2 && p.calls(ei) == 0);
    CHECK(p.entry_small_value(ei, 0, 1) == 2 && p.entry_small_value(ei, 2, 3) == 2);
    CHECK(p.entry_calls(fi) == 0);
    CHECK(p.to_json().find("\"entry_calls\": 2") != std::string::npos);

    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t) {
        ts.emplace_back([&] {
            for (int i = 0; i < 5000; ++i) p.hit(0x2000, none);
        });
    }
    for (auto& t : ts) t.join();
    CHECK(p.calls(static_cast<size_t>(p.find("init"))) == 20000);
    p.clear();
    CHECK(!p.active() && p.hit(0x5000, opus) == -1);
}

void test_files() {
    namespace af = playgta6::os::audio_files;
    CHECK(af::path_names_audio("x:/Game/Audio/sfx.rpf"));
    CHECK(af::path_names_audio("pack/AUDIO_music.rpf"));
    CHECK(!af::path_names_audio("x/common.rpf") && !af::path_names_audio("aud"));
    af::Stats s;
    s.record("a/common.rpf", true);
    s.record("a/audio1.rpf", true);
    s.record("a/audio2.rpf", false);
    CHECK(s.opens_total == 3 && s.audio_opens == 2 && s.audio_open_ok == 1 && s.audio_open_failed == 1);
    CHECK(s.failed_hash_count == 1 && s.failed_hash[0] == af::fnv1a("a/audio2.rpf"));
    const std::string j = s.to_json();
    CHECK(j.find("\"audio_open_failed\": 1") != std::string::npos && j.find("audio2") == std::string::npos);
    s.reset();
    CHECK(s.audio_opens == 0 && s.failed_hash_count == 0);
}
}  // namespace

// The registry scan reads groups of 16 control bytes + 16 {key, value} slots; a high control bit
// marks an empty slot (its key is ignored even if it matches).
void test_hash_registry_scan() {
    std::vector<uint8_t> mem(2 * kHashRegistryGroupBytes, 0);
    for (int s = 0; s < 16; ++s) {
        mem[s] = 0x80;
        mem[kHashRegistryGroupBytes + s] = 0x80;
    }
    auto put = [&](size_t group, int slot, uint8_t ctrl, uint32_t key, uint64_t value) {
        uint8_t* g = mem.data() + group * kHashRegistryGroupBytes;
        g[slot] = ctrl;
        uint8_t* e = g + 0x10 + slot * 0x10;
        for (int i = 0; i < 4; ++i) e[i] = static_cast<uint8_t>(key >> (8 * i));
        for (int i = 0; i < 8; ++i) e[8 + i] = static_cast<uint8_t>(value >> (8 * i));
    };
    put(0, 3, 0x21, 0x11111111u, 0x7ff000001000ull);
    put(1, 15, 0x05, 0x22222222u, 0);
    put(1, 0, 0x80, 0x33333333u, 0x1234);  // empty: not counted
    const uint64_t base = 0x10000;
    auto read = [&](uint64_t va, void* dst, size_t len) {
        if (va < base || va + len > base + mem.size()) return false;
        std::memcpy(dst, mem.data() + (va - base), len);
        return true;
    };
    const auto r = scan_hash_registry(read, base, 2, {0x11111111u, 0x22222222u, 0x33333333u});
    CHECK(r.readable && r.full_slots == 2);
    CHECK(r.present[0] && !r.value_null[0]);
    CHECK(r.present[1] && r.value_null[1]);
    CHECK(!r.present[2]);
    CHECK(!scan_hash_registry(read, base, 3, {1}).readable);  // third group unreadable
    const auto empty = scan_hash_registry(read, 0, 0, {1});
    CHECK(empty.readable && empty.full_slots == 0 && !empty.present[0]);
}

int main() {
    test_probes();
    test_stream_log();
    test_hash_registry_scan();
    test_files();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_audio_probes FAILED with %d failure(s).\n", g_failures);
        return 1;
    }
    std::printf("test_audio_probes PASSED\n");
    return 0;
}
