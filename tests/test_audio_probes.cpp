#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "guest/audio_probes.h"
#include "os/audio_file_stats.h"

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

int main() {
    test_probes();
    test_files();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_audio_probes FAILED with %d failure(s).\n", g_failures);
        return 1;
    }
    std::printf("test_audio_probes PASSED\n");
    return 0;
}
