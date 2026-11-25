// --audio-dump WAV writer: header fields and data length after frames pushed through the sink.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "os/audio_downmix.h"
#include "os/audio_wav_dump.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_audio_wav_dump.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

uint32_t rd32(const unsigned char* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24); }
uint16_t rd16(const unsigned char* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
}  // namespace

int main() {
    namespace wd = playgta6::os::audio_wav_dump;
    namespace ad = playgta6::os::audio_downmix;
    const auto path = std::filesystem::temp_directory_path() / "pg6_audio_dump_test.wav";
    const std::string p = path.string();
    {
        wd::Writer w;
        CHECK(w.open(p));
        ad::set_stereo_sink(&wd::Writer::sink, &w);
        std::vector<float> buf(2 * 480);
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<float>(i) / 1000.0f;
        for (int i = 0; i < 3; ++i) wd::Writer::sink(&w, buf.data(), 480, 48000);
        wd::Writer::sink(&w, buf.data(), 480, 44100);  // wrong rate is skipped
        ad::set_stereo_sink(nullptr, nullptr);
        w.close();
        CHECK(w.frames() == 1440);
    }
    std::FILE* f = std::fopen(p.c_str(), "rb");
    CHECK(f);
    if (f) {
        std::vector<unsigned char> d(16384);
        const size_t n = std::fread(d.data(), 1, d.size(), f);
        std::fclose(f);
        CHECK(n == 44 + 1440 * 8);
        CHECK(!std::memcmp(d.data(), "RIFF", 4));
        CHECK(rd32(&d[4]) == 36 + 1440 * 8);
        CHECK(!std::memcmp(&d[8], "WAVEfmt ", 8));
        CHECK(rd32(&d[16]) == 16);
        CHECK(rd16(&d[20]) == 3);
        CHECK(rd16(&d[22]) == 2);
        CHECK(rd32(&d[24]) == 48000);
        CHECK(rd32(&d[28]) == 48000 * 8);
        CHECK(rd16(&d[32]) == 8);
        CHECK(rd16(&d[34]) == 32);
        CHECK(!std::memcmp(&d[36], "data", 4));
        CHECK(rd32(&d[40]) == 1440 * 8);
    }
    std::filesystem::remove(path);
    if (g_failures == 0) std::printf("test_audio_wav_dump: ok\n");
    return g_failures ? 1 : 0;
}
