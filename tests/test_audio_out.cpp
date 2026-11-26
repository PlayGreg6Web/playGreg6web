// --audio-out: resampler, jitter buffer and underrun accounting (no device needed).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "os/audio_out.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_audio_out.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
}  // namespace

int main() {
    namespace ao = playgta6::os::audio_out;

    {  // passthrough keeps samples
        ao::Resampler r;
        r.configure(48000, 48000);
        std::vector<float> in = {1, 2, 3, 4}, out;
        r.process(in.data(), 2, out);
        CHECK(out == in);
    }
    {  // 48k -> 44.1k yields ~ratio frames across chunked input, and a constant stays constant
        ao::Resampler r;
        r.configure(48000, 44100);
        std::vector<float> in(2 * 480, 0.5f), out;
        for (int i = 0; i < 100; ++i) r.process(in.data(), 480, out);
        const double frames = static_cast<double>(out.size() / 2);
        CHECK(std::fabs(frames - 48000.0 * 44100.0 / 48000.0) < 4.0);
        bool flat = true;
        for (size_t i = 2; i < out.size(); ++i) flat = flat && std::fabs(out[i] - 0.5f) < 1e-5f;
        CHECK(flat);
    }
    {  // 48k -> 96k doubles
        ao::Resampler r;
        r.configure(48000, 96000);
        std::vector<float> in(2 * 100, 0.25f), out;
        r.process(in.data(), 100, out);
        CHECK(out.size() / 2 == 200);
    }
    {  // jitter buffer: no output before prebuffer, no underrun counted while priming
        ao::JitterBuffer jb;
        jb.configure(100, 20);
        std::vector<float> a(2 * 10, 1.0f), o(2 * 10, 9.0f);
        jb.push(a.data(), 10);
        CHECK(jb.pop(o.data(), 10) == 0);
        CHECK(o[0] == 0.0f);
        CHECK(jb.stats().underruns == 0);
        jb.push(a.data(), 10);  // now primed (20 frames)
        CHECK(jb.pop(o.data(), 10) == 10);
        CHECK(o[0] == 1.0f && o[19] == 1.0f);
        CHECK(jb.pop(o.data(), 10) == 10);
        // drained: next pop underruns, silence inserted and counted once
        CHECK(jb.pop(o.data(), 10) == 0);
        CHECK(o[0] == 0.0f);
        CHECK(jb.stats().underruns == 1);
        CHECK(jb.stats().underrun_frames == 10);
        // re-priming: a small push is not enough, still silence but not a new underrun event
        jb.push(a.data(), 10);
        CHECK(jb.pop(o.data(), 10) == 0);
        CHECK(jb.stats().underruns == 1);
        CHECK(jb.stats().underrun_frames == 20);
    }
    {  // partial underrun: real frames then silence
        ao::JitterBuffer jb;
        jb.configure(100, 5);
        std::vector<float> a(2 * 5, 1.0f), o(2 * 8, 9.0f);
        jb.push(a.data(), 5);
        CHECK(jb.pop(o.data(), 8) == 5);
        CHECK(o[9] == 1.0f && o[10] == 0.0f && o[15] == 0.0f);
        CHECK(jb.stats().underruns == 1 && jb.stats().underrun_frames == 3);
    }
    {  // overrun drops, wrap-around keeps order
        ao::JitterBuffer jb;
        jb.configure(8, 1);
        std::vector<float> a(2 * 12);
        for (size_t i = 0; i < 12; ++i) a[2 * i] = a[2 * i + 1] = static_cast<float>(i);
        jb.push(a.data(), 12);
        CHECK(jb.stats().overrun_frames == 4 && jb.buffered() == 8);
        std::vector<float> o(2 * 5);
        CHECK(jb.pop(o.data(), 5) == 5);
        jb.push(a.data() + 2 * 8, 4);
        std::vector<float> o2(2 * 7);
        CHECK(jb.pop(o2.data(), 7) == 7);
        CHECK(o[8] == 4.0f && o2[0] == 5.0f && o2[4] == 7.0f && o2[6] == 8.0f && o2[12] == 11.0f);
    }
    {  // feed: wrong rate ignored, right rate lands in the buffer
        ao::Feed f;
        f.configure(44100, 200, 10);
        std::vector<float> a(2 * 480, 0.1f);
        f.push(a.data(), 480, 44100);
        CHECK(f.buffer().stats().frames_in == 0);
        f.push(a.data(), 480, 48000);
        CHECK(f.buffer().stats().frames_in > 400);
    }
    {
        const std::string j = ao::report_json(false, 0, ao::Stats{}, "no \"device\"");
        CHECK(j.find("\"underruns\": 0") != std::string::npos);
        CHECK(j.find("\"device_open\": false") != std::string::npos);
        CHECK(j.find("\"render_channels\": 2") != std::string::npos);
        const std::string k = ao::report_json(true, 48000, ao::Stats{}, "", 6, 0x3F);
        CHECK(k.find("\"device_channels\": 6") != std::string::npos);
        CHECK(k.find("\"device_channel_mask\": 63") != std::string::npos);
    }

    if (g_failures == 0) std::printf("test_audio_out: all passed\n");
    return g_failures == 0 ? 0 : 1;
}
