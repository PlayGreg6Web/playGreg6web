// Boot report `audio` section: window bucketing and the JSON members.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "os/audio_report.h"
#include "os/xapu_connect.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_audio_report.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)
}  // namespace

int main() {
    namespace ar = playgta6::os::audio_report;
    CHECK(ar::window_index(0) == 0);
    CHECK(ar::window_index(9999) == 0);
    CHECK(ar::window_index(10000) == 1);
    bool clamped = false;
    CHECK(ar::window_index(ar::kMaxWindows * ar::kWindowMs - 1, &clamped) == ar::kMaxWindows - 1 && !clamped);
    CHECK(ar::window_index(ar::kMaxWindows * ar::kWindowMs * 5, &clamped) == ar::kMaxWindows - 1 && clamped);

    ar::Windows w;
    CHECK(w.nonsilent_windows().empty());
    w.record_at(100, 480, true);
    w.record_at(5000, 480, false);
    w.record_at(25000, 960, true);  // window 2; window 1 stays empty
    const std::vector<uint64_t> v = w.nonsilent_windows();
    CHECK(v.size() == 3 && v[0] == 480 && v[1] == 0 && v[2] == 960);
    w.record_at(ar::kMaxWindows * ar::kWindowMs + 1, 10, true);
    CHECK(w.clamped.load() == 1 && w.nonsilent_windows().size() == ar::kMaxWindows);

    ar::Totals t;
    t.render_frames = 1000;
    t.render_nonzero_frames = 250;
    t.xapu_decoded_packets = 7;
    const std::string json = ar::to_json(t, v, 0);
    CHECK(json.find("\"non_silent_fraction\": 0.250000") != std::string::npos);
    CHECK(json.find("\"xapu_decoded_packets\": 7") != std::string::npos);
    CHECK(json.find("\"non_silent_frames_per_window\": [480, 0, 960]") != std::string::npos);
    // Spatial-only title (no render client): passes count as kSpatialPassFrames frames each.
    ar::Totals s;
    s.spatial_passes = 100;
    s.spatial_nonzero_passes = 25;
    CHECK(ar::non_silent_fraction(s) == 0.25);
    CHECK(ar::to_json(s, {}, 0).find("\"non_silent_fraction\": 0.250000") != std::string::npos);
    CHECK(ar::to_json(s, {}, 0).find("\"spatial_nonzero_passes\": 25") != std::string::npos);
    // Both paths: 1000 render frames (250 non-zero) + 10 passes (5 non-zero) = 4800 + 1000 frames.
    t.spatial_passes = 10;
    t.spatial_nonzero_passes = 5;
    CHECK(ar::non_silent_fraction(t) == (250.0 + 2400.0) / (1000.0 + 4800.0));
    t.xapu_commands = 12;
    t.xapu_unhandled_commands = 1;
    t.xapu_type1 = 2;
    t.xapu_type2 = 2;
    t.xapu_type3 = 7;
    t.xma_frames = 9;
    t.xma_frames_decoded = 8;
    t.xma_bad_frames = 1;
    const std::string json2 = ar::to_json(t, {}, 0);
    CHECK(json2.find("\"xapu_commands\": 12, \"xapu_unhandled_commands\": 1") != std::string::npos);
    CHECK(json2.find("\"xapu_commands_by_type\": {\"1\": 2, \"2\": 2, \"3\": 7, \"5\": 0}") != std::string::npos);
    CHECK(json2.find("\"xma_frames\": 9, \"xma_frames_decoded\": 8, \"xma_bad_frames\": 1") != std::string::npos);
    CHECK(ar::to_json(ar::Totals{}, {}, 0).find("\"non_silent_fraction\": 0.000000") != std::string::npos);

    {  // XApuConnect params as the title sends them (raw dump, menu run)
        namespace xc = playgta6::os::xapu_connect;
        const uint32_t w[6] = {1, 256, 2, 0x104000, 0, 1};
        const xc::Params p = xc::parse(w);
        CHECK(p.outstanding_limit == 512 && p.memory_bytes == 0x104000 && !p.result_event);
        const uint32_t ev[6] = {1, 4, 2, 0x104000, 0, 9};
        CHECK(xc::parse(ev).result_event);
        const uint32_t bad[6] = {1, 0, 0, 2, 0, 0};
        CHECK(xc::parse(bad).outstanding_limit == 0 && xc::parse(bad).memory_bytes == xc::kFixedLayoutBytes);
    }

    if (g_failures == 0) std::printf("test_audio_report: ok\n");
    return g_failures == 0 ? 0 : 1;
}
