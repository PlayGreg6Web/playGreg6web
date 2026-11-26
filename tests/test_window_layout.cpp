#include "gpu/native_exec/window_layout.h"

#include <cstdio>

using namespace playgta6::gpu::native_exec;

static int g_fail = 0;
#define CHECK(c)                                                  \
    do {                                                          \
        if (!(c)) {                                               \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                             \
        }                                                         \
    } while (0)

int main() {
    CHECK(letterbox_rect(1920, 1080, 1280, 720) == (PresentRect{0, 0, 1280, 720}));
    CHECK(letterbox_rect(1920, 1080, 2000, 1080) == (PresentRect{40, 0, 1920, 1080}));
    CHECK(letterbox_rect(1920, 1080, 1920, 1200) == (PresentRect{0, 60, 1920, 1080}));
    CHECK(letterbox_rect(1024, 768, 1920, 1080) == (PresentRect{240, 0, 1440, 1080}));
    CHECK(letterbox_rect(0, 1080, 100, 100).w == 0);
    CHECK(letterbox_rect(1920, 1080, 0, 100).h == 0);
    {
        const PresentRect r = letterbox_rect(3840, 2160, 1, 1);
        CHECK(r.w == 1 && r.h == 1 && r.x == 0 && r.y == 0);
        const PresentRect big = letterbox_rect(1, 1, 0xFFFFFFFFu, 100);
        CHECK(big.w == 100 && big.h == 100);
    }
    {
        const PresentRect r = letterbox_rect(1920, 1080, 1920, 1200);
        int32_t sx = 0, sy = 0;
        CHECK(!client_to_source(r, 1920, 1080, 10, 10, &sx, &sy));
        CHECK(client_to_source(r, 1920, 1080, 960, 600, &sx, &sy) && sx == 960 && sy == 540);
    }
    CHECK(scale_for_dpi(1280, 96) == 1280);
    CHECK(scale_for_dpi(1280, 144) == 1920);
    CHECK(scale_for_dpi(100, 0) == 100);
    CHECK(scale_for_dpi(1, 120) == 1);

    {
        SavedWindowState s{-300, 40, 1280, 720, true}, o;
        CHECK(parse_window_state(format_window_state(s), &o) && o == s);
        CHECK(!parse_window_state("", &o));
        CHECK(!parse_window_state("w=1280\n", &o));
        CHECK(!parse_window_state("w=5\nh=5\n", &o));
        CHECK(parse_window_state("junk\nw=800\nh=600\nfoo=1\n", &o) && o.w == 800 && o.h == 600 && !o.maximized);
    }
    {
        const std::vector<MonitorRect> one{{0, 0, 1920, 1080}};
        SavedWindowState s{100, 100, 800, 600, false};
        CHECK(fit_state_to_monitors(&s, one) && s.x == 100 && s.y == 100 && s.w == 800);
        SavedWindowState gone{2500, 100, 800, 600, false};  // second monitor unplugged
        CHECK(fit_state_to_monitors(&gone, one) && gone.x == 560 && gone.y == 240);
        SavedWindowState huge{0, 0, 5000, 4000, false};
        CHECK(fit_state_to_monitors(&huge, one) && huge.w == 1920 && huge.h == 1080 && huge.x == 0);
        SavedWindowState edge{1900, 1070, 800, 600, false};  // barely visible
        CHECK(fit_state_to_monitors(&edge, one) && edge.x == 560);
        const std::vector<MonitorRect> two{{0, 0, 1920, 1080}, {1920, 0, 3840, 1080}};
        SavedWindowState on2{2000, 50, 800, 600, false};
        CHECK(fit_state_to_monitors(&on2, two) && on2.x == 2000 && on2.y == 50);
        CHECK(!fit_state_to_monitors(&on2, {}));
    }
    if (g_fail) return 1;
    std::printf("test_window_layout: ok\n");
    return 0;
}
