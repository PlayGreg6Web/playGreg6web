#include "gpu/native_exec/window_layout.h"
#include "gpu/native_exec/window_script.h"

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
    {  // window script
        std::vector<TimedWindowAction> v;
        std::string err;
        CHECK(parse_window_script("1000:resize=1100x700,2000:minimize,3000:restore,4000:fullscreen,5000:windowed,6000:probe,6000:maximize", &v, &err));
        CHECK(v.size() == 7 && v[0].action.kind == WindowActionKind::Resize && v[0].action.w == 1100 && v[0].action.h == 700);
        CHECK(v[1].at_ms == 2000 && v[1].action.kind == WindowActionKind::Minimize);
        CHECK(!parse_window_script("2000:minimize,1000:restore", &v, &err));
        CHECK(!parse_window_script("100:resize=5x5", &v, &err));
        CHECK(!parse_window_script("100:resize=800", &v, &err));
        CHECK(!parse_window_script("100:resize=800x600x1", &v, &err));
        CHECK(!parse_window_script("x:minimize", &v, &err));
        CHECK(!parse_window_script("100:dance", &v, &err));
        CHECK(parse_window_script("", &v, &err) && v.empty());
        WindowScriptRunner r;
        CHECK(parse_window_script("100:minimize,100:restore,500:probe", &v, &err));
        r.set(v);
        WindowAction a;
        CHECK(!r.pop_due(99, &a));
        CHECK(r.pop_due(100, &a) && a.kind == WindowActionKind::Minimize);
        CHECK(r.pop_due(100, &a) && a.kind == WindowActionKind::Restore);
        CHECK(!r.pop_due(400, &a) && !r.finished());
        CHECK(r.pop_due(9999, &a) && a.kind == WindowActionKind::Probe && r.finished());
    }
    {  // surface reconfigure state machine
        SurfaceState st;
        CHECK(st.plan(1280, 720) == SurfacePlan::Configure);
        st.on_configured(1280, 720);
        CHECK(st.plan(1280, 720) == SurfacePlan::Acquire);
        CHECK(st.plan(1100, 700) == SurfacePlan::Configure);  // resize
        st.on_configured(1100, 700);
        CHECK(st.plan(0, 0) == SurfacePlan::SkipMinimized);  // minimized
        CHECK(st.plan(0, 700) == SurfacePlan::SkipMinimized);
        CHECK(st.configured() && st.minimized_skips() == 2);
        CHECK(st.plan(1100, 700) == SurfacePlan::Acquire);  // restored to the same size: no reconfigure
        st.invalidate();                                    // e.g. format change
        CHECK(st.plan(1100, 700) == SurfacePlan::Configure);
        st.on_configured(1100, 700);
        CHECK(st.plan(1100, 700) == SurfacePlan::Acquire);
        CHECK(st.on_acquire(AcquireStatus::Optimal, 0) == AcquireNext::Use);
        CHECK(st.on_acquire(AcquireStatus::Suboptimal, 0) == AcquireNext::Use);
        CHECK(st.on_acquire(AcquireStatus::Timeout, 0) == AcquireNext::Drop);
        CHECK(st.on_acquire(AcquireStatus::Outdated, 0) == AcquireNext::Reconfigure);
        CHECK(st.on_acquire(AcquireStatus::Lost, 0) == AcquireNext::Reconfigure);
        CHECK(st.retries() == 2);
        CHECK(st.on_acquire(AcquireStatus::Outdated, 1) == AcquireNext::Drop);  // still failing: drop, clean configure next
        CHECK(st.plan(1100, 700) == SurfacePlan::Configure);
        CHECK(st.on_acquire(AcquireStatus::OutOfMemory, 0) == AcquireNext::Fatal);
        CHECK(st.on_acquire(AcquireStatus::DeviceLost, 0) == AcquireNext::Fatal);
        CHECK(st.configures() == 3);
    }
    {  // letterbox probe
        const uint32_t W = 8, H = 6;
        std::vector<uint8_t> px(W * H * 4, 0);
        const PresentRect r{0, 1, 8, 4};  // letterbox rows 1..4
        for (uint32_t y = 1; y < 5; ++y)
            for (uint32_t x = 0; x < W; ++x) px[(y * W + x) * 4 + 1] = 200;
        LetterboxProbe p = probe_letterbox(px.data(), W * 4, W, H, r);
        CHECK(p.inner_pixels == 32 && p.inner_nonblack == 32 && p.bar_pixels == 16 && p.bar_nonblack == 0 && p.ok());
        px[2 * 4] = 9;  // bar pixel (x=2,y=0) lit
        px[2 * 4 + 2] = 9;
        p = probe_letterbox(px.data(), W * 4, W, H, r);
        CHECK(p.bar_nonblack == 1 && !p.ok());
        std::vector<uint8_t> blank(W * H * 4, 0);
        {
            const LetterboxProbe b = probe_letterbox(blank.data(), W * 4, W, H, r);
            CHECK(b.ok() && b.blank());
        }
    }
    if (g_fail) return 1;
    std::printf("test_window_layout: ok\n");
    return 0;
}
