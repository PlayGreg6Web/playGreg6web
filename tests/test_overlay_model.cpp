// Overlay model (gpu/native_exec/overlay_model.h): glyph table, history ring, text lines, fault wrapping, the
// rasterized bitmap and the CPU composite twin. No GPU.
#include <cstdio>

#include "gpu/native_exec/overlay_model.h"
#include "os/window_hotkeys.h"

using namespace playgta6::gpu::native_exec;

static int g_failures = 0;
#define CHECK(expr)                                                                 \
    do {                                                                            \
        if (!(expr)) {                                                              \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #expr); \
            ++g_failures;                                                           \
        }                                                                           \
    } while (0)

static bool glyph_equal(const uint8_t* a, const uint8_t* b) {
    for (int i = 0; i < 5; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

int main() {
    // Glyphs: lower case folds to upper, out-of-range is '?', every printable glyph has ink except the space.
    CHECK(glyph_equal(overlay_glyph('a'), overlay_glyph('A')));
    CHECK(glyph_equal(overlay_glyph('\x01'), overlay_glyph('?')));
    CHECK(glyph_equal(overlay_glyph('{'), overlay_glyph('?')));
    CHECK(overlay_glyph('A')[0] == 0x7E && overlay_glyph('0')[0] == 0x3E);
    for (int c = 0x21; c <= 0x5F; ++c) {
        int ink = 0;
        for (int k = 0; k < 5; ++k) ink += overlay_glyph(static_cast<char>(c))[k] != 0;
        CHECK(ink > 0);
        for (int k = 0; k < 5; ++k) CHECK((overlay_glyph(static_cast<char>(c))[k] & 0x80) == 0);  // 7 rows
    }
    for (int k = 0; k < 5; ++k) CHECK(overlay_glyph(' ')[k] == 0);

    // History: oldest-first order, wraps at 120, averages over the newest n.
    OverlayHistory h;
    CHECK(h.size() == 0 && h.avg_ms(30) == 0.0);
    for (int i = 0; i < 130; ++i) h.push(i, 10);
    CHECK(h.size() == kOverlayHistory);
    CHECK(h.ms_at(0) == 10.0f && h.ms_at(kOverlayHistory - 1) == 129.0f);
    CHECK(h.avg_ms(2) == 128.5);
    CHECK(h.avg_draws(500) == 10.0);

    // Lines.
    OverlayStatus s;
    s.fps = 59.6;
    s.frame_ms = 16.78;
    s.draws_per_frame = 1234.4;
    s.present_mode = "mailbox";
    std::vector<std::string> l = overlay_lines(s);
    CHECK(l.size() == 5);
    CHECK(l[0] == "60 FPS  16.8 MS" && l[1] == "PRESENT mailbox" && l[2] == "DRAWS 1234/FRAME");
    CHECK(l[3] == "GPU N/A" && l[4] == "AUDIO OFF");
    s.gpu_ms = 5.234;
    s.audio_underruns = 3;
    l = overlay_lines(s);
    CHECK(l[3] == "GPU 5.23 MS" && l[4] == "AUDIO UNDERRUNS 3");
    for (const std::string& t : l) CHECK(static_cast<int>(t.size()) <= kOverlayCols);

    // Fault wrapping.
    CHECK(overlay_fault_lines("").empty());
    std::vector<std::string> f = overlay_fault_lines("access violation");
    CHECK(f.size() == 1 && f[0] == "STOPPED: access violation");
    f = overlay_fault_lines(std::string(500, 'x'));
    CHECK(f.size() == 6 && f.back().size() == static_cast<size_t>(kOverlayCols) &&
          f.back().compare(kOverlayCols - 3, 3, "...") == 0);

    // Bitmap: size grows with the fault rows, the background is 65 % black, text is opaque white, fault text red.
    OverlayBitmap bm = render_overlay(s, h);
    CHECK(bm.w == static_cast<uint32_t>(kOverlayWidth) && bm.h == static_cast<uint32_t>(overlay_height(s)));
    CHECK(bm.rgba.size() == size_t{bm.w} * bm.h * 4);
    CHECK(bm.rgba[0] == 0 && bm.rgba[3] == 166);
    s.fault = "guest fault";
    CHECK(overlay_height(s) == static_cast<int>(bm.h) + 2 + kOverlayLineH);
    OverlayBitmap bf = render_overlay(s, h);
    int white = 0, red = 0;
    for (size_t i = 0; i < bf.rgba.size(); i += 4) {
        if (bf.rgba[i] == 255 && bf.rgba[i + 1] == 255 && bf.rgba[i + 2] == 255 && bf.rgba[i + 3] == 255) ++white;
        if (bf.rgba[i] == 255 && bf.rgba[i + 1] == 90 && bf.rgba[i + 2] == 90 && bf.rgba[i + 3] == 255) ++red;
    }
    CHECK(white > 100 && red > 20);
    // The graph reaches full height for a >= 50 ms frame and a grey baseline exists.
    OverlayHistory slow;
    slow.push(80.0, 0);
    const OverlayBitmap bs = render_overlay(OverlayStatus{}, slow);
    const int gy = kOverlayPad + kOverlayLineH + 2;  // graph top row
    const size_t px = (size_t{static_cast<uint32_t>(gy)} * bs.w + (kOverlayPad + kOverlayHistory - 1)) * 4;
    CHECK(bs.rgba[px] == 255 && bs.rgba[px + 3] == 255);  // red column (> 33 ms) at the newest slot, top row

    // Composite twin: opaque ink replaces, background blends to 35 % of the destination, outside is untouched.
    CHECK(overlay_scale(720) == 1 && overlay_scale(1080) == 2 && overlay_scale(2160) == 3);
    const uint32_t dw = 640, dh = 480;
    std::vector<uint8_t> dst(size_t{dw} * dh * 4, 200);
    overlay_composite(dst, dw, dh, bm, 1, false);
    CHECK(dst[0] == (200 * 89 + 127) / 255 && dst[3] == 166 + (200 * 89 + 127) / 255);
    const size_t far = (size_t{dh - 1} * dw + dw - 1) * 4;
    CHECK(dst[far] == 200 && dst[far + 3] == 200);
    std::vector<uint8_t> dst2(size_t{dw} * dh * 4, 200);
    overlay_composite(dst2, dw, dh, bm, 2, false);  // 2x: pixel (1,1) comes from bitmap pixel (0,0)
    CHECK(dst2[(1 * dw + 1) * 4] == dst[0]);
    // A box larger than the canvas is clipped, not overrun.
    std::vector<uint8_t> tiny(size_t{16} * 16 * 4, 7);
    overlay_composite(tiny, 16, 16, bm, 3, true);

    // The hotkey: F3 by default, in the table with a help line.
    namespace wh = playgta6::os::window_hotkeys;
    const wh::Table t = wh::defaults();
    CHECK(t[static_cast<size_t>(wh::Action::Overlay)] == (wh::Chord{0x72, false}));
    CHECK(wh::find_duplicate(t).empty());
    CHECK(wh::help_text(t).find("overlay = F3") != std::string::npos);

    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_overlay_model: ok\n");
    return 0;
}
