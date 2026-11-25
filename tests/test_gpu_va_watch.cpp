// Synthetic tests for the GPU address watch (gpu/gpu_va_watch.h, harness --watch-gpu-va).
#include <cstdio>
#include <string>

#include "gpu/gpu_va_watch.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_gpu_va_watch.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

namespace w = playgta6::gpu::va_watch;

void test_spec() {
    std::string error;
    w::reset();
    CHECK(!w::active());
    CHECK(w::configure("", &error) && !w::active());
    CHECK(w::configure("0x1000-0x2000,0x8000+0x100,textures:3840x2160/1920x1080", &error) && w::active());
    CHECK(!w::configure("0x2000-0x1000", &error) && error.find("bad range") != std::string::npos);
    CHECK(!w::configure("0x1000", &error));
    CHECK(!w::configure("textures:12", &error) && error.find("bad texture size") != std::string::npos);
    CHECK(!w::configure("0xZZ+4", &error));
    w::reset();
}

void test_ranges() {
    std::string error;
    w::reset();
    CHECK(w::configure("0x1000-0x2000,0x8000+256", &error));
    w::set_frame(7);
    w::touch("root SRV (graphics)", 0x1800, 16, "root parameter 2");
    w::touch("root SRV (graphics)", 0x1800, 16, "root parameter 2");  // counted, logged once
    w::touch("root CBV (graphics)", 0x0fff, 2, "overlaps the start");
    w::touch("table slot (image)", 0x2000, 1, "just past the end");  // not watched
    w::touch("upload", 0x8080, 0, "size 0 counts as one byte");
    w::touch("upload", 0, 4096, "address 0 is never watched");
    const auto ev = w::events();
    CHECK(ev.size() == 3);
    if (ev.size() == 3) {
        CHECK(ev[0].kind == "root SRV (graphics)" && ev[0].count == 2 && ev[0].first_frame == 7 &&
              ev[0].range == "0x1000-0x2000");
        CHECK(ev[1].kind == "root CBV (graphics)" && ev[1].va == 0x0fff);
        CHECK(ev[2].kind == "upload" && ev[2].range == "0x8000+256");
    }
    const std::string s = w::summary();
    CHECK(s.find("3 distinct reference(s)") != std::string::npos && s.find("2x, first frame 7") != std::string::npos);
    w::reset();
    CHECK(w::summary().find("nothing referenced") != std::string::npos);
}

void test_texture_selector() {
    std::string error;
    w::reset();
    CHECK(w::configure("textures:64x32", &error));
    w::touch("upload", 0x40000, 16, "before any texture: nothing watched");
    w::note_texture(0x40000, 8192, 64, 32, 61, 12);  // watched from now on
    w::note_texture(0x90000, 8192, 32, 32, 61, 13);  // other size: not watched
    w::touch("descriptor SRV", 0x40000, 8192, "CreateShaderResourceView");
    w::touch("descriptor SRV", 0x90000, 8192, "CreateShaderResourceView");
    const auto ev = w::events();
    CHECK(ev.size() == 2);
    if (ev.size() == 2) {
        CHECK(ev[0].kind == "texture" && ev[0].range == "texture 64x32 id 12" &&
              ev[0].detail.find("DXGI format 61") != std::string::npos);
        CHECK(ev[1].kind == "descriptor SRV" && ev[1].va == 0x40000);
    }
    // A texture created over an explicitly watched range is reported too.
    w::reset();
    CHECK(w::configure("0x50000+0x1000", &error));
    w::note_texture(0x50000, 0x1000, 16, 16, 28, 5);
    CHECK(w::events().size() == 1 && w::events()[0].detail.find("created over the range") != std::string::npos);
    w::reset();
}

void test_descriptor_copies() {
    std::string error;
    w::reset();
    CHECK(w::configure("0x10000+0x1000", &error));
    w::watch_handle(0xA000, 0x10000);
    w::descriptor_copied(0xB000, 0xC000);  // unwatched source: nothing
    w::descriptor_copied(0xA000, 0xD000);
    w::descriptor_copied(0xD000, 0xE000);  // the copy is watched too
    w::descriptor_copied(0xB000, 0xD000);  // overwritten: no longer watched
    w::descriptor_copied(0xD000, 0xF000);
    const auto ev = w::events();
    CHECK(ev.size() == 2 && ev[0].kind == "descriptor copy" && ev[0].detail == "handle 0xa000 -> 0xd000" &&
          ev[1].detail == "handle 0xd000 -> 0xe000" && ev[1].range == "0x10000+0x1000");
    w::reset();
}

}  // namespace

int main() {
    test_spec();
    test_ranges();
    test_texture_selector();
    test_descriptor_copies();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_gpu_va_watch: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_gpu_va_watch: all checks passed\n");
    return 0;
}
