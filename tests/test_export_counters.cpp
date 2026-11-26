#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "guest/export_counters.h"

namespace {

int g_failures = 0;

#define CHECK(expr)                                                                   \
    do {                                                                              \
        if (!(expr)) {                                                                \
            std::fprintf(stderr, "FAIL [test_export_counters.cpp:%d]: %s\n", __LINE__, #expr); \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

using namespace playgta6::guest;

constexpr uint64_t kBase = 0x180000000ull;

// A minimal PE64 image: DOS header, NT headers, an export directory with three named exports
// (given out of address order), one empty slot and one forwarder.
std::vector<uint8_t> make_image() {
    std::vector<uint8_t> img(0x1000, 0);
    auto put32 = [&](size_t off, uint32_t v) { std::memcpy(&img[off], &v, 4); };
    auto put16 = [&](size_t off, uint16_t v) { std::memcpy(&img[off], &v, 2); };
    put16(0, 0x5A4D);
    put32(0x3C, 0x80);
    put32(0x80, 0x4550);
    put16(0x80 + 0x18, 0x20B);
    const uint32_t dir_rva = 0x200;
    const uint32_t dir_size = 0x200;
    put32(0x80 + 0x18 + 0x70, dir_rva);
    put32(0x80 + 0x18 + 0x74, dir_size);
    // IMAGE_EXPORT_DIRECTORY
    const uint32_t funcs = 0x240, names = 0x280, ords = 0x2A0;
    put32(dir_rva + 20, 5);  // functions
    put32(dir_rva + 24, 4);  // names
    put32(dir_rva + 28, funcs);
    put32(dir_rva + 32, names);
    put32(dir_rva + 36, ords);
    // The directory spans 0x200..0x400; a forwarder RVA lies inside it.
    const uint32_t fixed[5] = {0x3000, 0x2000, 0, 0x300, 0x2800};
    for (int i = 0; i < 5; ++i) {
        put32(funcs + 4 * i, fixed[i]);
    }
    const char* nm[4] = {"BinkOpenWithOptions", "BinkDoFrame", "BinkFwd", "BinkClose"};
    const uint16_t ordinal[4] = {0, 1, 3, 4};
    uint32_t str = 0x500;
    for (int i = 0; i < 4; ++i) {
        put32(names + 4 * i, str);
        put16(ords + 2 * i, ordinal[i]);
        std::memcpy(&img[str], nm[i], std::strlen(nm[i]) + 1);
        str += 0x20;
    }
    return img;
}

void test_enumerate() {
    const std::vector<uint8_t> img = make_image();
    auto read = [&](uint64_t va, void* dst, size_t n) {
        if (va < kBase || va - kBase + n > img.size()) {
            return false;
        }
        std::memcpy(dst, &img[va - kBase], n);
        return true;
    };
    std::vector<PeExport> out;
    CHECK(enumerate_pe_exports(read, kBase, out));
    CHECK(out.size() == 3);  // the forwarder is skipped
    bool open = false, frame = false, close = false;
    for (const PeExport& e : out) {
        open |= e.name == "BinkOpenWithOptions" && e.address == kBase + 0x3000;
        frame |= e.name == "BinkDoFrame" && e.address == kBase + 0x2000;
        close |= e.name == "BinkClose" && e.address == kBase + 0x2800;
    }
    CHECK(open && frame && close);

    std::vector<uint8_t> broken = img;
    broken[0x80] = 0;  // no PE signature
    auto read_broken = [&](uint64_t va, void* dst, size_t n) {
        if (va < kBase || va - kBase + n > broken.size()) {
            return false;
        }
        std::memcpy(dst, &broken[va - kBase], n);
        return true;
    };
    CHECK(!enumerate_pe_exports(read_broken, kBase, out));
    CHECK(out.empty());
}

void test_counters() {
    ExportCounters c;
    CHECK(!c.active());
    CHECK(c.hit(kBase + 0x2000) == -1);

    c.configure("Bink2Scarlett.dll", {{"BinkOpenWithOptions", kBase + 0x3000},
                                      {"BinkDoFrame", kBase + 0x2000},
                                      {"BinkClose", kBase + 0x2800},
                                      {"NoAddress", 0}});
    CHECK(c.active());
    CHECK(c.size() == 3);
    CHECK(c.module() == "Bink2Scarlett.dll");

    bool first = false;
    const long open = c.find("BinkOpenWithOptions");
    CHECK(open >= 0);
    CHECK(c.hit(kBase + 0x3000, &first) == open && first);
    CHECK(c.hit(kBase + 0x3000, &first) == open && !first);
    CHECK(c.calls(static_cast<size_t>(open)) == 2);
    CHECK(c.hit(kBase + 0x2001) == -1);  // inside the range, not an entry
    CHECK(c.hit(kBase + 0x1000) == -1);  // below
    CHECK(c.hit(kBase + 0x4000) == -1);  // above
    CHECK(c.calls(static_cast<size_t>(c.find("BinkDoFrame"))) == 0);
    CHECK(c.find("BinkNextFrame") == -1);

    // Concurrent counting loses no call.
    const long frame = c.find("BinkDoFrame");
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 10000; ++i) {
                c.hit(kBase + 0x2000);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    CHECK(c.calls(static_cast<size_t>(frame)) == 40000);

    c.clear();
    CHECK(!c.active());
    CHECK(c.hit(kBase + 0x3000) == -1);
}

}  // namespace

int main() {
    test_enumerate();
    test_counters();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_export_counters FAILED with %d failure(s).\n", g_failures);
        return 1;
    }
    std::printf("test_export_counters PASSED\n");
    return 0;
}
