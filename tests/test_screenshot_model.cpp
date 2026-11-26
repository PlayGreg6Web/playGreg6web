// Screenshot model (gpu/native_exec/screenshot_model.h): PNG encoder round trip, pixel conversion,
// --screenshot-at parsing, file naming. No GPU.
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "png_stored_decode.h"

using namespace playgta6::gpu::native_exec;

static int g_failures = 0;
#define CHECK(expr)                                                                 \
    do {                                                                            \
        if (!(expr)) {                                                              \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #expr); \
            ++g_failures;                                                           \
        }                                                                           \
    } while (0)

int main() {
    // Round trip, including an image larger than one stored block (65535 bytes).
    for (auto dims : {std::pair<uint32_t, uint32_t>{1, 1}, {7, 5}, {200, 150}}) {
        const uint32_t w = dims.first, h = dims.second;
        std::vector<uint8_t> rgb(size_t{w} * h * 3);
        for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<uint8_t>(i * 31 + i / 7);
        const std::vector<uint8_t> png = png_encode_rgb(rgb.data(), w, h);
        uint32_t dw = 0, dh = 0;
        std::vector<uint8_t> back;
        CHECK(png_test_decode_rgb(png, &dw, &dh, &back));
        CHECK(dw == w && dh == h && back == rgb);
    }
    CHECK(png_encode_rgb(nullptr, 1, 1).empty());
    const uint8_t one[3] = {1, 2, 3};
    CHECK(png_encode_rgb(one, 0, 1).empty());
    // Known CRC-32 check value.
    CHECK(png_crc32(reinterpret_cast<const uint8_t*>("123456789"), 9) == 0xCBF43926u);

    // Conversion, with row padding (2 px wide, rows 16 bytes apart).
    std::vector<uint8_t> bgra(32, 0xEE);
    const uint8_t p0[4] = {10, 20, 30, 255}, p1[4] = {40, 50, 60, 255};  // b,g,r,a
    std::memcpy(&bgra[0], p0, 4);
    std::memcpy(&bgra[4], p1, 4);
    std::memcpy(&bgra[16], p1, 4);
    std::memcpy(&bgra[20], p0, 4);
    std::vector<uint8_t> out = shot_to_rgb8(bgra.data(), 16, 2, 2, ShotFormat::Bgra8);
    const std::vector<uint8_t> want = {30, 20, 10, 60, 50, 40, 60, 50, 40, 30, 20, 10};
    CHECK(out == want);
    out = shot_to_rgb8(bgra.data(), 16, 2, 2, ShotFormat::Rgba8);  // same bytes read as rgba
    CHECK(out.size() == 12 && out[0] == 10 && out[2] == 30);
    const uint32_t a2 = 0x3FFu | (512u << 20) | (3u << 30);  // r=1.0 g=0 b~0.5
    out = shot_to_rgb8(reinterpret_cast<const uint8_t*>(&a2), 4, 1, 1, ShotFormat::Rgb10a2);
    CHECK(out.size() == 3 && out[0] == 255 && out[1] == 0 && out[2] == 128);
    const uint16_t half[4] = {0x3C00, 0x3800, 0x0000, 0x3C00};  // 1.0, 0.5, 0, 1
    out = shot_to_rgb8(reinterpret_cast<const uint8_t*>(half), 8, 1, 1, ShotFormat::Rgba16Float);
    CHECK(out.size() == 3 && out[0] == 255 && out[1] == 128 && out[2] == 0);
    CHECK(shot_to_rgb8(bgra.data(), 16, 2, 2, ShotFormat::Unsupported).empty());
    CHECK(shot_to_rgb8(bgra.data(), 4, 2, 2, ShotFormat::Bgra8).empty());  // row too short

    // --screenshot-at
    std::vector<uint64_t> v;
    std::string err;
    CHECK(parse_screenshot_at("30,5,5,,100", &v, &err) && v == std::vector<uint64_t>({5, 30, 100}));
    CHECK(!parse_screenshot_at("5,x", &v, &err) && !err.empty());
    CHECK(!parse_screenshot_at("-1", &v, &err));

    // Naming: timestamp shape and collision suffix.
    const std::string stamp = screenshot_timestamp(std::chrono::system_clock::now());
    CHECK(stamp.size() == 19 && stamp[8] == '-' && stamp[15] == '-');
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "pg6_shot_model_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto path1 = screenshot_unique_path(dir, stamp);
    CHECK(path1.filename() == stamp + ".png");
    std::ofstream(path1).put('x');
    CHECK(screenshot_unique_path(dir, stamp).filename() == stamp + "-2.png");
    std::filesystem::remove_all(dir);

    if (g_failures) return 1;
    std::printf("test_screenshot_model: ok\n");
    return 0;
}
