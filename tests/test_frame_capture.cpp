// gpu/frame_capture: a synthetic command stream through os::wgpu_submit is written as a
// .pg6trace with data payloads omitted (packets stay well-formed) and a sidecar with frame
// boundaries; capture stops after the requested frames.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "gpu/cmd_encoder.h"
#include "gpu/frame_capture.h"
#include "os/os_shim.h"

namespace {

int g_failures = 0;

void check_impl(bool ok, const char* expr, int line) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL [test_frame_capture.cpp:%d]: %s\n", line, expr);
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using playgta6::gpu::CommandEncoder;
namespace fc = playgta6::gpu;

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct Packet {
    uint16_t opcode;
    uint32_t payload_bytes;
    size_t offset;
};

std::vector<Packet> packets_of(const std::vector<uint8_t>& bytes, bool* well_formed) {
    std::vector<Packet> out;
    size_t at = 0;
    *well_formed = true;
    while (at < bytes.size()) {
        if (bytes.size() - at < sizeof(GpuPacketHeader)) {
            *well_formed = false;
            break;
        }
        GpuPacketHeader h{};
        std::memcpy(&h, bytes.data() + at, sizeof(h));
        if (h.payload_bytes > bytes.size() - at - sizeof(h)) {
            *well_formed = false;
            break;
        }
        out.push_back(Packet{h.opcode, h.payload_bytes, at});
        at += sizeof(h) + h.payload_bytes;
    }
    return out;
}

void submit(const CommandEncoder& e) {
    playgta6::os::wgpu_submit(e.data(), e.size());
}

void test_capture_omits_payloads_and_stops() {
    const std::string path = (std::filesystem::temp_directory_path() / "pg6_test_capture.pg6trace").string();
    fc::FrameCaptureConfig config;
    config.path = path;
    config.frames = 2;
    CHECK(fc::frame_capture_start(config));
    CHECK(!fc::frame_capture_start(config));  // one capture at a time
    static uint64_t calls = 0;
    fc::frame_capture_set_counters([](fc::FrameCaptureSample* s) {
        calls += 10;
        s->dispatches = calls;
        s->import_calls = calls / 10;
        s->reserved_bytes = 0x1000 * calls;
        s->committed_bytes = 0x100 * calls;
        s->counters = {{"handles_dispatcher", calls / 10}, {"heap_live_blocks", 7}};
    });

    const std::vector<uint8_t> blob(1000, 0xAB);
    CommandEncoder e;
    e.begin_frame();
    e.create_buffer(1, 0x28, 4096);
    e.write_buffer(1, 16, blob.data(), static_cast<uint32_t>(blob.size()));
    e.draw(3);
    e.present();
    submit(e);  // frame 0
    e.clear();
    e.begin_frame();
    e.draw(6);
    e.draw(9);
    submit(e);
    e.clear();
    e.present();
    submit(e);  // frame 1 (split over two submissions)
    e.clear();
    e.begin_frame();
    e.draw(12);
    e.present();
    submit(e);  // frame 2: not captured
    CHECK(fc::frame_capture_frames() == 2);
    fc::frame_capture_finish();
    fc::frame_capture_finish();  // idempotent

    bool ok = false;
    const std::vector<uint8_t> trace = read_file(path);
    const std::vector<Packet> ps = packets_of(trace, &ok);
    CHECK(ok);
    CHECK(ps.size() == 9);  // begin, create, write, draw, present | begin, draw, draw, present
    CHECK(ps[2].opcode == static_cast<uint16_t>(GpuOpcode::WriteBuffer));
    CHECK(ps[2].payload_bytes == sizeof(GpuWriteBufferPayload));  // data omitted
    GpuWriteBufferPayload wb{};
    std::memcpy(&wb, trace.data() + ps[2].offset + sizeof(GpuPacketHeader), sizeof(wb));
    CHECK(wb.id == 1 && wb.dst_offset == 16 && wb.data_len == 0);
    CHECK(ps[8].opcode == static_cast<uint16_t>(GpuOpcode::Present));
    CHECK(trace.size() < 300);

    const std::vector<uint8_t> side_bytes = read_file(path + ".json");
    const std::string side(side_bytes.begin(), side_bytes.end());
    CHECK(side.find("\"schema\": \"playgta6-frame-capture\"") != std::string::npos);
    CHECK(side.find("\"payloads\": \"omitted\"") != std::string::npos);
    CHECK(side.find("\"frames_complete\": 2") != std::string::npos);
    CHECK(side.find("\"presents_seen\": 3") != std::string::npos);
    CHECK(side.find("\"omitted_bytes\": 1000") != std::string::npos);
    CHECK(side.find("\"length\": 1000") != std::string::npos);
    const std::string frame1 = "\"index\": 1, \"offset\": " + std::to_string(ps[5].offset);
    CHECK(side.find(frame1) != std::string::npos);
    // One sample per Present, the third one after the captured frames.
    CHECK(side.find("\"dispatches\": 30, \"import_calls\": 3, \"reserved_bytes\": 122880, \"committed_bytes\": 7680, "
                    "\"counters\": {\"handles_dispatcher\": 3, \"heap_live_blocks\": 7}") != std::string::npos);
    fc::frame_capture_set_counters(nullptr);
    std::filesystem::remove(path);
    std::filesystem::remove(path + ".json");
}

void test_capture_keeps_payloads_on_request() {
    const std::string path = (std::filesystem::temp_directory_path() / "pg6_test_capture_full.pg6trace").string();
    fc::FrameCaptureConfig config;
    config.path = path;
    config.frames = 1;
    config.keep_payloads = true;
    CHECK(fc::frame_capture_start(config));
    const uint8_t data[64] = {1, 2, 3};
    CommandEncoder e;
    e.begin_frame();
    e.create_buffer(2, 0x28, 64);
    e.write_buffer(2, 0, data, sizeof(data));
    e.present();
    submit(e);
    fc::frame_capture_finish();
    bool ok = false;
    const std::vector<uint8_t> trace = read_file(path);
    const std::vector<Packet> ps = packets_of(trace, &ok);
    CHECK(ok && ps.size() == 4);
    CHECK(ps[2].payload_bytes == sizeof(GpuWriteBufferPayload) + sizeof(data));
    std::filesystem::remove(path);
    std::filesystem::remove(path + ".json");
}

void test_capture_thins_large_uploads() {
    const std::string path = (std::filesystem::temp_directory_path() / "pg6_test_capture_thin.pg6trace").string();
    fc::FrameCaptureConfig config;
    config.path = path;
    config.frames = 1;
    config.keep_payloads = true;
    config.thin_uploads = 3;
    config.thin_min_bytes = 100;
    CHECK(fc::frame_capture_start(config));
    const std::vector<uint8_t> big(1000, 7), small(50, 9);
    CommandEncoder e;
    e.begin_frame();
    for (int i = 0; i < 7; ++i) {
        e.write_buffer(5, 0, big.data(), static_cast<uint32_t>(big.size()));
    }
    e.write_buffer(6, 0, big.data(), static_cast<uint32_t>(big.size()));  // other buffer: first write kept
    for (int i = 0; i < 4; ++i) {
        e.write_buffer(7, 0, small.data(), static_cast<uint32_t>(small.size()));  // below threshold: all kept
    }
    e.present();
    submit(e);
    fc::frame_capture_finish();
    bool ok = false;
    const std::vector<uint8_t> trace = read_file(path);
    const std::vector<Packet> ps = packets_of(trace, &ok);
    CHECK(ok && ps.size() == 14);
    const uint32_t full = sizeof(GpuWriteBufferPayload) + 1000;
    for (int i = 0; i < 7; ++i) {
        const Packet& p = ps[1 + i];
        const bool kept = i % 3 == 0;  // writes 0, 3, 6
        CHECK(p.payload_bytes == (kept ? full : sizeof(GpuWriteBufferPayload)));
        GpuWriteBufferPayload wb{};
        std::memcpy(&wb, trace.data() + p.offset + sizeof(GpuPacketHeader), sizeof(wb));
        CHECK(wb.id == 5 && wb.data_len == (kept ? 1000u : 0u));
        CHECK(wb.reserved == (kept ? 0u : fc::kThinUploadMarker));
    }
    CHECK(ps[8].payload_bytes == full);
    for (int i = 9; i < 13; ++i) {
        CHECK(ps[i].payload_bytes == sizeof(GpuWriteBufferPayload) + 50);
    }
    const std::vector<uint8_t> side_bytes = read_file(path + ".json");
    const std::string side(side_bytes.begin(), side_bytes.end());
    CHECK(side.find("\"thinned_packets\": 4") != std::string::npos);
    CHECK(side.find("\"thinned_bytes\": 4000") != std::string::npos);
    std::filesystem::remove(path);
    std::filesystem::remove(path + ".json");
}

}  // namespace

int main() {
    test_capture_omits_payloads_and_stops();
    test_capture_keeps_payloads_on_request();
    test_capture_thins_large_uploads();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_frame_capture: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_frame_capture: all checks passed\n");
    return 0;
}
