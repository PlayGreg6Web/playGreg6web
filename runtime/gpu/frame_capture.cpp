#include "gpu/frame_capture.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/os_shim.h"

namespace playgta6::gpu {

namespace {

struct FrameRecord {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t packets = 0;
};

struct PresentSample {
    double wall_ms = 0;
    FrameCaptureSample counters;
};

struct Omitted {
    uint64_t packet = 0;
    uint16_t opcode = 0;
    uint32_t length = 0;
    uint64_t hash = 0;
};

struct CaptureState {
    std::mutex mutex;
    bool open = false;    // a capture runs (until finish)
    bool active = false;  // still writing packets
    FrameCaptureConfig config;
    std::FILE* file = nullptr;
    uint64_t bytes = 0;
    uint64_t packets = 0;
    FrameRecord current;
    std::vector<FrameRecord> frames;
    std::vector<PresentSample> presents;
    std::vector<Omitted> omitted;
    uint64_t omitted_bytes = 0;
    std::chrono::steady_clock::time_point start;
    std::atomic<FrameCaptureCounters> counters{nullptr};
};

CaptureState& state() {
    static CaptureState s;
    return s;
}

uint64_t fnv1a64(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// Offset of the u32 length of the trailing data in the payload, or -1.
int data_length_offset(uint16_t opcode) {
    switch (static_cast<GpuOpcode>(opcode)) {
        case GpuOpcode::WriteBuffer:
            return static_cast<int>(offsetof(GpuWriteBufferPayload, data_len));
        case GpuOpcode::CreateShaderModule:
            return static_cast<int>(offsetof(GpuCreateShaderModulePayload, code_len));
        case GpuOpcode::WriteTexture:
            return static_cast<int>(offsetof(GpuWriteTexturePayload, data_len));
        default:
            return -1;
    }
}

size_t fixed_payload_size(uint16_t opcode) {
    switch (static_cast<GpuOpcode>(opcode)) {
        case GpuOpcode::WriteBuffer:
            return sizeof(GpuWriteBufferPayload);
        case GpuOpcode::CreateShaderModule:
            return sizeof(GpuCreateShaderModulePayload);
        case GpuOpcode::WriteTexture:
            return sizeof(GpuWriteTexturePayload);
        default:
            return 0;
    }
}

void write_bytes(CaptureState& s, const void* p, size_t n) {
    if (n != 0 && std::fwrite(p, 1, n, s.file) != n) {
        s.active = false;  // disk error: stop writing, the sidecar still reports what was written
    }
    s.bytes += n;
    s.current.bytes += n;
}

void sample_present(CaptureState& s) {
    PresentSample p;
    p.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s.start).count();
    if (FrameCaptureCounters c = s.counters.load(std::memory_order_acquire)) {
        c(&p.counters);
    }
    s.presents.push_back(p);
}

void write_packet(CaptureState& s, const GpuPacketHeader& h, const uint8_t* packet) {
    const uint8_t* payload = packet + sizeof(h);
    const int len_off = data_length_offset(h.opcode);
    const size_t fixed = fixed_payload_size(h.opcode);
    if (!s.config.keep_payloads && len_off >= 0 && h.payload_bytes > fixed) {
        uint32_t data_len = 0;
        std::memcpy(&data_len, payload + len_off, sizeof(data_len));
        const size_t tail = h.payload_bytes - fixed;
        s.omitted.push_back(Omitted{s.packets, h.opcode, data_len, fnv1a64(payload + fixed, tail)});
        s.omitted_bytes += tail;
        GpuPacketHeader out = h;
        out.payload_bytes = static_cast<uint32_t>(fixed);
        std::vector<uint8_t> head(payload, payload + fixed);
        const uint32_t zero = 0;
        std::memcpy(head.data() + len_off, &zero, sizeof(zero));
        write_bytes(s, &out, sizeof(out));
        write_bytes(s, head.data(), head.size());
    } else {
        write_bytes(s, packet, sizeof(h) + h.payload_bytes);
    }
    ++s.packets;
    ++s.current.packets;
}

void observe_locked(CaptureState& s, const uint8_t* data, size_t len) {
    size_t at = 0;
    while (at + sizeof(GpuPacketHeader) <= len) {
        GpuPacketHeader h{};
        std::memcpy(&h, data + at, sizeof(h));
        if (h.payload_bytes > len - at - sizeof(h)) {
            break;  // a truncated packet: never produced by the encoder; ignore the rest
        }
        if (s.active) {
            write_packet(s, h, data + at);
        }
        at += sizeof(h) + h.payload_bytes;
        if (h.opcode != static_cast<uint16_t>(GpuOpcode::Present)) {
            continue;
        }
        sample_present(s);
        if (s.active) {
            s.frames.push_back(s.current);
            s.current = FrameRecord{};
            s.current.offset = s.bytes;
            if (s.frames.size() >= s.config.frames) {
                s.active = false;
                std::fflush(s.file);
            }
        }
    }
}

void observer(const uint8_t* data, size_t len) {
    frame_capture_observe(data, len);
}

std::string json_escape(const std::string& in) {
    std::string out;
    for (char c : in) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        if (static_cast<unsigned char>(c) < 0x20) {
            continue;
        }
        out += c;
    }
    return out;
}

}  // namespace

bool frame_capture_start(const FrameCaptureConfig& config) {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.open || config.path.empty() || config.frames == 0) {
        return false;
    }
    s.file = std::fopen(config.path.c_str(), "wb");
    if (s.file == nullptr) {
        return false;
    }
    s.config = config;
    s.open = true;
    s.active = true;
    s.bytes = s.packets = s.omitted_bytes = 0;
    s.current = FrameRecord{};
    s.frames.clear();
    s.presents.clear();
    s.omitted.clear();
    s.start = std::chrono::steady_clock::now();
    if (config.attach) {
        os::set_wgpu_submit_observer(&observer);
    }
    return true;
}

void frame_capture_observe(const uint8_t* data, size_t len) {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.open) {
        observe_locked(s, data, len);
    }
}

void frame_capture_finish() {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.open) {
        return;
    }
    if (s.config.attach) {
        os::set_wgpu_submit_observer(nullptr);
    }
    std::fclose(s.file);
    s.file = nullptr;
    s.open = false;
    s.active = false;
    std::FILE* side = std::fopen((s.config.path + ".json").c_str(), "wb");
    if (side == nullptr) {
        return;
    }
    std::string name = s.config.path;
    const size_t slash = name.find_last_of("/\\");
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    std::fprintf(side,
                 "{\n  \"schema\": \"playgta6-frame-capture\",\n  \"schema_version\": 1,\n  \"trace\": \"%s\",\n"
                 "  \"payloads\": \"%s\",\n  \"frames_requested\": %u,\n  \"frames_complete\": %zu,\n"
                 "  \"presents_seen\": %zu,\n  \"packets\": %" PRIu64 ",\n  \"bytes\": %" PRIu64 ",\n"
                 "  \"omitted_bytes\": %" PRIu64 ",\n  \"frames\": [",
                 json_escape(name).c_str(), s.config.keep_payloads ? "kept" : "omitted", s.config.frames,
                 s.frames.size(), s.presents.size(), s.packets, s.bytes, s.omitted_bytes);
    for (size_t i = 0; i < s.frames.size(); ++i) {
        const FrameRecord& f = s.frames[i];
        std::fprintf(side,
                     "%s\n    {\"index\": %zu, \"offset\": %" PRIu64 ", \"bytes\": %" PRIu64 ", \"packets\": %" PRIu64 "}",
                     i == 0 ? "" : ",", i, f.offset, f.bytes, f.packets);
    }
    std::fprintf(side, "%s],\n  \"presents\": [", s.frames.empty() ? "" : "\n  ");
    for (size_t i = 0; i < s.presents.size(); ++i) {
        const PresentSample& p = s.presents[i];
        std::fprintf(side,
                     "%s\n    {\"index\": %zu, \"wall_ms\": %.3f, \"dispatches\": %" PRIu64 ", \"import_calls\": %" PRIu64
                     ", \"reserved_bytes\": %" PRIu64 ", \"committed_bytes\": %" PRIu64 "}",
                     i == 0 ? "" : ",", i, p.wall_ms, p.counters.dispatches, p.counters.import_calls,
                     p.counters.reserved_bytes, p.counters.committed_bytes);
    }
    std::fprintf(side, "%s],\n  \"omitted\": [", s.presents.empty() ? "" : "\n  ");
    for (size_t i = 0; i < s.omitted.size(); ++i) {
        const Omitted& o = s.omitted[i];
        std::fprintf(side,
                     "%s\n    {\"packet\": %" PRIu64 ", \"opcode\": %u, \"length\": %u, \"fnv1a64\": \"0x%016" PRIx64 "\"}",
                     i == 0 ? "" : ",", o.packet, static_cast<unsigned>(o.opcode), o.length, o.hash);
    }
    std::fprintf(side, "%s]\n}\n", s.omitted.empty() ? "" : "\n  ");
    std::fclose(side);
}

void frame_capture_set_counters(FrameCaptureCounters counters) {
    state().counters.store(counters, std::memory_order_release);
}

uint32_t frame_capture_frames() {
    CaptureState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return static_cast<uint32_t>(s.frames.size());
}

}  // namespace playgta6::gpu
