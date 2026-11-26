// pg6_native_exec: replays a .pg6trace (gpu/frame_capture.h) through the native executor, frame by
// frame like tools/replay/replay.html, headless or in a window.
//
//   pg6_native_exec --trace <file.pg6trace> [--pack <file.wgsl.pack>] [--window] [--size WxH]
//                   [--frames N] [--digest-at 0,10,...] [--report <file.json>]
//
// --pack        WGSL pack for hash-only CreateShaderModule payloads (the replay page's --pack)
// --size        swapchain size (default 3840x2160, the replay page's canvas)
// --frames N    stop after frame N (default: all)
// --digest-at   headless: print the replay page's `digest frame` / `swapchain frame` lines for these
//               frames, so the output compares line by line with tools/replay_capture.py
// --window      present to a native window (Win32) instead; closing it stops the replay
// --gpu-validation  strict mode: backend validation + an error scope per packet (stats "validation")
// --report      write executor statistics (packets handled / not yet handled by opcode) as JSON
//
// Exit codes: 0 ok, 1 error, 77 no WebGPU adapter.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "gpu/native_exec/native_executor.h"
#include "gpu/native_exec/replay_frames.h"

using namespace playgta6::gpu::native_exec;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: pg6_native_exec --trace <file.pg6trace> [--window] [--size WxH] [--frames N]\n"
                 "                       [--digest-at 0,10,...] [--report <file.json>]\n");
    return 1;
}

bool parse_list(const std::string& text, std::set<uint64_t>* out) {
    size_t pos = 0;
    while (pos < text.size()) {
        size_t comma = text.find(',', pos);
        if (comma == std::string::npos) comma = text.size();
        const std::string part = text.substr(pos, comma - pos);
        if (!part.empty()) {
            char* end = nullptr;
            const unsigned long long v = std::strtoull(part.c_str(), &end, 10);
            if (!end || *end) return false;
            out->insert(v);
        }
        pos = comma + 1;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string trace, report, pack;
    ExecutorOptions opts;
    opts.width = 3840;
    opts.height = 2160;
    uint64_t max_frames = UINT64_MAX;
    std::set<uint64_t> digest_at;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (a == "--window") {
            opts.window = true;
        } else if (a == "--gpu-validation") {
            opts.gpu_validation = true;
        } else if (a == "--trace" && (v = next())) {
            trace = v;
        } else if (a == "--pack" && (v = next())) {
            pack = v;
        } else if (a == "--report" && (v = next())) {
            report = v;
        } else if (a == "--frames" && (v = next())) {
            max_frames = std::strtoull(v, nullptr, 10);
        } else if (a == "--digest-at" && (v = next())) {
            if (!parse_list(v, &digest_at)) return usage();
        } else if (a == "--size" && (v = next())) {
            unsigned w = 0, h = 0;
            if (std::sscanf(v, "%ux%u", &w, &h) != 2 || !w || !h) return usage();
            opts.width = w;
            opts.height = h;
        } else {
            return usage();
        }
    }
    if (trace.empty()) return usage();
    if (opts.window && !digest_at.empty()) {
        std::fprintf(stderr, "pg6_native_exec: --digest-at reads the headless swapchain; drop --window\n");
        return 1;
    }

    std::ifstream in(trace, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "pg6_native_exec: cannot open the trace\n");
        return 1;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto frames = split_replay_frames(bytes.data(), bytes.size());
    std::printf("trace: %zu frames\n", frames.size());

    NativeExecutor::CreateStatus status;
    std::string error;
    auto exec = NativeExecutor::create(opts, &status, &error);
    if (!exec) {
        std::fprintf(stderr, "pg6_native_exec: %s\n", error.c_str());
        return status == NativeExecutor::CreateStatus::NoAdapter ? 77 : 1;
    }
    if (!pack.empty() && !exec->load_wgsl_pack(pack, &error)) {
        std::fprintf(stderr, "pg6_native_exec: %s\n", error.c_str());
        return 1;
    }
    const AdapterDescription& ad = exec->adapter();
    std::printf("adapter: %s (%s)\n", ad.description.empty() ? ad.device.c_str() : ad.description.c_str(),
                ad.backend.c_str());

    int rc = 0;
    for (uint64_t i = 0; i < frames.size() && i <= max_frames; ++i) {
        exec->execute(frames[i].data(), frames[i].size());
        if (!exec->poll()) break;  // window closed
        if (digest_at.count(i)) {
            FrameImage img;
            if (!exec->read_last_frame(&img, &error)) {
                std::fprintf(stderr, "pg6_native_exec: frame %llu: %s\n", static_cast<unsigned long long>(i),
                             error.c_str());
                rc = 1;
                continue;
            }
            if (img.bytes_per_pixel != 4) {
                std::printf("frame %llu: %s swapchain, no replay digest\n", static_cast<unsigned long long>(i),
                            img.format.c_str());
                continue;
            }
            const SampledStats s = replay_sampled_stats(img.pixels.data(), img.width, img.height);
            std::printf("digest frame %llu: %s\n", static_cast<unsigned long long>(i),
                        replay_digest(img.pixels.data(), img.width, img.height).c_str());
            std::printf("swapchain frame %llu (%ux%u %s): non-black %llu/%llu (%.2f%%), distinct %llu\n",
                        static_cast<unsigned long long>(i), img.width, img.height, img.format.c_str(),
                        static_cast<unsigned long long>(s.non_black), static_cast<unsigned long long>(s.total),
                        s.total ? 100.0 * static_cast<double>(s.non_black) / static_cast<double>(s.total) : 0.0,
                        static_cast<unsigned long long>(s.distinct));
        }
    }
    exec->wait_idle();
    for (const std::string& e : exec->stats().device_error_log) std::printf("VALIDATION: %s\n", e.c_str());
    if (!exec->stats().last_packet_error.empty())
        std::printf("last packet error: %s\n", exec->stats().last_packet_error.c_str());
    const std::string json = exec->stats_json();
    std::printf("stats: %s\n", json.c_str());
    if (!report.empty()) {
        std::ofstream out(report, std::ios::binary);
        out << json << "\n";
        if (!out) rc = 1;
    }
    return rc;
}
