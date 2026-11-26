#include "gpu/pipeline_prewarm_host.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>

namespace playgta6::gpu {

namespace fs = std::filesystem;

namespace {
std::mutex g_mutex;
std::string g_path;
std::thread g_thread;
std::atomic<bool> g_stop{false};
}  // namespace

std::string default_warm_list_path(const std::string& title) {
    std::string base;
#if defined(_WIN32)
    if (const char* v = std::getenv("LOCALAPPDATA")) base = v;
#else
    if (const char* v = std::getenv("XDG_CACHE_HOME")) {
        base = v;
    } else if (const char* h = std::getenv("HOME")) {
        base = std::string(h) + "/.cache";
    }
#endif
    if (base.empty()) return "";
    std::string name;
    for (char c : title) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
                        c == '_' || c == '.';
        name += ok ? c : '_';
    }
    if (name.empty() || name == "." || name == "..") name = "title";
    return (fs::path(base) / "playgta6" / "pipeline-warm" / (name + ".pg6w")).string();
}

bool read_warm_list_file(const std::string& path, std::vector<uint8_t>* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

bool write_warm_list_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    if (path.empty() || bytes.size() <= kPrewarmListHeaderBytes) return false;
    std::error_code ec;
    const fs::path target(path);
    if (target.has_parent_path()) fs::create_directories(target.parent_path(), ec);
    const fs::path tmp = fs::path(path + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        out.flush();
        if (!out) {
            out.close();
            fs::remove(tmp, ec);
            return false;
        }
    }
    fs::rename(tmp, target, ec);  // replaces an existing file
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

uint32_t prewarm_host_configure(const std::string& path, std::string* message) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_path = (path == "off") ? std::string() : path;
    }
    std::string msg;
    uint32_t queued = 0;
    if (path.empty() || path == "off") {
        msg = "disabled";
    } else {
        std::vector<uint8_t> bytes;
        if (!read_warm_list_file(path, &bytes)) {
            msg = "no warm list yet";
        } else if (!pipeline_prewarm().load_warm_list(bytes.data(), bytes.size(), &msg, &queued)) {
            msg = "ignored: " + msg;  // corrupt or foreign file: start cold, the next save replaces it
            queued = 0;
        } else {
            msg = "loaded";
        }
    }
    if (message) *message = msg;
    return queued;
}

void prewarm_host_start(PrewarmBuilder builder) {
    prewarm_host_stop();
    g_stop = false;
    g_thread = std::thread([b = std::move(builder)]() {
        while (!g_stop && !b.ready()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        PrewarmTask tasks[8];
        while (!g_stop) {
            const size_t n = pipeline_prewarm().take(tasks, 8);
            if (n == 0) break;  // queue drained
            for (size_t i = 0; i < n; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = b.build(tasks[i].kind, tasks[i].key0, tasks[i].key1);
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - t0).count();
                pipeline_prewarm().complete(tasks[i].id, ok, static_cast<uint64_t>(ns));
            }
        }
    });
}

void prewarm_host_wait() {
    if (g_thread.joinable()) g_thread.join();
}

void prewarm_host_stop() {
    g_stop = true;
    if (g_thread.joinable()) g_thread.join();
}

bool prewarm_host_save() {
    std::string path;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        path = g_path;
    }
    if (path.empty()) return false;
    return write_warm_list_file(path, pipeline_prewarm().save_warm_list());
}

std::string prewarm_host_report_json() {
    const PrewarmProgress p = pipeline_prewarm().progress();
    const PrewarmUse u = pipeline_prewarm().use_stats();
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "{\"queued_total\": %u, \"prewarmed\": %u, \"hit_before_use\": %u, \"wasted\": %u, \"failed\": %u, "
                  "\"skipped\": %u, \"observed\": %u, \"build_ms\": %llu}",
                  p.total, u.prewarmed, u.hit, u.wasted, p.failed, p.skipped, p.observed,
                  static_cast<unsigned long long>(p.total_ns / 1000000));
    return buf;
}

}  // namespace playgta6::gpu
