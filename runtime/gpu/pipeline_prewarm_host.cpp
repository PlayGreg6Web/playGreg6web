#include "gpu/pipeline_prewarm_host.h"
#include "os/thread_name.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

namespace playgta6::gpu {

namespace fs = std::filesystem;

namespace {
std::mutex g_mutex;
std::string g_path;
std::thread g_thread;
std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_yield_ns{0};  // builder time spent waiting on may_build
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
    if (out == nullptr) return false;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const std::streampos end = in.tellg();
    if (end < 0) return false;
    const auto size = static_cast<size_t>(end);
    out->resize(size);
    if (size == 0) return true;
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(size))) {
        out->clear();
        return false;
    }
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
    g_yield_ns = 0;
    g_thread = std::thread([b = std::move(builder)]() {
        playgta6::os::name_this_thread(L"pg6-prewarm");
        while (!g_stop && !b.ready()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        PrewarmTask task{};
        while (!g_stop) {
            if (b.may_build && !b.may_build()) {
                const auto y0 = std::chrono::steady_clock::now();
                while (!g_stop && !b.may_build()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                g_yield_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - y0).count());
                if (g_stop) break;
            }
            if (pipeline_prewarm().take(&task, 1) == 0) break;  // queue drained
            const auto t0 = std::chrono::steady_clock::now();
            PrewarmBuild result;
            const PrewarmDetail detail = pipeline_prewarm().detail_of(task.id);
            if (detail && b.build_pipeline) {
                result = b.build_pipeline(task.id, task.kind, task.key0, task.key1, *detail);
            } else {
                result = b.build(task.kind, task.key0, task.key1) ? PrewarmBuild::Built
                                                                  : PrewarmBuild::Failed;
            }
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t0).count();
            if (result == PrewarmBuild::Lost) {
                pipeline_prewarm().complete_lost(task.id);
            } else {
                pipeline_prewarm().complete(task.id, result == PrewarmBuild::Built, static_cast<uint64_t>(ns));
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
    const PrewarmDiagnostics d = pipeline_prewarm().diagnostics();
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\"queued_total\": %u, \"prewarmed\": %u, \"hit_before_use\": %u, \"wasted\": %u, \"failed\": %u, "
                  "\"skipped\": %u, \"observed\": %u, \"pipelines_prewarmed\": %u, "
                  "\"pipelines_hit_before_use\": %u, \"pipelines_wasted\": %u, \"build_ms\": %llu, "
                  "\"longest_build_ms\": %llu, \"races_lost\": %u, \"retired_queued\": %u, \"pack_skipped\": %u, "
                  "\"priority_queued\": %u, \"yield_ms\": %llu}",
                  p.total, u.prewarmed, u.hit, u.wasted, p.failed, p.skipped, p.observed, u.pipelines_prewarmed,
                  u.pipelines_hit, u.pipelines_wasted,
                  static_cast<unsigned long long>(p.total_ns / 1000000),
                  static_cast<unsigned long long>(p.longest_ns / 1000000),
                  d.races_lost, d.retired_queued, d.pack_skipped, d.priority_queued,
                  static_cast<unsigned long long>(g_yield_ns.load() / 1000000));
    return buf;
}

}  // namespace playgta6::gpu
