#include "gpu/gpu_va_watch.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>

#include "os/os_shim.h"
#include "os/win32_sync.h"

namespace playgta6::gpu::va_watch {

namespace {

struct Range {
    uint64_t begin = 0;
    uint64_t end = 0;  // exclusive
    std::string label;
};

struct State {
    std::mutex mutex;
    std::vector<Range> ranges;
    std::vector<std::pair<uint32_t, uint32_t>> texture_sizes;
    std::map<std::tuple<std::string, uint64_t, std::string>, Event> events;
    std::vector<std::tuple<std::string, uint64_t, std::string>> order;
    std::map<uint64_t, std::pair<uint64_t, std::string>> handles;  // handle -> (va, range label)
};

State& state() {
    static State s;
    return s;
}

std::atomic<bool> g_active{false};
std::atomic<uint64_t> g_frame{0};

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%" PRIx64, v);
    return b;
}

bool parse_u64(std::string_view s, uint64_t* out) {
    if (s.empty()) {
        return false;
    }
    int base = 10;
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        base = 16;
        s.remove_prefix(2);
    }
    uint64_t v = 0;
    for (const char c : s) {
        uint64_t d;
        if (c >= '0' && c <= '9') {
            d = static_cast<uint64_t>(c - '0');
        } else if (base == 16 && c >= 'a' && c <= 'f') {
            d = static_cast<uint64_t>(c - 'a' + 10);
        } else if (base == 16 && c >= 'A' && c <= 'F') {
            d = static_cast<uint64_t>(c - 'A' + 10);
        } else {
            return false;
        }
        if (d >= static_cast<uint64_t>(base) || v > (UINT64_MAX - d) / static_cast<uint64_t>(base)) {
            return false;
        }
        v = v * static_cast<uint64_t>(base) + d;
    }
    *out = v;
    return true;
}

// Logs and counts one event; caller holds the mutex.
void record_locked(State& s, const char* kind, uint64_t va, uint64_t size, const std::string& range,
                   const std::string& detail) {
    auto key = std::make_tuple(std::string(kind), va, detail);
    auto it = s.events.find(key);
    if (it != s.events.end()) {
        ++it->second.count;
        return;
    }
    Event e;
    e.kind = kind;
    e.va = va;
    e.size = size;
    e.range = range;
    e.detail = detail;
    e.first_frame = g_frame.load(std::memory_order_relaxed);
    e.first_thread = os::win32::GetCurrentThreadId();
    e.count = 1;
    os::log("gpu-va-watch: frame " + std::to_string(e.first_frame) + " thread " + std::to_string(e.first_thread) + " " +
            kind + " " + hex(va) + " +" + std::to_string(size) + " [" + range + "] " + detail);
    s.order.push_back(key);
    s.events.emplace(std::move(key), std::move(e));
}

}  // namespace

bool configure(std::string_view spec, std::string* error) {
    std::vector<Range> ranges;
    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    size_t pos = 0;
    while (pos <= spec.size() && !spec.empty()) {
        const size_t comma = spec.find(',', pos);
        const std::string_view item = spec.substr(pos, comma == std::string_view::npos ? spec.size() - pos : comma - pos);
        pos = comma == std::string_view::npos ? spec.size() + 1 : comma + 1;
        if (item.empty()) {
            continue;
        }
        if (item.substr(0, 9) == "textures:") {
            std::string_view rest = item.substr(9);
            while (!rest.empty()) {
                const size_t slash = rest.find('/');
                const std::string_view wh = rest.substr(0, slash);
                rest = slash == std::string_view::npos ? std::string_view{} : rest.substr(slash + 1);
                const size_t x = wh.find('x');
                uint64_t w = 0, h = 0;
                if (x == std::string_view::npos || !parse_u64(wh.substr(0, x), &w) || !parse_u64(wh.substr(x + 1), &h) ||
                    w == 0 || h == 0 || w > 0xFFFFFFFFu || h > 0xFFFFFFFFu) {
                    *error = "bad texture size '" + std::string(wh) + "' (expected WxH)";
                    return false;
                }
                sizes.emplace_back(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
            }
            continue;
        }
        const size_t dash = item.find('-');
        const size_t plus = item.find('+');
        uint64_t a = 0, b = 0;
        if (dash != std::string_view::npos && parse_u64(item.substr(0, dash), &a) && parse_u64(item.substr(dash + 1), &b) &&
            b > a) {
            ranges.push_back({a, b, std::string(item)});
        } else if (plus != std::string_view::npos && parse_u64(item.substr(0, plus), &a) &&
                   parse_u64(item.substr(plus + 1), &b) && b != 0 && a + b > a) {
            ranges.push_back({a, a + b, std::string(item)});
        } else {
            *error = "bad range '" + std::string(item) + "' (expected 0xSTART-0xEND, 0xSTART+SIZE or textures:WxH)";
            return false;
        }
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.ranges = std::move(ranges);
    s.texture_sizes = std::move(sizes);
    g_active.store(!s.ranges.empty() || !s.texture_sizes.empty());
    return true;
}

void reset() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.ranges.clear();
    s.texture_sizes.clear();
    s.events.clear();
    s.order.clear();
    s.handles.clear();
    g_active.store(false);
    g_frame.store(0);
}

bool active() {
    return g_active.load(std::memory_order_relaxed);
}

void note_texture(uint64_t va, uint64_t size, uint32_t width, uint32_t height, uint32_t dxgi_format,
                  uint32_t resource_id) {
    if (!active() || va == 0) {
        return;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto& [w, h] : s.texture_sizes) {
        if (w == width && h == height) {
            const std::string label = "texture " + std::to_string(width) + "x" + std::to_string(height) + " id " +
                                      std::to_string(resource_id);
            s.ranges.push_back({va, va + std::max<uint64_t>(size, 1), label});
            record_locked(s, "texture", va, size, label, "created, DXGI format " + std::to_string(dxgi_format));
            return;
        }
    }
    const uint64_t end = va + std::max<uint64_t>(size, 1);
    for (const Range& r : s.ranges) {
        if (va < r.end && r.begin < end) {
            record_locked(s, "texture", va, size, r.label,
                          "created over the range, " + std::to_string(width) + "x" + std::to_string(height) +
                              " DXGI format " + std::to_string(dxgi_format) + " id " + std::to_string(resource_id));
            return;
        }
    }
}

bool touch(const char* kind, uint64_t va, uint64_t size, const std::string& detail) {
    if (!active() || va == 0) {
        return false;
    }
    const uint64_t end = va + std::max<uint64_t>(size, 1);
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const Range& r : s.ranges) {
        if (va < r.end && r.begin < end) {
            record_locked(s, kind, va, size, r.label, detail);
            return true;
        }
    }
    return false;
}

void watch_handle(uint64_t handle, uint64_t va) {
    if (!active() || handle == 0) {
        return;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::string label;
    for (const Range& r : s.ranges) {
        if (va >= r.begin && va < r.end) {
            label = r.label;
        }
    }
    s.handles[handle] = {va, label};
}

void descriptor_copied(uint64_t src_handle, uint64_t dst_handle) {
    if (!active()) {
        return;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const auto it = s.handles.find(src_handle);
    if (it == s.handles.end()) {
        s.handles.erase(dst_handle);  // overwritten by an unwatched descriptor
        return;
    }
    const auto [va, label] = it->second;
    record_locked(s, "descriptor copy", va, 0, label, "handle " + hex(src_handle) + " -> " + hex(dst_handle));
    s.handles[dst_handle] = {va, label};
}

void set_frame(uint64_t frame) {
    g_frame.store(frame, std::memory_order_relaxed);
}

std::vector<Event> events() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::vector<Event> out;
    out.reserve(s.order.size());
    for (const auto& key : s.order) {
        out.push_back(s.events.at(key));
    }
    return out;
}

std::string summary() {
    const std::vector<Event> ev = events();
    if (ev.empty()) {
        return "gpu-va-watch: nothing referenced the watched ranges\n";
    }
    std::string out = "gpu-va-watch: " + std::to_string(ev.size()) + " distinct reference(s)\n";
    for (const Event& e : ev) {
        out += "  " + e.kind + " " + hex(e.va) + " +" + std::to_string(e.size) + " [" + e.range + "] " + e.detail +
               ": " + std::to_string(e.count) + "x, first frame " + std::to_string(e.first_frame) + " thread " +
               std::to_string(e.first_thread) + "\n";
    }
    return out;
}

}  // namespace playgta6::gpu::va_watch
