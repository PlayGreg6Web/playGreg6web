// XMem CPU/GPU windows (O27). Design and evidence: xmem_windows.h.

#include "os/xmem_windows.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <mutex>
#include <unordered_map>

namespace playgta6::os::xmem_windows {

namespace {

constexpr uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

PlacementPolicy default_policy() {
#if defined(__EMSCRIPTEN__)
    return PlacementPolicy::PhantomGpuWindows;
#else
    return PlacementPolicy::Arena;
#endif
}

struct State {
    std::mutex mutex;
    PlacementPolicy policy = default_policy();
    uint64_t next_phantom = kPhantomBase;
    std::map<uint64_t, uint64_t> phantom;  // base -> size
    std::map<uint64_t, Window> windows;    // base -> window
    std::vector<AliasPair> pairs;
    std::map<uint64_t, AliasListener> listeners;
    uint64_t next_listener = 1;
    std::unordered_map<uint64_t, uint64_t> gpu_pages;  // GPU page VA -> pool page index
    uint64_t gpu_mapping_updates = 0;
    int64_t cpu_window_mapped = 0;
};

State& state() {
    static State s;
    return s;
}

bool window_containing_locked(const State& s, uint64_t va, uint64_t len, Window* out) {
    auto it = s.windows.upper_bound(va);
    if (it == s.windows.begin()) {
        return false;
    }
    --it;
    const Window& w = it->second;
    if (va - w.base >= w.size || len > w.size - (va - w.base)) {
        return false;
    }
    if (out != nullptr) {
        *out = w;
    }
    return true;
}

}  // namespace

bool classify(uint32_t allocation_type, uint32_t protect, uint64_t size, WindowKind* kind) {
    constexpr uint32_t kMemCommitBit = 0x1000u, kMemReserveBit = 0x2000u;
    if ((allocation_type & kMemReserveBit) == 0 || (allocation_type & kMemCommitBit) != 0 ||
        (allocation_type & kMemPhysical) == 0 || size < kMinWindowBytes) {
        return false;
    }
    if (kind != nullptr) {
        *kind = (protect & 0xFFu) == 0x01u ? WindowKind::Gpu : WindowKind::Cpu;  // PAGE_NOACCESS
    }
    return true;
}

PlacementPolicy policy() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.policy;
}

void set_policy(PlacementPolicy policy) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.policy = policy;
}

bool place_phantom(WindowKind kind) {
    const PlacementPolicy p = policy();
    return kind == WindowKind::Gpu &&
           (p == PlacementPolicy::PhantomGpuWindows || p == PlacementPolicy::PhantomGpuTranslatedCpu);
}

bool place_translated(WindowKind kind) {
    return kind == WindowKind::Cpu && policy() == PlacementPolicy::PhantomGpuTranslatedCpu;
}

uint64_t reserve_phantom(uint64_t size) {
    if (size == 0) {
        return 0;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const uint64_t bytes = align_up(size, kWindowPageSize);
    if (bytes > kPhantomLimit - s.next_phantom) {
        return 0;
    }
    const uint64_t base = s.next_phantom;
    s.next_phantom += bytes;
    s.phantom.emplace(base, bytes);
    return base;
}

bool is_phantom(uint64_t va, uint64_t len) {
    if (va < kPhantomBase || va >= kPhantomLimit) {
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.phantom.upper_bound(va);
    if (it == s.phantom.begin()) {
        return false;
    }
    --it;
    return va - it->first < it->second && len <= it->second - (va - it->first);
}

void note_window(const Window& window) {
    std::vector<AliasListener> to_call;
    AliasPair pair{};
    bool paired = false;
    {
        State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.windows[window.base] = window;
        // Pair with an unpaired window of the other kind and the same size (the most recent one).
        for (auto it = s.windows.rbegin(); it != s.windows.rend(); ++it) {
            const Window& other = it->second;
            if (other.base == window.base || other.kind == window.kind || other.size != window.size) {
                continue;
            }
            const bool taken = std::any_of(s.pairs.begin(), s.pairs.end(), [&](const AliasPair& p) {
                return p.cpu_base == other.base || p.gpu_base == other.base;
            });
            if (taken) {
                continue;
            }
            pair.cpu_base = window.kind == WindowKind::Cpu ? window.base : other.base;
            pair.gpu_base = window.kind == WindowKind::Gpu ? window.base : other.base;
            pair.size = window.size;
            s.pairs.push_back(pair);
            paired = true;
            break;
        }
        if (paired) {
            for (const auto& [id, fn] : s.listeners) {
                to_call.push_back(fn);
            }
        }
    }
    for (const AliasListener& fn : to_call) {
        fn(pair);
    }
}

bool forget_window(uint64_t base) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.windows.find(base);
    if (it == s.windows.end()) {
        return false;
    }
    if (it->second.kind == WindowKind::Gpu) {
        const uint64_t lo = it->second.base, hi = it->second.base + it->second.size;
        for (auto p = s.gpu_pages.begin(); p != s.gpu_pages.end();) {
            p = (p->first >= lo && p->first < hi) ? s.gpu_pages.erase(p) : std::next(p);
        }
    }
    s.pairs.erase(std::remove_if(s.pairs.begin(), s.pairs.end(),
                                 [&](const AliasPair& p) { return p.cpu_base == base || p.gpu_base == base; }),
                  s.pairs.end());
    s.phantom.erase(base);
    s.windows.erase(it);
    return true;
}

std::vector<Window> windows() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::vector<Window> out;
    for (const auto& [base, w] : s.windows) {
        out.push_back(w);
    }
    return out;
}

std::vector<AliasPair> alias_pairs() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.pairs;
}

bool find_window(uint64_t va, uint64_t len, Window* out) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return window_containing_locked(s, va, len, out);
}

uint64_t add_alias_listener(AliasListener listener) {
    std::vector<AliasPair> existing;
    uint64_t id = 0;
    {
        State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        id = s.next_listener++;
        s.listeners.emplace(id, listener);
        existing = s.pairs;
    }
    for (const AliasPair& p : existing) {
        listener(p);
    }
    return id;
}

void remove_alias_listener(uint64_t id) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.listeners.erase(id);
}

PageMappingResult apply_gpu_page_mappings(std::span<const PageMappingRange> ranges, uint64_t, uint64_t pool_pages) {
    PageMappingResult r;
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    ++s.gpu_mapping_updates;
    for (const PageMappingRange& range : ranges) {
        uint64_t page_va = range.gpu_va & ~(kWindowPageSize - 1);
        for (const PageMappingEntry& e : range.entries) {
            if (e.kind != 1 && e.kind != 2) {
                ++r.skipped_entries;
                continue;
            }
            for (uint32_t i = 0; i < e.count; ++i, page_va += kWindowPageSize) {
                Window w;
                if (!window_containing_locked(s, page_va, kWindowPageSize, &w) || w.kind != WindowKind::Gpu) {
                    ++r.outside_window;
                    continue;
                }
                if (e.kind == 1) {
                    const uint64_t pool_page = uint64_t{e.pool_page} + i;
                    if (pool_pages != 0 && pool_page >= pool_pages) {
                        ++r.outside_window;
                        continue;
                    }
                    s.gpu_pages[page_va] = pool_page;
                    ++r.mapped_pages;
                } else {
                    s.gpu_pages.erase(page_va);
                    ++r.unmapped_pages;
                }
            }
        }
    }
    return r;
}

int64_t gpu_page_pool_index(uint64_t gpu_va) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.gpu_pages.find(gpu_va & ~(kWindowPageSize - 1));
    return it == s.gpu_pages.end() ? -1 : static_cast<int64_t>(it->second);
}

void note_cpu_window_mapping(int64_t delta_bytes) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.cpu_window_mapped += delta_bytes;
}

Accounting accounting() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Accounting a;
    for (const auto& [base, w] : s.windows) {
        if (w.kind == WindowKind::Cpu) {
            ++a.cpu_windows;
            a.cpu_window_reserved += w.size;
        } else {
            ++a.gpu_windows;
            a.gpu_window_reserved += w.size;
        }
    }
    for (const auto& [base, size] : s.phantom) {
        a.phantom_reserved += size;
    }
    a.alias_pairs = s.pairs.size();
    a.cpu_window_mapped = s.cpu_window_mapped > 0 ? static_cast<uint64_t>(s.cpu_window_mapped) : 0;
    a.gpu_window_mapped = s.gpu_pages.size() * kWindowPageSize;
    a.gpu_mapping_updates = s.gpu_mapping_updates;
    return a;
}

std::string describe() {
    const Accounting a = accounting();
    char b[320];
    std::snprintf(b, sizeof(b),
                  "xmem windows: %" PRIu64 " CPU (%" PRIu64 " MiB reserved, %" PRIu64 " MiB mapped), %" PRIu64
                  " GPU (%" PRIu64 " MiB reserved, %" PRIu64 " MiB mapped), %" PRIu64 " alias pairs, %" PRIu64
                  " MiB phantom",
                  a.cpu_windows, a.cpu_window_reserved >> 20, a.cpu_window_mapped >> 20, a.gpu_windows,
                  a.gpu_window_reserved >> 20, a.gpu_window_mapped >> 20, a.alias_pairs, a.phantom_reserved >> 20);
    return b;
}

FitResult plan_fit(std::span<const PlannedReservation> sequence, uint64_t pool_bytes, uint64_t fixed_bytes,
                   PlacementPolicy policy) {
    FitResult r;
    uint64_t used = fixed_bytes;
    for (size_t i = 0; i < sequence.size(); ++i) {
        const PlannedReservation& q = sequence[i];
        const uint64_t bytes = align_up(q.size, kWindowPageSize);
        WindowKind kind{};
        const bool is_window = classify(q.allocation_type, q.protect, q.size, &kind);
        if (is_window && kind == WindowKind::Gpu &&
            (policy == PlacementPolicy::PhantomGpuWindows || policy == PlacementPolicy::PhantomGpuTranslatedCpu)) {
            r.phantom_bytes += bytes;
            continue;
        }
        if (is_window && kind == WindowKind::Cpu && policy == PlacementPolicy::PhantomGpuTranslatedCpu &&
            r.translated_bytes == 0) {
            r.translated_bytes += bytes;  // one translated window; it takes no pool address space
            continue;
        }
        if (used > pool_bytes || bytes > pool_bytes - used) {
            if (r.first_failure == SIZE_MAX) {
                r.first_failure = i;
            }
            continue;
        }
        used += bytes;
        r.identity_bytes += bytes;
    }
    r.fits = r.first_failure == SIZE_MAX;
    return r;
}

void reset_for_testing() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.policy = default_policy();
    s.next_phantom = kPhantomBase;
    s.phantom.clear();
    s.windows.clear();
    s.pairs.clear();
    s.listeners.clear();
    s.gpu_pages.clear();
    s.gpu_mapping_updates = 0;
    s.cpu_window_mapped = 0;
}

}  // namespace playgta6::os::xmem_windows
