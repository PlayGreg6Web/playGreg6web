#include "gpu/rt_arena.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <utility>

namespace playgta6::gpu::rt_arena {

const char* skip_reason_name(SkipReason r) {
    switch (r) {
    case SkipReason::RtOff: return "rt_off";
    case SkipReason::ArenaFull: return "arena_full";
    case SkipReason::NotResident: return "not_resident";
    case SkipReason::Atomic64: return "atomic64";
    case SkipReason::TranslatorPending: return "translator_pending";
    case SkipReason::Count: break;
    }
    return "?";
}

Arena::Arena(uint32_t arenas, uint32_t pages_per_arena)
    : arenas_(std::min(std::max(arenas, 1u), kArenaCount)),
      pages_per_arena_(std::max(pages_per_arena, 1u)),
      capacity_(static_cast<uint64_t>(arenas_) * pages_per_arena_),
      dir_(kTopEntries, 0) {}

void Arena::touch_dir(uint32_t word) {
    dirty_first_ = std::min(dirty_first_, word);
    dirty_last_ = std::max(dirty_last_, word);
}

bool Arena::take_dir_dirty(uint32_t* first, uint32_t* last) {
    if (dirty_first_ == 0xffffffffu) return false;
    *first = dirty_first_;
    *last = dirty_last_;
    dirty_first_ = 0xffffffffu;
    dirty_last_ = 0;
    return true;
}

bool Arena::set_entry(uint64_t page, uint32_t entry) {
    const uint32_t top = top_index(page);
    uint32_t t = dir_[top];
    if (t == 0) {
        if (entry == 0) return true;
        if (l2_used_ >= kMaxL2Tables) return false;
        t = ++l2_used_;  // 1-based table number
        dir_.resize(static_cast<size_t>(kTopEntries) + static_cast<size_t>(t) * kL2Entries, 0);
        dir_[top] = t;
        touch_dir(top);
    }
    const uint32_t word = kTopEntries + (t - 1) * kL2Entries + l2_index(page);
    dir_[word] = entry;
    touch_dir(word);
    return true;
}

uint32_t Arena::lookup_entry(uint64_t addr) const {
    const uint64_t page = page_of(addr);
    const uint32_t t = dir_[top_index(page)];
    if (t == 0) return 0;
    return dir_[kTopEntries + (t - 1) * kL2Entries + l2_index(page)];
}

bool Arena::alloc_slot(uint32_t* arena, uint32_t* slot) {
    uint64_t packed;
    if (!free_slots_.empty()) {
        packed = free_slots_.back();
        free_slots_.pop_back();
    } else if (next_fresh_ < capacity_) {
        packed = ((next_fresh_ / pages_per_arena_) << 32) | (next_fresh_ % pages_per_arena_);
        ++next_fresh_;
    } else {
        return false;
    }
    *arena = static_cast<uint32_t>(packed >> 32);
    *slot = static_cast<uint32_t>(packed);
    return true;
}

bool Arena::evict_one() {
    for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
        auto p = pages_.find(*it);
        Info& i = p->second;
        if (i.pins > 0 || i.epoch == epoch_ || i.gpu_owned) continue;
        free_slots_.push_back((static_cast<uint64_t>(entry_arena(i.entry)) << 32) | entry_slot(i.entry));
        set_entry(p->first, 0);
        lru_.erase(i.lru);
        pages_.erase(p);
        ++stats_.evictions;
        return true;
    }
    return false;
}

Ensure Arena::ensure_page(uint64_t page, bool pin) {
    if (page >> (kAddressBits - kPageShift)) {
        ++stats_.out_of_range;
        return Ensure::OutOfRange;
    }
    auto it = pages_.find(page);
    if (it != pages_.end()) {
        Info& i = it->second;
        i.last_frame = frame_;
        i.epoch = epoch_;
        lru_.splice(lru_.begin(), lru_, i.lru);
        if (pin) ++i.pins;
        if (i.dirty) {
            i.dirty = false;
            if (i.gpu_owned) {
                ++stats_.skipped_gpu_owned;
            } else {
                uploads_.push_back({page, entry_arena(i.entry), entry_slot(i.entry)});
                ++stats_.uploads;
                return Ensure::Uploaded;
            }
        }
        return Ensure::Resident;
    }
    // A fresh page may need a directory table; refuse before consuming a slot.
    if (dir_[top_index(page)] == 0 && l2_used_ >= kMaxL2Tables) {
        ++stats_.table_full;
        return Ensure::TableFull;
    }
    uint32_t arena = 0, slot = 0;
    while (!alloc_slot(&arena, &slot)) {
        if (!evict_one()) {
            ++stats_.arena_full;
            return Ensure::ArenaFull;
        }
    }
    Info info;
    info.entry = pack_entry(arena, slot);
    info.last_frame = frame_;
    info.epoch = epoch_;
    info.pins = pin ? 1 : 0;
    lru_.push_front(page);
    info.lru = lru_.begin();
    set_entry(page, info.entry);
    pages_.emplace(page, info);
    uploads_.push_back({page, arena, slot});
    ++stats_.uploads;
    return Ensure::Uploaded;
}

Ensure Arena::ensure_range(uint64_t addr, uint64_t size, bool pin) {
    const uint64_t span = size ? size : 1;
    if ((addr & kAddressMask) + span - 1 > kAddressMask) {
        ++stats_.out_of_range;
        return Ensure::OutOfRange;
    }
    const uint64_t first = page_of(addr);
    const uint64_t last = page_of((addr & kAddressMask) + span - 1);
    Ensure worst = Ensure::Resident;
    for (uint64_t p = first; p <= last; ++p) {
        const Ensure e = ensure_page(p, pin);
        if (e == Ensure::ArenaFull || e == Ensure::TableFull || e == Ensure::OutOfRange) return e;
        if (e == Ensure::Uploaded) worst = e;
    }
    return worst;
}

Ensure Arena::pin_structure(uint64_t id, uint64_t addr, uint64_t size) {
    release_structure(id);
    const uint64_t span = size ? size : 1;
    const uint64_t first = page_of(addr);
    const uint64_t last = page_of((addr & kAddressMask) + span - 1);
    std::vector<uint64_t> done;
    Ensure worst = Ensure::Resident;
    for (uint64_t p = first; p <= last; ++p) {
        const Ensure e = ensure_page(p, true);
        if (e == Ensure::ArenaFull || e == Ensure::TableFull || e == Ensure::OutOfRange) {
            for (uint64_t q : done) {
                auto it = pages_.find(q);
                if (it != pages_.end() && it->second.pins > 0) --it->second.pins;
            }
            return e;
        }
        if (e == Ensure::Uploaded) worst = e;
        done.push_back(p);
    }
    structures_[id] = std::move(done);
    return worst;
}

void Arena::release_structure(uint64_t id) {
    auto it = structures_.find(id);
    if (it == structures_.end()) return;
    for (uint64_t p : it->second) {
        auto pi = pages_.find(p);
        if (pi != pages_.end() && pi->second.pins > 0) --pi->second.pins;
    }
    structures_.erase(it);
}

void Arena::mark_gpu_written(uint64_t page) {
    auto it = pages_.find(page);
    if (it != pages_.end()) it->second.gpu_owned = true;
}

void Arena::mark_cpu_remapped(uint64_t addr, uint64_t size) {
    if (size == 0) return;
    for (uint64_t p = page_of(addr); p <= page_of(addr + size - 1); ++p) {
        auto it = pages_.find(p);
        if (it == pages_.end()) continue;
        it->second.gpu_owned = false;
        it->second.dirty = true;
    }
}

void Arena::mark_cpu_dirty(uint64_t page) {
    auto it = pages_.find(page);
    if (it != pages_.end()) it->second.dirty = true;
}

bool Arena::is_resident(uint64_t page) const { return pages_.count(page) != 0; }

std::vector<Upload> Arena::take_uploads() {
    std::vector<Upload> out;
    out.swap(uploads_);
    return out;
}

Stats Arena::stats() const {
    Stats s = stats_;
    s.resident = pages_.size();
    for (const auto& kv : pages_) {
        if (kv.second.pins) ++s.pinned;
        if (kv.second.gpu_owned) ++s.gpu_owned;
    }
    return s;
}

std::string Arena::json() const {
    const Stats s = stats();
    char buf[768];
    std::snprintf(buf, sizeof buf,
                  "{\"arena_bytes\": %llu, \"pages_resident\": %llu, \"pages_pinned\": %llu, "
                  "\"pages_gpu_owned\": %llu, \"uploads\": %llu, \"evictions\": %llu, "
                  "\"skipped_gpu_owned\": %llu, \"arena_full\": %llu, \"table_full\": %llu, "
                  "\"out_of_range\": %llu, \"rt_fault\": %llu, \"dispatches\": %llu, \"unreadable_pages\": %llu, \"dir_words\": %zu, \"skips\": {",
                  (unsigned long long)capacity_bytes(), (unsigned long long)s.resident,
                  (unsigned long long)s.pinned, (unsigned long long)s.gpu_owned,
                  (unsigned long long)s.uploads, (unsigned long long)s.evictions,
                  (unsigned long long)s.skipped_gpu_owned, (unsigned long long)s.arena_full,
                  (unsigned long long)s.table_full, (unsigned long long)s.out_of_range,
                  (unsigned long long)s.faults, (unsigned long long)s.dispatches, (unsigned long long)s.unreadable_pages, dir_.size());
    std::string out = buf;
    for (size_t i = 0; i < static_cast<size_t>(SkipReason::Count); ++i) {
        if (i) out += ", ";
        std::snprintf(buf, sizeof buf, "\"%s\": %llu", skip_reason_name(static_cast<SkipReason>(i)),
                      (unsigned long long)s.skips[i]);
        out += buf;
    }
    return out + "}}";
}

namespace {
std::mutex g_report_mutex;
const void* g_report_owner = nullptr;
std::function<std::string()> g_report_provider;
}  // namespace

void set_report_provider(const void* owner, std::function<std::string()> provider) {
    std::lock_guard<std::mutex> lock(g_report_mutex);
    g_report_owner = owner;
    g_report_provider = std::move(provider);
}

void clear_report_provider(const void* owner) {
    std::lock_guard<std::mutex> lock(g_report_mutex);
    if (g_report_owner == owner) {
        g_report_owner = nullptr;
        g_report_provider = nullptr;
    }
}

std::string report_json() {
    std::function<std::string()> provider;
    {
        std::lock_guard<std::mutex> lock(g_report_mutex);
        provider = g_report_provider;
    }
    return provider ? provider() : std::string("{}");
}

}  // namespace playgta6::gpu::rt_arena
