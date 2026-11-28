#include "gpu/gpu_time.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace playgta6::gpu::d3d12_guest {

const char* gpu_pass_kind_name(GpuPassKind k) {
    switch (k) {
    case GpuPassKind::Render: return "render";
    case GpuPassKind::Compute: return "compute";
    case GpuPassKind::Copy: return "copy";
    case GpuPassKind::Count: break;
    }
    return "?";
}

GpuTimeTable& gpu_time_table() {
    static GpuTimeTable t;
    return t;
}

namespace {
// Key = hash with the kind folded in (hash * 4 + kind would collide at the top bits, so mix).
uint64_t key_of(uint64_t hash, GpuPassKind kind) {
    return (hash * 1099511628211ull) ^ (static_cast<uint64_t>(kind) + 0x9e3779b97f4a7c15ull);
}
std::string hex16(uint64_t h) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%016" PRIx64, h);
    return b;
}
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (static_cast<unsigned char>(c) < 0x20) continue;
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}
}  // namespace

void GpuTimeTable::set_support(Support s, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mu_);
    support_ = s;
    reason_ = reason;
}
GpuTimeTable::Support GpuTimeTable::support() const {
    std::lock_guard<std::mutex> lock(mu_);
    return support_;
}

void GpuTimeTable::record(uint64_t hash, GpuPassKind kind, uint64_t ns) {
    std::lock_guard<std::mutex> lock(mu_);
    support_ = Support::Available;
    Stats* s = &other_;
    const uint64_t k = key_of(hash, kind);
    auto it = map_.find(k);
    if (it != map_.end()) {
        s = &it->second;
    } else if (map_.size() < kMaxKeys) {
        keys_[k] = {hash, kind};
        s = &map_[k];
    }
    s->passes += 1;
    s->total_ns += ns;
    s->max_ns = std::max(s->max_ns, ns);
}

bool GpuTimeTable::lookup(uint64_t hash, GpuPassKind kind, Stats* out) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = map_.find(key_of(hash, kind));
    if (it == map_.end() || it->second.passes == 0) return false;
    *out = it->second;
    return true;
}

void GpuTimeTable::note_skipped_frame() {
    std::lock_guard<std::mutex> lock(mu_);
    ++frames_skipped_;
}
void GpuTimeTable::note_skipped_passes(uint64_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    passes_skipped_ += n;
}
void GpuTimeTable::note_measured_frame(uint64_t gpu_ns) {
    std::lock_guard<std::mutex> lock(mu_);
    ++frames_measured_;
    frame_ns_ += gpu_ns;
}

void GpuTimeTable::measured_totals(uint64_t* frames, uint64_t* gpu_ns) const {
    std::lock_guard<std::mutex> lock(mu_);
    *frames = frames_measured_;
    *gpu_ns = frame_ns_;
}

void GpuTimeTable::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    support_ = Support::Unknown;
    reason_.clear();
    map_.clear();
    keys_.clear();
    other_ = Stats{};
    frames_measured_ = frames_skipped_ = passes_skipped_ = frame_ns_ = 0;
}

size_t GpuTimeTable::tracked() const {
    std::lock_guard<std::mutex> lock(mu_);
    return map_.size();
}

std::vector<GpuTimeTable::Entry> GpuTimeTable::top(size_t n) const {
    std::vector<Entry> v;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto& e : map_) {
            const auto& k = keys_.at(e.first);
            v.push_back({k.first, k.second, e.second});
        }
    }
    std::sort(v.begin(), v.end(), [](const Entry& a, const Entry& b) {
        if (a.stats.total_ns != b.stats.total_ns) return a.stats.total_ns > b.stats.total_ns;
        if (a.hash != b.hash) return a.hash < b.hash;
        return a.kind < b.kind;
    });
    if (v.size() > n) v.resize(n);
    return v;
}

std::string GpuTimeTable::json(size_t top_n) const {
    Support sup;
    std::string reason;
    Stats other;
    uint64_t fm, fs, ps, fns, total = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        sup = support_;
        reason = reason_;
        other = other_;
        fm = frames_measured_;
        fs = frames_skipped_;
        ps = passes_skipped_;
        fns = frame_ns_;
        for (const auto& e : map_) total += e.second.total_ns;
    }
    total += other.total_ns;
    const char* name = sup == Support::Available ? "available" : sup == Support::Unavailable ? "unavailable" : "unknown";
    if (reason.empty() && sup != Support::Available) {
        reason = "no timestamp-query results were fed (no GPU executor attached, or the adapter lacks timestamp-query)";
    }
    std::string o = std::string("{\"available\": ") + (sup == Support::Available ? "true" : "false") +
                    ", \"support\": \"" + name + "\"";
    if (sup != Support::Available) o += ", \"reason\": \"" + esc(reason) + "\"";
    o += ", \"frames_measured\": " + std::to_string(fm) + ", \"frames_skipped\": " + std::to_string(fs) +
         ", \"passes_skipped\": " + std::to_string(ps) + ", \"frame_gpu_ns\": " + std::to_string(fns) +
         ", \"total_ns\": " + std::to_string(total) + ", \"tracked\": " + std::to_string(tracked()) +
         ", \"other\": {\"passes\": " + std::to_string(other.passes) +
         ", \"total_ns\": " + std::to_string(other.total_ns) + "}, \"top\": [";
    bool first = true;
    for (const auto& e : top(top_n)) {
        o += std::string(first ? "" : ", ") + "{\"hash\": \"" + hex16(e.hash) + "\", \"kind\": \"" +
             gpu_pass_kind_name(e.kind) + "\", \"passes\": " + std::to_string(e.stats.passes) +
             ", \"total_ns\": " + std::to_string(e.stats.total_ns) + ", \"max_ns\": " + std::to_string(e.stats.max_ns) +
             ", \"avg_ns\": " + std::to_string(e.stats.passes ? e.stats.total_ns / e.stats.passes : 0) + "}";
        first = false;
    }
    return o + "]}";
}

GpuTimeRing::GpuTimeRing(size_t slots, size_t max_passes_per_frame)
    : slots_(std::max<size_t>(slots, 1)), max_passes_(std::max<size_t>(max_passes_per_frame, 1)) {}

int GpuTimeRing::begin_frame() {
    const size_t i = next_;
    next_ = (next_ + 1) % slots_.size();
    if (slots_[i].in_flight) {
        open_ = kNoSlot;
        return kNoSlot;
    }
    slots_[i].passes.clear();
    open_ = static_cast<int>(i);
    return open_;
}

bool GpuTimeRing::add_pass(uint64_t hash, GpuPassKind kind, uint32_t* b, uint32_t* e) {
    if (open_ == kNoSlot) return false;
    Slot& s = slots_[static_cast<size_t>(open_)];
    if (s.passes.size() >= max_passes_) {
        ++overflow_;
        return false;
    }
    const uint32_t q = static_cast<uint32_t>(s.passes.size() * 2);
    s.passes.emplace_back(hash, kind);
    if (b) *b = q;
    if (e) *e = q + 1;
    return true;
}

size_t GpuTimeRing::end_frame(int slot) {
    if (slot < 0 || static_cast<size_t>(slot) >= slots_.size()) return 0;
    Slot& s = slots_[static_cast<size_t>(slot)];
    s.in_flight = true;
    if (open_ == slot) open_ = kNoSlot;
    return s.passes.size();
}

void GpuTimeRing::complete(int slot, const uint64_t* ticks, size_t count, double period_ns, GpuTimeTable& table) {
    if (slot < 0 || static_cast<size_t>(slot) >= slots_.size()) return;
    Slot& s = slots_[static_cast<size_t>(slot)];
    if (!s.in_flight) return;
    uint64_t skipped = 0, frame_ns = 0;
    for (size_t p = 0; p < s.passes.size(); ++p) {
        if (ticks && p * 2 + 1 < count && ticks[p * 2] != 0 && ticks[p * 2 + 1] > ticks[p * 2]) {
            const uint64_t ns = static_cast<uint64_t>(static_cast<double>(ticks[p * 2 + 1] - ticks[p * 2]) * period_ns);
            table.record(s.passes[p].first, s.passes[p].second, ns);
            frame_ns += ns;
        } else {
            ++skipped;
        }
    }
    if (skipped) table.note_skipped_passes(skipped);
    table.note_measured_frame(frame_ns);
    s.in_flight = false;
    s.passes.clear();
}

void GpuTimeRing::abandon(int slot, GpuTimeTable& table) {
    if (slot < 0 || static_cast<size_t>(slot) >= slots_.size()) return;
    Slot& s = slots_[static_cast<size_t>(slot)];
    if (!s.in_flight) return;
    table.note_skipped_passes(s.passes.size());
    table.note_skipped_frame();
    s.in_flight = false;
    s.passes.clear();
}

}  // namespace playgta6::gpu::d3d12_guest
