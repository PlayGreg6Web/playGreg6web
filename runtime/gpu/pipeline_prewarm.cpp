#include "gpu/pipeline_prewarm.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <tuple>

namespace playgta6::gpu {

namespace {
void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
void put64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
uint32_t get32(const uint8_t* p) {
    return uint32_t{p[0]} | uint32_t{p[1]} << 8 | uint32_t{p[2]} << 16 | uint32_t{p[3]} << 24;
}
uint64_t get64(const uint8_t* p) { return uint64_t{get32(p)} | uint64_t{get32(p + 4)} << 32; }
// FNV-1a over the detail bytes, never 0 (0 marks a module-only entry).
uint64_t hash_detail(const std::vector<uint8_t>& d) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t b : d) {
        h ^= b;
        h *= 0x100000001b3ull;
    }
    return h == 0 ? 1 : h;
}
}  // namespace

void PipelinePrewarm::set_pack_index(std::shared_ptr<const ShaderPackIndex> index) {
    std::lock_guard<std::mutex> lock(mu_);
    pack_ = std::move(index);
}

void PipelinePrewarm::set_max_in_flight(uint32_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    max_in_flight_ = std::max<uint32_t>(n, 1);
}

bool PipelinePrewarm::resolvable(const Identity& i) const {
    if (!pack_) return true;
    if (pack_->find(i.k0) == pack_->end()) return false;
    return i.k1 == 0 || pack_->find(i.k1) != pack_->end();
}

bool PipelinePrewarm::load_warm_list(const uint8_t* data, size_t size, std::string* error, uint32_t* queued) {
    auto fail = [&](const char* msg) {
        if (error) *error = msg;
        return false;
    };
    if (queued) *queued = 0;
    if (data == nullptr || size < kPrewarmListHeaderBytes) return fail("warm list too short");
    if (std::memcmp(data, "PG6W", 4) != 0) return fail("warm list bad magic");
    const uint32_t version = get32(data + 4);
    if (version != kPrewarmListVersionModules && version != kPrewarmListVersionPipelines) {
        return fail("warm list unsupported version");
    }
    const uint32_t count = get32(data + 8);
    if (count > kPrewarmMaxEntries) return fail("warm list too many entries");
    if (size < kPrewarmListHeaderBytes + uint64_t{count} * kPrewarmListEntryBytes) return fail("warm list truncated");
    struct Parsed {
        Identity id;
        PrewarmDetail detail;
    };
    std::vector<Parsed> parsed;
    parsed.reserve(count);
    size_t pos = kPrewarmListHeaderBytes;
    for (uint32_t i = 0; i < count; ++i) {
        if (size - pos < kPrewarmListEntryBytes) return fail("warm list truncated");
        const uint8_t* e = data + pos;
        pos += kPrewarmListEntryBytes;
        Parsed p{Identity{e[0], get64(e + 8), get64(e + 16), 0}, nullptr};
        if (p.id.kind > kPrewarmKindCompute || p.id.k0 == 0) return fail("warm list bad entry");
        if (version == kPrewarmListVersionPipelines) {
            const uint32_t detail_len = get32(e + 4);
            if (detail_len > kPrewarmMaxDetailBytes) return fail("warm list bad entry");
            if (size - pos < detail_len) return fail("warm list truncated");
            if (detail_len != 0) {
                auto d = std::make_shared<std::vector<uint8_t>>(data + pos, data + pos + detail_len);
                p.id.k2 = hash_detail(*d);
                p.detail = std::move(d);
            }
            pos += detail_len;
        }
        parsed.push_back(std::move(p));
    }
    std::lock_guard<std::mutex> lock(mu_);
    uint32_t added = 0;
    for (Parsed& p : parsed) {
        const Identity id = p.id;
        if (by_identity_.count(id) != 0) continue;
        const bool seen = observed_set_.count(id) != 0;
        slots_.push_back({id, State::Queued, std::move(p.detail)});
        const uint32_t task = static_cast<uint32_t>(slots_.size());
        by_identity_.emplace(id, task);
        if (seen || !resolvable(id)) {
            slots_.back().state = State::Skipped;
            continue;
        }
        queue_.push_back(task);
        ++added;
    }
    if (queued) *queued = added;
    return true;
}

std::vector<uint8_t> PipelinePrewarm::save_warm_list() const {
    std::lock_guard<std::mutex> lock(mu_);
    // A module-only entry is redundant when the list carries a full pipeline for the same shaders
    // (building the pipeline builds the modules).
    std::set<std::tuple<uint32_t, uint64_t, uint64_t>> full;
    for (const Identity& id : observed_) {
        if (id.k2 != 0) full.emplace(id.kind, id.k0, id.k1);
    }
    std::vector<size_t> keep;
    keep.reserve(observed_.size());
    bool any_detail = false;
    for (size_t i = 0; i < observed_.size(); ++i) {
        const Identity& id = observed_[i];
        if (id.k2 == 0 && full.count({id.kind, id.k0, id.k1}) != 0) continue;
        if (id.k2 != 0) any_detail = true;
        keep.push_back(i);
    }
    std::vector<uint8_t> out;
    out.reserve(kPrewarmListHeaderBytes + keep.size() * kPrewarmListEntryBytes);
    out.push_back('P');
    out.push_back('G');
    out.push_back('6');
    out.push_back('W');
    put32(out, any_detail ? kPrewarmListVersionPipelines : kPrewarmListVersionModules);
    put32(out, static_cast<uint32_t>(keep.size()));
    put32(out, 0);
    for (size_t i : keep) {
        const Identity& id = observed_[i];
        const PrewarmDetail& detail = observed_detail_[i];
        out.push_back(static_cast<uint8_t>(id.kind));
        if (any_detail) {
            for (int j = 0; j < 3; ++j) out.push_back(0);
            put32(out, detail ? static_cast<uint32_t>(detail->size()) : 0);
        } else {
            for (int j = 0; j < 7; ++j) out.push_back(0);
        }
        put64(out, id.k0);
        put64(out, id.k1);
        if (any_detail && detail) out.insert(out.end(), detail->begin(), detail->end());
    }
    return out;
}

bool PipelinePrewarm::note_locked(const Identity& id, PrewarmDetail detail) {
    if (!observed_set_.emplace(id, static_cast<uint32_t>(observed_.size())).second) return false;
    if (observed_.size() < kPrewarmMaxEntries) {
        observed_.push_back(id);
        observed_detail_.push_back(std::move(detail));
    }
    auto it = by_identity_.find(id);
    if (it != by_identity_.end()) {
        Slot& s = slots_[it->second - 1];
        if (s.state == State::Queued) {
            s.state = State::Skipped;  // the title got there first; dropped from the queue on take()
        } else if (s.state == State::Done) {
            s.state = State::Used;  // built ahead of the title's first use
        }
    }
    return true;
}

bool PipelinePrewarm::note_title_pipeline(uint32_t kind, uint64_t key0, uint64_t key1) {
    if (key0 == 0 || kind > kPrewarmKindCompute) return false;
    std::lock_guard<std::mutex> lock(mu_);
    return note_locked(Identity{kind, key0, key1, 0}, nullptr);
}

bool PipelinePrewarm::note_title_pipeline_full(uint32_t kind, uint64_t key0, uint64_t key1,
                                               std::vector<uint8_t> detail) {
    if (detail.empty()) return note_title_pipeline(kind, key0, key1);
    if (key0 == 0 || kind > kPrewarmKindCompute || detail.size() > kPrewarmMaxDetailBytes) return false;
    const Identity id{kind, key0, key1, hash_detail(detail)};
    auto d = std::make_shared<const std::vector<uint8_t>>(std::move(detail));
    std::lock_guard<std::mutex> lock(mu_);
    return note_locked(id, std::move(d));
}

bool PipelinePrewarm::note_task_used(uint32_t task_id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (task_id == 0 || task_id > slots_.size()) return false;
    const Slot& s = slots_[task_id - 1];
    return note_locked(s.id, s.detail);
}

PrewarmDetail PipelinePrewarm::detail_of(uint32_t task_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (task_id == 0 || task_id > slots_.size()) return nullptr;
    return slots_[task_id - 1].detail;
}

size_t PipelinePrewarm::take(PrewarmTask* out, size_t max) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t n = 0;
    while (n < max && in_flight_ < max_in_flight_ && !queue_.empty()) {
        const uint32_t task = queue_.front();
        queue_.pop_front();
        Slot& s = slots_[task - 1];
        if (s.state != State::Queued) continue;
        s.state = State::InFlight;
        ++in_flight_;
        out[n++] = PrewarmTask{task, s.id.kind, s.id.k0, s.id.k1};
    }
    return n;
}

bool PipelinePrewarm::complete(uint32_t id, bool ok, uint64_t build_ns) {
    std::lock_guard<std::mutex> lock(mu_);
    if (id == 0 || id > slots_.size()) return false;
    Slot& s = slots_[id - 1];
    if (s.state != State::InFlight) return false;
    s.state = !ok ? State::Failed : observed_set_.count(s.id) != 0 ? State::Used : State::Done;  // used while building
    --in_flight_;
    longest_ns_ = std::max(longest_ns_, build_ns);
    total_ns_ += build_ns;
    return true;
}

bool PipelinePrewarm::complete_lost(uint32_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (id == 0 || id > slots_.size()) return false;
    Slot& s = slots_[id - 1];
    if (s.state != State::InFlight) return false;
    s.state = State::Skipped;
    --in_flight_;
    return true;
}

PrewarmProgress PipelinePrewarm::progress() const {
    std::lock_guard<std::mutex> lock(mu_);
    PrewarmProgress p;
    p.total = static_cast<uint32_t>(slots_.size());
    for (const Slot& s : slots_) {
        switch (s.state) {
            case State::Queued: ++p.queued; break;
            case State::InFlight: ++p.in_flight; break;
            case State::Done:
            case State::Used: ++p.done; break;
            case State::Failed: ++p.failed; break;
            case State::Skipped: ++p.skipped; break;
        }
    }
    p.observed = static_cast<uint32_t>(observed_.size());
    p.longest_ns = longest_ns_;
    p.total_ns = total_ns_;
    return p;
}

PrewarmUse PipelinePrewarm::use_stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    PrewarmUse u;
    for (const Slot& s : slots_) {
        const bool full = s.id.k2 != 0;
        if (s.state == State::Done) ++(full ? u.pipelines_wasted : u.wasted);
        if (s.state == State::Used) ++(full ? u.pipelines_hit : u.hit);
    }
    u.prewarmed = u.hit + u.wasted;
    u.pipelines_prewarmed = u.pipelines_hit + u.pipelines_wasted;
    return u;
}

void PipelinePrewarm::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    slots_.clear();
    by_identity_.clear();
    queue_.clear();
    observed_.clear();
    observed_detail_.clear();
    observed_set_.clear();
    in_flight_ = 0;
    longest_ns_ = 0;
    total_ns_ = 0;
}

PipelinePrewarm& pipeline_prewarm() {
    static PipelinePrewarm instance;
    return instance;
}

}  // namespace playgta6::gpu
