#include "gpu/pipeline_prewarm.h"

#include <algorithm>
#include <cstring>

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
    if (get32(data + 4) != 1) return fail("warm list unsupported version");
    const uint32_t count = get32(data + 8);
    if (count > kPrewarmMaxEntries) return fail("warm list too many entries");
    if (size < kPrewarmListHeaderBytes + uint64_t{count} * kPrewarmListEntryBytes) return fail("warm list truncated");
    std::vector<Identity> parsed;
    parsed.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* e = data + kPrewarmListHeaderBytes + size_t{i} * kPrewarmListEntryBytes;
        Identity id{e[0], get64(e + 8), get64(e + 16)};
        if (id.kind > kPrewarmKindCompute || id.k0 == 0) return fail("warm list bad entry");
        parsed.push_back(id);
    }
    std::lock_guard<std::mutex> lock(mu_);
    uint32_t added = 0;
    for (const Identity& id : parsed) {
        if (by_identity_.count(id) != 0) continue;
        const bool seen = observed_set_.count(id) != 0;
        slots_.push_back({id, State::Queued});
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
    std::vector<uint8_t> out;
    out.reserve(kPrewarmListHeaderBytes + observed_.size() * kPrewarmListEntryBytes);
    out.push_back('P');
    out.push_back('G');
    out.push_back('6');
    out.push_back('W');
    put32(out, 1);
    put32(out, static_cast<uint32_t>(observed_.size()));
    put32(out, 0);
    for (const Identity& id : observed_) {
        out.push_back(static_cast<uint8_t>(id.kind));
        for (int i = 0; i < 7; ++i) out.push_back(0);
        put64(out, id.k0);
        put64(out, id.k1);
    }
    return out;
}

bool PipelinePrewarm::note_title_pipeline(uint32_t kind, uint64_t key0, uint64_t key1) {
    if (key0 == 0 || kind > kPrewarmKindCompute) return false;
    const Identity id{kind, key0, key1};
    std::lock_guard<std::mutex> lock(mu_);
    if (!observed_set_.emplace(id, static_cast<uint32_t>(observed_.size())).second) return false;
    if (observed_.size() < kPrewarmMaxEntries) observed_.push_back(id);
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
        if (s.state == State::Done) ++u.wasted;
        if (s.state == State::Used) ++u.hit;
    }
    u.prewarmed = u.hit + u.wasted;
    return u;
}

void PipelinePrewarm::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    slots_.clear();
    by_identity_.clear();
    queue_.clear();
    observed_.clear();
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
