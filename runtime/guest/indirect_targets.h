#pragma once

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// Record of the indirect-call targets a run actually dispatched (`--record-indirect-targets`).
//
// File: text, one `0x<site> 0x<target>` pair per line (hex, `#` comments). `site` is the return
// address the guest `call` pushed (what [rsp] held at the dispatch), `target` the guest address
// that was dispatched; imports and host callables (synthetic addresses) are not recorded. The
// recompiler reads it back with `build-program --seed-targets` (recomp/src/trace_seed.rs) and
// lifts the targets as function entries. The file is derived from running the game: keep it in
// the per-title cache directory, never in git.
//
// Cost: each guest thread owns an IndirectTargetCollector (a hash set; one lookup per dispatch,
// nothing shared). New pairs are handed to the shared IndirectTargetLog in batches (every
// dispatch batch flush, at thread exit and at a fault) and the file is appended to at the end of
// the run and at a fault, only with pairs it does not hold yet.
namespace playgta6::guest {

struct IndirectTargetPair {
    uint64_t site = 0;
    uint64_t target = 0;
    bool operator<(const IndirectTargetPair& o) const {
        return site != o.site ? site < o.site : target < o.target;
    }
    bool operator==(const IndirectTargetPair& o) const { return site == o.site && target == o.target; }
};

struct IndirectTargetPairHash {
    size_t operator()(const IndirectTargetPair& p) const noexcept {
        const uint64_t h = (p.site * 0x9E3779B97F4A7C15ull) ^ ((p.target + 0x7F4A7C15ull) * 0xC2B2AE3D27D4EB4Full);
        return static_cast<size_t>(h ^ (h >> 29));
    }
};

// Per-thread, lock-free: remembers what it has seen and queues the new pairs.
class IndirectTargetCollector {
public:
    void note(uint64_t site, uint64_t target) {
        const IndirectTargetPair p{site, target};
        if (seen_.insert(p).second) {
            pending_.push_back(p);
        }
    }
    bool has_pending() const noexcept { return !pending_.empty(); }
    std::vector<IndirectTargetPair> take_pending() {
        std::vector<IndirectTargetPair> out;
        out.swap(pending_);
        return out;
    }
    void reset() {
        seen_.clear();
        pending_.clear();
    }

private:
    std::unordered_set<IndirectTargetPair, IndirectTargetPairHash> seen_;
    std::vector<IndirectTargetPair> pending_;
};

// Shared sink: the union of every thread's pairs plus what the file already holds.
class IndirectTargetLog {
public:
    explicit IndirectTargetLog(std::string path) : path_(std::move(path)) { load_existing(); }

    const std::string& path() const noexcept { return path_; }

    void merge(const std::vector<IndirectTargetPair>& pairs) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& p : pairs) {
            if (!on_disk_.count(p)) {
                fresh_.insert(p);
            }
        }
    }

    // Appends the pairs the file does not hold yet; returns how many were written (-1: the file
    // cannot be opened).
    long flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fresh_.empty()) {
            return 0;
        }
        std::FILE* f = std::fopen(path_.c_str(), "ab");
        if (f == nullptr) {
            return -1;
        }
        if (!existed_) {
            existed_ = true;
            std::fputs("# playgta6 recorded indirect targets: <site (return address)> <target>\n", f);
        }
        long n = 0;
        for (const auto& p : fresh_) {
            std::fprintf(f, "0x%" PRIx64 " 0x%" PRIx64 "\n", p.site, p.target);
            on_disk_.insert(p);
            ++n;
        }
        std::fclose(f);
        fresh_.clear();
        return n;
    }

    size_t known() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return on_disk_.size() + fresh_.size();
    }

private:
    void load_existing() {
        std::FILE* f = std::fopen(path_.c_str(), "rb");
        if (f == nullptr) {
            return;
        }
        existed_ = true;
        char line[256];
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            unsigned long long site = 0;
            unsigned long long target = 0;
            if (line[0] != '#' && std::sscanf(line, "%llx %llx", &site, &target) == 2) {
                on_disk_.insert({site, target});
            }
        }
        std::fclose(f);
    }

    std::string path_;
    bool existed_ = false;
    mutable std::mutex mutex_;
    std::set<IndirectTargetPair> on_disk_;
    std::set<IndirectTargetPair> fresh_;
};

}  // namespace playgta6::guest
