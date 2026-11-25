#pragma once

// Per-present samples without a GPU capture (R40): the boot report's `present_samples`. Every
// PresentX (the PresentPacer observer) records {present index, wall ms, draws issued since the last
// present, resources created since the last present} into a bounded ring that keeps the last
// `capacity` presents, so tools/screen_timeline.py can segment screens in normal runs.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace playgta6::gpu {

struct PresentSample {
    uint64_t index = 0;
    uint64_t wall_ms = 0;
    uint64_t draws = 0;
    uint64_t resources = 0;
};

class PresentSampleRing {
public:
    static constexpr size_t kDefaultCapacity = 20000;

    explicit PresentSampleRing(size_t capacity = kDefaultCapacity) : capacity_(capacity ? capacity : 1) {}

    // `draws_total` / `resources_total` are the running counters; the sample stores the deltas
    // since the previous record.
    void record(uint64_t wall_ms, uint64_t draws_total, uint64_t resources_total) {
        PresentSample s;
        s.index = total_;
        s.wall_ms = wall_ms;
        s.draws = draws_total >= last_draws_ ? draws_total - last_draws_ : 0;
        s.resources = resources_total >= last_resources_ ? resources_total - last_resources_ : 0;
        last_draws_ = draws_total;
        last_resources_ = resources_total;
        if (samples_.size() < capacity_) {
            samples_.push_back(s);
        } else {
            samples_[next_] = s;
            next_ = (next_ + 1) % capacity_;
        }
        ++total_;
    }

    uint64_t total() const { return total_; }
    size_t size() const { return samples_.size(); }
    size_t capacity() const { return capacity_; }

    // Oldest first.
    std::vector<PresentSample> snapshot() const {
        std::vector<PresentSample> out;
        out.reserve(samples_.size());
        for (size_t i = 0; i < samples_.size(); ++i) {
            out.push_back(samples_[(next_ + i) % samples_.size()]);
        }
        return out;
    }

    void clear() {
        samples_.clear();
        next_ = 0;
        total_ = last_draws_ = last_resources_ = 0;
    }

private:
    size_t capacity_;
    std::vector<PresentSample> samples_;
    size_t next_ = 0;
    uint64_t total_ = 0;
    uint64_t last_draws_ = 0;
    uint64_t last_resources_ = 0;
};

}  // namespace playgta6::gpu
