// Present pacing (O36). Design: present_pacing.h.

#include "gpu/present_pacing.h"

#include <algorithm>
#include <limits>

#include "os/host_sleep.h"
#include "os/thread_spans.h"

namespace playgta6::gpu {

namespace {

PresentPacer::Clock::time_point add_ns_clamped(PresentPacer::Clock::time_point base, int64_t ns) {
    if (ns <= 0) {
        return base;
    }
    const auto headroom =
        std::chrono::duration_cast<std::chrono::nanoseconds>(PresentPacer::Clock::time_point::max() - base)
            .count();
    if (headroom <= 0) {
        return base;
    }
    return base + std::chrono::nanoseconds(ns > headroom ? headroom : ns);
}

}  // namespace

PresentPacer::PresentPacer() {
    reset_for_testing();
}

void PresentPacer::reset_for_testing() {
    std::lock_guard<std::mutex> lock(mutex_);
    origin_ = Clock::now();
    now_ = [] { return Clock::now(); };
    sleep_until_ = [](Clock::time_point t) { os::host_sleep::sleep_until(t); };  // precise (R40)
    enabled_ = true;
    have_last_ = false;
    have_last_present_time_ = false;
    title_driven_ = false;
    honor_interval_ = false;
    last_flip_ = 0;
    last_present_time_ = Clock::time_point{};
    ++generation_;
    stats_ = Stats{};
}

void PresentPacer::set_clock_for_testing(std::function<Clock::time_point()> now,
                                         std::function<void(Clock::time_point)> sleep_until,
                                         Clock::time_point origin) {
    std::lock_guard<std::mutex> lock(mutex_);
    now_ = std::move(now);
    sleep_until_ = std::move(sleep_until);
    origin_ = origin;
    have_last_ = false;
    have_last_present_time_ = false;
    last_flip_ = 0;
    last_present_time_ = Clock::time_point{};
    ++generation_;
    stats_ = Stats{};
}

void PresentPacer::set_title_driven(bool on) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (title_driven_ == on) {
        return;
    }
    title_driven_ = on;
    have_last_ = false;
    last_flip_ = 0;
    have_last_present_time_ = false;
    last_present_time_ = Clock::time_point{};
    ++generation_;
}

void PresentPacer::set_honor_interval(bool on) {
    std::lock_guard<std::mutex> lock(mutex_);
    honor_interval_ = on;
}

bool PresentPacer::honor_interval() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return honor_interval_;
}

bool PresentPacer::title_driven() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return title_driven_;
}

uint64_t PresentPacer::vblank_at_locked(Clock::time_point t) const {
    if (t <= origin_) {
        return 0;
    }
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t - origin_).count();
    if (ns <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(ns / kVblankNs);
}

PresentPacer::Clock::time_point PresentPacer::vblank_start_locked(uint64_t v) const {
    const auto headroom =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::time_point::max() - origin_)
            .count();
    if (headroom <= 0) {
        return origin_;
    }
    const uint64_t max_v = static_cast<uint64_t>(headroom / kVblankNs);
    const uint64_t clamped_v = v > max_v ? max_v : v;
    return origin_ + std::chrono::nanoseconds(static_cast<int64_t>(clamped_v) * kVblankNs);
}

uint64_t PresentPacer::vblank_at(Clock::time_point t) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vblank_at_locked(t);
}

PresentPacer::Clock::time_point PresentPacer::vblank_start(uint64_t v) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vblank_start_locked(v);
}

uint64_t PresentPacer::present(uint32_t interval) {
    if (Observer observer = observer_.load(std::memory_order_acquire)) {
        observer();
    }
    std::unique_lock<std::mutex> lock(mutex_);
    const Clock::time_point now = now_();
    const uint64_t current = vblank_at_locked(now);
    if (!enabled_) {
        return current;
    }
    ++stats_.presents;
    stats_.last_interval = interval;
    const uint32_t clamped_interval = interval > kMaxInterval ? kMaxInterval : interval;
    if (title_driven_) {
        if (interval == 0) {
            ++stats_.immediate;
            last_present_time_ = now;
            have_last_present_time_ = true;
            return current;
        }
        const int64_t gap_ns =
            static_cast<int64_t>(title_gap_vblanks(clamped_interval, honor_interval_)) * kVblankNs;
        if (!have_last_present_time_ ||
            (now >= last_present_time_ &&
             std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_present_time_).count() >=
                 gap_ns)) {
            ++stats_.immediate;
            last_present_time_ = now;
            have_last_present_time_ = true;
            return current;
        }
        // Bound the sleep lead to at most `gap_ns` (<= kMaxSleepNs) ahead of `now`. If a concurrent
        // caller, early wakeup, or backward clock step left `last_present_time_ > now`, re-anchor
        // relative to `now` instead of stacking runaway future deadlines.
        const Clock::time_point max_until = add_ns_clamped(now, gap_ns);
        const Clock::time_point until =
            (last_present_time_ > now)
                ? max_until
                : std::min(add_ns_clamped(last_present_time_, gap_ns), max_until);
        last_present_time_ = until;
        have_last_present_time_ = true;
        const uint64_t flip_vblank = vblank_at_locked(until);
        const uint64_t gen = generation_;
        auto now_fn = now_;
        auto sleep_until = sleep_until_;
        lock.unlock();
        {
            const os::ThreadSpan frame_span(os::SpanKind::Pacer);
            sleep_until(until);
        }
        const Clock::time_point after = now_fn();
        lock.lock();
        if (generation_ == gen) {
            ++stats_.waits;
            const auto waited_us =
                std::chrono::duration_cast<std::chrono::microseconds>(after - now).count();
            if (waited_us > 0) {
                stats_.wait_us += static_cast<uint64_t>(waited_us);
            }
        }
        return flip_vblank;
    }
    if (interval == 0) {
        ++stats_.immediate;
        last_flip_ = current;
        have_last_ = true;
        return current;
    }
    uint64_t target =
        have_last_
            ? (last_flip_ <= std::numeric_limits<uint64_t>::max() - clamped_interval
                   ? last_flip_ + clamped_interval
                   : std::numeric_limits<uint64_t>::max())
            : (current < std::numeric_limits<uint64_t>::max() ? current + 1 : current);
    if (target <= current) {
        const uint64_t next_vblank =
            current < std::numeric_limits<uint64_t>::max() ? current + 1 : current;
        stats_.missed_vblanks += next_vblank - target;
        target = next_vblank;
    }
    // Never walk `target` more than `clamped_interval` vblanks ahead of `current` (guards against
    // concurrent callers stacking future flips or clock regressions).
    const uint64_t max_target =
        current <= std::numeric_limits<uint64_t>::max() - clamped_interval
            ? current + clamped_interval
            : std::numeric_limits<uint64_t>::max();
    if (target > max_target) {
        target = max_target;
    }
    last_flip_ = target;
    have_last_ = true;
    const Clock::time_point max_until = add_ns_clamped(now, kMaxSleepNs);
    const Clock::time_point vblank_until = vblank_start_locked(target);
    const Clock::time_point until = vblank_until > max_until ? max_until : vblank_until;
    const uint64_t gen = generation_;
    auto now_fn = now_;
    auto sleep_until = sleep_until_;
    lock.unlock();
    {
        const os::ThreadSpan frame_span(os::SpanKind::Pacer);
        sleep_until(until);
    }
    const Clock::time_point after = now_fn();
    lock.lock();
    if (generation_ == gen) {
        ++stats_.waits;
        const auto waited_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after - now).count();
        if (waited_us > 0) {
            stats_.wait_us += static_cast<uint64_t>(waited_us);
        }
    }
    return target;
}

PresentPacer::Stats PresentPacer::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void PresentPacer::set_enabled(bool on) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = on;
    have_last_ = false;
    last_flip_ = 0;
    have_last_present_time_ = false;
    last_present_time_ = Clock::time_point{};
    ++generation_;
}

bool PresentPacer::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

void PresentPacer::set_observer(Observer observer) {
    observer_.store(observer, std::memory_order_release);
}

PresentPacer& present_pacer() {
    static PresentPacer pacer;
    return pacer;
}

}  // namespace playgta6::gpu
