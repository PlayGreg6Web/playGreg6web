// Present pacing (O36). Design: present_pacing.h.

#include "gpu/present_pacing.h"

#include <thread>

namespace playgta6::gpu {

PresentPacer::PresentPacer() {
    reset_for_testing();
}

void PresentPacer::reset_for_testing() {
    std::lock_guard<std::mutex> lock(mutex_);
    origin_ = Clock::now();
    now_ = [] { return Clock::now(); };
    sleep_until_ = [](Clock::time_point t) { std::this_thread::sleep_until(t); };
    enabled_ = true;
    have_last_ = false;
    last_flip_ = 0;
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
    last_flip_ = 0;
    stats_ = Stats{};
}

uint64_t PresentPacer::vblank_at(Clock::time_point t) const {
    if (t <= origin_) {
        return 0;
    }
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t - origin_).count();
    return static_cast<uint64_t>(ns / kVblankNs);
}

PresentPacer::Clock::time_point PresentPacer::vblank_start(uint64_t v) const {
    return origin_ + std::chrono::nanoseconds(static_cast<int64_t>(v) * kVblankNs);
}

uint64_t PresentPacer::present(uint32_t interval) {
    if (Observer observer = observer_.load(std::memory_order_acquire)) {
        observer();
    }
    std::unique_lock<std::mutex> lock(mutex_);
    const Clock::time_point now = now_();
    const uint64_t current = vblank_at(now);
    if (!enabled_) {
        return current;
    }
    ++stats_.presents;
    stats_.last_interval = interval;
    if (interval == 0) {
        ++stats_.immediate;
        last_flip_ = current;
        have_last_ = true;
        return current;
    }
    uint64_t target = have_last_ ? last_flip_ + interval : current + 1;
    if (target <= current) {
        stats_.missed_vblanks += current + 1 - target;
        target = current + 1;
    }
    last_flip_ = target;
    have_last_ = true;
    const Clock::time_point until = vblank_start(target);
    auto sleep_until = sleep_until_;
    lock.unlock();
    sleep_until(until);
    lock.lock();
    ++stats_.waits;
    stats_.wait_us += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(until - now).count());
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
