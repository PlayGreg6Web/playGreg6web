#pragma once

// Present pacing (O36): a present honours the title's sync interval against a 60 Hz display, as
// the console does. Interval 0 does not wait; interval n waits for the n-th vblank after the
// previous paced present (1 = the next vblank). A present that comes after that vblank has passed
// flips at the next one, and the vblanks it was late by are counted as missed.
//
// The vblank clock is monotonic and process-wide (one display), the same in the native and the
// browser build, so title logic that counts frames sees console timing instead of however fast the
// host can present. Pacing is on by default; boot option "present_pacing": false (browser) or
// --no-present-pacing (native harness) turns it off for benchmarks.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>

namespace playgta6::gpu {

class PresentPacer {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr int64_t kVblankNs = 16'666'667;  // 60 Hz

    PresentPacer();

    // The vblank index at `t` (0 at the clock's origin) and the start of vblank `v`.
    uint64_t vblank_at(Clock::time_point t) const;
    Clock::time_point vblank_start(uint64_t v) const;

    // A present with sync interval `interval` (vblanks). Waits as described above (not while
    // pacing is off) and returns the vblank the present flips at.
    uint64_t present(uint32_t interval);

    struct Stats {
        uint64_t presents = 0;        // paced presents (pacing on)
        uint64_t immediate = 0;       // ... with interval 0
        uint64_t waits = 0;           // ... that waited for a vblank
        uint64_t wait_us = 0;         // total time waited
        uint64_t missed_vblanks = 0;  // vblanks presents came late by
        uint32_t last_interval = 0;
    };
    Stats stats() const;

    // The title drives the frame interval (SetFrameIntervalX): presents are not quantized to the
    // global 60 Hz vblank grid (zero phase beat when the title paces itself at <= 60 Hz), but
    // interval > 0 presents still enforce a 1-vblank (60 Hz) minimum gap from the previous present
    // so unpaced/catch-up bursts cannot flood asynchronous WebGPU queues (G74).
    void set_title_driven(bool on);
    bool title_driven() const;

    // Title-driven minimum gap between interval > 0 presents. false ("min1", default): one vblank
    // whatever the interval (menus the title paces at interval 2 run at 60 fps, faster than the
    // console). true ("honor", --present-interval honor): interval vblanks, so interval 2 (SetFrameIntervalX
    // 33333 us, period 3) gives the console's 30 fps.
    void set_honor_interval(bool on);
    bool honor_interval() const;
    // The gap in vblanks title-driven mode enforces for `interval` (> 0) in each mode.
    static uint32_t title_gap_vblanks(uint32_t interval, bool honor) {
        return honor ? (interval < 1 ? 1u : interval) : 1u;
    }

    void set_enabled(bool on);
    bool enabled() const;

    // Called at the start of every present, paced or not (the boot report's first_present and
    // FIRST_PRESENT milestone, guest/boot.h mark_first_present); nullptr: none.
    using Observer = void (*)();
    void set_observer(Observer observer);

    // Tests: the clock, its origin and the wait (default: steady_clock and sleep_until).
    void set_clock_for_testing(std::function<Clock::time_point()> now,
                               std::function<void(Clock::time_point)> sleep_until,
                               Clock::time_point origin);
    void reset_for_testing();

private:
    mutable std::mutex mutex_;
    Clock::time_point origin_;
    std::function<Clock::time_point()> now_;
    std::function<void(Clock::time_point)> sleep_until_;
    std::atomic<Observer> observer_{nullptr};
    bool enabled_ = true;
    bool title_driven_ = false;
    bool honor_interval_ = false;
    bool have_last_ = false;
    bool have_last_present_time_ = false;
    uint64_t last_flip_ = 0;
    Clock::time_point last_present_time_{};
    Stats stats_{};
};

// The process's display (shared by every queue and device).
PresentPacer& present_pacer();

}  // namespace playgta6::gpu
