// One owner for the fault/stop report. Several threads can trap or fault at the same moment
// (e.g. every guest worker hitting the same missing code); only the first one may build the
// report and write the dump, the rest park until the owner ends the process (the exit backstop
// covers an owner that hangs). Before this, the losers returned into the default handler, which
// called std::abort(): the first SIGABRT handler call reset the CRT's handler to default, so the
// next aborting thread fast-failed the process with 0xC0000409 and no report, or a loser killed
// the process while the owner was still writing the report.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <thread>

namespace playgta6::guest {

class FaultGate {
public:
    static uint64_t self() { return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())) | 1u; }

    void reset() { owner_.store(0, std::memory_order_release); }
    bool owned_by_me() const { return owner_.load(std::memory_order_acquire) == self(); }
    bool claimed() const { return owner_.load(std::memory_order_acquire) != 0; }

    // True when the caller owns the report (first caller, or the owner re-entering after a fault
    // inside its own report path). Any other thread never returns.
    bool claim_or_park() {
        uint64_t expected = 0;
        if (owner_.compare_exchange_strong(expected, self(), std::memory_order_acq_rel)) return true;
        if (expected == self()) return true;
        park_forever();
    }

    // Non-blocking: true when the caller just became the owner.
    bool try_claim() {
        uint64_t expected = 0;
        return owner_.compare_exchange_strong(expected, self(), std::memory_order_acq_rel);
    }

    [[noreturn]] static void park_forever() {
        for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
    }

private:
    std::atomic<uint64_t> owner_{0};
};

// Process-wide gate in front of std::abort(): the CRT resets the SIGABRT handler to default while
// it runs it, so a second thread aborting at the same moment would skip the report hook and exit
// (or fast-fail, 0xC0000409) with no report. The first aborting thread goes through; the others
// park until it ends the process.
inline FaultGate& abort_gate() {
    static FaultGate gate;
    return gate;
}

[[noreturn]] inline void abort_once() {
    abort_gate().claim_or_park();
    std::abort();
}

}  // namespace playgta6::guest
