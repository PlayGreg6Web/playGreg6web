#pragma once

// Per-thread time spans (the boot report's `frame_breakdown`): where a guest thread's wall time
// goes between two presents. Each span kind accumulates the time the calling thread spent inside
// a ThreadSpan scope of that kind (blocked in a guest wait, a contended lock, a sleep, the frame
// event / present pacer, or our D3D12 layer translating a submission). Scopes nest: an inner scope
// pauses the outer one, so every nanosecond is charged to exactly one kind (self time). The CPU
// kinds (our D3D12 layer) also record the thread's CPU time inside them, so their off-CPU part
// (blocking no span names) shows as wall minus CPU.
//
// Thread-local and lock-free: a scope costs two steady_clock reads (plus two thread CPU reads for
// a CPU kind). Read the totals of the calling thread with this_thread_spans().

#include <array>
#include <chrono>
#include <cstdint>

#include "os/thread_name.h"

namespace playgta6::os {

enum class SpanKind : int {
    Wait = 0,       // WaitForSingleObject / WaitForMultipleObjects with a non-zero timeout
    CondVar,        // SleepConditionVariableSRW / CS
    Lock,           // a contended EnterCriticalSection / AcquireSRWLock*
    Sleep,          // Sleep
    FrameEvent,     // WaitFrameEventX (blocking)
    Pacer,          // the present pacer's vblank wait
    SubmitBlocked,  // the live executor's queue was full (backpressure in LivePump::submit)
    D3dLock,        // a contended lock of our D3D12 layer (submission / CPU-buffer flush)
    D3dExecute,     // ExecuteCommandLists outside the parts below
    D3dFlush,       // ExecuteCommandLists: CPU-visible buffer / texture flush (write tracking, shadow diff)
    D3dAppend,      // ExecuteCommandLists: the lists' deferred copies and texture syncs
    D3dSubmit,      // ExecuteCommandLists: handing the packets to the sink (the executor queue)
    D3dPresent,     // PresentX outside the pacer wait
    Count
};

inline constexpr int kSpanKinds = static_cast<int>(SpanKind::Count);

// CPU work of our D3D12 layer; the other kinds are the thread blocked.
inline constexpr bool span_kind_is_cpu(SpanKind k) {
    return k == SpanKind::D3dExecute || k == SpanKind::D3dFlush || k == SpanKind::D3dAppend ||
           k == SpanKind::D3dSubmit || k == SpanKind::D3dPresent;
}

inline const char* span_kind_name(SpanKind k) {
    static const char* const names[kSpanKinds] = {
        "wait",       "condvar",   "lock",       "sleep",      "frame_event", "pacer",      "submit_blocked",
        "d3d_lock",   "d3d_execute", "d3d_flush", "d3d_append", "d3d_submit", "d3d_present"};
    return names[static_cast<int>(k)];
}

struct ThreadSpanTotals {
    std::array<uint64_t, kSpanKinds> ns{};
    std::array<uint64_t, kSpanKinds> cpu_ns{};  // CPU kinds only
};

namespace thread_spans_detail {
struct State {
    ThreadSpanTotals totals;
    int active = -1;  // the innermost open scope's kind, -1 = none
    std::chrono::steady_clock::time_point since{};
    uint64_t cpu_since = 0;  // thread CPU at `since` (CPU kinds)
};
inline State& state() {
    thread_local State s;
    return s;
}
inline uint64_t ns_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return b > a ? static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()) : 0u;
}
inline bool active_is_cpu(const State& s) { return s.active >= 0 && span_kind_is_cpu(static_cast<SpanKind>(s.active)); }
// Charges the active scope up to now (wall, and CPU when `cpu_now` was read).
inline void charge(State& s, std::chrono::steady_clock::time_point now, uint64_t cpu_now) {
    if (s.active < 0) return;
    s.totals.ns[static_cast<size_t>(s.active)] += ns_between(s.since, now);
    if (active_is_cpu(s) && cpu_now > s.cpu_since) s.totals.cpu_ns[static_cast<size_t>(s.active)] += cpu_now - s.cpu_since;
}
}  // namespace thread_spans_detail

// The calling thread's totals, including the elapsed part of a scope that is still open.
inline ThreadSpanTotals this_thread_spans() {
    namespace d = thread_spans_detail;
    d::State s = d::state();
    d::charge(s, std::chrono::steady_clock::now(), d::active_is_cpu(s) ? this_thread_cpu_ns() : 0);
    return s.totals;
}

class ThreadSpan {
public:
    // SpanKind::Count makes an inactive scope (no timing, no effect on an outer scope).
    explicit ThreadSpan(SpanKind kind) {
        if (kind == SpanKind::Count) {
            inactive_ = true;
            return;
        }
        namespace d = thread_spans_detail;
        d::State& s = d::state();
        const auto now = std::chrono::steady_clock::now();
        const uint64_t cpu = d::active_is_cpu(s) || span_kind_is_cpu(kind) ? this_thread_cpu_ns() : 0;
        d::charge(s, now, cpu);
        outer_ = s.active;
        s.active = static_cast<int>(kind);
        s.since = now;
        s.cpu_since = cpu;
    }
    ~ThreadSpan() {
        if (inactive_) {
            return;
        }
        namespace d = thread_spans_detail;
        d::State& s = d::state();
        const auto now = std::chrono::steady_clock::now();
        const bool outer_cpu = outer_ >= 0 && span_kind_is_cpu(static_cast<SpanKind>(outer_));
        const uint64_t cpu = d::active_is_cpu(s) || outer_cpu ? this_thread_cpu_ns() : 0;
        d::charge(s, now, cpu);
        s.active = outer_;
        s.since = now;
        s.cpu_since = cpu;
    }
    ThreadSpan(const ThreadSpan&) = delete;
    ThreadSpan& operator=(const ThreadSpan&) = delete;

private:
    int outer_ = -1;
    bool inactive_ = false;
};

// Locks `m`, charging the wait to `kind` only when it is contended.
template <class Mutex>
inline void lock_with_span(Mutex& m, SpanKind kind) {
    if (m.try_lock()) return;
    const ThreadSpan span(kind);
    m.lock();
}

}  // namespace playgta6::os
