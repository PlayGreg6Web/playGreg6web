#pragma once

// GPU -> CPU read-back ordering and non-blocking readback ring (G66, G84).
//
// Copies into CPU-visible memory (readback-heap buffers, texture regions copied into a placed
// footprint), occlusion queries, GPU pass timestamps, and compute readbacks (auto-exposure / GPU
// culling) become ReadbackBuffer / ReadbackTexture packets (playgta6_abi.h). The consumer runs
// them in stream order, writes the bytes into guest memory and then adds 1 to a u32 completion
// counter this gate owns. Fences are signalled on the CPU at submission (CommandQueue::Signal), so
// without a hold the title would see a fence complete before its read-back data arrived.
//
// The gate counts the read-backs it issued and tracks bounded ring metadata per readback kind so
// callers can poll completion non-blockingly (`is_complete`, `try_poll_slot`,
// `drain_completed_slots`) without stalling the main thread. A queue signal issued while some
// readbacks are still outstanding (or while earlier signals are held) is parked with the issue
// count at that moment and released, in order, once the completion counter reaches it. Release
// calls the signal itself (Fence::signal_from_queue), so the fence publish rules
// (DeviceOptions::fence_publish_lag) apply unchanged afterwards: this is an extra hold in front of
// them, not a change to them.
//
// A held signal is only ever released by the consumer's completion: nothing is published before the
// bytes landed. `timeout` is a diagnostic: a signal held longer logs a loud warning once and is
// counted (ReadbackTimeouts), but stays held. Builds without a consumer construct the gate with
// hold = false, so they never park.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace playgta6::gpu::d3d12_guest {

// Categorises GPU -> CPU readbacks tracked in the non-blocking readback ring.
enum class ReadbackKind : uint8_t {
    Buffer = 0,
    Texture = 1,
    OcclusionQuery = 2,
    TimestampQuery = 3,
    ComputeReadback = 4,
    Count = 5,
};

// Snapshot of a single readback ring slot returned by non-blocking query / drain helpers.
struct ReadbackSlot {
    uint64_t seq = 0;           // 1-based monotonic 64-bit sequence number
    uint64_t guest_va = 0;      // guest virtual address written by consumer
    uint64_t bytes = 0;         // byte count written by consumer (0 allowed)
    ReadbackKind kind = ReadbackKind::Buffer;
    uint32_t tag = 0;           // caller-defined tag (e.g. query index, pass id, dispatch id)
    uint32_t ring_index = 0;    // seq % ring_capacity
    bool completed = false;
};

class ReadbackGate {
public:
    static constexpr size_t kDefaultRingCapacity = 256;
    static constexpr size_t kMinRingCapacity = 16;
    static constexpr size_t kMaxRingCapacity = 16384;
    static constexpr size_t kKindCount = static_cast<size_t>(ReadbackKind::Count);

    struct Stats {
        uint64_t issued = 0;
        uint64_t bytes = 0;
        uint64_t completed = 0;       // monotonic 64-bit counter value seen (completions the consumer reported)
        uint64_t timeouts = 0;        // held signals that passed the diagnostic timeout
        uint64_t signals_held = 0;    // queue signals parked behind a read-back
        uint64_t signals_pending = 0; // parked now
        bool holding = false;         // signals are held (enabled and not given up)
        // Non-blocking readback ring telemetry (G84).
        uint64_t pending = 0;              // in-flight readbacks (issued - completed, clamped >= 0)
        uint64_t pending_high_water = 0;   // peak concurrent in-flight readbacks observed
        uint64_t ring_capacity = 0;        // bounded metadata ring capacity
        uint64_t ring_overflows = 0;       // ring slot overwritten while its prior entry was still pending
        std::array<uint64_t, kKindCount> issued_by_kind{};
        std::array<uint64_t, kKindCount> completed_by_kind{};
        std::array<uint64_t, kKindCount> bytes_by_kind{};
    };

    // `hold`: park queue signals behind outstanding read-backs (a consumer answers).
    ReadbackGate(bool hold, std::chrono::milliseconds timeout, size_t ring_capacity = kDefaultRingCapacity);
    ~ReadbackGate();
    ReadbackGate(const ReadbackGate&) = delete;
    ReadbackGate& operator=(const ReadbackGate&) = delete;

    // Host address of the u32 completion counter (the packets' done_addr).
    uint64_t done_address() const { return reinterpret_cast<uintptr_t>(&done_); }

    // Issues one read-back that will write `bytes` at guest address `guest_va`; returns its 1-based
    // 32-bit sequence number (for packet headers). The range is marked written (guest_window) when
    // it completes. Backward-compatible with 2-argument callers.
    uint32_t issue(uint64_t guest_va, uint64_t bytes, ReadbackKind kind = ReadbackKind::Buffer, uint32_t tag = 0);

    // Issues one typed read-back and returns its full 64-bit monotonic sequence number (wrap-safe).
    uint64_t issue64(uint64_t guest_va, uint64_t bytes, ReadbackKind kind = ReadbackKind::Buffer, uint32_t tag = 0);

    // Like issue(), plus `on_complete`, run once when the consumer has reported this read-back done
    // (delivered or failed: the callback must check what it was given), in sequence order, on
    // whichever thread observed the completion. It runs under the gate's lock: keep it short and
    // never call back into the gate. `guest_va` / `bytes` may be 0 when the bytes land in a scratch
    // buffer the callback owns (no guest range is then marked written).
    uint32_t issue_with_completion(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag,
                                   std::function<void()> on_complete);

    // Convenience typed issuers for non-blocking occlusion queries, GPU pass timestamps, and
    // compute readbacks (auto-exposure / GPU culling).
    ReadbackSlot issue_occlusion_query(uint64_t guest_va, uint64_t bytes = 8, uint32_t query_index = 0);
    ReadbackSlot issue_timestamp_query(uint64_t guest_va, uint64_t bytes = 8, uint32_t pass_id = 0);
    ReadbackSlot issue_compute_readback(uint64_t guest_va, uint64_t bytes, uint32_t tag = 0);

    // Non-blocking completion check for a 64-bit sequence number (0 is trivially complete).
    // Never blocks or stalls the caller.
    bool is_complete(uint64_t seq) const;

    // Non-blocking poll for a specific readback slot: advances completion tracking, marks newly
    // completed guest-window ranges written, and returns true if `seq` has completed. If `out` is
    // non-null, fills it with the slot metadata (when still resident in the ring, or a summary
    // with `seq` and `completed` if overwritten).
    bool try_poll_slot(uint64_t seq, ReadbackSlot* out = nullptr);

    // Non-blocking drain of newly completed slots in FIFO order (up to `max_slots`, 0 = all
    // available). Advances completion tracking first and never blocks.
    size_t drain_completed_slots(std::vector<ReadbackSlot>* out = nullptr, size_t max_slots = 0);

    // Non-blocking counts of currently pending and total completed readbacks (64-bit wrap-safe).
    uint64_t pending_count() const;
    uint64_t completed_count() const;
    size_t ring_capacity() const { return ring_capacity_; }

    // Optional consumer hint: wakes the background poller immediately with minimum latency.
    void notify();

    // A queue signal. If read-backs are outstanding or signals are already held, `release` is
    // parked (run later by poll / the poller thread, in order) and true is returned; otherwise false
    // and the caller signals now.
    bool park(std::function<void()> release);

    // Releases the parked signals whose read-backs completed (or logs diagnostic timeouts); returns
    // how many signals were released. Never blocks.
    size_t poll();

    Stats stats() const;

    // Test helper: seeds the internal 64-bit sequence and 32-bit atomic counter (only when no
    // readbacks or signals are pending) so 32-bit wraparound can be exercised deterministically.
    bool seed_sequence_for_testing(uint64_t base_seq);

private:
    struct Parked {
        uint64_t needed;  // completion count that releases it
        std::chrono::steady_clock::time_point at;
        std::function<void()> release;
        bool warned = false;
    };
    struct PendingRange {
        uint64_t seq = 0;
        uint64_t va = 0;
        uint64_t bytes = 0;
        ReadbackKind kind = ReadbackKind::Buffer;
        uint32_t tag = 0;
        std::function<void()> on_complete;
    };
    uint64_t peek_completed_locked() const;
    uint64_t completed_locked();
    void start_poller_locked();
    ReadbackSlot issue_slot_impl(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag,
                                 std::function<void()> on_complete = {});

    const std::chrono::milliseconds timeout_;
    const size_t ring_capacity_;
    alignas(8) std::atomic<uint32_t> done_{0};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool hold_;
    bool stop_ = false;
    uint64_t wake_gen_ = 0;
    uint64_t issued_ = 0;
    uint64_t bytes_ = 0;
    uint64_t seen_ = 0;           // completions whose ranges were marked written
    uint64_t reported_seen_ = 0;  // consumer completions already counted
    uint64_t done_wraps_ = 0;     // high 32 bits of the u32 counter
    uint32_t done_last_ = 0;
    uint64_t timeouts_ = 0;
    uint64_t signals_held_ = 0;
    uint64_t pending_high_water_ = 0;
    uint64_t ring_overflows_ = 0;
    std::array<uint64_t, kKindCount> issued_by_kind_{};
    std::array<uint64_t, kKindCount> completed_by_kind_{};
    std::array<uint64_t, kKindCount> bytes_by_kind_{};
    std::vector<ReadbackSlot> ring_;          // fixed-size metadata ring indexed by seq % ring_capacity_
    std::deque<PendingRange> ranges_;         // issued, not yet marked written
    std::deque<ReadbackSlot> completed_ring_; // bounded FIFO of newly completed slots for drain_completed_slots
    std::deque<Parked> parked_;
    std::thread poller_;
};

}  // namespace playgta6::gpu::d3d12_guest
