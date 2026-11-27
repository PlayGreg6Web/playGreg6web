#include "gpu/d3d12_readback.h"
#include "os/thread_name.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "gpu/d3d12_guest_objects.h"
#include "os/guest_window.h"
#include "os/os_shim.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

size_t clamp_ring_capacity(size_t requested) {
    if (requested == 0) {
        return ReadbackGate::kDefaultRingCapacity;
    }
    return std::clamp(requested, ReadbackGate::kMinRingCapacity, ReadbackGate::kMaxRingCapacity);
}

size_t kind_index(ReadbackKind kind) {
    const size_t idx = static_cast<size_t>(kind);
    return idx < ReadbackGate::kKindCount ? idx : 0u;
}

}  // namespace

ReadbackGate::ReadbackGate(bool hold, std::chrono::milliseconds timeout, size_t ring_capacity)
    : timeout_(timeout.count() < 0 ? std::chrono::milliseconds(0) : timeout),
      ring_capacity_(clamp_ring_capacity(ring_capacity)),
      hold_(hold),
      ring_(ring_capacity_) {}

ReadbackGate::~ReadbackGate() {
    // Move parked callbacks out before unlocking so any destructor invoked by dropping a parked
    // closure cannot deadlock if it re-enters stats() / poll() / issue().
    std::deque<Parked> dropped;
    std::thread poller_to_join;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        dropped.swap(parked_);
        if (poller_.joinable() && poller_.get_id() != std::this_thread::get_id()) {
            poller_to_join = std::move(poller_);
        }
    }
    cv_.notify_all();
    if (poller_to_join.joinable()) {
        poller_to_join.join();
    }
    dropped.clear();
}

ReadbackSlot ReadbackGate::issue_slot_impl(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag,
                                           std::function<void()> on_complete) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Advance completion tracking first so already-completed ring slots are not miscounted as
    // overflows when reusing the ring.
    const uint64_t done = completed_locked();
    const uint64_t seq = ++issued_;
    bytes_ += bytes;
    const size_t kidx = kind_index(kind);
    ++issued_by_kind_[kidx];
    bytes_by_kind_[kidx] += bytes;

    const uint64_t in_flight = seq > done ? (seq - done) : 0u;
    if (in_flight > pending_high_water_) {
        pending_high_water_ = in_flight;
    }

    const uint32_t ridx = static_cast<uint32_t>(seq % ring_capacity_);
    ReadbackSlot& slot = ring_[ridx];
    if (slot.seq != 0 && !slot.completed && slot.seq > done) {
        ++ring_overflows_;
    }
    slot.seq = seq;
    slot.guest_va = guest_va;
    slot.bytes = bytes;
    slot.kind = kind;
    slot.tag = tag;
    slot.ring_index = ridx;
    slot.completed = (done >= seq);

    ranges_.push_back(PendingRange{seq, guest_va, bytes, kind, tag, std::move(on_complete)});
    count_work(Work::ReadbacksIssued);
    if (bytes != 0) {
        count_work(Work::ReadbackBytes, bytes);
    }
    // If consumer already ran ahead of seq (rare synthetic case), reconcile immediately.
    if (done >= seq) {
        completed_locked();
    }
    return slot;
}

uint32_t ReadbackGate::issue(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag) {
    return static_cast<uint32_t>(issue_slot_impl(guest_va, bytes, kind, tag).seq);
}

uint32_t ReadbackGate::issue_with_completion(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag,
                                             std::function<void()> on_complete) {
    return static_cast<uint32_t>(issue_slot_impl(guest_va, bytes, kind, tag, std::move(on_complete)).seq);
}

uint64_t ReadbackGate::issue64(uint64_t guest_va, uint64_t bytes, ReadbackKind kind, uint32_t tag) {
    return issue_slot_impl(guest_va, bytes, kind, tag).seq;
}

ReadbackSlot ReadbackGate::issue_occlusion_query(uint64_t guest_va, uint64_t bytes, uint32_t query_index) {
    return issue_slot_impl(guest_va, bytes, ReadbackKind::OcclusionQuery, query_index);
}

ReadbackSlot ReadbackGate::issue_timestamp_query(uint64_t guest_va, uint64_t bytes, uint32_t pass_id) {
    return issue_slot_impl(guest_va, bytes, ReadbackKind::TimestampQuery, pass_id);
}

ReadbackSlot ReadbackGate::issue_compute_readback(uint64_t guest_va, uint64_t bytes, uint32_t tag) {
    return issue_slot_impl(guest_va, bytes, ReadbackKind::ComputeReadback, tag);
}

uint64_t ReadbackGate::peek_completed_locked() const {
    const uint32_t v = done_.load(std::memory_order_acquire);
    const uint64_t wraps = done_wraps_ + (v < done_last_ ? 1ull : 0ull);
    const uint64_t raw = (wraps << 32) | static_cast<uint64_t>(v);
    return std::max(raw, reported_seen_);
}

uint64_t ReadbackGate::completed_locked() {
    const uint32_t v = done_.load(std::memory_order_acquire);
    if (v < done_last_) {
        ++done_wraps_;
    }
    done_last_ = v;
    const uint64_t reported = std::max((done_wraps_ << 32) | static_cast<uint64_t>(v), reported_seen_);
    if (reported > reported_seen_) {
        count_work(Work::ReadbacksCompleted, reported - reported_seen_);
        reported_seen_ = reported;
    }
    const uint64_t done = reported;
    // The consumer wrote these ranges: lifted code tracking window writes must see them (O37).
    while (!ranges_.empty() && ranges_.front().seq <= done) {
        PendingRange r = std::move(ranges_.front());
        ranges_.pop_front();
        if (r.on_complete) {
            r.on_complete();
        }
        if (r.va != 0 && r.bytes != 0) {
            const size_t clamped = static_cast<size_t>(
                std::min<uint64_t>(r.bytes, static_cast<uint64_t>(std::numeric_limits<size_t>::max())));
            os::guest_window::mark_written(r.va, clamped);
        }
        ++completed_by_kind_[kind_index(r.kind)];
        const uint32_t ridx = static_cast<uint32_t>(r.seq % ring_capacity_);
        ReadbackSlot& slot = ring_[ridx];
        if (slot.seq == r.seq) {
            slot.completed = true;
        }
        ReadbackSlot finished{};
        finished.seq = r.seq;
        finished.guest_va = r.va;
        finished.bytes = r.bytes;
        finished.kind = r.kind;
        finished.tag = r.tag;
        finished.ring_index = ridx;
        finished.completed = true;
        if (completed_ring_.size() >= ring_capacity_) {
            completed_ring_.pop_front();
        }
        completed_ring_.push_back(finished);
        seen_ = std::max(seen_, r.seq);
    }
    seen_ = std::max(seen_, done);
    return done;
}

bool ReadbackGate::is_complete(uint64_t seq) const {
    if (seq == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return peek_completed_locked() >= seq;
}

bool ReadbackGate::try_poll_slot(uint64_t seq, ReadbackSlot* out) {
    if (seq == 0) {
        if (out != nullptr) {
            *out = ReadbackSlot{};
            out->completed = true;
        }
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t done = completed_locked();
    const bool ready = (seq <= issued_) && (done >= seq);
    if (out != nullptr) {
        const uint32_t ridx = static_cast<uint32_t>(seq % ring_capacity_);
        const ReadbackSlot& slot = ring_[ridx];
        if (slot.seq == seq) {
            *out = slot;
            out->completed = ready;
        } else {
            ReadbackSlot summary{};
            summary.seq = seq;
            summary.ring_index = ridx;
            summary.completed = ready;
            *out = summary;
        }
    }
    return ready;
}

size_t ReadbackGate::drain_completed_slots(std::vector<ReadbackSlot>* out, size_t max_slots) {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_locked();
    const size_t available = completed_ring_.size();
    const size_t take = (max_slots == 0 || max_slots > available) ? available : max_slots;
    if (out != nullptr) {
        out->reserve(out->size() + take);
        for (size_t i = 0; i < take; ++i) {
            out->push_back(completed_ring_.front());
            completed_ring_.pop_front();
        }
    } else {
        for (size_t i = 0; i < take; ++i) {
            completed_ring_.pop_front();
        }
    }
    return take;
}

uint64_t ReadbackGate::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t done = peek_completed_locked();
    return issued_ > done ? (issued_ - done) : 0u;
}

uint64_t ReadbackGate::completed_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peek_completed_locked();
}

void ReadbackGate::notify() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++wake_gen_;
    }
    cv_.notify_one();
}

bool ReadbackGate::park(std::function<void()> release) {
    poll();  // signals already due go first (keeps the order)
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hold_ || stop_) {
        return false;
    }
    if (parked_.empty() && completed_locked() >= issued_) {
        return false;
    }
    parked_.push_back({issued_, std::chrono::steady_clock::now(), std::move(release)});
    ++signals_held_;
    ++wake_gen_;
    count_work(Work::ReadbackSignalsHeld);
    start_poller_locked();
    cv_.notify_all();
    return true;
}

size_t ReadbackGate::poll() {
    std::vector<std::function<void()>> ready;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t done = completed_locked();
        const auto now = std::chrono::steady_clock::now();
        while (!parked_.empty()) {
            Parked& p = parked_.front();
            if (done < p.needed) {
                // Never publish a completion before the bytes arrived: the signal stays held. The
                // timeout is a diagnostic only (once per held signal, logged loudly).
                if (!p.warned && now - p.at >= timeout_) {
                    p.warned = true;
                    ++timeouts_;
                    count_work(Work::ReadbackTimeouts);
                    os::log("d3d12_xs: WARNING GPU read-back not completed after " + std::to_string(timeout_.count()) +
                            " ms; the guarding fence stays unsignalled (consumer stalled or present-bound?)");
                }
                break;
            }
            ready.push_back(std::move(p.release));
            parked_.pop_front();
        }
    }
    for (auto& r : ready) {
        if (r) {
            r();
        }
    }
    return ready.size();
}

void ReadbackGate::start_poller_locked() {
    if (poller_.joinable()) {
        return;
    }
    // Titles poll the fence's completed-value word directly, so releases cannot wait for an API call.
    // Use adaptive backoff so active completions wake quickly (~250 us) while idle or stalled
    // readbacks never busy-spin or waste CPU.
    poller_ = std::thread([this] {
        playgta6::os::name_this_thread(L"pg6-readback-poller");
        constexpr int64_t kMinPollUs = 250;
        constexpr int64_t kMaxPollUs = 8000;
        int64_t poll_us = kMinPollUs;
        uint64_t last_gen = 0;
        uint64_t last_done = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_) {
            if (parked_.empty()) {
                poll_us = kMinPollUs;
                cv_.wait(lock, [this] { return stop_ || !parked_.empty(); });
                last_gen = wake_gen_;
                continue;
            }
            if (wake_gen_ != last_gen) {
                poll_us = kMinPollUs;
                last_gen = wake_gen_;
            }
            // Cap sleep to the next unwarned signal's timeout deadline so diagnostic timeouts fire
            // promptly even while backed off.
            auto wait_dur = std::chrono::microseconds(poll_us);
            const Parked& front = parked_.front();
            if (!front.warned) {
                const auto now = std::chrono::steady_clock::now();
                const auto deadline = front.at + timeout_;
                if (deadline <= now) {
                    wait_dur = std::chrono::microseconds(0);
                } else {
                    const auto rem = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
                    if (rem < wait_dur) {
                        wait_dur = rem;
                    }
                }
            }
            if (wait_dur.count() > 0) {
                cv_.wait_for(lock, wait_dur, [this, last_gen] { return stop_ || wake_gen_ != last_gen; });
            }
            if (stop_) {
                break;
            }
            const uint64_t prev_done = last_done;
            lock.unlock();
            const size_t released = poll();
            lock.lock();
            const uint64_t cur_done = reported_seen_;
            last_done = cur_done;
            if (released > 0 || cur_done > prev_done || wake_gen_ != last_gen) {
                poll_us = kMinPollUs;
                last_gen = wake_gen_;
            } else {
                poll_us = std::min<int64_t>(poll_us * 2, kMaxPollUs);
            }
        }
    });
}

ReadbackGate::Stats ReadbackGate::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t done = peek_completed_locked();
    Stats s;
    s.issued = issued_;
    s.bytes = bytes_;
    s.completed = done;
    s.timeouts = timeouts_;
    s.signals_held = signals_held_;
    s.signals_pending = parked_.size();
    s.holding = hold_;
    s.pending = issued_ > done ? (issued_ - done) : 0u;
    s.pending_high_water = pending_high_water_;
    s.ring_capacity = ring_capacity_;
    s.ring_overflows = ring_overflows_;
    s.issued_by_kind = issued_by_kind_;
    s.completed_by_kind = completed_by_kind_;
    s.bytes_by_kind = bytes_by_kind_;
    return s;
}

bool ReadbackGate::seed_sequence_for_testing(uint64_t base_seq) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!parked_.empty() || !ranges_.empty()) {
        return false;
    }
    const uint32_t low = static_cast<uint32_t>(base_seq & 0xFFFFFFFFull);
    done_.store(low, std::memory_order_release);
    done_last_ = low;
    done_wraps_ = base_seq >> 32;
    issued_ = base_seq;
    seen_ = base_seq;
    reported_seen_ = base_seq;
    completed_ring_.clear();
    for (ReadbackSlot& slot : ring_) {
        slot = ReadbackSlot{};
    }
    return true;
}

}  // namespace playgta6::gpu::d3d12_guest
