// Live play: hands the GPU packet stream (os::wgpu_submit batches) to a native executor running on
// its own thread. Header-only and executor-agnostic (a template over the executor type) so the
// queue / threading logic is unit-tested with a fake executor, without Dawn or a GPU.
//
// Threads: the submit side (the guest's GPU thread) copies each batch into a bounded queue; the
// pump thread creates the executor (executors and their windows are thread-affine), executes the
// queued batches in order, and polls it (window messages, WebGPU callbacks) between batches and
// at least every idle_poll_ms while the queue is empty.
//
// Backpressure, like the browser worker: when the executor falls behind and the queue holds
// max_queued_batches batches or max_queued_bytes bytes, submit() blocks the submitting thread
// until room is made (never drops). Each blocked submit is counted with its wait time. After the
// window was closed (or stop()), submit() returns immediately and discards, so the guest never
// deadlocks behind a dead consumer.
//
// Executor requirements: uint32_t execute(const uint8_t*, size_t); bool poll() (false = window
// closed); void wait_idle(). Optional: bool window_focused() const, forwarded to on_focus on change.
#pragma once

#include "os/thread_name.h"
#include "os/thread_spans.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace playgta6::gpu::native_exec {

struct LivePumpOptions {
    size_t max_queued_batches = 64;
    size_t max_queued_bytes = size_t{256} << 20;  // a batch larger than this is admitted into an empty queue
    uint32_t idle_poll_ms = 8;                     // poll cadence while no batch is queued
};

struct LivePumpStats {
    uint64_t submitted = 0;            // batches accepted by submit()
    uint64_t submitted_bytes = 0;
    uint64_t executed = 0;             // batches handed to the executor
    uint64_t presents = 0;             // Present packets the executor reported
    uint64_t blocked_submits = 0;      // submits that had to wait for room (backpressure)
    uint64_t blocked_ns = 0;           // total wait of those submits
    uint64_t execute_ns = 0;           // time the pump thread spent inside the executor's execute()
    uint64_t dropped_after_close = 0;  // batches discarded after close / stop
    uint64_t discarded_on_close = 0;   // queued batches dropped when the window closed
    uint64_t max_queue_depth = 0;
    uint64_t polls = 0;
    bool window_closed = false;
};

namespace live_pump_detail {
template <class E, class = void>
struct HasFocus : std::false_type {};
template <class E>
struct HasFocus<E, std::void_t<decltype(std::declval<const E&>().window_focused())>> : std::true_type {};
}  // namespace live_pump_detail

template <class Executor>
class LivePump {
public:
    using Factory = std::function<std::unique_ptr<Executor>(std::string* error)>;

    explicit LivePump(LivePumpOptions options = {}) : options_(options) {}
    ~LivePump() { stop(); }
    LivePump(const LivePump&) = delete;
    LivePump& operator=(const LivePump&) = delete;

    // Starts the pump thread and waits until `factory` has run on it. False (with *error) when the
    // executor could not be created; the pump is then not running.
    // on_close runs once on the pump thread when the executor reports its window closed.
    // on_focus runs on the pump thread whenever the window's focus changes (and once at start).
    bool start(Factory factory, std::function<void()> on_close, std::function<void(bool)> on_focus,
               std::string* error) {
        on_close_ = std::move(on_close);
        on_focus_ = std::move(on_focus);
        thread_ = std::thread([this, f = std::move(factory)]() mutable { run(std::move(f)); });
        std::unique_lock<std::mutex> lock(mutex_);
        ready_cv_.wait(lock, [&] { return ready_; });
        if (!create_error_.empty()) {
            if (error) *error = create_error_;
            lock.unlock();
            thread_.join();
            return false;
        }
        return true;
    }

    // Copies the batch into the queue; blocks while the queue is full. Thread-safe.
    void submit(const uint8_t* data, size_t len) {
        if (len == 0) return;
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_ || stopping_) {
            ++stats_.dropped_after_close;
            return;
        }
        if (full_locked(len)) {
            ++stats_.blocked_submits;
            const os::ThreadSpan frame_span(os::SpanKind::SubmitBlocked);  // the submitting thread's frame_breakdown
            const auto t0 = std::chrono::steady_clock::now();
            room_cv_.wait(lock, [&] { return !full_locked(len) || closed_ || stopping_; });
            stats_.blocked_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            if (closed_ || stopping_) {
                ++stats_.dropped_after_close;
                return;
            }
        }
        queue_.emplace_back(data, data + len);
        queued_bytes_ += len;
        ++stats_.submitted;
        stats_.submitted_bytes += len;
        stats_.max_queue_depth = std::max<uint64_t>(stats_.max_queue_depth, queue_.size());
        work_cv_.notify_one();
    }

    // Blocks until every accepted batch was executed (or the window closed).
    void flush() {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_cv_.wait(lock, [&] { return (queue_.empty() && !busy_) || closed_ || !running_; });
    }

    // Executes what is queued, waits for the GPU, destroys the executor and joins the thread.
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!thread_.joinable()) return;
            stopping_ = true;
        }
        work_cv_.notify_all();
        room_cv_.notify_all();
        thread_.join();
    }

    // Ends the run now: queued batches are discarded, a submit() blocked on backpressure returns
    // (and later ones discard), and the pump thread exits after the batch it is executing. For a
    // wall-clock limit that must hold even while the guest is blocked behind a slow executor.
    void abort() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.discarded_on_close += queue_.size();
            queue_.clear();
            queued_bytes_ = 0;
            stopping_ = true;
        }
        work_cv_.notify_all();
        room_cv_.notify_all();
        idle_cv_.notify_all();
    }

    bool window_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    // True when nothing is queued and the executor is between batches: background work (pipeline
    // prewarm) submitted now does not land in front of a batch the guest already handed over.
    bool idle() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty() && !busy_;
    }
    LivePumpStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        LivePumpStats s = stats_;
        s.window_closed = closed_;
        return s;
    }

    // Runs on the pump thread after the last batch and wait_idle(), just before the executor is
    // destroyed (the only safe moment to read its stats). Set before start().
    void set_finish_hook(std::function<void(Executor&)> hook) { finish_hook_ = std::move(hook); }

    // Runs on the pump thread at most every `interval_ms` (checked after each poll): window title
    // updates and the like. Set before start().
    void set_tick_hook(std::function<void(Executor&)> hook, uint32_t interval_ms = 250) {
        tick_hook_ = std::move(hook);
        tick_interval_ms_ = interval_ms;
    }

    // One-line JSON for the boot report.
    std::string stats_json() const {
        const LivePumpStats s = stats();
        auto u = [](uint64_t v) { return std::to_string(v); };
        return "{\"submitted\": " + u(s.submitted) + ", \"submitted_bytes\": " + u(s.submitted_bytes) +
               ", \"executed\": " + u(s.executed) + ", \"presents\": " + u(s.presents) +
               ", \"blocked_submits\": " + u(s.blocked_submits) + ", \"blocked_ms\": " + u(s.blocked_ns / 1000000) +
               ", \"execute_ms\": " + u(s.execute_ns / 1000000) +
               ", \"dropped_after_close\": " + u(s.dropped_after_close) +
               ", \"discarded_on_close\": " + u(s.discarded_on_close) +
               ", \"max_queue_depth\": " + u(s.max_queue_depth) + ", \"polls\": " + u(s.polls) +
               ", \"window_closed\": " + (s.window_closed ? "true" : "false") + "}";
    }

private:
    bool full_locked(size_t incoming) const {
        if (queue_.empty()) return false;  // an oversized batch must still get in
        return queue_.size() >= options_.max_queued_batches || queued_bytes_ + incoming > options_.max_queued_bytes;
    }

    void run(Factory factory) {
        playgta6::os::name_this_thread(L"pg6-livepump");
        std::string error;
        std::unique_ptr<Executor> exec = factory ? factory(&error) : nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!exec) {
                create_error_ = error.empty() ? "executor creation failed" : error;
            } else {
                running_ = true;
            }
            ready_ = true;
        }
        ready_cv_.notify_all();
        if (!exec) return;

        bool focus = false;
        bool focus_known = false;
        auto last_tick = std::chrono::steady_clock::time_point{};
        auto poll_once = [&]() -> bool {
            const bool open = exec->poll();
            if (tick_hook_ && open) {
                const auto now_t = std::chrono::steady_clock::now();
                if (now_t - last_tick >= std::chrono::milliseconds(tick_interval_ms_)) {
                    last_tick = now_t;
                    tick_hook_(*exec);
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++stats_.polls;
            }
            if constexpr (live_pump_detail::HasFocus<Executor>::value) {
                const bool now = exec->window_focused();
                if ((!focus_known || now != focus) && on_focus_) on_focus_(now);
                focus = now;
                focus_known = true;
            }
            return open;
        };

        bool open = poll_once();  // initial window messages and focus
        std::vector<uint8_t> batch;
        while (open) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_cv_.wait_for(lock, std::chrono::milliseconds(options_.idle_poll_ms),
                                  [&] { return !queue_.empty() || stopping_; });
                if (queue_.empty()) {
                    if (stopping_) break;
                    busy_ = false;
                    idle_cv_.notify_all();
                } else {
                    batch = std::move(queue_.front());
                    queue_.pop_front();
                    queued_bytes_ -= batch.size();
                    busy_ = true;
                    room_cv_.notify_all();
                }
            }
            if (!batch.empty()) {
                const auto t0 = std::chrono::steady_clock::now();
                const uint32_t presents = exec->execute(batch.data(), batch.size());
                const auto ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++stats_.executed;
                    stats_.execute_ns += ns;
                    stats_.presents += presents;
                }
                batch.clear();
            }
            open = poll_once();
        }
        exec->wait_idle();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!open) {
                closed_ = true;
                stats_.discarded_on_close += queue_.size();
                queue_.clear();
                queued_bytes_ = 0;
            }
            busy_ = false;
            running_ = false;
        }
        room_cv_.notify_all();
        idle_cv_.notify_all();
        if (finish_hook_) finish_hook_(*exec);
        if (!open && on_close_) on_close_();
        exec.reset();  // on this thread: the window and device are thread-affine
    }

    const LivePumpOptions options_;
    mutable std::mutex mutex_;
    std::condition_variable work_cv_, room_cv_, idle_cv_, ready_cv_;
    std::deque<std::vector<uint8_t>> queue_;
    size_t queued_bytes_ = 0;
    bool ready_ = false, running_ = false, busy_ = false, closed_ = false, stopping_ = false;
    std::string create_error_;
    LivePumpStats stats_;
    std::function<void()> on_close_;
    std::function<void(bool)> on_focus_;
    std::function<void(Executor&)> finish_hook_;
    std::function<void(Executor&)> tick_hook_;
    uint32_t tick_interval_ms_ = 250;
    std::thread thread_;
};

}  // namespace playgta6::gpu::native_exec
