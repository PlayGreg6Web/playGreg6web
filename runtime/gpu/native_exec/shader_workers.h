// Native executor: worker threads that materialize pack shaders (read the WGSL from the pack, compile
// the Dawn shader module, run the text scans) off the executor's queue thread. A std-pack compute
// shader is ~3 MB of WGSL and takes 0.3-2.5 s to parse; done on the queue thread, 750 of them blocked
// it for 78 s of a 138 s boot (the live pump filled, the title's submits blocked, its watchdog fired).
// The queue thread hands a shader over when a pipeline first needs it and only waits for it when a
// draw / dispatch needs that pipeline (within the frame's pipeline wait budget, pipe_budget_model.h).
// Pure std, no GPU: the job is a closure.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace playgta6::gpu::native_exec {

// Worker count for `hw` hardware threads: a quarter of them, 1..4 (the title's own threads and
// Dawn's pipeline compiles need the rest). `env` (PG6_SHADER_WORKERS) overrides it; "0" = off.
inline unsigned shader_worker_count(unsigned hw, const char* env) {
    if (env && *env) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end != env && *end == '\0') return v > 16 ? 16u : static_cast<unsigned>(v);
    }
    const unsigned n = hw / 4;
    return n < 1 ? 1u : n > 4 ? 4u : n;
}

class ShaderWorkers {
public:
    ShaderWorkers() = default;
    ShaderWorkers(const ShaderWorkers&) = delete;
    ShaderWorkers& operator=(const ShaderWorkers&) = delete;
    ~ShaderWorkers() { stop(); }

    void start(unsigned n) {
        stop();
        stopping_ = false;
        for (unsigned i = 0; i < n; ++i) threads_.emplace_back([this] { run(); });
    }
    bool running() const { return !threads_.empty(); }
    unsigned size() const { return static_cast<unsigned>(threads_.size()); }

    void submit(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            jobs_.push_back(std::move(job));
            ++submitted_;
        }
        work_cv_.notify_one();
    }

    // Jobs finished so far (a cheap "anything new?" check for the queue thread).
    uint64_t completed() const { return completed_.load(std::memory_order_acquire); }
    // Jobs submitted and not finished yet.
    uint64_t outstanding() const {
        std::lock_guard<std::mutex> lock(mu_);
        return submitted_ - completed_.load(std::memory_order_acquire);
    }

    // Blocks until `ready()` holds or `timeout_ns` passed (UINT64_MAX = no limit); returns ready().
    // `ready` is re-checked after every finished job.
    template <class Pred>
    bool wait(Pred ready, uint64_t timeout_ns) {
        if (ready()) return true;
        std::unique_lock<std::mutex> lock(mu_);
        if (timeout_ns == UINT64_MAX) {
            done_cv_.wait(lock, [&] { return ready() || threads_.empty(); });
        } else {
            done_cv_.wait_for(lock, std::chrono::nanoseconds(timeout_ns), [&] { return ready() || threads_.empty(); });
        }
        return ready();
    }

    // Joins the threads; jobs not started yet are dropped (their shaders never become ready, so only
    // call this when nothing waits for them any more: teardown).
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopping_ = true;
            jobs_.clear();
        }
        work_cv_.notify_all();
        for (auto& t : threads_) t.join();
        threads_.clear();
        done_cv_.notify_all();
    }

private:
    void run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mu_);
                work_cv_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
                if (stopping_) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
            {
                std::lock_guard<std::mutex> lock(mu_);
                completed_.fetch_add(1, std::memory_order_release);
            }
            done_cv_.notify_all();
        }
    }

    mutable std::mutex mu_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<std::function<void()>> jobs_;
    std::vector<std::thread> threads_;
    bool stopping_ = false;
    uint64_t submitted_ = 0;
    std::atomic<uint64_t> completed_{0};
};

}  // namespace playgta6::gpu::native_exec
