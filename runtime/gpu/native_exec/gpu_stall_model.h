// Native executor: GPU stall watch. The in-world scene of the std-pack --live-executor boots ended with
// the title's own crash (sub_1423CF620, called by its hang watchdog sub_142215A60 after 30 s without a
// main-loop heartbeat) at ~170 s, ~293 s, ~399 s, ~542 s and ~612 s. Every time the live executor was inside
// one batch for 25-40 s: Dawn's Queue::Submit waited in WaitForSerial for a free command allocator (all in
// flight) while the GPU completed nothing, without a TDR. The pump queue filled, the guest thread that submits
// the next batch blocked in LivePump::submit holding the device lock, ExecuteCommandLists, the render thread
// and the main loop waited behind it. The oldest unfinished submit of every such stall (and of the TDRs
// before them) was the compute pass 0x56d744a4394aeb41 with its one indirect dispatch (cs 0x694fe11e60e62233).
//
// The executor signals a fence of its own after every tracked submit; a watcher thread reads the fence's
// completed value. When work is pending and the completed value has not moved for `threshold_ns`, the GPU
// is stalled: the stall is logged and counted (gpu_stall_events). With PG6_GPU_STALL_REMOVE=1 the executor
// also removes its D3D12 device (ID3D12Device5::RemoveDevice): Dawn's waits return and the recovery starts,
// but in the boot that tried it Device::Destroy then blocked behind the still-running GPU work, so it is
// opt-in. A single submit is kept under Windows' ~2 s TDR by the submit budget, so a GPU that completes
// nothing for several seconds is not busy with a long submit. Pure logic, no GPU.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace playgta6::gpu::native_exec {

constexpr uint64_t kDefaultGpuStallNs = 8000ull * 1000000ull;

// PG6_GPU_STALL_MS: milliseconds without GPU progress before a stall is reported; "0" = off. Anything
// else that is not a number keeps the default.
inline uint64_t gpu_stall_threshold_ns(const char* env) {
    if (!env || !*env || env[0] == '-') return kDefaultGpuStallNs;
    char* end = nullptr;
    const unsigned long long ms = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0') return kDefaultGpuStallNs;
    return ms > UINT64_MAX / 1000000ull ? UINT64_MAX : ms * 1000000ull;
}

class GpuStallWatch {
public:
    explicit GpuStallWatch(uint64_t threshold_ns = kDefaultGpuStallNs) : threshold_ns_(threshold_ns) {}

    // A new device (or none): forget the old fence's values.
    void reset() {
        pending_since_ = 0;
        last_completed_ = 0;
        fired_ = false;
    }

    // One watcher sample: the last value signaled after a submit, the fence's completed value, the time.
    // True once per stall (until the GPU makes progress again or reset()).
    bool sample(uint64_t signaled, uint64_t completed, uint64_t now_ns) {
        if (threshold_ns_ == 0) return false;
        if (completed >= signaled) {  // idle (a removed device reports UINT64_MAX)
            pending_since_ = 0;
            last_completed_ = completed;
            fired_ = false;
            return false;
        }
        if (pending_since_ == 0 || completed != last_completed_) {  // work pending, or the GPU progressed
            pending_since_ = now_ns;
            last_completed_ = completed;
            fired_ = false;
            return false;
        }
        if (fired_ || now_ns - pending_since_ < threshold_ns_) return false;
        fired_ = true;
        return true;
    }

    // How long the GPU has completed nothing while work is pending (0 when idle or progressing).
    uint64_t stalled_ns(uint64_t now_ns) const { return pending_since_ ? now_ns - pending_since_ : 0; }
    uint64_t threshold_ns() const { return threshold_ns_; }

private:
    uint64_t threshold_ns_;
    uint64_t pending_since_ = 0;
    uint64_t last_completed_ = 0;
    bool fired_ = false;
};

}  // namespace playgta6::gpu::native_exec
