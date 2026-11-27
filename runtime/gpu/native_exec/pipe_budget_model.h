// Native executor: per-frame wait budget for asynchronous pipeline creation (backlog 14). A pipeline
// the prewarm did not finish in time used to be awaited forever, so one slow compile froze the frame
// (and tripped the title's watchdog). Now a frame may wait ~0.5 s in total for normal pipelines and
// ~10 s for fullscreen passes (vertex-buffer-less pipelines: post-process / blit, which would leave a
// visibly broken frame if skipped); past the budget the draw is skipped (the compile keeps running and
// the pipeline is used once ready on a later frame) and counted as `pipeline_pending`. Pure logic, no GPU.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace playgta6::gpu::native_exec {

constexpr uint64_t kPipeWaitUnlimited = UINT64_MAX;
constexpr uint64_t kDefaultPipeWaitFrameNs = 500ull * 1000000ull;
constexpr uint64_t kDefaultPipeWaitFullscreenNs = 10000ull * 1000000ull;

// Milliseconds text -> ns. "inf" / "unlimited" / "-1" = unlimited (the old always-wait behaviour),
// "0" = never wait. Returns false (out untouched) on anything else.
inline bool parse_pipe_wait_ms(const char* s, uint64_t* out) {
    if (!s || !*s) return false;
    if (!std::strcmp(s, "inf") || !std::strcmp(s, "unlimited") || !std::strcmp(s, "-1")) {
        *out = kPipeWaitUnlimited;
        return true;
    }
    char* end = nullptr;
    const unsigned long long ms = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || s[0] == '-') return false;
    *out = ms > UINT64_MAX / 1000000ull ? kPipeWaitUnlimited : ms * 1000000ull;
    return true;
}

// Budget of one frame. Normal and fullscreen waits are charged to separate pools.
class PipeWaitBudget {
public:
    void configure(uint64_t frame_ns, uint64_t fullscreen_ns) {
        frame_ns_ = frame_ns;
        fullscreen_ns_ = fullscreen_ns;
    }
    void begin_frame() { spent_normal_ = spent_fullscreen_ = 0; }
    // How long a pipeline wait may still block (0 = do not wait, skip the draw).
    uint64_t allowance(bool fullscreen) const {
        const uint64_t cap = fullscreen ? fullscreen_ns_ : frame_ns_;
        const uint64_t spent = fullscreen ? spent_fullscreen_ : spent_normal_;
        if (cap == kPipeWaitUnlimited) return kPipeWaitUnlimited;
        return spent >= cap ? 0 : cap - spent;
    }
    void charge(bool fullscreen, uint64_t ns) { (fullscreen ? spent_fullscreen_ : spent_normal_) += ns; }
    uint64_t spent(bool fullscreen) const { return fullscreen ? spent_fullscreen_ : spent_normal_; }

private:
    uint64_t frame_ns_ = kDefaultPipeWaitFrameNs;
    uint64_t fullscreen_ns_ = kDefaultPipeWaitFullscreenNs;
    uint64_t spent_normal_ = 0;
    uint64_t spent_fullscreen_ = 0;
};

}  // namespace playgta6::gpu::native_exec
