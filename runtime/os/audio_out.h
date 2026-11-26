#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// --audio-out: live playback of the stereo float32 mix (audio_downmix::set_stereo_sink).
// This header holds the device-independent part (linear resampler, jitter buffer with underrun
// accounting, the sink glue); the Windows WASAPI device lives in audio_out_wasapi.h. The guest
// is never stalled: the sink only does a short locked copy, the device thread drains the buffer
// and plays silence (counted as an underrun) when it is empty.
namespace playgta6::os::audio_out {

constexpr uint32_t kSourceRate = 48000;

// Streaming linear-interpolation resampler for interleaved stereo.
class Resampler {
public:
    void configure(uint32_t in_rate, uint32_t out_rate) {
        step_ = out_rate ? static_cast<double>(in_rate) / static_cast<double>(out_rate) : 1.0;
        passthrough_ = in_rate == out_rate;
        pos_ = 0.0;
        prev_[0] = prev_[1] = 0.0f;
    }
    // Appends the resampled frames to `out`.
    void process(const float* in, uint32_t frames, std::vector<float>& out) {
        if (passthrough_) {
            out.insert(out.end(), in, in + static_cast<size_t>(frames) * 2);
            return;
        }
        for (uint32_t i = 0; i < frames; ++i) {
            const float x0 = in[2 * i], x1 = in[2 * i + 1];
            while (pos_ < 1.0) {
                const float t = static_cast<float>(pos_);
                out.push_back(prev_[0] + (x0 - prev_[0]) * t);
                out.push_back(prev_[1] + (x1 - prev_[1]) * t);
                pos_ += step_;
            }
            pos_ -= 1.0;
            prev_[0] = x0;
            prev_[1] = x1;
        }
    }

private:
    double step_ = 1.0;
    double pos_ = 0.0;
    float prev_[2] = {0.0f, 0.0f};
    bool passthrough_ = true;
};

struct Stats {
    uint64_t frames_in = 0;        // device-rate frames accepted into the buffer
    uint64_t frames_played = 0;    // real frames handed to the device
    uint64_t underruns = 0;        // times the buffer ran dry while playing
    uint64_t underrun_frames = 0;  // silence frames inserted after playback had started
    uint64_t overrun_frames = 0;   // frames dropped because the buffer was full
};

// Interleaved-stereo jitter buffer. Playback starts once `prebuffer` frames are queued; after an
// underrun it re-primes. push() never blocks beyond a short lock and drops what does not fit.
class JitterBuffer {
public:
    void configure(uint32_t capacity_frames, uint32_t prebuffer_frames) {
        std::lock_guard<std::mutex> lock(m_);
        cap_ = std::max<uint32_t>(capacity_frames, 1);
        prebuffer_ = std::min(prebuffer_frames, cap_);
        buf_.assign(static_cast<size_t>(cap_) * 2, 0.0f);
        head_ = count_ = 0;
        primed_ = false;
        was_playing_ = false;
        stats_ = Stats{};
    }

    void push(const float* stereo, uint32_t frames) {
        std::lock_guard<std::mutex> lock(m_);
        const uint32_t room = cap_ - count_;
        const uint32_t n = std::min(frames, room);
        for (uint32_t i = 0; i < n; ++i) {
            const size_t slot = (static_cast<size_t>(head_) + count_ + i) % cap_;
            buf_[2 * slot] = stereo[2 * i];
            buf_[2 * slot + 1] = stereo[2 * i + 1];
        }
        count_ += n;
        stats_.frames_in += n;
        stats_.overrun_frames += frames - n;
        if (!primed_ && count_ >= prebuffer_) primed_ = true;
    }

    // Fills `frames` interleaved stereo frames; the part that cannot be served is silence.
    // Returns the number of real frames written.
    uint32_t pop(float* out, uint32_t frames) {
        std::lock_guard<std::mutex> lock(m_);
        uint32_t n = 0;
        if (primed_) {
            n = std::min(frames, count_);
            for (uint32_t i = 0; i < n; ++i) {
                const size_t slot = (static_cast<size_t>(head_) + i) % cap_;
                out[2 * i] = buf_[2 * slot];
                out[2 * i + 1] = buf_[2 * slot + 1];
            }
            head_ = static_cast<uint32_t>((static_cast<size_t>(head_) + n) % cap_);
            count_ -= n;
            stats_.frames_played += n;
            if (n > 0) was_playing_ = true;
        }
        if (n < frames) {
            std::memset(out + 2 * static_cast<size_t>(n), 0, sizeof(float) * 2 * (frames - n));
            // Only a stream that already started counts; the initial priming is not an underrun.
            if (was_playing_) {
                if (primed_) ++stats_.underruns;
                stats_.underrun_frames += frames - n;
                primed_ = false;  // re-prime before resuming
            }
        }
        return n;
    }

    uint32_t buffered() const {
        std::lock_guard<std::mutex> lock(m_);
        return count_;
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(m_);
        return stats_;
    }

private:
    mutable std::mutex m_;
    std::vector<float> buf_;
    uint32_t cap_ = 1, prebuffer_ = 0, head_ = 0, count_ = 0;
    bool primed_ = false, was_playing_ = false;
    Stats stats_;
};

// Device-independent feed: resamples 48 kHz source frames to the device rate into the buffer.
class Feed {
public:
    void configure(uint32_t device_rate, uint32_t capacity_ms, uint32_t prebuffer_ms) {
        rate_ = device_rate;
        rs_.configure(kSourceRate, device_rate);
        jb_.configure(device_rate * capacity_ms / 1000, device_rate * prebuffer_ms / 1000);
    }
    void push(const float* stereo, uint32_t frames, uint32_t sample_rate) {
        if (!stereo || sample_rate != kSourceRate || frames == 0) return;
        std::lock_guard<std::mutex> lock(m_);
        scratch_.clear();
        rs_.process(stereo, frames, scratch_);
        jb_.push(scratch_.data(), static_cast<uint32_t>(scratch_.size() / 2));
    }
    static void sink(void* user, const float* stereo, uint32_t frames, uint32_t sample_rate) {
        static_cast<Feed*>(user)->push(stereo, frames, sample_rate);
    }
    JitterBuffer& buffer() { return jb_; }
    uint32_t device_rate() const { return rate_; }

private:
    std::mutex m_;
    Resampler rs_;
    JitterBuffer jb_;
    std::vector<float> scratch_;
    uint32_t rate_ = kSourceRate;
};

inline std::string report_json(bool device_open, uint32_t device_rate, const Stats& s, const std::string& error) {
    std::string o = "{\"enabled\": true, \"device_open\": ";
    o += device_open ? "true" : "false";
    o += ", \"device_rate\": " + std::to_string(device_rate);
    o += ", \"frames_in\": " + std::to_string(s.frames_in);
    o += ", \"frames_played\": " + std::to_string(s.frames_played);
    o += ", \"underruns\": " + std::to_string(s.underruns);
    o += ", \"underrun_frames\": " + std::to_string(s.underrun_frames);
    o += ", \"overrun_frames\": " + std::to_string(s.overrun_frames);
    std::string esc;
    for (char c : error) esc += (c == '"' || c == '\\') ? '_' : c;
    o += ", \"error\": \"" + esc + "\"}";
    return o;
}

}  // namespace playgta6::os::audio_out
