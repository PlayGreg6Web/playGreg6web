#pragma once

#include <algorithm>
#include <chrono>
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

// Underrun classification. An underrun is "guest stalled" when guest audio production stopped:
//  - nothing was ever pushed, or
//  - no push for at least the playback time the ring held at the last push (the guest stopped
//    producing and the ring simply drained), or
//  - no push for kGuestStallGapUs (far beyond any normal push cadence), or
//  - the guest frames stopped (no present for kFrameSpikeUs) around the underrun.
// "host starved" only when the guest kept pushing (recent push, within the ring's worth of time)
// but the render thread missed its period.
constexpr uint64_t kGuestStallGapUs = 50000;
constexpr uint64_t kFrameSpikeUs = 100000;
constexpr size_t kMaxUnderrunEvents = 32;

enum class UnderrunCause { GuestStalled, HostStarved };

// gap_us: time since the last push (UINT64_MAX when none ever arrived); fill_at_push_us: playback
// time buffered right after that push; present_age_us: time since the guest last presented a frame
// (UINT64_MAX when unknown).
inline UnderrunCause classify_underrun(uint64_t gap_us, uint64_t fill_at_push_us, uint64_t present_age_us) {
    if (gap_us == UINT64_MAX || gap_us >= kGuestStallGapUs || gap_us >= fill_at_push_us ||
        (present_age_us != UINT64_MAX && present_age_us >= kFrameSpikeUs))
        return UnderrunCause::GuestStalled;
    return UnderrunCause::HostStarved;
}

struct UnderrunEvent {
    uint64_t at_us = 0;            // steady clock
    uint32_t frames_missing = 0;   // silence frames inserted by this pop
    uint64_t gap_us = UINT64_MAX;  // since the last guest push (UINT64_MAX: never pushed)
    uint32_t fill_frames = 0;      // ring fill when the pop started
    uint64_t present_index = 0;    // last guest present seen
    uint64_t present_age_us = UINT64_MAX;
    UnderrunCause cause = UnderrunCause::HostStarved;
};

inline uint64_t steady_now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

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
    uint64_t underruns = 0;        // times the buffer ran dry while playing (= host_starved + guest_stalled)
    uint64_t underruns_host_starved = 0;  // dry, but audio had been arriving (recent push): host-side lateness
    uint64_t underruns_guest_stalled = 0; // dry guest stopped producing, see classify_underrun
    uint64_t underrun_frames = 0;  // silence frames inserted after playback had started
    uint64_t overrun_frames = 0;   // frames dropped because the buffer was full
    std::vector<UnderrunEvent> events;  // the first kMaxUnderrunEvents underruns
    uint64_t events_dropped = 0;
};

// Interleaved-stereo jitter buffer. Playback starts once `prebuffer` frames are queued; after an
// underrun it re-primes. push() never blocks beyond a short lock and drops what does not fit.
class JitterBuffer {
public:
    void configure(uint32_t capacity_frames, uint32_t prebuffer_frames, uint32_t rate = kSourceRate) {
        std::lock_guard<std::mutex> lock(m_);
        rate_ = std::max<uint32_t>(rate, 1);
        fill_at_push_ = 0;
        present_index_ = 0;
        present_change_us_ = 0;
        cap_ = std::max<uint32_t>(capacity_frames, 1);
        prebuffer_ = std::min(prebuffer_frames, cap_);
        buf_.assign(static_cast<size_t>(cap_) * 2, 0.0f);
        head_ = count_ = 0;
        primed_ = false;
        was_playing_ = false;
        stats_ = Stats{};
        last_push_us_ = 0;
    }

    void push(const float* stereo, uint32_t frames) { push(stereo, frames, steady_now_us()); }
    void push(const float* stereo, uint32_t frames, uint64_t now_us) {
        std::lock_guard<std::mutex> lock(m_);
        const uint32_t room = cap_ - count_;
        const uint32_t n = std::min(frames, room);
        for (uint32_t i = 0; i < n; ++i) {
            const size_t slot = (static_cast<size_t>(head_) + count_ + i) % cap_;
            buf_[2 * slot] = stereo[2 * i];
            buf_[2 * slot + 1] = stereo[2 * i + 1];
        }
        count_ += n;
        if (frames > 0) {
            last_push_us_ = now_us;
            fill_at_push_ = count_;
        }
        stats_.frames_in += n;
        stats_.overrun_frames += frames - n;
        if (!primed_ && count_ >= prebuffer_) primed_ = true;
    }

    // The guest present counter, sampled by the producer; its last change time stands in for "frames stopped".
    void note_present(uint64_t index) { note_present(index, steady_now_us()); }
    void note_present(uint64_t index, uint64_t now_us) {
        std::lock_guard<std::mutex> lock(m_);
        if (present_change_us_ == 0 || index != present_index_) {
            present_index_ = index;
            present_change_us_ = now_us;
        }
    }

    // Fills `frames` interleaved stereo frames; the part that cannot be served is silence.
    // Returns the number of real frames written.
    uint32_t pop(float* out, uint32_t frames) { return pop(out, frames, steady_now_us()); }
    uint32_t pop(float* out, uint32_t frames, uint64_t now_us) {
        std::lock_guard<std::mutex> lock(m_);
        uint32_t n = 0;
        const uint32_t fill_before = count_;
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
                if (primed_) {
                    ++stats_.underruns;
                    UnderrunEvent ev;
                    ev.at_us = now_us;
                    ev.frames_missing = frames - n;
                    ev.fill_frames = fill_before;
                    ev.present_index = present_index_;
                    if (last_push_us_ != 0) ev.gap_us = now_us >= last_push_us_ ? now_us - last_push_us_ : 0;
                    if (present_change_us_ != 0)
                        ev.present_age_us = now_us >= present_change_us_ ? now_us - present_change_us_ : 0;
                    ev.cause = classify_underrun(ev.gap_us, static_cast<uint64_t>(fill_at_push_) * 1000000ull / rate_,
                                                 ev.present_age_us);
                    if (ev.cause == UnderrunCause::GuestStalled)
                        ++stats_.underruns_guest_stalled;
                    else
                        ++stats_.underruns_host_starved;
                    if (stats_.events.size() < kMaxUnderrunEvents)
                        stats_.events.push_back(ev);
                    else
                        ++stats_.events_dropped;
                }
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
    uint64_t last_push_us_ = 0;
    uint32_t rate_ = kSourceRate, fill_at_push_ = 0;
    uint64_t present_index_ = 0, present_change_us_ = 0;
    Stats stats_;
};

// Device-independent feed: resamples 48 kHz source frames to the device rate into the buffer.
class Feed {
public:
    void configure(uint32_t device_rate, uint32_t capacity_ms, uint32_t prebuffer_ms) {
        rate_ = device_rate;
        rs_.configure(kSourceRate, device_rate);
        jb_.configure(device_rate * capacity_ms / 1000, device_rate * prebuffer_ms / 1000, device_rate);
    }
    // Optional producer-side present counter (the guest frame index) for underrun classification.
    void set_present_source(uint64_t (*fn)()) { present_fn_ = fn; }
    void push(const float* stereo, uint32_t frames, uint32_t sample_rate) {
        if (!stereo || sample_rate != kSourceRate || frames == 0) return;
        std::lock_guard<std::mutex> lock(m_);
        scratch_.clear();
        rs_.process(stereo, frames, scratch_);
        if (present_fn_) jb_.note_present(present_fn_());
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
    uint64_t (*present_fn_)() = nullptr;
};

// Render-thread scheduling class (--audio-mmcss / PLAYGTA6_AUDIO_MMCSS). Default ProAudio.
enum class MmcssMode { ProAudio, Audio, Off };

inline const char* mmcss_mode_name(MmcssMode m) {
    return m == MmcssMode::Audio ? "audio" : m == MmcssMode::Off ? "off" : "pro-audio";
}

// "pro-audio" | "audio" | "off" (case-insensitive). False (and *out untouched) for anything else.
inline bool parse_mmcss_mode(const std::string& text, MmcssMode* out) {
    std::string t;
    for (char c : text) t += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    if (t == "pro-audio") *out = MmcssMode::ProAudio;
    else if (t == "audio") *out = MmcssMode::Audio;
    else if (t == "off") *out = MmcssMode::Off;
    else return false;
    return true;
}

inline std::string report_json(bool device_open, uint32_t device_rate, const Stats& s, const std::string& error,
                        uint32_t device_channels = 0, uint32_t device_channel_mask = 0,
                        uint64_t epoch_us = 0, MmcssMode mmcss_mode = MmcssMode::ProAudio,
                        bool mmcss_registered = false) {
    std::string o = "{\"enabled\": true, \"device_open\": ";
    o += device_open ? "true" : "false";
    o += ", \"device_rate\": " + std::to_string(device_rate);
    // The device's real mix layout; we always render stereo into it (the engine up-mixes).
    o += ", \"device_channels\": " + std::to_string(device_channels);
    o += ", \"device_channel_mask\": " + std::to_string(device_channel_mask);
    o += ", \"render_channels\": 2";
    o += std::string(", \"mmcss_mode\": \"") + mmcss_mode_name(mmcss_mode) + "\"";
    o += std::string(", \"mmcss_registered\": ") + (mmcss_registered ? "true" : "false");
    o += ", \"frames_in\": " + std::to_string(s.frames_in);
    o += ", \"frames_played\": " + std::to_string(s.frames_played);
    o += ", \"underruns\": " + std::to_string(s.underruns);
    o += ", \"underruns_host_starved\": " + std::to_string(s.underruns_host_starved);
    o += ", \"underruns_guest_stalled\": " + std::to_string(s.underruns_guest_stalled);
    o += ", \"underrun_frames\": " + std::to_string(s.underrun_frames);
    o += ", \"overrun_frames\": " + std::to_string(s.overrun_frames);
    // Bounded list of underrun events; at_ms is relative to epoch_us (the boot start) so it lines up with
    // frame_times hitches. ms_since_* are -1 when unknown.
    o += ", \"underrun_events_dropped\": " + std::to_string(s.events_dropped);
    o += ", \"underrun_events\": [";
    auto ms = [](uint64_t us) {
        if (us == UINT64_MAX) return std::string("-1");
        return std::to_string(us / 1000) + "." + std::to_string(us % 1000 / 100);
    };
    for (size_t i = 0; i < s.events.size(); ++i) {
        const UnderrunEvent& e = s.events[i];
        o += i ? ", " : "";
        o += "{\"at_ms\": " + ms(e.at_us >= epoch_us ? e.at_us - epoch_us : 0);
        o += ", \"frames_missing\": " + std::to_string(e.frames_missing);
        o += ", \"ms_since_push\": " + ms(e.gap_us);
        o += ", \"ring_fill_frames\": " + std::to_string(e.fill_frames);
        o += ", \"ring_fill_ms\": " +
             ms(static_cast<uint64_t>(e.fill_frames) * 1000000ull / std::max<uint32_t>(device_rate, 1));
        o += ", \"present_index\": " + std::to_string(e.present_index);
        o += ", \"ms_since_present\": " + ms(e.present_age_us);
        o += std::string(", \"cause\": \"") +
             (e.cause == UnderrunCause::GuestStalled ? "guest_stalled" : "host_starved") + "\"}";
    }
    o += "]";
    std::string esc;
    for (char c : error) esc += (c == '"' || c == '\\') ? '_' : c;
    o += ", \"error\": \"" + esc + "\"}";
    return o;
}

}  // namespace playgta6::os::audio_out
