#include "os/audio_out_wasapi.h"
#include "os/thread_name.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <objbase.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#endif

namespace playgta6::os::audio_out {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kSoftLimitKnee = 0.9f;
constexpr uint32_t kTrimCrossfadeFrames = 64;
constexpr uint32_t kJitterCapacityMs = 400;
constexpr uint32_t kJitterPrebufferMs = 60;
constexpr uint32_t kWasapiTargetPadMs = 40;   // strictly < kJitterPrebufferMs to prevent stutter loop
constexpr uint32_t kTrimHighWatermarkMs = 180;
constexpr uint32_t kTrimTargetWatermarkMs = 80;

inline float sanitize_and_limit(float v) {
    if (!std::isfinite(v)) {
        return 0.0f;
    }
    constexpr float kHeadroom = 1.0f - kSoftLimitKnee;
    if (v > kSoftLimitKnee) {
        const float c = kSoftLimitKnee + kHeadroom * std::tanh((v - kSoftLimitKnee) / kHeadroom);
        return c > 1.0f ? 1.0f : c;
    }
    if (v < -kSoftLimitKnee) {
        const float c = kSoftLimitKnee + kHeadroom * std::tanh((-v - kSoftLimitKnee) / kHeadroom);
        return c > 1.0f ? -1.0f : -c;
    }
    return v;
}

inline float half_cosine_down(uint32_t pos, uint32_t total) {
    if (total == 0 || pos + 1 >= total) {
        return 0.0f;
    }
    const float t = static_cast<float>(pos + 1) / static_cast<float>(total);
    return 0.5f * (1.0f + std::cos(kPi * t));
}

inline float half_cosine_up(uint32_t pos, uint32_t total) {
    if (total == 0) {
        return 1.0f;
    }
    const float t = static_cast<float>(pos) / static_cast<float>(total);
    return 0.5f * (1.0f - std::cos(kPi * t));
}

}  // namespace

struct Wasapi::Impl {
#ifdef _WIN32
    static const GUID& clsid_enumerator() {
        static const GUID g = {0xBCDE0395, 0xE52F, 0x467C, {0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}};
        return g;
    }
    static const GUID& iid_enumerator() {
        static const GUID g = {0xA95664D2, 0x9614, 0x4F35, {0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6}};
        return g;
    }
    static const GUID& iid_audio_client() {
        static const GUID g = {0x1CB9AD4C, 0xDBFA, 0x4C32, {0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2}};
        return g;
    }
    static const GUID& iid_render_client() {
        static const GUID g = {0xF294ACFC, 0x3146, 0x4483, {0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2}};
        return g;
    }

    bool open_device_interfaces(uint32_t* rate, std::string* err) {
        IMMDeviceEnumerator* en = nullptr;
        HRESULT hr = CoCreateInstance(clsid_enumerator(), nullptr, CLSCTX_ALL, iid_enumerator(),
                                      reinterpret_cast<void**>(&en));
        if (FAILED(hr)) return fail(err, "no device enumerator", hr);
        hr = en->GetDefaultAudioEndpoint(eRender, eConsole, &dev_);
        en->Release();
        if (FAILED(hr)) return fail(err, "no default render endpoint", hr);
        hr = dev_->Activate(iid_audio_client(), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client_));
        if (FAILED(hr)) return fail(err, "cannot activate audio client", hr);
        WAVEFORMATEX* mix = nullptr;
        hr = client_->GetMixFormat(&mix);
        if (FAILED(hr) || !mix) return fail(err, "GetMixFormat failed", hr);
        const uint32_t dev_rate = mix->nSamplesPerSec ? mix->nSamplesPerSec : kSourceRate;
        dev_channels_ = mix->nChannels;
        dev_mask_ = 0;
        if (mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE && mix->cbSize >= 22) {
            dev_mask_ = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(mix)->dwChannelMask;
        } else if (mix->nChannels == 2) {
            dev_mask_ = 0x3;
        }
        CoTaskMemFree(mix);
        WAVEFORMATEX fmt = {};
        fmt.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
        fmt.nChannels = 2;
        fmt.nSamplesPerSec = dev_rate;
        fmt.wBitsPerSample = 32;
        fmt.nBlockAlign = 8;
        fmt.nAvgBytesPerSec = dev_rate * 8;
        // Request a 100 ms shared-mode buffer; run() writes ahead to at most kWasapiTargetPadMs
        // (40 ms < 60 ms JitterBuffer prebuffer) so post-underrun re-priming never oscillates.
        // AUTOCONVERTPCM lets the engine accept stereo float even on a surround mix format.
        // Event-driven: the engine signals every period, so the render thread wakes on the device
        // clock instead of a coarse Sleep(5) that a loaded host can stretch by tens of ms.
        DWORD flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        if (!event_) event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        event_driven_ = event_ != nullptr;
        if (event_driven_) flags |= AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
        hr = client_->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 1000000 /* 100 ms */, 0, &fmt, nullptr);
        if (FAILED(hr) && event_driven_) {
            // Some endpoints refuse the callback mode: fall back to polling.
            client_->Release();
            client_ = nullptr;
            event_driven_ = false;
            hr = dev_->Activate(iid_audio_client(), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client_));
            if (SUCCEEDED(hr))
                hr = client_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                         AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                                         1000000, 0, &fmt, nullptr);
        }
        if (FAILED(hr)) return fail(err, "IAudioClient::Initialize failed", hr);
        if (event_driven_ && FAILED(client_->SetEventHandle(event_))) {
            // Cannot happen after a successful callback-mode Initialize; poll rather than hang.
            event_driven_ = false;
        }
        hr = client_->GetBufferSize(&buffer_frames_);
        if (FAILED(hr)) return fail(err, "GetBufferSize failed", hr);
        hr = client_->GetService(iid_render_client(), reinterpret_cast<void**>(&render_));
        if (FAILED(hr)) return fail(err, "GetService(IAudioRenderClient) failed", hr);
        *rate = dev_rate;
        return true;
    }

    bool open_device(uint32_t* rate, std::string* err) {
        const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        com_inited_ = SUCCEEDED(co);
        com_inited_thread_id_ = com_inited_ ? GetCurrentThreadId() : 0;
        return open_device_interfaces(rate, err);
    }

    static bool fail(std::string* err, const char* what, HRESULT hr) {
        char b[128];
        std::snprintf(b, sizeof(b), "%s (hr=0x%08lx)", what, static_cast<unsigned long>(hr));
        if (err) *err = b;
        return false;
    }

    void release_interfaces() {
        if (render_) render_->Release();
        if (client_) client_->Release();
        if (dev_) dev_->Release();
        render_ = nullptr;
        client_ = nullptr;
        dev_ = nullptr;
        buffer_frames_ = 0;
    }

    void close_device() {
        release_interfaces();
        if (event_) {
            CloseHandle(event_);
            event_ = nullptr;
        }
        device_rate_ = 0;
        if (com_inited_ && GetCurrentThreadId() == com_inited_thread_id_) {
            CoUninitialize();
        }
        com_inited_ = false;
        com_inited_thread_id_ = 0;
    }

    void reset_declick_state(uint32_t rate) {
        declick_frames_ = std::max<uint32_t>(32, rate * 160 / kSourceRate);
        last_l_ = 0.0f;
        last_r_ = 0.0f;
        has_active_audio_ = false;
        in_underrun_ = false;
        fade_out_pos_ = 0;
        fade_out_total_ = 0;
        fade_out_start_l_ = 0.0f;
        fade_out_start_r_ = 0.0f;
        fade_in_pos_ = 0;
        fade_in_total_ = 0;
        fade_in_from_l_ = 0.0f;
        fade_in_from_r_ = 0.0f;
        trim_xfade_len_ = 0;
    }

    // Processes `want` stereo frames from `feed_->buffer()` into `dst`, applying NaN/Inf
    // sanitization, tanh soft-limiting, bounded backlog trim crossfade, half-cosine underrun
    // fade-out, and recovery fade-in. Returns true if the entire block is pure steady silence.
    bool render_frames(float* dst, uint32_t want, uint32_t rate) {
        const uint32_t high_water = std::max<uint32_t>(rate * kTrimHighWatermarkMs / 1000, want + 256);
        const uint32_t target_water = std::max<uint32_t>(rate * kTrimTargetWatermarkMs / 1000, rate * kJitterPrebufferMs / 1000);
        const uint32_t buffered = feed_->buffer().buffered();
        if (!in_underrun_ && buffered > high_water && buffered > target_water) {
            const uint32_t excess = buffered - target_water;
            const uint32_t max_trim = static_cast<uint32_t>(trim_scratch_.size() / 2);
            const uint32_t to_skip = std::min(excess, max_trim);
            if (to_skip > 0) {
                const uint32_t skipped = feed_->buffer().pop(trim_scratch_.data(), to_skip);
                if (skipped > 0 && has_active_audio_) {
                    const uint32_t xf = std::min<uint32_t>(kTrimCrossfadeFrames, skipped);
                    for (uint32_t i = 0; i < xf; ++i) {
                        trim_xfade_[2 * i] = sanitize_and_limit(trim_scratch_[2 * i]);
                        trim_xfade_[2 * i + 1] = sanitize_and_limit(trim_scratch_[2 * i + 1]);
                    }
                    trim_xfade_len_ = xf;
                }
            }
        }

        const uint32_t got = feed_->buffer().pop(dst, want);

        for (uint32_t i = 0; i < got * 2; ++i) {
            dst[i] = sanitize_and_limit(dst[i]);
        }

        if (trim_xfade_len_ > 0 && got > 0) {
            const uint32_t xf = std::min(trim_xfade_len_, got);
            for (uint32_t i = 0; i < xf; ++i) {
                const float t = static_cast<float>(i + 1) / static_cast<float>(xf + 1);
                const float w_new = 0.5f * (1.0f - std::cos(kPi * t));
                const float w_old = 1.0f - w_new;
                dst[2 * i] = trim_xfade_[2 * i] * w_old + dst[2 * i] * w_new;
                dst[2 * i + 1] = trim_xfade_[2 * i + 1] * w_old + dst[2 * i + 1] * w_new;
            }
            trim_xfade_len_ = 0;
        }

        if (got > 0) {
            if (in_underrun_) {
                if (fade_out_total_ > 0 && fade_out_pos_ < fade_out_total_) {
                    const float g = half_cosine_down(fade_out_pos_, fade_out_total_);
                    fade_in_from_l_ = fade_out_start_l_ * g;
                    fade_in_from_r_ = fade_out_start_r_ * g;
                } else {
                    fade_in_from_l_ = 0.0f;
                    fade_in_from_r_ = 0.0f;
                }
                fade_out_pos_ = 0;
                fade_out_total_ = 0;
                in_underrun_ = false;
                fade_in_pos_ = 0;
                fade_in_total_ = declick_frames_;
            }
            if (fade_in_total_ > 0 && fade_in_pos_ < fade_in_total_) {
                const uint32_t rem = fade_in_total_ - fade_in_pos_;
                const uint32_t ramp = std::min(got, rem);
                for (uint32_t i = 0; i < ramp; ++i) {
                    const float w_new = half_cosine_up(fade_in_pos_ + i, fade_in_total_);
                    const float w_old = 1.0f - w_new;
                    dst[2 * i] = fade_in_from_l_ * w_old + dst[2 * i] * w_new;
                    dst[2 * i + 1] = fade_in_from_r_ * w_old + dst[2 * i + 1] * w_new;
                }
                fade_in_pos_ += ramp;
                if (fade_in_pos_ >= fade_in_total_) {
                    fade_in_pos_ = 0;
                    fade_in_total_ = 0;
                }
            }
            last_l_ = dst[2 * (got - 1)];
            last_r_ = dst[2 * (got - 1) + 1];
            has_active_audio_ = true;
        }

        bool wrote_fade_out = false;
        if (got < want) {
            fade_in_pos_ = 0;
            fade_in_total_ = 0;
            trim_xfade_len_ = 0;
            if (!in_underrun_ && has_active_audio_) {
                fade_out_start_l_ = last_l_;
                fade_out_start_r_ = last_r_;
                fade_out_pos_ = 0;
                fade_out_total_ = declick_frames_;
                has_active_audio_ = false;
            }
            in_underrun_ = true;
            uint32_t pos = got;
            while (pos < want && fade_out_total_ > 0 && fade_out_pos_ < fade_out_total_) {
                const float g = half_cosine_down(fade_out_pos_, fade_out_total_);
                dst[2 * pos] = fade_out_start_l_ * g;
                dst[2 * pos + 1] = fade_out_start_r_ * g;
                ++fade_out_pos_;
                ++pos;
                wrote_fade_out = true;
            }
            if (fade_out_total_ > 0 && fade_out_pos_ >= fade_out_total_) {
                fade_out_pos_ = 0;
                fade_out_total_ = 0;
                last_l_ = 0.0f;
                last_r_ = 0.0f;
            }
            if (pos < want) {
                std::memset(dst + 2 * static_cast<size_t>(pos), 0, sizeof(float) * 2 * (want - pos));
            }
        }

        return got == 0 && !wrote_fade_out;
    }

    static bool is_device_lost(HRESULT hr) {
        return hr == AUDCLNT_E_DEVICE_INVALIDATED || hr == AUDCLNT_E_SERVICE_NOT_RUNNING;
    }

    // Waits for the next device period (event) or at most `max_ms`.
    void wait_period(DWORD max_ms) {
        if (event_driven_ && event_) {
            WaitForSingleObject(event_, max_ms);
        } else {
            Sleep(5);
        }
    }

    // Scheduling hardening for the render thread: MMCSS "Pro Audio" (resolved at runtime so no
    // avrt.lib link dependency), falling back to a raised thread priority.
    void boost_render_thread() {
        typedef HANDLE(WINAPI * AvSetFn)(LPCWSTR, LPDWORD);
        mmcss_registered_ = false;
        if (mmcss_mode_ == MmcssMode::Off) return;
        avrt_ = LoadLibraryW(L"avrt.dll");
        if (avrt_) {
            auto set = reinterpret_cast<AvSetFn>(GetProcAddress(avrt_, "AvSetMmThreadCharacteristicsW"));
            if (set) {
                DWORD task = 0;
                mmcss_ = set(mmcss_mode_ == MmcssMode::Audio ? L"Audio" : L"Pro Audio", &task);
            }
        }
        mmcss_registered_ = mmcss_ != nullptr;
        if (!mmcss_) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }

    void unboost_render_thread() {
        if (mmcss_ && avrt_) {
            typedef BOOL(WINAPI * AvRevertFn)(HANDLE);
            auto rev = reinterpret_cast<AvRevertFn>(GetProcAddress(avrt_, "AvRevertMmThreadCharacteristics"));
            if (rev) rev(mmcss_);
        }
        mmcss_ = nullptr;
        if (avrt_) FreeLibrary(avrt_);
        avrt_ = nullptr;
    }

    void run() {
        playgta6::os::name_this_thread(L"pg6-audio-out");
        boost_render_thread();
        const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        const bool thread_com = SUCCEEDED(co);
        uint32_t active_rate = device_rate_ ? device_rate_ : kSourceRate;
        reset_declick_state(active_rate);

        bool started = client_ && SUCCEEDED(client_->Start());
        while (running_) {
            if (!client_ || !render_ || !started) {
                // Drain excess buffered audio while endpoint is unavailable so guest never backs up.
                const uint32_t buffered = feed_->buffer().buffered();
                const uint32_t drain_keep = active_rate * kJitterPrebufferMs / 1000;
                if (buffered > drain_keep && !trim_scratch_.empty()) {
                    const uint32_t to_drop =
                        std::min<uint32_t>(buffered - drain_keep, static_cast<uint32_t>(trim_scratch_.size() / 2));
                    if (to_drop > 0) {
                        (void)feed_->buffer().pop(trim_scratch_.data(), to_drop);
                    }
                }
                for (int i = 0; i < 25 && running_; ++i) {
                    Sleep(10);
                }
                if (!running_) break;
                release_interfaces();
                uint32_t new_rate = 0;
                std::string reopen_err;
                if (open_device_interfaces(&new_rate, &reopen_err)) {
                    if (new_rate != active_rate) {
                        active_rate = new_rate;
                        device_rate_ = new_rate;
                        feed_->configure(new_rate, kJitterCapacityMs, kJitterPrebufferMs);
                        trim_scratch_.resize(static_cast<size_t>(new_rate * kJitterCapacityMs / 1000) * 2, 0.0f);
                    }
                    reset_declick_state(active_rate);
                    started = SUCCEEDED(client_->Start());
                }
                continue;
            }

            UINT32 padding = 0;
            HRESULT hr = client_->GetCurrentPadding(&padding);
            if (FAILED(hr)) {
                if (is_device_lost(hr)) {
                    started = false;
                }
                Sleep(5);
                continue;
            }

            // Cap WASAPI write-ahead strictly below JitterBuffer prebuffer (40 ms vs 60 ms) so
            // post-underrun re-priming never drains all buffered frames in one GetBuffer() call.
            const UINT32 target_pad = std::min<UINT32>(
                buffer_frames_, std::max<UINT32>(active_rate * kWasapiTargetPadMs / 1000, 256));
            if (target_pad > padding) {
                const UINT32 want = target_pad - padding;
                BYTE* data = nullptr;
                hr = render_->GetBuffer(want, &data);
                if (SUCCEEDED(hr) && data) {
                    const bool silent = render_frames(reinterpret_cast<float*>(data), want, active_rate);
                    hr = render_->ReleaseBuffer(want, silent ? AUDCLNT_BUFFERFLAGS_SILENT : 0);
                    if (FAILED(hr) && is_device_lost(hr)) {
                        started = false;
                    }
                } else if (FAILED(hr) && is_device_lost(hr)) {
                    started = false;
                }
            }
            wait_period(20);
        }
        unboost_render_thread();
        if (client_) {
            client_->Stop();
            client_->Reset();
        }
        if (thread_com) {
            CoUninitialize();
        }
    }

    IMMDevice* dev_ = nullptr;
    IAudioClient* client_ = nullptr;
    IAudioRenderClient* render_ = nullptr;
    UINT32 buffer_frames_ = 0;
    HANDLE event_ = nullptr;
    bool event_driven_ = false;
    HMODULE avrt_ = nullptr;
    HANDLE mmcss_ = nullptr;
    bool com_inited_ = false;
    DWORD com_inited_thread_id_ = 0;
    uint32_t declick_frames_ = 160;
    float last_l_ = 0.0f;
    float last_r_ = 0.0f;
    bool has_active_audio_ = false;
    bool in_underrun_ = false;
    uint32_t fade_out_pos_ = 0;
    uint32_t fade_out_total_ = 0;
    float fade_out_start_l_ = 0.0f;
    float fade_out_start_r_ = 0.0f;
    uint32_t fade_in_pos_ = 0;
    uint32_t fade_in_total_ = 0;
    float fade_in_from_l_ = 0.0f;
    float fade_in_from_r_ = 0.0f;
    std::vector<float> trim_scratch_;
    float trim_xfade_[kTrimCrossfadeFrames * 2] = {};
    uint32_t trim_xfade_len_ = 0;
#endif
    Feed* feed_ = nullptr;
    std::atomic<bool> running_{false};
    std::thread thread_;
    MmcssMode mmcss_mode_ = MmcssMode::ProAudio;
    std::atomic<bool> mmcss_registered_{false};
    uint32_t device_rate_ = 0;
    uint32_t dev_channels_ = 0;
    uint32_t dev_mask_ = 0;

    bool start(Feed& feed, std::string* error) {
#ifdef _WIN32
        feed_ = &feed;
        std::string err;
        uint32_t rate = 0;
        if (!open_device(&rate, &err)) {
            close_device();
            if (error) *error = err;
            return false;
        }
        feed.configure(rate, kJitterCapacityMs, kJitterPrebufferMs);
        trim_scratch_.assign(static_cast<size_t>(rate * kJitterCapacityMs / 1000) * 2, 0.0f);
        device_rate_ = rate;
        running_ = true;
        thread_ = std::thread([this] { run(); });
        return true;
#else
        (void)feed;
        if (error) *error = "WASAPI is only available on Windows";
        return false;
#endif
    }

    void stop() {
#ifdef _WIN32
        running_ = false;
        if (thread_.joinable()) thread_.join();
        close_device();
#endif
    }
};

Wasapi::Wasapi() : impl_(new Impl) {}
Wasapi::~Wasapi() {
    impl_->stop();
    delete impl_;
}
void Wasapi::set_mmcss_mode(MmcssMode mode) { impl_->mmcss_mode_ = mode; }
MmcssMode Wasapi::mmcss_mode() const { return impl_->mmcss_mode_; }
bool Wasapi::mmcss_registered() const { return impl_->mmcss_registered_.load(); }
bool Wasapi::start(Feed& feed, std::string* error) { return impl_->start(feed, error); }
void Wasapi::stop() { impl_->stop(); }
bool Wasapi::open() const { return impl_->device_rate_ != 0; }
uint32_t Wasapi::device_rate() const { return impl_->device_rate_; }
uint32_t Wasapi::device_channels() const { return impl_->dev_channels_; }
uint32_t Wasapi::device_channel_mask() const { return impl_->dev_mask_; }

}  // namespace playgta6::os::audio_out
