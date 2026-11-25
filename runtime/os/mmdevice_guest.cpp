// The console's default audio render endpoint for guest code: combase!CoCreateInstance(
// CLSID_MMDeviceEnumerator) and the MMDevice / WASAPI / spatial-audio objects behind it.
//
// A console always has one active default render endpoint. The executable's audio device setup
// (0x1443AEA20) creates the enumerator, registers an endpoint notification callback, enumerates
// the active render endpoints (reading PKEY_AudioEndpoint_FormFactor of each when there are
// several), takes the default render endpoint and registers it with its frame tracker
// (0x141E7F580): IAudioClient::GetMixFormat for the channel count, then ISpatialAudioClient
// (format enumerator, IsAudioObjectFormatSupported, ActivateSpatialAudioStream with an event).
// Its audio thread then waits on that event and, per period, calls BeginUpdatingAudioObjects,
// ActivateSpatialAudioObject / GetBuffer for each static object and EndUpdatingAudioObjects.
// Without an endpoint the tracker stays empty, and the tracker's per-frame update (0x145267720)
// returns with its lock held, which stops the title's device frame thread for good.
//
// Model: one endpoint, "{0.0.0.00000000}.{pg6-default-render}", form factor Speakers.
//   IAudioClient: shared-mode mix format 48 kHz float32 stereo (WAVEFORMATEXTENSIBLE), 10 ms
//     period; Initialize/Start/Stop/Reset, SetEventHandle, GetCurrentPadding from a real-time
//     clock, IAudioRenderClient (GetBuffer/ReleaseBuffer) and IAudioClock.
//   ISpatialAudioClient: static objects of 7.1.4 (no dynamic objects), object format 48 kHz
//     float32 mono, 480 frames per pass; the stream's event is set every 10 ms while started.
// Audio data is consumed at real-time rate and discarded natively; set_audio_render_sink()
// receives every pass's PCM (the browser path can forward it to web audio).
// ISpatialAudioObject::SetVolume scales that object's samples from the next EndUpdatingAudioObjects.
//
// PG6_AUDIO_TRACE=1 logs, every 500 spatial passes (5 s), how many object buffers the title filled
// with non-zero samples and its XApu / XDsp command counts. On the start-screen boot the title
// renders two static objects (FrontLeft/FrontRight) per pass and fills them with zeros except
// for ~0.7 s around the first XApu voice: the silence is upstream of this endpoint. The title's
// stream decoders (0x144E64B10) only produce PCM after an XApu type-3 result reporting the bytes
// of float output the APU wrote to the voice's output slot (result +0x34 -> voice +0x334), and
// the XApu model (acp_hal_guest.cpp) neither decodes nor sends such results.

#include "os/audio_downmix.h"
#include "os/audio_report.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett_guest.h"

namespace playgta6::os::scarlett::guest_api {

using com::GuestPtr;
using com::OutPtr;

namespace {

constexpr HResult kOk = com::S_OK_HR;
constexpr HResult kSFalse = 1;
constexpr HResult kENoInterface = static_cast<HResult>(0x80004002u);
constexpr HResult kENotFound = static_cast<HResult>(0x80070490u);       // HRESULT_FROM_WIN32(ERROR_NOT_FOUND)
constexpr HResult kClassNotReg = static_cast<HResult>(0x80040154u);     // REGDB_E_CLASSNOTREG
constexpr HResult kStgAccessDenied = static_cast<HResult>(0x80030005u);
constexpr HResult kNotInitialized = static_cast<HResult>(0x88890001u);  // AUDCLNT_E_NOT_INITIALIZED
constexpr HResult kAlreadyInitialized = static_cast<HResult>(0x88890002u);
constexpr HResult kNotStopped = static_cast<HResult>(0x88890005u);
constexpr HResult kBufferTooLarge = static_cast<HResult>(0x88890006u);
constexpr HResult kOutOfOrder = static_cast<HResult>(0x88890007u);
constexpr HResult kUnsupportedFormat = static_cast<HResult>(0x88890008u);
constexpr HResult kEventHandleNotSet = static_cast<HResult>(0x88890014u);

constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kPeriodFrames = 480;                // 10 ms
constexpr int64_t kPeriodHns = 100000;                 // 10 ms in 100 ns units
constexpr uint32_t kRenderBufferFrames = kPeriodFrames * 2;
constexpr uint32_t kStaticObjectMask = 0x1FFE;         // AudioObjectType FrontLeft .. TopBackRight (7.1.4)
constexpr uint32_t kAnyStaticObject = 0x3FFFE;         // FrontLeft .. BackCenter (not Dynamic)
constexpr uint32_t kDeviceStateActive = 1;
constexpr uint16_t kVtUi4 = 19;
constexpr uint16_t kVtLpwstr = 31;
constexpr uint16_t kVtBlob = 65;

constexpr com::Guid kClsidMMDeviceEnumerator = com::make_guid("bcde0395-e52f-467c-8e3d-c4579291692e");
constexpr com::Guid kIidIMMDeviceEnumerator = com::make_guid("a95664d2-9614-4f35-a746-de8db63617e6");
constexpr com::Guid kIidIMMDeviceCollection = com::make_guid("0bd7a1be-7a1a-44db-8397-cc5392387b5e");
constexpr com::Guid kIidIMMDevice = com::make_guid("d666063f-1587-4e43-81f1-b948e807363f");
constexpr com::Guid kIidIPropertyStore = com::make_guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99");
constexpr com::Guid kIidIAudioClient = com::make_guid("1cb9ad4c-dbfa-4c32-b178-c2f568a703b2");
constexpr com::Guid kIidIAudioRenderClient = com::make_guid("f294acfc-3146-4483-a7bf-addca7c260e2");
constexpr com::Guid kIidIAudioClock = com::make_guid("cd63314f-3fba-4a1b-812c-ef96358728e7");
constexpr com::Guid kIidISpatialAudioClient = com::make_guid("bbf8e066-aaaa-49be-9a4d-fd2a858ea27f");
constexpr com::Guid kIidIAudioFormatEnumerator = com::make_guid("dcdaa858-895a-4a22-a5eb-67bda506096d");
constexpr com::Guid kIidISpatialStreamBase = com::make_guid("feaaf403-c1d8-450d-aa05-e0ccee7502a8");
constexpr com::Guid kIidISpatialStream = com::make_guid("bab5f473-b423-477b-85f5-b5a332a04153");
constexpr com::Guid kIidISpatialObjectBase = com::make_guid("cce0b8f2-8d4d-4efb-a8cf-3d6ecf1c30e0");
constexpr com::Guid kIidISpatialObject = com::make_guid("dde28967-521b-46e5-8f00-bd6f2bc8ab1d");
constexpr com::Guid kPkeyFormFactorFmtid = com::make_guid("1da5d803-d492-4edd-8c23-e0c0ffee7f0e");  // pid 0
constexpr com::Guid kPkeyFriendlyNameFmtid = com::make_guid("a45c254e-df1c-4efd-8020-67d146a850e0");  // pid 14
constexpr com::Guid kSubtypeIeeeFloat = com::make_guid("00000003-0000-0010-8000-00aa00389b71");

constexpr char16_t kEndpointId[] = u"{0.0.0.00000000}.{pg6-default-render}";
constexpr char16_t kFriendlyName[] = u"Speakers";
constexpr uint32_t kFormFactorSpeakers = 1;

struct PropertyKey {
    com::Guid fmtid;
    uint32_t pid;
};
static_assert(sizeof(PropertyKey) == 20);

struct PropVariant {  // PROPVARIANT (x64): vt, three reserved words, then the 16-byte value union
    uint16_t vt;
    uint16_t reserved[3];
    uint64_t value[2];
};
static_assert(sizeof(PropVariant) == 24);

#pragma pack(push, 1)  // mmreg.h packs the wave formats to 1 byte
struct WaveFormatEx {
    uint16_t format_tag;
    uint16_t channels;
    uint32_t samples_per_sec;
    uint32_t avg_bytes_per_sec;
    uint16_t block_align;
    uint16_t bits_per_sample;
    uint16_t cb_size;
};
static_assert(sizeof(WaveFormatEx) == 18);

struct WaveFormatExtensible {
    WaveFormatEx format;
    uint16_t valid_bits_per_sample;
    uint32_t channel_mask;
    com::Guid sub_format;
};
#pragma pack(pop)
static_assert(sizeof(WaveFormatExtensible) == 40);

// SpatialAudioObjectRenderStreamActivationParams (x64)
struct SpatialActivationParams {
    uint64_t object_format;  // const WAVEFORMATEX*
    uint32_t static_object_type_mask;
    uint32_t min_dynamic_object_count;
    uint32_t max_dynamic_object_count;
    uint32_t category;
    uint64_t event_handle;
    uint64_t notify_object;
};
static_assert(sizeof(SpatialActivationParams) == 0x28);

WaveFormatExtensible mix_format() {
    WaveFormatExtensible f{};
    f.format = {0xFFFE, 2, kSampleRate, kSampleRate * 8, 8, 32, 22};
    f.valid_bits_per_sample = 32;
    f.channel_mask = 0x3;  // SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT
    f.sub_format = kSubtypeIeeeFloat;
    return f;
}

WaveFormatEx object_format() { return {3, 1, kSampleRate, kSampleRate * 4, 4, 32, 0}; }

// Guest memory the caller frees with CoTaskMemFree (combase), filled with `bytes`.
uint64_t co_task_copy(const void* bytes, size_t size) {
    const uint64_t va = pg6_win32_co_task_mem_alloc(size);
    if (va != 0 && !com::write_guest(va, bytes, size)) {
        pg6_win32_co_task_mem_free(va);
        return 0;
    }
    return va;
}

bool is_float_mono_48k(uint64_t va) {
    WaveFormatEx f{};
    if (va == 0 || !com::read_guest(va, f)) {
        return false;
    }
    const WaveFormatEx want = object_format();
    return f.format_tag == want.format_tag && f.channels == 1 && f.samples_per_sec == kSampleRate &&
           f.bits_per_sample == 32 && f.block_align == 4;
}

bool is_mix_format(uint64_t va) {
    WaveFormatEx f{};
    if (va == 0 || !com::read_guest(va, f)) {
        return false;
    }
    if (f.channels != 2 || f.samples_per_sec != kSampleRate || f.bits_per_sample != 32 || f.block_align != 8) {
        return false;
    }
    if (f.format_tag == 3) {
        return true;
    }
    WaveFormatExtensible x{};
    return f.format_tag == 0xFFFE && com::read_guest(va, x) && x.sub_format == kSubtypeIeeeFloat;
}

// ---------------------------------------------------------------------------
// The real-time engine: one 10 ms period clock for every started stream
// ---------------------------------------------------------------------------

struct StreamClock {
    std::atomic<bool> started{false};
    std::atomic<uint64_t> event{0};
    std::atomic<uint64_t> frames_played{0};
    std::atomic<uint64_t> periods{0};
};

struct Sink {
    std::mutex mutex;
    AudioRenderSink fn = nullptr;
    void* user = nullptr;
};
Sink& sink() {
    static Sink s;
    return s;
}

class Engine {
public:
    void add(const std::shared_ptr<StreamClock>& clock) {
        std::lock_guard<std::mutex> lock(mutex_);
        clocks_.push_back(clock);
        if (!manual_ && !thread_.joinable()) {
            stop_ = false;
            thread_ = std::thread([this] { run(); });
        }
        cv_.notify_all();
    }

    // One period for every started stream (the thread's step; tests call it directly).
    void tick() {
        std::vector<std::shared_ptr<StreamClock>> live;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            clocks_.erase(std::remove_if(clocks_.begin(), clocks_.end(),
                                         [](const std::weak_ptr<StreamClock>& w) { return w.expired(); }),
                          clocks_.end());
            for (const auto& w : clocks_) {
                if (auto c = w.lock()) {
                    live.push_back(std::move(c));
                }
            }
        }
        audio_downmix::flush_period(kPeriodFrames);
        for (const auto& c : live) {
            if (!c->started.load()) {
                continue;
            }
            c->frames_played.fetch_add(kPeriodFrames);
            c->periods.fetch_add(1);
            if (const uint64_t ev = c->event.load(); ev != 0) {
                win32::SetEvent(static_cast<win32::Handle>(ev));
            }
        }
    }

    void set_manual(bool manual) {
        std::thread old;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            manual_ = manual;
            if (manual && thread_.joinable()) {
                stop_ = true;
                cv_.notify_all();
                old = std::move(thread_);
            }
        }
        if (old.joinable()) {
            old.join();
        }
    }

    void reset() {
        set_manual(true);
        std::lock_guard<std::mutex> lock(mutex_);
        clocks_.clear();
        manual_ = false;
    }

private:
    void run() {
        auto next = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_) {
            next += std::chrono::microseconds(kPeriodHns / 10);
            cv_.wait_until(lock, next, [this] { return stop_; });
            if (stop_) {
                break;
            }
            lock.unlock();
            tick();
            lock.lock();
            const auto now = std::chrono::steady_clock::now();
            if (now - next > std::chrono::milliseconds(100)) {
                next = now;  // a long stall: do not burst the missed periods
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::weak_ptr<StreamClock>> clocks_;
    std::thread thread_;
    bool stop_ = false;
    bool manual_ = false;
};
Engine& engine() {
    static Engine e;
    return e;
}

std::atomic<uint64_t> g_endpoint_activations{0};
std::atomic<bool> g_endpoint_enabled{true};
std::atomic<uint64_t> g_spatial_passes{0};
std::atomic<uint64_t> g_render_frames{0};
std::atomic<uint64_t> g_spatial_object_buffers{0};   // object buffers read at EndUpdatingAudioObjects
std::atomic<uint64_t> g_spatial_nonzero_buffers{0};  // ... of them holding a non-zero sample
std::atomic<uint64_t> g_spatial_nonzero_passes{0};   // passes with any non-zero object buffer
std::atomic<uint64_t> g_render_nonzero_frames{0};
std::atomic<uint64_t> g_render_silent_releases{0};   // ReleaseBuffer with AUDCLNT_BUFFERFLAGS_SILENT

bool any_nonzero(const float* p, size_t n, float* peak) {
    bool nz = false;
    for (size_t i = 0; i < n; ++i) {
        const float a = p[i] < 0 ? -p[i] : p[i];
        if (a > 0.0f) {
            nz = true;
            if (peak != nullptr && a > *peak) {
                *peak = a;
            }
        }
    }
    return nz;
}

// PG6_AUDIO_TRACE=1: one diagnostic line per 500 spatial passes (5 s) with what the title wrote.
void trace_window(uint32_t objects_this_pass, float pass_peak) {
    static const bool on = [] {
        const char* e = std::getenv("PG6_AUDIO_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    if (!on) {
        return;
    }
    static std::mutex m;
    static uint64_t passes = 0, objects = 0, nz_last = 0, rf_last = 0, rnz_last = 0, xapu_last = 0, xdsp_last = 0;
    static float peak = 0.0f;
    std::lock_guard<std::mutex> lock(m);
    ++passes;
    objects += objects_this_pass;
    peak = std::max(peak, pass_peak);
    if (passes % 500 != 0) {
        return;
    }
    const uint64_t nz = g_spatial_nonzero_buffers.load(), rf = g_render_frames.load(),
                   rnz = g_render_nonzero_frames.load(), xa = xapu_stats().commands, xd = xdsp_stats().commands;
    char line[320];
    std::snprintf(line, sizeof(line),
                  "audio-trace: passes %llu: objects/pass %.2f, non-zero object buffers %llu, peak %.5f; render "
                  "frames %llu (non-zero %llu, silent releases %llu); xapu commands +%llu, xdsp commands +%llu",
                  static_cast<unsigned long long>(passes), static_cast<double>(objects) / 500.0,
                  static_cast<unsigned long long>(nz - nz_last), static_cast<double>(peak),
                  static_cast<unsigned long long>(rf - rf_last), static_cast<unsigned long long>(rnz - rnz_last),
                  static_cast<unsigned long long>(g_render_silent_releases.load()),
                  static_cast<unsigned long long>(xa - xapu_last), static_cast<unsigned long long>(xd - xdsp_last));
    com::detail::diagnostic(line);
    objects = 0;
    peak = 0.0f;
    nz_last = nz;
    rf_last = rf;
    rnz_last = rnz;
    xapu_last = xa;
    xdsp_last = xd;
}

void forward_samples(uint32_t object_type, const float* pcm, uint32_t frames, uint32_t channels) {
    Sink& s = sink();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.fn != nullptr && frames != 0) {
        s.fn(s.user, object_type, pcm, frames, channels, kSampleRate);
    }
}

void forward_pcm(uint32_t object_type, uint64_t buffer, uint32_t frames, uint32_t channels) {
    Sink& s = sink();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.fn == nullptr || buffer == 0 || frames == 0) {
        return;
    }
    std::vector<float> pcm(static_cast<size_t>(frames) * channels);
    if (com::read_guest(buffer, pcm.data(), pcm.size() * sizeof(float))) {
        s.fn(s.user, object_type, pcm.data(), frames, channels, kSampleRate);
    }
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

class PropertyStoreObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.IPropertyStore";

    HResult GetCount(GuestPtr<uint32_t> count) {  // 3
        return count.write(2u) ? kOk : com::E_POINTER_HR;
    }
    HResult GetAt(uint32_t index, GuestPtr<PropertyKey> key) {  // 4
        if (key.is_null()) {
            return com::E_POINTER_HR;
        }
        if (index > 1) {
            return com::E_INVALIDARG_HR;
        }
        key.write(index == 0 ? PropertyKey{kPkeyFormFactorFmtid, 0} : PropertyKey{kPkeyFriendlyNameFmtid, 14});
        return kOk;
    }
    HResult GetValue(const PropertyKey& key, GuestPtr<PropVariant> value) {  // 5
        if (value.is_null()) {
            return com::E_POINTER_HR;
        }
        PropVariant v{};
        if (key.fmtid == kPkeyFormFactorFmtid && key.pid == 0) {
            v.vt = kVtUi4;
            v.value[0] = kFormFactorSpeakers;
        } else if (key.fmtid == kPkeyFriendlyNameFmtid && key.pid == 14) {
            const uint64_t text = co_task_copy(kFriendlyName, sizeof(kFriendlyName));
            if (text == 0) {
                return com::E_OUTOFMEMORY_HR;
            }
            v.vt = kVtLpwstr;
            v.value[0] = text;
        }
        return value.write(v) ? kOk : com::E_POINTER_HR;  // unknown keys: VT_EMPTY
    }
    HResult SetValue(uint64_t /*key*/, uint64_t /*value*/) { return kStgAccessDenied; }  // 6
    HResult Commit() { return kStgAccessDenied; }                                       // 7
};

class AudioClientObj;

class RenderClientObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.IAudioRenderClient";
    explicit RenderClientObj(com::Ref<AudioClientObj> client) : client_(std::move(client)) {}

    HResult GetBuffer(uint32_t frames, OutPtr data);   // 3
    HResult ReleaseBuffer(uint32_t frames, uint32_t flags);  // 4

private:
    com::Ref<AudioClientObj> client_;
};

class AudioClockObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.IAudioClock";
    explicit AudioClockObj(std::shared_ptr<StreamClock> clock) : clock_(std::move(clock)) {}

    HResult GetFrequency(GuestPtr<uint64_t> frequency) {  // 3: position in bytes per second
        return frequency.write(uint64_t{kSampleRate} * 8) ? kOk : com::E_POINTER_HR;
    }
    HResult GetPosition(GuestPtr<uint64_t> position, GuestPtr<uint64_t> qpc) {  // 4
        if (position.is_null()) {
            return com::E_POINTER_HR;
        }
        position.write(clock_->frames_played.load() * 8);
        if (!qpc.is_null()) {
            const auto t = std::chrono::steady_clock::now().time_since_epoch();
            qpc.write(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 100));
        }
        return kOk;
    }
    HResult GetCharacteristics(GuestPtr<uint32_t> characteristics) {  // 5
        return characteristics.write(0u) ? kOk : com::E_POINTER_HR;
    }

private:
    std::shared_ptr<StreamClock> clock_;
};

class AudioClientObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.IAudioClient";
    AudioClientObj() : clock_(std::make_shared<StreamClock>()) {}

    // 3 Initialize(share mode, stream flags, buffer duration, periodicity, format, session)
    HResult Initialize(uint32_t share_mode, uint32_t flags, int64_t /*duration*/, int64_t /*periodicity*/,
                       uint64_t format, uint64_t /*session*/) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            return kAlreadyInitialized;
        }
        if (format == 0) {
            return com::E_POINTER_HR;
        }
        if (share_mode != 0 || !is_mix_format(format)) {  // shared mode, the mix format only
            return kUnsupportedFormat;
        }
        initialized_ = true;
        event_driven_ = (flags & 0x00040000u) != 0;  // AUDCLNT_STREAMFLAGS_EVENTCALLBACK
        engine().add(clock_);
        return kOk;
    }
    HResult GetBufferSize(GuestPtr<uint32_t> frames) {  // 4
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return kNotInitialized;
        }
        return frames.write(kRenderBufferFrames) ? kOk : com::E_POINTER_HR;
    }
    HResult GetStreamLatency(GuestPtr<int64_t> latency) {  // 5
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return kNotInitialized;
        }
        return latency.write(kPeriodHns) ? kOk : com::E_POINTER_HR;
    }
    HResult GetCurrentPadding(GuestPtr<uint32_t> padding) {  // 6
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return kNotInitialized;
        }
        return padding.write(padding_locked()) ? kOk : com::E_POINTER_HR;
    }
    HResult IsFormatSupported(uint32_t share_mode, uint64_t format, OutPtr closest) {  // 7
        if (!closest.is_null()) {
            closest.write(0);
        }
        if (format == 0) {
            return com::E_POINTER_HR;
        }
        if (share_mode == 0 && is_mix_format(format)) {
            return kOk;
        }
        if (share_mode == 0 && !closest.is_null()) {
            const WaveFormatExtensible f = mix_format();
            closest.write(co_task_copy(&f, sizeof(f)));
            return kSFalse;
        }
        return kUnsupportedFormat;
    }
    HResult GetMixFormat(OutPtr format) {  // 8
        if (format.is_null()) {
            return com::E_POINTER_HR;
        }
        const WaveFormatExtensible f = mix_format();
        const uint64_t va = co_task_copy(&f, sizeof(f));
        format.write(va);
        return va != 0 ? kOk : com::E_OUTOFMEMORY_HR;
    }
    HResult GetDevicePeriod(GuestPtr<int64_t> default_period, GuestPtr<int64_t> minimum_period) {  // 9
        if (default_period.is_null() && minimum_period.is_null()) {
            return com::E_POINTER_HR;
        }
        default_period.write(kPeriodHns);
        minimum_period.write(kPeriodHns / 3);
        return kOk;
    }
    HResult Start() {  // 10
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return kNotInitialized;
        }
        if (event_driven_ && clock_->event.load() == 0) {
            return kEventHandleNotSet;
        }
        if (clock_->started.load()) {
            return kNotStopped;
        }
        clock_->started = true;
        return kOk;
    }
    HResult Stop() {  // 11
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return kNotInitialized;
        }
        const bool was = clock_->started.exchange(false);
        return was ? kOk : kSFalse;
    }
    HResult Reset() {  // 12
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return kNotInitialized;
        }
        if (clock_->started.load()) {
            return kNotStopped;
        }
        written_ = 0;
        clock_->frames_played = 0;
        return kOk;
    }
    HResult SetEventHandle(uint64_t event) {  // 13
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return kNotInitialized;
        }
        if (event == 0) {
            return com::E_INVALIDARG_HR;
        }
        clock_->event = event;
        return kOk;
    }
    HResult GetService(const com::Guid& iid, OutPtr out);  // 14

    // Render client side (under the client's lock).
    HResult get_buffer(uint32_t frames, OutPtr data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data.is_null()) {
            return com::E_POINTER_HR;
        }
        data.write(0);
        if (pending_ != 0) {
            return kOutOfOrder;
        }
        if (frames > kRenderBufferFrames - padding_locked()) {
            return kBufferTooLarge;
        }
        if (frames == 0) {
            return kOk;
        }
        if (buffer_ == 0) {
            buffer_ = com::guest_alloc(uint64_t{kRenderBufferFrames} * 8);
            if (buffer_ == 0) {
                return com::E_OUTOFMEMORY_HR;
            }
        }
        pending_ = frames;
        data.write(buffer_);
        return kOk;
    }
    HResult release_buffer(uint32_t frames, uint32_t flags) {
        uint64_t buffer = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (frames > pending_ || (frames != 0 && pending_ == 0)) {
                return com::E_INVALIDARG_HR;
            }
            pending_ = 0;
            written_ += frames;
            buffer = (flags & 0x2u) != 0 ? 0 : buffer_;  // AUDCLNT_BUFFERFLAGS_SILENT
        }
        if ((flags & 0x2u) != 0) {
            g_render_silent_releases.fetch_add(1);
        }
        g_render_frames.fetch_add(frames);
        forward_pcm(0, buffer, frames, 2);
        if (buffer != 0 && frames != 0) {
            std::vector<float> stereo(static_cast<size_t>(frames) * 2);
            if (com::read_guest(buffer, stereo.data(), stereo.size() * sizeof(float))) {
                const bool nonzero = any_nonzero(stereo.data(), stereo.size(), nullptr);
                if (nonzero) {
                    g_render_nonzero_frames.fetch_add(frames);
                }
                audio_report::windows().record(frames, nonzero);
                audio_downmix::push_render_stereo(stereo.data(), frames);
            }
        }
        return kOk;
    }

    ~AudioClientObj() override {
        if (buffer_ != 0) {
            com::guest_free(buffer_);
        }
    }

private:
    uint32_t padding_locked() const {
        const uint64_t played = clock_->frames_played.load();
        return written_ > played ? static_cast<uint32_t>(std::min<uint64_t>(written_ - played, kRenderBufferFrames))
                                 : 0u;
    }

    std::mutex mutex_;
    std::shared_ptr<StreamClock> clock_;
    bool initialized_ = false;
    bool event_driven_ = false;
    uint64_t written_ = 0;
    uint32_t pending_ = 0;
    uint64_t buffer_ = 0;
};

HResult RenderClientObj::GetBuffer(uint32_t frames, OutPtr data) { return client_->get_buffer(frames, data); }
HResult RenderClientObj::ReleaseBuffer(uint32_t frames, uint32_t flags) {
    return client_->release_buffer(frames, flags);
}

class FormatEnumeratorObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.IAudioFormatEnumerator";
    FormatEnumeratorObj() {
        format_ = com::guest_alloc(sizeof(WaveFormatEx) + 6);
        if (format_ != 0) {
            com::write_guest(format_, object_format());
        }
    }
    ~FormatEnumeratorObj() override {
        if (format_ != 0) {
            com::guest_free(format_);
        }
    }
    HResult GetCount(GuestPtr<uint32_t> count) {  // 3
        return count.write(1u) ? kOk : com::E_POINTER_HR;
    }
    HResult GetFormat(uint32_t index, OutPtr format) {  // 4: the pointer stays owned by the enumerator
        if (format.is_null()) {
            return com::E_POINTER_HR;
        }
        if (index != 0 || format_ == 0) {
            format.write(0);
            return com::E_INVALIDARG_HR;
        }
        format.write(format_);
        return kOk;
    }

private:
    uint64_t format_ = 0;
};

class SpatialStreamObj;

class SpatialObjectObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.ISpatialAudioObject";
    SpatialObjectObj(SpatialStreamObj* stream, uint32_t type) : stream_(stream), type_(type) {}

    HResult GetBuffer(OutPtr buffer, GuestPtr<uint32_t> length);  // 3
    HResult SetEndOfStream(uint32_t /*frames*/) {                 // 4
        active_ = false;
        return kOk;
    }
    HResult IsActive(GuestPtr<uint32_t> active) {  // 5 (BOOL)
        return active.write(active_ ? 1u : 0u) ? kOk : com::E_POINTER_HR;
    }
    HResult GetAudioObjectType(GuestPtr<uint32_t> type) {  // 6
        return type.write(type_) ? kOk : com::E_POINTER_HR;
    }
    HResult SetPosition(float /*x*/, float /*y*/, float /*z*/) { return kOk; }  // 7 (dynamic objects only)
    // 8: an amplitude multiplier on the object's audio, applied from the pass that ends next.
    HResult SetVolume(float volume) {
        if (!(volume >= 0.0f && volume <= 1.0f)) {
            return com::E_INVALIDARG_HR;
        }
        volume_ = volume;
        return kOk;
    }

    uint32_t type() const { return type_; }
    bool active() const { return active_; }
    float volume() const { return volume_; }
    void detach() { stream_ = nullptr; }

private:
    SpatialStreamObj* stream_;  // the stream owns its objects (cleared before it goes away)
    uint32_t type_;
    std::atomic<bool> active_{true};
    std::atomic<float> volume_{1.0f};
};

class SpatialStreamObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.ISpatialAudioObjectRenderStream";
    SpatialStreamObj(uint32_t static_mask, uint64_t event) : clock_(std::make_shared<StreamClock>()), mask_(static_mask) {
        clock_->event = event;
        engine().add(clock_);
    }
    ~SpatialStreamObj() override {
        for (auto& o : objects_) {
            o.object->detach();
            if (o.buffer != 0) {
                com::guest_free(o.buffer);
            }
        }
    }

    HResult GetAvailableDynamicObjectCount(GuestPtr<uint32_t> count) {  // 3
        return count.write(0u) ? kOk : com::E_POINTER_HR;
    }
    HResult GetService(const com::Guid& /*iid*/, OutPtr out) {  // 4
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        out.write(0);
        return kENoInterface;
    }
    HResult Start() {  // 5
        clock_->started = true;
        return kOk;
    }
    HResult Stop() {  // 6
        clock_->started = false;
        return kOk;
    }
    HResult Reset() {  // 7
        std::lock_guard<std::mutex> lock(mutex_);
        clock_->frames_played = 0;
        return kOk;
    }
    HResult BeginUpdatingAudioObjects(GuestPtr<uint32_t> available_dynamic, GuestPtr<uint32_t> frames) {  // 8
        std::lock_guard<std::mutex> lock(mutex_);
        if (updating_) {
            return kOutOfOrder;
        }
        updating_ = true;
        available_dynamic.write(0u);
        frames.write(kPeriodFrames);
        return kOk;
    }
    HResult EndUpdatingAudioObjects() {  // 9
        struct Written {
            uint32_t type;
            uint64_t buffer;
            float volume;
        };
        std::vector<Written> written;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!updating_) {
                return kOutOfOrder;
            }
            updating_ = false;
            for (auto& o : objects_) {
                if (o.handed_out && o.object->active()) {
                    written.push_back({o.object->type(), o.buffer, o.object->volume()});
                }
                o.handed_out = false;
            }
        }
        g_spatial_passes.fetch_add(1);
        std::vector<uint32_t> types;
        std::vector<std::vector<float>> pcm;
        float pass_peak = 0.0f;
        bool pass_nonzero = false;
        for (const auto& [type, buffer, volume] : written) {
            std::vector<float> mono(kPeriodFrames);
            if (buffer != 0 && com::read_guest(buffer, mono.data(), mono.size() * sizeof(float))) {
                if (volume != 1.0f) {
                    for (float& v : mono) {
                        v *= volume;
                    }
                }
                forward_samples(type, mono.data(), kPeriodFrames, 1);
                g_spatial_object_buffers.fetch_add(1);
                if (any_nonzero(mono.data(), mono.size(), &pass_peak)) {
                    g_spatial_nonzero_buffers.fetch_add(1);
                    pass_nonzero = true;
                }
                types.push_back(type);
                pcm.push_back(std::move(mono));
            }
        }
        // The report's non-silent fraction and windows count spatial passes like render frames:
        // a title that only renders through static objects otherwise reports 0 frames.
        if (pass_nonzero) {
            g_spatial_nonzero_passes.fetch_add(1);
        }
        audio_report::windows().record(kPeriodFrames, pass_nonzero);
        std::vector<const float*> ptrs;
        for (const auto& v : pcm) {
            ptrs.push_back(v.data());
        }
        audio_downmix::push_spatial_pass(types.data(), ptrs.data(), static_cast<uint32_t>(types.size()),
                                         kPeriodFrames);
        trace_window(static_cast<uint32_t>(types.size()), pass_peak);
        return kOk;
    }
    HResult ActivateSpatialAudioObject(uint32_t type, OutPtr out);  // 10

    HResult object_buffer(const SpatialObjectObj* object, OutPtr buffer, GuestPtr<uint32_t> length) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer.is_null() || length.is_null()) {
            return com::E_POINTER_HR;
        }
        buffer.write(0);
        length.write(0u);
        if (!updating_) {
            return kOutOfOrder;
        }
        for (auto& o : objects_) {
            if (o.object.get() != object) {
                continue;
            }
            if (o.buffer == 0) {
                o.buffer = com::guest_alloc(uint64_t{kPeriodFrames} * 4);
                if (o.buffer == 0) {
                    return com::E_OUTOFMEMORY_HR;
                }
            }
            o.handed_out = true;
            buffer.write(o.buffer);
            length.write(kPeriodFrames * 4);
            return kOk;
        }
        return com::E_INVALIDARG_HR;
    }

    const std::shared_ptr<StreamClock>& clock() const { return clock_; }

private:
    struct Object {
        com::Ref<SpatialObjectObj> object;
        uint64_t buffer = 0;
        bool handed_out = false;
    };
    std::mutex mutex_;
    std::shared_ptr<StreamClock> clock_;
    uint32_t mask_;
    bool updating_ = false;
    std::vector<Object> objects_;
};

HResult SpatialObjectObj::GetBuffer(OutPtr buffer, GuestPtr<uint32_t> length) {
    if (stream_ == nullptr) {
        return kOutOfOrder;
    }
    return stream_->object_buffer(this, buffer, length);
}

using PSM = com::Methods<PropertyStoreObj>;
constexpr com::MethodDesc kPropertyStoreMethods[] = {
    PSM::method<&PropertyStoreObj::GetCount>("GetCount"), PSM::method<&PropertyStoreObj::GetAt>("GetAt"),
    PSM::method<&PropertyStoreObj::GetValue>("GetValue"), PSM::method<&PropertyStoreObj::SetValue>("SetValue"),
    PSM::method<&PropertyStoreObj::Commit>("Commit"),
};
constexpr auto kIPropertyStore =
    com::declare_interface<PropertyStoreObj>("IPropertyStore", kIidIPropertyStore, kPropertyStoreMethods);

using RCM = com::Methods<RenderClientObj>;
constexpr com::MethodDesc kRenderClientMethods[] = {
    RCM::method<&RenderClientObj::GetBuffer>("GetBuffer"),
    RCM::method<&RenderClientObj::ReleaseBuffer>("ReleaseBuffer"),
};
constexpr auto kIAudioRenderClient =
    com::declare_interface<RenderClientObj>("IAudioRenderClient", kIidIAudioRenderClient, kRenderClientMethods);

using ACM = com::Methods<AudioClockObj>;
constexpr com::MethodDesc kAudioClockMethods[] = {
    ACM::method<&AudioClockObj::GetFrequency>("GetFrequency"),
    ACM::method<&AudioClockObj::GetPosition>("GetPosition"),
    ACM::method<&AudioClockObj::GetCharacteristics>("GetCharacteristics"),
};
constexpr auto kIAudioClock = com::declare_interface<AudioClockObj>("IAudioClock", kIidIAudioClock, kAudioClockMethods);

using AUM = com::Methods<AudioClientObj>;
constexpr com::MethodDesc kAudioClientMethods[] = {
    AUM::method<&AudioClientObj::Initialize>("Initialize"),
    AUM::method<&AudioClientObj::GetBufferSize>("GetBufferSize"),
    AUM::method<&AudioClientObj::GetStreamLatency>("GetStreamLatency"),
    AUM::method<&AudioClientObj::GetCurrentPadding>("GetCurrentPadding"),
    AUM::method<&AudioClientObj::IsFormatSupported>("IsFormatSupported"),
    AUM::method<&AudioClientObj::GetMixFormat>("GetMixFormat"),
    AUM::method<&AudioClientObj::GetDevicePeriod>("GetDevicePeriod"),
    AUM::method<&AudioClientObj::Start>("Start"),
    AUM::method<&AudioClientObj::Stop>("Stop"),
    AUM::method<&AudioClientObj::Reset>("Reset"),
    AUM::method<&AudioClientObj::SetEventHandle>("SetEventHandle"),
    AUM::method<&AudioClientObj::GetService>("GetService"),
};
constexpr auto kIAudioClient = com::declare_interface<AudioClientObj>("IAudioClient", kIidIAudioClient, kAudioClientMethods);

HResult AudioClientObj::GetService(const com::Guid& iid, OutPtr out) {
    if (out.is_null()) {
        return com::E_POINTER_HR;
    }
    out.write(0);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return kNotInitialized;
        }
    }
    if (iid == kIidIAudioRenderClient) {
        com::Ref<RenderClientObj> rc = com::create_object<RenderClientObj>(
            com::implements(kIAudioRenderClient), com::Ref<AudioClientObj>::share(this));
        if (!rc) {
            return com::E_OUTOFMEMORY_HR;
        }
        out.write(rc.detach(kIAudioRenderClient));
        return kOk;
    }
    if (iid == kIidIAudioClock) {
        com::Ref<AudioClockObj> ck = com::create_object<AudioClockObj>(com::implements(kIAudioClock), clock_);
        if (!ck) {
            return com::E_OUTOFMEMORY_HR;
        }
        out.write(ck.detach(kIAudioClock));
        return kOk;
    }
    return kENoInterface;
}

using FEM = com::Methods<FormatEnumeratorObj>;
constexpr com::MethodDesc kFormatEnumeratorMethods[] = {
    FEM::method<&FormatEnumeratorObj::GetCount>("GetCount"),
    FEM::method<&FormatEnumeratorObj::GetFormat>("GetFormat"),
};
constexpr auto kIAudioFormatEnumerator = com::declare_interface<FormatEnumeratorObj>(
    "IAudioFormatEnumerator", kIidIAudioFormatEnumerator, kFormatEnumeratorMethods);

constexpr com::Guid kSpatialObjectBases[] = {kIidISpatialObjectBase};
using SOM = com::Methods<SpatialObjectObj>;
constexpr com::MethodDesc kSpatialObjectMethods[] = {
    SOM::method<&SpatialObjectObj::GetBuffer>("GetBuffer"),
    SOM::method<&SpatialObjectObj::SetEndOfStream>("SetEndOfStream"),
    SOM::method<&SpatialObjectObj::IsActive>("IsActive"),
    SOM::method<&SpatialObjectObj::GetAudioObjectType>("GetAudioObjectType"),
    SOM::method<&SpatialObjectObj::SetPosition>("SetPosition"),
    SOM::method<&SpatialObjectObj::SetVolume>("SetVolume"),
};
constexpr auto kISpatialAudioObject = com::declare_interface<SpatialObjectObj>(
    "ISpatialAudioObject", kIidISpatialObject, kSpatialObjectMethods, {.extra_iids = kSpatialObjectBases});

constexpr com::Guid kSpatialStreamBases[] = {kIidISpatialStreamBase};
using SSM = com::Methods<SpatialStreamObj>;
constexpr com::MethodDesc kSpatialStreamMethods[] = {
    SSM::method<&SpatialStreamObj::GetAvailableDynamicObjectCount>("GetAvailableDynamicObjectCount"),
    SSM::method<&SpatialStreamObj::GetService>("GetService"),
    SSM::method<&SpatialStreamObj::Start>("Start"),
    SSM::method<&SpatialStreamObj::Stop>("Stop"),
    SSM::method<&SpatialStreamObj::Reset>("Reset"),
    SSM::method<&SpatialStreamObj::BeginUpdatingAudioObjects>("BeginUpdatingAudioObjects"),
    SSM::method<&SpatialStreamObj::EndUpdatingAudioObjects>("EndUpdatingAudioObjects"),
    SSM::method<&SpatialStreamObj::ActivateSpatialAudioObject>("ActivateSpatialAudioObject"),
};
constexpr auto kISpatialStream = com::declare_interface<SpatialStreamObj>(
    "ISpatialAudioObjectRenderStream", kIidISpatialStream, kSpatialStreamMethods, {.extra_iids = kSpatialStreamBases});

HResult SpatialStreamObj::ActivateSpatialAudioObject(uint32_t type, OutPtr out) {
    if (out.is_null()) {
        return com::E_POINTER_HR;
    }
    out.write(0);
    // One static object per channel bit of the stream's mask; dynamic objects are not offered.
    if (type == 0 || (type & (type - 1)) != 0 || (type & mask_) == 0) {
        return com::E_INVALIDARG_HR;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& o : objects_) {
        if (o.object->type() == type && o.object->active()) {
            return com::E_INVALIDARG_HR;  // already active
        }
    }
    // Objects that reached their end of stream are done: forget them and their buffers.
    objects_.erase(std::remove_if(objects_.begin(), objects_.end(),
                                  [](Object& o) {
                                      if (o.object->active()) {
                                          return false;
                                      }
                                      o.object->detach();
                                      if (o.buffer != 0) {
                                          com::guest_free(o.buffer);
                                      }
                                      return true;
                                  }),
                   objects_.end());
    com::Ref<SpatialObjectObj> obj = com::create_object<SpatialObjectObj>(com::implements(kISpatialAudioObject), this, type);
    if (!obj) {
        return com::E_OUTOFMEMORY_HR;
    }
    objects_.push_back({obj, 0, false});
    out.write(obj.detach(kISpatialAudioObject));
    return kOk;
}

class SpatialClientObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.ISpatialAudioClient";

    HResult GetStaticObjectPosition(uint32_t type, GuestPtr<float> x, GuestPtr<float> y, GuestPtr<float> z) {  // 3
        if (x.is_null() || y.is_null() || z.is_null()) {
            return com::E_POINTER_HR;
        }
        if ((type & kStaticObjectMask) == 0 || (type & (type - 1)) != 0) {
            return com::E_INVALIDARG_HR;
        }
        // Unit-sphere positions of the 7.1.4 bed (right-handed: x right, y up, -z front).
        struct Pos {
            uint32_t type;
            float x, y, z;
        };
        static constexpr Pos kPos[] = {
            {0x2, -0.5f, 0.0f, -0.866f},  {0x4, 0.5f, 0.0f, -0.866f},   {0x8, 0.0f, 0.0f, -1.0f},
            {0x10, 0.0f, 0.0f, 0.0f},     {0x20, -1.0f, 0.0f, 0.0f},    {0x40, 1.0f, 0.0f, 0.0f},
            {0x80, -0.5f, 0.0f, 0.866f},  {0x100, 0.5f, 0.0f, 0.866f},  {0x200, -0.5f, 0.707f, -0.5f},
            {0x400, 0.5f, 0.707f, -0.5f}, {0x800, -0.5f, 0.707f, 0.5f}, {0x1000, 0.5f, 0.707f, 0.5f},
        };
        for (const Pos& p : kPos) {
            if (p.type == type) {
                x.write(p.x);
                y.write(p.y);
                z.write(p.z);
                return kOk;
            }
        }
        return com::E_INVALIDARG_HR;
    }
    HResult GetNativeStaticObjectTypeMask(GuestPtr<uint32_t> mask) {  // 4
        return mask.write(kStaticObjectMask) ? kOk : com::E_POINTER_HR;
    }
    HResult GetMaxDynamicObjectCount(GuestPtr<uint32_t> count) {  // 5
        return count.write(0u) ? kOk : com::E_POINTER_HR;
    }
    HResult GetSupportedAudioObjectFormatEnumerator(OutPtr out) {  // 6
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        com::Ref<FormatEnumeratorObj> e = com::create_object<FormatEnumeratorObj>(com::implements(kIAudioFormatEnumerator));
        if (!e) {
            out.write(0);
            return com::E_OUTOFMEMORY_HR;
        }
        out.write(e.detach(kIAudioFormatEnumerator));
        return kOk;
    }
    HResult GetMaxFrameCount(uint64_t format, GuestPtr<uint32_t> frames) {  // 7
        if (frames.is_null()) {
            return com::E_POINTER_HR;
        }
        if (!is_float_mono_48k(format)) {
            return kUnsupportedFormat;
        }
        return frames.write(kPeriodFrames) ? kOk : com::E_POINTER_HR;
    }
    HResult IsAudioObjectFormatSupported(uint64_t format) {  // 8
        if (format == 0) {
            return com::E_POINTER_HR;
        }
        return is_float_mono_48k(format) ? kOk : kUnsupportedFormat;
    }
    HResult IsSpatialAudioStreamAvailable(const com::Guid& iid, uint64_t /*params*/) {  // 9
        return iid == kIidISpatialStream || iid == kIidISpatialStreamBase ? kOk : kENoInterface;
    }
    // 10 ActivateSpatialAudioStream(const PROPVARIANT* activation params, REFIID, void**)
    HResult ActivateSpatialAudioStream(GuestPtr<PropVariant> params, const com::Guid& iid, OutPtr out) {
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        out.write(0);
        if (!(iid == kIidISpatialStream || iid == kIidISpatialStreamBase)) {
            return kENoInterface;
        }
        PropVariant pv{};
        if (!params.read(pv) || pv.vt != kVtBlob || static_cast<uint32_t>(pv.value[0]) < sizeof(SpatialActivationParams)) {
            return com::E_INVALIDARG_HR;
        }
        SpatialActivationParams p{};
        if (!com::read_guest(pv.value[1], p)) {
            return com::E_INVALIDARG_HR;
        }
        if (!is_float_mono_48k(p.object_format)) {
            return kUnsupportedFormat;
        }
        if ((p.static_object_type_mask & ~kAnyStaticObject) != 0 || p.max_dynamic_object_count != 0 ||
            p.min_dynamic_object_count != 0) {
            return com::E_INVALIDARG_HR;
        }
        if (p.event_handle == 0) {
            return kEventHandleNotSet;
        }
        com::Ref<SpatialStreamObj> s = com::create_object<SpatialStreamObj>(
            com::implements(kISpatialStream), p.static_object_type_mask, p.event_handle);
        if (!s) {
            return com::E_OUTOFMEMORY_HR;
        }
        out.write(s.detach(kISpatialStream));
        return kOk;
    }
};

using SCM = com::Methods<SpatialClientObj>;
constexpr com::MethodDesc kSpatialClientMethods[] = {
    SCM::method<&SpatialClientObj::GetStaticObjectPosition>("GetStaticObjectPosition"),
    SCM::method<&SpatialClientObj::GetNativeStaticObjectTypeMask>("GetNativeStaticObjectTypeMask"),
    SCM::method<&SpatialClientObj::GetMaxDynamicObjectCount>("GetMaxDynamicObjectCount"),
    SCM::method<&SpatialClientObj::GetSupportedAudioObjectFormatEnumerator>("GetSupportedAudioObjectFormatEnumerator"),
    SCM::method<&SpatialClientObj::GetMaxFrameCount>("GetMaxFrameCount"),
    SCM::method<&SpatialClientObj::IsAudioObjectFormatSupported>("IsAudioObjectFormatSupported"),
    SCM::method<&SpatialClientObj::IsSpatialAudioStreamAvailable>("IsSpatialAudioStreamAvailable"),
    SCM::method<&SpatialClientObj::ActivateSpatialAudioStream>("ActivateSpatialAudioStream"),
};
constexpr auto kISpatialAudioClient =
    com::declare_interface<SpatialClientObj>("ISpatialAudioClient", kIidISpatialAudioClient, kSpatialClientMethods);

class DeviceObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.IMMDevice";

    // 3 Activate(REFIID, DWORD clsctx, PROPVARIANT* params, void** out)
    HResult Activate(const com::Guid& iid, uint32_t /*clsctx*/, uint64_t /*params*/, OutPtr out) {
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        out.write(0);
        if (iid == kIidIAudioClient) {
            com::Ref<AudioClientObj> c = com::create_object<AudioClientObj>(com::implements(kIAudioClient));
            if (!c) {
                return com::E_OUTOFMEMORY_HR;
            }
            g_endpoint_activations.fetch_add(1);
            out.write(c.detach(kIAudioClient));
            return kOk;
        }
        if (iid == kIidISpatialAudioClient) {
            com::Ref<SpatialClientObj> c = com::create_object<SpatialClientObj>(com::implements(kISpatialAudioClient));
            if (!c) {
                return com::E_OUTOFMEMORY_HR;
            }
            g_endpoint_activations.fetch_add(1);
            out.write(c.detach(kISpatialAudioClient));
            return kOk;
        }
        return kENoInterface;
    }
    HResult OpenPropertyStore(uint32_t /*access*/, OutPtr out) {  // 4
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        com::Ref<PropertyStoreObj> s = com::create_object<PropertyStoreObj>(com::implements(kIPropertyStore));
        const uint64_t store = s ? s.detach(kIPropertyStore) : 0;
        out.write(store);
        return store != 0 ? kOk : com::E_OUTOFMEMORY_HR;
    }
    HResult GetId(OutPtr out) {  // 5: CoTaskMemAlloc'ed string
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        const uint64_t text = co_task_copy(kEndpointId, sizeof(kEndpointId));
        out.write(text);
        return text != 0 ? kOk : com::E_OUTOFMEMORY_HR;
    }
    HResult GetState(GuestPtr<uint32_t> state) {  // 6
        return state.write(kDeviceStateActive) ? kOk : com::E_POINTER_HR;
    }
};

using DVM = com::Methods<DeviceObj>;
constexpr com::MethodDesc kDeviceMethods[] = {
    DVM::method<&DeviceObj::Activate>("Activate"),
    DVM::method<&DeviceObj::OpenPropertyStore>("OpenPropertyStore"),
    DVM::method<&DeviceObj::GetId>("GetId"),
    DVM::method<&DeviceObj::GetState>("GetState"),
};
constexpr auto kIMMDevice = com::declare_interface<DeviceObj>("IMMDevice", kIidIMMDevice, kDeviceMethods);

HResult new_device(OutPtr out) {
    com::Ref<DeviceObj> d = com::create_object<DeviceObj>(com::implements(kIMMDevice));
    const uint64_t device = d ? d.detach(kIMMDevice) : 0;
    out.write(device);
    return device != 0 ? kOk : com::E_OUTOFMEMORY_HR;
}

class CollectionObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.IMMDeviceCollection";
    explicit CollectionObj(uint32_t count) : count_(count) {}

    HResult GetCount(GuestPtr<uint32_t> count) {  // 3
        return count.write(count_) ? kOk : com::E_POINTER_HR;
    }
    HResult Item(uint32_t index, OutPtr out) {  // 4
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        out.write(0);
        if (index >= count_) {
            return com::E_INVALIDARG_HR;
        }
        return new_device(out);
    }

private:
    uint32_t count_;
};

using CLM = com::Methods<CollectionObj>;
constexpr com::MethodDesc kCollectionMethods[] = {
    CLM::method<&CollectionObj::GetCount>("GetCount"),
    CLM::method<&CollectionObj::Item>("Item"),
};
constexpr auto kIMMDeviceCollection =
    com::declare_interface<CollectionObj>("IMMDeviceCollection", kIidIMMDeviceCollection, kCollectionMethods);

class EnumeratorObj : public com::ComObject {
public:
    static constexpr const char* kComClassName = "MMDevAPI.IMMDeviceEnumerator";

    // 3 EnumAudioEndpoints(EDataFlow, DWORD state mask, IMMDeviceCollection**)
    HResult EnumAudioEndpoints(uint32_t flow, uint32_t state_mask, OutPtr out) {
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        out.write(0);
        if (flow > 2 || state_mask == 0 || (state_mask & ~0xFu) != 0) {
            return com::E_INVALIDARG_HR;
        }
        const uint32_t count = (flow == 0 || flow == 2) && (state_mask & kDeviceStateActive) != 0 ? 1u : 0u;
        com::Ref<CollectionObj> c = com::create_object<CollectionObj>(com::implements(kIMMDeviceCollection), count);
        if (!c) {
            return com::E_OUTOFMEMORY_HR;
        }
        out.write(c.detach(kIMMDeviceCollection));
        return kOk;
    }
    // 4 GetDefaultAudioEndpoint(EDataFlow, ERole, IMMDevice**)
    HResult GetDefaultAudioEndpoint(uint32_t flow, uint32_t role, OutPtr out) {
        if (out.is_null()) {
            return com::E_POINTER_HR;
        }
        out.write(0);
        if (flow > 2 || role > 2) {
            return com::E_INVALIDARG_HR;
        }
        if (flow != 0) {
            return kENotFound;  // no capture endpoint
        }
        return new_device(out);
    }
    // 5 GetDevice(LPCWSTR id, IMMDevice**)
    HResult GetDevice(uint64_t id, OutPtr out) {
        if (out.is_null() || id == 0) {
            return com::E_POINTER_HR;
        }
        out.write(0);
        char16_t text[sizeof(kEndpointId) / sizeof(char16_t)] = {};
        if (!com::read_guest(id, text, sizeof(text)) || std::memcmp(text, kEndpointId, sizeof(kEndpointId)) != 0) {
            return kENotFound;
        }
        return new_device(out);
    }
    // 6 / 7 Register/UnregisterEndpointNotificationCallback: the one endpoint never changes.
    HResult RegisterEndpointNotificationCallback(uint64_t client) { return client != 0 ? kOk : com::E_POINTER_HR; }
    HResult UnregisterEndpointNotificationCallback(uint64_t client) { return client != 0 ? kOk : com::E_POINTER_HR; }
};

using ENM = com::Methods<EnumeratorObj>;
constexpr com::MethodDesc kEnumeratorMethods[] = {
    ENM::method<&EnumeratorObj::EnumAudioEndpoints>("EnumAudioEndpoints"),
    ENM::method<&EnumeratorObj::GetDefaultAudioEndpoint>("GetDefaultAudioEndpoint"),
    ENM::method<&EnumeratorObj::GetDevice>("GetDevice"),
    ENM::method<&EnumeratorObj::RegisterEndpointNotificationCallback>("RegisterEndpointNotificationCallback"),
    ENM::method<&EnumeratorObj::UnregisterEndpointNotificationCallback>("UnregisterEndpointNotificationCallback"),
};
constexpr auto kIMMDeviceEnumerator =
    com::declare_interface<EnumeratorObj>("IMMDeviceEnumerator", kIidIMMDeviceEnumerator, kEnumeratorMethods);

// combase!CoCreateInstance(REFCLSID, IUnknown* outer, DWORD clsctx, REFIID, void** out)
HResult co_create_instance(const com::Guid& clsid, uint64_t outer, uint32_t clsctx, const com::Guid& iid, OutPtr out) {
    if (out.is_null()) {
        return com::E_POINTER_HR;
    }
    out.write(0);
    if (!(clsid == kClsidMMDeviceEnumerator) || !g_endpoint_enabled.load()) {
        // Every other class: the host shim's answer (logs each unknown CLSID once).
        Pg6Guid host_clsid{};
        Pg6Guid host_iid{};
        std::memcpy(&host_clsid, &clsid, sizeof(host_clsid));
        std::memcpy(&host_iid, &iid, sizeof(host_iid));
        void* ignored = nullptr;
        const int32_t hr = pg6_win32_co_create_instance(&host_clsid, nullptr, clsctx, &host_iid, &ignored);
        return hr >= 0 ? kClassNotReg : hr;
    }
    if (outer != 0) {
        return static_cast<HResult>(0x80040110u);  // CLASS_E_NOAGGREGATION
    }
    if (!(iid == kIidIMMDeviceEnumerator || iid == com::IID_IUnknown_GUID)) {
        return kENoInterface;
    }
    com::Ref<EnumeratorObj> e = com::create_object<EnumeratorObj>(com::implements(kIMMDeviceEnumerator));
    if (!e) {
        return com::E_OUTOFMEMORY_HR;
    }
    out.write(e.detach(kIMMDeviceEnumerator));
    return kOk;
}

}  // namespace

void set_audio_render_sink(AudioRenderSink sink_fn, void* user) {
    Sink& s = sink();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.fn = sink_fn;
    s.user = user;
}

void set_audio_stereo_sink(AudioStereoSink sink_fn, void* user) { audio_downmix::set_stereo_sink(sink_fn, user); }

AudioEndpointStats audio_endpoint_stats() {
    AudioEndpointStats s;
    s.activations = g_endpoint_activations.load();
    s.spatial_passes = g_spatial_passes.load();
    s.render_frames = g_render_frames.load();
    s.spatial_object_buffers = g_spatial_object_buffers.load();
    s.spatial_nonzero_buffers = g_spatial_nonzero_buffers.load();
    s.spatial_nonzero_passes = g_spatial_nonzero_passes.load();
    s.render_nonzero_frames = g_render_nonzero_frames.load();
    s.render_silent_releases = g_render_silent_releases.load();
    return s;
}

void audio_engine_manual_clock(bool manual) { engine().set_manual(manual); }
void audio_engine_tick() { engine().tick(); }

void set_audio_endpoint_enabled(bool on) { g_endpoint_enabled.store(on); }
bool audio_endpoint_enabled() { return g_endpoint_enabled.load(); }

void reset_audio_endpoint() {
    engine().reset();
    set_audio_render_sink(nullptr, nullptr);
    audio_downmix::reset();
    g_endpoint_activations = 0;
    g_spatial_passes = 0;
    g_render_frames = 0;
    g_spatial_object_buffers = 0;
    g_spatial_nonzero_buffers = 0;
    g_spatial_nonzero_passes = 0;
    g_render_nonzero_frames = 0;
    audio_report::windows().reset();
    g_render_silent_releases = 0;
}

}  // namespace playgta6::os::scarlett::guest_api

extern "C" {

void pg6_gthunk_co_create_instance(void* state) {
    playgta6::guest::com::function_thunk<&playgta6::os::scarlett::guest_api::co_create_instance>()(state);
}

}  // extern "C"
