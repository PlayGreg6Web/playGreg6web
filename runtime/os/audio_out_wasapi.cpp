#include "os/audio_out_wasapi.h"

#include <atomic>
#include <cstdio>
#include <thread>

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

    bool open_device(uint32_t* rate, std::string* err) {
        const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        com_inited_ = SUCCEEDED(co);
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
        CoTaskMemFree(mix);
        WAVEFORMATEX fmt = {};
        fmt.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
        fmt.nChannels = 2;
        fmt.nSamplesPerSec = dev_rate;
        fmt.wBitsPerSample = 32;
        fmt.nBlockAlign = 8;
        fmt.nAvgBytesPerSec = dev_rate * 8;
        // AUTOCONVERTPCM lets the engine accept stereo float even on a surround mix format.
        hr = client_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                 AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                                 2000000 /* 200 ms */, 0, &fmt, nullptr);
        if (FAILED(hr)) return fail(err, "IAudioClient::Initialize failed", hr);
        hr = client_->GetBufferSize(&buffer_frames_);
        if (FAILED(hr)) return fail(err, "GetBufferSize failed", hr);
        hr = client_->GetService(iid_render_client(), reinterpret_cast<void**>(&render_));
        if (FAILED(hr)) return fail(err, "GetService(IAudioRenderClient) failed", hr);
        *rate = dev_rate;
        return true;
    }

    static bool fail(std::string* err, const char* what, HRESULT hr) {
        char b[128];
        std::snprintf(b, sizeof(b), "%s (hr=0x%08lx)", what, static_cast<unsigned long>(hr));
        if (err) *err = b;
        return false;
    }

    void close_device() {
        if (render_) render_->Release();
        if (client_) client_->Release();
        if (dev_) dev_->Release();
        render_ = nullptr;
        client_ = nullptr;
        dev_ = nullptr;
        device_rate_ = 0;
        if (com_inited_) CoUninitialize();
        com_inited_ = false;
    }

    void run() {
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        client_->Start();
        while (running_) {
            UINT32 padding = 0;
            if (SUCCEEDED(client_->GetCurrentPadding(&padding)) && buffer_frames_ > padding) {
                const UINT32 want = buffer_frames_ - padding;
                BYTE* data = nullptr;
                if (SUCCEEDED(render_->GetBuffer(want, &data)) && data) {
                    feed_->buffer().pop(reinterpret_cast<float*>(data), want);
                    render_->ReleaseBuffer(want, 0);
                }
            }
            Sleep(5);
        }
        client_->Stop();
        CoUninitialize();
    }

    IMMDevice* dev_ = nullptr;
    IAudioClient* client_ = nullptr;
    IAudioRenderClient* render_ = nullptr;
    UINT32 buffer_frames_ = 0;
    bool com_inited_ = false;
#endif
    Feed* feed_ = nullptr;
    std::atomic<bool> running_{false};
    std::thread thread_;
    uint32_t device_rate_ = 0;

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
        feed.configure(rate, 400, 60);
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
bool Wasapi::start(Feed& feed, std::string* error) { return impl_->start(feed, error); }
void Wasapi::stop() { impl_->stop(); }
bool Wasapi::open() const { return impl_->device_rate_ != 0; }
uint32_t Wasapi::device_rate() const { return impl_->device_rate_; }

}  // namespace playgta6::os::audio_out
