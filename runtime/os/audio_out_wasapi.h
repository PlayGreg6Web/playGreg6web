#pragma once

#include <cstdint>
#include <string>

#include "os/audio_out.h"

// Windows WASAPI shared-mode output on the default render endpoint (audio_out_wasapi.cpp; the
// windows headers live only there because they clash with os/win32_compat.h). On other
// platforms, or when no device is available, start() returns false with an error and the
// harness carries on silent.
namespace playgta6::os::audio_out {

class Wasapi {
public:
    Wasapi();
    ~Wasapi();
    Wasapi(const Wasapi&) = delete;
    Wasapi& operator=(const Wasapi&) = delete;

    bool start(Feed& feed, std::string* error);
    void stop();
    bool open() const;
    uint32_t device_rate() const;
    uint32_t device_channels() const;
    uint32_t device_channel_mask() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace playgta6::os::audio_out
