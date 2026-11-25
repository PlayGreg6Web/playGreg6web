#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "os/audio_downmix.h"

// --audio-dump: writes the stereo float32 downmix (audio_downmix::set_stereo_sink) to a standard
// WAV file (RIFF/WAVE, IEEE float format 3, 2 channels, 48000 Hz). The sizes in the header are
// patched on close(); the file stops growing after kMaxSeconds of audio.
namespace playgta6::os::audio_wav_dump {

constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kChannels = 2;
constexpr uint64_t kMaxSeconds = 600;  // 10 minutes
constexpr uint64_t kMaxFrames = kMaxSeconds * kSampleRate;
constexpr uint32_t kHeaderBytes = 44;

class Writer {
public:
    ~Writer() { close(); }

    bool open(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_) return false;
        file_ = std::fopen(path.c_str(), "wb");
        if (!file_) return false;
        frames_ = 0;
        write_header_locked();
        return true;
    }

    // Appends interleaved stereo frames; frames beyond the cap are dropped. Other rates are skipped.
    void append(const float* stereo, uint32_t frames, uint32_t sample_rate) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_ || !stereo || sample_rate != kSampleRate) return;
        const uint64_t room = kMaxFrames - frames_;
        const uint32_t n = frames > room ? static_cast<uint32_t>(room) : frames;
        if (n == 0) return;
        if (std::fwrite(stereo, sizeof(float) * kChannels, n, file_) == n) frames_ += n;
    }

    // Patches the header with the final size and closes the file.
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_) return;
        write_header_locked();
        std::fclose(file_);
        file_ = nullptr;
    }

    uint64_t frames() const { return frames_; }

    static void sink(void* user, const float* stereo, uint32_t frames, uint32_t sample_rate) {
        static_cast<Writer*>(user)->append(stereo, frames, sample_rate);
    }

private:
    static void put32(unsigned char* p, uint32_t v) {
        for (int i = 0; i < 4; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
    }
    static void put16(unsigned char* p, uint16_t v) {
        p[0] = static_cast<unsigned char>(v);
        p[1] = static_cast<unsigned char>(v >> 8);
    }

    void write_header_locked() {
        const uint32_t data_bytes = static_cast<uint32_t>(frames_ * sizeof(float) * kChannels);
        unsigned char h[kHeaderBytes];
        std::memcpy(h, "RIFF", 4);
        put32(h + 4, 36 + data_bytes);
        std::memcpy(h + 8, "WAVEfmt ", 8);
        put32(h + 16, 16);
        put16(h + 20, 3);  // WAVE_FORMAT_IEEE_FLOAT
        put16(h + 22, kChannels);
        put32(h + 24, kSampleRate);
        put32(h + 28, kSampleRate * kChannels * 4);
        put16(h + 32, kChannels * 4);
        put16(h + 34, 32);
        std::memcpy(h + 36, "data", 4);
        put32(h + 40, data_bytes);
        std::fseek(file_, 0, SEEK_SET);
        std::fwrite(h, 1, sizeof(h), file_);
        std::fseek(file_, 0, SEEK_END);
    }

    std::mutex mutex_;
    std::FILE* file_ = nullptr;
    uint64_t frames_ = 0;
};

}  // namespace playgta6::os::audio_wav_dump
