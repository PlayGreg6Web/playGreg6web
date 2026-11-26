#pragma once

// Recording of real host input (`playgta6_boot --record-input <file>`, with --host-input).
//
// Every change of the host reading (pad + keyboard, as HostInputSource publishes it) is written as one
// `state` line in the --input-script format (guest/input_script.h), stamped with the present index and
// the elapsed ms at the time of the reading. Replaying the file with --input-script writes the same
// states at the same present indices. Only readings of the real host are recorded, nothing synthesized.
//
//     # playgta6-input-script v2
//     state <present> <buttons_hex> <left_x> <left_y> <right_x> <right_y> <lt> <rt>   # <ms> ms

#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>

#include "abi/playgta6_abi.h"

namespace playgta6::guest {

inline constexpr const char* kInputScriptVersionLine = "# playgta6-input-script v2";

// One `state` line (no newline); `ms` < 0 omits the trailing time comment.
std::string format_state_line(uint64_t present, const playgta6::InputState& s, int64_t ms = -1);

class InputRecorder {
public:
    using PresentProbe = std::function<uint64_t()>;
    InputRecorder() = default;
    ~InputRecorder() { close(); }
    InputRecorder(const InputRecorder&) = delete;
    InputRecorder& operator=(const InputRecorder&) = delete;

    // Creates the file (and its directory) and writes the version line.
    bool open(const std::string& path, PresentProbe presents, std::string& error);
    void close();
    // Writes a line when `reading` differs from the last recorded one (the first is compared with idle).
    void record(const playgta6::InputState& reading);
    uint64_t lines() const { return lines_; }

private:
    std::mutex mutex_;
    std::FILE* file_ = nullptr;
    PresentProbe presents_;
    playgta6::InputState last_{};
    uint64_t lines_ = 0;
    int64_t t0_ns_ = 0;
};

}  // namespace playgta6::guest
