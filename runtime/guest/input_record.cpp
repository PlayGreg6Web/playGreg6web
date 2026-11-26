#include "guest/input_record.h"

#include <chrono>
#include <cstring>
#include <filesystem>

namespace playgta6::guest {

namespace {
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
bool same(const playgta6::InputState& a, const playgta6::InputState& b) {
    return a.buttons == b.buttons && a.left_stick_x == b.left_stick_x && a.left_stick_y == b.left_stick_y &&
           a.right_stick_x == b.right_stick_x && a.right_stick_y == b.right_stick_y &&
           a.left_trigger == b.left_trigger && a.right_trigger == b.right_trigger;
}
}  // namespace

std::string format_state_line(uint64_t present, const playgta6::InputState& s, int64_t ms) {
    char b[256];
    std::snprintf(b, sizeof(b), "state %llu 0x%x %.9g %.9g %.9g %.9g %.9g %.9g", static_cast<unsigned long long>(present),
                  s.buttons, s.left_stick_x, s.left_stick_y, s.right_stick_x, s.right_stick_y, s.left_trigger,
                  s.right_trigger);
    std::string line = b;
    if (ms >= 0) {
        line += "   # " + std::to_string(ms) + " ms";
    }
    return line;
}

bool InputRecorder::open(const std::string& path, PresentProbe presents, std::string& error) {
    close();
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    file_ = std::fopen(path.c_str(), "wb");
    if (file_ == nullptr) {
        error = "cannot write " + path;
        return false;
    }
    presents_ = std::move(presents);
    last_ = playgta6::InputState{};
    lines_ = 0;
    t0_ns_ = now_ns();
    std::fprintf(file_, "%s\n", kInputScriptVersionLine);
    std::fflush(file_);
    return true;
}

void InputRecorder::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_ != nullptr) {
        std::fclose(file_);
        file_ = nullptr;
    }
}

void InputRecorder::record(const playgta6::InputState& reading) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_ == nullptr || same(reading, last_)) {
        return;
    }
    last_ = reading;
    const uint64_t present = presents_ ? presents_() : 0;
    const std::string line = format_state_line(present, reading, (now_ns() - t0_ns_) / 1000000);
    std::fprintf(file_, "%s\n", line.c_str());
    std::fflush(file_);
    ++lines_;
}

}  // namespace playgta6::guest
