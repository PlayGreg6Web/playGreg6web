// Native executor: present mode selection and the host-side frame cap. Pure logic, no GPU (unit
// tested in tests/test_native_exec_model.cpp). The guest's own pacing is untouched: the cap only
// delays the host's present, and `auto` keeps the Mailbox -> Immediate -> Fifo preference.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace playgta6::gpu::native_exec {

enum class PresentChoice : uint8_t { Auto, Fifo, Mailbox, Immediate };

// Bit per mode in a supported-modes mask.
inline constexpr uint8_t kModeFifo = 1, kModeMailbox = 2, kModeImmediate = 4;

inline const char* present_choice_name(PresentChoice c) {
    switch (c) {
        case PresentChoice::Fifo: return "fifo";
        case PresentChoice::Mailbox: return "mailbox";
        case PresentChoice::Immediate: return "immediate";
        default: return "auto";
    }
}

inline bool parse_present_choice(const char* s, PresentChoice* out) {
    if (!s) return false;
    for (PresentChoice c : {PresentChoice::Auto, PresentChoice::Fifo, PresentChoice::Mailbox, PresentChoice::Immediate})
        if (std::strcmp(s, present_choice_name(c)) == 0) {
            *out = c;
            return true;
        }
    return false;
}

struct PresentSelection {
    PresentChoice actual = PresentChoice::Fifo;  // never Auto
    bool fell_back = false;                      // an explicit request was not supported
    std::string note;                            // why (log line), empty when nothing special happened
};

inline uint8_t present_mode_bit(PresentChoice c) {
    return c == PresentChoice::Fifo ? kModeFifo : c == PresentChoice::Mailbox ? kModeMailbox
         : c == PresentChoice::Immediate ? kModeImmediate : 0;
}

// supported: kMode* mask from the surface capabilities (0 = unknown: Fifo, the always-available mode).
inline PresentSelection select_present_mode(PresentChoice requested, uint8_t supported) {
    PresentSelection r;
    const PresentChoice best = (supported & kModeMailbox) ? PresentChoice::Mailbox
                             : (supported & kModeImmediate) ? PresentChoice::Immediate : PresentChoice::Fifo;
    if (requested == PresentChoice::Auto) {
        r.actual = best;
        return r;
    }
    if (requested == PresentChoice::Fifo || (supported & present_mode_bit(requested))) {
        r.actual = requested;
        return r;
    }
    r.fell_back = true;
    r.actual = best;
    r.note = std::string("present mode ") + present_choice_name(requested) + " not supported by the surface; using " +
             present_choice_name(r.actual);
    return r;
}

// Host-side frame cap: how long to wait before the next present may happen. max_fps 0 = off.
// anchor_ns / now_ns on one monotonic clock; anchor_ns 0 = no present yet.
inline uint64_t frame_cap_wait_ns(uint32_t max_fps, uint64_t anchor_ns, uint64_t now_ns) {
    if (max_fps == 0 || anchor_ns == 0) return 0;
    const uint64_t due = anchor_ns + 1000000000ull / max_fps;
    return now_ns >= due ? 0 : due - now_ns;
}

// The anchor after a present at now_ns: the due time while on schedule (no drift), now once a frame ran
// more than one interval late (no catch-up burst).
inline uint64_t frame_cap_next_anchor(uint32_t max_fps, uint64_t anchor_ns, uint64_t now_ns) {
    if (max_fps == 0 || anchor_ns == 0) return now_ns;
    const uint64_t interval = 1000000000ull / max_fps;
    const uint64_t due = anchor_ns + interval;
    return now_ns < due + interval ? (now_ns < due ? now_ns : due) : now_ns;
}

}  // namespace playgta6::gpu::native_exec
