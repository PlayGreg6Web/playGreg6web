#pragma once
// Streaming-drain diagnostics (#663, #1960): the title's "drain all streaming" function
// (main executable sub_141E10090) loops until a set of streaming counters is zero. It copies a 0x2c48
// byte block at `*(0x14764B240) + 0x13a80` and, for the category index it was given (< 0x31), sums five
// u64 counters of that category's 0xC0-byte entry (entries start 0x48 into the block; counters at
// +0x00/+0x18/+0x30/+0x48/+0x60); for "all" (index -1) it sums five u64 at block +0x24f0 + k*0x18.
// This reads those counters with checked guest reads at report time, so a fault or stall report says
// what the loop was still waiting on, and lists which guest threads are inside the function.
// Addresses and offsets only; no game data. Link-time VAs; the caller rebases them.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace playgta6::guest {

constexpr uint64_t kDrainLinkBase = 0x140000000ull;
constexpr uint64_t kDrainSingletonVa = 0x14764B240ull;  // global holding a pointer to the streaming owner
constexpr uint64_t kDrainBlockOffset = 0x13a80;
constexpr size_t kDrainBlockBytes = 0x2c48;
constexpr size_t kDrainEntriesOffset = 0x48;
constexpr size_t kDrainEntryStride = 0xC0;
constexpr size_t kDrainEntryCount = 0x31;
constexpr size_t kDrainCountersPerEntry = 5;
constexpr size_t kDrainCounterStride = 0x18;
constexpr size_t kDrainAllOffset = 0x24f0;
constexpr uint64_t kDrainFunctionVa = 0x141E10090ull;
constexpr uint64_t kDrainFunctionBytes = 0x17c;  // up to its ret at 0x141E1020B

struct DrainEntry {
    uint32_t index = 0;
    uint64_t counters[kDrainCountersPerEntry] = {};
};

struct StreamingDrainState {
    bool available = false;  // singleton pointer and block were readable
    uint64_t owner = 0;
    std::vector<DrainEntry> pending;  // categories with a non-zero counter sum
    uint64_t all[kDrainCountersPerEntry] = {};
    uint64_t all_sum = 0;
};

using DrainRead = std::function<bool(uint64_t va, void* dst, size_t len)>;

inline uint64_t drain_le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

// `singleton` = rebased address of the global. Never faults: every access goes through `read`.
inline StreamingDrainState read_streaming_drain(const DrainRead& read, uint64_t singleton) {
    StreamingDrainState out;
    uint8_t ptr[8];
    if (!read(singleton, ptr, sizeof ptr)) return out;
    out.owner = drain_le64(ptr);
    if (out.owner == 0 || out.owner + kDrainBlockOffset < out.owner) return out;
    std::vector<uint8_t> block(kDrainBlockBytes);
    if (!read(out.owner + kDrainBlockOffset, block.data(), block.size())) return out;
    out.available = true;
    for (size_t e = 0; e < kDrainEntryCount; ++e) {
        DrainEntry d;
        d.index = static_cast<uint32_t>(e);
        uint64_t sum = 0;
        for (size_t k = 0; k < kDrainCountersPerEntry; ++k) {
            d.counters[k] = drain_le64(block.data() + kDrainEntriesOffset + e * kDrainEntryStride + k * kDrainCounterStride);
            sum += d.counters[k];
        }
        if (sum != 0) out.pending.push_back(d);
    }
    for (size_t k = 0; k < kDrainCountersPerEntry; ++k) {
        out.all[k] = drain_le64(block.data() + kDrainAllOffset + k * kDrainCounterStride);
        out.all_sum += out.all[k];
    }
    return out;
}

// True when `rip` or any of `frame_functions` (function entries of the thread's stack-scan frames) lies
// inside the drain function at `function_va` (rebased).
inline bool thread_in_drain(uint64_t rip, const std::vector<uint64_t>& frame_functions, uint64_t function_va) {
    auto inside = [&](uint64_t a) { return a >= function_va && a < function_va + kDrainFunctionBytes; };
    if (inside(rip)) return true;
    for (uint64_t f : frame_functions) {
        if (inside(f)) return true;
    }
    return false;
}

inline std::string streaming_drain_json(const StreamingDrainState& s, const std::vector<uint32_t>& threads_in_drain) {
    char buf[160];
    std::string o = "{\"available\": ";
    o += s.available ? "true" : "false";
    std::snprintf(buf, sizeof buf, ", \"owner\": \"0x%llx\"", static_cast<unsigned long long>(s.owner));
    o += buf;
    if (s.available) {
        std::snprintf(buf, sizeof buf, ", \"all_sum\": %llu, \"all\": [", static_cast<unsigned long long>(s.all_sum));
        o += buf;
        for (size_t k = 0; k < kDrainCountersPerEntry; ++k) {
            o += (k == 0 ? "" : ", ") + std::to_string(s.all[k]);
        }
        o += "], \"pending_categories\": [";
        for (size_t i = 0; i < s.pending.size(); ++i) {
            o += (i == 0 ? "{\"index\": " : ", {\"index\": ") + std::to_string(s.pending[i].index) + ", \"counters\": [";
            for (size_t k = 0; k < kDrainCountersPerEntry; ++k) {
                o += (k == 0 ? "" : ", ") + std::to_string(s.pending[i].counters[k]);
            }
            o += "]}";
        }
        o += "]";
    }
    o += ", \"threads_in_drain\": [";
    for (size_t i = 0; i < threads_in_drain.size(); ++i) {
        o += (i == 0 ? "" : ", ") + std::to_string(threads_in_drain[i]);
    }
    return o + "]}";
}

}  // namespace playgta6::guest
