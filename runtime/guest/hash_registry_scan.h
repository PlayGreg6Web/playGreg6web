#pragma once
// Music-start diagnostics (backlog A4): reads the title's name-hash registry (the table the music
// controller's metadata lookups search) out of guest memory, to say how many entries it holds and
// whether given hashes are in it. Layout, as the title's lookup reads it: `groups` groups of 0x110
// bytes, each 16 control bytes (high bit set = empty slot, else the slot is full) then 16 slots of
// {u32 key, u32 pad, u64 value}. Only counts and hashes are reported, never game data.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace playgta6::guest {

struct HashRegistryScan {
    bool readable = false;   // every group could be read
    uint64_t full_slots = 0;
    std::vector<bool> present;  // one per queried hash
    std::vector<bool> value_null;  // the queried hash is present with a null value
};

constexpr uint64_t kHashRegistryGroupBytes = 0x110;
constexpr uint32_t kHashRegistryMaxGroups = 1u << 20;

inline HashRegistryScan scan_hash_registry(const std::function<bool(uint64_t va, void* dst, size_t len)>& read,
                                           uint64_t table, uint32_t groups, const std::vector<uint32_t>& hashes) {
    HashRegistryScan out;
    out.present.assign(hashes.size(), false);
    out.value_null.assign(hashes.size(), false);
    if (table == 0 || groups == 0 || groups > kHashRegistryMaxGroups) {
        out.readable = groups == 0;
        return out;
    }
    uint8_t group[kHashRegistryGroupBytes];
    for (uint32_t g = 0; g < groups; ++g) {
        if (!read(table + g * kHashRegistryGroupBytes, group, sizeof group)) {
            return out;
        }
        for (int s = 0; s < 16; ++s) {
            if ((group[s] & 0x80) != 0) {
                continue;
            }
            ++out.full_slots;
            const uint8_t* slot = group + 0x10 + s * 0x10;
            const uint32_t key = static_cast<uint32_t>(slot[0]) | (static_cast<uint32_t>(slot[1]) << 8) |
                                 (static_cast<uint32_t>(slot[2]) << 16) | (static_cast<uint32_t>(slot[3]) << 24);
            uint64_t value = 0;
            for (int i = 7; i >= 0; --i) {
                value = (value << 8) | slot[8 + i];
            }
            for (size_t h = 0; h < hashes.size(); ++h) {
                if (hashes[h] == key) {
                    out.present[h] = true;
                    out.value_null[h] = value == 0;
                }
            }
        }
    }
    out.readable = true;
    return out;
}

}  // namespace playgta6::guest
