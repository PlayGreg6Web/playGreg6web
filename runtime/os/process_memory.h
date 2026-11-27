#pragma once

#include <cstdint>

namespace playgta6::os {

// Private commit and working set of this host process (boot report only). False when the platform
// call is unavailable; the outputs are left untouched then.
bool process_memory_bytes(uint64_t* private_bytes, uint64_t* working_set_bytes, uint64_t* peak_private_bytes = nullptr);

// Free physical memory of the machine in bytes; 0 when the platform call is unavailable.
uint64_t available_physical_bytes();

// Dedicated video memory of the largest hardware adapter (DXGI) in bytes; 0 when unknown.
uint64_t dedicated_video_memory_bytes();

// Committed bytes of the process by kind, from a VirtualQuery walk (boot report only; the walk is
// cheap, a few thousand regions). `arena_lo`..`arena_hi` is the guest arena's host range.
struct ProcessCommitMap {
    uint64_t arena_private = 0;  // MEM_PRIVATE commit inside the arena range
    uint64_t arena_mapped = 0;   // MEM_MAPPED commit inside the arena range (shared page views)
    uint64_t other_private = 0;  // MEM_PRIVATE commit elsewhere (heaps, Dawn, staging, stacks)
    uint64_t other_mapped = 0;
    uint64_t image = 0;          // MEM_IMAGE commit (the harness exe, DLLs)
    // Of the private commit, the bytes actually resident in the working set (touched pages).
    uint64_t arena_private_resident = 0;
    uint64_t other_private_resident = 0;
    // Private allocations of >= 4 MiB outside the arena by exact allocation size: the 8 most common.
    static constexpr int kSizes = 8;
    uint64_t size_bytes[kSizes] = {};
    uint64_t size_count[kSizes] = {};
    // Largest private allocations outside the arena (AllocationBase merged), descending.
    static constexpr int kTop = 8;
    uint64_t top_bytes[kTop] = {};
    uint64_t top_base[kTop] = {};
};
bool process_commit_map(uint64_t arena_lo, uint64_t arena_hi, ProcessCommitMap* out);

}  // namespace playgta6::os
