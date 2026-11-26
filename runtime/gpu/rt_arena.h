// Paged GPU address arena for ray tracing / raw-address compute (docs/RAYTRACING_DESIGN.md section 3,
// stage R0/R2 runtime half). Pure bookkeeping: the 40-bit address -> page mapping, the two-level
// page directory packed exactly as the WGSL prelude reads it, residency with LRU eviction and
// pinning, GPU-ownership of written pages, and counters. The caller owns the actual buffers and
// performs the uploads this class lists. Contract C3 bindings: group 3, bindings 16-21.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu::rt_arena {

constexpr uint32_t kPageShift = 16;
constexpr uint64_t kPageBytes = 1ull << kPageShift;  // 64 KiB
constexpr uint32_t kArenaCount = 4;
constexpr uint64_t kArenaBytes = 512ull << 20;  // 512 MiB each
constexpr uint32_t kPagesPerArena = static_cast<uint32_t>(kArenaBytes / kPageBytes);  // 8192
constexpr uint32_t kAddressBits = 40;
constexpr uint64_t kAddressMask = (1ull << kAddressBits) - 1;
constexpr uint32_t kTopEntries = 4096;  // one per 256 MiB
constexpr uint32_t kL2Entries = 4096;   // one per 64 KiB page
constexpr uint32_t kMaxL2Tables = 56;   // keeps the directory under 1 MiB

// Contract C3 binding slots (group 3).
constexpr uint32_t kBindGroup = 3;
constexpr uint32_t kBindArena0 = 16;  // 16..19 = pg6_arena0..3
constexpr uint32_t kBindDirectory = 20;  // pg6_arena_dir
constexpr uint32_t kBindStats = 21;      // pg6_rt_stats
constexpr uint32_t kStatFaults = 0;      // index 0 of pg6_rt_stats

// Second-level entry: bit 31 valid, bits 30:29 arena, bits 28:0 page slot. 0 = not resident.
constexpr uint32_t kEntryValid = 0x80000000u;
constexpr uint32_t pack_entry(uint32_t arena, uint32_t slot) {
    return kEntryValid | ((arena & 3u) << 29) | (slot & 0x1fffffffu);
}
constexpr bool entry_valid(uint32_t e) { return (e & kEntryValid) != 0; }
constexpr uint32_t entry_arena(uint32_t e) { return (e >> 29) & 3u; }
constexpr uint32_t entry_slot(uint32_t e) { return e & 0x1fffffffu; }

constexpr uint64_t page_of(uint64_t addr) { return (addr & kAddressMask) >> kPageShift; }
constexpr uint32_t top_index(uint64_t page) { return static_cast<uint32_t>(page >> 12); }
constexpr uint32_t l2_index(uint64_t page) { return static_cast<uint32_t>(page & 0xfff); }
// Byte offset of a page slot inside its arena buffer.
constexpr uint64_t slot_byte_offset(uint32_t slot) { return static_cast<uint64_t>(slot) * kPageBytes; }

enum class SkipReason : uint8_t { RtOff, ArenaFull, NotResident, Atomic64, TranslatorPending, Count };
const char* skip_reason_name(SkipReason r);

enum class Ensure : uint8_t { Resident, Uploaded, ArenaFull, TableFull, OutOfRange };

struct Upload {
    uint64_t page;
    uint32_t arena;
    uint32_t slot;
};

struct Stats {
    uint64_t resident = 0, pinned = 0, gpu_owned = 0;
    uint64_t uploads = 0, evictions = 0, skipped_gpu_owned = 0;
    uint64_t arena_full = 0, table_full = 0, out_of_range = 0;
    uint64_t faults = 0;  // last value read back from pg6_rt_stats[0]
    uint64_t skips[static_cast<size_t>(SkipReason::Count)] = {};
};

class Arena {
public:
    // `arenas` and `pages_per_arena` are overridable for tests / smaller adapters.
    explicit Arena(uint32_t arenas = kArenaCount, uint32_t pages_per_arena = kPagesPerArena);

    uint64_t capacity_pages() const { return capacity_; }
    uint64_t capacity_bytes() const { return capacity_ * kPageBytes; }

    // Frame boundary: advances the LRU clock.
    void begin_frame() { ++frame_; }
    // Starts one RT-class dispatch: pages ensured since are protected from eviction until the next one.
    void begin_dispatch() { ++epoch_; }

    // Makes `page` resident (pinned when `pin`). New or re-dirtied pages are appended to the pending
    // upload list unless the page is GPU-authoritative.
    Ensure ensure_page(uint64_t page, bool pin = false);
    // Makes every page overlapping [addr, addr+size) resident. Returns the first failure (pages made
    // resident before it stay). size 0 touches one page.
    Ensure ensure_range(uint64_t addr, uint64_t size, bool pin = false);

    // Structures: pin a page set while the structure is live; release unpins it.
    Ensure pin_structure(uint64_t id, uint64_t addr, uint64_t size);
    void release_structure(uint64_t id);

    // Ownership. A builder/trace program wrote the page: CPU uploads skip it until freed/remapped.
    void mark_gpu_written(uint64_t page);
    // Title freed or remapped [addr, addr+size): drops GPU ownership and marks pages dirty.
    void mark_cpu_remapped(uint64_t addr, uint64_t size);
    // Guest memory changed under a page (write-mark): reupload if resident and not GPU-owned.
    void mark_cpu_dirty(uint64_t page);

    bool is_resident(uint64_t page) const;
    // 0 when not resident. Mirrors exactly what the WGSL prelude computes.
    uint32_t lookup_entry(uint64_t addr) const;

    // Pending uploads (clears the list).
    std::vector<Upload> take_uploads();

    // Directory words (top level, then L2 tables) and the [first,last] word range changed since the
    // last take_dir_dirty().
    const std::vector<uint32_t>& directory() const { return dir_; }
    bool take_dir_dirty(uint32_t* first, uint32_t* last);

    void note_skip(SkipReason r) { ++stats_.skips[static_cast<size_t>(r)]; }
    void set_fault_count(uint64_t f) { stats_.faults = f; }
    Stats stats() const;
    std::string json() const;

private:
    struct Info {
        uint32_t entry = 0;
        uint64_t last_frame = 0, epoch = 0;
        uint32_t pins = 0;
        bool gpu_owned = false, dirty = false;
        std::list<uint64_t>::iterator lru;
    };
    bool alloc_slot(uint32_t* arena, uint32_t* slot);
    bool evict_one();
    bool set_entry(uint64_t page, uint32_t entry);  // false: L2 table limit
    void touch_dir(uint32_t word);

    uint32_t arenas_, pages_per_arena_;
    uint64_t capacity_, frame_ = 1, epoch_ = 1, next_fresh_ = 0;
    std::vector<uint32_t> dir_;
    uint32_t l2_used_ = 0;
    uint32_t dirty_first_ = 0xffffffffu, dirty_last_ = 0;
    std::unordered_map<uint64_t, Info> pages_;
    std::list<uint64_t> lru_;           // front = most recent
    std::vector<uint64_t> free_slots_;  // packed arena<<32|slot
    std::unordered_map<uint64_t, std::vector<uint64_t>> structures_;
    std::vector<Upload> uploads_;
    Stats stats_;
};

}  // namespace playgta6::gpu::rt_arena
