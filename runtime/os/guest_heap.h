#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

#include "os/vmem.h"

// Guest allocator service (Track 2, Task O24)
// ===========================================
//
// Every allocation whose address is handed to guest code and that the guest may later free or
// query (xmem.dll XMemAlloc*, the C runtime's malloc family, strdup, argv/environment tables, the
// onexit table, exception strings, ...) comes from here and therefore lives in the guest address
// space (os/vmem.h): the title's gigabytes are accounted as guest memory, are valid guest
// addresses on every arena (identity or offset-mapped) and never touch the runtime's own heap,
// which is fenced below 4 GiB on wasm64.
//
// One allocator, one family
// -------------------------
// A GuestHeap is bound to one VirtualMemoryArena and works on guest virtual addresses (uint64_t).
// All blocks, whatever API produced them (malloc, calloc, realloc, _aligned_malloc, strdup,
// XMemAlloc, ...), live in one table keyed by address, so free() accepts any of them, in either
// order (an _aligned_malloc block may be released with free() and vice versa without corruption,
// which is more permissive than Windows). Foreign pointers and double frees are looked up, not
// dereferenced: they are counted (GuestHeapStats::invalid_frees), logged to stderr (first few per
// heap) and ignored, never undefined behaviour. All bookkeeping is host side, so guest buffer
// overruns cannot corrupt allocator state.
//
// Backends
// --------
// * Small / medium blocks (size <= kLargeThreshold, alignment <= 64 KiB): best-fit free lists with
//   address-ordered coalescing inside 2 MiB arena segments (reserved and committed in one piece
//   through VirtualMemoryArena::virtual_alloc). Blocks are 16-byte aligned (Win64
//   MEMORY_ALLOCATION_ALIGNMENT); larger alignments carve an aligned block out of a free range
//   and give the head and tail back. Segments stay reserved for the life of the heap (reused by
//   later allocations) - only the large path returns memory to the arena immediately.
// * Large blocks (size > kLargeThreshold = 128 KiB, or alignment > 64 KiB): one arena reservation
//   each, committed directly, so freeing a 512 MiB block releases its VA and commit at once.
//   Alignments up to 64 KiB come for free (reservations are 64 KiB aligned); larger ones are
//   placed by probing for an aligned hole and reserving it at an exact address. realloc shrinks
//   in place by decommitting the tail pages (the VA stays reserved so a later growth is a plain
//   commit) and moves blocks that fall under the threshold into the small backend.
//
// Semantics (MSVC): malloc(0) returns a unique non-null pointer; usable_size() returns the
// *requested* size (what _msize reports); realloc(p, 0) frees p and returns 0 without an error;
// realloc(0, n) is alloc(n); free(0) is a no-op; calloc/zeroed blocks are zero filled.
//
// Accounting and limits
// ---------------------
// stats() reports live/peak bytes and blocks in total, per family (XMem / CRT / Other) and for
// the large backend, failure counters and the committed bytes the heap holds in the arena. A soft
// limit (set_soft_limit, 0 = unlimited) caps the live requested bytes: allocations beyond it fail
// cleanly (return 0, HeapStatus::OutOfMemory) exactly like exhausting the arena does; the shims
// translate that into nullptr + errno (CRT) / last error (XMem). The C ABI at the bottom
// (pg6_guest_heap_*) exposes the process heap's numbers to the diagnostics HUD.
//
// Pointers: guest VA vs host pointer
// ----------------------------------
// The class deals in guest VAs only. The shims (ucrt_bridge, xbox_scarlett) are called by the
// import bridge with host-typed arguments and results; the bridge translates top-level pointer
// arguments and results with the one formula `host = guest_base + VA`, so a shim returns
// to_host(va) and looks blocks up with to_guest(ptr) and is correct on identity and offset
// arenas alike. What is identity-only: pointers the shims *store inside memory* that the
// guest reads must be guest VAs (the CRT tables below store them as VAs), but guest-visible
// pointers to *static host storage* (the address of the `argv` variable returned by
// __p___argv, FILE objects, errno) are only meaningful through the bridge's round trip
// host -> VA -> host, which the formula makes lossless, so they work as long as the host
// address is reachable from the guest base (always true here: the guest range spans the whole
// translated space the lifted code may address).
//
// Threading: every method is thread safe. The default heap follows get_vmem_arena(): when a test
// installs another arena the previous default heap is abandoned (its blocks stay with its arena)
// and a new one is created; the old heap objects are never freed so concurrent users stay valid.

namespace playgta6::os {

enum class HeapFamily : uint8_t {
    Crt = 0,    // malloc family, strdup, CRT tables handed to the guest
    XMem = 1,   // xmem.dll XMemAlloc / XMemAllocDefault (and mapping buffers)
    Other = 2,  // anything else
};
inline constexpr size_t kHeapFamilyCount = 3;

enum class HeapStatus : uint8_t {
    Ok = 0,
    OutOfMemory,       // soft limit reached, request too large or arena exhausted
    InvalidParameter,  // bad alignment, count * size overflow
    InvalidPointer,    // not a live block of this heap (foreign pointer / double free)
};

inline constexpr uint32_t kHeapZeroMemory = 0x1u;  // zero-fill the block (calloc semantics)

struct GuestHeapFamilyStats {
    uint64_t live_bytes = 0;
    uint64_t peak_bytes = 0;
    uint64_t live_blocks = 0;
    uint64_t total_allocs = 0;
};

struct GuestHeapStats {
    uint64_t live_bytes = 0;  // requested bytes of all live blocks
    uint64_t peak_bytes = 0;
    uint64_t live_blocks = 0;
    uint64_t total_allocs = 0;
    uint64_t total_frees = 0;
    uint64_t failed_allocs = 0;
    uint64_t invalid_frees = 0;     // foreign pointers and double frees (ignored)
    uint64_t soft_limit_bytes = 0;  // 0 = unlimited
    uint64_t large_live_bytes = 0;  // requested bytes of live large blocks
    uint64_t large_peak_bytes = 0;
    uint64_t large_live_blocks = 0;
    uint64_t committed_bytes = 0;   // arena bytes this heap currently commits (segments + large)
    GuestHeapFamilyStats family[kHeapFamilyCount];
};

struct HeapBlockInfo {
    uint64_t size = 0;       // requested size
    uint64_t alignment = 0;  // requested alignment (>= 16)
    HeapFamily family = HeapFamily::Crt;
    bool large = false;
};

class GuestHeap {
public:
    static constexpr uint64_t kMinAlignment = 16;
    static constexpr uint64_t kLargeThreshold = uint64_t{128} << 10;
    static constexpr uint64_t kSegmentSize = uint64_t{2} << 20;
    static constexpr uint64_t kMaxRequest = uint64_t{1} << 40;
    static constexpr uint64_t kInvalidSize = ~uint64_t{0};

    // The arena must outlive the heap (the destructor releases the heap's reservations).
    explicit GuestHeap(VirtualMemoryArena& arena);
    ~GuestHeap();
    GuestHeap(const GuestHeap&) = delete;
    GuestHeap& operator=(const GuestHeap&) = delete;

    // All return a guest VA, or 0 on failure with *status set (status may be null).
    // `alignment` must be a power of two (values below 16 are raised to 16).
    uint64_t alloc(uint64_t size,
                   uint64_t alignment = kMinAlignment,
                   HeapFamily family = HeapFamily::Crt,
                   uint32_t flags = 0,
                   HeapStatus* status = nullptr);
    uint64_t calloc(uint64_t count,
                    uint64_t size,
                    uint64_t alignment = kMinAlignment,
                    HeapFamily family = HeapFamily::Crt,
                    HeapStatus* status = nullptr);
    // alignment 0 keeps the block's alignment. realloc(0, n) allocates (family given),
    // realloc(p, 0) frees p and returns 0 with status Ok. On failure the old block is untouched.
    uint64_t realloc(uint64_t va,
                     uint64_t size,
                     uint64_t alignment = 0,
                     HeapFamily family = HeapFamily::Crt,
                     uint32_t flags = 0,
                     HeapStatus* status = nullptr);
    // Grows or shrinks in place only (_expand); returns va or 0 (block untouched).
    uint64_t expand(uint64_t va, uint64_t size, HeapStatus* status = nullptr);
    // free(0) succeeds; an unknown pointer returns false (InvalidPointer) and is counted.
    bool free(uint64_t va, HeapStatus* status = nullptr);

    // Requested size of a live block, or kInvalidSize.
    uint64_t usable_size(uint64_t va) const;
    bool block_info(uint64_t va, HeapBlockInfo* out) const;
    bool owns(uint64_t va) const { return block_info(va, nullptr); }

    GuestHeapStats stats() const;
    void set_soft_limit(uint64_t bytes);
    uint64_t soft_limit() const;

    VirtualMemoryArena& arena() const { return arena_; }
    uint8_t* to_host(uint64_t va) const { return va == 0 ? nullptr : arena_.translate(va); }
    uint64_t to_guest(const void* host) const {
        return host == nullptr ? 0 : arena_.guest_address(host);
    }

    // Forget every block without touching the arena (the arena is about to go away or be reused).
    void abandon();
    // Release every block and segment back to the arena.
    void release_all();

private:
    struct Block {
        uint64_t size = 0;       // requested
        uint64_t extent = 0;     // small: 16-byte footprint; large: reserved bytes
        uint64_t committed = 0;  // large: committed bytes (<= extent); small: == extent
        uint64_t alignment = kMinAlignment;
        HeapFamily family = HeapFamily::Crt;
        bool large = false;
    };

    bool charge_locked(uint64_t size, HeapFamily family);
    void uncharge_locked(uint64_t size, HeapFamily family);
    void resize_charge_locked(const Block& block, uint64_t new_size);
    void fail_locked(HeapStatus* status, HeapStatus why);

    uint64_t alloc_small_locked(uint64_t size, uint64_t alignment, HeapFamily family);
    uint64_t take_free_locked(uint64_t capacity, uint64_t alignment);
    bool grow_locked(uint64_t capacity, uint64_t alignment);
    void insert_free_locked(uint64_t addr, uint64_t size);
    void erase_free_locked(std::map<uint64_t, uint64_t>::iterator it);
    void add_free_locked(uint64_t addr, uint64_t size);
    uint64_t segment_base_locked(uint64_t addr) const;

    uint64_t reserve_large(uint64_t size, uint64_t alignment);
    void note_invalid_free(uint64_t va);
    int resize_in_place_locked(uint64_t va, Block& block, uint64_t size, uint32_t flags);
    void zero(uint64_t va, uint64_t len) const;

    VirtualMemoryArena& arena_;
    mutable std::mutex mutex_;
    bool abandoned_ = false;
    uint64_t soft_limit_ = 0;
    uint32_t invalid_logged_ = 0;

    std::map<uint64_t, uint64_t> segments_;                 // base -> size
    std::map<uint64_t, uint64_t> free_by_addr_;             // maximal free ranges
    std::set<std::pair<uint64_t, uint64_t>> free_by_size_;  // (size, addr)
    std::unordered_map<uint64_t, Block> live_;

    GuestHeapStats stats_;
};

// The heap of the process arena (win32::get_vmem_arena()); follows set_vmem_arena().
GuestHeap& default_guest_heap();

// C-runtime flavoured allocation family on the default heap, host-pointer typed (the form the
// import bridge calls). malloc / calloc / realloc / _aligned_* / _msize / _recalloc / _expand /
// strdup / _wcsdup: one family, any of free() / _aligned_free() releases any block. Failure
// returns nullptr and sets *errno_out (ENOMEM / EINVAL) when it is non-null.
namespace guest_crt {

void* malloc(size_t size, int32_t* errno_out = nullptr);
void* calloc(size_t count, size_t size, int32_t* errno_out = nullptr);
void* realloc(void* ptr, size_t size, int32_t* errno_out = nullptr);
void* recalloc(void* ptr, size_t count, size_t size, int32_t* errno_out = nullptr);
void* expand(void* ptr, size_t size, int32_t* errno_out = nullptr);
void free(void* ptr);
void* aligned_malloc(size_t size, size_t alignment, int32_t* errno_out = nullptr);
void* aligned_realloc(void* ptr, size_t size, size_t alignment, int32_t* errno_out = nullptr);
void aligned_free(void* ptr);
size_t msize(void* ptr, int32_t* errno_out = nullptr);  // (size_t)-1 + EINVAL on a foreign pointer
char* strdup(const char* str, int32_t* errno_out = nullptr);
uint16_t* wcsdup(const uint16_t* str, int32_t* errno_out = nullptr);

}  // namespace guest_crt

}  // namespace playgta6::os

extern "C" {

// Diagnostics HUD view of the process guest heap (all values bytes / counts).
typedef struct Pg6GuestHeapFamilyStats {
    uint64_t live_bytes;
    uint64_t peak_bytes;
    uint64_t live_blocks;
    uint64_t total_allocs;
} Pg6GuestHeapFamilyStats;

typedef struct Pg6GuestHeapStats {
    uint64_t live_bytes;
    uint64_t peak_bytes;
    uint64_t live_blocks;
    uint64_t total_allocs;
    uint64_t total_frees;
    uint64_t failed_allocs;
    uint64_t invalid_frees;
    uint64_t soft_limit_bytes;
    uint64_t large_live_bytes;
    uint64_t large_peak_bytes;
    uint64_t large_live_blocks;
    uint64_t committed_bytes;
    Pg6GuestHeapFamilyStats crt;
    Pg6GuestHeapFamilyStats xmem;
    Pg6GuestHeapFamilyStats other;
} Pg6GuestHeapStats;

// Returns 1 and fills *out, or 0 when out is null.
int32_t pg6_guest_heap_get_stats(Pg6GuestHeapStats* out);
// 0 = unlimited. Applies to the process heap (all families together).
void pg6_guest_heap_set_soft_limit(uint64_t bytes);

}  // extern "C"
