#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace playgta6::os {

inline constexpr uint64_t kPageSize = 4096;                  // 4 KiB
inline constexpr uint64_t kAllocationGranularity = 65536;    // 64 KiB
inline constexpr uint64_t kDefaultArenaBase = 0x0000000100000000ULL;

// Win32 / Xbox VirtualAlloc & VirtualFree flags.
inline constexpr uint32_t kMemCommit = 0x00001000u;
inline constexpr uint32_t kMemReserve = 0x00002000u;
inline constexpr uint32_t kMemDecommit = 0x00004000u;
inline constexpr uint32_t kMemRelease = 0x00008000u;
inline constexpr uint32_t kMemFree = 0x00010000u;

enum class MemState : uint32_t {
    Free = kMemFree,
    Reserve = kMemReserve,
    Commit = kMemCommit,
};

// Win32 / Xbox PAGE_* protection constants.
enum class PageProtect : uint32_t {
    NoAccess = 0x01u,          // PAGE_NOACCESS
    ReadOnly = 0x02u,          // PAGE_READONLY
    ReadWrite = 0x04u,         // PAGE_READWRITE
    ExecuteRead = 0x20u,       // PAGE_EXECUTE_READ
    ExecuteReadWrite = 0x40u,  // PAGE_EXECUTE_READWRITE
};

inline constexpr bool is_readable(PageProtect protect) {
    return protect == PageProtect::ReadOnly ||
           protect == PageProtect::ReadWrite ||
           protect == PageProtect::ExecuteRead ||
           protect == PageProtect::ExecuteReadWrite;
}

inline constexpr bool is_writable(PageProtect protect) {
    return protect == PageProtect::ReadWrite ||
           protect == PageProtect::ExecuteReadWrite;
}

inline constexpr bool is_executable(PageProtect protect) {
    return protect == PageProtect::ExecuteRead ||
           protect == PageProtect::ExecuteReadWrite;
}

inline constexpr uint64_t align_down(uint64_t value, uint64_t alignment) {
    return value & ~(alignment - 1u);
}

inline constexpr bool align_up_checked(uint64_t value, uint64_t alignment, uint64_t& out) {
    if (alignment == 0 || (alignment & (alignment - 1u)) != 0) {
        return false;
    }
    const uint64_t mask = alignment - 1u;
    if (value > UINT64_MAX - mask) {
        return false;
    }
    out = (value + mask) & ~mask;
    return true;
}

struct MemoryBasicInfo {
    uint64_t base_address = 0;
    uint64_t allocation_base = 0;
    uint64_t region_size = 0;
    MemState state = MemState::Free;
    PageProtect protect = PageProtect::NoAccess;
    PageProtect allocation_protect = PageProtect::NoAccess;
};

// Software virtual memory arena matching Win32/Xbox VirtualAlloc, VirtualProtect,
// VirtualFree, and VirtualQuery semantics for Wasm64 and native host execution.
class VirtualMemoryArena {
public:
    explicit VirtualMemoryArena(uint64_t default_base = kDefaultArenaBase);
    ~VirtualMemoryArena() = default;

    VirtualMemoryArena(const VirtualMemoryArena&) = delete;
    VirtualMemoryArena& operator=(const VirtualMemoryArena&) = delete;

    // Reserve a 64 KiB-aligned virtual address range (size rounded up to 4 KiB pages).
    // Returns the reserved base VA on success, or 0 on failure.
    uint64_t reserve(uint64_t hint_addr,
                     uint64_t size,
                     PageProtect allocation_protect = PageProtect::NoAccess);

    // Commit a 4 KiB page-aligned sub-range inside an existing reservation.
    // Newly committed pages are zero-initialized; already committed pages retain data
    // and receive the updated protection.
    bool commit(uint64_t addr,
                uint64_t size,
                PageProtect protect = PageProtect::ReadWrite);

    // Decommit a 4 KiB page-aligned sub-range, releasing its host backing storage.
    // If size == 0 and addr == reservation base, decommits the entire reservation.
    bool decommit(uint64_t addr, uint64_t size);

    // Change protection on a committed 4 KiB page-aligned sub-range.
    // Fails if any page in the range is uncommitted.
    bool protect(uint64_t addr,
                 uint64_t size,
                 PageProtect new_protect,
                 PageProtect* old_protect = nullptr);

    // Release an entire reservation by its exact base VA.
    bool release(uint64_t base_addr);

    // Unified Win32/Xbox VirtualAlloc wrapper (kMemReserve, kMemCommit, or both).
    uint64_t virtual_alloc(uint64_t addr,
                           uint64_t size,
                           uint32_t alloc_type,
                           PageProtect protect);

    // Unified Win32/Xbox VirtualFree wrapper (kMemDecommit or kMemRelease).
    bool virtual_free(uint64_t addr, uint64_t size, uint32_t free_type);

    // Query state and protection of the region starting at addr (Win32 VirtualQuery).
    MemoryBasicInfo query(uint64_t addr) const;

    // Safe bounds- and protection-checked memory access across committed extents.
    bool read(uint64_t addr, void* dst, size_t len) const;
    bool write(uint64_t addr, const void* src, size_t len);

    // Direct host pointer lookup when [addr, addr + len) resides in a single committed
    // extent with appropriate permissions. Returns nullptr otherwise.
    uint8_t* host_ptr(uint64_t addr, size_t len = 1);
    const uint8_t* host_ptr_const(uint64_t addr, size_t len = 1) const;

    uint64_t reserved_bytes() const;
    uint64_t committed_bytes() const;
    size_t reservation_count() const;

private:
    struct PageMeta {
        MemState state = MemState::Reserve;
        PageProtect protect = PageProtect::NoAccess;
    };

    struct CommittedExtent {
        uint64_t offset = 0;  // Byte offset from Reservation::base (page-aligned)
        uint64_t size = 0;    // Byte length (page-aligned, > 0)
        std::unique_ptr<uint8_t[]> data;
    };

    struct Reservation {
        uint64_t base = 0;
        uint64_t size = 0;              // Page-aligned reserved size
        uint64_t granularity_size = 0;  // 64 KiB-aligned VA slot occupancy
        PageProtect allocation_protect = PageProtect::NoAccess;
        std::vector<PageMeta> pages;
        std::vector<CommittedExtent> extents;  // Sorted by offset, non-overlapping
    };

    uint64_t reserve_locked(uint64_t hint_addr,
                            uint64_t size,
                            PageProtect allocation_protect);
    bool commit_locked(uint64_t addr, uint64_t size, PageProtect protect);
    bool decommit_locked(uint64_t addr, uint64_t size);
    bool release_locked(uint64_t base_addr);

    Reservation* find_reservation_locked(uint64_t addr);
    const Reservation* find_reservation_locked(uint64_t addr) const;
    bool range_overlaps_locked(uint64_t start, uint64_t end) const;
    uint64_t find_free_va_locked(uint64_t granularity_size) const;

    static bool is_valid_protect(PageProtect protect);

    uint64_t default_base_ = kDefaultArenaBase;
    mutable std::mutex mutex_;
    std::map<uint64_t, Reservation> reservations_;
    uint64_t reserved_bytes_ = 0;
    uint64_t committed_bytes_ = 0;
};

}  // namespace playgta6::os
