#pragma once

#include <cstdint>

#include "os/vmem.h"

// Internal to VirtualMemoryArena (vmem.cpp): the per-platform host backing implemented in
// vmem_backing.cpp. All ranges are page aligned and lie inside one reservation.
namespace playgta6::os::backing {

// Reserves `size` bytes of inaccessible address space; with `preferred` != 0 only exactly at
// that host address. nullptr on failure.
uint8_t* reserve(uint64_t size, uint64_t preferred);
void release(uint8_t* base, uint64_t size);
// Makes reserved (never or no longer committed) pages usable; they read as zero.
bool commit(uint8_t* p, uint64_t len, PageProtect protect);
// Gives committed pages back (where the platform can); the range may include uncommitted pages.
bool decommit(uint8_t* p, uint64_t len);
// Applies a protection to committed pages (no-op where protections cannot be enforced).
bool protect(uint8_t* p, uint64_t len, PageProtect protect);
bool enforces_protection();

}  // namespace playgta6::os::backing
