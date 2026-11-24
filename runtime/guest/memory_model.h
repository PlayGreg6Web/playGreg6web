#pragma once

// Guest memory model of a lifted program (task O29): how its lifted code translates guest
// addresses, recorded by the lifter (`playgta6-recomp build` / `build-program --memory-model`,
// recomp/src/lift/memory.rs) in the build manifests and in the generated program descriptor
// (Pg6Program::memory, guest/program_loader.h), and served by the boot (BootConfig::memory,
// guest/boot.h).
//
// Models
// ------
// * Flat (the default; no descriptor record): host = __playgta6_guest_base + VA everywhere.
// * Windowed: the translated CPU window of docs/wasm_abi.md "Translated CPU window" - guest VAs in
//   [kGuestWindowBase, kGuestWindowBase + kGuestWindowSize) translate through
//   __playgta6_window_table (os/guest_window.h), everything else as Flat. The boot turns the
//   window on, places the title's CPU window there (xmem_windows::PlacementPolicy::
//   PhantomGpuTranslatedCpu: GPU windows phantom, the first CPU window translated) and keeps the
//   arena out of the window range (a native arena that spans it gets one no-access reservation
//   over it, so no other allocation - a guest stack, a heap - can land there and host code that
//   still dereferences a window VA flat faults instead of touching unrelated memory).
//   Flags: kPg6MemoryStackFlat - rsp-relative accesses skip the window check (sound because
//   guest stacks are arena allocations, never in the window); kPg6MemoryContiguousRuns - no
//   straddle check, which needs adjacent mapped slots to be host-contiguous: the boot then has
//   XMem keep window runs contiguous (window-centric placement, os/xbox_scarlett.cpp
//   keep_window_run_locked: a page's bytes move next to its window neighbour's, merging runs
//   copies) and stops the guest with a fault at the first slot it cannot keep contiguous (a page
//   mapped at two adjacent slots, for one), so lifted code never reads the wrong bytes.
// * PageTable: a lifter prototype (every VA through a page table); no runtime support.
//
// A runtime build serves a model only if it can do everything the lifted code relies on;
// memory_model_unsupported() names the first thing it cannot do, and the program loader
// (check_program) and the boot (run_boot) refuse such a program with that reason.

#include <cstdint>
#include <string>

#include "os/vmem.h"

extern "C" {

// Pg6GuestMemoryModel::model
inline constexpr uint32_t kPg6MemoryModelFlat = 0u;
inline constexpr uint32_t kPg6MemoryModelWindowed = 1u;
inline constexpr uint32_t kPg6MemoryModelPageTable = 2u;

// Pg6GuestMemoryModel::flags
inline constexpr uint32_t kPg6MemoryStackFlat = 1u << 0;       // rsp-relative accesses flat
inline constexpr uint32_t kPg6MemoryContiguousRuns = 1u << 1;  // no straddle checks
inline constexpr uint32_t kPg6MemoryFlatConstant = 1u << 2;    // flat base is `flat_base`, not
                                                               // __playgta6_guest_base
inline constexpr uint32_t kPg6MemoryKnownFlags =
    kPg6MemoryStackFlat | kPg6MemoryContiguousRuns | kPg6MemoryFlatConstant;

// Pg6GuestMemoryModel::check (code shape of the window check; the runtime serves all three)
inline constexpr uint32_t kPg6MemoryCheckBranch = 0u;
inline constexpr uint32_t kPg6MemoryCheckSelect = 1u;
inline constexpr uint32_t kPg6MemoryCheckOutline = 2u;

// All zero = the Flat default.
struct Pg6GuestMemoryModel {
    uint32_t model;        // kPg6MemoryModel*
    uint32_t flags;        // kPg6Memory* flags
    uint32_t check;        // kPg6MemoryCheck* (Windowed)
    uint32_t reserved;     // 0
    uint64_t window_base;  // Windowed: first window VA; PageTable: 0
    uint64_t window_size;  // Windowed: window bytes; PageTable: number of 64 KiB slots
    uint64_t flat_base;    // with kPg6MemoryFlatConstant: the constant host - guest offset
};

}  // extern "C"

namespace playgta6::guest {

inline bool is_default_memory_model(const Pg6GuestMemoryModel& m) {
    return m.model == kPg6MemoryModelFlat && m.flags == 0;
}

// One-line description, e.g. "windowed [0x400000000, 0xa00000000) check=branch stack-flat".
std::string describe_memory_model(const Pg6GuestMemoryModel& model);

// Why this runtime build cannot run code lifted with `model` (empty: it can). With an arena the
// checks that depend on the guest address space run too.
std::string memory_model_unsupported(const Pg6GuestMemoryModel& model,
                                     const os::VirtualMemoryArena* arena = nullptr);

// Sets up what `model` needs in `arena` (see the header comment); a no-op for the default.
// Returns false with `error` set if the model cannot be served. Undone by release_memory_model
// (reset_boot calls it).
bool apply_memory_model(const Pg6GuestMemoryModel& model, os::VirtualMemoryArena& arena,
                        std::string& error);
void release_memory_model();

}  // namespace playgta6::guest
