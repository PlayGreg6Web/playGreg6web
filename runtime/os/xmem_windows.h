#pragma once

// XMem CPU/GPU windows (task O27): the large physical-page windows the title reserves, their
// CPU <-> GPU alias, the GPU-side page mappings and the placement policy that decides which
// windows need guest address space at all.
//
// What the title does (static analysis of the executable, local only; numbers and offsets only)
// -------------------------------------------------------------------------------------------
// * Console profile. One startup routine reads an OS console-mode registry DWORD (RegOpenKeyW /
//   RegQueryValueExW under HKLM) and maps it to a device type: 0x60 -> Series X (type 6), 0x64 /
//   0x68 -> development kits (7 / 8), key missing -> 1. Types other than 1, 6, 7, 8 make startup
//   fail (the Series S value 0x58 included). Every
//   memory size below is a constant chosen by "type == 6" (Series X) versus anything else; no size
//   is derived from XMemGetWorkingSetStatistics, GlobalMemoryStatusEx or the GPU.
// * Windows. Right after D3D12ConfigureVirtualMemoryX (whose 32-byte block carries
//   2 + ceil(S / 16 GiB)) the title reserves two windows of S bytes with XMemVirtualAlloc
//   (address null, MEM_RESERVE | MEM_64K_PAGES (0x20402000), no commit):
//     CPU window: XMem flags 0x21, PAGE_READWRITE (0x4);
//     GPU window: XMem flags 0x3, PAGE_GRAPHICS_NOACCESS | PAGE_NOACCESS (0x801).
//   S = 16 GiB on Series X, 24 GiB otherwise (0x6'0000'0000, what the native boot sees with no
//   registry value). A null result traps (int3) in both cases.
// * Page pool. A streaming page pool of P bytes (P = 0xF890'0000, 3.88 GiB, on Series X;
//   0x1'85C0'0000, 6.09 GiB, otherwise) is allocated with XMemAllocatePhysicalPages
//   (attributes, &count, pfns), mapped once into its own XMemVirtualAlloc view (flags 0x23,
//   PAGE_READWRITE | PAGE_GRAPHICS_READWRITE) and registered with the D3D12 device (Xbox slot 74).
//   The view's address is otherwise only compared against zero by guest code.
// * Streaming. Pool pages are mapped at window page k into the CPU window with
//   XMemMapPhysicalPagesScatter (VA array, count, pfns; pfn 0 / null array unmaps) and into the
//   GPU window, at the same page k, through queue slot 31 (d3d12_xs_xbox.h). The title converts
//   CPU-window addresses to GPU-window addresses itself (48 functions read the window bases for
//   such conversions) before handing them to D3D12, so D3D12 sees GPU-window addresses and the
//   CPU reads/writes the CPU window. The GPU window is PAGE_NOACCESS for the CPU on the console: a CPU access faults there.
// * Other memory (committed, fully backed, sizes are constants of the same profile switch):
//   heap budgets of 1.03 GiB + 96 + 100 + 80 + 16 + 16 + 8 + 1 MiB on Series X (1.81 GiB +
//   246 + 200 + ... MiB otherwise), a 220 MiB physical-page heap (Series X only), a placeholder
//   region split with VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) and backed by physical
//   pages (descriptor memory, sized from configuration), and smaller XMemVirtualAlloc commits
//   (e.g. 20 MiB). Peak dynamic use (XMemAlloc heaps, D3D12 resources) is not derivable statically.
//
// The model
// ---------
// * Recognition: an XMemVirtualAlloc reservation (no commit) with the MEM_PHYSICAL bit and at
//   least kMinWindowBytes is a window; CPU-inaccessible protection (PAGE_NOACCESS in the low byte)
//   makes it a GPU window, anything else a CPU window. A GPU window of the same size as an unpaired
//   CPU window (either order) forms an alias pair: GPU VA G + x <-> CPU VA C + x.
// * Alias wiring: listeners (the D3D12 device, Device::register_gpu_alias) receive every pair,
//   including pairs that existed before they subscribed.
// * GPU page table: queue slot 31 updates (map / unmap of pool pages at GPU-window pages) are
//   applied to a per-64 KiB-page table; D3D12 reads a GPU-window page from the CPU window page with
//   the same index, which holds the same pool page by construction (both windows use index k).
// * One copy of the bytes: a pool page's bytes live in exactly one guest location (its "home": the
//   CPU-window page it is mapped at, otherwise a store; os/xbox_scarlett.cpp). Phantom ranges are
//   never homes, so mapping a page into the GPU window or another phantom range moves nothing.
//
// Placement policy (Deliverable 3) - decision and rejected alternatives
// ----------------------------------------------------------------------
// Browser budget: wasm64 linear memory is at most 16 GiB; the runtime owns [0, 4 GiB) and the
// identity-mapped guest pool is [4 GiB, 16 GiB) = 12 GiB, with the image at 5 GiB.
// Demand on the smallest profile (Series X): CPU window 16 GiB + GPU window 16 GiB of address
// space, plus 3.88 GiB pool view, plus >= 1.6 GiB of committed fixed budgets and the image.
// * Phantom GPU windows (CHOSEN, PlacementPolicy::PhantomGpuWindows, the wasm default): a GPU
//   window is never dereferenced by CPU code (PAGE_NOACCESS on the console), so it needs no guest
//   address space. It is placed at kPhantomBase (above every arena, below the D3D12 fake GPU VA
//   range 0x7E00'0000'0000, far from the synthetic thunk addresses 0x5047'...), tracked here for
//   VirtualQuery. Risk: a stray CPU access faults on native identity arenas (unmapped) and traps
//   on wasm (beyond the 16 GiB memory); on a native arena with a non-zero guest base the address
//   translates to arbitrary host memory (guest_to_host rejects phantom ranges, lifted code cannot).
// * The CPU window does NOT fit: it is dereferenced (the title writes streaming data through it
//   and nothing proves otherwise), it must be addressable over its whole 16 GiB because the title
//   picks the window page index of each mapping, and 16 GiB > 12 GiB pool. Rejected options:
//   (i) report less memory - the only lever is the console type; Series X already gives the
//   smallest windows (16 GiB), other types give 24 GiB, the Series S value is refused by the title;
//   (ii) a bigger pool - the hard cap is 16 GiB of linear memory; even with the runtime squeezed to
//   1 GiB the pool is 15 GiB < 16 GiB, and the image must stay at 5 GiB;
//   (iv) overlapping / truncated reservations - unsound without a bound on the window page indices
//   the title uses (it allocates them itself; the design ratio is window = 4 x pool).
// * What would work (design only, needs an [ABI] PR and lifter work): (iii) a second translation
//   for one high range in the lifter's guest_ptr seam - CPU-window VAs (placed above 16 GiB)
//   translate through a 64 KiB page table into pool pages allocated on map; every guest access
//   pays an unsigned compare and a rarely-taken branch. Cost and lifter changes are not measured
//   here; the decision is left to an ABI task.
// Native builds keep PlacementPolicy::Arena (all windows in the 64 GiB arena) unless configured.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace playgta6::os::xmem_windows {

inline constexpr uint64_t kPhantomBase = 0x0000'6000'0000'0000ull;
inline constexpr uint64_t kPhantomLimit = 0x0000'7000'0000'0000ull;
inline constexpr uint64_t kWindowPageSize = 64ull << 10;
inline constexpr uint64_t kMinWindowBytes = 1ull << 30;
inline constexpr uint32_t kMemPhysical = 0x0040'0000u;  // MEM_PHYSICAL bit of the allocation type

enum class WindowKind : uint8_t { Cpu, Gpu };

enum class PlacementPolicy : uint8_t {
    Arena,              // every window is an arena reservation (native default)
    PhantomGpuWindows,  // GPU windows live in the phantom space (wasm default)
};

struct Window {
    uint64_t base = 0;
    uint64_t size = 0;
    WindowKind kind = WindowKind::Cpu;
    uint32_t protect = 0;
    bool phantom = false;
};

struct AliasPair {
    uint64_t cpu_base = 0;
    uint64_t gpu_base = 0;
    uint64_t size = 0;
};

// Window classification of an XMemVirtualAlloc request; false if it is not a window.
bool classify(uint32_t allocation_type, uint32_t protect, uint64_t size, WindowKind* kind);

PlacementPolicy policy();
void set_policy(PlacementPolicy policy);
// True if a window of this kind is placed in the phantom space under the current policy.
bool place_phantom(WindowKind kind);

// Phantom space: 64 KiB-aligned bump allocation (never reused); 0 when exhausted.
uint64_t reserve_phantom(uint64_t size);
bool is_phantom(uint64_t va, uint64_t len = 1);

// Window bookkeeping (called by the XMemVirtualAlloc shim after placement).
void note_window(const Window& window);
// Forgets a window (and its pair) on release; true if `base` was a window.
bool forget_window(uint64_t base);
std::vector<Window> windows();
std::vector<AliasPair> alias_pairs();
// Window containing [va, va + len), if any.
bool find_window(uint64_t va, uint64_t len, Window* out);

using AliasListener = std::function<void(const AliasPair&)>;
// Subscribes `listener` (called at once for every existing pair, then for each new pair, never
// with the internal lock held). Returns an id for remove_alias_listener.
uint64_t add_alias_listener(AliasListener listener);
void remove_alias_listener(uint64_t id);

// GPU page table (queue slot 31). `kind` 1 maps `count` pool pages starting at `pool_page` to
// consecutive GPU pages, 2 unmaps `count` GPU pages; other kinds are counted and skipped.
struct PageMappingEntry {
    uint32_t kind = 0;
    uint32_t pool_page = 0;
    uint32_t count = 0;
};
struct PageMappingRange {
    uint64_t gpu_va = 0;
    std::vector<PageMappingEntry> entries;
};
struct PageMappingResult {
    uint64_t mapped_pages = 0;
    uint64_t unmapped_pages = 0;
    uint64_t skipped_entries = 0;   // unknown kinds
    uint64_t outside_window = 0;    // pages not inside a GPU window
};
PageMappingResult apply_gpu_page_mappings(std::span<const PageMappingRange> ranges, uint64_t pool_va, uint64_t pool_pages);
// Pool page mapped at the GPU page containing `gpu_va`, or -1.
int64_t gpu_page_pool_index(uint64_t gpu_va);

// CPU-window residency (pages mapped into CPU windows by xmem).
void note_cpu_window_mapping(int64_t delta_bytes);

struct Accounting {
    uint64_t cpu_windows = 0, gpu_windows = 0, alias_pairs = 0;
    uint64_t cpu_window_reserved = 0, gpu_window_reserved = 0;
    uint64_t phantom_reserved = 0;
    uint64_t cpu_window_mapped = 0;  // bytes of pool pages mapped into CPU windows
    uint64_t gpu_window_mapped = 0;  // bytes mapped into GPU windows (queue slot 31)
    uint64_t gpu_mapping_updates = 0;
};
Accounting accounting();
std::string describe();  // one-line summary for diagnostics

// Budget planner: would this reservation sequence fit an identity guest pool of `pool_bytes` under
// `policy`? Reservations are placed first-fit in address order after `fixed_bytes` already used
// (image, runtime-owned guest memory); phantom-placed windows cost nothing.
struct PlannedReservation {
    uint64_t size = 0;
    uint32_t allocation_type = 0;
    uint32_t protect = 0;
};
struct FitResult {
    bool fits = false;
    uint64_t identity_bytes = 0;   // address space taken in the pool
    uint64_t phantom_bytes = 0;    // address space placed in the phantom space
    size_t first_failure = SIZE_MAX;
};
FitResult plan_fit(std::span<const PlannedReservation> sequence, uint64_t pool_bytes, uint64_t fixed_bytes,
                   PlacementPolicy policy);

void reset_for_testing();

}  // namespace playgta6::os::xmem_windows
