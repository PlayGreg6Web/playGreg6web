#pragma once

// Backlog 29: the "where did the memory go" map of the boot report. Pure bookkeeping: the boot
// gathers the numbers from the arena, the guest heap, XMem and the GPU side and this module sums
// them by kind and renders one JSON object. No dependencies besides the standard library, so the
// composition is unit-tested without a boot.
//
// Kinds (bytes are host memory the run holds right now, not reservations):
//   guest_arena    committed pages of the guest address-space arena (everything the guest can
//                  address: image, XMem pools, heaps, stacks); the largest term by far
//     guest_heap     ... of which blocks of the process heap (CRT / XMem / other families)
//     xmem_*         ... XMem: heap blocks, physical pages by where their bytes live, virtual commits
//   host_shadow    host copies of CPU-visible buffers kept to find what changed (resident chunks;
//                  the covered size is what an eager shadow would have cost)
//   gpu_textures   host GPU textures held against the texture budget (video memory, not heap)
//   gpu_parked     ... of which recycled small textures parked in the pool (still counted by the budget)
//
// Browser (wasm64) limit: linear memory is at most 16 GiB for the runtime (below 4 GiB), the
// identity-mapped guest pool (4..16 GiB) and every host-side buffer that lives in the wasm heap
// (shadows, staging). `wasm_headroom` is limit minus the bytes the wasm heap would hold: arena
// committed + host shadows (the GPU textures live in the browser's GPU process, not in wasm).
// Blockers recorded where they live (os/xmem_windows.h, "Placement policy"): the 16 GiB CPU
// window cannot be reserved whole inside 16 GiB of linear memory next to the 3.88 GiB pool view,
// so the window must stay translated and sparse; the memory the world needs (15-46 GB private on
// the reference emulators) exceeds a 16 GiB wasm memory unless render targets/depth stay GPU-only
// and physical-page homes commit on first touch (both are the XMem "unbacked" counters below).

#include <cstdint>
#include <string>

namespace playgta6::os {

inline constexpr uint64_t kWasmLinearMemoryLimit = uint64_t{16} << 30;

struct MemoryBudgetInput {
    uint64_t arena_reserved = 0;
    uint64_t arena_committed = 0;
    uint64_t guest_heap_committed = 0;
    uint64_t guest_heap_live = 0;
    // XMem (XMemMemorySummary)
    uint64_t xmem_heap_bytes = 0;
    uint64_t xmem_virtual_committed = 0;
    uint64_t xmem_pages_nominal = 0;
    uint64_t xmem_pages_resident = 0;
    uint64_t xmem_pages_store = 0;
    uint64_t xmem_pages_unbacked = 0;
    uint64_t xmem_owned_blocks = 0;
    // GPU side
    uint64_t shadow_covered = 0;
    uint64_t shadow_resident = 0;
    uint64_t gpu_texture_bytes = 0;
    uint64_t gpu_parked_bytes = 0;
};

struct MemoryBudgetTotals {
    uint64_t host_private = 0;  // arena committed + host shadows: heap the process holds
    uint64_t gpu_visible = 0;   // textures held against the budget (parked ones included)
    uint64_t wasm_heap = 0;     // what a browser build would hold in linear memory
    int64_t wasm_headroom = 0;  // limit - wasm_heap (negative: over the limit)
    // Nominal page bytes the title asked for that no host memory backs yet.
    uint64_t xmem_pages_saved = 0;
};

inline MemoryBudgetTotals memory_budget_totals(const MemoryBudgetInput& in) {
    MemoryBudgetTotals t;
    t.host_private = in.arena_committed + in.shadow_resident;
    t.gpu_visible = in.gpu_texture_bytes;  // parked textures stay counted in the budget
    t.wasm_heap = t.host_private;
    t.wasm_headroom = static_cast<int64_t>(kWasmLinearMemoryLimit) - static_cast<int64_t>(t.wasm_heap);
    t.xmem_pages_saved =
        in.xmem_pages_nominal > in.xmem_pages_resident ? in.xmem_pages_nominal - in.xmem_pages_resident : 0;
    return t;
}

// One JSON object (no trailing comma/newline), keys in a fixed order.
inline std::string memory_budget_json(const MemoryBudgetInput& in) {
    const MemoryBudgetTotals t = memory_budget_totals(in);
    auto n = [](uint64_t v) { return std::to_string(v); };
    std::string o = "{\"host_private_bytes\": " + n(t.host_private) + ", \"gpu_visible_bytes\": " + n(t.gpu_visible) +
                    ", \"guest_arena\": {\"reserved\": " + n(in.arena_reserved) +
                    ", \"committed\": " + n(in.arena_committed) + ", \"guest_heap_committed\": " +
                    n(in.guest_heap_committed) + ", \"guest_heap_live\": " + n(in.guest_heap_live) +
                    "}, \"xmem\": {\"heap_bytes\": " + n(in.xmem_heap_bytes) +
                    ", \"virtual_committed\": " + n(in.xmem_virtual_committed) +
                    ", \"pages_nominal\": " + n(in.xmem_pages_nominal) +
                    ", \"pages_resident\": " + n(in.xmem_pages_resident) +
                    ", \"pages_in_store\": " + n(in.xmem_pages_store) +
                    ", \"pages_unbacked\": " + n(in.xmem_pages_unbacked) +
                    ", \"pages_not_backed_bytes\": " + n(t.xmem_pages_saved) +
                    ", \"owned_blocks\": " + n(in.xmem_owned_blocks) +
                    "}, \"host_shadow\": {\"covered\": " + n(in.shadow_covered) +
                    ", \"resident\": " + n(in.shadow_resident) + "}, \"gpu\": {\"textures\": " +
                    n(in.gpu_texture_bytes) + ", \"parked_textures\": " + n(in.gpu_parked_bytes) +
                    "}, \"wasm\": {\"limit\": " + n(kWasmLinearMemoryLimit) + ", \"heap\": " + n(t.wasm_heap) +
                    ", \"headroom\": " + std::to_string(t.wasm_headroom) + "}}";
    return o;
}

}  // namespace playgta6::os
