// Residency planning and fault service for the paged RT arena (docs/RAYTRACING_DESIGN.md sections 3, 5,
// contract C2). Pure bookkeeping on top of gpu/rt_arena.h:
//  - AddressSource: the reflection `address_sources` of a program (which root params / root constants /
//    user-data dwords hold guest addresses, and which of them are BVH roots) and how to resolve them
//    against a dispatch's bound state into the guest ranges that have to be resident.
//  - BVH roots: the whole structure (header dw1 = size) and, through a TLAS, the structures its
//    instances reach. Needs the CPU to read the header/instances; pages a builder wrote on the GPU
//    cannot be walked -> skip reason structures_unavailable.
//  - FaultTracker: the GPU logs non-resident accesses in pg6_rt_stats (word 0 = count, word 1 = log
//    attempts, words 2..31 = page + 1); the tracker turns a readback into pages made resident for the
//    next dispatch, counted.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "gpu/rt_arena.h"

namespace playgta6::gpu::rt_residency {

// C2 reflection field (per program), JSON array:
//   "address_sources": [
//     {"kind": "root_param",     "index": 3, "extent": 65536},               // root CBV/SRV/UAV: 64-bit GPU VA
//     {"kind": "root_constants", "index": 0, "dword": 2, "bvh_root": true},  // lo = dword, hi = dword + 1
//     {"kind": "user_data",      "index": 4, "extent": 256}                  // lo = user-data dword, hi = index + 1
//   ]
// `extent` (bytes, default 1 = just the page of the address) is the range made resident from the
// address. `bvh_root` marks the address as the header of a BVH structure: the whole structure and,
// for a TLAS, every BLAS its instances name are made resident.
enum class SourceKind : uint8_t { RootParam, RootConstants, UserData };

struct AddressSource {
    SourceKind kind = SourceKind::RootParam;
    uint32_t index = 0;
    uint32_t dword = 0;  // root_constants only
    uint64_t extent = 1;
    bool bvh_root = false;
};

bool parse_address_sources(const std::string& json, std::vector<AddressSource>* out, std::string* error);

// Why a source did not resolve to an address (counted per reason in Plan::unresolved_by_reason).
enum class Unresolved : uint8_t { NoInput, Unbound, Null, DescriptorTable, OutOfRange, Count };
const char* unresolved_name(Unresolved r);

// Bound state of one dispatch. Any callback may be empty (the source is then unresolved).
struct DispatchInputs {
    std::function<bool(uint32_t index, uint64_t* va)> root_param;
    std::function<bool(uint32_t index, uint32_t dword, uint32_t* value)> root_constant;
    std::function<bool(uint32_t index, uint32_t* value)> user_data;
    // Optional: why user_data(index) failed (default Unbound).
    std::function<Unresolved(uint32_t index)> user_data_failure;
    // Guest memory read (bytes at a 40-bit address).
    std::function<bool(uint64_t addr, void* dst, size_t bytes)> read_guest;
};

struct Plan {
    std::vector<std::pair<uint64_t, uint64_t>> ranges;  // (address, size) to make resident
    uint32_t unresolved = 0;                            // sources the dispatch did not bind (or null)
    uint32_t unresolved_by_reason[static_cast<size_t>(Unresolved::Count)] = {};
    uint32_t structures = 0;                            // BVH structures walked
    uint32_t bad_headers = 0;                           // bvh_root addresses without a valid header
    bool structures_unavailable = false;                // GPU-written or unreadable structure
};

constexpr uint32_t kMaxInstances = 1u << 20;
constexpr uint64_t kMaxStructureBytes = 1ull << 30;

// Resolves `sources` against `in` (and walks BVH roots) into the ranges the dispatch needs. `arena`
// (optional) tells which pages are GPU-owned.
Plan plan_dispatch(const std::vector<AddressSource>& sources, const DispatchInputs& in, const rt_arena::Arena* arena);

struct Outcome {
    bool ok = true;
    rt_arena::SkipReason reason = rt_arena::SkipReason::Count;  // valid when !ok (already counted)
};

// plan_dispatch + ensure on the arena (begin_dispatch is the caller's). On failure the skip is counted
// on the arena and the dispatch must not be recorded.
Outcome prepare_dispatch(rt_arena::Arena& arena, const std::vector<AddressSource>& sources, const DispatchInputs& in,
                         Plan* plan_out = nullptr);

constexpr uint32_t kFaultLogEntries = 30;
constexpr uint32_t kStatsWords = 32;  // words of pg6_rt_stats the tracker reads

struct FaultStats {
    uint64_t faults = 0;                  // total counted by the GPU
    uint64_t pages_serviced = 0;          // newly made resident because of a fault
    uint64_t pages_already_resident = 0;  // logged, but resident by the time we looked (stale directory)
    uint64_t pages_unserviceable = 0;     // arena full, table full or out of range
    uint64_t unlogged = 0;                // faults whose page the 30-entry log could not hold
    uint64_t readbacks = 0;
};

class FaultTracker {
public:
    // `words` = the first kStatsWords dwords of pg6_rt_stats read back. Makes the logged pages resident
    // (they upload with the next dispatch's prepare) and updates the arena's rt_fault. Returns true when
    // the caller must zero words 1..31 (never word 0, the monotonic counter) before the next frame.
    bool consume(const uint32_t* words, size_t count, rt_arena::Arena& arena);
    const FaultStats& stats() const { return stats_; }
    std::string json() const;

private:
    uint32_t last_count_ = 0;
    FaultStats stats_;
};

}  // namespace playgta6::gpu::rt_residency
