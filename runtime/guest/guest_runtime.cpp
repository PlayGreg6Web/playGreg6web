#include "guest/guest_runtime.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include "os/os_shim.h"
#include "os/win32_kernel.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
// Keep the helpers in the link and export them, so lifted code instantiated as a separate module
// can import them from the runtime instance as well as link against them statically.
#define PG6_GUEST_HELPER EMSCRIPTEN_KEEPALIVE
#else
#define PG6_GUEST_HELPER
#endif

namespace playgta6::guest {

namespace {

constexpr size_t kChunkBits = 12;
constexpr size_t kChunkSize = size_t{1} << kChunkBits;
constexpr size_t kMaxChunks = static_cast<size_t>(kMaxSyntheticTargets / kChunkSize);

struct LiftedTable {
    const LiftedFunctionEntry* entries = nullptr;
    size_t count = 0;
    uint64_t lo = 0;
    uint64_t hi = 0;
};

struct SyntheticEntry {
    HostThunkFn thunk = nullptr;
    uint64_t image_base = 0;
    uint32_t slot_rva = 0;
    bool is_import = false;
    const char* name = nullptr;
};

// Readers (dispatch) only touch the atomics and the data they publish; everything else is
// guarded by `mutex`.
struct Registry {
    std::mutex mutex;

    LiftedTable tables[kMaxLiftedTables];
    std::atomic<size_t> table_count{0};
    std::atomic<size_t> function_count{0};

    // Published immutable snapshot for lock-free O(log N) zero-allocation shard dispatch.
    std::atomic<const ShardDispatchRegistry*> shard_registry{nullptr};
    std::vector<std::unique_ptr<ShardDispatchRegistry>> owned_shard_registries;
    std::atomic<size_t> shard_function_count{0};
    std::atomic<size_t> shard_count{0};

    std::atomic<SyntheticEntry*> chunks[kMaxChunks];
    std::vector<std::unique_ptr<SyntheticEntry[]>> owned_chunks;
    std::atomic<uint64_t> synthetic_count{1};  // index 0 is kGuestReturnSentinel

    std::deque<std::string> names;
    std::unordered_map<uint64_t, uint64_t> proc_to_synthetic;

    std::atomic<FaultHandler> fault_handler{nullptr};
};

Registry& registry() {
    static Registry instance;
    return instance;
}

std::atomic<DispatchObserver> g_dispatch_observer{nullptr};

// ---- Dispatch cache ------------------------------------------------------------------------
// A per-thread direct-mapped cache guest VA -> resolved lifted function or shard dispatcher in
// front of the registry lookup. Only resolved lifted targets are cached (no misses, no synthetic
// addresses: those resolve in O(1) already), so unknown targets still reach the slow path and
// raise the same fault. Every registry change bumps g_dispatch_generation after publishing; an
// entry is valid only for the generation it was filled under (read before the lookup), so no
// thread ever uses an entry that a later registration could shadow. The hit path is one acquire
// load of the generation plus thread-local memory: no locks, no shared writes.

std::atomic<uint64_t> g_dispatch_generation{1};  // 0 marks an empty cache entry

enum DispatchCacheKind : uint32_t { kCachedLifted = 1, kCachedShard = 2 };

struct alignas(32) DispatchCacheEntry {
    uint64_t va = 0;
    uint64_t generation = 0;
    union {
        LiftedFn lifted;
        Pg6ShardDispatchFn shard;
    } fn{nullptr};
    uint32_t local_idx = 0;
    uint32_t kind = 0;
};
static_assert(sizeof(DispatchCacheEntry) == 32, "two entries per cache line");

constexpr unsigned kDispatchCacheBits = 12;  // 4096 entries, 128 KiB per dispatching thread
constexpr size_t kDispatchCacheSize = size_t{1} << kDispatchCacheBits;

struct DispatchCache {
    DispatchCacheEntry entries[kDispatchCacheSize];
};

// Plain pointer (no TLS init guard on the hit path); owned by a thread_local that frees it.
thread_local DispatchCache* t_dispatch_cache = nullptr;

struct DispatchCacheOwner {
    DispatchCache* cache = nullptr;
    bool destroyed = false;
    ~DispatchCacheOwner() {
        t_dispatch_cache = nullptr;
        destroyed = true;
        delete cache;
        cache = nullptr;
    }
};

inline size_t dispatch_cache_index(uint64_t va) {
    return static_cast<size_t>((va * 0x9E3779B97F4A7C15ULL) >> (64 - kDispatchCacheBits));
}

// The calling thread's cache (allocated on first use), or nullptr if it cannot have one.
DispatchCache* dispatch_cache_for_fill() {
    if (DispatchCache* cache = t_dispatch_cache) {
        return cache;
    }
    thread_local DispatchCacheOwner owner;
    if (owner.destroyed) {
        return nullptr;  // dispatch from a later thread_local destructor: stay uncached
    }
    if (owner.cache == nullptr) {
        owner.cache = new (std::nothrow) DispatchCache{};
    }
    t_dispatch_cache = owner.cache;
    return owner.cache;
}

std::atomic<bool> g_dispatch_cache_enabled{true};

void fill_dispatch_cache(uint64_t va, uint64_t generation, LiftedFn lifted, Pg6ShardDispatchFn shard,
                         uint32_t local_idx) {
    if (!g_dispatch_cache_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    DispatchCache* cache = dispatch_cache_for_fill();
    if (cache == nullptr) {
        return;
    }
    DispatchCacheEntry& e = cache->entries[dispatch_cache_index(va)];
    e.va = va;
    e.generation = generation;
    if (lifted != nullptr) {
        e.fn.lifted = lifted;
        e.kind = kCachedLifted;
    } else {
        e.fn.shard = shard;
        e.kind = kCachedShard;
    }
    e.local_idx = local_idx;
}

// Called after a registry change is published (under the registry mutex).
void invalidate_dispatch_caches() {
    g_dispatch_generation.fetch_add(1, std::memory_order_release);
}

const LiftedFunctionEntry* find_entry(const LiftedTable& table, uint64_t va) {
    const LiftedFunctionEntry* end = table.entries + table.count;
    const LiftedFunctionEntry* it = std::lower_bound(
        table.entries, end, va, [](const LiftedFunctionEntry& e, uint64_t v) { return e.va < v; });
    return (it != end && it->va == va) ? it : nullptr;
}

// Highest registered lifted entry <= va, or 0.
uint64_t nearest_entry_at_or_below(uint64_t va) {
    Registry& reg = registry();
    const size_t n = reg.table_count.load(std::memory_order_acquire);
    uint64_t best = 0;
    for (size_t i = 0; i < n; ++i) {
        const LiftedTable& t = reg.tables[i];
        if (va < t.lo) {
            continue;
        }
        const LiftedFunctionEntry* end = t.entries + t.count;
        const LiftedFunctionEntry* it = std::upper_bound(
            t.entries, end, va, [](uint64_t v, const LiftedFunctionEntry& e) { return v < e.va; });
        if (it != t.entries) {
            best = std::max(best, (it - 1)->va);
        }
    }
    if (const ShardDispatchRegistry* shards = reg.shard_registry.load(std::memory_order_acquire)) {
        uint64_t shard_best = 0;
        if (shards->nearest_entry(va, shard_best)) {
            best = std::max(best, shard_best);
        }
    }
    return best;
}

const SyntheticEntry* find_synthetic(uint64_t va) {
    if (!is_synthetic_address(va)) {
        return nullptr;
    }
    const uint64_t offset = va - kSyntheticBase;
    if (offset % kSyntheticStride != 0) {
        return nullptr;
    }
    const uint64_t index = offset / kSyntheticStride;
    Registry& reg = registry();
    if (index == 0 || index >= reg.synthetic_count.load(std::memory_order_acquire)) {
        return nullptr;
    }
    const SyntheticEntry* chunk =
        reg.chunks[static_cast<size_t>(index >> kChunkBits)].load(std::memory_order_acquire);
    if (chunk == nullptr) {
        return nullptr;
    }
    return &chunk[static_cast<size_t>(index & (kChunkSize - 1))];
}

// Reserves the next synthetic slot (not yet visible to readers). Returns 0 when full.
uint64_t reserve_synthetic_locked(Registry& reg, SyntheticEntry** slot) {
    const uint64_t index = reg.synthetic_count.load(std::memory_order_relaxed);
    if (index >= kMaxSyntheticTargets) {
        return 0;
    }
    const size_t chunk_index = static_cast<size_t>(index >> kChunkBits);
    SyntheticEntry* chunk = reg.chunks[chunk_index].load(std::memory_order_relaxed);
    if (chunk == nullptr) {
        reg.owned_chunks.push_back(std::make_unique<SyntheticEntry[]>(kChunkSize));
        chunk = reg.owned_chunks.back().get();
        reg.chunks[chunk_index].store(chunk, std::memory_order_release);
    }
    *slot = &chunk[static_cast<size_t>(index & (kChunkSize - 1))];
    return kSyntheticBase + index * kSyntheticStride;
}

void publish_synthetic_locked(Registry& reg) {
    reg.synthetic_count.store(reg.synthetic_count.load(std::memory_order_relaxed) + 1,
                              std::memory_order_release);
}

// Writes one 8-byte IAT slot, temporarily making a read-only page writable.
bool write_slot(os::VirtualMemoryArena& arena, uint64_t va, uint64_t value) {
    if (arena.write(va, &value, sizeof(value))) {
        return true;
    }
    os::PageProtect old_protect = os::PageProtect::NoAccess;
    if (!arena.protect(va, sizeof(value), os::PageProtect::ReadWrite, &old_protect)) {
        return false;
    }
    const bool ok = arena.write(va, &value, sizeof(value));
    arena.protect(va, sizeof(value), old_protect);
    return ok;
}

struct Formatter {
    char* buffer;
    size_t size;
    size_t length = 0;

    void append(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char* dst = length < size ? buffer + length : nullptr;
        const size_t room = length < size ? size - length : 0;
        const int n = std::vsnprintf(dst, room, fmt, args);
        va_end(args);
        if (n > 0) {
            length += static_cast<size_t>(n);
        }
    }
};

[[noreturn]] void default_fault_handler(const FaultInfo& info) {
    char message[1024];
    format_fault(info, message, sizeof(message));
    os::log(message);
    std::abort();
}

}  // namespace

const char* fault_kind_name(FaultKind kind) {
    switch (kind) {
        case FaultKind::Trap:
            return "trap";
        case FaultKind::Unimplemented:
            return "unimplemented instruction";
        case FaultKind::DivideError:
            return "divide error";
        case FaultKind::UnknownTarget:
            return "indirect call to unknown target";
        case FaultKind::StackImbalance:
            return "stack imbalance after guest call";
        case FaultKind::BadStack:
            return "guest stack not writable";
        case FaultKind::GeneralProtection:
            return "general protection fault (#GP)";
        case FaultKind::UnhandledException:
            return "unhandled guest exception";
        case FaultKind::UnknownResumeTarget:
            return "context restore to an unknown resume target";
        case FaultKind::UnwindFailed:
            return "guest unwind failed";
        case FaultKind::UnimplementedSlot:
            return "unimplemented COM interface slot";
    }
    return "unknown fault";
}

// ---------------------------------------------------------------------------
// Lifted functions
// ---------------------------------------------------------------------------

bool register_lifted_functions(std::span<const LiftedFunctionEntry> table) {
    if (table.empty()) {
        return true;
    }
    if (table.data() == nullptr) {
        return false;
    }
    for (size_t i = 0; i < table.size(); ++i) {
        if (table[i].fn == nullptr || is_synthetic_address(table[i].va)) {
            return false;
        }
        if (i > 0 && table[i - 1].va >= table[i].va) {
            return false;
        }
    }
    const uint64_t lo = table.front().va;
    const uint64_t hi = table.back().va;

    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    const size_t n = reg.table_count.load(std::memory_order_relaxed);
    if (n >= kMaxLiftedTables) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (lo <= reg.tables[i].hi && reg.tables[i].lo <= hi) {
            return false;
        }
    }
    reg.tables[n] = LiftedTable{table.data(), table.size(), lo, hi};
    reg.function_count.fetch_add(table.size(), std::memory_order_relaxed);
    reg.table_count.store(n + 1, std::memory_order_release);
    invalidate_dispatch_caches();
    return true;
}

LiftedFn find_lifted_function(uint64_t va) {
    Registry& reg = registry();
    const size_t n = reg.table_count.load(std::memory_order_acquire);
    for (size_t i = 0; i < n; ++i) {
        const LiftedTable& t = reg.tables[i];
        if (va < t.lo || va > t.hi) {
            continue;
        }
        const LiftedFunctionEntry* e = find_entry(t, va);
        return e != nullptr ? e->fn : nullptr;  // table ranges never overlap
    }
    return nullptr;
}

size_t lifted_function_count() {
    const Registry& reg = registry();
    return reg.function_count.load(std::memory_order_relaxed) +
           reg.shard_function_count.load(std::memory_order_relaxed);
}

bool register_shard_dispatch_table(const Pg6ShardDispatchTable& table, std::string* error) {
    if (table.entry_count > 0 && table.entries != nullptr) {
        for (uint64_t i = 0; i < table.entry_count; ++i) {
            if (is_synthetic_address(table.entries[i].guest_va)) {
                if (error != nullptr) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf),
                                  "entry %" PRIu64 " uses synthetic guest_va 0x%" PRIx64,
                                  i, table.entries[i].guest_va);
                    *error = buf;
                }
                return false;
            }
        }
    }

    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    const ShardDispatchRegistry* current = reg.shard_registry.load(std::memory_order_relaxed);
    auto next = current != nullptr ? std::make_unique<ShardDispatchRegistry>(*current)
                                   : std::make_unique<ShardDispatchRegistry>();
    if (!next->add_table(table, error)) {
        return false;
    }
    const ShardDispatchRegistry* published = next.get();
    reg.owned_shard_registries.push_back(std::move(next));
    reg.shard_function_count.store(published->entry_count(), std::memory_order_relaxed);
    reg.shard_count.store(published->shard_count(), std::memory_order_relaxed);
    reg.shard_registry.store(published, std::memory_order_release);
    invalidate_dispatch_caches();
    return true;
}

bool find_shard_dispatch_target(uint64_t va, ResolvedShardTarget& out) {
    const ShardDispatchRegistry* shards = registry().shard_registry.load(std::memory_order_acquire);
    if (shards == nullptr) {
        out = ResolvedShardTarget{};
        return false;
    }
    return shards->lookup(va, out);
}

uint64_t nearest_lifted_function_entry(uint64_t va) {
    return nearest_entry_at_or_below(va);
}

size_t shard_dispatch_function_count() {
    return registry().shard_function_count.load(std::memory_order_relaxed);
}

size_t shard_dispatch_shard_count() {
    return registry().shard_count.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Imports and host callables
// ---------------------------------------------------------------------------

size_t bind_import_slots(os::VirtualMemoryArena& arena,
                         uint64_t image_base,
                         std::span<const ImportSlotBinding> slots) {
    Registry& reg = registry();
    size_t bound = 0;
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (const ImportSlotBinding& slot : slots) {
        const uint64_t slot_va = image_base + slot.slot_rva;
        char message[160];
        if (slot.thunk == nullptr) {
            std::snprintf(message, sizeof(message),
                          "guest runtime: import slot rva 0x%" PRIx32 " has no thunk, left unbound",
                          slot.slot_rva);
            os::log(message);
            continue;
        }
        SyntheticEntry* entry = nullptr;
        const uint64_t address = reserve_synthetic_locked(reg, &entry);
        if (address == 0) {
            os::log("guest runtime: synthetic address table full, import slots left unbound");
            break;
        }
        if (slot_va < image_base || !write_slot(arena, slot_va, address)) {
            std::snprintf(message, sizeof(message),
                          "guest runtime: import slot 0x%016" PRIx64 " is not writable guest memory",
                          slot_va);
            os::log(message);
            continue;
        }
        *entry = SyntheticEntry{slot.thunk, image_base, slot.slot_rva, true, nullptr};
        publish_synthetic_locked(reg);
        ++bound;
    }
    if (bound != 0) {
        invalidate_dispatch_caches();
    }
    return bound;
}

uint64_t register_host_callable(HostThunkFn thunk, const char* name, uint64_t proc_address) {
    if (thunk == nullptr) {
        return 0;
    }
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    SyntheticEntry* entry = nullptr;
    const uint64_t address = reserve_synthetic_locked(reg, &entry);
    if (address == 0) {
        return 0;
    }
    const char* stored_name = nullptr;
    if (name != nullptr) {
        reg.names.emplace_back(name);
        stored_name = reg.names.back().c_str();
    }
    *entry = SyntheticEntry{thunk, 0, 0, false, stored_name};
    publish_synthetic_locked(reg);
    if (proc_address != 0) {
        reg.proc_to_synthetic[proc_address] = address;
    }
    invalidate_dispatch_caches();
    return address;
}

uint64_t guest_callable_for_proc(uint64_t proc_address) {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    const auto it = reg.proc_to_synthetic.find(proc_address);
    return it != reg.proc_to_synthetic.end() ? it->second : 0;
}

bool describe_synthetic(uint64_t va, SyntheticTarget* out) {
    const SyntheticEntry* entry = find_synthetic(va);
    if (entry == nullptr) {
        return false;
    }
    if (out != nullptr) {
        *out = SyntheticTarget{entry->thunk, entry->image_base, entry->slot_rva, entry->is_import,
                               entry->name};
    }
    return true;
}

// ---------------------------------------------------------------------------
// Faults
// ---------------------------------------------------------------------------

FaultHandler set_fault_handler(FaultHandler handler) {
    return registry().fault_handler.exchange(handler, std::memory_order_acq_rel);
}

size_t format_fault(const FaultInfo& info, char* buffer, size_t size) {
    Formatter out{buffer, size};
    const auto kind = static_cast<FaultKind>(info.kind);
    out.append("guest fault: %s at 0x%016" PRIx64, fault_kind_name(kind), info.va);
    switch (kind) {
        case FaultKind::UnknownTarget:
            if (is_synthetic_address(info.va)) {
                out.append(" (synthetic address not bound to an import or host callable)");
            } else if (info.detail != 0) {
                out.append(" (not a lifted function entry or bound import; nearest entry below: "
                           "sub_%" PRIX64 " + 0x%" PRIx64 ")",
                           info.detail, info.va - info.detail);
            } else {
                out.append(" (not a lifted function entry or bound import)");
            }
            break;
        case FaultKind::StackImbalance:
            out.append(" (rsp after return 0x%016" PRIx64 ")", info.detail);
            break;
        case FaultKind::BadStack:
            out.append(" (stack address 0x%016" PRIx64 ")", info.detail);
            break;
        case FaultKind::GeneralProtection:
            out.append(" (xgetbv with invalid XCR index 0x%" PRIx64 ")", info.detail);
            break;
        case FaultKind::UnhandledException:
            out.append(" (exception code 0x%08" PRIx64 ")", info.detail);
            break;
        case FaultKind::UnknownResumeTarget:
            out.append(" (restored rsp 0x%016" PRIx64
                       "; no active guest frame resumes there: not a resume target of the function "
                       "owning that frame, or the frame is not on this thread's guest stack)",
                       info.detail);
            break;
        case FaultKind::UnwindFailed:
            out.append(" (target frame 0x%016" PRIx64 ")", info.detail);
            break;
        case FaultKind::UnimplementedSlot: {
            SyntheticTarget target;
            const bool named = describe_synthetic(info.va, &target) && target.name != nullptr;
            out.append(" (slot %" PRIu64 "%s%s)", info.detail, named ? ": " : "", named ? target.name : "");
            break;
        }
        default:
            break;
    }
    if (info.state != nullptr) {
        static const char* const kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp",
                                               "rsi", "rdi", "r8",  "r9",  "r10", "r11",
                                               "r12", "r13", "r14", "r15"};
        const GuestState& s = *info.state;
        for (unsigned i = 0; i < 16; ++i) {
            out.append("%s%-3s=%016" PRIx64, (i % 4 == 0) ? "\n  " : " ", kNames[i], s.gpr[i]);
        }
        out.append("\n  rip=%016" PRIx64 " gs_base=%016" PRIx64 " fs_base=%016" PRIx64, s.rip,
                   s.gs_base, s.fs_base);
    }
    return out.length;
}

DispatchObserver set_dispatch_observer(DispatchObserver observer) {
    return g_dispatch_observer.exchange(observer, std::memory_order_acq_rel);
}

bool set_dispatch_cache_enabled(bool enabled) {
    const bool previous = g_dispatch_cache_enabled.exchange(enabled, std::memory_order_relaxed);
    invalidate_dispatch_caches();  // a disabled cache must not keep serving old fills
    return previous;
}

uint64_t dispatch_cache_generation() {
    return g_dispatch_generation.load(std::memory_order_acquire);
}

void raise_fault(FaultKind kind, GuestState* state, uint64_t va, uint64_t detail) {
    const FaultInfo info{static_cast<uint32_t>(kind), 0, va, detail, state};
    if (FaultHandler handler = registry().fault_handler.load(std::memory_order_acquire)) {
        handler(&info);
    }
    default_fault_handler(info);
}

// ---------------------------------------------------------------------------
// Guest threads and entry calls
// ---------------------------------------------------------------------------

bool create_guest_thread(os::VirtualMemoryArena& arena, uint64_t stack_size, GuestThread& thread) {
    thread = GuestThread{};
    uint64_t size = 0;
    if (!os::align_up_checked(stack_size == 0 ? kDefaultGuestStackSize : stack_size, os::kPageSize,
                              size) ||
        size > UINT64_MAX - os::kPageSize) {
        return false;
    }
    const uint64_t base = arena.reserve(0, size + os::kPageSize, os::PageProtect::ReadWrite);
    if (base == 0) {
        return false;
    }
    // The lowest page stays reserved but uncommitted: arena accesses below the stack fail.
    if (!arena.commit(base + os::kPageSize, size, os::PageProtect::ReadWrite)) {
        arena.release(base);
        return false;
    }
    thread.stack_reservation = base;
    thread.stack_limit = base + os::kPageSize;
    thread.stack_top = thread.stack_limit + size;
    thread.state.gpr[GuestReg::Rsp] = thread.stack_top;
    return true;
}

void destroy_guest_thread(os::VirtualMemoryArena& arena, GuestThread& thread) {
    if (thread.stack_reservation != 0) {
        arena.release(thread.stack_reservation);
    }
    thread = GuestThread{};
}

namespace {

thread_local const GuestCallBoundary* t_innermost_call = nullptr;

// Links a call_guest activation into the thread's boundary chain for its lifetime.
struct BoundaryScope {
    GuestCallBoundary boundary;
    explicit BoundaryScope(const GuestCallBoundary& b) : boundary(b) { t_innermost_call = &boundary; }
    ~BoundaryScope() { t_innermost_call = boundary.outer; }
    BoundaryScope(const BoundaryScope&) = delete;
    BoundaryScope& operator=(const BoundaryScope&) = delete;
};

}  // namespace

const GuestCallBoundary* innermost_guest_call() {
    return t_innermost_call;
}

uint64_t call_guest(os::VirtualMemoryArena& arena,
                    GuestState& state,
                    uint64_t va,
                    std::span<const uint64_t> args) {
    return call_guest_tagged(arena, state, va, args, nullptr);
}

const void* top_level_call_tag() {
    static const uint64_t kTopLevel = 0x5047'544F'504C'564CULL;  // "PGTOPLVL"
    return &kTopLevel;
}

uint64_t call_guest_tagged(os::VirtualMemoryArena& arena,
                           GuestState& state,
                           uint64_t va,
                           std::span<const uint64_t> args,
                           const void* tag) {
    static constexpr uint32_t kArgRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8,
                                             GuestReg::R9};
    const uint64_t saved_rsp = state.gpr[GuestReg::Rsp];
    const size_t stack_args = args.size() > 4 ? args.size() - 4 : 0;
    // [return address][32-byte shadow space][stack arguments]
    const uint64_t frame_bytes = kGuestStackArgOffset + 8 * stack_args;
    if (saved_rsp < frame_bytes + 16) {
        raise_fault(FaultKind::BadStack, &state, va, saved_rsp);
    }
    const uint64_t entry_rsp = os::align_down(saved_rsp - (frame_bytes - 8), 16) - 8;

    bool written = false;
    {
        // Scoped so nothing with a destructor is live while guest code (which may fault and
        // unwind through extern "C" frames) runs.
        std::vector<uint64_t> frame(static_cast<size_t>(frame_bytes / 8), 0);
        frame[0] = kGuestReturnSentinel;
        for (size_t i = 0; i < stack_args; ++i) {
            frame[kGuestStackArgOffset / 8 + i] = args[4 + i];
        }
        written = arena.write(entry_rsp, frame.data(), frame.size() * sizeof(uint64_t));
    }
    if (!written) {
        raise_fault(FaultKind::BadStack, &state, va, entry_rsp);
    }
    for (size_t i = 0; i < args.size() && i < 4; ++i) {
        state.gpr[kArgRegs[i]] = args[i];
    }
    state.gpr[GuestReg::Rsp] = entry_rsp;

    {
        const BoundaryScope scope(GuestCallBoundary{t_innermost_call, entry_rsp, saved_rsp, tag});
        __playgta6_indirect_call(&state, va);
        if (state.unwind_pending != 0) {
            // A context restore is propagating. It belongs to an enclosing guest frame only if its
            // rsp lies above this activation's frame and an enclosing activation exists.
            if (state.gpr[GuestReg::Rsp] <= entry_rsp || scope.boundary.outer == nullptr ||
                tag == top_level_call_tag()) {
                raise_fault(FaultKind::UnknownResumeTarget, &state, state.resume_rip,
                            state.gpr[GuestReg::Rsp]);
            }
            return state.gpr[GuestReg::Rax];
        }
    }

    if (state.gpr[GuestReg::Rsp] != entry_rsp + 8) {
        raise_fault(FaultKind::StackImbalance, &state, va, state.gpr[GuestReg::Rsp]);
    }
    state.gpr[GuestReg::Rsp] = saved_rsp;
    return state.gpr[GuestReg::Rax];
}

void reset_guest_runtime() {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.table_count.store(0, std::memory_order_release);
    reg.function_count.store(0, std::memory_order_relaxed);
    for (auto& t : reg.tables) {
        t = LiftedTable{};
    }
    reg.shard_registry.store(nullptr, std::memory_order_release);
    reg.owned_shard_registries.clear();
    reg.shard_function_count.store(0, std::memory_order_relaxed);
    reg.shard_count.store(0, std::memory_order_relaxed);
    reg.synthetic_count.store(1, std::memory_order_release);
    for (auto& chunk : reg.chunks) {
        chunk.store(nullptr, std::memory_order_release);
    }
    reg.owned_chunks.clear();
    reg.names.clear();
    reg.proc_to_synthetic.clear();
    reg.fault_handler.store(nullptr, std::memory_order_release);
    g_dispatch_observer.store(nullptr, std::memory_order_release);
    invalidate_dispatch_caches();
}

}  // namespace playgta6::guest

// ---------------------------------------------------------------------------
// Runtime helpers called by lifted code
// ---------------------------------------------------------------------------

using playgta6::GuestState;
namespace guest = playgta6::guest;

extern "C" {

PG6_GUEST_HELPER void __playgta6_indirect_call(GuestState* state, uint64_t target) {
    state->rip = target;
    if (guest::DispatchObserver observer = guest::g_dispatch_observer.load(std::memory_order_acquire)) {
        observer(state, target);
    }
    // Fast path: this thread resolved `target` before and the registry has not changed since.
    const uint64_t generation = guest::g_dispatch_generation.load(std::memory_order_acquire);
    if (const guest::DispatchCache* cache = guest::t_dispatch_cache) {
        const guest::DispatchCacheEntry& e = cache->entries[guest::dispatch_cache_index(target)];
        if (e.va == target && e.generation == generation) {
            if (e.kind == guest::kCachedLifted) {
                e.fn.lifted(state);
            } else {
                e.fn.shard(e.local_idx, state);
            }
            return;
        }
    }
    if (guest::is_synthetic_address(target)) {
        if (target == guest::kGuestReturnSentinel) {
            return;  // a jump to call_guest's return address: back to the host caller
        }
        if (const guest::SyntheticEntry* entry = guest::find_synthetic(target)) {
            entry->thunk(state);
            return;
        }
    } else {
        if (playgta6::LiftedFn fn = guest::find_lifted_function(target)) {
            guest::fill_dispatch_cache(target, generation, fn, nullptr, 0);
            fn(state);
            return;
        }
        guest::ResolvedShardTarget shard_target{};
        if (guest::find_shard_dispatch_target(target, shard_target) &&
            shard_target.dispatcher != nullptr) {
            guest::fill_dispatch_cache(target, generation, nullptr, shard_target.dispatcher,
                                       shard_target.local_idx);
            shard_target.dispatcher(shard_target.local_idx, state);
            return;
        }
    }
    guest::raise_fault(guest::FaultKind::UnknownTarget, state, target,
                       guest::nearest_entry_at_or_below(target));
}

PG6_GUEST_HELPER void __playgta6_trap(GuestState* state, uint64_t va) {
    state->rip = va;
    if (state->unwind_pending != 0) {
        // The activation owning the restored rsp has no block at the resume rip.
        guest::raise_fault(guest::FaultKind::UnknownResumeTarget, state, state->resume_rip,
                           state->gpr[playgta6::GuestReg::Rsp]);
    }
    guest::raise_fault(guest::FaultKind::Trap, state, va);
}

PG6_GUEST_HELPER void __playgta6_unimplemented(GuestState* state, uint64_t va) {
    state->rip = va;
    guest::raise_fault(guest::FaultKind::Unimplemented, state, va);
}

PG6_GUEST_HELPER void __playgta6_divide_error(GuestState* state, uint64_t va) {
    state->rip = va;
    guest::raise_fault(guest::FaultKind::DivideError, state, va);
}

// ---------------------------------------------------------------------------
// C-ABI entry points
// ---------------------------------------------------------------------------

int32_t pg6_guest_register_lifted_functions(const Pg6LiftedFunctionEntry* entries, size_t count) {
    if (entries == nullptr && count != 0) {
        return 0;
    }
    return guest::register_lifted_functions(std::span<const Pg6LiftedFunctionEntry>(entries, count))
               ? 1
               : 0;
}

int32_t pg6_guest_register_shard_dispatch_table(const Pg6ShardDispatchTable* table) {
    if (table == nullptr) {
        return 0;
    }
    return guest::register_shard_dispatch_table(*table) ? 1 : 0;
}

playgta6::LiftedFn pg6_guest_find_lifted_function(uint64_t va) {
    return guest::find_lifted_function(va);
}

size_t pg6_guest_bind_import_slots(uint64_t image_base, const Pg6ImportSlotBinding* slots, size_t count) {
    if (slots == nullptr) {
        return 0;
    }
    return guest::bind_import_slots(*playgta6::os::win32::get_vmem_arena(), image_base,
                                    std::span<const Pg6ImportSlotBinding>(slots, count));
}

uint64_t pg6_guest_register_host_callable(Pg6HostThunkFn thunk, const char* name, uint64_t proc_address) {
    return guest::register_host_callable(thunk, name, proc_address);
}

uint64_t pg6_guest_callable_for_proc(uint64_t proc_address) {
    return guest::guest_callable_for_proc(proc_address);
}

Pg6GuestFaultHandler pg6_guest_set_fault_handler(Pg6GuestFaultHandler handler) {
    return guest::set_fault_handler(handler);
}

int32_t pg6_guest_thread_create(Pg6GuestThread* thread, uint64_t stack_size) {
    if (thread == nullptr) {
        return 0;
    }
    return guest::create_guest_thread(*playgta6::os::win32::get_vmem_arena(), stack_size, *thread) ? 1
                                                                                                  : 0;
}

void pg6_guest_thread_destroy(Pg6GuestThread* thread) {
    if (thread != nullptr) {
        guest::destroy_guest_thread(*playgta6::os::win32::get_vmem_arena(), *thread);
    }
}

uint64_t pg6_guest_call(GuestState* state, uint64_t va, const uint64_t* args, size_t count) {
    if (args == nullptr) {
        count = 0;
    }
    return guest::call_guest(*playgta6::os::win32::get_vmem_arena(), *state, va,
                             std::span<const uint64_t>(args, count));
}

}  // extern "C"
