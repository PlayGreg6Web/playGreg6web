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
    return registry().function_count.load(std::memory_order_relaxed);
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

uint64_t call_guest(os::VirtualMemoryArena& arena,
                    GuestState& state,
                    uint64_t va,
                    std::span<const uint64_t> args) {
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

    __playgta6_indirect_call(&state, va);

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
    reg.synthetic_count.store(1, std::memory_order_release);
    for (auto& chunk : reg.chunks) {
        chunk.store(nullptr, std::memory_order_release);
    }
    reg.owned_chunks.clear();
    reg.names.clear();
    reg.proc_to_synthetic.clear();
    reg.fault_handler.store(nullptr, std::memory_order_release);
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
    if (guest::is_synthetic_address(target)) {
        if (target == guest::kGuestReturnSentinel) {
            return;  // a jump to call_guest's return address: back to the host caller
        }
        if (const guest::SyntheticEntry* entry = guest::find_synthetic(target)) {
            entry->thunk(state);
            return;
        }
    } else if (playgta6::LiftedFn fn = guest::find_lifted_function(target)) {
        fn(state);
        return;
    }
    guest::raise_fault(guest::FaultKind::UnknownTarget, state, target,
                       guest::nearest_entry_at_or_below(target));
}

PG6_GUEST_HELPER void __playgta6_trap(GuestState* state, uint64_t va) {
    state->rip = va;
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
