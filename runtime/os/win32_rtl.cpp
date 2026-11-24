// Win32 shims required by the recompiled runtime DLLs, core part (task O26): critical section
// init with flags, fiber-local storage, one-time initialization, pointer encoding, interlocked
// SLIST, SwitchToThread, GetModuleHandleExW, RtlCaptureStackBackTrace (host form) and
// FreeLibraryWhenCallbackReturns. See os/win32_rtl.h for the design notes.

#include "os/win32_rtl.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "os/guest_window.h"
#include "os/vmem.h"
#include "os/win32_loader.h"

namespace playgta6::os::win32 {

namespace {

// ---------------------------------------------------------------------------
// FLS
// ---------------------------------------------------------------------------

constexpr uint32_t kFlsChunkSlots = 64;
constexpr uint32_t kFlsChunks = kFlsMaxSlots / kFlsChunkSlots;
constexpr int kFlsExitPasses = 4;

struct FlsChunk {
    std::atomic<uint64_t> value[kFlsChunkSlots];
    FlsChunk() {
        for (auto& v : value) {
            v.store(0, std::memory_order_relaxed);
        }
    }
};

// One per host thread that ever stored an FLS value. Chunks are created only by the owning
// thread (publish with release); FlsFree reads other threads' chunks under the registry lock.
struct FlsThread {
    std::atomic<FlsChunk*> chunk[kFlsChunks];
    FlsThread() {
        for (auto& c : chunk) {
            c.store(nullptr, std::memory_order_relaxed);
        }
    }
    ~FlsThread() {
        for (auto& c : chunk) {
            delete c.load(std::memory_order_relaxed);
        }
    }
};

struct FlsRegistry {
    std::mutex mutex;  // guards slot allocation, `threads` and thread teardown
    std::vector<FlsThread*> threads;
    std::atomic<uint8_t> allocated[kFlsMaxSlots];
    std::atomic<uint64_t> callback[kFlsMaxSlots];
    uint32_t allocated_count = 0;
    uint32_t search_from = 1;

    FlsRegistry() {
        for (uint32_t i = 0; i < kFlsMaxSlots; ++i) {
            allocated[i].store(0, std::memory_order_relaxed);
            callback[i].store(0, std::memory_order_relaxed);
        }
    }
};

// Leaked on purpose: thread-exit paths may run during static destruction.
FlsRegistry& fls_registry() {
    static auto* registry = new FlsRegistry();
    return *registry;
}

thread_local FlsThread* t_fls = nullptr;     // trivially destructible: safe in any exit order
thread_local bool t_fls_hooked = false;

void release_fls_thread(FlsThread* thread) {
    if (thread == nullptr) {
        return;
    }
    FlsRegistry& reg = fls_registry();
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = std::find(reg.threads.begin(), reg.threads.end(), thread);
        if (it != reg.threads.end()) {
            reg.threads.erase(it);
        }
    }
    delete thread;
}

// Reclaims the storage of threads that end without the guest exit hook (host test threads, ...).
struct FlsThreadSentinel {
    void touch() {}
    ~FlsThreadSentinel() {
        if (!t_fls_hooked && t_fls != nullptr) {
            FlsThread* thread = t_fls;
            t_fls = nullptr;
            release_fls_thread(thread);
        }
    }
};
thread_local FlsThreadSentinel t_fls_sentinel;

FlsThread* fls_thread_for_write() {
    if (t_fls != nullptr) {
        return t_fls;
    }
    t_fls_sentinel.touch();  // constructs the sentinel so its destructor runs at thread exit
    auto* thread = new (std::nothrow) FlsThread();
    if (thread == nullptr) {
        return nullptr;
    }
    FlsRegistry& reg = fls_registry();
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.threads.push_back(thread);
    }
    t_fls = thread;
    return thread;
}

// ---------------------------------------------------------------------------
// InitOnce
// ---------------------------------------------------------------------------

std::mutex& init_once_mutex() {
    static auto* m = new std::mutex();
    return *m;
}

std::condition_variable& init_once_cv() {
    static auto* cv = new std::condition_variable();
    return *cv;
}

void init_once_notify() {
    {
        std::lock_guard<std::mutex> lock(init_once_mutex());
    }
    init_once_cv().notify_all();
}

// ---------------------------------------------------------------------------
// Pointer encoding
// ---------------------------------------------------------------------------

constexpr uint64_t kPointerCookie = 0xA5C3'1E7B'92D4'6F08ull;

constexpr uint64_t rotr64(uint64_t v, unsigned n) {
    n &= 63;
    return n == 0 ? v : (v >> n) | (v << (64 - n));
}
constexpr uint64_t rotl64(uint64_t v, unsigned n) {
    n &= 63;
    return n == 0 ? v : (v << n) | (v >> (64 - n));
}

// ---------------------------------------------------------------------------
// SLIST
// ---------------------------------------------------------------------------

constexpr uint64_t kSListLockBit = uint64_t{1} << 63;
constexpr uint64_t kSListDepthMask = 0xFFFFu;
constexpr unsigned kSListSeqShift = 16;
constexpr uint64_t kSListSeqMask = 0xFFFFFFFFull << kSListSeqShift;

bool slist_header_ok(const void* head) {
    if (head == nullptr || (reinterpret_cast<uintptr_t>(head) & 7u) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    return true;
}

// Spin-locks the header; returns word 1 as it was (lock bit clear).
uint64_t slist_lock(Pg6SListHeader* head) {
    std::atomic_ref<uint64_t> word(head->depth_sequence);
    unsigned spins = 0;
    for (;;) {
        uint64_t v = word.load(std::memory_order_relaxed);
        if ((v & kSListLockBit) == 0 &&
            word.compare_exchange_weak(v, v | kSListLockBit, std::memory_order_acquire,
                                       std::memory_order_relaxed)) {
            return v;
        }
        if (++spins > 64) {
            std::this_thread::yield();
        }
    }
}

void slist_unlock(Pg6SListHeader* head, uint64_t new_word1) {
    std::atomic_ref<uint64_t> word(head->depth_sequence);
    word.store(new_word1 & ~kSListLockBit, std::memory_order_release);
}

uint64_t slist_bump(uint64_t word1, int depth_delta) {
    const uint64_t seq = ((word1 & kSListSeqMask) + (uint64_t{1} << kSListSeqShift)) & kSListSeqMask;
    const uint64_t depth = (word1 + static_cast<uint64_t>(static_cast<int64_t>(depth_delta))) & kSListDepthMask;
    return (word1 & ~(kSListSeqMask | kSListDepthMask)) | seq | depth;
}

// Host pointer to the SLIST_ENTRY.Next slot of a guest entry, nullptr if it is not 16-aligned or
// not mapped.
uint64_t* slist_next_slot(uint64_t entry) {
    VirtualMemoryArena* arena = get_vmem_arena();
    if (arena == nullptr || entry == 0 || (entry & 15u) != 0) {
        return nullptr;
    }
    if (guest_window::translated(entry)) [[unlikely]] {
        return static_cast<uint64_t*>(guest_window::translate(entry));  // aligned: within one slot
    }
    return reinterpret_cast<uint64_t*>(arena->host_ptr(entry, sizeof(uint64_t)));
}

uint64_t slist_first(Pg6SListHeader* head) {
    return std::atomic_ref<uint64_t>(head->first_entry).load(std::memory_order_relaxed);
}

void slist_set_first(Pg6SListHeader* head, uint64_t value) {
    std::atomic_ref<uint64_t>(head->first_entry).store(value, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// GetModuleHandleExW
// ---------------------------------------------------------------------------

bool read_guest_u16_string(uint64_t address, std::u16string* out) {
    VirtualMemoryArena* arena = get_vmem_arena();
    if (arena == nullptr) {
        return false;
    }
    constexpr size_t kMaxChars = 4096;
    for (size_t i = 0; i < kMaxChars; ++i) {
        char16_t c = 0;
        const uint64_t va = address + 2 * i;
        if (guest_window::translated(va) ? !guest_window::copy_from_guest(&c, va, sizeof(c))
                                         : !arena->read(va, &c, sizeof(c))) {
            return false;
        }
        if (c == 0) {
            return true;
        }
        out->push_back(c);
    }
    return false;
}

std::atomic<uint64_t> g_free_library_requests{0};

}  // namespace

// ---------------------------------------------------------------------------
// FLS
// ---------------------------------------------------------------------------

uint32_t FlsAlloc(uint64_t callback) {
    FlsRegistry& reg = fls_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (uint32_t n = 0; n < kFlsMaxSlots - 1; ++n) {
        // Lowest free slot first; search_from is only a hint that is reset by FlsFree.
        const uint32_t index = 1 + ((reg.search_from - 1 + n) % (kFlsMaxSlots - 1));
        if (reg.allocated[index].load(std::memory_order_relaxed) == 0) {
            reg.callback[index].store(callback, std::memory_order_relaxed);
            reg.allocated[index].store(1, std::memory_order_release);
            ++reg.allocated_count;
            reg.search_from = index + 1 < kFlsMaxSlots ? index + 1 : 1;
            return index;
        }
    }
    SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    return FLS_OUT_OF_INDEXES;
}

bool FlsFree(uint32_t index, FlsCallbackInvoker invoker, void* ctx) {
    FlsRegistry& reg = fls_registry();
    std::vector<uint64_t> values;
    uint64_t callback = 0;
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        if (index == 0 || index >= kFlsMaxSlots ||
            reg.allocated[index].load(std::memory_order_relaxed) == 0) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return false;
        }
        reg.allocated[index].store(0, std::memory_order_release);
        callback = reg.callback[index].exchange(0, std::memory_order_relaxed);
        --reg.allocated_count;
        reg.search_from = 1;  // lowest free slot first again
        const uint32_t chunk_index = index / kFlsChunkSlots;
        const uint32_t slot = index % kFlsChunkSlots;
        for (FlsThread* thread : reg.threads) {
            FlsChunk* chunk = thread->chunk[chunk_index].load(std::memory_order_acquire);
            if (chunk != nullptr) {
                const uint64_t v = chunk->value[slot].exchange(0, std::memory_order_acq_rel);
                if (v != 0) {
                    values.push_back(v);
                }
            }
        }
    }
    if (callback != 0 && invoker != nullptr) {
        for (const uint64_t v : values) {
            if (!invoker(ctx, callback, v)) {
                break;
            }
        }
    }
    SetLastError(ERROR_SUCCESS);
    return true;
}

uint64_t FlsGetValue(uint32_t index) {
    if (index == 0 || index >= kFlsMaxSlots ||
        fls_registry().allocated[index].load(std::memory_order_acquire) == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    SetLastError(ERROR_SUCCESS);
    const FlsThread* thread = t_fls;
    if (thread == nullptr) {
        return 0;
    }
    const FlsChunk* chunk = thread->chunk[index / kFlsChunkSlots].load(std::memory_order_acquire);
    return chunk != nullptr ? chunk->value[index % kFlsChunkSlots].load(std::memory_order_relaxed) : 0;
}

bool FlsSetValue(uint32_t index, uint64_t value) {
    FlsRegistry& reg = fls_registry();
    if (index == 0 || index >= kFlsMaxSlots ||
        reg.allocated[index].load(std::memory_order_acquire) == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    FlsThread* thread = t_fls;
    if (thread == nullptr) {
        if (value == 0) {
            SetLastError(ERROR_SUCCESS);
            return true;  // nothing stored yet: already "0" on this thread
        }
        thread = fls_thread_for_write();
        if (thread == nullptr) {
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return false;
        }
    }
    std::atomic<FlsChunk*>& slot_chunk = thread->chunk[index / kFlsChunkSlots];
    FlsChunk* chunk = slot_chunk.load(std::memory_order_acquire);
    if (chunk == nullptr) {
        chunk = new (std::nothrow) FlsChunk();
        if (chunk == nullptr) {
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return false;
        }
        slot_chunk.store(chunk, std::memory_order_release);
    }
    std::atomic<uint64_t>& cell = chunk->value[index % kFlsChunkSlots];
    cell.store(value, std::memory_order_release);
    if (reg.allocated[index].load(std::memory_order_acquire) == 0) {
        cell.store(0, std::memory_order_release);  // freed concurrently: do not leak into a reuse
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    SetLastError(ERROR_SUCCESS);
    return true;
}

void fls_mark_thread_hooked() {
    t_fls_hooked = true;
}

void fls_run_thread_exit(FlsCallbackInvoker invoker, void* ctx) {
    FlsThread* thread = t_fls;
    if (thread == nullptr) {
        return;
    }
    FlsRegistry& reg = fls_registry();
    bool stop = false;
    for (int pass = 0; pass < kFlsExitPasses && !stop; ++pass) {
        bool ran = false;
        for (uint32_t c = 0; c < kFlsChunks && !stop; ++c) {
            FlsChunk* chunk = thread->chunk[c].load(std::memory_order_acquire);
            if (chunk == nullptr) {
                continue;
            }
            for (uint32_t s = 0; s < kFlsChunkSlots && !stop; ++s) {
                const uint64_t v = chunk->value[s].exchange(0, std::memory_order_acq_rel);
                if (v == 0) {
                    continue;
                }
                const uint32_t index = c * kFlsChunkSlots + s;
                const uint64_t callback = reg.callback[index].load(std::memory_order_relaxed);
                if (callback != 0 && invoker != nullptr &&
                    reg.allocated[index].load(std::memory_order_acquire) != 0) {
                    ran = true;
                    if (!invoker(ctx, callback, v)) {
                        stop = true;
                    }
                }
            }
        }
        if (!ran) {
            break;
        }
    }
    t_fls = nullptr;
    release_fls_thread(thread);
}

uint32_t fls_allocated_count() {
    FlsRegistry& reg = fls_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return reg.allocated_count;
}

void fls_reset_for_tests() {
    FlsRegistry& reg = fls_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (uint32_t i = 0; i < kFlsMaxSlots; ++i) {
        reg.allocated[i].store(0, std::memory_order_relaxed);
        reg.callback[i].store(0, std::memory_order_relaxed);
    }
    reg.allocated_count = 0;
    reg.search_from = 1;
    for (FlsThread* thread : reg.threads) {
        for (auto& chunk : thread->chunk) {
            FlsChunk* c = chunk.load(std::memory_order_relaxed);
            if (c != nullptr) {
                for (auto& v : c->value) {
                    v.store(0, std::memory_order_relaxed);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// InitOnce
// ---------------------------------------------------------------------------

bool InitOnceRun(InitOnce* once, const InitOnceBody& body, uint64_t* context_out) {
    if (once == nullptr || (reinterpret_cast<uintptr_t>(once) & 7u) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    std::atomic_ref<uint64_t> state(once->ptr);
    for (;;) {
        const uint64_t v = state.load(std::memory_order_acquire);
        if ((v & 3u) == 2u) {
            if (context_out != nullptr) {
                *context_out = v & ~uint64_t{3};
            }
            return true;
        }
        if (v == 0) {
            uint64_t expected = 0;
            if (!state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                continue;
            }
            uint64_t context = 0;
            const InitOnceOutcome outcome = body(&context);
            if (outcome == InitOnceOutcome::Completed) {
                if ((context & 3u) != 0) {
                    state.store(0, std::memory_order_release);
                    init_once_notify();
                    SetLastError(ERROR_INVALID_PARAMETER);
                    return false;
                }
                state.store(context | 2u, std::memory_order_release);
                init_once_notify();
                if (context_out != nullptr) {
                    *context_out = context;
                }
                return true;
            }
            state.store(0, std::memory_order_release);  // failed or abandoned: retry next time
            init_once_notify();
            return false;
        }
        if ((v & 3u) == 1u) {
            std::unique_lock<std::mutex> lock(init_once_mutex());
            init_once_cv().wait(lock, [&] { return (state.load(std::memory_order_acquire) & 3u) != 1u; });
            continue;
        }
        SetLastError(ERROR_INVALID_PARAMETER);  // not an INIT_ONCE this module wrote
        return false;
    }
}

bool InitOnceExecuteOnce(InitOnce* once, Pg6InitOnceFn fn, void* parameter, void** context) {
    if (fn == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    uint64_t result = 0;
    const bool ok = InitOnceRun(
        once,
        [&](uint64_t* ctx) {
            void* out = nullptr;
            const int32_t r = fn(once, parameter, &out);
            *ctx = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(out));
            return r != 0 ? InitOnceOutcome::Completed : InitOnceOutcome::Failed;
        },
        &result);
    if (context != nullptr) {
        *context = ok ? reinterpret_cast<void*>(static_cast<uintptr_t>(result)) : nullptr;
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Pointer encoding, SLIST, threads, modules
// ---------------------------------------------------------------------------

uint64_t EncodePointer(uint64_t pointer) {
    return rotr64(pointer ^ kPointerCookie, static_cast<unsigned>(kPointerCookie & 63));
}

uint64_t DecodePointer(uint64_t pointer) {
    return rotl64(pointer, static_cast<unsigned>(kPointerCookie & 63)) ^ kPointerCookie;
}

uint64_t InterlockedPushEntrySList(SListHeader* head, uint64_t entry) {
    if (!slist_header_ok(head)) {
        return 0;
    }
    uint64_t* next = slist_next_slot(entry);
    if (next == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    const uint64_t word1 = slist_lock(head);
    const uint64_t first = slist_first(head);
    std::memcpy(next, &first, sizeof(first));
    slist_set_first(head, entry);
    slist_unlock(head, slist_bump(word1, +1));
    return first;
}

uint64_t InterlockedPopEntrySList(SListHeader* head) {
    if (!slist_header_ok(head)) {
        return 0;
    }
    const uint64_t word1 = slist_lock(head);
    const uint64_t first = slist_first(head);
    if (first == 0) {
        slist_unlock(head, word1);
        return 0;
    }
    uint64_t next_value = 0;
    const uint64_t* next = slist_next_slot(first);
    if (next == nullptr) {
        slist_unlock(head, word1);  // corrupted list: leave it as it was
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    std::memcpy(&next_value, next, sizeof(next_value));
    slist_set_first(head, next_value);
    slist_unlock(head, slist_bump(word1, -1));
    return first;
}

uint64_t InterlockedFlushSList(SListHeader* head) {
    if (!slist_header_ok(head)) {
        return 0;
    }
    const uint64_t word1 = slist_lock(head);
    const uint64_t first = slist_first(head);
    slist_set_first(head, 0);
    slist_unlock(head, slist_bump(word1 & ~kSListDepthMask, 0));
    return first;
}

uint16_t QueryDepthSList(const SListHeader* head) {
    if (!slist_header_ok(head)) {
        return 0;
    }
    auto* mutable_head = const_cast<Pg6SListHeader*>(head);
    return static_cast<uint16_t>(std::atomic_ref<uint64_t>(mutable_head->depth_sequence)
                                     .load(std::memory_order_acquire) &
                                 kSListDepthMask);
}

bool SwitchToThread() {
    std::this_thread::yield();
    return true;
}

bool GetModuleHandleExW(uint32_t flags, uint64_t name_or_address, Handle* module) {
    if (module == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    *module = kNullHandle;
    constexpr uint32_t kKnown = GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT |
                                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS;
    const bool pin = (flags & GET_MODULE_HANDLE_EX_FLAG_PIN) != 0;
    const bool unchanged = (flags & GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT) != 0;
    if ((flags & ~kKnown) != 0 || (pin && unchanged)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    Handle found = kNullHandle;
    if ((flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS) != 0) {
        found = find_module_by_address(name_or_address);
    } else if (name_or_address == 0) {
        found = GetModuleHandleW(nullptr);
    } else {
        std::u16string name;
        if (!read_guest_u16_string(name_or_address, &name)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return false;
        }
        found = GetModuleHandleW(name.c_str());
    }
    if (found == kNullHandle) {
        return false;  // last error set by the loader
    }
    if (!module_add_reference(found, pin, !unchanged)) {
        return false;
    }
    *module = found;
    SetLastError(ERROR_SUCCESS);
    return true;
}

uint64_t free_library_when_callback_returns_count() {
    return g_free_library_requests.load(std::memory_order_relaxed);
}

}  // namespace playgta6::os::win32

namespace win32 = playgta6::os::win32;

extern "C" {

int32_t pg6_win32_initialize_critical_section_ex(Pg6CriticalSection* critical_section,
                                                 uint32_t spin_count,
                                                 uint32_t flags) {
    if (critical_section == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // Windows accepts the CRITICAL_SECTION_* / RTL_CRITICAL_SECTION_FLAG_* bits (top byte).
    if ((flags & 0x00FFFFFFu) != 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // Constructs in place; whatever bytes the guest storage held are never read.
    win32::InitializeCriticalSectionWithSpinCount(critical_section, spin_count);
    return 1;
}

uint32_t pg6_win32_fls_alloc(uint64_t callback) {
    return win32::FlsAlloc(callback);
}

int32_t pg6_win32_fls_free(uint32_t fls_index) {
    return win32::FlsFree(fls_index) ? 1 : 0;
}

uint64_t pg6_win32_fls_get_value(uint32_t fls_index) {
    return win32::FlsGetValue(fls_index);
}

int32_t pg6_win32_fls_set_value(uint32_t fls_index, uint64_t fls_data) {
    return win32::FlsSetValue(fls_index, fls_data) ? 1 : 0;
}

uint64_t pg6_win32_encode_pointer(uint64_t pointer) {
    return win32::EncodePointer(pointer);
}

uint64_t pg6_win32_decode_pointer(uint64_t pointer) {
    return win32::DecodePointer(pointer);
}

int32_t pg6_win32_init_once_execute_once(Pg6InitOnce* init_once,
                                         Pg6InitOnceFn init_fn,
                                         void* parameter,
                                         void** context) {
    return win32::InitOnceExecuteOnce(init_once, init_fn, parameter, context) ? 1 : 0;
}

uint64_t pg6_win32_interlocked_push_entry_slist(Pg6SListHeader* list_head, uint64_t list_entry) {
    return win32::InterlockedPushEntrySList(list_head, list_entry);
}

uint64_t pg6_win32_interlocked_pop_entry_slist(Pg6SListHeader* list_head) {
    return win32::InterlockedPopEntrySList(list_head);
}

uint64_t pg6_win32_interlocked_flush_slist(Pg6SListHeader* list_head) {
    return win32::InterlockedFlushSList(list_head);
}

uint16_t pg6_win32_query_depth_slist(const Pg6SListHeader* list_head) {
    return win32::QueryDepthSList(list_head);
}

int32_t pg6_win32_switch_to_thread(void) {
    return win32::SwitchToThread() ? 1 : 0;
}

int32_t pg6_win32_get_module_handle_ex_w(uint32_t flags,
                                         uint64_t module_name_or_address,
                                         Pg6Handle* module) {
    return win32::GetModuleHandleExW(flags, module_name_or_address, module) ? 1 : 0;
}

uint16_t pg6_win32_rtl_capture_stack_back_trace(uint32_t,
                                                uint32_t,
                                                void** back_trace,
                                                uint32_t* back_trace_hash) {
    (void)back_trace;
    if (back_trace_hash != nullptr) {
        *back_trace_hash = 0;
    }
    return 0;  // the host form has no guest stack to walk
}

void pg6_win32_free_library_when_callback_returns(uint64_t instance, Pg6Handle module) {
    (void)instance;
    (void)module;  // guest modules are never unloaded: record the request only
    win32::g_free_library_requests.fetch_add(1, std::memory_order_relaxed);
}

}  // extern "C"
