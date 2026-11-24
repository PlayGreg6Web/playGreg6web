#include "guest/guest_callbacks.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "guest/export_thunks.h"
#include "os/os_shim.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/xasync_provider.h"
#include "os/xbox_middleware.h"

namespace playgta6::guest::callbacks {

namespace com = playgta6::guest::com;

namespace {

struct TrampolineRegistry {
    std::mutex mutex;
    std::unordered_set<Trampoline*> live;
};

TrampolineRegistry& trampolines() {
    static TrampolineRegistry instance;
    return instance;
}

// Takes a one-shot trampoline out of the registry and returns its fields (nullopt-like: function
// 0 if the pointer is not a live trampoline, which is reported).
Trampoline consume(void* context) {
    auto* t = static_cast<Trampoline*>(context);
    TrampolineRegistry& reg = trampolines();
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        if (reg.live.erase(t) == 0) {
            com::detail::diagnostic("guest callbacks: host callback with an unknown trampoline");
            return Trampoline{};
        }
    }
    const Trampoline copy = *t;
    delete t;
    return copy;
}

void task_queue_entry(void* context, bool canceled) {
    const Trampoline t = consume(context);
    if (t.function != 0) {
        com::GuestFunction<void(uint64_t, bool)>(t.function).call_top_level(t.context, canceled);
    }
}

void task_queue_terminated_entry(void* context) {
    const Trampoline t = consume(context);
    if (t.function != 0) {
        com::GuestFunction<void(uint64_t)>(t.function).call_top_level(t.context);
    }
}

uint32_t thread_start_entry(void* parameter) {
    const Trampoline t = consume(parameter);
    if (t.function == 0) {
        return 0;
    }
    playgta6::GuestState& state = com::current_guest_state(t.stack_size);
    return com::GuestFunction<uint32_t(uint64_t)>(t.function).call_top_level(state, t.context);
}

// ---------------------------------------------------------------------------
// XAsyncBlock shadows
// ---------------------------------------------------------------------------

struct AsyncShadow {
    Pg6XAsyncBlock block;  // first member: the shim's Pg6XAsyncBlock* converts back
    uint64_t guest_block = 0;
    uint64_t guest_callback = 0;
};
static_assert(std::is_standard_layout_v<AsyncShadow>);

struct ShadowRegistry {
    std::mutex mutex;
    std::unordered_map<uint64_t, std::unique_ptr<AsyncShadow>> active;
    std::vector<std::unique_ptr<AsyncShadow>> retired;  // replaced while still pending
};

ShadowRegistry& shadows() {
    static ShadowRegistry instance;
    return instance;
}

void async_completion_entry(Pg6XAsyncBlock* block) {
    auto* shadow = reinterpret_cast<AsyncShadow*>(block);
    // Mirror the shim's private state into the guest block (opaque to the guest).
    com::write_guest(shadow->guest_block + offsetof(GuestAsyncBlock, internal), shadow->block.internal,
                     sizeof(shadow->block.internal));
    if (shadow->guest_callback != 0) {
        com::GuestFunction<void(uint64_t)>(shadow->guest_callback).call_top_level(shadow->guest_block);
    }
}

// The shim still references the shadow: its operation runs or its completion callback has not
// returned yet.
bool pending(AsyncShadow& shadow) {
    return playgta6::os::gdk::xasync_in_flight(&shadow.block);
}

// Frees a shadow nobody references any more (its finished operation record goes too).
void drop_shadow(std::unique_ptr<AsyncShadow> shadow) {
    if (shadow) {
        playgta6::os::gdk::xasync_forget(&shadow->block);
    }
}

std::string hex(uint64_t value) {
    char buffer[24];
    std::snprintf(buffer, sizeof(buffer), "0x%016" PRIx64, value);
    return buffer;
}

}  // namespace

Trampoline* create_trampoline(uint64_t function, uint64_t context, uint64_t stack_size) {
    auto* t = new Trampoline{function, context, stack_size};
    TrampolineRegistry& reg = trampolines();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.live.insert(t);
    return t;
}

void destroy_trampoline(Trampoline* trampoline) {
    if (trampoline == nullptr) {
        return;
    }
    TrampolineRegistry& reg = trampolines();
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        if (reg.live.erase(trampoline) == 0) {
            return;
        }
    }
    delete trampoline;
}

uint64_t live_trampolines() {
    TrampolineRegistry& reg = trampolines();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return reg.live.size();
}

HostCallback<Pg6XTaskQueueCallback> adapt_task_queue_callback(uint64_t guest_function, uint64_t guest_context) {
    Trampoline* t = create_trampoline(guest_function, guest_context);
    return {&task_queue_entry, t, t};
}

HostCallback<Pg6XTaskQueueTerminatedCallback> adapt_task_queue_terminated_callback(uint64_t guest_function,
                                                                                   uint64_t guest_context) {
    Trampoline* t = create_trampoline(guest_function, guest_context);
    return {&task_queue_terminated_entry, t, t};
}

HostCallback<Pg6ThreadStartRoutine> adapt_thread_start(uint64_t guest_function,
                                                       uint64_t guest_parameter,
                                                       uint64_t stack_size) {
    Trampoline* t = create_trampoline(guest_function, guest_parameter, stack_size);
    return {&thread_start_entry, t, t};
}

Pg6XAsyncBlock* async_shadow_begin(uint64_t guest_block) {
    GuestAsyncBlock guest{};
    if (guest_block == 0 || !com::read_guest(guest_block, guest)) {
        com::detail::log_bad_pointer("XAsyncBlock", guest_block);
        return nullptr;
    }
    auto shadow = std::make_unique<AsyncShadow>();
    shadow->block.queue = reinterpret_cast<Pg6XTaskQueueHandle>(static_cast<uintptr_t>(guest.queue));
    shadow->block.context = nullptr;  // the guest context stays in the guest block
    shadow->block.callback = &async_completion_entry;
    shadow->guest_block = guest_block;
    shadow->guest_callback = guest.callback;
    Pg6XAsyncBlock* result = &shadow->block;

    std::unique_ptr<AsyncShadow> replaced;
    {
        ShadowRegistry& reg = shadows();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = reg.active.find(guest_block);
        if (it != reg.active.end()) {
            if (pending(*it->second)) {
                com::detail::diagnostic(
                    ("guest callbacks: XAsyncBlock " + hex(guest_block) + " reused while its operation is pending")
                        .c_str());
                reg.retired.push_back(std::move(it->second));
            } else {
                replaced = std::move(it->second);
            }
            it->second = std::move(shadow);
        } else {
            reg.active.emplace(guest_block, std::move(shadow));
        }
    }
    drop_shadow(std::move(replaced));  // outside the lock: may run a provider's Cleanup
    return result;
}

Pg6XAsyncBlock* async_shadow_find(uint64_t guest_block) {
    ShadowRegistry& reg = shadows();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.active.find(guest_block);
    return it != reg.active.end() ? &it->second->block : nullptr;
}

void async_shadow_end(uint64_t guest_block) {
    std::unique_ptr<AsyncShadow> ended;
    {
        ShadowRegistry& reg = shadows();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = reg.active.find(guest_block);
        if (it != reg.active.end() && !pending(*it->second)) {
            ended = std::move(it->second);
            reg.active.erase(it);
        }
    }
    drop_shadow(std::move(ended));
}

size_t live_async_shadows() {
    ShadowRegistry& reg = shadows();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return reg.active.size() + reg.retired.size();
}

void reset_callbacks() {
    {
        TrampolineRegistry& reg = trampolines();
        std::lock_guard<std::mutex> lock(reg.mutex);
        for (Trampoline* t : reg.live) {
            delete t;
        }
        reg.live.clear();
    }
    ShadowRegistry& reg = shadows();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.active.clear();
    reg.retired.clear();
}

// ---------------------------------------------------------------------------
// Import thunk implementations (marshaled by com::function_thunk)
// ---------------------------------------------------------------------------

namespace {

using com::CallContext;
using com::GuestPtr;

uint64_t create_thread_impl(CallContext& ctx,
                            uint64_t /*security_attributes*/,
                            uint64_t stack_size,
                            uint64_t start_address,
                            uint64_t parameter,
                            uint32_t creation_flags,
                            GuestPtr<uint32_t> thread_id) {
    (void)ctx;
    if (start_address == 0) {
        return pg6_win32_create_thread(nullptr, stack_size, nullptr, nullptr, creation_flags, nullptr);
    }
    HostCallback<Pg6ThreadStartRoutine> cb = adapt_thread_start(start_address, parameter, stack_size);
    uint32_t id = 0;
    const Pg6Handle handle =
        pg6_win32_create_thread(nullptr, stack_size, cb.function, cb.context, creation_flags, &id);
    if (handle == 0) {
        destroy_trampoline(cb.trampoline);
        return 0;
    }
    if (!thread_id.is_null() && !thread_id.write(id)) {
        com::detail::log_bad_pointer("CreateThread thread id", thread_id.va());
    }
    return handle;
}

int32_t submit_callback_impl(uint64_t queue, uint32_t port, uint64_t context, uint64_t callback) {
    auto* host_queue = reinterpret_cast<Pg6XTaskQueueHandle>(static_cast<uintptr_t>(queue));
    if (callback == 0) {
        return pg6_gdk_xtask_queue_submit_callback(host_queue, port, nullptr, nullptr);
    }
    HostCallback<Pg6XTaskQueueCallback> cb = adapt_task_queue_callback(callback, context);
    const int32_t hr = pg6_gdk_xtask_queue_submit_callback(host_queue, port, cb.context, cb.function);
    if (hr < 0) {
        destroy_trampoline(cb.trampoline);
    }
    return hr;
}

int32_t terminate_impl(uint64_t queue, bool wait, uint64_t context, uint64_t callback) {
    auto* host_queue = reinterpret_cast<Pg6XTaskQueueHandle>(static_cast<uintptr_t>(queue));
    if (callback == 0) {
        return pg6_gdk_xtask_queue_terminate(host_queue, wait ? 1 : 0, nullptr, nullptr);
    }
    HostCallback<Pg6XTaskQueueTerminatedCallback> cb = adapt_task_queue_terminated_callback(callback, context);
    const int32_t hr = pg6_gdk_xtask_queue_terminate(host_queue, wait ? 1 : 0, cb.context, cb.function);
    if (hr < 0) {
        destroy_trampoline(cb.trampoline);
    }
    return hr;
}

// Merge sort of element indices with the guest comparator (no out-of-bounds access even if the
// comparator is inconsistent).
void merge_sort(std::vector<uint64_t>& idx,
                std::vector<uint64_t>& tmp,
                size_t lo,
                size_t hi,
                const auto& less_or_equal) {
    if (hi - lo < 2) {
        return;
    }
    const size_t mid = lo + (hi - lo) / 2;
    merge_sort(idx, tmp, lo, mid, less_or_equal);
    merge_sort(idx, tmp, mid, hi, less_or_equal);
    size_t a = lo;
    size_t b = mid;
    size_t out = lo;
    while (a < mid && b < hi) {
        tmp[out++] = less_or_equal(idx[a], idx[b]) ? idx[a++] : idx[b++];
    }
    while (a < mid) {
        tmp[out++] = idx[a++];
    }
    while (b < hi) {
        tmp[out++] = idx[b++];
    }
    for (size_t i = lo; i < hi; ++i) {
        idx[i] = tmp[i];
    }
}

void qsort_impl(CallContext& ctx, uint64_t base, uint64_t count, uint64_t size, uint64_t compare) {
    if (count < 2 || size == 0) {
        return;
    }
    if (compare == 0 || base == 0 || count > (uint64_t{1} << 32) || size > (uint64_t{1} << 24) ||
        count * size > (uint64_t{1} << 34)) {
        com::detail::diagnostic("guest callbacks: qsort with invalid arguments ignored");
        return;
    }
    const com::GuestFunction<int32_t(uint64_t, uint64_t)> cmp(compare);
    std::vector<uint64_t> idx(static_cast<size_t>(count));
    for (size_t i = 0; i < idx.size(); ++i) {
        idx[i] = i;
    }
    std::vector<uint64_t> tmp(idx.size());
    bool abandoned = false;
    merge_sort(idx, tmp, 0, idx.size(), [&](uint64_t a, uint64_t b) {
        if (abandoned) {
            return true;  // no further guest calls; the order no longer matters
        }
        const bool le = cmp.call(ctx.state, base + a * size, base + b * size) <= 0;
        abandoned = unwind_pending(ctx.state);
        return le;
    });
    if (abandoned) {
        // The comparator raised and an outer guest frame unwinds: leave the array as it was (the
        // CRT's qsort leaves it partially sorted; either is allowed after a throwing comparator).
        return;
    }
    std::vector<uint8_t> original(static_cast<size_t>(count * size));
    if (!com::read_guest(base, original.data(), original.size())) {
        com::detail::log_bad_pointer("qsort array", base);
        return;
    }
    std::vector<uint8_t> sorted(original.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        std::memcpy(sorted.data() + i * size, original.data() + idx[i] * size, static_cast<size_t>(size));
    }
    if (!com::write_guest(base, sorted.data(), sorted.size())) {
        com::detail::log_bad_pointer("qsort array", base);
    }
}

GuestPtr<void> bsearch_impl(CallContext& ctx,
                            uint64_t key,
                            uint64_t base,
                            uint64_t count,
                            uint64_t size,
                            uint64_t compare) {
    if (compare == 0 || count == 0) {
        return GuestPtr<void>();
    }
    const com::GuestFunction<int32_t(uint64_t, uint64_t)> cmp(compare);
    uint64_t lo = 0;
    uint64_t hi = count;
    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        const uint64_t element = base + mid * size;
        const int32_t c = cmp.call(ctx.state, key, element);
        if (unwind_pending(ctx.state)) {
            return GuestPtr<void>();  // ignored: the thunk leaves rax to the restored context
        }
        if (c == 0) {
            return GuestPtr<void>(element);
        }
        if (c < 0) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return GuestPtr<void>();
}

void initterm_impl(CallContext& ctx, uint64_t first, uint64_t last) {
    for (uint64_t p = first; p < last; p += 8) {
        uint64_t fn = 0;
        if (!com::read_guest(p, fn)) {
            com::detail::log_bad_pointer("_initterm table entry", p);
            return;
        }
        if (fn != 0) {
            com::GuestFunction<void()>(fn).call(ctx.state);
            if (unwind_pending(ctx.state)) {
                return;
            }
        }
    }
}

int32_t initterm_e_impl(CallContext& ctx, uint64_t first, uint64_t last) {
    for (uint64_t p = first; p < last; p += 8) {
        uint64_t fn = 0;
        if (!com::read_guest(p, fn)) {
            com::detail::log_bad_pointer("_initterm_e table entry", p);
            return 0;
        }
        if (fn != 0) {
            const int32_t result = com::GuestFunction<int32_t()>(fn).call(ctx.state);
            if (unwind_pending(ctx.state)) {
                return 0;  // ignored, see bsearch
            }
            if (result != 0) {
                return result;
            }
        }
    }
    return 0;
}

uint64_t get_proc_address_impl(uint64_t module, uint64_t name_or_ordinal) {
    namespace win32 = playgta6::os::win32;
    std::string name;
    win32::ExportLookup found;
    if ((name_or_ordinal >> 16) == 0) {
        name = "#" + std::to_string(name_or_ordinal);
        found = win32::LookupExport(module, {}, static_cast<uint32_t>(name_or_ordinal));
    } else {
        // Read the NUL-terminated ASCII name from guest memory (bounded).
        for (uint64_t i = 0; i < 1024; ++i) {
            char c = 0;
            if (!com::read_guest(name_or_ordinal + i, &c, 1)) {
                com::detail::log_bad_pointer("GetProcAddress name", name_or_ordinal);
                win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
                return 0;
            }
            if (c == 0) {
                break;
            }
            name.push_back(c);
        }
        found = win32::LookupExport(module, name);
    }
    if (found.address == 0) {
        return 0;  // LookupExport set the last error
    }
    if (found.guest) {
        return found.address;  // an export of a recompiled guest module: a real guest VA
    }
    const uint64_t callable = guest_callable_for_proc(found.address);
    if (callable == 0) {
        // A host shim the generator could not give a guest-callable thunk (denied by the shim
        // audit, unmarshalable signature) or one the export table does not list. Optional APIs
        // fail cleanly so their callers take the fallback path.
        const char* reason = unavailable_reason_for_proc(found.address);
        com::detail::diagnostic(
            ("guest callbacks: GetProcAddress(" + hex(module) + ", " + name +
             ") found a host-only export with no guest-callable entry" +
             (reason != nullptr ? std::string(": ") + reason : std::string(" (not in the export thunk table)")))
                .c_str());
        win32::SetLastError(win32::ERROR_PROC_NOT_FOUND);
    }
    return callable;
}

// libHttpClient flat imports taking a guest XAsyncBlock: the shim completes a shadow block, whose
// completion entry calls the guest callback with the guest block (O22).
int32_t hc_cleanup_async_impl(uint64_t block) {
    Pg6XAsyncBlock* shadow = async_shadow_begin(block);
    return shadow != nullptr ? pg6_mw_hc_cleanup_async(shadow) : playgta6::os::gdk::E_INVALIDARG_HR;
}

int32_t hc_http_call_perform_async_impl(uint64_t call, uint64_t block) {
    Pg6XAsyncBlock* shadow = async_shadow_begin(block);
    if (shadow == nullptr) {
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }
    // The call handle is the shim's own opaque value, handed to the guest by HCHttpCallCreate.
    return pg6_mw_hc_http_call_perform_async(reinterpret_cast<Pg6HCCallHandle>(static_cast<uintptr_t>(call)),
                                             shadow);
}

}  // namespace

}  // namespace playgta6::guest::callbacks

namespace cb = playgta6::guest::callbacks;
namespace gcom = playgta6::guest::com;

extern "C" {

void pg6_gthunk_create_thread(void* state) {
    gcom::function_thunk<&cb::create_thread_impl>()(state);
}

void pg6_gthunk_xtask_queue_submit_callback(void* state) {
    gcom::function_thunk<&cb::submit_callback_impl>()(state);
}

void pg6_gthunk_xtask_queue_terminate(void* state) {
    gcom::function_thunk<&cb::terminate_impl>()(state);
}

void pg6_gthunk_qsort(void* state) {
    gcom::function_thunk<&cb::qsort_impl>()(state);
}

void pg6_gthunk_bsearch(void* state) {
    gcom::function_thunk<&cb::bsearch_impl>()(state);
}

void pg6_gthunk_initterm(void* state) {
    gcom::function_thunk<&cb::initterm_impl>()(state);
}

void pg6_gthunk_initterm_e(void* state) {
    gcom::function_thunk<&cb::initterm_e_impl>()(state);
}

void pg6_gthunk_get_proc_address(void* state) {
    gcom::function_thunk<&cb::get_proc_address_impl>()(state);
}

void pg6_gthunk_hc_cleanup_async(void* state) {
    gcom::function_thunk<&cb::hc_cleanup_async_impl>()(state);
}

void pg6_gthunk_hc_http_call_perform_async(void* state) {
    gcom::function_thunk<&cb::hc_http_call_perform_async_impl>()(state);
}

}  // extern "C"
