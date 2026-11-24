#include "guest/guest_rtl.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_exceptions.h"
#include "os/win32_kernel.h"
#include "os/win32_rtl.h"
#include "os/win32_sync.h"

#if defined(__cpp_exceptions) || defined(_CPPUNWIND) || defined(__EXCEPTIONS)
#define PG6_RTL_TRY try
#define PG6_RTL_CATCH_ALL catch (...)
#else
#define PG6_RTL_TRY if (true)
#define PG6_RTL_CATCH_ALL else if (false)
#endif

namespace playgta6::guest::rtl {

namespace com = playgta6::guest::com;
namespace win32 = playgta6::os::win32;

namespace {

// ---------------------------------------------------------------------------
// FLS
// ---------------------------------------------------------------------------

bool fls_invoke(void* ctx, uint64_t callback, uint64_t value) {
    auto* state = static_cast<GuestState*>(ctx);
    com::GuestFunction<void(uint64_t)>(callback).call_top_level(*state, value);
    return !unwind_pending(*state);
}

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

// Persistent record of one guest work object (its callback may run many times).
struct WorkRecord {
    uint64_t function = 0;
    uint64_t context = 0;
    std::atomic<uint32_t> pending{0};  // submitted callbacks that have not run yet
    bool closed = false;                // guarded by the registry mutex
};

// One-shot record of a TrySubmitThreadpoolCallback.
struct SimpleRecord {
    uint64_t function = 0;
    uint64_t context = 0;
};

struct PoolRegistry {
    std::mutex mutex;
    std::unordered_map<uint64_t, WorkRecord*> works;  // key: the PTP_WORK value handed to the guest
    std::atomic<uint64_t> simple_live{0};
};

PoolRegistry& pool() {
    static auto* instance = new PoolRegistry();
    return *instance;
}

void release_work_record_if_done(WorkRecord* record) {
    PoolRegistry& reg = pool();
    bool destroy = false;
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        destroy = record->closed && record->pending.load(std::memory_order_acquire) == 0;
    }
    if (destroy) {
        delete record;
    }
}

void work_entry(Pg6TpCallbackInstance* instance, void* context, Pg6TpWork* work) {
    auto* record = static_cast<WorkRecord*>(context);
    const uint64_t function = record->function;
    const uint64_t guest_context = record->context;
    PG6_RTL_TRY {
        GuestState& state = com::current_guest_state();
        com::GuestFunction<void(uint64_t, uint64_t, uint64_t)>(function).call_top_level(
            state, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(instance)), guest_context,
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(work)));
    }
    PG6_RTL_CATCH_ALL {
        // A fault inside a pool callback has been reported by the fault handler.
    }
    record->pending.fetch_sub(1, std::memory_order_acq_rel);
    release_work_record_if_done(record);
}

void simple_entry(Pg6TpCallbackInstance* instance, void* context) {
    std::unique_ptr<SimpleRecord> record(static_cast<SimpleRecord*>(context));
    pool().simple_live.fetch_sub(1, std::memory_order_relaxed);
    PG6_RTL_TRY {
        GuestState& state = com::current_guest_state();
        com::GuestFunction<void(uint64_t, uint64_t)>(record->function).call_top_level(
            state, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(instance)), record->context);
    }
    PG6_RTL_CATCH_ALL {
    }
}

WorkRecord* find_work(uint64_t token, const char* api) {
    PoolRegistry& reg = pool();
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = reg.works.find(token);
        if (it != reg.works.end()) {
            return it->second;
        }
    }
    char text[128];
    std::snprintf(text, sizeof(text), "%s: unknown thread-pool work 0x%016" PRIx64, api, token);
    com::detail::diagnostic(text);
    return nullptr;
}

// ---------------------------------------------------------------------------
// Import thunk implementations
// ---------------------------------------------------------------------------

int32_t fls_free_impl(com::CallContext& ctx, uint32_t index) {
    return win32::FlsFree(index, &fls_invoke, &ctx.state) ? 1 : 0;
}

int32_t init_once_impl(com::CallContext& ctx,
                       uint64_t once_va,
                       uint64_t callback,
                       uint64_t parameter,
                       com::GuestPtr<uint64_t> context_out) {
    auto* once = static_cast<Pg6InitOnce*>(com::guest_to_host(once_va, sizeof(Pg6InitOnce)));
    if (once == nullptr || callback == 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    GuestState& state = ctx.state;
    uint64_t result = 0;
    const bool ok = win32::InitOnceRun(
        once,
        [&](uint64_t* context) {
            const uint64_t slot = com::guest_alloc(sizeof(uint64_t));  // PVOID* Context, zeroed
            if (slot == 0) {
                win32::SetLastError(win32::ERROR_NOT_ENOUGH_MEMORY);
                return win32::InitOnceOutcome::Failed;
            }
            // Not a top-level call: an exception leaving the callback reaches the caller's frames.
            const int32_t r = com::GuestFunction<int32_t(uint64_t, uint64_t, uint64_t)>(callback).call(
                state, once_va, parameter, slot);
            uint64_t value = 0;
            const bool unwinding = unwind_pending(state);
            if (!unwinding) {
                com::read_guest(slot, value);
            }
            com::guest_free(slot);
            if (unwinding) {
                return win32::InitOnceOutcome::Abandoned;
            }
            *context = value;
            return r != 0 ? win32::InitOnceOutcome::Completed : win32::InitOnceOutcome::Failed;
        },
        &result);
    if (unwind_pending(state)) {
        return 0;
    }
    if (ok && !context_out.is_null() && !context_out.write(result)) {
        com::detail::log_bad_pointer("InitOnceExecuteOnce context", context_out.va());
    }
    return ok ? 1 : 0;
}

uint16_t capture_back_trace_impl(com::CallContext& ctx,
                                 uint32_t frames_to_skip,
                                 uint32_t frames_to_capture,
                                 com::GuestPtr<uint64_t> back_trace,
                                 com::GuestPtr<uint32_t> back_trace_hash) {
    constexpr uint32_t kMaxFrames = 1024;
    if (!back_trace_hash.is_null()) {
        back_trace_hash.write(0);
    }
    if (frames_to_capture == 0) {
        return 0;
    }
    if (back_trace.is_null()) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    os::VirtualMemoryArena& arena = com::guest_memory();
    GuestContext rec{};
    if (!capture_caller_context(arena, ctx.state, rec)) {
        return 0;
    }
    std::vector<uint64_t> frames;
    uint32_t skipped = 0;
    uint32_t steps = 0;
    while (steps++ < kMaxFrames && frames.size() < frames_to_capture) {
        const uint64_t pc = rec.rip;
        if (pc == 0 || pc_to_image_base(pc) == 0) {
            break;  // not guest code: the boundary of the outermost guest call
        }
        if (skipped < frames_to_skip) {
            ++skipped;
        } else {
            frames.push_back(pc);
        }
        const uint64_t old_rsp = rec.gpr[GuestReg::Rsp];
        uint64_t image_base = 0;
        const uint64_t entry = lookup_function_entry(arena, pc, &image_base);
        if (entry == 0) {
            uint64_t return_address = 0;  // leaf function: the return address is at [rsp]
            if (!com::read_guest(old_rsp, return_address)) {
                break;
            }
            rec.rip = return_address;
            rec.gpr[GuestReg::Rsp] = old_rsp + 8;
        } else {
            uint64_t handler_data = 0;
            uint64_t frame = 0;
            bool ok = false;
            virtual_unwind(arena, 0, image_base, pc, entry, rec, &handler_data, &frame, nullptr, &ok);
            if (!ok) {
                break;
            }
        }
        if (rec.gpr[GuestReg::Rsp] <= old_rsp) {
            break;  // no progress: corrupt unwind data
        }
    }
    if (!frames.empty() &&
        !com::write_guest(back_trace.va(), frames.data(), frames.size() * sizeof(uint64_t))) {
        com::detail::log_bad_pointer("RtlCaptureStackBackTrace buffer", back_trace.va());
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    uint32_t hash = 0;
    for (const uint64_t frame : frames) {
        hash += static_cast<uint32_t>(frame);
    }
    if (!back_trace_hash.is_null()) {
        back_trace_hash.write(hash);
    }
    return static_cast<uint16_t>(frames.size());
}

uint64_t create_work_impl(uint64_t callback, uint64_t context, uint64_t callback_environ) {
    (void)callback_environ;
    if (callback == 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    auto* record = new WorkRecord();
    record->function = callback;
    record->context = context;
    Pg6TpWork* work = win32::CreateThreadpoolWork(&work_entry, record, nullptr);
    if (work == nullptr) {
        delete record;
        return 0;
    }
    const uint64_t token = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(work));
    PoolRegistry& reg = pool();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.works[token] = record;
    return token;
}

void submit_work_impl(uint64_t token) {
    WorkRecord* record = find_work(token, "SubmitThreadpoolWork");
    if (record == nullptr) {
        return;
    }
    record->pending.fetch_add(1, std::memory_order_acq_rel);
    win32::SubmitThreadpoolWork(reinterpret_cast<Pg6TpWork*>(static_cast<uintptr_t>(token)));
}

void wait_work_impl(uint64_t token, bool cancel_pending) {
    WorkRecord* record = find_work(token, "WaitForThreadpoolWorkCallbacks");
    if (record == nullptr) {
        return;
    }
    win32::WaitForThreadpoolWorkCallbacks(reinterpret_cast<Pg6TpWork*>(static_cast<uintptr_t>(token)),
                                          cancel_pending);
    if (cancel_pending) {
        // Everything is idle now: submissions cancelled before they ran never call work_entry.
        record->pending.store(0, std::memory_order_release);
    }
}

void close_work_impl(uint64_t token) {
    PoolRegistry& reg = pool();
    WorkRecord* record = nullptr;
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = reg.works.find(token);
        if (it != reg.works.end()) {
            record = it->second;
            reg.works.erase(it);
            record->closed = true;
        }
    }
    if (record == nullptr) {
        char text[128];
        std::snprintf(text, sizeof(text), "CloseThreadpoolWork: unknown thread-pool work 0x%016" PRIx64, token);
        com::detail::diagnostic(text);
        return;
    }
    win32::CloseThreadpoolWork(reinterpret_cast<Pg6TpWork*>(static_cast<uintptr_t>(token)));
    release_work_record_if_done(record);
}

int32_t try_submit_callback_impl(uint64_t callback, uint64_t context, uint64_t callback_environ) {
    (void)callback_environ;
    if (callback == 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    auto* record = new SimpleRecord{callback, context};
    pool().simple_live.fetch_add(1, std::memory_order_relaxed);
    if (!win32::TrySubmitThreadpoolCallback(&simple_entry, record, nullptr)) {
        pool().simple_live.fetch_sub(1, std::memory_order_relaxed);
        delete record;
        return 0;
    }
    return 1;
}

}  // namespace

void run_fls_thread_exit(GuestState& state) {
    PG6_RTL_TRY {
        win32::fls_run_thread_exit(&fls_invoke, &state);
    }
    PG6_RTL_CATCH_ALL {
        win32::fls_run_thread_exit(nullptr, nullptr);  // a fault ended the destructors: just release
    }
}

uint64_t live_pool_records() {
    PoolRegistry& reg = pool();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return reg.works.size() + reg.simple_live.load(std::memory_order_relaxed);
}

void reset_rtl() {
    PoolRegistry& reg = pool();
    std::lock_guard<std::mutex> lock(reg.mutex);
    for (auto& [token, record] : reg.works) {
        (void)token;
        delete record;
    }
    reg.works.clear();
    reg.simple_live.store(0, std::memory_order_relaxed);
}

}  // namespace playgta6::guest::rtl

namespace rtl = playgta6::guest::rtl;
namespace gcom = playgta6::guest::com;

extern "C" {

void pg6_gthunk_fls_free(void* state) {
    gcom::function_thunk<&rtl::fls_free_impl>()(state);
}

void pg6_gthunk_init_once_execute_once(void* state) {
    gcom::function_thunk<&rtl::init_once_impl>()(state);
}

void pg6_gthunk_rtl_capture_stack_back_trace(void* state) {
    gcom::function_thunk<&rtl::capture_back_trace_impl>()(state);
}

void pg6_gthunk_create_threadpool_work(void* state) {
    gcom::function_thunk<&rtl::create_work_impl>()(state);
}

void pg6_gthunk_submit_threadpool_work(void* state) {
    gcom::function_thunk<&rtl::submit_work_impl>()(state);
}

void pg6_gthunk_wait_for_threadpool_work_callbacks(void* state) {
    gcom::function_thunk<&rtl::wait_work_impl>()(state);
}

void pg6_gthunk_close_threadpool_work(void* state) {
    gcom::function_thunk<&rtl::close_work_impl>()(state);
}

void pg6_gthunk_try_submit_threadpool_callback(void* state) {
    gcom::function_thunk<&rtl::try_submit_callback_impl>()(state);
}

}  // extern "C"
