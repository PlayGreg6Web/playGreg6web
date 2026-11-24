#include "guest/guest_crt_exit.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "os/os_shim.h"
#include "os/ucrt_bridge.h"

namespace playgta6::guest::crt_exit {

namespace com = playgta6::guest::com;

namespace {

constexpr uint64_t kInitialEntries = 32;
constexpr uint64_t kMaxGrowth = 512;
constexpr uint64_t kMaxEntries = uint64_t{1} << 24;

struct ExitState {
    std::recursive_mutex lock;                // UCRT's exit lock
    std::unordered_set<uint64_t> buffers;     // entry buffers allocated here
    uint64_t atexit_table = 0;
    uint64_t at_quick_exit_table = 0;
    std::atomic<uint64_t> tls_callback{0};
    std::atomic<bool> tls_callback_called{false};
};

ExitState& exit_state() {
    static ExitState instance;
    return instance;
}

std::string hex(uint64_t value) {
    char buffer[24];
    std::snprintf(buffer, sizeof(buffer), "0x%016" PRIx64, value);
    return buffer;
}

bool read_table(uint64_t table_va, GuestOnexitTable& table) {
    if (table_va == 0 || !com::read_guest(table_va, table)) {
        com::detail::log_bad_pointer("_onexit_table_t", table_va);
        return false;
    }
    return true;
}

bool is_empty(const GuestOnexitTable& t) {
    return t.first == 0 || t.first == kTableSentinel;
}

// Caller holds the exit lock.
void free_buffer(ExitState& s, uint64_t buffer) {
    if (buffer != 0 && s.buffers.erase(buffer) != 0) {
        com::guest_free(buffer);
    }
}

// Caller holds the exit lock.
uint64_t process_table(ExitState& s, uint64_t ExitState::*member) {
    if (s.*member == 0) {
        s.*member = com::guest_alloc(sizeof(GuestOnexitTable));  // zero-filled: an empty table
        if (s.*member == 0) {
            com::detail::diagnostic("crt exit: out of guest memory for the process exit table");
        }
    }
    return s.*member;
}

}  // namespace

int32_t initialize_table(uint64_t table_va) {
    ExitState& s = exit_state();
    std::lock_guard<std::recursive_mutex> lock(s.lock);
    GuestOnexitTable t{};
    if (!read_table(table_va, t)) {
        return -1;
    }
    if (t.first != t.end) {
        return 0;  // already initialized
    }
    const GuestOnexitTable empty{0, 0, 0};
    return com::write_guest(table_va, empty) ? 0 : -1;
}

int32_t register_function(uint64_t table_va, uint64_t function) {
    ExitState& s = exit_state();
    std::lock_guard<std::recursive_mutex> lock(s.lock);
    GuestOnexitTable t{};
    if (!read_table(table_va, t)) {
        return -1;
    }
    if (function == 0) {
        return 0;
    }
    if (is_empty(t)) {
        t = GuestOnexitTable{0, 0, 0};
    }
    if (t.last < t.first || t.end < t.last || ((t.last - t.first) % 8) != 0 || ((t.end - t.first) % 8) != 0) {
        com::detail::diagnostic(("crt exit: corrupt _onexit_table_t at " + hex(table_va)).c_str());
        return -1;
    }
    if (t.last == t.end) {
        const uint64_t count = (t.last - t.first) / 8;
        const uint64_t capacity = count == 0 ? kInitialEntries : count + std::min(count, kMaxGrowth);
        if (capacity > kMaxEntries) {
            return -1;
        }
        const uint64_t buffer = com::guest_alloc(capacity * 8);
        if (buffer == 0) {
            com::detail::diagnostic("crt exit: out of guest memory growing an onexit table");
            return -1;
        }
        if (count != 0) {
            std::vector<uint64_t> entries(static_cast<size_t>(count));
            if (!com::read_guest(t.first, entries.data(), entries.size() * 8) ||
                !com::write_guest(buffer, entries.data(), entries.size() * 8)) {
                com::guest_free(buffer);
                com::detail::log_bad_pointer("onexit table entries", t.first);
                return -1;
            }
        }
        free_buffer(s, t.first);
        s.buffers.insert(buffer);
        t = GuestOnexitTable{buffer, buffer + count * 8, buffer + capacity * 8};
    }
    if (!com::write_guest(t.last, function)) {
        com::detail::log_bad_pointer("onexit table entry", t.last);
        return -1;
    }
    t.last += 8;
    return com::write_guest(table_va, t) ? 0 : -1;
}

int32_t execute_table(GuestState& state, uint64_t table_va) {
    ExitState& s = exit_state();
    std::lock_guard<std::recursive_mutex> lock(s.lock);
    GuestOnexitTable t{};
    if (!read_table(table_va, t)) {
        return -1;
    }
    if (is_empty(t)) {
        return 0;
    }
    uint64_t first = t.first;
    uint64_t saved_first = t.first;
    uint64_t saved_last = t.last;
    uint64_t p = t.last;
    while (p > first) {
        p -= 8;
        uint64_t function = 0;
        if (!com::read_guest(p, function)) {
            com::detail::log_bad_pointer("onexit table entry", p);
            break;
        }
        if (function == 0) {
            continue;
        }
        const uint64_t cleared = 0;
        com::write_guest(p, cleared);  // each entry runs at most once
        com::GuestFunction<void()>(function).call(state);
        if (state.unwind_pending != 0) {
            return 0;  // a context restore (longjmp) out of a handler: leave at once
        }
        GuestOnexitTable now{};
        if (!com::read_guest(table_va, now)) {
            break;
        }
        if (now.first != saved_first || now.last != saved_last) {
            // A handler registered more handlers (or executed the table itself): restart from
            // the new end; entries already run are cleared.
            saved_first = now.first;
            saved_last = now.last;
            if (is_empty(now)) {
                break;
            }
            first = now.first;
            p = now.last;
        }
    }
    GuestOnexitTable now{};
    if (com::read_guest(table_va, now) && !is_empty(now)) {
        free_buffer(s, now.first);
    }
    const GuestOnexitTable empty{0, 0, 0};
    com::write_guest(table_va, empty);
    return 0;
}

uint64_t atexit_table() {
    ExitState& s = exit_state();
    std::lock_guard<std::recursive_mutex> lock(s.lock);
    return process_table(s, &ExitState::atexit_table);
}

uint64_t at_quick_exit_table() {
    ExitState& s = exit_state();
    std::lock_guard<std::recursive_mutex> lock(s.lock);
    return process_table(s, &ExitState::at_quick_exit_table);
}

void set_thread_local_exit_callback(uint64_t function) {
    uint64_t expected = 0;
    if (!exit_state().tls_callback.compare_exchange_strong(expected, function) && function != expected) {
        com::detail::diagnostic(("crt exit: second thread-local exit callback " + hex(function) +
                                 " ignored (already registered: " + hex(expected) + ")")
                                    .c_str());
    }
}

uint64_t thread_local_exit_callback() {
    return exit_state().tls_callback.load();
}

void run_exit_cleanup(GuestState& state, ExitMode mode) {
    ExitState& s = exit_state();
    std::lock_guard<std::recursive_mutex> lock(s.lock);
    if (mode == ExitMode::Full) {
        const uint64_t callback = s.tls_callback.load();
        if (callback != 0 && !s.tls_callback_called.exchange(true)) {
            // Thread-local destructors of the exiting thread run before the atexit handlers.
            com::GuestFunction<void(uint64_t, uint32_t, uint64_t)>(callback).call(state, 0, kDllProcessDetach, 0);
            if (state.unwind_pending != 0) {
                return;
            }
        }
        if (s.atexit_table != 0) {
            execute_table(state, s.atexit_table);
        }
    } else if (mode == ExitMode::Quick) {
        if (s.at_quick_exit_table != 0) {
            execute_table(state, s.at_quick_exit_table);
        }
    }
}

void exit_process(GuestState& state, int32_t code, ExitMode mode) {
    run_exit_cleanup(state, mode);
    if (mode == ExitMode::Full) {
        pg6_crt_exit(code);
    } else {
        pg6_crt__exit(code);
    }
    os::log("crt exit: guest exit(" + std::to_string(code) +
            ") was not stopped by an exit hook; ending the host process");
    std::_Exit(code);
}

void reset_crt_exit() {
    ExitState& s = exit_state();
    std::lock_guard<std::recursive_mutex> lock(s.lock);
    s.buffers.clear();
    s.atexit_table = 0;
    s.at_quick_exit_table = 0;
    s.tls_callback.store(0);
    s.tls_callback_called.store(false);
}

// ---------------------------------------------------------------------------
// Import thunk implementations (marshaled by com::function_thunk)
// ---------------------------------------------------------------------------

namespace {

using com::CallContext;

int32_t crt_atexit_impl(uint64_t function) {
    const uint64_t table = atexit_table();
    return table != 0 ? register_function(table, function) : -1;
}

int32_t crt_at_quick_exit_impl(uint64_t function) {
    const uint64_t table = at_quick_exit_table();
    return table != 0 ? register_function(table, function) : -1;
}

int32_t register_onexit_function_impl(uint64_t table, uint64_t function) {
    return register_function(table, function);
}

int32_t initialize_onexit_table_impl(uint64_t table) {
    return initialize_table(table);
}

int32_t execute_onexit_table_impl(CallContext& ctx, uint64_t table) {
    return execute_table(ctx.state, table);
}

void register_thread_local_exe_atexit_callback_impl(uint64_t function) {
    set_thread_local_exit_callback(function);
}

void exit_impl(CallContext& ctx, int32_t code) {
    exit_process(ctx.state, code, ExitMode::Full);
}

void quick_exit_impl(CallContext& ctx, int32_t code) {
    exit_process(ctx.state, code, ExitMode::Quick);
}

void underscore_exit_impl(CallContext& ctx, int32_t code) {
    exit_process(ctx.state, code, ExitMode::None);
}

void cexit_impl(CallContext& ctx) {
    run_exit_cleanup(ctx.state, ExitMode::Full);
    pg6_crt_cexit();  // host-registered handlers too
}

void c_exit_impl() {}

}  // namespace

}  // namespace playgta6::guest::crt_exit

namespace ce = playgta6::guest::crt_exit;
namespace gcom = playgta6::guest::com;

extern "C" {

void pg6_gthunk_crt_atexit(void* state) {
    gcom::function_thunk<&ce::crt_atexit_impl>()(state);
}

void pg6_gthunk_crt_at_quick_exit(void* state) {
    gcom::function_thunk<&ce::crt_at_quick_exit_impl>()(state);
}

void pg6_gthunk_register_onexit_function(void* state) {
    gcom::function_thunk<&ce::register_onexit_function_impl>()(state);
}

void pg6_gthunk_initialize_onexit_table(void* state) {
    gcom::function_thunk<&ce::initialize_onexit_table_impl>()(state);
}

void pg6_gthunk_execute_onexit_table(void* state) {
    gcom::function_thunk<&ce::execute_onexit_table_impl>()(state);
}

void pg6_gthunk_register_thread_local_exe_atexit_callback(void* state) {
    gcom::function_thunk<&ce::register_thread_local_exe_atexit_callback_impl>()(state);
}

void pg6_gthunk_exit(void* state) {
    gcom::function_thunk<&ce::exit_impl>()(state);
}

void pg6_gthunk_quick_exit(void* state) {
    gcom::function_thunk<&ce::quick_exit_impl>()(state);
}

void pg6_gthunk__exit(void* state) {
    gcom::function_thunk<&ce::underscore_exit_impl>()(state);
}

void pg6_gthunk__cexit(void* state) {
    gcom::function_thunk<&ce::cexit_impl>()(state);
}

void pg6_gthunk__c_exit(void* state) {
    gcom::function_thunk<&ce::c_exit_impl>()(state);
}

}  // extern "C"
