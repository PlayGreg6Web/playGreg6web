#include "os/crt_guest.h"

#include <atomic>

#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_runtime.h"

namespace playgta6::os::crt_guest {

namespace {

namespace gcom = playgta6::guest::com;

std::atomic<uint64_t> g_user_matherr{0};
std::atomic<uint64_t> g_invalid_parameter{0};
std::atomic<uint64_t> g_new_handler{0};

void set_user_matherr(uint64_t handler) {
    g_user_matherr.store(handler, std::memory_order_release);
}

uint64_t set_invalid_parameter(uint64_t handler) {
    return g_invalid_parameter.exchange(handler, std::memory_order_acq_rel);
}

uint64_t set_new(uint64_t handler) {
    return g_new_handler.exchange(handler, std::memory_order_acq_rel);
}

int32_t callnewh(gcom::CallContext& ctx, uint64_t size) {
    const uint64_t handler = g_new_handler.load(std::memory_order_acquire);
    if (handler == 0) {
        return 0;
    }
    const gcom::GuestFunction<int32_t(uint64_t)> fn(handler);
    const int32_t result = fn.call(ctx.state, size);
    if (playgta6::guest::unwind_pending(ctx.state)) {
        return 0;  // the handler threw: the unwind continues past this thunk
    }
    return result;
}

}  // namespace

uint64_t user_matherr_handler() {
    return g_user_matherr.load(std::memory_order_acquire);
}

uint64_t invalid_parameter_handler() {
    return g_invalid_parameter.load(std::memory_order_acquire);
}

uint64_t new_handler() {
    return g_new_handler.load(std::memory_order_acquire);
}

void reset_handlers() {
    g_user_matherr.store(0, std::memory_order_release);
    g_invalid_parameter.store(0, std::memory_order_release);
    g_new_handler.store(0, std::memory_order_release);
}

}  // namespace playgta6::os::crt_guest

namespace cg = playgta6::os::crt_guest;
namespace gcom = playgta6::guest::com;

extern "C" {

void pg6_gthunk_setusermatherr(void* state) {
    gcom::function_thunk<&cg::set_user_matherr>()(state);
}

void pg6_gthunk_set_invalid_parameter_handler(void* state) {
    gcom::function_thunk<&cg::set_invalid_parameter>()(state);
}

void pg6_gthunk_set_new_handler(void* state) {
    gcom::function_thunk<&cg::set_new>()(state);
}

void pg6_gthunk_callnewh(void* state) {
    gcom::function_thunk<&cg::callnewh>()(state);
}

void pg6_gthunk_beginthreadex(void* state) {
    // _beginthreadex(security, stack_size, start, arglist, initflag, thrdaddr) takes CreateThread's
    // arguments in CreateThread's order and returns the thread handle or 0.
    pg6_gthunk_create_thread(state);
}

}  // extern "C"
