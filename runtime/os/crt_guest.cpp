#include "os/crt_guest.h"

#include <atomic>

#include "guest/com_bridge.h"

namespace playgta6::os::crt_guest {

namespace {

std::atomic<uint64_t> g_user_matherr{0};
std::atomic<uint64_t> g_invalid_parameter{0};

void set_user_matherr(uint64_t handler) {
    g_user_matherr.store(handler, std::memory_order_release);
}

uint64_t set_invalid_parameter(uint64_t handler) {
    return g_invalid_parameter.exchange(handler, std::memory_order_acq_rel);
}

}  // namespace

uint64_t user_matherr_handler() {
    return g_user_matherr.load(std::memory_order_acquire);
}

uint64_t invalid_parameter_handler() {
    return g_invalid_parameter.load(std::memory_order_acquire);
}

void reset_handlers() {
    g_user_matherr.store(0, std::memory_order_release);
    g_invalid_parameter.store(0, std::memory_order_release);
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

}  // extern "C"
