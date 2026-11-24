#pragma once

#include <cstdint>

// Guest-state import thunks for C-runtime imports that take a guest callback (Track 2, O17).
//
// The pg6_crt_ shims of os/ucrt_bridge.h for these imports take a host function pointer, so the
// import bridge cannot bind them (a guest code address must never be stored or called as a host
// function pointer). The thunks below implement the imports for recompiled callers instead and
// keep the guest handler as a guest address:
//
//   __setusermatherr(handler)                 records the handler. The runtime's math shims never
//                                             report a _matherr condition, so it is never called.
//   _set_invalid_parameter_handler(handler)   swaps the recorded handler and returns the previous
//                                             one (0 at first). The runtime's
//                                             _invalid_parameter_noinfo does not call it: the
//                                             failing CRT call returns its error code instead.
//
// The import bridge binds them by name key (pg6_gthunk_setusermatherr <- __setusermatherr) ahead
// of the family shims.

namespace playgta6::os::crt_guest {

// The recorded guest handler addresses (0 = none).
uint64_t user_matherr_handler();
uint64_t invalid_parameter_handler();
// Tests: forget the recorded handlers.
void reset_handlers();

}  // namespace playgta6::os::crt_guest

extern "C" {

void pg6_gthunk_setusermatherr(void* state);
void pg6_gthunk_set_invalid_parameter_handler(void* state);

}  // extern "C"
