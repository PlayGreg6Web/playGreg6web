#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "guest/com_bridge.h"

// xgameruntime.dll provider (O16).
//
// The game has no static imports from xgameruntime.dll. Its statically linked GDK client code
// loads the DLL by name, resolves QueryApiImpl (plus InitializeApiImpl / UninitializeApiImpl)
// with GetProcAddress and, for every GDK API call, asks QueryApiImpl for the COM interface of that
// API family, calls one vtable method and releases the interface again. This module registers a
// virtual xgameruntime.dll in the O7 module registry whose exports are guest-callable thunks
// (com::function_thunk, synthetic addresses from register_host_callable, so the guest-facing
// GetProcAddress thunk pg6_gthunk_get_proc_address hands out dispatchable addresses), and
// answers QueryApiImpl with singleton bridge objects (com_bridge.h) whose mapped slots forward to
// the existing pg6_gdk_* shims (xgameruntime.h / xbox_offline.h). Every other slot is a logging
// stub (SlotPolicy::Trap): the first real boot reports exactly which interface/slot it needs.
//
// The interface identities, slot numbers and arities below come from a static survey of the
// game executable's QueryApiImpl call sites (aggregate evidence only: GUIDs, slot indices,
// call-site counts and argument counts); the mapping of slots to GDK functions and its
// confidence is documented next to each interface in xgameruntime_provider.cpp.

namespace playgta6::os::gdk::provider {

// QueryApiImpl answers REGDB_E_CLASSNOTREG (and *out = NULL) for API identities it does not know:
// the game's client code turns exactly this result into 0x89240107 plus an XErrorReport; any other
// failure is returned by the API wrapper as is.
inline constexpr int32_t kUnknownApiResult = static_cast<int32_t>(0x80040154u);

// Registers the virtual module and its exports (idempotent, thread-safe). Returns the module
// handle (a synthetic HMODULE from the O7 registry), 0 on failure.
uint64_t register_xgameruntime_provider();

// Guest-callable synthetic addresses of the exports (0 before registration).
uint64_t query_api_address();
uint64_t initialize_api_address();
uint64_t uninitialize_api_address();

struct ApiRequest {
    std::string provider;   // formatted GUIDs
    std::string interface_id;
    uint64_t count = 0;
    bool known = false;
};

// XGameGetXboxTitleId result (default 0: the title id is not known to the runtime yet).
void set_xbox_title_id(uint32_t title_id);

// Every distinct QueryApiImpl request so far (known and unknown), sorted by GUID.
std::vector<ApiRequest> api_requests();

// Tests only: forget the singleton objects and request counters (call with reset_com_bridge()).
void reset_xgameruntime_provider();

}  // namespace playgta6::os::gdk::provider
