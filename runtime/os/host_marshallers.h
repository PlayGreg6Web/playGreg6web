#pragma once

#include <cstdint>

// Hand-written guest-state import thunks (pg6_gthunk_*, bound by the import bridge ahead of the
// pg6_* shims) for two imports the generic marshaller cannot build a thunk for.
//
//   inet_ntoa  (WS2_32.dll ordinal 11)   `char* inet_ntoa(struct in_addr in)`: a struct passed by
//       value (Win64 passes the 4-byte in_addr in ecx) and a result that must live in guest memory.
//       Per the Winsock contract the text is in a per-thread buffer that stays valid until the
//       thread's next inet_ntoa call. Pure formatting: no network access.
//
//   HCTraceImplMessage  (libHttpClient)  `void HCTraceImplMessage(const HCTraceImplArea* area,
//       HCTraceLevel level, const char* format, ...)`: C-variadic. The variadic arguments are
//       r9 then the guest stack slots after the shadow space; the format and every %s string are
//       read from guest memory with the checked accessors and formatted by the MSVC printf
//       formatter (os/crt_format.h). The line goes to the runtime log as
//       "libHttpClient [<area>] <level>: <message>". Bounded: at most kHcTraceMaxFormat format
//       characters, kHcTraceMaxArgs variadic slots, kHcTraceMaxMessage message characters
//       (longer output is cut and marked with "..."). A malformed format or an unreadable
//       argument is logged as the raw format text with a note, never a guest fault.
inline constexpr uint32_t kHcTraceMaxFormat = 2048;
inline constexpr uint32_t kHcTraceMaxArgs = 32;
inline constexpr uint32_t kHcTraceMaxMessage = 1024;

extern "C" {

void pg6_gthunk_inet_ntoa(void* state);
void pg6_gthunk_hc_trace_impl_message(void* state);

}  // extern "C"
