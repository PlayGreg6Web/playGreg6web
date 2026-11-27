// Windows only: std::set_terminate is per thread in the MSVC runtime, so the handler installed in
// main() covers main() only; std::terminate() on any other thread (a guest unwind crossing a
// noexcept frame, a throwing std::thread body) fell through to abort(), where the CRT resets the
// SIGABRT handler while it runs it: with several threads terminating at once the later ones got no
// handler and the process died (exit 3, or 0xC0000409 fail-fast) with no report. A TLS callback of
// the executable runs on every new thread and installs the handler there. Include in exactly ONE
// translation unit of an executable and call install() once from main().
#pragma once
#ifdef _WIN32
#include <windows.h>

#include <exception>

namespace playgta6::guest::terminate_hook {

inline std::terminate_handler g_handler = nullptr;

inline void NTAPI tls_callback(PVOID, DWORD reason, PVOID) {
    if (reason == DLL_THREAD_ATTACH && g_handler != nullptr) std::set_terminate(g_handler);
}

inline void install(std::terminate_handler handler) {
    g_handler = handler;
    std::set_terminate(handler);  // this (main) thread
}

}  // namespace playgta6::guest::terminate_hook

#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:pg6_terminate_tls_callback")
#pragma const_seg(".CRT$XLB")
extern "C" const PIMAGE_TLS_CALLBACK pg6_terminate_tls_callback = &playgta6::guest::terminate_hook::tls_callback;
#pragma const_seg()
#endif
