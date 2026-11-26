#pragma once
// Windows host-fault minidump writer. MiniDumpWriteDump must not run on the faulting thread (its
// stack may be exhausted/corrupt and the process state is mid-exception), so a dedicated dumper
// thread is created up front, with dbghelp loaded and resolved before any fault. The fault
// handler only fills a request and signals it, then waits (bounded) so the exception pointers stay
// valid. Flags are small: thread info + the stacks the dumper walks + module list, never
// indirectly referenced memory (the guest's huge reserved address space made the writer crash and
// produce empty dumps) and never full memory.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// After windows.h.
#include <dbghelp.h>

#include <atomic>
#include <cstdio>
#include <string>

namespace playgta6::guest::minidump {

class Writer {
public:
    // Loads dbghelp and starts the dumper thread. Idempotent; false when unavailable (dumps are
    // then skipped, never crash).
    static bool init() {
        State& s = state();
        if (s.ready.load()) return true;
        HMODULE dbg = LoadLibraryA("dbghelp.dll");
        if (dbg == nullptr) return false;
        s.fn = reinterpret_cast<WriteFn>(GetProcAddress(dbg, "MiniDumpWriteDump"));
        s.request = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        s.done = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (s.fn == nullptr || s.request == nullptr || s.done == nullptr) return false;
        s.thread = CreateThread(nullptr, 1 << 20, &thread_main, nullptr, 0, nullptr);
        if (s.thread == nullptr) return false;
        s.ready.store(true);
        return true;
    }

    // Called from the faulting thread (handler context, or normal code with info == nullptr).
    // Blocks until the dumper finished or timeout_ms elapsed. Returns true when a dump was
    // written. Never throws; one dump per process.
    static bool dump(const char* path, EXCEPTION_POINTERS* info, DWORD timeout_ms = 20000) {
        State& s = state();
        if (!s.ready.load() || path == nullptr || s.busy.exchange(true)) return false;
        std::snprintf(s.path, sizeof(s.path), "%s", path);
        s.info = info;
        s.fault_thread = GetCurrentThreadId();
        s.ok = false;
        SetEvent(s.request);
        if (WaitForSingleObject(s.done, timeout_ms) != WAIT_OBJECT_0) return false;
        return s.ok;
    }

private:
    using WriteFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, DWORD, void*, void*, void*);
    struct State {
        std::atomic<bool> ready{false};
        std::atomic<bool> busy{false};
        WriteFn fn = nullptr;
        HANDLE request = nullptr, done = nullptr, thread = nullptr;
        char path[MAX_PATH * 2] = {};
        EXCEPTION_POINTERS* info = nullptr;
        DWORD fault_thread = 0;
        bool ok = false;
    };
    static State& state() {
        static State s;
        return s;
    }

    static DWORD WINAPI thread_main(void*) {
        State& s = state();
        for (;;) {
            if (WaitForSingleObject(s.request, INFINITE) != WAIT_OBJECT_0) return 0;
            s.ok = write(s);
            SetEvent(s.done);
        }
    }

    static bool write(State& s) {
        HANDLE file = CreateFileA(s.path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) return false;
        // The real struct: MINIDUMP_EXCEPTION_INFORMATION is pack(4) on every arch, so a hand-rolled
        // 8-aligned copy shifts the pointer field and MiniDumpWriteDump fails with ERROR_NOACCESS.
        MINIDUMP_EXCEPTION_INFORMATION ex{};
        ex.ThreadId = s.fault_thread;
        ex.ExceptionPointers = s.info;
        ex.ClientPointers = FALSE;
        constexpr DWORD kWithThreadInfo = 0x00001000;
        constexpr DWORD kWithUnloadedModules = 0x00000020;
        bool ok = false;
        __try {
            ok = s.fn(GetCurrentProcess(), GetCurrentProcessId(), file, kWithThreadInfo | kWithUnloadedModules,
                      s.info != nullptr ? &ex : nullptr, nullptr, nullptr) != FALSE;
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            ok = false;
        }
        if (!ok) std::fprintf(stderr, "[minidump] MiniDumpWriteDump failed: 0x%08lx\n", GetLastError());
        LARGE_INTEGER size{};
        const bool nonempty = GetFileSizeEx(file, &size) && size.QuadPart > 0;
        CloseHandle(file);
        return ok && nonempty;
    }
};

}  // namespace playgta6::guest::minidump
#endif
