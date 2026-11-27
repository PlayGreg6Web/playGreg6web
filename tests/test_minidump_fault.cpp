// Windows only: a child process faults on a host null dereference; the dump writer (a helper
// process started up front) must produce a non-empty dump and tools/minidump_stack.py must print
// the fault. --locked: another thread holds the process heap lock during the dump (it must not
// freeze the process); --hang: the watchdog path prints every thread's stack and writes a dump
// without an exception stream.
#ifdef _WIN32
#include "guest/minidump_writer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

using playgta6::guest::minidump::Writer;

static const char* g_dump_path = nullptr;

static LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    Writer::dump(g_dump_path, info);
    TerminateProcess(GetCurrentProcess(), 77);
    return EXCEPTION_CONTINUE_SEARCH;
}

#pragma optimize("", off)
static void deliberate_null_deref(volatile int* p) { *p = 1; }
#pragma optimize("", on)

static const char kFaultJson[] = "{\"kind\": \"unknown_target\", \"thread\": 7, \"rip\": \"0x1400\", \"registers\": {\"rax\": \"0x1\"}, "
                                  "\"source\": {\"return_address\": \"0x2000\"}, \"recent_entries\": [{\"target\": \"0x3000\", \"name\": null, \"from\": \"0x2000\"}]}";

#pragma optimize("", off)
static void guest_fault_path(const char* path) {
    CONTEXT ctx;
    RtlCaptureContext(&ctx);
    Writer::dump_guest_fault(path, &ctx, kFaultJson);
}
#pragma optimize("", on)

// Holds the process heap lock (as any thread inside HeapAlloc does at an unlucky moment) until the
// dump is done. An in-process MiniDumpWriteDump suspends this thread with the lock held and then
// needs the heap itself: the whole process froze (empty dump, watchdog never ran, #663).
static volatile LONG g_dump_finished = 0;
static HANDLE g_locked = nullptr;
static DWORD WINAPI hold_heap_lock(void*) {
    HeapLock(GetProcessHeap());
    SetEvent(g_locked);
    while (g_dump_finished == 0) SwitchToThread();
    HeapUnlock(GetProcessHeap());
    return 0;
}

static LONG CALLBACK locked_handler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    Writer::dump(g_dump_path, info);
    InterlockedExchange(&g_dump_finished, 1);
    TerminateProcess(GetCurrentProcess(), 77);
    return EXCEPTION_CONTINUE_SEARCH;
}

int main(int argc, char** argv) {
    if (const int helper = Writer::run_helper_if_requested(argc, argv); helper >= 0) return helper;
    if (argc >= 3 && std::strcmp(argv[1], "--child-locked") == 0) {
        g_dump_path = argv[2];
        if (!Writer::init()) return 3;
        g_locked = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        AddVectoredExceptionHandler(1, &locked_handler);
        HANDLE t = CreateThread(nullptr, 0, &hold_heap_lock, nullptr, 0, nullptr);
        if (t == nullptr || WaitForSingleObject(g_locked, 10000) != WAIT_OBJECT_0) return 5;
        deliberate_null_deref(nullptr);
        return 4;
    }
    if (argc >= 3 && std::strcmp(argv[1], "--child-hang") == 0) {
        // Watchdog path: no exception, every thread's stack printed by the helper plus a dump.
        if (!Writer::init()) return 3;
        g_locked = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        HANDLE t = CreateThread(nullptr, 0, &hold_heap_lock, nullptr, 0, nullptr);
        if (t == nullptr || WaitForSingleObject(g_locked, 10000) != WAIT_OBJECT_0) return 5;
        const bool ok = Writer::dump_threads(argv[2], "test hang");
        InterlockedExchange(&g_dump_finished, 1);
        TerminateProcess(GetCurrentProcess(), ok ? 77 : 6);
        return 6;
    }
    if (argc >= 3 && std::strcmp(argv[1], "--child-guest") == 0) {
        if (!Writer::init()) return 3;
        guest_fault_path(argv[2]);
        return 77;
    }
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0) {
        g_dump_path = argv[2];
        if (!Writer::init()) return 3;
        AddVectoredExceptionHandler(1, &handler);
        deliberate_null_deref(nullptr);
        return 4;
    }
    char exe[MAX_PATH];
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    char tmp[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    const std::string dump = std::string(tmp) + "pg6_minidump_fault_test_" + std::to_string(GetCurrentProcessId()) + ".dmp";
    DeleteFileA(dump.c_str());
    const std::string mode = argc >= 2 ? argv[1] : "";
    const bool guest_mode = mode == "--guest";
    const bool locked_mode = mode == "--locked";
    const bool hang_mode = mode == "--hang";
    const char* child = guest_mode ? "--child-guest" : locked_mode ? "--child-locked" : hang_mode ? "--child-hang" : "--child";
    std::string cmd = std::string("\"") + exe + "\" " + child + " \"" + dump + "\"";
    const std::string err_path = dump + ".stderr";
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE err = CreateFileA(err_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = err;
    si.hStdError = err;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) return 1;
    CloseHandle(err);
    const bool exited = WaitForSingleObject(pi.hProcess, 60000) == WAIT_OBJECT_0;
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    if (!exited) TerminateProcess(pi.hProcess, 99);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    std::string child_err;
    {
        std::ifstream e(err_path, std::ios::binary);
        child_err.assign((std::istreambuf_iterator<char>(e)), std::istreambuf_iterator<char>());
    }
    if (!exited) {
        std::fprintf(stderr, "FAIL: child hung (the dump froze the process)\n%s\n", child_err.c_str());
        return 1;
    }
    if (hang_mode && (child_err.find("[hang-stack] test hang:") == std::string::npos ||
                      child_err.find("[hang-stack] thread ") == std::string::npos ||
                      child_err.find("test_minidump_fault.exe+0x") == std::string::npos)) {
        std::fprintf(stderr, "FAIL: no thread stacks printed:\n%s\n", child_err.c_str());
        return 1;
    }
    DeleteFileA(err_path.c_str());
    std::ifstream f(dump, std::ios::binary | std::ios::ate);
    const long long size = f ? static_cast<long long>(f.tellg()) : -1;
    if (code != 77 || size <= 1024) {
        std::fprintf(stderr, "FAIL: child exit %lu, dump size %lld\n", code, size);
        return 1;
    }
    const char* script = std::getenv("PG6_MINIDUMP_STACK_PY");
    if (script != nullptr) {
        const std::string py = std::string("python -I \"") + script + "\" \"" + dump + "\"" + (hang_mode ? " --all-threads" : "") + " > \"" + dump + ".txt\"";
        if (std::system(py.c_str()) != 0) { std::fprintf(stderr, "FAIL: minidump_stack.py\n"); return 1; }
        std::ifstream t(dump + ".txt");
        std::string text((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        const char* want = guest_mode ? "exception 0xe0475046" : hang_mode ? "thread " : "exception 0xc0000005";
        if (text.find(want) == std::string::npos || text.find("test_minidump_fault.exe+0x") == std::string::npos ||
            (guest_mode && (text.find("guest fault: kind=unknown_target") == std::string::npos ||
                            text.find("-> 0x3000") == std::string::npos))) {
            std::fprintf(stderr, "FAIL: stack output:\n%s\n", text.c_str());
            return 1;
        }
        DeleteFileA((dump + ".txt").c_str());
    }
    DeleteFileA(dump.c_str());
    std::puts("minidump fault test OK");
    return 0;
}
#else
int main() { return 0; }
#endif
