// Windows only: a child process faults on a host null dereference; the dump writer (dedicated
// dumper thread) must produce a non-empty dump and tools/minidump_stack.py must print the fault.
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

int main(int argc, char** argv) {
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
    std::string cmd = std::string("\"") + exe + "\" --child \"" + dump + "\"";
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return 1;
    WaitForSingleObject(pi.hProcess, 60000);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    std::ifstream f(dump, std::ios::binary | std::ios::ate);
    const long long size = f ? static_cast<long long>(f.tellg()) : -1;
    if (code != 77 || size <= 1024) {
        std::fprintf(stderr, "FAIL: child exit %lu, dump size %lld\n", code, size);
        return 1;
    }
    const char* script = std::getenv("PG6_MINIDUMP_STACK_PY");
    if (script != nullptr) {
        const std::string py = std::string("python -I \"") + script + "\" \"" + dump + "\" > \"" + dump + ".txt\"";
        if (std::system(py.c_str()) != 0) { std::fprintf(stderr, "FAIL: minidump_stack.py\n"); return 1; }
        std::ifstream t(dump + ".txt");
        std::string text((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        if (text.find("exception 0xc0000005") == std::string::npos ||
            text.find("test_minidump_fault.exe+0x") == std::string::npos) {
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
