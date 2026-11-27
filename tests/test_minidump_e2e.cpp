// Windows only: end-to-end check of the three crash-evidence paths of playgta6_boot without a
// game, using the same harness code (guest/minidump_writer.h, guest/minidump.h,
// guest/exit_backstop.h). A parent starts this exe once per scenario as a child and requires,
// within a time bound, a non-empty minidump that tools/minidump_stack.py can parse plus the
// expected report/stderr fields:
//   --fault     host access violation: dump from the vectored handler, report carrying
//               fault_record.minidump, exit 9
//   --watchdog  hang caught by the watchdog: all-thread stacks + dump, report, exit 10
//   --multi     8 threads fault at once (null deref, abort(), terminate()): the first owns the
//               report + all-thread dump, the others park; exit 9, never 0xC0000409 / no report
//   --stuck     the report path after the watchdog stop hangs: the exit backstop (30 s in the
//               harness, 2 s here) prints stacks, writes its own dump and ends the process with
//               exit 10; no report exists
#ifdef _WIN32
#include "guest/exit_backstop.h"
#include "guest/fault_gate.h"
#include "guest/minidump.h"
#include "guest/minidump_writer.h"
#include "guest/terminate_hook.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <cstring>
#include <fstream>
#include <string>

namespace mdp = playgta6::guest::minidump;
namespace backstop = playgta6::guest::backstop;
using playgta6::guest::minidump::Writer;

static constexpr int kExitHostFault = 9, kExitWatchdog = 10;
static constexpr DWORD kBackstopMs = 2000;
static std::string g_report;

static std::string dump_path(const char* kind, DWORD pid) { return mdp::select_path(g_report, "host", kind, pid); }

static void write_report(const char* kind, const std::string& dump) {
    const std::string body = std::string("{\"fault_record\": {\"kind\": \"") + kind + "\"}}";
    std::ofstream(g_report, std::ios::binary) << mdp::annotate_report(body, dump);
}

static LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    backstop::arm(kExitHostFault);
    const std::string path = dump_path("host_access_violation", GetCurrentProcessId());
    if (Writer::dump(path.c_str(), info)) write_report("host_access_violation", path);
    TerminateProcess(GetCurrentProcess(), kExitHostFault);
    return EXCEPTION_CONTINUE_SEARCH;
}

#pragma optimize("", off)
static void deliberate_null_deref(volatile int* p) { *p = 1; }
#pragma optimize("", on)

static DWORD WINAPI watchdog_thread(void* stuck) {
    Sleep(1000);
    backstop::arm(kExitWatchdog);
    backstop::raw_stderr("[harness] watchdog: elapsed, the run did not end; thread stacks follow\n");
    const std::string path = dump_path("host_watchdog_timeout", GetCurrentProcessId());
    const bool ok = Writer::dump_threads(path.c_str(), "watchdog", 20000);
    if (stuck != nullptr) Sleep(INFINITE);  // the report path hangs: only the backstop can end this
    if (ok) write_report("host_watchdog_timeout", path);
    TerminateProcess(GetCurrentProcess(), kExitWatchdog);
    return 0;
}

// --multi: the harness's claim-or-park (guest/fault_gate.h) in front of one report writer.
static playgta6::guest::FaultGate g_gate;
static std::atomic<bool> g_go{false};
[[noreturn]] static void multi_fault() {
    backstop::arm(kExitHostFault);
    g_gate.claim_or_park();  // a loser never returns: the owner ends the process
    const std::string path = dump_path("host_multi_fault", GetCurrentProcessId());
    if (Writer::dump_threads(path.c_str(), "multi fault", 20000)) write_report("host_multi_fault", path);
    TerminateProcess(GetCurrentProcess(), kExitHostFault);
    for (;;) Sleep(1000);
}
static LONG CALLBACK multi_handler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    multi_fault();
}
static void multi_abort(int) {
    std::signal(SIGABRT, &multi_abort);  // the CRT resets it before the call
    multi_fault();
}
[[noreturn]] static void multi_terminate() { multi_fault(); }
static void throw_in_noexcept() noexcept { std::terminate(); }
static DWORD WINAPI multi_thread(void* arg) {
    const int i = static_cast<int>(reinterpret_cast<intptr_t>(arg));
    while (!g_go.load()) {}
    if (i % 3 == 0) deliberate_null_deref(nullptr);
    if (i % 3 == 1) playgta6::guest::abort_once();
    throw_in_noexcept();
    return 0;
}

static int run_child(const std::string& mode) {
    if (!Writer::init()) return 3;
    backstop::start(dump_path("report_stuck", GetCurrentProcessId()).c_str(), kExitWatchdog, kBackstopMs);
    if (mode == "--fault") {
        AddVectoredExceptionHandler(1, &handler);
        deliberate_null_deref(nullptr);
        return 4;
    }
    if (mode == "--multi") {
        AddVectoredExceptionHandler(1, &multi_handler);
        playgta6::guest::terminate_hook::install(&multi_terminate);
        std::signal(SIGABRT, &multi_abort);
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
        for (int i = 0; i < 8; ++i) CreateThread(nullptr, 0, &multi_thread, reinterpret_cast<void*>(static_cast<intptr_t>(i)), 0, nullptr);
        Sleep(200);
        g_go = true;
        Sleep(INFINITE);
        return 4;
    }
    CreateThread(nullptr, 0, &watchdog_thread, mode == "--stuck" ? &g_report : nullptr, 0, nullptr);
    Sleep(INFINITE);  // the "hung" run
    return 4;
}

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static long long size_of(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    return f ? static_cast<long long>(f.tellg()) : -1;
}

static bool has(const std::string& s, const char* t) { return s.find(t) != std::string::npos; }

static bool fail(const std::string& what, const std::string& detail) {
    std::fprintf(stderr, "FAIL: %s\n%s\n", what.c_str(), detail.c_str());
    return false;
}

struct Scenario {
    const char* mode;
    const char* kind;       // dump file kind (select_path)
    int exit_code;
    bool report;            // a report with fault_record.minidump is expected
    DWORD min_ms, max_ms;   // time bound
    const char* want_stack; // text the parsed dump must contain
    const char* stderr_a;   // text the child's stderr must contain
    const char* stderr_b;
};

static bool run_scenario(const char* exe, const char* script, const std::string& dir, const Scenario& sc) {
    const std::string mode = sc.mode;
    const std::string report = dir + "pg6_e2e" + mode + ".json";
    const std::string err_path = report + ".stderr";
    DeleteFileA(report.c_str());
    std::string cmd = std::string("\"") + exe + "\" --child " + mode + " \"" + report + "\"";
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
    const DWORD t0 = GetTickCount();
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi))
        return fail(mode + ": CreateProcess", cmd);
    CloseHandle(err);
    const bool exited = WaitForSingleObject(pi.hProcess, sc.max_ms) == WAIT_OBJECT_0;
    const DWORD elapsed = GetTickCount() - t0;
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    if (!exited) TerminateProcess(pi.hProcess, 99);
    const DWORD pid = pi.dwProcessId;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    const std::string child_err = slurp(err_path);
    if (!exited) return fail(mode + ": still running after " + std::to_string(sc.max_ms) + " ms", child_err);
    if (code != static_cast<DWORD>(sc.exit_code) || elapsed < sc.min_ms)
        return fail(mode + ": exit " + std::to_string(code) + " after " + std::to_string(elapsed) + " ms", child_err);
    if ((sc.stderr_a != nullptr && !has(child_err, sc.stderr_a)) || (sc.stderr_b != nullptr && !has(child_err, sc.stderr_b)))
        return fail(mode + ": expected stderr text missing", child_err);

    // The dump the scenario must have produced, found via the report when one exists.
    const std::string dump = mdp::select_path(report, "host", sc.kind, pid);
    std::string parse_arg = dump;
    if (sc.report) {
        const std::string body = slurp(report);
        if (!has(body, "\"fault_record\": {\"minidump\": \"") || !has(body, std::string(sc.kind).c_str()))
            return fail(mode + ": report lacks fault_record.minidump", body);
        parse_arg = report;
    }
    const long long size = size_of(dump);
    if (size <= 1024) return fail(mode + ": dump missing or empty: " + dump, std::to_string(size));
    if (script != nullptr) {
        const std::string out = dump + ".txt";
        const std::string py = std::string("python -I \"") + script + "\" \"" + parse_arg + "\"" + (mode == "--fault" ? "" : " --all-threads") + " > \"" + out + "\"";
        if (std::system(py.c_str()) != 0) return fail(mode + ": minidump_stack.py failed", slurp(out));
        const std::string text = slurp(out);
        if (!has(text, sc.want_stack) || !has(text, "test_minidump_e2e.exe+0x"))
            return fail(mode + ": parsed dump lacks '" + sc.want_stack + "'", text);
        DeleteFileA(out.c_str());
    }
    DeleteFileA(dump.c_str());
    DeleteFileA(report.c_str());
    DeleteFileA(err_path.c_str());
    std::printf("  %s OK (%lu ms, dump %lld bytes)\n", sc.mode, static_cast<unsigned long>(elapsed), size);
    return true;
}

int main(int argc, char** argv) {
    if (const int helper = Writer::run_helper_if_requested(argc, argv); helper >= 0) return helper;
    if (argc >= 4 && std::strcmp(argv[1], "--child") == 0) {
        g_report = argv[3];
        return run_child(argv[2]);
    }
    char exe[MAX_PATH];
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    char tmp[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    const char* script = std::getenv("PG6_MINIDUMP_STACK_PY");
    const Scenario scenarios[] = {
        {"--fault", "host_access_violation", kExitHostFault, true, 0, 30000, "exception 0xc0000005", nullptr, nullptr},
        {"--watchdog", "host_watchdog_timeout", kExitWatchdog, true, 1000, 30000, "thread ", "[harness] watchdog:",
         "[hang-stack] thread "},
        {"--multi", "host_multi_fault", kExitHostFault, true, 0, 30000, "thread ", nullptr, nullptr},
        {"--stuck", "report_stuck", kExitWatchdog, false, 1000 + kBackstopMs, 30000, "thread ", "[harness] exit backstop:",
         "[hang-stack] thread "},
    };
    bool ok = true;
    for (const Scenario& sc : scenarios) ok = run_scenario(exe, script, tmp, sc) && ok;
    if (!ok) return 1;
    std::puts("minidump e2e test OK");
    return 0;
}
#else
int main() { return 0; }
#endif
