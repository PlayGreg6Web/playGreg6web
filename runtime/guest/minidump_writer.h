#pragma once
// Windows host-fault / hang minidump writer.
//
// MiniDumpWriteDump runs in a helper process, never in the process being dumped. In-process it
// suspends every other thread and then allocates and walks the loader data itself; a thread
// suspended while it holds the process heap lock (any thread inside HeapAlloc, the audio device and
// WASAPI threads do that constantly) froze the whole process: the dump stayed empty and even the
// watchdog thread never ran again (#663: --audio-out boots that hung until the runner's timeout).
//
// The helper is the same executable started up front by init() with
// `--pg6-minidump-helper <mapping> <request> <done> <parent>` (inherited handles), so a fault only
// fills a shared request block, signals an event and waits (bounded): no heap, no loader lock, no
// thread creation in the faulting process. The helper also prints every thread's stack to the
// inherited stderr for hangs (dump_threads). Executables that use the writer call
// run_helper_if_requested() first thing in main().
//
// Flags are small: thread info + every thread's stack + module list, never indirectly referenced
// memory (the guest's huge reserved address space made the writer crash and produce empty dumps)
// and never full memory.
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
#include <psapi.h>
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace playgta6::guest::minidump {

// User stream carrying the guest fault_record JSON (UTF-8, no terminator). Read by
// tools/minidump_stack.py.
constexpr DWORD kFaultRecordStream = 0x50473601;
// Exception code of the synthetic exception stream written for guest faults.
constexpr DWORD kGuestFaultExceptionCode = 0xE0475046;

class Writer {
public:
    // Starts the helper process. Idempotent; false when unavailable (dumps are then skipped,
    // never crash).
    static bool init() {
        State& s = state();
        if (s.ready.load()) return true;
        if (s.failed.load()) return false;
        if (!start_helper(s)) {
            s.failed.store(true);
            return false;
        }
        s.ready.store(true);
        return true;
    }

    // Called from the faulting thread (handler context, or normal code with info == nullptr).
    // Blocks until the helper finished or timeout_ms elapsed. Returns true when a dump was
    // written. Never throws; one fault dump per process (dump_threads is separate).
    static bool dump(const char* path, EXCEPTION_POINTERS* info, DWORD timeout_ms = 20000,
                     const void* user = nullptr, size_t user_size = 0) {
        State& s = state();
        if (!s.ready.load() || path == nullptr || s.dumped.exchange(true)) return false;
        return request(s, kDoDump, path, nullptr, info, user, user_size, timeout_ms);
    }

    // Guest fault: the calling (faulting) thread's host context captured with RtlCaptureContext
    // becomes a synthetic exception stream (code kGuestFaultExceptionCode), so the host stack,
    // including lifted sub_ frames, can be walked; `json` is stored as the fault_record user stream.
    static bool dump_guest_fault(const char* path, CONTEXT* ctx, const std::string& json, DWORD timeout_ms = 20000) {
        EXCEPTION_RECORD rec{};
        rec.ExceptionCode = kGuestFaultExceptionCode;
        rec.ExceptionAddress = reinterpret_cast<PVOID>(ctx->Rip);
        EXCEPTION_POINTERS ep{&rec, ctx};
        return dump(path, &ep, timeout_ms, json.data(), json.size());
    }

    // Hang diagnostics (watchdog, stuck report path): the helper prints every thread's stack
    // ("[hang-stack] ..." lines, module+offset, to the inherited stderr) and, when `path` is
    // given, writes a dump of all threads without an exception stream. Safe while other threads
    // hold any lock. Returns true when the helper finished in time (and the dump, if asked for,
    // is non-empty).
    static bool dump_threads(const char* path, const char* reason, DWORD timeout_ms = 20000) {
        State& s = state();
        if (!s.ready.load()) return false;
        return request(s, kDoStacks | (path != nullptr ? kDoDump : 0u), path, reason, nullptr, nullptr, 0,
                       timeout_ms);
    }

    // Helper process entry: returns the helper's exit code when argv asks for the helper, -1
    // otherwise (the caller continues with its own main).
    static int run_helper_if_requested(int argc, char** argv) {
        if (argc < 6 || std::strcmp(argv[1], kHelperFlag) != 0) return -1;
        auto handle = [](const char* t) { return reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::strtoull(t, nullptr, 10))); };
        return helper_main(handle(argv[2]), handle(argv[3]), handle(argv[4]), handle(argv[5]));
    }

private:
    static constexpr const char* kHelperFlag = "--pg6-minidump-helper";
    static constexpr uint32_t kDoDump = 1, kDoStacks = 2;

    // Shared request block (file mapping). Written by the requester, read by the helper.
    struct Shared {
        uint32_t what;
        uint32_t fault_thread;
        uint64_t info;       // EXCEPTION_POINTERS* in the requester (0 = none)
        uint64_t user;       // user stream bytes in the requester
        uint64_t user_size;
        uint32_t ok;         // helper: 1 = done (dump non-empty when asked for)
        uint32_t reserved;
        char reason[128];
        char path[MAX_PATH * 2];
    };

    struct State {
        std::atomic<bool> ready{false};
        std::atomic<bool> failed{false};
        std::atomic<bool> dumped{false};
        std::atomic<bool> busy{false};
        HANDLE mapping = nullptr, request = nullptr, done = nullptr, helper = nullptr;
        Shared* shared = nullptr;
    };
    static State& state() {
        static State s;
        return s;
    }

    static void copy_text(char* dst, size_t cap, const char* src) {
        size_t i = 0;
        for (; src != nullptr && src[i] != 0 && i + 1 < cap; ++i) dst[i] = src[i];
        dst[i] = 0;
    }

    // No heap, no locks: fill the block, signal, wait.
    static bool request(State& s, uint32_t what, const char* path, const char* reason, EXCEPTION_POINTERS* info,
                        const void* user, size_t user_size, DWORD timeout_ms) {
        if (s.busy.exchange(true)) return false;
        Shared& r = *s.shared;
        r.what = what;
        r.fault_thread = GetCurrentThreadId();
        r.info = reinterpret_cast<uint64_t>(info);
        r.user = reinterpret_cast<uint64_t>(user);
        r.user_size = user != nullptr ? user_size : 0;
        r.ok = 0;
        copy_text(r.reason, sizeof(r.reason), reason);
        copy_text(r.path, sizeof(r.path), path);
        ResetEvent(s.done);
        SetEvent(s.request);
        const bool finished = WaitForSingleObject(s.done, timeout_ms) == WAIT_OBJECT_0;
        // A helper that did not finish still owns the block: leave busy set.
        if (!finished) return false;
        const bool ok = r.ok != 0;
        s.busy.store(false);
        return ok;
    }

    static bool start_helper(State& s) {
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        s.mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, &sa, PAGE_READWRITE, 0, sizeof(Shared), nullptr);
        if (s.mapping == nullptr) return false;
        s.shared = static_cast<Shared*>(MapViewOfFile(s.mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Shared)));
        s.request = CreateEventA(&sa, FALSE, FALSE, nullptr);
        s.done = CreateEventA(&sa, TRUE, FALSE, nullptr);
        HANDLE parent = nullptr;
        if (s.shared == nullptr || s.request == nullptr || s.done == nullptr ||
            !DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(), GetCurrentProcess(), &parent,
                             PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | PROCESS_DUP_HANDLE | SYNCHRONIZE, TRUE, 0))
            return false;
        wchar_t exe[MAX_PATH * 2];
        const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH * 2);
        if (n == 0 || n >= MAX_PATH * 2) {
            CloseHandle(parent);
            return false;
        }
        wchar_t cmd[MAX_PATH * 2 + 160];
        _snwprintf_s(cmd, _TRUNCATE, L"\"%ls\" --pg6-minidump-helper %llu %llu %llu %llu", exe,
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(s.mapping)),
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(s.request)),
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(s.done)),
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(parent)));
        // Inherit exactly the handles the helper needs (plus stderr for the stack lines), not
        // every inheritable handle of the harness.
        HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
        HANDLE inherit[5] = {s.mapping, s.request, s.done, parent, nullptr};
        DWORD count = 4;
        HANDLE err_dup = nullptr;
        if (err != nullptr && err != INVALID_HANDLE_VALUE &&
            DuplicateHandle(GetCurrentProcess(), err, GetCurrentProcess(), &err_dup, 0, TRUE, DUPLICATE_SAME_ACCESS))
            inherit[count++] = err_dup;
        SIZE_T attr_size = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
        std::vector<unsigned char> attr_buf(attr_size);
        auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
        bool ok = InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size) &&
                  UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, count * sizeof(HANDLE),
                                            nullptr, nullptr);
        STARTUPINFOEXW si{};
        si.StartupInfo.cb = sizeof(si);
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdError = err_dup;
        si.StartupInfo.hStdOutput = err_dup;
        si.lpAttributeList = attrs;
        PROCESS_INFORMATION pi{};
        if (ok)
            ok = CreateProcessW(nullptr, cmd, nullptr, nullptr, TRUE,
                                EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr,
                                nullptr, &si.StartupInfo, &pi) != FALSE;
        DeleteProcThreadAttributeList(attrs);
        CloseHandle(parent);
        if (err_dup != nullptr) CloseHandle(err_dup);
        if (!ok) return false;
        CloseHandle(pi.hThread);
        s.helper = pi.hProcess;
        return true;
    }

    // ---- helper process ----

    struct Module {
        uint64_t base, size;
        char name[64];
    };

    static std::vector<Module> list_modules(HANDLE process) {
        std::vector<Module> out;
        HMODULE mods[1024];
        DWORD needed = 0;
        if (!K32EnumProcessModulesEx(process, mods, sizeof(mods), &needed, LIST_MODULES_ALL)) return out;
        const DWORD n = std::min<DWORD>(needed / sizeof(HMODULE), 1024);
        for (DWORD i = 0; i < n; ++i) {
            MODULEINFO mi{};
            Module m{};
            if (!K32GetModuleInformation(process, mods[i], &mi, sizeof(mi))) continue;
            m.base = reinterpret_cast<uint64_t>(mi.lpBaseOfDll);
            m.size = mi.SizeOfImage;
            if (!K32GetModuleBaseNameA(process, mods[i], m.name, sizeof(m.name))) copy_text(m.name, sizeof(m.name), "?");
            out.push_back(m);
        }
        return out;
    }

    static const Module* module_for(const std::vector<Module>& mods, uint64_t a) {
        for (const Module& m : mods)
            if (a >= m.base && a < m.base + m.size) return &m;
        return nullptr;
    }

    // One line per thread: name, rip and the first return-address-looking stack words as
    // module+offset (tools/minidump_stack.py --all-threads on the dump symbolizes the same).
    static void print_stacks(HANDLE process, DWORD pid, const char* reason, DWORD requester) {
        const std::vector<Module> mods = list_modules(process);
        using DescFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
        auto describe = reinterpret_cast<DescFn>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription"));
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        std::vector<DWORD> tids;
        if (snap != INVALID_HANDLE_VALUE) {
            THREADENTRY32 te{};
            te.dwSize = sizeof(te);
            for (BOOL more = Thread32First(snap, &te); more; more = Thread32Next(snap, &te))
                if (te.th32OwnerProcessID == pid) tids.push_back(te.th32ThreadID);
            CloseHandle(snap);
        }
        std::fprintf(stderr, "[hang-stack] %s: %zu thread(s) of pid %lu (requested by thread %lu)\n",
                     reason[0] != 0 ? reason : "thread stacks", tids.size(), pid, requester);
        std::vector<uint64_t> words(4096);
        for (DWORD tid : tids) {
            HANDLE t = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid);
            if (t == nullptr) continue;
            char name[96] = "";
            PWSTR wname = nullptr;
            if (describe != nullptr && SUCCEEDED(describe(t, &wname)) && wname != nullptr) {
                WideCharToMultiByte(CP_UTF8, 0, wname, -1, name, sizeof(name), nullptr, nullptr);
                LocalFree(wname);
            }
            const bool suspended = SuspendThread(t) != static_cast<DWORD>(-1);
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            const bool have = suspended && GetThreadContext(t, &ctx);
            SIZE_T got = 0;
            if (have) ReadProcessMemory(process, reinterpret_cast<LPCVOID>(ctx.Rsp), words.data(), words.size() * 8, &got);
            if (have && got == 0) {
                // The stack end is closer than 32 KiB: read up to the page boundary chain.
                for (SIZE_T bytes = 16384; bytes >= 512 && got == 0; bytes /= 2)
                    ReadProcessMemory(process, reinterpret_cast<LPCVOID>(ctx.Rsp), words.data(), bytes, &got);
            }
            if (suspended) ResumeThread(t);
            CloseHandle(t);
            std::string line = "[hang-stack] thread " + std::to_string(tid);
            if (name[0] != 0) line += std::string(" \"") + name + "\"";
            if (tid == requester) line += " (requester)";
            char buf[96];
            if (!have) {
                line += ": no context";
            } else {
                const Module* m = module_for(mods, ctx.Rip);
                if (m != nullptr)
                    std::snprintf(buf, sizeof(buf), " rip %s+0x%llx:", m->name, static_cast<unsigned long long>(ctx.Rip - m->base));
                else
                    std::snprintf(buf, sizeof(buf), " rip 0x%llx:", static_cast<unsigned long long>(ctx.Rip));
                line += buf;
                int shown = 0;
                for (size_t i = 0; i < got / 8 && shown < 16; ++i) {
                    const Module* w = words[i] >= 0x10000 ? module_for(mods, words[i]) : nullptr;
                    if (w == nullptr) continue;
                    std::snprintf(buf, sizeof(buf), " %s+0x%llx", w->name, static_cast<unsigned long long>(words[i] - w->base));
                    line += buf;
                    ++shown;
                }
            }
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        std::fflush(stderr);
    }

    static bool write_dump(HANDLE process, DWORD pid, const Shared& r) {
        HMODULE dbg = LoadLibraryA("dbghelp.dll");
        using WriteFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, DWORD, void*, void*, void*);
        auto fn = dbg != nullptr ? reinterpret_cast<WriteFn>(GetProcAddress(dbg, "MiniDumpWriteDump")) : nullptr;
        if (fn == nullptr) return false;
        HANDLE file = CreateFileA(r.path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) return false;
        // The real struct: MINIDUMP_EXCEPTION_INFORMATION is pack(4) on every arch, so a hand-rolled
        // 8-aligned copy shifts the pointer field and MiniDumpWriteDump fails with ERROR_NOACCESS.
        // ClientPointers: the EXCEPTION_POINTERS (and what they point to) live in the dumped process.
        MINIDUMP_EXCEPTION_INFORMATION ex{};
        ex.ThreadId = r.fault_thread;
        ex.ExceptionPointers = reinterpret_cast<PEXCEPTION_POINTERS>(r.info);
        ex.ClientPointers = TRUE;
        constexpr DWORD kWithThreadInfo = 0x00001000;
        constexpr DWORD kWithUnloadedModules = 0x00000020;
        std::vector<unsigned char> user;
        MINIDUMP_USER_STREAM us{};
        MINIDUMP_USER_STREAM_INFORMATION usi{};
        if (r.user != 0 && r.user_size > 0 && r.user_size < (64u << 20)) {
            user.resize(static_cast<size_t>(r.user_size));
            SIZE_T got = 0;
            if (ReadProcessMemory(process, reinterpret_cast<LPCVOID>(r.user), user.data(), user.size(), &got) &&
                got == user.size()) {
                us.Type = kFaultRecordStream;
                us.BufferSize = static_cast<ULONG>(user.size());
                us.Buffer = user.data();
                usi.UserStreamCount = 1;
                usi.UserStreamArray = &us;
            }
        }
        const bool ok = fn(process, pid, file, kWithThreadInfo | kWithUnloadedModules, r.info != 0 ? &ex : nullptr,
                           usi.UserStreamCount != 0 ? &usi : nullptr, nullptr) != FALSE;
        if (!ok) std::fprintf(stderr, "[minidump] MiniDumpWriteDump failed: 0x%08lx\n", GetLastError());
        LARGE_INTEGER size{};
        const bool nonempty = GetFileSizeEx(file, &size) && size.QuadPart > 0;
        CloseHandle(file);
        return ok && nonempty;
    }

    static int helper_main(HANDLE mapping, HANDLE request, HANDLE done, HANDLE parent) {
        auto* r = static_cast<Shared*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Shared)));
        if (r == nullptr) return 2;
        const DWORD pid = GetProcessId(parent);
        HANDLE waits[2] = {parent, request};
        for (;;) {
            const DWORD w = WaitForMultipleObjects(2, waits, FALSE, INFINITE);
            if (w != WAIT_OBJECT_0 + 1) return 0;  // the parent exited (or the wait failed)
            bool ok = true;
            if (r->what & kDoStacks) print_stacks(parent, pid, r->reason, r->fault_thread);
            if (r->what & kDoDump) ok = write_dump(parent, pid, *r);
            r->ok = ok ? 1 : 0;
            SetEvent(done);
        }
    }
};

}  // namespace playgta6::guest::minidump
#endif
