// Native first-boot harness: runs a recompiled guest executable in an ordinary, debuggable host
// process through the boot sequence of guest/boot.h and writes its report.
//
//   playgta6_boot --pgim <guest.pgim> [options]
//
// tools/build_native_boot.py compiles the lifted shards of a `playgta6-recomp build --target
// native` output for the host, and links them, the generated import bridge and the registration
// table with the host runtime library and this file (defining PG6_BOOT_LINKED_GUEST, PG6_BOOT_MODULE
// and putting the build's stubs/ directory on the include path). Built without
// PG6_BOOT_LINKED_GUEST (the CMake target `playgta6_boot`) it has no lifted code and no imports:
// useful to check an image loads and see the first fault at the entry point.
//
// Options:
//   --pgim <file>        guest image (required)
//   --name <name>        main module name (default game.exe)
//   --game-dir <dir>     serve the guest's file reads (HTTPFS) from this host directory, read-only
//   --cwd <dir>          guest current directory string (default: the --game-dir path)
//   --cmdline <text>     guest command line (default: the module name)
//   --survey             unresolved imports return 0 and are listed (diagnostic, unsafe)
//   --survey-return <n>  value survey mode returns in rax instead of 0 (e.g. 1 = BOOL TRUE)
//   --max-calls <n>      stop after n dispatched calls (0 = unbounded)
//   --max-ms <n>         stop after n ms, checked at dispatches (0 = unbounded)
//   --watchdog-ms <n>    hard limit: report and terminate even without dispatches (default 0)
//   --stack <bytes>      guest stack size (default: the image's SizeOfStackReserve)
//   --host-stack-mb <n>  host stack of the boot thread (default 4096: lifted guest calls nest
//                        host calls, so deep guest recursion needs a deep host stack)
//   --report <file>      JSON report path (default boot-report.json)
//   --summary <file>     also write the text summary there (it always goes to stderr)
//   --import-trace <file> write every import call (sequence, thread, DLL!symbol, rcx rdx r8 r9
//                        [rsp+0x28]) there, up to --import-trace-limit calls (default 200000)
//   --console-mode <n>   console-mode registry value the title reads to pick its console type
//                        (default 0x60 = Xbox Series X; 0 = leave it unset)
//   --gpu-capture <file> write the GPU command stream of the first frames there as a .pg6trace
//                        plus <file>.json (gpu/frame_capture.h); data payloads are omitted
//   --gpu-capture-frames <n>  frames to capture (default 10)
//   --gpu-capture-payloads    keep WriteBuffer / WriteTexture / shader payloads in the capture
//   --no-tls-callbacks / --no-entry / --quiet
//
// Exit code: the BootStatus (0 = the guest exited; its own exit code is in the report),
// 9 = host fault inside guest code (access violation, illegal instruction, stack overflow, ...),
// 10 = watchdog, 64 = usage error. Host faults and the watchdog write the report like any other
// outcome before the process terminates: execution cannot continue after a host fault.
//
// Address space: lifted code addresses guest memory through __playgta6_guest_base (identity in
// the current lifter). The harness touches the process arena first so it can claim its guest
// address space before anything else maps there, and reports whether the placement is identity.
// The harness executable itself must not sit inside the guest range (the game wants
// 0x1'4000'0000, MSVC's default EXE base): build_native_boot.py links it at a high base.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu/frame_capture.h"
#include "guest/boot.h"
#include "os/httpfs_mailbox.h"
#include "os/image_loader.h"
#include "os/os_shim.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#endif

#ifdef PG6_BOOT_LINKED_GUEST
#include "pg6_import_thunks.h"
#ifndef PG6_BOOT_MODULE
#define PG6_BOOT_MODULE main
#endif
#define PG6_CAT2(a, b) a##b
#define PG6_CAT(a, b) PG6_CAT2(a, b)
extern "C" const Pg6LiftedFunctionEntry PG6_CAT(__playgta6_lifted_functions_, PG6_BOOT_MODULE)[];
extern "C" const int64_t PG6_CAT(__playgta6_lifted_function_count_, PG6_BOOT_MODULE);
static_assert(sizeof(Pg6ImportThunk) == sizeof(Pg6BootImport), "import table layout");
static_assert(offsetof(Pg6ImportThunk, thunk) == offsetof(Pg6BootImport, thunk), "import table layout");
static_assert(offsetof(Pg6ImportThunk, shim) == offsetof(Pg6BootImport, shim), "import table layout");
#endif

#ifdef PG6_BOOT_PROGRAM
#include "guest/program_loader.h"
extern "C" const Pg6Program pg6_program;  // pg6_program.cpp of the program build
#endif

namespace {

using namespace playgta6;

constexpr int kExitUsage = 64;
constexpr int kExitHostFault = 9;
constexpr int kExitWatchdog = 10;

struct Options {
    std::string pgim;
    std::string program_dir;  // program builds: <out> of build-program
    std::string name = "game.exe";
    std::string game_dir;
    std::string cwd;
    std::string cmdline;
    bool survey = false;
    uint64_t max_calls = 0;
    uint64_t survey_return = 0;
    uint64_t max_ms = 0;
    uint64_t watchdog_ms = 0;
    uint64_t stack = 0;
    uint64_t host_stack_mb = 4096;
    std::string report = "boot-report.json";
    std::string summary;
    std::string import_trace;  // --import-trace: names + arguments of every import call
    uint64_t import_trace_limit = 200000;
    uint64_t console_mode = playgta6::guest::kConsoleModeSeriesX;
    std::string gpu_capture;          // --gpu-capture: .pg6trace path
    uint64_t gpu_capture_frames = 10;
    bool gpu_capture_payloads = false;
    bool tls_callbacks = true;
    bool entry = true;
    bool quiet = false;
};

Options g_options;
std::atomic<bool> g_finishing{false};
int g_result = 0;

void usage() {
    std::fprintf(stderr,
                 "usage: playgta6_boot --pgim <guest.pgim> [--name <exe>] [--game-dir <dir>] [--cwd <dir>]\n"
                 "                     [--cmdline <text>] [--survey] [--max-calls N] [--max-ms N]\n"
                 "                     [--watchdog-ms N] [--stack BYTES] [--host-stack-mb N]\n"
                 "                     [--report <file>] [--summary <file>] [--no-tls-callbacks]\n"
                 "                     [--no-entry] [--quiet]\n"
                 "                     [--gpu-capture <file> [--gpu-capture-frames N] [--gpu-capture-payloads]]\n");
}

bool parse_u64(const char* text, uint64_t& out) {
    char* end = nullptr;
    out = std::strtoull(text, &end, 0);
    return end != text && *end == '\0';
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char*& out) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                return false;
            }
            out = argv[++i];
            return true;
        };
        const char* v = nullptr;
        if (a == "--survey") {
            o.survey = true;
        } else if (a == "--no-tls-callbacks") {
            o.tls_callbacks = false;
        } else if (a == "--no-entry") {
            o.entry = false;
        } else if (a == "--quiet") {
            o.quiet = true;
        } else if (a == "--gpu-capture-payloads") {
            o.gpu_capture_payloads = true;
        } else if (a == "-h" || a == "--help") {
            return false;
        } else if (a == "--pgim" || a == "--name" || a == "--game-dir" || a == "--cwd" ||
                   a == "--program-dir" ||
                   a == "--cmdline" || a == "--report" || a == "--summary" || a == "--import-trace" ||
                   a == "--gpu-capture") {
            if (!value(v)) {
                return false;
            }
            std::string& target = a == "--pgim"       ? o.pgim
                                  : a == "--gpu-capture" ? o.gpu_capture
                                  : a == "--program-dir" ? o.program_dir
                                  : a == "--name"     ? o.name
                                  : a == "--game-dir" ? o.game_dir
                                  : a == "--cwd"      ? o.cwd
                                  : a == "--cmdline"  ? o.cmdline
                                  : a == "--report"   ? o.report
                                  : a == "--import-trace" ? o.import_trace
                                                      : o.summary;
            target = v;
        } else if (a == "--max-calls" || a == "--max-ms" || a == "--watchdog-ms" || a == "--stack" ||
                   a == "--host-stack-mb" || a == "--survey-return" || a == "--import-trace-limit" ||
                   a == "--console-mode" || a == "--gpu-capture-frames") {
            if (!value(v)) {
                return false;
            }
            uint64_t& target = a == "--max-calls"     ? o.max_calls
                               : a == "--gpu-capture-frames" ? o.gpu_capture_frames
                               : a == "--console-mode" ? o.console_mode
                               : a == "--max-ms"      ? o.max_ms
                               : a == "--watchdog-ms" ? o.watchdog_ms
                               : a == "--import-trace-limit" ? o.import_trace_limit
                               : a == "--survey-return" ? o.survey_return
                               : a == "--stack"       ? o.stack
                                                      : o.host_stack_mb;
            if (!parse_u64(v, target)) {
                std::fprintf(stderr, "%s needs a number\n", a.c_str());
                return false;
            }
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return false;
        }
    }
#ifdef PG6_BOOT_PROGRAM
    if (o.program_dir.empty()) {
        std::fprintf(stderr, "--program-dir is required (this harness boots a program build)\n");
        return false;
    }
#else
    if (o.pgim.empty()) {
        std::fprintf(stderr, "--pgim is required\n");
        return false;
    }
#endif
    if (o.cmdline.empty()) {
        o.cmdline = o.name;
    }
    if (o.cwd.empty()) {
        o.cwd = o.game_dir;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Image file reader
// ---------------------------------------------------------------------------

class FileReader final : public os::pgim::ImageReader {
public:
    bool open(const std::string& path) {
        file_ = std::fopen(path.c_str(), "rb");
        if (file_ == nullptr) {
            return false;
        }
        std::error_code ec;
        size_ = std::filesystem::file_size(path, ec);
        return !ec;
    }
    ~FileReader() override {
        if (file_ != nullptr) {
            std::fclose(file_);
        }
    }
    uint64_t size() const override { return size_; }
    bool read(uint64_t offset, void* dst, size_t len) override {
        if (offset > size_ || len > size_ - offset) {
            return false;
        }
#ifdef _WIN32
        if (_fseeki64(file_, static_cast<int64_t>(offset), SEEK_SET) != 0) {
            return false;
        }
#else
        if (fseeko(file_, static_cast<off_t>(offset), SEEK_SET) != 0) {
            return false;
        }
#endif
        return std::fread(dst, 1, len, file_) == len;
    }

private:
    std::FILE* file_ = nullptr;
    uint64_t size_ = 0;
};

// ---------------------------------------------------------------------------
// HTTPFS responder serving a host directory (read-only)
// ---------------------------------------------------------------------------

class DirectoryServer {
public:
    explicit DirectoryServer(std::filesystem::path root) : root_(std::move(root)) {}

    void handle(os::HttpfsMailbox& mb) {
        using os::HttpfsMailboxClient;
        using os::HttpfsOp;
        using os::HttpfsStatus;
        std::lock_guard<std::mutex> lock(mutex_);
        switch (static_cast<HttpfsOp>(mb.op)) {
            case HttpfsOp::Open: {
                std::string rel(reinterpret_cast<const char*>(mb.payload),
                                static_cast<size_t>(std::min<uint64_t>(mb.length, os::kHttpfsPayloadBytes)));
                std::replace(rel.begin(), rel.end(), '\\', '/');
                const std::filesystem::path path = std::filesystem::path(rel).lexically_normal();
                if (rel.empty() || path.is_absolute() || path.has_root_name() ||
                    (!path.empty() && *path.begin() == "..")) {
                    HttpfsMailboxClient::complete_response(mb, -2, HttpfsStatus::Error);
                    return;
                }
                std::FILE* f = std::fopen((root_ / path).string().c_str(), "rb");
                if (f == nullptr) {
                    if (opened_.size() < 200) {
                        os::log("[harness] file not found in --game-dir: " + rel);
                    }
                    HttpfsMailboxClient::complete_response(mb, -2, HttpfsStatus::Error);
                    return;
                }
                const int32_t handle = next_++;
                files_[handle] = f;
                opened_.push_back(rel);
                HttpfsMailboxClient::complete_response(mb, handle, HttpfsStatus::ResponseReady);
                return;
            }
            case HttpfsOp::Read: {
                auto it = files_.find(mb.handle);
                if (it == files_.end()) {
                    HttpfsMailboxClient::complete_response(mb, -9, HttpfsStatus::Error);
                    return;
                }
                const size_t cap = mb.payload_capacity == 0 ? os::kHttpfsPayloadBytes : mb.payload_capacity;
                const size_t len = static_cast<size_t>(std::min<uint64_t>(mb.length, cap));
#ifdef _WIN32
                _fseeki64(it->second, static_cast<int64_t>(mb.offset), SEEK_SET);
#else
                fseeko(it->second, static_cast<off_t>(mb.offset), SEEK_SET);
#endif
                const size_t n = std::fread(mb.payload, 1, len, it->second);
                HttpfsMailboxClient::complete_response(mb, static_cast<int64_t>(n), HttpfsStatus::ResponseReady);
                return;
            }
            case HttpfsOp::Close: {
                auto it = files_.find(mb.handle);
                if (it == files_.end()) {
                    HttpfsMailboxClient::complete_response(mb, -9, HttpfsStatus::Error);
                    return;
                }
                std::fclose(it->second);
                files_.erase(it);
                HttpfsMailboxClient::complete_response(mb, 0, HttpfsStatus::ResponseReady);
                return;
            }
            default:
                HttpfsMailboxClient::complete_response(mb, -1, HttpfsStatus::Error);
        }
    }

private:
    std::filesystem::path root_;
    std::mutex mutex_;
    std::unordered_map<int32_t, std::FILE*> files_;
    std::vector<std::string> opened_;
    int32_t next_ = 1;
};

// ---------------------------------------------------------------------------
// Report output and host faults
// ---------------------------------------------------------------------------

void write_text(const std::string& path, const std::string& text) {
    if (path.empty()) {
        return;
    }
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
    } else {
        std::fprintf(stderr, "[harness] cannot write %s\n", path.c_str());
    }
}

// The boot logs its summary itself unless --quiet; host faults happen outside its logging.
void write_outputs(bool print_summary) {
    if (!g_options.gpu_capture.empty()) {
        gpu::frame_capture_finish();
        std::fprintf(stderr, "[harness] GPU capture: %u frame(s) written to %s\n", gpu::frame_capture_frames(),
                     g_options.gpu_capture.c_str());
    }
    const std::string report = guest::last_report_json();
    const std::string summary = guest::last_summary();
    write_text(g_options.report, report);
    write_text(g_options.summary, summary);
    if (!g_options.import_trace.empty()) {
        write_text(g_options.import_trace, guest::import_trace_text());
    }
    if (print_summary) {
        std::fprintf(stderr, "%s", summary.c_str());
    }
    std::fprintf(stderr, "[harness] report written to %s\n", g_options.report.c_str());
    std::fflush(stderr);
}

[[noreturn]] void terminate_now(int code) {
    std::fflush(nullptr);
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#endif
    std::_Exit(code);
}

// Reports a host fault once and terminates; later/concurrent faults just terminate.
[[noreturn]] void host_fault(const char* kind, uint64_t address, uint64_t pc, int code) {
    if (!g_finishing.exchange(true)) {
        if (guest::report_host_fault(kind, address, pc)) {
            write_outputs(true);
        } else {
            std::fprintf(stderr, "[harness] host fault %s at 0x%" PRIx64 " (pc 0x%" PRIx64 ") outside a boot\n",
                         kind, address, pc);
        }
    }
    terminate_now(code);
}

// BootConfig::off_thread_fault: the boot already recorded the fault of a non-boot guest thread.
void off_thread_fault() {
    if (!g_finishing.exchange(true)) {
        write_outputs(true);
    }
    terminate_now(static_cast<int>(guest::BootStatus::Fault));
}

#ifdef _WIN32
// Start of the host function containing pc, from the image's unwind table (.pdata).
uint64_t host_function_start(uint64_t pc) {
    DWORD64 image_base = 0;
    if (PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(pc, &image_base, nullptr)) {
        return image_base + entry->BeginAddress;
    }
    return 0;
}

size_t capture_host_stack(void** frames, size_t max_frames) {
    return RtlCaptureStackBackTrace(0, static_cast<DWORD>(std::min<size_t>(max_frames, 62)), frames, nullptr);
}

LONG CALLBACK vectored_handler(EXCEPTION_POINTERS* info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    const char* kind = nullptr;
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:
            kind = "host_access_violation";
            break;
        case EXCEPTION_IN_PAGE_ERROR:
            kind = "host_in_page_error";
            break;
        case EXCEPTION_ILLEGAL_INSTRUCTION:
            kind = "host_illegal_instruction";
            break;
        case EXCEPTION_PRIV_INSTRUCTION:
            kind = "host_privileged_instruction";
            break;
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
        case EXCEPTION_INT_OVERFLOW:
            kind = "host_integer_divide";
            break;
        case EXCEPTION_STACK_OVERFLOW:
            kind = "host_stack_overflow";
            break;
        case EXCEPTION_BREAKPOINT:
            kind = "host_breakpoint";
            break;
        default:
            return EXCEPTION_CONTINUE_SEARCH;  // C++ exceptions etc.
    }
    uint64_t address = 0;
    if ((code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_IN_PAGE_ERROR) &&
        info->ExceptionRecord->NumberParameters >= 2) {
        address = static_cast<uint64_t>(info->ExceptionRecord->ExceptionInformation[1]);
    }
    const uint64_t pc = reinterpret_cast<uint64_t>(info->ExceptionRecord->ExceptionAddress);
    host_fault(kind, address, pc, kExitHostFault);
}

void install_host_fault_handlers() {
    AddVectoredExceptionHandler(1, &vectored_handler);
}
#else
constexpr guest::HostFunctionStartFn host_function_start = nullptr;  // heuristic mapping

size_t capture_host_stack(void** frames, size_t max_frames) {
    const int n = backtrace(frames, static_cast<int>(std::min<size_t>(max_frames, 128)));
    return n > 0 ? static_cast<size_t>(n) : 0;
}

void signal_handler(int sig, siginfo_t* info, void* raw_context) {
    const char* kind = sig == SIGSEGV   ? "host_access_violation"
                       : sig == SIGBUS  ? "host_bus_error"
                       : sig == SIGILL  ? "host_illegal_instruction"
                       : sig == SIGFPE  ? "host_integer_divide"
                       : sig == SIGTRAP ? "host_breakpoint"
                                        : "host_signal";
    uint64_t pc = 0;
#if defined(__x86_64__) && defined(__linux__)
    pc = static_cast<uint64_t>(static_cast<ucontext_t*>(raw_context)->uc_mcontext.gregs[REG_RIP]);
#else
    (void)raw_context;
#endif
    host_fault(kind, reinterpret_cast<uint64_t>(info->si_addr), pc, kExitHostFault);
}

void install_host_fault_handlers() {
    static std::vector<uint8_t> alt_stack(1 << 20);
    stack_t ss{};
    ss.ss_sp = alt_stack.data();
    ss.ss_size = alt_stack.size();
    sigaltstack(&ss, nullptr);
    struct sigaction sa{};
    sa.sa_sigaction = &signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP}) {
        sigaction(sig, &sa, nullptr);
    }
}
#endif

// ---------------------------------------------------------------------------
// The boot
// ---------------------------------------------------------------------------

void run_boot_thread() {
#ifdef _WIN32
    ULONG guarantee = 256 * 1024;  // room for the report after a stack overflow
    SetThreadStackGuarantee(&guarantee);
#else
    install_host_fault_handlers();  // sigaltstack is per thread
#endif
#ifdef PG6_BOOT_PROGRAM
    // A program build: the executable plus guest DLLs (guest/program_loader.h, R12); every
    // module's image is <program-dir>/<module>/image/guest.pgim.
    guest::ProgramBoot program_boot;
    const std::filesystem::path program_dir(g_options.program_dir);
    const guest::ImageSource source =
        [&](const Pg6ProgramModule& module) -> std::unique_ptr<os::pgim::ImageReader> {
        auto reader = std::make_unique<FileReader>();
        const std::filesystem::path path = program_dir / module.module / "image" / "guest.pgim";
        if (!reader->open(path.string())) {
            std::fprintf(stderr, "[harness] cannot open %s\n", path.string().c_str());
            return nullptr;
        }
        return reader;
    };
    guest::ProgramLoadOptions load_options;
    load_options.module_directory = g_options.cwd.empty() ? std::string() : g_options.cwd + "\\";
    const guest::ProgramError program_error =
        guest::prepare_program_boot(pg6_program, source, load_options, program_boot);
    if (!program_error.ok()) {
        std::fprintf(stderr, "[harness] program not bootable: %s (module %zu) %s\n",
                     guest::program_error_name(program_error.code), program_error.module,
                     program_error.detail.c_str());
        g_result = kExitUsage;
        return;
    }
    guest::BootConfig& cfg = program_boot.config;
#else
    FileReader reader;
    if (!reader.open(g_options.pgim)) {
        std::fprintf(stderr, "[harness] cannot open %s\n", g_options.pgim.c_str());
        g_result = kExitUsage;
        return;
    }
    guest::BootConfig cfg;
    guest::BootModule exe;
    exe.image = &reader;
    exe.name = g_options.name;
#ifdef PG6_BOOT_LINKED_GUEST
    const auto* table = PG6_CAT(__playgta6_lifted_functions_, PG6_BOOT_MODULE);
    const auto count = static_cast<size_t>(PG6_CAT(__playgta6_lifted_function_count_, PG6_BOOT_MODULE));
    exe.lifted_tables.emplace_back(table, count);
    exe.imports = std::span<const Pg6BootImport>(reinterpret_cast<const Pg6BootImport*>(pg6_import_thunks),
                                                 pg6_import_thunk_count);
#endif
    cfg.modules.push_back(std::move(exe));
#endif
    cfg.command_line = g_options.cmdline;
    cfg.working_directory = g_options.cwd;
    cfg.stack_size = g_options.stack;
    cfg.survey = g_options.survey;
    cfg.survey_return = g_options.survey_return;
    cfg.max_dispatches = g_options.max_calls;
    cfg.max_wall_ms = g_options.max_ms;
    cfg.run_tls_callbacks = g_options.tls_callbacks;
    cfg.call_entry = g_options.entry;
    cfg.log_progress = !g_options.quiet;
    cfg.capture_host_stack = &capture_host_stack;
    cfg.import_trace_limit = g_options.import_trace.empty() ? 0 : g_options.import_trace_limit;
    cfg.host_function_start = host_function_start;
    cfg.console_mode = static_cast<uint32_t>(g_options.console_mode);
    cfg.off_thread_fault = &off_thread_fault;
    const guest::BootResult result = guest::run_boot(cfg);
    if (!g_finishing.exchange(true)) {
        write_outputs(g_options.quiet);
    }
    g_result = static_cast<int>(result.status);
}

#ifdef _WIN32
DWORD WINAPI boot_thread_entry(void*) {
    run_boot_thread();
    return 0;
}
#else
void* boot_thread_entry(void*) {
    run_boot_thread();
    return nullptr;
}
#endif

void start_watchdog() {
    if (g_options.watchdog_ms == 0) {
        return;
    }
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(g_options.watchdog_ms));
        // Reads the boot's state while the guest may still run: diagnostic, racy by nature.
        host_fault("host_watchdog_timeout", 0, 0, kExitWatchdog);
    }).detach();
}

}  // namespace

int main(int argc, char** argv) {
    if (!parse(argc, argv, g_options)) {
        usage();
        return kExitUsage;
    }
    // Claim the guest address space before anything else can map into it.
    os::VirtualMemoryArena* arena = os::win32::get_vmem_arena();
    std::fprintf(stderr,
                 "[harness] guest address space 0x%" PRIx64 "..0x%" PRIx64 ", %s (guest base 0x%" PRIx64 ")\n",
                 arena->space_base(), arena->space_base() + arena->space_size(),
                 arena->is_identity() ? "identity-mapped" : "NOT identity-mapped", arena->guest_base());
    if (!arena->is_identity()) {
        std::fprintf(stderr,
                     "[harness] warning: lifted code built without the guest-base seam addresses guest "
                     "memory 1:1 and will fault on the first guest memory access\n");
    }

    std::unique_ptr<DirectoryServer> server;
    std::unique_ptr<os::HttpfsMailbox> mailbox;
    std::unique_ptr<os::HttpfsMailboxClient> client;
    if (!g_options.game_dir.empty()) {
        server = std::make_unique<DirectoryServer>(std::filesystem::path(g_options.game_dir));
        mailbox = std::make_unique<os::HttpfsMailbox>();
        os::HttpfsMailboxClient::init_mailbox(*mailbox);
        client = std::make_unique<os::HttpfsMailboxClient>(
            mailbox.get(), [srv = server.get()](os::HttpfsMailbox& mb) { srv->handle(mb); });
        os::httpfs_attach_client(client.get());
        os::win32::set_httpfs_client(client.get());
    }

#ifdef _WIN32
    install_host_fault_handlers();
#endif
    if (!g_options.gpu_capture.empty()) {
        gpu::FrameCaptureConfig capture;
        capture.path = g_options.gpu_capture;
        capture.frames = static_cast<uint32_t>(std::max<uint64_t>(1, g_options.gpu_capture_frames));
        capture.keep_payloads = g_options.gpu_capture_payloads;
        if (!gpu::frame_capture_start(capture)) {
            std::fprintf(stderr, "[harness] cannot write the GPU capture %s\n", g_options.gpu_capture.c_str());
            return kExitUsage;
        }
        gpu::frame_capture_set_counters([](gpu::FrameCaptureSample* sample) {
            const guest::LiveCounters c = guest::live_counters();
            sample->dispatches = c.dispatches;
            sample->import_calls = c.import_calls;
            const os::VirtualMemoryArena* arena = os::win32::get_vmem_arena();
            sample->reserved_bytes = arena->reserved_bytes();
            sample->committed_bytes = arena->committed_bytes();
        });
    }
    start_watchdog();

    const uint64_t host_stack = g_options.host_stack_mb << 20;
#ifdef _WIN32
    HANDLE thread = CreateThread(nullptr, static_cast<SIZE_T>(host_stack), &boot_thread_entry, nullptr,
                                 STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (thread == nullptr) {
        std::fprintf(stderr, "[harness] cannot create the boot thread (host stack %" PRIu64 " MiB)\n",
                     g_options.host_stack_mb);
        return kExitUsage;
    }
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, static_cast<size_t>(host_stack));
    pthread_t thread;
    if (pthread_create(&thread, &attr, &boot_thread_entry, nullptr) != 0) {
        std::fprintf(stderr, "[harness] cannot create the boot thread\n");
        return kExitUsage;
    }
    pthread_join(thread, nullptr);
#endif
    std::fflush(nullptr);
    // Guest threads created by the boot may still run: do not run static destructors under them.
    std::_Exit(g_result);
}
