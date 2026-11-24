#include "guest/boot.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string_view>
#include <thread>
#include <utility>

#include "guest/com_bridge.h"
#include "guest/guest_crt_exit.h"
#include "guest/guest_exceptions.h"
#include "os/guest_window.h"
#include "os/os_shim.h"
#include "os/shim_exports.h"
#include "os/ucrt_bridge.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/win32_sync.h"
#include "os/xbox_scarlett.h"
#include "os/xgameruntime_provider.h"

#if defined(__cpp_exceptions) || defined(_CPPUNWIND) || defined(__EXCEPTIONS)
#define PG6_BOOT_HAS_EXCEPTIONS 1
#else
#define PG6_BOOT_HAS_EXCEPTIONS 0
#endif

namespace playgta6::guest {

namespace {

using os::PageProtect;
using os::VirtualMemoryArena;

constexpr uint32_t kGuestProcessId = 0x1000;  // ClientId.UniqueProcess of the guest process
constexpr uint64_t kDefaultStackSize = uint64_t{1} << 20;
constexpr size_t kMaxTlsCallbacks = 256;
constexpr size_t kMaxScanBytes = size_t{16} << 20;
constexpr uint64_t kWallCheckMask = 255;  // check the wall clock every 256 dispatches
constexpr uint32_t kTlsDirectoryIndex = 9;

// Thrown to leave guest code once a boot outcome is recorded.
struct BootStop {};

struct Frame {
    uint64_t stack_address = 0;
    uint64_t return_address = 0;
    uint64_t function = 0;
};

struct HostFrame {
    uint64_t host_pc = 0;
    uint64_t function = 0;
    uint64_t offset = 0;
};

// One dispatched (indirect or import) call: target, the return address the guest `call` pushed
// (whatever is at [rsp] for a tail jump), and the first two integer arguments.
struct DispatchRecord {
    uint64_t target = 0;
    uint64_t return_address = 0;
    uint64_t rcx = 0;
    uint64_t rdx = 0;
};

struct FaultRecord {
    bool present = false;
    std::string kind;
    uint64_t va = 0;
    uint64_t detail = 0;
    long import_index = -1;
    uint64_t host_address = 0;
    uint64_t host_pc = 0;
    bool has_host = false;
    bool has_registers = false;
    GuestState registers{};
    std::string backtrace_method = "none";
    std::vector<Frame> frames;
    bool has_host_frames = false;
    std::vector<HostFrame> host_frames;
    std::vector<DispatchRecord> recent_dispatches;  // faulting thread, oldest first
};

struct ExecRange {
    uint64_t begin = 0;
    uint64_t end = 0;
    bool bytes = false;  // executable bytes present in guest memory
};

struct TlsInfo {
    bool present = false;
    uint32_t index = 0;
    uint64_t template_start = 0;
    uint64_t template_size = 0;
    uint64_t zero_fill = 0;
    uint64_t alignment = 16;
    uint64_t index_va = 0;
    std::vector<uint64_t> callbacks;
    uint64_t callbacks_run = 0;
};

struct ModuleState {
    std::string name;
    std::string path;
    bool loaded = false;
    os::pgim::LoadedImage image;
    bool headers_mapped = false;
    uint32_t subsystem = 0;
    uint64_t stack_reserve = 0;
    uint32_t tls_rva = 0;
    uint32_t tls_size = 0;
    std::vector<std::span<const LiftedFunctionEntry>> tables;
    size_t shard_table_count = 0;
    uint64_t functions = 0;
    size_t import_count = 0;
    uint64_t imports_bound = 0;
    uint64_t imports_runtime = 0;  // unresolved by the bridge, bound to runtime thunks (R11)
    bool pdata = false;           // .pdata registered with the R11 unwinder
    uint64_t imports_unresolved = 0;
    TlsInfo tls;
};

struct ImportRecord {
    BootImport import{};
    size_t module = 0;
    uint64_t image_base = 0;
};

struct SurveyEntry {
    size_t import_index = 0;
    uint64_t first_seq = 0;
    uint64_t count = 0;
};

struct RecentImport {
    uint64_t seq = 0;
    size_t import_index = 0;
    uint64_t args[5] = {};  // rcx rdx r8 r9 [rsp+0x28] at the call
    uint32_t thread = 0;    // guest thread id (TEB ClientId)
};

// First call of an API worth a timeline entry (see kTimelineApis).
struct FirstCall {
    std::string name;
    uint64_t seq = 0;  // import call sequence number (0 for host callables)
    uint64_t elapsed_ms = 0;
    uint32_t thread = 0;
};

// One reserve / commit / decommit / release on the boot's arena.
struct ArenaRecord {
    uint64_t import_seq = 0;  // import call this happened under (0: runtime itself)
    long import_index = -1;
    os::ArenaEvent event = os::ArenaEvent::Reserve;
    bool ok = false;
    uint64_t va = 0;
    uint64_t size = 0;
    uint32_t protect = 0;
};

constexpr size_t kDefaultArenaLogLimit = 4096;

// APIs whose first call is a timeline milestone, with the W40 milestone name where one exists.
struct TimelineApi {
    const char* name;
    const char* milestone;  // W40 BOOT_MILESTONES name or nullptr
};
constexpr TimelineApi kTimelineApis[] = {
    {"_initterm", "CRT_INITTERM"},
    {"_initterm_e", "CRT_INITTERM"},
    {"XMemVirtualAlloc", "XMEM_INIT"},
    {"XMemAllocatePhysicalPages", nullptr},
    {"XGameRuntimeInitialize", "GDK_INIT"},
    {"QueryApiImpl", nullptr},
    {"InitializeApiImpl", nullptr},
    {"LoadLibraryA", nullptr},
    {"LoadLibraryW", nullptr},
    {"LoadLibraryExA", nullptr},
    {"LoadLibraryExW", nullptr},
    {"GetProcAddress", nullptr},
    {"CreateThread", nullptr},
    {"CreateFileA", nullptr},
    {"CreateFileW", nullptr},
    {"CreateFile2", nullptr},
    {"DStorageGetFactory", nullptr},
    {"GameInputCreate", nullptr},
    {"WSAStartup", nullptr},
    {"D3D12ConfigureVirtualMemoryX", nullptr},
    {"D3D12XboxCreateDevice", "D3D12_CREATE_DEVICE"},
    {"BinkOpen", "BINK_OPEN"},
};

struct Milestone {
    const char* name = "";
    uint64_t elapsed_ms = 0;
    uint64_t import_seq = 0;  // import calls made when the milestone was recorded
};

std::atomic<ReportExtension> g_report_extension{nullptr};

struct BootState {
    std::mutex mutex;  // recent ring, survey list

    bool active = false;   // a boot's process exists (run_boot ran, reset_boot not yet)
    bool running = false;  // guest code of the boot may run
    VirtualMemoryArena* arena = nullptr;

    // Options.
    bool survey = false;
    uint64_t survey_return = 0;
    uint64_t max_dispatches = 0;
    uint64_t max_wall_ms = 0;
    size_t recent_capacity = 32;
    size_t max_frames = 32;
    CaptureHostStackFn capture_host_stack = nullptr;
    HostFunctionStartFn host_function_start = nullptr;
    OffThreadFaultFn off_thread_fault = nullptr;
    GuestThreadStartedFn guest_thread_started = nullptr;
    std::atomic<bool> off_thread_fault_taken{false};
    bool log_progress = true;
    std::chrono::steady_clock::time_point start;

    // Modules (modules[0] = the executable) and their code.
    std::vector<ModuleState> modules;
    std::vector<ExecRange> exec;
    std::vector<std::span<const LiftedFunctionEntry>> tables;  // all modules
    size_t shard_table_count = 0;
    uint64_t functions_registered = 0;
    std::vector<std::pair<uint64_t, uint64_t>> host_map;  // host fn -> guest va (lazy)
    bool host_map_built = false;

    // Imports of all modules.
    std::vector<ImportRecord> imports;
    std::vector<size_t> by_slot;  // import indices sorted by (image_base, slot_rva)
    std::unique_ptr<std::atomic<uint64_t>[]> import_counts;
    uint64_t imports_bound = 0;
    uint64_t imports_unresolved = 0;

    // Process.
    uint64_t process_reservation = 0;
    uint64_t peb = 0;
    uint64_t process_parameters = 0;
    std::vector<size_t> tls_modules;  // module indices in TLS index order
    std::vector<std::unique_ptr<GuestThreadContext>> threads;  // owned contexts
    GuestThreadContext* main_thread = nullptr;

    // Counters.
    std::atomic<uint64_t> dispatches{0};
    std::atomic<uint64_t> import_calls{0};
    std::vector<RecentImport> recent;  // ring
    size_t recent_next = 0;
    std::vector<SurveyEntry> survey_called;
    uint64_t tls_callbacks_run = 0;
    bool entry_started = false;
    std::vector<Milestone> milestones;
    std::vector<FirstCall> first_calls;  // guarded by `mutex`
    // Import calls per (guest thread << 32 | import index) until the first Present (mark_first_present).
    std::unordered_map<uint64_t, uint64_t> pre_present_counts;  // guarded by `mutex`
    bool present_seen = false;                                  // guarded by `mutex`
    uint64_t present_import_seq = 0;
    uint64_t present_elapsed_ms = 0;
    std::vector<RecentImport> trace;     // optional import trace, guarded by `mutex`
    std::vector<std::string> trace_strings;  // string arguments of each trace entry
    struct CallableCalls {
        const char* name = nullptr;
        uint64_t calls = 0;
        uint64_t first_import_seq = 0;  // import calls made before the first call
        size_t order = 0;
    };
    std::mutex callable_mutex;
    std::unordered_map<uint64_t, CallableCalls> callable_calls;  // synthetic target -> calls
    size_t trace_limit = 0;
    bool trace_truncated = false;
    std::unordered_map<const GuestThreadContext*, RecentImport> thread_last;  // guarded by `mutex`
    // The live guest state of contexts whose state lives elsewhere (O16 lazy threads).
    std::unordered_map<const GuestThreadContext*, const GuestState*> thread_states;

    // Address-space log of the boot's arena (guarded by `arena_mutex`).
    std::mutex arena_mutex;
    std::vector<ArenaRecord> arena_log;
    uint64_t arena_events = 0;
    size_t arena_log_limit = kDefaultArenaLogLimit;
    uint64_t arena_failures = 0;
    std::unordered_map<uint64_t, uint64_t> live_reservations;  // base -> requested size
    uint64_t reserved_now = 0;
    uint64_t reserved_peak = 0;
    uint64_t commit_requests = 0;  // bytes requested by successful commits (not net)
    bool arena_observer_installed = false;
    os::ArenaObserver previous_arena_observer = nullptr;

    // Outcome.
    std::string phase = "config";
    BootStatus status = BootStatus::BadConfig;
    int32_t exit_code = 0;
    bool has_exit_code = false;
    std::string message;
    std::string memory_model = "flat";  // describe_memory_model of the boot's model (O29)
    FaultRecord fault;

    std::string last_report;
    std::string last_summary;

    // Hooks to restore.
    FaultHandler previous_fault_handler = nullptr;
    DispatchObserver previous_observer = nullptr;
    Pg6CrtExitHook previous_exit_hook = nullptr;
    bool hooks_installed = false;
    bool thread_hook_installed = false;
};

BootState& boot() {
    static BootState state;
    return state;
}

thread_local GuestThreadContext* t_active = nullptr;
thread_local bool t_running_boot = false;
// Import call (index into BootState::imports, sequence number) running on this thread, for
// attributing arena events to the API that caused them.
thread_local long t_last_import = -1;
thread_local uint64_t t_last_import_seq = 0;
// Ring of the last dispatched calls on this thread (fault reports).
constexpr size_t kDispatchRing = 32;
thread_local DispatchRecord t_dispatch_ring[kDispatchRing];
thread_local uint64_t t_dispatch_count = 0;

void logf(const char* fmt, ...) {
    if (!boot().log_progress) {
        return;
    }
    char line[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);
    os::log(line);
}

uint64_t elapsed_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - boot().start)
                                     .count());
}

// Records a W40 boot milestone and logs it as `[boot] <NAME> <text>`.
void milestone(const char* name, const char* fmt, ...) {
    BootState& b = boot();
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        b.milestones.push_back(Milestone{name, elapsed_ms(), b.import_calls.load(std::memory_order_relaxed)});
    }
    if (!b.log_progress) {
        return;
    }
    char text[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    char line[640];
    std::snprintf(line, sizeof(line), "[boot] %s %s", name, text);
    os::log(line);
}

// ---------------------------------------------------------------------------
// Guest memory helpers
// ---------------------------------------------------------------------------

bool read_u64(VirtualMemoryArena& arena, uint64_t va, uint64_t& out) {
    return arena.read(va, &out, sizeof(out));
}

bool read_u32(VirtualMemoryArena& arena, uint64_t va, uint32_t& out) {
    return arena.read(va, &out, sizeof(out));
}

// Writes guest memory, temporarily lifting a read-only protection (IAT-style data).
bool force_write(VirtualMemoryArena& arena, uint64_t va, const void* src, size_t len) {
    if (arena.write(va, src, len)) {
        return true;
    }
    const uint64_t page = os::align_down(va, os::kPageSize);
    uint64_t end = 0;
    if (!os::align_up_checked(va + len, os::kPageSize, end)) {
        return false;
    }
    PageProtect old_protect = PageProtect::NoAccess;
    if (!arena.protect(page, end - page, PageProtect::ReadWrite, &old_protect)) {
        return false;
    }
    const bool ok = arena.write(va, src, len);
    arena.protect(page, end - page, old_protect);
    return ok;
}

// Reserves and commits a read/write block in the arena. Returns its base or 0.
uint64_t alloc_block(VirtualMemoryArena& arena, uint64_t size) {
    uint64_t rounded = 0;
    if (!os::align_up_checked(size, os::kPageSize, rounded)) {
        return 0;
    }
    const uint64_t base = arena.reserve(0, rounded, PageProtect::ReadWrite);
    if (base == 0) {
        return 0;
    }
    if (!arena.commit(base, rounded, PageProtect::ReadWrite)) {
        arena.release(base);
        return 0;
    }
    return base;
}

std::u16string utf8_to_utf16(std::string_view text) {
    std::u16string out;
    size_t i = 0;
    while (i < text.size()) {
        const auto c = static_cast<uint8_t>(text[i]);
        uint32_t cp = 0xFFFD;
        size_t n = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            cp = ((c & 0x1Fu) << 6) | (static_cast<uint8_t>(text[i + 1]) & 0x3Fu);
            n = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            cp = ((c & 0x0Fu) << 12) | ((static_cast<uint8_t>(text[i + 1]) & 0x3Fu) << 6) |
                 (static_cast<uint8_t>(text[i + 2]) & 0x3Fu);
            n = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
            cp = ((c & 0x07u) << 18) | ((static_cast<uint8_t>(text[i + 1]) & 0x3Fu) << 12) |
                 ((static_cast<uint8_t>(text[i + 2]) & 0x3Fu) << 6) |
                 (static_cast<uint8_t>(text[i + 3]) & 0x3Fu);
            n = 4;
        }
        i += n;
        if (cp >= 0x10000 && cp <= 0x10FFFF) {
            cp -= 0x10000;
            out.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
            out.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
        } else {
            out.push_back(static_cast<char16_t>(cp > 0xFFFF ? 0xFFFD : cp));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

// Highest registered lifted entry <= va (0 if none), across both per-function and shard dispatch tables.
uint64_t nearest_entry(uint64_t va) {
    uint64_t best = nearest_lifted_function_entry(va);
    for (const auto& table : boot().tables) {
        if (table.empty() || va < table.front().va) {
            continue;
        }
        const auto it = std::upper_bound(
            table.begin(), table.end(), va,
            [](uint64_t v, const LiftedFunctionEntry& e) { return v < e.va; });
        if (it != table.begin()) {
            best = std::max(best, (it - 1)->va);
        }
    }
    return best;
}

long find_import(uint64_t image_base, uint32_t slot_rva) {
    const BootState& b = boot();
    const auto key = std::make_pair(image_base, slot_rva);
    const auto it = std::lower_bound(b.by_slot.begin(), b.by_slot.end(), key,
                                     [&](size_t index, const std::pair<uint64_t, uint32_t>& k) {
                                         const ImportRecord& r = b.imports[index];
                                         return std::make_pair(r.image_base, r.import.slot_rva) < k;
                                     });
    if (it != b.by_slot.end() && b.imports[*it].image_base == image_base &&
        b.imports[*it].import.slot_rva == slot_rva) {
        return static_cast<long>(*it);
    }
    return -1;
}

// Import index for a synthetic call target bound by this boot, or -1.
long import_for_target(uint64_t target) {
    SyntheticTarget info;
    if (!is_synthetic_address(target) || !describe_synthetic(target, &info) || !info.is_import) {
        return -1;
    }
    return find_import(info.image_base, info.slot_rva);
}

bool import_resolved(const BootImport& import) {
    return (import.flags & kPg6BootImportBound) != 0 && import.thunk != nullptr;
}

std::string import_symbol(const BootImport& import) {
    return import.symbol != nullptr ? std::string(import.symbol) : "#" + std::to_string(import.ordinal);
}

std::string import_label(const BootImport& import) {
    return std::string(import.dll != nullptr ? import.dll : "?") + "!" + import_symbol(import);
}

void build_host_map() {
    BootState& b = boot();
    if (b.host_map_built) {
        return;
    }
    b.host_map_built = true;
    size_t total = 0;
    for (const auto& table : b.tables) {
        total += table.size();
    }
    b.host_map.reserve(total);
    for (const auto& table : b.tables) {
        for (const LiftedFunctionEntry& e : table) {
            b.host_map.emplace_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(e.fn)), e.va);
        }
    }
    std::sort(b.host_map.begin(), b.host_map.end());
}

bool in_exec_range(uint64_t va, bool* bytes) {
    for (const ExecRange& r : boot().exec) {
        if (va >= r.begin && va < r.end) {
            *bytes = r.bytes;
            return true;
        }
    }
    return false;
}

// Do the 8 bytes before a return address end with a `call rel32` or `call r/m64` (FF /2)?
bool ends_with_call(const uint8_t (&b)[8]) {
    if (b[3] == 0xE8) {
        return true;
    }
    for (size_t start = 0; start + 2 <= 8; ++start) {
        if (b[start] != 0xFF) {
            continue;
        }
        const uint8_t modrm = b[start + 1];
        if (((modrm >> 3) & 7) != 2) {
            continue;
        }
        const uint8_t mod = modrm >> 6;
        const uint8_t rm = modrm & 7;
        size_t length = 2;
        if (mod != 3) {
            if (rm == 4) {
                if (start + 2 >= 8) {
                    continue;
                }
                const uint8_t sib = b[start + 2];
                length += 1;
                if (mod == 0 && (sib & 7) == 5) {
                    length += 4;
                }
            } else if (mod == 0 && rm == 5) {
                length += 4;  // rip-relative
            }
            if (mod == 1) {
                length += 1;
            } else if (mod == 2) {
                length += 4;
            }
        }
        if (start + length == 8) {
            return true;
        }
    }
    return false;
}

void scan_backtrace(FaultRecord& record, uint64_t rsp, const GuestThreadContext* thread = nullptr) {
    BootState& b = boot();
    if (b.arena == nullptr || rsp == 0) {
        return;
    }
    uint64_t top = rsp + 0x10000;
    const GuestThreadContext* owner = thread != nullptr ? thread : t_active;
    if (owner != nullptr && rsp >= owner->thread.stack_limit &&
        rsp < owner->thread.stack_top) {
        top = owner->thread.stack_top;
    }
    const uint64_t bytes = std::min<uint64_t>(top - rsp, kMaxScanBytes) & ~uint64_t{7};
    std::vector<uint64_t> words(static_cast<size_t>(bytes / 8));
    if (words.empty() || !b.arena->read(rsp, words.data(), words.size() * 8)) {
        // Fall back to the largest readable prefix, word by word.
        words.clear();
        for (uint64_t va = rsp; va + 8 <= rsp + bytes; va += 8) {
            uint64_t w = 0;
            if (!read_u64(*b.arena, va, w)) {
                break;
            }
            words.push_back(w);
        }
    }
    bool any_unverified = false;
    bool any_verified = false;
    for (size_t i = 0; i < words.size() && record.frames.size() < b.max_frames; ++i) {
        const uint64_t q = words[i];
        bool has_bytes = false;
        if (q < 8 || !in_exec_range(q, &has_bytes)) {
            continue;
        }
        const uint64_t entry = nearest_entry(q);
        if (entry == 0 || entry == q) {
            continue;
        }
        if (has_bytes) {
            uint8_t code[8];
            if (!b.arena->read(q - 8, code, sizeof(code)) || !ends_with_call(code)) {
                continue;
            }
            any_verified = true;
        } else {
            any_unverified = true;
        }
        record.frames.push_back(Frame{rsp + 8 * i, q, entry});
    }
    if (any_unverified) {
        record.backtrace_method = any_verified ? "stack-scan-mixed" : "stack-scan-unverified";
    } else {
        record.backtrace_method = "stack-scan-call-verified";
    }
}

void capture_host_frames(FaultRecord& record) {
    BootState& b = boot();
    if (b.capture_host_stack == nullptr) {
        return;
    }
    void* frames[128];
    const size_t n = b.capture_host_stack(frames, 128);
    record.has_host_frames = true;
    for (size_t i = 0; i < n && record.host_frames.size() < 64; ++i) {
        const uint64_t pc = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(frames[i]));
        uint64_t offset = 0;
        const uint64_t fn = guest_function_for_host_pc(pc, &offset);
        if (fn != 0) {
            record.host_frames.push_back(HostFrame{pc, fn, offset});
        }
    }
}

// Walks the guest stack with R11's RtlVirtualUnwind over the images' .pdata (guest_exceptions.h).
// `inside` = the fault is at `va` inside a function (the state's rsp is that function's rsp);
// otherwise rsp points at the return address of a call (imports, dispatch targets). Records the
// return addresses of the callers. Returns false (record untouched) when the first frame has no
// unwind data, so the caller can fall back to the stack scan.
bool unwind_backtrace(FaultRecord& record, const GuestState& state, uint64_t va, bool inside) {
    BootState& b = boot();
    VirtualMemoryArena& arena = *b.arena;
    auto ctx = std::make_unique<GuestContext>();
    std::memset(ctx.get(), 0, sizeof(*ctx));
    std::memcpy(ctx->gpr, state.gpr, sizeof(ctx->gpr));
    uint64_t stack_top = UINT64_MAX;
    if (t_active != nullptr && state.gpr[GuestReg::Rsp] >= t_active->thread.stack_limit &&
        state.gpr[GuestReg::Rsp] < t_active->thread.stack_top) {
        stack_top = t_active->thread.stack_top;
    }
    std::vector<Frame> frames;
    uint64_t pc = va;
    if (!inside) {
        uint64_t ret = 0;
        if (!read_u64(arena, ctx->gpr[GuestReg::Rsp], ret)) {
            return false;
        }
        frames.push_back(Frame{ctx->gpr[GuestReg::Rsp], ret, nearest_entry(ret)});
        ctx->gpr[GuestReg::Rsp] += 8;
        pc = ret;
    }
    if (pc_to_image_base(pc) == 0) {
        return false;  // no registered .pdata for this code: the stack scan does better
    }
    bool complete = false;
    for (size_t depth = 0; depth < 4 * b.max_frames && frames.size() < b.max_frames; ++depth) {
        if (pc == kGuestReturnSentinel) {
            complete = true;
            break;
        }
        bool has_bytes = false;
        if (!in_exec_range(pc, &has_bytes)) {
            break;
        }
        uint64_t image_base = 0;
        const uint64_t entry = lookup_function_entry(arena, pc, &image_base);
        const uint64_t rsp_before = ctx->gpr[GuestReg::Rsp];
        if (entry == 0) {
            uint64_t ret = 0;  // leaf: the return address is at rsp
            if (!read_u64(arena, rsp_before, ret)) {
                break;
            }
            ctx->rip = ret;
            ctx->gpr[GuestReg::Rsp] = rsp_before + 8;
        } else {
            ctx->rip = pc;
            uint64_t handler_data = 0;
            uint64_t establisher = 0;
            bool ok = false;
            virtual_unwind(arena, 0, image_base, pc, entry, *ctx, &handler_data, &establisher, nullptr, &ok);
            if (!ok) {
                break;
            }
        }
        if (ctx->gpr[GuestReg::Rsp] <= rsp_before || ctx->gpr[GuestReg::Rsp] > stack_top) {
            break;  // no progress or off the stack
        }
        pc = ctx->rip;
        if (pc == kGuestReturnSentinel) {
            complete = true;
            break;
        }
        frames.push_back(Frame{ctx->gpr[GuestReg::Rsp] - 8, pc, nearest_entry(pc)});
    }
    record.frames = std::move(frames);
    record.backtrace_method = complete ? "rtl-virtual-unwind" : "rtl-virtual-unwind-partial";
    return true;
}

// Fills the common parts of a fault record from a guest state. `inside` as for unwind_backtrace.
void capture_fault(const char* kind, GuestState* state, uint64_t va, uint64_t detail, bool inside = false) {
    BootState& b = boot();
    FaultRecord& r = b.fault;
    r = FaultRecord{};
    r.present = true;
    r.kind = kind;
    r.va = va;
    r.detail = detail;
    GuestState* regs = state != nullptr ? state : (t_active != nullptr ? &t_active->thread.state : nullptr);
    if (regs != nullptr) {
        r.has_registers = true;
        r.registers = *regs;
        if (!unwind_backtrace(r, *regs, va, inside)) {
            scan_backtrace(r, regs->gpr[GuestReg::Rsp]);
        }
    }
    capture_host_frames(r);
    const uint64_t count = std::min<uint64_t>(t_dispatch_count, kDispatchRing);
    for (uint64_t k = t_dispatch_count - count; k < t_dispatch_count; ++k) {
        r.recent_dispatches.push_back(t_dispatch_ring[k % kDispatchRing]);
    }
}

[[noreturn]] void stop_boot() {
#if PG6_BOOT_HAS_EXCEPTIONS
    throw BootStop{};
#else
    os::log("guest boot: stopping without exception support; aborting the process");
    std::abort();
#endif
}

// W40 TRAP_KIND for a fault kind (null for run bounds).
const char* w40_trap_kind(std::string_view kind) {
    if (kind == "unresolved_import") return "UNRESOLVED_IMPORT";
    if (kind == "unimplemented") return "UNIMPLEMENTED_OPCODE";
    if (kind == "unknown_target") return "INDIRECT_CALL_MISS";
    if (kind == "divide_error") return "DIVIDE_ERROR";
    if (kind == "dispatch_limit" || kind == "wall_time_limit") return nullptr;
    return "GUEST_TRAP";
}

// ---------------------------------------------------------------------------
// Hooks active while a boot runs
// ---------------------------------------------------------------------------

const char* fault_kind_id(FaultKind kind) {
    switch (kind) {
        case FaultKind::Trap:
            return "trap";
        case FaultKind::Unimplemented:
            return "unimplemented";
        case FaultKind::DivideError:
            return "divide_error";
        case FaultKind::UnknownTarget:
            return "unknown_target";
        case FaultKind::StackImbalance:
            return "stack_imbalance";
        case FaultKind::BadStack:
            return "bad_stack";
        case FaultKind::UnhandledException:
            return "unhandled_exception";
        case FaultKind::UnknownResumeTarget:
            return "unknown_resume_target";
        case FaultKind::UnwindFailed:
            return "unwind_failed";
        case FaultKind::UnimplementedSlot:
            return "unimplemented_slot";
        case FaultKind::GeneralProtection:
            return "general_protection";
        default:
            break;
    }
    return "unknown";
}

void log_trap();
void finish_report();

void boot_fault_handler(const FaultInfo* info) {
    BootState& b = boot();
    if (!b.running || info == nullptr) {
        return;  // the default handler logs and aborts
    }
    const bool other_thread = !t_running_boot;
    if (other_thread && (b.off_thread_fault == nullptr || b.off_thread_fault_taken.exchange(true))) {
        return;
    }
    const auto kind = static_cast<FaultKind>(info->kind);
    const bool inside = kind == FaultKind::Trap || kind == FaultKind::Unimplemented ||
                        kind == FaultKind::DivideError;
    capture_fault(fault_kind_id(kind), info->state, info->va, info->detail, inside);
    char message[256];
    std::snprintf(message, sizeof(message), "%s at 0x%016" PRIx64, fault_kind_name(kind), info->va);
    b.message = message;
    if (other_thread) {
        b.message += " on guest thread " + std::to_string(t_active != nullptr ? t_active->thread_id : 0);
    }
    b.status = BootStatus::Fault;
    if (other_thread) {
        log_trap();
        finish_report();
        b.off_thread_fault();  // does not return
        return;
    }
    stop_boot();
}

void stop_on_limit(GuestState* state, const char* kind, uint64_t target) {
    BootState& b = boot();
    capture_fault(kind, state, target, 0);
    char message[160];
    std::snprintf(message, sizeof(message), "run bound reached (%s) after %" PRIu64 " dispatches",
                  kind, b.dispatches.load(std::memory_order_relaxed));
    b.message = message;
    b.status = BootStatus::LimitReached;
    stop_boot();
}

// ` aN="..."` / ` aN=L"..."` for integer arguments that point at a readable, printable NUL-ended
// (or 48+ character) ASCII or UTF-16 string in guest memory (import trace only).
std::string describe_string_args(const uint64_t (&args)[5]) {
    BootState& b = boot();
    std::string out;
    if (b.arena == nullptr) {
        return out;
    }
    for (size_t i = 0; i < 4; ++i) {
        const uint64_t va = args[i];
        if (va < b.arena->space_base() || va - b.arena->space_base() >= b.arena->space_size()) {
            continue;
        }
        uint8_t bytes[96];
        size_t len = sizeof(bytes);
        while (len >= 8 && !b.arena->read(va, bytes, len)) {
            len /= 2;
        }
        if (len < 8) {
            continue;
        }
        auto printable = [](uint8_t c) { return c >= 0x20 && c < 0x7F; };
        std::string text;
        bool wide = false;
        size_t n = 0;
        while (n < len && printable(bytes[n])) {
            ++n;
        }
        if (n >= 3 && (n == len || bytes[n] == 0)) {
            text.assign(reinterpret_cast<const char*>(bytes), std::min<size_t>(n, 48));
        } else {
            size_t w = 0;
            while (2 * w + 1 < len && printable(bytes[2 * w]) && bytes[2 * w + 1] == 0) {
                ++w;
            }
            if (w >= 3 && (2 * w + 1 >= len || (bytes[2 * w] == 0 && bytes[2 * w + 1] == 0))) {
                wide = true;
                for (size_t k = 0; k < w && k < 48; ++k) {
                    text.push_back(static_cast<char>(bytes[2 * k]));
                }
            }
        }
        if (!text.empty()) {
            out += " a" + std::to_string(i) + (wide ? "=L\"" : "=\"") + text + "\"";
        }
    }
    return out;
}

// The registered name of a host callable at a synthetic address (COM slots, GetProcAddress
// callables), else null.
const char* synthetic_name(uint64_t target) {
    SyntheticTarget info;
    if (is_synthetic_address(target) && describe_synthetic(target, &info) && !info.is_import) {
        return info.name;
    }
    return nullptr;
}

// What a dispatched synthetic target is: a host callable's name or an import's dll!symbol
// (empty for lifted code and unknown targets).
std::string dispatch_label(uint64_t target) {
    if (const char* name = synthetic_name(target)) {
        return name;
    }
    const long index = import_for_target(target);
    return index >= 0 ? import_label(boot().imports[static_cast<size_t>(index)].import) : std::string();
}

void boot_dispatch_observer(GuestState* state, uint64_t target) {
    BootState& b = boot();
    if (!b.running) {
        return;
    }
    const uint64_t n = b.dispatches.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        DispatchRecord& d = t_dispatch_ring[t_dispatch_count++ % kDispatchRing];
        d.target = target;
        d.rcx = state->gpr[GuestReg::Rcx];
        d.rdx = state->gpr[GuestReg::Rdx];
        d.return_address = 0;
        if (b.arena != nullptr) {
            read_u64(*b.arena, state->gpr[GuestReg::Rsp], d.return_address);
        }
    }
    if (is_synthetic_address(target)) {
        const long index = import_for_target(target);
        if (index >= 0) {
            const uint64_t previous =
                b.import_counts[static_cast<size_t>(index)].fetch_add(1, std::memory_order_relaxed);
            const uint64_t seq = b.import_calls.fetch_add(1, std::memory_order_relaxed) + 1;
            t_last_import = index;
            t_last_import_seq = seq;
            RecentImport entry{seq, static_cast<size_t>(index)};
            entry.args[0] = state->gpr[GuestReg::Rcx];
            entry.args[1] = state->gpr[GuestReg::Rdx];
            entry.args[2] = state->gpr[GuestReg::R8];
            entry.args[3] = state->gpr[GuestReg::R9];
            if (b.arena != nullptr) {
                read_u64(*b.arena, state->gpr[GuestReg::Rsp] + 0x28, entry.args[4]);
            }
            entry.thread = t_active != nullptr ? t_active->thread_id : 0;
            const BootImport& imp = b.imports[static_cast<size_t>(index)].import;
            std::string strings;
            if (b.trace_limit != 0) {
                strings = describe_string_args(entry.args);
            }
            const char* first_call_milestone = nullptr;
            bool first_call = false;
            if (previous == 0 && imp.symbol != nullptr) {
                for (const TimelineApi& api : kTimelineApis) {
                    if (std::strcmp(api.name, imp.symbol) == 0) {
                        first_call = true;
                        first_call_milestone = api.milestone;
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(b.mutex);
                if (b.recent_capacity != 0) {
                    if (b.recent.size() < b.recent_capacity) {
                        b.recent.push_back(entry);
                    } else {
                        b.recent[b.recent_next] = entry;
                    }
                    b.recent_next = (b.recent_next + 1) % b.recent_capacity;
                }
                if (b.trace_limit != 0) {
                    if (b.trace.size() < b.trace_limit) {
                        b.trace.push_back(entry);
                        b.trace_strings.push_back(std::move(strings));
                    } else {
                        b.trace_truncated = true;
                    }
                }
                if (t_active != nullptr) {
                    b.thread_last[t_active] = entry;
                }
                if (!b.present_seen) {
                    ++b.pre_present_counts[(static_cast<uint64_t>(entry.thread) << 32) | static_cast<uint32_t>(index)];
                }
                if (first_call) {
                    b.first_calls.push_back(FirstCall{imp.symbol, seq, elapsed_ms(), entry.thread});
                }
            }
            if (first_call_milestone != nullptr) {
                milestone(first_call_milestone, "first %s (import call #%" PRIu64 ")", imp.symbol, seq);
            }
        } else {
            // Host callables handed out by GetProcAddress / providers (QueryApiImpl, ...).
            SyntheticTarget info;
            if (describe_synthetic(target, &info) && !info.is_import && info.name != nullptr) {
                {
                    // Per-callable call counts with the import-call number of the first call
                    // (COM methods: D3D12 / xgameruntime milestones and slot statistics).
                    std::lock_guard<std::mutex> lock(b.callable_mutex);
                    auto [it, inserted] = b.callable_calls.try_emplace(target);
                    if (inserted) {
                        it->second.name = info.name;
                        it->second.first_import_seq = b.import_calls.load(std::memory_order_relaxed);
                        it->second.order = b.callable_calls.size();
                    }
                    ++it->second.calls;
                }
                for (const TimelineApi& api : kTimelineApis) {
                    if (std::strcmp(api.name, info.name) != 0) {
                        continue;
                    }
                    std::lock_guard<std::mutex> lock(b.mutex);
                    const bool seen = std::any_of(b.first_calls.begin(), b.first_calls.end(),
                                                  [&](const FirstCall& f) { return f.name == info.name; });
                    if (!seen) {
                        b.first_calls.push_back(FirstCall{info.name, 0, elapsed_ms(),
                                                          t_active != nullptr ? t_active->thread_id : 0});
                    }
                }
            }
        }
    }
    if (!t_running_boot) {
        return;  // bounds can only stop the boot's own thread
    }
    if (b.max_dispatches != 0 && n > b.max_dispatches) {
        stop_on_limit(state, "dispatch_limit", target);
    }
    if (b.max_wall_ms != 0 && (n & kWallCheckMask) == 0 && elapsed_ms() > b.max_wall_ms) {
        stop_on_limit(state, "wall_time_limit", target);
    }
}

// Address-space log: what the guest (through shims) reserves and commits, in order, attributed to
// the import call running on the requesting thread. Runs under the arena's lock (no arena calls).
void boot_arena_observer(const os::VirtualMemoryArena& arena,
                         os::ArenaEvent event,
                         uint64_t va,
                         uint64_t size,
                         os::PageProtect protect,
                         bool ok) {
    BootState& b = boot();
    if (&arena != b.arena) {
        return;
    }
    std::lock_guard<std::mutex> lock(b.arena_mutex);
    ++b.arena_events;
    if (!ok) {
        ++b.arena_failures;
    }
    if (ok) {
        switch (event) {
            case os::ArenaEvent::Reserve:
                b.live_reservations[va] = size;
                b.reserved_now += size;
                b.reserved_peak = std::max(b.reserved_peak, b.reserved_now);
                break;
            case os::ArenaEvent::Release: {
                auto it = b.live_reservations.find(va);
                if (it != b.live_reservations.end()) {
                    b.reserved_now -= it->second;
                    b.live_reservations.erase(it);
                }
                break;
            }
            case os::ArenaEvent::Commit:
                b.commit_requests += size;
                break;
            case os::ArenaEvent::Decommit:
                break;
        }
    }
    if (b.arena_log.size() < b.arena_log_limit) {
        b.arena_log.push_back(ArenaRecord{t_last_import_seq, t_last_import, event, ok, va, size,
                                          static_cast<uint32_t>(protect)});
    }
}

// Bound to every IAT slot whose import did not resolve to a runtime shim.
void boot_unresolved_thunk(void* raw_state) {
    auto* state = static_cast<GuestState*>(raw_state);
    BootState& b = boot();
    const long index = import_for_target(state->rip);
    uint64_t return_address = 0;
    if (b.arena != nullptr) {
        read_u64(*b.arena, state->gpr[GuestReg::Rsp], return_address);
    }
    if (b.running && b.survey && index >= 0) {
        const auto i = static_cast<size_t>(index);
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(b.mutex);
            auto it = std::find_if(b.survey_called.begin(), b.survey_called.end(),
                                   [&](const SurveyEntry& e) { return e.import_index == i; });
            if (it == b.survey_called.end()) {
                b.survey_called.push_back(
                    SurveyEntry{i, b.import_calls.load(std::memory_order_relaxed), 1});
                first = true;
            } else {
                ++it->count;
            }
        }
        if (first) {
            logf("[import] unresolved import %s called from 0x%016" PRIx64
                 " (survey mode: returning 0x%" PRIx64 ")",
                 import_label(b.imports[i].import).c_str(), return_address, b.survey_return);
        }
        state->gpr[GuestReg::Rax] = b.survey_return;
        state->gpr[GuestReg::Rsp] += 8;  // like `ret`
        return;
    }
    if (b.running && t_running_boot) {
        capture_fault("unresolved_import", state, return_address, state->rip);
        b.fault.import_index = index;
        b.message = "call to unresolved import " +
                    (index >= 0 ? import_label(b.imports[static_cast<size_t>(index)].import)
                                : std::string("(unknown slot)"));
        b.status = BootStatus::Fault;
        stop_boot();
    }
    // Another guest thread during a boot: reported like any other off-thread fault (the hook
    // writes the report and parks the thread), once.
    if (b.running && b.off_thread_fault != nullptr && !b.off_thread_fault_taken.exchange(true)) {
        capture_fault("unresolved_import", state, return_address, state->rip);
        b.fault.import_index = index;
        b.message = "call to unresolved import " +
                    (index >= 0 ? import_label(b.imports[static_cast<size_t>(index)].import)
                                : std::string("(unknown slot)")) +
                    " on guest thread " + std::to_string(t_active != nullptr ? t_active->thread_id : 0);
        b.status = BootStatus::Fault;
        log_trap();
        finish_report();
        b.off_thread_fault();  // does not return
    }
    // Not the boot thread (or no boot): behave like the generated logging thunk.
    char message[256];
    std::snprintf(message, sizeof(message), "[trap] UNRESOLVED_IMPORT %s called outside the boot thread",
                  index >= 0 ? import_label(b.imports[static_cast<size_t>(index)].import).c_str() : "?");
    os::log(message);
    std::abort();
}

const char* crt_exit_kind_id(int32_t kind) {
    switch (kind) {
        case PG6_CRT_EXIT_NORMAL:
            return "exit";
        case PG6_CRT_EXIT_QUICK:
            return "quick_exit";
        case PG6_CRT_EXIT_ABORT:
            return "crt_abort";
        case PG6_CRT_EXIT_TERMINATE:
            return "crt_terminate";
        case PG6_CRT_EXIT_INVALID_PARAM:
            return "crt_invalid_parameter";
        case PG6_CRT_EXIT_PURECALL:
            return "crt_purecall";
        case PG6_CRT_EXIT_CXX_THROW:
            return "cxx_throw";
        case PG6_CRT_EXIT_LONGJMP:
            return "longjmp";
        default:
            return "crt_unknown";
    }
}

void boot_exit_hook(int32_t code, int32_t kind) {
    BootState& b = boot();
    if (!b.running || !t_running_boot) {
        if (b.previous_exit_hook != nullptr) {
            b.previous_exit_hook(code, kind);
        }
        return;
    }
    if (kind == PG6_CRT_EXIT_NORMAL || kind == PG6_CRT_EXIT_QUICK) {
        b.status = BootStatus::Exited;
        b.exit_code = code;
        b.has_exit_code = true;
        b.message = std::string("guest called ") + crt_exit_kind_id(kind) + "(" + std::to_string(code) + ")";
        stop_boot();
    }
    capture_fault(crt_exit_kind_id(kind), t_active != nullptr ? &t_active->thread.state : nullptr,
                  t_active != nullptr ? t_active->thread.state.rip : 0, static_cast<uint64_t>(code));
    b.status = (kind == PG6_CRT_EXIT_CXX_THROW || kind == PG6_CRT_EXIT_LONGJMP) ? BootStatus::Fault
                                                                                 : BootStatus::GuestAbort;
    b.message = std::string("guest terminated through the CRT: ") + crt_exit_kind_id(kind);
    stop_boot();
}

// TerminateProcess(GetCurrentProcess(), code) on the boot thread ends the run like _exit: no
// guest exit handlers run (guest/guest_crt_exit.h).
void boot_terminate_process_handler(uint32_t exit_code, void* /*user*/) {
    request_exit(static_cast<int32_t>(exit_code));
}

void install_hooks() {
    BootState& b = boot();
    b.previous_fault_handler = set_fault_handler(&boot_fault_handler);
    b.previous_observer = set_dispatch_observer(&boot_dispatch_observer);
    b.previous_exit_hook = pg6_crt_set_exit_hook(&boot_exit_hook);
    os::win32::set_process_exit_handler(&boot_terminate_process_handler, nullptr);
    b.hooks_installed = true;
}

void remove_hooks() {
    BootState& b = boot();
    if (!b.hooks_installed) {
        return;
    }
    set_fault_handler(b.previous_fault_handler);
    set_dispatch_observer(b.previous_observer);
    pg6_crt_set_exit_hook(b.previous_exit_hook);
    os::win32::set_process_exit_handler(nullptr, nullptr);
    b.hooks_installed = false;
}

// ---------------------------------------------------------------------------
// PE headers, PEB, TLS
// ---------------------------------------------------------------------------

// Reads what the boot needs from a module's mapped PE headers. Missing headers are not an error.
void read_pe_headers(ModuleState& m) {
    VirtualMemoryArena& arena = *boot().arena;
    const uint64_t base = m.image.base;
    uint16_t mz = 0;
    if (!m.headers_mapped || !arena.read(base, &mz, sizeof(mz)) || mz != 0x5A4D) {
        return;
    }
    uint32_t lfanew = 0;
    uint32_t signature = 0;
    if (!read_u32(arena, base + 0x3C, lfanew) || !read_u32(arena, base + lfanew, signature) ||
        signature != 0x00004550) {
        return;
    }
    const uint64_t opt = base + lfanew + 24;
    uint16_t magic = 0;
    if (!arena.read(opt, &magic, sizeof(magic)) || magic != 0x20B) {
        return;
    }
    uint16_t subsystem = 0;
    arena.read(opt + 68, &subsystem, sizeof(subsystem));
    m.subsystem = subsystem;
    read_u64(arena, opt + 72, m.stack_reserve);
    uint32_t count = 0;
    read_u32(arena, opt + 108, count);
    if (count > kTlsDirectoryIndex && m.tls_rva == 0) {
        read_u32(arena, opt + 112 + 8 * kTlsDirectoryIndex, m.tls_rva);
        read_u32(arena, opt + 112 + 8 * kTlsDirectoryIndex + 4, m.tls_size);
    }
}

// Parses a module's TLS directory and assigns its TLS index (written to AddressOfIndex).
bool read_tls_directory(ModuleState& m, uint32_t index, std::string& error) {
    VirtualMemoryArena& arena = *boot().arena;
    m.tls = TlsInfo{};
    if (m.tls_rva == 0) {
        return true;
    }
    const std::string where = " (module " + m.name + ")";
    if (m.tls_size != 0 && m.tls_size < sizeof(win64::TlsDirectory64)) {
        error = "TLS directory too small" + where;
        return false;
    }
    win64::TlsDirectory64 dir{};
    if (!arena.read(m.image.base + m.tls_rva, &dir, sizeof(dir))) {
        error = "TLS directory not readable" + where;
        return false;
    }
    if (dir.end_address_of_raw_data < dir.start_address_of_raw_data) {
        error = "TLS template end before start" + where;
        return false;
    }
    TlsInfo& tls = m.tls;
    tls.present = true;
    tls.index = index;
    tls.template_start = dir.start_address_of_raw_data;
    tls.template_size = dir.end_address_of_raw_data - dir.start_address_of_raw_data;
    tls.zero_fill = dir.size_of_zero_fill;
    const uint32_t align_bits = (dir.characteristics >> 20) & 0xF;
    tls.alignment = std::max<uint64_t>(align_bits == 0 ? 16 : (uint64_t{1} << (align_bits - 1)), 16);
    tls.index_va = dir.address_of_index;
    if (tls.template_size > (uint64_t{256} << 20) || tls.zero_fill > (uint64_t{256} << 20)) {
        error = "TLS template implausibly large" + where;
        return false;
    }
    if (dir.address_of_callbacks != 0) {
        for (size_t i = 0; i < kMaxTlsCallbacks; ++i) {
            uint64_t callback = 0;
            if (!read_u64(arena, dir.address_of_callbacks + 8 * i, callback)) {
                error = "TLS callback array not readable" + where;
                return false;
            }
            if (callback == 0) {
                break;
            }
            tls.callbacks.push_back(callback);
        }
    }
    if (tls.index_va != 0 && !force_write(arena, tls.index_va, &index, sizeof(index))) {
        error = "TLS AddressOfIndex not writable" + where;
        return false;
    }
    return true;
}

bool create_process_block(const BootConfig& config, std::string& error) {
    BootState& b = boot();
    VirtualMemoryArena& arena = *b.arena;
    const ModuleState& exe = b.modules.front();
    const std::u16string command_line = utf8_to_utf16(config.command_line);
    const std::u16string image_path = utf8_to_utf16(exe.path.empty() ? exe.name : exe.path);
    const std::u16string current_dir = utf8_to_utf16(config.working_directory);

    // [PEB 0x1000][process parameters 0x400][strings...]
    const uint64_t strings_offset = win64::kPebSize + sizeof(win64::ProcessParameters64);
    const uint64_t strings_bytes =
        2 * (command_line.size() + image_path.size() + current_dir.size() + 3) + 4 + 64;
    const uint64_t base = alloc_block(arena, strings_offset + strings_bytes);
    if (base == 0) {
        error = "cannot allocate the process block";
        return false;
    }
    b.process_reservation = base;
    b.peb = base;
    b.process_parameters = base + win64::kPebSize;

    uint64_t cursor = base + strings_offset;
    auto put_string = [&](const std::u16string& s, win64::UnicodeString64& out) -> bool {
        const uint64_t bytes = 2 * s.size();
        if (bytes + 2 > 0xFFFF) {
            return false;
        }
        std::u16string with_nul = s;
        with_nul.push_back(u'\0');
        if (!arena.write(cursor, with_nul.data(), with_nul.size() * 2)) {
            return false;
        }
        out.length = static_cast<uint16_t>(bytes);
        out.maximum_length = static_cast<uint16_t>(bytes + 2);
        out.buffer = cursor;
        cursor += (bytes + 2 + 7) & ~uint64_t{7};
        return true;
    };

    auto params = std::make_unique<win64::ProcessParameters64>();
    std::memset(params.get(), 0, sizeof(*params));
    params->maximum_length = sizeof(win64::ProcessParameters64);
    params->length = sizeof(win64::ProcessParameters64);
    params->flags = win64::kProcessParametersNormalized;
    if (!put_string(current_dir, params->current_directory) ||
        !put_string(image_path, params->image_path_name) ||
        !put_string(command_line, params->command_line)) {
        error = "process parameter strings too long";
        return false;
    }
    const uint32_t empty_environment = 0;  // two UTF-16 NULs
    if (!arena.write(cursor, &empty_environment, sizeof(empty_environment))) {
        error = "cannot write the environment block";
        return false;
    }
    params->environment = cursor;

    auto peb = std::make_unique<win64::Peb64>();
    std::memset(peb.get(), 0, sizeof(*peb));
    peb->image_base_address = exe.image.base;
    peb->process_parameters = b.process_parameters;
    peb->process_heap = static_cast<uint64_t>(os::win32::GetProcessHeap());
    const unsigned cpus = std::thread::hardware_concurrency();
    peb->number_of_processors = cpus == 0 ? 1 : cpus;
    peb->os_major_version = 10;
    peb->os_minor_version = 0;
    peb->os_build_number = 19041;
    peb->os_platform_id = 2;  // VER_PLATFORM_WIN32_NT
    peb->image_subsystem = exe.subsystem;
    if (!arena.write(b.process_parameters, params.get(), sizeof(*params)) ||
        !arena.write(b.peb, peb.get(), sizeof(*peb))) {
        error = "cannot write the PEB";
        return false;
    }
    return true;
}

// TEB + TLS vector + one TLS block per TLS module, in one reservation:
// [TEB][vector: 8 bytes per TLS index, padded to 16][block of index 0][block of index 1]...
bool setup_thread_block(GuestThreadContext& ctx) {
    BootState& b = boot();
    VirtualMemoryArena& arena = *b.arena;
    const size_t tls_count = b.tls_modules.size();
    const uint64_t vector_offset = win64::kTebSize;
    uint64_t cursor = vector_offset + ((8 * tls_count + 15) & ~uint64_t{15});
    std::vector<uint64_t> block_offsets;
    for (const size_t mi : b.tls_modules) {
        const TlsInfo& tls = b.modules[mi].tls;
        uint64_t offset = 0;
        if (!os::align_up_checked(cursor, tls.alignment, offset)) {
            return false;
        }
        block_offsets.push_back(offset);
        cursor = offset + std::max<uint64_t>(tls.template_size + tls.zero_fill, 8);
    }
    const uint64_t base = alloc_block(arena, cursor);
    if (base == 0) {
        return false;
    }
    ctx.block_reservation = base;
    ctx.teb = base;
    if (tls_count != 0) {
        ctx.tls_vector = base + vector_offset;
        std::vector<uint64_t> vector(tls_count);
        for (size_t k = 0; k < tls_count; ++k) {
            const TlsInfo& tls = b.modules[b.tls_modules[k]].tls;
            const uint64_t block = base + block_offsets[k];
            vector[k] = block;
            if (tls.template_size != 0) {
                std::vector<uint8_t> image_template(static_cast<size_t>(tls.template_size));
                if (!arena.read(tls.template_start, image_template.data(), image_template.size()) ||
                    !arena.write(block, image_template.data(), image_template.size())) {
                    return false;
                }
            }
            // Zero fill: the block was freshly committed, hence already zero.
        }
        ctx.tls_block = vector[0];
        if (!arena.write(ctx.tls_vector, vector.data(), vector.size() * 8)) {
            return false;
        }
    }
    auto teb = std::make_unique<win64::Teb64>();
    std::memset(teb.get(), 0, sizeof(*teb));
    teb->nt_tib.stack_base = ctx.thread.stack_top;
    teb->nt_tib.stack_limit = ctx.thread.stack_limit;
    teb->nt_tib.self = ctx.teb;
    teb->unique_process = kGuestProcessId;
    teb->unique_thread = ctx.thread_id;
    teb->thread_local_storage_pointer = ctx.tls_vector;
    teb->process_environment_block = b.peb;
    teb->deallocation_stack = ctx.thread.stack_reservation;
    if (!arena.write(ctx.teb, teb.get(), sizeof(*teb))) {
        return false;
    }
    set_teb(ctx.thread.state, ctx.teb);
    ctx.thread.state.mxcsr = kGuestMxcsrDefault;
    return true;
}

// Module run order for TLS callbacks: DLLs first (in order), the executable last.
std::vector<size_t> callback_order() {
    std::vector<size_t> order;
    const size_t n = boot().modules.size();
    for (size_t i = 1; i < n; ++i) {
        order.push_back(i);
    }
    if (n != 0) {
        order.push_back(0);
    }
    return order;
}

// O16 (com_bridge.h) creates guest threads lazily for host threads that call guest code
// (CreateThread start routines, task-queue and pool callbacks). Give each one a TEB and its
// implicit TLS blocks, bind its last-error slot and run the TLS callbacks with
// DLL_THREAD_ATTACH, as the Windows loader does for a new thread. The stack stays owned by O16.
void boot_thread_init_hook(GuestThread& thread) {
    BootState& b = boot();
    if (!b.active || b.arena == nullptr || b.peb == 0) {
        return;
    }
    auto ctx = std::make_unique<GuestThreadContext>();
    ctx->thread.stack_limit = thread.stack_limit;
    ctx->thread.stack_top = thread.stack_top;
    ctx->thread_id = os::win32::GetCurrentThreadId();
    if (!setup_thread_block(*ctx)) {
        os::log("guest boot: cannot allocate a TEB for a new guest thread");
        return;
    }
    thread.state.gs_base = ctx->teb;
    thread.state.mxcsr = kGuestMxcsrDefault;
    GuestThreadContext* raw = ctx.get();
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        b.threads.push_back(std::move(ctx));
        b.thread_states[raw] = &thread.state;
    }
    activate_guest_thread(raw);
    if (b.guest_thread_started != nullptr) {
        b.guest_thread_started(raw->thread_id);
    }
    for (const size_t mi : callback_order()) {
        const ModuleState& m = b.modules[mi];
        for (const uint64_t callback : m.tls.callbacks) {
            call_guest_args(*b.arena, thread.state, callback, m.image.base,
                            uint64_t{win64::kDllThreadAttach}, uint64_t{0});
        }
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

std::string hex(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "\"0x%" PRIx64 "\"", v);
    return buf;
}

// Guest VA of a host address inside the arena's guest address space, or 0.
uint64_t host_to_guest(uint64_t host) {
    const VirtualMemoryArena* arena = boot().arena;
    if (arena == nullptr || !arena->valid()) {
        return 0;
    }
    if (os::guest_window::enabled()) {
        // A fault inside a translated-window page reports the window VA (O29).
        const uint64_t window_va =
            os::guest_window::guest_address(reinterpret_cast<const void*>(static_cast<uintptr_t>(host)));
        if (os::guest_window::in_window(window_va)) {
            return window_va;
        }
    }
    const uint64_t va = host - arena->guest_base();
    return va >= arena->space_base() && va - arena->space_base() < arena->space_size() ? va : 0;
}

std::string hex_or_null(uint64_t v) {
    return v == 0 ? std::string("null") : hex(v);
}

std::string json_string(std::string_view s) {
    std::string out = "\"";
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char esc[8];
                    std::snprintf(esc, sizeof(esc), "\\u%04x", c);
                    out += esc;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
    return out;
}

std::string json_cstr(const char* s) {
    return s != nullptr ? json_string(s) : std::string("null");
}

const char* const kGprNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                   "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};

std::string function_json(uint64_t va) {
    const uint64_t entry = nearest_entry(va);
    if (entry == 0) {
        return "null";
    }
    return "{\"entry\": " + hex(entry) + ", \"offset\": " + hex(va - entry) + "}";
}

std::string import_name_json(const BootImport& imp) {
    return "\"dll\": " + json_cstr(imp.dll) + ", \"symbol\": " + json_string(import_symbol(imp));
}

std::string tls_json(const ModuleState& m) {
    const TlsInfo& tls = m.tls;
    std::string o = "{\"present\": " + std::string(tls.present ? "true" : "false");
    if (tls.present) {
        o += ", \"index\": " + std::to_string(tls.index) + ", \"template_start\": " +
             hex(tls.template_start) + ", \"template_size\": " + std::to_string(tls.template_size) +
             ", \"zero_fill\": " + std::to_string(tls.zero_fill) + ", \"alignment\": " +
             std::to_string(tls.alignment) + ", \"index_va\": " + hex_or_null(tls.index_va) +
             ", \"callbacks\": [";
        for (size_t k = 0; k < tls.callbacks.size(); ++k) {
            o += (k == 0 ? "" : ", ") + hex(tls.callbacks[k]);
        }
        o += "], \"callbacks_run\": " + std::to_string(tls.callbacks_run);
    }
    return o + "}";
}

std::string build_report() {
    BootState& b = boot();
    const ModuleState* exe = b.modules.empty() ? nullptr : &b.modules.front();
    std::string o;
    o.reserve(16384);
    o += "{\n";
    o += "  \"schema\": \"playgta6-boot-report\",\n";
    o += "  \"schema_version\": 2,\n";
    o += "  \"status\": " + json_string(boot_status_name(b.status)) + ",\n";
    o += "  \"status_code\": " + std::to_string(static_cast<int32_t>(b.status)) + ",\n";
    o += "  \"exit_code\": " + (b.has_exit_code ? std::to_string(b.exit_code) : std::string("null")) + ",\n";
    o += "  \"phase\": " + json_string(b.phase) + ",\n";
    o += "  \"message\": " + json_string(b.message) + ",\n";
    o += "  \"milestones\": [";
    for (size_t k = 0; k < b.milestones.size(); ++k) {
        o += (k == 0 ? "" : ", ") + std::string("{\"milestone\": ") + json_string(b.milestones[k].name) +
             ", \"elapsed_ms\": " + std::to_string(b.milestones[k].elapsed_ms) +
             ", \"import_seq\": " + std::to_string(b.milestones[k].import_seq) + "}";
    }
    o += "],\n";

    const FaultRecord& f = b.fault;
    if (f.present) {
        const char* trap_kind = w40_trap_kind(f.kind);
        o += "  \"fault\": {\"kind\": " + json_string(f.kind) + ", \"trap_kind\": " + json_cstr(trap_kind) +
             ", \"va\": " + hex(f.va) + ", \"detail\": " + hex(f.detail) +
             ", \"function\": " + function_json(f.va) +
             (f.kind == "unhandled_exception" ? ", \"exception_code\": " + hex(f.detail) : std::string()) +
             ", \"import\": ";
        if (f.import_index >= 0) {
            const ImportRecord& rec = b.imports[static_cast<size_t>(f.import_index)];
            o += "{" + import_name_json(rec.import) + ", \"slot_rva\": " + hex(rec.import.slot_rva) +
                 ", \"delay_load\": " + ((rec.import.flags & kPg6BootImportDelayLoad) != 0 ? "true" : "false") +
                 ", \"module\": " + json_string(b.modules[rec.module].name) + "}";
        } else {
            o += "null";
        }
        o += ", \"host_address\": " + (f.has_host ? hex(f.host_address) : std::string("null"));
        o += ", \"guest_address\": " + (f.has_host ? hex_or_null(host_to_guest(f.host_address)) : std::string("null"));
        o += ", \"host_pc\": " + (f.has_host ? hex(f.host_pc) : std::string("null"));
        o += "},\n";
    } else {
        o += "  \"fault\": null,\n";
    }

    const GuestState* regs = f.present && f.has_registers ? &f.registers
                             : (b.main_thread != nullptr ? &b.main_thread->thread.state : nullptr);
    if (regs != nullptr) {
        o += "  \"registers\": {";
        for (unsigned i = 0; i < 16; ++i) {
            o += json_string(kGprNames[i]) + ": " + hex(regs->gpr[i]) + ", ";
        }
        o += "\"rip\": " + hex(regs->rip) + ", \"gs_base\": " + hex(regs->gs_base) +
             ", \"fs_base\": " + hex(regs->fs_base) + ", \"mxcsr\": " + hex(regs->mxcsr) + "},\n";
    } else {
        o += "  \"registers\": null,\n";
    }

    o += "  \"backtrace\": {\"method\": " + json_string(f.backtrace_method) + ", \"frames\": [";
    for (size_t i = 0; i < f.frames.size(); ++i) {
        const Frame& fr = f.frames[i];
        o += (i == 0 ? "\n    " : ",\n    ");
        o += "{\"stack_address\": " + hex(fr.stack_address) + ", \"return_address\": " +
             hex(fr.return_address) + ", \"function\": " + hex(fr.function) +
             ", \"offset\": " + hex(fr.return_address - fr.function) + "}";
    }
    o += f.frames.empty() ? "]},\n" : "\n  ]},\n";

    o += "  \"recent_dispatches\": [";
    for (size_t i = 0; i < f.recent_dispatches.size(); ++i) {
        const DispatchRecord& d = f.recent_dispatches[i];
        o += (i == 0 ? "\n    " : ",\n    ");
        const std::string label = dispatch_label(d.target);
        o += "{\"target\": " + hex(d.target) + ", \"name\": " + (label.empty() ? "null" : json_string(label)) +
             ", \"return_address\": " + hex(d.return_address) + ", \"rcx\": " + hex(d.rcx) +
             ", \"rdx\": " + hex(d.rdx) + "}";
    }
    o += f.recent_dispatches.empty() ? "],\n" : "\n  ],\n";

    // Host callables (COM methods, GetProcAddress results) called by guest code, in first-call
    // order, with the number of import calls made before the first call.
    o += "  \"host_callable_calls\": [";
    {
        std::vector<std::pair<uint64_t, BootState::CallableCalls>> rows;
        {
            std::lock_guard<std::mutex> lock(b.callable_mutex);
            rows.assign(b.callable_calls.begin(), b.callable_calls.end());
        }
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& c) { return a.second.order < c.second.order; });
        for (size_t i = 0; i < rows.size(); ++i) {
            o += (i == 0 ? "\n    " : ",\n    ");
            o += "{\"name\": " + json_cstr(rows[i].second.name) + ", \"target\": " + hex(rows[i].first) +
                 ", \"calls\": " + std::to_string(rows[i].second.calls) + ", \"first_after_import_call\": " +
                 std::to_string(rows[i].second.first_import_seq) + "}";
        }
        o += rows.empty() ? "],\n" : "\n  ],\n";
    }

    // COM-bridge slots without an implementation that guest code called (named or not): what the
    // title asked of D3D12 / xgameruntime interfaces and got a stub for.
    o += "  \"com_stub_hits\": [";
    {
        const std::vector<com::StubHit> hits = com::stub_hit_report();
        for (size_t i = 0; i < hits.size(); ++i) {
            const com::StubHit& h = hits[i];
            o += (i == 0 ? "\n    " : ",\n    ");
            o += "{\"interface\": " + json_string(h.interface_name) + ", \"iid\": " +
                 json_string(com::format_guid(h.iid)) + ", \"slot\": " + std::to_string(h.slot) +
                 ", \"method\": " + json_string(h.method_name) + ", \"hits\": " + std::to_string(h.hits) + "}";
        }
        o += hits.empty() ? "],\n" : "\n  ],\n";
    }

    if (f.has_host_frames) {
        o += "  \"host_frames\": [";
        for (size_t i = 0; i < f.host_frames.size(); ++i) {
            const HostFrame& h = f.host_frames[i];
            o += (i == 0 ? "\n    " : ",\n    ");
            o += "{\"host_pc\": " + hex(h.host_pc) + ", \"function\": " + hex(h.function) +
                 ", \"offset\": " + hex(h.offset) + "}";
        }
        o += f.host_frames.empty() ? "],\n" : "\n  ],\n";
    } else {
        o += "  \"host_frames\": null,\n";
    }

    // Counters.
    o += "  \"counters\": {\"dispatches\": " + std::to_string(b.dispatches.load()) +
         ", \"import_calls\": " + std::to_string(b.import_calls.load()) + ", \"imports\": [";
    std::vector<std::pair<uint64_t, size_t>> called;
    for (size_t i = 0; i < b.imports.size(); ++i) {
        const uint64_t n = b.import_counts ? b.import_counts[i].load() : 0;
        if (n != 0) {
            called.emplace_back(n, i);
        }
    }
    std::stable_sort(called.begin(), called.end(),
                     [](const auto& a, const auto& c) { return a.first > c.first; });
    for (size_t k = 0; k < called.size(); ++k) {
        const BootImport& imp = b.imports[called[k].second].import;
        o += (k == 0 ? "\n    " : ",\n    ");
        o += "{" + import_name_json(imp) + ", \"resolved\": " + (import_resolved(imp) ? "true" : "false") +
             ", \"count\": " + std::to_string(called[k].first) + "}";
    }
    o += called.empty() ? "], \"recent\": [" : "\n  ], \"recent\": [";
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        std::vector<RecentImport> ordered = b.recent;
        std::sort(ordered.begin(), ordered.end(),
                  [](const RecentImport& a, const RecentImport& c) { return a.seq < c.seq; });
        for (size_t k = 0; k < ordered.size(); ++k) {
            o += (k == 0 ? "\n    " : ",\n    ");
            o += "{\"seq\": " + std::to_string(ordered[k].seq) + ", " +
                 import_name_json(b.imports[ordered[k].import_index].import) + ", \"thread\": " +
                 std::to_string(ordered[k].thread) + ", \"args\": [" + hex(ordered[k].args[0]) + ", " +
                 hex(ordered[k].args[1]) + ", " + hex(ordered[k].args[2]) + ", " + hex(ordered[k].args[3]) +
                 ", " + hex(ordered[k].args[4]) + "]}";
        }
        o += ordered.empty() ? "]},\n" : "\n  ]},\n";

        o += "  \"survey\": {\"enabled\": " + std::string(b.survey ? "true" : "false") +
             ", \"unresolved_called\": [";
        for (size_t k = 0; k < b.survey_called.size(); ++k) {
            const SurveyEntry& e = b.survey_called[k];
            o += (k == 0 ? "\n    " : ",\n    ");
            o += "{" + import_name_json(b.imports[e.import_index].import) +
                 ", \"first_seq\": " + std::to_string(e.first_seq) +
                 ", \"count\": " + std::to_string(e.count) + "}";
        }
        o += b.survey_called.empty() ? "]},\n" : "\n  ]},\n";
    }

    // Timeline of first calls, threads, address space.
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        o += "  \"first_calls\": [";
        for (size_t k = 0; k < b.first_calls.size(); ++k) {
            const FirstCall& f = b.first_calls[k];
            o += (k == 0 ? "" : ", ") + std::string("{\"name\": ") + json_string(f.name) +
                 ", \"import_seq\": " + std::to_string(f.seq) + ", \"elapsed_ms\": " +
                 std::to_string(f.elapsed_ms) + ", \"thread\": " + std::to_string(f.thread) + "}";
        }
        o += "],\n  \"threads\": [";
        for (size_t k = 0; k < b.threads.size(); ++k) {
            const GuestThreadContext* t = b.threads[k].get();
            const auto live = b.thread_states.find(t);
            const GuestState& s = live != b.thread_states.end() ? *live->second : t->thread.state;
            o += (k == 0 ? "\n    " : ",\n    ");
            o += "{\"thread_id\": " + std::to_string(t->thread_id) + ", \"teb\": " + hex(t->teb) +
                 ", \"stack_base\": " + hex(t->thread.stack_top) + ", \"stack_limit\": " +
                 hex(t->thread.stack_limit) + ", \"rsp\": " + hex(s.gpr[GuestReg::Rsp]) + ", \"rip\": " +
                 hex(s.rip) + ", \"last_import\": ";
            const auto last = b.thread_last.find(t);
            if (last != b.thread_last.end()) {
                o += "{\"seq\": " + std::to_string(last->second.seq) + ", " +
                     import_name_json(b.imports[last->second.import_index].import) + "}";
            } else {
                o += "null";
            }
            FaultRecord scan;
            scan_backtrace(scan, s.gpr[GuestReg::Rsp], t);
            o += ", \"stack_scan\": [";
            for (size_t i = 0; i < scan.frames.size() && i < 8; ++i) {
                o += (i == 0 ? "" : ", ") + hex(scan.frames[i].return_address);
            }
            o += "]}";
        }
        o += b.threads.empty() ? "],\n" : "\n  ],\n";
    }
    {
        // Arena totals first: the arena observer takes arena_mutex under the arena's lock.
        const uint64_t arena_reserved = b.arena != nullptr ? b.arena->reserved_bytes() : 0;
        const uint64_t arena_committed = b.arena != nullptr ? b.arena->committed_bytes() : 0;
        std::lock_guard<std::mutex> lock(b.arena_mutex);
        static const char* const kEventNames[] = {"reserve", "commit", "decommit", "release"};
        o += "  \"address_space\": {\"events\": " + std::to_string(b.arena_events) + ", \"failures\": " +
             std::to_string(b.arena_failures) + ", \"reserved_now\": " + hex(b.reserved_now) +
             ", \"reserved_peak\": " + hex(b.reserved_peak) + ", \"commit_requests\": " +
             hex(b.commit_requests) + ", \"arena_reserved\": " +
             hex(arena_reserved) + ", \"arena_committed\": " +
             hex(arena_committed) + ", \"log_truncated\": " +
             (b.arena_events > b.arena_log.size() ? "true" : "false") + ", \"log\": [";
        for (size_t k = 0; k < b.arena_log.size(); ++k) {
            const ArenaRecord& r = b.arena_log[k];
            o += (k == 0 ? "\n    " : ",\n    ");
            o += "{\"event\": " + json_string(kEventNames[static_cast<int>(r.event)]) + ", \"ok\": " +
                 (r.ok ? "true" : "false") + ", \"va\": " + hex(r.va) + ", \"size\": " + hex(r.size) +
                 ", \"protect\": " + hex(r.protect) + ", \"import_seq\": " + std::to_string(r.import_seq) +
                 ", \"import\": " +
                 (r.import_index >= 0 && static_cast<size_t>(r.import_index) < b.imports.size()
                      ? json_string(import_label(b.imports[static_cast<size_t>(r.import_index)].import))
                      : std::string("null")) +
                 "}";
        }
        o += b.arena_log.empty() ? "]},\n" : "\n  ]},\n";
    }

    // Startup summary.
    o += "  \"startup\": {\"image_base\": " + hex_or_null(exe != nullptr ? exe->image.base : 0) +
         ", \"entry_point\": " + hex_or_null(exe != nullptr ? exe->image.entry_point : 0) +
         ", \"modules\": [";
    for (size_t k = 0; k < b.modules.size(); ++k) {
        const ModuleState& m = b.modules[k];
        o += (k == 0 ? "\n    " : ",\n    ");
        o += "{\"name\": " + json_string(m.name) + ", \"image_base\": " + hex_or_null(m.image.base) +
             ", \"image_size\": " + hex(m.image.size_of_image) + ", \"entry_point\": " +
             hex_or_null(m.image.entry_point) + ", \"rebased\": " + (m.image.rebased ? "true" : "false") +
             ", \"lifted_tables\": " + std::to_string(m.tables.size()) +
             ", \"shard_dispatch_tables\": " + std::to_string(m.shard_table_count) + ", \"functions\": " +
             std::to_string(m.functions) + ", \"imports\": {\"total\": " +
             std::to_string(m.import_count) + ", \"bound\": " + std::to_string(m.imports_bound) +
             ", \"unresolved\": " + std::to_string(m.imports_unresolved) + ", \"runtime_thunks\": " +
             std::to_string(m.imports_runtime) + "}, \"pdata\": " + (m.pdata ? "true" : "false") +
             ", \"tls\": " + tls_json(m) + "}";
    }
    o += b.modules.empty() ? "],\n" : "\n  ],\n";
    o += "    \"lifted_tables\": " + std::to_string(b.tables.size()) +
         ", \"shard_dispatch_tables\": " + std::to_string(b.shard_table_count) +
         ", \"functions_registered\": " + std::to_string(b.functions_registered) + ",\n";
    o += "    \"imports\": {\"total\": " + std::to_string(b.imports.size()) +
         ", \"bound\": " + std::to_string(b.imports_bound) +
         ", \"unresolved\": " + std::to_string(b.imports_unresolved) + ", \"by_dll\": [";
    {
        std::vector<std::string> dlls;
        std::vector<std::pair<uint64_t, uint64_t>> counts;
        for (const ImportRecord& rec : b.imports) {
            const std::string dll = rec.import.dll != nullptr ? rec.import.dll : "?";
            auto it = std::find(dlls.begin(), dlls.end(), dll);
            const auto k = static_cast<size_t>(it - dlls.begin());
            if (it == dlls.end()) {
                dlls.push_back(dll);
                counts.emplace_back(0, 0);
            }
            (import_resolved(rec.import) ? counts[k].first : counts[k].second) += 1;
        }
        for (size_t k = 0; k < dlls.size(); ++k) {
            o += (k == 0 ? "\n      " : ",\n      ");
            o += "{\"dll\": " + json_string(dlls[k]) + ", \"bound\": " +
                 std::to_string(counts[k].first) + ", \"unresolved\": " +
                 std::to_string(counts[k].second) + "}";
        }
        o += dlls.empty() ? "]},\n" : "\n    ]},\n";
    }
    o += "    \"process\": {\"peb\": " + hex_or_null(b.peb) + ", \"process_parameters\": " +
         hex_or_null(b.process_parameters) + "},\n";
    if (b.main_thread != nullptr) {
        const GuestThreadContext& t = *b.main_thread;
        o += "    \"main_thread\": {\"teb\": " + hex(t.teb) + ", \"stack_base\": " +
             hex(t.thread.stack_top) + ", \"stack_limit\": " + hex(t.thread.stack_limit) +
             ", \"tls_vector\": " + hex_or_null(t.tls_vector) + ", \"tls_block\": " +
             hex_or_null(t.tls_block) + "}},\n";
    } else {
        o += "    \"main_thread\": null},\n";
    }
    {
        // O29: the memory model, how the title maps pool pages into its CPU windows (adjacent
        // slots holding pool-view neighbours are host-contiguous under the translated window) and
        // the translated window's own slot table.
        const os::scarlett::XMemWindowContiguity c = os::scarlett::xmem_window_contiguity();
        const os::guest_window::Stats w = os::guest_window::stats();
        o += "  \"memory\": {\"model\": " + json_string(b.memory_model) +
             ", \"cpu_window\": {\"maps\": " + std::to_string(c.maps) +
             ", \"neighbour_pairs\": " + std::to_string(c.neighbour_pairs) +
             ", \"neighbour_pairs_contiguous\": " + std::to_string(c.neighbour_pairs_contiguous) +
             ", \"neighbour_pairs_reversed\": " + std::to_string(c.neighbour_pairs_reversed) +
             ", \"run_kept\": " + std::to_string(c.run_kept) + ", \"run_moves\": " + std::to_string(c.run_moves) +
             ", \"run_breaks\": " + std::to_string(c.run_breaks) +
             ", \"run_relocations\": " + std::to_string(c.run_relocations) +
             ", \"run_bytes_moved\": " + std::to_string(c.run_bytes_moved) + ", \"regions\": [";
        for (size_t k = 0; k < c.regions.size(); ++k) {
            const auto& r = c.regions[k];
            o += std::string(k == 0 ? "" : ", ") + "{\"base\": " + hex(r.base) + ", \"size\": " + hex(r.size) +
                 ", \"translated\": " + (r.translated ? "true" : "false") +
                 ", \"mapped_slots\": " + std::to_string(r.mapped_slots) +
                 ", \"aliased_slots\": " + std::to_string(r.aliased_slots) + ", \"runs\": " + std::to_string(r.runs) +
                 ", \"longest_run\": " + std::to_string(r.longest_run) +
                 ", \"adjacent_pairs\": " + std::to_string(r.adjacent_pairs) +
                 ", \"aliased_pairs\": " + std::to_string(r.aliased_pairs) +
                 ", \"contiguous_pairs\": " + std::to_string(r.contiguous_pairs) + "}";
        }
        o += std::string("]}, \"window\": {\"enabled\": ") + (os::guest_window::enabled() ? "true" : "false") +
             ", \"mapped_slots\": " + std::to_string(w.mapped_slots) +
             ", \"contiguous_pairs\": " + std::to_string(w.contiguous_pairs) +
             ", \"split_pairs\": " + std::to_string(w.split_pairs) + ", \"map_calls\": " +
             std::to_string(w.map_calls) + "}},\n";
    }
    // Schema 2: the import calls up to the first Present, per guest thread and import (null until a
    // Present was reported through mark_first_present), then the extension's members (per-Present samples).
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        if (!b.present_seen) {
            o += "  \"first_present\": null,\n";
        } else {
            std::vector<std::pair<uint64_t, uint64_t>> rows(b.pre_present_counts.begin(), b.pre_present_counts.end());
            std::sort(rows.begin(), rows.end());
            o += "  \"first_present\": {\"import_seq\": " + std::to_string(b.present_import_seq) +
                 ", \"elapsed_ms\": " + std::to_string(b.present_elapsed_ms) + ", \"imports\": [";
            for (size_t k = 0; k < rows.size(); ++k) {
                const size_t index = static_cast<size_t>(rows[k].first & 0xffffffffu);
                o += (k == 0 ? "\n    " : ",\n    ");
                o += "{\"thread\": " + std::to_string(rows[k].first >> 32) + ", ";
                if (index < b.imports.size()) {
                    o += import_name_json(b.imports[index].import);
                } else {
                    o += "\"dll\": null, \"symbol\": null";
                }
                o += ", \"count\": " + std::to_string(rows[k].second) + "}";
            }
            o += rows.empty() ? "]},\n" : "\n  ]},\n";
        }
    }
    if (const ReportExtension extension = g_report_extension.load(std::memory_order_acquire)) {
        const std::string members = extension();
        if (!members.empty()) {
            o += "  " + members + ",\n";
        }
    }
    o += "  \"elapsed_ms\": " + std::to_string(elapsed_ms()) + "\n";
    o += "}\n";
    return o;
}

std::string function_text(uint64_t va) {
    const uint64_t entry = nearest_entry(va);
    char buf[64];
    if (entry == 0) {
        std::snprintf(buf, sizeof(buf), "(no lifted function)");
    } else {
        std::snprintf(buf, sizeof(buf), "sub_%" PRIX64 "+0x%" PRIx64, entry, va - entry);
    }
    return buf;
}

std::string build_summary() {
    BootState& b = boot();
    std::string s;
    char line[512];
    std::snprintf(line, sizeof(line), "guest boot: %s (phase %s) after %" PRIu64 " ms",
                  boot_status_name(b.status), b.phase.c_str(), elapsed_ms());
    s += line;
    if (b.has_exit_code) {
        std::snprintf(line, sizeof(line), ", exit code %d", b.exit_code);
        s += line;
    }
    s += "\n";
    if (!b.message.empty()) {
        s += "  " + b.message + "\n";
    }
    for (const ModuleState& m : b.modules) {
        std::snprintf(line, sizeof(line),
                      "  module %s: image 0x%" PRIx64 " (0x%" PRIx64 " bytes), entry 0x%" PRIx64
                      ", %" PRIu64 " lifted functions, imports %zu (%" PRIu64 " bound, %" PRIu64
                      " of them to R11 runtime thunks, %" PRIu64 " unresolved), .pdata %s\n",
                      m.name.c_str(), m.image.base, m.image.size_of_image, m.image.entry_point,
                      m.functions, m.import_count, m.imports_bound, m.imports_runtime,
                      m.imports_unresolved, m.pdata ? "registered" : "absent");
        s += line;
        if (m.tls.present) {
            std::snprintf(line, sizeof(line),
                          "    TLS index %u: template 0x%" PRIx64 " (%" PRIu64 " bytes + %" PRIu64
                          " zero-fill, align %" PRIu64 "), %zu callback(s), %" PRIu64 " ran\n",
                          m.tls.index, m.tls.template_start, m.tls.template_size, m.tls.zero_fill,
                          m.tls.alignment, m.tls.callbacks.size(), m.tls.callbacks_run);
            s += line;
        }
    }
    std::snprintf(line, sizeof(line),
                  "  entry point %s; dispatched calls %" PRIu64 ", import calls %" PRIu64 "\n",
                  b.entry_started ? "started" : "not started", b.dispatches.load(),
                  b.import_calls.load());
    s += line;
    const FaultRecord& f = b.fault;
    if (f.present) {
        std::snprintf(line, sizeof(line), "  fault %s at 0x%016" PRIx64 " in %s", f.kind.c_str(), f.va,
                      function_text(f.va).c_str());
        s += line;
        if (f.import_index >= 0) {
            s += ": " + import_label(b.imports[static_cast<size_t>(f.import_index)].import);
        }
        if (f.has_host) {
            std::snprintf(line, sizeof(line), " (host address 0x%" PRIx64 ", host pc 0x%" PRIx64 ")",
                          f.host_address, f.host_pc);
            s += line;
        }
        s += "\n";
        if (f.has_registers) {
            for (unsigned i = 0; i < 16; ++i) {
                std::snprintf(line, sizeof(line), "%s%-3s=%016" PRIx64, (i % 4 == 0) ? "    " : " ",
                              kGprNames[i], f.registers.gpr[i]);
                s += line;
                if (i % 4 == 3) {
                    s += "\n";
                }
            }
            std::snprintf(line, sizeof(line), "    rip=%016" PRIx64 " gs_base=%016" PRIx64 "\n",
                          f.registers.rip, f.registers.gs_base);
            s += line;
        }
        s += "  backtrace (" + f.backtrace_method + "):\n";
        for (const Frame& fr : f.frames) {
            std::snprintf(line, sizeof(line), "    [rsp 0x%" PRIx64 "] 0x%016" PRIx64 " %s\n",
                          fr.stack_address, fr.return_address, function_text(fr.return_address).c_str());
            s += line;
        }
        if (f.has_host_frames) {
            s += "  host frames (lifted functions, innermost first):\n";
            for (const HostFrame& h : f.host_frames) {
                std::snprintf(line, sizeof(line), "    sub_%" PRIX64 " (+0x%" PRIx64 " host)\n", h.function,
                              h.offset);
                s += line;
            }
        }
        if (!f.recent_dispatches.empty()) {
            s += "  last dispatched calls on the faulting thread (oldest first):\n";
            for (const DispatchRecord& d : f.recent_dispatches) {
                const std::string label = dispatch_label(d.target);
                std::snprintf(line, sizeof(line),
                              "    -> 0x%016" PRIx64 "%s%s from 0x%016" PRIx64 " %s rcx=%" PRIx64 " rdx=%" PRIx64 "\n",
                              d.target, label.empty() ? "" : " ", label.c_str(), d.return_address,
                              function_text(d.return_address).c_str(), d.rcx, d.rdx);
                s += line;
            }
        }
    }
    {
        const std::vector<com::StubHit> hits = com::stub_hit_report();
        if (!hits.empty()) {
            s += "  COM stub slots called (interface, slot, method, calls):\n";
            for (const com::StubHit& h : hits) {
                s += "    " + h.interface_name + " slot " + std::to_string(h.slot) +
                     (h.method_name.empty() ? std::string() : " (" + h.method_name + ")") + " x" +
                     std::to_string(h.hits) + "\n";
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        if (!b.recent.empty()) {
            std::vector<RecentImport> ordered = b.recent;
            std::sort(ordered.begin(), ordered.end(),
                      [](const RecentImport& a, const RecentImport& c) { return a.seq < c.seq; });
            s += "  last imports called:\n";
            for (const RecentImport& r : ordered) {
                std::snprintf(line, sizeof(line), "    #%" PRIu64 " %s\n", r.seq,
                              import_label(b.imports[r.import_index].import).c_str());
                s += line;
            }
        }
        if (b.survey) {
            std::snprintf(line, sizeof(line), "  survey: %zu unresolved import(s) called, in order:\n",
                          b.survey_called.size());
            s += line;
            for (const SurveyEntry& e : b.survey_called) {
                std::snprintf(line, sizeof(line), "    %s x%" PRIu64 "\n",
                              import_label(b.imports[e.import_index].import).c_str(), e.count);
                s += line;
            }
        }
        if (!b.first_calls.empty()) {
            s += "  first calls:";
            for (const FirstCall& f : b.first_calls) {
                std::snprintf(line, sizeof(line), " %s@#%" PRIu64, f.name.c_str(), f.seq);
                s += line;
            }
            s += "\n";
        }
        std::snprintf(line, sizeof(line), "  guest threads: %zu\n", b.threads.size());
        s += line;
    }
    {
        std::lock_guard<std::mutex> lock(b.arena_mutex);
        std::snprintf(line, sizeof(line),
                      "  address space: %" PRIu64 " events (%" PRIu64 " failed), reserved now 0x%" PRIx64
                      " peak 0x%" PRIx64 ", commit requests 0x%" PRIx64 "\n",
                      b.arena_events, b.arena_failures, b.reserved_now, b.reserved_peak, b.commit_requests);
        s += line;
    }
    return s;
}

// Logs the stopping fault in W40's `[trap]` format.
void log_trap() {
    BootState& b = boot();
    const FaultRecord& f = b.fault;
    if (!f.present) {
        return;
    }
    const char* trap_kind = w40_trap_kind(f.kind);
    std::string text = std::string(trap_kind != nullptr ? "[trap] " : "[boot] ") +
                       (trap_kind != nullptr ? trap_kind : "run bound") + " at 0x";
    char va[24];
    std::snprintf(va, sizeof(va), "%" PRIx64, f.va);
    text += va;
    if (f.import_index >= 0) {
        text += " " + import_label(b.imports[static_cast<size_t>(f.import_index)].import);
    }
    text += " (" + f.kind + ", " + function_text(f.va) + ")";
    logf("%s", text.c_str());
}

void finish_report() {
    BootState& b = boot();
    b.last_report = build_report();
    b.last_summary = build_summary();
}

BootStats collect_stats() {
    BootState& b = boot();
    BootStats st{};
    st.status = static_cast<int32_t>(b.status);
    st.exit_code = b.exit_code;
    if (!b.modules.empty()) {
        st.image_base = b.modules.front().image.base;
        st.entry_point = b.modules.front().image.entry_point;
    }
    st.functions_registered = b.functions_registered;
    st.imports_total = b.imports.size();
    st.imports_bound = b.imports_bound;
    st.imports_unresolved = b.imports_unresolved;
    for (const ModuleState& m : b.modules) {
        st.tls_callbacks += m.tls.callbacks.size();
    }
    st.tls_callbacks_run = b.tls_callbacks_run;
    st.dispatches = b.dispatches.load();
    st.import_calls = b.import_calls.load();
    st.unresolved_called = b.survey_called.size();
    st.entry_started = b.entry_started ? 1 : 0;
    return st;
}

void fail_setup(BootStatus status, const std::string& message) {
    BootState& b = boot();
    b.status = status;
    b.message = message;
    logf("guest boot: %s failed: %s", b.phase.c_str(), message.c_str());
}

// call_guest returns early while an R11 context restore propagates (GuestState::unwind_pending);
// one still pending when a top-level boot call returns has no owner: stop with a report.
void check_unowned_restore(GuestState& state) {
    if (state.unwind_pending == 0) {
        return;
    }
    BootState& b = boot();
    capture_fault("unowned_restore", &state, state.resume_rip, state.gpr[GuestReg::Rsp]);
    char message[160];
    std::snprintf(message, sizeof(message),
                  "a context restore to 0x%016" PRIx64 " was still pending when the guest call returned",
                  state.resume_rip);
    b.message = message;
    b.status = BootStatus::Fault;
    stop_boot();
}

// Runs the guest part (TLS callbacks + entry). Returns normally when the entry returned.
void run_guest(const BootConfig& config) {
    BootState& b = boot();
    GuestThreadContext& main = *b.main_thread;
    VirtualMemoryArena& arena = *b.arena;
    for (const size_t mi : callback_order()) {
        ModuleState& m = b.modules[mi];
        if (config.run_tls_callbacks) {
            b.phase = "tls_callback";
            for (const uint64_t callback : m.tls.callbacks) {
                logf("guest boot: TLS callback 0x%016" PRIx64 " of %s (DLL_PROCESS_ATTACH)", callback,
                     m.name.c_str());
                call_guest_args(arena, main.thread.state, callback, m.image.base,
                                uint64_t{win64::kDllProcessAttach}, uint64_t{0});
                check_unowned_restore(main.thread.state);
                ++m.tls.callbacks_run;
                ++b.tls_callbacks_run;
            }
        }
        if (config.call_dll_entry_points && mi != 0 && m.image.entry_point != 0) {
            b.phase = "dll_main";
            logf("guest boot: DllMain 0x%016" PRIx64 " of %s (DLL_PROCESS_ATTACH)",
                 m.image.entry_point, m.name.c_str());
            const uint64_t ok = call_guest_args(arena, main.thread.state, m.image.entry_point,
                                                m.image.base, uint64_t{win64::kDllProcessAttach},
                                                uint64_t{0});
            if (static_cast<uint32_t>(ok) == 0) {
                fail_setup(BootStatus::ProcessSetupFailed, "DllMain of " + m.name + " returned FALSE");
                return;
            }
        }
    }
    const ModuleState& exe = b.modules.front();
    if (!config.call_entry) {
        b.status = BootStatus::Exited;
        b.message = "entry point not called (call_entry = false)";
        return;
    }
    if (exe.image.entry_point == 0) {
        fail_setup(BootStatus::BadConfig, "image has no entry point");
        return;
    }
    b.phase = "entry";
    b.entry_started = true;
    milestone("ENTRY_CALL", "calling entry point 0x%016" PRIx64, exe.image.entry_point);
    const uint64_t rax = call_guest_args(arena, main.thread.state, exe.image.entry_point, b.peb);
    check_unowned_restore(main.thread.state);
    // Returning from the entry point exits the process: run the guest exit handlers registered
    // so far (MSVC's startup code calls exit itself and never gets here).
    crt_exit::run_exit_cleanup(main.thread.state, crt_exit::ExitMode::Full);
    b.status = BootStatus::Exited;
    b.exit_code = static_cast<int32_t>(static_cast<uint32_t>(rax));
    b.has_exit_code = true;
    b.message = "entry point returned";
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const char* boot_status_name(BootStatus status) {
    switch (status) {
        case BootStatus::Exited:
            return "exited";
        case BootStatus::BadConfig:
            return "bad_config";
        case BootStatus::LoadFailed:
            return "load_failed";
        case BootStatus::RegisterFailed:
            return "register_failed";
        case BootStatus::ProcessSetupFailed:
            return "process_setup_failed";
        case BootStatus::Fault:
            return "fault";
        case BootStatus::GuestAbort:
            return "guest_abort";
        case BootStatus::LimitReached:
            return "limit_reached";
    }
    return "unknown";
}

std::string guest_thread_last_import(uint32_t thread_id) {
    BootState& b = boot();
    std::lock_guard<std::mutex> lock(b.mutex);
    for (const auto& [thread, entry] : b.thread_last) {
        if (thread->thread_id == thread_id) {
            return import_label(b.imports[entry.import_index].import);
        }
    }
    return {};
}

uint64_t guest_function_for_host_pc(uint64_t host_pc, uint64_t* offset) {
    BootState& b = boot();
    build_host_map();
    const auto& map = b.host_map;
    auto it = std::upper_bound(map.begin(), map.end(), host_pc,
                               [](uint64_t pc, const std::pair<uint64_t, uint64_t>& e) {
                                   return pc < e.first;
                               });
    if (it == map.begin()) {
        return 0;
    }
    --it;
    if (b.host_function_start != nullptr) {
        // Exact: the host function containing the PC must be a lifted function itself.
        const uint64_t start = b.host_function_start(host_pc);
        if (start == 0 || start != it->first) {
            return 0;
        }
    } else if (host_pc - it->first >= (uint64_t{1} << 20)) {
        // Heuristic: lifted functions are far smaller than 1 MiB of host code.
        return 0;
    }
    if (offset != nullptr) {
        *offset = host_pc - it->first;
    }
    return it->second;
}

void release_guest_thread(GuestThreadContext& context) {
    BootState& b = boot();
    if (t_active == &context) {
        activate_guest_thread(nullptr);
    }
    if (b.arena != nullptr) {
        if (context.block_reservation != 0) {
            b.arena->release(context.block_reservation);
        }
        destroy_guest_thread(*b.arena, context.thread);
    }
    context = GuestThreadContext{};
}

bool prepare_guest_thread(uint64_t stack_size, uint32_t thread_id, GuestThreadContext& out) {
    BootState& b = boot();
    out = GuestThreadContext{};
    if (!b.active || b.arena == nullptr || b.peb == 0) {
        return false;
    }
    if (!create_guest_thread(*b.arena, stack_size == 0 ? kDefaultStackSize : stack_size, out.thread)) {
        return false;
    }
    out.thread_id = thread_id != 0 ? thread_id : os::win32::GetCurrentThreadId();
    if (!setup_thread_block(out)) {
        release_guest_thread(out);
        return false;
    }
    return true;
}

GuestThreadContext* activate_guest_thread(GuestThreadContext* context) {
    GuestThreadContext* previous = t_active;
    t_active = context;
    uint32_t* slot = nullptr;
    BootState& b = boot();
    if (context != nullptr && b.arena != nullptr) {
        slot = reinterpret_cast<uint32_t*>(b.arena->host_ptr(
            context->teb + offsetof(win64::Teb64, last_error_value), sizeof(uint32_t)));
    }
    os::win32::SetThreadLastErrorSlot(slot);
    // GetCurrentThreadStackLimits reports the guest stack, like TEB StackLimit/StackBase.
    os::win32::set_current_thread_stack_limits(context != nullptr ? context->thread.stack_limit : 0,
                                               context != nullptr ? context->thread.stack_top : 0);
    return previous;
}

size_t run_tls_callbacks(GuestThreadContext& context, uint32_t reason) {
    BootState& b = boot();
    size_t ran = 0;
    if (b.arena == nullptr) {
        return 0;
    }
    for (const size_t mi : callback_order()) {
        const ModuleState& m = b.modules[mi];
        for (const uint64_t callback : m.tls.callbacks) {
            call_guest_args(*b.arena, context.thread.state, callback, m.image.base, uint64_t{reason},
                            uint64_t{0});
            ++ran;
        }
    }
    return ran;
}

bool request_exit(int32_t exit_code) {
    BootState& b = boot();
    if (!b.running || !t_running_boot) {
        return false;
    }
    b.status = BootStatus::Exited;
    b.exit_code = exit_code;
    b.has_exit_code = true;
    b.message = "guest requested process exit (" + std::to_string(exit_code) + ")";
    stop_boot();
}

bool report_host_fault(const char* kind, uint64_t host_address, uint64_t host_pc) {
    BootState& b = boot();
    if (!b.active) {
        return false;
    }
    const char* name = kind != nullptr ? kind : "host_fault";
    capture_fault(name, nullptr, 0, 0);
    FaultRecord& f = b.fault;
    f.has_host = true;
    f.host_address = host_address;
    f.host_pc = host_pc;
    uint64_t offset = 0;
    const uint64_t fn = guest_function_for_host_pc(host_pc, &offset);
    f.va = fn;  // the containing lifted function's entry (the exact guest VA is not known)
    f.detail = offset;
    b.status = BootStatus::Fault;
    char function[32] = "unknown";
    if (fn != 0) {
        std::snprintf(function, sizeof(function), "sub_%" PRIX64, fn);
    }
    char where[64] = "outside the guest address space";
    if (const uint64_t va = host_to_guest(host_address); va != 0) {
        std::snprintf(where, sizeof(where), "guest VA 0x%" PRIx64, va);
    }
    char message[320];
    std::snprintf(message, sizeof(message),
                  "host %s at host address 0x%" PRIx64 " = %s (host pc 0x%" PRIx64
                  ", lifted function %s)",
                  name, host_address, where, host_pc, function);
    b.message = message;
    log_trap();
    finish_report();
    return true;
}

std::string last_report_json() {
    return boot().last_report;
}

void set_report_extension(ReportExtension extension) {
    g_report_extension.store(extension, std::memory_order_release);
}

void mark_first_present() {
    BootState& b = boot();
    std::lock_guard<std::mutex> lock(b.mutex);
    if (!b.present_seen) {
        b.present_seen = true;
        b.present_import_seq = b.import_calls.load(std::memory_order_relaxed);
        b.present_elapsed_ms = elapsed_ms();
    }
}

std::string last_summary() {
    return boot().last_summary;
}

LiveCounters live_counters() {
    BootState& b = boot();
    LiveCounters c;
    c.dispatches = b.dispatches.load(std::memory_order_relaxed);
    c.import_calls = b.import_calls.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(b.arena_mutex);
        c.arena_events = b.arena_events;
    }
    return c;
}

std::string import_trace_text() {
    BootState& b = boot();
    std::lock_guard<std::mutex> lock(b.mutex);
    std::string out;
    out.reserve(b.trace.size() * 96);
    char line[512];
    for (const RecentImport& r : b.trace) {
        std::snprintf(line, sizeof(line),
                      "%" PRIu64 " %u %s %" PRIx64 " %" PRIx64 " %" PRIx64 " %" PRIx64 " %" PRIx64 "\n",
                      r.seq, r.thread, import_label(b.imports[r.import_index].import).c_str(), r.args[0],
                      r.args[1], r.args[2], r.args[3], r.args[4]);
        out += line;
        const size_t k = static_cast<size_t>(&r - b.trace.data());
        if (k < b.trace_strings.size() && !b.trace_strings[k].empty()) {
            out.pop_back();  // the newline
            out += b.trace_strings[k];
            out += '\n';
        }
    }
    if (b.trace_truncated) {
        out += "# truncated at the trace limit\n";
    }
    return out;
}

void reset_boot() {
    BootState& b = boot();
    remove_hooks();
    if (b.arena_observer_installed) {
        os::set_arena_observer(b.previous_arena_observer);
        b.arena_observer_installed = false;
    }
    if (b.thread_hook_installed) {
        com::set_guest_thread_init_hook(nullptr);
        b.thread_hook_installed = false;
    }
    if (t_active != nullptr) {
        activate_guest_thread(nullptr);
    }
    if (b.arena != nullptr) {
        for (auto& thread : b.threads) {
            if (thread->block_reservation != 0) {
                b.arena->release(thread->block_reservation);
            }
            destroy_guest_thread(*b.arena, thread->thread);
        }
        if (b.process_reservation != 0) {
            b.arena->release(b.process_reservation);
        }
        for (const ModuleState& m : b.modules) {
            if (m.loaded) {
                os::pgim::unload_image(*b.arena, m.image);
            }
        }
    }
    release_memory_model();
    reset_guest_runtime();
    crt_exit::reset_crt_exit();
    reset_guest_exceptions();
    com::set_guest_memory(nullptr);
    b.active = false;
    b.running = false;
    b.arena = nullptr;
    b.modules.clear();
    b.exec.clear();
    b.tables.clear();
    b.shard_table_count = 0;
    b.functions_registered = 0;
    b.host_map.clear();
    b.host_map_built = false;
    b.imports.clear();
    b.by_slot.clear();
    b.import_counts.reset();
    b.imports_bound = 0;
    b.imports_unresolved = 0;
    b.process_reservation = 0;
    b.peb = 0;
    b.process_parameters = 0;
    b.tls_modules.clear();
    b.threads.clear();
    b.main_thread = nullptr;
    b.dispatches.store(0);
    b.import_calls.store(0);
    b.recent.clear();
    b.recent_next = 0;
    b.survey_called.clear();
    b.tls_callbacks_run = 0;
    b.entry_started = false;
    b.milestones.clear();
    b.first_calls.clear();
    b.pre_present_counts.clear();
    b.present_seen = false;
    b.present_import_seq = 0;
    b.present_elapsed_ms = 0;
    b.trace.clear();
    b.trace_strings.clear();
    {
        std::lock_guard<std::mutex> lock(b.callable_mutex);
        b.callable_calls.clear();
    }
    b.trace_limit = 0;
    b.trace_truncated = false;
    b.thread_last.clear();
    b.thread_states.clear();
    t_last_import = -1;
    t_last_import_seq = 0;
    t_dispatch_count = 0;
    {
        std::lock_guard<std::mutex> lock(b.arena_mutex);
        b.arena_log.clear();
        b.arena_events = 0;
        b.arena_failures = 0;
        b.live_reservations.clear();
        b.reserved_now = 0;
        b.reserved_peak = 0;
        b.commit_requests = 0;
    }
    b.phase = "config";
    b.status = BootStatus::BadConfig;
    b.exit_code = 0;
    b.has_exit_code = false;
    b.message.clear();
    b.fault = FaultRecord{};
}

BootResult run_boot(const BootConfig& config) {
    reset_boot();
    BootState& b = boot();
    b.start = std::chrono::steady_clock::now();
    b.log_progress = config.log_progress;
    b.survey = config.survey;
    b.survey_return = config.survey_return;
    b.max_dispatches = config.max_dispatches;
    b.max_wall_ms = config.max_wall_ms;
    b.recent_capacity = config.recent_import_count;
    b.max_frames = config.max_backtrace_frames;
    b.capture_host_stack = config.capture_host_stack;
    b.trace_limit = config.import_trace_limit;
    b.previous_arena_observer = os::set_arena_observer(&boot_arena_observer);
    b.arena_observer_installed = true;
    b.host_function_start = config.host_function_start;
    b.off_thread_fault = config.off_thread_fault;
    b.guest_thread_started = config.guest_thread_started;
    {
        std::lock_guard<std::mutex> lock(b.arena_mutex);
        b.arena_log_limit = config.arena_log_limit;
    }
    b.off_thread_fault_taken = false;
    b.arena = config.arena != nullptr ? config.arena : os::win32::get_vmem_arena();
    // One arena everywhere: O16 bridge objects and lazily created guest threads live in it too.
    com::set_guest_memory(config.arena);
    // Guest-visible process state: O7 export registry (incl. the virtual xgameruntime.dll provider,
    // idempotent) and the command line GetCommandLineA/W report.
    os::win32::register_all_shim_exports();
    os::win32::set_command_line(config.command_line.empty() ? "game.exe" : config.command_line);
    if (config.console_mode != 0) {
        // Console-mode value the title reads at startup to pick its console type.
        constexpr uint32_t kRegDword = 4;
        const uint32_t mode = config.console_mode;
        os::win32::registry_set_value("HKLM\\Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM",
                                      u"ConsoleMode", kRegDword, &mode, sizeof(mode));
    }
    os::gdk::provider::set_xbox_live_sandbox_id(config.sandbox_id);
    os::gdk::provider::set_user_age_group(config.user_age_group);
    b.active = true;

    auto finish = [&]() {
        BootResult result;
        result.status = b.status;
        finish_report();
        result.stats = collect_stats();
        result.report_json = b.last_report;
        result.summary = b.last_summary;
        return result;
    };

    b.phase = "config";
    if (config.modules.empty()) {
        fail_setup(BootStatus::BadConfig, "no modules");
        return finish();
    }
    if (std::string error; !apply_memory_model(config.memory, *b.arena, error)) {
        fail_setup(BootStatus::BadConfig, error);
        return finish();
    }
    b.memory_model = describe_memory_model(config.memory);
    if (!is_default_memory_model(config.memory)) {
        logf("guest boot: memory model %s", b.memory_model.c_str());
    }
    for (size_t i = 0; i < config.modules.size(); ++i) {
        const BootModule& in = config.modules[i];
        if (in.image == nullptr) {
            fail_setup(BootStatus::BadConfig, "module " + std::to_string(i) + " has no image reader");
            return finish();
        }
        ModuleState m;
        m.name = !in.name.empty() ? in.name : (i == 0 ? "game.exe" : "module" + std::to_string(i) + ".dll");
        m.path = in.path;
        m.tls_rva = in.tls_directory_rva;
        m.tls_size = in.tls_directory_size;
        b.modules.push_back(std::move(m));
    }

    // 1. Images.
    b.phase = "load";
    for (size_t i = 0; i < config.modules.size(); ++i) {
        ModuleState& m = b.modules[i];
        os::pgim::ImageReader& reader = *config.modules[i].image;
        os::pgim::ImageLayout layout;
        os::pgim::ImageError err = os::pgim::parse_layout(reader, layout);
        if (!err.ok()) {
            fail_setup(BootStatus::LoadFailed, "invalid PGIM container for " + m.name + ": " +
                                                   os::pgim::error_name(err.code) + " " + err.detail);
            return finish();
        }
        const os::pgim::ModuleRegistration registration{m.name, m.path};
        os::pgim::LoadOptions load_options;
        load_options.register_main_module = i == 0 ? &registration : nullptr;
        err = os::pgim::load_image(reader, *b.arena, load_options, m.image);
        if (!err.ok()) {
            fail_setup(BootStatus::LoadFailed, "cannot load the PGIM image of " + m.name + ": " +
                                                   os::pgim::error_name(err.code) + " " + err.detail);
            return finish();
        }
        m.loaded = true;
        m.headers_mapped = layout.header.headers_mapped();
        for (const os::pgim::SectionEntry& s : layout.sections) {
            if ((s.protection & os::pgim::kProtExecute) != 0 && !s.is_headers()) {
                b.exec.push_back(ExecRange{m.image.base + s.rva, m.image.base + s.end(),
                                           !s.bytes_omitted() && !layout.header.executable_bytes_omitted()});
            }
        }
        milestone("PGIM_LOAD", "%s loaded at 0x%016" PRIx64 " (0x%" PRIx64 " bytes, %zu sections%s)",
                  m.name.c_str(), m.image.base, m.image.size_of_image, m.image.sections.size(),
                  m.image.rebased ? ", rebased" : "");
    }

    // 2. Lifted code and imports.
    b.phase = "register";
    for (size_t i = 0; i < config.modules.size(); ++i) {
        ModuleState& m = b.modules[i];
        for (const auto& table : config.modules[i].lifted_tables) {
            if (!register_lifted_functions(table)) {
                fail_setup(BootStatus::RegisterFailed, "lifted function table of " + m.name +
                                                           " rejected (unsorted, duplicate or "
                                                           "overlapping entries)");
                return finish();
            }
            m.tables.push_back(table);
            m.functions += table.size();
            b.tables.push_back(table);
        }
        for (const Pg6ShardDispatchTable* shard_table : config.modules[i].shard_dispatch_tables) {
            if (shard_table == nullptr) {
                fail_setup(BootStatus::RegisterFailed,
                           "shard dispatch table of " + m.name + " rejected: null table pointer");
                return finish();
            }
            std::string shard_error;
            if (!register_shard_dispatch_table(*shard_table, &shard_error)) {
                fail_setup(BootStatus::RegisterFailed,
                           "shard dispatch table of " + m.name + " rejected: " + shard_error);
                return finish();
            }
            ++m.shard_table_count;
            m.functions += shard_table->entry_count;
            ++b.shard_table_count;
        }
    }
    b.functions_registered = lifted_function_count();
    b.phase = "bind";
    register_exception_exports();
    for (size_t i = 0; i < config.modules.size(); ++i) {
        ModuleState& m = b.modules[i];
        m.import_count = config.modules[i].imports.size();
        for (const BootImport& imp : config.modules[i].imports) {
            ImportRecord rec{imp, i, m.image.base};
            // R11 implements the exception APIs (RtlVirtualUnwind, RaiseException, ...) as runtime
            // thunks rather than C-ABI shims, so the import bridge leaves them unresolved.
            // (also when the bridge bound a shim: the guest-state thunks own these APIs)
            if (imp.dll != nullptr && imp.symbol != nullptr) {
                if (HostThunkFn thunk = exception_thunk_for(imp.dll, imp.symbol)) {
                    rec.import.thunk = thunk;
                    rec.import.flags |= kPg6BootImportBound;
                    rec.import.shim = "guest_exceptions";
                    ++m.imports_runtime;
                }
            }
            b.imports.push_back(rec);
        }
    }
    b.by_slot.resize(b.imports.size());
    for (size_t i = 0; i < b.by_slot.size(); ++i) {
        b.by_slot[i] = i;
    }
    auto slot_key = [&](size_t i) { return std::make_pair(b.imports[i].image_base, b.imports[i].import.slot_rva); };
    std::sort(b.by_slot.begin(), b.by_slot.end(), [&](size_t x, size_t y) { return slot_key(x) < slot_key(y); });
    for (size_t i = 1; i < b.by_slot.size(); ++i) {
        if (slot_key(b.by_slot[i - 1]) == slot_key(b.by_slot[i])) {
            fail_setup(BootStatus::BadConfig, "two imports share one IAT slot");
            return finish();
        }
    }
    b.import_counts = std::make_unique<std::atomic<uint64_t>[]>(b.imports.size() + 1);
    for (size_t i = 0; i < config.modules.size(); ++i) {
        ModuleState& m = b.modules[i];
        std::vector<ImportSlotBinding> bindings;
        bindings.reserve(m.import_count);
        for (const ImportRecord& rec : b.imports) {
            if (rec.module != i) {
                continue;
            }
            const BootImport& imp = rec.import;
            const bool resolved = import_resolved(imp);
            bindings.push_back(ImportSlotBinding{imp.slot_rva, resolved ? imp.thunk : &boot_unresolved_thunk});
            (resolved ? m.imports_bound : m.imports_unresolved) += 1;
        }
        b.imports_bound += m.imports_bound;
        b.imports_unresolved += m.imports_unresolved;
        const size_t bound = bind_import_slots(*b.arena, m.image.base, bindings);
        if (bound != bindings.size()) {
            fail_setup(BootStatus::RegisterFailed,
                       "only " + std::to_string(bound) + " of " + std::to_string(bindings.size()) +
                           " IAT slots of " + m.name + " could be bound");
            return finish();
        }
    }
    logf("guest boot: %" PRIu64 " lifted functions in %zu table(s) and %zu shard dispatch table(s) (%zu shard(s)); "
         "%zu imports, %" PRIu64 " bound to shims, %" PRIu64 " without a shim%s",
         b.functions_registered, b.tables.size(), b.shard_table_count, shard_dispatch_shard_count(),
         b.imports.size(), b.imports_bound, b.imports_unresolved,
         b.survey ? " [survey mode]" : "");
    if (!config.export_thunks.empty()) {
        const ExportThunkStats exports = register_export_thunks(config.export_thunks);
        logf("guest boot: %zu export thunks: %zu new guest-callable entries, %zu sharing an existing "
             "entry, %zu without one, %zu not in the module registry%s",
             exports.rows, exports.registered, exports.shared, exports.unavailable,
             exports.unregistered, exports.failed != 0 ? " (synthetic address table full)" : "");
    }
    if (config.after_load) {
        b.phase = "after_load";
        std::string after_load_error;
        if (!config.after_load(*b.arena, after_load_error)) {
            fail_setup(BootStatus::RegisterFailed, after_load_error);
            return finish();
        }
    }
    // R11 unwinder: every image's .pdata (RtlLookupFunctionEntry, exception dispatch, backtraces),
    // unless after_load (R12's register_program_modules) registered it already.
    for (ModuleState& m : b.modules) {
        m.pdata = pc_to_image_base(m.image.base) != 0 || register_image_from_headers(*b.arena, m.image.base);
    }
    // XMem allocation hooks: the platform picks up a module's exported `__XMemAllocationHooks__`
    // structure (the first module, in load order, that exports one).
    os::scarlett::xmem_set_guest_hook_table(0);
    for (const ModuleState& m : b.modules) {
        const os::win32::Handle module = os::win32::find_module_by_address(m.image.base);
        const os::win32::ExportLookup hooks =
            module != 0 ? os::win32::LookupExport(module, "__XMemAllocationHooks__") : os::win32::ExportLookup{};
        if (hooks.guest && hooks.address != 0) {
            os::scarlett::xmem_set_guest_hook_table(hooks.address);
            logf("guest boot: XMem allocation hooks from %s at 0x%016" PRIx64, m.name.c_str(), hooks.address);
            break;
        }
    }

    // 3. Process block, TLS directories, main thread.
    b.phase = "process";
    std::string error;
    for (ModuleState& m : b.modules) {
        read_pe_headers(m);
    }
    if (!create_process_block(config, error)) {
        fail_setup(BootStatus::ProcessSetupFailed, error);
        return finish();
    }
    for (size_t i = 0; i < b.modules.size(); ++i) {
        if (!read_tls_directory(b.modules[i], static_cast<uint32_t>(b.tls_modules.size()), error)) {
            fail_setup(BootStatus::ProcessSetupFailed, error);
            return finish();
        }
        if (b.modules[i].tls.present) {
            b.tls_modules.push_back(i);
        }
    }
    b.phase = "main_thread";
    uint64_t stack_size = config.stack_size;
    if (stack_size == 0) {
        stack_size = b.modules.front().stack_reserve != 0 ? b.modules.front().stack_reserve : kDefaultStackSize;
    }
    auto main = std::make_unique<GuestThreadContext>();
    if (!prepare_guest_thread(stack_size, 0, *main)) {
        fail_setup(BootStatus::ProcessSetupFailed, "cannot create the main guest thread");
        return finish();
    }
    b.main_thread = main.get();
    b.threads.push_back(std::move(main));
    activate_guest_thread(b.main_thread);
    if (b.guest_thread_started != nullptr) {
        b.guest_thread_started(b.main_thread->thread_id);
    }
    com::set_guest_thread_init_hook(&boot_thread_init_hook);
    b.thread_hook_installed = true;
    size_t callbacks = 0;
    for (const ModuleState& m : b.modules) {
        callbacks += m.tls.callbacks.size();
    }
    milestone("TLS_INIT", "tls ready: %zu module block(s), %zu callback(s); TEB 0x%016" PRIx64
              ", stack 0x%" PRIx64 "..0x%" PRIx64 ", PEB 0x%016" PRIx64,
              b.tls_modules.size(), callbacks, b.main_thread->teb, b.main_thread->thread.stack_limit,
              b.main_thread->thread.stack_top, b.peb);

    // 4. Guest code.
    install_hooks();
    b.running = true;
    t_running_boot = true;
#if PG6_BOOT_HAS_EXCEPTIONS
    try {
        run_guest(config);
    } catch (const BootStop&) {
        // Outcome already recorded.
    }
#else
    run_guest(config);
#endif
    t_running_boot = false;
    b.running = false;
    remove_hooks();
    log_trap();
    BootResult result = finish();
    if (b.log_progress) {
        os::log(result.summary);  // not through logf: the summary is longer than its buffer
    }
    return result;
}

}  // namespace playgta6::guest

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

namespace guest = playgta6::guest;

namespace {

size_t copy_out(const std::string& text, char* buffer, size_t size) {
    if (buffer != nullptr && size != 0) {
        const size_t n = std::min(size - 1, text.size());
        std::memcpy(buffer, text.data(), n);
        buffer[n] = '\0';
    }
    return text.size();
}

}  // namespace

extern "C" {

int32_t pg6_guest_boot_run(const Pg6GuestBootConfig* config, Pg6GuestBootStats* stats) {
    guest::BootConfig cfg;
    std::vector<std::unique_ptr<playgta6::os::pgim::ImageReader>> readers;
    if (config != nullptr) {
        for (size_t mi = 0; config->modules != nullptr && mi < config->module_count; ++mi) {
            const Pg6GuestBootModule& in = config->modules[mi];
            guest::BootModule module;
            if (in.image_bytes != nullptr) {
                readers.push_back(std::make_unique<playgta6::os::pgim::MemoryImageReader>(
                    in.image_bytes, static_cast<size_t>(in.image_size)));
            } else if (in.image_read != nullptr) {
                Pg6BootReadFn fn = in.image_read;
                void* ctx = in.image_read_context;
                readers.push_back(std::make_unique<playgta6::os::pgim::FunctionImageReader>(
                    in.image_size,
                    [fn, ctx](uint64_t offset, void* dst, size_t len) { return fn(ctx, offset, dst, len) != 0; }));
            } else {
                readers.push_back(nullptr);
            }
            module.image = readers.back().get();
            if (in.name != nullptr) module.name = in.name;
            if (in.path != nullptr) module.path = in.path;
            for (size_t i = 0; i < in.lifted_table_count; ++i) {
                if (in.lifted_tables != nullptr && in.lifted_table_counts != nullptr &&
                    in.lifted_tables[i] != nullptr) {
                    module.lifted_tables.emplace_back(in.lifted_tables[i], in.lifted_table_counts[i]);
                }
            }
            for (size_t i = 0; i < in.shard_dispatch_table_count; ++i) {
                if (in.shard_dispatch_tables != nullptr && in.shard_dispatch_tables[i] != nullptr) {
                    module.shard_dispatch_tables.push_back(in.shard_dispatch_tables[i]);
                }
            }
            if (in.imports != nullptr) {
                module.imports = std::span<const Pg6BootImport>(in.imports, in.import_count);
            }
            cfg.modules.push_back(std::move(module));
        }
        if (config->command_line != nullptr) cfg.command_line = config->command_line;
        if (config->working_directory != nullptr) cfg.working_directory = config->working_directory;
        cfg.stack_size = config->stack_size;
        cfg.max_dispatches = config->max_dispatches;
        cfg.max_wall_ms = config->max_wall_ms;
        cfg.survey = config->survey != 0;
        cfg.run_tls_callbacks = config->skip_tls_callbacks == 0;
        cfg.call_entry = config->skip_entry == 0;
        cfg.log_progress = config->quiet == 0;
        cfg.capture_host_stack = config->capture_host_stack;
    }
    const guest::BootResult result = guest::run_boot(cfg);
    if (stats != nullptr) {
        *stats = result.stats;
    }
    return static_cast<int32_t>(result.status);
}

size_t pg6_guest_boot_report_json(char* buffer, size_t size) {
    return copy_out(guest::last_report_json(), buffer, size);
}

size_t pg6_guest_boot_summary(char* buffer, size_t size) {
    return copy_out(guest::last_summary(), buffer, size);
}

void pg6_guest_boot_reset(void) {
    guest::reset_boot();
}

int32_t pg6_guest_boot_prepare_thread(Pg6GuestThreadContext* context, uint64_t stack_size) {
    if (context == nullptr) {
        return 0;
    }
    return guest::prepare_guest_thread(stack_size, 0, *context) ? 1 : 0;
}

void pg6_guest_boot_release_thread(Pg6GuestThreadContext* context) {
    if (context != nullptr) {
        guest::release_guest_thread(*context);
    }
}

Pg6GuestThreadContext* pg6_guest_boot_activate_thread(Pg6GuestThreadContext* context) {
    return guest::activate_guest_thread(context);
}

int32_t pg6_guest_boot_request_exit(int32_t exit_code) {
    return guest::request_exit(exit_code) ? 1 : 0;
}

int32_t pg6_guest_boot_report_host_fault(const char* kind, uint64_t host_address, uint64_t host_pc) {
    return guest::report_host_fault(kind, host_address, host_pc) ? 1 : 0;
}

}  // extern "C"
