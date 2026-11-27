#include "guest/boot.h"

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string_view>
#include <thread>
#include <utility>

#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_xs_exports.h"
#include "guest/audio_probes.h"
#include "guest/fault_gate.h"
#include "guest/guest_probe.h"
#include "guest/hash_registry_scan.h"
#include "guest/streaming_drain_probe.h"
#include "os/audio_stream_log.h"
#include "guest/export_counters.h"
#include "os/audio_file_stats.h"
#include "gpu/d3d12_xs_xbox.h"
#include "gpu/descriptor_heap.h"
#include "gpu/frame_breakdown.h"
#include "os/host_sleep.h"
#include "os/thread_name.h"
#include "os/thread_spans.h"
#include "gpu/frame_time_hist.h"
#include "gpu/phase_hash_counts.h"
#include "gpu/present_pacing.h"
#include "gpu/present_samples.h"
#include "gpu/small_texture_pool.h"
#include "gpu/texture_budget.h"
#include "os/guest_heap.h"
#include "os/memory_budget.h"
#include "os/process_memory.h"
#include "guest/com_bridge.h"
#include "guest/guest_crt_exit.h"
#include "guest/guest_exceptions.h"
#include "os/xgamesave_local.h"
#include "os/guest_window.h"
#include "os/os_shim.h"
#include "os/shim_exports.h"
#include "os/xbox_middleware.h"
#include "os/ucrt_bridge.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/win32_sync.h"
#include "gpu/pipeline_prewarm_host.h"
#include "guest/indirect_targets.h"
#include "os/xbox_scarlett.h"
#include "os/xbox_scarlett_guest.h"
#include "os/xgameruntime_provider.h"

#if defined(__cpp_exceptions) || defined(_CPPUNWIND) || defined(__EXCEPTIONS)
#define PG6_BOOT_HAS_EXCEPTIONS 1
#else
#define PG6_BOOT_HAS_EXCEPTIONS 0
#endif

namespace playgta6::guest {

namespace {
std::atomic<uint64_t> g_host_pack_index_bytes{0};
std::atomic<uint64_t> g_host_stage_reflection_bytes{0};
}  // namespace

void note_host_table_bytes(uint64_t shader_pack_index, uint64_t stage_reflection) {
    g_host_pack_index_bytes.store(shader_pack_index, std::memory_order_relaxed);
    g_host_stage_reflection_bytes.store(stage_reflection, std::memory_order_relaxed);
}
uint64_t host_shader_pack_index_bytes() { return g_host_pack_index_bytes.load(std::memory_order_relaxed); }
uint64_t host_stage_reflection_bytes() { return g_host_stage_reflection_bytes.load(std::memory_order_relaxed); }

namespace {
std::atomic<bool> g_stop_requested{false};  // request_boot_stop

using os::PageProtect;
using os::VirtualMemoryArena;

constexpr uint32_t kGuestProcessId = 0x1000;  // ClientId.UniqueProcess of the guest process
constexpr uint64_t kDefaultStackSize = uint64_t{1} << 20;
constexpr size_t kMaxTlsCallbacks = 256;
constexpr size_t kMaxScanBytes = size_t{16} << 20;
constexpr uint64_t kWallCheckMask = 255;  // check the wall clock every 256 dispatches
constexpr uint32_t kDispatchBatchSize = 256;  // per-thread dispatch batch flush threshold
constexpr size_t kRecentPerThread = 32;       // per-thread lock-free recent-import ring size
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
    // Fault record (BootConfig::fault_record): context captured with the fault.
    uint32_t thread_id = 0;
    uint64_t elapsed_ms = 0;
    bool has_present = false;
    uint64_t presents = 0;
    uint64_t draws_per_frame = 0;
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

// Guest addresses the recompiler missed, by fault kind: a count plus the distinct addresses in
// first-seen order (capped at kMissTargetCap; `count` keeps counting past the cap).
struct MissList {
    uint64_t count = 0;
    std::vector<uint64_t> targets;
};
constexpr size_t kMissTargetCap = 256;

struct RecentImport {
    uint64_t seq = 0;
    size_t import_index = 0;
    uint64_t args[5] = {};  // rcx rdx r8 r9 [rsp+0x28] at the call
    uint32_t thread = 0;    // guest thread id (TEB ClientId)
    uint64_t return_address = 0;  // guest return address of the call
    const GuestThreadContext* ctx = nullptr;  // owning guest thread context (for thread_last)
};

// Shared dispatch accumulator owned by a single boot run; kept alive via shared_ptr by
// thread-local observers so thread-exit flushes cannot race with reset_boot() or static teardown.
struct DispatchBatchTarget {
    std::atomic<uint64_t> dispatches{0};
    std::atomic<bool> active{true};
};

// Per-thread lock-free ring buffer of recent post-present import calls (single writer, concurrent
// readers via per-entry sequence lock). Eliminates global BootState::mutex contention on steady-
// state post-present import dispatches.
struct alignas(64) ThreadSlot {
    struct Entry {
        std::atomic<uint64_t> version{0};  // odd = write in progress, even = published
        std::atomic<uint64_t> seq{0};
        std::atomic<size_t> import_index{0};
        std::atomic<uint64_t> args[5] = {};
        std::atomic<uint32_t> thread{0};
        std::atomic<uint64_t> return_address{0};
        std::atomic<const GuestThreadContext*> ctx{nullptr};
    };

    std::atomic<uint64_t> write_count{0};
    std::atomic<uint32_t> thread_id{0};
    Entry ring[kRecentPerThread];

    // Live / exited snapshot of an O16 lazy thread's GuestState (cold path only: touched once
    // at thread init, once at thread exit, and during report synthesis). Prevents build_report()
    // from dereferencing a dangling pointer into an exited thread's TLS.
    std::mutex exit_mutex;
    const GuestState* live_state = nullptr;
    GuestState exited_state{};
    bool has_exited_state = false;

    void record(const RecentImport& entry) noexcept {
        if (entry.thread != 0) {
            thread_id.store(entry.thread, std::memory_order_relaxed);
        }
        const uint64_t idx = write_count.load(std::memory_order_relaxed);
        Entry& slot = ring[idx % kRecentPerThread];
        const uint64_t v = slot.version.load(std::memory_order_relaxed);
        slot.version.store(v + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        slot.seq.store(entry.seq, std::memory_order_relaxed);
        slot.import_index.store(entry.import_index, std::memory_order_relaxed);
        for (size_t i = 0; i < 5; ++i) {
            slot.args[i].store(entry.args[i], std::memory_order_relaxed);
        }
        slot.thread.store(entry.thread, std::memory_order_relaxed);
        slot.return_address.store(entry.return_address, std::memory_order_relaxed);
        slot.ctx.store(entry.ctx, std::memory_order_relaxed);
        slot.version.store(v + 2, std::memory_order_release);
        write_count.store(idx + 1, std::memory_order_release);
    }

    static bool read_entry(const Entry& slot, RecentImport& out) noexcept {
        for (int attempt = 0; attempt < 8; ++attempt) {
            const uint64_t v1 = slot.version.load(std::memory_order_acquire);
            if (v1 == 0 || (v1 & 1u) != 0) {
                continue;
            }
            RecentImport tmp{};
            tmp.seq = slot.seq.load(std::memory_order_relaxed);
            tmp.import_index = slot.import_index.load(std::memory_order_relaxed);
            for (size_t i = 0; i < 5; ++i) {
                tmp.args[i] = slot.args[i].load(std::memory_order_relaxed);
            }
            tmp.thread = slot.thread.load(std::memory_order_relaxed);
            tmp.return_address = slot.return_address.load(std::memory_order_relaxed);
            tmp.ctx = slot.ctx.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint64_t v2 = slot.version.load(std::memory_order_relaxed);
            if (v1 == v2) {
                out = tmp;
                return out.seq != 0;
            }
        }
        return false;
    }

    void snapshot_ring(std::vector<RecentImport>& out) const {
        const uint64_t total = write_count.load(std::memory_order_acquire);
        const size_t count = static_cast<size_t>(std::min<uint64_t>(total, kRecentPerThread));
        const uint64_t start = total - count;
        for (uint64_t k = start; k < total; ++k) {
            RecentImport item;
            if (read_entry(ring[k % kRecentPerThread], item)) {
                out.push_back(item);
            }
        }
    }

    bool read_latest(RecentImport& out) const noexcept {
        const uint64_t total = write_count.load(std::memory_order_acquire);
        if (total == 0) {
            return false;
        }
        if (read_entry(ring[(total - 1) % kRecentPerThread], out)) {
            return true;
        }
        if (total >= 2) {
            return read_entry(ring[(total - 2) % kRecentPerThread], out);
        }
        return false;
    }
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
std::atomic<FrameCounterSource> g_frame_counter_source{nullptr};
// Set before boot (set_build_info_json), possibly from a static initializer: function-local so the
// order of translation-unit initialisation does not matter.
std::string& build_info_json() {
    static std::string json;
    return json;
}

struct BootState {
    std::mutex mutex;  // recent ring, survey list

    bool active = false;                 // a boot's process exists (run_boot ran, reset_boot not yet)
    std::atomic<bool> running{false};    // guest code of the boot may run
    std::atomic<uint64_t> generation{1}; // bumped on every reset_boot()
    VirtualMemoryArena* arena = nullptr;

    // Options.
    bool survey = false;
    uint64_t survey_return = 0;
    uint64_t max_dispatches = 0;
    uint64_t max_wall_ms = 0;
    size_t recent_capacity = 32;
    size_t max_frames = 32;
    bool fault_record = true;  // BootConfig::fault_record
    std::shared_ptr<IndirectTargetLog> indirect_log;  // BootConfig::record_indirect_targets (null: off)
    CaptureHostStackFn capture_host_stack = nullptr;
    HostFunctionStartFn host_function_start = nullptr;
    OffThreadFaultFn off_thread_fault = nullptr;
    std::function<void(std::string_view)> report_sink;  // O36
    GuestThreadStartedFn guest_thread_started = nullptr;
    FaultGate fault_gate;  // first faulting thread owns capture_fault + the report; others park
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
    std::vector<long> synthetic_to_import;  // O(1) synthetic index -> import index (-1 if none)
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
    std::shared_ptr<DispatchBatchTarget> dispatch_target = std::make_shared<DispatchBatchTarget>();
    std::atomic<uint64_t> import_calls{0};
    std::vector<RecentImport> recent;  // ring (pre-present / traced / first-call)
    size_t recent_next = 0;
    std::vector<std::shared_ptr<ThreadSlot>> thread_slots;  // per-thread post-present rings, guarded by `mutex`
    std::vector<SurveyEntry> survey_called;
    // Code misses (guarded by `mutex`): indirect calls/jumps to no lifted entry, context restores
    // to no resume target, and traps. Every boot thread's fault is counted, even ones not reported.
    MissList miss_unknown_target;
    MissList miss_unknown_resume_target;
    MissList miss_trap;
    uint64_t tls_callbacks_run = 0;
    bool entry_started = false;
    std::vector<Milestone> milestones;
    std::vector<FirstCall> first_calls;  // guarded by `mutex`
    // Import calls per (guest thread << 32 | import index) until the first Present (mark_first_present).
    std::unordered_map<uint64_t, uint64_t> pre_present_counts;  // guarded by `mutex`
    bool present_seen = false;                                  // guarded by `mutex`
    std::atomic<bool> present_seen_atomic{false};               // lock-free mirror of `present_seen`
    uint64_t present_import_seq = 0;
    uint64_t present_elapsed_ms = 0;
    gpu::PresentSampleRing present_samples;  // guarded by `mutex` (R40)
    gpu::FrameTimeStats frame_times;         // guarded by `mutex`: whole-run frame-time histogram
    gpu::FrameBreakdown frame_breakdown;     // guarded by `mutex`: where each phase's frame time goes
    int64_t last_present_us = -1;            // guarded by `mutex`: steady clock of the previous present
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
    // Live/exited guest state slots of contexts whose state lives elsewhere (O16 lazy threads).
    std::unordered_map<const GuestThreadContext*, std::shared_ptr<ThreadSlot>> thread_state_slots;

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

    // B2: call counters for the exports of the lifted Bink2Scarlett.dll and how its exports bound.
    ExportCounters bink;
    ProbeSet guest_probes;     // --probe / --probe-file (guest/guest_probe.h)
    AudioProbes audio_probes;  // A4: title audio-stack call counters (main executable addresses)
    std::atomic<uint64_t> music_controller{0};  // rcx at the first music_controller_init entry (guest VA)
    GuestWatch watch;          // --watch-guest-va (guest/guest_watch.h)
    std::mutex watch_mutex;
    std::string watch_last_present;            // snapshot logged at the previous changed present
    std::atomic<uint64_t> watch_presents{0};
    bool bink_present = false;        // a loaded module named Bink2Scarlett.dll
    uint64_t bink_exports_lifted = 0; // registry exports that are guest (lifted) code
    uint64_t bink_exports_shim = 0;   // registry exports that are host shims (would shadow the DLL)

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
constexpr size_t kDispatchRing = 64;
thread_local DispatchRecord t_dispatch_ring[kDispatchRing];
thread_local uint64_t t_dispatch_count = 0;
thread_local ProbeFrames t_probe_frames;  // entries of `ret` probes waiting for their exit hook
thread_local uint64_t t_probe_dispatch_seq = 0;  // seq of the line behind t_probe_dispatch_va
thread_local uint64_t t_probe_dispatch_va = 0;  // a --probe hit the dispatch observer logged; its entry hook skips it

// Per-thread observer state: batches dispatch counter increments and owns a lock-free
// post-present recent-import ring. Its destructor flushes pending dispatches on thread exit.
struct ThreadLocalObserver {
    uint64_t generation = 0;
    uint32_t dispatch_pending = 0;
    std::shared_ptr<DispatchBatchTarget> dispatch_target;
    std::shared_ptr<ThreadSlot> slot;
    std::shared_ptr<IndirectTargetLog> indirect_log;  // null unless recording
    IndirectTargetCollector indirect_targets;

    void flush_indirect_targets() noexcept {
        if (indirect_log && indirect_targets.has_pending()) {
            try {
                indirect_log->merge(indirect_targets.take_pending());
            } catch (...) {
            }
        }
    }

    void flush_dispatches() noexcept {
        flush_indirect_targets();
        if (dispatch_pending != 0) {
            if (dispatch_target && dispatch_target->active.load(std::memory_order_relaxed)) {
                dispatch_target->dispatches.fetch_add(dispatch_pending, std::memory_order_relaxed);
            }
            dispatch_pending = 0;
        }
    }

    void reset() noexcept {
        dispatch_pending = 0;
        generation = 0;
        dispatch_target.reset();
        slot.reset();
        indirect_log.reset();
        indirect_targets.reset();
    }

    void snapshot_state_on_exit() noexcept {
        if (slot) {
            std::lock_guard<std::mutex> lock(slot->exit_mutex);
            if (slot->live_state != nullptr) {
                slot->exited_state = *slot->live_state;
                slot->live_state = nullptr;
                slot->has_exited_state = true;
            }
        }
    }

    ~ThreadLocalObserver() {
        flush_dispatches();
        snapshot_state_on_exit();
    }
};

thread_local ThreadLocalObserver t_observer;

void bind_thread_observer(BootState& b) {
    t_observer.flush_dispatches();
    std::lock_guard<std::mutex> lock(b.mutex);
    const uint64_t gen = b.generation.load(std::memory_order_acquire);
    t_observer.dispatch_target = b.dispatch_target;
    t_observer.indirect_log = b.indirect_log;
    t_observer.indirect_targets.reset();
    t_observer.slot = std::make_shared<ThreadSlot>();
    b.thread_slots.push_back(t_observer.slot);
    t_observer.generation = gen;
    t_last_import = -1;
    t_last_import_seq = 0;
    t_dispatch_count = 0;
}

uint64_t current_dispatches(BootState& b) {
    std::shared_ptr<DispatchBatchTarget> target;
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        target = b.dispatch_target;
    }
    return target ? target->dispatches.load(std::memory_order_relaxed) : 0;
}

// Merges the shared `b.recent` ring and all per-thread lock-free `ThreadSlot` rings into the
// newest `b.recent_capacity` import calls in ascending `seq` order. Caller holds `b.mutex`.
std::vector<RecentImport> collect_recent_imports_locked(const BootState& b) {
    if (b.recent_capacity == 0) {
        return {};
    }
    std::vector<RecentImport> ordered = b.recent;
    for (const auto& slot : b.thread_slots) {
        slot->snapshot_ring(ordered);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const RecentImport& a, const RecentImport& c) { return a.seq < c.seq; });
    ordered.erase(std::unique(ordered.begin(), ordered.end(),
                              [](const RecentImport& a, const RecentImport& c) { return a.seq == c.seq; }),
                  ordered.end());
    if (ordered.size() > b.recent_capacity) {
        ordered.erase(ordered.begin(),
                      ordered.end() - static_cast<std::ptrdiff_t>(b.recent_capacity));
    }
    return ordered;
}

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
    const BootState& b = boot();
    if (!is_synthetic_address(target)) {
        return -1;
    }
    const uint64_t offset = target - kSyntheticBase;
    if ((offset & (kSyntheticStride - 1)) != 0) {
        return -1;
    }
    const size_t synth_idx = static_cast<size_t>(offset / kSyntheticStride);
    if (synth_idx < b.synthetic_to_import.size()) {
        const long cached = b.synthetic_to_import[synth_idx];
        if (cached >= 0) {
            return cached;
        }
    }
    SyntheticTarget info;
    if (!describe_synthetic(target, &info) || !info.is_import) {
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
FaultHookFn g_fault_hook = nullptr;
std::string fault_record_json(const FaultRecord& f);

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
    r.thread_id = t_active != nullptr ? t_active->thread_id : 0;
    r.elapsed_ms = elapsed_ms();
    if (b.fault_record) {
        std::unique_lock<std::mutex> lock(b.mutex, std::try_to_lock);
        if (lock.owns_lock() && b.present_samples.total() != 0) {
            r.has_present = true;
            r.presents = b.present_samples.total();
            r.draws_per_frame = b.present_samples.last().draws;
        }
    }
    const uint64_t count = b.fault_record ? std::min<uint64_t>(t_dispatch_count, kDispatchRing) : 0;
    for (uint64_t k = t_dispatch_count - count; k < t_dispatch_count; ++k) {
        r.recent_dispatches.push_back(t_dispatch_ring[k % kDispatchRing]);
    }
    if (g_fault_hook != nullptr && b.fault_record) {
        g_fault_hook(fault_record_json(r));
    }
}

void log_trap();
void log_fault_brief();
void finish_report();

[[noreturn]] void stop_boot() {
#if PG6_BOOT_HAS_EXCEPTIONS
    throw BootStop{};
#else
    log_trap();
    os::log("guest boot: stopping without exception support; aborting the process");
    std::fflush(stderr);
    abort_once();
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

void record_code_miss(FaultKind kind, uint64_t va) {
    BootState& b = boot();
    MissList* list = nullptr;
    switch (kind) {
        case FaultKind::UnknownTarget:
            list = &b.miss_unknown_target;
            break;
        case FaultKind::UnknownResumeTarget:
            list = &b.miss_unknown_resume_target;
            break;
        case FaultKind::Trap:
            list = &b.miss_trap;
            break;
        default:
            return;
    }
    std::lock_guard<std::mutex> lock(b.mutex);
    ++list->count;
    if (list->targets.size() < kMissTargetCap &&
        std::find(list->targets.begin(), list->targets.end(), va) == list->targets.end()) {
        list->targets.push_back(va);
    }
}

void boot_fault_handler(const FaultInfo* info) {
    BootState& b = boot();
    if (!b.running.load(std::memory_order_acquire) || info == nullptr) {
        return;  // the default handler logs and aborts
    }
    t_observer.flush_dispatches();
    {
        const auto miss_kind = static_cast<FaultKind>(info->kind);
        record_code_miss(miss_kind, miss_kind == FaultKind::UnknownResumeTarget && info->state != nullptr
                                        ? info->state->resume_rip
                                        : info->va);
    }
    const bool other_thread = !t_running_boot;
    if (other_thread && b.off_thread_fault == nullptr) {
        return;
    }
    b.fault_gate.claim_or_park();  // concurrent faults on other threads park behind the first
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
    b.fault_gate.claim_or_park();
    t_observer.flush_dispatches();
    capture_fault(kind, state, target, 0);
    char message[160];
    std::snprintf(message, sizeof(message), "run bound reached (%s) after %" PRIu64 " dispatches",
                  kind, current_dispatches(b));
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

// B2: finds the lifted Bink2Scarlett.dll, counts calls to every named export, and records which
// of the module registry's exports are guest code (lifted) and which are host shims (B1: a shim
// binding would silently refuse every movie).
void setup_bink_counters(BootState& b) {
    b.bink.clear();
    b.bink_present = false;
    b.bink_exports_lifted = 0;
    b.bink_exports_shim = 0;
    if (b.arena == nullptr) {
        return;
    }
    struct FallbackLog {
        const BootState& b;
        ~FallbackLog() {
            if (!b.bink_present) {
                logf("guest boot: no lifted Bink2Scarlett.dll in this program: Bink imports use the no-decoder "
                     "pg6_mw_bink_* shim fallback (movies are refused)");
            }
        }
    } fallback_log{b};
    for (const ModuleState& m : b.modules) {
        if (!m.loaded || m.image.base == 0 || os::win32::canonical_module_name(m.name) != "bink2scarlett.dll") {
            continue;
        }
        b.bink_present = true;
        std::vector<PeExport> exports;
        VirtualMemoryArena& arena = *b.arena;
        enumerate_pe_exports([&arena](uint64_t va, void* dst, size_t n) { return arena.read(va, dst, n); },
                             m.image.base, exports);
        const os::win32::Handle handle = os::win32::find_module_by_address(m.image.base);
        for (const PeExport& e : exports) {
            const os::win32::ExportLookup lookup =
                handle != 0 ? os::win32::LookupExport(handle, e.name) : os::win32::ExportLookup{};
            if (handle == 0 || lookup.address == 0) {
                pg6_bink_note_lookup_null(e.name.c_str());
                continue;  // not in the registry at all: neither lifted nor shim
            }
            (lookup.guest ? b.bink_exports_lifted : b.bink_exports_shim) += 1;
        }
        const size_t n = exports.size();
        b.bink.configure(m.name, std::move(exports));
        logf("guest boot: %s is lifted: %zu named exports counted, %" PRIu64 " bind to lifted code, %" PRIu64
             " to a host shim%s",
             m.name.c_str(), n, b.bink_exports_lifted, b.bink_exports_shim,
             b.bink_exports_shim != 0 ? " (WARNING: the no-decoder shim shadows the DLL)" : "");
        return;
    }
}

// A4: one stream-log event per probe hit (first kMaxPerKind per probe; the counters keep counting).
// The metadata lookup is hot, so only its three music-controller hashes are logged.
void audio_probe_event(BootState& b, long idx, const uint64_t args[AudioProbes::kArgs], bool entry) {
    if (idx < 0) {
        return;
    }
    const std::string& name = b.audio_probes.name(static_cast<size_t>(idx));
    if (name == "metadata_hash_lookup") {
        const uint32_t h = static_cast<uint32_t>(args[0]);
        if (h != 0x9205c7a1u && h != 0xd3594a47u && h != 0x4aa37aa8u) {
            return;
        }
    } else if (name == "music_controller_init") {
        uint64_t zero = 0;
        b.music_controller.compare_exchange_strong(zero, args[0]);
    } else if (name == "metadata_registry_insert") {
        // rdx points at the key; only the three music hashes are logged (the counters count all).
        uint32_t h = 0;
        if (b.arena == nullptr || !read_u32(*b.arena, args[1], h) ||
            (h != 0x9205c7a1u && h != 0xd3594a47u && h != 0x4aa37aa8u)) {
            return;
        }
        os::audio_stream_log::log().record("probe_entry_metadata_registry_insert_music", h);
        return;
    }
    os::audio_stream_log::log().record((std::string(entry ? "probe_entry_" : "probe_call_") + name).c_str(), args[0],
                                       args[1], args[2]);
}

// A4: the music controller's looked-up objects, read from guest memory at report time: null means the
// metadata lookup (by name hash) found nothing. Offsets are fields of the controller object.
std::string music_controller_json(BootState& b) {
    const uint64_t obj = b.music_controller.load();
    if (obj == 0 || b.arena == nullptr) {
        return "{\"init_seen\": false}";
    }
    static const struct { const char* name; uint64_t off; } kSlots[] = {
        {"lookup_2758", 0x2758}, {"lookup_3058", 0x3058}, {"lookup_3060", 0x3060}, {"lookup_3068", 0x3068},
        {"lookup_3078", 0x3078}, {"lookup_3080", 0x3080}, {"lookup_3088", 0x3088}, {"lookup_3090", 0x3090},
    };
    char head[96];
    std::snprintf(head, sizeof head, "{\"init_seen\": true, \"object\": \"0x%" PRIx64 "\", \"slots\": {", obj);
    std::string o = head;
    bool first = true;
    for (const auto& sl : kSlots) {
        uint64_t v = 0;
        const bool ok = read_u64(*b.arena, obj + sl.off, v);
        o += std::string(first ? "" : ", ") + "\"" + sl.name + "\": " + (!ok ? "\"unreadable\"" : v == 0 ? "\"null\"" : "\"set\"");
        first = false;
    }
    o += "}, \"flags\": {";
    uint64_t w = 0;
    if (read_u64(*b.arena, obj + 0x3180, w)) {
        o += "\"b3180\": " + std::to_string(w & 0xff) + ", \"b3181\": " + std::to_string((w >> 8) & 0xff);
    }
    return o + "}}";
}

// A4: the name-hash registry the music controller's metadata lookups search (globals of the main
// executable: group count, entry count, table pointer), read at report time: entries in total and
// whether each music hash is registered. "absent" in a populated registry means nothing registered
// that name, so the lookup's null is not a lookup bug.
std::string metadata_registry_json(BootState& b) {
    if (b.arena == nullptr || b.modules.empty() || !b.modules[0].loaded || b.modules[0].image.base == 0) {
        return "{\"available\": false}";
    }
    const uint64_t g = b.modules[0].image.base + (0x149ab7d18ull - 0x140000000ull);
    uint32_t groups = 0, entries = 0;
    uint64_t table = 0;
    if (!read_u32(*b.arena, g, groups) || !read_u32(*b.arena, g + 4, entries) || !read_u64(*b.arena, g + 8, table)) {
        return "{\"available\": false}";
    }
    static const std::vector<uint32_t> kMusic = {0x9205c7a1u, 0xd3594a47u, 0x4aa37aa8u};
    VirtualMemoryArena* arena = b.arena;
    const HashRegistryScan scan = scan_hash_registry(
        [arena](uint64_t va, void* dst, size_t len) { return arena->read(va, dst, len); }, table, groups, kMusic);
    std::string o = "{\"available\": true, \"groups\": " + std::to_string(groups) +
                    ", \"entries\": " + std::to_string(entries) + ", \"readable\": " +
                    (scan.readable ? "true" : "false") + ", \"full_slots\": " + std::to_string(scan.full_slots) +
                    ", \"music_hashes\": {";
    for (size_t i = 0; i < kMusic.size(); ++i) {
        char key[32];
        std::snprintf(key, sizeof key, "\"0x%08x\": ", kMusic[i]);
        o += std::string(i == 0 ? "" : ", ") + key +
             (!scan.present[i] ? "\"absent\"" : scan.value_null[i] ? "\"null_value\"" : "\"present\"");
    }
    return o + "}}";
}

// A4: the title's audio stack as guest addresses (main executable, linked at 0x140000000). The
// decoder factory is reached by vtable, so the dispatch observer sees it; a direct call between
// two lifted functions would not show (`calls`, "indirect_calls_only"); a build with
// `--probe-functions` also fills `entry_calls` from an entry counter, direct calls included.
void setup_audio_probes(BootState& b) {
    b.audio_probes.clear();
    if (b.modules.empty() || !b.modules[0].loaded || b.modules[0].image.base == 0) {
        return;
    }
    constexpr uint64_t kLinkBase = 0x140000000ull;
    const uint64_t base = b.modules[0].image.base;
    struct Def { const char* name; uint64_t va; };
    static const Def kDefs[] = {
        {"decoder_factory", 0x144ee95f0ull},          // codec 11 Opus, 5 XMA, 0..3 PCM/ADPCM
        {"decoder_factory_caller_a", 0x140078560ull},
        {"decoder_factory_caller_b", 0x144ee8a10ull},
        {"music_controller_init", 0x145b67eb0ull},
        {"stream_decoder_create", 0x140078f00ull},
        {"stream_consumer", 0x140078120ull},
        {"metadata_hash_lookup", 0x140c8a070ull},      // ecx = name hash; only the music hashes are logged
        {"metadata_registry_build", 0x141e74390ull},   // startup pass that fills that name-hash registry
        {"metadata_registry_insert", 0x141d73cc0ull},  // its insert; rdx = &key, only the music hashes are logged
    };
    std::vector<AudioProbe> probes;
    for (const Def& d : kDefs) {
        probes.push_back(AudioProbe{d.name, base + (d.va - kLinkBase)});
    }
    b.audio_probes.configure(std::move(probes));
}

// `--probe` / `--probe-file`: resolve the specs against the loaded modules (a main-executable link VA
// is rebased onto the image, `module+off` onto that module) and open the JSONL file. A spec that
// does not resolve is reported and skipped.
void setup_guest_probes(BootState& b, const BootConfig& config) {
    b.guest_probes.clear();
    if (config.probe_specs.empty()) {
        return;
    }
    constexpr uint64_t kLinkBase = 0x140000000ull;
    std::vector<ProbeSpec> specs;
    for (const std::string& text : config.probe_specs) {
        ProbeSpec s;
        std::string err;
        if (!parse_probe_spec(text, &s, &err)) {
            logf("[probe] ignoring '%s': %s", text.c_str(), err.c_str());
            continue;
        }
        if (!s.module.empty()) {
            const ModuleState* found = nullptr;
            for (const ModuleState& m : b.modules) {
                std::string a = m.name, z = s.module;
                std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                std::transform(z.begin(), z.end(), z.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (m.loaded && m.image.base != 0 && a == z) {
                    found = &m;
                    break;
                }
            }
            if (found == nullptr) {
                logf("[probe] ignoring '%s': module %s is not loaded", text.c_str(), s.module.c_str());
                continue;
            }
            s.addr += found->image.base;
        } else if (s.addr >= kLinkBase && !b.modules.empty() && b.modules[0].loaded && b.modules[0].image.base != 0) {
            s.addr = b.modules[0].image.base + (s.addr - kLinkBase);
        }
        specs.push_back(std::move(s));
    }
    b.guest_probes.configure(std::move(specs), config.probe_out);
    for (size_t i = 0; i < b.guest_probes.size(); ++i) {
        logf("[probe] %s at 0x%" PRIx64 " -> %s",
             b.guest_probes.spec(i).name.empty() ? "probe" : b.guest_probes.spec(i).name.c_str(),
             b.guest_probes.address(i), config.probe_out.empty() ? "(counters only)" : config.probe_out.c_str());
    }
}

// One `--probe` hit: counted always, logged (as a JSONL line) under the rate limits. `entry` is the
// `__playgta6_probe_hit` entry counter, otherwise the dispatch observer. Reads go through the
// arena's checked copy with the diagnostic-read scope (never blocks on the arena lock).
void guest_probe_hit(BootState& b, GuestState* state, long idx, uint64_t va, bool entry) {
    if (idx < 0) {
        return;
    }
    const bool wants_ret = b.guest_probes.spec(static_cast<size_t>(idx)).ret;
    if (entry) {
        if (t_probe_dispatch_va == va) {
            t_probe_dispatch_va = 0;  // the dispatch observer already logged this execution
            if (wants_ret) {
                t_probe_frames.push(idx, va, state->gpr[GuestReg::Rsp], t_probe_dispatch_seq);
            }
            t_probe_dispatch_seq = 0;
            return;
        }
    } else {
        t_probe_dispatch_va = va;
        t_probe_dispatch_seq = 0;
    }
    if (!b.guest_probes.count_hit(static_cast<size_t>(idx), entry)) {
        if (entry && wants_ret) {
            t_probe_frames.push(idx, va, state->gpr[GuestReg::Rsp], 0);  // exit still counted, unlogged entry
        }
        return;
    }
    ProbeHit hit;
    hit.t_ms = elapsed_ms();
    hit.thread = t_active != nullptr ? t_active->thread_id : 0u;
    hit.source = entry ? "entry" : "dispatch";
    for (int i = 0; i < 16; ++i) {
        hit.regs[i] = state->gpr[i];
    }
    VirtualMemoryArena* arena = b.arena;
    const uint64_t seq = b.guest_probes.log_hit(static_cast<size_t>(idx), hit, [arena](uint64_t a, void* dst, size_t n) {
        if (arena == nullptr) {
            return false;
        }
        VirtualMemoryArena::DiagnosticReads scope;
        return arena->read(a, dst, n);
    });
    if (entry && wants_ret) {
        t_probe_frames.push(idx, va, state->gpr[GuestReg::Rsp], seq);
    } else if (!entry) {
        t_probe_dispatch_seq = seq;
    }
}

// Exit of a function listed in `--probe-functions` (the lifter calls `__playgta6_probe_exit` on each
// return path): pairs it with the entry frame of this thread and logs the return value.
void guest_probe_exit(BootState& b, GuestState* state, uint64_t va) {
    const long idx = b.guest_probes.find(va);
    if (idx < 0 || !b.guest_probes.spec(static_cast<size_t>(idx)).ret || state->unwind_pending != 0) {
        return;  // a context restore passing through returns no value (its frame is dropped later)
    }
    ProbeFrames::Frame frame;
    const bool paired = t_probe_frames.pop(va, state->gpr[GuestReg::Rsp], &frame);
    if (paired && frame.entry_seq == 0) {
        b.guest_probes.count_exit_only(static_cast<size_t>(idx));  // its entry line was rate limited
        return;
    }
    b.guest_probes.log_exit(static_cast<size_t>(idx), t_active != nullptr ? t_active->thread_id : 0u, elapsed_ms(),
                            paired ? frame.entry_seq : 0, paired, state->gpr[GuestReg::Rax], state->xmm[0]);
}

// First call of a counted Bink2Scarlett.dll export: BinkOpenWithOptions is the BINK_OPEN milestone
// (the import-name milestone cannot see a guest-to-guest call), the others are logged.
void bink_first_call(BootState& b, size_t index) {
    const std::string& name = b.bink.name(index);
    if (name == "BinkOpenWithOptions") {
        milestone("BINK_OPEN", "first %s call (lifted %s)", name.c_str(), b.bink.module().c_str());
    } else if (b.log_progress) {
        os::log(("[bink] first call of lifted export " + name).c_str());
    }
}

void boot_dispatch_observer(GuestState* state, uint64_t target) {
    BootState& b = boot();
    if (!b.running.load(std::memory_order_relaxed)) {
        return;
    }
    const uint64_t gen = b.generation.load(std::memory_order_acquire);
    if (t_observer.generation != gen) {
        bind_thread_observer(b);
    }
    if (++t_observer.dispatch_pending >= kDispatchBatchSize) {
        t_observer.flush_dispatches();
    }
    if (b.bink.active()) {
        bool first = false;
        const long bink_index = b.bink.hit(target, &first);
        if (first && bink_index >= 0) {
            bink_first_call(b, static_cast<size_t>(bink_index));
        }
    }
    if (b.guest_probes.active()) {
        guest_probe_hit(b, state, b.guest_probes.find(target), target, false);
    }
    if (b.audio_probes.active()) {
        const uint64_t args[AudioProbes::kArgs] = {state->gpr[GuestReg::Rcx], state->gpr[GuestReg::Rdx],
                                                   state->gpr[GuestReg::R8], state->gpr[GuestReg::R9]};
        bool first = false;
        const long idx = b.audio_probes.hit(target, args, &first);
        audio_probe_event(b, idx, args, false);
        if (first && idx >= 0 && b.log_progress) {
            os::log(("[audio-probe] first call of " + b.audio_probes.name(static_cast<size_t>(idx))).c_str());
        }
    }
    const uint64_t rsp = state->gpr[GuestReg::Rsp];
    uint64_t return_address = 0;
    {
        // With the fault record off only the latest slot is kept (event-trace caller / import
        // return address), so the write cost stays one slot.
        if (!b.fault_record) {
            t_dispatch_count = 1;
        }
        DispatchRecord& d = b.fault_record ? t_dispatch_ring[t_dispatch_count++ % kDispatchRing]
                                           : t_dispatch_ring[0];
        d.target = target;
        d.rcx = state->gpr[GuestReg::Rcx];
        d.rdx = state->gpr[GuestReg::Rdx];
        if (t_active != nullptr && b.arena != nullptr && rsp >= t_active->thread.stack_limit &&
            rsp <= t_active->thread.stack_top - 8) {
            // The thread's own stack is committed read/write for its whole life: read it like
            // lifted code does, without the arena lock (taken on every dispatch, it serialized
            // all guest threads).
            std::memcpy(&return_address, b.arena->translate(rsp), sizeof(return_address));
        } else if (b.arena != nullptr) {
            read_u64(*b.arena, rsp, return_address);
        }
        d.return_address = return_address;
        if (t_observer.indirect_log && !is_synthetic_address(target)) {
            t_observer.indirect_targets.note(return_address, target);
        }
    }
    if (is_synthetic_address(target)) {
        // Fast O(1) lookup for statically bound IAT slots; only fall back to describe_synthetic
        // once for host callables or dynamically bound slots.
        long index = -1;
        SyntheticTarget info;
        bool has_info = false;
        const uint64_t synth_off = target - kSyntheticBase;
        if ((synth_off & (kSyntheticStride - 1)) == 0) {
            const size_t synth_idx = static_cast<size_t>(synth_off / kSyntheticStride);
            if (synth_idx < b.synthetic_to_import.size()) {
                index = b.synthetic_to_import[synth_idx];
            }
        }
        if (index < 0 && describe_synthetic(target, &info)) {
            has_info = true;
            if (info.is_import) {
                index = find_import(info.image_base, info.slot_rva);
            }
        }
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
            if (t_active != nullptr && b.arena != nullptr && rsp >= t_active->thread.stack_limit &&
                t_active->thread.stack_top >= 0x30 && rsp <= t_active->thread.stack_top - 0x30) {
                std::memcpy(&entry.args[4], b.arena->translate(rsp + 0x28), sizeof(entry.args[4]));
            } else if (b.arena != nullptr) {
                read_u64(*b.arena, rsp + 0x28, entry.args[4]);
            }
            entry.thread = t_active != nullptr ? t_active->thread_id : 0;
            entry.return_address = return_address;
            entry.ctx = t_active;
            const BootImport& imp = b.imports[static_cast<size_t>(index)].import;
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
            if (b.present_seen_atomic.load(std::memory_order_relaxed) && b.trace_limit == 0 && !first_call) {
                // Steady-state post-present fast path: zero global mutex, zero map lookup.
                t_observer.slot->record(entry);
            } else {
                std::string strings;
                if (b.trace_limit != 0) {
                    strings = describe_string_args(entry.args);
                }
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
        } else if (has_info && !info.is_import && info.name != nullptr) {
            // Host callables handed out by GetProcAddress / providers (QueryApiImpl, ...).
            bool inserted = false;
            {
                // Per-callable call counts with the import-call number of the first call
                // (COM methods: D3D12 / xgameruntime milestones and slot statistics).
                std::lock_guard<std::mutex> lock(b.callable_mutex);
                auto [it, ins] = b.callable_calls.try_emplace(target);
                inserted = ins;
                if (inserted) {
                    it->second.name = info.name;
                    it->second.first_import_seq = b.import_calls.load(std::memory_order_relaxed);
                    it->second.order = b.callable_calls.size();
                }
                ++it->second.calls;
            }
            if (inserted) {
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
                    break;
                }
            }
        }
    }
    if (!t_running_boot) {
        return;  // bounds can only stop the boot's own thread
    }
    if (g_stop_requested.load(std::memory_order_relaxed)) {
        t_observer.flush_dispatches();
        stop_on_limit(state, "stop_requested", target);
    }
    if (b.max_dispatches != 0) {
        const uint64_t total =
            t_observer.dispatch_target->dispatches.load(std::memory_order_relaxed) + t_observer.dispatch_pending;
        if (total > b.max_dispatches) {
            t_observer.flush_dispatches();
            stop_on_limit(state, "dispatch_limit", target);
        }
    }
    if (b.max_wall_ms != 0 && (t_dispatch_count & kWallCheckMask) == 0 && elapsed_ms() > b.max_wall_ms) {
        t_observer.flush_dispatches();
        stop_on_limit(state, "wall_time_limit", target);
    }
}

// --watch-guest-va: one line per runtime-side operation touching a watched range. Runs under the
// arena's lock, so the range itself is not read here (a decommitted page would fault).
void watch_arena_event(BootState& b, os::ArenaEvent event, uint64_t va, uint64_t size, bool ok) {
    if (!b.log_progress || !b.watch.take_line()) {
        return;
    }
    static const char* const kNames[] = {"reserve", "commit", "decommit", "release", "reset", "write"};
    const long imp = t_last_import;
    const std::string import = imp >= 0 && static_cast<size_t>(imp) < b.imports.size()
                                   ? import_label(b.imports[static_cast<size_t>(imp)].import)
                                   : std::string("runtime");
    logf("[watch] t=%" PRIu64 " thread=%u %s va=0x%" PRIx64 " size=0x%" PRIx64 " ok=%d under=%s import_seq=%" PRIu64,
         elapsed_ms(), t_active != nullptr ? t_active->thread_id : 0u, kNames[static_cast<int>(event)], va, size,
         ok ? 1 : 0, import.c_str(), t_last_import_seq);
}

// Contents of the watched ranges, read through the arena's checked copy.
std::string watch_snapshot(BootState& b) {
    return b.watch.snapshot([&b](uint64_t va, void* dst, size_t len) {
        return b.arena != nullptr && b.arena->read(va, dst, len);
    });
}

// At each present, one line when a watched range changed since the previous present: how the
// contents move frame by frame without a probe build (guest stores themselves are not seen).
void watch_at_present(BootState& b) {
    if (!b.watch.active() || !b.log_progress) {
        return;
    }
    std::string now = watch_snapshot(b);
    const uint64_t presents = b.watch_presents.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        std::lock_guard<std::mutex> lock(b.watch_mutex);
        if (now == b.watch_last_present) {
            return;
        }
        b.watch_last_present = now;
    }
    if (!b.watch.take_line()) {
        return;
    }
    char head[96];
    std::snprintf(head, sizeof(head), "[watch] t=%" PRIu64 " thread=%u present %" PRIu64, elapsed_ms(),
                  t_active != nullptr ? t_active->thread_id : 0u, presents);
    os::log(std::string(head) + now);
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
    if (event == os::ArenaEvent::Write || event == os::ArenaEvent::Reset) {
        // Not part of the address-space log: only a watched range cares.
        if (b.watch.active() && b.watch.overlaps(va, size)) {
            watch_arena_event(b, event, va, size, ok);
        }
        return;
    }
    std::lock_guard<std::mutex> lock(b.arena_mutex);
    if (b.watch.active()) {
        uint64_t span = size;
        if (event == os::ArenaEvent::Release) {
            auto it = b.live_reservations.find(va);
            span = it != b.live_reservations.end() ? it->second : 1;
        }
        if (b.watch.overlaps(va, span)) {
            watch_arena_event(b, event, va, span, ok);
        }
    }
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
    const bool running = b.running.load(std::memory_order_acquire);
    if (running && b.survey && index >= 0) {
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
    if (running && t_running_boot) {
        b.fault_gate.claim_or_park();
        t_observer.flush_dispatches();
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
    if (running && b.off_thread_fault != nullptr) {
        b.fault_gate.claim_or_park();
        t_observer.flush_dispatches();
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
    abort_once();
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
    if (!b.running.load(std::memory_order_acquire) || !t_running_boot) {
        if (b.previous_exit_hook != nullptr) {
            b.previous_exit_hook(code, kind);
        }
        return;
    }
    t_observer.flush_dispatches();
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

// XLaunchNewGame of the running title: the guest's constrained-change then app-state routines run
// with TRUE on the calling guest thread, then a ~1 s grace period for work they started.
void boot_relaunch_invoke(const os::win32::AppNotification& call, void* /*user*/) {
    com::GuestFunction<void(bool, uint64_t)> routine(call.routine);
    routine(call.flag != 0, call.context);
}

void boot_relaunch_grace(uint32_t grace_ms, void* /*user*/) {
    std::this_thread::sleep_for(std::chrono::milliseconds(grace_ms));
}

void boot_relaunch_prepare(void* /*user*/) {
    os::win32::run_relaunch_suspend_sequence(&boot_relaunch_invoke, nullptr, &boot_relaunch_grace, nullptr);
    // Saves the title wrote during the suspend routines must be on disk before the process is replaced.
    os::gdk::gamesave::flush_all();
    gpu::prewarm_host_save();  // the pipelines this run created, for the next launch's prewarm
}

// Guest return address of this thread's latest dispatched call (PG6_EVENT_TRACE callers).
uint64_t boot_current_guest_caller() {
    return t_dispatch_count != 0 ? t_dispatch_ring[(t_dispatch_count - 1) % kDispatchRing].return_address : 0;
}

void install_hooks() {
    BootState& b = boot();
    os::win32::set_event_trace_caller(&boot_current_guest_caller);
    b.previous_fault_handler = set_fault_handler(&boot_fault_handler);
    b.previous_observer = set_dispatch_observer(&boot_dispatch_observer);
    b.previous_exit_hook = pg6_crt_set_exit_hook(&boot_exit_hook);
    os::win32::set_process_exit_handler(&boot_terminate_process_handler, nullptr);
    os::win32::set_relaunch_prepare_hook(&boot_relaunch_prepare, nullptr);
    b.hooks_installed = true;
}

void remove_hooks() {
    BootState& b = boot();
    t_observer.flush_dispatches();
    if (!b.hooks_installed) {
        return;
    }
    set_fault_handler(b.previous_fault_handler);
    set_dispatch_observer(b.previous_observer);
    pg6_crt_set_exit_hook(b.previous_exit_hook);
    os::win32::set_process_exit_handler(nullptr, nullptr);
    os::win32::set_relaunch_prepare_hook(nullptr, nullptr);
    os::win32::set_event_trace_caller(nullptr);
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
    static const unsigned hw = std::thread::hardware_concurrency();
    uint64_t affinity_mask = os::win32::get_process_affinity_mask();
    uint32_t affinity_bits = 0;
    while (affinity_mask != 0) {
        affinity_bits += static_cast<uint32_t>(affinity_mask & 1u);
        affinity_mask >>= 1;
    }
    const uint32_t cpus =
        std::min<uint32_t>(hw == 0 ? 1u : hw, affinity_bits == 0 ? 1u : affinity_bits);
    peb->number_of_processors = std::max<uint32_t>(1u, std::min<uint32_t>(cpus, 64u));
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
    const uint64_t gen = b.generation.load(std::memory_order_acquire);
    if (t_observer.generation != gen) {
        bind_thread_observer(b);
    }
    {
        std::lock_guard<std::mutex> slock(t_observer.slot->exit_mutex);
        t_observer.slot->live_state = &thread.state;
        t_observer.slot->has_exited_state = false;
    }
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        b.threads.push_back(std::move(ctx));
        b.thread_state_slots[raw] = t_observer.slot;
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

// "dll!symbol" names of a module's imports with no shim and no runtime thunk (capped), so an
// unresolved import is named in the report instead of only counted.
std::string unresolved_names_json(size_t module) {
    constexpr size_t kCap = 64;
    const BootState& b = boot();
    std::string o = "[";
    size_t listed = 0;
    for (const ImportRecord& rec : b.imports) {
        if (rec.module != module || import_resolved(rec.import)) {
            continue;
        }
        if (listed++ < kCap) {
            o += (listed == 1 ? "" : ", ") + json_string(import_label(rec.import));
        }
    }
    return o + "]";
}

// B1/B2: the lifted Bink2Scarlett.dll's per-export call counts, how its exports bound, and the
// fallback shim's own call count (non-zero: the shim, not the DLL, answered an open).
std::string bink_json() {
    const BootState& b = boot();
    uint64_t shim_opens = 0;
    uint64_t shim_refused = 0;
    pg6_bink_shim_stats(&shim_opens, &shim_refused);
    uint64_t imports_to_shim = 0;
    for (const ImportRecord& rec : b.imports) {
        if (rec.import.dll != nullptr && os::win32::canonical_module_name(rec.import.dll) == "bink2scarlett.dll" &&
            rec.import.shim != nullptr && import_resolved(rec.import)) {
            ++imports_to_shim;
        }
    }
    const char* binding = !b.bink_present ? "absent"
                          : b.bink_exports_shim != 0 ? (b.bink_exports_lifted != 0 ? "mixed" : "shim")
                                                      : "lifted";
    std::string o = "  \"bink\": {\"module_loaded\": " + std::string(b.bink_present ? "true" : "false") +
                    ", \"binding\": \"" + binding + "\", \"exports_lifted\": " +
                    std::to_string(b.bink_exports_lifted) + ", \"exports_shim\": " +
                    std::to_string(b.bink_exports_shim) + ", \"imports_to_shim\": " +
                    std::to_string(imports_to_shim) + ", \"shim_open_calls\": " + std::to_string(shim_opens) +
                    ", \"shim_refused_opens\": " + std::to_string(shim_refused) +
                    ", \"lookup_null\": " + std::to_string(pg6_bink_lookup_null_count()) + ", \"lookup_null_names\": [";
    for (size_t i = 0;; ++i) {
        char nm[160];
        if (pg6_bink_lookup_null_name(i, nm, sizeof(nm)) == 0) {
            break;
        }
        o += (i == 0 ? "" : ", ") + json_string(nm);
    }
    o += "], \"open_log\": [";
    for (size_t i = 0;; ++i) {
        char path[300];
        uint32_t flags = 0;
        int opened = 0;
        if (pg6_bink_open_log(i, path, sizeof(path), &flags, &opened) == 0) {
            break;
        }
        o += (i == 0 ? "" : ", ") + std::string("{\"path\": ") + json_string(path) + ", \"flags\": " +
             std::to_string(flags) + ", \"opened\": " + (opened != 0 ? "true" : "false") + "}";
    }
    o += "], \"exports\": [";
    uint64_t total = 0;
    for (size_t i = 0; i < b.bink.size(); ++i) {
        const uint64_t calls = b.bink.calls(i);
        total += calls;
        o += (i == 0 ? "" : ", ") + std::string("{\"name\": ") + json_string(b.bink.name(i)) +
             ", \"address\": " + hex(b.bink.address(i)) + ", \"calls\": " + std::to_string(calls) + "}";
    }
    return o + "], \"total_calls\": " + std::to_string(total) + "}";
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

// Nearest named export of a loaded module at or below `va` (fault record only; reads the mapped
// export directory). Returns false when no module covers `va` or none of its exports precedes it.
struct ExportHit {
    std::string module;
    std::string name;
    uint64_t address = 0;
};

bool nearest_export(uint64_t va, ExportHit& out) {
    BootState& b = boot();
    if (b.arena == nullptr) {
        return false;
    }
    VirtualMemoryArena& arena = *b.arena;
    for (const ModuleState& m : b.modules) {
        const uint64_t base = m.image.base;
        if (!m.loaded || base == 0 || va < base || va - base >= m.image.size_of_image) {
            continue;
        }
        uint32_t lfanew = 0;
        uint8_t nt[0x90];
        if (!arena.read(base + 0x3C, &lfanew, sizeof(lfanew)) || lfanew > 0x1000 ||
            !arena.read(base + lfanew, nt, sizeof(nt))) {
            return false;
        }
        uint32_t sig = 0;
        std::memcpy(&sig, nt, 4);
        uint16_t magic = 0;
        std::memcpy(&magic, nt + 0x18, 2);
        if (sig != 0x4550 || magic != 0x20B) {
            return false;
        }
        uint32_t dir_rva = 0;
        uint32_t dir_size = 0;
        std::memcpy(&dir_rva, nt + 0x18 + 0x70, 4);
        std::memcpy(&dir_size, nt + 0x18 + 0x74, 4);
        uint32_t ed[10];
        if (dir_rva == 0 || dir_size == 0 || !arena.read(base + dir_rva, ed, sizeof(ed))) {
            return false;
        }
        const uint32_t n_funcs = ed[5];
        const uint32_t n_names = ed[6];
        const uint32_t funcs = ed[7];
        const uint32_t names = ed[8];
        const uint32_t ords = ed[9];
        if (n_names > 200000 || n_funcs > 200000) {
            return false;
        }
        const uint64_t rva = va - base;
        uint64_t best_rva = 0;
        std::string best_name;
        for (uint32_t i = 0; i < n_names; ++i) {
            uint32_t name_rva = 0;
            uint16_t ord = 0;
            uint32_t func_rva = 0;
            if (!arena.read(base + names + 4ULL * i, &name_rva, 4) ||
                !arena.read(base + ords + 2ULL * i, &ord, 2) || ord >= n_funcs ||
                !arena.read(base + funcs + 4ULL * ord, &func_rva, 4)) {
                continue;
            }
            if (func_rva > rva || func_rva < best_rva || (func_rva >= dir_rva && func_rva < dir_rva + dir_size)) {
                continue;  // above va, worse than the best, or a forwarder string
            }
            char name[128] = {};
            if (!arena.read(base + name_rva, name, sizeof(name) - 1)) {
                continue;
            }
            best_rva = func_rva;
            best_name = name;
        }
        if (best_name.empty()) {
            return false;
        }
        out.module = m.name;
        out.name = best_name;
        out.address = base + best_rva;
        return true;
    }
    return false;
}

// {"address", "entry", "offset", "symbol": {module, name, offset} | null} for a code address
// (null when no lifted function precedes it and no export either).
std::string code_location_json(uint64_t va) {
    const uint64_t entry = nearest_entry(va);
    ExportHit sym;
    const bool has_sym = nearest_export(va, sym);
    if (entry == 0 && !has_sym) {
        return "null";
    }
    std::string o = "{\"address\": " + hex(va) + ", \"entry\": " + (entry != 0 ? hex(entry) : std::string("null")) +
                    ", \"offset\": " + (entry != 0 ? hex(va - entry) : std::string("null")) + ", \"symbol\": ";
    if (has_sym) {
        o += "{\"module\": " + json_string(sym.module) + ", \"name\": " + json_string(sym.name) +
             ", \"offset\": " + hex(va - sym.address) + "}";
    } else {
        o += "null";
    }
    return o + "}";
}

// The compact fault record (BootConfig::fault_record): everything needed to triage a recompiler
// gap from the report alone. Consumed by tools/fault_report.py.
std::string fault_record_json(const FaultRecord& f) {
    const GuestState* regs = f.has_registers ? &f.registers : nullptr;
    const bool code_miss = f.kind == "unknown_target" || f.kind == "unknown_resume_target" || f.kind == "trap";
    std::string o = "{\"kind\": " + json_string(f.kind) + ", \"code_miss\": " + (code_miss ? "true" : "false") +
                    ", \"thread\": " + std::to_string(f.thread_id) +
                    ", \"elapsed_ms\": " + std::to_string(f.elapsed_ms) +
                    ", \"rip\": " + (regs != nullptr ? hex(regs->rip) : std::string("null")) +
                    ", \"target\": " + hex(f.va) + ", \"detail\": " + hex(f.detail) + ", \"present\": " +
                    (f.has_present ? "{\"presents\": " + std::to_string(f.presents) +
                                         ", \"draws_per_frame\": " + std::to_string(f.draws_per_frame) + "}"
                                   : std::string("null")) +
                    ", \"function\": " + code_location_json(f.va);
    // Source of an indirect miss: the call site's return address of the faulting dispatch.
    std::string source = "null";
    if (f.kind == "unknown_target") {
        for (size_t i = f.recent_dispatches.size(); i-- > 0;) {
            if (f.recent_dispatches[i].target == f.va) {
                source = "{\"return_address\": " + hex(f.recent_dispatches[i].return_address) +
                         ", \"location\": " + code_location_json(f.recent_dispatches[i].return_address) + "}";
                break;
            }
        }
    } else if (f.kind == "unknown_resume_target" && regs != nullptr) {
        source = "{\"resume_rip\": " + hex(regs->resume_rip) +
                 ", \"location\": " + code_location_json(regs->resume_rip) + "}";
    }
    o += ", \"source\": " + source;
    if (f.kind == "unimplemented_slot") {
        const std::string slot = com::last_unimplemented_slot_json();
        if (!slot.empty()) {
            o += ", \"unimplemented_slot\": " + slot;
        }
    }
    o += ", \"registers\": ";
    if (regs != nullptr) {
        o += "{";
        for (unsigned i = 0; i < 16; ++i) {
            o += (i == 0 ? "" : ", ") + json_string(kGprNames[i]) + ": " + hex(regs->gpr[i]);
        }
        o += "}";
    } else {
        o += "null";
    }
    o += ", \"call_stack\": {\"method\": " + json_string(f.backtrace_method) + ", \"frames\": [";
    for (size_t i = 0; i < f.frames.size(); ++i) {
        o += (i == 0 ? "" : ", ") + code_location_json(f.frames[i].return_address);
    }
    o += "]}, \"recent_entries\": [";
    for (size_t i = 0; i < f.recent_dispatches.size(); ++i) {
        const DispatchRecord& d = f.recent_dispatches[i];
        const std::string label = dispatch_label(d.target);
        o += (i == 0 ? "" : ", ") + std::string("{\"target\": ") + hex(d.target) + ", \"name\": " +
             (label.empty() ? std::string("null") : json_string(label)) + ", \"from\": " + hex(d.return_address) + "}";
    }
    return o + "]}";
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
        if (b.fault_record) {
            o += "  \"fault_record\": " + fault_record_json(f) + ",\n";
        }
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

    o += "  \"com_calls\": [";
    {
        const std::vector<com::TraceRow> rows = com::trace_report();
        for (size_t i = 0; i < rows.size(); ++i) {
            const com::TraceRow& r = rows[i];
            o += (i == 0 ? "\n    " : ",\n    ");
            o += "{\"interface\": " + json_string(r.interface_name) + ", \"slot\": " +
                 std::to_string(r.slot) + ", \"count\": " + std::to_string(r.count) +
                 ", \"hresult\": " + hex(static_cast<uint32_t>(r.hresult)) + "}";
        }
        o += rows.empty() ? "],\n" : "\n  ],\n";
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
    t_observer.flush_dispatches();
    o += "  \"counters\": {\"dispatches\": " + std::to_string(current_dispatches(b)) +
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
        const std::vector<RecentImport> ordered = collect_recent_imports_locked(b);
        for (size_t k = 0; k < ordered.size(); ++k) {
            o += (k == 0 ? "\n    " : ",\n    ");
            o += "{\"seq\": " + std::to_string(ordered[k].seq) + ", " +
                 import_name_json(b.imports[ordered[k].import_index].import) + ", \"thread\": " +
                 std::to_string(ordered[k].thread) + ", \"args\": [" + hex(ordered[k].args[0]) + ", " +
                 hex(ordered[k].args[1]) + ", " + hex(ordered[k].args[2]) + ", " + hex(ordered[k].args[3]) +
                 ", " + hex(ordered[k].args[4]) + "]}";
        }
        o += ordered.empty() ? "]},\n" : "\n  ]},\n";

        o += bink_json() + ",\n";
        o += "  \"guest_probes\": " + b.guest_probes.to_json() + ",\n";
        o += "  \"audio_probes\": " + b.audio_probes.to_json() + ",\n";
        o += "  \"audio_stream_log\": " + os::audio_stream_log::log().to_json() + ",\n";
        o += "  \"music_controller\": " + music_controller_json(b) + ",\n";
        o += "  \"metadata_registry\": " + metadata_registry_json(b) + ",\n";
        o += "  \"audio_files\": " + os::audio_files::stats().to_json() + ",\n";
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

        // Code the recompiler missed (addresses only; feed back with `recomp build --entry`).
        const auto miss_json = [](const char* name, const MissList& m) {
            std::string j = std::string("\"") + name + "\": {\"count\": " + std::to_string(m.count) +
                            ", \"targets\": [";
            for (size_t k = 0; k < m.targets.size(); ++k) {
                j += (k == 0 ? "" : ", ") + hex(m.targets[k]);
            }
            return j + "]}";
        };
        o += "  \"code_misses\": {" + miss_json("unknown_target", b.miss_unknown_target) + ", " +
             miss_json("unknown_resume_target", b.miss_unknown_resume_target) + ", " +
             miss_json("trap", b.miss_trap) + "},\n";
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
        std::unordered_map<const GuestThreadContext*, RecentImport> merged_thread_last = b.thread_last;
        for (const auto& slot : b.thread_slots) {
            RecentImport latest;
            if (slot->read_latest(latest) && latest.ctx != nullptr) {
                auto it = merged_thread_last.find(latest.ctx);
                if (it == merged_thread_last.end() || latest.seq > it->second.seq) {
                    merged_thread_last[latest.ctx] = latest;
                }
            }
        }
        std::vector<uint32_t> drain_threads;  // threads inside the title's streaming-drain loop
        const bool drain_base_ok = !b.modules.empty() && b.modules[0].loaded && b.modules[0].image.base != 0;
        const uint64_t drain_base = drain_base_ok ? b.modules[0].image.base : 0;
        for (size_t k = 0; k < b.threads.size(); ++k) {
            const GuestThreadContext* t = b.threads[k].get();
            GuestState snapshot = t->thread.state;
            const auto slot_it = b.thread_state_slots.find(t);
            if (slot_it != b.thread_state_slots.end()) {
                const auto& slot = slot_it->second;
                std::lock_guard<std::mutex> slock(slot->exit_mutex);
                if (slot->live_state != nullptr) {
                    snapshot = *slot->live_state;
                } else if (slot->has_exited_state) {
                    snapshot = slot->exited_state;
                }
            }
            const GuestState& s = snapshot;
            o += (k == 0 ? "\n    " : ",\n    ");
            o += "{\"thread_id\": " + std::to_string(t->thread_id) + ", \"teb\": " + hex(t->teb) +
                 ", \"stack_base\": " + hex(t->thread.stack_top) + ", \"stack_limit\": " +
                 hex(t->thread.stack_limit) + ", \"rsp\": " + hex(s.gpr[GuestReg::Rsp]) + ", \"rip\": " +
                 hex(s.rip) + ", \"last_import\": ";
            const auto last = merged_thread_last.find(t);
            if (last != merged_thread_last.end()) {
                o += "{\"seq\": " + std::to_string(last->second.seq) + ", " +
                     import_name_json(b.imports[last->second.import_index].import) + "}";
            } else {
                o += "null";
            }
            FaultRecord scan;
            scan_backtrace(scan, s.gpr[GuestReg::Rsp], t);
            if (drain_base_ok) {
                std::vector<uint64_t> fns;
                for (const auto& f : scan.frames) fns.push_back(f.function);
                if (thread_in_drain(s.rip, fns, drain_base + (kDrainFunctionVa - kDrainLinkBase))) {
                    drain_threads.push_back(t->thread_id);
                }
            }
            o += ", \"stack_scan\": [";
            for (size_t i = 0; i < scan.frames.size() && i < 8; ++i) {
                o += (i == 0 ? "" : ", ") + hex(scan.frames[i].return_address);
            }
            o += "]}";
        }
        o += b.threads.empty() ? "],\n" : "\n  ],\n";
        // The title's "drain all streaming" loop: the counters it waits on and who is inside it.
        if (drain_base_ok && b.arena != nullptr) {
            VirtualMemoryArena* arena = b.arena;
            const StreamingDrainState st = read_streaming_drain(
                [arena](uint64_t va, void* dst, size_t len) { return arena->read(va, dst, len); },
                drain_base + (kDrainSingletonVa - kDrainLinkBase));
            o += "  \"streaming_drain\": " + streaming_drain_json(st, drain_threads) + ",\n";
        } else {
            o += "  \"streaming_drain\": {\"available\": false},\n";
        }
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
             std::to_string(m.imports_runtime) + ", \"unresolved_names\": " + unresolved_names_json(k) +
             "}, \"pdata\": " + (m.pdata ? "true" : "false") +
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
             ", \"run_bytes_moved\": " + std::to_string(c.run_bytes_moved) +
             ", \"alias_slots\": " + std::to_string(c.alias_slots) +
             ", \"alias_conflicts\": " + std::to_string(c.alias_conflicts) + ", \"regions\": [";
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
             std::to_string(w.map_calls) + ", \"write_marks_verified\": " +
             (os::guest_window::write_marks_verified() ? "true" : "false") +
             ", \"write_mark_misses\": " + std::to_string(os::guest_window::verify_misses()) + "}},\n";
    }
    // O36: present pacing (gpu/present_pacing.h).
    {
        const gpu::PresentPacer::Stats ps = gpu::present_pacer().stats();
        o += std::string("  \"present_pacing\": {\"enabled\": ") +
             (gpu::present_pacer().enabled() ? "true" : "false") + ", \"present_interval\": \"" +
             (gpu::present_pacer().honor_interval() ? "honor" : "min1") + "\", \"presents\": " + std::to_string(ps.presents) +
             ", \"immediate\": " + std::to_string(ps.immediate) + ", \"vblank_waits\": " + std::to_string(ps.waits) +
             ", \"wait_ms\": " + std::to_string(ps.wait_us / 1000) +
             ", \"missed_vblanks\": " + std::to_string(ps.missed_vblanks) +
             ", \"last_interval\": " + std::to_string(ps.last_interval) + "},\n";
    }
    // Backlog 37: host texture memory in use against the budget, the size cap in force and what it
    // dropped. Backlog 34: peak per-draw bind-group slots of one recording epoch and budget overflows.
    {
        const gpu::TextureBudgetStats tb = gpu::texture_budget().stats();
        const gpu::d3d12_state::DescriptorLiveCounts dc = gpu::d3d12_state::descriptor_live_counts();
        o += "  \"texture_budget\": {\"budget_mb\": " + std::to_string(tb.budget_bytes >> 20) +
             ", \"used_mb\": " + std::to_string(tb.used_bytes >> 20) +
             ", \"peak_used_mb\": " + std::to_string(tb.peak_used_bytes >> 20) +
             ", \"base_cap\": " + std::to_string(tb.base_cap) + ", \"current_cap\": " + std::to_string(tb.current_cap) +
             ", \"cap_changes\": " + std::to_string(tb.cap_changes) +
             ", \"textures_capped\": " + std::to_string(tb.textures_capped) +
             ", \"mips_dropped\": " + std::to_string(tb.mips_dropped) +
             ", \"mb_not_uploaded\": " + std::to_string(tb.bytes_not_uploaded >> 20) +
             ", \"textures_restored\": " + std::to_string(tb.textures_restored) +
             ", \"bind_group_slots\": {\"budget\": " + std::to_string(dc.bind_group_slot_budget) +
             ", \"peak_per_epoch\": " + std::to_string(dc.peak_bind_groups_per_epoch) +
             ", \"budget_overflows\": " + std::to_string(dc.bind_group_budget_overflows) +
             ", \"draws_dropped\": " + std::to_string(dc.draws_dropped_no_slot) + "}},\n";
        // Backlog 38: small host textures recycled instead of destroyed and recreated.
        const gpu::SmallTexturePoolStats sp = gpu::small_texture_pool().stats();
        o += "  \"small_texture_pool\": {\"max_chain_bytes\": " + std::to_string(gpu::kSmallTextureMaxBytes) +
             ", \"hits\": " + std::to_string(sp.hits) + ", \"misses\": " + std::to_string(sp.misses) +
             ", \"bytes_reused\": " + std::to_string(sp.bytes_reused) +
             ", \"textures_created\": " + std::to_string(sp.creates) +
             ", \"bytes_created\": " + std::to_string(sp.bytes_created) +
             ", \"create_ms\": " + std::to_string(sp.create_ns / 1000000u) +
             ", \"parked\": " + std::to_string(sp.parked_count) +
             ", \"parked_bytes\": " + std::to_string(sp.parked_bytes) +
             ", \"peak_parked_bytes\": " + std::to_string(sp.peak_parked_bytes) +
             ", \"parked_total\": " + std::to_string(sp.parked_total) +
             ", \"evicted\": " + std::to_string(sp.evicted) + ", \"rejected\": " + std::to_string(sp.rejected) +
             ", \"address_misses\": " + std::to_string(sp.address_misses) +
             ", \"address_evictions\": " + std::to_string(sp.address_evictions) +
             ", \"address_recreates\": " + std::to_string(sp.address_recreates) + "},\n";
        // Chains over 64 KiB, pooled only when the consumer zeroes reused textures on the GPU (ClearTexture).
        const gpu::SmallTexturePoolStats lp = gpu::large_texture_pool().stats();
        o += "  \"large_texture_pool\": {\"gpu_clear\": " + std::string(gpu::gpu_texture_clear() ? "true" : "false") +
             ", \"max_chain_bytes\": " + std::to_string(gpu::kLargeTextureMaxBytes) +
             ", \"hits\": " + std::to_string(lp.hits) + ", \"misses\": " + std::to_string(lp.misses) +
             ", \"bytes_reused\": " + std::to_string(lp.bytes_reused) + ", \"parked\": " + std::to_string(lp.parked_count) +
             ", \"parked_bytes\": " + std::to_string(lp.parked_bytes) +
             ", \"peak_parked_bytes\": " + std::to_string(lp.peak_parked_bytes) +
             ", \"parked_total\": " + std::to_string(lp.parked_total) + ", \"evicted\": " + std::to_string(lp.evicted) +
             ", \"rejected\": " + std::to_string(lp.rejected) + "},\n";
    }
    // Backlog 29: committed memory by kind (os/memory_budget.h).
    {
        os::MemoryBudgetInput mb;
        mb.arena_reserved = b.arena != nullptr ? b.arena->reserved_bytes() : 0;
        mb.arena_committed = b.arena != nullptr ? b.arena->committed_bytes() : 0;
        const os::GuestHeapStats hs = os::default_guest_heap().stats();
        mb.guest_heap_committed = hs.committed_bytes;
        mb.guest_heap_live = hs.live_bytes;
        const os::scarlett::XMemMemorySummary xm = os::scarlett::xmem_memory_summary();
        mb.xmem_heap_bytes = xm.heap_allocated_bytes;
        mb.xmem_virtual_committed = xm.virtual_committed_bytes;
        mb.xmem_pages_nominal = xm.physical_pages_bytes;
        mb.xmem_pages_resident = xm.resident_page_bytes;
        mb.xmem_pages_store = xm.store_bytes;
        mb.xmem_pages_unbacked = xm.pages_unbacked;
        mb.xmem_owned_blocks = xm.owned_block_bytes;
        if (const auto device = gpu::d3d12_xs_exports::current_device()) {
            const gpu::d3d12_guest::DeviceStats ds = device->stats();
            mb.shadow_covered = ds.shadow_bytes;
            mb.shadow_resident = ds.shadow_resident_bytes;
        }
        mb.shader_pack_index_bytes = host_shader_pack_index_bytes();
        mb.stage_reflection_bytes = host_stage_reflection_bytes();
        os::process_memory_bytes(&mb.process_private_bytes, &mb.process_working_set_bytes,
                               &mb.process_peak_private_bytes);
        if (b.arena != nullptr) {
            const uint64_t lo = b.arena->guest_base() + b.arena->space_base();
            os::process_commit_map(lo, lo + b.arena->space_size(), &mb.commit_map);
        }
        mb.gpu_texture_bytes = gpu::texture_budget().stats().used_bytes;
        mb.gpu_parked_bytes =
            gpu::small_texture_pool().stats().parked_bytes + gpu::large_texture_pool().stats().parked_bytes;
        o += "  \"memory_budget\": " + os::memory_budget_json(mb) + ",\n";
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
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        o += "  \"frame_times\": " + b.frame_times.to_json() + ",\n";
        o += "  \"frame_breakdown\": " + b.frame_breakdown.to_json() + ",\n";
    }
    o += "  \"guest_sleep\": " + os::host_sleep::guest_sleep_stats_json() + ",\n";
    o += "  \"guest_timed_wait\": " + os::host_sleep::guest_timed_wait_stats_json() + ",\n";
    // Threads parked in a WaitFor* call for over a second, with the state of what they wait on.
    o += "  \"blocked_waits\": " + os::win32::blocked_waits_json(1000) + ",\n";
    // R40: per-present samples without a GPU capture (the last gpu::PresentSampleRing::kDefaultCapacity).
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        o += "  \"present_samples_total\": " + std::to_string(b.present_samples.total()) + ",\n";
        o += "  \"present_samples\": [";
        const std::vector<gpu::PresentSample> samples = b.present_samples.snapshot();
        for (size_t k = 0; k < samples.size(); ++k) {
            const gpu::PresentSample& ps = samples[k];
            o += (k == 0 ? "" : ",");
            o += "{\"index\": " + std::to_string(ps.index) + ", \"wall_ms\": " + std::to_string(ps.wall_ms) +
                 ", \"draws\": " + std::to_string(ps.draws) + ", \"resources\": " + std::to_string(ps.resources) + "}";
        }
        o += "],\n";
    }
    if (!build_info_json().empty()) {
        o += "  \"build\": " + build_info_json() + ",\n";
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
    t_observer.flush_dispatches();
    std::snprintf(line, sizeof(line),
                  "  entry point %s; dispatched calls %" PRIu64 ", import calls %" PRIu64 "\n",
                  b.entry_started ? "started" : "not started", current_dispatches(b),
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
        const std::vector<RecentImport> ordered = collect_recent_imports_locked(b);
        if (!ordered.empty()) {
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
    if (b.watch.active()) {
        char head[96];
        std::snprintf(head, sizeof(head), "[watch] t=%" PRIu64 " thread=%u at fault", f.elapsed_ms, f.thread_id);
        if (b.log_progress) {
            os::log(std::string(head) + watch_snapshot(b));
        }
    }
    if (b.fault_record) {
        // Compact sanitized lines next to the trap line; tools/fault_report.py prints the full record.
        ExportHit sym;
        const bool has_sym = nearest_export(f.va, sym);
        char line[512];
        std::snprintf(line, sizeof(line),
                      "[fault] %s rip=0x%" PRIx64 " target=0x%" PRIx64 " fn=%s%s%s thread=%u t=%" PRIu64 "ms",
                      f.kind.c_str(), f.has_registers ? f.registers.rip : uint64_t{0}, f.va,
                      function_text(f.va).c_str(), has_sym ? " sym=" : "", has_sym ? sym.name.c_str() : "",
                      f.thread_id, f.elapsed_ms);
        std::string out = line;
        if (f.has_present) {
            out += " presents=" + std::to_string(f.presents) + " draws/frame=" + std::to_string(f.draws_per_frame);
        }
        if (!f.frames.empty()) {
            out += " caller=" + function_text(f.frames.front().return_address);
            constexpr size_t kMaxCallers = 4;
            const size_t caller_count = std::min(f.frames.size(), kMaxCallers);
            out += " callers=";
            for (size_t i = 0; i < caller_count; ++i) {
                if (i != 0) {
                    out += ",";
                }
                out += function_text(f.frames[i].return_address);
            }
        }
        if (b.log_progress) {
            os::log(out);
        }
        if (f.has_registers) {
            const GuestState& r = f.registers;
            logf("[fault-regs] rax=0x%" PRIx64 " rcx=0x%" PRIx64 " rdx=0x%" PRIx64 " r8=0x%" PRIx64
                 " r9=0x%" PRIx64 " rsp=0x%" PRIx64 " rbx=0x%" PRIx64 " rsi=0x%" PRIx64 " rdi=0x%" PRIx64,
                 r.gpr[GuestReg::Rax], r.gpr[GuestReg::Rcx], r.gpr[GuestReg::Rdx], r.gpr[GuestReg::R8],
                 r.gpr[GuestReg::R9], r.gpr[GuestReg::Rsp], r.gpr[GuestReg::Rbx], r.gpr[GuestReg::Rsi],
                 r.gpr[GuestReg::Rdi]);
        }
    }
    if (b.log_progress) {
        std::fflush(stderr);
    }
}

// One compact "[fault-record] {...}" log line, emitted AFTER the report chunks: a browser run keeps only the
// tail of the log (the HUD cap drops the head of the chunked report, which is where the fault lives), so
// this line is what lets the runner name the fault (kind, function, import, unimplemented COM slot).
void log_fault_brief() {
    BootState& b = boot();
    const FaultRecord& f = b.fault;
    if (!f.present || !b.fault_record || !b.log_progress) {
        return;
    }
    // The kind's first character is \u-escaped (still valid JSON): web/telemetry/boot_tracer.js turns any
    // wasm-tagged line with the bare word "trap" (kind "trap") into a fatal WASM_TRAP, which would
    // overwrite the real trap line.
    char lead[8];
    std::snprintf(lead, sizeof(lead), "\\u%04x",
                  f.kind.empty() ? 0u : static_cast<unsigned>(static_cast<unsigned char>(f.kind[0])));
    const std::string kind_json = "\"" + std::string(lead) + (f.kind.empty() ? std::string() : f.kind.substr(1)) + "\"";
    std::string o = "{\"kind\": " + kind_json + ", \"va\": " + hex(f.va) + ", \"detail\": " + hex(f.detail) +
                    ", \"rip\": " + (f.has_registers ? hex(f.registers.rip) : std::string("null")) +
                    ", \"thread\": " + std::to_string(f.thread_id) +
                    ", \"elapsed_ms\": " + std::to_string(f.elapsed_ms) +
                    ", \"function\": " + function_json(f.va) + ", \"import\": ";
    if (f.import_index >= 0) {
        const ImportRecord& rec = b.imports[static_cast<size_t>(f.import_index)];
        o += "{" + import_name_json(rec.import) + ", \"module\": " + json_string(b.modules[rec.module].name) + "}";
    } else {
        o += "null";
    }
    o += ", \"unimplemented_slot\": ";
    const std::string slot = f.kind == "unimplemented_slot" ? com::last_unimplemented_slot_json() : std::string();
    o += slot.empty() ? std::string("null") : slot;
    o += ", \"callers\": [";
    const size_t callers = std::min<size_t>(f.frames.size(), 4);
    for (size_t i = 0; i < callers; ++i) {
        o += (i == 0 ? "" : ", ") + code_location_json(f.frames[i].return_address);
    }
    o += "]";
    // Locating info, same keys as the report's fault_record (tools/fault_report.py renders either), bounded:
    // the source site of the indirect call, the 16 GPRs, up to 16 guest stack frames (unwind info, else a
    // stack scan) and the faulting thread's last (up to 64) dispatches as bare target/from pairs.
    const GuestState* regs = f.has_registers ? &f.registers : nullptr;
    std::string source = "null";
    if (f.kind == "unknown_target") {
        for (size_t i = f.recent_dispatches.size(); i-- > 0;) {
            if (f.recent_dispatches[i].target == f.va) {
                source = "{\"return_address\": " + hex(f.recent_dispatches[i].return_address) +
                         ", \"location\": " + code_location_json(f.recent_dispatches[i].return_address) + "}";
                break;
            }
        }
    } else if (f.kind == "unknown_resume_target" && regs != nullptr) {
        source = "{\"resume_rip\": " + hex(regs->resume_rip) +
                 ", \"location\": " + code_location_json(regs->resume_rip) + "}";
    }
    o += ", \"source\": " + source + ", \"registers\": ";
    if (regs != nullptr) {
        o += "{";
        for (unsigned i = 0; i < 16; ++i) {
            o += (i == 0 ? "" : ", ") + json_string(kGprNames[i]) + ": " + hex(regs->gpr[i]);
        }
        o += "}";
    } else {
        o += "null";
    }
    o += ", \"call_stack\": {\"method\": " + json_string(f.backtrace_method) + ", \"frames\": [";
    const size_t nframes = std::min<size_t>(f.frames.size(), 16);
    for (size_t i = 0; i < nframes; ++i) {
        o += (i == 0 ? "" : ", ") + code_location_json(f.frames[i].return_address);
    }
    o += "]}, \"recent_entries\": [";
    for (size_t i = 0; i < f.recent_dispatches.size(); ++i) {
        const DispatchRecord& d = f.recent_dispatches[i];
        o += (i == 0 ? "" : ", ") + std::string("{\"target\": ") + hex(d.target) + ", \"from\": " +
             hex(d.return_address) + "}";
    }
    o += "]}";
    os::log("[fault-record] " + o);
}

void finish_report() {
    BootState& b = boot();
    t_observer.flush_indirect_targets();
    if (b.indirect_log) {
        b.indirect_log->flush();
    }
    b.last_report = build_report();
    b.last_summary = build_summary();
    if (b.report_sink) {
        b.report_sink(b.last_report);
    }
    log_fault_brief();
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
    t_observer.flush_dispatches();
    st.dispatches = current_dispatches(b);
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
    bool found = false;
    RecentImport entry{};
    for (const auto& [thread, rec] : b.thread_last) {
        if (thread->thread_id == thread_id && (!found || rec.seq > entry.seq)) {
            entry = rec;
            found = true;
        }
    }
    for (const auto& slot : b.thread_slots) {
        if (slot->thread_id.load(std::memory_order_relaxed) != thread_id) {
            continue;
        }
        RecentImport latest;
        if (slot->read_latest(latest) && latest.thread == thread_id && (!found || latest.seq > entry.seq)) {
            entry = latest;
            found = true;
        }
    }
    if (!found || entry.import_index >= b.imports.size()) {
        return {};
    }
    const BootImport& imp = b.imports[entry.import_index].import;
    std::string label = import_label(imp);
    // Waits and sleeps: the guest caller and the handle / time argument, so a profile
    // says which wait a blocked thread is in (--profile).
    const std::string symbol = import_symbol(imp);
    const bool sleep = symbol == "Sleep" || symbol == "SleepEx";
    const bool wait_one = symbol == "WaitForSingleObject" || symbol == "WaitForSingleObjectEx";
    const bool wait_many = symbol == "WaitForMultipleObjects" || symbol == "WaitForMultipleObjectsEx";
    if (sleep || wait_one || wait_many) {
        char text[96];
        if (sleep) {
            std::snprintf(text, sizeof(text), " ms=%" PRIu64, entry.args[0] & 0xFFFFFFFFu);
        } else if (wait_one) {
            std::snprintf(text, sizeof(text), " h=0x%" PRIx64 " ms=%" PRId32, entry.args[0],
                          static_cast<int32_t>(entry.args[1]));
        } else {
            std::snprintf(text, sizeof(text), " n=%" PRIu64 " ms=%" PRId32, entry.args[0] & 0xFFFFFFFFu,
                          static_cast<int32_t>(entry.args[3]));
        }
        label += text;
        std::snprintf(text, sizeof(text), " ret 0x%" PRIx64, entry.return_address);
        label += text;
    }
    return label;
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
    t_observer.flush_dispatches();
    BootState& b = boot();
    if (previous != nullptr && previous != context && t_observer.slot != nullptr &&
        t_observer.generation == b.generation.load(std::memory_order_acquire)) {
        RecentImport latest;
        if (t_observer.slot->read_latest(latest) && latest.ctx == previous) {
            std::lock_guard<std::mutex> lock(b.mutex);
            auto it = b.thread_last.find(previous);
            if (it == b.thread_last.end() || latest.seq > it->second.seq) {
                b.thread_last[previous] = latest;
            }
        }
    }
    t_active = context;
    uint32_t* slot = nullptr;
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
    if (!b.running.load(std::memory_order_acquire) || !t_running_boot) {
        return false;
    }
    t_observer.flush_dispatches();
    b.status = BootStatus::Exited;
    b.exit_code = exit_code;
    b.has_exit_code = true;
    b.message = "guest requested process exit (" + std::to_string(exit_code) + ")";
    stop_boot();
}

void request_boot_stop() { g_stop_requested.store(true, std::memory_order_relaxed); }

bool report_host_fault(const char* kind, uint64_t host_address, uint64_t host_pc) {
    BootState& b = boot();
    if (!b.active) {
        return false;
    }
    // The faulting thread may hold the arena lock (a host fault inside a held guest view): the
    // backtrace and report reads below must not block on it.
    const VirtualMemoryArena::DiagnosticReads diagnostic_reads;
    t_observer.flush_dispatches();
    const char* name = kind != nullptr ? kind : "host_fault";
    if (b.fault.present && (std::strcmp(name, "host_terminate") == 0 || std::strcmp(name, "host_abort") == 0)) {
        // The guest fault was already recorded (the unwind that should have stopped the boot hit a
        // noexcept frame or the CRT aborted): keep that record and write the report.
        b.message += std::string(" [process terminated while unwinding: ") + name + "]";
        log_trap();
        finish_report();
        return true;
    }
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

void set_fault_hook(FaultHookFn hook) {
    g_fault_hook = hook;
}

std::string last_report_json() {
    return boot().last_report;
}

namespace {

// The value after `"key":` in a flat JSON object; empty if absent.
std::string_view json_value(std::string_view json, std::string_view key) {
    const std::string quoted = "\"" + std::string(key) + "\"";
    size_t at = json.find(quoted);
    if (at == std::string_view::npos) {
        return {};
    }
    at = json.find(':', at + quoted.size());
    if (at == std::string_view::npos) {
        return {};
    }
    ++at;
    while (at < json.size() && (json[at] == ' ' || json[at] == '\t' || json[at] == '\n' || json[at] == '\r')) {
        ++at;
    }
    size_t end = at;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != '\n') {
        ++end;
    }
    while (end > at && (json[end - 1] == ' ' || json[end - 1] == '\r' || json[end - 1] == '\t')) {
        --end;
    }
    return json.substr(at, end - at);
}

bool json_uint(std::string_view json, std::string_view key, uint64_t& out) {
    const std::string_view v = json_value(json, key);
    if (v.empty()) {
        return false;
    }
    uint64_t n = 0;
    for (const char c : v) {
        if (c < '0' || c > '9') {
            return false;
        }
        const uint64_t digit = static_cast<uint64_t>(c - '0');
        if (n > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        n = n * 10 + digit;
    }
    out = n;
    return true;
}

bool json_uint32(std::string_view json, std::string_view key, uint32_t& out) {
    uint64_t n = 0;
    if (!json_uint(json, key, n) || n > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(n);
    return true;
}

bool json_string(std::string_view json, std::string_view key, std::string& out) {
    const std::string quoted = "\"" + std::string(key) + "\"";
    size_t search_from = 0;
    while (search_from < json.size()) {
        const size_t key_pos = json.find(quoted, search_from);
        if (key_pos == std::string_view::npos) {
            return false;
        }
        size_t at = key_pos + quoted.size();
        while (at < json.size() && (json[at] == ' ' || json[at] == '\t' || json[at] == '\n' || json[at] == '\r')) {
            ++at;
        }
        if (at >= json.size() || json[at] != ':') {
            search_from = key_pos + quoted.size();
            continue;
        }
        ++at;
        while (at < json.size() && (json[at] == ' ' || json[at] == '\t' || json[at] == '\n' || json[at] == '\r')) {
            ++at;
        }
        if (at >= json.size() || json[at] != '"') {
            return false;
        }
        ++at;
        std::string decoded;
        while (at < json.size()) {
            const char c = json[at++];
            if (c == '"') {
                out = std::move(decoded);
                return true;
            }
            if (static_cast<unsigned char>(c) < 0x20u) {
                return false;
            }
            if (c != '\\') {
                decoded.push_back(c);
                continue;
            }
            if (at >= json.size()) {
                return false;
            }
            const char esc = json[at++];
            switch (esc) {
                case '"':
                    decoded.push_back('"');
                    break;
                case '\\':
                    decoded.push_back('\\');
                    break;
                case '/':
                    decoded.push_back('/');
                    break;
                case 'b':
                    decoded.push_back('\b');
                    break;
                case 'f':
                    decoded.push_back('\f');
                    break;
                case 'n':
                    decoded.push_back('\n');
                    break;
                case 'r':
                    decoded.push_back('\r');
                    break;
                case 't':
                    decoded.push_back('\t');
                    break;
                case 'u': {
                    if (at + 4 > json.size()) {
                        return false;
                    }
                    uint32_t cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = json[at++];
                        uint32_t nibble = 0;
                        if (h >= '0' && h <= '9') {
                            nibble = static_cast<uint32_t>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            nibble = 10u + static_cast<uint32_t>(h - 'a');
                        } else if (h >= 'A' && h <= 'F') {
                            nibble = 10u + static_cast<uint32_t>(h - 'A');
                        } else {
                            return false;
                        }
                        cp = (cp << 4) | nibble;
                    }
                    if (cp < 0x80u) {
                        decoded.push_back(static_cast<char>(cp));
                    } else if (cp < 0x800u) {
                        decoded.push_back(static_cast<char>(0xC0u | (cp >> 6)));
                        decoded.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
                    } else {
                        decoded.push_back(static_cast<char>(0xE0u | (cp >> 12)));
                        decoded.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
                        decoded.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
                    }
                    break;
                }
                default:
                    return false;
            }
        }
        return false;
    }
    return false;
}

}  // namespace

bool apply_boot_options_json(std::string_view json, BootConfig& config) {
    size_t first = json.find_first_not_of(" \t\r\n");
    size_t last = json.find_last_not_of(" \t\r\n");
    if (first == std::string_view::npos || json[first] != '{' || json[last] != '}') {
        return false;
    }
    uint64_t n = 0;
    if (json_uint(json, "max_wall_ms", n)) {
        config.max_wall_ms = n;
    }
    if (json_uint(json, "max_dispatches", n)) {
        config.max_dispatches = n;
    }
    if (json_uint(json, "arena_log_limit", n)) {
        config.arena_log_limit = static_cast<size_t>(n);
    }
    if (json_uint(json, "import_trace_limit", n)) {
        config.import_trace_limit = static_cast<size_t>(n);
    }
    uint32_t u32 = 0;
    if (json_uint32(json, "fence_publish_lag_frames", u32)) {
        config.fence_publish_lag_frames = u32;
    }
    if (json_uint32(json, "fence_publish_cap_ms", u32)) {
        config.fence_publish_cap_ms = u32;
    }
    if (json_uint32(json, "gpu_memory_budget_mb", u32)) {
        config.gpu_memory_budget_mb = u32;
    }
    if (json_uint32(json, "texture_size_cap", u32)) {
        config.texture_size_cap = u32;
    }
    if (json_uint32(json, "bind_group_slot_budget", u32)) {
        config.bind_group_slot_budget = u32;
    }
    if (json_uint32(json, "present_max_interval", u32)) {
        config.present_max_interval = u32;
    }
    const std::string_view verify = json_value(json, "verify_window_marks");
    if (verify == "true" || verify == "false") {
        config.verify_window_marks = verify == "true";
    }
    const std::string_view pacing = json_value(json, "present_pacing");
    if (pacing == "true" || pacing == "false") {
        config.present_pacing = pacing == "true";
    }
    std::string pace_interval;
    if (json_string(json, "present_interval", pace_interval) &&
        (pace_interval == "honor" || pace_interval == "min1")) {
        config.present_honor_interval = pace_interval == "honor";
    }
    std::string dump_dir;
    if (json_string(json, "dump_pipelines_dir", dump_dir) && !dump_dir.empty()) {
        config.dump_pipelines_dir = std::move(dump_dir);
    }
    const auto read_bool = [&](const char* key, bool& out) {
        const std::string_view v = json_value(json, key);
        if (v == "true" || v == "false") {
            out = v == "true";
        }
    };
    read_bool("fence_publish_lag", config.fence_publish_lag);
    read_bool("defer_gpu_objects", config.defer_gpu_objects);
    read_bool("xmem_shared_pages", config.xmem_shared_pages);
    read_bool("present_frame_waits", config.present_frame_waits);
    read_bool("audio_endpoint", config.audio_endpoint);
    const std::string_view to_log = json_value(json, "report_to_log");
    if (to_log == "true") {
        config.report_sink = [](std::string_view report) { log_report_json(report); };
    } else if (to_log == "false") {
        config.report_sink = nullptr;
    }
    return true;
}

void log_report_json(std::string_view report_json,
                     size_t chunk_bytes,
                     const std::function<void(std::string_view)>& write) {
    if (chunk_bytes < 8) {
        chunk_bytes = 8;
    }
    std::vector<std::string_view> chunks;
    size_t at = 0;
    while (at < report_json.size()) {
        size_t end = std::min(report_json.size(), at + chunk_bytes);
        // Never split a UTF-8 sequence: back off to the lead byte of the sequence at `end`.
        while (end < report_json.size() && end > at + 1 &&
               (static_cast<unsigned char>(report_json[end]) & 0xC0u) == 0x80u) {
            --end;
        }
        chunks.push_back(report_json.substr(at, end - at));
        at = end;
    }
    const size_t n = chunks.size();
    for (size_t i = 0; i < n; ++i) {
        std::string line = "[boot-report " + std::to_string(i + 1) + "/" + std::to_string(n) + "] ";
        line.append(chunks[i]);
        if (write) {
            write(line);
        } else {
            os::log(line);
        }
    }
}

bool load_boot_options(BootConfig& config, const BootFileReader& read, const std::string& path) {
    std::vector<uint8_t> bytes;
    if (!read || !read(path, bytes)) {
        return true;  // no options file: run as configured
    }
    const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    const bool ok = apply_boot_options_json(text, config);
    os::log(ok ? "[boot] BOOT_OPTIONS " + path + " applied" : "[boot] BOOT_OPTIONS " + path + " ignored (not a JSON object)");
    return ok;
}

void set_build_info_json(std::string json) {
    build_info_json() = std::move(json);
}

void set_report_extension(ReportExtension extension) {
    g_report_extension.store(extension, std::memory_order_release);
}

void set_frame_counter_source(FrameCounterSource source) {
    g_frame_counter_source.store(source, std::memory_order_release);
}

// The present pacer's observer: every PresentX records a sample (R40); the first marks first_present.
static void record_present() {
    namespace d3d = gpu::d3d12_guest;
    const uint64_t draws = d3d::work_total(d3d::Work::DrawsIssued);
    const uint64_t resources = d3d::work_total(d3d::Work::ResourcesCreated);
    // The interval the previous present asked for (its wait is what this gap contains).
    const gpu::PresentPacer::Stats ps = gpu::present_pacer().stats();
    const uint32_t gap = ps.presents == 0 || ps.last_interval == 0
                             ? 0u
                             : gpu::PresentPacer::title_gap_vblanks(ps.last_interval, gpu::present_pacer().honor_interval());
    const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now().time_since_epoch()).count();
    // frame_breakdown: the present thread's cumulative clocks (sampled before the lock).
    gpu::FrameSample fs;
    fs.thread_key = static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    fs.thread_cpu_ns = os::this_thread_cpu_ns();
    fs.spans = os::this_thread_spans();
    fs.process_cpu_ns = os::process_cpu_ns();
    if (const FrameCounterSource src = g_frame_counter_source.load(std::memory_order_acquire)) {
        src(&fs.executor_ns, &fs.submitted_bytes);
    }
    fs.texture_uploads = d3d::work_total(d3d::Work::TextureUploads);
    fs.resources_created = resources;
    fs.shadow_compared_bytes = d3d::work_total(d3d::Work::ShadowComparedBytes);
    fs.shadow_full_bytes = d3d::work_total(d3d::Work::ShadowFullCompareBytes);
    {
        BootState& b = boot();
        std::lock_guard<std::mutex> lock(b.mutex);
        b.present_samples.record(elapsed_ms(), draws, resources);
        if (b.last_present_us >= 0 && now_us >= b.last_present_us) {
            b.frame_times.add(static_cast<uint64_t>(now_us - b.last_present_us), b.present_samples.last().draws, gap,
                              elapsed_ms());
            b.frame_breakdown.add(static_cast<uint64_t>(now_us - b.last_present_us) * 1000u,
                                  b.present_samples.last().draws, fs, elapsed_ms());
            gpu::d3d12_guest::set_current_gpu_phase(
                gpu::d3d12_guest::gpu_phase_for_draws(b.present_samples.last().draws));
        }
        b.last_present_us = now_us;
    }
    watch_at_present(boot());
    mark_first_present();
}

uint64_t boot_elapsed_ms() { return elapsed_ms(); }

uint64_t boot_present_count() {
    BootState& b = boot();
    std::lock_guard<std::mutex> lock(b.mutex);
    return b.present_samples.total();
}

void mark_first_present() {
    BootState& b = boot();
    uint64_t seq = 0;
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        if (b.present_seen) {
            return;
        }
        b.present_seen = true;
        b.present_seen_atomic.store(true, std::memory_order_release);
        b.present_import_seq = seq = b.import_calls.load(std::memory_order_relaxed);
        b.present_elapsed_ms = elapsed_ms();
    }
    milestone("FIRST_PRESENT", "first PresentX (import call #%" PRIu64 ")", seq);
}

std::string last_summary() {
    return boot().last_summary;
}

PresentStatus present_status() {
    BootState& b = boot();
    std::lock_guard<std::mutex> lock(b.mutex);
    const gpu::PresentSample last = b.present_samples.last();
    PresentStatus s;
    s.presents = b.present_samples.total();
    s.draws_per_frame = last.draws;
    s.wall_ms = last.wall_ms;
    return s;
}

LiveCounters live_counters() {
    BootState& b = boot();
    t_observer.flush_dispatches();
    LiveCounters c;
    c.dispatches = current_dispatches(b);
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
    b.running.store(false, std::memory_order_release);
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
    b.synthetic_to_import.clear();
    b.import_counts.reset();
    b.imports_bound = 0;
    b.imports_unresolved = 0;
    b.bink.clear();
    b.bink_present = false;
    b.bink_exports_lifted = 0;
    b.bink_exports_shim = 0;
    b.audio_probes.clear();
    b.guest_probes.clear();
    b.music_controller = 0;
    os::audio_stream_log::log().reset();
    os::audio_stream_log::log().set_sources(&boot_elapsed_ms, &boot_present_count);
    os::audio_files::stats().reset();
    b.process_reservation = 0;
    b.peb = 0;
    b.process_parameters = 0;
    b.tls_modules.clear();
    b.threads.clear();
    b.main_thread = nullptr;
    {
        std::lock_guard<std::mutex> lock(b.mutex);
        if (b.dispatch_target) {
            b.dispatch_target->active.store(false, std::memory_order_release);
        }
        b.dispatch_target = std::make_shared<DispatchBatchTarget>();
        b.thread_slots.clear();
        b.present_seen = false;
        b.present_seen_atomic.store(false, std::memory_order_release);
        b.generation.fetch_add(1, std::memory_order_release);
    }
    t_observer.reset();
    b.import_calls.store(0);
    b.recent.clear();
    b.recent_next = 0;
    b.survey_called.clear();
    b.miss_unknown_target = {};
    b.miss_unknown_resume_target = {};
    b.miss_trap = {};
    b.tls_callbacks_run = 0;
    b.entry_started = false;
    b.milestones.clear();
    b.first_calls.clear();
    b.pre_present_counts.clear();
    b.present_import_seq = 0;
    b.present_elapsed_ms = 0;
    b.present_samples.clear();
    b.frame_times.clear();
    b.frame_breakdown.clear();
    gpu::d3d12_guest::set_current_gpu_phase(0);
    b.last_present_us = -1;
    b.trace.clear();
    b.trace_strings.clear();
    {
        std::lock_guard<std::mutex> lock(b.callable_mutex);
        b.callable_calls.clear();
    }
    b.trace_limit = 0;
    b.trace_truncated = false;
    b.thread_last.clear();
    b.thread_state_slots.clear();
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
    b.fault_record = config.fault_record;
    b.watch.configure(config.watch_guest_va);
    b.watch_last_present.clear();
    b.watch_presents.store(0, std::memory_order_relaxed);
    for (const WatchRange& r : b.watch.ranges()) {
        if (r.indirect) {
            logf("[watch] watching guest *0x%" PRIx64 "+0x%" PRIx64 " (%" PRIu64 " bytes)", r.va, r.offset, r.len);
        } else {
            logf("[watch] watching guest 0x%" PRIx64 "..0x%" PRIx64, r.va, r.va + r.len);
        }
    }
    b.indirect_log = config.record_indirect_targets.empty()
                         ? nullptr
                         : std::make_shared<IndirectTargetLog>(config.record_indirect_targets);
    b.capture_host_stack = config.capture_host_stack;
    b.trace_limit = config.import_trace_limit;
    b.previous_arena_observer = os::set_arena_observer(&boot_arena_observer);
    b.arena_observer_installed = true;
    b.host_function_start = config.host_function_start;
    b.off_thread_fault = config.off_thread_fault;
    b.report_sink = config.report_sink;
    os::guest_window::set_write_marks_verified(config.verify_window_marks);
    gpu::present_pacer().set_enabled(config.present_pacing);
    gpu::present_pacer().set_honor_interval(config.present_honor_interval);
    {
        // No budget configured: 60% of the GPU's dedicated memory (textures share it with buffers, render
        // targets, Dawn's descriptor heaps and the desktop; without a budget the in-world scene grew to
        // 7.2 GB of textures on a 12 GB card and the device was lost in CreateDescriptorHeap, E_OUTOFMEMORY).
        uint64_t budget = uint64_t{config.gpu_memory_budget_mb} << 20;
        if (budget == 0) {
            const uint64_t vram = os::dedicated_video_memory_bytes();
            budget = vram / 10 * 6;
            if (budget != 0) {
                std::fprintf(stderr, "[gpu] texture budget: %llu MiB (60%% of %llu MiB dedicated video memory)\n",
                             static_cast<unsigned long long>(budget >> 20), static_cast<unsigned long long>(vram >> 20));
            }
        }
        gpu::texture_budget().set_budget_bytes(budget);
    }
    gpu::texture_budget().set_base_cap(config.texture_size_cap);
    gpu::d3d12_state::set_bind_group_slot_budget(config.bind_group_slot_budget);
    gpu::d3d12_guest::xbox::set_present_max_interval(config.present_max_interval);
    // A switched-off defer_gpu_objects disables the harness setting; the fence publish lag (off by
    // default since #663), its tuning fields, and any non-empty dump_pipelines_dir from
    // boot-options.json are wired through from BootConfig before the guest creates its D3D12 device.
    {
        gpu::d3d12_guest::DeviceOptions device_options = gpu::d3d12_xs_exports::device_options();
        const bool lag_changed = config.fence_publish_lag != device_options.fence_publish_lag;
        const bool defer_changed = !config.defer_gpu_objects && device_options.defer_gpu_objects;
        const bool lag_frames_changed = device_options.fence_publish_lag_frames != config.fence_publish_lag_frames;
        const bool lag_cap_changed = device_options.fence_publish_cap_ms != config.fence_publish_cap_ms;
        const bool dump_dir_changed =
            !config.dump_pipelines_dir.empty() && device_options.dump_dir != config.dump_pipelines_dir;
        if (lag_changed || defer_changed || lag_frames_changed || lag_cap_changed || dump_dir_changed) {
            device_options.fence_publish_lag = config.fence_publish_lag;
            device_options.defer_gpu_objects = device_options.defer_gpu_objects && config.defer_gpu_objects;
            device_options.fence_publish_lag_frames = config.fence_publish_lag_frames;
            device_options.fence_publish_cap_ms = config.fence_publish_cap_ms;
            if (!config.dump_pipelines_dir.empty()) {
                device_options.dump_dir = config.dump_pipelines_dir;
            }
            gpu::d3d12_xs_exports::set_device_options(std::move(device_options));
        }
        if (dump_dir_changed && b.log_progress) {
            os::log("[boot] PIPELINE_DUMP: " + config.dump_pipelines_dir);
        }
    }
    if (!config.xmem_shared_pages) {
        os::win32::set_process_shared_zone(false);
    }
    if (!config.present_frame_waits) {
        gpu::d3d12_guest::xbox::set_present_token_waits(false);
    }
    if (!config.audio_endpoint) {
        os::scarlett::guest_api::set_audio_endpoint_enabled(false);
    }
    // Every present marks the first one (report first_present, FIRST_PRESENT milestone), with or
    // without a GPU capture.
    gpu::present_pacer().set_observer(&record_present);
    b.guest_thread_started = config.guest_thread_started;
    {
        std::lock_guard<std::mutex> lock(b.arena_mutex);
        b.arena_log_limit = config.arena_log_limit;
    }
    b.fault_gate.reset();
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
    // Build O(1) synthetic-index -> import-index table once so boot_dispatch_observer avoids
    // describe_synthetic() + binary search on every static IAT call.
    b.synthetic_to_import.clear();
    const size_t synth_cap = b.imports.size() + 1024;
    for (size_t i = 0; i < b.imports.size(); ++i) {
        uint64_t synth_va = 0;
        if (read_u64(*b.arena, b.imports[i].image_base + b.imports[i].import.slot_rva, synth_va) &&
            is_synthetic_address(synth_va) && ((synth_va - kSyntheticBase) & (kSyntheticStride - 1)) == 0) {
            const size_t synth_idx = static_cast<size_t>((synth_va - kSyntheticBase) / kSyntheticStride);
            if (synth_idx < synth_cap) {
                if (synth_idx >= b.synthetic_to_import.size()) {
                    b.synthetic_to_import.resize(synth_idx + 1, -1);
                }
                b.synthetic_to_import[synth_idx] = static_cast<long>(i);
            }
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
    // After after_load: the guest modules and their exports enter the module registry there, so the
    // lifted/shim classification of Bink2Scarlett.dll's exports is only meaningful from here on.
    setup_bink_counters(b);
    setup_audio_probes(b);
    setup_guest_probes(b, config);
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
    b.running.store(true, std::memory_order_release);
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
    b.running.store(false, std::memory_order_release);
    remove_hooks();
    log_trap();
    if (!config.dump_pipelines_dir.empty()) {
        if (auto dev = gpu::d3d12_xs_exports::current_device()) {
            dev->flush_pipeline_dump();
        }
    }
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

// Entry counter of a function listed in a build's `--probe-functions` file: the lifter calls this
// at the entry of each listed function, so direct calls count too. Not emitted without such a list.
extern "C" void __playgta6_probe_hit(playgta6::GuestState* state, uint64_t va) {
    using namespace playgta6::guest;
    BootState& b = boot();
    if (b.watch.active() && b.log_progress && b.watch.take_entry(va) && b.watch.take_line()) {
        // Entry order across threads, with what the watched ranges hold at that moment.
        char head[160];
        std::snprintf(head, sizeof(head), "[watch] t=%" PRIu64 " thread=%u enter sub_%" PRIX64 " rcx=0x%" PRIx64
                      " rdx=0x%" PRIx64, elapsed_ms(), t_active != nullptr ? t_active->thread_id : 0u, va,
                      state->gpr[playgta6::GuestReg::Rcx], state->gpr[playgta6::GuestReg::Rdx]);
        playgta6::os::log(std::string(head) + watch_snapshot(b));  // longer than logf's buffer
    }
    if (b.guest_probes.active()) {
        guest_probe_hit(b, state, b.guest_probes.find(va), va, true);
    }
    if (!b.audio_probes.active()) {
        return;
    }
    const uint64_t args[AudioProbes::kArgs] = {state->gpr[playgta6::GuestReg::Rcx], state->gpr[playgta6::GuestReg::Rdx],
                                               state->gpr[playgta6::GuestReg::R8], state->gpr[playgta6::GuestReg::R9]};
    bool first = false;
    const long idx = b.audio_probes.entry_hit(va, args, &first);
    audio_probe_event(b, idx, args, true);
    if (first && idx >= 0 && b.log_progress) {
        playgta6::os::log(("[audio-probe] first entry of " + b.audio_probes.name(static_cast<size_t>(idx))).c_str());
    }
}

extern "C" void __playgta6_probe_exit(playgta6::GuestState* state, uint64_t va) {
    using namespace playgta6::guest;
    BootState& b = boot();
    if (b.guest_probes.active()) {
        guest_probe_exit(b, state, va);
    }
}

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
