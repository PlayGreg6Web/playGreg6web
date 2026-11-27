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
//   --present-interval honor|min1  title-driven present gap: honor = the title's sync interval (30 fps menus like the console), min1 = 1 vblank (default)
//   --no-present-pacing  presents do not wait for the 60 Hz vblank (benchmarks; gpu/present_pacing.h)
//   --no-xmem-shared-pages  physical pages mapped at several addresses get per-address copies
//                        (the pre-O36 behaviour; A/B runs, os/xbox_scarlett.h)
//   --stack <bytes>      guest stack size (default: the image's SizeOfStackReserve)
//   --host-stack-mb <n>  host stack of the boot thread (default 4096: lifted guest calls nest
//                        host calls, so deep guest recursion needs a deep host stack)
//   --report <file>      JSON report path (default boot-report.json)
//   --summary <file>     also write the text summary there (it always goes to stderr)
//   --import-trace <file> write every import call (sequence, thread, DLL!symbol, rcx rdx r8 r9
//                        [rsp+0x28]) there, up to --import-trace-limit calls (default 200000)
//   --dstorage-workers <n>  DirectStorage (XCRDAPI) worker threads, 1..16 (default 1)
//   --bcpack-tables <file>  BCPack format tables for DirectStorage texture requests (default: the
//                        PLAYGTA6_BCPACK_TABLES environment variable; none = BCPack requests fail
//                        E_NOTIMPL). User-supplied, never shipped: docs/BCPACK.md
//   --console-mode <n>   console-mode registry value the title reads to pick its console type
//                        (default 0x60 = Xbox Series X; 0 = leave it unset)
//   --sandbox-id <text>  Xbox Live sandbox the emulated console reports (default RETAIL, a retail
//                        console; "" = none: XSystemGetXboxLiveSandboxId fails with E_NOTIMPL)
//   --user-age-group <unknown|child|teen|adult>  the local user's age group XUserGetAgeGroup
//                        reports (default unknown: no account data offline)
//   --save-dir <dir>     host directory of the local XGameSave store (default pg6-saves in the
//                        working directory); saves persist there between runs, no cloud sync
//   --gpu-capture <file> write the GPU command stream of the first frames there as a .pg6trace
//                        plus <file>.json (gpu/frame_capture.h); data payloads are omitted
//   --audio-dump <file.wav>  write the stereo float32 48 kHz mix (os/audio_downmix) to a WAV file, max 10 minutes
//   --audio-out          play the stereo mix live (Windows WASAPI shared mode, default render endpoint);
//                        off by default; no device = warning only; underruns are counted in the report
//   --gpu-capture-frames <n>  frames to capture (default 10)
//   --gpu-capture-thin-uploads <n>  with payloads: keep only every nth write of a texture/buffer > 4 MiB
//   --gpu-capture-payloads    keep WriteBuffer / WriteTexture / shader payloads in the capture
//   --gpu-capture-start-ms <n>  start the capture's frames late, n ms after boot (gpu/frame_capture.h
//                        "Late start"); objects and the newest writes are kept, commands dropped
//   --gpu-capture-start-at-script  start the capture's frames at the input script's `capture` step
//   --live-digest <file>  (with --window, tools/live_vs_replay.py) per-frame replay digests of the live executor
//   --screenshot-at N[,N...]  with --window: save those present indices (0-based) as PNG; F12 saves the current frame
//                to %LOCALAPPDATA%\playgta6\screenshots
//   --present-mode auto|fifo|mailbox|immediate / --max-fps N (0 = off)  with --window: surface present mode, host-side cap
//   --overlay            (with --window) start with the on-screen stats overlay on (F3 toggles it, rebindable [hotkeys] overlay; gpu/native_exec/overlay_model.h)
//   --show-target <selector>  (with --window) present a render target / texture instead of the swapchain image (read-only; Ctrl+F7 mip, Ctrl+F8 layer, Ctrl+F9/F10 target, rebindable in keymap.ini [hotkeys]; gpu/native_exec/show_target_model.h)
//   --window-script <script>  (with --window) timed resize / minimize / restore / fullscreen / probe actions (gpu/native_exec/window_script.h)
//   --live-executor [--window-size WxH]  the same live path as --window (LivePump + native executor, the guest's GPU stream,
//                        --live-digest / --screenshot-at / --gpu-validation / gpu_time / --gpu-capture / --host-input), but
//                        presenting into the executor's offscreen swapchain texture (same format, size and blit): no window.
//   --gpu-time-split / --gpu-time-log <file>  (with --window) time each pipeline inside a pass (timestamps at every pipeline change); one cumulative gpu_time line per second (tools/gpu_time_top.py --log ... --phase NAME=A-B)
//   --window [--window-size WxH]  live play: hand the GPU packet stream to the native Dawn executor
//                        (docs/NATIVE_EXECUTOR_DESIGN.md) on its own thread and present to a window.
//                        Only in builds with -DPLAYGTA6_NATIVE_EXECUTOR=ON. Works together with
//                        --gpu-capture and --host-input (keyboard only while the window has focus).
//                        Closing the window stops the boot cleanly (normal report, stop_requested).
//   --arena-log-limit <n>     address-space events kept in the report log (default 4096)
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
#include <csignal>
#include <exception>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu/d3d12_guest_resources.h"
#include "gpu/rt_arena.h"
#include "gpu/binding_fallbacks.h"
#include "gpu/drop_table.h"
#include "gpu/pipeline_prewarm_host.h"
#include "gpu/gpu_time.h"
#include "gpu/present_pacing.h"
#include "gpu/d3d12_xs_exports.h"
#include "gpu/d3d12_xs_xbox.h"
#include "gpu/frame_capture.h"
#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/host_input.h"
#include "guest/input_record.h"
#include "gpu/native_exec/present_model.h"
#include "gpu/native_exec/screenshot_model.h"
#include "gpu/native_exec/show_target_model.h"
#ifdef PLAYGTA6_NATIVE_EXECUTOR
#include "gpu/native_exec/blob_cache.h"
#include "gpu/native_exec/live_pump.h"
#include "gpu/native_exec/native_executor.h"
#endif
#include "guest/input_script.h"
#include "guest/keymap.h"
#include "guest/stop_message.h"
#include "gpu/gpu_va_watch.h"
#include "gpu/present_pacing.h"
#include "os/guest_heap.h"
#include "os/host_sleep.h"
#include "guest/boot.h"
#include "guest/minidump.h"
#include "os/httpfs_mailbox.h"
#include "os/image_loader.h"
#include "os/audio_report.h"
#include "os/audio_out.h"
#include "os/audio_out_wasapi.h"
#include "os/audio_wav_dump.h"
#include "os/bcpack.h"
#include "os/os_shim.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_middleware.h"
#include "os/xbox_scarlett.h"
#include "os/xgameruntime_provider.h"
#include "os/xgamesave_local.h"
#include "os/xbox_scarlett_guest.h"
#include "wasm_boot.h"

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
#pragma comment(lib, "dbghelp.lib")
#include "guest/minidump_writer.h"
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

// Build metadata written by tools/build_native_boot.py beside the harness (build-info.json has the same
// content). Absent when the harness is built by hand.
#if defined(__has_include)
#if __has_include("pg6_build_info.h")
#include "pg6_build_info.h"
#endif
#endif
#ifndef PG6_BUILD_INFO_JSON
#define PG6_BUILD_INFO_JSON "{\"source_commit\": null}"
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
    uint64_t dstorage_workers = 1;  // --dstorage-workers: DirectStorage worker threads (1..16)
    std::string bcpack_tables;      // --bcpack-tables: user-supplied BCPack table file (os/bcpack.h)
    uint64_t arena_log_limit = 4096;  // --arena-log-limit
    uint64_t console_mode = playgta6::guest::kConsoleModeSeriesX;
    std::string sandbox_id = "RETAIL";  // --sandbox-id ("" = no sandbox)
    std::string user_age_group = "unknown";  // --user-age-group
    std::string save_dir;                    // --save-dir
    std::string gpu_capture;          // --gpu-capture: .pg6trace path
    uint64_t gpu_capture_frames = 10;
    bool gpu_capture_payloads = false;
    uint64_t gpu_capture_thin_uploads = 0;
    uint64_t gpu_capture_start_ms = 0;    // --gpu-capture-start-ms: late start (0 = from boot)
    bool gpu_capture_start_at_script = false;  // --gpu-capture-start-at-script
    bool native_pipelines = false;  // d3d12 DeviceOptions::refuse_native_pipelines = false
    bool fence_lag = false;         // d3d12 DeviceOptions::fence_publish_lag = true (#663)
    bool eager_gpu_objects = false; // d3d12 DeviceOptions::defer_gpu_objects = false
    std::string profile;            // --profile: per-thread host PC samples (Windows)
    uint64_t profile_interval_us = 2000;  // --profile-interval-us
    uint64_t profile_start_ms = 0;        // --profile-start-ms: sample only from this time on (a screen's window)
    std::string audio_dump;         // --audio-dump: WAV of the stereo mix
    bool audio_out = false;         // --audio-out: live WASAPI playback of the stereo mix
    std::string dump_pipelines;     // --dump-pipelines: d3d12 DeviceOptions::dump_dir (any dir outside a repo checkout)
    std::string pipeline_warm_list; // --pipeline-warm-list: warm list file ("off" disables; default per title under the cache dir)
    std::string window_script;      // --window-script <script>
    bool gpu_time_split = false;    // --gpu-time-split: per-pipeline GPU timing inside a pass (native executor)
    std::string gpu_time_log;       // --gpu-time-log <file>: with --window, a gpu_time snapshot line per second
    std::string live_digest;        // --live-digest <file>: per-frame replay digests from the live executor
    std::string dawn_cache;         // --dawn-cache: Dawn blob cache root ("off" disables; default %LOCALAPPDATA%/playgta6/dawn-cache)
    uint64_t dawn_cache_max_mb = 2048;  // --dawn-cache-max-mb: LRU cap of the title's cache directory (0 = unbounded)
    bool no_async_pipelines = false;    // --no-async-pipelines: live executor creates pipelines synchronously
    std::string shader_pack;        // --shader-pack: WGPK whose keys resolve pipeline stages
    bool gpu_validation = false;    // --gpu-validation: strict Dawn validation in the live executor
    bool live_executor = false;     // --live-executor: --window's live path into the offscreen swapchain (no window)
    bool window = false;            // --window: live GPU stream to the native executor (PLAYGTA6_NATIVE_EXECUTOR builds)
    std::string show_target;        // --show-target <selector>
    bool overlay = false;           // --overlay: start with the stats overlay on (F3 toggles)
    uint32_t window_width = 1280, window_height = 720;  // --window-size WxH
    playgta6::gpu::native_exec::PresentChoice present_mode = playgta6::gpu::native_exec::PresentChoice::Auto;  // --present-mode
    uint32_t max_fps = 0;           // --max-fps: host-side present cap (0 = off)
    std::vector<uint64_t> screenshot_at;  // --screenshot-at: present indices saved as PNG (F12 always works)
    bool host_input = false;        // --host-input: real XInput pad + keyboard (guest/host_input.h)
    std::string keymap_file;        // --keymap: key/mouse rebinding file for --host-input (guest/keymap.h)
    std::string record_input;       // --record-input: write the real --host-input readings (guest/input_record.h)
    std::string input_script;      // --input-script: timed controller buttons (guest/input_script.h)
    std::string watch_gpu_va;       // --watch-gpu-va: GPU address ranges to report references to (gpu/gpu_va_watch.h)
    // --wgpu-features: pso::WgpuFeature bits of the WebGPU device that replays the run. The bridge
    // picks texture formats (and whether guest texels can be uploaded as they are) by them; the
    // default is the browser player's negotiated set (boot::kDefaultWasmWgpuFeatures).
    uint64_t wgpu_features = playgta6::boot::kDefaultWasmWgpuFeatures;
    bool tls_callbacks = true;
    bool entry = true;
    bool quiet = false;
    std::string record_indirect_targets;  // --record-indirect-targets <file>
    bool dump_on_fault = false;  // --dump-on-fault: also write a minidump on a guest fault
    bool fault_record = true;    // --no-fault-record: no fault_record / entry ring (benchmarks)
    std::vector<playgta6::guest::WatchRange> watch;  // --watch-guest-va <addr>[:len] (repeatable)
    bool present_honor_interval = false;  // --present-interval honor|min1 (default min1)
    bool present_pacing = true;  // O36: --no-present-pacing turns it off (benchmarks)
};

Options g_options;
std::string g_minidump_path;  // set once a fault minidump was written (recorded in fault_record)
std::atomic<bool> g_finishing{false};
int g_result = 0;

void usage() {
    std::fprintf(stderr,
                 "usage: playgta6_boot --pgim <guest.pgim> [--name <exe>] [--game-dir <dir>] [--cwd <dir>]\n"
                 "                     [--cmdline <text>] [--survey] [--max-calls N] [--max-ms N]\n"
                 "                     [--watchdog-ms N] [--stack BYTES] [--host-stack-mb N]\n"
                 "                     [--report <file>] [--summary <file>] [--no-tls-callbacks]\n"
                 "                     [--no-entry] [--quiet] [--no-fault-record] [--dump-on-fault] [--watch-guest-va <addr>[:len]] [--record-indirect-targets <file>] [--no-present-pacing] [--no-xmem-shared-pages]\n"
                 "                     [--gpu-capture <file> [--gpu-capture-frames N] [--gpu-capture-payloads]\n"
                 "                      [--gpu-capture-start-ms N | --gpu-capture-start-at-script]]\n"
                 "                     [--native-pipelines] [--fence-lag] [--eager-gpu-objects]\n"
                 "                     [--dump-pipelines <dir outside the repo, or under work/>]\n"
                 "                     [--pipeline-warm-list <file|off>]\n"
                 "                     [--dawn-cache <dir|off>] [--dawn-cache-max-mb N] [--no-async-pipelines]\n"
                 "                     [--shader-pack <file.wgsl.pack>] [--input-script <file>] [--host-input]\n"
                 "                     [--record-input <file>] [--keymap <file>]\n"
                 "                     [--wgpu-features <mask>] [--audio-dump <file.wav>] [--audio-out]\n"
                 "                     [--bcpack-tables <file>]\n"
                 "                     [--watch-gpu-va <0xA-0xB|0xA+SIZE|textures:WxH[/WxH]>[,...]]\n");
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
        } else if (a == "--dump-on-fault") {
            o.dump_on_fault = true;
        } else if (a == "--no-fault-record") {
            o.fault_record = false;
        } else if (a == "--watch-guest-va" && i + 1 < argc) {
            playgta6::guest::WatchRange r;
            if (!playgta6::guest::parse_watch_spec(argv[++i], &r)) return false;
            o.watch.push_back(r);
        } else if (a == "--present-interval" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v != "honor" && v != "min1") return false;
            o.present_honor_interval = v == "honor";
        } else if (a == "--no-present-pacing") {
            o.present_pacing = false;
        } else if (a == "--no-xmem-shared-pages") {
            playgta6::os::win32::set_process_shared_zone(false);
        } else if (a == "--gpu-capture-payloads") {
            o.gpu_capture_payloads = true;
        } else if (a == "--gpu-capture-start-at-script") {
            o.gpu_capture_start_at_script = true;
        } else if (a == "--fence-lag") {
            o.fence_lag = true;
        } else if (a == "--no-fence-lag") {
            o.fence_lag = false;  // the default since #663; kept for old command lines
        } else if (a == "--eager-gpu-objects") {
            o.eager_gpu_objects = true;
        } else if (a == "--window") {
            o.window = true;
        } else if (a == "--live-executor") {
            o.live_executor = true;
        } else if (a == "--gpu-validation") {
            o.gpu_validation = true;
        } else if (a == "--no-async-pipelines") {
            o.no_async_pipelines = true;
        } else if (a == "--dawn-cache-max-mb" && i + 1 < argc) {
            o.dawn_cache_max_mb = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--present-mode" && i + 1 < argc) {
            if (!playgta6::gpu::native_exec::parse_present_choice(argv[++i], &o.present_mode)) return false;
        } else if (a == "--screenshot-at" && i + 1 < argc) {
            std::string err;
            if (!playgta6::gpu::native_exec::parse_screenshot_at(argv[++i], &o.screenshot_at, &err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return false;
            }
        } else if (a == "--overlay") {
            o.overlay = true;
        } else if (a == "--show-target" && i + 1 < argc) {
            o.show_target = argv[++i];
            playgta6::gpu::native_exec::ShowTargetSpec spec;
            std::string err;
            if (!playgta6::gpu::native_exec::parse_show_target(o.show_target, &spec, &err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return false;
            }
        } else if (a == "--max-fps" && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long v = std::strtoul(argv[++i], &end, 10);
            if (!end || *end || v > 100000) return false;
            o.max_fps = static_cast<uint32_t>(v);
        } else if (a == "--window-size" && i + 1 < argc) {
            unsigned w = 0, h = 0;
            if (std::sscanf(argv[++i], "%ux%u", &w, &h) != 2 || w == 0 || h == 0) return false;
            o.window_width = w;
            o.window_height = h;
        } else if (a == "--gpu-time-split") {
            o.gpu_time_split = true;
        } else if (a == "--host-input") {
            o.host_input = true;
        } else if (a == "--native-pipelines") {
            o.native_pipelines = true;
        } else if (a == "--audio-out") {
            o.audio_out = true;
        } else if (a == "-h" || a == "--help") {
            return false;
        } else if (a == "--pgim" || a == "--name" || a == "--game-dir" || a == "--cwd" ||
                   a == "--program-dir" ||
                   a == "--cmdline" || a == "--report" || a == "--summary" || a == "--import-trace" ||
                   a == "--gpu-capture" || a == "--sandbox-id" || a == "--dump-pipelines" ||
                   a == "--shader-pack" || a == "--pipeline-warm-list" || a == "--dawn-cache" || a == "--live-digest" || a == "--gpu-time-log" || a == "--window-script" || a == "--input-script" || a == "--record-input" || a == "--keymap" || a == "--watch-gpu-va" ||
                   a == "--user-age-group" || a == "--profile" || a == "--save-dir" || a == "--audio-dump" ||
                   a == "--bcpack-tables" || a == "--record-indirect-targets") {
            if (!value(v)) {
                return false;
            }
            std::string& target = a == "--pgim"       ? o.pgim
                                  : a == "--gpu-capture" ? o.gpu_capture
                                  : a == "--sandbox-id" ? o.sandbox_id
                                  : a == "--dump-pipelines" ? o.dump_pipelines
                                  : a == "--shader-pack" ? o.shader_pack
                                  : a == "--pipeline-warm-list" ? o.pipeline_warm_list
                                  : a == "--dawn-cache" ? o.dawn_cache
                                  : a == "--live-digest" ? o.live_digest
                                  : a == "--gpu-time-log" ? o.gpu_time_log
                                  : a == "--window-script" ? o.window_script
                                  : a == "--input-script" ? o.input_script
                                  : a == "--record-input" ? o.record_input
                                  : a == "--keymap" ? o.keymap_file
                                  : a == "--watch-gpu-va" ? o.watch_gpu_va
                                  : a == "--user-age-group" ? o.user_age_group
                                  : a == "--save-dir" ? o.save_dir
                                  : a == "--audio-dump" ? o.audio_dump
                                  : a == "--bcpack-tables" ? o.bcpack_tables
                                  : a == "--record-indirect-targets" ? o.record_indirect_targets
                                  : a == "--profile" ? o.profile
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
                   a == "--console-mode" || a == "--gpu-capture-frames" || a == "--gpu-capture-thin-uploads" || a == "--arena-log-limit" ||
                   a == "--gpu-capture-start-ms" ||
                   a == "--dstorage-workers" || a == "--profile-interval-us" || a == "--profile-start-ms" || a == "--wgpu-features") {
            if (!value(v)) {
                return false;
            }
            uint64_t& target = a == "--max-calls"     ? o.max_calls
                               : a == "--gpu-capture-frames" ? o.gpu_capture_frames
                               : a == "--gpu-capture-thin-uploads" ? o.gpu_capture_thin_uploads
                               : a == "--gpu-capture-start-ms" ? o.gpu_capture_start_ms
                               : a == "--arena-log-limit" ? o.arena_log_limit
                               : a == "--console-mode" ? o.console_mode
                               : a == "--dstorage-workers" ? o.dstorage_workers
                               : a == "--profile-interval-us" ? o.profile_interval_us
                               : a == "--profile-start-ms" ? o.profile_start_ms
                               : a == "--wgpu-features" ? o.wgpu_features
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
    if (playgta6::os::gdk::provider::UserAgeGroup group{};
        !playgta6::os::gdk::provider::parse_user_age_group(o.user_age_group, &group)) {
        std::fprintf(stderr, "--user-age-group must be unknown, child, teen or adult\n");
        return false;
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

// ---- --profile: where the guest threads spend their time ---------------------------------------
// A sampler thread suspends every guest thread each interval, reads its host instruction pointer
// and resumes it; the PC is classified after the resume (no lock is taken while a thread is
// suspended): a lifted function ("guest sub_X"), a system module ("os <module>", i.e. blocked or
// running in the OS: waits, file I/O, allocation) or the runtime ("runtime"); the last two carry
// the thread's last import, i.e. the shim it is in. Written per thread, by share of samples.
#ifdef _WIN32
struct SampledThread {
    uint32_t id;
    HANDLE handle;
};
std::mutex g_profile_mutex;
std::vector<SampledThread> g_profile_threads;
std::atomic<bool> g_profile_stop{false};
std::thread g_profile_thread;
std::map<uint32_t, std::map<std::string, uint64_t>> g_profile_labels;  // sampler thread, then writer

void profile_thread_started(uint32_t thread_id) {
    HANDLE h = nullptr;
    if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h,
                        THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0)) {
        std::lock_guard<std::mutex> lock(g_profile_mutex);
        g_profile_threads.push_back({thread_id, h});
    }
}

// Host function containing `pc` from the harness PDB (build_native_boot.py --debug-info) or a
// system module's exports, or "" (no symbols). Only the sampler thread calls it (DbgHelp is
// single-threaded); results are cached per PC.
std::string profile_symbol(uint64_t pc) {
    static bool initialized = false;
    static bool available = false;
    static std::unordered_map<uint64_t, std::string> cache;
    if (!initialized) {
        initialized = true;
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        available = SymInitialize(GetCurrentProcess(), nullptr, TRUE) != FALSE;
    }
    if (!available) {
        return {};
    }
    const auto it = cache.find(pc);
    if (it != cache.end()) {
        return it->second;
    }
    alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + 256] = {};
    auto* info = reinterpret_cast<SYMBOL_INFO*>(storage);
    info->SizeOfStruct = sizeof(SYMBOL_INFO);
    info->MaxNameLen = 255;
    DWORD64 displacement = 0;
    std::string name;
    if (SymFromAddr(GetCurrentProcess(), pc, &displacement, info)) {
        name.assign(info->Name, info->NameLen);
    }
    cache.emplace(pc, name);
    return name;
}

// The return address of the function containing `ctx.Rip`: a leaf function (no unwind data) has
// it on top of the stack (`top`, copied while the thread was suspended); otherwise one virtual
// unwind step (on the live stack, after the resume: a statistical answer, not an exact one).
uint64_t profile_caller(const CONTEXT& ctx, uint64_t top) {
    DWORD64 image_base = 0;
    PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
    if (entry == nullptr) {
        return top;
    }
    CONTEXT unwound = ctx;
    void* handler_data = nullptr;
    DWORD64 establisher = 0;
    __try {
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, entry, &unwound, &handler_data, &establisher,
                         nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return 0;
    }
    return unwound.Rip;
}

std::string profile_module(uint64_t pc) {
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(static_cast<uintptr_t>(pc)), &module) &&
        module != GetModuleHandleW(nullptr)) {
        char path[MAX_PATH] = {};
        GetModuleFileNameA(module, path, MAX_PATH);
        const char* base = std::strrchr(path, '\\');
        return std::string("os ") + (base != nullptr ? base + 1 : path);
    }
    return "runtime";
}

std::string profile_label(const CONTEXT& ctx, uint64_t top, uint32_t thread_id) {
    const uint64_t pc = ctx.Rip;
    uint64_t offset = 0;
    if (const uint64_t fn = guest::guest_function_for_host_pc(pc, &offset)) {
        char text[40];
        std::snprintf(text, sizeof(text), "guest sub_%" PRIX64, fn);
        return text;
    }
    std::string where = profile_module(pc);
    const std::string symbol = profile_symbol(pc);
    if (!symbol.empty()) {
        where += " " + symbol;
    }
    // In a system DLL: name who called it (a memchr or a wait says little without its caller).
    if (where.compare(0, 3, "os ") == 0) {
        if (const uint64_t caller = profile_caller(ctx, top)) {
            uint64_t caller_offset = 0;
            if (const uint64_t fn = guest::guest_function_for_host_pc(caller, &caller_offset)) {
                char text[48];
                std::snprintf(text, sizeof(text), " <- guest sub_%" PRIX64, fn);
                where += text;
            } else {
                const std::string caller_symbol = profile_symbol(caller);
                where += " <- " + (caller_symbol.empty() ? profile_module(caller) : caller_symbol);
            }
        }
    }
    const std::string last = guest::guest_thread_last_import(thread_id);
    return where + " (last import " + (last.empty() ? "none" : last) + ")";
}

void profile_loop(uint64_t interval_us, uint64_t start_ms) {
    const auto begin = std::chrono::steady_clock::now();
    while (!g_profile_stop.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() - begin < std::chrono::milliseconds(start_ms)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        std::vector<SampledThread> threads;
        {
            std::lock_guard<std::mutex> lock(g_profile_mutex);
            threads = g_profile_threads;
        }
        for (const SampledThread& t : threads) {
            if (SuspendThread(t.handle) == static_cast<DWORD>(-1)) {
                continue;
            }
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_FULL;
            const bool ok = GetThreadContext(t.handle, &ctx) != 0;
            uint64_t top = 0;  // a leaf function's return address (profile_caller)
            if (ok && ctx.Rsp != 0) {
                std::memcpy(&top, reinterpret_cast<const void*>(static_cast<uintptr_t>(ctx.Rsp)), sizeof(top));
            }
            ResumeThread(t.handle);
            if (ok) {
                ++g_profile_labels[t.id][profile_label(ctx, top, t.id)];
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(interval_us));
    }
}

void start_profile() {
    if (!g_options.profile.empty()) {
        g_profile_thread = std::thread(profile_loop, std::max<uint64_t>(100, g_options.profile_interval_us),
                                    g_options.profile_start_ms);
    }
}

void finish_profile() {
    if (g_options.profile.empty() || !g_profile_thread.joinable() || g_profile_stop.exchange(true)) {
        return;
    }
    g_profile_thread.join();
    std::string out;
    for (const auto& [id, labels] : g_profile_labels) {
        uint64_t total = 0;
        std::map<std::string, uint64_t> kinds;
        for (const auto& [label, n] : labels) {
            total += n;
            kinds[label.substr(0, label.find(' '))] += n;
        }
        std::vector<std::pair<uint64_t, std::string>> top;
        for (const auto& [label, n] : labels) {
            top.emplace_back(n, label);
        }
        std::sort(top.rbegin(), top.rend());
        char line[512];
        std::snprintf(line, sizeof(line), "thread %u: %" PRIu64 " samples;", id, total);
        out += line;
        for (const auto& [kind, n] : kinds) {
            std::snprintf(line, sizeof(line), " %s %.1f%%", kind.c_str(), 100.0 * n / std::max<uint64_t>(1, total));
            out += line;
        }
        out += "\n";
        for (size_t i = 0; i < top.size() && i < 12; ++i) {
            std::snprintf(line, sizeof(line), "  %5.1f%%  %s\n", 100.0 * top[i].first / std::max<uint64_t>(1, total),
                          top[i].second.c_str());
            out += line;
        }
    }
    write_text(g_options.profile, out);
    std::fprintf(stderr, "[harness] thread profile written to %s\n", g_options.profile.c_str());
}
#else
void profile_thread_started(uint32_t) {}
void start_profile() {}
void finish_profile() {}
#endif

playgta6::os::audio_out::Feed g_audio_feed;
playgta6::os::audio_out::Wasapi g_audio_wasapi;
bool g_audio_out_active = false;  // device opened and feeding
std::string g_audio_out_error;
#ifdef PLAYGTA6_NATIVE_EXECUTOR
namespace ne = playgta6::gpu::native_exec;
ne::LivePump<ne::NativeExecutor> g_live;
bool g_live_started = false;
std::atomic<bool> g_window_focused{true};
std::atomic<playgta6::guest::host_input::HostInputSource*> g_host_source{nullptr};
std::mutex g_live_final_mutex;
std::string g_live_final_stats = "null";  // the executor's stats_json, captured when the pump stops
std::atomic<bool> g_live_deadline_active{false};
std::atomic<bool> g_live_stats_active{false};  // the 5 s [live] fps line thread  // the --max-ms watcher thread of a live run
std::atomic<bool> g_guest_faulted{false};  // set before the window is held open on a fault
std::string g_fault_text;                  // one-line fault summary for the overlay, written before g_guest_faulted
std::string g_stop_message;                // the "Stopped: ..." / "Crashed: ..." banner (guest/stop_message.h), same timing

// Window title on the pump thread, about 4 times a second: fps, frames presented, draws per frame
// of the newest interval, and the fault state.
void live_title_tick(ne::NativeExecutor& exec) {
    struct Sample {
        std::chrono::steady_clock::time_point t{};
        uint64_t presents = 0, draws = 0;
        double fps = 0, draws_per_frame = 0;
    };
    static Sample last;
    // Recent stats for a report written after a guest fault (the finish hook only runs on a clean stop).
    static auto last_snapshot = std::chrono::steady_clock::time_point{};
    if (std::chrono::steady_clock::now() - last_snapshot >= std::chrono::seconds(1)) {
        last_snapshot = std::chrono::steady_clock::now();
        std::string json = exec.stats_json();
        std::lock_guard<std::mutex> lock(g_live_final_mutex);
        g_live_final_stats = std::move(json);
    }
    const ne::ExecutorStats& st = exec.stats();
    const auto now = std::chrono::steady_clock::now();
    if (last.t.time_since_epoch().count() != 0) {
        const double dt = std::chrono::duration<double>(now - last.t).count();
        const uint64_t dp = st.present_packets - last.presents;
        if (dt > 0 && dp > 0) {
            last.fps = static_cast<double>(dp) / dt;
            last.draws_per_frame = static_cast<double>(st.draws - last.draws) / static_cast<double>(dp);
        } else if (dt > 1.0) {
            last.fps = 0;
        }
    }
    last.t = now;
    last.presents = st.present_packets;
    last.draws = st.draws;
    const std::string mode = exec.present_mode_name();
    char buf[240];
    if (g_guest_faulted.load()) {
        std::snprintf(buf, sizeof buf, "playgta6 | GUEST FAULT - last frame kept, details in the console | frame %llu | close the window to exit",
                      static_cast<unsigned long long>(st.present_packets));
    } else {
        std::snprintf(buf, sizeof buf, "playgta6 | %.0f fps | frame %llu | %.0f draws/frame | %s | cap %u | interval %s | ok | F11 fullscreen",
                      last.fps, static_cast<unsigned long long>(st.present_packets), last.draws_per_frame, mode.empty() ? "?" : mode.c_str(), g_options.max_fps,
                      g_options.present_honor_interval ? "honor" : "min1");
    }
    exec.set_window_title(buf);
    // Overlay extras (F3): live-audio underruns and why the guest stopped.
    exec.set_overlay_extra(g_audio_out_active ? static_cast<int64_t>(g_audio_feed.buffer().stats().underruns) : -1,
                           g_guest_faulted.load() ? g_fault_text : std::string());
    exec.set_overlay_banner(g_guest_faulted.load() ? g_stop_message : std::string());
}

void live_observer(const uint8_t* data, size_t len) {
    if (!g_options.gpu_capture.empty()) {
        gpu::frame_capture_observe(data, len);
    }
    g_live.submit(data, len);  // blocks while the executor is behind (bounded queue, counted)
}

void apply_window_focus(bool focused) {
    g_window_focused.store(focused);
    if (auto* source = g_host_source.load()) {
        source->set_keyboard_enabled(focused);
    }
}

// Starts the pump; false (after printing why) when the executor cannot be created.
bool start_live_window() {
    ne::ExecutorOptions options;
    options.window = g_options.window;  // --live-executor: the offscreen swapchain, same size / format / blit
    options.width = g_options.window_width;
    options.height = g_options.window_height;
    options.present_width = g_options.window_width;
    options.present_height = g_options.window_height;
    options.present_mode = g_options.present_mode;
    options.show_target = g_options.show_target;
    options.overlay = g_options.overlay;
    options.max_fps = g_options.max_fps;
    options.screenshot_at = g_options.screenshot_at;
    // Persistent Dawn blob cache per title / Dawn build / adapter (repeat runs skip the DXC and driver
    // compiles), and pipelines compiled on Dawn's worker threads while the pump keeps consuming packets.
    if (g_options.dawn_cache != "off") {
        options.blob_cache_root = g_options.dawn_cache.empty() ? ne::BlobCache::default_root() : g_options.dawn_cache;
        options.blob_cache_title = g_options.name.empty() ? "title" : g_options.name;
        options.blob_cache_max_bytes = g_options.dawn_cache_max_mb << 20;
    }
    options.async_pipelines = !g_options.no_async_pipelines;
    options.gpu_validation = g_options.gpu_validation;
    options.frame_digest_path = g_options.live_digest;
    options.gpu_time_split = g_options.gpu_time_split;
    options.window_script = g_options.window_script;
    // The guest renders at a fixed resolution: a window resize letterboxes the stream instead of stretching it.
    options.fixed_stream_size = true;
    g_live.set_tick_hook(&live_title_tick, 250);
    g_live.set_finish_hook([](ne::NativeExecutor& exec) {
        std::lock_guard<std::mutex> lock(g_live_final_mutex);
        g_live_final_stats = exec.stats_json();
    });
    std::string error;
    const bool ok = g_live.start(
        [options](std::string* err) -> std::unique_ptr<ne::NativeExecutor> {
            ne::NativeExecutor::CreateStatus status;
            auto exec = ne::NativeExecutor::create(options, &status, err);
            if (exec) {
                // The same WGPK the guest-side index uses (v1 or v2): hash-only shader payloads resolve to WGSL here.
                if (!g_options.shader_pack.empty()) {
                    std::string pack_error;
                    if (exec->load_wgsl_pack(g_options.shader_pack, &pack_error))
                        std::fprintf(stderr, "[harness] native executor shader pack loaded: %s\n", g_options.shader_pack.c_str());
                    else
                        std::fprintf(stderr, "[harness] native executor: shader pack %s: %s\n", g_options.shader_pack.c_str(), pack_error.c_str());
                }
                // The producer's read-back packets carry absolute host addresses (ReadbackGate /
                // CommandList::readback_target), so the results land straight in the running guest.
                ne::GuestMemory mem;
                mem.host_addresses = true;
                exec->attach_guest_memory(mem);
            }
            return exec;
        },
        [] {
            std::fprintf(stderr, "[harness] window closed: stopping the boot\n");
            guest::request_boot_stop();
        },
        [](bool focused) { apply_window_focus(focused); }, &error);
    if (!ok) {
        std::fprintf(stderr, "[harness] --window: %s\n", error.c_str());
        return false;
    }
    g_live_started = true;
    playgta6::os::set_wgpu_submit_observer(&live_observer);
    // Every 5 s: where the frame time goes (so a run below 60 fps names its cap): presents/s, the share of
    // wall time the guest's submit() was blocked behind a full executor queue, and the share it slept in
    // the present pacer (title-driven waits). Neither high = the guest's own CPU is the limit.
    g_live_stats_active.store(true);
    std::thread([] {
        namespace gp = playgta6::gpu;
        auto prev_t = std::chrono::steady_clock::now();
        ne::LivePumpStats prev = g_live.stats();
        gp::PresentPacer::Stats prev_p = gp::present_pacer().stats();
        while (g_live_stats_active.load() && !g_live.window_closed()) {
            for (int i = 0; i < 50 && g_live_stats_active.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (!g_live_stats_active.load()) break;
            const auto now = std::chrono::steady_clock::now();
            const ne::LivePumpStats cur = g_live.stats();
            const gp::PresentPacer::Stats cp = gp::present_pacer().stats();
            const double secs = std::chrono::duration<double>(now - prev_t).count();
            std::fprintf(stderr,
                         "[live] %.1f fps, submit blocked %.1f%%, pacer slept %.1f%%, queue max %llu, interval %u\n",
                         static_cast<double>(cur.presents - prev.presents) / secs,
                         100.0 * static_cast<double>(cur.blocked_ns - prev.blocked_ns) / (secs * 1e9),
                         100.0 * static_cast<double>(cp.wait_us - prev_p.wait_us) / (secs * 1e6),
                         static_cast<unsigned long long>(cur.max_queue_depth), cp.last_interval);
            prev_t = now;
            prev = cur;
            prev_p = cp;
        }
    }).detach();
    // --gpu-time-log: one JSON line per second (elapsed ms, presents, the cumulative gpu_time table), so a
    // tool can cut a run into phases by differencing two lines. Counts and nanoseconds only.
    if (!g_options.gpu_time_log.empty()) {
        std::thread([] {
            std::FILE* f = std::fopen(g_options.gpu_time_log.c_str(), "wb");
            if (!f) return;
            const auto t0 = std::chrono::steady_clock::now();
            while (g_live_stats_active.load() && !g_live.window_closed()) {
                for (int i = 0; i < 10 && g_live_stats_active.load(); ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(f, "{\"t_ms\": %lld, \"presents\": %llu, \"gpu_time\": %s}\n", static_cast<long long>(ms),
                             static_cast<unsigned long long>(g_live.stats().presents),
                             playgta6::gpu::d3d12_guest::gpu_time_table().json(1024).c_str());
                std::fflush(f);
            }
            std::fclose(f);
        }).detach();
    }
    // --max-ms must end a live run even while the guest is blocked behind the executor (the boot's own
    // limit is only checked on indirect dispatches, which a thread stuck in submit never reaches).
    if (g_options.max_ms != 0) {
        g_live_deadline_active.store(true);
        std::thread([] {
            const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(g_options.max_ms);
            while (g_live_deadline_active.load() && std::chrono::steady_clock::now() < end)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (!g_live_deadline_active.load()) return;
            std::fprintf(stderr, "[harness] --max-ms %llu reached: stopping the live run\n",
                         static_cast<unsigned long long>(g_options.max_ms));
            guest::request_boot_stop();
            g_live.abort();
        }).detach();
    }
    return true;
}

// A guest fault must not close the window at once: write the report now (the launcher prints the
// fault report from it), flip the title to the fault state and keep the last frame up until the
// user closes the window (or the watchdog fires).
void hold_window_on_fault(bool faulted, const std::string& summary) {
    if (!faulted || !g_live_started || !g_options.window || g_live.window_closed()) return;
    g_fault_text = summary.substr(0, summary.find('\n'));
    {
        namespace sm = playgta6::guest::stop_message;
        const char* slots = std::getenv("PLAYGTA6_HARD_STOP_SLOTS");
        std::string dir = g_options.report;
        const size_t cut = dir.find_last_of("/\\");
        dir = cut == std::string::npos ? std::string() : dir.substr(0, cut);
        g_stop_message = sm::select(sm::facts_from_report(guest::last_report_json(), dir),
                                    sm::parse_pending(slots != nullptr ? slots : ""));
        std::fprintf(stderr, "[harness] %s\n", g_stop_message.c_str());
    }
    write_text(g_options.report, guest::last_report_json());
    std::fprintf(stderr, "[harness] guest fault: window kept open on the last frame; close it to exit\n");
    std::fflush(stderr);
    g_guest_faulted.store(true);
    while (!g_live.window_closed()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// After the boot: detach, drain the queue, destroy the executor on its thread.
void stop_live_window() {
    if (!g_live_started) return;
    g_live_deadline_active.store(false);
    g_live_stats_active.store(false);
    playgta6::os::set_wgpu_submit_observer(nullptr);
    g_live.stop();
    const ne::LivePumpStats s = g_live.stats();
    std::fprintf(stderr,
                 "[harness] native executor: %llu batch(es), %llu present(s), submit blocked %llu time(s) / %llu ms\n",
                 static_cast<unsigned long long>(s.executed), static_cast<unsigned long long>(s.presents),
                 static_cast<unsigned long long>(s.blocked_submits), static_cast<unsigned long long>(s.blocked_ns / 1000000));
    {
        std::lock_guard<std::mutex> lock(g_live_final_mutex);
        std::fprintf(stderr, "[harness] native executor stats: %s\n", g_live_final_stats.c_str());
    }
    g_live_started = false;
}
#endif

// The boot logs its summary itself unless --quiet; host faults happen outside its logging.
playgta6::os::audio_wav_dump::Writer g_audio_dump;

// --audio-dump and --audio-out share the single stereo sink slot.
void audio_fanout_sink(void*, const float* stereo, uint32_t frames, uint32_t sample_rate) {
    if (!g_options.audio_dump.empty()) g_audio_dump.append(stereo, frames, sample_rate);
    if (g_audio_out_active) g_audio_feed.push(stereo, frames, sample_rate);
}

#ifdef _WIN32
bool write_minidump(const std::string& path, EXCEPTION_POINTERS* info);
#endif

void write_outputs(bool print_summary) {
    finish_profile();
    gpu::prewarm_host_stop();
    gpu::prewarm_host_save();
    {
        const os::GuestHeapStats heap = os::default_guest_heap().stats();
        std::fprintf(stderr,
                     "[harness] guest heap lock: %llu acquisitions, %llu contended, %.3f ms blocked "
                     "(%llu allocs, %llu frees)\n",
                     static_cast<unsigned long long>(heap.lock_acquisitions),
                     static_cast<unsigned long long>(heap.lock_contended),
                     static_cast<double>(heap.lock_wait_ns) / 1e6,
                     static_cast<unsigned long long>(heap.total_allocs),
                     static_cast<unsigned long long>(heap.total_frees));
    }
    if (!g_options.audio_dump.empty() || g_options.audio_out) {
        playgta6::os::audio_downmix::set_stereo_sink(nullptr, nullptr);
        g_audio_dump.close();
    }
    if (g_options.audio_out) {
        g_audio_wasapi.stop();
        g_audio_out_active = false;
    }
    if (!g_options.watch_gpu_va.empty()) {
        std::fprintf(stderr, "%s", gpu::va_watch::summary().c_str());
    }
    if (!g_options.gpu_capture.empty()) {
        gpu::frame_capture_finish();
        std::fprintf(stderr, "[harness] GPU capture: %u frame(s) written to %s\n", gpu::frame_capture_frames(),
                     g_options.gpu_capture.c_str());
    }
    if (!g_options.dump_pipelines.empty()) {
        if (auto device = gpu::d3d12_xs_exports::current_device()) {
            device->flush_pipeline_dump();
        }
        std::fprintf(stderr, "[harness] pipeline dump written to %s\n", g_options.dump_pipelines.c_str());
    }
    std::string report = guest::last_report_json();
#ifdef _WIN32
    if (g_options.dump_on_fault && g_minidump_path.empty() && report.find("\"fault_record\": {") != std::string::npos) {
        const std::string path = guest::minidump::select_path(g_options.report, "guest", "fault",
                                                              static_cast<uint32_t>(GetCurrentProcessId()));
        if (write_minidump(path, nullptr)) g_minidump_path = path;
    }
#endif
    report = guest::minidump::annotate_report(report, g_minidump_path);
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
    playgta6::guest::host_input::silence_host_rumble();
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

// A guest fault unwinds out of the guest with an exception; when that exception crosses a noexcept
// frame (or the CRT aborts), the process would die with 0xC0000409 and no report. The terminate
// and SIGABRT handlers write the already recorded fault record + report first.
[[noreturn]] void terminate_handler_report() {
    host_fault("host_terminate", 0, 0, kExitHostFault);
}

void abort_signal_report(int) {
    host_fault("host_abort", 0, 0, kExitHostFault);
}

// BootConfig::off_thread_fault: the boot already recorded the fault of a non-boot guest thread.
void off_thread_fault() {
    if (!g_finishing.exchange(true)) {
        write_outputs(true);
    }
    terminate_now(static_cast<int>(guest::BootStatus::Fault));
}

#ifdef _WIN32
// Writes the dump via the dedicated dumper thread (guest/minidump_writer.h); `info` may be null
// (guest faults). Returns true on success. Never throws; safe to call from the vectored handler.
bool write_minidump(const std::string& path, EXCEPTION_POINTERS* info) {
    return guest::minidump::Writer::init() && guest::minidump::Writer::dump(path.c_str(), info);
}

// Fault hook (guest::set_fault_hook): runs on the faulting guest thread inside the fault path, so
// RtlCaptureContext here is that thread's live host context (lifted sub_ frames below it). The
// dump carries it as a synthetic exception stream plus the fault_record JSON user stream.
// NOINLINE keeps the captured context a real frame.
__declspec(noinline) void guest_fault_hook(const std::string& fault_record_json) {
    if (!g_options.dump_on_fault || !g_minidump_path.empty()) return;
    CONTEXT ctx;
    RtlCaptureContext(&ctx);
    const std::string path = guest::minidump::select_path(g_options.report, "guest", "fault",
                                                          static_cast<uint32_t>(GetCurrentProcessId()));
    if (guest::minidump::Writer::init() && guest::minidump::Writer::dump_guest_fault(path.c_str(), &ctx, fault_record_json))
        g_minidump_path = path;
}

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

// Prints the first return-address-looking words on the faulting thread's stack as module+offset
// (a call through a null pointer has pc 0 and no unwind info, so the caller is at [rsp]).
void print_host_stack_hint(uint64_t rsp) {
    std::fprintf(stderr, "[host-stack] fault thread %lu, rsp 0x%" PRIx64 ":", GetCurrentThreadId(), rsp);
    int printed = 0;
    for (int i = 0; i < 256 && printed < 12; ++i) {
        uint64_t word = 0;
        __try {
            word = *reinterpret_cast<const uint64_t*>(rsp + static_cast<uint64_t>(i) * 8);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            break;
        }
        HMODULE module = nullptr;
        if (word < 0x10000 || !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                                      GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                                  reinterpret_cast<LPCSTR>(word), &module)) {
            continue;
        }
        char name[MAX_PATH] = {};
        GetModuleFileNameA(module, name, MAX_PATH);
        const char* base = std::strrchr(name, '\\');
        std::fprintf(stderr, " %s+0x%" PRIx64, base != nullptr ? base + 1 : name,
                     word - reinterpret_cast<uint64_t>(module));
        ++printed;
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
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
    print_host_stack_hint(info->ContextRecord->Rsp);
    if (code != EXCEPTION_BREAKPOINT && !g_finishing.load()) {
        const std::string path = guest::minidump::select_path(g_options.report, "host", kind,
                                                              static_cast<uint32_t>(GetCurrentProcessId()));
        if (write_minidump(path, info)) {
            g_minidump_path = path;
            std::fprintf(stderr, "[host-stack] minidump written to %s\n", path.c_str());
        }
    }
    host_fault(kind, address, pc, kExitHostFault);
}

void install_host_fault_handlers() {
    guest::minidump::Writer::init();  // dbghelp + dumper thread before any fault
    guest::set_fault_hook(&guest_fault_hook);
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
    cfg.present_pacing = g_options.present_pacing;
    cfg.present_honor_interval = g_options.present_honor_interval;
    cfg.fence_publish_lag = g_options.fence_lag;  // run_boot applies it to the device options
    cfg.run_tls_callbacks = g_options.tls_callbacks;
    cfg.call_entry = g_options.entry;
    cfg.log_progress = !g_options.quiet;
    cfg.fault_record = g_options.fault_record;
    cfg.watch_guest_va = g_options.watch;
    cfg.record_indirect_targets = g_options.record_indirect_targets;
    cfg.capture_host_stack = &capture_host_stack;
    cfg.import_trace_limit = g_options.import_trace.empty() ? 0 : g_options.import_trace_limit;
    cfg.arena_log_limit = static_cast<size_t>(g_options.arena_log_limit);
    cfg.host_function_start = host_function_start;
    cfg.console_mode = static_cast<uint32_t>(g_options.console_mode);
    cfg.sandbox_id = g_options.sandbox_id;
    playgta6::os::gdk::gamesave::set_save_dir(g_options.save_dir);
    playgta6::os::gdk::provider::parse_user_age_group(g_options.user_age_group, &cfg.user_age_group);
    cfg.off_thread_fault = &off_thread_fault;
    cfg.guest_thread_started = &profile_thread_started;
    const guest::BootResult result = guest::run_boot(cfg);
#ifdef PLAYGTA6_NATIVE_EXECUTOR
    hold_window_on_fault(result.status == guest::BootStatus::Fault, result.summary);
    stop_live_window();
#endif
    playgta6::os::gdk::gamesave::flush_all();  // exit: nothing the title saved stays only in cache
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
    guest::set_build_info_json(PG6_BUILD_INFO_JSON);
    os::host_sleep::raise_timer_resolution();  // guest timed waits otherwise wake on the 15.6 ms tick
    os::scarlett::guest_api::set_dstorage_worker_threads(static_cast<uint32_t>(g_options.dstorage_workers));
    {
        // BCPack tables: user-supplied only (docs/BCPACK.md). A bad --bcpack-tables file is a usage
        // error; a bad PLAYGTA6_BCPACK_TABLES only warns (BCPack requests then fail E_NOTIMPL).
        const char* env = std::getenv("PLAYGTA6_BCPACK_TABLES");
        const bool from_flag = !g_options.bcpack_tables.empty();
        const std::string path = from_flag ? g_options.bcpack_tables : (env != nullptr ? env : "");
        if (!path.empty()) {
            std::string error;
            if (os::bcpack::load_tables_file(path, &error)) {
                std::fprintf(stderr, "[harness] BCPack tables loaded from %s (%s)\n", path.c_str(),
                             os::bcpack::supported_formats(*os::bcpack::active_tables()).c_str());
            } else {
                std::fprintf(stderr, "[harness] %s: BCPack tables not loaded: %s\n", from_flag ? "error" : "warning",
                             error.c_str());
                if (from_flag) {
                    return kExitUsage;
                }
            }
        }
    }
    // Write tracking on the guest address space (Windows): the GPU layer then compares only the
    // written pages of CPU-visible buffers at each ExecuteCommandLists instead of whole buffers.
    os::set_write_tracking(true);
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
    std::set_terminate(&terminate_handler_report);
    std::signal(SIGABRT, &abort_signal_report);
    {
        gpu::d3d12_guest::DeviceOptions device_options;
        device_options.refuse_native_pipelines = !g_options.native_pipelines;
        device_options.fence_publish_lag = g_options.fence_lag;
        device_options.defer_gpu_objects = !g_options.eager_gpu_objects;
        // The features of the WebGPU device that replays the run (by default the browser player's):
        // without them BC and 16-bit normalized textures get substitute formats and their guest
        // texels are never uploaded.
        device_options.bridge.features = static_cast<uint32_t>(g_options.wgpu_features & gpu::pso::WgpuFeature::All);
        std::fprintf(stderr, "[harness] WebGPU features 0x%x\n", device_options.bridge.features);
        if (!g_options.dump_pipelines.empty()) {
            // The dump is derived from the title: any directory, but never inside a repository
            // checkout (an ancestor holding `.git`) unless it sits under a gitignored `work` dir.
            std::error_code ec;
            const std::filesystem::path dir = std::filesystem::absolute(g_options.dump_pipelines, ec);
            bool under_work = false;
            for (const auto& part : dir) {
                under_work = under_work || part == "work";
            }
            bool in_repo = false;
            for (std::filesystem::path p = dir; !ec && !in_repo && p.has_relative_path(); p = p.parent_path()) {
                in_repo = std::filesystem::exists(p / ".git", ec);
            }
            if (ec || (in_repo && !under_work)) {
                std::fprintf(stderr, "[harness] --dump-pipelines must not name a directory inside a repository checkout (use a path outside it, or under a `work` directory)\n");
                return kExitUsage;
            }
            device_options.dump_dir = dir.string();
        }
        if (!g_options.shader_pack.empty()) {
            // Only the header + entry table are read (a merged pack is several GB).
            std::ifstream in(g_options.shader_pack, std::ios::binary);
            std::vector<uint8_t> bytes(gpu::kShaderPackHeaderBytes);
            in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            bytes.resize(static_cast<size_t>(in.gcount()));
            uint32_t pack_count = 0, pack_version = 0;
            auto index = std::make_shared<gpu::ShaderPackIndex>();
            std::string error;
            bool pack_ok = gpu::parse_shader_pack_header(bytes.data(), bytes.size(), &pack_count, &error, &pack_version);
            if (pack_ok) {
                const size_t toc_bytes = gpu::kShaderPackHeaderBytes + size_t{pack_count} * gpu::shader_pack_entry_bytes(pack_version);
                bytes.resize(toc_bytes);
                in.read(reinterpret_cast<char*>(bytes.data() + gpu::kShaderPackHeaderBytes),
                        static_cast<std::streamsize>(toc_bytes - gpu::kShaderPackHeaderBytes));
                std::error_code size_ec;
                const uint64_t file_size = std::filesystem::file_size(g_options.shader_pack, size_ec);
                if (size_ec || static_cast<size_t>(in.gcount()) != toc_bytes - gpu::kShaderPackHeaderBytes) {
                    error = "WGPK entry table exceeds the file";
                    pack_ok = false;
                } else {
                    pack_ok = gpu::parse_shader_pack_index_toc(bytes.data(), bytes.size(), file_size, index.get(), &error);
                }
            }
            if (!pack_ok) {
                std::fprintf(stderr, "[harness] --shader-pack %s: %s\n", g_options.shader_pack.c_str(), error.c_str());
                return kExitUsage;
            }
            std::fprintf(stderr, "[harness] shader pack %s: %zu key(s)\n", g_options.shader_pack.c_str(), index->size());
            uint64_t index_bytes = uint64_t{index->bucket_count()} * 8 + uint64_t{index->size()} * 32;
            device_options.shader_pack = std::move(index);
            // Stage reflection sidecar (G62): <pack>.reflect.json next to the pack, when present.
            const std::string sidecar = g_options.shader_pack + ".reflect.json";
            if (std::ifstream rin(sidecar, std::ios::binary); rin) {
                const std::string json((std::istreambuf_iterator<char>(rin)), std::istreambuf_iterator<char>());
                auto reflection = std::make_shared<gpu::StageReflectionIndex>();
                if (!gpu::parse_stage_reflection_json(json, reflection.get(), &error)) {
                    std::fprintf(stderr, "[harness] %s: %s\n", sidecar.c_str(), error.c_str());
                    return kExitUsage;
                }
                std::fprintf(stderr, "[harness] stage reflection %s: %zu stage(s)\n", sidecar.c_str(), reflection->size());
                uint64_t reflection_bytes = uint64_t{reflection->bucket_count()} * 8;
                for (const auto& kv : *reflection) {
                    reflection_bytes += 24 + sizeof(gpu::StageReflection) +
                                        kv.second.vertex_inputs.capacity() * sizeof(gpu::ReflectedVertexInput) +
                                        kv.second.bindings.capacity() * sizeof(gpu::ReflectedBinding) +
                                        kv.second.address_sources.capacity() * sizeof(kv.second.address_sources[0]);
                }
                guest::note_host_table_bytes(index_bytes, reflection_bytes);
                device_options.stage_reflection = std::move(reflection);
            }
        }
        // Pipeline prewarm: warm list from the per-title cache, built ahead of first use in the background.
        if (device_options.shader_pack) {
            gpu::pipeline_prewarm().set_pack_index(device_options.shader_pack);
            const std::string warm_path = g_options.pipeline_warm_list.empty()
                                              ? gpu::default_warm_list_path(g_options.name.empty() ? "title" : g_options.name)
                                              : g_options.pipeline_warm_list;
            std::string message;
            const uint32_t queued = gpu::prewarm_host_configure(warm_path, &message);
            std::fprintf(stderr, "[harness] pipeline warm list %s: %s (%u queued)\n", warm_path.c_str(), message.c_str(), queued);
            if (queued != 0) {
                gpu::prewarm_host_start(
                    {[]() { return static_cast<bool>(gpu::d3d12_xs_exports::current_device()); },
                     [](uint32_t kind, uint64_t k0, uint64_t k1) {
                         auto device = gpu::d3d12_xs_exports::current_device();
                         return device && device->prewarm_pack_modules(kind, k0, k1);
                     },
                     [](uint32_t task, uint32_t kind, uint64_t k0, uint64_t k1, const std::vector<uint8_t>& detail) {
                         auto device = gpu::d3d12_xs_exports::current_device();
                         if (!device) return gpu::PrewarmBuild::Failed;
                         switch (device->prewarm_pipeline(task, kind, k0, k1, detail)) {
                             case gpu::bridge::PrewarmPipelineResult::Built: return gpu::PrewarmBuild::Built;
                             case gpu::bridge::PrewarmPipelineResult::Lost: return gpu::PrewarmBuild::Lost;
                             default: return gpu::PrewarmBuild::Failed;
                         }
                     }});
            }
        }
        gpu::d3d12_xs_exports::set_device_options(std::move(device_options));
    }
    // R40: the report always carries the D3D12 layer's work totals (draws / dispatches emitted vs
    // skipped by reason), and the capture's per-Present samples when a capture runs.
    {
        const bool with_presents = !g_options.gpu_capture.empty();
        static bool s_with_presents = false;
        s_with_presents = with_presents;
        guest::set_report_extension([]() {
            namespace dg = gpu::d3d12_guest;
            std::string o = "\"gpu_work\": {";
            for (size_t i = 0; i < static_cast<size_t>(dg::Work::Count); ++i) {
                const dg::Work w = static_cast<dg::Work>(i);
                o += std::string(i == 0 ? "" : ", ") + "\"" + dg::work_name(w) + "\": " + std::to_string(dg::work_total(w));
            }
            o += ", \"drops_by_program\": " + dg::drop_table().by_program_json();
            o += "}";
            o += ",\n  \"pipeline_prewarm\": " + gpu::prewarm_host_report_json();
#ifdef PLAYGTA6_NATIVE_EXECUTOR
            {
                // stop_live_window() ran before the report: use the stats it captured on the executor thread.
                std::lock_guard<std::mutex> lock(g_live_final_mutex);
                if (g_live_final_stats != "null") o += ",\n  \"native_executor\": " + g_live_final_stats;
                else if (g_live_started) o += ",\n  \"native_executor\": " + g_live.stats_json();
            }
#endif
            o += ",\n  \"rt\": " + gpu::rt_arena::report_json();
            // Per-pipeline GPU time (#1608): fed by the native executor's timestamp queries
            // (--window); "available": false with the reason when nothing was measured.
            o += ",\n  \"gpu_time\": " + dg::gpu_time_table().json();
            o += ",\n  \"gpu_drop_table\": " + dg::drop_table().detail_json();
            o += ",\n  \"binding_fallbacks\": " + dg::binding_fallback_table().detail_json();
            {
                namespace ar = os::audio_report;
                const os::scarlett::guest_api::AudioEndpointStats a = os::scarlett::guest_api::audio_endpoint_stats();
                const os::scarlett::guest_api::XApuStats x = os::scarlett::guest_api::xapu_stats();
                ar::Totals t;
                t.render_frames = a.render_frames;
                t.render_nonzero_frames = a.render_nonzero_frames;
                t.render_silent_releases = a.render_silent_releases;
                t.spatial_passes = a.spatial_passes;
                t.spatial_object_buffers = a.spatial_object_buffers;
                t.spatial_nonzero_buffers = a.spatial_nonzero_buffers;
                t.spatial_nonzero_passes = a.spatial_nonzero_passes;
                t.xapu_decoded_packets = x.decoded_packets;
                t.xapu_failed_packets = x.failed_packets;
                t.xapu_results_posted = x.type3_results;
                t.xapu_commands = x.commands;
                t.xapu_unhandled_commands = x.unhandled_commands;
                t.xapu_type1 = x.type1_commands;
                t.xapu_type2 = x.type2_commands;
                t.xapu_type3 = x.type3_commands;
                t.xapu_type5 = x.type5_commands;
                const Pg6XmaContextCensus xma = pg6_mw_xma_context_census();
                t.xma_contexts = xma.contexts;
                t.xma_contexts_touched = xma.touched;
                t.xma_contexts_input_valid = xma.input_valid;
                t.xma_contexts_output_pending = xma.output_pending;
                const os::scarlett::guest_api::XmaCommandStats xc = os::scarlett::guest_api::xma_command_stats();
                t.xma_contexts_enabled = xc.enabled_now;
                t.xma_enable_commands = xc.enables + xc.mask_enables;
                t.xma_read_advances = xc.read_advances;
                t.xma_frames = xc.frames;
                t.xma_blocks_written = xc.blocks_written;
                t.xma_input_exhausted = xc.input_exhausted;
                o += ",\n  \"audio\": " + ar::to_json(t, ar::windows().nonsilent_windows(), ar::windows().clamped.load());
                if (g_options.audio_out) {
                    o += ",\n  \"audio_out\": " + playgta6::os::audio_out::report_json(
                                                      g_audio_wasapi.open(), g_audio_wasapi.device_rate(),
                                                      g_audio_feed.buffer().stats(), g_audio_out_error,
                                                      g_audio_wasapi.device_channels(),
                                                      g_audio_wasapi.device_channel_mask(),
                                                      playgta6::os::audio_out::steady_now_us() -
                                                          playgta6::guest::boot_elapsed_ms() * 1000ull);
                }
            }
            if (s_with_presents) {
                o += ",\n  \"presents\": " + gpu::frame_capture_presents_json();
            }
            return o;
        });
    }
    if (!g_options.gpu_capture.empty()) {
        gpu::FrameCaptureConfig capture;
        capture.path = g_options.gpu_capture;
        capture.frames = static_cast<uint32_t>(std::max<uint64_t>(1, g_options.gpu_capture_frames));
        capture.keep_payloads = g_options.gpu_capture_payloads;
        capture.thin_uploads = static_cast<uint32_t>(g_options.gpu_capture_thin_uploads);
        capture.start_ms = g_options.gpu_capture_start_ms;
        capture.attach = !(g_options.window || g_options.live_executor);  // with --window / --live-executor the live observer feeds the capture
        capture.deferred = g_options.gpu_capture_start_ms != 0 || g_options.gpu_capture_start_at_script;
        if (capture.deferred) {
            std::fprintf(stderr, "[harness] GPU capture deferred until %s\n",
                         g_options.gpu_capture_start_at_script ? "the input script's capture step"
                                                               : (std::to_string(capture.start_ms) + " ms").c_str());
        }
        if (!gpu::frame_capture_start(capture)) {
            std::fprintf(stderr, "[harness] cannot write the GPU capture %s\n", g_options.gpu_capture.c_str());
            return kExitUsage;
        }
        gpu::frame_capture_set_counters([](gpu::FrameCaptureSample* sample) {
            guest::mark_first_present();  // sampled at every Present; only the first one counts
            const guest::LiveCounters c = guest::live_counters();
            sample->dispatches = c.dispatches;
            sample->import_calls = c.import_calls;
            const os::VirtualMemoryArena* arena = os::win32::get_vmem_arena();
            sample->reserved_bytes = arena->reserved_bytes();
            sample->committed_bytes = arena->committed_bytes();
            // Runtime-side state, to tell growth of the runtime from the title's own.
            const os::win32::KernelHandleCounts h = os::win32::live_kernel_handles();
            const guest::com::BridgeStats com = guest::com::bridge_stats();
            const os::GuestHeapStats heap = os::default_guest_heap().stats();
            const gpu::d3d12_state::DescriptorLiveCounts dcount = gpu::d3d12_state::descriptor_live_counts();
            const gpu::bridge::BridgeLiveCounts bcount = gpu::bridge::bridge_live_counts();
            const gpu::d3d12_guest::xbox::FrameWaitTotals waits = gpu::d3d12_guest::xbox::frame_wait_totals();
            sample->counters = {
                {"handles_dispatcher", h.dispatcher},
                {"handles_file", h.file},
                {"handles_opaque", h.opaque},
                {"com_live_objects", com.live_objects},
                {"com_guest_bytes", com.guest_bytes_in_use},
                {"lazy_guest_threads", com.lazy_guest_threads},
                {"async_shadows", static_cast<uint64_t>(guest::callbacks::live_async_shadows())},
                {"heap_live_blocks", heap.live_blocks},
                {"heap_live_bytes", heap.live_bytes},
                {"heap_total_allocs", heap.total_allocs},
                {"heap_lock_acquisitions", heap.lock_acquisitions},
                {"heap_lock_contended", heap.lock_contended},
                {"heap_lock_wait_us", heap.lock_wait_ns / 1000},
                // Live descriptor-side objects (backlog 28): all must plateau over a long run.
                {"descriptor_heaps", dcount.heaps},
                {"descriptor_heap_slots", dcount.heap_slots},
                {"bind_groups_cached", bcount.cached_bind_groups},
                {"interned_views", bcount.interned_views},
                {"samplers_live", bcount.samplers},
                {"command_lists_live", bcount.command_lists},
                {"frame_waits", waits.waits},
                {"frame_wait_us", waits.wait_us},
                {"frame_polls", waits.polls},
                {"arena_events", c.arena_events},
            };
            // Recorded GPU work by outcome (gpu/d3d12_guest_objects.h, Work).
            namespace dg = gpu::d3d12_guest;
            for (size_t w = 0; w < static_cast<size_t>(dg::Work::Count); ++w) {
                sample->counters.emplace_back(dg::work_name(static_cast<dg::Work>(w)),
                                              dg::work_total(static_cast<dg::Work>(w)));
            }
            // DirectStorage (XCRDAPI): what the title's asset loading gets done.
            const os::scarlett::guest_api::DStorageStats ds = os::scarlett::guest_api::dstorage_stats();
            sample->counters.emplace_back("dstorage_requests_completed", ds.requests_completed);
            sample->counters.emplace_back("dstorage_requests_failed", ds.requests_failed);
            sample->counters.emplace_back("dstorage_bcpack_requests", ds.bcpack_requests);
            sample->counters.emplace_back("dstorage_bcpack_decoded", ds.bcpack_decoded);
            sample->counters.emplace_back("dstorage_bytes_delivered", ds.bytes_delivered);
            sample->counters.emplace_back("dstorage_decompressed_requests", ds.decompressed_requests);
            sample->counters.emplace_back("dstorage_file_reads", ds.file_reads);
            // Textures the title fills by writing their memory (no upload call): resource textures
            // and view-only textures (the streamed ones), with the view-only part on its own too.
            const gpu::d3d12_guest::TextureProbeTotals probe = gpu::d3d12_guest::probe_texture_contents(256);
            sample->counters.emplace_back("textures_with_memory", probe.registered);
            sample->counters.emplace_back("textures_with_data", probe.with_data);
            sample->counters.emplace_back("view_textures_with_memory", probe.views_registered);
            sample->counters.emplace_back("view_textures_with_data", probe.views_with_data);
        });
    }
    if (g_options.window && g_options.live_executor) {
        std::fprintf(stderr, "[harness] --window and --live-executor are exclusive\n");
        return kExitUsage;
    }
    if (g_options.window || g_options.live_executor) {
#ifdef PLAYGTA6_NATIVE_EXECUTOR
        if (!start_live_window()) {
            return kExitUsage;
        }
#else
        std::fprintf(stderr, "[harness] --window / --live-executor needs a build with -DPLAYGTA6_NATIVE_EXECUTOR=ON\n");
        return kExitUsage;
#endif
    }
    // --input-script: a test tool that presses buttons on a schedule through the same InputState
    // the browser's input bridge writes (the title reads it through the GameInput adapter).
    static playgta6::InputState script_input{};
    static std::unique_ptr<playgta6::guest::InputScriptPlayer> script_player;
    if (!g_options.watch_gpu_va.empty()) {
        std::string error;
        if (!gpu::va_watch::configure(g_options.watch_gpu_va, &error)) {
            std::fprintf(stderr, "[harness] --watch-gpu-va: %s\n", error.c_str());
            return kExitUsage;
        }
        std::fprintf(stderr, "[harness] watching GPU addresses: %s\n", g_options.watch_gpu_va.c_str());
    }
    if (!g_options.audio_dump.empty()) {
        if (!g_audio_dump.open(g_options.audio_dump)) {
            std::fprintf(stderr, "[harness] cannot write %s\n", g_options.audio_dump.c_str());
            return kExitUsage;
        }
    }
    if (g_options.audio_out) {
        g_audio_feed.set_present_source([]() -> uint64_t { return playgta6::guest::boot_present_count(); });
        g_audio_out_active = g_audio_wasapi.start(g_audio_feed, &g_audio_out_error);
        if (!g_audio_out_active) {
            std::fprintf(stderr, "[harness] --audio-out: no audio device, continuing silent (%s)\n",
                         g_audio_out_error.c_str());
        }
    }
    if (!g_options.audio_dump.empty() || g_audio_out_active) {
        playgta6::os::audio_downmix::set_stereo_sink(&audio_fanout_sink, nullptr);
    }
    if (!g_options.input_script.empty()) {
        std::ifstream in(g_options.input_script, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<playgta6::guest::InputScriptEvent> events;
        std::string error;
        if (!in.good() && !in.eof()) {
            std::fprintf(stderr, "[harness] cannot read the input script %s\n", g_options.input_script.c_str());
            return kExitUsage;
        }
        if (!playgta6::guest::parse_input_script(text, events, error)) {
            std::fprintf(stderr, "[harness] input script %s: %s\n", g_options.input_script.c_str(), error.c_str());
            return kExitUsage;
        }
        std::fprintf(stderr, "[harness] input script %s: %zu event(s)\n", g_options.input_script.c_str(), events.size());
        playgta6::os::scarlett::set_game_input_state(&script_input);
        const bool has_capture_step =
            std::any_of(events.begin(), events.end(), [](const playgta6::guest::InputScriptEvent& e) {
                return e.kind == playgta6::guest::InputStepKind::Capture;
            });
        if (g_options.gpu_capture_start_at_script && !has_capture_step) {
            std::fprintf(stderr, "[harness] --gpu-capture-start-at-script: the input script has no capture step\n");
            return kExitUsage;
        }
        script_player = std::make_unique<playgta6::guest::InputScriptPlayer>(std::move(events), &script_input);
        // wait / repeat steps follow the live present counters (the report's present_samples).
        script_player->set_status_probe([] {
            const guest::PresentStatus p = guest::present_status();
            return playgta6::guest::InputStatus{p.presents, p.draws_per_frame};
        });
        if (g_options.gpu_capture_start_at_script) {
            script_player->set_on_capture([] { gpu::frame_capture_begin(); });
        }
        script_player->set_log(true);
        script_player->start();
    } else if (g_options.host_input) {
        static playgta6::InputState host_state{};
        static playgta6::guest::host_input::HostInputSource host_source(&host_state);
        {
            playgta6::guest::keymap::Keymap map;
            std::vector<std::string> errors;
            std::string used;
            if (!playgta6::guest::keymap::load_keymap(g_options.keymap_file, map, errors, used)) {
                for (const std::string& e : errors) {
                    std::fprintf(stderr, "[harness] keymap: %s\n", e.c_str());
                }
                return kExitUsage;
            }
            host_source.set_keymap(map);
            std::fprintf(stderr, "[harness] keymap: %s%s\n", used.empty() ? "built-in table" : used.c_str(),
                         map.mouse_look ? " (mouse look on)" : "");
        }
        playgta6::os::scarlett::set_game_input_state(&host_state);
        playgta6::os::scarlett::set_game_input_press_sampler(true);
        static playgta6::guest::InputRecorder recorder;
        if (!g_options.record_input.empty()) {
            std::string error;
            if (!recorder.open(g_options.record_input, [] { return guest::present_status().presents; }, error)) {
                std::fprintf(stderr, "[harness] --record-input: %s\n", error.c_str());
                return kExitUsage;
            }
            host_source.set_recorder(&recorder);
            std::fprintf(stderr, "[harness] recording host input to %s\n", g_options.record_input.c_str());
        }
#ifdef PLAYGTA6_NATIVE_EXECUTOR
        g_host_source.store(&host_source);
#endif
        host_source.start();
        std::fprintf(stderr, "[harness] host input: XInput pad + keyboard\n");
    }
    if (!g_options.record_input.empty() && !g_options.host_input) {
        std::fprintf(stderr, "[harness] --record-input needs --host-input\n");
        return kExitUsage;
    }
    start_watchdog();
    start_profile();

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
