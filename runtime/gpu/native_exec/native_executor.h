// Native executor: consumes the GPU packet stream (docs/gpu_commands.md) with a native WebGPU
// implementation (Dawn, the library Chrome's WebGPU is built on) and presents to an offscreen
// swapchain texture (headless) or a native window. It mirrors the semantics of the browser worker
// (web/workers/wgpu_worker.js) packet by packet; packets it does not implement yet are skipped by
// payload_bytes and counted per opcode, never approximated. Built only with
// -DPLAYGTA6_NATIVE_EXECUTOR=ON. Design and milestones: docs/NATIVE_EXECUTOR_DESIGN.md.
//
// Threading (milestone 0): an executor and its window belong to the thread that created them;
// execute() and every other call must come from that thread.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "gpu/native_exec/present_model.h"
#include <vector>

#include "gpu/native_exec/m2_model.h"

namespace playgta6::gpu::d3d12_guest {
class GpuTimeTable;
}

namespace playgta6::gpu::native_exec {

struct ExecutorOptions {
    bool window = false;  // present to a native window instead of the offscreen swapchain
    uint32_t width = 1280;  // swapchain (and initial client area) size
    uint32_t height = 720;
    std::string title = "playgta6 native executor";
    // Swapchain blit (WgpuSwapchain's intermediate texture + fullscreen blit, plus letterboxing):
    // headless, a canvas of present_width x present_height (both non-zero) receives the stream
    // (width x height) scaled into the largest centred aspect-preserving rectangle, black bars
    // around it. With a window, fixed_stream_size keeps the stream at width x height and does the same
    // into the window's client area (default: the stream follows the window size). A stream format
    // that is not a canvas format (e.g. rgba8unorm-srgb) always goes through the blit.
    uint32_t present_width = 0;
    uint32_t present_height = 0;
    bool fixed_stream_size = false;
    // Test hook (--window-script, window_script.h): timed resize / minimize / restore / fullscreen
    // actions and read-back probes driven from poll(), counted from the first presented frame.
    std::string window_script;
    // GPU pass timestamps (#1608): measured with timestamp queries when the adapter has the
    // feature and fed into this table (nullptr = the process-wide gpu_time_table(), the one the
    // boot report prints). Nothing is estimated: without the feature the table says unavailable.
    // Test hook: do not request float32-filterable even when the adapter has it, so the code paths of a
    // device without it (r16float depth twins, r32float dummies) run on real hardware.
    bool disable_float32_filterable = false;
    bool gpu_timing = true;
    d3d12_guest::GpuTimeTable* gpu_time = nullptr;
    // Persistent Dawn blob cache (blob_cache.h): compiled shaders / pipelines are stored under
    // <blob_cache_root>/<blob_cache_title>/<Dawn build>-<adapter>/ and loaded by later runs, so DXC and
    // the driver compile each pipeline once per machine instead of once per run. Empty root = off
    // (the default: tests and tools never touch the user's cache); boot_main passes the default root.
    std::string blob_cache_root;
    std::string blob_cache_title = "title";
    uint64_t blob_cache_max_bytes = uint64_t{2} << 30;  // LRU-trimmed to this (0 = unbounded)
    // Live play: v2 render / compute pipelines are created with CreateRenderPipelineAsync /
    // CreateComputePipelineAsync, so Dawn compiles them on its own worker threads while the executor
    // keeps consuming packets (the prewarm batches create hundreds ahead of use). The pipeline is
    // awaited, never skipped, when a pass first uses it, so output is identical to sync creation.
    bool async_pipelines = false;
    // Strict validation (--gpu-validation): Dawn backend validation (D3D12 debug layer when installed),
    // immediate error handling, and an error scope around every packet. Errors are deduped by text in
    // stats_json()'s "validation" block with the first packet index / opcode. Slower; diagnosis only.
    bool gpu_validation = false;
    // Live-vs-replay check (tools/live_vs_replay.py): when non-empty, every presented frame's stream image
    // (before the blit; 4-byte formats) is read back and a line `digest frame N: <16 hex> WxH <format>` is
    // appended here, N counting presents from 0 - the same digest pg6_native_exec --digest-at prints.
    // Slow (one synchronous read-back per frame); a window surface then gets CopySrc usage.
    std::string frame_digest_path;
    // Window present mode (--present-mode) and host-side frame cap (--max-fps, 0 = off). An unsupported
    // mode falls back (logged); the guest's own pacing is not changed.
    PresentChoice present_mode = PresentChoice::Auto;
    uint32_t max_fps = 0;
    // Screenshots: F12 in the window, and these 0-based present indices (--screenshot-at) save the presented
    // stream image (before the letterbox) as a PNG into screenshot_dir (empty: %LOCALAPPDATA%/playgta6/screenshots).
    std::vector<uint64_t> screenshot_at;
    std::string screenshot_dir;
    // --show-target <selector> (show_target_model.h): present this render target / texture instead of the
    // swapchain image. Read-only, the guest's frame is unchanged. Empty = off. With a window F7 / F8 / F9 /
    // F10 (mip, layer, previous, next target) take over; set_show_target changes it at run time.
    std::string show_target;
};

inline constexpr size_t kOpcodeSlots = 64;

struct ExecutorStats {
    uint64_t batches = 0;
    uint64_t packets = 0;
    uint64_t bytes = 0;
    uint64_t present_packets = 0;
    uint64_t frames_submitted = 0;   // the worker's frameCount
    uint64_t present_failures = 0;   // no swapchain texture / submit failed: the frame is dropped
    uint64_t truncated_batches = 0;  // CommandStream "truncated packet" (rest of the batch dropped)
    uint64_t op_errors = 0;          // the worker's _recordOpValidationError for payload checks
    uint64_t device_errors = 0;      // uncaptured WebGPU errors (validation / OOM / internal)
    uint64_t device_lost = 0;
    uint64_t dropped_while_lost = 0;  // batches ignored after device loss
    uint64_t render_pass_missing_texture = 0;  // BeginRenderPass dropped: unknown attachment texture
    uint64_t render_pass_dropped_colors = 0;   // attachments dropped for a size mismatch
    uint64_t swapchain_format_unsupported = 0;  // stream asked for a swapchain format that is no colour format
    uint64_t blit_frames = 0;  // frames presented through the intermediate texture + blit
    uint64_t surface_configures = 0;         // surface (re)configurations (resize, format change, lost surface)
    uint64_t surface_minimized_skips = 0;    // presents skipped because the client area was empty (minimized)
    uint64_t surface_acquire_retries = 0;    // acquires retried after Outdated / Lost / Error
    uint64_t window_actions = 0;             // --window-script actions applied
    uint64_t probes = 0;                     // letterbox probes run
    uint64_t screenshots_saved = 0, screenshots_failed = 0;  // F12 / --screenshot-at PNGs written / failed
    uint64_t probes_failed = 0;              // ... whose bars were not black or whose picture was blank
    uint64_t show_target_frames = 0;         // presents that drew the --show-target texture instead of the swapchain image
    uint64_t show_target_unavailable = 0;    // presents where the chosen target could not be shown (the swapchain image stays)
    // Milestone 1 (resources, pipelines, draws). "packet errors" are the worker's `_packetError`
    // (a v2 / texture packet it rejects); op errors above are its `_recordOpValidationError`.
    uint64_t packet_errors = 0;
    std::array<uint64_t, kOpcodeSlots> packet_errors_by_opcode{};
    std::string last_packet_error;
    uint64_t shader_modules = 0;
    uint64_t shader_pack_hits = 0;
    uint64_t shader_pack_misses = 0;  // a hash-only shader the pack does not hold (compiled as text: invalid)
    uint64_t textures_created = 0;
    uint64_t texture_uploads = 0;
    uint64_t texture_upload_bytes = 0;
    uint64_t samplers_created = 0;
    uint64_t bind_group_layouts = 0;
    uint64_t pipeline_layouts = 0;
    uint64_t render_pipelines = 0;
    uint64_t compute_pipelines = 0;
    uint64_t bind_groups = 0;
    uint64_t draws = 0;
    uint64_t draws_skipped = 0;        // _executePresentRenderDraw skips (no pipeline / group 0 unbound)
    uint64_t dispatches = 0;
    uint64_t dispatches_skipped = 0;   // zero-size or missing pipeline (_preparePassCommandsForFlush)
    uint64_t dispatches_clamped = 0;
    uint64_t auto_layout_pipelines = 0;  // layout 0 ("auto"): the worker's WGSL remap is not mirrored
    uint64_t manifest_layouts_unavailable = 0;
    // layout:'auto' bind group remap (the worker's _getOrCreateRemappedBindGroup).
    uint64_t auto_layout_remap_pipelines = 0;  // an explicit layout that did not fit the WGSL @group(0) bindings
    uint64_t remapped_bind_groups = 0;         // group 0 bind groups built for such pipelines
    uint64_t remap_cache_hits = 0;
    uint64_t remap_dynamic_offsets = 0;        // dynamic offsets folded into static buffer offsets
    uint64_t remap_dynamic_clamps = 0;         // folded offsets clamped to the buffer / out of bounds (dummy)
    uint64_t remap_fallback_samplers = 0;      // sampler bindings no resource matched: default sampler
    uint64_t remap_dummies = 0;                // bindings no resource matched: dummy resource
    // Milestone 2: read-backs (39/40), v1 pipelines / bind groups (8/15/16/19), pass timestamps.
    uint64_t readbacks_requested = 0;
    uint64_t readbacks_delivered = 0;  // bytes written to guest memory
    uint64_t readbacks_failed = 0;     // counter advanced without data (copy / map failed, no memory, out of range)
    uint64_t readback_bytes = 0;
    uint64_t early_readback_flushes = 0;  // submits that ran the recorded work up to a read-back (no present)
    uint64_t v1_render_pipelines = 0;
    uint64_t v1_compute_pipelines = 0;
    uint64_t v1_bind_groups = 0;
    uint64_t reflection_unavailable = 0;  // v1 pipelines built without a matching reflection record (layout:auto)
    // Reflection loader (reflection_model.h: web/workers/reflection_loader.js): .reflect.json sidecars
    // (v1..v3) and layout-manifest.json documents feed v1 pipelines (layout, entry points, vertex
    // attributes), v1 bind groups (layouts, fallback samplers) and ManifestReference layouts.
    uint64_t reflection_documents = 0;      // documents loaded (sidecars + manifests)
    uint64_t reflection_load_errors = 0;    // documents the loader rejected (nothing of them is registered)
    uint64_t reflection_shaders = 0;        // stage / shader records in the registry
    uint64_t reflected_render_pipelines = 0;   // v1 render pipelines built with a reflected layout
    uint64_t reflected_compute_pipelines = 0;  // v1 compute pipelines built with a reflected layout
    uint64_t reflected_vertex_layouts = 0;     // v1 render pipelines with the reflected vertex buffer layout
    uint64_t reflection_vertex_fallbacks = 0;  // reflected vertex layout rejected: the single-attribute default
    uint64_t reflection_layout_errors = 0;     // a reflected layout the loader's merge rejected (pipeline not created)
    uint64_t reflection_binding_conflicts = 0; // ... of which two stages disagreed on a binding
    uint64_t bind_group_depth_layout_dummies = 0;  // depth view vs float/sint/uint layout entry (or the reverse): dummy bound
    uint64_t depth_float_copies = 0;       // depth aspect re-copied into an r32float twin (float layout entry meets a depth texture)
    uint64_t depth_twins_created = 0;      // ... of which r32uint/r32sint (uint/sint entry on a depth texture): depth_int_twins_created
    uint64_t depth_int_twins_created = 0;
    uint64_t depth_half_twins_created = 0; // r16float twins (float entry, device without float32-filterable): half-precision depth
    uint64_t depth_stencil_direct_binds = 0;  // stencil-aspect views bound to a uint entry (no twin: the stencil aspect is uint)
    uint64_t reflection_layout_unsupported = 0;  // layout entries Dawn cannot take (externalTexture, unknown storage format)
    uint64_t manifest_layouts_resolved = 0;    // ManifestReference layouts served from a loaded layout-manifest.json
    uint64_t v1_synth_samplers = 0;            // v1 bind group samplers synthesized from the layout (default samplers)
    std::string last_reflection_error;
    uint64_t timer_frames = 0;            // submits with timestamp queries attached
    uint64_t timer_passes = 0;            // passes that carried timestamp writes
    uint64_t timer_frames_skipped = 0;    // no free query slot (the previous one was still in flight)
    bool timer_supported = false;
    bool limits_raised = false;  // device created with requestNegotiatedDevice's raised limits
    std::array<uint64_t, kOpcodeSlots> handled{};    // by opcode
    std::array<uint64_t, kOpcodeSlots> unhandled{};  // by opcode: not implemented yet, skipped
    uint64_t unknown_opcodes = 0;                    // opcode >= kOpcodeSlots
    // Profile (ns, wall time on the executor thread): per opcode, and the phases of a batch / present.
    std::array<uint64_t, kOpcodeSlots> opcode_ns{};
    uint64_t ns_execute = 0, ns_process_events = 0, ns_flush_readbacks = 0, ns_poll = 0;
    uint64_t ns_present_acquire = 0, ns_present_encode = 0, ns_present_submit = 0, ns_present_swap = 0;
    // Persistent blob cache (blob_cache.h; a snapshot taken when stats() is read).
    bool blob_cache_enabled = false;
    std::string blob_cache_dir;
    uint64_t blob_hits = 0;       // Dawn loads served from disk (shader / pipeline compiles skipped)
    uint64_t blob_misses = 0;     // lookups Dawn had to compile for
    uint64_t blob_stores = 0;
    uint64_t blob_bytes_loaded = 0;
    uint64_t blob_bytes_stored = 0;
    uint64_t blob_evictions = 0;  // LRU removals to stay under the cap
    uint64_t blob_corrupt = 0;
    uint64_t blob_entries = 0;
    uint64_t blob_total_bytes = 0;
    uint64_t blob_max_bytes = 0;
    // Async pipeline creation (ExecutorOptions::async_pipelines).
    uint64_t async_pipelines = 0;          // pipelines handed to Dawn's worker threads
    uint64_t async_pipeline_waits = 0;     // first uses that still had to wait for the compile
    uint64_t async_pipeline_wait_ns = 0;
    uint64_t async_pipeline_failures = 0;  // creation failed: the pipeline became an error object
    std::string last_device_error;
    std::vector<std::string> device_error_log;  // the first 20 uncaptured device errors
};

struct ShowTargetCandidate {
    uint32_t id = 0;
    bool depth = false;
};

struct AdapterDescription {
    std::string backend;
    std::string vendor;
    std::string architecture;
    std::string device;
    std::string description;
};

// A presented frame read back from the GPU: rows tightly packed (width * 4 bytes, or * 8 for
// rgba16float), in the swapchain's own format.
struct FrameImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::string format;  // WebGPU name: "bgra8unorm", "rgba8unorm", "rgba16float"
    uint32_t bytes_per_pixel = 4;
    std::vector<uint8_t> pixels;
};

class NativeExecutor {
public:
    enum class CreateStatus { Ok, NoAdapter, Failed };

    // nullptr with *status NoAdapter when the host has no WebGPU adapter (tests skip), Failed for
    // any other error (*error says what).
    static std::unique_ptr<NativeExecutor> create(const ExecutorOptions& options, CreateStatus* status,
                                                  std::string* error);
    ~NativeExecutor();
    NativeExecutor(const NativeExecutor&) = delete;
    NativeExecutor& operator=(const NativeExecutor&) = delete;

    // Executes one batch of complete packets (as handed to os::wgpu_submit). Returns the number of
    // Present packets in it (the worker's `presents`).
    uint32_t execute(const uint8_t* data, size_t len);

    // Loads a .wgsl.pack (WGPK v1) for hash-only CreateShaderModule payloads, like the replay page's
    // `processor.loadWgslPack`. Only the entry table is read now; WGSL is read on demand.
    // A `<path>.reflect.json` sidecar next to the pack is loaded too (the replay page / boot
    // orchestrator derive the same URL); a bad sidecar is counted (reflection_load_errors), it does
    // not fail the pack.
    bool load_wgsl_pack(const std::string& path, std::string* error);

    // Loads a reflection document: a `playgta6-stage-reflection` sidecar (`*.reflect.json`, versions
    // 1..3) or an S6 `layout-manifest.json`, like `processor.loadReflectionManifest`. Later documents
    // overwrite records with the same hash. False + *error when the loader rejects the document.
    bool load_reflection(const std::string& path, std::string* error);
    bool load_reflection_text(const std::string& json, std::string* error);

    // Processes WebGPU callbacks and, with a window, window messages. Returns false once the window
    // was closed (always true headless).
    bool poll();
    // True while the presenting window has keyboard focus (false headless).
    bool window_focused() const;
    // Title bar text of the live window (no-op without one). Call from the pump thread.
    void set_window_title(const std::string& title);
    // --show-target at run time: selector grammar in show_target_model.h; "" switches it off. False + *error
    // for a selector that does not parse.
    bool set_show_target(const std::string& selector, std::string* error);
    // The hotkeys as bits (show_target_model.h kHotkeyMip / Layer / Prev / Next): what poll() does with F7..F10.
    void show_target_hotkeys(uint32_t bits);
    // What the last present showed ("tex 12 depth 1280x720 depth32float mip 0/1 layer 0/1"), the reason it could
    // not ("unavailable: ..."), "" when the swapchain image was shown.
    std::string shown_target_label() const;
    // Render targets of the last presented frame (the hotkey cycle list), sorted by id.
    std::vector<ShowTargetCandidate> shown_target_candidates() const;
    // Present mode in effect once the surface is configured ("" before / headless).
    std::string present_mode_name() const;
    // Path of the newest screenshot written (wait_idle() first), "" if none.
    std::string last_screenshot_path() const;
    // Blocks until the GPU finished all submitted work and every pending read-back / timestamp
    // read was delivered.
    void wait_idle();

    // Where read-back results (ReadbackBuffer / ReadbackTexture) are written: the worker's
    // attachGuestMemory. Without it a read-back writes nothing and does not move its counter.
    // The memory must outlive the executor's pending read-backs (wait_idle() drains them).
    void attach_guest_memory(const GuestMemory& memory);

    uint64_t frame_count() const;
    const ExecutorStats& stats() const;
    const AdapterDescription& adapter() const;
    std::string stats_json() const;

    // Headless only: the swapchain contents after the last submitted frame.
    bool read_last_frame(FrameImage* out, std::string* error);
    // Reads back mip `mip` of a stream texture (needs COPY_SRC), tightly packed rows. Uncompressed
    // 2D textures only.
    bool read_texture(uint32_t id, uint32_t mip, std::vector<uint8_t>* out, uint32_t* width, uint32_t* height,
                      std::string* error);
    // Reads a stream buffer (it needs COPY_SRC usage; WebGPU rejects the copy otherwise).
    bool read_buffer(uint32_t id, uint64_t offset, uint64_t size, std::vector<uint8_t>* out, std::string* error);

    struct Impl;

private:
    explicit NativeExecutor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace playgta6::gpu::native_exec
