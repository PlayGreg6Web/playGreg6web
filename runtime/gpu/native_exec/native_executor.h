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
#include <vector>

namespace playgta6::gpu::native_exec {

struct ExecutorOptions {
    bool window = false;  // present to a native window instead of the offscreen swapchain
    uint32_t width = 1280;  // swapchain (and initial client area) size
    uint32_t height = 720;
    std::string title = "playgta6 native executor";
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
    uint64_t swapchain_format_unsupported = 0;  // stream asked for a non-canvas swapchain format
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
    std::array<uint64_t, kOpcodeSlots> handled{};    // by opcode
    std::array<uint64_t, kOpcodeSlots> unhandled{};  // by opcode: not implemented yet, skipped
    uint64_t unknown_opcodes = 0;                    // opcode >= kOpcodeSlots
    std::string last_device_error;
    std::vector<std::string> device_error_log;  // the first 20 uncaptured device errors
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
    bool load_wgsl_pack(const std::string& path, std::string* error);

    // Processes WebGPU callbacks and, with a window, window messages. Returns false once the window
    // was closed (always true headless).
    bool poll();
    // True while the presenting window has keyboard focus (false headless).
    bool window_focused() const;
    // Blocks until the GPU finished all submitted work.
    void wait_idle();

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
