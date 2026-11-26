// Dawn-backed native executor (milestone 0). Every handler names the worker function it mirrors;
// docs/NATIVE_EXECUTOR_DESIGN.md has the coverage table.
#include "gpu/native_exec/native_executor.h"

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <variant>

#include "abi/playgta6_abi.h"
#include "gpu/native_exec/native_window.h"
#include "gpu/native_exec/stream_model.h"

namespace playgta6::gpu::native_exec {

namespace {

std::string to_string(wgpu::StringView sv) {
    const std::string_view v = sv;
    return std::string(v);
}

const char* backend_name(wgpu::BackendType b) {
    switch (b) {
        case wgpu::BackendType::D3D12: return "d3d12";
        case wgpu::BackendType::D3D11: return "d3d11";
        case wgpu::BackendType::Vulkan: return "vulkan";
        case wgpu::BackendType::Metal: return "metal";
        case wgpu::BackendType::OpenGL: return "opengl";
        case wgpu::BackendType::OpenGLES: return "opengles";
        case wgpu::BackendType::Null: return "null";
        default: return "other";
    }
}

// Canvas formats (WgpuSwapchain isValidCanvasFormat): the stream may switch the swapchain to them
// directly (#709). Other formats need the worker's intermediate texture + blit (not in M0).
// Codes: docs/gpu_commands.md "WebGPU enum codes".
bool canvas_format_for_code(uint8_t code, wgpu::TextureFormat* out) {
    switch (code) {
        case 22: *out = wgpu::TextureFormat::RGBA8Unorm; return true;
        case 27: *out = wgpu::TextureFormat::BGRA8Unorm; return true;
        case 40: *out = wgpu::TextureFormat::RGBA16Float; return true;
        default: return false;
    }
}

const char* format_name(wgpu::TextureFormat f) {
    switch (f) {
        case wgpu::TextureFormat::RGBA8Unorm: return "rgba8unorm";
        case wgpu::TextureFormat::BGRA8Unorm: return "bgra8unorm";
        case wgpu::TextureFormat::RGBA16Float: return "rgba16float";
        default: return "other";
    }
}

uint32_t format_bytes(wgpu::TextureFormat f) { return f == wgpu::TextureFormat::RGBA16Float ? 8 : 4; }

constexpr uint64_t kWaitForever = UINT64_MAX;

struct BeginPassCmd {
    RenderPassPlan plan;
};
struct EndPassCmd {};
using PassCmd = std::variant<BeginPassCmd, EndPassCmd>;

struct BufferEntry {
    wgpu::Buffer buffer;
    uint64_t size = 0;  // descriptor size (GPUBuffer.size, also for an invalid buffer)
    uint32_t usage = 0;
};

}  // namespace

struct NativeExecutor::Impl {
    ExecutorOptions options;
    wgpu::Instance instance;
    wgpu::Adapter adapter_handle;
    wgpu::Device device;
    wgpu::Queue queue;
    AdapterDescription adapter;
    ExecutorStats stats;
    uint64_t frame_count = 0;
    bool lost = false;

    // Swapchain: an offscreen texture headless, a configured surface with a window.
    std::unique_ptr<NativeWindow> window;
    wgpu::Surface surface;
    wgpu::TextureFormat swap_format = wgpu::TextureFormat::BGRA8Unorm;  // Chrome's preferred canvas format on Windows
    uint32_t swap_width = 0, swap_height = 0;
    bool surface_configured = false;
    wgpu::Texture offscreen;

    // Per-frame state (wgpu_worker.js _resetFrameState).
    ClearValue clear_color = kDefaultClearColor;
    std::vector<PassCmd> pass_cmds;

    std::unordered_map<uint32_t, BufferEntry> buffers;

    // ---- setup ----

    bool init(CreateStatus* status, std::string* error) {
        static const wgpu::InstanceFeatureName kFeatures[] = {wgpu::InstanceFeatureName::TimedWaitAny};
        wgpu::InstanceDescriptor idesc{};
        idesc.requiredFeatureCount = 1;
        idesc.requiredFeatures = kFeatures;
        instance = wgpu::CreateInstance(&idesc);
        if (!instance) {
            *status = CreateStatus::Failed;
            *error = "wgpuCreateInstance failed";
            return false;
        }
        swap_width = options.width ? options.width : 1;
        swap_height = options.height ? options.height : 1;
        if (options.window) {
            window = NativeWindow::create(options.title, swap_width, swap_height, error);
            if (!window) {
                *status = CreateStatus::Failed;
                return false;
            }
#ifdef _WIN32
            wgpu::SurfaceSourceWindowsHWND src{};
            src.hinstance = window->native_instance();
            src.hwnd = window->native_handle();
            wgpu::SurfaceDescriptor sdesc{};
            sdesc.nextInChain = &src;
            surface = instance.CreateSurface(&sdesc);
#endif
            if (!surface) {
                *status = CreateStatus::Failed;
                *error = "CreateSurface failed";
                return false;
            }
        }

        wgpu::RequestAdapterOptions aopts{};
        aopts.compatibleSurface = surface;
        struct AdapterResult {
            wgpu::Adapter adapter;
            std::string message;
        } ar;
        instance.WaitAny(
            instance.RequestAdapter(
                &aopts, wgpu::CallbackMode::WaitAnyOnly,
                [](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView msg, AdapterResult* out) {
                    if (s == wgpu::RequestAdapterStatus::Success) out->adapter = std::move(a);
                    else out->message = to_string(msg);
                },
                &ar),
            kWaitForever);
        if (!ar.adapter) {
            *status = CreateStatus::NoAdapter;
            *error = "no WebGPU adapter" + (ar.message.empty() ? std::string() : ": " + ar.message);
            return false;
        }
        adapter_handle = ar.adapter;
        wgpu::AdapterInfo info;
        adapter_handle.GetInfo(&info);
        adapter.backend = backend_name(info.backendType);
        adapter.vendor = to_string(info.vendor);
        adapter.architecture = to_string(info.architecture);
        adapter.device = to_string(info.device);
        adapter.description = to_string(info.description);

        // M0: a bare device, like the worker's last fallback in requestNegotiatedDevice (feature
        // and limit negotiation is milestone 1).
        wgpu::DeviceDescriptor ddesc{};
        ddesc.SetUncapturedErrorCallback(
            [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView msg, Impl* self) {
                ++self->stats.device_errors;
                self->stats.last_device_error = to_string(msg);
            },
            this);
        ddesc.SetDeviceLostCallback(
            wgpu::CallbackMode::AllowSpontaneous,
            [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView msg, Impl* self) {
                if (reason == wgpu::DeviceLostReason::Destroyed) return;  // our own teardown
                ++self->stats.device_lost;
                self->lost = true;
                self->stats.last_device_error = to_string(msg);
            },
            this);
        struct DeviceResult {
            wgpu::Device device;
            std::string message;
        } dr;
        instance.WaitAny(
            adapter_handle.RequestDevice(
                &ddesc, wgpu::CallbackMode::WaitAnyOnly,
                [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView msg, DeviceResult* out) {
                    if (s == wgpu::RequestDeviceStatus::Success) out->device = std::move(d);
                    else out->message = to_string(msg);
                },
                &dr),
            kWaitForever);
        if (!dr.device) {
            *status = CreateStatus::Failed;
            *error = "RequestDevice failed: " + dr.message;
            return false;
        }
        device = dr.device;
        queue = device.GetQueue();
        if (!surface) create_offscreen();
        *status = CreateStatus::Ok;
        return true;
    }

    void create_offscreen() {
        wgpu::TextureDescriptor td{};
        td.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc |
                   wgpu::TextureUsage::TextureBinding;
        td.dimension = wgpu::TextureDimension::e2D;
        td.size = {swap_width, swap_height, 1};
        td.format = swap_format;
        offscreen = device.CreateTexture(&td);
    }

    void configure_surface(uint32_t w, uint32_t h) {
        wgpu::SurfaceConfiguration cfg{};
        cfg.device = device;
        cfg.format = swap_format;
        cfg.usage = wgpu::TextureUsage::RenderAttachment;
        cfg.width = w;
        cfg.height = h;
        cfg.alphaMode = wgpu::CompositeAlphaMode::Opaque;  // the worker's DEFAULT_ALPHA_MODE
        cfg.presentMode = wgpu::PresentMode::Fifo;
        surface.Configure(&cfg);
        swap_width = w;
        swap_height = h;
        surface_configured = true;
    }

    // WgpuSwapchain.configureForStreamFormat for canvas formats (#709).
    void set_swapchain_format(uint8_t code) {
        wgpu::TextureFormat f;
        if (!canvas_format_for_code(code, &f)) {
            ++stats.swapchain_format_unsupported;
            return;
        }
        if (f == swap_format) return;
        swap_format = f;
        if (surface) surface_configured = false;
        else create_offscreen();
    }

    // ---- packets ----

    void reset_frame_state() {
        clear_color = kDefaultClearColor;
        pass_cmds.clear();
    }

    bool dispatch(const Packet& p) {
        const uint16_t op = p.opcode;
        bool handled = true;
        switch (static_cast<GpuOpcode>(op)) {
            case GpuOpcode::Nop:
                break;
            case GpuOpcode::BeginFrame:
                reset_frame_state();
                break;
            case GpuOpcode::ClearColor:
                if (auto c = decode_clear_color(p)) clear_color = *c;
                else ++stats.op_errors;
                break;
            case GpuOpcode::CreateBuffer:
                if (auto c = decode_create_buffer(p)) create_buffer(*c);
                else ++stats.op_errors;
                break;
            case GpuOpcode::WriteBuffer:
                if (p.payload_bytes >= sizeof(GpuWriteBufferPayload)) write_buffer(p);
                else ++stats.op_errors;
                break;
            case GpuOpcode::Present:
                ++stats.present_packets;
                present();
                break;
            case GpuOpcode::DestroyResource:
                if (p.payload_bytes >= 4) destroy_resource(read_payload<uint32_t>(p));
                break;
            case GpuOpcode::BeginRenderPass:
                if (auto plan = decode_begin_render_pass(p)) begin_render_pass(*plan);
                break;
            case GpuOpcode::EndRenderPass:
                pass_cmds.emplace_back(EndPassCmd{});
                break;
            default:
                handled = false;
                break;
        }
        if (op >= kOpcodeSlots) ++stats.unknown_opcodes;
        else if (handled) ++stats.handled[op];
        else ++stats.unhandled[op];
        return !lost;
    }

    // _handleCreateBuffer: re-creating an id replaces the buffer.
    void create_buffer(const GpuCreateBufferPayload& c) {
        wgpu::BufferDescriptor bd{};
        bd.size = c.byte_size;
        bd.usage = static_cast<wgpu::BufferUsage>(c.usage_flags);
        BufferEntry e;
        e.buffer = device.CreateBuffer(&bd);
        e.size = c.byte_size;
        e.usage = c.usage_flags;
        buffers[c.id] = std::move(e);
    }

    // _handleWriteBuffer (+ staging_pool.js stageBufferUpload). Both paths land on the queue before
    // the frame's passes, which are encoded at Present, exactly like the worker's.
    void write_buffer(const Packet& p) {
        const uint32_t id = read_payload<uint32_t>(p);
        auto it = buffers.find(id);
        if (it == buffers.end()) return;
        const WriteBufferPlan plan = plan_write_buffer(p, it->second.size);
        switch (plan.path) {
            case WriteBufferPath::Skip:
                return;
            case WriteBufferPath::Padded: {
                if (plan.copy_size == plan.data_len) {
                    queue.WriteBuffer(it->second.buffer, plan.dst_offset, plan.data, plan.data_len);
                } else {
                    std::vector<uint8_t> padded(plan.copy_size, 0);
                    std::memcpy(padded.data(), plan.data, plan.data_len);
                    queue.WriteBuffer(it->second.buffer, plan.dst_offset, padded.data(), padded.size());
                }
                return;
            }
            case WriteBufferPath::Direct: {
                if (plan.data_len > kMaxDirectWriteChunkBytes) {
                    for (uint64_t off = 0; off < plan.data_len; off += kMaxDirectWriteChunkBytes) {
                        const uint64_t n = std::min<uint64_t>(kMaxDirectWriteChunkBytes, plan.data_len - off);
                        queue.WriteBuffer(it->second.buffer, plan.dst_offset + off, plan.data + off, n);
                    }
                } else {
                    queue.WriteBuffer(it->second.buffer, plan.dst_offset, plan.data, plan.data_len);
                }
                return;
            }
        }
    }

    // _handleDestroyResource: the id leaves every table; Dawn keeps the object alive until the GPU
    // is done with it (the worker's deferred destroy).
    void destroy_resource(uint32_t id) { buffers.erase(id); }

    // render_state_packets.js _beginRenderPass: every attachment must resolve now (else the packet
    // is dropped); the pass itself is encoded at Present.
    void begin_render_pass(const RenderPassPlan& plan) {
        for (uint32_t i = 0; i < plan.color_count; ++i) {
            const ColorAttachmentPlan& c = plan.colors[i];
            if (c.texture_id == 0) continue;
            if (c.texture_id != kGpuSwapchainTextureId) {
                ++stats.render_pass_missing_texture;  // M0 creates no textures
                return;
            }
            if (c.view.format != 0) set_swapchain_format(c.view.format);
        }
        if (plan.has_depth_stencil) {
            ++stats.render_pass_missing_texture;
            return;
        }
        pass_cmds.emplace_back(BeginPassCmd{plan});
    }

    // ---- Present (wgpu_worker.js _flushPresent) ----

    struct FrameTarget {
        wgpu::Texture texture;
        wgpu::TextureView view;
        bool written = false;  // the canvas texture starts each frame as transparent black
    };

    bool acquire(FrameTarget* t) {
        if (!surface) {
            t->texture = offscreen;
            t->view = offscreen.CreateView();
            return true;
        }
        uint32_t w = 0, h = 0;
        window->client_size(&w, &h);
        if (w == 0 || h == 0) return false;  // minimized
        if (!surface_configured || w != swap_width || h != swap_height) configure_surface(w, h);
        for (int attempt = 0; attempt < 2; ++attempt) {
            wgpu::SurfaceTexture st;
            surface.GetCurrentTexture(&st);
            if (st.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal ||
                st.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
                t->texture = st.texture;
                t->view = st.texture.CreateView();
                return true;
            }
            if (st.status != wgpu::SurfaceGetCurrentTextureStatus::Outdated &&
                st.status != wgpu::SurfaceGetCurrentTextureStatus::Lost)
                return false;
            configure_surface(w, h);
        }
        return false;
    }

    // A swapchain color attachment. A 'load' of the canvas before anything was drawn this frame
    // reads transparent black (a fresh canvas texture), so it becomes a clear to 0 here.
    wgpu::RenderPassColorAttachment swapchain_attachment(FrameTarget& t, LoadOp load, StoreOp store,
                                                         const ClearValue& clear) {
        wgpu::RenderPassColorAttachment a{};
        a.view = t.view;
        if (load == LoadOp::Clear) {
            a.loadOp = wgpu::LoadOp::Clear;
            a.clearValue = {clear.r, clear.g, clear.b, clear.a};
        } else if (!t.written) {
            a.loadOp = wgpu::LoadOp::Clear;
            a.clearValue = {0, 0, 0, 0};
        } else {
            a.loadOp = wgpu::LoadOp::Load;
        }
        a.storeOp = store == StoreOp::Discard ? wgpu::StoreOp::Discard : wgpu::StoreOp::Store;
        t.written = true;
        return a;
    }

    void present() {
        FrameTarget target;
        if (lost || !acquire(&target)) {
            ++stats.present_failures;
            reset_frame_state();
            return;
        }
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
        wgpu::RenderPassEncoder active;
        bool render_pass_opened = false;
        auto end_active = [&] {
            if (active) {
                active.End();
                active = nullptr;
            }
        };
        for (const PassCmd& cmd : pass_cmds) {
            if (const auto* b = std::get_if<BeginPassCmd>(&cmd)) {
                end_active();
                const RenderPassPlan& plan = b->plan;
                std::array<AttachmentSize, kGpuMaxColorAttachments> sizes{};
                for (uint32_t i = 0; i < plan.color_count; ++i) {
                    if (plan.colors[i].texture_id == kGpuSwapchainTextureId) sizes[i] = {swap_width, swap_height};
                }
                const uint32_t dropped = drop_mismatched_colors(plan, sizes, AttachmentSize{});
                std::array<wgpu::RenderPassColorAttachment, kGpuMaxColorAttachments> colors{};
                for (uint32_t i = 0; i < plan.color_count; ++i) {
                    const ColorAttachmentPlan& c = plan.colors[i];
                    if (c.texture_id == 0 || (dropped & (1u << i))) {
                        if (dropped & (1u << i)) ++stats.render_pass_dropped_colors;
                        colors[i] = wgpu::RenderPassColorAttachment{};  // null attachment
                        continue;
                    }
                    colors[i] = swapchain_attachment(target, c.load, c.store, c.clear);
                }
                wgpu::RenderPassDescriptor rp{};
                rp.colorAttachmentCount = plan.color_count;
                rp.colorAttachments = colors.data();
                active = encoder.BeginRenderPass(&rp);
                render_pass_opened = true;
            } else {
                end_active();  // EndRenderPass
            }
        }
        end_active();
        if (!render_pass_opened) {
            // The v1 clearing swapchain pass (_ensurePresentRenderPass with renderPassOpened false).
            wgpu::RenderPassColorAttachment a = swapchain_attachment(target, LoadOp::Clear, StoreOp::Store, clear_color);
            wgpu::RenderPassDescriptor rp{};
            rp.colorAttachmentCount = 1;
            rp.colorAttachments = &a;
            encoder.BeginRenderPass(&rp).End();
        }
        wgpu::CommandBuffer cb = encoder.Finish();
        queue.Submit(1, &cb);
        ++frame_count;
        ++stats.frames_submitted;
        if (surface) surface.Present();
        reset_frame_state();
    }

    // ---- read-back helpers ----

    bool map_read(wgpu::Buffer& staging, uint64_t size, std::string* error) {
        struct MapResult {
            bool ok = false;
            std::string message;
        } mr;
        instance.WaitAny(staging.MapAsync(
                             wgpu::MapMode::Read, 0, static_cast<size_t>(size), wgpu::CallbackMode::WaitAnyOnly,
                             [](wgpu::MapAsyncStatus s, wgpu::StringView msg, MapResult* out) {
                                 out->ok = s == wgpu::MapAsyncStatus::Success;
                                 if (!out->ok) out->message = to_string(msg);
                             },
                             &mr),
                         kWaitForever);
        if (!mr.ok && error) *error = "MapAsync failed: " + mr.message;
        return mr.ok;
    }

    // Runs `record` inside a validation error scope; false (and *error) if WebGPU rejected it.
    template <class F>
    bool validated(F&& record, std::string* error) {
        device.PushErrorScope(wgpu::ErrorFilter::Validation);
        record();
        struct ScopeResult {
            bool error = false;
            std::string message;
        } sr;
        instance.WaitAny(device.PopErrorScope(
                             wgpu::CallbackMode::WaitAnyOnly,
                             [](wgpu::PopErrorScopeStatus, wgpu::ErrorType type, wgpu::StringView msg, ScopeResult* out) {
                                 out->error = type != wgpu::ErrorType::NoError;
                                 out->message = to_string(msg);
                             },
                             &sr),
                         kWaitForever);
        if (sr.error && error) *error = "validation error: " + sr.message;
        return !sr.error;
    }
};

std::unique_ptr<NativeExecutor> NativeExecutor::create(const ExecutorOptions& options, CreateStatus* status,
                                                       std::string* error) {
    CreateStatus local_status;
    std::string local_error;
    if (!status) status = &local_status;
    if (!error) error = &local_error;
    auto impl = std::make_unique<Impl>();
    impl->options = options;
    if (!impl->init(status, error)) return nullptr;
    return std::unique_ptr<NativeExecutor>(new NativeExecutor(std::move(impl)));
}

NativeExecutor::NativeExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

NativeExecutor::~NativeExecutor() {
    if (impl_ && impl_->device) {
        impl_->buffers.clear();
        impl_->device.Destroy();
    }
}

uint32_t NativeExecutor::execute(const uint8_t* data, size_t len) {
    Impl& s = *impl_;
    ++s.stats.batches;
    if (s.lost) {
        ++s.stats.dropped_while_lost;
        return 0;
    }
    const uint64_t presents_before = s.stats.present_packets;
    const WalkResult r = walk_packets(data, len, [&](const Packet& p) { return s.dispatch(p); });
    s.stats.packets += r.packets;
    s.stats.bytes += r.bytes;
    if (r.truncated) ++s.stats.truncated_batches;
    s.instance.ProcessEvents();
    return static_cast<uint32_t>(s.stats.present_packets - presents_before);
}

bool NativeExecutor::poll() {
    impl_->instance.ProcessEvents();
    if (impl_->window) return impl_->window->pump();
    return true;
}

void NativeExecutor::wait_idle() {
    Impl& s = *impl_;
    s.instance.WaitAny(s.queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                                                   [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}),
                       kWaitForever);
}

uint64_t NativeExecutor::frame_count() const { return impl_->frame_count; }
const ExecutorStats& NativeExecutor::stats() const { return impl_->stats; }
const AdapterDescription& NativeExecutor::adapter() const { return impl_->adapter; }

std::string NativeExecutor::stats_json() const {
    const ExecutorStats& st = impl_->stats;
    std::string j = "{";
    auto num = [&](const char* k, uint64_t v) {
        j += "\"";
        j += k;
        j += "\": " + std::to_string(v) + ", ";
    };
    num("batches", st.batches);
    num("packets", st.packets);
    num("bytes", st.bytes);
    num("present_packets", st.present_packets);
    num("frames_submitted", st.frames_submitted);
    num("present_failures", st.present_failures);
    num("truncated_batches", st.truncated_batches);
    num("op_errors", st.op_errors);
    num("device_errors", st.device_errors);
    num("device_lost", st.device_lost);
    num("dropped_while_lost", st.dropped_while_lost);
    num("render_pass_missing_texture", st.render_pass_missing_texture);
    num("render_pass_dropped_colors", st.render_pass_dropped_colors);
    num("swapchain_format_unsupported", st.swapchain_format_unsupported);
    num("unknown_opcodes", st.unknown_opcodes);
    auto table = [&](const char* k, const std::array<uint64_t, kOpcodeSlots>& t) {
        j += "\"";
        j += k;
        j += "\": {";
        bool first = true;
        for (size_t i = 0; i < t.size(); ++i) {
            if (!t[i]) continue;
            if (!first) j += ", ";
            first = false;
            j += "\"" + std::to_string(i) + "\": " + std::to_string(t[i]);
        }
        j += "}";
    };
    table("handled", st.handled);
    j += ", ";
    table("unhandled", st.unhandled);
    j += ", \"adapter\": {\"backend\": \"" + impl_->adapter.backend + "\"}";
    j += "}";
    return j;
}

bool NativeExecutor::read_last_frame(FrameImage* out, std::string* error) {
    Impl& s = *impl_;
    if (!s.offscreen) {
        if (error) *error = "read_last_frame: only the headless swapchain can be read back";
        return false;
    }
    const uint32_t bpp = format_bytes(s.swap_format);
    const uint32_t row = s.swap_width * bpp;
    const uint32_t padded_row = (row + 255u) & ~255u;
    const uint64_t size = uint64_t{padded_row} * s.swap_height;
    wgpu::BufferDescriptor bd{};
    bd.size = size;
    bd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer staging = s.device.CreateBuffer(&bd);
    const bool ok = s.validated(
        [&] {
            wgpu::CommandEncoder enc = s.device.CreateCommandEncoder();
            wgpu::TexelCopyTextureInfo src{};
            src.texture = s.offscreen;
            wgpu::TexelCopyBufferInfo dst{};
            dst.buffer = staging;
            dst.layout.bytesPerRow = padded_row;
            dst.layout.rowsPerImage = s.swap_height;
            wgpu::Extent3D extent{s.swap_width, s.swap_height, 1};
            enc.CopyTextureToBuffer(&src, &dst, &extent);
            wgpu::CommandBuffer cb = enc.Finish();
            s.queue.Submit(1, &cb);
        },
        error);
    if (!ok || !s.map_read(staging, size, error)) return false;
    const auto* mapped = static_cast<const uint8_t*>(staging.GetConstMappedRange(0, static_cast<size_t>(size)));
    out->width = s.swap_width;
    out->height = s.swap_height;
    out->format = format_name(s.swap_format);
    out->bytes_per_pixel = bpp;
    out->pixels.resize(size_t{row} * s.swap_height);
    for (uint32_t y = 0; y < s.swap_height; ++y) {
        std::memcpy(out->pixels.data() + size_t{row} * y, mapped + size_t{padded_row} * y, row);
    }
    staging.Unmap();
    return true;
}

bool NativeExecutor::read_buffer(uint32_t id, uint64_t offset, uint64_t size, std::vector<uint8_t>* out,
                                 std::string* error) {
    Impl& s = *impl_;
    auto it = s.buffers.find(id);
    if (it == s.buffers.end()) {
        if (error) *error = "read_buffer: unknown buffer " + std::to_string(id);
        return false;
    }
    const uint64_t copy = (size + 3) & ~uint64_t{3};
    wgpu::BufferDescriptor bd{};
    bd.size = copy;
    bd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer staging = s.device.CreateBuffer(&bd);
    const bool ok = s.validated(
        [&] {
            wgpu::CommandEncoder enc = s.device.CreateCommandEncoder();
            enc.CopyBufferToBuffer(it->second.buffer, offset, staging, 0, copy);
            wgpu::CommandBuffer cb = enc.Finish();
            s.queue.Submit(1, &cb);
        },
        error);
    if (!ok || !s.map_read(staging, copy, error)) return false;
    const auto* mapped = static_cast<const uint8_t*>(staging.GetConstMappedRange(0, static_cast<size_t>(copy)));
    out->assign(mapped, mapped + size);
    staging.Unmap();
    return true;
}

}  // namespace playgta6::gpu::native_exec
