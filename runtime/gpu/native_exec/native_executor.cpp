// Dawn-backed native executor (milestones 0 and 1). Every handler names the worker function it
// mirrors (web/workers/wgpu_worker.js, render_state_packets.js, texture_packets.js);
// docs/NATIVE_EXECUTOR_DESIGN.md has the coverage table.
#include "gpu/native_exec/native_executor.h"

#include <webgpu/webgpu_cpp.h>

#include <dawn/native/DawnNative.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "abi/playgta6_abi.h"
#include "gpu/drop_table.h"
#include "gpu/gpu_time.h"
#include "gpu/present_pacing.h"
#include "gpu/native_exec/blit_model.h"
#include "gpu/native_exec/window_script.h"
#include "gpu/native_exec/blob_cache.h"
#include "gpu/native_exec/clear_model.h"
#include "gpu/native_exec/error_scope_model.h"
#include "gpu/native_exec/fragment_outputs_model.h"
#include "gpu/native_exec/layout_model.h"
#include "gpu/native_exec/m1_model.h"
#include "gpu/native_exec/m2_model.h"
#include "gpu/native_exec/native_window.h"
#include "gpu/native_exec/overlay_model.h"
#include "gpu/native_exec/pipe_budget_model.h"
#include "gpu/native_exec/shader_text_model.h"
#include "gpu/native_exec/present_model.h"
#include "gpu/native_exec/reflection_model.h"
#include "gpu/native_exec/replay_frames.h"
#include "gpu/native_exec/screenshot_model.h"
#include "gpu/native_exec/shader_workers.h"
#include "gpu/native_exec/show_target_model.h"
#include "gpu/native_exec/stream_model.h"
#include "os/process_memory.h"

namespace playgta6::gpu::native_exec {

namespace {

// Set on the shader worker threads: a device error raised there is queued for the executor thread
// instead of touching the stats from a second thread.
thread_local bool tl_shader_worker = false;

uint64_t host_private_mb() {
    uint64_t priv = 0, ws = 0;
    return playgta6::os::process_memory_bytes(&priv, &ws, nullptr) ? priv >> 20 : 0;
}

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
// directly (#709). Other formats need the worker's intermediate texture + blit (not yet).
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

// ---- code tables (docs/gpu_commands.md "WebGPU enum codes") ----

wgpu::TextureFormat wgpu_format(uint8_t code) {
    using F = wgpu::TextureFormat;
    static const F kT[kFormatCodeCount] = {
        F::Undefined, F::R8Unorm, F::R8Snorm, F::R8Uint, F::R8Sint, F::R16Unorm, F::R16Snorm, F::R16Uint,
        F::R16Sint, F::R16Float, F::RG8Unorm, F::RG8Snorm, F::RG8Uint, F::RG8Sint, F::R32Uint, F::R32Sint,
        F::R32Float, F::RG16Unorm, F::RG16Snorm, F::RG16Uint, F::RG16Sint, F::RG16Float, F::RGBA8Unorm,
        F::RGBA8UnormSrgb, F::RGBA8Snorm, F::RGBA8Uint, F::RGBA8Sint, F::BGRA8Unorm, F::BGRA8UnormSrgb,
        F::RGB9E5Ufloat, F::RGB10A2Uint, F::RGB10A2Unorm, F::RG11B10Ufloat, F::RG32Uint, F::RG32Sint,
        F::RG32Float, F::RGBA16Unorm, F::RGBA16Snorm, F::RGBA16Uint, F::RGBA16Sint, F::RGBA16Float,
        F::RGBA32Uint, F::RGBA32Sint, F::RGBA32Float, F::Stencil8, F::Depth16Unorm, F::Depth24Plus,
        F::Depth24PlusStencil8, F::Depth32Float, F::Depth32FloatStencil8, F::BC1RGBAUnorm,
        F::BC1RGBAUnormSrgb, F::BC2RGBAUnorm, F::BC2RGBAUnormSrgb, F::BC3RGBAUnorm, F::BC3RGBAUnormSrgb,
        F::BC4RUnorm, F::BC4RSnorm, F::BC5RGUnorm, F::BC5RGSnorm, F::BC6HRGBUfloat, F::BC6HRGBFloat,
        F::BC7RGBAUnorm, F::BC7RGBAUnormSrgb};
    return kT[code < kFormatCodeCount ? code : 0];
}

wgpu::VertexFormat wgpu_vertex_format(uint8_t code, uint32_t* bytes) {
    using V = wgpu::VertexFormat;
    struct E {
        V f;
        uint32_t b;
    };
    static const E kT[42] = {
        {V::Uint8, 0},      {V::Uint8, 1},      {V::Uint8x2, 2},    {V::Uint8x4, 4},    {V::Sint8, 1},
        {V::Sint8x2, 2},    {V::Sint8x4, 4},    {V::Unorm8, 1},     {V::Unorm8x2, 2},   {V::Unorm8x4, 4},
        {V::Snorm8, 1},     {V::Snorm8x2, 2},   {V::Snorm8x4, 4},   {V::Uint16, 2},     {V::Uint16x2, 4},
        {V::Uint16x4, 8},   {V::Sint16, 2},     {V::Sint16x2, 4},   {V::Sint16x4, 8},   {V::Unorm16, 2},
        {V::Unorm16x2, 4},  {V::Unorm16x4, 8},  {V::Snorm16, 2},    {V::Snorm16x2, 4},  {V::Snorm16x4, 8},
        {V::Float16, 2},    {V::Float16x2, 4},  {V::Float16x4, 8},  {V::Float32, 4},    {V::Float32x2, 8},
        {V::Float32x3, 12}, {V::Float32x4, 16}, {V::Uint32, 4},     {V::Uint32x2, 8},   {V::Uint32x3, 12},
        {V::Uint32x4, 16},  {V::Sint32, 4},     {V::Sint32x2, 8},   {V::Sint32x3, 12},  {V::Sint32x4, 16},
        {V::Unorm10_10_10_2, 4}, {V::Unorm8x4BGRA, 4}};
    if (code == 0 || code >= 42) {
        *bytes = 0;
        return V::Uint8;
    }
    *bytes = kT[code].b;
    return kT[code].f;
}

wgpu::BlendFactor wgpu_blend_factor(uint8_t c) {
    using B = wgpu::BlendFactor;
    static const B kT[17] = {B::Zero, B::One, B::Src, B::OneMinusSrc, B::SrcAlpha, B::OneMinusSrcAlpha, B::Dst,
                             B::OneMinusDst, B::DstAlpha, B::OneMinusDstAlpha, B::SrcAlphaSaturated, B::Constant,
                             B::OneMinusConstant, B::Src1, B::OneMinusSrc1, B::Src1Alpha, B::OneMinusSrc1Alpha};
    return c < 17 ? kT[c] : B::One;  // decodeBlendComponent: `|| 'one'`
}
wgpu::BlendFactor wgpu_blend_dst_factor(uint8_t c) { return c < 17 ? wgpu_blend_factor(c) : wgpu::BlendFactor::Zero; }
wgpu::BlendOperation wgpu_blend_op(uint8_t c) {
    using O = wgpu::BlendOperation;
    static const O kT[5] = {O::Add, O::Subtract, O::ReverseSubtract, O::Min, O::Max};
    return c < 5 ? kT[c] : O::Add;
}
// COMPARE_FUNCTION_NAMES; 0 / out of range -> Undefined.
wgpu::CompareFunction wgpu_compare(uint8_t c) {
    using C = wgpu::CompareFunction;
    static const C kT[9] = {C::Undefined, C::Never, C::Less, C::Equal, C::LessEqual, C::Greater, C::NotEqual,
                            C::GreaterEqual, C::Always};
    return c < 9 ? kT[c] : C::Undefined;
}
wgpu::StencilOperation wgpu_stencil_op(uint8_t c) {
    using S = wgpu::StencilOperation;
    static const S kT[8] = {S::Keep, S::Zero, S::Replace, S::Invert, S::IncrementClamp, S::DecrementClamp,
                            S::IncrementWrap, S::DecrementWrap};
    return c < 8 ? kT[c] : S::Keep;
}
wgpu::PrimitiveTopology wgpu_topology(uint8_t c) {
    using T = wgpu::PrimitiveTopology;
    static const T kT[5] = {T::TriangleList, T::TriangleStrip, T::LineList, T::LineStrip, T::PointList};
    return c < 5 ? kT[c] : T::TriangleList;
}
wgpu::TextureViewDimension wgpu_view_dim(uint8_t c) {
    using D = wgpu::TextureViewDimension;
    static const D kT[7] = {D::Undefined, D::e1D, D::e2D, D::e2DArray, D::Cube, D::CubeArray, D::e3D};
    return c < 7 ? kT[c] : D::Undefined;
}
wgpu::AddressMode wgpu_address(uint32_t c) {
    using A = wgpu::AddressMode;
    return c == 1 ? A::Repeat : c == 2 ? A::MirrorRepeat : A::ClampToEdge;
}

// resolveStorageTextureFormat (built-in tables; texture_formats.js is not loaded in the replay page).
struct StorageResolution {
    uint8_t storage_format = 0;  // 0 = none
    bool supported = false;
    bool linear_sibling = false;
};

struct DeviceFeatures {
    bool bc = false, bc_sliced_3d = false, depth32_stencil8 = false, float32_filterable = false,
         float32_blendable = false, bgra8_storage = false, rg11b10_renderable = false, tier1 = false,
         timestamp_query = false, timestamp_inside_passes = false;
};

StorageResolution resolve_storage_format(uint8_t f, const DeviceFeatures& d) {
    StorageResolution r;
    if (!format_valid(f) || format_is_depth_or_stencil(f) || format_is_bc(f)) return r;
    switch (f) {
        case 22: case 24: case 25: case 26: case 38: case 39: case 40: case 14: case 15: case 16: case 33:
        case 34: case 35: case 41: case 42: case 43:
            r.storage_format = f;
            r.supported = true;
            return r;
        case 27:
            if (d.bgra8_storage) { r.storage_format = 27; r.supported = true; }
            return r;
        case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: case 9: case 10: case 11: case 12:
        case 13: case 17: case 18: case 19: case 20: case 21: case 30: case 31: case 32: case 36: case 37:
            if (d.tier1) { r.storage_format = f; r.supported = true; }
            return r;
        case 23:
            r.storage_format = 22; r.supported = true; r.linear_sibling = true;
            return r;
        case 28:
            if (d.bgra8_storage) { r.storage_format = 27; r.supported = true; r.linear_sibling = true; }
            return r;
        default: return r;
    }
}

bool render_attachment_supported(uint8_t f, const DeviceFeatures& d) {
    if (!format_valid(f) || format_is_bc(f)) return false;
    if (f == 29 || f == 2 || f == 11 || f == 24) return false;  // NON_RENDERABLE_CORE_FORMATS
    if (f == 32) return d.rg11b10_renderable;
    if (f == 5 || f == 6 || f == 17 || f == 18 || f == 36 || f == 37) return d.tier1;
    return true;
}

// isBlendableTextureFormat
bool blendable(uint8_t f, const DeviceFeatures& d) {
    if (!format_valid(f)) return false;
    if (format_is_integer(f) || format_is_depth_or_stencil(f) || format_is_bc(f)) return false;
    switch (f) {
        case 1: case 9: case 10: case 21: case 22: case 23: case 27: case 28: case 31: case 40: return true;
        default: break;
    }
    if (format_is_float32(f)) return d.float32_blendable;
    if (f == 32) return d.rg11b10_renderable;
    if (f == 5 || f == 6 || f == 17 || f == 18 || f == 36 || f == 37) return d.tier1;
    return false;
}

struct LayoutEntry {  // decodeBindGroupLayoutEntry (kind: 0 buffer, 1 sampler, 2 texture, 3 storage texture)
    uint32_t binding = 0, visibility = 0;
    uint8_t kind = 0, subtype = 0, view_dim = 2, storage_format = 0;
    bool dynamic = false, multisampled = false;
    uint64_t min_binding_size = 0;
};

struct BglEntry {
    wgpu::BindGroupLayout layout;
    std::vector<LayoutEntry> entries;
    bool entries_known = true;
    // Over a per-stage limit on its own (an invalid layout): its bind groups are recorded for the remap
    // but not created, and never bound (an invalid bind group in a pass invalidates the command buffer).
    bool over_limits = false;
};

struct PlEntry {
    wgpu::PipelineLayout layout;
    bool group0_has_entries = false;
    // The worker's `pipelineLayoutMeta.group0Entries`: nullopt for a manifest layout or an empty slot.
    std::optional<std::vector<Group0LayoutEntry>> group0_entries;
    // A v2 explicit layout's groups (empty for a manifest layout): each group's layout, its entries
    // as the WGSL check and the per-stage limit counts read them (fit_pipeline_layout).
    std::vector<wgpu::BindGroupLayout> bgls;
    std::vector<FitExplicitGroup> fit_groups;
    std::vector<std::vector<LayoutEntry>> entry_groups;
};

// Groups of a fitted pipeline layout the stream does not know about (built from the WGSL, or padding):
// bound with dummy resources before a draw / dispatch unless the pass bound that group itself.
struct FitDummies {
    uint8_t mask = 0;
    uint8_t force = 0;  // of `mask`: the title's groups replaced by padding (bound over whatever the pass bound)
    std::array<wgpu::BindGroup, kFitMaxGroups> groups;
};

// A bind group's recorded descriptor (`bindGroupDescriptors`): the real (non-dummy) resources with
// the facts the layout:'auto' remap matches on, and the handles to bind.
struct BgHandles {
    wgpu::Buffer buffer;
    wgpu::TextureView view;
    wgpu::Sampler sampler;
};
struct BgDesc {
    std::vector<RemapResource> model;
    std::vector<BgHandles> handles;  // parallel to `model`
};

// `renderPipelineAutoLayoutMeta` / `computePipelineMeta`: a pipeline whose explicit layout did not
// fit its WGSL gets layout:'auto' and the worker's group 0 bind group remap.
struct AutoLayoutMeta {
    bool remap = false;
    std::vector<WgslBinding> bindings;
    uint64_t gen = 0;  // bumps when the pipeline id is (re)created: invalidates cached remapped groups
    // Group 0 layout built up front from the shader reflection (the pipeline was created with it as an
    // explicit layout): the remap binds against it without the pipeline having finished compiling.
    wgpu::BindGroupLayout reflected_bgl0;
    // A fitted group 0 that keeps the title's binding numbers (its layout was over the per-stage limits):
    // the remap binds each WGSL binding from the same binding of the title's bind group, and a missing
    // one gets a dummy of the title's entry.
    bool by_binding = false;
    std::vector<LayoutEntry> by_binding_entries;
};

struct RemapCacheEntry {
    wgpu::BindGroup group;  // null: the remap came out empty
    const BgDesc* desc = nullptr;
    uint64_t gen = 0;
};

// A shader whose WGSL comes from the pack is materialized (read from the pack, entry point scanned,
// module compiled) on first use by a pipeline: a title creates ~1400 modules of which only a few
// are ever used, and compiling each synchronously (about 50 ms for multi-MB WGSL) stalled the
// executor for over a minute. Fields are mutable: ensure_shader() fills them through const refs.
// Live mode hands the materialization to a shader worker (shader_workers.h): while `queued` and not
// `ready`, the worker owns every mutable field below; the queue thread reads them after `ready`.
struct ShaderEntry {
    mutable wgpu::ShaderModule module;
    mutable std::string code;
    mutable std::optional<std::string> explicit_ep;
    uint32_t stage = 0;
    uint64_t hash = 0;  // the pack hash of a hash-only payload (0 = inline WGSL): the GPU-time key
    uint64_t reflect_hash = 0;  // the worker's `hashHex`: the pack hash, else FNV-1a 64 of the WGSL text
    mutable bool pending = false;  // pack-backed and not yet materialized (nor handed to a worker)
    // Text scans of `code` memoized per shader (a compute WGSL can be 16 MB; every pipeline built
    // from the module rescanned it): resolved entry point per stage, statically referenced @group(0),
    // whether it declares @group(0) at all, the fragment outputs of an entry point.
    mutable std::string ep_cache[3];
    mutable std::optional<std::vector<WgslBinding>> group0_cache;
    mutable std::optional<std::vector<std::vector<WgslBinding>>> groups_cache;  // every @group (layout fit)
    mutable std::optional<bool> declares_group0_cache;
    mutable std::string frag_mask_ep;
    mutable std::optional<std::optional<uint32_t>> frag_mask_cache;
    // Worker hand-off (queue thread: `queued`, `accounted`; worker: the results, then `ready`).
    mutable bool queued = false;
    mutable bool accounted = false;  // the worker's timings were folded into the executor stats
    mutable std::atomic<bool> ready{false};
    mutable uint64_t worker_ns = 0;       // module compile on the worker
    mutable uint64_t worker_scan_ns = 0;  // text scans on the worker
    mutable uint32_t worker_scans = 0;
    mutable bool worker_miss = false;     // the pack no longer delivered it (compiled as the hash text)
    mutable uint64_t worker_code_bytes = 0;  // size of the WGSL the worker compiled (`code` is dropped afterwards)
    mutable bool code_dropped = false;       // pack-backed text released after its scans + module (code_of() re-reads it)
    mutable uint64_t worker_private_mb = 0;  // host process private bytes right after the compile, MB
};
using ShaderRef = std::shared_ptr<ShaderEntry>;

// Result of an asynchronous pipeline creation (ExecutorOptions::async_pipelines), shared with the
// completion callback. The callback runs on the executor thread (ProcessEvents / WaitAny).
struct PendingPipe {
    bool done = false;
    wgpu::CreatePipelineAsyncStatus status = wgpu::CreatePipelineAsyncStatus::Success;
    std::string message;
    wgpu::RenderPipeline render;
    wgpu::ComputePipeline compute;
};

// A pipeline whose compile may still be running on a Dawn worker thread: `pipeline` is only valid
// once Impl::resolve() has returned it.
struct AsyncSlot {
    mutable std::shared_ptr<PendingPipe> pending;  // null: created synchronously / already resolved
    mutable wgpu::Future future;
    bool fullscreen = false;  // vertex-buffer-less pipeline: the long wait budget (pipe_budget_model.h)
    mutable bool failed = false;  // the async compile failed: draws / dispatches using it are dropped, never submitted
    uint64_t vs_hash = 0, fs_hash = 0;  // stage shader hashes, for the failure log
    // Deferred: a shader of it is still being materialized on a shader worker, so the pipeline is not
    // created yet (Impl::deferred_render / deferred_compute hold its packet). `pending` is set and never done.
    mutable bool deferred = false;
    uint32_t deferred_id = 0;
    bool deferred_compute = false;
};

struct RenderPipelineEntry : AsyncSlot {
    mutable wgpu::RenderPipeline pipeline;
    AutoLayoutMeta auto_meta;
    bool expects_group0 = false;
    uint64_t perf_hash = 0;  // GPU-time key: the fragment shader's hash, else the vertex shader's
    uint64_t quarantined_hash = 0;  // the stage hash that is on the shader quarantine (draws are skipped), else 0
    FitDummies fit;
};

struct ComputePipelineEntry : AsyncSlot {
    mutable wgpu::ComputePipeline pipeline;
    FitDummies fit;
};

struct BufferEntry {
    wgpu::Buffer buffer;
    uint64_t size = 0;  // descriptor size (GPUBuffer.size, also for an invalid buffer)
    uint32_t usage = 0;
};

struct TwinSlot {
    wgpu::Texture tex;
    uint64_t gen = 0;  // depth_gen the twin was last copied at
};

struct TextureEntry {
    wgpu::Texture texture;
    wgpu::TextureView default_view;
    TextureGeometry geom;
    uint32_t usage = 0;
    uint32_t sample_count = 1;
    bool promoted_1d = false;
    bool bc3d_fallback = false;  // 3D BC texture created as rgba8unorm: uploads are skipped
    std::unordered_map<uint64_t, wgpu::TextureView> views;
    // Depth twins (color copies of the depth plane bound where a layout entry cannot take a depth
    // view, see depth_twin below): each refreshed when depth_gen moved past its slot's gen.
    std::array<TwinSlot, kDepthReadCount> twins;  // indexed by DepthRead (Twin* kinds only)
    uint64_t depth_gen = 1;  // bumps on every write to the depth texture
};

// ---- pass commands (wgpu_worker.js `_passCommands`) ----

struct ColorAtt {
    uint32_t texture_id = 0;
    wgpu::TextureView view;  // resolved at packet time
    uint32_t base_mip = 0, base_layer = 0;  // the attachment view's subresource (--show-target default)
    bool swapchain = false;
    AttachmentSize size;  // texture attachments; a swapchain attachment's size is read at encode time
    LoadOp load = LoadOp::Load;
    StoreOp store = StoreOp::Store;
    ClearValue clear;
};
struct BeginPassCmd {
    uint32_t color_count = 0;
    std::array<ColorAtt, kGpuMaxColorAttachments> colors;
    bool has_ds = false;
    wgpu::TextureView ds_view;
    AttachmentSize ds_size;
    GpuRenderPassDepthStencilAttachment ds{};
};
struct EndPassCmd {};
struct SetPipelineCmd {
    uint32_t id;
    wgpu::RenderPipeline at_packet;
};
struct SetVertexBufferCmd {
    uint32_t slot, id;
    uint64_t offset;
};
struct SetIndexBufferCmd {
    uint32_t id, format;
    uint64_t offset;
};
struct DrawCmd {
    uint32_t vertex_count, instance_count, first_vertex, first_instance;
};
struct DrawIndexedCmd {
    uint32_t index_count, instance_count, first_index;
    int32_t base_vertex;
    uint32_t first_instance;
};
struct IndirectCmd {
    uint8_t kind;  // 0 draw, 1 drawIndexed, 2 dispatch
    uint32_t id;
    uint64_t offset;
};
struct SetComputePipelineCmd {
    uint32_t id;
};
struct DispatchCmd {
    uint32_t x, y, z;
};
struct ViewportCmd {
    Viewport vp;
};
struct ScissorCmd {
    uint32_t x, y, w, h;
};
struct BlendConstantCmd {
    float r, g, b, a;
};
struct StencilRefCmd {
    uint32_t ref;
};
struct SetBindGroup2Cmd {
    uint32_t group;
    wgpu::BindGroup bind_group;  // resolved at packet time
    std::vector<uint32_t> offsets;
    uint32_t bind_group_id = 0;
    std::shared_ptr<BgDesc> desc;  // the recorded descriptor at packet time (the layout:'auto' remap reads it)
};
// v1 SetBindGroup: the group is resolved by id when the frame is encoded (`this.bindGroups.get(id)`
// with the packet-time handle as fallback), and goes to whichever pass is open.
struct SetBindGroup1Cmd {
    uint32_t group, id;
    wgpu::BindGroup at_packet;
};
// ReadbackBuffer / ReadbackTexture: runs in stream order with the frame's passes.
struct ReadbackCmd {
    ReadbackRequest req;
};
// CopyTexture (kGpuOpcodeCopyTexture): a texture-to-texture copy, in stream order like a read-back.
struct CopyTextureCmd {
    GpuCopyTexturePayload copy;
};
using PassCmd = std::variant<BeginPassCmd, EndPassCmd, SetPipelineCmd, SetVertexBufferCmd, SetIndexBufferCmd,
                             DrawCmd, DrawIndexedCmd, IndirectCmd, SetComputePipelineCmd, DispatchCmd, ViewportCmd,
                             ScissorCmd, BlendConstantCmd, StencilRefCmd, SetBindGroup2Cmd, SetBindGroup1Cmd,
                             ReadbackCmd, CopyTextureCmd>;

// One query set per frame in flight (GpuTimeRing slot): pass begin / end timestamps, resolved into a
// buffer and copied to a mappable one at the end of the submit.
constexpr size_t kTimerSlots = 4;
constexpr size_t kTimerMaxPasses = 256;
struct TimerSlot {
    wgpu::QuerySet queries;
    wgpu::Buffer resolve;
    wgpu::Buffer staging;
};

struct Impl_ReadbackEntry {
    ReadbackRequest req;
    ReadbackLayout layout;
    wgpu::Buffer staging;  // null: the copy was not recorded (the counter still advances)
};

}  // namespace

struct NativeExecutor::Impl {
    ExecutorOptions options;
    BlobCache blob_cache;  // before the device: Dawn may call the hooks until the device is gone
    wgpu::Instance instance;
    std::unique_ptr<dawn::native::Instance> native_instance;  // --adapter warp: enumerates the CPU adapter
    wgpu::Adapter adapter_handle;
    wgpu::Device device;
    wgpu::Queue queue;
    AdapterDescription adapter;
    ExecutorStats stats;
    DeviceFeatures features;
    uint32_t max_workgroups_per_dim = 65535;
    StageLimits stage_limits;  // the device's per-stage binding limits (fit_pipeline_layout)
    uint64_t fit_dummy_owner = 0;
    uint64_t frame_count = 0;
    std::atomic<bool> lost{false};  // set by Dawn's device-lost callback (any thread)
    bool blob_cache_opened = false;

    // Swapchain: an offscreen texture headless, a configured surface with a window.
    std::unique_ptr<NativeWindow> window;
    wgpu::Surface surface;
    wgpu::TextureFormat swap_format = wgpu::TextureFormat::BGRA8Unorm;  // Chrome's preferred canvas format on Windows
    uint32_t swap_width = 0, swap_height = 0;  // the canvas (surface / offscreen texture) size
    bool surface_configured = false;
    SurfaceState surface_state;  // reconfigure state machine (window_script.h)
    WindowScriptRunner script;
    uint64_t script_start_ns = 0;
    bool probe_pending = false;
    // Screenshots (F12 / --screenshot-at): the stream image of a present is copied into a read-back buffer
    // inside that frame's submit, mapped asynchronously (delivered by ProcessEvents, never waited for), then
    // converted / PNG-encoded / written on a short-lived worker thread. The pump thread only ever sets the
    // window title.
    std::filesystem::path shot_dir;
    size_t shot_next = 0;  // index into options.screenshot_at
    std::mutex shot_mu;
    std::vector<std::thread> shot_threads;
    std::atomic<uint64_t> shot_saved{0}, shot_failed{0};
    std::atomic<bool> shot_done{false};
    std::string shot_last_path;  // under shot_mu
    std::string last_title;
    uint64_t shot_title_until_ns = 0;
    wgpu::Texture offscreen;
    // The stream's own render target (WgpuSwapchain `_streamFormat` + intermediate texture): used when the
    // stream format is not a canvas format, or the stream size differs from the canvas size. Presenting
    // then blits it into the canvas (blit_model.h). Otherwise the stream renders into the canvas directly.
    wgpu::TextureFormat stream_format = wgpu::TextureFormat::BGRA8Unorm;
    uint32_t stream_w = 0, stream_h = 0;
    wgpu::Texture intermediate;
    wgpu::TextureView intermediate_view;
    wgpu::TextureFormat intermediate_format = wgpu::TextureFormat::Undefined;
    uint32_t intermediate_w = 0, intermediate_h = 0;
    wgpu::ShaderModule blit_module;
    wgpu::Sampler blit_sampler;
    wgpu::RenderPipeline blit_pipeline;
    wgpu::TextureFormat blit_pipeline_format = wgpu::TextureFormat::Undefined;
    wgpu::BindGroup blit_bind_group;

    // Per-frame state (wgpu_worker.js _resetFrameState).
    ClearValue clear_color = kDefaultClearColor;
    std::vector<PassCmd> pass_cmds;

    std::unordered_map<uint32_t, BufferEntry> buffers;
    std::unordered_map<uint32_t, TextureEntry> textures;
    // ClearTexture (clear_model.h): textures whose zeroing is encoded at the start of the next
    // encoder that is submitted (or flushed before a write that needs it).
    std::unordered_map<uint32_t, PendingClear> pending_clears;
    wgpu::Buffer zero_buffer;  // CopySrc, never written: Dawn zero-initializes it
    std::unordered_map<uint32_t, wgpu::Sampler> samplers;
    std::unordered_map<uint32_t, ShaderRef> shaders;
    std::unordered_map<uint32_t, BglEntry> bgls;
    std::unordered_map<uint32_t, PlEntry> pipeline_layouts;
    std::unordered_map<uint64_t, wgpu::BindGroupLayout> layouts_by_content;
    std::unordered_map<uint32_t, RenderPipelineEntry> render_pipelines;
    std::unordered_map<uint32_t, ComputePipelineEntry> compute_pipelines;
    std::unordered_map<uint32_t, wgpu::BindGroup> bind_groups;
    std::unordered_map<std::string, wgpu::Buffer> dummy_buffers;
    std::unordered_map<std::string, wgpu::Sampler> dummy_samplers;
    std::unordered_map<std::string, wgpu::TextureView> dummy_views;
    std::vector<wgpu::Texture> dummy_textures;
    std::set<uint32_t> depth_twin_sources;  // texture ids that own a depth twin
    std::unordered_map<uint32_t, wgpu::RenderPipeline> depth_copy_pipelines;  // DepthRead * 256 + source format
    wgpu::BindGroupLayout empty_layout;
    wgpu::ShaderModule empty_fragment;
    wgpu::ShaderModule error_vertex_module;
    wgpu::ShaderModule error_compute_module;
    std::shared_ptr<WgslPack> pack;  // shared with the shader workers
    // Shader workers (live mode, shader_workers.h) and the pipelines waiting for their shaders: the
    // packet (with the pipeline layout as it was at packet time and the shader entries) is replayed
    // once the shaders are ready, by pump_deferred() between packets or by a draw that needs it.
    ShaderWorkers shader_workers;
    bool use_shader_workers = false;
    struct DeferredPipe {
        uint16_t opcode = 0;
        std::vector<uint8_t> payload;
        std::optional<PlEntry> layout;
        ShaderRef a, b;  // vertex + fragment, or compute
    };
    std::unordered_map<uint32_t, DeferredPipe> deferred_render, deferred_compute;
    uint64_t deferred_seen_completed = 0;
    // Uncaptured device errors raised on a shader worker thread: drained into the stats by the queue thread.
    std::mutex worker_error_mu;
    std::vector<std::string> worker_errors;  // the first few messages
    uint64_t worker_error_count = 0;
    uint64_t worker_errors_logged = 0;
    // Reflection loader (web/workers/reflection_loader.js ReflectionRegistry) and the layouts a v1
    // pipeline was built with (`_pipelineBindGroupLayouts` / `_pipelineReflectionGroups` and the compute twins).
    reflect::ReflectionRegistry reflection;
    struct ReflectedLayout {
        wgpu::PipelineLayout pipeline_layout;
        std::vector<wgpu::BindGroupLayout> bgls;
        std::vector<std::vector<reflect::LayoutEntryDesc>> groups;
    };
    std::unordered_map<uint32_t, ReflectedLayout> reflected_render, reflected_compute;
    std::unordered_map<std::string, wgpu::BindGroupLayout> manifest_bgls;  // by layout id hex

    // ---- milestone 2 state ----
    GuestMemory guest_mem;
    bool negotiated_limits_applied = false;  // the device was created with the raised limits
    std::vector<Impl_ReadbackEntry> recorded_readbacks;  // recorded into the encoder being built
    bool readback_flush_wanted = false;
    uint64_t pending_maps = 0;  // read-back / timer map callbacks not delivered yet
    bool early_flushed = false;  // a read-back flush already submitted work of this frame
    std::unordered_map<uint32_t, uint64_t> compute_hash;  // GPU-time key of a compute pipeline
    std::unordered_set<uint32_t> compute_bind_groups;     // v1 bind groups made for a compute pipeline
    wgpu::Sampler default_filtering_sampler;  // getOrCreateDefaultSampler('filtering')
    // layout:'auto' bind group remap (_getOrCreateRemappedBindGroup)
    std::unordered_map<uint32_t, std::shared_ptr<BgDesc>> bg_descs;
    std::unordered_map<uint32_t, AutoLayoutMeta> compute_meta;
    std::unordered_set<uint32_t> comparison_samplers;
    std::unordered_map<std::string, RemapCacheEntry> remap_cache;
    uint64_t pipeline_gen_counter = 0;
    wgpu::Sampler fallback_samplers[3];  // getOrCreateDefaultSampler: filtering, non-filtering, comparison
    struct Group0Src {
        uint32_t id = 0;
        std::shared_ptr<BgDesc> desc;
        std::vector<uint32_t> dyn;
    };
    struct RemapState {  // _preparePassCommandsForFlush's per-flush state
        int mode = 0;    // 0 none, 1 render, 2 compute
        uint32_t render_pipe = 0;
        bool render_auto = false, group0_render = false, has_last_render = false;
        Group0Src last_render;
        uint32_t compute_pipe = 0;
        bool compute_valid = false, compute_auto = false, group0_compute = false, has_last_compute = false;
        Group0Src last_compute;
        bool had_explicit_compute = false;
    } remap;
    d3d12_guest::GpuTimeTable* time_table = nullptr;
    std::unique_ptr<d3d12_guest::GpuTimeRing> time_ring;
    std::vector<TimerSlot> timer_slots;

    void snapshot_blob_stats();

    // ---- strict validation (--gpu-validation) ----
    struct ValidationEntry {
        uint64_t count = 0;
        uint64_t first_packet = 0;
        uint16_t first_opcode = 0;
        std::string kind;  // scope-validation / scope-internal / uncaptured / log
        std::map<uint16_t, uint64_t> by_opcode;
    };
    std::mutex validation_mutex;
    std::map<std::string, ValidationEntry> validation_errors;
    uint64_t validation_total = 0;
    uint64_t validation_scoped_packets = 0;
    uint64_t validation_packet_index = 0;
    uint64_t validation_cur_packet = 0;
    uint16_t validation_cur_opcode = 0;

    void record_validation(const char* kind, const std::string& text, uint64_t packet, uint16_t opcode) {
        std::string key = text.size() > 700 ? text.substr(0, 700) : text;
        std::lock_guard<std::mutex> lock(validation_mutex);
        ++validation_total;
        ValidationEntry& e = validation_errors[key];
        if (e.count++ == 0) {
            e.first_packet = packet;
            e.first_opcode = opcode;
            e.kind = kind;
        }
        ++e.by_opcode[opcode];
    }

    // ---- diagnostic error scopes (--gpu-error-scopes, error_scope_model.h) ----
    ErrorScopeLog error_scope_log;
    std::atomic<uint64_t> error_scoped_total{0};  // scoped errors recorded (drives the auto mode's cut-off)

    std::string render_pipeline_ctx(uint32_t id) const {
        auto it = render_pipelines.find(id);
        if (it == render_pipelines.end()) return "rp" + std::to_string(id) + "(unknown)";
        return "rp" + std::to_string(id) + "(vs=" + hash_hex(it->second.vs_hash) + " fs=" + hash_hex(it->second.fs_hash) + ")";
    }
    std::string compute_pipeline_ctx(uint32_t id) const {
        auto it = compute_hash.find(id);
        return "cp" + std::to_string(id) + "(cs=" + (it == compute_hash.end() ? std::string("?") : hash_hex(it->second)) + ")";
    }
    // The pipelines and bind groups the frame's recorded passes use (encoded and submitted by the Present).
    std::string frame_scope_context() const {
        std::vector<std::string> items;
        std::set<std::string> seen;
        auto add = [&](std::string v) {
            if (items.size() < 24 && seen.insert(v).second) items.push_back(std::move(v));
        };
        for (const PassCmd& c : pass_cmds) {
            if (const auto* sp = std::get_if<SetPipelineCmd>(&c)) add(render_pipeline_ctx(sp->id));
            else if (const auto* cp = std::get_if<SetComputePipelineCmd>(&c)) add(compute_pipeline_ctx(cp->id));
            else if (const auto* b1 = std::get_if<SetBindGroup1Cmd>(&c)) add("bg" + std::to_string(b1->id));
            else if (const auto* b2 = std::get_if<SetBindGroup2Cmd>(&c)) add("bg" + std::to_string(b2->bind_group_id));
        }
        std::string out;
        for (const std::string& i : items) out += (out.empty() ? "" : " ") + i;
        return out;
    }
    // The resource ids a packet names, read after the packet ran (pipelines then hold their shader hashes).
    std::string packet_scope_context(const Packet& p) const {
        const uint32_t n = p.payload_bytes;
        switch (static_cast<GpuOpcode>(p.opcode)) {
            case GpuOpcode::CreateRenderPipeline2:
                if (n >= sizeof(GpuCreateRenderPipeline2Payload)) {
                    const auto h = read_payload<GpuCreateRenderPipeline2Payload>(p);
                    return render_pipeline_ctx(h.id) + " vs_module=" + std::to_string(h.vs_module_id) +
                           " fs_module=" + std::to_string(h.fs_module_id) + " layout=" + std::to_string(h.pipeline_layout_id);
                }
                break;
            case GpuOpcode::CreateComputePipeline2:
                if (n >= sizeof(GpuCreateComputePipeline2Payload)) {
                    const auto h = read_payload<GpuCreateComputePipeline2Payload>(p);
                    return compute_pipeline_ctx(h.id) + " cs_module=" + std::to_string(h.cs_module_id);
                }
                break;
            case GpuOpcode::CreateRenderPipeline:
                if (n >= sizeof(GpuCreateRenderPipelinePayload)) {
                    const auto h = read_payload<GpuCreateRenderPipelinePayload>(p);
                    return "rp" + std::to_string(h.id) + " vs_module=" + std::to_string(h.vs_shader_id) +
                           " fs_module=" + std::to_string(h.fs_shader_id);
                }
                break;
            case GpuOpcode::CreateComputePipeline:
                if (n >= sizeof(GpuCreateComputePipelinePayload)) {
                    const auto h = read_payload<GpuCreateComputePipelinePayload>(p);
                    return "cp" + std::to_string(h.id) + " cs_module=" + std::to_string(h.cs_shader_id);
                }
                break;
            case GpuOpcode::CreateBindGroup2:
                if (n >= sizeof(GpuCreateBindGroup2Payload)) {
                    const auto h = read_payload<GpuCreateBindGroup2Payload>(p);
                    std::string c = "bg" + std::to_string(h.id) + " layout=" + std::to_string(h.layout_id) + " resources=";
                    const uint64_t cnt = std::min<uint64_t>(h.entry_count, (n - 16) / sizeof(GpuBindGroupEntry2));
                    for (uint64_t i = 0; i < cnt && i < 16; ++i) {
                        GpuBindGroupEntry2 e;
                        std::memcpy(&e, p.payload + 16 + i * sizeof(GpuBindGroupEntry2), sizeof e);
                        c += (i ? "," : "") + std::to_string(e.resource_type) + ":" + std::to_string(e.resource_id);
                    }
                    return c;
                }
                break;
            case GpuOpcode::CreateBindGroup:
                if (n >= sizeof(GpuCreateBindGroupPayload)) {
                    const auto h = read_payload<GpuCreateBindGroupPayload>(p);
                    return "bg" + std::to_string(h.id) + " pipeline=" + std::to_string(h.pipeline_id);
                }
                break;
            case GpuOpcode::CreateShaderModule:
                if (n >= 4) return "shader_module=" + std::to_string(read_payload<uint32_t>(p));
                break;
            case GpuOpcode::SetPipeline:
                if (n >= 4) return render_pipeline_ctx(read_payload<uint32_t>(p));
                break;
            case GpuOpcode::SetComputePipeline:
                if (n >= 4) return compute_pipeline_ctx(read_payload<uint32_t>(p));
                break;
            default:
                break;
        }
        return "";
    }

    // ---- setup ----

    bool init(CreateStatus* status, std::string* error) {
        static const wgpu::InstanceFeatureName kFeatures[] = {wgpu::InstanceFeatureName::TimedWaitAny};
        wgpu::InstanceDescriptor idesc{};
        idesc.requiredFeatureCount = 1;
        idesc.requiredFeatures = kFeatures;
        // Strict mode: Dawn backend validation (the D3D12 debug layer where installed); its messages
        // arrive through the logging callback and are counted with the packet being executed.
        dawn::native::DawnInstanceDescriptor dawn_idesc;
        if (options.gpu_validation) {
            dawn_idesc.backendValidationLevel = dawn::native::BackendValidationLevel::Full;
            dawn_idesc.SetLoggingCallback(
                [](wgpu::LoggingType type, wgpu::StringView msg, Impl* self) {
                    if (type != wgpu::LoggingType::Error && type != wgpu::LoggingType::Warning) return;
                    self->record_validation("log", std::string(type == wgpu::LoggingType::Error ? "[error] " : "[warning] ") + to_string(msg),
                                            self->validation_cur_packet, self->validation_cur_opcode);
                },
                this);
            idesc.nextInChain = &dawn_idesc;
        }
        if (options.force_fallback_adapter) {
            native_instance = std::make_unique<dawn::native::Instance>(&idesc);
            wgpuInstanceAddRef(native_instance->Get());
            instance = wgpu::Instance::Acquire(native_instance->Get());
        } else {
            instance = wgpu::CreateInstance(&idesc);
        }
        if (!instance) {
            *status = CreateStatus::Failed;
            *error = "wgpuCreateInstance failed";
            return false;
        }
        swap_width = options.width ? options.width : 1;
        swap_height = options.height ? options.height : 1;
        stream_w = swap_width;
        stream_h = swap_height;
        if (!options.window && options.present_width && options.present_height) {
            swap_width = options.present_width;  // headless canvas of its own size: the stream is blitted into it
            swap_height = options.present_height;
        }
        if (!options.window_script.empty()) {
            std::vector<TimedWindowAction> actions;
            if (!parse_window_script(options.window_script, &actions, error)) {
                *status = CreateStatus::Failed;
                return false;
            }
            script.set(std::move(actions));
        }
        shot_dir = options.screenshot_dir.empty() ? default_screenshot_dir() : std::filesystem::path(options.screenshot_dir);
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

        return acquire_device(status, error);
    }

    // Adapter + device (+ offscreen canvas, timers): the first creation and every device-loss recovery.
    bool acquire_device(CreateStatus* status, std::string* error) {
        features = DeviceFeatures{};
        timer_slots.clear();
        time_ring.reset();
        wgpu::RequestAdapterOptions aopts{};
        aopts.compatibleSurface = surface;
        // Experimental features (timestamps inside passes) are only exposed by an adapter made with unsafe APIs.
        static const char* const kAdapterUnsafe[] = {"allow_unsafe_apis"};
        wgpu::DawnTogglesDescriptor adapter_toggles{};
        if (options.gpu_time_split) {
            adapter_toggles.enabledToggleCount = 1;
            adapter_toggles.enabledToggles = kAdapterUnsafe;
            aopts.nextInChain = &adapter_toggles;
        }
        struct AdapterResult {
            wgpu::Adapter adapter;
            std::string message;
        } ar;
        if (native_instance) {
            // --adapter warp: Dawn's forceFallbackAdapter finds no D3D12 WARP (it is listed as a CPU adapter by plain
            // enumeration), so pick the CPU adapter from the enumeration of this very instance.
            for (dawn::native::Adapter& cand : native_instance->EnumerateAdapters(&aopts)) {
                wgpu::AdapterInfo ci;
                wgpu::Adapter(cand.Get()).GetInfo(&ci);
                if (ci.adapterType != wgpu::AdapterType::CPU) continue;
                wgpuAdapterAddRef(cand.Get());
                ar.adapter = wgpu::Adapter::Acquire(cand.Get());
                break;
            }
            if (!ar.adapter) ar.message = "no software (CPU) adapter";
        } else instance.WaitAny(
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

        // The features tools/replay/replay.html requests (those the adapter has), default limits:
        // the replay page is the reference the executor is compared against.
        struct Want {
            wgpu::FeatureName name;
            bool DeviceFeatures::*flag;
        };
        const Want wants[] = {
            {wgpu::FeatureName::TextureCompressionBC, &DeviceFeatures::bc},
            {wgpu::FeatureName::Depth32FloatStencil8, &DeviceFeatures::depth32_stencil8},
            {wgpu::FeatureName::Float32Filterable, &DeviceFeatures::float32_filterable},
            {wgpu::FeatureName::BGRA8UnormStorage, &DeviceFeatures::bgra8_storage},
            {wgpu::FeatureName::RG11B10UfloatRenderable, &DeviceFeatures::rg11b10_renderable},
            {wgpu::FeatureName::TextureFormatsTier1, &DeviceFeatures::tier1},
        };
        std::vector<wgpu::FeatureName> required;
        for (const Want& w : wants) {
            if (w.name == wgpu::FeatureName::Float32Filterable && options.disable_float32_filterable) continue;
            if (adapter_handle.HasFeature(w.name)) {
                required.push_back(w.name);
                features.*(w.flag) = true;
            }
        }
        // Live mode: pack shader modules are compiled on shader workers (shader_workers.h), which needs
        // a device whose API calls may come from several threads.
        bool want_worker_sync = false;
        if (options.async_pipelines && !options.gpu_validation &&
            adapter_handle.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization)) {
            required.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
            want_worker_sync = true;
        }
        // GPU pass timestamps (gpu_timer.js requests 'timestamp-query' when the adapter has it).
        // Chrome quantizes timestamp queries to 100 us; native Dawn gets exact values (the toggle
        // is turned off), which is what per-pass GPU time needs.
        wgpu::DawnTogglesDescriptor toggles{};
        static const char* const kDisabledToggles[] = {"timestamp_quantization"};
        // Persistent blob cache: Dawn asks for / hands over its compiled shader and pipeline blobs
        // through these hooks; the store lives under <root>/<title>/<Dawn build>-<adapter>.
        // (The C++ wrapper has no callback converters for these raw-pointer hooks: the C struct is
        // filled directly; wgpu::ChainedStruct has the layout of WGPUChainedStruct.)
        WGPUDawnCacheDeviceDescriptor cache_desc = WGPU_DAWN_CACHE_DEVICE_DESCRIPTOR_INIT;
        bool have_cache = false;
        std::string iso_key;
        if (!options.blob_cache_root.empty()) {
#ifdef PG6_DAWN_COMMIT
            const std::string dawn_build = std::string(PG6_DAWN_COMMIT).substr(0, 12);
#else
            const std::string dawn_build = "dawn";
#endif
            const std::string adapter_key = adapter.vendor + "-" + adapter.device + "-" + adapter.backend + "-" +
                                            std::to_string(BlobCache::fnv1a(adapter.description.data(), adapter.description.size()) & 0xffffffffu) +
                                            "-" + std::to_string(info.vendorID) + "-" + std::to_string(info.deviceID);
            const std::string dir = BlobCache::versioned_dir(options.blob_cache_root, options.blob_cache_title, dawn_build, adapter_key);
            if (blob_cache_opened || blob_cache.open(dir, options.blob_cache_max_bytes)) {
                if (!blob_cache_opened) BlobCache::prune_stale_siblings((std::filesystem::path(dir).parent_path()).string(), dir, 30);
                blob_cache_opened = true;
                have_cache = true;
                iso_key = "pg6-" + BlobCache::sanitize(options.blob_cache_title);
                cache_desc.isolationKey = {iso_key.data(), iso_key.size()};  // iso outlives RequestDevice (below)
                cache_desc.dawnLoadCacheDataCallbackInfo.callback = [](size_t key_size, const uint8_t* key, size_t value_size,
                                                                       uint8_t* value, void* c, void*) -> size_t {
                    return static_cast<BlobCache*>(c)->load(key, key_size, value, value_size);
                };
                cache_desc.dawnLoadCacheDataCallbackInfo.userdata1 = &blob_cache;
                cache_desc.dawnStoreCacheDataCallbackInfo.callback = [](size_t key_size, const uint8_t* key, size_t value_size,
                                                                        const uint8_t* value, void* c, void*) {
                    static_cast<BlobCache*>(c)->store(key, key_size, value, value_size);
                };
                cache_desc.dawnStoreCacheDataCallbackInfo.userdata1 = &blob_cache;
            }
        }
        static const char* const kValidationToggles[] = {"enable_immediate_error_handling", "use_user_defined_labels_in_backend"};
        std::vector<const char*> enabled_toggles;
        if (options.gpu_validation) enabled_toggles.assign(std::begin(kValidationToggles), std::end(kValidationToggles));
        const bool want_timing = options.gpu_timing && adapter_handle.HasFeature(wgpu::FeatureName::TimestampQuery);
        if (want_timing) {
            required.push_back(wgpu::FeatureName::TimestampQuery);
            features.timestamp_query = true;
            // gpu_time_split: timestamps inside a pass (an experimental Dawn feature behind allow_unsafe_apis).
            if (options.gpu_time_split &&
                adapter_handle.HasFeature(wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses)) {
                required.push_back(wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses);
                features.timestamp_inside_passes = true;
                enabled_toggles.push_back("allow_unsafe_apis");
            }
            toggles.disabledToggleCount = 1;
            toggles.disabledToggles = kDisabledToggles;
        }
        if (!enabled_toggles.empty()) {
            toggles.enabledToggleCount = enabled_toggles.size();
            toggles.enabledToggles = enabled_toggles.data();
        }
        // Limits (requestNegotiatedDevice): each optional limit above the WebGPU default, capped.
        wgpu::Limits adapter_limits{};
        wgpu::Limits required_limits{};
        bool have_limits = false;
        if (adapter_handle.GetLimits(&adapter_limits) == wgpu::Status::Success) {
            const uint64_t supported[kOptionalLimitCount] = {
                adapter_limits.maxSampledTexturesPerShaderStage, adapter_limits.maxSamplersPerShaderStage,
                adapter_limits.maxStorageBuffersPerShaderStage,  adapter_limits.maxUniformBuffersPerShaderStage,
                adapter_limits.maxStorageTexturesPerShaderStage, adapter_limits.maxBindingsPerBindGroup,
                adapter_limits.maxBufferSize,                    adapter_limits.maxStorageBufferBindingSize,
                adapter_limits.maxComputeWorkgroupStorageSize};
            uint64_t want[kOptionalLimitCount];
            for (size_t i = 0; i < kOptionalLimitCount; ++i) {
                const auto v = negotiated_limit(kOptionalLimitSpecs[i], supported[i]);
                want[i] = v ? *v : 0;
                have_limits = have_limits || v.has_value();
            }
            required_limits.maxSampledTexturesPerShaderStage = want[0] ? static_cast<uint32_t>(want[0]) : WGPU_LIMIT_U32_UNDEFINED;
            required_limits.maxSamplersPerShaderStage = want[1] ? static_cast<uint32_t>(want[1]) : WGPU_LIMIT_U32_UNDEFINED;
            required_limits.maxStorageBuffersPerShaderStage = want[2] ? static_cast<uint32_t>(want[2]) : WGPU_LIMIT_U32_UNDEFINED;
            required_limits.maxUniformBuffersPerShaderStage = want[3] ? static_cast<uint32_t>(want[3]) : WGPU_LIMIT_U32_UNDEFINED;
            required_limits.maxStorageTexturesPerShaderStage = want[4] ? static_cast<uint32_t>(want[4]) : WGPU_LIMIT_U32_UNDEFINED;
            required_limits.maxBindingsPerBindGroup = want[5] ? static_cast<uint32_t>(want[5]) : WGPU_LIMIT_U32_UNDEFINED;
            required_limits.maxBufferSize = want[6] ? want[6] : WGPU_LIMIT_U64_UNDEFINED;
            required_limits.maxStorageBufferBindingSize = want[7] ? want[7] : WGPU_LIMIT_U64_UNDEFINED;
            required_limits.maxComputeWorkgroupStorageSize = want[8] ? static_cast<uint32_t>(want[8]) : WGPU_LIMIT_U32_UNDEFINED;
            // Wide MRT sets (e.g. 5+ RGBA16F targets = 40 B/sample) exceed the 32 B WebGPU default; the
            // adapter's own cap is a faithful superset, so ask for it (the D3D12 backend allows 64).
            if (adapter_limits.maxColorAttachmentBytesPerSample > 32) {
                required_limits.maxColorAttachmentBytesPerSample = adapter_limits.maxColorAttachmentBytesPerSample;
                have_limits = true;
            }
        }
        // Features + limits, then features only, then a bare device (the worker's order; the
        // timestamp feature rides along with the features and is dropped with them).
        struct DeviceResult {
            wgpu::Device device;
            std::string message;
        } dr;
        enum Attempt { kFeaturesAndLimits, kFeaturesOnly, kBare };
        for (const Attempt attempt : {kFeaturesAndLimits, kFeaturesOnly, kBare}) {
            if (attempt == kFeaturesAndLimits && !have_limits) continue;
            wgpu::DeviceDescriptor ddesc{};
            const bool with_features = attempt != kBare;
            if ((with_features && want_timing) || options.gpu_validation) ddesc.nextInChain = &toggles;
            if (have_cache) {
                cache_desc.chain.next = const_cast<WGPUChainedStruct*>(reinterpret_cast<const WGPUChainedStruct*>(ddesc.nextInChain));
                ddesc.nextInChain = reinterpret_cast<const wgpu::ChainedStruct*>(&cache_desc.chain);
            }
            if (with_features) {
                ddesc.requiredFeatureCount = required.size();
                ddesc.requiredFeatures = required.data();
            }
            if (attempt == kFeaturesAndLimits) ddesc.requiredLimits = &required_limits;
            ddesc.SetUncapturedErrorCallback(
                [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView msg, Impl* self) {
                    if (tl_shader_worker) {  // drained by the executor thread (drain_worker_errors)
                        std::lock_guard<std::mutex> lock(self->worker_error_mu);
                        ++self->worker_error_count;
                        if (self->worker_errors.size() < 64) self->worker_errors.push_back(to_string(msg));
                        return;
                    }
                    ++self->stats.device_errors;
                    self->stats.last_device_error = to_string(msg);
                    if (self->stats.device_error_log.size() < 20) self->stats.device_error_log.push_back(to_string(msg));
                    if (self->options.gpu_validation)
                        self->record_validation("uncaptured", to_string(msg), self->validation_cur_packet, self->validation_cur_opcode);
                },
                this);
            ddesc.SetDeviceLostCallback(
                wgpu::CallbackMode::AllowSpontaneous,
                [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView msg, Impl* self) {
                    if (reason == wgpu::DeviceLostReason::Destroyed) return;  // our own teardown
                    ++self->stats.device_lost;
                    ++self->stats.device_lost_events;
                    self->stats.last_device_error = to_string(msg);
                    self->lost = true;
                },
                this);
            dr = DeviceResult{};
            instance.WaitAny(
                adapter_handle.RequestDevice(
                    &ddesc, wgpu::CallbackMode::WaitAnyOnly,
                    [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView msg, DeviceResult* out) {
                        if (s == wgpu::RequestDeviceStatus::Success) out->device = std::move(d);
                        else out->message = to_string(msg);
                    },
                    &dr),
                kWaitForever);
            if (dr.device) {
                negotiated_limits_applied = attempt == kFeaturesAndLimits;
                stats.limits_raised = negotiated_limits_applied;
                if (attempt == kBare) {  // the features were dropped with the request
                    features = DeviceFeatures{};
                    required.clear();
                    want_worker_sync = false;
                }
                break;
            }
        }
        if (!dr.device) {
            *status = CreateStatus::Failed;
            *error = "RequestDevice failed: " + dr.message;
            return false;
        }
        device = dr.device;
        queue = device.GetQueue();
        if (want_worker_sync && !shader_workers.running()) {
            const unsigned n = shader_worker_count(std::thread::hardware_concurrency(), std::getenv("PG6_SHADER_WORKERS"));
            if (n) shader_workers.start(n);
        }
        // A rebuilt device without the feature keeps the (idle) threads but compiles on this thread again.
        use_shader_workers = want_worker_sync && shader_workers.running();
        stats.shader_workers = use_shader_workers ? shader_workers.size() : 0;
        wgpu::Limits limits{};
        if (device.GetLimits(&limits) == wgpu::Status::Success) {
            if (limits.maxComputeWorkgroupsPerDimension >= 1) max_workgroups_per_dim = limits.maxComputeWorkgroupsPerDimension;
            stage_limits.uniform_buffers = limits.maxUniformBuffersPerShaderStage;
            stage_limits.storage_buffers = limits.maxStorageBuffersPerShaderStage;
            stage_limits.sampled_textures = limits.maxSampledTexturesPerShaderStage;
            stage_limits.samplers = limits.maxSamplersPerShaderStage;
            stage_limits.storage_textures = limits.maxStorageTexturesPerShaderStage;
            stats.limit_uniform_buffers_per_stage = stage_limits.uniform_buffers;
            stats.limit_storage_buffers_per_stage = stage_limits.storage_buffers;
            stats.limit_sampled_textures_per_stage = stage_limits.sampled_textures;
            stats.limit_samplers_per_stage = stage_limits.samplers;
            stats.limit_storage_textures_per_stage = stage_limits.storage_textures;
            // The adapter's own per-stage maxima (requested above where they exceed the defaults): a
            // layout over these cannot be raised into range, only fitted to the WGSL.
            std::fprintf(stderr,
                         "[native_exec] per-stage limits (adapter / device): uniform buffers %u / %u, storage buffers "
                         "%u / %u, sampled textures %u / %u, samplers %u / %u, storage textures %u / %u\n",
                         adapter_limits.maxUniformBuffersPerShaderStage, limits.maxUniformBuffersPerShaderStage,
                         adapter_limits.maxStorageBuffersPerShaderStage, limits.maxStorageBuffersPerShaderStage,
                         adapter_limits.maxSampledTexturesPerShaderStage, limits.maxSampledTexturesPerShaderStage,
                         adapter_limits.maxSamplersPerShaderStage, limits.maxSamplersPerShaderStage,
                         adapter_limits.maxStorageTexturesPerShaderStage, limits.maxStorageTexturesPerShaderStage);
        }
        if (!surface) create_offscreen();
        if (options.gpu_timing) setup_timers(adapter_handle.HasFeature(wgpu::FeatureName::TimestampQuery));
        *status = CreateStatus::Ok;
        return true;
    }

    void setup_timers(bool adapter_has_feature) {
        time_table = options.gpu_time ? options.gpu_time : &d3d12_guest::gpu_time_table();
        stats.timer_supported = features.timestamp_query;
        stats.timer_split_supported = features.timestamp_inside_passes;
        if (options.gpu_time_split && !features.timestamp_inside_passes)
            std::fprintf(stderr, "[native] gpu_time_split: the adapter lacks timestamp-query-inside-passes; passes are timed whole\n");
        if (!adapter_has_feature) {
            time_table->set_support(d3d12_guest::GpuTimeTable::Support::Unavailable, "adapter lacks timestamp-query");
            return;
        }
        time_ring = std::make_unique<d3d12_guest::GpuTimeRing>(kTimerSlots, kTimerMaxPasses);
        for (size_t i = 0; i < kTimerSlots; ++i) {
            TimerSlot slot;
            wgpu::QuerySetDescriptor qd{};
            qd.type = wgpu::QueryType::Timestamp;
            qd.count = static_cast<uint32_t>(kTimerMaxPasses * 2);
            slot.queries = device.CreateQuerySet(&qd);
            wgpu::BufferDescriptor rb{};
            rb.size = kTimerMaxPasses * 2 * 8;
            rb.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
            slot.resolve = device.CreateBuffer(&rb);
            wgpu::BufferDescriptor sb{};
            sb.size = rb.size;
            sb.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
            slot.staging = device.CreateBuffer(&sb);
            timer_slots.push_back(std::move(slot));
        }
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

    // Fifo makes GetCurrentTexture block for the vertical blank (and far longer behind a busy compositor):
    // the guest's frame pacing is its own, so auto prefers Mailbox (no tearing), then Immediate; Fifo is the
    // always-supported fallback. --present-mode forces one (validated against the surface, falls back with a
    // log line). Rendering is unchanged, only when a frame reaches the screen.
    wgpu::PresentMode pick_present_mode() {
        if (!present_mode_known) {
            present_mode_known = true;
            uint8_t mask = 0;
            wgpu::SurfaceCapabilities caps{};
            if (surface.GetCapabilities(adapter_handle, &caps) == wgpu::Status::Success) {
                for (size_t i = 0; i < caps.presentModeCount; ++i) {
                    if (caps.presentModes[i] == wgpu::PresentMode::Fifo) mask |= kModeFifo;
                    if (caps.presentModes[i] == wgpu::PresentMode::Mailbox) mask |= kModeMailbox;
                    if (caps.presentModes[i] == wgpu::PresentMode::Immediate) mask |= kModeImmediate;
                }
            }
            const PresentSelection sel = select_present_mode(options.present_mode, mask);
            if (!sel.note.empty()) std::fprintf(stderr, "[native_exec] %s\n", sel.note.c_str());
            present_mode_name = present_choice_name(sel.actual);
            present_mode = sel.actual == PresentChoice::Mailbox ? wgpu::PresentMode::Mailbox
                         : sel.actual == PresentChoice::Immediate ? wgpu::PresentMode::Immediate
                                                                  : wgpu::PresentMode::Fifo;
            std::fprintf(stderr, "[native_exec] present mode: %s (requested %s)\n", present_mode_name.c_str(),
                         present_choice_name(options.present_mode));
        }
        return present_mode;
    }
    std::string present_mode_name;
    uint64_t cap_anchor_ns = 0;
    // --max-fps: sleep (then yield-spin the last 0.5 ms) until the next present is due.
    void apply_frame_cap() {
        if (options.max_fps == 0) return;
        const uint64_t now = now_ns();
        const uint64_t wait = frame_cap_wait_ns(options.max_fps, cap_anchor_ns, now);
        if (wait > 600000ull) std::this_thread::sleep_for(std::chrono::nanoseconds(wait - 500000ull));
        while (now_ns() < now + wait) std::this_thread::yield();
        cap_anchor_ns = frame_cap_next_anchor(options.max_fps, cap_anchor_ns, now_ns());
    }
    bool present_mode_known = false;
    wgpu::PresentMode present_mode = wgpu::PresentMode::Fifo;

    void configure_surface(uint32_t w, uint32_t h) {
        wgpu::SurfaceConfiguration cfg{};
        cfg.device = device;
        cfg.format = swap_format;
        cfg.usage = wgpu::TextureUsage::RenderAttachment;
        cfg.usage |= wgpu::TextureUsage::CopySrc;  // F12 / --screenshot-at read the stream image back (also the digest / probe)
        cfg.width = w;
        cfg.height = h;
        cfg.alphaMode = wgpu::CompositeAlphaMode::Opaque;  // the worker's DEFAULT_ALPHA_MODE
        cfg.presentMode = pick_present_mode();
        surface.Configure(&cfg);
        swap_width = w;
        swap_height = h;
        if (!options.fixed_stream_size) {
            stream_w = w;
            stream_h = h;
        }
        surface_configured = true;
        surface_state.on_configured(w, h);
        ++stats.surface_configures;
    }

    // Normal play presents straight to the swapchain (no extra full-frame copy). Only while the F3 overlay or the
    // stop banner is active does the frame go through the intermediate texture, so the last frame survives the
    // present and can be redrawn with the overlay / banner after the guest stopped (overlay_needs_intermediate).
    bool blit_active() const {
        return stream_format != swap_format || stream_w != swap_width || stream_h != swap_height ||
               overlay_needs_intermediate(window != nullptr, overlay_on, !ov_banner.empty());
    }

    // ---- stats overlay (overlay_model.h): drawn after every copy of the guest's frame, never read back ----
    bool overlay_on = false;
    bool overlay_dirty = false;
    bool have_last_frame = false;  // the intermediate texture holds the last presented stream image
    OverlayHistory ov_hist;
    uint64_t ov_last_present_ns = 0, ov_last_draws = 0, ov_redraw_ns = 0, ov_gpu_sample_ns = 0, ov_snapshot_ns = 0;
    uint64_t ov_gpu_frames = 0, ov_gpu_total_ns = 0;
    double ov_gpu_ms = -1;
    int64_t ov_underruns = -1;
    std::string ov_fault;
    std::string ov_banner;  // the stop message, drawn even with the stats overlay off
    wgpu::Texture ov_tex;
    wgpu::TextureView ov_view;
    uint32_t ov_tex_w = 0, ov_tex_h = 0;
    wgpu::Sampler ov_sampler;
    wgpu::RenderPipeline ov_pipeline;
    wgpu::TextureFormat ov_pipeline_format = wgpu::TextureFormat::Undefined;
    wgpu::BindGroup ov_bg;

    OverlayStatus overlay_status() {
        OverlayStatus s;
        const uint64_t now = now_ns();
        const size_t n = 30;
        s.frame_ms = ov_hist.avg_ms(n);
        const bool recent = ov_last_present_ns && now - ov_last_present_ns < 500000000ull && s.frame_ms > 0;
        s.fps = recent ? 1000.0 / s.frame_ms : 0.0;
        s.draws_per_frame = ov_hist.avg_draws(n);
        if (time_table && now - ov_gpu_sample_ns >= 250000000ull) {
            ov_gpu_sample_ns = now;
            uint64_t fr = 0, ns = 0;
            time_table->measured_totals(&fr, &ns);
            if (fr > ov_gpu_frames) ov_gpu_ms = static_cast<double>(ns - ov_gpu_total_ns) / static_cast<double>(fr - ov_gpu_frames) / 1e6;
            ov_gpu_frames = fr;
            ov_gpu_total_ns = ns;
        }
        s.gpu_ms = ov_gpu_ms;
        s.audio_underruns = ov_underruns;
        s.present_mode = present_mode_name;
        if (gpu::present_pacer().honor_interval()) s.present_mode += " HONOR";
        s.fault = ov_fault;
        s.banner = ov_banner;
        s.show_stats = overlay_on;
        return s;
    }

    // Draws the box on top of the canvas (load, alpha blend). Nothing here touches the guest's frame.
    void encode_overlay(wgpu::CommandEncoder& enc, const wgpu::TextureView& canvas_view) {
        const OverlayBitmap bm = render_overlay(overlay_status(), ov_hist);
        if (!ov_tex || ov_tex_w != bm.w || ov_tex_h != bm.h) {
            wgpu::TextureDescriptor td{};
            td.label = {"playgta6-overlay", WGPU_STRLEN};
            td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
            td.dimension = wgpu::TextureDimension::e2D;
            td.size = {bm.w, bm.h, 1};
            td.format = wgpu::TextureFormat::RGBA8Unorm;
            ov_tex = device.CreateTexture(&td);
            ov_view = ov_tex.CreateView();
            ov_tex_w = bm.w;
            ov_tex_h = bm.h;
            ov_bg = nullptr;
        }
        wgpu::TexelCopyTextureInfo dst{};
        dst.texture = ov_tex;
        wgpu::TexelCopyBufferLayout layout{};
        layout.bytesPerRow = bm.w * 4;
        layout.rowsPerImage = bm.h;
        const wgpu::Extent3D size{bm.w, bm.h, 1};
        queue.WriteTexture(&dst, bm.rgba.data(), bm.rgba.size(), &layout, &size);
        if (!blit_module) {
            wgpu::ShaderSourceWGSL wgsl{};
            wgsl.code = {kBlitWgsl, WGPU_STRLEN};
            wgpu::ShaderModuleDescriptor md{};
            md.nextInChain = &wgsl;
            md.label = {"playgta6-swapchain-blit-wgsl", WGPU_STRLEN};
            blit_module = device.CreateShaderModule(&md);
        }
        if (!ov_sampler) {
            wgpu::SamplerDescriptor sd{};
            sd.magFilter = wgpu::FilterMode::Nearest;
            sd.minFilter = wgpu::FilterMode::Nearest;
            ov_sampler = device.CreateSampler(&sd);
        }
        if (!ov_pipeline || ov_pipeline_format != swap_format) {
            wgpu::BlendState blend{};  // premultiplied source: One / OneMinusSrcAlpha
            blend.color = {wgpu::BlendOperation::Add, wgpu::BlendFactor::One, wgpu::BlendFactor::OneMinusSrcAlpha};
            blend.alpha = {wgpu::BlendOperation::Add, wgpu::BlendFactor::One, wgpu::BlendFactor::OneMinusSrcAlpha};
            wgpu::ColorTargetState ct{};
            ct.format = swap_format;
            ct.blend = &blend;
            wgpu::FragmentState fs{};
            fs.module = blit_module;
            fs.entryPoint = {"fs_main", 7};
            fs.targetCount = 1;
            fs.targets = &ct;
            wgpu::RenderPipelineDescriptor rd{};
            rd.label = {"playgta6-overlay-pipeline", WGPU_STRLEN};
            rd.vertex.module = blit_module;
            rd.vertex.entryPoint = {"vs_main", 7};
            rd.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
            rd.primitive.cullMode = wgpu::CullMode::None;
            rd.fragment = &fs;
            ov_pipeline = device.CreateRenderPipeline(&rd);
            ov_pipeline_format = swap_format;
            ov_bg = nullptr;
        }
        if (!ov_bg) {
            wgpu::BindGroupEntry e[2]{};
            e[0].binding = 0;
            e[0].textureView = ov_view;
            e[1].binding = 1;
            e[1].sampler = ov_sampler;
            wgpu::BindGroupDescriptor bd{};
            bd.label = {"playgta6-overlay-bg", WGPU_STRLEN};
            bd.layout = ov_pipeline.GetBindGroupLayout(0);
            bd.entryCount = 2;
            bd.entries = e;
            ov_bg = device.CreateBindGroup(&bd);
        }
        const uint32_t scale = overlay_scale(swap_height);
        const uint32_t vw = std::min(bm.w * scale, swap_width), vh = std::min(bm.h * scale, swap_height);
        if (!vw || !vh) return;
        wgpu::RenderPassColorAttachment a{};
        a.view = canvas_view;
        a.loadOp = wgpu::LoadOp::Load;
        a.storeOp = wgpu::StoreOp::Store;
        wgpu::RenderPassDescriptor rp{};
        rp.label = {"playgta6-overlay-pass", WGPU_STRLEN};
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &a;
        wgpu::RenderPassEncoder pass = enc.BeginRenderPass(&rp);
        pass.SetViewport(0.0f, 0.0f, static_cast<float>(bm.w * scale), static_cast<float>(bm.h * scale), 0.0f, 1.0f);
        pass.SetScissorRect(0, 0, vw, vh);
        pass.SetPipeline(ov_pipeline);
        pass.SetBindGroup(0, ov_bg);
        pass.Draw(3);
        pass.End();
        ++stats.overlay_frames;
    }

    // poll(): with the guest not presenting (stopped on a fault, loading) the last frame is redrawn from the
    // intermediate texture when the overlay was toggled, the fault text changed, or every 250 ms for a resize.
    void overlay_idle_redraw() {
        if (!window || !surface || lost || fatal || show_spec || frame_target) return;
        // No frozen frame (the guest stopped after a direct present, banner switched on late): the banner is drawn
        // on a dark canvas instead.
        const bool frozen = have_last_frame && intermediate;
        if (!frozen && ov_banner.empty()) return;
        const uint64_t now = now_ns();
        const bool due = overlay_dirty || ((overlay_on || !ov_banner.empty()) && now - ov_last_present_ns > 250000000ull &&
                                           now - ov_redraw_ns > 250000000ull);
        if (!due) return;
        overlay_dirty = false;
        ov_redraw_ns = now;
        FrameTarget t;
        if (!acquire_canvas(&t)) return;
        wgpu::CommandEncoder enc = device.CreateCommandEncoder();
        if (frozen) {
            encode_blit(enc, t.canvas_view);
        } else {
            wgpu::RenderPassColorAttachment a{};
            a.view = t.canvas_view;
            a.loadOp = wgpu::LoadOp::Clear;
            a.storeOp = wgpu::StoreOp::Store;
            a.clearValue = {0.04, 0.04, 0.05, 1.0};
            wgpu::RenderPassDescriptor rp{};
            rp.colorAttachmentCount = 1;
            rp.colorAttachments = &a;
            enc.BeginRenderPass(&rp).End();
        }
        if (overlay_on || !ov_banner.empty()) encode_overlay(enc, t.canvas_view);
        wgpu::CommandBuffer cb = enc.Finish();
        queue.Submit(1, &cb);
        surface.Present();
    }

    void ensure_intermediate() {
        if (intermediate && intermediate_format == stream_format && intermediate_w == stream_w &&
            intermediate_h == stream_h)
            return;
        if (intermediate) intermediate.Destroy();  // before the new allocation, like the worker (no VRAM leak)
        have_last_frame = false;
        blit_bind_group = nullptr;
        wgpu::TextureDescriptor td{};
        td.label = {"playgta6-swapchain-intermediate", WGPU_STRLEN};
        td.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                   wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
        td.dimension = wgpu::TextureDimension::e2D;
        td.size = {stream_w, stream_h, 1};
        td.format = stream_format;
        intermediate = device.CreateTexture(&td);
        intermediate_view = intermediate.CreateView();
        intermediate_format = stream_format;
        intermediate_w = stream_w;
        intermediate_h = stream_h;
    }

    // WgpuSwapchain._ensureBlitPipeline / _ensureBlitBindGroup / blitToCanvasIfNeeded: a fullscreen
    // triangle sampling the intermediate texture into the canvas, inside the letterbox viewport.
    void encode_blit(wgpu::CommandEncoder& enc, const wgpu::TextureView& canvas_view) {
        if (!blit_module) {
            wgpu::ShaderSourceWGSL wgsl{};
            wgsl.code = {kBlitWgsl, WGPU_STRLEN};
            wgpu::ShaderModuleDescriptor md{};
            md.nextInChain = &wgsl;
            md.label = {"playgta6-swapchain-blit-wgsl", WGPU_STRLEN};
            blit_module = device.CreateShaderModule(&md);
            wgpu::SamplerDescriptor sd{};
            sd.magFilter = wgpu::FilterMode::Linear;
            sd.minFilter = wgpu::FilterMode::Linear;
            sd.addressModeU = wgpu::AddressMode::ClampToEdge;
            sd.addressModeV = wgpu::AddressMode::ClampToEdge;
            blit_sampler = device.CreateSampler(&sd);
        }
        if (!blit_pipeline || blit_pipeline_format != swap_format) {
            wgpu::ColorTargetState ct{};
            ct.format = swap_format;
            wgpu::FragmentState fs{};
            fs.module = blit_module;
            fs.entryPoint = {"fs_main", 7};
            fs.targetCount = 1;
            fs.targets = &ct;
            wgpu::RenderPipelineDescriptor rd{};
            rd.label = {"playgta6-swapchain-blit-pipeline", WGPU_STRLEN};
            rd.vertex.module = blit_module;
            rd.vertex.entryPoint = {"vs_main", 7};
            rd.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
            rd.primitive.cullMode = wgpu::CullMode::None;
            rd.fragment = &fs;
            blit_pipeline = device.CreateRenderPipeline(&rd);
            blit_pipeline_format = swap_format;
            blit_bind_group = nullptr;
        }
        if (!blit_bind_group) {
            wgpu::BindGroupEntry e[2]{};
            e[0].binding = 0;
            e[0].textureView = intermediate_view;
            e[1].binding = 1;
            e[1].sampler = blit_sampler;
            wgpu::BindGroupDescriptor bd{};
            bd.label = {"playgta6-swapchain-blit-bg", WGPU_STRLEN};
            bd.layout = blit_pipeline.GetBindGroupLayout(0);
            bd.entryCount = 2;
            bd.entries = e;
            blit_bind_group = device.CreateBindGroup(&bd);
        }
        wgpu::RenderPassColorAttachment a{};
        a.view = canvas_view;
        a.loadOp = wgpu::LoadOp::Clear;
        a.storeOp = wgpu::StoreOp::Store;
        a.clearValue = {0, 0, 0, 1};
        wgpu::RenderPassDescriptor rp{};
        rp.label = {"playgta6-swapchain-blit-pass", WGPU_STRLEN};
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &a;
        wgpu::RenderPassEncoder pass = enc.BeginRenderPass(&rp);
        // With a window the rect is the window's own (DPI-aware) present rect, headless the plain letterbox.
        const PresentRect r = window ? window->present_rect(stream_w, stream_h)
                                     : letterbox_rect(stream_w, stream_h, swap_width, swap_height);
        if (r.w && r.h) {
            pass.SetViewport(static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.w),
                             static_cast<float>(r.h), 0.0f, 1.0f);
            pass.SetPipeline(blit_pipeline);
            pass.SetBindGroup(0, blit_bind_group);
            pass.Draw(3);
        }
        pass.End();
        ++stats.blit_frames;
    }

    // ---- --show-target: present any render target / texture instead of the swapchain image ----
    // Read-only: the frame is encoded and the digest copy taken exactly as without it; afterwards one more
    // pass draws the chosen texture over the canvas (letterboxed, black bars). The guest's frame is unchanged.
    std::optional<ShowTargetSpec> show_spec;        // --show-target / set_show_target
    std::vector<PassTarget> frame_passes;           // the render passes encoded for the presented frame
    uint32_t frame_pass_count = 0;
    std::vector<TargetRef> show_candidates;         // the hotkey cycle list (render targets of the last frame)
    bool show_manual = false;                       // a hotkey took over from the selector
    std::optional<TargetRef> show_manual_target;    // nullopt = off (swapchain)
    int64_t show_manual_layer = -1, show_manual_mip = -1;
    std::optional<TargetRef> show_last;             // what the last present showed
    ShowTexture show_last_tex;
    ShowSubresource show_last_sub;
    std::string show_label;                         // title overlay text of the last present ("" = nothing shown)
    wgpu::RenderPipeline show_pipe[2];
    wgpu::BindGroupLayout show_bgl[2];
    wgpu::Buffer show_ub;
    wgpu::TextureFormat show_pipe_format = wgpu::TextureFormat::Undefined;

    bool ensure_show_pipeline(bool ms) {
        if (show_pipe_format != swap_format) {
            show_pipe[0] = show_pipe[1] = nullptr;
            show_bgl[0] = show_bgl[1] = nullptr;
            show_pipe_format = swap_format;
        }
        if (show_pipe[ms]) return true;
        const std::string code = show_target_wgsl(ms);
        wgpu::ShaderSourceWGSL wgsl{};
        wgsl.code = {code.data(), code.size()};
        wgpu::ShaderModuleDescriptor md{};
        md.nextInChain = &wgsl;
        md.label = {"playgta6-show-target-wgsl", WGPU_STRLEN};
        wgpu::ShaderModule mod = device.CreateShaderModule(&md);
        wgpu::BindGroupLayoutEntry le[2]{};
        le[0].binding = 0;
        le[0].visibility = wgpu::ShaderStage::Fragment;
        le[0].texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
        le[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
        le[0].texture.multisampled = ms;
        le[1].binding = 1;
        le[1].visibility = wgpu::ShaderStage::Fragment;
        le[1].buffer.type = wgpu::BufferBindingType::Uniform;
        le[1].buffer.minBindingSize = 48;
        wgpu::BindGroupLayoutDescriptor bgd{};
        bgd.entryCount = 2;
        bgd.entries = le;
        show_bgl[ms] = device.CreateBindGroupLayout(&bgd);
        wgpu::PipelineLayoutDescriptor pld{};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &show_bgl[ms];
        wgpu::ColorTargetState ct{};
        ct.format = swap_format;
        wgpu::FragmentState fs{};
        fs.module = mod;
        fs.entryPoint = "fs";
        fs.targetCount = 1;
        fs.targets = &ct;
        wgpu::RenderPipelineDescriptor rd{};
        rd.label = {"playgta6-show-target-pipeline", WGPU_STRLEN};
        rd.layout = device.CreatePipelineLayout(&pld);
        rd.vertex.module = mod;
        rd.vertex.entryPoint = "vs";
        rd.fragment = &fs;
        show_pipe[ms] = device.CreateRenderPipeline(&rd);
        if (!show_ub) {
            wgpu::BufferDescriptor bd{};
            bd.size = 48;
            bd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
            show_ub = device.CreateBuffer(&bd);
        }
        return static_cast<bool>(show_pipe[ms]);
    }

    // The title bar: the host's title + what --show-target shows + the screenshot notice.
    std::string window_title() const {
        std::string t = show_target_title(last_title.empty() ? options.title : last_title, show_label);
        return shot_title_until_ns ? t + " | screenshot saved" : t;
    }

    void set_show_label(const std::string& label) {
        if (label == show_label) return;
        show_label = label;
        if (window) window->set_title(window_title());
    }

    // True when the chosen target was drawn into the canvas.
    bool encode_show_target(wgpu::CommandEncoder& enc, const wgpu::TextureView& canvas_view) {
        show_candidates = playgta6::gpu::native_exec::show_target_candidates(frame_passes);
        ShowTargetSpec spec = *show_spec;
        TargetResolution res;
        if (show_manual) {
            spec.layer = show_manual_layer;
            spec.mip = show_manual_mip;
            if (show_manual_target) res.target = show_manual_target;
            else res.reason = "off";
        } else {
            res = resolve_show_target(spec, frame_passes);
        }
        show_last.reset();
        if (!res.target) {
            const bool quiet = res.reason == "off" || spec.kind == ShowTargetSpec::Kind::Cycle;
            if (!quiet) ++stats.show_target_unavailable;
            set_show_label(quiet ? std::string() : "unavailable: " + res.reason);
            return false;
        }
        const TargetRef ref = *res.target;
        auto fail = [&](const std::string& why) {
            ++stats.show_target_unavailable;
            set_show_label("tex " + std::to_string(ref.id) + (ref.depth ? " depth" : "") + ": " + why);
            return false;
        };
        auto tit = textures.find(ref.id);
        if (tit == textures.end()) return fail("no such texture (not created or destroyed)");
        TextureEntry& t = tit->second;
        ShowTexture st;
        st.format = t.geom.format;
        st.dimension = t.geom.dimension;
        st.usage = t.usage;
        st.sample_count = t.sample_count;
        st.width = t.geom.width;
        st.height = t.geom.height;
        st.layers = t.geom.depth_or_layers;
        st.mips = t.geom.mips;
        if (t.bc3d_fallback) st.dimension = 2;
        const std::string why = show_target_unsupported(st, ref.depth);
        if (!why.empty()) return fail("cannot be shown: " + why);
        const ShowSubresource sub = show_target_subresource(spec, ref, st);
        wgpu::TextureViewDescriptor vd{};
        vd.dimension = wgpu::TextureViewDimension::e2D;
        vd.baseMipLevel = sub.mip;
        vd.mipLevelCount = 1;
        vd.baseArrayLayer = sub.layer;
        vd.arrayLayerCount = 1;
        wgpu::TextureView view;
        if (ref.depth) {
            if (!depth_twin(ref.id, t, DepthRead::TwinF32)) return fail("depth twin could not be created");
            refresh_depth_twins(enc);
            vd.format = wgpu::TextureFormat::R32Float;
            view = t.twins[static_cast<size_t>(DepthRead::TwinF32)].tex.CreateView(&vd);
        } else {
            view = t.texture.CreateView(&vd);
        }
        const bool ms = t.sample_count > 1 && !ref.depth;
        if (!view || !ensure_show_pipeline(ms)) return fail("display pipeline unavailable");
        const uint32_t mw = show_target_mip_extent(st.width, sub.mip), mh = show_target_mip_extent(st.height, sub.mip);
        const PresentRect r = window ? window->present_rect(mw, mh) : letterbox_rect(mw, mh, swap_width, swap_height);
        if (!r.w || !r.h) return false;  // minimized
        const float u[8] = {static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.w),
                            static_cast<float>(r.h), static_cast<float>(mw), static_cast<float>(mh), spec.lo, spec.hi};
        uint32_t ubits[12] = {};
        std::memcpy(ubits, u, sizeof u);
        ubits[8] = ref.depth ? 1u : 0u;
        queue.WriteBuffer(show_ub, 0, ubits, sizeof ubits);
        wgpu::BindGroupEntry be[2]{};
        be[0].binding = 0;
        be[0].textureView = view;
        be[1].binding = 1;
        be[1].buffer = show_ub;
        be[1].size = 48;
        wgpu::BindGroupDescriptor bd{};
        bd.layout = show_bgl[ms];
        bd.entryCount = 2;
        bd.entries = be;
        wgpu::BindGroup group = device.CreateBindGroup(&bd);
        wgpu::RenderPassColorAttachment a{};
        a.view = canvas_view;
        a.loadOp = wgpu::LoadOp::Clear;
        a.storeOp = wgpu::StoreOp::Store;
        a.clearValue = {0, 0, 0, 1};
        wgpu::RenderPassDescriptor rp{};
        rp.label = {"playgta6-show-target-pass", WGPU_STRLEN};
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &a;
        wgpu::RenderPassEncoder pass = enc.BeginRenderPass(&rp);
        pass.SetViewport(static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.w),
                         static_cast<float>(r.h), 0.0f, 1.0f);
        pass.SetPipeline(show_pipe[ms]);
        pass.SetBindGroup(0, group);
        pass.Draw(3);
        pass.End();
        ++stats.show_target_frames;
        TargetRef shown = ref;
        shown.mip = sub.mip;
        shown.layer = sub.layer;
        show_last = shown;
        show_last_tex = st;
        show_last_sub = sub;
        set_show_label(playgta6::gpu::native_exec::show_target_label(ref, st, sub, spec));
        return true;
    }

    // F7 / F8 / F9 / F10 (show_target_model.h kHotkey*): the first one takes over from the selector.
    void show_hotkeys(uint32_t bits) {
        if (!show_spec || !bits) return;
        if (!show_manual) {
            show_manual = true;
            show_manual_target = show_last;
            show_manual_layer = show_manual_mip = -1;
        }
        if (bits & (kHotkeyPrev | kHotkeyNext)) {
            show_manual_target = show_target_cycle(show_candidates, show_manual_target, (bits & kHotkeyNext) != 0);
            show_manual_layer = show_manual_mip = -1;
        }
        if (show_last) {
            if (bits & kHotkeyMip) show_manual_mip = show_target_step(show_last_sub.mip, show_last_tex.mips);
            if (bits & kHotkeyLayer) show_manual_layer = show_target_step(show_last_sub.layer, show_last_tex.layers);
        }
    }

    // WgpuSwapchain.configureForStreamFormat for canvas formats (#709).
    void set_swapchain_format(uint8_t code) {
        wgpu::TextureFormat f;
        if (!canvas_format_for_code(code, &f)) {
            // Not a canvas format: the canvas keeps its format and the stream renders into an intermediate
            // texture of this one, blitted on present (configureForStreamFormat's blit fallback).
            if (!is_blit_stream_format_code(code)) {
                ++stats.swapchain_format_unsupported;
                return;
            }
            stream_format = wgpu_format(code);
            return;
        }
        stream_format = f;
        if (f == swap_format) return;
        swap_format = f;
        if (surface) {
            surface_configured = false;
            surface_state.invalidate();
        } else {
            create_offscreen();
        }
    }

    // ---- device loss recovery (docs/NATIVE_EXECUTOR_DESIGN.md "Device loss") ----
    //
    // Everything the executor owns on the GPU is built from packets, so the packets that build it are the
    // recoverable state: creations and buffer / texture writes are journaled (a destroyed id's entries are
    // pruned, a rewrite of the same range replaces the old write), and after a loss the journal is
    // re-dispatched onto a fresh adapter / device / surface. Pipelines come back through the blob cache,
    // shader modules from the pack. What the GPU itself produced (render targets, compute / storage
    // output) is re-created empty and counted; the next frame writes it again. A state the journal
    // cannot rebuild (disabled, over budget, device without the old features, no device) ends the run
    // with a fault record instead.
    struct JournalEntry {
        uint16_t opcode = 0;
        uint16_t flags = 0;
        uint32_t id = 0;
        std::string key;  // write dedup key ("" for creations)
        std::vector<uint8_t> payload;
        bool live = true;
    };
    std::vector<JournalEntry> journal;
    std::unordered_map<std::string, size_t> journal_write_keys;
    std::unordered_map<uint32_t, std::vector<size_t>> journal_by_id;
    size_t journal_dead = 0;
    bool journal_overflowed = false;
    bool fatal = false;
    bool drop_next_present = false;
    bool replaying = false;

    static bool journaled_opcode(uint16_t op) {
        switch (op) {
            case 3: case 4: case 5: case 8: case 12: case 13: case 14: case 15: case 19:
            case 22: case 23: case 24: case 25: case 32: case 34: case 35:
                return true;
            default:
                return false;
        }
    }

    void journal_record(const Packet& p) {
        if (!journaled_opcode(p.opcode) || options.recovery_journal_bytes == 0 || p.payload_bytes < 4 || !p.payload) return;
        uint32_t id;
        std::memcpy(&id, p.payload, 4);
        std::string key;
        if (p.opcode == 4 && p.payload_bytes >= 16) key = std::string("b") + std::string(reinterpret_cast<const char*>(p.payload), 12);
        else if (p.opcode == 13 && p.payload_bytes >= 48) key = std::string("t") + std::string(reinterpret_cast<const char*>(p.payload), 44);
        if (!key.empty()) {
            auto it = journal_write_keys.find(key);
            if (it != journal_write_keys.end() && journal[it->second].live) {
                JournalEntry& e = journal[it->second];
                const uint64_t after = stats.journal_bytes - e.payload.size() + p.payload_bytes;
                if (after > options.recovery_journal_bytes) {
                    ++stats.journal_overflows;
                    journal_overflowed = true;
                    return;
                }
                stats.journal_bytes = after;
                e.payload.assign(p.payload, p.payload + p.payload_bytes);
                e.flags = p.flags;
                return;
            }
        }
        if (stats.journal_bytes + p.payload_bytes > options.recovery_journal_bytes) {
            ++stats.journal_overflows;
            journal_overflowed = true;
            return;
        }
        JournalEntry e;
        e.opcode = p.opcode;
        e.flags = p.flags;
        e.id = id;
        e.key = key;
        e.payload.assign(p.payload, p.payload + p.payload_bytes);
        stats.journal_bytes += p.payload_bytes;
        if (!key.empty()) journal_write_keys[key] = journal.size();
        journal_by_id[id].push_back(journal.size());
        journal.push_back(std::move(e));
    }

    void journal_prune(uint32_t id) {
        auto it = journal_by_id.find(id);
        if (it == journal_by_id.end()) return;
        for (size_t idx : it->second) {
            JournalEntry& e = journal[idx];
            if (!e.live) continue;
            stats.journal_bytes -= e.payload.size();
            if (!e.key.empty()) journal_write_keys.erase(e.key);
            std::vector<uint8_t>().swap(e.payload);
            e.live = false;
            ++journal_dead;
        }
        journal_by_id.erase(it);
        if (journal_dead > 4096 && journal_dead * 2 > journal.size()) journal_compact();
    }

    void journal_compact() {
        std::vector<JournalEntry> kept;
        kept.reserve(journal.size() - journal_dead);
        journal_write_keys.clear();
        journal_by_id.clear();
        for (JournalEntry& e : journal) {
            if (!e.live) continue;
            if (!e.key.empty()) journal_write_keys[e.key] = kept.size();
            journal_by_id[e.id].push_back(kept.size());
            kept.push_back(std::move(e));
        }
        journal = std::move(kept);
        journal_dead = 0;
    }

    // Drops every object of the lost device (the executor's own state; the journal stays).
    void release_device_objects() {
        pass_cmds.clear();
        recorded_readbacks.clear();
        readback_flush_wanted = false;
        frame_target.reset();
        early_flushed = false;
        remap = RemapState{};
        buffers.clear();
        textures.clear();
        samplers.clear();
        shaders.clear();
        bgls.clear();
        pipeline_layouts.clear();
        layouts_by_content.clear();
        render_pipelines.clear();
        compute_pipelines.clear();
        deferred_render.clear();
        deferred_compute.clear();
        bind_groups.clear();
        dummy_buffers.clear();
        dummy_samplers.clear();
        dummy_views.clear();
        dummy_textures.clear();
        depth_twin_sources.clear();
        depth_copy_pipelines.clear();
        empty_layout = nullptr;
        empty_fragment = nullptr;
        error_vertex_module = nullptr;
        error_compute_module = nullptr;
        reflected_render.clear();
        reflected_compute.clear();
        manifest_bgls.clear();
        compute_hash.clear();
        compute_bind_groups.clear();
        bg_descs.clear();
        compute_meta.clear();
        comparison_samplers.clear();
        remap_cache.clear();
        default_filtering_sampler = nullptr;
        for (auto& s : fallback_samplers) s = nullptr;
        offscreen = nullptr;
        intermediate = nullptr;
        intermediate_view = nullptr;
        intermediate_format = wgpu::TextureFormat::Undefined;
        intermediate_w = intermediate_h = 0;
        blit_module = nullptr;
        blit_sampler = nullptr;
        blit_pipeline = nullptr;
        blit_pipeline_format = wgpu::TextureFormat::Undefined;
        blit_bind_group = nullptr;
        have_last_frame = false;
        ov_tex = nullptr;
        ov_view = nullptr;
        ov_tex_w = ov_tex_h = 0;
        ov_sampler = nullptr;
        ov_pipeline = nullptr;
        ov_pipeline_format = wgpu::TextureFormat::Undefined;
        ov_bg = nullptr;
        timer_slots.clear();
        time_ring.reset();
        if (surface && surface_configured) surface.Unconfigure();
        surface_configured = false;
        if (device) device.Destroy();
        device = nullptr;
        queue = nullptr;
        adapter_handle = nullptr;
    }

    static std::string json_escape(const std::string& s) {
        std::string o;
        for (char ch : s) {
            if (ch == '"' || ch == '\\') { o += '\\'; o += ch; }
            else if (static_cast<unsigned char>(ch) < 0x20) o += ' ';
            else o += ch;
        }
        return o;
    }

    // The run cannot continue: a clear fault record (stats_json), poll() returns false.
    bool fail_fatal(const std::string& why) {
        fatal = true;
        lost = true;
        ++stats.recovery_failures;
        stats.faulted = true;
        stats.fault_record = "{\"kind\": \"gpu_device_lost\", \"reason\": \"" + json_escape(why) + "\", \"device_error\": \"" +
                             json_escape(stats.last_device_error) + "\", \"device_lost_events\": " +
                             std::to_string(stats.device_lost_events) + ", \"recoveries\": " + std::to_string(stats.recoveries) +
                             ", \"frames_submitted\": " + std::to_string(stats.frames_submitted) +
                             ", \"packets\": " + std::to_string(stats.packets) + ", \"adapter\": \"" +
                             json_escape(adapter.description) + "\", \"backend\": \"" + json_escape(adapter.backend) + "\"}";
        std::fprintf(stderr, "[native_exec] fatal: GPU device lost and not recoverable: %s\n", why.c_str());
        return false;
    }

    static bool features_equal(const DeviceFeatures& a, const DeviceFeatures& b) {
        return a.bc == b.bc && a.depth32_stencil8 == b.depth32_stencil8 && a.float32_filterable == b.float32_filterable &&
               a.bgra8_storage == b.bgra8_storage && a.rg11b10_renderable == b.rg11b10_renderable && a.tier1 == b.tier1 &&
               a.timestamp_query == b.timestamp_query;
    }

    // True: the device is usable (nothing was lost, or the recovery worked); false: fatal, the run is over.
    bool ensure_device() {
        if (fatal) return false;
        if (!lost) return true;
        return recover();
    }

    bool recover() {
        const uint64_t t0 = now_ns();
        if (options.recovery_journal_bytes == 0) return fail_fatal("the resource journal is disabled (recovery_journal_bytes = 0)");
        if (journal_overflowed)
            return fail_fatal("the resource journal outgrew its budget (" + std::to_string(options.recovery_journal_bytes) +
                              " bytes): resources cannot be rebuilt");
        std::fprintf(stderr, "[native_exec] GPU device lost (%s): recovering\n", stats.last_device_error.c_str());
        // The frame being built is gone; its read-backs complete as failed so no fence stays held.
        const bool partial_frame = !pass_cmds.empty() || early_flushed;
        fail_dropped_readbacks();
        start_recorded_readbacks(false);
        const DeviceFeatures before = features;
        const ClearValue keep_clear = clear_color;
        const uint64_t replayed_before = stats.recovery_replayed_packets;
        std::string last_error = "no attempt";
        bool ok = false;
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
            release_device_objects();
            if (attempt) std::this_thread::sleep_for(std::chrono::milliseconds(250));
            lost = false;
            CreateStatus st = CreateStatus::Failed;
            std::string err;
            if (!acquire_device(&st, &err)) {
                last_error = err;
                continue;
            }
            if (!features_equal(before, features)) return fail_fatal("the new device lacks features the old one had");
            const uint64_t errors_before = stats.packet_errors;
            replaying = true;
            for (const JournalEntry& e : journal) {
                if (!e.live) continue;
                Packet p;
                p.opcode = e.opcode;
                p.flags = e.flags;
                p.payload_bytes = static_cast<uint32_t>(e.payload.size());
                p.payload = e.payload.data();
                dispatch_op(p);
                ++stats.recovery_replayed_packets;
                if (lost) break;
            }
            replaying = false;
            stats.recovery_replay_errors += stats.packet_errors - errors_before;
            if (lost) {
                last_error = "the device was lost again while the resources were rebuilt";
                continue;
            }
            ok = true;
        }
        if (!ok) return fail_fatal("could not create a new device: " + last_error);
        clear_color = keep_clear;
        for (const auto& b : buffers)
            if (b.second.usage & 0x0080u) ++stats.recovery_gpu_resources_reset;  // Storage
        for (const auto& t : textures)
            if (t.second.usage & 0x0018u) ++stats.recovery_gpu_resources_reset;  // StorageBinding | RenderAttachment
        if (partial_frame) drop_next_present = true;
        ++stats.recoveries;
        stats.recovery_ns_total += now_ns() - t0;
        std::fprintf(stderr, "[native_exec] device recovered in %llu ms (%llu packet(s) replayed)\n",
                     static_cast<unsigned long long>((now_ns() - t0) / 1000000ull),
                     static_cast<unsigned long long>(stats.recovery_replayed_packets - replayed_before));
        return true;
    }

    // ---- packets ----

    void reset_frame_state() {
        frame_passes.clear();
        frame_pass_count = 0;
        clear_color = kDefaultClearColor;
        pass_cmds.clear();
        remap = RemapState{};
    }

    // The worker's `_packetError` (a rejected v2 / texture packet), counted per opcode.
    void packet_error(uint16_t op, const std::string& message) {
        ++stats.packet_errors;
        if (op < kOpcodeSlots) ++stats.packet_errors_by_opcode[op];
        stats.last_packet_error = message;
    }

    static uint64_t now_ns() {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    static constexpr uint64_t kSlowPacketNs = 1000ull * 1000000ull;
    std::string slow_note;  // what the current packet materialized (ensure_shader), for the slow log

    void note_slow_packet(uint16_t opcode, uint64_t ns) {
        ++stats.slow_packets;
        stats.slow_packet_max_ns = std::max(stats.slow_packet_max_ns, ns);
        if (stats.slow_packets <= 32) {
            std::fprintf(stderr, "[native_exec] slow packet: opcode %u took %llu ms%s%s\n", static_cast<unsigned>(opcode),
                         static_cast<unsigned long long>(ns / 1000000), slow_note.empty() ? "" : "; ", slow_note.c_str());
        }
    }

    bool dispatch(const Packet& p) {
        if (lost || fatal) {
            if (!ensure_device()) {
                ++stats.dropped_while_lost;
                return false;
            }
        }
        journal_record(p);
        if (p.opcode == static_cast<uint16_t>(GpuOpcode::DestroyResource) && p.payload_bytes >= 4) journal_prune(read_payload<uint32_t>(p));
        pump_deferred();
        const uint64_t t0 = now_ns();
        bool r;
        const bool diag_scopes = error_scope_active(options.gpu_error_scopes, stats.device_errors, error_scoped_total.load());
        if (options.gpu_validation || diag_scopes) {
            validation_cur_packet = validation_packet_index++;
            validation_cur_opcode = p.opcode;
            ++validation_scoped_packets;
            const bool is_present = p.opcode == static_cast<uint16_t>(GpuOpcode::Present);
            const uint64_t present_index = stats.present_packets;
            // A present encodes, finishes and submits the recorded passes: take their context before it is consumed.
            std::string ctx = diag_scopes && is_present ? frame_scope_context() : std::string();
            device.PushErrorScope(wgpu::ErrorFilter::Internal);
            device.PushErrorScope(wgpu::ErrorFilter::OutOfMemory);
            device.PushErrorScope(wgpu::ErrorFilter::Validation);
            r = dispatch_op(p);
            if (diag_scopes) {
                const std::string pc = packet_scope_context(p);
                if (!pc.empty()) ctx += (ctx.empty() ? "" : " ") + pc;
            }
            struct Kind {
                const char* scope_kind;
                const char* diag_kind;
            };
            for (const Kind k : {Kind{"scope-validation", "validation"}, Kind{"scope-oom", "out-of-memory"},
                                 Kind{"scope-internal", "internal"}}) {
                struct Ctx {
                    Impl* self;
                    uint64_t packet;
                    uint16_t opcode;
                    uint64_t present_index;
                    Kind kind;
                    bool validation, diag;
                    std::string context;
                };
                auto* c = new Ctx{this, validation_cur_packet, p.opcode, present_index, k, options.gpu_validation, diag_scopes, ctx};
                device.PopErrorScope(
                    wgpu::CallbackMode::AllowSpontaneous,
                    [](wgpu::PopErrorScopeStatus, wgpu::ErrorType type, wgpu::StringView msg, Ctx* c) {
                        if (type != wgpu::ErrorType::NoError && type != wgpu::ErrorType::Unknown) {
                            if (c->validation) c->self->record_validation(c->kind.scope_kind, to_string(msg), c->packet, c->opcode);
                            if (c->diag) {
                                c->self->error_scope_log.record(c->kind.diag_kind, to_string(msg), error_scope_phase(c->opcode),
                                                                c->context, c->present_index, c->packet, c->opcode);
                                ++c->self->error_scoped_total;
                            }
                        }
                        delete c;
                    },
                    c);
            }
        } else {
            r = dispatch_op(p);
        }
        const uint64_t dt = now_ns() - t0;
        if (p.opcode < kOpcodeSlots) stats.opcode_ns[p.opcode] += dt;
        if (dt >= kSlowPacketNs) note_slow_packet(p.opcode, dt);
        slow_note.clear();
        return r;
    }

    bool dispatch_op(const Packet& p) {
        const uint16_t op = p.opcode;
        bool handled = true;
        const uint32_t n = p.payload_bytes;
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
                if (n >= sizeof(GpuWriteBufferPayload)) write_buffer(p);
                else ++stats.op_errors;
                break;
            case GpuOpcode::CreateShaderModule:
                if (n >= 16) create_shader_module(p);
                else ++stats.op_errors;
                break;
            case GpuOpcode::Present:
                ++stats.present_packets;
                present();
                pipe_budget.begin_frame();
                break;
            case GpuOpcode::DestroyResource:
                if (n >= 4) destroy_resource(read_payload<uint32_t>(p));
                break;
            case GpuOpcode::CreateTexture:
                if (n >= 32) create_texture_v1(p);
                else ++stats.op_errors;
                break;
            case GpuOpcode::CreateTexture2:
                create_texture_v2(p);
                break;
            case GpuOpcode::WriteTexture:
                if (n >= 48) write_texture(p);
                else ++stats.op_errors;
                break;
            case kGpuOpcodeClearTexture:
                if (n >= sizeof(GpuClearTexturePayload)) clear_texture(read_payload<GpuClearTexturePayload>(p).texture_id);
                else packet_error(op, "ClearTexture: payload too short");
                break;
            case GpuOpcode::CreateSampler:
                if (n >= 32) create_sampler_v1(p);
                else ++stats.op_errors;
                break;
            case GpuOpcode::CreateSampler2:
                if (n >= 24) create_sampler_v2(p);
                else packet_error(op, "CreateSampler2: malformed or truncated v2 packet");
                break;
            case GpuOpcode::SetPipeline:
                if (n >= 4) {
                    const uint32_t id = read_payload<uint32_t>(p);
                    auto it = render_pipelines.find(id);
                    pass_cmds.emplace_back(SetPipelineCmd{id, it != render_pipelines.end() && !it->second.pending
                                                                ? it->second.pipeline
                                                                : wgpu::RenderPipeline()});
                } else {
                    ++stats.op_errors;
                }
                break;
            case GpuOpcode::SetVertexBuffer:
                if (n >= 16) {
                    const auto v = read_payload<GpuSetVertexBufferPayload>(p);
                    pass_cmds.emplace_back(SetVertexBufferCmd{v.slot, v.buffer_id, v.offset});
                } else {
                    ++stats.op_errors;
                }
                break;
            case GpuOpcode::SetIndexBuffer:
                if (n >= 16) {
                    const auto v = read_payload<GpuSetIndexBufferPayload>(p);
                    pass_cmds.emplace_back(SetIndexBufferCmd{v.buffer_id, v.index_format, v.offset});
                } else {
                    ++stats.op_errors;
                }
                break;
            case GpuOpcode::Draw:
                if (n >= 16) {
                    const auto v = read_payload<GpuDrawPayload>(p);
                    pass_cmds.emplace_back(DrawCmd{v.vertex_count, v.instance_count, v.first_vertex, v.first_instance});
                } else {
                    ++stats.op_errors;
                }
                break;
            case GpuOpcode::DrawIndexed:
                if (n >= 20) {
                    const auto v = read_payload<GpuDrawIndexedPayload>(p);
                    pass_cmds.emplace_back(
                        DrawIndexedCmd{v.index_count, v.instance_count, v.first_index, v.base_vertex, v.first_instance});
                } else {
                    ++stats.op_errors;
                }
                break;
            case GpuOpcode::SetComputePipeline:
                if (n >= 4) pass_cmds.emplace_back(SetComputePipelineCmd{read_payload<uint32_t>(p)});
                break;
            case GpuOpcode::DispatchCompute:
                if (n >= 12) {
                    const auto v = read_payload<GpuDispatchComputePayload>(p);
                    pass_cmds.emplace_back(DispatchCmd{v.workgroup_x, v.workgroup_y, v.workgroup_z});
                }
                break;
            case GpuOpcode::DrawIndirect:
            case GpuOpcode::DrawIndexedIndirect:
            case GpuOpcode::DispatchIndirect:
                if (n >= 16) {
                    const auto v = read_payload<GpuIndirectPayload>(p);
                    pass_cmds.emplace_back(IndirectCmd{
                        static_cast<uint8_t>(op == static_cast<uint16_t>(GpuOpcode::DrawIndirect)          ? 0
                                             : op == static_cast<uint16_t>(GpuOpcode::DrawIndexedIndirect) ? 1
                                                                                                           : 2),
                        v.buffer_id, v.offset});
                } else {
                    ++stats.op_errors;
                }
                break;
            // ABI v2 (render_state_packets.js handlePacket). A rejected packet is counted as a
            // packet error and is still "handled" (the worker counts it too).
            case GpuOpcode::CreateBindGroupLayout:
                if (!create_bind_group_layout(p)) packet_error(op, pending_error("malformed bind group layout"));
                break;
            case GpuOpcode::CreatePipelineLayout:
                if (!create_pipeline_layout(p)) packet_error(op, pending_error("malformed pipeline layout"));
                break;
            case GpuOpcode::CreateRenderPipeline2:
                if (!(n >= 16 + sizeof(GpuRenderPipelineKey) && create_render_pipeline2(p)))
                    packet_error(op, pending_error("malformed or truncated v2 packet"));
                break;
            case GpuOpcode::CreateComputePipeline2:
                if (!(n >= 32 && create_compute_pipeline2(p))) packet_error(op, pending_error("malformed or truncated v2 packet"));
                break;
            case GpuOpcode::BeginRenderPass:
                if (!(n >= sizeof(GpuBeginRenderPassPayload) && begin_render_pass(p)))
                    packet_error(op, pending_error("malformed or truncated v2 packet"));
                break;
            case GpuOpcode::EndRenderPass:
                pass_cmds.emplace_back(EndPassCmd{});
                break;
            case GpuOpcode::SetViewport:
                if (n >= 24) {
                    const auto v = read_payload<GpuSetViewportPayload>(p);
                    Viewport vp;
                    vp.x = v.x; vp.y = v.y; vp.width = v.width; vp.height = v.height;
                    vp.min_depth = v.min_depth; vp.max_depth = v.max_depth;
                    pass_cmds.emplace_back(ViewportCmd{vp});
                } else {
                    packet_error(op, "malformed or truncated v2 packet");
                }
                break;
            case GpuOpcode::SetScissorRect:
                if (n >= 16) {
                    const auto v = read_payload<GpuSetScissorRectPayload>(p);
                    pass_cmds.emplace_back(ScissorCmd{v.x, v.y, v.width, v.height});
                } else {
                    packet_error(op, "malformed or truncated v2 packet");
                }
                break;
            case GpuOpcode::SetBlendConstant:
                if (n >= 16) {
                    const auto v = read_payload<GpuSetBlendConstantPayload>(p);
                    pass_cmds.emplace_back(BlendConstantCmd{v.r, v.g, v.b, v.a});
                } else {
                    packet_error(op, "malformed or truncated v2 packet");
                }
                break;
            case GpuOpcode::SetStencilReference:
                if (n >= 4) pass_cmds.emplace_back(StencilRefCmd{read_payload<uint32_t>(p)});
                else packet_error(op, "malformed or truncated v2 packet");
                break;
            case GpuOpcode::CreateBindGroup2:
                if (!create_bind_group2(p)) packet_error(op, pending_error("malformed or truncated v2 packet"));
                break;
            case GpuOpcode::SetBindGroup2:
                if (!set_bind_group2(p)) packet_error(op, pending_error("malformed or truncated v2 packet"));
                break;
            // v1 pipelines and bind groups (wgpu_worker.js, no reflection registry loaded).
            case GpuOpcode::CreateRenderPipeline:
                if (n >= sizeof(GpuCreateRenderPipelinePayload)) create_render_pipeline1(p);
                else ++stats.op_errors;
                break;
            case GpuOpcode::CreateComputePipeline:
                if (n >= sizeof(GpuCreateComputePipelinePayload)) create_compute_pipeline1(p);
                else ++stats.op_errors;
                break;
            case GpuOpcode::CreateBindGroup:
                if (n >= sizeof(GpuCreateBindGroupPayload)) create_bind_group1(p);
                else ++stats.op_errors;
                break;
            case GpuOpcode::SetBindGroup:
                if (n >= sizeof(GpuSetBindGroupPayload)) {
                    const auto v = read_payload<GpuSetBindGroupPayload>(p);
                    auto it = bind_groups.find(v.bind_group_id);
                    pass_cmds.emplace_back(SetBindGroup1Cmd{v.group_index, v.bind_group_id,
                                                            it != bind_groups.end() ? it->second : wgpu::BindGroup()});
                } else {
                    ++stats.op_errors;
                }
                break;
            case kGpuOpcodeCopyTexture:
                if (n >= sizeof(GpuCopyTexturePayload)) {
                    pass_cmds.emplace_back(CopyTextureCmd{read_payload<GpuCopyTexturePayload>(p)});
                } else {
                    packet_error(op, "CopyTexture: payload too short");
                }
                break;
            case GpuOpcode::ReadbackBuffer:
            case GpuOpcode::ReadbackTexture:
                if (auto r = decode_readback(p)) {
                    ++stats.readbacks_requested;
                    readback_flush_wanted = true;
                    pass_cmds.emplace_back(ReadbackCmd{*r});
                } else {
                    ++stats.op_errors;
                }
                break;
            default:
                handled = false;
                break;
        }
        if (op >= kOpcodeSlots) ++stats.unknown_opcodes;
        else if (handled) ++stats.handled[op];
        else ++stats.unhandled[op];
        return !fatal;
    }

    std::string packet_error_text;
    bool packet_error_set = false;
    bool fail(const std::string& message) {
        packet_error_text = message;
        packet_error_set = true;
        return false;
    }
    std::string pending_error(const char* fallback) {
        std::string s = packet_error_set ? packet_error_text : std::string(fallback);
        packet_error_set = false;
        return s;
    }

    // ---- buffers (M0) ----

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
    // is done with it (the worker's deferred destroy). Bind groups and pass commands that already
    // hold a handle keep it (the worker resolves SetBindGroup2 / SetPipeline at packet time too).
    void destroy_resource(uint32_t id) {
        buffers.erase(id);
        textures.erase(id);
        pending_clears.erase(id);
        samplers.erase(id);
        bind_groups.erase(id);
        shaders.erase(id);
        render_pipelines.erase(id);
        compute_pipelines.erase(id);
        deferred_render.erase(id);
        deferred_compute.erase(id);
        reflected_render.erase(id);
        reflected_compute.erase(id);
        compute_hash.erase(id);
        compute_bind_groups.erase(id);
        bgls.erase(id);
        pipeline_layouts.erase(id);
        bg_descs.erase(id);
        compute_meta.erase(id);
        comparison_samplers.erase(id);
    }

    // ---- shaders (wgpu_worker.js _handleCreateShaderModule / _resolveShaderPayload) ----

    bool load_pack(const std::string& path, std::string* error) {
        auto p = std::make_unique<WgslPack>();
        if (!p->load(path, error)) return false;
        pack = std::move(p);
        // The boot orchestrator's deriveReflectionSidecarUrl: `<pack>.reflect.json`, optional.
        std::error_code ec;
        const std::string sidecar = path + ".reflect.json";
        if (std::filesystem::is_regular_file(sidecar, ec)) {
            std::string rerr;
            if (!load_reflection_file(sidecar, &rerr)) stats.last_reflection_error = sidecar + ": " + rerr;
        }
        return true;
    }

    bool load_reflection_text(const std::string& text, std::string* error) {
        const std::string err = reflection.load(text);
        stats.reflection_shaders = reflection.shader_count();
        if (!err.empty()) {
            ++stats.reflection_load_errors;
            stats.last_reflection_error = err;
            if (error) *error = err;
            return false;
        }
        ++stats.reflection_documents;
        return true;
    }

    bool load_reflection_file(const std::string& path, std::string* error) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            ++stats.reflection_load_errors;
            if (error) *error = "cannot open the reflection document";
            return false;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return load_reflection_text(ss.str(), error);
    }

    // _resolveShaderReflection: the registry record of a shader by its hash, then by "<stage>:<entry>".
    // (`registerShaderReflectionKey` has no native producer, so no explicit key.)
    const reflect::StageRecord* resolve_shader_reflection(const ShaderEntry& e, const char* stage_name) {
        if (reflection.shader_count() == 0) return nullptr;
        if (const auto* r = reflection.lookup_hash(e.reflect_hash)) return r;
        if (e.explicit_ep && !e.explicit_ep->empty())
            if (const auto* r = reflection.lookup_name(std::string(stage_name) + ":" + *e.explicit_ep)) return r;
        return nullptr;
    }

    static std::optional<uint8_t> format_code_by_name(const std::string& name) {
        for (uint8_t c = 1; c < kFormatCodeCount; ++c)
            if (name == format_info(c).name) return c;
        return std::nullopt;
    }

    // `pipelineCache.createBindGroupLayoutFromReflection(desc)` / `device.createBindGroupLayout`.
    // false: an entry Dawn cannot take (counted; the worker's createBindGroupLayout would throw).
    bool make_bgl_from_entries(const std::vector<reflect::LayoutEntryDesc>& entries, wgpu::BindGroupLayout* out) {
        using K = reflect::LayoutEntryDesc::Kind;
        std::vector<wgpu::BindGroupLayoutEntry> wes(entries.size());
        for (size_t i = 0; i < entries.size(); ++i) {
            const reflect::LayoutEntryDesc& le = entries[i];
            wgpu::BindGroupLayoutEntry& w = wes[i];
            w.binding = le.binding;
            w.visibility = static_cast<wgpu::ShaderStage>(le.visibility);
            switch (le.kind) {
                case K::Buffer:
                    w.buffer.type = le.buffer_type == 0   ? wgpu::BufferBindingType::Uniform
                                    : le.buffer_type == 1 ? wgpu::BufferBindingType::Storage
                                                          : wgpu::BufferBindingType::ReadOnlyStorage;
                    w.buffer.hasDynamicOffset = le.has_dynamic_offset;
                    w.buffer.minBindingSize = le.min_binding_size;
                    break;
                case K::Sampler:
                    w.sampler.type = le.sampler_type == 0   ? wgpu::SamplerBindingType::Filtering
                                     : le.sampler_type == 1 ? wgpu::SamplerBindingType::NonFiltering
                                                            : wgpu::SamplerBindingType::Comparison;
                    break;
                case K::Texture: {
                    static const wgpu::TextureSampleType kS[5] = {
                        wgpu::TextureSampleType::Float, wgpu::TextureSampleType::UnfilterableFloat,
                        wgpu::TextureSampleType::Depth, wgpu::TextureSampleType::Sint, wgpu::TextureSampleType::Uint};
                    w.texture.sampleType = kS[le.sample_type < 5 ? le.sample_type : 0];
                    // 'float' + multisampled is invalid in WebGPU (a Texture2DMS<float4> SRV): non-filterable
                    if (le.multisampled && w.texture.sampleType == wgpu::TextureSampleType::Float)
                        w.texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
                    w.texture.viewDimension = wgpu_view_dim(le.view_dim);
                    w.texture.multisampled = le.multisampled;
                    break;
                }
                case K::StorageTexture: {
                    const auto fc = format_code_by_name(le.storage_format);
                    if (!fc) {
                        ++stats.reflection_layout_unsupported;
                        return false;
                    }
                    w.storageTexture.access = le.access == 0   ? wgpu::StorageTextureAccess::WriteOnly
                                              : le.access == 1 ? wgpu::StorageTextureAccess::ReadOnly
                                                               : wgpu::StorageTextureAccess::ReadWrite;
                    w.storageTexture.format = wgpu_format(*fc);
                    w.storageTexture.viewDimension = wgpu_view_dim(le.view_dim);
                    break;
                }
                case K::ExternalTexture:  // the pinned Dawn has no ExternalTextureBindingLayout
                    ++stats.reflection_layout_unsupported;
                    return false;
            }
        }
        wgpu::BindGroupLayoutDescriptor d{};
        d.entryCount = wes.size();
        d.entries = wes.data();
        *out = device.CreateBindGroupLayout(&d);
        return true;
    }

    // _buildReflectedPipelineLayout: false + !*error for no reflection record at all (the pipeline
    // stays on layout:'auto'); false + *error when the loader's merge or Dawn rejects it (the
    // worker's op error, the pipeline is not created).
    bool build_reflected_layout(const std::vector<const reflect::StageRecord*>& refls, ReflectedLayout* out, bool* error) {
        *error = false;
        if (refls.empty()) return false;
        reflect::LayoutBuildResult r = reflect::build_bind_group_layout_descriptors(refls);
        if (!r.ok) {
            ++stats.reflection_layout_errors;
            if (r.conflict) ++stats.reflection_binding_conflicts;
            stats.last_reflection_error = r.error;
            ++stats.op_errors;
            *error = true;
            return false;
        }
        ReflectedLayout l;
        for (auto& g : r.groups) {
            wgpu::BindGroupLayout bgl;
            if (!make_bgl_from_entries(g, &bgl)) {
                ++stats.op_errors;
                *error = true;
                return false;
            }
            l.bgls.push_back(std::move(bgl));
        }
        wgpu::PipelineLayoutDescriptor pd{};
        pd.bindGroupLayoutCount = l.bgls.size();
        pd.bindGroupLayouts = l.bgls.data();
        l.pipeline_layout = device.CreatePipelineLayout(&pd);
        l.groups = std::move(r.groups);
        *out = std::move(l);
        return true;
    }

    // A bind group layout of WGSL bindings (a fitted group): buffers without a minimum size (checked
    // at the draw against the shader, like the title's own entries), the rest as the remap synthesizes.
    wgpu::BindGroupLayout wgsl_bgl(const std::vector<FitBinding>& fb) {
        std::vector<wgpu::BindGroupLayoutEntry> wes(fb.size());
        for (size_t i = 0; i < fb.size(); ++i) {
            const SynthLayoutEntry se = synth_layout_entry(fb[i].b);
            wgpu::BindGroupLayoutEntry& w = wes[i];
            w.binding = se.binding;
            w.visibility = static_cast<wgpu::ShaderStage>(fb[i].visibility);
            if (se.kind == 0) {
                w.buffer.type = se.subtype == 0   ? wgpu::BufferBindingType::Uniform
                                : se.subtype == 1 ? wgpu::BufferBindingType::Storage
                                                  : wgpu::BufferBindingType::ReadOnlyStorage;
            } else if (se.kind == 1) {
                w.sampler.type = se.subtype == 2 ? wgpu::SamplerBindingType::Comparison : wgpu::SamplerBindingType::Filtering;
            } else if (se.kind == 2) {
                static const wgpu::TextureSampleType kS[5] = {
                    wgpu::TextureSampleType::Float, wgpu::TextureSampleType::UnfilterableFloat,
                    wgpu::TextureSampleType::Depth, wgpu::TextureSampleType::Sint, wgpu::TextureSampleType::Uint};
                w.texture.sampleType = kS[se.subtype < 5 ? se.subtype : 0];
                if (se.multisampled && w.texture.sampleType == wgpu::TextureSampleType::Float)
                    w.texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
                w.texture.viewDimension = wgpu_view_dim(se.view_dim);
                w.texture.multisampled = se.multisampled;
            } else {
                const StorageResolution sr = resolve_storage_format(se.storage_format, features);
                const uint8_t f = sr.supported && sr.storage_format ? sr.storage_format : 22;
                w.storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
                w.storageTexture.format = wgpu_format(f);
                w.storageTexture.viewDimension = wgpu_view_dim(se.view_dim == 4 || se.view_dim == 5 ? 3 : se.view_dim);
            }
        }
        wgpu::BindGroupLayoutDescriptor d{};
        d.entryCount = wes.size();
        d.entries = wes.data();
        return device.CreateBindGroupLayout(&d);
    }

    // A bind group of dummy resources for a fitted group the stream does not bind.
    wgpu::BindGroup fit_dummy_group(const wgpu::BindGroupLayout& bgl, const std::vector<FitBinding>& fb) {
        const uint64_t owner = (1ull << 63) | ++fit_dummy_owner;  // distinct storage-texture dummies
        std::vector<wgpu::BindGroupEntry> wes(fb.size());
        for (size_t i = 0; i < fb.size(); ++i) {
            const SynthLayoutEntry se = synth_layout_entry(fb[i].b);
            LayoutEntry le;
            le.binding = se.binding;
            le.kind = se.kind;
            le.subtype = se.subtype;
            le.view_dim = se.kind == 3 && (se.view_dim == 4 || se.view_dim == 5) ? 3 : se.view_dim;
            le.storage_format = se.storage_format;
            le.multisampled = se.multisampled;
            le.min_binding_size = 65536;  // the largest uniform binding a WGSL struct can ask for by default
            const Dummy dm = dummy_resource(le, owner);
            wes[i].binding = se.binding;
            wes[i].buffer = dm.buffer;
            wes[i].sampler = dm.sampler;
            wes[i].textureView = dm.view;
            wes[i].size = wgpu::kWholeSize;
        }
        wgpu::BindGroupDescriptor bd{};
        bd.layout = bgl;
        bd.entryCount = wes.size();
        bd.entries = wes.data();
        return device.CreateBindGroup(&bd);
    }

    // The title's explicit layout fitted to the pipeline's WGSL (layout_model.h): null when the layout
    // serves the WGSL as it is (or is a manifest layout, or would stay over the limits). A group 0 that
    // the remap binds (`meta.remap`, or the title's layout is over a per-stage limit: the remap then
    // binds only what the WGSL references) is built from the WGSL and handed to the remap in
    // `meta.reflected_bgl0`; groups above 0 that the title's layout lacks are built from the WGSL and
    // get dummy bind groups (`dummies`) for passes that do not bind them.
    wgpu::PipelineLayout fit_pipeline_layout(const PlEntry& pl, const std::vector<FitStage>& stages, AutoLayoutMeta& meta,
                                             const std::vector<WgslBinding>& group0, FitDummies* dummies) {
        if (pl.fit_groups.empty()) return {};
        size_t ng = 0;
        for (const FitStage& st : stages)
            if (st.groups) ng = std::max(ng, st.groups->size());
        std::vector<std::vector<FitBinding>> wg(ng);
        for (size_t g = 0; g < ng; ++g) wg[g] = merge_fit_group(stages, g);
        auto limits_of = [](const std::vector<LayoutEntry>& es, std::vector<LimitEntry>* out) {
            for (const LayoutEntry& le : es) out->push_back({le.visibility, le.kind, le.subtype});
        };
        std::vector<LimitEntry> title_entries;
        for (const auto& eg : pl.entry_groups) limits_of(eg, &title_entries);
        const bool over = !stage_limit_excess(title_entries, stage_limits).empty();
        bool upper = false;
        for (size_t g = 1; g < ng; ++g) upper = upper || !wg[g].empty();
        const bool remap0 = meta.remap || over;
        std::vector<FitSource> src = fit_layout_groups(pl.fit_groups, wg, remap0);
        if (!layout_needs_fit(src, pl.fit_groups.size(), over) && !(meta.remap && upper)) return {};
        std::vector<LimitEntry> fitted;
        for (size_t g = 0; g < src.size(); ++g) {
            if (src[g] == FitSource::Explicit) limits_of(pl.entry_groups[g], &fitted);
            else if (src[g] == FitSource::Wgsl)
                for (const FitBinding& f : wg[g]) fitted.push_back(limit_entry_of(f));
        }
        uint8_t force = 0;
        if (!stage_limit_excess(fitted, stage_limits).empty()) {
            std::vector<FitSource> padded = src;
            force = replace_unused_groups(&padded, wg);
            fitted.clear();
            for (size_t g = 0; g < padded.size(); ++g) {
                if (padded[g] == FitSource::Explicit) limits_of(pl.entry_groups[g], &fitted);
                else if (padded[g] == FitSource::Wgsl)
                    for (const FitBinding& f : wg[g]) fitted.push_back(limit_entry_of(f));
            }
            const std::string excess = stage_limit_excess(fitted, stage_limits);
            if (!force || !excess.empty()) {
                if (++stats.layout_fit_still_over <= 8)
                    std::fprintf(stderr, "[native_exec] layout fit: the WGSL alone is over a per-stage limit: %s\n",
                                 excess.c_str());
                return {};
            }
            src = std::move(padded);
            for (uint32_t g = 0; g < kFitMaxGroups; ++g)
                if (force & (1u << g)) ++stats.layout_fit_unused_groups;
        }
        std::vector<wgpu::BindGroupLayout> bgls(src.size());
        FitDummies fd;
        // Over the limits with a group 0 that matches the WGSL: the title's own entries for the bindings
        // the WGSL references, visible to the stages that reference them (dynamic offsets fold into the
        // remap's static ones).
        const bool by_binding = remap0 && !meta.remap && !wg.empty() && !wg[0].empty() && !pl.entry_groups.empty();
        std::vector<LayoutEntry> bb;
        if (by_binding) {
            for (const FitBinding& f : wg[0])
                for (const LayoutEntry& le : pl.entry_groups[0])
                    if (le.binding == f.b.binding) {
                        LayoutEntry x = le;
                        x.visibility = f.visibility;
                        x.dynamic = false;
                        // The title's storage entry for a WGSL read-only binding: Dawn wants the WGSL's access.
                        if (x.kind == 0)
                            x.subtype = f.b.kind == WgslBindingKind::Uniform ? 0 : f.b.kind == WgslBindingKind::Storage ? 1 : 2;
                        bb.push_back(x);
                    }
            std::vector<wgpu::BindGroupLayoutEntry> wes(bb.size());
            for (size_t i = 0; i < bb.size(); ++i) to_wgpu_layout_entry(bb[i], &wes[i]);
            wgpu::BindGroupLayoutDescriptor d{};
            d.entryCount = wes.size();
            d.entries = wes.data();
            bgls[0] = device.CreateBindGroupLayout(&d);
        }
        for (size_t g = 0; g < src.size(); ++g) {
            if (src[g] == FitSource::Explicit) {
                bgls[g] = pl.bgls[g];
                continue;
            }
            if (g == 0 && by_binding) continue;
            const std::vector<FitBinding> none;
            const std::vector<FitBinding>& fb = src[g] == FitSource::Wgsl ? wg[g] : none;
            bgls[g] = fb.empty() ? empty_bgl() : wgsl_bgl(fb);
            if (g > 0 && src[g] == FitSource::Wgsl) ++stats.layout_fit_wgsl_groups;
            // Group 0 with bindings is the remap's; everything else the stream does not bind gets dummies.
            if (g == 0 && !fb.empty()) continue;
            fd.mask |= static_cast<uint8_t>(1u << g);
            fd.groups[g] = fit_dummy_group(bgls[g], fb);
        }
        wgpu::PipelineLayoutDescriptor pd{};
        pd.bindGroupLayoutCount = bgls.size();
        pd.bindGroupLayouts = bgls.data();
        wgpu::PipelineLayout fitted_pl = device.CreatePipelineLayout(&pd);
        if (remap0) {
            if (!meta.remap) {
                meta.remap = true;
                meta.bindings = group0;
                meta.by_binding = by_binding;
                meta.by_binding_entries = std::move(bb);
            }
            meta.reflected_bgl0 = bgls[0];
        }
        if (over) ++stats.layout_fit_over_limits;
        ++stats.layout_fit_pipelines;
        fd.force = force;
        *dummies = std::move(fd);
        return fitted_pl;
    }

    // v2 remap pipelines: an explicit layout from the stages' reflection instead of layout:'auto', so
    // nothing has to wait for the compiled pipeline to get its group 0 layout. Null (stay on 'auto',
    // budgeted wait later) unless every stage has a record and its group 0 binds exactly the WGSL's
    // bindings.
    wgpu::PipelineLayout reflected_remap_layout(AutoLayoutMeta& meta, const std::vector<const reflect::StageRecord*>& refls,
                                                size_t stage_count) {
        if (refls.size() != stage_count) return {};
        reflect::LayoutBuildResult r = reflect::build_bind_group_layout_descriptors(refls);
        if (!r.ok || r.groups.empty()) return {};
        std::vector<uint32_t> nums;
        for (const auto& e : r.groups[0]) nums.push_back(e.binding);
        if (!remap_bindings_match(nums, meta.bindings)) return {};
        std::vector<wgpu::BindGroupLayout> bgls;
        for (auto& g : r.groups) {
            wgpu::BindGroupLayout bgl;
            if (!make_bgl_from_entries(g, &bgl)) return {};
            bgls.push_back(std::move(bgl));
        }
        wgpu::PipelineLayoutDescriptor pd{};
        pd.bindGroupLayoutCount = bgls.size();
        pd.bindGroupLayouts = bgls.data();
        wgpu::PipelineLayout pl = device.CreatePipelineLayout(&pd);
        meta.reflected_bgl0 = bgls[0];
        ++stats.remap_reflected_layouts;
        return pl;
    }

    void create_shader_module(const Packet& p) {
        const auto h = read_payload<GpuCreateShaderModulePayload>(p);
        if (uint64_t{16} + h.code_len > p.payload_bytes) {
            ++stats.op_errors;  // truncated payload
            return;
        }
        const std::string raw(reinterpret_cast<const char*>(p.payload) + 16, h.code_len);
        const std::string_view trimmed = trim_view(raw);
        std::string code = raw;
        uint64_t shader_hash = 0;
        const uint64_t text_hash = reflect::fnv1a64(raw);
        if (trimmed.empty()) {
            // The worker looks up a synthetic `empty:<stage>:<id>` hash, which no pack holds: the empty
            // text compiles as an (empty) module.
        } else if (auto hash = parse_hash_only_shader_payload(trimmed)) {
            shader_hash = *hash;
            if (pack && pack->has(*hash)) {
                ++stats.shader_pack_hits;
                auto e = std::make_shared<ShaderEntry>();
                e->stage = h.stage;
                e->hash = shader_hash;
                e->reflect_hash = shader_hash;
                e->pending = true;
                shaders[h.id] = std::move(e);
                ++stats.shader_modules;
                return;
            } else {
                ++stats.shader_pack_misses;  // the hash text itself is compiled (and rejected)
            }
        }
        const char* stage_name = h.stage == 0 ? "vertex" : h.stage == 1 ? "fragment" : h.stage == 2 ? "compute" : "";
        auto e = std::make_shared<ShaderEntry>();
        e->stage = h.stage;
        e->hash = shader_hash;
        e->reflect_hash = text_hash;
        e->explicit_ep = extract_wgsl_entry_point(code, stage_name);
        e->module = make_module(code);
        e->code = std::move(code);
        shaders[h.id] = std::move(e);
        ++stats.shader_modules;
    }

    static const char* stage_name_of(uint32_t stage) {
        return stage == 0 ? "vertex" : stage == 1 ? "fragment" : stage == 2 ? "compute" : "";
    }

    // Shader worker job (any thread but the executor's): everything ensure_shader() and the pipeline
    // builders would do with the WGSL text, so the executor only picks up the results. Touches only
    // the entry (owned by the worker until `ready`), the pack (locked reads) and the device (thread-safe
    // with ImplicitDeviceSynchronization).
    static void materialize_on_worker(const wgpu::Device& device, const std::shared_ptr<WgslPack>& pk, ShaderEntry& e) {
        tl_shader_worker = true;
        std::optional<std::string> hit = pk ? pk->get(e.hash) : std::nullopt;
        if (hit && !hit->empty()) {
            e.code = std::move(*hit);
        } else {
            char buf[32];
            std::snprintf(buf, sizeof buf, "hash:%llx", static_cast<unsigned long long>(e.hash));
            e.code = buf;
            e.worker_miss = true;
        }
        const char* stage_name = stage_name_of(e.stage);
        e.explicit_ep = extract_wgsl_entry_point(e.code, stage_name);
        const uint64_t t0 = now_ns();
        wgpu::ShaderSourceWGSL wgsl{};
        wgsl.code = {e.code.data(), e.code.size()};
        wgpu::ShaderModuleDescriptor desc{};
        desc.nextInChain = &wgsl;
        e.module = device.CreateShaderModule(&desc);
        const uint64_t t1 = now_ns();
        e.worker_ns = t1 - t0;
        e.worker_private_mb = host_private_mb();
        // The scans the pipeline builders need (memoized: entry_point_of / group0_of / ... find them).
        if (*stage_name) {
            e.ep_cache[stage_slot(stage_name)] = resolve_shader_entry_point(e.code, e.explicit_ep, stage_name);
            ++e.worker_scans;
        }
        e.group0_cache = parse_wgsl_group0_bindings(e.code);
        e.groups_cache = parse_wgsl_group_bindings(e.code);
        e.declares_group0_cache = wgsl_declares_group0(e.code);
        e.worker_scans += 3;
        if (e.stage == 1) {
            e.frag_mask_ep = e.ep_cache[1];
            e.frag_mask_cache = wgsl_fragment_output_mask(e.code, e.frag_mask_ep);
            ++e.worker_scans;
        }
        e.worker_scan_ns = now_ns() - t1;
        e.worker_code_bytes = e.code.size();
        // The module and every scan the builders memoize are done: a pack-backed text (~3 MB each, ~2 GB over
        // a std-pack boot) is not kept; code_of() re-reads it from the pack on the rare rescan.
        static const bool keep_text = std::getenv("PG6_KEEP_WGSL") != nullptr;  // diagnosis: the old behaviour
        if (!keep_text && shader_text_droppable(e.hash, e.worker_miss)) {
            std::string().swap(e.code);
            e.code_dropped = true;
        }
        tl_shader_worker = false;
        e.ready.store(true, std::memory_order_release);
    }

    // Live mode: true while `e` is (or has just been handed over to be) materialized on a shader
    // worker, i.e. a pipeline built from it has to wait. False: usable on this thread now (inline
    // WGSL, already materialized, or no workers: ensure_shader() compiles it here as before).
    bool shader_on_worker(const ShaderRef& e) {
        if (!e || !use_shader_workers) return false;
        if (e->pending) {
            e->pending = false;
            e->queued = true;
            ++stats.shader_async_queued;
            shader_workers.submit([dev = device, pk = pack, e] { materialize_on_worker(dev, pk, *e); });
            return true;
        }
        if (!e->queued) return false;
        if (!e->ready.load(std::memory_order_acquire)) return true;
        account_worker_shader(*e);
        return false;
    }

    // v1 pipelines are created synchronously: a shader a v2 pipeline already handed to a worker is
    // waited for (unbudgeted, like the sync compile it replaces).
    void shader_here_now(const ShaderRef& e) {
        if (!e || !e->queued) return;
        if (!shader_ready(e)) {
            const uint64_t t0 = now_ns();
            ++stats.shader_async_waits;
            shader_workers.wait([&] { return shader_ready(e); }, kPipeWaitUnlimited);
            stats.shader_async_wait_ns += now_ns() - t0;
        }
        account_worker_shader(*e);
    }

    static bool shader_ready(const ShaderRef& e) { return !e || !e->queued || e->ready.load(std::memory_order_acquire); }

    // Folds a worker's results into the stats (once, on this thread).
    static constexpr uint64_t kSlowShaderLogNs = 1000ull * 1000000ull;
    uint64_t slow_shader_logs = 0;
    void account_worker_shader(const ShaderEntry& e) {
        if (e.accounted || !e.queued) return;
        e.accounted = true;
        ++stats.shader_materialized;
        ++stats.shader_async;
        stats.shader_async_ns += e.worker_ns;
        if (e.worker_ns > stats.shader_async_max_ns) {
            stats.shader_async_max_ns = e.worker_ns;
            stats.shader_async_max_hash = e.hash;
            stats.shader_async_max_bytes = e.worker_code_bytes;
        }
        stats.shader_async_private_mb_max = std::max(stats.shader_async_private_mb_max, e.worker_private_mb);
        stats.shader_scans += e.worker_scans;
        stats.shader_scan_ns += e.worker_scan_ns;
        if (e.worker_miss) ++stats.shader_pack_misses;
        if (e.worker_ns >= kSlowShaderLogNs && slow_shader_logs < 32) {
            ++slow_shader_logs;
            std::fprintf(stderr, "[native_exec] shader worker: 0x%016llx, %llu WGSL bytes, module %llu ms, host private %llu MB\n",
                         static_cast<unsigned long long>(e.hash), static_cast<unsigned long long>(e.worker_code_bytes),
                         static_cast<unsigned long long>(e.worker_ns / 1000000),
                         static_cast<unsigned long long>(e.worker_private_mb));
        }
    }

    // Device errors raised on a worker thread (the uncaptured-error callback queued them).
    void drain_worker_errors() {
        std::vector<std::string> errs;
        uint64_t count = 0;
        {
            std::lock_guard<std::mutex> lock(worker_error_mu);
            if (!worker_error_count) return;
            count = worker_error_count;
            worker_error_count = 0;
            errs.swap(worker_errors);
        }
        stats.device_errors += count;
        for (std::string& m : errs) {
            if (worker_errors_logged++ < 8) std::fprintf(stderr, "[native_exec] shader worker error: %s\n", m.substr(0, 400).c_str());
            stats.last_device_error = m;
            if (stats.device_error_log.size() < 20) stats.device_error_log.push_back(m);
        }
    }

    // Materializes a pack-backed shader (see ShaderEntry). An entry the pack cannot deliver any more
    // falls back to the hash text, which compiles (and is rejected) like a pack miss.
    void ensure_shader(const ShaderEntry& e) {
        if (!e.pending) return;
        e.pending = false;
        std::optional<std::string> hit = pack ? pack->get(e.hash) : std::nullopt;
        if (hit && !hit->empty()) {
            e.code = std::move(*hit);
        } else {
            char buf[32];
            std::snprintf(buf, sizeof buf, "hash:%llx", static_cast<unsigned long long>(e.hash));
            e.code = buf;
            ++stats.shader_pack_misses;
        }
        const char* stage_name = e.stage == 0 ? "vertex" : e.stage == 1 ? "fragment" : e.stage == 2 ? "compute" : "";
        e.explicit_ep = extract_wgsl_entry_point(e.code, stage_name);
        const uint64_t t0 = now_ns();
        e.module = make_module(e.code);  // WGSL parse + resolve in Dawn/tint, on this thread
        const uint64_t dt = now_ns() - t0;
        ++stats.shader_materialized;
        stats.shader_materialize_ns += dt;
        if (dt > stats.shader_materialize_max_ns) {
            stats.shader_materialize_max_ns = dt;
            stats.shader_materialize_max_hash = e.hash;
            stats.shader_materialize_max_bytes = e.code.size();
        }
        char note[128];
        std::snprintf(note, sizeof note, "shader 0x%016llx, %zu WGSL bytes, module %llu ms, host private %llu MB",
                      static_cast<unsigned long long>(e.hash), e.code.size(),
                      static_cast<unsigned long long>(dt / 1000000),
                      static_cast<unsigned long long>(dt >= kSlowPacketNs ? host_private_mb() : 0));
        slow_note = note;
    }

    // The shader's WGSL text. A pack-backed shader's text is dropped once its module and scans exist
    // (materialize_on_worker); a later rescan re-reads it from the pack into one reusable scratch string.
    std::string code_scratch;
    const std::string& code_of(const ShaderEntry& e) {
        if (!e.code_dropped) return e.code;
        std::optional<std::string> hit = pack ? pack->get(e.hash) : std::nullopt;
        if (hit && !hit->empty()) {
            code_scratch = std::move(*hit);
        } else {
            char buf[32];
            std::snprintf(buf, sizeof buf, "hash:%llx", static_cast<unsigned long long>(e.hash));
            code_scratch = buf;
        }
        ++stats.shader_code_rereads;
        return code_scratch;
    }

    static int stage_slot(std::string_view stage) { return stage == "vertex" ? 0 : stage == "fragment" ? 1 : 2; }

    const std::string& entry_point_of(const ShaderEntry& e, std::string_view stage) {
        std::string& c = e.ep_cache[stage_slot(stage)];
        if (c.empty()) {
            const uint64_t t0 = now_ns();
            c = resolve_shader_entry_point(code_of(e), e.explicit_ep, stage);
            note_scan(now_ns() - t0);
        }
        return c;
    }

    const std::vector<WgslBinding>& group0_of(const ShaderEntry& e) {
        if (!e.group0_cache) {
            const uint64_t t0 = now_ns();
            e.group0_cache = parse_wgsl_group0_bindings(code_of(e));
            note_scan(now_ns() - t0);
        }
        return *e.group0_cache;
    }

    const std::vector<std::vector<WgslBinding>>& groups_of(const ShaderEntry& e) {
        if (!e.groups_cache) {
            const uint64_t t0 = now_ns();
            e.groups_cache = parse_wgsl_group_bindings(e.code);
            note_scan(now_ns() - t0);
        }
        return *e.groups_cache;
    }

    bool declares_group0_of(const ShaderEntry& e) {
        if (!e.declares_group0_cache) {
            const uint64_t t0 = now_ns();
            e.declares_group0_cache = wgsl_declares_group0(code_of(e));
            note_scan(now_ns() - t0);
        }
        return *e.declares_group0_cache;
    }

    std::optional<uint32_t> fragment_outputs_of(const ShaderEntry& e, const std::string& ep) {
        if (!e.frag_mask_cache || e.frag_mask_ep != ep) {
            const uint64_t t0 = now_ns();
            e.frag_mask_ep = ep;
            e.frag_mask_cache = wgsl_fragment_output_mask(code_of(e), ep);
            note_scan(now_ns() - t0);
        }
        return *e.frag_mask_cache;
    }

    void note_scan(uint64_t ns) {
        ++stats.shader_scans;
        stats.shader_scan_ns += ns;
    }

    wgpu::ShaderModule make_module(const std::string& code) {
        wgpu::ShaderSourceWGSL wgsl{};
        wgsl.code = {code.data(), code.size()};
        wgpu::ShaderModuleDescriptor desc{};
        desc.nextInChain = &wgsl;
        return device.CreateShaderModule(&desc);
    }

    // ---- textures (texture_packets.js TexturePackets._create / handleWriteTexture) ----

    void create_texture_v1(const Packet& p) {
        const auto c = read_payload<GpuCreateTexturePayload>(p);
        const uint8_t dimension = c.dimension <= 2 ? static_cast<uint8_t>(c.dimension) : 1;
        const uint32_t w = std::max<uint32_t>(1, c.width);
        const uint32_t h = dimension == 0 ? 1 : std::max<uint32_t>(1, c.height);
        const uint32_t d = dimension == 0 ? 1 : std::max<uint32_t>(1, c.depth_or_layers);
        const uint8_t fmt = v1_format_code(c.format);
        std::vector<uint8_t> views;
        if (uint8_t sib = srgb_sibling(fmt)) views.push_back(sib);
        create_texture(static_cast<uint16_t>(GpuOpcode::CreateTexture), c.id, w, h, d,
                       clamp_mip_levels(c.mip_levels, w, h, d, dimension), 1, c.usage_flags, dimension, fmt, views);
    }

    void create_texture_v2(const Packet& p) {
        const uint16_t op = static_cast<uint16_t>(GpuOpcode::CreateTexture2);
        if (p.payload_bytes < sizeof(GpuCreateTexture2Payload)) {
            packet_error(op, "CreateTexture2: payload too short");
            return;
        }
        const auto c = read_payload<GpuCreateTexture2Payload>(p);
        if (!format_valid(c.format)) {
            packet_error(op, "CreateTexture2 " + std::to_string(c.id) + ": invalid texture format code");
            return;
        }
        if (c.dimension > 2) {
            packet_error(op, "CreateTexture2 " + std::to_string(c.id) + ": invalid dimension");
            return;
        }
        if (c.view_format_count > kGpuMaxTextureViewFormats) {
            packet_error(op, "CreateTexture2 " + std::to_string(c.id) + ": view_format_count too large");
            return;
        }
        std::vector<uint8_t> views;
        for (uint32_t i = 0; i < c.view_format_count; ++i) {
            if (!format_valid(c.view_formats[i])) {
                packet_error(op, "CreateTexture2 " + std::to_string(c.id) + ": invalid view format code");
                return;
            }
            views.push_back(c.view_formats[i]);
        }
        const uint32_t w = std::max<uint32_t>(1, c.width);
        const uint32_t h = c.dimension == 0 ? 1 : std::max<uint32_t>(1, c.height);
        const uint32_t d = c.dimension == 0 ? 1 : std::max<uint32_t>(1, c.depth_or_layers);
        const uint32_t mips = clamp_mip_levels(c.mip_levels, w, h, d, c.dimension);
        const uint32_t samples = clamp_sample_count(c.sample_count, mips, c.dimension, d);
        create_texture(op, c.id, w, h, d, mips, samples, c.usage_flags, c.dimension, c.format, views);
    }

    // TexturePackets._create: sRGB sibling view formats, 3D-BC fallback, 1D -> 2D promotion.
    void create_texture(uint16_t op, uint32_t id, uint32_t w, uint32_t h, uint32_t d, uint32_t mips,
                        uint32_t samples, uint32_t usage, uint8_t dimension, uint8_t format,
                        const std::vector<uint8_t>& requested_views) {
        textures.erase(id);
        pending_clears.erase(id);  // a new texture is zero already
        std::vector<uint8_t> view_formats;
        uint8_t eff_format = format;
        bool bc3d = false;
        if (dimension == 2 && format_is_bc(format) && !features.bc_sliced_3d) {
            bc3d = true;
            eff_format = 22;
            for (uint8_t v : requested_views)
                if (!format_is_bc(v)) view_formats.push_back(v);
        } else {
            view_formats = resolve_compatible_view_formats(format, requested_views);
        }
        bool promoted = false;
        uint32_t eff_h = h, eff_d = d, eff_mips = mips, eff_samples = samples;
        uint8_t eff_dim = dimension;
        if (dimension == 0) {  // promote1DTo2D (default on)
            eff_dim = 1;
            eff_h = 1;
            eff_d = 1;
            eff_mips = 1;
            eff_samples = 1;
            promoted = true;
        }
        wgpu::TextureDescriptor td{};
        // Every texture can be a CopyTexture source (D3D12 copies between any two textures).
        td.usage = static_cast<wgpu::TextureUsage>(usage) | wgpu::TextureUsage::CopySrc;
        td.dimension = eff_dim == 0 ? wgpu::TextureDimension::e1D
                       : eff_dim == 2 ? wgpu::TextureDimension::e3D
                                      : wgpu::TextureDimension::e2D;
        td.size = {w, eff_h, eff_d};
        td.format = wgpu_format(eff_format);
        td.mipLevelCount = eff_mips;
        td.sampleCount = eff_samples;
        std::vector<wgpu::TextureFormat> vf;
        for (uint8_t v : view_formats) vf.push_back(wgpu_format(v));
        td.viewFormatCount = vf.size();
        td.viewFormats = vf.data();
        TextureEntry e;
        e.texture = device.CreateTexture(&td);
        if (!e.texture) {
            packet_error(op, "CreateTexture " + std::to_string(id) + ": createTexture failed");
            return;
        }
        e.default_view = e.texture.CreateView();
        e.geom.width = w;
        e.geom.height = eff_h;
        e.geom.depth_or_layers = eff_d;
        e.geom.mips = eff_mips;
        e.geom.dimension = eff_dim;
        e.geom.requested_1d = dimension == 0;
        e.geom.format = eff_format;
        e.usage = usage;
        e.sample_count = eff_samples;
        e.promoted_1d = promoted;
        e.bc3d_fallback = bc3d;
        textures[id] = std::move(e);
        ++stats.textures_created;
    }

    // WriteTexture (opcode 13): the effective upload of TexturePackets.handleWriteTexture.
    void write_texture(const Packet& p) {
        const auto w = read_payload<GpuWriteTexturePayload>(p);
        if (uint64_t{48} + w.data_len > p.payload_bytes) return;
        auto it = textures.find(w.texture_id);
        if (it == textures.end()) return;
        TextureEntry& t = it->second;
        if (t.bc3d_fallback) return;
        WriteTextureArgs a;
        a.mip = w.mip_level; a.ox = w.origin_x; a.oy = w.origin_y; a.oz = w.origin_z;
        a.width = w.width; a.height = w.height; a.depth = w.depth_or_layers;
        a.bytes_per_row = w.bytes_per_row; a.rows_per_image = w.rows_per_image; a.data_len = w.data_len;
        const WriteTexturePlan plan = plan_write_texture(a, t.geom);
        if (plan.status == WriteTexturePlan::Status::BadMip) {
            packet_error(static_cast<uint16_t>(GpuOpcode::WriteTexture), "WriteTexture: mipLevel out of bounds");
            return;
        }
        if (plan.status == WriteTexturePlan::Status::BadRowBytes) {
            packet_error(static_cast<uint16_t>(GpuOpcode::WriteTexture), "WriteTexture: compressed bytesPerRow too small");
            return;
        }
        if (plan.status != WriteTexturePlan::Status::Write || w.data_len == 0) return;
        {
            // WebGPU requires bytesPerRow >= the row's bytes for a multi-row / multi-slice copy; Dawn
            // rejects such a write whole (nothing is written). Reject it here, counted, instead of
            // handing Dawn an invalid call (found by --gpu-validation: 128-wide writes with a
            // 32-texel pitch on a 128x128 mip chain).
            const FormatInfo& fi = format_info(t.geom.format);
            if (!fi.compressed && fi.bytes > 0 && plan.bytes_per_row > 0 && (plan.height > 1 || plan.depth > 1) &&
                plan.bytes_per_row < uint64_t{plan.width} * fi.bytes) {
                packet_error(static_cast<uint16_t>(GpuOpcode::WriteTexture), "WriteTexture: bytesPerRow smaller than the row");
                return;
            }
        }
        if (!pending_clears.empty()) {
            // The write lands on the queue before the encoder the pending clear goes into.
            const auto pc = pending_clears.find(w.texture_id);
            if (pc != pending_clears.end() &&
                !pc->second.note_write(plan.mip, plan.ox, plan.oy, plan.oz, plan.width, plan.height, plan.depth)) {
                flush_pending_clears();
            }
        }
        wgpu::TexelCopyTextureInfo dst{};
        dst.texture = t.texture;
        dst.mipLevel = plan.mip;
        dst.origin = {plan.ox, plan.oy, plan.oz};
        dst.aspect = wgpu::TextureAspect::All;
        wgpu::TexelCopyBufferLayout layout{};
        layout.offset = 0;
        layout.bytesPerRow = plan.bytes_per_row ? plan.bytes_per_row : wgpu::kCopyStrideUndefined;
        layout.rowsPerImage = plan.rows_per_image ? plan.rows_per_image : wgpu::kCopyStrideUndefined;
        const wgpu::Extent3D size{plan.width, plan.height, plan.depth};
        queue.WriteTexture(&dst, p.payload + 48, w.data_len, &layout, &size);
        ++t.depth_gen;
        ++stats.texture_uploads;
        stats.texture_upload_bytes += w.data_len;
    }

    // ---- ClearTexture (opcode 63, clear_model.h) ----

    static constexpr uint64_t kZeroBufferBytes = 4ull << 20;

    // Records the texture as pending: its zero copies go into the next submitted encoder.
    void clear_texture(uint32_t id) {
        const uint16_t op = static_cast<uint16_t>(kGpuOpcodeClearTexture);
        const auto it = textures.find(id);
        if (it == textures.end()) {
            ++stats.clears_missing;  // never created here (a lazy create): it is created zero later
            return;
        }
        const TextureEntry& t = it->second;
        const FormatInfo& fi = format_info(t.geom.format);
        if (format_is_depth_or_stencil(t.geom.format) || t.sample_count > 1 || fi.bytes == 0 ||
            (t.usage & GpuTextureUsage::CopyDst) == 0) {
            packet_error(op, "ClearTexture " + std::to_string(id) + ": not a single-sample color COPY_DST texture");
            return;
        }
        ClearGeometry g;
        g.width = t.geom.width;
        g.height = t.geom.height;
        g.depth_or_layers = t.geom.depth_or_layers;
        g.mips = t.geom.mips;
        g.dimension = t.geom.dimension == 2 ? 2 : 1;
        g.block_w = fi.compressed ? fi.block_w : 1;
        g.block_h = fi.compressed ? fi.block_h : 1;
        g.block_bytes = fi.bytes;
        pending_clears[id].reset(g);
        ++stats.textures_cleared;
    }

    // Copies from the zero buffer over every still-pending part (into an encoder with no open pass).
    void encode_pending_clears(wgpu::CommandEncoder& enc) {
        if (pending_clears.empty()) return;
        if (!zero_buffer) {
            wgpu::BufferDescriptor bd{};
            bd.size = kZeroBufferBytes;
            bd.usage = wgpu::BufferUsage::CopySrc;
            zero_buffer = device.CreateBuffer(&bd);
        }
        std::vector<ZeroCopy> copies;
        for (const auto& [id, pc] : pending_clears) {
            const auto it = textures.find(id);
            if (it == textures.end()) continue;
            if (!pc.pending()) {
                ++stats.clears_covered;  // every texel was written since the clear
                continue;
            }
            copies.clear();
            pc.zero_copies(kZeroBufferBytes, &copies);
            for (const ZeroCopy& c : copies) {
                wgpu::TexelCopyBufferInfo src{};
                src.buffer = zero_buffer;
                src.layout.offset = 0;
                src.layout.bytesPerRow = c.bytes_per_row;
                src.layout.rowsPerImage = c.rows_per_image;
                wgpu::TexelCopyTextureInfo dst{};
                dst.texture = it->second.texture;
                dst.mipLevel = c.mip;
                dst.origin = {0, c.y, c.z};
                dst.aspect = wgpu::TextureAspect::All;
                const wgpu::Extent3D extent{c.width, c.height, c.depth};
                enc.CopyBufferToTexture(&src, &dst, &extent);
                ++stats.clear_copies;
                stats.clear_bytes += c.bytes;
            }
            ++it->second.depth_gen;
        }
        pending_clears.clear();
    }

    // A write needs a pending clear done first: submit the pending clears on their own.
    void flush_pending_clears() {
        wgpu::CommandEncoder enc = device.CreateCommandEncoder();
        encode_pending_clears(enc);
        wgpu::CommandBuffer cb = enc.Finish();
        queue.Submit(1, &cb);
        ++stats.clear_flushes;
    }

    // ---- depth twins ----
    //
    // A depth texture bound where the layout entry's sample type cannot take a depth view is served by
    // a twin: a same-extent color texture refreshed from the depth plane by a fullscreen pass (depth
    // textures cannot be copied to color formats). Kinds (layout_model.h DepthRead):
    //   F32 r32float, F16 r16float (device without float32-filterable: the only filterable float
    //   format), U32 r32uint / S32 r32sint (the raw stored value, see depth_twin_wgsl).

    static wgpu::TextureFormat twin_format(DepthRead k) {
        switch (k) {
            case DepthRead::TwinF16: return wgpu::TextureFormat::R16Float;
            case DepthRead::TwinU32: return wgpu::TextureFormat::R32Uint;
            case DepthRead::TwinS32: return wgpu::TextureFormat::R32Sint;
            default: return wgpu::TextureFormat::R32Float;
        }
    }

    // The twin of kind `k` of a depth texture (created on first need): same extent/mips/layers.
    wgpu::Texture depth_twin(uint32_t id, TextureEntry& t, DepthRead k) {
        TwinSlot& slot = t.twins[static_cast<size_t>(k)];
        if (slot.tex) return slot.tex;
        wgpu::TextureDescriptor td{};
        td.dimension = wgpu::TextureDimension::e2D;
        td.size = {t.geom.width, t.geom.height, t.geom.depth_or_layers};
        td.format = twin_format(k);
        td.mipLevelCount = t.geom.mips;
        td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment;
        slot.tex = device.CreateTexture(&td);
        if (slot.tex) {
            depth_twin_sources.insert(id);
            slot.gen = 0;
            ++stats.depth_twins_created;
            if (k == DepthRead::TwinU32 || k == DepthRead::TwinS32) ++stats.depth_int_twins_created;
            if (k == DepthRead::TwinF16) ++stats.depth_half_twins_created;
        }
        return slot.tex;
    }

    wgpu::TextureView depth_twin_view(TextureEntry& t, DepthRead k, const GpuTextureViewDesc& d) {
        wgpu::TextureViewDescriptor vd{};
        vd.format = twin_format(k);
        if (d.dimension) vd.dimension = wgpu_view_dim(d.dimension);
        vd.baseMipLevel = d.base_mip_level;
        if (d.mip_level_count) vd.mipLevelCount = d.mip_level_count;
        vd.baseArrayLayer = d.base_array_layer;
        if (d.array_layer_count) vd.arrayLayerCount = d.array_layer_count;
        return t.twins[static_cast<size_t>(k)].tex.CreateView(&vd);
    }

    wgpu::RenderPipeline depth_copy_pipeline_for(DepthRead k, uint8_t src_format) {
        const uint32_t key = static_cast<uint32_t>(k) * 256u + src_format;
        auto it = depth_copy_pipelines.find(key);
        if (it != depth_copy_pipelines.end()) return it->second;
        const std::string code = depth_twin_wgsl(k, src_format);
        wgpu::ShaderSourceWGSL wgsl{};
        wgsl.code = code.c_str();
        wgpu::ShaderModuleDescriptor md{};
        md.nextInChain = &wgsl;
        wgpu::ShaderModule mod = device.CreateShaderModule(&md);
        wgpu::ColorTargetState ct{};
        ct.format = twin_format(k);
        wgpu::FragmentState fs{};
        fs.module = mod;
        fs.entryPoint = "fs";
        fs.targetCount = 1;
        fs.targets = &ct;
        wgpu::RenderPipelineDescriptor pd{};
        pd.vertex.module = mod;
        pd.vertex.entryPoint = "vs";
        pd.fragment = &fs;
        wgpu::RenderPipeline p = device.CreateRenderPipeline(&pd);
        depth_copy_pipelines[key] = p;
        return p;
    }

    // Before every pass begins (no pass open): re-copy the depth plane of each stale twin, one
    // fullscreen pass per mip/layer, so the pass about to run reads the depth written so far.
    void refresh_depth_twins(wgpu::CommandEncoder& enc) {
        for (auto sit = depth_twin_sources.begin(); sit != depth_twin_sources.end();) {
            auto tit = textures.find(*sit);
            if (tit == textures.end()) { sit = depth_twin_sources.erase(sit); continue; }
            ++sit;
            TextureEntry& t = tit->second;
            for (size_t ki = 0; ki < t.twins.size(); ++ki) {
                TwinSlot& slot = t.twins[ki];
                if (!slot.tex || slot.gen == t.depth_gen) continue;
                const DepthRead k = static_cast<DepthRead>(ki);
                wgpu::RenderPipeline pipe = depth_copy_pipeline_for(k, t.geom.format);
                if (!pipe) continue;
                const wgpu::BindGroupLayout bgl = pipe.GetBindGroupLayout(0);
                for (uint32_t m = 0; m < t.geom.mips; ++m) {
                    for (uint32_t l = 0; l < t.geom.depth_or_layers; ++l) {
                        wgpu::TextureViewDescriptor sv{};
                        sv.dimension = wgpu::TextureViewDimension::e2D;
                        sv.aspect = wgpu::TextureAspect::DepthOnly;
                        sv.baseMipLevel = m; sv.mipLevelCount = 1;
                        sv.baseArrayLayer = l; sv.arrayLayerCount = 1;
                        wgpu::TextureViewDescriptor dv = sv;
                        dv.format = twin_format(k);
                        dv.aspect = wgpu::TextureAspect::All;
                        wgpu::BindGroupEntry be{};
                        be.binding = 0;
                        be.textureView = t.texture.CreateView(&sv);
                        wgpu::BindGroupDescriptor bd{};
                        bd.layout = bgl;
                        bd.entryCount = 1;
                        bd.entries = &be;
                        wgpu::RenderPassColorAttachment ca{};
                        ca.view = slot.tex.CreateView(&dv);
                        ca.loadOp = wgpu::LoadOp::Clear;
                        ca.storeOp = wgpu::StoreOp::Store;
                        wgpu::RenderPassDescriptor rp{};
                        rp.colorAttachmentCount = 1;
                        rp.colorAttachments = &ca;
                        wgpu::RenderPassEncoder pass = enc.BeginRenderPass(&rp);
                        pass.SetPipeline(pipe);
                        pass.SetBindGroup(0, device.CreateBindGroup(&bd));
                        pass.Draw(3);
                        pass.End();
                    }
                }
                slot.gen = t.depth_gen;
                ++stats.depth_float_copies;
            }
        }
    }

    // TexturePackets / RenderStatePackets.getOrCreateTextureView (W347 1D promotion, 3D-BC fallback).
    wgpu::TextureView texture_view(TextureEntry& t, const GpuTextureViewDesc* d) {
        uint8_t fmt = d ? d->format : 0, dim = d ? d->dimension : 0, aspect = d ? d->aspect : 0;
        uint32_t base_mip = d ? d->base_mip_level : 0, mip_count = d ? d->mip_level_count : 0;
        uint32_t base_layer = d ? d->base_array_layer : 0, layer_count = d ? d->array_layer_count : 0;
        if (fmt && t.bc3d_fallback && format_is_bc(fmt)) fmt = t.geom.format;
        if (dim == 1 && (t.geom.dimension == 1 || t.promoted_1d)) {  // explicit 1d view of a 2d / promoted texture
            dim = 2;
            base_layer = 0;
            layer_count = 1;
            if (t.promoted_1d || t.geom.mips == 1) {
                base_mip = 0;
                mip_count = 1;
            }
        }
        if (!fmt && !dim && !aspect && !base_mip && !mip_count && !base_layer && !layer_count) return t.default_view;
        const uint64_t key = uint64_t{fmt} | (uint64_t{dim} << 8) | (uint64_t{aspect} << 16) |
                             (uint64_t{base_mip & 0xffff} << 24) | (uint64_t{mip_count & 0xffff} << 40) |
                             (uint64_t{base_layer & 0xff} << 56);
        const uint64_t key2 = key ^ (uint64_t{layer_count} * 0x9E3779B97F4A7C15ull) ^
                              (uint64_t{base_layer >> 8} * 0xC2B2AE3D27D4EB4Full);
        auto vit = t.views.find(key2);
        if (vit != t.views.end()) return vit->second;
        wgpu::TextureViewDescriptor vd{};
        if (fmt) vd.format = wgpu_format(fmt);
        if (dim) vd.dimension = wgpu_view_dim(dim);
        vd.aspect = aspect == 1 ? wgpu::TextureAspect::DepthOnly
                    : aspect == 2 ? wgpu::TextureAspect::StencilOnly
                                  : wgpu::TextureAspect::Undefined;
        vd.baseMipLevel = base_mip;
        if (mip_count) vd.mipLevelCount = mip_count;
        vd.baseArrayLayer = base_layer;
        if (layer_count) vd.arrayLayerCount = layer_count;
        // An aspect view's format must be that aspect's own format (WebGPU "Texture view format
        // compatibility"): depth-only of a depth+stencil texture is its depth format, stencil-only is stencil8.
        if (vd.aspect == wgpu::TextureAspect::DepthOnly) {
            if (vd.format == wgpu::TextureFormat::Depth32FloatStencil8) vd.format = wgpu::TextureFormat::Depth32Float;
            else if (vd.format == wgpu::TextureFormat::Depth24PlusStencil8) vd.format = wgpu::TextureFormat::Depth24Plus;
        } else if (vd.aspect == wgpu::TextureAspect::StencilOnly) {
            if (vd.format == wgpu::TextureFormat::Depth32FloatStencil8 || vd.format == wgpu::TextureFormat::Depth24PlusStencil8)
                vd.format = wgpu::TextureFormat::Stencil8;
        }
        wgpu::TextureView v = t.texture.CreateView(&vd);
        if (t.views.size() >= 64) t.views.erase(t.views.begin());
        t.views[key2] = v;
        return v;
    }

    // ---- samplers ----

    void create_sampler_v1(const Packet& p) {  // _handleCreateSampler
        const auto c = read_payload<GpuCreateSamplerPayload>(p);
        wgpu::SamplerDescriptor sd{};
        sd.addressModeU = wgpu_address(c.address_u);
        sd.addressModeV = wgpu_address(c.address_v);
        sd.addressModeW = wgpu_address(c.address_w);
        sd.magFilter = c.mag_filter == 1 ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
        sd.minFilter = c.min_filter == 1 ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
        sd.mipmapFilter = c.mipmap_filter == 1 ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest;
        sd.maxAnisotropy = static_cast<uint16_t>(std::min<uint32_t>(std::max<uint32_t>(1, c.max_anisotropy), 65535));
        samplers[c.id] = device.CreateSampler(&sd);
        comparison_samplers.erase(c.id);
        ++stats.samplers_created;
    }

    void create_sampler_v2(const Packet& p) {  // RenderStatePackets._createSampler / decodeSampler2Packet
        const auto c = read_payload<GpuCreateSampler2Payload>(p);
        wgpu::SamplerDescriptor sd{};
        sd.addressModeU = wgpu_address(c.address_u);
        sd.addressModeV = wgpu_address(c.address_v);
        sd.addressModeW = wgpu_address(c.address_w);
        sd.magFilter = c.mag_filter == 1 ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
        sd.minFilter = c.min_filter == 1 ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
        sd.mipmapFilter = c.mipmap_filter == 1 ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest;
        sd.lodMinClamp = c.lod_min_clamp;
        sd.lodMaxClamp = c.lod_max_clamp;
        sd.maxAnisotropy = std::max<uint16_t>(1, c.max_anisotropy);
        if (c.compare != 0) sd.compare = wgpu_compare(c.compare);
        samplers[c.id] = device.CreateSampler(&sd);
        if (c.compare != 0) comparison_samplers.insert(c.id);
        else comparison_samplers.erase(c.id);
        ++stats.samplers_created;
    }

    // ---- layouts (RenderStatePackets._createBindGroupLayout / _createPipelineLayout) ----

    wgpu::BindGroupLayout empty_bgl() {
        if (!empty_layout) {
            wgpu::BindGroupLayoutDescriptor d{};
            empty_layout = device.CreateBindGroupLayout(&d);
        }
        return empty_layout;
    }

    // The layout object a manifest id resolves to (one per id, shared by every reference).
    bool manifest_bgl(const std::string& hex, const std::vector<reflect::LayoutEntryDesc>& entries, wgpu::BindGroupLayout* out) {
        auto it = manifest_bgls.find(hex);
        if (it != manifest_bgls.end()) {
            *out = it->second;
            return true;
        }
        if (!make_bgl_from_entries(entries, out)) return false;
        manifest_bgls[hex] = *out;
        return true;
    }

    // decodeBindGroupLayoutEntry -> the WebGPU entry (create_bind_group_layout, fitted group 0).
    static void to_wgpu_layout_entry(const LayoutEntry& le, wgpu::BindGroupLayoutEntry* out) {
        wgpu::BindGroupLayoutEntry& w = *out;
        w.binding = le.binding;
        w.visibility = static_cast<wgpu::ShaderStage>(le.visibility);
        if (le.kind == 0) {
            w.buffer.type = le.subtype == 0   ? wgpu::BufferBindingType::Uniform
                            : le.subtype == 1 ? wgpu::BufferBindingType::Storage
                                              : wgpu::BufferBindingType::ReadOnlyStorage;
            w.buffer.hasDynamicOffset = le.dynamic;
            w.buffer.minBindingSize = le.min_binding_size;
        } else if (le.kind == 1) {
            w.sampler.type = le.subtype == 0   ? wgpu::SamplerBindingType::Filtering
                             : le.subtype == 1 ? wgpu::SamplerBindingType::NonFiltering
                                               : wgpu::SamplerBindingType::Comparison;
        } else if (le.kind == 2) {
            static const wgpu::TextureSampleType kS[5] = {
                wgpu::TextureSampleType::Float, wgpu::TextureSampleType::UnfilterableFloat,
                wgpu::TextureSampleType::Depth, wgpu::TextureSampleType::Sint, wgpu::TextureSampleType::Uint};
            w.texture.sampleType = kS[le.subtype];
            // 'float' + multisampled is invalid in WebGPU (a Texture2DMS<float4> SRV): non-filterable
            if (le.multisampled && w.texture.sampleType == wgpu::TextureSampleType::Float)
                w.texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
            w.texture.viewDimension = wgpu_view_dim(le.view_dim);
            w.texture.multisampled = le.multisampled;
        } else {
            w.storageTexture.access = le.subtype == 0   ? wgpu::StorageTextureAccess::WriteOnly
                                      : le.subtype == 1 ? wgpu::StorageTextureAccess::ReadOnly
                                                        : wgpu::StorageTextureAccess::ReadWrite;
            w.storageTexture.format = wgpu_format(le.storage_format);
            w.storageTexture.viewDimension = wgpu_view_dim(le.view_dim);
        }
    }

    bool create_bind_group_layout(const Packet& p) {
        if (p.payload_bytes < sizeof(GpuCreateBindGroupLayoutPayload)) return false;
        const auto h = read_payload<GpuCreateBindGroupLayoutPayload>(p);
        if (h.flags & GpuLayoutFlags::ManifestReference) {
            // `resolveManifestLayout('bindGroupLayout', contentId)`: the loaded layout-manifest.json's
            // bind_group_layouts[contentId] (nothing loaded: unavailable, as in the replay page).
            const auto* entries = reflection.manifest_bind_group_layout(h.content_id);
            if (!entries) {
                ++stats.manifest_layouts_unavailable;
                return fail("manifest bind group layout unavailable");
            }
            wgpu::BindGroupLayout layout;
            if (!manifest_bgl(reflect::format_hash(h.content_id), *entries, &layout))
                return fail("manifest bind group layout unavailable");
            BglEntry b;
            b.layout = layout;
            b.entries_known = false;  // `entries: null`
            bgls[h.id] = std::move(b);
            ++stats.bind_group_layouts;
            ++stats.manifest_layouts_resolved;
            return true;
        }
        if (uint64_t{24} + uint64_t{h.entry_count} * 24 > p.payload_bytes) return false;
        std::vector<LayoutEntry> entries;
        std::vector<bool> seen_dummy;
        for (uint32_t i = 0; i < h.entry_count; ++i) {
            const auto e = read_payload<GpuBindGroupLayoutEntry>(p, 24 + size_t{i} * 24);
            LayoutEntry le;
            le.binding = e.binding;
            le.visibility = e.visibility & 0x7;
            le.kind = e.type;
            le.subtype = e.binding_type;
            le.dynamic = (e.flags & 0x1) != 0;
            le.multisampled = (e.flags & 0x2) != 0;
            le.view_dim = e.view_dimension != 0 ? e.view_dimension : 2;
            le.storage_format = e.storage_format;
            le.min_binding_size = e.min_binding_size;
            if (le.kind == 0) {
                if (le.subtype > 2) return false;
                if (le.subtype == 1 && (le.visibility & 0x1)) le.visibility &= ~0x1u;
            } else if (le.kind == 1) {
                if (le.subtype > 2) return false;
            } else if (le.kind == 2) {
                if (le.subtype > 4 || le.view_dim > 6) return false;
            } else if (le.kind == 3) {
                if (le.subtype > 2 || !format_valid(le.storage_format) || le.view_dim > 6) return false;
                if ((le.subtype == 0 || le.subtype == 2) && (le.visibility & 0x1)) le.visibility &= ~0x1u;
            } else {
                return false;
            }
            for (const LayoutEntry& prev : entries)
                if (prev.binding == le.binding) return fail("bind group layout: duplicate binding");
            if (le.kind == 3) {
                const StorageResolution sr = resolve_storage_format(le.storage_format, features);
                le.storage_format = sr.supported && sr.storage_format ? sr.storage_format : 22;
                if (le.view_dim == 4 || le.view_dim == 5) le.view_dim = 3;
            }
            entries.push_back(le);
        }
        wgpu::BindGroupLayout layout;
        const uint64_t content = h.content_id;
        auto cit = content ? layouts_by_content.find(content) : layouts_by_content.end();
        if (cit != layouts_by_content.end()) {
            layout = cit->second;
        } else {
            std::vector<wgpu::BindGroupLayoutEntry> wes(entries.size());
            for (size_t i = 0; i < entries.size(); ++i) to_wgpu_layout_entry(entries[i], &wes[i]);
            wgpu::BindGroupLayoutDescriptor d{};
            d.entryCount = wes.size();
            d.entries = wes.data();
            layout = device.CreateBindGroupLayout(&d);
            if (content) layouts_by_content[content] = layout;
        }
        BglEntry b;
        b.layout = layout;
        {
            std::vector<LimitEntry> le;
            for (const LayoutEntry& x : entries) le.push_back({x.visibility, x.kind, x.subtype});
            b.over_limits = !stage_limit_excess(le, stage_limits).empty();
            if (b.over_limits) ++stats.over_limit_bind_group_layouts;
        }
        b.entries = std::move(entries);
        bgls[h.id] = std::move(b);
        ++stats.bind_group_layouts;
        return true;
    }

    bool create_pipeline_layout(const Packet& p) {
        if (p.payload_bytes < sizeof(GpuCreatePipelineLayoutPayload)) return false;
        const auto h = read_payload<GpuCreatePipelineLayoutPayload>(p);
        if (h.flags & GpuLayoutFlags::ManifestReference) {
            // `resolveManifestLayout('pipelineLayout', contentId)`: pipeline_layouts[contentId] of the
            // loaded manifest; its meta is the worker's default (no group 0 entries).
            const auto* ids = reflection.manifest_pipeline_layout(h.content_id);
            if (!ids) {
                ++stats.manifest_layouts_unavailable;
                return fail("manifest pipeline layout unavailable");
            }
            std::vector<wgpu::BindGroupLayout> layouts;
            for (const std::string& hex : *ids) {
                wgpu::BindGroupLayout l;
                const auto* entries = reflection.manifest_bind_group_layout_hex(hex);
                if (!entries || !manifest_bgl(hex, *entries, &l)) return fail("manifest pipeline layout unavailable");
                layouts.push_back(l);
            }
            wgpu::PipelineLayoutDescriptor d{};
            d.bindGroupLayoutCount = layouts.size();
            d.bindGroupLayouts = layouts.data();
            PlEntry e;
            e.layout = device.CreatePipelineLayout(&d);
            pipeline_layouts[h.id] = std::move(e);
            ++stats.pipeline_layouts;
            ++stats.manifest_layouts_resolved;
            return true;
        }
        if (uint64_t{24} + uint64_t{h.bind_group_layout_count} * 4 > p.payload_bytes) return false;
        std::vector<wgpu::BindGroupLayout> layouts;
        std::vector<FitExplicitGroup> fit_groups;
        std::vector<std::vector<LayoutEntry>> entry_groups;
        const BglEntry* group0 = nullptr;
        for (uint32_t g = 0; g < h.bind_group_layout_count; ++g) {
            const uint32_t bgl_id = read_payload<uint32_t>(p, 24 + size_t{g} * 4);
            const BglEntry* rec = nullptr;
            if (bgl_id != 0) {
                auto it = bgls.find(bgl_id);
                if (it == bgls.end()) return fail("pipeline layout: unknown bind group layout " + std::to_string(bgl_id));
                rec = &it->second;
            }
            if (g == 0) group0 = rec;
            layouts.push_back(rec ? rec->layout : empty_bgl());
            FitExplicitGroup fg;
            std::vector<LayoutEntry> eg;
            if (!rec || rec->entries_known) {
                fg.entries.emplace();
                if (rec) {
                    for (const LayoutEntry& le : rec->entries) fg.entries->push_back({le.binding, le.kind, le.subtype});
                    eg = rec->entries;
                }
            }
            fit_groups.push_back(std::move(fg));
            entry_groups.push_back(std::move(eg));
        }
        wgpu::PipelineLayoutDescriptor d{};
        d.bindGroupLayoutCount = layouts.size();
        d.bindGroupLayouts = layouts.data();
        PlEntry e;
        e.layout = device.CreatePipelineLayout(&d);
        e.bgls = layouts;
        e.fit_groups = std::move(fit_groups);
        e.entry_groups = std::move(entry_groups);
        e.group0_has_entries = group0 && (!group0->entries_known || !group0->entries.empty());
        if (group0 && group0->entries_known) {
            std::vector<Group0LayoutEntry> g0;
            for (const LayoutEntry& le : group0->entries) g0.push_back({le.binding, le.kind, le.subtype});
            e.group0_entries = std::move(g0);
        }
        pipeline_layouts[h.id] = std::move(e);
        ++stats.pipeline_layouts;
        return true;
    }

    // ---- pipelines ----

    bool create_render_pipeline2(const Packet& p) {
        const auto h = read_payload<GpuCreateRenderPipeline2Payload>(p);
        const PlEntry* pl = nullptr;
        if (h.pipeline_layout_id != 0) {
            auto lit = pipeline_layouts.find(h.pipeline_layout_id);
            if (lit == pipeline_layouts.end()) return fail("unknown pipeline layout " + std::to_string(h.pipeline_layout_id));
            pl = &lit->second;
        }
        auto vit = shaders.find(h.vs_module_id);
        if (vit == shaders.end()) return fail("unknown vertex shader module " + std::to_string(h.vs_module_id));
        ShaderRef vs = vit->second, fs;
        if (h.fs_module_id != 0) {
            auto fit = shaders.find(h.fs_module_id);
            if (fit == shaders.end()) return fail("unknown fragment shader module " + std::to_string(h.fs_module_id));
            fs = fit->second;
        }
        deferred_render.erase(h.id);
        const bool vs_away = shader_on_worker(vs);
        const bool fs_away = shader_on_worker(fs);
        if (vs_away || fs_away) {
            // Built once both shaders are ready (pump_deferred / a draw that needs it within budget).
            RenderPipelineEntry e;
            e.pending = std::make_shared<PendingPipe>();
            e.deferred = true;
            e.deferred_id = h.id;
            e.fullscreen = std::min<uint32_t>(h.key.vertex_buffer_count, 8) == 0;
            e.vs_hash = vs->hash;
            e.fs_hash = fs ? fs->hash : 0;
            e.auto_meta.gen = ++pipeline_gen_counter;
            e.perf_hash = fs && fs->hash ? fs->hash : vs->hash;
            e.quarantined_hash = quarantined_stage(fs ? fs->hash : 0, vs->hash);
            render_pipelines[h.id] = std::move(e);
            defer_pipeline(deferred_render, h.id, p, pl, std::move(vs), std::move(fs));
            return true;
        }
        return build_render_pipeline2(h, pl, *vs, fs.get());
    }

    void defer_pipeline(std::unordered_map<uint32_t, DeferredPipe>& m, uint32_t id, const Packet& p, const PlEntry* pl,
                        ShaderRef a, ShaderRef b) {
        DeferredPipe d;
        d.opcode = p.opcode;
        d.payload.assign(p.payload, p.payload + p.payload_bytes);
        if (pl) d.layout = *pl;
        d.a = std::move(a);
        d.b = std::move(b);
        m[id] = std::move(d);
        ++stats.pipelines_deferred;
        stats.pipelines_deferred_max = std::max<uint64_t>(stats.pipelines_deferred_max, deferred_render.size() + deferred_compute.size());
    }

    bool build_render_pipeline2(const GpuCreateRenderPipeline2Payload& h, const PlEntry* pl, const ShaderEntry& vs,
                                const ShaderEntry* fs) {
        const GpuRenderPipelineKey& key = h.key;
        wgpu::PipelineLayout layout;
        bool auto_layout = pl == nullptr;
        bool group0_has_entries = false;
        const std::optional<std::vector<Group0LayoutEntry>>* group0_entries = nullptr;
        if (!auto_layout) {
            layout = pl->layout;
            group0_has_entries = pl->group0_has_entries;
            group0_entries = &pl->group0_entries;
        } else {
            ++stats.auto_layout_pipelines;
        }
        ensure_shader(vs);

        // decodeRenderPipelineKey + validateVertexBufferLayouts
        const uint32_t target_count = std::min<uint32_t>(key.color_target_count, 8);
        const uint32_t buffer_count = std::min<uint32_t>(key.vertex_buffer_count, 8);
        const uint32_t attr_count = std::min<uint32_t>(key.vertex_attribute_count, 16);
        std::vector<wgpu::VertexAttribute> attrs[8];
        bool used_location[16] = {};
        uint32_t attached = 0;
        for (uint32_t i = 0; i < attr_count; ++i) {
            const GpuVertexAttributeKey& a = key.vertex_attributes[i];
            if (a.buffer_index >= buffer_count) continue;
            if (key.vertex_buffers[a.buffer_index].step_mode == 0) continue;
            uint32_t bytes = 0;
            wgpu::VertexAttribute va{};
            va.format = wgpu_vertex_format(a.format, &bytes);
            if (!bytes) return fail("render pipeline vertex layout invalid: bad vertex format");
            if (a.shader_location >= 16) return fail("render pipeline vertex layout invalid: shaderLocation");
            if (used_location[a.shader_location])
                return fail("render pipeline vertex layout invalid: duplicate shaderLocation");
            used_location[a.shader_location] = true;
            const uint32_t stride = key.vertex_buffers[a.buffer_index].array_stride;
            if (stride > 0 && a.offset + bytes > stride)
                return fail("render pipeline vertex layout invalid: attribute exceeds arrayStride");
            va.offset = a.offset;
            va.shaderLocation = a.shader_location;
            attrs[a.buffer_index].push_back(va);
            ++attached;
        }
        std::vector<wgpu::VertexBufferLayout> vbs(buffer_count);
        for (uint32_t i = 0; i < buffer_count; ++i) {
            const GpuVertexBufferKey& b = key.vertex_buffers[i];
            if (b.step_mode == 0) continue;  // null slot
            if (b.step_mode > 2) return fail("render pipeline vertex layout invalid: stepMode");
            if (b.array_stride % 4 != 0) return fail("render pipeline vertex layout invalid: arrayStride");
            vbs[i].stepMode = b.step_mode == 1 ? wgpu::VertexStepMode::Vertex : wgpu::VertexStepMode::Instance;
            vbs[i].arrayStride = b.array_stride;
            vbs[i].attributeCount = attrs[i].size();
            vbs[i].attributes = attrs[i].data();
        }
        if (attached != attr_count) return fail("render pipeline vertex layout dropped attributes");

        std::vector<wgpu::ColorTargetState> targets(target_count);
        std::vector<wgpu::BlendState> blends(target_count);
        for (uint32_t i = 0; i < target_count; ++i) {
            const GpuColorTargetKey& t = key.color_targets[i];
            if (t.format == 0) continue;  // null target
            if (!format_valid(t.format)) return fail("render pipeline target: invalid texture format");
            targets[i].format = wgpu_format(t.format);
            targets[i].writeMask = static_cast<wgpu::ColorWriteMask>(t.write_mask & 0x0f);
        }

        wgpu::RenderPipelineDescriptor rd{};
        wgpu::FragmentState fragment{};
        if (fs) {
            ensure_shader(*fs);
            for (uint32_t i = 0; i < target_count; ++i) {
                const GpuColorTargetKey& t = key.color_targets[i];
                if (t.format == 0 || !t.blend_enabled) continue;
                if (!blendable(t.format, features)) continue;  // sanitizeColorTargetBlendState strips the blend
                blends[i].color.operation = wgpu_blend_op(t.color.operation);
                blends[i].color.srcFactor = wgpu_blend_factor(t.color.src_factor);
                blends[i].color.dstFactor = wgpu_blend_dst_factor(t.color.dst_factor);
                blends[i].alpha.operation = wgpu_blend_op(t.alpha.operation);
                blends[i].alpha.srcFactor = wgpu_blend_factor(t.alpha.src_factor);
                blends[i].alpha.dstFactor = wgpu_blend_dst_factor(t.alpha.dst_factor);
                targets[i].blend = &blends[i];
            }
            fragment.module = fs->module;
            const std::string& ep = entry_point_of(*fs, "fragment");
            // A target the fragment stage never writes must have writeMask 0 (Dawn validation).
            if (const auto outs = fragment_outputs_of(*fs, ep)) {
                for (uint32_t i = 0; i < target_count && i < 32; ++i) {
                    if (targets[i].format == wgpu::TextureFormat::Undefined || (*outs >> i & 1u)) continue;
                    if (targets[i].writeMask != wgpu::ColorWriteMask::None) ++stats.write_masks_cleared;
                    targets[i].writeMask = wgpu::ColorWriteMask::None;
                }
            }
            fragment.entryPoint = {ep.data(), ep.size()};
            fragment.targetCount = targets.size();
            fragment.targets = targets.data();
            rd.fragment = &fragment;
            fragment_ep_storage = ep;
            fragment.entryPoint = {fragment_ep_storage.data(), fragment_ep_storage.size()};
        } else if (target_count > 0) {
            if (!empty_fragment) empty_fragment = make_module("@fragment fn fs_main() {}\n");
            for (uint32_t i = 0; i < target_count; ++i) targets[i].writeMask = wgpu::ColorWriteMask::None;
            fragment.module = empty_fragment;
            fragment_ep_storage = "fs_main";
            fragment.entryPoint = {fragment_ep_storage.data(), fragment_ep_storage.size()};
            fragment.targetCount = targets.size();
            fragment.targets = targets.data();
            rd.fragment = &fragment;
        }

        wgpu::DepthStencilState ds{};
        const GpuDepthStencilKey& dk = key.depth_stencil;
        if (dk.format != 0 && format_valid(dk.format)) {
            const bool has_depth = format_is_depth(dk.format);
            ds.format = wgpu_format(dk.format);
            ds.depthWriteEnabled = has_depth && dk.depth_write_enabled ? wgpu::OptionalBool::True : wgpu::OptionalBool::False;
            auto face = [](const GpuStencilFaceKey& f) {
                wgpu::StencilFaceState s{};
                s.compare = f.compare != 0 && f.compare < 9 ? wgpu_compare(f.compare) : wgpu::CompareFunction::Always;
                s.failOp = wgpu_stencil_op(f.fail_op);
                s.depthFailOp = wgpu_stencil_op(f.depth_fail_op);
                s.passOp = wgpu_stencil_op(f.pass_op);
                return s;
            };
            ds.stencilFront = face(dk.stencil_front);
            ds.stencilBack = face(dk.stencil_back);
            ds.stencilReadMask = dk.stencil_read_mask;
            ds.stencilWriteMask = dk.stencil_write_mask;
            ds.depthBias = dk.depth_bias;
            float slope, clamp;
            std::memcpy(&slope, &dk.depth_bias_slope_scale_bits, 4);
            std::memcpy(&clamp, &dk.depth_bias_clamp_bits, 4);
            ds.depthBiasSlopeScale = std::isfinite(slope) ? slope : 0.f;
            ds.depthBiasClamp = std::isfinite(clamp) ? clamp : 0.f;
            const wgpu::CompareFunction cmp = wgpu_compare(dk.depth_compare);
            if (has_depth) ds.depthCompare = cmp != wgpu::CompareFunction::Undefined ? cmp : wgpu::CompareFunction::Always;
            else if (cmp == wgpu::CompareFunction::Always) ds.depthCompare = wgpu::CompareFunction::Always;
            rd.depthStencil = &ds;
        }

        const GpuPrimitiveKey& pk = key.primitive;
        rd.primitive.topology = wgpu_topology(pk.topology);
        rd.primitive.frontFace = pk.front_face == 1 ? wgpu::FrontFace::CW : wgpu::FrontFace::CCW;
        rd.primitive.cullMode = pk.cull_mode == 1 ? wgpu::CullMode::Front : pk.cull_mode == 2 ? wgpu::CullMode::Back
                                                                                               : wgpu::CullMode::None;
        if (pk.strip_index_format != 0 && (pk.topology == 1 || pk.topology == 3))
            rd.primitive.stripIndexFormat = pk.strip_index_format == 1   ? wgpu::IndexFormat::Uint16
                                            : pk.strip_index_format == 2 ? wgpu::IndexFormat::Uint32
                                                                         : wgpu::IndexFormat::Undefined;
        // GPUPrimitiveState.unclippedDepth needs the depth-clip-control feature, which the replay
        // page does not request: the worker's pipeline fails validation, and so does Dawn's.
        if (pk.unclipped_depth) rd.primitive.unclippedDepth = true;
        rd.multisample.count = key.multisample.count ? key.multisample.count : 1;
        rd.multisample.mask = key.multisample.mask;
        rd.multisample.alphaToCoverageEnabled = key.multisample.alpha_to_coverage_enabled != 0;

        // An explicit layout that does not satisfy the statically referenced @group(0) bindings of the
        // WGSL becomes layout:'auto' with the worker's bind group remap (isExplicitLayoutCompatibleWithWgsl).
        AutoLayoutMeta auto_meta;
        if (!auto_layout && group0_entries) {
            std::vector<WgslBinding> wb =
                merge_wgsl_group0_bindings(group0_of(vs), fs ? group0_of(*fs) : std::vector<WgslBinding>{});
            if (!explicit_layout_compatible(group0_entries->has_value() ? &**group0_entries : nullptr, wb)) {
                auto_meta.remap = true;
                auto_meta.bindings = std::move(wb);
                ++stats.auto_layout_remap_pipelines;
            }
        }
        auto_meta.gen = ++pipeline_gen_counter;
        FitDummies fit;
        wgpu::PipelineLayout fitted_pl;
        if (!auto_layout) {
            const std::vector<FitStage> st = {{kFitVertex, &groups_of(vs)}, {kFitFragment, fs ? &groups_of(*fs) : nullptr}};
            fitted_pl = fit_pipeline_layout(
                *pl, st, auto_meta, merge_wgsl_group0_bindings(group0_of(vs), fs ? group0_of(*fs) : std::vector<WgslBinding>{}),
                &fit);
        }
        wgpu::PipelineLayout reflected_pl;
        if (!fitted_pl && auto_meta.remap && reflection.shader_count() != 0) {
            std::vector<const reflect::StageRecord*> rf;
            if (const auto* r = resolve_shader_reflection(vs, "vertex")) rf.push_back(r);
            if (fs) if (const auto* r = resolve_shader_reflection(*fs, "fragment")) rf.push_back(r);
            reflected_pl = reflected_remap_layout(auto_meta, rf, fs ? 2 : 1);
        }
        rd.layout = fitted_pl ? fitted_pl : reflected_pl ? reflected_pl : auto_layout || auto_meta.remap ? nullptr : layout;
        rd.vertex.module = vs.module;
        const std::string& vep = entry_point_of(vs, "vertex");
        rd.vertex.entryPoint = {vep.data(), vep.size()};
        rd.vertex.bufferCount = vbs.size();
        rd.vertex.buffers = vbs.data();

        RenderPipelineEntry e;
        e.fullscreen = vbs.empty();
        e.vs_hash = vs.hash;
        e.fs_hash = fs ? fs->hash : 0;
        if (options.async_pipelines) {
            create_render_async(rd, &e);
        } else {
            e.pipeline = device.CreateRenderPipeline(&rd);
        }
        bool expects = declares_group0_of(vs) || (fs && declares_group0_of(*fs));
        if (!expects && !auto_layout && group0_has_entries) expects = true;
        e.expects_group0 = expects;
        e.auto_meta = std::move(auto_meta);
        e.fit = std::move(fit);
        e.perf_hash = fs && fs->hash ? fs->hash : vs.hash;
        e.quarantined_hash = quarantined_stage(fs ? fs->hash : 0, vs.hash);
        render_pipelines[h.id] = std::move(e);
        ++stats.render_pipelines;
        return true;
    }
    std::string fragment_ep_storage;

    // ---- async pipeline creation (live mode) ----

    void create_render_async(const wgpu::RenderPipelineDescriptor& rd, RenderPipelineEntry* e) {
        auto state = std::make_shared<PendingPipe>();
        e->future = device.CreateRenderPipelineAsync(
            &rd, wgpu::CallbackMode::AllowProcessEvents,
            [state](wgpu::CreatePipelineAsyncStatus s, wgpu::RenderPipeline pl, wgpu::StringView msg) {
                state->done = true;
                state->status = s;
                if (s == wgpu::CreatePipelineAsyncStatus::Success) state->render = std::move(pl);
                else state->message = to_string(msg);
            });
        e->pending = std::move(state);
        ++stats.async_pipelines;
    }

    void create_compute_async(const wgpu::ComputePipelineDescriptor& cd, ComputePipelineEntry* e) {
        auto state = std::make_shared<PendingPipe>();
        e->future = device.CreateComputePipelineAsync(
            &cd, wgpu::CallbackMode::AllowProcessEvents,
            [state](wgpu::CreatePipelineAsyncStatus s, wgpu::ComputePipeline pl, wgpu::StringView msg) {
                state->done = true;
                state->status = s;
                if (s == wgpu::CreatePipelineAsyncStatus::Success) state->compute = std::move(pl);
                else state->message = to_string(msg);
            });
        e->pending = std::move(state);
        ++stats.async_pipelines;
    }

    PipeWaitBudget pipe_budget;

    // ---- pipelines deferred on their shaders (shader workers, live mode) ----

    // Waits up to `timeout_ns` (kPipeWaitUnlimited = no limit) for the shaders of a deferred pipeline,
    // then builds it (Create*PipelineAsync: the compile itself is then Dawn's). True once it is no
    // longer deferred. `slot` is the map entry itself and is rewritten in place by the build.
    bool settle(const AsyncSlot& slot, uint64_t timeout_ns) {
        if (!slot.deferred) return true;
        const bool compute = slot.deferred_compute;
        auto& m = compute ? deferred_compute : deferred_render;
        auto it = m.find(slot.deferred_id);
        if (it == m.end()) {  // cannot happen (erased together with the entry); never leave it hanging
            slot.deferred = false;
            slot.pending.reset();
            slot.failed = true;
            return true;
        }
        const ShaderRef a = it->second.a, b = it->second.b;
        auto ready = [&] { return shader_ready(a) && shader_ready(b); };
        if (!ready()) {
            if (timeout_ns == 0) return false;
            const uint64_t t0 = now_ns();
            ++stats.shader_async_waits;
            const bool ok = shader_workers.wait(ready, timeout_ns);
            stats.shader_async_wait_ns += now_ns() - t0;
            if (!ok) return false;
        }
        build_deferred(compute, slot.deferred_id);
        return true;
    }

    // Budgeted (the frame's pipeline wait budget, as for a still-compiling pipeline): false = the
    // shaders are not ready in time; the caller drops the draw / dispatch and counts it.
    bool settle_budgeted(const AsyncSlot& slot) {
        if (!slot.deferred) return true;
        const uint64_t allow = pipe_budget.allowance(slot.fullscreen);
        const uint64_t t0 = now_ns();
        const bool built = settle(slot, allow);
        pipe_budget.charge(slot.fullscreen, now_ns() - t0);
        return built;
    }

    // Replays a deferred pipeline packet with its captured layout and shaders. A failure (the
    // packet's own validation) leaves a failed entry: draws / dispatches with it are dropped and counted.
    void build_deferred(bool compute, uint32_t id) {
        auto& m = compute ? deferred_compute : deferred_render;
        auto it = m.find(id);
        if (it == m.end()) return;
        DeferredPipe d = std::move(it->second);
        m.erase(it);
        account_worker_shader(*d.a);
        if (d.b) account_worker_shader(*d.b);
        const PlEntry* pl = d.layout ? &*d.layout : nullptr;
        bool ok;
        if (compute) {
            GpuCreateComputePipeline2Payload h;
            std::memcpy(&h, d.payload.data(), std::min(sizeof h, d.payload.size()));
            ok = build_compute_pipeline2(h, pl, *d.a);
        } else {
            GpuCreateRenderPipeline2Payload h;
            std::memcpy(&h, d.payload.data(), std::min(sizeof h, d.payload.size()));
            ok = build_render_pipeline2(h, pl, *d.a, d.b.get());
        }
        ++stats.pipelines_deferred_built;
        if (ok) return;
        packet_error(d.opcode, pending_error("deferred pipeline build failed"));
        if (compute) {
            auto e = compute_pipelines.find(id);
            if (e == compute_pipelines.end()) return;
            e->second.deferred = false;
            e->second.pending.reset();
            e->second.failed = true;
            e->second.pipeline = make_error_compute_pipeline();
        } else {
            auto e = render_pipelines.find(id);
            if (e == render_pipelines.end()) return;
            e->second.deferred = false;
            e->second.pending.reset();
            e->second.failed = true;
            e->second.pipeline = make_error_render_pipeline();
        }
    }

    // Between packets: builds every deferred pipeline whose shaders finished, so Dawn starts
    // compiling it ahead of its first use. Cheap when no worker finished anything since the last call.
    void pump_deferred() {
        drain_worker_errors();
        if (deferred_render.empty() && deferred_compute.empty()) return;
        const uint64_t done = shader_workers.completed();
        if (done == deferred_seen_completed) return;
        deferred_seen_completed = done;
        std::vector<uint32_t> ids;
        for (int pass = 0; pass < 2; ++pass) {
            const bool compute = pass == 1;
            ids.clear();
            for (const auto& [id, d] : compute ? deferred_compute : deferred_render)
                if (shader_ready(d.a) && shader_ready(d.b)) ids.push_back(id);
            for (uint32_t id : ids) build_deferred(compute, id);
        }
    }

    // Waits for the entry's compile within the frame's wait budget (pipe_budget_model.h); true once it
    // finished. A pipeline built ahead of use is already done and costs nothing here. False: still
    // compiling past the budget, the caller skips the draw / dispatch (the compile keeps running).
    bool await_pipe(const AsyncSlot& slot) {
        if (slot.deferred) {
            if (!settle_budgeted(slot)) return false;
            if (!slot.pending) return true;  // its build failed: resolve() hands out nothing
        }
        if (slot.pending->done) return true;
        instance.ProcessEvents();
        if (slot.pending->done) return true;
        const uint64_t allow = pipe_budget.allowance(slot.fullscreen);
        if (allow == 0) return false;
        ++stats.async_pipeline_waits;
        const uint64_t t0 = now_ns();
        instance.WaitAny(slot.future, allow);
        const uint64_t dt = now_ns() - t0;
        stats.async_pipeline_wait_ns += dt;
        pipe_budget.charge(slot.fullscreen, dt);
        return slot.pending->done;
    }

    // Blocks without a budget (bind group layout / remap paths that need the pipeline's layout now).
    void await_pipe_forever(const AsyncSlot& slot) {
        if (slot.deferred) {
            settle(slot, kPipeWaitUnlimited);
            if (!slot.pending) return;
        }
        if (slot.pending->done) return;
        ++stats.async_pipeline_waits;
        const uint64_t t0 = now_ns();
        instance.WaitAny(slot.future, kWaitForever);
        stats.async_pipeline_wait_ns += now_ns() - t0;
    }

    // Still compiling (no wait): the draw / dispatch using it is dropped.
    bool pipe_still_pending(const AsyncSlot& slot) {
        if (!slot.pending) return false;
        if (slot.deferred) return true;  // not even created: only SetPipeline / a binding settles it (budgeted)
        if (slot.pending->done) return false;
        instance.ProcessEvents();
        return !slot.pending->done;
    }

    // The quarantined hash among a render pipeline's stages (fragment first, as perf_hash), else 0.
    uint64_t quarantined_stage(uint64_t fs_hash, uint64_t vs_hash) const {
        if (options.quarantine.empty()) return 0;
        if (options.quarantine.contains(fs_hash)) return fs_hash;
        if (options.quarantine.contains(vs_hash)) return vs_hash;
        return 0;
    }

    // A draw / dispatch skipped by the shader quarantine: counted per phase in the drop table, never silent.
    void note_quarantined(uint64_t hash, bool dispatch) {
        ++stats.quarantine_skips;
        d3d12_guest::drop_table().record(hash, d3d12_guest::DropTable::Reason::Quarantined, dispatch);
    }

    void note_pipeline_pending(uint64_t hash, bool dispatch) {
        ++stats.pipeline_pending_skips;
        d3d12_guest::drop_table().record(hash, d3d12_guest::DropTable::Reason::PipelinePending, dispatch);
    }

    static std::string hash_hex(uint64_t h) {
        char b[24];
        std::snprintf(b, sizeof b, "0x%016llx", static_cast<unsigned long long>(h));
        return b;
    }

    // A draw / dispatch skipped because its async pipeline failed to compile: counted per hash, never submitted
    // (an invalid pipeline in the pass would invalidate the whole command buffer).
    void note_pipeline_failed(uint64_t hash, bool dispatch) {
        ++stats.pipeline_failed_skips;
        if (hash != 0 && std::find(stats.failed_pipeline_hashes.begin(), stats.failed_pipeline_hashes.end(), hash) ==
                             stats.failed_pipeline_hashes.end())
            stats.failed_pipeline_hashes.push_back(hash);
        d3d12_guest::drop_table().record(hash, d3d12_guest::DropTable::Reason::PipelineFailed, dispatch);
    }

    void pipe_failed(const PendingPipe& st, const char* what, const AsyncSlot& slot) {
        ++stats.async_pipeline_failures;
        slot.failed = true;
        const std::string msg = std::string("async ") + what + " pipeline [vs=" + hash_hex(slot.vs_hash) +
                                " fs=" + hash_hex(slot.fs_hash) + "]: " + st.message;
        stats.last_device_error = msg;
        ++stats.async_failure_kinds[classify_pipeline_failure(st.message)];
        if (stats.device_error_log.size() < 20) stats.device_error_log.push_back(msg);
        if (stats.async_pipeline_failures <= 16) std::fprintf(stderr, "[native_exec] %s\n", msg.substr(0, 600).c_str());
    }

    // An invalid-by-validation pipeline (the entry point does not exist in a valid module). The
    // descriptors must carry a real module: Dawn dereferences a null module while validating it
    // (ObjectBase::GetDevice on null: a host access violation that killed live runs).
    wgpu::RenderPipeline make_error_render_pipeline() {
        if (!error_vertex_module)
            error_vertex_module = make_module("@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(); }\n");
        wgpu::RenderPipelineDescriptor bad{};
        bad.vertex.module = error_vertex_module;
        bad.vertex.entryPoint = {"pg6_missing_entry", WGPU_STRLEN};
        return device.CreateRenderPipeline(&bad);
    }

    wgpu::ComputePipeline make_error_compute_pipeline() {
        if (!error_compute_module) error_compute_module = make_module("@compute @workgroup_size(1) fn cs_main() {}\n");
        wgpu::ComputePipelineDescriptor bad{};
        bad.compute.module = error_compute_module;
        bad.compute.entryPoint = {"pg6_missing_entry", WGPU_STRLEN};
        return device.CreateComputePipeline(&bad);
    }

    const wgpu::RenderPipeline& resolve(const RenderPipelineEntry& e) {
        if (e.pending) {
            await_pipe_forever(e);
            if (!e.pending) return e.pipeline;  // a deferred build that failed
            if (e.pending->render) {
                e.pipeline = e.pending->render;
            } else {
                pipe_failed(*e.pending, "render", e);
                // What sync creation yields for an invalid descriptor: an error pipeline (the draw
                // that uses it is then dropped by validation), not a null handle.
                e.pipeline = make_error_render_pipeline();
            }
            e.pending.reset();
        }
        return e.pipeline;
    }

    // Budgeted: null while the compile is still running past the frame's wait budget.
    const wgpu::RenderPipeline* resolve_budgeted(const RenderPipelineEntry& e) {
        if (e.pending && !await_pipe(e)) return nullptr;
        const wgpu::RenderPipeline& p = resolve(e);
        return e.failed ? nullptr : &p;  // a failed compile is never handed to an encoder
    }
    const wgpu::ComputePipeline* resolve_budgeted(const ComputePipelineEntry& e) {
        if (e.pending && !await_pipe(e)) return nullptr;
        const wgpu::ComputePipeline& p = resolve(e);
        return e.failed ? nullptr : &p;
    }

    const wgpu::ComputePipeline& resolve(const ComputePipelineEntry& e) {
        if (e.pending) {
            await_pipe_forever(e);
            if (!e.pending) return e.pipeline;
            if (e.pending->compute) {
                e.pipeline = e.pending->compute;
            } else {
                pipe_failed(*e.pending, "compute", e);
                e.pipeline = make_error_compute_pipeline();
            }
            e.pending.reset();
        }
        return e.pipeline;
    }

    bool create_compute_pipeline2(const Packet& p) {
        const auto h = read_payload<GpuCreateComputePipeline2Payload>(p);
        const PlEntry* pl = nullptr;
        if (h.pipeline_layout_id != 0) {
            auto lit = pipeline_layouts.find(h.pipeline_layout_id);
            if (lit == pipeline_layouts.end()) return fail("unknown pipeline layout " + std::to_string(h.pipeline_layout_id));
            pl = &lit->second;
        }
        auto cit = shaders.find(h.cs_module_id);
        if (cit == shaders.end()) return fail("unknown compute shader module " + std::to_string(h.cs_module_id));
        ShaderRef cs = cit->second;
        deferred_compute.erase(h.id);
        if (shader_on_worker(cs)) {
            ComputePipelineEntry ce;
            ce.pending = std::make_shared<PendingPipe>();
            ce.deferred = true;
            ce.deferred_id = h.id;
            ce.deferred_compute = true;
            ce.vs_hash = cs->hash;
            compute_pipelines[h.id] = std::move(ce);
            compute_meta.erase(h.id);  // known once built (the remap needs the WGSL's group 0 bindings)
            compute_hash[h.id] = cs->hash;
            defer_pipeline(deferred_compute, h.id, p, pl, std::move(cs), nullptr);
            return true;
        }
        return build_compute_pipeline2(h, pl, *cs);
    }

    bool build_compute_pipeline2(const GpuCreateComputePipeline2Payload& h, const PlEntry* pl, const ShaderEntry& cs) {
        wgpu::PipelineLayout layout;
        const bool auto_layout = pl == nullptr;
        const std::optional<std::vector<Group0LayoutEntry>>* group0_entries = nullptr;
        if (!auto_layout) {
            layout = pl->layout;
            group0_entries = &pl->group0_entries;
        } else {
            ++stats.auto_layout_pipelines;
        }
        ensure_shader(cs);
        const std::string& ep = entry_point_of(cs, "compute");
        AutoLayoutMeta auto_meta;
        if (!auto_layout && group0_entries) {
            std::vector<WgslBinding> wb = group0_of(cs);
            if (!explicit_layout_compatible(group0_entries->has_value() ? &**group0_entries : nullptr, wb)) {
                auto_meta.remap = true;
                auto_meta.bindings = std::move(wb);
                ++stats.auto_layout_remap_pipelines;
            }
        }
        auto_meta.gen = ++pipeline_gen_counter;
        wgpu::ComputePipelineDescriptor cd{};
        FitDummies fit;
        wgpu::PipelineLayout fitted_pl;
        if (!auto_layout) fitted_pl = fit_pipeline_layout(*pl, {{kFitCompute, &groups_of(cs)}}, auto_meta, group0_of(cs), &fit);
        wgpu::PipelineLayout reflected_pl;
        if (!fitted_pl && auto_meta.remap && reflection.shader_count() != 0) {
            std::vector<const reflect::StageRecord*> rf;
            if (const auto* r = resolve_shader_reflection(cs, "compute")) rf.push_back(r);
            reflected_pl = reflected_remap_layout(auto_meta, rf, 1);
        }
        cd.layout = fitted_pl ? fitted_pl : reflected_pl ? reflected_pl : auto_layout || auto_meta.remap ? nullptr : layout;
        cd.compute.module = cs.module;
        cd.compute.entryPoint = {ep.data(), ep.size()};
        ComputePipelineEntry ce;
        ce.fit = std::move(fit);
        if (options.async_pipelines) {
            create_compute_async(cd, &ce);
        } else {
            ce.pipeline = device.CreateComputePipeline(&cd);
        }
        compute_pipelines[h.id] = std::move(ce);
        compute_meta[h.id] = std::move(auto_meta);
        compute_hash[h.id] = cs.hash;
        ++stats.compute_pipelines;
        return true;
    }

    // ---- v1 pipelines and bind groups (wgpu_worker.js _handleCreateRenderPipeline / ...ComputePipeline /
    // _handleCreateBindGroup) ----
    //
    // The worker consults its reflection registry for an explicit layout, the entry points and the
    // vertex attributes. The registry (reflection_model.h) is filled by `load_reflection` / a
    // `<pack>.reflect.json` next to the pack; a shader it has no record of takes the path of an empty
    // registry (`_resolveShaderReflection` returns null): layout 'auto', `vs_main` / `fs_main` /
    // `cs_main`, one float vertex attribute (counted as `reflection_unavailable`). The G34 fallback
    // shader profiles belong to the worker's shader router, which has no native counterpart.

    void create_render_pipeline1(const Packet& p) {
        const auto h = read_payload<GpuCreateRenderPipelinePayload>(p);
        auto vit = shaders.find(h.vs_shader_id);
        if (vit == shaders.end()) {
            ++stats.op_errors;  // unknown vertex shader module
            return;
        }
        const ShaderEntry& vs = *vit->second;
        shader_here_now(vit->second);
        ensure_shader(vs);
        const ShaderEntry* fs = nullptr;
        if (h.fs_shader_id != 0) {
            auto fit = shaders.find(h.fs_shader_id);
            if (fit == shaders.end()) {
                ++stats.op_errors;  // unknown fragment shader module
                return;
            }
            fs = fit->second.get();
            shader_here_now(fit->second);
            ensure_shader(*fs);
        }
        const reflect::StageRecord* vs_refl = resolve_shader_reflection(vs, "vertex");
        const reflect::StageRecord* fs_refl = fs ? resolve_shader_reflection(*fs, "fragment") : nullptr;
        std::vector<const reflect::StageRecord*> matched;
        if (vs_refl) matched.push_back(vs_refl);
        if (fs_refl) matched.push_back(fs_refl);
        ReflectedLayout reflected;
        bool layout_error = false;
        const bool has_reflected = build_reflected_layout(matched, &reflected, &layout_error);
        if (layout_error) return;  // the worker's exception: op error, no pipeline
        if (has_reflected) {
            ++stats.reflected_render_pipelines;
        } else {
            ++stats.reflection_unavailable;
        }

        const V1VertexLayout vl = plan_v1_vertex_layout(h.vertex_stride);
        wgpu::VertexAttribute attr{};
        attr.shaderLocation = 0;
        attr.offset = 0;
        attr.format = vl.components == 3   ? wgpu::VertexFormat::Float32x3
                      : vl.components == 2 ? wgpu::VertexFormat::Float32x2
                                           : wgpu::VertexFormat::Float32;
        wgpu::VertexBufferLayout vb{};
        vb.arrayStride = vl.stride;
        vb.stepMode = wgpu::VertexStepMode::Vertex;
        vb.attributeCount = 1;
        vb.attributes = &attr;
        bool has_buffer = vl.has_buffer;

        // buildVertexBufferLayouts(vsRefl, vertexStride): a rejected layout keeps the default.
        std::vector<wgpu::VertexAttribute> reflected_attrs;
        if (vs_refl && h.vertex_stride > 0 && !vs_refl->vertex_inputs.empty()) {
            std::vector<reflect::VertexBufferLayoutDesc> vbl;
            std::string verr;
            if (reflect::build_vertex_buffer_layouts(*vs_refl, h.vertex_stride, &vbl, &verr) && !vbl.empty()) {
                for (const auto& a : vbl[0].attributes) {
                    uint32_t bytes = 0;
                    wgpu::VertexAttribute va{};
                    va.format = wgpu_vertex_format(a.format, &bytes);
                    va.offset = a.offset;
                    va.shaderLocation = a.location;
                    reflected_attrs.push_back(va);
                }
                vb.arrayStride = vbl[0].array_stride;
                vb.attributeCount = reflected_attrs.size();
                vb.attributes = reflected_attrs.data();
                has_buffer = true;
                ++stats.reflected_vertex_layouts;
            } else {
                ++stats.reflection_vertex_fallbacks;
            }
        }
        const std::string vs_ep = vs_refl && !vs_refl->entry_point.empty() ? vs_refl->entry_point : "vs_main";
        const std::string fs_ep = fs_refl && !fs_refl->entry_point.empty() ? fs_refl->entry_point : "fs_main";

        wgpu::RenderPipelineDescriptor rd{};
        if (has_reflected) rd.layout = reflected.pipeline_layout;  // else nullptr: 'auto'
        rd.vertex.module = vs.module;
        rd.vertex.entryPoint = {vs_ep.data(), vs_ep.size()};
        rd.vertex.bufferCount = has_buffer ? 1 : 0;
        rd.vertex.buffers = &vb;
        rd.primitive.topology = wgpu_topology(static_cast<uint8_t>(std::min<uint32_t>(h.topology, 255)));
        wgpu::ColorTargetState target{};
        target.format = stream_format;  // _getRenderTargetFormat()
        wgpu::FragmentState fragment{};
        if (fs) {
            fragment.module = fs->module;
            fragment.entryPoint = {fs_ep.data(), fs_ep.size()};
            fragment.targetCount = 1;
            fragment.targets = &target;
            rd.fragment = &fragment;
        }
        RenderPipelineEntry e;
        e.pipeline = device.CreateRenderPipeline(&rd);
        e.expects_group0 = declares_group0_of(vs) || (fs && declares_group0_of(*fs));
        e.auto_meta.gen = ++pipeline_gen_counter;
        e.perf_hash = fs && fs->hash ? fs->hash : vs.hash;
        e.quarantined_hash = quarantined_stage(fs ? fs->hash : 0, vs.hash);
        render_pipelines[h.id] = std::move(e);
        if (has_reflected) reflected_render[h.id] = std::move(reflected);
        else reflected_render.erase(h.id);
        ++stats.render_pipelines;
        ++stats.v1_render_pipelines;
    }

    void create_compute_pipeline1(const Packet& p) {
        const auto h = read_payload<GpuCreateComputePipelinePayload>(p);
        auto cit = shaders.find(h.cs_shader_id);
        if (cit == shaders.end()) return;  // `_buildComputePipelineDescriptorV1` returns null silently
        const ShaderEntry& cs = *cit->second;
        shader_here_now(cit->second);
        ensure_shader(cs);
        const reflect::StageRecord* cs_refl = resolve_shader_reflection(cs, "compute");
        ReflectedLayout reflected;
        bool layout_error = false;
        const bool has_reflected = cs_refl && build_reflected_layout({cs_refl}, &reflected, &layout_error);
        if (layout_error) return;
        if (has_reflected) {
            ++stats.reflected_compute_pipelines;
        } else {
            ++stats.reflection_unavailable;
        }
        const std::string cs_ep = cs_refl && !cs_refl->entry_point.empty() ? cs_refl->entry_point : "cs_main";
        wgpu::ComputePipelineDescriptor cd{};
        if (has_reflected) cd.layout = reflected.pipeline_layout;  // else nullptr: 'auto'
        cd.compute.module = cs.module;
        cd.compute.entryPoint = {cs_ep.data(), cs_ep.size()};
        compute_pipelines[h.id].pipeline = device.CreateComputePipeline(&cd);
        if (has_reflected) reflected_compute[h.id] = std::move(reflected);
        else reflected_compute.erase(h.id);
        compute_meta.erase(h.id);
        compute_hash[h.id] = cs.hash;
        ++stats.compute_pipelines;
        ++stats.v1_compute_pipelines;
    }

    void create_bind_group1(const Packet& p) {
        const auto h = read_payload<GpuCreateBindGroupPayload>(p);
        if (uint64_t{16} + uint64_t{h.entry_count} * sizeof(GpuBindGroupEntry) > p.payload_bytes) {
            ++stats.op_errors;  // truncated payload
            return;
        }
        wgpu::BindGroupLayout layout;
        bool compute = false;
        // getBindGroupLayout: the reflected layout of the pipeline first, else pipeline.getBindGroupLayout.
        const std::vector<reflect::LayoutEntryDesc>* synth_entries = nullptr;
        std::vector<reflect::LayoutEntryDesc> synth_from_bgl;
        auto reflected_group = [&](const std::unordered_map<uint32_t, ReflectedLayout>& m) -> const ReflectedLayout* {
            auto it = m.find(h.pipeline_id);
            return it != m.end() && h.group_index < it->second.bgls.size() ? &it->second : nullptr;
        };
        const ReflectedLayout* rl = reflected_group(reflected_render);
        const ReflectedLayout* cl = reflected_group(reflected_compute);
        auto rit = render_pipelines.find(h.pipeline_id);
        auto cit = compute_pipelines.find(h.pipeline_id);
        if (rit != render_pipelines.end()) {
            if (rl) layout = rl->bgls[h.group_index];
            else if (cl) layout = cl->bgls[h.group_index];
            else {
                // No reflection: the only source of the layout is the compiled pipeline. Budgeted; past it
                // the bind group is skipped and counted as pipeline_pending.
                const wgpu::RenderPipeline* rp = resolve_budgeted(rit->second);
                if (!rp) { if (rit->second.failed) note_pipeline_failed(rit->second.perf_hash, false); else note_pipeline_pending(rit->second.perf_hash, false); return; }
                layout = rp->GetBindGroupLayout(h.group_index);
            }
        } else if (cit != compute_pipelines.end()) {
            if (rl) layout = rl->bgls[h.group_index];
            else if (cl) layout = cl->bgls[h.group_index];
            else {
                const wgpu::ComputePipeline* cp = resolve_budgeted(cit->second);
                if (!cp) { note_pipeline_pending(0, true); return; }
                layout = cp->GetBindGroupLayout(h.group_index);
            }
            compute = true;
        } else {
            auto bit = bgls.find(h.pipeline_id);  // a v2 bind group layout id works here too
            if (bit == bgls.end()) {
                ++stats.op_errors;  // unknown pipeline or bind group layout
                return;
            }
            layout = bit->second.layout;
        }
        // _getBindGroupLayoutEntriesForSynthesis: the reflected group, else the v2 layout record's entries.
        if (rl) {
            synth_entries = &rl->groups[h.group_index];
        } else if (cl) {
            synth_entries = &cl->groups[h.group_index];
        } else if (rit == render_pipelines.end() && cit == compute_pipelines.end()) {
            auto bit = bgls.find(h.pipeline_id);
            if (bit != bgls.end() && bit->second.entries_known && !bit->second.entries.empty()) {
                for (const LayoutEntry& le : bit->second.entries) {
                    if (le.kind != 1) continue;
                    reflect::LayoutEntryDesc d;
                    d.kind = reflect::LayoutEntryDesc::Kind::Sampler;
                    d.binding = le.binding;
                    d.sampler_type = le.subtype;
                    synth_from_bgl.push_back(d);
                }
                synth_entries = &synth_from_bgl;
            }
        }
        struct Entry {
            uint32_t binding;
            wgpu::Buffer buffer;
            uint64_t offset = 0, size = wgpu::kWholeSize;
            wgpu::TextureView view;
            wgpu::Sampler sampler;
        };
        std::vector<Entry> entries;
        for (uint32_t i = 0; i < h.entry_count; ++i) {
            const auto e = read_payload<GpuBindGroupEntry>(p, 16 + size_t{i} * sizeof(GpuBindGroupEntry));
            Entry pe;
            pe.binding = e.binding;
            if (e.resource_type == static_cast<uint32_t>(GpuBindResourceType::Buffer)) {
                auto it = buffers.find(e.resource_id);
                if (it == buffers.end()) {
                    ++stats.op_errors;  // unknown buffer
                    return;
                }
                pe.buffer = it->second.buffer;
                pe.offset = e.offset;
                if (e.size > 0) pe.size = e.size;
            } else if (e.resource_type == static_cast<uint32_t>(GpuBindResourceType::TextureView)) {
                auto it = textures.find(e.resource_id);
                if (it == textures.end() || !it->second.default_view) {
                    ++stats.op_errors;  // unknown texture
                    return;
                }
                pe.view = it->second.default_view;
            } else if (e.resource_type == static_cast<uint32_t>(GpuBindResourceType::Sampler)) {
                auto it = samplers.find(e.resource_id);
                if (it != samplers.end()) {
                    pe.sampler = it->second;
                } else if (e.resource_id == 0) {
                    if (!default_filtering_sampler) {
                        wgpu::SamplerDescriptor sd{};
                        sd.magFilter = wgpu::FilterMode::Linear;
                        sd.minFilter = wgpu::FilterMode::Linear;
                        sd.mipmapFilter = wgpu::MipmapFilterMode::Linear;
                        default_filtering_sampler = device.CreateSampler(&sd);
                    }
                    pe.sampler = default_filtering_sampler;
                } else {
                    ++stats.op_errors;  // unknown sampler
                    return;
                }
            } else {
                ++stats.op_errors;  // invalid resourceType
                return;
            }
            entries.push_back(std::move(pe));
        }
        // synthesizeFallbackSamplerEntries: a sampler slot of the layout nothing is bound to gets the
        // default sampler of its kind; the entries then come back keyed (last wins) and sorted by binding.
        if (synth_entries) {
            std::vector<uint32_t> bound;
            for (const Entry& pe : entries) bound.push_back(pe.binding);
            std::vector<Entry> added;
            for (const reflect::LayoutEntryDesc& le : *synth_entries) {
                if (le.kind != reflect::LayoutEntryDesc::Kind::Sampler) continue;
                if (std::find(bound.begin(), bound.end(), le.binding) != bound.end()) continue;
                Entry se;
                se.binding = le.binding;
                se.sampler = fallback_sampler_kind(le.sampler_type < 3 ? le.sampler_type : 0);
                added.push_back(std::move(se));
                bound.push_back(le.binding);
                ++stats.v1_synth_samplers;
            }
            if (!added.empty()) {
                std::map<uint32_t, Entry> by_binding;
                for (Entry& pe : entries) by_binding[pe.binding] = std::move(pe);
                for (Entry& se : added) by_binding[se.binding] = std::move(se);
                entries.clear();
                for (auto& kv : by_binding) entries.push_back(std::move(kv.second));
            }
        }
        std::vector<wgpu::BindGroupEntry> wes(entries.size());
        for (size_t i = 0; i < entries.size(); ++i) {
            wes[i].binding = entries[i].binding;
            wes[i].buffer = entries[i].buffer;
            wes[i].offset = entries[i].offset;
            wes[i].size = entries[i].size;
            wes[i].sampler = entries[i].sampler;
            wes[i].textureView = entries[i].view;
        }
        wgpu::BindGroupDescriptor bd{};
        bd.layout = layout;
        bd.entryCount = wes.size();
        bd.entries = wes.data();
        bind_groups[h.id] = device.CreateBindGroup(&bd);
        if (compute) compute_bind_groups.insert(h.id);
        else compute_bind_groups.erase(h.id);
        ++stats.bind_groups;
        ++stats.v1_bind_groups;
    }

    // ---- bind groups (RenderStatePackets._createBindGroup / _dummyResource / _setBindGroup) ----

    struct Dummy {
        wgpu::Buffer buffer;
        wgpu::Sampler sampler;
        wgpu::TextureView view;
    };

    Dummy dummy_resource(const LayoutEntry& le, uint64_t owner = 0) {
        Dummy d;
        if (le.kind == 0) {
            const uint64_t size = std::max<uint64_t>(256, (le.min_binding_size + 3) / 4 * 4);
            const std::string key = "buffer:" + std::to_string(size);
            auto it = dummy_buffers.find(key);
            if (it == dummy_buffers.end()) {
                wgpu::BufferDescriptor bd{};
                bd.size = size;
                bd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
                it = dummy_buffers.emplace(key, device.CreateBuffer(&bd)).first;
            }
            d.buffer = it->second;
            return d;
        }
        if (le.kind == 1) {
            const std::string key = "sampler:" + std::to_string(le.subtype);
            auto it = dummy_samplers.find(key);
            if (it == dummy_samplers.end()) {
                wgpu::SamplerDescriptor sd{};
                if (le.subtype == 2) sd.compare = wgpu::CompareFunction::Always;
                it = dummy_samplers.emplace(key, device.CreateSampler(&sd)).first;
            }
            d.sampler = it->second;
            return d;
        }
        const bool storage = le.kind == 3;
        uint8_t format = 22;
        if (storage) {
            const StorageResolution sr = resolve_storage_format(le.storage_format, features);
            format = sr.supported && sr.storage_format ? sr.storage_format : 22;
        } else {
            static const uint8_t kByType[5] = {22, 16, 48, 26, 25};  // float, unfilterable, depth, sint, uint
            format = kByType[le.subtype < 5 ? le.subtype : 0];
        }
        const bool ms = !storage && le.multisampled;
        if (ms && (!render_attachment_supported(format, features) || format_is_float32(format) || format_is_integer(format)))
            format = 22;
        std::string key = "tex:" + std::to_string(format) + ":" + std::to_string(le.view_dim) + ":" +
                                std::to_string(ms) + ":" + std::to_string(storage);
        // Writable storage textures must not alias inside one pass (Dawn: "writable storage texture
        // used twice"): each (owner, binding) gets its own dummy.
        if (storage) key += ":o" + std::to_string(owner) + "b" + std::to_string(le.binding);
        auto it = dummy_views.find(key);
        if (it == dummy_views.end()) {
            const uint32_t layers = le.view_dim == 4 || le.view_dim == 5 ? 6 : 1;
            wgpu::TextureDescriptor td{};
            td.dimension = le.view_dim == 1 ? wgpu::TextureDimension::e1D
                           : le.view_dim == 6 ? wgpu::TextureDimension::e3D
                                              : wgpu::TextureDimension::e2D;
            td.size = {1, 1, layers};
            td.format = wgpu_format(format);
            td.usage = storage ? wgpu::TextureUsage::StorageBinding : wgpu::TextureUsage::TextureBinding;
            td.sampleCount = ms && td.dimension == wgpu::TextureDimension::e2D && layers == 1 ? 4 : 1;
            wgpu::Texture tex = device.CreateTexture(&td);
            dummy_textures.push_back(tex);
            wgpu::TextureViewDescriptor vd{};
            vd.dimension = wgpu_view_dim(le.view_dim);
            it = dummy_views.emplace(key, tex.CreateView(&vd)).first;
        }
        d.view = it->second;
        return d;
    }

    bool create_bind_group2(const Packet& p) {
        if (p.payload_bytes < 16) return false;
        const auto h = read_payload<GpuCreateBindGroup2Payload>(p);
        if (uint64_t{16} + uint64_t{h.entry_count} * 48 > p.payload_bytes) return false;
        auto lit = bgls.find(h.layout_id);
        if (lit == bgls.end()) return fail("bind group " + std::to_string(h.id) + ": unknown bind group layout");
        const BglEntry& rec = lit->second;
        struct Pending {
            uint32_t binding;
            wgpu::Buffer buffer;
            uint64_t offset = 0, size = wgpu::kWholeSize;
            wgpu::Sampler sampler;
            wgpu::TextureView view;
        };
        std::vector<Pending> entries;
        std::vector<uint32_t> seen;
        auto desc = std::make_shared<BgDesc>();
        // dynamicOffsetIndicesForLayoutEntries: the rank of a dynamic-offset buffer binding
        std::vector<uint32_t> dyn_bindings;
        for (const LayoutEntry& x : rec.entries)
            if (x.kind == 0 && x.dynamic) dyn_bindings.push_back(x.binding);
        std::sort(dyn_bindings.begin(), dyn_bindings.end());
        for (uint32_t i = 0; i < h.entry_count; ++i) {
            const auto e = read_payload<GpuBindGroupEntry2>(p, 16 + size_t{i} * 48);
            if (std::find(seen.begin(), seen.end(), e.binding) != seen.end())
                return fail("bind group: duplicate binding " + std::to_string(e.binding));
            seen.push_back(e.binding);
            const LayoutEntry* le = nullptr;
            for (const LayoutEntry& x : rec.entries)
                if (x.binding == e.binding) { le = &x; break; }
            const bool expects_buffer = le && le->kind == 0, expects_texture = le && (le->kind == 2 || le->kind == 3),
                       expects_sampler = le && le->kind == 1;
            const uint32_t type = e.resource_type;
            const bool kind_mismatch = e.resource_id != 0 && le &&
                                       ((expects_buffer && type != 0) || (expects_texture && type != 1) ||
                                        (expects_sampler && type != 2));
            Pending pe;
            pe.binding = e.binding;
            bool have = false;
            // The descriptor entry the layout:'auto' remap matches on (`descriptorResources`): only
            // real resources are recorded; a dummy entry is never picked.
            bool rec_real = false;
            RemapResource rm;
            BgHandles rh;
            rm.binding = e.binding;
            auto use_dummy = [&](const LayoutEntry& l) {
                const Dummy d = dummy_resource(l, h.layout_id);
                pe.buffer = d.buffer;
                pe.offset = 0;
                pe.size = wgpu::kWholeSize;
                pe.sampler = d.sampler;
                pe.view = d.view;
                have = true;
            };
            if (e.resource_id == 0) {
                if (le) use_dummy(*le);
            } else if (type == 0) {
                auto bit = buffers.find(e.resource_id);
                if (bit != buffers.end()) {
                    const BufferEntry& b = bit->second;
                    const uint64_t buf_offset = e.offset, requested = e.size;
                    const uint64_t buf_size = b.size > 0 ? b.size : 0;
                    uint32_t required_usage = 0;
                    if (expects_buffer) required_usage = le->subtype == 0 ? 0x40u : 0x80u;
                    const bool usage_mismatch = expects_buffer && required_usage && !(b.usage & required_usage);
                    uint64_t min_binding = expects_buffer && !usage_mismatch ? le->min_binding_size : 0;
                    if (buf_size && buf_offset >= buf_size) {
                        if (le) use_dummy(*le);
                    } else {
                        uint64_t eff = requested;
                        if (buf_size && eff > 0 && buf_offset + eff > buf_size) eff = buf_size - buf_offset;
                        const uint64_t remaining = buf_size ? buf_size - buf_offset : 0;
                        bool undersized = false;
                        if (min_binding > 0) {
                            const uint64_t bound = eff > 0 ? eff : remaining;
                            if (buf_size && bound < min_binding) {
                                if (remaining >= min_binding) eff = min_binding;
                                else if (le) undersized = true;
                            }
                        }
                        if (undersized) {
                            use_dummy(*le);
                        } else {
                            rec_real = true;  // realResource is set even when the entry itself is a dummy
                            if (!kind_mismatch && !usage_mismatch && le && le->kind == 0)
                                rm.kind = le->subtype == 0   ? WgslBindingKind::Uniform
                                          : le->subtype == 1 ? WgslBindingKind::Storage
                                                             : WgslBindingKind::ReadOnlyStorage;
                            else
                                rm.kind = (b.usage & 0x80u) && !(b.usage & 0x40u) ? WgslBindingKind::Storage
                                                                                   : WgslBindingKind::Uniform;
                            rm.buffer_usage = b.usage;
                            rm.buffer_offset = buf_offset;
                            rm.buffer_size = eff;
                            rm.buffer_total = buf_size;
                            if (le && le->kind == 0) {
                                const auto rank = std::lower_bound(dyn_bindings.begin(), dyn_bindings.end(), e.binding);
                                if (rank != dyn_bindings.end() && *rank == e.binding) {
                                    rm.has_dynamic_offset = true;
                                    rm.dynamic_offset_index = static_cast<uint32_t>(rank - dyn_bindings.begin());
                                }
                            }
                            rh.buffer = b.buffer;
                            if (kind_mismatch || usage_mismatch) {
                                use_dummy(*le);
                            } else {
                                pe.buffer = b.buffer;
                                pe.offset = buf_offset;
                                pe.size = eff > 0 ? eff : wgpu::kWholeSize;
                                have = true;
                            }
                        }
                    }
                }
            } else if (type == 1) {
                auto tit = textures.find(e.resource_id);
                const GpuTextureViewDesc* vd = &e.view;
                if (tit != textures.end()) {
                    TextureEntry& t = tit->second;
                    uint8_t tex_format = e.view.format ? e.view.format : t.geom.format;
                    uint8_t rec_format = 0;  // the descriptor's textureFormat (0 = unknown)
                    if (e.view.format && t.bc3d_fallback && format_is_bc(e.view.format)) tex_format = t.geom.format;
                    rec_format = tex_format;
                    if (!kind_mismatch && le && le->kind == 3 && tex_format) {
                        const StorageResolution sr = resolve_storage_format(tex_format, features);
                        if (!sr.supported || !sr.storage_format) {
                            use_dummy(*le);
                        } else if (sr.linear_sibling) {
                            GpuTextureViewDesc lv = e.view;
                            lv.format = sr.storage_format;
                            lv.mip_level_count = 1;
                            pe.view = texture_view(t, &lv);
                            have = static_cast<bool>(pe.view);
                            if (!have) use_dummy(*le);
                            else { rec_real = true; rec_format = sr.storage_format; rh.view = pe.view; }
                        } else {
                            GpuTextureViewDesc sv = e.view;
                            sv.mip_level_count = 1;
                            pe.view = texture_view(t, e.view.mip_level_count == 1 ? vd : &sv);
                            have = static_cast<bool>(pe.view);
                            if (have) { rec_real = true; rh.view = pe.view; }
                        }
                    } else if (!kind_mismatch && le && le->kind == 2 && le->subtype == 0 && tex_format &&
                               format_is_float32(tex_format) && !features.float32_filterable) {
                        use_dummy(*le);
                    } else if (!kind_mismatch && le && le->kind == 2 && tex_format && format_is_depth(tex_format) &&
                               pick_depth_read(le->subtype, le->multisampled, tex_format, e.view.aspect,
                                               t.sample_count, t.geom.dimension, features.float32_filterable) !=
                                   DepthRead::Direct) {
                        const DepthRead dr = pick_depth_read(le->subtype, le->multisampled, tex_format, e.view.aspect,
                                                             t.sample_count, t.geom.dimension, features.float32_filterable);
                        if (dr == DepthRead::StencilDirect) {
                            GpuTextureViewDesc sv = e.view;
                            sv.aspect = 2;  // the stencil plane (a uint entry); Dawn rejects an all-aspect view
                            pe.view = texture_view(t, &sv);
                            have = static_cast<bool>(pe.view);
                            if (have) ++stats.depth_stencil_direct_binds;
                        } else if (dr != DepthRead::Dummy && depth_twin(e.resource_id, t, dr)) {
                            pe.view = depth_twin_view(t, dr, e.view);
                            have = static_cast<bool>(pe.view);
                        }
                        if (!have) {
                            ++stats.bind_group_depth_layout_dummies;
                            use_dummy(*le);
                        }
                    } else {
                        pe.view = texture_view(t, vd);
                        if (pe.view) {
                            rec_real = true;
                            rh.view = pe.view;
                            if (kind_mismatch) use_dummy(*le);
                            else have = true;
                        }
                    }
                    if (rec_real) {
                        if (!kind_mismatch && le && le->kind == 3) rm.kind = WgslBindingKind::StorageTexture;
                        else if (!kind_mismatch && le && le->kind == 2) rm.kind = WgslBindingKind::Texture;
                        else
                            rm.kind = (t.usage & 0x8u) && !(t.usage & 0x4u) ? WgslBindingKind::StorageTexture
                                                                             : WgslBindingKind::Texture;
                        rm.texture_usage = t.usage;
                        if (rec_format) rm.format = rec_format;
                        // textureViewDimension: the view's, with the 1d <-> 2d promotion rules, else the texture's
                        const uint8_t vdim = e.view.dimension;
                        const bool actual_1d = t.geom.dimension == 0;
                        uint8_t dim;
                        if (vdim != 0) {
                            if (vdim == 1 && (t.geom.dimension == 1 || t.promoted_1d)) dim = 2;
                            else if (actual_1d && vdim != 1) dim = 1;
                            else dim = vdim;
                        } else {
                            const uint8_t base = t.geom.dimension == 0 ? 1 : t.geom.dimension == 2 ? 6 : 2;
                            dim = base == 2 && t.geom.depth_or_layers > 1 ? 3 : base;
                        }
                        rm.view_dim = dim;
                        rm.multisampled = t.sample_count > 1;
                    }
                }
            } else if (type == 2) {
                auto sit = samplers.find(e.resource_id);
                if (sit != samplers.end()) {
                    rec_real = true;
                    rm.kind = WgslBindingKind::Sampler;
                    rm.comparison = comparison_samplers.count(e.resource_id) != 0;
                    rh.sampler = sit->second;
                    if (kind_mismatch) {
                        use_dummy(*le);
                    } else {
                        pe.sampler = sit->second;
                        have = true;
                    }
                }
            }
            if (!have)
                return fail("bind group " + std::to_string(h.id) + ": binding " + std::to_string(e.binding) +
                            " resource " + std::to_string(e.resource_id) + " unavailable");
            if (rec_real) {
                desc->model.push_back(rm);
                desc->handles.push_back(std::move(rh));
            }
            entries.push_back(std::move(pe));
        }
        std::vector<wgpu::BindGroupEntry> wes(entries.size());
        for (size_t i = 0; i < entries.size(); ++i) {
            wes[i].binding = entries[i].binding;
            wes[i].buffer = entries[i].buffer;
            wes[i].offset = entries[i].offset;
            wes[i].size = entries[i].size;
            wes[i].sampler = entries[i].sampler;
            wes[i].textureView = entries[i].view;
        }
        wgpu::BindGroupDescriptor bd{};
        bd.layout = rec.layout;
        bd.entryCount = wes.size();
        bd.entries = wes.data();
        if (rec.over_limits) bind_groups[h.id] = nullptr;  // recorded (desc) for the remap only
        else bind_groups[h.id] = device.CreateBindGroup(&bd);
        bg_descs[h.id] = std::move(desc);
        compute_bind_groups.erase(h.id);  // a v2 group of the same id is not a v1 compute group
        ++stats.bind_groups;
        return true;
    }

    bool set_bind_group2(const Packet& p) {
        if (p.payload_bytes < sizeof(GpuSetBindGroup2Payload)) return false;
        const auto h = read_payload<GpuSetBindGroup2Payload>(p);
        if (uint64_t{16} + uint64_t{h.dynamic_offset_count} * 4 > p.payload_bytes) return false;
        auto it = bind_groups.find(h.bind_group_id);
        if (it == bind_groups.end()) return fail("unknown bind group " + std::to_string(h.bind_group_id));
        SetBindGroup2Cmd c;
        c.group = h.group_index;
        c.bind_group = it->second;
        c.bind_group_id = h.bind_group_id;
        if (auto dit = bg_descs.find(h.bind_group_id); dit != bg_descs.end()) c.desc = dit->second;
        for (uint32_t i = 0; i < h.dynamic_offset_count; ++i) c.offsets.push_back(read_payload<uint32_t>(p, 16 + size_t{i} * 4));
        pass_cmds.emplace_back(std::move(c));
        return true;
    }

    // ---- BeginRenderPass (render_state_packets.js _beginRenderPass) ----

    // _textureAttachmentSize
    AttachmentSize texture_attachment_size(const TextureEntry& t, const GpuTextureViewDesc& v) {
        return mip_size(AttachmentSize{t.geom.width, t.geom.height}, v.base_mip_level);
    }

    bool begin_render_pass(const Packet& p) {
        const auto plan = decode_begin_render_pass(p);
        if (!plan) return false;
        BeginPassCmd cmd;
        cmd.color_count = plan->color_count;
        for (uint32_t i = 0; i < plan->color_count; ++i) {
            const ColorAttachmentPlan& c = plan->colors[i];
            if (c.texture_id == 0) continue;
            ColorAtt& a = cmd.colors[i];
            a.texture_id = c.texture_id;
            a.load = c.load;
            a.store = c.store;
            a.clear = c.clear;
            if (c.texture_id == kGpuSwapchainTextureId) {
                a.swapchain = true;
                if (c.view.format != 0) set_swapchain_format(c.view.format);  // #709
                continue;
            }
            auto it = textures.find(c.texture_id);
            if (it == textures.end()) {
                ++stats.render_pass_missing_texture;
                return fail("render pass: unknown texture " + std::to_string(c.texture_id));
            }
            a.base_mip = c.view.base_mip_level;
            a.base_layer = c.view.base_array_layer;
            a.view = texture_view(it->second, &c.view);
            if (!a.view) {
                ++stats.render_pass_missing_texture;
                return fail("render pass: texture " + std::to_string(c.texture_id) + " createView failed");
            }
            a.size = texture_attachment_size(it->second, c.view);
        }
        if (plan->has_depth_stencil) {
            const GpuRenderPassDepthStencilAttachment& d = plan->depth_stencil;
            auto it = d.texture_id == kGpuSwapchainTextureId ? textures.end() : textures.find(d.texture_id);
            if (it == textures.end()) {
                ++stats.render_pass_missing_texture;
                return fail("render pass: unknown texture " + std::to_string(d.texture_id));
            }
            cmd.has_ds = true;
            cmd.ds = d;
            cmd.ds_view = texture_view(it->second, &d.view);
            if (!cmd.ds_view) {
                ++stats.render_pass_missing_texture;
                return fail("render pass: depth texture createView failed");
            }
            cmd.ds_size = texture_attachment_size(it->second, d.view);
        }
        pass_cmds.emplace_back(std::move(cmd));
        return true;
    }

    // ---- Present (wgpu_worker.js _flushPresent) ----

    struct FrameTarget {
        wgpu::Texture texture;
        wgpu::TextureView view;  // what the stream renders into (the canvas, or the intermediate texture)
        wgpu::TextureView canvas_view;  // the canvas the frame is presented to
        wgpu::Texture canvas_texture;   // its texture (the probe reads it back); empty headless
        bool blit = false;  // view is the intermediate texture: Present blits it into canvas_view
        bool written = false;  // the render target starts each frame as transparent black
    };

    // The canvas texture of the current frame: kept across an early read-back flush, released by Present.
    std::optional<FrameTarget> frame_target;
    bool acquire_target() {
        if (frame_target) return true;
        FrameTarget t;
        if (!acquire(&t)) return false;
        frame_target = std::move(t);
        return true;
    }

    bool acquire(FrameTarget* t) {
        if (!acquire_canvas(t)) return false;
        t->blit = blit_active();
        if (t->blit) {
            ensure_intermediate();
            t->texture = intermediate;
            t->view = intermediate_view;
        } else {
            t->view = t->canvas_view;
        }
        return true;
    }

    bool acquire_canvas(FrameTarget* t) {
        if (!surface) {
            t->texture = offscreen;
            t->canvas_texture = offscreen;
            t->canvas_view = offscreen.CreateView();
            return true;
        }
        uint32_t w = 0, h = 0;
        window->client_size(&w, &h);
        const SurfacePlan plan = surface_state.plan(w, h);
        if (plan == SurfacePlan::SkipMinimized) {
            ++stats.surface_minimized_skips;  // the frame is dropped; the surface keeps its configuration
            return false;
        }
        if (plan == SurfacePlan::Configure || !surface_configured) configure_surface(w, h);
        for (int attempt = 0; attempt < 2; ++attempt) {
            wgpu::SurfaceTexture st;
            surface.GetCurrentTexture(&st);
            AcquireStatus as = AcquireStatus::Error;
            switch (st.status) {
                case wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal: as = AcquireStatus::Optimal; break;
                case wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal: as = AcquireStatus::Suboptimal; break;
                case wgpu::SurfaceGetCurrentTextureStatus::Timeout: as = AcquireStatus::Timeout; break;
                case wgpu::SurfaceGetCurrentTextureStatus::Outdated: as = AcquireStatus::Outdated; break;
                case wgpu::SurfaceGetCurrentTextureStatus::Lost: as = AcquireStatus::Lost; break;
                case wgpu::SurfaceGetCurrentTextureStatus::Error: as = AcquireStatus::Error; break;
                default: as = AcquireStatus::Error; break;
            }
            switch (surface_state.on_acquire(as, attempt)) {
                case AcquireNext::Use:
                    t->texture = st.texture;
                    t->canvas_texture = st.texture;
                    t->canvas_view = st.texture.CreateView();
                    return true;
                case AcquireNext::Reconfigure:
                    ++stats.surface_acquire_retries;
                    window->client_size(&w, &h);  // the size may have changed since the plan
                    if (w == 0 || h == 0) {
                        ++stats.surface_minimized_skips;
                        return false;
                    }
                    configure_surface(w, h);
                    break;
                case AcquireNext::Drop:
                case AcquireNext::Fatal:
                    return false;
            }
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

    // The encode-time state of _flushPresent (`_presentState`) plus _preparePassCommandsForFlush.
    struct Encoder {
        Impl& s;
        FrameTarget& target;
        wgpu::CommandEncoder enc;
        wgpu::RenderPassEncoder rpass;
        wgpu::ComputePassEncoder cpass;
        bool render_pass_opened = false;
        bool explicit_pass = false;       // the active render pass came from BeginRenderPass
        std::optional<AttachmentSize> explicit_size;
        bool group0_bound = false;
        uint8_t bound_groups = 0;  // groups the active pass bound (bit = group): fitted dummies skip them
        // The stream's own bind groups 1..3 of the active pass, to restore after a padded pipeline's
        // draw bound its padding over them (`overridden`).
        struct StreamGroup {
            wgpu::BindGroup group;
            std::vector<uint32_t> offsets;
        };
        std::array<StreamGroup, kFitMaxGroups> stream_groups;
        uint8_t overridden = 0;
        std::optional<uint32_t> active_pipeline;
        std::optional<uint32_t> active_compute;
        // _preparePassCommandsForFlush compute filter state
        bool had_explicit_compute = false, compute_valid = false;
        bool compute_failed = false;  // the compute pipeline's async compile failed
        bool compute_pending = false;  // compute pipeline still compiling past the wait budget
        uint64_t compute_pending_hash = 0;
        uint64_t compute_quarantined_hash = 0;  // the bound compute shader's hash when it is quarantined
        // GPU pass timestamps: the commands being encoded (for the pass's pipeline look-ahead) and
        // the ring slot of this submit (-2: not opened yet, -1: no free slot).
        const std::vector<PassCmd>* cmds = nullptr;
        size_t idx = 0;
        int timer_slot = -2;
        size_t timer_passes = 0;
        // gpu_time_split: the open per-pipeline segment of the active pass (its end query, key and kind).
        bool seg_open = false;
        uint32_t seg_end = 0;
        uint64_t seg_hash = 0;

        // gpu_time_split: the pass-level write keeps only the begin query; the end is written inside the
        // pass (seg_close) so later segments can follow.
        void seg_begin(wgpu::PassTimestampWrites& tw, bool timed, uint64_t hash) {
            seg_open = false;
            if (!timed || !s.features.timestamp_inside_passes) return;
            seg_open = true;
            seg_end = tw.endOfPassWriteIndex;
            seg_hash = hash;
            tw.endOfPassWriteIndex = wgpu::kQuerySetIndexUndefined;
        }
        wgpu::QuerySet timer_queries() const { return s.timer_slots[static_cast<size_t>(timer_slot)].queries; }
        void seg_close() {  // writes the open segment's end timestamp into the active pass
            if (!seg_open) return;
            seg_open = false;
            if (rpass) rpass.WriteTimestamp(timer_queries(), seg_end);
            else if (cpass) cpass.WriteTimestamp(timer_queries(), seg_end);
        }
        // A SetPipeline / SetComputePipeline in the active pass: a different timing key closes the segment
        // and starts the next one.
        void seg_switch(uint64_t hash, d3d12_guest::GpuPassKind kind) {
            if (!seg_open || hash == seg_hash) return;
            seg_close();
            uint32_t b = 0, en = 0;
            if (!s.time_ring->add_pass(hash, kind, &b, &en)) return;
            if (rpass) rpass.WriteTimestamp(timer_queries(), b);
            else if (cpass) cpass.WriteTimestamp(timer_queries(), b);
            else return;
            seg_open = true;
            seg_end = en;
            seg_hash = hash;
            ++s.stats.timer_segments;
        }

        void end_active() {
            seg_close();
            if (rpass) { rpass.End(); rpass = nullptr; }
            if (cpass) { cpass.End(); cpass = nullptr; }
            explicit_pass = false;
            group0_bound = false;
            bound_groups = 0;
            overridden = 0;
            for (auto& sg : stream_groups) sg = StreamGroup{};
            active_pipeline.reset();
            active_compute.reset();
        }
        void ensure_render() {  // _ensurePresentRenderPass
            if (rpass) return;
            end_active();
            s.refresh_depth_twins(enc);
            wgpu::RenderPassColorAttachment a = s.swapchain_attachment(
                target, render_pass_opened ? LoadOp::Load : LoadOp::Clear, StoreOp::Store, s.clear_color);
            wgpu::RenderPassDescriptor rp{};
            rp.colorAttachmentCount = 1;
            rp.colorAttachments = &a;
            wgpu::PassTimestampWrites tw{};
            if (s.timer_pass(*this, d3d12_guest::GpuPassKind::Render, s.lookahead_hash(*this, idx, false), &tw))
                rp.timestampWrites = &tw;
            seg_begin(tw, rp.timestampWrites != nullptr, s.lookahead_hash(*this, idx, false));
            rpass = enc.BeginRenderPass(&rp);
            render_pass_opened = true;
        }
        void ensure_compute() {
            if (cpass) return;
            end_active();
            s.refresh_depth_twins(enc);
            wgpu::ComputePassDescriptor cd{};
            wgpu::PassTimestampWrites tw{};
            if (s.timer_pass(*this, d3d12_guest::GpuPassKind::Compute, s.lookahead_hash(*this, idx, true), &tw))
                cd.timestampWrites = &tw;
            seg_begin(tw, cd.timestampWrites != nullptr, s.lookahead_hash(*this, idx, true));
            cpass = enc.BeginComputePass(&cd);
        }
        std::optional<AttachmentSize> swapchain_size() const { return AttachmentSize{s.stream_w, s.stream_h}; }
        // _getActivePassTargetSize
        std::optional<AttachmentSize> target_size() const {
            if (explicit_pass) return explicit_size;
            return swapchain_size();
        }
    };

    void begin_explicit_pass(Encoder& e, const BeginPassCmd& b) {
        if (show_spec) {
            PassTarget pt;
            pt.pass = frame_pass_count++;
            for (uint32_t i = 0; i < b.color_count && !pt.color_id; ++i) {
                if (b.colors[i].swapchain || b.colors[i].texture_id == 0) continue;
                pt.color_id = b.colors[i].texture_id;
                pt.color_mip = b.colors[i].base_mip;
                pt.color_layer = b.colors[i].base_layer;
            }
            if (b.has_ds && b.ds.texture_id != 0) {
                pt.depth_id = b.ds.texture_id;
                pt.depth_mip = b.ds.view.base_mip_level;
                pt.depth_layer = b.ds.view.base_array_layer;
            }
            frame_passes.push_back(pt);
        }
        e.end_active();
        refresh_depth_twins(e.enc);
        if (b.has_ds && !b.ds.depth_read_only && b.ds.depth_load_op != 0) {
            auto wit = textures.find(b.ds.texture_id);
            if (wit != textures.end()) ++wit->second.depth_gen;
        }
        std::array<AttachmentSize, kGpuMaxColorAttachments> sizes{};
        RenderPassPlan plan;
        plan.color_count = b.color_count;
        for (uint32_t i = 0; i < b.color_count; ++i) {
            plan.colors[i].texture_id = b.colors[i].texture_id;
            sizes[i] = b.colors[i].swapchain ? AttachmentSize{stream_w, stream_h} : b.colors[i].size;
        }
        plan.has_depth_stencil = b.has_ds;
        const uint32_t dropped = drop_mismatched_colors(plan, sizes, b.ds_size);
        // sanitizeRenderPassAttachments reference size: first color with a known size, else depth.
        AttachmentSize ref{};
        for (uint32_t i = 0; i < b.color_count; ++i)
            if (b.colors[i].texture_id != 0 && sizes[i].known()) { ref = sizes[i]; break; }
        if (!ref.known() && b.has_ds) ref = b.ds_size;
        bool drop_ds = b.has_ds && ref.known() && b.ds_size.known() && !(b.ds_size == ref);

        std::array<wgpu::RenderPassColorAttachment, kGpuMaxColorAttachments> colors{};
        for (uint32_t i = 0; i < b.color_count; ++i) {
            const ColorAtt& c = b.colors[i];
            if (c.texture_id == 0 || (dropped & (1u << i))) {
                if (dropped & (1u << i)) ++stats.render_pass_dropped_colors;
                continue;
            }
            if (c.swapchain) {
                colors[i] = swapchain_attachment(e.target, c.load, c.store, c.clear);
            } else {
                colors[i].view = c.view;
                colors[i].loadOp = c.load == LoadOp::Clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
                colors[i].storeOp = c.store == StoreOp::Discard ? wgpu::StoreOp::Discard : wgpu::StoreOp::Store;
                if (c.load == LoadOp::Clear) colors[i].clearValue = {c.clear.r, c.clear.g, c.clear.b, c.clear.a};
            }
        }
        wgpu::RenderPassDepthStencilAttachment ds{};
        wgpu::RenderPassDescriptor rp{};
        rp.colorAttachmentCount = b.color_count;
        rp.colorAttachments = colors.data();
        if (b.has_ds && !drop_ds) {
            const GpuRenderPassDepthStencilAttachment& d = b.ds;
            auto lop = [](uint8_t v) {
                return v == 1 ? wgpu::LoadOp::Load : v == 2 ? wgpu::LoadOp::Clear : wgpu::LoadOp::Undefined;
            };
            auto sop = [](uint8_t v) {
                return v == 1 ? wgpu::StoreOp::Store : v == 2 ? wgpu::StoreOp::Discard : wgpu::StoreOp::Undefined;
            };
            ds.view = b.ds_view;
            if (d.depth_load_op != 0 && !d.depth_read_only) {
                ds.depthLoadOp = lop(d.depth_load_op);
                ds.depthStoreOp = sop(d.depth_store_op);
                if (ds.depthLoadOp == wgpu::LoadOp::Clear) {
                    const double c = std::min(1.0, std::max(0.0, finite_or(d.depth_clear_value, 1)));
                    ds.depthClearValue = static_cast<float>(or_zero(c));
                }
            }
            if (d.stencil_load_op != 0 && !d.stencil_read_only) {
                ds.stencilLoadOp = lop(d.stencil_load_op);
                ds.stencilStoreOp = sop(d.stencil_store_op);
                if (ds.stencilLoadOp == wgpu::LoadOp::Clear) ds.stencilClearValue = d.stencil_clear_value;
            }
            if (d.depth_read_only) ds.depthReadOnly = true;
            if (d.stencil_read_only) ds.stencilReadOnly = true;
            rp.depthStencilAttachment = &ds;
        } else if (drop_ds) {
            ++stats.render_pass_dropped_colors;
        }
        wgpu::PassTimestampWrites tw{};
        if (timer_pass(e, d3d12_guest::GpuPassKind::Render, lookahead_hash(e, e.idx + 1, false), &tw))
            rp.timestampWrites = &tw;
        e.seg_begin(tw, rp.timestampWrites != nullptr, lookahead_hash(e, e.idx + 1, false));
        e.rpass = e.enc.BeginRenderPass(&rp);
        e.render_pass_opened = true;
        e.explicit_pass = true;
        // BeginRenderPassCommand.run: the pass's target size is the sanitized reference size, else the
        // swapchain size (swapchain target) or the static attachment size.
        if (ref.known()) e.explicit_size = ref;
        else e.explicit_size = std::nullopt;
        // A new pass resets the compute filter state (_preparePassCommandsForFlush).
        e.had_explicit_compute = false;
        e.compute_valid = false;
    }

    void run_cmd(Encoder& e, const PassCmd& cmd) {
        if (const auto* b = std::get_if<BeginPassCmd>(&cmd)) {
            remap_pass_boundary(true);
            begin_explicit_pass(e, *b);
        } else if (std::get_if<EndPassCmd>(&cmd)) {
            remap_pass_boundary(false);
            if (e.rpass) e.end_active();  // EndRenderPass: `if (ctl.passKind === 'render') ctl.endPass()`
            e.explicit_pass = false;
            e.had_explicit_compute = false;
            e.compute_valid = false;
        } else if (const auto* sp = std::get_if<SetPipelineCmd>(&cmd)) {
            // A pipeline still waiting for its shaders is built first (within the budget): the remap
            // state below reads what the build decides.
            if (auto dit = render_pipelines.find(sp->id); dit != render_pipelines.end()) settle_budgeted(dit->second);
            remap_set_render_pipeline(sp->id);
            e.ensure_render();
            e.active_pipeline = sp->id;
            e.had_explicit_compute = false;
            e.compute_valid = false;
            auto it = render_pipelines.find(sp->id);
            const wgpu::RenderPipeline* plp = it != render_pipelines.end() ? resolve_budgeted(it->second) : &sp->at_packet;
            if (!plp) {
                // Still compiling past the budget (or failed): draw_allowed() drops the draws.
            } else if (*plp) {
                e.rpass.SetPipeline(*plp);
            } else {
                ++stats.op_errors;  // SetPipeline: render pipeline not found
            }
            if (e.seg_open) e.seg_switch(it != render_pipelines.end() ? it->second.perf_hash : 0, d3d12_guest::GpuPassKind::Render);
        } else if (const auto* vb = std::get_if<SetVertexBufferCmd>(&cmd)) {
            e.ensure_render();
            auto it = buffers.find(vb->id);
            if (it != buffers.end()) e.rpass.SetVertexBuffer(vb->slot, it->second.buffer, vb->offset, wgpu::kWholeSize);
        } else if (const auto* ib = std::get_if<SetIndexBufferCmd>(&cmd)) {
            e.ensure_render();
            auto it = buffers.find(ib->id);
            if (it != buffers.end())
                e.rpass.SetIndexBuffer(it->second.buffer,
                                       ib->format == 1 ? wgpu::IndexFormat::Uint32 : wgpu::IndexFormat::Uint16, ib->offset,
                                       wgpu::kWholeSize);
        } else if (const auto* d = std::get_if<DrawCmd>(&cmd)) {
            e.had_explicit_compute = false;
            e.compute_valid = false;
            remap_before_draw(e);
            if (draw_allowed(e)) {
                e.rpass.Draw(d->vertex_count, d->instance_count, d->first_vertex, d->first_instance);
                ++stats.draws;
            }
        } else if (const auto* di = std::get_if<DrawIndexedCmd>(&cmd)) {
            e.had_explicit_compute = false;
            e.compute_valid = false;
            remap_before_draw(e);
            if (draw_allowed(e)) {
                e.rpass.DrawIndexed(di->index_count, di->instance_count, di->first_index, di->base_vertex,
                                    di->first_instance);
                ++stats.draws;
            }
        } else if (const auto* in = std::get_if<IndirectCmd>(&cmd)) {
            auto it = buffers.find(in->id);
            if (it == buffers.end()) {
                ++stats.op_errors;  // indirect buffer not found
                return;
            }
            if (in->kind == 2) {
                if (e.compute_quarantined_hash) {
                    note_quarantined(e.compute_quarantined_hash, true);
                    return;
                }
                if (e.had_explicit_compute && !e.compute_valid && (e.compute_pending || e.compute_failed)) {
                    // As a direct dispatch: no pipeline was set (still compiling past the budget / failed).
                    ++stats.dispatches_skipped;
                    if (e.compute_failed) note_pipeline_failed(e.compute_pending_hash, true);
                    else note_pipeline_pending(e.compute_pending_hash, true);
                    return;
                }
                e.ensure_compute();
                bind_compute_fit_dummies(e);
                e.cpass.DispatchWorkgroupsIndirect(it->second.buffer, in->offset);
                ++stats.dispatches;
            } else if (draw_allowed(e)) {
                if (in->kind == 1) e.rpass.DrawIndexedIndirect(it->second.buffer, in->offset);
                else e.rpass.DrawIndirect(it->second.buffer, in->offset);
                ++stats.draws;
            }
        } else if (const auto* sc = std::get_if<SetComputePipelineCmd>(&cmd)) {
            if (auto dit = compute_pipelines.find(sc->id); dit != compute_pipelines.end()) settle_budgeted(dit->second);
            remap_set_compute_pipeline(sc->id);
            e.ensure_compute();
            e.active_compute = sc->id;
            e.had_explicit_compute = true;
            auto it = compute_pipelines.find(sc->id);
            e.compute_valid = it != compute_pipelines.end();
            e.compute_pending_hash = 0;
            e.compute_pending = false;
            e.compute_failed = false;
            e.compute_quarantined_hash = 0;
            if (!options.quarantine.empty()) {
                auto qh = compute_hash.find(sc->id);
                if (qh != compute_hash.end() && options.quarantine.contains(qh->second))
                    e.compute_quarantined_hash = qh->second;
            }
            if (e.compute_valid) {
                if (const wgpu::ComputePipeline* cp = resolve_budgeted(it->second)) {
                    e.cpass.SetPipeline(*cp);
                } else {
                    e.compute_valid = false;
                    e.compute_failed = it->second.failed;
                    e.compute_pending = !e.compute_failed;
                    auto hit = compute_hash.find(sc->id);
                    e.compute_pending_hash = hit != compute_hash.end() ? hit->second : 0;
                }
            }
            if (e.seg_open) {
                auto hit = compute_hash.find(sc->id);
                e.seg_switch(hit != compute_hash.end() ? hit->second : 0, d3d12_guest::GpuPassKind::Compute);
            }
        } else if (const auto* dc = std::get_if<DispatchCmd>(&cmd)) {
            if (e.had_explicit_compute && e.compute_quarantined_hash) {
                note_quarantined(e.compute_quarantined_hash, true);
                return;
            }
            if (e.had_explicit_compute && !e.compute_valid) {
                ++stats.dispatches_skipped;
                if (e.compute_failed) note_pipeline_failed(e.compute_pending_hash, true);
                else if (e.compute_pending) note_pipeline_pending(e.compute_pending_hash, true);
                return;
            }
            // validateComputeDispatch
            if (dc->x == 0 || dc->y == 0 || dc->z == 0) {
                ++stats.dispatches_skipped;
                return;
            }
            uint32_t x = dc->x, y = dc->y, z = dc->z;
            if (x > max_workgroups_per_dim || y > max_workgroups_per_dim || z > max_workgroups_per_dim) {
                ++stats.dispatches_clamped;
                x = std::min(x, max_workgroups_per_dim);
                y = std::min(y, max_workgroups_per_dim);
                z = std::min(z, max_workgroups_per_dim);
            }
            remap_before_dispatch(e);
            e.ensure_compute();
            bind_compute_fit_dummies(e);
            e.cpass.DispatchWorkgroups(x, y, z);
            ++stats.dispatches;
        } else if (const auto* vp = std::get_if<ViewportCmd>(&cmd)) {
            if (!e.rpass) return;
            const auto ts = e.target_size();
            const Viewport c = clamp_viewport_to_target(vp->vp, ts ? ts->width : 0, ts ? ts->height : 0);
            e.rpass.SetViewport(static_cast<float>(c.x), static_cast<float>(c.y), static_cast<float>(c.width),
                                static_cast<float>(c.height), static_cast<float>(c.min_depth),
                                static_cast<float>(c.max_depth));
        } else if (const auto* sr = std::get_if<ScissorCmd>(&cmd)) {
            if (!e.rpass) return;
            const auto ts = e.target_size();
            const Rect c = clamp_scissor_to_target(sr->x, sr->y, sr->w, sr->h, ts ? ts->width : 0, ts ? ts->height : 0);
            e.rpass.SetScissorRect(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y), static_cast<uint32_t>(c.width),
                                   static_cast<uint32_t>(c.height));
        } else if (const auto* bc = std::get_if<BlendConstantCmd>(&cmd)) {
            if (!e.rpass) return;
            const wgpu::Color color{bc->r, bc->g, bc->b, bc->a};
            e.rpass.SetBlendConstant(&color);
        } else if (const auto* st = std::get_if<StencilRefCmd>(&cmd)) {
            if (!e.rpass) return;
            e.rpass.SetStencilReference(st->ref);
        } else if (const auto* bg = std::get_if<SetBindGroup2Cmd>(&cmd)) {
            if (bg->group == 0 && remap_set_group0(e, *bg)) return;  // layout:'auto' remap
            if (!e.rpass && !e.cpass) return;  // `if (!ctl.pass) return`
            if (!bg->bind_group) {  // a group of an over-limit layout: only a fitted pipeline can use it (remap)
                ++stats.over_limit_binds_skipped;
                return;
            }
            if (e.rpass) e.rpass.SetBindGroup(bg->group, bg->bind_group, bg->offsets.size(), bg->offsets.data());
            else e.cpass.SetBindGroup(bg->group, bg->bind_group, bg->offsets.size(), bg->offsets.data());
            if (bg->group == 0) e.group0_bound = true;
            if (bg->group < 8) e.bound_groups |= static_cast<uint8_t>(1u << bg->group);
            if (bg->group > 0 && bg->group < kFitMaxGroups) {
                e.stream_groups[bg->group] = {bg->bind_group, bg->offsets};
                e.overridden &= static_cast<uint8_t>(~(1u << bg->group));
            }
        } else if (const auto* b1 = std::get_if<SetBindGroup1Cmd>(&cmd)) {
            // `this.bindGroups.get(id) || cmd.bindGroup`; a missing group is a validation error.
            auto it = bind_groups.find(b1->id);
            wgpu::BindGroup bg = it != bind_groups.end() ? it->second : b1->at_packet;
            if (!bg) {
                ++stats.op_errors;  // bind group not found
                return;
            }
            if (!e.rpass && !e.cpass) {
                if (compute_bind_groups.count(b1->id)) e.ensure_compute();
                else e.ensure_render();
            }
            if (e.rpass) e.rpass.SetBindGroup(b1->group, bg, 0, nullptr);
            else e.cpass.SetBindGroup(b1->group, bg, 0, nullptr);
            if (b1->group == 0) e.group0_bound = true;
            if (b1->group < 8) e.bound_groups |= static_cast<uint8_t>(1u << b1->group);
            if (b1->group > 0 && b1->group < kFitMaxGroups) {
                e.stream_groups[b1->group] = {bg, {}};
                e.overridden &= static_cast<uint8_t>(~(1u << b1->group));
            }
        } else if (const auto* rb = std::get_if<ReadbackCmd>(&cmd)) {
            e.end_active();  // _recordReadbackCopy: pc.endPass()
            record_readback_copy(e.enc, rb->req);
        } else if (const auto* ct = std::get_if<CopyTextureCmd>(&cmd)) {
            e.end_active();
            record_texture_copy(e.enc, ct->copy);
        }
    }


    // ---- layout:'auto' bind group remap (_getOrCreateRemappedBindGroup / _preparePassCommandsForFlush) ----

    // pipelineCache.getOrCreateDefaultSampler('filtering' | 'comparison')
    wgpu::Sampler fallback_sampler(bool comparison) { return fallback_sampler_kind(comparison ? 2 : 0); }
    // kind: 0 filtering, 1 non-filtering, 2 comparison (DEFAULT_SAMPLER_DESCRIPTORS)
    wgpu::Sampler fallback_sampler_kind(int kind) {
        wgpu::Sampler& s = fallback_samplers[kind];
        if (!s) {
            wgpu::SamplerDescriptor sd{};
            const bool nearest = kind == 1;
            sd.magFilter = nearest ? wgpu::FilterMode::Nearest : wgpu::FilterMode::Linear;
            sd.minFilter = nearest ? wgpu::FilterMode::Nearest : wgpu::FilterMode::Linear;
            sd.mipmapFilter = kind == 0 ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest;
            if (kind == 2) sd.compare = wgpu::CompareFunction::LessEqual;
            s = device.CreateSampler(&sd);
        }
        return s;
    }

    struct RemapResult {
        enum class Status { Ok, Empty, Unsupported } status = Status::Unsupported;
        wgpu::BindGroup group;
    };

    // Builds (or reuses) the bind group for the auto layout's group 0 from the recorded descriptor of
    // the group the stream bound: each WGSL binding takes the first matching real resource, else a
    // default sampler / a dummy. Dynamic offsets are folded into the static buffer offsets.
    RemapResult get_remapped(bool compute, uint32_t pipeline_id, const AutoLayoutMeta& meta, uint32_t bg_id,
                             const std::shared_ptr<BgDesc>& desc, const std::vector<uint32_t>& dyn) {
        RemapResult out;
        const bool has_pipeline = compute ? compute_pipelines.count(pipeline_id) != 0 : render_pipelines.count(pipeline_id) != 0;
        if (!has_pipeline) return out;  // 'unsupported': the original command stays
        out.status = RemapResult::Status::Empty;
        if (meta.bindings.empty()) return out;

        static const std::vector<RemapResource> kNone;
        const std::vector<RemapResource>& av = desc ? desc->model : kNone;
        std::string dyn_key;
        {
            bool any = false;
            for (uint32_t v : dyn) any = any || v > 0;
            if (any) {
                bool non_zero = false;
                std::string parts;
                for (const RemapResource& r : av) {
                    if (!r.has_dynamic_offset || r.is_dummy) continue;
                    const uint32_t v = r.dynamic_offset_index < dyn.size() ? dyn[r.dynamic_offset_index] : 0;
                    if (v > 0) non_zero = true;
                    if (!parts.empty()) parts += ',';
                    parts += std::to_string(r.binding) + "=" + std::to_string(v);
                }
                if (non_zero) dyn_key = parts;
            }
        }
        auto dit = bg_descs.find(bg_id);
        const bool live = bg_id == 0 || (dit != bg_descs.end() && dit->second == desc);
        const std::string key = std::string(compute ? "c:" : "r:") + std::to_string(pipeline_id) + ":" +
                                std::to_string(bg_id) + (dyn_key.empty() ? "" : "@" + dyn_key);
        if (live) {
            auto cit = remap_cache.find(key);
            if (cit != remap_cache.end() && cit->second.desc == desc.get() && cit->second.gen == meta.gen) {
                ++stats.remap_cache_hits;
                if (cit->second.group) {
                    out.status = RemapResult::Status::Ok;
                    out.group = cit->second.group;
                }
                return out;
            }
        }

        wgpu::BindGroupLayout auto_bgl = meta.reflected_bgl0;
        if (!auto_bgl) {
            // layout:'auto' pipeline without reflection: its layout needs the compiled pipeline. Budgeted; past
            // it nothing is bound (the draw / dispatch is skipped as pipeline_pending) and nothing is cached.
            if (compute) {
                const wgpu::ComputePipeline* cp = resolve_budgeted(compute_pipelines.find(pipeline_id)->second);
                if (!cp) { note_pipeline_pending(0, true); return out; }
                auto_bgl = cp->GetBindGroupLayout(0);
            } else {
                const RenderPipelineEntry& re = render_pipelines.find(pipeline_id)->second;
                const wgpu::RenderPipeline* rp = resolve_budgeted(re);
                if (!rp) { if (re.failed) note_pipeline_failed(re.perf_hash, false); else note_pipeline_pending(re.perf_hash, false); return out; }
                auto_bgl = rp->GetBindGroupLayout(0);
            }
        }
        wgpu::BindGroup group;
        if (auto_bgl) {
            RemapPicker picker(av, dyn, features.float32_filterable);
            std::vector<wgpu::BindGroupEntry> wes(meta.bindings.size());
            for (size_t i = 0; i < meta.bindings.size(); ++i) {
                const WgslBinding& b = meta.bindings[i];
                wgpu::BindGroupEntry& w = wes[i];
                w.binding = b.binding;
                const PickResult r = meta.by_binding ? picker.pick_binding(b) : picker.pick(b);
                if (r.kind == PickResult::Kind::Resource) {
                    const BgHandles& hnd = desc->handles[r.index];
                    w.buffer = hnd.buffer;
                    w.textureView = hnd.view;
                    w.sampler = hnd.sampler;
                    if (hnd.buffer) {
                        w.offset = r.offset;
                        w.size = r.size > 0 ? r.size : wgpu::kWholeSize;
                        if (r.folded) ++stats.remap_dynamic_offsets;
                        if (r.dyn_clamped) ++stats.remap_dynamic_clamps;
                    }
                } else if (r.kind == PickResult::Kind::FallbackSampler) {
                    w.sampler = fallback_sampler(b.comparison);
                    ++stats.remap_fallback_samplers;
                } else {
                    if (r.dyn_oob) ++stats.remap_dynamic_clamps;
                    const SynthLayoutEntry se = synth_layout_entry(b);
                    LayoutEntry le;
                    le.binding = se.binding;
                    le.kind = se.kind;
                    le.subtype = se.subtype;
                    le.view_dim = se.view_dim;
                    le.storage_format = se.storage_format;
                    le.multisampled = se.multisampled;
                    le.min_binding_size = se.min_binding_size;
                    for (const LayoutEntry& x : meta.by_binding_entries)
                        if (x.binding == b.binding) {
                            le = x;
                            le.min_binding_size = std::max<uint64_t>(x.min_binding_size, se.min_binding_size);
                        }
                    const Dummy d = dummy_resource(le);
                    w.buffer = d.buffer;
                    w.sampler = d.sampler;
                    w.textureView = d.view;
                    w.offset = 0;
                    w.size = wgpu::kWholeSize;
                    ++stats.remap_dummies;
                }
            }
            wgpu::BindGroupDescriptor bd{};
            bd.layout = auto_bgl;
            bd.entryCount = wes.size();
            bd.entries = wes.data();
            group = device.CreateBindGroup(&bd);
            ++stats.remapped_bind_groups;
        }
        if (live) {
            if (remap_cache.size() >= 8192) remap_cache.clear();
            remap_cache[key] = RemapCacheEntry{group, desc.get(), meta.gen};
        }
        if (group) {
            out.status = RemapResult::Status::Ok;
            out.group = group;
        }
        return out;
    }

    // Before a draw / dispatch: the fitted groups the pass has not bound get their dummies.
    // Every draw / dispatch: a group a padded pipeline bound over the stream's own is restored first
    // (unless this pipeline pads it too).
    void bind_fit_dummies(Encoder& e, const FitDummies& fd) {
        if (!e.rpass && !e.cpass) return;
        for (uint32_t g = 1; g < kFitMaxGroups && e.overridden; ++g) {
            const uint8_t bit = static_cast<uint8_t>(1u << g);
            if (!(e.overridden & bit) || (fd.force & bit)) continue;
            e.overridden &= static_cast<uint8_t>(~bit);
            const auto& sg = e.stream_groups[g];
            if (!sg.group) continue;
            if (e.rpass) e.rpass.SetBindGroup(g, sg.group, sg.offsets.size(), sg.offsets.data());
            else e.cpass.SetBindGroup(g, sg.group, sg.offsets.size(), sg.offsets.data());
            ++stats.layout_fit_restores;
        }
        if (!fd.mask) return;
        for (uint32_t g = 0; g < kFitMaxGroups; ++g) {
            const uint8_t bit = static_cast<uint8_t>(1u << g);
            const bool force = (fd.force & bit) != 0;
            if (!(fd.mask & bit) || !fd.groups[g]) continue;
            if (!force && (e.bound_groups & bit)) continue;
            if (force && (e.overridden & bit)) continue;  // still bound from the previous draw
            if (e.rpass) e.rpass.SetBindGroup(g, fd.groups[g], 0, nullptr);
            else e.cpass.SetBindGroup(g, fd.groups[g], 0, nullptr);
            e.bound_groups |= bit;
            if (force) e.overridden |= bit;
            ++stats.layout_fit_dummy_binds;
        }
    }

    void bind_compute_fit_dummies(Encoder& e) {
        static const FitDummies kNone;
        if (!e.active_compute || !e.compute_valid) return bind_fit_dummies(e, kNone);
        auto it = compute_pipelines.find(*e.active_compute);
        bind_fit_dummies(e, it != compute_pipelines.end() ? it->second.fit : kNone);
    }

    // RemappedSetBindGroupCommand.run: `if (!ctl.pass) return`.
    void apply_remapped(Encoder& e, const wgpu::BindGroup& g) {
        if (!e.rpass && !e.cpass) return;
        if (e.rpass) {
            e.rpass.SetBindGroup(0, g, 0, nullptr);
            e.group0_bound = true;  // _recordPassBindGroup
        } else {
            e.cpass.SetBindGroup(0, g, 0, nullptr);
        }
        e.bound_groups |= 1u;
    }

    void remap_reset_compute() {
        remap.compute_pipe = 0;
        remap.compute_valid = false;
        remap.compute_auto = false;
        remap.had_explicit_compute = false;
        remap.group0_compute = false;
        remap.has_last_compute = false;
    }
    void remap_reset_render() {
        remap.render_pipe = 0;
        remap.render_auto = false;
        remap.group0_render = false;
        remap.has_last_render = false;
    }
    void remap_pass_boundary(bool begin) {
        remap.mode = begin ? 1 : 0;
        remap_reset_compute();
        remap_reset_render();
    }
    void remap_set_render_pipeline(uint32_t id) {
        remap.mode = 1;
        remap_reset_compute();
        remap.render_pipe = id;
        auto it = render_pipelines.find(id);
        remap.render_auto = it != render_pipelines.end() && it->second.auto_meta.remap;
        remap.group0_render = false;
    }
    void remap_set_compute_pipeline(uint32_t id) {
        remap.mode = 2;
        remap_reset_render();
        remap.had_explicit_compute = true;
        remap.compute_pipe = id;
        remap.compute_valid = compute_pipelines.count(id) != 0;
        auto it = compute_meta.find(id);
        remap.compute_auto = remap.compute_valid && it != compute_meta.end() && it->second.remap;
        remap.group0_compute = false;
    }

    // SetBindGroup2 of group 0: true = the command was replaced (the original is not run).
    bool remap_set_group0(Encoder& e, const SetBindGroup2Cmd& c) {
        const Group0Src src{c.bind_group_id, c.desc, c.offsets};
        if (remap.mode == 2) {
            remap.last_compute = src;
            remap.has_last_compute = true;
            if (remap.compute_valid && remap.compute_auto) {
                auto mit = compute_meta.find(remap.compute_pipe);
                const RemapResult r = get_remapped(true, remap.compute_pipe, mit->second, c.bind_group_id, c.desc, c.offsets);
                remap.group0_compute = true;
                if (r.status == RemapResult::Status::Ok) {
                    apply_remapped(e, r.group);
                    return true;
                }
                if (r.status == RemapResult::Status::Empty) return true;
            }
        } else if (remap.mode == 1) {
            remap.last_render = src;
            remap.has_last_render = true;
            if (remap.render_auto) {
                auto pit = render_pipelines.find(remap.render_pipe);
                if (pit != render_pipelines.end()) {
                    const RemapResult r =
                        get_remapped(false, remap.render_pipe, pit->second.auto_meta, c.bind_group_id, c.desc, c.offsets);
                    remap.group0_render = true;
                    if (r.status == RemapResult::Status::Ok) {
                        apply_remapped(e, r.group);
                        return true;
                    }
                    if (r.status == RemapResult::Status::Empty) return true;  // NoopRenderGroup0Command
                } else {
                    remap.group0_render = true;
                }
            }
        } else {
            remap.last_render = src;
            remap.has_last_render = true;
            remap.last_compute = src;
            remap.has_last_compute = true;
        }
        return false;
    }

    // A draw / dispatch of an auto-layout pipeline with no group 0 set yet in this pass binds the
    // remap of the last group 0 the stream bound (or none: dummies).
    void remap_before_draw(Encoder& e) {
        remap.mode = 1;
        remap_reset_compute();
        if (!remap.render_auto || remap.group0_render) return;
        auto pit = render_pipelines.find(remap.render_pipe);
        const Group0Src none;
        const Group0Src& src = remap.has_last_render ? remap.last_render : none;
        remap.group0_render = true;
        if (pit == render_pipelines.end()) return;
        const RemapResult r = get_remapped(false, remap.render_pipe, pit->second.auto_meta, src.id, src.desc, src.dyn);
        if (r.status == RemapResult::Status::Ok) apply_remapped(e, r.group);
    }
    void remap_before_dispatch(Encoder& e) {
        if (!remap.compute_auto || remap.group0_compute) return;
        auto mit = compute_meta.find(remap.compute_pipe);
        const Group0Src none;
        const Group0Src& src = remap.has_last_compute ? remap.last_compute : none;
        remap.group0_compute = true;
        if (mit == compute_meta.end()) return;
        const RemapResult r = get_remapped(true, remap.compute_pipe, mit->second, src.id, src.desc, src.dyn);
        if (r.status == RemapResult::Status::Ok) apply_remapped(e, r.group);
    }

    // ---- GPU pass timestamps (gpu_timer.js, #1608) ----

    // The key of the pass that is about to open: the pipeline its first SetPipeline /
    // SetComputePipeline selects (a pass can only be timed as a whole; WebGPU has no timestamp
    // writes inside a pass). Render pipelines count under the fragment shader's hash (else the
    // vertex shader's), compute pipelines under the compute shader's; 0 = no pipeline or an inline
    // WGSL shader. `from` is the first command that belongs to the pass.
    uint64_t lookahead_hash(const Encoder& e, size_t from, bool compute) const {
        if (!e.cmds) return 0;
        const std::vector<PassCmd>& cmds = *e.cmds;
        for (size_t j = from; j < cmds.size(); ++j) {
            const PassCmd& c = cmds[j];
            if (std::get_if<BeginPassCmd>(&c) || std::get_if<EndPassCmd>(&c) || std::get_if<ReadbackCmd>(&c) ||
                std::get_if<CopyTextureCmd>(&c))
                break;
            const auto* ind = std::get_if<IndirectCmd>(&c);
            const bool compute_op = std::get_if<SetComputePipelineCmd>(&c) || std::get_if<DispatchCmd>(&c) ||
                                    (ind && ind->kind == 2);
            const bool render_op = std::get_if<SetPipelineCmd>(&c) || std::get_if<SetVertexBufferCmd>(&c) ||
                                   std::get_if<SetIndexBufferCmd>(&c) || std::get_if<DrawCmd>(&c) ||
                                   std::get_if<DrawIndexedCmd>(&c) || (ind && ind->kind != 2);
            if (compute ? render_op : compute_op) break;
            if (compute) {
                if (const auto* sc = std::get_if<SetComputePipelineCmd>(&c)) {
                    auto it = compute_hash.find(sc->id);
                    return it != compute_hash.end() ? it->second : 0;
                }
            } else if (const auto* sp = std::get_if<SetPipelineCmd>(&c)) {
                auto it = render_pipelines.find(sp->id);
                return it != render_pipelines.end() ? it->second.perf_hash : 0;
            }
        }
        return 0;
    }

    // Reserves the query pair of one pass in this submit's ring slot; false = the pass goes untimed.
    bool timer_pass(Encoder& e, d3d12_guest::GpuPassKind kind, uint64_t hash, wgpu::PassTimestampWrites* out) {
        if (!time_ring) return false;
        if (e.timer_slot == -2) {
            e.timer_slot = time_ring->begin_frame();
            if (e.timer_slot < 0) {
                ++stats.timer_frames_skipped;
                time_table->note_skipped_frame();
            }
        }
        if (e.timer_slot < 0) return false;
        uint32_t b = 0, en = 0;
        if (!time_ring->add_pass(hash, kind, &b, &en)) return false;
        out->querySet = timer_slots[static_cast<size_t>(e.timer_slot)].queries;
        out->beginningOfPassWriteIndex = b;
        out->endOfPassWriteIndex = en;
        ++e.timer_passes;
        ++stats.timer_passes;
        return true;
    }

    // Before encoder.Finish(): resolves the slot's queries and copies them to the mappable buffer.
    void timer_resolve(Encoder& e) {
        if (e.timer_slot < 0) return;
        const size_t n = time_ring->end_frame(e.timer_slot);
        if (n == 0) {
            time_ring->abandon(e.timer_slot, *time_table);
            e.timer_slot = -1;
            return;
        }
        TimerSlot& ts = timer_slots[static_cast<size_t>(e.timer_slot)];
        e.enc.ResolveQuerySet(ts.queries, 0, static_cast<uint32_t>(n * 2), ts.resolve, 0);
        e.enc.CopyBufferToBuffer(ts.resolve, 0, ts.staging, 0, n * 2 * 8);
        ++stats.timer_frames;
        e.timer_passes = n;
    }

    // After queue.Submit(): maps the slot; the table is fed when the GPU is done (poll / wait_idle).
    void timer_collect(const Encoder& e) {
        if (e.timer_slot < 0) return;
        struct Pending {
            Impl* self;
            int slot;
            size_t count;
        };
        auto* pend = new Pending{this, e.timer_slot, e.timer_passes * 2};
        TimerSlot& ts = timer_slots[static_cast<size_t>(e.timer_slot)];
        ++pending_maps;
        ts.staging.MapAsync(
            wgpu::MapMode::Read, 0, pend->count * 8, wgpu::CallbackMode::AllowProcessEvents,
            [](wgpu::MapAsyncStatus st, wgpu::StringView, Pending* pd) {
                Impl& s = *pd->self;
                TimerSlot& slot = s.timer_slots[static_cast<size_t>(pd->slot)];
                if (st == wgpu::MapAsyncStatus::Success) {
                    const auto* ticks = static_cast<const uint64_t*>(slot.staging.GetConstMappedRange(0, pd->count * 8));
                    // Dawn resolves timestamps to nanoseconds: the period is 1.
                    s.time_ring->complete(pd->slot, ticks, ticks ? pd->count : 0, 1.0, *s.time_table);
                    slot.staging.Unmap();
                } else {
                    s.time_ring->abandon(pd->slot, *s.time_table);
                }
                --s.pending_maps;
                delete pd;
            },
            pend);
    }

    // ---- read-backs (wgpu_worker.js _recordReadbackCopy / _startRecordedReadbacks) ----

    // Same format up to sRGB-ness (copyTextureToTexture's copy-compatibility).
    static bool copy_compatible_formats(uint8_t a, uint8_t b) {
        if (a == b) return true;
        std::string x = format_info(a).name, y = format_info(b).name;
        for (std::string* f : {&x, &y}) {
            const size_t k = f->find("-srgb");
            if (k != std::string::npos) f->erase(k, 5);
        }
        return x == y;
    }

    // CopyTexture (opcode 62): validated against the tracked geometry (a bad copy is a packet error,
    // never a Dawn validation error), then recorded as copyTextureToTexture.
    void record_texture_copy(wgpu::CommandEncoder& enc, const GpuCopyTexturePayload& c) {
        const uint16_t op = static_cast<uint16_t>(kGpuOpcodeCopyTexture);
        const auto di = textures.find(c.dst_id);
        const auto si = textures.find(c.src_id);
        if (di == textures.end() || si == textures.end()) {
            packet_error(op, "CopyTexture: unknown texture " + std::to_string(di == textures.end() ? c.dst_id : c.src_id));
            return;
        }
        TextureEntry& dt = di->second;
        const TextureEntry& st = si->second;
        const TextureGeometry& dg = dt.geom;
        const TextureGeometry& sg = st.geom;
        auto bad = [&](const char* why) { packet_error(op, std::string("CopyTexture: ") + why); };
        if (dg.dimension != sg.dimension || dt.promoted_1d || st.promoted_1d || dt.bc3d_fallback || st.bc3d_fallback) {
            return bad("textures are not both 2D or both 3D");
        }
        if (!copy_compatible_formats(dg.format, sg.format) || dt.sample_count != st.sample_count) {
            return bad("formats or sample counts are not copy-compatible");
        }
        if (c.src_mip >= sg.mips || c.dst_mip >= dg.mips) return bad("mip level out of range");
        if (c.width == 0 || c.height == 0 || c.depth_or_layers == 0) return;  // empty: nothing to copy
        const FormatInfo& fi = format_info(dg.format);
        const bool whole_only = format_is_depth_or_stencil(dg.format) || dt.sample_count > 1;
        const bool is3d = dg.dimension == 2;
        const uint32_t bw = fi.compressed ? fi.block_w : 1, bh = fi.compressed ? fi.block_h : 1;
        auto up = [](uint32_t v, uint32_t b) { return (v + b - 1) / b * b; };
        auto mip_dim = [](uint32_t v, uint32_t m) { return std::max<uint32_t>(1u, v >> m); };
        auto aligned = [](uint32_t v, uint32_t b) { return b <= 1 || v % b == 0; };
        // Compressed textures are addressed in their physical (block rounded) mip size.
        const uint32_t sw = up(mip_dim(sg.width, c.src_mip), bw), sh = up(mip_dim(sg.height, c.src_mip), bh);
        const uint32_t dw = up(mip_dim(dg.width, c.dst_mip), bw), dh = up(mip_dim(dg.height, c.dst_mip), bh);
        const uint32_t sd = is3d ? mip_dim(sg.depth_or_layers, c.src_mip) : sg.depth_or_layers;
        const uint32_t dd = is3d ? mip_dim(dg.depth_or_layers, c.dst_mip) : dg.depth_or_layers;
        const uint64_t sx = uint64_t{c.src_x} + c.width, sy = uint64_t{c.src_y} + c.height,
                       sz = uint64_t{c.src_z} + c.depth_or_layers;
        const uint64_t dx = uint64_t{c.dst_x} + c.width, dy = uint64_t{c.dst_y} + c.height,
                       dz = uint64_t{c.dst_z} + c.depth_or_layers;
        if (sx > sw || sy > sh || sz > sd || dx > dw || dy > dh || dz > dd) return bad("region out of range");
        if (!aligned(c.src_x, bw) || !aligned(c.dst_x, bw) || !aligned(c.src_y, bh) || !aligned(c.dst_y, bh) ||
            !aligned(c.width, bw) || !aligned(c.height, bh)) {
            return bad("compressed region is not block aligned");
        }
        if (whole_only && (c.width != sw || c.height != sh || c.depth_or_layers != sd || sw != dw || sh != dh ||
                           sd != dd || c.src_x || c.src_y || c.src_z || c.dst_x || c.dst_y || c.dst_z)) {
            return bad("depth/stencil or multisampled textures are copied whole");
        }
        wgpu::TexelCopyTextureInfo src{};
        src.texture = st.texture;
        src.mipLevel = c.src_mip;
        src.origin = {c.src_x, c.src_y, c.src_z};
        src.aspect = wgpu::TextureAspect::All;
        wgpu::TexelCopyTextureInfo dst{};
        dst.texture = dt.texture;
        dst.mipLevel = c.dst_mip;
        dst.origin = {c.dst_x, c.dst_y, c.dst_z};
        dst.aspect = wgpu::TextureAspect::All;
        const wgpu::Extent3D extent{c.width, c.height, c.depth_or_layers};
        enc.CopyTextureToTexture(&src, &dst, &extent);
        ++dt.depth_gen;
        ++stats.texture_copies;
    }

    void record_readback_copy(wgpu::CommandEncoder& enc, const ReadbackRequest& req) {
        Impl_ReadbackEntry entry;
        entry.req = req;
        wgpu::BufferDescriptor sd{};
        sd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        if (req.kind == ReadbackRequest::Kind::Buffer) {
            auto it = buffers.find(req.id);
            ReadbackLayout l = readback_staging_layout(req);
            if (it != buffers.end() && l.valid) {
                const uint64_t src_size = it->second.size;
                if (l.copy_offset + l.staging_size > src_size)
                    l.staging_size = (src_size > l.copy_offset ? src_size - l.copy_offset : 0) & ~uint64_t{3};
                if (l.staging_size > 0 && l.skip + req.size <= l.staging_size) {
                    sd.size = l.staging_size;
                    entry.staging = device.CreateBuffer(&sd);
                    enc.CopyBufferToBuffer(it->second.buffer, l.copy_offset, entry.staging, 0, l.staging_size);
                    entry.layout = l;
                }
            }
        } else {
            auto it = textures.find(req.id);
            if (it != textures.end()) {
                TextureEntry& t = it->second;
                const ReadbackLayout l = readback_staging_layout(req, readback_block_height(t.geom.format, false));
                if (l.valid && l.row_bytes > 0 && static_cast<uint64_t>(l.row_bytes) <= req.bytes_per_row &&
                    l.staging_row <= 0xffffffffull) {
                    sd.size = l.staging_size;
                    entry.staging = device.CreateBuffer(&sd);
                    wgpu::TexelCopyTextureInfo src{};
                    src.texture = t.texture;
                    src.mipLevel = req.mip_level;
                    src.origin = {req.origin_x, req.origin_y, req.origin_z};
                    wgpu::TexelCopyBufferInfo dst{};
                    dst.buffer = entry.staging;
                    dst.layout.bytesPerRow = static_cast<uint32_t>(l.staging_row);
                    dst.layout.rowsPerImage = static_cast<uint32_t>(l.staging_rows_per_image);
                    const wgpu::Extent3D extent{req.width, req.height, std::max<uint32_t>(1, req.depth_or_layers)};
                    enc.CopyTextureToBuffer(&src, &dst, &extent);
                    entry.layout = l;
                }
            }
        }
        recorded_readbacks.push_back(std::move(entry));
    }

    void finish_readback(const Impl_ReadbackEntry& e, const uint8_t* mapped, size_t len) {
        if (deliver_readback(guest_mem, e.req, e.layout, mapped, len)) {
            ++stats.readbacks_delivered;
            stats.readback_bytes += e.req.size;
        } else {
            ++stats.readbacks_failed;
        }
    }

    // After the submit: map every staging buffer, then deliver (or fail) each read-back.
    void start_recorded_readbacks(bool submitted) {
        std::vector<Impl_ReadbackEntry> entries = std::move(recorded_readbacks);
        recorded_readbacks.clear();
        for (Impl_ReadbackEntry& e : entries) {
            if (!submitted || !e.staging) {
                finish_readback(e, nullptr, 0);
                if (e.staging) e.staging.Destroy();
                continue;
            }
            struct Pending {
                Impl* self;
                Impl_ReadbackEntry entry;
            };
            auto* pend = new Pending{this, std::move(e)};
            ++pending_maps;
            const uint64_t size = pend->entry.staging.GetSize();
            pend->entry.staging.MapAsync(
                wgpu::MapMode::Read, 0, static_cast<size_t>(size), wgpu::CallbackMode::AllowProcessEvents,
                [](wgpu::MapAsyncStatus st, wgpu::StringView, Pending* pd) {
                    Impl& s = *pd->self;
                    wgpu::Buffer& staging = pd->entry.staging;
                    if (st == wgpu::MapAsyncStatus::Success) {
                        const size_t n = static_cast<size_t>(staging.GetSize());
                        const auto* mapped = static_cast<const uint8_t*>(staging.GetConstMappedRange(0, n));
                        s.finish_readback(pd->entry, mapped, mapped ? n : 0);
                        staging.Unmap();
                    } else {
                        s.finish_readback(pd->entry, nullptr, 0);
                    }
                    staging.Destroy();
                    --s.pending_maps;
                    delete pd;
                },
                pend);
        }
    }

    // A frame that cannot be presented is dropped (the worker's `_resetFrameState`); its read-backs
    // complete as failed so the producer's fence hold is released.
    void fail_dropped_readbacks() {
        for (const PassCmd& c : pass_cmds) {
            if (const auto* rb = std::get_if<ReadbackCmd>(&c)) {
                Impl_ReadbackEntry e;
                e.req = rb->req;
                finish_readback(e, nullptr, 0);
            }
        }
    }

    // _flushReadbacksNow: submits everything recorded up to the last read-back (no present, no frame
    // accounting) so the maps can start while the title waits on the fence that guards the read-back.
    // The commands after it stay queued for the frame.
    void flush_readbacks_now() {
        readback_flush_wanted = false;
        if (lost) return;
        size_t last = pass_cmds.size();
        for (size_t i = pass_cmds.size(); i-- > 0;) {
            if (std::get_if<ReadbackCmd>(&pass_cmds[i])) {
                last = i;
                break;
            }
        }
        if (last == pass_cmds.size()) return;
        if (!acquire_target()) {  // `_notePresentSubmitFailure`: the commands stay queued
            start_recorded_readbacks(false);
            return;
        }
        std::vector<PassCmd> head(std::make_move_iterator(pass_cmds.begin()),
                                  std::make_move_iterator(pass_cmds.begin() + static_cast<std::ptrdiff_t>(last + 1)));
        std::vector<PassCmd> tail(std::make_move_iterator(pass_cmds.begin() + static_cast<std::ptrdiff_t>(last + 1)),
                                  std::make_move_iterator(pass_cmds.end()));
        Encoder e{*this, *frame_target, device.CreateCommandEncoder()};
        encode_pending_clears(e.enc);
        e.render_pass_opened = true;  // readbackOnly: the canvas already holds this frame's work
        e.cmds = &head;
        for (size_t i = 0; i < head.size(); ++i) {
            e.idx = i;
            run_cmd(e, head[i]);
        }
        e.end_active();
        timer_resolve(e);
        wgpu::CommandBuffer cb = e.enc.Finish();
        queue.Submit(1, &cb);
        timer_collect(e);
        start_recorded_readbacks(true);
        ++stats.early_readback_flushes;
        early_flushed = true;
        pass_cmds = std::move(tail);
    }

    // _executePresentRenderDraw: opens the lazy render pass, then skips a draw whose pipeline is
    // unavailable or whose expected @group(0) bind group is not bound.
    bool draw_allowed(Encoder& e) {
        e.ensure_render();
        if (e.active_pipeline) {
            auto it = render_pipelines.find(*e.active_pipeline);
            if (it == render_pipelines.end()) {
                ++stats.draws_skipped;
                ++stats.op_errors;
                return false;
            }
            if (it->second.quarantined_hash) {
                note_quarantined(it->second.quarantined_hash, false);
                return false;
            }
            if (it->second.failed) {
                ++stats.draws_skipped;
                note_pipeline_failed(it->second.perf_hash, false);
                return false;
            }
            if (pipe_still_pending(it->second)) {
                ++stats.draws_skipped;
                note_pipeline_pending(it->second.perf_hash, false);
                return false;
            }
            if (it->second.expects_group0 && !e.group0_bound) {
                ++stats.draws_skipped;
                ++stats.op_errors;
                return false;
            }
            bind_fit_dummies(e, it->second.fit);
        }
        return true;
    }

    void present() {
        const uint64_t ta = now_ns();
        const bool recovery_drop = drop_next_present;  // the frame under construction when the device was lost
        drop_next_present = false;
        const bool have_target = !lost && !recovery_drop && acquire_target();
        stats.ns_present_acquire += now_ns() - ta;
        if (!have_target) {
            ++stats.present_failures;
            if (recovery_drop) ++stats.recovery_frames_dropped;
            fail_dropped_readbacks();
            reset_frame_state();
            frame_target.reset();
            early_flushed = false;
            return;
        }
        const uint64_t tp0 = now_ns();
        FrameTarget& target = *frame_target;
        Encoder e{*this, target, device.CreateCommandEncoder()};
        encode_pending_clears(e.enc);
        e.render_pass_opened = early_flushed;  // `ps.renderPassOpened = readbackOnly || earlyFlushedThisFrame`
        e.cmds = &pass_cmds;
        for (size_t i = 0; i < pass_cmds.size(); ++i) {
            e.idx = i;
            run_cmd(e, pass_cmds[i]);
        }
        e.end_active();
        if (!e.render_pass_opened) {
            // The v1 clearing swapchain pass (_ensurePresentRenderPass with renderPassOpened false).
            wgpu::RenderPassColorAttachment a = swapchain_attachment(target, LoadOp::Clear, StoreOp::Store, clear_color);
            wgpu::RenderPassDescriptor rp{};
            rp.colorAttachmentCount = 1;
            rp.colorAttachments = &a;
            wgpu::PassTimestampWrites tw{};
            if (timer_pass(e, d3d12_guest::GpuPassKind::Render, 0, &tw)) rp.timestampWrites = &tw;
            e.enc.BeginRenderPass(&rp).End();
        }
        wgpu::Buffer digest_buf;
        uint32_t digest_padded_row = 0, digest_w = 0, digest_h = 0;
        wgpu::TextureFormat digest_format = stream_format;
        if (!options.frame_digest_path.empty() && target.texture && format_bytes(digest_format) == 4) {
            digest_w = target.blit ? stream_w : swap_width;
            digest_h = target.blit ? stream_h : swap_height;
            digest_padded_row = (digest_w * 4 + 255u) & ~255u;
            wgpu::BufferDescriptor bd{};
            bd.size = uint64_t{digest_padded_row} * digest_h;
            bd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
            digest_buf = device.CreateBuffer(&bd);
            wgpu::TexelCopyTextureInfo src{};
            src.texture = target.texture;
            wgpu::TexelCopyBufferInfo dst{};
            dst.buffer = digest_buf;
            dst.layout.bytesPerRow = digest_padded_row;
            dst.layout.rowsPerImage = digest_h;
            wgpu::Extent3D extent{digest_w, digest_h, 1};
            e.enc.CopyTextureToBuffer(&src, &dst, &extent);
        }
        const bool shown = show_spec.has_value() && encode_show_target(e.enc, target.canvas_view);
        if (target.blit && !shown) encode_blit(e.enc, target.canvas_view);
        // Direct present: the swapchain image is gone after Present, so keep a snapshot (about once a second, not
        // per frame) that the stop banner can be redrawn over if the guest stops (overlay_snapshot_due).
        bool snapped = false;
        if (!target.blit && surface && target.canvas_texture && overlay_snapshot_due(now_ns(), ov_snapshot_ns)) {
            ensure_intermediate();
            wgpu::TexelCopyTextureInfo cs{}, cd{};
            cs.texture = target.canvas_texture;
            cd.texture = intermediate;
            const wgpu::Extent3D ce{stream_w, stream_h, 1};
            e.enc.CopyTextureToTexture(&cs, &cd, &ce);
            ov_snapshot_ns = now_ns();
            snapped = true;
        }
        wgpu::Buffer probe_buf;
        uint32_t probe_row = 0, probe_w = 0, probe_h = 0;
        PresentRect probe_rect;
        if (probe_pending && surface && target.canvas_texture) {
            probe_pending = false;
            if (format_bytes(swap_format) == 4) {
                probe_w = swap_width;
                probe_h = swap_height;
                probe_row = (probe_w * 4 + 255u) & ~255u;
                probe_rect = target.blit ? window->present_rect(stream_w, stream_h)
                                         : PresentRect{0, 0, probe_w, probe_h};
                wgpu::BufferDescriptor bd{};
                bd.size = uint64_t{probe_row} * probe_h;
                bd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
                probe_buf = device.CreateBuffer(&bd);
                wgpu::TexelCopyTextureInfo src{};
                src.texture = target.canvas_texture;
                wgpu::TexelCopyBufferInfo dst{};
                dst.buffer = probe_buf;
                dst.layout.bytesPerRow = probe_row;
                dst.layout.rowsPerImage = probe_h;
                wgpu::Extent3D extent{probe_w, probe_h, 1};
                e.enc.CopyTextureToBuffer(&src, &dst, &extent);
            }
        }
        wgpu::Buffer shot_buf;
        uint32_t shot_row = 0, shot_w = 0, shot_h = 0;
        ShotFormat shot_fmt = ShotFormat::Unsupported;
        if (shot_wanted() && target.texture) {
            shot_fmt = shot_format_of(stream_format);
            shot_w = target.blit ? stream_w : swap_width;
            shot_h = target.blit ? stream_h : swap_height;
            if (shot_fmt == ShotFormat::Unsupported || shot_w == 0 || shot_h == 0) {
                ++shot_failed;
                std::fprintf(stderr, "[native_exec] screenshot: stream format not supported\n");
            } else {
                shot_row = (shot_w * static_cast<uint32_t>(shot_format_bytes(shot_fmt)) + 255u) & ~255u;
                wgpu::BufferDescriptor bd{};
                bd.size = uint64_t{shot_row} * shot_h;
                bd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
                shot_buf = device.CreateBuffer(&bd);
                wgpu::TexelCopyTextureInfo src{};
                src.texture = target.texture;
                wgpu::TexelCopyBufferInfo dst{};
                dst.buffer = shot_buf;
                dst.layout.bytesPerRow = shot_row;
                dst.layout.rowsPerImage = shot_h;
                wgpu::Extent3D extent{shot_w, shot_h, 1};
                e.enc.CopyTextureToBuffer(&src, &dst, &extent);
            }
        }
        if (overlay_on || !ov_banner.empty()) encode_overlay(e.enc, target.canvas_view);  // after every copy of the guest's frame
        timer_resolve(e);
        wgpu::CommandBuffer cb = e.enc.Finish();
        const uint64_t tp1 = now_ns();
        stats.ns_present_encode += tp1 - tp0;
        queue.Submit(1, &cb);
        timer_collect(e);
        start_recorded_readbacks(true);
        if (shot_buf) shot_collect(shot_buf, shot_row, shot_w, shot_h, shot_fmt);
        if (digest_buf) write_frame_digest(digest_buf, digest_padded_row, digest_w, digest_h, digest_format);
        if (probe_buf) run_probe(probe_buf, probe_row, probe_w, probe_h, probe_rect, target.blit);
        ++frame_count;
        ++stats.frames_submitted;
        {
            const uint64_t pn = now_ns();
            if (ov_last_present_ns) ov_hist.push(static_cast<double>(pn - ov_last_present_ns) / 1e6, static_cast<uint32_t>(stats.draws - ov_last_draws));
            ov_last_present_ns = pn;
            ov_last_draws = stats.draws;
            if (target.blit || snapped) have_last_frame = true;
        }
        const uint64_t tp2 = now_ns();
        stats.ns_present_submit += tp2 - tp1;
        if (surface) surface.Present();
        if (options.window) apply_frame_cap();
        stats.ns_present_swap += now_ns() - tp2;
        frame_target.reset();
        early_flushed = false;
        reset_frame_state();
    }

    // ---- screenshots ----

    static ShotFormat shot_format_of(wgpu::TextureFormat f) {
        switch (f) {
            case wgpu::TextureFormat::RGBA8Unorm:
            case wgpu::TextureFormat::RGBA8UnormSrgb: return ShotFormat::Rgba8;
            case wgpu::TextureFormat::BGRA8Unorm:
            case wgpu::TextureFormat::BGRA8UnormSrgb: return ShotFormat::Bgra8;
            case wgpu::TextureFormat::RGB10A2Unorm: return ShotFormat::Rgb10a2;
            case wgpu::TextureFormat::RGBA16Float: return ShotFormat::Rgba16Float;
            default: return ShotFormat::Unsupported;
        }
    }

    // True when the frame being presented (index frame_count) is to be captured: F12 pressed since the
    // last present, or a --screenshot-at index. Consumes the request.
    bool shot_wanted() {
        bool want = window && window->take_screenshot_request();
        while (shot_next < options.screenshot_at.size() && options.screenshot_at[shot_next] <= frame_count) {
            if (options.screenshot_at[shot_next] == frame_count) want = true;
            ++shot_next;
        }
        return want;
    }

    // After queue.Submit(): maps the buffer; ProcessEvents delivers it once the GPU is done.
    void shot_collect(wgpu::Buffer buf, uint32_t row, uint32_t w, uint32_t h, ShotFormat fmt) {
        struct Pending {
            Impl* self;
            wgpu::Buffer buf;
            uint32_t row, w, h;
            ShotFormat fmt;
        };
        auto* pend = new Pending{this, buf, row, w, h, fmt};
        ++pending_maps;
        const size_t size = static_cast<size_t>(uint64_t{row} * h);
        buf.MapAsync(
            wgpu::MapMode::Read, 0, size, wgpu::CallbackMode::AllowProcessEvents,
            [](wgpu::MapAsyncStatus st, wgpu::StringView, Pending* pd) {
                Impl& s = *pd->self;
                if (st == wgpu::MapAsyncStatus::Success) {
                    const size_t sz = static_cast<size_t>(uint64_t{pd->row} * pd->h);
                    const auto* mapped = static_cast<const uint8_t*>(pd->buf.GetConstMappedRange(0, sz));
                    std::vector<uint8_t> copy(mapped, mapped + sz);
                    pd->buf.Unmap();
                    s.shot_spawn(std::move(copy), pd->row, pd->w, pd->h, pd->fmt);
                } else {
                    ++s.shot_failed;
                    std::fprintf(stderr, "[native_exec] screenshot: read-back map failed\n");
                }
                --s.pending_maps;
                delete pd;
            },
            pend);
    }

    void shot_spawn(std::vector<uint8_t> data, uint32_t row, uint32_t w, uint32_t h, ShotFormat fmt) {
        const std::filesystem::path dir = shot_dir;
        const std::string stamp = screenshot_timestamp(std::chrono::system_clock::now());
        std::lock_guard<std::mutex> lock(shot_mu);
        shot_threads.emplace_back([this, dir, stamp, d = std::move(data), row, w, h, fmt]() {
            std::vector<uint8_t> rgb = shot_to_rgb8(d.data(), row, w, h, fmt);
            const std::vector<uint8_t> png = png_encode_rgb(rgb.data(), w, h);
            std::error_code ec;
            if (!dir.empty()) std::filesystem::create_directories(dir, ec);
            std::string path;
            bool ok = !png.empty() && !dir.empty();
            if (ok) {
                std::lock_guard<std::mutex> l(shot_mu);  // one writer at a time picks a free file name
                const std::filesystem::path p = screenshot_unique_path(dir, stamp);
                std::ofstream f(p, std::ios::binary);
                f.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
                f.flush();
                ok = static_cast<bool>(f);
                if (ok) {
                    shot_last_path = p.string();
                    path = shot_last_path;
                }
            }
            if (ok) {
                ++shot_saved;
                shot_done = true;
                std::fprintf(stderr, "[native_exec] screenshot saved: %s (%ux%u)\n", path.c_str(), w, h);
            } else {
                ++shot_failed;
                std::fprintf(stderr, "[native_exec] screenshot: could not write into '%s'\n", dir.string().c_str());
            }
        });
    }

    void shot_join() {
        std::vector<std::thread> t;
        {
            std::lock_guard<std::mutex> lock(shot_mu);
            t.swap(shot_threads);
        }
        for (auto& th : t) th.join();
    }

    // Pump thread: shows "saved" in the title for a moment after a screenshot, then restores it.
    void shot_title_tick() {
        if (!window) return;
        if (shot_done.exchange(false)) {
            shot_title_until_ns = now_ns() + 2500000000ull;
            window->set_title(window_title());
        } else if (shot_title_until_ns && now_ns() >= shot_title_until_ns) {
            shot_title_until_ns = 0;
            window->set_title(window_title());
        }
    }

    // --window-script `probe`: the presented canvas (after the blit) is read back and the letterbox bars
    // must be black with a lit picture inside the present rect.
    void run_probe(wgpu::Buffer& buf, uint32_t padded_row, uint32_t w, uint32_t h, const PresentRect& r, bool blitted) {
        std::string err;
        const uint64_t size = uint64_t{padded_row} * h;
        ++stats.probes;
        if (!map_read(buf, size, &err)) {
            ++stats.probes_failed;
            std::fprintf(stderr, "[native_exec] window probe: %s\n", err.c_str());
            return;
        }
        const auto* mapped = static_cast<const uint8_t*>(buf.GetConstMappedRange(0, static_cast<size_t>(size)));
        const LetterboxProbe p = probe_letterbox(mapped, padded_row, w, h, r);
        buf.Unmap();
        if (!p.ok()) ++stats.probes_failed;
        std::fprintf(stderr,
                     "[native_exec] window probe %s: canvas %ux%u stream %ux%u blit=%d rect %d,%d %ux%u bars %llu nonblack %llu "
                     "inner %llu nonblack %llu\n",
                     p.ok() ? (p.blank() ? "OK(blank picture)" : "OK") : "FAIL", w, h, stream_w, stream_h, blitted ? 1 : 0, r.x, r.y, r.w, r.h,
                     static_cast<unsigned long long>(p.bar_pixels), static_cast<unsigned long long>(p.bar_nonblack),
                     static_cast<unsigned long long>(p.inner_pixels), static_cast<unsigned long long>(p.inner_nonblack));
    }

    // Drives the --window-script from poll(): actions come due relative to the first presented frame.
    void run_window_script() {
        if (script.finished() || !window) return;
        if (frame_count == 0) return;
        if (!script_start_ns) script_start_ns = now_ns();
        const uint64_t elapsed_ms = (now_ns() - script_start_ns) / 1000000;
        WindowAction a;
        while (script.pop_due(elapsed_ms, &a)) {
            if (a.kind == WindowActionKind::Probe) {
                probe_pending = true;
                continue;
            }
            const bool ok = window->apply_action(a);
            ++stats.window_actions;
            std::fprintf(stderr, "[native_exec] window script @%llu ms: action %d %ux%u -> %s\n",
                         static_cast<unsigned long long>(elapsed_ms), static_cast<int>(a.kind), a.w, a.h,
                         ok ? "ok" : "failed");
        }
    }

    // frame_digest_path: the replay digest of the frame just submitted (see ExecutorOptions).
    void write_frame_digest(wgpu::Buffer& buf, uint32_t padded_row, uint32_t w, uint32_t h, wgpu::TextureFormat f) {
        std::string err;
        const uint64_t size = uint64_t{padded_row} * h;
        if (!map_read(buf, size, &err)) {
            std::fprintf(stderr, "[native_exec] frame %llu digest: %s\n", static_cast<unsigned long long>(frame_count),
                         err.c_str());
            return;
        }
        const auto* mapped = static_cast<const uint8_t*>(buf.GetConstMappedRange(0, static_cast<size_t>(size)));
        std::vector<uint8_t> px(size_t{w} * 4 * h);
        for (uint32_t y = 0; y < h; ++y) std::memcpy(px.data() + size_t{w} * 4 * y, mapped + size_t{padded_row} * y, size_t{w} * 4);
        buf.Unmap();
        if (!digest_out.is_open()) digest_out.open(options.frame_digest_path, std::ios::binary | std::ios::app);
        digest_out << "digest frame " << frame_count << ": " << replay_digest(px.data(), w, h) << " " << w << "x" << h
                   << " " << format_name(f) << "\n";
        digest_out.flush();
    }
    std::ofstream digest_out;

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
    {
        // Env overrides for diagnosis (PG6_PIPELINE_WAIT_MS / PG6_PIPELINE_WAIT_FULLSCREEN_MS: ms, "inf", "0").
        uint64_t fr = impl->options.pipeline_wait_frame_ns, fs = impl->options.pipeline_wait_fullscreen_ns;
        parse_pipe_wait_ms(std::getenv("PG6_PIPELINE_WAIT_MS"), &fr);
        parse_pipe_wait_ms(std::getenv("PG6_PIPELINE_WAIT_FULLSCREEN_MS"), &fs);
        impl->pipe_budget.configure(fr, fs);
    }
    impl->overlay_on = options.overlay;
    if (!impl->init(status, error)) return nullptr;
    std::unique_ptr<NativeExecutor> ex(new NativeExecutor(std::move(impl)));
    if (!options.show_target.empty() && !ex->set_show_target(options.show_target, error)) {
        *status = CreateStatus::Failed;
        return nullptr;
    }
    return ex;
}

NativeExecutor::NativeExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

NativeExecutor::~NativeExecutor() {
    if (impl_) impl_->shot_join();
    if (impl_) impl_->shader_workers.stop();  // before the device goes: a running compile finishes first
    if (impl_ && impl_->device) {
        impl_->pass_cmds.clear();
        impl_->buffers.clear();
        impl_->textures.clear();
        impl_->device.Destroy();
        // Maps that were still pending complete (as failures) now: their read-backs still advance
        // the producer's counter and their callback state is freed.
        for (int i = 0; i < 100 && impl_->pending_maps > 0; ++i) impl_->instance.ProcessEvents();
    }
}

bool NativeExecutor::load_wgsl_pack(const std::string& path, std::string* error) {
    std::string local;
    return impl_->load_pack(path, error ? error : &local);
}

bool NativeExecutor::load_reflection(const std::string& path, std::string* error) {
    std::string local;
    return impl_->load_reflection_file(path, error ? error : &local);
}

bool NativeExecutor::load_reflection_text(const std::string& json, std::string* error) {
    std::string local;
    return impl_->load_reflection_text(json, error ? error : &local);
}

uint32_t NativeExecutor::execute(const uint8_t* data, size_t len) {
    Impl& s = *impl_;
    ++s.stats.batches;
    if (s.lost && !s.ensure_device()) {
        ++s.stats.dropped_while_lost;
        return 0;
    }
    const uint64_t presents_before = s.stats.present_packets;
    const uint64_t t0 = Impl::now_ns();
    const WalkResult r = walk_packets(data, len, [&](const Packet& p) { return s.dispatch(p); });
    s.stats.packets += r.packets;
    s.stats.bytes += r.bytes;
    if (r.truncated) ++s.stats.truncated_batches;
    if (s.lost) s.ensure_device();  // lost during this batch: rebuild before the read-back flush
    const uint64_t t1 = Impl::now_ns();
    if (s.readback_flush_wanted) s.flush_readbacks_now();  // at the end of every batch, like the worker
    const uint64_t t2 = Impl::now_ns();
    s.instance.ProcessEvents();
    if (s.lost) s.ensure_device();  // the loss callback arrives here
    const uint64_t t3 = Impl::now_ns();
    s.stats.ns_execute += t1 - t0;
    s.stats.ns_flush_readbacks += t2 - t1;
    s.stats.ns_process_events += t3 - t2;
    return static_cast<uint32_t>(s.stats.present_packets - presents_before);
}

bool NativeExecutor::poll() {
    const uint64_t t0 = Impl::now_ns();
    impl_->instance.ProcessEvents();
    if (impl_->lost) impl_->ensure_device();  // a driver reset between batches: rebuild now, not at the next packet
    if (impl_->fatal) return false;           // not recoverable: end the run (fault record in stats_json)
    const bool open = impl_->window ? impl_->window->pump() : true;
    if (impl_->window && impl_->show_spec) impl_->show_hotkeys(impl_->window->take_hotkeys());
    impl_->run_window_script();
    if (impl_->window && impl_->window->take_overlay_toggle()) set_overlay(!impl_->overlay_on);
    impl_->overlay_idle_redraw();
    impl_->shot_title_tick();
    impl_->stats.ns_poll += Impl::now_ns() - t0;
    return open;
}

void NativeExecutor::set_overlay(bool on) {
    if (impl_->overlay_on == on) return;
    impl_->overlay_on = on;
    impl_->overlay_dirty = true;
}

bool NativeExecutor::overlay_enabled() const { return impl_->overlay_on; }

void NativeExecutor::set_overlay_extra(int64_t audio_underruns, const std::string& fault) {
    Impl& s = *impl_;
    s.ov_underruns = audio_underruns;
    if (fault != s.ov_fault) {
        s.ov_fault = fault;
        s.overlay_dirty = true;
    }
}

void NativeExecutor::set_overlay_banner(const std::string& text) {
    Impl& s = *impl_;
    if (text != s.ov_banner) {
        s.ov_banner = text;
        s.overlay_dirty = true;
    }
}

void NativeExecutor::set_window_title(const std::string& title) {
    impl_->last_title = title;
    if (impl_->window) impl_->window->set_title(impl_->window_title());
}

std::string NativeExecutor::last_screenshot_path() const {
    std::lock_guard<std::mutex> lock(impl_->shot_mu);
    return impl_->shot_last_path;
}

bool NativeExecutor::set_show_target(const std::string& selector, std::string* error) {
    Impl& s = *impl_;
    if (selector.empty()) {
        s.show_spec.reset();
        s.show_manual = false;
        s.set_show_label("");
        if (s.window) s.window->enable_hotkeys(false);
        return true;
    }
    ShowTargetSpec spec;
    if (!parse_show_target(selector, &spec, error)) return false;
    s.show_spec = spec;
    s.show_manual = false;
    s.show_manual_target.reset();
    s.show_manual_layer = s.show_manual_mip = -1;
    if (s.window) s.window->enable_hotkeys(true);
    return true;
}

void NativeExecutor::show_target_hotkeys(uint32_t bits) { impl_->show_hotkeys(bits); }

std::string NativeExecutor::shown_target_label() const { return impl_->show_label; }

std::vector<ShowTargetCandidate> NativeExecutor::shown_target_candidates() const {
    std::vector<ShowTargetCandidate> v;
    for (const TargetRef& t : impl_->show_candidates) v.push_back({t.id, t.depth});
    return v;
}

std::string NativeExecutor::present_mode_name() const { return impl_->present_mode_name; }

void NativeExecutor::debug_force_device_lost(const std::string& message) {
    if (impl_->device) impl_->device.ForceLoss(wgpu::DeviceLostReason::Unknown, {message.data(), message.size()});
    impl_->instance.ProcessEvents();  // delivers the lost callback
}

bool NativeExecutor::faulted() const { return impl_->fatal; }

bool NativeExecutor::window_focused() const { return impl_->window && impl_->window->focused(); }

void NativeExecutor::wait_idle() {
    Impl& s = *impl_;
    s.instance.WaitAny(s.queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                                                   [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}),
                       kWaitForever);
    // Read-back and timer maps complete in ProcessEvents once the GPU is done.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (s.pending_maps > 0 && std::chrono::steady_clock::now() < deadline) {
        s.instance.ProcessEvents();
        if (s.pending_maps > 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    s.shot_join();  // screenshot files are on disk when this returns
    s.stats.screenshots_saved = s.shot_saved;
    s.stats.screenshots_failed = s.shot_failed;
}

void NativeExecutor::attach_guest_memory(const GuestMemory& memory) { impl_->guest_mem = memory; }

uint64_t NativeExecutor::frame_count() const { return impl_->frame_count; }
void NativeExecutor::Impl::snapshot_blob_stats() {
    const BlobCacheStats b = blob_cache.stats();
    stats.blob_cache_enabled = b.enabled;
    stats.blob_cache_dir = b.dir;
    stats.blob_hits = b.hits;
    stats.blob_misses = b.misses;
    stats.blob_stores = b.stores;
    stats.blob_bytes_loaded = b.bytes_loaded;
    stats.blob_bytes_stored = b.bytes_stored;
    stats.blob_evictions = b.evictions;
    stats.blob_corrupt = b.corrupt;
    stats.blob_entries = b.entries;
    stats.blob_total_bytes = b.total_bytes;
    stats.blob_max_bytes = b.max_bytes;
}

const ExecutorStats& NativeExecutor::stats() const {
    impl_->snapshot_blob_stats();
    return impl_->stats;
}
const AdapterDescription& NativeExecutor::adapter() const { return impl_->adapter; }

std::string NativeExecutor::stats_json() const {
    impl_->snapshot_blob_stats();
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
    num("device_lost_events", st.device_lost_events);
    num("recoveries", st.recoveries);
    num("recovery_failures", st.recovery_failures);
    num("recovery_ms", st.recovery_ns_total / 1000000ull);
    num("recovery_replayed_packets", st.recovery_replayed_packets);
    num("recovery_replay_errors", st.recovery_replay_errors);
    num("recovery_frames_dropped", st.recovery_frames_dropped);
    num("recovery_gpu_resources_reset", st.recovery_gpu_resources_reset);
    num("journal_bytes", st.journal_bytes);
    num("journal_overflows", st.journal_overflows);
    if (st.faulted) j += "\"fault_record\": " + st.fault_record + ", ";
    num("render_pass_missing_texture", st.render_pass_missing_texture);
    num("render_pass_dropped_colors", st.render_pass_dropped_colors);
    num("swapchain_format_unsupported", st.swapchain_format_unsupported);
    num("blit_frames", st.blit_frames);
    num("surface_configures", st.surface_configures);
    num("surface_minimized_skips", st.surface_minimized_skips);
    num("surface_acquire_retries", st.surface_acquire_retries);
    num("window_actions", st.window_actions);
    num("probes", st.probes);
    num("overlay_frames", st.overlay_frames);
    num("screenshots_saved", impl_->shot_saved);
    num("screenshots_failed", impl_->shot_failed);
    num("probes_failed", st.probes_failed);
    num("packet_errors", st.packet_errors);
    num("shader_modules", st.shader_modules);
    num("shader_pack_hits", st.shader_pack_hits);
    num("shader_pack_misses", st.shader_pack_misses);
    num("shader_code_rereads", st.shader_code_rereads);
    num("shader_materialized", st.shader_materialized);
    num("shader_materialize_ms", st.shader_materialize_ns / 1000000ull);
    num("shader_materialize_max_ms", st.shader_materialize_max_ns / 1000000ull);
    {
        char h[24];
        std::snprintf(h, sizeof h, "0x%016llx", static_cast<unsigned long long>(st.shader_materialize_max_hash));
        j += std::string("\"shader_materialize_max_hash\": \"") + h + "\", ";
    }
    num("shader_materialize_max_bytes", st.shader_materialize_max_bytes);
    num("shader_scans", st.shader_scans);
    num("shader_scan_ms", st.shader_scan_ns / 1000000ull);
    num("shader_workers", st.shader_workers);
    num("shader_async_queued", st.shader_async_queued);
    num("shader_async", st.shader_async);
    num("shader_async_ms", st.shader_async_ns / 1000000ull);
    num("shader_async_max_ms", st.shader_async_max_ns / 1000000ull);
    {
        char h[24];
        std::snprintf(h, sizeof h, "0x%016llx", static_cast<unsigned long long>(st.shader_async_max_hash));
        j += std::string("\"shader_async_max_hash\": \"") + h + "\", ";
    }
    num("shader_async_max_bytes", st.shader_async_max_bytes);
    num("shader_async_private_mb_max", st.shader_async_private_mb_max);
    num("shader_async_waits", st.shader_async_waits);
    num("shader_async_wait_ms", st.shader_async_wait_ns / 1000000ull);
    num("pipelines_deferred", st.pipelines_deferred);
    num("pipelines_deferred_built", st.pipelines_deferred_built);
    num("pipelines_deferred_max", st.pipelines_deferred_max);
    num("slow_packets", st.slow_packets);
    num("slow_packet_max_ms", st.slow_packet_max_ns / 1000000ull);
    num("textures_created", st.textures_created);
    num("texture_uploads", st.texture_uploads);
    num("texture_upload_bytes", st.texture_upload_bytes);
    num("textures_cleared", st.textures_cleared);
    num("clear_copies", st.clear_copies);
    num("clear_bytes", st.clear_bytes);
    num("clear_flushes", st.clear_flushes);
    num("clears_covered", st.clears_covered);
    num("clears_missing", st.clears_missing);
    num("texture_copies", st.texture_copies);
    num("samplers_created", st.samplers_created);
    num("bind_group_layouts", st.bind_group_layouts);
    num("pipeline_layouts", st.pipeline_layouts);
    num("render_pipelines", st.render_pipelines);
    num("compute_pipelines", st.compute_pipelines);
    num("bind_groups", st.bind_groups);
    num("draws", st.draws);
    num("draws_skipped", st.draws_skipped);
    num("dispatches", st.dispatches);
    num("dispatches_skipped", st.dispatches_skipped);
    num("dispatches_clamped", st.dispatches_clamped);
    num("auto_layout_pipelines", st.auto_layout_pipelines);
    num("manifest_layouts_unavailable", st.manifest_layouts_unavailable);
    num("auto_layout_remap_pipelines", st.auto_layout_remap_pipelines);
    num("remapped_bind_groups", st.remapped_bind_groups);
    num("remap_cache_hits", st.remap_cache_hits);
    num("remap_dynamic_offsets", st.remap_dynamic_offsets);
    num("remap_dynamic_clamps", st.remap_dynamic_clamps);
    num("remap_fallback_samplers", st.remap_fallback_samplers);
    num("remap_dummies", st.remap_dummies);
    num("readbacks_requested", st.readbacks_requested);
    num("readbacks_delivered", st.readbacks_delivered);
    num("readbacks_failed", st.readbacks_failed);
    num("readback_bytes", st.readback_bytes);
    num("early_readback_flushes", st.early_readback_flushes);
    num("v1_render_pipelines", st.v1_render_pipelines);
    num("v1_compute_pipelines", st.v1_compute_pipelines);
    num("v1_bind_groups", st.v1_bind_groups);
    num("reflection_unavailable", st.reflection_unavailable);
    num("reflection_documents", st.reflection_documents);
    num("reflection_load_errors", st.reflection_load_errors);
    num("reflection_shaders", st.reflection_shaders);
    num("reflected_render_pipelines", st.reflected_render_pipelines);
    num("reflected_compute_pipelines", st.reflected_compute_pipelines);
    num("reflected_vertex_layouts", st.reflected_vertex_layouts);
    num("reflection_vertex_fallbacks", st.reflection_vertex_fallbacks);
    num("reflection_layout_errors", st.reflection_layout_errors);
    num("reflection_binding_conflicts", st.reflection_binding_conflicts);
    num("reflection_layout_unsupported", st.reflection_layout_unsupported);
    num("bind_group_depth_layout_dummies", st.bind_group_depth_layout_dummies);
    num("depth_float_copies", st.depth_float_copies);
    num("depth_twins_created", st.depth_twins_created);
    num("depth_int_twins_created", st.depth_int_twins_created);
    num("depth_half_twins_created", st.depth_half_twins_created);
    num("depth_stencil_direct_binds", st.depth_stencil_direct_binds);
    num("show_target_frames", st.show_target_frames);
    num("show_target_unavailable", st.show_target_unavailable);
    num("manifest_layouts_resolved", st.manifest_layouts_resolved);
    num("v1_synth_samplers", st.v1_synth_samplers);
    num("timer_frames", st.timer_frames);
    num("timer_passes", st.timer_passes);
    num("timer_segments", st.timer_segments);
    num("timer_split_supported", st.timer_split_supported ? 1 : 0);
    num("timer_frames_skipped", st.timer_frames_skipped);
    num("timer_supported", st.timer_supported ? 1 : 0);
    num("limits_raised", st.limits_raised ? 1 : 0);
    num("async_pipelines", st.async_pipelines);
    num("async_pipeline_waits", st.async_pipeline_waits);
    num("async_pipeline_wait_ns", st.async_pipeline_wait_ns);
    num("pipeline_pending_skips", st.pipeline_pending_skips);
    num("remap_reflected_layouts", st.remap_reflected_layouts);
    num("layout_fit_pipelines", st.layout_fit_pipelines);
    num("layout_fit_wgsl_groups", st.layout_fit_wgsl_groups);
    num("layout_fit_over_limits", st.layout_fit_over_limits);
    num("layout_fit_still_over", st.layout_fit_still_over);
    num("layout_fit_dummy_binds", st.layout_fit_dummy_binds);
    num("layout_fit_unused_groups", st.layout_fit_unused_groups);
    num("layout_fit_restores", st.layout_fit_restores);
    num("over_limit_bind_group_layouts", st.over_limit_bind_group_layouts);
    num("over_limit_binds_skipped", st.over_limit_binds_skipped);
    j += "\"async_failure_kinds\": {";
    {
        bool first = true;
        for (const auto& [k, v] : st.async_failure_kinds) {
            j += (first ? "\"" : ", \"") + k + "\": " + std::to_string(v);
            first = false;
        }
    }
    j += "}, ";
    num("limit_uniform_buffers_per_stage", st.limit_uniform_buffers_per_stage);
    num("limit_storage_buffers_per_stage", st.limit_storage_buffers_per_stage);
    num("limit_sampled_textures_per_stage", st.limit_sampled_textures_per_stage);
    num("limit_samplers_per_stage", st.limit_samplers_per_stage);
    num("limit_storage_textures_per_stage", st.limit_storage_textures_per_stage);
    num("quarantine_skips", st.quarantine_skips);
    num("async_pipeline_failures", st.async_pipeline_failures);
    num("pipeline_failed_skips", st.pipeline_failed_skips);
    j += "\"failed_pipeline_hashes\": [";
    for (size_t i = 0; i < st.failed_pipeline_hashes.size(); ++i) {
        char hb[32];
        std::snprintf(hb, sizeof hb, "%s\"0x%016llx\"", i ? ", " : "",
                      static_cast<unsigned long long>(st.failed_pipeline_hashes[i]));
        j += hb;
    }
    j += "], ";
    num("write_masks_cleared", st.write_masks_cleared);
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
    num("ns_execute", st.ns_execute);
    num("ns_process_events", st.ns_process_events);
    num("ns_flush_readbacks", st.ns_flush_readbacks);
    num("ns_poll", st.ns_poll);
    num("ns_present_acquire", st.ns_present_acquire);
    num("ns_present_encode", st.ns_present_encode);
    num("ns_present_submit", st.ns_present_submit);
    num("ns_present_swap", st.ns_present_swap);
    j += "\"present_mode_requested\": \"" + std::string(present_choice_name(impl_->options.present_mode)) +
         "\", \"present_mode_actual\": \"" + impl_->present_mode_name + "\", ";
    num("max_fps", impl_->options.max_fps);
    table("opcode_ns", st.opcode_ns);
    j += ", ";
    table("handled", st.handled);
    j += ", ";
    table("unhandled", st.unhandled);
    j += ", ";
    table("packet_errors_by_opcode", st.packet_errors_by_opcode);
    {
        // Persistent Dawn blob cache (blob_cache.h): hits are shader / pipeline compiles a previous run
        // already paid for; misses were compiled now (and stored).
        std::string dir;
        for (char c : st.blob_cache_dir) {
            if (c == '"' || c == '\\') dir += '\\';
            dir += c;
        }
        j += ", \"blob_cache\": {\"enabled\": ";
        j += st.blob_cache_enabled ? "true" : "false";
        j += ", \"dir\": \"" + dir + "\"";
        auto bn = [&](const char* k, uint64_t v) { j += std::string(", \"") + k + "\": " + std::to_string(v); };
        bn("hits", st.blob_hits);
        bn("misses", st.blob_misses);
        bn("stores", st.blob_stores);
        bn("bytes_loaded", st.blob_bytes_loaded);
        bn("bytes_stored", st.blob_bytes_stored);
        bn("evictions", st.blob_evictions);
        bn("corrupt", st.blob_corrupt);
        bn("entries", st.blob_entries);
        bn("total_bytes", st.blob_total_bytes);
        bn("max_bytes", st.blob_max_bytes);
        j += "}, ";
    }
    if (impl_->options.gpu_error_scopes || impl_->error_scope_log.total() > 0) {
        j += "\"error_scopes\": " + impl_->error_scope_log.json() + ", ";
    }
    if (impl_->options.gpu_validation) {
        auto esc = [](const std::string& in) {
            std::string o;
            for (char c : in) {
                if (c == '"' || c == '\\') {
                    o += '\\';
                    o += c;
                } else if (static_cast<unsigned char>(c) >= 0x20) {
                    o += c;
                }
            }
            return o;
        };
        std::lock_guard<std::mutex> lock(impl_->validation_mutex);
        std::vector<const std::pair<const std::string, Impl::ValidationEntry>*> rows;
        for (const auto& kv : impl_->validation_errors) rows.push_back(&kv);
        std::sort(rows.begin(), rows.end(), [](auto* a, auto* b) { return a->second.count > b->second.count; });
        j += "\"validation\": {\"scoped_packets\": " + std::to_string(impl_->validation_scoped_packets) +
             ", \"total_errors\": " + std::to_string(impl_->validation_total) +
             ", \"unique_errors\": " + std::to_string(rows.size()) + ", \"errors\": [";
        for (size_t i = 0; i < rows.size() && i < 60; ++i) {
            const auto& e = rows[i]->second;
            if (i) j += ", ";
            j += "{\"count\": " + std::to_string(e.count) + ", \"kind\": \"" + e.kind + "\", \"first_packet\": " +
                 std::to_string(e.first_packet) + ", \"first_opcode\": " + std::to_string(e.first_opcode) + ", \"opcodes\": {";
            bool first = true;
            for (const auto& oc : e.by_opcode) {
                if (!first) j += ", ";
                first = false;
                j += "\"" + std::to_string(oc.first) + "\": " + std::to_string(oc.second);
            }
            j += "}, \"text\": \"" + esc(rows[i]->first) + "\"}";
        }
        j += "]}, ";
    }
    j += "\"adapter\": {\"backend\": \"" + impl_->adapter.backend + "\"}";
    j += ", \"device_error_log\": [";
    for (size_t i = 0; i < st.device_error_log.size(); ++i) {
        if (i) j += ", ";
        j += "\"";
        for (char c : st.device_error_log[i]) {
            if (c == '"' || c == '\\') {
                j += '\\';
                j += c;
            } else if (static_cast<unsigned char>(c) >= 0x20) {
                j += c;
            }
        }
        j += "\"";
    }
    j += "]";
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

bool NativeExecutor::read_texture(uint32_t id, uint32_t mip, std::vector<uint8_t>* out, uint32_t* width,
                                  uint32_t* height, std::string* error) {
    Impl& s = *impl_;
    auto it = s.textures.find(id);
    if (it == s.textures.end()) {
        if (error) *error = "read_texture: unknown texture " + std::to_string(id);
        return false;
    }
    const TextureEntry& t = it->second;
    const FormatInfo& fi = format_info(t.geom.format);
    if (fi.compressed || t.geom.dimension == 2) {
        if (error) *error = "read_texture: uncompressed 2D textures only";
        return false;
    }
    const uint32_t w = mip_dimension(t.geom.width, mip), h = mip_dimension(t.geom.height, mip);
    const uint32_t row = w * fi.bytes, padded = (row + 255u) & ~255u;
    const uint64_t size = uint64_t{padded} * h;
    wgpu::BufferDescriptor bd{};
    bd.size = size;
    bd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer staging = s.device.CreateBuffer(&bd);
    const bool ok = s.validated(
        [&] {
            wgpu::CommandEncoder enc = s.device.CreateCommandEncoder();
            wgpu::TexelCopyTextureInfo src{};
            src.texture = t.texture;
            src.mipLevel = mip;
            wgpu::TexelCopyBufferInfo dst{};
            dst.buffer = staging;
            dst.layout.bytesPerRow = padded;
            dst.layout.rowsPerImage = h;
            wgpu::Extent3D extent{w, h, 1};
            enc.CopyTextureToBuffer(&src, &dst, &extent);
            wgpu::CommandBuffer cb = enc.Finish();
            s.queue.Submit(1, &cb);
        },
        error);
    if (!ok || !s.map_read(staging, size, error)) return false;
    const auto* mapped = static_cast<const uint8_t*>(staging.GetConstMappedRange(0, static_cast<size_t>(size)));
    out->resize(size_t{row} * h);
    for (uint32_t y = 0; y < h; ++y) std::memcpy(out->data() + size_t{row} * y, mapped + size_t{padded} * y, row);
    staging.Unmap();
    if (width) *width = w;
    if (height) *height = h;
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
