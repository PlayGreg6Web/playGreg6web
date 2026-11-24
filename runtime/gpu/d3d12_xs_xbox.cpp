// Xbox-specific D3D12 slots (G27): frame events and pacing, PresentX planes, pipeline-state streams
// and overrides, views at GPU addresses, command-list staging and the recorded-only Xbox slots.
// Evidence, contracts and confidence: d3d12_xs_xbox.h.

#include "gpu/d3d12_xs_xbox.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu/d3d12_guest_resources.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

constexpr uint32_t kInfinite = 0xFFFFFFFFu;
constexpr uint64_t kMaxStreamBytes = 64u << 10;
constexpr uint32_t kMaxOverrides = 16;
constexpr uint32_t kMaxPlanes = 8;
constexpr uint32_t kMaxAllocationInfos = 4096;

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%" PRIx64, v);
    return b;
}

constexpr size_t align8(size_t v) {
    return (v + 7) & ~size_t{7};
}

constexpr uint64_t align_up(uint64_t v, uint64_t a) {
    return a == 0 ? v : (v + a - 1) / a * a;
}

}  // namespace

namespace xbox {

// ---------------------------------------------------------------------------------------------
// Pipeline-state streams
// ---------------------------------------------------------------------------------------------

namespace {

// D3D12 defaults for a stream without the corresponding subobject (CD3DX12 defaults).
void set_stream_defaults(GuestGraphicsPipelineStateDesc* g) {
    g->SampleMask = 0xFFFFFFFFu;
    g->SampleCount = 1;
    g->PrimitiveTopologyType = 3;  // TRIANGLE
    for (GuestRenderTargetBlendDesc& rt : g->BlendState.RenderTarget) {
        rt.SrcBlend = 2;        // ONE
        rt.DestBlend = 1;       // ZERO
        rt.BlendOp = 1;         // ADD
        rt.SrcBlendAlpha = 2;
        rt.DestBlendAlpha = 1;
        rt.BlendOpAlpha = 1;
        rt.LogicOp = 4;         // NOOP
        rt.RenderTargetWriteMask = 0xF;
    }
    g->RasterizerState.FillMode = 3;  // SOLID
    g->RasterizerState.CullMode = 3;  // BACK
    g->RasterizerState.DepthClipEnable = 1;
    GuestDepthStencilDesc& ds = g->DepthStencilState;
    ds.DepthEnable = 1;
    ds.DepthWriteMask = 1;  // ALL
    ds.DepthFunc = 2;       // LESS
    ds.StencilReadMask = 0xFF;
    ds.StencilWriteMask = 0xFF;
    for (GuestDepthStencilOpDesc* op : {&ds.FrontFace, &ds.BackFace}) {
        op->StencilFailOp = op->StencilDepthFailOp = op->StencilPassOp = 1;  // KEEP
        op->StencilFunc = 8;                                                  // ALWAYS
    }
}

struct SubobjectLayout {
    size_t data_offset;  // from the subobject start
    size_t size;
};

bool layout_of(uint32_t type, SubobjectLayout* out) {
    switch (type) {
    case 0:   // ROOT_SIGNATURE
        *out = {8, 8};
        return true;
    case 1: case 2: case 3: case 4: case 5: case 6:  // VS PS DS HS GS CS
    case 23: case 24:                                // AS MS
    case 19:                                         // CACHED_PSO
    case 28:                                         // SERIALIZED_ROOT_SIGNATURE
        *out = {8, 16};
        return true;
    case 12:  // INPUT_LAYOUT
        *out = {8, 16};
        return true;
    case 7:   // STREAM_OUTPUT
        *out = {8, 32};
        return true;
    case 22:  // VIEW_INSTANCING
        *out = {8, 24};
        return true;
    case 8:   // BLEND
        *out = {4, sizeof(GuestBlendDesc)};
        return true;
    case 9: case 13: case 14: case 16: case 18: case 20:  // SAMPLE_MASK IB_STRIP_CUT TOPOLOGY DSV_FORMAT NODE_MASK FLAGS
        *out = {4, 4};
        return true;
    case 10: case 26:  // RASTERIZER, RASTERIZER1
        *out = {4, 44};
        return true;
    case 27:  // RASTERIZER2
        *out = {4, 40};
        return true;
    case 11:  // DEPTH_STENCIL
        *out = {4, 52};
        return true;
    case 21:  // DEPTH_STENCIL1
        *out = {4, 56};
        return true;
    case 25:  // DEPTH_STENCIL2
        *out = {4, 60};
        return true;
    case 15:  // RENDER_TARGET_FORMATS
        *out = {4, 36};
        return true;
    case 17:  // SAMPLE_DESC
        *out = {4, 8};
        return true;
    default:
        return false;
    }
}

template <typename T>
T load(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

}  // namespace

bool parse_pipeline_stream(const uint8_t* bytes, size_t size, StreamPipeline* out, std::string* error) {
    StreamPipeline p;
    set_stream_defaults(&p.graphics);
    bool has_cs = false, has_mesh = false;
    size_t off = 0;
    while (off < size) {
        if (size - off < 4) {
            *error = "truncated subobject at offset " + std::to_string(off);
            return false;
        }
        const uint32_t type = load<uint32_t>(bytes + off);
        SubobjectLayout l{};
        if (!layout_of(type, &l)) {
            *error = "unknown subobject type " + std::to_string(type) + " at offset " + std::to_string(off);
            return false;
        }
        if (off + l.data_offset + l.size > size) {
            *error = "subobject type " + std::to_string(type) + " exceeds the stream";
            return false;
        }
        const uint8_t* d = bytes + off + l.data_offset;
        GuestGraphicsPipelineStateDesc& g = p.graphics;
        switch (type) {
        case 0:
            g.pRootSignature = p.compute.pRootSignature = load<uint64_t>(d);
            break;
        case 1: std::memcpy(&g.VS, d, 16); break;
        case 2: std::memcpy(&g.PS, d, 16); break;
        case 3: std::memcpy(&g.DS, d, 16); break;
        case 4: std::memcpy(&g.HS, d, 16); break;
        case 5: std::memcpy(&g.GS, d, 16); break;
        case 6:
            std::memcpy(&p.compute.CS, d, 16);
            has_cs = true;
            break;
        case 23: case 24: has_mesh = true; break;
        case 7: std::memcpy(&g.StreamOutput, d, 32); break;
        case 8: std::memcpy(&g.BlendState, d, sizeof(GuestBlendDesc)); break;
        case 9: g.SampleMask = load<uint32_t>(d); break;
        case 10: std::memcpy(&g.RasterizerState, d, 44); break;
        case 26:
        case 27: {  // float depth bias (RASTERIZER1/2): converted to the integer field
            GuestRasterizerDesc r{};
            r.FillMode = load<uint32_t>(d);
            r.CullMode = load<uint32_t>(d + 4);
            r.FrontCounterClockwise = load<int32_t>(d + 8);
            r.DepthBias = static_cast<int32_t>(load<float>(d + 12));
            r.DepthBiasClamp = load<float>(d + 16);
            r.SlopeScaledDepthBias = load<float>(d + 20);
            r.DepthClipEnable = load<int32_t>(d + 24);
            if (type == 26) {
                r.MultisampleEnable = load<int32_t>(d + 28);
                r.AntialiasedLineEnable = load<int32_t>(d + 32);
                r.ForcedSampleCount = load<uint32_t>(d + 36);
                r.ConservativeRaster = load<uint32_t>(d + 40);
            } else {
                r.ForcedSampleCount = load<uint32_t>(d + 32);
                r.ConservativeRaster = load<uint32_t>(d + 36);
            }
            g.RasterizerState = r;
            break;
        }
        case 11:
        case 21:  // DEPTH_STENCIL1 = DEPTH_STENCIL + DepthBoundsTestEnable (not supported, ignored)
            std::memcpy(&g.DepthStencilState, d, 52);
            break;
        case 25: {  // DEPTH_STENCIL2: per-face stencil masks (front face masks are used)
            GuestDepthStencilDesc ds{};
            ds.DepthEnable = load<int32_t>(d);
            ds.DepthWriteMask = load<uint32_t>(d + 4);
            ds.DepthFunc = load<uint32_t>(d + 8);
            ds.StencilEnable = load<int32_t>(d + 12);
            std::memcpy(&ds.FrontFace, d + 16, 16);
            ds.StencilReadMask = d[32];
            ds.StencilWriteMask = d[33];
            std::memcpy(&ds.BackFace, d + 36, 16);
            g.DepthStencilState = ds;
            break;
        }
        case 12: std::memcpy(&g.InputLayout, d, 16); break;
        case 13: g.IBStripCutValue = load<uint32_t>(d); break;
        case 14: g.PrimitiveTopologyType = load<uint32_t>(d); break;
        case 15:
            std::memcpy(g.RTVFormats, d, 32);
            g.NumRenderTargets = load<uint32_t>(d + 32);
            break;
        case 16: g.DSVFormat = load<uint32_t>(d); break;
        case 17:
            g.SampleCount = load<uint32_t>(d);
            g.SampleQuality = load<uint32_t>(d + 4);
            break;
        case 18: g.NodeMask = p.compute.NodeMask = load<uint32_t>(d); break;
        case 19: std::memcpy(&g.CachedPSO, d, 16); break;
        case 20: g.Flags = p.compute.Flags = load<uint32_t>(d); break;
        default:  // VIEW_INSTANCING, SERIALIZED_ROOT_SIGNATURE
            ++p.ignored_subobjects;
            break;
        }
        ++p.subobjects;
        off = align8(off + l.data_offset + l.size);
    }
    p.kind = has_mesh ? StreamPipelineKind::Mesh : (has_cs ? StreamPipelineKind::Compute : StreamPipelineKind::Graphics);
    *out = p;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Frame pacing
// ---------------------------------------------------------------------------------------------

FramePacer::FramePacer() : origin_(Clock::now()) {}

uint64_t FramePacer::frame_locked(Clock::time_point now) const {
    if (now < origin_) {
        return base_frame_;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - origin_).count();
    return base_frame_ + static_cast<uint64_t>(elapsed) / interval_us_;
}

FramePacer::Clock::time_point FramePacer::start_of_locked(uint64_t frame) const {
    return origin_ + std::chrono::microseconds(static_cast<int64_t>((frame - base_frame_) * interval_us_));
}

void FramePacer::set_interval(uint32_t interval_us, uint32_t period) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto now = Clock::now();
    const uint64_t current = frame_locked(now);
    // Keep the tick count continuous: the current tick starts now at the new interval.
    base_frame_ = current;
    origin_ = now;
    interval_us_ = std::clamp(interval_us == 0 ? kDefaultIntervalUs : interval_us, kMinIntervalUs, kMaxIntervalUs);
    period_ = period;
    cv_.notify_all();
}

uint32_t FramePacer::interval_us() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return interval_us_;
}

uint32_t FramePacer::period() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return period_;
}

uint64_t FramePacer::current_frame() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frame_locked(Clock::now());
}

FramePacer::WaitResult FramePacer::wait_after(uint64_t after, uint32_t timeout_ms, uint64_t* frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto start = Clock::now();
    const bool bounded = timeout_ms != kInfinite;
    const auto deadline = start + std::chrono::milliseconds(bounded ? timeout_ms : 0);
    for (;;) {
        const auto now = Clock::now();
        const uint64_t current = frame_locked(now);
        if (shutdown_) {
            *frame = current;
            return WaitResult::Shutdown;
        }
        if (current > after) {
            *frame = current;
            return WaitResult::Ok;
        }
        if (bounded && now >= deadline) {
            *frame = current;
            return WaitResult::Timeout;
        }
        auto wake = start_of_locked(after + 1);
        if (bounded && deadline < wake) {
            wake = deadline;
        }
        // Woken by the tick, a new interval or shutdown; spurious wake-ups loop.
        cv_.wait_until(lock, wake);
    }
}

void FramePacer::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    cv_.notify_all();
}

bool FramePacer::is_shut_down() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdown_;
}

}  // namespace xbox

// ---------------------------------------------------------------------------------------------
// Device: frame events
// ---------------------------------------------------------------------------------------------

HResult Device::SetFrameIntervalX(uint64_t, uint32_t interval_us, uint32_t period, uint32_t) {
    pacer_.set_interval(interval_us, period);
    count(xbox::Counter::FrameIntervalChanges);
    return kSOk;
}

HResult Device::ScheduleFrameEventX(uint32_t, uint32_t, uint64_t, uint32_t) {
    // Events are ticks of the pacer; the type and offset only shift the event inside the frame.
    count(xbox::Counter::FrameEventsScheduled);
    return kSOk;
}

HResult Device::WaitFrameEventX(uint32_t type, uint32_t timeout_ms, uint64_t, uint32_t, GuestPtr<uint64_t> token) {
    if (token.is_null()) {
        note_bad_argument("WaitFrameEventX: null token pointer");
        return kEInvalidArg;
    }
    uint64_t in = 0;
    token.read(in);
    count(xbox::Counter::FrameWaits);
    uint64_t frame = 0;
    if (timeout_ms == 0) {
        // Poll: the current tick, never a failure (the executable traps if this fails).
        frame = pacer_.current_frame();
    } else {
        // Type 2 with a token waits until that frame is over; otherwise until a tick newer than
        // the last one handed out.
        const uint64_t after = (type == 2 && in != 0) ? in : last_frame_token_.load();
        const auto r = pacer_.wait_after(after, timeout_ms, &frame);
        if (r == xbox::FramePacer::WaitResult::Timeout) {
            count(xbox::Counter::FrameWaitTimeouts);
            token.write(frame);
            return static_cast<HResult>(0x887A0027u);  // DXGI_ERROR_WAIT_TIMEOUT
        }
    }
    uint64_t prev = last_frame_token_.load();
    while (frame > prev && !last_frame_token_.compare_exchange_weak(prev, frame)) {
    }
    token.write(frame);
    return kSOk;
}

HResult Device::QueryFrameStatistics_unverified(uint64_t, uint32_t, GuestPtr<uint32_t> count_out, uint64_t) {
    // No statistics: the caller reads entries only when the count is non-zero.
    if (!count_out.is_null()) {
        count_out.write(0u);
    }
    return kSOk;
}

// ---------------------------------------------------------------------------------------------
// Device: no-op and refused slots
// ---------------------------------------------------------------------------------------------

HResult Device::Slot49_SetFeatureToggle_unverified(uint32_t, uint32_t) {
    count(xbox::Counter::FeatureToggles);
    return kSOk;
}

HResult Device::Slot50_RegisterRemovalCallbacks_unverified(uint64_t, uint64_t, uint64_t) {
    count(xbox::Counter::XboxNoOps);
    return kSOk;
}

HResult Device::Slot61_unverified(uint64_t, uint64_t) {
    count(xbox::Counter::XboxNoOps);
    return kSOk;
}

HResult Device::Slot119_unverified(uint64_t, uint32_t, uint32_t) {
    count(xbox::Counter::XboxNoOps);
    return kSOk;
}

HResult Device::Slot132_unverified(uint32_t) {
    count(xbox::Counter::XboxNoOps);
    return kSOk;
}

HResult Device::RegisterPagePool_unverified(uint64_t, uint32_t, GuestPtr<uint64_t> handle) {
    const uint64_t h = 0x9000 + counter(xbox::Counter::PagePools) + 1;
    count(xbox::Counter::PagePools);
    if (!handle.is_null()) {
        handle.write(h);
    }
    return kSOk;
}

HResult Device::CreatePipelineStateFromBlob_unverified(com::CallContext& ctx) {
    // Slot 62: (desc, 0, 0, riid, out) -> out at position 5; slot 67: (&desc, riid, out) -> out at 3.
    // The blob format is the title's own: refuse and clear whichever out pointer is a valid
    // pointer to a null-initialised slot (both callers take their failure path).
    count(xbox::Counter::BlobPipelineStatesRefused);
    for (unsigned pos : {3u, 5u}) {
        const uint64_t out = ctx.int_arg(pos);
        uint64_t current = 1;
        if (out != 0 && com::read_guest(out, &current, sizeof(current)) && current == 0) {
            const uint64_t zero = 0;
            com::write_guest(out, &zero, sizeof(zero));
        }
    }
    return kENotImpl;
}

HResult Device::CreateStateObject(uint64_t, const Guid&, OutPtr out) {
    count(xbox::Counter::StateObjectsRefused);
    if (out) {
        out.write(0);
    }
    return kENotImpl;
}

// ---------------------------------------------------------------------------------------------
// Device: pipeline states
// ---------------------------------------------------------------------------------------------

HResult Device::CreatePipelineStateFromStream(uint64_t stream_desc, const Guid& riid, OutPtr out) {
    xbox::GuestPipelineStateStreamDesc sd{};
    if (stream_desc == 0 || !com::read_guest(stream_desc, sd) || sd.SizeInBytes == 0 ||
        sd.SizeInBytes > kMaxStreamBytes || sd.pPipelineStateSubobjectStream == 0) {
        note_bad_argument("CreatePipelineState (stream): unreadable stream descriptor");
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(sd.SizeInBytes));
    if (!com::read_guest(sd.pPipelineStateSubobjectStream, bytes.data(), bytes.size())) {
        note_bad_argument("CreatePipelineState (stream): unreadable stream at " + hex(sd.pPipelineStateSubobjectStream));
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    xbox::StreamPipeline p;
    std::string error;
    if (!xbox::parse_pipeline_stream(bytes.data(), bytes.size(), &p, &error)) {
        note_bad_argument("CreatePipelineState (stream): " + error);
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    count(xbox::Counter::StreamPipelineStates);
    switch (p.kind) {
    case xbox::StreamPipelineKind::Compute:
        return create_compute_pso(p.compute, riid, out);
    case xbox::StreamPipelineKind::Mesh:
        count(xbox::Counter::PlaceholderPipelineStates);
        return create_placeholder_pso("mesh/amplification shaders", riid, out);
    case xbox::StreamPipelineKind::Graphics:
        break;
    }
    return create_graphics_pso(p.graphics, riid, out);
}

HResult Device::CreatePipelineStateWithOverrides(InterfaceArg<IPipelineStateHost> base, uint32_t n, uint64_t overrides,
                                                 const Guid& riid, OutPtr out) {
    if (!base || n > kMaxOverrides || (n != 0 && overrides == 0)) {
        note_bad_argument("CreatePipelineStateWithOverrides: invalid base pipeline state or override array");
        if (out) {
            out.write(0);
        }
        return kEInvalidArg;
    }
    auto* state = static_cast<PipelineState*>(base.get());
    const GraphicsPsoSource* source = state->graphics_source();
    if (source == nullptr) {
        // Compute or placeholder base: nothing to override, hand out the base itself.
        note_bad_argument("CreatePipelineStateWithOverrides: base is not a graphics pipeline (base returned)");
        return hand_out(com::Ref<PipelineState>::share(state), riid, out);
    }
    GraphicsPsoSource derived = *source;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t entry = overrides + uint64_t{xbox::kPsoOverrideStride} * i;
        uint32_t kind = 0;
        if (!com::read_guest(entry, &kind, sizeof(kind))) {
            note_bad_argument("CreatePipelineStateWithOverrides: unreadable override " + std::to_string(i));
            if (out) {
                out.write(0);
            }
            return kEInvalidArg;
        }
        bool ok = true;
        switch (static_cast<xbox::PsoOverrideKind>(kind)) {
        case xbox::PsoOverrideKind::DepthStencil: {
            GuestDepthStencilDesc g{};
            ok = com::read_guest(entry + 4, g);
            convert_depth_stencil_desc(g, &derived.desc.DepthStencilState);
            break;
        }
        case xbox::PsoOverrideKind::Rasterizer: {
            GuestRasterizerDesc g{};
            ok = com::read_guest(entry + 4, g);
            convert_rasterizer_desc(g, &derived.desc.RasterizerState);
            break;
        }
        case xbox::PsoOverrideKind::Blend: {
            GuestBlendDesc g{};
            ok = com::read_guest(entry + 4, g);
            convert_blend_desc(g, &derived.desc.BlendState);
            break;
        }
        default:
            count(xbox::Counter::UnknownOverrideKinds);  // kind 5 (and anything else): ignored
            break;
        }
        if (!ok) {
            note_bad_argument("CreatePipelineStateWithOverrides: unreadable override " + std::to_string(i));
            if (out) {
                out.write(0);
            }
            return kEInvalidArg;
        }
    }
    count(xbox::Counter::OverridePipelineStates);
    return create_graphics_pso_from(std::move(derived), riid, out);
}

// ---------------------------------------------------------------------------------------------
// Device: views at GPU addresses, allocation info, alias windows
// ---------------------------------------------------------------------------------------------

namespace {

// Element size of a buffer view (structured: stride; raw: 4; typed: the format's size).
uint32_t buffer_element_size(uint32_t format, uint32_t stride, bool raw) {
    if (stride != 0) {
        return stride;
    }
    if (raw) {
        return 4;
    }
    const DxgiFormatLayout l = dxgi_format_layout(format);
    return l.known && l.block_width == 1 ? l.block_bytes : 0;
}

}  // namespace

void Device::CreateShaderResourceViewAtAddress(uint64_t, uint64_t view_desc, uint64_t dest) {
    GuestShaderResourceViewDesc g{};
    uint64_t va = 0;
    if (view_desc == 0 || !com::read_guest(view_desc, g) ||
        !com::read_guest(view_desc + xbox::kViewAtAddressOffset, &va, sizeof(va))) {
        note_bad_argument("CreateShaderResourceView (Xbox slot 70): unreadable descriptor");
        return;
    }
    if (g.ViewDimension != d3d::ViewDimension::Buffer) {
        note_bad_argument("CreateShaderResourceView (Xbox slot 70): only buffer views are supported (dimension " +
                          std::to_string(g.ViewDimension) + ")");
        return;
    }
    d3d::ViewDesc vd;
    vd.format = g.Format;
    vd.dimension = g.ViewDimension;
    vd.component_mapping = g.Shader4ComponentMapping;
    vd.first_element = uint64_t{g.u[0]} | (uint64_t{g.u[1]} << 32);
    vd.num_elements = g.u[2];
    vd.structure_byte_stride = g.u[3];
    vd.flags = g.u[4];
    d3d::GpuVaResolution res;
    uint32_t id = 0;
    if (state_bridge().resolve_gpu_va(va, &res)) {
        const uint32_t elem = buffer_element_size(g.Format, g.u[3], (g.u[4] & 1) != 0);
        if (elem == 0 || res.offset % elem != 0) {
            note_bad_argument("CreateShaderResourceView (Xbox slot 70): address " + hex(va) +
                              " is not element-aligned inside its buffer");
            return;
        }
        id = res.resource_id;
        vd.first_element += res.offset / elem;
    } else {
        note_bad_argument("CreateShaderResourceView (Xbox slot 70): unknown GPU address " + hex(va) +
                          " (null descriptor written)");
    }
    count(xbox::Counter::ViewsAtAddress);
    const d3d::Status s = heaps_.create_shader_resource_view(id, &vd, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateShaderResourceView (Xbox slot 70): ") + d3d::status_name(s));
    }
}

void Device::CreateUnorderedAccessViewAtAddress(uint64_t, uint64_t view_desc, uint64_t dest) {
    GuestUnorderedAccessViewDesc g{};
    uint64_t va = 0;
    if (view_desc == 0 || !com::read_guest(view_desc, g) ||
        !com::read_guest(view_desc + xbox::kViewAtAddressOffset, &va, sizeof(va))) {
        note_bad_argument("CreateUnorderedAccessView (Xbox slot 71): unreadable descriptor");
        return;
    }
    if (g.ViewDimension != d3d::ViewDimension::Buffer) {
        note_bad_argument("CreateUnorderedAccessView (Xbox slot 71): only buffer views are supported");
        return;
    }
    d3d::ViewDesc vd;
    vd.format = g.Format;
    vd.dimension = g.ViewDimension;
    vd.first_element = uint64_t{g.u[0]} | (uint64_t{g.u[1]} << 32);
    vd.num_elements = g.u[2];
    vd.structure_byte_stride = g.u[3];
    vd.counter_offset_in_bytes = uint64_t{g.u[4]} | (uint64_t{g.u[5]} << 32);
    vd.flags = g.u[6];
    d3d::GpuVaResolution res;
    uint32_t id = 0;
    if (state_bridge().resolve_gpu_va(va, &res)) {
        const uint32_t elem = buffer_element_size(g.Format, g.u[3], (g.u[6] & 1) != 0);
        if (elem == 0 || res.offset % elem != 0) {
            note_bad_argument("CreateUnorderedAccessView (Xbox slot 71): address " + hex(va) +
                              " is not element-aligned inside its buffer");
            return;
        }
        id = res.resource_id;
        vd.first_element += res.offset / elem;
    } else {
        note_bad_argument("CreateUnorderedAccessView (Xbox slot 71): unknown GPU address " + hex(va) +
                          " (null descriptor written)");
    }
    count(xbox::Counter::ViewsAtAddress);
    const d3d::Status s = heaps_.create_unordered_access_view(id, 0, &vd, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateUnorderedAccessView (Xbox slot 71): ") + d3d::status_name(s));
    }
}

GuestResourceAllocationInfo Device::GetResourceAllocationInfo1(uint32_t visible_mask, uint32_t n, uint64_t descs,
                                                               uint64_t infos) {
    const GuestResourceAllocationInfo total = GetResourceAllocationInfo(visible_mask, n, descs);
    if (infos == 0 || total.SizeInBytes == UINT64_MAX || n > kMaxAllocationInfos) {
        return total;
    }
    uint64_t offset = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const GuestResourceAllocationInfo one =
            GetResourceAllocationInfo(visible_mask, 1, descs + sizeof(GuestResourceDesc) * i);
        offset = align_up(offset, one.Alignment);
        const xbox::GuestResourceAllocationInfo1 info{offset, one.Alignment, one.SizeInBytes};
        if (!com::write_guest(infos + sizeof(info) * i, info)) {
            note_bad_argument("GetResourceAllocationInfo1: unwritable info array");
            break;
        }
        offset += one.SizeInBytes;
    }
    return total;
}

void Device::register_gpu_alias(uint64_t gpu_base, uint64_t cpu_base, uint64_t size) {
    std::lock_guard<std::mutex> lock(alias_mutex_);
    aliases_.push_back(AliasWindow{gpu_base, cpu_base, size});
}

uint64_t Device::cpu_address_for_gpu(uint64_t gpu_va) const {
    std::lock_guard<std::mutex> lock(alias_mutex_);
    for (const AliasWindow& w : aliases_) {
        if (gpu_va >= w.gpu_base && gpu_va - w.gpu_base < w.size) {
            return w.cpu_base + (gpu_va - w.gpu_base);
        }
    }
    return gpu_va;
}

// ---------------------------------------------------------------------------------------------
// Allocator and queue
// ---------------------------------------------------------------------------------------------

HResult CommandAllocator::SetAllocationCallback_unverified(uint64_t callback, uint64_t) {
    callback_.store(callback);
    return kSOk;
}

HResult CommandQueue::SuspendX(uint32_t) {
    suspended_.store(true);
    device_->count(xbox::Counter::Suspends);
    return kSOk;
}

HResult CommandQueue::ResumeX() {
    suspended_.store(false);
    device_->count(xbox::Counter::Resumes);
    return kSOk;
}

HResult CommandQueue::Slot31_UpdatePageMappings_unverified(com::CallContext&) {
    device_->count(xbox::Counter::PageMappingsIgnored);
    return kSOk;
}

HResult CommandQueue::PresentX(uint32_t plane_count, uint64_t planes, uint64_t) {
    if (plane_count == 0) {
        device_->count(xbox::Counter::PresentsWithoutPlanes);
        return kSOk;
    }
    xbox::GuestPresentPlaneX plane{};
    if (plane_count > kMaxPlanes || planes == 0 || !com::read_guest(planes, plane)) {
        device_->note_bad_argument("PresentX: unreadable plane array (" + std::to_string(plane_count) + " planes)");
        return kEInvalidArg;
    }
    if (plane.Token != 0) {
        device_->last_present_token.store(plane.Token);
    }
    uint64_t resource = 0;
    if (plane.ResourceCount != 0 && plane.ppResources != 0 &&
        com::read_guest(plane.ppResources, &resource, sizeof(resource)) && resource != 0) {
        com::detail::Resolved r;
        if (com::detail::lookup_interface(resource, &r) && r.iface == &kID3D12Resource.info) {
            auto* res = static_cast<Resource*>(static_cast<IResourceHost*>(r.host));
            if (!res->is_back_buffer()) {
                if (device_->set_back_buffer(resource)) {
                    device_->count(xbox::Counter::BackBufferMarks);
                } else {
                    device_->note_bad_argument("PresentX: plane 0 resource " + hex(resource) +
                                               " is not a 2D texture with a GPU object");
                }
            }
        } else {
            device_->note_bad_argument("PresentX: plane 0 resource " + hex(resource) + " is not a resource");
        }
    } else {
        device_->note_bad_argument("PresentX: plane 0 has no readable resource");
    }
    if (plane_count > 1) {
        device_->count(xbox::Counter::ExtraPlanesIgnored, plane_count - 1);
    }
    device_->count(xbox::Counter::PresentsWithPlanes);
    device_->present();
    return kSOk;
}

// ---------------------------------------------------------------------------------------------
// Command list
// ---------------------------------------------------------------------------------------------

void CommandList::initialize_guest_payload() {
    std::vector<uint8_t> zeros(static_cast<size_t>(kComPayloadSize), 0);
    com::write_guest(payload_address(), zeros.data(), zeros.size());
}

void CommandList::SetPipelineStateX(com::CallContext& ctx) {
    if (!usable()) {
        return;
    }
    // The executable stages the PSO's 32-byte payload at list + 0x544 and passes only `this`.
    uint64_t staged[2] = {};
    const uint64_t staged_va = identity() + xbox::kListStagedPipelineOffset;
    uint64_t pointer = 0;
    if (com::read_guest(staged_va, staged, sizeof(staged)) && staged[0] == xbox::kPipelineStateTag) {
        pointer = staged[1];
        const uint64_t zero = 0;
        com::write_guest(staged_va, &zero, sizeof(zero));  // consumed: a later explicit call uses rdx
    } else {
        pointer = ctx.int_arg(1);
    }
    auto state = InterfaceArg<IPipelineStateHost>::resolve(pointer);
    if (!state) {
        note("SetPipelineState: no staged pipeline state and invalid argument " + hex(pointer));
        return;
    }
    bind_pipeline(static_cast<PipelineState*>(state.get()));
}

void CommandList::ExecuteIndirect(com::CallContext&) {
    if (usable()) {
        // No indirect draw/dispatch packet exists in the GPU ABI: diagnosed and skipped.
        device_->count(xbox::Counter::IndirectSkipped);
        device_->note_skipped_draw();
    }
}

void CommandList::OMSetDepthBounds(float, float) {
    if (usable()) {
        device_->count(xbox::Counter::DepthBoundsIgnored);  // no depth-bounds test in WebGPU
    }
}

void CommandList::RSSetShadingRate(uint32_t, uint64_t) {
    if (usable()) {
        device_->count(xbox::Counter::ShadingRateIgnored);  // full-rate shading
    }
}

void CommandList::RSSetShadingRateImage(InterfaceArg<IResourceHost>) {
    if (usable()) {
        device_->count(xbox::Counter::ShadingRateIgnored);
    }
}

void CommandList::note_xbox_slot(uint32_t slot) {
    ++xbox_calls_[slot];
    device_->count(xbox::Counter::XboxNoOps);
}

std::vector<std::pair<uint32_t, uint64_t>> CommandList::xbox_slot_calls() const {
    return {xbox_calls_.begin(), xbox_calls_.end()};
}

}  // namespace playgta6::gpu::d3d12_guest
