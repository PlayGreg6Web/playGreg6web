// ID3D12GraphicsCommandList on the state bridge, command list creation and ExecuteCommandLists.
// See d3d12_guest_resources.h for the data-flow model.

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#include "gpu/d3d12_guest_resources.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

constexpr uint32_t kMaxViewports = 16;
constexpr uint32_t kMaxRects = 64;
constexpr uint32_t kMaxBarriers = 4096;
constexpr uint32_t kMaxListsPerExecute = 1024;
constexpr uint64_t kMaxWriteChunk = 1u << 20;

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%" PRIx64, v);
    return b;
}

template <typename T>
bool read_n(uint64_t va, uint32_t count, std::vector<T>* out) {
    out->assign(count, T{});
    return count == 0 || (va != 0 && com::read_guest(va, out->data(), sizeof(T) * count));
}

Resource* as_resource(uint64_t va) {
    auto arg = InterfaceArg<IResourceHost>::resolve(va);
    return arg ? static_cast<Resource*>(arg.get()) : nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Device: command lists
// ---------------------------------------------------------------------------------------------

HResult Device::create_command_list(uint32_t type, InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid,
                                    OutPtr out) {
    if (type > 3) {
        note_bad_argument("CreateCommandList: unsupported type " + std::to_string(type));
        return kEInvalidArg;
    }
    if (initial_state.invalid()) {
        note_bad_argument("CreateCommandList: invalid initial pipeline state " + hex(initial_state.va()));
        return kEInvalidArg;
    }
    auto list = com::create_object<CommandList>(com::implements(kID3D12GraphicsCommandList), device_ref(this), type);
    if (list) {
        list->initialize_guest_payload();
        list->reset(initial_state ? static_cast<PipelineState*>(initial_state.get()) : nullptr);
    }
    return hand_out(std::move(list), riid, out);
}

HResult Device::CreateCommandList(uint32_t, uint32_t type, InterfaceArg<ICommandAllocatorHost> allocator,
                                  InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid, OutPtr out) {
    if (!allocator) {
        note_bad_argument("CreateCommandList: invalid allocator " + hex(allocator.va()));
        return kEInvalidArg;
    }
    return create_command_list(type, std::move(initial_state), riid, out);
}

HResult Device::CreateCommandListX(const GuestCommandListDescX* desc, InterfaceArg<ICommandAllocatorHost> allocator,
                                   InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid, OutPtr out) {
    if (desc == nullptr || !allocator) {
        note_bad_argument("CreateCommandList (Xbox slot 73): unreadable descriptor or invalid allocator");
        return kEInvalidArg;
    }
    return create_command_list(desc->Type, std::move(initial_state), riid, out);
}

// ---------------------------------------------------------------------------------------------
// Queue: ExecuteCommandLists
// ---------------------------------------------------------------------------------------------

void CommandQueue::ExecuteCommandListsX(com::CallContext& ctx) {
    // (NumCommandLists, ppCommandLists, frame token, flags): the token (device slot 80's output, 0
    // outside a frame) is recorded; the flags are not interpreted.
    const uint32_t count = static_cast<uint32_t>(ctx.int_arg(1));
    const uint64_t lists = ctx.int_arg(2);
    if (ctx.int_arg(3) != 0) {
        device_->last_execute_token.store(ctx.int_arg(3));
        device_->count(xbox::Counter::ExecutesWithToken);
    }
    std::vector<uint64_t> pointers;
    if (count > kMaxListsPerExecute || !read_n(lists, count, &pointers)) {
        device_->note_bad_argument("ExecuteCommandLists: unreadable list array " + hex(lists));
        return;
    }
    execute(pointers);
}

void CommandQueue::execute(const std::vector<uint64_t>& lists) {
    std::vector<std::shared_ptr<com::ComObject>> pins;
    std::vector<CommandList*> resolved;
    for (uint64_t p : lists) {
        com::detail::Resolved r;
        if (!com::detail::lookup_interface(p, &r) || r.iface != &kID3D12GraphicsCommandList.info) {
            device_->note_bad_argument("ExecuteCommandLists: not a command list " + hex(p));
            continue;
        }
        auto* list = static_cast<CommandList*>(static_cast<IGraphicsCommandListHost*>(r.host));
        if (list->is_open()) {
            device_->note_bad_argument("ExecuteCommandLists: command list " + hex(p) + " is not closed");
            continue;
        }
        pins.push_back(std::move(r.object));
        resolved.push_back(list);
    }
    // One submission: changed CPU-visible buffers, then each list's CPU-sourced copies and packets.
    std::vector<CommandEncoder> parts(1 + resolved.size());
    device_->flush_cpu_buffers(parts[0]);
    std::vector<d3d::StateDiagnostic> state_diags;
    for (size_t i = 0; i < resolved.size(); ++i) {
        resolved[i]->append_submission(parts[i + 1], &state_diags);
    }
    std::vector<std::pair<const uint8_t*, size_t>> spans;
    for (const CommandEncoder& e : parts) {
        spans.emplace_back(e.data(), e.size());
    }
    for (CommandList* list : resolved) {
        spans.emplace_back(list->encoder().data(), list->encoder().size());
    }
    // parts[i + 1] (deferred copies of list i) must precede list i's packets: interleave.
    std::vector<std::pair<const uint8_t*, size_t>> ordered;
    ordered.push_back(spans[0]);
    for (size_t i = 0; i < resolved.size(); ++i) {
        ordered.push_back(spans[1 + i]);
        ordered.push_back(spans[1 + resolved.size() + i]);
    }
    device_->submit_parts(ordered);
    for (const d3d::StateDiagnostic& d : state_diags) {
        device_->note_bad_argument(std::string("resource state at ExecuteCommandLists: ") +
                                   d3d::state_diagnostic_name(d.code) + " (resource " + std::to_string(d.resource_id) +
                                   ")");
    }
}

// ---------------------------------------------------------------------------------------------
// Command list
// ---------------------------------------------------------------------------------------------

CommandList::CommandList(com::Ref<Device> device, uint32_t type)
    : ChildBase(std::move(device)), type_(type), bridge_(device_->state_bridge(), encoder_) {
    device_->command_lists.fetch_add(1);
}

CommandList::~CommandList() {
    device_->command_lists.fetch_sub(1);
}

void CommandList::note(const std::string& message) {
    ++diag_count_;
    device_->note_bad_argument(message);
}

bool CommandList::usable() {
    if (!open_) {
        note("ID3D12GraphicsCommandList: call on a closed command list ignored");
        return false;
    }
    return true;
}

Resource* CommandList::resource(const InterfaceArg<IResourceHost>& arg) const {
    return arg ? static_cast<Resource*>(arg.get()) : nullptr;
}

void CommandList::reset(PipelineState* initial_state) {
    encoder_.clear();
    deferred_.clear();
    pinned_.clear();
    graphics_pso_ = nullptr;
    compute_pso_ = nullptr;
    bridge_.Reset(nullptr);
    open_ = true;
    if (initial_state != nullptr) {
        bind_pipeline(initial_state);
    }
}

HResult CommandList::Close() {
    if (!open_) {
        note("ID3D12GraphicsCommandList::Close on a closed list");
        return kEFail;
    }
    bridge_.Close();
    open_ = false;
    return kSOk;
}

HResult CommandList::Reset(InterfaceArg<ICommandAllocatorHost> allocator, InterfaceArg<IPipelineStateHost> initial_state) {
    if (!allocator || initial_state.invalid()) {
        note("ID3D12GraphicsCommandList::Reset: invalid allocator or pipeline state");
        return kEInvalidArg;
    }
    if (open_) {
        note("ID3D12GraphicsCommandList::Reset on an open list");
        return kEFail;
    }
    reset(initial_state ? static_cast<PipelineState*>(initial_state.get()) : nullptr);
    return kSOk;
}

void CommandList::ClearState(InterfaceArg<IPipelineStateHost> state) {
    if (!usable()) {
        return;
    }
    // Keep the recorded packets; drop the bound state and rebind `state`.
    graphics_pso_ = nullptr;
    compute_pso_ = nullptr;
    bridge_.SetPipelineState(nullptr);
    bridge_.SetComputePipelineState(nullptr);
    if (state) {
        bind_pipeline(static_cast<PipelineState*>(state.get()));
    }
}

void CommandList::bind_pipeline(PipelineState* state) {
    pinned_.push_back(com::Ref<com::ComObject>::share(state));
    if (state->is_compute()) {
        compute_pso_ = state;
        bridge_.SetComputePipelineState(state->resolved() ? state->compute() : nullptr);
    } else {
        graphics_pso_ = state;
        bridge_.SetPipelineState(state->resolved() ? state->graphics() : nullptr);
    }
    if (!state->resolved()) {
        note("pipeline state without shader modules bound: its draws/dispatches are skipped");
    }
}

void CommandList::SetPipelineState(InterfaceArg<IPipelineStateHost> state) {
    if (!usable()) {
        return;
    }
    if (!state) {
        note("SetPipelineState: invalid pipeline state " + hex(state.va()));
        return;
    }
    bind_pipeline(static_cast<PipelineState*>(state.get()));
}

void CommandList::DrawInstanced(uint32_t vertices, uint32_t instances, uint32_t start_vertex, uint32_t start_instance) {
    if (!usable()) {
        return;
    }
    if (graphics_pso_ != nullptr && !graphics_pso_->resolved()) {
        device_->note_skipped_draw();
        return;
    }
    bridge_.DrawInstanced(vertices, instances, start_vertex, start_instance);
}

void CommandList::DrawIndexedInstanced(uint32_t indices, uint32_t instances, uint32_t start_index, int32_t base_vertex,
                                       uint32_t start_instance) {
    if (!usable()) {
        return;
    }
    if (graphics_pso_ != nullptr && !graphics_pso_->resolved()) {
        device_->note_skipped_draw();
        return;
    }
    bridge_.DrawIndexedInstanced(indices, instances, start_index, base_vertex, start_instance);
}

void CommandList::Dispatch(uint32_t x, uint32_t y, uint32_t z) {
    if (!usable()) {
        return;
    }
    if (compute_pso_ != nullptr && !compute_pso_->resolved()) {
        device_->note_skipped_draw();
        return;
    }
    bridge_.Dispatch(x, y, z);
}

// ---- Copies ----

void CommandList::CopyBufferRegion(InterfaceArg<IResourceHost> dst_arg, uint64_t dst_offset,
                                   InterfaceArg<IResourceHost> src_arg, uint64_t src_offset, uint64_t bytes) {
    if (!usable()) {
        return;
    }
    Resource* dst = resource(dst_arg);
    Resource* src = resource(src_arg);
    if (dst == nullptr || src == nullptr || !dst->is_buffer() || !src->is_buffer() ||
        dst_offset + bytes > dst->desc().Width || src_offset + bytes > src->desc().Width) {
        note("CopyBufferRegion: invalid resources or range");
        return;
    }
    if (src->cpu_address() == 0) {
        device_->note_unsupported_copy("CopyBufferRegion: GPU-to-GPU buffer copy has no packet (skipped)");
        return;
    }
    pinned_.push_back(com::Ref<com::ComObject>::share(src));
    pinned_.push_back(com::Ref<com::ComObject>::share(dst));
    DeferredWrite w;
    w.id = dst->id();
    w.dst_offset = dst_offset;
    w.src = src->cpu_address() + src_offset;
    w.size = bytes;
    deferred_.push_back(w);
}

void CommandList::CopyResource(InterfaceArg<IResourceHost> dst_arg, InterfaceArg<IResourceHost> src_arg) {
    if (!usable()) {
        return;
    }
    Resource* dst = resource(dst_arg);
    Resource* src = resource(src_arg);
    if (dst == nullptr || src == nullptr) {
        note("CopyResource: invalid resources");
        return;
    }
    if (dst->is_buffer() && src->is_buffer() && src->cpu_address() != 0) {
        pinned_.push_back(com::Ref<com::ComObject>::share(src));
        DeferredWrite w;
        w.id = dst->id();
        w.src = src->cpu_address();
        w.size = std::min(dst->desc().Width, src->desc().Width);
        deferred_.push_back(w);
        return;
    }
    device_->note_unsupported_copy("CopyResource: only CPU-visible buffer sources are supported (skipped)");
}

void CommandList::CopyTextureRegion(com::CallContext& ctx) {
    // (pDst, DstX, DstY, DstZ, pSrc, pSrcBox)
    if (!usable()) {
        return;
    }
    GuestTextureCopyLocation dst{}, src{};
    if (!com::read_guest(ctx.int_arg(1), dst) || !com::read_guest(ctx.int_arg(5), src)) {
        note("CopyTextureRegion: unreadable copy locations");
        return;
    }
    Resource* tex = as_resource(dst.pResource);
    Resource* buf = as_resource(src.pResource);
    if (tex == nullptr || buf == nullptr || dst.Type != 0 || src.Type != 1 || tex->is_buffer() || !buf->is_buffer()) {
        device_->note_unsupported_copy("CopyTextureRegion: only buffer footprint -> texture subresource is supported");
        return;
    }
    if (buf->cpu_address() == 0 || !tex->has_gpu_texture()) {
        device_->note_unsupported_copy("CopyTextureRegion: source not CPU-visible or texture without GPU object");
        return;
    }
    const uint32_t sub = static_cast<uint32_t>(dst.PlacedFootprint.Offset);  // SubresourceIndex
    if (sub >= tex->subresource_count()) {
        note("CopyTextureRegion: subresource " + std::to_string(sub) + " out of range");
        return;
    }
    const GuestSubresourceFootprint& fp = src.PlacedFootprint.Footprint;
    const DxgiFormatLayout l = dxgi_format_layout(fp.Format);
    if (!l.known || fp.RowPitch == 0) {
        note("CopyTextureRegion: unknown footprint format " + std::to_string(fp.Format));
        return;
    }
    GuestBox box{0, 0, 0, fp.Width, fp.Height, fp.Depth};
    if (ctx.int_arg(6) != 0 && !com::read_guest(ctx.int_arg(6), box)) {
        note("CopyTextureRegion: unreadable source box");
        return;
    }
    const uint32_t w = box.right > box.left ? box.right - box.left : 0;
    const uint32_t h = box.bottom > box.top ? box.bottom - box.top : 0;
    const uint32_t d = box.back > box.front ? box.back - box.front : 0;
    if (w == 0 || h == 0 || d == 0) {
        return;
    }
    const uint32_t rows = (h + l.block_height - 1) / l.block_height;
    const uint32_t rows_per_slice = (fp.Height + l.block_height - 1) / l.block_height;
    const uint64_t row_bytes = uint64_t{(w + l.block_width - 1) / l.block_width} * l.block_bytes;
    const uint64_t start = src.PlacedFootprint.Offset + uint64_t{box.front} * rows_per_slice * fp.RowPitch +
                           uint64_t{box.top / l.block_height} * fp.RowPitch +
                           uint64_t{box.left / l.block_width} * l.block_bytes;
    const uint64_t len = uint64_t{d - 1} * rows_per_slice * fp.RowPitch + uint64_t{rows - 1} * fp.RowPitch + row_bytes;
    if (start + len > buf->desc().Width) {
        note("CopyTextureRegion: footprint exceeds the source buffer");
        return;
    }
    const uint32_t mip = sub % tex->mip_levels();
    const uint32_t layer = (sub / tex->mip_levels()) % tex->array_layers();
    const bool is3d = tex->desc().Dimension == ResourceDimension::Texture3D;
    pinned_.push_back(com::Ref<com::ComObject>::share(buf));
    DeferredWrite wr;
    wr.texture = true;
    wr.id = tex->id();
    wr.src = buf->cpu_address() + start;
    wr.size = len;
    wr.tex.texture_id = tex->id();
    wr.tex.mip_level = mip;
    wr.tex.origin_x = static_cast<uint32_t>(ctx.int_arg(2));
    wr.tex.origin_y = static_cast<uint32_t>(ctx.int_arg(3));
    wr.tex.origin_z = is3d ? static_cast<uint32_t>(ctx.int_arg(4)) : layer;
    wr.tex.width = w;
    wr.tex.height = h;
    wr.tex.depth_or_layers = d;
    wr.tex.bytes_per_row = fp.RowPitch;
    wr.tex.rows_per_image = rows_per_slice;
    deferred_.push_back(wr);
}

void CommandList::ResolveSubresource(com::CallContext&) {
    if (usable()) {
        device_->note_unsupported_copy("ResolveSubresource: no resolve packet (skipped)");
    }
}

void CommandList::ClearUnorderedAccessView(com::CallContext&) {
    if (usable()) {
        device_->note_unsupported_copy("ClearUnorderedAccessView: no UAV clear packet (skipped)");
    }
}

void CommandList::QueryOperation(com::CallContext&) {
    // Queries (BeginQuery / EndQuery / ResolveQueryData) have no packet; results stay zero.
}

void CommandList::append_submission(CommandEncoder& out, std::vector<d3d::StateDiagnostic>* state_diagnostics) {
    for (const DeferredWrite& w : deferred_) {
        std::vector<uint8_t> bytes(static_cast<size_t>(w.size));
        if (!com::read_guest(w.src, bytes.data(), bytes.size())) {
            device_->note_bad_argument("ExecuteCommandLists: copy source " + hex(w.src) + " no longer readable");
            continue;
        }
        if (w.texture) {
            out.write_texture(w.tex.texture_id, w.tex.mip_level, w.tex.origin_x, w.tex.origin_y, w.tex.origin_z,
                              w.tex.width, w.tex.height, w.tex.depth_or_layers, w.tex.bytes_per_row,
                              w.tex.rows_per_image, bytes.data(), static_cast<uint32_t>(bytes.size()));
        } else {
            for (uint64_t off = 0; off < w.size; off += kMaxWriteChunk) {
                const uint64_t n = std::min(kMaxWriteChunk, w.size - off);
                out.write_buffer(w.id, static_cast<uint32_t>(w.dst_offset + off), bytes.data() + off,
                                 static_cast<uint32_t>(n));
            }
        }
        device_->note_upload(w.size);
    }
    device_->resource_states().execute(bridge_.resource_states(), state_diagnostics);
}

// ---- Fixed-function state ----

void CommandList::IASetPrimitiveTopology(uint32_t topology) {
    if (usable()) {
        bridge_.IASetPrimitiveTopology(static_cast<pso::D3DPrimitiveTopology>(topology));
    }
}

void CommandList::RSSetViewports(uint32_t count, uint64_t viewports) {
    std::vector<pso::D3D12Viewport> v;
    static_assert(sizeof(pso::D3D12Viewport) == sizeof(GuestViewport));
    if (!usable()) {
        return;
    }
    if (count > kMaxViewports || !read_n(viewports, count, &v)) {
        note("RSSetViewports: unreadable viewports");
        return;
    }
    bridge_.RSSetViewports(count, v.data());
}

void CommandList::RSSetScissorRects(uint32_t count, uint64_t rects) {
    std::vector<pso::D3D12Rect> r;
    static_assert(sizeof(pso::D3D12Rect) == sizeof(GuestRect));
    if (!usable()) {
        return;
    }
    if (count > kMaxViewports || !read_n(rects, count, &r)) {
        note("RSSetScissorRects: unreadable rects");
        return;
    }
    bridge_.RSSetScissorRects(count, r.data());
}

void CommandList::OMSetBlendFactor(uint64_t factor) {
    if (!usable()) {
        return;
    }
    float f[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (factor != 0 && !com::read_guest(factor, f, sizeof(f))) {
        note("OMSetBlendFactor: unreadable factor");
        return;
    }
    bridge_.OMSetBlendFactor(f);
}

void CommandList::OMSetStencilRef(uint32_t ref) {
    if (usable()) {
        bridge_.OMSetStencilRef(ref);
    }
}

void CommandList::ResourceBarrier(uint32_t count, uint64_t barriers) {
    if (!usable()) {
        return;
    }
    std::vector<GuestResourceBarrier> g;
    if (count > kMaxBarriers || !read_n(barriers, count, &g)) {
        note("ResourceBarrier: unreadable barrier array");
        return;
    }
    std::vector<d3d::ResourceBarrier> b;
    b.reserve(count);
    auto id_of = [&](uint64_t va) -> uint32_t {
        Resource* r = as_resource(va);
        return r != nullptr ? r->id() : 0u;
    };
    for (const GuestResourceBarrier& x : g) {
        d3d::ResourceBarrier r;
        r.type = static_cast<d3d::ResourceBarrierType>(x.Type);
        r.flags = x.Flags;
        switch (x.Type) {
        case 0:
            r.resource_id = id_of(x.pResource);
            r.subresource = x.u0;
            r.state_before = x.u1;
            r.state_after = x.StateAfter;
            if (r.resource_id == 0) {
                note("ResourceBarrier: transition of an unknown resource " + hex(x.pResource));
                continue;
            }
            break;
        case 1:
            r.resource_before = id_of(x.pResource);
            r.resource_after = id_of(uint64_t{x.u0} | (uint64_t{x.u1} << 32));
            break;
        case 2:
            r.resource_id = id_of(x.pResource);
            break;
        default:
            note("ResourceBarrier: invalid type " + std::to_string(x.Type));
            continue;
        }
        b.push_back(r);
    }
    bridge_.ResourceBarrier(static_cast<uint32_t>(b.size()), b.data());
}

void CommandList::SetDescriptorHeaps(uint32_t count, uint64_t heaps) {
    if (!usable()) {
        return;
    }
    std::vector<uint64_t> ptrs;
    if (count > 2 || !read_n(heaps, count, &ptrs)) {
        note("SetDescriptorHeaps: unreadable heap array or more than 2 heaps");
        return;
    }
    std::vector<d3d::DescriptorHeap*> list;
    for (uint64_t p : ptrs) {
        auto arg = InterfaceArg<IDescriptorHeapHost>::resolve(p);
        if (!arg) {
            note("SetDescriptorHeaps: invalid heap " + hex(p));
            return;
        }
        pinned_.push_back(com::Ref<com::ComObject>::share(arg.object()));
        list.push_back(static_cast<DescriptorHeap*>(arg.get())->heap());
    }
    const d3d::Status s = bridge_.SetDescriptorHeaps(static_cast<uint32_t>(list.size()), list.data());
    if (s != d3d::Status::Ok) {
        note(std::string("SetDescriptorHeaps: ") + d3d::status_name(s));
    }
}

namespace {
const d3d::RootSignature* signature_of(const InterfaceArg<IRootSignatureHost>& arg) {
    return arg ? &static_cast<RootSignature*>(arg.get())->signature() : nullptr;
}
}  // namespace

void CommandList::SetComputeRootSignature(InterfaceArg<IRootSignatureHost> signature) {
    if (!usable()) {
        return;
    }
    if (signature) {
        pinned_.push_back(com::Ref<com::ComObject>::share(signature.object()));
    }
    const d3d::Status s = bridge_.SetComputeRootSignature(signature_of(signature));
    if (s != d3d::Status::Ok) {
        note(std::string("SetComputeRootSignature: ") + d3d::status_name(s));
    }
}

void CommandList::SetGraphicsRootSignature(InterfaceArg<IRootSignatureHost> signature) {
    if (!usable()) {
        return;
    }
    if (signature) {
        pinned_.push_back(com::Ref<com::ComObject>::share(signature.object()));
    }
    const d3d::Status s = bridge_.SetGraphicsRootSignature(signature_of(signature));
    if (s != d3d::Status::Ok) {
        note(std::string("SetGraphicsRootSignature: ") + d3d::status_name(s));
    }
}

#define PG6_ROOT_CALL(expr, what)                                       \
    do {                                                                \
        if (!usable()) {                                                \
            return;                                                     \
        }                                                               \
        const d3d::Status s_ = (expr);                                  \
        if (s_ != d3d::Status::Ok) {                                    \
            note(std::string(what ": ") + d3d::status_name(s_));        \
        }                                                               \
    } while (0)

void CommandList::SetComputeRootDescriptorTable(uint32_t index, uint64_t base) {
    PG6_ROOT_CALL(bridge_.SetComputeRootDescriptorTable(index, d3d::GpuDescriptorHandle{base}), "SetComputeRootDescriptorTable");
}

void CommandList::SetGraphicsRootDescriptorTable(uint32_t index, uint64_t base) {
    PG6_ROOT_CALL(bridge_.SetGraphicsRootDescriptorTable(index, d3d::GpuDescriptorHandle{base}), "SetGraphicsRootDescriptorTable");
}

void CommandList::SetComputeRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) {
    PG6_ROOT_CALL(bridge_.SetComputeRoot32BitConstants(index, 1, &value, offset), "SetComputeRoot32BitConstant");
}

void CommandList::SetGraphicsRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) {
    PG6_ROOT_CALL(bridge_.SetGraphicsRoot32BitConstants(index, 1, &value, offset), "SetGraphicsRoot32BitConstant");
}

void CommandList::SetComputeRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values, uint32_t offset) {
    std::vector<uint32_t> v;
    if (count > d3d::kMaxRootSignatureDwords || !read_n(values, count, &v)) {
        note("SetComputeRoot32BitConstants: unreadable values");
        return;
    }
    PG6_ROOT_CALL(bridge_.SetComputeRoot32BitConstants(index, count, v.data(), offset), "SetComputeRoot32BitConstants");
}

void CommandList::SetGraphicsRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values, uint32_t offset) {
    std::vector<uint32_t> v;
    if (count > d3d::kMaxRootSignatureDwords || !read_n(values, count, &v)) {
        note("SetGraphicsRoot32BitConstants: unreadable values");
        return;
    }
    PG6_ROOT_CALL(bridge_.SetGraphicsRoot32BitConstants(index, count, v.data(), offset), "SetGraphicsRoot32BitConstants");
}

void CommandList::SetComputeRootConstantBufferView(uint32_t index, uint64_t va) {
    PG6_ROOT_CALL(bridge_.SetComputeRootConstantBufferView(index, va), "SetComputeRootConstantBufferView");
}

void CommandList::SetGraphicsRootConstantBufferView(uint32_t index, uint64_t va) {
    PG6_ROOT_CALL(bridge_.SetGraphicsRootConstantBufferView(index, va), "SetGraphicsRootConstantBufferView");
}

void CommandList::SetComputeRootShaderResourceView(uint32_t, uint64_t) {
    if (usable()) {
        note("SetComputeRootShaderResourceView: compute root SRVs are not supported by the state bridge (ignored)");
    }
}

void CommandList::SetGraphicsRootShaderResourceView(uint32_t index, uint64_t va) {
    PG6_ROOT_CALL(bridge_.SetGraphicsRootShaderResourceView(index, va), "SetGraphicsRootShaderResourceView");
}

void CommandList::SetComputeRootUnorderedAccessView(uint32_t, uint64_t) {
    if (usable()) {
        note("SetComputeRootUnorderedAccessView: compute root UAVs are not supported by the state bridge (ignored)");
    }
}

void CommandList::SetGraphicsRootUnorderedAccessView(uint32_t index, uint64_t va) {
    PG6_ROOT_CALL(bridge_.SetGraphicsRootUnorderedAccessView(index, va), "SetGraphicsRootUnorderedAccessView");
}

#undef PG6_ROOT_CALL

// ---- Input assembler ----

void CommandList::IASetIndexBuffer(uint64_t view) {
    if (!usable()) {
        return;
    }
    if (view == 0) {
        bridge_.IASetIndexBuffer(nullptr);
        return;
    }
    GuestIndexBufferView g{};
    if (!com::read_guest(view, g)) {
        note("IASetIndexBuffer: unreadable view");
        return;
    }
    d3d::GpuVaResolution res;
    if (!device_->state_bridge().resolve_gpu_va(g.BufferLocation, &res)) {
        note("IASetIndexBuffer: unknown GPU virtual address " + hex(g.BufferLocation));
        bridge_.IASetIndexBuffer(nullptr);
        return;
    }
    if (g.Format != 57 && g.Format != 42) {
        note("IASetIndexBuffer: invalid index format " + std::to_string(g.Format));
        return;
    }
    d3d12_xs::D3D12IndexBufferView v;
    v.buffer_id = res.resource_id;
    v.offset = res.offset;
    v.size_in_bytes = g.SizeInBytes;
    v.format = g.Format == 57 ? GpuIndexFormat::Uint16 : GpuIndexFormat::Uint32;
    bridge_.IASetIndexBuffer(&v);
}

void CommandList::IASetVertexBuffers(uint32_t start, uint32_t count, uint64_t views) {
    if (!usable()) {
        return;
    }
    if (start >= pso::kD3D12IaVertexInputSlotCount || count > pso::kD3D12IaVertexInputSlotCount - start) {
        note("IASetVertexBuffers: slot range out of bounds");
        return;
    }
    if (views == 0) {
        bridge_.IASetVertexBuffers(start, count, nullptr);
        return;
    }
    std::vector<GuestVertexBufferView> g;
    if (!read_n(views, count, &g)) {
        note("IASetVertexBuffers: unreadable views");
        return;
    }
    std::vector<d3d12_xs::D3D12VertexBufferView> v(count);
    for (uint32_t i = 0; i < count; ++i) {
        d3d::GpuVaResolution res;
        if (g[i].BufferLocation != 0 && device_->state_bridge().resolve_gpu_va(g[i].BufferLocation, &res)) {
            v[i].buffer_id = res.resource_id;
            v[i].offset = res.offset;
        } else if (g[i].BufferLocation != 0) {
            note("IASetVertexBuffers: unknown GPU virtual address " + hex(g[i].BufferLocation));
        }
        v[i].size_in_bytes = g[i].SizeInBytes;
        v[i].stride_in_bytes = g[i].StrideInBytes;
    }
    bridge_.IASetVertexBuffers(start, count, v.data());
}

void CommandList::SOSetTargets(uint32_t, uint32_t count, uint64_t) {
    if (usable() && count != 0) {
        note("SOSetTargets: stream output is not supported (ignored)");
    }
}

// ---- Output merger ----

void CommandList::OMSetRenderTargets(uint32_t count, uint64_t rtvs, bool single_range, uint64_t dsv) {
    if (!usable()) {
        return;
    }
    if (count > 8) {
        note("OMSetRenderTargets: more than 8 render targets");
        return;
    }
    std::vector<d3d::CpuDescriptorHandle> handles;
    if (!read_n(rtvs, count == 0 ? 0u : (single_range ? 1u : count), &handles)) {
        note("OMSetRenderTargets: unreadable render target handles");
        return;
    }
    d3d::CpuDescriptorHandle depth{};
    if (dsv != 0 && !com::read_guest(dsv, depth)) {
        note("OMSetRenderTargets: unreadable depth-stencil handle");
        return;
    }
    bridge_.OMSetRenderTargets(count, handles.empty() ? nullptr : handles.data(), single_range,
                               dsv != 0 ? &depth : nullptr);
}

void CommandList::ClearDepthStencilView(uint64_t dsv, uint32_t flags, float depth, uint8_t stencil,
                                        uint32_t rect_count, uint64_t rects) {
    if (!usable()) {
        return;
    }
    std::vector<pso::D3D12Rect> r;
    if (rect_count > kMaxRects || !read_n(rects, rect_count, &r)) {
        note("ClearDepthStencilView: unreadable rects");
        return;
    }
    bridge_.ClearDepthStencilView(d3d::CpuDescriptorHandle{dsv}, flags, depth, stencil, rect_count,
                                  r.empty() ? nullptr : r.data());
}

void CommandList::ClearRenderTargetView(uint64_t rtv, uint64_t color, uint32_t rect_count, uint64_t rects) {
    if (!usable()) {
        return;
    }
    float c[4] = {};
    std::vector<pso::D3D12Rect> r;
    if (color == 0 || !com::read_guest(color, c, sizeof(c)) || rect_count > kMaxRects || !read_n(rects, rect_count, &r)) {
        note("ClearRenderTargetView: unreadable color or rects");
        return;
    }
    bridge_.ClearRenderTargetView(d3d::CpuDescriptorHandle{rtv}, c, rect_count, r.empty() ? nullptr : r.data());
}

// ---------------------------------------------------------------------------------------------
// Vtable (Xbox ID3D12GraphicsCommandList order, see d3d12_xs_xbox.h)
// ---------------------------------------------------------------------------------------------

namespace {

using CL = Methods<IGraphicsCommandListHost>;
using H = IGraphicsCommandListHost;
constexpr com::MethodDesc kCommandListMethods[] = {
    CL::method<&IObjectHost::GetPrivateData>("GetPrivateData"),                       // 3
    CL::method<&IObjectHost::SetPrivateData>("SetPrivateData"),                       // 4
    CL::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),     // 5
    CL::method<&IObjectHost::SetName>("SetName"),                                     // 6
    CL::method<&IDeviceChildHost::GetDevice>("GetDevice"),                            // 7
    CL::method<&H::GetType>("GetType"),                                               // 8
    CL::method<&H::Close>("Close"),                                                   // 9
    CL::method<&H::Reset>("Reset"),                                                   // 10
    CL::method<&H::ClearState>("ClearState"),                                         // 11
    CL::method<&H::DrawInstanced>("DrawInstanced"),                                   // 12
    CL::method<&H::DrawIndexedInstanced>("DrawIndexedInstanced"),                     // 13
    CL::method<&H::Dispatch>("Dispatch"),                                             // 14
    CL::method<&H::CopyBufferRegion>("CopyBufferRegion"),                             // 15
    CL::method<&H::CopyTextureRegion>("CopyTextureRegion"),                           // 16
    CL::method<&H::CopyResource>("CopyResource"),                                     // 17
    CL::stub("CopyTiles"),                                                            // 18
    CL::method<&H::ResolveSubresource>("ResolveSubresource"),                         // 19
    CL::method<&H::IASetPrimitiveTopology>("IASetPrimitiveTopology"),                 // 20
    CL::method<&H::RSSetViewports>("RSSetViewports"),                                 // 21
    CL::method<&H::RSSetScissorRects>("RSSetScissorRects"),                           // 22
    CL::method<&H::OMSetBlendFactor>("OMSetBlendFactor"),                             // 23
    CL::method<&H::OMSetStencilRef>("OMSetStencilRef"),                               // 24
    CL::method<&H::SetPipelineStateX>("SetPipelineState (staged)"),                   // 25
    CL::method<&H::ResourceBarrier>("ResourceBarrier"),                               // 26
    CL::stub("ExecuteBundle"),                                                        // 27
    // Xbox order from here on (d3d12_xs_xbox.h): queries and predication moved up, public 28..51 +4.
    CL::method<&H::QueryOperation>("BeginQuery"),                                     // 28
    CL::method<&H::QueryOperation>("EndQuery"),                                       // 29
    CL::method<&H::QueryOperation>("ResolveQueryData"),                               // 30
    CL::method<&H::SetPredication>("SetPredication (inferred)"),                      // 31
    CL::method<&H::SetDescriptorHeaps>("SetDescriptorHeaps"),                         // 32
    CL::method<&H::SetComputeRootSignature>("SetComputeRootSignature"),               // 33
    CL::method<&H::SetGraphicsRootSignature>("SetGraphicsRootSignature"),             // 34
    CL::method<&H::SetComputeRootDescriptorTable>("SetComputeRootDescriptorTable"),   // 35
    CL::method<&H::SetGraphicsRootDescriptorTable>("SetGraphicsRootDescriptorTable"), // 36
    CL::method<&H::SetComputeRoot32BitConstant>("SetComputeRoot32BitConstant"),       // 37
    CL::method<&H::SetGraphicsRoot32BitConstant>("SetGraphicsRoot32BitConstant"),     // 38
    CL::method<&H::SetComputeRoot32BitConstants>("SetComputeRoot32BitConstants"),     // 39
    CL::method<&H::SetGraphicsRoot32BitConstants>("SetGraphicsRoot32BitConstants"),   // 40
    CL::method<&H::SetComputeRootConstantBufferView>("SetComputeRootConstantBufferView"),    // 41
    CL::method<&H::SetGraphicsRootConstantBufferView>("SetGraphicsRootConstantBufferView"),  // 42
    CL::method<&H::SetComputeRootShaderResourceView>("SetComputeRootShaderResourceView"),    // 43
    CL::method<&H::SetGraphicsRootShaderResourceView>("SetGraphicsRootShaderResourceView"),  // 44
    CL::method<&H::SetComputeRootUnorderedAccessView>("SetComputeRootUnorderedAccessView"),  // 45
    CL::method<&H::SetGraphicsRootUnorderedAccessView>("SetGraphicsRootUnorderedAccessView"),  // 46
    CL::method<&H::IASetIndexBuffer>("IASetIndexBuffer"),                             // 47
    CL::method<&H::IASetVertexBuffers>("IASetVertexBuffers"),                         // 48
    CL::method<&H::SOSetTargets>("SOSetTargets"),                                     // 49
    CL::method<&H::OMSetRenderTargets>("OMSetRenderTargets"),                         // 50
    CL::method<&H::ClearDepthStencilView>("ClearDepthStencilView"),                   // 51
    CL::method<&H::ClearRenderTargetView>("ClearRenderTargetView"),                   // 52
    CL::method<&H::ClearUnorderedAccessView>("ClearUnorderedAccessViewUint"),         // 53
    CL::method<&H::ClearUnorderedAccessView>("ClearUnorderedAccessViewFloat"),        // 54
    CL::method<&H::DiscardResource>("DiscardResource"),                               // 55
    CL::method<&H::DebugMarker>("SetMarker (inferred)"),                              // 56
    CL::method<&H::DebugMarker>("BeginEvent (inferred)"),                             // 57
    CL::method<&H::DebugMarker>("EndEvent (inferred)"),                               // 58
    CL::method<&H::XboxSlot<59>>("Xbox slot 59"),                                     // 59
    CL::method<&H::XboxSlot<60>>("Xbox slot 60"),                                     // 60
    CL::method<&H::ExecuteIndirect>("ExecuteIndirect"),                               // 61
    CL::method<&H::XboxSlot<62>>("Xbox slot 62"),                                     // 62
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 63..66
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),                          // 67..69
    CL::method<&H::OMSetDepthBounds>("OMSetDepthBounds"),                             // 70
    CL::stub(nullptr),                                                                // 71
    CL::method<&H::XboxSlot<72>>("Xbox slot 72"),                                     // 72
    CL::stub(nullptr), CL::stub(nullptr),                                             // 73..74
    CL::method<&H::XboxSlot<75>>("Xbox slot 75"),                                     // 75
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 76..79
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 80..83
    CL::stub(nullptr), CL::stub(nullptr),                                             // 84..85
    CL::method<&H::XboxSlot<86>>("Xbox slot 86"),                                     // 86
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),                          // 87..89
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 90..93
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 94..97
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 98..101
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 102..105
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 106..109
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 110..113
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 114..117
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 118..121
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 122..125
    CL::method<&H::RSSetShadingRate>("RSSetShadingRate"),                             // 126
    CL::method<&H::RSSetShadingRateImage>("RSSetShadingRateImage"),                   // 127
    CL::method<&H::XboxSlot<128>>("Xbox slot 128"),                                   // 128
    CL::method<&H::XboxSlot<129>>("Xbox slot 129"),                                   // 129
    CL::stub(nullptr),                                                                // 130
    CL::method<&H::XboxSlot<131>>("Xbox slot 131 (register write)"),                  // 131
    CL::method<&H::XboxSlot<132>>("Xbox slot 132 (register write)"),                  // 132
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 133..136
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 137..140
    CL::stub(nullptr),                                                                // 141
    CL::method<&H::XboxSlot<142>>("Xbox slot 142"),                                   // 142
};
constexpr Guid kCommandListIids[] = {
    IID_ID3D12Object,
    IID_ID3D12DeviceChild,
    IID_ID3D12CommandList,
    com::make_guid("553103fb-1fe7-4557-bb38-946d7d0e7ca7"),  // ID3D12GraphicsCommandList1
    com::make_guid("38c3e585-ff17-412c-9150-4fc6f9d72a28"),  // ID3D12GraphicsCommandList2
    com::make_guid("6fda83a7-b84c-4e38-9ac8-c7bd22016b3d"),  // ID3D12GraphicsCommandList3
    com::make_guid("8754318e-d3a9-4541-98cf-645b50dc4874"),  // ID3D12GraphicsCommandList4
    com::make_guid("55050859-4024-474c-87f5-6472eaee44ea"),  // ID3D12GraphicsCommandList5
    com::make_guid("c3827890-e548-4cfa-96cf-5689a9370f80"),  // ID3D12GraphicsCommandList6
    // Requested by the title from CreateCommandList (Xbox slot 73), first for a direct list: the
    // Xbox runtime's own graphics-command-list IID (the vtable this interface implements, Xbox
    // slots included, was built from the title's command-list calls).
    com::make_guid("c3827890-4024-474c-98cf-645b51dc4874"),
};

constexpr com::InterfaceOptions list_options() {
    com::InterfaceOptions o;
    o.slot_count = 192;  // Xbox-specific slots up to at least 132 are called (logging stubs)
    o.policy = com::SlotPolicy::ReturnNotImpl;
    o.extra_iids = kCommandListIids;
    return o;
}

}  // namespace

const com::Interface<IGraphicsCommandListHost> kID3D12GraphicsCommandList =
    com::declare_interface<IGraphicsCommandListHost>("ID3D12GraphicsCommandList", IID_ID3D12GraphicsCommandList,
                                                     kCommandListMethods, list_options());

}  // namespace playgta6::gpu::d3d12_guest
