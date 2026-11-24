#pragma once

// Guest D3D12 resources, heaps, pipeline states and graphics command lists (G24, part 2), on top
// of the device objects in d3d12_guest_objects.h.
//
// Memory and GPU virtual addresses
// --------------------------------
// The console is UMA: a buffer's GPU virtual address is the CPU address of its memory, and the
// executable creates most resources at addresses it allocated itself (device slot 133). The layer
// follows that model:
//  * Buffers in CPU-visible memory (UPLOAD / READBACK / CUSTOM heaps with CPU access, and every
//    resource created at a caller address that is guest memory) get guest memory; their GPU VA is
//    that guest address. Committed ones get a fresh 64 KiB-aligned guest allocation, placed ones the
//    heap's memory + offset.
//  * Buffers without CPU access (DEFAULT heaps, reserved resources) get a GPU VA from the device's
//    fake range (Device::kFakeGpuVaBase); they have no guest memory.
//  * Textures: on the console every resource has a GPU VA, the address of its memory, and the
//    executable uses it (it translates a texture's GPU VA into a CPU pointer and writes the
//    texels itself). A texture's GPU VA is the memory it was created in: the caller address
//    (slot 133), the placed heap's memory + offset, or its committed XMemAlloc memory; 0 only
//    for a texture the layer gives no memory (committed textures other than render targets).
//    Its descriptor words (descriptor_words.h) carry that address; a texture without memory gets
//    a descriptor address from the fake GPU VA range instead (not reported as its GPU VA).
// Every buffer is registered with the state bridge (VA -> buffer id) so CBVs, root views and
// vertex/index buffer views resolve; every texture with its format, size and sample count.
//
// How data reaches the GPU process
// --------------------------------
// The packet ABI has WriteBuffer / WriteTexture (CPU -> GPU) and nothing for GPU -> CPU or GPU -> GPU.
//  * CPU-visible buffers keep a host shadow of their contents. At every ExecuteCommandLists the
//    device compares each tracked buffer with its shadow in 4 KiB pages and emits WriteBuffer for
//    the changed runs (before the command lists' packets). Map/Unmap need no bookkeeping, so
//    persistently mapped buffers and memory the guest writes without Map work the same way. Cost:
//    one memcmp of every tracked buffer per submission.
//  * CopyBufferRegion / CopyResource from a CPU-visible buffer and CopyTextureRegion from a placed
//    footprint in a CPU-visible buffer become WriteBuffer / WriteTexture packets built from guest
//    memory at ExecuteCommandLists time (the source holds what the GPU would read when the list
//    executes). Copies from GPU-only memory, into CPU-visible memory, resolves and queries have no
//    packet: they are diagnosed (Device::stats().unsupported_copies) and skipped.
//  * The WebGPU worker applies WriteBuffer / WriteTexture as queue writes, i.e. before the frame's
//    command buffer runs; a list that rewrites a buffer between two draws of the same frame sees
//    the last write in both (limitation of the worker's frame model, not of this layer).
//
// Shaders
// -------
// PSO creation receives DXIL containers. A shader's identity is FNV-1a 64 over the exact container
// bytes (pShaderBytecode[0, BytecodeLength)), the same function and "0x%016x" spelling the offline
// pipeline uses for its SPIR-V hashes (shaders/src/convert.rs). The offline manifest keys modules by
// the SPIR-V hash, so it needs a DXIL identity -> module entry; Device::register_shader_module
// installs that mapping (StateBridgeDevice::register_shader_module) and the pipeline layout for the
// shader pair goes through StateBridgeDevice::register_pipeline_layout. A PSO whose shaders have no
// module is "unresolved": binding it diagnoses, records the identities (Device::unresolved_shaders)
// and every draw with it is skipped (Device::stats().skipped_draws), never emitted with module 0.
// Console-native pipelines (Xbox slot 57) carry GPU machine code instead of DXIL: their stages get
// native_stage_identity() identities and they are always unresolved (d3d12_xs_xbox.h).
//
// Command lists
// -------------
// One bridge::CommandListBridge + CommandEncoder per command list (single recording thread, like
// D3D12). Reset clears the encoder; ExecuteCommandLists (queue Xbox slot 35) submits, per list:
// pending shadow uploads, the list's deferred CPU-sourced copies, then its packet stream, and
// commits its resource states to the device registry (G6 ResourceStateRegistry::execute).

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gpu/d3d12_guest_objects.h"

namespace playgta6::gpu::d3d12_guest {

// FNV-1a 64 over `size` bytes (shader identity, see the header comment).
uint64_t shader_identity(const void* data, size_t size);
// D3D12_SHADER_RESOURCE_VIEW_DESC / D3D12_UNORDERED_ACCESS_VIEW_DESC -> registry view description
// (false for an invalid view dimension). Also used by the Xbox views at an address (slots 70 / 71).
bool parse_srv_desc(const GuestShaderResourceViewDesc& g, d3d::ViewDesc* out);
bool parse_uav_desc(const GuestUnorderedAccessViewDesc& g, d3d::ViewDesc* out);
// Identity of a console-native stage (Xbox slot 57): FNV-1a 64 over the code bytes followed by the
// stage's metadata bytes (one hash over the concatenation).
uint64_t native_stage_identity(const void* code, size_t code_size, const void* metadata, size_t metadata_size);

// Diagnostics for textures whose memory the title writes directly (no upload call): every texture
// with readable guest memory is registered; probe_texture_contents() looks at up to `budget` of
// them (round robin) and reports each the first time its first 64 KiB are not all zero (format,
// size, mips, layout; once, through the log). Nothing is uploaded.
struct TextureProbeTotals {
    uint64_t registered = 0;  // live textures with guest memory
    uint64_t with_data = 0;   // of all registered so far, seen with non-zero memory
};
TextureProbeTotals probe_texture_contents(size_t budget);

class Heap final : public ChildBase<IHeapHost> {
public:
    static constexpr const char* kComClassName = "ID3D12Heap";
    Heap(com::Ref<Device> device, const GuestHeapDesc& desc);
    ~Heap() override;
    GuestHeapDesc GetDesc() override { return desc_; }
    bool valid() const { return base_ != 0; }
    bool cpu_visible() const { return cpu_visible_; }
    uint64_t base() const { return base_; }  // guest address (CPU-visible) or fake GPU VA
    const GuestHeapDesc& desc() const { return desc_; }

private:
    GuestHeapDesc desc_;
    bool cpu_visible_ = false;
    bool owns_guest_memory_ = false;
    uint64_t base_ = 0;
};

class Resource final : public ChildBase<IResourceHost> {
public:
    static constexpr const char* kComClassName = "ID3D12Resource";

    struct Placement {
        GuestHeapProperties heap{};
        uint32_t heap_flags = 0;
        com::Ref<Heap> placed_heap;  // placed resources
        uint64_t heap_offset = 0;
        uint64_t address = 0;        // resources created at a caller address (Xbox slot 133)
        bool reserved = false;
    };

    Resource(com::Ref<Device> device, const GuestResourceDesc& desc, Placement placement, uint32_t initial_state);
    ~Resource() override;
    // Creates the GPU object (packets) and registrations; false (diagnosed) if it cannot exist.
    bool initialize();

    // ---- IResourceHost ----
    HResult Map(uint32_t subresource, uint64_t read_range, OutPtr data) override;
    void Unmap(uint32_t subresource, uint64_t written_range) override;
    GuestResourceDesc GetDesc() override { return desc_; }
    uint64_t GetGPUVirtualAddress() override { return gpu_va_; }
    HResult WriteToSubresource(com::CallContext& ctx) override;
    HResult ReadFromSubresource(com::CallContext& ctx) override;
    HResult GetHeapProperties(GuestPtr<GuestHeapProperties> properties, GuestPtr<uint32_t> flags) override;
    ResourceDesc1Result GetDesc1() override;

    // ---- Runtime side ----
    uint32_t id() const { return id_; }
    bool is_buffer() const { return desc_.Dimension == ResourceDimension::Buffer; }
    const GuestResourceDesc& desc() const { return desc_; }
    uint64_t gpu_va() const { return gpu_va_; }
    uint64_t cpu_address() const { return cpu_va_; }  // guest address of CPU-visible memory, 0 if none
    uint32_t mip_levels() const { return mips_; }
    uint32_t array_layers() const;
    uint32_t subresource_count() const;
    bool has_gpu_texture() const { return gpu_texture_; }
    // Shadow comparison (CPU-visible buffers): appends WriteBuffer packets for changed pages.
    void flush_shadow(CommandEncoder& encoder);
    // Marks the texture as the presented back buffer (Device::set_back_buffer).
    void mark_back_buffer();
    bool is_back_buffer() const { return back_buffer_; }

private:
    GuestResourceDesc desc_;
    Placement placement_;
    uint32_t initial_state_;
    uint32_t id_ = 0;
    uint32_t mips_ = 1;
    uint64_t gpu_va_ = 0;
    uint64_t descriptor_va_ = 0;  // textures: the address their descriptor words carry
    uint64_t cpu_va_ = 0;
    void* xmem_memory_ = nullptr;  // committed render-target memory from XMemAlloc (see initialize)
    uint64_t owned_guest_memory_ = 0;
    bool gpu_texture_ = false;
    bool tracked_ = false;
    std::atomic<bool> back_buffer_{false};
    std::vector<uint8_t> shadow_;
    friend TextureProbeTotals probe_texture_contents(size_t budget);
    bool probe_registered_ = false;
    bool probe_seen_data_ = false;
};

// Host-side copy of a graphics pipeline description (Xbox device slot 55 derives new pipeline
// states from an existing one, so the description must outlive the guest structs it came from).
struct GraphicsPsoSource {
    pso::D3D12GraphicsPsoDesc desc{};
    std::vector<std::string> semantic_names;
    std::vector<pso::D3D12InputElementDesc> input;
    com::Ref<RootSignature> root_signature;
    bool rect_list = false;  // Xbox rectangle topology type (desc says Triangle; d3d12_xs_xbox.h)
    // The console-native stages the DXIL vertex / pixel shader joins to (identity 0: not joined).
    Device::NativeStageRecord native_vs, native_ps;
    Device::DxilDigest vs_digest{}, ps_digest{};  // HASH digests of the DXIL containers (zero: none)
    uint64_t pack_vs_key = 0, pack_ps_key = 0;   // shader pack keys the stages resolved with (0: none)

    GraphicsPsoSource() = default;
    GraphicsPsoSource(const GraphicsPsoSource& other) { *this = other; }
    GraphicsPsoSource& operator=(const GraphicsPsoSource& other);
    // Points desc.pInputElementDescs and every SemanticName at this object's own storage.
    void fix_pointers();
};

// The pipeline layout Device::resolve_with_pack registers for a pack-resolved pipeline: one group,
// bindings numbered from 0 in root-parameter order (table ranges descriptor by descriptor, an
// unbounded range as one; root CBV/SRV/UAV and root constants one each; then static samplers).
// CBVs and root constants are uniform buffers, SRVs 2D float textures, UAVs storage buffers,
// samplers filtering samplers; every binding is visible to the vertex and fragment stages.
bridge::PipelineLayoutDesc layout_from_root_signature(const d3d::RootSignatureDesc& rs);

// D3D12 guest struct -> pipeline-cache description (shared by the public and the Xbox creators).
void convert_blend_desc(const GuestBlendDesc& g, pso::D3D12BlendDesc* out);
void convert_rasterizer_desc(const GuestRasterizerDesc& g, pso::D3D12RasterizerDesc* out);
void convert_depth_stencil_desc(const GuestDepthStencilDesc& g, pso::D3D12DepthStencilDesc* out);

class PipelineState final : public ChildBase<IPipelineStateHost> {
public:
    static constexpr const char* kComClassName = "ID3D12PipelineState";
    // Guest payload at pso + 0x10. The executable copies these 32 bytes into the command list
    // (list + 0x544 .. 0x563) before calling SetPipelineState with only `this` (see d3d12_xs_xbox.h):
    // payload[0] = kPipelineStateTag, payload[1] = the PSO's own interface pointer.
    static constexpr uint64_t kComPayloadSize = 32;
    PipelineState(com::Ref<Device> device, com::Ref<RootSignature> root_signature);
    ~PipelineState() override;
    HResult GetCachedBlob(OutPtr blob) override;

    // Writes the guest payload (must run once right after create_object).
    void initialize_guest_payload();
    bool is_compute() const { return compute_ != nullptr; }
    bool resolved() const { return resolved_; }
    bridge::GraphicsPipelineState* graphics() const { return graphics_.get(); }
    bridge::ComputePipelineState* compute() const { return compute_.get(); }
    const std::vector<uint64_t>& shader_identities() const { return identities_; }
    // Graphics description (null for compute and placeholder pipeline states).
    const GraphicsPsoSource* graphics_source() const { return source_.get(); }
    // Why a placeholder was created instead of a pipeline (mesh shaders, ...); empty otherwise.
    const std::string& unsupported_reason() const { return unsupported_; }
    // Console-native graphics pipeline (Xbox slot 57): the range layout, a copy of the header
    // record (kept, not decoded) and the stage identities. Null for every other pipeline state.
    struct Native {
        xbox::NativePipeline layout;
        std::vector<uint8_t> header;
        uint64_t vertex_identity = 0;
        uint64_t pixel_identity = 0;  // 0 = no pixel stage
        uint64_t vertex_code_identity = 0;  // FNV-1a 64 over the code range only
        uint64_t pixel_code_identity = 0;
    };
    const Native* native() const { return native_.get(); }
    // Index of this pipeline in the pipeline dump, kNoDumpIndex when not dumped.
    static constexpr uint32_t kNoDumpIndex = 0xFFFFFFFFu;
    uint32_t dump_index() const { return dump_index_; }

private:
    friend class Device;
    com::Ref<RootSignature> root_signature_;
    std::unique_ptr<bridge::GraphicsPipelineState> graphics_;
    std::unique_ptr<bridge::ComputePipelineState> compute_;
    std::unique_ptr<GraphicsPsoSource> source_;
    std::unique_ptr<Native> native_;
    uint32_t dump_index_ = kNoDumpIndex;
    std::vector<uint64_t> identities_;
    std::string unsupported_;
    bool resolved_ = false;
};

class CommandList final : public ChildBase<IGraphicsCommandListHost> {
public:
    static constexpr const char* kComClassName = "ID3D12GraphicsCommandList";
    // Guest payload at list + 0x10, zero-filled (layout: d3d12_xs_xbox.h, "inlined command-list
    // accesses"): the title's inlined fast paths compare write cursors at list + 0x18 / 0x28 / 0x30,
    // which stay zero so every call takes the vtable path, and it stages pipeline states at
    // list + 0x544.
    static constexpr uint64_t kComPayloadSize = 0x580;
    CommandList(com::Ref<Device> device, uint32_t type);
    ~CommandList() override;

    // ---- IGraphicsCommandListHost ----
    uint32_t GetType() override { return type_; }
    HResult Close() override;
    HResult Reset(InterfaceArg<ICommandAllocatorHost> allocator, InterfaceArg<IPipelineStateHost> initial_state) override;
    HResult ResetX(InterfaceArg<ICommandAllocatorHost> allocator, InterfaceArg<IPipelineStateHost> initial_state,
                   uint64_t extra) override;
    void ClearState(InterfaceArg<IPipelineStateHost> state) override;
    void DrawInstanced(uint32_t vertices, uint32_t instances, uint32_t start_vertex, uint32_t start_instance) override;
    void DrawIndexedInstanced(uint32_t indices, uint32_t instances, uint32_t start_index, int32_t base_vertex,
                              uint32_t start_instance) override;
    void Dispatch(uint32_t x, uint32_t y, uint32_t z) override;
    void CopyBufferRegion(InterfaceArg<IResourceHost> dst, uint64_t dst_offset, InterfaceArg<IResourceHost> src,
                          uint64_t src_offset, uint64_t bytes) override;
    void CopyTextureRegion(com::CallContext& ctx) override;
    void CopyResource(InterfaceArg<IResourceHost> dst, InterfaceArg<IResourceHost> src) override;
    void ResolveSubresource(com::CallContext& ctx) override;
    void IASetPrimitiveTopology(uint32_t topology) override;
    void RSSetViewports(uint32_t count, uint64_t viewports) override;
    void RSSetScissorRects(uint32_t count, uint64_t rects) override;
    void OMSetBlendFactor(uint64_t factor) override;
    void OMSetStencilRef(uint32_t ref) override;
    void SetPipelineState(InterfaceArg<IPipelineStateHost> state) override;
    void ResourceBarrier(uint32_t count, uint64_t barriers) override;
    void SetDescriptorHeaps(uint32_t count, uint64_t heaps) override;
    void SetComputeRootSignature(InterfaceArg<IRootSignatureHost> signature) override;
    void SetGraphicsRootSignature(InterfaceArg<IRootSignatureHost> signature) override;
    void SetComputeRootDescriptorTable(uint32_t index, uint64_t base) override;
    void SetGraphicsRootDescriptorTable(uint32_t index, uint64_t base) override;
    void SetComputeRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) override;
    void SetGraphicsRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) override;
    void SetComputeRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values, uint32_t offset) override;
    void SetGraphicsRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values, uint32_t offset) override;
    void SetComputeRootConstantBufferView(uint32_t index, uint64_t va) override;
    void SetGraphicsRootConstantBufferView(uint32_t index, uint64_t va) override;
    void SetComputeRootShaderResourceView(uint32_t index, uint64_t va) override;
    void SetGraphicsRootShaderResourceView(uint32_t index, uint64_t va) override;
    void SetComputeRootUnorderedAccessView(uint32_t index, uint64_t va) override;
    void SetGraphicsRootUnorderedAccessView(uint32_t index, uint64_t va) override;
    void IASetIndexBuffer(uint64_t view) override;
    void IASetVertexBuffers(uint32_t start, uint32_t count, uint64_t views) override;
    void SOSetTargets(uint32_t start, uint32_t count, uint64_t views) override;
    void OMSetRenderTargets(uint32_t count, uint64_t rtvs, bool single_range, uint64_t dsv) override;
    void ClearDepthStencilView(uint64_t dsv, uint32_t flags, float depth, uint8_t stencil, uint32_t rect_count,
                               uint64_t rects) override;
    void ClearRenderTargetView(uint64_t rtv, uint64_t color, uint32_t rect_count, uint64_t rects) override;
    void ClearUnorderedAccessView(com::CallContext& ctx) override;
    void DiscardResource(InterfaceArg<IResourceHost>, uint64_t) override {}
    void QueryOperation(com::CallContext& ctx) override;
    void SetPredication(com::CallContext&) override {}
    void DebugMarker(com::CallContext&) override {}
    // Xbox slots (d3d12_xs_xbox.cpp).
    void SetPipelineStateX(com::CallContext& ctx) override;
    void ExecuteIndirect(com::CallContext& ctx) override;
    void OMSetDepthBounds(float min_depth, float max_depth) override;
    void RSSetShadingRate(uint32_t rate, uint64_t combiners) override;
    void RSSetShadingRateImage(InterfaceArg<IResourceHost> image) override;
    void WriteValue32(uint64_t address, uint32_t value, uint32_t a3, uint32_t a4) override;
    void WriteValue64(uint64_t address, uint32_t a2, uint64_t value, uint32_t a4) override;
    void Barrier(uint32_t group_count, uint64_t groups) override;
    void note_xbox_slot(uint32_t slot) override;
    // Immediate value writes recorded by Xbox slots 71 / 74, applied to guest memory in order when
    // the list is executed (work completes at submission in this runtime, as fences do).
    void apply_immediate_writes();
    size_t immediate_write_count() const { return immediate_writes_.size(); }

    // ---- Runtime side ----
    bool is_open() const { return open_; }
    // Appends this list's submission (deferred CPU-sourced copies, then its packets) to `out` and
    // commits its resource states. The list must be closed.
    void append_submission(CommandEncoder& out, std::vector<d3d::StateDiagnostic>* state_diagnostics);
    const bridge::CommandListBridge& bridge_state() const { return bridge_; }
    const CommandEncoder& encoder() const { return encoder_; }
    uint64_t diagnostics_count() const { return diag_count_; }

    // Reset with a host PSO (initial state at creation).
    void reset(PipelineState* initial_state);
    // Calls of Xbox-specific slots recorded without effect (slot -> count).
    std::vector<std::pair<uint32_t, uint64_t>> xbox_slot_calls() const;
    // Must run once right after create_object (zero-fills the guest payload).
    void initialize_guest_payload();

private:
    struct DeferredWrite {
        bool texture = false;
        uint32_t id = 0;
        uint64_t dst_offset = 0;      // buffers
        GpuWriteTexturePayload tex{};  // textures (data_len set)
        uint64_t src = 0;             // guest address of the source bytes
        uint64_t size = 0;
    };

    void note(const std::string& message);
    bool usable();
    Resource* resource(const InterfaceArg<IResourceHost>& arg) const;
    void bind_pipeline(PipelineState* state);

    uint32_t type_;
    bool open_ = false;
    CommandEncoder encoder_;
    bridge::CommandListBridge bridge_;
    PipelineState* graphics_pso_ = nullptr;  // current graphics PSO (kept alive by `pinned_`)
    PipelineState* compute_pso_ = nullptr;   // current compute PSO
    std::vector<com::Ref<com::ComObject>> pinned_;  // objects referenced by the recording
    std::vector<DeferredWrite> deferred_;
    // What is bound for graphics, kept for describing skipped draws (describe_skipped_draw).
    struct RootArgRecord {
        uint8_t kind = 0;  // 0 unset, 1 descriptor table, 2 constants, 3 CBV, 4 SRV, 5 UAV
        uint32_t count = 0;
        uint64_t value = 0;  // table: GPU handle; root descriptors: GPU VA
    };
    std::array<RootArgRecord, 64> graphics_root_args_{};
    struct VertexBufferRecord {
        uint64_t va = 0;
        uint32_t size = 0;
        uint32_t stride = 0;
    };
    std::array<VertexBufferRecord, 32> vertex_buffers_{};
    VertexBufferRecord index_buffer_{};  // stride = DXGI format
    std::vector<uint64_t> rtv_handles_;
    uint64_t dsv_handle_ = 0;
    void record_root_arg(uint32_t index, uint8_t kind, uint64_t value, uint32_t count);
    const d3d::RootSignature* graphics_root_signature_ = nullptr;
    bool rect_list_ = false;  // IASetPrimitiveTopology(xbox::kTopologyRectList)
    bool skip_rect_list_draw(const char* call, uint32_t count, uint32_t instances, const char* reason);
    // Rectangle-list expansion (d3d12_xs_xbox.h): each rectangle's three vertices become two
    // triangles (v0 v1 v2, v2 v3 v0) with v3 = v0 + v2 - v1 for every per-vertex attribute, written
    // into list-owned vertex buffers at ExecuteCommandLists. False (with `why`) when not possible.
    bool expand_rect_list_draw(uint32_t vertices, uint32_t instances, uint32_t start_vertex, uint32_t start_instance,
                               std::string* why);
    struct RectExpansion {
        uint32_t buffer_id = 0;
        uint64_t src = 0;  // guest address of the first source vertex
        uint32_t stride = 0;
        uint32_t rects = 0;
        std::vector<std::pair<uint32_t, uint32_t>> elements;  // (byte offset, DXGI format)
    };
    std::vector<RectExpansion> rect_expansions_;
    std::vector<com::Ref<Resource>> rect_buffers_;  // reused across recordings
    size_t rect_buffers_used_ = 0;
    std::array<d3d12_xs::D3D12VertexBufferView, 32> bridge_vb_views_{};  // as last passed to the bridge
    void note_draw_for_dump();
    void describe_skipped_draw(const char* call, uint32_t count, uint32_t instances, const char* reason);
    void describe_dropped_draw(const char* call, uint32_t count, uint32_t instances);
    struct ImmediateWrite {
        uint64_t address = 0;
        uint64_t value = 0;
        uint32_t size = 0;  // 4 or 8
    };
    std::vector<ImmediateWrite> immediate_writes_;
    uint64_t diag_count_ = 0;
    std::map<uint32_t, uint64_t> xbox_calls_;
};

extern const com::Interface<IResourceHost> kID3D12Resource;
extern const com::Interface<IHeapHost> kID3D12Heap;
extern const com::Interface<IPipelineStateHost> kID3D12PipelineState;
extern const com::Interface<IGraphicsCommandListHost> kID3D12GraphicsCommandList;

}  // namespace playgta6::gpu::d3d12_guest
