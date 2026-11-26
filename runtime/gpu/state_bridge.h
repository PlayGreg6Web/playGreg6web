#pragma once

// D3D12 command-list state -> GPU packet ABI v2 bridge.
//
// Drives the G7 PSO cache / output-merger tracker (pso_cache.h) and the G6 descriptor heap / root
// signature / bind group / resource state library (descriptor_heap.h) from one per-command-list API
// that the d3d12_xs shim layer calls, and emits the resulting packets (docs/gpu_commands.md v2)
// through CommandEncoder.
//
//   d3d12_xs shim --> CommandListBridge --+--> pso::CommandListStateTracker --StateEvents--+
//                                         +--> d3d12_state::RootArgumentState / BindGroupCache +--> packets
//                                         +--> d3d12_state::CommandListStateTracker (barriers)  |
//                     StateBridgeDevice (shared): PipelineCache, layouts, samplers, views, VA map
//
// Packet emission model
// ---------------------
// * Device-level objects (bind group layouts, pipeline layouts, pipelines, samplers, the root
//   constant ring buffer) are created once per device and their creation packet is written into the
//   stream of the command list that first needs them. Streams must therefore be executed in an order
//   where that list runs first (true for a single recording thread; with several recording threads the
//   submitter must preserve recording order, the same limitation G7's PipelineCache has).
// * Bind groups are per command list (G6 BindGroupCache) and are created, re-created and retired
//   (DestroyResource) in the list's own stream.
// * WebGPU resets pipeline, bind groups, vertex/index buffers and dynamic state at every pass begin;
//   the bridge re-emits whatever a draw needs after each BeginRenderPass.
//
// Ids
// ---
// All packet-level ids come from one allocator shared with the d3d12_xs device (BridgeOptions::
// id_allocator), so pipelines, layouts, samplers and bind groups never collide with buffers/textures.
// View ids handed to G7 are bridge-internal (they never appear in packets).
//
// Threading: StateBridgeDevice is thread-safe; CommandListBridge belongs to one recording thread.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "abi/playgta6_abi.h"
#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_xs.h"
#include "gpu/depth_twin.h"
#include "gpu/descriptor_heap.h"
#include "gpu/indirect_clamp_batch.h"
#include "gpu/pso_cache.h"

namespace playgta6::gpu {
struct PrewarmPipelineDesc;  // prewarm_pipeline_desc.h
}

namespace playgta6::gpu::bridge {

namespace pso = playgta6::gpu::pso;
namespace d3d = playgta6::gpu::d3d12_state;

// =====================================================================================================
// Seam conversions between the G5 (d3d12_xs / ABI v1), G6 (d3d12_state) and G7 (pso) type families.
// =====================================================================================================

// v1 GpuIndexFormat (SetIndexBuffer) <-> G7 / key index format code.
pso::WgpuIndexFormat to_wgpu_index_format(GpuIndexFormat format) noexcept;
// G6 DESCRIPTOR_RANGE_TYPE-style DXGI value (uint32) -> G7 DxgiFormat.
inline constexpr pso::DxgiFormat to_dxgi(uint32_t dxgi_format) noexcept {
    return static_cast<pso::DxgiFormat>(dxgi_format);
}
// G6 SRV/UAV/RTV view dimension (D3D12_*_DIMENSION) -> GpuTextureViewDimension (Undefined when the
// view has no WebGPU equivalent or is the default view).
GpuTextureViewDimension to_view_dimension(uint32_t d3d12_view_dimension, bool* supported) noexcept;
// G6 SamplerDesc (D3D12_SAMPLER_DESC numbering) -> CreateSampler2 payload (id left 0).
GpuCreateSampler2Payload to_sampler_payload(const d3d::SamplerDesc& desc, bool* exact) noexcept;
// The sampler bound to a layout entry of `layout_sampler_type` (GpuSamplerBindingType): a
// comparison sampler (D3D12 filter reduction 1) bound where the layout, and so the shader, samples
// without comparison drops the comparison (WebGPU rejects a comparison sampler there); everything
// else is returned unchanged. *changed reports a drop.
d3d::SamplerDesc sampler_for_binding(const d3d::SamplerDesc& desc, uint8_t layout_sampler_type, bool* changed) noexcept;

// =====================================================================================================
// Registration records
// =====================================================================================================

// Texture metadata the bridge needs to resolve views (registered by the d3d12_xs device when it
// creates a texture or swapchain buffer).
struct TextureInfo {
    pso::DxgiFormat format = pso::DxgiFormat::UNKNOWN;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth_or_array_size = 1;
    uint32_t mip_levels = 1;
    uint32_t sample_count = 1;
    GpuTextureDimension dimension = GpuTextureDimension::Dim2D;
    bool swapchain = false;  // render pass attachments name kGpuSwapchainTextureId instead of the id
    // Address the texture's descriptor words carry (descriptor_words.h): its memory address on the
    // console model, or a runtime-assigned range; 0 = none.
    uint64_t gpu_va = 0;
    // Guest D3D12_TEXTURE_LAYOUT / Xbox swizzle layout (0 = unknown/linear, 1 = row-major,
    // 0x100 = XG linear, >= 0x101 = tiled console swizzle mode).
    uint32_t layout = 0;
    // Created with pso::FormatUsage::SampledOnly (never a render target): its views map with it too.
    bool sampled_only = false;
    // Created with STORAGE_BINDING usage (storage views of it are valid).
    bool storage = false;
    // G66 partly-held mip chains: first mip level from which the chain holds content (the title
    // streams only the small levels; the larger ones are unmapped or belong to another texture).
    // Sampled views start no lower than it. 0 = the whole chain is held. Set with
    // StateBridgeDevice::set_resident_mip_floor, kept when the same description is registered again.
    uint32_t resident_mip_floor = 0;
    // Texture size cap (gpu/texture_budget.h): top mip levels the host texture lacks because it was
    // allocated at the capped size (its level 0 is guest level host_mip_offset). The fields above
    // stay in guest terms; views and writes are shifted by this. Set with set_host_mip_offset.
    uint32_t host_mip_offset = 0;
};

// Shifts a view's guest mip range onto a host texture that lacks the top `offset` levels: the view
// starts no lower than guest level `offset` (at least the view's last level stays) and its base and
// count become host terms. `levels` is the guest level count; count 0 = all remaining levels.
void shift_view_to_host_mips(uint32_t levels, uint32_t offset, uint16_t* base, uint16_t* count) noexcept;

// The sampled-view mip range clamped to the held levels: `base` / `count` (count 0 = all remaining
// levels) of a view of a texture with `mip_levels` levels, raised so the view starts no lower than
// max(resident_floor, floor(min_lod_clamp)) (D3D12 ResourceMinLODClamp counts from the resource's
// level 0). At least the view's (or the chain's) last level stays. Returns true when it raised base.
bool clamp_view_to_resident_mips(uint32_t mip_levels, uint32_t resident_floor, float min_lod_clamp, uint16_t* base,
                                 uint16_t* count) noexcept;
// The first held level of a mip chain from per-subresource upload marks (index mip + slice * mips;
// non-zero = that subresource's guest memory was mapped and uploaded): the lowest level from which
// every smaller level is held in some slice. 0 when the smallest level is not held (nothing
// streamed, or levels filled on the GPU): such a chain is not clamped.
// Only a streamable (reserved/tiled) resource is clamped: for any other, an all-zero level is a
// black level the title uploaded, not a level that is not loaded, so the floor is 0.
uint32_t held_mip_floor(std::span<const uint64_t> subresource_marks, uint32_t mips, uint32_t layers,
                        bool streamable = true) noexcept;

// One binding of a bind group layout: the WebGPU layout entry plus the D3D register join key (S6
// layout-manifest.json `bindings` record). `entry.binding` and `join.binding` must match.
struct LayoutBinding {
    GpuBindGroupLayoutEntry entry{};
    d3d::ShaderBinding join{};
};

struct GroupLayoutDesc {
    uint64_t content_id = 0;  // S6 bind group layout id (0 = not from the manifest)
    std::vector<LayoutBinding> bindings;
};

// The pipeline layout of one shader combination (S6 `pipelines` / `shaders` entry).
struct PipelineLayoutDesc {
    uint64_t content_id = 0;  // S6 pipeline layout id (0 = not from the manifest)
    std::vector<GroupLayoutDesc> groups;  // group 0..n-1
};

// =====================================================================================================
// Diagnostics
// =====================================================================================================

namespace BridgeDiag {
inline constexpr uint64_t UnknownShaderModule = 1ull << 0;      // shader identity has no module id
inline constexpr uint64_t MissingPipelineLayout = 1ull << 1;    // no layout registered: "auto"
inline constexpr uint64_t RootArgumentNotSet = 1ull << 2;       // bind group skipped (draw skipped)
inline constexpr uint64_t BindGroupBuildFailed = 1ull << 3;     // other G6 lookup failure
inline constexpr uint64_t NullBinding = 1ull << 4;              // null/unresolved descriptor bound
inline constexpr uint64_t ClearFallbackDropped = 1ull << 5;     // partial-rect clear: no packet yet
inline constexpr uint64_t UsageConflict = 1ull << 6;            // WebGPU usage scope violation
inline constexpr uint64_t UnknownTexture = 1ull << 7;           // view of an unregistered texture
inline constexpr uint64_t UnsupportedViewDimension = 1ull << 8; // 1D array / MS array / 3D attachment
inline constexpr uint64_t SamplerApproximated = 1ull << 9;      // border / mirror-once / min-max filter
inline constexpr uint64_t DrawSkipped = 1ull << 10;             // see pso tracker diagnostics too
inline constexpr uint64_t RootConstantRingWrapped = 1ull << 11; // ring reused: size it per frame
inline constexpr uint64_t NoComputePipeline = 1ull << 12;       // Dispatch without a usable pipeline
inline constexpr uint64_t CommandListClosed = 1ull << 13;       // call on a closed list ignored
inline constexpr uint64_t ComponentMappingIgnored = 1ull << 14; // non-identity SRV swizzle
inline constexpr uint64_t UnalignedUniform = 1ull << 15;        // uniform binding offset % 256, not relocated
inline constexpr uint64_t UnalignedStorage = 1ull << 16;        // storage binding offset % 256, not relocated
inline constexpr uint64_t NullStorageDispatchDropped = 1ull << 18;  // a dispatch with a null writable storage binding was dropped
inline constexpr uint64_t StorageViewNoUsage = 1ull << 17;      // storage view of a texture without storage usage
                                                                // or of another format than the stage's: null
}  // namespace BridgeDiag

// =====================================================================================================
// Device-level state
// =====================================================================================================

struct BridgeOptions {
    // Shared resource id allocator (must be thread-safe, never return 0). Default: counter from 1.
    std::function<uint32_t()> id_allocator;
    // Drop dispatches that have a null writable storage binding (counted, never faked as done).
    bool drop_null_storage_dispatch = true;
    uint32_t features = 0;  // pso::WgpuFeature bits enabled on the WebGPU device
    pso::MidPassClearPolicy mid_pass_clear = pso::MidPassClearPolicy::SplitPass;
    uint32_t max_texture_dimension_2d = 8192;
    // Uniform ring for root constants (bound with dynamic offsets). Must hold one frame's worth.
    uint32_t root_constant_ring_bytes = 1u << 20;
    // Uniform-buffer bindings whose offset is not a multiple of kUniformOffsetAlignment (WebGPU's
    // minUniformBufferOffsetAlignment), or that are smaller than the stage's uniform block with no
    // room behind them in their buffer, are relocated through this hook: given the binding's GPU
    // address, its size and the size the binding must have, it returns a buffer id and an aligned
    // offset holding the same `size` bytes followed by zeros up to `binding_size` rounded to 256
    // (false: no relocation, the binding is emitted as it is and diagnosed).
    std::function<bool(uint64_t gpu_va, uint64_t size, uint64_t binding_size, uint32_t* buffer_id, uint64_t* offset)>
        relocate_uniform;
    // O36: the same for read-only storage bindings (WebGPU's minStorageBufferOffsetAlignment is 256
    // as well; Xbox structured / raw buffer views start at any element): returns a buffer id and a
    // 256-aligned offset holding the `size` bytes at `gpu_va`, refreshed when the guest writes them
    // (false: emitted as it is and diagnosed). The translated stage indexes the binding from its
    // first byte, so the view's bytes must start the binding: an aligned-down binding would need an
    // offset the stage does not take.
    std::function<bool(uint64_t gpu_va, uint64_t size, uint32_t* buffer_id, uint64_t* offset)> relocate_storage;
    // G66 (#691): bumped whenever the two hooks above hand an existing home to another range (LRU
    // eviction). A cached bind group bound to a relocated home re-resolves its relocations when it
    // changed and is re-created if one moved; otherwise it would read another draw's bytes.
    std::function<uint64_t()> relocation_epoch;
    // #663: true for a buffer holding storage relocation homes. A slot source decoding descriptor
    // memory binds these homes directly (no recorded relocation), so a memory-backed bind group
    // that binds one is rebuilt from its slots when relocation_epoch changed.
    std::function<bool(uint32_t buffer_id)> relocation_home;
    // Root tables in heaps this source reads are built from descriptor memory (G48): at recording,
    // and again at ExecuteCommandLists (CommandListBridge::refresh_memory_bind_groups). Not owned.
    const d3d::DescriptorSlotSource* slot_source = nullptr;
    // Guest texture-memory sync hook (G49b): invoked when a texture SRV is bound in a root
    // descriptor table at draw/dispatch or submission refresh time. Inspects guest memory backing
    // `resource_id` and emits WriteTexture into `encoder` for each subresource whose texels
    // changed; returns true when a WriteTexture packet was emitted.
    std::function<bool(uint32_t resource_id, CommandEncoder& encoder)> sync_texture_memory;
    // Device capability (G62, #482): the consumer binds binding arrays (GpuLayoutFlags::BindingArrays /
    // GpuBindGroupFlags::BindingArrays). Off by default: lane-indexed heap windows are then refused at
    // the reflection join (counted) and no layout with binding_array_size != 0 is emitted.
    bool binding_arrays = false;
};

inline constexpr uint64_t kUniformOffsetAlignment = 256;
// WebGPU's default maxUniformBufferBindingSize (also D3D12's 4096 x 16-byte constant buffer limit).
inline constexpr uint64_t kMaxUniformBindingSize = 65536;
// Bind groups one command list keeps cached across Resets (O36); see CommandListBridge::Reset.
inline constexpr size_t kMaxCachedBindGroups = 4096;

// O36 / backlog 28: process-wide gauges of live bridge-side objects (every StateBridgeDevice and
// command list), for the boot report and the plateau check of a long run.
struct BridgeLiveCounts {
    uint64_t interned_views = 0;
    uint64_t samplers = 0;
    uint64_t cached_bind_groups = 0;  // sum over command lists (also d3d12_state::descriptor_live_counts)
    uint64_t command_lists = 0;       // live CommandListBridge objects
};
BridgeLiveCounts bridge_live_counts() noexcept;

class GraphicsPipelineState;
class ComputePipelineState;

// G66: ExecuteIndirect with a single-argument command signature. The guest's argument records are
// first passed through a clamp compute pass (indirect_clamp_wgsl) that applies the count buffer,
// clamps out-of-range values and writes them compacted into a bridge-owned ring, from which one
// DrawIndirect / DrawIndexedIndirect / DispatchIndirect per command reads.
enum class IndirectKind : uint8_t {
    Draw = 0,         // 16-byte record: vertex count, instance count, first vertex, first instance
    DrawIndexed = 1,  // 20-byte record: index count, instance count, first index, base vertex, first instance
    Dispatch = 2,     // 12-byte record: thread group counts x, y, z
    // G66 mesh pipelines: 12-byte DispatchMesh record (group counts x, y, z) for the bound mesh
    // pipeline state. The clamp pass writes a 28-byte pair: a dispatch record (x, 1, 1) for the
    // mesh stage run as compute, then a draw record (x * max_primitives * 3, 1, 0, 0).
    DispatchMesh = 3,
};
struct IndirectExecution {
    IndirectKind kind = IndirectKind::Draw;
    uint32_t max_count = 0;
    uint32_t stride = 0;       // command signature ByteStride
    uint32_t args_buffer = 0;  // buffer resource id
    uint64_t args_offset = 0;
    uint32_t count_buffer = 0;  // 0 = no count buffer (max_count commands)
    uint64_t count_offset = 0;
};
// Commands one ExecuteIndirect emits at most (more are dropped and counted).
inline constexpr uint32_t kMaxIndirectCommands = 256;
// WebGPU's default maxComputeWorkgroupsPerDimension: dispatch arguments are clamped to it.
inline constexpr uint32_t kMaxIndirectWorkgroups = 65535;
// The clamp pass's output ring (one window of kMaxIndirectCommands records per execution).
inline constexpr uint32_t kIndirectRingBytes = 1u << 18;
inline constexpr uint32_t kIndirectWindowBytes = kMaxIndirectCommands * 28u;
// Encoded size of one clamp pass (params write, pipeline, bind group, dispatch): a batch slot.
inline constexpr uint32_t kIndirectClampSlotBytes = 120;
// Bytes the clamp pass writes per command of `kind` (the stride of the indirect packets' records).
uint32_t indirect_output_bytes(IndirectKind kind) noexcept;
// The clamp compute shader (entry point cs_main, workgroup size 64).
const char* indirect_clamp_wgsl() noexcept;
class CommandListBridge;

// ---- G66: mesh pipelines (triangle output, x-dimension groups) emulated with compute + draw ----
// A mesh pipeline state is two pipelines:
//   * the mesh stage translated to a WGSL compute module (pack stage "compute", key
//     mesh_compute_pack_key(mesh stage key), entry point cs_main, one workgroup per mesh group,
//     workgroup_id.x = the group index). Its pipeline layout is the stage's own (groups 0..2, bound
//     from the graphics root arguments) plus group kMeshOutputGroup, binding 0: the runtime's
//     output buffer (storage, read_write), one record per group (mesh_group_record_words):
//       word 0: vertex count, word 1: primitive count (SetMeshOutputCounts), words 2..3: 0,
//       then max_vertices vertices of vertex_words words (SV_Position xyzw as f32 bits, then the
//       attributes as vec4<f32> per location 0..), then max_primitives x 3 u32 local indices;
//   * a draw with a runtime vertex shader (mesh_draw_vertex_wgsl) that pulls those triangles from
//     the same buffer (group kMeshOutputGroup, read-only storage) and the title's pixel shader:
//     groups * max_primitives * 3 vertices, unused primitives degenerate.
struct MeshOutputLayout {
    uint32_t max_vertices = 0;    // 1..256
    uint32_t max_primitives = 0;  // 1..256
    uint32_t vertex_words = 0;    // 4 (position) + 4 per attribute location (at most kMaxMeshAttributes)
};
inline constexpr uint32_t kMaxMeshAttributes = 15;
inline constexpr uint32_t kMeshOutputGroup = 3;
// The runtime's mesh output buffer: every mesh dispatch writes it from offset 0 (GPU-ordered).
inline constexpr uint64_t kMeshOutputBytes = 32ull << 20;
bool valid_mesh_output_layout(const MeshOutputLayout& layout) noexcept;
uint32_t mesh_group_record_words(const MeshOutputLayout& layout) noexcept;
// Groups one mesh dispatch runs at most: what fits in the output buffer, at most kMaxIndirectWorkgroups.
uint32_t mesh_max_groups(const MeshOutputLayout& layout) noexcept;
// The pack key of a mesh stage's compute translation.
inline constexpr uint64_t kMeshComputeKeySalt = 0x4d53435300000000ull;  // "MSCS"
constexpr uint64_t mesh_compute_pack_key(uint64_t mesh_stage_key) noexcept {
    return mesh_stage_key ^ kMeshComputeKeySalt;
}
// The runtime vertex shader of a mesh draw (entry point vs_main) and the shader identity it is
// registered under (one per output layout).
std::string mesh_draw_vertex_wgsl(const MeshOutputLayout& layout);
uint64_t mesh_draw_vertex_identity(const MeshOutputLayout& layout) noexcept;
class MeshPipelineState;

struct ViewRecord {
    uint32_t texture_id = 0;  // packet texture id (kGpuSwapchainTextureId for swapchain buffers)
    uint32_t resource_id = 0; // runtime resource id (usage tracking)
    GpuTextureViewDesc view{};
    d3d::SubresourceRange range{};
    d3d::ResourceUsage usage = d3d::ResourceUsage::RenderTarget;
};

// Outcome of building one warm-list pipeline ahead of the title (StateBridgeDevice::prewarm_pipeline).
enum class PrewarmPipelineResult : uint8_t {
    Built,   // created and emitted ahead of the title
    Lost,    // the title created (or is creating) it, or owns a layout packet still in its own list
    Failed,  // the description could not be replayed
};

class StateBridgeDevice final : public pso::DescriptorViewResolver {
public:
    // `resource_states` (optional) is the queue-level registry the per-list barrier trackers report to.
    explicit StateBridgeDevice(d3d::DescriptorHeapRegistry& heaps,
                               BridgeOptions options = {},
                               const d3d::ResourceStateRegistry* resource_states = nullptr);
    ~StateBridgeDevice() override;
    StateBridgeDevice(const StateBridgeDevice&) = delete;
    StateBridgeDevice& operator=(const StateBridgeDevice&) = delete;

    // ---- Registration (called by the d3d12_xs device) ----
    void register_texture(uint32_t resource_id, const TextureInfo& info);
    // Buffer placed at [gpu_va, gpu_va + byte_size) for CBV / root descriptor address resolution.
    // `allocated_size`: the WebGPU buffer's size when it is larger than the guest's `byte_size`
    // (zero padding a binding may extend into, see buffer_allocated_size); 0 = byte_size.
    void register_buffer(uint32_t resource_id, uint64_t gpu_va, uint64_t byte_size, uint64_t allocated_size = 0);
    // Bytes the WebGPU buffer has (>= the guest size; 0 for an unknown buffer).
    uint64_t buffer_allocated_size(uint32_t resource_id) const;
    void unregister_resource(uint32_t resource_id);
    // Shader identity (D3D12GraphicsPsoDesc::vs/ps/cs) -> CreateShaderModule id.
    void register_shader_module(uint64_t shader_identity, uint32_t module_id);
    // Layout for a shader combination: (vs, ps) for graphics, (cs, 0) for compute.
    void register_pipeline_layout(uint64_t first_shader, uint64_t second_shader, PipelineLayoutDesc layout);

    // ---- Pipeline state objects (ID3D12Device::Create*PipelineState) ----
    std::unique_ptr<GraphicsPipelineState> create_graphics_pipeline_state(
        const pso::D3D12GraphicsPsoDesc& desc,
        const d3d::RootSignature* root_signature);
    std::unique_ptr<ComputePipelineState> create_compute_pipeline_state(
        const pso::D3D12ComputePsoDesc& desc,
        const d3d::RootSignature* root_signature);
    // G66 mesh pipeline: `draw_desc` is the pipeline's graphics state with vs =
    // mesh_draw_vertex_identity(output) (its module and its (vs, ps) layout registered), no input
    // layout and triangle topology; `mesh_shader` is the identity the mesh stage's compute module
    // and (mesh_shader, 0) layout are registered under. Diagnostics as for the other states.
    std::unique_ptr<MeshPipelineState> create_mesh_pipeline_state(const pso::D3D12GraphicsPsoDesc& draw_desc,
                                                                  uint64_t mesh_shader, const MeshOutputLayout& output,
                                                                  const d3d::RootSignature* root_signature);
    // Final Release of a PSO: drops its pipeline-cache references and emits DestroyResource for the
    // pipelines the cache evicted (only with EvictUnreferenced caches).
    void release_pipeline_state(std::unique_ptr<GraphicsPipelineState> pso, CommandEncoder& encoder);
    void release_mesh_pipeline_state(std::unique_ptr<MeshPipelineState> pso, CommandEncoder& encoder);

    // ---- pso::DescriptorViewResolver over the G6 heaps ----
    uint64_t rtv_descriptor_increment() const override;
    bool resolve_rtv(uint64_t cpu_handle, pso::RenderTargetViewInfo* out) const override;
    bool resolve_dsv(uint64_t cpu_handle, pso::DepthStencilViewInfo* out) const override;
    // Texture id behind an RTV/DSV descriptor (0 if null or unknown).
    uint32_t attachment_texture(uint64_t cpu_handle, d3d::DescriptorKind kind) const {
        d3d::ViewDescriptor v{};
        return cpu_handle != 0 && resolve_view_descriptor(cpu_handle, kind, &v) ? v.resource_id : 0u;
    }

    // Resolves a CBV / root descriptor GPU virtual address (G6 GpuVaResolver contract).
    bool resolve_gpu_va(uint64_t gpu_address, d3d::GpuVaResolution* out) const;
    static bool gpu_va_trampoline(void* user, uint64_t gpu_address, d3d::GpuVaResolution* out);

    const ViewRecord* view(uint32_t view_id) const;
    pso::PipelineCache& pipeline_cache() noexcept { return *pipeline_cache_; }
    d3d::DescriptorHeapRegistry& heaps() noexcept { return heaps_; }
    const BridgeOptions& options() const noexcept { return options_; }
    // GPU address of a registered buffer (0 when unknown).
    uint64_t buffer_gpu_va(uint32_t resource_id) const;
    uint32_t allocate_id() { return options_.id_allocator(); }
    // The registered description of texture `resource_id` (false if it is not a texture).
    bool texture_info(uint32_t resource_id, TextureInfo* out) const;
    // The texture whose descriptor address (TextureInfo::gpu_va) is exactly `gpu_va` (0 if none).
    uint32_t texture_at(uint64_t gpu_va) const;
    // G66 partly-held mip chains: records the first held mip level of texture `resource_id` (from the
    // title's uploads / mappings: the levels whose memory holds content). Sampled views bound after
    // this start there; cached bind groups over the texture are re-created when it changes.
    // Returns false for an unknown texture.
    bool set_resident_mip_floor(uint32_t resource_id, uint32_t first_resident_mip);
    // Records that the host texture of `resource_id` was re-created without its top `offset` levels
    // (0 = full size). Views and cached bind groups over it are retired so they are made again
    // against the new allocation. Returns false for an unknown texture.
    bool set_host_mip_offset(uint32_t resource_id, uint32_t offset);
    // Maps guest mip `*mip` of the texture onto its host level. False when the host texture lacks
    // that level (a write to it is skipped); unknown textures map unchanged.
    bool translate_host_mip(uint32_t resource_id, uint32_t* mip) const;
    // Bumped whenever a texture's resident mip floor changes.
    uint64_t residency_epoch() const noexcept { return residency_epoch_.load(std::memory_order_acquire); }
    // G66 (#691): ids of textures / buffers unregistered since they were registered (destroyed by the
    // device: guest Release, address-texture LRU, relocation teardown). A bind group built while
    // such a resource lived still names it: a cached hit must not bind it again, as WebGPU then
    // rejects the whole submit ("Destroyed texture used in a submit"). The epoch bumps on every one.
    bool retired(uint32_t resource_id) const;
    uint64_t retire_epoch() const noexcept { return retire_epoch_.load(std::memory_order_acquire); }

    // ---- Native pipeline prewarm (pipeline_prewarm.h) ----
    // Pack key of a pack shader module id (0 = not a pack module); set by the Device. Needed to describe a
    // pipeline the title creates so the next run can build it ahead of time.
    void set_module_pack_key_resolver(std::function<uint64_t(uint32_t module_id)> resolver);
    // Turn the title-side hand-off on: with a warm list that carries full pipelines, title paths that
    // might meet a prewarmed object wait for an in-progress build first (see prewarm_pipeline()).
    void set_prewarm_sync(bool on) noexcept { prewarm_sync_.store(on, std::memory_order_release); }
    // Builds the host pipeline `desc` describes: interns and emits its layout, takes the pipeline from the
    // pipeline cache (the very id the title's own request for the same state gets) and emits its creation
    // packet through `submit` (called with the packets while the hand-off lock is held, so a concurrent
    // title request that sees the pipeline also sees its packets submitted). One build per pipeline.
    // Safe from any thread.
    PrewarmPipelineResult prewarm_pipeline(const PrewarmPipelineDesc& desc, uint32_t vs_module_id,
                                           uint32_t fs_module_id, uint32_t task_id,
                                           const std::function<void(CommandEncoder&)>& submit);

    // ---- Frame-safe resource lifetime & deferred DestroyResource (G70, #691) ----
    // Records that GPU resource `resource_id` (texture, buffer, intermediate, or bind group) is
    // referenced by commands in the current frame (since the last Present) and updates its LRU tick.
    void note_resource_used(uint32_t resource_id);
    bool is_resource_used_in_frame(uint32_t resource_id) const;
    uint64_t resource_last_use_tick(uint32_t resource_id) const;
    // Emits DestroyResource(resource_id) into `encoder`, or defers it until after Present if
    // `resource_id` was referenced in the current frame. Returns true if deferred.
    bool destroy_or_defer_resource(uint32_t resource_id, CommandEncoder& encoder);
    // Unconditionally queues `resource_id` for DestroyResource after Present (deduplicated).
    void defer_destroy_resource(uint32_t resource_id);
    // Removes `resource_id` from the deferred queue (if pending) and emits DestroyResource into `encoder`.
    // Safe only for bind group ids (GPUBindGroup has no WebGPU destroy() that invalidates a submit).
    bool take_deferred_destroy(uint32_t resource_id, CommandEncoder& encoder);
    // Emits DestroyResource for every deferred resource and starts a new frame usage window.
    // Call strictly AFTER Present (or at the start of the next frame after Present).
    size_t flush_deferred_destroys(CommandEncoder& encoder);
    // Emits Present into `encoder`, then immediately flushes all deferred DestroyResource packets
    // after Present so no resource used by the frame is destroyed before queue.submit().
    void present(CommandEncoder& encoder);
    // Syncs with a host Present counter (e.g. Device::stats().presents). When `present_count`
    // has advanced past the last seen count, flushes deferred destroys into `encoder` and returns true.
    bool advance_present_frame(uint64_t present_count, CommandEncoder& encoder);
    size_t deferred_destroy_count() const;

private:
    friend class CommandListBridge;
    friend class GraphicsPipelineState;
    friend class ComputePipelineState;

    struct GroupLayoutRecord {
        uint32_t id = 0;
        GpuCreateBindGroupLayoutPayload header{};
        // Per bind group entry: a binding array of N elements (binding_array_size N) is N copies of its
        // layout entry, joins[i].array_index 0..N-1 (one GpuBindGroupEntry2 each).
        std::vector<GpuBindGroupLayoutEntry> entries;
        std::vector<d3d::ShaderBinding> joins;  // same order as entries
        // The emitted GpuBindGroupLayoutEntry list: one per binding (arrays not expanded).
        std::vector<GpuBindGroupLayoutEntry> layout_entries;
        bool emitted = false;
        bool emitted_by_prewarm = false;  // its packets went out with a prewarm submit, not in a title list
        // G66 mesh pipelines: a group the caller binds itself (the mesh output buffer).
        bool runtime_owned = false;
    };
    struct PipelineLayoutRecord {
        uint32_t id = 0;
        uint64_t content_id = 0;
        std::vector<GroupLayoutRecord*> groups;
        bool emitted = false;
        bool emitted_by_prewarm = false;
    };

    uint32_t intern_view(const ViewRecord& record) const;
    std::vector<uint32_t> retire_views(uint32_t texture_id);
    void attach_bridge(CommandListBridge* bridge);
    void detach_bridge(CommandListBridge* bridge);
    void notify_bridges_invalidated(std::span<const uint32_t> resource_ids, std::span<const uint32_t> view_ids);
    size_t flush_deferred_destroys_impl(CommandEncoder& encoder, bool respect_unexecuted);

public:
    // O36 (tests): live interned views and sampler count.
    size_t live_view_count() const;
    uint32_t test_intern_view(const ViewRecord& record) const { return intern_view(record); }

private:
    bool resolve_view_descriptor(uint64_t cpu_handle, d3d::DescriptorKind kind, d3d::ViewDescriptor* out) const;
    PipelineLayoutRecord* find_layout(uint64_t first_shader, uint64_t second_shader);
    uint32_t module_id(uint64_t shader_identity, bool* known) const;
    // Emits the layout's creation packets the first time any list needs them.
    void ensure_layout_emitted(PipelineLayoutRecord* layout, CommandEncoder& encoder);
    void ensure_layout_emitted_impl(PipelineLayoutRecord* layout, CommandEncoder& encoder, bool by_prewarm);
    // Interns `desc` (sorts its bindings) into group / pipeline layout records; objects_mutex_ held.
    // Null when the layout is refused.
    PipelineLayoutRecord* intern_pipeline_layout_locked(PipelineLayoutDesc& desc);
    // Title-side hand-off with the prewarm thread: waits for a build in progress (no-op without a
    // prewarm warm list).
    void prewarm_barrier();
    // The title reached pipeline `pipeline_id` without creating it: when the prewarm built it, counts the
    // hit and enters it into the observed list.
    void note_pipeline_reached(uint32_t pipeline_id);
    // The title created a pipeline: describes it into the warm list.
    void record_graphics_pipeline(const pso::GraphicsPipelineKey& key, uint32_t vs_module, uint32_t fs_module,
                                  const PipelineLayoutRecord* layout);
    void record_compute_pipeline(const pso::ComputePipelineKey& key, uint32_t cs_module,
                                 const PipelineLayoutRecord* layout);
    bool describe_layout(const PipelineLayoutRecord* layout, PrewarmPipelineDesc* desc);
    // Returns the sampler id for `desc`, emitting CreateSampler2 on first use.
    uint32_t intern_sampler(const d3d::SamplerDesc& desc, CommandEncoder& encoder, uint64_t* diagnostics);
    // Root constant ring buffer id; emits its CreateBuffer on first use.
    uint32_t ensure_root_constant_ring(CommandEncoder& encoder);
    // Shared (0,0,0,1) vertex buffer for attributes no input element feeds (G62); created on first use.
    uint32_t ensure_default_vertex_buffer(CommandEncoder& encoder);
    // Reserves `bytes` (256-byte aligned start) in the ring and returns the offset.
    uint32_t allocate_root_constants(uint32_t bytes, bool* wrapped);
    // True exactly once per compute pipeline id: the caller emits CreateComputePipeline2.
    bool claim_compute_creation(uint32_t pipeline_id);

    // G66 ExecuteIndirect clamp pass: pipeline, group layout and output ring (created through
    // `encoder` on first use), its bind group over (args, count) buffers, and a ring window.
    struct IndirectClamp {
        uint32_t group_layout = 0;
        uint32_t pipeline = 0;
        uint32_t ring = 0;
    };
    IndirectClamp indirect_clamp(CommandEncoder& encoder);
    uint32_t indirect_clamp_group(const IndirectClamp& clamp, uint32_t params_ring, uint32_t args, uint32_t count,
                                  CommandEncoder& encoder);
    // True when the (args, count) clamp bind group already exists (a batched clamp may use it).
    bool indirect_clamp_group_known(uint32_t args, uint32_t count);
    uint32_t allocate_indirect_window();

    // G66 mesh pipelines: `base` (at most kMeshOutputGroup groups) padded with empty groups and
    // followed by the runtime-owned output group (binding 0, `visibility`, `binding_type`); null
    // when `base` has more groups.
    PipelineLayoutRecord* mesh_layout(PipelineLayoutRecord* base, uint32_t visibility, uint8_t binding_type);
    // The output buffer's bind group for runtime group layout `group` (created through `encoder`
    // on first use).
    uint32_t mesh_output_group(const GroupLayoutRecord& group, CommandEncoder& encoder);

    // #482 storage-format fallback: the intermediate texture of `resource_id` in `format` (same size,
    // dimension and mips), created through `encoder` on first use.
    uint32_t intermediate_storage_texture(uint32_t resource_id, const TextureInfo& info, uint8_t format,
                                          CommandEncoder& encoder);
    // The conversion pipeline writing `dst_format` views of `dimension` (GpuTextureViewDimension), and
    // its bind group for (src, dst, subresource); created through `encoder` on first use.
    struct StorageConversionPipeline {
        uint32_t group_layout = 0;
        uint32_t pipeline = 0;
    };
    StorageConversionPipeline storage_conversion_pipeline(uint8_t dst_format, uint8_t dimension,
                                                          CommandEncoder& encoder);
    uint32_t storage_conversion_group(const StorageConversionPipeline& pipeline, uint8_t dimension, uint32_t src,
                                      uint8_t src_format, uint32_t dst, uint8_t dst_format, uint32_t mip,
                                      uint32_t base_layer, uint32_t layer_count, CommandEncoder& encoder);
    // Drops the intermediates and conversion bind groups of `resource_id` (queued for frame-safe destroy)
    // and returns the retired intermediate texture ids so cached bind groups can be invalidated too.
    std::vector<uint32_t> retire_intermediates(uint32_t resource_id);

    // Backlog 35: depth <-> colour twin copies. `twin_format` is the storage format (R32Float or
    // R32Uint), `depth_format` the depth texture's WebGPU format. The pipelines (a compute pass
    // sampling the depth aspect into the twin; a fullscreen draw writing frag_depth from it) and
    // their bind groups are created through `encoder` on first use.
    struct DepthTwinPipelines {
        uint32_t in_layout = 0;
        uint32_t in_pipeline = 0;
        uint32_t out_layout = 0;
        uint32_t out_pipeline = 0;
    };
    DepthTwinPipelines depth_twin_pipelines(uint8_t depth_format, uint8_t twin_format, CommandEncoder& encoder);
    // Bind group of the compute copy (depth texture sampled, twin stored) or of the draw (twin read).
    uint32_t depth_twin_group(uint32_t layout, uint32_t depth_texture, uint32_t twin, bool into_twin,
                              uint8_t twin_format, uint32_t mip, uint32_t layer, CommandEncoder& encoder);
    void flush_retired_locked(CommandEncoder& encoder);

    d3d::DescriptorHeapRegistry& heaps_;
    BridgeOptions options_;
    const d3d::ResourceStateRegistry* resource_states_ = nullptr;
    std::unique_ptr<pso::PipelineCache> pipeline_cache_;

    mutable std::shared_mutex resources_mutex_;
    std::unordered_map<uint32_t, TextureInfo> textures_;
    std::map<uint64_t, std::pair<uint32_t, uint64_t>> buffers_by_va_;  // va -> (id, size)
    std::unordered_map<uint32_t, uint64_t> buffer_va_;                  // id -> va
    std::unordered_map<uint32_t, uint64_t> buffer_allocated_;           // id -> WebGPU buffer size
    std::unordered_map<uint64_t, uint32_t> textures_by_va_;             // descriptor address -> id
    std::unordered_set<uint32_t> retired_resources_;  // see retired(); an id registered again leaves it
    std::atomic<uint64_t> retire_epoch_{0};
    std::atomic<uint64_t> residency_epoch_{0};
    std::unordered_map<uint64_t, uint32_t> shader_modules_;

    mutable std::mutex views_mutex_;
    mutable std::deque<ViewRecord> views_;  // index = view_id - 1 (stable addresses)
    mutable std::unordered_map<std::string, uint32_t> view_ids_;
    // O36: views die with their texture: the id is recycled (free list) and the interned key dropped.
    mutable std::deque<std::string> view_keys_;                               // index = view_id - 1; empty = free
    mutable std::unordered_map<uint32_t, std::vector<uint32_t>> views_by_texture_;
    mutable std::vector<uint32_t> free_views_;
    mutable size_t views_tracked_ = 0;     // under views_mutex_: this device's share of the live-view gauge
    size_t samplers_tracked_ = 0;          // under objects_mutex_: its share of the sampler gauge

    std::mutex objects_mutex_;  // layouts, samplers, root constant ring
    std::vector<std::unique_ptr<GroupLayoutRecord>> group_layouts_;
    std::vector<std::unique_ptr<PipelineLayoutRecord>> pipeline_layouts_;
    std::map<std::pair<uint64_t, uint64_t>, PipelineLayoutRecord*> layouts_by_shaders_;
    std::unordered_map<std::string, uint32_t> samplers_;
    std::unordered_set<uint32_t> emitted_compute_;
    // Pipeline prewarm hand-off. Lock order: prewarm_mutex_, then objects_mutex_.
    std::function<uint64_t(uint32_t)> module_pack_key_;
    std::atomic<bool> prewarm_sync_{false};
    std::mutex prewarm_mutex_;
    std::unordered_map<uint32_t, uint32_t> prewarm_pipelines_;  // pipeline id -> prewarm task id, until the title reaches it
    uint32_t ring_buffer_id_ = 0;
    uint32_t default_vertex_buffer_id_ = 0;
    uint32_t ring_cursor_ = 0;
    std::map<std::pair<uint32_t, uint8_t>, uint32_t> intermediates_;     // (texture, format) -> texture
    std::map<uint32_t, DepthTwinPipelines> depth_twin_pipelines_;  // depth format << 8 | twin format
    std::map<uint32_t, StorageConversionPipeline> conversion_pipelines_;  // dst format << 8 | dimension
    std::map<std::array<uint32_t, 6>, uint32_t> conversion_groups_;      // (layout, src, dst, mip, layers)
    std::vector<uint32_t> retired_ids_;                                   // DestroyResource at next safe point
    IndirectClamp indirect_clamp_{};
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> indirect_groups_;  // (args, count) -> bind group
    uint32_t indirect_cursor_ = 0;
    std::map<std::pair<PipelineLayoutRecord*, uint32_t>, PipelineLayoutRecord*> mesh_layouts_;
    GroupLayoutRecord* empty_group_layout_ = nullptr;
    uint32_t mesh_output_buffer_ = 0;
    std::map<uint32_t, uint32_t> mesh_output_groups_;  // runtime group layout id -> bind group

    mutable std::mutex bridges_mutex_;
    std::vector<CommandListBridge*> bridges_;

    mutable std::mutex retire_mutex_;
    std::unordered_set<uint32_t> frame_used_resources_;
    std::unordered_map<uint32_t, uint64_t> resource_last_use_;
    uint64_t resource_use_tick_ = 0;
    std::vector<uint32_t> deferred_destroys_;
    std::unordered_set<uint32_t> deferred_destroy_set_;
    uint64_t synced_present_count_ = 0;
};

// Graphics PSO as seen by the bridge: the G7 GraphicsPso plus everything needed to emit its packets.
class GraphicsPipelineState {
public:
    pso::GraphicsPso& pso() noexcept { return pso_; }
    const d3d::RootSignature* root_signature() const noexcept { return root_signature_; }
    uint64_t diagnostics() const noexcept { return diagnostics_; }

private:
    friend class StateBridgeDevice;
    friend class CommandListBridge;
    GraphicsPipelineState(const pso::D3D12GraphicsPsoDesc& desc, const pso::PipelineKeyOptions& options)
        : pso_(desc, options) {}

    pso::GraphicsPso pso_;
    const d3d::RootSignature* root_signature_ = nullptr;
    uint32_t vs_module_id_ = 0;
    uint32_t fs_module_id_ = 0;
    StateBridgeDevice::PipelineLayoutRecord* layout_ = nullptr;
    std::vector<d3d::BindGroupPlan> plans_;  // one per layout group
    uint64_t diagnostics_ = 0;               // BridgeDiag bits found at creation
};

class ComputePipelineState {
public:
    uint32_t pipeline_id() const noexcept { return pipeline_id_; }
    const d3d::RootSignature* root_signature() const noexcept { return root_signature_; }
    uint64_t diagnostics() const noexcept { return diagnostics_; }

private:
    friend class StateBridgeDevice;
    friend class CommandListBridge;
    ComputePipelineState() = default;

    pso::ComputePipelineKey key_{};
    uint32_t pipeline_id_ = 0;
    const d3d::RootSignature* root_signature_ = nullptr;
    uint32_t cs_module_id_ = 0;
    StateBridgeDevice::PipelineLayoutRecord* layout_ = nullptr;
    std::vector<d3d::BindGroupPlan> plans_;
    uint64_t diagnostics_ = 0;
};

// G66: a mesh pipeline state (see MeshOutputLayout).
class MeshPipelineState {
public:
    GraphicsPipelineState* draw() const noexcept { return draw_.get(); }
    ComputePipelineState* stage() const noexcept { return stage_.get(); }
    const MeshOutputLayout& output() const noexcept { return output_; }
    uint64_t diagnostics() const noexcept { return diagnostics_; }

private:
    friend class StateBridgeDevice;
    friend class CommandListBridge;
    std::unique_ptr<GraphicsPipelineState> draw_;
    std::unique_ptr<ComputePipelineState> stage_;
    MeshOutputLayout output_{};
    uint64_t diagnostics_ = 0;
};

// =====================================================================================================
// Per-command-list recording API (ID3D12GraphicsCommandList subset)
// =====================================================================================================

struct BridgeStats {
    uint64_t draws = 0;
    uint64_t draws_skipped = 0;
    uint64_t dispatches = 0;
    uint64_t render_passes = 0;
    uint64_t pipelines_created = 0;
    uint64_t bind_groups_created = 0;
    uint64_t bind_groups_retired = 0;
    uint64_t bind_groups_invalidated = 0;       // G70: cached bind groups evicted on resource/view unregister
    uint64_t bind_groups_relocation_moved = 0;  // G66: cache hits re-created, a relocated home moved
    uint64_t bind_groups_resource_retired = 0;  // G66: cache hits re-built, a bound resource was destroyed
    uint64_t retired_resources_nulled = 0;      // G66: bindings of destroyed textures/buffers bound null
    uint64_t bind_group_cache_hits = 0;    // G66: bind group lookups answered without a CreateBindGroup
    uint64_t bind_group_cache_misses = 0;  // ... rebuilt (first use or changed descriptors)
    uint64_t memory_bind_groups = 0;            // bind groups over descriptor memory (recorded)
    uint64_t memory_bind_groups_refreshed = 0;  // ... rebuilt at ExecuteCommandLists (words changed)
    uint64_t root_constant_uploads = 0;
    uint64_t clear_fallbacks_dropped = 0;
    uint64_t usage_conflicts = 0;
    uint64_t texture_memory_syncs = 0;
    // Uniform bindings smaller than the stage's block with no room in their buffer, bound as a
    // relocated copy zero-extended to the block size.
    uint64_t uniforms_zero_extended = 0;
    // Read-only storage bindings at an offset not a multiple of 256, bound as a relocated copy (O36).
    uint64_t storage_relocated = 0;
    // Storage-texture bindings of a texture created without STORAGE_BINDING usage, bound as null
    // (a storage view of it would invalidate the bind group and the whole command buffer).
    uint64_t storage_views_without_usage = 0;
    // ... of a texture whose format is not the stage's storage format, bound as null too.
    uint64_t storage_views_format_mismatch = 0;
    // Redirected storage views whose texture is also a render attachment of the draw's pass: counted, not converted.
    uint64_t storage_views_attachment_alias = 0;
    // Compute storage-texture bindings of another format than the stage's bound instead to an
    // intermediate texture in the stage's format (#482: both formats storage-capable and float),
    // converted into the real texture after the dispatch (and from it before, for read access).
    uint64_t storage_views_converted = 0;
    // Conversion dispatches emitted for them (intermediate <-> real texture).
    uint64_t storage_conversions = 0;
    // Storage-texture bindings of a depth/stencil-format texture that could NOT be given a twin
    // (backlog 35): bound null because WebGPU has no storage access on depth formats, so their
    // writes are lost. Counted here, never faked. A twin needs a compute-only 2D binding of the
    // depth aspect in the storage format r32float or r32uint; the stencil plane has no twin (a
    // fragment cannot write the stencil value per pixel).
    uint64_t storage_views_depth_untwinned = 0;
    // Backlog 35: storage-texture bindings of a depth texture bound to a colour-format twin
    // instead, the depth -> twin copies (compute) and twin -> depth copies (a fullscreen draw
    // writing frag_depth) emitted for them. Copies in are skipped while the twin is known equal.
    uint64_t depth_twin_bindings = 0;
    uint64_t depth_twin_copies_in = 0;
    uint64_t depth_twin_copies_out = 0;
    // Dispatches (direct and indirect) dropped because a writable storage binding is null: reads
    // would give zero and atomics no progress, so work-queue / spin-wait loops can hang the device
    // (backlog 39). Their other writes are lost. Counted per dispatch; the distinct pipelines hit
    // are in null_storage_dispatch_pipelines (each logged once through diagnostics).
    uint64_t dispatches_dropped_null_storage = 0;
    uint64_t null_storage_dispatch_pipelines = 0;
    // Heap-indexed bindings (SM 6.6 directly indexed heaps) whose root-constant index fell outside
    // the bound heap: bound as null.
    uint64_t heap_index_misses = 0;
    // ... of those, CB-indexed bindings (G62) whose index could not be read from the constant buffer.
    uint64_t cb_index_misses = 0;
    // G66 ExecuteIndirect: executions emitted, the indirect draws/dispatches they emitted, executions
    // skipped (invalid arguments, no pipeline, storage redirects), and commands dropped over
    // kMaxIndirectCommands.
    uint64_t indirect_executions = 0;
    uint64_t indirect_commands = 0;
    uint64_t indirect_skipped = 0;
    uint64_t indirect_truncated = 0;
    uint64_t indirect_clamps_batched = 0;  // clamps recorded ahead of their draws (backlog 41)
    // G66 mesh pipelines: mesh dispatches emitted (direct, and indirect commands), and mesh work
    // skipped (no mesh pipeline state bound, invalid group counts, bindings that cannot be built).
    uint64_t mesh_dispatches = 0;
    uint64_t mesh_skipped = 0;
    // G66 partly-held mip chains: sampled texture bindings emitted with their base mip level raised
    // to the texture's first held level or the view's min-LOD clamp, and cached bind groups
    // re-created because a bound texture's held levels changed.
    uint64_t mip_clamped_binds = 0;
    uint64_t bind_groups_residency_changed = 0;
};

class CommandListBridge {
public:
    CommandListBridge(StateBridgeDevice& device, CommandEncoder& encoder);
    ~CommandListBridge();
    CommandListBridge(const CommandListBridge&) = delete;
    CommandListBridge& operator=(const CommandListBridge&) = delete;

    // ID3D12GraphicsCommandList::Reset (state back to defaults; the encoder is not cleared).
    void Reset(GraphicsPipelineState* initial_state = nullptr);
    // Flushes pending clears, ends the open render pass and finishes the list.
    void Close();
    // Emits Present into the encoder and flushes end-of-frame deferred DestroyResource packets after it.
    void Present();
    // Drains any pending resource/view invalidations queued by StateBridgeDevice::unregister_resource.
    void flush_invalidations();
    size_t cached_bind_group_count() const;
    bool is_open() const noexcept { return open_; }

    // Pipeline state.
    void SetPipelineState(GraphicsPipelineState* pso);
    void SetComputePipelineState(ComputePipelineState* pso);
    // G66: binds a mesh pipeline state (graphics slot); SetPipelineState unbinds it.
    void SetMeshPipelineState(MeshPipelineState* pso);

    // Output merger / clears.
    void OMSetRenderTargets(uint32_t num_render_target_descriptors,
                            const d3d::CpuDescriptorHandle* render_target_descriptors,
                            bool rts_single_handle_to_descriptor_range,
                            const d3d::CpuDescriptorHandle* depth_stencil_descriptor);
    void ClearRenderTargetView(d3d::CpuDescriptorHandle rtv,
                               const float color_rgba[4],
                               uint32_t num_rects = 0,
                               const pso::D3D12Rect* rects = nullptr);
    void ClearDepthStencilView(d3d::CpuDescriptorHandle dsv,
                               uint32_t clear_flags,
                               float depth,
                               uint8_t stencil,
                               uint32_t num_rects = 0,
                               const pso::D3D12Rect* rects = nullptr);
    void OMSetBlendFactor(const float blend_factor[4]);
    void OMSetStencilRef(uint32_t stencil_ref);

    // Rasterizer.
    void RSSetViewports(uint32_t num_viewports, const pso::D3D12Viewport* viewports);
    void RSSetScissorRects(uint32_t num_rects, const pso::D3D12Rect* rects);

    // Input assembler.
    void IASetPrimitiveTopology(pso::D3DPrimitiveTopology topology);
    void IASetVertexBuffers(uint32_t start_slot, uint32_t num_views, const d3d12_xs::D3D12VertexBufferView* views);
    void IASetIndexBuffer(const d3d12_xs::D3D12IndexBufferView* view);

    // Root signature and arguments.
    d3d::Status SetDescriptorHeaps(uint32_t count, d3d::DescriptorHeap* const* heaps);
    d3d::Status SetGraphicsRootSignature(const d3d::RootSignature* signature);
    d3d::Status SetComputeRootSignature(const d3d::RootSignature* signature);
    d3d::Status SetGraphicsRootDescriptorTable(uint32_t parameter_index, d3d::GpuDescriptorHandle base);
    d3d::Status SetComputeRootDescriptorTable(uint32_t parameter_index, d3d::GpuDescriptorHandle base);
    d3d::Status SetGraphicsRoot32BitConstants(uint32_t parameter_index, uint32_t num_values,
                                              const uint32_t* values, uint32_t dest_offset);
    d3d::Status SetComputeRoot32BitConstants(uint32_t parameter_index, uint32_t num_values,
                                             const uint32_t* values, uint32_t dest_offset);
    d3d::Status SetGraphicsRootConstantBufferView(uint32_t parameter_index, uint64_t gpu_address);
    d3d::Status SetComputeRootConstantBufferView(uint32_t parameter_index, uint64_t gpu_address);
    d3d::Status SetGraphicsRootShaderResourceView(uint32_t parameter_index, uint64_t gpu_address);
    d3d::Status SetGraphicsRootUnorderedAccessView(uint32_t parameter_index, uint64_t gpu_address);

    // Work.
    void DrawInstanced(uint32_t vertex_count_per_instance, uint32_t instance_count,
                       uint32_t start_vertex_location, uint32_t start_instance_location);
    void DrawIndexedInstanced(uint32_t index_count_per_instance, uint32_t instance_count,
                              uint32_t start_index_location, int32_t base_vertex_location,
                              uint32_t start_instance_location);
    void Dispatch(uint32_t thread_group_count_x, uint32_t thread_group_count_y, uint32_t thread_group_count_z);
    // G66: ExecuteIndirect (see IndirectExecution). False when skipped (counted in stats().indirect_skipped).
    bool ExecuteIndirect(const IndirectExecution& execution);
    // G66: DispatchMesh with the bound mesh pipeline state (x-dimension groups only: y and z must
    // be 1). False when skipped (counted in stats().mesh_skipped).
    bool DispatchMesh(uint32_t thread_group_count_x, uint32_t thread_group_count_y, uint32_t thread_group_count_z);

    // Barriers: flush pending clears and end the render pass (WebGPU usage scopes), then track states.
    void ResourceBarrier(uint32_t num_barriers, const d3d::ResourceBarrier* barriers);

    // G66 read-back: ends the open render pass (pending clears first, as for a copy) and emits a
    // ReadbackBuffer / ReadbackTexture packet naming `resource_id`. Returns the encoder offset of
    // the payload (the caller patches its sequence number at execution), or SIZE_MAX when the list
    // is not writable.
    size_t Readback(GpuOpcode opcode, const void* payload, uint32_t payload_bytes, uint32_t resource_id);

    // Emits a WriteTexture packet into this command list's encoder stream (G49b).
    void WriteTexture(uint32_t texture_id, uint32_t mip_level, uint32_t origin_x, uint32_t origin_y,
                      uint32_t origin_z, uint32_t width, uint32_t height, uint32_t depth_or_layers,
                      uint32_t bytes_per_row, uint32_t rows_per_image, const void* data, uint32_t data_len);

    // ExecuteCommandLists: re-reads the descriptor memory of every bind group this list recorded over
    // memory-backed tables (the GPU reads a table when the draw executes) and rewrites the recorded
    // CreateBindGroup2 entries in place where the words changed. Packets it needs first (samplers)
    // go to `prelude`. Returns the number of bind groups rebuilt.
    uint32_t refresh_memory_bind_groups(CommandEncoder& prelude);

    // Inspection.
    uint64_t diagnostics() const noexcept { return diagnostics_; }
    uint64_t state_diagnostics() const noexcept { return tracker_.diagnostics(); }        // pso::StateDiag
    uint64_t pipeline_diagnostics() const noexcept { return tracker_.pipeline_diagnostics(); }  // pso::PipelineDiag
    const std::vector<d3d::UsageConflict>& usage_conflicts() const noexcept { return usage_conflicts_; }
    const d3d::CommandListStateTracker& resource_states() const noexcept { return resource_states_; }
    const BridgeStats& stats() const noexcept { return stats_; }
    const pso::TrackerStats& tracker_stats() const noexcept { return tracker_.stats(); }

private:
    friend class StateBridgeDevice;
    struct BoundGroup {
        uint32_t bind_group_id = 0;
        std::vector<uint32_t> dynamic_offsets;
        uint32_t null_write_entries = 0;  // writable storage bindings of the group bound null
    };
    struct VertexBinding {
        uint32_t buffer_id = 0;
        uint64_t offset = 0;
    };

    bool writable() noexcept;
    void drain_events();
    void emit_begin_render_pass(const pso::RenderPassDesc& pass);
    void end_usage_scope();
    void invalidate_pass_bindings();
    bool apply_bind_groups(d3d::BindPoint bind_point,
                           StateBridgeDevice::PipelineLayoutRecord* layout,
                           const std::vector<d3d::BindGroupPlan>& plans,
                           std::vector<BoundGroup>* bound,
                           bool track_usage);
    bool emit_bind_group(d3d::BindPoint bind_point,
                         const StateBridgeDevice::GroupLayoutRecord& group,
                         d3d::CachedBindGroup* cached,
                         const d3d::BindGroupPlan* plan = nullptr);
    // The CreateBindGroup2 entries for `cached` (samplers are interned through `side`).
    void gpu_entries(const StateBridgeDevice::GroupLayoutRecord& group, const d3d::CachedBindGroup& cached,
                     CommandEncoder& side, std::vector<GpuBindGroupEntry2>* out);
    // G66 (#691): true when a relocation `cached` was emitted with now resolves to another home.
    bool relocations_moved(d3d::CachedBindGroup* cached);
    bool binds_relocation_home(const d3d::CachedBindGroup& cached) const;
    // G66 (#691): true when `cached` binds a texture or buffer the device destroyed since it was built.
    bool binds_retired(d3d::CachedBindGroup* cached) const;
    // G66 partly-held mip chains: hash of (texture, clamped base mip) over a group's texture entries,
    // and whether a cached group's clamps changed since it was emitted (the residency epoch moved).
    uint64_t residency_key(const d3d::CachedBindGroup& cached) const;
    bool residency_changed(d3d::CachedBindGroup* cached) const;
    // Relocations and interned views the last gpu_entries call made.
    std::vector<d3d::CachedBindGroup::Relocation> pending_relocations_;
    std::vector<uint32_t> pending_view_ids_;
    void invalidate_resources_locked(std::span<const uint32_t> resource_ids,
                                     std::span<const uint32_t> view_ids,
                                     std::vector<uint32_t>* retired_bgs);
    bool try_invalidate_immediately(std::span<const uint32_t> resource_ids,
                                    std::span<const uint32_t> view_ids,
                                    std::vector<uint32_t>* retired_bgs);
    void queue_invalidation(std::span<const uint32_t> resource_ids, std::span<const uint32_t> view_ids);
    void drain_invalidations_locked();
    void note_recorded_resource(uint32_t resource_id);
    void clear_recorded_resources(bool executed);
    void collect_unexecuted_resources(std::unordered_set<uint32_t>* out) const;
    void mark_executed_and_pin_in_frame();
    void note_bind_group_used_locked(const d3d::CachedBindGroup& cached, uint32_t bind_group_id);
    void sync_bound_textures(const StateBridgeDevice::GroupLayoutRecord& group, const d3d::CachedBindGroup& cached,
                             CommandEncoder& target);
    void add_usage(const d3d::BindGroupEntry& entry, const d3d::ShaderBinding& join);
    GpuTextureViewDesc texture_view_desc(const d3d::ViewDescriptor& view);
    uint32_t upload_root_constants(d3d::BindPoint bind_point, uint32_t parameter_index);
    bool prepare_render_draw();
    // Ends the render pass and binds the compute PSO and its bind groups (null: nothing to dispatch).
    ComputePipelineState* prepare_compute_dispatch();
    ComputePipelineState* prepare_compute_dispatch(ComputePipelineState* state, d3d::BindPoint bind_point);
    // G66 mesh pipelines: the mesh stage's compute pipeline and bindings (output group included),
    // and the draw's render pass and bindings. False when they cannot be prepared.
    bool prepare_mesh_stage();
    bool prepare_mesh_draw();
    // G66: emits the clamp pass for `e` (count commands) and returns the ring offset of its output.
    bool emit_indirect_clamp(const IndirectExecution& e, uint32_t count, uint32_t* ring, uint32_t* out_offset);
    void apply_input_assembler(bool indexed);

    // #482: a compute storage-texture binding redirected to an intermediate texture in the stage's
    // storage format.
    struct StorageRedirect {
        size_t entry = 0;            // index in the bind group's entries
        uint32_t texture = 0;        // real texture
        uint32_t intermediate = 0;   // bound instead
        uint8_t format = 0;          // real texture's format
        uint8_t storage_format = 0;  // the stage's (and the intermediate's) format
        uint8_t dimension = 0;       // GpuTextureViewDimension: 2D, 2D array or 3D
        uint8_t access = 0;          // GpuStorageTextureAccess
        uint32_t mip = 0;
        uint32_t base_layer = 0;
        uint32_t layer_count = 1;
        uint32_t extent[3] = {1, 1, 1};  // texels of the view (z: depth or layers)
        bool depth_twin = false;     // backlog 35: `intermediate` is the colour twin of a depth plane
        uint8_t depth_format = 0;    // depth_twin: the depth texture's WebGPU format
        bool same_target(const StorageRedirect& o) const noexcept {
            return entry == o.entry && texture == o.texture && intermediate == o.intermediate &&
                   format == o.format && storage_format == o.storage_format && dimension == o.dimension &&
                   mip == o.mip && base_layer == o.base_layer && layer_count == o.layer_count;
        }
    };
    bool storage_redirect(const GpuBindGroupLayoutEntry& layout, uint32_t resource_id, const TextureInfo& info,
                          uint8_t own_format, const GpuTextureViewDesc& view, CommandEncoder& side,
                          StorageRedirect* out);
    // Emits one conversion dispatch: real -> intermediate (`into_intermediate`) or back.
    void emit_storage_conversion(const StorageRedirect& r, bool into_intermediate);
    // Backlog 35: a compute storage binding of a depth texture bound to a colour twin. True (and
    // `out` filled) when the binding can be twinned.
    bool depth_twin_redirect(const GpuBindGroupLayoutEntry& layout, uint32_t resource_id, const TextureInfo& info,
                             const GpuTextureViewDesc& view, CommandEncoder& side, StorageRedirect* out);
    // Depth plane -> twin (compute), skipped while the twin is known equal; twin -> depth plane (draw).
    bool emit_depth_twin_in(const StorageRedirect& r);
    void emit_depth_twin_out(const StorageRedirect& r);
    void emit_depth_twin_out_sub(const StorageRedirect& r, uint32_t mip, uint32_t layer);
    // Writes the twin's pending subresources of `texture` back into it (before a copy out of it).
    void flush_depth_twin(uint32_t texture);
    // The last redirect that twinned each depth texture (its twin and formats), for flush_depth_twin.
    std::unordered_map<uint32_t, StorageRedirect> depth_twin_flush_;
    DepthTwinTracker depth_twin_;
    // Fragment-stage redirects of the open render pass: converted out when the pass ends.
    std::vector<StorageRedirect> pass_redirects_;

    StateBridgeDevice& device_;
    CommandEncoder& encoder_;
    pso::CommandListStateTracker tracker_;
    d3d::RootArgumentState root_args_;
    d3d::BindGroupCache bind_groups_;
    d3d::CommandListStateTracker resource_states_;
    d3d::UsageScope usage_scope_;
    std::vector<d3d::UsageConflict> usage_conflicts_;

    bool open_ = false;
    GraphicsPipelineState* graphics_pso_ = nullptr;
    ComputePipelineState* compute_pso_ = nullptr;
    MeshPipelineState* mesh_pso_ = nullptr;
    std::unordered_map<uint32_t, uint32_t> empty_bind_groups_;  // group layout id -> bind group id
    // Bind groups recorded over descriptor memory: where their entries sit in the encoder, what
    // they were built from.
    struct MemoryBindGroup {
        size_t entries_offset = 0;  // byte offset of the GpuBindGroupEntry2 array in encoder_
        const StateBridgeDevice::GroupLayoutRecord* group = nullptr;
        d3d::BindGroupPlan plan;
        d3d::CachedBindGroup built;
        bool invalidated = false;
        // #663: binds a relocation home (BridgeOptions::relocation_home), and the relocation
        // epoch its entries were resolved at.
        bool binds_home = false;
        uint64_t relocation_epoch = 0;
    };
    std::vector<MemoryBindGroup> memory_groups_;
    mutable std::mutex cache_mutex_;
    mutable std::mutex pending_mutex_;
    mutable std::mutex recorded_mutex_;
    std::unordered_set<uint32_t> recorded_resources_;
    bool executed_ = false;
    std::vector<uint32_t> pending_invalidated_resources_;
    std::vector<uint32_t> pending_invalidated_views_;
    std::unordered_set<uint64_t> invalidated_key_hashes_;
    std::vector<uint32_t> invalidated_bgs_to_retire_;
    std::atomic<uint64_t> pending_invalidation_epoch_{0};
    uint64_t seen_invalidation_epoch_ = 0;
    // Storage redirects the last gpu_entries call made, and those of each bind group (by gpu id).
    std::vector<StorageRedirect> pending_redirects_;
    uint32_t pending_null_writes_ = 0;  // null writable storage bindings of the last gpu_entries call
    std::unordered_set<uint32_t> null_storage_pipelines_;  // compute pipelines already counted
    // True (and counted) when a bound group has a null writable storage binding: the dispatch is dropped.
    bool drop_null_storage_dispatch(uint32_t pipeline_id);
    std::array<uint32_t, pso::kMaxColorTargets + 1> om_attachments_{};  // texture ids of the bound RTVs + DSV
    std::unordered_map<uint32_t, std::vector<StorageRedirect>> redirects_;
    static constexpr size_t kNoPacket = ~size_t{0};  // created by an earlier recording
    void note_memory_group(const StateBridgeDevice::GroupLayoutRecord& group, const d3d::BindGroupPlan& plan,
                           const d3d::CachedBindGroup& cached, size_t entries_offset);

    // Input assembler state.
    std::array<VertexBinding, pso::kMaxVertexBuffers> vertex_buffers_{};
    uint32_t vertex_dirty_ = 0;
    d3d12_xs::D3D12IndexBufferView index_buffer_{};
    bool index_dirty_ = false;

    // State applied in the open WebGPU pass.
    bool render_pass_open_ = false;
    std::vector<BoundGroup> render_groups_;
    std::vector<BoundGroup> compute_groups_;
    std::unordered_set<uint32_t> scoped_bind_groups_;  // bind groups already added to the usage scope
    bool compute_pass_valid_ = false;
    uint32_t compute_pipeline_bound_ = 0;
    IndirectClampBatch clamp_batch_;  // backlog 41: the open run of batched ExecuteIndirect clamps
    // Root constants last uploaded per (bind point, parameter): values + ring offset.
    struct RootConstantUpload {
        std::vector<uint32_t> values;
        uint32_t offset = 0;
        bool valid = false;
    };
    std::array<std::array<RootConstantUpload, d3d::kMaxRootParameters>, d3d::kBindPointCount> root_uploads_{};

    uint64_t diagnostics_ = 0;
    BridgeStats stats_{};
};

}  // namespace playgta6::gpu::bridge
