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
#include "gpu/descriptor_heap.h"
#include "gpu/pso_cache.h"

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
};

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
inline constexpr uint64_t StorageViewNoUsage = 1ull << 17;      // storage view of a texture without storage usage
                                                                // or of another format than the stage's: null
}  // namespace BridgeDiag

// =====================================================================================================
// Device-level state
// =====================================================================================================

struct BridgeOptions {
    // Shared resource id allocator (must be thread-safe, never return 0). Default: counter from 1.
    std::function<uint32_t()> id_allocator;
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

class GraphicsPipelineState;
class ComputePipelineState;

struct ViewRecord {
    uint32_t texture_id = 0;  // packet texture id (kGpuSwapchainTextureId for swapchain buffers)
    uint32_t resource_id = 0; // runtime resource id (usage tracking)
    GpuTextureViewDesc view{};
    d3d::SubresourceRange range{};
    d3d::ResourceUsage usage = d3d::ResourceUsage::RenderTarget;
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
    // Final Release of a PSO: drops its pipeline-cache references and emits DestroyResource for the
    // pipelines the cache evicted (only with EvictUnreferenced caches).
    void release_pipeline_state(std::unique_ptr<GraphicsPipelineState> pso, CommandEncoder& encoder);

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
    };
    struct PipelineLayoutRecord {
        uint32_t id = 0;
        uint64_t content_id = 0;
        std::vector<GroupLayoutRecord*> groups;
        bool emitted = false;
    };

    uint32_t intern_view(const ViewRecord& record) const;
    void retire_views(uint32_t texture_id);

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
    // Returns the sampler id for `desc`, emitting CreateSampler2 on first use.
    uint32_t intern_sampler(const d3d::SamplerDesc& desc, CommandEncoder& encoder, uint64_t* diagnostics);
    // Root constant ring buffer id; emits its CreateBuffer on first use.
    uint32_t ensure_root_constant_ring(CommandEncoder& encoder);
    // Reserves `bytes` (256-byte aligned start) in the ring and returns the offset.
    uint32_t allocate_root_constants(uint32_t bytes, bool* wrapped);
    // True exactly once per compute pipeline id: the caller emits CreateComputePipeline2.
    bool claim_compute_creation(uint32_t pipeline_id);

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
    // Drops the intermediates and conversion bind groups of `resource_id` (destroyed at next use).
    void retire_intermediates(uint32_t resource_id);
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
    std::unordered_map<uint64_t, uint32_t> shader_modules_;

    mutable std::mutex views_mutex_;
    mutable std::deque<ViewRecord> views_;  // index = view_id - 1 (stable addresses)
    mutable std::unordered_map<std::string, uint32_t> view_ids_;
    // O36: views die with their texture: the id is recycled (free list) and the interned key dropped.
    mutable std::deque<std::string> view_keys_;                               // index = view_id - 1; empty = free
    mutable std::unordered_map<uint32_t, std::vector<uint32_t>> views_by_texture_;
    mutable std::vector<uint32_t> free_views_;

    std::mutex objects_mutex_;  // layouts, samplers, root constant ring
    std::vector<std::unique_ptr<GroupLayoutRecord>> group_layouts_;
    std::vector<std::unique_ptr<PipelineLayoutRecord>> pipeline_layouts_;
    std::map<std::pair<uint64_t, uint64_t>, PipelineLayoutRecord*> layouts_by_shaders_;
    std::unordered_map<std::string, uint32_t> samplers_;
    std::unordered_set<uint32_t> emitted_compute_;
    uint32_t ring_buffer_id_ = 0;
    uint32_t ring_cursor_ = 0;
    std::map<std::pair<uint32_t, uint8_t>, uint32_t> intermediates_;     // (texture, format) -> texture
    std::map<uint32_t, StorageConversionPipeline> conversion_pipelines_;  // dst format << 8 | dimension
    std::map<std::array<uint32_t, 6>, uint32_t> conversion_groups_;      // (layout, src, dst, mip, layers)
    std::vector<uint32_t> retired_ids_;                                   // DestroyResource at next use
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
    uint64_t bind_groups_relocation_moved = 0;  // G66: cache hits re-created, a relocated home moved
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
    // Heap-indexed bindings (SM 6.6 directly indexed heaps) whose root-constant index fell outside
    // the bound heap: bound as null.
    uint64_t heap_index_misses = 0;
    // ... of those, CB-indexed bindings (G62) whose index could not be read from the constant buffer.
    uint64_t cb_index_misses = 0;
};

class CommandListBridge {
public:
    CommandListBridge(StateBridgeDevice& device, CommandEncoder& encoder);

    // ID3D12GraphicsCommandList::Reset (state back to defaults; the encoder is not cleared).
    void Reset(GraphicsPipelineState* initial_state = nullptr);
    // Flushes pending clears, ends the open render pass and finishes the list.
    void Close();
    bool is_open() const noexcept { return open_; }

    // Pipeline state.
    void SetPipelineState(GraphicsPipelineState* pso);
    void SetComputePipelineState(ComputePipelineState* pso);

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

    // Barriers: flush pending clears and end the render pass (WebGPU usage scopes), then track states.
    void ResourceBarrier(uint32_t num_barriers, const d3d::ResourceBarrier* barriers);

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
    struct BoundGroup {
        uint32_t bind_group_id = 0;
        std::vector<uint32_t> dynamic_offsets;
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
    // Relocations the last gpu_entries call made.
    std::vector<d3d::CachedBindGroup::Relocation> pending_relocations_;
    void sync_bound_textures(const StateBridgeDevice::GroupLayoutRecord& group, const d3d::CachedBindGroup& cached,
                             CommandEncoder& target);
    void add_usage(const d3d::BindGroupEntry& entry, const d3d::ShaderBinding& join);
    GpuTextureViewDesc texture_view_desc(const d3d::ViewDescriptor& view);
    uint32_t upload_root_constants(d3d::BindPoint bind_point, uint32_t parameter_index);
    bool prepare_render_draw();
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
    std::unordered_map<uint32_t, uint32_t> empty_bind_groups_;  // group layout id -> bind group id
    // Bind groups recorded over descriptor memory: where their entries sit in the encoder, what
    // they were built from.
    struct MemoryBindGroup {
        size_t entries_offset = 0;  // byte offset of the GpuBindGroupEntry2 array in encoder_
        const StateBridgeDevice::GroupLayoutRecord* group = nullptr;
        d3d::BindGroupPlan plan;
        d3d::CachedBindGroup built;
    };
    std::vector<MemoryBindGroup> memory_groups_;
    // Storage redirects the last gpu_entries call made, and those of each bind group (by gpu id).
    std::vector<StorageRedirect> pending_redirects_;
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
