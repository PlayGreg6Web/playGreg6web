#pragma once

// D3D12-XS descriptor heaps, root signatures, root-argument state, bind-group
// caching and resource state tracking.
//
// This is pure CPU-side bookkeeping used by the d3d12_xs shim layer before any
// WebGPU bind group can be built. It emits no GPU packets.
//
// Threading contract
// ------------------
// * DescriptorHeapRegistry: create_heap/destroy_heap and handle resolution are
//   thread-safe. Descriptor writes (create_*_view, create_sampler, copies) may
//   run concurrently from several threads as long as their destination ranges
//   are disjoint (same rule as D3D12). Writing a slot while another thread reads
//   it (e.g. while a bind group is compiled from it) is an application error,
//   exactly as in D3D12; it cannot crash but the read may observe a torn record.
// * DescriptorHeap generation counters are atomic and may be read from any
//   thread.
// * RootArgumentState, BindGroupCache, CommandListStateTracker and UsageScope
//   are single-threaded objects (one per command list / recording thread).
// * ResourceStateRegistry is thread-safe (internal shared_mutex).
// * Destroying a heap or unregistering a resource that is still referenced by a
//   recording command list is an application error (as in D3D12).
//
// Register-slot model (join with the shader-side reflection, S6 / shaders/src/reflect.rs)
// ------------------------------------------------------------------------------------
// WebGPU group/binding numbers are owned by the shader: S6 reflects every used
// binding as {group, binding, register: {class, index, space}, root_constants}
// in layout-manifest.json. This file never invents group/binding numbers; it
// joins S6 bindings to root-signature contents through the D3D register:
//   1. A root signature is flattened into an ordered list of BindingSlots: root
//      parameters in declaration order (a descriptor table contributes one slot
//      per range, a root CBV/SRV/UAV or root-constants parameter one slot), then
//      every static sampler. Root constants are a CBV-class slot (the shader sees
//      a cbuffer) with source == RootConstants.
//   2. Register lookup (class, space, register, stage) -> the unique slot whose
//      [base_register, base_register + descriptor_count) contains the register
//      (unbounded: >= base_register) and whose visibility includes the stage;
//      for table slots the descriptor sits at table_offset + (register -
//      base_register) from the table start. Compute pipelines ignore visibility.
//      Register classes: b = Cbv, t = Srv, u = Uav, s = Sampler.
//   3. An S6 binding with root_constants: true (register null) joins the first
//      root-constants parameter whose visibility includes the stage.
// BindGroupPlan precomputes this join once per (root signature, bind group
// layout); BindGroupCache then compiles/caches bind groups from it.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu::d3d12_state {

// ---------------------------------------------------------------------------
// D3D12 enums and constants. Numeric values match the public D3D12 API. These
// are namespaced so they cannot clash with declarations later added to
// d3d12_xs.h.
// ---------------------------------------------------------------------------

enum class DescriptorHeapType : uint32_t {
    CbvSrvUav = 0,
    Sampler = 1,
    Rtv = 2,
    Dsv = 3,
};
inline constexpr uint32_t kDescriptorHeapTypeCount = 4;

namespace DescriptorHeapFlags {
inline constexpr uint32_t None = 0x0u;
inline constexpr uint32_t ShaderVisible = 0x1u;
}  // namespace DescriptorHeapFlags

enum class DescriptorRangeType : uint32_t {
    Srv = 0,
    Uav = 1,
    Cbv = 2,
    Sampler = 3,
};

enum class RootParameterType : uint32_t {
    DescriptorTable = 0,
    Constants32Bit = 1,
    Cbv = 2,
    Srv = 3,
    Uav = 4,
};

enum class ShaderVisibility : uint32_t {
    All = 0,
    Vertex = 1,
    Hull = 2,
    Domain = 3,
    Geometry = 4,
    Pixel = 5,
    Amplification = 6,
    Mesh = 7,
};

enum class RootSignatureVersion : uint32_t {
    V1_0 = 0x1,
    V1_1 = 0x2,
    V1_2 = 0x3,
};

namespace RootSignatureFlags {
inline constexpr uint32_t None = 0x0u;
inline constexpr uint32_t AllowInputAssemblerInputLayout = 0x1u;
inline constexpr uint32_t DenyVertexShaderRootAccess = 0x2u;
inline constexpr uint32_t DenyHullShaderRootAccess = 0x4u;
inline constexpr uint32_t DenyDomainShaderRootAccess = 0x8u;
inline constexpr uint32_t DenyGeometryShaderRootAccess = 0x10u;
inline constexpr uint32_t DenyPixelShaderRootAccess = 0x20u;
inline constexpr uint32_t AllowStreamOutput = 0x40u;
inline constexpr uint32_t LocalRootSignature = 0x80u;
inline constexpr uint32_t DenyAmplificationShaderRootAccess = 0x100u;
inline constexpr uint32_t DenyMeshShaderRootAccess = 0x200u;
inline constexpr uint32_t CbvSrvUavHeapDirectlyIndexed = 0x400u;
inline constexpr uint32_t SamplerHeapDirectlyIndexed = 0x800u;
}  // namespace RootSignatureFlags

namespace DescriptorRangeFlags {
inline constexpr uint32_t None = 0x0u;
inline constexpr uint32_t DescriptorsVolatile = 0x1u;
inline constexpr uint32_t DataVolatile = 0x2u;
inline constexpr uint32_t DataStaticWhileSetAtExecute = 0x4u;
inline constexpr uint32_t DataStatic = 0x8u;
inline constexpr uint32_t DescriptorsStaticKeepingBufferBoundsChecks = 0x10000u;
}  // namespace DescriptorRangeFlags

namespace RootDescriptorFlags {
inline constexpr uint32_t None = 0x0u;
inline constexpr uint32_t DataVolatile = 0x2u;
inline constexpr uint32_t DataStaticWhileSetAtExecute = 0x4u;
inline constexpr uint32_t DataStatic = 0x8u;
}  // namespace RootDescriptorFlags

inline constexpr uint32_t kDescriptorRangeOffsetAppend = 0xffffffffu;
inline constexpr uint32_t kUnboundedDescriptorCount = 0xffffffffu;
inline constexpr uint32_t kMaxRootSignatureDwords = 64;
inline constexpr uint32_t kMaxRootParameters = 64;
inline constexpr uint32_t kMaxStaticSamplers = 2032;
inline constexpr uint32_t kMaxShaderVisibleSamplers = 2048;
inline constexpr uint32_t kDefaultShader4ComponentMapping = 0x1688u;

namespace ResourceStates {
inline constexpr uint32_t Common = 0x0u;
inline constexpr uint32_t VertexAndConstantBuffer = 0x1u;
inline constexpr uint32_t IndexBuffer = 0x2u;
inline constexpr uint32_t RenderTarget = 0x4u;
inline constexpr uint32_t UnorderedAccess = 0x8u;
inline constexpr uint32_t DepthWrite = 0x10u;
inline constexpr uint32_t DepthRead = 0x20u;
inline constexpr uint32_t NonPixelShaderResource = 0x40u;
inline constexpr uint32_t PixelShaderResource = 0x80u;
inline constexpr uint32_t StreamOut = 0x100u;
inline constexpr uint32_t IndirectArgument = 0x200u;
inline constexpr uint32_t Predication = 0x200u;
inline constexpr uint32_t CopyDest = 0x400u;
inline constexpr uint32_t CopySource = 0x800u;
inline constexpr uint32_t ResolveDest = 0x1000u;
inline constexpr uint32_t ResolveSource = 0x2000u;
inline constexpr uint32_t VideoDecodeRead = 0x10000u;
inline constexpr uint32_t VideoDecodeWrite = 0x20000u;
inline constexpr uint32_t VideoProcessRead = 0x40000u;
inline constexpr uint32_t VideoProcessWrite = 0x80000u;
inline constexpr uint32_t VideoEncodeRead = 0x200000u;
inline constexpr uint32_t RaytracingAccelerationStructure = 0x400000u;
inline constexpr uint32_t VideoEncodeWrite = 0x800000u;
inline constexpr uint32_t ShadingRateSource = 0x1000000u;
inline constexpr uint32_t GenericRead = 0x1u | 0x2u | 0x40u | 0x80u | 0x200u | 0x800u;
inline constexpr uint32_t AllShaderResource = 0x40u | 0x80u;
inline constexpr uint32_t Present = 0x0u;
// States that are writable and therefore must not be combined with any other bit.
inline constexpr uint32_t WriteMask = RenderTarget | UnorderedAccess | DepthWrite | StreamOut |
                                      CopyDest | ResolveDest | VideoDecodeWrite |
                                      VideoProcessWrite | VideoEncodeWrite;
}  // namespace ResourceStates

// Tracker-internal sentinel: "state not known yet in this command list".
inline constexpr uint32_t kStateUnknown = 0xffffffffu;

enum class ResourceBarrierType : uint32_t {
    Transition = 0,
    Aliasing = 1,
    Uav = 2,
};

namespace ResourceBarrierFlags {
inline constexpr uint32_t None = 0x0u;
inline constexpr uint32_t BeginOnly = 0x1u;
inline constexpr uint32_t EndOnly = 0x2u;
}  // namespace ResourceBarrierFlags

inline constexpr uint32_t kAllSubresources = 0xffffffffu;

// D3D12_SRV_DIMENSION / D3D12_UAV_DIMENSION / D3D12_RTV_DIMENSION share these values.
namespace ViewDimension {
inline constexpr uint32_t Unknown = 0;
inline constexpr uint32_t Buffer = 1;
inline constexpr uint32_t Texture1D = 2;
inline constexpr uint32_t Texture1DArray = 3;
inline constexpr uint32_t Texture2D = 4;
inline constexpr uint32_t Texture2DArray = 5;
inline constexpr uint32_t Texture2DMs = 6;
inline constexpr uint32_t Texture2DMsArray = 7;
inline constexpr uint32_t Texture3D = 8;
inline constexpr uint32_t TextureCube = 9;       // SRV only
inline constexpr uint32_t TextureCubeArray = 10;  // SRV only
}  // namespace ViewDimension

// D3D12_DSV_DIMENSION (different numbering from the other view dimensions).
namespace DsvDimension {
inline constexpr uint32_t Unknown = 0;
inline constexpr uint32_t Texture1D = 1;
inline constexpr uint32_t Texture1DArray = 2;
inline constexpr uint32_t Texture2D = 3;
inline constexpr uint32_t Texture2DArray = 4;
inline constexpr uint32_t Texture2DMs = 5;
inline constexpr uint32_t Texture2DMsArray = 6;
}  // namespace DsvDimension

namespace BufferViewFlags {
inline constexpr uint32_t None = 0x0u;
inline constexpr uint32_t Raw = 0x1u;
}  // namespace BufferViewFlags

namespace DsvFlags {
inline constexpr uint32_t None = 0x0u;
inline constexpr uint32_t ReadOnlyDepth = 0x1u;
inline constexpr uint32_t ReadOnlyStencil = 0x2u;
}  // namespace DsvFlags

// D3D12_CPU_DESCRIPTOR_HANDLE / D3D12_GPU_DESCRIPTOR_HANDLE (opaque 64-bit values).
struct CpuDescriptorHandle {
    uint64_t ptr = 0;
};
struct GpuDescriptorHandle {
    uint64_t ptr = 0;
};

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

enum class Status : uint32_t {
    Ok = 0,
    InvalidArgument,
    InvalidHandle,          // not a handle produced by this registry / wrong CPU-vs-GPU tag
    StaleHandle,            // heap destroyed (or slot reused)
    HandleOutOfRange,       // index (+ count) beyond heap capacity
    MisalignedHandle,       // byte offset not a multiple of the increment size
    WrongHeapType,
    HeapNotShaderVisible,
    HeapNotBound,           // table handle does not point into a currently bound heap
    TooManyHeaps,
    CountMismatch,          // CopyDescriptors source/destination totals differ
    InvalidRootParameterIndex,
    RootParameterTypeMismatch,
    RootSignatureNotSet,
    RootArgumentNotSet,
    ConstantsOutOfRange,
    // Root signature validation / blob parsing.
    UnsupportedVersion,
    InvalidParameterType,
    InvalidShaderVisibility,
    InvalidRangeType,
    EmptyDescriptorTable,
    ZeroDescriptorRange,
    MixedSamplerTable,
    AppendAfterUnboundedRange,
    DescriptorOffsetOverflow,
    RegisterOverlap,
    RootSignatureTooLarge,
    TooManyStaticSamplers,
    BlobTruncated,
    BlobBadMagic,
    BlobMissingRootSignaturePart,
    BlobOffsetOutOfBounds,
    // Resource state tracking.
    UnknownResource,
    ResourceAlreadyRegistered,
};

const char* status_name(Status status) noexcept;

// ---------------------------------------------------------------------------
// Descriptor records
// ---------------------------------------------------------------------------

enum class DescriptorKind : uint8_t {
    Empty = 0,  // never written
    Cbv = 1,
    Srv = 2,
    Uav = 3,
    Sampler = 4,
    Rtv = 5,
    Dsv = 6,
};

// D3D12_CONSTANT_BUFFER_VIEW_DESC.
struct ConstantBufferViewDesc {
    uint64_t buffer_location = 0;  // GPU virtual address
    uint32_t size_in_bytes = 0;    // multiple of 256
};

// Flattened D3D12_{SHADER_RESOURCE,UNORDERED_ACCESS,RENDER_TARGET,DEPTH_STENCIL}_VIEW_DESC.
// `dimension` uses ViewDimension for SRV/UAV/RTV and DsvDimension for DSV.
// Texture fields: first_mip = MostDetailedMip / MipSlice, mip_levels (SRV only;
// 0xffff = all remaining), first_array_slice / array_size (also FirstWSlice /
// WSize for 3D UAV/RTV, First2DArrayFace / NumCubes for cube arrays).
struct ViewDesc {
    uint32_t format = 0;  // DXGI_FORMAT
    uint32_t dimension = ViewDimension::Unknown;
    uint32_t component_mapping = kDefaultShader4ComponentMapping;  // SRV only
    uint32_t flags = 0;  // BufferViewFlags for SRV/UAV buffers, DsvFlags for DSV
    uint64_t first_element = 0;
    uint32_t num_elements = 0;
    uint32_t structure_byte_stride = 0;
    uint64_t counter_offset_in_bytes = 0;  // UAV buffer counter
    uint32_t first_mip = 0;
    uint32_t mip_levels = 0xffffu;
    uint32_t first_array_slice = 0;
    uint32_t array_size = 1;
    uint32_t plane_slice = 0;
    float min_lod_clamp = 0.0f;
};

// Stored CBV/SRV/UAV/RTV/DSV descriptor (64 bytes, no padding, trivially copyable).
// resource_id is the runtime's resource id (0 = null descriptor).
struct ViewDescriptor {
    DescriptorKind kind = DescriptorKind::Empty;
    uint8_t dimension = 0;
    uint16_t flags = 0;
    uint32_t format = 0;
    uint32_t resource_id = 0;
    uint32_t counter_resource_id = 0;
    uint64_t first_element_or_address = 0;  // buffer views: FirstElement; CBV: GPU VA
    uint64_t counter_offset_in_bytes = 0;
    uint32_t num_elements_or_size = 0;  // buffer views: NumElements; CBV: SizeInBytes
    uint32_t structure_byte_stride = 0;
    uint32_t component_mapping = 0;
    uint16_t first_mip = 0;
    uint16_t mip_levels = 0;
    uint32_t first_array_slice = 0;
    uint32_t array_size = 0;
    uint32_t plane_slice = 0;
    float min_lod_clamp = 0.0f;
};
static_assert(sizeof(ViewDescriptor) == 64, "ViewDescriptor must stay 64 bytes");

// D3D12_SAMPLER_DESC (+ D3D12_SAMPLER_FLAGS for desc2 / root signature 1.2).
struct SamplerDesc {
    uint32_t filter = 0x15;  // D3D12_FILTER_MIN_MAG_MIP_LINEAR
    uint32_t address_u = 1;  // D3D12_TEXTURE_ADDRESS_MODE_WRAP
    uint32_t address_v = 1;
    uint32_t address_w = 1;
    float mip_lod_bias = 0.0f;
    uint32_t max_anisotropy = 1;
    uint32_t comparison_func = 0;
    float border_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float min_lod = 0.0f;
    float max_lod = 3.402823466e+38f;
    uint32_t flags = 0;
};

// Stored sampler descriptor (64 bytes).
struct SamplerDescriptor {
    DescriptorKind kind = DescriptorKind::Empty;
    uint8_t reserved[7] = {0, 0, 0, 0, 0, 0, 0};
    SamplerDesc desc{};
};
static_assert(sizeof(SamplerDescriptor) == 64, "SamplerDescriptor must stay 64 bytes");

// ---------------------------------------------------------------------------
// Descriptor heaps
// ---------------------------------------------------------------------------

// GetDescriptorHandleIncrementSize values exposed to the game.
inline constexpr uint32_t descriptor_handle_increment_size(DescriptorHeapType type) noexcept {
    switch (type) {
    case DescriptorHeapType::CbvSrvUav:
        return 32;
    case DescriptorHeapType::Sampler:
        return 16;
    case DescriptorHeapType::Rtv:
        return 32;
    case DescriptorHeapType::Dsv:
        return 32;
    }
    return 0;
}

inline constexpr uint32_t kMaxDescriptorsPerHeap = 1u << 24;  // keeps byte offsets < 2^32
inline constexpr uint32_t kMaxDescriptorHeaps = 1u << 16;
// Slots per generation block (range_changed_since cost is O(count / kGenerationBlockSize)).
inline constexpr uint32_t kGenerationBlockSize = 64;

struct DescriptorHeapDesc {
    DescriptorHeapType type = DescriptorHeapType::CbvSrvUav;
    uint32_t num_descriptors = 0;
    uint32_t flags = DescriptorHeapFlags::None;
    uint32_t node_mask = 0;
    // Non-zero: the heap's CPU descriptor handles are this address + index * increment (an
    // Xbox descriptor heap placed in title memory). Zero: tagged registry handles.
    uint64_t cpu_address = 0;
};

class DescriptorHeap {
public:
    DescriptorHeapType type() const noexcept { return desc_.type; }
    uint32_t capacity() const noexcept { return desc_.num_descriptors; }
    bool shader_visible() const noexcept {
        return (desc_.flags & DescriptorHeapFlags::ShaderVisible) != 0;
    }
    const DescriptorHeapDesc& desc() const noexcept { return desc_; }
    uint32_t serial() const noexcept { return serial_; }
    uint32_t increment() const noexcept { return descriptor_handle_increment_size(desc_.type); }

    // GetCPUDescriptorHandleForHeapStart / GetGPUDescriptorHandleForHeapStart
    // (GPU start is {0} for non-shader-visible heaps, as in D3D12).
    CpuDescriptorHandle cpu_start() const noexcept;
    GpuDescriptorHandle gpu_start() const noexcept;
    CpuDescriptorHandle cpu_handle(uint32_t index) const noexcept;
    GpuDescriptorHandle gpu_handle(uint32_t index) const noexcept;

    // nullptr when out of range or when the heap stores the other record type.
    const ViewDescriptor* view(uint32_t index) const noexcept;
    const SamplerDescriptor* sampler(uint32_t index) const noexcept;

    // Generation tracking. Every write that changes a slot's content stamps it
    // with a new, heap-wide monotonically increasing generation (> 0). Writes of
    // identical content do not bump the generation.
    uint64_t generation() const noexcept { return generation_.load(std::memory_order_acquire); }
    uint64_t slot_generation(uint32_t index) const noexcept;
    // True if any slot in [first, first + count) was changed after generation `since`.
    // O(1) when nothing in the heap changed, otherwise O(count / kGenerationBlockSize + 2 * kGenerationBlockSize).
    bool range_changed_since(uint32_t first, uint32_t count, uint64_t since) const noexcept;

    // Index-based writes (callers validate the index). Return true if content changed.
    bool write_view(uint32_t index, const ViewDescriptor& record) noexcept;
    bool write_sampler(uint32_t index, const SamplerDescriptor& record) noexcept;

private:
    friend class DescriptorHeapRegistry;
    DescriptorHeap(const DescriptorHeapDesc& desc, uint32_t serial);
    void stamp(uint32_t index) noexcept;

    DescriptorHeapDesc desc_{};
    uint32_t serial_ = 0;
    std::vector<ViewDescriptor> views_;
    std::vector<SamplerDescriptor> samplers_;
    std::unique_ptr<uint64_t[]> slot_generations_;
    std::unique_ptr<std::atomic<uint64_t>[]> block_generations_;
    std::atomic<uint64_t> generation_{0};
};

struct ResolvedDescriptor {
    DescriptorHeap* heap = nullptr;
    uint32_t index = 0;
};

// O36 / backlog 28: process-wide gauges of live descriptor objects (every registry and cache in the
// process), for the boot report and the per-minute plateau check of a long run.
struct DescriptorLiveCounts {
    uint64_t heaps = 0;               // live descriptor heaps
    uint64_t heap_slots = 0;          // descriptor slots across them (capacity)
    uint64_t cached_bind_groups = 0;  // BindGroupCache entries, each a live WebGPU bind group
    // Backlog 34: most bind-group entries one cache held within a single recording epoch (the peak
    // of per-draw tables a frame needed), the soft slot budget (0 = none), and the entries made
    // past that budget. The cache grows instead of dropping a draw, so draws dropped for lack of
    // slots stays 0 by construction; the overflow counter is what shows a budget was too small.
    uint64_t peak_bind_groups_per_epoch = 0;
    uint64_t bind_group_slot_budget = 0;
    uint64_t bind_group_budget_overflows = 0;
    uint64_t draws_dropped_no_slot = 0;
};
DescriptorLiveCounts descriptor_live_counts() noexcept;
// Soft per-cache budget of bind-group slots (0 = unlimited). Exceeding it never drops a draw: the
// entry is still built and counted in bind_group_budget_overflows.
void set_bind_group_slot_budget(uint64_t slots) noexcept;
// Resets the peak and overflow counters (tests, per-run reports).
void reset_bind_group_slot_counters() noexcept;

// Device-level owner of all descriptor heaps; resolves opaque handles.
//
// Handle encoding (64-bit): high 32 bits = tag(2) | reuse generation(14) | heap
// slot(16); low 32 bits = byte offset (index * increment). CPU tag = 01b, GPU
// tag = 10b. Games may do arbitrary pointer arithmetic on handles; every use
// re-validates tag, heap liveness, alignment and range.
class DescriptorHeapRegistry {
public:
    DescriptorHeapRegistry();
    ~DescriptorHeapRegistry();
    DescriptorHeapRegistry(const DescriptorHeapRegistry&) = delete;
    DescriptorHeapRegistry& operator=(const DescriptorHeapRegistry&) = delete;

    Status create_heap(const DescriptorHeapDesc& desc, DescriptorHeap** out_heap);
    void destroy_heap(DescriptorHeap* heap);
    // Required alignment of a constant-buffer view's address and size (power of two). D3D12 on PC
    // requires 256 (the default); the Xbox title passes 16-byte aligned views.
    void set_cbv_alignment(uint32_t alignment) noexcept { cbv_alignment_ = alignment; }
    uint32_t heap_count() const noexcept;

    Status resolve_cpu(CpuDescriptorHandle handle, ResolvedDescriptor* out) const noexcept;
    Status resolve_gpu(GpuDescriptorHandle handle, ResolvedDescriptor* out) const noexcept;

    // ID3D12Device::Create*View / CreateSampler. A null desc for SRV/UAV/RTV/DSV
    // means "default view of the whole resource" (dimension Unknown) and requires
    // a non-zero resource; a zero resource with a desc writes a null descriptor.
    // A null CBV desc writes a null CBV.
    Status create_constant_buffer_view(const ConstantBufferViewDesc* desc, CpuDescriptorHandle dst) noexcept;
    Status create_shader_resource_view(uint32_t resource_id, const ViewDesc* desc, CpuDescriptorHandle dst) noexcept;
    Status create_unordered_access_view(
        uint32_t resource_id,
        uint32_t counter_resource_id,
        const ViewDesc* desc,
        CpuDescriptorHandle dst) noexcept;
    Status create_render_target_view(uint32_t resource_id, const ViewDesc* desc, CpuDescriptorHandle dst) noexcept;
    Status create_depth_stencil_view(uint32_t resource_id, const ViewDesc* desc, CpuDescriptorHandle dst) noexcept;
    Status create_sampler(const SamplerDesc& desc, CpuDescriptorHandle dst) noexcept;

    // ID3D12Device::CopyDescriptorsSimple. Overlapping source/destination ranges
    // behave like memmove.
    Status copy_descriptors_simple(
        uint32_t num_descriptors,
        CpuDescriptorHandle dst_start,
        CpuDescriptorHandle src_start,
        DescriptorHeapType type) noexcept;

    // ID3D12Device::CopyDescriptors. Null size arrays mean "every range has size 1".
    // All ranges are validated before anything is written; if any destination
    // range overlaps any source range the sources are staged first, so the
    // result equals "read all sources, then write all destinations".
    Status copy_descriptors(
        uint32_t num_dst_ranges,
        const CpuDescriptorHandle* dst_range_starts,
        const uint32_t* dst_range_sizes,
        uint32_t num_src_ranges,
        const CpuDescriptorHandle* src_range_starts,
        const uint32_t* src_range_sizes,
        DescriptorHeapType type);

    // Decodes the heap serial from a handle without validating liveness.
    static uint32_t handle_serial(uint64_t handle_ptr) noexcept {
        return static_cast<uint32_t>(handle_ptr >> 32) & 0x3fffffffu;
    }

private:
    Status resolve(uint64_t ptr, uint32_t tag, ResolvedDescriptor* out) const noexcept;
    Status resolve_range(CpuDescriptorHandle start, uint32_t count, DescriptorHeapType type, ResolvedDescriptor* out) const noexcept;
    Status write_view_at(CpuDescriptorHandle dst, const ViewDescriptor& record) noexcept;

    struct PlacedHeapRange {
        uint64_t start = 0;
        uint64_t end = 0;
        DescriptorHeap* heap = nullptr;
        uint32_t inc_mask = 0;
        uint32_t inc_shift = 0;
    };

    std::unique_ptr<std::atomic<DescriptorHeap*>[]> slots_;
    std::vector<uint16_t> slot_generation_;  // guarded by mutex_
    std::vector<uint32_t> free_slots_;       // guarded by mutex_
    std::unordered_map<const DescriptorHeap*, uint32_t> owned_;  // heap -> slot, guarded by mutex_
    uint32_t next_unused_slot_ = 0;          // guarded by mutex_
    std::atomic<uint32_t> live_heaps_{0};
    mutable std::mutex mutex_;
    // Heaps placed at a CPU address (DescriptorHeapDesc::cpu_address), sorted by start address.
    std::vector<PlacedHeapRange> placed_;  // guarded by placed_mutex_
    mutable std::shared_mutex placed_mutex_;
    uint32_t cbv_alignment_ = 256;
};

// ---------------------------------------------------------------------------
// Root signatures
// ---------------------------------------------------------------------------

// D3D12_DESCRIPTOR_RANGE1 (flags ignored for version 1.0).
struct DescriptorRangeDesc {
    DescriptorRangeType range_type = DescriptorRangeType::Srv;
    uint32_t num_descriptors = 1;  // kUnboundedDescriptorCount = unbounded
    uint32_t base_shader_register = 0;
    uint32_t register_space = 0;
    uint32_t flags = DescriptorRangeFlags::None;
    uint32_t offset_in_descriptors_from_table_start = kDescriptorRangeOffsetAppend;
};

// D3D12_ROOT_PARAMETER1. `ranges` is used for tables; shader_register /
// register_space for root descriptors and constants; num_32bit_values for
// constants; flags (D3D12_ROOT_DESCRIPTOR_FLAGS) for root descriptors.
struct RootParameterDesc {
    RootParameterType parameter_type = RootParameterType::DescriptorTable;
    ShaderVisibility shader_visibility = ShaderVisibility::All;
    std::vector<DescriptorRangeDesc> ranges;
    uint32_t shader_register = 0;
    uint32_t register_space = 0;
    uint32_t num_32bit_values = 0;
    uint32_t flags = RootDescriptorFlags::None;
};

// D3D12_STATIC_SAMPLER_DESC1 (field order matches the serialized layout).
struct StaticSamplerDesc {
    uint32_t filter = 0x15;
    uint32_t address_u = 1;
    uint32_t address_v = 1;
    uint32_t address_w = 1;
    float mip_lod_bias = 0.0f;
    uint32_t max_anisotropy = 1;
    uint32_t comparison_func = 0;
    uint32_t border_color = 0;  // D3D12_STATIC_BORDER_COLOR
    float min_lod = 0.0f;
    float max_lod = 3.402823466e+38f;
    uint32_t shader_register = 0;
    uint32_t register_space = 0;
    ShaderVisibility shader_visibility = ShaderVisibility::All;
    uint32_t flags = 0;  // D3D12_SAMPLER_FLAGS (version 1.2)
};

struct RootSignatureDesc {
    RootSignatureVersion version = RootSignatureVersion::V1_1;
    std::vector<RootParameterDesc> parameters;
    std::vector<StaticSamplerDesc> static_samplers;
    uint32_t flags = RootSignatureFlags::None;
};

inline constexpr uint32_t kNoIndex = 0xffffffffu;

struct RootSignatureError {
    Status status = Status::Ok;
    uint32_t parameter_index = kNoIndex;  // static sampler index for sampler errors
    uint32_t range_index = kNoIndex;
    uint64_t byte_offset = 0;  // blob parsing only
    bool ok() const noexcept { return status == Status::Ok; }
};

// GPUShaderStage bits.
namespace WebGpuStage {
inline constexpr uint32_t Vertex = 0x1u;
inline constexpr uint32_t Fragment = 0x2u;
inline constexpr uint32_t Compute = 0x4u;
}  // namespace WebGpuStage

// ALL -> vertex|fragment|compute, VERTEX -> vertex, PIXEL -> fragment,
// AMPLIFICATION/MESH -> compute (lowered to compute), HULL/DOMAIN/GEOMETRY -> 0.
uint32_t webgpu_stage_mask(ShaderVisibility visibility) noexcept;

enum class SlotSource : uint32_t {
    DescriptorTable = 0,
    RootDescriptor = 1,
    RootConstants = 2,
    StaticSampler = 3,
};

struct BindingSlot {
    DescriptorRangeType kind = DescriptorRangeType::Srv;  // root constants -> Cbv
    SlotSource source = SlotSource::DescriptorTable;
    uint32_t root_parameter_index = kNoIndex;  // static sampler index for StaticSampler
    uint32_t range_index = kNoIndex;           // descriptor tables only
    uint32_t register_space = 0;
    uint32_t base_register = 0;
    uint32_t descriptor_count = 1;  // kUnboundedDescriptorCount for unbounded ranges
    uint32_t table_offset = 0;      // descriptors from table start, OFFSET_APPEND resolved
    ShaderVisibility visibility = ShaderVisibility::All;
    uint32_t num_32bit_values = 0;  // RootConstants only
    bool unbounded = false;
};

struct BindingLayout {
    std::vector<BindingSlot> slots;
    bool has_unbounded_ranges = false;

    // Register lookup as described in the file header. `stage_mask` is a
    // WebGpuStage mask (WebGpuStage::Compute matches any visibility). Returns
    // nullptr if no slot matches. For table slots *out_table_offset receives the
    // descriptor's offset from the table start (0 for other slots).
    const BindingSlot* find(
        DescriptorRangeType register_class,
        uint32_t register_space,
        uint32_t shader_register,
        uint32_t stage_mask,
        uint32_t* out_table_offset) const noexcept;
};

class RootSignature {
public:
    struct ParameterInfo {
        RootParameterType type = RootParameterType::DescriptorTable;
        DescriptorHeapType table_heap_type = DescriptorHeapType::CbvSrvUav;  // tables only
        uint32_t table_extent = 0;  // descriptors a table spans from its base (bounded part)
        bool table_unbounded = false;
        uint32_t constants_offset = 0;  // dword offset into the per-bind-point constants store
        uint32_t num_32bit_values = 0;
        uint32_t first_slot = 0;
        uint32_t slot_count = 0;
    };

    // Validates and normalizes `desc` (version 1.0 flags are expanded to their
    // 1.1 equivalents: everything volatile), resolves OFFSET_APPEND, computes the
    // content hash and the binding layout.
    static RootSignatureError build(const RootSignatureDesc& desc, RootSignature* out);

    const RootSignatureDesc& desc() const noexcept { return desc_; }
    // Stable FNV-1a 64 hash of the normalized content (independent of version
    // encoding, struct padding and platform). Suitable as a cache key.
    uint64_t hash() const noexcept { return hash_; }
    uint32_t parameter_count() const noexcept { return static_cast<uint32_t>(params_.size()); }
    const ParameterInfo& parameter(uint32_t index) const noexcept { return params_[index]; }
    const BindingLayout& layout() const noexcept { return layout_; }
    uint32_t cost_dwords() const noexcept { return cost_dwords_; }
    uint32_t total_constant_dwords() const noexcept { return total_constant_dwords_; }

private:
    RootSignatureDesc desc_{};
    std::vector<ParameterInfo> params_;
    BindingLayout layout_{};
    uint64_t hash_ = 0;
    uint32_t cost_dwords_ = 0;
    uint32_t total_constant_dwords_ = 0;
};

// Parses a serialized root signature: either a raw RTS0 part or a DXBC
// container holding an "RTS0" part (what ID3D12Device::CreateRootSignature
// receives). Supports versions 1.0, 1.1 and 1.2. Every read is bounds-checked;
// errors carry the failing byte offset. The result still needs RootSignature::build.
RootSignatureError parse_root_signature_blob(const void* data, size_t size, RootSignatureDesc* out);

// Serializes to a raw RTS0 part in desc.version's format (D3D12SerializeVersionedRootSignature).
std::vector<uint8_t> serialize_root_signature(const RootSignatureDesc& desc);

// ---------------------------------------------------------------------------
// Root argument state (per command list)
// ---------------------------------------------------------------------------

enum class BindPoint : uint32_t {
    Graphics = 0,
    Compute = 1,
};
inline constexpr uint32_t kBindPointCount = 2;

struct RootArgument {
    DescriptorHeap* heap = nullptr;  // tables
    uint32_t base_index = 0;         // tables
    uint64_t gpu_address = 0;        // root CBV/SRV/UAV
    bool is_set = false;
};

class RootArgumentState {
public:
    // ID3D12GraphicsCommandList::Reset: clears heaps, signatures and arguments.
    void reset() noexcept;

    // SetDescriptorHeaps: at most one CBV_SRV_UAV and one SAMPLER heap, both
    // shader-visible. Replacing a heap clears (and dirties) every table argument
    // that pointed into the old heap.
    Status set_descriptor_heaps(uint32_t count, DescriptorHeap* const* heaps) noexcept;
    DescriptorHeap* bound_heap(DescriptorHeapType type) const noexcept;

    // Set{Graphics,Compute}RootSignature. Setting a different signature clears
    // and dirties all root arguments of that bind point; re-setting the same one
    // keeps them.
    Status set_root_signature(BindPoint bind_point, const RootSignature* signature) noexcept;
    const RootSignature* root_signature(BindPoint bind_point) const noexcept;

    // Set{Graphics,Compute}RootDescriptorTable. The handle must point into the
    // currently bound shader-visible heap of the table's type, and the table's
    // bounded extent must fit in the heap.
    Status set_descriptor_table(BindPoint bind_point, uint32_t parameter_index, GpuDescriptorHandle base) noexcept;
    // Set{Graphics,Compute}Root{ConstantBuffer,ShaderResource,UnorderedAccess}View.
    Status set_root_view(
        BindPoint bind_point,
        uint32_t parameter_index,
        RootParameterType view_type,
        uint64_t gpu_address) noexcept;
    // Set{Graphics,Compute}Root32BitConstant(s).
    Status set_root_constants(
        BindPoint bind_point,
        uint32_t parameter_index,
        uint32_t num_values,
        const uint32_t* values,
        uint32_t dest_offset) noexcept;
    Status set_root_constant(BindPoint bind_point, uint32_t parameter_index, uint32_t value, uint32_t dest_offset) noexcept {
        return set_root_constants(bind_point, parameter_index, 1, &value, dest_offset);
    }

    const RootArgument& argument(BindPoint bind_point, uint32_t parameter_index) const noexcept;
    std::span<const uint32_t> root_constants(BindPoint bind_point, uint32_t parameter_index) const noexcept;

    // Bit i = root parameter i changed since clear_dirty. A bind group built
    // from a BindGroupPlan is dirty when (dirty_parameters & plan.parameter_mask()) != 0.
    uint64_t dirty_parameters(BindPoint bind_point) const noexcept;
    void clear_dirty(BindPoint bind_point) noexcept;
    // Bit i = root parameter i (table / root view / constants) not set yet; must
    // be 0 before Draw/Dispatch.
    uint64_t unset_parameters(BindPoint bind_point) const noexcept;

private:
    struct PointState {
        const RootSignature* signature = nullptr;
        std::array<RootArgument, kMaxRootParameters> args{};
        std::array<uint32_t, kMaxRootSignatureDwords> constants{};
        uint64_t dirty = 0;
    };
    PointState& point(BindPoint bind_point) noexcept { return points_[static_cast<uint32_t>(bind_point) & 1u]; }
    const PointState& point(BindPoint bind_point) const noexcept {
        return points_[static_cast<uint32_t>(bind_point) & 1u];
    }

    std::array<PointState, kBindPointCount> points_{};
    std::array<DescriptorHeap*, 2> heaps_{};  // [CbvSrvUav, Sampler]
};

// ---------------------------------------------------------------------------
// Bind-group compilation and caching
// ---------------------------------------------------------------------------

// One entry of a WebGPU bind group layout as reflected by S6 (a
// layout-manifest.json "bindings" record): the shader's own @binding plus its
// D3D register join key.
// A descriptor the stage reads straight from a bound heap (SM 6.6 ResourceDescriptorHeap[i] /
// SamplerDescriptorHeap[i], root-signature flags CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED /
// SAMPLER_HEAP_DIRECTLY_INDEXED): the descriptor at byte offset (root constant << shift) of the
// bound heap of that type.
enum class HeapAccess : uint8_t { None = 0, Sampler = 1, Resource = 2 };

// Where a heap-indexed descriptor's index comes from (G62, #482). RootConstantIndex: S34, the index
// is a root constant the stage holds in an SGPR. The others read the 32-bit index from a constant
// buffer at draw time (SM 6.6 ResourceDescriptorHeap[cb.index]): the dword at `cb_byte_offset` of
// the root constants of a constants parameter, of the root CBV, or of the CBV in a table slot.
enum class HeapIndexSource : uint8_t {
    RootConstantIndex = 0,
    CbRootConstants = 1,
    CbRootDescriptor = 2,
    CbTable = 3,
    // No index is read: the index is 0 (plus ShaderBinding::array_index). A lane-indexed heap window
    // without a CB window base (G62).
    Constant = 4,
};

struct ShaderBinding {
    uint32_t binding = 0;  // WebGPU @binding inside the group
    DescriptorRangeType register_class = DescriptorRangeType::Srv;  // b=Cbv, t=Srv, u=Uav, s=Sampler
    uint32_t shader_register = 0;
    uint32_t register_space = 0;
    bool root_constants = false;  // S6 "root_constants": true (register is null)
    // Heap-indexed descriptor (register/space unused): the index is 32-bit value `heap_index_dword`
    // of root-constants parameter `heap_index_parameter`, shifted left by `heap_shift` into a byte
    // offset of the heap.
    HeapAccess heap = HeapAccess::None;
    uint32_t heap_index_parameter = kNoIndex;
    uint32_t heap_index_dword = 0;
    uint32_t heap_shift = 0;
    // CB-indexed heap descriptor (heap_index_source != RootConstantIndex): heap_index_parameter is
    // the root parameter holding (constants) or pointing at (root CBV, table) the constant buffer;
    // the index is ((dword at cb_byte_offset) >> cb_bit_offset) & cb_mask, the byte offset in the
    // heap that index << heap_shift. CbTable: the CBV is descriptor cb_table_slot of the table.
    HeapIndexSource heap_index_source = HeapIndexSource::RootConstantIndex;
    uint32_t cb_table_slot = 0;
    uint32_t cb_byte_offset = 0;
    uint32_t cb_bit_offset = 0;
    uint32_t cb_mask = 0xffffffffu;
    // Non-power-of-two descriptor step (G62, #482; S63 lane-indexed windows the stage scales with
    // s_mul_i32, e.g. * 24): when non-zero the byte offset is index * heap_stride instead of
    // index << heap_shift (32-bit wrapping either way). 0 = shift.
    uint32_t heap_stride = 0;
    // Binding array element (G62, #482; a lane-indexed heap window bound as a WGSL binding_array):
    // element `array_index` of an array of `array_size` heap descriptors. The element reads the heap
    // index above plus array_index (the window starts at that index). array_size 0 = not an array.
    // The bridge expands a layout entry of N elements into N joins, array_index 0..N-1.
    uint32_t array_size = 0;
    uint32_t array_index = 0;
    // Direct table slot (register/space unused): descriptor `table_slot` of descriptor-table
    // parameter `table_parameter`, whether or not the root signature declares a range there. Console
    // stages read their tables as memory, past the ranges a title's root signature declares (G62).
    uint32_t table_parameter = kNoIndex;
    uint32_t table_slot = 0;
    // Inline sampler (register/space unused): the sampler the stage builds from constants in its
    // code, decoded from its S# words at the join (stage_reflection.h).
    bool inline_sampler = false;
    SamplerDesc sampler{};
};

// Two joins bind the same descriptor.
inline bool same_join(const ShaderBinding& a, const ShaderBinding& b) noexcept {
    return a.binding == b.binding && a.register_class == b.register_class && a.shader_register == b.shader_register &&
           a.register_space == b.register_space && a.root_constants == b.root_constants && a.heap == b.heap &&
           a.heap_index_parameter == b.heap_index_parameter && a.heap_index_dword == b.heap_index_dword &&
           a.heap_shift == b.heap_shift && a.heap_index_source == b.heap_index_source &&
           a.cb_table_slot == b.cb_table_slot && a.cb_byte_offset == b.cb_byte_offset &&
           a.cb_bit_offset == b.cb_bit_offset && a.cb_mask == b.cb_mask && a.heap_stride == b.heap_stride &&
           a.array_size == b.array_size &&
           a.array_index == b.array_index && a.table_parameter == b.table_parameter &&
           a.table_slot == b.table_slot && a.inline_sampler == b.inline_sampler &&
           std::memcmp(&a.sampler, &b.sampler, sizeof(a.sampler)) == 0;
}

enum class PlanSource : uint32_t {
    Unresolved = 0,  // no matching root-signature slot: compiles to a Null entry
    DescriptorTable = 1,
    RootDescriptor = 2,
    RootConstants = 3,
    StaticSampler = 4,
    HeapIndexed = 5,  // ShaderBinding::heap: root_parameter_index is the constants parameter
    InlineSampler = 6,  // ShaderBinding::inline_sampler: PlanEntry::sampler
};

struct PlanEntry {
    uint32_t binding = 0;
    PlanSource source = PlanSource::Unresolved;
    DescriptorRangeType register_class = DescriptorRangeType::Srv;
    uint32_t root_parameter_index = kNoIndex;  // static sampler index for StaticSampler
    uint32_t table_offset = 0;                 // DescriptorTable: descriptor offset from the table start
    uint32_t heap_dword = 0;                   // HeapIndexed: the index constant within the parameter
    uint32_t heap_shift = 0;                   // HeapIndexed: byte offset = index << heap_shift
    // HeapIndexed with a CB index (ShaderBinding::heap_index_source): root_parameter_index is the
    // CB's parameter, table_offset the CBV's slot for CbTable, heap_dword unused.
    HeapIndexSource heap_index_source = HeapIndexSource::RootConstantIndex;
    uint32_t cb_byte_offset = 0;
    uint32_t cb_bit_offset = 0;
    uint32_t cb_mask = 0xffffffffu;
    uint32_t heap_stride = 0;      // HeapIndexed: non-zero = byte offset is index * heap_stride (ShaderBinding::heap_stride)
    uint32_t heap_index_bias = 0;  // HeapIndexed: added to the index (binding array element, ShaderBinding::array_index)
    SamplerDesc sampler{};                     // InlineSampler
};

// The (root signature x bind group layout) join, computed once and reused for
// every draw. Entries are sorted by binding.
class BindGroupPlan {
public:
    struct TableWindow {
        uint32_t parameter_index = 0;
        uint32_t first_offset = 0;  // lowest table offset read by the plan
        uint32_t count = 0;         // offsets [first_offset, first_offset + count)
    };

    // stage_mask: WebGpuStage bits of the pipeline stages that use this group
    // (WebGpuStage::Compute for compute pipelines).
    static BindGroupPlan build(
        const RootSignature& signature,
        std::span<const ShaderBinding> bindings,
        uint32_t stage_mask);

    const std::vector<PlanEntry>& entries() const noexcept { return entries_; }
    const std::vector<TableWindow>& table_windows() const noexcept { return windows_; }
    // Bit i = root parameter i feeds this bind group.
    uint64_t parameter_mask() const noexcept { return parameter_mask_; }
    bool has_heap_indexed() const noexcept { return has_heap_indexed_; }
    uint32_t unresolved_count() const noexcept { return unresolved_; }
    uint64_t signature_hash() const noexcept { return signature_hash_; }
    // Content hash of the join (signature hash + entries).
    uint64_t hash() const noexcept { return hash_; }

private:
    std::vector<PlanEntry> entries_;
    std::vector<TableWindow> windows_;
    uint64_t parameter_mask_ = 0;
    uint32_t unresolved_ = 0;
    uint64_t signature_hash_ = 0;
    uint64_t hash_ = 0;
    bool has_heap_indexed_ = false;
};

// The first three values match GpuBindResourceType (G5 CreateBindGroup entries).
enum class BindResourceKind : uint32_t {
    Buffer = 0,
    TextureView = 1,
    Sampler = 2,
    RootConstants = 3,  // root constants: the caller uploads root_constants() into a uniform buffer
    Null = 4,           // empty/null/mismatched descriptor, unresolved binding or unknown GPU VA
};

struct BindGroupEntry {
    uint32_t binding = 0;
    BindResourceKind kind = BindResourceKind::Null;
    uint32_t resource_id = 0;
    uint32_t root_parameter_index = kNoIndex;
    uint64_t offset = 0;
    uint64_t size = 0;  // 0 = to the end of the buffer
    ViewDescriptor view{};     // source descriptor (views / CBVs)
    SamplerDesc sampler{};     // samplers (descriptor or static sampler)
};

struct GpuVaResolution {
    uint32_t resource_id = 0;
    uint64_t offset = 0;         // byte offset of the VA inside the resource
    uint64_t resource_size = 0;  // total byte size of the resource
};
// Maps a GPU virtual address (CBV BufferLocation / root view address) to a
// runtime buffer. Return false if the address is unknown.
using GpuVaResolver = bool (*)(void* user, uint64_t gpu_address, GpuVaResolution* out);

inline constexpr uint64_t kMaxUniformBindingSize = 65536;

// Descriptor memory (G48). On the console the GPU reads a root table's descriptors from the heap's
// memory when the draw executes; the title fills that memory itself (it copies descriptor words
// between slots). A slot source reads table slots from that memory instead of the registry
// records: a heap it `reads` is decoded slot by slot, and a cached bind group over it is
// revalidated by comparing the words of the windows it read.
class DescriptorSlotSource {
public:
    virtual ~DescriptorSlotSource() = default;
    virtual bool reads(const DescriptorHeap& heap) const = 0;
    // Bytes of slots [first, first + count); false (and *out empty) when unreadable.
    virtual bool snapshot(const DescriptorHeap& heap, uint32_t first, uint32_t count, std::vector<uint8_t>* out) const = 0;
    // The view slot `index` holds for a binding of `range` (kind of the binding; resource 0 /
    // address 0 when nothing binds).
    virtual ViewDescriptor view(const DescriptorHeap& heap, uint32_t index, DescriptorRangeType range) const = 0;
    // The sampler slot `index` holds; false when none.
    virtual bool sampler(const DescriptorHeap& heap, uint32_t index, SamplerDesc* out) const = 0;
    // `bytes` bytes of guest memory at GPU address `gpu_address` (constant-buffer contents for
    // CB-indexed heap bindings, HeapIndexSource); false when unreadable. Default: none readable.
    virtual bool read_gpu_memory(uint64_t gpu_address, void* out, size_t bytes) const {
        (void)gpu_address;
        (void)out;
        (void)bytes;
        return false;
    }
};

struct CachedBindGroup {
    uint64_t layout_key = 0;
    std::vector<BindGroupEntry> entries;  // sorted by binding
    uint32_t gpu_id = 0;  // WebGPU bind group id, assigned by the caller after a rebuild
    uint32_t null_entries = 0;
    // Writable storage bindings (storage texture write/read-write, read-write storage buffer) the
    // bridge bound null: a dispatch through them is dropped (guard rail, backlog 39).
    uint32_t null_write_entries = 0;
    uint64_t last_used_epoch = 0;  // BindGroupCache epoch of the last lookup that returned it
    // G66 (#691): entries the bridge bound as a relocated copy (relocate_uniform / relocate_storage)
    // and the home it got, with the relocation epoch at that time. A cache hit does not relocate
    // again, so a home the device has since handed to another range must be noticed here.
    struct Relocation {
        uint32_t entry = 0;
        bool storage = false;
        uint64_t va = 0;
        uint64_t size = 0;
        uint64_t binding_size = 0;
        uint32_t buffer_id = 0;
        uint64_t offset = 0;
    };
    std::vector<Relocation> relocations;
    uint64_t relocation_epoch = 0;
    // G66 (#691): the bridge's resource retire epoch when the entries were last checked live.
    uint64_t retire_epoch = 0;
    // G66 partly-held mip chains: the bridge's residency epoch when the entries' mip clamps were
    // last checked, and a hash of (texture, clamped base mip) over its texture entries.
    uint64_t residency_epoch = 0;
    uint64_t residency_key = 0;
    // G70 (#691): bridge-interned view ids backing TextureView entries in this bind group, so
    // retiring a texture's interned views immediately invalidates cached bind groups referencing them.
    std::vector<uint32_t> view_ids;
    uint64_t key_hash = 0;

    // Internal validity data.
    std::vector<uint64_t> key;
    std::vector<uint64_t> built_generation;  // per plan table window
    // Memory-backed tables (DescriptorSlotSource): the words of every window read at the build and
    // the GPU handle of each table parameter's base (root parameter index, handle).
    bool from_memory = false;
    std::vector<uint8_t> built_words;
    std::vector<std::pair<uint32_t, uint64_t>> table_bases;
    // Heap-indexed entries: (plan entry index, GPU handle of the slot read, its generation at the
    // build); handle 0 when the index fell outside the heap.
    struct HeapSlot {
        uint32_t entry = 0;
        uint64_t gpu_handle = 0;
        uint64_t generation = 0;
    };
    std::vector<HeapSlot> heap_slots;
};

// Re-reads the memory-backed table entries of `group` (built for `plan`) through `source`: true
// when the windows' words changed since the build (or `force_rebuild` is true), in which case the
// table entries and built_words are rebuilt (other entries are kept). Heaps are looked up again by
// their handles.
bool refresh_table_entries(const DescriptorHeapRegistry& heaps,
                           const DescriptorSlotSource& source,
                           const BindGroupPlan& plan,
                           GpuVaResolver resolver,
                           void* resolver_user,
                           CachedBindGroup* group,
                           bool force_rebuild = false);

struct BindGroupLookup {
    Status status = Status::Ok;
    CachedBindGroup* bind_group = nullptr;  // nullptr when the plan has no entries
    bool rebuilt = false;                   // entries changed: emit CreateBindGroup, set gpu_id
    uint32_t retired_gpu_id = 0;            // previous gpu_id of a rebuilt entry (destroy it)
};

// Compiles one WebGPU bind group from a BindGroupPlan and the current root
// arguments, and caches it. Key = (layout_key, plan hash, per-table (heap
// serial, base index), per-root-view GPU address); root-constant values are not
// part of the key. layout_key is caller-defined (the BindGroupLayout id, or the
// pipeline id while auto-layouts are used). A hit is revalidated against the
// heap generations of exactly the descriptors the plan reads: O(1) when the heap
// did not change, else O(window / kGenerationBlockSize).
class BindGroupCache {
public:
    BindGroupCache() = default;
    ~BindGroupCache();
    BindGroupCache(const BindGroupCache&) = delete;
    BindGroupCache& operator=(const BindGroupCache&) = delete;
    BindGroupLookup get_or_build(
        const RootArgumentState& state,
        BindPoint bind_point,
        const BindGroupPlan& plan,
        uint64_t layout_key,
        GpuVaResolver resolver,
        void* resolver_user);

    // Drops every entry, appending their gpu ids (if non-zero) to `retired`.
    void clear(std::vector<uint32_t>* retired);
    // Drops one entry (a pointer a lookup returned); returns its gpu id (0 when none / not cached).
    uint32_t erase(const CachedBindGroup* group);
    // G70 (#691): immediately evicts every cached entry referencing `resource_id` (directly in
    // entries/views or in relocations) and appends their non-zero gpu ids to `retired`.
    size_t evict_resource(uint32_t resource_id, std::vector<uint32_t>* retired = nullptr);
    // G70 (#691): immediately evicts every cached entry referencing any id in `resource_ids`
    // (entries/views/relocations) or any interned view id in `view_ids`, appending retired gpu ids
    // and optionally the evicted entries' cache key hashes.
    size_t evict_resources_and_views(std::span<const uint32_t> resource_ids,
                                     std::span<const uint32_t> view_ids,
                                     std::vector<uint32_t>* retired = nullptr,
                                     std::vector<uint64_t>* evicted_key_hashes = nullptr);
    // Records every resource, view and relocation buffer id `group` references in the eviction
    // filter. Call it after anything that adds ids to a group the cache holds (the cache does it
    // for its own builds; the bridge after it fills in relocations and view ids). Ids are only
    // ever added between rebuilds, so the filter may over-report but never miss a referenced id.
    void note_group_ids(const CachedBindGroup& group) noexcept;
    // Bounds the cache between recordings (O36). Keys carry root-view GPU addresses and table base
    // indices, which ring allocators change per draw, so without a bound the cache (and the WebGPU
    // bind groups behind it) grows for the whole run. When more than `max_entries` are cached, drops
    // the entries the current epoch did not use, then everything if still above the bound; the gpu
    // ids of dropped entries go to `retired`. Then starts a new epoch. Only call it when no open
    // recording references the entries (command-list Reset). Returns the number dropped.
    size_t end_epoch(size_t max_entries, std::vector<uint32_t>* retired);
    uint64_t epoch() const noexcept { return epoch_; }
    // Tables in heaps the source reads come from descriptor memory (nullptr: registry only).
    void set_slot_source(const DescriptorSlotSource* source) noexcept { slot_source_ = source; }
    const DescriptorSlotSource* slot_source() const noexcept { return slot_source_; }
    size_t size() const noexcept { return size_; }
    uint64_t hits() const noexcept { return hits_; }
    uint64_t misses() const noexcept { return misses_; }
    // Heap-indexed bindings whose index fell outside the bound heap (or no heap was bound, or the
    // byte offset is not a whole descriptor): bound as null, counted.
    uint64_t heap_index_misses() const noexcept { return heap_index_misses_; }
    // CB-indexed bindings (HeapIndexSource) whose index could not be read: no constant buffer, an
    // unreadable one, or byte offset outside it (also counted in heap_index_misses).
    uint64_t cb_index_misses() const noexcept { return cb_index_misses_; }

private:
    struct Slot {
        uint64_t hash = 0;
        std::unique_ptr<CachedBindGroup> entry;
        uint8_t state = 0;  // 0 = Empty, 1 = Occupied, 2 = Tombstone
    };

    Slot* find_slot(uint64_t hash) noexcept;
    void rehash(size_t new_cap);
    void insert_entry(uint64_t hash, std::unique_ptr<CachedBindGroup> entry);

    // Keeps the process-wide live-entry gauge equal to size_ after a mutation.
    void track_size() noexcept;
    size_t tracked_size_ = 0;

    void build_entries(
        const RootArgumentState& state,
        BindPoint bind_point,
        const RootSignature& signature,
        const BindGroupPlan& plan,
        GpuVaResolver resolver,
        void* resolver_user,
        CachedBindGroup* out);

    // Words of the plan's memory-backed windows under the current arguments (false: none).
    bool memory_words(const RootArgumentState& state, BindPoint bind_point, const BindGroupPlan& plan,
                      std::vector<uint8_t>* out) const;

    // Eviction filter: one bit per (hashed) resource id and view id some cached group references.
    // A retire of an id that is not set cannot match any group, which skips the full table scan
    // (a title retires far more resources than its bind groups ever bind).
    static constexpr size_t kIdFilterBits = size_t{1} << 16;
    static size_t id_filter_bit(uint32_t id, bool view) noexcept {
        return static_cast<size_t>(((id + (view ? 0x5bd1e995u : 0u)) * 0x9E3779B1u) >> 16) & (kIdFilterBits - 1);
    }
    bool filter_may_hold(uint32_t id, bool view) const noexcept {
        const size_t b = id_filter_bit(id, view);
        return ((id_filter_[b >> 6] >> (b & 63)) & 1u) != 0;
    }
    void rebuild_id_filter() noexcept;
    std::vector<uint64_t> id_filter_ = std::vector<uint64_t>(kIdFilterBits / 64, 0);
    size_t filter_notes_ = 0;  // note_group_ids calls since the filter was last rebuilt

    std::vector<Slot> table_;
    size_t size_ = 0;
    size_t tombstones_ = 0;
    uint64_t last_hash_ = 0;
    CachedBindGroup* last_entry_ = nullptr;
    std::vector<uint64_t> scratch_key_;
    std::vector<uint8_t> scratch_words_;
    mutable std::vector<uint8_t> scratch_window_;
    const DescriptorSlotSource* slot_source_ = nullptr;
    uint64_t epoch_ = 1;
    size_t peak_epoch_entries_ = 0;  // largest size_ in the open epoch
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t heap_index_misses_ = 0;
    uint64_t cb_index_misses_ = 0;
};

// ---------------------------------------------------------------------------
// Resource state tracking
// ---------------------------------------------------------------------------

// D3D12CalcSubresource.
inline constexpr uint32_t calc_subresource(
    uint32_t mip_slice,
    uint32_t array_slice,
    uint32_t plane_slice,
    uint32_t mip_levels,
    uint32_t array_size) noexcept {
    return mip_slice + array_slice * mip_levels + plane_slice * mip_levels * array_size;
}

// D3D12_RESOURCE_BARRIER, flattened.
struct ResourceBarrier {
    ResourceBarrierType type = ResourceBarrierType::Transition;
    uint32_t flags = ResourceBarrierFlags::None;
    uint32_t resource_id = 0;       // Transition / UAV (0 = all UAV accesses)
    uint32_t subresource = kAllSubresources;
    uint32_t state_before = ResourceStates::Common;
    uint32_t state_after = ResourceStates::Common;
    uint32_t resource_before = 0;   // Aliasing
    uint32_t resource_after = 0;    // Aliasing

    static ResourceBarrier transition(
        uint32_t resource_id,
        uint32_t before,
        uint32_t after,
        uint32_t subresource = kAllSubresources,
        uint32_t flags = ResourceBarrierFlags::None) noexcept {
        ResourceBarrier b;
        b.type = ResourceBarrierType::Transition;
        b.flags = flags;
        b.resource_id = resource_id;
        b.subresource = subresource;
        b.state_before = before;
        b.state_after = after;
        return b;
    }
    static ResourceBarrier uav(uint32_t resource_id) noexcept {
        ResourceBarrier b;
        b.type = ResourceBarrierType::Uav;
        b.resource_id = resource_id;
        return b;
    }
    static ResourceBarrier aliasing(uint32_t before, uint32_t after) noexcept {
        ResourceBarrier b;
        b.type = ResourceBarrierType::Aliasing;
        b.resource_before = before;
        b.resource_after = after;
        return b;
    }
};

struct ResourceStateDesc {
    uint32_t subresource_count = 1;
    uint32_t initial_state = ResourceStates::Common;
    // Buffers and simultaneous-access textures: implicit promotion from COMMON on
    // first use and decay to COMMON at ExecuteCommandLists.
    bool implicit_promotion = false;
};

enum class StateDiagnosticCode : uint32_t {
    BeforeStateMismatch = 0,      // barrier StateBefore != tracked state
    UnknownResource,              // resource id never registered
    SubresourceOutOfRange,
    InvalidStateCombination,      // write state combined with other bits
    RedundantTransition,          // StateBefore == StateAfter
    InvalidBarrierFlags,          // both BEGIN_ONLY and END_ONLY, or flags on UAV/aliasing barriers
    SplitBarrierAlreadyBegun,
    SplitBarrierWithoutBegin,     // END_ONLY on a subresource with no matching BEGIN_ONLY
    SplitBarrierMismatch,         // END_ONLY after-state differs from BEGIN_ONLY
    UseDuringSplitBarrier,        // access/transition while a split barrier is in flight
    AccessStateMismatch,          // note_access: tracked state lacks the required bits
    ExecuteStateMismatch,         // list expected state differs from global state at execute
    InvalidBarrierType,
};

const char* state_diagnostic_name(StateDiagnosticCode code) noexcept;

struct StateDiagnostic {
    StateDiagnosticCode code = StateDiagnosticCode::BeforeStateMismatch;
    uint32_t resource_id = 0;
    uint32_t subresource = kAllSubresources;
    uint32_t expected_state = 0;  // tracked state (or list expectation at execute)
    uint32_t actual_state = 0;    // state named by the barrier / access (or global state)
    uint32_t barrier_index = kNoIndex;
};

class CommandListStateTracker;

// Device-level ("global", queue-visible) resource states.
class ResourceStateRegistry {
public:
    Status register_resource(uint32_t resource_id, const ResourceStateDesc& desc);
    void unregister_resource(uint32_t resource_id);
    bool contains(uint32_t resource_id) const;
    // kStateUnknown if unknown resource / subresource.
    uint32_t state(uint32_t resource_id, uint32_t subresource) const;
    // Returns false if unknown.
    bool describe(uint32_t resource_id, uint32_t* subresource_count, bool* implicit_promotion) const;

    // ExecuteCommandLists for one list: checks the list's expected initial
    // states against the global states (diagnostics only; implicit promotion
    // from COMMON is accepted), then commits the list's final states, applying
    // state decay for implicit-promotion resources. Diagnostics are appended.
    void execute(const CommandListStateTracker& list, std::vector<StateDiagnostic>* diagnostics);

private:
    struct GlobalResource {
        uint32_t subresource_count = 1;
        bool implicit_promotion = false;
        uint32_t uniform_state = 0;
        std::vector<uint32_t> states;  // empty while uniform
    };
    mutable std::shared_mutex mutex_;
    std::unordered_map<uint32_t, GlobalResource> resources_;
};

// Per-command-list resource state tracker. States start unknown; the first
// barrier/access per subresource records the state the list expects at
// execution time, resolved by ResourceStateRegistry::execute.
class CommandListStateTracker {
public:
    explicit CommandListStateTracker(const ResourceStateRegistry* registry) noexcept : registry_(registry) {}

    void reset();
    void resource_barrier(uint32_t count, const ResourceBarrier* barriers);
    // Records that the list accesses a subresource in `required_states` (e.g.
    // PIXEL_SHADER_RESOURCE for an SRV bound at a draw). Applies implicit promotion.
    void note_access(uint32_t resource_id, uint32_t subresource, uint32_t required_states);

    // Local state (kStateUnknown if not touched in this list or if unknown resource).
    uint32_t state(uint32_t resource_id, uint32_t subresource) const;
    // State the list expects at execute (kStateUnknown if none).
    uint32_t expected_initial_state(uint32_t resource_id, uint32_t subresource) const;
    bool in_split_barrier(uint32_t resource_id, uint32_t subresource) const;
    // True while the resource keeps a single state for all subresources.
    bool is_uniform(uint32_t resource_id) const;

    const std::vector<StateDiagnostic>& diagnostics() const noexcept { return diagnostics_; }
    uint64_t dropped_diagnostics() const noexcept { return dropped_diagnostics_; }
    uint64_t uav_barrier_count() const noexcept { return uav_barriers_; }
    uint64_t aliasing_barrier_count() const noexcept { return aliasing_barriers_; }

    static constexpr size_t kMaxDiagnostics = 4096;

private:
    friend class ResourceStateRegistry;
    struct SubState {
        uint32_t state = kStateUnknown;
        uint32_t expected = kStateUnknown;  // state required at execute
        uint32_t split_after = kStateUnknown;
        uint32_t expected_is_access = 0;  // 1: expected only needs to be a subset (access)
        bool operator==(const SubState&) const = default;
    };
    struct LocalResource {
        uint32_t id = 0;
        uint32_t subresource_count = 1;
        bool implicit_promotion = false;
        bool uniform = true;
        SubState all{};
        std::vector<SubState> subs;  // used when !uniform
    };

    LocalResource* touch(uint32_t resource_id, uint32_t barrier_index);
    const LocalResource* find(uint32_t resource_id) const;
    const SubState* sub_state(uint32_t resource_id, uint32_t subresource) const;
    void transition_sub(LocalResource& r, SubState& s, uint32_t sub, const ResourceBarrier& b, uint32_t index);
    void access_sub(LocalResource& r, SubState& s, uint32_t sub, uint32_t required);
    void maybe_collapse(LocalResource& r);
    void diag(StateDiagnosticCode code, uint32_t id, uint32_t sub, uint32_t expected, uint32_t actual, uint32_t index);

    struct IndexSlot {
        uint32_t key = 0;
        uint32_t value = kNoIndex;
    };

    uint32_t index_find(uint32_t key) const noexcept;
    void index_insert(uint32_t key, uint32_t value);

    const ResourceStateRegistry* registry_ = nullptr;
    std::vector<LocalResource> resources_;
    std::vector<IndexSlot> index_table_;
    size_t index_size_ = 0;
    std::vector<StateDiagnostic> diagnostics_;
    uint64_t dropped_diagnostics_ = 0;
    uint64_t uav_barriers_ = 0;
    uint64_t aliasing_barriers_ = 0;
};

// ---------------------------------------------------------------------------
// WebGPU usage-scope conflict query
// ---------------------------------------------------------------------------

// WebGPU forbids using a subresource both as a writable attachment/storage and
// in any other way inside one usage scope (render pass draw / dispatch).
enum class ResourceUsage : uint8_t {
    Sampled = 0,        // SRV
    StorageRead = 1,    // read-only storage
    StorageWrite = 2,   // UAV (may alias other StorageWrite)
    RenderTarget = 3,   // RTV
    DepthStencilWrite = 4,
    DepthStencilRead = 5,  // read-only DSV
};

inline constexpr uint32_t kAllMips = 0xffffffffu;

struct SubresourceRange {
    uint32_t first_mip = 0;
    uint32_t mip_count = kAllMips;
    uint32_t first_slice = 0;
    uint32_t slice_count = kAllMips;
    uint32_t first_plane = 0;
    uint32_t plane_count = kAllMips;
};

// Subresource range addressed by a view descriptor (SRV/UAV/RTV/DSV).
SubresourceRange view_subresource_range(const ViewDescriptor& view) noexcept;
// Usage implied by a view descriptor (SRV -> Sampled, UAV -> StorageWrite,
// RTV -> RenderTarget, DSV -> DepthStencilWrite or DepthStencilRead if read-only depth).
ResourceUsage view_usage(const ViewDescriptor& view) noexcept;

struct UsageConflict {
    uint32_t resource_id = 0;
    ResourceUsage first = ResourceUsage::Sampled;
    ResourceUsage second = ResourceUsage::Sampled;
    uint32_t first_index = 0;  // index of the add() call
    uint32_t second_index = 0;
};

class UsageScope {
public:
    void clear() noexcept { usages_.clear(); }
    void add(uint32_t resource_id, const SubresourceRange& range, ResourceUsage usage);
    // Appends every conflicting pair; returns the number found.
    size_t find_conflicts(std::vector<UsageConflict>* out);

private:
    struct Usage {
        uint32_t resource_id;
        uint32_t index;
        SubresourceRange range;
        ResourceUsage usage;
    };
    std::vector<Usage> usages_;
    std::vector<Usage> sorted_;
};

}  // namespace playgta6::gpu::d3d12_state
