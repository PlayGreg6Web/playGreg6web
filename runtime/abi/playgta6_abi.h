#pragma once

#include <cstddef>
#include <cstdint>

namespace playgta6 {

inline constexpr uint32_t kAbiVersion = 1;

// Shared input snapshot written by web/input/input_bridge.js into a SharedArrayBuffer.
struct InputState {
    uint32_t seq;
    uint32_t buttons;
    int32_t mouse_dx;
    int32_t mouse_dy;
    float left_stick_x;
    float left_stick_y;
    float right_stick_x;
    float right_stick_y;
    float left_trigger;
    float right_trigger;
};
static_assert(sizeof(InputState) == 40, "InputState layout must stay 40 bytes");

// Opcodes encoded into the GPU command buffer (runtime/gpu -> web/workers/wgpu_worker.js).
enum class GpuOpcode : uint16_t {
    Nop = 0,
    BeginFrame = 1,
    ClearColor = 2,
    CreateBuffer = 3,
    WriteBuffer = 4,
    CreateShaderModule = 5,
    Draw = 6,
    Present = 7,
    CreateRenderPipeline = 8,
    SetPipeline = 9,
    SetVertexBuffer = 10,
    DestroyResource = 11,
    CreateTexture = 12,
    WriteTexture = 13,
    CreateSampler = 14,
    CreateBindGroup = 15,
    SetBindGroup = 16,
    SetIndexBuffer = 17,
    DrawIndexed = 18,
    CreateComputePipeline = 19,
    SetComputePipeline = 20,
    DispatchCompute = 21,
    // v2 (docs/gpu_commands.md "Opcodes (v2)"): explicit layouts, pipeline keys, render passes and
    // dynamic state. Additive only; every v1 opcode and payload above keeps its meaning.
    CreateBindGroupLayout = 22,
    CreatePipelineLayout = 23,
    CreateRenderPipeline2 = 24,
    CreateComputePipeline2 = 25,
    BeginRenderPass = 26,
    EndRenderPass = 27,
    SetViewport = 28,
    SetScissorRect = 29,
    SetBlendConstant = 30,
    SetStencilReference = 31,
    CreateBindGroup2 = 32,
    SetBindGroup2 = 33,
    CreateSampler2 = 34,
    // v2 texture creation: full texture format codes, sample count and view formats.
    CreateTexture2 = 35,
};

// WebGPU-compatible GPUBufferUsage bitmask values for GpuCreateBufferPayload::usage_flags.
namespace GpuBufferUsage {
inline constexpr uint32_t MapRead = 0x0001u;
inline constexpr uint32_t MapWrite = 0x0002u;
inline constexpr uint32_t CopySrc = 0x0004u;
inline constexpr uint32_t CopyDst = 0x0008u;
inline constexpr uint32_t Index = 0x0010u;
inline constexpr uint32_t Vertex = 0x0020u;
inline constexpr uint32_t Uniform = 0x0040u;
inline constexpr uint32_t Storage = 0x0080u;
inline constexpr uint32_t Indirect = 0x0100u;
inline constexpr uint32_t QueryResolve = 0x0200u;
}  // namespace GpuBufferUsage

// WebGPU-compatible GPUTextureUsage bitmask values for GpuCreateTexturePayload::usage_flags.
namespace GpuTextureUsage {
inline constexpr uint32_t CopySrc = 0x0001u;
inline constexpr uint32_t CopyDst = 0x0002u;
inline constexpr uint32_t TextureBinding = 0x0004u;
inline constexpr uint32_t StorageBinding = 0x0008u;
inline constexpr uint32_t RenderAttachment = 0x0010u;
}  // namespace GpuTextureUsage

enum class GpuTextureFormat : uint32_t {
    Rgba8Unorm = 0,
    Bgra8Unorm = 1,
    Rgba16Float = 2,
    Depth24Plus = 3,
    Depth32Float = 4,
    Bc7RgbaUnorm = 5,
};

enum class GpuTextureDimension : uint32_t {
    Dim1D = 0,
    Dim2D = 1,
    Dim3D = 2,
};

enum class GpuAddressMode : uint32_t {
    ClampToEdge = 0,
    Repeat = 1,
    MirrorRepeat = 2,
};

enum class GpuFilterMode : uint32_t {
    Nearest = 0,
    Linear = 1,
};

enum class GpuBindResourceType : uint32_t {
    Buffer = 0,
    TextureView = 1,
    Sampler = 2,
};

enum class GpuIndexFormat : uint32_t {
    Uint16 = 0,
    Uint32 = 1,
};

enum class GpuShaderStage : uint32_t {
    Vertex = 0,
    Fragment = 1,
    Compute = 2,
};

enum class GpuPrimitiveTopology : uint32_t {
    TriangleList = 0,
    TriangleStrip = 1,
    LineList = 2,
    LineStrip = 3,
    PointList = 4,
};

struct GpuPacketHeader {
    uint16_t opcode;
    uint16_t flags;
    uint32_t payload_bytes;
};
static_assert(sizeof(GpuPacketHeader) == 8, "GpuPacketHeader layout must stay 8 bytes");

struct GpuClearColorPayload {
    float r;
    float g;
    float b;
    float a;
};
static_assert(sizeof(GpuClearColorPayload) == 16, "GpuClearColorPayload layout must stay 16 bytes");

struct GpuCreateBufferPayload {
    uint32_t id;
    uint32_t usage_flags;
    uint64_t byte_size;
};
static_assert(sizeof(GpuCreateBufferPayload) == 16, "GpuCreateBufferPayload layout must stay 16 bytes");

// Followed immediately by `data_len` raw bytes in the command stream.
struct GpuWriteBufferPayload {
    uint32_t id;
    uint32_t dst_offset;
    uint32_t data_len;
    uint32_t reserved;
};
static_assert(sizeof(GpuWriteBufferPayload) == 16, "GpuWriteBufferPayload layout must stay 16 bytes");

// Followed immediately by `code_len` UTF-8 WGSL bytes in the command stream.
struct GpuCreateShaderModulePayload {
    uint32_t id;
    uint32_t stage;
    uint32_t code_len;
    uint32_t reserved;
};
static_assert(sizeof(GpuCreateShaderModulePayload) == 16, "GpuCreateShaderModulePayload layout must stay 16 bytes");

struct GpuCreateRenderPipelinePayload {
    uint32_t id;
    uint32_t vs_shader_id;
    uint32_t fs_shader_id;
    uint32_t vertex_stride;
    uint32_t topology;
    uint32_t flags;
};
static_assert(sizeof(GpuCreateRenderPipelinePayload) == 24, "GpuCreateRenderPipelinePayload layout must stay 24 bytes");

struct GpuSetPipelinePayload {
    uint32_t pipeline_id;
};
static_assert(sizeof(GpuSetPipelinePayload) == 4, "GpuSetPipelinePayload layout must stay 4 bytes");

struct GpuSetVertexBufferPayload {
    uint32_t slot;
    uint32_t buffer_id;
    uint64_t offset;
};
static_assert(sizeof(GpuSetVertexBufferPayload) == 16, "GpuSetVertexBufferPayload layout must stay 16 bytes");

struct GpuDrawPayload {
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
};
static_assert(sizeof(GpuDrawPayload) == 16, "GpuDrawPayload layout must stay 16 bytes");

struct GpuDestroyResourcePayload {
    uint32_t id;
};
static_assert(sizeof(GpuDestroyResourcePayload) == 4, "GpuDestroyResourcePayload layout must stay 4 bytes");

struct GpuCreateTexturePayload {
    uint32_t id;
    uint32_t width;
    uint32_t height;
    uint32_t depth_or_layers;
    uint32_t mip_levels;
    uint32_t format;
    uint32_t usage_flags;
    uint32_t dimension;
};
static_assert(sizeof(GpuCreateTexturePayload) == 32, "GpuCreateTexturePayload layout must stay 32 bytes");

// Followed immediately by `data_len` raw texel bytes in the command stream.
struct GpuWriteTexturePayload {
    uint32_t texture_id;
    uint32_t mip_level;
    uint32_t origin_x;
    uint32_t origin_y;
    uint32_t origin_z;
    uint32_t width;
    uint32_t height;
    uint32_t depth_or_layers;
    uint32_t bytes_per_row;
    uint32_t rows_per_image;
    uint32_t data_len;
    uint32_t reserved;
};
static_assert(sizeof(GpuWriteTexturePayload) == 48, "GpuWriteTexturePayload layout must stay 48 bytes");

struct GpuCreateSamplerPayload {
    uint32_t id;
    uint32_t address_u;
    uint32_t address_v;
    uint32_t address_w;
    uint32_t mag_filter;
    uint32_t min_filter;
    uint32_t mipmap_filter;
    uint32_t max_anisotropy;
};
static_assert(sizeof(GpuCreateSamplerPayload) == 32, "GpuCreateSamplerPayload layout must stay 32 bytes");

struct GpuBindGroupEntry {
    uint32_t binding;
    uint32_t resource_type;
    uint32_t resource_id;
    uint32_t reserved;
    uint64_t offset;
    uint64_t size;
};
static_assert(sizeof(GpuBindGroupEntry) == 32, "GpuBindGroupEntry layout must stay 32 bytes");

// Followed immediately by `entry_count` packed GpuBindGroupEntry structs in the command stream.
struct GpuCreateBindGroupPayload {
    uint32_t id;
    uint32_t pipeline_id;
    uint32_t group_index;
    uint32_t entry_count;
};
static_assert(sizeof(GpuCreateBindGroupPayload) == 16, "GpuCreateBindGroupPayload layout must stay 16 bytes");

struct GpuSetBindGroupPayload {
    uint32_t group_index;
    uint32_t bind_group_id;
};
static_assert(sizeof(GpuSetBindGroupPayload) == 8, "GpuSetBindGroupPayload layout must stay 8 bytes");

struct GpuSetIndexBufferPayload {
    uint32_t buffer_id;
    uint32_t index_format;
    uint64_t offset;
};
static_assert(sizeof(GpuSetIndexBufferPayload) == 16, "GpuSetIndexBufferPayload layout must stay 16 bytes");

struct GpuDrawIndexedPayload {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t base_vertex;
    uint32_t first_instance;
};
static_assert(sizeof(GpuDrawIndexedPayload) == 20, "GpuDrawIndexedPayload layout must stay 20 bytes");

struct GpuCreateComputePipelinePayload {
    uint32_t id;
    uint32_t cs_shader_id;
    uint32_t flags;
    uint32_t reserved;
};
static_assert(sizeof(GpuCreateComputePipelinePayload) == 16, "GpuCreateComputePipelinePayload layout must stay 16 bytes");

struct GpuSetComputePipelinePayload {
    uint32_t pipeline_id;
};
static_assert(sizeof(GpuSetComputePipelinePayload) == 4, "GpuSetComputePipelinePayload layout must stay 4 bytes");

struct GpuDispatchComputePayload {
    uint32_t workgroup_x;
    uint32_t workgroup_y;
    uint32_t workgroup_z;
};
static_assert(sizeof(GpuDispatchComputePayload) == 12, "GpuDispatchComputePayload layout must stay 12 bytes");

// Guest x86-64 CPU state of recompiled code (recomp/src/lift, `%playgta6.GuestState`).
// 704 bytes, little endian, one per guest thread. Every lifted function is
// `void sub_<VA>(GuestState* state)`; see docs/wasm_abi.md for the calling convention.
// The offsets below are checked against the lifter's STATE_* constants by
// recomp/tests/guest_state_abi.rs. New fields are appended at the end (bump kGuestStateSize);
// existing offsets never change.
struct GuestState {
    uint64_t gpr[16];     // rax rcx rdx rbx rsp rbp rsi rdi r8..r15 (x86 encoding order)
    uint64_t rip;         // reserved for the runtime; lifted code does not maintain it
    uint8_t flags[8];     // CF PF AF ZF SF OF DF pad, one byte each, always 0 or 1
    uint64_t fs_base;     // `fs` segment base
    uint64_t gs_base;     // `gs` segment base (TEB on Win64)
    uint64_t xmm[16][2];     // xmm0..xmm15 (bits 0..127 of ymm0..ymm15), low qword first
    uint64_t ymm_hi[16][2];  // bits 128..255 of ymm0..ymm15, low qword first
    uint32_t mxcsr;          // SSE control/status register; initialise to kGuestMxcsrDefault
    uint32_t reserved[3];    // zero; keeps the size a multiple of 16
    // Pending context restore (exception unwind, longjmp; runtime/guest/guest_exceptions.h). While
    // `unwind_pending` is nonzero, the registers above hold the restored context and every lifted
    // activation returns to its caller without touching the state, until the activation that owns
    // the restored `rsp` clears the flag and continues at `resume_rip`.
    uint64_t resume_rip;     // guest address execution continues at
    uint8_t unwind_pending;  // 0 or 1
    uint8_t reserved2[7];    // zero; keeps the size a multiple of 16
};

inline constexpr uint64_t kGuestStateGprOffset = 0;
inline constexpr uint64_t kGuestStateRipOffset = 128;
inline constexpr uint64_t kGuestStateFlagsOffset = 136;
inline constexpr uint64_t kGuestStateFsBaseOffset = 144;
inline constexpr uint64_t kGuestStateGsBaseOffset = 152;
inline constexpr uint64_t kGuestStateXmmOffset = 160;
inline constexpr uint64_t kGuestStateXmmStride = 16;
inline constexpr uint64_t kGuestStateYmmHiOffset = 416;
inline constexpr uint64_t kGuestStateYmmHiStride = 16;
inline constexpr uint64_t kGuestStateMxcsrOffset = 672;
inline constexpr uint64_t kGuestStateResumeRipOffset = 688;
inline constexpr uint64_t kGuestStateUnwindPendingOffset = 696;
inline constexpr uint64_t kGuestStateSize = 704;
// Power-on / new-thread MXCSR (round to nearest, all exceptions masked, FTZ/DAZ off).
inline constexpr uint32_t kGuestMxcsrDefault = 0x1F80;
// Fifth Win64 argument relative to `rsp` at callee entry: return address + 32-byte shadow space.
inline constexpr uint64_t kGuestStackArgOffset = 40;

static_assert(sizeof(GuestState) == kGuestStateSize, "GuestState layout must stay 704 bytes");
static_assert(alignof(GuestState) == 8, "GuestState must stay 8-byte aligned");
static_assert(offsetof(GuestState, gpr) == kGuestStateGprOffset, "GuestState::gpr offset");
static_assert(offsetof(GuestState, rip) == kGuestStateRipOffset, "GuestState::rip offset");
static_assert(offsetof(GuestState, flags) == kGuestStateFlagsOffset, "GuestState::flags offset");
static_assert(offsetof(GuestState, fs_base) == kGuestStateFsBaseOffset, "GuestState::fs_base offset");
static_assert(offsetof(GuestState, gs_base) == kGuestStateGsBaseOffset, "GuestState::gs_base offset");
static_assert(offsetof(GuestState, xmm) == kGuestStateXmmOffset, "GuestState::xmm offset");
static_assert(sizeof(GuestState::xmm) == 16 * kGuestStateXmmStride, "GuestState::xmm stride");
static_assert(offsetof(GuestState, ymm_hi) == kGuestStateYmmHiOffset, "GuestState::ymm_hi offset");
static_assert(sizeof(GuestState::ymm_hi) == 16 * kGuestStateYmmHiStride, "GuestState::ymm_hi stride");
static_assert(offsetof(GuestState, mxcsr) == kGuestStateMxcsrOffset, "GuestState::mxcsr offset");
static_assert(offsetof(GuestState, resume_rip) == kGuestStateResumeRipOffset,
              "GuestState::resume_rip offset");
static_assert(offsetof(GuestState, unwind_pending) == kGuestStateUnwindPendingOffset,
              "GuestState::unwind_pending offset");

// Indices into GuestState::gpr (x86 register encoding).
namespace GuestReg {
inline constexpr uint32_t Rax = 0;
inline constexpr uint32_t Rcx = 1;
inline constexpr uint32_t Rdx = 2;
inline constexpr uint32_t Rbx = 3;
inline constexpr uint32_t Rsp = 4;
inline constexpr uint32_t Rbp = 5;
inline constexpr uint32_t Rsi = 6;
inline constexpr uint32_t Rdi = 7;
inline constexpr uint32_t R8 = 8;
inline constexpr uint32_t R9 = 9;
inline constexpr uint32_t R10 = 10;
inline constexpr uint32_t R11 = 11;
inline constexpr uint32_t R12 = 12;
inline constexpr uint32_t R13 = 13;
inline constexpr uint32_t R14 = 14;
inline constexpr uint32_t R15 = 15;
}  // namespace GuestReg

// Indices into GuestState::flags.
namespace GuestFlag {
inline constexpr uint32_t CF = 0;
inline constexpr uint32_t PF = 1;
inline constexpr uint32_t AF = 2;
inline constexpr uint32_t ZF = 3;
inline constexpr uint32_t SF = 4;
inline constexpr uint32_t OF = 5;
inline constexpr uint32_t DF = 6;
}  // namespace GuestFlag

// A lifted guest function (`sub_<VA>`).
using LiftedFn = void (*)(GuestState* state);

// Guest address space (docs/wasm_abi.md "Guest Address Space"). Guest virtual addresses reach host
// memory through one translation shared by lifted code, the import bridge and the runtime:
//     host address = __playgta6_guest_base + guest VA   (64-bit wrapping add)
// The runtime backs the guest address space with one contiguous host region starting at the guest
// VA below, so guest pointers stay valid across page and allocation boundaries. In the browser
// build the mapping is the identity (__playgta6_guest_base == 0): the guest address space is the
// linear memory from kGuestAddressSpaceBase upwards, the runtime's own memory lies below it.
inline constexpr uint64_t kGuestAddressSpaceBase = 4294967296;  // 0x1'0000'0000

// =====================================================================================================
// Packet ABI v2 (opcodes 22..34). See docs/gpu_commands.md "Opcodes (v2)".
//
// Byte-sized enum fields below that are documented as "WebGPU enum codes" (texture format, vertex
// format, blend factor/operation, compare function, stencil operation, cull mode, front face, index
// format, vertex step mode) use the code tables in docs/gpu_commands.md, which are the numbering of
// the playgta6::gpu::pso::Wgpu* enums in runtime/gpu/pso_cache.h. Primitive topology codes are
// GpuPrimitiveTopology.
// =====================================================================================================

// Texture id that names the canvas texture of the current frame (getCurrentTexture()).
inline constexpr uint32_t kGpuSwapchainTextureId = 0xFFFFFFFFu;

inline constexpr uint32_t kGpuMaxColorAttachments = 8;
inline constexpr uint32_t kGpuMaxVertexBuffers = 8;
inline constexpr uint32_t kGpuMaxVertexAttributes = 16;

enum class GpuLoadOp : uint8_t {
    Undefined = 0,  // only for depth/stencil aspects that are read-only or absent
    Load = 1,
    Clear = 2,
};

enum class GpuStoreOp : uint8_t {
    Undefined = 0,  // only for depth/stencil aspects that are read-only or absent
    Store = 1,
    Discard = 2,
};

enum class GpuTextureAspect : uint8_t {
    All = 0,
    DepthOnly = 1,
    StencilOnly = 2,
};

// GPUTextureViewDimension; Undefined = WebGPU default for the texture.
enum class GpuTextureViewDimension : uint8_t {
    Undefined = 0,
    Dim1D = 1,
    Dim2D = 2,
    Dim2DArray = 3,
    Cube = 4,
    CubeArray = 5,
    Dim3D = 6,
};

// Which GPUBindGroupLayoutEntry member a GpuBindGroupLayoutEntry describes.
enum class GpuBindingLayoutType : uint8_t {
    Buffer = 0,
    Sampler = 1,
    Texture = 2,
    StorageTexture = 3,
};

enum class GpuBufferBindingType : uint8_t {
    Uniform = 0,
    Storage = 1,
    ReadOnlyStorage = 2,
};

enum class GpuSamplerBindingType : uint8_t {
    Filtering = 0,
    NonFiltering = 1,
    Comparison = 2,
};

enum class GpuTextureSampleType : uint8_t {
    Float = 0,
    UnfilterableFloat = 1,
    Depth = 2,
    Sint = 3,
    Uint = 4,
};

enum class GpuStorageTextureAccess : uint8_t {
    WriteOnly = 0,
    ReadOnly = 1,
    ReadWrite = 2,
};

// GPUShaderStage bits for GpuBindGroupLayoutEntry::visibility.
namespace GpuShaderStageBits {
inline constexpr uint32_t Vertex = 0x1u;
inline constexpr uint32_t Fragment = 0x2u;
inline constexpr uint32_t Compute = 0x4u;
}  // namespace GpuShaderStageBits

namespace GpuBindGroupLayoutEntryFlags {
inline constexpr uint8_t HasDynamicOffset = 0x1u;  // buffer entries
inline constexpr uint8_t Multisampled = 0x2u;      // texture entries
}  // namespace GpuBindGroupLayoutEntryFlags

namespace GpuLayoutFlags {
// The object is the one layout-manifest.json lists under `content_id`; the packet carries no entries.
inline constexpr uint32_t ManifestReference = 0x1u;
}  // namespace GpuLayoutFlags

// GPUTextureViewDescriptor subset used by render pass attachments and bind group entries.
struct GpuTextureViewDesc {
    uint8_t format;             // WebGPU texture format code; 0 = the texture's format
    uint8_t dimension;          // GpuTextureViewDimension
    uint8_t aspect;             // GpuTextureAspect
    uint8_t reserved;
    uint16_t base_mip_level;
    uint16_t mip_level_count;   // 0 = all remaining mips
    uint32_t base_array_layer;
    uint32_t array_layer_count; // 0 = all remaining layers
};
static_assert(sizeof(GpuTextureViewDesc) == 16, "GpuTextureViewDesc layout must stay 16 bytes");

// One GPUBindGroupLayoutEntry (same shape as an S6 layout-manifest.json entry).
struct GpuBindGroupLayoutEntry {
    uint32_t binding;
    uint32_t visibility;        // GpuShaderStageBits
    uint8_t type;               // GpuBindingLayoutType
    uint8_t binding_type;       // GpuBufferBindingType / GpuSamplerBindingType / GpuTextureSampleType /
                                // GpuStorageTextureAccess, selected by `type`
    uint8_t view_dimension;     // texture / storageTexture: GpuTextureViewDimension (Undefined = "2d")
    uint8_t flags;              // GpuBindGroupLayoutEntryFlags
    uint8_t storage_format;     // storageTexture: WebGPU texture format code
    uint8_t reserved[3];
    uint64_t min_binding_size;  // buffer: minBindingSize (0 = none)
};
static_assert(sizeof(GpuBindGroupLayoutEntry) == 24, "GpuBindGroupLayoutEntry layout must stay 24 bytes");
static_assert(offsetof(GpuBindGroupLayoutEntry, min_binding_size) == 16);

// Followed immediately by `entry_count` packed GpuBindGroupLayoutEntry structs.
struct GpuCreateBindGroupLayoutPayload {
    uint32_t id;
    uint32_t entry_count;
    uint32_t flags;        // GpuLayoutFlags
    uint32_t reserved;
    uint64_t content_id;   // S6 layout-manifest.json bind group layout id (0 = none)
};
static_assert(sizeof(GpuCreateBindGroupLayoutPayload) == 24, "GpuCreateBindGroupLayoutPayload layout must stay 24 bytes");

// Followed immediately by `bind_group_layout_count` u32 bind group layout ids (group 0..n-1;
// 0 = empty layout).
struct GpuCreatePipelineLayoutPayload {
    uint32_t id;
    uint32_t bind_group_layout_count;
    uint32_t flags;        // GpuLayoutFlags
    uint32_t reserved;
    uint64_t content_id;   // S6 layout-manifest.json pipeline layout id (0 = none)
};
static_assert(sizeof(GpuCreatePipelineLayoutPayload) == 24, "GpuCreatePipelineLayoutPayload layout must stay 24 bytes");

// ---- Render pipeline key: byte-for-byte playgta6::gpu::pso::GraphicsPipelineKey (376 bytes). ----

struct GpuBlendComponentKey {
    uint8_t operation;   // blend operation code
    uint8_t src_factor;  // blend factor code
    uint8_t dst_factor;
    uint8_t reserved;
};
static_assert(sizeof(GpuBlendComponentKey) == 4);

struct GpuColorTargetKey {
    uint8_t format;         // texture format code; 0 = null target (hole), all other fields 0
    uint8_t blend_enabled;  // 0 = GPUColorTargetState.blend omitted
    uint8_t write_mask;     // GPUColorWrite bits
    uint8_t reserved;
    GpuBlendComponentKey color;
    GpuBlendComponentKey alpha;
};
static_assert(sizeof(GpuColorTargetKey) == 12);

struct GpuStencilFaceKey {
    uint8_t compare;        // compare function code
    uint8_t fail_op;        // stencil operation code
    uint8_t depth_fail_op;
    uint8_t pass_op;
};
static_assert(sizeof(GpuStencilFaceKey) == 4);

struct GpuDepthStencilKey {
    uint8_t format;               // texture format code; 0 = no depthStencil state, all other fields 0
    uint8_t depth_write_enabled;
    uint8_t depth_compare;        // compare function code (0 = undefined)
    uint8_t reserved;
    GpuStencilFaceKey stencil_front;
    GpuStencilFaceKey stencil_back;
    uint32_t stencil_read_mask;
    uint32_t stencil_write_mask;
    int32_t depth_bias;
    uint32_t depth_bias_slope_scale_bits;  // IEEE-754 f32 bits
    uint32_t depth_bias_clamp_bits;        // IEEE-754 f32 bits
};
static_assert(sizeof(GpuDepthStencilKey) == 32);

struct GpuPrimitiveKey {
    uint8_t topology;            // GpuPrimitiveTopology
    uint8_t strip_index_format;  // index format code (0 = undefined)
    uint8_t front_face;          // front face code
    uint8_t cull_mode;           // cull mode code
    uint8_t unclipped_depth;
    uint8_t reserved[3];
};
static_assert(sizeof(GpuPrimitiveKey) == 8);

struct GpuMultisampleKey {
    uint32_t count;  // 1 or 4
    uint32_t mask;
    uint8_t alpha_to_coverage_enabled;
    uint8_t reserved[3];
};
static_assert(sizeof(GpuMultisampleKey) == 12);

struct GpuVertexBufferKey {
    uint32_t array_stride;
    uint8_t step_mode;        // vertex step mode code; 0 = unused slot (null buffer layout)
    uint8_t attribute_count;  // attributes of this buffer (consecutive in vertex_attributes)
    uint8_t reserved[2];
};
static_assert(sizeof(GpuVertexBufferKey) == 8);

struct GpuVertexAttributeKey {
    uint32_t offset;
    uint8_t format;           // vertex format code
    uint8_t shader_location;
    uint8_t buffer_index;
    uint8_t reserved;
};
static_assert(sizeof(GpuVertexAttributeKey) == 8);

struct GpuRenderPipelineKey {
    uint64_t vs_shader;           // shader identity (informational; modules travel as ids)
    uint64_t ps_shader;           // 0 = no pixel shader
    uint32_t root_signature_id;   // informational
    uint32_t color_target_count;  // highest non-null target + 1
    GpuColorTargetKey color_targets[kGpuMaxColorAttachments];
    GpuDepthStencilKey depth_stencil;
    GpuPrimitiveKey primitive;
    GpuMultisampleKey multisample;
    uint32_t vertex_buffer_count;     // highest used slot + 1
    uint32_t vertex_attribute_count;  // sorted by (buffer_index, shader_location)
    GpuVertexBufferKey vertex_buffers[kGpuMaxVertexBuffers];
    GpuVertexAttributeKey vertex_attributes[kGpuMaxVertexAttributes];
    uint32_t reserved;
};
static_assert(sizeof(GpuRenderPipelineKey) == 376, "GpuRenderPipelineKey layout must stay 376 bytes");
static_assert(offsetof(GpuRenderPipelineKey, color_targets) == 24);
static_assert(offsetof(GpuRenderPipelineKey, depth_stencil) == 120);
static_assert(offsetof(GpuRenderPipelineKey, primitive) == 152);
static_assert(offsetof(GpuRenderPipelineKey, multisample) == 160);
static_assert(offsetof(GpuRenderPipelineKey, vertex_buffer_count) == 172);
static_assert(offsetof(GpuRenderPipelineKey, vertex_attribute_count) == 176);
static_assert(offsetof(GpuRenderPipelineKey, vertex_buffers) == 180);
static_assert(offsetof(GpuRenderPipelineKey, vertex_attributes) == 244);
static_assert(offsetof(GpuRenderPipelineKey, reserved) == 372);

struct GpuCreateRenderPipeline2Payload {
    uint32_t id;
    uint32_t vs_module_id;        // CreateShaderModule id (entry point vs_main)
    uint32_t fs_module_id;        // CreateShaderModule id (entry point fs_main); 0 = none
    uint32_t pipeline_layout_id;  // CreatePipelineLayout id; 0 = layout "auto"
    GpuRenderPipelineKey key;
};
static_assert(sizeof(GpuCreateRenderPipeline2Payload) == 392, "GpuCreateRenderPipeline2Payload layout must stay 392 bytes");
static_assert(offsetof(GpuCreateRenderPipeline2Payload, key) == 16);

// Byte-for-byte playgta6::gpu::pso::ComputePipelineKey (16 bytes).
struct GpuComputePipelineKey {
    uint64_t cs_shader;           // shader identity (informational)
    uint32_t root_signature_id;   // informational
    uint32_t reserved;
};
static_assert(sizeof(GpuComputePipelineKey) == 16);

struct GpuCreateComputePipeline2Payload {
    uint32_t id;
    uint32_t cs_module_id;        // CreateShaderModule id (entry point cs_main)
    uint32_t pipeline_layout_id;  // 0 = layout "auto"
    uint32_t reserved;
    GpuComputePipelineKey key;
};
static_assert(sizeof(GpuCreateComputePipeline2Payload) == 32, "GpuCreateComputePipeline2Payload layout must stay 32 bytes");

// ---- Render passes ----

struct GpuRenderPassColorAttachment {
    uint32_t texture_id;  // 0 = no attachment at this index; kGpuSwapchainTextureId = canvas
    uint8_t load_op;      // GpuLoadOp (Load or Clear)
    uint8_t store_op;     // GpuStoreOp (Store or Discard)
    uint16_t reserved;
    GpuTextureViewDesc view;
    float clear_value[4];  // RGBA, used when load_op == Clear
};
static_assert(sizeof(GpuRenderPassColorAttachment) == 40, "GpuRenderPassColorAttachment layout must stay 40 bytes");

struct GpuRenderPassDepthStencilAttachment {
    uint32_t texture_id;  // 0 = no depth-stencil attachment
    uint8_t depth_load_op;     // GpuLoadOp; Undefined when depth is read-only or absent
    uint8_t depth_store_op;    // GpuStoreOp; Undefined when depth is read-only or absent
    uint8_t stencil_load_op;   // GpuLoadOp; Undefined when stencil is read-only or absent
    uint8_t stencil_store_op;  // GpuStoreOp; Undefined when stencil is read-only or absent
    GpuTextureViewDesc view;
    float depth_clear_value;
    uint32_t stencil_clear_value;
    uint8_t depth_read_only;
    uint8_t stencil_read_only;
    uint16_t reserved0;
    uint32_t reserved1;
};
static_assert(sizeof(GpuRenderPassDepthStencilAttachment) == 40, "GpuRenderPassDepthStencilAttachment layout must stay 40 bytes");

struct GpuBeginRenderPassPayload {
    uint32_t color_count;  // 0..kGpuMaxColorAttachments; entries >= color_count are ignored
    uint32_t reserved;
    GpuRenderPassColorAttachment color[kGpuMaxColorAttachments];
    GpuRenderPassDepthStencilAttachment depth_stencil;
};
static_assert(sizeof(GpuBeginRenderPassPayload) == 368, "GpuBeginRenderPassPayload layout must stay 368 bytes");
static_assert(offsetof(GpuBeginRenderPassPayload, depth_stencil) == 328);

// ---- Dynamic render pass state ----

struct GpuSetViewportPayload {
    float x;
    float y;
    float width;
    float height;
    float min_depth;
    float max_depth;
};
static_assert(sizeof(GpuSetViewportPayload) == 24, "GpuSetViewportPayload layout must stay 24 bytes");

struct GpuSetScissorRectPayload {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};
static_assert(sizeof(GpuSetScissorRectPayload) == 16, "GpuSetScissorRectPayload layout must stay 16 bytes");

struct GpuSetBlendConstantPayload {
    float r;
    float g;
    float b;
    float a;
};
static_assert(sizeof(GpuSetBlendConstantPayload) == 16, "GpuSetBlendConstantPayload layout must stay 16 bytes");

struct GpuSetStencilReferencePayload {
    uint32_t reference;
};
static_assert(sizeof(GpuSetStencilReferencePayload) == 4, "GpuSetStencilReferencePayload layout must stay 4 bytes");

// ---- Bind groups against explicit layouts ----

// GpuBindGroupEntry (v1, first 32 bytes) + a texture view descriptor for TextureView entries.
struct GpuBindGroupEntry2 {
    uint32_t binding;
    uint32_t resource_type;  // GpuBindResourceType
    uint32_t resource_id;    // 0 = null binding (the worker binds a zero resource matching the layout)
    uint32_t reserved;
    uint64_t offset;         // Buffer
    uint64_t size;           // Buffer; 0 = to the end of the buffer
    GpuTextureViewDesc view; // TextureView; all-zero = default view
};
static_assert(sizeof(GpuBindGroupEntry2) == 48, "GpuBindGroupEntry2 layout must stay 48 bytes");
static_assert(offsetof(GpuBindGroupEntry2, view) == 32);

// Followed immediately by `entry_count` packed GpuBindGroupEntry2 structs.
struct GpuCreateBindGroup2Payload {
    uint32_t id;
    uint32_t layout_id;  // CreateBindGroupLayout id
    uint32_t entry_count;
    uint32_t reserved;
};
static_assert(sizeof(GpuCreateBindGroup2Payload) == 16, "GpuCreateBindGroup2Payload layout must stay 16 bytes");

// Followed immediately by `dynamic_offset_count` u32 dynamic offsets (layout binding order).
struct GpuSetBindGroup2Payload {
    uint32_t group_index;
    uint32_t bind_group_id;
    uint32_t dynamic_offset_count;
    uint32_t reserved;
};
static_assert(sizeof(GpuSetBindGroup2Payload) == 16, "GpuSetBindGroup2Payload layout must stay 16 bytes");

struct GpuCreateSampler2Payload {
    uint32_t id;
    uint8_t address_u;       // GpuAddressMode
    uint8_t address_v;
    uint8_t address_w;
    uint8_t mag_filter;      // GpuFilterMode
    uint8_t min_filter;
    uint8_t mipmap_filter;
    uint8_t compare;         // compare function code; 0 = not a comparison sampler
    uint8_t reserved0;
    uint16_t max_anisotropy; // >= 1
    uint16_t reserved1;
    float lod_min_clamp;
    float lod_max_clamp;
};
static_assert(sizeof(GpuCreateSampler2Payload) == 24, "GpuCreateSampler2Payload layout must stay 24 bytes");

inline constexpr uint32_t kGpuMaxTextureViewFormats = 8;

// CreateTexture2 (opcode 35): a GPUTextureDescriptor. `format` and `view_formats` use the v2 texture
// format codes ("WebGPU enum codes"), not the v1 GpuTextureFormat of CreateTexture.
struct GpuCreateTexture2Payload {
    uint32_t id;
    uint32_t width;
    uint32_t height;
    uint32_t depth_or_layers;    // depth for 3D textures, array layers otherwise
    uint32_t mip_levels;
    uint32_t sample_count;       // 1 or 4
    uint32_t usage_flags;        // GpuTextureUsage
    uint8_t dimension;           // GpuTextureDimension
    uint8_t format;              // v2 texture format code (never 0)
    uint8_t view_format_count;   // <= kGpuMaxTextureViewFormats
    uint8_t reserved;
    uint8_t view_formats[kGpuMaxTextureViewFormats];  // first view_format_count entries used, rest 0
};
static_assert(sizeof(GpuCreateTexture2Payload) == 40, "GpuCreateTexture2Payload layout must stay 40 bytes");

}  // namespace playgta6

// Runtime helpers called by lifted code (implemented by the runtime, not by JS).
extern "C" {
// Guest -> host translation base (see kGuestAddressSpaceBase): host = __playgta6_guest_base + VA.
// Defined by the runtime, set before guest code runs and constant while guest code runs; lifted
// code may load it once per function. 0 means identity (always the case in the browser build).
extern uint64_t __playgta6_guest_base;

// Run the guest code at `target` as if called; the caller already pushed the return address.
// Returns when that code executes its `ret`.
void __playgta6_indirect_call(playgta6::GuestState* state, uint64_t target);
// Guest trap (int3/ud2/__fastfail/hlt), invalid or unmapped code, or a call that was assumed
// never to return, at guest address `va`.
[[noreturn]] void __playgta6_trap(playgta6::GuestState* state, uint64_t va);
// The instruction at `va` could not be lifted.
[[noreturn]] void __playgta6_unimplemented(playgta6::GuestState* state, uint64_t va);
// x86 divide error (#DE) raised by div/idiv at `va`.
[[noreturn]] void __playgta6_divide_error(playgta6::GuestState* state, uint64_t va);
// CPU identification helpers. Lifted code stores its registers to the state before the call and
// reloads them after it; the helper reads/writes only the state's GPRs (32-bit results are stored
// zero-extended into the 64-bit slots).
// cpuid: in EAX = leaf, ECX = sub-leaf; out EAX EBX ECX EDX.
void __playgta6_cpuid(playgta6::GuestState* state);
// rdtsc: out EDX:EAX = time-stamp counter (monotonic per thread).
void __playgta6_rdtsc(playgta6::GuestState* state);
// rdtscp: as rdtsc, plus ECX = IA32_TSC_AUX (processor id).
void __playgta6_rdtscp(playgta6::GuestState* state);
// xgetbv: in ECX = XCR index; out EDX:EAX = XCR[ECX]. An index without an XCR raises #GP (does
// not return).
void __playgta6_xgetbv(playgta6::GuestState* state);
}  // extern "C"
