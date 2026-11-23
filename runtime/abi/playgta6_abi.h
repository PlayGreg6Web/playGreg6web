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
// 416 bytes, little endian, one per guest thread. Every lifted function is
// `void sub_<VA>(GuestState* state)`; see docs/wasm_abi.md for the calling convention.
// The offsets below are checked against the lifter's STATE_* constants by
// recomp/tests/guest_state_abi.rs. New fields are appended after `xmm` (bump kGuestStateSize).
struct GuestState {
    uint64_t gpr[16];     // rax rcx rdx rbx rsp rbp rsi rdi r8..r15 (x86 encoding order)
    uint64_t rip;         // reserved for the runtime; lifted code does not maintain it
    uint8_t flags[8];     // CF PF AF ZF SF OF DF pad, one byte each, always 0 or 1
    uint64_t fs_base;     // `fs` segment base
    uint64_t gs_base;     // `gs` segment base (TEB on Win64)
    uint64_t xmm[16][2];  // xmm0..xmm15, low qword first
};

inline constexpr uint64_t kGuestStateGprOffset = 0;
inline constexpr uint64_t kGuestStateRipOffset = 128;
inline constexpr uint64_t kGuestStateFlagsOffset = 136;
inline constexpr uint64_t kGuestStateFsBaseOffset = 144;
inline constexpr uint64_t kGuestStateGsBaseOffset = 152;
inline constexpr uint64_t kGuestStateXmmOffset = 160;
inline constexpr uint64_t kGuestStateXmmStride = 16;
inline constexpr uint64_t kGuestStateSize = 416;
// Fifth Win64 argument relative to `rsp` at callee entry: return address + 32-byte shadow space.
inline constexpr uint64_t kGuestStackArgOffset = 40;

static_assert(sizeof(GuestState) == kGuestStateSize, "GuestState layout must stay 416 bytes");
static_assert(alignof(GuestState) == 8, "GuestState must stay 8-byte aligned");
static_assert(offsetof(GuestState, gpr) == kGuestStateGprOffset, "GuestState::gpr offset");
static_assert(offsetof(GuestState, rip) == kGuestStateRipOffset, "GuestState::rip offset");
static_assert(offsetof(GuestState, flags) == kGuestStateFlagsOffset, "GuestState::flags offset");
static_assert(offsetof(GuestState, fs_base) == kGuestStateFsBaseOffset, "GuestState::fs_base offset");
static_assert(offsetof(GuestState, gs_base) == kGuestStateGsBaseOffset, "GuestState::gs_base offset");
static_assert(offsetof(GuestState, xmm) == kGuestStateXmmOffset, "GuestState::xmm offset");
static_assert(sizeof(GuestState::xmm) == 16 * kGuestStateXmmStride, "GuestState::xmm stride");

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

}  // namespace playgta6

// Runtime helpers called by lifted code (implemented by the runtime, not by JS).
extern "C" {
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
}  // extern "C"
