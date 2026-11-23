#pragma once

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

}  // namespace playgta6
