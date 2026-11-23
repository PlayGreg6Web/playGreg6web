#include "gpu/cmd_encoder.h"

#include <cstring>

namespace playgta6::gpu {

void CommandEncoder::clear() {
    buf_.clear();
}

void CommandEncoder::write_packet(GpuOpcode opcode, const void* payload, uint32_t payload_bytes) {
    write_packet_with_tail(opcode, payload, payload_bytes, nullptr, 0);
}

void CommandEncoder::write_packet_with_tail(
    GpuOpcode opcode,
    const void* head,
    uint32_t head_bytes,
    const void* tail,
    uint32_t tail_bytes) {
    const uint32_t safe_head_bytes = (head != nullptr) ? head_bytes : 0u;
    const uint32_t safe_tail_bytes = (tail != nullptr) ? tail_bytes : 0u;
    const uint32_t payload_bytes = safe_head_bytes + safe_tail_bytes;

    const GpuPacketHeader header{
        static_cast<uint16_t>(opcode),
        0,
        payload_bytes,
    };

    const size_t offset = buf_.size();
    buf_.resize(offset + sizeof(GpuPacketHeader) + payload_bytes);
    std::memcpy(buf_.data() + offset, &header, sizeof(GpuPacketHeader));
    if (safe_head_bytes > 0) {
        std::memcpy(buf_.data() + offset + sizeof(GpuPacketHeader), head, safe_head_bytes);
    }
    if (safe_tail_bytes > 0) {
        std::memcpy(
            buf_.data() + offset + sizeof(GpuPacketHeader) + safe_head_bytes,
            tail,
            safe_tail_bytes);
    }
}

void CommandEncoder::nop() {
    write_packet(GpuOpcode::Nop, nullptr, 0);
}

void CommandEncoder::begin_frame() {
    write_packet(GpuOpcode::BeginFrame, nullptr, 0);
}

void CommandEncoder::clear_color(float r, float g, float b, float a) {
    const GpuClearColorPayload payload{r, g, b, a};
    write_packet(GpuOpcode::ClearColor, &payload, sizeof(payload));
}

void CommandEncoder::create_buffer(uint32_t id, uint32_t usage_flags, uint64_t byte_size) {
    const GpuCreateBufferPayload payload{id, usage_flags, byte_size};
    write_packet(GpuOpcode::CreateBuffer, &payload, sizeof(payload));
}

void CommandEncoder::write_buffer(
    uint32_t id,
    uint32_t dst_offset,
    const void* data,
    uint32_t data_len) {
    const uint32_t actual_len = (data != nullptr) ? data_len : 0u;
    const GpuWriteBufferPayload payload{id, dst_offset, actual_len, 0u};
    write_packet_with_tail(GpuOpcode::WriteBuffer, &payload, sizeof(payload), data, actual_len);
}

void CommandEncoder::create_shader_module(
    uint32_t id,
    GpuShaderStage stage,
    std::string_view wgsl_code) {
    const auto code_len = static_cast<uint32_t>(wgsl_code.size());
    const GpuCreateShaderModulePayload payload{
        id,
        static_cast<uint32_t>(stage),
        code_len,
        0u,
    };
    write_packet_with_tail(
        GpuOpcode::CreateShaderModule,
        &payload,
        sizeof(payload),
        wgsl_code.data(),
        code_len);
}

void CommandEncoder::create_render_pipeline(
    uint32_t id,
    uint32_t vs_shader_id,
    uint32_t fs_shader_id,
    uint32_t vertex_stride,
    GpuPrimitiveTopology topology,
    uint32_t flags) {
    const GpuCreateRenderPipelinePayload payload{
        id,
        vs_shader_id,
        fs_shader_id,
        vertex_stride,
        static_cast<uint32_t>(topology),
        flags,
    };
    write_packet(GpuOpcode::CreateRenderPipeline, &payload, sizeof(payload));
}

void CommandEncoder::set_pipeline(uint32_t pipeline_id) {
    const GpuSetPipelinePayload payload{pipeline_id};
    write_packet(GpuOpcode::SetPipeline, &payload, sizeof(payload));
}

void CommandEncoder::set_vertex_buffer(uint32_t slot, uint32_t buffer_id, uint64_t offset) {
    const GpuSetVertexBufferPayload payload{slot, buffer_id, offset};
    write_packet(GpuOpcode::SetVertexBuffer, &payload, sizeof(payload));
}

void CommandEncoder::draw(
    uint32_t vertex_count,
    uint32_t instance_count,
    uint32_t first_vertex,
    uint32_t first_instance) {
    const GpuDrawPayload payload{
        vertex_count,
        instance_count,
        first_vertex,
        first_instance,
    };
    write_packet(GpuOpcode::Draw, &payload, sizeof(payload));
}

void CommandEncoder::present() {
    write_packet(GpuOpcode::Present, nullptr, 0);
}

void CommandEncoder::destroy_resource(uint32_t id) {
    const GpuDestroyResourcePayload payload{id};
    write_packet(GpuOpcode::DestroyResource, &payload, sizeof(payload));
}

}  // namespace playgta6::gpu
