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

void CommandEncoder::create_texture(
    uint32_t id,
    uint32_t width,
    uint32_t height,
    uint32_t depth_or_layers,
    uint32_t mip_levels,
    GpuTextureFormat format,
    uint32_t usage_flags,
    GpuTextureDimension dimension) {
    const GpuCreateTexturePayload payload{
        id,
        width,
        height,
        depth_or_layers,
        mip_levels,
        static_cast<uint32_t>(format),
        usage_flags,
        static_cast<uint32_t>(dimension),
    };
    write_packet(GpuOpcode::CreateTexture, &payload, sizeof(payload));
}

void CommandEncoder::write_texture(
    uint32_t texture_id,
    uint32_t mip_level,
    uint32_t origin_x,
    uint32_t origin_y,
    uint32_t origin_z,
    uint32_t width,
    uint32_t height,
    uint32_t depth_or_layers,
    uint32_t bytes_per_row,
    uint32_t rows_per_image,
    const void* data,
    uint32_t data_len) {
    const uint32_t actual_len = (data != nullptr) ? data_len : 0u;
    const GpuWriteTexturePayload payload{
        texture_id,
        mip_level,
        origin_x,
        origin_y,
        origin_z,
        width,
        height,
        depth_or_layers,
        bytes_per_row,
        rows_per_image,
        actual_len,
        0u,
    };
    write_packet_with_tail(GpuOpcode::WriteTexture, &payload, sizeof(payload), data, actual_len);
}

void CommandEncoder::create_sampler(
    uint32_t id,
    GpuAddressMode address_u,
    GpuAddressMode address_v,
    GpuAddressMode address_w,
    GpuFilterMode mag_filter,
    GpuFilterMode min_filter,
    GpuFilterMode mipmap_filter,
    uint32_t max_anisotropy) {
    const GpuCreateSamplerPayload payload{
        id,
        static_cast<uint32_t>(address_u),
        static_cast<uint32_t>(address_v),
        static_cast<uint32_t>(address_w),
        static_cast<uint32_t>(mag_filter),
        static_cast<uint32_t>(min_filter),
        static_cast<uint32_t>(mipmap_filter),
        max_anisotropy,
    };
    write_packet(GpuOpcode::CreateSampler, &payload, sizeof(payload));
}

void CommandEncoder::create_bind_group(
    uint32_t id,
    uint32_t pipeline_id,
    uint32_t group_index,
    const GpuBindGroupEntry* entries,
    uint32_t entry_count) {
    const uint32_t actual_count = (entries != nullptr) ? entry_count : 0u;
    const GpuCreateBindGroupPayload payload{
        id,
        pipeline_id,
        group_index,
        actual_count,
    };
    const uint32_t tail_bytes = actual_count * static_cast<uint32_t>(sizeof(GpuBindGroupEntry));
    write_packet_with_tail(GpuOpcode::CreateBindGroup, &payload, sizeof(payload), entries, tail_bytes);
}

void CommandEncoder::set_bind_group(uint32_t group_index, uint32_t bind_group_id) {
    const GpuSetBindGroupPayload payload{group_index, bind_group_id};
    write_packet(GpuOpcode::SetBindGroup, &payload, sizeof(payload));
}

void CommandEncoder::set_index_buffer(
    uint32_t buffer_id,
    GpuIndexFormat index_format,
    uint64_t offset) {
    const GpuSetIndexBufferPayload payload{
        buffer_id,
        static_cast<uint32_t>(index_format),
        offset,
    };
    write_packet(GpuOpcode::SetIndexBuffer, &payload, sizeof(payload));
}

void CommandEncoder::draw_indexed(
    uint32_t index_count,
    uint32_t instance_count,
    uint32_t first_index,
    int32_t base_vertex,
    uint32_t first_instance) {
    const GpuDrawIndexedPayload payload{
        index_count,
        instance_count,
        first_index,
        base_vertex,
        first_instance,
    };
    write_packet(GpuOpcode::DrawIndexed, &payload, sizeof(payload));
}

void CommandEncoder::create_compute_pipeline(uint32_t id, uint32_t cs_shader_id, uint32_t flags) {
    const GpuCreateComputePipelinePayload payload{id, cs_shader_id, flags, 0u};
    write_packet(GpuOpcode::CreateComputePipeline, &payload, sizeof(payload));
}

void CommandEncoder::set_compute_pipeline(uint32_t pipeline_id) {
    const GpuSetComputePipelinePayload payload{pipeline_id};
    write_packet(GpuOpcode::SetComputePipeline, &payload, sizeof(payload));
}

void CommandEncoder::dispatch_compute(
    uint32_t workgroup_x,
    uint32_t workgroup_y,
    uint32_t workgroup_z) {
    const GpuDispatchComputePayload payload{workgroup_x, workgroup_y, workgroup_z};
    write_packet(GpuOpcode::DispatchCompute, &payload, sizeof(payload));
}

void CommandEncoder::emit_packet(
    GpuOpcode opcode,
    const void* head,
    uint32_t head_bytes,
    const void* tail,
    uint32_t tail_bytes) {
    write_packet_with_tail(opcode, head, head_bytes, tail, tail_bytes);
}

}  // namespace playgta6::gpu
