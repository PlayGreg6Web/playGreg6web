#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu {

class CommandEncoder {
public:
    void clear();
    void nop();
    void begin_frame();
    void clear_color(float r, float g, float b, float a);
    void create_buffer(uint32_t id, uint32_t usage_flags, uint64_t byte_size);
    void write_buffer(uint32_t id, uint32_t dst_offset, const void* data, uint32_t data_len);
    void create_shader_module(uint32_t id, GpuShaderStage stage, std::string_view wgsl_code);
    void create_render_pipeline(
        uint32_t id,
        uint32_t vs_shader_id,
        uint32_t fs_shader_id,
        uint32_t vertex_stride,
        GpuPrimitiveTopology topology = GpuPrimitiveTopology::TriangleList,
        uint32_t flags = 0);
    void set_pipeline(uint32_t pipeline_id);
    void set_vertex_buffer(uint32_t slot, uint32_t buffer_id, uint64_t offset = 0);
    void draw(
        uint32_t vertex_count,
        uint32_t instance_count = 1,
        uint32_t first_vertex = 0,
        uint32_t first_instance = 0);
    void present();
    void destroy_resource(uint32_t id);

    const uint8_t* data() const { return buf_.data(); }
    size_t size() const { return buf_.size(); }

private:
    void write_packet(GpuOpcode opcode, const void* payload, uint32_t payload_bytes);
    void write_packet_with_tail(
        GpuOpcode opcode,
        const void* head,
        uint32_t head_bytes,
        const void* tail,
        uint32_t tail_bytes);
    std::vector<uint8_t> buf_;
};

}  // namespace playgta6::gpu
