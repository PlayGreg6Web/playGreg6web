#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include "abi/playgta6_abi.h"
#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_xs.h"
#include "gpu/gpu_ring.h"

namespace {

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                                   \
        }                                                                                   \
    } while (0)

struct DecodedPacket {
    playgta6::GpuPacketHeader header{};
    const uint8_t* payload = nullptr;
};

std::vector<DecodedPacket> decode_stream(const uint8_t* data, size_t size) {
    std::vector<DecodedPacket> packets;
    size_t offset = 0;
    while (offset + sizeof(playgta6::GpuPacketHeader) <= size) {
        DecodedPacket pkt{};
        std::memcpy(&pkt.header, data + offset, sizeof(pkt.header));
        offset += sizeof(pkt.header);
        CHECK(offset + pkt.header.payload_bytes <= size);
        pkt.payload = data + offset;
        offset += pkt.header.payload_bytes;
        packets.push_back(pkt);
    }
    CHECK(offset == size);
    return packets;
}

template <typename T>
T read_pod(const uint8_t* ptr) {
    T out{};
    std::memcpy(&out, ptr, sizeof(T));
    return out;
}

void test_legacy_frame_packets() {
    playgta6::gpu::CommandEncoder enc;
    enc.begin_frame();
    enc.clear_color(0.1f, 0.2f, 0.3f, 1.0f);
    enc.present();

    // 8 (BeginFrame) + (8 + 16) (ClearColor) + 8 (Present) = 40 bytes
    CHECK(enc.size() == 40);

    const auto packets = decode_stream(enc.data(), enc.size());
    CHECK(packets.size() == 3);
    CHECK(packets[0].header.opcode == static_cast<uint16_t>(playgta6::GpuOpcode::BeginFrame));
    CHECK(packets[0].header.payload_bytes == 0);

    CHECK(packets[1].header.opcode == static_cast<uint16_t>(playgta6::GpuOpcode::ClearColor));
    CHECK(packets[1].header.payload_bytes == 16);
    const auto clear = read_pod<playgta6::GpuClearColorPayload>(packets[1].payload);
    CHECK(clear.r == 0.1f && clear.g == 0.2f && clear.b == 0.3f && clear.a == 1.0f);

    CHECK(packets[2].header.opcode == static_cast<uint16_t>(playgta6::GpuOpcode::Present));
    CHECK(packets[2].header.payload_bytes == 0);
}

void test_full_command_encoder_packets() {
    using namespace playgta6;

    gpu::CommandEncoder enc;
    const std::array<float, 9> vertices = {
        0.0f, 0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
    };
    const std::string_view vs_wgsl = "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    const std::string_view fs_wgsl = "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }";

    enc.nop();
    enc.begin_frame();
    enc.create_buffer(1, GpuBufferUsage::Vertex | GpuBufferUsage::CopyDst, sizeof(vertices));
    enc.write_buffer(1, 0, vertices.data(), static_cast<uint32_t>(sizeof(vertices)));
    enc.create_shader_module(2, GpuShaderStage::Vertex, vs_wgsl);
    enc.create_shader_module(3, GpuShaderStage::Fragment, fs_wgsl);
    enc.create_render_pipeline(4, 2, 3, 12, GpuPrimitiveTopology::TriangleList, 0x3u);
    enc.clear_color(0.25f, 0.5f, 0.75f, 1.0f);
    enc.set_pipeline(4);
    enc.set_vertex_buffer(0, 1, 0);
    enc.draw(3, 1, 0, 0);
    enc.present();
    enc.destroy_resource(1);

    const auto packets = decode_stream(enc.data(), enc.size());
    CHECK(packets.size() == 13);

    CHECK(packets[0].header.opcode == static_cast<uint16_t>(GpuOpcode::Nop));
    CHECK(packets[0].header.payload_bytes == 0);

    CHECK(packets[1].header.opcode == static_cast<uint16_t>(GpuOpcode::BeginFrame));
    CHECK(packets[1].header.payload_bytes == 0);

    CHECK(packets[2].header.opcode == static_cast<uint16_t>(GpuOpcode::CreateBuffer));
    CHECK(packets[2].header.payload_bytes == sizeof(GpuCreateBufferPayload));
    const auto cb = read_pod<GpuCreateBufferPayload>(packets[2].payload);
    CHECK(cb.id == 1);
    CHECK(cb.usage_flags == (GpuBufferUsage::Vertex | GpuBufferUsage::CopyDst));
    CHECK(cb.byte_size == sizeof(vertices));

    CHECK(packets[3].header.opcode == static_cast<uint16_t>(GpuOpcode::WriteBuffer));
    CHECK(packets[3].header.payload_bytes == sizeof(GpuWriteBufferPayload) + sizeof(vertices));
    const auto wb = read_pod<GpuWriteBufferPayload>(packets[3].payload);
    CHECK(wb.id == 1 && wb.dst_offset == 0 && wb.data_len == sizeof(vertices) && wb.reserved == 0);
    CHECK(std::memcmp(packets[3].payload + sizeof(GpuWriteBufferPayload), vertices.data(), sizeof(vertices)) == 0);

    CHECK(packets[4].header.opcode == static_cast<uint16_t>(GpuOpcode::CreateShaderModule));
    CHECK(packets[4].header.payload_bytes == sizeof(GpuCreateShaderModulePayload) + vs_wgsl.size());
    const auto vs = read_pod<GpuCreateShaderModulePayload>(packets[4].payload);
    CHECK(vs.id == 2 && vs.stage == static_cast<uint32_t>(GpuShaderStage::Vertex));
    CHECK(vs.code_len == vs_wgsl.size() && vs.reserved == 0);
    CHECK(std::string_view(reinterpret_cast<const char*>(packets[4].payload + sizeof(GpuCreateShaderModulePayload)), vs.code_len) == vs_wgsl);

    CHECK(packets[5].header.opcode == static_cast<uint16_t>(GpuOpcode::CreateShaderModule));
    const auto fs = read_pod<GpuCreateShaderModulePayload>(packets[5].payload);
    CHECK(fs.id == 3 && fs.stage == static_cast<uint32_t>(GpuShaderStage::Fragment));
    CHECK(fs.code_len == fs_wgsl.size() && fs.reserved == 0);
    CHECK(std::string_view(reinterpret_cast<const char*>(packets[5].payload + sizeof(GpuCreateShaderModulePayload)), fs.code_len) == fs_wgsl);

    CHECK(packets[6].header.opcode == static_cast<uint16_t>(GpuOpcode::CreateRenderPipeline));
    CHECK(packets[6].header.payload_bytes == sizeof(GpuCreateRenderPipelinePayload));
    const auto rp = read_pod<GpuCreateRenderPipelinePayload>(packets[6].payload);
    CHECK(rp.id == 4 && rp.vs_shader_id == 2 && rp.fs_shader_id == 3);
    CHECK(rp.vertex_stride == 12 && rp.topology == static_cast<uint32_t>(GpuPrimitiveTopology::TriangleList) && rp.flags == 0x3u);

    CHECK(packets[7].header.opcode == static_cast<uint16_t>(GpuOpcode::ClearColor));
    const auto cc = read_pod<GpuClearColorPayload>(packets[7].payload);
    CHECK(cc.r == 0.25f && cc.g == 0.5f && cc.b == 0.75f && cc.a == 1.0f);

    CHECK(packets[8].header.opcode == static_cast<uint16_t>(GpuOpcode::SetPipeline));
    CHECK(packets[8].header.payload_bytes == sizeof(GpuSetPipelinePayload));
    CHECK(read_pod<GpuSetPipelinePayload>(packets[8].payload).pipeline_id == 4);

    CHECK(packets[9].header.opcode == static_cast<uint16_t>(GpuOpcode::SetVertexBuffer));
    CHECK(packets[9].header.payload_bytes == sizeof(GpuSetVertexBufferPayload));
    const auto vb = read_pod<GpuSetVertexBufferPayload>(packets[9].payload);
    CHECK(vb.slot == 0 && vb.buffer_id == 1 && vb.offset == 0);

    CHECK(packets[10].header.opcode == static_cast<uint16_t>(GpuOpcode::Draw));
    CHECK(packets[10].header.payload_bytes == sizeof(GpuDrawPayload));
    const auto dr = read_pod<GpuDrawPayload>(packets[10].payload);
    CHECK(dr.vertex_count == 3 && dr.instance_count == 1 && dr.first_vertex == 0 && dr.first_instance == 0);

    CHECK(packets[11].header.opcode == static_cast<uint16_t>(GpuOpcode::Present));
    CHECK(packets[11].header.payload_bytes == 0);

    CHECK(packets[12].header.opcode == static_cast<uint16_t>(GpuOpcode::DestroyResource));
    CHECK(packets[12].header.payload_bytes == sizeof(GpuDestroyResourcePayload));
    CHECK(read_pod<GpuDestroyResourcePayload>(packets[12].payload).id == 1);
}

void test_d3d12_xs_shim_flow() {
    using namespace playgta6;
    using namespace playgta6::gpu::d3d12_xs;

    gpu::CommandEncoder enc;
    D3D12Device device(&enc);
    CHECK(device.ref_count() == 1);
    CHECK(device.AddRef() == 2);
    CHECK(device.Release() == 1);

    D3D12ResourceHandle vb_handle{};
    const std::array<float, 6> verts = {0.0f, 0.5f, -0.5f, -0.5f, 0.5f, -0.5f};
    CHECK(Succeeded(device.CreateCommittedResource(
        D3D12BufferDesc{sizeof(verts), GpuBufferUsage::Vertex | GpuBufferUsage::CopyDst},
        &vb_handle)));
    CHECK(vb_handle.id == 1);
    CHECK(vb_handle.byte_size == sizeof(verts));

    // Invalid arg check.
    CHECK(Failed(device.CreateCommittedResource(D3D12BufferDesc{0, GpuBufferUsage::Vertex}, &vb_handle)));

    D3D12ShaderHandle vs_handle{};
    D3D12ShaderHandle fs_handle{};
    CHECK(Succeeded(device.CreateShaderModule(GpuShaderStage::Vertex, "fn vs_main() {}", &vs_handle)));
    CHECK(Succeeded(device.CreateShaderModule(GpuShaderStage::Fragment, "fn fs_main() {}", &fs_handle)));
    CHECK(vs_handle.id == 2);
    CHECK(fs_handle.id == 3);

    D3D12PipelineStateHandle pso_handle{};
    CHECK(Succeeded(device.CreateGraphicsPipelineState(
        D3D12GraphicsPipelineStateDesc{
            vs_handle.id,
            fs_handle.id,
            8,
            GpuPrimitiveTopology::TriangleStrip,
            0,
        },
        &pso_handle)));
    CHECK(pso_handle.id == 4);

    D3D12GraphicsCommandList cmd_list;
    CHECK(Succeeded(device.CreateCommandList(&cmd_list)));
    CHECK(!cmd_list.is_open());
    CHECK(Failed(cmd_list.Close()));

    CHECK(Succeeded(cmd_list.Reset()));
    CHECK(cmd_list.is_open());

    const float clear_rgba[4] = {0.05f, 0.1f, 0.2f, 1.0f};
    cmd_list.CopyBufferRegion(vb_handle, 0, verts.data(), static_cast<uint32_t>(sizeof(verts)));
    cmd_list.ClearRenderTargetView(clear_rgba);
    cmd_list.SetPipelineState(pso_handle);
    cmd_list.IASetVertexBuffers(0, D3D12VertexBufferView{vb_handle.id, 0, static_cast<uint32_t>(sizeof(verts)), 8});
    cmd_list.DrawInstanced(3, 2, 0, 1);
    CHECK(Succeeded(cmd_list.Close()));
    CHECK(!cmd_list.is_open());

    // Calls while closed must be ignored.
    cmd_list.DrawInstanced(99, 1, 0, 0);

    PresentX present_x;
    CHECK(Succeeded(device.CreatePresentX(&present_x)));
    CHECK(Succeeded(present_x.Present(1, 0)));
    CHECK(present_x.present_count() == 1);
    CHECK(present_x.last_sync_interval() == 1);

    CHECK(Succeeded(device.DestroyResource(vb_handle.id)));

    const auto packets = decode_stream(enc.data(), enc.size());
    // CreateBuffer, CreateShaderModule(VS), CreateShaderModule(FS), CreateRenderPipeline,
    // BeginFrame (from Reset), WriteBuffer, ClearColor, SetPipeline, SetVertexBuffer, Draw,
    // Present, DestroyResource => 12 packets.
    CHECK(packets.size() == 12);
    CHECK(packets[0].header.opcode == static_cast<uint16_t>(GpuOpcode::CreateBuffer));
    CHECK(packets[1].header.opcode == static_cast<uint16_t>(GpuOpcode::CreateShaderModule));
    CHECK(packets[2].header.opcode == static_cast<uint16_t>(GpuOpcode::CreateShaderModule));
    CHECK(packets[3].header.opcode == static_cast<uint16_t>(GpuOpcode::CreateRenderPipeline));
    CHECK(packets[4].header.opcode == static_cast<uint16_t>(GpuOpcode::BeginFrame));
    CHECK(packets[5].header.opcode == static_cast<uint16_t>(GpuOpcode::WriteBuffer));
    CHECK(packets[6].header.opcode == static_cast<uint16_t>(GpuOpcode::ClearColor));
    CHECK(packets[7].header.opcode == static_cast<uint16_t>(GpuOpcode::SetPipeline));
    CHECK(packets[8].header.opcode == static_cast<uint16_t>(GpuOpcode::SetVertexBuffer));
    CHECK(packets[9].header.opcode == static_cast<uint16_t>(GpuOpcode::Draw));
    CHECK(packets[10].header.opcode == static_cast<uint16_t>(GpuOpcode::Present));
    CHECK(packets[11].header.opcode == static_cast<uint16_t>(GpuOpcode::DestroyResource));

    const auto draw_payload = read_pod<GpuDrawPayload>(packets[9].payload);
    CHECK(draw_payload.vertex_count == 3);
    CHECK(draw_payload.instance_count == 2);
    CHECK(draw_payload.first_vertex == 0);
    CHECK(draw_payload.first_instance == 1);

    const auto destroy_payload = read_pod<GpuDestroyResourcePayload>(packets[11].payload);
    CHECK(destroy_payload.id == vb_handle.id);
}

void test_gpu_ring_wraparound_and_backpressure() {
    using namespace playgta6::gpu;

    // Non-power-of-two capacity must be rejected.
    std::vector<uint8_t> bad_mem(kGpuRingHeaderBytes + 96u, 0u);
    CHECK(!init_gpu_ring_memory(bad_mem.data(), bad_mem.size()));
    CHECK(!GpuRingWriter::init_in_place(bad_mem.data(), bad_mem.size()).valid());

    // Valid 64-byte ring buffer (total 128 bytes).
    constexpr uint32_t kCapacity = 64u;
    std::vector<uint8_t> mem(gpu_ring_total_bytes(kCapacity), 0xCCu);
    auto writer = GpuRingWriter::init_in_place(mem.data(), mem.size());
    CHECK(writer.valid());
    CHECK(writer.capacity_bytes() == kCapacity);
    CHECK(writer.available_bytes() == 0u);
    CHECK(writer.free_bytes() == kCapacity);
    CHECK(writer.header()->magic == kGpuRingMagic);
    CHECK(writer.header()->version == kGpuRingVersion);

    auto reader = GpuRingReader::attach(mem.data(), mem.size());
    CHECK(reader.valid());
    CHECK(reader.capacity_bytes() == kCapacity);

    // Fill 48 bytes, drain 48 bytes -> read_pos = write_pos = 48.
    std::array<uint8_t, 48> chunk1{};
    for (size_t i = 0; i < chunk1.size(); ++i) {
        chunk1[i] = static_cast<uint8_t>(i + 1u);
    }
    CHECK(writer.write_bytes(chunk1.data(), chunk1.size()));
    CHECK(writer.available_bytes() == 48u);
    CHECK(writer.free_bytes() == 16u);
    CHECK(writer.notify_seq() == 1u);

    // Backpressure: writing 17 bytes when only 16 are free must fail atomically.
    std::array<uint8_t, 17> too_big{};
    too_big.fill(0xEEu);
    CHECK(!writer.write_bytes(too_big.data(), too_big.size()));
    CHECK(writer.dropped_packets() == 1u);
    CHECK(writer.available_bytes() == 48u);
    CHECK(writer.free_bytes() == 16u);

    // Writing > capacity must also fail and increment dropped_packets.
    std::array<uint8_t, 65> oversized{};
    CHECK(!writer.write_bytes(oversized.data(), oversized.size()));
    CHECK(writer.dropped_packets() == 2u);

    // Drain chunk1 (contiguous zero-copy path) and verify contents.
    std::vector<uint8_t> drained;
    const size_t drained1 = reader.drain_to([&](const uint8_t* ptr, size_t len) {
        CHECK(ptr == reader.data());
        drained.assign(ptr, ptr + len);
    });
    CHECK(drained1 == chunk1.size());
    CHECK(drained.size() == chunk1.size());
    CHECK(std::memcmp(drained.data(), chunk1.data(), chunk1.size()) == 0);
    CHECK(reader.available_bytes() == 0u);
    CHECK(writer.free_bytes() == kCapacity);

    // Now write 40 bytes starting at offset 48 -> wraps across [48..64) and [0..24).
    std::array<uint8_t, 40> wrapped{};
    for (size_t i = 0; i < wrapped.size(); ++i) {
        wrapped[i] = static_cast<uint8_t>(0xA0u + i);
    }
    CHECK(writer.write_bytes(wrapped.data(), wrapped.size()));
    CHECK(writer.available_bytes() == 40u);
    CHECK(writer.free_bytes() == 24u);

    // Fill the remaining 24 bytes exactly to reach 100% capacity.
    std::array<uint8_t, 24> tail{};
    for (size_t i = 0; i < tail.size(); ++i) {
        tail[i] = static_cast<uint8_t>(0x50u + i);
    }
    CHECK(writer.write_bytes(tail.data(), tail.size()));
    CHECK(writer.available_bytes() == kCapacity);
    CHECK(writer.free_bytes() == 0u);

    // Full ring rejects even a 1-byte write without corrupting buffered bytes.
    const uint8_t one_byte = 0xFFu;
    CHECK(!writer.write_bytes(&one_byte, 1u));
    CHECK(writer.dropped_packets() == 3u);

    // Drain wrapped 64 bytes as one contiguous slice via drain_to.
    drained.clear();
    const size_t drained2 = reader.drain_to([&](const uint8_t* ptr, size_t len) {
        drained.assign(ptr, ptr + len);
    });
    CHECK(drained2 == kCapacity);
    CHECK(drained.size() == kCapacity);
    CHECK(std::memcmp(drained.data(), wrapped.data(), wrapped.size()) == 0);
    CHECK(std::memcmp(drained.data() + wrapped.size(), tail.data(), tail.size()) == 0);
    CHECK(reader.available_bytes() == 0u);
    CHECK(writer.free_bytes() == kCapacity);

    // Verify monotonic uint32_t counter wraparound across 0xFFFFFFFF -> 0.
    constexpr uint32_t kNearWrap = 0xFFFFFFF0u;
    writer.header()->write_pos.store(kNearWrap, std::memory_order_relaxed);
    writer.header()->read_pos.store(kNearWrap, std::memory_order_relaxed);
    std::array<uint8_t, 32> wrap32{};
    for (size_t i = 0; i < wrap32.size(); ++i) {
        wrap32[i] = static_cast<uint8_t>(i ^ 0x5Au);
    }
    CHECK(writer.write_bytes(wrap32.data(), wrap32.size()));
    CHECK(writer.write_pos() == 0x00000010u);
    CHECK(reader.available_bytes() == 32u);
    std::array<uint8_t, 32> wrap32_out{};
    CHECK(reader.read_exact(wrap32_out.data(), wrap32_out.size()));
    CHECK(reader.read_pos() == 0x00000010u);
    CHECK(std::memcmp(wrap32_out.data(), wrap32.data(), wrap32.size()) == 0);
}

void test_gpu_ring_command_encoder_roundtrip() {
    using namespace playgta6;
    using namespace playgta6::gpu;

    // Build a realistic multi-command frame with variable-length payloads.
    CommandEncoder enc;
    const std::array<float, 9> vertices = {
        0.0f, 0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
    };
    const std::string_view vs_wgsl =
        "@vertex fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }";
    const std::string_view fs_wgsl =
        "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.8, 0.4, 1.0); }";

    enc.begin_frame();
    enc.create_buffer(1, GpuBufferUsage::Vertex | GpuBufferUsage::CopyDst, sizeof(vertices));
    enc.write_buffer(1, 0, vertices.data(), static_cast<uint32_t>(sizeof(vertices)));
    enc.create_shader_module(2, GpuShaderStage::Vertex, vs_wgsl);
    enc.create_shader_module(3, GpuShaderStage::Fragment, fs_wgsl);
    enc.create_render_pipeline(4, 2, 3, 12, GpuPrimitiveTopology::TriangleList, 0u);
    enc.clear_color(0.1f, 0.2f, 0.3f, 1.0f);
    enc.set_pipeline(4);
    enc.set_vertex_buffer(0, 1, 0);
    enc.draw(3, 1, 0, 0);
    enc.present();
    enc.destroy_resource(1);

    // Choose a 512-byte ring so repeated frames wrap around multiple times.
    constexpr uint32_t kCapacity = 512u;
    CHECK(enc.size() < kCapacity);
    std::vector<uint8_t> mem(gpu_ring_total_bytes(kCapacity), 0u);
    auto writer = GpuRingWriter::init_in_place(mem.data(), mem.size());
    auto reader = GpuRingReader::attach(mem.data(), mem.size());
    CHECK(writer.valid() && reader.valid());

    // Stream 12 consecutive frames (forcing many wrap-arounds at different offsets).
    for (int iter = 0; iter < 12; ++iter) {
        CHECK(writer.write_packet_stream(enc.data(), enc.size()));

        std::vector<uint8_t> received;
        const size_t n = reader.drain_to([&](const uint8_t* ptr, size_t len) {
            received.assign(ptr, ptr + len);
        });
        CHECK(n == enc.size());
        CHECK(received.size() == enc.size());
        CHECK(std::memcmp(received.data(), enc.data(), enc.size()) == 0);

        const auto packets = decode_stream(received.data(), received.size());
        CHECK(packets.size() == 12);
        CHECK(packets[0].header.opcode == static_cast<uint16_t>(GpuOpcode::BeginFrame));
        CHECK(packets[10].header.opcode == static_cast<uint16_t>(GpuOpcode::Present));
        CHECK(packets[11].header.opcode == static_cast<uint16_t>(GpuOpcode::DestroyResource));
    }

    // Also test write_packet + read_packet directly with header straddling the boundary.
    std::vector<uint8_t> tiny_mem(gpu_ring_total_bytes(64u), 0u);
    auto tiny_writer = GpuRingWriter::init_in_place(tiny_mem.data(), tiny_mem.size());
    auto tiny_reader = GpuRingReader::attach(tiny_mem.data(), tiny_mem.size());
    CHECK(tiny_writer.valid() && tiny_reader.valid());

    // Advance offset to 60 so the next 8-byte GpuPacketHeader splits 4+4 across boundary.
    std::array<uint8_t, 60> pad{};
    CHECK(tiny_writer.write_bytes(pad.data(), pad.size()));
    std::array<uint8_t, 60> discard{};
    CHECK(tiny_reader.read_exact(discard.data(), discard.size()));

    const GpuClearColorPayload clear_in{0.25f, 0.5f, 0.75f, 1.0f};
    CHECK(tiny_writer.write_packet(GpuOpcode::ClearColor, &clear_in, sizeof(clear_in), 0x0007u));

    // A second large packet (48 bytes total > 40 bytes free) must fail without corrupting first.
    std::array<uint8_t, 40> big_payload{};
    CHECK(!tiny_writer.write_packet(GpuOpcode::WriteBuffer, big_payload.data(), 40u));
    CHECK(tiny_writer.dropped_packets() == 1u);

    GpuPacketHeader out_hdr{};
    std::vector<uint8_t> out_payload;
    CHECK(tiny_reader.read_packet(out_hdr, out_payload));
    CHECK(out_hdr.opcode == static_cast<uint16_t>(GpuOpcode::ClearColor));
    CHECK(out_hdr.flags == 0x0007u);
    CHECK(out_hdr.payload_bytes == sizeof(GpuClearColorPayload));
    CHECK(out_payload.size() == sizeof(GpuClearColorPayload));
    const auto clear_out = read_pod<GpuClearColorPayload>(out_payload.data());
    CHECK(clear_out.r == 0.25f && clear_out.g == 0.5f && clear_out.b == 0.75f && clear_out.a == 1.0f);
    CHECK(!tiny_reader.read_packet(out_hdr, out_payload));
}

}  // namespace

int main() {
    test_legacy_frame_packets();
    test_full_command_encoder_packets();
    test_d3d12_xs_shim_flow();
    test_gpu_ring_wraparound_and_backpressure();
    test_gpu_ring_command_encoder_roundtrip();
    return 0;
}
