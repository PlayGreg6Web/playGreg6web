// Native executor stream model (gpu/native_exec/stream_model.h, replay_frames.h): packet walking,
// payload decoding and the replay frame split / digest, checked against the browser worker's rules.
// No GPU involved.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/native_exec/replay_frames.h"
#include "gpu/native_exec/stream_model.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_native_exec_model.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::gpu::native_exec;

std::vector<Packet> walk_all(const uint8_t* data, size_t len, WalkResult* out = nullptr) {
    std::vector<Packet> v;
    WalkResult r = walk_packets(data, len, [&](const Packet& p) {
        v.push_back(p);
        return true;
    });
    if (out) *out = r;
    return v;
}

Packet single(const gpu::CommandEncoder& enc) {
    auto v = walk_all(enc.data(), enc.size());
    return v.empty() ? Packet{} : v[0];
}

void test_walk() {
    gpu::CommandEncoder enc;
    enc.begin_frame();
    enc.clear_color(0.25f, 0.5f, 0.75f, 1.0f);
    enc.present();
    WalkResult r;
    auto packets = walk_all(enc.data(), enc.size(), &r);
    CHECK(packets.size() == 3);
    CHECK(!r.truncated && r.bytes == enc.size());
    CHECK(packets[1].opcode == static_cast<uint16_t>(GpuOpcode::ClearColor) && packets[1].payload_bytes == 16);
    CHECK(packets[2].offset == 8 + 24);
    CHECK(complete_prefix(enc.data(), enc.size()) == enc.size());

    // A header whose payload runs past the end stops the walk (the worker's CommandStream error).
    std::vector<uint8_t> bytes(enc.data(), enc.data() + enc.size());
    const GpuPacketHeader bad{static_cast<uint16_t>(GpuOpcode::WriteBuffer), 0, 100};
    const size_t at = bytes.size();
    bytes.resize(at + 8 + 10);
    std::memcpy(bytes.data() + at, &bad, 8);
    packets = walk_all(bytes.data(), bytes.size(), &r);
    CHECK(packets.size() == 3);
    CHECK(r.truncated && r.truncated_opcode == 4 && r.truncated_need == 100 && r.truncated_remaining == 10);
    CHECK(complete_prefix(bytes.data(), bytes.size()) == at);
    // Fewer than 8 trailing bytes are ignored silently.
    bytes.resize(at + 5);
    walk_all(bytes.data(), bytes.size(), &r);
    CHECK(!r.truncated && r.packets == 3);

    // The visitor can stop the walk (device lost mid-batch).
    int seen = 0;
    r = walk_packets(enc.data(), enc.size(), [&](const Packet&) { return ++seen < 2; });
    CHECK(seen == 2 && r.stopped && r.packets == 2);
}

void test_clear_color() {
    gpu::CommandEncoder enc;
    enc.clear_color(1.0f, 0.1f, 0.0f, 0.5f);
    auto c = decode_clear_color(single(enc));
    CHECK(c.has_value());
    CHECK(c->r == 1.0 && c->g == static_cast<double>(0.1f) && c->b == 0.0 && c->a == 0.5);
    gpu::CommandEncoder shorter;
    const float three[3] = {1, 1, 1};
    shorter.emit_packet(GpuOpcode::ClearColor, three, sizeof three);
    CHECK(!decode_clear_color(single(shorter)).has_value());
    CHECK(kDefaultClearColor == (ClearValue{0.05, 0.07, 0.11, 1.0}));

    CHECK(sanitize_clear_component(std::numeric_limits<float>::quiet_NaN(), 1.0) == 1.0);
    CHECK(sanitize_clear_component(std::numeric_limits<float>::infinity(), 0.0) == 0.0);
    const double z = sanitize_clear_component(-0.0f, 0.0);
    CHECK(z == 0.0 && !std::signbit(z));
    CHECK(sanitize_clear_component(0.5f, 0.0) == 0.5);
}

GpuBeginRenderPassPayload empty_pass() {
    GpuBeginRenderPassPayload bp;
    std::memset(&bp, 0, sizeof bp);
    return bp;
}

void test_begin_render_pass() {
    GpuBeginRenderPassPayload bp = empty_pass();
    bp.color_count = 12;  // clamped to 8
    bp.color[0].texture_id = kGpuSwapchainTextureId;
    bp.color[0].load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
    bp.color[0].store_op = static_cast<uint8_t>(GpuStoreOp::Discard);
    bp.color[0].clear_value[0] = std::numeric_limits<float>::quiet_NaN();
    bp.color[0].clear_value[1] = 0.25f;
    bp.color[0].clear_value[2] = -0.0f;
    bp.color[0].clear_value[3] = std::numeric_limits<float>::infinity();
    bp.color[1].texture_id = 9;
    bp.color[1].load_op = 0;   // undefined -> load
    bp.color[1].store_op = 7;  // unknown -> store
    bp.color[1].view.format = 27;
    gpu::CommandEncoder enc;
    enc.emit_packet(GpuOpcode::BeginRenderPass, &bp, sizeof bp);
    auto plan = decode_begin_render_pass(single(enc));
    CHECK(plan.has_value());
    CHECK(plan->color_count == 8);
    CHECK(plan->colors[0].load == LoadOp::Clear && plan->colors[0].store == StoreOp::Discard);
    CHECK(plan->colors[0].clear == (ClearValue{0.0, 0.25, 0.0, 1.0}));
    CHECK(plan->colors[1].texture_id == 9 && plan->colors[1].load == LoadOp::Load &&
          plan->colors[1].store == StoreOp::Store && plan->colors[1].view.format == 27);
    CHECK(plan->colors[2].texture_id == 0);
    CHECK(!plan->has_depth_stencil);

    gpu::CommandEncoder shorter;
    shorter.emit_packet(GpuOpcode::BeginRenderPass, &bp, 367);
    CHECK(!decode_begin_render_pass(single(shorter)).has_value());

    // Size sanitizer: the first sized color attachment is the reference.
    RenderPassPlan p;
    p.color_count = 3;
    p.colors[0].texture_id = 1;
    p.colors[1].texture_id = 2;
    p.colors[2].texture_id = 3;
    std::array<AttachmentSize, kGpuMaxColorAttachments> sizes{};
    sizes[0] = {};          // unknown: skipped
    sizes[1] = {64, 32};    // reference
    sizes[2] = {32, 32};    // dropped
    CHECK(drop_mismatched_colors(p, sizes, {}) == 0b100);
    sizes[2] = {64, 32};
    CHECK(drop_mismatched_colors(p, sizes, {}) == 0);
    // Null attachments are never the reference (nor dropped); the depth-stencil attachment does not
    // outrank a sized color attachment.
    sizes = {};
    sizes[0] = {16, 16};
    p.colors[0].texture_id = 0;
    p.has_depth_stencil = true;
    sizes[1] = {8, 8};
    sizes[2] = {4, 4};
    CHECK(drop_mismatched_colors(p, sizes, {4, 4}) == 0b100);
    CHECK(mip_size({64, 32}, 2) == (AttachmentSize{16, 8}));
    CHECK(mip_size({64, 32}, 6) == (AttachmentSize{1, 1}));
    CHECK(mip_size({64, 32}, 40) == (AttachmentSize{1, 1}));
}

void test_write_buffer() {
    const uint8_t data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto plan_for = [&](uint32_t offset, uint32_t len, uint64_t size) {
        gpu::CommandEncoder enc;
        enc.write_buffer(5, offset, data, len);
        return plan_write_buffer(single(enc), size);
    };
    WriteBufferPlan p = plan_for(4, 5, 16);  // staged: padded to 8 bytes
    CHECK(p.path == WriteBufferPath::Padded && p.copy_size == 8 && p.data_len == 5 && p.id == 5);
    CHECK(p.data && p.data[0] == 1 && p.data[4] == 5);
    p = plan_for(8, 8, 16);
    CHECK(p.path == WriteBufferPath::Padded && p.copy_size == 8);
    p = plan_for(12, 5, 16);  // the padded copy would overflow: direct (WebGPU rejects the size)
    CHECK(p.path == WriteBufferPath::Direct);
    p = plan_for(2, 4, 16);  // unaligned offset: direct (WebGPU rejects the offset)
    CHECK(p.path == WriteBufferPath::Direct && p.dst_offset == 2);
    p = plan_for(0, 0, 16);
    CHECK(p.path == WriteBufferPath::Direct && p.data_len == 0);

    // data_len past the payload: the worker returns without writing.
    GpuWriteBufferPayload w{5, 0, 64, 0};
    gpu::CommandEncoder enc;
    enc.emit_packet(GpuOpcode::WriteBuffer, &w, sizeof w, data, sizeof data);
    CHECK(plan_write_buffer(single(enc), 1024).path == WriteBufferPath::Skip);
}

void test_split_frames() {
    gpu::CommandEncoder enc;
    enc.create_buffer(1, GpuBufferUsage::CopyDst, 64);
    enc.begin_frame();
    enc.present();
    const size_t frame0 = enc.size();
    // A thinned upload: data_len 0, reserved == 'thin'.
    GpuWriteBufferPayload thin{1, 0, 0, kThinMarker};
    enc.emit_packet(GpuOpcode::WriteBuffer, &thin, sizeof thin);
    enc.clear_color(1, 0, 0, 1);
    enc.present();
    enc.begin_frame();  // trailing packets: a final frame
    auto frames = split_replay_frames(enc.data(), enc.size());
    CHECK(frames.size() == 3);
    CHECK(frames[0].size() == frame0);
    CHECK(frames[1].size() == 24 + 8);  // the thin packet is gone
    CHECK(frames[2].size() == 8);
    WalkResult r;
    auto packets = walk_all(frames[1].data(), frames[1].size(), &r);
    CHECK(packets.size() == 2 && packets[0].opcode == static_cast<uint16_t>(GpuOpcode::ClearColor));
}

void test_digest() {
    // Reference value computed by tools/replay/replay.html's digest code on the same pixels.
    uint8_t px[3 * 2 * 4];
    for (size_t i = 0; i < sizeof px; ++i) px[i] = static_cast<uint8_t>((i * 37 + 11) & 255);
    CHECK(replay_digest(px, 3, 2) == "1876ef7b336afd4d");

    std::vector<uint8_t> img(8 * 8 * 4, 0);
    // Sampled pixels are (0,0), (4,0), (0,4), (4,4); alpha does not count.
    img[3] = 255;                       // (0,0): alpha only -> black
    img[(0 * 8 + 4) * 4 + 1] = 10;      // (4,0)
    img[(4 * 8 + 4) * 4 + 1] = 10;      // (4,4): same colour
    img[(1 * 8 + 1) * 4] = 99;          // not sampled
    const SampledStats s = replay_sampled_stats(img.data(), 8, 8);
    CHECK(s.total == 4 && s.non_black == 2 && s.distinct == 2);
}

}  // namespace

int main() {
    test_walk();
    test_clear_color();
    test_begin_render_pass();
    test_write_buffer();
    test_split_frames();
    test_digest();
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_native_exec_model: ok\n");
    return 0;
}
