// Native executor, GPU-independent part: walking the packet stream and decoding the packets the
// native executor handles, with the exact rules of the browser worker (web/workers/wgpu_worker.js,
// web/workers/render_state_packets.js). No WebGPU types here, so this header is unit-tested on
// every host build (tests/test_native_exec_model.cpp) while the Dawn-backed executor
// (native_executor.h) stays behind PLAYGTA6_NATIVE_EXECUTOR. Design: docs/NATIVE_EXECUTOR_DESIGN.md.
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu::native_exec {

static_assert(std::endian::native == std::endian::little, "packet payloads are read in place (little-endian)");

// One packet of a stream (GpuPacketHeader + payload), pointing into the caller's buffer.
struct Packet {
    uint16_t opcode = 0;
    uint16_t flags = 0;
    uint32_t payload_bytes = 0;
    const uint8_t* payload = nullptr;
    size_t offset = 0;  // of the header, from the start of the walked range
};

struct WalkResult {
    size_t packets = 0;
    size_t bytes = 0;  // header + payload bytes of the packets visited
    // The worker's `CommandStream` validation error: a header whose payload runs past the end
    // stops the walk (the rest of the batch is dropped). Fewer than 8 trailing bytes are ignored.
    bool truncated = false;
    uint16_t truncated_opcode = 0;
    uint32_t truncated_need = 0;
    size_t truncated_remaining = 0;
    bool stopped = false;  // the visitor returned false
};

// WgpuCommandProcessor._executeCommandView: visits packets in order; `visit(const Packet&)` returns
// false to stop (the worker stops a batch when the device is lost mid-batch).
template <class Visit>
WalkResult walk_packets(const uint8_t* data, size_t len, Visit&& visit) {
    WalkResult r;
    size_t offset = 0;
    while (offset + sizeof(GpuPacketHeader) <= len) {
        GpuPacketHeader h;
        std::memcpy(&h, data + offset, sizeof h);
        const size_t payload_offset = offset + sizeof h;
        if (payload_offset + h.payload_bytes > len) {
            r.truncated = true;
            r.truncated_opcode = h.opcode;
            r.truncated_need = h.payload_bytes;
            r.truncated_remaining = len - payload_offset;
            break;
        }
        Packet p;
        p.opcode = h.opcode;
        p.flags = h.flags;
        p.payload_bytes = h.payload_bytes;
        p.payload = data + payload_offset;
        p.offset = offset;
        ++r.packets;
        r.bytes += sizeof h + h.payload_bytes;
        offset = payload_offset + h.payload_bytes;
        if (!visit(p)) {
            r.stopped = true;
            break;
        }
    }
    return r;
}

// Length of the longest prefix of `data` made of complete packets (for feeding a file in chunks).
inline size_t complete_prefix(const uint8_t* data, size_t len) {
    size_t offset = 0;
    while (offset + sizeof(GpuPacketHeader) <= len) {
        GpuPacketHeader h;
        std::memcpy(&h, data + offset, sizeof h);
        const size_t end = offset + sizeof h + h.payload_bytes;
        if (end > len) break;
        offset = end;
    }
    return offset;
}

template <class T>
T read_payload(const Packet& p, size_t at = 0) {
    T v;
    std::memcpy(&v, p.payload + at, sizeof v);
    return v;
}

// ---- Clear values ----

struct ClearValue {
    double r = 0, g = 0, b = 0, a = 0;
    bool operator==(const ClearValue&) const = default;
};

// DEFAULT_CLEAR_COLOR of wgpu_worker.js: the v1 lazy swapchain pass clears to it unless the frame
// sent a ClearColor (BeginFrame resets it). The f32 -> double conversion is exact, as in JS.
inline constexpr ClearValue kDefaultClearColor{0.05, 0.07, 0.11, 1.0};

// ClearColor (opcode 2): nullopt = the worker's "truncated payload" validation error (< 16 bytes).
inline std::optional<ClearValue> decode_clear_color(const Packet& p) {
    if (p.payload_bytes < sizeof(GpuClearColorPayload)) return std::nullopt;
    const auto c = read_payload<GpuClearColorPayload>(p);
    return ClearValue{c.r, c.g, c.b, c.a};
}

// render_state_packets.js buildRenderPassDescriptor: `finiteOr(v, fallback) || 0` (non-finite ->
// fallback; the `|| 0` turns -0 into +0).
inline double sanitize_clear_component(float v, double fallback) {
    const double d = std::isfinite(v) ? static_cast<double>(v) : fallback;
    return d == 0.0 ? 0.0 : d;
}

// ---- BeginRenderPass (opcode 26) ----

enum class LoadOp : uint8_t { Load, Clear };
enum class StoreOp : uint8_t { Store, Discard };

struct ColorAttachmentPlan {
    uint32_t texture_id = 0;  // 0 = null attachment
    LoadOp load = LoadOp::Load;
    StoreOp store = StoreOp::Store;
    ClearValue clear{};  // used when load == Clear
    GpuTextureViewDesc view{};
};

struct RenderPassPlan {
    uint32_t color_count = 0;  // min(payload color_count, 8)
    std::array<ColorAttachmentPlan, kGpuMaxColorAttachments> colors{};
    bool has_depth_stencil = false;
    GpuRenderPassDepthStencilAttachment depth_stencil{};
};

// decodeBeginRenderPass + the load/store/clear rules of buildRenderPassDescriptor. nullopt when the
// payload is shorter than 368 bytes (the worker counts the packet as failed and records nothing).
inline std::optional<RenderPassPlan> decode_begin_render_pass(const Packet& p) {
    if (p.payload_bytes < sizeof(GpuBeginRenderPassPayload)) return std::nullopt;
    const auto bp = read_payload<GpuBeginRenderPassPayload>(p);
    RenderPassPlan plan;
    plan.color_count = bp.color_count < kGpuMaxColorAttachments ? bp.color_count : kGpuMaxColorAttachments;
    for (uint32_t i = 0; i < plan.color_count; ++i) {
        const GpuRenderPassColorAttachment& c = bp.color[i];
        ColorAttachmentPlan& out = plan.colors[i];
        out.texture_id = c.texture_id;
        if (c.texture_id == 0) continue;
        // 'clear' / 'load' as sent; anything else (undefined) becomes 'load'. Same for store.
        out.load = c.load_op == static_cast<uint8_t>(GpuLoadOp::Clear) ? LoadOp::Clear : LoadOp::Load;
        out.store = c.store_op == static_cast<uint8_t>(GpuStoreOp::Discard) ? StoreOp::Discard : StoreOp::Store;
        out.view = c.view;
        if (out.load == LoadOp::Clear) {
            out.clear = ClearValue{sanitize_clear_component(c.clear_value[0], 0.0),
                                   sanitize_clear_component(c.clear_value[1], 0.0),
                                   sanitize_clear_component(c.clear_value[2], 0.0),
                                   sanitize_clear_component(c.clear_value[3], 1.0)};
        }
    }
    if (bp.depth_stencil.texture_id != 0) {
        plan.has_depth_stencil = true;
        plan.depth_stencil = bp.depth_stencil;
    }
    return plan;
}

struct AttachmentSize {
    uint32_t width = 0, height = 0;  // 0 = unknown
    bool known() const { return width > 0 && height > 0; }
    bool operator==(const AttachmentSize&) const = default;
};

// _textureAttachmentSize: the mip's size of the view's base mip level.
inline AttachmentSize mip_size(AttachmentSize base, uint32_t mip) {
    if (!base.known() || mip == 0) return base;
    if (mip >= 31) return {1, 1};
    return {base.width >> mip ? base.width >> mip : 1, base.height >> mip ? base.height >> mip : 1};
}

// sanitizeRenderPassAttachments: the first color attachment with a known size (else the
// depth-stencil attachment) is the reference; color attachments of another known size are dropped.
// Returns the bit mask of dropped color attachments.
inline uint32_t drop_mismatched_colors(const RenderPassPlan& plan,
                                       const std::array<AttachmentSize, kGpuMaxColorAttachments>& color_sizes,
                                       AttachmentSize depth_stencil_size) {
    AttachmentSize ref{};
    for (uint32_t i = 0; i < plan.color_count; ++i) {
        if (plan.colors[i].texture_id != 0 && color_sizes[i].known()) {
            ref = color_sizes[i];
            break;
        }
    }
    if (!ref.known() && plan.has_depth_stencil) ref = depth_stencil_size;
    uint32_t dropped = 0;
    if (!ref.known()) return 0;
    for (uint32_t i = 0; i < plan.color_count; ++i) {
        if (plan.colors[i].texture_id == 0 || !color_sizes[i].known()) continue;
        if (!(color_sizes[i] == ref)) dropped |= 1u << i;
    }
    return dropped;
}

// ---- CreateBuffer / WriteBuffer (opcodes 3, 4) ----

inline std::optional<GpuCreateBufferPayload> decode_create_buffer(const Packet& p) {
    if (p.payload_bytes < sizeof(GpuCreateBufferPayload)) return std::nullopt;
    return read_payload<GpuCreateBufferPayload>(p);
}

// wgpu_worker.js MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES.
inline constexpr uint32_t kMaxDirectWriteChunkBytes = 4u * 1024u * 1024u;

enum class WriteBufferPath : uint8_t {
    Skip,    // data_len runs past the payload: the worker returns without writing
    Padded,  // staged copy: data zero-padded to a multiple of 4 (staging_pool.js stageBufferUpload)
    Direct,  // queue.writeBuffer(buffer, dst_offset, data), in 4 MiB chunks when larger
};

struct WriteBufferPlan {
    WriteBufferPath path = WriteBufferPath::Skip;
    uint32_t id = 0;
    uint32_t dst_offset = 0;
    uint32_t data_len = 0;
    uint32_t copy_size = 0;  // Padded: data_len rounded up to 4
    const uint8_t* data = nullptr;
};

// _handleWriteBuffer. `buffer_size` is the destination's descriptor size (the caller has already
// checked that the buffer exists; a missing buffer is a silent no-op in the worker). A Direct write
// keeps WebGPU's own validation: an unaligned offset or size is rejected by queue.writeBuffer, the
// way Chrome (which is Dawn) rejects it.
inline WriteBufferPlan plan_write_buffer(const Packet& p, uint64_t buffer_size) {
    WriteBufferPlan plan;
    if (p.payload_bytes < sizeof(GpuWriteBufferPayload)) return plan;
    const auto w = read_payload<GpuWriteBufferPayload>(p);
    if (uint64_t{sizeof(GpuWriteBufferPayload)} + w.data_len > p.payload_bytes) return plan;
    plan.id = w.id;
    plan.dst_offset = w.dst_offset;
    plan.data_len = w.data_len;
    plan.data = p.payload + sizeof(GpuWriteBufferPayload);
    if (w.data_len > 0 && w.dst_offset % 4 == 0) {
        const uint64_t copy = (uint64_t{w.data_len} + 3) & ~uint64_t{3};
        if (uint64_t{w.dst_offset} + copy <= buffer_size) {
            plan.path = WriteBufferPath::Padded;
            plan.copy_size = static_cast<uint32_t>(copy);
            return plan;
        }
    }
    plan.path = WriteBufferPath::Direct;
    return plan;
}

}  // namespace playgta6::gpu::native_exec
