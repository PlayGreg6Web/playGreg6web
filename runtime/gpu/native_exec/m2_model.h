// Native executor milestone 2, GPU-independent part: GPU -> CPU read-backs (ReadbackBuffer /
// ReadbackTexture) and the v1 pipeline rules, with the exact arithmetic of the browser worker
// (web/workers/wgpu_worker.js `decodeReadbackPayload`, `readbackStagingLayout`, `readbackPieces`,
// `deliverReadback`). No WebGPU types, so tests/test_native_exec_model.cpp runs it on every host
// build. Design: docs/NATIVE_EXECUTOR_DESIGN.md.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "gpu/native_exec/m1_model.h"
#include "gpu/native_exec/stream_model.h"

namespace playgta6::gpu::native_exec {

// ---- Read-back requests (opcodes 39 / 40) ----

struct ReadbackRequest {
    enum class Kind : uint8_t { Buffer, Texture } kind = Kind::Buffer;
    uint32_t seq = 0;
    uint32_t id = 0;  // buffer id / texture id
    uint64_t src_offset = 0;  // buffer
    uint64_t size = 0;        // buffer: bytes; texture: dst_size
    uint64_t dst_addr = 0;
    uint64_t done_addr = 0;
    // texture
    uint32_t mip_level = 0, origin_x = 0, origin_y = 0, origin_z = 0;
    uint32_t width = 0, height = 0, depth_or_layers = 0, bytes_per_row = 0, rows_per_image = 0;
};

// decodeReadbackPayload: nullopt = the worker's "truncated payload" validation error.
inline std::optional<ReadbackRequest> decode_readback(const Packet& p) {
    ReadbackRequest r;
    if (p.opcode == static_cast<uint16_t>(GpuOpcode::ReadbackBuffer)) {
        if (p.payload_bytes < sizeof(GpuReadbackBufferPayload)) return std::nullopt;
        const auto v = read_payload<GpuReadbackBufferPayload>(p);
        r.kind = ReadbackRequest::Kind::Buffer;
        r.seq = v.seq;
        r.id = v.buffer_id;
        r.src_offset = v.src_offset;
        r.size = v.size;
        r.dst_addr = v.dst_addr;
        r.done_addr = v.done_addr;
        return r;
    }
    if (p.opcode == static_cast<uint16_t>(GpuOpcode::ReadbackTexture)) {
        if (p.payload_bytes < sizeof(GpuReadbackTexturePayload)) return std::nullopt;
        const auto v = read_payload<GpuReadbackTexturePayload>(p);
        r.kind = ReadbackRequest::Kind::Texture;
        r.seq = v.seq;
        r.id = v.texture_id;
        r.mip_level = v.mip_level;
        r.origin_x = v.origin_x;
        r.origin_y = v.origin_y;
        r.origin_z = v.origin_z;
        r.width = v.width;
        r.height = v.height;
        r.depth_or_layers = v.depth_or_layers;
        r.bytes_per_row = v.bytes_per_row;
        r.rows_per_image = v.rows_per_image;
        r.dst_addr = v.dst_addr;
        r.done_addr = v.done_addr;
        r.size = v.dst_size;
        return r;
    }
    return std::nullopt;
}

// readbackStagingLayout. The worker computes in JS doubles (exact below 2^53); a request whose
// numbers leave that range is invalid here (the worker would work on rounded values).
struct ReadbackLayout {
    bool valid = true;
    // buffer
    uint64_t copy_offset = 0, skip = 0;
    // texture
    uint32_t rows = 0, depth = 0;
    int64_t row_bytes = 0;
    uint64_t staging_row = 0, staging_rows_per_image = 0;
    uint64_t staging_size = 0;
};

inline constexpr uint64_t kMaxExactNumber = uint64_t{1} << 53;

inline ReadbackLayout readback_staging_layout(const ReadbackRequest& req, uint32_t block_height = 1) {
    ReadbackLayout l;
    if (req.kind == ReadbackRequest::Kind::Buffer) {
        if (req.src_offset >= kMaxExactNumber || req.size >= kMaxExactNumber) {
            l.valid = false;
            return l;
        }
        const uint64_t start = req.src_offset - (req.src_offset % 4);
        const uint64_t end = req.src_offset + req.size;
        const uint64_t aligned_end = end + ((4 - (end % 4)) % 4);
        l.copy_offset = start;
        l.skip = req.src_offset - start;
        l.staging_size = std::max<uint64_t>(4, aligned_end - start);
        return l;
    }
    const uint32_t bh = std::max<uint32_t>(1, block_height);
    l.rows = std::max<uint32_t>(1, (req.height + bh - 1) / bh);
    l.depth = std::max<uint32_t>(1, req.depth_or_layers);
    const uint64_t pitch = req.bytes_per_row;
    if (req.size >= kMaxExactNumber) {
        l.valid = false;
        return l;
    }
    l.row_bytes = static_cast<int64_t>(req.size) -
                  static_cast<int64_t>((uint64_t{l.depth - 1} * req.rows_per_image + (l.rows - 1)) * pitch);
    const uint64_t rb = static_cast<uint64_t>(std::max<int64_t>(1, l.row_bytes));
    l.staging_row = (rb + 255) / 256 * 256;
    l.staging_rows_per_image = l.rows;
    l.staging_size = l.staging_row * l.rows * l.depth;
    return l;
}

// readbackPieces: where each run of mapped staging bytes lands in the guest layout.
struct ReadbackPiece {
    uint64_t dst_offset;  // from dst_addr
    uint64_t src_offset;  // into the mapped staging bytes
    uint64_t length;
};

inline std::vector<ReadbackPiece> readback_pieces(const ReadbackRequest& req, const ReadbackLayout& l) {
    std::vector<ReadbackPiece> pieces;
    if (req.kind == ReadbackRequest::Kind::Buffer) {
        pieces.push_back({0, l.skip, req.size});
        return pieces;
    }
    if (!(l.row_bytes > 0) || static_cast<uint64_t>(l.row_bytes) > req.bytes_per_row) return pieces;
    const uint64_t rb = static_cast<uint64_t>(l.row_bytes);
    for (uint32_t z = 0; z < l.depth; ++z) {
        for (uint32_t r = 0; r < l.rows; ++r) {
            const uint64_t src = (uint64_t{z} * l.staging_rows_per_image + r) * l.staging_row;
            const uint64_t dst = (uint64_t{z} * req.rows_per_image + r) * req.bytes_per_row;
            if (dst + rb > req.size) return pieces;  // the worker returns what it has
            pieces.push_back({dst, src, rb});
        }
    }
    return pieces;
}

// The memory read-back results are written to: the worker's `attachGuestMemory(buffer)`.
//  - base + size: `dst_addr` / `done_addr` are offsets into [base, base + size), like the wasm
//    linear-memory offsets of the browser runtime.
//  - host_addresses: they are absolute host addresses (what the native producer's ReadbackGate
//    hands out: `done_address()` is a host pointer). Nothing can be bounds-checked then.
// Nothing attached: nothing is written and the counter does not move (the worker's `!memory`).
struct GuestMemory {
    uint8_t* base = nullptr;
    uint64_t size = 0;
    bool host_addresses = false;

    bool attached() const { return host_addresses || base != nullptr; }
    // The bytes [addr, addr + len), or nullptr when they are not inside the memory.
    uint8_t* at(uint64_t addr, uint64_t len) const {
        if (host_addresses) return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(addr));
        if (!base || addr > size || len > size - addr) return nullptr;
        return base + addr;
    }
};

// deliverReadback: writes the pieces (when `mapped` is non-null: a failed copy writes nothing) and
// then, always, adds 1 to the u32 completion counter at done_addr (the counter write is atomic, the
// native equivalent of Atomics.add + Atomics.notify). Returns true when the bytes were written.
inline bool deliver_readback(const GuestMemory& mem, const ReadbackRequest& req, const ReadbackLayout& layout,
                             const uint8_t* mapped, size_t mapped_len) {
    if (!mem.attached()) return false;
    bool written = false;
    if (mapped) {
        if (uint8_t* dst = mem.at(req.dst_addr, req.size)) {
            for (const ReadbackPiece& piece : readback_pieces(req, layout)) {
                if (piece.src_offset >= mapped_len) continue;
                const uint64_t n = std::min<uint64_t>(piece.length, mapped_len - piece.src_offset);
                std::memcpy(dst + piece.dst_offset, mapped + piece.src_offset, static_cast<size_t>(n));
            }
            written = true;
        }
    }
    if (req.done_addr % 4 == 0) {
        if (uint8_t* counter = mem.at(req.done_addr, 4)) {
            std::atomic_ref<uint32_t> c(*reinterpret_cast<uint32_t*>(counter));
            c.fetch_add(1, std::memory_order_release);
            c.notify_all();
        }
    }
    return written;
}

// textureFormatBlockHeight: block-compressed formats read back in rows of 4-texel blocks. A 3D BC
// texture that was created as rgba8unorm (the fallback) has a plain format.
inline uint32_t readback_block_height(uint8_t effective_format, bool bc3d_fallback) {
    return !bc3d_fallback && format_is_bc(effective_format) ? 4 : 1;
}

// ---- v1 pipelines (opcodes 8, 19) without a reflection registry ----

// _handleCreateRenderPipeline when `_resolveShaderReflection` finds nothing (no reflection JSON
// loaded, as on the replay page): one vertex buffer of `vertexStride` bytes with a single attribute
// at location 0, offset 0; none for a zero stride.
struct V1VertexLayout {
    bool has_buffer = false;
    uint32_t stride = 0;
    uint8_t components = 0;  // 1, 2 or 3 x float32
};

inline V1VertexLayout plan_v1_vertex_layout(uint32_t vertex_stride) {
    V1VertexLayout l;
    if (vertex_stride == 0) return l;
    l.has_buffer = true;
    l.stride = vertex_stride;
    l.components = vertex_stride >= 12 ? 3 : vertex_stride >= 8 ? 2 : 1;
    return l;
}

// ---- Device negotiation (wgpu_worker.js `requestNegotiatedDevice`) ----

// OPTIONAL_ADAPTER_LIMIT_SPECS: a limit is requested (at min(supported, cap)) only when that is
// above the WebGPU default the spec pins, so an adapter at the default asks for nothing.
struct LimitSpec {
    const char* name;
    uint64_t default_min;
    uint64_t cap;
};

inline constexpr LimitSpec kOptionalLimitSpecs[] = {
    {"maxSampledTexturesPerShaderStage", 16, 256},
    {"maxSamplersPerShaderStage", 16, 64},
    {"maxStorageBuffersPerShaderStage", 8, 32},
    {"maxUniformBuffersPerShaderStage", 12, 32},
    {"maxStorageTexturesPerShaderStage", 4, 16},
    {"maxBindingsPerBindGroup", 1000, 4096},
    {"maxBufferSize", 256ull * 1024 * 1024, 1024ull * 1024 * 1024},
    {"maxStorageBufferBindingSize", 128ull * 1024 * 1024, 512ull * 1024 * 1024},
    {"maxComputeWorkgroupStorageSize", 16384, 65536},
};
inline constexpr size_t kOptionalLimitCount = sizeof(kOptionalLimitSpecs) / sizeof(kOptionalLimitSpecs[0]);

// selectSupportedAdapterLimits for one limit: the value to require, or nullopt (the worker's
// `clamped > defaultMin` test; a non-supported value is never requested).
inline std::optional<uint64_t> negotiated_limit(const LimitSpec& spec, uint64_t adapter_supported) {
    const uint64_t clamped = std::min(adapter_supported, spec.cap);
    if (clamped > spec.default_min) return clamped;
    return std::nullopt;
}

}  // namespace playgta6::gpu::native_exec
