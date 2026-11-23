#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu {

// Magic "GPUR" (0x47505552) matching web/workers/gpu_ring.js.
inline constexpr uint32_t kGpuRingMagic = 0x47505552u;
inline constexpr uint32_t kGpuRingVersion = 1u;
inline constexpr size_t kGpuRingHeaderBytes = 64u;

constexpr bool is_power_of_two(uint32_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

// Fixed 64-byte shared control header placed at byte offset 0 of a SharedArrayBuffer
// or shared linear memory region, immediately followed by `capacity_bytes` of ring data.
struct GpuRingHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t capacity_bytes;
    uint32_t reserved0;
    std::atomic<uint32_t> write_pos;
    std::atomic<uint32_t> read_pos;
    std::atomic<uint32_t> dropped_packets;
    std::atomic<uint32_t> notify_seq;
    uint32_t padding[8];
};

static_assert(sizeof(std::atomic<uint32_t>) == 4, "std::atomic<uint32_t> must be 4 bytes");
static_assert(sizeof(GpuRingHeader) == kGpuRingHeaderBytes, "GpuRingHeader layout must stay 64 bytes");
static_assert(offsetof(GpuRingHeader, magic) == 0, "GpuRingHeader::magic offset mismatch");
static_assert(offsetof(GpuRingHeader, version) == 4, "GpuRingHeader::version offset mismatch");
static_assert(offsetof(GpuRingHeader, capacity_bytes) == 8, "GpuRingHeader::capacity_bytes offset mismatch");
static_assert(offsetof(GpuRingHeader, reserved0) == 12, "GpuRingHeader::reserved0 offset mismatch");
static_assert(offsetof(GpuRingHeader, write_pos) == 16, "GpuRingHeader::write_pos offset mismatch");
static_assert(offsetof(GpuRingHeader, read_pos) == 20, "GpuRingHeader::read_pos offset mismatch");
static_assert(offsetof(GpuRingHeader, dropped_packets) == 24, "GpuRingHeader::dropped_packets offset mismatch");
static_assert(offsetof(GpuRingHeader, notify_seq) == 28, "GpuRingHeader::notify_seq offset mismatch");
static_assert(offsetof(GpuRingHeader, padding) == 32, "GpuRingHeader::padding offset mismatch");

inline size_t gpu_ring_total_bytes(uint32_t capacity_bytes) noexcept {
    return kGpuRingHeaderBytes + static_cast<size_t>(capacity_bytes);
}

inline bool validate_gpu_ring_header(const GpuRingHeader* header) noexcept {
    if (header == nullptr) {
        return false;
    }
    return header->magic == kGpuRingMagic &&
           header->version == kGpuRingVersion &&
           is_power_of_two(header->capacity_bytes);
}

inline bool init_gpu_ring_header(GpuRingHeader* header, uint32_t capacity_bytes) noexcept {
    if (header == nullptr || !is_power_of_two(capacity_bytes)) {
        return false;
    }
    std::memset(static_cast<void*>(header), 0, sizeof(GpuRingHeader));
    auto* initialized = new (header) GpuRingHeader();
    initialized->version = kGpuRingVersion;
    initialized->capacity_bytes = capacity_bytes;
    initialized->reserved0 = 0u;
    initialized->write_pos.store(0u, std::memory_order_relaxed);
    initialized->read_pos.store(0u, std::memory_order_relaxed);
    initialized->dropped_packets.store(0u, std::memory_order_relaxed);
    initialized->notify_seq.store(0u, std::memory_order_relaxed);
    for (uint32_t& pad : initialized->padding) {
        pad = 0u;
    }
    std::atomic_thread_fence(std::memory_order_release);
    initialized->magic = kGpuRingMagic;
    return true;
}

inline bool init_gpu_ring_memory(void* memory, size_t total_bytes) noexcept {
    if (memory == nullptr || total_bytes <= kGpuRingHeaderBytes) {
        return false;
    }
    const size_t data_bytes = total_bytes - kGpuRingHeaderBytes;
    if (data_bytes > 0x80000000u) {
        return false;
    }
    const auto capacity_bytes = static_cast<uint32_t>(data_bytes);
    if (!is_power_of_two(capacity_bytes)) {
        return false;
    }
    return init_gpu_ring_header(static_cast<GpuRingHeader*>(memory), capacity_bytes);
}

// Lock-free Single-Producer side of the SPSC GPU command ring buffer.
class GpuRingWriter {
public:
    GpuRingWriter() = default;

    GpuRingWriter(GpuRingHeader* header, uint8_t* data) noexcept {
        if (validate_gpu_ring_header(header) && data != nullptr) {
            header_ = header;
            data_ = data;
            capacity_ = header->capacity_bytes;
            mask_ = capacity_ - 1u;
        }
    }

    static GpuRingWriter init_in_place(void* memory, size_t total_bytes) noexcept {
        if (!init_gpu_ring_memory(memory, total_bytes)) {
            return {};
        }
        auto* base = static_cast<uint8_t*>(memory);
        return GpuRingWriter(
            reinterpret_cast<GpuRingHeader*>(base),
            base + kGpuRingHeaderBytes);
    }

    static GpuRingWriter attach(void* memory, size_t total_bytes) noexcept {
        if (memory == nullptr || total_bytes < kGpuRingHeaderBytes) {
            return {};
        }
        auto* base = static_cast<uint8_t*>(memory);
        auto* header = reinterpret_cast<GpuRingHeader*>(base);
        if (!validate_gpu_ring_header(header)) {
            return {};
        }
        if (total_bytes < gpu_ring_total_bytes(header->capacity_bytes)) {
            return {};
        }
        return GpuRingWriter(header, base + kGpuRingHeaderBytes);
    }

    bool valid() const noexcept { return header_ != nullptr && data_ != nullptr; }
    uint32_t capacity_bytes() const noexcept { return capacity_; }
    GpuRingHeader* header() const noexcept { return header_; }
    uint8_t* data() const noexcept { return data_; }

    uint32_t write_pos() const noexcept {
        return valid() ? header_->write_pos.load(std::memory_order_relaxed) : 0u;
    }

    uint32_t read_pos() const noexcept {
        return valid() ? header_->read_pos.load(std::memory_order_acquire) : 0u;
    }

    uint32_t available_bytes() const noexcept {
        if (!valid()) {
            return 0u;
        }
        const uint32_t w = header_->write_pos.load(std::memory_order_relaxed);
        const uint32_t r = header_->read_pos.load(std::memory_order_acquire);
        return static_cast<uint32_t>(w - r);
    }

    uint32_t free_bytes() const noexcept {
        if (!valid()) {
            return 0u;
        }
        const uint32_t used = available_bytes();
        return used >= capacity_ ? 0u : (capacity_ - used);
    }

    uint32_t dropped_packets() const noexcept {
        return valid() ? header_->dropped_packets.load(std::memory_order_relaxed) : 0u;
    }

    uint32_t notify_seq() const noexcept {
        return valid() ? header_->notify_seq.load(std::memory_order_relaxed) : 0u;
    }

    // All-or-nothing byte write. Returns false (and increments dropped_packets) without
    // modifying write_pos or ring data when there is insufficient free space.
    bool write_bytes(const void* src, size_t len) noexcept {
        if (!valid()) {
            return false;
        }
        if (len == 0u) {
            return true;
        }
        if (src == nullptr) {
            return false;
        }
        if (len > capacity_) {
            header_->dropped_packets.fetch_add(1u, std::memory_order_relaxed);
            return false;
        }

        const uint32_t w = header_->write_pos.load(std::memory_order_relaxed);
        const uint32_t r = header_->read_pos.load(std::memory_order_acquire);
        const uint32_t used = static_cast<uint32_t>(w - r);
        const uint32_t free = capacity_ - used;
        if (len > free) {
            header_->dropped_packets.fetch_add(1u, std::memory_order_relaxed);
            return false;
        }

        copy_in(w, static_cast<const uint8_t*>(src), len);
        header_->write_pos.store(
            static_cast<uint32_t>(w + static_cast<uint32_t>(len)),
            std::memory_order_release);
        header_->notify_seq.fetch_add(1u, std::memory_order_release);
        return true;
    }

    // Writes a framed GpuPacketHeader + payload atomically without splitting across full
    // backpressure or allocating temporary buffers.
    bool write_packet(
        GpuOpcode opcode,
        const void* payload,
        uint32_t payload_bytes,
        uint16_t flags = 0u) noexcept {
        if (!valid()) {
            return false;
        }
        if (payload_bytes > 0u && payload == nullptr) {
            return false;
        }
        const uint64_t total_len64 =
            static_cast<uint64_t>(sizeof(GpuPacketHeader)) + static_cast<uint64_t>(payload_bytes);
        if (total_len64 > capacity_) {
            header_->dropped_packets.fetch_add(1u, std::memory_order_relaxed);
            return false;
        }

        const auto total_len = static_cast<uint32_t>(total_len64);
        const uint32_t w = header_->write_pos.load(std::memory_order_relaxed);
        const uint32_t r = header_->read_pos.load(std::memory_order_acquire);
        const uint32_t used = static_cast<uint32_t>(w - r);
        const uint32_t free = capacity_ - used;
        if (total_len > free) {
            header_->dropped_packets.fetch_add(1u, std::memory_order_relaxed);
            return false;
        }

        const GpuPacketHeader pkt_header{
            static_cast<uint16_t>(opcode),
            flags,
            payload_bytes,
        };
        copy_in(w, reinterpret_cast<const uint8_t*>(&pkt_header), sizeof(pkt_header));
        if (payload_bytes > 0u) {
            copy_in(
                static_cast<uint32_t>(w + static_cast<uint32_t>(sizeof(pkt_header))),
                static_cast<const uint8_t*>(payload),
                payload_bytes);
        }

        header_->write_pos.store(static_cast<uint32_t>(w + total_len), std::memory_order_release);
        header_->notify_seq.fetch_add(1u, std::memory_order_release);
        return true;
    }

    // Validates packet framing across `stream` and writes the whole stream atomically.
    bool write_packet_stream(const uint8_t* stream, size_t stream_bytes) noexcept {
        if (!valid()) {
            return false;
        }
        if (stream_bytes == 0u) {
            return true;
        }
        if (stream == nullptr) {
            return false;
        }

        size_t offset = 0u;
        while (offset < stream_bytes) {
            if (stream_bytes - offset < sizeof(GpuPacketHeader)) {
                return false;
            }
            GpuPacketHeader pkt_header{};
            std::memcpy(&pkt_header, stream + offset, sizeof(pkt_header));
            offset += sizeof(pkt_header);
            if (pkt_header.payload_bytes > stream_bytes - offset) {
                return false;
            }
            offset += pkt_header.payload_bytes;
        }

        return write_bytes(stream, stream_bytes);
    }

private:
    void copy_in(uint32_t pos, const uint8_t* src, size_t len) noexcept {
        if (len == 0u) {
            return;
        }
        const uint32_t offset = pos & mask_;
        const size_t first = std::min<size_t>(len, static_cast<size_t>(capacity_ - offset));
        std::memcpy(data_ + offset, src, first);
        if (len > first) {
            std::memcpy(data_, src + first, len - first);
        }
    }

    GpuRingHeader* header_ = nullptr;
    uint8_t* data_ = nullptr;
    uint32_t capacity_ = 0u;
    uint32_t mask_ = 0u;
};

// Lock-free Single-Consumer side of the SPSC GPU command ring buffer.
class GpuRingReader {
public:
    GpuRingReader() = default;

    GpuRingReader(GpuRingHeader* header, const uint8_t* data) noexcept {
        if (validate_gpu_ring_header(header) && data != nullptr) {
            header_ = header;
            data_ = data;
            capacity_ = header->capacity_bytes;
            mask_ = capacity_ - 1u;
        }
    }

    static GpuRingReader attach(void* memory, size_t total_bytes) noexcept {
        if (memory == nullptr || total_bytes < kGpuRingHeaderBytes) {
            return {};
        }
        auto* base = static_cast<uint8_t*>(memory);
        auto* header = reinterpret_cast<GpuRingHeader*>(base);
        if (!validate_gpu_ring_header(header)) {
            return {};
        }
        if (total_bytes < gpu_ring_total_bytes(header->capacity_bytes)) {
            return {};
        }
        return GpuRingReader(header, base + kGpuRingHeaderBytes);
    }

    bool valid() const noexcept { return header_ != nullptr && data_ != nullptr; }
    uint32_t capacity_bytes() const noexcept { return capacity_; }
    GpuRingHeader* header() const noexcept { return header_; }
    const uint8_t* data() const noexcept { return data_; }

    uint32_t write_pos() const noexcept {
        return valid() ? header_->write_pos.load(std::memory_order_acquire) : 0u;
    }

    uint32_t read_pos() const noexcept {
        return valid() ? header_->read_pos.load(std::memory_order_relaxed) : 0u;
    }

    uint32_t available_bytes() const noexcept {
        if (!valid()) {
            return 0u;
        }
        const uint32_t w = header_->write_pos.load(std::memory_order_acquire);
        const uint32_t r = header_->read_pos.load(std::memory_order_relaxed);
        return static_cast<uint32_t>(w - r);
    }

    uint32_t free_bytes() const noexcept {
        if (!valid()) {
            return 0u;
        }
        const uint32_t used = available_bytes();
        return used >= capacity_ ? 0u : (capacity_ - used);
    }

    uint32_t dropped_packets() const noexcept {
        return valid() ? header_->dropped_packets.load(std::memory_order_relaxed) : 0u;
    }

    uint32_t notify_seq() const noexcept {
        return valid() ? header_->notify_seq.load(std::memory_order_acquire) : 0u;
    }

    size_t read_bytes(void* dst, size_t max_len) noexcept {
        if (!valid() || max_len == 0u || dst == nullptr) {
            return 0u;
        }
        const uint32_t r = header_->read_pos.load(std::memory_order_relaxed);
        const uint32_t w = header_->write_pos.load(std::memory_order_acquire);
        const uint32_t avail = static_cast<uint32_t>(w - r);
        const size_t to_read = std::min<size_t>(avail, max_len);
        if (to_read == 0u) {
            return 0u;
        }
        copy_out(r, static_cast<uint8_t*>(dst), to_read);
        header_->read_pos.store(
            static_cast<uint32_t>(r + static_cast<uint32_t>(to_read)),
            std::memory_order_release);
        return to_read;
    }

    bool read_exact(void* dst, size_t len) noexcept {
        if (!valid()) {
            return false;
        }
        if (len == 0u) {
            return true;
        }
        if (dst == nullptr) {
            return false;
        }
        const uint32_t r = header_->read_pos.load(std::memory_order_relaxed);
        const uint32_t w = header_->write_pos.load(std::memory_order_acquire);
        const uint32_t avail = static_cast<uint32_t>(w - r);
        if (len > avail) {
            return false;
        }
        copy_out(r, static_cast<uint8_t*>(dst), len);
        header_->read_pos.store(
            static_cast<uint32_t>(r + static_cast<uint32_t>(len)),
            std::memory_order_release);
        return true;
    }

    // Reads a single complete framed GpuPacketHeader + payload. Returns false without
    // advancing read_pos if a complete packet is not yet available.
    bool read_packet(GpuPacketHeader& out_header, std::vector<uint8_t>& out_payload) {
        if (!valid()) {
            return false;
        }
        const uint32_t r = header_->read_pos.load(std::memory_order_relaxed);
        const uint32_t w = header_->write_pos.load(std::memory_order_acquire);
        const uint32_t avail = static_cast<uint32_t>(w - r);
        if (avail < sizeof(GpuPacketHeader)) {
            return false;
        }

        GpuPacketHeader pkt_header{};
        copy_out(r, reinterpret_cast<uint8_t*>(&pkt_header), sizeof(pkt_header));
        const uint64_t total_len64 =
            static_cast<uint64_t>(sizeof(GpuPacketHeader)) +
            static_cast<uint64_t>(pkt_header.payload_bytes);
        if (total_len64 > avail) {
            return false;
        }

        out_header = pkt_header;
        out_payload.resize(pkt_header.payload_bytes);
        if (pkt_header.payload_bytes > 0u) {
            copy_out(
                static_cast<uint32_t>(r + static_cast<uint32_t>(sizeof(pkt_header))),
                out_payload.data(),
                pkt_header.payload_bytes);
        }

        header_->read_pos.store(
            static_cast<uint32_t>(r + static_cast<uint32_t>(total_len64)),
            std::memory_order_release);
        return true;
    }

    size_t drain_all(std::vector<uint8_t>& out) {
        if (!valid()) {
            return 0u;
        }
        const uint32_t r = header_->read_pos.load(std::memory_order_relaxed);
        const uint32_t w = header_->write_pos.load(std::memory_order_acquire);
        const uint32_t avail = static_cast<uint32_t>(w - r);
        if (avail == 0u) {
            return 0u;
        }
        const size_t prev_size = out.size();
        out.resize(prev_size + avail);
        copy_out(r, out.data() + prev_size, avail);
        header_->read_pos.store(static_cast<uint32_t>(r + avail), std::memory_order_release);
        return avail;
    }

    // Invokes `cb(const uint8_t* ptr, size_t len)` with a single contiguous byte slice.
    // Zero-copy when the queued bytes do not wrap around; linearizes into an internal
    // scratch buffer on wrap-around so framed GpuPacketHeader packets are never split.
    template <typename Callback>
    size_t drain_to(Callback&& cb) {
        if (!valid()) {
            return 0u;
        }
        const uint32_t r = header_->read_pos.load(std::memory_order_relaxed);
        const uint32_t w = header_->write_pos.load(std::memory_order_acquire);
        const uint32_t avail = static_cast<uint32_t>(w - r);
        if (avail == 0u) {
            return 0u;
        }

        const uint32_t offset = r & mask_;
        if (static_cast<size_t>(offset) + static_cast<size_t>(avail) <= static_cast<size_t>(capacity_)) {
            cb(data_ + offset, avail);
        } else {
            if (scratch_.size() < avail) {
                scratch_.resize(avail);
            }
            copy_out(r, scratch_.data(), avail);
            cb(scratch_.data(), avail);
        }

        header_->read_pos.store(static_cast<uint32_t>(r + avail), std::memory_order_release);
        return avail;
    }

private:
    void copy_out(uint32_t pos, uint8_t* dst, size_t len) const noexcept {
        if (len == 0u) {
            return;
        }
        const uint32_t offset = pos & mask_;
        const size_t first = std::min<size_t>(len, static_cast<size_t>(capacity_ - offset));
        std::memcpy(dst, data_ + offset, first);
        if (len > first) {
            std::memcpy(dst + first, data_, len - first);
        }
    }

    GpuRingHeader* header_ = nullptr;
    const uint8_t* data_ = nullptr;
    uint32_t capacity_ = 0u;
    uint32_t mask_ = 0u;
    std::vector<uint8_t> scratch_;
};

}  // namespace playgta6::gpu
