#pragma once

// Recycler for small host textures (backlog 38).
//
// A host texture is a driver allocation of at least 64 KiB however few bytes it holds, and the title
// creates and drops thousands of tiny ones while it streams. Instead of destroying a small texture
// that is let go of and creating an identical one a moment later, the owner parks its host id here
// and takes it back when a texture of the same shape (format, extent, mips, dimension, storage use)
// is needed. The pool is bookkeeping only: it holds ids and byte counts and emits no GPU packets.
//
// A parked texture still holds its old texels. The owner must zero every texel of every mip on
// reuse so no stale content is ever visible: with a ClearTexture packet (GPU-side, when the
// consumer has it: gpu_texture_clear()) or else with CPU zero writes (zero_fill_writes).
//
// Two pools: small_texture_pool() for chains up to kSmallTextureMaxBytes (with either clear), and
// large_texture_pool() for bigger chains up to kLargeTextureMaxBytes, used only with the GPU clear
// (CPU zero uploads of those would cost more than the creations they save).
//
// Parked textures stay counted in the texture budget (texture_budget.h): the owner calls
// note_created once when the host texture is really created and note_destroyed only when it is
// really destroyed, including when the pool evicts or drains it.
//
// Thread-safe.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu {

inline constexpr uint64_t kSmallTextureMaxBytes = 64u * 1024u;      // largest mip chain that is pooled
inline constexpr uint32_t kSmallTexturePoolMaxCount = 512;          // parked textures at most
inline constexpr uint64_t kSmallTexturePoolMaxBytes = 16ull << 20;  // parked bytes at most
inline constexpr uint64_t kLargeTextureMaxBytes = 32ull << 20;      // largest chain of the large pool
inline constexpr uint32_t kLargeTexturePoolMaxCount = 1024;
inline constexpr uint64_t kLargeTexturePoolMaxBytes = 256ull << 20;
// The large pool is trimmed while the texture budget's use (parked bytes included) is above this
// share of the budget, so parked textures never push live ones into the size cap (texture_budget.h).
inline constexpr uint32_t kLargeTexturePoolBudgetPercent = 60;

// The GPU consumer zeroes textures itself (kGpuOpcodeClearTexture). Enabled by live native
// and browser boots, otherwise off for consumers that do not implement the packet.
void set_gpu_texture_clear(bool on) noexcept;
bool gpu_texture_clear() noexcept;

// The GPU consumer copies texture to texture itself (kGpuOpcodeCopyTexture): D3D12
// copies whose destination is neither a render target nor a UAV are emitted as that packet instead of
// being skipped. Enabled by live native and browser boots, otherwise off for consumers that do not
// implement the packet.
void set_gpu_texture_copy(bool on) noexcept;
bool gpu_texture_copy() noexcept;

struct SmallTextureShape {
    uint64_t owner = 0;  // the device that owns the host texture ids (ids are per device)
    uint32_t dxgi_format = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint8_t dim = 0;
    bool storage = false;

    bool operator==(const SmallTextureShape& o) const noexcept {
        return owner == o.owner && dxgi_format == o.dxgi_format && width == o.width && height == o.height && depth == o.depth &&
               mip_levels == o.mip_levels && dim == o.dim && storage == o.storage;
    }
};

struct SmallTextureShapeHash {
    size_t operator()(const SmallTextureShape& s) const noexcept {
        auto mix = [](uint64_t h, uint64_t v) noexcept -> uint64_t {
            h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
            return h;
        };
        uint64_t h = s.owner * 0x9E3779B97F4A7C15ull;
        h = mix(h, (uint64_t{s.dxgi_format} << 32) | s.width);
        h = mix(h, (uint64_t{s.height} << 32) | s.depth);
        h = mix(h, (uint64_t{s.mip_levels} << 16) | (uint64_t{s.dim} << 8) | (s.storage ? 1ull : 0ull));
        return static_cast<size_t>(h);
    }
};

struct SmallTexturePoolStats {
    uint64_t creates = 0;        // host textures really created by the owner
    uint64_t hits = 0;           // acquires served by a parked texture
    uint64_t misses = 0;         // acquires of a poolable shape that found none
    uint64_t bytes_reused = 0;   // texture bytes served by hits
    uint64_t bytes_created = 0;  // texture bytes of the creates
    uint64_t parked_total = 0;   // textures parked over the run
    uint64_t evicted = 0;        // parked textures destroyed for room
    uint64_t rejected = 0;       // releases not parked (too big, or pool disabled)
    uint64_t create_ns = 0;      // time the owner spent creating host textures
    uint64_t address_misses = 0;     // address-texture lookups that needed a texture (hit or create)
    uint64_t address_evictions = 0;  // address textures dropped by the LRU (count ceiling or VRAM reclaim)
    uint64_t address_recreates = 0;  // misses on a key the LRU had dropped before (LRU thrash)
    uint32_t parked_count = 0;
    uint64_t parked_bytes = 0;
    uint64_t peak_parked_bytes = 0;
};

class SmallTexturePool {
public:
    SmallTexturePool() = default;
    // A pool of chains in (min_chain_bytes, max_chain_bytes] with its own count / byte limits.
    SmallTexturePool(uint64_t min_chain_bytes, uint64_t max_chain_bytes, uint32_t max_count, uint64_t max_bytes) noexcept
        : min_chain_(min_chain_bytes),
          max_chain_(max_chain_bytes),
          default_max_count_(max_count),
          default_max_bytes_(max_bytes),
          max_count_(max_count),
          max_bytes_(max_bytes) {}

    static bool poolable(uint64_t chain_bytes) noexcept {
        return chain_bytes != 0 && chain_bytes <= kSmallTextureMaxBytes;
    }
    // This pool's chain range.
    bool accepts(uint64_t chain_bytes) const noexcept {
        return chain_bytes != 0 && chain_bytes > min_chain_ && chain_bytes <= max_chain_;
    }
    uint64_t max_chain_bytes() const noexcept { return max_chain_; }

    void set_limits(uint32_t max_count, uint64_t max_bytes) noexcept;
    void set_enabled(bool enabled) noexcept;
    void reset() noexcept;

    // A parked texture of this shape, or 0 (counts a hit or, for a poolable shape, a miss).
    uint32_t acquire(const SmallTextureShape& shape, uint64_t chain_bytes) noexcept;

    // Parks `id`. Returns false when it is not poolable or the pool is off: the owner destroys it.
    // Ids pushed out to make room are appended to `evicted` with their byte counts: the owner
    // destroys those and notes the bytes destroyed in the budget.
    bool release(const SmallTextureShape& shape, uint32_t id, uint64_t chain_bytes, std::vector<uint32_t>* evicted,
                 std::vector<uint64_t>* evicted_bytes);

    // Removes every parked texture of `owner` (device teardown, memory pressure).
    void drain(uint64_t owner, std::vector<uint32_t>* ids, std::vector<uint64_t>* bytes);

    // Immediately evicts oldest parked textures until parked_count <= max_count and parked_bytes <= max_bytes.
    // Returns the number of textures evicted.
    size_t trim(uint32_t max_count, uint64_t max_bytes, std::vector<uint32_t>* evicted,
                std::vector<uint64_t>* evicted_bytes);

    // The owner reports a real host texture creation.
    void note_create(uint64_t chain_bytes, uint64_t ns) noexcept;

    // The owner's address-texture LRU: a miss (`recreate`: of a key it dropped before), an eviction.
    void note_address_miss(bool recreate) noexcept;
    void note_address_eviction() noexcept;

    // Number of distinct shapes currently parked in the pool.
    size_t distinct_shapes() const;

    SmallTexturePoolStats stats() const;

private:
    static constexpr uint32_t kInvalidNode = UINT32_MAX;
    struct Node {
        SmallTextureShape shape;
        uint32_t id = 0;
        uint64_t bytes = 0;
        uint32_t lru_prev = kInvalidNode;
        uint32_t lru_next = kInvalidNode;
        uint32_t bucket_prev = kInvalidNode;
        uint32_t bucket_next = kInvalidNode;
        bool active = false;
    };

    uint32_t alloc_node_locked();
    void unlink_lru_locked(uint32_t idx) noexcept;
    void unlink_and_free_locked(uint32_t idx) noexcept;

    mutable std::mutex mutex_;
    std::vector<Node> nodes_;
    std::vector<uint32_t> free_nodes_;
    std::unordered_map<SmallTextureShape, uint32_t, SmallTextureShapeHash> buckets_;  // shape -> newest (LIFO head)
    uint32_t lru_head_ = kInvalidNode;                                                 // oldest
    uint32_t lru_tail_ = kInvalidNode;                                                 // newest
    uint64_t min_chain_ = 0;
    uint64_t max_chain_ = kSmallTextureMaxBytes;
    uint32_t default_max_count_ = kSmallTexturePoolMaxCount;
    uint64_t default_max_bytes_ = kSmallTexturePoolMaxBytes;
    uint32_t max_count_ = kSmallTexturePoolMaxCount;
    uint64_t max_bytes_ = kSmallTexturePoolMaxBytes;
    bool enabled_ = true;
    std::atomic<bool> enabled_atomic_{true};
    SmallTexturePoolStats stats_;
};

SmallTexturePool& small_texture_pool() noexcept;
// Chains in (kSmallTextureMaxBytes, kLargeTextureMaxBytes]; only used with gpu_texture_clear().
SmallTexturePool& large_texture_pool() noexcept;
// The pool a chain of `chain_bytes` goes to: the large pool when it takes it and the GPU clear is
// on, else the small pool (which declines, and counts, a chain too big for it).
SmallTexturePool& texture_pool_for(uint64_t chain_bytes) noexcept;

// One zero write: a region of one mip level (depth_or_layers is slices of a 3D mip or array layers).
struct ZeroWrite {
    uint32_t mip = 0;
    uint32_t width = 0;  // texels, rounded up to whole blocks
    uint32_t height = 0;
    uint32_t depth_or_layers = 1;
    uint32_t bytes_per_row = 0;
    uint32_t rows_per_image = 0;  // block rows
    uint64_t bytes = 0;
};

// The writes that overwrite every texel of every mip of a texture with zeros. Empty when the
// block description is unknown (block_w/h/bytes == 0): such a texture must not be pooled.
std::vector<ZeroWrite> zero_fill_writes(uint32_t width, uint32_t height, uint32_t depth_or_layers, uint32_t mips,
                                        bool is_3d, uint32_t block_w, uint32_t block_h, uint32_t block_bytes);

}  // namespace playgta6::gpu
