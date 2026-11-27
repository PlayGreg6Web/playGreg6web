#include "gpu/small_texture_pool.h"

#include <algorithm>

namespace playgta6::gpu {

uint32_t SmallTexturePool::alloc_node_locked() {
    if (!free_nodes_.empty()) {
        const uint32_t idx = free_nodes_.back();
        free_nodes_.pop_back();
        return idx;
    }
    const uint32_t idx = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back();
    if (free_nodes_.capacity() < nodes_.size()) {
        free_nodes_.reserve(nodes_.capacity());
    }
    return idx;
}

void SmallTexturePool::unlink_lru_locked(uint32_t idx) noexcept {
    Node& n = nodes_[idx];
    if (n.lru_prev != kInvalidNode) {
        nodes_[n.lru_prev].lru_next = n.lru_next;
    } else {
        lru_head_ = n.lru_next;
    }
    if (n.lru_next != kInvalidNode) {
        nodes_[n.lru_next].lru_prev = n.lru_prev;
    } else {
        lru_tail_ = n.lru_prev;
    }
    n.lru_prev = kInvalidNode;
    n.lru_next = kInvalidNode;
}

void SmallTexturePool::unlink_and_free_locked(uint32_t idx) noexcept {
    Node& n = nodes_[idx];
    unlink_lru_locked(idx);
    if (n.bucket_prev != kInvalidNode) {
        nodes_[n.bucket_prev].bucket_next = n.bucket_next;
    } else {
        auto it = buckets_.find(n.shape);
        if (it != buckets_.end()) {
            if (n.bucket_next != kInvalidNode) {
                it->second = n.bucket_next;
            } else {
                buckets_.erase(it);
            }
        }
    }
    if (n.bucket_next != kInvalidNode) {
        nodes_[n.bucket_next].bucket_prev = n.bucket_prev;
    }
    n.bucket_prev = kInvalidNode;
    n.bucket_next = kInvalidNode;
    n.active = false;
    free_nodes_.push_back(idx);
}

void SmallTexturePool::set_limits(uint32_t max_count, uint64_t max_bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    max_count_ = max_count;
    max_bytes_ = max_bytes;
}

void SmallTexturePool::set_enabled(bool enabled) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
    enabled_atomic_.store(enabled, std::memory_order_relaxed);
}

void SmallTexturePool::reset() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    nodes_.clear();
    free_nodes_.clear();
    buckets_.clear();
    lru_head_ = kInvalidNode;
    lru_tail_ = kInvalidNode;
    stats_ = SmallTexturePoolStats{};
    max_count_ = default_max_count_;
    max_bytes_ = default_max_bytes_;
    enabled_ = true;
    enabled_atomic_.store(true, std::memory_order_relaxed);
}

uint32_t SmallTexturePool::acquire(const SmallTextureShape& shape, uint64_t chain_bytes) noexcept {
    if (!accepts(chain_bytes) || !enabled_atomic_.load(std::memory_order_relaxed)) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        return 0;
    }
    // Newest first (LIFO per shape): the most recently parked texture is the most likely still resident.
    auto it = buckets_.find(shape);
    if (it != buckets_.end()) {
        const uint32_t idx = it->second;
        Node& n = nodes_[idx];
        const uint32_t id = n.id;
        const uint64_t bytes = n.bytes;
        if (n.bucket_next != kInvalidNode) {
            nodes_[n.bucket_next].bucket_prev = kInvalidNode;
            it->second = n.bucket_next;
        } else {
            buckets_.erase(it);
        }
        unlink_lru_locked(idx);
        n.bucket_prev = kInvalidNode;
        n.bucket_next = kInvalidNode;
        n.active = false;
        free_nodes_.push_back(idx);
        stats_.hits += 1;
        stats_.bytes_reused += bytes;
        stats_.parked_bytes -= bytes;
        stats_.parked_count -= 1;
        return id;
    }
    stats_.misses += 1;
    return 0;
}

bool SmallTexturePool::release(const SmallTextureShape& shape, uint32_t id, uint64_t chain_bytes,
                               std::vector<uint32_t>* evicted, std::vector<uint64_t>* evicted_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || id == 0 || !accepts(chain_bytes) || max_count_ == 0 || chain_bytes > max_bytes_) {
        stats_.rejected += 1;
        return false;
    }
    while (lru_head_ != kInvalidNode &&
           (stats_.parked_count >= max_count_ || stats_.parked_bytes + chain_bytes > max_bytes_)) {
        const uint32_t old_idx = lru_head_;
        const uint32_t old_id = nodes_[old_idx].id;
        const uint64_t old_bytes = nodes_[old_idx].bytes;
        if (evicted != nullptr) {
            evicted->push_back(old_id);
        }
        if (evicted_bytes != nullptr) {
            evicted_bytes->push_back(old_bytes);
        }
        stats_.parked_bytes -= old_bytes;
        stats_.parked_count -= 1;
        stats_.evicted += 1;
        unlink_and_free_locked(old_idx);
    }
    const uint32_t idx = alloc_node_locked();
    Node& n = nodes_[idx];
    n.shape = shape;
    n.id = id;
    n.bytes = chain_bytes;
    n.active = true;
    // Append to global LRU tail (newest).
    n.lru_prev = lru_tail_;
    n.lru_next = kInvalidNode;
    if (lru_tail_ != kInvalidNode) {
        nodes_[lru_tail_].lru_next = idx;
    } else {
        lru_head_ = idx;
    }
    lru_tail_ = idx;
    // Push onto shape bucket LIFO head.
    auto [it, inserted] = buckets_.emplace(shape, idx);
    if (inserted) {
        n.bucket_prev = kInvalidNode;
        n.bucket_next = kInvalidNode;
    } else {
        const uint32_t old_head = it->second;
        n.bucket_prev = kInvalidNode;
        n.bucket_next = old_head;
        nodes_[old_head].bucket_prev = idx;
        it->second = idx;
    }
    stats_.parked_total += 1;
    stats_.parked_bytes += chain_bytes;
    stats_.peak_parked_bytes = std::max(stats_.peak_parked_bytes, stats_.parked_bytes);
    stats_.parked_count += 1;
    return true;
}

void SmallTexturePool::drain(uint64_t owner, std::vector<uint32_t>* ids, std::vector<uint64_t>* bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t curr = lru_head_;
    while (curr != kInvalidNode) {
        const uint32_t next = nodes_[curr].lru_next;
        if (nodes_[curr].shape.owner == owner) {
            if (ids != nullptr) {
                ids->push_back(nodes_[curr].id);
            }
            if (bytes != nullptr) {
                bytes->push_back(nodes_[curr].bytes);
            }
            stats_.parked_bytes -= nodes_[curr].bytes;
            stats_.parked_count -= 1;
            unlink_and_free_locked(curr);
        }
        curr = next;
    }
}

size_t SmallTexturePool::trim(uint32_t max_count, uint64_t max_bytes, std::vector<uint32_t>* evicted,
                              std::vector<uint64_t>* evicted_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    while (lru_head_ != kInvalidNode && (stats_.parked_count > max_count || stats_.parked_bytes > max_bytes)) {
        const uint32_t old_idx = lru_head_;
        const uint32_t old_id = nodes_[old_idx].id;
        const uint64_t old_bytes = nodes_[old_idx].bytes;
        if (evicted != nullptr) {
            evicted->push_back(old_id);
        }
        if (evicted_bytes != nullptr) {
            evicted_bytes->push_back(old_bytes);
        }
        stats_.parked_bytes -= old_bytes;
        stats_.parked_count -= 1;
        stats_.evicted += 1;
        unlink_and_free_locked(old_idx);
        ++count;
    }
    return count;
}

void SmallTexturePool::note_create(uint64_t chain_bytes, uint64_t ns) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.creates += 1;
    stats_.bytes_created += chain_bytes;
    stats_.create_ns += ns;
}

void SmallTexturePool::note_address_miss(bool recreate) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.address_misses += 1;
    stats_.address_recreates += recreate ? 1u : 0u;
}

void SmallTexturePool::note_address_eviction() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.address_evictions += 1;
}

size_t SmallTexturePool::distinct_shapes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buckets_.size();
}

SmallTexturePoolStats SmallTexturePool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

SmallTexturePool& small_texture_pool() noexcept {
    static SmallTexturePool pool;
    return pool;
}

SmallTexturePool& large_texture_pool() noexcept {
    static SmallTexturePool pool(kSmallTextureMaxBytes, kLargeTextureMaxBytes, kLargeTexturePoolMaxCount,
                                 kLargeTexturePoolMaxBytes);
    return pool;
}

namespace {
std::atomic<bool> g_gpu_texture_clear{false};
std::atomic<bool> g_gpu_texture_copy{false};
}  // namespace

void set_gpu_texture_copy(bool on) noexcept {
    g_gpu_texture_copy.store(on, std::memory_order_relaxed);
}

bool gpu_texture_copy() noexcept {
    return g_gpu_texture_copy.load(std::memory_order_relaxed);
}

void set_gpu_texture_clear(bool on) noexcept {
    g_gpu_texture_clear.store(on, std::memory_order_relaxed);
}

bool gpu_texture_clear() noexcept {
    return g_gpu_texture_clear.load(std::memory_order_relaxed);
}

SmallTexturePool& texture_pool_for(uint64_t chain_bytes) noexcept {
    SmallTexturePool& large = large_texture_pool();
    return gpu_texture_clear() && large.accepts(chain_bytes) ? large : small_texture_pool();
}

std::vector<ZeroWrite> zero_fill_writes(uint32_t width, uint32_t height, uint32_t depth_or_layers, uint32_t mips,
                                        bool is_3d, uint32_t block_w, uint32_t block_h, uint32_t block_bytes) {
    std::vector<ZeroWrite> out;
    if (block_w == 0 || block_h == 0 || block_bytes == 0 || width == 0 || height == 0 || mips == 0) {
        return out;
    }
    out.reserve(mips);
    const uint32_t depth = std::max<uint32_t>(1u, depth_or_layers);
    for (uint32_t m = 0; m < mips; ++m) {
        const uint32_t w = std::max<uint32_t>(1u, width >> m);
        const uint32_t h = std::max<uint32_t>(1u, height >> m);
        const uint32_t d = is_3d ? std::max<uint32_t>(1u, depth >> m) : depth;
        ZeroWrite z;
        z.mip = m;
        const uint32_t bx = (w + block_w - 1u) / block_w;
        const uint32_t by = (h + block_h - 1u) / block_h;
        z.width = bx * block_w;
        z.height = by * block_h;
        z.depth_or_layers = d;
        z.bytes_per_row = bx * block_bytes;
        z.rows_per_image = by;
        z.bytes = uint64_t{z.bytes_per_row} * by * d;
        out.push_back(z);
    }
    return out;
}

}  // namespace playgta6::gpu
