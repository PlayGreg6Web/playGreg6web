#include "gpu/texture_budget.h"

#include <algorithm>
#include <cstdio>

namespace playgta6::gpu {

uint64_t texture_chain_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t mips, uint32_t block_w,
                             uint32_t block_h, uint32_t block_bytes, uint32_t start_mip) noexcept {
    if (block_w == 0 || block_h == 0 || block_bytes == 0) {
        return 0;
    }
    uint64_t total = 0;
    for (uint32_t m = start_mip; m < std::max(mips, 1u); ++m) {
        const uint32_t w = m >= 32 ? 1u : std::max(1u, width >> m);
        const uint32_t h = m >= 32 ? 1u : std::max(1u, height >> m);
        const uint32_t d = m >= 32 ? 1u : std::max(1u, depth >> m);
        total += uint64_t{(w + block_w - 1) / block_w} * ((h + block_h - 1) / block_h) * d * block_bytes;
    }
    return total;
}

uint32_t cap_floor_for(uint32_t width, uint32_t height, uint32_t mips, uint32_t cap) noexcept {
    if (cap == 0 || mips <= 1) {
        return 0;
    }
    const uint32_t side = std::max(width, height);
    uint32_t drop = 0;
    while (drop + 1 < mips && (drop >= 32 ? 1u : std::max(1u, side >> drop)) > cap) {
        ++drop;
    }
    return drop;
}

uint32_t cap_for_pressure(uint64_t used_bytes, uint64_t budget_bytes, uint32_t base_cap) noexcept {
    uint32_t tier = 0;
    if (budget_bytes != 0) {
        const auto at_least = [&](uint32_t pct) {
            return static_cast<long double>(used_bytes) * 100.0L >= static_cast<long double>(budget_bytes) * pct;
        };
        if (at_least(kTextureCapHighPercent)) {
            tier = kTextureCapHigh;
        } else if (at_least(kTextureCapMediumPercent)) {
            tier = kTextureCapMedium;
        }
    }
    if (tier == 0) {
        return base_cap;
    }
    return base_cap == 0 ? tier : std::min(base_cap, tier);
}

void TextureBudget::set_budget_bytes(uint64_t bytes) noexcept { budget_.store(bytes, std::memory_order_relaxed); }

void TextureBudget::set_base_cap(uint32_t cap) noexcept { base_cap_.store(cap, std::memory_order_relaxed); }

void TextureBudget::reset() noexcept {
    budget_ = 0;
    used_ = 0;
    peak_ = 0;
    base_cap_ = 0;
    last_cap_ = 0;
    cap_changes_ = 0;
    capped_ = 0;
    mips_dropped_ = 0;
    bytes_skipped_ = 0;
    restored_ = 0;
}

void TextureBudget::note_created(uint64_t bytes) noexcept {
    const uint64_t now = used_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    uint64_t peak = peak_.load(std::memory_order_relaxed);
    while (now > peak && !peak_.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
    }
}

void TextureBudget::note_destroyed(uint64_t bytes) noexcept {
    uint64_t cur = used_.load(std::memory_order_relaxed);
    while (!used_.compare_exchange_weak(cur, cur > bytes ? cur - bytes : 0, std::memory_order_relaxed)) {
    }
}

uint32_t TextureBudget::current_cap() noexcept {
    const uint32_t cap = cap_for_pressure(used_.load(std::memory_order_relaxed), budget_.load(std::memory_order_relaxed),
                                          base_cap_.load(std::memory_order_relaxed));
    const uint32_t prev = last_cap_.exchange(cap, std::memory_order_relaxed);
    if (prev != cap) {
        const uint64_t n = cap_changes_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 64) {  // bounded log: a use hovering at a threshold must not flood
            std::fprintf(stderr, "[gpu] texture size cap %u -> %u px (%llu of %llu MiB in use)\n", prev, cap,
                         static_cast<unsigned long long>(used_.load(std::memory_order_relaxed) >> 20),
                         static_cast<unsigned long long>(budget_.load(std::memory_order_relaxed) >> 20));
        }
    }
    return cap;
}

void TextureBudget::note_capped(uint32_t mips, uint64_t bytes) noexcept {
    capped_.fetch_add(1, std::memory_order_relaxed);
    mips_dropped_.fetch_add(mips, std::memory_order_relaxed);
    bytes_skipped_.fetch_add(bytes, std::memory_order_relaxed);
}

TextureBudgetStats TextureBudget::stats() const noexcept {
    TextureBudgetStats s;
    s.budget_bytes = budget_.load(std::memory_order_relaxed);
    s.used_bytes = used_.load(std::memory_order_relaxed);
    s.peak_used_bytes = peak_.load(std::memory_order_relaxed);
    s.base_cap = base_cap_.load(std::memory_order_relaxed);
    s.current_cap = last_cap_.load(std::memory_order_relaxed);
    s.cap_changes = cap_changes_.load(std::memory_order_relaxed);
    s.textures_capped = capped_.load(std::memory_order_relaxed);
    s.mips_dropped = mips_dropped_.load(std::memory_order_relaxed);
    s.bytes_not_uploaded = bytes_skipped_.load(std::memory_order_relaxed);
    s.textures_restored = restored_.load(std::memory_order_relaxed);
    return s;
}

TextureBudget& texture_budget() noexcept {
    static TextureBudget instance;
    return instance;
}

}  // namespace playgta6::gpu
