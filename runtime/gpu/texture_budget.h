#pragma once

// Host GPU memory budget and streamed-texture size cap (backlog 37).
//
// All textures of the title together do not fit a typical host GPU; paging then costs seconds per
// frame or the device is lost. The budget tracker counts the bytes of live host textures against a
// configured budget and derives a size cap from the share in use: below kTextureCapMediumPercent of
// the budget only the configured base cap applies (0 = none), from there up to
// kTextureCapHighPercent the cap is 512 px, above that 256 px. A texture whose largest side is over
// the cap uploads only the mip levels at or below it (the top levels are dropped and the texture's
// views start at the first held level: a lower resolution, not invented content). When memory
// frees and the cap loosens, the dropped levels are uploaded again the next time the texture is
// bound.
//
// A capped texture whose host allocation can be replaced is also *reallocated* at the capped size
// (fewer levels, smaller base; the bridge maps guest mip indices onto the host levels), so the
// dropped levels cost no GPU memory; the budget counts the smaller chain. Loosening reallocates at
// the larger size and re-uploads from guest memory, but only when the restored use still stays
// below the pressure tier the cap came from (cap_with_extra / restore_fits), so a texture does not
// flap between the sizes as its own shrinking moves the use across a threshold.
//
// Pure bookkeeping, thread-safe, no GPU packets.

#include <atomic>
#include <cstdint>

namespace playgta6::gpu {

inline constexpr uint32_t kTextureCapMediumPercent = 70;
inline constexpr uint32_t kTextureCapHighPercent = 85;
inline constexpr uint32_t kTextureCapMedium = 512;
inline constexpr uint32_t kTextureCapHigh = 256;

// Bytes of the mip chain (levels start_mip..mips-1) of a width x height x depth texture whose
// blocks are block_w x block_h texels of block_bytes bytes.
uint64_t texture_chain_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t mips, uint32_t block_w,
                             uint32_t block_h, uint32_t block_bytes, uint32_t start_mip = 0) noexcept;

// Number of top mip levels to drop so no side of the first level that remains exceeds `cap`
// texels. 0 when cap is 0 (no cap) or the texture already fits; never drops the last level.
uint32_t cap_floor_for(uint32_t width, uint32_t height, uint32_t mips, uint32_t cap) noexcept;

// The cap for a given memory use: base_cap (0 = none) below the medium threshold, then the
// tighter of base_cap and the pressure tier. budget_bytes == 0 means no budget (base cap only).
uint32_t cap_for_pressure(uint64_t used_bytes, uint64_t budget_bytes, uint32_t base_cap) noexcept;

struct TextureBudgetStats {
    uint64_t budget_bytes = 0;
    uint64_t used_bytes = 0;
    uint64_t peak_used_bytes = 0;
    uint32_t base_cap = 0;
    uint32_t current_cap = 0;         // 0 = none
    uint64_t cap_changes = 0;         // times the effective cap moved
    uint64_t textures_capped = 0;     // upload passes that dropped at least one top level
    uint64_t mips_dropped = 0;        // top levels not uploaded, summed over those passes
    uint64_t bytes_not_uploaded = 0;  // texel bytes those levels would have taken
    uint64_t textures_restored = 0;   // textures that got dropped levels back after the cap loosened
    uint64_t host_reallocs = 0;       // host textures re-created at another size by the cap
    uint64_t host_bytes_saved = 0;    // bytes the currently shrunk host textures are below full size
};

class TextureBudget {
public:
    void set_budget_bytes(uint64_t bytes) noexcept;
    void set_base_cap(uint32_t cap) noexcept;
    void reset() noexcept;  // tests and per-run reports: counters, use and configuration

    // A host texture of `bytes` was created / destroyed.
    void note_created(uint64_t bytes) noexcept;
    void note_destroyed(uint64_t bytes) noexcept;

    // The cap in force now (re-derived from current use; logs when it moves).
    uint32_t current_cap() noexcept;

    // Records that an upload pass dropped `mips` top levels (`bytes` of texels) of one texture.
    void note_capped(uint32_t mips, uint64_t bytes) noexcept;
    void note_restored() noexcept { restored_.fetch_add(1, std::memory_order_relaxed); }

    // The cap that would be in force if `extra_bytes` more were in use (a restore that grows a texture).
    uint32_t cap_with_extra(uint64_t extra_bytes) const noexcept;
    // True when growing a width x height x mips texture to start at level `target_floor`, costing
    // `extra_bytes` more, still leaves a cap that allows that floor (hysteresis for restores).
    bool restore_fits(uint32_t width, uint32_t height, uint32_t mips, uint32_t target_floor,
                      uint64_t extra_bytes) const noexcept;

    // A host texture was re-created at another size: `saved_delta` is the change of its bytes below
    // full size (positive = shrunk further). The used bytes move by the same amount.
    void note_host_realloc(int64_t saved_delta) noexcept;

    // A shrunk texture goes away: its caller then note_destroyed()s the full size, so the saved bytes
    // are counted back in first (net: the shrunk bytes leave the use).
    void note_shrunk_released(uint64_t saved_bytes) noexcept;

    TextureBudgetStats stats() const noexcept;

private:
    std::atomic<uint64_t> budget_{0};
    std::atomic<uint64_t> used_{0};
    std::atomic<uint64_t> peak_{0};
    std::atomic<uint32_t> base_cap_{0};
    std::atomic<uint32_t> last_cap_{0};
    std::atomic<uint64_t> cap_changes_{0};
    std::atomic<uint64_t> capped_{0};
    std::atomic<uint64_t> mips_dropped_{0};
    std::atomic<uint64_t> bytes_skipped_{0};
    std::atomic<uint64_t> restored_{0};
    std::atomic<uint64_t> reallocs_{0};
    std::atomic<uint64_t> saved_{0};
};

TextureBudget& texture_budget() noexcept;

}  // namespace playgta6::gpu
