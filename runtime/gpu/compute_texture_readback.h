#pragma once

// Read-back of compute-written float textures into guest memory in console layout (backlog 8).
//
// A title that runs a compute pass into a small float texture (an ocean height / displacement
// map) and then reads the texels on the CPU finds them in the texture's own guest memory, tiled
// the way the console GPU would have left them. The host GPU wrote its own texture instead, so
// the runtime reads the texture back (the G66 ReadbackTexture packet, into tight linear rows in
// a scratch buffer) and re-tiles the rows into the guest surface with gfx10::tile_surface_subresource,
// the exact inverse of the detiler the upload path uses.
//
// Only real GPU results are ever written: the scratch buffer is pre-filled with a sentinel, and a
// read-back the consumer reports complete without having written it (a dropped frame, a failed
// map) leaves the sentinel and is discarded (was_delivered).
// Pure C++20, no GPU or guest-memory dependency, so the unit test runs on every host.

#include <cstddef>
#include <cstdint>

#include "gpu/gfx10_detile.h"

namespace playgta6::gpu::compute_readback {

// Largest tight linear size of one read-back (a 256x256 R32_FLOAT map is 256 KiB).
constexpr uint64_t kMaxLinearBytes = 1ull << 20;

// Texture facts the plan needs (a tracked guest texture).
struct Target {
    uint32_t dxgi_format = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t array_size = 1;
    uint32_t mip_levels = 1;
    uint32_t d3d12_layout = 0;  // console layout: 0x100 | swizzle mode
    bool is_2d = false;         // a single-sample 2D (or 2D array) texture
};

struct Plan {
    bool ok = false;
    const char* reason = nullptr;  // why not, when !ok
    gfx10::TextureLayoutDesc desc{};
    uint32_t mip = 0;
    uint32_t mip_width = 0;
    uint32_t mip_height = 0;
    uint32_t row_bytes = 0;        // tight row bytes of the mip
    uint64_t linear_bytes = 0;     // row_bytes * mip_height
    uint64_t surface_bytes = 0;    // the whole console surface from its base address
};

// Float colour formats a compute pass writes through a storage view and the title reads back
// (R32G32B32A32 / R16G16B16A16 / R32G32 / R16G16 / R32 / R16 FLOAT).
bool is_readback_float_format(uint32_t dxgi_format) noexcept;

// Eligibility and geometry of reading back mip `mip` (array slice 0). !ok when the texture is not a
// single-sample 2D console-layout float texture, the mip is out of range, or it exceeds kMaxLinearBytes.
Plan plan_readback(const Target& target, uint32_t mip) noexcept;

// The scratch pre-fill. Real float data is never all-sentinel (every texel a NaN with all bits set).
constexpr uint32_t kSentinelWord = 0xFFFFFFFFu;
void fill_sentinel(uint8_t* buf, size_t bytes) noexcept;
// True when the consumer wrote at least one byte of the scratch (it is not entirely sentinel).
bool was_delivered(const uint8_t* buf, size_t bytes) noexcept;

// Re-tiles the tight rows of `linear` (plan.linear_bytes) into the console surface (`surface`,
// surface_size >= plan.surface_bytes). false (and nothing written) when a size is short or the
// scratch was never delivered. Other bytes of the surface are kept.
bool retile_into_guest(const Plan& plan, const uint8_t* linear, size_t linear_bytes, uint8_t* surface,
                       size_t surface_size) noexcept;

}  // namespace playgta6::gpu::compute_readback
