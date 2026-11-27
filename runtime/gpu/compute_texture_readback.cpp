#include "gpu/compute_texture_readback.h"

#include <algorithm>
#include <cstring>

namespace playgta6::gpu::compute_readback {

bool is_readback_float_format(uint32_t f) noexcept {
    switch (f) {
        case 2:   // R32G32B32A32_FLOAT
        case 10:  // R16G16B16A16_FLOAT
        case 16:  // R32G32_FLOAT
        case 34:  // R16G16_FLOAT
        case 41:  // R32_FLOAT
        case 54:  // R16_FLOAT
            return true;
        default:
            return false;
    }
}

Plan plan_readback(const Target& t, uint32_t mip) noexcept {
    Plan p;
    if (!t.is_2d) {
        p.reason = "not a single-sample 2D texture";
        return p;
    }
    if (!is_readback_float_format(t.dxgi_format)) {
        p.reason = "not a float colour format";
        return p;
    }
    if (t.d3d12_layout < gfx10::kConsoleLinearLayout ||
        gfx10::decode_d3d12_layout(t.d3d12_layout) == gfx10::SwizzleMode::Unsupported) {
        p.reason = "not a console layout";
        return p;
    }
    const gfx10::FormatGeometry geo = gfx10::format_geometry(t.dxgi_format);
    if (!geo.supported || geo.compressed || t.width == 0 || t.height == 0 || mip >= std::max(1u, t.mip_levels)) {
        p.reason = "unsupported geometry";
        return p;
    }
    p.desc.dxgi_format = t.dxgi_format;
    p.desc.width = t.width;
    p.desc.height = t.height;
    p.desc.depth = 1;
    p.desc.array_size = std::max(1u, t.array_size);
    p.desc.mip_levels = std::max(1u, t.mip_levels);
    p.desc.d3d12_layout = t.d3d12_layout;
    p.mip = mip;
    p.mip_width = std::max(1u, t.width >> mip);
    p.mip_height = std::max(1u, t.height >> mip);
    p.row_bytes = p.mip_width * geo.bytes_per_block;
    p.linear_bytes = uint64_t{p.row_bytes} * p.mip_height;
    if (p.linear_bytes > kMaxLinearBytes) {
        p.reason = "larger than the read-back cap";
        return p;
    }
    p.surface_bytes = gfx10::surface_bytes(p.desc);
    if (p.surface_bytes == 0) {
        p.reason = "no surface placement";
        return p;
    }
    p.ok = true;
    return p;
}

void fill_sentinel(uint8_t* buf, size_t bytes) noexcept {
    if (buf != nullptr) {
        std::memset(buf, 0xFF, bytes);
    }
}

bool was_delivered(const uint8_t* buf, size_t bytes) noexcept {
    if (buf == nullptr) {
        return false;
    }
    for (size_t i = 0; i + 4 <= bytes; i += 4) {
        uint32_t w;
        std::memcpy(&w, buf + i, 4);
        if (w != kSentinelWord) {
            return true;
        }
    }
    return false;
}

bool retile_into_guest(const Plan& plan, const uint8_t* linear, size_t linear_bytes, uint8_t* surface,
                       size_t surface_size) noexcept {
    if (!plan.ok || linear == nullptr || surface == nullptr || linear_bytes < plan.linear_bytes ||
        surface_size < plan.surface_bytes || !was_delivered(linear, linear_bytes)) {
        return false;
    }
    return gfx10::tile_surface_subresource(plan.desc, plan.mip, 0u, linear, linear_bytes, surface, surface_size);
}

}  // namespace playgta6::gpu::compute_readback
