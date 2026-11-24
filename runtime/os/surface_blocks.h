#pragma once

#include <cstdint>

// Surface sizes in the GPU's 64 KiB-block layout (the console GPU generation's "64 KiB" swizzle
// modes), for the cases this file covers: a single-sample 2D surface, one mip level, one array
// slice, of a format with 1x1 elements (no block compression) of 1, 2, 4, 8 or 16 bytes.
//
// Rule: the surface is padded to whole 64 KiB blocks in both dimensions. A block holds
// 65,536 / bytes-per-element elements and is as square as possible, the width taking the extra
// factor of two:
//   1 byte -> 256 x 256, 2 -> 256 x 128, 4 -> 128 x 128, 8 -> 128 x 64, 16 -> 64 x 64.
// The surface is 64 KiB aligned and its size is a whole number of blocks.
//
// Not covered (callers keep their own sizing for these): surfaces smaller than one block in
// either dimension (the hardware then picks smaller block modes, whose choice is not modelled),
// MSAA, mip chains, arrays, 3D, depth/stencil planes and block-compressed formats.

namespace playgta6::os::surface {

inline constexpr uint64_t kBlockBytes = 64u * 1024u;

struct BlockDims {
    uint32_t width = 0;   // elements
    uint32_t height = 0;  // elements
};

// Block dimensions for an element size of 1, 2, 4, 8 or 16 bytes; false for any other size.
inline constexpr bool block64k_dims(uint32_t bytes_per_element, BlockDims* out) {
    BlockDims d{};
    switch (bytes_per_element) {
        case 1: d = {256, 256}; break;
        case 2: d = {256, 128}; break;
        case 4: d = {128, 128}; break;
        case 8: d = {128, 64}; break;
        case 16: d = {64, 64}; break;
        default: return false;
    }
    if (out != nullptr) {
        *out = d;
    }
    return true;
}

struct Block64kLayout {
    uint32_t padded_width = 0;   // elements, a multiple of the block width
    uint32_t padded_height = 0;  // elements, a multiple of the block height
    uint64_t row_pitch_bytes = 0;
    uint64_t size_bytes = 0;     // a multiple of kBlockBytes
    uint64_t alignment = kBlockBytes;
};

// The layout of a covered surface (see above); false when the surface is not covered.
inline constexpr bool block64k_layout_2d(uint32_t width, uint32_t height, uint32_t bytes_per_element,
                                         Block64kLayout* out) {
    BlockDims d{};
    if (!block64k_dims(bytes_per_element, &d) || width < d.width || height < d.height) {
        return false;
    }
    Block64kLayout l{};
    l.padded_width = (width + d.width - 1) / d.width * d.width;
    l.padded_height = (height + d.height - 1) / d.height * d.height;
    l.row_pitch_bytes = static_cast<uint64_t>(l.padded_width) * bytes_per_element;
    l.size_bytes = l.row_pitch_bytes * l.padded_height;
    if (out != nullptr) {
        *out = l;
    }
    return true;
}

}  // namespace playgta6::os::surface
