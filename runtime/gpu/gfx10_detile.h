#pragma once

// AMD GFX10 / RDNA2 console texture detiler (G49a): converts tiled Xbox Series texture memory
// into tightly-packed linear texels/blocks for WebGPU WriteTexture upload.
//
// The _S and _D equations (256B/4KB/64KB, thin) and the thick 3D _S equations (4KB/64KB) are AMD
// addrlib's public gfx10 RB+ pattern tables (gfx10SwizzlePattern.h). The _R / VAR modes and the
// _X pipe-bank XOR are not from addrlib (gfx10 defines no non-XOR _R or VAR_S/D pattern, and the
// XOR term depends on the pipe configuration); they stay unverified.
// Pure C++20, zero external dependencies, Wasm64-safe, bounds-checked on every read.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace playgta6::gpu::gfx10 {

// Documented AMD GFX10 SWIZZLE_MODE_ENUM values (5-bit hardware sw_mode field).
enum class SwizzleMode : uint8_t {
    Linear = 0,
    Sw256BS = 1,
    Sw256BD = 2,
    Sw256BR = 3,
    Sw4KBS = 5,
    Sw4KBD = 6,
    Sw4KBR = 7,
    Sw64KBS = 9,
    Sw64KBD = 10,
    Sw64KBR = 11,
    SwVarS = 13,
    SwVarD = 14,
    SwVarR = 15,
    Sw64KBSX = 25,
    Sw64KBDX = 26,
    Sw64KBRX = 27,
    SwVarSX = 29,
    SwVarDX = 30,
    SwVarRX = 31,
    Unsupported = 255,
};

struct TextureLayoutDesc {
    uint32_t dxgi_format = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t d3d12_layout = 0;
    // Optional explicit source row pitch in bytes for the subresource (0 = default tight/tiled pitch).
    // For Linear mode, pitch_bytes must be >= row_width_bytes when non-zero.
    // For tiled modes, pitch_bytes must be a multiple of bytes_per_block and >= row_width_bytes when non-zero.
    uint32_t pitch_bytes = 0;
};

struct FormatGeometry {
    uint32_t block_width = 0;
    uint32_t block_height = 0;
    uint32_t bytes_per_block = 0;
    bool compressed = false;
    bool supported = false;
};

struct DetileResult {
    bool ok = false;
    uint32_t row_pitch_bytes = 0;
    uint32_t rows = 0;
    uint32_t slices = 0;
    size_t tiled_bytes_consumed = 0;
    const char* error = nullptr;
};

// Decode D3D12_TEXTURE_LAYOUT values:
//   0 (UNKNOWN) / 1 (ROW_MAJOR) -> SwizzleMode::Linear
//   0x100..0x11F (Xbox 0x100 | swizzle_mode) -> corresponding GFX10 SwizzleMode if documented
//   All unrecognized or unsupported layout values -> SwizzleMode::Unsupported.
SwizzleMode decode_d3d12_layout(uint32_t d3d12_layout) noexcept;

// Block dimensions and byte size for supported DXGI formats (1, 2, 4, 8, or 16 bytes per block).
FormatGeometry format_geometry(uint32_t dxgi_format) noexcept;

// True if `mode` is supported by the detiler (Linear and all documented 256B/4KB/64KB/Var modes).
bool is_swizzle_supported(SwizzleMode mode) noexcept;

// Compute the byte offset of 2D block/texel (bx, by) in a surface with `bytes_per_block` (1,2,4,8,16)
// and `pitch_in_blocks` stride. Returns SIZE_MAX if `mode` or `bytes_per_block` is invalid.
size_t tiled_byte_offset_2d(SwizzleMode mode, uint32_t bx, uint32_t by, uint32_t bytes_per_block,
                            uint32_t pitch_in_blocks) noexcept;

// Source bytes detile_subresource needs for subresource `mip` (whole tiles; for Linear the pitched
// rows). 0 if the format, layout or dimensions are not supported.
size_t tiled_footprint_bytes(const TextureLayoutDesc& desc, uint32_t mip) noexcept;

// Detile a single subresource (mip level `mip`) from `src[0..src_bytes)` into tightly-packed linear
// rows in `linear_out`. Supports 2D (depth == 1) and 3D (depth > 1) textures in linear and tiled
// GFX10 modes. Never reads past `src_bytes`; returns ok = false with a descriptive `error` on any
// unsupported format/layout, invalid dimensions, or truncated source buffer.
DetileResult detile_subresource(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* src,
                                size_t src_bytes, std::vector<uint8_t>& linear_out);

// Inverse encoder for synthetic round-trip unit tests: tiles tightly-packed linear subresource
// bytes into the exact tiled/pitched GPU layout expected by detile_subresource.
bool tile_subresource_for_test(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* linear_src,
                               size_t linear_src_bytes, std::vector<uint8_t>& tiled_out);

}  // namespace playgta6::gpu::gfx10
