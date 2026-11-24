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
    // 2D array slices (depth must be 1 when > 1). Thin layouts repeat the whole mip chain per
    // slice (addrlib: slice stride = mip-chain size), so the surface is chain x array_size.
    uint32_t array_size = 1;
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

// Whole-surface view (all mips and slices from the resource's base address), following AMD
// addrlib's gfx10 placement (Gfx10Lib::ComputeSurfaceInfoMacroTiled / MicroTiled and
// ComputeSurfaceAddrFromCoordMacroTiled): mip sizes in elements are rounded up per level; in
// 4KB/64KB modes the mip tail (one block holding every mip that fits) comes first and the larger
// mips follow smallest-first, so mip 0 sits last, or inside the tail block at a fixed coordinate
// when the whole chain fits there; 256B modes store the mips smallest-first without a tail.
// Linear layouts keep mip 0 at offset 0 (only mip 0 is supported for them): the real linear mip
// rule is not established. addrlib's gfx10 ADDR_SW_LINEAR chain (256 B pitch, mips
// smallest-first) is what the sampler needs, but it disagrees with the tight allocation answer
// the title has been given so far (#665).
struct MipPlacement {
    bool ok = false;
    const char* error = nullptr;
    uint64_t surface_bytes = 0;       // bytes of the whole surface from the base address
    uint64_t macro_block_offset = 0;  // offset of the mip's first block (0 for mips in the tail)
    bool in_tail = false;
    uint32_t tail_x = 0;  // element coordinate of the mip inside the tail block
    uint32_t tail_y = 0;
    uint32_t tail_z = 0;
    // addrlib ADDR2_MIP_INFO-style figures for reporting a layout (XG GetResourceLayout):
    uint32_t pitch_elements = 0;   // padded pitch (tail mips: the tail region, halved per level)
    uint32_t height_elements = 0;  // padded height
    uint64_t slice_bytes = 0;      // pitch * height * bytes per element
    uint64_t size_bytes = 0;       // slice_bytes * padded depth (single mip: the whole surface)
    uint64_t offset_bytes = 0;     // ADDR2_MIP_INFO::offset (tail mips: tail offset x depth blocks)
    uint64_t block_bytes = 0;      // swizzle block size (base alignment)
};
MipPlacement mip_placement(const TextureLayoutDesc& desc, uint32_t mip) noexcept;

// Bytes of the whole surface (desc.mip_levels mips, all slices) from the base address; 0 if unsupported.
size_t surface_bytes(const TextureLayoutDesc& desc) noexcept;

// Detile mip `mip` from the whole surface starting at the resource's base address (see
// MipPlacement). Output rows are tight, with D3D mip dimensions (max(1, size >> mip)).
DetileResult detile_from_surface(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* surface,
                                 size_t surface_bytes, std::vector<uint8_t>& linear_out);

// One subresource of the whole surface: mip `mip` of array slice `array_slice` (3D: every depth
// slice of the mip; array_slice must be 0). Tail mips are read inside the tail block at their
// addrlib coordinates. Linear layouts: mip 0 only; array slices only when the slice stride is the
// same under every candidate linear rule (one mip, row bytes a multiple of 256 or an explicit
// pitch_bytes).
DetileResult detile_surface_subresource(const TextureLayoutDesc& desc, uint32_t mip, uint32_t array_slice,
                                        const uint8_t* surface, size_t surface_bytes,
                                        std::vector<uint8_t>& linear_out);

// Inverse of detile_surface_subresource for tiled layouts; other bytes of `surface` are kept.
bool tile_surface_subresource(const TextureLayoutDesc& desc, uint32_t mip, uint32_t array_slice,
                              const uint8_t* linear_src, size_t linear_src_bytes, uint8_t* surface,
                              size_t surface_size);

// Inverse of detile_from_surface: writes the tight rows of mip `mip` (D3D mip dimensions) into
// the whole surface at `surface` (surface_size >= surface_bytes(desc)); other bytes are kept.
// Tiled layouts only.
bool tile_into_surface(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* linear_src,
                       size_t linear_src_bytes, uint8_t* surface, size_t surface_size);

// Inverse of detile_from_surface for synthetic tests: writes mip `mip` into `surface` (resized to
// surface_bytes(desc) if smaller; other bytes are kept).
bool tile_into_surface_for_test(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* linear_src,
                                size_t linear_src_bytes, std::vector<uint8_t>& surface);

// Inverse encoder for synthetic round-trip unit tests: tiles tightly-packed linear subresource
// bytes into the exact tiled/pitched GPU layout expected by detile_subresource.
bool tile_subresource_for_test(const TextureLayoutDesc& desc, uint32_t mip, const uint8_t* linear_src,
                               size_t linear_src_bytes, std::vector<uint8_t>& tiled_out);

}  // namespace playgta6::gpu::gfx10
