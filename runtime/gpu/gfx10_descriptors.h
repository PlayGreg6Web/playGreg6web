#pragma once

// Hardware descriptor words of the console GPU (AMD gfx10 / RDNA2 family): the four-dword buffer
// resource (V#), the eight-dword image resource (T#) and the four-dword sampler state (S#).
//
// Source of every field: the public AMD driver definitions (GPUOpen PAL,
// src/core/hw/gfxip/gfx9/chip/gfx10_sq_ko_reg.h: unions sq_buf_rsrc_t, sq_img_rsrc_t and
// sq_img_samp_t with their "gfx10" variants; enumerations from gfx9_plus_merged_enum.h: BUF_FMT,
// IMG_FMT, SQ_SEL_XYZW01, SQ_RSRC_IMG_TYPE, SQ_OOB_SELECT, SWIZZLE_MODE_ENUM, SQ_TEX_CLAMP,
// SQ_TEX_XY_FILTER, SQ_TEX_MIP_FILTER, SQ_TEX_Z_FILTER, SQ_TEX_DEPTH_COMPARE, SQ_TEX_ANISO_RATIO,
// SQ_IMG_FILTER_TYPE, SQ_TEX_BORDER_COLOR). Fixed-point encodings that the headers give only as
// field widths follow how the open-source drivers fill them: image and metadata addresses are
// stored as address >> 8 (256-byte units), LODs as unsigned 4.8 fixed point, the LOD bias as
// signed 6.8 fixed point.
//
// Layout summary (dword.bits):
//   V#  0.0-31 base[31:0]; 1.0-15 base[47:32], 1.16-29 stride, 1.30 cache_swizzle, 1.31
//       swizzle_enable; 2 num_records; 3.0-11 dst_sel x/y/z/w (3 bits each), 3.12-18 format (BUF_FMT),
//       3.21-22 index_stride, 3.23 add_tid_enable, 3.24 resource_level (1 on gfx10), 3.28-29
//       oob_select, 3.30-31 type (0 = buffer).
//   T#  0.0-31 + 1.0-7 base >> 8 (40 bits); 1.8-19 min_lod; 1.20-28 format (IMG_FMT); 1.30-31 +
//       2.0-13 width - 1; 2.14-29 height - 1; 2.31 resource_level; 3.0-11 dst_sel; 3.12-15
//       base_level; 3.16-19 last_level; 3.20-24 sw_mode; 3.25-27 bc_swizzle; 3.28-31 type (8..15);
//       4.0-15 depth - 1 (3D) or last array slice; 4.16-31 base_array; 5.0-3 array_pitch; 5.4-7
//       max_mip; 5.8-19 min_lod_warn; 5.20-22 perf_mod; 5.23 corner_samples; 5.25 lod_hdw_cnt_en;
//       5.26 prt_default; 5.31 big_page; 6.0-7 counter_bank_id; 6.10 iterate_256; 6.15-16
//       max_uncompressed_block_size; 6.17-18 max_compressed_block_size; 6.19 meta_pipe_aligned; 6.20
//       write_compress_enable; 6.21 compression_en; 6.22 alpha_is_on_msb; 6.23 color_transform;
//       6.24-31 + 7.0-31 meta_data_address >> 8 (40 bits).
//   S#  0.0-8 clamp x/y/z; 0.9-11 max_aniso_ratio; 0.12-14 depth_compare_func; 0.15
//       force_unnormalized; 0.16-18 aniso_threshold; 0.19 mc_coord_trunc; 0.20 force_degamma;
//       0.21-26 aniso_bias; 0.27 trunc_coord; 0.28 disable_cube_wrap; 0.29-30 filter_mode; 0.31
//       skip_degamma; 1.0-11 min_lod; 1.12-23 max_lod; 1.24-27 perf_mip; 1.28-31 perf_z; 2.0-13
//       lod_bias; 2.14-19 lod_bias_sec; 2.20-21 xy_mag_filter; 2.22-23 xy_min_filter; 2.24-25
//       z_filter; 2.26-27 mip_filter; 2.28 mip_point_preclamp; 2.29 aniso_override; 2.30 blend_prt;
//       2.31 deriv_adjust_en; 3.0-11 border_color_ptr; 3.30-31 border_color_type.
// A buffer and an image descriptor are told apart by dword 3 bit 31 (an image type is 8..15, a
// buffer type is 0).

#include <array>
#include <cstdint>

#include "gpu/descriptor_heap.h"

namespace playgta6::gpu::gfx10 {

inline constexpr uint32_t kBufferDwords = 4;
inline constexpr uint32_t kImageDwords = 8;
inline constexpr uint32_t kSamplerDwords = 4;

// SQ_SEL_XYZW01.
namespace Sel {
inline constexpr uint8_t Zero = 0;
inline constexpr uint8_t One = 1;
inline constexpr uint8_t X = 4;
inline constexpr uint8_t Y = 5;
inline constexpr uint8_t Z = 6;
inline constexpr uint8_t W = 7;
}  // namespace Sel
using Swizzle = std::array<uint8_t, 4>;
inline constexpr Swizzle kIdentitySwizzle = {Sel::X, Sel::Y, Sel::Z, Sel::W};

// SQ_RSRC_IMG_TYPE.
namespace ImageType {
inline constexpr uint32_t Tex1D = 8;
inline constexpr uint32_t Tex2D = 9;
inline constexpr uint32_t Tex3D = 10;
inline constexpr uint32_t Cube = 11;
inline constexpr uint32_t Tex1DArray = 12;
inline constexpr uint32_t Tex2DArray = 13;
inline constexpr uint32_t Tex2DMsaa = 14;
inline constexpr uint32_t Tex2DMsaaArray = 15;
}  // namespace ImageType

// SQ_OOB_SELECT.
namespace Oob {
inline constexpr uint32_t IndexAndOffset = 0;
inline constexpr uint32_t IndexOnly = 1;
inline constexpr uint32_t NumRecords0 = 2;
inline constexpr uint32_t Complete = 3;  // raw: num_records counts bytes
}  // namespace Oob

// BUF_FMT / IMG_FMT values used by the layer (both enumerations agree on 0..77).
namespace Fmt {
inline constexpr uint32_t Invalid = 0;
inline constexpr uint32_t F32 = 22;  // 32_FLOAT
}  // namespace Fmt

struct BufferResource {
    uint64_t base = 0;  // 48-bit byte address
    uint32_t stride = 0;
    bool cache_swizzle = false;
    bool swizzle_enable = false;
    uint32_t num_records = 0;
    Swizzle dst_sel = kIdentitySwizzle;
    uint32_t format = Fmt::Invalid;  // BUF_FMT
    uint32_t index_stride = 0;
    bool add_tid_enable = false;
    bool resource_level = true;
    uint32_t oob_select = Oob::IndexAndOffset;
    uint32_t type = 0;
};

struct ImageResource {
    uint64_t base = 0;  // byte address, 256-byte aligned
    uint32_t min_lod = 0;  // 4.8 fixed point
    uint32_t format = Fmt::Invalid;  // IMG_FMT
    uint32_t width = 1;
    uint32_t height = 1;
    bool resource_level = true;
    Swizzle dst_sel = kIdentitySwizzle;
    uint32_t base_level = 0;
    uint32_t last_level = 0;
    uint32_t sw_mode = 0;  // SWIZZLE_MODE_ENUM (0 = SW_LINEAR)
    uint32_t bc_swizzle = 0;
    uint32_t type = ImageType::Tex2D;
    uint32_t depth = 1;  // depth (3D) or last array slice + 1 (arrays, cubes: faces)
    uint32_t base_array = 0;
    uint32_t array_pitch = 0;
    uint32_t max_mip = 0;  // mip levels of the resource - 1
    uint32_t min_lod_warn = 0;
    uint32_t perf_mod = 0;
    bool corner_samples = false;
    bool lod_hdw_cnt_en = false;
    bool prt_default = false;
    bool big_page = false;
    uint32_t counter_bank_id = 0;
    bool iterate_256 = false;
    uint32_t max_uncompressed_block_size = 0;
    uint32_t max_compressed_block_size = 0;
    bool meta_pipe_aligned = false;
    bool write_compress_enable = false;
    bool compression_en = false;
    bool alpha_is_on_msb = false;
    bool color_transform = false;
    uint64_t meta_data_address = 0;  // byte address, 256-byte aligned
};

struct SamplerState {
    uint32_t clamp_x = 0, clamp_y = 0, clamp_z = 0;  // SQ_TEX_CLAMP
    uint32_t max_aniso_ratio = 0;                    // SQ_TEX_ANISO_RATIO
    uint32_t depth_compare_func = 0;                 // SQ_TEX_DEPTH_COMPARE
    bool force_unnormalized = false;
    uint32_t aniso_threshold = 0;
    bool mc_coord_trunc = false;
    bool force_degamma = false;
    uint32_t aniso_bias = 0;
    bool trunc_coord = false;
    bool disable_cube_wrap = false;
    uint32_t filter_mode = 0;  // SQ_IMG_FILTER_TYPE
    bool skip_degamma = false;
    uint32_t min_lod = 0;  // 4.8 fixed point
    uint32_t max_lod = 0;  // 4.8 fixed point
    uint32_t perf_mip = 0;
    uint32_t perf_z = 0;
    uint32_t lod_bias = 0;  // signed 6.8 fixed point (14 bits, two's complement)
    uint32_t lod_bias_sec = 0;
    uint32_t xy_mag_filter = 0;  // SQ_TEX_XY_FILTER
    uint32_t xy_min_filter = 0;
    uint32_t z_filter = 0;    // SQ_TEX_Z_FILTER
    uint32_t mip_filter = 0;  // SQ_TEX_MIP_FILTER
    bool mip_point_preclamp = false;
    bool aniso_override = false;
    bool blend_prt = false;
    bool deriv_adjust_en = false;
    uint32_t border_color_ptr = 0;
    uint32_t border_color_type = 0;  // SQ_TEX_BORDER_COLOR
};

void encode(const BufferResource& d, uint32_t out[kBufferDwords]) noexcept;
void encode(const ImageResource& d, uint32_t out[kImageDwords]) noexcept;
void encode(const SamplerState& d, uint32_t out[kSamplerDwords]) noexcept;
BufferResource decode_buffer(const uint32_t in[kBufferDwords]) noexcept;
ImageResource decode_image(const uint32_t in[kImageDwords]) noexcept;
SamplerState decode_sampler(const uint32_t in[kSamplerDwords]) noexcept;

// What a 32-byte CBV/SRV/UAV heap slot holds. Buffer: a plausible V# in dwords 0..3 (dwords 4..7 are
// not inspected); Image: a plausible T#; Zero: all 32 bytes zero; Unknown: anything else (free-list
// links, data, words of an unsupported generation).
enum class SlotContent : uint8_t { Zero, Buffer, Image, Unknown };
SlotContent classify_slot(const uint32_t words[8]) noexcept;
// Plausibility checks: type field, destination selects in SQ_SEL_XYZW01, a format that is not
// reserved, reserved bits zero; images also resource_level = 1 and base_level <= last_level.
bool plausible_buffer(const uint32_t words[kBufferDwords]) noexcept;
bool plausible_image(const uint32_t words[kImageDwords]) noexcept;

// ---- Formats ----
// DXGI format -> hardware format and the destination selects that make the hardware's channel
// order read as the DXGI format (e.g. B8G8R8A8 = 8_8_8_8 read as Z, Y, X, W). False when the
// format has no gfx10 image format here (video, palettized, typeless without a canonical view).
struct HardwareFormat {
    uint32_t format = Fmt::Invalid;  // IMG_FMT (equals BUF_FMT for values below 78)
    Swizzle swizzle = kIdentitySwizzle;
};
bool hardware_format(uint32_t dxgi_format, HardwareFormat* out) noexcept;
// Hardware format + selects -> DXGI format: the table entry with the same format and selects, else
// the first entry with the same format (*exact = false). 0 when the format is not in the table.
uint32_t dxgi_format(uint32_t hardware_format, const Swizzle& dst_sel, bool* exact = nullptr) noexcept;
// Bytes per element of a typed-buffer format (BUF_FMT), 0 if unknown or block-compressed.
uint32_t buffer_format_bytes(uint32_t buf_format) noexcept;

// D3D12 Shader4ComponentMapping (3 bits per output channel: 0..3 = source channel, 4 = 0, 5 = 1)
// applied on top of a format's own selects.
Swizzle compose_component_mapping(const Swizzle& format_swizzle, uint32_t shader4_component_mapping) noexcept;
// The inverse for identity-format selects: the D3D12 mapping that produces `dst_sel` from a format
// whose own selects are `format_swizzle` (false when a select cannot be expressed).
bool component_mapping_from(const Swizzle& format_swizzle, const Swizzle& dst_sel, uint32_t* mapping) noexcept;

// ---- Samplers ----
// D3D12_SAMPLER_DESC <-> S#. The border colour is reduced to the three fixed colours (transparent
// black, opaque black, opaque white); other colours are reported through *exact = false.
SamplerState sampler_from_d3d12(const d3d12_state::SamplerDesc& desc, bool* exact = nullptr) noexcept;
d3d12_state::SamplerDesc d3d12_from_sampler(const SamplerState& s) noexcept;

// Fixed-point helpers (exposed for tests).
uint32_t lod_to_u4_8(float lod) noexcept;          // clamped to [0, 15.996]
float u4_8_to_lod(uint32_t v) noexcept;
uint32_t bias_to_s6_8(float bias) noexcept;        // clamped to [-32, 31.996]
float s6_8_to_bias(uint32_t v) noexcept;

}  // namespace playgta6::gpu::gfx10
