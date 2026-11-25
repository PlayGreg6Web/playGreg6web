// gfx10 descriptor words (runtime/gpu/gfx10_descriptors.h): field positions against the public
// driver definitions, round trips, slot classification, format and sampler mappings; plus the
// STORAGE_BINDING usage of UAV textures whose descriptors those formats feed (G60). Synthetic
// values only.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_guest_resources.h"
#include "gpu/gfx10_descriptors.h"
#include "gpu/pso_cache.h"
#include "guest/com_bridge.h"
#include "os/vmem.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_gfx10_descriptors.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu::gfx10;
namespace d3d = playgta6::gpu::d3d12_state;

// Field positions: one field at a time, compared with the shifts of gfx10_sq_ko_reg.h.
void test_buffer_field_positions() {
    uint32_t w[4];
    BufferResource b;
    b.dst_sel = {0, 0, 0, 0};
    b.resource_level = false;
    encode(b, w);
    CHECK(w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 0);

    BufferResource s = b;
    s.base = 0x0000123456789abcull;
    encode(s, w);
    CHECK(w[0] == 0x56789abcu && w[1] == 0x1234u);  // base_address: 48 bits from dword 0 bit 0
    s = b;
    s.stride = 1;
    encode(s, w);
    CHECK(w[1] == (1u << 16));  // SqBufRsrcTWord1StrideShift = 16
    s = b;
    s.cache_swizzle = true;
    encode(s, w);
    CHECK(w[1] == (1u << 30));  // Gfx10SqBufRsrcTWord1CacheSwizzleShift = 30
    s = b;
    s.swizzle_enable = true;
    encode(s, w);
    CHECK(w[1] == (1u << 31));  // Gfx10SqBufRsrcTWord1SwizzleEnableShift = 31
    s = b;
    s.num_records = 0xdeadbeef;
    encode(s, w);
    CHECK(w[2] == 0xdeadbeefu);
    s = b;
    s.dst_sel = {1, 0, 0, 0};
    encode(s, w);
    CHECK(w[3] == 1u);
    s.dst_sel = {0, 1, 0, 0};
    encode(s, w);
    CHECK(w[3] == (1u << 3));
    s.dst_sel = {0, 0, 1, 0};
    encode(s, w);
    CHECK(w[3] == (1u << 6));
    s.dst_sel = {0, 0, 0, 1};
    encode(s, w);
    CHECK(w[3] == (1u << 9));
    s = b;
    s.format = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 12));  // Gfx10SqBufRsrcTWord3FormatShift = 12
    s = b;
    s.index_stride = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 21));
    s = b;
    s.add_tid_enable = true;
    encode(s, w);
    CHECK(w[3] == (1u << 23));
    s = b;
    s.resource_level = true;
    encode(s, w);
    CHECK(w[3] == (1u << 24));  // Gfx10SqBufRsrcTWord3ResourceLevelShift = 24
    s = b;
    s.oob_select = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 28));
    s = b;
    s.type = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 30));
}

void test_image_field_positions() {
    uint32_t w[8];
    ImageResource z;
    z.width = 1;
    z.height = 1;
    z.depth = 1;
    z.dst_sel = {0, 0, 0, 0};
    z.type = 0;
    z.resource_level = false;
    encode(z, w);
    for (uint32_t i = 0; i < 8; ++i) {
        CHECK(w[i] == 0);
    }
    ImageResource s = z;
    s.base = 0x0012'3456'789a'bc00ull;  // stored >> 8 in 40 bits
    encode(s, w);
    CHECK(w[0] == 0x56789abcu && (w[1] & 0xffu) == 0x34u);
    s = z;
    s.min_lod = 1;
    encode(s, w);
    CHECK(w[1] == (1u << 8));  // Gfx10SqImgRsrcTWord1MinLodShift = 8
    s = z;
    s.format = 1;
    encode(s, w);
    CHECK(w[1] == (1u << 20));  // Gfx10SqImgRsrcTWord1FormatShift = 20
    s = z;
    s.width = 2;
    encode(s, w);
    CHECK(w[1] == (1u << 30) && w[2] == 0);  // Gfx10SqImgRsrcTWord1WidthShift = 30
    s.width = 5;
    encode(s, w);
    CHECK(w[1] == 0 && w[2] == 1u);  // width - 1 = 4: bit 2 of the width lands in dword 2 bit 0
    s = z;
    s.height = 2;
    encode(s, w);
    CHECK(w[2] == (1u << 14));  // Gfx10SqImgRsrcTWord2HeightShift = 14
    s = z;
    s.resource_level = true;
    encode(s, w);
    CHECK(w[2] == (1u << 31));
    s = z;
    s.base_level = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 12));
    s = z;
    s.last_level = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 16));
    s = z;
    s.sw_mode = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 20));  // SqImgRsrcTWord3SwModeShift = 20
    s = z;
    s.bc_swizzle = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 25));
    s = z;
    s.type = ImageType::Tex2D;
    encode(s, w);
    CHECK(w[3] == (9u << 28));  // SqImgRsrcTWord3TypeShift = 28, SQ_RSRC_IMG_2D = 9
    s = z;
    s.depth = 2;
    encode(s, w);
    CHECK(w[4] == 1u);
    s = z;
    s.base_array = 1;
    encode(s, w);
    CHECK(w[4] == (1u << 16));  // Gfx10SqImgRsrcTWord4BaseArrayShift = 16
    s = z;
    s.max_mip = 1;
    encode(s, w);
    CHECK(w[5] == (1u << 4));  // Gfx10SqImgRsrcTWord5MaxMipShift = 4
    s = z;
    s.compression_en = true;
    encode(s, w);
    CHECK(w[6] == (1u << 21));  // SqImgRsrcTWord6CompressionEnShift = 21
    s = z;
    s.meta_data_address = 0x0012'3456'789a'bc00ull;
    encode(s, w);
    CHECK(w[6] == (0xbcu << 24) && w[7] == 0x3456789au);  // SqImgRsrcTWord6MetaDataAddressShift = 24
}

void test_sampler_field_positions() {
    uint32_t w[4];
    SamplerState z;
    encode(z, w);
    CHECK(w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 0);
    SamplerState s = z;
    s.clamp_y = 1;
    encode(s, w);
    CHECK(w[0] == (1u << 3));
    s = z;
    s.depth_compare_func = 1;
    encode(s, w);
    CHECK(w[0] == (1u << 12));  // SqImgSampTWord0DepthCompareFuncShift = 12
    s = z;
    s.filter_mode = 1;
    encode(s, w);
    CHECK(w[0] == (1u << 29));
    s = z;
    s.max_lod = 1;
    encode(s, w);
    CHECK(w[1] == (1u << 12));
    s = z;
    s.xy_min_filter = 1;
    encode(s, w);
    CHECK(w[2] == (1u << 22));  // SqImgSampTWord2XyMinFilterShift = 22
    s = z;
    s.mip_filter = 1;
    encode(s, w);
    CHECK(w[2] == (1u << 26));
    s = z;
    s.border_color_type = 1;
    encode(s, w);
    CHECK(w[3] == (1u << 30));  // SqImgSampTWord3BorderColorTypeShift = 30
    s = z;
    s.border_color_ptr = 1;
    encode(s, w);
    CHECK(w[3] == 1u);  // Gfx10SqImgSampTWord3BorderColorPtrShift = 0
}

void test_round_trips() {
    BufferResource b;
    b.base = 0x0000'7e00'1234'5670ull;
    b.stride = 48;
    b.num_records = 4096;
    b.dst_sel = {Sel::Z, Sel::Y, Sel::X, Sel::One};
    b.format = 56;
    b.index_stride = 3;
    b.add_tid_enable = true;
    b.oob_select = Oob::Complete;
    uint32_t w[8] = {};
    encode(b, w);
    const BufferResource b2 = decode_buffer(w);
    CHECK(b2.base == b.base && b2.stride == b.stride && b2.num_records == b.num_records && b2.dst_sel == b.dst_sel &&
          b2.format == b.format && b2.index_stride == 3 && b2.add_tid_enable && b2.resource_level &&
          b2.oob_select == Oob::Complete && b2.type == 0);
    CHECK(plausible_buffer(w));
    CHECK(classify_slot(w) == SlotContent::Buffer);

    ImageResource i;
    i.base = 0x0000'00a1'b2c3'd400ull;
    i.min_lod = 0x123;
    i.format = 181;
    i.width = 3840;
    i.height = 2160;
    i.dst_sel = {Sel::X, Sel::Y, Sel::Z, Sel::One};
    i.base_level = 2;
    i.last_level = 11;
    i.sw_mode = 27;
    i.bc_swizzle = 5;
    i.type = ImageType::Tex2DArray;
    i.depth = 6;
    i.base_array = 2;
    i.array_pitch = 3;
    i.max_mip = 11;
    i.min_lod_warn = 0x456;
    i.perf_mod = 4;
    i.corner_samples = true;
    i.lod_hdw_cnt_en = true;
    i.prt_default = true;
    i.big_page = true;
    i.counter_bank_id = 0x5a;
    i.iterate_256 = true;
    i.max_uncompressed_block_size = 2;
    i.max_compressed_block_size = 1;
    i.meta_pipe_aligned = true;
    i.write_compress_enable = true;
    i.compression_en = true;
    i.alpha_is_on_msb = true;
    i.color_transform = true;
    i.meta_data_address = 0x0000'00f0'0000'1200ull;
    encode(i, w);
    const ImageResource i2 = decode_image(w);
    CHECK(i2.base == i.base && i2.min_lod == i.min_lod && i2.format == i.format && i2.width == 3840 &&
          i2.height == 2160 && i2.dst_sel == i.dst_sel && i2.base_level == 2 && i2.last_level == 11 &&
          i2.sw_mode == 27 && i2.bc_swizzle == 5 && i2.type == ImageType::Tex2DArray && i2.depth == 6 &&
          i2.base_array == 2 && i2.array_pitch == 3 && i2.max_mip == 11 && i2.min_lod_warn == 0x456 &&
          i2.perf_mod == 4 && i2.corner_samples && i2.lod_hdw_cnt_en && i2.prt_default && i2.big_page &&
          i2.counter_bank_id == 0x5a && i2.iterate_256 && i2.max_uncompressed_block_size == 2 &&
          i2.max_compressed_block_size == 1 && i2.meta_pipe_aligned && i2.write_compress_enable &&
          i2.compression_en && i2.alpha_is_on_msb && i2.color_transform &&
          i2.meta_data_address == i.meta_data_address && i2.resource_level);
    CHECK(plausible_image(w));
    CHECK(classify_slot(w) == SlotContent::Image);

    SamplerState s;
    s.clamp_x = 6;
    s.clamp_y = 2;
    s.clamp_z = 1;
    s.max_aniso_ratio = 4;
    s.depth_compare_func = 3;
    s.force_unnormalized = true;
    s.aniso_threshold = 5;
    s.aniso_bias = 33;
    s.filter_mode = 2;
    s.min_lod = 0x10;
    s.max_lod = 0xfff;
    s.perf_mip = 9;
    s.perf_z = 3;
    s.lod_bias = 0x3f00;
    s.lod_bias_sec = 7;
    s.xy_mag_filter = 3;
    s.xy_min_filter = 2;
    s.z_filter = 2;
    s.mip_filter = 2;
    s.mip_point_preclamp = true;
    s.aniso_override = true;
    s.blend_prt = true;
    s.deriv_adjust_en = true;
    s.border_color_ptr = 0x7ff;
    s.border_color_type = 3;
    uint32_t sw[4];
    encode(s, sw);
    const SamplerState s2 = decode_sampler(sw);
    uint32_t sw2[4];
    encode(s2, sw2);
    CHECK(std::memcmp(sw, sw2, sizeof(sw)) == 0);
    CHECK(s2.clamp_x == 6 && s2.max_aniso_ratio == 4 && s2.filter_mode == 2 && s2.lod_bias == 0x3f00 &&
          s2.border_color_ptr == 0x7ff && s2.border_color_type == 3 && s2.deriv_adjust_en);
}

void test_classify_slot() {
    uint32_t zero[8] = {};
    CHECK(classify_slot(zero) == SlotContent::Zero);
    // A free-list link (next index in the first dword) and plain float data are not descriptors.
    uint32_t link[8] = {42, 0, 0, 0, 0, 0, 0, 0};
    CHECK(classify_slot(link) == SlotContent::Unknown);
    uint32_t floats[8] = {0x3f800000, 0xbf800000, 0, 0xffffffff, 0x3f800000, 0x3f800000, 0, 0};
    CHECK(classify_slot(floats) == SlotContent::Unknown);
    // Buffer descriptors: resource_level is not required (constant-buffer words with dword 3 =
    // 0x0002403C: selects X, W, 0, 0, format 36, resource_level 0); a reserved format or a non-zero
    // type is rejected.
    BufferResource b;
    b.format = Fmt::F32;
    uint32_t w[8] = {};
    encode(b, w);
    CHECK(classify_slot(w) == SlotContent::Buffer);
    const uint32_t cbv[8] = {0x1000, 0x00100000, 6, 0x0002403Cu, 0, 0, 0, 0};
    CHECK(classify_slot(cbv) == SlotContent::Buffer);
    CHECK(decode_buffer(cbv).stride == 16 && decode_buffer(cbv).num_records == 6 && !decode_buffer(cbv).resource_level);
    b.type = 1;
    encode(b, w);
    CHECK(classify_slot(w) == SlotContent::Unknown);
    b.type = 0;
    b.format = 100;
    encode(b, w);
    CHECK(classify_slot(w) == SlotContent::Unknown);
    // Image with base_level > last_level or a reserved format is rejected.
    ImageResource i;
    i.format = 56;
    encode(i, w);
    CHECK(classify_slot(w) == SlotContent::Image);
    i.base_level = 3;
    i.last_level = 1;
    encode(i, w);
    CHECK(classify_slot(w) == SlotContent::Unknown);
    i.base_level = 0;
    i.format = 155;
    encode(i, w);
    CHECK(classify_slot(w) == SlotContent::Unknown);
}

// G60: typeless, depth/stencil-plane and Xbox DXGI formats map to gfx10 image formats; reverse
// lookups of the shared hardware formats keep their canonical DXGI format.
void test_g60_formats() {
    struct Case {
        uint32_t dxgi;
        uint32_t hw;
        Swizzle sel;
    };
    const Swizzle x001 = {Sel::X, Sel::Zero, Sel::Zero, Sel::One};
    const Swizzle xyz1 = {Sel::X, Sel::Y, Sel::Z, Sel::One};
    const Swizzle zyxw = {Sel::Z, Sel::Y, Sel::X, Sel::W};
    const Swizzle zyx1 = {Sel::Z, Sel::Y, Sel::X, Sel::One};
    const Case cases[] = {
        {115, 136, zyxw},              // B4G4R4A4_UNORM -> 4_4_4_4_UNORM
        {118, 7, x001},                // D16_UNORM_S8_UINT -> 16_UNORM (depth plane)
        {119, 7, x001},                // R16_UNORM_X8_TYPELESS
        {120, 5, x001},                // X16_TYPELESS_G8_UINT -> 8_UINT (stencil plane)
        {19, 22, x001},                // R32G8X24_TYPELESS -> 32_FLOAT
        {22, 5, x001},                 // X32_TYPELESS_G8X24_UINT -> 8_UINT
        {44, 141, x001},               // R24G8_TYPELESS -> 8_24_UNORM
        {47, 5, x001},                 // X24_TYPELESS_G8_UINT -> 8_UINT
        {90, 56, zyxw},                // B8G8R8A8_TYPELESS -> 8_8_8_8_UNORM
        {92, 56, zyx1},                // B8G8R8X8_TYPELESS
        {5, 72, xyz1},                 // R32G32B32_TYPELESS -> 32_32_32_UINT
        {89, 50, kIdentitySwizzle},    // R10G10B10_XR_BIAS_A2_UNORM -> 2_10_10_10_UNORM
    };
    for (const Case& c : cases) {
        HardwareFormat f;
        CHECK(hardware_format(c.dxgi, &f) && f.format == c.hw && f.swizzle == c.sel);
        // T# encode/decode keeps the format and selects, and the slot classifies as an image
        // (valid_image_format accepts every new hardware format).
        ImageResource img;
        img.base = 0x0000000100040000ull;
        img.format = f.format;
        img.width = 64;
        img.height = 32;
        img.dst_sel = compose_component_mapping(f.swizzle, 0x1688u);
        img.type = ImageType::Tex2D;
        uint32_t words[kImageDwords] = {};
        encode(img, words);
        CHECK(plausible_image(words));
        CHECK(classify_slot(words) == SlotContent::Image);
        const ImageResource dec = decode_image(words);
        CHECK(dec.format == c.hw && dec.dst_sel == c.sel && dec.width == 64 && dec.height == 32);
        // The reverse lookup lands on a DXGI format with the same hardware format and selects.
        bool exact = false;
        const uint32_t back = dxgi_format(dec.format, dec.dst_sel, &exact);
        HardwareFormat bf;
        CHECK(exact && hardware_format(back, &bf) && bf.format == c.hw && bf.swizzle == c.sel);
    }
    // The new hardware format reverses to its only DXGI format; aliases do not take over the
    // canonical reverse mappings of shared hardware formats.
    bool exact = false;
    CHECK(dxgi_format(136, zyxw, &exact) == 115 && exact);
    CHECK(dxgi_format(7, x001, &exact) == 56 && exact);    // R16_UNORM, not D16 / Xbox D16S8
    CHECK(dxgi_format(5, x001, &exact) == 62 && exact);    // R8_UINT, not a stencil plane
    CHECK(dxgi_format(22, x001, &exact) == 41 && exact);   // R32_FLOAT
    CHECK(dxgi_format(141, x001, &exact) == 45 && exact);  // D24_UNORM_S8_UINT
    CHECK(dxgi_format(72, xyz1, &exact) == 7 && exact);    // R32G32B32_UINT
    CHECK(dxgi_format(50, kIdentitySwizzle, &exact) == 24 && exact);  // R10G10B10A2_UNORM
    CHECK(dxgi_format(56, zyxw, &exact) == 87 && exact);   // B8G8R8A8_UNORM
    CHECK(dxgi_format(56, zyx1, &exact) == 88 && exact);   // B8G8R8X8_UNORM
    CHECK(dxgi_format(56, kIdentitySwizzle, &exact) == 28 && exact);  // R8G8B8A8_UNORM
    CHECK(dxgi_format(136, kIdentitySwizzle, &exact) == 115 && !exact);
    // 4_4_4_4 is image-only: not a buffer format.
    CHECK(buffer_format_bytes(136) == 0);
}

void test_formats() {
    HardwareFormat f;
    CHECK(hardware_format(28, &f) && f.format == 56 && f.swizzle == kIdentitySwizzle);  // R8G8B8A8_UNORM
    CHECK(hardware_format(87, &f) && f.format == 56 && f.swizzle == (Swizzle{Sel::Z, Sel::Y, Sel::X, Sel::W}));
    CHECK(hardware_format(29, &f) && f.format == 130);  // 8_8_8_8_SRGB
    CHECK(hardware_format(24, &f) && f.format == 50);   // R10G10B10A2_UNORM -> 2_10_10_10_UNORM
    CHECK(hardware_format(26, &f) && f.format == 36);   // R11G11B10_FLOAT -> 10_11_11_FLOAT
    CHECK(hardware_format(98, &f) && f.format == 181);  // BC7_UNORM
    CHECK(hardware_format(2, &f) && f.format == 77);    // R32G32B32A32_FLOAT
    CHECK(hardware_format(61, &f) && f.format == 1 &&
          f.swizzle == (Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One}));  // R8_UNORM -> 8_UNORM (X,0,0,1)
    CHECK(hardware_format(65, &f) && f.format == 1 &&
          f.swizzle == (Swizzle{Sel::Zero, Sel::Zero, Sel::Zero, Sel::X}));  // A8_UNORM -> 8_UNORM (0,0,0,X)
    CHECK(!hardware_format(100, &f));                   // video format
    bool exact = false;
    CHECK(dxgi_format(56, Swizzle{Sel::Z, Sel::Y, Sel::X, Sel::W}, &exact) == 87 && exact);
    CHECK(dxgi_format(56, kIdentitySwizzle, &exact) == 28 && exact);
    CHECK(dxgi_format(56, Swizzle{Sel::W, Sel::Z, Sel::Y, Sel::X}, &exact) == 28 && !exact);
    CHECK(dxgi_format(1, Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One}, &exact) == 61 && exact);
    CHECK(dxgi_format(1, Swizzle{Sel::Zero, Sel::Zero, Sel::Zero, Sel::X}, &exact) == 65 && exact);
    CHECK(dxgi_format(1, kIdentitySwizzle, &exact) == 61 && !exact);
    CHECK(dxgi_format(200, kIdentitySwizzle, &exact) == 0 && !exact);
    // A8_UNORM (65) T# image descriptor encode/decode round-trip preserves distinct DXGI format 65.
    {
        HardwareFormat a8;
        CHECK(hardware_format(65, &a8));
        ImageResource img;
        img.base = 0x0000'0001'0002'0000ull;
        img.format = a8.format;
        img.width = 256;
        img.height = 128;
        img.dst_sel = compose_component_mapping(a8.swizzle, 0x1688u);
        img.type = ImageType::Tex2D;
        uint32_t words[kImageDwords] = {};
        encode(img, words);
        CHECK(plausible_image(words));
        CHECK(classify_slot(words) == SlotContent::Image);
        const ImageResource dec = decode_image(words);
        CHECK(dec.format == 1u && dec.dst_sel == (Swizzle{Sel::Zero, Sel::Zero, Sel::Zero, Sel::X}));
        bool a8_exact = false;
        CHECK(dxgi_format(dec.format, dec.dst_sel, &a8_exact) == 65u && a8_exact);
    }
    // Every DXGI format of the table maps to a hardware format that maps back to a DXGI format with
    // the same hardware format.
    for (uint32_t dxgi = 1; dxgi <= 120; ++dxgi) {
        HardwareFormat h;
        if (!hardware_format(dxgi, &h)) {
            continue;
        }
        const uint32_t back = dxgi_format(h.format, h.swizzle);
        HardwareFormat h2;
        CHECK(back != 0 && hardware_format(back, &h2) && h2.format == h.format);
    }
    test_g60_formats();
    CHECK(buffer_format_bytes(Fmt::F32) == 4 && buffer_format_bytes(77) == 16 && buffer_format_bytes(74) == 12 &&
          buffer_format_bytes(14) == 2 && buffer_format_bytes(1) == 1 && buffer_format_bytes(0) == 0);
}

void test_component_mapping() {
    // D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING = 0x1688 (R, G, B, A).
    CHECK(compose_component_mapping(kIdentitySwizzle, 0x1688) == kIdentitySwizzle);
    const Swizzle bgra = {Sel::Z, Sel::Y, Sel::X, Sel::W};
    CHECK(compose_component_mapping(bgra, 0x1688) == bgra);
    // (A, 0, 1, R): 3 | 4 << 3 | 5 << 6 | 0 << 9.
    const uint32_t m = 3u | (4u << 3) | (5u << 6) | (1u << 12);
    CHECK(compose_component_mapping(kIdentitySwizzle, m) == (Swizzle{Sel::W, Sel::Zero, Sel::One, Sel::X}));
    uint32_t back = 0;
    CHECK(component_mapping_from(kIdentitySwizzle, Swizzle{Sel::W, Sel::Zero, Sel::One, Sel::X}, &back) && back == m);
    CHECK(component_mapping_from(bgra, bgra, &back) && back == 0x1688);
    // A select the format does not provide cannot be expressed.
    CHECK(!component_mapping_from(Swizzle{Sel::X, Sel::Zero, Sel::Zero, Sel::One},
                                  Swizzle{Sel::Y, Sel::Zero, Sel::Zero, Sel::One}, &back));
}

void test_samplers() {
    CHECK(lod_to_u4_8(0.0f) == 0 && lod_to_u4_8(1.0f) == 256 && lod_to_u4_8(1000.0f) == 0xfff &&
          lod_to_u4_8(-1.0f) == 0);
    CHECK(u4_8_to_lod(256) == 1.0f);
    CHECK(bias_to_s6_8(-1.0f) == 0x3f00 && s6_8_to_bias(0x3f00) == -1.0f && s6_8_to_bias(bias_to_s6_8(2.5f)) == 2.5f);

    d3d::SamplerDesc d;  // MIN_MAG_MIP_LINEAR, wrap
    d.max_lod = 13.0f;
    bool exact = false;
    SamplerState s = sampler_from_d3d12(d, &exact);
    CHECK(exact && s.xy_mag_filter == 1 && s.xy_min_filter == 1 && s.mip_filter == 2 && s.clamp_x == 0 &&
          s.max_lod == 13u * 256u && s.border_color_type == 0);
    d3d::SamplerDesc back = d3d12_from_sampler(s);
    CHECK(back.filter == 0x15 && back.address_u == 1 && back.max_lod == 13.0f && back.min_lod == 0.0f);

    d3d::SamplerDesc c;
    c.filter = 0x80 | 0x15;  // COMPARISON_MIN_MAG_MIP_LINEAR
    c.comparison_func = 4;   // LESS_EQUAL
    c.address_u = 4;         // BORDER
    c.address_v = 3;         // CLAMP
    c.address_w = 5;         // MIRROR_ONCE
    c.border_color[0] = c.border_color[1] = c.border_color[2] = c.border_color[3] = 1.0f;
    c.mip_lod_bias = -0.5f;
    s = sampler_from_d3d12(c, &exact);
    CHECK(exact && s.depth_compare_func == 3 && s.clamp_x == 6 && s.clamp_y == 2 && s.clamp_z == 3 &&
          s.border_color_type == 2 && s.filter_mode == 0);
    back = d3d12_from_sampler(s);
    CHECK(back.filter == (0x80u | 0x15u) && back.comparison_func == 4 && back.address_u == 4 && back.address_v == 3 &&
          back.address_w == 5 && back.border_color[0] == 1.0f && back.mip_lod_bias == -0.5f);

    d3d::SamplerDesc a;
    a.filter = 0x55;  // ANISOTROPIC
    a.max_anisotropy = 8;
    s = sampler_from_d3d12(a, &exact);
    CHECK(s.xy_mag_filter == 3 && s.max_aniso_ratio == 3);
    back = d3d12_from_sampler(s);
    CHECK(back.filter == 0x55 && back.max_anisotropy == 8);

    d3d::SamplerDesc mx;
    mx.filter = 0x180 | 0x15;  // MAXIMUM_MIN_MAG_MIP_LINEAR
    s = sampler_from_d3d12(mx, &exact);
    CHECK(s.filter_mode == 2 && d3d12_from_sampler(s).filter == (0x180u | 0x15u));

    d3d::SamplerDesc odd;
    odd.border_color[0] = 0.5f;
    sampler_from_d3d12(odd, &exact);
    CHECK(!exact);
}

// G60: a BGRA8 UAV texture gets STORAGE_BINDING only when the WebGPU device has
// "bgra8unorm-storage"; without it the texture keeps its other usages and the device notes why.
struct GpuPacket {
    playgta6::GpuOpcode opcode{};
    std::vector<uint8_t> payload;
};

std::vector<GpuPacket> split_packets(const std::vector<std::vector<uint8_t>>& captured) {
    std::vector<uint8_t> all;
    for (const auto& c : captured) {
        all.insert(all.end(), c.begin(), c.end());
    }
    std::vector<GpuPacket> out;
    size_t off = 0;
    while (off + sizeof(playgta6::GpuPacketHeader) <= all.size()) {
        playgta6::GpuPacketHeader h{};
        std::memcpy(&h, all.data() + off, sizeof(h));
        off += sizeof(h);
        if (off + h.payload_bytes > all.size()) {
            break;
        }
        GpuPacket p;
        p.opcode = static_cast<playgta6::GpuOpcode>(h.opcode);
        p.payload.assign(all.begin() + static_cast<std::ptrdiff_t>(off),
                         all.begin() + static_cast<std::ptrdiff_t>(off + h.payload_bytes));
        off += h.payload_bytes;
        out.push_back(std::move(p));
    }
    CHECK(off == all.size());
    return out;
}

void test_bgra8_uav_storage_usage() {
    namespace com = playgta6::guest::com;
    namespace dg = playgta6::gpu::d3d12_guest;
    namespace pso = playgta6::gpu::pso;
    using playgta6::GpuCreateTexture2Payload;
    using playgta6::GpuOpcode;
    namespace GpuTextureUsage = playgta6::GpuTextureUsage;

    playgta6::os::VirtualMemoryArena arena;
    com::reset_com_bridge();
    com::set_guest_memory(&arena);
    for (const bool enabled : {true, false}) {
        std::vector<std::vector<uint8_t>> captured;
        dg::DeviceOptions options;
        options.sink = [&captured](const uint8_t* data, size_t size) { captured.emplace_back(data, data + size); };
        options.bridge.features = enabled ? pso::WgpuFeature::Bgra8UnormStorage : 0u;
        options.defer_gpu_objects = false;  // the create packet is checked at creation
        com::Ref<dg::Device> device = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), options);
        CHECK(device);
        if (!device) {
            continue;
        }
        dg::GuestResourceDesc desc{};
        desc.Dimension = 3;  // TEXTURE2D
        desc.Width = 64;
        desc.Height = 64;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = 87;  // B8G8R8A8_UNORM
        desc.SampleCount = 1;
        desc.Flags = dg::ResourceFlags::AllowUnorderedAccess;
        dg::Resource::Placement placement;
        placement.heap.Type = dg::HeapType::Default;
        com::Ref<dg::Resource> resource =
            com::create_object<dg::Resource>(com::implements(dg::kID3D12Resource), dg::device_ref(device.get()), desc,
                                             std::move(placement), uint32_t{0});
        CHECK(resource && resource->initialize() && resource->has_gpu_texture());

        const std::vector<GpuPacket> ps = split_packets(captured);
        const GpuPacket* create = nullptr;
        size_t creates = 0;
        for (const GpuPacket& p : ps) {
            if (p.opcode == GpuOpcode::CreateTexture2) {
                create = &p;
                ++creates;
            }
        }
        CHECK(creates == 1 && create != nullptr && create->payload.size() >= sizeof(GpuCreateTexture2Payload));
        if (create != nullptr && create->payload.size() >= sizeof(GpuCreateTexture2Payload)) {
            GpuCreateTexture2Payload t{};
            std::memcpy(&t, create->payload.data(), sizeof(t));
            CHECK(t.format == static_cast<uint8_t>(pso::WgpuTextureFormat::Bgra8Unorm));
            CHECK((t.usage_flags & GpuTextureUsage::TextureBinding) != 0);
            CHECK(((t.usage_flags & GpuTextureUsage::StorageBinding) != 0) == enabled);
        }
        bool warned = false;
        for (const std::string& d : device->diagnostics()) {
            warned = warned || d.find("has no WebGPU storage binding") != std::string::npos;
        }
        CHECK(warned == !enabled);
        resource = {};
        device = {};
    }
    com::reset_com_bridge();
    com::set_guest_memory(nullptr);
}

}  // namespace

int main() {
    test_buffer_field_positions();
    test_image_field_positions();
    test_sampler_field_positions();
    test_round_trips();
    test_classify_slot();
    test_formats();
    test_component_mapping();
    test_samplers();
    test_bgra8_uav_storage_usage();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_gfx10_descriptors failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All gfx10_descriptors_tests passed.\n");
    return 0;
}
