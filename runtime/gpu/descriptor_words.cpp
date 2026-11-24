#include "gpu/descriptor_words.h"

#include <algorithm>
#include <cstring>

#include "gpu/state_bridge.h"

namespace playgta6::gpu::bridge {

namespace d3d = playgta6::gpu::d3d12_state;
namespace hw = playgta6::gpu::gfx10;

namespace {

struct TextureShape {
    uint32_t dimension = 2;  // 1, 2, 3
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth_or_layers = 1;
    uint32_t mips = 1;
    uint32_t samples = 1;
    uint32_t format = 0;  // DXGI
};

uint32_t log2_u32(uint32_t v) noexcept {
    uint32_t r = 0;
    while (v > 1) {
        v >>= 1;
        ++r;
    }
    return r;
}

WordsStatus image_words(uint64_t address, const TextureShape& t, const d3d::ViewDescriptor& v, uint32_t out[8]) {
    if (address == 0) {
        return WordsStatus::NoAddress;
    }
    if (address % 256 != 0) {
        return WordsStatus::MisalignedAddress;
    }
    const uint32_t dxgi = v.format != 0 ? v.format : t.format;
    hw::HardwareFormat f;
    if (!hw::hardware_format(dxgi, &f)) {
        return WordsStatus::UnsupportedFormat;
    }
    hw::ImageResource img;
    img.base = address;
    img.format = f.format;
    img.dst_sel = v.kind == d3d::DescriptorKind::Srv ? hw::compose_component_mapping(f.swizzle, v.component_mapping)
                                                     : f.swizzle;
    img.width = std::max(1u, t.width);
    img.height = std::max(1u, t.height);
    const uint32_t mips = std::max(1u, t.mips);
    img.max_mip = mips - 1;
    img.min_lod = v.kind == d3d::DescriptorKind::Srv ? hw::lod_to_u4_8(v.min_lod_clamp) : 0u;

    // Mip range: SRVs carry (first, count; 0xffff = the rest), UAVs one mip.
    const uint32_t first_mip = std::min<uint32_t>(v.first_mip, mips - 1);
    uint32_t last_mip = first_mip;
    if (v.kind == d3d::DescriptorKind::Srv) {
        const uint32_t count = v.mip_levels == 0xffffu || v.mip_levels == 0 ? mips - first_mip : v.mip_levels;
        last_mip = std::min(mips - 1, first_mip + count - 1);
    }
    img.base_level = first_mip;
    img.last_level = last_mip;

    uint32_t dimension = v.dimension;
    const uint32_t layers = std::max(1u, t.depth_or_layers);
    if (dimension == d3d::ViewDimension::Unknown) {  // default view of the whole resource
        if (t.dimension == 1) {
            dimension = layers > 1 ? d3d::ViewDimension::Texture1DArray : d3d::ViewDimension::Texture1D;
        } else if (t.dimension == 3) {
            dimension = d3d::ViewDimension::Texture3D;
        } else if (t.samples > 1) {
            dimension = layers > 1 ? d3d::ViewDimension::Texture2DMsArray : d3d::ViewDimension::Texture2DMs;
        } else {
            dimension = layers > 1 ? d3d::ViewDimension::Texture2DArray : d3d::ViewDimension::Texture2D;
        }
    }
    const uint32_t first_slice = std::min(v.first_array_slice, layers - 1);
    const uint32_t slice_count =
        v.array_size == 0xffffffffu || v.array_size == 0 ? layers - first_slice : std::min(v.array_size, layers - first_slice);
    switch (dimension) {
    case d3d::ViewDimension::Texture1D:
        img.type = hw::ImageType::Tex1D;
        break;
    case d3d::ViewDimension::Texture1DArray:
        img.type = hw::ImageType::Tex1DArray;
        img.base_array = first_slice;
        img.depth = first_slice + slice_count;
        break;
    case d3d::ViewDimension::Texture2D:
        img.type = hw::ImageType::Tex2D;
        break;
    case d3d::ViewDimension::Texture2DArray:
        img.type = hw::ImageType::Tex2DArray;
        img.base_array = first_slice;
        img.depth = first_slice + slice_count;
        break;
    case d3d::ViewDimension::Texture2DMs:
    case d3d::ViewDimension::Texture2DMsArray:
        img.type = dimension == d3d::ViewDimension::Texture2DMs ? hw::ImageType::Tex2DMsaa : hw::ImageType::Tex2DMsaaArray;
        img.base_level = 0;
        img.last_level = log2_u32(std::max(1u, t.samples));  // MSAA: last_level holds log2(samples)
        img.max_mip = img.last_level;
        if (dimension == d3d::ViewDimension::Texture2DMsArray) {
            img.base_array = first_slice;
            img.depth = first_slice + slice_count;
        }
        break;
    case d3d::ViewDimension::Texture3D:
        img.type = hw::ImageType::Tex3D;
        img.depth = layers;
        break;
    case d3d::ViewDimension::TextureCube:
        img.type = hw::ImageType::Cube;
        img.depth = 6;
        break;
    case d3d::ViewDimension::TextureCubeArray: {
        img.type = hw::ImageType::Cube;
        // SRV cube arrays: First2DArrayFace / NumCubes in the array fields.
        const uint32_t cubes = v.array_size == 0xffffffffu || v.array_size == 0 ? (layers - first_slice) / 6 : v.array_size;
        img.base_array = first_slice;
        img.depth = first_slice + 6 * std::max(1u, cubes);
        break;
    }
    default:
        return WordsStatus::UnsupportedView;
    }
    hw::encode(img, out);
    return WordsStatus::Ok;
}

}  // namespace

void constant_buffer_words(uint64_t address, uint32_t size, uint32_t out[4]) noexcept {
    out[0] = static_cast<uint32_t>(address);
    out[1] = static_cast<uint32_t>(address >> 32) & 0xffffu;
    out[1] |= kConsoleCbvStride << 16;
    out[2] = static_cast<uint32_t>((uint64_t{size} + kConsoleCbvStride - 1) / kConsoleCbvStride);
    out[3] = kConsoleCbvWord3;
}

const char* words_status_name(WordsStatus status) noexcept {
    switch (status) {
    case WordsStatus::Ok: return "ok";
    case WordsStatus::Null: return "null descriptor";
    case WordsStatus::UnknownResource: return "unknown resource";
    case WordsStatus::NoAddress: return "texture without a descriptor address";
    case WordsStatus::MisalignedAddress: return "texture address not 256-byte aligned";
    case WordsStatus::UnsupportedFormat: return "format without a hardware format";
    case WordsStatus::UnsupportedView: return "view without descriptor words";
    }
    return "?";
}

WordsStatus view_words(const StateBridgeDevice& device, const d3d::ViewDescriptor& v, uint32_t out[8]) {
    std::memset(out, 0, kViewSlotBytes);
    switch (v.kind) {
    case d3d::DescriptorKind::Cbv:
        constant_buffer_words(v.first_element_or_address, v.num_elements_or_size, out);
        return v.first_element_or_address == 0 ? WordsStatus::Null : WordsStatus::Ok;
    case d3d::DescriptorKind::Srv:
    case d3d::DescriptorKind::Uav:
        break;
    case d3d::DescriptorKind::Empty:
        return WordsStatus::Null;
    default:
        return WordsStatus::UnsupportedView;
    }
    if (v.resource_id == 0) {
        return WordsStatus::Null;
    }
    TextureInfo tex;
    if (device.texture_info(v.resource_id, &tex)) {
        if (v.dimension == d3d::ViewDimension::Buffer) {
            return WordsStatus::UnsupportedView;
        }
        TextureShape t;
        t.dimension = tex.dimension == GpuTextureDimension::Dim1D ? 1u : (tex.dimension == GpuTextureDimension::Dim3D ? 3u : 2u);
        t.width = tex.width;
        t.height = tex.height;
        t.depth_or_layers = tex.depth_or_array_size;
        t.mips = tex.mip_levels;
        t.samples = tex.sample_count;
        t.format = static_cast<uint32_t>(tex.format);
        return image_words(tex.gpu_va, t, v, out);
    }
    const uint64_t va = device.buffer_gpu_va(v.resource_id);
    if (va == 0) {
        return WordsStatus::UnknownResource;
    }
    if (v.dimension != d3d::ViewDimension::Buffer && v.dimension != d3d::ViewDimension::Unknown) {
        return WordsStatus::UnsupportedView;  // e.g. acceleration structures
    }
    return buffer_words_at_address(va, v, out);
}

WordsStatus buffer_words_at_address(uint64_t va, const d3d::ViewDescriptor& v, uint32_t out[8]) {
    std::memset(out, 0, kViewSlotBytes);
    hw::BufferResource b;
    const bool raw = (v.flags & d3d::BufferViewFlags::Raw) != 0;
    if (v.structure_byte_stride != 0) {
        b.base = va + v.first_element_or_address * v.structure_byte_stride;
        b.stride = v.structure_byte_stride;
        b.num_records = v.num_elements_or_size;
        b.format = hw::Fmt::F32;
        b.oob_select = hw::Oob::IndexAndOffset;
    } else if (raw || v.format == 0) {
        b.base = va + v.first_element_or_address * 4u;
        b.num_records = v.num_elements_or_size * 4u;
        b.format = hw::Fmt::F32;
        b.oob_select = hw::Oob::Complete;
    } else {
        hw::HardwareFormat f;
        const uint32_t bytes = hw::hardware_format(v.format, &f) ? hw::buffer_format_bytes(f.format) : 0u;
        if (bytes == 0) {
            return WordsStatus::UnsupportedFormat;
        }
        b.base = va + v.first_element_or_address * bytes;
        b.stride = bytes;
        b.num_records = v.num_elements_or_size;
        b.format = f.format;
        b.dst_sel = v.kind == d3d::DescriptorKind::Srv ? hw::compose_component_mapping(f.swizzle, v.component_mapping)
                                                       : f.swizzle;
        b.oob_select = hw::Oob::IndexAndOffset;
    }
    hw::encode(b, out);
    return WordsStatus::Ok;
}

const char* decode_status_name(DecodeStatus status) noexcept {
    switch (status) {
    case DecodeStatus::Ok: return "ok";
    case DecodeStatus::Zero: return "zero words (null descriptor)";
    case DecodeStatus::NotDescriptor: return "no descriptor words";
    case DecodeStatus::KindMismatch: return "descriptor of another kind";
    case DecodeStatus::UnresolvedAddress: return "buffer address in no known buffer";
    case DecodeStatus::UnknownTexture: return "image address of no known texture";
    case DecodeStatus::UnsupportedFormat: return "image format without a DXGI format";
    case DecodeStatus::Count: break;
    }
    return "?";
}

namespace {

d3d::DescriptorKind kind_of(d3d::DescriptorRangeType range) noexcept {
    switch (range) {
    case d3d::DescriptorRangeType::Cbv: return d3d::DescriptorKind::Cbv;
    case d3d::DescriptorRangeType::Uav: return d3d::DescriptorKind::Uav;
    default: return d3d::DescriptorKind::Srv;
    }
}

// Byte size of a buffer descriptor's window: records of `stride` bytes when the stride is set and
// out-of-bounds checks count records, else num_records bytes.
uint64_t buffer_bytes(const hw::BufferResource& b) noexcept {
    if (b.stride != 0 && b.oob_select != hw::Oob::Complete) {
        return uint64_t{b.num_records} * b.stride;
    }
    return b.num_records;
}

uint32_t srgb_sibling(uint32_t dxgi) noexcept {
    static constexpr uint32_t kPairs[][2] = {{28, 29}, {87, 91}, {88, 93}, {71, 72}, {74, 75}, {77, 78}, {98, 99}};
    for (const auto& p : kPairs) {
        if (p[0] == dxgi) {
            return p[1];
        }
        if (p[1] == dxgi) {
            return p[0];
        }
    }
    return 0;
}

uint32_t view_dimension(const hw::ImageResource& img) noexcept {
    switch (img.type) {
    case hw::ImageType::Tex1D: return d3d::ViewDimension::Texture1D;
    case hw::ImageType::Tex1DArray: return d3d::ViewDimension::Texture1DArray;
    case hw::ImageType::Tex2D: return d3d::ViewDimension::Texture2D;
    case hw::ImageType::Tex2DArray: return d3d::ViewDimension::Texture2DArray;
    case hw::ImageType::Tex2DMsaa: return d3d::ViewDimension::Texture2DMs;
    case hw::ImageType::Tex2DMsaaArray: return d3d::ViewDimension::Texture2DMsArray;
    case hw::ImageType::Tex3D: return d3d::ViewDimension::Texture3D;
    case hw::ImageType::Cube:
        return img.depth > img.base_array + 6 ? d3d::ViewDimension::TextureCubeArray : d3d::ViewDimension::TextureCube;
    default: return d3d::ViewDimension::Unknown;
    }
}

}  // namespace

DecodeStatus decode_view_words(const StateBridgeDevice& device, const uint32_t w[8], d3d::DescriptorRangeType range,
                               d3d::ViewDescriptor* out) {
    *out = d3d::ViewDescriptor{};
    out->kind = kind_of(range);
    const hw::SlotContent content = hw::classify_slot(w);
    if (content == hw::SlotContent::Zero) {
        return DecodeStatus::Zero;
    }
    if (content == hw::SlotContent::Unknown || range == d3d::DescriptorRangeType::Sampler) {
        return content == hw::SlotContent::Unknown ? DecodeStatus::NotDescriptor : DecodeStatus::KindMismatch;
    }
    if (content == hw::SlotContent::Buffer) {
        const hw::BufferResource b = hw::decode_buffer(w);
        const uint64_t bytes = buffer_bytes(b);
        if (range == d3d::DescriptorRangeType::Cbv) {
            out->first_element_or_address = b.base;
            out->num_elements_or_size = static_cast<uint32_t>(std::min<uint64_t>(bytes, 0xffffffffu));
            d3d::GpuVaResolution res;
            return device.resolve_gpu_va(b.base, &res) ? DecodeStatus::Ok : DecodeStatus::UnresolvedAddress;
        }
        d3d::GpuVaResolution res;
        if (!device.resolve_gpu_va(b.base, &res)) {
            return DecodeStatus::UnresolvedAddress;
        }
        out->dimension = static_cast<uint8_t>(d3d::ViewDimension::Buffer);
        out->resource_id = res.resource_id;
        out->structure_byte_stride = 1;
        out->first_element_or_address = res.offset;
        out->num_elements_or_size = static_cast<uint32_t>(
            std::min<uint64_t>(bytes, res.resource_size > res.offset ? res.resource_size - res.offset : 0));
        return DecodeStatus::Ok;
    }
    // Image words.
    if (range == d3d::DescriptorRangeType::Cbv) {
        return DecodeStatus::KindMismatch;
    }
    const hw::ImageResource img = hw::decode_image(w);
    const uint32_t id = device.texture_at(img.base);
    if (id == 0) {
        return DecodeStatus::UnknownTexture;
    }
    bool exact = false;
    const uint32_t dxgi = hw::dxgi_format(img.format, img.dst_sel, &exact);
    if (dxgi == 0) {
        return DecodeStatus::UnsupportedFormat;
    }
    out->resource_id = id;
    // The view format: the decoded one when it is the texture's own format or its sRGB sibling (what
    // a WebGPU view of the texture may use), else the texture's format (0).
    TextureInfo tex;
    const uint32_t own = device.texture_info(id, &tex) ? static_cast<uint32_t>(tex.format) : 0u;
    out->format = dxgi == own || srgb_sibling(dxgi) == own ? dxgi : 0u;
    out->dimension = static_cast<uint8_t>(view_dimension(img));
    if (out->kind == d3d::DescriptorKind::Srv) {
        hw::HardwareFormat f;
        uint32_t mapping = d3d::kDefaultShader4ComponentMapping;
        if (!exact && hw::hardware_format(dxgi, &f)) {
            hw::component_mapping_from(f.swizzle, img.dst_sel, &mapping);
        }
        out->component_mapping = mapping;
        out->min_lod_clamp = hw::u4_8_to_lod(img.min_lod);
    }
    const bool msaa = img.type == hw::ImageType::Tex2DMsaa || img.type == hw::ImageType::Tex2DMsaaArray;
    out->first_mip = static_cast<uint16_t>(msaa ? 0u : img.base_level);
    out->mip_levels = static_cast<uint16_t>(msaa || out->kind == d3d::DescriptorKind::Uav
                                                ? 1u
                                                : img.last_level - img.base_level + 1u);
    out->first_array_slice = img.base_array;
    out->array_size = img.depth > img.base_array ? img.depth - img.base_array : 1u;
    if (out->dimension == d3d::ViewDimension::TextureCube || out->dimension == d3d::ViewDimension::TextureCubeArray) {
        out->array_size = std::max(1u, out->array_size / 6u);  // cube arrays: NumCubes
    }
    return DecodeStatus::Ok;
}

DecodeStatus decode_sampler_words(const uint32_t w[4], d3d::SamplerDesc* out) {
    if (w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 0) {
        return DecodeStatus::Zero;
    }
    const hw::SamplerState s = hw::decode_sampler(w);
    // Unused dword 3 bits 12..29 must be zero; enumerations in range.
    if ((w[3] & 0x3ffff000u) != 0 || s.max_aniso_ratio > 4 || s.filter_mode > 2 || s.z_filter > 2) {
        return DecodeStatus::NotDescriptor;
    }
    *out = hw::d3d12_from_sampler(s);
    return DecodeStatus::Ok;
}

void sampler_words(const d3d::SamplerDesc& desc, uint32_t out[4]) {
    hw::encode(hw::sampler_from_d3d12(desc), out);
}

WordsStatus image_words_at_address(uint64_t address, const AddressTexture& texture, const d3d::ViewDescriptor& view,
                                   uint32_t out[8]) {
    std::memset(out, 0, kViewSlotBytes);
    TextureShape t;
    t.dimension = texture.dimension == 2 ? 1u : (texture.dimension == 4 ? 3u : 2u);
    t.width = static_cast<uint32_t>(std::min<uint64_t>(texture.width, 0xffffffffull));
    t.height = texture.height;
    t.depth_or_layers = texture.depth_or_array_size;
    t.mips = texture.mip_levels;
    t.samples = texture.sample_count;
    t.format = texture.format;
    return image_words(address, t, view, out);
}

}  // namespace playgta6::gpu::bridge
