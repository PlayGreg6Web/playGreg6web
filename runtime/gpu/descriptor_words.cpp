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
