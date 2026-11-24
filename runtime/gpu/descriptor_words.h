#pragma once

// The descriptor words the console's D3D12 leaves at a descriptor's CPU address (G48).
//
// On the console a CPU descriptor handle is a pointer into memory the title can read: Create*View,
// CreateSampler and CopyDescriptors write the GPU's own descriptor words there (gfx10_descriptors.h),
// and the title copies those 32-byte slots itself (memcpy of whole slots) into the shader-visible
// heap when it fills a root table. A heap slot written only in the runtime's registry stays empty
// in title memory, and every table the title assembles from it reads zeros.
//
// This module builds the words from the runtime's view records:
//   * CBV: exactly the words of the title's own inlined CreateConstantBufferView (11 accessors in
//     the executable write descriptors themselves when the byte at device + 0x18 is 0; the runtime
//     sets it to 1 so they call the API): 16 bytes, buffer resource with base = BufferLocation,
//     stride 16, num_records = SizeInBytes rounded up to 16 bytes / 16, dword 3 = 0x0002403C. A null
//     CBV is the same with base 0 and num_records 0. Dwords 4..7 are left as they are (the title's
//     writer stores two qwords).
//   * Buffer SRV/UAV: base = buffer GPU address + FirstElement * element size. Raw: stride 0,
//     num_records in bytes, 32_FLOAT, oob 3. Structured: stride = StructureByteStride, num_records =
//     NumElements, oob 0. Typed: stride = element size, num_records = NumElements, the format's
//     buffer format and selects, oob 0.
//   * Texture SRV/UAV: image resource with the texture's descriptor address (TextureInfo::gpu_va,
//     256-byte aligned), the view's format and selects (SRV component mapping applied), mip range,
//     array range and image type of the view, sw_mode 0 (the runtime does not tile textures).
//   * Sampler: sampler state (16 bytes; sampler heaps have a 16-byte increment).
//   * Null SRV / UAV descriptors (no resource): zero words.
// View slots are 32 bytes: buffer words fill dwords 0..3, dwords 4..7 are zero (CBVs: untouched).
// Only the CBV words have ground truth in the title; the others follow the open-source drivers'
// conventions for the public fields (the title only copies them as opaque 32-byte slots).

#include <cstdint>

#include "gpu/descriptor_heap.h"
#include "gpu/gfx10_descriptors.h"

namespace playgta6::gpu::bridge {

class StateBridgeDevice;

inline constexpr uint32_t kViewSlotBytes = 32;
inline constexpr uint32_t kSamplerSlotBytes = 16;
inline constexpr uint32_t kCbvWordBytes = 16;  // what a CBV write stores (dwords 0..3)
// Constant-buffer words of the console (see above): stride and dword 3.
inline constexpr uint32_t kConsoleCbvStride = 16;
inline constexpr uint32_t kConsoleCbvWord3 = 0x0002403Cu;

// The console's constant-buffer words for [address, address + size) (address 0: null CBV).
void constant_buffer_words(uint64_t address, uint32_t size, uint32_t out[4]) noexcept;

enum class WordsStatus : uint8_t {
    Ok,
    Null,               // null descriptor: zero words
    UnknownResource,    // the view names a resource the bridge does not know: zero words
    NoAddress,          // a texture without a descriptor address: zero words
    MisalignedAddress,  // a texture address that is not 256-byte aligned: zero words
    UnsupportedFormat,  // no hardware format for the view format: zero words
    UnsupportedView,    // RTV / DSV / acceleration structure: zero words
};
const char* words_status_name(WordsStatus status) noexcept;

// Words for a CBV/SRV/UAV record (32 bytes, `out` zeroed first).
WordsStatus view_words(const StateBridgeDevice& device, const d3d12_state::ViewDescriptor& view,
                       uint32_t out[kViewSlotBytes / 4]);
// Words for a sampler (16 bytes).
void sampler_words(const d3d12_state::SamplerDesc& desc, uint32_t out[kSamplerSlotBytes / 4]);

// Buffer words for a buffer view at a GPU address (Xbox device slots 70 / 71 name the address, not a
// resource): FirstElement counts from the address.
WordsStatus buffer_words_at_address(uint64_t va, const d3d12_state::ViewDescriptor& view,
                                    uint32_t out[kViewSlotBytes / 4]);

// A texture the title describes only by address and resource description (Xbox device slot 70 /
// 71: no resource object exists). Same encoding as a texture view, base = `address`.
struct AddressTexture {
    uint32_t dimension = 3;  // D3D12_RESOURCE_DIMENSION (2 1D, 3 2D, 4 3D)
    uint64_t width = 1;
    uint32_t height = 1;
    uint32_t depth_or_array_size = 1;
    uint32_t mip_levels = 1;
    uint32_t format = 0;  // DXGI
    uint32_t sample_count = 1;
};
WordsStatus image_words_at_address(uint64_t address, const AddressTexture& texture,
                                   const d3d12_state::ViewDescriptor& view, uint32_t out[kViewSlotBytes / 4]);

}  // namespace playgta6::gpu::bridge
