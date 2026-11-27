// Buffer bindings fitted into the device's binding-size rules before CreateBindGroup: Dawn rejects a
// uniform binding over maxUniformBufferBindingSize (64 KiB, the D3D12 CBV maximum the title's shaders
// address anyway), a storage binding over maxStorageBufferBindingSize or not a multiple of 4, and a
// binding under the layout's minBindingSize. A rejected bind group invalidates every pass that sets it.
// Pure; no GPU involved.
#pragma once

#include <algorithm>
#include <cstdint>

#include "gpu/native_exec/layout_model.h"

namespace playgta6::gpu::native_exec {

struct BufferBindingFit {
    bool ok = true;         // false: no range of the buffer fits; bind a dummy
    uint64_t size = 0;      // the explicit size to bind (never "whole")
    bool clamped = false;   // the requested / whole range was cut to the device maximum or to 4 bytes
};

// buffer_size: the buffer's size (0 = unknown: nothing is fitted, size 0 = whole). requested: 0 = whole.
// min_binding: the layout entry's minBindingSize (0 = none). max_binding: the device limit for the kind.
inline BufferBindingFit fit_buffer_binding(bool storage, uint64_t buffer_size, uint64_t offset, uint64_t requested,
                                           uint64_t min_binding, uint64_t max_binding) {
    BufferBindingFit f;
    if (buffer_size == 0) {
        f.size = requested;
        return f;
    }
    if (offset >= buffer_size) {
        f.ok = false;
        return f;
    }
    const uint64_t remaining = buffer_size - offset;
    uint64_t size = requested ? std::min(requested, remaining) : remaining;
    if (max_binding && size > max_binding) {
        size = max_binding;
        f.clamped = true;
    }
    if (storage && size % 4) {
        size &= ~uint64_t{3};
        f.clamped = true;
    }
    if (min_binding && size < min_binding) {
        if (remaining < min_binding || (max_binding && min_binding > max_binding)) {
            f.ok = false;
            return f;
        }
        size = min_binding;
    }
    if (size == 0) {
        f.ok = false;
        return f;
    }
    f.size = size;
    return f;
}

// An integer-format view (r32uint, rgba8sint, ...) bound to a float / unfilterable-float / depth layout
// entry, or a non-integer view bound to a sint / uint entry (or the wrong signedness): Dawn rejects the
// bind group ("None of the supported sample types (Uint) ... match the expected sample types (Float)").
// Depth formats are left to depth_sample_type_mismatch / pick_depth_read.
// layout_sample_type: 0 float, 1 unfilterable-float, 2 depth, 3 sint, 4 uint.
inline bool integer_sample_type_mismatch(uint8_t layout_sample_type, uint8_t format) {
    if (!format_valid(format) || format_is_depth(format)) return false;
    const WgslSampleType actual = texture_sample_type(format, true);
    const bool actual_int = actual == WgslSampleType::Sint || actual == WgslSampleType::Uint;
    if (layout_sample_type == 3) return actual != WgslSampleType::Sint;
    if (layout_sample_type == 4) return actual != WgslSampleType::Uint;
    return actual_int;
}

// A float (filterable) layout entry bound to a view WebGPU samples only as unfilterable-float: the
// 16-bit normalized formats (r16unorm & co., "texture-formats-tier1") and, without "float32-filterable",
// the 32-bit float ones. The entry is paired with a filtering sampler, so Dawn rejects the bind group
// ("None of the supported sample types (UnfilterableFloat) ... match the expected sample types (Float)").
// layout_sample_type: 0 float, 1 unfilterable-float, 2 depth, 3 sint, 4 uint.
inline bool format_is_norm16(uint8_t format) {
    if (!format_valid(format) || format_is_depth(format)) return false;
    const std::string_view n = format_info(format).name;
    return n.find("16unorm") != std::string_view::npos || n.find("16snorm") != std::string_view::npos;
}
inline bool unfilterable_float_mismatch(uint8_t layout_sample_type, uint8_t format, bool float32_filterable) {
    if (layout_sample_type != 0 || !format_valid(format) || format_is_depth(format)) return false;
    return format_is_norm16(format) || (format_is_float32(format) && !float32_filterable);
}

}  // namespace playgta6::gpu::native_exec
