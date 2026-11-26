#pragma once

// Full pipeline description for the native pipeline warm list (pipeline_prewarm.h, list version 2).
//
// Everything the host needs to create a title pipeline identically ahead of the title's first use:
//   * the pipeline state key (pso::GraphicsPipelineKey / pso::ComputePipelineKey: stage identities, render
//     target formats, blend / depth / stencil / raster / multisample state, vertex buffer + attribute
//     layout), byte for byte what the creation packet carries;
//   * the shader pack keys of its stages (the module ids are the deterministic PackModuleTable ids);
//   * its pipeline layout: the bind group layouts (entries + register joins) the layout interns from.
// No game bytes: only keys, state and layout shapes.
//
// Blob layout (little endian): u32 version (1), u32 kind, u64 pack_key0, u64 pack_key1,
// u32 key_bytes + the key bytes, u32 has_layout, [layout: u64 content_id, u32 group_count, per group
// {u64 content_id, u32 binding_count, per binding {GpuBindGroupLayoutEntry (24 bytes), ShaderBinding fields}}].

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu/pipeline_prewarm.h"
#include "gpu/pso_cache.h"
#include "gpu/state_bridge.h"

namespace playgta6::gpu {

inline constexpr uint32_t kPrewarmDescVersion = 1;

struct PrewarmPipelineDesc {
    uint32_t kind = kPrewarmKindGraphics;
    uint64_t pack_key0 = 0;  // graphics: vertex pack key; compute: compute pack key
    uint64_t pack_key1 = 0;  // graphics: fragment pack key (0 = none)
    pso::GraphicsPipelineKey graphics{};
    pso::ComputePipelineKey compute{};
    bool has_layout = false;  // false = layout "auto" (pipeline_layout_id 0)
    bridge::PipelineLayoutDesc layout;
};

std::vector<uint8_t> encode_prewarm_pipeline_desc(const PrewarmPipelineDesc& desc);
// False when the blob is truncated, from another version or malformed.
bool decode_prewarm_pipeline_desc(const uint8_t* data, size_t size, PrewarmPipelineDesc* out);

}  // namespace playgta6::gpu
