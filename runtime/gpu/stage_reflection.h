#pragma once

// Stage reflection (G62, #651): what a translated (pack-resolved) stage reads, in terms both the
// translator and the D3D12 guest layer can compute, so the layer can build the WebGPU vertex
// state and pipeline layout the translated WGSL expects instead of numbering by convention.
//
// Per stage:
//   - vertex inputs: @location <- (vertex-buffer slot, byte offset in the vertex), with the fetch
//     format when the stage reads inside an input-layout element and the component type it expects;
//   - resource bindings: @group/@binding <- where the console stage loads the descriptor from: the
//     user-data SGPR holding a descriptor-table pointer plus the descriptor's byte offset in that
//     table, root constants or a root descriptor at a user-data SGPR, or a static sampler; plus
//     the binding's kind and minimum size.
//
// User data -> root parameter (console ABI, established from the stage metadata of every pipeline
// the title creates): root parameters occupy consecutive user-data SGPRs from s8 in graphics
// stages and from s4 in compute stages; a descriptor table takes one SGPR (32-bit pointer), root
// constants one per 32-bit value and a root descriptor two. A table's descriptors are 32-byte slots:
// slot = table byte offset / 32, slots numbered as D3D12 numbers them (ranges appended in order
// unless they give an explicit offset).
//
// Source: a sidecar `<pack>.reflect.json` next to the shader pack, keyed by the pack key (stage
// identity): {"schema": "playgta6-stage-reflection", "version": 1, "stages": {"0x<key>": {
//   "stage": "vertex" | "fragment" | "compute",
//   "vertex_inputs": [{"location", "vb_slot", "byte_offset", "components", "component_type":
//                      "f32" | "u32" | "i32", "fetch_format": "<WebGPU vertex format>" (optional)}],
//   "bindings": [{"group", "binding", "kind": "uniform" | "storage" | "read_only_storage" | "texture" |
//                 "storage_texture" | "sampler" | "comparison_sampler", "view_dimension" ("2d" ...),
//                 "sample_type" ("float" | "unfilterable_float" | "depth" | "sint" | "uint"),
//                 "multisampled", "min_binding_size_bytes",
//                 "source": {"user_sgpr", "table_byte_offset"} | {"user_sgpr", "root_constants": true} |
//                           {"user_sgpr", "root_descriptor": true} | {"static_sampler": <index>}}]}}}
// Stages without an entry keep the layer's conventions (layout_from_root_signature, input element i
// -> @location(i)).

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "abi/playgta6_abi.h"
#include "gpu/descriptor_heap.h"
#include "gpu/pso_cache.h"
#include "gpu/state_bridge.h"

namespace playgta6::gpu {

namespace d3d = playgta6::gpu::d3d12_state;

enum class ReflectedComponentType : uint8_t { F32 = 0, U32 = 1, I32 = 2 };

struct ReflectedVertexInput {
    uint32_t location = 0;
    uint32_t vb_slot = 0;
    uint32_t byte_offset = 0;
    uint32_t components = 4;
    ReflectedComponentType component_type = ReflectedComponentType::F32;
    pso::WgpuVertexFormat fetch_format = pso::WgpuVertexFormat::Undefined;  // Undefined: not given
};

enum class ReflectedBindingKind : uint8_t {
    Uniform = 0,
    Storage,
    ReadOnlyStorage,
    Texture,
    StorageTexture,
    Sampler,
    ComparisonSampler,
};

enum class ReflectedSource : uint8_t { Table = 0, RootConstants, RootDescriptor, StaticSampler };

struct ReflectedBinding {
    uint32_t group = 0;
    uint32_t binding = 0;
    ReflectedBindingKind kind = ReflectedBindingKind::Uniform;
    GpuTextureViewDimension view_dimension = GpuTextureViewDimension::Dim2D;
    GpuTextureSampleType sample_type = GpuTextureSampleType::Float;
    bool multisampled = false;
    uint8_t storage_format = 0;  // storage textures: WebGPU texture format code (0 = not given)
    uint64_t min_binding_size = 0;
    ReflectedSource source = ReflectedSource::Table;
    uint32_t user_sgpr = 0;           // Table / RootConstants / RootDescriptor
    uint32_t table_byte_offset = 0;   // Table
    uint32_t static_sampler = 0;      // StaticSampler: index in the root signature's static samplers
};

enum class ReflectedStage : uint8_t { Vertex = 1, Fragment = 2, Compute = 3 };

struct StageReflection {
    ReflectedStage stage = ReflectedStage::Vertex;
    std::vector<ReflectedVertexInput> vertex_inputs;
    std::vector<ReflectedBinding> bindings;
};

// Pack key (stage identity) -> reflection.
using StageReflectionIndex = std::unordered_map<uint64_t, StageReflection>;

// Parses a sidecar (see above). False + message when malformed; unknown members are ignored.
bool parse_stage_reflection_json(std::string_view json, StageReflectionIndex* out, std::string* error);

// The root parameter whose user data starts at `user_sgpr` (-1 when none does).
inline constexpr uint32_t kGraphicsUserDataBase = 8;
inline constexpr uint32_t kComputeUserDataBase = 4;
inline constexpr uint32_t kTableSlotBytes = 32;
int root_parameter_for_user_sgpr(const d3d::RootSignatureDesc& rs, bool compute, uint32_t user_sgpr);

// Pipeline layout for pack-resolved stages from their reflection joined to the root signature.
// `stages`: each stage's reflection with its GpuShaderStageBits. Bindings that two stages share
// (same group and binding) must agree on kind and source. False (with one line per failed key in
// *report) when a binding cannot be joined; no partial layout is produced then.
struct ReflectedStageRef {
    uint32_t visibility = 0;  // GpuShaderStageBits
    const StageReflection* reflection = nullptr;
};
bool join_pipeline_layout(const d3d::RootSignatureDesc& rs, bool compute, const std::vector<ReflectedStageRef>& stages,
                          bridge::PipelineLayoutDesc* out, std::string* report);

// Vertex attributes for a pack-resolved vertex stage: each reflected input is matched to the input
// element of its slot that covers its byte offset. An element starting at that offset gives its own
// format (its component class must match the input's: float for unorm/snorm/float formats, uint,
// sint); an input inside an element needs its fetch format. Elements no input reads are dropped.
// False (one line per failed location in *report) when an input has no element or the classes
// differ.
bool join_vertex_inputs(const StageReflection& vs, const pso::D3D12InputElementDesc* elements, uint32_t count,
                        std::vector<pso::VertexInputFetch>* out, std::string* report);

}  // namespace playgta6::gpu
