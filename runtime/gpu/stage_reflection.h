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
// constants one per 32-bit value and a root descriptor two. When the root signature enables SM 6.6
// directly-indexed descriptor heaps (`CbvSrvUavHeapDirectlyIndexed` = 0x400 and/or
// `SamplerHeapDirectlyIndexed` = 0x800), the console appends implicit 1-SGPR heap base pointers
// immediately after the explicit root parameters: `s_res = base + total_root_dwords` (when 0x400),
// then `s_smp = base + total_root_dwords + (has_res ? 1 : 0)` (when 0x800). CBV_SRV_UAV table and
// resource-heap descriptors are 32-byte slots (`slot = table_byte_offset / 32`); sampler table and
// sampler-heap descriptors are 16-byte slots (`slot = table_byte_offset / 16`), matching
// `descriptor_handle_increment_size` and GFX10 S# size.
// Slots are numbered as D3D12 numbers them (ranges appended in order unless they give an explicit
// offset).
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
//                           {"user_sgpr", "root_descriptor": true} | {"static_sampler": <index>} |
//                           {"sampler_heap" | "resource_heap": {"base_user_sgpr", "stride",
//                                                               "index": {"user_sgpr", "shift"}}} |
//                           {"inline_sampler": {"sdwords": [d0, d1, d2, d3]}} |
//                           {"cbv_indexed_resource_heap" | "cbv_indexed_sampler_heap": {
//                               "cbv": {"user_sgpr", "root_descriptor": true} | {"user_sgpr", "table_byte_offset"} |
//                                      {"user_sgpr", "root_constants": true},
//                               "byte_offset", "shift", "bit_offset" (0), "mask" (0xffffffff), "stride" (opt.)}} |
//                           {"lane_indexed_resource_heap" | "lane_indexed_sampler_heap": {
//                               "max_count", "shift", "stride" (opt.),
//                               "window_base" (opt.): {"cbv": <as above>, "byte_offset", "bit_offset" (0),
//                                                      "mask" (0xffffffff)}}} |
//                           {"unresolved": true | "<reason>"} (version >= 3)}]}}}
// An unresolved source (G62, #482; version >= 3): the translator reflected the binding but could
// not trace where the stage loads its descriptor from (mostly SM 6.6 bindless textures whose heap
// index is a constant-buffer dword at an offset computed from a root constant, which the CB-indexed
// source cannot express yet). The stage still joins: such a binding is bound as the worker's
// default for its kind (an unresolved register, like a sampler without a static sampler), counted
// in JoinStats::unresolved_bindings, instead of the whole stage falling back to conventions.
// Writable kinds (storage, storage_texture) cannot be unresolved: their writes would be lost.
// A lane-indexed heap source (G62, #482; SM 6.6 ResourceDescriptorHeap[NonUniformResourceIndex(i)]:
// a VGPR index, v_readfirstlane_b32 waterfall loops): each lane picks its own descriptor, so the
// stage reads a window of "max_count" consecutive heap descriptors as a WGSL
// binding_array<T, max_count>. Element k of the window is the descriptor at byte offset
// (window_base + k) << shift of the bound heap; window_base is the CB dword read like a CB-indexed
// source ((dword >> bit_offset) & mask), 0 without "window_base". The stage subtracts the same base
// from its lane index and bounds-checks against max_count. Such a binding joins as a binding array
// (GpuBindGroupLayoutEntry::binding_array_size = max_count; the bridge binds heap descriptors
// window_base + 0 .. max_count - 1, see d3d::ShaderBinding::array_size) when the consumer binds
// binding arrays (JoinOptions::binding_arrays, from bridge::BridgeOptions::binding_arrays; off by
// default). Otherwise it degrades to a single scalar heap binding at window_base + 0 (or constant
// slot 0 without window_base, binding_array_size = 0) and is counted in JoinStats::lane_indexed_bindings.
// A CB-indexed heap source (G62, #482; SM 6.6 ResourceDescriptorHeap[cb.index]): the index is the
// 32-bit dword at "byte_offset" of a constant buffer: the root CBV, the CBV in a table slot or the
// root constants of the root parameter whose user data starts at "cbv"."user_sgpr". The descriptor
// is the one at byte offset ((dword >> bit_offset) & mask) << shift of the bound heap, read at
// draw time; an index that cannot be read or falls outside the heap binds null (counted).
// An inline sampler (#482): the stage builds the sampler from constants in its code; "sdwords" are
// the four GFX10 S# dwords, decoded at the join (bridge::decode_sampler_words) and bound as that
// sampler. Words that do not decode as a sampler fail the join.
// A heap source (S34; SM 6.6 SamplerDescriptorHeap[i] / ResourceDescriptorHeap[i] under the root-
// signature flags SAMPLER_HEAP_DIRECTLY_INDEXED / CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED): the index is
// a root constant (the user SGPR inside a constants parameter); the descriptor is the one at byte
// offset (index << shift) of the bound heap, chosen at draw time.
// A root-descriptor source over a descriptor table (G62, #482): the translator reports
// "root_descriptor" when the stage builds a buffer from the user-data SGPR pair itself, but the
// root parameter that starts at that SGPR may be a descriptor table (one SGPR, a 32-bit pointer).
// Such a binding joins as the table's descriptor at "table_byte_offset" when the sidecar gives it
// next to "root_descriptor": true (version >= 2, kStageReflectionVersionRootTableOffset), else as
// the table's first descriptor (slot 0). Counted in JoinStats::root_descriptor_tables.
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

enum class ReflectedSource : uint8_t {
    Table = 0,
    RootConstants,
    RootDescriptor,
    StaticSampler,
    SamplerHeap,
    ResourceHeap,
    InlineSampler,  // the stage builds the S# from constants in its code (sampler_words)
    CbvIndexedSamplerHeap,   // heap index read from a constant buffer (cb_*), G62
    CbvIndexedResourceHeap,
    LaneIndexedSamplerHeap,  // per-lane index into a window of the heap (lane_*), G62
    LaneIndexedResourceHeap,
    Unresolved,  // origin not traced by the translator: bound as the default (version >= 3), G62
};

// Largest lane-indexed window a sidecar may ask for (WebGPU's per-stage sampled-texture limit we
// request is at most 256).
inline constexpr uint32_t kMaxLaneIndexedWindow = 256;

// Where a CB-indexed heap source's constant buffer is (the root parameter at user_sgpr).
enum class ReflectedCbv : uint8_t { RootDescriptor = 0, Table = 1, RootConstants = 2 };

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
    // SamplerHeap / ResourceHeap (S34): the descriptor at byte offset (user SGPR heap_index_sgpr <<
    // heap_shift) of the heap whose base is in user SGPR heap_base_sgpr; heap_stride is the
    // descriptor size the stage assumes.
    uint32_t heap_base_sgpr = 0;
    uint32_t heap_stride = 0;
    uint32_t heap_index_sgpr = 0;
    uint32_t heap_shift = 0;
    // InlineSampler: the four GFX10 S# dwords the stage builds in SGPRs (decoded at the join).
    uint32_t sampler_words[4] = {0, 0, 0, 0};
    // CbvIndexed*Heap: the CB is the root parameter at user_sgpr (a Table CB: the CBV at
    // table_byte_offset); index = (dword at cb_byte_offset >> cb_bit_offset) & cb_mask, byte offset in
    // the heap index << heap_shift; heap_stride as given (0 when not).
    ReflectedCbv cbv = ReflectedCbv::RootDescriptor;
    uint32_t cb_byte_offset = 0;
    uint32_t cb_bit_offset = 0;
    uint32_t cb_mask = 0xffffffffu;
    // LaneIndexed*Heap: a window of lane_max_count descriptors starting at heap index window_base
    // (heap_shift, heap_stride as for the heap sources). lane_has_window_base: the base is read like
    // a CB-indexed index (cbv, user_sgpr, table_byte_offset, cb_*); otherwise it is 0.
    uint32_t lane_max_count = 0;
    bool lane_has_window_base = false;
    // RootDescriptor: the sidecar gave table_byte_offset with it (version >= 2), used when the root
    // parameter at user_sgpr is a descriptor table; otherwise such a table joins at slot 0.
    bool root_table_offset_given = false;
};

enum class ReflectedStage : uint8_t { Vertex = 1, Fragment = 2, Compute = 3 };

// Inter-stage parameter semantic mapping entry from Xbox Series FSSM stage metadata (#2085):
//   - vs_outputs (OSG1): semantic_reg -> exp param<location>
//   - ps_inputs  (ISG1): semantic_reg -> v_interp attr<location>, with optional component mask (0..15)
struct ReflectedStageParam {
    uint8_t semantic_reg = 0;
    uint32_t location = 0;
    uint8_t mask = 0x0fu;
    bool has_mask = false;
};

// Non-identity SPI_PS_INPUT_CNTL inter-stage @location remap: rewrite PS input @location(ps_location)
// to read VS output @location(vs_location) for matching FSSM semantic_reg (#2085).
struct InterstageLocationRemap {
    uint8_t semantic_reg = 0;
    uint32_t ps_location = 0;
    uint32_t vs_location = 0;
};

struct StageReflection {
    ReflectedStage stage = ReflectedStage::Vertex;
    std::vector<ReflectedVertexInput> vertex_inputs;
    std::vector<ReflectedBinding> bindings;
    // Optional FSSM inter-stage parameter semantic maps (#2085).
    std::vector<ReflectedStageParam> vs_outputs;
    std::vector<ReflectedStageParam> ps_inputs;
    // G66: a mesh stage translated to compute (pack key bridge::mesh_compute_pack_key): its output
    // record layout, from the optional "mesh_output": {"max_vertices", "max_primitives",
    // "vertex_words"} (see bridge::MeshOutputLayout; validated by bridge::valid_mesh_output_layout).
    bool has_mesh_output = false;
    bridge::MeshOutputLayout mesh_output{};
    // RT arena bindings (docs/RAYTRACING_DESIGN.md C3): the stage uses the runtime-owned prelude
    // (group 3, bindings 16-21). Bit i = binding 16 + i. Set by a stage-level "rt_arena": true (all
    // six) or by bindings {"group": 3, "binding": 16..21, "source": {"rt_arena": true}}; those
    // bindings are runtime-owned and are not joined to the root signature.
    uint32_t rt_arena_mask = 0;
    // C2 "address_sources" (rt_residency.h): which root params / constants hold guest addresses.
    std::vector<rt_residency::AddressSource> address_sources;
};

// Pack key (stage identity) -> reflection.
using StageReflectionIndex = std::unordered_map<uint64_t, StageReflection>;

// Sidecar versions this parser reads. Version 2 adds the optional "table_byte_offset" of a
// "root_descriptor" source (ignored in version 1 documents); version 3 adds the "unresolved" source
// (malformed in older documents).
inline constexpr uint32_t kStageReflectionVersion = 1;
inline constexpr uint32_t kStageReflectionVersionRootTableOffset = 2;
inline constexpr uint32_t kStageReflectionVersionUnresolved = 3;
inline constexpr uint32_t kStageReflectionMaxVersion = 3;

// Parses a sidecar (see above). False + message when malformed; unknown members are ignored.
bool parse_stage_reflection_json(std::string_view json, StageReflectionIndex* out, std::string* error);

// The root parameter whose user data starts at `user_sgpr` (-1 when none does).
inline constexpr uint32_t kGraphicsUserDataBase = 8;
inline constexpr uint32_t kComputeUserDataBase = 4;
// The mesh stage (lowered to compute) reads the graphics root arguments from s16 on: its reflections
// put the first root parameter's table at s16 (s8..s15 hold the mesh/NGG system values).
inline constexpr uint32_t kMeshUserDataBase = 16;
inline constexpr uint32_t kTableSlotBytes = 32;
// `base` (0 = the graphics or compute default) overrides the first root parameter's user SGPR.
int root_parameter_for_user_sgpr(const d3d::RootSignatureDesc& rs, bool compute, uint32_t user_sgpr, uint32_t base = 0);
// The root parameter whose user data *covers* `user_sgpr` (-1 when none), with the dword index into it.
int root_parameter_containing_user_sgpr(const d3d::RootSignatureDesc& rs, bool compute, uint32_t user_sgpr,
                                        uint32_t* dword, uint32_t base = 0);

// Pipeline layout for pack-resolved stages from their reflection joined to the root signature.
// `stages`: each stage's reflection with its GpuShaderStageBits. Bindings that two stages share
// (same group and binding) must agree on kind and source. False (with one line per failed key in
// *report) when a binding cannot be joined; no partial layout is produced then.
struct ReflectedStageRef {
    uint32_t visibility = 0;  // GpuShaderStageBits
    const StageReflection* reflection = nullptr;
    // Optional per-stage first root parameter's user SGPR (0: use JoinOptions::user_data_base / default).
    uint32_t user_data_base = 0;
};
// Join statistics (added to on success only, except lane_indexed_bindings).
struct JoinStats {
    // Table bindings joined by slot (ShaderBinding::table_parameter): the root signature does not
    // declare the slot, or declares it as another class.
    uint64_t table_slots_outside_rs = 0;
    // Lane-indexed heap bindings (LaneIndexed*Heap) degraded to a single scalar heap binding at the
    // window base because the consumer does not bind binding arrays (JoinOptions::binding_arrays off).
    // Counted for telemetry whether or not the join succeeds.
    uint64_t lane_indexed_bindings = 0;
    // Lane-indexed heap bindings joined as binding arrays (JoinOptions::binding_arrays; on success).
    uint64_t lane_indexed_arrays = 0;
    // Root-descriptor bindings whose user SGPR starts a descriptor table, joined as a table slot
    // (the given table_byte_offset, else the first descriptor; on success).
    uint64_t root_descriptor_tables = 0;
    // Vertex inputs no input-layout element covers, read as D3D12's (0, 0, 0, 1) defaults from a
    // shared constant buffer (join_vertex_inputs; on success).
    uint64_t vertex_inputs_defaulted = 0;
    // Bindings with an unresolved source, bound as the worker's default (on success).
    uint64_t unresolved_bindings = 0;
    // Non-compute stages whose user-data SGPRs resolved via GFX10.3 shifted-base fallback (e.g.
    // s16 for NGG/merged VS or s12 for 4-SGPR-shifted VS) instead of the default s8 (on success).
    uint64_t shifted_user_data_base_stages = 0;
    // Bindings resolved via SM 6.6 trailing implicit directly-indexed descriptor heap SGPRs
    // (`s_res` for CbvSrvUavHeapDirectlyIndexed, `s_smp` for SamplerHeapDirectlyIndexed; on success).
    uint64_t implicit_heap_bindings = 0;
};
struct JoinOptions {
    // The consumer binds binding arrays (GpuLayoutFlags::BindingArrays): lane-indexed heap windows
    // join as arrays instead of failing the join.
    bool binding_arrays = false;
    // First root parameter's user SGPR (0: the graphics / compute default); kMeshUserDataBase for a mesh stage.
    uint32_t user_data_base = 0;
};
bool join_pipeline_layout(const d3d::RootSignatureDesc& rs, bool compute, const std::vector<ReflectedStageRef>& stages,
                          bridge::PipelineLayoutDesc* out, std::string* report, JoinStats* stats = nullptr,
                          const JoinOptions& options = {});

// Vertex attributes for a pack-resolved vertex stage: each reflected input is matched to the input
// element of its slot that covers its byte offset. An element starting at that offset gives its own
// format (its component class must match the input's: float for unorm/snorm/float formats, uint,
// sint); an input inside an element needs its fetch format. Elements no input reads are dropped.
// An input no element covers is not a failure: D3D12 reads (0, 0, 0, 1), so it becomes a
// `constant_default` fetch. False (one line per failed location in *report) when the classes differ.
bool join_vertex_inputs(const StageReflection& vs, const pso::D3D12InputElementDesc* elements, uint32_t count,
                        std::vector<pso::VertexInputFetch>* out, std::string* report, JoinStats* stats = nullptr);

// Compute non-identity PS->VS @location remaps by matching `ps.ps_inputs[i].semantic_reg` against
// `vs.vs_outputs[j].semantic_reg` (SPI_PS_INPUT_CNTL, #2085). Skips unused PS inputs (`has_mask && mask == 0`)
// and identity mappings (`ps_location == vs_location`). Results are sorted by ascending `ps_location`.
std::vector<InterstageLocationRemap> compute_interstage_location_remaps(const StageReflection& vs,
                                                                        const StageReflection& ps);

}  // namespace playgta6::gpu
