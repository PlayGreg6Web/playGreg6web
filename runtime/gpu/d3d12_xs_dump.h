#pragma once

// Pipeline dump (G37): local files for the shader track, written while the title runs.
//
// Off unless DeviceOptions::dump_dir is set (boot harness: --dump-pipelines <dir>, which must lie
// outside a repository checkout, or under a `work` directory). The files are derived from the running title, like captures: keep
// them out of the repository.
//
// Layout of <dir>:
//   native/<code-identity>.bin    exact code range of a console-native stage (Xbox slot 57)
//   native/<code-identity>.meta   that stage's metadata range; a second metadata range seen with
//                                 the same code is written as <code-identity>-<identity>.meta
//   dxil/<identity>.dxbc          a DXBC container passed to a pipeline (slots 86, 10, 11, 62)
//   manifest.json                 rewritten every 60 presents and on flush
// Every identity is spelled "0x%016x" (FNV-1a 64, shader_identity / native_stage_identity). A file
// is named by the FNV-1a 64 of its own bytes (.bin, .dxbc), so any tool that hashes the file gets
// the identity the guest layer uses; the code+metadata identity is in the manifest.
//
// manifest.json (schema "playgta6-pipeline-dump", schema_version 1):
//   native_stages: [{code_identity, identity, stage ("vertex" | "pixel" | "compute" | "mesh"), layout ("A" | "B"),
//                    code_bytes, metadata_bytes, code_file, metadata_file}]
//   dxil: [{identity, hash_digest (32 hex digits, the HASH part; "" if none), kind (DXIL program
//           kind: 0 pixel, 1 vertex, 5 compute, ...; -1 unknown), bytes, file}]
//   root_signatures: [{index, version ("1.0" | "1.1" | "1.2"), flags, static_samplers,
//                      parameters: [{type ("table" | "constants" | "cbv" | "srv" | "uav"),
//                                    visibility, register, space, num_32bit_values,
//                                    ranges: [{type ("srv" | "uav" | "cbv" | "sampler"), count
//                                              (-1 unbounded), base_register, space, offset}]}]}]
//   pipelines: [{index, created (pipeline states with this entry), draws (draw calls recorded with
//                it bound, skipped ones included),
//                vs / ps: null or {dxil_identity, hash_digest, native_code_identity,
//                                  native_identity} (native_* "" when not joined),
//                root_signature (index or -1), topology_type, rect_list, sample_mask,
//                sample_count, rtv_formats [DXGI], dsv_format, ib_strip_cut,
//                blend {alpha_to_coverage, independent, targets: [{enable, logic_enable, src, dst,
//                       op, src_alpha, dst_alpha, op_alpha, logic_op, write_mask}] (one per
//                       render target)},
//                depth_stencil {depth_enable, depth_write_mask, depth_func, stencil_enable,
//                               read_mask, write_mask, front [fail, depth_fail, pass, func],
//                               back [...], depth_bounds},
//                rasterizer {fill, cull, front_ccw, depth_bias, depth_bias_clamp,
//                            slope_scaled_depth_bias, depth_clip, multisample, antialiased_line,
//                            forced_sample_count, conservative},
//                input_layout: [{semantic, index, format, slot, offset, per_instance, step_rate}]}]
//   compute_pipelines: [{index, created, dispatches,
//                        cs: null or {dxil_identity, hash_digest, native_code_identity,
//                                     native_identity},
//                        root_signature (index or -1)}]
// Enum values are the public D3D12 / DXGI numbers.

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace playgta6::gpu::d3d12_state {
struct RootSignatureDesc;
}

namespace playgta6::gpu::d3d12_guest {

struct GraphicsPsoSource;

class PipelineDump {
public:
    explicit PipelineDump(std::string dir);
    bool ok() const { return ok_; }
    const std::string& dir() const { return dir_; }

    enum class NativeStageKind : uint8_t { Vertex = 0, Pixel = 1, Compute = 2, Mesh = 3 };

    void native_stage(uint64_t code_identity, uint64_t identity, bool pixel, bool layout_b, const uint8_t* code,
                      size_t code_size, const uint8_t* metadata, size_t metadata_size);
    void native_stage(uint64_t code_identity, uint64_t identity, NativeStageKind stage, bool layout_b,
                      const uint8_t* code, size_t code_size, const uint8_t* metadata, size_t metadata_size);
    void dxil(uint64_t identity, const std::array<uint8_t, 16>* hash_digest, int kind, const uint8_t* bytes,
              size_t size);
    // Records a graphics pipeline (deduplicated by its manifest entry); returns its index.
    // `root_signature` identifies the root signature object (null: none).
    uint32_t pipeline(const GraphicsPsoSource& source, const std::array<uint8_t, 16>& vs_digest,
                      const std::array<uint8_t, 16>& ps_digest, const void* root_signature,
                      const d3d12_state::RootSignatureDesc* root_desc);
    void draw(uint32_t pipeline_index);
    // Records a compute pipeline (deduplicated by its manifest entry); returns its index.
    uint32_t compute_pipeline(uint64_t cs_identity, const std::array<uint8_t, 16>& cs_digest,
                              uint64_t native_code_identity, uint64_t native_identity, const void* root_signature,
                              const d3d12_state::RootSignatureDesc* root_desc);
    void dispatch(uint32_t compute_pipeline_index);
    // Writes manifest.json (atomically: temporary file, then rename).
    void write_manifest();

private:
    bool write_file(const std::string& path, const uint8_t* bytes, size_t size);
    int intern_root_signature_locked(const void* root_signature, const d3d12_state::RootSignatureDesc* root_desc);

    std::string dir_;
    bool ok_ = false;
    mutable std::mutex mutex_;
    std::vector<std::string> native_entries_;
    std::unordered_set<uint64_t> native_seen_;          // stage identities
    std::unordered_map<uint64_t, uint64_t> code_meta_;  // code identity -> identity of the .meta written
    std::vector<std::string> dxil_entries_;
    std::unordered_set<uint64_t> dxil_seen_;
    std::vector<std::string> root_entries_;
    std::unordered_map<const void*, int> root_index_;
    struct Pipeline {
        std::string body;  // manifest entry without index and counters
        uint64_t created = 0;
        uint64_t draws = 0;
    };
    std::vector<Pipeline> pipelines_;
    std::map<std::string, uint32_t> pipeline_index_;
    struct ComputePipeline {
        std::string body;  // manifest entry without index and counters
        uint64_t created = 0;
        uint64_t dispatches = 0;
    };
    std::vector<ComputePipeline> compute_pipelines_;
    std::map<std::string, uint32_t> compute_pipeline_index_;
};

}  // namespace playgta6::gpu::d3d12_guest
