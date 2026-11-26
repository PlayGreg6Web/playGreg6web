// Native executor reflection loader (gpu/native_exec/reflection_model.h), checked against the rules of
// web/workers/reflection_loader.js: the `playgta6-stage-reflection` sidecar (v1..v3), the S6
// layout-manifest.json, the registry lookups, the merged bind group layout descriptors and the
// reflected vertex buffer layout. No GPU involved.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/native_exec/m1_model.h"
#include "gpu/native_exec/reflection_model.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_native_exec_reflection.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu::native_exec;
using namespace playgta6::gpu::native_exec::reflect;
using K = LayoutEntryDesc::Kind;

std::string sidecar(int version, const std::string& stages) {
    return std::string(R"({"schema":"playgta6-stage-reflection","version":)") + std::to_string(version) + R"(,"stages":{)" + stages + "}}";
}

const char* kVs = R"("0x00000000000000a1":{"stage":"vertex","entry_point":"vtx",
  "vertex_inputs":[
    {"location":2,"vb_slot":0,"byte_offset":8,"components":4,"component_type":"f32","fetch_format":"unorm8x4"},
    {"location":0,"vb_slot":0,"byte_offset":0,"components":2,"component_type":"f32","fetch_format":"float32x2"}],
  "bindings":[
    {"group":1,"binding":0,"kind":"uniform","min_binding_size_bytes":16,"source":{"user_sgpr":8,"table_byte_offset":0}},
    {"group":1,"binding":3,"kind":"texture","view_dimension":"2d","sample_type":"unfilterable_float","source":{"user_sgpr":8,"table_byte_offset":32}},
    {"group":1,"binding":4,"kind":"sampler","source":{"static_sampler":0}}]})";
const char* kFs = R"("0xA2":{"stage":"pixel",
  "bindings":[
    {"group":1,"binding":1,"kind":"uniform","source":{"user_sgpr":9,"root_constants":true}},
    {"group":1,"binding":3,"kind":"texture","view_dimension":"2d","sample_type":"float","source":{"user_sgpr":8,"table_byte_offset":32}},
    {"group":1,"binding":4,"kind":"sampler","source":{"static_sampler":0}}]})";

void test_json() {
    Json j;
    parse_json(R"({"a":[1,2.5,-3e2,true,false,null,"x\u00e9\n"],"b":{"c":"d"},"a":7,"9":1,"1":2,"z":0})", &j);
    CHECK(j.is_obj());
    // a duplicate key keeps its first position and takes the last value; array-index keys come first.
    CHECK(j.o.size() == 5);
    CHECK(j.o[0].first == "1" && j.o[1].first == "9" && j.o[2].first == "a" && j.o[3].first == "b" && j.o[4].first == "z");
    CHECK(j.get("a")->is_num() && j.get("a")->n == 7);
    CHECK(j.get("missing") == nullptr);
    Json k;
    parse_json(R"(["x\u00e9\ud83d\ude00", -3e2, 2.5e-1])", &k);
    CHECK(k.a[0].s == "x\xc3\xa9\xf0\x9f\x98\x80" && k.a[1].n == -300 && k.a[2].n == 0.25);
    const char* bad[] = {"", "{", "[1,]", "{\"a\":1,}", "01", "1.", "\"a", "tru", "{\"a\" 1}", "[1] x", "'a'", "{\"a\":\"\\x\"}", "-"};
    for (const char* b : bad) {
        Json x;
        bool threw = false;
        try {
            parse_json(b, &x);
        } catch (const ReflectError&) {
            threw = true;
        }
        CHECK(threw);
    }
}

void test_sidecar_stage_records() {
    ReflectionRegistry reg;
    CHECK(reg.shader_count() == 0);
    const std::string err = reg.load(sidecar(3, std::string(kVs) + "," + kFs));
    CHECK(err.empty());
    CHECK(reg.shader_count() == 2);
    const StageRecord* vs = reg.by_hash(0xa1);
    const StageRecord* fs = reg.by_hash(0xa2);
    CHECK(vs && fs);
    CHECK(vs->stage == "vertex" && vs->entry_point == "vtx" && vs->visibility == kVisVertex);
    CHECK(fs->stage == "fragment" && fs->entry_point == "fs_main" && fs->visibility == kVisFragment);  // "pixel" + default entry
    CHECK(vs->hash_hex == "0x00000000000000a1");
    // vertex inputs sorted by location; the format is the fetch format
    CHECK(vs->vertex_inputs.size() == 2 && vs->vertex_inputs[0].location == 0 && vs->vertex_inputs[1].location == 2);
    CHECK(vs->vertex_inputs[0].format == "float32x2" && vs->vertex_inputs[1].format == "unorm8x4");
    // group layouts: group 0 empty, group 1 sorted by binding, visibility of the stage
    CHECK(vs->group_layouts.size() == 2 && vs->group_layouts[0].empty() && vs->group_layouts[1].size() == 3);
    const auto& g1 = vs->group_layouts[1];
    CHECK(g1[0].binding == 0 && g1[0].kind == K::Buffer && g1[0].buffer_type == 0 && g1[0].min_binding_size == 16);
    CHECK(g1[1].binding == 3 && g1[1].kind == K::Texture && g1[1].sample_type == 1 && g1[1].view_dim == 2);  // unfilterable-float
    CHECK(g1[2].binding == 4 && g1[2].kind == K::Sampler && g1[2].sampler_type == 0 && g1[2].visibility == kVisVertex);
    // lookups: by hash only (sidecar stages are not indexed by entry point)
    CHECK(reg.lookup_hash(0xa1) == vs && reg.lookup_hash(0x77) == nullptr);
    CHECK(reg.lookup_name("vertex:vtx") == nullptr);
    CHECK(reg.stats().lookup_hits == 1 && reg.stats().lookup_misses == 2);
}

void test_sidecar_validation() {
    auto rejects = [](int version, const std::string& stages) {
        ReflectionRegistry r;
        const bool bad = !r.load(sidecar(version, stages)).empty();
        return bad && r.shader_count() == 0;
    };
    auto accepts = [](int version, const std::string& stages) {
        ReflectionRegistry r;
        return r.load(sidecar(version, stages)).empty() && r.shader_count() == 1;
    };
    auto stage = [](const std::string& body) { return R"("0x1":{"stage":"compute",)" + body + "}"; };
    auto bnd = [](const std::string& b) { return R"("bindings":[)" + b + "]"; };
    const std::string storage = R"({"group":0,"binding":0,"kind":"storage","source":)";

    CHECK(accepts(3, stage(bnd(storage + R"({"user_sgpr":4,"table_byte_offset":0}})"))));
    // version window and schema
    ReflectionRegistry r0;
    CHECK(!r0.load(R"({"schema":"other","version":1,"stages":{}})").empty());
    CHECK(!r0.load(R"({"schema":"playgta6-stage-reflection","version":0,"stages":{}})").empty());
    CHECK(!r0.load(R"({"schema":"playgta6-stage-reflection","version":4,"stages":{}})").empty());
    CHECK(!r0.load(R"({"schema":"playgta6-stage-reflection","version":1.5,"stages":{}})").empty());
    CHECK(!r0.load(R"({"schema":"playgta6-stage-reflection","version":1})").empty());  // stages missing
    CHECK(r0.load(R"({"schema":"playgta6-stage-reflection","version":1,"stages":{}})").empty() && r0.shader_count() == 0);
    // bad hash / stage / duplicates
    CHECK(rejects(1, R"("0xzz":{"stage":"compute","bindings":[]})"));
    CHECK(rejects(1, R"("0x1":{"stage":"geometry","bindings":[]})"));
    CHECK(rejects(1, R"("0x1":{"stage":"compute","bindings":[]},"0x0000000000000001":{"stage":"compute","bindings":[]})"));
    CHECK(rejects(1, R"("0x1":{"stage":"compute"})"));  // bindings missing
    CHECK(rejects(1, R"("0x1":5)"));
    CHECK(rejects(1, R"("0x11111111111111111":{"stage":"compute","bindings":[]})"));  // 17 digits
    // duplicate slot, unknown kind, missing source, bad group
    CHECK(rejects(1, stage(bnd(storage + R"({"static_sampler":0}},)" + storage + R"({"static_sampler":0}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"image","source":{"static_sampler":0}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"sampler"})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":-1,"binding":0,"kind":"sampler","source":{"static_sampler":0}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0.5,"kind":"sampler","source":{"static_sampler":0}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":64,"binding":0,"kind":"sampler","source":{"static_sampler":0}})"))));  // group limit
    CHECK(accepts(1, stage(bnd(R"({"group":63,"binding":0,"kind":"sampler","source":{"static_sampler":0}})"))));
    // unresolved sources: version >= 3 only, never for a writable kind
    const std::string unresolved = R"({"group":0,"binding":0,"kind":"texture","source":{"unresolved":"bindless"}})";
    CHECK(accepts(3, stage(bnd(unresolved))));
    CHECK(rejects(2, stage(bnd(unresolved))));
    CHECK(rejects(3, stage(bnd(storage + R"({"unresolved":true}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"storage_texture","storage_format":"rgba8unorm","source":{"unresolved":true}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","source":{"unresolved":false}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","source":{"unresolved":"x","user_sgpr":1}})"))));
    // root_descriptor over a table: the offset is read from version 2 on (and then validated)
    const std::string rd = R"({"group":0,"binding":0,"kind":"uniform","source":{"user_sgpr":8,"root_descriptor":true,"table_byte_offset":)";
    CHECK(accepts(1, stage(bnd(rd + R"(-4}})"))));
    CHECK(accepts(2, stage(bnd(rd + R"(64}})"))));
    CHECK(rejects(2, stage(bnd(rd + R"(-4}})"))));
    // user_sgpr sources must pick exactly one mode
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"uniform","source":{"user_sgpr":8}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"uniform","source":{"user_sgpr":8,"root_constants":true,"root_descriptor":true}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"uniform","source":{"table_byte_offset":8}})"))));
    // heap / inline sampler / CBV-indexed / lane-indexed sources
    CHECK(accepts(3, stage(bnd(R"({"group":0,"binding":0,"kind":"sampler","source":{"sampler_heap":{"base_user_sgpr":8,"stride":16,"index":{"user_sgpr":9,"shift":4}}}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"sampler","source":{"sampler_heap":{"base_user_sgpr":8,"stride":0,"index":{"user_sgpr":9,"shift":4}}}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"sampler","source":{"sampler_heap":{"base_user_sgpr":8,"stride":16,"index":{"user_sgpr":9,"shift":32}}}})"))));
    CHECK(accepts(3, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","source":{"resource_heap":true}})"))));
    CHECK(accepts(3, stage(bnd(R"({"group":0,"binding":0,"kind":"sampler","source":{"inline_sampler":{"sdwords":[1,2,3,4294967295]}}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"sampler","source":{"inline_sampler":{"sdwords":[1,2,3]}}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"sampler","source":{"inline_sampler":{"sdwords":[1,2,3,4294967296]}}})"))));
    const std::string cbv = R"({"cbv":{"user_sgpr":9,"root_descriptor":true},"byte_offset":8,"shift":5,"bit_offset":2,"mask":65535,"stride":32})";
    CHECK(accepts(3, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","source":{"cbv_indexed_resource_heap":)" + cbv + "}}"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","source":{"cbv_indexed_resource_heap":{"cbv":{"user_sgpr":9,"root_descriptor":true},"byte_offset":6,"shift":5}}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","source":{"cbv_indexed_resource_heap":{"cbv":{"user_sgpr":9},"byte_offset":4,"shift":5}}})"))));
    CHECK(accepts(3, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","source":{"lane_indexed_resource_heap":{"max_count":16,"shift":5,"window_base":)" + cbv + "}}}")))) ;
    CHECK(rejects(3, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","source":{"lane_indexed_resource_heap":{"max_count":257,"shift":5}}})"))));
    // RT arena: runtime-owned (group 3, binding 16..21), left out of the layout
    const std::string arena = R"({"group":3,"binding":17,"kind":"storage","source":{"rt_arena":true}})";
    CHECK(accepts(3, stage(bnd(arena))));
    CHECK(rejects(3, stage(bnd(R"({"group":3,"binding":22,"kind":"storage","source":{"rt_arena":true}})"))));
    CHECK(rejects(3, stage(bnd(R"({"group":2,"binding":17,"kind":"storage","source":{"rt_arena":true}})"))));
    CHECK(rejects(3, stage(R"("rt_arena":"yes","bindings":[])")));
    CHECK(accepts(3, stage(R"("rt_arena":true,"bindings":[])")));
    {
        ReflectionRegistry r;
        CHECK(r.load(sidecar(3, stage(bnd(arena + "," + storage + R"({"user_sgpr":4,"table_byte_offset":0}})")))).empty());
        const StageRecord* s = r.by_hash(1);
        CHECK(s && s->group_layouts.size() == 1 && s->group_layouts[0].size() == 1);  // the arena binding is not in the layout
    }
    // texture details
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","view_dimension":"cube-sphere","source":{"static_sampler":0}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","sample_type":"half","source":{"static_sampler":0}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"texture","multisampled":true,"view_dimension":"2d-array","source":{"static_sampler":0}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"storage_texture","view_dimension":"cube","source":{"static_sampler":0}})"))));
    CHECK(rejects(1, stage(bnd(R"({"group":0,"binding":0,"kind":"uniform","min_binding_size_bytes":-1,"source":{"static_sampler":0}})"))));
    // mesh_output
    CHECK(accepts(3, stage(R"("mesh_output":{"max_vertices":64,"max_primitives":126,"vertex_words":20},"bindings":[])")));
    CHECK(rejects(3, stage(R"("mesh_output":{"max_vertices":64,"max_primitives":126,"vertex_words":18},"bindings":[])")));
    CHECK(rejects(3, stage(R"("mesh_output":{"max_vertices":0,"max_primitives":1,"vertex_words":4},"bindings":[])")));
    // vertex inputs
    CHECK(rejects(1, stage(R"("vertex_inputs":[{"location":-1}],"bindings":[])")));
    CHECK(rejects(1, stage(R"("vertex_inputs":{},"bindings":[])")));
    CHECK(rejects(1, stage(R"("vertex_input_mode":"","bindings":[])")));
    CHECK(accepts(1, stage(R"("vertex_input_mode":"attributes","bindings":[])")));
    // a failed document registers nothing, an earlier one stays
    ReflectionRegistry keep;
    CHECK(keep.load(sidecar(1, stage(bnd(storage + R"({"static_sampler":0}})")))).empty());
    CHECK(!keep.load(sidecar(1, R"("0x2":{"stage":"compute","bindings":[]},"0x3":{"stage":"nope","bindings":[]})")).empty());
    CHECK(keep.shader_count() == 1 && keep.by_hash(1) && !keep.by_hash(2));
    CHECK(keep.stats().load_errors == 1);
}

void test_sidecar_kind_aliases_and_storage_formats() {
    ReflectionRegistry r;
    const std::string stg = R"("0x5":{"stage":"fragment","bindings":[
      {"group":0,"binding":0,"kind":"READ-ONLY-STORAGE","source":{"static_sampler":0}},
      {"group":0,"binding":1,"kind":"texture_2d","source":{"static_sampler":0}},
      {"group":0,"binding":2,"kind":"comparison-sampler","source":{"static_sampler":0}},
      {"group":0,"binding":3,"kind":"storage_texture","storage_format":"rgba16float","view_dimension":"2D_ARRAY","source":{"static_sampler":0}},
      {"group":0,"binding":4,"kind":"storage_texture","storage_format":7,"source":{"static_sampler":0}},
      {"group":0,"binding":5,"kind":"storage_texture","source":{"static_sampler":0}}]})";
    CHECK(r.load(sidecar(1, stg)).empty());
    const StageRecord* s = r.by_hash(5);
    CHECK(s);
    // a storage texture without a string format yields no layout entry (layout_complete = false)
    CHECK(!s->layout_complete && s->group_layouts.size() == 1 && s->group_layouts[0].size() == 4);
    const auto& g = s->group_layouts[0];
    CHECK(g[0].kind == K::Buffer && g[0].buffer_type == 2);
    CHECK(g[1].kind == K::Texture && g[1].view_dim == 2 && g[1].sample_type == 0);
    CHECK(g[2].kind == K::Sampler && g[2].sampler_type == 2);
    CHECK(g[3].kind == K::StorageTexture && g[3].binding == 3 && g[3].storage_format == "rgba16float" && g[3].view_dim == 3 && g[3].access == 0);
}

void test_layout_merge() {
    ReflectionRegistry reg;
    CHECK(reg.load(sidecar(3, std::string(kVs) + "," + kFs)).empty());
    const StageRecord* vs = reg.by_hash(0xa1);
    const StageRecord* fs = reg.by_hash(0xa2);
    LayoutBuildResult r = build_bind_group_layout_descriptors({vs, fs});
    CHECK(r.ok && r.groups.size() == 2 && r.groups[0].empty());
    const auto& g1 = r.groups[1];
    CHECK(g1.size() == 4);
    CHECK(g1[0].binding == 0 && g1[0].visibility == kVisVertex);
    CHECK(g1[1].binding == 1 && g1[1].visibility == kVisFragment && g1[1].min_binding_size == 0);
    // the texture both stages use: float wins over unfilterable-float, visibilities are OR-ed
    CHECK(g1[2].binding == 3 && g1[2].sample_type == 0 && g1[2].visibility == (kVisVertex | kVisFragment));
    CHECK(g1[3].binding == 4 && g1[3].kind == K::Sampler && g1[3].visibility == (kVisVertex | kVisFragment));
    // order of the stages does not change the result
    LayoutBuildResult r2 = build_bind_group_layout_descriptors({fs, vs});
    CHECK(r2.ok && r2.groups[1].size() == 4 && r2.groups[1][2].sample_type == 0);
    // nothing reflected -> no groups (but a valid, empty layout)
    ReflectionRegistry empty;
    CHECK(empty.load(sidecar(1, R"("0x9":{"stage":"compute","bindings":[]})")).empty());
    LayoutBuildResult e = build_bind_group_layout_descriptors({empty.by_hash(9)});
    CHECK(e.ok && e.groups.empty());

    // conflicts: a different buffer type / sampler type / view dimension / kind
    auto conflict = [](const std::string& a, const std::string& b) {
        ReflectionRegistry g;
        const std::string doc = sidecar(1, R"("0x1":{"stage":"vertex","bindings":[)" + a + R"(]},"0x2":{"stage":"fragment","bindings":[)" + b + "]}");
        if (!g.load(doc).empty()) return false;
        LayoutBuildResult x = build_bind_group_layout_descriptors({g.by_hash(1), g.by_hash(2)});
        return !x.ok && x.conflict;
    };
    const std::string src = R"("source":{"static_sampler":0}})";
    CHECK(conflict(R"({"group":0,"binding":0,"kind":"uniform",)" + src, R"({"group":0,"binding":0,"kind":"storage",)" + src));
    CHECK(conflict(R"({"group":0,"binding":0,"kind":"sampler",)" + src, R"({"group":0,"binding":0,"kind":"comparison_sampler",)" + src));
    CHECK(conflict(R"({"group":0,"binding":0,"kind":"texture","view_dimension":"2d",)" + src, R"({"group":0,"binding":0,"kind":"texture","view_dimension":"3d",)" + src));
    CHECK(conflict(R"({"group":0,"binding":0,"kind":"texture","sample_type":"depth",)" + src, R"({"group":0,"binding":0,"kind":"texture","sample_type":"float",)" + src));
    CHECK(conflict(R"({"group":0,"binding":0,"kind":"uniform",)" + src, R"({"group":0,"binding":0,"kind":"sampler",)" + src));
    // buffers merge to the larger min binding size
    {
        ReflectionRegistry g;
        CHECK(g.load(sidecar(1, R"("0x1":{"stage":"vertex","bindings":[{"group":0,"binding":0,"kind":"uniform","min_binding_size_bytes":16,)" + src +
                                    R"(]},"0x2":{"stage":"fragment","bindings":[{"group":0,"binding":0,"kind":"uniform","min_binding_size_bytes":64,)" + src + "]}")).empty());
        LayoutBuildResult x = build_bind_group_layout_descriptors({g.by_hash(1), g.by_hash(2)});
        CHECK(x.ok && x.groups[0][0].min_binding_size == 64);
    }
    // WebGPU's baseline: bind group index 4 is rejected (only with entries in it)
    ReflectionRegistry hi;
    CHECK(hi.load(sidecar(1, R"("0x1":{"stage":"compute","bindings":[{"group":4,"binding":0,"kind":"sampler",)" + src + "]}")).empty());
    LayoutBuildResult h = build_bind_group_layout_descriptors({hi.by_hash(1)});
    CHECK(!h.ok && !h.conflict);
}

void test_vertex_layout() {
    ReflectionRegistry reg;
    CHECK(reg.load(sidecar(3, std::string(kVs) + "," + kFs)).empty());
    const StageRecord* vs = reg.by_hash(0xa1);
    std::vector<VertexBufferLayoutDesc> vb;
    std::string err;
    CHECK(build_vertex_buffer_layouts(*vs, 12, &vb, &err));
    CHECK(vb.size() == 1 && vb[0].array_stride == 12 && vb[0].attributes.size() == 2);
    CHECK(vb[0].attributes[0].location == 0 && vb[0].attributes[0].offset == 0 && vb[0].attributes[0].format == 29);  // float32x2
    CHECK(vb[0].attributes[1].location == 2 && vb[0].attributes[1].offset == 8 && vb[0].attributes[1].format == 9);   // unorm8x4
    // a larger stride is taken as is; a stride below the attribute extent is rejected
    CHECK(build_vertex_buffer_layouts(*vs, 32, &vb, &err) && vb[0].array_stride == 32);
    CHECK(!build_vertex_buffer_layouts(*vs, 8, &vb, &err) && !err.empty());
    // no vertex inputs: nothing to build
    CHECK(build_vertex_buffer_layouts(*reg.by_hash(0xa2), 12, &vb, &err) && vb.empty());
    // inputs without a format cannot be laid out
    ReflectionRegistry nf;
    CHECK(nf.load(sidecar(1, R"("0x1":{"stage":"vertex","vertex_inputs":[{"location":0}],"bindings":[]})")).empty());
    CHECK(!build_vertex_buffer_layouts(*nf.by_hash(1), 16, &vb, &err));
    ReflectionRegistry bf;
    CHECK(bf.load(sidecar(1, R"("0x1":{"stage":"vertex","vertex_inputs":[{"location":0,"fetch_format":"float32x9"}],"bindings":[]})")).empty());
    CHECK(!build_vertex_buffer_layouts(*bf.by_hash(1), 16, &vb, &err));
    // `format` wins over `fetch_format`; duplicate locations are rejected
    ReflectionRegistry fm;
    CHECK(fm.load(sidecar(1, R"("0x1":{"stage":"vertex","vertex_inputs":[{"location":1,"format":"uint32","fetch_format":"float32x4"},{"location":1,"format":"uint32"}],"bindings":[]})")).empty());
    CHECK(fm.by_hash(1)->vertex_inputs[0].format == "uint32");
    CHECK(!build_vertex_buffer_layouts(*fm.by_hash(1), 16, &vb, &err));
    // running offsets follow location order across mixed sizes
    ReflectionRegistry mx;
    CHECK(mx.load(sidecar(1, R"("0x1":{"stage":"vertex","vertex_inputs":[
        {"location":3,"fetch_format":"float32x3"},{"location":1,"fetch_format":"uint16x2"},{"location":2,"fetch_format":"float16x4"}],"bindings":[]})")).empty());
    CHECK(build_vertex_buffer_layouts(*mx.by_hash(1), 24, &vb, &err));
    CHECK(vb[0].attributes.size() == 3 && vb[0].attributes[0].location == 1 && vb[0].attributes[0].offset == 0 &&
          vb[0].attributes[1].location == 2 && vb[0].attributes[1].offset == 4 && vb[0].attributes[2].location == 3 && vb[0].attributes[2].offset == 12);
}

const char* kManifest = R"({
  "schema_version": 1,
  "register_convention": {"source": "dxc-spirv", "shifts": {"b": 0, "t": 0}},
  "bind_group_layouts": {
    "0x00000000000000aa": {"entries": [
      {"binding": 1, "visibility": 4, "buffer": {"type": "storage", "hasDynamicOffset": true, "minBindingSize": 16}},
      {"binding": 0, "visibility": 5, "texture": {"sampleType": "depth", "viewDimension": "cube"}}]},
    "0x00000000000000ab": {"entries": []}
  },
  "pipeline_layouts": {"0x00000000000000bb": {"bindGroupLayouts": ["0xaa", "0xab"]}},
  "shaders": {
    "0x00000000000000c1": {"stage": "vertex", "entry_point": "vs_main", "name": "fade",
      "pipeline_layout": "0xbb", "bind_group_layouts": ["0xaa", "0xab"],
      "vertex_inputs": [{"location": 1, "type": "vec2<f32>", "format": "float32x2"}, {"location": 0, "type": "vec4<f32>", "format": "float32x4"}],
      "bindings": [{"group": 0, "binding": 1, "resource": "buffer", "source": {"user_sgpr": 8, "table_byte_offset": 0}}]},
    "0x00000000000000c2": {"stage": "fragment", "entry_point": "fs_main", "bind_group_layouts": ["0xaa"],
      "fragment_outputs": [{"location": 0, "type": "vec4<f32>"}],
      "requires_lowering": [{"kind": "x", "group": null, "binding": null, "detail": "d"}],
      "limit_violations": [{"limit": "maxSamplers", "stage": "fragment", "count": 17, "max": 16}]},
    "0x00000000000000c3": {"stage": "compute", "entry_point": "cs_main", "workgroup_size": [8, 8, 1], "layout_complete": false}
  },
  "pipelines": {"x": {"stages": {"vertex": "0xc1", "fragment": "0xc2"}, "pipeline_layout": "0xbb", "bind_group_layouts": ["0xaa"]}}
})";

void test_manifest() {
    ReflectionRegistry reg;
    CHECK(reg.load(kManifest).empty());
    CHECK(reg.shader_count() == 3 && reg.bind_group_layout_count() == 2 && reg.pipeline_layout_count() == 1);
    const StageRecord* vs = reg.by_hash(0xc1);
    CHECK(vs && vs->entry_point == "vs_main" && vs->name == "fade" && vs->from_sidecar == false);
    // manifest shaders are indexed by entry point, "<stage>:<entry>" and name
    CHECK(reg.lookup_name("vertex:vs_main") == vs);
    CHECK(reg.lookup_name("fade") == vs);
    CHECK(reg.lookup_name("vs_main") == vs);
    CHECK(reg.lookup_name("fragment:fs_main") == reg.by_hash(0xc2));
    CHECK(reg.lookup_name("compute:cs_main") == reg.by_hash(0xc3));
    CHECK(reg.lookup_name("vertex:nope") == nullptr);
    CHECK(vs->vertex_inputs.size() == 2 && vs->vertex_inputs[0].location == 0 && vs->vertex_inputs[0].format == "float32x4");
    CHECK(!reg.by_hash(0xc3)->layout_complete);
    // layouts: entries sorted by binding, visibility kept, per group in the order the shader lists them
    CHECK(vs->group_layouts.size() == 2 && vs->group_layouts[0].size() == 2 && vs->group_layouts[1].empty());
    const auto& g0 = vs->group_layouts[0];
    CHECK(g0[0].binding == 0 && g0[0].kind == K::Texture && g0[0].sample_type == 2 && g0[0].view_dim == 4 && g0[0].visibility == 5);
    CHECK(g0[1].binding == 1 && g0[1].kind == K::Buffer && g0[1].buffer_type == 1 && g0[1].has_dynamic_offset && g0[1].min_binding_size == 16);
    // the layout tables for ManifestReference layouts
    const auto* t = reg.manifest_bind_group_layout(0xaa);
    CHECK(t && t->size() == 2 && (*t)[0].binding == 0);
    CHECK(reg.manifest_bind_group_layout(0xab) && reg.manifest_bind_group_layout(0xab)->empty());
    CHECK(reg.manifest_bind_group_layout(0xac) == nullptr);
    const auto* pl = reg.manifest_pipeline_layout(0xbb);
    CHECK(pl && pl->size() == 2 && (*pl)[0] == "0x00000000000000aa" && (*pl)[1] == "0x00000000000000ab");
    CHECK(reg.manifest_pipeline_layout(0xaa) == nullptr);
    // vs + fs merge (visibility 1|2 on the shared buffer would conflict: the fragment one is a texture-free group)
    LayoutBuildResult m = build_bind_group_layout_descriptors({vs, reg.by_hash(0xc2)});
    CHECK(m.ok && m.groups.size() == 1 && m.groups[0].size() == 2);
    // the sidecar a later document brings replaces nothing of the manifest's other shaders
    CHECK(reg.load(sidecar(1, R"("0xc1":{"stage":"vertex","entry_point":"alt","bindings":[]})")).empty());
    CHECK(reg.shader_count() == 3 && reg.by_hash(0xc1)->entry_point == "alt" && reg.by_hash(0xc1)->from_sidecar);
    CHECK(reg.lookup_name("vertex:vs_main") != nullptr);  // the old name index is not purged (as in the loader)
    CHECK(reg.stats().manifests_loaded == 2 && reg.stats().sidecars_loaded == 1);
}

void test_manifest_validation() {
    auto rejects = [](const std::string& doc) {
        ReflectionRegistry r;
        return !r.load(doc).empty() && r.shader_count() == 0 && r.bind_group_layout_count() == 0;
    };
    CHECK(rejects("[]"));
    CHECK(rejects(R"({"shaders":{}})"));                       // schema_version missing
    CHECK(rejects(R"({"schema_version":2,"shaders":{}})"));
    CHECK(rejects(R"({"schema_version":1})"));                  // shaders missing
    CHECK(rejects(R"({"schema_version":1,"shaders":[]})"));
    ReflectionRegistry ok;
    CHECK(ok.load(R"({"schema_version":1,"shaders":{}})").empty());
    const std::string bgl = R"("bind_group_layouts":{"0xaa":{"entries":[)";
    auto doc = [&](const std::string& entries, const std::string& rest = "") {
        return R"({"schema_version":1,)" + bgl + entries + R"(]}},)" + rest + R"("shaders":{}})";
    };
    CHECK(!rejects(doc(R"({"binding":0,"visibility":1,"buffer":{}})")));  // accepted: type defaults to uniform
    CHECK(rejects(doc(R"({"binding":0,"visibility":1})")));                       // no resource member
    CHECK(rejects(doc(R"({"binding":0,"visibility":1,"buffer":{},"sampler":{}})")));  // two members
    CHECK(rejects(doc(R"({"binding":0,"visibility":8,"buffer":{}})")));          // visibility outside 1..7
    CHECK(rejects(doc(R"({"binding":0,"visibility":0,"buffer":{}})")));
    CHECK(rejects(doc(R"({"binding":0,"buffer":{}})")));                         // visibility required without a fallback
    CHECK(rejects(doc(R"({"binding":0,"visibility":1,"buffer":{"type":"weird"}})")));
    CHECK(rejects(doc(R"({"binding":0,"visibility":1,"buffer":{"hasDynamicOffset":1}})")));
    CHECK(rejects(doc(R"({"binding":0,"visibility":1,"buffer":{"minBindingSize":-4}})")));
    CHECK(rejects(doc(R"({"binding":0,"visibility":1,"texture":{"multisampled":true,"viewDimension":"3d"}})")));
    CHECK(rejects(doc(R"({"binding":0,"visibility":1,"storageTexture":{"format":""}})")));
    CHECK(rejects(doc(R"({"binding":0,"visibility":1,"storageTexture":{"format":"r32float","viewDimension":"cube"}})")));
    CHECK(rejects(doc(R"({"binding":0,"visibility":1,"sampler":{}},{"binding":0,"visibility":1,"sampler":{}})")));  // duplicate binding
    CHECK(!rejects(doc(R"({"binding":0,"visibility":2,"externalTexture":{}})")));
    CHECK(!rejects(doc(R"({"binding":0,"visibility":2,"storage_texture":{"format":"r32float","view_dimension":"3d","access":"read-write"}})")));
    // references must resolve
    CHECK(rejects(doc("", R"("pipeline_layouts":{"0xbb":{"bindGroupLayouts":["0xcc"]}},)")));
    CHECK(rejects(doc("", R"("pipeline_layouts":{"0xbb":{"bindGroupLayouts":["0xaa"]},"0x00000000000000BB":{"bindGroupLayouts":[]}},)")));
    CHECK(rejects(R"({"schema_version":1,"shaders":{"0x1":{"stage":"vertex","entry_point":"v","bind_group_layouts":["0x5"]}}})"));
    CHECK(rejects(R"({"schema_version":1,"shaders":{"0x1":{"stage":"vertex","entry_point":""}}})"));
    CHECK(rejects(R"({"schema_version":1,"shaders":{"0x1":{"stage":"pixel","entry_point":"v"}}})"));  // "pixel" is sidecar-only
    CHECK(rejects(R"({"schema_version":1,"shaders":{"0x1":{"stage":"compute","entry_point":"c","workgroup_size":[1,0,1]}}})"));
    CHECK(rejects(R"({"schema_version":1,"shaders":{"0x1":{"stage":"compute","entry_point":"c","bindings":[{"group":0,"binding":0,"resource":"foo"}]}}})"));
    CHECK(rejects(R"({"schema_version":1,"shaders":{"0x1":{"stage":"compute","entry_point":"c","pipeline_layout":"0x9"}},"pipeline_layouts":{}})"));
    CHECK(!rejects(R"({"schema_version":1,"shaders":{"0x1":{"stage":"compute","entry_point":"c","pipeline_layout":"0x9"}}})"));  // no table: not checked
    CHECK(rejects(R"({"schema_version":1,"shaders":{},"pipelines":{"p":{"stages":{"vertex":"0x1"}}}})"));  // unknown shader
    // more than 64 bind group layouts in a shader is over the reflection limit
    std::string many = R"({"schema_version":1,"bind_group_layouts":{"0xaa":{"entries":[]}},"shaders":{"0x1":{"stage":"compute","entry_point":"c","bind_group_layouts":[)";
    for (int i = 0; i < 65; ++i) many += std::string(i ? "," : "") + "\"0xaa\"";
    many += "]}}}";
    CHECK(rejects(many));
    // layouts beyond group 3 are fine while they hold no entries
    ReflectionRegistry six;
    CHECK(six.load(R"({"schema_version":1,"bind_group_layouts":{"0xaa":{"entries":[{"binding":0,"visibility":4,"sampler":{}}]},"0xab":{"entries":[]}},
        "shaders":{"0x1":{"stage":"compute","entry_point":"c","bind_group_layouts":["0xaa","0xab","0xab","0xab","0xab","0xab"]}}})").empty());
    LayoutBuildResult r = build_bind_group_layout_descriptors({six.by_hash(1)});
    CHECK(r.ok && r.groups.size() == 1);
    ReflectionRegistry seven;
    CHECK(seven.load(R"({"schema_version":1,"bind_group_layouts":{"0xaa":{"entries":[{"binding":0,"visibility":4,"sampler":{}}]},"0xab":{"entries":[]}},
        "shaders":{"0x1":{"stage":"compute","entry_point":"c","bind_group_layouts":["0xab","0xab","0xab","0xab","0xaa"]}}})").empty());
    CHECK(!build_bind_group_layout_descriptors({seven.by_hash(1)}).ok);
}

// WGPK v2 pack (24-byte entries, 64-bit offsets) next to its sidecar: the entry table resolves the
// shader text and the sidecar record is keyed by the same hash.
void write_pack(const std::string& path, uint32_t version, const std::vector<std::pair<uint64_t, std::string>>& shaders) {
    const size_t entry_bytes = version == 2 ? 24 : 20;
    std::vector<uint8_t> out(16 + shaders.size() * entry_bytes, 0);
    std::memcpy(out.data(), "WGPK", 4);
    const uint32_t count = static_cast<uint32_t>(shaders.size());
    std::memcpy(out.data() + 4, &version, 4);
    std::memcpy(out.data() + 8, &count, 4);
    uint64_t offset = out.size();
    std::vector<uint8_t> blob;
    for (size_t i = 0; i < shaders.size(); ++i) {
        uint8_t* e = out.data() + 16 + i * entry_bytes;
        const uint64_t hash = shaders[i].first;
        const uint32_t length = static_cast<uint32_t>(shaders[i].second.size());
        std::memcpy(e, &hash, 8);
        e[8] = 2;  // fragment
        if (version == 2) {
            std::memcpy(e + 12, &length, 4);
            std::memcpy(e + 16, &offset, 8);
        } else {
            const uint32_t o32 = static_cast<uint32_t>(offset);
            std::memcpy(e + 12, &o32, 4);
            std::memcpy(e + 16, &length, 4);
        }
        blob.insert(blob.end(), shaders[i].second.begin(), shaders[i].second.end());
        offset += length;
    }
    out.insert(out.end(), blob.begin(), blob.end());
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f) {
        std::fwrite(out.data(), 1, out.size(), f);
        std::fclose(f);
    }
}

void test_pack_and_sidecar_hashes() {
    for (uint32_t version : {1u, 2u}) {
        const std::string path = "test_native_exec_reflection_v" + std::to_string(version) + ".wgsl.pack";
        const uint64_t hash = 0x1122334455667788ull;
        const std::string wgsl = "@fragment fn frag_entry() -> @location(0) vec4f { return vec4f(1.0); }";
        write_pack(path, version, {{hash, wgsl}});
        {
            WgslPack pack;
            std::string err;
            CHECK(pack.load(path, &err));
            CHECK(pack.has(hash) && pack.get(hash) && *pack.get(hash) == wgsl);
        }  // closed before the file is removed

        // the sidecar sits at `<pack>.reflect.json` and is keyed by the pack hash (any hex spelling)
        ReflectionRegistry reg;
        CHECK(reg.load(sidecar(3, R"("0X1122334455667788":{"stage":"fragment","entry_point":"frag_entry","bindings":[]})")).empty());
        const StageRecord* s = reg.lookup_hash(hash);
        CHECK(s && s->entry_point == "frag_entry" && s->hash_hex == format_hash(hash));
        // the worker keys inline shaders by the FNV-1a 64 of their text
        CHECK(fnv1a64("") == 0xcbf29ce484222325ull);
        CHECK(fnv1a64("a") == 0xaf63dc4c8601ec8cull);
        std::remove(path.c_str());
    }
    CHECK(format_hash(0xabc) == "0x0000000000000abc");
    CHECK(parse_hash_string(" 0xABC ") && *parse_hash_string(" 0xABC ") == 0xabc);
    CHECK(!parse_hash_string("") && !parse_hash_string("0x") && !parse_hash_string("12345678901234567") && !parse_hash_string("0xg"));
}

}  // namespace

int main() {
    test_json();
    test_sidecar_stage_records();
    test_sidecar_validation();
    test_sidecar_kind_aliases_and_storage_formats();
    test_layout_merge();
    test_vertex_layout();
    test_manifest();
    test_manifest_validation();
    test_pack_and_sidecar_hashes();
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_native_exec_reflection: ok\n");
    return 0;
}
