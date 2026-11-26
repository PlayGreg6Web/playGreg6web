#include "gpu/stage_reflection.h"

#include "gpu/descriptor_words.h"
#include "gpu/rt_arena.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

namespace playgta6::gpu {

namespace {

// ---------------------------------------------------------------------------------------------
// Minimal JSON reader (objects, arrays, strings, numbers, true/false/null).
// ---------------------------------------------------------------------------------------------

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    const Json* get(std::string_view key) const {
        if (type != Type::Object) {
            return nullptr;
        }
        for (const auto& [k, v] : object) {
            if (k == key) {
                return &v;
            }
        }
        return nullptr;
    }
};

class Reader {
public:
    explicit Reader(std::string_view s) : s_(s) {}

    bool parse(Json* out, std::string* error) {
        if (!value(out, 0)) {
            *error = "malformed JSON near byte " + std::to_string(at_);
            return false;
        }
        skip();
        if (at_ != s_.size()) {
            *error = "trailing data at byte " + std::to_string(at_);
            return false;
        }
        return true;
    }

private:
    void skip() {
        while (at_ < s_.size() && (s_[at_] == ' ' || s_[at_] == '\n' || s_[at_] == '\r' || s_[at_] == '\t')) {
            ++at_;
        }
    }
    bool literal(std::string_view word) {
        if (s_.substr(at_, word.size()) != word) {
            return false;
        }
        at_ += word.size();
        return true;
    }
    bool string(std::string* out) {
        if (at_ >= s_.size() || s_[at_] != '"') {
            return false;
        }
        ++at_;
        while (at_ < s_.size() && s_[at_] != '"') {
            char c = s_[at_++];
            if (c == '\\') {
                if (at_ >= s_.size()) {
                    return false;
                }
                const char e = s_[at_++];
                switch (e) {
                    case 'n': c = '\n'; break;
                    case 't': c = '\t'; break;
                    case 'r': c = '\r'; break;
                    case 'b': c = '\b'; break;
                    case 'f': c = '\f'; break;
                    case 'u':
                        if (at_ + 4 > s_.size()) {
                            return false;
                        }
                        at_ += 4;  // non-ASCII escapes are not needed by this format
                        c = '?';
                        break;
                    default: c = e; break;
                }
            }
            out->push_back(c);
        }
        if (at_ >= s_.size()) {
            return false;
        }
        ++at_;
        return true;
    }
    bool value(Json* out, int depth) {
        if (depth > 32) {
            return false;
        }
        skip();
        if (at_ >= s_.size()) {
            return false;
        }
        const char c = s_[at_];
        if (c == '{') {
            ++at_;
            out->type = Json::Type::Object;
            skip();
            if (at_ < s_.size() && s_[at_] == '}') {
                ++at_;
                return true;
            }
            for (;;) {
                skip();
                std::string key;
                if (!string(&key)) {
                    return false;
                }
                skip();
                if (at_ >= s_.size() || s_[at_] != ':') {
                    return false;
                }
                ++at_;
                Json v;
                if (!value(&v, depth + 1)) {
                    return false;
                }
                out->object.emplace_back(std::move(key), std::move(v));
                skip();
                if (at_ < s_.size() && s_[at_] == ',') {
                    ++at_;
                    continue;
                }
                if (at_ < s_.size() && s_[at_] == '}') {
                    ++at_;
                    return true;
                }
                return false;
            }
        }
        if (c == '[') {
            ++at_;
            out->type = Json::Type::Array;
            skip();
            if (at_ < s_.size() && s_[at_] == ']') {
                ++at_;
                return true;
            }
            for (;;) {
                Json v;
                if (!value(&v, depth + 1)) {
                    return false;
                }
                out->array.push_back(std::move(v));
                skip();
                if (at_ < s_.size() && s_[at_] == ',') {
                    ++at_;
                    continue;
                }
                if (at_ < s_.size() && s_[at_] == ']') {
                    ++at_;
                    return true;
                }
                return false;
            }
        }
        if (c == '"') {
            out->type = Json::Type::String;
            return string(&out->string);
        }
        if (literal("true")) {
            out->type = Json::Type::Bool;
            out->boolean = true;
            return true;
        }
        if (literal("false")) {
            out->type = Json::Type::Bool;
            return true;
        }
        if (literal("null")) {
            return true;
        }
        const size_t start = at_;
        while (at_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[at_])) || s_[at_] == '-' ||
                                   s_[at_] == '+' || s_[at_] == '.' || s_[at_] == 'e' || s_[at_] == 'E')) {
            ++at_;
        }
        if (at_ == start) {
            return false;
        }
        out->type = Json::Type::Number;
        out->number = std::strtod(std::string(s_.substr(start, at_ - start)).c_str(), nullptr);
        return true;
    }

    std::string_view s_;
    size_t at_ = 0;
};

bool get_u32(const Json& o, std::string_view key, uint32_t* out, bool required = true) {
    const Json* v = o.get(key);
    if (v == nullptr) {
        return !required;
    }
    if (v->type != Json::Type::Number || v->number < 0 || v->number > 4294967295.0 ||
        v->number != static_cast<double>(static_cast<uint32_t>(v->number))) {
        return false;
    }
    *out = static_cast<uint32_t>(v->number);
    return true;
}

std::string get_string(const Json& o, std::string_view key) {
    const Json* v = o.get(key);
    return v != nullptr && v->type == Json::Type::String ? v->string : std::string();
}

bool get_bool(const Json& o, std::string_view key) {
    const Json* v = o.get(key);
    return v != nullptr && v->type == Json::Type::Bool && v->boolean;
}

bool vertex_format_from_name(std::string_view name, pso::WgpuVertexFormat* out) {
    using F = pso::WgpuVertexFormat;
    static const std::pair<std::string_view, F> kNames[] = {
        {"uint8", F::Uint8},         {"uint8x2", F::Uint8x2},     {"uint8x4", F::Uint8x4},
        {"sint8", F::Sint8},         {"sint8x2", F::Sint8x2},     {"sint8x4", F::Sint8x4},
        {"unorm8", F::Unorm8},       {"unorm8x2", F::Unorm8x2},   {"unorm8x4", F::Unorm8x4},
        {"snorm8", F::Snorm8},       {"snorm8x2", F::Snorm8x2},   {"snorm8x4", F::Snorm8x4},
        {"uint16", F::Uint16},       {"uint16x2", F::Uint16x2},   {"uint16x4", F::Uint16x4},
        {"sint16", F::Sint16},       {"sint16x2", F::Sint16x2},   {"sint16x4", F::Sint16x4},
        {"unorm16", F::Unorm16},     {"unorm16x2", F::Unorm16x2}, {"unorm16x4", F::Unorm16x4},
        {"snorm16", F::Snorm16},     {"snorm16x2", F::Snorm16x2}, {"snorm16x4", F::Snorm16x4},
        {"float16", F::Float16},     {"float16x2", F::Float16x2}, {"float16x4", F::Float16x4},
        {"float32", F::Float32},     {"float32x2", F::Float32x2}, {"float32x3", F::Float32x3},
        {"float32x4", F::Float32x4}, {"uint32", F::Uint32},       {"uint32x2", F::Uint32x2},
        {"uint32x3", F::Uint32x3},   {"uint32x4", F::Uint32x4},   {"sint32", F::Sint32},
        {"sint32x2", F::Sint32x2},   {"sint32x3", F::Sint32x3},   {"sint32x4", F::Sint32x4},
        {"unorm10-10-10-2", F::Unorm10_10_10_2},
    };
    for (const auto& [n, f] : kNames) {
        if (n == name) {
            *out = f;
            return true;
        }
    }
    return false;
}

bool view_dimension_from_name(std::string_view name, GpuTextureViewDimension* out) {
    using D = GpuTextureViewDimension;
    static const std::pair<std::string_view, D> kNames[] = {
        {"1d", D::Dim1D}, {"2d", D::Dim2D}, {"2d-array", D::Dim2DArray}, {"2d_array", D::Dim2DArray},
        {"cube", D::Cube}, {"cube-array", D::CubeArray}, {"cube_array", D::CubeArray}, {"3d", D::Dim3D},
    };
    for (const auto& [n, d] : kNames) {
        if (n == name) {
            *out = d;
            return true;
        }
    }
    return false;
}

// The CB dword of a CB-indexed index or a lane-indexed window base (G62): "cbv" {user_sgpr,
// root_descriptor | table_byte_offset | root_constants}, "byte_offset", optional "bit_offset" and
// "mask" in `body`. False when malformed.
bool parse_cb_dword(const Json& body, ReflectedBinding* r) {
    const Json* cbv = body.get("cbv");
    bool ok = cbv != nullptr && cbv->type == Json::Type::Object && get_u32(*cbv, "user_sgpr", &r->user_sgpr) &&
              get_u32(body, "byte_offset", &r->cb_byte_offset) && get_u32(body, "bit_offset", &r->cb_bit_offset, false) &&
              get_u32(body, "mask", &r->cb_mask, false);
    if (ok) {
        const bool is_root_desc = get_bool(*cbv, "root_descriptor");
        const bool is_root_const = get_bool(*cbv, "root_constants");
        const bool has_table_off = cbv->get("table_byte_offset") != nullptr;
        const int where = (is_root_desc ? 1 : 0) + (is_root_const ? 1 : 0) + (has_table_off ? 1 : 0);
        if (where != 1) {
            return false;
        }
        if (is_root_desc) {
            r->cbv = ReflectedCbv::RootDescriptor;
        } else if (is_root_const) {
            r->cbv = ReflectedCbv::RootConstants;
        } else {
            r->cbv = ReflectedCbv::Table;
            ok = get_u32(*cbv, "table_byte_offset", &r->table_byte_offset);
        }
    }
    return ok && r->cb_bit_offset < 32 && r->cb_mask != 0 && r->cb_byte_offset % 4 == 0;
}

bool parse_stage(const Json& s, uint32_t version, StageReflection* out, std::string* error) {
    const std::string stage = get_string(s, "stage");
    if (stage == "vertex") {
        out->stage = ReflectedStage::Vertex;
    } else if (stage == "fragment" || stage == "pixel") {
        out->stage = ReflectedStage::Fragment;
    } else if (stage == "compute") {
        out->stage = ReflectedStage::Compute;
    } else {
        *error = "unknown stage \"" + stage + "\"";
        return false;
    }
    if (const Json* mesh = s.get("mesh_output"); mesh != nullptr) {
        bridge::MeshOutputLayout& m = out->mesh_output;
        if (mesh->type != Json::Type::Object || !get_u32(*mesh, "max_vertices", &m.max_vertices) ||
            !get_u32(*mesh, "max_primitives", &m.max_primitives) || !get_u32(*mesh, "vertex_words", &m.vertex_words) ||
            !bridge::valid_mesh_output_layout(m)) {
            *error = "mesh_output needs max_vertices and max_primitives in 1..256 and vertex_words 4 + 4 * n (n <= " +
                     std::to_string(bridge::kMaxMeshAttributes) + ")";
            return false;
        }
        out->has_mesh_output = true;
    }
    if (get_bool(s, "rt_arena")) {
        out->rt_arena_mask = 0x3fu;
    }
    if (const Json* srcs = s.get("address_sources"); srcs != nullptr) {
        if (srcs->type != Json::Type::Array) {
            *error = "address_sources must be an array";
            return false;
        }
        for (const Json& e : srcs->array) {
            rt_residency::AddressSource a;
            const Json* kind = e.get("kind");
            if (kind == nullptr || kind->type != Json::Type::String || !get_u32(e, "index", &a.index)) {
                *error = "an address source needs a string kind and a numeric index";
                return false;
            }
            if (kind->string == "root_param") a.kind = rt_residency::SourceKind::RootParam;
            else if (kind->string == "root_constants") a.kind = rt_residency::SourceKind::RootConstants;
            else if (kind->string == "user_data") a.kind = rt_residency::SourceKind::UserData;
            else {
                *error = "unknown address source kind \"" + kind->string + "\"";
                return false;
            }
            get_u32(e, "dword", &a.dword, false);
            if (const Json* x = e.get("extent"); x != nullptr && x->type == Json::Type::Number && x->number >= 1) {
                a.extent = static_cast<uint64_t>(x->number);
            }
            a.bvh_root = get_bool(e, "bvh_root");
            out->address_sources.push_back(a);
        }
    }
    if (const Json* inputs = s.get("vertex_inputs"); inputs != nullptr && inputs->type == Json::Type::Array) {
        for (const Json& in : inputs->array) {
            ReflectedVertexInput v;
            if (!get_u32(in, "location", &v.location) || !get_u32(in, "vb_slot", &v.vb_slot) ||
                !get_u32(in, "byte_offset", &v.byte_offset) || !get_u32(in, "components", &v.components, false)) {
                *error = "vertex input needs location, vb_slot and byte_offset";
                return false;
            }
            const std::string type = get_string(in, "component_type");
            v.component_type = type == "u32" ? ReflectedComponentType::U32
                               : type == "i32" ? ReflectedComponentType::I32
                                               : ReflectedComponentType::F32;
            const std::string fmt = get_string(in, "fetch_format");
            if (!fmt.empty() && !vertex_format_from_name(fmt, &v.fetch_format)) {
                *error = "unknown fetch_format \"" + fmt + "\"";
                return false;
            }
            out->vertex_inputs.push_back(v);
        }
    }
    if (const Json* bindings = s.get("bindings"); bindings != nullptr && bindings->type == Json::Type::Array) {
        for (const Json& b : bindings->array) {
            ReflectedBinding r;
            if (!get_u32(b, "group", &r.group) || !get_u32(b, "binding", &r.binding)) {
                *error = "binding needs group and binding";
                return false;
            }
            if (const Json* rt_src = b.get("source");
                rt_src != nullptr && rt_src->type == Json::Type::Object && get_bool(*rt_src, "rt_arena")) {
                if (r.group != rt_arena::kBindGroup || r.binding < rt_arena::kBindArena0 ||
                    r.binding > rt_arena::kBindStats) {
                    *error = "an rt_arena binding must be group 3, binding 16..21";
                    return false;
                }
                out->rt_arena_mask |= 1u << (r.binding - rt_arena::kBindArena0);
                continue;  // runtime-owned: not joined
            }
            const std::string kind = get_string(b, "kind");
            static const std::pair<std::string_view, ReflectedBindingKind> kKinds[] = {
                {"uniform", ReflectedBindingKind::Uniform},
                {"storage", ReflectedBindingKind::Storage},
                {"read_only_storage", ReflectedBindingKind::ReadOnlyStorage},
                {"texture", ReflectedBindingKind::Texture},
                {"storage_texture", ReflectedBindingKind::StorageTexture},
                {"sampler", ReflectedBindingKind::Sampler},
                {"comparison_sampler", ReflectedBindingKind::ComparisonSampler},
            };
            bool known = false;
            for (const auto& [n, k] : kKinds) {
                if (n == kind) {
                    r.kind = k;
                    known = true;
                }
            }
            if (!known) {
                *error = "unknown binding kind \"" + kind + "\"";
                return false;
            }
            const std::string dim = get_string(b, "view_dimension");
            if (!dim.empty() && !view_dimension_from_name(dim, &r.view_dimension)) {
                *error = "unknown view_dimension \"" + dim + "\"";
                return false;
            }
            const std::string st = get_string(b, "sample_type");
            r.sample_type = st == "unfilterable_float" ? GpuTextureSampleType::UnfilterableFloat
                            : st == "depth"            ? GpuTextureSampleType::Depth
                            : st == "sint"             ? GpuTextureSampleType::Sint
                            : st == "uint"             ? GpuTextureSampleType::Uint
                                                       : GpuTextureSampleType::Float;
            r.multisampled = get_bool(b, "multisampled");
            uint32_t storage_format = 0;
            uint32_t min_size = 0;
            if (!get_u32(b, "storage_format", &storage_format, false) ||
                !get_u32(b, "min_binding_size_bytes", &min_size, false)) {
                *error = "bad storage_format or min_binding_size_bytes";
                return false;
            }
            r.storage_format = static_cast<uint8_t>(storage_format);
            r.min_binding_size = min_size;
            const Json* src = b.get("source");
            if (src == nullptr || src->type != Json::Type::Object) {
                *error = "binding " + std::to_string(r.group) + ":" + std::to_string(r.binding) + " has no source";
                return false;
            }
            const Json* heap = src->get("sampler_heap");
            const bool sampler_heap = heap != nullptr;
            if (heap == nullptr) {
                heap = src->get("resource_heap");
            }
            if (heap != nullptr) {
                r.source = sampler_heap ? ReflectedSource::SamplerHeap : ReflectedSource::ResourceHeap;
                const Json* index = heap->type == Json::Type::Object ? heap->get("index") : nullptr;
                if (index == nullptr || index->type != Json::Type::Object ||
                    !get_u32(*heap, "base_user_sgpr", &r.heap_base_sgpr) || !get_u32(*heap, "stride", &r.heap_stride) ||
                    !get_u32(*index, "user_sgpr", &r.heap_index_sgpr) || !get_u32(*index, "shift", &r.heap_shift) ||
                    r.heap_shift >= 32) {
                    *error = std::string(sampler_heap ? "sampler_heap" : "resource_heap") +
                             " needs base_user_sgpr, stride and index {user_sgpr, shift}";
                    return false;
                }
            } else if (const Json* cb_heap = src->get("cbv_indexed_sampler_heap") != nullptr
                                                 ? src->get("cbv_indexed_sampler_heap")
                                                 : src->get("cbv_indexed_resource_heap");
                       cb_heap != nullptr) {
                const bool sampler_cb = src->get("cbv_indexed_sampler_heap") != nullptr;
                const char* name = sampler_cb ? "cbv_indexed_sampler_heap" : "cbv_indexed_resource_heap";
                r.source = sampler_cb ? ReflectedSource::CbvIndexedSamplerHeap : ReflectedSource::CbvIndexedResourceHeap;
                const bool ok = cb_heap->type == Json::Type::Object && parse_cb_dword(*cb_heap, &r) &&
                                get_u32(*cb_heap, "shift", &r.heap_shift) &&
                                get_u32(*cb_heap, "stride", &r.heap_stride, false);
                if (!ok || r.heap_shift >= 32) {
                    *error = std::string(name) +
                             " needs cbv {user_sgpr, root_descriptor | table_byte_offset | root_constants}, a dword-aligned "
                             "byte_offset, shift < 32 and (optional) bit_offset < 32, non-zero mask, stride";
                    return false;
                }
            } else if (const Json* lane = src->get("lane_indexed_sampler_heap") != nullptr
                                              ? src->get("lane_indexed_sampler_heap")
                                              : src->get("lane_indexed_resource_heap");
                       lane != nullptr) {
                const bool sampler_lane = src->get("lane_indexed_sampler_heap") != nullptr;
                const char* name = sampler_lane ? "lane_indexed_sampler_heap" : "lane_indexed_resource_heap";
                r.source = sampler_lane ? ReflectedSource::LaneIndexedSamplerHeap : ReflectedSource::LaneIndexedResourceHeap;
                bool ok = lane->type == Json::Type::Object && get_u32(*lane, "max_count", &r.lane_max_count) &&
                          get_u32(*lane, "shift", &r.heap_shift) && get_u32(*lane, "stride", &r.heap_stride, false) &&
                          r.lane_max_count != 0 && r.lane_max_count <= kMaxLaneIndexedWindow && r.heap_shift < 32;
                if (const Json* base = ok ? lane->get("window_base") : nullptr; base != nullptr) {
                    r.lane_has_window_base = true;
                    ok = base->type == Json::Type::Object && parse_cb_dword(*base, &r);
                }
                if (!ok) {
                    *error = std::string(name) + " needs max_count in 1.." + std::to_string(kMaxLaneIndexedWindow) +
                             ", shift < 32, (optional) stride and (optional) window_base {cbv {user_sgpr, "
                             "root_descriptor | table_byte_offset | root_constants}, a dword-aligned byte_offset, "
                             "bit_offset < 32, non-zero mask}";
                    return false;
                }
            } else if (const Json* inline_sampler = src->get("inline_sampler"); inline_sampler != nullptr) {
                r.source = ReflectedSource::InlineSampler;
                const Json* words =
                    inline_sampler->type == Json::Type::Object ? inline_sampler->get("sdwords") : nullptr;
                bool ok = words != nullptr && words->type == Json::Type::Array && words->array.size() == 4;
                for (size_t i = 0; ok && i < 4; ++i) {
                    const Json& w = words->array[i];
                    ok = w.type == Json::Type::Number && w.number >= 0 && w.number <= 4294967295.0 &&
                         w.number == static_cast<double>(static_cast<uint32_t>(w.number));
                    if (ok) {
                        r.sampler_words[i] = static_cast<uint32_t>(w.number);
                    }
                }
                if (!ok) {
                    *error = "inline_sampler needs sdwords: four 32-bit unsigned integers";
                    return false;
                }
            } else if (const Json* unresolved = src->get("unresolved"); unresolved != nullptr) {
                const bool marked = (unresolved->type == Json::Type::Bool && unresolved->boolean) ||
                                    (unresolved->type == Json::Type::String && !unresolved->string.empty());
                if (version < kStageReflectionVersionUnresolved || !marked || src->object.size() != 1) {
                    *error = "unresolved source needs \"unresolved\": true or a reason string, and version >= 3";
                    return false;
                }
                if (r.kind == ReflectedBindingKind::Storage || r.kind == ReflectedBindingKind::StorageTexture) {
                    *error = std::string("a ") + (r.kind == ReflectedBindingKind::Storage ? "storage" : "storage_texture") +
                             " binding cannot have an unresolved source";
                    return false;
                }
                r.source = ReflectedSource::Unresolved;
            } else if (src->get("static_sampler") != nullptr) {
                r.source = ReflectedSource::StaticSampler;
                if (!get_u32(*src, "static_sampler", &r.static_sampler)) {
                    *error = "bad static_sampler";
                    return false;
                }
            } else {
                if (!get_u32(*src, "user_sgpr", &r.user_sgpr)) {
                    *error = "source needs user_sgpr";
                    return false;
                }
                if (get_bool(*src, "root_constants")) {
                    r.source = ReflectedSource::RootConstants;
                } else if (get_bool(*src, "root_descriptor")) {
                    r.source = ReflectedSource::RootDescriptor;
                    if (version >= kStageReflectionVersionRootTableOffset &&
                        src->get("table_byte_offset") != nullptr) {
                        if (!get_u32(*src, "table_byte_offset", &r.table_byte_offset)) {
                            *error = "bad root_descriptor table_byte_offset";
                            return false;
                        }
                        r.root_table_offset_given = true;
                    }
                } else if (!get_u32(*src, "table_byte_offset", &r.table_byte_offset)) {
                    *error = "source needs table_byte_offset, root_constants or root_descriptor";
                    return false;
                }
            }
            out->bindings.push_back(r);
        }
    }
    return true;
}

uint32_t parameter_dwords(const d3d::RootParameterDesc& p) {
    switch (p.parameter_type) {
        case d3d::RootParameterType::DescriptorTable: return 1;
        case d3d::RootParameterType::Constants32Bit: return p.num_32bit_values;
        default: return 2;
    }
}

const char* kind_name(ReflectedBindingKind k) {
    switch (k) {
        case ReflectedBindingKind::Uniform: return "uniform";
        case ReflectedBindingKind::Storage: return "storage";
        case ReflectedBindingKind::ReadOnlyStorage: return "read_only_storage";
        case ReflectedBindingKind::Texture: return "texture";
        case ReflectedBindingKind::StorageTexture: return "storage_texture";
        case ReflectedBindingKind::Sampler: return "sampler";
        case ReflectedBindingKind::ComparisonSampler: return "comparison_sampler";
    }
    return "?";
}

std::string heap_scale_str(const ReflectedBinding& b) {
    if (b.heap_stride != 0 && (b.heap_shift >= 32 || b.heap_stride != (1u << b.heap_shift))) {
        return " * " + std::to_string(b.heap_stride);
    }
    return " << " + std::to_string(b.heap_shift);
}

std::string binding_key(const ReflectedBinding& b) {
    std::string k = "@group(" + std::to_string(b.group) + ") @binding(" + std::to_string(b.binding) + ")";
    switch (b.source) {
        case ReflectedSource::Table:
            k += " <- table at s" + std::to_string(b.user_sgpr) + " + " + std::to_string(b.table_byte_offset);
            break;
        case ReflectedSource::RootConstants: k += " <- root constants at s" + std::to_string(b.user_sgpr); break;
        case ReflectedSource::RootDescriptor:
            k += " <- root descriptor at s" + std::to_string(b.user_sgpr);
            if (b.root_table_offset_given) {
                k += " + " + std::to_string(b.table_byte_offset);
            }
            break;
        case ReflectedSource::StaticSampler: k += " <- static sampler " + std::to_string(b.static_sampler); break;
        case ReflectedSource::InlineSampler: {
            char w[64];
            std::snprintf(w, sizeof(w), " <- inline sampler %08x %08x %08x %08x", b.sampler_words[0], b.sampler_words[1],
                          b.sampler_words[2], b.sampler_words[3]);
            k += w;
            break;
        }
        case ReflectedSource::SamplerHeap:
        case ReflectedSource::ResourceHeap:
            k += std::string(" <- ") + (b.source == ReflectedSource::SamplerHeap ? "sampler" : "resource") +
                 " heap[s" + std::to_string(b.heap_index_sgpr) + heap_scale_str(b) + "]";
            break;
        case ReflectedSource::CbvIndexedSamplerHeap:
        case ReflectedSource::CbvIndexedResourceHeap:
            k += std::string(" <- ") + (b.source == ReflectedSource::CbvIndexedSamplerHeap ? "sampler" : "resource") +
                 " heap[cb(" +
                 (b.cbv == ReflectedCbv::RootDescriptor ? "root CBV"
                  : b.cbv == ReflectedCbv::RootConstants ? "root constants"
                                                          : "table + " + std::to_string(b.table_byte_offset)) +
                 " at s" + std::to_string(b.user_sgpr) + ")[" + std::to_string(b.cb_byte_offset) + "]" +
                 heap_scale_str(b) + "]";
            break;
        case ReflectedSource::LaneIndexedSamplerHeap:
        case ReflectedSource::LaneIndexedResourceHeap:
            k += std::string(" <- ") + (b.source == ReflectedSource::LaneIndexedSamplerHeap ? "sampler" : "resource") +
                 " heap[(" +
                 (!b.lane_has_window_base ? std::string("0")
                  : "cb(" +
                        (b.cbv == ReflectedCbv::RootDescriptor ? std::string("root CBV")
                         : b.cbv == ReflectedCbv::RootConstants ? std::string("root constants")
                                                                 : "table + " + std::to_string(b.table_byte_offset)) +
                        " at s" + std::to_string(b.user_sgpr) + ")[" + std::to_string(b.cb_byte_offset) + "]") +
                 " + lane)" + heap_scale_str(b) + "] x " + std::to_string(b.lane_max_count);
            break;
        case ReflectedSource::Unresolved:
            k += " <- unresolved";
            break;
    }
    return k;
}

// The root-constants parameter holding user SGPR `user_sgpr` and the 32-bit value's index in it
// (-1 when no constants parameter covers it).
int root_constant_for_user_sgpr(const d3d::RootSignatureDesc& rs, bool compute, uint32_t user_sgpr, uint32_t* dword,
                                uint32_t base = 0) {
    uint32_t sgpr = base != 0 ? base : compute ? kComputeUserDataBase : kGraphicsUserDataBase;
    for (size_t i = 0; i < rs.parameters.size(); ++i) {
        const uint32_t n = parameter_dwords(rs.parameters[i]);
        if (user_sgpr >= sgpr && user_sgpr < sgpr + n) {
            if (rs.parameters[i].parameter_type != d3d::RootParameterType::Constants32Bit) {
                return -1;
            }
            *dword = user_sgpr - sgpr;
            return static_cast<int>(i);
        }
        sgpr += n;
    }
    return -1;
}

bool kind_accepts(ReflectedBindingKind kind, d3d::DescriptorRangeType cls) {
    switch (kind) {
        case ReflectedBindingKind::Uniform:
            return cls == d3d::DescriptorRangeType::Cbv || cls == d3d::DescriptorRangeType::Srv;
        case ReflectedBindingKind::ReadOnlyStorage:
            return cls == d3d::DescriptorRangeType::Srv || cls == d3d::DescriptorRangeType::Uav ||
                   cls == d3d::DescriptorRangeType::Cbv;
        case ReflectedBindingKind::Storage:
        case ReflectedBindingKind::StorageTexture:
            return cls == d3d::DescriptorRangeType::Uav;
        case ReflectedBindingKind::Texture:
            return cls == d3d::DescriptorRangeType::Srv;
        case ReflectedBindingKind::Sampler:
        case ReflectedBindingKind::ComparisonSampler:
            return cls == d3d::DescriptorRangeType::Sampler;
    }
    return false;
}

// The (class, register, space) of descriptor `slot` of a table (D3D12 numbering: ranges in order,
// appended unless they give an explicit offset). False when no range covers the slot.
bool table_slot_register(const d3d::RootParameterDesc& p, uint32_t slot, d3d::DescriptorRangeType* cls,
                         uint32_t* reg, uint32_t* space) {
    uint64_t next = 0;
    for (const d3d::DescriptorRangeDesc& r : p.ranges) {
        const uint64_t start =
            r.offset_in_descriptors_from_table_start == d3d::kDescriptorRangeOffsetAppend ? next
                                                                                          : r.offset_in_descriptors_from_table_start;
        const bool unbounded = r.num_descriptors == d3d::kUnboundedDescriptorCount;
        if (slot >= start && (unbounded || slot < start + r.num_descriptors)) {
            *cls = r.range_type;
            *reg = r.base_shader_register + static_cast<uint32_t>(slot - start);
            *space = r.register_space;
            return true;
        }
        next = unbounded ? UINT64_MAX / 2 : start + r.num_descriptors;
    }
    return false;
}

bool same_entry(const GpuBindGroupLayoutEntry& a, const GpuBindGroupLayoutEntry& b) {
    return a.type == b.type && a.binding_type == b.binding_type && a.view_dimension == b.view_dimension &&
           a.flags == b.flags && a.storage_format == b.storage_format && a.binding_array_size == b.binding_array_size;
}

bool same_binding(const bridge::LayoutBinding& a, const bridge::LayoutBinding& b) {
    return same_entry(a.entry, b.entry) && a.join.register_class == b.join.register_class &&
           a.join.shader_register == b.join.shader_register && a.join.register_space == b.join.register_space &&
           d3d::same_join(a.join, b.join);
}

pso::WgpuVertexFormat element_class(pso::WgpuVertexFormat f, ReflectedComponentType* cls) {
    using F = pso::WgpuVertexFormat;
    switch (f) {
        case F::Uint8: case F::Uint8x2: case F::Uint8x4: case F::Uint16: case F::Uint16x2: case F::Uint16x4:
        case F::Uint32: case F::Uint32x2: case F::Uint32x3: case F::Uint32x4:
            *cls = ReflectedComponentType::U32;
            break;
        case F::Sint8: case F::Sint8x2: case F::Sint8x4: case F::Sint16: case F::Sint16x2: case F::Sint16x4:
        case F::Sint32: case F::Sint32x2: case F::Sint32x3: case F::Sint32x4:
            *cls = ReflectedComponentType::I32;
            break;
        default:
            *cls = ReflectedComponentType::F32;
            break;
    }
    return f;
}

const char* class_name(ReflectedComponentType c) {
    return c == ReflectedComponentType::U32 ? "u32" : c == ReflectedComponentType::I32 ? "i32" : "f32";
}

// 32-bit component count of a Float32/Uint32/Sint32 format (0 for any other format).
uint32_t dword_components(pso::WgpuVertexFormat f) {
    using F = pso::WgpuVertexFormat;
    switch (f) {
        case F::Float32: case F::Uint32: case F::Sint32: return 1;
        case F::Float32x2: case F::Uint32x2: case F::Sint32x2: return 2;
        case F::Float32x3: case F::Uint32x3: case F::Sint32x3: return 3;
        case F::Float32x4: case F::Uint32x4: case F::Sint32x4: return 4;
        default: return 0;
    }
}

pso::WgpuVertexFormat dword_format(ReflectedComponentType cls, uint32_t n) {
    using F = pso::WgpuVertexFormat;
    static constexpr F kF32[] = {F::Float32, F::Float32x2, F::Float32x3, F::Float32x4};
    static constexpr F kU32[] = {F::Uint32, F::Uint32x2, F::Uint32x3, F::Uint32x4};
    static constexpr F kI32[] = {F::Sint32, F::Sint32x2, F::Sint32x3, F::Sint32x4};
    const F* t = cls == ReflectedComponentType::U32 ? kU32 : cls == ReflectedComponentType::I32 ? kI32 : kF32;
    return t[n - 1];
}

}  // namespace

bool parse_stage_reflection_json(std::string_view json, StageReflectionIndex* out, std::string* error) {
    Json root;
    if (!Reader(json).parse(&root, error)) {
        return false;
    }
    if (root.type != Json::Type::Object || get_string(root, "schema") != "playgta6-stage-reflection") {
        *error = "not a playgta6-stage-reflection document";
        return false;
    }
    uint32_t version = 0;
    if (!get_u32(root, "version", &version) || version < kStageReflectionVersion ||
        version > kStageReflectionMaxVersion) {
        *error = "unsupported reflection version";
        return false;
    }
    const Json* stages = root.get("stages");
    if (stages == nullptr || stages->type != Json::Type::Object) {
        *error = "missing \"stages\"";
        return false;
    }
    for (const auto& [key, s] : stages->object) {
        char* end = nullptr;
        const uint64_t k = std::strtoull(key.c_str(), &end, 16);
        if (key.size() < 3 || end == nullptr || *end != '\0' || k == 0) {
            *error = "bad stage key \"" + key + "\"";
            return false;
        }
        StageReflection r;
        std::string why;
        if (!parse_stage(s, version, &r, &why)) {
            *error = "stage " + key + ": " + why;
            return false;
        }
        (*out)[k] = std::move(r);
    }
    return true;
}

int root_parameter_containing_user_sgpr(const d3d::RootSignatureDesc& rs, bool compute, uint32_t user_sgpr,
                                        uint32_t* dword, uint32_t base) {
    uint32_t sgpr = base != 0 ? base : compute ? kComputeUserDataBase : kGraphicsUserDataBase;
    for (size_t i = 0; i < rs.parameters.size(); ++i) {
        const uint32_t n = parameter_dwords(rs.parameters[i]);
        if (user_sgpr >= sgpr && user_sgpr < sgpr + n) {
            if (dword != nullptr) {
                *dword = user_sgpr - sgpr;
            }
            return static_cast<int>(i);
        }
        sgpr += n;
    }
    return -1;
}

int root_parameter_for_user_sgpr(const d3d::RootSignatureDesc& rs, bool compute, uint32_t user_sgpr, uint32_t base) {
    uint32_t sgpr = base != 0 ? base : compute ? kComputeUserDataBase : kGraphicsUserDataBase;
    for (size_t i = 0; i < rs.parameters.size(); ++i) {
        if (sgpr == user_sgpr) {
            return static_cast<int>(i);
        }
        sgpr += parameter_dwords(rs.parameters[i]);
    }
    return -1;
}

namespace {

// The constant buffer a CB-indexed index or a lane-indexed window base is read from (G62): the
// root parameter at b.user_sgpr, checked against b.cbv. False with the reason when it does not fit.
bool join_cb_dword(const d3d::RootSignatureDesc& rs, bool compute, const ReflectedBinding& b, int* pi_out,
                   d3d::HeapIndexSource* from, uint32_t* cb_table_slot, std::string* why, uint32_t base) {
    if (b.cb_byte_offset % 4 != 0) {
        *why = "byte offset " + std::to_string(b.cb_byte_offset) + " is not dword-aligned";
        return false;
    }
    if (b.cb_bit_offset >= 32 || b.cb_mask == 0) {
        *why = "bit_offset must be < 32 and mask must be non-zero";
        return false;
    }
    const int pi = root_parameter_for_user_sgpr(rs, compute, b.user_sgpr, base);
    if (pi < 0) {
        *why = "no root parameter starts at s" + std::to_string(b.user_sgpr);
        return false;
    }
    const d3d::RootParameterDesc& p = rs.parameters[static_cast<size_t>(pi)];
    const std::string pname = "root parameter " + std::to_string(pi);
    switch (b.cbv) {
        case ReflectedCbv::RootDescriptor:
            if (p.parameter_type != d3d::RootParameterType::Cbv) {
                *why = pname + " is not a root CBV";
                return false;
            }
            *from = d3d::HeapIndexSource::CbRootDescriptor;
            break;
        case ReflectedCbv::RootConstants:
            if (p.parameter_type != d3d::RootParameterType::Constants32Bit || b.cb_byte_offset / 4 >= p.num_32bit_values) {
                *why = pname + " is not root constants covering byte " + std::to_string(b.cb_byte_offset);
                return false;
            }
            *from = d3d::HeapIndexSource::CbRootConstants;
            break;
        case ReflectedCbv::Table: {
            bool sampler_table = false;
            for (const d3d::DescriptorRangeDesc& r : p.ranges) {
                sampler_table = sampler_table || r.range_type == d3d::DescriptorRangeType::Sampler;
            }
            if (p.parameter_type != d3d::RootParameterType::DescriptorTable || sampler_table) {
                *why = pname + " is not a CBV_SRV_UAV descriptor table";
                return false;
            }
            if (b.table_byte_offset % kTableSlotBytes != 0) {
                *why = pname + ": byte offset " + std::to_string(b.table_byte_offset) + " is not a descriptor boundary";
                return false;
            }
            *from = d3d::HeapIndexSource::CbTable;
            *cb_table_slot = b.table_byte_offset / kTableSlotBytes;
            break;
        }
    }
    *pi_out = pi;
    return true;
}

}  // namespace

bool join_pipeline_layout(const d3d::RootSignatureDesc& rs, bool compute, const std::vector<ReflectedStageRef>& stages,
                          bridge::PipelineLayoutDesc* out, std::string* report, JoinStats* stats,
                          const JoinOptions& options) {
    struct JoinedBinding {
        bridge::LayoutBinding lb{};
        bool unresolved = false;
        bool outside_rs = false;
        bool lane_array = false;
        bool root_table = false;
    };
    std::map<std::pair<uint32_t, uint32_t>, JoinedBinding> joined;
    uint64_t lane_indexed = 0;
    bool rt_arena = false;
    std::vector<rt_residency::AddressSource> rt_sources;
    bool ok = true;
    auto fail = [&](const ReflectedBinding& b, const std::string& why) {
        *report += binding_key(b) + ": " + why + "\n";
        ok = false;
    };
    for (const ReflectedStageRef& st : stages) {
        if (st.reflection == nullptr) {
            continue;
        }
        if ((st.reflection->rt_arena_mask & ~0x3fu) != 0) {
            *report += "rt_arena_mask has bits outside group 3 bindings 16..21\n";
            ok = false;
        }
        if (st.reflection->rt_arena_mask != 0) {
            rt_arena = true;
        }
        rt_sources.insert(rt_sources.end(), st.reflection->address_sources.begin(), st.reflection->address_sources.end());
        for (const ReflectedBinding& b : st.reflection->bindings) {
            bridge::LayoutBinding lb;
            lb.entry.binding = b.binding;
            lb.entry.visibility = st.visibility;
            bool is_unresolved = false;
            bool is_outside_rs = false;
            bool is_lane_array = false;
            bool is_root_table = false;
            // Where the descriptor comes from, as a root-signature register.
            d3d::DescriptorRangeType cls = d3d::DescriptorRangeType::Srv;
            uint32_t reg = 0, space = 0;
            if (b.source == ReflectedSource::SamplerHeap || b.source == ReflectedSource::ResourceHeap) {
                if (b.heap_shift >= 32) {
                    fail(b, "heap shift " + std::to_string(b.heap_shift) + " must be < 32");
                    continue;
                }
                // Heap-indexed: the index must be a root constant; the class follows the kind.
                uint32_t dword = 0;
                const int pi = root_constant_for_user_sgpr(rs, compute, b.heap_index_sgpr, &dword, options.user_data_base);
                if (pi < 0) {
                    fail(b, "s" + std::to_string(b.heap_index_sgpr) + " is not inside a root-constants parameter");
                    continue;
                }
                const bool sampler_kind =
                    b.kind == ReflectedBindingKind::Sampler || b.kind == ReflectedBindingKind::ComparisonSampler;
                if ((b.source == ReflectedSource::SamplerHeap) != sampler_kind) {
                    fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot come from the " +
                                (b.source == ReflectedSource::SamplerHeap ? "sampler" : "resource") + " heap");
                    continue;
                }
                cls = b.kind == ReflectedBindingKind::Uniform ? d3d::DescriptorRangeType::Cbv
                      : sampler_kind                          ? d3d::DescriptorRangeType::Sampler
                      : (b.kind == ReflectedBindingKind::Storage || b.kind == ReflectedBindingKind::StorageTexture)
                          ? d3d::DescriptorRangeType::Uav
                          : d3d::DescriptorRangeType::Srv;
                lb.join.heap = sampler_kind ? d3d::HeapAccess::Sampler : d3d::HeapAccess::Resource;
                lb.join.heap_index_parameter = static_cast<uint32_t>(pi);
                lb.join.heap_index_dword = dword;
                lb.join.heap_shift = b.heap_shift;
                if (b.heap_stride != 0 && b.heap_stride != (1u << b.heap_shift)) {
                    lb.join.heap_stride = b.heap_stride;
                }
            } else if (b.source == ReflectedSource::CbvIndexedSamplerHeap ||
                       b.source == ReflectedSource::CbvIndexedResourceHeap) {
                if (b.heap_shift >= 32) {
                    fail(b, "heap shift " + std::to_string(b.heap_shift) + " must be < 32");
                    continue;
                }
                // CB-indexed heap (G62): the CB is the root parameter at user_sgpr; the index is read
                // from it at draw time, the class follows the kind.
                int pi = -1;
                d3d::HeapIndexSource from = d3d::HeapIndexSource::CbRootDescriptor;
                std::string why;
                if (!join_cb_dword(rs, compute, b, &pi, &from, &lb.join.cb_table_slot, &why, options.user_data_base)) {
                    fail(b, why);
                    continue;
                }
                const bool sampler_kind =
                    b.kind == ReflectedBindingKind::Sampler || b.kind == ReflectedBindingKind::ComparisonSampler;
                if ((b.source == ReflectedSource::CbvIndexedSamplerHeap) != sampler_kind) {
                    fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot come from the " +
                                (b.source == ReflectedSource::CbvIndexedSamplerHeap ? "sampler" : "resource") + " heap");
                    continue;
                }
                cls = b.kind == ReflectedBindingKind::Uniform ? d3d::DescriptorRangeType::Cbv
                      : sampler_kind                          ? d3d::DescriptorRangeType::Sampler
                      : (b.kind == ReflectedBindingKind::Storage || b.kind == ReflectedBindingKind::StorageTexture)
                          ? d3d::DescriptorRangeType::Uav
                          : d3d::DescriptorRangeType::Srv;
                lb.join.heap = sampler_kind ? d3d::HeapAccess::Sampler : d3d::HeapAccess::Resource;
                lb.join.heap_index_parameter = static_cast<uint32_t>(pi);
                lb.join.heap_index_source = from;
                lb.join.heap_shift = b.heap_shift;
                if (b.heap_stride != 0 && b.heap_stride != (1u << b.heap_shift)) {
                    lb.join.heap_stride = b.heap_stride;
                }
                lb.join.cb_byte_offset = b.cb_byte_offset;
                lb.join.cb_bit_offset = b.cb_bit_offset;
                lb.join.cb_mask = b.cb_mask;
            } else if (b.source == ReflectedSource::LaneIndexedSamplerHeap ||
                       b.source == ReflectedSource::LaneIndexedResourceHeap) {
                if (b.lane_max_count == 0 || b.lane_max_count > kMaxLaneIndexedWindow || b.heap_shift >= 32) {
                    fail(b, "lane-indexed heap needs max_count in 1.." + std::to_string(kMaxLaneIndexedWindow) +
                                " and shift < 32");
                    continue;
                }
                // Lane-indexed heap window (G62): validated like a CB-indexed source. With a consumer
                // that binds binding arrays (JoinOptions::binding_arrays) it joins as an array of
                // lane_max_count heap descriptors starting at the window base; otherwise it is refused
                // with its own reason.
                const bool sampler_kind =
                    b.kind == ReflectedBindingKind::Sampler || b.kind == ReflectedBindingKind::ComparisonSampler;
                if ((b.source == ReflectedSource::LaneIndexedSamplerHeap) != sampler_kind) {
                    fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot come from the " +
                                (b.source == ReflectedSource::LaneIndexedSamplerHeap ? "sampler" : "resource") +
                                " heap");
                    continue;
                }
                int pi = -1;
                d3d::HeapIndexSource from = d3d::HeapIndexSource::Constant;
                uint32_t slot = 0;
                if (b.lane_has_window_base) {
                    std::string why;
                    if (!join_cb_dword(rs, compute, b, &pi, &from, &slot, &why, options.user_data_base)) {
                        fail(b, "window base: " + why);
                        continue;
                    }
                }
                if (!options.binding_arrays) {
                    ++lane_indexed;
                    fail(b, "lane-indexed heap window of " + std::to_string(b.lane_max_count) +
                                " descriptors needs a binding array; the consumer does not bind binding arrays");
                    continue;
                }
                cls = b.kind == ReflectedBindingKind::Uniform ? d3d::DescriptorRangeType::Cbv
                      : sampler_kind                          ? d3d::DescriptorRangeType::Sampler
                      : (b.kind == ReflectedBindingKind::Storage || b.kind == ReflectedBindingKind::StorageTexture)
                          ? d3d::DescriptorRangeType::Uav
                          : d3d::DescriptorRangeType::Srv;
                lb.join.heap = sampler_kind ? d3d::HeapAccess::Sampler : d3d::HeapAccess::Resource;
                lb.join.heap_index_source = from;
                lb.join.heap_shift = b.heap_shift;
                // S63 emits a non-power-of-two scale (s_mul_i32 * 24) as shift 0 plus that stride;
                // a power-of-two stride is the shift itself.
                if (b.heap_stride != 0 && b.heap_stride != (1u << b.heap_shift)) {
                    lb.join.heap_stride = b.heap_stride;
                }
                lb.join.array_size = b.lane_max_count;
                if (b.lane_has_window_base) {
                    lb.join.heap_index_parameter = static_cast<uint32_t>(pi);
                    lb.join.cb_table_slot = slot;
                    lb.join.cb_byte_offset = b.cb_byte_offset;
                    lb.join.cb_bit_offset = b.cb_bit_offset;
                    lb.join.cb_mask = b.cb_mask;
                }
                lb.entry.binding_array_size = static_cast<uint16_t>(b.lane_max_count);
                is_lane_array = true;
            } else if (b.source == ReflectedSource::InlineSampler) {
                if (b.kind != ReflectedBindingKind::Sampler && b.kind != ReflectedBindingKind::ComparisonSampler) {
                    fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot take an inline sampler");
                    continue;
                }
                d3d::SamplerDesc desc;
                const bridge::DecodeStatus st = bridge::decode_sampler_words(b.sampler_words, &desc);
                if (st != bridge::DecodeStatus::Ok) {
                    fail(b, std::string("the inline sampler words do not decode (") + bridge::decode_status_name(st) +
                                ")");
                    continue;
                }
                cls = d3d::DescriptorRangeType::Sampler;
                reg = space = d3d::kNoIndex;
                lb.join.inline_sampler = true;
                lb.join.sampler = desc;
            } else if (b.source == ReflectedSource::Unresolved) {
                if (b.kind == ReflectedBindingKind::Storage || b.kind == ReflectedBindingKind::StorageTexture) {
                    fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot have an unresolved source");
                    continue;
                }
                // The translator could not trace the descriptor (#482): bind the worker's default
                // for the kind (an unresolved register), so the rest of the stage still joins.
                const bool sampler_kind =
                    b.kind == ReflectedBindingKind::Sampler || b.kind == ReflectedBindingKind::ComparisonSampler;
                cls = b.kind == ReflectedBindingKind::Uniform ? d3d::DescriptorRangeType::Cbv
                      : sampler_kind                          ? d3d::DescriptorRangeType::Sampler
                                                              : d3d::DescriptorRangeType::Srv;
                reg = space = d3d::kNoIndex;
                is_unresolved = true;
            } else if (b.source == ReflectedSource::StaticSampler) {
                cls = d3d::DescriptorRangeType::Sampler;
                if (b.static_sampler < rs.static_samplers.size()) {
                    reg = rs.static_samplers[b.static_sampler].shader_register;
                    space = rs.static_samplers[b.static_sampler].register_space;
                } else {
                    // A sampler the console stage builds from constants in its code: the root
                    // signature has no static sampler for it. Bound as the default sampler (an
                    // unresolved join binds null, i.e. the worker's default), not a join failure.
                    reg = space = d3d::kNoIndex;
                }
            } else {
                uint32_t into = 0;
                const int pi = root_parameter_containing_user_sgpr(rs, compute, b.user_sgpr, &into, options.user_data_base);
                if (pi < 0) {
                    fail(b, "no root parameter covers s" + std::to_string(b.user_sgpr));
                    continue;
                }
                if (into != 0 && (b.source == ReflectedSource::Table || b.source == ReflectedSource::RootDescriptor) &&
                    rs.parameters[static_cast<size_t>(pi)].parameter_type != d3d::RootParameterType::DescriptorTable) {
                    fail(b, "s" + std::to_string(b.user_sgpr) + " is dword " + std::to_string(into) + " of root parameter " +
                                std::to_string(pi) + ", not a table pointer or root descriptor");
                    continue;
                }
                const d3d::RootParameterDesc& p = rs.parameters[static_cast<size_t>(pi)];
                const std::string pname = "root parameter " + std::to_string(pi);
                ReflectedSource source = b.source;
                uint32_t table_byte_offset = b.table_byte_offset;
                if (source == ReflectedSource::RootDescriptor &&
                    p.parameter_type == d3d::RootParameterType::DescriptorTable) {
                    // The stage builds its buffer from the SGPR that holds the table pointer (#482):
                    // join the table's descriptor the sidecar names, else its first one.
                    source = ReflectedSource::Table;
                    table_byte_offset = b.root_table_offset_given ? b.table_byte_offset : 0;
                    is_root_table = true;
                }
                switch (source) {
                    case ReflectedSource::Table: {
                        if (p.parameter_type != d3d::RootParameterType::DescriptorTable) {
                            fail(b, pname + " is not a descriptor table");
                            continue;
                        }
                        if (table_byte_offset % kTableSlotBytes != 0) {
                            fail(b, pname + ": byte offset " + std::to_string(table_byte_offset) +
                                        " is not a descriptor boundary");
                            continue;
                        }
                        const bool read_only_kind = b.kind != ReflectedBindingKind::Storage &&
                                                    b.kind != ReflectedBindingKind::StorageTexture;
                        const bool declared =
                            table_slot_register(p, table_byte_offset / kTableSlotBytes, &cls, &reg, &space);
                        if (!declared || (read_only_kind && !kind_accepts(b.kind, cls)) ||
                            (!read_only_kind && cls != d3d::DescriptorRangeType::Uav)) {
                            // A slot the root signature does not declare, or declares as another class:
                            // console stages read their tables as memory, past and regardless of the
                            // declared ranges. Join the slot itself, with the class the stage reads.
                            // That includes a writable binding over a declared SRV or CBV: the stage
                            // stores through the descriptor (image_store / buffer_store), so the slot
                            // holds a UAV-capable view on the console (#482: shared root signatures
                            // whose space-1 table lists a CBV where the stage's compact layout has a
                            // UAV). It is bound as a storage view; a slot that holds no UAV at draw
                            // time binds null (translate_view), as does a texture created without
                            // storage usage (see the state bridge).
                            const bool sampler_kind = b.kind == ReflectedBindingKind::Sampler ||
                                                      b.kind == ReflectedBindingKind::ComparisonSampler;
                            bool sampler_table = false;
                            for (const d3d::DescriptorRangeDesc& r : p.ranges) {
                                sampler_table = sampler_table || r.range_type == d3d::DescriptorRangeType::Sampler;
                            }
                            if (sampler_kind != sampler_table) {
                                fail(b, pname + " has no descriptor at byte offset " +
                                            std::to_string(table_byte_offset) + " and is a " +
                                            (sampler_table ? "sampler" : "resource") + " table");
                                continue;
                            }
                            cls = b.kind == ReflectedBindingKind::Uniform ? d3d::DescriptorRangeType::Cbv
                                  : sampler_kind                         ? d3d::DescriptorRangeType::Sampler
                                  : (b.kind == ReflectedBindingKind::Storage ||
                                     b.kind == ReflectedBindingKind::StorageTexture)
                                      ? d3d::DescriptorRangeType::Uav
                                      : d3d::DescriptorRangeType::Srv;
                            reg = space = 0;
                            lb.join.table_parameter = static_cast<uint32_t>(pi);
                            lb.join.table_slot = table_byte_offset / kTableSlotBytes;
                            is_outside_rs = true;
                        }
                        break;
                    }
                    case ReflectedSource::RootConstants:
                        if (p.parameter_type != d3d::RootParameterType::Constants32Bit) {
                            fail(b, pname + " is not root constants");
                            continue;
                        }
                        cls = d3d::DescriptorRangeType::Cbv;
                        reg = p.shader_register;
                        space = p.register_space;
                        lb.join.root_constants = true;
                        break;
                    case ReflectedSource::RootDescriptor:
                        if (p.parameter_type == d3d::RootParameterType::Cbv) {
                            cls = d3d::DescriptorRangeType::Cbv;
                        } else if (p.parameter_type == d3d::RootParameterType::Srv) {
                            cls = d3d::DescriptorRangeType::Srv;
                        } else if (p.parameter_type == d3d::RootParameterType::Uav) {
                            cls = d3d::DescriptorRangeType::Uav;
                        } else {
                            fail(b, pname + " is not a root descriptor");
                            continue;
                        }
                        reg = p.shader_register;
                        space = p.register_space;
                        break;
                    case ReflectedSource::StaticSampler:
                    case ReflectedSource::SamplerHeap:
                    case ReflectedSource::ResourceHeap:
                    case ReflectedSource::InlineSampler:
                    case ReflectedSource::CbvIndexedSamplerHeap:
                    case ReflectedSource::CbvIndexedResourceHeap:
                    case ReflectedSource::LaneIndexedSamplerHeap:
                    case ReflectedSource::LaneIndexedResourceHeap:
                    case ReflectedSource::Unresolved:
                        break;
                }
            }
            if (!lb.join.root_constants && !kind_accepts(b.kind, cls)) {
                fail(b, std::string("a ") + kind_name(b.kind) + " binding cannot take a " +
                            (cls == d3d::DescriptorRangeType::Cbv   ? "CBV"
                             : cls == d3d::DescriptorRangeType::Srv ? "SRV"
                             : cls == d3d::DescriptorRangeType::Uav ? "UAV"
                                                                    : "sampler"));
                continue;
            }
            if (lb.join.root_constants && b.kind != ReflectedBindingKind::Uniform) {
                fail(b, "root constants bind as a uniform buffer only");
                continue;
            }
            lb.join.register_class = cls;
            lb.join.shader_register = reg;
            lb.join.register_space = space;
            // The WebGPU entry, from the reflection.
            switch (b.kind) {
                case ReflectedBindingKind::Uniform:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuBufferBindingType::Uniform);
                    lb.entry.min_binding_size = b.min_binding_size;
                    break;
                case ReflectedBindingKind::Storage:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuBufferBindingType::Storage);
                    lb.entry.min_binding_size = b.min_binding_size;
                    break;
                case ReflectedBindingKind::ReadOnlyStorage:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage);
                    lb.entry.min_binding_size = b.min_binding_size;
                    break;
                case ReflectedBindingKind::Texture:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
                    lb.entry.binding_type = static_cast<uint8_t>(b.sample_type);
                    lb.entry.view_dimension = static_cast<uint8_t>(b.view_dimension);
                    if (b.multisampled) {
                        lb.entry.flags |= GpuBindGroupLayoutEntryFlags::Multisampled;
                    }
                    break;
                case ReflectedBindingKind::StorageTexture:
                    if (b.storage_format == 0) {
                        fail(b, "a storage texture needs its format");
                        continue;
                    }
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture);
                    // The translator declares storage textures write-only (texture_storage_*<fmt, write>);
                    // the layout must match, and read-write access is not even allowed for formats
                    // such as rgba8unorm (WebGPU rejects the layout, the pipeline and the frame).
                    lb.entry.binding_type = static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly);
                    lb.entry.view_dimension = static_cast<uint8_t>(b.view_dimension);
                    lb.entry.storage_format = b.storage_format;
                    break;
                case ReflectedBindingKind::Sampler:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Sampler);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuSamplerBindingType::Filtering);
                    break;
                case ReflectedBindingKind::ComparisonSampler:
                    lb.entry.type = static_cast<uint8_t>(GpuBindingLayoutType::Sampler);
                    lb.entry.binding_type = static_cast<uint8_t>(GpuSamplerBindingType::Comparison);
                    break;
            }
            const std::pair<uint32_t, uint32_t> key{b.group, b.binding};
            auto it = joined.find(key);
            if (it == joined.end()) {
                joined.emplace(key, JoinedBinding{lb, is_unresolved, is_outside_rs, is_lane_array, is_root_table});
            } else if (same_binding(it->second.lb, lb)) {
                it->second.lb.entry.visibility |= lb.entry.visibility;
                it->second.lb.entry.min_binding_size =
                    std::max(it->second.lb.entry.min_binding_size, lb.entry.min_binding_size);
                it->second.unresolved = it->second.unresolved && is_unresolved;
                it->second.outside_rs = it->second.outside_rs || is_outside_rs;
                it->second.lane_array = it->second.lane_array || is_lane_array;
                it->second.root_table = it->second.root_table || is_root_table;
            } else if ((it->second.unresolved || is_unresolved) && same_entry(it->second.lb.entry, lb.entry)) {
                // One stage traced this binding's descriptor and another could not (version >= 3
                // "unresolved"): when their WebGPU layout entries agree, keep the resolved source.
                if (it->second.unresolved) {
                    const uint32_t vis = it->second.lb.entry.visibility | lb.entry.visibility;
                    const uint64_t min_sz = std::max(it->second.lb.entry.min_binding_size, lb.entry.min_binding_size);
                    it->second.lb = lb;
                    it->second.lb.entry.visibility = vis;
                    it->second.lb.entry.min_binding_size = min_sz;
                    it->second.unresolved = false;
                    it->second.outside_rs = is_outside_rs;
                    it->second.lane_array = is_lane_array;
                    it->second.root_table = is_root_table;
                } else {
                    it->second.lb.entry.visibility |= lb.entry.visibility;
                    it->second.lb.entry.min_binding_size =
                        std::max(it->second.lb.entry.min_binding_size, lb.entry.min_binding_size);
                }
            } else {
                fail(b, "the stages disagree on this binding's kind or source");
            }
        }
    }
    if (stats != nullptr) {
        stats->lane_indexed_bindings += lane_indexed;
    }
    if (!ok) {
        return false;
    }
    bridge::PipelineLayoutDesc layout;
    uint64_t outside_rs = 0;
    uint64_t lane_arrays = 0;
    uint64_t root_tables = 0;
    uint64_t unresolved = 0;
    for (const auto& [key, jb] : joined) {
        if (layout.groups.size() <= key.first) {
            layout.groups.resize(key.first + 1);
        }
        layout.groups[key.first].bindings.push_back(jb.lb);
        if (jb.outside_rs) {
            ++outside_rs;
        }
        if (jb.lane_array) {
            ++lane_arrays;
        }
        if (jb.root_table) {
            ++root_tables;
        }
        if (jb.unresolved) {
            ++unresolved;
        }
    }
    if (layout.groups.empty()) {
        layout.groups.resize(1);  // stages that bind nothing still get one (empty) group
    }
    if (rt_arena) {
        if (layout.groups.size() > rt_arena::kBindGroup) {
            *report += "rt_arena stage: the runtime owns group 3, the stage joined " +
                       std::to_string(layout.groups.size()) + " groups\n";
            return false;
        }
        layout.rt_arena = true;
        layout.rt_address_sources = std::move(rt_sources);
    }
    *out = std::move(layout);
    if (stats != nullptr) {
        stats->table_slots_outside_rs += outside_rs;
        stats->lane_indexed_arrays += lane_arrays;
        stats->root_descriptor_tables += root_tables;
        stats->unresolved_bindings += unresolved;
    }
    return true;
}

bool join_vertex_inputs(const StageReflection& vs, const pso::D3D12InputElementDesc* elements, uint32_t count,
                        std::vector<pso::VertexInputFetch>* out, std::string* report, JoinStats* stats) {
    struct Element {
        uint32_t slot, offset, size;
        pso::WgpuVertexFormat format;
        const char* name;
        uint32_t index;
    };
    std::vector<Element> placed;
    uint32_t slot_end[pso::kD3D12IaVertexInputSlotCount] = {};
    for (uint32_t i = 0; i < count && elements != nullptr; ++i) {
        const pso::D3D12InputElementDesc& e = elements[i];
        const pso::VertexFormatMapping vf = pso::map_vertex_format(e.Format);
        if (!vf.supported() || e.InputSlot >= pso::kD3D12IaVertexInputSlotCount) {
            continue;
        }
        uint32_t offset = e.AlignedByteOffset;
        if (offset == pso::kD3D12AppendAlignedElement) {
            const uint32_t a = std::min(4u, vf.byte_size);
            offset = (slot_end[e.InputSlot] + a - 1) / a * a;
        }
        slot_end[e.InputSlot] = std::max(slot_end[e.InputSlot], offset + vf.byte_size);
        placed.push_back(Element{e.InputSlot, offset, vf.byte_size, vf.format, e.SemanticName, e.SemanticIndex});
    }
    bool ok = true;
    uint64_t defaulted = 0;
    std::vector<pso::VertexInputFetch> fetches;
    for (const ReflectedVertexInput& in : vs.vertex_inputs) {
        const std::string key = "@location(" + std::to_string(in.location) + ") <- slot " + std::to_string(in.vb_slot) +
                                " + " + std::to_string(in.byte_offset);
        const Element* hit = nullptr;
        for (const Element& el : placed) {
            if (el.slot == in.vb_slot && in.byte_offset >= el.offset && in.byte_offset < el.offset + el.size) {
                hit = &el;
                break;
            }
        }
        if (hit == nullptr) {
            // D3D12 reads (0, 0, 0, 1) for an attribute the layout does not provide.
            pso::VertexInputFetch d;
            d.location = in.location;
            d.constant_default = true;
            d.offset = in.component_type == ReflectedComponentType::F32 ? 0u : pso::kDefaultVertexIntOffset;
            d.format = dword_format(in.component_type, std::min(std::max(in.components, 1u), 4u));
            fetches.push_back(d);
            ++defaulted;
            continue;
        }
        pso::VertexInputFetch f;
        f.slot = in.vb_slot;
        f.offset = in.byte_offset;
        f.location = in.location;
        if (in.byte_offset == hit->offset && in.fetch_format == pso::WgpuVertexFormat::Undefined) {
            f.format = hit->format;
            // A dword fetch wider than its element (#691: one 4-dword load of TEXCOORD float2 @4 plus
            // POSITION float2 @12) spans the following 32-bit elements of the same class; widen the
            // attribute over them, or the extra components read WebGPU's (0, 1) defaults.
            const uint32_t have = dword_components(hit->format);
            if (have != 0 && in.components > have && in.components <= 4) {
                ReflectedComponentType hit_cls;
                element_class(hit->format, &hit_cls);
                uint32_t covered = have;
                bool extended = true;
                while (covered < in.components && extended) {
                    extended = false;
                    for (const Element& el : placed) {
                        ReflectedComponentType el_cls;
                        element_class(el.format, &el_cls);
                        const uint32_t n = dword_components(el.format);
                        if (el.slot == hit->slot && el.offset == hit->offset + covered * 4u && n != 0 &&
                            el_cls == hit_cls) {
                            covered += n;
                            extended = true;
                            break;
                        }
                    }
                }
                if (covered >= in.components) {
                    f.format = dword_format(hit_cls, in.components);
                }
            }
        } else if (in.fetch_format != pso::WgpuVertexFormat::Undefined) {
            f.format = in.fetch_format;  // the stage's own fetch (inside an element, or a reinterpretation)
        } else {
            *report += key + ": reads inside element " + std::string(hit->name ? hit->name : "?") +
                       std::to_string(hit->index) + " and has no fetch_format\n";
            ok = false;
            continue;
        }
        ReflectedComponentType cls;
        element_class(f.format, &cls);
        if (cls != in.component_type && dword_components(f.format) != 0) {
            // Same 32-bit bits, other class (f32 vs u32/i32): the vertex format is ours, so reinterpret it.
            f.format = dword_format(in.component_type, dword_components(f.format));
            cls = in.component_type;
        }
        if (cls != in.component_type) {
            *report += key + ": the stage expects " + class_name(in.component_type) + ", the attribute gives " +
                       class_name(cls) + "\n";
            ok = false;
            continue;
        }
        fetches.push_back(f);
    }
    if (!ok) {
        return false;
    }
    *out = std::move(fetches);
    if (stats != nullptr) {
        stats->vertex_inputs_defaulted += defaulted;
    }
    return true;
}

}  // namespace playgta6::gpu
