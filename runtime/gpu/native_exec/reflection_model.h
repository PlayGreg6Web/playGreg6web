// Native executor, GPU-independent part of the reflection loader: a port of the parts of
// web/workers/reflection_loader.js (`ReflectionRegistry`, `parseStageReflectionSidecar`,
// `parseReflectionManifest`, `buildBindGroupLayoutDescriptors`, `buildVertexBufferLayouts`) that
// decide what the worker's `_resolveShaderReflection` / `_buildReflectedPipelineLayout` hand to
// WebGPU: the stage record of a shader (entry point, per-group layout entries, vertex inputs), the
// merged bind group layout descriptors of a pipeline and the reflected vertex buffer layout.
// Validation follows the loader: a document it rejects is rejected here, nothing is repaired.
// Binding *sources* (the console ABI side: user SGPRs, heaps, static samplers) never reach WebGPU;
// they are validated exactly like the loader does and then dropped. The producer-side join to the
// root signature, `diagnoseStageBindings` and `synthesizeMissingStaticSamplers` are not ported.
// No WebGPU types: tests/test_native_exec_reflection.cpp runs on every host build.
// Design: docs/NATIVE_EXECUTOR_DESIGN.md.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace playgta6::gpu::native_exec::reflect {

// ---- JSON (the subset of JSON.parse's behaviour the loader depends on) ----

struct Json {
    enum class T : uint8_t { Null, Bool, Num, Str, Arr, Obj };
    T t = T::Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    bool is_obj() const { return t == T::Obj; }
    bool is_arr() const { return t == T::Arr; }
    bool is_str() const { return t == T::Str; }
    bool is_num() const { return t == T::Num; }
    bool is_bool() const { return t == T::Bool; }
    bool is_null() const { return t == T::Null; }
    // `obj.key` : nullptr is JS `undefined`; a present null is a Json of type Null.
    const Json* get(std::string_view k) const {
        if (t != T::Obj) return nullptr;
        for (const auto& kv : o)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct ReflectError : std::runtime_error {
    explicit ReflectError(const std::string& m) : std::runtime_error(m) {}
};

[[noreturn]] inline void fail(const std::string& msg, const std::string& path = std::string()) {
    throw ReflectError(path.empty() ? msg : (path + ": " + msg));
}

namespace detail {

class JsonParser {
public:
    explicit JsonParser(std::string_view s) : s_(s) {}
    void parse(Json* out) {
        skip_ws();
        value(out, 0);
        skip_ws();
        if (i_ != s_.size()) err("unexpected trailing characters");
    }

private:
    std::string_view s_;
    size_t i_ = 0;

    [[noreturn]] void err(const char* m) const { fail(std::string("malformed JSON: ") + m + " at offset " + std::to_string(i_)); }
    void skip_ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
    }
    bool lit(std::string_view w) {
        if (s_.substr(i_, w.size()) == w) {
            i_ += w.size();
            return true;
        }
        return false;
    }
    static void put_utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) {
            o += static_cast<char>(cp);
        } else if (cp < 0x800) {
            o += static_cast<char>(0xC0 | (cp >> 6));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            o += static_cast<char>(0xE0 | (cp >> 12));
            o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            o += static_cast<char>(0xF0 | (cp >> 18));
            o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    uint32_t hex4() {
        if (i_ + 4 > s_.size()) err("bad unicode escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else err("bad unicode escape");
        }
        return v;
    }
    void string(std::string* out) {
        ++i_;  // opening quote
        for (;;) {
            if (i_ >= s_.size()) err("unterminated string");
            const unsigned char c = static_cast<unsigned char>(s_[i_++]);
            if (c == '"') return;
            if (c < 0x20) err("control character in string");
            if (c != '\\') {
                out->push_back(static_cast<char>(c));
                continue;
            }
            if (i_ >= s_.size()) err("unterminated escape");
            const char e = s_[i_++];
            switch (e) {
                case '"': out->push_back('"'); break;
                case '\\': out->push_back('\\'); break;
                case '/': out->push_back('/'); break;
                case 'b': out->push_back('\b'); break;
                case 'f': out->push_back('\f'); break;
                case 'n': out->push_back('\n'); break;
                case 'r': out->push_back('\r'); break;
                case 't': out->push_back('\t'); break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                        const size_t save = i_;
                        i_ += 2;
                        const uint32_t lo = hex4();
                        if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else i_ = save;
                    }
                    put_utf8(*out, cp);
                    break;
                }
                default: err("bad escape");
            }
        }
    }
    void number(Json* out) {
        const size_t start = i_;
        if (i_ < s_.size() && s_[i_] == '-') ++i_;
        if (i_ >= s_.size()) err("bad number");
        if (s_[i_] == '0') {
            ++i_;
        } else if (s_[i_] >= '1' && s_[i_] <= '9') {
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        } else {
            err("bad number");
        }
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') err("bad number");
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') err("bad number");
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        const std::string tok(s_.substr(start, i_ - start));
        out->t = Json::T::Num;
        out->n = std::strtod(tok.c_str(), nullptr);
    }
    // A JS object lists canonical array-index keys (decimal, no leading zero, < 2^32 - 1) first in
    // ascending order, then the other keys in insertion order.
    static bool array_index(const std::string& k, uint64_t* v) {
        if (k.empty() || k.size() > 10) return false;
        if (k.size() > 1 && k[0] == '0') return false;
        uint64_t x = 0;
        for (char c : k) {
            if (c < '0' || c > '9') return false;
            x = x * 10 + static_cast<uint64_t>(c - '0');
        }
        if (x >= 4294967295ull) return false;
        *v = x;
        return true;
    }
    static void order_keys(Json* obj) {
        bool any = false;
        uint64_t tmp;
        for (const auto& kv : obj->o)
            if (array_index(kv.first, &tmp)) {
                any = true;
                break;
            }
        if (!any) return;
        std::stable_sort(obj->o.begin(), obj->o.end(), [](const auto& x, const auto& y) {
            uint64_t a, b;
            const bool ia = array_index(x.first, &a), ib = array_index(y.first, &b);
            if (ia && ib) return a < b;
            return ia && !ib;
        });
    }
    void value(Json* out, int depth) {
        if (depth > 256) err("nesting too deep");
        if (i_ >= s_.size()) err("unexpected end");
        const char c = s_[i_];
        if (c == '{') {
            ++i_;
            out->t = Json::T::Obj;
            skip_ws();
            if (i_ < s_.size() && s_[i_] == '}') {
                ++i_;
                return;
            }
            for (;;) {
                skip_ws();
                if (i_ >= s_.size() || s_[i_] != '"') err("expected a key");
                std::string key;
                string(&key);
                skip_ws();
                if (i_ >= s_.size() || s_[i_] != ':') err("expected ':'");
                ++i_;
                skip_ws();
                Json v;
                value(&v, depth + 1);
                bool dup = false;
                for (auto& kv : out->o)
                    if (kv.first == key) {
                        kv.second = std::move(v);  // a duplicate key keeps its first position, last value
                        dup = true;
                        break;
                    }
                if (!dup) out->o.emplace_back(std::move(key), std::move(v));
                skip_ws();
                if (i_ < s_.size() && s_[i_] == ',') {
                    ++i_;
                    continue;
                }
                if (i_ < s_.size() && s_[i_] == '}') {
                    ++i_;
                    break;
                }
                err("expected ',' or '}'");
            }
            order_keys(out);
        } else if (c == '[') {
            ++i_;
            out->t = Json::T::Arr;
            skip_ws();
            if (i_ < s_.size() && s_[i_] == ']') {
                ++i_;
                return;
            }
            for (;;) {
                skip_ws();
                out->a.emplace_back();
                value(&out->a.back(), depth + 1);
                skip_ws();
                if (i_ < s_.size() && s_[i_] == ',') {
                    ++i_;
                    continue;
                }
                if (i_ < s_.size() && s_[i_] == ']') {
                    ++i_;
                    break;
                }
                err("expected ',' or ']'");
            }
        } else if (c == '"') {
            out->t = Json::T::Str;
            string(&out->s);
        } else if (c == 't') {
            if (!lit("true")) err("bad literal");
            out->t = Json::T::Bool;
            out->b = true;
        } else if (c == 'f') {
            if (!lit("false")) err("bad literal");
            out->t = Json::T::Bool;
            out->b = false;
        } else if (c == 'n') {
            if (!lit("null")) err("bad literal");
            out->t = Json::T::Null;
        } else {
            number(out);
        }
    }
};

}  // namespace detail

inline void parse_json(std::string_view text, Json* out) { detail::JsonParser(text).parse(out); }

// ---- constants (reflection_loader.js) ----

inline constexpr uint32_t kManifestSchemaVersion = 1;
inline constexpr uint32_t kStageReflectionMinVersion = 1;
inline constexpr uint32_t kStageReflectionVersionRootTableOffset = 2;
inline constexpr uint32_t kStageReflectionVersionUnresolved = 3;
inline constexpr uint32_t kStageReflectionMaxVersion = 3;
inline constexpr uint32_t kMaxWebGpuBindGroups = 4;
inline constexpr uint32_t kMaxReflectedGroups = 64;
inline constexpr uint32_t kMaxHeapIndexShift = 31;
inline constexpr uint32_t kMaxLaneIndexedWindow = 256;
inline constexpr uint32_t kMaxMeshVertices = 256, kMaxMeshPrimitives = 256, kMaxMeshAttributes = 15;
inline constexpr uint32_t kRtArenaGroup = 3, kRtArenaFirstBinding = 16, kRtArenaLastBinding = 21;
inline constexpr uint32_t kRtArenaFullMask = 0x3f;
inline constexpr uint32_t kInlineSamplerSdwords = 4;
inline constexpr uint32_t kVisVertex = 1, kVisFragment = 2, kVisCompute = 4, kValidVisibilityMask = 7;

// ---- helpers (isNonNegativeInt, normalizeShaderHash, ...) ----

inline bool is_u32_int(const Json* v) {
    return v && v->t == Json::T::Num && std::isfinite(v->n) && v->n == std::floor(v->n) && v->n >= 0 && v->n <= 4294967295.0;
}
inline bool is_js_int(const Json* v) { return v && v->t == Json::T::Num && std::isfinite(v->n) && v->n == std::floor(v->n); }
inline uint32_t as_u32(const Json* v) { return static_cast<uint32_t>(v->n); }
inline bool is_bool(const Json* v) { return v && v->t == Json::T::Bool; }
inline bool is_str(const Json* v) { return v && v->t == Json::T::Str; }
inline bool nonempty_str(const Json* v) { return is_str(v) && !v->s.empty(); }

inline std::string js_trim(std::string_view s) {
    size_t a = 0, b = s.size();
    auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };
    while (a < b && ws(s[a])) ++a;
    while (b > a && ws(s[b - 1])) --b;
    return std::string(s.substr(a, b - a));
}
inline std::string js_lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

inline std::string format_hash(uint64_t h) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "0x%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// normalizeShaderHash for a string: optional 0x, 1..16 hex digits.
inline std::optional<uint64_t> parse_hash_string(std::string_view raw) {
    std::string t = js_trim(raw);
    size_t i = 0;
    if (t.size() >= 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) i = 2;
    const size_t n = t.size() - i;
    if (n < 1 || n > 16) return std::nullopt;
    uint64_t v = 0;
    for (size_t k = i; k < t.size(); ++k) {
        const char c = t[k];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= static_cast<uint64_t>(c - '0');
        else if (c >= 'a' && c <= 'f') v |= static_cast<uint64_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= static_cast<uint64_t>(c - 'A' + 10);
        else return std::nullopt;
    }
    return v;
}

// validateHashKey over a JSON value (a string, or a non-negative safe integer).
inline uint64_t hash_from_json(const Json& v, const std::string& path) {
    if (v.t == Json::T::Str) {
        if (auto h = parse_hash_string(v.s)) return *h;
        fail("invalid 64-bit hex hash \"" + v.s + "\"", path);
    }
    if (v.t == Json::T::Num && std::isfinite(v.n) && v.n == std::floor(v.n) && v.n >= 0 && v.n <= 9007199254740991.0)
        return static_cast<uint64_t>(v.n);
    fail("invalid 64-bit hex hash", path);
}

inline uint64_t fnv1a64(std::string_view bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : bytes) {
        h ^= c;
        h *= 0x00000100000001b3ull;
    }
    return h;
}

// ---- layout entries (GPUBindGroupLayoutEntry) ----

struct LayoutEntryDesc {
    enum class Kind : uint8_t { Buffer, Sampler, Texture, StorageTexture, ExternalTexture };
    uint32_t binding = 0;
    uint32_t visibility = 0;
    Kind kind = Kind::Buffer;
    uint8_t buffer_type = 0;   // 0 uniform, 1 storage, 2 read-only-storage
    bool has_dynamic_offset = false;
    uint64_t min_binding_size = 0;
    uint8_t sampler_type = 0;  // 0 filtering, 1 non-filtering, 2 comparison
    uint8_t sample_type = 0;   // 0 float, 1 unfilterable-float, 2 depth, 3 sint, 4 uint
    uint8_t view_dim = 2;      // GPU enum code: 1 1d, 2 2d, 3 2d-array, 4 cube, 5 cube-array, 6 3d
    bool multisampled = false;
    uint8_t access = 0;        // storage texture: 0 write-only, 1 read-only, 2 read-write
    std::string storage_format;  // WebGPU name, as the document wrote it
};

inline const char* view_dim_name(uint8_t c) {
    static const char* k[7] = {"", "1d", "2d", "2d-array", "cube", "cube-array", "3d"};
    return c < 7 ? k[c] : "";
}
inline std::optional<uint8_t> view_dim_code(std::string_view n, bool storage) {
    static const std::pair<const char*, uint8_t> k[6] = {{"1d", 1}, {"2d", 2}, {"2d-array", 3}, {"cube", 4}, {"cube-array", 5}, {"3d", 6}};
    for (const auto& e : k)
        if (n == e.first) {
            if (storage && (e.second == 4 || e.second == 5)) return std::nullopt;
            return e.second;
        }
    return std::nullopt;
}
inline std::optional<uint8_t> sample_type_code(std::string_view n) {
    static const char* k[5] = {"float", "unfilterable-float", "depth", "sint", "uint"};
    for (uint8_t i = 0; i < 5; ++i)
        if (n == k[i]) return i;
    return std::nullopt;
}

// ---- vertex formats (VERTEX_FORMAT_BYTE_SIZES); the code is the GPU enum code (docs/gpu_commands.md) ----

struct VertexFormatName {
    const char* name;
    uint8_t code;
    uint32_t bytes;
};
inline const VertexFormatName* vertex_format_by_name(std::string_view n) {
    static const VertexFormatName k[] = {
        {"uint8", 1, 1},     {"uint8x2", 2, 2},     {"uint8x4", 3, 4},     {"sint8", 4, 1},      {"sint8x2", 5, 2},
        {"sint8x4", 6, 4},   {"unorm8", 7, 1},      {"unorm8x2", 8, 2},    {"unorm8x4", 9, 4},   {"snorm8", 10, 1},
        {"snorm8x2", 11, 2}, {"snorm8x4", 12, 4},   {"uint16", 13, 2},     {"uint16x2", 14, 4},  {"uint16x4", 15, 8},
        {"sint16", 16, 2},   {"sint16x2", 17, 4},   {"sint16x4", 18, 8},   {"unorm16", 19, 2},   {"unorm16x2", 20, 4},
        {"unorm16x4", 21, 8}, {"snorm16", 22, 2},   {"snorm16x2", 23, 4},  {"snorm16x4", 24, 8}, {"float16", 25, 2},
        {"float16x2", 26, 4}, {"float16x4", 27, 8}, {"float32", 28, 4},    {"float32x2", 29, 8}, {"float32x3", 30, 12},
        {"float32x4", 31, 16}, {"uint32", 32, 4},   {"uint32x2", 33, 8},   {"uint32x3", 34, 12}, {"uint32x4", 35, 16},
        {"sint32", 36, 4},   {"sint32x2", 37, 8},   {"sint32x3", 38, 12},  {"sint32x4", 39, 16}, {"unorm10-10-10-2", 40, 4},
        {"unorm8x4-bgra", 41, 4}};
    for (const auto& e : k)
        if (n == e.name) return &e;
    return nullptr;
}

// ---- stage / shader records ----

struct VertexInputRec {
    uint32_t location = 0;
    std::string format;  // WebGPU vertex format; empty = null (no format: the vertex layout cannot be built)
};

struct StageRecord {
    uint64_t hash = 0;
    std::string hash_hex;
    std::string stage;  // "vertex" | "fragment" | "compute"
    uint32_t visibility = 0;
    std::string entry_point;
    std::string name;  // empty = none
    std::vector<std::vector<LayoutEntryDesc>> group_layouts;  // `bindGroupLayouts[g].entries`, sorted by binding
    bool layout_complete = true;
    std::vector<VertexInputRec> vertex_inputs;  // sorted by location
    bool from_sidecar = false;
};

// ---- layout entry normalization (normalizeLayoutEntry) ----

namespace detail {

inline int32_t to_int32(double d) {
    if (!std::isfinite(d)) return 0;
    const double m = std::fmod(std::trunc(d), 4294967296.0);
    const uint32_t u = static_cast<uint32_t>(m < 0 ? m + 4294967296.0 : m);
    return static_cast<int32_t>(u);
}

inline const Json* first_defined(const Json* a, const Json* b) { return a ? a : b; }

}  // namespace detail

inline LayoutEntryDesc normalize_layout_entry(const Json& raw, const std::string& path, std::optional<uint32_t> fallback_visibility) {
    if (!raw.is_obj()) fail("layout entry must be an object", path);
    const Json* binding = raw.get("binding");
    if (!is_u32_int(binding)) fail("invalid binding index", path + ".binding");
    const Json* vis = raw.get("visibility");
    double visibility;
    if (!vis && fallback_visibility) {
        visibility = *fallback_visibility;
    } else if (vis && vis->t == Json::T::Num) {
        visibility = vis->n;
    } else {
        fail("invalid visibility bitmask", path + ".visibility");
    }
    if (!(std::isfinite(visibility) && visibility == std::floor(visibility)) || visibility < 1 ||
        (detail::to_int32(visibility) & ~static_cast<int32_t>(kValidVisibilityMask)) != 0)
        fail("invalid visibility bitmask", path + ".visibility");

    const Json* rb = raw.get("buffer");
    const Json* rs = raw.get("sampler");
    const Json* rt = raw.get("texture");
    const Json* rst = detail::first_defined(raw.get("storageTexture"), raw.get("storage_texture"));
    const Json* ret = detail::first_defined(raw.get("externalTexture"), raw.get("external_texture"));
    const int members = (rb ? 1 : 0) + (rs ? 1 : 0) + (rt ? 1 : 0) + (rst ? 1 : 0) + (ret ? 1 : 0);
    if (members != 1)
        fail("layout entry must define exactly one resource member (buffer, sampler, texture, storageTexture, or externalTexture)", path);

    LayoutEntryDesc e;
    e.binding = as_u32(binding);
    e.visibility = static_cast<uint32_t>(visibility) & 0xffffffffu;

    if (rb) {
        if (!rb->is_obj()) fail("buffer layout must be an object", path + ".buffer");
        const Json* type = rb->get("type");
        if (!type) {
            e.buffer_type = 0;
        } else if (is_str(type) && type->s == "uniform") {
            e.buffer_type = 0;
        } else if (is_str(type) && type->s == "storage") {
            e.buffer_type = 1;
        } else if (is_str(type) && type->s == "read-only-storage") {
            e.buffer_type = 2;
        } else {
            fail("invalid buffer type", path + ".buffer.type");
        }
        const Json* dyn = detail::first_defined(rb->get("hasDynamicOffset"), rb->get("has_dynamic_offset"));
        if (dyn && !is_bool(dyn)) fail("hasDynamicOffset must be a boolean", path + ".buffer.hasDynamicOffset");
        e.has_dynamic_offset = dyn && dyn->b;
        const Json* ms = detail::first_defined(rb->get("minBindingSize"), rb->get("min_binding_size"));
        if (ms) {
            if (!is_js_int(ms) || ms->n < 0) fail("minBindingSize must be a non-negative integer", path + ".buffer.minBindingSize");
            e.min_binding_size = static_cast<uint64_t>(ms->n);
        }
        e.kind = LayoutEntryDesc::Kind::Buffer;
    } else if (rs) {
        if (!rs->is_obj()) fail("sampler layout must be an object", path + ".sampler");
        const Json* type = rs->get("type");
        if (!type || (is_str(type) && type->s == "filtering")) e.sampler_type = 0;
        else if (is_str(type) && type->s == "non-filtering") e.sampler_type = 1;
        else if (is_str(type) && type->s == "comparison") e.sampler_type = 2;
        else fail("invalid sampler type", path + ".sampler.type");
        e.kind = LayoutEntryDesc::Kind::Sampler;
    } else if (rt) {
        if (!rt->is_obj()) fail("texture layout must be an object", path + ".texture");
        const Json* st = detail::first_defined(rt->get("sampleType"), rt->get("sample_type"));
        e.sample_type = 0;
        if (st) {
            auto c = is_str(st) ? sample_type_code(st->s) : std::nullopt;
            if (!c) fail("invalid texture sampleType", path + ".texture.sampleType");
            e.sample_type = *c;
        }
        const Json* vd = detail::first_defined(rt->get("viewDimension"), rt->get("view_dimension"));
        e.view_dim = 2;
        if (vd) {
            auto c = is_str(vd) ? view_dim_code(vd->s, false) : std::nullopt;
            if (!c) fail("invalid texture viewDimension", path + ".texture.viewDimension");
            e.view_dim = *c;
        }
        const Json* msd = rt->get("multisampled");
        if (msd && !is_bool(msd)) fail("multisampled must be a boolean", path + ".texture.multisampled");
        e.multisampled = msd && msd->b;
        if (e.multisampled && e.view_dim != 2) fail("multisampled texture must have viewDimension \"2d\"", path + ".texture");
        e.kind = LayoutEntryDesc::Kind::Texture;
    } else if (rst) {
        if (!rst->is_obj()) fail("storageTexture layout must be an object", path + ".storageTexture");
        const Json* acc = rst->get("access");
        if (!acc || (is_str(acc) && acc->s == "write-only")) e.access = 0;
        else if (is_str(acc) && acc->s == "read-only") e.access = 1;
        else if (is_str(acc) && acc->s == "read-write") e.access = 2;
        else fail("invalid storageTexture access", path + ".storageTexture.access");
        const Json* fmt = rst->get("format");
        if (!nonempty_str(fmt)) fail("storageTexture format must be a non-empty string", path + ".storageTexture.format");
        e.storage_format = fmt->s;
        const Json* vd = detail::first_defined(rst->get("viewDimension"), rst->get("view_dimension"));
        e.view_dim = 2;
        if (vd) {
            auto c = is_str(vd) ? view_dim_code(vd->s, true) : std::nullopt;
            if (!c) fail("invalid storageTexture viewDimension", path + ".storageTexture.viewDimension");
            e.view_dim = *c;
        }
        e.kind = LayoutEntryDesc::Kind::StorageTexture;
    } else {
        if (!ret->is_obj()) fail("externalTexture layout must be an object", path + ".externalTexture");
        e.kind = LayoutEntryDesc::Kind::ExternalTexture;
    }
    return e;
}

// ---- binding sources (parseBindingSource and its helpers): validated, then dropped ----

namespace detail {

inline bool has(const Json& o, std::string_view k) { return o.get(k) != nullptr; }

inline void check_shift(const Json* v, const char* what, const std::string& path) {
    if (!is_u32_int(v) || as_u32(v) > kMaxHeapIndexShift) fail(std::string(what) + " must be an integer in 0..31", path);
}

inline void parse_cbv_dword_spec(const Json* body, const std::string& path) {
    if (!body || !body->is_obj()) fail("CBV dword descriptor must be an object", path);
    const Json* cbv = body->get("cbv");
    if (!cbv || !cbv->is_obj()) fail("cbv must be an object", path + ".cbv");
    if (!is_u32_int(cbv->get("user_sgpr"))) fail("cbv.user_sgpr must be a non-negative uint32 safe integer", path + ".cbv.user_sgpr");
    if (has(*cbv, "root_descriptor") && !is_bool(cbv->get("root_descriptor"))) fail("cbv.root_descriptor must be a boolean", path + ".cbv.root_descriptor");
    if (has(*cbv, "root_constants") && !is_bool(cbv->get("root_constants"))) fail("cbv.root_constants must be a boolean", path + ".cbv.root_constants");
    const bool rd = cbv->get("root_descriptor") && cbv->get("root_descriptor")->b;
    const bool rc = cbv->get("root_constants") && cbv->get("root_constants")->b;
    const bool tbl = has(*cbv, "table_byte_offset");
    if ((rd ? 1 : 0) + (rc ? 1 : 0) + (tbl ? 1 : 0) != 1)
        fail("cbv must specify exactly one of root_descriptor: true, root_constants: true, or table_byte_offset", path + ".cbv");
    if (!rd && !rc && !is_u32_int(cbv->get("table_byte_offset")))
        fail("cbv.table_byte_offset must be a non-negative uint32 safe integer", path + ".cbv.table_byte_offset");
    const Json* bo = body->get("byte_offset");
    if (!is_u32_int(bo) || as_u32(bo) % 4 != 0) fail("byte_offset must be a dword-aligned non-negative uint32 integer", path + ".byte_offset");
    const Json* bit = body->get("bit_offset");
    if (bit && !bit->is_null() && (!is_u32_int(bit) || as_u32(bit) > kMaxHeapIndexShift))
        fail("bit_offset must be an integer in 0..31", path + ".bit_offset");
    const Json* mask = body->get("mask");
    if (mask && !mask->is_null() && (!is_u32_int(mask) || as_u32(mask) == 0)) fail("mask must be a non-zero uint32 integer", path + ".mask");
}

inline void check_opt_stride(const Json& body, const char* kind, const std::string& path) {
    const Json* st = body.get("stride");
    if (st && !st->is_null() && !is_u32_int(st)) fail(std::string(kind) + ".stride must be a non-negative uint32 integer", path + ".stride");
}

}  // namespace detail

struct SourceInfo {
    enum class Kind : uint8_t {
        Table, RootConstants, RootDescriptor, StaticSampler, SamplerHeap, ResourceHeap, InlineSampler,
        CbvIndexedSamplerHeap, CbvIndexedResourceHeap, LaneIndexedSamplerHeap, LaneIndexedResourceHeap,
        Unresolved, RtArena
    } kind = Kind::Table;
};

inline SourceInfo parse_binding_source(const Json& src, const std::string& path, uint32_t version, const std::string& binding_kind) {
    using namespace detail;
    using K = SourceInfo::Kind;
    if (!src.is_obj()) fail("binding source must be an object or null", path);
    const bool sampler_heap = has(src, "sampler_heap"), resource_heap = has(src, "resource_heap");
    const bool cbv_s = has(src, "cbv_indexed_sampler_heap"), cbv_r = has(src, "cbv_indexed_resource_heap");
    const bool lane_s = has(src, "lane_indexed_sampler_heap"), lane_r = has(src, "lane_indexed_resource_heap");
    const bool inline_s = has(src, "inline_sampler"), unresolved = has(src, "unresolved"), rt_arena = has(src, "rt_arena");
    const bool static_s = has(src, "static_sampler"), user_sgpr = has(src, "user_sgpr"), table_off = has(src, "table_byte_offset");
    const bool root_c = has(src, "root_constants"), root_d = has(src, "root_descriptor");

    if (sampler_heap && resource_heap) fail("binding source cannot define both sampler_heap and resource_heap", path);
    const bool v2v3 = cbv_s || cbv_r || lane_s || lane_r || unresolved || rt_arena;

    const Json* rh = src.get("resource_heap");
    if (resource_heap && rh->t == Json::T::Bool && rh->b) {
        if (v2v3 || inline_s || static_s || table_off || root_c || root_d)
            fail("resource_heap source cannot be combined with other binding source fields", path);
        if (user_sgpr && !is_u32_int(src.get("user_sgpr"))) fail("user_sgpr must be a non-negative uint32 safe integer", path + ".user_sgpr");
        return {K::ResourceHeap};
    }
    if (sampler_heap || resource_heap) {
        const char* key = sampler_heap ? "sampler_heap" : "resource_heap";
        if (v2v3 || inline_s || static_s || user_sgpr || table_off || root_c || root_d)
            fail(std::string(key) + " source cannot be combined with other binding source fields", path);
        const Json* h = src.get(key);
        const std::string hp = path + "." + key;
        if (!h->is_obj()) fail(std::string(key) + " must be an object", hp);
        if (!is_u32_int(h->get("base_user_sgpr"))) fail("base_user_sgpr must be a non-negative uint32 safe integer", hp + ".base_user_sgpr");
        if (!is_u32_int(h->get("stride")) || as_u32(h->get("stride")) == 0) fail("stride must be a positive uint32 safe integer (> 0)", hp + ".stride");
        const Json* idx = h->get("index");
        if (!idx || !idx->is_obj()) fail("index must be an object with user_sgpr and shift", hp + ".index");
        if (!is_u32_int(idx->get("user_sgpr"))) fail("index.user_sgpr must be a non-negative uint32 safe integer", hp + ".index.user_sgpr");
        check_shift(idx->get("shift"), "index.shift", hp + ".index.shift");
        return {sampler_heap ? K::SamplerHeap : K::ResourceHeap};
    }
    if (cbv_s || cbv_r) {
        if (cbv_s && cbv_r) fail("binding source cannot define both cbv_indexed_sampler_heap and cbv_indexed_resource_heap", path);
        const char* key = cbv_s ? "cbv_indexed_sampler_heap" : "cbv_indexed_resource_heap";
        if (lane_s || lane_r || inline_s || unresolved || rt_arena || static_s || user_sgpr || table_off || root_c || root_d)
            fail(std::string(key) + " source cannot be combined with other binding source fields", path);
        const Json* b = src.get(key);
        const std::string bp = path + "." + key;
        if (!b->is_obj()) fail(std::string(key) + " must be an object", bp);
        parse_cbv_dword_spec(b, bp);
        check_shift(b->get("shift"), "shift", bp + ".shift");
        check_opt_stride(*b, key, bp);
        return {cbv_s ? K::CbvIndexedSamplerHeap : K::CbvIndexedResourceHeap};
    }
    if (lane_s || lane_r) {
        if (lane_s && lane_r) fail("binding source cannot define both lane_indexed_sampler_heap and lane_indexed_resource_heap", path);
        const char* key = lane_s ? "lane_indexed_sampler_heap" : "lane_indexed_resource_heap";
        if (inline_s || unresolved || rt_arena || static_s || user_sgpr || table_off || root_c || root_d)
            fail(std::string(key) + " source cannot be combined with other binding source fields", path);
        const Json* b = src.get(key);
        const std::string bp = path + "." + key;
        if (!b->is_obj()) fail(std::string(key) + " must be an object", bp);
        const Json* mc = b->get("max_count");
        if (!is_u32_int(mc) || as_u32(mc) < 1 || as_u32(mc) > kMaxLaneIndexedWindow) fail("max_count must be an integer in 1..256", bp + ".max_count");
        check_shift(b->get("shift"), "shift", bp + ".shift");
        check_opt_stride(*b, key, bp);
        const Json* wb = b->get("window_base");
        if (wb && !wb->is_null()) parse_cbv_dword_spec(wb, bp + ".window_base");
        return {lane_s ? K::LaneIndexedSamplerHeap : K::LaneIndexedResourceHeap};
    }
    if (inline_s) {
        if (unresolved || rt_arena || static_s || user_sgpr || table_off || root_c || root_d)
            fail("inline_sampler source cannot be combined with other binding source fields", path);
        const Json* in = src.get("inline_sampler");
        const std::string ip = path + ".inline_sampler";
        if (!in->is_obj()) fail("inline_sampler must be an object", ip);
        const Json* w = in->get("sdwords");
        if (!w || !w->is_arr() || w->a.size() != kInlineSamplerSdwords)
            fail("inline_sampler.sdwords must be an array of 4 unsigned 32-bit integers", ip + ".sdwords");
        for (size_t i = 0; i < w->a.size(); ++i)
            if (!is_u32_int(&w->a[i])) fail("inline_sampler.sdwords[" + std::to_string(i) + "] must be an unsigned 32-bit integer", ip + ".sdwords[" + std::to_string(i) + "]");
        return {K::InlineSampler};
    }
    if (rt_arena) {
        if (unresolved || static_s || user_sgpr || table_off || root_c || root_d)
            fail("rt_arena source cannot be combined with other binding source fields", path);
        const Json* v = src.get("rt_arena");
        if (!(v->t == Json::T::Bool && v->b)) fail("rt_arena source must be true", path + ".rt_arena");
        return {K::RtArena};
    }
    if (unresolved) {
        if (static_s || user_sgpr || table_off || root_c || root_d)
            fail("unresolved source cannot be combined with other binding source fields", path);
        const Json* u = src.get("unresolved");
        const bool marked = (u->t == Json::T::Bool && u->b) || (u->t == Json::T::Str && !u->s.empty());
        if (version < kStageReflectionVersionUnresolved || !marked)
            fail("unresolved source needs \"unresolved\": true or a non-empty reason string, and version >= 3", path + ".unresolved");
        if (binding_kind == "storage" || binding_kind == "storage_texture")
            fail("a " + binding_kind + " binding cannot have an unresolved source", path + ".unresolved");
        return {K::Unresolved};
    }
    if (static_s) {
        if (user_sgpr || table_off || root_c || root_d) fail("static_sampler source cannot be combined with user_sgpr source fields", path);
        if (!is_u32_int(src.get("static_sampler"))) fail("static_sampler must be a non-negative uint32 safe integer", path + ".static_sampler");
        return {K::StaticSampler};
    }
    if (user_sgpr) {
        if (!is_u32_int(src.get("user_sgpr"))) fail("user_sgpr must be a non-negative uint32 safe integer", path + ".user_sgpr");
        if (root_c && !is_bool(src.get("root_constants"))) fail("root_constants must be a boolean", path + ".root_constants");
        if (root_d && !is_bool(src.get("root_descriptor"))) fail("root_descriptor must be a boolean", path + ".root_descriptor");
        const bool rc = root_c && src.get("root_constants")->b;
        const bool rd = root_d && src.get("root_descriptor")->b;
        const bool tbl = table_off;
        if (rc) {
            if (rd || tbl) fail("user_sgpr source must specify exactly one of table_byte_offset, root_constants: true, or root_descriptor: true", path);
            return {K::RootConstants};
        }
        if (rd) {
            if (tbl && version >= kStageReflectionVersionRootTableOffset && !is_u32_int(src.get("table_byte_offset")))
                fail("bad root_descriptor table_byte_offset", path + ".table_byte_offset");
            return {K::RootDescriptor};
        }
        if (!tbl) fail("user_sgpr source must specify exactly one of table_byte_offset, root_constants: true, or root_descriptor: true", path);
        if (!is_u32_int(src.get("table_byte_offset"))) fail("table_byte_offset must be a non-negative uint32 safe integer", path + ".table_byte_offset");
        return {K::Table};
    }
    if (table_off || root_c || root_d) fail("binding source requires user_sgpr alongside table_byte_offset, root_constants, or root_descriptor", path);
    fail("unrecognized binding source", path);
}

inline void parse_mesh_output_layout(const Json& m, const std::string& path) {
    if (!m.is_obj()) fail("mesh_output must be an object", path);
    const Json* mv = detail::first_defined(m.get("max_vertices"), m.get("maxVertices"));
    const Json* mp = detail::first_defined(m.get("max_primitives"), m.get("maxPrimitives"));
    const Json* vw = detail::first_defined(m.get("vertex_words"), m.get("vertexWords"));
    if (!is_u32_int(mv) || as_u32(mv) < 1 || as_u32(mv) > kMaxMeshVertices) fail("mesh_output.max_vertices must be an integer in 1..256", path + ".max_vertices");
    if (!is_u32_int(mp) || as_u32(mp) < 1 || as_u32(mp) > kMaxMeshPrimitives) fail("mesh_output.max_primitives must be an integer in 1..256", path + ".max_primitives");
    const uint32_t max_words = 4 + 4 * kMaxMeshAttributes;
    if (!is_u32_int(vw) || as_u32(vw) < 4 || as_u32(vw) % 4 != 0 || as_u32(vw) > max_words)
        fail("mesh_output.vertex_words must be 4 + 4 * n", path + ".vertex_words");
}

// ---- shared helpers of the two document kinds ----

inline std::optional<std::string> normalize_stage_name(const Json* raw, bool allow_pixel) {
    if (!is_str(raw)) return std::nullopt;
    const std::string l = js_lower(js_trim(raw->s));
    if (allow_pixel && l == "pixel") return std::string("fragment");
    if (l == "vertex" || l == "fragment" || l == "compute") return l;
    return std::nullopt;
}
inline uint32_t stage_visibility(const std::string& stage) { return stage == "vertex" ? kVisVertex : stage == "fragment" ? kVisFragment : kVisCompute; }

// resolveVertexInputMode (only its validation matters here).
inline void check_vertex_input_mode(const Json& st, const std::string& path) {
    const Json* m = st.get("vertex_input_mode");
    if (!m) m = st.get("vertexInputMode");
    if (!m) {
        const Json* bs = st.get("binding_summary");
        if (bs && bs->is_obj() && bs->get("vertex_input_mode")) m = bs->get("vertex_input_mode");
        else {
            const Json* bs2 = st.get("bindingSummary");
            if (bs2 && bs2->is_obj() && bs2->get("vertexInputMode")) m = bs2->get("vertexInputMode");
        }
    }
    if (m && !m->is_null() && (!is_str(m) || js_trim(m->s).empty())) fail("vertex_input_mode must be a non-empty string", path + ".vertex_input_mode");
}

// ---- sidecar (`playgta6-stage-reflection`, versions 1..3) ----

namespace detail {

struct SidecarBinding {
    uint32_t group = 0, binding = 0;
    std::string kind;  // uniform | storage | read_only_storage | texture | storage_texture | sampler | comparison_sampler
    uint64_t min_binding_size = 0;
    std::string storage_format;  // empty unless a non-empty string was given
    uint8_t view_dim = 2, sample_type = 0;
    bool multisampled = false;
};

inline std::string sidecar_kind(const std::string& raw) {
    const std::string k = js_lower(js_trim(raw));
    if (k == "uniform") return "uniform";
    if (k == "storage") return "storage";
    if (k == "read_only_storage" || k == "read-only-storage") return "read_only_storage";
    if (k == "texture" || k == "texture_2d") return "texture";
    if (k == "storage_texture" || k == "storage-texture" || k == "storagetexture") return "storage_texture";
    if (k == "sampler") return "sampler";
    if (k == "comparison_sampler" || k == "comparison-sampler" || k == "sampler_comparison") return "comparison_sampler";
    return std::string();
}

inline std::string dash_norm(std::string s) {
    s = js_lower(js_trim(s));
    for (char& c : s)
        if (c == '_') c = '-';
    return s;
}

// buildSidecarLayoutEntry: nullopt when the loader produces no entry (storage texture without a format).
inline std::optional<LayoutEntryDesc> sidecar_layout_entry(const SidecarBinding& r, uint32_t visibility) {
    LayoutEntryDesc e;
    e.binding = r.binding;
    e.visibility = visibility;
    if (r.kind == "uniform" || r.kind == "storage" || r.kind == "read_only_storage") {
        e.kind = LayoutEntryDesc::Kind::Buffer;
        e.buffer_type = r.kind == "uniform" ? 0 : r.kind == "storage" ? 1 : 2;
        e.min_binding_size = r.min_binding_size;
        return e;
    }
    if (r.kind == "texture") {
        e.kind = LayoutEntryDesc::Kind::Texture;
        e.sample_type = r.sample_type;
        e.view_dim = r.view_dim;
        e.multisampled = r.multisampled;
        return e;
    }
    if (r.kind == "storage_texture") {
        if (r.storage_format.empty()) return std::nullopt;
        e.kind = LayoutEntryDesc::Kind::StorageTexture;
        e.access = 0;  // write-only
        e.storage_format = r.storage_format;
        e.view_dim = r.view_dim;
        return e;
    }
    e.kind = LayoutEntryDesc::Kind::Sampler;
    e.sampler_type = r.kind == "sampler" ? 0 : 2;
    return e;
}

inline VertexInputRec parse_sidecar_vertex_input(const Json& raw, const std::string& path) {
    if (!raw.is_obj()) fail("vertex_input must be an object", path);
    if (!is_u32_int(raw.get("location"))) fail("invalid location", path + ".location");
    const Json* slot = first_defined(raw.get("vb_slot"), raw.get("vbSlot"));
    if (slot && !is_u32_int(slot)) fail("vb_slot must be a non-negative uint32 integer", path + ".vb_slot");
    const Json* off = first_defined(raw.get("byte_offset"), raw.get("byteOffset"));
    if (off && !is_u32_int(off)) fail("byte_offset must be a non-negative uint32 integer", path + ".byte_offset");
    const Json* comp = raw.get("components");
    if (comp && !comp->is_null() && !is_u32_int(comp)) fail("components must be a non-negative uint32 integer", path + ".components");
    const Json* ct = first_defined(raw.get("component_type"), raw.get("componentType"));
    if (ct && !ct->is_null() && !is_str(ct)) fail("component_type must be a string or null", path + ".component_type");
    const Json* ff = first_defined(raw.get("fetch_format"), raw.get("fetchFormat"));
    if (ff && !ff->is_null() && !is_str(ff)) fail("fetch_format must be a string or null", path + ".fetch_format");
    const Json* fmt = raw.get("format");
    if (fmt && !fmt->is_null() && !is_str(fmt)) fail("format must be a string or null", path + ".format");
    VertexInputRec v;
    v.location = as_u32(raw.get("location"));
    const bool have_ff = ff && !ff->is_null();
    if (fmt && !fmt->is_null()) v.format = fmt->s;
    else if (have_ff) v.format = ff->s;
    return v;
}

}  // namespace detail

inline std::vector<std::shared_ptr<StageRecord>> parse_stage_reflection_sidecar(const Json& raw) {
    using namespace detail;
    if (!raw.is_obj()) fail("stage reflection sidecar root must be a JSON object");
    const Json* schema = raw.get("schema");
    if (!is_str(schema) || schema->s != "playgta6-stage-reflection") fail("unsupported stage reflection schema", "schema");
    const Json* ver = raw.get("version");
    if (!is_js_int(ver) || ver->n < kStageReflectionMinVersion || ver->n > kStageReflectionMaxVersion)
        fail("unsupported stage reflection version (expected 1..3)", "version");
    const uint32_t version = static_cast<uint32_t>(ver->n);
    const Json* stages_j = raw.get("stages");
    if (!stages_j || !stages_j->is_obj()) fail("stages must be an object", "stages");

    std::vector<std::shared_ptr<StageRecord>> out;
    std::set<uint64_t> seen_hashes;
    for (const auto& kv : stages_j->o) {
        const std::string st_path = "stages[\"" + kv.first + "\"]";
        const Json& rs = kv.second;
        const auto hash = parse_hash_string(kv.first);
        if (!hash) fail("invalid 64-bit hex hash \"" + kv.first + "\"", st_path);
        if (!seen_hashes.insert(*hash).second) fail("duplicate stage hash " + format_hash(*hash), st_path);
        if (!rs.is_obj()) fail("stage record must be an object", st_path);

        auto stage = normalize_stage_name(rs.get("stage"), true);
        if (!stage) fail("unsupported shader stage (expected \"vertex\", \"fragment\", \"pixel\", or \"compute\")", st_path + ".stage");
        const uint32_t visibility = stage_visibility(*stage);

        const Json* mesh = first_defined(rs.get("mesh_output"), rs.get("meshOutput"));
        if (mesh && !mesh->is_null()) parse_mesh_output_layout(*mesh, st_path + ".mesh_output");
        const Json* rta = rs.get("rt_arena");
        if (rta && !is_bool(rta)) fail("rt_arena must be a boolean", st_path + ".rt_arena");

        const Json* bindings_j = rs.get("bindings");
        if (!bindings_j || !bindings_j->is_arr()) fail("bindings must be an array", st_path + ".bindings");

        std::set<std::pair<uint32_t, uint32_t>> seen_slots;
        std::map<uint32_t, std::vector<LayoutEntryDesc>> group_entries;
        int64_t max_group = -1;
        bool layout_complete = true;
        for (size_t i = 0; i < bindings_j->a.size(); ++i) {
            const Json& b = bindings_j->a[i];
            const std::string bp = st_path + ".bindings[" + std::to_string(i) + "]";
            if (!b.is_obj()) fail("binding record must be an object", bp);
            if (!is_u32_int(b.get("group"))) fail("invalid group", bp + ".group");
            if (!is_u32_int(b.get("binding"))) fail("invalid binding", bp + ".binding");
            const uint32_t group = as_u32(b.get("group")), binding = as_u32(b.get("binding"));
            const Json* bsrc = b.get("source");
            if (bsrc && bsrc->is_obj() && bsrc->get("rt_arena") && bsrc->get("rt_arena")->t == Json::T::Bool && bsrc->get("rt_arena")->b) {
                parse_binding_source(*bsrc, bp + ".source", version, std::string());
                if (group != kRtArenaGroup || binding < kRtArenaFirstBinding || binding > kRtArenaLastBinding)
                    fail("an rt_arena binding must be group 3, binding 16..21", bp);
                continue;  // runtime-owned, not part of the layout
            }
            if (!seen_slots.insert({group, binding}).second)
                fail("duplicate binding at @group(" + std::to_string(group) + ") @binding(" + std::to_string(binding) + ")", bp);
            const Json* name = b.get("name");
            if (name && !name->is_null() && !is_str(name)) fail("name must be a string or null", bp + ".name");
            const Json* kind_j = b.get("kind");
            if (!is_str(kind_j)) fail("kind must be a string", bp + ".kind");
            SidecarBinding rec;
            rec.group = group;
            rec.binding = binding;
            rec.kind = sidecar_kind(kind_j->s);
            if (rec.kind.empty()) fail("unknown binding kind \"" + kind_j->s + "\"", bp + ".kind");
            if (!bsrc || bsrc->is_null()) fail("binding source is required in stage reflection sidecar", bp + ".source");
            parse_binding_source(*bsrc, bp + ".source", version, rec.kind);

            const Json* mbs = first_defined(b.get("min_binding_size_bytes"), b.get("minBindingSizeBytes"));
            if (mbs && !mbs->is_null()) {
                if (!is_u32_int(mbs)) fail("min_binding_size_bytes must be a non-negative uint32 safe integer", bp + ".min_binding_size_bytes");
                rec.min_binding_size = as_u32(mbs);
            }
            const Json* sf = first_defined(b.get("storage_format"), b.get("storageFormat"));
            if (sf && !sf->is_null()) {
                if ((!is_str(sf) || sf->s.empty()) && !is_u32_int(sf))
                    fail("storage_format must be a non-empty string or non-negative integer", bp + ".storage_format");
                if (is_str(sf)) rec.storage_format = sf->s;  // an integer code leaves the layout entry out (typeof !== 'string')
            }
            const bool is_tex = rec.kind == "texture", is_stex = rec.kind == "storage_texture";
            const Json* vd = first_defined(b.get("view_dimension"), b.get("viewDimension"));
            if (is_tex || is_stex) {
                uint8_t code = 2;
                if (vd && !vd->is_null()) {
                    if (!is_str(vd)) fail("view_dimension must be a string", bp + ".view_dimension");
                    auto c = view_dim_code(dash_norm(vd->s), is_stex);
                    if (!c) fail("invalid view_dimension \"" + vd->s + "\"", bp + ".view_dimension");
                    code = *c;
                }
                rec.view_dim = code;
            }
            const Json* stj = first_defined(b.get("sample_type"), b.get("sampleType"));
            if (is_tex) {
                uint8_t code = 0;
                if (stj && !stj->is_null()) {
                    if (!is_str(stj)) fail("sample_type must be a string", bp + ".sample_type");
                    auto c = sample_type_code(dash_norm(stj->s));
                    if (!c) fail("invalid sample_type \"" + stj->s + "\"", bp + ".sample_type");
                    code = *c;
                }
                rec.sample_type = code;
            }
            const Json* msj = b.get("multisampled");
            if (msj && !is_bool(msj)) fail("multisampled must be a boolean", bp + ".multisampled");
            rec.multisampled = msj && msj->b;
            if (is_tex && rec.multisampled && rec.view_dim != 2) fail("multisampled texture must have view_dimension \"2d\"", bp + ".view_dimension");

            if (auto entry = sidecar_layout_entry(rec, visibility)) {
                max_group = std::max<int64_t>(max_group, group);
                group_entries[group].push_back(std::move(*entry));
            } else {
                layout_complete = false;
            }
        }

        auto rec = std::make_shared<StageRecord>();
        rec->from_sidecar = true;
        if (max_group >= 0) {
            if (max_group >= static_cast<int64_t>(kMaxReflectedGroups))
                fail("bind group index " + std::to_string(max_group) + " exceeds reflection limit 64", st_path + ".bindings");
            rec->group_layouts.resize(static_cast<size_t>(max_group) + 1);
            for (auto& g : group_entries) {
                auto& v = g.second;
                std::stable_sort(v.begin(), v.end(), [](const LayoutEntryDesc& x, const LayoutEntryDesc& y) { return x.binding < y.binding; });
                rec->group_layouts[g.first] = std::move(v);
            }
        }

        const Json* vi = first_defined(rs.get("vertex_inputs"), rs.get("vertexInputs"));
        if (vi && !vi->is_null()) {
            if (!vi->is_arr()) fail("vertex_inputs must be an array", st_path + ".vertex_inputs");
            for (size_t i = 0; i < vi->a.size(); ++i)
                rec->vertex_inputs.push_back(parse_sidecar_vertex_input(vi->a[i], st_path + ".vertex_inputs[" + std::to_string(i) + "]"));
            std::stable_sort(rec->vertex_inputs.begin(), rec->vertex_inputs.end(),
                             [](const VertexInputRec& x, const VertexInputRec& y) { return x.location < y.location; });
        }
        check_vertex_input_mode(rs, st_path);

        const Json* ep = rs.get("entry_point");
        const Json* ep2 = rs.get("entryPoint");
        rec->entry_point = nonempty_str(ep) ? ep->s : nonempty_str(ep2) ? ep2->s
                           : *stage == "vertex" ? "vs_main" : *stage == "fragment" ? "fs_main" : "cs_main";
        const Json* nm = rs.get("name");
        if (nonempty_str(nm)) rec->name = nm->s;
        rec->hash = *hash;
        rec->hash_hex = format_hash(*hash);
        rec->stage = *stage;
        rec->visibility = visibility;
        rec->layout_complete = layout_complete;
        out.push_back(std::move(rec));
    }
    return out;
}

// ---- S6 layout manifest (`layout-manifest.json`, schema_version 1) ----

struct ManifestTables {
    std::unordered_map<std::string, std::vector<LayoutEntryDesc>> bind_group_layouts;  // id hex -> entries sorted by binding
    std::unordered_map<std::string, std::vector<std::string>> pipeline_layouts;        // id hex -> bind group layout id hexes
    std::vector<std::shared_ptr<StageRecord>> shaders;
    size_t pipelines = 0;
};

namespace detail {

inline void parse_binding_record(const Json& raw, const std::string& path) {
    if (!raw.is_obj()) fail("binding record must be an object", path);
    if (!is_u32_int(raw.get("group"))) fail("invalid group", path + ".group");
    if (!is_u32_int(raw.get("binding"))) fail("invalid binding", path + ".binding");
    const Json* name = raw.get("name");
    if (name && !name->is_null() && !is_str(name)) fail("name must be a string or null", path + ".name");
    const Json* res = raw.get("resource");
    static const char* kRes[6] = {"buffer", "sampler", "texture", "storageTexture", "externalTexture", "accelerationStructure"};
    bool ok = false;
    if (is_str(res))
        for (const char* r : kRes)
            if (res->s == r) ok = true;
    if (!ok) fail("invalid resource kind", path + ".resource");
    const Json* inl = raw.get("in_layout");
    if (inl && !is_bool(inl)) fail("in_layout must be a boolean", path + ".in_layout");
    const Json* reg = raw.get("register");
    if (reg && !reg->is_null()) {
        if (!reg->is_obj()) fail("register must be an object or null", path + ".register");
        const Json* cl = reg->get("class");
        if (!is_str(cl) || !(cl->s == "b" || cl->s == "t" || cl->s == "u" || cl->s == "s")) fail("invalid register class", path + ".register.class");
        if (!is_u32_int(reg->get("index")) || !is_u32_int(reg->get("space"))) fail("register index and space must be non-negative integers", path + ".register");
    }
    const Json* sp = raw.get("spirv");
    if (sp && !sp->is_null()) {
        if (!sp->is_obj()) fail("spirv must be an object or null", path + ".spirv");
        if (!is_u32_int(sp->get("set")) || !is_u32_int(sp->get("binding"))) fail("spirv set and binding must be non-negative integers", path + ".spirv");
    }
    const Json* rc = raw.get("root_constants");
    if (rc && !is_bool(rc)) fail("root_constants must be a boolean", path + ".root_constants");
    const Json* ba = raw.get("binding_array");
    if (ba && !ba->is_null()) {
        if (!ba->is_obj()) fail("binding_array must be an object or null", path + ".binding_array");
        const Json* c = ba->get("count");
        // `count !== null && !isNonNegativeInt(count)`: an absent count (undefined) is rejected too.
        if (!(c && c->is_null()) && !is_u32_int(c)) fail("binding_array.count must be a non-negative integer or null", path + ".binding_array.count");
    }
    const Json* src = raw.get("source");
    if (src && !src->is_null()) parse_binding_source(*src, path + ".source", kStageReflectionMaxVersion, std::string());
    const Json* mbs = first_defined(raw.get("min_binding_size_bytes"), raw.get("minBindingSizeBytes"));
    if (mbs && !mbs->is_null() && !is_u32_int(mbs)) fail("min_binding_size_bytes must be a non-negative uint32 safe integer", path + ".min_binding_size_bytes");
    const Json* sf = first_defined(raw.get("storage_format"), raw.get("storageFormat"));
    if (sf && !sf->is_null() && (!is_str(sf) || sf->s.empty()) && !is_u32_int(sf))
        fail("storage_format must be a non-empty string or non-negative integer", path + ".storage_format");
}

inline void parse_limit_violation(const Json& raw, const std::string& path) {
    if (!raw.is_obj()) fail("limit_violation item must be an object", path);
    if (!nonempty_str(raw.get("limit"))) fail("limit must be a non-empty string", path + ".limit");
    const Json* st = raw.get("stage");
    if (st && !st->is_null() && !normalize_stage_name(st, false)) fail("unsupported shader stage", path + ".stage");
    const Json* c = raw.get("count");
    if (!is_js_int(c) || c->n < 0) fail("count must be a non-negative integer", path + ".count");
    const Json* m = raw.get("max");
    if (!is_js_int(m) || m->n < 0) fail("max must be a non-negative integer", path + ".max");
}

inline void parse_array_with(const Json* arr, const char* what, const std::string& path, void (*fn)(const Json&, const std::string&)) {
    if (!arr) return;
    if (!arr->is_arr()) fail(std::string(what) + " must be an array", path);
    for (size_t i = 0; i < arr->a.size(); ++i) fn(arr->a[i], path + "[" + std::to_string(i) + "]");
}

inline void check_fragment_output(const Json& raw, const std::string& path) {
    if (!raw.is_obj()) fail("fragment_output must be an object", path);
    if (!is_u32_int(raw.get("location"))) fail("invalid location", path + ".location");
    if (!nonempty_str(raw.get("type"))) fail("type must be a non-empty string", path + ".type");
}
inline void check_lowering_need(const Json& raw, const std::string& path) {
    if (!raw.is_obj()) fail("requires_lowering item must be an object", path);
    if (!nonempty_str(raw.get("kind"))) fail("kind must be a non-empty string", path + ".kind");
    const Json* g = raw.get("group");
    if (g && !g->is_null() && !is_u32_int(g)) fail("group must be a non-negative integer or null", path + ".group");
    const Json* bd = raw.get("binding");
    if (bd && !bd->is_null() && !is_u32_int(bd)) fail("binding must be a non-negative integer or null", path + ".binding");
    if (!is_str(raw.get("detail"))) fail("detail must be a string", path + ".detail");
}

}  // namespace detail

inline ManifestTables parse_reflection_manifest(const Json& raw) {
    using namespace detail;
    ManifestTables out;
    if (!raw.is_obj()) fail("manifest root must be a JSON object");
    const Json* sv = raw.get("schema_version");
    if (!sv || sv->t != Json::T::Num || sv->n != kManifestSchemaVersion) fail("unsupported schema_version (expected 1)", "schema_version");

    // register_convention
    if (const Json* rc = raw.get("register_convention"); rc && !rc->is_null()) {
        if (!rc->is_obj()) fail("register_convention must be an object", "register_convention");
        const Json* src = rc->get("source");
        if (src && !nonempty_str(src)) fail("source must be a non-empty string", "register_convention.source");
        const Json* sh = rc->get("shifts");
        if (sh && !sh->is_obj()) fail("shifts must be an object", "register_convention.shifts");
        if (sh)
            for (const char* k : {"b", "t", "u", "s"})
                if (const Json* v = sh->get(k); v && !is_u32_int(v)) fail("shift must be a non-negative uint32 integer", std::string("register_convention.shifts.") + k);
    }

    std::set<uint64_t> seen_bgl;
    if (const Json* t = raw.get("bind_group_layouts")) {
        if (!t->is_obj()) fail("bind_group_layouts must be an object", "bind_group_layouts");
        for (const auto& kv : t->o) {
            const std::string bp = "bind_group_layouts[\"" + kv.first + "\"]";
            const auto id = parse_hash_string(kv.first);
            if (!id) fail("invalid 64-bit hex hash \"" + kv.first + "\"", bp);
            if (!seen_bgl.insert(*id).second) fail("duplicate bind group layout id " + format_hash(*id), bp);
            const Json* ents = kv.second.get("entries");
            if (!kv.second.is_obj() || !ents || !ents->is_arr()) fail("bind group layout must be an object with an \"entries\" array", bp);
            std::set<uint32_t> seen_bind;
            std::vector<LayoutEntryDesc> entries;
            for (size_t i = 0; i < ents->a.size(); ++i) {
                const std::string ep = bp + ".entries[" + std::to_string(i) + "]";
                LayoutEntryDesc e = normalize_layout_entry(ents->a[i], ep, std::nullopt);
                if (!seen_bind.insert(e.binding).second) fail("duplicate binding index " + std::to_string(e.binding), ep);
                entries.push_back(std::move(e));
            }
            std::stable_sort(entries.begin(), entries.end(), [](const LayoutEntryDesc& x, const LayoutEntryDesc& y) { return x.binding < y.binding; });
            out.bind_group_layouts[format_hash(*id)] = std::move(entries);
        }
    }

    std::set<uint64_t> seen_pl;
    const Json* pls = raw.get("pipeline_layouts");
    if (pls) {
        if (!pls->is_obj()) fail("pipeline_layouts must be an object", "pipeline_layouts");
        for (const auto& kv : pls->o) {
            const std::string pp = "pipeline_layouts[\"" + kv.first + "\"]";
            const auto id = parse_hash_string(kv.first);
            if (!id) fail("invalid 64-bit hex hash \"" + kv.first + "\"", pp);
            if (!seen_pl.insert(*id).second) fail("duplicate pipeline layout id " + format_hash(*id), pp);
            const Json* bgls = kv.second.get("bindGroupLayouts");
            if (!kv.second.is_obj() || !bgls || !bgls->is_arr()) fail("pipeline layout must be an object with a \"bindGroupLayouts\" array", pp);
            std::vector<std::string> ids;
            for (size_t g = 0; g < bgls->a.size(); ++g) {
                const std::string gp = pp + ".bindGroupLayouts[" + std::to_string(g) + "]";
                const std::string hex = format_hash(hash_from_json(bgls->a[g], gp));
                if (!out.bind_group_layouts.count(hex)) fail("references unknown bind_group_layout " + hex, gp);
                ids.push_back(hex);
            }
            out.pipeline_layouts[format_hash(*id)] = std::move(ids);
        }
    }

    const Json* shaders = raw.get("shaders");
    if (!shaders || !shaders->is_obj()) fail("shaders must be an object", "shaders");
    std::set<uint64_t> seen_hash;
    for (const auto& kv : shaders->o) {
        const std::string sp = "shaders[\"" + kv.first + "\"]";
        const auto hash = parse_hash_string(kv.first);
        if (!hash) fail("invalid 64-bit hex hash \"" + kv.first + "\"", sp);
        if (!seen_hash.insert(*hash).second) fail("duplicate shader hash " + format_hash(*hash), sp);
        const Json& rs = kv.second;
        if (!rs.is_obj()) fail("shader record must be an object", sp);
        auto stage = normalize_stage_name(rs.get("stage"), false);
        if (!stage) fail("unsupported shader stage (expected \"vertex\", \"fragment\", or \"compute\")", sp + ".stage");

        const Json* ep = first_defined(rs.get("entry_point"), rs.get("entryPoint"));
        if (!nonempty_str(ep)) fail("entry_point must be a non-empty string", sp + ".entry_point");
        const Json* nm = rs.get("name");
        if (nm && !nm->is_null() && !is_str(nm)) fail("name must be a string", sp + ".name");

        if (const Json* pl = rs.get("pipeline_layout"); pl && !pl->is_null()) {
            const std::string hex = format_hash(hash_from_json(*pl, sp + ".pipeline_layout"));
            if (pls && !out.pipeline_layouts.count(hex)) fail("references unknown pipeline_layout " + hex, sp + ".pipeline_layout");
        }

        auto rec = std::make_shared<StageRecord>();
        const Json* ids = rs.get("bind_group_layouts");
        if (ids) {
            if (!ids->is_arr()) fail("bind_group_layouts must be an array of layout ids", sp + ".bind_group_layouts");
            if (ids->a.size() > kMaxReflectedGroups) fail("bind_group_layouts length exceeds reflection limit 64", sp + ".bind_group_layouts");
            for (size_t g = 0; g < ids->a.size(); ++g) {
                const std::string gp = sp + ".bind_group_layouts[" + std::to_string(g) + "]";
                const std::string hex = format_hash(hash_from_json(ids->a[g], gp));
                auto it = out.bind_group_layouts.find(hex);
                if (it == out.bind_group_layouts.end()) fail("references unknown bind_group_layout " + hex, gp);
                rec->group_layouts.push_back(it->second);
            }
        }
        if (const Json* lc = rs.get("layout_complete")) {
            if (!is_bool(lc)) fail("layout_complete must be a boolean", sp + ".layout_complete");
            rec->layout_complete = lc->b;
        }
        if (const Json* wg = rs.get("workgroup_size"); wg && !wg->is_null()) {
            bool ok = wg->is_arr() && wg->a.size() == 3;
            if (ok)
                for (const Json& n : wg->a)
                    if (!is_u32_int(&n) || n.n < 1) ok = false;
            if (!ok) fail("workgroup_size must be null or a 3-element array of positive integers", sp + ".workgroup_size");
        }
        if (const Json* bs = rs.get("bindings")) {
            if (!bs->is_arr()) fail("bindings must be an array", sp + ".bindings");
            for (size_t i = 0; i < bs->a.size(); ++i) parse_binding_record(bs->a[i], sp + ".bindings[" + std::to_string(i) + "]");
        }
        const Json* vi = first_defined(rs.get("vertex_inputs"), rs.get("vertexInputs"));
        if (vi) {
            if (!vi->is_arr()) fail("vertex_inputs must be an array", sp + ".vertex_inputs");
            for (size_t i = 0; i < vi->a.size(); ++i) {
                const Json& v = vi->a[i];
                const std::string vp = sp + ".vertex_inputs[" + std::to_string(i) + "]";
                if (!v.is_obj()) fail("vertex_input must be an object", vp);
                if (!is_u32_int(v.get("location"))) fail("invalid location", vp + ".location");
                if (const Json* t = v.get("type"); t && !is_str(t)) fail("type must be a string", vp + ".type");
                const Json* f = v.get("format");
                if (f && !f->is_null() && !is_str(f)) fail("format must be a string or null", vp + ".format");
                VertexInputRec r;
                r.location = as_u32(v.get("location"));
                if (f && is_str(f)) r.format = f->s;
                rec->vertex_inputs.push_back(std::move(r));
            }
            std::stable_sort(rec->vertex_inputs.begin(), rec->vertex_inputs.end(),
                             [](const VertexInputRec& x, const VertexInputRec& y) { return x.location < y.location; });
        }
        if (const Json* fo = first_defined(rs.get("fragment_outputs"), rs.get("fragmentOutputs")))
            parse_array_with(fo, "fragment_outputs", sp + ".fragment_outputs", check_fragment_output);
        parse_array_with(rs.get("requires_lowering"), "requires_lowering", sp + ".requires_lowering", check_lowering_need);
        parse_array_with(rs.get("limit_violations"), "limit_violations", sp + ".limit_violations", parse_limit_violation);
        check_vertex_input_mode(rs, sp);

        rec->hash = *hash;
        rec->hash_hex = format_hash(*hash);
        rec->stage = *stage;
        rec->visibility = stage_visibility(*stage);
        rec->entry_point = ep->s;
        if (nonempty_str(nm)) rec->name = nm->s;
        out.shaders.push_back(std::move(rec));
    }

    if (const Json* ps = raw.get("pipelines")) {
        if (!ps->is_obj()) fail("pipelines must be an object", "pipelines");
        std::set<uint64_t> shader_hashes;
        for (const auto& s : out.shaders) shader_hashes.insert(s->hash);
        std::set<std::string> keys;
        for (const auto& kv : ps->o) {
            const std::string pp = "pipelines[\"" + kv.first + "\"]";
            const Json& rp = kv.second;
            if (!rp.is_obj()) fail("pipeline record must be an object", pp);
            const Json* st = rp.get("stages");
            if (!st || !st->is_obj()) fail("stages must be an object", pp + ".stages");
            std::vector<std::pair<std::pair<int, uint64_t>, std::string>> list;
            for (const auto& sk : st->o) {
                const std::string kp = pp + ".stages." + sk.first;
                const std::string l = js_lower(js_trim(sk.first));
                if (!(l == "vertex" || l == "fragment" || l == "compute")) fail("unsupported shader stage", kp);
                const uint64_t h = hash_from_json(sk.second, kp);
                if (!shader_hashes.count(h)) fail("references unknown shader " + format_hash(h), kp);
                list.push_back({{l == "vertex" ? 0 : l == "fragment" ? 1 : 2, h}, format_hash(h)});
            }
            if (const Json* pl = rp.get("pipeline_layout"); pl && !pl->is_null()) {
                const std::string hex = format_hash(hash_from_json(*pl, pp + ".pipeline_layout"));
                if (pls && !out.pipeline_layouts.count(hex)) fail("references unknown pipeline_layout " + hex, pp + ".pipeline_layout");
            }
            if (const Json* bg = rp.get("bind_group_layouts")) {
                if (!bg->is_arr()) fail("bind_group_layouts must be an array", pp + ".bind_group_layouts");
                for (size_t g = 0; g < bg->a.size(); ++g) {
                    const std::string gp = pp + ".bind_group_layouts[" + std::to_string(g) + "]";
                    const std::string hex = format_hash(hash_from_json(bg->a[g], gp));
                    if (!out.bind_group_layouts.count(hex)) fail("references unknown bind_group_layout " + hex, gp);
                }
            }
            if (const Json* lc = rp.get("layout_complete"); lc && !is_bool(lc)) fail("layout_complete must be a boolean", pp + ".layout_complete");
            parse_array_with(rp.get("limit_violations"), "limit_violations", pp + ".limit_violations", parse_limit_violation);
            std::sort(list.begin(), list.end());
            std::string key;
            for (const auto& e : list) key += (key.empty() ? "" : "+") + e.second;
            keys.insert(key);  // later pipelines with the same canonical key replace earlier ones
        }
        out.pipelines = keys.size();
    }
    return out;
}

// ---- merging and the layout / vertex builders ----

// mergeLayoutEntries; an empty optional + *reason on conflict.
inline std::optional<LayoutEntryDesc> merge_layout_entries(const LayoutEntryDesc& a, const LayoutEntryDesc& b, std::string* reason) {
    using K = LayoutEntryDesc::Kind;
    if (a.kind != b.kind) {
        *reason = "resource kind mismatch";
        return std::nullopt;
    }
    LayoutEntryDesc m = a;
    m.visibility = a.visibility | b.visibility;
    switch (a.kind) {
        case K::Buffer:
            if (a.buffer_type != b.buffer_type) {
                *reason = "buffer type mismatch";
                return std::nullopt;
            }
            if (a.has_dynamic_offset != b.has_dynamic_offset) {
                *reason = "buffer hasDynamicOffset mismatch";
                return std::nullopt;
            }
            m.min_binding_size = std::max(a.min_binding_size, b.min_binding_size);
            return m;
        case K::Sampler:
            if (a.sampler_type != b.sampler_type) {
                *reason = "sampler type mismatch";
                return std::nullopt;
            }
            return m;
        case K::Texture:
            if (a.view_dim != b.view_dim) {
                *reason = "texture viewDimension mismatch";
                return std::nullopt;
            }
            if (a.multisampled != b.multisampled) {
                *reason = "texture multisampled mismatch";
                return std::nullopt;
            }
            if (a.sample_type != b.sample_type) {
                const bool float_pair = (a.sample_type == 0 && b.sample_type == 1) || (a.sample_type == 1 && b.sample_type == 0);
                if (!float_pair) {
                    *reason = "texture sampleType mismatch";
                    return std::nullopt;
                }
                m.sample_type = 0;  // "float" wins over "unfilterable-float"
            }
            return m;
        case K::StorageTexture:
            if (a.access != b.access || a.storage_format != b.storage_format || a.view_dim != b.view_dim) {
                *reason = "storageTexture mismatch";
                return std::nullopt;
            }
            return m;
        case K::ExternalTexture: return m;
    }
    return m;
}

struct LayoutBuildResult {
    bool ok = false;
    bool conflict = false;  // a BindingConflictError (counted separately by the registry)
    std::string error;
    std::vector<std::vector<LayoutEntryDesc>> groups;  // GPUBindGroupLayoutDescriptor.entries per group, dense 0..max
};

// buildBindGroupLayoutDescriptors (ReflectionRegistry options: maxBindGroups 4, no requireLayoutComplete).
inline LayoutBuildResult build_bind_group_layout_descriptors(const std::vector<const StageRecord*>& refls, uint32_t max_bind_groups = kMaxWebGpuBindGroups) {
    LayoutBuildResult r;
    std::map<uint32_t, std::map<uint32_t, LayoutEntryDesc>> groups;
    int64_t max_group_seen = -1;
    for (const StageRecord* st : refls) {
        for (size_t g = 0; g < st->group_layouts.size(); ++g) {
            for (const LayoutEntryDesc& e : st->group_layouts[g]) {
                if (g >= max_bind_groups) {
                    r.error = "Bind group index " + std::to_string(g) + " exceeds WebGPU maxBindGroups limit (" + std::to_string(max_bind_groups) + ")";
                    return r;
                }
                max_group_seen = std::max<int64_t>(max_group_seen, static_cast<int64_t>(g));
                auto& gm = groups[static_cast<uint32_t>(g)];
                auto it = gm.find(e.binding);
                if (it == gm.end()) {
                    gm.emplace(e.binding, e);
                } else {
                    std::string reason;
                    auto merged = merge_layout_entries(it->second, e, &reason);
                    if (!merged) {
                        r.conflict = true;
                        r.error = "Bind group " + std::to_string(g) + " binding " + std::to_string(e.binding) + " conflict: " + reason;
                        return r;
                    }
                    it->second = std::move(*merged);
                }
            }
        }
    }
    r.groups.resize(static_cast<size_t>(max_group_seen + 1));
    for (auto& g : groups)
        for (auto& e : g.second) r.groups[g.first].push_back(std::move(e.second));  // std::map: ascending binding
    r.ok = true;
    return r;
}

struct VertexAttributeDesc {
    uint32_t location = 0;
    uint32_t offset = 0;
    uint8_t format = 0;  // GPU vertex format code
};
struct VertexBufferLayoutDesc {
    uint32_t array_stride = 0;
    std::vector<VertexAttributeDesc> attributes;
};

// buildVertexBufferLayouts(vsRefl, stride): one vertex-step buffer; offsets run on in location order.
inline bool build_vertex_buffer_layouts(const StageRecord& vs, uint32_t explicit_stride, std::vector<VertexBufferLayoutDesc>* out, std::string* error) {
    out->clear();
    if (vs.vertex_inputs.empty()) return true;
    struct In {
        uint32_t location;
        uint8_t format;
        uint32_t bytes;
    };
    std::vector<In> ins;
    for (const VertexInputRec& v : vs.vertex_inputs) {
        const VertexFormatName* f = v.format.empty() ? nullptr : vertex_format_by_name(v.format);
        if (!f) {
            *error = "Vertex input @location(" + std::to_string(v.location) + ") has unsupported or missing GPUVertexFormat";
            return false;
        }
        ins.push_back({v.location, f->code, f->bytes});
    }
    std::stable_sort(ins.begin(), ins.end(), [](const In& x, const In& y) { return x.location < y.location; });
    VertexBufferLayoutDesc vb;
    uint64_t running = 0, max_end = 0;
    std::set<uint32_t> seen;
    for (const In& in : ins) {
        if (!seen.insert(in.location).second) {
            *error = "Duplicate vertex input @location(" + std::to_string(in.location) + ")";
            return false;
        }
        const uint64_t offset = running;
        const uint64_t end = offset + in.bytes;
        max_end = std::max(max_end, end);
        running = end;
        vb.attributes.push_back({in.location, static_cast<uint32_t>(offset), in.format});
    }
    if (explicit_stride < max_end) {
        *error = "Explicit vertex stride (" + std::to_string(explicit_stride) + ") is smaller than required attribute extent (" + std::to_string(max_end) + " bytes)";
        return false;
    }
    vb.array_stride = explicit_stride;
    out->push_back(std::move(vb));
    return true;
}

// ---- the registry (ReflectionRegistry: loadManifest / getShaderReflection / getStats) ----

class ReflectionRegistry {
public:
    struct Stats {
        uint64_t manifests_loaded = 0, sidecars_loaded = 0, load_errors = 0;
        uint64_t lookup_hits = 0, lookup_misses = 0;
        uint64_t layouts_built = 0, pipeline_layouts_built = 0, vertex_layouts_built = 0, binding_conflicts = 0;
    };

    // loadManifest: a `playgta6-stage-reflection` document (without `schema_version`) is a sidecar,
    // everything else is parsed as a layout manifest. Returns "" on success, else the error.
    std::string load(std::string_view json_text) {
        try {
            Json root;
            parse_json(json_text, &root);
            if (root.is_obj() && root.get("schema") && is_str(root.get("schema")) && root.get("schema")->s == "playgta6-stage-reflection" &&
                !root.get("schema_version")) {
                auto stages = parse_stage_reflection_sidecar(root);
                for (auto& s : stages) register_record(s, false);
                ++stats_.manifests_loaded;
                ++stats_.sidecars_loaded;
                return std::string();
            }
            ManifestTables t = parse_reflection_manifest(root);
            for (auto& kv : t.bind_group_layouts) bgls_[kv.first] = std::move(kv.second);
            for (auto& kv : t.pipeline_layouts) pls_[kv.first] = std::move(kv.second);
            for (auto& s : t.shaders) register_record(s, true);
            ++stats_.manifests_loaded;
            return std::string();
        } catch (const ReflectError& e) {
            ++stats_.load_errors;
            return e.what();
        }
    }

    size_t shader_count() const { return by_hash_.size(); }
    size_t bind_group_layout_count() const { return bgls_.size(); }
    size_t pipeline_layout_count() const { return pls_.size(); }
    const Stats& stats() const { return stats_; }
    Stats& mutable_stats() { return stats_; }

    // getShaderReflection(bigint | hex string | name)
    const StageRecord* by_hash(uint64_t h) const {
        auto it = by_hash_.find(h);
        return it == by_hash_.end() ? nullptr : it->second.get();
    }
    // The worker's lookups by name: `"<stage>:<entryPoint>"` (manifest shaders) and shader names.
    const StageRecord* by_name(const std::string& n) const {
        auto it = by_name_.find(n);
        return it == by_name_.end() ? nullptr : it->second.get();
    }
    const StageRecord* lookup_hash(uint64_t h) { return count(by_hash(h)); }
    const StageRecord* lookup_name(const std::string& n) { return count(by_name(n)); }

    // Manifest tables for ManifestReference layouts: nullptr when not loaded.
    const std::vector<LayoutEntryDesc>* manifest_bind_group_layout(uint64_t id) const {
        auto it = bgls_.find(format_hash(id));
        return it == bgls_.end() ? nullptr : &it->second;
    }
    const std::vector<std::string>* manifest_pipeline_layout(uint64_t id) const {
        auto it = pls_.find(format_hash(id));
        return it == pls_.end() ? nullptr : &it->second;
    }
    const std::vector<LayoutEntryDesc>* manifest_bind_group_layout_hex(const std::string& hex) const {
        auto it = bgls_.find(hex);
        return it == bgls_.end() ? nullptr : &it->second;
    }

private:
    std::unordered_map<uint64_t, std::shared_ptr<StageRecord>> by_hash_;
    std::unordered_map<std::string, std::shared_ptr<StageRecord>> by_name_;
    std::unordered_map<std::string, std::vector<LayoutEntryDesc>> bgls_;
    std::unordered_map<std::string, std::vector<std::string>> pls_;
    Stats stats_;

    const StageRecord* count(const StageRecord* r) {
        r ? ++stats_.lookup_hits : ++stats_.lookup_misses;
        return r;
    }
    // _registerShaderRecord: a later record with the same hash replaces the earlier one; manifest
    // shaders are also indexed by entry point, "<stage>:<entry>" and name, sidecar stages by name only.
    void register_record(const std::shared_ptr<StageRecord>& s, bool index_entry_names) {
        by_hash_[s->hash] = s;
        if (index_entry_names) {
            by_name_[s->entry_point] = s;
            by_name_[s->stage + ":" + s->entry_point] = s;
            if (!s->name.empty() && s->name != s->entry_point) by_name_[s->name] = s;
        } else if (!s->name.empty()) {
            by_name_[s->name] = s;
        }
    }
};

}  // namespace playgta6::gpu::native_exec::reflect
