#include "gpu/prewarm_pipeline_desc.h"

#include <cstring>
#include <type_traits>

namespace playgta6::gpu {

namespace {

constexpr uint32_t kMaxGroups = 16;
constexpr uint32_t kMaxBindingsPerGroup = 4096;

static_assert(std::is_trivially_copyable_v<GpuBindGroupLayoutEntry>);
static_assert(std::is_trivially_copyable_v<d3d12_state::SamplerDesc>);

class Writer {
public:
    explicit Writer(std::vector<uint8_t>& out) : out_(out) {}
    void u8(uint32_t v) { out_.push_back(static_cast<uint8_t>(v)); }
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) out_.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) out_.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
    void raw(const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        out_.insert(out_.end(), b, b + n);
    }

private:
    std::vector<uint8_t>& out_;
};

class Reader {
public:
    Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
    bool u8(uint32_t* v) {
        if (n_ - pos_ < 1) return false;
        *v = p_[pos_++];
        return true;
    }
    bool u32(uint32_t* v) {
        if (n_ - pos_ < 4) return false;
        *v = uint32_t{p_[pos_]} | uint32_t{p_[pos_ + 1]} << 8 | uint32_t{p_[pos_ + 2]} << 16 |
             uint32_t{p_[pos_ + 3]} << 24;
        pos_ += 4;
        return true;
    }
    bool u64(uint64_t* v) {
        uint32_t lo = 0;
        uint32_t hi = 0;
        if (!u32(&lo) || !u32(&hi)) return false;
        *v = uint64_t{lo} | uint64_t{hi} << 32;
        return true;
    }
    bool raw(void* p, size_t n) {
        if (n_ - pos_ < n) return false;
        std::memcpy(p, p_ + pos_, n);
        pos_ += n;
        return true;
    }
    size_t remaining() const { return n_ - pos_; }

private:
    const uint8_t* p_;
    size_t n_;
    size_t pos_ = 0;
};

void write_binding(Writer& w, const bridge::LayoutBinding& b) {
    w.raw(&b.entry, sizeof(b.entry));
    const d3d12_state::ShaderBinding& j = b.join;
    w.u32(j.binding);
    w.u32(static_cast<uint32_t>(j.register_class));
    w.u32(j.shader_register);
    w.u32(j.register_space);
    w.u8(j.root_constants ? 1 : 0);
    w.u8(static_cast<uint32_t>(j.heap));
    w.u32(j.heap_index_parameter);
    w.u32(j.heap_index_dword);
    w.u32(j.heap_shift);
    w.u8(static_cast<uint32_t>(j.heap_index_source));
    w.u32(j.cb_table_slot);
    w.u32(j.cb_byte_offset);
    w.u32(j.cb_bit_offset);
    w.u32(j.cb_mask);
    w.u32(j.heap_stride);
    w.u32(j.array_size);
    w.u32(j.array_index);
    w.u32(j.table_parameter);
    w.u32(j.table_slot);
    w.u8(j.inline_sampler ? 1 : 0);
    w.raw(&j.sampler, sizeof(j.sampler));
}

bool read_binding(Reader& r, bridge::LayoutBinding* b) {
    namespace d3d = d3d12_state;
    d3d::ShaderBinding& j = b->join;
    uint32_t v = 0;
    if (!r.raw(&b->entry, sizeof(b->entry))) return false;
    if (!r.u32(&j.binding)) return false;
    if (!r.u32(&v)) return false;
    j.register_class = static_cast<d3d::DescriptorRangeType>(v);
    if (!r.u32(&j.shader_register) || !r.u32(&j.register_space)) return false;
    if (!r.u8(&v)) return false;
    j.root_constants = v != 0;
    if (!r.u8(&v)) return false;
    j.heap = static_cast<d3d::HeapAccess>(v);
    if (!r.u32(&j.heap_index_parameter) || !r.u32(&j.heap_index_dword) || !r.u32(&j.heap_shift)) return false;
    if (!r.u8(&v)) return false;
    j.heap_index_source = static_cast<d3d::HeapIndexSource>(v);
    if (!r.u32(&j.cb_table_slot) || !r.u32(&j.cb_byte_offset) || !r.u32(&j.cb_bit_offset) || !r.u32(&j.cb_mask) ||
        !r.u32(&j.heap_stride) || !r.u32(&j.array_size) || !r.u32(&j.array_index) || !r.u32(&j.table_parameter) ||
        !r.u32(&j.table_slot)) {
        return false;
    }
    if (!r.u8(&v)) return false;
    j.inline_sampler = v != 0;
    return r.raw(&j.sampler, sizeof(j.sampler));
}

}  // namespace

std::vector<uint8_t> encode_prewarm_pipeline_desc(const PrewarmPipelineDesc& d) {
    std::vector<uint8_t> out;
    out.reserve(512);
    Writer w(out);
    w.u32(kPrewarmDescVersion);
    w.u32(d.kind);
    w.u64(d.pack_key0);
    w.u64(d.pack_key1);
    if (d.kind == kPrewarmKindCompute) {
        w.u32(sizeof(d.compute));
        w.raw(&d.compute, sizeof(d.compute));
    } else {
        w.u32(sizeof(d.graphics));
        w.raw(&d.graphics, sizeof(d.graphics));
    }
    w.u32(d.has_layout ? 1 : 0);
    if (d.has_layout) {
        w.u64(d.layout.content_id);
        w.u32(static_cast<uint32_t>(d.layout.groups.size()));
        for (const bridge::GroupLayoutDesc& g : d.layout.groups) {
            w.u64(g.content_id);
            w.u32(static_cast<uint32_t>(g.bindings.size()));
            for (const bridge::LayoutBinding& b : g.bindings) write_binding(w, b);
        }
    }
    return out;
}

bool decode_prewarm_pipeline_desc(const uint8_t* data, size_t size, PrewarmPipelineDesc* out) {
    if (data == nullptr || out == nullptr) return false;
    Reader r(data, size);
    PrewarmPipelineDesc d;
    uint32_t version = 0;
    uint32_t key_bytes = 0;
    uint32_t has_layout = 0;
    if (!r.u32(&version) || version != kPrewarmDescVersion) return false;
    if (!r.u32(&d.kind) || d.kind > kPrewarmKindCompute) return false;
    if (!r.u64(&d.pack_key0) || !r.u64(&d.pack_key1) || !r.u32(&key_bytes)) return false;
    if (d.kind == kPrewarmKindCompute) {
        if (key_bytes != sizeof(d.compute) || !r.raw(&d.compute, sizeof(d.compute))) return false;
    } else {
        if (key_bytes != sizeof(d.graphics) || !r.raw(&d.graphics, sizeof(d.graphics))) return false;
    }
    if (!r.u32(&has_layout) || has_layout > 1) return false;
    d.has_layout = has_layout != 0;
    if (d.has_layout) {
        uint32_t groups = 0;
        if (!r.u64(&d.layout.content_id) || !r.u32(&groups) || groups > kMaxGroups) return false;
        d.layout.groups.resize(groups);
        for (bridge::GroupLayoutDesc& g : d.layout.groups) {
            uint32_t bindings = 0;
            if (!r.u64(&g.content_id) || !r.u32(&bindings) || bindings > kMaxBindingsPerGroup) return false;
            // A binding takes well over 64 bytes: refuse a count the blob cannot hold.
            if (bindings > r.remaining() / 64) return false;
            g.bindings.resize(bindings);
            for (bridge::LayoutBinding& b : g.bindings) {
                if (!read_binding(r, &b)) return false;
            }
        }
    }
    if (r.remaining() != 0) return false;
    *out = std::move(d);
    return true;
}

}  // namespace playgta6::gpu
