#include "gpu/rt_residency.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>

#include "gpu/gfx10_descriptors.h"
#include "gpu/rt_bvh.h"
#include "gpu/rt_json.h"

namespace playgta6::gpu::rt_residency {

bool parse_address_sources(const std::string& json, std::vector<AddressSource>* out, std::string* error) {
    out->clear();
    rt_json::Value root;
    std::string err;
    if (!rt_json::parse(json, &root, &err) || root.type != rt_json::Value::Type::Array) {
        if (error) *error = err.empty() ? "address_sources must be an array" : err;
        return false;
    }
    for (const rt_json::Value& e : root.arr) {
        AddressSource s;
        const rt_json::Value* kind = e.type == rt_json::Value::Type::Object ? e.get("kind") : nullptr;
        const rt_json::Value* index = e.type == rt_json::Value::Type::Object ? e.get("index") : nullptr;
        if (kind == nullptr || kind->type != rt_json::Value::Type::String || index == nullptr ||
            index->type != rt_json::Value::Type::Number) {
            if (error) *error = "an address source needs a string kind and a numeric index";
            return false;
        }
        if (kind->str == "root_param") s.kind = SourceKind::RootParam;
        else if (kind->str == "root_constants") s.kind = SourceKind::RootConstants;
        else if (kind->str == "user_data") s.kind = SourceKind::UserData;
        else {
            if (error) *error = "unknown address source kind '" + kind->str + "'";
            return false;
        }
        s.index = static_cast<uint32_t>(index->u64());
        if (const rt_json::Value* d = e.get("dword")) s.dword = static_cast<uint32_t>(d->u64());
        if (const rt_json::Value* x = e.get("extent")) s.extent = std::max<uint64_t>(x->u64(1), 1);
        if (const rt_json::Value* b = e.get("bvh_root")) s.bvh_root = b->type == rt_json::Value::Type::Bool && b->b;
        out->push_back(s);
    }
    return true;
}

namespace {
constexpr uint32_t kMaxDepth = 4;
constexpr uint32_t kInstanceChunk = 64;  // instances read per guest read (4 KiB)

class Walker {
public:
    Walker(const DispatchInputs& in, const rt_arena::Arena* arena, Plan* plan) : in_(in), arena_(arena), plan_(plan) {}

    void walk(uint64_t addr, uint32_t depth) {
        addr &= rt_bvh::kAddressMask;
        if (depth > kMaxDepth || !visited_.insert(addr).second) return;
        uint32_t h[16];
        if (!read(addr, h, sizeof h) || gpu_owned(addr, sizeof h)) {
            plan_->structures_unavailable = true;
            return;
        }
        const uint64_t size = static_cast<uint64_t>(h[1]) * rt_bvh::kNodeGranularityBytes;
        if ((h[0] & 0xFFFFu) != rt_bvh::kHeaderVersion || size < sizeof h || size > kMaxStructureBytes) {
            ++plan_->bad_headers;
            plan_->ranges.emplace_back(addr, sizeof h);
            return;
        }
        ++plan_->structures;
        plan_->ranges.emplace_back(addr, size);
        if ((h[0] & rt_bvh::kHeaderFlagBottomLevel) != 0) return;
        const uint32_t count = h[3];
        const uint64_t offset = h[13];
        if (count > kMaxInstances || offset + static_cast<uint64_t>(count) * 64u > size) {
            ++plan_->bad_headers;
            return;
        }
        std::vector<uint32_t> words;
        for (uint32_t first = 0; first < count; first += kInstanceChunk) {
            const uint32_t n = std::min(kInstanceChunk, count - first);
            const uint64_t at = addr + offset + static_cast<uint64_t>(first) * 64u;
            words.resize(static_cast<size_t>(n) * 16u);
            if (!read(at, words.data(), words.size() * 4) || gpu_owned(at, words.size() * 4)) {
                plan_->structures_unavailable = true;
                return;
            }
            for (uint32_t i = 0; i < n; ++i) {
                const rt_bvh::PackedInstanceView v = rt_bvh::unpack_instance_node(&words[i * 16]);
                if (v.blas_address != 0 && v.blas_root_node != rt_bvh::kNoChild) walk(v.blas_address, depth + 1);
            }
        }
    }

private:
    bool read(uint64_t addr, void* dst, size_t bytes) const { return in_.read_guest && in_.read_guest(addr, dst, bytes); }
    bool gpu_owned(uint64_t addr, uint64_t size) const {
        if (arena_ == nullptr || size == 0) return false;
        for (uint64_t p = rt_arena::page_of(addr); p <= rt_arena::page_of(addr + size - 1); ++p) {
            if (arena_->is_gpu_owned(p)) return true;
        }
        return false;
    }

    const DispatchInputs& in_;
    const rt_arena::Arena* arena_;
    Plan* plan_;
    std::set<uint64_t> visited_;
};

Unresolved resolve(const AddressSource& s, const DispatchInputs& in, uint64_t* addr) {
    switch (s.kind) {
    case SourceKind::RootParam:
        if (!in.root_param) return Unresolved::NoInput;
        return in.root_param(s.index, addr) ? Unresolved::Count : Unresolved::Unbound;
    case SourceKind::RootConstants: {
        uint32_t lo = 0, hi = 0;
        if (!in.root_constant) return Unresolved::NoInput;
        if (!in.root_constant(s.index, s.dword, &lo) || !in.root_constant(s.index, s.dword + 1, &hi)) {
            return Unresolved::Unbound;
        }
        *addr = (static_cast<uint64_t>(hi) << 32) | lo;
        return Unresolved::Count;
    }
    case SourceKind::UserData: {
        uint32_t lo = 0, hi = 0;
        if (!in.user_data) return Unresolved::NoInput;
        if (!in.user_data(s.index, &lo) || !in.user_data(s.index + 1, &hi)) {
            return in.user_data_failure ? in.user_data_failure(s.index) : Unresolved::Unbound;
        }
        *addr = (static_cast<uint64_t>(hi) << 32) | lo;
        return Unresolved::Count;
    }
    }
    return Unresolved::Unbound;
}
}  // namespace

const char* unresolved_name(Unresolved r) {
    switch (r) {
    case Unresolved::NoInput: return "no_input";
    case Unresolved::Unbound: return "unbound";
    case Unresolved::Null: return "null";
    case Unresolved::DescriptorTable: return "descriptor_table";
    case Unresolved::OutOfRange: return "out_of_range";
    default: return "?";
    }
}

const char* table_skip_name(TableSkip r) {
    switch (r) {
    case TableSkip::Zero: return "zero";
    case TableSkip::NotBuffer: return "not_buffer";
    case TableSkip::Unreadable: return "unreadable";
    case TableSkip::NullBase: return "null_base";
    case TableSkip::Truncated: return "truncated";
    default: return "?";
    }
}

namespace {
// Follows `t` into buffer descriptors; true when the table could be walked at all.
bool follow_table(const DescriptorTableRef& t, const AddressSource& s, const DispatchInputs& in, Walker& walker, Plan& plan) {
    if (!in.read_guest || t.address == 0 || t.stride < 32 || t.stride > kMaxTableSlotBytes) return false;
    ++plan.tables_followed;
    const uint32_t count = std::min(t.count, kMaxTableDescriptors);
    if (t.truncated || t.count > kMaxTableDescriptors) ++plan.table_skips[static_cast<size_t>(TableSkip::Truncated)];
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t w[kMaxTableSlotBytes / 4] = {};
        if (!in.read_guest(t.address + uint64_t{i} * t.stride, w, 32)) {
            ++plan.table_skips[static_cast<size_t>(TableSkip::Unreadable)];
            continue;
        }
        switch (gfx10::classify_slot(w)) {
        case gfx10::SlotContent::Zero: ++plan.table_skips[static_cast<size_t>(TableSkip::Zero)]; continue;
        case gfx10::SlotContent::Buffer: break;
        default: ++plan.table_skips[static_cast<size_t>(TableSkip::NotBuffer)]; continue;
        }
        const gfx10::BufferResource b = gfx10::decode_buffer(w);
        if (b.base == 0) {
            ++plan.table_skips[static_cast<size_t>(TableSkip::NullBase)];
            continue;
        }
        ++plan.table_buffers;
        if (s.bvh_root) {
            walker.walk(b.base, 0);
        } else {
            const uint64_t bytes = std::min<uint64_t>(uint64_t{b.num_records} * std::max<uint32_t>(b.stride, 1u), kMaxStructureBytes);
            plan.ranges.emplace_back(b.base, std::max<uint64_t>(bytes, 1));
        }
    }
    return true;
}
}  // namespace

Plan plan_dispatch(const std::vector<AddressSource>& sources, const DispatchInputs& in, const rt_arena::Arena* arena) {
    Plan plan;
    Walker walker(in, arena, &plan);
    for (const AddressSource& s : sources) {
        uint64_t addr = 0;
        Unresolved why = resolve(s, in, &addr);
        if (why == Unresolved::Count && addr == 0) why = Unresolved::Null;
        if (why == Unresolved::DescriptorTable && in.descriptor_table) {
            DescriptorTableRef table;
            if (in.descriptor_table(s.index, &table) && follow_table(table, s, in, walker, plan)) continue;
        }
        if (why != Unresolved::Count) {
            ++plan.unresolved;
            ++plan.unresolved_by_reason[static_cast<size_t>(why)];
            continue;
        }
        if (s.bvh_root) {
            walker.walk(addr, 0);
        } else {
            plan.ranges.emplace_back(addr, s.extent);
        }
    }
    return plan;
}

Outcome prepare_dispatch(rt_arena::Arena& arena, const std::vector<AddressSource>& sources, const DispatchInputs& in,
                         Plan* plan_out) {
    Plan plan = plan_dispatch(sources, in, &arena);
    Outcome out;
    const auto fail = [&](rt_arena::SkipReason r) {
        arena.note_skip(r);
        out.ok = false;
        out.reason = r;
    };
    if (plan.structures_unavailable) {
        fail(rt_arena::SkipReason::StructuresUnavailable);
    } else {
        for (const auto& [addr, size] : plan.ranges) {
            const rt_arena::Ensure e = arena.ensure_range(addr, size);
            if (e == rt_arena::Ensure::ArenaFull || e == rt_arena::Ensure::TableFull) {
                fail(rt_arena::SkipReason::ArenaFull);
                break;
            }
            if (e == rt_arena::Ensure::OutOfRange) {
                fail(rt_arena::SkipReason::NotResident);
                break;
            }
        }
    }
    if (plan_out) *plan_out = std::move(plan);
    return out;
}

bool FaultTracker::consume(const uint32_t* words, size_t count, rt_arena::Arena& arena) {
    if (words == nullptr || count < kStatsWords) return false;
    ++stats_.readbacks;
    const uint32_t delta = words[0] - last_count_;  // wraps like the GPU counter
    last_count_ = words[0];
    stats_.faults += delta;
    arena.set_fault_count(stats_.faults);
    if (delta == 0 && words[1] == 0) return false;

    const uint32_t attempts = words[1];
    const uint32_t logged = std::min(attempts, kFaultLogEntries);
    std::vector<uint64_t> seen;
    for (uint32_t i = 0; i < logged; ++i) {
        const uint32_t v = words[2 + i];
        if (v == 0) continue;  // slot reserved but not yet visible: counted as unlogged below
        const uint64_t page = static_cast<uint64_t>(v - 1);
        if (std::find(seen.begin(), seen.end(), page) != seen.end()) continue;
        seen.push_back(page);
        if (arena.is_resident(page)) {
            ++stats_.pages_already_resident;
            continue;
        }
        const rt_arena::Ensure e = arena.ensure_page(page);
        if (e == rt_arena::Ensure::Resident || e == rt_arena::Ensure::Uploaded) ++stats_.pages_serviced;
        else ++stats_.pages_unserviceable;
    }
    if (attempts > kFaultLogEntries) stats_.unlogged += attempts - kFaultLogEntries;
    return true;
}

std::string FaultTracker::json() const {
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "{\"rt_fault\": %llu, \"pages_serviced\": %llu, \"pages_already_resident\": %llu, "
                  "\"pages_unserviceable\": %llu, \"unlogged\": %llu, \"readbacks\": %llu}",
                  (unsigned long long)stats_.faults, (unsigned long long)stats_.pages_serviced,
                  (unsigned long long)stats_.pages_already_resident, (unsigned long long)stats_.pages_unserviceable,
                  (unsigned long long)stats_.unlogged, (unsigned long long)stats_.readbacks);
    return buf;
}

}  // namespace playgta6::gpu::rt_residency
