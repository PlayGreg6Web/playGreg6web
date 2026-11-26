#include "gpu/descriptor_heap.h"

#include <algorithm>
#include <bit>
#include <iterator>
#include <cstring>
#include <limits>

namespace playgta6::gpu::d3d12_state {

namespace {

constexpr uint32_t kCpuTag = 0x40000000u;
constexpr uint32_t kGpuTag = 0x80000000u;
constexpr uint32_t kTagMask = 0xc0000000u;
constexpr uint32_t kSlotMask = 0xffffu;
constexpr uint32_t kSlotGenerationMask = 0x3fffu;

struct Fnv64 {
    uint64_t h = 0xcbf29ce484222325ull;
    void byte(uint8_t b) noexcept {
        h ^= b;
        h *= 0x100000001b3ull;
    }
    void u32(uint32_t v) noexcept {
        for (int i = 0; i < 4; ++i) {
            byte(static_cast<uint8_t>(v >> (8 * i)));
        }
    }
    void u64(uint64_t v) noexcept {
        u32(static_cast<uint32_t>(v));
        u32(static_cast<uint32_t>(v >> 32));
    }
    void f32(float v) noexcept { u32(std::bit_cast<uint32_t>(v)); }
};

bool is_sampler_heap(DescriptorHeapType type) noexcept {
    return type == DescriptorHeapType::Sampler;
}

bool valid_heap_type(DescriptorHeapType type) noexcept {
    return static_cast<uint32_t>(type) < kDescriptorHeapTypeCount;
}

uint32_t increment_shift(uint32_t inc) noexcept {
    return inc == 16u ? 4u : 5u;
}

uint16_t clamp_u16(uint32_t v) noexcept {
    return static_cast<uint16_t>(std::min<uint32_t>(v, 0xffffu));
}

}  // namespace

const char* status_name(Status status) noexcept {
    switch (status) {
    case Status::Ok: return "Ok";
    case Status::InvalidArgument: return "InvalidArgument";
    case Status::InvalidHandle: return "InvalidHandle";
    case Status::StaleHandle: return "StaleHandle";
    case Status::HandleOutOfRange: return "HandleOutOfRange";
    case Status::MisalignedHandle: return "MisalignedHandle";
    case Status::WrongHeapType: return "WrongHeapType";
    case Status::HeapNotShaderVisible: return "HeapNotShaderVisible";
    case Status::HeapNotBound: return "HeapNotBound";
    case Status::TooManyHeaps: return "TooManyHeaps";
    case Status::CountMismatch: return "CountMismatch";
    case Status::InvalidRootParameterIndex: return "InvalidRootParameterIndex";
    case Status::RootParameterTypeMismatch: return "RootParameterTypeMismatch";
    case Status::RootSignatureNotSet: return "RootSignatureNotSet";
    case Status::RootArgumentNotSet: return "RootArgumentNotSet";
    case Status::ConstantsOutOfRange: return "ConstantsOutOfRange";
    case Status::UnsupportedVersion: return "UnsupportedVersion";
    case Status::InvalidParameterType: return "InvalidParameterType";
    case Status::InvalidShaderVisibility: return "InvalidShaderVisibility";
    case Status::InvalidRangeType: return "InvalidRangeType";
    case Status::EmptyDescriptorTable: return "EmptyDescriptorTable";
    case Status::ZeroDescriptorRange: return "ZeroDescriptorRange";
    case Status::MixedSamplerTable: return "MixedSamplerTable";
    case Status::AppendAfterUnboundedRange: return "AppendAfterUnboundedRange";
    case Status::DescriptorOffsetOverflow: return "DescriptorOffsetOverflow";
    case Status::RegisterOverlap: return "RegisterOverlap";
    case Status::RootSignatureTooLarge: return "RootSignatureTooLarge";
    case Status::TooManyStaticSamplers: return "TooManyStaticSamplers";
    case Status::BlobTruncated: return "BlobTruncated";
    case Status::BlobBadMagic: return "BlobBadMagic";
    case Status::BlobMissingRootSignaturePart: return "BlobMissingRootSignaturePart";
    case Status::BlobOffsetOutOfBounds: return "BlobOffsetOutOfBounds";
    case Status::UnknownResource: return "UnknownResource";
    case Status::ResourceAlreadyRegistered: return "ResourceAlreadyRegistered";
    }
    return "Unknown";
}

// ===========================================================================
// DescriptorHeap
// ===========================================================================

DescriptorHeap::DescriptorHeap(const DescriptorHeapDesc& desc, uint32_t serial)
    : desc_(desc), serial_(serial) {
    const uint32_t n = desc.num_descriptors;
    if (is_sampler_heap(desc.type)) {
        samplers_.resize(n);
    } else {
        views_.resize(n);
    }
    slot_generations_ = std::make_unique<uint64_t[]>(n);
    const uint32_t blocks = (n + kGenerationBlockSize - 1) / kGenerationBlockSize;
    block_generations_ = std::make_unique<std::atomic<uint64_t>[]>(blocks);
    for (uint32_t b = 0; b < blocks; ++b) {
        block_generations_[b].store(0, std::memory_order_relaxed);
    }
}

CpuDescriptorHandle DescriptorHeap::cpu_start() const noexcept {
    return cpu_handle(0);
}

GpuDescriptorHandle DescriptorHeap::gpu_start() const noexcept {
    return gpu_handle(0);
}

CpuDescriptorHandle DescriptorHeap::cpu_handle(uint32_t index) const noexcept {
    if (desc_.cpu_address != 0) {
        return CpuDescriptorHandle{desc_.cpu_address + static_cast<uint64_t>(index) * increment()};
    }
    return CpuDescriptorHandle{
        (static_cast<uint64_t>(kCpuTag | serial_) << 32) |
        (static_cast<uint64_t>(index) * increment())};
}

GpuDescriptorHandle DescriptorHeap::gpu_handle(uint32_t index) const noexcept {
    if (!shader_visible()) {
        return GpuDescriptorHandle{0};
    }
    return GpuDescriptorHandle{
        (static_cast<uint64_t>(kGpuTag | serial_) << 32) |
        (static_cast<uint64_t>(index) * increment())};
}

const ViewDescriptor* DescriptorHeap::view(uint32_t index) const noexcept {
    return index < views_.size() ? &views_[index] : nullptr;
}

const SamplerDescriptor* DescriptorHeap::sampler(uint32_t index) const noexcept {
    return index < samplers_.size() ? &samplers_[index] : nullptr;
}

uint64_t DescriptorHeap::slot_generation(uint32_t index) const noexcept {
    return index < capacity() ? slot_generations_[index] : 0;
}

void DescriptorHeap::stamp(uint32_t index) noexcept {
    const uint64_t gen = generation_.fetch_add(1, std::memory_order_acq_rel) + 1;
    slot_generations_[index] = gen;
    std::atomic<uint64_t>& block = block_generations_[index / kGenerationBlockSize];
    uint64_t current = block.load(std::memory_order_relaxed);
    while (current < gen &&
           !block.compare_exchange_weak(current, gen, std::memory_order_release, std::memory_order_relaxed)) {
    }
}

bool DescriptorHeap::write_view(uint32_t index, const ViewDescriptor& record) noexcept {
    if (index >= views_.size()) {
        return false;
    }
    if (std::memcmp(&views_[index], &record, sizeof(ViewDescriptor)) == 0) {
        return false;
    }
    views_[index] = record;
    stamp(index);
    return true;
}

bool DescriptorHeap::write_sampler(uint32_t index, const SamplerDescriptor& record) noexcept {
    if (index >= samplers_.size()) {
        return false;
    }
    if (std::memcmp(&samplers_[index], &record, sizeof(SamplerDescriptor)) == 0) {
        return false;
    }
    samplers_[index] = record;
    stamp(index);
    return true;
}

bool DescriptorHeap::range_changed_since(uint32_t first, uint32_t count, uint64_t since) const noexcept {
    const uint32_t cap = capacity();
    if (count == 0 || first >= cap) {
        return false;
    }
    if (generation() <= since) {
        return false;
    }
    const uint32_t last = first + std::min(count, cap - first);  // exclusive
    const uint32_t b0 = first / kGenerationBlockSize;
    const uint32_t b1 = (last - 1) / kGenerationBlockSize;
    for (uint32_t b = b0; b <= b1; ++b) {
        if (block_generations_[b].load(std::memory_order_acquire) <= since) {
            continue;
        }
        const uint32_t block_lo = b * kGenerationBlockSize;
        const uint32_t block_hi = block_lo + kGenerationBlockSize;
        const uint32_t lo = std::max(first, block_lo);
        const uint32_t hi = std::min(last, block_hi);
        if (lo == block_lo && hi == block_hi) {
            return true;
        }
        for (uint32_t i = lo; i < hi; ++i) {
            if (slot_generations_[i] > since) {
                return true;
            }
        }
    }
    return false;
}

// ===========================================================================
// DescriptorHeapRegistry
// ===========================================================================

namespace {
std::atomic<uint64_t> g_live_heaps{0};
std::atomic<uint64_t> g_live_heap_slots{0};
std::atomic<uint64_t> g_live_cached_bind_groups{0};
std::atomic<uint64_t> g_peak_bind_groups_epoch{0};
std::atomic<uint64_t> g_bind_group_slot_budget{0};
std::atomic<uint64_t> g_bind_group_budget_overflows{0};

void raise_peak(uint64_t value) noexcept {
    uint64_t cur = g_peak_bind_groups_epoch.load(std::memory_order_relaxed);
    while (value > cur && !g_peak_bind_groups_epoch.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
    }
}
}  // namespace

void set_bind_group_slot_budget(uint64_t slots) noexcept {
    g_bind_group_slot_budget.store(slots, std::memory_order_relaxed);
}

void reset_bind_group_slot_counters() noexcept {
    g_peak_bind_groups_epoch.store(0, std::memory_order_relaxed);
    g_bind_group_budget_overflows.store(0, std::memory_order_relaxed);
}

DescriptorLiveCounts descriptor_live_counts() noexcept {
    DescriptorLiveCounts c;
    c.heaps = g_live_heaps.load(std::memory_order_relaxed);
    c.heap_slots = g_live_heap_slots.load(std::memory_order_relaxed);
    c.cached_bind_groups = g_live_cached_bind_groups.load(std::memory_order_relaxed);
    c.peak_bind_groups_per_epoch = g_peak_bind_groups_epoch.load(std::memory_order_relaxed);
    c.bind_group_slot_budget = g_bind_group_slot_budget.load(std::memory_order_relaxed);
    c.bind_group_budget_overflows = g_bind_group_budget_overflows.load(std::memory_order_relaxed);
    return c;
}

DescriptorHeapRegistry::DescriptorHeapRegistry()
    : slots_(std::make_unique<std::atomic<DescriptorHeap*>[]>(kMaxDescriptorHeaps)),
      slot_generation_(kMaxDescriptorHeaps, 0) {
    for (uint32_t i = 0; i < kMaxDescriptorHeaps; ++i) {
        slots_[i].store(nullptr, std::memory_order_relaxed);
    }
}

DescriptorHeapRegistry::~DescriptorHeapRegistry() {
    for (uint32_t i = 0; i < next_unused_slot_; ++i) {
        DescriptorHeap* heap = slots_[i].exchange(nullptr, std::memory_order_acq_rel);
        if (heap != nullptr) {
            g_live_heaps.fetch_sub(1, std::memory_order_relaxed);
            g_live_heap_slots.fetch_sub(heap->capacity(), std::memory_order_relaxed);
        }
        delete heap;
    }
}

Status DescriptorHeapRegistry::create_heap(const DescriptorHeapDesc& desc, DescriptorHeap** out_heap) {
    if (out_heap == nullptr) {
        return Status::InvalidArgument;
    }
    *out_heap = nullptr;
    if (!valid_heap_type(desc.type) || desc.num_descriptors == 0 ||
        desc.num_descriptors > kMaxDescriptorsPerHeap ||
        (desc.flags & ~DescriptorHeapFlags::ShaderVisible) != 0) {
        return Status::InvalidArgument;
    }
    const bool shader_visible = (desc.flags & DescriptorHeapFlags::ShaderVisible) != 0;
    if (shader_visible && (desc.type == DescriptorHeapType::Rtv || desc.type == DescriptorHeapType::Dsv)) {
        return Status::InvalidArgument;
    }
    if (shader_visible && desc.type == DescriptorHeapType::Sampler &&
        desc.num_descriptors > kMaxShaderVisibleSamplers) {
        return Status::InvalidArgument;
    }

    const uint32_t inc = descriptor_handle_increment_size(desc.type);
    const uint64_t placed_bytes = static_cast<uint64_t>(desc.num_descriptors) * inc;
    if (desc.cpu_address != 0) {
        // CPU handles are plain addresses: below the tag bits, aligned, not overlapping another
        // placed heap.
        if ((desc.cpu_address >> 62) != 0 || desc.cpu_address > UINT64_MAX - placed_bytes ||
            ((desc.cpu_address + placed_bytes) >> 62) != 0 || (desc.cpu_address & 7u) != 0) {
            return Status::InvalidArgument;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (desc.cpu_address != 0) {
        const uint64_t end = desc.cpu_address + placed_bytes;
        std::shared_lock<std::shared_mutex> placed_lock(placed_mutex_);
        auto next = std::lower_bound(
            placed_.begin(), placed_.end(), desc.cpu_address,
            [](const PlacedHeapRange& r, uint64_t addr) { return r.start < addr; });
        if (next != placed_.end() && next->start < end) {
            return Status::InvalidArgument;
        }
        if (next != placed_.begin() && std::prev(next)->end > desc.cpu_address) {
            return Status::InvalidArgument;
        }
    }
    uint32_t slot = 0;
    if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
    } else if (next_unused_slot_ < kMaxDescriptorHeaps) {
        slot = next_unused_slot_++;
    } else {
        return Status::TooManyHeaps;
    }
    const uint32_t serial = (static_cast<uint32_t>(slot_generation_[slot]) << 16) | slot;
    auto* heap = new DescriptorHeap(desc, serial);
    owned_.emplace(heap, slot);
    slots_[slot].store(heap, std::memory_order_release);
    if (desc.cpu_address != 0) {
        const uint64_t end = desc.cpu_address + placed_bytes;
        const uint32_t shift = increment_shift(inc);
        std::unique_lock<std::shared_mutex> placed_lock(placed_mutex_);
        auto pos = std::lower_bound(
            placed_.begin(), placed_.end(), desc.cpu_address,
            [](const PlacedHeapRange& r, uint64_t addr) { return r.start < addr; });
        placed_.insert(pos, PlacedHeapRange{desc.cpu_address, end, heap, inc - 1u, shift});
    }
    live_heaps_.fetch_add(1, std::memory_order_relaxed);
    g_live_heaps.fetch_add(1, std::memory_order_relaxed);
    g_live_heap_slots.fetch_add(desc.num_descriptors, std::memory_order_relaxed);
    *out_heap = heap;
    return Status::Ok;
}

void DescriptorHeapRegistry::destroy_heap(DescriptorHeap* heap) {
    if (heap == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = owned_.find(heap);
    if (it == owned_.end()) {
        return;  // not ours / already destroyed (never dereferenced)
    }
    const uint32_t slot = it->second;
    owned_.erase(it);
    if (heap->desc().cpu_address != 0) {
        const uint64_t addr = heap->desc().cpu_address;
        std::unique_lock<std::shared_mutex> placed_lock(placed_mutex_);
        auto pit = std::lower_bound(
            placed_.begin(), placed_.end(), addr,
            [](const PlacedHeapRange& r, uint64_t a) { return r.start < a; });
        if (pit != placed_.end() && pit->start == addr) {
            placed_.erase(pit);
        }
    }
    slots_[slot].store(nullptr, std::memory_order_release);
    slot_generation_[slot] = static_cast<uint16_t>((slot_generation_[slot] + 1) & kSlotGenerationMask);
    free_slots_.push_back(slot);
    live_heaps_.fetch_sub(1, std::memory_order_relaxed);
    g_live_heaps.fetch_sub(1, std::memory_order_relaxed);
    g_live_heap_slots.fetch_sub(heap->capacity(), std::memory_order_relaxed);
    delete heap;
}

uint32_t DescriptorHeapRegistry::heap_count() const noexcept {
    return live_heaps_.load(std::memory_order_relaxed);
}

Status DescriptorHeapRegistry::resolve(uint64_t ptr, uint32_t tag, ResolvedDescriptor* out) const noexcept {
    const uint32_t hi = static_cast<uint32_t>(ptr >> 32);
    if (tag == kCpuTag && (hi & kTagMask) == 0 && ptr != 0) {
        // A CPU handle of a heap placed at an address.
        std::shared_lock<std::shared_mutex> placed_lock(placed_mutex_);
        const size_t n = placed_.size();
        const PlacedHeapRange* found = nullptr;
        if (n <= 8) {
            for (size_t i = 0; i < n; ++i) {
                if (ptr < placed_[i].start) {
                    break;
                }
                if (ptr < placed_[i].end) {
                    found = &placed_[i];
                    break;
                }
            }
        } else {
            auto it = std::upper_bound(
                placed_.begin(), placed_.end(), ptr,
                [](uint64_t addr, const PlacedHeapRange& r) { return addr < r.start; });
            if (it != placed_.begin()) {
                const PlacedHeapRange& cand = *std::prev(it);
                if (ptr < cand.end) {
                    found = &cand;
                }
            }
        }
        if (found == nullptr) {
            return Status::InvalidHandle;
        }
        const uint64_t offset = ptr - found->start;
        if ((offset & found->inc_mask) != 0) {
            return Status::MisalignedHandle;
        }
        if (out != nullptr) {
            out->heap = found->heap;
            out->index = static_cast<uint32_t>(offset >> found->inc_shift);
        }
        return Status::Ok;
    }
    if ((hi & kTagMask) != tag) {
        return Status::InvalidHandle;
    }
    const uint32_t serial = hi & ~kTagMask;
    const DescriptorHeap* heap = slots_[serial & kSlotMask].load(std::memory_order_acquire);
    if (heap == nullptr || heap->serial() != serial) {
        return Status::StaleHandle;
    }
    if (tag == kGpuTag && !heap->shader_visible()) {
        return Status::InvalidHandle;
    }
    const uint32_t offset = static_cast<uint32_t>(ptr);
    const uint32_t inc = heap->increment();
    if ((offset & (inc - 1u)) != 0) {
        return Status::MisalignedHandle;
    }
    const uint32_t index = offset >> increment_shift(inc);
    if (index >= heap->capacity()) {
        return Status::HandleOutOfRange;
    }
    if (out != nullptr) {
        out->heap = const_cast<DescriptorHeap*>(heap);
        out->index = index;
    }
    return Status::Ok;
}

Status DescriptorHeapRegistry::resolve_cpu(CpuDescriptorHandle handle, ResolvedDescriptor* out) const noexcept {
    return resolve(handle.ptr, kCpuTag, out);
}

Status DescriptorHeapRegistry::resolve_gpu(GpuDescriptorHandle handle, ResolvedDescriptor* out) const noexcept {
    return resolve(handle.ptr, kGpuTag, out);
}

Status DescriptorHeapRegistry::resolve_range(
    CpuDescriptorHandle start,
    uint32_t count,
    DescriptorHeapType type,
    ResolvedDescriptor* out) const noexcept {
    ResolvedDescriptor r;
    const Status s = resolve(start.ptr, kCpuTag, &r);
    if (s != Status::Ok) {
        return s;
    }
    if (r.heap->type() != type) {
        return Status::WrongHeapType;
    }
    if (static_cast<uint64_t>(r.index) + count > r.heap->capacity()) {
        return Status::HandleOutOfRange;
    }
    *out = r;
    return Status::Ok;
}

Status DescriptorHeapRegistry::write_view_at(CpuDescriptorHandle dst, const ViewDescriptor& record) noexcept {
    DescriptorHeapType type = DescriptorHeapType::CbvSrvUav;
    if (record.kind == DescriptorKind::Rtv) {
        type = DescriptorHeapType::Rtv;
    } else if (record.kind == DescriptorKind::Dsv) {
        type = DescriptorHeapType::Dsv;
    }
    ResolvedDescriptor r;
    const Status s = resolve_range(dst, 1, type, &r);
    if (s != Status::Ok) {
        return s;
    }
    r.heap->write_view(r.index, record);
    return Status::Ok;
}

namespace {

ViewDescriptor make_view_record(DescriptorKind kind, uint32_t resource_id, uint32_t counter_id, const ViewDesc* desc) {
    ViewDescriptor r{};
    r.kind = kind;
    r.resource_id = resource_id;
    r.counter_resource_id = counter_id;
    if (desc == nullptr) {
        r.dimension = static_cast<uint8_t>(ViewDimension::Unknown);
        r.component_mapping = kind == DescriptorKind::Srv ? kDefaultShader4ComponentMapping : 0;
        r.mip_levels = kind == DescriptorKind::Srv ? 0xffffu : 1u;
        r.array_size = 0xffffffffu;
        return r;
    }
    r.format = desc->format;
    r.dimension = static_cast<uint8_t>(desc->dimension);
    r.flags = static_cast<uint16_t>(desc->flags);
    r.first_element_or_address = desc->first_element;
    r.num_elements_or_size = desc->num_elements;
    r.structure_byte_stride = desc->structure_byte_stride;
    r.counter_offset_in_bytes = kind == DescriptorKind::Uav ? desc->counter_offset_in_bytes : 0;
    r.component_mapping = kind == DescriptorKind::Srv ? desc->component_mapping : 0;
    r.first_mip = clamp_u16(desc->first_mip);
    r.mip_levels = kind == DescriptorKind::Srv ? clamp_u16(desc->mip_levels) : 1u;
    r.first_array_slice = desc->first_array_slice;
    r.array_size = desc->array_size;
    r.plane_slice = desc->plane_slice;
    r.min_lod_clamp = kind == DescriptorKind::Srv ? desc->min_lod_clamp : 0.0f;
    return r;
}

bool valid_view_desc(DescriptorKind kind, const ViewDesc* desc, uint32_t resource_id) noexcept {
    if (desc == nullptr) {
        return resource_id != 0;  // null descriptors need a desc to know their dimension
    }
    if (desc->flags > 0xffffu) {
        return false;
    }
    switch (kind) {
    case DescriptorKind::Srv:
        return desc->dimension <= ViewDimension::TextureCubeArray + 1;  // +1: RT acceleration structure
    case DescriptorKind::Uav:
    case DescriptorKind::Rtv:
        return desc->dimension <= ViewDimension::Texture3D;
    case DescriptorKind::Dsv:
        return desc->dimension <= DsvDimension::Texture2DMsArray;
    default:
        return false;
    }
}

}  // namespace

Status DescriptorHeapRegistry::create_constant_buffer_view(
    const ConstantBufferViewDesc* desc,
    CpuDescriptorHandle dst) noexcept {
    ViewDescriptor r{};
    r.kind = DescriptorKind::Cbv;
    if (desc != nullptr) {
        if ((desc->size_in_bytes % cbv_alignment_) != 0 || (desc->buffer_location % cbv_alignment_) != 0) {
            return Status::InvalidArgument;
        }
        r.first_element_or_address = desc->buffer_location;
        r.num_elements_or_size = desc->size_in_bytes;
    }
    return write_view_at(dst, r);
}

Status DescriptorHeapRegistry::create_shader_resource_view(
    uint32_t resource_id,
    const ViewDesc* desc,
    CpuDescriptorHandle dst) noexcept {
    if (!valid_view_desc(DescriptorKind::Srv, desc, resource_id)) {
        return Status::InvalidArgument;
    }
    return write_view_at(dst, make_view_record(DescriptorKind::Srv, resource_id, 0, desc));
}

Status DescriptorHeapRegistry::create_unordered_access_view(
    uint32_t resource_id,
    uint32_t counter_resource_id,
    const ViewDesc* desc,
    CpuDescriptorHandle dst) noexcept {
    if (!valid_view_desc(DescriptorKind::Uav, desc, resource_id)) {
        return Status::InvalidArgument;
    }
    return write_view_at(dst, make_view_record(DescriptorKind::Uav, resource_id, counter_resource_id, desc));
}

Status DescriptorHeapRegistry::create_render_target_view(
    uint32_t resource_id,
    const ViewDesc* desc,
    CpuDescriptorHandle dst) noexcept {
    if (!valid_view_desc(DescriptorKind::Rtv, desc, resource_id)) {
        return Status::InvalidArgument;
    }
    return write_view_at(dst, make_view_record(DescriptorKind::Rtv, resource_id, 0, desc));
}

Status DescriptorHeapRegistry::create_depth_stencil_view(
    uint32_t resource_id,
    const ViewDesc* desc,
    CpuDescriptorHandle dst) noexcept {
    if (!valid_view_desc(DescriptorKind::Dsv, desc, resource_id)) {
        return Status::InvalidArgument;
    }
    return write_view_at(dst, make_view_record(DescriptorKind::Dsv, resource_id, 0, desc));
}

Status DescriptorHeapRegistry::create_sampler(const SamplerDesc& desc, CpuDescriptorHandle dst) noexcept {
    ResolvedDescriptor r;
    const Status s = resolve_range(dst, 1, DescriptorHeapType::Sampler, &r);
    if (s != Status::Ok) {
        return s;
    }
    SamplerDescriptor record{};
    record.kind = DescriptorKind::Sampler;
    record.desc = desc;
    r.heap->write_sampler(r.index, record);
    return Status::Ok;
}

namespace {

void copy_one(DescriptorHeap* dst, uint32_t di, const DescriptorHeap* src, uint32_t si) noexcept {
    if (is_sampler_heap(dst->type())) {
        const SamplerDescriptor record = *src->sampler(si);
        dst->write_sampler(di, record);
    } else {
        const ViewDescriptor record = *src->view(si);
        dst->write_view(di, record);
    }
}

struct CopyScratch {
    std::vector<ResolvedDescriptor> dst;
    std::vector<ResolvedDescriptor> src;
    std::vector<ViewDescriptor> views;
    std::vector<SamplerDescriptor> samplers;
};

CopyScratch& copy_scratch() {
    thread_local CopyScratch scratch;
    return scratch;
}

}  // namespace

Status DescriptorHeapRegistry::copy_descriptors_simple(
    uint32_t num_descriptors,
    CpuDescriptorHandle dst_start,
    CpuDescriptorHandle src_start,
    DescriptorHeapType type) noexcept {
    if (!valid_heap_type(type)) {
        return Status::InvalidArgument;
    }
    if (num_descriptors == 0) {
        return Status::Ok;
    }
    ResolvedDescriptor dst;
    ResolvedDescriptor src;
    Status s = resolve_range(dst_start, num_descriptors, type, &dst);
    if (s != Status::Ok) {
        return s;
    }
    s = resolve_range(src_start, num_descriptors, type, &src);
    if (s != Status::Ok) {
        return s;
    }
    if (dst.heap == src.heap && dst.index == src.index) {
        return Status::Ok;
    }
    if (dst.heap == src.heap && dst.index > src.index) {
        // memmove semantics: copy backwards when the destination is ahead.
        for (uint32_t i = num_descriptors; i-- > 0;) {
            copy_one(dst.heap, dst.index + i, src.heap, src.index + i);
        }
    } else {
        for (uint32_t i = 0; i < num_descriptors; ++i) {
            copy_one(dst.heap, dst.index + i, src.heap, src.index + i);
        }
    }
    return Status::Ok;
}

Status DescriptorHeapRegistry::copy_descriptors(
    uint32_t num_dst_ranges,
    const CpuDescriptorHandle* dst_range_starts,
    const uint32_t* dst_range_sizes,
    uint32_t num_src_ranges,
    const CpuDescriptorHandle* src_range_starts,
    const uint32_t* src_range_sizes,
    DescriptorHeapType type) {
    if (!valid_heap_type(type) || (num_dst_ranges > 0 && dst_range_starts == nullptr) ||
        (num_src_ranges > 0 && src_range_starts == nullptr)) {
        return Status::InvalidArgument;
    }
    auto size_of = [](const uint32_t* sizes, uint32_t i) { return sizes == nullptr ? 1u : sizes[i]; };

    uint64_t dst_total = 0;
    uint64_t src_total = 0;
    for (uint32_t i = 0; i < num_dst_ranges; ++i) {
        dst_total += size_of(dst_range_sizes, i);
    }
    for (uint32_t i = 0; i < num_src_ranges; ++i) {
        src_total += size_of(src_range_sizes, i);
    }
    if (dst_total != src_total) {
        return Status::CountMismatch;
    }
    if (dst_total == 0) {
        return Status::Ok;
    }

    CopyScratch& scratch = copy_scratch();
    scratch.dst.resize(num_dst_ranges);
    scratch.src.resize(num_src_ranges);
    for (uint32_t i = 0; i < num_dst_ranges; ++i) {
        const uint32_t n = size_of(dst_range_sizes, i);
        if (n == 0) {
            continue;
        }
        const Status s = resolve_range(dst_range_starts[i], n, type, &scratch.dst[i]);
        if (s != Status::Ok) {
            return s;
        }
    }
    for (uint32_t i = 0; i < num_src_ranges; ++i) {
        const uint32_t n = size_of(src_range_sizes, i);
        if (n == 0) {
            continue;
        }
        const Status s = resolve_range(src_range_starts[i], n, type, &scratch.src[i]);
        if (s != Status::Ok) {
            return s;
        }
    }

    bool overlap = false;
    for (uint32_t d = 0; d < num_dst_ranges && !overlap; ++d) {
        const uint32_t dn = size_of(dst_range_sizes, d);
        if (dn == 0) {
            continue;
        }
        for (uint32_t s = 0; s < num_src_ranges; ++s) {
            const uint32_t sn = size_of(src_range_sizes, s);
            if (sn == 0 || scratch.src[s].heap != scratch.dst[d].heap) {
                continue;
            }
            const uint64_t d0 = scratch.dst[d].index;
            const uint64_t s0 = scratch.src[s].index;
            if (d0 < s0 + sn && s0 < d0 + dn) {
                overlap = true;
                break;
            }
        }
    }

    const bool samplers = is_sampler_heap(type);
    if (overlap) {
        // Stage every source descriptor, then scatter.
        scratch.views.clear();
        scratch.samplers.clear();
        for (uint32_t s = 0; s < num_src_ranges; ++s) {
            const uint32_t sn = size_of(src_range_sizes, s);
            for (uint32_t k = 0; k < sn; ++k) {
                if (samplers) {
                    scratch.samplers.push_back(*scratch.src[s].heap->sampler(scratch.src[s].index + k));
                } else {
                    scratch.views.push_back(*scratch.src[s].heap->view(scratch.src[s].index + k));
                }
            }
        }
        size_t cursor = 0;
        for (uint32_t d = 0; d < num_dst_ranges; ++d) {
            const uint32_t dn = size_of(dst_range_sizes, d);
            for (uint32_t k = 0; k < dn; ++k, ++cursor) {
                if (samplers) {
                    scratch.dst[d].heap->write_sampler(scratch.dst[d].index + k, scratch.samplers[cursor]);
                } else {
                    scratch.dst[d].heap->write_view(scratch.dst[d].index + k, scratch.views[cursor]);
                }
            }
        }
        return Status::Ok;
    }

    uint32_t s = 0;
    uint32_t s_off = 0;
    for (uint32_t d = 0; d < num_dst_ranges; ++d) {
        const uint32_t dn = size_of(dst_range_sizes, d);
        for (uint32_t k = 0; k < dn; ++k) {
            while (s_off >= size_of(src_range_sizes, s)) {
                ++s;
                s_off = 0;
            }
            copy_one(scratch.dst[d].heap, scratch.dst[d].index + k, scratch.src[s].heap, scratch.src[s].index + s_off);
            ++s_off;
        }
    }
    return Status::Ok;
}

// ===========================================================================
// Root signatures
// ===========================================================================

uint32_t webgpu_stage_mask(ShaderVisibility visibility) noexcept {
    switch (visibility) {
    case ShaderVisibility::All:
        return WebGpuStage::Vertex | WebGpuStage::Fragment | WebGpuStage::Compute;
    case ShaderVisibility::Vertex:
        return WebGpuStage::Vertex;
    case ShaderVisibility::Pixel:
        return WebGpuStage::Fragment;
    case ShaderVisibility::Amplification:
    case ShaderVisibility::Mesh:
        return WebGpuStage::Compute;
    case ShaderVisibility::Hull:
    case ShaderVisibility::Domain:
    case ShaderVisibility::Geometry:
        return 0;
    }
    return 0;
}

const BindingSlot* BindingLayout::find(
    DescriptorRangeType register_class,
    uint32_t register_space,
    uint32_t shader_register,
    uint32_t stage_mask,
    uint32_t* out_table_offset) const noexcept {
    const bool compute = (stage_mask & WebGpuStage::Compute) != 0;
    for (const BindingSlot& slot : slots) {
        if (slot.kind != register_class || slot.register_space != register_space ||
            shader_register < slot.base_register) {
            continue;
        }
        const uint32_t rel = shader_register - slot.base_register;
        if (!slot.unbounded && rel >= slot.descriptor_count) {
            continue;
        }
        if (!compute && (webgpu_stage_mask(slot.visibility) & stage_mask) == 0) {
            continue;
        }
        if (out_table_offset != nullptr) {
            *out_table_offset = slot.source == SlotSource::DescriptorTable
                                    ? static_cast<uint32_t>(std::min<uint64_t>(
                                          static_cast<uint64_t>(slot.table_offset) + rel, 0xffffffffull))
                                    : 0;
        }
        return &slot;
    }
    return nullptr;
}

namespace {

bool visibilities_overlap(ShaderVisibility a, ShaderVisibility b) noexcept {
    return a == ShaderVisibility::All || b == ShaderVisibility::All || a == b;
}

struct RegisterInterval {
    uint32_t cls;
    uint32_t space;
    uint64_t lo;
    uint64_t hi;  // inclusive
    ShaderVisibility visibility;
    uint32_t order;
    uint32_t parameter_index;
    uint32_t range_index;
};

uint32_t range_class(DescriptorRangeType type) noexcept {
    return static_cast<uint32_t>(type);
}

uint32_t root_view_class(RootParameterType type) noexcept {
    switch (type) {
    case RootParameterType::Srv:
        return range_class(DescriptorRangeType::Srv);
    case RootParameterType::Uav:
        return range_class(DescriptorRangeType::Uav);
    default:
        return range_class(DescriptorRangeType::Cbv);
    }
}

}  // namespace

RootSignatureError RootSignature::build(const RootSignatureDesc& input, RootSignature* out) {
    RootSignatureError err;
    if (out == nullptr) {
        err.status = Status::InvalidArgument;
        return err;
    }
    RootSignature rs;
    rs.desc_ = input;
    RootSignatureDesc& desc = rs.desc_;

    const uint32_t version = static_cast<uint32_t>(desc.version);
    if (version < 1 || version > 3) {
        err.status = Status::UnsupportedVersion;
        return err;
    }
    if (desc.parameters.size() > kMaxRootParameters) {
        err.status = Status::RootSignatureTooLarge;
        return err;
    }
    if (desc.static_samplers.size() > kMaxStaticSamplers) {
        err.status = Status::TooManyStaticSamplers;
        return err;
    }
    if (desc.version == RootSignatureVersion::V1_0) {
        // 1.0 semantics expressed in 1.1 flags: everything volatile.
        for (RootParameterDesc& p : desc.parameters) {
            for (DescriptorRangeDesc& r : p.ranges) {
                r.flags = r.range_type == DescriptorRangeType::Sampler
                              ? DescriptorRangeFlags::DescriptorsVolatile
                              : DescriptorRangeFlags::DescriptorsVolatile | DescriptorRangeFlags::DataVolatile;
            }
            if (p.parameter_type != RootParameterType::DescriptorTable &&
                p.parameter_type != RootParameterType::Constants32Bit) {
                p.flags = RootDescriptorFlags::DataVolatile;
            }
        }
    }
    if (desc.version != RootSignatureVersion::V1_2) {
        for (StaticSamplerDesc& s : desc.static_samplers) {
            s.flags = 0;
        }
    }

    const uint32_t param_count = static_cast<uint32_t>(desc.parameters.size());
    rs.params_.resize(param_count);
    std::vector<std::vector<uint32_t>> resolved_offsets(param_count);
    std::vector<RegisterInterval> intervals;
    uint32_t cost = 0;
    uint32_t constants = 0;
    uint32_t order = 0;

    for (uint32_t i = 0; i < param_count; ++i) {
        const RootParameterDesc& p = desc.parameters[i];
        ParameterInfo& info = rs.params_[i];
        info.type = p.parameter_type;
        if (static_cast<uint32_t>(p.shader_visibility) > static_cast<uint32_t>(ShaderVisibility::Mesh)) {
            err.status = Status::InvalidShaderVisibility;
            err.parameter_index = i;
            return err;
        }
        switch (p.parameter_type) {
        case RootParameterType::DescriptorTable: {
            if (p.ranges.empty()) {
                err.status = Status::EmptyDescriptorTable;
                err.parameter_index = i;
                return err;
            }
            const bool sampler_table = p.ranges[0].range_type == DescriptorRangeType::Sampler;
            info.table_heap_type = sampler_table ? DescriptorHeapType::Sampler : DescriptorHeapType::CbvSrvUav;
            uint64_t cursor = 0;
            uint64_t extent = 0;
            bool unbounded_seen = false;
            resolved_offsets[i].resize(p.ranges.size());
            for (uint32_t j = 0; j < p.ranges.size(); ++j) {
                const DescriptorRangeDesc& r = p.ranges[j];
                if (static_cast<uint32_t>(r.range_type) > static_cast<uint32_t>(DescriptorRangeType::Sampler)) {
                    err.status = Status::InvalidRangeType;
                } else if ((r.range_type == DescriptorRangeType::Sampler) != sampler_table) {
                    err.status = Status::MixedSamplerTable;
                } else if (r.num_descriptors == 0) {
                    err.status = Status::ZeroDescriptorRange;
                } else if (r.offset_in_descriptors_from_table_start == kDescriptorRangeOffsetAppend && unbounded_seen) {
                    err.status = Status::AppendAfterUnboundedRange;
                }
                if (err.status != Status::Ok) {
                    err.parameter_index = i;
                    err.range_index = j;
                    return err;
                }
                const uint64_t offset = r.offset_in_descriptors_from_table_start == kDescriptorRangeOffsetAppend
                                            ? cursor
                                            : r.offset_in_descriptors_from_table_start;
                const bool unbounded = r.num_descriptors == kUnboundedDescriptorCount;
                const uint64_t end = offset + (unbounded ? 0 : r.num_descriptors);
                const uint64_t reg_hi = unbounded ? 0xffffffffull
                                                  : static_cast<uint64_t>(r.base_shader_register) + r.num_descriptors - 1;
                if (end >= 0xffffffffull || reg_hi > 0xffffffffull) {
                    err.status = Status::DescriptorOffsetOverflow;
                    err.parameter_index = i;
                    err.range_index = j;
                    return err;
                }
                resolved_offsets[i][j] = static_cast<uint32_t>(offset);
                if (unbounded) {
                    unbounded_seen = true;
                    info.table_unbounded = true;
                }
                cursor = end;
                extent = std::max(extent, end);
                intervals.push_back(RegisterInterval{
                    range_class(r.range_type), r.register_space, r.base_shader_register, reg_hi,
                    p.shader_visibility, order++, i, j});
            }
            info.table_extent = static_cast<uint32_t>(extent);
            cost += 1;
            break;
        }
        case RootParameterType::Constants32Bit:
            if (p.num_32bit_values == 0) {
                err.status = Status::InvalidArgument;
                err.parameter_index = i;
                return err;
            }
            info.constants_offset = constants;
            info.num_32bit_values = p.num_32bit_values;
            constants += std::min<uint32_t>(p.num_32bit_values, kMaxRootSignatureDwords + 1);
            cost += std::min<uint32_t>(p.num_32bit_values, kMaxRootSignatureDwords + 1);
            intervals.push_back(RegisterInterval{
                range_class(DescriptorRangeType::Cbv), p.register_space, p.shader_register, p.shader_register,
                p.shader_visibility, order++, i, kNoIndex});
            break;
        case RootParameterType::Cbv:
        case RootParameterType::Srv:
        case RootParameterType::Uav:
            cost += 2;
            intervals.push_back(RegisterInterval{
                root_view_class(p.parameter_type), p.register_space, p.shader_register, p.shader_register,
                p.shader_visibility, order++, i, kNoIndex});
            break;
        default:
            err.status = Status::InvalidParameterType;
            err.parameter_index = i;
            return err;
        }
        if (cost > kMaxRootSignatureDwords) {
            err.status = Status::RootSignatureTooLarge;
            err.parameter_index = i;
            return err;
        }
    }

    for (uint32_t k = 0; k < desc.static_samplers.size(); ++k) {
        const StaticSamplerDesc& s = desc.static_samplers[k];
        if (static_cast<uint32_t>(s.shader_visibility) > static_cast<uint32_t>(ShaderVisibility::Mesh)) {
            err.status = Status::InvalidShaderVisibility;
            err.parameter_index = k;
            return err;
        }
        intervals.push_back(RegisterInterval{
            range_class(DescriptorRangeType::Sampler), s.register_space, s.shader_register, s.shader_register,
            s.shader_visibility, order++, kNoIndex, k});
    }

    // Register overlap check: sort by (class, space, lo) and sweep.
    std::sort(intervals.begin(), intervals.end(), [](const RegisterInterval& a, const RegisterInterval& b) {
        if (a.cls != b.cls) return a.cls < b.cls;
        if (a.space != b.space) return a.space < b.space;
        if (a.lo != b.lo) return a.lo < b.lo;
        return a.order < b.order;
    });
    for (size_t a = 0; a < intervals.size(); ++a) {
        for (size_t b = a + 1; b < intervals.size(); ++b) {
            const RegisterInterval& x = intervals[a];
            const RegisterInterval& y = intervals[b];
            if (y.cls != x.cls || y.space != x.space || y.lo > x.hi) {
                break;
            }
            if (visibilities_overlap(x.visibility, y.visibility)) {
                const RegisterInterval& later = x.order > y.order ? x : y;
                err.status = Status::RegisterOverlap;
                err.parameter_index = later.parameter_index;
                err.range_index = later.range_index;
                return err;
            }
        }
    }

    // Flatten into binding slots.
    BindingLayout& layout = rs.layout_;
    auto add_slot = [&](const BindingSlot& slot) { layout.slots.push_back(slot); };
    for (uint32_t i = 0; i < param_count; ++i) {
        const RootParameterDesc& p = desc.parameters[i];
        ParameterInfo& info = rs.params_[i];
        info.first_slot = static_cast<uint32_t>(layout.slots.size());
        if (p.parameter_type == RootParameterType::DescriptorTable) {
            for (uint32_t j = 0; j < p.ranges.size(); ++j) {
                const DescriptorRangeDesc& r = p.ranges[j];
                BindingSlot slot;
                slot.kind = r.range_type;
                slot.source = SlotSource::DescriptorTable;
                slot.root_parameter_index = i;
                slot.range_index = j;
                slot.register_space = r.register_space;
                slot.base_register = r.base_shader_register;
                slot.descriptor_count = r.num_descriptors;
                slot.table_offset = resolved_offsets[i][j];
                slot.visibility = p.shader_visibility;
                slot.unbounded = r.num_descriptors == kUnboundedDescriptorCount;
                layout.has_unbounded_ranges = layout.has_unbounded_ranges || slot.unbounded;
                add_slot(slot);
            }
        } else {
            BindingSlot slot;
            slot.kind = p.parameter_type == RootParameterType::Srv   ? DescriptorRangeType::Srv
                        : p.parameter_type == RootParameterType::Uav ? DescriptorRangeType::Uav
                                                                     : DescriptorRangeType::Cbv;
            slot.source = p.parameter_type == RootParameterType::Constants32Bit ? SlotSource::RootConstants
                                                                                 : SlotSource::RootDescriptor;
            slot.root_parameter_index = i;
            slot.register_space = p.register_space;
            slot.base_register = p.shader_register;
            slot.descriptor_count = 1;
            slot.visibility = p.shader_visibility;
            slot.num_32bit_values = p.parameter_type == RootParameterType::Constants32Bit ? p.num_32bit_values : 0;
            add_slot(slot);
        }
        info.slot_count = static_cast<uint32_t>(layout.slots.size()) - info.first_slot;
    }
    for (uint32_t k = 0; k < desc.static_samplers.size(); ++k) {
        const StaticSamplerDesc& s = desc.static_samplers[k];
        BindingSlot slot;
        slot.kind = DescriptorRangeType::Sampler;
        slot.source = SlotSource::StaticSampler;
        slot.root_parameter_index = k;
        slot.register_space = s.register_space;
        slot.base_register = s.shader_register;
        slot.descriptor_count = 1;
        slot.visibility = s.shader_visibility;
        add_slot(slot);
    }

    // Stable content hash over the normalized description.
    Fnv64 h;
    h.u32(0x30535452u);  // "RTS0"
    h.u32(desc.flags);
    h.u32(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
        const RootParameterDesc& p = desc.parameters[i];
        h.u32(static_cast<uint32_t>(p.parameter_type));
        h.u32(static_cast<uint32_t>(p.shader_visibility));
        if (p.parameter_type == RootParameterType::DescriptorTable) {
            h.u32(static_cast<uint32_t>(p.ranges.size()));
            for (uint32_t j = 0; j < p.ranges.size(); ++j) {
                const DescriptorRangeDesc& r = p.ranges[j];
                h.u32(static_cast<uint32_t>(r.range_type));
                h.u32(r.num_descriptors);
                h.u32(r.base_shader_register);
                h.u32(r.register_space);
                h.u32(r.flags);
                h.u32(resolved_offsets[i][j]);
            }
        } else {
            h.u32(p.shader_register);
            h.u32(p.register_space);
            h.u32(p.parameter_type == RootParameterType::Constants32Bit ? p.num_32bit_values : p.flags);
        }
    }
    h.u32(static_cast<uint32_t>(desc.static_samplers.size()));
    for (const StaticSamplerDesc& s : desc.static_samplers) {
        h.u32(s.filter);
        h.u32(s.address_u);
        h.u32(s.address_v);
        h.u32(s.address_w);
        h.f32(s.mip_lod_bias);
        h.u32(s.max_anisotropy);
        h.u32(s.comparison_func);
        h.u32(s.border_color);
        h.f32(s.min_lod);
        h.f32(s.max_lod);
        h.u32(s.shader_register);
        h.u32(s.register_space);
        h.u32(static_cast<uint32_t>(s.shader_visibility));
        h.u32(s.flags);
    }
    rs.hash_ = h.h;
    rs.cost_dwords_ = cost;
    rs.total_constant_dwords_ = constants;
    *out = std::move(rs);
    return err;
}

// ---------------------------------------------------------------------------
// RTS0 blob parsing / serialization
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kFourccDxbc = 0x43425844u;  // "DXBC"
constexpr uint32_t kFourccRts0 = 0x30535452u;  // "RTS0"
constexpr uint32_t kRts0HeaderBytes = 24;
constexpr uint32_t kRootParameterBytes = 12;
constexpr uint32_t kStaticSamplerBytes = 52;
constexpr uint32_t kStaticSamplerBytesV12 = 56;

class BlobReader {
public:
    BlobReader(const uint8_t* data, size_t size, uint64_t base) : data_(data), size_(size), base_(base) {}

    bool u32(uint64_t offset, uint32_t* out) const noexcept {
        if (offset > size_ || size_ - offset < 4) {
            return false;
        }
        const uint8_t* p = data_ + offset;
        *out = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
        return true;
    }
    bool f32(uint64_t offset, float* out) const noexcept {
        uint32_t bits = 0;
        if (!u32(offset, &bits)) {
            return false;
        }
        *out = std::bit_cast<float>(bits);
        return true;
    }
    bool span_ok(uint64_t offset, uint64_t count, uint64_t stride) const noexcept {
        if (offset > size_) {
            return false;
        }
        if (stride != 0 && count > (size_ - offset) / stride) {
            return false;
        }
        return true;
    }
    uint64_t absolute(uint64_t offset) const noexcept { return base_ + offset; }

private:
    const uint8_t* data_;
    size_t size_;
    uint64_t base_;
};

RootSignatureError blob_error(Status status, uint64_t offset, uint32_t param = kNoIndex, uint32_t range = kNoIndex) {
    RootSignatureError e;
    e.status = status;
    e.byte_offset = offset;
    e.parameter_index = param;
    e.range_index = range;
    return e;
}

RootSignatureError parse_rts0(const BlobReader& in, RootSignatureDesc* out) {
    uint32_t header[6] = {};
    for (uint32_t i = 0; i < 6; ++i) {
        if (!in.u32(i * 4ull, &header[i])) {
            return blob_error(Status::BlobTruncated, in.absolute(i * 4ull));
        }
    }
    const uint32_t version = header[0];
    if (version < 1 || version > 3) {
        return blob_error(Status::UnsupportedVersion, in.absolute(0));
    }
    const uint32_t num_params = header[1];
    const uint32_t params_offset = header[2];
    const uint32_t num_samplers = header[3];
    const uint32_t samplers_offset = header[4];
    const bool v10 = version == 1;
    const bool v12 = version == 3;

    RootSignatureDesc desc;
    desc.version = static_cast<RootSignatureVersion>(version);
    desc.flags = header[5];
    if (!in.span_ok(params_offset, num_params, kRootParameterBytes)) {
        return blob_error(Status::BlobOffsetOutOfBounds, in.absolute(8));
    }
    desc.parameters.resize(num_params);
    for (uint32_t i = 0; i < num_params; ++i) {
        const uint64_t at = static_cast<uint64_t>(params_offset) + static_cast<uint64_t>(i) * kRootParameterBytes;
        uint32_t type = 0;
        uint32_t visibility = 0;
        uint32_t payload = 0;
        in.u32(at, &type);
        in.u32(at + 4, &visibility);
        in.u32(at + 8, &payload);
        RootParameterDesc& p = desc.parameters[i];
        p.parameter_type = static_cast<RootParameterType>(type);
        p.shader_visibility = static_cast<ShaderVisibility>(visibility);
        switch (p.parameter_type) {
        case RootParameterType::DescriptorTable: {
            uint32_t num_ranges = 0;
            uint32_t ranges_offset = 0;
            if (!in.u32(payload, &num_ranges) || !in.u32(payload + 4ull, &ranges_offset)) {
                return blob_error(Status::BlobOffsetOutOfBounds, in.absolute(at + 8), i);
            }
            const uint32_t stride = v10 ? 20u : 24u;
            if (!in.span_ok(ranges_offset, num_ranges, stride)) {
                return blob_error(Status::BlobOffsetOutOfBounds, in.absolute(payload + 4ull), i);
            }
            p.ranges.resize(num_ranges);
            for (uint32_t j = 0; j < num_ranges; ++j) {
                const uint64_t r = static_cast<uint64_t>(ranges_offset) + static_cast<uint64_t>(j) * stride;
                DescriptorRangeDesc& range = p.ranges[j];
                uint32_t range_type = 0;
                in.u32(r, &range_type);
                range.range_type = static_cast<DescriptorRangeType>(range_type);
                in.u32(r + 4, &range.num_descriptors);
                in.u32(r + 8, &range.base_shader_register);
                in.u32(r + 12, &range.register_space);
                if (v10) {
                    range.flags = 0;
                    in.u32(r + 16, &range.offset_in_descriptors_from_table_start);
                } else {
                    in.u32(r + 16, &range.flags);
                    in.u32(r + 20, &range.offset_in_descriptors_from_table_start);
                }
            }
            break;
        }
        case RootParameterType::Constants32Bit:
            if (!in.span_ok(payload, 1, 12)) {
                return blob_error(Status::BlobOffsetOutOfBounds, in.absolute(at + 8), i);
            }
            in.u32(payload, &p.shader_register);
            in.u32(payload + 4ull, &p.register_space);
            in.u32(payload + 8ull, &p.num_32bit_values);
            break;
        case RootParameterType::Cbv:
        case RootParameterType::Srv:
        case RootParameterType::Uav: {
            const uint32_t bytes = v10 ? 8u : 12u;
            if (!in.span_ok(payload, 1, bytes)) {
                return blob_error(Status::BlobOffsetOutOfBounds, in.absolute(at + 8), i);
            }
            in.u32(payload, &p.shader_register);
            in.u32(payload + 4ull, &p.register_space);
            if (!v10) {
                in.u32(payload + 8ull, &p.flags);
            }
            break;
        }
        default:
            return blob_error(Status::InvalidParameterType, in.absolute(at), i);
        }
    }

    const uint32_t sampler_stride = v12 ? kStaticSamplerBytesV12 : kStaticSamplerBytes;
    if (!in.span_ok(samplers_offset, num_samplers, sampler_stride)) {
        return blob_error(Status::BlobOffsetOutOfBounds, in.absolute(16));
    }
    desc.static_samplers.resize(num_samplers);
    for (uint32_t k = 0; k < num_samplers; ++k) {
        const uint64_t s = static_cast<uint64_t>(samplers_offset) + static_cast<uint64_t>(k) * sampler_stride;
        StaticSamplerDesc& d = desc.static_samplers[k];
        uint32_t visibility = 0;
        in.u32(s + 0, &d.filter);
        in.u32(s + 4, &d.address_u);
        in.u32(s + 8, &d.address_v);
        in.u32(s + 12, &d.address_w);
        in.f32(s + 16, &d.mip_lod_bias);
        in.u32(s + 20, &d.max_anisotropy);
        in.u32(s + 24, &d.comparison_func);
        in.u32(s + 28, &d.border_color);
        in.f32(s + 32, &d.min_lod);
        in.f32(s + 36, &d.max_lod);
        in.u32(s + 40, &d.shader_register);
        in.u32(s + 44, &d.register_space);
        in.u32(s + 48, &visibility);
        d.shader_visibility = static_cast<ShaderVisibility>(visibility);
        d.flags = 0;
        if (v12) {
            in.u32(s + 52, &d.flags);
        }
    }
    *out = std::move(desc);
    return RootSignatureError{};
}

class BlobWriter {
public:
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
        }
    }
    void f32(float v) { u32(std::bit_cast<uint32_t>(v)); }
    void patch(size_t offset, uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            bytes[offset + i] = static_cast<uint8_t>(v >> (8 * i));
        }
    }
    uint32_t pos() const { return static_cast<uint32_t>(bytes.size()); }
    std::vector<uint8_t> bytes;
};

}  // namespace

RootSignatureError parse_root_signature_blob(const void* data, size_t size, RootSignatureDesc* out) {
    if (out == nullptr || (data == nullptr && size != 0)) {
        return blob_error(Status::InvalidArgument, 0);
    }
    const auto* bytes = static_cast<const uint8_t*>(data);
    const BlobReader whole(bytes, size, 0);
    uint32_t magic = 0;
    if (!whole.u32(0, &magic)) {
        return blob_error(Status::BlobTruncated, 0);
    }
    if (magic != kFourccDxbc) {
        if (magic < 1 || magic > 3) {
            return blob_error(Status::BlobBadMagic, 0);
        }
        return parse_rts0(whole, out);
    }
    // DXBC container: magic, 16-byte digest, u16 major, u16 minor, u32 total size,
    // u32 part count, u32 part offsets[part count]; each part = fourcc, size, data.
    uint32_t part_count = 0;
    if (!whole.u32(28, &part_count)) {
        return blob_error(Status::BlobTruncated, 28);
    }
    if (!whole.span_ok(32, part_count, 4)) {
        return blob_error(Status::BlobOffsetOutOfBounds, 28);
    }
    for (uint32_t i = 0; i < part_count; ++i) {
        uint32_t part_offset = 0;
        whole.u32(32ull + 4ull * i, &part_offset);
        uint32_t fourcc = 0;
        uint32_t part_size = 0;
        if (!whole.u32(part_offset, &fourcc) || !whole.u32(part_offset + 4ull, &part_size)) {
            return blob_error(Status::BlobOffsetOutOfBounds, 32ull + 4ull * i);
        }
        if (fourcc != kFourccRts0) {
            continue;
        }
        const uint64_t data_offset = part_offset + 8ull;
        if (!whole.span_ok(data_offset, part_size, 1)) {
            return blob_error(Status::BlobOffsetOutOfBounds, part_offset + 4ull);
        }
        const BlobReader part(bytes + data_offset, part_size, data_offset);
        return parse_rts0(part, out);
    }
    return blob_error(Status::BlobMissingRootSignaturePart, 0);
}

std::vector<uint8_t> serialize_root_signature(const RootSignatureDesc& desc) {
    const uint32_t version = static_cast<uint32_t>(desc.version);
    const bool v10 = version == 1;
    const bool v12 = version == 3;
    BlobWriter w;
    const uint32_t num_params = static_cast<uint32_t>(desc.parameters.size());
    const uint32_t num_samplers = static_cast<uint32_t>(desc.static_samplers.size());
    w.u32(version);
    w.u32(num_params);
    w.u32(kRts0HeaderBytes);
    w.u32(num_samplers);
    const size_t samplers_offset_at = w.pos();
    w.u32(0);
    w.u32(desc.flags);
    const uint32_t params_at = w.pos();
    for (uint32_t i = 0; i < num_params; ++i) {
        w.u32(static_cast<uint32_t>(desc.parameters[i].parameter_type));
        w.u32(static_cast<uint32_t>(desc.parameters[i].shader_visibility));
        w.u32(0);  // payload offset, patched below
    }
    for (uint32_t i = 0; i < num_params; ++i) {
        const RootParameterDesc& p = desc.parameters[i];
        w.patch(params_at + i * kRootParameterBytes + 8, w.pos());
        switch (p.parameter_type) {
        case RootParameterType::DescriptorTable:
            w.u32(static_cast<uint32_t>(p.ranges.size()));
            w.u32(w.pos() + 4);
            for (const DescriptorRangeDesc& r : p.ranges) {
                w.u32(static_cast<uint32_t>(r.range_type));
                w.u32(r.num_descriptors);
                w.u32(r.base_shader_register);
                w.u32(r.register_space);
                if (!v10) {
                    w.u32(r.flags);
                }
                w.u32(r.offset_in_descriptors_from_table_start);
            }
            break;
        case RootParameterType::Constants32Bit:
            w.u32(p.shader_register);
            w.u32(p.register_space);
            w.u32(p.num_32bit_values);
            break;
        default:
            w.u32(p.shader_register);
            w.u32(p.register_space);
            if (!v10) {
                w.u32(p.flags);
            }
            break;
        }
    }
    w.patch(samplers_offset_at, w.pos());
    for (const StaticSamplerDesc& s : desc.static_samplers) {
        w.u32(s.filter);
        w.u32(s.address_u);
        w.u32(s.address_v);
        w.u32(s.address_w);
        w.f32(s.mip_lod_bias);
        w.u32(s.max_anisotropy);
        w.u32(s.comparison_func);
        w.u32(s.border_color);
        w.f32(s.min_lod);
        w.f32(s.max_lod);
        w.u32(s.shader_register);
        w.u32(s.register_space);
        w.u32(static_cast<uint32_t>(s.shader_visibility));
        if (v12) {
            w.u32(s.flags);
        }
    }
    return std::move(w.bytes);
}

// ===========================================================================
// RootArgumentState
// ===========================================================================

namespace {

const RootArgument kEmptyArgument{};

uint32_t heap_slot(DescriptorHeapType type) noexcept {
    return type == DescriptorHeapType::Sampler ? 1u : 0u;
}

}  // namespace

void RootArgumentState::reset() noexcept {
    for (PointState& p : points_) {
        p = PointState{};
    }
    heaps_ = {};
}

Status RootArgumentState::set_descriptor_heaps(uint32_t count, DescriptorHeap* const* heaps) noexcept {
    if (count > 2 || (count > 0 && heaps == nullptr)) {
        return Status::InvalidArgument;
    }
    std::array<DescriptorHeap*, 2> next{};
    for (uint32_t i = 0; i < count; ++i) {
        DescriptorHeap* heap = heaps[i];
        if (heap == nullptr) {
            return Status::InvalidArgument;
        }
        if (heap->type() != DescriptorHeapType::CbvSrvUav && heap->type() != DescriptorHeapType::Sampler) {
            return Status::WrongHeapType;
        }
        if (!heap->shader_visible()) {
            return Status::HeapNotShaderVisible;
        }
        const uint32_t slot = heap_slot(heap->type());
        if (next[slot] != nullptr) {
            return Status::InvalidArgument;
        }
        next[slot] = heap;
    }
    for (uint32_t slot = 0; slot < 2; ++slot) {
        if (next[slot] == heaps_[slot] || heaps_[slot] == nullptr) {
            continue;
        }
        for (PointState& p : points_) {
            if (p.signature == nullptr) {
                continue;
            }
            for (uint32_t i = 0; i < p.signature->parameter_count(); ++i) {
                RootArgument& arg = p.args[i];
                if (arg.is_set && arg.heap == heaps_[slot]) {
                    arg = RootArgument{};
                    p.dirty |= 1ull << i;
                }
            }
        }
    }
    heaps_ = next;
    return Status::Ok;
}

DescriptorHeap* RootArgumentState::bound_heap(DescriptorHeapType type) const noexcept {
    if (type != DescriptorHeapType::CbvSrvUav && type != DescriptorHeapType::Sampler) {
        return nullptr;
    }
    return heaps_[heap_slot(type)];
}

Status RootArgumentState::set_root_signature(BindPoint bind_point, const RootSignature* signature) noexcept {
    if (signature == nullptr) {
        return Status::InvalidArgument;
    }
    PointState& p = point(bind_point);
    if (p.signature == signature) {
        return Status::Ok;
    }
    p = PointState{};
    p.signature = signature;
    const uint32_t n = signature->parameter_count();
    p.dirty = n >= 64 ? ~0ull : ((1ull << n) - 1);
    return Status::Ok;
}

const RootSignature* RootArgumentState::root_signature(BindPoint bind_point) const noexcept {
    return point(bind_point).signature;
}

Status RootArgumentState::set_descriptor_table(
    BindPoint bind_point,
    uint32_t parameter_index,
    GpuDescriptorHandle base) noexcept {
    PointState& p = point(bind_point);
    if (p.signature == nullptr) {
        return Status::RootSignatureNotSet;
    }
    if (parameter_index >= p.signature->parameter_count()) {
        return Status::InvalidRootParameterIndex;
    }
    const RootSignature::ParameterInfo& info = p.signature->parameter(parameter_index);
    if (info.type != RootParameterType::DescriptorTable) {
        return Status::RootParameterTypeMismatch;
    }
    const uint32_t hi = static_cast<uint32_t>(base.ptr >> 32);
    if ((hi & kTagMask) != kGpuTag) {
        return Status::InvalidHandle;
    }
    const uint32_t serial = hi & ~kTagMask;
    DescriptorHeap* heap = heaps_[heap_slot(info.table_heap_type)];
    if (heap == nullptr || heap->serial() != serial) {
        DescriptorHeap* other = heaps_[heap_slot(info.table_heap_type) ^ 1u];
        if (other != nullptr && other->serial() == serial) {
            return Status::WrongHeapType;
        }
        return Status::HeapNotBound;
    }
    const uint32_t offset = static_cast<uint32_t>(base.ptr);
    const uint32_t inc = heap->increment();
    if ((offset & (inc - 1u)) != 0) {
        return Status::MisalignedHandle;
    }
    const uint32_t index = offset >> increment_shift(inc);
    if (index >= heap->capacity() || static_cast<uint64_t>(index) + info.table_extent > heap->capacity()) {
        return Status::HandleOutOfRange;
    }
    RootArgument& arg = p.args[parameter_index];
    if (!(arg.is_set && arg.heap == heap && arg.base_index == index)) {
        arg = RootArgument{heap, index, 0, true};
        p.dirty |= 1ull << parameter_index;
    }
    return Status::Ok;
}

Status RootArgumentState::set_root_view(
    BindPoint bind_point,
    uint32_t parameter_index,
    RootParameterType view_type,
    uint64_t gpu_address) noexcept {
    if (view_type != RootParameterType::Cbv && view_type != RootParameterType::Srv &&
        view_type != RootParameterType::Uav) {
        return Status::InvalidArgument;
    }
    PointState& p = point(bind_point);
    if (p.signature == nullptr) {
        return Status::RootSignatureNotSet;
    }
    if (parameter_index >= p.signature->parameter_count()) {
        return Status::InvalidRootParameterIndex;
    }
    if (p.signature->parameter(parameter_index).type != view_type) {
        return Status::RootParameterTypeMismatch;
    }
    RootArgument& arg = p.args[parameter_index];
    if (!(arg.is_set && arg.gpu_address == gpu_address)) {
        arg = RootArgument{nullptr, 0, gpu_address, true};
        p.dirty |= 1ull << parameter_index;
    }
    return Status::Ok;
}

Status RootArgumentState::set_root_constants(
    BindPoint bind_point,
    uint32_t parameter_index,
    uint32_t num_values,
    const uint32_t* values,
    uint32_t dest_offset) noexcept {
    PointState& p = point(bind_point);
    if (p.signature == nullptr) {
        return Status::RootSignatureNotSet;
    }
    if (parameter_index >= p.signature->parameter_count()) {
        return Status::InvalidRootParameterIndex;
    }
    const RootSignature::ParameterInfo& info = p.signature->parameter(parameter_index);
    if (info.type != RootParameterType::Constants32Bit) {
        return Status::RootParameterTypeMismatch;
    }
    if (num_values > 0 && values == nullptr) {
        return Status::InvalidArgument;
    }
    if (static_cast<uint64_t>(dest_offset) + num_values > info.num_32bit_values) {
        return Status::ConstantsOutOfRange;
    }
    uint32_t* dst = p.constants.data() + info.constants_offset + dest_offset;
    RootArgument& arg = p.args[parameter_index];
    const bool changed = num_values > 0 && std::memcmp(dst, values, static_cast<size_t>(num_values) * 4u) != 0;
    if (changed) {
        std::memcpy(dst, values, static_cast<size_t>(num_values) * 4u);
    }
    if (changed || !arg.is_set) {
        arg.is_set = true;
        p.dirty |= 1ull << parameter_index;
    }
    return Status::Ok;
}

const RootArgument& RootArgumentState::argument(BindPoint bind_point, uint32_t parameter_index) const noexcept {
    const PointState& p = point(bind_point);
    if (p.signature == nullptr || parameter_index >= p.signature->parameter_count()) {
        return kEmptyArgument;
    }
    return p.args[parameter_index];
}

std::span<const uint32_t> RootArgumentState::root_constants(
    BindPoint bind_point,
    uint32_t parameter_index) const noexcept {
    const PointState& p = point(bind_point);
    if (p.signature == nullptr || parameter_index >= p.signature->parameter_count()) {
        return {};
    }
    const RootSignature::ParameterInfo& info = p.signature->parameter(parameter_index);
    if (info.type != RootParameterType::Constants32Bit) {
        return {};
    }
    return std::span<const uint32_t>(p.constants.data() + info.constants_offset, info.num_32bit_values);
}

uint64_t RootArgumentState::dirty_parameters(BindPoint bind_point) const noexcept {
    return point(bind_point).dirty;
}

void RootArgumentState::clear_dirty(BindPoint bind_point) noexcept {
    point(bind_point).dirty = 0;
}

uint64_t RootArgumentState::unset_parameters(BindPoint bind_point) const noexcept {
    const PointState& p = point(bind_point);
    if (p.signature == nullptr) {
        return 0;
    }
    uint64_t mask = 0;
    for (uint32_t i = 0; i < p.signature->parameter_count(); ++i) {
        if (!p.args[i].is_set) {
            mask |= 1ull << i;
        }
    }
    return mask;
}

// ===========================================================================
// BindGroupCache
// ===========================================================================

namespace {

SamplerDesc static_sampler_to_desc(const StaticSamplerDesc& s) noexcept {
    SamplerDesc d;
    d.filter = s.filter;
    d.address_u = s.address_u;
    d.address_v = s.address_v;
    d.address_w = s.address_w;
    d.mip_lod_bias = s.mip_lod_bias;
    d.max_anisotropy = s.max_anisotropy;
    d.comparison_func = s.comparison_func;
    const bool white = s.border_color == 2 || s.border_color == 4;
    const bool opaque = s.border_color != 0;
    const float rgb = white ? 1.0f : 0.0f;
    d.border_color[0] = rgb;
    d.border_color[1] = rgb;
    d.border_color[2] = rgb;
    d.border_color[3] = opaque ? 1.0f : 0.0f;
    d.min_lod = s.min_lod;
    d.max_lod = s.max_lod;
    d.flags = s.flags;
    return d;
}

DescriptorKind expected_descriptor_kind(DescriptorRangeType type) noexcept {
    switch (type) {
    case DescriptorRangeType::Srv: return DescriptorKind::Srv;
    case DescriptorRangeType::Uav: return DescriptorKind::Uav;
    case DescriptorRangeType::Cbv: return DescriptorKind::Cbv;
    case DescriptorRangeType::Sampler: return DescriptorKind::Sampler;
    }
    return DescriptorKind::Empty;
}

void translate_view(
    const ViewDescriptor& v,
    DescriptorRangeType range_type,
    GpuVaResolver resolver,
    void* user,
    BindGroupEntry* e) noexcept {
    e->view = v;
    e->kind = BindResourceKind::Null;
    if (v.kind != expected_descriptor_kind(range_type)) {
        return;
    }
    if (v.kind == DescriptorKind::Cbv) {
        GpuVaResolution res;
        if (v.first_element_or_address != 0 && resolver != nullptr &&
            resolver(user, v.first_element_or_address, &res)) {
            e->kind = BindResourceKind::Buffer;
            e->resource_id = res.resource_id;
            e->offset = res.offset;
            e->size = v.num_elements_or_size;
        }
        return;
    }
    if (v.resource_id == 0) {
        return;
    }
    e->resource_id = v.resource_id;
    if (v.dimension == ViewDimension::Buffer) {
        e->kind = BindResourceKind::Buffer;
        uint64_t element_size = 0;
        if (v.structure_byte_stride != 0) {
            element_size = v.structure_byte_stride;
        } else if ((v.flags & BufferViewFlags::Raw) != 0) {
            element_size = 4;
        }
        // Typed buffer views (format-sized elements) bind the whole buffer; the
        // element window stays available in `view`.
        e->offset = v.first_element_or_address * element_size;
        e->size = static_cast<uint64_t>(v.num_elements_or_size) * element_size;
        return;
    }
    e->kind = BindResourceKind::TextureView;
}

// One descriptor-table entry of a bind group: slot `index` of `heap`, from the registry record or
// (when `source` reads the heap) from descriptor memory.
void table_entry(const DescriptorSlotSource* source,
                 const DescriptorHeap& heap,
                 uint32_t index,
                 DescriptorRangeType register_class,
                 GpuVaResolver resolver,
                 void* user,
                 BindGroupEntry* e) {
    e->kind = BindResourceKind::Null;
    if (index >= heap.capacity()) {
        return;
    }
    if (source != nullptr && source->reads(heap)) {
        if (register_class == DescriptorRangeType::Sampler) {
            SamplerDesc s;
            if (source->sampler(heap, index, &s)) {
                e->kind = BindResourceKind::Sampler;
                e->sampler = s;
            }
        } else {
            translate_view(source->view(heap, index, register_class), register_class, resolver, user, e);
        }
        return;
    }
    if (register_class == DescriptorRangeType::Sampler) {
        const SamplerDescriptor* s = heap.sampler(index);
        if (s != nullptr && s->kind == DescriptorKind::Sampler) {
            e->kind = BindResourceKind::Sampler;
            e->sampler = s->desc;
        }
    } else if (const ViewDescriptor* v = heap.view(index)) {
        translate_view(*v, register_class, resolver, user, e);
    }
}

uint64_t hash_key(const std::vector<uint64_t>& key) noexcept {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (static_cast<uint64_t>(key.size()) * 0xbf58476d1ce4e5b9ull);
    for (uint64_t v : key) {
        v += 0x9e3779b97f4a7c15ull;
        v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull;
        v = (v ^ (v >> 27)) * 0x94d049bb133111ebull;
        v ^= v >> 31;
        h ^= v;
        h = std::rotl(h, 27) * 5u + 0x52dce729u;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

}  // namespace

BindGroupPlan BindGroupPlan::build(
    const RootSignature& signature,
    std::span<const ShaderBinding> bindings,
    uint32_t stage_mask) {
    BindGroupPlan plan;
    plan.signature_hash_ = signature.hash();
    plan.entries_.reserve(bindings.size());
    const bool compute = (stage_mask & WebGpuStage::Compute) != 0;
    for (const ShaderBinding& b : bindings) {
        PlanEntry e;
        e.binding = b.binding;
        e.register_class = b.root_constants ? DescriptorRangeType::Cbv : b.register_class;
        if (b.inline_sampler) {
            e.source = PlanSource::InlineSampler;
            e.register_class = DescriptorRangeType::Sampler;
            e.sampler = b.sampler;
            plan.entries_.push_back(e);
            continue;
        }
        if (b.heap != HeapAccess::None) {
            // Heap-indexed: the index is a root constant or a constant-buffer dword; the descriptor
            // comes from the bound heap.
            bool index_ok = b.heap_index_source == HeapIndexSource::Constant;
            if (!index_ok && b.heap_index_parameter < signature.parameter_count()) {
                const auto& ip = signature.parameter(b.heap_index_parameter);
                switch (b.heap_index_source) {
                case HeapIndexSource::RootConstantIndex:
                    index_ok = ip.type == RootParameterType::Constants32Bit && b.heap_index_dword < ip.num_32bit_values;
                    break;
                case HeapIndexSource::CbRootConstants:
                    index_ok = ip.type == RootParameterType::Constants32Bit && b.cb_byte_offset % 4 == 0 &&
                               b.cb_byte_offset / 4 < ip.num_32bit_values;
                    break;
                case HeapIndexSource::CbRootDescriptor:
                    index_ok = ip.type == RootParameterType::Cbv && b.cb_byte_offset % 4 == 0;
                    break;
                case HeapIndexSource::CbTable:
                    index_ok = ip.type == RootParameterType::DescriptorTable &&
                               ip.table_heap_type == DescriptorHeapType::CbvSrvUav && b.cb_byte_offset % 4 == 0;
                    break;
                case HeapIndexSource::Constant:
                    break;
                }
                index_ok = index_ok && (b.heap_index_source == HeapIndexSource::RootConstantIndex ||
                                        (b.cb_bit_offset < 32 && b.cb_mask != 0));
            }
            if (index_ok && b.heap_shift < 32 &&
                (b.heap == HeapAccess::Sampler) == (b.register_class == DescriptorRangeType::Sampler)) {
                e.source = PlanSource::HeapIndexed;
                plan.has_heap_indexed_ = true;
                e.root_parameter_index = b.heap_index_source == HeapIndexSource::Constant ? kNoIndex
                                                                                           : b.heap_index_parameter;
                e.heap_dword = b.heap_index_source == HeapIndexSource::RootConstantIndex ? b.heap_index_dword : 0;
                e.heap_shift = b.heap_shift;
                e.heap_stride = b.heap_stride;
                e.heap_index_source = b.heap_index_source;
                e.heap_index_bias = b.array_index;
                if (b.heap_index_source != HeapIndexSource::RootConstantIndex &&
                    b.heap_index_source != HeapIndexSource::Constant) {
                    e.table_offset = b.heap_index_source == HeapIndexSource::CbTable ? b.cb_table_slot : 0;
                    e.cb_byte_offset = b.cb_byte_offset;
                    e.cb_bit_offset = b.cb_bit_offset;
                    e.cb_mask = b.cb_mask;
                }
                if (e.root_parameter_index != kNoIndex) {
                    plan.parameter_mask_ |= 1ull << e.root_parameter_index;
                }
            } else {
                ++plan.unresolved_;
            }
            plan.entries_.push_back(e);
            continue;
        }
        if (b.table_parameter != kNoIndex) {
            // Direct table slot: the descriptor at that offset from the table's base, whatever
            // ranges the root signature declares (the heap type must match the class).
            const bool sampler = b.register_class == DescriptorRangeType::Sampler;
            if (b.table_parameter < signature.parameter_count() &&
                signature.parameter(b.table_parameter).type == RootParameterType::DescriptorTable &&
                (signature.parameter(b.table_parameter).table_heap_type == DescriptorHeapType::Sampler) == sampler) {
                e.source = PlanSource::DescriptorTable;
                e.root_parameter_index = b.table_parameter;
                e.table_offset = b.table_slot;
                plan.parameter_mask_ |= 1ull << e.root_parameter_index;
            } else {
                ++plan.unresolved_;
            }
            plan.entries_.push_back(e);
            continue;
        }
        const BindingSlot* slot = nullptr;
        uint32_t table_offset = 0;
        if (b.root_constants) {
            for (const BindingSlot& s : signature.layout().slots) {
                if (s.source == SlotSource::RootConstants &&
                    (compute || (webgpu_stage_mask(s.visibility) & stage_mask) != 0)) {
                    slot = &s;
                    break;
                }
            }
        } else {
            slot = signature.layout().find(b.register_class, b.register_space, b.shader_register, stage_mask,
                                           &table_offset);
        }
        if (slot == nullptr) {
            ++plan.unresolved_;
        } else {
            e.root_parameter_index = slot->root_parameter_index;
            switch (slot->source) {
            case SlotSource::DescriptorTable:
                e.source = PlanSource::DescriptorTable;
                e.table_offset = table_offset;
                break;
            case SlotSource::RootDescriptor:
                e.source = PlanSource::RootDescriptor;
                break;
            case SlotSource::RootConstants:
                e.source = PlanSource::RootConstants;
                break;
            case SlotSource::StaticSampler:
                e.source = PlanSource::StaticSampler;
                break;
            }
            if (e.source != PlanSource::StaticSampler) {
                plan.parameter_mask_ |= 1ull << e.root_parameter_index;
            }
        }
        plan.entries_.push_back(e);
    }
    std::stable_sort(plan.entries_.begin(), plan.entries_.end(),
                     [](const PlanEntry& a, const PlanEntry& b) { return a.binding < b.binding; });

    // One window per table parameter covering every offset the plan reads.
    for (uint32_t i = 0; i < signature.parameter_count(); ++i) {
        if ((plan.parameter_mask_ & (1ull << i)) == 0 ||
            signature.parameter(i).type != RootParameterType::DescriptorTable) {
            continue;
        }
        uint32_t lo = 0xffffffffu;
        uint32_t hi = 0;
        bool any = false;
        for (const PlanEntry& e : plan.entries_) {
            if (e.source == PlanSource::DescriptorTable && e.root_parameter_index == i) {
                lo = std::min(lo, e.table_offset);
                hi = std::max(hi, e.table_offset);
                any = true;
            }
        }
        if (any) {  // a table only a CB-indexed entry reads its index through binds no slot
            plan.windows_.push_back(TableWindow{i, lo, hi - lo + 1});
        }
    }

    Fnv64 h;
    h.u64(plan.signature_hash_);
    h.u32(static_cast<uint32_t>(plan.entries_.size()));
    for (const PlanEntry& e : plan.entries_) {
        h.u32(e.binding);
        h.u32(static_cast<uint32_t>(e.source));
        h.u32(static_cast<uint32_t>(e.register_class));
        h.u32(e.root_parameter_index);
        h.u32(e.table_offset);
        h.u32(e.heap_dword);
        h.u32(e.heap_shift);
        h.u32(static_cast<uint32_t>(e.heap_index_source));
        h.u32(e.cb_byte_offset);
        h.u32(e.cb_bit_offset);
        h.u32(e.cb_mask);
        h.u32(e.heap_index_bias);
        h.u32(e.heap_stride);
        if (e.source == PlanSource::InlineSampler) {
            uint32_t words[sizeof(SamplerDesc) / 4] = {};
            std::memcpy(words, &e.sampler, sizeof(words));
            for (uint32_t w : words) {
                h.u32(w);
            }
        }
    }
    plan.hash_ = h.h;
    return plan;
}

namespace {

// The 32-bit index dword of a CB-indexed plan entry (HeapIndexSource) under the current arguments:
// read from the root constants, or from guest memory at the root CBV / the table slot's CBV address
// through `source`. False when there is no such dword (no CBV, unreadable, offset outside the CBV).
bool cb_index_dword(const RootArgumentState& state, BindPoint bind_point, const PlanEntry& p,
                    const DescriptorSlotSource* source, uint32_t* dword) {
    switch (p.heap_index_source) {
    case HeapIndexSource::RootConstantIndex:
        return false;
    case HeapIndexSource::Constant:
        *dword = 0;
        return true;
    case HeapIndexSource::CbRootConstants: {
        const std::span<const uint32_t> values = state.root_constants(bind_point, p.root_parameter_index);
        if (p.cb_byte_offset / 4 >= values.size()) {
            return false;
        }
        *dword = values[p.cb_byte_offset / 4];
        return true;
    }
    case HeapIndexSource::CbRootDescriptor: {
        const RootArgument& arg = state.argument(bind_point, p.root_parameter_index);
        return arg.is_set && arg.gpu_address != 0 && source != nullptr &&
               source->read_gpu_memory(arg.gpu_address + p.cb_byte_offset, dword, 4);
    }
    case HeapIndexSource::CbTable: {
        const RootArgument& arg = state.argument(bind_point, p.root_parameter_index);
        if (!arg.is_set || arg.heap == nullptr) {
            return false;
        }
        const uint64_t slot = static_cast<uint64_t>(arg.base_index) + p.table_offset;
        if (slot >= arg.heap->capacity()) {
            return false;
        }
        ViewDescriptor v;
        if (source != nullptr && source->reads(*arg.heap)) {
            v = source->view(*arg.heap, static_cast<uint32_t>(slot), DescriptorRangeType::Cbv);
        } else if (const ViewDescriptor* r = arg.heap->view(static_cast<uint32_t>(slot))) {
            v = *r;
        }
        if (v.kind != DescriptorKind::Cbv || v.first_element_or_address == 0 ||
            (v.num_elements_or_size != 0 && uint64_t{p.cb_byte_offset} + 4 > v.num_elements_or_size)) {
            return false;
        }
        return source != nullptr && source->read_gpu_memory(v.first_element_or_address + p.cb_byte_offset, dword, 4);
    }
    }
    return false;
}

// The heap byte offset a heap-indexed plan entry selects under the current arguments: false when
// the index is not available (*cb_miss set when it is a CB index that could not be read).
// RDNA2 scalar-load soffset is 32-bit: the scaled index wraps in hardware.
uint32_t scaled_heap_index(const PlanEntry& p, uint32_t index) {
    return p.heap_stride != 0 ? index * p.heap_stride : index << p.heap_shift;
}

bool heap_byte_offset(const RootArgumentState& state, BindPoint bind_point, const PlanEntry& p,
                      const DescriptorSlotSource* source, uint64_t* byte_offset, bool* cb_miss = nullptr) {
    if (p.heap_index_source == HeapIndexSource::RootConstantIndex) {
        const std::span<const uint32_t> values = state.root_constants(bind_point, p.root_parameter_index);
        if (p.heap_dword >= values.size()) {
            return false;
        }
        *byte_offset = scaled_heap_index(p, values[p.heap_dword] + p.heap_index_bias);
        return true;
    }
    uint32_t dword = 0;
    if (!cb_index_dword(state, bind_point, p, source, &dword)) {
        if (cb_miss != nullptr) {
            *cb_miss = true;
        }
        return false;
    }
    // 32-bit, as above; a binding array element adds its index to the window base.
    *byte_offset = scaled_heap_index(p, ((dword >> p.cb_bit_offset) & p.cb_mask) + p.heap_index_bias);
    return true;
}

// The heap slot a heap-indexed plan entry reads under the current root arguments: false (and
// nothing set) when no heap of that type is bound, the index is not available, or the byte offset
// is not a whole descriptor inside the heap.
bool heap_slot(const RootArgumentState& state, BindPoint bind_point, const PlanEntry& p,
               const DescriptorSlotSource* source, DescriptorHeap** heap, uint32_t* index, bool* cb_miss = nullptr) {
    const DescriptorHeap* h = state.bound_heap(p.register_class == DescriptorRangeType::Sampler
                                                   ? DescriptorHeapType::Sampler
                                                   : DescriptorHeapType::CbvSrvUav);
    uint64_t byte_offset = 0;
    if (!heap_byte_offset(state, bind_point, p, source, &byte_offset, cb_miss) || h == nullptr) {
        return false;
    }
    const uint32_t inc = h->increment();
    if (inc == 0 || (byte_offset & (inc - 1u)) != 0) {
        return false;
    }
    const uint64_t idx = byte_offset >> increment_shift(inc);
    if (idx >= h->capacity()) {
        return false;
    }
    *heap = const_cast<DescriptorHeap*>(h);
    *index = static_cast<uint32_t>(idx);
    return true;
}

}  // namespace

void BindGroupCache::build_entries(
    const RootArgumentState& state,
    BindPoint bind_point,
    const RootSignature& signature,
    const BindGroupPlan& plan,
    GpuVaResolver resolver,
    void* resolver_user,
    CachedBindGroup* out) {
    out->key = scratch_key_;
    out->entries.clear();
    out->null_entries = 0;
    out->built_generation.clear();
    for (const BindGroupPlan::TableWindow& w : plan.table_windows()) {
        out->built_generation.push_back(state.argument(bind_point, w.parameter_index).heap->generation());
    }
    out->built_words.clear();
    out->table_bases.clear();
    out->heap_slots.clear();
    out->from_memory = memory_words(state, bind_point, plan, &out->built_words);
    if (out->from_memory) {
        for (const BindGroupPlan::TableWindow& w : plan.table_windows()) {
            const RootArgument& arg = state.argument(bind_point, w.parameter_index);
            out->table_bases.emplace_back(w.parameter_index, arg.heap->gpu_handle(arg.base_index).ptr);
        }
    }
    out->entries.reserve(plan.entries().size());
    const RootSignatureDesc& desc = signature.desc();
    for (const PlanEntry& p : plan.entries()) {
        BindGroupEntry e;
        e.binding = p.binding;
        e.root_parameter_index = p.root_parameter_index;
        switch (p.source) {
        case PlanSource::Unresolved:
            break;
        case PlanSource::DescriptorTable: {
            const RootArgument& arg = state.argument(bind_point, p.root_parameter_index);
            const uint64_t index = static_cast<uint64_t>(arg.base_index) + p.table_offset;
            if (index < arg.heap->capacity()) {
                table_entry(slot_source_, *arg.heap, static_cast<uint32_t>(index), p.register_class, resolver,
                            resolver_user, &e);
            }
            break;
        }
        case PlanSource::RootDescriptor: {
            const RootArgument& arg = state.argument(bind_point, p.root_parameter_index);
            GpuVaResolution res;
            if (arg.gpu_address != 0 && resolver != nullptr && resolver(resolver_user, arg.gpu_address, &res)) {
                e.kind = BindResourceKind::Buffer;
                e.resource_id = res.resource_id;
                e.offset = res.offset;
                if (p.register_class == DescriptorRangeType::Cbv) {
                    const uint64_t remaining = res.resource_size > res.offset ? res.resource_size - res.offset : 0;
                    e.size = std::min(remaining, kMaxUniformBindingSize);
                }
            }
            break;
        }
        case PlanSource::RootConstants:
            e.kind = BindResourceKind::RootConstants;
            e.size = static_cast<uint64_t>(signature.parameter(p.root_parameter_index).num_32bit_values) * 4u;
            break;
        case PlanSource::StaticSampler:
            e.kind = BindResourceKind::Sampler;
            e.root_parameter_index = kNoIndex;
            e.sampler = static_sampler_to_desc(desc.static_samplers[p.root_parameter_index]);
            break;
        case PlanSource::InlineSampler:
            e.kind = BindResourceKind::Sampler;
            e.root_parameter_index = kNoIndex;
            e.sampler = p.sampler;
            break;
        case PlanSource::HeapIndexed: {
            DescriptorHeap* heap = nullptr;
            uint32_t index = 0;
            CachedBindGroup::HeapSlot slot;
            slot.entry = static_cast<uint32_t>(out->entries.size());
            bool cb_miss = false;
            if (heap_slot(state, bind_point, p, slot_source_, &heap, &index, &cb_miss)) {
                slot.gpu_handle = heap->gpu_handle(index).ptr;
                slot.generation = heap->generation();
                table_entry(slot_source_, *heap, index, p.register_class, resolver, resolver_user, &e);
            } else {
                ++heap_index_misses_;  // index outside the bound heap: bound as null, counted
                cb_index_misses_ += cb_miss ? 1u : 0u;
            }
            out->heap_slots.push_back(slot);
            break;
        }
        }
        out->null_entries += e.kind == BindResourceKind::Null ? 1u : 0u;
        out->entries.push_back(e);
    }
}

namespace {

constexpr uint8_t kBindSlotEmpty = 0;
constexpr uint8_t kBindSlotOccupied = 1;
constexpr uint8_t kBindSlotTombstone = 2;
constexpr size_t kBindGroupInitialCapacity = 32;

}  // namespace

BindGroupCache::Slot* BindGroupCache::find_slot(uint64_t hash) noexcept {
    if (table_.empty()) {
        return nullptr;
    }
    const size_t mask = table_.size() - 1;
    size_t idx = static_cast<size_t>(hash) & mask;
    for (;;) {
        Slot& s = table_[idx];
        if (s.state == kBindSlotEmpty) {
            return nullptr;
        }
        if (s.state == kBindSlotOccupied && s.hash == hash) {
            return &s;
        }
        idx = (idx + 1) & mask;
    }
}

void BindGroupCache::rehash(size_t new_cap) {
    std::vector<Slot> next(new_cap);
    const size_t mask = new_cap - 1;
    for (Slot& s : table_) {
        if (s.state != kBindSlotOccupied) {
            continue;
        }
        size_t idx = static_cast<size_t>(s.hash) & mask;
        while (next[idx].state == kBindSlotOccupied) {
            idx = (idx + 1) & mask;
        }
        next[idx].hash = s.hash;
        next[idx].entry = std::move(s.entry);
        next[idx].state = kBindSlotOccupied;
    }
    table_ = std::move(next);
    tombstones_ = 0;
}

void BindGroupCache::insert_entry(uint64_t hash, std::unique_ptr<CachedBindGroup> entry) {
    if (table_.empty()) {
        table_.resize(kBindGroupInitialCapacity);
    } else if ((size_ + tombstones_ + 1) * 10 >= table_.size() * 7) {
        const size_t new_cap = tombstones_ > size_ ? table_.size() : table_.size() * 2;
        rehash(new_cap);
    }
    const size_t mask = table_.size() - 1;
    size_t idx = static_cast<size_t>(hash) & mask;
    for (;;) {
        Slot& s = table_[idx];
        if (s.state != kBindSlotOccupied) {
            if (s.state == kBindSlotTombstone) {
                --tombstones_;
            }
            s.hash = hash;
            s.entry = std::move(entry);
            s.state = kBindSlotOccupied;
            ++size_;
            return;
        }
        idx = (idx + 1) & mask;
    }
}

BindGroupLookup BindGroupCache::get_or_build(
    const RootArgumentState& state,
    BindPoint bind_point,
    const BindGroupPlan& plan,
    uint64_t layout_key,
    GpuVaResolver resolver,
    void* resolver_user) {
    BindGroupLookup result;
    const RootSignature* signature = state.root_signature(bind_point);
    if (signature == nullptr) {
        result.status = Status::RootSignatureNotSet;
        return result;
    }
    if (plan.signature_hash() != signature->hash()) {
        result.status = Status::InvalidArgument;  // plan was built for another root signature
        return result;
    }
    if (plan.entries().empty()) {
        return result;
    }
    scratch_key_.clear();
    scratch_key_.push_back(layout_key);
    scratch_key_.push_back(plan.hash());
    uint64_t rem = plan.parameter_mask();
    const uint32_t param_count = signature->parameter_count();
    while (rem != 0) {
        const uint32_t i = static_cast<uint32_t>(std::countr_zero(rem));
        rem &= rem - 1;
        if (i >= param_count) {
            break;
        }
        const RootArgument& arg = state.argument(bind_point, i);
        if (!arg.is_set) {
            result.status = Status::RootArgumentNotSet;
            return result;
        }
        switch (signature->parameter(i).type) {
        case RootParameterType::DescriptorTable:
            scratch_key_.push_back((static_cast<uint64_t>(arg.heap->serial()) << 32) | arg.base_index);
            break;
        case RootParameterType::Constants32Bit:
            scratch_key_.push_back(i);
            break;
        default:
            scratch_key_.push_back(arg.gpu_address);
            break;
        }
    }
    // Heap-indexed entries: the bound heap and the byte offset the root constant (or the constant-
    // buffer dword, read now: G62 CB-indexed heaps) selects.
    if (plan.has_heap_indexed()) {
        for (const PlanEntry& p : plan.entries()) {
            if (p.source != PlanSource::HeapIndexed) {
                continue;
            }
            const DescriptorHeap* heap = state.bound_heap(
                p.register_class == DescriptorRangeType::Sampler ? DescriptorHeapType::Sampler : DescriptorHeapType::CbvSrvUav);
            uint64_t byte_offset = 0;
            scratch_key_.push_back(heap != nullptr ? heap->serial() : 0u);
            scratch_key_.push_back(heap_byte_offset(state, bind_point, p, slot_source_, &byte_offset) ? byte_offset : ~0ull);
        }
    }
    const uint64_t h = hash_key(scratch_key_);
    CachedBindGroup* hit_entry = nullptr;
    Slot* existing_slot = nullptr;
    if (last_entry_ != nullptr && last_hash_ == h && last_entry_->key == scratch_key_) {
        hit_entry = last_entry_;
    } else {
        existing_slot = find_slot(h);
        if (existing_slot != nullptr && existing_slot->entry->key == scratch_key_) {
            hit_entry = existing_slot->entry.get();
            last_hash_ = h;
            last_entry_ = hit_entry;
        }
    }
    if (hit_entry != nullptr) {
        CachedBindGroup& cached = *hit_entry;
        cached.last_used_epoch = epoch_;
        bool changed = false;
        const std::vector<BindGroupPlan::TableWindow>& windows = plan.table_windows();
        for (size_t t = 0; t < windows.size() && !changed; ++t) {
            const RootArgument& arg = state.argument(bind_point, windows[t].parameter_index);
            const uint64_t first = static_cast<uint64_t>(arg.base_index) + windows[t].first_offset;
            if (first < arg.heap->capacity()) {
                changed = arg.heap->range_changed_since(
                    static_cast<uint32_t>(first), windows[t].count, cached.built_generation[t]);
            }
        }
        for (const CachedBindGroup::HeapSlot& s : cached.heap_slots) {
            DescriptorHeap* heap = nullptr;
            uint32_t index = 0;
            if (!changed && s.gpu_handle != 0 && s.entry < plan.entries().size() &&
                heap_slot(state, bind_point, plan.entries()[s.entry], slot_source_, &heap, &index)) {
                changed = heap->range_changed_since(index, 1, s.generation);
            }
        }
        if (!changed && cached.from_memory) {
            // Slots the title writes itself have no generation: compare the words.
            changed = !memory_words(state, bind_point, plan, &scratch_words_) || scratch_words_ != cached.built_words;
        }
        result.bind_group = &cached;
        if (!changed) {
            ++hits_;
            return result;
        }
        ++misses_;
        result.rebuilt = true;
        result.retired_gpu_id = cached.gpu_id;
        cached.gpu_id = 0;
        build_entries(state, bind_point, *signature, plan, resolver, resolver_user, &cached);
        note_group_ids(cached);
        return result;
    }
    ++misses_;
    if (existing_slot != nullptr) {
        // Hash collision with a different key: replace in-place.
        result.retired_gpu_id = existing_slot->entry->gpu_id;
        if (last_entry_ == existing_slot->entry.get()) {
            last_entry_ = nullptr;
            last_hash_ = 0;
        }
        auto entry = std::make_unique<CachedBindGroup>();
        entry->layout_key = layout_key;
        entry->key_hash = h;
        entry->last_used_epoch = epoch_;
        build_entries(state, bind_point, *signature, plan, resolver, resolver_user, entry.get());
        note_group_ids(*entry);
        CachedBindGroup* ptr = entry.get();
        result.bind_group = ptr;
        result.rebuilt = true;
        const uint64_t budget = g_bind_group_slot_budget.load(std::memory_order_relaxed);
        if (budget != 0 && (size_ - 1) >= budget) {
            g_bind_group_budget_overflows.fetch_add(1, std::memory_order_relaxed);  // grows, never drops the draw
        }
        existing_slot->entry = std::move(entry);
        last_hash_ = h;
        last_entry_ = ptr;
        return result;
    }
    auto entry = std::make_unique<CachedBindGroup>();
    entry->layout_key = layout_key;
    entry->key_hash = h;
    entry->last_used_epoch = epoch_;
    build_entries(state, bind_point, *signature, plan, resolver, resolver_user, entry.get());
        note_group_ids(*entry);
    CachedBindGroup* ptr = entry.get();
    result.bind_group = ptr;
    result.rebuilt = true;
    const uint64_t budget = g_bind_group_slot_budget.load(std::memory_order_relaxed);
    if (budget != 0 && size_ >= budget) {
        g_bind_group_budget_overflows.fetch_add(1, std::memory_order_relaxed);  // grows, never drops the draw
    }
    insert_entry(h, std::move(entry));
    last_hash_ = h;
    last_entry_ = ptr;
    peak_epoch_entries_ = std::max(peak_epoch_entries_, size_);
    raise_peak(peak_epoch_entries_);
    track_size();
    return result;
}

bool BindGroupCache::memory_words(const RootArgumentState& state, BindPoint bind_point, const BindGroupPlan& plan,
                                  std::vector<uint8_t>* out) const {
    out->clear();
    if (slot_source_ == nullptr) {
        return false;
    }
    bool any = false;
    for (const BindGroupPlan::TableWindow& w : plan.table_windows()) {
        const RootArgument& arg = state.argument(bind_point, w.parameter_index);
        if (arg.heap == nullptr || !slot_source_->reads(*arg.heap)) {
            continue;
        }
        const uint64_t first = static_cast<uint64_t>(arg.base_index) + w.first_offset;
        if (first >= arg.heap->capacity()) {
            continue;
        }
        const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(w.count, arg.heap->capacity() - first));
        slot_source_->snapshot(*arg.heap, static_cast<uint32_t>(first), count, &scratch_window_);
        out->insert(out->end(), scratch_window_.begin(), scratch_window_.end());
        any = true;
    }
    // Heap-indexed slots the title writes itself, after the windows.
    if (plan.has_heap_indexed()) {
        for (const PlanEntry& p : plan.entries()) {
            DescriptorHeap* heap = nullptr;
            uint32_t index = 0;
            if (p.source != PlanSource::HeapIndexed || !heap_slot(state, bind_point, p, slot_source_, &heap, &index) ||
                !slot_source_->reads(*heap)) {
                continue;
            }
            slot_source_->snapshot(*heap, index, 1, &scratch_window_);
            out->insert(out->end(), scratch_window_.begin(), scratch_window_.end());
            any = true;
        }
    }
    return any;
}

bool refresh_table_entries(const DescriptorHeapRegistry& heaps,
                           const DescriptorSlotSource& source,
                           const BindGroupPlan& plan,
                           GpuVaResolver resolver,
                           void* resolver_user,
                           CachedBindGroup* group,
                           bool force_rebuild) {
    if (!group->from_memory) {
        return false;
    }
    // Resolve every table base again (the heap may be gone) and read the windows.
    std::vector<std::pair<uint32_t, ResolvedDescriptor>> bases;
    std::vector<uint8_t> words;
    std::vector<uint8_t> window;
    for (const BindGroupPlan::TableWindow& w : plan.table_windows()) {
        auto it = std::find_if(group->table_bases.begin(), group->table_bases.end(),
                               [&](const auto& b) { return b.first == w.parameter_index; });
        ResolvedDescriptor r;
        if (it == group->table_bases.end() || heaps.resolve_gpu(GpuDescriptorHandle{it->second}, &r) != Status::Ok) {
            continue;
        }
        bases.emplace_back(w.parameter_index, r);
        if (!source.reads(*r.heap)) {
            continue;
        }
        const uint64_t first = static_cast<uint64_t>(r.index) + w.first_offset;
        if (first >= r.heap->capacity()) {
            continue;
        }
        const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(w.count, r.heap->capacity() - first));
        source.snapshot(*r.heap, static_cast<uint32_t>(first), count, &window);
        words.insert(words.end(), window.begin(), window.end());
    }
    // Heap-indexed slots, by the handle read at the build (the root constants are part of the key).
    std::vector<std::pair<uint32_t, ResolvedDescriptor>> heap_slots;
    for (const CachedBindGroup::HeapSlot& s : group->heap_slots) {
        ResolvedDescriptor r;
        if (s.gpu_handle == 0 || heaps.resolve_gpu(GpuDescriptorHandle{s.gpu_handle}, &r) != Status::Ok) {
            continue;
        }
        heap_slots.emplace_back(s.entry, r);
        if (source.reads(*r.heap)) {
            source.snapshot(*r.heap, r.index, 1, &window);
            words.insert(words.end(), window.begin(), window.end());
        }
    }
    if (!force_rebuild && words == group->built_words) {
        return false;
    }
    group->built_words = std::move(words);
    group->null_entries = 0;
    for (size_t i = 0; i < group->entries.size() && i < plan.entries().size(); ++i) {
        const PlanEntry& p = plan.entries()[i];
        BindGroupEntry& e = group->entries[i];
        if (p.source == PlanSource::DescriptorTable) {
            auto it = std::find_if(bases.begin(), bases.end(), [&](const auto& b) { return b.first == p.root_parameter_index; });
            e = BindGroupEntry{};
            e.binding = p.binding;
            e.root_parameter_index = p.root_parameter_index;
            if (it != bases.end()) {
                table_entry(&source, *it->second.heap, static_cast<uint32_t>(std::min<uint64_t>(
                                                           uint64_t{it->second.index} + p.table_offset, 0xffffffffu)),
                            p.register_class, resolver, resolver_user, &e);
            }
        } else if (p.source == PlanSource::HeapIndexed) {
            auto it = std::find_if(heap_slots.begin(), heap_slots.end(), [&](const auto& s) { return s.first == i; });
            if (it != heap_slots.end()) {
                e = BindGroupEntry{};
                e.binding = p.binding;
                e.root_parameter_index = p.root_parameter_index;
                table_entry(&source, *it->second.heap, it->second.index, p.register_class, resolver, resolver_user, &e);
            }
        }
        group->null_entries += e.kind == BindResourceKind::Null ? 1u : 0u;
    }
    return true;
}

void BindGroupCache::clear(std::vector<uint32_t>* retired) {
    for (Slot& s : table_) {
        if (s.state == kBindSlotOccupied) {
            if (retired != nullptr && s.entry->gpu_id != 0) {
                retired->push_back(s.entry->gpu_id);
            }
            s.entry.reset();
        }
        s.state = kBindSlotEmpty;
        s.hash = 0;
    }
    size_ = 0;
    tombstones_ = 0;
    last_hash_ = 0;
    last_entry_ = nullptr;
    std::fill(id_filter_.begin(), id_filter_.end(), uint64_t{0});
    filter_notes_ = 0;
    track_size();
}

BindGroupCache::~BindGroupCache() {
    table_.clear();
    size_ = 0;
    tombstones_ = 0;
    last_hash_ = 0;
    last_entry_ = nullptr;
    track_size();
}

void BindGroupCache::track_size() noexcept {
    const size_t now = size_;
    if (now >= tracked_size_) {
        g_live_cached_bind_groups.fetch_add(now - tracked_size_, std::memory_order_relaxed);
    } else {
        g_live_cached_bind_groups.fetch_sub(tracked_size_ - now, std::memory_order_relaxed);
    }
    tracked_size_ = now;
}

uint32_t BindGroupCache::erase(const CachedBindGroup* group) {
    if (group == nullptr || table_.empty()) {
        return 0;
    }
    Slot* s = find_slot(group->key_hash);
    if (s != nullptr && s->entry.get() == group) {
        const uint32_t id = s->entry->gpu_id;
        if (last_entry_ == group) {
            last_entry_ = nullptr;
            last_hash_ = 0;
        }
        s->entry.reset();
        s->state = kBindSlotTombstone;
        --size_;
        ++tombstones_;
        track_size();
        return id;
    }
    return 0;
}

size_t BindGroupCache::evict_resource(uint32_t resource_id, std::vector<uint32_t>* retired) {
    if (resource_id == 0) {
        return 0;
    }
    return evict_resources_and_views(std::span<const uint32_t>(&resource_id, 1), {}, retired);
}

void BindGroupCache::note_group_ids(const CachedBindGroup& g) noexcept {
    auto set = [this](uint32_t id, bool view) noexcept {
        if (id != 0) {
            const size_t b = id_filter_bit(id, view);
            id_filter_[b >> 6] |= uint64_t{1} << (b & 63);
        }
    };
    for (const BindGroupEntry& e : g.entries) {
        set(e.resource_id, false);
        set(e.view.resource_id, false);
        set(e.view.counter_resource_id, false);
    }
    for (const CachedBindGroup::Relocation& r : g.relocations) {
        set(r.buffer_id, false);
    }
    for (const uint32_t vid : g.view_ids) {
        set(vid, true);
    }
    ++filter_notes_;
}

void BindGroupCache::rebuild_id_filter() noexcept {
    std::fill(id_filter_.begin(), id_filter_.end(), uint64_t{0});
    for (const Slot& s : table_) {
        if (s.state == kBindSlotOccupied) {
            note_group_ids(*s.entry);
        }
    }
    filter_notes_ = 0;
}

size_t BindGroupCache::evict_resources_and_views(std::span<const uint32_t> resource_ids,
                                                 std::span<const uint32_t> view_ids,
                                                 std::vector<uint32_t>* retired,
                                                 std::vector<uint64_t>* evicted_key_hashes) {
    if (size_ == 0 || (resource_ids.empty() && view_ids.empty())) {
        return 0;
    }
    {
        bool maybe = false;
        for (const uint32_t id : resource_ids) {
            if (id != 0 && filter_may_hold(id, false)) {
                maybe = true;
                break;
            }
        }
        for (size_t i = 0; !maybe && i < view_ids.size(); ++i) {
            maybe = view_ids[i] != 0 && filter_may_hold(view_ids[i], true);
        }
        if (!maybe) {
            return 0;  // no cached group references any of the ids
        }
    }
    auto contains = [](std::span<const uint32_t> ids, uint32_t id) noexcept -> bool {
        if (id == 0) {
            return false;
        }
        for (const uint32_t candidate : ids) {
            if (candidate == id) {
                return true;
            }
        }
        return false;
    };
    auto references_any = [&](const CachedBindGroup& g) noexcept -> bool {
        if (!resource_ids.empty()) {
            for (const BindGroupEntry& e : g.entries) {
                if (contains(resource_ids, e.resource_id) ||
                    contains(resource_ids, e.view.resource_id) ||
                    contains(resource_ids, e.view.counter_resource_id)) {
                    return true;
                }
            }
            for (const CachedBindGroup::Relocation& r : g.relocations) {
                if (contains(resource_ids, r.buffer_id)) {
                    return true;
                }
            }
        }
        if (!view_ids.empty()) {
            for (const uint32_t vid : g.view_ids) {
                if (contains(view_ids, vid)) {
                    return true;
                }
            }
        }
        return false;
    };
    size_t dropped = 0;
    for (Slot& s : table_) {
        if (s.state == kBindSlotOccupied && references_any(*s.entry)) {
            if (retired != nullptr && s.entry->gpu_id != 0) {
                retired->push_back(s.entry->gpu_id);
            }
            if (evicted_key_hashes != nullptr) {
                evicted_key_hashes->push_back(s.hash);
            }
            if (last_entry_ == s.entry.get()) {
                last_entry_ = nullptr;
                last_hash_ = 0;
            }
            s.entry.reset();
            s.state = kBindSlotTombstone;
            --size_;
            ++tombstones_;
            ++dropped;
        }
    }
    if (size_ == 0 && tombstones_ > 0) {
        for (Slot& s : table_) {
            s.state = kBindSlotEmpty;
            s.hash = 0;
        }
        tombstones_ = 0;
    } else if (tombstones_ > size_ && !table_.empty() && (size_ + tombstones_) * 10 >= table_.size() * 7) {
        rehash(table_.size());
    }
    if (dropped != 0 && filter_notes_ > 2 * size_ + 256) {
        rebuild_id_filter();  // the dropped groups' bits are stale
    }
    track_size();
    return dropped;
}

size_t BindGroupCache::end_epoch(size_t max_entries, std::vector<uint32_t>* retired) {
    size_t dropped = 0;
    if (size_ > max_entries) {
        for (Slot& s : table_) {
            if (s.state == kBindSlotOccupied && s.entry->last_used_epoch < epoch_) {
                if (retired != nullptr && s.entry->gpu_id != 0) {
                    retired->push_back(s.entry->gpu_id);
                }
                if (last_entry_ == s.entry.get()) {
                    last_entry_ = nullptr;
                    last_hash_ = 0;
                }
                s.entry.reset();
                s.state = kBindSlotTombstone;
                --size_;
                ++tombstones_;
                ++dropped;
            }
        }
        if (size_ > max_entries) {
            dropped += size_;
            clear(retired);
        } else if (size_ == 0 && tombstones_ > 0) {
            for (Slot& s : table_) {
                s.state = kBindSlotEmpty;
                s.hash = 0;
            }
            tombstones_ = 0;
        } else if (tombstones_ > size_ && !table_.empty()) {
            rehash(table_.size());
        }
    }
    if (filter_notes_ > 2 * size_ + 256) {
        rebuild_id_filter();
    }
    track_size();
    peak_epoch_entries_ = size_;  // the next epoch's peak starts from what survived
    ++epoch_;
    return dropped;
}

// ===========================================================================
// Resource state tracking
// ===========================================================================

namespace {

bool valid_state_combination(uint32_t state) noexcept {
    const uint32_t writes = state & ResourceStates::WriteMask;
    if (writes == 0) {
        return true;
    }
    // A write state must stand alone (single bit, nothing else).
    return state == writes && std::has_single_bit(writes);
}

constexpr uint32_t kTexturePromotableStates = ResourceStates::NonPixelShaderResource |
                                               ResourceStates::PixelShaderResource |
                                               ResourceStates::CopyDest | ResourceStates::CopySource;

bool access_satisfied(uint32_t state, uint32_t required) noexcept {
    if (required == ResourceStates::Common) {
        return state == ResourceStates::Common;
    }
    return (state & required) == required;
}

}  // namespace

const char* state_diagnostic_name(StateDiagnosticCode code) noexcept {
    switch (code) {
    case StateDiagnosticCode::BeforeStateMismatch: return "BeforeStateMismatch";
    case StateDiagnosticCode::UnknownResource: return "UnknownResource";
    case StateDiagnosticCode::SubresourceOutOfRange: return "SubresourceOutOfRange";
    case StateDiagnosticCode::InvalidStateCombination: return "InvalidStateCombination";
    case StateDiagnosticCode::RedundantTransition: return "RedundantTransition";
    case StateDiagnosticCode::InvalidBarrierFlags: return "InvalidBarrierFlags";
    case StateDiagnosticCode::SplitBarrierAlreadyBegun: return "SplitBarrierAlreadyBegun";
    case StateDiagnosticCode::SplitBarrierWithoutBegin: return "SplitBarrierWithoutBegin";
    case StateDiagnosticCode::SplitBarrierMismatch: return "SplitBarrierMismatch";
    case StateDiagnosticCode::UseDuringSplitBarrier: return "UseDuringSplitBarrier";
    case StateDiagnosticCode::AccessStateMismatch: return "AccessStateMismatch";
    case StateDiagnosticCode::ExecuteStateMismatch: return "ExecuteStateMismatch";
    case StateDiagnosticCode::InvalidBarrierType: return "InvalidBarrierType";
    }
    return "Unknown";
}

// --- ResourceStateRegistry -------------------------------------------------

Status ResourceStateRegistry::register_resource(uint32_t resource_id, const ResourceStateDesc& desc) {
    if (resource_id == 0 || desc.subresource_count == 0) {
        return Status::InvalidArgument;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    GlobalResource r;
    r.subresource_count = desc.subresource_count;
    r.implicit_promotion = desc.implicit_promotion;
    r.uniform_state = desc.initial_state;
    if (!resources_.emplace(resource_id, std::move(r)).second) {
        return Status::ResourceAlreadyRegistered;
    }
    return Status::Ok;
}

void ResourceStateRegistry::unregister_resource(uint32_t resource_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    resources_.erase(resource_id);
}

bool ResourceStateRegistry::contains(uint32_t resource_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return resources_.count(resource_id) != 0;
}

uint32_t ResourceStateRegistry::state(uint32_t resource_id, uint32_t subresource) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = resources_.find(resource_id);
    if (it == resources_.end()) {
        return kStateUnknown;
    }
    const GlobalResource& r = it->second;
    if (subresource == kAllSubresources) {
        if (r.states.empty()) {
            return r.uniform_state;
        }
        for (uint32_t s : r.states) {
            if (s != r.states[0]) {
                return kStateUnknown;
            }
        }
        return r.states[0];
    }
    if (subresource >= r.subresource_count) {
        return kStateUnknown;
    }
    return r.states.empty() ? r.uniform_state : r.states[subresource];
}

bool ResourceStateRegistry::describe(uint32_t resource_id, uint32_t* subresource_count, bool* implicit_promotion) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = resources_.find(resource_id);
    if (it == resources_.end()) {
        return false;
    }
    if (subresource_count != nullptr) {
        *subresource_count = it->second.subresource_count;
    }
    if (implicit_promotion != nullptr) {
        *implicit_promotion = it->second.implicit_promotion;
    }
    return true;
}

void ResourceStateRegistry::execute(const CommandListStateTracker& list, std::vector<StateDiagnostic>* diagnostics) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto report = [&](StateDiagnosticCode code, uint32_t id, uint32_t sub, uint32_t expected, uint32_t actual) {
        if (diagnostics != nullptr) {
            diagnostics->push_back(StateDiagnostic{code, id, sub, expected, actual, kNoIndex});
        }
    };
    for (const CommandListStateTracker::LocalResource& local : list.resources_) {
        auto it = resources_.find(local.id);
        if (it == resources_.end()) {
            report(StateDiagnosticCode::UnknownResource, local.id, kAllSubresources, 0, 0);
            continue;
        }
        GlobalResource& g = it->second;
        const uint32_t count = std::min(local.subresource_count, g.subresource_count);
        const bool local_uniform = local.uniform;
        if (local_uniform && g.states.empty()) {
            // Fast path: both uniform.
            const CommandListStateTracker::SubState& s = local.all;
            uint32_t final_state = g.uniform_state;
            bool promoted = false;
            if (s.expected != kStateUnknown) {
                const uint32_t gs = g.uniform_state;
                bool ok = s.expected_is_access ? access_satisfied(gs, s.expected) : gs == s.expected;
                if (!ok && gs == ResourceStates::Common &&
                    (g.implicit_promotion || (s.expected_is_access && (s.expected & ~kTexturePromotableStates) == 0))) {
                    ok = true;
                    promoted = true;
                }
                if (!ok) {
                    report(StateDiagnosticCode::ExecuteStateMismatch, local.id, kAllSubresources, s.expected, gs);
                }
            }
            if (s.state != kStateUnknown) {
                final_state = s.split_after != kStateUnknown ? s.split_after : s.state;
                if (g.implicit_promotion || (promoted && s.state == s.expected)) {
                    final_state = ResourceStates::Common;
                }
            }
            g.uniform_state = final_state;
            continue;
        }
        if (g.states.empty()) {
            g.states.assign(g.subresource_count, g.uniform_state);
        }
        for (uint32_t sub = 0; sub < count; ++sub) {
            const CommandListStateTracker::SubState& s = local_uniform ? local.all : local.subs[sub];
            uint32_t& gs = g.states[sub];
            bool promoted = false;
            if (s.expected != kStateUnknown) {
                bool ok = s.expected_is_access ? access_satisfied(gs, s.expected) : gs == s.expected;
                if (!ok && gs == ResourceStates::Common &&
                    (g.implicit_promotion || (s.expected_is_access && (s.expected & ~kTexturePromotableStates) == 0))) {
                    ok = true;
                    promoted = true;
                }
                if (!ok) {
                    report(StateDiagnosticCode::ExecuteStateMismatch, local.id, sub, s.expected, gs);
                }
            }
            if (s.state != kStateUnknown) {
                gs = s.split_after != kStateUnknown ? s.split_after : s.state;
                if (g.implicit_promotion || (promoted && s.state == s.expected)) {
                    gs = ResourceStates::Common;
                }
            }
        }
        bool all_equal = true;
        for (uint32_t v : g.states) {
            all_equal = all_equal && v == g.states[0];
        }
        if (all_equal) {
            g.uniform_state = g.states[0];
            g.states.clear();
            g.states.shrink_to_fit();
        }
    }
}

// --- CommandListStateTracker -----------------------------------------------

namespace {

constexpr size_t kTrackerIndexInitialCapacity = 64;

inline uint32_t hash_resource_id(uint32_t x) noexcept {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

}  // namespace

uint32_t CommandListStateTracker::index_find(uint32_t key) const noexcept {
    if (index_table_.empty()) {
        return kNoIndex;
    }
    const size_t mask = index_table_.size() - 1;
    size_t idx = hash_resource_id(key) & mask;
    for (;;) {
        const IndexSlot& s = index_table_[idx];
        if (s.value == kNoIndex) {
            return kNoIndex;
        }
        if (s.key == key) {
            return s.value;
        }
        idx = (idx + 1) & mask;
    }
}

void CommandListStateTracker::index_insert(uint32_t key, uint32_t value) {
    if (index_table_.empty()) {
        index_table_.resize(kTrackerIndexInitialCapacity);
    } else if ((index_size_ + 1) * 10 >= index_table_.size() * 7) {
        const size_t new_cap = index_table_.size() * 2;
        std::vector<IndexSlot> next(new_cap);
        const size_t mask = new_cap - 1;
        for (const IndexSlot& s : index_table_) {
            if (s.value == kNoIndex) {
                continue;
            }
            size_t idx = hash_resource_id(s.key) & mask;
            while (next[idx].value != kNoIndex) {
                idx = (idx + 1) & mask;
            }
            next[idx] = s;
        }
        index_table_ = std::move(next);
    }
    const size_t mask = index_table_.size() - 1;
    size_t idx = hash_resource_id(key) & mask;
    while (index_table_[idx].value != kNoIndex) {
        idx = (idx + 1) & mask;
    }
    index_table_[idx] = IndexSlot{key, value};
    ++index_size_;
}

void CommandListStateTracker::reset() {
    resources_.clear();
    if (index_size_ != 0) {
        for (IndexSlot& s : index_table_) {
            s.value = kNoIndex;
        }
        index_size_ = 0;
    }
    diagnostics_.clear();
    dropped_diagnostics_ = 0;
    uav_barriers_ = 0;
    aliasing_barriers_ = 0;
}

void CommandListStateTracker::diag(
    StateDiagnosticCode code,
    uint32_t id,
    uint32_t sub,
    uint32_t expected,
    uint32_t actual,
    uint32_t index) {
    if (diagnostics_.size() >= kMaxDiagnostics) {
        ++dropped_diagnostics_;
        return;
    }
    diagnostics_.push_back(StateDiagnostic{code, id, sub, expected, actual, index});
}

CommandListStateTracker::LocalResource* CommandListStateTracker::touch(uint32_t resource_id, uint32_t barrier_index) {
    const uint32_t existing = index_find(resource_id);
    if (existing != kNoIndex) {
        return &resources_[existing];
    }
    uint32_t count = 0;
    bool promotion = false;
    if (registry_ == nullptr || !registry_->describe(resource_id, &count, &promotion)) {
        diag(StateDiagnosticCode::UnknownResource, resource_id, kAllSubresources, 0, 0, barrier_index);
        return nullptr;
    }
    LocalResource r;
    r.id = resource_id;
    r.subresource_count = count;
    r.implicit_promotion = promotion;
    index_insert(resource_id, static_cast<uint32_t>(resources_.size()));
    resources_.push_back(std::move(r));
    return &resources_.back();
}

const CommandListStateTracker::LocalResource* CommandListStateTracker::find(uint32_t resource_id) const {
    const uint32_t existing = index_find(resource_id);
    return existing == kNoIndex ? nullptr : &resources_[existing];
}

const CommandListStateTracker::SubState* CommandListStateTracker::sub_state(
    uint32_t resource_id,
    uint32_t subresource) const {
    const LocalResource* r = find(resource_id);
    if (r == nullptr) {
        return nullptr;
    }
    if (r->uniform) {
        return (subresource == kAllSubresources || subresource < r->subresource_count) ? &r->all : nullptr;
    }
    if (subresource == kAllSubresources || subresource >= r->subresource_count) {
        return nullptr;
    }
    return &r->subs[subresource];
}

uint32_t CommandListStateTracker::state(uint32_t resource_id, uint32_t subresource) const {
    const SubState* s = sub_state(resource_id, subresource);
    return s == nullptr ? kStateUnknown : s->state;
}

uint32_t CommandListStateTracker::expected_initial_state(uint32_t resource_id, uint32_t subresource) const {
    const SubState* s = sub_state(resource_id, subresource);
    return s == nullptr ? kStateUnknown : s->expected;
}

bool CommandListStateTracker::in_split_barrier(uint32_t resource_id, uint32_t subresource) const {
    const SubState* s = sub_state(resource_id, subresource);
    return s != nullptr && s->split_after != kStateUnknown;
}

bool CommandListStateTracker::is_uniform(uint32_t resource_id) const {
    const LocalResource* r = find(resource_id);
    return r == nullptr || r->uniform;
}

void CommandListStateTracker::maybe_collapse(LocalResource& r) {
    if (r.uniform) {
        return;
    }
    for (const SubState& s : r.subs) {
        if (!(s == r.subs[0])) {
            return;
        }
    }
    r.all = r.subs[0];
    r.subs.clear();
    r.uniform = true;
}

void CommandListStateTracker::transition_sub(
    LocalResource& r,
    SubState& s,
    uint32_t sub,
    const ResourceBarrier& b,
    uint32_t index) {
    const uint32_t before = b.state_before;
    const uint32_t after = b.state_after;
    if ((b.flags & ResourceBarrierFlags::EndOnly) != 0) {
        if (s.split_after == kStateUnknown) {
            if (s.state == kStateUnknown) {
                // Split begun in an earlier command list: the queue state is `after` by now.
                s.expected = after;
                s.expected_is_access = 0;
            } else {
                diag(StateDiagnosticCode::SplitBarrierWithoutBegin, r.id, sub, s.state, before, index);
            }
        } else if (s.split_after != after) {
            diag(StateDiagnosticCode::SplitBarrierMismatch, r.id, sub, s.split_after, after, index);
        }
        s.state = after;
        s.split_after = kStateUnknown;
        return;
    }
    const bool begin = (b.flags & ResourceBarrierFlags::BeginOnly) != 0;
    if (s.split_after != kStateUnknown) {
        diag(begin ? StateDiagnosticCode::SplitBarrierAlreadyBegun : StateDiagnosticCode::UseDuringSplitBarrier,
             r.id, sub, s.split_after, before, index);
        s.state = s.split_after;
        s.split_after = kStateUnknown;
    }
    if (s.state == kStateUnknown) {
        s.expected = before;
        s.expected_is_access = 0;
        s.state = before;
    } else if (s.state != before && !(r.implicit_promotion && s.state == ResourceStates::Common)) {
        diag(StateDiagnosticCode::BeforeStateMismatch, r.id, sub, s.state, before, index);
    }
    if (before == after) {
        diag(StateDiagnosticCode::RedundantTransition, r.id, sub, before, after, index);
    }
    if (begin) {
        s.state = before;
        s.split_after = after;
    } else {
        s.state = after;
    }
}

void CommandListStateTracker::resource_barrier(uint32_t count, const ResourceBarrier* barriers) {
    if (barriers == nullptr) {
        return;
    }
    for (uint32_t k = 0; k < count; ++k) {
        const ResourceBarrier& b = barriers[k];
        switch (b.type) {
        case ResourceBarrierType::Transition: {
            if (b.flags > (ResourceBarrierFlags::BeginOnly | ResourceBarrierFlags::EndOnly) ||
                b.flags == (ResourceBarrierFlags::BeginOnly | ResourceBarrierFlags::EndOnly)) {
                diag(StateDiagnosticCode::InvalidBarrierFlags, b.resource_id, b.subresource, 0, b.flags, k);
                break;
            }
            if (!valid_state_combination(b.state_before) || !valid_state_combination(b.state_after)) {
                diag(StateDiagnosticCode::InvalidStateCombination, b.resource_id, b.subresource, b.state_before,
                     b.state_after, k);
            }
            LocalResource* r = touch(b.resource_id, k);
            if (r == nullptr) {
                break;
            }
            if (b.subresource == kAllSubresources) {
                if (r->uniform) {
                    transition_sub(*r, r->all, kAllSubresources, b, k);
                } else {
                    for (uint32_t sub = 0; sub < r->subresource_count; ++sub) {
                        transition_sub(*r, r->subs[sub], sub, b, k);
                    }
                    maybe_collapse(*r);
                }
                break;
            }
            if (b.subresource >= r->subresource_count) {
                diag(StateDiagnosticCode::SubresourceOutOfRange, b.resource_id, b.subresource, r->subresource_count,
                     b.subresource, k);
                break;
            }
            if (r->subresource_count == 1) {
                transition_sub(*r, r->all, 0, b, k);
                break;
            }
            if (r->uniform) {
                r->subs.assign(r->subresource_count, r->all);
                r->uniform = false;
            }
            transition_sub(*r, r->subs[b.subresource], b.subresource, b, k);
            maybe_collapse(*r);
            break;
        }
        case ResourceBarrierType::Uav:
            ++uav_barriers_;
            if (b.flags != ResourceBarrierFlags::None) {
                diag(StateDiagnosticCode::InvalidBarrierFlags, b.resource_id, kAllSubresources, 0, b.flags, k);
            }
            if (b.resource_id != 0) {
                touch(b.resource_id, k);
            }
            break;
        case ResourceBarrierType::Aliasing:
            ++aliasing_barriers_;
            if (b.flags != ResourceBarrierFlags::None) {
                diag(StateDiagnosticCode::InvalidBarrierFlags, b.resource_after, kAllSubresources, 0, b.flags, k);
            }
            if (b.resource_before != 0) {
                touch(b.resource_before, k);
            }
            if (b.resource_after != 0) {
                touch(b.resource_after, k);
            }
            break;
        default:
            diag(StateDiagnosticCode::InvalidBarrierType, b.resource_id, b.subresource, 0,
                 static_cast<uint32_t>(b.type), k);
            break;
        }
    }
}

void CommandListStateTracker::access_sub(LocalResource& r, SubState& s, uint32_t sub, uint32_t required) {
    if (s.split_after != kStateUnknown) {
        diag(StateDiagnosticCode::UseDuringSplitBarrier, r.id, sub, s.split_after, required, kNoIndex);
        return;
    }
    if (s.state == kStateUnknown) {
        s.expected = required;
        s.expected_is_access = 1;
        s.state = required;
        return;
    }
    if (access_satisfied(s.state, required)) {
        return;
    }
    if (s.state == ResourceStates::Common &&
        (r.implicit_promotion || (required & ~kTexturePromotableStates) == 0)) {
        s.state = required;  // implicit promotion
        return;
    }
    diag(StateDiagnosticCode::AccessStateMismatch, r.id, sub, s.state, required, kNoIndex);
}

void CommandListStateTracker::note_access(uint32_t resource_id, uint32_t subresource, uint32_t required_states) {
    LocalResource* r = touch(resource_id, kNoIndex);
    if (r == nullptr) {
        return;
    }
    if (subresource == kAllSubresources) {
        if (r->uniform) {
            access_sub(*r, r->all, kAllSubresources, required_states);
        } else {
            for (uint32_t sub = 0; sub < r->subresource_count; ++sub) {
                access_sub(*r, r->subs[sub], sub, required_states);
            }
            maybe_collapse(*r);
        }
        return;
    }
    if (subresource >= r->subresource_count) {
        diag(StateDiagnosticCode::SubresourceOutOfRange, resource_id, subresource, r->subresource_count, subresource,
             kNoIndex);
        return;
    }
    if (r->subresource_count == 1) {
        access_sub(*r, r->all, 0, required_states);
        return;
    }
    if (r->uniform) {
        r->subs.assign(r->subresource_count, r->all);
        r->uniform = false;
    }
    access_sub(*r, r->subs[subresource], subresource, required_states);
    maybe_collapse(*r);
}

// ===========================================================================
// UsageScope
// ===========================================================================

namespace {

bool read_only(ResourceUsage u) noexcept {
    return u == ResourceUsage::Sampled || u == ResourceUsage::StorageRead || u == ResourceUsage::DepthStencilRead;
}

bool spans_overlap(uint32_t a_first, uint32_t a_count, uint32_t b_first, uint32_t b_count) noexcept {
    const uint64_t a_end = a_count == kAllMips ? std::numeric_limits<uint64_t>::max()
                                               : static_cast<uint64_t>(a_first) + a_count;
    const uint64_t b_end = b_count == kAllMips ? std::numeric_limits<uint64_t>::max()
                                               : static_cast<uint64_t>(b_first) + b_count;
    return a_first < b_end && b_first < a_end;
}

bool ranges_overlap(const SubresourceRange& a, const SubresourceRange& b) noexcept {
    return spans_overlap(a.first_mip, a.mip_count, b.first_mip, b.mip_count) &&
           spans_overlap(a.first_slice, a.slice_count, b.first_slice, b.slice_count) &&
           spans_overlap(a.first_plane, a.plane_count, b.first_plane, b.plane_count);
}

}  // namespace

SubresourceRange view_subresource_range(const ViewDescriptor& view) noexcept {
    SubresourceRange r;  // default: everything
    const uint32_t dim = view.dimension;
    auto mips = [&](uint32_t first, uint32_t count) {
        r.first_mip = first;
        r.mip_count = count;
    };
    auto slices = [&](uint32_t first, uint32_t count) {
        r.first_slice = first;
        r.slice_count = count;
    };
    switch (view.kind) {
    case DescriptorKind::Srv: {
        const uint32_t mip_count = view.mip_levels == 0xffffu ? kAllMips : view.mip_levels;
        switch (dim) {
        case ViewDimension::Texture1D:
        case ViewDimension::Texture2D:
            mips(view.first_mip, mip_count);
            slices(0, 1);
            r.first_plane = view.plane_slice;
            r.plane_count = 1;
            break;
        case ViewDimension::Texture1DArray:
        case ViewDimension::Texture2DArray:
            mips(view.first_mip, mip_count);
            slices(view.first_array_slice, view.array_size);
            r.first_plane = view.plane_slice;
            r.plane_count = 1;
            break;
        case ViewDimension::Texture2DMs:
            mips(0, 1);
            slices(0, 1);
            break;
        case ViewDimension::Texture2DMsArray:
            mips(0, 1);
            slices(view.first_array_slice, view.array_size);
            break;
        case ViewDimension::Texture3D:
            mips(view.first_mip, mip_count);
            break;
        case ViewDimension::TextureCube:
            mips(view.first_mip, mip_count);
            slices(0, 6);
            break;
        case ViewDimension::TextureCubeArray:
            mips(view.first_mip, mip_count);
            slices(view.first_array_slice,
                   view.array_size >= kAllMips / 6 ? kAllMips : view.array_size * 6);
            break;
        default:
            break;
        }
        break;
    }
    case DescriptorKind::Uav:
    case DescriptorKind::Rtv:
        switch (dim) {
        case ViewDimension::Texture1D:
        case ViewDimension::Texture2D:
            mips(view.first_mip, 1);
            slices(0, 1);
            r.first_plane = view.plane_slice;
            r.plane_count = 1;
            break;
        case ViewDimension::Texture1DArray:
        case ViewDimension::Texture2DArray:
            mips(view.first_mip, 1);
            slices(view.first_array_slice, view.array_size);
            r.first_plane = view.plane_slice;
            r.plane_count = 1;
            break;
        case ViewDimension::Texture2DMs:
            mips(0, 1);
            slices(0, 1);
            break;
        case ViewDimension::Texture2DMsArray:
            mips(0, 1);
            slices(view.first_array_slice, view.array_size);
            break;
        case ViewDimension::Texture3D:
            mips(view.first_mip, 1);
            break;
        default:
            break;
        }
        break;
    case DescriptorKind::Dsv:
        switch (dim) {
        case DsvDimension::Texture1D:
        case DsvDimension::Texture2D:
            mips(view.first_mip, 1);
            slices(0, 1);
            break;
        case DsvDimension::Texture1DArray:
        case DsvDimension::Texture2DArray:
            mips(view.first_mip, 1);
            slices(view.first_array_slice, view.array_size);
            break;
        case DsvDimension::Texture2DMs:
            mips(0, 1);
            slices(0, 1);
            break;
        case DsvDimension::Texture2DMsArray:
            mips(0, 1);
            slices(view.first_array_slice, view.array_size);
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
    return r;
}

ResourceUsage view_usage(const ViewDescriptor& view) noexcept {
    switch (view.kind) {
    case DescriptorKind::Uav:
        return ResourceUsage::StorageWrite;
    case DescriptorKind::Rtv:
        return ResourceUsage::RenderTarget;
    case DescriptorKind::Dsv:
        return (view.flags & DsvFlags::ReadOnlyDepth) != 0 ? ResourceUsage::DepthStencilRead
                                                           : ResourceUsage::DepthStencilWrite;
    default:
        return ResourceUsage::Sampled;
    }
}

void UsageScope::add(uint32_t resource_id, const SubresourceRange& range, ResourceUsage usage) {
    usages_.push_back(Usage{resource_id, static_cast<uint32_t>(usages_.size()), range, usage});
}

size_t UsageScope::find_conflicts(std::vector<UsageConflict>* out) {
    sorted_ = usages_;
    std::sort(sorted_.begin(), sorted_.end(), [](const Usage& a, const Usage& b) {
        return a.resource_id != b.resource_id ? a.resource_id < b.resource_id : a.index < b.index;
    });
    size_t found = 0;
    for (size_t i = 0; i < sorted_.size(); ++i) {
        for (size_t j = i + 1; j < sorted_.size() && sorted_[j].resource_id == sorted_[i].resource_id; ++j) {
            const Usage& a = sorted_[i];
            const Usage& b = sorted_[j];
            const bool compatible = (read_only(a.usage) && read_only(b.usage)) ||
                                    (a.usage == ResourceUsage::StorageWrite && b.usage == ResourceUsage::StorageWrite);
            if (compatible || !ranges_overlap(a.range, b.range)) {
                continue;
            }
            ++found;
            if (out != nullptr) {
                out->push_back(UsageConflict{a.resource_id, a.usage, b.usage, a.index, b.index});
            }
        }
    }
    return found;
}

}  // namespace playgta6::gpu::d3d12_state
