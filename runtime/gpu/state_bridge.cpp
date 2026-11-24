#include "gpu/state_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace playgta6::gpu::bridge {

// =====================================================================================================
// Seam: the three type families must agree wherever values cross without conversion.
// =====================================================================================================

static_assert(static_cast<uint32_t>(d3d::BindResourceKind::Buffer) == static_cast<uint32_t>(GpuBindResourceType::Buffer));
static_assert(static_cast<uint32_t>(d3d::BindResourceKind::TextureView) ==
              static_cast<uint32_t>(GpuBindResourceType::TextureView));
static_assert(static_cast<uint32_t>(d3d::BindResourceKind::Sampler) == static_cast<uint32_t>(GpuBindResourceType::Sampler));
static_assert(d3d::DsvFlags::ReadOnlyDepth == pso::D3D12DsvFlags::ReadOnlyDepth);
static_assert(d3d::DsvFlags::ReadOnlyStencil == pso::D3D12DsvFlags::ReadOnlyStencil);
static_assert(d3d::WebGpuStage::Vertex == GpuShaderStageBits::Vertex);
static_assert(d3d::WebGpuStage::Fragment == GpuShaderStageBits::Fragment);
static_assert(d3d::WebGpuStage::Compute == GpuShaderStageBits::Compute);
static_assert(static_cast<uint8_t>(pso::WgpuLoadOp::Load) == static_cast<uint8_t>(GpuLoadOp::Load));
static_assert(static_cast<uint8_t>(pso::WgpuLoadOp::Clear) == static_cast<uint8_t>(GpuLoadOp::Clear));
static_assert(static_cast<uint8_t>(pso::WgpuStoreOp::Store) == static_cast<uint8_t>(GpuStoreOp::Store));
static_assert(static_cast<uint8_t>(pso::WgpuStoreOp::Discard) == static_cast<uint8_t>(GpuStoreOp::Discard));
static_assert(static_cast<uint8_t>(pso::WgpuTextureAspect::DepthOnly) == static_cast<uint8_t>(GpuTextureAspect::DepthOnly));
static_assert(static_cast<uint8_t>(pso::WgpuTextureAspect::StencilOnly) ==
              static_cast<uint8_t>(GpuTextureAspect::StencilOnly));
static_assert(sizeof(GpuRenderPipelineKey) == sizeof(pso::GraphicsPipelineKey));
static_assert(sizeof(GpuComputePipelineKey) == sizeof(pso::ComputePipelineKey));
static_assert(pso::kD3D12SimultaneousRenderTargetCount == kGpuMaxColorAttachments);
static_assert(static_cast<uint32_t>(pso::D3D12ComparisonFunc::Always) == 8);  // D3D12_COMPARISON_FUNC

namespace {

constexpr uint32_t kRootConstantAlignment = 256;  // minUniformBufferOffsetAlignment (default limit)

uint32_t align_up(uint32_t value, uint32_t alignment) noexcept {
    return (value + alignment - 1u) / alignment * alignment;
}

template <typename T>
std::string bytes_key(const T& value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(T));
}

uint32_t mip_extent(uint32_t extent, uint32_t mip) noexcept {
    return mip >= 32 ? 1u : std::max(1u, extent >> mip);
}

uint8_t texture_format_code(pso::DxgiFormat format, pso::FormatUsage usage, uint32_t features,
                            pso::WgpuTextureAspect* aspect) {
    const pso::TextureFormatMapping m = pso::map_texture_format(format, usage, features);
    if (aspect != nullptr) {
        *aspect = m.aspect;
    }
    return static_cast<uint8_t>(m.format);
}

GpuAddressMode to_address_mode(uint32_t d3d12_mode, bool* exact) noexcept {
    switch (d3d12_mode) {
    case 1: return GpuAddressMode::Repeat;        // WRAP
    case 2: return GpuAddressMode::MirrorRepeat;  // MIRROR
    case 3: return GpuAddressMode::ClampToEdge;   // CLAMP
    case 4:                                       // BORDER: no border colors in WebGPU
        *exact = false;
        return GpuAddressMode::ClampToEdge;
    case 5:  // MIRROR_ONCE
        *exact = false;
        return GpuAddressMode::MirrorRepeat;
    default:
        *exact = false;
        return GpuAddressMode::ClampToEdge;
    }
}

}  // namespace

pso::WgpuIndexFormat to_wgpu_index_format(GpuIndexFormat format) noexcept {
    return format == GpuIndexFormat::Uint32 ? pso::WgpuIndexFormat::Uint32 : pso::WgpuIndexFormat::Uint16;
}

GpuTextureViewDimension to_view_dimension(uint32_t dim, bool* supported) noexcept {
    namespace V = d3d::ViewDimension;
    *supported = true;
    switch (dim) {
    case V::Unknown: return GpuTextureViewDimension::Undefined;
    case V::Texture1D: return GpuTextureViewDimension::Dim1D;
    case V::Texture2D:
    case V::Texture2DMs: return GpuTextureViewDimension::Dim2D;
    case V::Texture2DArray: return GpuTextureViewDimension::Dim2DArray;
    case V::Texture3D: return GpuTextureViewDimension::Dim3D;
    case V::TextureCube: return GpuTextureViewDimension::Cube;
    case V::TextureCubeArray: return GpuTextureViewDimension::CubeArray;
    default:  // 1D arrays and multisampled arrays have no WebGPU view dimension
        *supported = false;
        return GpuTextureViewDimension::Undefined;
    }
}

GpuCreateSampler2Payload to_sampler_payload(const d3d::SamplerDesc& desc, bool* exact) noexcept {
    bool ok = true;
    GpuCreateSampler2Payload p{};
    p.address_u = static_cast<uint8_t>(to_address_mode(desc.address_u, &ok));
    p.address_v = static_cast<uint8_t>(to_address_mode(desc.address_v, &ok));
    p.address_w = static_cast<uint8_t>(to_address_mode(desc.address_w, &ok));
    // D3D12_FILTER: mip = bits 0..1, mag = bits 2..3, min = bits 4..5, anisotropic = 0x40,
    // reduction = bits 7..8 (0 standard, 1 comparison, 2 minimum, 3 maximum).
    const uint32_t f = desc.filter;
    const bool anisotropic = (f & 0x40u) != 0;
    const uint32_t reduction = (f >> 7) & 0x3u;
    p.mipmap_filter = static_cast<uint8_t>(anisotropic ? 1u : (f & 0x1u));
    p.mag_filter = static_cast<uint8_t>(anisotropic ? 1u : ((f >> 2) & 0x1u));
    p.min_filter = static_cast<uint8_t>(anisotropic ? 1u : ((f >> 4) & 0x1u));
    p.max_anisotropy = static_cast<uint16_t>(anisotropic ? std::clamp<uint32_t>(desc.max_anisotropy, 1u, 16u) : 1u);
    if (reduction == 1) {
        const auto cmp = pso::map_compare_function(static_cast<pso::D3D12ComparisonFunc>(desc.comparison_func));
        p.compare = static_cast<uint8_t>(cmp ? *cmp : pso::WgpuCompareFunction::Never);
        ok = ok && cmp.has_value();
    } else if (reduction != 0) {
        ok = false;  // min/max reduction filters have no WebGPU equivalent
    }
    if (desc.mip_lod_bias != 0.0f) {
        ok = false;
    }
    const float lod_min = std::isfinite(desc.min_lod) ? std::clamp(desc.min_lod, 0.0f, 32.0f) : 0.0f;
    const float lod_max = std::isfinite(desc.max_lod) ? std::clamp(desc.max_lod, lod_min, 32.0f) : 32.0f;
    p.lod_min_clamp = lod_min;
    p.lod_max_clamp = std::max(lod_min, lod_max);
    if (exact != nullptr) {
        *exact = ok;
    }
    return p;
}

d3d::SamplerDesc sampler_for_binding(const d3d::SamplerDesc& desc, uint8_t layout_sampler_type, bool* changed) noexcept {
    d3d::SamplerDesc out = desc;
    const bool comparison = ((desc.filter >> 7) & 0x3u) == 1u;
    const bool drop = comparison && layout_sampler_type != static_cast<uint8_t>(GpuSamplerBindingType::Comparison);
    if (drop) {
        out.filter &= ~(0x3u << 7);
    }
    if (changed != nullptr) {
        *changed = drop;
    }
    return out;
}

// =====================================================================================================
// StateBridgeDevice
// =====================================================================================================

StateBridgeDevice::StateBridgeDevice(d3d::DescriptorHeapRegistry& heaps,
                                     BridgeOptions options,
                                     const d3d::ResourceStateRegistry* resource_states)
    : heaps_(heaps), options_(std::move(options)), resource_states_(resource_states) {
    if (!options_.id_allocator) {
        auto counter = std::make_shared<std::atomic<uint32_t>>(1u);
        options_.id_allocator = [counter] { return counter->fetch_add(1u, std::memory_order_relaxed); };
    }
    options_.root_constant_ring_bytes =
        std::max(align_up(options_.root_constant_ring_bytes, kRootConstantAlignment), kRootConstantAlignment);
    pso::PipelineCacheOptions cache_options;
    cache_options.id_allocator = options_.id_allocator;
    pipeline_cache_ = std::make_unique<pso::PipelineCache>(std::move(cache_options));
}

StateBridgeDevice::~StateBridgeDevice() = default;

void StateBridgeDevice::register_texture(uint32_t resource_id, const TextureInfo& info) {
    std::unique_lock lock(resources_mutex_);
    if (auto it = textures_.find(resource_id); it != textures_.end() && it->second.gpu_va != 0) {
        if (auto v = textures_by_va_.find(it->second.gpu_va); v != textures_by_va_.end() && v->second == resource_id) {
            textures_by_va_.erase(v);
        }
    }
    textures_[resource_id] = info;
    if (info.gpu_va != 0) {
        textures_by_va_[info.gpu_va] = resource_id;
    }
}

uint32_t StateBridgeDevice::texture_at(uint64_t gpu_va) const {
    std::shared_lock lock(resources_mutex_);
    auto it = textures_by_va_.find(gpu_va);
    return it != textures_by_va_.end() ? it->second : 0u;
}

void StateBridgeDevice::register_buffer(uint32_t resource_id, uint64_t gpu_va, uint64_t byte_size) {
    std::unique_lock lock(resources_mutex_);
    if (auto it = buffer_va_.find(resource_id); it != buffer_va_.end()) {
        buffers_by_va_.erase(it->second);
    }
    buffers_by_va_[gpu_va] = {resource_id, byte_size};
    buffer_va_[resource_id] = gpu_va;
}

uint64_t StateBridgeDevice::buffer_gpu_va(uint32_t resource_id) const {
    std::shared_lock lock(resources_mutex_);
    auto it = buffer_va_.find(resource_id);
    return it != buffer_va_.end() ? it->second : 0;
}

void StateBridgeDevice::unregister_resource(uint32_t resource_id) {
    std::unique_lock lock(resources_mutex_);
    if (auto it = textures_.find(resource_id); it != textures_.end()) {
        if (auto v = textures_by_va_.find(it->second.gpu_va); v != textures_by_va_.end() && v->second == resource_id) {
            textures_by_va_.erase(v);
        }
        textures_.erase(it);
    }
    if (auto it = buffer_va_.find(resource_id); it != buffer_va_.end()) {
        buffers_by_va_.erase(it->second);
        buffer_va_.erase(it);
    }
}

void StateBridgeDevice::register_shader_module(uint64_t shader_identity, uint32_t module_id) {
    std::unique_lock lock(resources_mutex_);
    shader_modules_[shader_identity] = module_id;
}

bool StateBridgeDevice::texture_info(uint32_t resource_id, TextureInfo* out) const {
    std::shared_lock lock(resources_mutex_);
    const auto it = textures_.find(resource_id);
    if (it == textures_.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

uint32_t StateBridgeDevice::module_id(uint64_t shader_identity, bool* known) const {
    if (shader_identity == 0) {
        *known = true;
        return 0;
    }
    std::shared_lock lock(resources_mutex_);
    const auto it = shader_modules_.find(shader_identity);
    *known = it != shader_modules_.end();
    return *known ? it->second : 0u;
}

void StateBridgeDevice::register_pipeline_layout(uint64_t first_shader, uint64_t second_shader, PipelineLayoutDesc desc) {
    std::lock_guard lock(objects_mutex_);
    std::vector<GroupLayoutRecord*> groups;
    for (GroupLayoutDesc& g : desc.groups) {
        std::stable_sort(g.bindings.begin(), g.bindings.end(), [](const LayoutBinding& a, const LayoutBinding& b) {
            return a.entry.binding < b.entry.binding;
        });
        auto record = std::make_unique<GroupLayoutRecord>();
        bool modified = false;
        for (LayoutBinding b : g.bindings) {
            b.join.binding = b.entry.binding;
            // Root constants are uploaded to the shared ring and bound with a dynamic offset.
            if (b.join.root_constants && b.entry.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
                (b.entry.flags & GpuBindGroupLayoutEntryFlags::HasDynamicOffset) == 0) {
                b.entry.flags |= GpuBindGroupLayoutEntryFlags::HasDynamicOffset;
                modified = true;
            }
            record->entries.push_back(b.entry);
            record->joins.push_back(b.join);
        }
        // A layout that differs from the manifest's no longer carries the manifest id.
        record->header.content_id = modified ? 0 : g.content_id;
        record->header.entry_count = static_cast<uint32_t>(record->entries.size());

        GroupLayoutRecord* found = nullptr;
        for (const auto& existing : group_layouts_) {
            if (existing->header.content_id == record->header.content_id &&
                existing->entries.size() == record->entries.size() &&
                std::memcmp(existing->entries.data(), record->entries.data(),
                            record->entries.size() * sizeof(GpuBindGroupLayoutEntry)) == 0) {
                const bool same_joins = std::equal(
                    existing->joins.begin(), existing->joins.end(), record->joins.begin(),
                    [](const d3d::ShaderBinding& a, const d3d::ShaderBinding& b) {
                        return a.binding == b.binding && a.register_class == b.register_class &&
                               a.shader_register == b.shader_register && a.register_space == b.register_space &&
                               a.root_constants == b.root_constants;
                    });
                if (same_joins) {
                    found = existing.get();
                    break;
                }
            }
        }
        if (found == nullptr) {
            record->id = allocate_id();
            record->header.id = record->id;
            found = record.get();
            group_layouts_.push_back(std::move(record));
        }
        groups.push_back(found);
    }

    PipelineLayoutRecord* layout = nullptr;
    for (const auto& existing : pipeline_layouts_) {
        if (existing->groups == groups && existing->content_id == desc.content_id) {
            layout = existing.get();
            break;
        }
    }
    if (layout == nullptr) {
        auto record = std::make_unique<PipelineLayoutRecord>();
        record->id = allocate_id();
        record->content_id = desc.content_id;
        record->groups = std::move(groups);
        layout = record.get();
        pipeline_layouts_.push_back(std::move(record));
    }
    layouts_by_shaders_[{first_shader, second_shader}] = layout;
}

StateBridgeDevice::PipelineLayoutRecord* StateBridgeDevice::find_layout(uint64_t first_shader, uint64_t second_shader) {
    std::lock_guard lock(objects_mutex_);
    const auto it = layouts_by_shaders_.find({first_shader, second_shader});
    return it == layouts_by_shaders_.end() ? nullptr : it->second;
}

void StateBridgeDevice::ensure_layout_emitted(PipelineLayoutRecord* layout, CommandEncoder& encoder) {
    if (layout == nullptr) {
        return;
    }
    std::lock_guard lock(objects_mutex_);
    if (layout->emitted) {
        return;
    }
    std::vector<uint32_t> group_ids;
    for (GroupLayoutRecord* group : layout->groups) {
        if (!group->emitted) {
            encoder.emit_packet(GpuOpcode::CreateBindGroupLayout, &group->header, sizeof(group->header),
                                group->entries.data(),
                                static_cast<uint32_t>(group->entries.size() * sizeof(GpuBindGroupLayoutEntry)));
            group->emitted = true;
        }
        group_ids.push_back(group->id);
    }
    GpuCreatePipelineLayoutPayload header{};
    header.id = layout->id;
    header.bind_group_layout_count = static_cast<uint32_t>(group_ids.size());
    header.content_id = layout->content_id;
    encoder.emit_packet(GpuOpcode::CreatePipelineLayout, &header, sizeof(header), group_ids.data(),
                        static_cast<uint32_t>(group_ids.size() * sizeof(uint32_t)));
    layout->emitted = true;
}

uint32_t StateBridgeDevice::intern_sampler(const d3d::SamplerDesc& desc, CommandEncoder& encoder, uint64_t* diagnostics) {
    bool exact = true;
    GpuCreateSampler2Payload payload = to_sampler_payload(desc, &exact);
    if (!exact && diagnostics != nullptr) {
        *diagnostics |= BridgeDiag::SamplerApproximated;
    }
    const std::string key = bytes_key(payload);
    std::lock_guard lock(objects_mutex_);
    if (const auto it = samplers_.find(key); it != samplers_.end()) {
        return it->second;
    }
    payload.id = allocate_id();
    encoder.emit_packet(GpuOpcode::CreateSampler2, &payload, sizeof(payload));
    samplers_.emplace(key, payload.id);
    return payload.id;
}

uint32_t StateBridgeDevice::ensure_root_constant_ring(CommandEncoder& encoder) {
    std::lock_guard lock(objects_mutex_);
    if (ring_buffer_id_ == 0) {
        ring_buffer_id_ = allocate_id();
        encoder.create_buffer(ring_buffer_id_, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst,
                              options_.root_constant_ring_bytes);
    }
    return ring_buffer_id_;
}

uint32_t StateBridgeDevice::allocate_root_constants(uint32_t bytes, bool* wrapped) {
    std::lock_guard lock(objects_mutex_);
    const uint32_t size = align_up(std::max(bytes, 1u), kRootConstantAlignment);
    *wrapped = false;
    if (ring_cursor_ + size > options_.root_constant_ring_bytes) {
        ring_cursor_ = 0;
        *wrapped = true;
    }
    const uint32_t offset = ring_cursor_;
    ring_cursor_ += size;
    return offset;
}

bool StateBridgeDevice::claim_compute_creation(uint32_t pipeline_id) {
    std::lock_guard lock(objects_mutex_);
    return emitted_compute_.insert(pipeline_id).second;
}

uint32_t StateBridgeDevice::intern_view(const ViewRecord& record) const {
    std::string key;
    key.reserve(64);
    key.append(bytes_key(record.texture_id));
    key.append(bytes_key(record.resource_id));
    key.append(bytes_key(record.view));
    key.append(bytes_key(record.range));
    key.push_back(static_cast<char>(record.usage));
    std::lock_guard lock(views_mutex_);
    if (const auto it = view_ids_.find(key); it != view_ids_.end()) {
        return it->second;
    }
    views_.push_back(record);
    const auto id = static_cast<uint32_t>(views_.size());
    view_ids_.emplace(std::move(key), id);
    return id;
}

const ViewRecord* StateBridgeDevice::view(uint32_t view_id) const {
    std::lock_guard lock(views_mutex_);
    return (view_id == 0 || view_id > views_.size()) ? nullptr : &views_[view_id - 1];
}

bool StateBridgeDevice::resolve_view_descriptor(uint64_t cpu_handle, d3d::DescriptorKind kind,
                                                d3d::ViewDescriptor* out) const {
    d3d::ResolvedDescriptor resolved{};
    if (heaps_.resolve_cpu(d3d::CpuDescriptorHandle{cpu_handle}, &resolved) != d3d::Status::Ok ||
        resolved.heap == nullptr) {
        return false;
    }
    const d3d::ViewDescriptor* v = resolved.heap->view(resolved.index);
    if (v == nullptr || v->kind != kind) {
        return false;
    }
    *out = *v;
    return true;
}

uint64_t StateBridgeDevice::rtv_descriptor_increment() const {
    return d3d::descriptor_handle_increment_size(d3d::DescriptorHeapType::Rtv);
}

bool StateBridgeDevice::resolve_rtv(uint64_t cpu_handle, pso::RenderTargetViewInfo* out) const {
    d3d::ViewDescriptor v{};
    if (!resolve_view_descriptor(cpu_handle, d3d::DescriptorKind::Rtv, &v)) {
        return false;
    }
    *out = pso::RenderTargetViewInfo{};
    if (v.resource_id == 0) {
        return true;  // null descriptor: unbound slot
    }
    TextureInfo tex{};
    if (!texture_info(v.resource_id, &tex)) {
        return false;
    }
    const pso::DxgiFormat format = v.format != 0 ? to_dxgi(v.format) : tex.format;
    const uint32_t mip = v.first_mip;
    uint32_t layer = 0;
    switch (v.dimension) {
    case d3d::ViewDimension::Texture1DArray:
    case d3d::ViewDimension::Texture2DArray:
    case d3d::ViewDimension::Texture2DMsArray:
        layer = v.first_array_slice;
        break;
    default:
        break;
    }
    ViewRecord record{};
    record.resource_id = v.resource_id;
    record.texture_id = tex.swapchain ? kGpuSwapchainTextureId : v.resource_id;
    record.view.format = texture_format_code(format, pso::FormatUsage::RenderTarget, options_.features, nullptr);
    record.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
    record.view.base_mip_level = static_cast<uint16_t>(mip);
    record.view.mip_level_count = 1;
    record.view.base_array_layer = layer;
    record.view.array_layer_count = 1;
    record.range = d3d::SubresourceRange{mip, 1, layer, 1, 0, d3d::kAllMips};
    record.usage = d3d::ResourceUsage::RenderTarget;
    out->view_id = intern_view(record);
    out->format = format;
    out->width = mip_extent(tex.width, mip);
    out->height = mip_extent(tex.height, mip);
    out->sample_count = tex.sample_count;
    return true;
}

bool StateBridgeDevice::resolve_dsv(uint64_t cpu_handle, pso::DepthStencilViewInfo* out) const {
    d3d::ViewDescriptor v{};
    if (!resolve_view_descriptor(cpu_handle, d3d::DescriptorKind::Dsv, &v)) {
        return false;
    }
    *out = pso::DepthStencilViewInfo{};
    if (v.resource_id == 0) {
        return true;
    }
    TextureInfo tex{};
    if (!texture_info(v.resource_id, &tex)) {
        return false;
    }
    const pso::DxgiFormat format = v.format != 0 ? to_dxgi(v.format) : tex.format;
    const uint32_t mip = v.first_mip;
    uint32_t layer = 0;
    switch (v.dimension) {
    case d3d::DsvDimension::Texture1DArray:
    case d3d::DsvDimension::Texture2DArray:
    case d3d::DsvDimension::Texture2DMsArray:
        layer = v.first_array_slice;
        break;
    default:
        break;
    }
    const uint32_t flags = v.flags & (d3d::DsvFlags::ReadOnlyDepth | d3d::DsvFlags::ReadOnlyStencil);
    ViewRecord record{};
    record.resource_id = v.resource_id;
    record.texture_id = tex.swapchain ? kGpuSwapchainTextureId : v.resource_id;
    record.view.format = texture_format_code(format, pso::FormatUsage::DepthStencil, options_.features, nullptr);
    record.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
    record.view.base_mip_level = static_cast<uint16_t>(mip);
    record.view.mip_level_count = 1;
    record.view.base_array_layer = layer;
    record.view.array_layer_count = 1;
    record.range = d3d::SubresourceRange{mip, 1, layer, 1, 0, d3d::kAllMips};
    record.usage = (flags & d3d::DsvFlags::ReadOnlyDepth) != 0 ? d3d::ResourceUsage::DepthStencilRead
                                                               : d3d::ResourceUsage::DepthStencilWrite;
    out->view_id = intern_view(record);
    out->format = format;
    out->width = mip_extent(tex.width, mip);
    out->height = mip_extent(tex.height, mip);
    out->sample_count = tex.sample_count;
    out->flags = flags;
    return true;
}

bool StateBridgeDevice::resolve_gpu_va(uint64_t gpu_address, d3d::GpuVaResolution* out) const {
    std::shared_lock lock(resources_mutex_);
    auto it = buffers_by_va_.upper_bound(gpu_address);
    if (it == buffers_by_va_.begin()) {
        return false;
    }
    --it;
    const uint64_t base = it->first;
    const auto [resource_id, size] = it->second;
    if (gpu_address - base >= size) {
        return false;
    }
    out->resource_id = resource_id;
    out->offset = gpu_address - base;
    out->resource_size = size;
    return true;
}

bool StateBridgeDevice::gpu_va_trampoline(void* user, uint64_t gpu_address, d3d::GpuVaResolution* out) {
    return static_cast<const StateBridgeDevice*>(user)->resolve_gpu_va(gpu_address, out);
}

namespace {

uint32_t group_stage_mask(const std::vector<GpuBindGroupLayoutEntry>& entries, uint32_t pipeline_stages) {
    uint32_t mask = 0;
    for (const GpuBindGroupLayoutEntry& e : entries) {
        mask |= e.visibility;
    }
    mask &= pipeline_stages;
    return mask != 0 ? mask : pipeline_stages;
}

}  // namespace

std::unique_ptr<GraphicsPipelineState> StateBridgeDevice::create_graphics_pipeline_state(
    const pso::D3D12GraphicsPsoDesc& desc,
    const d3d::RootSignature* root_signature) {
    pso::PipelineKeyOptions key_options;
    key_options.features = options_.features;
    std::unique_ptr<GraphicsPipelineState> state(new GraphicsPipelineState(desc, key_options));
    state->root_signature_ = root_signature;
    bool vs_known = true;
    bool ps_known = true;
    state->vs_module_id_ = module_id(desc.vs, &vs_known);
    state->fs_module_id_ = module_id(desc.ps, &ps_known);
    if (!vs_known || !ps_known) {
        state->diagnostics_ |= BridgeDiag::UnknownShaderModule;
    }
    state->layout_ = find_layout(desc.vs, desc.ps);
    if (state->layout_ == nullptr) {
        state->diagnostics_ |= BridgeDiag::MissingPipelineLayout;
    } else if (root_signature == nullptr) {
        state->diagnostics_ |= BridgeDiag::BindGroupBuildFailed;
    } else {
        for (const GroupLayoutRecord* group : state->layout_->groups) {
            const uint32_t stages = group_stage_mask(group->entries, d3d::WebGpuStage::Vertex | d3d::WebGpuStage::Fragment);
            state->plans_.push_back(d3d::BindGroupPlan::build(*root_signature, group->joins, stages));
        }
    }
    return state;
}

std::unique_ptr<ComputePipelineState> StateBridgeDevice::create_compute_pipeline_state(
    const pso::D3D12ComputePsoDesc& desc,
    const d3d::RootSignature* root_signature) {
    std::unique_ptr<ComputePipelineState> state(new ComputePipelineState());
    state->key_ = pso::make_compute_pipeline_key(desc);
    state->root_signature_ = root_signature;
    bool known = true;
    state->cs_module_id_ = module_id(desc.cs, &known);
    if (!known || state->cs_module_id_ == 0) {
        state->diagnostics_ |= BridgeDiag::UnknownShaderModule;
    }
    state->layout_ = find_layout(desc.cs, 0);
    if (state->layout_ == nullptr) {
        state->diagnostics_ |= BridgeDiag::MissingPipelineLayout;
    } else if (root_signature == nullptr) {
        state->diagnostics_ |= BridgeDiag::BindGroupBuildFailed;
    } else {
        for (const GroupLayoutRecord* group : state->layout_->groups) {
            state->plans_.push_back(d3d::BindGroupPlan::build(*root_signature, group->joins, d3d::WebGpuStage::Compute));
        }
    }
    if (state->cs_module_id_ != 0) {
        state->pipeline_id_ = pipeline_cache_->acquire(state->key_).pipeline_id;
    }
    return state;
}

void StateBridgeDevice::release_pipeline_state(std::unique_ptr<GraphicsPipelineState> state, CommandEncoder& encoder) {
    if (state == nullptr) {
        return;
    }
    for (uint32_t id : state->pso_.release(*pipeline_cache_)) {
        encoder.destroy_resource(id);
    }
}

// =====================================================================================================
// CommandListBridge
// =====================================================================================================

namespace {

pso::TrackerOptions tracker_options(const BridgeOptions& options) {
    pso::TrackerOptions t;
    t.mid_pass_clear = options.mid_pass_clear;
    t.features = options.features;
    t.max_texture_dimension_2d = options.max_texture_dimension_2d;
    return t;
}

}  // namespace

CommandListBridge::CommandListBridge(StateBridgeDevice& device, CommandEncoder& encoder)
    : device_(device),
      encoder_(encoder),
      tracker_(tracker_options(device.options())),
      resource_states_(device.resource_states_) {
    bind_groups_.set_slot_source(device.options().slot_source);
    Reset();
}

void CommandListBridge::Reset(GraphicsPipelineState* initial_state) {
    tracker_.reset();
    root_args_.reset();
    resource_states_.reset();
    usage_scope_.clear();
    usage_conflicts_.clear();
    graphics_pso_ = nullptr;
    compute_pso_ = nullptr;
    vertex_buffers_ = {};
    vertex_dirty_ = 0;
    index_buffer_ = {};
    index_dirty_ = false;
    render_pass_open_ = false;
    render_groups_.clear();
    compute_groups_.clear();
    scoped_bind_groups_.clear();
    memory_groups_.clear();
    compute_pass_valid_ = false;
    compute_pipeline_bound_ = 0;
    for (auto& point : root_uploads_) {
        for (RootConstantUpload& upload : point) {
            upload.valid = false;
        }
    }
    diagnostics_ = 0;
    stats_ = BridgeStats{};
    open_ = true;
    if (initial_state != nullptr) {
        SetPipelineState(initial_state);
    }
}

bool CommandListBridge::writable() noexcept {
    if (!open_) {
        diagnostics_ |= BridgeDiag::CommandListClosed;
        return false;
    }
    return true;
}

void CommandListBridge::Close() {
    if (!writable()) {
        return;
    }
    tracker_.close();
    drain_events();
    open_ = false;
}

void CommandListBridge::SetPipelineState(GraphicsPipelineState* state) {
    if (!writable()) {
        return;
    }
    graphics_pso_ = state;
    tracker_.SetPipelineState(state != nullptr ? &state->pso_ : nullptr);
}

void CommandListBridge::SetComputePipelineState(ComputePipelineState* state) {
    if (!writable()) {
        return;
    }
    compute_pso_ = state;
}

void CommandListBridge::OMSetRenderTargets(uint32_t num_render_target_descriptors,
                                           const d3d::CpuDescriptorHandle* render_target_descriptors,
                                           bool rts_single_handle_to_descriptor_range,
                                           const d3d::CpuDescriptorHandle* depth_stencil_descriptor) {
    if (!writable()) {
        return;
    }
    std::array<uint64_t, pso::kMaxColorTargets> handles{};
    const uint32_t n = std::min(num_render_target_descriptors, pso::kMaxColorTargets);
    if (render_target_descriptors != nullptr) {
        const uint32_t copy = rts_single_handle_to_descriptor_range ? std::min(n, 1u) : n;
        for (uint32_t i = 0; i < copy; ++i) {
            handles[i] = render_target_descriptors[i].ptr;
        }
    }
    const uint64_t dsv = depth_stencil_descriptor != nullptr ? depth_stencil_descriptor->ptr : 0;
    tracker_.OMSetRenderTargets(n, render_target_descriptors != nullptr ? handles.data() : nullptr,
                                rts_single_handle_to_descriptor_range,
                                depth_stencil_descriptor != nullptr ? &dsv : nullptr, device_);
}

void CommandListBridge::ClearRenderTargetView(d3d::CpuDescriptorHandle rtv,
                                              const float color_rgba[4],
                                              uint32_t num_rects,
                                              const pso::D3D12Rect* rects) {
    if (!writable()) {
        return;
    }
    pso::RenderTargetViewInfo info{};
    if (!device_.resolve_rtv(rtv.ptr, &info) || info.view_id == 0) {
        diagnostics_ |= BridgeDiag::UnknownTexture;
        return;
    }
    tracker_.ClearRenderTargetView(info, color_rgba, num_rects, rects);
    drain_events();
}

void CommandListBridge::ClearDepthStencilView(d3d::CpuDescriptorHandle dsv,
                                              uint32_t clear_flags,
                                              float depth,
                                              uint8_t stencil,
                                              uint32_t num_rects,
                                              const pso::D3D12Rect* rects) {
    if (!writable()) {
        return;
    }
    pso::DepthStencilViewInfo info{};
    if (!device_.resolve_dsv(dsv.ptr, &info) || info.view_id == 0) {
        diagnostics_ |= BridgeDiag::UnknownTexture;
        return;
    }
    tracker_.ClearDepthStencilView(info, clear_flags, depth, stencil, num_rects, rects);
    drain_events();
}

void CommandListBridge::OMSetBlendFactor(const float blend_factor[4]) {
    if (writable()) {
        tracker_.OMSetBlendFactor(blend_factor);
    }
}

void CommandListBridge::OMSetStencilRef(uint32_t stencil_ref) {
    if (writable()) {
        tracker_.OMSetStencilRef(stencil_ref);
    }
}

void CommandListBridge::RSSetViewports(uint32_t num_viewports, const pso::D3D12Viewport* viewports) {
    if (writable()) {
        tracker_.RSSetViewports(num_viewports, viewports);
    }
}

void CommandListBridge::RSSetScissorRects(uint32_t num_rects, const pso::D3D12Rect* rects) {
    if (writable()) {
        tracker_.RSSetScissorRects(num_rects, rects);
    }
}

void CommandListBridge::IASetPrimitiveTopology(pso::D3DPrimitiveTopology topology) {
    if (writable()) {
        tracker_.IASetPrimitiveTopology(topology);
    }
}

void CommandListBridge::IASetVertexBuffers(uint32_t start_slot,
                                           uint32_t num_views,
                                           const d3d12_xs::D3D12VertexBufferView* views) {
    if (!writable() || start_slot >= pso::kMaxVertexBuffers) {
        return;
    }
    const uint32_t count = std::min(num_views, pso::kMaxVertexBuffers - start_slot);
    std::array<uint32_t, pso::kMaxVertexBuffers> strides{};
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t slot = start_slot + i;
        vertex_buffers_[slot] = views != nullptr ? VertexBinding{views[i].buffer_id, views[i].offset} : VertexBinding{};
        strides[i] = views != nullptr ? views[i].stride_in_bytes : 0u;
        vertex_dirty_ |= 1u << slot;
    }
    tracker_.set_vertex_strides(start_slot, count, views != nullptr ? strides.data() : nullptr);
}

void CommandListBridge::IASetIndexBuffer(const d3d12_xs::D3D12IndexBufferView* view) {
    if (!writable()) {
        return;
    }
    index_buffer_ = view != nullptr ? *view : d3d12_xs::D3D12IndexBufferView{};
    index_dirty_ = true;
    if (view != nullptr) {
        tracker_.set_index_format(to_wgpu_index_format(view->format));
    }
}

d3d::Status CommandListBridge::SetDescriptorHeaps(uint32_t count, d3d::DescriptorHeap* const* heaps) {
    return writable() ? root_args_.set_descriptor_heaps(count, heaps) : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootSignature(const d3d::RootSignature* signature) {
    return writable() ? root_args_.set_root_signature(d3d::BindPoint::Graphics, signature) : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetComputeRootSignature(const d3d::RootSignature* signature) {
    return writable() ? root_args_.set_root_signature(d3d::BindPoint::Compute, signature) : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootDescriptorTable(uint32_t parameter_index, d3d::GpuDescriptorHandle base) {
    return writable() ? root_args_.set_descriptor_table(d3d::BindPoint::Graphics, parameter_index, base)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetComputeRootDescriptorTable(uint32_t parameter_index, d3d::GpuDescriptorHandle base) {
    return writable() ? root_args_.set_descriptor_table(d3d::BindPoint::Compute, parameter_index, base)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRoot32BitConstants(uint32_t parameter_index, uint32_t num_values,
                                                            const uint32_t* values, uint32_t dest_offset) {
    return writable() ? root_args_.set_root_constants(d3d::BindPoint::Graphics, parameter_index, num_values, values,
                                                      dest_offset)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetComputeRoot32BitConstants(uint32_t parameter_index, uint32_t num_values,
                                                           const uint32_t* values, uint32_t dest_offset) {
    return writable() ? root_args_.set_root_constants(d3d::BindPoint::Compute, parameter_index, num_values, values,
                                                      dest_offset)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootConstantBufferView(uint32_t parameter_index, uint64_t gpu_address) {
    return writable() ? root_args_.set_root_view(d3d::BindPoint::Graphics, parameter_index, d3d::RootParameterType::Cbv,
                                                 gpu_address)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetComputeRootConstantBufferView(uint32_t parameter_index, uint64_t gpu_address) {
    return writable() ? root_args_.set_root_view(d3d::BindPoint::Compute, parameter_index, d3d::RootParameterType::Cbv,
                                                 gpu_address)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootShaderResourceView(uint32_t parameter_index, uint64_t gpu_address) {
    return writable() ? root_args_.set_root_view(d3d::BindPoint::Graphics, parameter_index, d3d::RootParameterType::Srv,
                                                 gpu_address)
                      : d3d::Status::InvalidArgument;
}

d3d::Status CommandListBridge::SetGraphicsRootUnorderedAccessView(uint32_t parameter_index, uint64_t gpu_address) {
    return writable() ? root_args_.set_root_view(d3d::BindPoint::Graphics, parameter_index, d3d::RootParameterType::Uav,
                                                 gpu_address)
                      : d3d::Status::InvalidArgument;
}

// ---- Event translation ----

void CommandListBridge::drain_events() {
    const std::vector<pso::StateEvent>& events = tracker_.events();
    if (events.empty()) {
        return;
    }
    for (const pso::StateEvent& e : events) {
        switch (e.type) {
        case pso::StateEventType::BeginRenderPass:
            emit_begin_render_pass(tracker_.passes()[e.index]);
            break;
        case pso::StateEventType::EndRenderPass:
            encoder_.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
            end_usage_scope();
            render_pass_open_ = false;
            break;
        case pso::StateEventType::SetPipeline:
            if (e.pipeline_key != nullptr && graphics_pso_ != nullptr) {
                GpuCreateRenderPipeline2Payload payload{};
                payload.id = e.pipeline_id;
                payload.vs_module_id = graphics_pso_->vs_module_id_;
                payload.fs_module_id = graphics_pso_->fs_module_id_;
                if (graphics_pso_->layout_ != nullptr) {
                    device_.ensure_layout_emitted(graphics_pso_->layout_, encoder_);
                    payload.pipeline_layout_id = graphics_pso_->layout_->id;
                }
                std::memcpy(&payload.key, e.pipeline_key, sizeof(payload.key));
                encoder_.emit_packet(GpuOpcode::CreateRenderPipeline2, &payload, sizeof(payload));
                ++stats_.pipelines_created;
                diagnostics_ |= graphics_pso_->diagnostics_;
            }
            encoder_.set_pipeline(e.pipeline_id);
            break;
        case pso::StateEventType::SetViewport: {
            const GpuSetViewportPayload p{e.viewport.x, e.viewport.y, e.viewport.width,
                                          e.viewport.height, e.viewport.min_depth, e.viewport.max_depth};
            encoder_.emit_packet(GpuOpcode::SetViewport, &p, sizeof(p));
            break;
        }
        case pso::StateEventType::SetScissorRect: {
            const GpuSetScissorRectPayload p{e.scissor.x, e.scissor.y, e.scissor.width, e.scissor.height};
            encoder_.emit_packet(GpuOpcode::SetScissorRect, &p, sizeof(p));
            break;
        }
        case pso::StateEventType::SetBlendConstant: {
            const GpuSetBlendConstantPayload p{e.blend_constant[0], e.blend_constant[1], e.blend_constant[2],
                                               e.blend_constant[3]};
            encoder_.emit_packet(GpuOpcode::SetBlendConstant, &p, sizeof(p));
            break;
        }
        case pso::StateEventType::SetStencilReference: {
            const GpuSetStencilReferencePayload p{e.stencil_reference};
            encoder_.emit_packet(GpuOpcode::SetStencilReference, &p, sizeof(p));
            break;
        }
        case pso::StateEventType::ClearFallback:
            // Partial-rect clears need a worker-side clear draw that ABI v2 does not have yet.
            ++stats_.clear_fallbacks_dropped;
            diagnostics_ |= BridgeDiag::ClearFallbackDropped;
            break;
        }
    }
    tracker_.clear_events();
}

void CommandListBridge::emit_begin_render_pass(const pso::RenderPassDesc& pass) {
    GpuBeginRenderPassPayload p{};
    p.color_count = std::min(pass.color_count, kGpuMaxColorAttachments);
    usage_scope_.clear();
    scoped_bind_groups_.clear();
    for (uint32_t i = 0; i < p.color_count; ++i) {
        const pso::ColorAttachmentDesc& c = pass.color[i];
        const ViewRecord* v = c.view_id != 0 ? device_.view(c.view_id) : nullptr;
        if (v == nullptr) {
            continue;
        }
        GpuRenderPassColorAttachment& a = p.color[i];
        a.texture_id = v->texture_id;
        a.load_op = static_cast<uint8_t>(c.load_op);
        a.store_op = static_cast<uint8_t>(c.store_op);
        a.view = v->view;
        std::memcpy(a.clear_value, c.clear_value.data(), sizeof(a.clear_value));
        usage_scope_.add(v->resource_id, v->range, v->usage);
    }
    if (pass.has_depth_stencil) {
        const pso::DepthStencilAttachmentDesc& d = pass.depth_stencil;
        if (const ViewRecord* v = device_.view(d.view_id); v != nullptr) {
            GpuRenderPassDepthStencilAttachment& a = p.depth_stencil;
            a.texture_id = v->texture_id;
            a.depth_load_op = static_cast<uint8_t>(d.depth_load_op);
            a.depth_store_op = static_cast<uint8_t>(d.depth_store_op);
            a.stencil_load_op = static_cast<uint8_t>(d.stencil_load_op);
            a.stencil_store_op = static_cast<uint8_t>(d.stencil_store_op);
            a.view = v->view;
            a.depth_clear_value = d.depth_clear_value;
            a.stencil_clear_value = d.stencil_clear_value;
            a.depth_read_only = d.depth_read_only ? 1 : 0;
            a.stencil_read_only = d.stencil_read_only ? 1 : 0;
            usage_scope_.add(v->resource_id, v->range, v->usage);
        }
    }
    encoder_.emit_packet(GpuOpcode::BeginRenderPass, &p, sizeof(p));
    render_pass_open_ = true;
    ++stats_.render_passes;
    invalidate_pass_bindings();
    compute_pass_valid_ = false;
}

void CommandListBridge::invalidate_pass_bindings() {
    render_groups_.clear();
    vertex_dirty_ = 0;
    for (uint32_t slot = 0; slot < pso::kMaxVertexBuffers; ++slot) {
        if (vertex_buffers_[slot].buffer_id != 0) {
            vertex_dirty_ |= 1u << slot;
        }
    }
    index_dirty_ = index_buffer_.buffer_id != 0;
}

void CommandListBridge::end_usage_scope() {
    const size_t before = usage_conflicts_.size();
    usage_scope_.find_conflicts(&usage_conflicts_);
    if (usage_conflicts_.size() != before) {
        diagnostics_ |= BridgeDiag::UsageConflict;
        stats_.usage_conflicts += usage_conflicts_.size() - before;
    }
    usage_scope_.clear();
    scoped_bind_groups_.clear();
}

// ---- Bind groups ----

GpuTextureViewDesc CommandListBridge::texture_view_desc(const d3d::ViewDescriptor& v) {
    namespace V = d3d::ViewDimension;
    GpuTextureViewDesc d{};
    const uint32_t features = device_.options().features;
    pso::WgpuTextureAspect aspect = pso::WgpuTextureAspect::All;
    if (v.format != 0) {
        d.format = texture_format_code(to_dxgi(v.format), pso::FormatUsage::Sampled, features, &aspect);
    } else if (TextureInfo tex{}; device_.texture_info(v.resource_id, &tex)) {
        texture_format_code(tex.format, pso::FormatUsage::Sampled, features, &aspect);  // aspect of default views
    } else {
        diagnostics_ |= BridgeDiag::UnknownTexture;
    }
    d.aspect = static_cast<uint8_t>(aspect);
    bool supported = true;
    d.dimension = static_cast<uint8_t>(to_view_dimension(v.dimension, &supported));
    if (!supported) {
        diagnostics_ |= BridgeDiag::UnsupportedViewDimension;
    }
    const bool uav = v.kind == d3d::DescriptorKind::Uav;
    d.base_mip_level = v.first_mip;
    d.mip_level_count = uav ? 1 : (v.mip_levels == 0xffffu ? 0 : v.mip_levels);
    switch (v.dimension) {
    case V::Texture1D:
    case V::Texture2D:
    case V::Texture2DMs:
        d.array_layer_count = 1;
        break;
    case V::Texture1DArray:
    case V::Texture2DArray:
    case V::Texture2DMsArray:
        d.base_array_layer = v.first_array_slice;
        d.array_layer_count = v.array_size;
        break;
    case V::TextureCube:
        d.array_layer_count = 6;
        break;
    case V::TextureCubeArray:
        d.base_array_layer = v.first_array_slice;  // First2DArrayFace
        d.array_layer_count = v.array_size * 6u;   // NumCubes
        break;
    default:  // default view / 3D: whole texture
        break;
    }
    if (v.kind == d3d::DescriptorKind::Srv && v.component_mapping != d3d::kDefaultShader4ComponentMapping) {
        diagnostics_ |= BridgeDiag::ComponentMappingIgnored;
    }
    return d;
}

uint32_t CommandListBridge::upload_root_constants(d3d::BindPoint bind_point, uint32_t parameter_index) {
    if (parameter_index >= d3d::kMaxRootParameters) {
        return 0;
    }
    const std::span<const uint32_t> values = root_args_.root_constants(bind_point, parameter_index);
    RootConstantUpload& upload = root_uploads_[static_cast<uint32_t>(bind_point) & 1u][parameter_index];
    if (upload.valid && std::equal(values.begin(), values.end(), upload.values.begin(), upload.values.end())) {
        return upload.offset;
    }
    const uint32_t buffer_id = device_.ensure_root_constant_ring(encoder_);
    // The whole 256-byte ring slot is written (zero past the values): it is what the binding
    // covers (see the RootConstants case of to_bind_entry).
    const uint32_t bytes = align_up(std::max<uint32_t>(static_cast<uint32_t>(values.size()) * 4u, 1u),
                                    kRootConstantAlignment);
    bool wrapped = false;
    const uint32_t offset = device_.allocate_root_constants(bytes, &wrapped);
    if (wrapped) {
        diagnostics_ |= BridgeDiag::RootConstantRingWrapped;
    }
    std::vector<uint32_t> data(bytes / 4u, 0u);
    std::copy(values.begin(), values.end(), data.begin());
    encoder_.write_buffer(buffer_id, offset, data.data(), bytes);
    upload.values.assign(values.begin(), values.end());
    upload.offset = offset;
    upload.valid = true;
    ++stats_.root_constant_uploads;
    return offset;
}

bool CommandListBridge::emit_bind_group(d3d::BindPoint /*bind_point*/,
                                        const StateBridgeDevice::GroupLayoutRecord& group,
                                        d3d::CachedBindGroup* cached,
                                        const d3d::BindGroupPlan* plan) {
    if (cached->entries.size() != group.entries.size()) {
        diagnostics_ |= BridgeDiag::BindGroupBuildFailed;
        return false;
    }
    std::vector<GpuBindGroupEntry2> entries;
    gpu_entries(group, *cached, encoder_, &entries);
    GpuCreateBindGroup2Payload header{};
    header.id = device_.allocate_id();
    header.layout_id = group.id;
    header.entry_count = static_cast<uint32_t>(entries.size());
    const uint32_t entry_bytes = static_cast<uint32_t>(entries.size() * sizeof(GpuBindGroupEntry2));
    encoder_.emit_packet(GpuOpcode::CreateBindGroup2, &header, sizeof(header), entries.data(), entry_bytes);
    cached->gpu_id = header.id;
    ++stats_.bind_groups_created;
    if (cached->from_memory && plan != nullptr) {
        note_memory_group(group, *plan, *cached, encoder_.size() - entry_bytes);
    }
    return true;
}

void CommandListBridge::note_memory_group(const StateBridgeDevice::GroupLayoutRecord& group,
                                          const d3d::BindGroupPlan& plan,
                                          const d3d::CachedBindGroup& cached,
                                          size_t entries_offset) {
    for (MemoryBindGroup& m : memory_groups_) {
        if (m.built.gpu_id == cached.gpu_id) {
            if (entries_offset != kNoPacket) {
                m.entries_offset = entries_offset;
                m.built = cached;
            }
            return;
        }
    }
    memory_groups_.push_back(MemoryBindGroup{entries_offset, &group, plan, cached});
    ++stats_.memory_bind_groups;
}

uint32_t CommandListBridge::refresh_memory_bind_groups(CommandEncoder& prelude) {
    const d3d::DescriptorSlotSource* source = device_.options().slot_source;
    if (source == nullptr) {
        return 0;
    }
    uint32_t refreshed = 0;
    for (MemoryBindGroup& m : memory_groups_) {
        const bool changed = d3d::refresh_table_entries(device_.heaps(), *source, m.plan,
                                                        &StateBridgeDevice::gpu_va_trampoline, &device_, &m.built);
        if (changed) {
            std::vector<GpuBindGroupEntry2> entries;
            gpu_entries(*m.group, m.built, prelude, &entries);
            const size_t bytes = entries.size() * sizeof(GpuBindGroupEntry2);
            if (m.entries_offset != kNoPacket) {
                encoder_.patch(m.entries_offset, entries.data(), bytes);
            } else {
                // Created by an earlier recording of this list: create it again under the same id.
                GpuCreateBindGroup2Payload header{};
                header.id = m.built.gpu_id;
                header.layout_id = m.group->id;
                header.entry_count = static_cast<uint32_t>(entries.size());
                prelude.emit_packet(GpuOpcode::CreateBindGroup2, &header, sizeof(header), entries.data(),
                                    static_cast<uint32_t>(bytes));
            }
            ++refreshed;
            ++stats_.memory_bind_groups_refreshed;
        }
        sync_bound_textures(*m.group, m.built, prelude);
    }
    return refreshed;
}

void CommandListBridge::sync_bound_textures(const StateBridgeDevice::GroupLayoutRecord& group,
                                            const d3d::CachedBindGroup& cached,
                                            CommandEncoder& target) {
    if (!device_.options().sync_texture_memory) {
        return;
    }
    for (size_t i = 0; i < cached.entries.size() && i < group.joins.size(); ++i) {
        const d3d::BindGroupEntry& e = cached.entries[i];
        if (e.kind == d3d::BindResourceKind::TextureView && e.resource_id != 0 &&
            (e.view.kind == d3d::DescriptorKind::Srv ||
             group.joins[i].register_class == d3d::DescriptorRangeType::Srv)) {
            if (device_.options().sync_texture_memory(e.resource_id, target)) {
                ++stats_.texture_memory_syncs;
            }
        }
    }
}

void CommandListBridge::gpu_entries(const StateBridgeDevice::GroupLayoutRecord& group,
                                    const d3d::CachedBindGroup& cached,
                                    CommandEncoder& side,
                                    std::vector<GpuBindGroupEntry2>* result) {
    std::vector<GpuBindGroupEntry2>& entries = *result;
    entries.assign(cached.entries.size(), GpuBindGroupEntry2{});
    for (size_t i = 0; i < entries.size() && i < group.entries.size(); ++i) {
        const d3d::BindGroupEntry& e = cached.entries[i];
        const GpuBindGroupLayoutEntry& layout = group.entries[i];
        GpuBindGroupEntry2& out = entries[i];
        out.binding = layout.binding;
        switch (e.kind) {
        case d3d::BindResourceKind::Buffer:
            out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
            out.resource_id = e.resource_id;
            out.offset = e.offset;
            out.size = e.size;
            // WebGPU binds uniform buffers only at 256-byte aligned offsets; the title's constant
            // buffers are 16-byte aligned (Xbox). Relocate to a copy at an aligned offset.
            if (layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) &&
                layout.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Uniform) &&
                e.offset % kUniformOffsetAlignment != 0) {
                const uint64_t base = device_.buffer_gpu_va(e.resource_id);
                uint32_t id = 0;
                uint64_t offset = 0;
                // A reflected stage (G62) gives the binding's minimum size: the home covers it.
                const uint64_t home = std::max<uint64_t>(e.size, layout.min_binding_size);
                if (base != 0 && e.size != 0 && device_.options().relocate_uniform &&
                    device_.options().relocate_uniform(base + e.offset, home, &id, &offset)) {
                    out.resource_id = id;
                    out.offset = offset;
                    // The relocated home is a whole multiple of 256 bytes (zero past the view).
                    out.size = align_up(home, kUniformOffsetAlignment);
                } else {
                    diagnostics_ |= BridgeDiag::UnalignedUniform;
                }
            } else if (layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Buffer) && e.size != 0 &&
                       (layout.min_binding_size != 0
                            ? layout.min_binding_size > e.size
                            : layout.binding_type == static_cast<uint8_t>(GpuBufferBindingType::Uniform) &&
                                  e.size % kUniformOffsetAlignment != 0)) {
                // Xbox constant-buffer views are 16-byte multiples. Bind up to what the stage reads:
                // its reflected minimum (G62), else the 256-byte minimum of PC D3D12, when the
                // buffer has the room.
                const uint64_t target = layout.min_binding_size != 0 ? layout.min_binding_size
                                                                     : align_up(e.size, kUniformOffsetAlignment);
                const uint64_t base = device_.buffer_gpu_va(e.resource_id);
                d3d::GpuVaResolution res;
                if (base != 0 && device_.resolve_gpu_va(base + e.offset, &res)) {
                    out.size = std::min<uint64_t>(target, res.resource_size - res.offset);
                }
            }
            break;
        case d3d::BindResourceKind::TextureView:
            out.resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
            out.resource_id = e.resource_id;
            out.view = texture_view_desc(e.view);
            break;
        case d3d::BindResourceKind::Sampler: {
            out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Sampler);
            bool dropped = false;
            const d3d::SamplerDesc s =
                layout.type == static_cast<uint8_t>(GpuBindingLayoutType::Sampler)
                    ? sampler_for_binding(e.sampler, layout.binding_type, &dropped)
                    : e.sampler;
            if (dropped) {
                diagnostics_ |= BridgeDiag::SamplerApproximated;
            }
            out.resource_id = device_.intern_sampler(s, side, &diagnostics_);
            break;
        }
        case d3d::BindResourceKind::RootConstants:
            out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
            out.resource_id = device_.ensure_root_constant_ring(side);
            // Bind the whole ring slot (256 bytes, zero past the values), like a constant-buffer
            // view padded to the PC D3D12 minimum: stages may declare the constants as a larger
            // uniform block, and WebGPU rejects a binding smaller than the shader's block.
            out.size = align_up(std::max<uint32_t>(static_cast<uint32_t>(e.size), 1u), kRootConstantAlignment);
            break;
        case d3d::BindResourceKind::Null:
            switch (static_cast<GpuBindingLayoutType>(layout.type)) {
            case GpuBindingLayoutType::Buffer:
                out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
                break;
            case GpuBindingLayoutType::Sampler:
                out.resource_type = static_cast<uint32_t>(GpuBindResourceType::Sampler);
                break;
            default:
                out.resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
                break;
            }
            diagnostics_ |= BridgeDiag::NullBinding;
            break;
        }
    }
}

void CommandListBridge::add_usage(const d3d::BindGroupEntry& entry, const d3d::ShaderBinding& join) {
    if (entry.resource_id == 0) {
        return;
    }
    if (entry.kind == d3d::BindResourceKind::TextureView) {
        usage_scope_.add(entry.resource_id, d3d::view_subresource_range(entry.view), d3d::view_usage(entry.view));
    } else if (entry.kind == d3d::BindResourceKind::Buffer) {
        d3d::ResourceUsage usage = d3d::ResourceUsage::Sampled;
        if (join.register_class == d3d::DescriptorRangeType::Uav) {
            usage = d3d::ResourceUsage::StorageWrite;
        } else if (join.register_class == d3d::DescriptorRangeType::Srv) {
            usage = d3d::ResourceUsage::StorageRead;
        }
        usage_scope_.add(entry.resource_id, d3d::SubresourceRange{}, usage);
    }
}

bool CommandListBridge::apply_bind_groups(d3d::BindPoint bind_point,
                                          StateBridgeDevice::PipelineLayoutRecord* layout,
                                          const std::vector<d3d::BindGroupPlan>& plans,
                                          std::vector<BoundGroup>* bound,
                                          bool track_usage) {
    bound->resize(layout->groups.size());
    for (size_t g = 0; g < layout->groups.size(); ++g) {
        const StateBridgeDevice::GroupLayoutRecord& group = *layout->groups[g];
        uint32_t bind_group_id = 0;
        std::vector<uint32_t> offsets;
        d3d::CachedBindGroup* cached = nullptr;
        if (group.entries.empty()) {
            auto it = empty_bind_groups_.find(group.id);
            if (it == empty_bind_groups_.end()) {
                device_.ensure_layout_emitted(layout, encoder_);
                GpuCreateBindGroup2Payload header{};
                header.id = device_.allocate_id();
                header.layout_id = group.id;
                encoder_.emit_packet(GpuOpcode::CreateBindGroup2, &header, sizeof(header));
                ++stats_.bind_groups_created;
                it = empty_bind_groups_.emplace(group.id, header.id).first;
            }
            bind_group_id = it->second;
        } else {
            const d3d::BindGroupLookup lookup = bind_groups_.get_or_build(
                root_args_, bind_point, plans[g], group.id, &StateBridgeDevice::gpu_va_trampoline, &device_);
            if (lookup.status != d3d::Status::Ok || lookup.bind_group == nullptr) {
                diagnostics_ |= lookup.status == d3d::Status::RootArgumentNotSet ? BridgeDiag::RootArgumentNotSet
                                                                                  : BridgeDiag::BindGroupBuildFailed;
                return false;
            }
            cached = lookup.bind_group;
            if (lookup.rebuilt || cached->gpu_id == 0) {
                device_.ensure_layout_emitted(layout, encoder_);
                if (!emit_bind_group(bind_point, group, cached, &plans[g])) {
                    return false;
                }
                if (lookup.retired_gpu_id != 0) {
                    encoder_.destroy_resource(lookup.retired_gpu_id);
                    ++stats_.bind_groups_retired;
                }
            } else if (cached->from_memory) {
                note_memory_group(group, plans[g], *cached, kNoPacket);  // created by an earlier recording
            }
            sync_bound_textures(group, *cached, encoder_);
            bind_group_id = cached->gpu_id;
            for (size_t i = 0; i < group.entries.size(); ++i) {
                if ((group.entries[i].flags & GpuBindGroupLayoutEntryFlags::HasDynamicOffset) == 0) {
                    continue;
                }
                const d3d::BindGroupEntry& e = cached->entries[i];
                offsets.push_back(e.kind == d3d::BindResourceKind::RootConstants
                                      ? upload_root_constants(bind_point, e.root_parameter_index)
                                      : 0u);
            }
        }
        BoundGroup& b = (*bound)[g];
        if (b.bind_group_id != bind_group_id || b.dynamic_offsets != offsets) {
            const GpuSetBindGroup2Payload header{static_cast<uint32_t>(g), bind_group_id,
                                                 static_cast<uint32_t>(offsets.size()), 0u};
            encoder_.emit_packet(GpuOpcode::SetBindGroup2, &header, sizeof(header), offsets.data(),
                                 static_cast<uint32_t>(offsets.size() * sizeof(uint32_t)));
            b.bind_group_id = bind_group_id;
            b.dynamic_offsets = std::move(offsets);
        }
        if (track_usage && cached != nullptr &&
            std::find(scoped_bind_groups_.begin(), scoped_bind_groups_.end(), bind_group_id) ==
                scoped_bind_groups_.end()) {
            scoped_bind_groups_.push_back(bind_group_id);
            for (size_t i = 0; i < cached->entries.size(); ++i) {
                add_usage(cached->entries[i], group.joins[i]);
            }
        }
    }
    return true;
}

// ---- Work ----

bool CommandListBridge::prepare_render_draw() {
    const bool ok = tracker_.prepare_draw(&device_.pipeline_cache());
    drain_events();
    if (ok && graphics_pso_ != nullptr && graphics_pso_->layout_ != nullptr &&
        graphics_pso_->plans_.size() == graphics_pso_->layout_->groups.size() &&
        !apply_bind_groups(d3d::BindPoint::Graphics, graphics_pso_->layout_, graphics_pso_->plans_, &render_groups_,
                           true)) {
        ++stats_.draws_skipped;
        diagnostics_ |= BridgeDiag::DrawSkipped;
        return false;
    }
    if (!ok) {
        ++stats_.draws_skipped;
        diagnostics_ |= BridgeDiag::DrawSkipped;
    }
    return ok;
}

void CommandListBridge::apply_input_assembler(bool indexed) {
    for (uint32_t slot = 0; slot < pso::kMaxVertexBuffers; ++slot) {
        if ((vertex_dirty_ & (1u << slot)) != 0 && vertex_buffers_[slot].buffer_id != 0) {
            encoder_.set_vertex_buffer(slot, vertex_buffers_[slot].buffer_id, vertex_buffers_[slot].offset);
        }
    }
    vertex_dirty_ = 0;
    if (indexed && index_dirty_ && index_buffer_.buffer_id != 0) {
        encoder_.set_index_buffer(index_buffer_.buffer_id, index_buffer_.format, index_buffer_.offset);
        index_dirty_ = false;
    }
}

void CommandListBridge::DrawInstanced(uint32_t vertex_count_per_instance,
                                      uint32_t instance_count,
                                      uint32_t start_vertex_location,
                                      uint32_t start_instance_location) {
    if (!writable() || !prepare_render_draw()) {
        return;
    }
    apply_input_assembler(false);
    encoder_.draw(vertex_count_per_instance, instance_count, start_vertex_location, start_instance_location);
    ++stats_.draws;
}

void CommandListBridge::DrawIndexedInstanced(uint32_t index_count_per_instance,
                                             uint32_t instance_count,
                                             uint32_t start_index_location,
                                             int32_t base_vertex_location,
                                             uint32_t start_instance_location) {
    if (!writable()) {
        return;
    }
    if (index_buffer_.buffer_id == 0) {
        ++stats_.draws_skipped;
        diagnostics_ |= BridgeDiag::DrawSkipped;
        return;
    }
    if (!prepare_render_draw()) {
        return;
    }
    apply_input_assembler(true);
    encoder_.draw_indexed(index_count_per_instance, instance_count, start_index_location, base_vertex_location,
                          start_instance_location);
    ++stats_.draws;
}

void CommandListBridge::Dispatch(uint32_t thread_group_count_x,
                                 uint32_t thread_group_count_y,
                                 uint32_t thread_group_count_z) {
    if (!writable()) {
        return;
    }
    // G7: pending clears must land and the render pass must end before compute work.
    tracker_.flush_pending_clears();
    tracker_.end_render_pass();
    drain_events();
    ComputePipelineState* state = compute_pso_;
    if (state == nullptr || state->pipeline_id_ == 0) {
        diagnostics_ |= BridgeDiag::NoComputePipeline;
        return;
    }
    if (device_.claim_compute_creation(state->pipeline_id_)) {
        GpuCreateComputePipeline2Payload payload{};
        payload.id = state->pipeline_id_;
        payload.cs_module_id = state->cs_module_id_;
        if (state->layout_ != nullptr) {
            device_.ensure_layout_emitted(state->layout_, encoder_);
            payload.pipeline_layout_id = state->layout_->id;
        }
        std::memcpy(&payload.key, &state->key_, sizeof(payload.key));
        encoder_.emit_packet(GpuOpcode::CreateComputePipeline2, &payload, sizeof(payload));
        ++stats_.pipelines_created;
        diagnostics_ |= state->diagnostics_;
    }
    if (!compute_pass_valid_ || compute_pipeline_bound_ != state->pipeline_id_) {
        encoder_.set_compute_pipeline(state->pipeline_id_);
        compute_groups_.clear();
        compute_pass_valid_ = true;
        compute_pipeline_bound_ = state->pipeline_id_;
    }
    if (state->layout_ != nullptr && state->plans_.size() == state->layout_->groups.size() &&
        !apply_bind_groups(d3d::BindPoint::Compute, state->layout_, state->plans_, &compute_groups_, false)) {
        return;
    }
    encoder_.dispatch_compute(thread_group_count_x, thread_group_count_y, thread_group_count_z);
    ++stats_.dispatches;
}

void CommandListBridge::ResourceBarrier(uint32_t num_barriers, const d3d::ResourceBarrier* barriers) {
    if (!writable()) {
        return;
    }
    // G7: anything that reads or copies a render target must see pending clears, and WebGPU usage
    // scopes end with the pass.
    tracker_.flush_pending_clears();
    tracker_.end_render_pass();
    drain_events();
    if (num_barriers != 0 && barriers != nullptr) {
        resource_states_.resource_barrier(num_barriers, barriers);
    }
}

void CommandListBridge::WriteTexture(uint32_t texture_id, uint32_t mip_level, uint32_t origin_x, uint32_t origin_y,
                                     uint32_t origin_z, uint32_t width, uint32_t height, uint32_t depth_or_layers,
                                     uint32_t bytes_per_row, uint32_t rows_per_image, const void* data,
                                     uint32_t data_len) {
    if (!writable()) {
        return;
    }
    encoder_.write_texture(texture_id, mip_level, origin_x, origin_y, origin_z, width, height, depth_or_layers,
                           bytes_per_row, rows_per_image, data, data_len);
}

}  // namespace playgta6::gpu::bridge
