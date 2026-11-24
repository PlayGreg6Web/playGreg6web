#include "gpu/d3d12_guest_resources.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#include "os/os_shim.h"
#include "os/surface_blocks.h"
#include "os/xbox_scarlett.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

constexpr uint64_t kShadowPage = 4096;
constexpr uint64_t kMaxWriteChunk = 1u << 20;
constexpr uint64_t kMaxShaderBytes = 64u << 20;
constexpr uint32_t kMaxInputElements = 32;

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%" PRIx64, v);
    return b;
}

constexpr uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

bool heap_cpu_visible(const GuestHeapProperties& p) {
    if (p.Type == HeapType::Upload || p.Type == HeapType::Readback) {
        return true;
    }
    // CUSTOM: CPUPageProperty 2 WRITE_COMBINE / 3 WRITE_BACK are CPU accessible.
    return p.Type == HeapType::Custom && (p.CPUPageProperty == 2 || p.CPUPageProperty == 3);
}

uint32_t full_mip_count(const GuestResourceDesc& d) {
    uint64_t m = std::max<uint64_t>({d.Width, d.Height,
                                     d.Dimension == ResourceDimension::Texture3D ? d.DepthOrArraySize : 1u});
    uint32_t n = 1;
    while (m > 1) {
        m >>= 1;
        ++n;
    }
    return n;
}

float f32(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint64_t u64_at(const uint32_t* u, size_t i) {
    return uint64_t{u[i]} | (uint64_t{u[i + 1]} << 32);
}

uint32_t mips_or_all(uint32_t v) {
    return v == 0xFFFFFFFFu ? 0xffffu : v;
}

}  // namespace

uint64_t shader_identity(const void* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ull;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x00000100000001b3ull;
    }
    return h;
}

// ---------------------------------------------------------------------------------------------
// Device: shaders, VA ranges, tracked buffers, diagnostics
// ---------------------------------------------------------------------------------------------

bool Device::shader_known(uint64_t identity) const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return known_shaders_.count(identity) != 0;
}

void Device::register_shader_module(uint64_t identity, uint32_t module_id) {
    bridge_->register_shader_module(identity, module_id);
    std::lock_guard<std::mutex> lock(shader_mutex_);
    known_shaders_.insert(identity);
    unresolved_shaders_.erase(std::remove(unresolved_shaders_.begin(), unresolved_shaders_.end(), identity),
                              unresolved_shaders_.end());
}

void Device::note_unresolved_shader(uint64_t identity) {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    if (std::find(unresolved_shaders_.begin(), unresolved_shaders_.end(), identity) == unresolved_shaders_.end()) {
        unresolved_shaders_.push_back(identity);
        char name[24];
        std::snprintf(name, sizeof(name), "0x%016" PRIx64, identity);
        os::log(std::string("d3d12_xs: no shader module for DXIL identity ") + name + " (draws with it are skipped)");
    }
}

std::vector<uint64_t> Device::unresolved_shaders() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return unresolved_shaders_;
}

uint64_t Device::allocate_gpu_va(uint64_t size) {
    std::lock_guard<std::mutex> lock(va_mutex_);
    const uint64_t va = next_fake_va_;
    next_fake_va_ += align_up(std::max<uint64_t>(size, 1), 64u << 10);
    return va;
}

void Device::track_cpu_buffer(Resource* resource) {
    std::lock_guard<std::mutex> lock(tracked_mutex_);
    tracked_.push_back(resource);
}

void Device::untrack_cpu_buffer(Resource* resource) {
    std::lock_guard<std::mutex> lock(tracked_mutex_);
    tracked_.erase(std::remove(tracked_.begin(), tracked_.end(), resource), tracked_.end());
}

void Device::flush_cpu_buffers(CommandEncoder& encoder) {
    std::lock_guard<std::mutex> lock(tracked_mutex_);
    for (Resource* r : tracked_) {
        r->flush_shadow(encoder);
    }
}

bool Device::set_back_buffer(uint64_t resource_interface_pointer) {
    com::detail::Resolved r;
    if (!com::detail::lookup_interface(resource_interface_pointer, &r) || r.iface != &kID3D12Resource.info) {
        return false;
    }
    auto* resource = static_cast<Resource*>(static_cast<IResourceHost*>(r.host));
    if (resource->desc().Dimension != ResourceDimension::Texture2D || !resource->has_gpu_texture()) {
        return false;
    }
    resource->mark_back_buffer();
    return true;
}

void Device::note_unsupported_copy(const std::string& message) {
    unsupported_copies_.fetch_add(1, std::memory_order_relaxed);
    note_bad_argument(message);
}

// ---------------------------------------------------------------------------------------------
// Heap
// ---------------------------------------------------------------------------------------------

Heap::Heap(com::Ref<Device> device, const GuestHeapDesc& desc) : ChildBase(std::move(device)), desc_(desc) {
    cpu_visible_ = heap_cpu_visible(desc.Properties);
    const uint64_t size = align_up(std::max<uint64_t>(desc.SizeInBytes, 1), 64u << 10);
    if (cpu_visible_) {
        base_ = com::guest_memory().virtual_alloc(0, size, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        owns_guest_memory_ = base_ != 0;
    } else {
        base_ = device_->allocate_gpu_va(size);
    }
    device_->memory_heaps.fetch_add(1);
}

Heap::~Heap() {
    if (owns_guest_memory_) {
        com::guest_memory().release(base_);
    }
    device_->memory_heaps.fetch_sub(1);
}

// ---------------------------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------------------------

Resource::Resource(com::Ref<Device> device, const GuestResourceDesc& desc, Placement placement, uint32_t initial_state)
    : ChildBase(std::move(device)), desc_(desc), placement_(std::move(placement)), initial_state_(initial_state) {
    device_->resources.fetch_add(1);
}

Resource::~Resource() {
    if (tracked_) {
        device_->untrack_cpu_buffer(this);
    }
    if (id_ != 0) {
        device_->state_bridge().unregister_resource(id_);
        device_->resource_states().unregister_resource(id_);
        CommandEncoder enc;
        enc.destroy_resource(id_);
        if (is_buffer() || gpu_texture_) {
            device_->submit(enc);
        }
    }
    if (owned_guest_memory_ != 0) {
        com::guest_memory().release(owned_guest_memory_);
    }
    if (xmem_memory_ != nullptr) {
        pg6_xbox_XMemFree(xmem_memory_, kD3D12ResourceXMemAttributes);
    }
    device_->resources.fetch_sub(1);
}

uint64_t committed_memory_size(const GuestResourceDesc& d) {
    // The 64 KiB-block surface size where os/surface_blocks.h covers the resource, else the
    // linear estimate.
    const DxgiFormatLayout l = dxgi_format_layout(d.Format);
    os::surface::Block64kLayout block{};
    if (d.Dimension == ResourceDimension::Texture2D && d.MipLevels <= 1 && d.DepthOrArraySize <= 1 &&
        d.SampleCount <= 1 && l.known && l.block_width == 1 && l.block_height == 1 && l.planes == 1 &&
        d.Width <= 0xFFFFFFFFull &&
        os::surface::block64k_layout_2d(static_cast<uint32_t>(d.Width), d.Height, l.block_bytes, &block)) {
        return block.size_bytes;
    }
    return resource_size_estimate(d);
}

uint32_t Resource::array_layers() const {
    return desc_.Dimension == ResourceDimension::Texture3D || is_buffer() ? 1u
                                                                          : std::max<uint32_t>(1, desc_.DepthOrArraySize);
}

uint32_t Resource::subresource_count() const {
    if (is_buffer()) {
        return 1;
    }
    const DxgiFormatLayout l = dxgi_format_layout(desc_.Format);
    return mips_ * array_layers() * (l.known ? l.planes : 1u);
}

bool Resource::initialize() {
    Device& dev = *device_;
    id_ = dev.allocate_id();
    if (is_buffer()) {
        const uint64_t size = desc_.Width;
        if (placement_.address != 0) {
            gpu_va_ = placement_.address;
            const uint64_t cpu = dev.cpu_address_for_gpu(placement_.address);
            // Inside an alias window the CPU pages may be mapped later (streaming): keep the CPU
            // address; flush_shadow reads whatever is mapped at each submission.
            if (cpu != placement_.address || com::guest_to_host(cpu, static_cast<size_t>(size)) != nullptr) {
                cpu_va_ = cpu;
            } else {
                dev.note_bad_argument("resource at " + hex(placement_.address) +
                                      " is not guest memory: contents cannot be uploaded");
            }
        } else if (placement_.placed_heap) {
            gpu_va_ = placement_.placed_heap->base() + placement_.heap_offset;
            cpu_va_ = placement_.placed_heap->cpu_visible() ? gpu_va_ : 0;
        } else if (placement_.reserved || !heap_cpu_visible(placement_.heap)) {
            gpu_va_ = dev.allocate_gpu_va(size);
        } else {
            const uint64_t va = com::guest_memory().virtual_alloc(0, align_up(size, 64u << 10),
                                                                  os::kMemReserve | os::kMemCommit,
                                                                  os::PageProtect::ReadWrite);
            if (va == 0) {
                dev.note_bad_argument("out of guest memory for a " + std::to_string(size) + "-byte buffer");
                id_ = 0;
                return false;
            }
            owned_guest_memory_ = va;
            gpu_va_ = cpu_va_ = va;
        }
        CommandEncoder enc;
        enc.create_buffer(id_,
                          GpuBufferUsage::Vertex | GpuBufferUsage::Index | GpuBufferUsage::Uniform |
                              GpuBufferUsage::Storage | GpuBufferUsage::Indirect | GpuBufferUsage::CopyDst |
                              GpuBufferUsage::CopySrc,
                          align_up(size, 4));
        dev.submit(enc);
        dev.state_bridge().register_buffer(id_, gpu_va_, size);
        if (cpu_va_ != 0) {
            shadow_.assign(static_cast<size_t>(size), 0);
            dev.track_cpu_buffer(this);
            tracked_ = true;
        }
    } else {
        mips_ = desc_.MipLevels == 0 ? full_mip_count(desc_) : desc_.MipLevels;
        const bool depth = (desc_.Flags & ResourceFlags::AllowDepthStencil) != 0;
        const bool rt = (desc_.Flags & ResourceFlags::AllowRenderTarget) != 0;
        if (rt && !depth && !placement_.placed_heap && !placement_.reserved && placement_.address == 0) {
            // The console's D3D12 allocates committed-resource memory with XMemAlloc, so a
            // title's XMem allocation hooks see it (and may place it in arenas of their own).
            // Only committed render targets for now: they are what the boot shows (other
            // resource kinds are left as they were until a title shows their accounting).
            const uint64_t size = committed_memory_size(desc_);
            xmem_memory_ = pg6_xbox_XMemAlloc(static_cast<size_t>(size), kD3D12ResourceXMemAttributes);
            if (xmem_memory_ == nullptr) {
                dev.note_bad_argument("XMemAlloc of " + std::to_string(size) + " bytes for a render target failed");
                id_ = 0;
                return false;
            }
        }
        const pso::FormatUsage usage = depth ? pso::FormatUsage::DepthStencil
                                             : (rt ? pso::FormatUsage::RenderTarget : pso::FormatUsage::Sampled);
        const pso::TextureFormatMapping m = pso::map_texture_format(static_cast<pso::DxgiFormat>(desc_.Format), usage,
                                                                    dev.state_bridge().options().features);
        if (!m.supported()) {
            dev.note_bad_argument("texture format " + std::to_string(desc_.Format) +
                                  " has no WebGPU equivalent: resource created without a GPU texture");
        } else {
            GpuCreateTexture2Payload p{};
            p.id = id_;
            p.width = static_cast<uint32_t>(desc_.Width);
            p.height = std::max<uint32_t>(1, desc_.Height);
            p.depth_or_layers = std::max<uint32_t>(1, desc_.DepthOrArraySize);
            p.mip_levels = mips_;
            p.sample_count = pso::map_sample_count(desc_.SampleCount);
            p.usage_flags = GpuTextureUsage::CopySrc | GpuTextureUsage::CopyDst;
            if ((desc_.Flags & ResourceFlags::DenyShaderResource) == 0) {
                p.usage_flags |= GpuTextureUsage::TextureBinding;
            }
            if (rt || depth) {
                p.usage_flags |= GpuTextureUsage::RenderAttachment;
            }
            if ((desc_.Flags & ResourceFlags::AllowUnorderedAccess) != 0) {
                p.usage_flags |= GpuTextureUsage::StorageBinding;
            }
            p.dimension = static_cast<uint8_t>(desc_.Dimension == ResourceDimension::Texture1D
                                                   ? GpuTextureDimension::Dim1D
                                                   : (desc_.Dimension == ResourceDimension::Texture3D
                                                          ? GpuTextureDimension::Dim3D
                                                          : GpuTextureDimension::Dim2D));
            p.format = static_cast<uint8_t>(m.format);
            CommandEncoder enc;
            enc.emit_packet(GpuOpcode::CreateTexture2, &p, sizeof(p));
            dev.submit(enc);
            gpu_texture_ = true;
            bridge::TextureInfo info;
            info.format = static_cast<pso::DxgiFormat>(desc_.Format);
            info.width = p.width;
            info.height = p.height;
            info.depth_or_array_size = p.depth_or_layers;
            info.mip_levels = mips_;
            info.sample_count = p.sample_count;
            info.dimension = static_cast<GpuTextureDimension>(p.dimension);
            dev.state_bridge().register_texture(id_, info);
        }
    }
    d3d::ResourceStateDesc sd;
    sd.subresource_count = subresource_count();
    sd.initial_state = initial_state_;
    sd.implicit_promotion = is_buffer() || (desc_.Flags & ResourceFlags::AllowSimultaneousAccess) != 0;
    dev.resource_states().register_resource(id_, sd);
    return true;
}

void Resource::mark_back_buffer() {
    bridge::TextureInfo info;
    info.format = static_cast<pso::DxgiFormat>(desc_.Format);
    info.width = static_cast<uint32_t>(desc_.Width);
    info.height = std::max<uint32_t>(1, desc_.Height);
    info.depth_or_array_size = 1;
    info.mip_levels = mips_;
    info.sample_count = 1;
    info.dimension = GpuTextureDimension::Dim2D;
    info.swapchain = true;
    device_->state_bridge().register_texture(id_, info);
    back_buffer_.store(true);
}

void Resource::flush_shadow(CommandEncoder& encoder) {
    if (cpu_va_ == 0 || shadow_.empty()) {
        return;
    }
    const uint64_t size = shadow_.size();
    const auto* live = static_cast<const uint8_t*>(com::guest_to_host(cpu_va_, static_cast<size_t>(size)));
    std::vector<uint8_t> copy;
    if (live == nullptr) {
        // Not mapped in one piece (e.g. an alias-window buffer whose pages are mapped one by one):
        // unreadable pages keep their shadow contents, i.e. count as unchanged.
        copy.assign(shadow_.begin(), shadow_.end());
        if (!com::read_guest(cpu_va_, copy.data(), copy.size())) {
            for (uint64_t page = 0; page < size; page += kShadowPage) {
                const uint64_t n = std::min(kShadowPage, size - page);
                com::read_guest(cpu_va_ + page, copy.data() + page, static_cast<size_t>(n));
            }
        }
        live = copy.data();
    }
    uint64_t run_start = 0;
    bool in_run = false;
    uint64_t uploaded = 0;
    auto emit = [&](uint64_t begin, uint64_t end) {
        for (uint64_t off = begin; off < end; off += kMaxWriteChunk) {
            const uint64_t n = std::min(kMaxWriteChunk, end - off);
            encoder.write_buffer(id_, static_cast<uint32_t>(off), live + off, static_cast<uint32_t>(n));
            std::memcpy(shadow_.data() + off, live + off, static_cast<size_t>(n));
            uploaded += n;
        }
    };
    for (uint64_t page = 0; page < size; page += kShadowPage) {
        const uint64_t n = std::min(kShadowPage, size - page);
        const bool changed = std::memcmp(shadow_.data() + page, live + page, static_cast<size_t>(n)) != 0;
        if (changed && !in_run) {
            run_start = page;
            in_run = true;
        } else if (!changed && in_run) {
            emit(run_start, page);
            in_run = false;
        }
    }
    if (in_run) {
        emit(run_start, size);
    }
    if (uploaded != 0) {
        device_->note_upload(uploaded);
    }
}

HResult Resource::Map(uint32_t subresource, uint64_t, OutPtr data) {
    if (!is_buffer() || cpu_va_ == 0 || subresource != 0) {
        device_->note_bad_argument("ID3D12Resource::Map on a resource without CPU access (id " + std::to_string(id_) +
                                   ")");
        if (data) {
            data.write(0);
        }
        return kEInvalidArg;
    }
    if (data) {
        data.write(cpu_va_);
    }
    return kSOk;
}

void Resource::Unmap(uint32_t, uint64_t) {
    // Nothing to do: CPU-visible buffers are compared with their shadow at ExecuteCommandLists.
}

HResult Resource::WriteToSubresource(com::CallContext& ctx) {
    // (DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch)
    const uint32_t sub = static_cast<uint32_t>(ctx.int_arg(1));
    const uint64_t box_va = ctx.int_arg(2);
    const uint64_t src = ctx.int_arg(3);
    const uint32_t row_pitch = static_cast<uint32_t>(ctx.int_arg(4));
    const uint32_t depth_pitch = static_cast<uint32_t>(ctx.int_arg(5));
    if (is_buffer() || !gpu_texture_ || sub >= subresource_count() || src == 0) {
        device_->note_bad_argument("WriteToSubresource: unsupported target or arguments");
        return kEInvalidArg;
    }
    const uint32_t mip = sub % mips_;
    const uint32_t layer = (sub / mips_) % array_layers();
    const uint32_t w = std::max<uint32_t>(1, static_cast<uint32_t>(desc_.Width >> mip));
    const uint32_t h = std::max<uint32_t>(1, desc_.Height >> mip);
    const uint32_t d = desc_.Dimension == ResourceDimension::Texture3D ? std::max<uint32_t>(1, desc_.DepthOrArraySize >> mip) : 1u;
    GuestBox box{0, 0, 0, w, h, d};
    if (box_va != 0 && !com::read_guest(box_va, box)) {
        device_->note_bad_argument("WriteToSubresource: unreadable box");
        return kEInvalidArg;
    }
    const DxgiFormatLayout l = dxgi_format_layout(desc_.Format);
    const uint32_t bw = std::max<uint32_t>(1, box.right - box.left);
    const uint32_t bh = std::max<uint32_t>(1, box.bottom - box.top);
    const uint32_t bd = std::max<uint32_t>(1, box.back - box.front);
    const uint32_t rows = (bh + l.block_height - 1) / l.block_height;
    const uint64_t len = uint64_t{depth_pitch} * (bd - 1) + uint64_t{row_pitch} * rows;
    std::vector<uint8_t> bytes(static_cast<size_t>(len));
    if (!com::read_guest(src, bytes.data(), bytes.size())) {
        device_->note_bad_argument("WriteToSubresource: unreadable source");
        return kEInvalidArg;
    }
    CommandEncoder enc;
    const bool is3d = desc_.Dimension == ResourceDimension::Texture3D;
    enc.write_texture(id_, mip, box.left, box.top, is3d ? box.front : layer, bw, bh, bd, row_pitch,
                      row_pitch != 0 ? depth_pitch / row_pitch : rows, bytes.data(), static_cast<uint32_t>(len));
    device_->submit(enc);
    device_->note_upload(len);
    return kSOk;
}

HResult Resource::ReadFromSubresource(com::CallContext&) {
    device_->note_unsupported_copy("ReadFromSubresource: GPU -> CPU readback has no packet");
    return kENotImpl;
}

HResult Resource::GetHeapProperties(GuestPtr<GuestHeapProperties> properties, GuestPtr<uint32_t> flags) {
    if (placement_.reserved) {
        return kEInvalidArg;  // D3D12: reserved resources have no heap
    }
    GuestHeapProperties p = placement_.placed_heap ? placement_.placed_heap->desc().Properties : placement_.heap;
    if (properties) {
        properties.write(p);
    }
    if (flags) {
        flags.write(placement_.placed_heap ? placement_.placed_heap->desc().Flags : placement_.heap_flags);
    }
    return kSOk;
}

ResourceDesc1Result Resource::GetDesc1() {
    ResourceDesc1Result r{};
    std::memcpy(r.bytes, &desc_, 52);  // the shared D3D12_RESOURCE_DESC prefix; mip region 0
    return r;
}

// ---------------------------------------------------------------------------------------------
// Device: resources, heaps
// ---------------------------------------------------------------------------------------------

namespace {
bool validate_desc(Device& dev, const GuestResourceDesc& d) {
    if (d.Dimension < ResourceDimension::Buffer || d.Dimension > ResourceDimension::Texture3D || d.Width == 0 ||
        d.Layout > 3) {
        dev.note_bad_argument("invalid resource description (dimension " + std::to_string(d.Dimension) + ", width " +
                              std::to_string(d.Width) + ", layout " + std::to_string(d.Layout) + ")");
        return false;
    }
    if (d.Dimension == ResourceDimension::Buffer && (d.Format != 0 || d.Height > 1)) {
        dev.note_bad_argument("invalid buffer description (format " + std::to_string(d.Format) + ")");
        return false;
    }
    if (d.Dimension != ResourceDimension::Buffer &&
        (d.Width > 0x10000 || d.Height == 0 || d.DepthOrArraySize == 0 || !dxgi_format_layout(d.Format).known)) {
        dev.note_bad_argument("invalid texture description (" + std::to_string(d.Width) + "x" +
                              std::to_string(d.Height) + "x" + std::to_string(d.DepthOrArraySize) + ", format " +
                              std::to_string(d.Format) + ")");
        return false;
    }
    return true;
}
}  // namespace

HResult Device::create_resource(const GuestResourceDesc& desc, const GuestHeapProperties& heap, uint32_t heap_flags,
                                uint32_t initial_state, com::Ref<Heap> placed_heap, uint64_t heap_offset,
                                uint64_t address, bool reserved, const Guid& riid, OutPtr out) {
    if (!validate_desc(*this, desc)) {
        return kEInvalidArg;
    }
    if (placed_heap && heap_offset + (desc.Dimension == ResourceDimension::Buffer ? desc.Width : 0) >
                           placed_heap->desc().SizeInBytes) {
        note_bad_argument("placed resource exceeds its heap");
        return kEInvalidArg;
    }
    Resource::Placement pl;
    pl.heap = heap;
    pl.heap_flags = heap_flags;
    pl.placed_heap = std::move(placed_heap);
    pl.heap_offset = heap_offset;
    pl.address = address;
    pl.reserved = reserved;
    auto resource =
        com::create_object<Resource>(com::implements(kID3D12Resource), device_ref(this), desc, std::move(pl), initial_state);
    if (resource && !resource->initialize()) {
        return kEOutOfMemory;
    }
    return hand_out(std::move(resource), riid, out);
}

HResult Device::CreateCommittedResource(com::CallContext& ctx) {
    // (pHeapProperties, HeapFlags, pDesc, InitialResourceState, pOptimizedClearValue, riid, ppvResource)
    GuestHeapProperties props{};
    GuestResourceDesc desc{};
    Guid riid{};
    if (!com::read_guest(ctx.int_arg(1), props) || !com::read_guest(ctx.int_arg(3), desc) ||
        !com::read_guest(ctx.int_arg(6), riid)) {
        note_bad_argument("CreateCommittedResource: unreadable arguments");
        return kEInvalidArg;
    }
    return create_resource(desc, props, static_cast<uint32_t>(ctx.int_arg(2)), static_cast<uint32_t>(ctx.int_arg(4)),
                           {}, 0, 0, false, riid, OutPtr(ctx.int_arg(7)));
}

HResult Device::CreateCommittedResource3(com::CallContext& ctx) {
    // (pHeapProperties, HeapFlags, pDesc (DESC1), InitialLayout, pOptimizedClearValue,
    //  pProtectedSession, NumCastableFormats, pCastableFormats, riid, ppvResource)
    GuestHeapProperties props{};
    GuestResourceDesc desc{};
    Guid riid{};
    if (!com::read_guest(ctx.int_arg(1), props) || !com::read_guest(ctx.int_arg(3), &desc, 52) ||
        !com::read_guest(ctx.int_arg(9), riid)) {
        note_bad_argument("CreateCommittedResource (Xbox slot 127): unreadable arguments");
        return kEInvalidArg;
    }
    // Enhanced-barrier layouts start every resource in COMMON for the state tracker.
    return create_resource(desc, props, static_cast<uint32_t>(ctx.int_arg(2)), d3d::ResourceStates::Common, {}, 0, 0,
                           false, riid, OutPtr(ctx.int_arg(10)));
}

HResult Device::CreateHeap(const GuestHeapDesc* desc, const Guid& riid, OutPtr out) {
    if (desc == nullptr || desc->SizeInBytes == 0 || desc->Properties.Type < 1 || desc->Properties.Type > 4) {
        note_bad_argument("CreateHeap: invalid descriptor");
        return kEInvalidArg;
    }
    auto heap = com::create_object<Heap>(com::implements(kID3D12Heap), device_ref(this), *desc);
    if (heap && !heap->valid()) {
        note_bad_argument("CreateHeap: out of guest memory for " + std::to_string(desc->SizeInBytes) + " bytes");
        return kEOutOfMemory;
    }
    return hand_out(std::move(heap), riid, out);
}

HResult Device::CreatePlacedResource(com::CallContext& ctx) {
    // (pHeap, HeapOffset, pDesc, InitialState, pOptimizedClearValue, riid, ppvResource)
    auto heap_arg = InterfaceArg<IHeapHost>::resolve(ctx.int_arg(1));
    GuestResourceDesc desc{};
    Guid riid{};
    if (!heap_arg || !com::read_guest(ctx.int_arg(3), desc) || !com::read_guest(ctx.int_arg(6), riid)) {
        note_bad_argument("CreatePlacedResource: invalid heap or unreadable arguments");
        return kEInvalidArg;
    }
    auto heap = com::Ref<Heap>::share(static_cast<Heap*>(heap_arg.get()));
    return create_resource(desc, GuestHeapProperties{}, 0, static_cast<uint32_t>(ctx.int_arg(4)), std::move(heap),
                           ctx.int_arg(2), 0, false, riid, OutPtr(ctx.int_arg(7)));
}

HResult Device::CreatePlacedResourceX(com::CallContext& ctx) {
    // Xbox slot 133: (address, pDesc, InitialState, pOptimizedClearValue, riid, ppvResource)
    const uint64_t address = ctx.int_arg(1);
    GuestResourceDesc desc{};
    Guid riid{};
    if (address == 0 || !com::read_guest(ctx.int_arg(2), &desc, 52) || !com::read_guest(ctx.int_arg(5), riid)) {
        note_bad_argument("CreatePlacedResource (Xbox slot 133): unreadable arguments");
        return kEInvalidArg;
    }
    GuestHeapProperties props{};
    props.Type = HeapType::Custom;
    props.CPUPageProperty = 2;
    return create_resource(desc, props, 0, static_cast<uint32_t>(ctx.int_arg(3)), {}, 0, address, false, riid,
                           OutPtr(ctx.int_arg(6)));
}

HResult Device::CreateReservedResource(uint64_t desc_va, uint32_t state, uint64_t, const Guid& riid, OutPtr out) {
    GuestResourceDesc desc{};
    if (!com::read_guest(desc_va, desc)) {
        note_bad_argument("CreateReservedResource: unreadable descriptor");
        return kEInvalidArg;
    }
    return create_resource(desc, GuestHeapProperties{}, 0, state, {}, 0, 0, true, riid, out);
}

// ---------------------------------------------------------------------------------------------
// Device: views
// ---------------------------------------------------------------------------------------------

namespace {
uint32_t resource_id(const InterfaceArg<IResourceHost>& r) {
    return r ? static_cast<Resource*>(r.get())->id() : 0u;
}
}  // namespace

void Device::CreateShaderResourceView(InterfaceArg<IResourceHost> resource, uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestShaderResourceViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateShaderResourceView: unreadable descriptor");
            return;
        }
        vd.format = g.Format;
        vd.dimension = g.ViewDimension;
        vd.component_mapping = g.Shader4ComponentMapping;
        const uint32_t* u = g.u;
        switch (g.ViewDimension) {
        case d3d::ViewDimension::Buffer:
            vd.first_element = u64_at(u, 0);
            vd.num_elements = u[2];
            vd.structure_byte_stride = u[3];
            vd.flags = u[4];
            break;
        case d3d::ViewDimension::Texture1D:
        case d3d::ViewDimension::Texture3D:
        case d3d::ViewDimension::TextureCube:
            vd.first_mip = u[0];
            vd.mip_levels = mips_or_all(u[1]);
            vd.min_lod_clamp = f32(u[2]);
            break;
        case d3d::ViewDimension::Texture1DArray:
        case d3d::ViewDimension::TextureCubeArray:
            vd.first_mip = u[0];
            vd.mip_levels = mips_or_all(u[1]);
            vd.first_array_slice = u[2];
            vd.array_size = u[3];
            vd.min_lod_clamp = f32(u[4]);
            break;
        case d3d::ViewDimension::Texture2D:
            vd.first_mip = u[0];
            vd.mip_levels = mips_or_all(u[1]);
            vd.plane_slice = u[2];
            vd.min_lod_clamp = f32(u[3]);
            break;
        case d3d::ViewDimension::Texture2DArray:
            vd.first_mip = u[0];
            vd.mip_levels = mips_or_all(u[1]);
            vd.first_array_slice = u[2];
            vd.array_size = u[3];
            vd.plane_slice = u[4];
            vd.min_lod_clamp = f32(u[5]);
            break;
        case d3d::ViewDimension::Texture2DMs:
            break;
        case d3d::ViewDimension::Texture2DMsArray:
            vd.first_array_slice = u[0];
            vd.array_size = u[1];
            break;
        case 11:  // RAYTRACING_ACCELERATION_STRUCTURE: Location
            vd.first_element = u64_at(u, 0);
            break;
        default:
            note_bad_argument("CreateShaderResourceView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    const d3d::Status s = heaps_.create_shader_resource_view(resource_id(resource), desc, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateShaderResourceView: ") + d3d::status_name(s));
    }
}

void Device::CreateUnorderedAccessView(InterfaceArg<IResourceHost> resource, InterfaceArg<IResourceHost> counter,
                                       uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestUnorderedAccessViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateUnorderedAccessView: unreadable descriptor");
            return;
        }
        vd.format = g.Format;
        vd.dimension = g.ViewDimension;
        const uint32_t* u = g.u;
        switch (g.ViewDimension) {
        case d3d::ViewDimension::Buffer:
            vd.first_element = u64_at(u, 0);
            vd.num_elements = u[2];
            vd.structure_byte_stride = u[3];
            vd.counter_offset_in_bytes = u64_at(u, 4);
            vd.flags = u[6];
            break;
        case d3d::ViewDimension::Texture1D:
            vd.first_mip = u[0];
            break;
        case d3d::ViewDimension::Texture1DArray:
        case d3d::ViewDimension::Texture3D:
            vd.first_mip = u[0];
            vd.first_array_slice = u[1];
            vd.array_size = u[2];
            break;
        case d3d::ViewDimension::Texture2D:
            vd.first_mip = u[0];
            vd.plane_slice = u[1];
            break;
        case d3d::ViewDimension::Texture2DArray:
            vd.first_mip = u[0];
            vd.first_array_slice = u[1];
            vd.array_size = u[2];
            vd.plane_slice = u[3];
            break;
        case d3d::ViewDimension::Texture2DMs:
            break;
        case d3d::ViewDimension::Texture2DMsArray:
            vd.first_array_slice = u[0];
            vd.array_size = u[1];
            break;
        default:
            note_bad_argument("CreateUnorderedAccessView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    const d3d::Status s = heaps_.create_unordered_access_view(resource_id(resource), resource_id(counter), desc,
                                                              d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateUnorderedAccessView: ") + d3d::status_name(s));
    }
}

void Device::CreateRenderTargetView(InterfaceArg<IResourceHost> resource, uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestRenderTargetViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateRenderTargetView: unreadable descriptor");
            return;
        }
        vd.format = g.Format;
        vd.dimension = g.ViewDimension;
        const uint32_t* u = g.u;
        switch (g.ViewDimension) {
        case d3d::ViewDimension::Buffer:
            vd.first_element = u64_at(u, 0);
            vd.num_elements = u[2];
            break;
        case d3d::ViewDimension::Texture1D:
            vd.first_mip = u[0];
            break;
        case d3d::ViewDimension::Texture1DArray:
        case d3d::ViewDimension::Texture3D:
            vd.first_mip = u[0];
            vd.first_array_slice = u[1];
            vd.array_size = u[2];
            break;
        case d3d::ViewDimension::Texture2D:
            vd.first_mip = u[0];
            vd.plane_slice = u[1];
            break;
        case d3d::ViewDimension::Texture2DArray:
            vd.first_mip = u[0];
            vd.first_array_slice = u[1];
            vd.array_size = u[2];
            vd.plane_slice = u[3];
            break;
        case d3d::ViewDimension::Texture2DMs:
            break;
        case d3d::ViewDimension::Texture2DMsArray:
            vd.first_array_slice = u[0];
            vd.array_size = u[1];
            break;
        default:
            note_bad_argument("CreateRenderTargetView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    const d3d::Status s = heaps_.create_render_target_view(resource_id(resource), desc, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateRenderTargetView: ") + d3d::status_name(s));
    }
}

void Device::CreateDepthStencilView(InterfaceArg<IResourceHost> resource, uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestDepthStencilViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateDepthStencilView: unreadable descriptor");
            return;
        }
        vd.format = g.Format;
        vd.dimension = g.ViewDimension;
        vd.flags = g.Flags;
        const uint32_t* u = g.u;
        switch (g.ViewDimension) {
        case d3d::DsvDimension::Texture1D:
        case d3d::DsvDimension::Texture2D:
            vd.first_mip = u[0];
            break;
        case d3d::DsvDimension::Texture1DArray:
        case d3d::DsvDimension::Texture2DArray:
            vd.first_mip = u[0];
            vd.first_array_slice = u[1];
            vd.array_size = u[2];
            break;
        case d3d::DsvDimension::Texture2DMs:
            break;
        case d3d::DsvDimension::Texture2DMsArray:
            vd.first_array_slice = u[0];
            vd.array_size = u[1];
            break;
        default:
            note_bad_argument("CreateDepthStencilView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    const d3d::Status s = heaps_.create_depth_stencil_view(resource_id(resource), desc, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateDepthStencilView: ") + d3d::status_name(s));
    }
}

// ---------------------------------------------------------------------------------------------
// Pipeline states
// ---------------------------------------------------------------------------------------------

PipelineState::PipelineState(com::Ref<Device> device, com::Ref<RootSignature> root_signature)
    : ChildBase(std::move(device)), root_signature_(std::move(root_signature)) {
    device_->pipeline_states.fetch_add(1);
}

PipelineState::~PipelineState() {
    if (graphics_) {
        CommandEncoder enc;
        device_->state_bridge().release_pipeline_state(std::move(graphics_), enc);
        if (enc.size() != 0) {
            device_->submit(enc);
        }
    }
    device_->pipeline_states.fetch_sub(1);
}

HResult PipelineState::GetCachedBlob(OutPtr blob) {
    if (blob.is_null()) {
        return kEPointer;
    }
    auto b = Blob::create(nullptr, 0);
    blob.write(b ? b.detach(kID3DBlob) : 0);
    return b ? kSOk : kEOutOfMemory;
}

namespace {

// Reads a shader's bytecode and returns its identity (0 = no shader). False on unreadable input.
bool read_shader(Device& dev, const GuestShaderBytecode& code, const char* stage, uint64_t* identity) {
    *identity = 0;
    if (code.pShaderBytecode == 0 || code.BytecodeLength == 0) {
        return true;
    }
    if (code.BytecodeLength > kMaxShaderBytes) {
        dev.note_bad_argument(std::string("PSO: ") + stage + " bytecode too large");
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(code.BytecodeLength));
    if (!com::read_guest(code.pShaderBytecode, bytes.data(), bytes.size())) {
        dev.note_bad_argument(std::string("PSO: unreadable ") + stage + " bytecode at " + hex(code.pShaderBytecode));
        return false;
    }
    *identity = shader_identity(bytes.data(), bytes.size());
    return true;
}

com::Ref<RootSignature> root_signature_ref(uint64_t va) {
    if (va == 0) {
        return {};
    }
    auto arg = InterfaceArg<IRootSignatureHost>::resolve(va);
    return arg ? com::Ref<RootSignature>::share(static_cast<RootSignature*>(arg.get())) : com::Ref<RootSignature>{};
}

}  // namespace

GraphicsPsoSource& GraphicsPsoSource::operator=(const GraphicsPsoSource& other) {
    if (this != &other) {
        desc = other.desc;
        semantic_names = other.semantic_names;
        input = other.input;
        root_signature = other.root_signature;
        fix_pointers();
    }
    return *this;
}

void GraphicsPsoSource::fix_pointers() {
    for (size_t i = 0; i < input.size() && i < semantic_names.size(); ++i) {
        input[i].SemanticName = semantic_names[i].c_str();
    }
    desc.pInputElementDescs = input.empty() ? nullptr : input.data();
    desc.NumInputElements = static_cast<uint32_t>(input.size());
}

void convert_blend_desc(const GuestBlendDesc& g, pso::D3D12BlendDesc* out) {
    out->AlphaToCoverageEnable = g.AlphaToCoverageEnable != 0;
    out->IndependentBlendEnable = g.IndependentBlendEnable != 0;
    for (int i = 0; i < 8; ++i) {
        const GuestRenderTargetBlendDesc& s = g.RenderTarget[i];
        pso::D3D12RenderTargetBlendDesc& t = out->RenderTarget[i];
        t.BlendEnable = s.BlendEnable != 0;
        t.LogicOpEnable = s.LogicOpEnable != 0;
        t.SrcBlend = static_cast<pso::D3D12Blend>(s.SrcBlend);
        t.DestBlend = static_cast<pso::D3D12Blend>(s.DestBlend);
        t.BlendOp = static_cast<pso::D3D12BlendOp>(s.BlendOp);
        t.SrcBlendAlpha = static_cast<pso::D3D12Blend>(s.SrcBlendAlpha);
        t.DestBlendAlpha = static_cast<pso::D3D12Blend>(s.DestBlendAlpha);
        t.BlendOpAlpha = static_cast<pso::D3D12BlendOp>(s.BlendOpAlpha);
        t.LogicOp = static_cast<pso::D3D12LogicOp>(s.LogicOp);
        t.RenderTargetWriteMask = s.RenderTargetWriteMask;
    }
}

void convert_rasterizer_desc(const GuestRasterizerDesc& r, pso::D3D12RasterizerDesc* out) {
    out->FillMode = static_cast<pso::D3D12FillMode>(r.FillMode);
    out->CullMode = static_cast<pso::D3D12CullMode>(r.CullMode);
    out->FrontCounterClockwise = r.FrontCounterClockwise != 0;
    out->DepthBias = r.DepthBias;
    out->DepthBiasClamp = r.DepthBiasClamp;
    out->SlopeScaledDepthBias = r.SlopeScaledDepthBias;
    out->DepthClipEnable = r.DepthClipEnable != 0;
    out->MultisampleEnable = r.MultisampleEnable != 0;
    out->AntialiasedLineEnable = r.AntialiasedLineEnable != 0;
    out->ForcedSampleCount = r.ForcedSampleCount;
    out->ConservativeRaster = static_cast<pso::D3D12ConservativeRasterizationMode>(r.ConservativeRaster);
}

void convert_depth_stencil_desc(const GuestDepthStencilDesc& ds, pso::D3D12DepthStencilDesc* out) {
    out->DepthEnable = ds.DepthEnable != 0;
    out->DepthWriteMask = static_cast<pso::D3D12DepthWriteMask>(ds.DepthWriteMask);
    out->DepthFunc = static_cast<pso::D3D12ComparisonFunc>(ds.DepthFunc);
    out->StencilEnable = ds.StencilEnable != 0;
    out->StencilReadMask = ds.StencilReadMask;
    out->StencilWriteMask = ds.StencilWriteMask;
    auto op = [](const GuestDepthStencilOpDesc& s) {
        pso::D3D12DepthStencilOpDesc o;
        o.StencilFailOp = static_cast<pso::D3D12StencilOp>(s.StencilFailOp);
        o.StencilDepthFailOp = static_cast<pso::D3D12StencilOp>(s.StencilDepthFailOp);
        o.StencilPassOp = static_cast<pso::D3D12StencilOp>(s.StencilPassOp);
        o.StencilFunc = static_cast<pso::D3D12ComparisonFunc>(s.StencilFunc);
        return o;
    };
    out->FrontFace = op(ds.FrontFace);
    out->BackFace = op(ds.BackFace);
}

void PipelineState::initialize_guest_payload() {
    const uint64_t payload[2] = {xbox::kPipelineStateTag, identity()};
    com::write_guest(payload_address(), payload, sizeof(payload));
}

HResult Device::CreateGraphicsPipelineState(uint64_t desc_va, const Guid& riid, OutPtr out) {
    GuestGraphicsPipelineStateDesc g{};
    if (desc_va == 0 || !com::read_guest(desc_va, g)) {
        note_bad_argument("CreateGraphicsPipelineState: unreadable descriptor");
        return kEInvalidArg;
    }
    return create_graphics_pso(g, riid, out);
}

HResult Device::create_graphics_pso(const GuestGraphicsPipelineStateDesc& g, const Guid& riid, OutPtr out) {
    GraphicsPsoSource src;
    src.root_signature = root_signature_ref(g.pRootSignature);
    if (g.pRootSignature != 0 && !src.root_signature) {
        note_bad_argument("CreateGraphicsPipelineState: invalid root signature " + hex(g.pRootSignature));
        return kEInvalidArg;
    }
    pso::D3D12GraphicsPsoDesc& d = src.desc;
    if (!read_shader(*this, g.VS, "VS", &d.vs) || !read_shader(*this, g.PS, "PS", &d.ps) ||
        !read_shader(*this, g.DS, "DS", &d.ds) || !read_shader(*this, g.HS, "HS", &d.hs) ||
        !read_shader(*this, g.GS, "GS", &d.gs)) {
        return kEInvalidArg;
    }
    if (d.vs == 0) {
        note_bad_argument("CreateGraphicsPipelineState: no vertex shader");
        return kEInvalidArg;
    }
    if (g.NumRenderTargets > 8 || g.InputLayout.NumElements > kMaxInputElements || g.PrimitiveTopologyType > 4) {
        note_bad_argument("CreateGraphicsPipelineState: out-of-range counts or topology type");
        return kEInvalidArg;
    }
    d.stream_output_enabled = g.StreamOutput.NumEntries > 0;
    convert_blend_desc(g.BlendState, &d.BlendState);
    d.SampleMask = g.SampleMask;
    convert_rasterizer_desc(g.RasterizerState, &d.RasterizerState);
    convert_depth_stencil_desc(g.DepthStencilState, &d.DepthStencilState);
    // Input layout: semantic names are guest strings, copied into the source.
    std::vector<GuestInputElementDesc> elements(g.InputLayout.NumElements);
    src.semantic_names.assign(g.InputLayout.NumElements, std::string());
    src.input.assign(g.InputLayout.NumElements, pso::D3D12InputElementDesc{});
    for (uint32_t i = 0; i < g.InputLayout.NumElements; ++i) {
        if (!com::read_guest(g.InputLayout.pInputElementDescs + sizeof(GuestInputElementDesc) * i, elements[i])) {
            note_bad_argument("CreateGraphicsPipelineState: unreadable input element " + std::to_string(i));
            return kEInvalidArg;
        }
        for (uint32_t c = 0; c < 64; ++c) {
            char ch = 0;
            if (!com::read_guest(elements[i].SemanticName + c, &ch, 1)) {
                note_bad_argument("CreateGraphicsPipelineState: unreadable semantic name of element " + std::to_string(i));
                return kEInvalidArg;
            }
            if (ch == 0) {
                break;
            }
            src.semantic_names[i].push_back(ch);
        }
    }
    for (uint32_t i = 0; i < g.InputLayout.NumElements; ++i) {
        const GuestInputElementDesc& e = elements[i];
        pso::D3D12InputElementDesc& t = src.input[i];
        t.SemanticIndex = e.SemanticIndex;
        t.Format = static_cast<pso::DxgiFormat>(e.Format);
        t.InputSlot = e.InputSlot;
        t.AlignedByteOffset = e.AlignedByteOffset;
        t.InputSlotClass = static_cast<pso::D3D12InputClassification>(e.InputSlotClass);
        t.InstanceDataStepRate = e.InstanceDataStepRate;
    }
    d.IBStripCutValue = static_cast<pso::D3D12IndexBufferStripCutValue>(g.IBStripCutValue);
    d.PrimitiveTopologyType = static_cast<pso::D3D12PrimitiveTopologyType>(g.PrimitiveTopologyType);
    d.NumRenderTargets = g.NumRenderTargets;
    for (int i = 0; i < 8; ++i) {
        d.RTVFormats[i] = static_cast<pso::DxgiFormat>(g.RTVFormats[i]);
    }
    d.DSVFormat = static_cast<pso::DxgiFormat>(g.DSVFormat);
    d.SampleDesc.Count = g.SampleCount;
    d.SampleDesc.Quality = g.SampleQuality;
    return create_graphics_pso_from(std::move(src), riid, out);
}

HResult Device::create_graphics_pso_from(GraphicsPsoSource source, const Guid& riid, OutPtr out) {
    auto src = std::make_unique<GraphicsPsoSource>(source);  // copy: fixes the internal pointers
    const com::Ref<RootSignature>& rs = src->root_signature;
    pso::D3D12GraphicsPsoDesc& d = src->desc;
    d.root_signature_id = rs ? static_cast<uint32_t>(rs->signature().hash()) : 0u;
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this), rs);
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->graphics_ = bridge_->create_graphics_pipeline_state(d, rs ? &rs->signature() : nullptr);
    state->identities_ = {d.vs, d.ps};
    state->resolved_ = (state->graphics_->diagnostics() & bridge::BridgeDiag::UnknownShaderModule) == 0;
    if (!state->resolved_) {
        for (uint64_t id : {d.vs, d.ps}) {
            if (id != 0 && !shader_known(id)) {
                note_unresolved_shader(id);
            }
        }
    }
    state->source_ = std::move(src);
    return hand_out(std::move(state), riid, out);
}

HResult Device::create_placeholder_pso(const std::string& reason, const Guid& riid, OutPtr out) {
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this),
                                                   com::Ref<RootSignature>{});
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->unsupported_ = reason;
    note_bad_argument("pipeline state not supported (" + reason + "): draws with it are skipped");
    return hand_out(std::move(state), riid, out);
}

HResult Device::CreateComputePipelineState(uint64_t desc_va, const Guid& riid, OutPtr out) {
    GuestComputePipelineStateDesc g{};
    if (desc_va == 0 || !com::read_guest(desc_va, g)) {
        note_bad_argument("CreateComputePipelineState: unreadable descriptor");
        return kEInvalidArg;
    }
    return create_compute_pso(g, riid, out);
}

HResult Device::create_compute_pso(const GuestComputePipelineStateDesc& g, const Guid& riid, OutPtr out) {
    com::Ref<RootSignature> rs = root_signature_ref(g.pRootSignature);
    if (g.pRootSignature != 0 && !rs) {
        note_bad_argument("CreateComputePipelineState: invalid root signature " + hex(g.pRootSignature));
        return kEInvalidArg;
    }
    pso::D3D12ComputePsoDesc d{};
    if (!read_shader(*this, g.CS, "CS", &d.cs)) {
        return kEInvalidArg;
    }
    if (d.cs == 0) {
        note_bad_argument("CreateComputePipelineState: no compute shader");
        return kEInvalidArg;
    }
    d.root_signature_id = rs ? static_cast<uint32_t>(rs->signature().hash()) : 0u;
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this), rs);
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->compute_ = bridge_->create_compute_pipeline_state(d, rs ? &rs->signature() : nullptr);
    state->identities_ = {d.cs};
    state->resolved_ = (state->compute_->diagnostics() & bridge::BridgeDiag::UnknownShaderModule) == 0;
    if (!state->resolved_ && !shader_known(d.cs)) {
        note_unresolved_shader(d.cs);
    }
    return hand_out(std::move(state), riid, out);
}

// ---------------------------------------------------------------------------------------------
// Vtables
// ---------------------------------------------------------------------------------------------

namespace {

using ResM = Methods<IResourceHost>;
constexpr com::MethodDesc kResourceMethods[] = {
    ResM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    ResM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    ResM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    ResM::method<&IObjectHost::SetName>("SetName"),
    ResM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    ResM::method<&IResourceHost::Map>("Map"),                                   // 8
    ResM::method<&IResourceHost::Unmap>("Unmap"),                               // 9
    ResM::sret<&IResourceHost::GetDesc>("GetDesc"),                             // 10
    ResM::method<&IResourceHost::GetGPUVirtualAddress>("GetGPUVirtualAddress"), // 11
    ResM::method<&IResourceHost::WriteToSubresource>("WriteToSubresource"),     // 12
    ResM::method<&IResourceHost::ReadFromSubresource>("ReadFromSubresource"),   // 13
    ResM::method<&IResourceHost::GetHeapProperties>("GetHeapProperties"),       // 14
    ResM::stub("GetProtectedResourceSession"),                                  // 15 (ID3D12Resource1)
    ResM::sret<&IResourceHost::GetDesc1>("GetDesc1"),                           // 16 (ID3D12Resource2)
};
constexpr Guid kResourceIids[] = {
    IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable,
    com::make_guid("9d5e227a-4430-4161-88b3-3eca6bb16e19"),  // ID3D12Resource1
    com::make_guid("be36ec3b-ea85-4aeb-a45a-e9d76404a495"),  // ID3D12Resource2
};

using HeapObjM = Methods<IHeapHost>;
constexpr com::MethodDesc kHeapMethods[] = {
    HeapObjM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    HeapObjM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    HeapObjM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    HeapObjM::method<&IObjectHost::SetName>("SetName"),
    HeapObjM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    HeapObjM::sret<&IHeapHost::GetDesc>("GetDesc"),  // 8
};
constexpr Guid kHeapIids[] = {IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable,
                              com::make_guid("572f7389-2168-49e3-9693-d6df5871bf6d")};  // ID3D12Heap1

using PsoM = Methods<IPipelineStateHost>;
constexpr com::MethodDesc kPipelineStateMethods[] = {
    PsoM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    PsoM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    PsoM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    PsoM::method<&IObjectHost::SetName>("SetName"),
    PsoM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    PsoM::method<&IPipelineStateHost::GetCachedBlob>("GetCachedBlob"),  // 8
};
constexpr Guid kPageable[] = {IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable};

constexpr com::InterfaceOptions opts(uint32_t slots, std::span<const Guid> extra) {
    com::InterfaceOptions o;
    o.slot_count = slots;
    o.policy = com::SlotPolicy::ReturnNotImpl;
    o.extra_iids = extra;
    return o;
}

}  // namespace

const com::Interface<IResourceHost> kID3D12Resource = com::declare_interface<IResourceHost>(
    "ID3D12Resource", IID_ID3D12Resource, kResourceMethods, opts(24, kResourceIids));
const com::Interface<IHeapHost> kID3D12Heap =
    com::declare_interface<IHeapHost>("ID3D12Heap", IID_ID3D12Heap, kHeapMethods, opts(16, kHeapIids));
const com::Interface<IPipelineStateHost> kID3D12PipelineState = com::declare_interface<IPipelineStateHost>(
    "ID3D12PipelineState", IID_ID3D12PipelineState, kPipelineStateMethods, opts(16, kPageable));

}  // namespace playgta6::gpu::d3d12_guest
