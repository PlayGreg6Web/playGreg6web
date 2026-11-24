#include "gpu/d3d12_guest_objects.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>

#include "os/os_shim.h"
#include "os/win32_kernel.h"
#include "os/xmem_windows.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

constexpr size_t kMaxDiagnostics = 256;
constexpr uint32_t kMaxRangesPerTable = 4096;

std::atomic<uint64_t> g_sret_this_first{0};
std::atomic<uint64_t> g_sret_result_first{0};

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%" PRIx64, v);
    return b;
}

// Reads `count` elements of `stride` bytes (>= sizeof(T)) starting at guest `va`.
template <typename T>
bool read_array(uint64_t va, uint32_t count, size_t stride, std::vector<T>* out) {
    out->assign(count, T{});
    if (count == 0) {
        return true;
    }
    if (va == 0) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (!com::read_guest(va + stride * i, &(*out)[i], sizeof(T))) {
            return false;
        }
    }
    return true;
}

constexpr uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Struct returns
// ---------------------------------------------------------------------------------------------

SretStats sret_stats() {
    return SretStats{g_sret_this_first.load(), g_sret_result_first.load()};
}

void reset_sret_stats() {
    g_sret_this_first = 0;
    g_sret_result_first = 0;
}

void detail::note_sret(bool this_first) {
    (this_first ? g_sret_this_first : g_sret_result_first).fetch_add(1, std::memory_order_relaxed);
}

bool detail::resolve_sret_this(GuestState& state, const void* host_tag, com::detail::Resolved* out,
                               bool* this_first) {
    const uint64_t rcx = state.gpr[GuestReg::Rcx];
    const uint64_t rdx = state.gpr[GuestReg::Rdx];
    if (com::detail::lookup_interface(rcx, out) && out->iface->host_tag == host_tag) {
        *this_first = true;
        note_sret(true);
        return true;
    }
    if (com::detail::lookup_interface(rdx, out) && out->iface->host_tag == host_tag) {
        *this_first = false;
        note_sret(false);
        return true;
    }
    // Neither register holds the object: diagnose and fault exactly like a method thunk does.
    *out = com::detail::resolve_this_or_fault(state, host_tag);
    *this_first = true;
    return true;
}

// ---------------------------------------------------------------------------------------------
// DXGI format layout (bytes per element/block)
// ---------------------------------------------------------------------------------------------

DxgiFormatLayout dxgi_format_layout(uint32_t format) {
    DxgiFormatLayout l;
    auto set = [&](uint32_t bytes, uint32_t bw = 1, uint32_t bh = 1, uint32_t planes = 1) {
        l.block_bytes = bytes;
        l.block_width = bw;
        l.block_height = bh;
        l.planes = planes;
        l.known = true;
    };
    if (format >= 1 && format <= 4) set(16);
    else if (format >= 5 && format <= 8) set(12);
    else if (format >= 9 && format <= 18) set(8);
    else if (format >= 19 && format <= 22) set(8, 1, 1, 2);
    else if (format >= 23 && format <= 43) set(4);
    else if (format >= 44 && format <= 47) set(4, 1, 1, 2);
    else if (format >= 48 && format <= 59) set(2);
    else if (format >= 60 && format <= 65) set(1);
    else if (format == 67) set(4);
    else if (format == 68 || format == 69) set(4, 2, 1);
    else if (format >= 70 && format <= 72) set(8, 4, 4);
    else if (format >= 73 && format <= 78) set(16, 4, 4);
    else if (format >= 79 && format <= 81) set(8, 4, 4);
    else if (format >= 82 && format <= 84) set(16, 4, 4);
    else if (format == 85 || format == 86) set(2);
    else if (format >= 87 && format <= 93) set(4);
    else if (format >= 94 && format <= 99) set(16, 4, 4);
    else if (format == 115) set(2);
    return l;
}

uint32_t plane_block_bytes(uint32_t format, uint32_t plane) {
    const DxgiFormatLayout l = dxgi_format_layout(format);
    if (l.planes == 2) {
        return plane == 0 ? 4u : 1u;  // depth (D32 / D24) plane, then the 8-bit stencil plane
    }
    return l.block_bytes;
}

// ---------------------------------------------------------------------------------------------
// Private data
// ---------------------------------------------------------------------------------------------

namespace {
std::string guid_key(const Guid& g) {
    return std::string(reinterpret_cast<const char*>(&g), sizeof(g));
}
// WKPDID_D3DDebugObjectNameW: SetName stores the name under this GUID as well.
constexpr Guid kDebugObjectNameW = com::make_guid("4cca5fd8-921f-42c8-8566-70caf2a9b741");
}  // namespace

HResult PrivateData::get(const Guid& guid, GuestPtr<uint32_t> data_size, uint64_t data) {
    uint32_t size = 0;
    if (data_size.is_null() || !data_size.read(size)) {
        return kEInvalidArg;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(guid_key(guid));
    if (it == entries_.end()) {
        data_size.write(0u);
        return kDxgiErrorNotFound;
    }
    const uint32_t stored = static_cast<uint32_t>(it->second.size());
    if (data == 0) {
        data_size.write(stored);
        return kSOk;
    }
    if (size < stored) {
        data_size.write(stored);
        return kDxgiErrorMoreData;
    }
    if (stored != 0 && !com::write_guest(data, it->second.data(), stored)) {
        return kEInvalidArg;
    }
    data_size.write(stored);
    return kSOk;
}

HResult PrivateData::set(const Guid& guid, uint32_t data_size, uint64_t data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (data == 0 || data_size == 0) {
        entries_.erase(guid_key(guid));
        return kSOk;
    }
    std::vector<uint8_t> bytes(data_size);
    if (!com::read_guest(data, bytes.data(), bytes.size())) {
        return kEInvalidArg;
    }
    entries_[guid_key(guid)] = std::move(bytes);
    return kSOk;
}

HResult PrivateData::set_interface(const Guid& guid, uint64_t unknown) {
    // The pointer value is stored (8 bytes); guest objects are not reference counted from here.
    std::lock_guard<std::mutex> lock(mutex_);
    if (unknown == 0) {
        entries_.erase(guid_key(guid));
        return kSOk;
    }
    std::vector<uint8_t> bytes(sizeof(unknown));
    std::memcpy(bytes.data(), &unknown, sizeof(unknown));
    entries_[guid_key(guid)] = std::move(bytes);
    return kSOk;
}

HResult PrivateData::set_name(uint64_t name) {
    std::u16string text;
    if (name != 0) {
        for (uint64_t i = 0; i < 4096; ++i) {
            char16_t c = 0;
            if (!com::read_guest(name + 2 * i, &c, sizeof(c))) {
                return kEInvalidArg;
            }
            if (c == 0) {
                break;
            }
            text.push_back(c);
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    name_ = text;
    std::vector<uint8_t> bytes((text.size() + 1) * sizeof(char16_t), 0);
    std::memcpy(bytes.data(), text.data(), text.size() * sizeof(char16_t));
    entries_[guid_key(kDebugObjectNameW)] = std::move(bytes);
    return kSOk;
}

std::u16string PrivateData::name() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return name_;
}

// ---------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------

Device::Device(DeviceOptions options) : options_(std::move(options)) {
    // The title creates constant-buffer views at 16-byte aligned addresses and sizes (Xbox; PC
    // D3D12 requires 256).
    heaps_.set_cbv_alignment(16);
    bridge::BridgeOptions bo = options_.bridge;
    bo.id_allocator = [this] { return allocate_id(); };
    bo.relocate_uniform = [this](uint64_t va, uint64_t size, uint32_t* id, uint64_t* offset) {
        return relocate_uniform(va, size, id, offset);
    };
    bridge_ = std::make_unique<bridge::StateBridgeDevice>(heaps_, std::move(bo), &resource_states_);
    sink_ = options_.sink;
    // O27: the XMem CPU/GPU window pairs, whether reserved before or after this device.
    alias_listener_ = os::xmem_windows::add_alias_listener([this](const os::xmem_windows::AliasPair& p) {
        register_gpu_alias(p.gpu_base, p.cpu_base, p.size);
    });
    if (!options_.dump_dir.empty()) {
        dump_ = std::make_unique<PipelineDump>(options_.dump_dir);
        if (!dump_->ok()) {
            os::log("d3d12_xs: cannot create the pipeline dump directory " + options_.dump_dir);
            dump_.reset();
        }
    }
}

void Device::flush_pipeline_dump() {
    if (dump_) {
        dump_->write_manifest();
    }
}

Device::~Device() {
    flush_pipeline_dump();
    os::xmem_windows::remove_alias_listener(alias_listener_);
    pacer_.shutdown();
}

void Device::initialize_guest_payload() {
    const uint8_t one = 1;
    com::write_guest(identity() + kInlineApiFlagOffset, &one, sizeof(one));
}

void Device::set_sink(PacketSink sink) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    sink_ = std::move(sink);
}

void Device::submit(const uint8_t* data, size_t size) {
    submit_parts({{data, size}});
}

void Device::submit_parts(const std::vector<std::pair<const uint8_t*, size_t>>& parts) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    auto emit = [&](const uint8_t* bytes, size_t n) {
        if (n == 0) {
            return;
        }
        if (sink_) {
            sink_(bytes, n);
        } else {
            os::wgpu_submit(bytes, n);
        }
    };
    if (!frame_open_) {
        CommandEncoder begin;
        begin.begin_frame();
        emit(begin.data(), begin.size());
        frame_open_ = true;
    }
    for (const auto& [data, size] : parts) {
        emit(data, size);
    }
    ++submissions_;
}

void Device::present() {
    if (dump_ && dump_presents_.fetch_add(1) % 60 == 59) {
        dump_->write_manifest();
    }
    CommandEncoder enc;
    enc.present();
    std::lock_guard<std::mutex> lock(submit_mutex_);
    CommandEncoder begin;
    if (!frame_open_) {
        begin.begin_frame();
    }
    for (const CommandEncoder* e : {&begin, &enc}) {
        if (e->size() == 0) {
            continue;
        }
        if (sink_) {
            sink_(e->data(), e->size());
        } else {
            os::wgpu_submit(e->data(), e->size());
        }
    }
    frame_open_ = false;
    ++presents_;
}

DeviceStats Device::stats() const {
    DeviceStats s;
    s.queues = queues.load();
    s.command_allocators = allocators.load();
    s.descriptor_heaps = descriptor_heaps.load();
    s.root_signatures = root_signatures.load();
    s.fences = fences.load();
    {
        std::lock_guard<std::mutex> lock(submit_mutex_);
        s.presents = presents_;
        s.submissions = submissions_;
    }
    s.bad_arguments = bad_arguments_.load();
    s.resources = resources.load();
    s.heaps = memory_heaps.load();
    s.pipeline_states = pipeline_states.load();
    s.command_lists = command_lists.load();
    s.upload_bytes = upload_bytes_.load();
    s.skipped_draws = skipped_draws_.load();
    s.unsupported_copies = unsupported_copies_.load();
    return s;
}

void Device::note_bad_argument(const std::string& message) {
    bad_arguments_.fetch_add(1, std::memory_order_relaxed);
    const std::string line = "d3d12_xs: " + message;
    os::log(line);
    std::lock_guard<std::mutex> lock(diag_mutex_);
    if (diags_.size() < kMaxDiagnostics) {
        diags_.push_back(line);
    }
}

std::vector<std::string> Device::diagnostics() const {
    std::lock_guard<std::mutex> lock(diag_mutex_);
    return diags_;
}

com::Ref<Device> device_ref(Device* device) {
    return com::Ref<Device>::share(device);
}

HResult Device::create_queue(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) {
    if (desc == nullptr) {
        note_bad_argument("CreateCommandQueue: unreadable descriptor");
        return kEInvalidArg;
    }
    // DIRECT 0, COMPUTE 2, COPY 3 (BUNDLE 1 cannot be a queue; video queues are not supported).
    // Flags: bit 0 is DISABLE_GPU_TIMEOUT; bit 1 is an Xbox queue flag the title sets on its
    // compute queues (flags 1, 2 and 3 on three of them). Both only steer scheduling on the
    // console; queues here execute the same way whatever they say, so they are accepted.
    constexpr uint32_t kKnownQueueFlags = 0x3;
    if ((desc->Type != 0 && desc->Type != 2 && desc->Type != 3) || (desc->Flags & ~kKnownQueueFlags) != 0) {
        note_bad_argument("CreateCommandQueue: unsupported type " + std::to_string(desc->Type) + " / flags " +
                          std::to_string(desc->Flags));
        return kEInvalidArg;
    }
    auto queue = com::create_object<CommandQueue>(com::implements(kID3D12CommandQueue), device_ref(this), *desc);
    return hand_out(std::move(queue), riid, out);
}

HResult Device::CreateCommandQueue(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) {
    return create_queue(desc, riid, out);
}

HResult Device::CreateCommandQueueX(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) {
    return create_queue(desc, riid, out);
}

HResult Device::CreateCommandAllocator(uint32_t type, const Guid& riid, OutPtr out) {
    if (type > 3) {
        note_bad_argument("CreateCommandAllocator: unsupported type " + std::to_string(type));
        return kEInvalidArg;
    }
    auto allocator =
        com::create_object<CommandAllocator>(com::implements(kID3D12CommandAllocator), device_ref(this), type);
    return hand_out(std::move(allocator), riid, out);
}

HResult Device::CheckFeatureSupport(uint32_t feature, uint64_t data, uint32_t size) {
    auto fill = [&](const void* value, uint32_t expected) -> HResult {
        if (size != expected || data == 0) {
            note_bad_argument("CheckFeatureSupport(" + std::to_string(feature) + "): size " + std::to_string(size) +
                              " != " + std::to_string(expected));
            return kEInvalidArg;
        }
        return com::write_guest(data, value, expected) ? kSOk : kEInvalidArg;
    };
    auto read_u32 = [&](uint32_t offset, uint32_t* v) { return data != 0 && com::read_guest(data + offset, v, 4); };
    switch (feature) {
    case 0: {  // D3D12_OPTIONS (15 x u32)
        uint32_t o[15] = {};
        o[3] = 0;   // TiledResourcesTier: not supported
        o[4] = 3;   // ResourceBindingTier 3 (G6: unbounded ranges, full heaps)
        o[9] = 40;  // MaxGPUVirtualAddressBitsPerResource
        o[14] = 2;  // ResourceHeapTier 2
        return fill(o, sizeof(o));
    }
    case 1: {  // ARCHITECTURE { NodeIndex, TileBasedRenderer, UMA, CacheCoherentUMA }
        uint32_t node = 0;
        if (size != 16 || !read_u32(0, &node)) {
            return fill(nullptr, 16);
        }
        const uint32_t a[4] = {node, 0, 1, 1};
        return fill(a, sizeof(a));
    }
    case 2: {  // FEATURE_LEVELS { u32 Num; u64 pRequested; u32 MaxSupported }
        if (size != 24 || data == 0) {
            return fill(nullptr, 24);
        }
        uint32_t num = 0;
        uint64_t requested = 0;
        read_u32(0, &num);
        com::read_guest(data + 8, &requested, sizeof(requested));
        std::vector<uint32_t> levels;
        if (num > 64 || !read_array(requested, num, 4, &levels)) {
            note_bad_argument("CheckFeatureSupport(FEATURE_LEVELS): unreadable level list");
            return kEInvalidArg;
        }
        uint32_t best = 0;
        for (uint32_t level : levels) {
            if (level <= 0xc200u && level > best) {
                best = level;  // up to D3D_FEATURE_LEVEL_12_2
            }
        }
        return com::write_guest(data + 16, &best, 4) ? kSOk : kEInvalidArg;
    }
    case 3: {  // FORMAT_SUPPORT { Format, Support1, Support2 }
        uint32_t format = 0;
        if (size != 12 || !read_u32(0, &format)) {
            return fill(nullptr, 12);
        }
        uint32_t support1 = 0;
        const auto dxgi = static_cast<pso::DxgiFormat>(format);
        const auto sampled = pso::map_texture_format(dxgi, pso::FormatUsage::Sampled, options_.bridge.features);
        const auto rt = pso::map_texture_format(dxgi, pso::FormatUsage::RenderTarget, options_.bridge.features);
        const auto ds = pso::map_texture_format(dxgi, pso::FormatUsage::DepthStencil, options_.bridge.features);
        if (format == 0) {
            support1 = 0x1;  // BUFFER
        }
        if (sampled.supported()) {
            support1 |= 0x1 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x4000;  // textures, load, sample, mip
        }
        if (rt.supported() && pso::is_color_renderable(rt.format, options_.bridge.features)) {
            support1 |= 0x10000 | 0x200000;  // RENDER_TARGET, MULTISAMPLE_RENDERTARGET
            if (pso::is_blendable(rt.format, options_.bridge.features)) {
                support1 |= 0x20000;  // BLENDABLE
            }
        }
        if (ds.supported() && pso::has_depth_aspect(ds.format)) {
            support1 |= 0x40000;  // DEPTH_STENCIL
        }
        if (format == 57 || format == 42) {
            support1 |= 0x4;  // IA_INDEX_BUFFER
        }
        if (pso::map_vertex_format(dxgi).supported()) {
            support1 |= 0x2;  // IA_VERTEX_BUFFER
        }
        if (support1 == 0) {
            const uint32_t r[3] = {format, 0, 0};
            fill(r, sizeof(r));
            return kEFail;  // D3D12: unsupported format
        }
        const uint32_t r[3] = {format, support1, 0};
        return fill(r, sizeof(r));
    }
    case 4: {  // MULTISAMPLE_QUALITY_LEVELS { Format, SampleCount, Flags, NumQualityLevels }
        uint32_t v[4] = {};
        if (size != 16 || !read_u32(0, &v[0]) || !read_u32(4, &v[1])) {
            return fill(nullptr, 16);
        }
        read_u32(8, &v[2]);
        const auto rt = pso::map_texture_format(static_cast<pso::DxgiFormat>(v[0]), pso::FormatUsage::RenderTarget,
                                                options_.bridge.features);
        const auto ds = pso::map_texture_format(static_cast<pso::DxgiFormat>(v[0]), pso::FormatUsage::DepthStencil,
                                                options_.bridge.features);
        const bool usable = rt.supported() || ds.supported();
        v[3] = (v[1] == 1 || (v[1] == 4 && usable)) ? 1u : 0u;
        return fill(v, sizeof(v));
    }
    case 5: {  // FORMAT_INFO { Format, u8 PlaneCount }
        uint32_t format = 0;
        if (size != 8 || !read_u32(0, &format)) {
            return fill(nullptr, 8);
        }
        const DxgiFormatLayout l = dxgi_format_layout(format);
        if (!l.known && format != 0) {
            return kEInvalidArg;
        }
        uint8_t r[8] = {};
        std::memcpy(r, &format, 4);
        r[4] = static_cast<uint8_t>(format == 0 ? 1 : l.planes);
        return fill(r, sizeof(r));
    }
    case 6: {  // GPU_VIRTUAL_ADDRESS_SUPPORT
        const uint32_t r[2] = {40, 44};
        return fill(r, sizeof(r));
    }
    case 7: {  // SHADER_MODEL { HighestShaderModel } (in: requested)
        uint32_t requested = 0;
        if (size != 4 || !read_u32(0, &requested)) {
            return fill(nullptr, 4);
        }
        const uint32_t highest = std::min<uint32_t>(requested, 0x66u);
        return fill(&highest, 4);
    }
    case 12: {  // ROOT_SIGNATURE { HighestVersion } (in: requested)
        uint32_t requested = 0;
        if (size != 4 || !read_u32(0, &requested)) {
            return fill(nullptr, 4);
        }
        const uint32_t highest = std::min<uint32_t>(requested, 0x3u);
        return fill(&highest, 4);
    }
    case 27: {  // OPTIONS5 { SRVOnlyTiledResourceTier3, RenderPassesTier, RaytracingTier }
        const uint32_t o[3] = {0, 0, 0};  // no ray tracing (d3d12_xs_xbox.h)
        return fill(o, sizeof(o));
    }
    case 30: {  // OPTIONS6 { AdditionalShadingRates, PerPrimitiveVRS..., VariableShadingRateTier, ... }
        const uint32_t o[5] = {0, 0, 0, 0, 0};  // no variable-rate shading: full-rate output
        return fill(o, sizeof(o));
    }
    case 32: {  // OPTIONS7 { MeshShaderTier, SamplerFeedbackTier }
        const uint32_t o[2] = {0, 0};
        return fill(o, sizeof(o));
    }
    default:
        note_bad_argument("CheckFeatureSupport: feature " + std::to_string(feature) + " not answered");
        return kEInvalidArg;
    }
}

HResult Device::CreateDescriptorHeap(const GuestDescriptorHeapDesc* desc, const Guid& riid, OutPtr out) {
    if (desc == nullptr) {
        note_bad_argument("CreateDescriptorHeap: unreadable descriptor");
        return kEInvalidArg;
    }
    if (desc->Type >= d3d::kDescriptorHeapTypeCount || desc->NumDescriptors == 0 ||
        desc->NumDescriptors > d3d::kMaxDescriptorsPerHeap || desc->Flags > 1 ||
        ((desc->Flags & 1) != 0 && desc->Type >= 2)) {
        note_bad_argument("CreateDescriptorHeap: invalid descriptor (type " + std::to_string(desc->Type) +
                          ", count " + std::to_string(desc->NumDescriptors) + ", flags " +
                          std::to_string(desc->Flags) + ")");
        return kEInvalidArg;
    }
    d3d::DescriptorHeapDesc hd;
    hd.type = static_cast<d3d::DescriptorHeapType>(desc->Type);
    hd.num_descriptors = desc->NumDescriptors;
    hd.flags = desc->Flags;
    hd.node_mask = desc->NodeMask;
    // Xbox descriptor: 16 more bytes follow the public ones, {u64 address, u64 size}. A non-zero
    // address places the heap's descriptors in title memory: the title fills `size` with exactly
    // NumDescriptors * GetDescriptorHandleIncrementSize(Type) and expects the heap's CPU start to
    // equal `address`. Anything else in those bytes is ignored (a public 16-byte descriptor).
    uint64_t placement[2] = {0, 0};
    const uint64_t desc_va = com::host_to_guest(desc);
    const uint64_t placed_size =
        static_cast<uint64_t>(desc->NumDescriptors) * d3d::descriptor_handle_increment_size(hd.type);
    if (com::read_guest(desc_va + sizeof(GuestDescriptorHeapDesc), placement, sizeof(placement)) &&
        placement[0] != 0 && placement[1] == placed_size) {
        hd.cpu_address = placement[0];
    }
    // Every other heap gets guest memory of its own: on the console a CPU descriptor handle is a
    // pointer, and the title writes through it (it keeps free lists inside unused descriptors).
    // Descriptor contents written through the API stay in the registry.
    const bool placed = hd.cpu_address != 0;
    uint64_t owned_memory = 0;
    if (!placed) {
        owned_memory = com::guest_alloc(placed_size);
        if (owned_memory == 0) {
            note_bad_argument("CreateDescriptorHeap: no guest memory for " + std::to_string(placed_size) + " bytes");
            return kEOutOfMemory;
        }
        hd.cpu_address = owned_memory;
    }
    d3d::DescriptorHeap* heap = nullptr;
    const d3d::Status status = heaps_.create_heap(hd, &heap);
    if (status != d3d::Status::Ok) {
        if (owned_memory != 0) {
            com::guest_free(owned_memory);
        }
        note_bad_argument(std::string("CreateDescriptorHeap: ") + d3d::status_name(status));
        return status == d3d::Status::TooManyHeaps ? kEOutOfMemory : kEInvalidArg;
    }
    if (placed) {
        count(xbox::Counter::PlacedDescriptorHeaps);
    }
    auto object = com::create_object<DescriptorHeap>(com::implements(kID3D12DescriptorHeap), device_ref(this),
                                                     heap, *desc);
    if (!object) {
        heaps_.destroy_heap(heap);
        if (owned_memory != 0) {
            com::guest_free(owned_memory);
        }
    } else {
        object->set_owned_memory(owned_memory);
    }
    return hand_out(std::move(object), riid, out);
}

uint32_t Device::GetDescriptorHandleIncrementSize(uint32_t type) {
    if (type >= d3d::kDescriptorHeapTypeCount) {
        return 0;
    }
    return d3d::descriptor_handle_increment_size(static_cast<d3d::DescriptorHeapType>(type));
}

HResult Device::CreateRootSignature(uint32_t, uint64_t blob, uint64_t length, const Guid& riid, OutPtr out) {
    if (blob == 0 || length == 0 || length > (64u << 20)) {
        note_bad_argument("CreateRootSignature: invalid blob " + hex(blob) + " length " + std::to_string(length));
        return kEInvalidArg;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    if (!com::read_guest(blob, bytes.data(), bytes.size())) {
        note_bad_argument("CreateRootSignature: unreadable blob " + hex(blob));
        return kEInvalidArg;
    }
    d3d::RootSignatureDesc desc;
    d3d::RootSignatureError err = d3d::parse_root_signature_blob(bytes.data(), bytes.size(), &desc);
    if (!err.ok()) {
        note_bad_argument(std::string("CreateRootSignature: blob parse failed: ") + d3d::status_name(err.status) +
                          " at byte " + std::to_string(err.byte_offset));
        return kEInvalidArg;
    }
    auto signature = std::make_unique<d3d::RootSignature>();
    err = d3d::RootSignature::build(desc, signature.get());
    if (!err.ok()) {
        note_bad_argument(std::string("CreateRootSignature: invalid root signature: ") + d3d::status_name(err.status));
        return kEInvalidArg;
    }
    {
        // The parser accepts versions 1..3 only (1.0, 1.1, 1.2).
        const uint32_t version = static_cast<uint32_t>(desc.version);
        std::lock_guard<std::mutex> lock(shader_mutex_);
        if (++root_signature_versions_[version - 1] == 1) {
            os::log("d3d12_xs: first root signature of version 1." + std::to_string(version - 1));
        }
    }
    auto object = com::create_object<RootSignature>(com::implements(kID3D12RootSignature), device_ref(this),
                                                    std::move(signature));
    return hand_out(std::move(object), riid, out);
}

void Device::CreateConstantBufferView(uint64_t desc_va, uint64_t dest) {
    d3d::ConstantBufferViewDesc cbv;
    const d3d::ConstantBufferViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestConstantBufferViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateConstantBufferView: unreadable descriptor " + hex(desc_va));
            return;
        }
        cbv.buffer_location = g.BufferLocation;
        cbv.size_in_bytes = g.SizeInBytes;
        desc = &cbv;
    }
    const d3d::Status s = heaps_.create_constant_buffer_view(desc, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateConstantBufferView: ") + d3d::status_name(s) + " (address " +
                          hex(desc != nullptr ? desc->buffer_location : 0) + ", size " +
                          std::to_string(desc != nullptr ? desc->size_in_bytes : 0) + ", handle " + hex(dest) + ")");
    }
}

void Device::CreateSampler(uint64_t desc_va, uint64_t dest) {
    GuestSamplerDesc g{};
    if (desc_va == 0 || !com::read_guest(desc_va, g)) {
        note_bad_argument("CreateSampler: unreadable descriptor " + hex(desc_va));
        return;
    }
    d3d::SamplerDesc sd;
    sd.filter = g.Filter;
    sd.address_u = g.AddressU;
    sd.address_v = g.AddressV;
    sd.address_w = g.AddressW;
    sd.mip_lod_bias = g.MipLODBias;
    sd.max_anisotropy = g.MaxAnisotropy;
    sd.comparison_func = g.ComparisonFunc;
    std::memcpy(sd.border_color, g.BorderColor, sizeof(sd.border_color));
    sd.min_lod = g.MinLOD;
    sd.max_lod = g.MaxLOD;
    const d3d::Status s = heaps_.create_sampler(sd, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateSampler: ") + d3d::status_name(s) + " (handle " + hex(dest) + ")");
    }
}

void Device::CopyDescriptors(com::CallContext& ctx) {
    // (NumDestDescriptorRanges, pDestDescriptorRangeStarts, pDestDescriptorRangeSizes,
    //  NumSrcDescriptorRanges, pSrcDescriptorRangeStarts, pSrcDescriptorRangeSizes, DescriptorHeapsType)
    const uint32_t num_dst = static_cast<uint32_t>(ctx.int_arg(1));
    const uint64_t dst_starts = ctx.int_arg(2);
    const uint64_t dst_sizes = ctx.int_arg(3);
    const uint32_t num_src = static_cast<uint32_t>(ctx.int_arg(4));
    const uint64_t src_starts = ctx.int_arg(5);
    const uint64_t src_sizes = ctx.int_arg(6);
    const uint32_t type = static_cast<uint32_t>(ctx.int_arg(7));
    constexpr uint32_t kMaxRanges = 1u << 16;
    std::vector<d3d::CpuDescriptorHandle> ds, ss;
    std::vector<uint32_t> dz, sz;
    if (num_dst > kMaxRanges || num_src > kMaxRanges || type >= d3d::kDescriptorHeapTypeCount ||
        !read_array(dst_starts, num_dst, 8, &ds) || !read_array(src_starts, num_src, 8, &ss) ||
        (dst_sizes != 0 && !read_array(dst_sizes, num_dst, 4, &dz)) ||
        (src_sizes != 0 && !read_array(src_sizes, num_src, 4, &sz))) {
        note_bad_argument("CopyDescriptors: unreadable range arrays");
        return;
    }
    const d3d::Status s = heaps_.copy_descriptors(num_dst, ds.data(), dst_sizes != 0 ? dz.data() : nullptr, num_src,
                                                  ss.data(), src_sizes != 0 ? sz.data() : nullptr,
                                                  static_cast<d3d::DescriptorHeapType>(type));
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CopyDescriptors: ") + d3d::status_name(s));
    }
}

void Device::CopyDescriptorsSimple(uint32_t count, uint64_t dest, uint64_t src, uint32_t type) {
    if (type >= d3d::kDescriptorHeapTypeCount) {
        note_bad_argument("CopyDescriptorsSimple: invalid heap type " + std::to_string(type));
        return;
    }
    const d3d::Status s = heaps_.copy_descriptors_simple(count, d3d::CpuDescriptorHandle{dest},
                                                         d3d::CpuDescriptorHandle{src},
                                                         static_cast<d3d::DescriptorHeapType>(type));
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CopyDescriptorsSimple: ") + d3d::status_name(s));
    }
}

uint64_t resource_size_estimate(const GuestResourceDesc& d) {
    if (d.Dimension == ResourceDimension::Buffer) {
        return d.Width;
    }
    const DxgiFormatLayout l = dxgi_format_layout(d.Format);
    const uint32_t bytes = l.known ? l.block_bytes : 4u;
    const uint32_t mips = d.MipLevels == 0 ? 1u : d.MipLevels;
    const uint32_t layers = d.Dimension == ResourceDimension::Texture3D ? 1u : std::max<uint32_t>(1, d.DepthOrArraySize);
    uint64_t total = 0;
    for (uint32_t m = 0; m < mips; ++m) {
        const uint64_t w = std::max<uint64_t>(1, d.Width >> m);
        const uint64_t h = std::max<uint64_t>(1, d.Height >> m);
        const uint64_t z = d.Dimension == ResourceDimension::Texture3D ? std::max<uint64_t>(1, d.DepthOrArraySize >> m) : 1;
        const uint64_t bw = (w + l.block_width - 1) / std::max<uint32_t>(1, l.block_width);
        const uint64_t bh = (h + l.block_height - 1) / std::max<uint32_t>(1, l.block_height);
        total += bw * bh * z * bytes * l.planes;
    }
    return total * layers * std::max<uint32_t>(1, d.SampleCount);
}

GuestResourceAllocationInfo Device::GetResourceAllocationInfo(uint32_t, uint32_t count, uint64_t descs) {
    GuestResourceAllocationInfo info{0, 0};
    std::vector<GuestResourceDesc> list;
    if (count > 4096 || !read_array(descs, count, sizeof(GuestResourceDesc), &list)) {
        note_bad_argument("GetResourceAllocationInfo: unreadable descriptor array");
        info.SizeInBytes = UINT64_MAX;  // D3D12 reports invalid input with SizeInBytes = UINT64_MAX
        return info;
    }
    for (const GuestResourceDesc& d : list) {
        if (d.Dimension < ResourceDimension::Buffer || d.Dimension > ResourceDimension::Texture3D) {
            note_bad_argument("GetResourceAllocationInfo: invalid dimension " + std::to_string(d.Dimension));
            info.SizeInBytes = UINT64_MAX;
            return info;
        }
        const uint64_t alignment = d.Alignment != 0 ? d.Alignment : (d.SampleCount > 1 ? (4u << 20) : (64u << 10));
        info.SizeInBytes = align_up(info.SizeInBytes, alignment) + align_up(resource_size_estimate(d), alignment);
        info.Alignment = std::max(info.Alignment, alignment);
    }
    return info;
}

HeapPropertiesResult Device::GetCustomHeapProperties(uint32_t, uint32_t heap_type) {
    HeapPropertiesResult r{};
    r.props.Type = HeapType::Custom;
    switch (heap_type) {
    case HeapType::Default:
        r.props.CPUPageProperty = 1;  // NOT_AVAILABLE
        break;
    case HeapType::Upload:
        r.props.CPUPageProperty = 2;  // WRITE_COMBINE
        break;
    case HeapType::Readback:
        r.props.CPUPageProperty = 3;  // WRITE_BACK
        break;
    default:
        note_bad_argument("GetCustomHeapProperties: invalid heap type " + std::to_string(heap_type));
        r.props.CPUPageProperty = 0;
        break;
    }
    r.props.MemoryPoolPreference = 1;  // L0: unified memory (the console is UMA)
    r.props.CreationNodeMask = 1;
    r.props.VisibleNodeMask = 1;
    return r;
}

HResult Device::CreateFence(uint64_t initial_value, uint32_t flags, const Guid& riid, OutPtr out) {
    auto fence = com::create_object<Fence>(com::implements(kID3D12Fence), device_ref(this), initial_value, flags);
    return hand_out(std::move(fence), riid, out);
}

void Device::GetCopyableFootprints(com::CallContext& ctx) {
    // (pResourceDesc, FirstSubresource, NumSubresources, BaseOffset, pLayouts, pNumRows,
    //  pRowSizeInBytes, pTotalBytes)
    const uint64_t desc_va = ctx.int_arg(1);
    const uint32_t first = static_cast<uint32_t>(ctx.int_arg(2));
    const uint32_t num = static_cast<uint32_t>(ctx.int_arg(3));
    const uint64_t base_offset = ctx.int_arg(4);
    const uint64_t layouts = ctx.int_arg(5);
    const uint64_t num_rows = ctx.int_arg(6);
    const uint64_t row_sizes = ctx.int_arg(7);
    const uint64_t total_bytes = ctx.int_arg(8);
    GuestResourceDesc d{};
    const uint64_t invalid = UINT64_MAX;
    if (desc_va == 0 || !com::read_guest(desc_va, d) || d.Dimension < 1 || d.Dimension > 4) {
        note_bad_argument("GetCopyableFootprints: unreadable or invalid resource descriptor " + hex(desc_va));
        if (total_bytes != 0) {
            com::write_guest(total_bytes, invalid);
        }
        return;
    }
    const DxgiFormatLayout l = dxgi_format_layout(d.Format);
    const uint32_t mips = d.Dimension == ResourceDimension::Buffer ? 1u : std::max<uint32_t>(1, d.MipLevels);
    const uint32_t layers =
        (d.Dimension == ResourceDimension::Texture3D || d.Dimension == ResourceDimension::Buffer) ? 1u
                                                                                                   : std::max<uint32_t>(1, d.DepthOrArraySize);
    const uint32_t planes = d.Dimension == ResourceDimension::Buffer ? 1u : (l.known ? l.planes : 1u);
    const uint64_t subresources = uint64_t{mips} * layers * planes;
    if ((d.Dimension != ResourceDimension::Buffer && !l.known) || uint64_t{first} + num > subresources) {
        note_bad_argument("GetCopyableFootprints: unknown format " + std::to_string(d.Format) +
                          " or subresource range out of bounds");
        if (total_bytes != 0) {
            com::write_guest(total_bytes, invalid);
        }
        return;
    }
    uint64_t offset = base_offset;
    uint64_t total = 0;
    for (uint32_t i = 0; i < num; ++i) {
        const uint32_t sub = first + i;
        const uint32_t mip = sub % mips;
        const uint32_t plane = sub / (mips * layers);
        GuestPlacedSubresourceFootprint fp{};
        uint32_t rows = 1;
        uint64_t row_size = 0;
        if (d.Dimension == ResourceDimension::Buffer) {
            fp.Footprint = GuestSubresourceFootprint{0, static_cast<uint32_t>(d.Width), 1, 1,
                                                     static_cast<uint32_t>(align_up(d.Width, 256))};
            row_size = d.Width;
        } else {
            const uint32_t w = std::max<uint32_t>(1, static_cast<uint32_t>(d.Width >> mip));
            const uint32_t h = std::max<uint32_t>(1, d.Height >> mip);
            const uint32_t z = d.Dimension == ResourceDimension::Texture3D ? std::max<uint32_t>(1, d.DepthOrArraySize >> mip) : 1u;
            const uint32_t bw = (w + l.block_width - 1) / l.block_width;
            rows = (h + l.block_height - 1) / l.block_height;
            row_size = uint64_t{bw} * plane_block_bytes(d.Format, plane);
            // Block-compressed footprints keep whole blocks (width/height rounded up to the block).
            fp.Footprint = GuestSubresourceFootprint{d.Format, bw * l.block_width, rows * l.block_height, z,
                                                     static_cast<uint32_t>(align_up(row_size, 256))};
        }
        offset = align_up(offset, 512);
        fp.Offset = offset;
        const uint64_t size = uint64_t{fp.Footprint.RowPitch} * rows * fp.Footprint.Depth;
        if (layouts != 0) {
            com::write_guest(layouts + sizeof(GuestPlacedSubresourceFootprint) * i, fp);
        }
        if (num_rows != 0) {
            com::write_guest(num_rows + 4ull * i, rows);
        }
        if (row_sizes != 0) {
            com::write_guest(row_sizes + 8ull * i, row_size);
        }
        total = (offset - base_offset) + size - (fp.Footprint.RowPitch - row_size);
        offset += size;
    }
    if (total_bytes != 0) {
        com::write_guest(total_bytes, total);
    }
}

HResult Device::CreateQueryHeap(const GuestQueryHeapDesc* desc, const Guid& riid, OutPtr out) {
    if (desc == nullptr || desc->Type > 7 || desc->Count == 0) {
        note_bad_argument("CreateQueryHeap: invalid descriptor");
        return kEInvalidArg;
    }
    auto heap = com::create_object<QueryHeap>(com::implements(kID3D12QueryHeap), device_ref(this), *desc);
    return hand_out(std::move(heap), riid, out);
}

HResult Device::CreateCommandSignature(const GuestCommandSignatureDesc* desc, uint64_t, const Guid& riid,
                                       OutPtr out) {
    if (desc == nullptr || desc->ByteStride == 0 || desc->NumArgumentDescs == 0 || desc->pArgumentDescs == 0) {
        note_bad_argument("CreateCommandSignature: invalid descriptor");
        return kEInvalidArg;
    }
    auto signature = com::create_object<CommandSignature>(com::implements(kID3D12CommandSignature), device_ref(this),
                                                          desc->ByteStride, desc->NumArgumentDescs);
    return hand_out(std::move(signature), riid, out);
}

// ---------------------------------------------------------------------------------------------
// Blob
// ---------------------------------------------------------------------------------------------

Blob::Blob(uint64_t size) : size_(size) {
    if (size != 0) {
        buffer_ = com::guest_alloc(size);
    }
}

Blob::~Blob() {
    if (buffer_ != 0) {
        com::guest_free(buffer_);
    }
}

com::Ref<Blob> Blob::create(const void* bytes, size_t size) {
    auto blob = com::create_object<Blob>(com::implements(kID3DBlob), static_cast<uint64_t>(size));
    if (!blob || !blob->valid()) {
        return {};
    }
    if (size != 0 && !com::write_guest(blob->buffer_, bytes, size)) {
        return {};
    }
    return blob;
}

// ---------------------------------------------------------------------------------------------
// Allocator, fence, descriptor heap, root signature, queue
// ---------------------------------------------------------------------------------------------

CommandAllocator::CommandAllocator(com::Ref<Device> device, uint32_t type)
    : ChildBase(std::move(device)), type_(type) {
    device_->allocators.fetch_add(1);
}

CommandAllocator::~CommandAllocator() {
    device_->allocators.fetch_sub(1);
}

HResult CommandAllocator::Reset() {
    resets_.fetch_add(1, std::memory_order_relaxed);
    return kSOk;
}

Fence::Fence(com::Ref<Device> device, uint64_t initial_value, uint32_t flags)
    : ChildBase(std::move(device)), value_(initial_value), flags_(flags) {
    device_->fences.fetch_add(1);
}

Fence::~Fence() {
    if (value_word_ != 0) {
        com::guest_free(value_word_);
    }
    device_->fences.fetch_sub(1);
}

uint64_t Fence::CompletedValueAddress() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (value_word_ == 0) {
        value_word_ = com::guest_alloc(sizeof(uint64_t));
        if (value_word_ == 0) {
            device_->note_bad_argument("ID3D12Fence Xbox slot 12: out of guest memory for the completed value");
            return 0;
        }
        publish_value_locked();
    }
    return value_word_;
}

// Guest code reads the word while other threads signal: one aligned 64-bit store.
void Fence::publish_value_locked() {
    if (value_word_ == 0) {
        return;
    }
    if (auto* word = static_cast<uint64_t*>(com::guest_to_host(value_word_, sizeof(uint64_t)))) {
        std::atomic_ref<uint64_t>(*word).store(value_.load(std::memory_order_acquire), std::memory_order_release);
    }
}

namespace {
std::mutex& fence_wait_mutex() {
    static std::mutex m;
    return m;
}
std::condition_variable& fence_wait_cv() {
    static std::condition_variable cv;
    return cv;
}
}  // namespace

HResult Fence::SetEventOnCompletion(uint64_t value, uint64_t event_handle) {
    if (event_handle == 0) {
        // D3D12: a null event blocks until the fence reaches `value`.
        std::unique_lock<std::mutex> lock(fence_wait_mutex());
        fence_wait_cv().wait(lock, [&] { return value_.load(std::memory_order_acquire) >= value; });
        return kSOk;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (value_.load(std::memory_order_acquire) < value) {
            waits_.emplace(value, event_handle);
            return kSOk;
        }
    }
    pg6_win32_set_event(event_handle);
    return kSOk;
}

HResult Fence::Signal(uint64_t value) {
    std::vector<uint64_t> to_set;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        value_.store(value, std::memory_order_release);
        publish_value_locked();
        fire_events_locked(&to_set);
    }
    for (uint64_t handle : to_set) {
        pg6_win32_set_event(handle);
    }
    {
        std::lock_guard<std::mutex> lock(fence_wait_mutex());
    }
    fence_wait_cv().notify_all();
    return kSOk;
}

void Fence::fire_events_locked(std::vector<uint64_t>* to_set) {
    const uint64_t v = value_.load(std::memory_order_acquire);
    auto end = waits_.upper_bound(v);
    for (auto it = waits_.begin(); it != end; ++it) {
        to_set->push_back(it->second);
    }
    waits_.erase(waits_.begin(), end);
}

DescriptorHeap::DescriptorHeap(com::Ref<Device> device, d3d::DescriptorHeap* heap, const GuestDescriptorHeapDesc& desc)
    : ChildBase(std::move(device)), heap_(heap), desc_(desc) {
    device_->descriptor_heaps.fetch_add(1);
}

DescriptorHeap::~DescriptorHeap() {
    device_->heaps().destroy_heap(heap_);
    if (owned_memory_ != 0) {
        com::guest_free(owned_memory_);
    }
    device_->descriptor_heaps.fetch_sub(1);
}

RootSignature::RootSignature(com::Ref<Device> device, std::unique_ptr<d3d::RootSignature> signature)
    : ChildBase(std::move(device)), signature_(std::move(signature)) {
    device_->root_signatures.fetch_add(1);
}

RootSignature::~RootSignature() {
    device_->root_signatures.fetch_sub(1);
}

CommandQueue::CommandQueue(com::Ref<Device> device, const GuestCommandQueueDesc& desc)
    : ChildBase(std::move(device)), desc_(desc) {
    device_->queues.fetch_add(1);
}

CommandQueue::~CommandQueue() {
    device_->queues.fetch_sub(1);
}

HResult CommandQueue::Signal(InterfaceArg<IFenceHost> fence, uint64_t value) {
    if (!fence) {
        device_->note_bad_argument("ID3D12CommandQueue::Signal: invalid fence " + hex(fence.va()));
        return kEInvalidArg;
    }
    // Work is handed to the GPU process at ExecuteCommandLists time and executes in submission
    // order; the runtime has no GPU completion feedback, so a queue signal completes when it is
    // issued (everything submitted before it has been handed off).
    static_cast<Fence*>(fence.get())->signal_from_queue(value);
    return kSOk;
}

HResult CommandQueue::Wait(InterfaceArg<IFenceHost> fence, uint64_t value) {
    if (!fence) {
        device_->note_bad_argument("ID3D12CommandQueue::Wait: invalid fence " + hex(fence.va()));
        return kEInvalidArg;
    }
    // Submissions are executed in order on one GPU timeline; a GPU wait for a value the CPU has not
    // signalled yet cannot be expressed in the packet stream and is only recorded.
    if (fence->GetCompletedValue() < value) {
        device_->note_bad_argument("ID3D12CommandQueue::Wait: fence value " + std::to_string(value) +
                                   " not reached at submission (not enforced)");
    }
    return kSOk;
}

HResult CommandQueue::GetTimestampFrequency(GuestPtr<uint64_t> frequency) {
    if (frequency.is_null() || !frequency.write(device_->timestamp_frequency())) {
        return kEInvalidArg;
    }
    return kSOk;
}

// ---------------------------------------------------------------------------------------------
// Root signature reader
// ---------------------------------------------------------------------------------------------

bool read_versioned_root_signature(uint64_t desc_va, d3d::RootSignatureDesc* out, std::string* error) {
    GuestVersionedRootSignatureDesc desc{};
    if (desc_va == 0 || !com::read_guest(desc_va, desc)) {
        *error = "unreadable D3D12_VERSIONED_ROOT_SIGNATURE_DESC at " + hex(desc_va);
        return false;
    }
    return read_versioned_root_signature(desc, out, error);
}

bool read_versioned_root_signature(const GuestVersionedRootSignatureDesc& desc, d3d::RootSignatureDesc* out,
                                   std::string* error) {
    if (desc.Version < 1 || desc.Version > 3) {
        *error = "unsupported root signature version " + std::to_string(desc.Version);
        return false;
    }
    if (desc.NumParameters > d3d::kMaxRootParameters || desc.NumStaticSamplers > d3d::kMaxStaticSamplers) {
        *error = "too many parameters / static samplers";
        return false;
    }
    const bool v10 = desc.Version == 1;
    d3d::RootSignatureDesc r;
    r.version = static_cast<d3d::RootSignatureVersion>(desc.Version);
    r.flags = desc.Flags;
    std::vector<GuestRootParameter> params;
    if (!read_array(desc.pParameters, desc.NumParameters, sizeof(GuestRootParameter), &params)) {
        *error = "unreadable root parameter array at " + hex(desc.pParameters);
        return false;
    }
    for (uint32_t i = 0; i < params.size(); ++i) {
        const GuestRootParameter& p = params[i];
        if (p.ParameterType > 4 || p.ShaderVisibility > 7) {
            *error = "root parameter " + std::to_string(i) + ": invalid type " + std::to_string(p.ParameterType) +
                     " / visibility " + std::to_string(p.ShaderVisibility);
            return false;
        }
        d3d::RootParameterDesc rp;
        rp.parameter_type = static_cast<d3d::RootParameterType>(p.ParameterType);
        rp.shader_visibility = static_cast<d3d::ShaderVisibility>(p.ShaderVisibility);
        const uint32_t low_q = static_cast<uint32_t>(p.q);
        switch (rp.parameter_type) {
        case d3d::RootParameterType::DescriptorTable: {
            const uint32_t n = p.w0;
            if (n > kMaxRangesPerTable) {
                *error = "root parameter " + std::to_string(i) + ": " + std::to_string(n) + " ranges";
                return false;
            }
            if (v10) {
                std::vector<GuestDescriptorRange> ranges;
                if (!read_array(p.q, n, sizeof(GuestDescriptorRange), &ranges)) {
                    *error = "root parameter " + std::to_string(i) + ": unreadable ranges at " + hex(p.q);
                    return false;
                }
                for (const auto& g : ranges) {
                    d3d::DescriptorRangeDesc d;
                    d.range_type = static_cast<d3d::DescriptorRangeType>(g.RangeType);
                    d.num_descriptors = g.NumDescriptors;
                    d.base_shader_register = g.BaseShaderRegister;
                    d.register_space = g.RegisterSpace;
                    d.offset_in_descriptors_from_table_start = g.OffsetInDescriptorsFromTableStart;
                    rp.ranges.push_back(d);
                }
            } else {
                std::vector<GuestDescriptorRange1> ranges;
                if (!read_array(p.q, n, sizeof(GuestDescriptorRange1), &ranges)) {
                    *error = "root parameter " + std::to_string(i) + ": unreadable ranges at " + hex(p.q);
                    return false;
                }
                for (const auto& g : ranges) {
                    d3d::DescriptorRangeDesc d;
                    d.range_type = static_cast<d3d::DescriptorRangeType>(g.RangeType);
                    d.num_descriptors = g.NumDescriptors;
                    d.base_shader_register = g.BaseShaderRegister;
                    d.register_space = g.RegisterSpace;
                    d.flags = g.Flags;
                    d.offset_in_descriptors_from_table_start = g.OffsetInDescriptorsFromTableStart;
                    rp.ranges.push_back(d);
                }
            }
            break;
        }
        case d3d::RootParameterType::Constants32Bit:
            rp.shader_register = p.w0;
            rp.register_space = p.w1;
            rp.num_32bit_values = low_q;
            break;
        default:  // root CBV / SRV / UAV
            rp.shader_register = p.w0;
            rp.register_space = p.w1;
            rp.flags = v10 ? 0u : low_q;
            break;
        }
        r.parameters.push_back(std::move(rp));
    }
    const size_t sampler_stride = desc.Version == 3 ? kGuestStaticSamplerDesc1Size : sizeof(GuestStaticSamplerDesc);
    for (uint32_t i = 0; i < desc.NumStaticSamplers; ++i) {
        GuestStaticSamplerDesc g{};
        uint32_t flags = 0;
        const uint64_t va = desc.pStaticSamplers + sampler_stride * i;
        if (desc.pStaticSamplers == 0 || !com::read_guest(va, g) ||
            (desc.Version == 3 && !com::read_guest(va + sizeof(GuestStaticSamplerDesc), &flags, 4))) {
            *error = "unreadable static sampler " + std::to_string(i);
            return false;
        }
        d3d::StaticSamplerDesc s;
        s.filter = g.Filter;
        s.address_u = g.AddressU;
        s.address_v = g.AddressV;
        s.address_w = g.AddressW;
        s.mip_lod_bias = g.MipLODBias;
        s.max_anisotropy = g.MaxAnisotropy;
        s.comparison_func = g.ComparisonFunc;
        s.border_color = g.BorderColor;
        s.min_lod = g.MinLOD;
        s.max_lod = g.MaxLOD;
        s.shader_register = g.ShaderRegister;
        s.register_space = g.RegisterSpace;
        if (g.ShaderVisibility > 7) {
            *error = "static sampler " + std::to_string(i) + ": invalid visibility";
            return false;
        }
        s.shader_visibility = static_cast<d3d::ShaderVisibility>(g.ShaderVisibility);
        s.flags = flags;
        r.static_samplers.push_back(s);
    }
    *out = std::move(r);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Vtables
// ---------------------------------------------------------------------------------------------

namespace {

using DevM = Methods<IDeviceHost>;
constexpr com::MethodDesc kDeviceMethods[] = {
    DevM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),                 // 3
    DevM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),                 // 4
    DevM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),  // 5
    DevM::method<&IObjectHost::SetName>("SetName"),                               // 6
    DevM::method<&IDeviceHost::GetNodeCount>("GetNodeCount"),                     // 7
    DevM::method<&IDeviceHost::CreateCommandQueue>("CreateCommandQueue"),         // 8
    DevM::method<&IDeviceHost::CreateCommandAllocator>("CreateCommandAllocator"), // 9
    DevM::method<&IDeviceHost::CreateGraphicsPipelineState>("CreateGraphicsPipelineState"),  // 10
    DevM::method<&IDeviceHost::CreateComputePipelineState>("CreateComputePipelineState"),    // 11
    DevM::method<&IDeviceHost::CreateCommandList>("CreateCommandList"),                      // 12
    DevM::method<&IDeviceHost::CheckFeatureSupport>("CheckFeatureSupport"),       // 13
    DevM::method<&IDeviceHost::CreateDescriptorHeap>("CreateDescriptorHeap"),     // 14
    DevM::method<&IDeviceHost::GetDescriptorHandleIncrementSize>("GetDescriptorHandleIncrementSize"),  // 15
    DevM::method<&IDeviceHost::CreateRootSignature>("CreateRootSignature"),       // 16
    DevM::method<&IDeviceHost::CreateConstantBufferView>("CreateConstantBufferView"),  // 17
    DevM::method<&IDeviceHost::CreateShaderResourceView>("CreateShaderResourceView"),    // 18
    DevM::method<&IDeviceHost::CreateUnorderedAccessView>("CreateUnorderedAccessView"),  // 19
    DevM::method<&IDeviceHost::CreateRenderTargetView>("CreateRenderTargetView"),        // 20
    DevM::method<&IDeviceHost::CreateDepthStencilView>("CreateDepthStencilView"),        // 21
    DevM::method<&IDeviceHost::CreateSampler>("CreateSampler"),                   // 22
    DevM::method<&IDeviceHost::CopyDescriptors>("CopyDescriptors"),               // 23
    DevM::method<&IDeviceHost::CopyDescriptorsSimple>("CopyDescriptorsSimple"),   // 24
    DevM::sret<&IDeviceHost::GetResourceAllocationInfo>("GetResourceAllocationInfo"),  // 25
    DevM::sret<&IDeviceHost::GetCustomHeapProperties>("GetCustomHeapProperties"),      // 26
    DevM::method<&IDeviceHost::CreateCommittedResource>("CreateCommittedResource"),  // 27
    DevM::method<&IDeviceHost::CreateHeap>("CreateHeap"),                            // 28
    DevM::method<&IDeviceHost::CreatePlacedResource>("CreatePlacedResource"),        // 29
    DevM::method<&IDeviceHost::CreateReservedResource>("CreateReservedResource"),    // 30
    DevM::stub("CreateSharedHandle"),                                             // 31
    DevM::stub("OpenSharedHandle"),                                               // 32
    DevM::stub("OpenSharedHandleByName"),                                         // 33
    DevM::method<&IDeviceHost::MakeResident>("MakeResident"),                     // 34
    DevM::method<&IDeviceHost::Evict>("Evict"),                                   // 35
    DevM::method<&IDeviceHost::CreateFence>("CreateFence"),                       // 36
    DevM::method<&IDeviceHost::GetDeviceRemovedReason>("GetDeviceRemovedReason"), // 37
    DevM::method<&IDeviceHost::GetCopyableFootprints>("GetCopyableFootprints"),   // 38
    DevM::method<&IDeviceHost::CreateQueryHeap>("CreateQueryHeap"),               // 39
    DevM::method<&IDeviceHost::SetStablePowerState>("SetStablePowerState"),       // 40
    DevM::method<&IDeviceHost::CreateCommandSignature>("CreateCommandSignature"), // 41
    DevM::stub("GetResourceTiling"),                                              // 42
    DevM::stub("GetAdapterLuid"),                                                 // 43
    DevM::stub(nullptr),                                                          // 44
    DevM::stub(nullptr),                                                          // 45
    DevM::stub(nullptr),                                                          // 46
    DevM::stub(nullptr),                                                          // 47
    DevM::method<&IDeviceHost::CreateCommandQueueX>("CreateCommandQueue (Xbox slot 48)"),  // 48
    DevM::method<&IDeviceHost::Slot49_SetFeatureToggle_unverified>("Xbox slot 49 (feature toggle)"),  // 49
    DevM::method<&IDeviceHost::Slot50_RegisterRemovalCallbacks_unverified>("Xbox slot 50"),  // 50
    DevM::stub(nullptr),  // 51
    DevM::stub(nullptr),  // 52
    DevM::stub(nullptr),  // 53
    DevM::stub(nullptr),  // 54
    DevM::method<&IDeviceHost::CreatePipelineStateWithOverrides>("CreatePipelineStateWithOverrides (Xbox slot 55)"),  // 55
    DevM::stub(nullptr),  // 56
    DevM::method<&IDeviceHost::CreateNativePipelineState>("CreatePipelineState native (Xbox slot 57)"),  // 57
    DevM::stub(nullptr),  // 58
    DevM::stub(nullptr),  // 59
    DevM::stub(nullptr),  // 60
    DevM::method<&IDeviceHost::Slot61_unverified>("Xbox slot 61"),  // 61
    DevM::method<&IDeviceHost::CreateComputePipelineStateX>("CreateComputePipelineState (Xbox slot 62)"),  // 62
    DevM::stub(nullptr),  // 63
    DevM::stub(nullptr),  // 64
    DevM::stub(nullptr),  // 65
    DevM::stub(nullptr),  // 66
    DevM::method<&IDeviceHost::CreatePipelineStateFromBlob_unverified>("native compute PSO (Xbox slot 67)"),  // 67
    DevM::stub(nullptr),  // 68
    DevM::stub(nullptr),  // 69
    DevM::method<&IDeviceHost::CreateShaderResourceViewAtAddress>("SRV at address (Xbox slot 70)"),  // 70
    DevM::method<&IDeviceHost::CreateUnorderedAccessViewAtAddress>("UAV at address (Xbox slot 71)"),  // 71
    DevM::stub(nullptr),  // 72
    DevM::method<&IDeviceHost::CreateCommandListX>("CreateCommandList (Xbox slot 73)"),  // 73
    DevM::method<&IDeviceHost::RegisterPagePool_unverified>("page pool (Xbox slot 74)"),  // 74
    DevM::stub(nullptr),  // 75
    DevM::stub(nullptr),  // 76
    DevM::stub(nullptr),  // 77
    DevM::method<&IDeviceHost::SetFrameIntervalX>("SetFrameIntervalX (Xbox slot 78)"),  // 78
    DevM::method<&IDeviceHost::ScheduleFrameEventX>("ScheduleFrameEventX (Xbox slot 79)"),  // 79
    DevM::method<&IDeviceHost::WaitFrameEventX>("WaitFrameEventX (Xbox slot 80)"),  // 80
    DevM::method<&IDeviceHost::QueryFrameStatistics_unverified>("frame statistics (Xbox slot 81)"),  // 81
    DevM::stub(nullptr),  // 82
    DevM::stub(nullptr),  // 83
    DevM::stub(nullptr),  // 84
    DevM::stub(nullptr),  // 85
    DevM::method<&IDeviceHost::CreatePipelineStateFromStream>("CreatePipelineState stream (Xbox slot 86)"),  // 86
    DevM::stub(nullptr),  // 87
    DevM::stub(nullptr),  // 88
    DevM::stub(nullptr),  // 89
    DevM::stub(nullptr),  // 90
    DevM::stub(nullptr),  // 91
    DevM::stub(nullptr),  // 92
    DevM::stub(nullptr),  // 93
    DevM::stub(nullptr),  // 94
    DevM::stub(nullptr),  // 95
    DevM::stub(nullptr),  // 96
    DevM::stub(nullptr),  // 97
    DevM::stub(nullptr),  // 98
    DevM::stub(nullptr),  // 99
    DevM::stub(nullptr),  // 100
    DevM::method<&IDeviceHost::CreateStateObject>("CreateStateObject (Xbox slot 101)"),  // 101
    DevM::stub(nullptr),  // 102
    DevM::stub(nullptr),  // 103
    DevM::stub(nullptr),  // 104
    DevM::stub(nullptr),  // 105
    DevM::stub(nullptr),  // 106
    DevM::stub(nullptr),  // 107
    DevM::sret<&IDeviceHost::GetResourceAllocationInfo1>("GetResourceAllocationInfo1 (Xbox slot 108)"),  // 108
    DevM::stub(nullptr),  // 109
    DevM::stub(nullptr),  // 110
    DevM::stub(nullptr),  // 111
    DevM::stub(nullptr),  // 112
    DevM::stub(nullptr),  // 113
    DevM::stub(nullptr),  // 114
    DevM::stub(nullptr),  // 115
    DevM::stub(nullptr),  // 116
    DevM::stub(nullptr),  // 117
    DevM::stub(nullptr),  // 118
    DevM::method<&IDeviceHost::Slot119_unverified>("Xbox slot 119"),  // 119
    DevM::stub(nullptr),  // 120
    DevM::stub(nullptr),  // 121
    DevM::stub(nullptr),  // 122
    DevM::stub(nullptr),  // 123
    DevM::stub(nullptr),  // 124
    DevM::stub(nullptr),  // 125
    DevM::stub(nullptr),  // 126
    DevM::method<&IDeviceHost::CreateCommittedResource3>("CreateCommittedResource3 (Xbox slot 127)"),  // 127
    DevM::stub(nullptr),  // 128
    DevM::stub(nullptr),  // 129
    DevM::stub(nullptr),  // 130
    DevM::stub(nullptr),  // 131
    DevM::method<&IDeviceHost::Slot132_unverified>("Xbox slot 132"),  // 132
    DevM::method<&IDeviceHost::CreatePlacedResourceX>("CreatePlacedResource (Xbox slot 133)"),  // 133
};
constexpr Guid kDeviceIids[] = {
    IID_ID3D12Object,
    IID_ID3D12Device,
    com::make_guid("77acce80-638e-4e65-8895-c1f23386863e"),  // ID3D12Device1
    com::make_guid("30baa41e-b15b-475c-a0bb-1af5c5b64328"),  // ID3D12Device2
    com::make_guid("81dadc15-2bad-4392-93c5-101345c4aa98"),  // ID3D12Device3
    com::make_guid("e865df17-a9ee-46f9-a463-3098315aa2e5"),  // ID3D12Device4
    com::make_guid("8b4f173b-2fea-4b80-8f58-4307191ab95d"),  // ID3D12Device5
    com::make_guid("c70b221b-40e4-4a17-89af-025a0727a6dc"),  // ID3D12Device6
    com::make_guid("5c014b53-68a1-4b9b-8bd1-dd6046b9358b"),  // ID3D12Device7
    com::make_guid("9218e6bb-f944-4f7e-a75c-b1b2c7b701f3"),  // ID3D12Device8
    com::make_guid("4c80e962-f032-4f60-bc9e-ebc2cfa1d83c"),  // ID3D12Device9
    com::make_guid("517f8718-aa66-49f9-b02b-a7ab89c06031"),  // ID3D12Device10
};

using QueueM = Methods<ICommandQueueHost>;
constexpr com::MethodDesc kQueueMethods[] = {
    QueueM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),                 // 3
    QueueM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),                 // 4
    QueueM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),  // 5
    QueueM::method<&IObjectHost::SetName>("SetName"),                               // 6
    QueueM::method<&IDeviceChildHost::GetDevice>("GetDevice"),                      // 7
    QueueM::stub(nullptr), QueueM::stub(nullptr), QueueM::stub(nullptr), QueueM::stub(nullptr),  // 8..11
    QueueM::stub(nullptr), QueueM::stub(nullptr), QueueM::stub(nullptr), QueueM::stub(nullptr),  // 12..15
    QueueM::method<&ICommandQueueHost::Signal>("Signal"),                           // 16
    QueueM::method<&ICommandQueueHost::Wait>("Wait"),                               // 17
    QueueM::method<&ICommandQueueHost::GetTimestampFrequency>("GetTimestampFrequency"),  // 18
    QueueM::stub(nullptr), QueueM::stub(nullptr), QueueM::stub(nullptr), QueueM::stub(nullptr),  // 19..22
    QueueM::method<&ICommandQueueHost::SuspendX>("SuspendX (Xbox slot 23)"),        // 23
    QueueM::method<&ICommandQueueHost::ResumeX>("ResumeX (Xbox slot 24)"),          // 24
    QueueM::stub(nullptr), QueueM::stub(nullptr),                                   // 25..26
    QueueM::stub(nullptr), QueueM::stub(nullptr), QueueM::stub(nullptr), QueueM::stub(nullptr),  // 27..30
    QueueM::method<&ICommandQueueHost::Slot31_UpdatePageMappings_unverified>("page mappings (Xbox slot 31)"),  // 31
    QueueM::method<&ICommandQueueHost::PresentX>("PresentX (Xbox slot 32)"),        // 32
    QueueM::stub(nullptr), QueueM::stub(nullptr),                                   // 33..34
    QueueM::method<&ICommandQueueHost::ExecuteCommandListsX>("ExecuteCommandLists (Xbox slot 35)"),  // 35
};
constexpr Guid kQueueIids[] = {IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable};

using AllocM = Methods<ICommandAllocatorHost>;
constexpr com::MethodDesc kAllocatorMethods[] = {
    AllocM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    AllocM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    AllocM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    AllocM::method<&IObjectHost::SetName>("SetName"),
    AllocM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    AllocM::method<&ICommandAllocatorHost::Reset>("Reset"),  // 8
    AllocM::method<&ICommandAllocatorHost::SetAllocationCallback_unverified>("allocation callback (Xbox slot 9)"),  // 9
};
constexpr Guid kPageableIids[] = {IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable};

using FenceM = Methods<IFenceHost>;
constexpr com::MethodDesc kFenceMethods[] = {
    FenceM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    FenceM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    FenceM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    FenceM::method<&IObjectHost::SetName>("SetName"),
    FenceM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    FenceM::method<&IFenceHost::GetCompletedValue>("GetCompletedValue"),        // 8
    FenceM::method<&IFenceHost::SetEventOnCompletion>("SetEventOnCompletion"),  // 9
    FenceM::method<&IFenceHost::Signal>("Signal"),                              // 10
    FenceM::method<&IFenceHost::GetCreationFlags>("GetCreationFlags"),          // 11 (ID3D12Fence1)
    FenceM::method<&IFenceHost::CompletedValueAddress>("completed-value address (Xbox slot 12)"),  // 12
};
constexpr Guid kFenceIids[] = {IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable, IID_ID3D12Fence1};

using HeapM = Methods<IDescriptorHeapHost>;
constexpr com::MethodDesc kDescriptorHeapMethods[] = {
    HeapM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    HeapM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    HeapM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    HeapM::method<&IObjectHost::SetName>("SetName"),
    HeapM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    HeapM::sret<&IDescriptorHeapHost::GetDesc>("GetDesc"),                                                 // 8
    HeapM::sret<&IDescriptorHeapHost::GetCPUDescriptorHandleForHeapStart>("GetCPUDescriptorHandleForHeapStart"),  // 9
    HeapM::sret<&IDescriptorHeapHost::GetGPUDescriptorHandleForHeapStart>("GetGPUDescriptorHandleForHeapStart"),  // 10
};

template <typename Host>
struct ChildOnly {
    using M = Methods<Host>;
    static constexpr com::MethodDesc kMethods[] = {
        M::template method<&IObjectHost::GetPrivateData>("GetPrivateData"),
        M::template method<&IObjectHost::SetPrivateData>("SetPrivateData"),
        M::template method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
        M::template method<&IObjectHost::SetName>("SetName"),
        M::template method<&IDeviceChildHost::GetDevice>("GetDevice"),
    };
};

using BlobM = com::Methods<IBlobHost>;
constexpr com::MethodDesc kBlobMethods[] = {
    BlobM::method<&IBlobHost::GetBufferPointer>("GetBufferPointer"),  // 3
    BlobM::method<&IBlobHost::GetBufferSize>("GetBufferSize"),        // 4
};

constexpr com::InterfaceOptions options(uint32_t slots, std::span<const Guid> extra) {
    com::InterfaceOptions o;
    o.slot_count = slots;
    o.policy = com::SlotPolicy::ReturnNotImpl;
    o.extra_iids = extra;
    return o;
}

}  // namespace

const com::Interface<IDeviceHost> kID3D12Device = com::declare_interface<IDeviceHost>(
    "ID3D12Device", IID_ID3D12Device11, kDeviceMethods, options(160, kDeviceIids));
const com::Interface<ICommandQueueHost> kID3D12CommandQueue = com::declare_interface<ICommandQueueHost>(
    "ID3D12CommandQueue", IID_ID3D12CommandQueue, kQueueMethods, options(64, kQueueIids));
const com::Interface<ICommandAllocatorHost> kID3D12CommandAllocator = com::declare_interface<ICommandAllocatorHost>(
    "ID3D12CommandAllocator", IID_ID3D12CommandAllocator, kAllocatorMethods, options(16, kPageableIids));
const com::Interface<IFenceHost> kID3D12Fence =
    com::declare_interface<IFenceHost>("ID3D12Fence", IID_ID3D12Fence, kFenceMethods, options(24, kFenceIids));
const com::Interface<IDescriptorHeapHost> kID3D12DescriptorHeap = com::declare_interface<IDescriptorHeapHost>(
    "ID3D12DescriptorHeap", IID_ID3D12DescriptorHeap, kDescriptorHeapMethods, options(16, kPageableIids));
const com::Interface<IRootSignatureHost> kID3D12RootSignature = com::declare_interface<IRootSignatureHost>(
    "ID3D12RootSignature", IID_ID3D12RootSignature, ChildOnly<IRootSignatureHost>::kMethods,
    options(16, kQueueIids));
const com::Interface<IQueryHeapHost> kID3D12QueryHeap = com::declare_interface<IQueryHeapHost>(
    "ID3D12QueryHeap", IID_ID3D12QueryHeap, ChildOnly<IQueryHeapHost>::kMethods, options(16, kPageableIids));
const com::Interface<ICommandSignatureHost> kID3D12CommandSignature = com::declare_interface<ICommandSignatureHost>(
    "ID3D12CommandSignature", IID_ID3D12CommandSignature, ChildOnly<ICommandSignatureHost>::kMethods,
    options(16, kPageableIids));
const com::Interface<IBlobHost> kID3DBlob =
    com::declare_interface<IBlobHost>("ID3DBlob", IID_ID3DBlob, kBlobMethods, options(8, {}));

}  // namespace playgta6::gpu::d3d12_guest
