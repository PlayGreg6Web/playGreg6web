#include "gpu/d3d12_guest_resources.h"
#include "gpu/gpu_va_watch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#include "gpu/gfx10_detile.h"
#include "os/guest_window.h"
#include "os/os_shim.h"
#include "os/vmem.h"
#include "os/surface_blocks.h"
#include "os/xbox_scarlett.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

// A texture known only by a view at an address (Device::ensure_address_texture): the first 64 KiB
// of its memory, read through the CPU window.
struct ViewProbe {
    uint64_t cpu = 0;
    uint64_t len = 0;
    bool seen_data = false;
};

struct TextureProbe {
    std::mutex mutex;
    std::vector<Resource*> textures;
    size_t cursor = 0;
    uint64_t with_data = 0;
    std::vector<ViewProbe> views;
    size_t view_cursor = 0;
    uint64_t views_with_data = 0;
};

TextureProbe& texture_probe() {
    static TextureProbe p;
    return p;
}

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

// Formats WebGPU allows STORAGE_BINDING for: the core set, plus the "texture-formats-tier1" set
// and BGRA8 with "bgra8unorm-storage". Both texture creation paths (committed/placed resources and
// address-only textures) gate STORAGE_BINDING through this one predicate.
bool storage_capable(pso::WgpuTextureFormat f, uint32_t features) {
    using F = pso::WgpuTextureFormat;
    switch (f) {
        case F::Rgba8Unorm: case F::Rgba8Snorm: case F::Rgba8Uint: case F::Rgba8Sint:
        case F::Rgba16Uint: case F::Rgba16Sint: case F::Rgba16Float:
        case F::R32Uint: case F::R32Sint: case F::R32Float:
        case F::Rg32Uint: case F::Rg32Sint: case F::Rg32Float:
        case F::Rgba32Uint: case F::Rgba32Sint: case F::Rgba32Float:
            return true;
        case F::R8Unorm: case F::R8Snorm: case F::R8Uint: case F::R8Sint:
        case F::Rg8Unorm: case F::Rg8Snorm: case F::Rg8Uint: case F::Rg8Sint:
        case F::R16Unorm: case F::R16Snorm: case F::R16Uint: case F::R16Sint: case F::R16Float:
        case F::Rg16Unorm: case F::Rg16Snorm: case F::Rg16Uint: case F::Rg16Sint: case F::Rg16Float:
        case F::Rgba16Unorm: case F::Rgba16Snorm:
        case F::Rgb10a2Uint: case F::Rgb10a2Unorm: case F::Rg11b10Ufloat:
            return (features & pso::WgpuFeature::TextureFormatsTier1) != 0;
        case F::Bgra8Unorm:
            return (features & pso::WgpuFeature::Bgra8UnormStorage) != 0;
        default:
            return false;
    }
}

uint32_t mips_or_all(uint32_t v) {
    return v == 0xFFFFFFFFu ? 0xffffu : v;
}

// Bytes of the whole console-layout surface from its base address (all mips and slices: on GFX10
// mip 0 is stored last, or inside the mip-tail block; see gfx10::MipPlacement). Returns 0 for
// D3D12 layouts 0-3 or unrecognized formats/modes.
uint64_t tiled_surface_bytes(uint32_t d3d12_layout, uint32_t dxgi_format, uint32_t width, uint32_t height,
                             uint32_t depth, uint32_t mip_levels, uint32_t array_size) noexcept {
    const gfx10::SwizzleMode mode = gfx10::decode_d3d12_layout(d3d12_layout);
    if (mode == gfx10::SwizzleMode::Unsupported ||
        (mode == gfx10::SwizzleMode::Linear && d3d12_layout != gfx10::kConsoleLinearLayout)) {
        return 0;
    }
    gfx10::TextureLayoutDesc desc{};
    desc.dxgi_format = dxgi_format;
    desc.width = width;
    desc.height = height;
    desc.depth = depth;
    desc.array_size = array_size;
    desc.mip_levels = std::max<uint32_t>(1u, mip_levels);
    desc.d3d12_layout = d3d12_layout;
    return gfx10::surface_bytes(desc);
}

bool read_guest_best_effort(uint64_t cpu_va, uint8_t* dst, size_t size) {
    if (size == 0) {
        return false;
    }
    if (com::read_guest(cpu_va, dst, size)) {
        return true;
    }
    uint8_t probe = 0;
    if (!com::read_guest(cpu_va, &probe, 1)) {
        return false;
    }
    std::memset(dst, 0, size);
    size_t off = 0;
    while (off < size) {
        const uint64_t cur = cpu_va + off;
        const size_t to_page_end = static_cast<size_t>(kShadowPage - (cur & (kShadowPage - 1)));
        const size_t step = std::min(size - off, to_page_end);
        if (!com::read_guest(cur, dst + off, step)) {
            if (off == 0) {
                dst[0] = probe;
            }
            break;
        }
        off += step;
    }
    return true;
}

}  // namespace

// O36: whether two byte ranges are equal. Without write tracking (wasm linear memory has none),
// flush_shadow compares every tracked CPU-visible buffer with its shadow at each submission: the
// real program tracks ~183 MB, compared twice a frame, the browser boot's frame-time bottleneck.
// The wasm64 libc memcmp compares 4 bytes per step; this compares 64 with wasm SIMD (the browser
// target requires SIMD anyway: lifted code is built with it), about 1.5x faster on the real
// program. The host memcmp is vectorized.
#if defined(__EMSCRIPTEN__)
__attribute__((target("simd128"))) bool same_bytes(const uint8_t* a, const uint8_t* b, size_t n) {
    typedef uint64_t u64x2 __attribute__((vector_size(16)));
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        u64x2 x[4];
        u64x2 y[4];
        std::memcpy(x, a + i, 64);
        std::memcpy(y, b + i, 64);
        const u64x2 d = (x[0] ^ y[0]) | (x[1] ^ y[1]) | (x[2] ^ y[2]) | (x[3] ^ y[3]);
        if ((d[0] | d[1]) != 0) {
            return false;
        }
    }
    for (; i < n; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}
#else
bool same_bytes(const uint8_t* a, const uint8_t* b, size_t n) {
    return std::memcmp(a, b, n) == 0;
}
#endif

namespace {

// R40: helper threads for the shadow comparison. Buffers without write tracking are compared whole
// at every submission: a memory-bandwidth-bound memcmp on the submitting (guest render) thread
// (before shared page views were tracked, os::take_written_pages, ~72% of its time in the menu
// flow: XMem physical pages are file views, which cannot carry MEM_WRITE_WATCH). Split
// across the caller and three helpers it runs ~1.7x faster in the boot (3.7 -> 6.1 GB/s; ~2x on
// an idle machine). run() hands out parts to the helpers and the caller alike and returns when all
// are done; concurrent callers take turns.
#if !defined(__EMSCRIPTEN__)
class ComparePool {
public:
    static ComparePool& get() {
        static ComparePool* pool = new ComparePool();  // never destroyed: helpers run until exit
        return *pool;
    }

    unsigned helpers() const { return static_cast<unsigned>(threads_.size()); }

    void run(unsigned parts, const std::function<void(unsigned)>& fn) {
        std::lock_guard<std::mutex> turn(run_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            fn_ = &fn;
            parts_ = parts;
            next_.store(0, std::memory_order_relaxed);
            done_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
        }
        wake_.notify_all();
        drain(fn, parts);
        // What is left is at most one part per helper (tens of microseconds): spin rather than pay
        // a blocking wait's wake-up latency once per buffer.
        auto finished = [&] {
            return done_.load(std::memory_order_acquire) == parts && active_.load(std::memory_order_acquire) == 0;
        };
        spin_until(finished);
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, finished);  // (under the lock: no helper can join after this)
        fn_ = nullptr;                   // a helper waking late finds no work
    }

private:
    ComparePool() {
        const unsigned cores = std::thread::hardware_concurrency();
        const unsigned helpers = std::min(3u, cores / 4);  // bandwidth-bound: 4 threads get most of it
        for (unsigned i = 0; i < helpers; ++i) {
            threads_.emplace_back([this] { helper(); });
        }
        for (std::thread& t : threads_) {
            t.detach();
        }
    }

    void drain(const std::function<void(unsigned)>& fn, unsigned parts) {
        for (;;) {
            const unsigned k = next_.fetch_add(1, std::memory_order_relaxed);
            if (k >= parts) {
                return;
            }
            fn(k);
            if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == parts) {
                std::lock_guard<std::mutex> lock(mutex_);
                finished_.notify_all();
            }
        }
    }

    // Spins (yielding) for up to ~200 us until `ready()`.
    template <typename Ready>
    static void spin_until(const Ready& ready) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
        while (!ready()) {
            if (std::chrono::steady_clock::now() > deadline) {
                return;
            }
            std::this_thread::yield();
        }
    }

    void helper() {
        uint64_t seen = 0;
        for (;;) {
            // A flush compares its buffers back to back: wait for the next one spinning first.
            spin_until([&] { return generation_.load(std::memory_order_acquire) != seen; });
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return generation_.load(std::memory_order_relaxed) != seen; });
            seen = generation_.load(std::memory_order_relaxed);
            if (fn_ == nullptr) {
                continue;
            }
            const std::function<void(unsigned)>* fn = fn_;
            const unsigned parts = parts_;
            active_.fetch_add(1, std::memory_order_relaxed);  // (under mutex_, while fn_ is set)
            lock.unlock();
            drain(*fn, parts);
            if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> relock(mutex_);
                finished_.notify_all();
            }
        }
    }

    std::mutex run_mutex_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable finished_;
    const std::function<void(unsigned)>* fn_ = nullptr;  // guarded by mutex_
    unsigned parts_ = 0;                                 // guarded by mutex_
    std::atomic<uint64_t> generation_{0};                // changed under mutex_
    std::atomic<unsigned> active_{0};                    // helpers inside drain(); joins under mutex_
    std::atomic<unsigned> next_{0};
    std::atomic<unsigned> done_{0};
    std::vector<std::thread> threads_;
};
#endif

}  // namespace

void compare_pages(size_t pages,
                   uint64_t bytes,
                   const std::function<bool(size_t)>& same,
                   std::vector<uint8_t>* changed) {
    changed->assign(pages, 0);
    auto compare = [&](size_t first, size_t last) {
        for (size_t i = first; i < last; ++i) {
            (*changed)[i] = same(i) ? 0 : 1;
        }
    };
#if !defined(__EMSCRIPTEN__)
    constexpr uint64_t kPartBytes = 256u << 10;
    constexpr uint64_t kParallelMinBytes = 1u << 20;
    ComparePool& pool = ComparePool::get();
    if (pool.helpers() != 0 && pages != 0 && bytes >= kParallelMinBytes) {
        const uint64_t page_size = std::max<uint64_t>(1, bytes / pages);
        const size_t per_part = static_cast<size_t>(std::max<uint64_t>(1, kPartBytes / page_size));
        const unsigned parts = static_cast<unsigned>((pages + per_part - 1) / per_part);
        pool.run(parts, [&](unsigned k) {
            const size_t first = static_cast<size_t>(k) * per_part;
            compare(first, std::min(pages, first + per_part));
        });
        return;
    }
#endif
    compare(0, pages);
}

uint64_t shader_identity(const void* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ull;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x00000100000001b3ull;
    }
    return h;
}

namespace {

inline uint64_t load_u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint64_t rotl64(uint64_t v, int r) {
    return (v << r) | (v >> (64 - r));
}

constexpr uint64_t kContentPrime1 = 0x9E3779B185EBCA87ull;
constexpr uint64_t kContentPrime2 = 0xC2B2AE3D27D4EB4Full;

inline uint64_t content_round(uint64_t acc, uint64_t word) {
    return rotl64(acc + word * kContentPrime2, 31) * kContentPrime1;
}

}  // namespace

uint64_t content_hash(const void* data, size_t size) {
    // Four independent lanes (xxHash64-style rounds) so the multiplies overlap.
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t a = kContentPrime1 + kContentPrime2;
    uint64_t b = kContentPrime2;
    uint64_t c = 0;
    uint64_t d = 0 - kContentPrime1;
    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        a = content_round(a, load_u64(p + i));
        b = content_round(b, load_u64(p + i + 8));
        c = content_round(c, load_u64(p + i + 16));
        d = content_round(d, load_u64(p + i + 24));
    }
    uint64_t h = rotl64(a, 1) + rotl64(b, 7) + rotl64(c, 12) + rotl64(d, 18) + static_cast<uint64_t>(size);
    for (; i + 8 <= size; i += 8) {
        h = rotl64(h ^ content_round(0, load_u64(p + i)), 27) * kContentPrime1 + 0x85EBCA77C2B2AE63ull;
    }
    for (; i < size; ++i) {
        h = rotl64(h ^ (p[i] * 0x27D4EB2F165667C5ull), 11) * kContentPrime1;
    }
    h ^= h >> 33;
    h *= kContentPrime2;
    h ^= h >> 29;
    return h;
}

bool all_zero_bytes(const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t i = 0;
    for (; i + 64 <= size; i += 64) {
        uint64_t acc = 0;
        for (size_t k = 0; k < 64; k += 8) {
            acc |= load_u64(p + i + k);
        }
        if (acc != 0) {
            return false;
        }
    }
    for (; i < size; ++i) {
        if (p[i] != 0) {
            return false;
        }
    }
    return true;
}

uint64_t native_stage_identity(const void* code, size_t code_size, const void* metadata, size_t metadata_size) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const auto& [data, size] : {std::pair<const void*, size_t>{code, code_size}, {metadata, metadata_size}}) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i) {
            h ^= p[i];
            h *= 0x00000100000001b3ull;
        }
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

void Device::note_unresolved_shader(uint64_t identity, ShaderStage stage) {
    static constexpr const char* kStageNames[] = {"vertex", "pixel", "compute"};
    const size_t index = static_cast<size_t>(stage);
    std::lock_guard<std::mutex> lock(shader_mutex_);
    if (std::find(unresolved_shaders_.begin(), unresolved_shaders_.end(), identity) == unresolved_shaders_.end()) {
        unresolved_shaders_.push_back(identity);
        ++unresolved_by_stage_[index];
        char name[24];
        std::snprintf(name, sizeof(name), "0x%016" PRIx64, identity);
        auto facts = shader_sizes_.find(identity);
        const int kind = facts != shader_sizes_.end() ? facts->second.kind : -1;
        os::log(std::string("d3d12_xs: no shader module for DXIL identity ") + name + " (" + kStageNames[index] +
                " shader, DXIL kind " + std::to_string(kind) + "; " +
                (stage == ShaderStage::Compute ? "dispatches" : "draws") + " with it are skipped)");
    }
}

std::array<uint32_t, 3> Device::unresolved_shader_stages() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return unresolved_by_stage_;
}

void Device::note_native_stages(uint64_t vertex_identity, uint64_t pixel_identity) {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    const bool first = native_stages_[0].empty();
    if (vertex_identity != 0) {
        native_stages_[0].insert(vertex_identity);
    }
    if (pixel_identity != 0) {
        native_stages_[1].insert(pixel_identity);
    }
    if (first) {
        os::log("d3d12_xs: console-native graphics pipelines (Xbox slot 57): stage code is GPU machine code, "
                "not DXIL; their draws are skipped (distinct stages are counted, not listed)");
    }
}

std::array<uint32_t, 2> Device::native_stage_counts() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return {static_cast<uint32_t>(native_stages_[0].size()), static_cast<uint32_t>(native_stages_[1].size())};
}

std::array<uint32_t, 3> Device::root_signature_versions() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return root_signature_versions_;
}

void Device::note_shader_size(uint64_t identity, uint64_t bytes, int kind) {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    shader_sizes_[identity] = ShaderFacts{bytes, kind};
}

uint64_t Device::shader_size(uint64_t identity) const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    auto it = shader_sizes_.find(identity);
    return it != shader_sizes_.end() ? it->second.bytes : 0;
}

int Device::shader_kind(uint64_t identity) const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    auto it = shader_sizes_.find(identity);
    return it != shader_sizes_.end() ? it->second.kind : -1;
}

bool Device::claim_skipped_draw_report(const void* pipeline_state) {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return described_pipelines_.insert(pipeline_state).second;
}

void Device::add_skipped_draw_report(const std::string& text) {
    os::log(text);
    std::lock_guard<std::mutex> lock(shader_mutex_);
    if (skipped_draw_reports_.size() < 64) {
        skipped_draw_reports_.push_back(text);
    }
}

std::vector<std::string> Device::skipped_draw_reports() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return skipped_draw_reports_;
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
    slot_index_.remove(resource);
}

void Device::track_guest_texture(const TrackedTexture& tex) {
    if (tex.resource_id == 0 || tex.gpu_va == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    tracked_textures_[tex.resource_id] = tex;
}

void Device::untrack_guest_texture(uint32_t resource_id) {
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    tracked_textures_.erase(resource_id);
    copy_destinations_.erase(resource_id);
}

void Device::note_texture_copy_destination(uint32_t resource_id) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    copy_destinations_.insert(resource_id);
}

bool Device::tracked_texture_info(uint32_t resource_id, TrackedTexture* out) const {
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    const auto it = tracked_textures_.find(resource_id);
    if (it == tracked_textures_.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = it->second;
    }
    return true;
}

void Device::set_detile_fn(DetileFn fn) {
    std::lock_guard<std::mutex> lock(detile_mutex_);
    detile_texture_ = std::move(fn);
}

namespace {
std::atomic<uint64_t> g_submissions{0};
thread_local uint64_t t_submission = 0;  // 0 = outside ExecuteCommandLists
}  // namespace

void Device::begin_submission() {
    t_submission = g_submissions.fetch_add(1, std::memory_order_relaxed) + 1;
}

void Device::end_submission() {
    t_submission = 0;
}

bool Device::sync_bound_texture_memory(uint32_t resource_id, CommandEncoder& encoder) {
    if (resource_id == 0) {
        return false;
    }
    TrackedTexture tt{};
    {
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        if (copy_destinations_.count(resource_id) != 0) {
            return false;  // a recorded copy defines the texels (note_texture_copy_destination)
        }
        const auto it = tracked_textures_.find(resource_id);
        if (it != tracked_textures_.end()) {
            if (t_submission != 0 && it->second.synced_submission == t_submission) {
                return false;  // already synced into an earlier part of this submission
            }
            it->second.synced_submission = t_submission;
            tt = it->second;
        } else {
            bridge::TextureInfo info{};
            if (!bridge_->texture_info(resource_id, &info) || info.swapchain || info.gpu_va == 0 ||
                info.gpu_va >= kFakeGpuVaBase || info.sample_count > 1) {
                return false;
            }
            tt.resource_id = resource_id;
            tt.gpu_va = info.gpu_va;
            tt.dxgi_format = static_cast<uint32_t>(info.format);
            tt.width = info.width;
            tt.height = info.height;
            tt.depth = info.depth_or_array_size;
            tt.mip_levels = info.mip_levels;
            tt.layout = info.layout;
            tt.dimension = info.dimension == GpuTextureDimension::Dim1D
                               ? ResourceDimension::Texture1D
                               : (info.dimension == GpuTextureDimension::Dim3D ? ResourceDimension::Texture3D
                                                                             : ResourceDimension::Texture2D);
            tt.sample_count = info.sample_count;
            tt.synced_submission = t_submission;
            tracked_textures_[resource_id] = tt;
        }
    }
    if (tt.gpu_va == 0 || tt.width == 0 || tt.height == 0 || tt.sample_count > 1) {
        return false;
    }
    const DxgiFormatLayout l = dxgi_format_layout(tt.dxgi_format);
    if (!l.known || l.block_bytes == 0 || l.block_width == 0 || l.block_height == 0 || l.planes > 1) {
        return false;
    }
    bridge::TextureInfo created{};
    const bool sampled_only = bridge_->texture_info(resource_id, &created) && created.sampled_only;
    const pso::TextureFormatMapping mapping = pso::map_texture_format(
        static_cast<pso::DxgiFormat>(tt.dxgi_format),
        sampled_only ? pso::FormatUsage::SampledOnly : pso::FormatUsage::Sampled, bridge_->options().features);
    if (!mapping.supported() || pso::has_depth_aspect(mapping.format) || pso::has_stencil_aspect(mapping.format)) {
        return false;
    }
    // 16-bit normalized texels for a SampledOnly texture: converted to its float representation
    // (pso::convert_norm16_texels; exact as float32, reported once when it is float16).
    uint8_t probe_dst[4] = {};
    const uint8_t probe_src[2] = {};
    const bool convert_norm16 = (mapping.diagnostics & pso::FormatDiag::NeedsCpuConversion) != 0 &&
                                pso::convert_norm16_texels(static_cast<pso::DxgiFormat>(tt.dxgi_format),
                                                           mapping.format, probe_src, 1, probe_dst);
    if (convert_norm16 && (mapping.diagnostics & pso::FormatDiag::LossyConversion) != 0) {
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
            first = lossy_reported_.insert(tt.dxgi_format).second;
        }
        if (first) {
            os::log(std::string("d3d12_xs: DXGI format ") + std::to_string(tt.dxgi_format) +
                    " is sampled as " + pso::texture_format_name(mapping.format) +
                    " (16-bit normalized formats are not filterable in WebGPU and float32-filterable is not "
                    "enabled): uploads round each value to float16 (11-bit mantissa)");
        }
    }
    // A8_UNORM held as rgba8unorm: each texel widened to (0, 0, 0, a).
    const bool convert_a8 = (mapping.diagnostics & pso::FormatDiag::NeedsCpuConversion) != 0 &&
                            pso::widen_a8_texels(static_cast<pso::DxgiFormat>(tt.dxgi_format), mapping.format,
                                                 probe_src, 1, probe_dst);
    if ((mapping.diagnostics & pso::FormatDiag::NeedsCpuConversion) != 0 && !convert_norm16 && !convert_a8) {
        // The GPU texture has a substitute format (e.g. rgba8unorm for BC without the feature):
        // the guest's texels do not fit it and are not converted. Skip, reported once per texture.
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
            first = conversion_skipped_.insert(resource_id).second;
        }
        if (first) {
            note_unsupported_copy("texture upload: DXGI format " + std::to_string(tt.dxgi_format) +
                                  " has a substitute WebGPU format that needs CPU conversion (not implemented); "
                                  "texels not uploaded");
        }
        return false;
    }
    const uint32_t w = tt.width;
    const uint32_t h = (tt.dimension == ResourceDimension::Texture1D) ? 1u : std::max<uint32_t>(1u, tt.height);
    const uint32_t d = (tt.dimension == ResourceDimension::Texture3D) ? std::max<uint32_t>(1u, tt.depth) : 1u;
    const uint32_t layers = (tt.dimension == ResourceDimension::Texture3D) ? 1u : std::max<uint32_t>(1u, tt.depth);
    const uint32_t mips = std::max<uint32_t>(1u, tt.mip_levels);
    const uint32_t blocks_x = (w + l.block_width - 1u) / l.block_width;
    const uint32_t blocks_y = (h + l.block_height - 1u) / l.block_height;
    const uint64_t tight_row_bytes = uint64_t{blocks_x} * l.block_bytes;
    const uint64_t tight_mip0_bytes = tight_row_bytes * blocks_y * d;
    va_watch::touch("upload (bound texture memory)", tt.gpu_va, tight_mip0_bytes * layers,
                    "texture id " + std::to_string(resource_id) + ", DXGI format " + std::to_string(tt.dxgi_format));
    constexpr uint64_t kMaxSyncTextureBytes = 64u << 20;
    if (tight_mip0_bytes == 0 || align_up(tight_row_bytes, 256u) * blocks_y * d > kMaxSyncTextureBytes) {
        return false;
    }

    // Console layouts (0x100 | swizzle mode, linear 0x100 included) are placed by the GFX10
    // surface code through the detile hook; D3D12 layouts 0-3 are read as tight rows.
    const bool is_tiled = tt.layout >= gfx10::kConsoleLinearLayout;
    const bool is_tiny = (w <= 2u && h <= 2u);
    DetileFn detile_fn;
    {
        std::lock_guard<std::mutex> lock(detile_mutex_);
        detile_fn = detile_texture_;
    }
    if (is_tiled && !detile_fn && !is_tiny) {
        return false;
    }

    // The subresources this layout places (subresource index = mip + slice * mips). Console
    // layouts: every mip and array slice of the GFX10 surface (gfx10::MipPlacement; linear 0x100
    // follows addrlib's ADDR_SW_LINEAR chain, #678). D3D12 layouts 0-3: mip 0 at the base; array
    // slices only where every candidate rule gives the same slice stride (one mip, rows a
    // multiple of 256 B).
    struct Sub {
        uint32_t mip;
        uint32_t slice;
    };
    std::vector<Sub> subs;
    uint64_t tiled_bytes = 0;
    if (is_tiled && detile_fn) {
        tiled_bytes = tiled_surface_bytes(tt.layout, tt.dxgi_format, w, h, d, mips, layers);
    }
    const bool whole_surface = tiled_bytes != 0;
    const uint32_t linear_layers = (!is_tiled && layers > 1u && mips == 1u && tight_row_bytes % 256u == 0u) ? layers : 1u;
    for (uint32_t s = 0; s < (whole_surface ? layers : linear_layers); ++s) {
        for (uint32_t m = 0; m < (whole_surface ? mips : 1u); ++m) {
            subs.push_back({m, s});
        }
    }

    // Safely read the source bytes via CPU-window translation (never crashes on unmapped/guard pages).
    const uint64_t cpu_va = cpu_address_for_gpu(tt.gpu_va);
    std::vector<uint8_t> raw;
    bool read_ok = false;
    if (is_tiled && detile_fn) {
        if (tiled_bytes == 0) {
            GuestResourceDesc tmp{};
            tmp.Dimension = tt.dimension;
            tmp.Width = w;
            tmp.Height = h;
            tmp.DepthOrArraySize = static_cast<uint16_t>(std::min<uint32_t>(d, 0xFFFFu));
            tmp.MipLevels = 1;
            tmp.Format = tt.dxgi_format;
            tmp.SampleCount = 1;
            tmp.Layout = tt.layout;
            tiled_bytes = committed_memory_size(tmp);
        }
        const uint64_t read_size = std::max<uint64_t>(tiled_bytes, tight_mip0_bytes);
        if (read_size > 0 && read_size <= kMaxSyncTextureBytes) {
            raw.resize(static_cast<size_t>(read_size));
            read_ok = com::read_guest(cpu_va, raw.data(), raw.size());
        }
    }
    if (!read_ok) {
        raw.resize(static_cast<size_t>(tight_mip0_bytes * linear_layers));
        read_ok = com::read_guest(cpu_va, raw.data(), raw.size());
    }
    if (!read_ok) {
        return false;
    }

    const bool all_zero = all_zero_bytes(raw.data(), raw.size());
    if (all_zero && !tt.ever_uploaded) {
        return false;
    }

    const uint64_t hash = content_hash(raw.data(), raw.size());
    if (tt.ever_uploaded && hash == tt.last_upload_hash) {
        return false;
    }

    // Stage every placed subresource; only those whose texels changed since their last upload
    // are written (per-subresource dirty tracking), so one changed mip costs one upload.
    struct Staged {
        Sub sub;
        uint32_t copy_w, copy_h, copy_d, row_pitch, rows;
        uint64_t hash;
        std::vector<uint8_t> bytes;
    };
    std::vector<Staged> staged;
    for (const Sub& sub : subs) {
        const uint32_t mw = std::max(1u, w >> sub.mip);
        const uint32_t mh = std::max(1u, h >> sub.mip);
        const uint32_t md = std::max(1u, d >> sub.mip);
        const uint32_t bx = (mw + l.block_width - 1u) / l.block_width;
        const uint32_t by = (mh + l.block_height - 1u) / l.block_height;
        const uint64_t row_bytes = uint64_t{bx} * l.block_bytes;
        const uint64_t row_pitch = align_up(row_bytes, 256u);
        Staged st{sub, bx * l.block_width, by * l.block_height, md, static_cast<uint32_t>(row_pitch), by, 0, {}};
        st.bytes.assign(static_cast<size_t>(row_pitch * by * md), 0);
        bool detiled = false;
        if (is_tiled && detile_fn) {
            TextureDetileRequest req{};
            req.dxgi_format = tt.dxgi_format;
            req.width = w;
            req.height = h;
            req.depth = d;
            req.mip_levels = mips;
            req.layout = tt.layout;
            req.block_bytes = l.block_bytes;
            req.block_width = l.block_width;
            req.block_height = l.block_height;
            req.dst_row_pitch = st.row_pitch;
            req.array_size = whole_surface ? layers : 1u;
            req.mip = sub.mip;
            req.array_slice = sub.slice;
            detiled = detile_fn(req, raw.data(), raw.size(), st.bytes.data(), st.bytes.size());
            if (!detiled && (sub.mip != 0u || sub.slice != 0u)) {
                continue;  // the hook does not place this subresource
            }
            if (!detiled && !is_tiny) {
                return false;
            }
        }
        if (!detiled) {
            const uint64_t slice_off = uint64_t{sub.slice} * tight_mip0_bytes;
            const uint32_t total_rows = by * md;
            for (uint32_t r = 0; r < total_rows; ++r) {
                std::memcpy(st.bytes.data() + static_cast<size_t>(r) * row_pitch,
                            raw.data() + static_cast<size_t>(slice_off + r * row_bytes), static_cast<size_t>(row_bytes));
            }
        }
        if (convert_norm16) {
            // Widen each row of 16-bit normalized components into the texture's float format.
            const size_t comps = static_cast<size_t>(row_bytes / 2u);
            const size_t out_bpc = mapping.format == pso::WgpuTextureFormat::R32Float ||
                                           mapping.format == pso::WgpuTextureFormat::Rg32Float ||
                                           mapping.format == pso::WgpuTextureFormat::Rgba32Float
                                       ? 4u
                                       : 2u;
            const uint64_t out_pitch = align_up(comps * out_bpc, 256u);
            std::vector<uint8_t> converted(static_cast<size_t>(out_pitch * by * md), 0);
            for (uint32_t r = 0; r < by * md; ++r) {
                pso::convert_norm16_texels(static_cast<pso::DxgiFormat>(tt.dxgi_format), mapping.format,
                                           st.bytes.data() + static_cast<size_t>(r) * row_pitch, comps,
                                           converted.data() + static_cast<size_t>(r) * out_pitch);
            }
            st.bytes = std::move(converted);
            st.row_pitch = static_cast<uint32_t>(out_pitch);
        }
        if (convert_a8) {
            const uint64_t out_pitch = align_up(row_bytes * 4u, 256u);
            std::vector<uint8_t> converted(static_cast<size_t>(out_pitch * by * md), 0);
            for (uint32_t r = 0; r < by * md; ++r) {
                pso::widen_a8_texels(static_cast<pso::DxgiFormat>(tt.dxgi_format), mapping.format,
                                     st.bytes.data() + static_cast<size_t>(r) * row_pitch, static_cast<size_t>(row_bytes),
                                     converted.data() + static_cast<size_t>(r) * out_pitch);
            }
            st.bytes = std::move(converted);
            st.row_pitch = static_cast<uint32_t>(out_pitch);
        }
        st.hash = content_hash(st.bytes.data(), st.bytes.size());
        staged.push_back(std::move(st));
    }

    {
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        const auto it = tracked_textures_.find(resource_id);
        if (it == tracked_textures_.end()) {
            return false;
        }
        TrackedTexture& cur = it->second;
        if (cur.ever_uploaded && cur.last_upload_hash == hash) {
            return false;
        }
        cur.last_upload_hash = hash;
        cur.ever_uploaded = true;
        cur.subresource_hashes.resize(size_t{mips} * layers, 0);
        auto out = staged.begin();
        for (Staged& st : staged) {
            uint64_t& seen = cur.subresource_hashes[st.sub.mip + size_t{st.sub.slice} * mips];
            const bool zero = all_zero_bytes(st.bytes.data(), st.bytes.size());
            if (seen == st.hash || (seen == 0 && zero)) {
                continue;  // unchanged, or still zero as the GPU texture was created
            }
            seen = st.hash;
            if (&*out != &st) {
                *out = std::move(st);
            }
            ++out;
        }
        staged.erase(out, staged.end());
    }

    for (const Staged& st : staged) {
        encoder.write_texture(resource_id, st.sub.mip, 0u, 0u, st.sub.slice, st.copy_w, st.copy_h, st.copy_d,
                              st.row_pitch, st.rows, st.bytes.data(), static_cast<uint32_t>(st.bytes.size()));
        count_work(Work::TextureUploads);
        note_upload(st.bytes.size());
        note_texture_upload();
    }
    return !staged.empty();
}

uint32_t Device::ensure_address_texture(uint64_t gpu_va, uint32_t dxgi_format, uint32_t width, uint32_t height,
                                        uint32_t depth, uint32_t mip_levels, uint32_t d3d12_layout,
                                        GpuTextureDimension dim, bool uav) {
    if (gpu_va == 0 || width < 1u || height < 1u || width > 16384u || height > 16384u) {
        return 0;
    }
    const uint32_t safe_height = dim == GpuTextureDimension::Dim1D ? 1u : height;
    const uint32_t safe_depth = std::max<uint32_t>(1u, depth);
    if (safe_depth > 2048u) {
        return 0;
    }
    uint32_t max_mips = 1u;
    uint32_t extent = std::max<uint32_t>(width, safe_height);
    if (dim == GpuTextureDimension::Dim3D) {
        extent = std::max<uint32_t>(extent, safe_depth);
    }
    while (extent > 1u) {
        extent >>= 1u;
        ++max_mips;
    }
    const uint32_t safe_mips = mip_levels == 0u ? max_mips : std::min<uint32_t>(mip_levels, max_mips);

    // A texture known only by a view is never a render target: SampledOnly.
    const pso::TextureFormatMapping mapping = pso::map_texture_format(
        static_cast<pso::DxgiFormat>(dxgi_format), pso::FormatUsage::SampledOnly, bridge_->options().features);
    if (!mapping.supported() || pso::has_depth_aspect(mapping.format) || pso::has_stencil_aspect(mapping.format)) {
        return 0;
    }

    AddressTextureKey key{};
    key.gpu_va = gpu_va;
    key.dxgi_format = dxgi_format;
    key.width = width;
    key.height = safe_height;
    key.depth = safe_depth;
    key.mip_levels = safe_mips;
    key.d3d12_layout = d3d12_layout;
    key.dim = static_cast<uint8_t>(dim);

    std::lock_guard<std::mutex> lock(address_textures_mutex_);
    const auto it = address_textures_.find(key);
    if (it != address_textures_.end()) {
        return it->second;
    }

    const uint32_t id = allocate_id();
    GpuCreateTexture2Payload payload{};
    payload.id = id;
    payload.width = width;
    payload.height = safe_height;
    payload.depth_or_layers = safe_depth;
    payload.mip_levels = safe_mips;
    payload.sample_count = 1;
    payload.dimension = static_cast<uint8_t>(dim);
    payload.usage_flags = GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst;
    if (uav && storage_capable(mapping.format, bridge_->options().features)) {
        payload.usage_flags |= GpuTextureUsage::StorageBinding;
    }
    payload.format = static_cast<uint8_t>(mapping.format);
    if (const pso::WgpuTextureFormat sibling = pso::srgb_sibling(mapping.format);
        sibling != pso::WgpuTextureFormat::Undefined) {
        payload.view_format_count = 1;  // other descriptors may view it sRGB / linear
        payload.view_formats[0] = static_cast<uint8_t>(sibling);
    }

    CommandEncoder enc;
    enc.emit_packet(GpuOpcode::CreateTexture2, &payload, sizeof(payload));
    submit(enc);

    bridge::TextureInfo info{};
    info.format = static_cast<pso::DxgiFormat>(dxgi_format);
    info.width = width;
    info.height = safe_height;
    info.depth_or_array_size = safe_depth;
    info.mip_levels = safe_mips;
    info.sample_count = 1;
    info.dimension = dim;
    info.gpu_va = gpu_va;
    info.layout = d3d12_layout;
    info.sampled_only = true;
    info.storage = (payload.usage_flags & GpuTextureUsage::StorageBinding) != 0;
    bridge_->register_texture(id, info);

    if (gpu_va < kFakeGpuVaBase) {
        TrackedTexture tt{};
        tt.resource_id = id;
        tt.gpu_va = gpu_va;
        tt.dxgi_format = dxgi_format;
        tt.width = width;
        tt.height = safe_height;
        tt.depth = safe_depth;
        tt.mip_levels = safe_mips;
        tt.layout = d3d12_layout;
        tt.dimension = dim == GpuTextureDimension::Dim1D
                           ? ResourceDimension::Texture1D
                           : (dim == GpuTextureDimension::Dim3D ? ResourceDimension::Texture3D
                                                                : ResourceDimension::Texture2D);
        tt.sample_count = 1;
        track_guest_texture(tt);
        // "Textures with data" counter: the surface's bytes (console layouts), else the tight mip 0.
        const bool is3d = dim == GpuTextureDimension::Dim3D;
        uint64_t size = tiled_surface_bytes(d3d12_layout, dxgi_format, width, safe_height, is3d ? safe_depth : 1u,
                                            safe_mips, is3d ? 1u : safe_depth);
        if (size == 0) {
            const DxgiFormatLayout l = dxgi_format_layout(dxgi_format);
            size = l.known && l.block_width != 0 && l.block_height != 0
                       ? uint64_t{(width + l.block_width - 1u) / l.block_width} *
                             ((safe_height + l.block_height - 1u) / l.block_height) * l.block_bytes
                       : 0u;
        }
        if (size != 0) {
            register_view_texture_probe(cpu_address_for_gpu(gpu_va), size);
        }
    }

    address_textures_.emplace(key, id);
    return id;
}

void Device::flush_cpu_buffers(CommandEncoder& encoder) {
    std::lock_guard<std::mutex> flush_lock(flush_mutex_);
    // O36: one pass over the translated window's write marks for the buffers and the relocated
    // ranges below. A relocated range in the window whose slots are unmarked has not been written
    // since its bytes were last read (the same protocol as the buffers: marks taken before the
    // read), so it is not read and compared again.
    mark_pass_.begin();
    std::vector<uint64_t> marked;
    auto unwritten = [&](uint64_t cpu, uint64_t size) {
        marked.clear();
        return mark_pass_.take(cpu, static_cast<size_t>(size), &marked) && marked.empty();
    };
    {
        std::lock_guard<std::mutex> lock(tracked_mutex_);
        // O36: take every dirty window slot once, then visit only the buffers overlapping one (plus
        // the buffers not indexed yet: non-window, or whose first full comparison is still due).
        // Verification mode compares unmarked pages too, so it visits everything.
        std::vector<uint64_t> dirty;
        const bool scanned = mark_pass_.take_dirty(&dirty);
        if (!scanned || os::guest_window::write_marks_verified()) {
            slot_index_.clear();
            for (Resource* r : tracked_) {
                r->set_slot_indexed(false);
                r->flush_shadow(encoder, &mark_pass_);
            }
        } else {
            std::vector<const void*> visit;
            slot_index_.collect(dirty, &visit);
            for (Resource* r : tracked_) {
                if (r->slot_indexed()) {
                    continue;
                }
                r->flush_shadow(encoder, &mark_pass_);
                if (r->window_marks_in_use()) {
                    slot_index_.add(r, r->cpu_address(), r->shadow_bytes());
                    r->set_slot_indexed(true);
                }
            }
            for (const void* p : visit) {
                Resource* r = const_cast<Resource*>(static_cast<const Resource*>(p));
                r->flush_shadow(encoder, &mark_pass_);
                if (!r->window_marks_in_use()) {
                    slot_index_.remove(r);
                    r->set_slot_indexed(false);
                }
            }
        }
    }
    // Relocated uniform ranges: refresh the aligned copies whose guest bytes changed.
    {
        std::lock_guard<std::mutex> lock(relocation_mutex_);
        std::vector<uint8_t> bytes;
        for (auto& [key, r] : relocations_) {
            const uint64_t cpu = cpu_address_for_gpu(r.va);
            if (unwritten(cpu, r.size) && !r.shadow.empty()) {
                continue;
            }
            bytes.resize(static_cast<size_t>(r.size));  // only the view's bytes: the home past them stays zero
            if (!com::read_guest(cpu, bytes.data(), bytes.size()) || bytes == r.shadow) {
                continue;
            }
            encoder.write_buffer(r.buffer_id, static_cast<uint32_t>(r.offset), bytes.data(), static_cast<uint32_t>(r.size));
            r.shadow = bytes;
            note_upload(r.size);
        }
    }
    // Relocated storage buffers (G51): refresh any relocated range whose guest bytes changed.
    {
        std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
        std::vector<uint8_t> bytes;
        for (auto& [key, r] : storage_relocations_) {
            const uint64_t write_len = align_up(r.size, 4u);
            const uint64_t cpu = cpu_address_for_gpu(r.va);
            if (unwritten(cpu, r.size) && r.ever_uploaded) {
                continue;
            }
            bytes.resize(static_cast<size_t>(r.size));
            if (!read_guest_best_effort(cpu, bytes.data(), bytes.size())) {
                continue;
            }
            const uint64_t hash = content_hash(bytes.data(), bytes.size());
            if (r.ever_uploaded && hash == r.last_hash) {
                continue;
            }
            bytes.resize(static_cast<size_t>(write_len), 0);
            for (uint64_t off = 0; off < write_len; off += kMaxWriteChunk) {
                const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, write_len - off);
                encoder.write_buffer(r.buffer_id, static_cast<uint32_t>(r.offset + off),
                                     bytes.data() + off, static_cast<uint32_t>(n));
            }
            count_work(Work::BufferUploads);
            note_upload(write_len);
            r.last_hash = hash;
            r.ever_uploaded = true;
        }
    }
    // Relocated index / vertex buffers (G58): re-upload ring buffers the title wrote since.
    {
        std::lock_guard<std::mutex> lock(ia_relocation_mutex_);
        std::vector<uint8_t> bytes;
        for (auto& [key, r] : ia_relocations_) {
            const uint64_t cpu = cpu_address_for_gpu(key.first);
            if (unwritten(cpu, key.second)) {
                continue;  // (uploaded when it was created)
            }
            bytes.assign(static_cast<size_t>(key.second), 0);
            if (!read_guest_best_effort(cpu, bytes.data(), bytes.size())) {
                continue;
            }
            const uint64_t hash = content_hash(bytes.data(), bytes.size());
            if (hash == r.last_hash) {
                continue;
            }
            bytes.resize(static_cast<size_t>(r.buffer_bytes), 0);
            for (uint64_t off = 0; off < r.buffer_bytes; off += kMaxWriteChunk) {
                const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, r.buffer_bytes - off);
                encoder.write_buffer(r.buffer_id, static_cast<uint32_t>(off), bytes.data() + off,
                                     static_cast<uint32_t>(n));
            }
            count_work(Work::BufferUploads);
            note_upload(r.buffer_bytes);
            r.last_hash = hash;
        }
    }
}

bool Device::relocate_uniform(uint64_t va, uint64_t size, uint64_t binding_size, uint32_t* buffer_id,
                              uint64_t* offset) {
    constexpr uint64_t kChunk = 1u << 20;
    const uint64_t need = align_up(std::max(size, binding_size), bridge::kUniformOffsetAlignment);
    if (size == 0 || need > (64u << 10)) {
        return false;
    }
    uint8_t probe = 0;
    if (!com::read_guest(cpu_address_for_gpu(va), &probe, 1)) {
        return false;  // GPU-only memory: no bytes to copy
    }
    std::lock_guard<std::mutex> lock(relocation_mutex_);
    auto it = relocations_.find({va, size, need});
    if (it == relocations_.end()) {
        // A fresh home: the chunk is zero-initialised and homes are never reused, so the bytes
        // past `size` stay zero (zero-extension for WebGPU's minimum binding size, not data).
        if (relocation_buffer_ == 0 || relocation_used_ + need > kChunk) {
            relocation_buffer_ = allocate_id();
            CommandEncoder enc;
            enc.create_buffer(relocation_buffer_, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, kChunk);
            submit(enc);
            relocation_used_ = 0;
        }
        RelocatedUniform r;
        r.va = va;
        r.size = size;
        r.buffer_id = relocation_buffer_;
        r.offset = relocation_used_;
        relocation_used_ += need;
        it = relocations_.emplace(std::make_tuple(va, size, need), std::move(r)).first;
    }
    *buffer_id = it->second.buffer_id;
    *offset = it->second.offset;
    return true;
}

size_t Device::relocated_uniform_count() const {
    std::lock_guard<std::mutex> lock(relocation_mutex_);
    return relocations_.size();
}

bool Device::relocate_storage_buffer(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset,
                                     bool require_zero_offset) {
    constexpr uint64_t kStorageOffsetAlignment = 256;
    constexpr uint64_t kChunk = uint64_t{4u} << 20;
    constexpr uint64_t kMaxBytes = uint64_t{16u} << 20;
    constexpr size_t kMaxEntries = 512;

    if (va == 0 || size == 0 || size > kMaxBytes || buffer_id == nullptr || offset == nullptr) {
        return false;
    }
    const uint64_t cpu_va = cpu_address_for_gpu(va);
    uint8_t probe = 0;
    if (!com::read_guest(cpu_va, &probe, 1)) {
        return false;  // unmapped / GPU-only memory: no bytes to copy
    }

    const uint64_t need = align_up(size, kStorageOffsetAlignment);
    const bool dedicated = require_zero_offset || need > kChunk;
    const RelocatedStorageKey key{va, size, dedicated};

    std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
    const uint64_t seq = ++storage_relocation_seq_;
    if (auto it = storage_relocations_.find(key); it != storage_relocations_.end()) {
        it->second.last_used_seq = seq;
        *buffer_id = it->second.buffer_id;
        *offset = it->second.offset;
        return true;
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(size), 0);
    if (!read_guest_best_effort(cpu_va, bytes.data(), bytes.size())) {
        return false;
    }
    const uint64_t hash = content_hash(bytes.data(), bytes.size());
    const uint64_t write_len = align_up(size, 4u);
    bytes.resize(static_cast<size_t>(write_len), 0);

    CommandEncoder enc;
    uint32_t target_buf = 0;
    uint64_t target_off = 0;
    uint64_t target_cap = 0;

    if (storage_relocations_.size() >= kMaxEntries) {
        auto reuse_it = storage_relocations_.end();
        auto lru_it = storage_relocations_.begin();
        for (auto it = storage_relocations_.begin(); it != storage_relocations_.end(); ++it) {
            if (it->second.last_used_seq < lru_it->second.last_used_seq) {
                lru_it = it;
            }
            if (it->second.dedicated == dedicated && it->second.slot_capacity >= need) {
                if (reuse_it == storage_relocations_.end() ||
                    it->second.last_used_seq < reuse_it->second.last_used_seq) {
                    reuse_it = it;
                }
            }
        }
        if (reuse_it != storage_relocations_.end()) {
            target_buf = reuse_it->second.buffer_id;
            target_off = reuse_it->second.offset;
            target_cap = reuse_it->second.slot_capacity;
            storage_relocations_.erase(reuse_it);
        } else {
            if (lru_it->second.dedicated) {
                const uint32_t old_id = lru_it->second.buffer_id;
                bridge_->unregister_resource(old_id);
                enc.destroy_resource(old_id);
                storage_relocation_buffers_.erase(
                    std::remove(storage_relocation_buffers_.begin(), storage_relocation_buffers_.end(), old_id),
                    storage_relocation_buffers_.end());
            }
            storage_relocations_.erase(lru_it);
        }
    }

    if (target_buf == 0) {
        if (dedicated) {
            target_buf = allocate_id();
            target_off = 0;
            target_cap = need;
            enc.create_buffer(target_buf, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, target_cap);
            bridge_->register_buffer(target_buf, allocate_gpu_va(target_cap), target_cap);
            storage_relocation_buffers_.push_back(target_buf);
        } else {
            if (storage_relocation_buffer_ == 0 || storage_relocation_used_ + need > kChunk) {
                storage_relocation_buffer_ = allocate_id();
                enc.create_buffer(storage_relocation_buffer_, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, kChunk);
                bridge_->register_buffer(storage_relocation_buffer_, allocate_gpu_va(kChunk), kChunk);
                storage_relocation_buffers_.push_back(storage_relocation_buffer_);
                storage_relocation_used_ = 0;
            }
            target_buf = storage_relocation_buffer_;
            target_off = storage_relocation_used_;
            target_cap = need;
            storage_relocation_used_ += need;
        }
    }

    for (uint64_t off = 0; off < write_len; off += kMaxWriteChunk) {
        const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, write_len - off);
        enc.write_buffer(target_buf, static_cast<uint32_t>(target_off + off),
                         bytes.data() + off, static_cast<uint32_t>(n));
    }
    submit(enc);
    count_work(Work::BufferUploads);
    note_upload(write_len);

    RelocatedStorage entry{};
    entry.va = va;
    entry.size = size;
    entry.buffer_id = target_buf;
    entry.offset = target_off;
    entry.slot_capacity = target_cap;
    entry.last_hash = hash;
    entry.last_used_seq = seq;
    entry.ever_uploaded = true;
    entry.dedicated = dedicated;
    storage_relocations_.emplace(key, entry);

    *buffer_id = target_buf;
    *offset = target_off;
    return true;
}

size_t Device::relocated_storage_count() const {
    std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
    return storage_relocations_.size();
}

bool Device::relocate_ia_buffer(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset) {
    constexpr uint64_t kMaxBytes = uint64_t{16u} << 20;
    constexpr uint64_t kMaxTotalBytes = uint64_t{256u} << 20;
    constexpr size_t kMaxEntries = 256;
    // WebGPU wants vertex buffer offsets 4-byte aligned and index offsets aligned to the index
    // size (2 or 4), so a sub-range is only served from a covering buffer at a 4-byte aligned offset.
    constexpr uint64_t kOffsetAlignment = 4;

    if (va == 0 || size == 0 || size > kMaxBytes || va + size < va || buffer_id == nullptr || offset == nullptr) {
        return false;
    }
    const uint64_t cpu_va = cpu_address_for_gpu(va);
    uint8_t probe = 0;
    if (!com::read_guest(cpu_va, &probe, 1)) {
        return false;  // unmapped / GPU-only memory: no bytes to copy
    }

    std::lock_guard<std::mutex> lock(ia_relocation_mutex_);
    const uint64_t seq = ++ia_relocation_seq_;
    // A relocation whose [base, base + size) covers the request: the title's ring buffer pointer
    // advanced within a range already mirrored (keys sort by base VA, so scan those <= va).
    for (auto it = ia_relocations_.upper_bound(std::make_pair(va, ~uint64_t{0})); it != ia_relocations_.begin();) {
        --it;
        const uint64_t base = it->first.first;
        const uint64_t mirrored = it->first.second;
        const uint64_t delta = va - base;
        if (delta % kOffsetAlignment == 0 && delta <= mirrored && size <= mirrored - delta) {
            it->second.last_used_seq = seq;
            *buffer_id = it->second.buffer_id;
            *offset = delta;
            return true;
        }
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(size), 0);
    if (!read_guest_best_effort(cpu_va, bytes.data(), bytes.size())) {
        return false;
    }
    const uint64_t hash = content_hash(bytes.data(), bytes.size());
    const uint64_t buffer_bytes = align_up(size, 4u);
    bytes.resize(static_cast<size_t>(buffer_bytes), 0);

    // Bounded like the storage relocations: the least recently bound mirror is destroyed.
    CommandEncoder enc;
    while (!ia_relocations_.empty() &&
           (ia_relocations_.size() >= kMaxEntries || ia_relocation_bytes_ + buffer_bytes > kMaxTotalBytes)) {
        auto lru = ia_relocations_.begin();
        for (auto it = ia_relocations_.begin(); it != ia_relocations_.end(); ++it) {
            if (it->second.last_used_seq < lru->second.last_used_seq) {
                lru = it;
            }
        }
        enc.destroy_resource(lru->second.buffer_id);
        ia_relocation_bytes_ -= lru->second.buffer_bytes;
        ia_relocations_.erase(lru);
    }

    const uint32_t id = allocate_id();
    enc.create_buffer(id, GpuBufferUsage::Vertex | GpuBufferUsage::Index | GpuBufferUsage::CopyDst, buffer_bytes);
    for (uint64_t off = 0; off < buffer_bytes; off += kMaxWriteChunk) {
        const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, buffer_bytes - off);
        enc.write_buffer(id, static_cast<uint32_t>(off), bytes.data() + off, static_cast<uint32_t>(n));
    }
    submit(enc);
    count_work(Work::BufferUploads);
    note_upload(buffer_bytes);

    RelocatedIa entry;
    entry.buffer_bytes = buffer_bytes;
    entry.buffer_id = id;
    entry.last_hash = hash;
    entry.last_used_seq = seq;
    ia_relocations_.emplace(std::make_pair(va, size), entry);
    ia_relocation_bytes_ += buffer_bytes;

    *buffer_id = id;
    *offset = 0;
    return true;
}

size_t Device::relocated_ia_count() const {
    std::lock_guard<std::mutex> lock(ia_relocation_mutex_);
    return ia_relocations_.size();
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
    count_work(Work::CopiesSkipped);
    unsupported_copies_.fetch_add(1, std::memory_order_relaxed);
    note_bad_argument(message);
}

bool Device::upload_needs_conversion(uint32_t dxgi_format) const {
    const pso::TextureFormatMapping m = pso::map_texture_format(static_cast<pso::DxgiFormat>(dxgi_format),
                                                                pso::FormatUsage::Sampled, bridge_->options().features);
    // A8_UNORM is converted on upload (widen_a8_rows).
    return m.supported() && (m.diagnostics & pso::FormatDiag::NeedsCpuConversion) != 0 &&
           static_cast<pso::DxgiFormat>(dxgi_format) != pso::DxgiFormat::A8_UNORM;
}

bool widen_a8_rows(const uint8_t* src, size_t src_size, uint32_t width, uint32_t rows, uint32_t slices,
                   uint32_t src_pitch, uint64_t slice_pitch, std::vector<uint8_t>* out, uint32_t* out_pitch) {
    const uint64_t pitch = align_up(uint64_t{width} * 4u, 256u);
    if (rows == 0 || slices == 0 ||
        uint64_t{slices - 1} * slice_pitch + uint64_t{rows - 1} * src_pitch + width > src_size) {
        return false;
    }
    out->assign(static_cast<size_t>(pitch * rows * slices), 0);
    for (uint32_t z = 0; z < slices; ++z) {
        for (uint32_t r = 0; r < rows; ++r) {
            pso::widen_a8_texels(pso::DxgiFormat::A8_UNORM, pso::WgpuTextureFormat::Rgba8Unorm,
                                 src + z * slice_pitch + uint64_t{r} * src_pitch, width,
                                 out->data() + (uint64_t{z} * rows + r) * pitch);
        }
    }
    *out_pitch = static_cast<uint32_t>(pitch);
    return true;
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
    if (id_ != 0) {
        count_work(Work::ResourcesDestroyed);
    }
    if (probe_registered_) {
        TextureProbe& p = texture_probe();
        std::lock_guard<std::mutex> lock(p.mutex);
        p.textures.erase(std::remove(p.textures.begin(), p.textures.end(), this), p.textures.end());
    }
    if (tracked_) {
        device_->untrack_cpu_buffer(this);
        device_->note_shadow_bytes(-static_cast<int64_t>(shadow_size_), -static_cast<int64_t>(shadow_resident_));
    }
    if (texture_tracked_) {
        device_->untrack_guest_texture(id_);
    }
    if (id_ != 0) {
        device_->state_bridge().unregister_resource(id_);
        device_->resource_states().unregister_resource(id_);
        // A GPU object whose create packet was never written (never used) needs no DestroyResource.
        if ((is_buffer() || gpu_texture_) && !device_->release_deferred(id_)) {
            CommandEncoder enc;
            enc.destroy_resource(id_);
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
    // size estimate (console layouts, including linear 0x100, use their GFX10 placement there).
    const DxgiFormatLayout l = dxgi_format_layout(d.Format);
    os::surface::Block64kLayout block{};
    if (d.Layout != gfx10::kConsoleLinearLayout && d.Dimension == ResourceDimension::Texture2D &&
        d.MipLevels <= 1 && d.DepthOrArraySize <= 1 &&
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
        // A small buffer is allocated with at least kSmallBufferPadding bytes: the title binds whole
        // 16-byte buffers (written by the GPU) as constant buffers to stages that declare uniform
        // blocks of up to a few KiB, and WebGPU needs the binding to cover the declared block. The
        // bytes past the guest's size are never written, so they stay zero (the zero-extension
        // WebGPU needs, not data; on the console the stage reads whatever memory follows).
        constexpr uint64_t kSmallBufferPadding = 4096;
        const uint64_t allocated = std::max<uint64_t>(align_up(size, 4), kSmallBufferPadding);
        CommandEncoder enc;
        enc.create_buffer(id_,
                          GpuBufferUsage::Vertex | GpuBufferUsage::Index | GpuBufferUsage::Uniform |
                              GpuBufferUsage::Storage | GpuBufferUsage::Indirect | GpuBufferUsage::CopyDst |
                              GpuBufferUsage::CopySrc,
                          allocated);
        dev.defer_create(id_, enc);  // written before the first packet that uses it (G66)
        dev.state_bridge().register_buffer(id_, gpu_va_, size, allocated);
        if (cpu_va_ != 0) {
            // O36: no host memory yet (see shadow_chunks_): the chunks come with the first uploads.
            shadow_size_ = size;
            shadow_chunks_.resize(static_cast<size_t>((size + kShadowChunk - 1) / kShadowChunk));
            dev.note_shadow_bytes(static_cast<int64_t>(size), 0);
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
        } else if (!depth && !placement_.placed_heap && !placement_.reserved && placement_.address == 0 &&
                   heap_cpu_visible(placement_.heap)) {
            const uint64_t size = align_up(std::max<uint64_t>(committed_memory_size(desc_), 1), 64u << 10);
            const uint64_t va = com::guest_memory().virtual_alloc(0, size, os::kMemReserve | os::kMemCommit,
                                                                  os::PageProtect::ReadWrite);
            if (va == 0) {
                dev.note_bad_argument("out of guest memory for a " + std::to_string(size) + "-byte texture");
                id_ = 0;
                return false;
            }
            owned_guest_memory_ = va;
        }
        // The texture's GPU VA is its memory (see the header comment).
        if (placement_.address != 0) {
            gpu_va_ = placement_.address;
            cpu_va_ = dev.cpu_address_for_gpu(gpu_va_);
        } else if (placement_.placed_heap) {
            gpu_va_ = placement_.placed_heap->base() + placement_.heap_offset;
            if (placement_.placed_heap->cpu_visible()) {
                cpu_va_ = gpu_va_;
            }
        } else if (xmem_memory_ != nullptr) {
            gpu_va_ = com::host_to_guest(xmem_memory_);
            cpu_va_ = dev.cpu_address_for_gpu(gpu_va_);
        } else if (owned_guest_memory_ != 0) {
            gpu_va_ = cpu_va_ = owned_guest_memory_;
        }
        if (gpu_va_ != 0 && xmem_memory_ == nullptr) {
            TextureProbe& p = texture_probe();
            std::lock_guard<std::mutex> lock(p.mutex);
            p.textures.push_back(this);
            probe_registered_ = true;
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
            p.dimension = static_cast<uint8_t>(desc_.Dimension == ResourceDimension::Texture1D
                                                   ? GpuTextureDimension::Dim1D
                                                   : (desc_.Dimension == ResourceDimension::Texture3D
                                                          ? GpuTextureDimension::Dim3D
                                                          : GpuTextureDimension::Dim2D));
            p.usage_flags = GpuTextureUsage::CopySrc | GpuTextureUsage::CopyDst;
            if ((desc_.Flags & ResourceFlags::DenyShaderResource) == 0) {
                p.usage_flags |= GpuTextureUsage::TextureBinding;
            }
            if ((rt || depth) && p.dimension != static_cast<uint8_t>(GpuTextureDimension::Dim1D)) {
                p.usage_flags |= GpuTextureUsage::RenderAttachment;
            }
            if ((desc_.Flags & ResourceFlags::AllowUnorderedAccess) != 0) {
                if (p.sample_count == 1 && storage_capable(m.format, dev.state_bridge().options().features)) {
                    p.usage_flags |= GpuTextureUsage::StorageBinding;
                } else {
                    // WebGPU rejects the whole texture when STORAGE_BINDING is not allowed for its
                    // format; such a texture keeps its other usages (its UAV bindings cannot work).
                    dev.note_bad_argument(std::string("UAV texture format ") + pso::texture_format_name(m.format) +
                                          " has no WebGPU storage binding: created without storage usage");
                }
            }
            p.format = static_cast<uint8_t>(m.format);
            // Views of a D3D12 resource may switch between the sRGB and the linear format (an
            // _SRGB texture rendered through a UNORM RTV, and the reverse): list the sibling.
            if (const pso::WgpuTextureFormat sibling = pso::srgb_sibling(m.format);
                sibling != pso::WgpuTextureFormat::Undefined) {
                p.view_format_count = 1;
                p.view_formats[0] = static_cast<uint8_t>(sibling);
            }
            CommandEncoder enc;
            enc.emit_packet(GpuOpcode::CreateTexture2, &p, sizeof(p));
            dev.defer_create(id_, enc);  // written before the first packet that uses it (G66)
            gpu_texture_ = true;
            // The address its descriptor words carry (descriptor_words.h): its GPU VA, or for a
            // texture without memory a range of the device's fake GPU VA space.
            descriptor_va_ = gpu_va_ != 0 ? gpu_va_
                                          : dev.allocate_gpu_va(std::max<uint64_t>(committed_memory_size(desc_), 1));
            bridge::TextureInfo info;
            info.format = static_cast<pso::DxgiFormat>(desc_.Format);
            info.width = p.width;
            info.height = p.height;
            info.depth_or_array_size = p.depth_or_layers;
            info.mip_levels = mips_;
            info.sample_count = p.sample_count;
            info.dimension = static_cast<GpuTextureDimension>(p.dimension);
            info.gpu_va = descriptor_va_;
            info.layout = desc_.Layout;
            info.storage = (p.usage_flags & GpuTextureUsage::StorageBinding) != 0;
            dev.state_bridge().register_texture(id_, info);
            va_watch::note_texture(descriptor_va_, committed_memory_size(desc_), p.width, p.height, desc_.Format, id_);
            if (gpu_va_ != 0 && gpu_va_ < Device::kFakeGpuVaBase) {
                TrackedTexture tt{};
                tt.resource_id = id_;
                tt.gpu_va = gpu_va_;
                tt.dxgi_format = desc_.Format;
                tt.width = p.width;
                tt.height = p.height;
                tt.depth = p.depth_or_layers;
                tt.mip_levels = mips_;
                tt.layout = desc_.Layout;
                tt.dimension = desc_.Dimension;
                tt.sample_count = p.sample_count;
                dev.track_guest_texture(tt);
                texture_tracked_ = true;
            }
        }
    }
    d3d::ResourceStateDesc sd;
    sd.subresource_count = subresource_count();
    sd.initial_state = initial_state_;
    sd.implicit_promotion = is_buffer() || (desc_.Flags & ResourceFlags::AllowSimultaneousAccess) != 0;
    dev.resource_states().register_resource(id_, sd);
    count_work(Work::ResourcesCreated);
    return true;
}

TextureProbeTotals probe_texture_contents(size_t budget) {
    constexpr uint64_t kProbeBytes = 64u << 10;
    TextureProbe& p = texture_probe();
    std::lock_guard<std::mutex> lock(p.mutex);
    for (size_t n = 0; n < budget && !p.textures.empty(); ++n) {
        p.cursor %= p.textures.size();
        Resource* r = p.textures[p.cursor++];
        if (r->probe_seen_data_) {
            continue;
        }
        const uint64_t len = std::min<uint64_t>(kProbeBytes, std::max<uint64_t>(1, resource_size_estimate(r->desc_)));
        // GPU-window addresses read their bytes from the CPU window (Device::register_gpu_alias).
        const uint64_t cpu = r->device_->cpu_address_for_gpu(r->gpu_va_);
        const auto* bytes = static_cast<const uint8_t*>(com::guest_to_host(cpu, static_cast<size_t>(len)));
        if (bytes == nullptr || all_zero_bytes(bytes, static_cast<size_t>(len))) {
            continue;
        }
        r->probe_seen_data_ = true;
        ++p.with_data;
        if (p.with_data <= 64) {  // the log names resource textures; view-only ones are counted only
            const GuestResourceDesc& d = r->desc_;
            char line[256];
            std::snprintf(line, sizeof(line),
                          "d3d12_xs: texture data written directly: format %u, %" PRIu64 "x%ux%u, %u mip(s), "
                          "layout 0x%x, flags 0x%x, dimension %u (texture %u of %zu registered)",
                          d.Format, d.Width, d.Height, static_cast<unsigned>(d.DepthOrArraySize), r->mips_,
                          d.Layout, d.Flags, d.Dimension, static_cast<unsigned>(p.with_data), p.textures.size());
            os::log(line);
        }
    }
    for (size_t n = 0; n < budget && !p.views.empty(); ++n) {
        p.view_cursor %= p.views.size();
        ViewProbe& v = p.views[p.view_cursor++];
        if (v.seen_data) {
            continue;
        }
        const auto* bytes = static_cast<const uint8_t*>(com::guest_to_host(v.cpu, static_cast<size_t>(v.len)));
        if (bytes == nullptr || all_zero_bytes(bytes, static_cast<size_t>(v.len))) {
            continue;
        }
        v.seen_data = true;
        ++p.views_with_data;
    }
    TextureProbeTotals t;
    t.views_registered = p.views.size();
    t.views_with_data = p.views_with_data;
    t.registered = p.textures.size() + t.views_registered;
    t.with_data = p.with_data + t.views_with_data;
    return t;
}

void register_view_texture_probe(uint64_t cpu_address, uint64_t size) {
    if (cpu_address == 0) {
        return;
    }
    TextureProbe& p = texture_probe();
    std::lock_guard<std::mutex> lock(p.mutex);
    p.views.push_back(ViewProbe{cpu_address, std::min<uint64_t>(64u << 10, std::max<uint64_t>(1, size)), false});
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
    info.gpu_va = descriptor_va_;
    device_->state_bridge().register_texture(id_, info);
    back_buffer_.store(true);
}

bool Resource::shadow_same(uint64_t off, const uint8_t* live, uint64_t n) const {
    const uint8_t* chunk = shadow_chunks_[static_cast<size_t>(off / kShadowChunk)].get();
    if (chunk == nullptr) {
        return all_zero_bytes(live, static_cast<size_t>(n));
    }
    return same_bytes(chunk + off % kShadowChunk, live, static_cast<size_t>(n));
}

void Resource::shadow_store(uint64_t off, const uint8_t* src, uint64_t n) {
    while (n != 0) {
        const size_t index = static_cast<size_t>(off / kShadowChunk);
        const uint64_t in_chunk = off % kShadowChunk;
        const uint64_t take = std::min(n, kShadowChunk - in_chunk);
        std::unique_ptr<uint8_t[]>& chunk = shadow_chunks_[index];
        if (chunk == nullptr && !all_zero_bytes(src, static_cast<size_t>(take))) {
            const uint64_t chunk_bytes = std::min(kShadowChunk, shadow_size_ - index * kShadowChunk);
            chunk = std::make_unique<uint8_t[]>(static_cast<size_t>(chunk_bytes));  // zeroed
            shadow_resident_ += chunk_bytes;
            device_->note_shadow_bytes(0, static_cast<int64_t>(chunk_bytes));
        }
        if (chunk != nullptr) {
            std::memcpy(chunk.get() + in_chunk, src, static_cast<size_t>(take));
        }
        off += take;
        src += take;
        n -= take;
    }
}

void Resource::flush_shadow(CommandEncoder& encoder, os::guest_window::MarkPass* marks) {
    if (cpu_va_ == 0 || shadow_size_ == 0) {
        return;
    }
    const uint64_t size = shadow_size_;
    const auto* live = static_cast<const uint8_t*>(com::guest_to_host(cpu_va_, static_cast<size_t>(size)));
    std::vector<uint8_t> copy;
    if (live == nullptr) {
        // Not mapped in one piece (e.g. an alias-window buffer whose pages are mapped one by one):
        // unreadable pages keep their shadow contents, i.e. count as unchanged.
        copy.assign(static_cast<size_t>(size), 0);
        for (size_t i = 0; i < shadow_chunks_.size(); ++i) {
            if (shadow_chunks_[i] != nullptr) {
                const uint64_t off = i * kShadowChunk;
                std::memcpy(copy.data() + off, shadow_chunks_[i].get(),
                            static_cast<size_t>(std::min(kShadowChunk, size - off)));
            }
        }
        if (!com::read_guest(cpu_va_, copy.data(), copy.size())) {
            for (uint64_t page = 0; page < size; page += kShadowPage) {
                const uint64_t n = std::min(kShadowPage, size - page);
                com::read_guest(cpu_va_ + page, copy.data() + page, static_cast<size_t>(n));
            }
        }
        live = copy.data();
    }
    // Written pages since the previous flush (where the arena tracks writes): only those can differ
    // from the shadow. The tracking is reset before the comparison, so a write racing with it is
    // reported again at the next flush.
    std::vector<uint64_t> written;
    const bool tracked = copy.empty() && os::take_written_pages(live, size, &write_cursor_, &written);
    const uint64_t skew = reinterpret_cast<uintptr_t>(live) & (os::kWrittenPageSize - 1);
    std::vector<bool> candidate;
    if (tracked && write_tracked_) {
        candidate.assign(static_cast<size_t>((size + kShadowPage - 1) / kShadowPage), false);
        for (uint64_t off : written) {
            const uint64_t lo = off > skew ? off - skew : 0;
            const uint64_t hi = std::min<uint64_t>(size, off + os::kWrittenPageSize - skew);
            for (uint64_t page = lo / kShadowPage * kShadowPage; page < hi; page += kShadowPage) {
                candidate[static_cast<size_t>(page / kShadowPage)] = true;
            }
        }
    }
    write_tracked_ = tracked;
    // O37: the translated window keeps write marks per 64 KiB slot instead (the browser has no OS
    // write watch): the same protocol, with the slot marks taken (cleared) before the comparison
    // and the first flush after the marks became usable comparing everything.
    bool window_candidates = false;
    if (!tracked && os::guest_window::translated(cpu_va_)) {
        std::vector<uint64_t> slots;
        const bool marked = marks != nullptr ? marks->take(cpu_va_, static_cast<size_t>(size), &slots)
                                             : os::guest_window::take_written(cpu_va_, static_cast<size_t>(size), &slots);
        if (marked && window_marked_) {
            candidate.assign(static_cast<size_t>((size + kShadowPage - 1) / kShadowPage), false);
            for (uint64_t off : slots) {
                const uint64_t slot_end = ((cpu_va_ + off) | (os::guest_window::kSlotBytes - 1)) + 1 - cpu_va_;
                const uint64_t hi = std::min<uint64_t>(size, slot_end);
                for (uint64_t page = off / kShadowPage * kShadowPage; page < hi; page += kShadowPage) {
                    candidate[static_cast<size_t>(page / kShadowPage)] = true;
                }
            }
            window_candidates = true;
        }
        window_marked_ = marked;
    }
    // Verification: compare the unmarked pages too and count changes found there (a writer that
    // did not mark); such a page is uploaded all the same.
    const bool verify = window_candidates && os::guest_window::write_marks_verified();
    uint64_t run_start = 0;
    bool in_run = false;
    uint64_t uploaded = 0;
    auto emit = [&](uint64_t begin, uint64_t end) {
        va_watch::touch("upload (CPU-visible buffer flush)", gpu_va_ + begin, end - begin,
                        "buffer id " + std::to_string(id_));
        for (uint64_t off = begin; off < end; off += kMaxWriteChunk) {
            const uint64_t n = std::min(kMaxWriteChunk, end - off);
            count_work(Work::BufferUploads);
        encoder.write_buffer(id_, static_cast<uint32_t>(off), live + off, static_cast<uint32_t>(n));
            shadow_store(off, live + off, n);
            uploaded += n;
        }
    };
    if (!candidate.empty() && !verify) {
        // O36: only the listed pages (a buffer of 100k pages with a handful written costs a
        // handful of comparisons, not a walk over every page).
        uint64_t run_end = 0;
        uint64_t compared = 0;
        for (auto it = std::find(candidate.begin(), candidate.end(), true); it != candidate.end();
             it = std::find(it + 1, candidate.end(), true)) {
            const uint64_t page = static_cast<uint64_t>(it - candidate.begin()) * kShadowPage;
            const uint64_t n = std::min(kShadowPage, size - page);
            compared += n;
            if (shadow_same(page, live + page, n)) {
                continue;
            }
            if (in_run && run_end == page) {
                run_end = page + n;
                continue;
            }
            if (in_run) {
                emit(run_start, run_end);
            }
            run_start = page;
            run_end = page + n;
            in_run = true;
        }
        if (in_run) {
            emit(run_start, run_end);
        }
        device_->note_shadow_compared(compared);
        if (uploaded != 0) {
            device_->note_upload(uploaded);
        }
        return;
    }
    // Every page is compared here (no candidates, or verification): the bulk of the work when the
    // buffer has no write tracking (R40: split across helper threads, compare_pages).
    std::vector<uint8_t> differs;
    compare_pages(static_cast<size_t>((size + kShadowPage - 1) / kShadowPage), size,
                  [&](size_t i) {
                      const uint64_t page = static_cast<uint64_t>(i) * kShadowPage;
                      return shadow_same(page, live + page, std::min(kShadowPage, size - page));
                  },
                  &differs);
    device_->note_shadow_compared(size);
    for (uint64_t page = 0; page < size; page += kShadowPage) {
        const bool listed = candidate.empty() || candidate[static_cast<size_t>(page / kShadowPage)];
        const bool changed = differs[static_cast<size_t>(page / kShadowPage)] != 0;
        if (changed && !listed && !os::guest_window::write_marked(cpu_va_ + page)) {
            // (A slot marked by now was written while this flush compared: not missed, and its
            // mark reports it again.)
            os::guest_window::note_verify_miss(cpu_va_ + page);
        }
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
    // Textures: on the console a texture's memory is ordinary CPU-addressable memory, and Map of
    // subresource 0 returns its base. Titles add the subresource offsets they take from the XG
    // layout themselves and write texels in that layout; the dirty-tracked texture-memory upload
    // (sync_bound_texture_memory) carries those writes to the GPU. Other subresources are refused
    // and reported rather than guessed.
    if (!is_buffer() && cpu_va_ != 0 && subresource != 0) {
        device_->note_bad_argument("ID3D12Resource::Map of texture subresource " + std::to_string(subresource) +
                                   " (id " + std::to_string(id_) + "): only subresource 0 is mapped");
        if (data) {
            data.write(0);
        }
        return kEInvalidArg;
    }
    if (cpu_va_ == 0 || subresource != 0) {
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
    if (device_->upload_needs_conversion(desc_.Format)) {
        device_->note_unsupported_copy("WriteToSubresource: DXGI format " + std::to_string(desc_.Format) +
                                       " needs CPU conversion for its WebGPU format (not implemented); skipped");
        return kSOk;
    }
    va_watch_resource("upload (WriteToSubresource)", this, "subresource " + std::to_string(sub));
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
    count_work(Work::TextureUploads);
    if (desc_.Format == static_cast<uint32_t>(pso::DxgiFormat::A8_UNORM)) {
        std::vector<uint8_t> wide;
        uint32_t pitch = 0;
        if (!widen_a8_rows(bytes.data(), bytes.size(), bw, rows, bd, row_pitch, depth_pitch, &wide, &pitch)) {
            device_->note_bad_argument("WriteToSubresource: A8 source smaller than its box");
            return kEInvalidArg;
        }
        enc.write_texture(id_, mip, box.left, box.top, is3d ? box.front : layer, bw, bh, bd, pitch, rows, wide.data(),
                          static_cast<uint32_t>(wide.size()));
        device_->submit(enc);
        device_->note_upload(wide.size());
        device_->note_texture_upload();
        return kSOk;
    }
    enc.write_texture(id_, mip, box.left, box.top, is3d ? box.front : layer, bw, bh, bd, row_pitch,
                      row_pitch != 0 ? depth_pitch / row_pitch : rows, bytes.data(), static_cast<uint32_t>(len));
    device_->submit(enc);
    device_->note_upload(len);
    device_->note_texture_upload();
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
// Xbox: a texture's layout field may also hold 0x100 | an XG swizzle mode. The executable builds
// 0x100, 0x102, 0x10A and 0x100 | a per-texture mode byte itself, and XGSuggestSwizzleMode writes
// into that field (kXgDefaultSwizzleMode). The guest layer keeps no CPU image of textures, so the
// mode does not change what is created; it is accepted, not interpreted.
bool xbox_swizzle_layout(uint32_t layout) { return (layout & ~0xFFu) == 0x100u; }

bool validate_desc(Device& dev, const GuestResourceDesc& d) {
    const bool layout_ok =
        d.Layout <= 3 || (d.Dimension != ResourceDimension::Buffer && xbox_swizzle_layout(d.Layout));
    if (d.Dimension < ResourceDimension::Buffer || d.Dimension > ResourceDimension::Texture3D || d.Width == 0 ||
        !layout_ok) {
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

com::Ref<Resource> Device::create_internal_buffer(uint64_t size) {
    GuestResourceDesc desc{};
    desc.Dimension = ResourceDimension::Buffer;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleCount = 1;
    desc.Layout = 1;  // ROW_MAJOR
    Resource::Placement pl;
    pl.heap.Type = HeapType::Default;
    auto resource = com::create_object<Resource>(com::implements(kID3D12Resource), device_ref(this), desc, std::move(pl),
                                                 uint32_t{0});
    if (!resource || !resource->initialize()) {
        return {};
    }
    return resource;
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

bool parse_srv_desc(const GuestShaderResourceViewDesc& g, d3d::ViewDesc* out) {
    d3d::ViewDesc vd;
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
        return false;
    }
    *out = vd;
    return true;
}

bool parse_uav_desc(const GuestUnorderedAccessViewDesc& g, d3d::ViewDesc* out) {
    d3d::ViewDesc vd;
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
        return false;
    }
    *out = vd;
    return true;
}

void Device::CreateShaderResourceView(InterfaceArg<IResourceHost> resource, uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestShaderResourceViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateShaderResourceView: unreadable descriptor");
            return;
        }
        if (!parse_srv_desc(g, &vd)) {
            note_bad_argument("CreateShaderResourceView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    if (va_watch::active()) {
        va_watch_resource("descriptor SRV", static_cast<const Resource*>(resource.get()), "CreateShaderResourceView at handle " + hex(dest), dest);
    }
    const d3d::Status s = heaps_.create_shader_resource_view(resource_id(resource), desc, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateShaderResourceView: ") + d3d::status_name(s));
        return;
    }
    store_descriptor_words(dest);
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
        if (!parse_uav_desc(g, &vd)) {
            note_bad_argument("CreateUnorderedAccessView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    if (va_watch::active()) {
        va_watch_resource("descriptor UAV", static_cast<const Resource*>(resource.get()), "CreateUnorderedAccessView at handle " + hex(dest), dest);
    }
    const d3d::Status s = heaps_.create_unordered_access_view(resource_id(resource), resource_id(counter), desc,
                                                              d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateUnorderedAccessView: ") + d3d::status_name(s));
        return;
    }
    store_descriptor_words(dest);
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
    if (va_watch::active()) {
        va_watch_resource("descriptor RTV", static_cast<const Resource*>(resource.get()), "CreateRenderTargetView at handle " + hex(dest));
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

// The shader kind a DXIL container declares (DXIL part, program header: kind = version >> 16:
// 0 pixel, 1 vertex, 2 geometry, 3 hull, 4 domain, 5 compute, 13 mesh, 14 amplification, ...),
// or -1 when the bytes are not a DXBC container with a DXIL part.
int dxil_shader_kind(const uint8_t* p, size_t n) {
    auto u32 = [&](size_t at) {
        uint32_t v = 0;
        std::memcpy(&v, p + at, 4);
        return v;
    };
    if (n < 32 || u32(0) != 0x43425844u) {  // "DXBC"
        return -1;
    }
    const uint32_t parts = u32(28);
    if (parts > 64 || 32 + uint64_t{parts} * 4 > n) {
        return -1;
    }
    for (uint32_t i = 0; i < parts; ++i) {
        const uint32_t off = u32(32 + i * 4);
        if (uint64_t{off} + 12 > n) {
            return -1;
        }
        const uint32_t fourcc = u32(off);
        if (fourcc == 0x4C495844u) {  // "DXIL"
            return static_cast<int>(u32(off + 8) >> 16);
        }
    }
    return -1;
}

// The 16-byte digest of a DXBC container's HASH part (after its u32 flags); false without one.
bool dxil_hash_digest(const uint8_t* p, size_t n, Device::DxilDigest* out) {
    auto u32 = [&](size_t at) {
        uint32_t v = 0;
        std::memcpy(&v, p + at, 4);
        return v;
    };
    if (n < 32 || u32(0) != 0x43425844u) {  // "DXBC"
        return false;
    }
    const uint32_t parts = u32(28);
    if (parts > 64 || 32 + uint64_t{parts} * 4 > n) {
        return false;
    }
    for (uint32_t i = 0; i < parts; ++i) {
        const uint32_t off = u32(32 + i * 4);
        if (uint64_t{off} + 8 > n) {
            return false;
        }
        if (u32(off) == 0x48534148u && u32(off + 4) >= 20 && uint64_t{off} + 8 + 20 <= n) {  // "HASH"
            std::memcpy(out->data(), p + off + 12, 16);
            return true;
        }
    }
    return false;
}

// Reads a shader's bytecode and returns its identity (0 = no shader). False on unreadable input.
// `digest` (optional) receives the container's HASH digest; it stays all-zero without one.
bool read_shader(Device& dev, const GuestShaderBytecode& code, const char* stage, uint64_t* identity,
                 Device::DxilDigest* digest = nullptr) {
    *identity = 0;
    if (digest != nullptr) {
        digest->fill(0);
    }
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
    dev.note_shader_size(*identity, bytes.size(), dxil_shader_kind(bytes.data(), bytes.size()));
    Device::DxilDigest d{};
    const bool has_digest = dxil_hash_digest(bytes.data(), bytes.size(), &d);
    if (digest != nullptr) {
        *digest = has_digest ? d : Device::DxilDigest{};
    }
    if (PipelineDump* dump = dev.pipeline_dump()) {
        dump->dxil(*identity, has_digest ? &d : nullptr, dxil_shader_kind(bytes.data(), bytes.size()), bytes.data(),
                   bytes.size());
    }
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
        rect_list = other.rect_list;
        native_vs = other.native_vs;
        pack_vs_key = other.pack_vs_key;
        pack_ps_key = other.pack_ps_key;
        vs_fetches = other.vs_fetches;
        reflected = other.reflected;
        vs_digest = other.vs_digest;
        ps_digest = other.ps_digest;
        native_ps = other.native_ps;
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
    desc.vs_input_fetches = reflected ? vs_fetches.data() : nullptr;
    desc.vs_input_fetch_count = reflected ? static_cast<uint32_t>(vs_fetches.size()) : 0u;
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
    if (!read_shader(*this, g.VS, "VS", &d.vs, &src.vs_digest) || !read_shader(*this, g.PS, "PS", &d.ps, &src.ps_digest) ||
        !read_shader(*this, g.DS, "DS", &d.ds) || !read_shader(*this, g.HS, "HS", &d.hs) ||
        !read_shader(*this, g.GS, "GS", &d.gs)) {
        return kEInvalidArg;
    }
    if (d.vs == 0) {
        note_bad_argument("CreateGraphicsPipelineState: no vertex shader");
        return kEInvalidArg;
    }
    // Xbox rectangle topology type: kept as triangles plus a flag (d3d12_xs_xbox.h).
    src.rect_list = g.PrimitiveTopologyType == xbox::kTopologyTypeRect;
    const uint32_t topology_type = src.rect_list ? 3u : g.PrimitiveTopologyType;
    if (g.NumRenderTargets > 8 || g.InputLayout.NumElements > kMaxInputElements || topology_type > 4) {
        note_bad_argument("CreateGraphicsPipelineState: out-of-range counts or topology type (render targets " +
                          std::to_string(g.NumRenderTargets) + ", input elements " +
                          std::to_string(g.InputLayout.NumElements) + ", topology type " +
                          std::to_string(g.PrimitiveTopologyType) + ")");
        return kEInvalidArg;
    }
    if (src.rect_list) {
        count(xbox::Counter::RectListPipelineStates);
    }
    // Join each DXIL stage to the console-native stage compiled from the same shader (only stages
    // indexed by slot 57 with refuse_native_pipelines; d3d12_xs_xbox.h).
    if (native_index_size() != 0) {
        static const DxilDigest kZero{};
        for (auto [digest, record] : {std::pair{&src.vs_digest, &src.native_vs}, std::pair{&src.ps_digest, &src.native_ps}}) {
            if (*digest == kZero) {
                continue;
            }
            count(find_native_stage(*digest, record) ? xbox::Counter::DxilStagesJoined
                                                     : xbox::Counter::DxilStagesNotJoined);
        }
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
    d.PrimitiveTopologyType = static_cast<pso::D3D12PrimitiveTopologyType>(topology_type);
    d.NumRenderTargets = g.NumRenderTargets;
    for (int i = 0; i < 8; ++i) {
        d.RTVFormats[i] = static_cast<pso::DxgiFormat>(g.RTVFormats[i]);
    }
    d.DSVFormat = static_cast<pso::DxgiFormat>(g.DSVFormat);
    d.SampleDesc.Count = g.SampleCount;
    d.SampleDesc.Quality = g.SampleQuality;
    return create_graphics_pso_from(std::move(src), riid, out);
}

// Pipeline layout from a root signature (convention in Device::resolve_with_pack).
bridge::PipelineLayoutDesc layout_from_root_signature(const d3d::RootSignatureDesc& rs) {
    // WebGPU core per-stage binding limits per resource class: keep every resource class
    // represented and every emitted BindGroupLayout valid even when a root signature declares
    // oversized descriptor tables (e.g. 240 SRVs / 19 UAVs / 17 CBVs / 24 static samplers).
    constexpr uint32_t kMaxCbvBindingsPerStage = 12u;
    constexpr uint32_t kMaxSrvBindingsPerStage = 16u;
    constexpr uint32_t kMaxUavBindingsPerStage = 8u;
    constexpr uint32_t kMaxSamplerBindingsPerStage = 16u;

    bridge::GroupLayoutDesc group;
    uint32_t binding = 0;
    uint32_t cbv_count = 0;
    uint32_t srv_count = 0;
    uint32_t uav_count = 0;
    uint32_t sampler_count = 0;
    auto add = [&](GpuBindingLayoutType type, uint8_t binding_type, uint8_t view_dimension, d3d::DescriptorRangeType cls,
                   uint32_t reg, uint32_t space, bool root_constants) {
        bridge::LayoutBinding b;
        b.entry.binding = binding;
        b.entry.visibility = GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment;
        b.entry.type = static_cast<uint8_t>(type);
        b.entry.binding_type = binding_type;
        b.entry.view_dimension = view_dimension;
        b.join.binding = binding;
        b.join.register_class = cls;
        b.join.shader_register = reg;
        b.join.register_space = space;
        b.join.root_constants = root_constants;
        group.bindings.push_back(b);
        ++binding;
    };
    auto add_class = [&](d3d::DescriptorRangeType cls, uint32_t reg, uint32_t space, bool root_constants) {
        switch (cls) {
            case d3d::DescriptorRangeType::Cbv:
                if (cbv_count >= kMaxCbvBindingsPerStage) {
                    return;
                }
                ++cbv_count;
                add(GpuBindingLayoutType::Buffer, static_cast<uint8_t>(GpuBufferBindingType::Uniform), 0, cls, reg, space,
                    root_constants);
                break;
            case d3d::DescriptorRangeType::Srv:
                if (srv_count >= kMaxSrvBindingsPerStage) {
                    return;
                }
                ++srv_count;
                add(GpuBindingLayoutType::Texture, static_cast<uint8_t>(GpuTextureSampleType::Float),
                    static_cast<uint8_t>(GpuTextureViewDimension::Dim2D), cls, reg, space, false);
                break;
            case d3d::DescriptorRangeType::Uav:
                if (uav_count >= kMaxUavBindingsPerStage) {
                    return;
                }
                ++uav_count;
                add(GpuBindingLayoutType::Buffer, static_cast<uint8_t>(GpuBufferBindingType::Storage), 0, cls, reg, space,
                    false);
                break;
            case d3d::DescriptorRangeType::Sampler:
                if (sampler_count >= kMaxSamplerBindingsPerStage) {
                    return;
                }
                ++sampler_count;
                add(GpuBindingLayoutType::Sampler, static_cast<uint8_t>(GpuSamplerBindingType::Filtering), 0, cls, reg,
                    space, false);
                break;
        }
    };
    for (const d3d::RootParameterDesc& p : rs.parameters) {
        switch (p.parameter_type) {
            case d3d::RootParameterType::DescriptorTable:
                for (const d3d::DescriptorRangeDesc& r : p.ranges) {
                    const uint32_t n = r.num_descriptors == d3d::kUnboundedDescriptorCount ? 1 : r.num_descriptors;
                    for (uint32_t i = 0; i < n; ++i) {
                        add_class(r.range_type, r.base_shader_register + i, r.register_space, false);
                    }
                }
                break;
            case d3d::RootParameterType::Constants32Bit:
                add_class(d3d::DescriptorRangeType::Cbv, p.shader_register, p.register_space, true);
                break;
            case d3d::RootParameterType::Cbv:
                add_class(d3d::DescriptorRangeType::Cbv, p.shader_register, p.register_space, false);
                break;
            case d3d::RootParameterType::Srv:
                add_class(d3d::DescriptorRangeType::Srv, p.shader_register, p.register_space, false);
                break;
            case d3d::RootParameterType::Uav:
                add_class(d3d::DescriptorRangeType::Uav, p.shader_register, p.register_space, false);
                break;
        }
    }
    for (const d3d::StaticSamplerDesc& s : rs.static_samplers) {
        add_class(d3d::DescriptorRangeType::Sampler, s.shader_register, s.register_space, false);
    }
    bridge::PipelineLayoutDesc layout;
    layout.groups.push_back(std::move(group));
    return layout;
}

size_t Device::reflection_join_failures() const {
    std::lock_guard<std::mutex> lock(pack_mutex_);
    return reflection_failures_.size();
}

bool Device::resolve_with_pack(GraphicsPsoSource& src, uint64_t* vs_key, uint64_t* ps_key) {
    *vs_key = *ps_key = 0;
    src.reflected = false;
    src.vs_fetches.clear();
    src.fix_pointers();
    const ShaderPackIndex* index = options_.shader_pack.get();
    const pso::D3D12GraphicsPsoDesc& d = src.desc;
    if (index == nullptr || d.vs == 0 || d.ds != 0 || d.hs != 0 || d.gs != 0) {
        return false;
    }
    auto key_for = [&](uint64_t dxil, const NativeStageRecord& native, PackStage stage, uint64_t* key) {
        for (uint64_t k : {native.identity != 0 ? native.code_identity : 0, dxil}) {
            auto it = k != 0 ? index->find(k) : index->end();
            if (it != index->end() && it->second == stage) {
                *key = k;
                return true;
            }
        }
        return false;
    };
    if (!key_for(d.vs, src.native_vs, PackStage::Vertex, vs_key) ||
        (d.ps != 0 && !key_for(d.ps, src.native_ps, PackStage::Fragment, ps_key))) {
        *vs_key = *ps_key = 0;
        return false;
    }
    // Reflection join (G62): when every stage has reflection, the vertex attributes and the
    // pipeline layout come from it; a pipeline that cannot be joined is not resolved.
    const StageReflectionIndex* refl = options_.stage_reflection.get();
    const StageReflection* vs_refl = nullptr;
    const StageReflection* ps_refl = nullptr;
    if (refl != nullptr) {
        auto vit = refl->find(*vs_key);
        auto pit = d.ps != 0 ? refl->find(*ps_key) : refl->end();
        vs_refl = vit != refl->end() ? &vit->second : nullptr;
        ps_refl = pit != refl->end() ? &pit->second : nullptr;
        if (d.ps != 0 && (vs_refl == nullptr) != (ps_refl == nullptr)) {
            vs_refl = ps_refl = nullptr;  // only one stage reflected: keep the conventions for both
        }
    }
    bridge::PipelineLayoutDesc reflected_layout;
    if (vs_refl != nullptr) {
        std::string report;
        bool ok = static_cast<bool>(src.root_signature);
        if (!ok) {
            report = "no root signature\n";
        }
        std::vector<pso::VertexInputFetch> fetches;
        if (ok) {
            ok = join_vertex_inputs(*vs_refl, src.input.data(), static_cast<uint32_t>(src.input.size()), &fetches,
                                    &report);
            std::vector<ReflectedStageRef> stages{{GpuShaderStageBits::Vertex, vs_refl}};
            if (ps_refl != nullptr) {
                stages.push_back({GpuShaderStageBits::Fragment, ps_refl});
            }
            JoinStats join_stats;
            ok = join_pipeline_layout(src.root_signature->signature().desc(), /*compute=*/false, stages,
                                      &reflected_layout, &report, &join_stats) && ok;
            if (ok) {
                count_work(Work::ReflectionTableSlotsOutsideRs, join_stats.table_slots_outside_rs);
            }
        }
        if (!ok) {
            const uint64_t rs_hash = src.root_signature ? src.root_signature->signature().hash() : 0;
            std::lock_guard<std::mutex> lock(pack_mutex_);
            if (reflection_failures_.insert({*vs_key, *ps_key, rs_hash}).second) {
                char head[96];
                std::snprintf(head, sizeof(head), "d3d12_xs: reflection join failed (vertex 0x%016" PRIx64
                              ", pixel 0x%016" PRIx64 "):\n", *vs_key, *ps_key);
                os::log(std::string(head) + report);
            }
            *vs_key = *ps_key = 0;
            return false;
        }
        src.vs_fetches = std::move(fetches);
        src.reflected = true;
        src.fix_pointers();
    }
    std::lock_guard<std::mutex> lock(pack_mutex_);
    auto module_for = [&](uint64_t key, GpuShaderStage stage) {
        auto it = pack_modules_.find(key);
        if (it != pack_modules_.end()) {
            return it->second;
        }
        const uint32_t id = allocate_id();
        CommandEncoder enc;
        enc.create_shader_module(id, stage, pack_key_code(key));
        submit(enc);
        pack_modules_.emplace(key, id);
        return id;
    };
    register_shader_module(d.vs, module_for(*vs_key, GpuShaderStage::Vertex));
    if (d.ps != 0) {
        register_shader_module(d.ps, module_for(*ps_key, GpuShaderStage::Fragment));
    }
    if (src.root_signature && pack_layouts_.insert({d.vs, d.ps}).second) {
        bridge_->register_pipeline_layout(d.vs, d.ps,
                                          src.reflected ? std::move(reflected_layout)
                                                        : layout_from_root_signature(src.root_signature->signature().desc()));
    }
    return true;
}

HResult Device::create_graphics_pso_from(GraphicsPsoSource source, const Guid& riid, OutPtr out) {
    auto src = std::make_unique<GraphicsPsoSource>(source);  // copy: fixes the internal pointers
    const com::Ref<RootSignature>& rs = src->root_signature;
    pso::D3D12GraphicsPsoDesc& d = src->desc;
    d.root_signature_id = rs ? static_cast<uint32_t>(rs->signature().hash()) : 0u;
    resolve_with_pack(*src, &src->pack_vs_key, &src->pack_ps_key);
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this), rs);
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->graphics_ = bridge_->create_graphics_pipeline_state(d, rs ? &rs->signature() : nullptr);
    state->identities_ = {d.vs, d.ps};
    state->resolved_ = (state->graphics_->diagnostics() & bridge::BridgeDiag::UnknownShaderModule) == 0;
    if (!state->resolved_) {
        if (d.vs != 0 && !shader_known(d.vs)) {
            note_unresolved_shader(d.vs, ShaderStage::Vertex);
        }
        if (d.ps != 0 && !shader_known(d.ps)) {
            note_unresolved_shader(d.ps, ShaderStage::Pixel);
        }
    }
    if (dump_) {
        state->dump_index_ = dump_->pipeline(*src, src->vs_digest, src->ps_digest, rs.get(),
                                             rs ? &rs->signature().desc() : nullptr);
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

bool Device::read_native_stage(const xbox::NativeStage& s, bool index, NativeStageRecord* out, bool pixel,
                               bool layout_b) {
    *out = NativeStageRecord{};
    if (!s.present()) {
        return true;
    }
    std::vector<uint8_t> code(static_cast<size_t>(s.code.size));
    std::vector<uint8_t> meta(static_cast<size_t>(s.metadata.size));
    if (!com::read_guest(s.code.address, code.data(), code.size()) ||
        !com::read_guest(s.metadata.address, meta.data(), meta.size())) {
        note_bad_argument("CreatePipelineState (native): unreadable stage at " + hex(s.code.address));
        return false;
    }
    out->identity = native_stage_identity(code.data(), code.size(), meta.data(), meta.size());
    out->code_identity = shader_identity(code.data(), code.size());
    out->code_bytes = static_cast<uint32_t>(code.size());
    out->metadata_bytes = static_cast<uint32_t>(meta.size());
    note_shader_size(out->identity, code.size());
    if (dump_) {
        dump_->native_stage(out->code_identity, out->identity, pixel, layout_b, code.data(), code.size(), meta.data(),
                            meta.size());
    }
    if (!index) {
        return true;
    }
    std::lock_guard<std::mutex> lock(shader_mutex_);
    if (!indexed_native_.insert(out->identity).second) {
        return true;
    }
    static const DxilDigest kZero{};
    for (size_t o = xbox::kNativeDigestFirst; o + 16 <= meta.size(); o += xbox::kNativeDigestStride) {
        DxilDigest d;
        std::memcpy(d.data(), meta.data() + o, 16);
        if (d != kZero) {
            native_by_digest_.emplace(d, *out);
        }
    }
    if (meta.size() >= 28) {
        DxilDigest d;
        std::memcpy(d.data(), meta.data() + meta.size() - 28, 16);
        if (d != kZero) {
            native_by_digest_.emplace(d, *out);
        }
    }
    return true;
}

bool Device::find_native_stage(const DxilDigest& digest, NativeStageRecord* out) const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    auto it = native_by_digest_.find(digest);
    if (it == native_by_digest_.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

size_t Device::native_index_size() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return indexed_native_.size();
}

HResult Device::create_native_pso(const xbox::NativePipeline& pipeline, const Guid& riid, OutPtr out) {
    com::Ref<RootSignature> rs = root_signature_ref(pipeline.root_signature);
    if (pipeline.root_signature != 0 && !rs) {
        note_bad_argument("CreatePipelineState (native): invalid root signature " + hex(pipeline.root_signature));
        return kEInvalidArg;
    }
    auto native = std::make_unique<PipelineState::Native>();
    native->layout = pipeline;
    native->header.resize(static_cast<size_t>(pipeline.header.size));
    if (!com::read_guest(pipeline.header.address, native->header.data(), native->header.size())) {
        note_bad_argument("CreatePipelineState (native): unreadable header record at " + hex(pipeline.header.address));
        return kEInvalidArg;
    }
    NativeStageRecord vs, ps;
    if (!read_native_stage(pipeline.vertex, false, &vs, false, pipeline.vertex_alt) ||
        !read_native_stage(pipeline.pixel, false, &ps, true, pipeline.vertex_alt)) {
        return kEInvalidArg;
    }
    native->vertex_identity = vs.identity;
    native->pixel_identity = ps.identity;
    native->vertex_code_identity = vs.code_identity;
    native->pixel_code_identity = ps.code_identity;
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this), rs);
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->identities_ = {native->vertex_identity, native->pixel_identity};
    state->unsupported_ = "console-native pipeline: GPU machine code and a register-state record, not translated";
    note_native_stages(native->vertex_identity, native->pixel_identity);
    state->native_ = std::move(native);
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
    DxilDigest cs_digest{};
    if (!read_shader(*this, g.CS, "CS", &d.cs, &cs_digest)) {
        return kEInvalidArg;
    }
    if (d.cs == 0) {
        note_bad_argument("CreateComputePipelineState: no compute shader");
        return kEInvalidArg;
    }
    NativeStageRecord native_cs{};
    static const DxilDigest kZero{};
    if (native_index_size() != 0 && cs_digest != kZero) {
        count(find_native_stage(cs_digest, &native_cs) ? xbox::Counter::DxilStagesJoined
                                                       : xbox::Counter::DxilStagesNotJoined);
    }
    if (const ShaderPackIndex* pack = options_.shader_pack.get(); pack != nullptr) {
        uint64_t pack_key = 0;
        for (uint64_t k : {native_cs.identity != 0 ? native_cs.code_identity : uint64_t{0}, d.cs}) {
            auto it = k != 0 ? pack->find(k) : pack->end();
            if (it != pack->end() && it->second == PackStage::Compute) {
                pack_key = k;
                break;
            }
        }
        // Reflection join (G62) for the compute stage.
        bridge::PipelineLayoutDesc reflected_layout;
        bool reflected = false;
        if (const StageReflectionIndex* refl = options_.stage_reflection.get(); refl != nullptr && pack_key != 0) {
            if (auto rit = refl->find(pack_key); rit != refl->end()) {
                std::string report;
                JoinStats join_stats;
                const bool ok = static_cast<bool>(rs) &&
                                join_pipeline_layout(rs->signature().desc(), /*compute=*/true,
                                                     {{GpuShaderStageBits::Compute, &rit->second}}, &reflected_layout,
                                                     &report, &join_stats);
                if (ok) {
                    reflected = true;
                    count_work(Work::ReflectionTableSlotsOutsideRs, join_stats.table_slots_outside_rs);
                } else {
                    const uint64_t rs_hash = rs ? rs->signature().hash() : 0;
                    std::lock_guard<std::mutex> lock(pack_mutex_);
                    if (reflection_failures_.insert({pack_key, 0, rs_hash}).second) {
                        char head[80];
                        std::snprintf(head, sizeof(head), "d3d12_xs: reflection join failed (compute 0x%016" PRIx64 "):\n",
                                      pack_key);
                        os::log(std::string(head) + (rs ? report : std::string("no root signature\n")));
                    }
                    pack_key = 0;  // not resolved: dispatches with it are skipped
                }
            }
        }
        if (pack_key != 0) {
            std::lock_guard<std::mutex> lock(pack_mutex_);
            uint32_t module_id = 0;
            auto it = pack_modules_.find(pack_key);
            if (it != pack_modules_.end()) {
                module_id = it->second;
            } else {
                module_id = allocate_id();
                CommandEncoder enc;
                enc.create_shader_module(module_id, GpuShaderStage::Compute, pack_key_code(pack_key));
                submit(enc);
                pack_modules_.emplace(pack_key, module_id);
            }
            register_shader_module(d.cs, module_id);
            if (rs && pack_layouts_.insert({d.cs, 0}).second) {
                bridge::PipelineLayoutDesc layout =
                    reflected ? std::move(reflected_layout) : layout_from_root_signature(rs->signature().desc());
                for (bridge::GroupLayoutDesc& group : layout.groups) {
                    for (bridge::LayoutBinding& b : group.bindings) {
                        b.entry.visibility = GpuShaderStageBits::Compute;
                    }
                }
                bridge_->register_pipeline_layout(d.cs, 0, std::move(layout));
            }
        }
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
        note_unresolved_shader(d.cs, ShaderStage::Compute);
    }
    if (dump_) {
        state->dump_index_ = dump_->compute_pipeline(d.cs, cs_digest, native_cs.code_identity, native_cs.identity,
                                                     rs.get(), rs ? &rs->signature().desc() : nullptr);
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

bool va_watch_resource(const char* kind, const Resource* r, const std::string& detail, uint64_t handle) {
    if (r == nullptr || !va_watch::active()) {
        return false;
    }
    const uint64_t size = r->is_buffer() ? r->desc().Width : committed_memory_size(r->desc());
    const uint64_t va = r->is_buffer() ? r->gpu_va() : r->descriptor_va();
    if (!va_watch::touch(kind, va, size, "resource id " + std::to_string(r->id()) + ", " + detail)) {
        return false;
    }
    va_watch::watch_handle(handle, va);
    return true;
}

}  // namespace playgta6::gpu::d3d12_guest
