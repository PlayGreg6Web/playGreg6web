// Native executor milestones 0 and 1 on the host's real GPU (headless): device, offscreen swapchain,
// clear/present, buffers, textures, shaders, layouts, pipelines, draws and dispatches, with every
// result read back from the GPU. Exits 77
// (CTest: skipped) when the host has no WebGPU adapter. Built with -DPLAYGTA6_NATIVE_EXECUTOR=ON.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/native_exec/native_executor.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_native_executor.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::gpu::native_exec;

constexpr uint32_t kW = 64, kH = 32;

struct Px {
    uint8_t b, g, r, a;
};

uint32_t run(NativeExecutor& ex, const gpu::CommandEncoder& enc) { return ex.execute(enc.data(), enc.size()); }

// Every pixel of the last frame equals `want` (bgra8unorm swapchain).
bool frame_is(NativeExecutor& ex, Px want) {
    FrameImage img;
    std::string err;
    if (!ex.read_last_frame(&img, &err)) {
        std::fprintf(stderr, "read_last_frame: %s\n", err.c_str());
        return false;
    }
    if (img.width != kW || img.height != kH || img.format != "bgra8unorm") return false;
    for (size_t i = 0; i < img.pixels.size(); i += 4) {
        if (img.pixels[i] != want.b || img.pixels[i + 1] != want.g || img.pixels[i + 2] != want.r ||
            img.pixels[i + 3] != want.a) {
            std::fprintf(stderr, "pixel %zu = %u,%u,%u,%u (bgra), want %u,%u,%u,%u\n", i / 4, img.pixels[i],
                         img.pixels[i + 1], img.pixels[i + 2], img.pixels[i + 3], want.b, want.g, want.r, want.a);
            return false;
        }
    }
    return true;
}

GpuBeginRenderPassPayload swapchain_pass(GpuLoadOp load, float r, float g, float b, float a) {
    GpuBeginRenderPassPayload bp;
    std::memset(&bp, 0, sizeof bp);
    bp.color_count = 1;
    bp.color[0].texture_id = kGpuSwapchainTextureId;
    bp.color[0].load_op = static_cast<uint8_t>(load);
    bp.color[0].store_op = static_cast<uint8_t>(GpuStoreOp::Store);
    bp.color[0].clear_value[0] = r;
    bp.color[0].clear_value[1] = g;
    bp.color[0].clear_value[2] = b;
    bp.color[0].clear_value[3] = a;
    return bp;
}

void test_clear_present(NativeExecutor& ex) {
    // No ClearColor: the v1 swapchain pass clears to DEFAULT_CLEAR_COLOR (0.05, 0.07, 0.11, 1).
    gpu::CommandEncoder f0;
    f0.begin_frame();
    f0.present();
    CHECK(run(ex, f0) == 1);
    CHECK(ex.frame_count() == 1);
    CHECK(frame_is(ex, {28, 18, 13, 255}));

    gpu::CommandEncoder f1;
    f1.begin_frame();
    f1.clear_color(1.0f, 0.0f, 0.0f, 1.0f);
    f1.present();
    run(ex, f1);
    CHECK(frame_is(ex, {0, 0, 255, 255}));

    // BeginFrame resets the clear colour.
    gpu::CommandEncoder f2;
    f2.clear_color(0.0f, 1.0f, 0.0f, 1.0f);
    f2.begin_frame();
    f2.present();
    run(ex, f2);
    CHECK(frame_is(ex, {28, 18, 13, 255}));

    // An explicit pass that loads the canvas first reads a fresh (transparent black) texture, and
    // then no clearing v1 pass runs.
    gpu::CommandEncoder f3;
    f3.begin_frame();
    f3.clear_color(1.0f, 1.0f, 1.0f, 1.0f);
    auto load = swapchain_pass(GpuLoadOp::Load, 0, 0, 0, 0);
    f3.emit_packet(GpuOpcode::BeginRenderPass, &load, sizeof load);
    f3.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    f3.present();
    run(ex, f3);
    CHECK(frame_is(ex, {0, 0, 0, 0}));

    // Clear, then a second pass that loads: the clear survives.
    gpu::CommandEncoder f4;
    f4.begin_frame();
    auto clear = swapchain_pass(GpuLoadOp::Clear, 0.0f, 1.0f, 0.0f, 1.0f);
    f4.emit_packet(GpuOpcode::BeginRenderPass, &clear, sizeof clear);
    f4.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    f4.emit_packet(GpuOpcode::BeginRenderPass, &load, sizeof load);
    f4.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    f4.present();
    run(ex, f4);
    CHECK(frame_is(ex, {0, 255, 0, 255}));

    // A pass on a texture the executor does not know is dropped; the v1 clearing pass runs.
    const uint64_t missing_before = ex.stats().render_pass_missing_texture;
    gpu::CommandEncoder f5;
    f5.begin_frame();
    auto unknown = clear;
    unknown.color[0].texture_id = 12345;
    f5.emit_packet(GpuOpcode::BeginRenderPass, &unknown, sizeof unknown);
    f5.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    f5.clear_color(0.0f, 0.0f, 1.0f, 1.0f);
    f5.present();
    run(ex, f5);
    CHECK(ex.stats().render_pass_missing_texture == missing_before + 1);
    CHECK(frame_is(ex, {255, 0, 0, 255}));
    CHECK(ex.frame_count() == 6);
}

void test_buffers(NativeExecutor& ex) {
    const uint8_t data[5] = {0xa1, 0xa2, 0xa3, 0xa4, 0xa5};
    gpu::CommandEncoder enc;
    enc.create_buffer(7, GpuBufferUsage::CopySrc | GpuBufferUsage::CopyDst, 16);
    enc.write_buffer(7, 4, data, 5);   // staged: zero-padded to 8 bytes
    enc.write_buffer(7, 12, data, 3);  // staged: padded to 4, fits exactly
    run(ex, enc);
    std::vector<uint8_t> got;
    std::string err;
    CHECK(ex.read_buffer(7, 0, 16, &got, &err));
    const std::vector<uint8_t> want = {0, 0, 0, 0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0, 0, 0, 0xa1, 0xa2, 0xa3, 0};
    CHECK(got == want);

    // An unaligned offset goes to queue.writeBuffer, which WebGPU rejects: a device error, no write.
    const uint64_t errors_before = ex.stats().device_errors;
    gpu::CommandEncoder bad;
    bad.write_buffer(7, 2, data, 4);
    run(ex, bad);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().device_errors == errors_before + 1);
    CHECK(ex.read_buffer(7, 0, 16, &got, &err) && got == want);

    // Re-creating an id replaces the buffer; DestroyResource removes it.
    gpu::CommandEncoder again;
    again.create_buffer(7, GpuBufferUsage::CopySrc | GpuBufferUsage::CopyDst, 8);
    run(ex, again);
    CHECK(ex.read_buffer(7, 0, 8, &got, &err) && got == std::vector<uint8_t>(8, 0));
    gpu::CommandEncoder destroy;
    destroy.destroy_resource(7);
    run(ex, destroy);
    CHECK(!ex.read_buffer(7, 0, 8, &got, &err));
}

void test_stream_accounting(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder enc;
    enc.set_bind_group(0, 1);  // v1 SetBindGroup (needs the reflection loader): skipped and counted
    enc.nop();
    std::vector<uint8_t> bytes(enc.data(), enc.data() + enc.size());
    const GpuPacketHeader truncated{static_cast<uint16_t>(GpuOpcode::WriteBuffer), 0, 64};
    const size_t at = bytes.size();
    bytes.resize(at + 8 + 4);
    std::memcpy(bytes.data() + at, &truncated, 8);
    CHECK(ex.execute(bytes.data(), bytes.size()) == 0);
    const ExecutorStats& after = ex.stats();
    CHECK(after.unhandled[static_cast<size_t>(GpuOpcode::SetBindGroup)] ==
          before.unhandled[static_cast<size_t>(GpuOpcode::SetBindGroup)] + 1);
    CHECK(after.handled[0] == before.handled[0] + 1);
    CHECK(after.truncated_batches == before.truncated_batches + 1);
    CHECK(after.packets == before.packets + 2);
    const std::string json = ex.stats_json();
    CHECK(json.find("\"unhandled\": {\"16\": 1}") != std::string::npos);
}

// ---- milestone 1: textures, shaders, layouts, pipelines, draws, dispatches ----

template <class T>
void append(std::vector<uint8_t>& v, const T& x) {
    const auto* b = reinterpret_cast<const uint8_t*>(&x);
    v.insert(v.end(), b, b + sizeof x);
}

template <class T>
std::vector<uint8_t> bytes_of(const T& x) {
    std::vector<uint8_t> v;
    append(v, x);
    return v;
}

void emit(gpu::CommandEncoder& enc, GpuOpcode op, const std::vector<uint8_t>& payload) {
    enc.emit_packet(op, payload.data(), static_cast<uint32_t>(payload.size()));
}

GpuBindGroupLayoutEntry layout_entry(uint32_t binding, uint32_t visibility, GpuBindingLayoutType type, uint8_t subtype,
                                     uint8_t view_dim = 0) {
    GpuBindGroupLayoutEntry e;
    std::memset(&e, 0, sizeof e);
    e.binding = binding;
    e.visibility = visibility;
    e.type = static_cast<uint8_t>(type);
    e.binding_type = subtype;
    e.view_dimension = view_dim;
    return e;
}

void create_layouts(gpu::CommandEncoder& enc, uint32_t bgl_id, uint32_t pl_id,
                    const std::vector<GpuBindGroupLayoutEntry>& entries) {
    std::vector<uint8_t> bgl;
    append(bgl, GpuCreateBindGroupLayoutPayload{bgl_id, static_cast<uint32_t>(entries.size()), 0, 0, 0});
    for (const auto& e : entries) append(bgl, e);
    emit(enc, GpuOpcode::CreateBindGroupLayout, bgl);
    std::vector<uint8_t> pl;
    append(pl, GpuCreatePipelineLayoutPayload{pl_id, 1, 0, 0, 0});
    append(pl, bgl_id);
    emit(enc, GpuOpcode::CreatePipelineLayout, pl);
}

GpuBindGroupEntry2 bg_entry(uint32_t binding, GpuBindResourceType type, uint32_t resource) {
    GpuBindGroupEntry2 e;
    std::memset(&e, 0, sizeof e);
    e.binding = binding;
    e.resource_type = static_cast<uint32_t>(type);
    e.resource_id = resource;
    return e;
}

void create_bind_group(gpu::CommandEncoder& enc, uint32_t id, uint32_t layout, const std::vector<GpuBindGroupEntry2>& es) {
    std::vector<uint8_t> v;
    append(v, GpuCreateBindGroup2Payload{id, layout, static_cast<uint32_t>(es.size()), 0});
    for (const auto& e : es) append(v, e);
    emit(enc, GpuOpcode::CreateBindGroup2, v);
}

void set_bind_group2(gpu::CommandEncoder& enc, uint32_t group, uint32_t id) {
    emit(enc, GpuOpcode::SetBindGroup2, bytes_of(GpuSetBindGroup2Payload{group, id, 0, 0}));
}

GpuCreateTexture2Payload texture2(uint32_t id, uint32_t w, uint32_t h, uint32_t usage, uint8_t format) {
    GpuCreateTexture2Payload ct;
    std::memset(&ct, 0, sizeof ct);
    ct.id = id;
    ct.width = w;
    ct.height = h;
    ct.depth_or_layers = 1;
    ct.mip_levels = 1;
    ct.sample_count = 1;
    ct.usage_flags = usage;
    ct.dimension = 1;
    ct.format = format;
    return ct;
}

GpuCreateRenderPipeline2Payload pipeline_to_bgra(uint32_t id, uint32_t vs, uint32_t fs, uint32_t layout) {
    GpuCreateRenderPipeline2Payload p;
    std::memset(&p, 0, sizeof p);
    p.id = id;
    p.vs_module_id = vs;
    p.fs_module_id = fs;
    p.pipeline_layout_id = layout;
    p.key.color_target_count = 1;
    p.key.color_targets[0].format = 27;  // bgra8unorm
    p.key.color_targets[0].write_mask = 15;
    p.key.multisample.count = 1;
    p.key.multisample.mask = 0xffffffffu;
    return p;
}

const char* kTexturedTriangle = R"(
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return textureSample(tex, smp, vec2f(0.5, 0.5)); }
)";

void test_textured_draw(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    // A 2x2 rgba8unorm texture whose texels are all (255, 128, 0, 255), uploaded with a padded row pitch.
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(20, 2, 2, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst | GpuTextureUsage::CopySrc, 22)));
    std::vector<uint8_t> texels(256 + 8, 0);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            texels[y * 256 + x * 4 + 0] = 255;
            texels[y * 256 + x * 4 + 1] = 128;
            texels[y * 256 + x * 4 + 3] = 255;
        }
    GpuWriteTexturePayload wt;
    std::memset(&wt, 0, sizeof wt);
    wt.texture_id = 20;
    wt.width = 2;
    wt.height = 2;
    wt.depth_or_layers = 1;
    wt.bytes_per_row = 256;
    wt.rows_per_image = 2;
    wt.data_len = static_cast<uint32_t>(texels.size());
    std::vector<uint8_t> wtv = bytes_of(wt);
    wtv.insert(wtv.end(), texels.begin(), texels.end());
    emit(setup, GpuOpcode::WriteTexture, wtv);
    GpuCreateSampler2Payload sp;
    std::memset(&sp, 0, sizeof sp);
    sp.id = 21;
    sp.max_anisotropy = 1;
    sp.lod_max_clamp = 32.0f;
    emit(setup, GpuOpcode::CreateSampler2, bytes_of(sp));
    setup.create_shader_module(30, GpuShaderStage::Vertex, kTexturedTriangle);
    setup.create_shader_module(31, GpuShaderStage::Fragment, kTexturedTriangle);
    create_layouts(setup, 40, 41,
                   {layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture, 0, 2),
                    layout_entry(1, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Sampler, 0)});
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(50, 30, 31, 41)));
    create_bind_group(setup, 60, 40, {bg_entry(0, GpuBindResourceType::TextureView, 20), bg_entry(1, GpuBindResourceType::Sampler, 21)});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    frame.set_pipeline(50);
    set_bind_group2(frame, 0, 60);
    frame.draw(3);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    const ExecutorStats& after = ex.stats();
    CHECK(after.device_errors == before.device_errors);
    CHECK(after.packet_errors == before.packet_errors);
    CHECK(after.draws == before.draws + 1);
    CHECK(after.texture_uploads == before.texture_uploads + 1);
    CHECK(frame_is(ex, {0, 128, 255, 255}));  // bgra of (255, 128, 0, 255)

    // The uploaded texels read back exactly (rgba order, rows tightly packed).
    std::vector<uint8_t> px;
    uint32_t w = 0, h = 0;
    std::string err;
    CHECK(ex.read_texture(20, 0, &px, &w, &h, &err) && w == 2 && h == 2);
    CHECK(px.size() == 16 && px[0] == 255 && px[1] == 128 && px[2] == 0 && px[3] == 255 && px[12] == 255);

    // A draw whose pipeline reads an unbound @group(0) is skipped, not submitted.
    gpu::CommandEncoder skip;
    skip.begin_frame();
    skip.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    skip.set_pipeline(50);
    skip.draw(3);
    skip.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    skip.present();
    run(ex, skip);
    CHECK(ex.stats().draws_skipped == before.draws_skipped + 1);
    CHECK(frame_is(ex, {0, 0, 0, 255}));
}

const char* kCompute = R"(
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = i * 2u + 1u; }
)";

void test_compute(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(70, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_buffer(71, GpuBufferUsage::Indirect | GpuBufferUsage::CopyDst, 12);
    const uint32_t args[3] = {1, 1, 1};
    setup.write_buffer(71, 0, args, 12);
    setup.create_shader_module(72, GpuShaderStage::Compute, kCompute);
    create_layouts(setup, 73, 74, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 75;
    cp.cs_module_id = 72;
    cp.pipeline_layout_id = 74;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 76, 73, {bg_entry(0, GpuBindResourceType::Buffer, 70)});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(75);
    set_bind_group2(frame, 0, 76);
    frame.dispatch_compute(0, 1, 1);  // zero-size dispatches are skipped (validateComputeDispatch)
    frame.indirect(GpuOpcode::DispatchIndirect, 71, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(70, 0, 16, &out, &err));
    const uint32_t want[4] = {1, 3, 5, 7};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    CHECK(ex.stats().dispatches == before.dispatches + 1);
    CHECK(ex.stats().dispatches_skipped == before.dispatches_skipped + 1);
    CHECK(ex.stats().device_errors == before.device_errors);
}

const char* kPackPath = "test_native_executor.wgsl.pack";

void test_pack_and_formats(NativeExecutor& ex) {
    // A one-entry WGPK pack: a hash-only CreateShaderModule resolves through it.
    const std::string wgsl = "@compute @workgroup_size(1) fn cs_main() {}\n";
    std::vector<uint8_t> pack = {'W', 'G', 'P', 'K', 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    append(pack, uint64_t{0x00c0ffee12345678ull});
    pack.push_back(3);  // compute
    pack.insert(pack.end(), 3, 0);
    append(pack, static_cast<uint32_t>(16 + 20));
    append(pack, static_cast<uint32_t>(wgsl.size()));
    pack.insert(pack.end(), wgsl.begin(), wgsl.end());
    const std::string path = kPackPath;
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        if (f) {
            std::fwrite(pack.data(), 1, pack.size(), f);
            std::fclose(f);
        }
    }
    std::string err;
    CHECK(ex.load_wgsl_pack(path, &err));
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder enc;
    enc.create_shader_module(80, GpuShaderStage::Compute, "0x00c0ffee12345678");
    enc.create_shader_module(81, GpuShaderStage::Compute, "0x00c0ffee00000000");  // not in the pack
    run(ex, enc);
    CHECK(ex.stats().shader_pack_hits == before.shader_pack_hits + 1);
    CHECK(ex.stats().shader_pack_misses == before.shader_pack_misses + 1);

    // A BC7 texture takes a 4x4 block upload (16 bytes) without a device error.
    ex.wait_idle();
    ex.poll();
    const uint64_t errors_before_bc = ex.stats().device_errors;
    const uint64_t uploads = ex.stats().texture_uploads;
    gpu::CommandEncoder bc;
    emit(bc, GpuOpcode::CreateTexture2,
         bytes_of(texture2(90, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 62)));
    GpuWriteTexturePayload wt;
    std::memset(&wt, 0, sizeof wt);
    wt.texture_id = 90;
    wt.width = 4;
    wt.height = 4;
    wt.depth_or_layers = 1;
    wt.bytes_per_row = 16;
    wt.rows_per_image = 1;
    wt.data_len = 16;
    std::vector<uint8_t> v = bytes_of(wt);
    v.insert(v.end(), 16, 0x11);
    emit(bc, GpuOpcode::WriteTexture, v);
    run(ex, bc);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().texture_uploads == uploads + 1);
    CHECK(ex.stats().device_errors == errors_before_bc);
}

}  // namespace

int main() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        if (status == NativeExecutor::CreateStatus::NoAdapter) {
            std::printf("test_native_executor: skipped (%s)\n", error.c_str());
            return 77;
        }
        std::fprintf(stderr, "FAIL: NativeExecutor::create: %s\n", error.c_str());
        return 1;
    }
    std::printf("adapter: %s (%s)\n", ex->adapter().description.c_str(), ex->adapter().backend.c_str());
    test_clear_present(*ex);
    test_buffers(*ex);
    test_textured_draw(*ex);
    test_compute(*ex);
    test_pack_and_formats(*ex);
    test_stream_accounting(*ex);
    CHECK(ex->stats().device_lost == 0);
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    ex.reset();
    std::remove(kPackPath);
    std::printf("test_native_executor: ok\n");
    return 0;
}
