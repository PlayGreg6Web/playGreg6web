// Native executor milestones 0 to 2 on the host's real GPU (headless): device, offscreen swapchain,
// clear/present, buffers, textures, shaders, layouts, pipelines, draws and dispatches, v1 pipelines,
// GPU -> CPU read-backs into (host) memory and GPU pass timestamps, with every result read back from the GPU. Exits 77
// (CTest: skipped) when the host has no WebGPU adapter. Built with -DPLAYGTA6_NATIVE_EXECUTOR=ON.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/gpu_time.h"
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
    const uint32_t zero = 0;
    enc.emit_packet(static_cast<GpuOpcode>(50), &zero, 4);  // an opcode no milestone implements: skipped and counted
    enc.nop();
    std::vector<uint8_t> bytes(enc.data(), enc.data() + enc.size());
    const GpuPacketHeader truncated{static_cast<uint16_t>(GpuOpcode::WriteBuffer), 0, 64};
    const size_t at = bytes.size();
    bytes.resize(at + 8 + 4);
    std::memcpy(bytes.data() + at, &truncated, 8);
    CHECK(ex.execute(bytes.data(), bytes.size()) == 0);
    const ExecutorStats& after = ex.stats();
    CHECK(after.unhandled[50] == before.unhandled[50] + 1);
    CHECK(after.handled[0] == before.handled[0] + 1);
    CHECK(after.truncated_batches == before.truncated_batches + 1);
    CHECK(after.packets == before.packets + 2);
    const std::string json = ex.stats_json();
    CHECK(json.find("\"unhandled\": {\"50\": 1}") != std::string::npos);
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

// An explicit layout that does not fit the WGSL (binding 0 in the layout, binding 3 in the shader) gets
// layout:'auto' and the worker's group 0 remap: the buffer bound at 0 serves binding 3, with the
// dynamic offset folded into the static offset.
const char* kComputeBinding3 = R"(
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = i * 3u + 2u; }
)";

void test_auto_layout_remap(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(110, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 512);
    setup.create_shader_module(111, GpuShaderStage::Compute, kComputeBinding3);
    GpuBindGroupLayoutEntry le = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1);
    le.flags = 1;  // hasDynamicOffset
    create_layouts(setup, 112, 113, {le});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 114;
    cp.cs_module_id = 111;
    cp.pipeline_layout_id = 113;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    GpuBindGroupEntry2 be = bg_entry(0, GpuBindResourceType::Buffer, 110);
    be.size = 16;
    create_bind_group(setup, 115, 112, {be});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(114);
    std::vector<uint8_t> sb;
    append(sb, GpuSetBindGroup2Payload{0, 115, 1, 0});
    append(sb, uint32_t{256});
    emit(frame, GpuOpcode::SetBindGroup2, sb);
    frame.dispatch_compute(1, 1, 1);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(110, 256, 16, &out, &err));
    const uint32_t want[4] = {2, 5, 8, 11};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    const ExecutorStats& s = ex.stats();
    CHECK(s.auto_layout_remap_pipelines == before.auto_layout_remap_pipelines + 1);
    CHECK(s.remapped_bind_groups == before.remapped_bind_groups + 1);
    CHECK(s.remap_dynamic_offsets == before.remap_dynamic_offsets + 1);
    CHECK(s.device_errors == before.device_errors);
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

// ---- milestone 2: read-backs, v1 pipelines, GPU pass timestamps ----

uint32_t u32_at(const std::vector<uint8_t>& mem, size_t off) {
    uint32_t v;
    std::memcpy(&v, mem.data() + off, 4);
    return v;
}

GpuReadbackBufferPayload rb_buffer(uint32_t id, uint64_t src, uint64_t size, uint64_t dst, uint64_t done) {
    GpuReadbackBufferPayload r;
    std::memset(&r, 0, sizeof r);
    r.seq = 1;
    r.buffer_id = id;
    r.src_offset = src;
    r.size = size;
    r.dst_addr = dst;
    r.done_addr = done;
    return r;
}

void test_readbacks(NativeExecutor& ex) {
    std::vector<uint8_t> mem(8192, 0xEE);
    GuestMemory gm;
    gm.base = mem.data();
    gm.size = mem.size();
    std::memset(mem.data(), 0, 16);  // the done counters
    ex.attach_guest_memory(gm);
    const ExecutorStats before = ex.stats();

    // A compute dispatch and the read-backs of its result, in one batch with no Present: the executor
    // submits the recorded work at the end of the batch (the worker's early read-back flush).
    gpu::CommandEncoder enc;
    enc.create_buffer(100, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    enc.create_shader_module(101, GpuShaderStage::Compute, kCompute);
    create_layouts(enc, 102, 103, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 104;
    cp.cs_module_id = 101;
    cp.pipeline_layout_id = 103;
    emit(enc, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(enc, 105, 102, {bg_entry(0, GpuBindResourceType::Buffer, 100)});
    enc.set_compute_pipeline(104);
    set_bind_group2(enc, 0, 105);
    enc.dispatch_compute(1);
    emit(enc, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(100, 4, 8, 256, 0)));
    emit(enc, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(100, 5, 6, 512, 4)));  // unaligned source range
    CHECK(run(ex, enc) == 0);
    ex.wait_idle();
    ex.poll();
    const ExecutorStats& a = ex.stats();
    CHECK(a.frames_submitted == before.frames_submitted);  // no frame was presented
    CHECK(a.early_readback_flushes == before.early_readback_flushes + 1);
    CHECK(a.readbacks_requested == before.readbacks_requested + 2);
    CHECK(a.readbacks_delivered == before.readbacks_delivered + 2);
    CHECK(a.readback_bytes == before.readback_bytes + 14);
    const uint8_t want0[8] = {3, 0, 0, 0, 5, 0, 0, 0};
    const uint8_t want1[6] = {0, 0, 0, 5, 0, 0};
    CHECK(std::memcmp(mem.data() + 256, want0, 8) == 0);
    CHECK(mem[255] == 0xEE && mem[264] == 0xEE);
    CHECK(std::memcmp(mem.data() + 512, want1, 6) == 0);
    CHECK(mem[511] == 0xEE && mem[518] == 0xEE);
    CHECK(u32_at(mem, 0) == 1 && u32_at(mem, 4) == 1);

    // A texture region lands in the guest's footprint: rows 16 bytes apart, the gap untouched.
    gpu::CommandEncoder tex;
    emit(tex, GpuOpcode::CreateTexture2,
         bytes_of(texture2(110, 2, 2, GpuTextureUsage::CopySrc | GpuTextureUsage::CopyDst, 22)));
    std::vector<uint8_t> texels(256 + 8, 0);
    for (int i = 0; i < 8; ++i) texels[i] = static_cast<uint8_t>(1 + i);
    for (int i = 0; i < 8; ++i) texels[256 + i] = static_cast<uint8_t>(9 + i);
    GpuWriteTexturePayload wt;
    std::memset(&wt, 0, sizeof wt);
    wt.texture_id = 110;
    wt.width = 2;
    wt.height = 2;
    wt.depth_or_layers = 1;
    wt.bytes_per_row = 256;
    wt.rows_per_image = 2;
    wt.data_len = static_cast<uint32_t>(texels.size());
    std::vector<uint8_t> wtv = bytes_of(wt);
    wtv.insert(wtv.end(), texels.begin(), texels.end());
    emit(tex, GpuOpcode::WriteTexture, wtv);
    GpuReadbackTexturePayload rt;
    std::memset(&rt, 0, sizeof rt);
    rt.seq = 3;
    rt.texture_id = 110;
    rt.width = 2;
    rt.height = 2;
    rt.depth_or_layers = 1;
    rt.bytes_per_row = 16;
    rt.rows_per_image = 2;
    rt.dst_addr = 1024;
    rt.done_addr = 8;
    rt.dst_size = 24;
    emit(tex, GpuOpcode::ReadbackTexture, bytes_of(rt));
    run(ex, tex);
    ex.wait_idle();
    ex.poll();
    for (int i = 0; i < 8; ++i) CHECK(mem[1024 + i] == 1 + i);
    for (int i = 0; i < 8; ++i) CHECK(mem[1040 + i] == 9 + i);
    for (int i = 8; i < 16; ++i) CHECK(mem[1024 + i] == 0xEE);
    CHECK(u32_at(mem, 8) == 1);
    CHECK(ex.stats().readbacks_delivered == before.readbacks_delivered + 3);

    // Failed copies advance the counter without writing: an unknown buffer, and a destination that
    // does not fit in the memory. A counter address that is not 4-aligned is left alone (the data
    // still lands).
    gpu::CommandEncoder bad;
    emit(bad, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(9999, 0, 4, 2048, 12)));
    emit(bad, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(100, 0, 8, 8190, 12)));
    emit(bad, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(100, 0, 4, 2056, 14)));  // data lands, the counter does not move
    run(ex, bad);
    ex.wait_idle();
    ex.poll();
    CHECK(mem[2048] == 0xEE && mem[8190] == 0xEE && mem[2056] == 1);
    CHECK(u32_at(mem, 12) == 2);
    CHECK(ex.stats().readbacks_failed == before.readbacks_failed + 2);

    // No memory attached: nothing is written and the counter does not move.
    ex.attach_guest_memory(GuestMemory{});
    gpu::CommandEncoder none;
    emit(none, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(100, 0, 4, 3000, 12)));
    run(ex, none);
    ex.wait_idle();
    ex.poll();
    CHECK(u32_at(mem, 12) == 2 && mem[3000] == 0xEE);
    CHECK(ex.stats().readbacks_failed == before.readbacks_failed + 3);

    // A truncated payload is an op error, nothing is recorded.
    const uint64_t errors = ex.stats().op_errors;
    gpu::CommandEncoder trunc;
    trunc.emit_packet(GpuOpcode::ReadbackBuffer, &gm, 16);
    run(ex, trunc);
    CHECK(ex.stats().op_errors == errors + 1);
    CHECK(ex.stats().device_errors == before.device_errors);
}

void test_v1(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    std::vector<uint8_t> mem(256, 0);
    GuestMemory gm;
    gm.base = mem.data();
    gm.size = mem.size();
    ex.attach_guest_memory(gm);

    gpu::CommandEncoder setup;
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(120, 2, 2, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 22)));
    std::vector<uint8_t> texels(256 + 8, 0);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            texels[y * 256 + x * 4 + 0] = 10;
            texels[y * 256 + x * 4 + 1] = 200;
            texels[y * 256 + x * 4 + 2] = 30;
            texels[y * 256 + x * 4 + 3] = 255;
        }
    GpuWriteTexturePayload wt;
    std::memset(&wt, 0, sizeof wt);
    wt.texture_id = 120;
    wt.width = 2;
    wt.height = 2;
    wt.depth_or_layers = 1;
    wt.bytes_per_row = 256;
    wt.rows_per_image = 2;
    wt.data_len = static_cast<uint32_t>(texels.size());
    std::vector<uint8_t> wtv = bytes_of(wt);
    wtv.insert(wtv.end(), texels.begin(), texels.end());
    emit(setup, GpuOpcode::WriteTexture, wtv);
    setup.create_sampler(121);
    setup.create_shader_module(122, GpuShaderStage::Vertex, kTexturedTriangle);
    setup.create_shader_module(123, GpuShaderStage::Fragment, kTexturedTriangle);
    setup.create_render_pipeline(124, 122, 123, 0);  // v1: layout 'auto', vs_main / fs_main
    const GpuBindGroupEntry entries[2] = {
        {0, static_cast<uint32_t>(GpuBindResourceType::TextureView), 120, 0, 0, 0},
        {1, static_cast<uint32_t>(GpuBindResourceType::Sampler), 121, 0, 0, 0},
    };
    setup.create_bind_group(125, 124, 0, entries, 2);
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_pipeline(124);
    frame.set_bind_group(0, 125);
    frame.draw(3);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().device_errors == before.device_errors);
    CHECK(ex.stats().v1_render_pipelines == before.v1_render_pipelines + 1);
    CHECK(ex.stats().v1_bind_groups == before.v1_bind_groups + 1);
    CHECK(ex.stats().draws == before.draws + 1);
    CHECK(frame_is(ex, {30, 200, 10, 255}));

    // A v1 draw without its bind group is skipped like the v2 one.
    gpu::CommandEncoder skip;
    skip.begin_frame();
    skip.set_pipeline(124);
    skip.draw(3);
    skip.present();
    run(ex, skip);
    CHECK(ex.stats().draws_skipped == before.draws_skipped + 1);

    // v1 compute pipeline + bind group (buffer binding), result read back into memory.
    gpu::CommandEncoder comp;
    comp.create_buffer(130, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    comp.create_shader_module(131, GpuShaderStage::Compute, kCompute);
    comp.create_compute_pipeline(132, 131);
    const GpuBindGroupEntry bge = {0, static_cast<uint32_t>(GpuBindResourceType::Buffer), 130, 0, 0, 0};
    comp.create_bind_group(133, 132, 0, &bge, 1);
    comp.begin_frame();
    comp.set_compute_pipeline(132);
    comp.set_bind_group(0, 133);
    comp.dispatch_compute(1);
    emit(comp, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(130, 0, 16, 64, 0)));
    comp.present();
    run(ex, comp);
    ex.wait_idle();
    ex.poll();
    const uint32_t want[4] = {1, 3, 5, 7};
    CHECK(std::memcmp(mem.data() + 64, want, 16) == 0);
    CHECK(u32_at(mem, 0) == 1);
    CHECK(ex.stats().v1_compute_pipelines == before.v1_compute_pipelines + 1);
    CHECK(ex.stats().dispatches == before.dispatches + 1);
    CHECK(ex.stats().device_errors == before.device_errors);

    // Unknown shader / unknown resources are validation errors that create nothing.
    const uint64_t errors = ex.stats().op_errors;
    gpu::CommandEncoder bad;
    bad.create_render_pipeline(140, 9999, 0, 0);
    const GpuBindGroupEntry missing = {0, static_cast<uint32_t>(GpuBindResourceType::Buffer), 8888, 0, 0, 0};
    bad.create_bind_group(141, 132, 0, &missing, 1);
    run(ex, bad);
    CHECK(ex.stats().op_errors == errors + 2);
    ex.attach_guest_memory(GuestMemory{});
}

const char* kHeavyCompute = R"(
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  var a = id.x;
  for (var i = 0u; i < 60000u; i = i + 1u) { a = a * 1664525u + 1013904223u; }
  out[id.x] = a;
}
)";

// GPU pass timestamps feed a GpuTimeTable with the pipeline's shader hash as the key.
void test_timers() {
    gpu::d3d12_guest::GpuTimeTable table;
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.gpu_time = &table;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        std::fprintf(stderr, "FAIL: second NativeExecutor::create: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    if (!ex->stats().timer_supported) {
        std::printf("test_native_executor: adapter lacks timestamp-query: timer checks skipped\n");
        CHECK(table.support() == gpu::d3d12_guest::GpuTimeTable::Support::Unavailable);
        return;
    }
    const uint64_t kHash = 0x00c0ffee0badf00dull;
    const std::string wgsl = kHeavyCompute;
    std::vector<uint8_t> pack = {'W', 'G', 'P', 'K', 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    append(pack, kHash);
    pack.push_back(3);
    pack.insert(pack.end(), 3, 0);
    append(pack, static_cast<uint32_t>(16 + 20));
    append(pack, static_cast<uint32_t>(wgsl.size()));
    pack.insert(pack.end(), wgsl.begin(), wgsl.end());
    const std::string path = "test_native_executor_timer.wgsl.pack";
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        if (f) {
            std::fwrite(pack.data(), 1, pack.size(), f);
            std::fclose(f);
        }
    }
    std::string err;
    CHECK(ex->load_wgsl_pack(path, &err));

    gpu::CommandEncoder setup;
    setup.create_buffer(200, GpuBufferUsage::Storage, 64 * 64 * 4);
    setup.create_shader_module(201, GpuShaderStage::Compute, "0x00c0ffee0badf00d");
    create_layouts(setup, 202, 203, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 204;
    cp.cs_module_id = 201;
    cp.pipeline_layout_id = 203;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 205, 202, {bg_entry(0, GpuBindResourceType::Buffer, 200)});
    run(*ex, setup);
    for (int f = 0; f < 3; ++f) {
        gpu::CommandEncoder frame;
        frame.begin_frame();
        frame.set_compute_pipeline(204);
        set_bind_group2(frame, 0, 205);
        frame.dispatch_compute(64);
        frame.present();
        run(*ex, frame);
        ex->wait_idle();  // one frame in flight at a time: no slot is skipped
    }
    ex->poll();
    CHECK(ex->stats().device_errors == 0);
    CHECK(ex->stats().timer_frames == 3);
    CHECK(ex->stats().timer_passes == 6);  // the compute pass and the v1 clearing pass of each frame
    CHECK(table.support() == gpu::d3d12_guest::GpuTimeTable::Support::Available);
    bool found = false;
    uint64_t compute_ns = 0;
    for (const auto& e : table.top(16)) {
        if (e.hash == kHash && e.kind == gpu::d3d12_guest::GpuPassKind::Compute) {
            found = true;
            compute_ns = e.stats.total_ns;
            CHECK(e.stats.passes == 3);
        }
    }
    CHECK(found);
    CHECK(compute_ns > 0);
    const std::string json = table.json();
    CHECK(json.find("\"frames_measured\": 3") != std::string::npos);
    std::printf("timer: %llu ns of compute over 3 passes\n", static_cast<unsigned long long>(compute_ns));
    ex.reset();  // closes the pack
    std::remove(path.c_str());
}

}  // namespace

// The live boot (--window) attaches absolute host addresses: the native producer's ReadbackGate
// hands out host pointers for dst_addr / done_addr. The bytes are the compute result still in
// buffer 100 (bytes 4..12: 3, 5 as u32), read back from the GPU.
void test_readbacks_host_addresses(NativeExecutor& ex) {
    alignas(8) uint8_t dst[16];
    std::memset(dst, 0xEE, sizeof dst);
    alignas(8) uint32_t done = 0;
    GuestMemory gm;
    gm.host_addresses = true;
    ex.attach_guest_memory(gm);
    const uint64_t before = ex.stats().readbacks_delivered;
    gpu::CommandEncoder enc;
    emit(enc, GpuOpcode::ReadbackBuffer,
         bytes_of(rb_buffer(100, 4, 8, reinterpret_cast<uintptr_t>(dst), reinterpret_cast<uintptr_t>(&done))));
    run(ex, enc);
    ex.wait_idle();
    ex.poll();
    const uint8_t want[8] = {3, 0, 0, 0, 5, 0, 0, 0};
    CHECK(std::memcmp(dst, want, 8) == 0);
    CHECK(dst[8] == 0xEE);
    CHECK(std::atomic_ref<uint32_t>(done).load() == 1);
    CHECK(ex.stats().readbacks_delivered == before + 1);
    ex.attach_guest_memory(GuestMemory{});
}

// Swapchain blit (WgpuSwapchain intermediate texture + blit, letterboxed): a 64x32 stream presented to a
// 128x128 canvas lands in a 128x64 band with black bars; a non-canvas (sRGB) stream format goes through the
// same path and comes out as the colour that was cleared (sampler decode, no re-encode).
bool px_near(const FrameImage& img, uint32_t x, uint32_t y, int b, int g, int r, int a, int tol) {
    const uint8_t* p = &img.pixels[(size_t{y} * img.width + x) * 4];
    auto close_to = [&](int got, int want) { return got >= want - tol && got <= want + tol; };
    const bool ok = close_to(p[0], b) && close_to(p[1], g) && close_to(p[2], r) && close_to(p[3], a);
    if (!ok)
        std::fprintf(stderr, "pixel (%u,%u) = %u,%u,%u,%u (bgra), want %d,%d,%d,%d\n", x, y, p[0], p[1], p[2], p[3], b,
                     g, r, a);
    return ok;
}

void test_blit() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.present_width = 128;
    opts.present_height = 128;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        std::fprintf(stderr, "FAIL: blit NativeExecutor::create: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    gpu::CommandEncoder f;
    f.begin_frame();
    f.clear_color(1.0f, 0.0f, 0.0f, 1.0f);
    f.present();
    CHECK(run(*ex, f) == 1);
    CHECK(ex->stats().blit_frames == 1);
    FrameImage img;
    CHECK(ex->read_last_frame(&img, &error));
    CHECK(img.width == 128 && img.height == 128 && img.format == "bgra8unorm");
    if (img.pixels.size() == 128u * 128u * 4u) {
        CHECK(px_near(img, 64, 5, 0, 0, 0, 255, 0));     // top bar
        CHECK(px_near(img, 64, 122, 0, 0, 0, 255, 0));   // bottom bar
        CHECK(px_near(img, 64, 64, 0, 0, 255, 255, 1));  // inside the band: the cleared red
        CHECK(px_near(img, 2, 40, 0, 0, 255, 255, 1));
        CHECK(px_near(img, 125, 90, 0, 0, 255, 255, 1));
    }

    // sRGB stream format (code 28 bgra8unorm-srgb): clear 0.5 is stored encoded and sampled decoded.
    GpuBeginRenderPassPayload bp = swapchain_pass(GpuLoadOp::Clear, 0.5f, 0.5f, 0.5f, 1.0f);
    bp.color[0].view.format = 28;
    gpu::CommandEncoder g;
    g.begin_frame();
    g.emit_packet(GpuOpcode::BeginRenderPass, &bp, sizeof bp);
    g.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    g.present();
    CHECK(run(*ex, g) == 1);
    CHECK(ex->stats().blit_frames == 2);
    CHECK(ex->stats().swapchain_format_unsupported == 0);
    CHECK(ex->read_last_frame(&img, &error));
    if (img.pixels.size() == 128u * 128u * 4u) {
        CHECK(px_near(img, 64, 5, 0, 0, 0, 255, 0));
        CHECK(px_near(img, 64, 64, 128, 128, 128, 255, 2));
    }
    CHECK(ex->stats().device_errors == 0);
    CHECK(ex->stats().device_lost == 0);

    // A same-size stream in a canvas format does not blit.
    ExecutorOptions plain;
    plain.width = kW;
    plain.height = kH;
    auto ex2 = NativeExecutor::create(plain, &status, &error);
    if (ex2) {
        gpu::CommandEncoder h;
        h.begin_frame();
        h.present();
        run(*ex2, h);
        CHECK(ex2->stats().blit_frames == 0);
    }
}

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
    std::printf("limits_raised: %d\n", ex->stats().limits_raised ? 1 : 0);
    test_clear_present(*ex);
    test_buffers(*ex);
    test_textured_draw(*ex);
    test_compute(*ex);
    test_auto_layout_remap(*ex);
    test_pack_and_formats(*ex);
    test_stream_accounting(*ex);
    test_readbacks(*ex);
    test_readbacks_host_addresses(*ex);
    test_v1(*ex);
    CHECK(ex->stats().device_lost == 0);
    test_timers();
    test_blit();
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    ex.reset();
    std::remove(kPackPath);
    std::printf("test_native_executor: ok\n");
    return 0;
}
