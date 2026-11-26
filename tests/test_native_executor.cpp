// Native executor milestones 0 to 2 on the host's real GPU (headless): device, offscreen swapchain,
// clear/present, buffers, textures, shaders, layouts, pipelines, draws and dispatches, v1 pipelines,
// GPU -> CPU read-backs into (host) memory and GPU pass timestamps, with every result read back from the GPU. Exits 77
// (CTest: skipped) when the host has no WebGPU adapter. Built with -DPLAYGTA6_NATIVE_EXECUTOR=ON.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/gpu_time.h"
#include "gpu/native_exec/native_executor.h"
#include "gpu/native_exec/reflection_model.h"
#include "png_stored_decode.h"

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

// Reflection loader (reflection_model.h): a `.reflect.json` sidecar keyed by the hash the worker gives an
// inline shader (FNV-1a 64 of its text) names the entry points, the vertex attributes and the layout of
// v1 pipelines; a layout-manifest.json serves ManifestReference layouts. Each case only works because of
// the reflection record (the WGSL has no vs_main / cs_main, location 2 / group 1 do not exist in the
// default layout), and every result is read back from the GPU.
const char* kReflVs = R"(
struct Scale { v: vec4f };
@group(1) @binding(0) var<uniform> scale: Scale;
struct VOut { @builtin(position) p: vec4f, @location(0) c: vec4f };
@vertex fn vtx(@location(0) pos: vec2f, @location(2) col: vec4f) -> VOut {
  var o: VOut;
  o.p = vec4f(pos * scale.v.x, 0.0, 1.0);
  o.c = col;
  return o;
}
)";
const char* kReflFs = R"(
struct Tint { v: vec4f };
@group(1) @binding(1) var<uniform> tint: Tint;
@fragment fn frg(@location(0) c: vec4f) -> @location(0) vec4f { return c * tint.v; }
)";
const char* kReflCs = R"(
@group(1) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(4) fn my_cs(@builtin(local_invocation_index) i: u32) { out[i] = i * 5u + 3u; }
)";
// Same bindings as kReflFs but a storage buffer where kReflVs has a uniform: the merge must reject.
const char* kReflFsConflict = R"(
@group(1) @binding(0) var<storage, read> tint: array<vec4f>;
@fragment fn frg(@location(0) c: vec4f) -> @location(0) vec4f { return c * tint[0]; }
)";

std::string hash_hex_of(const char* wgsl) { return gpu::native_exec::reflect::format_hash(gpu::native_exec::reflect::fnv1a64(wgsl)); }

std::string reflection_sidecar() {
    return std::string(R"({"schema":"playgta6-stage-reflection","version":3,"stages":{")") + hash_hex_of(kReflVs) +
           R"(":{"stage":"vertex","entry_point":"vtx",
  "vertex_inputs":[
    {"location":2,"vb_slot":0,"byte_offset":8,"components":4,"component_type":"f32","fetch_format":"unorm8x4"},
    {"location":0,"vb_slot":0,"byte_offset":0,"components":2,"component_type":"f32","fetch_format":"float32x2"}],
  "bindings":[{"group":1,"binding":0,"kind":"uniform","min_binding_size_bytes":16,"source":{"user_sgpr":8,"table_byte_offset":0}}]},")" +
           hash_hex_of(kReflFs) + R"(":{"stage":"fragment","entry_point":"frg",
  "bindings":[{"group":1,"binding":1,"kind":"uniform","min_binding_size_bytes":16,"source":{"user_sgpr":8,"table_byte_offset":32}}]},")" +
           hash_hex_of(kReflCs) + R"(":{"stage":"compute","entry_point":"my_cs",
  "bindings":[{"group":1,"binding":2,"kind":"storage","source":{"user_sgpr":4,"table_byte_offset":0}}]},")" +
           hash_hex_of(kReflFsConflict) + R"(":{"stage":"fragment","entry_point":"frg",
  "bindings":[{"group":1,"binding":0,"kind":"read_only_storage","source":{"user_sgpr":8,"table_byte_offset":0}}]}}})";
}

const char* kLayoutManifest = R"({
  "schema_version": 1,
  "bind_group_layouts": {"0x00000000000000aa": {"entries": [{"binding": 0, "visibility": 4, "buffer": {"type": "storage"}}]}},
  "pipeline_layouts": {"0x00000000000000bb": {"bindGroupLayouts": ["0xaa"]}},
  "shaders": {}
})";

void test_reflection() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto exu = NativeExecutor::create(opts, &status, &error);
    if (!exu) {
        CHECK(status == NativeExecutor::CreateStatus::NoAdapter);
        return;
    }
    NativeExecutor& ex = *exu;

    // ManifestReference layouts need a loaded layout-manifest.json: without one they are unavailable.
    auto manifest_packets = [](gpu::CommandEncoder& enc) {
        std::vector<uint8_t> bgl;
        append(bgl, GpuCreateBindGroupLayoutPayload{300, 0, GpuLayoutFlags::ManifestReference, 0, 0xaa});
        emit(enc, GpuOpcode::CreateBindGroupLayout, bgl);
        std::vector<uint8_t> pl;
        append(pl, GpuCreatePipelineLayoutPayload{301, 1, GpuLayoutFlags::ManifestReference, 0, 0xbb});
        emit(enc, GpuOpcode::CreatePipelineLayout, pl);
    };
    gpu::CommandEncoder before_load;
    manifest_packets(before_load);
    run(ex, before_load);
    CHECK(ex.stats().manifest_layouts_unavailable == 2 && ex.stats().manifest_layouts_resolved == 0);
    CHECK(ex.stats().packet_errors >= 2);

    std::string err;
    CHECK(!ex.load_reflection_text("{", &err) && !err.empty());
    CHECK(!ex.load_reflection_text(R"({"schema_version":1,"shaders":{},"bind_group_layouts":{"0xaa":{"entries":[{"binding":0,"visibility":9,"buffer":{}}]}}})", &err));
    CHECK(ex.stats().reflection_load_errors == 2 && ex.stats().reflection_documents == 0 && ex.stats().reflection_shaders == 0);
    CHECK(ex.load_reflection_text(kLayoutManifest, &err));
    CHECK(ex.stats().reflection_documents == 1);

    gpu::CommandEncoder setup;
    manifest_packets(setup);
    setup.create_buffer(70, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_shader_module(72, GpuShaderStage::Compute, kCompute);
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 75;
    cp.cs_module_id = 72;
    cp.pipeline_layout_id = 301;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 76, 300, {bg_entry(0, GpuBindResourceType::Buffer, 70)});
    run(ex, setup);
    CHECK(ex.stats().manifest_layouts_resolved == 2 && ex.stats().manifest_layouts_unavailable == 2);
    gpu::CommandEncoder mframe;
    mframe.begin_frame();
    mframe.set_compute_pipeline(75);
    set_bind_group2(mframe, 0, 76);
    mframe.dispatch_compute(1, 1, 1);
    mframe.present();
    run(ex, mframe);
    ex.wait_idle();
    std::vector<uint8_t> out;
    CHECK(ex.read_buffer(70, 0, 16, &out, &err));
    const uint32_t want_m[4] = {1, 3, 5, 7};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want_m, 16) == 0);

    // The sidecar: no record yet, so a v1 compute pipeline on kReflCs would look for cs_main and fail.
    CHECK(ex.load_reflection_text(reflection_sidecar(), &err));
    CHECK(ex.stats().reflection_documents == 2 && ex.stats().reflection_shaders == 4);

    const ExecutorStats before = ex.stats();
    // v1 compute: entry point `my_cs`, layout group 1 binding 2 from the record.
    gpu::CommandEncoder comp;
    comp.create_buffer(80, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    comp.create_shader_module(81, GpuShaderStage::Compute, kReflCs);
    comp.create_compute_pipeline(82, 81);
    const GpuBindGroupEntry cbe = {2, static_cast<uint32_t>(GpuBindResourceType::Buffer), 80, 0, 0, 0};
    comp.create_bind_group(83, 82, 1, &cbe, 1);
    comp.begin_frame();
    comp.set_compute_pipeline(82);
    comp.set_bind_group(1, 83);
    comp.dispatch_compute(1);
    comp.present();
    run(ex, comp);
    ex.wait_idle();
    CHECK(ex.read_buffer(80, 0, 16, &out, &err));
    const uint32_t want_c[4] = {3, 8, 13, 18};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want_c, 16) == 0);
    CHECK(ex.stats().reflected_compute_pipelines == before.reflected_compute_pipelines + 1);
    CHECK(ex.stats().reflection_unavailable == before.reflection_unavailable);

    // v1 render: entry points vtx / frg, attributes at locations 0 (float32x2) and 2 (unorm8x4) of one
    // 12-byte vertex, uniforms at group 1. Colour (1, 1, 0, 1) * tint (0, 1, 1, 1) = green everywhere.
    struct Vtx {
        float x, y;
        uint8_t c[4];
    };
    const Vtx verts[3] = {{-1, -1, {255, 255, 0, 255}}, {3, -1, {255, 255, 0, 255}}, {-1, 3, {255, 255, 0, 255}}};
    static_assert(sizeof(Vtx) == 12, "vertex layout");
    const float scale[4] = {1, 1, 1, 1};
    const float tint[4] = {0, 1, 1, 1};
    gpu::CommandEncoder rs;
    rs.create_buffer(90, GpuBufferUsage::Vertex | GpuBufferUsage::CopyDst, sizeof verts);
    rs.write_buffer(90, 0, verts, sizeof verts);
    rs.create_buffer(91, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, 16);
    rs.write_buffer(91, 0, scale, 16);
    rs.create_buffer(92, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, 16);
    rs.write_buffer(92, 0, tint, 16);
    rs.create_shader_module(93, GpuShaderStage::Vertex, kReflVs);
    rs.create_shader_module(94, GpuShaderStage::Fragment, kReflFs);
    rs.create_render_pipeline(95, 93, 94, sizeof(Vtx));
    const GpuBindGroupEntry rbe[2] = {
        {0, static_cast<uint32_t>(GpuBindResourceType::Buffer), 91, 0, 0, 0},
        {1, static_cast<uint32_t>(GpuBindResourceType::Buffer), 92, 0, 0, 0},
    };
    rs.create_bind_group(96, 95, 1, rbe, 2);
    rs.begin_frame();
    rs.set_pipeline(95);
    rs.set_bind_group(1, 96);
    rs.set_vertex_buffer(0, 90);
    rs.draw(3);
    rs.present();
    run(ex, rs);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().reflected_render_pipelines == before.reflected_render_pipelines + 1);
    CHECK(ex.stats().reflected_vertex_layouts == before.reflected_vertex_layouts + 1);
    CHECK(ex.stats().reflection_vertex_fallbacks == before.reflection_vertex_fallbacks);
    CHECK(ex.stats().draws == before.draws + 1 && ex.stats().draws_skipped == before.draws_skipped);
    CHECK(frame_is(ex, {0, 255, 0, 255}));

    // A merge conflict (uniform in the vertex stage, storage in the fragment stage at the same slot) is
    // the worker's thrown BindingConflictError: an op error, no pipeline, no draw.
    // (Control: a draw on a pipeline id that does not exist costs what it costs; the rejected layout adds one.)
    const uint64_t op_errors0 = ex.stats().op_errors;
    gpu::CommandEncoder control;
    control.begin_frame();
    control.set_pipeline(98);
    control.draw(3);
    control.present();
    run(ex, control);
    const uint64_t missing_cost = ex.stats().op_errors - op_errors0;
    const uint64_t op_errors = ex.stats().op_errors;
    gpu::CommandEncoder bad;
    bad.create_shader_module(97, GpuShaderStage::Fragment, kReflFsConflict);
    bad.create_render_pipeline(98, 93, 97, sizeof(Vtx));
    bad.begin_frame();
    bad.set_pipeline(98);
    bad.draw(3);
    bad.present();
    run(ex, bad);
    CHECK(ex.stats().op_errors == op_errors + missing_cost + 1);
    CHECK(ex.stats().reflection_layout_errors == 1 && ex.stats().reflection_binding_conflicts == 1);
    CHECK(ex.stats().draws_skipped == before.draws_skipped + 2);  // the control's and the conflict's draw

    // A shader without a record keeps the empty-registry path.
    gpu::CommandEncoder plain;
    plain.create_buffer(99, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    plain.create_shader_module(100, GpuShaderStage::Compute, kCompute);
    plain.create_compute_pipeline(101, 100);
    run(ex, plain);
    CHECK(ex.stats().reflection_unavailable == before.reflection_unavailable + 1);
    ex.wait_idle();
    CHECK(ex.stats().device_errors == 0);
    CHECK(ex.stats().device_lost == 0);
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

// A pipeline whose async creation fails (invalid WGSL) resolves to an error pipeline at first use:
// the draw is dropped, nothing crashes (a null module in the placeholder descriptor used to be a
// host access violation inside Dawn's validation).
void test_failed_async_pipeline_resolves(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_shader_module(80, GpuShaderStage::Vertex, "this is not wgsl");
    setup.create_shader_module(81, GpuShaderStage::Fragment, "this is not wgsl");
    create_layouts(setup, 82, 83, {});
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(84, 80, 81, 83)));
    run(ex, setup);
    gpu::CommandEncoder frame;
    frame.begin_frame();
    auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    frame.set_pipeline(84);
    frame.draw(3);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().async_pipeline_failures > before.async_pipeline_failures);
}

// Live-mode options: pipelines are created asynchronously (awaited at first use, same output) and
// Dawn's compiled blobs persist in an on-disk cache that a second executor reuses.
void test_async_pipelines_and_blob_cache() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "pg6_native_exec_blob_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.blob_cache_root = root.string();
    opts.blob_cache_title = "native-exec-test";
    opts.async_pipelines = true;
    uint64_t first_stores = 0;
    for (int run_index = 0; run_index < 2; ++run_index) {
        NativeExecutor::CreateStatus status;
        std::string error;
        auto ex = NativeExecutor::create(opts, &status, &error);
        if (!ex) {
            CHECK(status == NativeExecutor::CreateStatus::NoAdapter);
            return;
        }
        CHECK(ex->stats().blob_cache_enabled);
        test_textured_draw(*ex);  // checks the frame pixels: async creation must not change output
        test_compute(*ex);
        test_auto_layout_remap(*ex);
        ex->wait_idle();
        const ExecutorStats& st = ex->stats();
        CHECK(st.async_pipelines > 0);
        CHECK(st.async_pipeline_failures == 0);
        CHECK(st.device_errors == 0);
        if (run_index == 0) test_failed_async_pipeline_resolves(*ex);
        CHECK(ex->stats_json().find("\"blob_cache\": {\"enabled\": true") != std::string::npos);
        std::printf("blob cache run %d: hits=%llu misses=%llu stores=%llu entries=%llu bytes=%llu (async waits %llu)\n",
                    run_index, static_cast<unsigned long long>(st.blob_hits), static_cast<unsigned long long>(st.blob_misses),
                    static_cast<unsigned long long>(st.blob_stores), static_cast<unsigned long long>(st.blob_entries),
                    static_cast<unsigned long long>(st.blob_total_bytes), static_cast<unsigned long long>(st.async_pipeline_waits));
        if (run_index == 0) {
            first_stores = st.blob_stores;
        } else if (first_stores > 0) {
            // The backend wrote blobs the first time: the second device must be served from disk.
            CHECK(st.blob_hits > 0);
        }
    }
    fs::remove_all(root, ec);
}

void test_depth_texture_float_layout(NativeExecutor& ex) {
    // A float layout entry with a depth texture bound: the executor binds the entry's dummy instead
    // of letting Dawn reject the bind group; the 'depth' layout entry takes the real view.
    const uint64_t before = ex.stats().bind_group_depth_layout_dummies;
    const uint64_t twins_before = ex.stats().depth_twins_created, copies_before = ex.stats().depth_float_copies;
    const uint64_t errors_before = ex.stats().device_errors;
    gpu::CommandEncoder e;
    e.begin_frame();
    emit(e, GpuOpcode::CreateTexture2,
         bytes_of(texture2(900, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::RenderAttachment, 48)));
    create_layouts(e, 901, 902, {layout_entry(0, 2, GpuBindingLayoutType::Texture, 0, 2)});
    create_bind_group(e, 903, 901, {bg_entry(0, GpuBindResourceType::TextureView, 900)});
    create_layouts(e, 904, 905, {layout_entry(0, 2, GpuBindingLayoutType::Texture, 2, 2)});
    create_bind_group(e, 906, 904, {bg_entry(0, GpuBindResourceType::TextureView, 900)});
    // A pass begins: a stale depth twin is re-copied into the encoder before it.
    GpuBeginRenderPassPayload bp = swapchain_pass(GpuLoadOp::Clear, 0.0f, 0.0f, 0.0f, 1.0f);
    e.emit_packet(GpuOpcode::BeginRenderPass, &bp, sizeof bp);
    e.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    e.present();
    run(ex, e);
    ex.wait_idle();
    const ExecutorStats& st = ex.stats();
    CHECK(st.bind_groups >= 2);
    CHECK(st.device_errors == errors_before);
    std::printf("depth-as-float: twins=%llu copies=%llu dummies=%llu\n", (unsigned long long)st.depth_twins_created,
                (unsigned long long)st.depth_float_copies, (unsigned long long)st.bind_group_depth_layout_dummies);
    // The float entry always reads a real twin: r32float, or r16float without float32-filterable.
    CHECK(st.depth_twins_created == twins_before + 1);
    CHECK(st.bind_group_depth_layout_dummies == before);
    CHECK(st.depth_float_copies > copies_before);
}

// ---- depth read variants ----
// A depth texture (cleared by an explicit depth-only pass) read through a layout entry of each sample
// type; the fragment shader turns the value into a color, the swapchain pixel is checked. Every step
// re-clears the depth, so a stale twin (invalidation) would show the previous step's value.
struct DepthStep {
    float depth;
    uint32_t stencil;
    int r, g, b;  // expected pixel (tolerance `tol`)
};

const char* kDepthVs = R"(
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
)";

bool depth_read_case(NativeExecutor& ex, const char* name, uint32_t base, uint8_t format, uint32_t samples,
                     uint8_t layout_type, bool layout_ms, uint8_t view_aspect, const std::string& fs_body,
                     const std::vector<DepthStep>& steps, int tol) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    GpuCreateTexture2Payload ct = texture2(base, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::RenderAttachment, format);
    ct.sample_count = samples;
    emit(setup, GpuOpcode::CreateTexture2, bytes_of(ct));
    GpuBindGroupLayoutEntry le = layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture, layout_type, 2);
    if (layout_ms) le.flags |= 2;
    create_layouts(setup, base + 1, base + 2, {le});
    GpuBindGroupEntry2 be = bg_entry(0, GpuBindResourceType::TextureView, base);
    be.view.aspect = view_aspect;
    create_bind_group(setup, base + 3, base + 1, {be});
    const std::string src = std::string(kDepthVs) + fs_body;
    setup.create_shader_module(base + 4, GpuShaderStage::Vertex, src.c_str());
    setup.create_shader_module(base + 5, GpuShaderStage::Fragment, src.c_str());
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(base + 6, base + 4, base + 5, base + 2)));
    run(ex, setup);
    bool ok = true;
    const bool has_stencil = format == 47 || format == 49;
    for (const DepthStep& st : steps) {
        gpu::CommandEncoder frame;
        frame.begin_frame();
        GpuBeginRenderPassPayload dp;
        std::memset(&dp, 0, sizeof dp);
        dp.depth_stencil.texture_id = base;
        dp.depth_stencil.depth_load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
        dp.depth_stencil.depth_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
        dp.depth_stencil.depth_clear_value = st.depth;
        if (has_stencil) {
            dp.depth_stencil.stencil_load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
            dp.depth_stencil.stencil_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
            dp.depth_stencil.stencil_clear_value = st.stencil;
        }
        frame.emit_packet(GpuOpcode::BeginRenderPass, &dp, sizeof dp);
        frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
        GpuBeginRenderPassPayload pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
        frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
        frame.set_pipeline(base + 6);
        set_bind_group2(frame, 0, base + 3);
        frame.draw(3);
        frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
        frame.present();
        run(ex, frame);
        ex.wait_idle();
        ex.poll();
        FrameImage img;
        std::string err;
        const bool got = ex.read_last_frame(&img, &err);
        const bool px = got && px_near(img, 1, 1, st.b, st.g, st.r, 255, tol);
        if (!px) std::fprintf(stderr, "  depth read case '%s' (depth %.3f stencil %u)\n", name, st.depth, st.stencil);
        ok = ok && px;
    }
    const ExecutorStats& after = ex.stats();
    if (after.device_errors != before.device_errors) std::fprintf(stderr, "  depth read case '%s': device errors\n", name);
    ok = ok && after.device_errors == before.device_errors && after.packet_errors == before.packet_errors;
    CHECK(ok);
    return ok;
}

void test_depth_read_variants(NativeExecutor& ex) {
    // The raw bytes 31..8 of the u32 -> r, g, b.
    const std::string bytes_u =
        "  return vec4f(f32((v >> 24u) & 255u) / 255.0, f32((v >> 16u) & 255u) / 255.0, f32((v >> 8u) & 255u) / 255.0, 1.0);\n}\n";
    const ExecutorStats s0 = ex.stats();

    // float entry: the depth value itself (r32float; r16float without float32-filterable, hence tolerance 2)
    depth_read_case(ex, "float entry / depth32float", 1000, 48, 1, 0, false, 0,
                    "@group(0) @binding(0) var t: texture_2d<f32>;\n@fragment fn fs_main() -> @location(0) vec4f {\n"
                    "  return vec4f(textureLoad(t, vec2i(1, 1), 0).x, 0.0, 0.0, 1.0);\n}\n",
                    {{0.25f, 0, 64, 0, 0}, {0.75f, 0, 191, 0, 0}, {0.5f, 0, 128, 0, 0}}, 2);
    // uint entry: depth32float's f32 bit pattern (D3D12 R32_UINT view of the R32_TYPELESS family)
    const std::string u32_src = "@group(0) @binding(0) var t: texture_2d<u32>;\n@fragment fn fs_main() -> @location(0) vec4f {\n"
                                "  let v = textureLoad(t, vec2i(1, 1), 0).x;\n" + bytes_u;
    depth_read_case(ex, "uint entry / depth32float bits", 1010, 48, 1, 4, false, 0, u32_src,
                    {{0.5f, 0, 63, 0, 0}, {0.25f, 0, 62, 128, 0}, {1.0f, 0, 63, 128, 0}}, 1);
    // sint entry: the same 32 bits as i32
    depth_read_case(ex, "sint entry / depth32float bits", 1020, 48, 1, 3, false, 0,
                    "@group(0) @binding(0) var t: texture_2d<i32>;\n@fragment fn fs_main() -> @location(0) vec4f {\n"
                    "  let v = bitcast<u32>(textureLoad(t, vec2i(1, 1), 0).x);\n" + bytes_u,
                    {{0.5f, 0, 63, 0, 0}, {0.25f, 0, 62, 128, 0}}, 1);
    // depth16unorm / depth24plus through uint: the raw unorm (1.0 -> all ones, 0.0 -> 0)
    depth_read_case(ex, "uint entry / depth16unorm raw", 1030, 45, 1, 4, false, 0, u32_src,
                    {{1.0f, 0, 0, 0, 255}, {0.0f, 0, 0, 0, 0}, {1.0f, 0, 0, 0, 255}}, 1);
    depth_read_case(ex, "uint entry / depth24plus raw", 1040, 46, 1, 4, false, 0, u32_src,
                    {{1.0f, 0, 0, 255, 255}, {0.0f, 0, 0, 0, 0}}, 1);
    // stencil plane of depth24plus-stencil8 / depth32float-stencil8 through a uint entry (direct stencil-aspect view)
    const std::string stencil_src = "@group(0) @binding(0) var t: texture_2d<u32>;\n@fragment fn fs_main() -> @location(0) vec4f {\n"
                                    "  return vec4f(f32(textureLoad(t, vec2i(1, 1), 0).x) / 255.0, 0.0, 0.0, 1.0);\n}\n";
    depth_read_case(ex, "uint entry / stencil aspect (24+8)", 1050, 47, 1, 4, false, 2, stencil_src,
                    {{0.5f, 7, 7, 0, 0}, {0.5f, 200, 200, 0, 0}}, 1);
    depth_read_case(ex, "uint entry / all aspect (32+8)", 1060, 49, 1, 4, false, 0, stencil_src,
                    {{0.5f, 9, 9, 0, 0}, {0.5f, 33, 33, 0, 0}}, 1);
    // multisampled depth, unfilterable-float multisampled entry (and a 'float' ms entry, created unfilterable): per-sample loads
    const std::string ms_src = "@group(0) @binding(0) var t: texture_multisampled_2d<f32>;\n@fragment fn fs_main() -> @location(0) vec4f {\n"
                               "  return vec4f(textureLoad(t, vec2i(1, 1), 2).x, textureLoad(t, vec2i(1, 1), 3).x, 0.0, 1.0);\n}\n";
    depth_read_case(ex, "unfilterable ms entry / ms depth", 1070, 48, 4, 1, true, 0, ms_src,
                    {{0.25f, 0, 64, 64, 0}, {0.75f, 0, 191, 191, 0}}, 2);
    depth_read_case(ex, "float ms entry / ms depth", 1080, 48, 4, 0, true, 0, ms_src, {{0.5f, 0, 128, 128, 0}}, 2);

    const ExecutorStats& s1 = ex.stats();
    std::printf("depth variants: twins=%llu int=%llu half=%llu stencil_direct=%llu copies=%llu dummies=%llu\n",
                (unsigned long long)(s1.depth_twins_created - s0.depth_twins_created),
                (unsigned long long)(s1.depth_int_twins_created - s0.depth_int_twins_created),
                (unsigned long long)(s1.depth_half_twins_created - s0.depth_half_twins_created),
                (unsigned long long)(s1.depth_stencil_direct_binds - s0.depth_stencil_direct_binds),
                (unsigned long long)(s1.depth_float_copies - s0.depth_float_copies),
                (unsigned long long)(s1.bind_group_depth_layout_dummies - s0.bind_group_depth_layout_dummies));
    CHECK(s1.depth_int_twins_created == s0.depth_int_twins_created + 4);  // u32 x3 (32/16/24) + s32
    CHECK(s1.depth_stencil_direct_binds == s0.depth_stencil_direct_binds + 2);
    CHECK(s1.bind_group_depth_layout_dummies == s0.bind_group_depth_layout_dummies);  // every case read real data
}

// A device without float32-filterable (the feature is withheld): a float entry reads the depth through a
// real r16float twin (the only filterable float format), so the value is correct to half precision.
void test_depth_read_no_float32_filterable() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.disable_float32_filterable = true;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        std::fprintf(stderr, "FAIL: no-float32-filterable NativeExecutor::create: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    const ExecutorStats s0 = ex->stats();
    depth_read_case(*ex, "float entry / depth32float / no float32-filterable", 1100, 48, 1, 0, false, 0,
                    "@group(0) @binding(0) var t: texture_2d<f32>;\n@fragment fn fs_main() -> @location(0) vec4f {\n"
                    "  return vec4f(textureLoad(t, vec2i(1, 1), 0).x, 0.0, 0.0, 1.0);\n}\n",
                    {{0.25f, 0, 64, 0, 0}, {0.75f, 0, 191, 0, 0}, {0.5f, 0, 128, 0, 0}}, 2);
    const ExecutorStats& s1 = ex->stats();
    CHECK(s1.depth_half_twins_created == s0.depth_half_twins_created + 1);
    CHECK(s1.bind_group_depth_layout_dummies == s0.bind_group_depth_layout_dummies);
    // the uint path does not depend on the feature
    depth_read_case(*ex, "uint entry / depth32float bits / no float32-filterable", 1110, 48, 1, 4, false, 0,
                    "@group(0) @binding(0) var t: texture_2d<u32>;\n@fragment fn fs_main() -> @location(0) vec4f {\n"
                    "  let v = textureLoad(t, vec2i(1, 1), 0).x;\n"
                    "  return vec4f(f32((v >> 24u) & 255u) / 255.0, f32((v >> 16u) & 255u) / 255.0, 0.0, 1.0);\n}\n",
                    {{0.5f, 0, 63, 0, 0}, {0.25f, 0, 62, 128, 0}}, 1);
    std::printf("no float32-filterable: half twins=%llu\n", (unsigned long long)s1.depth_half_twins_created);
}

// --screenshot-at on a headless executor: present 1 is read back asynchronously, converted from the
// bgra8 swapchain and written as a PNG whose pixels are the frame (no letterbox, no blit re-encode).
void test_screenshot() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "pg6_shot_gpu_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.screenshot_at = {1};
    opts.screenshot_dir = dir.string();
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        CHECK(status == NativeExecutor::CreateStatus::NoAdapter);
        return;
    }
    for (int i = 0; i < 3; ++i) {
        gpu::CommandEncoder f;
        f.begin_frame();
        f.clear_color(i == 1 ? 1.0f : 0.0f, i == 1 ? 0.5f : 1.0f, 0.0f, 1.0f);  // present 1 is orange, others green
        f.present();
        run(*ex, f);
    }
    ex->wait_idle();
    CHECK(ex->stats().screenshots_saved == 1 && ex->stats().screenshots_failed == 0);
    const std::string path = ex->last_screenshot_path();
    CHECK(!path.empty());
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> png((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> rgb;
    CHECK(png_test_decode_rgb(png, &w, &h, &rgb));
    CHECK(w == kW && h == kH && rgb.size() == size_t{kW} * kH * 3);
    bool all = !rgb.empty();
    for (size_t i = 0; i + 2 < rgb.size(); i += 3) all = all && rgb[i] == 255 && rgb[i + 1] >= 126 && rgb[i + 1] <= 129 && rgb[i + 2] == 0;
    CHECK(all);
    ex.reset();
    in.close();
    ec.clear();
    std::filesystem::remove_all(dir, ec);
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
    test_reflection();
    CHECK(ex->stats().device_lost == 0);
    test_timers();
    test_blit();
    test_depth_texture_float_layout(*ex);
    test_depth_read_variants(*ex);
    test_depth_read_no_float32_filterable();
    test_async_pipelines_and_blob_cache();
    test_screenshot();
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    ex.reset();
    std::remove(kPackPath);
    std::printf("test_native_executor: ok\n");
    return 0;
}
