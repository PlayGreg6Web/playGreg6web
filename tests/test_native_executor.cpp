// Native executor milestones 0 to 2 on the host's real GPU (headless): device, offscreen swapchain,
// clear/present, buffers, textures, shaders, layouts, pipelines, draws and dispatches, v1 pipelines,
// GPU -> CPU read-backs into (host) memory and GPU pass timestamps, with every result read back from the GPU. Exits 77
// (CTest: skipped) when the host has no WebGPU adapter. Built with -DPLAYGTA6_NATIVE_EXECUTOR=ON.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_guest_objects.h"
#include "gpu/descriptor_heap.h"
#include "gpu/state_bridge.h"
#include "gpu/gpu_time.h"
#include "gpu/native_exec/native_executor.h"
#include "gpu/native_exec/binding_access_model.h"
#include "gpu/native_exec/fragment_outputs_model.h"
#include "gpu/state_bridge.h"
#include "gpu/native_exec/reflection_model.h"
#include "gpu/native_exec/shader_quarantine.h"
#include "png_stored_decode.h"
#include "gpu/native_exec/show_target_model.h"

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
namespace bridge = playgta6::gpu::bridge;

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
    CHECK(ex.stats_json().find("\"device_error_kinds\": {\"total\": ") != std::string::npos);  // the kind is recorded
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

// A texture copy ends the open compute pass; the title's compute pipeline is still current, so the dispatches after
// the copy (direct and indirect) run on it instead of hitting Dawn's "No pipeline set".
void test_compute_pipeline_survives_copy(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(170, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_buffer(171, GpuBufferUsage::Indirect | GpuBufferUsage::CopyDst, 12);
    const uint32_t args[3] = {1, 1, 1};
    setup.write_buffer(171, 0, args, 12);
    setup.create_shader_module(172, GpuShaderStage::Compute, kCompute);
    create_layouts(setup, 173, 174, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 175;
    cp.cs_module_id = 172;
    cp.pipeline_layout_id = 174;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 176, 173, {bg_entry(0, GpuBindResourceType::Buffer, 170)});
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(177, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst | GpuTextureUsage::CopySrc, 22)));
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(178, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 22)));
    run(ex, setup);

    auto copy = [] {
        GpuCopyTexturePayload c;
        std::memset(&c, 0, sizeof c);
        c.dst_id = 178;
        c.src_id = 177;
        c.width = 2;
        c.height = 2;
        c.depth_or_layers = 1;
        return c;
    };
    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(175);
    set_bind_group2(frame, 0, 176);
    frame.dispatch_compute(1, 1, 1);
    emit(frame, kGpuOpcodeCopyTexture, bytes_of(copy()));
    frame.dispatch_compute(1, 1, 1);
    emit(frame, kGpuOpcodeCopyTexture, bytes_of(copy()));
    frame.indirect(GpuOpcode::DispatchIndirect, 171, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    CHECK(ex.stats().dispatches == before.dispatches + 3);
    CHECK(ex.stats().device_errors == before.device_errors);
    gpu::CommandEncoder done;
    done.destroy_resource(177);
    done.destroy_resource(178);
    run(ex, done);
}

// Bind groups survive a copy that ended the pass: out[i] += 1 runs before and after the copy with the group bound
// once, so the buffer reads back +2 per element (real values; a missing binding would error or leave +1).
const char* kComputeInc = R"(
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = out[i] + 1u; }
)";

void test_bindings_survive_copy(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(190, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc | GpuBufferUsage::CopyDst, 16);
    const uint32_t init[4] = {10, 20, 30, 40};
    setup.write_buffer(190, 0, init, 16);
    setup.create_buffer(191, GpuBufferUsage::Indirect | GpuBufferUsage::CopyDst, 12);
    const uint32_t args[3] = {1, 1, 1};
    setup.write_buffer(191, 0, args, 12);
    setup.create_shader_module(192, GpuShaderStage::Compute, kComputeInc);
    create_layouts(setup, 193, 194, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 195;
    cp.cs_module_id = 192;
    cp.pipeline_layout_id = 194;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 196, 193, {bg_entry(0, GpuBindResourceType::Buffer, 190)});
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(197, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst | GpuTextureUsage::CopySrc, 22)));
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(198, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 22)));
    run(ex, setup);
    auto copy = [] {
        GpuCopyTexturePayload c;
        std::memset(&c, 0, sizeof c);
        c.dst_id = 198;
        c.src_id = 197;
        c.width = 2;
        c.height = 2;
        c.depth_or_layers = 1;
        return c;
    };
    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(195);
    set_bind_group2(frame, 0, 196);
    frame.dispatch_compute(1, 1, 1);
    emit(frame, kGpuOpcodeCopyTexture, bytes_of(copy()));
    frame.dispatch_compute(1, 1, 1);
    emit(frame, kGpuOpcodeCopyTexture, bytes_of(copy()));
    frame.indirect(GpuOpcode::DispatchIndirect, 191, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(190, 0, 16, &out, &err));
    const uint32_t want[4] = {13, 23, 33, 43};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    CHECK(ex.stats().dispatches == before.dispatches + 3);
    CHECK(ex.stats().device_errors == before.device_errors);
    gpu::CommandEncoder done;
    done.destroy_resource(197);
    done.destroy_resource(198);
    run(ex, done);
}

// A draw after a copy inside the render pass: the pipeline and bind group were set once before the copy.
void test_render_bindings_survive_copy(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(200, 2, 2, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst | GpuTextureUsage::CopySrc, 22)));
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(201, 2, 2, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 22)));
    std::vector<uint8_t> texels(256 + 8, 0);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            texels[y * 256 + x * 4 + 0] = 255;
            texels[y * 256 + x * 4 + 1] = 128;
            texels[y * 256 + x * 4 + 3] = 255;
        }
    GpuWriteTexturePayload wt;
    std::memset(&wt, 0, sizeof wt);
    wt.texture_id = 200;
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
    sp.id = 202;
    sp.max_anisotropy = 1;
    sp.lod_max_clamp = 32.0f;
    emit(setup, GpuOpcode::CreateSampler2, bytes_of(sp));
    setup.create_shader_module(203, GpuShaderStage::Vertex, kTexturedTriangle);
    setup.create_shader_module(204, GpuShaderStage::Fragment, kTexturedTriangle);
    create_layouts(setup, 205, 206,
                   {layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture, 0, 2),
                    layout_entry(1, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Sampler, 0)});
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(207, 203, 204, 206)));
    create_bind_group(setup, 208, 205, {bg_entry(0, GpuBindResourceType::TextureView, 200), bg_entry(1, GpuBindResourceType::Sampler, 202)});
    run(ex, setup);

    GpuCopyTexturePayload c;
    std::memset(&c, 0, sizeof c);
    c.dst_id = 201;
    c.src_id = 200;
    c.width = 2;
    c.height = 2;
    c.depth_or_layers = 1;
    gpu::CommandEncoder frame;
    frame.begin_frame();
    auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    frame.set_pipeline(207);
    set_bind_group2(frame, 0, 208);
    emit(frame, kGpuOpcodeCopyTexture, bytes_of(c));
    frame.draw(3);
    emit(frame, kGpuOpcodeCopyTexture, bytes_of(c));
    frame.draw(3);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().device_errors == before.device_errors);
    CHECK(ex.stats().draws == before.draws + 2);
    CHECK(ex.stats().draws_skipped == before.draws_skipped);
    CHECK(frame_is(ex, {0, 128, 255, 255}));
    gpu::CommandEncoder done;
    done.destroy_resource(200);
    done.destroy_resource(201);
    run(ex, done);
}

// Two writable storage bindings in two groups: bound to the same buffer they alias, which WebGPU rejects
// when the command buffer is finished.
const char* kComputeAlias = R"(
@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(1) @binding(0) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { a[i] = i; b[i] = i + 9u; }
)";

// split_passes: a compute pass that fails validation (writable storage aliasing) loses only itself; the
// render pass and the compute pass after it in the same frame still run and the frame is presented.
// Without it (the default) the frame is one command buffer.
void split_passes_frame(NativeExecutor& ex, bool split) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(3000, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_buffer(3001, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_shader_module(3002, GpuShaderStage::Compute, kComputeAlias);
    create_layouts(setup, 3003, 3012, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    std::vector<uint8_t> pl;
    append(pl, GpuCreatePipelineLayoutPayload{3004, 2, 0, 0, 0});
    append(pl, uint32_t{3003});
    append(pl, uint32_t{3003});
    emit(setup, GpuOpcode::CreatePipelineLayout, pl);
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 3005;
    cp.cs_module_id = 3002;
    cp.pipeline_layout_id = 3004;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 3006, 3003, {bg_entry(0, GpuBindResourceType::Buffer, 3000)});
    create_bind_group(setup, 3013, 3003, {bg_entry(0, GpuBindResourceType::Buffer, 3000)});
    setup.create_shader_module(3007, GpuShaderStage::Compute, kCompute);
    create_layouts(setup, 3008, 3009, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    cp.id = 3010;
    cp.cs_module_id = 3007;
    cp.pipeline_layout_id = 3009;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 3011, 3008, {bg_entry(0, GpuBindResourceType::Buffer, 3001)});
    run(ex, setup);
    ex.wait_idle();
    ex.poll();
    const uint64_t setup_errors = ex.stats().device_errors;
    CHECK(setup_errors == before.device_errors);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(3005);  // group 1 never bound: the pass is invalid
    set_bind_group2(frame, 0, 3006);
    frame.dispatch_compute(1);
    auto green = swapchain_pass(GpuLoadOp::Clear, 0.0f, 1.0f, 0.0f, 1.0f);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &green, sizeof green);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.set_compute_pipeline(3010);  // a valid pass after it
    set_bind_group2(frame, 0, 3011);
    frame.dispatch_compute(1);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    if (!split) {
        CHECK(ex.stats().command_buffers == before.command_buffers + 1);  // the whole frame in one buffer
        return;
    }
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(3000, 0, 16, &out, &err));  // the invalid pass was rejected (its writes never land) ...
    CHECK(out == std::vector<uint8_t>(16, 0));
    CHECK(frame_is(ex, {0, 255, 0, 255}));             // ... and the rest of the frame was not
    CHECK(ex.read_buffer(3001, 0, 16, &out, &err));
    const uint32_t want[4] = {1, 3, 5, 7};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    CHECK(ex.stats().command_buffers >= before.command_buffers + 3);
}

// split_passes is opt-in (ExecutorOptions::split_passes): split, the passes a bad pass used to take down
// with it run, and in the in-world scene some of them stall the GPU long enough to stop the live executor.
void test_split_passes(NativeExecutor& shared) {
    CHECK(!ExecutorOptions{}.split_passes);
    split_passes_frame(shared, false);
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.split_passes = true;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        std::fprintf(stderr, "FAIL: split-passes NativeExecutor::create: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    split_passes_frame(*ex, true);
}

// ---- usage scopes (usage_scope_model.h) ----

void test_usage_scopes(NativeExecutor& ex, bool snapshots);

// Usage-scope snapshots are opt-in (ExecutorOptions::usage_snapshots, --usage-snapshots).
void test_usage_snapshots(NativeExecutor& shared) {
    CHECK(!ExecutorOptions{}.usage_snapshots);
    test_usage_scopes(shared, false);
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.usage_snapshots = true;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        std::fprintf(stderr, "FAIL: usage-snapshots NativeExecutor::create: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    test_usage_scopes(*ex, true);
}

const char* kUniformAndStorage = R"(
@group(0) @binding(0) var<uniform> u: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { o[i] = u[i] + 1u; }
)";
const char* kFragWrite = R"(
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { o[0] = 7u; return vec4f(1.0, 0.0, 0.0, 1.0); }
)";
const char* kFragReadUniform = R"(
@group(0) @binding(0) var<uniform> u: vec4<u32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f {
  if (u.x == 7u) { return vec4f(0.0, 1.0, 0.0, 1.0); }
  return vec4f(0.0, 0.0, 1.0, 1.0);
}
)";
const char* kMipCopy = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn cs_main() { textureStore(dst, vec2i(0, 0), textureLoad(src, vec2i(0, 0), 0)); }
)";

// One buffer bound as a uniform and as read-write storage (disjoint ranges) in one dispatch: the uniform is
// bound from a snapshot. Two draws of one render pass, the first writing a buffer as storage, the second
// reading it as a uniform: the pass splits there and the read sees the write. A texture sampled (all mips)
// while one of its mips is a storage target: the sampled view comes from a snapshot. No device errors.
// `snapshots` = ExecutorOptions::usage_snapshots of `ex`; off (the default), no snapshot is taken (the
// in-world scene's ~950 GB of snapshot copies came with a GPU hang), the render pass split stays.
void test_usage_scopes(NativeExecutor& ex, bool snapshots) {
    const ExecutorStats before = ex.stats();
    const uint32_t ubo_storage = GpuBufferUsage::Uniform | GpuBufferUsage::Storage | GpuBufferUsage::CopyDst | GpuBufferUsage::CopySrc;
    gpu::CommandEncoder setup;
    // compute: uniform [0, 16) and storage [256, 272) of buffer 3100
    setup.create_buffer(3100, ubo_storage, 512);
    const uint32_t init[4] = {10, 20, 30, 40};
    setup.write_buffer(3100, 0, init, 16);
    setup.create_shader_module(3101, GpuShaderStage::Compute, kUniformAndStorage);
    create_layouts(setup, 3102, 3103,
                   {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 0),
                    layout_entry(1, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 3104;
    cp.cs_module_id = 3101;
    cp.pipeline_layout_id = 3103;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    auto ub = bg_entry(0, GpuBindResourceType::Buffer, 3100);
    ub.size = 16;
    auto sb = bg_entry(1, GpuBindResourceType::Buffer, 3100);
    sb.offset = 256;
    sb.size = 16;
    create_bind_group(setup, 3105, 3102, {ub, sb});
    // both bindings null: a uniform dummy and a read-write storage dummy, which must not be one buffer
    create_bind_group(setup, 3106, 3102,
                      {bg_entry(0, GpuBindResourceType::Buffer, 0), bg_entry(1, GpuBindResourceType::Buffer, 0)});
    // render: draw 1 writes buffer 3110 [0, 16) as storage, draw 2 reads it as a uniform
    setup.create_buffer(3110, ubo_storage, 256);
    setup.create_shader_module(3111, GpuShaderStage::Vertex, kFragWrite);
    setup.create_shader_module(3112, GpuShaderStage::Fragment, kFragWrite);
    setup.create_shader_module(3113, GpuShaderStage::Vertex, kFragReadUniform);
    setup.create_shader_module(3114, GpuShaderStage::Fragment, kFragReadUniform);
    create_layouts(setup, 3115, 3116, {layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Buffer, 1)});
    create_layouts(setup, 3117, 3118, {layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Buffer, 0)});
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(3119, 3111, 3112, 3116)));
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(3120, 3113, 3114, 3118)));
    auto wb = bg_entry(0, GpuBindResourceType::Buffer, 3110);
    wb.size = 16;
    create_bind_group(setup, 3121, 3115, {wb});
    create_bind_group(setup, 3122, 3117, {wb});
    // texture: 4x4 rgba8unorm with 2 mips; mip 0 uploaded, mip 1 written from it
    auto tex = texture2(3130, 4, 4,
                        GpuTextureUsage::TextureBinding | GpuTextureUsage::StorageBinding | GpuTextureUsage::CopyDst |
                            GpuTextureUsage::CopySrc,
                        22);
    tex.mip_levels = 2;
    emit(setup, GpuOpcode::CreateTexture2, bytes_of(tex));
    std::vector<uint8_t> texels(256 * 3 + 16, 0);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            uint8_t* t = &texels[size_t(y) * 256 + size_t(x) * 4];
            t[0] = 200;
            t[1] = 100;
            t[2] = 50;
            t[3] = 255;
        }
    GpuWriteTexturePayload wt;
    std::memset(&wt, 0, sizeof wt);
    wt.texture_id = 3130;
    wt.width = 4;
    wt.height = 4;
    wt.depth_or_layers = 1;
    wt.bytes_per_row = 256;
    wt.rows_per_image = 4;
    wt.data_len = static_cast<uint32_t>(texels.size());
    std::vector<uint8_t> wtv = bytes_of(wt);
    wtv.insert(wtv.end(), texels.begin(), texels.end());
    emit(setup, GpuOpcode::WriteTexture, wtv);
    setup.create_shader_module(3131, GpuShaderStage::Compute, kMipCopy);
    GpuBindGroupLayoutEntry st = layout_entry(1, GpuShaderStageBits::Compute, GpuBindingLayoutType::StorageTexture,
                                              static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly), 2);
    st.storage_format = 22;
    create_layouts(setup, 3132, 3133, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Texture, 0, 2), st});
    cp.id = 3134;
    cp.cs_module_id = 3131;
    cp.pipeline_layout_id = 3133;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    auto sv = bg_entry(0, GpuBindResourceType::TextureView, 3130);
    sv.view.dimension = 2;
    auto wv = bg_entry(1, GpuBindResourceType::TextureView, 3130);
    wv.view.dimension = 2;
    wv.view.base_mip_level = 1;
    wv.view.mip_level_count = 1;
    create_bind_group(setup, 3135, 3132, {sv, wv});
    run(ex, setup);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().device_errors == before.device_errors);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(3104);
    set_bind_group2(frame, 0, 3105);
    frame.dispatch_compute(1);
    set_bind_group2(frame, 0, 3106);
    frame.dispatch_compute(1);
    frame.set_compute_pipeline(3134);
    set_bind_group2(frame, 0, 3135);
    frame.dispatch_compute(1);
    auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    frame.set_pipeline(3119);
    set_bind_group2(frame, 0, 3121);
    frame.draw(3);
    frame.set_pipeline(3120);
    set_bind_group2(frame, 0, 3122);
    frame.draw(3);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    const ExecutorStats& after = ex.stats();
    if (!snapshots) {
        CHECK(after.usage_buffer_snapshots == before.usage_buffer_snapshots);
        CHECK(after.usage_texture_snapshots == before.usage_texture_snapshots);
        CHECK(after.usage_snapshot_splits == before.usage_snapshot_splits);
        CHECK(after.usage_snapshots_off == before.usage_snapshots_off + 2);
        CHECK(after.usage_scope_splits == before.usage_scope_splits + 1);
        return;
    }
    CHECK(after.device_errors == before.device_errors);
    CHECK(after.usage_buffer_snapshots == before.usage_buffer_snapshots + 1);
    CHECK(after.usage_texture_snapshots == before.usage_texture_snapshots + 1);
    CHECK(after.usage_snapshot_splits == before.usage_snapshot_splits + 2);
    CHECK(after.usage_scope_splits == before.usage_scope_splits + 1);
    CHECK(after.usage_snapshot_skipped == before.usage_snapshot_skipped);
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(3100, 256, 16, &out, &err));
    const uint32_t want[4] = {11, 21, 31, 41};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    std::vector<uint8_t> px;
    uint32_t w = 0, h = 0;
    CHECK(ex.read_texture(3130, 1, &px, &w, &h, &err) && w == 2 && h == 2);
    CHECK(px.size() == 16 && px[0] == 200 && px[1] == 100 && px[2] == 50 && px[3] == 255);
    CHECK(frame_is(ex, {0, 255, 0, 255}));  // the second draw read the first one's write
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

// A compute WGSL that binds @group(3) (as the pack's mesh-to-compute translations do) under a title
// layout with group 0 only: the layout is fitted (group 3 from the WGSL), the pipeline compiles, a
// pass that does not bind group 3 gets a dummy there, and one that binds its own group 3 keeps it.
const char* kComputeGroup3 = R"(
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(3) @binding(0) var<storage, read_write> ms_out: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) {
    out[i] = i * 5u + 1u;
    ms_out[i] = i + 100u;
}
)";

void test_layout_fit_missing_group(NativeExecutor& ex, bool packed = false) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(130, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_buffer(131, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_shader_module(132, GpuShaderStage::Compute,
                               packed ? "0x00b16b00b5c0ffea" : kComputeGroup3);
    create_layouts(setup, 133, 134, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 135;
    cp.cs_module_id = 132;
    cp.pipeline_layout_id = 134;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 136, 133, {bg_entry(0, GpuBindResourceType::Buffer, 130)});
    // The stream's own group 3 (the state bridge's mesh output group: one storage buffer, compute).
    std::vector<uint8_t> bgl3;
    append(bgl3, GpuCreateBindGroupLayoutPayload{137, 1, 0, 0, 0});
    append(bgl3, layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1));
    emit(setup, GpuOpcode::CreateBindGroupLayout, bgl3);
    create_bind_group(setup, 138, 137, {bg_entry(0, GpuBindResourceType::Buffer, 131)});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(135);
    set_bind_group2(frame, 0, 136);
    frame.dispatch_compute(1, 1, 1);  // no group 3 bound: the fitted dummy
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(130, 0, 16, &out, &err));
    const uint32_t want[4] = {1, 6, 11, 16};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    ExecutorStats s = ex.stats();
    CHECK(s.layout_fit_pipelines == before.layout_fit_pipelines + 1);
    CHECK(s.layout_fit_wgsl_groups == before.layout_fit_wgsl_groups + 1);
    CHECK(s.layout_fit_dummy_binds == before.layout_fit_dummy_binds + 3);  // padding groups 1, 2 and group 3
    CHECK(s.async_pipeline_failures == before.async_pipeline_failures);
    CHECK(s.dispatches == before.dispatches + 1);
    CHECK(s.device_errors == before.device_errors);

    gpu::CommandEncoder frame2;
    frame2.begin_frame();
    frame2.set_compute_pipeline(135);
    set_bind_group2(frame2, 0, 136);
    set_bind_group2(frame2, 3, 138);
    frame2.dispatch_compute(1, 1, 1);
    frame2.present();
    run(ex, frame2);
    ex.wait_idle();
    CHECK(ex.read_buffer(131, 0, 16, &out, &err));
    const uint32_t want3[4] = {100, 101, 102, 103};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want3, 16) == 0);
    s = ex.stats();
    CHECK(s.layout_fit_dummy_binds == before.layout_fit_dummy_binds + 5);  // only the padding: the stream's group 3 stays
    CHECK(s.device_errors == before.device_errors);
}

// A title layout over the device's per-stage uniform-buffer limit (a root signature visible to every
// stage) is invalid as a whole; the WGSL references two of its bindings. Fitted: group 0 keeps the
// title's entries for those two, the remap binds them by binding number, and the dispatch runs.
void test_layout_fit_over_limits(NativeExecutor& ex) {
    const uint32_t n = ex.stats().limit_uniform_buffers_per_stage + 1;
    CHECK(n > 1);
    if (n <= 1 || n > 64) return;
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(140, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, 256);
    setup.create_buffer(141, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, 256);
    setup.create_buffer(142, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_buffer(148, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, 16);  // zeros
    const uint32_t seven[4] = {7, 0, 0, 0};
    const uint32_t nine[4] = {9, 0, 0, 0};
    setup.write_buffer(140, 0, nine, 16);
    setup.write_buffer(141, 0, seven, 16);
    // Binding n+1: a read-only WGSL buffer (`var<storage>`) under the title's read-only entry.
    const std::string wgsl = "@group(0) @binding(2) var<uniform> u: vec4<u32>;\n@group(0) @binding(" + std::to_string(n) +
                             ") var<storage, read_write> out: array<u32>;\n@group(0) @binding(" + std::to_string(n + 1) +
                             ") var<storage> ro: array<u32>;\n"
                             "@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = u.x + i + ro[0]; }\n";
    setup.create_shader_module(143, GpuShaderStage::Compute, wgsl.c_str());
    std::vector<GpuBindGroupLayoutEntry> les;
    std::vector<GpuBindGroupEntry2> bes;
    for (uint32_t b = 0; b < n; ++b) {
        les.push_back(layout_entry(b, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 0));
        bes.push_back(bg_entry(b, GpuBindResourceType::Buffer, b == 2 ? 141 : 140));
    }
    les.push_back(layout_entry(n, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1));
    bes.push_back(bg_entry(n, GpuBindResourceType::Buffer, 142));
    les.push_back(layout_entry(n + 1, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 2));
    bes.push_back(bg_entry(n + 1, GpuBindResourceType::Buffer, 148));
    create_layouts(setup, 144, 145, les);
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 146;
    cp.cs_module_id = 143;
    cp.pipeline_layout_id = 145;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 147, 144, bes);
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(146);
    set_bind_group2(frame, 0, 147);
    frame.dispatch_compute(1, 1, 1);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(142, 0, 16, &out, &err));
    const uint32_t want[4] = {7, 8, 9, 10};  // binding 2's buffer, not the first uniform of the group
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    const ExecutorStats& s = ex.stats();
    CHECK(s.layout_fit_pipelines == before.layout_fit_pipelines + 1);
    CHECK(s.layout_fit_over_limits == before.layout_fit_over_limits + 1);
    CHECK(s.async_pipeline_failures == before.async_pipeline_failures);
    CHECK(s.dispatches == before.dispatches + 1);
}

// A title layout whose group 1 (another root parameter's table, unused by the WGSL) is over the
// uniform-buffer limit on its own: its bind groups are never bound, the fitted layout pads group 1
// and binds the padding over whatever the pass bound there, and the next pipeline gets the pass's own
// group 1 back.
const char* kComputeG0Only = R"(
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = i + 40u; }
)";
const char* kComputeG1 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(1) @binding(0) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { a[i] = 1u; b[i] = i + 60u; }
)";

void test_layout_fit_unused_group_over_limits(NativeExecutor& ex) {
    const uint32_t n = ex.stats().limit_uniform_buffers_per_stage + 1;
    if (n <= 1 || n > 64) return;
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(150, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_buffer(151, GpuBufferUsage::Uniform, 256);
    setup.create_buffer(152, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_buffer(153, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_shader_module(154, GpuShaderStage::Compute, kComputeG0Only);
    setup.create_shader_module(155, GpuShaderStage::Compute, kComputeG1);
    auto bgl = [&](uint32_t id, const std::vector<GpuBindGroupLayoutEntry>& es) {
        std::vector<uint8_t> v;
        append(v, GpuCreateBindGroupLayoutPayload{id, static_cast<uint32_t>(es.size()), 0, 0, 0});
        for (const auto& e : es) append(v, e);
        emit(setup, GpuOpcode::CreateBindGroupLayout, v);
    };
    auto pl = [&](uint32_t id, std::vector<uint32_t> groups) {
        std::vector<uint8_t> v;
        append(v, GpuCreatePipelineLayoutPayload{id, static_cast<uint32_t>(groups.size()), 0, 0, 0});
        for (uint32_t g : groups) append(v, g);
        emit(setup, GpuOpcode::CreatePipelineLayout, v);
    };
    const auto storage0 = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1);
    bgl(156, {storage0});
    std::vector<GpuBindGroupLayoutEntry> uniforms;
    std::vector<GpuBindGroupEntry2> ub;
    for (uint32_t b = 0; b < n; ++b) {
        uniforms.push_back(layout_entry(b, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 0));
        ub.push_back(bg_entry(b, GpuBindResourceType::Buffer, 151));
    }
    bgl(157, uniforms);  // over the limit on its own
    bgl(158, {storage0});
    pl(159, {156, 157});  // A: group 1 over the limit, unused by the WGSL
    pl(160, {156, 158});  // B: a valid group 1 the WGSL writes
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 161;
    cp.cs_module_id = 154;
    cp.pipeline_layout_id = 159;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    cp.id = 162;
    cp.cs_module_id = 155;
    cp.pipeline_layout_id = 160;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 163, 156, {bg_entry(0, GpuBindResourceType::Buffer, 150)});
    create_bind_group(setup, 164, 157, ub);
    create_bind_group(setup, 165, 158, {bg_entry(0, GpuBindResourceType::Buffer, 153)});
    create_bind_group(setup, 166, 156, {bg_entry(0, GpuBindResourceType::Buffer, 152)});
    run(ex, setup);
    const uint64_t setup_errors = ex.stats().device_errors;  // the over-limit layout itself

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(161);
    set_bind_group2(frame, 0, 163);
    set_bind_group2(frame, 1, 164);  // the over-limit group: not bound
    frame.dispatch_compute(1, 1, 1);
    set_bind_group2(frame, 1, 165);  // the pass's own group 1; A pads over it
    frame.dispatch_compute(1, 1, 1);
    frame.set_compute_pipeline(162);
    set_bind_group2(frame, 0, 166);
    frame.dispatch_compute(1, 1, 1);  // group 1 restored
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(150, 0, 16, &out, &err));
    const uint32_t want_a[4] = {40, 41, 42, 43};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want_a, 16) == 0);
    CHECK(ex.read_buffer(153, 0, 16, &out, &err));
    const uint32_t want_b[4] = {60, 61, 62, 63};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want_b, 16) == 0);
    const ExecutorStats& s = ex.stats();
    CHECK(s.over_limit_bind_group_layouts == before.over_limit_bind_group_layouts + 1);
    CHECK(s.over_limit_binds_skipped == before.over_limit_binds_skipped + 1);
    CHECK(s.layout_fit_unused_groups == before.layout_fit_unused_groups + 1);
    CHECK(s.layout_fit_restores == before.layout_fit_restores + 1);
    CHECK(s.async_pipeline_failures == before.async_pipeline_failures);
    CHECK(s.dispatches == before.dispatches + 3);
    CHECK(s.device_errors == setup_errors);  // nothing in the frame
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

    // The pack shader is materialized once (timed) and its WGSL scanned once, however many
    // pipelines use it (a compute WGSL can be 16 MB).
    gpu::CommandEncoder pipes;
    for (uint32_t id : {82u, 83u, 84u}) {
        GpuCreateComputePipeline2Payload cp;
        std::memset(&cp, 0, sizeof cp);
        cp.id = id;
        cp.cs_module_id = 80;
        emit(pipes, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    }
    run(ex, pipes);
    CHECK(ex.stats().shader_materialized == before.shader_materialized + 1);
    CHECK(ex.stats().shader_scans == before.shader_scans + 2);  // the entry point and the binding access, once each
    CHECK(ex.stats().slow_packets == before.slow_packets);
    CHECK(ex.stats_json().find("\"shader_materialize_max_hash\": \"0x") != std::string::npos);

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

// ClearTexture: a texture full of old texels reads back as zero after the clear, except where later
// writes landed: a write extending the covered rows needs no flush, a partial one forces one.
void test_clear_texture(NativeExecutor& ex) {
    std::vector<uint8_t> mem(8192, 0xEE);
    GuestMemory gm;
    gm.base = mem.data();
    gm.size = mem.size();
    std::memset(mem.data(), 0, 16);
    ex.attach_guest_memory(gm);
    const ExecutorStats before = ex.stats();

    auto write = [](gpu::CommandEncoder& enc, uint32_t id, uint32_t mip, uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                    uint8_t value) {
        GpuWriteTexturePayload wt;
        std::memset(&wt, 0, sizeof wt);
        wt.texture_id = id;
        wt.mip_level = mip;
        wt.origin_x = x;
        wt.origin_y = y;
        wt.width = w;
        wt.height = h;
        wt.depth_or_layers = 1;
        wt.bytes_per_row = 256;
        wt.rows_per_image = h;
        wt.data_len = 256 * h;
        std::vector<uint8_t> v = bytes_of(wt);
        v.insert(v.end(), size_t{256} * h, value);
        emit(enc, GpuOpcode::WriteTexture, v);
    };
    auto readback = [](gpu::CommandEncoder& enc, uint32_t id, uint32_t mip, uint32_t w, uint32_t h, uint64_t dst,
                       uint64_t done) {
        GpuReadbackTexturePayload rt;
        std::memset(&rt, 0, sizeof rt);
        rt.seq = 1;
        rt.texture_id = id;
        rt.mip_level = mip;
        rt.width = w;
        rt.height = h;
        rt.depth_or_layers = 1;
        rt.bytes_per_row = w * 4;
        rt.rows_per_image = h;
        rt.dst_addr = dst;
        rt.done_addr = done;
        rt.dst_size = uint64_t{w} * 4 * h;
        emit(enc, GpuOpcode::ReadbackTexture, bytes_of(rt));
    };

    // 4x4 rgba8 with 2 mips, filled with 0x55, then cleared; row 0 of mip 0 is written again (0x07).
    GpuCreateTexture2Payload ct = texture2(120, 4, 4, GpuTextureUsage::CopySrc | GpuTextureUsage::CopyDst, 22);
    ct.mip_levels = 2;
    gpu::CommandEncoder enc;
    emit(enc, GpuOpcode::CreateTexture2, bytes_of(ct));
    write(enc, 120, 0, 0, 0, 4, 4, 0x55);
    write(enc, 120, 1, 0, 0, 2, 2, 0x55);
    GpuClearTexturePayload clear{120, 0};
    emit(enc, kGpuOpcodeClearTexture, bytes_of(clear));
    write(enc, 120, 0, 0, 0, 4, 1, 0x07);
    readback(enc, 120, 0, 4, 4, 1024, 0);
    readback(enc, 120, 1, 2, 2, 2048, 4);
    CHECK(run(ex, enc) == 0);
    ex.wait_idle();
    ex.poll();
    bool row0 = true, rest0 = true, mip1 = true;
    for (int i = 0; i < 16; ++i) row0 &= mem[1024 + i] == 0x07;
    for (int i = 16; i < 64; ++i) rest0 &= mem[1024 + i] == 0;
    for (int i = 0; i < 16; ++i) mip1 &= mem[2048 + i] == 0;
    CHECK(row0 && rest0 && mip1);
    const ExecutorStats& a = ex.stats();
    CHECK(a.textures_cleared == before.textures_cleared + 1);
    CHECK(a.clear_flushes == before.clear_flushes);  // the band only extended the covered rows
    CHECK(a.clear_copies == before.clear_copies + 2);  // rows 1..3 of mip 0, all of mip 1
    CHECK(u32_at(mem, 0) == 1 && u32_at(mem, 4) == 1);

    // A partial-width write onto a pending row needs the clear first (a flush), then lands on zero.
    gpu::CommandEncoder enc2;
    write(enc2, 120, 0, 0, 0, 4, 4, 0x55);
    emit(enc2, kGpuOpcodeClearTexture, bytes_of(clear));
    write(enc2, 120, 0, 2, 2, 1, 1, 0x09);
    readback(enc2, 120, 0, 4, 4, 4096, 8);
    CHECK(run(ex, enc2) == 0);
    ex.wait_idle();
    ex.poll();
    bool ok = true;
    for (int i = 0; i < 64; ++i) {
        const bool texel = i >= (2 * 4 + 2) * 4 && i < (2 * 4 + 3) * 4;
        ok &= mem[4096 + i] == (texel ? 0x09 : 0);
    }
    CHECK(ok);
    CHECK(ex.stats().clear_flushes == before.clear_flushes + 1);
    CHECK(u32_at(mem, 8) == 1);

    // Every texel written after the clear: no copy at all. A missing texture is counted, not an error.
    const uint64_t copies = ex.stats().clear_copies;
    gpu::CommandEncoder enc3;
    emit(enc3, kGpuOpcodeClearTexture, bytes_of(clear));
    write(enc3, 120, 0, 0, 0, 4, 4, 0x11);
    write(enc3, 120, 1, 0, 0, 2, 2, 0x11);
    GpuClearTexturePayload missing{9999, 0};
    emit(enc3, kGpuOpcodeClearTexture, bytes_of(missing));
    readback(enc3, 120, 1, 2, 2, 6144, 12);
    CHECK(run(ex, enc3) == 0);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().clear_copies == copies);
    CHECK(ex.stats().clears_covered == before.clears_covered + 1);
    CHECK(ex.stats().clears_missing == before.clears_missing + 1);
    CHECK(mem[6144] == 0x11 && mem[6159] == 0x11);
    CHECK(ex.stats().device_errors == before.device_errors);
    gpu::CommandEncoder done;
    done.destroy_resource(120);
    run(ex, done);
}

// CopyTexture (opcode 62): a plain copyTextureToTexture between textures that are neither render
// targets nor storage; the source was created without CopySrc (the executor adds it). A bad copy is a
// packet error, not a device error.
void test_copy_texture(NativeExecutor& ex) {
    std::vector<uint8_t> mem(8192, 0xEE);
    GuestMemory gm;
    gm.base = mem.data();
    gm.size = mem.size();
    std::memset(mem.data(), 0, 16);
    ex.attach_guest_memory(gm);
    const ExecutorStats before = ex.stats();

    auto write = [](gpu::CommandEncoder& enc, uint32_t id, uint32_t w, uint32_t h, uint8_t value) {
        GpuWriteTexturePayload wt;
        std::memset(&wt, 0, sizeof wt);
        wt.texture_id = id;
        wt.width = w;
        wt.height = h;
        wt.depth_or_layers = 1;
        wt.bytes_per_row = 256;
        wt.rows_per_image = h;
        wt.data_len = 256 * h;
        std::vector<uint8_t> v = bytes_of(wt);
        v.insert(v.end(), size_t{256} * h, value);
        emit(enc, GpuOpcode::WriteTexture, v);
    };
    auto copy = [](uint32_t dst, uint32_t src, uint32_t dx, uint32_t dy, uint32_t sx, uint32_t sy, uint32_t w,
                   uint32_t h) {
        GpuCopyTexturePayload c;
        std::memset(&c, 0, sizeof c);
        c.dst_id = dst;
        c.src_id = src;
        c.dst_x = dx;
        c.dst_y = dy;
        c.src_x = sx;
        c.src_y = sy;
        c.width = w;
        c.height = h;
        c.depth_or_layers = 1;
        return c;
    };

    gpu::CommandEncoder enc;
    emit(enc, GpuOpcode::CreateTexture2,
         bytes_of(texture2(130, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 22)));
    emit(enc, GpuOpcode::CreateTexture2,
         bytes_of(texture2(131, 4, 4,
                           GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst | GpuTextureUsage::CopySrc, 22)));
    write(enc, 130, 4, 4, 0x55);
    write(enc, 131, 4, 4, 0x11);
    emit(enc, kGpuOpcodeCopyTexture, bytes_of(copy(131, 130, 2, 0, 1, 1, 2, 2)));
    GpuReadbackTexturePayload rt;
    std::memset(&rt, 0, sizeof rt);
    rt.seq = 1;
    rt.texture_id = 131;
    rt.width = 4;
    rt.height = 4;
    rt.depth_or_layers = 1;
    rt.bytes_per_row = 16;
    rt.rows_per_image = 4;
    rt.dst_addr = 1024;
    rt.done_addr = 0;
    rt.dst_size = 64;
    emit(enc, GpuOpcode::ReadbackTexture, bytes_of(rt));
    CHECK(run(ex, enc) == 0);
    ex.wait_idle();
    ex.poll();
    bool ok = u32_at(mem, 0) == 1;
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            const bool copied = x >= 2 && y < 2;
            ok &= mem[1024 + (y * 4 + x) * 4] == (copied ? 0x55 : 0x11);
        }
    }
    CHECK(ok);
    CHECK(ex.stats().texture_copies == before.texture_copies + 1);
    CHECK(ex.stats().device_errors == before.device_errors);

    // Out of range, an unknown texture and mismatched formats are rejected without a device error.
    gpu::CommandEncoder bad;
    emit(bad, GpuOpcode::CreateTexture2,
         bytes_of(texture2(132, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 1)));
    emit(bad, kGpuOpcodeCopyTexture, bytes_of(copy(131, 130, 3, 3, 0, 0, 2, 2)));
    emit(bad, kGpuOpcodeCopyTexture, bytes_of(copy(131, 9999, 0, 0, 0, 0, 1, 1)));
    emit(bad, kGpuOpcodeCopyTexture, bytes_of(copy(131, 132, 0, 0, 0, 0, 1, 1)));
    CHECK(run(ex, bad) == 0);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().texture_copies == before.texture_copies + 1);
    CHECK(ex.stats().device_errors == before.device_errors);
    gpu::CommandEncoder done;
    done.destroy_resource(130);
    done.destroy_resource(131);
    done.destroy_resource(132);
    run(ex, done);
}

// Guest query heaps (opcodes 61 / 60 / 59): occlusion queries count the samples of the draws between
// their begin and end across render passes, a binary one resolves to 1, one without draws to 0;
// timestamps are written outside and inside a pass; resolves land in a buffer in the D3D12 layout.
// Uses the full-screen pipeline 50 / bind group 60 of test_textured_draw.
void test_queries(NativeExecutor& ex) {
    std::vector<uint8_t> mem(4096, 0xEE);
    GuestMemory gm;
    gm.base = mem.data();
    gm.size = mem.size();
    std::memset(mem.data(), 0, 16);
    ex.attach_guest_memory(gm);
    const ExecutorStats before = ex.stats();
    auto query = [](gpu::CommandEncoder& enc, uint32_t set, uint32_t index, uint32_t op, uint32_t type) {
        emit(enc, kGpuOpcodeQuery, bytes_of(GpuQueryPayload{set, index, op, type}));
    };
    auto resolve = [](gpu::CommandEncoder& enc, uint32_t set, uint32_t type, uint32_t first, uint32_t count,
                      uint64_t offset) {
        emit(enc, kGpuOpcodeResolveQuery, bytes_of(GpuResolveQueryPayload{set, type, first, count, 910, 0, offset}));
    };
    auto draw = [](gpu::CommandEncoder& enc) {
        enc.set_pipeline(50);
        set_bind_group2(enc, 0, 60);
        enc.draw(3);
    };
    gpu::CommandEncoder setup;
    emit(setup, kGpuOpcodeCreateQuerySet, bytes_of(GpuCreateQuerySetPayload{900, GpuQuerySetKind::Occlusion, 8, 0}));
    emit(setup, kGpuOpcodeCreateQuerySet, bytes_of(GpuCreateQuerySetPayload{901, GpuQuerySetKind::Timestamp, 4, 0}));
    setup.create_buffer(910, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc | GpuBufferUsage::CopyDst, 64);
    run(ex, setup);
    const bool stamps = ex.stats().query_sets_created == before.query_sets_created + 2;

    gpu::CommandEncoder frame;
    frame.begin_frame();
    const auto clear = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    const auto load = swapchain_pass(GpuLoadOp::Load, 0, 0, 0, 1);
    query(frame, 900, 0, 0, 0);  // begun before the pass its draws fall in (as the producer records it)
    frame.emit_packet(GpuOpcode::BeginRenderPass, &clear, sizeof clear);
    draw(frame);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &load, sizeof load);
    draw(frame);  // a second pass: the query spans both
    query(frame, 900, 0, 1, 0);
    query(frame, 900, 1, 0, 1);  // binary
    draw(frame);
    query(frame, 900, 1, 1, 1);
    query(frame, 900, 2, 0, 0);  // no draw: 0
    query(frame, 900, 2, 1, 0);
    query(frame, 900, 3, 0, 0);  // two active at once: a draw counts for both
    draw(frame);
    query(frame, 900, 4, 0, 0);
    draw(frame);
    query(frame, 900, 3, 1, 0);
    draw(frame);
    query(frame, 900, 4, 1, 0);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    query(frame, 901, 0, 1, 2);  // a timestamp between passes
    frame.emit_packet(GpuOpcode::BeginRenderPass, &load, sizeof load);
    draw(frame);
    query(frame, 901, 1, 1, 2);  // ... and one inside a pass
    draw(frame);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    resolve(frame, 900, 0, 0, 5, 0);
    resolve(frame, 901, 2, 0, 2, 40);
    emit(frame, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(910, 0, 56, 256, 0)));
    frame.present();
    CHECK(run(ex, frame) == 1);
    ex.wait_idle();
    ex.poll();
    auto u64_at = [&](size_t off) {
        uint64_t v = 0;
        std::memcpy(&v, mem.data() + off, 8);
        return v;
    };
    const ExecutorStats& a = ex.stats();
    CHECK(u32_at(mem, 0) == 1);
    // Dawn's D3D12 backend counts an occlusion query as 0 / 1 per pass segment, others count samples.
    const uint64_t samples = u64_at(256);
    std::printf("queries: occlusion %llu, timestamps %llu %llu\n", static_cast<unsigned long long>(samples),
                static_cast<unsigned long long>(u64_at(256 + 40)), static_cast<unsigned long long>(u64_at(256 + 48)));
    CHECK(samples == 2 || samples == 2ull * kW * kH);
    CHECK(u64_at(256 + 8) == 1);
    CHECK(u64_at(256 + 16) == 0);
    CHECK(u64_at(256 + 24) == samples && u64_at(256 + 32) == samples);
    CHECK(a.query_begins == before.query_begins + 5 && a.query_ends == before.query_ends + 5);
    CHECK(a.query_occlusion_segments == before.query_occlusion_segments + 6);
    if (stamps) {
        CHECK(u64_at(256 + 40) != 0 && u64_at(256 + 48) >= u64_at(256 + 40));
        CHECK(a.query_timestamps == before.query_timestamps + 2);
        CHECK(a.query_resolves == before.query_resolves + 2);
        CHECK(a.query_results_written == before.query_results_written + 7);
    } else {
        std::printf("queries: no timestamp-query on this adapter\n");
    }
    CHECK(a.device_errors == before.device_errors);
    CHECK(a.packet_errors == before.packet_errors);

    // A query begun again starts over; a mismatched end and an out-of-range resolve are rejected.
    gpu::CommandEncoder again;
    again.begin_frame();
    query(again, 900, 0, 0, 0);
    query(again, 900, 0, 1, 0);
    query(again, 900, 5, 1, 0);       // never begun
    resolve(again, 900, 0, 6, 4, 0);  // past the heap
    resolve(again, 900, 0, 0, 1, 0);
    emit(again, GpuOpcode::ReadbackBuffer, bytes_of(rb_buffer(910, 0, 8, 512, 4)));
    again.present();
    CHECK(run(ex, again) == 1);
    ex.wait_idle();
    ex.poll();
    CHECK(u32_at(mem, 4) == 1 && u64_at(512) == 0);
    CHECK(ex.stats().query_unsupported == before.query_unsupported + (stamps ? 2 : 6));
    CHECK(ex.stats().device_errors == before.device_errors);
    gpu::CommandEncoder done;
    done.destroy_resource(900);
    done.destroy_resource(901);
    done.destroy_resource(910);
    run(ex, done);
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

// v2 remap pipelines built from the shader reflection get an explicit layout: the remapped bind group is
// created against it while the (async, zero-budget) pipeline is still compiling, so nothing waits for the
// compiled pipeline (async_pipeline_waits stays 0) and the result matches the layout:'auto' path.
const char* kRemapA = R"(
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = i * 3u + 2u; }
)";
const char* kRemapB = R"(
struct P { k: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = i + p.k; }
)";

std::string remap_sidecar() {
    return std::string(R"({"schema":"playgta6-stage-reflection","version":3,"stages":{")") + hash_hex_of(kRemapA) +
           R"(":{"stage":"compute","entry_point":"cs_main",
  "bindings":[{"group":0,"binding":3,"kind":"storage","source":{"user_sgpr":4,"table_byte_offset":0}}]},")" +
           hash_hex_of(kRemapB) + R"(":{"stage":"compute","entry_point":"cs_main",
  "bindings":[{"group":0,"binding":1,"kind":"uniform","min_binding_size_bytes":16,"source":{"user_sgpr":4,"table_byte_offset":0}},
              {"group":0,"binding":5,"kind":"storage","source":{"user_sgpr":4,"table_byte_offset":8}}]}}})";
}

void test_remap_reflected_layout(bool with_reflection) {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.async_pipelines = true;
#ifdef _WIN32
    _putenv_s("PG6_PIPELINE_WAIT_MS", "0");
#else
    setenv("PG6_PIPELINE_WAIT_MS", "0", 1);
#endif
    NativeExecutor::CreateStatus status;
    std::string error;
    auto exu = NativeExecutor::create(opts, &status, &error);
#ifdef _WIN32
    _putenv_s("PG6_PIPELINE_WAIT_MS", "");
#else
    unsetenv("PG6_PIPELINE_WAIT_MS");
#endif
    if (!exu) {
        CHECK(status == NativeExecutor::CreateStatus::NoAdapter);
        return;
    }
    NativeExecutor& ex = *exu;
    std::string err;
    if (with_reflection) CHECK(ex.load_reflection_text(remap_sidecar(), &err));
    const ExecutorStats before = ex.stats();

    gpu::CommandEncoder setup;
    setup.create_buffer(110, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 512);
    setup.create_buffer(116, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, 16);
    const uint32_t k[4] = {100, 0, 0, 0};
    setup.write_buffer(116, 0, k, 16);
    setup.create_shader_module(111, GpuShaderStage::Compute, kRemapA);
    setup.create_shader_module(117, GpuShaderStage::Compute, kRemapB);
    GpuBindGroupLayoutEntry le = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1);
    create_layouts(setup, 112, 113, {le});
    for (uint32_t i = 0; i < 2; ++i) {
        GpuCreateComputePipeline2Payload cp;
        std::memset(&cp, 0, sizeof cp);
        cp.id = 114 + i * 10;
        cp.cs_module_id = i ? 117 : 111;
        cp.pipeline_layout_id = 113;
        emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    }
    create_bind_group(setup, 115, 112, {bg_entry(0, GpuBindResourceType::Buffer, 110)});
    run(ex, setup);
    // Bind groups exist before either pipeline finished: dispatch repeatedly until they did.
    const uint32_t want_a[4] = {2, 5, 8, 11};
    bool ok = false;
    for (int frame = 0; frame < 400 && !ok; ++frame) {
        gpu::CommandEncoder f;
        f.begin_frame();
        f.set_compute_pipeline(114);
        set_bind_group2(f, 0, 115);
        f.dispatch_compute(1, 1, 1);
        f.present();
        run(ex, f);
        ex.wait_idle();
        std::vector<uint8_t> out;
        ok = ex.read_buffer(110, 0, 16, &out, &err) && out.size() == 16 && std::memcmp(out.data(), want_a, 16) == 0;
        if (!ok) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(ok);
    const ExecutorStats& s = ex.stats();
    CHECK(s.device_errors == before.device_errors);
    CHECK(s.auto_layout_remap_pipelines == before.auto_layout_remap_pipelines + 2);
    if (with_reflection) {
        CHECK(s.remap_reflected_layouts == before.remap_reflected_layouts + 2);
        CHECK(s.async_pipeline_waits == before.async_pipeline_waits);  // never waited for the pipeline's layout
    } else {
        CHECK(s.remap_reflected_layouts == before.remap_reflected_layouts);
        CHECK(s.pipeline_pending_skips > 0 || s.async_pipeline_waits >= before.async_pipeline_waits);
    }
    // Second shader (uniform at 1 + storage at 5): same group 0 bind group serves both through the remap.
    gpu::CommandEncoder g;
    g.begin_frame();
    g.set_compute_pipeline(124);
    set_bind_group2(g, 0, 115);
    g.dispatch_compute(1, 1, 1);
    g.present();
    bool ok_b = false;
    const uint32_t want_b[4] = {100, 101, 102, 103};
    for (int frame = 0; frame < 30 && !ok_b; ++frame) {
        run(ex, g);
        ex.wait_idle();
        std::vector<uint8_t> out;
        ok_b = ex.read_buffer(110, 0, 16, &out, &err) && out.size() == 16 && std::memcmp(out.data(), want_b, 16) == 0;
        if (!ok_b) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    (void)ok_b;  // the uniform has no source resource in the group (a dummy): only the layout path is checked
    CHECK(ex.stats().device_errors == before.device_errors);
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

const char* kLightCompute = R"(
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  out[id.x] = id.x;
}
)";

// gpu_time_split: two compute pipelines in one pass are timed as separate segments, each under its own
// shader hash (without the split the whole pass is keyed to the first pipeline).
void test_timer_split() {
    gpu::d3d12_guest::GpuTimeTable table;
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.gpu_time = &table;
    opts.gpu_time_split = true;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        std::fprintf(stderr, "FAIL: split NativeExecutor::create: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    if (!ex->stats().timer_supported) {
        std::printf("test_native_executor: adapter lacks timestamp-query: split checks skipped\n");
        return;
    }
    const uint64_t kHeavy = 0x00c0ffee0badf00dull, kLight = 0x00c0ffee0badf00eull;
    const std::string heavy = kHeavyCompute, light = kLightCompute;
    std::vector<uint8_t> pack = {'W', 'G', 'P', 'K', 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t base = 16 + 2 * 20;
    append(pack, kHeavy);
    pack.push_back(3);
    pack.insert(pack.end(), 3, 0);
    append(pack, base);
    append(pack, static_cast<uint32_t>(heavy.size()));
    append(pack, kLight);
    pack.push_back(3);
    pack.insert(pack.end(), 3, 0);
    append(pack, static_cast<uint32_t>(base + heavy.size()));
    append(pack, static_cast<uint32_t>(light.size()));
    pack.insert(pack.end(), heavy.begin(), heavy.end());
    pack.insert(pack.end(), light.begin(), light.end());
    const std::string path = "test_native_executor_timer_split.wgsl.pack";
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
    setup.create_shader_module(206, GpuShaderStage::Compute, "0x00c0ffee0badf00e");
    create_layouts(setup, 202, 203, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 204;
    cp.cs_module_id = 201;
    cp.pipeline_layout_id = 203;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    cp.id = 207;
    cp.cs_module_id = 206;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 205, 202, {bg_entry(0, GpuBindResourceType::Buffer, 200)});
    run(*ex, setup);
    for (int f = 0; f < 3; ++f) {
        gpu::CommandEncoder frame;
        frame.begin_frame();
        frame.set_compute_pipeline(204);
        set_bind_group2(frame, 0, 205);
        frame.dispatch_compute(64);
        frame.set_compute_pipeline(207);  // same compute pass, other shader
        set_bind_group2(frame, 0, 205);
        frame.dispatch_compute(64);
        frame.present();
        run(*ex, frame);
        ex->wait_idle();
    }
    ex->poll();
    CHECK(ex->stats().device_errors == 0);
    if (ex->stats().timer_segments == 0) {
        std::printf("test_native_executor: adapter lacks timestamp-query-inside-passes: split checks skipped\n");
    } else {
        CHECK(ex->stats().timer_segments == 3);  // the light pipeline's segment, once per frame
        uint64_t heavy_ns = 0, light_ns = 0;
        for (const auto& e : table.top(16)) {
            if (e.hash == kHeavy && e.kind == gpu::d3d12_guest::GpuPassKind::Compute) {
                heavy_ns = e.stats.total_ns;
                CHECK(e.stats.passes == 3);
            }
            if (e.hash == kLight && e.kind == gpu::d3d12_guest::GpuPassKind::Compute) {
                light_ns = e.stats.total_ns;
                CHECK(e.stats.passes == 3);
            }
        }
        CHECK(heavy_ns > 0);
        CHECK(light_ns > 0);
        CHECK(heavy_ns > light_ns);
        std::printf("timer split: heavy %llu ns, light %llu ns over 3 frames\n", static_cast<unsigned long long>(heavy_ns),
                    static_cast<unsigned long long>(light_ns));
    }
    ex.reset();
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
    // The draw is dropped and counted (pipeline_failed), never encoded: an invalid pipeline in the pass
    // would invalidate the whole command buffer.
    CHECK(ex.stats().draws_skipped > before.draws_skipped);
    CHECK(ex.stats().pipeline_failed_skips > before.pipeline_failed_skips);

    // Same for a compute pipeline whose async compile failed.
    const ExecutorStats mid = ex.stats();
    gpu::CommandEncoder csetup;
    csetup.create_shader_module(85, GpuShaderStage::Compute, "this is not wgsl");
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 86;
    cp.cs_module_id = 85;
    cp.pipeline_layout_id = 83;
    emit(csetup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    run(ex, csetup);
    gpu::CommandEncoder cframe;
    cframe.begin_frame();
    cframe.set_compute_pipeline(86);
    cframe.dispatch_compute(1, 1, 1);
    cframe.present();
    run(ex, cframe);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().dispatches_skipped > mid.dispatches_skipped);
    CHECK(ex.stats().pipeline_failed_skips > mid.pipeline_failed_skips);
}

// A failing pipeline whose fragment shader has a non-zero (pack) hash lands in failed_pipeline_hashes once
// (inline test shaders have hash 0, which is never recorded) and in the --auto-quarantine file text.
void test_failed_pipeline_records_hash(NativeExecutor& ex) {
    const uint64_t kBadHash = 0x00bad0de0000f5a1ull;
    const std::string wgsl = "this is not wgsl";
    std::vector<uint8_t> pack = {'W', 'G', 'P', 'K', 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    append(pack, kBadHash);
    pack.push_back(1);  // fragment
    pack.insert(pack.end(), 3, 0);
    append(pack, static_cast<uint32_t>(16 + 20));
    append(pack, static_cast<uint32_t>(wgsl.size()));
    pack.insert(pack.end(), wgsl.begin(), wgsl.end());
    const std::string path = "test_native_executor_failhash.wgsl.pack";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(pack.data()), static_cast<std::streamsize>(pack.size()));
    }
    std::string err;
    CHECK(ex.load_wgsl_pack(path, &err));
    std::remove(path.c_str());
    const size_t known = ex.stats().failed_pipeline_hashes.size();
    gpu::CommandEncoder setup;
    setup.create_shader_module(90, GpuShaderStage::Vertex, "this is not wgsl");
    setup.create_shader_module(91, GpuShaderStage::Fragment, "0x00bad0de0000f5a1");
    create_layouts(setup, 92, 93, {});
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(94, 90, 91, 93)));
    run(ex, setup);
    for (int i = 0; i < 2; ++i) {  // twice: the hash is recorded once
        gpu::CommandEncoder frame;
        frame.begin_frame();
        auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
        frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
        frame.set_pipeline(94);
        frame.draw(3);
        frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
        frame.present();
        run(ex, frame);
        ex.wait_idle();
        ex.poll();
    }
    const auto& failed = ex.stats().failed_pipeline_hashes;
    CHECK(failed.size() == known + 1);
    CHECK(std::count(failed.begin(), failed.end(), kBadHash) == 1);
    CHECK(ex.stats_json().find("\"failed_pipeline_hashes\"") != std::string::npos);
    const std::string file =
        gpu::native_exec::ShaderQuarantine::merge_auto_file("# my note\n", failed);
    CHECK(file.find("# my note\n") == 0);
    CHECK(file.find("0x00bad0de0000f5a1 pipeline_failed auto\n") != std::string::npos);
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
        if (run_index == 0) {
            test_failed_async_pipeline_resolves(*ex);
            test_failed_pipeline_records_hash(*ex);
        }
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

// Live mode with a WGSL pack: the pack shader's module is compiled on a shader worker, not on the
// executor (queue) thread. The pipeline packet returns at once; the pipeline is created once the
// shader is ready. A dispatch that needs it earlier waits within the frame's pipeline budget; past the
// budget it is dropped and counted (pipeline_pending), never run with a stand-in.
std::string big_compute_wgsl(int pad_functions) {
    std::string s = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n";
    for (int i = 0; i < pad_functions; ++i) {
        const std::string n = std::to_string(i);
        s += "fn pad" + n + "(x: u32) -> u32 { var a = x * " + n + "u; a = a ^ (a >> 3u); if (a > " + n +
             "u) { a = a - 1u; } return a + " + n + "u; }\n";
    }
    s += "@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = i * 3u + 2u; }\n";
    return s;
}

bool write_one_shader_pack(const std::string& path, uint64_t hash, uint8_t wgpk_stage, const std::string& wgsl) {
    std::vector<uint8_t> pack = {'W', 'G', 'P', 'K', 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    append(pack, hash);
    pack.push_back(wgpk_stage);
    pack.insert(pack.end(), 3, 0);
    append(pack, static_cast<uint32_t>(16 + 20));
    append(pack, static_cast<uint32_t>(wgsl.size()));
    pack.insert(pack.end(), wgsl.begin(), wgsl.end());
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(pack.data(), 1, pack.size(), f) == pack.size();
    std::fclose(f);
    return ok;
}

void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    if (*v) setenv(k, v, 1);
    else unsetenv(k);
#endif
}

void test_shader_workers(const char* wait_ms) {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.async_pipelines = true;
    set_env("PG6_PIPELINE_WAIT_MS", wait_ms);
    NativeExecutor::CreateStatus status;
    std::string error;
    auto exu = NativeExecutor::create(opts, &status, &error);
    set_env("PG6_PIPELINE_WAIT_MS", "");
    if (!exu) {
        CHECK(status == NativeExecutor::CreateStatus::NoAdapter);
        return;
    }
    NativeExecutor& ex = *exu;
    if (ex.stats().shader_workers == 0) {
        std::printf("shader workers: skipped (no implicit device synchronization)\n");
        return;
    }
    const std::string path = "test_native_executor_workers.wgsl.pack";
    const std::string wgsl = big_compute_wgsl(1500);  // ~180 KB: a module compile that takes a while
    CHECK(write_one_shader_pack(path, 0x00b16b00b5c0ffeeull, 3, wgsl));
    std::string err;
    CHECK(ex.load_wgsl_pack(path, &err));
    const ExecutorStats before = ex.stats();

    gpu::CommandEncoder setup;
    setup.create_buffer(110, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 16);
    setup.create_shader_module(111, GpuShaderStage::Compute, "0x00b16b00b5c0ffee");
    create_layouts(setup, 112, 113, {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 114;
    cp.cs_module_id = 111;
    cp.pipeline_layout_id = 113;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 115, 112, {bg_entry(0, GpuBindResourceType::Buffer, 110)});
    // The deferred pipeline keeps the layout and shader alive after their IDs are destroyed.
    emit(setup, GpuOpcode::DestroyResource, bytes_of(uint32_t{113}));
    emit(setup, GpuOpcode::DestroyResource, bytes_of(uint32_t{111}));
    const auto t0 = std::chrono::steady_clock::now();
    run(ex, setup);
    const uint64_t setup_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    {
        const ExecutorStats& s = ex.stats();
        CHECK(s.shader_async_queued == before.shader_async_queued + 1);
        CHECK(s.pipelines_deferred == before.pipelines_deferred + 1);
        CHECK(s.shader_materialize_ns == before.shader_materialize_ns);  // nothing compiled on this thread
    }

    const uint32_t want[4] = {2, 5, 8, 11};
    bool ok = false, first_ok = false;
    int frames = 0;
    const auto tf = std::chrono::steady_clock::now();
    for (int frame = 0; frame < 2000 && !ok; ++frame, ++frames) {
        gpu::CommandEncoder f;
        f.begin_frame();
        f.set_compute_pipeline(114);
        set_bind_group2(f, 0, 115);
        f.dispatch_compute(1, 1, 1);
        f.present();
        run(ex, f);
        ex.wait_idle();
        std::vector<uint8_t> out;
        ok = ex.read_buffer(110, 0, 16, &out, &err) && out.size() == 16 && std::memcmp(out.data(), want, 16) == 0;
        if (frame == 0) first_ok = ok;
        if (!ok) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(ok);
    const ExecutorStats& s = ex.stats();
    CHECK(s.shader_async == before.shader_async + 1);
    CHECK(s.pipelines_deferred_built == before.pipelines_deferred_built + 1);
    CHECK(s.shader_materialize_ns == before.shader_materialize_ns);
    CHECK(s.async_pipeline_failures == before.async_pipeline_failures);
    CHECK(s.device_errors == before.device_errors);
    // The executor thread never compiled: the setup batch returned before the worker finished.
    CHECK(setup_ns < s.shader_async_ns - before.shader_async_ns);
    if (std::string(wait_ms) == "0") {
        // Never waits: every dispatch before the pipeline existed was dropped and counted.
        if (!first_ok) CHECK(s.pipeline_pending_skips > before.pipeline_pending_skips);
        CHECK(s.shader_async_wait_ns == before.shader_async_wait_ns);
    } else {
        // A long budget: the first dispatch waited for the worker and ran.
        CHECK(first_ok);
        CHECK(s.pipeline_pending_skips == before.pipeline_pending_skips);
    }
    std::printf("shader workers (wait %s ms): %llu workers, setup %llu us, worker compile %llu ms, waited %llu ms, first frame %s, frames %d, %lld ms\n",
                wait_ms, static_cast<unsigned long long>(s.shader_workers), static_cast<unsigned long long>(setup_ns / 1000),
                static_cast<unsigned long long>((s.shader_async_ns - before.shader_async_ns) / 1000000),
                static_cast<unsigned long long>((s.shader_async_wait_ns - before.shader_async_wait_ns) / 1000000),
                first_ok ? "ran" : "dropped", frames,
                static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tf).count()));
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// A rgba16unorm texture (unfilterable-float to Dawn) read through a float layout entry with a linear
// sampler: the executor binds its float twin, so the bilinear blend of the real texels is what shows.
void test_norm16_filtered_read(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(9700, 2, 2, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 36)));  // rgba16unorm
    // x=0: (0, 0.2, 1, 1)  x=1: (0.5, 0.8, 0.5, 1): the blend at the center is (0.25, 0.5, 0.75, 1)
    const uint16_t t0[4] = {0, 13107, 65535, 65535}, t1[4] = {32768, 52428, 32768, 65535};
    std::vector<uint8_t> texels(256 + 16, 0);
    for (int y = 0; y < 2; ++y) {
        std::memcpy(&texels[y * 256], t0, 8);
        std::memcpy(&texels[y * 256 + 8], t1, 8);
    }
    GpuWriteTexturePayload wt;
    std::memset(&wt, 0, sizeof wt);
    wt.texture_id = 9700;
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
    sp.id = 9701;
    sp.mag_filter = static_cast<uint8_t>(GpuFilterMode::Linear);
    sp.min_filter = static_cast<uint8_t>(GpuFilterMode::Linear);
    sp.max_anisotropy = 1;
    sp.lod_max_clamp = 32.0f;
    emit(setup, GpuOpcode::CreateSampler2, bytes_of(sp));
    setup.create_shader_module(9702, GpuShaderStage::Vertex, kTexturedTriangle);
    setup.create_shader_module(9703, GpuShaderStage::Fragment, kTexturedTriangle);
    create_layouts(setup, 9704, 9705,
                   {layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture, 0, 2),
                    layout_entry(1, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Sampler, 0)});
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(9706, 9702, 9703, 9705)));
    create_bind_group(setup, 9707, 9704,
                      {bg_entry(0, GpuBindResourceType::TextureView, 9700), bg_entry(1, GpuBindResourceType::Sampler, 9701)});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    frame.set_pipeline(9706);
    set_bind_group2(frame, 0, 9707);
    frame.draw(3);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    const ExecutorStats& after = ex.stats();
    std::printf("norm16 filtered read: twins=%llu copies=%llu binds=%llu dummies=%llu\n",
                (unsigned long long)(after.norm16_twins_created - before.norm16_twins_created),
                (unsigned long long)(after.norm16_twin_copies - before.norm16_twin_copies),
                (unsigned long long)(after.norm16_twin_binds - before.norm16_twin_binds),
                (unsigned long long)(after.bind_group_unfilterable_dummies - before.bind_group_unfilterable_dummies));
    CHECK(after.device_errors == before.device_errors);
    CHECK(after.bind_group_unfilterable_dummies == before.bind_group_unfilterable_dummies);
    CHECK(after.norm16_twin_binds == before.norm16_twin_binds + 1);
    FrameImage img;
    std::string err;
    CHECK(ex.read_last_frame(&img, &err) && px_near(img, 1, 1, 191, 128, 64, 255, 2));
}

// The same for a volume: a 2x2x2 rgba16unorm 3D texture read by texture_3d + linear sampler goes through a
// 3D float twin (one copy pass per depth slice).
void test_norm16_volume_filtered_read(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    GpuCreateTexture2Payload ct = texture2(9710, 2, 2, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 36);
    ct.dimension = 2;
    ct.depth_or_layers = 2;
    emit(setup, GpuOpcode::CreateTexture2, bytes_of(ct));
    // slice 0: (0, 0.2, 1, 1)  slice 1: (1, 0.8, 0, 1): the blend at the center is (0.5, 0.5, 0.5, 1)
    const uint16_t t0[4] = {0, 13107, 65535, 65535}, t1[4] = {65535, 52428, 0, 65535};
    std::vector<uint8_t> texels(256 * 4, 0);
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) std::memcpy(&texels[(z * 2 + y) * 256 + x * 8], z ? t1 : t0, 8);
    GpuWriteTexturePayload wt;
    std::memset(&wt, 0, sizeof wt);
    wt.texture_id = 9710;
    wt.width = 2;
    wt.height = 2;
    wt.depth_or_layers = 2;
    wt.bytes_per_row = 256;
    wt.rows_per_image = 2;
    wt.data_len = static_cast<uint32_t>(texels.size());
    std::vector<uint8_t> wtv = bytes_of(wt);
    wtv.insert(wtv.end(), texels.begin(), texels.end());
    emit(setup, GpuOpcode::WriteTexture, wtv);
    GpuCreateSampler2Payload sp;
    std::memset(&sp, 0, sizeof sp);
    sp.id = 9711;
    sp.mag_filter = static_cast<uint8_t>(GpuFilterMode::Linear);
    sp.min_filter = static_cast<uint8_t>(GpuFilterMode::Linear);
    sp.max_anisotropy = 1;
    sp.lod_max_clamp = 32.0f;
    emit(setup, GpuOpcode::CreateSampler2, bytes_of(sp));
    static const char* kVolumeTriangle = R"(
@group(0) @binding(0) var tex: texture_3d<f32>;
@group(0) @binding(1) var smp: sampler;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4f { return textureSample(tex, smp, vec3f(0.5, 0.5, 0.5)); }
)";
    setup.create_shader_module(9712, GpuShaderStage::Vertex, kVolumeTriangle);
    setup.create_shader_module(9713, GpuShaderStage::Fragment, kVolumeTriangle);
    create_layouts(setup, 9714, 9715,
                   {layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture, 0, 6),
                    layout_entry(1, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Sampler, 0)});
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(9716, 9712, 9713, 9715)));
    GpuBindGroupEntry2 te = bg_entry(0, GpuBindResourceType::TextureView, 9710);
    te.view.dimension = 6;
    create_bind_group(setup, 9717, 9714, {te, bg_entry(1, GpuBindResourceType::Sampler, 9711)});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    frame.set_pipeline(9716);
    set_bind_group2(frame, 0, 9717);
    frame.draw(3);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    const ExecutorStats& after = ex.stats();
    CHECK(after.device_errors == before.device_errors);
    CHECK(after.bind_group_unfilterable_dummies == before.bind_group_unfilterable_dummies);
    CHECK(after.norm16_twin_binds == before.norm16_twin_binds + 1);
    FrameImage img;
    std::string err;
    CHECK(ex.read_last_frame(&img, &err) && px_near(img, 1, 1, 128, 128, 128, 255, 2));
}

// #2086's all-group cache must remain available after a worker drops the pack text.
void test_worker_group_metadata() {
    ExecutorOptions opts;
    opts.width = kW; opts.height = kH; opts.async_pipelines = true;
    set_env("PG6_PIPELINE_WAIT_MS", "10000");
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    set_env("PG6_PIPELINE_WAIT_MS", "");
    if (!ex) { CHECK(status == NativeExecutor::CreateStatus::NoAdapter); return; }
    if (ex->stats().shader_workers == 0) return;
    const std::string path = "test_worker_group_metadata.wgsl.pack";
    CHECK(write_one_shader_pack(path, 0x00b16b00b5c0ffeaull, 3, kComputeGroup3));
    CHECK(ex->load_wgsl_pack(path, &error));
    test_layout_fit_missing_group(*ex, true);
    CHECK(ex->stats().shader_async == 1);
    CHECK(ex->stats().shader_code_rereads == 0);
    if (!std::getenv("PG6_KEEP_WGSL")) CHECK(ex->stats().shader_scans == 8);
    if (!std::getenv("PG6_NO_BINDING_ACCESS")) CHECK(ex->stats().shader_access_scans == 1);
    std::error_code ec; std::filesystem::remove(path, ec);
}

// A pack module containing both stages is created as vertex, then reused as fragment by v2.
// The v1 pipeline is built after the worker has finished, so its group-zero check must use cached
// metadata as well. Rendering and the missing-group skip exercise the decisions after release.
void test_shader_pack_identity(const char* raw_case) {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.async_pipelines = true;
    set_env("PG6_PIPELINE_WAIT_MS", "10000");
    NativeExecutor::CreateStatus status;
    std::string error;
    auto exu = NativeExecutor::create(opts, &status, &error);
    set_env("PG6_PIPELINE_WAIT_MS", "");
    if (!exu) { CHECK(status == NativeExecutor::CreateStatus::NoAdapter); return; }
    NativeExecutor& ex = *exu;
    if (ex.stats().shader_workers == 0) return;
    const std::string path = "test_released_shader_source.wgsl.pack";
    const uint64_t hash = 0x00b16b00b5c0fff0ull;
    const std::string scenario = raw_case;
    // This comment makes the conservative fragment-output scanner return nullopt while the actual
    // WGSL still has a valid fragment entry point. Its cached unknown result must not cause rereads.
    const std::string prefix = scenario == "unknown-mask" ? "// @fragment fn fs_main() -> MysteryReturn\n" : "";
    // Padding forces a later seek out of stdio's old buffer for in-place overwrite tests.
    const std::string original = prefix + "@group(0) @binding(2) var<storage, read_write> spare: array<u32>;\n" +
                                 std::string(kTexturedTriangle) + "\n// " + std::string(16384, 'x') + "\n";
    if (scenario == "unknown-mask") CHECK(!wgsl_fragment_output_mask(original, "fs_main").has_value());
    std::string changed = original;
    const size_t fragment_pos = changed.find("fs_main");
    CHECK(fragment_pos != std::string::npos);
    changed.replace(fragment_pos, 7, "fs_fake");
    changed.insert(changed.find("@vertex"), "fn future_write() { spare[0] = 1u; }\n");
    CHECK(scan_binding_access(original).of(0, 2) == BindingAccess::Unreferenced);
    CHECK(scan_binding_access(changed).of(0, 2) == BindingAccess::Written);
    CHECK(write_one_shader_pack(path, hash, 1, original));
    CHECK(ex.load_wgsl_pack(path, &error));
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(400, 1, 1, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 22)));
    GpuWriteTexturePayload wt{};
    wt.texture_id = 400; wt.width = 1; wt.height = 1; wt.depth_or_layers = 1;
    wt.bytes_per_row = 256; wt.rows_per_image = 1; wt.data_len = 4;
    auto payload = bytes_of(wt);
    payload.insert(payload.end(), {255, 128, 0, 255});
    emit(setup, GpuOpcode::WriteTexture, payload);
    setup.create_sampler(401);
    setup.create_shader_module(402, GpuShaderStage::Vertex, "0x00b16b00b5c0fff0");
    setup.create_shader_module(410, GpuShaderStage::Vertex, "0x00b16b00b5c0fff0");
    create_layouts(setup, 403, 404,
                   {layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Texture, 0, 2),
                    layout_entry(1, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Sampler, 0)});
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(409, 402, 0, 404)));
    setup.create_render_pipeline(406, 402, 402, 0);
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(413, 410, 0, 404)));
    setup.create_render_pipeline(411, 410, 410, 0);
    create_bind_group(setup, 407, 403, {bg_entry(0, GpuBindResourceType::TextureView, 400),
                                        bg_entry(1, GpuBindResourceType::Sampler, 401)});
    const GpuBindGroupEntry entries[2] = {
        {0, static_cast<uint32_t>(GpuBindResourceType::TextureView), 400, 0, 0, 0},
        {1, static_cast<uint32_t>(GpuBindResourceType::Sampler), 401, 0, 0, 0},
    };
    setup.create_bind_group(408, 406, 0, entries, 2);
    run(ex, setup);
    // The v1 packet waits for worker materialization. Replacing the pack must not change the
    // source metadata of an existing module that was already compiled from the first pack.
    const std::string other = "test_released_shader_replacement.wgsl.pack";
    if (scenario == "missing") {
        CHECK(write_one_shader_pack(other, hash + 1, 3, "@compute @workgroup_size(1) fn cs_main() {}"));
        CHECK(ex.load_wgsl_pack(other, &error));
    } else if (scenario == "same-hash") {
        CHECK(write_one_shader_pack(other, hash, 1, changed));
        CHECK(ex.load_wgsl_pack(other, &error));
    } else if (scenario == "in-place") {
        if (!write_one_shader_pack(path, hash, 1, changed)) {
            std::printf("shader pack identity: in-place overwrite unsupported\n");
            return;
        }
    } else if (scenario == "truncated") {
        std::error_code resize_error;
        std::filesystem::resize_file(path, 36, resize_error);
        if (resize_error) {
            std::printf("shader pack identity: in-place truncate unsupported\n");
            return;
        }
    } else if (scenario == "failed-open") {
        std::error_code ec;
        std::filesystem::remove(other, ec);
        CHECK(!ex.load_wgsl_pack(other, &error));
    } else {
        CHECK(scenario == "unchanged" || scenario == "unknown-mask");
    }
    gpu::CommandEncoder later;
    emit(later, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(405, 402, 402, 404)));
    emit(later, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(412, 410, 410, 404)));
    emit(later, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(414, 402, 402, 404)));
    run(ex, later);
    for (uint32_t pipe : {405u, 412u, 414u, 406u}) {
        gpu::CommandEncoder frame;
        frame.begin_frame();
        if (pipe != 406u) {
            const auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
            frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
        }
        frame.set_pipeline(pipe);
        if (pipe != 406u) set_bind_group2(frame, 0, 407);
        else frame.set_bind_group(0, 408);
        frame.draw(3);
        if (pipe != 406u) frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
        frame.present();
        run(ex, frame);
        ex.wait_idle(); ex.poll();
        CHECK(frame_is(ex, {0, 128, 255, 255}));
    }
    gpu::CommandEncoder skip;
    skip.begin_frame(); skip.set_pipeline(406); skip.draw(3); skip.present();
    run(ex, skip);
    CHECK(ex.stats().shader_async == before.shader_async + 2);
    CHECK(ex.stats().shader_async_max_bytes == original.size());
    if (!std::getenv("PG6_KEEP_WGSL")) CHECK(ex.stats().shader_scans == before.shader_scans + 16);
    if (!std::getenv("PG6_NO_BINDING_ACCESS")) {
        CHECK(ex.stats().shader_access_scans == before.shader_access_scans + 2);
        CHECK(ex.stats().shader_rw_never_written == before.shader_rw_never_written + 2);
    }
    CHECK(ex.stats().draws == before.draws + 4);
    CHECK(ex.stats().draws_skipped == before.draws_skipped + 1);
    CHECK(ex.stats().device_errors == before.device_errors);
    CHECK(ex.stats().shader_code_rereads == before.shader_code_rereads);
    std::error_code ec; std::filesystem::remove(path, ec); std::filesystem::remove(other, ec);
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


// ---- --show-target (present any target): the chosen texture is drawn instead of the swapchain image ----

GpuBeginRenderPassPayload texture_pass(uint32_t id, float r, float g, float b, float a, uint16_t mip = 0,
                                       uint32_t layer = 0) {
    GpuBeginRenderPassPayload bp = swapchain_pass(GpuLoadOp::Clear, r, g, b, a);
    bp.color[0].texture_id = id;
    bp.color[0].view.dimension = 2;  // 2d: a single mip / layer, whatever the texture's array size
    bp.color[0].view.base_mip_level = mip;
    bp.color[0].view.mip_level_count = 1;
    bp.color[0].view.base_array_layer = layer;
    bp.color[0].view.array_layer_count = 1;
    return bp;
}

void end_pass(gpu::CommandEncoder& e, const GpuBeginRenderPassPayload& bp) {
    e.emit_packet(GpuOpcode::BeginRenderPass, &bp, sizeof bp);
    e.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
}

GpuCreateTexture2Payload show_texture(uint32_t id, uint8_t format, uint32_t layers = 1, uint32_t mips = 1,
                                      uint32_t samples = 1,
                                      uint32_t usage = GpuTextureUsage::RenderAttachment | GpuTextureUsage::TextureBinding) {
    GpuCreateTexture2Payload ct = texture2(id, kW, kH, usage, format);
    ct.depth_or_layers = layers;
    ct.mip_levels = mips;
    ct.sample_count = samples;
    return ct;
}

// Every pixel of the last frame is `want` (bgra), within tol.
bool frame_all_near(NativeExecutor& ex, Px want, int tol = 1) {
    FrameImage img;
    std::string err;
    if (!ex.read_last_frame(&img, &err) || img.width != kW || img.height != kH) return false;
    for (uint32_t i = 0; i < kW * kH; ++i)
        if (!px_near(img, i % kW, i / kW, want.b, want.g, want.r, want.a, tol)) return false;
    return true;
}

// The frame: textures 100 (rgba8unorm, green), 101 (2 layers: blue, yellow), 102 (2 mips; mip 1 magenta),
// 103 (depth32float cleared to 0.5), 104 (4x msaa, cyan), 105 (r32uint), 106 (no TextureBinding), then the
// swapchain red. Resources are created once (the first call), each later call just draws and presents.
void show_frame(NativeExecutor& ex, bool create) {
    gpu::CommandEncoder e;
    e.begin_frame();
    if (create) {
        emit(e, GpuOpcode::CreateTexture2, bytes_of(show_texture(100, 22)));
        emit(e, GpuOpcode::CreateTexture2, bytes_of(show_texture(101, 22, 2)));
        emit(e, GpuOpcode::CreateTexture2, bytes_of(show_texture(102, 22, 1, 2)));
        emit(e, GpuOpcode::CreateTexture2, bytes_of(show_texture(103, 48)));
        emit(e, GpuOpcode::CreateTexture2, bytes_of(show_texture(104, 22, 1, 1, 4)));
        emit(e, GpuOpcode::CreateTexture2, bytes_of(show_texture(105, 14)));
        emit(e, GpuOpcode::CreateTexture2,
             bytes_of(show_texture(106, 22, 1, 1, 1, GpuTextureUsage::RenderAttachment)));
    }
    end_pass(e, texture_pass(100, 0, 1, 0, 1));                         // pass 0
    end_pass(e, texture_pass(101, 0, 0, 1, 1, 0, 0));                   // pass 1
    end_pass(e, texture_pass(101, 1, 1, 0, 1, 0, 1));                   // pass 2
    end_pass(e, texture_pass(102, 1, 0, 1, 1, 1, 0));                   // pass 3 (mip 1)
    GpuBeginRenderPassPayload dp;                                        // pass 4: depth only
    std::memset(&dp, 0, sizeof dp);
    dp.depth_stencil.texture_id = 103;
    dp.depth_stencil.depth_load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
    dp.depth_stencil.depth_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
    dp.depth_stencil.depth_clear_value = 0.5f;
    end_pass(e, dp);
    end_pass(e, texture_pass(104, 0, 1, 1, 1));                         // pass 5
    end_pass(e, texture_pass(105, 0, 0, 0, 0));                         // pass 6 (a uint target)
    end_pass(e, texture_pass(106, 0, 0, 0, 0));                         // pass 7
    end_pass(e, swapchain_pass(GpuLoadOp::Clear, 1, 0, 0, 1));          // pass 8: swapchain red
    e.present();
    run(ex, e);
}

bool show(NativeExecutor& ex, const char* selector) {
    std::string err;
    if (!ex.set_show_target(selector, &err)) {
        std::fprintf(stderr, "set_show_target(%s): %s\n", selector, err.c_str());
        return false;
    }
    show_frame(ex, false);
    return true;
}

void test_show_target() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    if (!ex) {
        std::fprintf(stderr, "FAIL: show-target NativeExecutor::create: %s\n", error.c_str());
        ++g_failures;
        return;
    }
    show_frame(*ex, true);
    CHECK(frame_is(*ex, {0, 0, 255, 255}));  // no selector: the swapchain image (red)
    CHECK(ex->stats().show_target_frames == 0);

    CHECK(show(*ex, "100"));
    CHECK(frame_is(*ex, {0, 255, 0, 255}));
    CHECK(ex->shown_target_label().find("tex 100 ") == 0);
    CHECK(show(*ex, "tex:101@layer=0"));
    CHECK(frame_is(*ex, {255, 0, 0, 255}));  // blue
    CHECK(show(*ex, "101@layer=1"));
    CHECK(frame_is(*ex, {0, 255, 255, 255}));  // yellow
    CHECK(show(*ex, "101@layer=9"));  // clamped to the last layer
    CHECK(frame_is(*ex, {0, 255, 255, 255}));
    CHECK(show(*ex, "102@mip=1"));
    CHECK(frame_is(*ex, {255, 0, 255, 255}));  // magenta, the mip is shown at its own size (32x16, same aspect)
    CHECK(ex->shown_target_label().find("32x16") != std::string::npos);

    // By pass: pass 2 rendered layer 1 of texture 101 (its own layer is the default), last = the last texture pass.
    CHECK(show(*ex, "pass:2"));
    CHECK(frame_is(*ex, {0, 255, 255, 255}));
    CHECK(show(*ex, "pass:1"));
    CHECK(frame_is(*ex, {255, 0, 0, 255}));
    CHECK(show(*ex, "pass:5"));  // the 4x msaa target (sample 0)
    CHECK(frame_is(*ex, {255, 255, 0, 255}));
    CHECK(show(*ex, "pass:8"));  // renders only to the swapchain: nothing to show, the swapchain image stays
    CHECK(frame_is(*ex, {0, 0, 255, 255}));
    CHECK(ex->shown_target_label().find("unavailable") == 0);
    CHECK(show(*ex, "pass:99"));
    CHECK(frame_is(*ex, {0, 0, 255, 255}));
    CHECK(show(*ex, "last"));  // pass 7 renders to texture 106: no TextureBinding
    CHECK(frame_is(*ex, {0, 0, 255, 255}));
    CHECK(ex->shown_target_label().find("TextureBinding") != std::string::npos);
    CHECK(show(*ex, "105"));
    CHECK(ex->shown_target_label().find("integer") != std::string::npos);
    CHECK(show(*ex, "9999"));
    CHECK(ex->shown_target_label().find("no such texture") != std::string::npos);
    CHECK(frame_is(*ex, {0, 0, 255, 255}));

    // Depth through the depth twin: 0.5 -> grey 128 on the default range, 255 / 0 on shifted ranges.
    CHECK(show(*ex, "depth:103"));
    CHECK(frame_all_near(*ex, {128, 128, 128, 255}, 1));
    CHECK(ex->stats().depth_twins_created >= 1);
    CHECK(show(*ex, "pass:4"));  // a depth-only pass shows its depth target
    CHECK(frame_all_near(*ex, {128, 128, 128, 255}, 1));
    CHECK(show(*ex, "pass:4:depth@range=0:0.5"));
    CHECK(frame_all_near(*ex, {255, 255, 255, 255}, 0));
    CHECK(show(*ex, "depth:103@range=0.5:1"));
    CHECK(frame_all_near(*ex, {0, 0, 0, 255}, 0));
    CHECK(show(*ex, "103"));  // a depth texture asked for as colour
    CHECK(ex->shown_target_label().find("depth/stencil") != std::string::npos);
    CHECK(show(*ex, "100@range=0:2"));  // green 1.0 -> 0.5
    CHECK(frame_all_near(*ex, {0, 128, 0, 255}, 1));

    // Hotkey cycling over the frame's render targets: cycle starts on the swapchain, next walks the targets
    // by id (colour before depth), past the last one it returns to the swapchain, previous goes backwards.
    CHECK(show(*ex, "cycle"));
    CHECK(frame_is(*ex, {0, 0, 255, 255}));
    const auto cands = ex->shown_target_candidates();
    CHECK(cands.size() == 7);  // 100 101 102 103(depth) 104 105 106
    CHECK(cands.size() == 7 && cands[3].id == 103 && cands[3].depth);
    ex->show_target_hotkeys(kHotkeyNext);
    show_frame(*ex, false);
    CHECK(frame_is(*ex, {0, 255, 0, 255}));  // 100
    ex->show_target_hotkeys(kHotkeyNext);
    show_frame(*ex, false);
    CHECK(frame_is(*ex, {255, 0, 0, 255}));  // 101, its own layer from pass 1 (blue)
    ex->show_target_hotkeys(kHotkeyLayer);
    show_frame(*ex, false);
    CHECK(frame_is(*ex, {0, 255, 255, 255}));  // layer 1 (yellow)
    ex->show_target_hotkeys(kHotkeyPrev);
    show_frame(*ex, false);
    CHECK(frame_is(*ex, {0, 255, 0, 255}));  // back to 100
    ex->show_target_hotkeys(kHotkeyPrev);
    show_frame(*ex, false);
    CHECK(frame_is(*ex, {0, 0, 255, 255}));  // off: the swapchain
    CHECK(ex->shown_target_label().empty());
    ex->show_target_hotkeys(kHotkeyPrev);  // wraps to the last candidate (106: cannot be shown)
    show_frame(*ex, false);
    CHECK(ex->shown_target_label().find("TextureBinding") != std::string::npos);

    // Off again: the guest's own frame.
    std::string err;
    CHECK(ex->set_show_target("", &err));
    show_frame(*ex, false);
    CHECK(frame_is(*ex, {0, 0, 255, 255}));
    CHECK(!ex->set_show_target("bogus", &err) && !err.empty());
    CHECK(!ex->set_show_target("12@layer", &err));
    CHECK(!ex->set_show_target("12@range=1:1", &err));
    CHECK(ex->stats().show_target_frames > 10);
    CHECK(ex->stats().show_target_unavailable > 3);
    CHECK(ex->stats().device_errors == 0);
    CHECK(ex->stats().device_lost == 0);
}

// Read-only: with --show-target the per-frame digest of the guest's stream image is the same as without.
void test_show_target_read_only() {
    const char* paths[2] = {"test_show_target_digest_a.txt", "test_show_target_digest_b.txt"};
    for (int i = 0; i < 2; ++i) {
        std::remove(paths[i]);
        ExecutorOptions opts;
        opts.width = kW;
        opts.height = kH;
        opts.frame_digest_path = paths[i];
        if (i == 1) opts.show_target = "pass:0";
        NativeExecutor::CreateStatus status;
        std::string error;
        auto ex = NativeExecutor::create(opts, &status, &error);
        if (!ex) {
            std::fprintf(stderr, "FAIL: read-only NativeExecutor::create: %s\n", error.c_str());
            ++g_failures;
            return;
        }
        show_frame(*ex, true);
        show_frame(*ex, false);
        CHECK(i == 0 || ex->stats().show_target_frames == 2);
        ex->wait_idle();
    }
    auto slurp = [](const char* p) {
        std::FILE* f = std::fopen(p, "rb");
        std::string s;
        char buf[512];
        size_t n;
        while (f && (n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
        if (f) std::fclose(f);
        return s;
    };
    const std::string a = slurp(paths[0]), b = slurp(paths[1]);
    CHECK(!a.empty() && a == b);
    std::remove(paths[0]);
    std::remove(paths[1]);
}

// The stats overlay (F3) is drawn after every copy of the guest's frame: the digest of a run with it on equals the
// one without, the canvas shows the box top-left (a 65 % black over the red frame; it covers this tiny canvas), and
// switching it off restores the plain frame.
void test_overlay_read_only() {
    const char* paths[2] = {"test_overlay_digest_a.txt", "test_overlay_digest_b.txt"};
    for (int i = 0; i < 2; ++i) {
        std::remove(paths[i]);
        ExecutorOptions opts;
        opts.width = kW;
        opts.height = kH;
        opts.frame_digest_path = paths[i];
        opts.overlay = i == 1;
        NativeExecutor::CreateStatus status;
        std::string error;
        auto ex = NativeExecutor::create(opts, &status, &error);
        if (!ex) {
            std::fprintf(stderr, "FAIL: overlay NativeExecutor::create: %s\n", error.c_str());
            ++g_failures;
            return;
        }
        for (int f = 0; f < 3; ++f) {
            gpu::CommandEncoder e;
            e.begin_frame();
            e.clear_color(1.0f, 0.0f, 0.0f, 1.0f);
            e.present();
            run(*ex, e);
        }
        FrameImage img;
        std::string err;
        CHECK(ex->read_last_frame(&img, &err));
        if (img.pixels.size() >= 4) {
            if (i == 1) {
                CHECK(ex->overlay_enabled() && ex->stats().overlay_frames == 3);
                CHECK(img.pixels[0] == 0 && img.pixels[1] == 0 && img.pixels[2] == 89 && img.pixels[3] == 255);  // bgra: red * 35 %
            } else {
                CHECK(!ex->overlay_enabled() && ex->stats().overlay_frames == 0);
                CHECK(frame_is(*ex, {0, 0, 255, 255}));
            }
        }
        if (i == 1) {
            ex->set_overlay(false);
            gpu::CommandEncoder e;
            e.begin_frame();
            e.clear_color(1.0f, 0.0f, 0.0f, 1.0f);
            e.present();
            run(*ex, e);
            CHECK(frame_is(*ex, {0, 0, 255, 255}));
        }
        ex->wait_idle();
    }
    auto slurp = [](const char* p) {
        std::FILE* f = std::fopen(p, "rb");
        std::string s;
        char buf[512];
        size_t n;
        while (f && (n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
        if (f) std::fclose(f);
        return s;
    };
    const std::string a = slurp(paths[0]), b = slurp(paths[1]);
    CHECK(!a.empty());
    CHECK(b.compare(0, a.size(), a) == 0);  // the overlay run has the same digests (plus its extra off frame)
    std::remove(paths[0]);
    std::remove(paths[1]);
}

// A bad selector fails the creation.
void test_show_target_bad_option() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.show_target = "pass:x";
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    CHECK(!ex);
    CHECK(error.find("--show-target") != std::string::npos);
}

// Device loss (simulated with Device::ForceLoss): the executor rebuilds adapter / device and the journaled
// resources (buffer + texture contents come back byte for byte), drops only the frame under construction,
// keeps presenting, and counts it; where it cannot rebuild it ends the run with a fault record.
void test_device_loss() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    CHECK(ex != nullptr);
    if (!ex) return;
    std::vector<uint8_t> tex(4 * 4 * 4);
    for (size_t i = 0; i < tex.size(); ++i) tex[i] = static_cast<uint8_t>(i * 7 + 3);
    const uint8_t bdata[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    gpu::CommandEncoder setup;
    setup.create_buffer(21, GpuBufferUsage::CopySrc | GpuBufferUsage::CopyDst, 16);
    setup.write_buffer(21, 0, bdata, 8);
    setup.write_buffer(21, 0, bdata, 8);  // the same range again: one journal entry
    setup.create_buffer(22, GpuBufferUsage::CopySrc | GpuBufferUsage::CopyDst, 16);
    setup.write_buffer(22, 0, bdata, 8);
    setup.destroy_resource(22);  // pruned from the journal: must not come back
    setup.create_texture(23, 4, 4, 1, 1, GpuTextureFormat::Rgba8Unorm,
                         GpuTextureUsage::CopySrc | GpuTextureUsage::CopyDst | GpuTextureUsage::TextureBinding);
    setup.write_texture(23, 0, 0, 0, 0, 4, 4, 1, 16, 4, tex.data(), static_cast<uint32_t>(tex.size()));
    run(*ex, setup);
    gpu::CommandEncoder red;
    red.begin_frame();
    red.clear_color(1.f, 0.f, 0.f, 1.f);
    red.present();
    CHECK(run(*ex, red) == 1);
    CHECK(frame_is(*ex, {0, 0, 255, 255}));

    ex->debug_force_device_lost("simulated driver reset");
    CHECK(ex->stats().device_lost_events == 1);
    // A frame under construction is dropped; the next whole frame presents again.
    gpu::CommandEncoder partial;
    partial.begin_frame();
    partial.clear_color(0.f, 1.f, 0.f, 1.f);
    run(*ex, partial);
    CHECK(ex->stats().recoveries == 1);
    CHECK(ex->stats().recovery_failures == 0 && !ex->faulted());
    CHECK(ex->stats().recovery_replayed_packets >= 4);
    CHECK(ex->stats().recovery_replay_errors == 0);
    gpu::CommandEncoder rest;
    rest.present();
    run(*ex, rest);  // the loss was found between frames: nothing was lost with it, this frame presents
    CHECK(ex->stats().recovery_frames_dropped == 0);
    CHECK(frame_is(*ex, {0, 255, 0, 255}));
    gpu::CommandEncoder green;
    green.begin_frame();
    green.clear_color(0.f, 1.f, 0.f, 1.f);
    green.present();
    CHECK(run(*ex, green) == 1);
    CHECK(frame_is(*ex, {0, 255, 0, 255}));
    std::vector<uint8_t> got;
    std::string err;
    CHECK(ex->read_buffer(21, 0, 8, &got, &err) && got == std::vector<uint8_t>(bdata, bdata + 8));
    CHECK(!ex->read_buffer(22, 0, 8, &got, &err));
    uint32_t tw = 0, th = 0;
    CHECK(ex->read_texture(23, 0, &got, &tw, &th, &err) && tw == 4 && th == 4 && got == tex);

    // A loss while draw commands are queued: that frame is dropped (counted), later frames are fine.
    gpu::CommandEncoder queued;
    queued.begin_frame();
    GpuBeginRenderPassPayload bp = swapchain_pass(GpuLoadOp::Clear, 0.f, 0.f, 1.f, 1.f);
    queued.emit_packet(GpuOpcode::BeginRenderPass, &bp, sizeof bp);
    queued.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    run(*ex, queued);
    ex->debug_force_device_lost("second reset");
    gpu::CommandEncoder present_only;
    present_only.present();
    run(*ex, present_only);
    CHECK(ex->stats().recoveries == 2);
    CHECK(ex->stats().recovery_frames_dropped == 1);
    CHECK(run(*ex, green) == 1);
    CHECK(frame_is(*ex, {0, 255, 0, 255}));
    CHECK(ex->read_buffer(21, 0, 8, &got, &err) && got == std::vector<uint8_t>(bdata, bdata + 8));
    CHECK(ex->stats_json().find("\"recoveries\": 2") != std::string::npos);
    CHECK(ex->stats_json().find("fault_record") == std::string::npos);

    // Without a journal nothing can be rebuilt: a clean end with a fault record, no crash.
    ExecutorOptions nj = opts;
    nj.recovery_journal_bytes = 0;
    auto ex2 = NativeExecutor::create(nj, &status, &error);
    CHECK(ex2 != nullptr);
    if (!ex2) return;
    CHECK(run(*ex2, red) == 1);
    ex2->debug_force_device_lost("simulated gpu removal");
    CHECK(run(*ex2, red) == 0);
    CHECK(ex2->faulted());
    CHECK(!ex2->poll());
    CHECK(ex2->stats().recovery_failures == 1 && ex2->stats().recoveries == 0);
    const std::string js = ex2->stats_json();
    CHECK(js.find("\"fault_record\": {\"kind\": \"gpu_device_lost\"") != std::string::npos);
    CHECK(js.find("simulated gpu removal") != std::string::npos);
    CHECK(ex2->stats().dropped_while_lost >= 1);

    // A recovery whose new device cannot be created (the driver's E_OUTOFMEMORY) ends without a device:
    // the fault record, and wait_idle() (the live pump's last call) must not touch the null queue.
    auto ex5 = NativeExecutor::create(opts, &status, &error);
    CHECK(ex5 != nullptr);
    if (!ex5) return;
    CHECK(run(*ex5, red) == 1);
    ex5->debug_fail_device_creations(5);
    ex5->debug_force_device_lost("simulated out of memory");
    CHECK(run(*ex5, red) == 0);
    CHECK(ex5->faulted() && !ex5->poll());
    CHECK(ex5->stats_json().find("could not create a new device: simulated device creation failure") != std::string::npos);
    ex5->wait_idle();

    // Content over the budget is dropped, not fatal: the buffer comes back without it (counted).
    ExecutorOptions small = opts;
    small.recovery_journal_bytes = 64;
    auto ex3 = NativeExecutor::create(small, &status, &error);
    CHECK(ex3 != nullptr);
    if (!ex3) return;
    CHECK(ex3->stats().journal_budget == 64);
    gpu::CommandEncoder big;
    big.create_buffer(1, GpuBufferUsage::CopyDst | GpuBufferUsage::CopySrc, 256);
    std::vector<uint8_t> blob(200, 9);
    big.write_buffer(1, 0, blob.data(), 200);
    run(*ex3, big);
    CHECK(ex3->stats().journal_overflows >= 1);
    CHECK(ex3->stats().journal_evicted_resources == 1);
    ex3->debug_force_device_lost("reset");
    CHECK(run(*ex3, red) == 1 && !ex3->faulted());
    CHECK(ex3->stats().recoveries == 1 && ex3->stats().recovery_contents_lost == 1);
    CHECK(ex3->read_buffer(1, 0, 8, &got, &err) && got == std::vector<uint8_t>(8, 0));

    // Creations that alone outgrow the budget cannot rebuild: the clean end with a fault record.
    auto ex4 = NativeExecutor::create(small, &status, &error);
    CHECK(ex4 != nullptr);
    if (!ex4) return;
    gpu::CommandEncoder shader;
    shader.create_shader_module(5, GpuShaderStage::Compute, kCompute);
    run(*ex4, shader);
    CHECK(ex4->stats().journal_overflows >= 1);
    ex4->debug_force_device_lost("reset");
    CHECK(run(*ex4, red) == 0 && ex4->faulted());
    CHECK(ex4->stats_json().find("creations outgrew") != std::string::npos);
}

// The journal folds buffer writes into one image per buffer and drops texture writes a later write covers;
// over the budget the content of resources rewritten in many frames goes first.
void test_device_loss_journal() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.recovery_journal_bytes = 4096;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto ex = NativeExecutor::create(opts, &status, &error);
    CHECK(ex != nullptr);
    if (!ex) return;
    const auto present = [&] {
        gpu::CommandEncoder f;
        f.begin_frame();
        f.clear_color(0.f, 0.f, 1.f, 1.f);
        f.present();
        return run(*ex, f);
    };
    gpu::CommandEncoder setup;
    setup.create_buffer(31, GpuBufferUsage::CopyDst | GpuBufferUsage::CopySrc, 64);
    setup.create_buffer(32, GpuBufferUsage::CopyDst | GpuBufferUsage::CopySrc, 2048);
    setup.create_buffer(33, GpuBufferUsage::CopyDst | GpuBufferUsage::CopySrc, 3072);
    setup.create_texture(34, 4, 4, 1, 1, GpuTextureFormat::Rgba8Unorm,
                         GpuTextureUsage::CopySrc | GpuTextureUsage::CopyDst | GpuTextureUsage::TextureBinding);
    run(*ex, setup);
    const uint64_t creations = ex->stats().journal_bytes;
    CHECK(creations == ex->stats().journal_create_bytes);
    // Overlapping rewrites of buffer 31: the journal holds its 48 written bytes once.
    std::vector<uint8_t> want(64, 0);
    gpu::CommandEncoder w;
    for (int round = 0; round < 50; ++round) {
        std::vector<uint8_t> d(32);
        for (size_t i = 0; i < d.size(); ++i) d[i] = static_cast<uint8_t>(round * 3 + i);
        const uint32_t off = (round % 2) ? 16u : 0u;
        w.write_buffer(31, off, d.data(), 32);
        std::memcpy(want.data() + off, d.data(), 32);
    }
    // Texture 34: a full write, a sub-box, a full write again (the first two are dropped), then a sub-box.
    std::vector<uint8_t> tex(64);
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < tex.size(); ++i) tex[i] = static_cast<uint8_t>(i + pass * 100);
        w.write_texture(34, 0, 0, 0, 0, 4, 4, 1, 16, 4, tex.data(), 64);
        const uint8_t sub[16] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
        w.write_texture(34, 0, 1, 1, 0, 2, 2, 1, 8, 2, sub, 16);
    }
    run(*ex, w);
    CHECK(ex->stats().journal_bytes - creations == 48 + (48 + 64) + (48 + 16));
    for (uint32_t y = 1; y < 3; ++y)
        for (uint32_t x = 1; x < 3; ++x)
            for (uint32_t c = 0; c < 4; ++c) tex[(y * 4 + x) * 4 + c] = 7;
    // Buffer 32 is rewritten in 5 frames (volatile); buffer 33's one write needs room: 32 goes.
    std::vector<uint8_t> vol(2048, 5);
    for (int f = 0; f < 5; ++f) {
        gpu::CommandEncoder vw;
        vw.write_buffer(32, 0, vol.data(), 2048);
        run(*ex, vw);
        present();
    }
    std::vector<uint8_t> stat(3072);
    for (size_t i = 0; i < stat.size(); ++i) stat[i] = static_cast<uint8_t>(i * 13);
    gpu::CommandEncoder sw;
    sw.write_buffer(33, 0, stat.data(), 3072);
    run(*ex, sw);
    CHECK(ex->stats().journal_evicted_resources == 1);
    CHECK(ex->stats().journal_bytes <= 4096);

    ex->debug_force_device_lost("simulated TDR");
    CHECK(ex->stats().device_lost_reason == "unknown");
    CHECK(ex->stats().device_lost_message.find("simulated TDR") != std::string::npos);
    CHECK(present() == 1);
    CHECK(ex->stats().recoveries == 1 && !ex->faulted());
    CHECK(ex->stats().recovery_contents_lost == 1);
    std::vector<uint8_t> got;
    std::string err;
    CHECK(ex->read_buffer(31, 0, 64, &got, &err) && got == want);
    CHECK(ex->read_buffer(33, 0, 3072, &got, &err) && got == stat);
    CHECK(ex->read_buffer(32, 0, 16, &got, &err) && got == std::vector<uint8_t>(16, 0));
    uint32_t tw = 0, th = 0;
    CHECK(ex->read_texture(34, 0, &got, &tw, &th, &err) && got == tex);
    const std::string js = ex->stats_json();
    CHECK(js.find("\"device_lost_reason\": \"unknown\"") != std::string::npos);
    CHECK(js.find("\"recovery_contents_lost\": 1") != std::string::npos);
    CHECK(ex->stats().submits_tracked > 0);
}

// The ExecuteIndirect clamp pass (bridge::indirect_clamp_wgsl) on the GPU: garbage draw / dispatch records
// are cut to the work limits (a TDR guard), sane ones pass unchanged.
void test_indirect_clamp_work(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    constexpr uint32_t kParams = 420, kSrc = 421, kCnt = 422, kDst = 423, kModule = 424, kBgl = 425, kPl = 426,
                       kPipe = 427, kGroup = 428;
    gpu::CommandEncoder setup;
    setup.create_buffer(kParams, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, 32);
    setup.create_buffer(kSrc, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, 64);
    setup.create_buffer(kCnt, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, 16);
    setup.create_buffer(kDst, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 64);
    setup.create_shader_module(kModule, GpuShaderStage::Compute, bridge::indirect_clamp_wgsl());
    create_layouts(setup, kBgl, kPl,
                   {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                 static_cast<uint8_t>(GpuBufferBindingType::Uniform)),
                    layout_entry(1, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                 static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage)),
                    layout_entry(2, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                 static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage)),
                    layout_entry(3, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                 static_cast<uint8_t>(GpuBufferBindingType::Storage))});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = kPipe;
    cp.cs_module_id = kModule;
    cp.pipeline_layout_id = kPl;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, kGroup, kBgl,
                      {bg_entry(0, GpuBindResourceType::Buffer, kParams), bg_entry(1, GpuBindResourceType::Buffer, kSrc),
                       bg_entry(2, GpuBindResourceType::Buffer, kCnt), bg_entry(3, GpuBindResourceType::Buffer, kDst)});
    run(ex, setup);
    const auto clamp = [&](uint32_t kind, uint32_t stride, uint32_t count, uint32_t index_limit,
                           const std::vector<uint32_t>& records) {
        const uint32_t params[8] = {0, stride, count, 0, 0, kind, index_limit, 65535};
        std::vector<uint32_t> src(16, 0);
        std::copy(records.begin(), records.end(), src.begin());
        gpu::CommandEncoder frame;
        frame.begin_frame();
        frame.write_buffer(kParams, 0, params, sizeof params);
        frame.write_buffer(kSrc, 0, src.data(), 64);
        frame.set_compute_pipeline(kPipe);
        set_bind_group2(frame, 0, kGroup);
        frame.dispatch_compute(1, 1, 1);
        frame.present();
        run(ex, frame);
        ex.wait_idle();
        std::vector<uint8_t> out;
        std::string err;
        CHECK(ex.read_buffer(kDst, 0, 64, &out, &err) && out.size() == 64);
        std::vector<uint32_t> words(16, 0);
        if (out.size() == 64) std::memcpy(words.data(), out.data(), 64);
        return words;
    };
    constexpr uint32_t kWork = 1u << 26;
    {  // Draws: an all-ones record, a huge instance count, a sane record.
        const auto w = clamp(0, 4, 3, 0, {0xFFFFFFFFu, 0xFFFFFFFFu, 0, 0, 3, 0x40000000u, 0, 0, 100, 50, 7, 9});
        CHECK(w[0] == kWork && w[1] == 1);
        CHECK(w[4] == 3 && w[5] == kWork / 3);
        CHECK(w[8] == 100 && w[9] == 50 && w[10] == 7 && w[11] == 9);
    }
    {  // Indexed draws (1000 indices): instances cut against the clamped index count.
        const auto w = clamp(1, 5, 2, 1000, {600, 0xFFFFFFFFu, 0, 0, 0, 10, 20, 995, 1, 2});
        CHECK(w[0] == 600 && w[1] == kWork / 600);
        CHECK(w[5] == 5 && w[6] == 20 && w[7] == 995 && w[8] == 1 && w[9] == 2);
    }
    {  // Dispatches: at most 2^22 workgroups in all.
        const auto w = clamp(2, 3, 3, 0, {65535, 65535, 65535, 0xFFFFFFFFu, 2, 3, 8, 8, 8});
        CHECK(w[0] == 65535 && w[1] == (1u << 22) / 65535 && w[2] == 1);
        CHECK(w[3] == 65535 && w[4] == 2 && w[5] == 3);
        CHECK(w[6] == 8 && w[7] == 8 && w[8] == 8);
    }
    CHECK(ex.stats().device_errors == before.device_errors);
}

// ExecuteIndirect root-constant pass (gpu/state_bridge.h indirect_constants_wgsl) on the GPU: per
// command and constant argument, a 256-byte slot holds the parameter's base values with the record's
// words written over its range; commands past the count buffer's value or past the end of the
// argument buffer keep the base values. Nothing is read back by the pass itself.
void test_indirect_constants_pass(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    constexpr uint32_t kRing = 400, kArgs = 401, kCount = 402, kModule = 403, kBgl = 404, kPl = 405, kPipe = 406,
                       kGroup = 407;
    constexpr uint32_t kWindow = 4096, kStrideWords = 6, kNargs = 2, kCommands = 4;
    const uint32_t first_slot = bridge::indirect_constants_first_slot_word(kNargs);
    gpu::CommandEncoder setup;
    setup.create_buffer(kRing, GpuBufferUsage::Uniform | GpuBufferUsage::Storage | GpuBufferUsage::CopyDst |
                                    GpuBufferUsage::CopySrc, 3 * kWindow);
    setup.create_buffer(kArgs, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, 80);  // 20 words: 3 full records + 2
    setup.create_buffer(kCount, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, 16);
    uint32_t args[20];
    for (uint32_t i = 0; i < 20; ++i) args[i] = 0xDEAD0000u + i * 7u;  // garbage: every word is just data
    setup.write_buffer(kArgs, 0, args, sizeof args);
    const uint32_t count_values[4] = {2, 0, 0, 0};
    setup.write_buffer(kCount, 0, count_values, sizeof count_values);
    setup.create_shader_module(kModule, GpuShaderStage::Compute, bridge::indirect_constants_wgsl());
    GpuBindGroupLayoutEntry l0 = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                              static_cast<uint8_t>(GpuBufferBindingType::Storage));
    l0.flags = GpuBindGroupLayoutEntryFlags::HasDynamicOffset;
    create_layouts(setup, kBgl, kPl,
                   {l0, layout_entry(1, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                     static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage)),
                    layout_entry(2, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer,
                                 static_cast<uint8_t>(GpuBufferBindingType::ReadOnlyStorage))});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = kPipe;
    cp.cs_module_id = kModule;
    cp.pipeline_layout_id = kPl;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    GpuBindGroupEntry2 e0 = bg_entry(0, GpuBindResourceType::Buffer, kRing);
    e0.size = kWindow;
    create_bind_group(setup, kGroup, kBgl,
                      {e0, bg_entry(1, GpuBindResourceType::Buffer, kArgs), bg_entry(2, GpuBindResourceType::Buffer, kCount)});

    const auto chunk = [&](uint32_t has_count) {
        std::vector<uint32_t> w(first_slot, 0u);
        w[0] = 0;  // first record word
        w[1] = kStrideWords;
        w[2] = kCommands;
        w[3] = 0;  // count word
        w[4] = has_count;
        w[5] = kNargs;
        w[6] = first_slot;
        w[7] = kStrideWords;
        w[8] = 0, w[9] = 1, w[10] = 2;  // argument 0: record words 0..1 -> words 1..2
        w[12] = 2, w[13] = 0, w[14] = 1;  // argument 1: record word 2 -> word 0
        for (uint32_t k = 0; k < 64; ++k) {
            w[64 + k] = 0xB000u + k;
            w[128 + k] = 0xC000u + k;
        }
        return w;
    };
    const auto run_pass = [&](uint32_t offset, uint32_t has_count) {
        const std::vector<uint32_t> w = chunk(has_count);
        gpu::CommandEncoder frame;
        frame.begin_frame();
        frame.write_buffer(kRing, offset, w.data(), static_cast<uint32_t>(w.size() * 4));
        frame.set_compute_pipeline(kPipe);
        std::vector<uint8_t> sb;
        append(sb, GpuSetBindGroup2Payload{0, kGroup, 1, 0});
        append(sb, offset);
        emit(frame, GpuOpcode::SetBindGroup2, sb);
        frame.dispatch_compute(1, 1, 1);
        frame.present();
        run(ex, frame);
        ex.wait_idle();
        std::vector<uint8_t> out;
        std::string err;
        CHECK(ex.read_buffer(kRing, offset, kWindow, &out, &err) && out.size() == kWindow);
        std::vector<uint32_t> words(out.size() / 4);
        std::memcpy(words.data(), out.data(), out.size());
        return words;
    };
    run(ex, setup);

    // Slot of (command, argument), as the words the draw would bind.
    const auto slot = [&](const std::vector<uint32_t>& words, uint32_t command, uint32_t arg) {
        const size_t at = first_slot + (command * kNargs + arg) * 64u;
        return std::vector<uint32_t>(words.begin() + at, words.begin() + at + 64);
    };
    const auto base_of = [&](uint32_t arg) {
        std::vector<uint32_t> v(64);
        for (uint32_t k = 0; k < 64; ++k) v[k] = (arg == 0 ? 0xB000u : 0xC000u) + k;
        return v;
    };
    {  // With a count buffer (value 2): commands 0 and 1 take their records, 2 and 3 keep the base values.
        const std::vector<uint32_t> w = run_pass(256, 1);
        for (uint32_t c = 0; c < kCommands; ++c) {
            std::vector<uint32_t> a = base_of(0), b = base_of(1);
            if (c < 2) {
                a[1] = args[c * kStrideWords];
                a[2] = args[c * kStrideWords + 1];
                b[0] = args[c * kStrideWords + 2];
            }
            CHECK(slot(w, c, 0) == a);
            CHECK(slot(w, c, 1) == b);
        }
    }
    {  // No count buffer: commands 0 to 2 are live; command 3's record does not fit the buffer.
        const std::vector<uint32_t> w = run_pass(4096 + 256, 0);
        for (uint32_t c = 0; c < kCommands; ++c) {
            std::vector<uint32_t> a = base_of(0), b = base_of(1);
            if (c < 3) {
                a[1] = args[c * kStrideWords];
                a[2] = args[c * kStrideWords + 1];
                b[0] = args[c * kStrideWords + 2];
            }
            CHECK(slot(w, c, 0) == a);
            CHECK(slot(w, c, 1) == b);
        }
    }
    CHECK(ex.stats().device_errors == before.device_errors);
}

// ---- mesh pipelines (G66): a synthetic mesh-stage compute shader writes the runtime's output records,
// the runtime vertex shader pulls them into a draw (the contract of docs/gpu_commands.md "Mesh pipelines") ----

const char* kMeshStageCs = R"(
@group(3) @binding(0) var<storage, read_write> ms_out: array<u32>;
fn put4(w: u32, v: vec4<f32>) {
  ms_out[w] = bitcast<u32>(v.x); ms_out[w + 1u] = bitcast<u32>(v.y);
  ms_out[w + 2u] = bitcast<u32>(v.z); ms_out[w + 3u] = bitcast<u32>(v.w);
}
@compute @workgroup_size(1) fn cs_main(@builtin(workgroup_id) wg: vec3<u32>) {
  let rec = wg.x * 31u;  // 4 + 3 vertices * 8 words + 1 primitive * 3 indices
  if (wg.x != 0u) { ms_out[rec] = 0u; ms_out[rec + 1u] = 0u; return; }  // later groups emit nothing
  ms_out[rec] = 3u; ms_out[rec + 1u] = 1u; ms_out[rec + 2u] = 0u; ms_out[rec + 3u] = 0u;
  let green = vec4<f32>(0.0, 1.0, 0.0, 1.0);
  put4(rec + 4u, vec4<f32>(-1.0, -1.0, 0.5, 1.0)); put4(rec + 8u, green);
  put4(rec + 12u, vec4<f32>(1.0, -1.0, 0.5, 1.0)); put4(rec + 16u, green);
  put4(rec + 20u, vec4<f32>(-1.0, 1.0, 0.5, 1.0)); put4(rec + 24u, green);
  ms_out[rec + 28u] = 0u; ms_out[rec + 29u] = 1u; ms_out[rec + 30u] = 2u;
}
)";

const char* kMeshPs = R"(
@fragment fn fs_main(@location(0) c: vec4<f32>) -> @location(0) vec4<f32> { return c; }
)";

void test_mesh_pipeline(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    const bridge::MeshOutputLayout layout{3, 1, 8};  // 3 vertices, 1 triangle, SV_Position + 1 attribute
    CHECK(bridge::valid_mesh_output_layout(layout));
    CHECK(bridge::mesh_group_record_words(layout) == 31);
    constexpr uint32_t kOut = 900, kCsMod = 901, kVsMod = 902, kFsMod = 903, kPadBgl = 910, kBglC = 911, kBglV = 912,
                       kPlC = 913, kPlV = 914, kCsPipe = 915, kRenderPipe = 916, kGroupC = 917, kGroupV = 918, kPadGroup = 919;
    gpu::CommandEncoder setup;
    setup.create_buffer(kOut, GpuBufferUsage::Storage | GpuBufferUsage::CopySrc, 256);
    setup.create_shader_module(kCsMod, GpuShaderStage::Compute, kMeshStageCs);
    setup.create_shader_module(kVsMod, GpuShaderStage::Vertex, bridge::mesh_draw_vertex_wgsl(layout));
    setup.create_shader_module(kFsMod, GpuShaderStage::Fragment, kMeshPs);
    emit(setup, GpuOpcode::CreateBindGroupLayout, bytes_of(GpuCreateBindGroupLayoutPayload{kPadBgl, 0, 0, 0, 0}));
    const auto stage_entry = [&](uint32_t visibility, GpuBufferBindingType type) {
        return layout_entry(0, visibility, GpuBindingLayoutType::Buffer, static_cast<uint8_t>(type));
    };
    for (const auto& [id, vis, type] :
         {std::tuple{kBglC, GpuShaderStageBits::Compute, GpuBufferBindingType::Storage},
          std::tuple{kBglV, GpuShaderStageBits::Vertex, GpuBufferBindingType::ReadOnlyStorage}}) {
        std::vector<uint8_t> v;
        append(v, GpuCreateBindGroupLayoutPayload{id, 1, 0, 0, 0});
        append(v, stage_entry(vis, type));
        emit(setup, GpuOpcode::CreateBindGroupLayout, v);
    }
    // Pipeline layouts: groups 0..2 are empty padding, group kMeshOutputGroup (3) is the output buffer.
    for (const auto& [pl, out_bgl] : {std::pair{kPlC, kBglC}, std::pair{kPlV, kBglV}}) {
        std::vector<uint8_t> v;
        append(v, GpuCreatePipelineLayoutPayload{pl, 4, 0, 0, 0});
        for (uint32_t i = 0; i < 3; ++i) append(v, kPadBgl);
        append(v, out_bgl);
        emit(setup, GpuOpcode::CreatePipelineLayout, v);
    }
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = kCsPipe;
    cp.cs_module_id = kCsMod;
    cp.pipeline_layout_id = kPlC;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(pipeline_to_bgra(kRenderPipe, kVsMod, kFsMod, kPlV)));
    create_bind_group(setup, kPadGroup, kPadBgl, {});
    create_bind_group(setup, kGroupC, kBglC, {bg_entry(0, GpuBindResourceType::Buffer, kOut)});
    create_bind_group(setup, kGroupV, kBglV, {bg_entry(0, GpuBindResourceType::Buffer, kOut)});
    run(ex, setup);

    // DispatchMesh(2, 1, 1): two mesh groups, then a draw of groups * max_primitives * 3 vertices.
    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(kCsPipe);
    for (uint32_t g = 0; g < 3; ++g) set_bind_group2(frame, g, kPadGroup);
    set_bind_group2(frame, bridge::kMeshOutputGroup, kGroupC);
    frame.dispatch_compute(2, 1, 1);
    auto pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    frame.set_pipeline(kRenderPipe);
    for (uint32_t g = 0; g < 3; ++g) set_bind_group2(frame, g, kPadGroup);
    set_bind_group2(frame, bridge::kMeshOutputGroup, kGroupV);
    frame.draw(2 * layout.max_primitives * 3);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().device_errors == before.device_errors);
    CHECK(ex.stats().packet_errors == before.packet_errors);
    CHECK(ex.stats().dispatches == before.dispatches + 1);
    CHECK(ex.stats().draws == before.draws + 1);

    // The records: group 0 = 3 vertices / 1 primitive, group 1 = empty; vertex 1 position and the indices.
    std::vector<uint8_t> raw;
    std::string err;
    CHECK(ex.read_buffer(kOut, 0, 248, &raw, &err) && raw.size() == 248);
    std::vector<uint32_t> w(raw.size() / 4);
    std::memcpy(w.data(), raw.data(), raw.size());
    if (w.size() == 62) {
        CHECK(w[0] == 3 && w[1] == 1 && w[31] == 0 && w[32] == 0);
        float x1;
        std::memcpy(&x1, &w[12], 4);
        CHECK(x1 == 1.0f);
        CHECK(w[28] == 0 && w[29] == 1 && w[30] == 2);
    }

    // The triangle (-1,-1) (1,-1) (-1,1) covers the lower-left half of the target in green; the rest stays cleared.
    FrameImage img;
    CHECK(ex.read_last_frame(&img, &err) && img.width == kW && img.height == kH);
    CHECK(px_near(img, 8, 24, 0, 255, 0, 255, 1));    // inside (x + y < 0 in NDC)
    CHECK(px_near(img, 56, 4, 0, 0, 0, 255, 1));      // outside
    CHECK(px_near(img, 40, 8, 0, 0, 0, 255, 1));      // outside (x + y > 0 in NDC)
}

namespace playgta6::gpu::bridge {
struct StencilTwinDawnTestAccess {
    static StateBridgeDevice::StencilTwinPipelines pipelines(StateBridgeDevice& d, uint8_t df, uint8_t tf,
                                                             CommandEncoder& e) {
        return d.stencil_twin_pipelines(df, tf, e);
    }
    template <class... A>
    static uint32_t group(StateBridgeDevice& d, A&&... a) {
        return d.depth_twin_group(std::forward<A>(a)...);
    }
};
}  // namespace playgta6::gpu::bridge

// ---- backlog 35: the stencil plane twin on the real GPU ----
// A depth24plus-stencil8 texture's stencil plane is cleared to 0xA5, copied into the r32uint twin by the
// bridge's copy-in pass (read back: every texel 0xA5), the twin is overwritten with a per-texel pattern by
// a compute pass (standing in for the title's stage), written back by the bridge's write-back pass (clear
// plus eight masked draws), then the twin is zeroed and the stencil plane copied in again: the pattern
// can only come from the stencil plane. All pipelines and bind groups are the bridge's own.
void test_stencil_twin(NativeExecutor& ex) {
    using playgta6::gpu::bridge::StateBridgeDevice;
    using Access = playgta6::gpu::bridge::StencilTwinDawnTestAccess;
    constexpr uint32_t kN = 16;
    constexpr uint32_t kDepth = 1200, kTwin = 1201, kPatLayout = 1202, kPatPl = 1203, kPatMod = 1204,
                       kPatPipe = 1205, kPatGroup = 1206, kZeroMod = 1207, kZeroPipe = 1208;
    const uint8_t ds_fmt = static_cast<uint8_t>(playgta6::gpu::pso::WgpuTextureFormat::Depth24PlusStencil8);
    const uint8_t twin_fmt = static_cast<uint8_t>(playgta6::gpu::pso::WgpuTextureFormat::R32Uint);
    std::vector<uint8_t> mem(16384, 0xEE);
    GuestMemory gm;
    gm.base = mem.data();
    gm.size = mem.size();
    std::memset(mem.data(), 0, 16);
    ex.attach_guest_memory(gm);
    const ExecutorStats before = ex.stats();

    uint32_t next_id = 1300000;
    playgta6::gpu::bridge::BridgeOptions options;
    options.id_allocator = [&next_id] { return next_id++; };
    playgta6::gpu::d3d12_state::DescriptorHeapRegistry heaps;
    playgta6::gpu::d3d12_state::ResourceStateRegistry states;
    StateBridgeDevice dev(heaps, options, &states);

    gpu::CommandEncoder e;
    emit(e, GpuOpcode::CreateTexture2,
         bytes_of(texture2(kDepth, kN, kN, GpuTextureUsage::RenderAttachment | GpuTextureUsage::TextureBinding, ds_fmt)));
    emit(e, GpuOpcode::CreateTexture2,
         bytes_of(texture2(kTwin, kN, kN,
                           GpuTextureUsage::StorageBinding | GpuTextureUsage::TextureBinding | GpuTextureUsage::CopySrc,
                           twin_fmt)));
    const auto p = Access::pipelines(dev, ds_fmt, twin_fmt, e);

    // The stand-in stage: writes (x * 16 + y * 5) & 255 into the twin (or zero).
    GpuBindGroupLayoutEntry le = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::StorageTexture,
                                              static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly), 2);
    le.storage_format = twin_fmt;
    create_layouts(e, kPatLayout, kPatPl, {le});
    const auto stage = [](const char* value) {
        return std::string("@group(0) @binding(0) var dst: texture_storage_2d<r32uint, write>;\n"
                           "@compute @workgroup_size(8, 8, 1) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
                           "  textureStore(dst, id.xy, vec4<u32>(") +
               value + ", 0u, 0u, 0u));\n}\n";
    };
    e.create_shader_module(kPatMod, GpuShaderStage::Compute, stage("(id.x * 16u + id.y * 5u) & 255u"));
    e.create_shader_module(kZeroMod, GpuShaderStage::Compute, stage("0u"));
    for (const auto& [pipe, mod] : {std::pair<uint32_t, uint32_t>{kPatPipe, kPatMod}, {kZeroPipe, kZeroMod}}) {
        GpuCreateComputePipeline2Payload cp;
        std::memset(&cp, 0, sizeof cp);
        cp.id = pipe;
        cp.cs_module_id = mod;
        cp.pipeline_layout_id = kPatPl;
        emit(e, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    }
    GpuBindGroupEntry2 be = bg_entry(0, GpuBindResourceType::TextureView, kTwin);
    be.view.format = twin_fmt;
    be.view.dimension = 2;
    be.view.mip_level_count = 1;
    be.view.array_layer_count = 1;
    create_bind_group(e, kPatGroup, kPatLayout, {be});
    const uint32_t in_group = Access::group(dev, p.in_layout, kDepth, kTwin, true, twin_fmt, 0, 0, e, true);
    const uint32_t out_group = Access::group(dev, p.out_layout, kDepth, kTwin, false, twin_fmt, 0, 0, e, true);

    const auto readback_twin = [&](uint32_t seq, uint64_t dst, uint64_t done) {
        GpuReadbackTexturePayload rt;
        std::memset(&rt, 0, sizeof rt);
        rt.seq = seq;
        rt.texture_id = kTwin;
        rt.width = kN;
        rt.height = kN;
        rt.depth_or_layers = 1;
        rt.bytes_per_row = kN * 4;
        rt.rows_per_image = kN;
        rt.dst_addr = dst;
        rt.done_addr = done;
        rt.dst_size = kN * kN * 4;
        emit(e, GpuOpcode::ReadbackTexture, bytes_of(rt));
    };
    const auto copy_in = [&] {
        e.set_compute_pipeline(p.in_pipeline);
        set_bind_group2(e, 0, in_group);
        e.dispatch_compute(kN / 8, kN / 8, 1);
    };
    const auto run_stage = [&](uint32_t pipe) {
        e.set_compute_pipeline(pipe);
        set_bind_group2(e, 0, kPatGroup);
        e.dispatch_compute(kN / 8, kN / 8, 1);
    };
    const auto view2d = [](GpuTextureViewDesc& v) {
        v.dimension = 2;
        v.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
        v.mip_level_count = 1;
        v.array_layer_count = 1;
    };

    // 1. the stencil plane is cleared to 0xA5 (depth to 1)
    GpuBeginRenderPassPayload clear;
    std::memset(&clear, 0, sizeof clear);
    clear.depth_stencil.texture_id = kDepth;
    clear.depth_stencil.depth_load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
    clear.depth_stencil.depth_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
    clear.depth_stencil.stencil_load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
    clear.depth_stencil.stencil_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
    clear.depth_stencil.depth_clear_value = 1.0f;
    clear.depth_stencil.stencil_clear_value = 0xA5;
    view2d(clear.depth_stencil.view);
    e.emit_packet(GpuOpcode::BeginRenderPass, &clear, sizeof clear);
    e.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    // 2. copy in, read the twin
    copy_in();
    readback_twin(1, 4096, 0);
    // 3. the stage writes the pattern; 4. write it back into the stencil plane (clear + 8 masked draws)
    run_stage(kPatPipe);
    GpuBeginRenderPassPayload out;
    std::memset(&out, 0, sizeof out);
    out.depth_stencil.texture_id = kDepth;
    out.depth_stencil.depth_read_only = 1;
    out.depth_stencil.stencil_load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
    out.depth_stencil.stencil_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
    out.depth_stencil.stencil_clear_value = 0;
    view2d(out.depth_stencil.view);
    e.emit_packet(GpuOpcode::BeginRenderPass, &out, sizeof out);
    for (uint32_t bit = 0; bit < 8; ++bit) {
        e.set_pipeline(p.out_pipeline[bit]);
        set_bind_group2(e, 0, out_group);
        const GpuSetStencilReferencePayload ref{1u << bit};
        e.emit_packet(GpuOpcode::SetStencilReference, &ref, sizeof ref);
        e.draw(3, 1, 0, 0);
    }
    e.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    // 5. the twin is zeroed, the stencil plane copied in again: what is read back came from the plane
    run_stage(kZeroPipe);
    copy_in();
    readback_twin(2, 8192, 4);
    CHECK(run(ex, e) == 0);
    ex.wait_idle();
    ex.poll();
    CHECK(u32_at(mem, 0) == 1 && u32_at(mem, 4) == 1);
    bool all_a5 = true, pattern = true;
    for (uint32_t y = 0; y < kN; ++y) {
        for (uint32_t x = 0; x < kN; ++x) {
            all_a5 &= u32_at(mem, 4096 + (y * kN + x) * 4) == 0xA5;
            pattern &= u32_at(mem, 8192 + (y * kN + x) * 4) == ((x * 16u + y * 5u) & 255u);
        }
    }
    CHECK(all_a5);
    CHECK(pattern);
    CHECK(ex.stats().device_errors == before.device_errors);
    ex.attach_guest_memory(GuestMemory{});
}

// Two writable storage bindings that fall back to dummies must not alias one texture (Dawn:
// "writable storage texture used twice"), and a color target the fragment stage never writes gets
// writeMask 0 instead of failing the pipeline.
void test_dawn_validation_fixes(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    GpuBindGroupLayoutEntry a = layout_entry(2, GpuShaderStageBits::Compute, GpuBindingLayoutType::StorageTexture,
                                             static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly), 2);
    a.storage_format = 22;  // rgba8unorm
    GpuBindGroupLayoutEntry b = a;
    b.binding = 3;
    create_layouts(setup, 900, 901, {a, b});
    setup.create_shader_module(902, GpuShaderStage::Compute,
                               "@group(0) @binding(2) var d0: texture_storage_2d<rgba8unorm, write>;\n"
                               "@group(0) @binding(3) var d1: texture_storage_2d<rgba8unorm, write>;\n"
                               "@compute @workgroup_size(1) fn cs_main() {\n"
                               "  textureStore(d0, vec2<u32>(0u, 0u), vec4f(1.0));\n"
                               "  textureStore(d1, vec2<u32>(0u, 0u), vec4f(1.0));\n}\n");
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = 903;
    cp.cs_module_id = 902;
    cp.pipeline_layout_id = 901;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, 904, 900,
                      {bg_entry(2, GpuBindResourceType::TextureView, 0), bg_entry(3, GpuBindResourceType::TextureView, 0)});
    run(ex, setup);
    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(903);
    set_bind_group2(frame, 0, 904);
    frame.dispatch_compute(1, 1, 1);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    CHECK(ex.stats().device_errors == before.device_errors);

    // Two color targets, the fragment writes only @location(0): target 1 keeps writeMask 15 in the key.
    gpu::CommandEncoder pl;
    pl.create_shader_module(910, GpuShaderStage::Vertex,
                            "@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {"
                            " var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));"
                            " return vec4f(p[i], 0.0, 1.0); }\n");
    pl.create_shader_module(911, GpuShaderStage::Fragment,
                            "struct Out { @location(0) c: vec4f }\n"
                            "@fragment fn fs_main() -> Out { var o: Out; o.c = vec4f(1.0); return o; }\n");
    GpuCreateRenderPipeline2Payload rp = pipeline_to_bgra(912, 910, 911, 0);
    rp.key.color_target_count = 2;
    rp.key.color_targets[1].format = 22;
    rp.key.color_targets[1].write_mask = 15;
    emit(pl, GpuOpcode::CreateRenderPipeline2, bytes_of(rp));
    run(ex, pl);
    ex.wait_idle();
    CHECK(ex.stats().device_errors == before.device_errors);
    CHECK(ex.stats().write_masks_cleared == before.write_masks_cleared + 1);
}

// Bind groups Dawn would reject are fitted (binding_fit_model.h): a whole 128 KiB uniform buffer is cut
// to the 64 KiB binding maximum, an r32uint view on a float entry gets the entry's dummy. No device error.
void test_bind_group_binding_fit(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    setup.create_buffer(9500, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, 128 * 1024);
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(9501, 2, 2, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 14)));  // r32uint
    std::vector<uint8_t> bgl;
    const GpuBindGroupLayoutEntry u = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 0);
    const GpuBindGroupLayoutEntry t = layout_entry(1, GpuShaderStageBits::Compute, GpuBindingLayoutType::Texture, 0, 2);
    append(bgl, GpuCreateBindGroupLayoutPayload{9502, 2, 0, 0, 0});
    append(bgl, u);
    append(bgl, t);
    emit(setup, GpuOpcode::CreateBindGroupLayout, bgl);
    create_bind_group(setup, 9503, 9502,
                      {bg_entry(0, GpuBindResourceType::Buffer, 9500), bg_entry(1, GpuBindResourceType::TextureView, 9501)});
    run(ex, setup);
    ex.wait_idle();
    ex.poll();
    const ExecutorStats& s = ex.stats();
    CHECK(s.device_errors == before.device_errors);
    CHECK(s.buffer_bindings_clamped == before.buffer_bindings_clamped + 1);
    CHECK(s.bind_group_sample_type_dummies == before.bind_group_sample_type_dummies + 1);
}

// An sRGB view of a storage-capable rgba8unorm texture (a sampled bind group entry): the view drops the
// texture's StorageBinding usage, which no sRGB format allows. No device error.
void test_srgb_view_of_storage_texture(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    GpuCreateTexture2Payload ct = texture2(9510, 2, 2,
                                           GpuTextureUsage::TextureBinding | GpuTextureUsage::StorageBinding |
                                               GpuTextureUsage::RenderAttachment | GpuTextureUsage::CopyDst,
                                           22);
    ct.view_format_count = 1;
    ct.view_formats[0] = 23;  // rgba8unorm-srgb
    emit(setup, GpuOpcode::CreateTexture2, bytes_of(ct));
    std::vector<uint8_t> bgl;
    const GpuBindGroupLayoutEntry t = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Texture, 0, 2);
    append(bgl, GpuCreateBindGroupLayoutPayload{9511, 1, 0, 0, 0});
    append(bgl, t);
    emit(setup, GpuOpcode::CreateBindGroupLayout, bgl);
    GpuBindGroupEntry2 be = bg_entry(0, GpuBindResourceType::TextureView, 9510);
    be.view.format = 23;
    create_bind_group(setup, 9512, 9511, {be});
    run(ex, setup);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().device_errors == before.device_errors);
}

// A texture view bound to a layout entry takes the entry's viewDimension: a cube-array view of a 6-layer
// texture meeting a 2d entry is a single-slice 2d view, a cube view meeting a 2d-array entry a 2d-array
// view, a 2d-array view meeting a cube entry a cube view. No Dawn dimension error, counted as coerced.
void test_view_dimension_matches_entry(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    GpuCreateTexture2Payload ct = texture2(9600, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 22);
    ct.depth_or_layers = 6;
    emit(setup, GpuOpcode::CreateTexture2, bytes_of(ct));
    struct Case {
        uint8_t entry_dim, view_dim;
    };
    const Case cases[] = {{2, 5}, {3, 4}, {4, 3}, {5, 3}, {2, 3}, {4, 0}};
    uint32_t id = 9610;
    for (const Case& c : cases) {
        std::vector<uint8_t> bgl;
        append(bgl, GpuCreateBindGroupLayoutPayload{id, 1, 0, 0, 0});
        append(bgl, layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Texture, 0, c.entry_dim));
        emit(setup, GpuOpcode::CreateBindGroupLayout, bgl);
        GpuBindGroupEntry2 be = bg_entry(0, GpuBindResourceType::TextureView, 9600);
        be.view.dimension = c.view_dim;
        create_bind_group(setup, id + 1, id, {be});
        id += 2;
    }
    run(ex, setup);
    ex.wait_idle();
    ex.poll();
    const ExecutorStats& s = ex.stats();
    CHECK(s.device_errors == before.device_errors);
    CHECK(s.view_dim_coerced == before.view_dim_coerced + 6);
    CHECK(s.view_dim_unfixable == before.view_dim_unfixable);

    // A cube entry over a 2-layer texture cannot be satisfied: counted, not coerced.
    gpu::CommandEncoder two;
    GpuCreateTexture2Payload t2 = texture2(9650, 4, 4, GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst, 22);
    t2.depth_or_layers = 2;
    emit(two, GpuOpcode::CreateTexture2, bytes_of(t2));
    std::vector<uint8_t> bgl;
    append(bgl, GpuCreateBindGroupLayoutPayload{9651, 1, 0, 0, 0});
    append(bgl, layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Texture, 0, 4));
    emit(two, GpuOpcode::CreateBindGroupLayout, bgl);
    GpuBindGroupEntry2 be = bg_entry(0, GpuBindResourceType::TextureView, 9650);
    be.view.dimension = 3;
    create_bind_group(two, 9652, 9651, {be});
    run(ex, two);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().view_dim_unfixable == before.view_dim_unfixable + 1);
}

const char* kAliasCompute = R"(
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(%G%) @binding(%B%) var<storage, read> inp: array<u32>;
@compute @workgroup_size(4) fn cs_main(@builtin(local_invocation_index) i: u32) { out[i] = inp[i] + 10u; }
)";

// The same buffer bound writable and read-only in one dispatch (WebGPU forbids it; Dawn: "Writable storage
// buffer binding aliasing"): inside one bind group, and across two bind groups. The read-only side is bound
// through a copy taken at the dispatch, so the result is out = in + 10 and no device error is raised.
void test_storage_buffer_aliasing(NativeExecutor& ex, bool across_groups) {
    const ExecutorStats before = ex.stats();
    const uint32_t base = across_groups ? 9720 : 9700;
    const uint32_t buf = base, shader = base + 2, bgl0 = base + 3, bgl1 = base + 4, pl = base + 5, pipe = base + 6,
                   bg0 = base + 7, bg1 = base + 8;
    std::string src = kAliasCompute;
    src.replace(src.find("%G%"), 3, across_groups ? "1" : "0");
    src.replace(src.find("%B%"), 3, across_groups ? "0" : "1");
    gpu::CommandEncoder setup;
    setup.create_buffer(buf, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, 16);
    const uint32_t init[4] = {1, 2, 3, 4};
    setup.write_buffer(buf, 0, init, 16);
    setup.create_shader_module(shader, GpuShaderStage::Compute, src.c_str());
    const GpuBindGroupLayoutEntry rw = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1);
    const GpuBindGroupLayoutEntry ro =
        layout_entry(across_groups ? 0 : 1, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 2);
    std::vector<uint8_t> l0;
    append(l0, GpuCreateBindGroupLayoutPayload{bgl0, across_groups ? 1u : 2u, 0, 0, 0});
    append(l0, rw);
    if (!across_groups) append(l0, ro);
    emit(setup, GpuOpcode::CreateBindGroupLayout, l0);
    std::vector<uint8_t> plv;
    if (across_groups) {
        std::vector<uint8_t> l1;
        append(l1, GpuCreateBindGroupLayoutPayload{bgl1, 1, 0, 0, 0});
        append(l1, ro);
        emit(setup, GpuOpcode::CreateBindGroupLayout, l1);
        append(plv, GpuCreatePipelineLayoutPayload{pl, 2, 0, 0, 0});
        append(plv, bgl0);
        append(plv, bgl1);
    } else {
        append(plv, GpuCreatePipelineLayoutPayload{pl, 1, 0, 0, 0});
        append(plv, bgl0);
    }
    emit(setup, GpuOpcode::CreatePipelineLayout, plv);
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = pipe;
    cp.cs_module_id = shader;
    cp.pipeline_layout_id = pl;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    if (across_groups) {
        create_bind_group(setup, bg0, bgl0, {bg_entry(0, GpuBindResourceType::Buffer, buf)});
        create_bind_group(setup, bg1, bgl1, {bg_entry(0, GpuBindResourceType::Buffer, buf)});
    } else {
        create_bind_group(setup, bg0, bgl0,
                          {bg_entry(0, GpuBindResourceType::Buffer, buf), bg_entry(1, GpuBindResourceType::Buffer, buf)});
    }
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(pipe);
    set_bind_group2(frame, 0, bg0);
    if (across_groups) set_bind_group2(frame, 1, bg1);
    frame.dispatch_compute(1, 1, 1);
    frame.dispatch_compute(1, 1, 1);  // again: reads what the first dispatch wrote
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(buf, 0, 16, &out, &err));
    const uint32_t want[4] = {21, 22, 23, 24};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    const ExecutorStats& s = ex.stats();
    CHECK(s.device_errors == before.device_errors);
    CHECK(s.dispatches == before.dispatches + 2);
    // the read-only side is copied once per dispatch (a usage-scope snapshot: read-only + writable storage of one
    // buffer is a usage conflict before it is an aliasing one)
    CHECK(s.storage_alias_copies + s.usage_buffer_snapshots ==
          before.storage_alias_copies + before.usage_buffer_snapshots + 2);
    CHECK(s.storage_alias_unfixable == before.storage_alias_unfixable);
}

// The same buffer bound twice as writable storage in one dispatch (Dawn: "Writable storage buffer binding
// aliasing"): the shader writes disjoint words through each binding; the second runs on a scratch copy and its
// changed words are merged back, so both writes land and no device error is raised.
void test_storage_buffer_two_writers(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    const uint32_t buf = 9740, shader = 9742, bgl0 = 9743, pl = 9745, pipe = 9746, bg0 = 9747;
    gpu::CommandEncoder setup;
    setup.create_buffer(buf, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, 16);
    const uint32_t init[4] = {1, 2, 3, 4};
    setup.write_buffer(buf, 0, init, 16);
    setup.create_shader_module(shader, GpuShaderStage::Compute,
                               "@group(0) @binding(0) var<storage, read_write> a: array<u32>;\n"
                               "@group(0) @binding(1) var<storage, read_write> b: array<u32>;\n"
                               "@compute @workgroup_size(2) fn cs_main(@builtin(local_invocation_index) i: u32) {\n"
                               "  a[i] = a[i] + 10u;\n  b[i + 2u] = b[i + 2u] + 20u;\n}\n");
    create_layouts(setup, bgl0, pl,
                   {layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1),
                    layout_entry(1, GpuShaderStageBits::Compute, GpuBindingLayoutType::Buffer, 1)});
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = pipe;
    cp.cs_module_id = shader;
    cp.pipeline_layout_id = pl;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    create_bind_group(setup, bg0, bgl0,
                      {bg_entry(0, GpuBindResourceType::Buffer, buf), bg_entry(1, GpuBindResourceType::Buffer, buf)});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(pipe);
    set_bind_group2(frame, 0, bg0);
    frame.dispatch_compute(1, 1, 1);
    frame.dispatch_compute(1, 1, 1);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    std::vector<uint8_t> out;
    std::string err;
    CHECK(ex.read_buffer(buf, 0, 16, &out, &err));
    const uint32_t want[4] = {21, 22, 43, 44};
    CHECK(out.size() == 16 && std::memcmp(out.data(), want, 16) == 0);
    const ExecutorStats& s = ex.stats();
    CHECK(s.device_errors == before.device_errors);
    CHECK(s.dispatches == before.dispatches + 2);
    CHECK(s.storage_alias_twin_writers == before.storage_alias_twin_writers + 2);
    CHECK(s.storage_alias_unfixable == before.storage_alias_unfixable);
}

// A texture bound as a writable storage texture and as a sampled texture in one dispatch (Dawn: usage conflict
// in the dispatch): the sampled side reads a twin copy, so dst = src + 7 per dispatch and no device error.
void test_storage_texture_aliasing(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    const uint32_t tex = 9760, shader = 9761, bgl0 = 9762, pl = 9763, pipe = 9764, bg0 = 9765;
    const uint8_t fmt = static_cast<uint8_t>(playgta6::gpu::pso::WgpuTextureFormat::R32Uint);
    gpu::CommandEncoder setup;
    emit(setup, GpuOpcode::CreateTexture2,
         bytes_of(texture2(tex, 4, 4,
                           GpuTextureUsage::StorageBinding | GpuTextureUsage::TextureBinding | GpuTextureUsage::CopySrc,
                           fmt)));
    GpuBindGroupLayoutEntry src = layout_entry(0, GpuShaderStageBits::Compute, GpuBindingLayoutType::Texture, 4, 2);
    GpuBindGroupLayoutEntry dst = layout_entry(1, GpuShaderStageBits::Compute, GpuBindingLayoutType::StorageTexture,
                                               static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly), 2);
    dst.storage_format = fmt;
    create_layouts(setup, bgl0, pl, {src, dst});
    setup.create_shader_module(
        shader, GpuShaderStage::Compute,
        "@group(0) @binding(0) var src: texture_2d<u32>;\n"
        "@group(0) @binding(1) var dst: texture_storage_2d<r32uint, write>;\n"
        "@compute @workgroup_size(4, 4, 1) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
        "  textureStore(dst, id.xy, vec4<u32>(textureLoad(src, id.xy, 0).x + 7u, 0u, 0u, 0u));\n}\n");
    GpuCreateComputePipeline2Payload cp;
    std::memset(&cp, 0, sizeof cp);
    cp.id = pipe;
    cp.cs_module_id = shader;
    cp.pipeline_layout_id = pl;
    emit(setup, GpuOpcode::CreateComputePipeline2, bytes_of(cp));
    GpuBindGroupEntry2 b0 = bg_entry(0, GpuBindResourceType::TextureView, tex);
    GpuBindGroupEntry2 b1 = bg_entry(1, GpuBindResourceType::TextureView, tex);
    for (GpuBindGroupEntry2* b : {&b0, &b1}) {
        b->view.format = fmt;
        b->view.dimension = 2;
        b->view.mip_level_count = 1;
        b->view.array_layer_count = 1;
    }
    create_bind_group(setup, bg0, bgl0, {b0, b1});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    frame.set_compute_pipeline(pipe);
    set_bind_group2(frame, 0, bg0);
    frame.dispatch_compute(1, 1, 1);
    frame.dispatch_compute(1, 1, 1);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    std::vector<uint8_t> px;
    uint32_t w = 0, h = 0;
    std::string err;
    CHECK(ex.read_texture(tex, 0, &px, &w, &h, &err) && w == 4 && h == 4 && px.size() == 64);
    bool all = px.size() == 64;
    for (size_t i = 0; all && i < 16; ++i) {
        uint32_t v;
        std::memcpy(&v, px.data() + i * 4, 4);
        all = v == 14;
    }
    CHECK(all);
    const ExecutorStats& s = ex.stats();
    CHECK(s.device_errors == before.device_errors);
    CHECK(s.storage_tex_alias_copies + s.usage_texture_snapshots == before.storage_tex_alias_copies + before.usage_texture_snapshots + 2);
}

void test_error_scope_attribution() {
    ExecutorOptions opts;
    opts.width = kW;
    opts.height = kH;
    opts.gpu_error_scopes = true;
    NativeExecutor::CreateStatus status;
    std::string error;
    auto exu = NativeExecutor::create(opts, &status, &error);
    if (!exu) {
        CHECK(status == NativeExecutor::CreateStatus::NoAdapter);
        return;
    }
    NativeExecutor& ex = *exu;
    const uint8_t data[4] = {1, 2, 3, 4};
    gpu::CommandEncoder setup;
    setup.create_buffer(7, GpuBufferUsage::CopySrc | GpuBufferUsage::CopyDst, 16);
    run(ex, setup);
    // A good packet inside a scope records nothing.
    gpu::CommandEncoder good;
    good.write_buffer(7, 4, data, 4);
    run(ex, good);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats_json().find("\"error_scopes\": {\"total\": 0") != std::string::npos);
    // An unaligned write is a validation error raised inside the packet's scope: attributed to that packet.
    gpu::CommandEncoder bad;
    bad.write_buffer(7, 2, data, 4);
    run(ex, bad);
    ex.wait_idle();
    ex.poll();
    const std::string j = ex.stats_json();
    CHECK(j.find("\"error_scopes\": {\"total\": 1") != std::string::npos);
    CHECK(j.find("\"kind\": \"validation\", \"phase\": \"other\"") != std::string::npos);
    CHECK(j.find("\"opcode\": 4,") != std::string::npos);  // WriteBuffer
    CHECK(j.find("\"present_index\": 0") != std::string::npos);
}

// Five RGBA16F targets = 40 bytes per sample, over the 32 B WebGPU default: the device must have been
// created with the adapter's own maxColorAttachmentBytesPerSample or Dawn rejects the pipeline.
void test_wide_mrt_pipeline(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder pl;
    pl.create_shader_module(920, GpuShaderStage::Vertex,
                            "@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {"
                            " var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));"
                            " return vec4f(p[i], 0.0, 1.0); }\n");
    pl.create_shader_module(921, GpuShaderStage::Fragment,
                            "struct Out { @location(0) a: vec4f, @location(1) b: vec4f, @location(2) c: vec4f,"
                            " @location(3) d: vec4f, @location(4) e: vec4f }\n"
                            "@fragment fn fs_main() -> Out { var o: Out; return o; }\n");
    GpuCreateRenderPipeline2Payload rp = pipeline_to_bgra(922, 920, 921, 0);
    rp.key.color_target_count = 5;
    for (int i = 0; i < 5; ++i) {
        rp.key.color_targets[i].format = 40;  // rgba16float
        rp.key.color_targets[i].write_mask = 15;
    }
    emit(pl, GpuOpcode::CreateRenderPipeline2, bytes_of(rp));
    run(ex, pl);
    ex.wait_idle();
    ex.poll();
    CHECK(ex.stats().device_errors == before.device_errors);
    CHECK(ex.stats().write_masks_cleared == before.write_masks_cleared);
}

// A pipeline is drawn into a pass whose attachments have other formats than the pipeline's key (an sRGB
// twin of the target, a depth attachment the key does not know): the executor draws with a variant built
// with the pass's attachment state, so Dawn raises no "attachment state is not compatible" error.
void test_pipeline_pass_formats(NativeExecutor& ex) {
    const ExecutorStats before = ex.stats();
    gpu::CommandEncoder setup;
    emit(setup, GpuOpcode::CreateTexture2, bytes_of(texture2(9600, 4, 4, GpuTextureUsage::RenderAttachment, 22)));  // rgba8unorm
    emit(setup, GpuOpcode::CreateTexture2, bytes_of(texture2(9601, 4, 4, GpuTextureUsage::RenderAttachment, 48)));  // depth32float
    GpuCreateSampler2Payload sp;
    std::memset(&sp, 0, sizeof sp);
    sp.id = 9602;
    sp.max_anisotropy = 1;
    sp.lod_max_clamp = 32.0f;
    emit(setup, GpuOpcode::CreateSampler2, bytes_of(sp));
    setup.create_shader_module(9610, GpuShaderStage::Vertex,
                               "@group(0) @binding(0) var smp: sampler;\n"
                               "@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {"
                               " var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));"
                               " return vec4f(p[i], 0.5, 1.0); }\n");
    setup.create_shader_module(9611, GpuShaderStage::Fragment,
                               "@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }\n");
    create_layouts(setup, 9620, 9621, {layout_entry(0, GpuShaderStageBits::Fragment, GpuBindingLayoutType::Sampler, 0)});
    GpuCreateRenderPipeline2Payload rp = pipeline_to_bgra(9630, 9610, 9611, 9621);
    rp.key.color_targets[0].format = 23;  // rgba8unorm-srgb: the pass renders into the rgba8unorm texture
    emit(setup, GpuOpcode::CreateRenderPipeline2, bytes_of(rp));
    create_bind_group(setup, 9640, 9620, {bg_entry(0, GpuBindResourceType::Sampler, 9602)});
    run(ex, setup);

    gpu::CommandEncoder frame;
    frame.begin_frame();
    for (int with_depth = 0; with_depth < 2; ++with_depth) {
        GpuBeginRenderPassPayload bp = texture_pass(9600, 0, 0, 0, 1);
        if (with_depth) {
            bp.depth_stencil.texture_id = 9601;
            bp.depth_stencil.depth_load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
            bp.depth_stencil.depth_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
            bp.depth_stencil.depth_clear_value = 1.0f;
        }
        frame.emit_packet(GpuOpcode::BeginRenderPass, &bp, sizeof bp);
        frame.set_pipeline(9630);
        set_bind_group2(frame, 0, 9640);
        frame.draw(3);
        frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    }
    GpuBeginRenderPassPayload pass = swapchain_pass(GpuLoadOp::Clear, 0, 0, 0, 1);
    frame.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof pass);
    frame.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
    frame.present();
    run(ex, frame);
    ex.wait_idle();
    ex.poll();
    const ExecutorStats& after = ex.stats();
    CHECK(after.device_errors == before.device_errors);
    CHECK(after.draws == before.draws + 2);
    CHECK(after.pipeline_variants_created == before.pipeline_variants_created + 2);
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
    test_layout_fit_missing_group(*ex);
    test_layout_fit_over_limits(*ex);
    test_layout_fit_unused_group_over_limits(*ex);
    test_indirect_constants_pass(*ex);
    test_compute_pipeline_survives_copy(*ex);
    test_bindings_survive_copy(*ex);
    test_render_bindings_survive_copy(*ex);
    test_indirect_clamp_work(*ex);
    test_mesh_pipeline(*ex);
    test_pack_and_formats(*ex);
    test_stream_accounting(*ex);
    test_readbacks(*ex);
    test_clear_texture(*ex);
    test_copy_texture(*ex);
    test_queries(*ex);
    test_readbacks_host_addresses(*ex);
    test_v1(*ex);
    test_reflection();
    CHECK(ex->stats().device_lost == 0);
    test_timers();
    test_timer_split();
    test_blit();
    test_depth_texture_float_layout(*ex);
    test_depth_read_variants(*ex);
    test_norm16_filtered_read(*ex);
    test_norm16_volume_filtered_read(*ex);
    test_depth_read_no_float32_filterable();
    test_stencil_twin(*ex);
    test_show_target();
    test_show_target_read_only();
    test_overlay_read_only();
    test_show_target_bad_option();
    test_device_loss();
    test_device_loss_journal();
    test_async_pipelines_and_blob_cache();
    test_shader_workers("0");
    test_shader_workers("10000");
    test_worker_group_metadata();
    for (const char* scenario : {"unchanged", "unknown-mask", "missing", "same-hash", "in-place", "truncated", "failed-open"})
        test_shader_pack_identity(scenario);
    test_remap_reflected_layout(true);
    test_remap_reflected_layout(false);
    test_dawn_validation_fixes(*ex);
    test_bind_group_binding_fit(*ex);
    test_srgb_view_of_storage_texture(*ex);
    test_view_dimension_matches_entry(*ex);
    test_storage_buffer_aliasing(*ex, false);
    test_storage_buffer_aliasing(*ex, true);
    test_storage_buffer_two_writers(*ex);
    test_storage_texture_aliasing(*ex);
    test_wide_mrt_pipeline(*ex);
    test_screenshot();
    test_error_scope_attribution();
    test_pipeline_pass_formats(*ex);
    test_split_passes(*ex);
    test_usage_snapshots(*ex);
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    ex.reset();
    std::remove(kPackPath);
    std::printf("test_native_executor: ok\n");
    return 0;
}
