// Native executor milestone 0 on the host's real GPU (headless): device, offscreen swapchain,
// clear/present and the first packet types, with every result read back from the GPU. Exits 77
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
    enc.draw(3);  // not implemented in milestone 0: skipped and counted, never approximated
    enc.nop();
    std::vector<uint8_t> bytes(enc.data(), enc.data() + enc.size());
    const GpuPacketHeader truncated{static_cast<uint16_t>(GpuOpcode::WriteBuffer), 0, 64};
    const size_t at = bytes.size();
    bytes.resize(at + 8 + 4);
    std::memcpy(bytes.data() + at, &truncated, 8);
    CHECK(ex.execute(bytes.data(), bytes.size()) == 0);
    const ExecutorStats& after = ex.stats();
    CHECK(after.unhandled[static_cast<size_t>(GpuOpcode::Draw)] ==
          before.unhandled[static_cast<size_t>(GpuOpcode::Draw)] + 1);
    CHECK(after.handled[0] == before.handled[0] + 1);
    CHECK(after.truncated_batches == before.truncated_batches + 1);
    CHECK(after.packets == before.packets + 2);
    const std::string json = ex.stats_json();
    CHECK(json.find("\"unhandled\": {\"6\": 1}") != std::string::npos);
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
    test_stream_accounting(*ex);
    CHECK(ex->stats().device_lost == 0);
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_native_executor: ok\n");
    return 0;
}
