// gen_native_exec_trace <out.pg6trace>: a tiny synthetic trace for the native-executor CI smoke test
// (tests/native_exec_warp_smoke.py). Four frames, each a swapchain clear: red, green, blue, red again.
// No game data; built from the CommandEncoder the executor tests use.
#include <cstdio>
#include <cstring>
#include <fstream>

#include "gpu/cmd_encoder.h"

using namespace playgta6;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: gen_native_exec_trace <out.pg6trace>\n");
        return 1;
    }
    static const float colors[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
    std::ofstream out(argv[1], std::ios::binary);
    for (const auto& c : colors) {
        gpu::CommandEncoder f;
        f.begin_frame();
        GpuBeginRenderPassPayload bp;
        std::memset(&bp, 0, sizeof bp);
        bp.color_count = 1;
        bp.color[0].texture_id = kGpuSwapchainTextureId;
        bp.color[0].load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
        bp.color[0].store_op = static_cast<uint8_t>(GpuStoreOp::Store);
        bp.color[0].clear_value[0] = c[0];
        bp.color[0].clear_value[1] = c[1];
        bp.color[0].clear_value[2] = c[2];
        bp.color[0].clear_value[3] = 1.0f;
        f.emit_packet(GpuOpcode::BeginRenderPass, &bp, sizeof bp);
        f.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
        f.present();
        out.write(reinterpret_cast<const char*>(f.data()), static_cast<std::streamsize>(f.size()));
    }
    return out ? 0 : 1;
}
