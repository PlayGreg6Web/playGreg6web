// Native executor: swapchain blit model (web/workers/wgpu_swapchain.js `blitToCanvasIfNeeded`,
// `configureForStreamFormat`). Pure logic, no GPU: which stream formats need the intermediate
// texture + fullscreen-triangle blit, the blit shader (the worker's, verbatim), and the letterbox
// viewport used when the stream size differs from the presenting canvas / window size.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gpu/native_exec/window_layout.h"  // PresentRect / letterbox_rect (the window's present rect)

namespace playgta6::gpu::native_exec {

// The worker's BLIT_FULLSCREEN_WGSL. The source is sampled with a linear sampler; an sRGB stream
// format is decoded by the sampler and written to the unorm canvas as is, exactly like the worker
// (no re-encode), so the blit never changes the colour pipeline.
inline constexpr const char* kBlitWgsl = R"(struct BlitVsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> BlitVsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );
  var out: BlitVsOut;
  out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
  out.uv = uv[vid];
  return out;
}

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

@fragment
fn fs_main(in: BlitVsOut) -> @location(0) vec4<f32> {
  return textureSample(blit_tex, blit_sampler, in.uv);
}
)";

// Format codes (docs/gpu_commands.md "WebGPU enum codes"): the canvas formats (isValidCanvasFormat).
inline bool is_canvas_format_code(uint8_t code) { return code == 22 || code == 27 || code == 40; }

// A stream swapchain format that is not a canvas format but can be rendered into: colour formats
// (codes 1..43: r8unorm .. rgba32float, sRGB and packed ones included). The worker accepts any
// non-canvas name and lets WebGPU validation reject what cannot work (uint formats cannot be
// sampled as f32, for one); the executor does the same. Depth/stencil (44..48) and block
// compressed (49+) codes are not swapchain formats.
inline bool is_blit_stream_format_code(uint8_t code) { return code >= 1 && code <= 43 && !is_canvas_format_code(code); }

// True when presenting needs a blit at all: a non-canvas stream format, or a stream size that
// differs from the canvas size.
inline bool blit_needed(bool stream_format_is_canvas_format, uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                        uint32_t dst_h) {
    return !stream_format_is_canvas_format || src_w != dst_w || src_h != dst_h;
}

}  // namespace playgta6::gpu::native_exec
