// Frame capture: the GPU command stream handed to os::wgpu_submit, written as a .pg6trace (the
// flat packet stream web/workers/gpu_recorder.js exports and tools/gpu_trace.py reads) plus a JSON
// sidecar with frame boundaries and per-frame counters.
//
// Frames are delimited by Present packets: frame 0 is everything up to and including the first
// Present (resource creation included), frame i the packets after Present i-1 up to Present i.
// Capture stops after `frames` complete frames.
//
// Payloads: by default the trailing data of WriteBuffer, WriteTexture and CreateShaderModule
// packets is omitted (the packet stays well-formed with data_len / code_len 0, so the file stays
// small and parseable); the sidecar records each omitted payload's original length and FNV-1a 64
// hash. `keep_payloads` writes them unchanged.
//
// Sidecar (<path>.json): {schema "playgta6-frame-capture", schema_version 1, trace, payloads
// ("omitted" | "kept"), frames_requested, frames_complete, presents_seen, packets, bytes,
// frames [{index, offset, bytes, packets}], presents [{index, wall_ms, dispatches, import_calls,
// reserved_bytes, committed_bytes, counters {name: value}}], omitted [{packet, opcode, length,
// fnv1a64}], omitted_bytes}.
// `presents` has one sample per Present seen, also after the captured frames (a timeline of the
// whole run): wall time since the capture started and the counters of frame_capture_set_counters
// at that Present.
//
// Capture files are derived from the running title: keep them out of the repository.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace playgta6::gpu {

struct FrameCaptureConfig {
    std::string path;          // .pg6trace output; the sidecar is path + ".json"
    uint32_t frames = 10;      // complete frames to capture (>= 1)
    bool keep_payloads = false;
    bool attach = true;        // install the os::wgpu_submit observer (tests feed frame_capture_observe)
};

// Starts a capture (replacing none: false if one is running or the file cannot be opened).
bool frame_capture_start(const FrameCaptureConfig& config);
// Feeds a batch of complete packets (the observer does this for wgpu_submit).
void frame_capture_observe(const uint8_t* data, size_t len);
// Flushes the trace and writes the sidecar; idempotent; detaches the observer.
void frame_capture_finish();

// Optional source of the counters sampled at every Present.
struct FrameCaptureSample {
    uint64_t dispatches = 0;
    uint64_t import_calls = 0;
    uint64_t reserved_bytes = 0;   // guest address space reserved
    uint64_t committed_bytes = 0;  // guest memory committed
    // Further named runtime counters (live handles by kind, COM bridge objects, heap blocks, ...),
    // written as a "counters" object in the sample.
    std::vector<std::pair<std::string, uint64_t>> counters;
};
using FrameCaptureCounters = void (*)(FrameCaptureSample* sample);
void frame_capture_set_counters(FrameCaptureCounters counters);

// The per-Present samples of the running (or last) capture as a JSON array, the same text as the
// sidecar's "presents" value (the boot report's "presents" reuses it).
std::string frame_capture_presents_json();

// Frames completed so far by the running (or last) capture.
uint32_t frame_capture_frames();

}  // namespace playgta6::gpu
