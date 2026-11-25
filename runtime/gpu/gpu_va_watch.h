#pragma once

// GPU address watch (G62 diagnostic, harness option --watch-gpu-va): reports every place the D3D12
// guest layer sees a watched GPU address range referenced, so one can tell whether (and how) the
// title ever uses some memory on the GPU, e.g. the frame buffers a video decoder writes.
//
// Watched ranges come from a spec, a comma-separated list of
//   0xSTART-0xEND      [START, END)
//   0xSTART+SIZE       [START, START + SIZE) (SIZE decimal or 0x hex)
//   textures:WxH[/WxH...]   the memory of every texture resource created with one of these sizes
// Reported kinds: texture creation (for size selectors), descriptor creation (SRV / UAV / RTV /
// CBV over a watched resource), descriptor-table slot reads (root tables read from descriptor
// memory), root descriptors (CBV / SRV / UAV addresses), copy sources and destinations, and
// uploads (texture memory sync, WriteToSubresource, CPU-visible buffer flushes).
//
// Each distinct (kind, watched address, detail) is logged once, as
//   gpu-va-watch: frame <present count> thread <guest thread id> <kind> <va> +<size> [<range>] <detail>
// and counted; summary() lists everything seen with counts. Inactive (no cost beyond one atomic
// load per hook) unless configure() was given a non-empty spec.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace playgta6::gpu::va_watch {

// Parses `spec` and activates the watch. False + message on a malformed spec (nothing changes).
bool configure(std::string_view spec, std::string* error);
// Clears ranges, selectors and events (tests).
void reset();
// True when anything is watched (ranges or texture-size selectors).
bool active();

// A texture resource was created over [va, va + size): watched from now on when its size matches a
// `textures:` selector (reported as kind "texture"), reported when it lies in a watched range.
void note_texture(uint64_t va, uint64_t size, uint32_t width, uint32_t height, uint32_t dxgi_format,
                  uint32_t resource_id);

// Reports a reference to [va, va + size) (size 0 counts as 1 byte) when it overlaps a watched range;
// true when it did.
bool touch(const char* kind, uint64_t va, uint64_t size, const std::string& detail);

// Descriptor handles that hold a watched address: a descriptor created over a watched range is
// remembered by its CPU handle, and a CopyDescriptors* from such a handle is reported (kind
// "descriptor copy") and makes the destination handle watched too.
void watch_handle(uint64_t handle, uint64_t va);
void descriptor_copied(uint64_t src_handle, uint64_t dst_handle);

// The current frame number (the device's present count) stamped on events.
void set_frame(uint64_t frame);

struct Event {
    std::string kind;
    uint64_t va = 0;
    uint64_t size = 0;
    std::string range;   // the watched range it hit
    std::string detail;
    uint64_t first_frame = 0;
    uint32_t first_thread = 0;
    uint64_t count = 0;
};
std::vector<Event> events();
// One line per distinct event with its count, or a line saying nothing was referenced.
std::string summary();

}  // namespace playgta6::gpu::va_watch
