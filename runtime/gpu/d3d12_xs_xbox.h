#pragma once

// Xbox-specific D3D12 device, queue and command-list slots on the startup and frame path (G27),
// on top of the G24 guest objects (d3d12_guest_objects.h / d3d12_guest_resources.h).
//
// Everything below was established by static analysis of game_scarlett_master.exe (local only;
// only slot indices, argument shapes, struct offsets, counts and public method names are
// recorded). "Confidence" is about the method's identity; the implemented contract only relies on
// the argument shapes and on how the caller uses the result, which were read at every call site.
//
// Inlined command-list accesses (the most important finding)
// -----------------------------------------------------------
// The title does not only talk to command lists through the vtable. Every hot method (draws,
// topology, stencil ref, root constants/tables, ...) has an inlined fast path guarded by
// `mov rcx,[list+18h]; cmp rcx,[list+28h]` (draws) or `cmp rcx,[list+30h]` (state/root): when
// there is room it writes PM4-like packets straight into the list's command buffer at
// [list+0x18] and never calls the vtable. The slow path is the vtable call. The fast-path-only
// fields are bytes list+0x140..0x143, dword tables at list+0x144 / list+0x344 (indexed by root
// slot) and byte list+0x548. With the cursor and limits at list+0x18/0x28/0x30 equal to zero the
// fast path is never taken, so a zero-filled guest payload of CommandList::kComPayloadSize bytes
// turns every call into a vtable call. The title's command-allocator slot 9 registers a callback
// that would hand out command-buffer memory; it is recorded and never called, so the cursor stays
// zero.
// One write is unconditional: before calling SetPipelineState (slot 25) with only `this`, the
// title copies 32 bytes from the PSO object (vmovaps [pso+0x10], [pso+0x20]) into the list at
// list+0x544 / list+0x554. Pipeline states therefore carry a 32-byte payload (kPipelineStateTag,
// own interface pointer) and SetPipelineState reads the staged copy (falling back to rdx for
// callers that pass the PSO). Only 278 PM4-style immediates exist in 220 functions in total;
// besides the command-list fast paths no other code builds command buffers. Resources, descriptor
// heaps and PSOs showed no other inlined field reads; the device byte at device+0x18 (= 1, G24)
// keeps 11 inlined descriptor writers on their API path.
//
// Device slots (Xbox order; public slots 0..43 are in d3d12_guest_objects.h)
// --------------------------------------------------------------------------
//  slot  sites  shape / result use                                   contract (confidence)
//   49    1    (0x11F, 1) behind a system-version check; ignored    S_OK no-op (low: feature toggle)
//   50    1    (callback, 0, callback) on the device-removed path  S_OK no-op (low)
//   55    1    (base PSO, n, overrides[n] of 0x14C bytes, riid, out) CreatePipelineStateWithOverrides
//              override = {u32 kind; desc}: kind 0 DEPTH_STENCIL_DESC, 1 RASTERIZER_DESC, 2 BLEND_DESC
//              (byte layouts match the public structs), kind 5 unknown (ignored, counted). The
//              result is read unconditionally: a pipeline state is always handed out. (high shape)
//   57    1    (desc, riid, out), desc = D3D12_COMPUTE_PIPELINE_STATE_DESC-shaped (root signature
//              first, from a compile queue)                       CreateComputePipelineState (medium)
//   61    1    (a, b); ignored                                     S_OK no-op (low)
//   62    1    ({blob, .., flags=1}, 0, 0, riid, out); failure frees the blob and returns false
//   67    1    (&{blob}, riid, out) from a PSO cache; failure falls back to slot 62
//              Both create a PSO from a title-compiled blob (format unknown): E_NOTIMPL, *out = 0.
//   70    2+   (RESOURCE_DESC (buffer), SRV desc + u64 GPU address at +0x28, CPU handle)
//   71    2    (RESOURCE_DESC, UAV desc + u64 GPU address at +0x28, CPU handle)
//              Buffer views at a GPU address without a resource object (high shape): resolved
//              through the state bridge's VA map.
//   74    1    (address from XMemVirtualAlloc+XMemMapPhysicalPages, pages, &handle); traps on
//              failure                                            page-pool registration: handle (low)
//   78    1    (output=0, interval us, period, flags=0); checked  SetFrameIntervalX-like (medium)
//   79    3    (type, offset us, 0, flags=0); traps on failure    ScheduleFrameEventX-like (medium)
//   80    3    (type, timeout ms, 0, flags, &token): (2, INFINITE, 0, 0, &in/out token),
//              (0, 0, 0, 2, &frame token) [traps on failure], (0, INFINITE, 0, 0, &tmp) [traps]
//                                                                 WaitFrameEventX-like (medium)
//   81    1    (previous token, 0x11100, &count (in: 6), entries of 0x38 bytes); entries read only
//              when S_OK and count != 0                           frame statistics: count = 0 (low)
//   86    1    ({SizeInBytes 0x2B0, pStream}, riid, out)          CreatePipelineState(stream) (high):
//              the stream uses the public subobject types and alignment (root signature, VS, PS,
//              DS, HS|AS, GS|MS, BLEND, SAMPLE_MASK, RASTERIZER, DEPTH_STENCIL1, SAMPLE_DESC,
//              INPUT_LAYOUT, PRIMITIVE_TOPOLOGY, RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT).
//  101    1    (state-object desc of type RAYTRACING_PIPELINE, riid ID3D12StateObject, out); the
//              caller handles failure                             CreateStateObject: E_NOTIMPL (high)
//  108    2    struct return, hidden result in rcx, this in rdx, (mask 1, count 1, descs, &info1);
//              the 16-byte result is used                          GetResourceAllocationInfo1 (high)
//  119    1    (buffer, 0x80, 0) in a device-removed callback      S_OK no-op (low)
//  132    1    (3) on the device-removed path                      S_OK no-op (low)
//  127, 133: G24 (resources). No device slot allocates slot 133's address: the title passes its
//  own memory. One helper takes the address of a memory block (or a resource's
//  GetGPUVirtualAddress) and, when it lies in a CPU window [C, C+S), translates it to the GPU
//  window [G, G+S). Both windows are XMemVirtualAlloc reservations of equal size made right after
//  D3D12ConfigureVirtualMemoryX; pages of a physical page pool (registered through slot 74) are
//  mapped into the CPU window through xmem and into the GPU window through queue slot 31. Other
//  addresses are passed unchanged (UMA: GPU address = CPU address). Device::register_gpu_alias
//  makes buffers at GPU-window addresses read their bytes from the CPU window; the device
//  subscribes to the window pairs the XMem layer recognises (os/xmem_windows.h, O27).
//
// Queue slots
// -----------
//  23 (0) / 24 () in a suspend/resume handler: SuspendX / ResumeX-like, S_OK (medium).
//  31 (range count, ranges, page-pool address, pool pages, 0) on a second queue: page-mapping
//     update of the GPU window (medium shape). Range = 0x18 bytes {u64 GPU-window address, u32
//     entry count, u64 entries}, entry = 12 bytes {u32 kind (1 map, 2 unmap), u32 pool page, u32
//     page count}; the same window pages are mapped into the CPU window through xmem. Applied
//     to the XMem GPU page table (os/xmem_windows.h, O27).
//  32 PresentX(plane count, planes, params) (high shape). Plane = GuestPresentPlaneX (0x48 bytes):
//     token (the slot-80 frame token), resource count, &resource pointer, source rects, ... Plane
//     0's first resource is the back buffer; also called with (0, 0, 0) during suspend.
//  35 ExecuteCommandLists(count, lists, frame token, flags = 3) (high shape). The token is the
//     slot-80 output of the current frame, 0 outside a frame.
//
// Frame model
// -----------
// The title's submission thread waits for the frame-origin event (slot 80, type 0, flags 2,
// timeout 0) when it starts a frame, executes the frame's lists with that token (queue 35),
// waits for the previous frame's token (slot 80 type 2) before presenting, presents the planes
// (queue 32) and clears its token. Here a frame event is a tick of a steady clock at the slot-78
// interval (default 16667 us, clamped to [1 ms, 100 ms]); tokens are tick numbers. A wait with a
// timeout blocks (condition variable, no spinning) until the next tick, which bounds the frame
// rate; a zero timeout returns the current tick at once (the title traps if it fails). Shutdown
// (Device::shutdown_frame_events) releases every waiter and makes later waits return at once.
// Present emits Present (and BeginFrame before the next submission, G24); plane 0's resource is
// marked as the back buffer the first time it is presented, so render passes into it target the
// canvas from then on (one lost frame per back buffer). Further planes are counted, not composed.
//
// Command-list slots (Xbox order, from the two context-owned lists and the pools)
// -------------------------------------------------------------------------------
//  0..27 public order (12/13/14 draws/dispatch, 20..24 IA/RS/OM state, 25 SetPipelineState
//  (staged), 26 ResourceBarrier, 27 ExecuteBundle). 28 BeginQuery (heap, type, index), 29 EndQuery,
//  30 ResolveQueryData (7 args), 31 SetPredication (inferred). 32..55 = public 28..51 (+4): 32
//  SetDescriptorHeaps (2 heaps), 33/34 root signatures, 35/36 tables, 37/38 constant, 39/40
//  constants, 41/42 CBV, 43/44 SRV, 45/46 UAV, 47 IASetIndexBuffer, 48 IASetVertexBuffers,
//  49 SOSetTargets, 50 OMSetRenderTargets, 51 ClearDepthStencilView, 52 ClearRenderTargetView,
//  53/54 ClearUnorderedAccessView, 55 DiscardResource. 56..58 debug markers (inferred), 61
//  ExecuteIndirect (signature, max 1, args, offset, 0, 0). 70 OMSetDepthBounds (two floats),
//  126 RSSetShadingRate (rate, combiners), 127 RSSetShadingRateImage (resource or 0).
//  Xbox-only, meaning not established, recorded without effect: 59, 60, 62 (4 args), 72 (5 args,
//  query related), 75 (1 arg), 86, 128 (&{1, value}), 129 (value, 1, 1), 131 (register-like
//  (offset, value, 0, mask): 0xC280/0x20000 per frame, 0xC3C0/0xC380 at creation), 132
//  ((value, 0, 0, mask)), 142 (value or -1).
//
// Raytracing
// ----------
// CheckFeatureSupport(OPTIONS5) reports RaytracingTier NOT_SUPPORTED and CreateStateObject (101)
// fails with E_NOTIMPL; the one caller takes its failure path. Effects that need ray tracing
// (ray-traced reflections, shadows, global illumination) stay off; acceleration-structure builds
// and DispatchRays are never reached.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include "gpu/d3d12_guest_structs.h"

namespace playgta6::gpu::d3d12_guest::xbox {

// ---------------------------------------------------------------------------------------------
// Guest layouts
// ---------------------------------------------------------------------------------------------

// Pipeline-state payload tag ("PG6PSO01" little endian) and the command-list staging offsets.
inline constexpr uint64_t kPipelineStateTag = 0x31304F5350364750ull;
inline constexpr uint64_t kListStagedPipelineOffset = 0x544;  // from list base (16 + 16 bytes)
inline constexpr uint64_t kListFastPathCursorOffset = 0x18;   // must stay 0 (see above)

struct GuestPresentPlaneX {  // queue slot 32 plane (0x48 bytes)
    uint64_t Token;
    uint32_t ResourceCount;
    uint32_t pad0;
    uint64_t ppResources;   // guest ID3D12Resource* const*
    uint64_t pSrcRects;     // RECT per resource
    uint64_t pDestPlacement;
    uint32_t ColorSpace;
    uint32_t reserved[7];
};
static_assert(sizeof(GuestPresentPlaneX) == 0x48 && offsetof(GuestPresentPlaneX, ppResources) == 0x10 &&
              offsetof(GuestPresentPlaneX, ColorSpace) == 0x28);

struct GuestPipelineStateStreamDesc {  // D3D12_PIPELINE_STATE_STREAM_DESC
    uint64_t SizeInBytes;
    uint64_t pPipelineStateSubobjectStream;
};
static_assert(sizeof(GuestPipelineStateStreamDesc) == 16);

struct GuestResourceAllocationInfo1 {  // D3D12_RESOURCE_ALLOCATION_INFO1
    uint64_t Offset;
    uint64_t Alignment;
    uint64_t SizeInBytes;
};
static_assert(sizeof(GuestResourceAllocationInfo1) == 24);

// Device slot 55 override entry: u32 kind, then the description (BLEND_DESC is the largest).
inline constexpr uint32_t kPsoOverrideStride = 0x14C;
enum class PsoOverrideKind : uint32_t { DepthStencil = 0, Rasterizer = 1, Blend = 2 };

// Device slots 70/71: the public view description followed by the buffer's GPU address.
inline constexpr uint64_t kViewAtAddressOffset = 0x28;

// ---------------------------------------------------------------------------------------------
// Pipeline-state streams (public D3D12_PIPELINE_STATE_SUBOBJECT_TYPE values and layout)
// ---------------------------------------------------------------------------------------------

enum class StreamPipelineKind { Graphics, Compute, Mesh };

struct StreamPipeline {
    StreamPipelineKind kind = StreamPipelineKind::Graphics;
    GuestGraphicsPipelineStateDesc graphics{};
    GuestComputePipelineStateDesc compute{};
    uint32_t subobjects = 0;
    uint32_t ignored_subobjects = 0;  // known types without an equivalent (view instancing, ...)
};

// Parses a subobject stream: each subobject starts pointer-aligned with a u32 type; its data
// follows at its natural alignment. Defaults follow D3D12 (sample mask ~0, one sample, DEPTH_
// STENCIL defaults with depth enabled). False + message on a malformed or unknown subobject.
bool parse_pipeline_stream(const uint8_t* bytes, size_t size, StreamPipeline* out, std::string* error);

// ---------------------------------------------------------------------------------------------
// Frame pacing
// ---------------------------------------------------------------------------------------------

class FramePacer {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr uint32_t kDefaultIntervalUs = 16667;
    static constexpr uint32_t kMinIntervalUs = 1000;
    static constexpr uint32_t kMaxIntervalUs = 100000;

    FramePacer();
    // Frame-event interval (slot 78). `period` is recorded; its meaning is not established.
    void set_interval(uint32_t interval_us, uint32_t period);
    uint32_t interval_us() const;
    uint32_t period() const;
    // Current tick (>= 1).
    uint64_t current_frame() const;
    enum class WaitResult { Ok, Timeout, Shutdown };
    // Blocks until the tick after `after` has started, at most `timeout_ms` (0xFFFFFFFF: no
    // limit, which is still bounded by one interval). *frame = the tick reached (current one on
    // timeout or shutdown).
    WaitResult wait_after(uint64_t after, uint32_t timeout_ms, uint64_t* frame);
    void shutdown();
    bool is_shut_down() const;

private:
    uint64_t frame_locked(Clock::time_point now) const;
    Clock::time_point start_of_locked(uint64_t frame) const;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    Clock::time_point origin_;
    uint64_t base_frame_ = 1;
    uint32_t interval_us_ = kDefaultIntervalUs;
    uint32_t period_ = 1;
    bool shutdown_ = false;
};

// ---------------------------------------------------------------------------------------------
// Counters (Device::counter)
// ---------------------------------------------------------------------------------------------

enum class Counter : size_t {
    FrameWaits,
    FrameWaitTimeouts,
    FrameIntervalChanges,
    FrameEventsScheduled,
    PresentsWithPlanes,
    PresentsWithoutPlanes,
    BackBufferMarks,
    ExtraPlanesIgnored,
    ExecutesWithToken,
    IndirectSkipped,
    DepthBoundsIgnored,
    ShadingRateIgnored,
    StreamPipelineStates,
    OverridePipelineStates,
    PlaceholderPipelineStates,
    UnknownOverrideKinds,
    BlobPipelineStatesRefused,
    StateObjectsRefused,
    ViewsAtAddress,
    PagePools,
    PageMappingsApplied,
    PlacedDescriptorHeaps,  // descriptor heaps placed at a title address (CreateDescriptorHeap)
    Suspends,
    Resumes,
    FeatureToggles,
    XboxNoOps,
    Count
};

}  // namespace playgta6::gpu::d3d12_guest::xbox
