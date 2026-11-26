#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#include "guest/com_bridge.h"

// Guest interfaces for the Xbox Scarlett system DLLs on the COM bridge (O23).
//
// xbox_scarlett.h (O12) implements XCRDAPI.dll (DirectStorage), GameInput.dll and xg_xs.dll as
// host C++ objects whose methods take host pointers. Recompiled guest code cannot call those: it
// makes vtable calls on the interface pointer it gets back. This module gives the guest bridge
// objects (guest/com_bridge.h) instead, with vtables laid out the way the title calls them, and
// implements the guest-facing entry points as guest-state import thunks (pg6_gthunk_*, bound by
// R5 in recomp/src/stubs.rs ahead of any family shim).
//
// Evidence (static survey of the title executable; aggregate data only: guest addresses, GUIDs,
// slot indices, call-site counts, argument counts and struct offsets).
//
// XCRDAPI.dll!DStorageGetFactory(REFIID, void**) -> HRESULT
//   One call site, from a one-time init function; IID {62959d11-d45f-4884-a0a3-f7862ea243ae};
//   the factory is kept in one global and released at shutdown (slot 2, 1 tail-call site).
//   Factory slots used (arity includes `this`, h/m/l = confidence of the name):
//     s4  OpenFile(PCWSTR path, REFIID, void**)       2 sites, arity 4, IID {43704f9f-27c6-4a9f-
//                                                     8815-d16527894df4}, UTF-16 MAX_PATH buffer  h
//     s5  CreateStatusArray(UINT32 capacity, PCSTR name, REFIID, void**)  1 site, arity 5,
//                                                     capacity 1, IID {cd66d752-2786-44f7-9070-
//                                                     e2a6fdad3f1b}                                 h
//     s8  (UINT32) -> HRESULT                         1 site, arity 2, argument 0x2000000 (32 MiB)
//                                                     right after creation: SetStagingBufferSize  m
//     s10 CreateQueue(const DSTORAGE_QUEUE_DESC*, REFIID, void**)  1 site, arity 4, IID
//                                                     {3c02cb0c-c050-4195-99a1-a1ce94ba33c5}      h
//   The PC IDStorageFactory order (CreateQueue, OpenFile, CreateStatusArray, SetDebugFlags,
//   SetStagingBufferSize) does not hold: CreateQueue is slot 10 and the staging size is slot 8.
//   Slots 3, 6, 7 and 9 are never called and stay unnamed logging stubs.
//   Queue (one per engine stream wrapper, stored in a heap object; 'sdsm' wrapper of 0x80 bytes):
//     s3  EnqueueRequest(const DSTORAGE_REQUEST*)     1 site, arity 2                             h
//     s4  EnqueueStatus(IDStorageStatusArray*, UINT32) 1 site, arity 3, index 0                   h
//     s6  Submit()                                    1 site, arity 1                             h
//     s8  Close()                                     1 site, arity 1, in the wrapper teardown,
//                                                     which never calls Release on the queue      m
//     s5 EnqueueSignal, s7 CancelRequestsWithTag, s9 GetErrorEvent, s10 RetrieveErrorRecord,
//     s11 Query: never called; implemented at the PC IDStorageQueue positions, which the used
//     slots 3/4/6/8 agree with                                                                    l
//   Status array: s3 IsComplete(UINT32) -> bool, 2 sites, arity 2, polled with Sleep(1) after
//     EnqueueStatus + Submit (h); s2 Release, 1 site; s4 GetHResult never called (l).
//   File: no slot is ever called (the engine keeps the IDStorageFile* in a file handle and never
//     releases it; sizes come from Win32 CreateFileW/ReadFile on the same file). s3 Close and
//     s4 GetFileInformation follow the PC order (l).
//   DSTORAGE_QUEUE_DESC as filled by the title (0x20 bytes, same layout as the PC struct):
//     +0x00 u64 source type (0 file / 1 memory), +0x08 u16 capacity (count + 63), +0x0A i8
//     priority (0, 1 or 2), +0x10 const char* name, +0x18 u64 = 1 (the PC Device slot).
//   DSTORAGE_REQUEST as filled by the title's only request-building site (stack struct, zeroed
//   +0x10..+0x48 before the field writes, so 0x48 bytes):
//     +0x00 u64 options: bits 0..7 compression format (1 when the engine's per-entry codec is 3,
//           else 0); bit 22 (0x400000) source = memory (set for memory-source queues); no other
//           bit is ever set, so every destination is the memory destination (type 0)
//     +0x08 u64 destination buffer (guest memory), +0x10 u64 0, +0x18 u32 destination size
//           (the uncompressed size for compressed requests, else the source size)
//     +0x20 u64 source: IDStorageFile* (file source) or memory pointer (memory source)
//     +0x28 u64 source offset (0 for memory sources), +0x30 u32 source size (bytes on disk)
//     +0x34 u32 0 (uncompressed size field, unused by the title), +0x38 u64 cancellation tag (0),
//     +0x40 const char* name (null)
//   Compared with the public PC layout (Options 8 bytes with SourceType at bit 8, Source union at
//   +0x08, Destination at +0x20) the destination comes first and the memory-source flag is bit 22.
//   No GPU destinations (D3D12 resources) are used. Compression format 1 is the only one: the
//   archives read through DirectStorage (.ttpf, header magic "RDS5", not encrypted) contain
//   RFC 1950 zlib streams at 256-byte aligned offsets that inflate to whole 4 KiB pages, so
//   format 1 is implemented here as zlib (with a raw DEFLATE fallback).
//
// Request execution
// -----------------
// Guest threads only enqueue: EnqueueRequest copies the 0x48-byte request and pins the host side
// of the file / status array objects (no COM reference: a guest Release while a request is
// pending does not cut the read short, and no bridge object ever holds a COM reference to
// another, so reset_com_bridge() can drop them in any order). Submit (or the automatic submit at half capacity) hands
// the batch to a small pool of runtime worker threads (one by default, set_dstorage_worker_threads)
// that execute each queue in FIFO order; a queue's status and signal entries are fences: they
// complete after every request enqueued before them, with the first failure HRESULT of the
// requests since the previous status entry. Between two fences the requests are sorted by file
// and offset, and requests of one file whose ranges are at most `max_gap` bytes apart are read
// with one call into the file layer (at most `max_span` bytes, and never more than the staging
// buffer size) and scattered from a host staging buffer; a lone uncompressed request is read
// straight into guest memory. Reads go through the O12 host file objects (virtual file table,
// host read callback, HTTPFS mailbox, /userdata overlay) by way of a per-worker host queue.
// The queue is bounded by its capacity: an enqueue on a full queue submits what is pending and
// blocks until a worker has made room. Close() drops unsubmitted entries (their status entries
// complete with E_ABORT) and waits for submitted work; workers stop in shutdown/reset.
//
// HRESULTs in status entries and error records: E_INVALIDARG for a malformed request (bad source
// object, null or unwritable destination, destination smaller than the data), E_NOTIMPL for an
// unsupported compression format or option bit, ERROR_HANDLE_EOF (0x80070026) for a read past the
// end of the file, ERROR_INVALID_DATA (0x8007000D) for a corrupt or size-mismatched zlib stream,
// E_HANDLE for a closed file, and whatever else the file layer reports. Cancelled requests are
// skipped without counting as failures.
//
// GameInput.dll ordinal 1 = GameInputCreate(IGameInput**) -> HRESULT
//   One call site, in a lazy getter that caches the interface in one global (AddRef/Release on
//   the swap); 10 callers of the getter. No IID is involved. IGameInput slots used (the GameInput
//   v0 public order holds for every used slot):
//     s4  GetCurrentReading(kind, device, reading**)        2 sites, arity 4, kinds 0x40000
//                                                           (Gamepad) and 0x10 (Keyboard), the
//                                                           device found by enumeration         h
//     s5  GetNextReading(ref, kind, device, reading**)      1 site, arity 5, kind Keyboard      h
//     s6  GetPreviousReading(ref, kind, device, reading**)  1 site, arity 5, kind Gamepad       h
//     s9  RegisterDeviceCallback(device, kind, statusFilter, enumerationKind, context, callback,
//         token*)                                           2 sites, arity 8: device null, kinds
//                                                           Gamepad / Keyboard, filter 1
//                                                           (Connected), enumeration 2
//                                                           (Blocking)                          h
//     s13 UnregisterCallback(token, timeoutUs) -> bool      2 sites, arity 3 (5000, UINT64_MAX) h
//     s16 FindDeviceFromId(const APP_LOCAL_DEVICE_ID*, device**)  1 site, arity 3               h
//   IGameInputReading: s14 GetKeyCount (1 site), s15 GetKeyState(count, GameInputKeyState*) (1
//     site, 12-byte entries, scan code at +0 and virtual key at +8 read), s22 GetGamepadState
//     (3 sites, bool result, buttons at +0 compared between readings), s1/s2 AddRef/Release on
//     every reading (h).
//   IGameInputDevice: s3 GetDeviceInfo (2 sites, in both device callbacks; the 32-byte
//     APP_LOCAL_DEVICE_ID at +0x20 of GameInputDeviceInfo is copied), s10 SetRumbleState(const
//     GameInputRumbleParams*) (1 site), s1/s2 (h).
//   Device callbacks test bit 0 (Connected) of the current status (5th argument) and AddRef the
//   device they keep. Unused slots are named logging stubs in the v0 order (low confidence).
//
// xg_xs.dll!XGCreateTextureComputer(const XG_RESOURCE_DESC*, XGTextureAddressComputer**) -> HRESULT
//   4 call sites. The computer is not an IUnknown: slot 1 drops it (3 sites), slot 0 is never
//   called (AddRef). Used slots:
//     s2  GetResourceLayout(XG_RESOURCE_LAYOUT*)   2 sites: 0x18-byte header + 4 planes of 0x6C8
//                                                  bytes, 15 mip entries of 0x70 bytes from plane
//                                                  +0x38; the title reads mip +0x00 (u64), +0x10
//                                                  (u64), +0x1C (u32) and copies whole planes    m
//     s6  GetTexelElementOffsetBytes(plane, mip, x, y, zOrSlice, sample, bitOffset*) -> UINT64, 2 sites
//         (element coordinates; the title divides BC texel coordinates by 4)                          m
//     s8  CopyIntoSubresource(dst, plane, subresource, src, rowPitch, slicePitch)  1 site, arity 7 m
//     s10 GetResourceTiling(numTiles*, packedMipInfo*, tileShape*, numTilings*, firstTiling,
//         tilings*)                                1 site, arity 7, the D3D12 GetResourceTiling
//                                                  shapes (12-byte packed-mip info, tile shape and
//                                                  subresource tiling entries)                   m
//   XG_RESOURCE_DESC as built by the title: +0x00 u32 dimension (2/3/4), +0x08 u64 alignment,
//   +0x10 u64 width, +0x18 u32 height, +0x1C u16 depth or array size, +0x1E u16 mip levels,
//   +0x20 u32 DXGI format, +0x24 u32 sample count, +0x28 u32 sample quality, +0x2C u32 swizzle
//   mode (0x100 | n), +0x30 u32 flags.
// xg_xs.dll!XGSuggestSwizzleMode(const XG_RESOURCE_DESC*, UINT32 (0), UINT32* mode) -> HRESULT
//   1 call site, writing desc +0x2C; the one-argument host shim (O12) does not match it.
// Fidelity: for swizzle modes verified against AMD addrlib (256B/4KB/64KB _S and _D, thick 3D
// for _S) GetResourceLayout reports the GFX10 surface (sizes, mip offsets, pitches) and
// CopyIntoSubresource writes that layout (gpu/gfx10_detile.h, the placement the runtime's detiler
// reads); other descriptors get linear layouts (row pitch aligned to 256 bytes, mips one after
// another). GetResourceTiling uses the D3D12 standard 64 KiB tile shapes (the same block grid as
// GFX10 64KB modes; the packed-mip threshold is D3D12's). For the verified 64KB modes the tiles
// are numbered in GFX10 memory order (mip tail first as one tile, then the larger mips
// smallest-first), which the title assumes: it places the tail at tile 0 itself.

namespace playgta6::os::scarlett::guest_api {

namespace com = playgta6::guest::com;
using com::HResult;

// ---------------------------------------------------------------------------
// XCRDAPI.dll (DirectStorage on Xbox) identities and guest structs
// ---------------------------------------------------------------------------

inline constexpr com::Guid kIidDStorageFactory = com::make_guid("62959d11-d45f-4884-a0a3-f7862ea243ae");
inline constexpr com::Guid kIidDStorageFile = com::make_guid("43704f9f-27c6-4a9f-8815-d16527894df4");
inline constexpr com::Guid kIidDStorageQueue = com::make_guid("3c02cb0c-c050-4195-99a1-a1ce94ba33c5");
inline constexpr com::Guid kIidDStorageStatusArray = com::make_guid("cd66d752-2786-44f7-9070-e2a6fdad3f1b");

inline constexpr HResult kHrEof = static_cast<HResult>(0x80070026u);          // ERROR_HANDLE_EOF
inline constexpr HResult kHrInvalidData = static_cast<HResult>(0x8007000Du);  // ERROR_INVALID_DATA
inline constexpr HResult kHrFileNotFound = static_cast<HResult>(0x80070002u);
inline constexpr HResult kHrPending = static_cast<HResult>(0x8000000Au);
inline constexpr HResult kHrAbort = static_cast<HResult>(0x80004004u);
inline constexpr HResult kHrHandle = static_cast<HResult>(0x80070006u);

// DSTORAGE_REQUEST_OPTIONS, public GDK layout:
//   bit 0      ZlibDecompress
//   bits 1-4   BcpackMode: the BC format of a BCPack stream (os/bcpack.h). Decoded only with
//              user-supplied format tables (docs/BCPACK.md); without them the request fails
//              E_NOTIMPL. With bit 0 too, zlib runs first into uncompressed_size bytes.
//   bit 21     DestinationIsPhysicalPages: `destination` is a guest array of physical page
//              addresses (one per page of the destination) instead of a contiguous buffer
//   bit 22     source is memory (not an IDStorageFile)
// Any other bit set fails the request with E_NOTIMPL.
inline constexpr uint64_t kDStorageOptionZlib = 1ull << 0;
inline constexpr uint64_t kDStorageOptionBcpackShift = 1;
inline constexpr uint64_t kDStorageOptionBcpackMask = 0xFull << kDStorageOptionBcpackShift;
inline constexpr uint64_t kDStorageOptionDestinationIsPhysicalPages = 1ull << 21;
inline constexpr uint64_t kDStorageOptionSourceMemory = 1ull << 22;
inline constexpr uint64_t kDStorageOptionKnownBits = kDStorageOptionZlib | kDStorageOptionBcpackMask |
                                                     kDStorageOptionDestinationIsPhysicalPages |
                                                     kDStorageOptionSourceMemory;

struct DStorageOptions {
    bool zlib = false;
    uint32_t bcpack_mode = 0;  // 0: none
    bool physical_pages = false;
    bool source_memory = false;
    uint64_t unknown_bits = 0;
    bool compressed() const { return zlib || bcpack_mode != 0; }
};

constexpr DStorageOptions parse_dstorage_options(uint64_t options) {
    DStorageOptions o;
    o.zlib = (options & kDStorageOptionZlib) != 0;
    o.bcpack_mode = static_cast<uint32_t>((options & kDStorageOptionBcpackMask) >> kDStorageOptionBcpackShift);
    o.physical_pages = (options & kDStorageOptionDestinationIsPhysicalPages) != 0;
    o.source_memory = (options & kDStorageOptionSourceMemory) != 0;
    o.unknown_bits = options & ~kDStorageOptionKnownBits;
    return o;
}

struct DStorageQueueDesc {
    uint64_t source_type;  // 0 file, 1 memory
    uint16_t capacity;
    int8_t priority;
    uint8_t reserved[5];
    uint64_t name;    // const char*
    uint64_t device;  // the title writes 1
};
static_assert(sizeof(DStorageQueueDesc) == 0x20 && std::is_trivially_copyable_v<DStorageQueueDesc>);

struct DStorageRequest {
    uint64_t options;
    uint64_t destination;
    uint64_t destination_reserved;
    uint32_t destination_size;
    uint32_t reserved0;
    uint64_t source;  // IDStorageFile* or memory pointer
    uint64_t source_offset;
    uint32_t source_size;
    uint32_t uncompressed_size;  // 0: the destination size
    uint64_t cancellation_tag;
    uint64_t name;  // const char*
};
static_assert(sizeof(DStorageRequest) == 0x48 && std::is_trivially_copyable_v<DStorageRequest>);
static_assert(offsetof(DStorageRequest, source) == 0x20 && offsetof(DStorageRequest, source_size) == 0x30);
static_assert(offsetof(DStorageRequest, cancellation_tag) == 0x38);

// DSTORAGE_QUEUE_INFO (PC layout; the title never calls Query).
struct DStorageQueueInfo {
    DStorageQueueDesc desc;
    uint16_t empty_slot_count;
    uint16_t request_count_until_auto_submit;
    uint32_t reserved;
};
static_assert(sizeof(DStorageQueueInfo) == 0x28);

// DSTORAGE_ERROR_RECORD (PC layout: FailureCount, then DSTORAGE_ERROR_FIRST_FAILURE { HRESULT,
// DSTORAGE_COMMAND_TYPE, parameters }; for a request the parameters are the file name, the
// request name and the request). The title never calls RetrieveErrorRecord.
inline constexpr uint32_t kDStorageCommandRequest = 0;
inline constexpr uint32_t kDStorageCommandStatus = 1;
inline constexpr uint32_t kDStorageCommandSignal = 2;
struct DStorageErrorRecord {
    uint32_t failure_count;
    uint32_t reserved0;  // DSTORAGE_ERROR_FIRST_FAILURE is 8-byte aligned (its union holds pointers)
    HResult hresult;
    uint32_t command_type;
    char16_t filename[260];
    char request_name[260];
    uint8_t reserved1[4];
    DStorageRequest request;
};
static_assert(offsetof(DStorageErrorRecord, hresult) == 8 && offsetof(DStorageErrorRecord, filename) == 16);
static_assert(offsetof(DStorageErrorRecord, request) == 800 && sizeof(DStorageErrorRecord) == 872);

// BY_HANDLE_FILE_INFORMATION (52 bytes, 4-byte aligned FILETIMEs).
struct ByHandleFileInformation {
    uint32_t file_attributes;
    uint32_t creation_time[2];
    uint32_t last_access_time[2];
    uint32_t last_write_time[2];
    uint32_t volume_serial_number;
    uint32_t file_size_high;
    uint32_t file_size_low;
    uint32_t number_of_links;
    uint32_t file_index_high;
    uint32_t file_index_low;
};
static_assert(sizeof(ByHandleFileInformation) == 52 && offsetof(ByHandleFileInformation, file_size_high) == 32);

// ---------------------------------------------------------------------------
// DirectStorage runtime controls and accounting
// ---------------------------------------------------------------------------

struct DStorageStats {
    uint64_t requests_completed = 0;  // executed, successful or not (cancelled excluded)
    uint64_t requests_failed = 0;
    uint64_t requests_cancelled = 0;
    uint64_t bytes_delivered = 0;     // bytes written to destinations
    uint64_t file_reads = 0;          // calls into the file layer
    uint64_t coalesced_requests = 0;  // requests served by a read shared with another request
    uint64_t decompressed_requests = 0;
    uint64_t submits = 0;             // explicit + automatic
    uint64_t live_queues = 0;
    uint64_t physical_page_requests = 0;  // requests with DestinationIsPhysicalPages (option bit 21)
    uint64_t bcpack_requests = 0;  // BCPack requests (option bits 1-4)
    uint64_t bcpack_decoded = 0;   // ... unpacked into their destination (needs the user's tables)
};

DStorageStats dstorage_stats();

// Worker threads executing submitted requests (default 1, at most 16). Takes effect for workers
// started afterwards (the pool starts on the first Submit; reset_xbox_scarlett_guest() stops it).
void set_dstorage_worker_threads(uint32_t count);
// Coalescing of nearby reads of one file (defaults: gap 64 KiB, span 4 MiB; gap 0 still merges
// exactly adjacent ranges, span 0 disables coalescing).
void set_dstorage_coalescing(uint64_t max_gap, uint64_t max_span);

// zlib (RFC 1950) or raw DEFLATE (RFC 1951) decompression, used for compression format 1.
enum class InflateStatus { Ok, BadHeader, BadData, OutputTooSmall, Truncated, ChecksumMismatch };
// Decompresses one complete stream from src into dst; *out_len = bytes produced. A source that
// starts with a valid zlib header is a zlib stream (Adler-32 checked), anything else raw DEFLATE.
InflateStatus inflate_stream(const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_cap, size_t* out_len);

// ---------------------------------------------------------------------------
// GameInput.dll (v0) guest structs and constants
// ---------------------------------------------------------------------------

inline constexpr uint32_t kGameInputKindKeyboard = 0x00000010u;
inline constexpr uint32_t kGameInputKindMouse = 0x00000020u;
inline constexpr uint32_t kGameInputKindGamepad = 0x00040000u;
inline constexpr uint32_t kGameInputDeviceConnected = 0x1u;
inline constexpr uint32_t kGameInputDeviceInputEnabled = 0x2u;
inline constexpr HResult kGameInputDeviceNotFound = static_cast<HResult>(0x838A0002u);
inline constexpr HResult kGameInputReadingNotFound = static_cast<HResult>(0x838A0003u);

struct GameInputGamepadState {
    uint32_t buttons;  // GameInputGamepadButtons: Menu 0x1, View 0x2, A 0x4, B 0x8, X 0x10, Y 0x20, ...
    float left_trigger;
    float right_trigger;
    float left_thumbstick_x;
    float left_thumbstick_y;
    float right_thumbstick_x;
    float right_thumbstick_y;
};
static_assert(sizeof(GameInputGamepadState) == 28);

struct GameInputKeyState {
    uint32_t scan_code;
    uint32_t code_point;
    uint8_t virtual_key;
    uint8_t is_dead_key;
    uint8_t reserved[2];
};
static_assert(sizeof(GameInputKeyState) == 12);

struct AppLocalDeviceId {
    uint8_t value[32];
};

struct GameInputRumbleParams {
    float low_frequency;
    float high_frequency;
    float left_trigger;
    float right_trigger;
};

// GameInputDeviceInfo of the virtual devices, kept in each device object's guest payload. Only
// deviceId (+0x20) is read by the title; the other filled fields follow the v0 layout (low
// confidence).
inline constexpr uint64_t kGameInputDeviceInfoSize = 0x100;
inline constexpr uint64_t kDeviceInfoVendorId = 0x04;
inline constexpr uint64_t kDeviceInfoProductId = 0x06;
inline constexpr uint64_t kDeviceInfoDeviceId = 0x20;
inline constexpr uint64_t kDeviceInfoSupportedInput = 0x68;

// The virtual gamepad's last SetRumbleState (host integration and tests).
GameInputRumbleParams virtual_gamepad_rumble();

// ---------------------------------------------------------------------------
// xg_xs.dll guest structs
// ---------------------------------------------------------------------------

struct XgResourceDesc {
    uint32_t dimension;  // D3D12_RESOURCE_DIMENSION: 2 Texture1D, 3 Texture2D, 4 Texture3D
    uint32_t reserved0;
    uint64_t alignment;
    uint64_t width;
    uint32_t height;
    uint16_t depth_or_array_size;
    uint16_t mip_levels;
    uint32_t format;  // DXGI_FORMAT
    uint32_t sample_count;
    uint32_t sample_quality;
    uint32_t swizzle_mode;
    uint32_t flags;
};
inline constexpr size_t kXgResourceDescBytes = 0x34;  // bytes the title writes
static_assert(offsetof(XgResourceDesc, flags) == 0x30);

inline constexpr uint64_t kXgLayoutPlaneOffset = 0x18;
inline constexpr uint64_t kXgLayoutPlaneSize = 0x6C8;
inline constexpr uint64_t kXgLayoutMaxPlanes = 4;
inline constexpr uint64_t kXgPlaneMipOffset = 0x38;
inline constexpr uint64_t kXgMipLayoutSize = 0x70;
inline constexpr uint64_t kXgMaxMips = 15;
inline constexpr uint64_t kXgResourceLayoutSize = kXgLayoutPlaneOffset + kXgLayoutMaxPlanes * kXgLayoutPlaneSize;
static_assert(kXgPlaneMipOffset + kXgMaxMips * kXgMipLayoutSize == kXgLayoutPlaneSize);

// Mip entry written by GetResourceLayout. The title reads +0x00, +0x10 and +0x1C; the field names
// are the hypothesis (total size, 2D slice size, row pitch in bytes) the values follow.
struct XgMipLayout {
    uint64_t size_bytes;
    uint64_t offset_bytes;
    uint64_t slice_2d_size_bytes;
    uint32_t pitch_pixels;
    uint32_t pitch_bytes;
    uint32_t alignment_bytes;
    uint32_t padded_width_elements;
    uint32_t padded_height_elements;
    uint32_t padded_depth_or_array_size;
    uint32_t width_elements;
    uint32_t height_elements;
    uint32_t depth_or_array_size;
    uint32_t sample_count;
    uint8_t reserved[0x70 - 0x40];
};
static_assert(sizeof(XgMipLayout) == kXgMipLayoutSize && offsetof(XgMipLayout, pitch_bytes) == 0x1C);

// D3D12 tiling structs used by GetResourceTiling.
struct XgPackedMipInfo {
    uint8_t num_standard_mips;
    uint8_t num_packed_mips;
    uint8_t reserved[2];
    uint32_t num_tiles_for_packed_mips;
    uint32_t start_tile_index_in_overall_resource;
};
struct XgTileShape {
    uint32_t width_in_texels;
    uint32_t height_in_texels;
    uint32_t depth_in_texels;
};
struct XgSubresourceTiling {
    uint32_t width_in_tiles;
    uint16_t height_in_tiles;
    uint16_t depth_in_tiles;
    uint32_t start_tile_index_in_overall_resource;
};
static_assert(sizeof(XgPackedMipInfo) == 12 && sizeof(XgTileShape) == 12 && sizeof(XgSubresourceTiling) == 12);
inline constexpr uint32_t kXgPackedTile = 0xFFFFFFFFu;  // D3D12_PACKED_TILE
inline constexpr uint32_t kXgDefaultSwizzleMode = 0x100u;  // the mode the title itself uses by default

// ---------------------------------------------------------------------------
// Registration and testing
// ---------------------------------------------------------------------------

// Registers the guest-callable exports (XCRDAPI.dll!DStorageGetFactory, ...) in the O7 module
// registry with their guest-callable synthetic addresses (idempotent, thread-safe). Called by
// win32::register_all_shim_exports().
void register_xbox_scarlett_guest_exports();

// AcpHal.dll!AcpHalCreate's guest object (acp_hal_guest.cpp): what a test can observe of the
// IAcpHal at guest address `hal` (valid = false for anything else).
struct AcpHalState {
    bool valid = false;
    bool connected = false;
    uint64_t commands_2 = 0;  // submitted commands with id 2 / id 5
    uint64_t commands_5 = 0;
};
AcpHalState acp_hal_state(uint64_t hal);

// The AcpHal XMA context commands (acp_hal_guest.cpp): 5 / 7 enable / disable one context
// (payload word: index), 6 / 8 enable / disable many (payload: bitmask words, bit i = context i),
// 18 advance a context's output read offset (payload: index, blocks consumed). A worker thread
// fills enabled contexts' output rings (os/xma_decoder.h; silence until a WMA Pro decoder exists).
struct XmaCommandStats {
    uint64_t enables = 0, disables = 0, mask_enables = 0, mask_disables = 0, read_advances = 0;
    uint64_t rejected = 0;     // bad payload or context index
    uint32_t enabled_now = 0;  // contexts enabled at the moment
    // The decode worker: XMA frames walked, 256-byte output blocks written, loops taken, times a
    // voice ran out of input, frames or buffers it could not walk.
    uint64_t frames = 0, blocks_written = 0, loops = 0, input_exhausted = 0, bad_frames = 0;
};
XmaCommandStats xma_command_stats();
bool xma_context_enabled(uint32_t index);
void reset_xma_commands();  // tests only
// Tests only: use this context array instead of AcpHalAllocateShapeContexts' (va 0: back to it).
void set_xma_contexts_for_tests(uint64_t va, uint32_t count);
// Tests only: resolve APU addresses with `map` (empty: the APU heap again), keep the worker
// thread from starting, and run one decode pass on the calling thread.
void set_xma_apu_map_for_tests(std::function<uint8_t*(uint32_t, size_t)> map);
void set_xma_worker_allowed_for_tests(bool allowed);
void pump_xma_for_tests();

// chad.dll XDsp guest model (acp_hal_guest.cpp): open connections and commands accepted.
struct XDspStats {
    uint64_t connections = 0;
    uint64_t commands = 0;
};
XDspStats xdsp_stats();

// xapu.dll guest model (acp_hal_guest.cpp): open connections, commands accepted, results not
// yet dequeued, and decode commands (type 3) decoded / failed with the float32 bytes written.
struct XApuStats {
    uint64_t connections = 0;
    uint64_t commands = 0;
    uint64_t pending_results = 0;
    uint64_t decoded_packets = 0;
    uint64_t failed_packets = 0;
    uint64_t decoded_bytes = 0;
    uint64_t type3_results = 0;  // type-3 results posted (one per decode command)
    uint64_t results_posted = 0;  // results of every type posted
    uint64_t type1_commands = 0, type2_commands = 0, type3_commands = 0, type5_commands = 0;
    uint64_t unhandled_commands = 0;  // commands of an unknown type (accepted, no result)
};
XApuStats xapu_stats();

// PG6_AUDIO_TRACE=1 (read once): audio diagnostics (XApu voice lifetimes, audio file requests).
bool audio_trace_enabled();

// The default audio render endpoint (mmdevice_guest.cpp): combase!CoCreateInstance(
// CLSID_MMDeviceEnumerator) -> IMMDeviceEnumerator with one active eRender endpoint, its
// IAudioClient (48 kHz float32 stereo mix format, 10 ms period) and ISpatialAudioClient (7.1.4
// static objects, 48 kHz float32 mono, 480 frames per pass). Rendered audio is consumed at the
// real-time rate and discarded; a sink set here receives each pass's PCM (interleaved float32,
// `channels` per frame; object_type 0 for an IAudioRenderClient buffer, else the
// AudioObjectType of one spatial static object). Called on the guest's audio thread: copy, do
// not block.
using AudioRenderSink = void (*)(void* user, uint32_t object_type, const float* pcm, uint32_t frames,
                                 uint32_t channels, uint32_t sample_rate);
void set_audio_render_sink(AudioRenderSink sink, void* user);
// The stereo downmix (os/audio_downmix.h): every period the 7.1.4 objects and the render
// stream are mixed into interleaved float32 stereo at 48 kHz, pushed into the PCM ring
// (playgta6_audio_ring) and handed to this callback.
using AudioStereoSink = void (*)(void* user, const float* stereo, uint32_t frames, uint32_t sample_rate);
void set_audio_stereo_sink(AudioStereoSink sink, void* user);

struct AudioEndpointStats {
    uint64_t activations = 0;     // IMMDevice::Activate calls that returned a client
    uint64_t spatial_passes = 0;  // completed Begin/EndUpdatingAudioObjects passes
    uint64_t render_frames = 0;   // frames released through IAudioRenderClient
    // What the title actually wrote (tells "the title renders silence" from "we lose its audio"):
    uint64_t spatial_object_buffers = 0;   // static-object buffers read at EndUpdatingAudioObjects
    uint64_t spatial_nonzero_buffers = 0;  // ... of them holding a non-zero sample
    uint64_t spatial_nonzero_passes = 0;   // passes with a non-zero sample in any object buffer
    uint64_t render_nonzero_frames = 0;    // released render frames of buffers with a non-zero sample
    uint64_t render_silent_releases = 0;   // ReleaseBuffer calls with AUDCLNT_BUFFERFLAGS_SILENT
};
AudioEndpointStats audio_endpoint_stats();

// Tests only: with a manual clock the 10 ms period thread is stopped and audio_engine_tick()
// advances every started stream by one period (padding drains, the stream's event is set).
void audio_engine_manual_clock(bool manual);
void audio_engine_tick();
// Tests only: stops the period thread, forgets the streams, the sink and the statistics.
void reset_audio_endpoint();
// Boot option audio_endpoint (on by default): off, CoCreateInstance(CLSID_MMDeviceEnumerator)
// answers REGDB_E_CLASSNOTREG like any unknown class. Survives reset_audio_endpoint.
void set_audio_endpoint_enabled(bool on);
bool audio_endpoint_enabled();

// Tests only: stops the DirectStorage workers (waiting for running work), forgets the factory
// singleton and zeroes the statistics. Call before reset_com_bridge().
void reset_xbox_scarlett_guest();

namespace detail {
void reset_input_guest();  // GameInput / XG part of reset_xbox_scarlett_guest()
}  // namespace detail

}  // namespace playgta6::os::scarlett::guest_api

extern "C" {

// XCRDAPI.dll!DStorageGetFactory(REFIID riid, void** ppv) -> HRESULT
void pg6_gthunk_dstorage_get_factory(void* state);
// GameInput.dll ordinal 1: GameInputCreate(IGameInput** game_input) -> HRESULT
void pg6_gthunk_game_input_create(void* state);
// xg_xs.dll!XGCreateTextureComputer(const XG_RESOURCE_DESC*, XGTextureAddressComputer**) -> HRESULT
void pg6_gthunk_xg_create_texture_computer(void* state);
// xg_xs.dll!XGSuggestSwizzleMode(const XG_RESOURCE_DESC*, UINT32, UINT32* mode) -> HRESULT
void pg6_gthunk_xg_suggest_swizzle_mode(void* state);
// AcpHal.dll!AcpHalCreate(IAcpHal** hal) -> HRESULT (guest object without an IUnknown prefix)
void pg6_gthunk_acp_hal_create(void* state);
// chad.dll XDsp (acp_hal_guest.cpp has the call shapes)
void pg6_gthunk_xdsp_connect_with_maximum_stream_limit(void* state);
void pg6_gthunk_xdsp_activate(void* state);
void pg6_gthunk_xdsp_deactivate(void* state);
void pg6_gthunk_xdsp_disconnect(void* state);
void pg6_gthunk_xdsp_submit_command(void* state);
// xapu.dll (acp_hal_guest.cpp has the call shapes)
void pg6_gthunk_xapu_connect(void* state);
void pg6_gthunk_xapu_disconnect(void* state);
void pg6_gthunk_xapu_enqueue_command(void* state);
void pg6_gthunk_xapu_get_result_count(void* state);
void pg6_gthunk_xapu_dequeue_result(void* state);
// combase.dll!CoCreateInstance(REFCLSID, IUnknown*, DWORD, REFIID, void**) -> HRESULT
// (CLSID_MMDeviceEnumerator: mmdevice_guest.cpp; every other class: the host shim's answer)
void pg6_gthunk_co_create_instance(void* state);

}  // extern "C"
