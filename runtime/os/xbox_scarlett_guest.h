#pragma once

#include <cstddef>
#include <cstdint>
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
// Guest threads only enqueue: EnqueueRequest copies the 0x48-byte request and takes references
// to the file / status array objects. Submit (or the automatic submit at half capacity) hands
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

inline constexpr uint64_t kDStorageOptionFormatMask = 0xFFu;
inline constexpr uint64_t kDStorageOptionSourceMemory = 1ull << 22;
inline constexpr uint32_t kDStorageFormatNone = 0;
inline constexpr uint32_t kDStorageFormatZlib = 1;

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
// Registration and testing
// ---------------------------------------------------------------------------

// Registers the guest-callable exports (XCRDAPI.dll!DStorageGetFactory, ...) in the O7 module
// registry with their guest-callable synthetic addresses (idempotent, thread-safe). Called by
// win32::register_all_shim_exports().
void register_xbox_scarlett_guest_exports();

// Tests only: stops the DirectStorage workers (waiting for running work), forgets the factory
// singleton and zeroes the statistics. Call before reset_com_bridge().
void reset_xbox_scarlett_guest();

}  // namespace playgta6::os::scarlett::guest_api

extern "C" {

// XCRDAPI.dll!DStorageGetFactory(REFIID riid, void** ppv) -> HRESULT
void pg6_gthunk_dstorage_get_factory(void* state);

}  // extern "C"
