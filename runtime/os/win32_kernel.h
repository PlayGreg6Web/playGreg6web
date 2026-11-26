#pragma once

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "os/httpfs_mailbox.h"
#include "os/threading.h"
#include "os/vmem.h"

// Portable C-ABI types (prefixed with Pg6* to avoid collisions with <windows.h> on MSVC).
using Pg6Handle = uint64_t;

struct Pg6MemoryBasicInformation {
    uint64_t base_address = 0;
    uint64_t allocation_base = 0;
    uint32_t allocation_protect = 0;
    uint32_t partition_id = 0;
    uint64_t region_size = 0;
    uint32_t state = 0;
    uint32_t protect = 0;
    uint32_t type = 0;
};

struct Pg6FileTime {
    uint32_t low_date_time = 0;
    uint32_t high_date_time = 0;
};

// SRWLOCK (8 bytes on x64). Like CRITICAL_SECTION below, the guest owns this storage: the host lock
// lives in a table keyed by the lock's address, so no host object is ever placed in (or copied
// over) guest memory. A host std::shared_mutex is 8 bytes on MSVC but 56 on wasm64/POSIX, and
// writing it in place corrupted the guest variables that follow the SRWLOCK.
struct Pg6SrwLock {
    uint64_t reserved = 0;
};
static_assert(sizeof(Pg6SrwLock) == 8, "SRWLOCK is pointer-sized on x64");

// CRITICAL_SECTION (RTL_CRITICAL_SECTION, 40 bytes on x64). The guest owns this storage and may hand
// in uninitialized or statically zeroed memory, so the runtime never interprets it as a pointer:
// the lock itself lives in a table keyed by the section's address (created by Initialize*, or on
// first Enter for zero-initialized sections). Initialize writes the Windows initial field values
// and Enter/Leave keep RecursionCount and OwningThread current for code that reads them inline.
struct Pg6CriticalSection {
    uint64_t debug_info;      // 0x00 (-1: no debug info, as Windows)
    int32_t lock_count;       // 0x08 (-1 after initialization)
    int32_t recursion_count;  // 0x0C
    uint64_t owning_thread;   // 0x10 (thread id while owned, else 0)
    uint64_t lock_semaphore;  // 0x18
    uint64_t spin_count;      // 0x20
};

namespace playgta6::os::win32 {

using Handle = Pg6Handle;
using MemoryBasicInformation = Pg6MemoryBasicInformation;
using FileTime = Pg6FileTime;
using SrwLock = Pg6SrwLock;
using CriticalSection = Pg6CriticalSection;

// Handle sentinels.
inline constexpr Handle kNullHandle = 0;
inline constexpr Handle INVALID_HANDLE_VALUE = ~uint64_t{0};

// VirtualAlloc / VirtualFree flags.
inline constexpr uint32_t MEM_COMMIT = kMemCommit;
inline constexpr uint32_t MEM_RESERVE = kMemReserve;
inline constexpr uint32_t MEM_DECOMMIT = kMemDecommit;
inline constexpr uint32_t MEM_RELEASE = kMemRelease;
inline constexpr uint32_t MEM_FREE = kMemFree;
inline constexpr uint32_t MEM_PRIVATE = 0x00020000u;

// Page protection flags.
inline constexpr uint32_t PAGE_NOACCESS = static_cast<uint32_t>(PageProtect::NoAccess);
inline constexpr uint32_t PAGE_READONLY = static_cast<uint32_t>(PageProtect::ReadOnly);
inline constexpr uint32_t PAGE_READWRITE = static_cast<uint32_t>(PageProtect::ReadWrite);
inline constexpr uint32_t PAGE_EXECUTE_READ = static_cast<uint32_t>(PageProtect::ExecuteRead);
inline constexpr uint32_t PAGE_EXECUTE_READWRITE =
    static_cast<uint32_t>(PageProtect::ExecuteReadWrite);

// TLS sentinel.
inline constexpr uint32_t TLS_OUT_OF_INDEXES = TlsTable::kInvalidIndex;

// WaitForSingleObject constants.
inline constexpr uint32_t WAIT_OBJECT_0 = 0x00000000u;
inline constexpr uint32_t WAIT_TIMEOUT = 0x00000102u;
inline constexpr uint32_t WAIT_FAILED = 0xFFFFFFFFu;
inline constexpr uint32_t INFINITE_TIMEOUT = 0xFFFFFFFFu;

// File access / sharing / creation / seek constants.
inline constexpr uint32_t GENERIC_READ = 0x80000000u;
inline constexpr uint32_t GENERIC_WRITE = 0x40000000u;

inline constexpr uint32_t FILE_SHARE_READ = 0x00000001u;
inline constexpr uint32_t FILE_SHARE_WRITE = 0x00000002u;
inline constexpr uint32_t FILE_SHARE_DELETE = 0x00000004u;

inline constexpr uint32_t CREATE_NEW = 1u;
inline constexpr uint32_t CREATE_ALWAYS = 2u;
inline constexpr uint32_t OPEN_EXISTING = 3u;
inline constexpr uint32_t OPEN_ALWAYS = 4u;
inline constexpr uint32_t TRUNCATE_EXISTING = 5u;

inline constexpr uint32_t FILE_ATTRIBUTE_NORMAL = 0x00000080u;

inline constexpr uint32_t FILE_BEGIN = 0u;
inline constexpr uint32_t FILE_CURRENT = 1u;
inline constexpr uint32_t FILE_END = 2u;

// Common Win32 error codes for GetLastError / SetLastError.
inline constexpr uint32_t ERROR_SUCCESS = 0u;
inline constexpr uint32_t ERROR_FILE_NOT_FOUND = 2u;
inline constexpr uint32_t ERROR_PATH_NOT_FOUND = 3u;
inline constexpr uint32_t ERROR_ACCESS_DENIED = 5u;
inline constexpr uint32_t ERROR_INVALID_HANDLE = 6u;
inline constexpr uint32_t ERROR_NOT_ENOUGH_MEMORY = 8u;
inline constexpr uint32_t ERROR_WRITE_PROTECT = 19u;
inline constexpr uint32_t ERROR_READ_FAULT = 30u;
inline constexpr uint32_t ERROR_HANDLE_EOF = 38u;
inline constexpr uint32_t ERROR_FILE_EXISTS = 80u;
inline constexpr uint32_t ERROR_INVALID_PARAMETER = 87u;
inline constexpr uint32_t ERROR_INVALID_ADDRESS = 487u;
inline constexpr uint32_t ERROR_NEGATIVE_SEEK = 131u;
inline constexpr uint32_t ERROR_ALREADY_EXISTS = 183u;
inline constexpr uint32_t ERROR_NO_MORE_ITEMS = 259u;

// ---------------------------------------------------------------------------
// Injectable backends & overlay helpers
// ---------------------------------------------------------------------------

VirtualMemoryArena& default_vmem_arena();
// O36: whether the process arena keeps its shared zone (vmem.h kProcessSharedZoneSize: XMem
// physical pages are one memory at every address they are mapped at). On by default; takes effect
// only before the process arena is first used (playgta6_boot --no-xmem-shared-pages: A/B runs).
void set_process_shared_zone(bool on);
void set_vmem_arena(VirtualMemoryArena* arena);
VirtualMemoryArena* get_vmem_arena();

// Live entries of the kernel handle table by kind: dispatcher objects (events, semaphores,
// mutexes, threads, timers, ...), files, and opaque objects (find handles, registry keys, ...).
struct KernelHandleCounts {
    uint64_t dispatcher = 0;
    uint64_t file = 0;
    uint64_t opaque = 0;
};
KernelHandleCounts live_kernel_handles();

TlsTable& default_tls_table();
void set_tls_table(TlsTable* table);
TlsTable* get_tls_table();

void set_httpfs_client(HttpfsMailboxClient* client);
HttpfsMailboxClient* get_httpfs_client();

// Handle-table seam for os/win32_sync: waitable dispatcher objects (events, semaphores, mutexes,
// threads) are implemented there but share this HANDLE namespace and CloseHandle.
class DispatcherObject;
Handle insert_dispatcher_object(std::shared_ptr<DispatcherObject> object);
// Returns nullptr for invalid handles and for non-dispatcher handles (e.g. files).
std::shared_ptr<DispatcherObject> lookup_dispatcher_object(Handle handle);

// Handle-table seam for os/win32_compat (O15): opaque, tagged payload objects (find handles,
// registry keys). lookup/remove return nullptr when the handle is invalid or carries another tag.
// CloseHandle also releases them.
Handle insert_opaque_object(uint32_t tag, std::shared_ptr<void> payload);
std::shared_ptr<void> lookup_opaque_object(Handle handle, uint32_t tag);
std::shared_ptr<void> remove_opaque_object(Handle handle, uint32_t tag);

// In-memory /userdata overlay helpers for writeable saves and tests.
void clear_userdata_overlay();
bool userdata_file_exists(std::string_view path);
bool read_userdata_file(std::string_view path, std::vector<uint8_t>& out);
// Backlog 29: size / ranged read without copying the whole overlay file (a DirectStorage read of a
// few KiB used to copy the file first). Range: false when the file is missing or the range is
// outside it.
bool userdata_file_size(std::string_view path, uint64_t* size);
bool read_userdata_range(std::string_view path, uint64_t offset, void* dst, size_t len);
void write_userdata_file(std::string_view path, std::vector<uint8_t> data);

// Path conversion and normalization helpers.
// Virtual filesystem seam for os/win32_compat (O15): the /userdata overlay as a small tree of
// files and directories with FILETIME timestamps (100 ns ticks since 1601). Keys are the
// normalized "/userdata/..." form produced by normalize_path(). Every function returns a Win32
// error code (ERROR_SUCCESS on success) and is thread-safe. Directories exist explicitly (created
// here) or implicitly (a file lives below them); "/userdata" itself always exists.
struct UserdataEntry {
    std::string name;  // last path component (empty for stat results)
    bool is_directory = false;
    uint64_t size = 0;
    uint64_t created = 0;
    uint64_t accessed = 0;
    uint64_t written = 0;
    uint32_t attributes = 0;  // settable FILE_ATTRIBUTE_* bits stored by userdata_set_attributes
};

uint32_t userdata_stat(std::string_view key, UserdataEntry* out);
uint32_t userdata_create_directory(std::string_view key);
uint32_t userdata_remove_directory(std::string_view key);
uint32_t userdata_delete_file(std::string_view key);
uint32_t userdata_list_directory(std::string_view key, std::vector<UserdataEntry>* out);
uint32_t userdata_rename(std::string_view from, std::string_view to, bool replace_existing);
// Null time pointers are left unchanged.
uint32_t userdata_set_times(std::string_view key,
                            const uint64_t* created,
                            const uint64_t* accessed,
                            const uint64_t* written);
// Stored attribute bits: a non-zero value is returned verbatim (0x80 = NORMAL, none set); 0 means
// the entry was never given explicit attributes.
uint32_t userdata_set_attributes(std::string_view key, uint32_t attributes);
// Sum of all file sizes in the overlay.
uint64_t userdata_total_bytes();

// File handles created by CreateFile*. Returns false (ERROR_INVALID_HANDLE) for other handles.
// userdata_key stays empty for read-only game-data (HTTPFS) files.
struct FileHandleInfo {
    bool is_userdata = false;
    uint32_t desired_access = 0;
    uint64_t cursor = 0;
    std::string userdata_key;
};
bool query_file_handle(Handle handle, FileHandleInfo* out);
// SetEndOfFile: truncates or zero-extends the userdata file to the handle's cursor.
bool set_end_of_file_at_cursor(Handle handle);

std::string utf16_to_utf8(const char16_t* src);
std::string normalize_path(std::string_view raw_path, bool* is_userdata = nullptr);

// ---------------------------------------------------------------------------
// Virtual memory shims
// ---------------------------------------------------------------------------

uint64_t VirtualAlloc(uint64_t address,
                      uint64_t size,
                      uint32_t allocation_type,
                      uint32_t protect);

bool VirtualProtect(uint64_t address,
                    uint64_t size,
                    uint32_t new_protect,
                    uint32_t* old_protect);

bool VirtualFree(uint64_t address, uint64_t size, uint32_t free_type);

size_t VirtualQuery(uint64_t address,
                    MemoryBasicInformation* buffer,
                    size_t length);

// ---------------------------------------------------------------------------
// Threading & TLS shims
// ---------------------------------------------------------------------------

uint32_t TlsAlloc();
bool TlsFree(uint32_t tls_index);
void* TlsGetValue(uint32_t tls_index);
bool TlsSetValue(uint32_t tls_index, void* value);

void InitializeSRWLock(SrwLock* srw_lock);
void AcquireSRWLockExclusive(SrwLock* srw_lock);
void ReleaseSRWLockExclusive(SrwLock* srw_lock);
void AcquireSRWLockShared(SrwLock* srw_lock);
void ReleaseSRWLockShared(SrwLock* srw_lock);
bool TryAcquireSRWLockExclusive(SrwLock* srw_lock);
bool TryAcquireSRWLockShared(SrwLock* srw_lock);

// ---------------------------------------------------------------------------
// Synchronization shims (CriticalSection & Event)
// ---------------------------------------------------------------------------

void InitializeCriticalSection(CriticalSection* critical_section);
void InitializeCriticalSectionWithSpinCount(CriticalSection* critical_section, uint32_t spin_count);
void DeleteCriticalSection(CriticalSection* critical_section);
void EnterCriticalSection(CriticalSection* critical_section);
bool TryEnterCriticalSection(CriticalSection* critical_section);
void LeaveCriticalSection(CriticalSection* critical_section);

// Events and WaitForSingleObject are implemented in os/win32_sync.cpp on the shared dispatcher,
// so events can be combined with semaphores, mutexes and threads in WaitForMultipleObjects.
Handle CreateEventA(void* security_attributes,
                    bool manual_reset,
                    bool initial_state,
                    const char* name);

Handle CreateEventW(void* security_attributes,
                    bool manual_reset,
                    bool initial_state,
                    const char16_t* name);

bool SetEvent(Handle event_handle);
bool ResetEvent(Handle event_handle);
uint32_t WaitForSingleObject(Handle handle, uint32_t timeout_ms);

// ---------------------------------------------------------------------------
// File I/O shims (HttpfsMailboxClient + /userdata overlay)
// ---------------------------------------------------------------------------

Handle CreateFileA(const char* file_name,
                   uint32_t desired_access,
                   uint32_t share_mode,
                   void* security_attributes,
                   uint32_t creation_disposition,
                   uint32_t flags_and_attributes,
                   Handle template_file = kNullHandle);

Handle CreateFileW(const char16_t* file_name,
                   uint32_t desired_access,
                   uint32_t share_mode,
                   void* security_attributes,
                   uint32_t creation_disposition,
                   uint32_t flags_and_attributes,
                   Handle template_file = kNullHandle);

bool ReadFile(Handle file_handle,
              void* buffer,
              uint32_t bytes_to_read,
              uint32_t* bytes_read,
              void* overlapped = nullptr);

bool WriteFile(Handle file_handle,
               const void* buffer,
               uint32_t bytes_to_write,
               uint32_t* bytes_written,
               void* overlapped = nullptr);

bool SetFilePointerEx(Handle file_handle,
                      int64_t distance_to_move,
                      int64_t* new_file_pointer,
                      uint32_t move_method);

bool GetFileSizeEx(Handle file_handle, int64_t* file_size);

bool CloseHandle(Handle handle);

// ---------------------------------------------------------------------------
// Timing & thread-local error state
// ---------------------------------------------------------------------------

bool QueryPerformanceCounter(int64_t* performance_count);
bool QueryPerformanceFrequency(int64_t* frequency);
uint64_t GetTickCount64();
void GetSystemTimeAsFileTime(FileTime* system_time_as_file_time);

// Stall compensation: host work that knowingly blocks guest threads (pipeline builds, capture
// flushes, synchronous loads) must not count against guest watchdogs. Stalls of at least
// kHostStallThresholdNs are subtracted from QueryPerformanceCounter/GetTickCount64 for all
// threads; the guest clock stays monotonic. Shorter stalls are ignored (off unless a stall happens).
constexpr int64_t kHostStallThresholdNs = 50'000'000LL;
void ReportHostStall(int64_t stall_ns);
int64_t TotalHostStallNs();
void ResetHostStallForTesting();

// RAII: times a host-side region with a steady clock and reports it when >= the threshold.
class HostStallScope {
public:
    HostStallScope() : start_(std::chrono::steady_clock::now()) {}
    ~HostStallScope() {
        const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start_)
                               .count();
        if (ns >= kHostStallThresholdNs) {
            ReportHostStall(ns);
        }
    }
    HostStallScope(const HostStallScope&) = delete;
    HostStallScope& operator=(const HostStallScope&) = delete;

private:
    std::chrono::steady_clock::time_point start_;
};

uint32_t GetLastError();
void SetLastError(uint32_t error_code);

// Redirects this host thread's last-error value to `slot` (the running guest thread's TEB
// LastErrorValue, gs:[0x68], see guest/boot.h) so GetLastError/SetLastError and inline guest
// reads of the TEB agree; nullptr restores the host thread_local. Returns the previous slot.
uint32_t* SetThreadLastErrorSlot(uint32_t* slot);

}  // namespace playgta6::os::win32

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints (collision-free with Windows SDK / kernel32.lib)
// ---------------------------------------------------------------------------

extern "C" {

uint64_t pg6_win32_virtual_alloc(uint64_t address,
                                 uint64_t size,
                                 uint32_t allocation_type,
                                 uint32_t protect);
int32_t pg6_win32_virtual_protect(uint64_t address,
                                  uint64_t size,
                                  uint32_t new_protect,
                                  uint32_t* old_protect);
int32_t pg6_win32_virtual_free(uint64_t address, uint64_t size, uint32_t free_type);
size_t pg6_win32_virtual_query(uint64_t address,
                               Pg6MemoryBasicInformation* buffer,
                               size_t length);

uint32_t pg6_win32_tls_alloc(void);
int32_t pg6_win32_tls_free(uint32_t tls_index);
void* pg6_win32_tls_get_value(uint32_t tls_index);
int32_t pg6_win32_tls_set_value(uint32_t tls_index, void* value);

void pg6_win32_initialize_srw_lock(Pg6SrwLock* srw_lock);
void pg6_win32_acquire_srw_lock_exclusive(Pg6SrwLock* srw_lock);
void pg6_win32_release_srw_lock_exclusive(Pg6SrwLock* srw_lock);
void pg6_win32_acquire_srw_lock_shared(Pg6SrwLock* srw_lock);
void pg6_win32_release_srw_lock_shared(Pg6SrwLock* srw_lock);
int32_t pg6_win32_try_acquire_srw_lock_exclusive(Pg6SrwLock* srw_lock);
int32_t pg6_win32_try_acquire_srw_lock_shared(Pg6SrwLock* srw_lock);

void pg6_win32_initialize_critical_section(Pg6CriticalSection* critical_section);
void pg6_win32_delete_critical_section(Pg6CriticalSection* critical_section);
void pg6_win32_enter_critical_section(Pg6CriticalSection* critical_section);
int32_t pg6_win32_try_enter_critical_section(Pg6CriticalSection* critical_section);
void pg6_win32_leave_critical_section(Pg6CriticalSection* critical_section);

Pg6Handle pg6_win32_create_event_a(void* security_attributes,
                                   int32_t manual_reset,
                                   int32_t initial_state,
                                   const char* name);
Pg6Handle pg6_win32_create_event_w(void* security_attributes,
                                   int32_t manual_reset,
                                   int32_t initial_state,
                                   const char16_t* name);
int32_t pg6_win32_set_event(Pg6Handle event_handle);
int32_t pg6_win32_reset_event(Pg6Handle event_handle);
uint32_t pg6_win32_wait_for_single_object(Pg6Handle handle, uint32_t timeout_ms);

Pg6Handle pg6_win32_create_file_a(const char* file_name,
                                  uint32_t desired_access,
                                  uint32_t share_mode,
                                  void* security_attributes,
                                  uint32_t creation_disposition,
                                  uint32_t flags_and_attributes,
                                  Pg6Handle template_file);
Pg6Handle pg6_win32_create_file_w(const char16_t* file_name,
                                  uint32_t desired_access,
                                  uint32_t share_mode,
                                  void* security_attributes,
                                  uint32_t creation_disposition,
                                  uint32_t flags_and_attributes,
                                  Pg6Handle template_file);
// CREATEFILE2_EXTENDED_PARAMETERS.
struct Pg6CreateFile2Params {
    uint32_t size;
    uint32_t file_attributes;
    uint32_t file_flags;
    uint32_t security_qos_flags;
    void* security_attributes;
    Pg6Handle template_file;
};
static_assert(sizeof(Pg6CreateFile2Params) == 32, "CREATEFILE2_EXTENDED_PARAMETERS is 32 bytes");
// CreateFile2: CreateFileW with the attributes, flags and QoS flags taken from `params` (optional).
Pg6Handle pg6_win32_create_file_2(const char16_t* file_name,
                                  uint32_t desired_access,
                                  uint32_t share_mode,
                                  uint32_t creation_disposition,
                                  const Pg6CreateFile2Params* params);
int32_t pg6_win32_read_file(Pg6Handle file_handle,
                            void* buffer,
                            uint32_t bytes_to_read,
                            uint32_t* bytes_read,
                            void* overlapped);
int32_t pg6_win32_write_file(Pg6Handle file_handle,
                             const void* buffer,
                             uint32_t bytes_to_write,
                             uint32_t* bytes_written,
                             void* overlapped);
int32_t pg6_win32_set_file_pointer_ex(Pg6Handle file_handle,
                                      int64_t distance_to_move,
                                      int64_t* new_file_pointer,
                                      uint32_t move_method);
int32_t pg6_win32_get_file_size_ex(Pg6Handle file_handle, int64_t* file_size);
int32_t pg6_win32_close_handle(Pg6Handle handle);

int32_t pg6_win32_query_performance_counter(int64_t* performance_count);
int32_t pg6_win32_query_performance_frequency(int64_t* frequency);
uint64_t pg6_win32_get_tick_count64(void);
void pg6_win32_get_system_time_as_file_time(Pg6FileTime* system_time_as_file_time);
uint32_t pg6_win32_get_last_error(void);
void pg6_win32_set_last_error(uint32_t error_code);

}  // extern "C"
