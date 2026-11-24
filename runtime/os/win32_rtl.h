#pragma once

// Win32 shims required by the recompiled runtime DLLs (task O26): the imports of
// vcruntime140 / vcruntime140_1 / msvcp140 and the program itself that no other shim module
// covers (fiber-local storage, one-time initialization, interlocked singly linked lists, pointer
// encoding, locale/string classification, a few file calls).
//
// Conventions follow os/win32_kernel.h: C-ABI names are pg6_win32_<snake_case api name>, BOOL is
// int32_t, UTF-16 is char16_t, errors go through the thread-local last error, pointer parameters
// are *host* pointers (the import thunks translate guest VAs). Values that are not host pointers
// even though the Win32 type is a pointer - guest function addresses, FLS values, encoded
// pointers, SLIST entries, thread-pool handles - are uint64_t so the thunks leave them alone.
//
// Guest callbacks
// ---------------
// FlsAlloc destructors, InitOnceExecuteOnce callbacks and thread-pool work callbacks are *guest
// function addresses*. This module never calls guest code itself (it is the OS layer): the
// FLS table stores the destructor as an opaque uint64_t and hands it to a caller-supplied
// invoker; InitOnce runs a caller-supplied body. The guest side lives in guest/guest_rtl.{h,cpp}
// (pg6_gthunk_fls_free, pg6_gthunk_init_once_execute_once, the thread-pool work family and
// pg6_gthunk_rtl_capture_stack_back_trace, which need the guest state).
//
// Fiber-local storage (FLS)
// -------------------------
// The runtime has no fibers, so FLS is thread-local storage with a per-slot destructor. Storage
// is host-side and keyed by host thread (one guest thread == one host thread); nothing is kept
// in the guest TEB, so guest code reading TEB.FlsData sees null and cannot be confused with the
// TlsAlloc table (separate index space, TEB.TlsSlots is untouched). Slot 0 is never returned
// (some code uses 0 as "not allocated"); at most kFlsMaxSlots - 1 slots exist; the lowest free
// slot is reused first. A slot's value is 0 in every thread right after FlsAlloc.
// Destructors: run with the non-null value (the slot is cleared first) when the thread exits
// (fls_run_thread_exit, called by the guest thread-exit paths: lazily created guest threads,
// CreateThread / thread-pool threads, and the CRT exit funnel for the calling thread) and, for
// every thread's non-null value, when FlsFree frees the slot (on the calling thread, as the CRT
// expects). Values set by a destructor are processed again (up to 4 passes). A thread that exits
// without the guest hook never runs destructors but its storage is reclaimed.
//
// One-time initialization
// -----------------------
// INIT_ONCE is 8 bytes (INIT_ONCE_STATIC_INIT == 0). State lives in those 8 bytes: 0 =
// uninitialized, 1 = initializer running, (context | 2) = completed (context low two bits are
// reserved and must be zero, as on Windows). Other threads wait on a process-wide condition
// variable while the state is 1. A failed (or abandoned) initializer resets the state to 0 so the
// next caller retries.
//
// Interlocked SLIST
// -----------------
// SLIST_HEADER (16 bytes, 16-aligned) layout used by all four operations (zero == empty, which
// is what InitializeSListHead writes): bytes 0..7 first entry (guest address, 0 if empty);
// bytes 8..15: bit 63 = spin lock held, bits 16..47 = operation sequence number, bits 0..15 =
// depth. Operations take the spin lock (host-side atomic on the guest header), so they are
// mutually atomic but not lock-free. SLIST_ENTRY.Next (8 bytes at the entry) holds the guest
// address of the next entry; entries must be 16-aligned guest addresses in the VirtualMemoryArena.
//
// Pointer encoding: Encode(p) = rotr64(p ^ cookie, cookie & 63) with a fixed per-process cookie
// (deterministic across runs). EncodePointer(NULL) is not NULL (as on Windows) and
// DecodePointer(EncodePointer(p)) == p for every p including NULL.

#include <cstddef>
#include <cstdint>
#include <functional>

#include "os/win32_kernel.h"

// INIT_ONCE (8 bytes).
struct Pg6InitOnce {
    uint64_t ptr = 0;
};
static_assert(sizeof(Pg6InitOnce) == 8, "INIT_ONCE is pointer-sized on x64");

// BY_HANDLE_FILE_INFORMATION (52 bytes).
struct Pg6ByHandleFileInformation {
    uint32_t file_attributes;
    Pg6FileTime creation_time;
    Pg6FileTime last_access_time;
    Pg6FileTime last_write_time;
    uint32_t volume_serial_number;
    uint32_t file_size_high;
    uint32_t file_size_low;
    uint32_t number_of_links;
    uint32_t file_index_high;
    uint32_t file_index_low;
};
static_assert(sizeof(Pg6ByHandleFileInformation) == 52, "BY_HANDLE_FILE_INFORMATION is 52 bytes");
static_assert(offsetof(Pg6ByHandleFileInformation, creation_time) == 4, "ftCreationTime");
static_assert(offsetof(Pg6ByHandleFileInformation, volume_serial_number) == 28, "dwVolumeSerialNumber");
static_assert(offsetof(Pg6ByHandleFileInformation, file_size_high) == 32, "nFileSizeHigh");
static_assert(offsetof(Pg6ByHandleFileInformation, number_of_links) == 40, "nNumberOfLinks");
static_assert(offsetof(Pg6ByHandleFileInformation, file_index_low) == 48, "nFileIndexLow");

// CPINFO (20 bytes).
struct Pg6CpInfo {
    uint32_t max_char_size;
    uint8_t default_char[2];
    uint8_t lead_byte[12];
    uint8_t reserved[2];
};
static_assert(sizeof(Pg6CpInfo) == 20, "CPINFO is 20 bytes");
static_assert(offsetof(Pg6CpInfo, default_char) == 4, "DefaultChar");
static_assert(offsetof(Pg6CpInfo, lead_byte) == 6, "LeadByte");

// SLIST_HEADER (16 bytes).
struct alignas(16) Pg6SListHeader {
    uint64_t first_entry;
    uint64_t depth_sequence;
};
static_assert(sizeof(Pg6SListHeader) == 16, "SLIST_HEADER is 16 bytes on x64");

// PINIT_ONCE_FN host form: BOOL (*)(PINIT_ONCE, PVOID, PVOID*). Registered export only; guest
// code reaches InitOnceExecuteOnce through pg6_gthunk_init_once_execute_once.
using Pg6InitOnceFn = int32_t (*)(Pg6InitOnce* once, void* parameter, void** context);

namespace playgta6::os::win32 {

using InitOnce = Pg6InitOnce;
using ByHandleFileInformation = Pg6ByHandleFileInformation;
using CpInfo = Pg6CpInfo;
using SListHeader = Pg6SListHeader;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr uint32_t FLS_OUT_OF_INDEXES = 0xFFFFFFFFu;
inline constexpr uint32_t kFlsMaxSlots = 4096;  // index 0 is reserved

inline constexpr uint32_t GET_MODULE_HANDLE_EX_FLAG_PIN = 0x1u;
inline constexpr uint32_t GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT = 0x2u;
inline constexpr uint32_t GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS = 0x4u;

inline constexpr uint32_t CT_CTYPE1 = 0x1u;
inline constexpr uint32_t CT_CTYPE2 = 0x2u;
inline constexpr uint32_t CT_CTYPE3 = 0x4u;

inline constexpr uint32_t LCMAP_LOWERCASE = 0x00000100u;
inline constexpr uint32_t LCMAP_UPPERCASE = 0x00000200u;
inline constexpr uint32_t LCMAP_SORTKEY = 0x00000400u;
inline constexpr uint32_t LCMAP_BYTEREV = 0x00000800u;
inline constexpr uint32_t LOCALE_RETURN_NUMBER = 0x20000000u;
inline constexpr uint32_t LOCALE_SNAME = 0x0000005Cu;

inline constexpr int32_t FindExInfoStandard = 0;
inline constexpr int32_t FindExInfoBasic = 1;
inline constexpr int32_t FindExSearchNameMatch = 0;
inline constexpr int32_t FindExSearchLimitToDirectories = 1;
inline constexpr int32_t FindExSearchLimitToDevices = 2;

inline constexpr uint32_t COPY_FILE_FAIL_IF_EXISTS = 0x00000001u;
inline constexpr uint32_t SYMBOLIC_LINK_FLAG_DIRECTORY = 0x1u;
inline constexpr uint32_t SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE = 0x2u;

inline constexpr uint32_t ERROR_PRIVILEGE_NOT_HELD = 1314u;

// HRESULT_FROM_WIN32.
inline constexpr int32_t hresult_from_win32(uint32_t error) {
    return error == 0 ? 0 : static_cast<int32_t>((error & 0xFFFFu) | 0x80070000u);
}

// ---------------------------------------------------------------------------
// FLS (see the overview above). Values and destructors are guest values (uint64_t).
// ---------------------------------------------------------------------------

// Called once per non-null value of a slot with a destructor; `ctx` is the caller's pointer.
// Returns false to stop running further destructors (a guest unwind is pending).
using FlsCallbackInvoker = bool (*)(void* ctx, uint64_t callback, uint64_t value);

// Returns FLS_OUT_OF_INDEXES (ERROR_NOT_ENOUGH_MEMORY) when every slot is in use.
uint32_t FlsAlloc(uint64_t callback);
// With an invoker the destructor runs for every thread's non-null value on the calling thread;
// without one the values are dropped. ERROR_INVALID_PARAMETER for a bad/unallocated index.
bool FlsFree(uint32_t index, FlsCallbackInvoker invoker = nullptr, void* ctx = nullptr);
// Sets last error to ERROR_SUCCESS (like Windows) or ERROR_INVALID_PARAMETER (returns 0).
uint64_t FlsGetValue(uint32_t index);
bool FlsSetValue(uint32_t index, uint64_t value);
// Runs the calling thread's destructors and releases its storage. No-op if it never set a value.
void fls_run_thread_exit(FlsCallbackInvoker invoker, void* ctx);
// Marks the calling thread as one whose exit path calls fls_run_thread_exit (the thread-local
// storage is then only released there). Used by the guest thread bookkeeping.
void fls_mark_thread_hooked();
uint32_t fls_allocated_count();
// Tests: forget every slot and every thread's values (no destructors run).
void fls_reset_for_tests();

// ---------------------------------------------------------------------------
// One-time initialization
// ---------------------------------------------------------------------------

enum class InitOnceOutcome { Completed, Failed, Abandoned };
// Fills *context (low two bits must be zero) on Completed.
using InitOnceBody = std::function<InitOnceOutcome(uint64_t* context)>;
// Exactly-once core shared by the host export and the guest thunk. Returns TRUE if the once is
// (now) complete; *context_out (optional) receives the stored context.
bool InitOnceRun(InitOnce* once, const InitOnceBody& body, uint64_t* context_out);
bool InitOnceExecuteOnce(InitOnce* once, Pg6InitOnceFn fn, void* parameter, void** context);

// ---------------------------------------------------------------------------
// Pointer encoding, SLIST, threads, modules
// ---------------------------------------------------------------------------

uint64_t EncodePointer(uint64_t pointer);
uint64_t DecodePointer(uint64_t pointer);

uint64_t InterlockedPushEntrySList(SListHeader* head, uint64_t entry);
uint64_t InterlockedPopEntrySList(SListHeader* head);
uint64_t InterlockedFlushSList(SListHeader* head);
uint16_t QueryDepthSList(const SListHeader* head);

bool SwitchToThread();

// FROM_ADDRESS: `name_or_address` is a guest address, else a guest pointer to a UTF-16 name
// (0 = the main module). *module is zeroed first.
bool GetModuleHandleExW(uint32_t flags, uint64_t name_or_address, Handle* module);

// Number of FreeLibraryWhenCallbackReturns requests recorded (guest modules are never unloaded).
uint64_t free_library_when_callback_returns_count();

}  // namespace playgta6::os::win32

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints
// ---------------------------------------------------------------------------

extern "C" {

// ----- synchronization / one-time init / FLS / misc ------------------------------------------
int32_t pg6_win32_initialize_critical_section_ex(Pg6CriticalSection* critical_section,
                                                 uint32_t spin_count,
                                                 uint32_t flags);
uint32_t pg6_win32_fls_alloc(uint64_t callback);
int32_t pg6_win32_fls_free(uint32_t fls_index);
uint64_t pg6_win32_fls_get_value(uint32_t fls_index);
int32_t pg6_win32_fls_set_value(uint32_t fls_index, uint64_t fls_data);
uint64_t pg6_win32_encode_pointer(uint64_t pointer);
uint64_t pg6_win32_decode_pointer(uint64_t pointer);
int32_t pg6_win32_init_once_execute_once(Pg6InitOnce* init_once,
                                         Pg6InitOnceFn init_fn,
                                         void* parameter,
                                         void** context);
uint64_t pg6_win32_interlocked_push_entry_slist(Pg6SListHeader* list_head, uint64_t list_entry);
uint64_t pg6_win32_interlocked_pop_entry_slist(Pg6SListHeader* list_head);
uint64_t pg6_win32_interlocked_flush_slist(Pg6SListHeader* list_head);
uint16_t pg6_win32_query_depth_slist(const Pg6SListHeader* list_head);
int32_t pg6_win32_switch_to_thread(void);
int32_t pg6_win32_get_module_handle_ex_w(uint32_t flags,
                                         uint64_t module_name_or_address,
                                         Pg6Handle* module);
// Host form: no guest state, so no frames can be captured (returns 0). Guest code binds
// pg6_gthunk_rtl_capture_stack_back_trace, which walks the guest stack.
uint16_t pg6_win32_rtl_capture_stack_back_trace(uint32_t frames_to_skip,
                                                uint32_t frames_to_capture,
                                                void** back_trace,
                                                uint32_t* back_trace_hash);
void pg6_win32_free_library_when_callback_returns(uint64_t instance, Pg6Handle module);

// ----- locale / strings -----------------------------------------------------------------------
int32_t pg6_win32_lc_map_string_ex(const char16_t* locale_name,
                                   uint32_t map_flags,
                                   const char16_t* src_str,
                                   int32_t cch_src,
                                   char16_t* dest_str,
                                   int32_t cch_dest,
                                   void* version_information,
                                   void* reserved,
                                   uint64_t sort_handle);
int32_t pg6_win32_compare_string_ex(const char16_t* locale_name,
                                    uint32_t cmp_flags,
                                    const char16_t* string1,
                                    int32_t cch_count1,
                                    const char16_t* string2,
                                    int32_t cch_count2,
                                    void* version_information,
                                    void* reserved,
                                    int64_t lparam);
int32_t pg6_win32_get_locale_info_ex(const char16_t* locale_name,
                                     uint32_t lc_type,
                                     char16_t* lc_data,
                                     int32_t cch_data);
int32_t pg6_win32_get_string_type_w(uint32_t info_type,
                                    const char16_t* src_str,
                                    int32_t cch_src,
                                    uint16_t* char_type);
int32_t pg6_win32_get_cp_info(uint32_t code_page, Pg6CpInfo* cp_info);

// ----- files -----------------------------------------------------------------------------------
Pg6Handle pg6_win32_find_first_file_ex_w(const char16_t* file_name,
                                         int32_t info_level_id,
                                         void* find_file_data,
                                         int32_t search_op,
                                         void* search_filter,
                                         uint32_t additional_flags);
int32_t pg6_win32_get_file_information_by_handle(Pg6Handle file,
                                                 Pg6ByHandleFileInformation* file_information);
uint32_t pg6_win32_get_temp_path_w(uint32_t buffer_length, char16_t* buffer);
int32_t pg6_win32_copy_file2(const char16_t* existing_file_name,
                             const char16_t* new_file_name,
                             void* extended_parameters);
int32_t pg6_win32_create_hard_link_w(const char16_t* file_name,
                                     const char16_t* existing_file_name,
                                     void* security_attributes);
uint8_t pg6_win32_create_symbolic_link_w(const char16_t* symlink_file_name,
                                         const char16_t* target_file_name,
                                         uint32_t flags);

}  // extern "C"
