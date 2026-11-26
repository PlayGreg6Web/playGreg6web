#pragma once

// Remaining Win32 core surface (task O15): the api-ms-win-core-* API sets that O5/O6/O7 did not
// cover (file system, process/thread, system info, time, locale, strings, debug, error handling,
// COM helpers), the ADVAPI32 registry reads, bcrypt (RNG only), and a minimal USER32 window fake.
//
// Conventions follow os/win32_kernel.h: C-ABI names are pg6_win32_<snake_case api name>, BOOL is
// int32_t (1/0), HANDLE is Pg6Handle, UTF-16 is char16_t, errors are reported through the
// thread-local last error. Pointer parameters are *host* pointers (the import thunks translate
// guest VAs); a parameter that is a *guest address that must stay a guest address* (the heap model
// of os/win32_loader.h: HeapAlloc results, LocalFree/CoTaskMemFree arguments, function addresses
// handed to SetUnhandledExceptionFilter) is a uint64_t. Signatures use only integer / pointer
// types the R5 import-bridge scanner classifies, so every shim can bind.
//
// File system: directories, timestamps and attributes extend the in-memory /userdata overlay of
// os/win32_kernel.cpp (see the "Virtual filesystem seam" there); a file created through
// CreateFileW/WriteFile is visible to GetFileAttributesW/FindFirstFileW and vice versa. The
// read-only game data (HTTPFS) can be probed by name (GetFileAttributes*, single-file
// FindFirstFileW) but not enumerated, because the HTTPFS protocol has no directory listing.
//
// Not implemented here (other tasks): exception dispatch and RtlCaptureContext/unwinding (R11),
// callbacks into guest code such as window procedures and thread start routines (O16 trampolines),
// COM interfaces (O16), the CRT (O13), Xbox middleware (O14).

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "os/win32_kernel.h"

// ---------------------------------------------------------------------------
// Guest-visible struct layouts (Win64)
// ---------------------------------------------------------------------------

// WIN32_FIND_DATAW.
struct Pg6Win32FindDataW {
    uint32_t file_attributes;
    Pg6FileTime creation_time;
    Pg6FileTime last_access_time;
    Pg6FileTime last_write_time;
    uint32_t file_size_high;
    uint32_t file_size_low;
    uint32_t reserved0;
    uint32_t reserved1;
    char16_t file_name[260];
    char16_t alternate_file_name[14];
};
static_assert(sizeof(Pg6Win32FindDataW) == 592, "WIN32_FIND_DATAW is 592 bytes");
static_assert(offsetof(Pg6Win32FindDataW, creation_time) == 4, "ftCreationTime offset");
static_assert(offsetof(Pg6Win32FindDataW, last_write_time) == 20, "ftLastWriteTime offset");
static_assert(offsetof(Pg6Win32FindDataW, file_size_high) == 28, "nFileSizeHigh offset");
static_assert(offsetof(Pg6Win32FindDataW, file_name) == 44, "cFileName offset");
static_assert(offsetof(Pg6Win32FindDataW, alternate_file_name) == 564, "cAlternateFileName offset");

// WIN32_FILE_ATTRIBUTE_DATA.
struct Pg6Win32FileAttributeData {
    uint32_t file_attributes;
    Pg6FileTime creation_time;
    Pg6FileTime last_access_time;
    Pg6FileTime last_write_time;
    uint32_t file_size_high;
    uint32_t file_size_low;
};
static_assert(sizeof(Pg6Win32FileAttributeData) == 36, "WIN32_FILE_ATTRIBUTE_DATA is 36 bytes");

// SYSTEMTIME.
struct Pg6SystemTime {
    uint16_t year;
    uint16_t month;
    uint16_t day_of_week;
    uint16_t day;
    uint16_t hour;
    uint16_t minute;
    uint16_t second;
    uint16_t milliseconds;
};
static_assert(sizeof(Pg6SystemTime) == 16, "SYSTEMTIME is 16 bytes");

// TIME_ZONE_INFORMATION.
struct Pg6TimeZoneInformation {
    int32_t bias;
    char16_t standard_name[32];
    Pg6SystemTime standard_date;
    int32_t standard_bias;
    char16_t daylight_name[32];
    Pg6SystemTime daylight_date;
    int32_t daylight_bias;
};
static_assert(sizeof(Pg6TimeZoneInformation) == 172, "TIME_ZONE_INFORMATION is 172 bytes");

// SYSTEM_INFO (x64).
struct Pg6SystemInfo {
    uint16_t processor_architecture;
    uint16_t reserved;
    uint32_t page_size;
    uint64_t minimum_application_address;
    uint64_t maximum_application_address;
    uint64_t active_processor_mask;
    uint32_t number_of_processors;
    uint32_t processor_type;
    uint32_t allocation_granularity;
    uint16_t processor_level;
    uint16_t processor_revision;
};
static_assert(sizeof(Pg6SystemInfo) == 48, "SYSTEM_INFO is 48 bytes on x64");
static_assert(offsetof(Pg6SystemInfo, minimum_application_address) == 8, "lpMinimumApplicationAddress");
static_assert(offsetof(Pg6SystemInfo, number_of_processors) == 32, "dwNumberOfProcessors");
static_assert(offsetof(Pg6SystemInfo, allocation_granularity) == 40, "dwAllocationGranularity");

// MEMORYSTATUSEX.
struct Pg6MemoryStatusEx {
    uint32_t length;
    uint32_t memory_load;
    uint64_t total_phys;
    uint64_t avail_phys;
    uint64_t total_page_file;
    uint64_t avail_page_file;
    uint64_t total_virtual;
    uint64_t avail_virtual;
    uint64_t avail_extended_virtual;
};
static_assert(sizeof(Pg6MemoryStatusEx) == 64, "MEMORYSTATUSEX is 64 bytes");

// SYSTEM_LOGICAL_PROCESSOR_INFORMATION (x64): mask, relationship, 16-byte union.
struct Pg6LogicalProcessorInformation {
    uint64_t processor_mask;
    uint32_t relationship;
    uint32_t reserved;
    uint64_t data[2];
};
static_assert(sizeof(Pg6LogicalProcessorInformation) == 32, "SYSTEM_LOGICAL_PROCESSOR_INFORMATION");

// STARTUPINFOW (x64).
struct Pg6StartupInfoW {
    uint32_t cb;
    uint32_t reserved_pad0;
    char16_t* reserved;
    char16_t* desktop;
    char16_t* title;
    uint32_t x;
    uint32_t y;
    uint32_t x_size;
    uint32_t y_size;
    uint32_t x_count_chars;
    uint32_t y_count_chars;
    uint32_t fill_attribute;
    uint32_t flags;
    uint16_t show_window;
    uint16_t cb_reserved2;
    uint32_t reserved_pad1;
    uint8_t* reserved2;
    Pg6Handle std_input;
    Pg6Handle std_output;
    Pg6Handle std_error;
};
static_assert(sizeof(Pg6StartupInfoW) == 104, "STARTUPINFOW is 104 bytes on x64");
static_assert(offsetof(Pg6StartupInfoW, flags) == 60, "dwFlags offset");
static_assert(offsetof(Pg6StartupInfoW, std_input) == 80, "hStdInput offset");

// GROUP_AFFINITY.
struct Pg6GroupAffinity {
    uint64_t mask;
    uint16_t group;
    uint16_t reserved[3];
};
static_assert(sizeof(Pg6GroupAffinity) == 16, "GROUP_AFFINITY is 16 bytes");

// GUID.
struct Pg6Guid {
    uint32_t data1;
    uint16_t data2;
    uint16_t data3;
    uint8_t data4[8];
};
static_assert(sizeof(Pg6Guid) == 16, "GUID is 16 bytes");

// WNDCLASSEXA (x64).
struct Pg6WndClassExA {
    uint32_t size;
    uint32_t style;
    uint64_t window_proc;  // guest function address; never called by the fake
    int32_t class_extra;
    int32_t window_extra;
    Pg6Handle instance;
    Pg6Handle icon;
    Pg6Handle cursor;
    Pg6Handle background;
    const char* menu_name;
    const char* class_name;
    Pg6Handle icon_small;
};
static_assert(sizeof(Pg6WndClassExA) == 80, "WNDCLASSEXA is 80 bytes on x64");
static_assert(offsetof(Pg6WndClassExA, window_proc) == 8, "lpfnWndProc offset");
static_assert(offsetof(Pg6WndClassExA, class_name) == 64, "lpszClassName offset");

// MSG (x64).
struct Pg6Msg {
    Pg6Handle hwnd;
    uint32_t message;
    uint32_t reserved_pad0;
    uint64_t wparam;
    int64_t lparam;
    uint32_t time;
    int32_t pt_x;
    int32_t pt_y;
    uint32_t private_data;
};
static_assert(sizeof(Pg6Msg) == 48, "MSG is 48 bytes on x64");
static_assert(offsetof(Pg6Msg, wparam) == 16, "wParam offset");
static_assert(offsetof(Pg6Msg, time) == 32, "time offset");

namespace playgta6::os::win32 {

using Win32FindDataW = Pg6Win32FindDataW;
using Win32FileAttributeData = Pg6Win32FileAttributeData;
using SystemTime = Pg6SystemTime;
using SystemInfo = Pg6SystemInfo;
using Guid = Pg6Guid;
using Msg = Pg6Msg;

// File attributes (FILE_ATTRIBUTE_NORMAL lives in win32_kernel.h).
inline constexpr uint32_t FILE_ATTRIBUTE_READONLY = 0x00000001u;
inline constexpr uint32_t FILE_ATTRIBUTE_HIDDEN = 0x00000002u;
inline constexpr uint32_t FILE_ATTRIBUTE_SYSTEM = 0x00000004u;
inline constexpr uint32_t FILE_ATTRIBUTE_DIRECTORY = 0x00000010u;
inline constexpr uint32_t FILE_ATTRIBUTE_ARCHIVE = 0x00000020u;
inline constexpr uint32_t FILE_ATTRIBUTE_TEMPORARY = 0x00000100u;
inline constexpr uint32_t INVALID_FILE_ATTRIBUTES = 0xFFFFFFFFu;
inline constexpr int32_t GetFileExInfoStandard = 0;
inline constexpr uint32_t MOVEFILE_REPLACE_EXISTING = 0x1u;
inline constexpr uint32_t MOVEFILE_COPY_ALLOWED = 0x2u;
inline constexpr uint32_t MOVEFILE_WRITE_THROUGH = 0x8u;

// FILETIME (100 ns ticks since 1601-01-01) <-> Unix time.
inline constexpr uint64_t kFileTimeUnixEpochTicks = 116444736000000000ULL;
inline constexpr uint64_t kFileTimeTicksPerSecond = 10000000ULL;
inline uint64_t file_time_ticks(const FileTime& ft) {
    return (static_cast<uint64_t>(ft.high_date_time) << 32) | ft.low_date_time;
}
inline FileTime file_time_from_ticks(uint64_t ticks) {
    FileTime ft;
    ft.low_date_time = static_cast<uint32_t>(ticks & 0xFFFFFFFFull);
    ft.high_date_time = static_cast<uint32_t>(ticks >> 32);
    return ft;
}

// Code pages and MultiByteToWideChar / WideCharToMultiByte flags.
inline constexpr uint32_t CP_ACP = 0u;
inline constexpr uint32_t CP_OEMCP = 1u;
inline constexpr uint32_t CP_MACCP = 2u;
inline constexpr uint32_t CP_THREAD_ACP = 3u;
inline constexpr uint32_t CP_LATIN1 = 28591u;
inline constexpr uint32_t CP_USASCII = 20127u;
inline constexpr uint32_t CP_WINDOWS_1252 = 1252u;
inline constexpr uint32_t CP_UTF8 = 65001u;
inline constexpr uint32_t MB_PRECOMPOSED = 0x1u;
inline constexpr uint32_t MB_COMPOSITE = 0x2u;
inline constexpr uint32_t MB_USEGLYPHCHARS = 0x4u;
inline constexpr uint32_t MB_ERR_INVALID_CHARS = 0x8u;
inline constexpr uint32_t WC_COMPOSITECHECK = 0x200u;
inline constexpr uint32_t WC_DISCARDNS = 0x10u;
inline constexpr uint32_t WC_SEPCHARS = 0x20u;
inline constexpr uint32_t WC_DEFAULTCHAR = 0x40u;
inline constexpr uint32_t WC_ERR_INVALID_CHARS = 0x80u;
inline constexpr uint32_t WC_NO_BEST_FIT_CHARS = 0x400u;

// CompareString results and flags.
inline constexpr int32_t CSTR_LESS_THAN = 1;
inline constexpr int32_t CSTR_EQUAL = 2;
inline constexpr int32_t CSTR_GREATER_THAN = 3;
inline constexpr uint32_t NORM_IGNORECASE = 0x1u;

// FormatMessage flags.
inline constexpr uint32_t FORMAT_MESSAGE_ALLOCATE_BUFFER = 0x100u;
inline constexpr uint32_t FORMAT_MESSAGE_IGNORE_INSERTS = 0x200u;
inline constexpr uint32_t FORMAT_MESSAGE_FROM_STRING = 0x400u;
inline constexpr uint32_t FORMAT_MESSAGE_FROM_SYSTEM = 0x1000u;
inline constexpr uint32_t FORMAT_MESSAGE_MAX_WIDTH_MASK = 0xFFu;

// Registry.
inline constexpr uint32_t REG_NONE = 0u;
inline constexpr uint32_t REG_SZ = 1u;
inline constexpr uint32_t REG_BINARY = 3u;
inline constexpr uint32_t REG_DWORD = 4u;
inline constexpr uint64_t HKEY_CLASSES_ROOT = 0xFFFFFFFF80000000ULL;
inline constexpr uint64_t HKEY_CURRENT_USER = 0xFFFFFFFF80000001ULL;
inline constexpr uint64_t HKEY_LOCAL_MACHINE = 0xFFFFFFFF80000002ULL;
inline constexpr uint64_t HKEY_USERS = 0xFFFFFFFF80000003ULL;

// Pseudo-handles and ids.
inline constexpr Handle kCurrentProcessPseudoHandle = ~uint64_t{0};  // GetCurrentProcess()
inline constexpr uint32_t kCurrentProcessId = 0x1000u;
inline constexpr uint32_t THREAD_PRIORITY_ERROR_RETURN = 0x7FFFFFFFu;

// NTSTATUS values used by the bcrypt shims.
inline constexpr int32_t STATUS_SUCCESS = 0;
inline constexpr int32_t STATUS_NOT_FOUND = static_cast<int32_t>(0xC0000225u);
inline constexpr int32_t STATUS_INVALID_PARAMETER = static_cast<int32_t>(0xC000000Du);
inline constexpr int32_t STATUS_INVALID_HANDLE = static_cast<int32_t>(0xC0000008u);
inline constexpr int32_t STATUS_NOT_SUPPORTED = static_cast<int32_t>(0xC00000BBu);
inline constexpr uint32_t BCRYPT_USE_SYSTEM_PREFERRED_RNG = 0x2u;

// HRESULTs used by the COM helpers.
inline constexpr int32_t S_OK = 0;
inline constexpr int32_t S_FALSE = 1;
inline constexpr int32_t E_POINTER = static_cast<int32_t>(0x80004003u);
inline constexpr int32_t E_INVALIDARG = static_cast<int32_t>(0x80070057u);
inline constexpr int32_t REGDB_E_CLASSNOTREG = static_cast<int32_t>(0x80040154u);
inline constexpr int32_t RPC_E_CHANGED_MODE = static_cast<int32_t>(0x80010106u);
inline constexpr uint32_t COINIT_MULTITHREADED = 0x0u;
inline constexpr uint32_t COINIT_APARTMENTTHREADED = 0x2u;

// Window messages used by the fake.
inline constexpr uint32_t WM_QUIT = 0x0012u;
inline constexpr uint32_t PM_REMOVE = 0x0001u;

// Win32 error codes of this module.
inline constexpr uint32_t ERROR_NO_MORE_FILES = 18u;
inline constexpr uint32_t ERROR_NOT_SUPPORTED = 50u;
inline constexpr uint32_t ERROR_INVALID_NAME = 123u;
inline constexpr uint32_t ERROR_DIR_NOT_EMPTY = 145u;
inline constexpr uint32_t ERROR_MORE_DATA = 234u;
inline constexpr uint32_t ERROR_DIRECTORY = 267u;
inline constexpr uint32_t ERROR_MR_MID_NOT_FOUND = 317u;
inline constexpr uint32_t ERROR_INVALID_FLAGS = 1004u;
inline constexpr uint32_t ERROR_NO_UNICODE_TRANSLATION = 1113u;
inline constexpr uint32_t ERROR_INVALID_WINDOW_HANDLE = 1400u;
inline constexpr uint32_t ERROR_CANNOT_FIND_WND_CLASS = 1407u;
inline constexpr uint32_t ERROR_CLASS_ALREADY_EXISTS = 1410u;

// ---------------------------------------------------------------------------
// Runtime-facing configuration and hooks (not Win32 APIs)
// ---------------------------------------------------------------------------

// TerminateProcess(GetCurrentProcess(), code): there is no process-exit path in the runtime yet
// (a browser worker cannot exit the process; the page owns the lifetime), so the request is
// recorded and the installed handler, if any, is called on the calling thread. The handler is
// expected to stop guest execution (it must not call std::exit from a worker thread). Without a
// handler the call returns TRUE after recording. nullptr clears the handler.
using ProcessExitHandler = void (*)(uint32_t exit_code, void* user);
void set_process_exit_handler(ProcessExitHandler handler, void* user);
bool process_exit_requested(uint32_t* exit_code);  // false until TerminateProcess ran
void reset_process_exit_state();

// Command line returned by GetCommandLineA (default "game.exe").
void set_command_line(std::string_view utf8);

// Stack bounds reported for the calling thread by GetCurrentThreadStackLimits. The guest runtime
// knows the real (emulated) stack; without a call the thread reports a very wide range so
// stack-probe checks pass. (0, 0) restores the default.
void set_current_thread_stack_limits(uint64_t low, uint64_t high);

// Physical memory GlobalMemoryStatusEx reports as total (default 12 GiB, a console game budget).
void set_total_physical_memory(uint64_t bytes);

// SetUnhandledExceptionFilter bookkeeping (dispatch is task R11): the registered guest address.
uint64_t get_unhandled_exception_filter();

// Number of live RegisterAppStateChangeNotification / RegisterAppConstrainedChangeNotification
// registrations (the runtime never calls them back on its own).
size_t app_notification_registration_count();

// Process lifetime management (PLM). Both callbacks are `void CALLBACK(BOOLEAN flag, PVOID context)`:
// flag is `quiesced` for app-state routines and `constrained` for constrained-change routines.
enum class AppNotificationKind : uint8_t { State, Constrained };
struct AppNotification {
    AppNotificationKind kind;
    uint64_t routine;  // guest address
    uint8_t flag;      // BOOLEAN quiesced / constrained
    uint64_t context;
};
// Calls one guest routine; the runtime supplies it (only it can enter guest code).
using AppNotificationInvoker = void (*)(const AppNotification& call, void* user);
size_t app_notification_registration_count(AppNotificationKind kind);
// Suspend in the documented GDK order: every constrained-change routine with TRUE, then every
// app-state routine with quiesced = TRUE (a title's suspend handler can wait on work its constrained
// handler started). Resume: app-state routines with FALSE, then constrained-change routines with
// FALSE. Each group runs in registration order. Returns the number of routines called.
size_t dispatch_app_suspend(AppNotificationInvoker invoker, void* user);
size_t dispatch_app_resume(AppNotificationInvoker invoker, void* user);

// XLaunchNewGame of the running title: the guest stops with this exit code, and the native harness
// (tools/run_native_boot.py) and the browser boot turn it into a relaunch with the same arguments.
// ASCII "RELN" so it cannot collide with a title's own exit codes in practice.
inline constexpr uint32_t kRelaunchSelfExitCode = 0x52454C4Eu;

// Guest restart (XLaunchNewGame of the running title): the new arguments are recorded, the hook
// (the boot installs one that runs the guest's suspend routines) runs, then the process stops with
// kRelaunchSelfExitCode. The host (native main, browser loader) restarts with the same options plus
// these arguments. Cleared by reset_relaunch_request.
using RelaunchPrepareHook = void (*)(void* user);
void set_relaunch_prepare_hook(RelaunchPrepareHook hook, void* user);
// Records `args` and runs the prepare hook (no-op without one).
void begin_guest_relaunch(std::string_view args);
bool relaunch_requested(std::string* args_out);
void reset_relaunch_request();
// The relaunch suspend sequence: constrained-change routines, then app-state routines (suspend order,
// dispatch_app_suspend), then `grace` (sleep for the grace period, ~1 s). Returns routines called.
using RelaunchGraceFn = void (*)(uint32_t grace_ms, void* user);
inline constexpr uint32_t kRelaunchGraceMs = 1000;
size_t run_relaunch_suspend_sequence(AppNotificationInvoker invoker, void* invoker_user,
                                     RelaunchGraceFn grace, void* grace_user,
                                     uint32_t grace_ms = kRelaunchGraceMs);

// In-memory registry (empty by default). Keys are "HKLM\\Software\\Vendor" style paths (root names
// HKCR, HKCU, HKLM, HKU or their long forms; case-insensitive); values are stored per key.
void registry_set_value(std::string_view key_path,
                        std::u16string_view value_name,
                        uint32_t type,
                        const void* data,
                        size_t size);
void registry_clear();

// Window fake: the runtime posts messages that PeekMessageA then returns. hwnd 0 = thread message.
// Returns false for an unknown non-zero hwnd. reset_window_state drops classes, windows, queue.
bool post_window_message(Pg6Handle hwnd, uint32_t message, uint64_t wparam, int64_t lparam);
size_t window_count();
size_t queued_message_count();
uint64_t dispatched_message_count();
void reset_window_state();

}  // namespace playgta6::os::win32

extern "C" {

// ----- api-ms-win-core-file / file-l2 ---------------------------------------------------------
int32_t pg6_win32_create_directory_w(const char16_t* path_name, void* security_attributes);
int32_t pg6_win32_remove_directory_w(const char16_t* path_name);
int32_t pg6_win32_delete_file_w(const char16_t* file_name);
uint32_t pg6_win32_get_file_attributes_a(const char* file_name);
uint32_t pg6_win32_get_file_attributes_w(const char16_t* file_name);
int32_t pg6_win32_get_file_attributes_ex_w(const char16_t* file_name,
                                           int32_t info_level_id,
                                           void* file_information);
int32_t pg6_win32_set_file_attributes_w(const char16_t* file_name, uint32_t file_attributes);
Pg6Handle pg6_win32_find_first_file_w(const char16_t* file_name, Pg6Win32FindDataW* find_data);
int32_t pg6_win32_find_next_file_w(Pg6Handle find_file, Pg6Win32FindDataW* find_data);
int32_t pg6_win32_find_close(Pg6Handle find_file);
int32_t pg6_win32_get_file_time(Pg6Handle file,
                                Pg6FileTime* creation_time,
                                Pg6FileTime* last_access_time,
                                Pg6FileTime* last_write_time);
int32_t pg6_win32_set_file_time(Pg6Handle file,
                                const Pg6FileTime* creation_time,
                                const Pg6FileTime* last_access_time,
                                const Pg6FileTime* last_write_time);
int32_t pg6_win32_file_time_to_local_file_time(const Pg6FileTime* file_time,
                                               Pg6FileTime* local_file_time);
int32_t pg6_win32_local_file_time_to_file_time(const Pg6FileTime* local_file_time,
                                               Pg6FileTime* file_time);
int32_t pg6_win32_flush_file_buffers(Pg6Handle file);
int32_t pg6_win32_set_end_of_file(Pg6Handle file);
int32_t pg6_win32_get_disk_free_space_ex_w(const char16_t* directory_name,
                                           uint64_t* free_bytes_available_to_caller,
                                           uint64_t* total_number_of_bytes,
                                           uint64_t* total_number_of_free_bytes);
int32_t pg6_win32_move_file_ex_w(const char16_t* existing_file_name,
                                 const char16_t* new_file_name,
                                 uint32_t flags);

// ----- processthreads / processtopology / processenvironment / synch leftovers ---------------
Pg6Handle pg6_win32_get_current_process(void);
uint32_t pg6_win32_get_current_process_id(void);
uint32_t pg6_win32_get_process_id(Pg6Handle process);
void pg6_win32_get_startup_info_w(Pg6StartupInfoW* startup_info);
int32_t pg6_win32_get_thread_priority(Pg6Handle thread);
int32_t pg6_win32_set_thread_priority(Pg6Handle thread, int32_t priority);
Pg6Handle pg6_win32_open_thread(uint32_t desired_access, int32_t inherit_handle, uint32_t thread_id);
uint32_t pg6_win32_suspend_thread(Pg6Handle thread);
int32_t pg6_win32_terminate_process(Pg6Handle process, uint32_t exit_code);
int32_t pg6_win32_set_process_priority_boost(Pg6Handle process, int32_t disable_priority_boost);
void pg6_win32_get_current_thread_stack_limits(uint64_t* low_limit, uint64_t* high_limit);
int32_t pg6_win32_is_processor_feature_present(uint32_t processor_feature);
int32_t pg6_win32_get_thread_group_affinity(Pg6Handle thread, Pg6GroupAffinity* group_affinity);
int32_t pg6_win32_set_thread_group_affinity(Pg6Handle thread,
                                            const Pg6GroupAffinity* group_affinity,
                                            Pg6GroupAffinity* previous_group_affinity);
// PROCESSOR_NUMBER.
struct Pg6ProcessorNumber {
    uint16_t group;
    uint8_t number;
    uint8_t reserved;
};
static_assert(sizeof(Pg6ProcessorNumber) == 4, "PROCESSOR_NUMBER is 4 bytes");
// SetThreadIdealProcessorEx: records the ideal processor (a scheduling hint; the host scheduler
// is not told) and reports the previous one (initially group 0, processor 0).
int32_t pg6_win32_set_thread_ideal_processor_ex(Pg6Handle thread,
                                                const Pg6ProcessorNumber* ideal_processor,
                                                Pg6ProcessorNumber* previous_ideal_processor);
char* pg6_win32_get_command_line_a(void);
void pg6_win32_sleep(uint32_t milliseconds);
int32_t pg6_win32_initialize_critical_section_and_spin_count(Pg6CriticalSection* critical_section,
                                                             uint32_t spin_count);
Pg6Handle pg6_win32_create_semaphore_ex_w(void* security_attributes,
                                          int32_t initial_count,
                                          int32_t maximum_count,
                                          const char16_t* name,
                                          uint32_t flags,
                                          uint32_t desired_access);
uint64_t pg6_win32_virtual_query_ex(Pg6Handle process,
                                    uint64_t address,
                                    Pg6MemoryBasicInformation* buffer,
                                    uint64_t length);
uint64_t pg6_win32_local_free(uint64_t memory);
void pg6_win32_initialize_s_list_head(void* list_head);

// ----- sysinfo / datetime / timezone ----------------------------------------------------------
void pg6_win32_get_system_info(Pg6SystemInfo* system_info);
void pg6_win32_get_native_system_info(Pg6SystemInfo* system_info);
void pg6_win32_get_local_time(Pg6SystemTime* system_time);
void pg6_win32_get_system_time(Pg6SystemTime* system_time);
void pg6_win32_get_system_time_precise_as_file_time(Pg6FileTime* system_time_as_file_time);
int32_t pg6_win32_global_memory_status_ex(Pg6MemoryStatusEx* buffer);
int32_t pg6_win32_get_logical_processor_information(void* buffer, uint32_t* returned_length);
int32_t pg6_win32_file_time_to_system_time(const Pg6FileTime* file_time, Pg6SystemTime* system_time);
int32_t pg6_win32_system_time_to_file_time(const Pg6SystemTime* system_time, Pg6FileTime* file_time);
uint32_t pg6_win32_get_time_zone_information(Pg6TimeZoneInformation* time_zone_information);
int32_t pg6_win32_get_date_format_ex(const char16_t* locale_name,
                                     uint32_t flags,
                                     const Pg6SystemTime* date,
                                     const char16_t* format,
                                     char16_t* date_str,
                                     int32_t cch_date,
                                     const char16_t* calendar);
int32_t pg6_win32_get_time_format_ex(const char16_t* locale_name,
                                     uint32_t flags,
                                     const Pg6SystemTime* time,
                                     const char16_t* format,
                                     char16_t* time_str,
                                     int32_t cch_time);

// ----- string / localization ------------------------------------------------------------------
int32_t pg6_win32_multi_byte_to_wide_char(uint32_t code_page,
                                          uint32_t flags,
                                          const char* multi_byte_str,
                                          int32_t cb_multi_byte,
                                          char16_t* wide_char_str,
                                          int32_t cch_wide_char);
int32_t pg6_win32_wide_char_to_multi_byte(uint32_t code_page,
                                          uint32_t flags,
                                          const char16_t* wide_char_str,
                                          int32_t cch_wide_char,
                                          char* multi_byte_str,
                                          int32_t cb_multi_byte,
                                          const char* default_char,
                                          int32_t* used_default_char);
int32_t pg6_win32_lstrcmp_w(const char16_t* string1, const char16_t* string2);
int32_t pg6_win32_lstrcmpi_w(const char16_t* string1, const char16_t* string2);
int32_t pg6_win32_lstrlen_w(const char16_t* string);
int32_t pg6_win32_compare_string_w(uint32_t locale,
                                   uint32_t flags,
                                   const char16_t* string1,
                                   int32_t count1,
                                   const char16_t* string2,
                                   int32_t count2);
int32_t pg6_win32_compare_string_ordinal(const char16_t* string1,
                                         int32_t count1,
                                         const char16_t* string2,
                                         int32_t count2,
                                         int32_t ignore_case);
int32_t pg6_win32_get_user_default_locale_name(char16_t* locale_name, int32_t cch_locale_name);
int32_t pg6_win32_get_user_default_geo_name(char16_t* geo_name, int32_t count);
uint32_t pg6_win32_locale_name_to_lcid(const char16_t* name, uint32_t flags);
uint32_t pg6_win32_format_message_a(uint32_t flags,
                                    const void* source,
                                    uint32_t message_id,
                                    uint32_t language_id,
                                    char* buffer,
                                    uint32_t size,
                                    void* arguments);

// ----- debug / error handling / app lifecycle / WER / named pipes -----------------------------
int32_t pg6_win32_is_debugger_present(void);
void pg6_win32_output_debug_string_a(const char* output_string);
uint64_t pg6_win32_set_unhandled_exception_filter(uint64_t top_level_exception_filter);
int32_t pg6_win32_unhandled_exception_filter(void* exception_info);
uint32_t pg6_win32_register_app_state_change_notification(uint64_t routine,
                                                          uint64_t context,
                                                          uint64_t* registration);
void pg6_win32_unregister_app_state_change_notification(uint64_t registration);
uint32_t pg6_win32_register_app_constrained_change_notification(uint64_t routine,
                                                                uint64_t context,
                                                                uint64_t* registration);
void pg6_win32_unregister_app_constrained_change_notification(uint64_t registration);
int32_t pg6_win32_wer_register_file(const char16_t* file, uint32_t file_type, uint32_t flags);
int32_t pg6_win32_set_named_pipe_handle_state(Pg6Handle named_pipe,
                                              uint32_t* mode,
                                              uint32_t* max_collection_count,
                                              uint32_t* collect_data_timeout);
int32_t pg6_win32_wait_named_pipe_w(const char16_t* named_pipe_name, uint32_t timeout);

// ----- COM helpers (no interfaces; CoCreateInstance fails with REGDB_E_CLASSNOTREG) -----------
int32_t pg6_win32_co_initialize_ex(void* reserved, uint32_t co_init);
void pg6_win32_co_uninitialize(void);
int32_t pg6_win32_co_create_instance(const Pg6Guid* clsid,
                                     void* outer,
                                     uint32_t cls_context,
                                     const Pg6Guid* iid,
                                     void** object);
uint64_t pg6_win32_co_task_mem_alloc(uint64_t size);
void pg6_win32_co_task_mem_free(uint64_t memory);
int32_t pg6_win32_co_create_guid(Pg6Guid* guid);
int32_t pg6_win32_string_from_guid2(const Pg6Guid* guid, char16_t* buffer, int32_t cch_max);
int32_t pg6_win32_prop_variant_clear(void* prop_variant);

// ----- ADVAPI32 registry (in-memory, empty by default) ----------------------------------------
int32_t pg6_win32_reg_open_key_w(Pg6Handle key, const char16_t* sub_key, Pg6Handle* result);
int32_t pg6_win32_reg_query_value_ex_w(Pg6Handle key,
                                       const char16_t* value_name,
                                       uint32_t* reserved,
                                       uint32_t* type,
                                       uint8_t* data,
                                       uint32_t* data_size);
int32_t pg6_win32_reg_close_key(Pg6Handle key);

// ----- bcrypt (RNG only) ----------------------------------------------------------------------
int32_t pg6_win32_bcrypt_open_algorithm_provider(Pg6Handle* algorithm,
                                                 const char16_t* algorithm_id,
                                                 const char16_t* implementation,
                                                 uint32_t flags);
int32_t pg6_win32_bcrypt_gen_random(Pg6Handle algorithm,
                                    uint8_t* buffer,
                                    uint32_t byte_count,
                                    uint32_t flags);
int32_t pg6_win32_bcrypt_close_algorithm_provider(Pg6Handle algorithm, uint32_t flags);

// ----- USER32 window fake (ext-ms-win-rtcore-ntuser-*) ----------------------------------------
uint16_t pg6_win32_register_class_ex_a(const Pg6WndClassExA* window_class);
Pg6Handle pg6_win32_create_window_ex_a(uint32_t ex_style,
                                       const char* class_name,
                                       const char* window_name,
                                       uint32_t style,
                                       int32_t x,
                                       int32_t y,
                                       int32_t width,
                                       int32_t height,
                                       Pg6Handle parent,
                                       Pg6Handle menu,
                                       Pg6Handle instance,
                                       void* parameter);
int64_t pg6_win32_def_window_proc_a(Pg6Handle hwnd, uint32_t message, uint64_t wparam, int64_t lparam);
int32_t pg6_win32_show_window(Pg6Handle hwnd, int32_t cmd_show);
int32_t pg6_win32_peek_message_a(Pg6Msg* msg,
                                 Pg6Handle hwnd,
                                 uint32_t filter_min,
                                 uint32_t filter_max,
                                 uint32_t remove_message);
int32_t pg6_win32_translate_message(const Pg6Msg* msg);
int64_t pg6_win32_dispatch_message_a(const Pg6Msg* msg);

}  // extern "C"
