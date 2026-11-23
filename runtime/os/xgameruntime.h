#pragma once

#include <cstddef>
#include <cstdint>

// Opaque C-ABI handles (prefixed with Pg6* to avoid collisions with Xbox GDK SDK headers).
struct Pg6XTaskQueue;
struct Pg6XUser;
struct Pg6XAsyncBlock;

using Pg6HResult = int32_t;
using Pg6XTaskQueueHandle = Pg6XTaskQueue*;
using Pg6XUserHandle = Pg6XUser*;
using Pg6XTaskQueueCallback = void (*)(void* context, bool canceled);
using Pg6XTaskQueueTerminatedCallback = void (*)(void* context);
using Pg6XAsyncCompletionRoutine = void (*)(Pg6XAsyncBlock* async_block);

struct Pg6XAsyncBlock {
    Pg6XTaskQueueHandle queue = nullptr;
    void* context = nullptr;
    Pg6XAsyncCompletionRoutine callback = nullptr;
    uint8_t internal[32]{};
};

struct Pg6XVersion {
    uint16_t major = 0;
    uint16_t minor = 0;
    uint16_t build = 0;
    uint16_t revision = 0;
};

struct Pg6XSystemAnalyticsInfo {
    Pg6XVersion os_version{};
    Pg6XVersion hosting_os_version{};
    char family[64]{};
    char form[64]{};
};

namespace playgta6::os::gdk {

using HResult = Pg6HResult;
using XTaskQueueHandle = Pg6XTaskQueueHandle;
using XUserHandle = Pg6XUserHandle;
using XTaskQueueCallback = Pg6XTaskQueueCallback;
using XTaskQueueTerminatedCallback = Pg6XTaskQueueTerminatedCallback;
using XAsyncBlock = Pg6XAsyncBlock;
using XAsyncCompletionRoutine = Pg6XAsyncCompletionRoutine;
using XVersion = Pg6XVersion;
using XSystemAnalyticsInfo = Pg6XSystemAnalyticsInfo;

// Standard GDK / COM HRESULT codes.
inline constexpr HResult S_OK_HR = 0;
inline constexpr HResult E_FAIL_HR = static_cast<HResult>(0x80004005u);
inline constexpr HResult E_POINTER_HR = static_cast<HResult>(0x80004003u);
inline constexpr HResult E_ABORT_HR = static_cast<HResult>(0x80004004u);
inline constexpr HResult E_HANDLE_HR = static_cast<HResult>(0x80070006u);
inline constexpr HResult E_OUTOFMEMORY_HR = static_cast<HResult>(0x8007000Eu);
inline constexpr HResult E_INVALIDARG_HR = static_cast<HResult>(0x80070057u);
inline constexpr HResult E_NOT_SUFFICIENT_BUFFER_HR = static_cast<HResult>(0x8007007Au);
inline constexpr HResult E_PENDING_HR = static_cast<HResult>(0x8000000Au);
inline constexpr HResult E_GAMERUNTIME_NOT_INITIALIZED = static_cast<HResult>(0x89240100u);

inline constexpr bool succeeded(HResult hr) {
    return hr >= 0;
}

inline constexpr bool failed(HResult hr) {
    return hr < 0;
}

// Deterministic offline local profile constants (zero network/DRM dependencies).
inline constexpr uint64_t kOfflineUserXuid = 0x0009000000000001ULL;
inline constexpr const char* kOfflineUserGamertag = "OfflinePlayer";

// XTaskQueue enums matching Xbox GDK XTaskQueue.h (Manual is 0 there; guest code passes the raw
// values, e.g. through the xgameruntime.dll provider).
enum class XTaskQueueDispatchMode : uint32_t {
    Manual = 0,
    ThreadPool = 1,
    SerializedThreadPool = 2,
    Immediate = 3,
};

enum class XTaskQueuePort : uint32_t {
    Work = 0,
    Completion = 1,
};

// XUser enums matching Xbox GDK XUser.h.
enum class XUserAddOptions : uint32_t {
    None = 0x00u,
    AddDefaultUserSilently = 0x01u,
    AllowGuests = 0x02u,
    AddDefaultUserAllowingUI = 0x04u,
};

enum class XUserGamertagComponent : uint32_t {
    Classic = 0,
    Modern = 1,
    ModernSuffix = 2,
    UniqueModern = 3,
};

// XSystem device types matching Xbox GDK XSystem.h.
enum class XSystemDeviceType : uint32_t {
    Unknown = 0x00u,
    Pc = 0x01u,
    XboxOne = 0x02u,
    XboxOneS = 0x03u,
    XboxOneX = 0x04u,
    XboxOneXDevkit = 0x05u,
    XboxScarlettLockhart = 0x06u,  // Xbox Series S
    XboxScarlettAnaconda = 0x07u,  // Xbox Series X
    XboxScarlettDevkit = 0x08u,
};

// ---------------------------------------------------------------------------
// Game Runtime lifecycle
// ---------------------------------------------------------------------------

HResult XGameRuntimeInitialize();
void XGameRuntimeUninitialize();
bool XGameRuntimeIsInitialized();

// ---------------------------------------------------------------------------
// XTaskQueue C-ABI handle API (backed by playgta6::os::XTaskQueue)
// ---------------------------------------------------------------------------

HResult XTaskQueueCreate(XTaskQueueDispatchMode work_mode,
                         XTaskQueueDispatchMode completion_mode,
                         XTaskQueueHandle* queue);

void XTaskQueueCloseHandle(XTaskQueueHandle queue);

// Adds a reference: *duplicated_handle is the same queue and needs its own XTaskQueueCloseHandle.
HResult XTaskQueueDuplicateHandle(XTaskQueueHandle queue, XTaskQueueHandle* duplicated_handle);

HResult XTaskQueueSubmitCallback(XTaskQueueHandle queue,
                                 XTaskQueuePort port,
                                 void* callback_context,
                                 XTaskQueueCallback callback);

bool XTaskQueueDispatch(XTaskQueueHandle queue,
                        XTaskQueuePort port,
                        uint32_t timeout_ms = 0);

HResult XTaskQueueTerminate(XTaskQueueHandle queue,
                            bool wait,
                            void* callback_context = nullptr,
                            XTaskQueueTerminatedCallback callback = nullptr);

bool XTaskQueueGetCurrentProcessTaskQueue(XTaskQueueHandle* queue);

// ---------------------------------------------------------------------------
// XAsyncBlock status / result helpers
// ---------------------------------------------------------------------------

HResult XAsyncGetStatus(XAsyncBlock* async_block, bool wait);
HResult XAsyncGetResultSize(XAsyncBlock* async_block, size_t* buffer_size);

// ---------------------------------------------------------------------------
// Offline XUser stub (zero network / DRM)
// ---------------------------------------------------------------------------

HResult XUserAddAsync(XUserAddOptions options, XAsyncBlock* async_block);
HResult XUserAddResult(XAsyncBlock* async_block, XUserHandle* new_user);
HResult XUserGetId(XUserHandle user, uint64_t* user_id);
HResult XUserGetGamertag(XUserHandle user,
                         XUserGamertagComponent gamertag_component,
                         size_t gamertag_size,
                         char* gamertag,
                         size_t* gamertag_used);
void XUserCloseHandle(XUserHandle user);

// ---------------------------------------------------------------------------
// XSystem device & analytics info
// ---------------------------------------------------------------------------

XSystemDeviceType XSystemGetDeviceType();
XSystemAnalyticsInfo XSystemGetAnalyticsInfo();

}  // namespace playgta6::os::gdk

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints (collision-free with Windows/GDK SDK headers)
// ---------------------------------------------------------------------------

extern "C" {

Pg6HResult pg6_gdk_xgame_runtime_initialize(void);
void pg6_gdk_xgame_runtime_uninitialize(void);
int32_t pg6_gdk_xgame_runtime_is_initialized(void);

Pg6HResult pg6_gdk_xtask_queue_create(uint32_t work_mode,
                                      uint32_t completion_mode,
                                      Pg6XTaskQueueHandle* queue);
void pg6_gdk_xtask_queue_close_handle(Pg6XTaskQueueHandle queue);
Pg6HResult pg6_gdk_xtask_queue_duplicate_handle(Pg6XTaskQueueHandle queue,
                                                Pg6XTaskQueueHandle* duplicated_handle);
Pg6HResult pg6_gdk_xtask_queue_submit_callback(Pg6XTaskQueueHandle queue,
                                               uint32_t port,
                                               void* callback_context,
                                               Pg6XTaskQueueCallback callback);
int32_t pg6_gdk_xtask_queue_dispatch(Pg6XTaskQueueHandle queue,
                                     uint32_t port,
                                     uint32_t timeout_ms);
Pg6HResult pg6_gdk_xtask_queue_terminate(Pg6XTaskQueueHandle queue,
                                         int32_t wait,
                                         void* callback_context,
                                         Pg6XTaskQueueTerminatedCallback callback);
int32_t pg6_gdk_xtask_queue_get_current_process_task_queue(Pg6XTaskQueueHandle* queue);

Pg6HResult pg6_gdk_xasync_get_status(Pg6XAsyncBlock* async_block, int32_t wait);
Pg6HResult pg6_gdk_xasync_get_result_size(Pg6XAsyncBlock* async_block, size_t* buffer_size);

Pg6HResult pg6_gdk_xuser_add_async(uint32_t options, Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xuser_add_result(Pg6XAsyncBlock* async_block, Pg6XUserHandle* new_user);
Pg6HResult pg6_gdk_xuser_get_id(Pg6XUserHandle user, uint64_t* user_id);
Pg6HResult pg6_gdk_xuser_get_gamertag(Pg6XUserHandle user,
                                      uint32_t gamertag_component,
                                      size_t gamertag_size,
                                      char* gamertag,
                                      size_t* gamertag_used);
void pg6_gdk_xuser_close_handle(Pg6XUserHandle user);

uint32_t pg6_gdk_xsystem_get_device_type(void);
Pg6XSystemAnalyticsInfo pg6_gdk_xsystem_get_analytics_info(void);

}  // extern "C"
