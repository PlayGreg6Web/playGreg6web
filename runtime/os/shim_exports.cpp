#include "os/shim_exports.h"

#include <mutex>
#include <string_view>

#include "os/win32_kernel.h"
#include "os/win32_sync.h"
#include "os/xbox_offline.h"
#include "os/xgameruntime.h"

namespace {

// Module tags used by the table (`E(tag, ...)`). Placements follow where the real DLLs export the
// API; see the header comment for the uncertain ones.
#define PG6_MOD_K32 "KERNEL32.dll"
#define PG6_MOD_ADV "ADVAPI32.dll"
#define PG6_MOD_WS2 "WS2_32.dll"
#define PG6_MOD_WHTTP "WINHTTP.dll"
#define PG6_MOD_XGR "xgameruntime.dll"
#define PG6_MOD_XBL "Microsoft.Xbox.Services.141.GDK.C.Thunks.dll"

// Second name of the XSAPI thunk module (toolset-independent spelling); same export set as XBL.
constexpr const char kXblAliasModule[] = "Microsoft.Xbox.Services.GDK.C.Thunks.dll";

// One line per shim: E(module tag, exported API name, C-ABI symbol). Adding a shim = one line here.
// tests/test_shim_exports.py parses these lines; keep the E(...) one-per-line shape.
#define PG6_SHIM_EXPORTS(E) \
    E(K32, "GetProcessHeap", pg6_win32_get_process_heap) \
    E(K32, "HeapCreate", pg6_win32_heap_create) \
    E(K32, "HeapDestroy", pg6_win32_heap_destroy) \
    E(K32, "HeapAlloc", pg6_win32_heap_alloc) \
    E(K32, "HeapReAlloc", pg6_win32_heap_realloc) \
    E(K32, "HeapFree", pg6_win32_heap_free) \
    E(K32, "HeapSize", pg6_win32_heap_size) \
    E(K32, "HeapValidate", pg6_win32_heap_validate) \
    E(K32, "GetModuleHandleA", pg6_win32_get_module_handle_a) \
    E(K32, "GetModuleHandleW", pg6_win32_get_module_handle_w) \
    E(K32, "LoadLibraryA", pg6_win32_load_library_a) \
    E(K32, "LoadLibraryW", pg6_win32_load_library_w) \
    E(K32, "LoadLibraryExA", pg6_win32_load_library_ex_a) \
    E(K32, "LoadLibraryExW", pg6_win32_load_library_ex_w) \
    E(K32, "FreeLibrary", pg6_win32_free_library) \
    E(K32, "GetModuleFileNameA", pg6_win32_get_module_file_name_a) \
    E(K32, "GetModuleFileNameW", pg6_win32_get_module_file_name_w) \
    E(K32, "GetProcAddress", pg6_win32_get_proc_address) \
    E(K32, "VirtualAlloc", pg6_win32_virtual_alloc) \
    E(K32, "VirtualProtect", pg6_win32_virtual_protect) \
    E(K32, "VirtualFree", pg6_win32_virtual_free) \
    E(K32, "VirtualQuery", pg6_win32_virtual_query) \
    E(K32, "TlsAlloc", pg6_win32_tls_alloc) \
    E(K32, "TlsFree", pg6_win32_tls_free) \
    E(K32, "TlsGetValue", pg6_win32_tls_get_value) \
    E(K32, "TlsSetValue", pg6_win32_tls_set_value) \
    E(K32, "InitializeSRWLock", pg6_win32_initialize_srw_lock) \
    E(K32, "AcquireSRWLockExclusive", pg6_win32_acquire_srw_lock_exclusive) \
    E(K32, "ReleaseSRWLockExclusive", pg6_win32_release_srw_lock_exclusive) \
    E(K32, "AcquireSRWLockShared", pg6_win32_acquire_srw_lock_shared) \
    E(K32, "ReleaseSRWLockShared", pg6_win32_release_srw_lock_shared) \
    E(K32, "TryAcquireSRWLockExclusive", pg6_win32_try_acquire_srw_lock_exclusive) \
    E(K32, "TryAcquireSRWLockShared", pg6_win32_try_acquire_srw_lock_shared) \
    E(K32, "InitializeCriticalSection", pg6_win32_initialize_critical_section) \
    E(K32, "DeleteCriticalSection", pg6_win32_delete_critical_section) \
    E(K32, "EnterCriticalSection", pg6_win32_enter_critical_section) \
    E(K32, "TryEnterCriticalSection", pg6_win32_try_enter_critical_section) \
    E(K32, "LeaveCriticalSection", pg6_win32_leave_critical_section) \
    E(K32, "CreateEventA", pg6_win32_create_event_a) \
    E(K32, "CreateEventW", pg6_win32_create_event_w) \
    E(K32, "SetEvent", pg6_win32_set_event) \
    E(K32, "ResetEvent", pg6_win32_reset_event) \
    E(K32, "WaitForSingleObject", pg6_win32_wait_for_single_object) \
    E(K32, "CreateFileA", pg6_win32_create_file_a) \
    E(K32, "CreateFileW", pg6_win32_create_file_w) \
    E(K32, "ReadFile", pg6_win32_read_file) \
    E(K32, "WriteFile", pg6_win32_write_file) \
    E(K32, "SetFilePointerEx", pg6_win32_set_file_pointer_ex) \
    E(K32, "GetFileSizeEx", pg6_win32_get_file_size_ex) \
    E(K32, "CloseHandle", pg6_win32_close_handle) \
    E(K32, "QueryPerformanceCounter", pg6_win32_query_performance_counter) \
    E(K32, "QueryPerformanceFrequency", pg6_win32_query_performance_frequency) \
    E(K32, "GetTickCount64", pg6_win32_get_tick_count64) \
    E(K32, "GetSystemTimeAsFileTime", pg6_win32_get_system_time_as_file_time) \
    E(K32, "GetLastError", pg6_win32_get_last_error) \
    E(K32, "SetLastError", pg6_win32_set_last_error) \
    E(K32, "CreateSemaphoreA", pg6_win32_create_semaphore_a) \
    E(K32, "CreateSemaphoreW", pg6_win32_create_semaphore_w) \
    E(K32, "ReleaseSemaphore", pg6_win32_release_semaphore) \
    E(K32, "CreateMutexA", pg6_win32_create_mutex_a) \
    E(K32, "CreateMutexW", pg6_win32_create_mutex_w) \
    E(K32, "ReleaseMutex", pg6_win32_release_mutex) \
    E(K32, "WaitForSingleObjectEx", pg6_win32_wait_for_single_object_ex) \
    E(K32, "WaitForMultipleObjects", pg6_win32_wait_for_multiple_objects) \
    E(K32, "WaitForMultipleObjectsEx", pg6_win32_wait_for_multiple_objects_ex) \
    E(K32, "InitializeConditionVariable", pg6_win32_initialize_condition_variable) \
    E(K32, "SleepConditionVariableSRW", pg6_win32_sleep_condition_variable_srw) \
    E(K32, "SleepConditionVariableCS", pg6_win32_sleep_condition_variable_cs) \
    E(K32, "WakeConditionVariable", pg6_win32_wake_condition_variable) \
    E(K32, "WakeAllConditionVariable", pg6_win32_wake_all_condition_variable) \
    E(K32, "CreateThread", pg6_win32_create_thread) \
    E(K32, "ResumeThread", pg6_win32_resume_thread) \
    E(K32, "GetExitCodeThread", pg6_win32_get_exit_code_thread) \
    E(K32, "GetCurrentThread", pg6_win32_get_current_thread) \
    E(K32, "GetCurrentThreadId", pg6_win32_get_current_thread_id) \
    E(K32, "SetThreadAffinityMask", pg6_win32_set_thread_affinity_mask) \
    E(K32, "CreateThreadpoolWork", pg6_win32_create_threadpool_work) \
    E(K32, "SubmitThreadpoolWork", pg6_win32_submit_threadpool_work) \
    E(K32, "WaitForThreadpoolWorkCallbacks", pg6_win32_wait_for_threadpool_work_callbacks) \
    E(K32, "CloseThreadpoolWork", pg6_win32_close_threadpool_work) \
    E(K32, "TrySubmitThreadpoolCallback", pg6_win32_try_submit_threadpool_callback) \
    E(XGR, "XGameRuntimeInitialize", pg6_gdk_xgame_runtime_initialize) \
    E(XGR, "XGameRuntimeUninitialize", pg6_gdk_xgame_runtime_uninitialize) \
    E(XGR, "XGameRuntimeIsInitialized", pg6_gdk_xgame_runtime_is_initialized) \
    E(XGR, "XTaskQueueCreate", pg6_gdk_xtask_queue_create) \
    E(XGR, "XTaskQueueCloseHandle", pg6_gdk_xtask_queue_close_handle) \
    E(XGR, "XTaskQueueSubmitCallback", pg6_gdk_xtask_queue_submit_callback) \
    E(XGR, "XTaskQueueDispatch", pg6_gdk_xtask_queue_dispatch) \
    E(XGR, "XTaskQueueTerminate", pg6_gdk_xtask_queue_terminate) \
    E(XGR, "XTaskQueueGetCurrentProcessTaskQueue", pg6_gdk_xtask_queue_get_current_process_task_queue) \
    E(XGR, "XAsyncGetStatus", pg6_gdk_xasync_get_status) \
    E(XGR, "XAsyncGetResultSize", pg6_gdk_xasync_get_result_size) \
    E(XGR, "XUserAddAsync", pg6_gdk_xuser_add_async) \
    E(XGR, "XUserAddResult", pg6_gdk_xuser_add_result) \
    E(XGR, "XUserGetId", pg6_gdk_xuser_get_id) \
    E(XGR, "XUserGetGamertag", pg6_gdk_xuser_get_gamertag) \
    E(XGR, "XUserCloseHandle", pg6_gdk_xuser_close_handle) \
    E(XGR, "XSystemGetDeviceType", pg6_gdk_xsystem_get_device_type) \
    E(XGR, "XSystemGetAnalyticsInfo", pg6_gdk_xsystem_get_analytics_info) \
    E(XGR, "XNetworkingGetConnectivityHint", pg6_gdk_xnetworking_get_connectivity_hint) \
    E(XGR, "XNetworkingRegisterConnectivityHintChanged", pg6_gdk_xnetworking_register_connectivity_hint_changed) \
    E(XGR, "XNetworkingUnregisterConnectivityHintChanged", pg6_gdk_xnetworking_unregister_connectivity_hint_changed) \
    E(XGR, "XUserGetTokenAndSignatureAsync", pg6_gdk_xuser_get_token_and_signature_async) \
    E(XGR, "XUserGetTokenAndSignatureResultSize", pg6_gdk_xuser_get_token_and_signature_result_size) \
    E(XGR, "XUserGetTokenAndSignatureResult", pg6_gdk_xuser_get_token_and_signature_result) \
    E(XGR, "XUserCheckPrivilege", pg6_gdk_xuser_check_privilege) \
    E(XGR, "XUserResolvePrivilegeWithUiAsync", pg6_gdk_xuser_resolve_privilege_with_ui_async) \
    E(XGR, "XUserResolvePrivilegeWithUiResult", pg6_gdk_xuser_resolve_privilege_with_ui_result) \
    E(XGR, "XStoreCreateContext", pg6_gdk_xstore_create_context) \
    E(XGR, "XStoreCloseContextHandle", pg6_gdk_xstore_close_context_handle) \
    E(XGR, "XStoreQueryGameLicenseAsync", pg6_gdk_xstore_query_game_license_async) \
    E(XGR, "XStoreQueryGameLicenseResult", pg6_gdk_xstore_query_game_license_result) \
    E(XGR, "XStoreQueryAddOnLicensesAsync", pg6_gdk_xstore_query_add_on_licenses_async) \
    E(XGR, "XStoreQueryAddOnLicensesResultCount", pg6_gdk_xstore_query_add_on_licenses_result_count) \
    E(XGR, "XStoreQueryAssociatedProductsAsync", pg6_gdk_xstore_query_associated_products_async) \
    E(XGR, "XStoreQueryAssociatedProductsResult", pg6_gdk_xstore_query_associated_products_result) \
    E(XGR, "XStoreQueryProductForCurrentGameAsync", pg6_gdk_xstore_query_product_for_current_game_async) \
    E(XGR, "XStoreQueryProductForCurrentGameResult", pg6_gdk_xstore_query_product_for_current_game_result) \
    E(XGR, "XStoreRegisterGameLicenseChanged", pg6_gdk_xstore_register_game_license_changed) \
    E(XGR, "XStoreUnregisterGameLicenseChanged", pg6_gdk_xstore_unregister_game_license_changed) \
    E(WS2, "WSAStartup", pg6_win32_wsa_startup) \
    E(WS2, "WSACleanup", pg6_win32_wsa_cleanup) \
    E(WS2, "WSAGetLastError", pg6_win32_wsa_get_last_error) \
    E(WS2, "WSASetLastError", pg6_win32_wsa_set_last_error) \
    E(WS2, "socket", pg6_win32_socket) \
    E(WS2, "WSASocketW", pg6_win32_wsa_socket_w) \
    E(WS2, "closesocket", pg6_win32_closesocket) \
    E(WS2, "connect", pg6_win32_connect) \
    E(WS2, "bind", pg6_win32_bind) \
    E(WS2, "listen", pg6_win32_listen) \
    E(WS2, "accept", pg6_win32_accept) \
    E(WS2, "send", pg6_win32_send) \
    E(WS2, "recv", pg6_win32_recv) \
    E(WS2, "sendto", pg6_win32_sendto) \
    E(WS2, "recvfrom", pg6_win32_recvfrom) \
    E(WS2, "setsockopt", pg6_win32_setsockopt) \
    E(WS2, "getsockopt", pg6_win32_getsockopt) \
    E(WS2, "ioctlsocket", pg6_win32_ioctlsocket) \
    E(WS2, "shutdown", pg6_win32_shutdown) \
    E(WS2, "select", pg6_win32_select) \
    E(WS2, "getaddrinfo", pg6_win32_getaddrinfo) \
    E(WS2, "GetAddrInfoW", pg6_win32_get_addr_info_w) \
    E(WS2, "freeaddrinfo", pg6_win32_freeaddrinfo) \
    E(WS2, "FreeAddrInfoW", pg6_win32_free_addr_info_w) \
    E(WHTTP, "WinHttpOpen", pg6_win32_winhttp_open) \
    E(WHTTP, "WinHttpConnect", pg6_win32_winhttp_connect) \
    E(WHTTP, "WinHttpOpenRequest", pg6_win32_winhttp_open_request) \
    E(WHTTP, "WinHttpSendRequest", pg6_win32_winhttp_send_request) \
    E(WHTTP, "WinHttpReceiveResponse", pg6_win32_winhttp_receive_response) \
    E(WHTTP, "WinHttpQueryDataAvailable", pg6_win32_winhttp_query_data_available) \
    E(WHTTP, "WinHttpReadData", pg6_win32_winhttp_read_data) \
    E(WHTTP, "WinHttpSetStatusCallback", pg6_win32_winhttp_set_status_callback) \
    E(WHTTP, "WinHttpSetOption", pg6_win32_winhttp_set_option) \
    E(WHTTP, "WinHttpSetTimeouts", pg6_win32_winhttp_set_timeouts) \
    E(WHTTP, "WinHttpCloseHandle", pg6_win32_winhttp_close_handle) \
    E(XBL, "XblInitialize", pg6_gdk_xbl_initialize) \
    E(XBL, "XblCleanupAsync", pg6_gdk_xbl_cleanup_async) \
    E(XBL, "XblContextCreateHandle", pg6_gdk_xbl_context_create_handle) \
    E(XBL, "XblContextDuplicateHandle", pg6_gdk_xbl_context_duplicate_handle) \
    E(XBL, "XblContextCloseHandle", pg6_gdk_xbl_context_close_handle) \
    E(XBL, "XblContextGetXboxUserId", pg6_gdk_xbl_context_get_xbox_user_id) \
    E(XBL, "XblAchievementsGetAchievementsForTitleIdAsync", pg6_gdk_xbl_achievements_get_achievements_for_title_id_async) \
    E(XBL, "XblAchievementsGetAchievementsForTitleIdResult", pg6_gdk_xbl_achievements_get_achievements_for_title_id_result) \
    E(XBL, "XblAchievementsUpdateAchievementAsync", pg6_gdk_xbl_achievements_update_achievement_async) \
    E(XBL, "XblProfileGetUserProfileAsync", pg6_gdk_xbl_profile_get_user_profile_async) \
    E(XBL, "XblProfileGetUserProfileResult", pg6_gdk_xbl_profile_get_user_profile_result) \
    E(XBL, "XblPresenceSetPresenceAsync", pg6_gdk_xbl_presence_set_presence_async) \
    E(XBL, "XblUserStatisticsGetSingleUserStatisticAsync", pg6_gdk_xbl_user_statistics_get_single_user_statistic_async) \
    E(XBL, "XblUserStatisticsGetSingleUserStatisticResultSize", pg6_gdk_xbl_user_statistics_get_single_user_statistic_result_size) \
    E(XBL, "XblSocialGetSocialRelationshipsAsync", pg6_gdk_xbl_social_get_social_relationships_async) \
    E(XBL, "XblSocialGetSocialRelationshipsResult", pg6_gdk_xbl_social_get_social_relationships_result) \
    E(XBL, "XblTitleStorageGetQuotaAsync", pg6_gdk_xbl_title_storage_get_quota_async) \
    E(XBL, "XblTitleStorageGetQuotaResult", pg6_gdk_xbl_title_storage_get_quota_result) \
    E(XBL, "XblMultiplayerGetSessionAsync", pg6_gdk_xbl_multiplayer_get_session_async) \
    E(XBL, "XblMultiplayerGetSessionResult", pg6_gdk_xbl_multiplayer_get_session_result) \
    E(XBL, "XblEventsWriteInGameEvent", pg6_gdk_xbl_events_write_in_game_event) \
    E(ADV, "EventRegister", pg6_win32_event_register) \
    E(ADV, "EventUnregister", pg6_win32_event_unregister) \
    E(ADV, "EventSetInformation", pg6_win32_event_set_information) \
    E(ADV, "EventEnabled", pg6_win32_event_enabled) \
    E(ADV, "EventProviderEnabled", pg6_win32_event_provider_enabled) \
    E(ADV, "EventWrite", pg6_win32_event_write) \
    E(ADV, "EventWriteTransfer", pg6_win32_event_write_transfer)

#define PG6_SHIM_ENTRY(mod, name, fn) \
    Pg6ShimExport{PG6_MOD_##mod, name, #fn, playgta6::os::win32::proc_address(&fn)},

const Pg6ShimExport kTable[] = {PG6_SHIM_EXPORTS(PG6_SHIM_ENTRY)};

constexpr size_t kTableCount = sizeof(kTable) / sizeof(kTable[0]);

size_t register_all_once() {
    using namespace playgta6::os::win32;
    // Contract names the registered families are reached through, beyond O7's defaults
    // (api-ms-win-core-* already maps to KERNEL32.dll): ETW lives in api-ms-win-eventing-*.
    register_api_set_rule("api-ms-win-eventing-", "advapi32.dll");

    for (const Pg6ShimExport& e : kTable) {
        register_export(e.module, e.name, 0, e.address);
        if (std::string_view(e.module) == PG6_MOD_XBL) {
            register_export(kXblAliasModule, e.name, 0, e.address);
        }
    }
    return kTableCount;
}

}  // namespace

namespace playgta6::os::win32 {

size_t register_all_shim_exports() {
    static std::once_flag once;
    static size_t registered = 0;
    std::call_once(once, [] { registered = register_all_once(); });
    return registered;
}

const Pg6ShimExport* shim_export_table(size_t* count) {
    if (count != nullptr) {
        *count = kTableCount;
    }
    return kTable;
}

}  // namespace playgta6::os::win32

extern "C" {

size_t pg6_loader_register_all_shim_exports(void) {
    return playgta6::os::win32::register_all_shim_exports();
}

const Pg6ShimExport* pg6_loader_shim_export_table(size_t* count) {
    return playgta6::os::win32::shim_export_table(count);
}

}  // extern "C"
