#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "os/xgameruntime.h"

// Offline-mode safe stubs for networking (ws2_32, winhttp, XNetworking), Xbox Live (XSAPI,
// Live-dependent XUser calls), XStore, and telemetry (ETW / TraceLogging, XblEvents).
//
// The product is strictly local single-player: no stub performs any network I/O. Every entry
// point either fails with the documented "offline / network unreachable" result a title already
// handles for a console without connectivity, or succeeds as a local no-op (context/handle
// creation, telemetry). Calls taking an XAsyncBlock either fail synchronously without touching
// the block (invalid arguments) or return S_OK and complete the block exactly once through the
// block's XTaskQueue completion port (O5 `xgameruntime`), so the title callback always fires.

// Opaque C-ABI handles (prefixed with Pg6* to avoid collisions with GDK / Windows SDK headers).
// Handle values are tokens from an internal table and are never dereferenced.
struct Pg6XblContext;
struct Pg6XStoreContext;
struct Pg6OfflineResult;  // Achievement/social/session/product result handles (never produced).
struct Pg6AddrInfo;       // ADDRINFOA / ADDRINFOW (never produced).

using Pg6XblContextHandle = Pg6XblContext*;
using Pg6XStoreContextHandle = Pg6XStoreContext*;
using Pg6OfflineResultHandle = Pg6OfflineResult*;
using Pg6Socket = uint64_t;     // SOCKET (UINT_PTR on 64-bit).
using Pg6HInternet = uint64_t;  // WinHTTP HINTERNET.
using Pg6RegHandle = uint64_t;  // ETW REGHANDLE.

// WSADATA (64-bit layout).
struct Pg6WsaData {
    uint16_t version = 0;
    uint16_t high_version = 0;
    uint16_t max_sockets = 0;
    uint16_t max_udp_dg = 0;
    uint64_t vendor_info = 0;
    char description[257]{};
    char system_status[129]{};
};
static_assert(sizeof(Pg6WsaData) == 408, "Pg6WsaData must match the 64-bit WSADATA layout");

// XNetworkingConnectivityHint.
struct Pg6XNetworkingConnectivityHint {
    uint32_t connectivity_level = 0;
    uint32_t connectivity_cost = 0;
    uint32_t iana_interface_type = 0;
    bool network_initialized = false;
    bool approaching_data_limit = false;
    bool over_data_limit = false;
    bool roaming = false;
};
static_assert(sizeof(Pg6XNetworkingConnectivityHint) == 16,
              "Pg6XNetworkingConnectivityHint must match XNetworkingConnectivityHint");

// XTaskQueueRegistrationToken.
struct Pg6XTaskQueueRegistrationToken {
    uint64_t token = 0;
};

using Pg6WinHttpStatusCallback = void (*)(Pg6HInternet handle,
                                          uint64_t context,
                                          uint32_t internet_status,
                                          void* status_information,
                                          uint32_t status_information_length);
using Pg6XNetworkingConnectivityHintChangedCallback =
    void (*)(void* context, const Pg6XNetworkingConnectivityHint* connectivity_hint);
using Pg6XStoreGameLicenseChangedCallback = void (*)(void* context);
using Pg6EtwEnableCallback = void (*)(const void* source_id,
                                      uint32_t is_enabled,
                                      uint8_t level,
                                      uint64_t match_any_keyword,
                                      uint64_t match_all_keyword,
                                      const void* filter_data,
                                      void* callback_context);

namespace playgta6::os::offline {

using HResult = Pg6HResult;

// The single HRESULT every Live / Store / Live-dependent XUser network operation reports:
// HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE).
inline constexpr HResult kOfflineHResult = static_cast<HResult>(0x800704CFu);

// WinSock.
inline constexpr Pg6Socket INVALID_SOCKET_VALUE = ~uint64_t{0};
inline constexpr int32_t SOCKET_ERROR_VALUE = -1;
inline constexpr uint16_t kWinsockHighVersion = 0x0202u;  // MAKEWORD(2, 2)
inline constexpr int32_t WSAEFAULT = 10014;
inline constexpr int32_t WSAEINVAL = 10022;
inline constexpr int32_t WSAENOTSOCK = 10038;
inline constexpr int32_t WSAENETDOWN = 10050;
inline constexpr int32_t WSAVERNOTSUPPORTED = 10092;
inline constexpr int32_t WSANOTINITIALISED = 10093;
inline constexpr int32_t WSAHOST_NOT_FOUND = 11001;  // == EAI_NONAME

// WinHTTP (GetLastError values).
inline constexpr uint32_t ERROR_WINHTTP_NAME_NOT_RESOLVED = 12007u;
inline constexpr uint32_t ERROR_WINHTTP_INCORRECT_HANDLE_STATE = 12019u;

// ETW status codes.
inline constexpr uint32_t ETW_ERROR_SUCCESS = 0u;
inline constexpr uint32_t ETW_ERROR_INVALID_HANDLE = 6u;
inline constexpr uint32_t ETW_ERROR_INVALID_PARAMETER = 87u;

// XNetworkingConnectivityLevelHint / XNetworkingConnectivityCostHint.
enum class ConnectivityLevel : uint32_t {
    Unknown = 0,
    None = 1,
    LocalAccess = 2,
    InternetAccess = 3,
    ConstrainedInternetAccess = 4,
};

enum class ConnectivityCost : uint32_t {
    Unknown = 0,
    Unrestricted = 1,
    Fixed = 2,
    Variable = 3,
};

// XUserPrivilegeDenyReason.
enum class PrivilegeDenyReason : uint32_t {
    None = 0,
    PurchaseRequired = 1,
    Restricted = 2,
    Banned = 3,
    Unknown = 0xFFFFFFFFu,
};

// ---------------------------------------------------------------------------
// Bring-up diagnostics (thread-safe)
// ---------------------------------------------------------------------------

// Number of calls made to the stub for `import_name` (the original DLL export name, e.g.
// "WSAStartup", "XblContextCreateHandle"). Unknown or never-called names return 0.
uint64_t stub_hit_count(std::string_view import_name);

// (import name, call count) for every stub called at least once, sorted by name.
std::vector<std::pair<std::string, uint64_t>> stub_hit_snapshot();

// Telemetry events accepted and dropped (payloads are never read or stored).
uint64_t dropped_telemetry_event_count();

// Number of live dummy handles (WinHTTP, Xbl/XStore contexts, ETW providers, registrations).
size_t open_dummy_handle_count();

// Zeroes stub hit counters and the dropped-telemetry counter (handles are unaffected).
void reset_offline_diagnostics();

}  // namespace playgta6::os::offline

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints. Boolean results are 1/0 (int32_t) like O5; BOOLEAN is uint8_t.
// ---------------------------------------------------------------------------

extern "C" {

// ws2_32: WSAStartup/WSACleanup succeed; sockets never open (WSAENETDOWN); name resolution fails
// with WSAHOST_NOT_FOUND. WSA last error shares the Win32 thread-local GetLastError slot.
int32_t pg6_win32_wsa_startup(uint16_t version_requested, Pg6WsaData* wsa_data);
int32_t pg6_win32_wsa_cleanup(void);
int32_t pg6_win32_wsa_get_last_error(void);
void pg6_win32_wsa_set_last_error(int32_t error);
Pg6Socket pg6_win32_socket(int32_t af, int32_t type, int32_t protocol);
Pg6Socket pg6_win32_wsa_socket_w(int32_t af,
                                 int32_t type,
                                 int32_t protocol,
                                 void* protocol_info,
                                 uint32_t group,
                                 uint32_t flags);
int32_t pg6_win32_closesocket(Pg6Socket s);
int32_t pg6_win32_connect(Pg6Socket s, const void* name, int32_t name_length);
int32_t pg6_win32_bind(Pg6Socket s, const void* name, int32_t name_length);
int32_t pg6_win32_listen(Pg6Socket s, int32_t backlog);
Pg6Socket pg6_win32_accept(Pg6Socket s, void* address, int32_t* address_length);
int32_t pg6_win32_send(Pg6Socket s, const char* buffer, int32_t length, int32_t flags);
int32_t pg6_win32_recv(Pg6Socket s, char* buffer, int32_t length, int32_t flags);
int32_t pg6_win32_sendto(Pg6Socket s,
                         const char* buffer,
                         int32_t length,
                         int32_t flags,
                         const void* to,
                         int32_t to_length);
int32_t pg6_win32_recvfrom(Pg6Socket s,
                           char* buffer,
                           int32_t length,
                           int32_t flags,
                           void* from,
                           int32_t* from_length);
int32_t pg6_win32_setsockopt(Pg6Socket s,
                             int32_t level,
                             int32_t option_name,
                             const char* option_value,
                             int32_t option_length);
int32_t pg6_win32_getsockopt(Pg6Socket s,
                             int32_t level,
                             int32_t option_name,
                             char* option_value,
                             int32_t* option_length);
int32_t pg6_win32_ioctlsocket(Pg6Socket s, int32_t command, uint32_t* argument);
int32_t pg6_win32_shutdown(Pg6Socket s, int32_t how);
int32_t pg6_win32_select(int32_t nfds,
                         void* read_fds,
                         void* write_fds,
                         void* except_fds,
                         const void* timeout);
int32_t pg6_win32_getaddrinfo(const char* node_name,
                              const char* service_name,
                              const void* hints,
                              Pg6AddrInfo** result);
int32_t pg6_win32_get_addr_info_w(const char16_t* node_name,
                                  const char16_t* service_name,
                                  const void* hints,
                                  Pg6AddrInfo** result);
void pg6_win32_freeaddrinfo(Pg6AddrInfo* info);
void pg6_win32_free_addr_info_w(Pg6AddrInfo* info);

// winhttp: session/connect/request handles are local dummies (as on an offline PC); sending a
// request fails with ERROR_WINHTTP_NAME_NOT_RESOLVED and no status callback is ever invoked.
Pg6HInternet pg6_win32_winhttp_open(const char16_t* agent,
                                    uint32_t access_type,
                                    const char16_t* proxy_name,
                                    const char16_t* proxy_bypass,
                                    uint32_t flags);
Pg6HInternet pg6_win32_winhttp_connect(Pg6HInternet session,
                                       const char16_t* server_name,
                                       uint16_t server_port,
                                       uint32_t reserved);
Pg6HInternet pg6_win32_winhttp_open_request(Pg6HInternet connection,
                                            const char16_t* verb,
                                            const char16_t* object_name,
                                            const char16_t* version,
                                            const char16_t* referrer,
                                            const char16_t* const* accept_types,
                                            uint32_t flags);
int32_t pg6_win32_winhttp_send_request(Pg6HInternet request,
                                       const char16_t* headers,
                                       uint32_t headers_length,
                                       const void* optional,
                                       uint32_t optional_length,
                                       uint32_t total_length,
                                       uint64_t context);
int32_t pg6_win32_winhttp_receive_response(Pg6HInternet request, void* reserved);
int32_t pg6_win32_winhttp_query_data_available(Pg6HInternet request, uint32_t* bytes_available);
int32_t pg6_win32_winhttp_read_data(Pg6HInternet request,
                                    void* buffer,
                                    uint32_t bytes_to_read,
                                    uint32_t* bytes_read);
Pg6WinHttpStatusCallback pg6_win32_winhttp_set_status_callback(Pg6HInternet handle,
                                                               Pg6WinHttpStatusCallback callback,
                                                               uint32_t notification_flags,
                                                               uint64_t reserved);
int32_t pg6_win32_winhttp_set_option(Pg6HInternet handle,
                                     uint32_t option,
                                     const void* buffer,
                                     uint32_t buffer_length);
int32_t pg6_win32_winhttp_set_timeouts(Pg6HInternet handle,
                                       int32_t resolve_timeout,
                                       int32_t connect_timeout,
                                       int32_t send_timeout,
                                       int32_t receive_timeout);
int32_t pg6_win32_winhttp_close_handle(Pg6HInternet handle);

// XNetworking: initialized network with no connectivity; change notifications never fire.
Pg6HResult pg6_gdk_xnetworking_get_connectivity_hint(Pg6XNetworkingConnectivityHint* hint);
Pg6HResult pg6_gdk_xnetworking_register_connectivity_hint_changed(
    Pg6XTaskQueueHandle queue,
    void* context,
    Pg6XNetworkingConnectivityHintChangedCallback callback,
    Pg6XTaskQueueRegistrationToken* token);
int32_t pg6_gdk_xnetworking_unregister_connectivity_hint_changed(
    Pg6XTaskQueueRegistrationToken token,
    int32_t wait);

// Live-dependent XUser calls (the local offline user itself is O5's XUserAddAsync).
Pg6HResult pg6_gdk_xuser_get_token_and_signature_async(Pg6XUserHandle user,
                                                       uint32_t options,
                                                       const char* method,
                                                       const char* url,
                                                       size_t header_count,
                                                       const void* headers,
                                                       size_t body_size,
                                                       const void* body_buffer,
                                                       Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xuser_get_token_and_signature_result_size(Pg6XAsyncBlock* async_block,
                                                             size_t* buffer_size);
Pg6HResult pg6_gdk_xuser_get_token_and_signature_result(Pg6XAsyncBlock* async_block,
                                                        size_t buffer_size,
                                                        void* buffer,
                                                        void** ptr_to_buffer,
                                                        size_t* buffer_used);
Pg6HResult pg6_gdk_xuser_check_privilege(Pg6XUserHandle user,
                                         uint32_t options,
                                         uint32_t privilege,
                                         bool* has_privilege,
                                         uint32_t* deny_reason);
Pg6HResult pg6_gdk_xuser_resolve_privilege_with_ui_async(Pg6XUserHandle user,
                                                         uint32_t options,
                                                         uint32_t privilege,
                                                         Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xuser_resolve_privilege_with_ui_result(Pg6XAsyncBlock* async_block);

// Xbox Live (XSAPI): initialization and contexts are local and succeed; every service call
// completes its XAsyncBlock with kOfflineHResult. Struct-typed results (user profile) are not
// modelled and are left untouched on failure.
Pg6HResult pg6_gdk_xbl_initialize(const void* init_args);
Pg6HResult pg6_gdk_xbl_cleanup_async(Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_context_create_handle(Pg6XUserHandle user, Pg6XblContextHandle* context);
Pg6HResult pg6_gdk_xbl_context_duplicate_handle(Pg6XblContextHandle context,
                                                Pg6XblContextHandle* duplicated);
void pg6_gdk_xbl_context_close_handle(Pg6XblContextHandle context);
Pg6HResult pg6_gdk_xbl_context_get_xbox_user_id(Pg6XblContextHandle context, uint64_t* xuid);

Pg6HResult pg6_gdk_xbl_achievements_get_achievements_for_title_id_async(
    Pg6XblContextHandle context,
    uint64_t xbox_user_id,
    uint32_t title_id,
    uint32_t type,
    int32_t unlocked_only,
    uint32_t order_by,
    uint32_t skip_items,
    uint32_t max_items,
    Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_achievements_get_achievements_for_title_id_result(
    Pg6XAsyncBlock* async_block,
    Pg6OfflineResultHandle* result);
Pg6HResult pg6_gdk_xbl_achievements_update_achievement_async(Pg6XblContextHandle context,
                                                             uint64_t xbox_user_id,
                                                             const char* achievement_id,
                                                             uint32_t percent_complete,
                                                             Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_profile_get_user_profile_async(Pg6XblContextHandle context,
                                                      uint64_t xbox_user_id,
                                                      Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_profile_get_user_profile_result(Pg6XAsyncBlock* async_block,
                                                       void* profile);
Pg6HResult pg6_gdk_xbl_presence_set_presence_async(Pg6XblContextHandle context,
                                                   int32_t is_user_active_in_title,
                                                   const void* rich_presence_ids,
                                                   Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_user_statistics_get_single_user_statistic_async(
    Pg6XblContextHandle context,
    uint64_t xbox_user_id,
    const char* service_configuration_id,
    const char* statistic_name,
    Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_user_statistics_get_single_user_statistic_result_size(
    Pg6XAsyncBlock* async_block,
    size_t* result_size);
Pg6HResult pg6_gdk_xbl_social_get_social_relationships_async(Pg6XblContextHandle context,
                                                             uint64_t xbox_user_id,
                                                             uint32_t filter,
                                                             size_t start_index,
                                                             size_t max_items,
                                                             Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_social_get_social_relationships_result(Pg6XAsyncBlock* async_block,
                                                              Pg6OfflineResultHandle* result);
Pg6HResult pg6_gdk_xbl_title_storage_get_quota_async(Pg6XblContextHandle context,
                                                     const char* service_configuration_id,
                                                     uint32_t storage_type,
                                                     Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_title_storage_get_quota_result(Pg6XAsyncBlock* async_block,
                                                      size_t* used_bytes,
                                                      size_t* quota_bytes);
Pg6HResult pg6_gdk_xbl_multiplayer_get_session_async(Pg6XblContextHandle context,
                                                     const void* session_reference,
                                                     Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xbl_multiplayer_get_session_result(Pg6XAsyncBlock* async_block,
                                                      Pg6OfflineResultHandle* session);
Pg6HResult pg6_gdk_xbl_events_write_in_game_event(Pg6XblContextHandle context,
                                                  const char* event_name,
                                                  const char* dimensions_json,
                                                  const char* measurements_json);

// XStore: store contexts are local; every query completes with kOfflineHResult (no license or
// product data is fabricated). Struct-typed results (game license) are left untouched.
Pg6HResult pg6_gdk_xstore_create_context(Pg6XUserHandle user,
                                         Pg6XStoreContextHandle* store_context);
void pg6_gdk_xstore_close_context_handle(Pg6XStoreContextHandle store_context);
Pg6HResult pg6_gdk_xstore_query_game_license_async(Pg6XStoreContextHandle store_context,
                                                   Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xstore_query_game_license_result(Pg6XAsyncBlock* async_block,
                                                    void* license);
Pg6HResult pg6_gdk_xstore_query_add_on_licenses_async(Pg6XStoreContextHandle store_context,
                                                      Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xstore_query_add_on_licenses_result_count(Pg6XAsyncBlock* async_block,
                                                             uint32_t* count);
Pg6HResult pg6_gdk_xstore_query_associated_products_async(Pg6XStoreContextHandle store_context,
                                                          uint32_t product_kinds,
                                                          uint32_t max_items_per_page,
                                                          Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xstore_query_associated_products_result(Pg6XAsyncBlock* async_block,
                                                           Pg6OfflineResultHandle* query);
Pg6HResult pg6_gdk_xstore_query_product_for_current_game_async(
    Pg6XStoreContextHandle store_context,
    Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xstore_query_product_for_current_game_result(Pg6XAsyncBlock* async_block,
                                                                Pg6OfflineResultHandle* query);
Pg6HResult pg6_gdk_xstore_register_game_license_changed(
    Pg6XStoreContextHandle store_context,
    Pg6XTaskQueueHandle queue,
    void* context,
    Pg6XStoreGameLicenseChangedCallback callback,
    Pg6XTaskQueueRegistrationToken* token);
int32_t pg6_gdk_xstore_unregister_game_license_changed(Pg6XStoreContextHandle store_context,
                                                       Pg6XTaskQueueRegistrationToken token,
                                                       int32_t wait);

// advapi32 ETW / TraceLogging: providers register as dummies, nobody ever enables them, and
// every event write is accepted and dropped without reading the payload.
uint32_t pg6_win32_event_register(const void* provider_id,
                                  Pg6EtwEnableCallback enable_callback,
                                  void* callback_context,
                                  Pg6RegHandle* reg_handle);
uint32_t pg6_win32_event_unregister(Pg6RegHandle reg_handle);
uint32_t pg6_win32_event_set_information(Pg6RegHandle reg_handle,
                                         uint32_t information_class,
                                         const void* event_information,
                                         uint32_t information_length);
uint8_t pg6_win32_event_enabled(Pg6RegHandle reg_handle, const void* event_descriptor);
uint8_t pg6_win32_event_provider_enabled(Pg6RegHandle reg_handle, uint8_t level, uint64_t keyword);
uint32_t pg6_win32_event_write(Pg6RegHandle reg_handle,
                               const void* event_descriptor,
                               uint32_t user_data_count,
                               const void* user_data);
uint32_t pg6_win32_event_write_transfer(Pg6RegHandle reg_handle,
                                        const void* event_descriptor,
                                        const void* activity_id,
                                        const void* related_activity_id,
                                        uint32_t user_data_count,
                                        const void* user_data);

}  // extern "C"
