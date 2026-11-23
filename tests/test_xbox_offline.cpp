#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "os/win32_kernel.h"
#include "os/xbox_offline.h"
#include "os/xgameruntime.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xbox_offline.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;
using namespace playgta6::os::offline;
using gdk::E_ABORT_HR;
using gdk::E_INVALIDARG_HR;
using gdk::S_OK_HR;

// Garbage values used to prove out-parameters are initialised on every path.
template <typename T>
T* garbage_ptr() {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(0xBAADF00Du));
}

struct CallbackProbe {
    std::atomic<int> calls{0};
    std::atomic<int32_t> status_in_callback{0};
};

void probe_callback(Pg6XAsyncBlock* block) {
    auto* probe = static_cast<CallbackProbe*>(block->context);
    // Titles typically read the status from inside the completion callback.
    probe->status_in_callback.store(pg6_gdk_xasync_get_status(block, 0));
    probe->calls.fetch_add(1);
}

bool wait_for_calls(const CallbackProbe& probe, int expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (probe.calls.load() < expected) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

Pg6XTaskQueueHandle make_manual_queue() {
    Pg6XTaskQueueHandle queue = nullptr;
    CHECK(gdk::XTaskQueueCreate(gdk::XTaskQueueDispatchMode::Manual,
                                gdk::XTaskQueueDispatchMode::Manual,
                                &queue) == S_OK_HR);
    return queue;
}

Pg6XUserHandle make_offline_user() {
    Pg6XTaskQueueHandle queue = make_manual_queue();
    Pg6XAsyncBlock async{};
    async.queue = queue;
    CHECK(pg6_gdk_xuser_add_async(1u, &async) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 0, 0) == 1);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 1, 0) == 1);
    Pg6XUserHandle user = nullptr;
    CHECK(pg6_gdk_xuser_add_result(&async, &user) == S_OK_HR);
    pg6_gdk_xtask_queue_close_handle(queue);
    return user;
}

// Drives one offline async begin on a Manual queue and verifies the exactly-once contract.
template <typename Begin>
void expect_offline_completion(Begin begin, int32_t expected_status, Pg6XAsyncBlock& async) {
    Pg6XTaskQueueHandle queue = make_manual_queue();
    CallbackProbe probe;
    async = Pg6XAsyncBlock{};
    async.queue = queue;
    async.context = &probe;
    async.callback = &probe_callback;

    CHECK(begin(&async) == S_OK_HR);
    CHECK(probe.calls.load() == 0);  // Not before the title pumps its completion port.
    CHECK(pg6_gdk_xasync_get_status(&async, 0) == expected_status);
    size_t size = 77;
    CHECK(pg6_gdk_xasync_get_result_size(&async, &size) ==
          (expected_status == S_OK_HR ? S_OK_HR : expected_status));
    CHECK(size == 0);

    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 0, 0) == 0);  // Nothing on the work port.
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 1, 0) == 1);
    CHECK(probe.calls.load() == 1);
    CHECK(probe.status_in_callback.load() == expected_status);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 1, 0) == 0);  // Exactly once.
    CHECK(probe.calls.load() == 1);
    CHECK(pg6_gdk_xasync_get_status(&async, 1) == expected_status);

    async.queue = nullptr;
    pg6_gdk_xtask_queue_close_handle(queue);
}

// ---------------------------------------------------------------------------
// 1. ws2_32
// ---------------------------------------------------------------------------

void test_winsock() {
    reset_offline_diagnostics();

    // Before WSAStartup.
    CHECK(pg6_win32_socket(2, 1, 6) == INVALID_SOCKET_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSANOTINITIALISED);
    CHECK(pg6_win32_wsa_cleanup() == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSANOTINITIALISED);
    Pg6AddrInfo* info = garbage_ptr<Pg6AddrInfo>();
    CHECK(pg6_win32_getaddrinfo("example.com", "80", nullptr, &info) == WSANOTINITIALISED);
    CHECK(info == nullptr);

    // WSAStartup.
    CHECK(pg6_win32_wsa_startup(0x0202u, nullptr) == WSAEFAULT);
    Pg6WsaData data{};
    CHECK(pg6_win32_wsa_startup(0x0000u, &data) == WSAVERNOTSUPPORTED);
    std::memset(&data, 0xCD, sizeof(data));
    CHECK(pg6_win32_wsa_startup(0x0202u, &data) == 0);
    CHECK(data.version == 0x0202u);
    CHECK(data.high_version == 0x0202u);
    CHECK(data.vendor_info == 0);
    CHECK(std::strlen(data.description) > 0 && std::strlen(data.description) < 257);
    CHECK(pg6_win32_wsa_startup(0x0303u, &data) == 0);  // Newer request -> 2.2.
    CHECK(data.version == 0x0202u);
    CHECK(pg6_win32_wsa_startup(0x0101u, &data) == 0);  // Older request is honoured.
    CHECK(data.version == 0x0101u);

    // Sockets never open; operations on sockets report WSAENOTSOCK.
    CHECK(pg6_win32_socket(2, 2, 17) == INVALID_SOCKET_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSAENETDOWN);
    CHECK(pg6_win32_wsa_socket_w(23, 1, 6, nullptr, 0, 0) == INVALID_SOCKET_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSAENETDOWN);

    const Pg6Socket bogus = 42;
    char buf[8]{};
    int32_t len = 8;
    uint32_t arg = 1;
    CHECK(pg6_win32_connect(bogus, buf, 16) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSAENOTSOCK);
    CHECK(pg6_win32_bind(bogus, buf, 16) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_listen(bogus, 4) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_accept(bogus, nullptr, nullptr) == INVALID_SOCKET_VALUE);
    CHECK(pg6_win32_send(bogus, buf, 8, 0) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_recv(bogus, buf, 8, 0) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_sendto(bogus, buf, 8, 0, buf, 16) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_recvfrom(bogus, buf, 8, 0, buf, &len) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_setsockopt(bogus, 0xFFFF, 4, buf, 4) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_getsockopt(bogus, 0xFFFF, 4, buf, &len) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_ioctlsocket(bogus, 0, &arg) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_shutdown(bogus, 2) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_closesocket(INVALID_SOCKET_VALUE) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSAENOTSOCK);
    CHECK(pg6_win32_select(0, buf, nullptr, nullptr, nullptr) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSAENOTSOCK);
    CHECK(pg6_win32_select(0, nullptr, nullptr, nullptr, nullptr) == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSAEINVAL);

    // Name resolution: host not found, result pointer cleared.
    info = garbage_ptr<Pg6AddrInfo>();
    CHECK(pg6_win32_getaddrinfo("example.com", "443", nullptr, &info) == WSAHOST_NOT_FOUND);
    CHECK(info == nullptr);
    CHECK(pg6_win32_wsa_get_last_error() == WSAHOST_NOT_FOUND);
    info = garbage_ptr<Pg6AddrInfo>();
    CHECK(pg6_win32_get_addr_info_w(u"localhost", nullptr, nullptr, &info) == WSAHOST_NOT_FOUND);
    CHECK(info == nullptr);
    CHECK(pg6_win32_getaddrinfo(nullptr, nullptr, nullptr, nullptr) == WSAEINVAL);
    pg6_win32_freeaddrinfo(nullptr);
    pg6_win32_free_addr_info_w(nullptr);

    // WSASetLastError shares the Win32 thread-local slot.
    pg6_win32_wsa_set_last_error(12345);
    CHECK(pg6_win32_wsa_get_last_error() == 12345);
    CHECK(win32::GetLastError() == 12345u);

    // Balanced cleanup (three successful startups).
    CHECK(pg6_win32_wsa_cleanup() == 0);
    CHECK(pg6_win32_wsa_cleanup() == 0);
    CHECK(pg6_win32_wsa_cleanup() == 0);
    CHECK(pg6_win32_wsa_cleanup() == SOCKET_ERROR_VALUE);
    CHECK(pg6_win32_wsa_get_last_error() == WSANOTINITIALISED);

    CHECK(stub_hit_count("WSAStartup") == 5);
    CHECK(stub_hit_count("WSACleanup") == 5);
    CHECK(stub_hit_count("socket") == 2);
    CHECK(stub_hit_count("getaddrinfo") == 3);
}

// ---------------------------------------------------------------------------
// 2. winhttp
// ---------------------------------------------------------------------------

void test_winhttp() {
    const size_t baseline = open_dummy_handle_count();

    const Pg6HInternet session = pg6_win32_winhttp_open(u"agent", 0, nullptr, nullptr, 0);
    CHECK(session != 0);
    const Pg6HInternet session2 = pg6_win32_winhttp_open(u"agent", 0, nullptr, nullptr, 0);
    CHECK(session2 != 0 && session2 != session);

    CHECK(pg6_win32_winhttp_connect(session, nullptr, 443, 0) == 0);
    CHECK(win32::GetLastError() == win32::ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_winhttp_connect(0x1234, u"example.com", 443, 0) == 0);
    CHECK(win32::GetLastError() == win32::ERROR_INVALID_HANDLE);

    const Pg6HInternet connection = pg6_win32_winhttp_connect(session, u"example.com", 443, 0);
    CHECK(connection != 0 && connection != session && connection != session2);
    // A session is not a connection.
    CHECK(pg6_win32_winhttp_open_request(session, u"GET", u"/", nullptr, nullptr, nullptr, 0) ==
          0);
    CHECK(win32::GetLastError() == win32::ERROR_INVALID_HANDLE);
    const Pg6HInternet request =
        pg6_win32_winhttp_open_request(connection, u"GET", u"/", nullptr, nullptr, nullptr, 0);
    CHECK(request != 0 && request != connection);

    CHECK(pg6_win32_winhttp_set_status_callback(request, nullptr, 0xFFFFFFFFu, 0) == nullptr);
    CHECK(reinterpret_cast<uintptr_t>(
              pg6_win32_winhttp_set_status_callback(0x1234, nullptr, 0, 0)) == ~uintptr_t{0});
    CHECK(pg6_win32_winhttp_set_option(0, 1, nullptr, 0) == 1);
    CHECK(pg6_win32_winhttp_set_option(request, 1, nullptr, 0) == 1);
    CHECK(pg6_win32_winhttp_set_timeouts(session, 1, 1, 1, 1) == 1);
    CHECK(pg6_win32_winhttp_set_timeouts(0x1234, 1, 1, 1, 1) == 0);

    CHECK(pg6_win32_winhttp_send_request(request, nullptr, 0, nullptr, 0, 0, 0) == 0);
    CHECK(win32::GetLastError() == ERROR_WINHTTP_NAME_NOT_RESOLVED);
    CHECK(pg6_win32_winhttp_send_request(connection, nullptr, 0, nullptr, 0, 0, 0) == 0);
    CHECK(win32::GetLastError() == win32::ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_winhttp_receive_response(request, nullptr) == 0);
    CHECK(win32::GetLastError() == ERROR_WINHTTP_INCORRECT_HANDLE_STATE);
    uint32_t available = 99;
    CHECK(pg6_win32_winhttp_query_data_available(request, &available) == 0);
    CHECK(available == 0);
    char buffer[16]{};
    uint32_t read = 99;
    CHECK(pg6_win32_winhttp_read_data(request, buffer, sizeof(buffer), &read) == 0);
    CHECK(read == 0);
    CHECK(win32::GetLastError() == ERROR_WINHTTP_INCORRECT_HANDLE_STATE);

    CHECK(open_dummy_handle_count() == baseline + 4);
    CHECK(pg6_win32_winhttp_close_handle(request) == 1);
    CHECK(pg6_win32_winhttp_close_handle(request) == 0);  // Double close.
    CHECK(win32::GetLastError() == win32::ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_winhttp_close_handle(connection) == 1);
    CHECK(pg6_win32_winhttp_close_handle(session) == 1);
    CHECK(pg6_win32_winhttp_close_handle(session2) == 1);
    CHECK(pg6_win32_winhttp_close_handle(0) == 0);
    CHECK(open_dummy_handle_count() == baseline);
}

// ---------------------------------------------------------------------------
// 3. XNetworking
// ---------------------------------------------------------------------------

std::atomic<int> g_connectivity_notifications{0};

void connectivity_changed(void*, const Pg6XNetworkingConnectivityHint*) {
    g_connectivity_notifications.fetch_add(1);
}

void test_xnetworking() {
    const size_t baseline = open_dummy_handle_count();

    CHECK(pg6_gdk_xnetworking_get_connectivity_hint(nullptr) == E_INVALIDARG_HR);
    Pg6XNetworkingConnectivityHint hint{};
    std::memset(&hint, 0xFF, sizeof(hint));
    CHECK(pg6_gdk_xnetworking_get_connectivity_hint(&hint) == S_OK_HR);
    CHECK(hint.connectivity_level == static_cast<uint32_t>(ConnectivityLevel::None));
    CHECK(hint.connectivity_cost == static_cast<uint32_t>(ConnectivityCost::Unknown));
    CHECK(hint.iana_interface_type == 0);
    CHECK(hint.network_initialized);
    CHECK(!hint.approaching_data_limit && !hint.over_data_limit && !hint.roaming);

    Pg6XTaskQueueHandle queue = make_manual_queue();
    CHECK(pg6_gdk_xnetworking_register_connectivity_hint_changed(
              queue, nullptr, &connectivity_changed, nullptr) == E_INVALIDARG_HR);
    Pg6XTaskQueueRegistrationToken token{0xDEADu};
    CHECK(pg6_gdk_xnetworking_register_connectivity_hint_changed(
              queue, nullptr, nullptr, &token) == E_INVALIDARG_HR);
    CHECK(token.token == 0);

    Pg6XTaskQueueRegistrationToken token_a{};
    Pg6XTaskQueueRegistrationToken token_b{};
    CHECK(pg6_gdk_xnetworking_register_connectivity_hint_changed(
              queue, nullptr, &connectivity_changed, &token_a) == S_OK_HR);
    CHECK(pg6_gdk_xnetworking_register_connectivity_hint_changed(
              nullptr, nullptr, &connectivity_changed, &token_b) == S_OK_HR);
    CHECK(token_a.token != 0 && token_b.token != 0 && token_a.token != token_b.token);

    // Never fires.
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 0, 0) == 0);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 1, 0) == 0);
    CHECK(g_connectivity_notifications.load() == 0);

    CHECK(pg6_gdk_xnetworking_unregister_connectivity_hint_changed(token_a, 1) == 1);
    CHECK(pg6_gdk_xnetworking_unregister_connectivity_hint_changed(token_a, 1) == 0);
    CHECK(pg6_gdk_xnetworking_unregister_connectivity_hint_changed(token_b, 0) == 1);
    CHECK(pg6_gdk_xnetworking_unregister_connectivity_hint_changed(
              Pg6XTaskQueueRegistrationToken{}, 0) == 0);
    CHECK(open_dummy_handle_count() == baseline);

    pg6_gdk_xtask_queue_close_handle(queue);
}

// ---------------------------------------------------------------------------
// 4. Xbox Live (XSAPI) + async completion contract
// ---------------------------------------------------------------------------

void test_xbox_live(Pg6XUserHandle user) {
    const size_t baseline = open_dummy_handle_count();

    CHECK(pg6_gdk_xbl_initialize(nullptr) == E_INVALIDARG_HR);
    const uint64_t fake_init_args[4]{};
    CHECK(pg6_gdk_xbl_initialize(fake_init_args) == S_OK_HR);

    // Context lifetime coordinated with O5's offline user.
    CHECK(pg6_gdk_xbl_context_create_handle(user, nullptr) == E_INVALIDARG_HR);
    Pg6XblContextHandle context = garbage_ptr<Pg6XblContext>();
    CHECK(pg6_gdk_xbl_context_create_handle(nullptr, &context) == E_INVALIDARG_HR);
    CHECK(context == nullptr);
    CHECK(pg6_gdk_xbl_context_create_handle(user, &context) == S_OK_HR);
    CHECK(context != nullptr);
    Pg6XblContextHandle other = nullptr;
    CHECK(pg6_gdk_xbl_context_create_handle(user, &other) == S_OK_HR);
    CHECK(other != nullptr && other != context);

    uint64_t xuid = 1;
    CHECK(pg6_gdk_xbl_context_get_xbox_user_id(context, &xuid) == S_OK_HR);
    CHECK(xuid == gdk::kOfflineUserXuid);
    CHECK(pg6_gdk_xbl_context_get_xbox_user_id(context, nullptr) == E_INVALIDARG_HR);

    Pg6XblContextHandle duplicate = garbage_ptr<Pg6XblContext>();
    CHECK(pg6_gdk_xbl_context_duplicate_handle(garbage_ptr<Pg6XblContext>(), &duplicate) ==
          E_INVALIDARG_HR);
    CHECK(duplicate == nullptr);
    CHECK(pg6_gdk_xbl_context_duplicate_handle(context, &duplicate) == S_OK_HR);
    CHECK(duplicate == context);
    pg6_gdk_xbl_context_close_handle(duplicate);
    CHECK(pg6_gdk_xbl_context_get_xbox_user_id(context, &xuid) == S_OK_HR);  // Still alive.
    pg6_gdk_xbl_context_close_handle(other);
    pg6_gdk_xbl_context_close_handle(nullptr);  // Harmless.

    // Every service call completes exactly once with the offline HRESULT.
    Pg6XAsyncBlock async{};
    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xbl_achievements_get_achievements_for_title_id_async(
                context, xuid, 0x1234u, 0, 0, 0, 0, 100, b);
        },
        kOfflineHResult,
        async);
    Pg6OfflineResultHandle result = garbage_ptr<Pg6OfflineResult>();
    CHECK(pg6_gdk_xbl_achievements_get_achievements_for_title_id_result(&async, &result) ==
          kOfflineHResult);
    CHECK(result == nullptr);
    CHECK(pg6_gdk_xbl_achievements_get_achievements_for_title_id_result(&async, nullptr) ==
          E_INVALIDARG_HR);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xbl_achievements_update_achievement_async(context, xuid, "1", 100, b);
        },
        kOfflineHResult,
        async);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xbl_profile_get_user_profile_async(context, xuid, b);
        },
        kOfflineHResult,
        async);
    unsigned char profile[64];
    std::memset(profile, 0xAB, sizeof(profile));
    CHECK(pg6_gdk_xbl_profile_get_user_profile_result(&async, profile) == kOfflineHResult);
    CHECK(profile[0] == 0xAB && profile[63] == 0xAB);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xbl_presence_set_presence_async(context, 1, nullptr, b);
        },
        kOfflineHResult,
        async);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xbl_user_statistics_get_single_user_statistic_async(
                context, xuid, "scid", "stat", b);
        },
        kOfflineHResult,
        async);
    size_t size = 99;
    CHECK(pg6_gdk_xbl_user_statistics_get_single_user_statistic_result_size(&async, &size) ==
          kOfflineHResult);
    CHECK(size == 0);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xbl_social_get_social_relationships_async(context, xuid, 0, 0, 10, b);
        },
        kOfflineHResult,
        async);
    result = garbage_ptr<Pg6OfflineResult>();
    CHECK(pg6_gdk_xbl_social_get_social_relationships_result(&async, &result) == kOfflineHResult);
    CHECK(result == nullptr);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xbl_title_storage_get_quota_async(context, "scid", 0, b);
        },
        kOfflineHResult,
        async);
    size_t used = 5;
    size_t quota = 5;
    CHECK(pg6_gdk_xbl_title_storage_get_quota_result(&async, &used, &quota) == kOfflineHResult);
    CHECK(used == 0 && quota == 0);
    quota = 5;
    CHECK(pg6_gdk_xbl_title_storage_get_quota_result(&async, nullptr, &quota) == E_INVALIDARG_HR);
    CHECK(quota == 0);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xbl_multiplayer_get_session_async(context, nullptr, b);
        },
        kOfflineHResult,
        async);
    result = garbage_ptr<Pg6OfflineResult>();
    CHECK(pg6_gdk_xbl_multiplayer_get_session_result(&async, &result) == kOfflineHResult);
    CHECK(result == nullptr);

    // Invalid arguments fail synchronously and leave the block untouched (no callback).
    {
        CallbackProbe probe;
        Pg6XAsyncBlock untouched{};
        untouched.context = &probe;
        untouched.callback = &probe_callback;
        std::memset(untouched.internal, 0x5A, sizeof(untouched.internal));
        Pg6XAsyncBlock snapshot = untouched;
        CHECK(pg6_gdk_xbl_profile_get_user_profile_async(nullptr, xuid, &untouched) ==
              E_INVALIDARG_HR);
        CHECK(pg6_gdk_xbl_profile_get_user_profile_async(
                  garbage_ptr<Pg6XblContext>(), xuid, &untouched) == E_INVALIDARG_HR);
        CHECK(pg6_gdk_xbl_profile_get_user_profile_async(context, xuid, nullptr) ==
              E_INVALIDARG_HR);
        CHECK(std::memcmp(snapshot.internal, untouched.internal, sizeof(untouched.internal)) == 0);
        CHECK(probe.calls.load() == 0);

        // Result getter on a block never started by an offline stub.
        Pg6XAsyncBlock foreign{};
        result = garbage_ptr<Pg6OfflineResult>();
        CHECK(pg6_gdk_xbl_social_get_social_relationships_result(&foreign, &result) ==
              E_INVALIDARG_HR);
        CHECK(result == nullptr);
        CHECK(pg6_gdk_xbl_social_get_social_relationships_result(nullptr, &result) ==
              E_INVALIDARG_HR);
    }

    // Immediate completion port: callback runs before the begin call returns.
    {
        Pg6XTaskQueueHandle immediate = nullptr;
        CHECK(gdk::XTaskQueueCreate(gdk::XTaskQueueDispatchMode::Immediate,
                                    gdk::XTaskQueueDispatchMode::Immediate,
                                    &immediate) == S_OK_HR);
        CallbackProbe probe;
        Pg6XAsyncBlock block{};
        block.queue = immediate;
        block.context = &probe;
        block.callback = &probe_callback;
        CHECK(pg6_gdk_xbl_presence_set_presence_async(context, 0, nullptr, &block) == S_OK_HR);
        CHECK(probe.calls.load() == 1);
        CHECK(probe.status_in_callback.load() == kOfflineHResult);
        pg6_gdk_xtask_queue_close_handle(immediate);
    }

    // Null queue: completes on the process task queue (thread pool), exactly once.
    {
        CallbackProbe probe;
        Pg6XAsyncBlock block{};
        block.context = &probe;
        block.callback = &probe_callback;
        CHECK(pg6_gdk_xbl_profile_get_user_profile_async(context, xuid, &block) == S_OK_HR);
        CHECK(wait_for_calls(probe, 1));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK(probe.calls.load() == 1);
        CHECK(probe.status_in_callback.load() == kOfflineHResult);
        CHECK(pg6_gdk_xasync_get_status(&block, 1) == kOfflineHResult);
    }

    // Terminated queue: begin fails synchronously, block untouched, callback never runs.
    {
        Pg6XTaskQueueHandle dead = make_manual_queue();
        CHECK(pg6_gdk_xtask_queue_terminate(dead, 1, nullptr, nullptr) == S_OK_HR);
        CallbackProbe probe;
        Pg6XAsyncBlock block{};
        block.queue = dead;
        block.context = &probe;
        block.callback = &probe_callback;
        CHECK(pg6_gdk_xbl_profile_get_user_profile_async(context, xuid, &block) == E_ABORT_HR);
        const Pg6XAsyncBlock zero{};
        CHECK(std::memcmp(zero.internal, block.internal, sizeof(block.internal)) == 0);
        CHECK(probe.calls.load() == 0);
        pg6_gdk_xtask_queue_close_handle(dead);
    }

    // Queue terminated while pending: callback still fires once, with E_ABORT.
    {
        Pg6XTaskQueueHandle queue = make_manual_queue();
        CallbackProbe probe;
        Pg6XAsyncBlock block{};
        block.queue = queue;
        block.context = &probe;
        block.callback = &probe_callback;
        CHECK(pg6_gdk_xbl_profile_get_user_profile_async(context, xuid, &block) == S_OK_HR);
        CHECK(pg6_gdk_xtask_queue_terminate(queue, 1, nullptr, nullptr) == S_OK_HR);
        CHECK(probe.calls.load() == 1);
        CHECK(probe.status_in_callback.load() == E_ABORT_HR);
        CHECK(pg6_gdk_xasync_get_status(&block, 0) == E_ABORT_HR);
        pg6_gdk_xtask_queue_close_handle(queue);
    }

    // Telemetry through XblEvents is accepted and dropped.
    const uint64_t dropped = dropped_telemetry_event_count();
    CHECK(pg6_gdk_xbl_events_write_in_game_event(context, "PlayerSessionStart", "{}", "{}") ==
          S_OK_HR);
    CHECK(pg6_gdk_xbl_events_write_in_game_event(context, nullptr, "{}", "{}") ==
          E_INVALIDARG_HR);
    CHECK(pg6_gdk_xbl_events_write_in_game_event(nullptr, "Evt", nullptr, nullptr) ==
          E_INVALIDARG_HR);
    CHECK(dropped_telemetry_event_count() == dropped + 1);

    pg6_gdk_xbl_context_close_handle(context);
    CHECK(pg6_gdk_xbl_context_get_xbox_user_id(context, &xuid) == E_INVALIDARG_HR);
    CHECK(xuid == 0);
    CHECK(open_dummy_handle_count() == baseline);

    // XblCleanupAsync completes successfully through the same path.
    expect_offline_completion(
        [](Pg6XAsyncBlock* b) { return pg6_gdk_xbl_cleanup_async(b); }, S_OK_HR, async);
    result = garbage_ptr<Pg6OfflineResult>();
    CHECK(pg6_gdk_xbl_multiplayer_get_session_result(&async, &result) == E_INVALIDARG_HR);
    CHECK(result == nullptr);
    CHECK(pg6_gdk_xbl_cleanup_async(nullptr) == E_INVALIDARG_HR);
}

// ---------------------------------------------------------------------------
// 5. XStore
// ---------------------------------------------------------------------------

void test_xstore(Pg6XUserHandle user) {
    const size_t baseline = open_dummy_handle_count();

    CHECK(pg6_gdk_xstore_create_context(user, nullptr) == E_INVALIDARG_HR);
    Pg6XStoreContextHandle store = nullptr;
    CHECK(pg6_gdk_xstore_create_context(user, &store) == S_OK_HR);
    CHECK(store != nullptr);

    Pg6XAsyncBlock async{};
    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) { return pg6_gdk_xstore_query_game_license_async(store, b); },
        kOfflineHResult,
        async);
    unsigned char license[128];
    std::memset(license, 0xAB, sizeof(license));
    CHECK(pg6_gdk_xstore_query_game_license_result(&async, license) == kOfflineHResult);
    CHECK(license[0] == 0xAB && license[127] == 0xAB);  // No license data is fabricated.
    CHECK(pg6_gdk_xstore_query_game_license_result(&async, nullptr) == E_INVALIDARG_HR);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) { return pg6_gdk_xstore_query_add_on_licenses_async(store, b); },
        kOfflineHResult,
        async);
    uint32_t count = 9;
    CHECK(pg6_gdk_xstore_query_add_on_licenses_result_count(&async, &count) == kOfflineHResult);
    CHECK(count == 0);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xstore_query_associated_products_async(store, 0xFFu, 25, b);
        },
        kOfflineHResult,
        async);
    Pg6OfflineResultHandle query = garbage_ptr<Pg6OfflineResult>();
    CHECK(pg6_gdk_xstore_query_associated_products_result(&async, &query) == kOfflineHResult);
    CHECK(query == nullptr);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xstore_query_product_for_current_game_async(store, b);
        },
        kOfflineHResult,
        async);
    query = garbage_ptr<Pg6OfflineResult>();
    CHECK(pg6_gdk_xstore_query_product_for_current_game_result(&async, &query) ==
          kOfflineHResult);
    CHECK(query == nullptr);

    Pg6XAsyncBlock block{};
    CHECK(pg6_gdk_xstore_query_game_license_async(nullptr, &block) == E_INVALIDARG_HR);

    // License-changed registration never fires and unregisters once.
    Pg6XTaskQueueRegistrationToken token{7};
    CHECK(pg6_gdk_xstore_register_game_license_changed(
              nullptr, nullptr, nullptr, [](void*) {}, &token) == E_INVALIDARG_HR);
    CHECK(token.token == 0);
    CHECK(pg6_gdk_xstore_register_game_license_changed(
              store, nullptr, nullptr, [](void*) {}, &token) == S_OK_HR);
    CHECK(token.token != 0);
    CHECK(pg6_gdk_xstore_unregister_game_license_changed(store, token, 1) == 1);
    CHECK(pg6_gdk_xstore_unregister_game_license_changed(store, token, 1) == 0);

    pg6_gdk_xstore_close_context_handle(store);
    CHECK(pg6_gdk_xstore_query_game_license_async(store, &block) == E_INVALIDARG_HR);
    CHECK(open_dummy_handle_count() == baseline);
}

// ---------------------------------------------------------------------------
// 6. Live-dependent XUser calls
// ---------------------------------------------------------------------------

void test_xuser_live(Pg6XUserHandle user) {
    Pg6XAsyncBlock block{};
    CHECK(pg6_gdk_xuser_get_token_and_signature_async(
              user, 0, "GET", nullptr, 0, nullptr, 0, nullptr, &block) == E_INVALIDARG_HR);
    CHECK(pg6_gdk_xuser_get_token_and_signature_async(
              nullptr, 0, "GET", "https://x", 0, nullptr, 0, nullptr, &block) == E_INVALIDARG_HR);

    Pg6XAsyncBlock async{};
    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xuser_get_token_and_signature_async(
                user, 0, "GET", "https://example.com/", 0, nullptr, 0, nullptr, b);
        },
        kOfflineHResult,
        async);
    size_t size = 3;
    CHECK(pg6_gdk_xuser_get_token_and_signature_result_size(&async, &size) == kOfflineHResult);
    CHECK(size == 0);
    char buffer[32];
    void* ptr = buffer;
    size_t used = 3;
    CHECK(pg6_gdk_xuser_get_token_and_signature_result(
              &async, sizeof(buffer), buffer, &ptr, &used) == kOfflineHResult);
    CHECK(ptr == nullptr && used == 0);

    bool has = true;
    uint32_t reason = 0;
    CHECK(pg6_gdk_xuser_check_privilege(user, 0, 254, &has, &reason) == S_OK_HR);
    CHECK(!has);
    CHECK(reason == static_cast<uint32_t>(PrivilegeDenyReason::Unknown));
    has = true;
    CHECK(pg6_gdk_xuser_check_privilege(user, 0, 254, &has, nullptr) == S_OK_HR);
    CHECK(!has);
    has = true;
    CHECK(pg6_gdk_xuser_check_privilege(nullptr, 0, 254, &has, &reason) == E_INVALIDARG_HR);
    CHECK(!has);
    CHECK(pg6_gdk_xuser_check_privilege(user, 0, 254, nullptr, &reason) == E_INVALIDARG_HR);

    expect_offline_completion(
        [&](Pg6XAsyncBlock* b) {
            return pg6_gdk_xuser_resolve_privilege_with_ui_async(user, 0, 254, b);
        },
        kOfflineHResult,
        async);
    CHECK(pg6_gdk_xuser_resolve_privilege_with_ui_result(&async) == kOfflineHResult);
    CHECK(pg6_gdk_xuser_resolve_privilege_with_ui_async(user, 0, 254, nullptr) ==
          E_INVALIDARG_HR);
}

// ---------------------------------------------------------------------------
// 7. ETW / TraceLogging telemetry
// ---------------------------------------------------------------------------

void test_telemetry() {
    const size_t baseline = open_dummy_handle_count();
    const unsigned char provider_guid[16] = {1, 2, 3, 4};
    const unsigned char descriptor[16] = {};

    CHECK(pg6_win32_event_register(provider_guid, nullptr, nullptr, nullptr) ==
          ETW_ERROR_INVALID_PARAMETER);
    Pg6RegHandle handle = 0xFFFF;
    CHECK(pg6_win32_event_register(nullptr, nullptr, nullptr, &handle) ==
          ETW_ERROR_INVALID_PARAMETER);
    CHECK(handle == 0);
    CHECK(pg6_win32_event_register(provider_guid, nullptr, nullptr, &handle) == ETW_ERROR_SUCCESS);
    CHECK(handle != 0);
    Pg6RegHandle handle2 = 0;
    CHECK(pg6_win32_event_register(provider_guid, nullptr, nullptr, &handle2) ==
          ETW_ERROR_SUCCESS);
    CHECK(handle2 != 0 && handle2 != handle);

    CHECK(pg6_win32_event_set_information(handle, 2, descriptor, sizeof(descriptor)) ==
          ETW_ERROR_SUCCESS);
    CHECK(pg6_win32_event_set_information(0x77, 2, descriptor, sizeof(descriptor)) ==
          ETW_ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_event_enabled(handle, descriptor) == 0);
    CHECK(pg6_win32_event_provider_enabled(handle, 5, ~uint64_t{0}) == 0);

    // Payload pointers are never dereferenced: an unreadable pointer must not crash.
    const void* unreadable = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x10));
    const uint64_t before = dropped_telemetry_event_count();
    CHECK(pg6_win32_event_write(handle, descriptor, 4, unreadable) == ETW_ERROR_SUCCESS);
    CHECK(pg6_win32_event_write_transfer(handle, descriptor, unreadable, unreadable, 8,
                                         unreadable) == ETW_ERROR_SUCCESS);
    CHECK(pg6_win32_event_write(0, descriptor, 0, nullptr) == ETW_ERROR_SUCCESS);
    CHECK(pg6_win32_event_write(handle, nullptr, 0, nullptr) == ETW_ERROR_INVALID_PARAMETER);
    CHECK(dropped_telemetry_event_count() == before + 3);

    CHECK(pg6_win32_event_unregister(handle) == ETW_ERROR_SUCCESS);
    CHECK(pg6_win32_event_unregister(handle) == ETW_ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_event_unregister(handle2) == ETW_ERROR_SUCCESS);
    CHECK(open_dummy_handle_count() == baseline);
}

// ---------------------------------------------------------------------------
// 8. Hit counters: snapshot, reset, thread-safety
// ---------------------------------------------------------------------------

void test_hit_counters() {
    reset_offline_diagnostics();
    CHECK(stub_hit_count("WSAStartup") == 0);
    CHECK(stub_hit_count("NotAnImport") == 0);
    CHECK(dropped_telemetry_event_count() == 0);

    for (int i = 0; i < 3; ++i) {
        pg6_win32_wsa_get_last_error();
    }
    CHECK(stub_hit_count("WSAGetLastError") == 3);
    const auto snapshot = stub_hit_snapshot();
    CHECK(snapshot.size() == 1);
    CHECK(!snapshot.empty() && snapshot[0].first == "WSAGetLastError" &&
          snapshot[0].second == 3);

    constexpr int kThreads = 8;
    constexpr int kIterations = 1000;
    const size_t baseline = open_dummy_handle_count();
    const unsigned char descriptor[16] = {};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&descriptor]() {
            for (int i = 0; i < kIterations; ++i) {
                pg6_win32_event_write(0, descriptor, 0, nullptr);
                const Pg6HInternet session =
                    pg6_win32_winhttp_open(u"agent", 0, nullptr, nullptr, 0);
                pg6_win32_winhttp_close_handle(session);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    constexpr uint64_t kTotal = static_cast<uint64_t>(kThreads) * kIterations;
    CHECK(stub_hit_count("EventWrite") == kTotal);
    CHECK(stub_hit_count("WinHttpOpen") == kTotal);
    CHECK(stub_hit_count("WinHttpCloseHandle") == kTotal);
    CHECK(dropped_telemetry_event_count() == kTotal);
    CHECK(open_dummy_handle_count() == baseline);

    const auto full = stub_hit_snapshot();
    CHECK(full.size() == 4);
    for (size_t i = 1; i < full.size(); ++i) {
        CHECK(full[i - 1].first < full[i].first);
    }

    reset_offline_diagnostics();
    CHECK(stub_hit_count("EventWrite") == 0);
    CHECK(stub_hit_snapshot().empty());
    CHECK(dropped_telemetry_event_count() == 0);
}

}  // namespace

int main() {
    CHECK(gdk::XGameRuntimeInitialize() == S_OK_HR);
    Pg6XUserHandle user = make_offline_user();
    CHECK(user != nullptr);

    test_winsock();
    test_winhttp();
    test_xnetworking();
    test_xbox_live(user);
    test_xstore(user);
    test_xuser_live(user);
    test_telemetry();
    test_hit_counters();

    pg6_gdk_xuser_close_handle(user);
    gdk::XGameRuntimeUninitialize();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_xbox_offline failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_xbox_offline_tests passed.\n");
    return 0;
}
