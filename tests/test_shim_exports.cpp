#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "os/shim_exports.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/xasync_provider.h"
#include "os/xbox_offline.h"
#include "os/xgameruntime.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_shim_exports.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os::win32;

Pg6ProcAddress by_name(Pg6Handle module, const char* name) {
    return pg6_loader_get_proc_address_by_name(module, name);
}

void test_registration_idempotent() {
    const size_t first = pg6_loader_register_all_shim_exports();
    CHECK(first == 293);
    CHECK(register_all_shim_exports() == first);
    CHECK(pg6_loader_register_all_shim_exports() == first);

    size_t count = 0;
    const Pg6ShimExport* table = shim_export_table(&count);
    CHECK(table != nullptr);
    CHECK(count == first);
    CHECK(pg6_loader_shim_export_table(nullptr) == table);

    // Concurrent re-registration is a no-op as well.
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([first] { CHECK(register_all_shim_exports() == first); });
    }
    for (auto& t : threads) {
        t.join();
    }
}

void test_table_counts_and_lookup() {
    size_t count = 0;
    const Pg6ShimExport* table = shim_export_table(&count);

    std::map<std::string, size_t> per_module;
    std::set<std::string> seen;
    for (size_t i = 0; i < count; ++i) {
        const Pg6ShimExport& e = table[i];
        CHECK(e.module != nullptr && e.name != nullptr && e.symbol != nullptr);
        CHECK(e.address != 0);
        CHECK(std::strncmp(e.symbol, "pg6_", 4) == 0);
        CHECK(seen.insert(std::string(e.module) + "!" + e.name).second);  // no duplicate exports
        ++per_module[e.module];

        const Pg6Handle mod = pg6_loader_register_virtual_module(e.module, 1);
        CHECK(mod != 0);
        CHECK(by_name(mod, e.name) == e.address);
        CHECK(pg6_loader_resolve_import(e.module, e.name) == e.address);
    }
    CHECK(per_module.size() == 10);
    CHECK(per_module["KERNEL32.dll"] == 154);
    CHECK(per_module["ADVAPI32.dll"] == 10);
    CHECK(per_module["WS2_32.dll"] == 24);
    CHECK(per_module["WINHTTP.dll"] == 15);
    CHECK(per_module["xgameruntime.dll"] == 48);
    CHECK(per_module["Microsoft.Xbox.Services.141.GDK.C.Thunks.dll"] == 21);
    CHECK(per_module["bcrypt.dll"] == 3);
    CHECK(per_module["USER32.dll"] == 7);
    CHECK(per_module["combase.dll"] == 8);
    CHECK(per_module["d3d12_xs.dll"] == 3);

    // The unversioned XSAPI alias exposes the same exports.
    const Pg6Handle versioned = pg6_win32_get_module_handle_a("Microsoft.Xbox.Services.141.GDK.C.Thunks.dll");
    const Pg6Handle alias = pg6_win32_get_module_handle_a("microsoft.xbox.services.gdk.c.thunks.dll");
    CHECK(versioned != 0 && alias != 0 && versioned != alias);
    CHECK(by_name(alias, "XblInitialize") != 0);
    CHECK(by_name(alias, "XblInitialize") == by_name(versioned, "XblInitialize"));
}

void test_module_lookup() {
    const Pg6Handle k32 = pg6_win32_get_module_handle_a("KERNEL32.dll");
    CHECK(k32 != 0);
    CHECK(pg6_win32_get_module_handle_a("kernel32") == k32);
    CHECK(pg6_win32_get_module_handle_a("Kernel32.DLL") == k32);
    CHECK(pg6_win32_load_library_a("kernel32") == k32);
    CHECK(pg6_win32_load_library_a("api-ms-win-core-synch-l1-2-0") == k32);
    CHECK(pg6_win32_load_library_a("API-MS-WIN-CORE-FILE-L1-1-0.DLL") == k32);

    const Pg6Handle adv = pg6_win32_get_module_handle_a("advapi32");
    CHECK(adv != 0 && adv != k32);
    CHECK(pg6_win32_load_library_a("api-ms-win-eventing-provider-l1-1-0.dll") == adv);

    const char* others[] = {"WS2_32", "ws2_32.dll", "winhttp", "WINHTTP.DLL", "xgameruntime",
                            "XGameRuntime.dll"};
    for (const char* name : others) {
        CHECK(pg6_win32_get_module_handle_a(name) != 0);
        CHECK(pg6_win32_load_library_a(name) != 0);
    }
    CHECK(pg6_win32_get_module_handle_a("ws2_32") != pg6_win32_get_module_handle_a("winhttp"));

    const char16_t wide[] = u"xgameruntime";
    CHECK(pg6_win32_get_module_handle_w(wide) == pg6_win32_get_module_handle_a("xgameruntime.dll"));
}

void test_samples_per_family() {
    struct Sample {
        const char* module;
        const char* name;
        Pg6ProcAddress expected;
    };
    const Sample samples[] = {
        {"kernel32", "VirtualAlloc", proc_address(&pg6_win32_virtual_alloc)},
        {"kernel32", "HeapReAlloc", proc_address(&pg6_win32_heap_realloc)},
        {"kernel32", "CreateFileW", proc_address(&pg6_win32_create_file_w)},
        {"kernel32", "InitializeSRWLock", proc_address(&pg6_win32_initialize_srw_lock)},
        {"api-ms-win-core-file-l1-1-0", "ReadFile", proc_address(&pg6_win32_read_file)},
        {"advapi32", "EventWrite", proc_address(&pg6_win32_event_write)},
        {"api-ms-win-eventing-provider-l1-1-0", "EventRegister",
         proc_address(&pg6_win32_event_register)},
        {"ws2_32", "WSAStartup", proc_address(&pg6_win32_wsa_startup)},
        {"ws2_32", "getaddrinfo", proc_address(&pg6_win32_getaddrinfo)},
        {"ws2_32", "GetAddrInfoW", proc_address(&pg6_win32_get_addr_info_w)},
        {"winhttp", "WinHttpOpen", proc_address(&pg6_win32_winhttp_open)},
        {"winhttp", "WinHttpCrackUrl", proc_address(&pg6_win32_winhttp_crack_url)},
        {"api-ms-win-core-string-l1-1-0", "MultiByteToWideChar",
         proc_address(&pg6_win32_multi_byte_to_wide_char)},
        {"api-ms-win-core-file-l1-1-0", "FindFirstFileW", proc_address(&pg6_win32_find_first_file_w)},
        {"api-ms-win-core-com-l1-1-0", "CoCreateGuid", proc_address(&pg6_win32_co_create_guid)},
        {"ext-ms-win-rtcore-ntuser-message-l1-1-0", "TranslateMessage",
         proc_address(&pg6_win32_translate_message)},
        {"user32", "CreateWindowExA", proc_address(&pg6_win32_create_window_ex_a)},
        {"advapi32", "RegOpenKeyW", proc_address(&pg6_win32_reg_open_key_w)},
        {"bcrypt", "BCryptGenRandom", proc_address(&pg6_win32_bcrypt_gen_random)},
        {"xgameruntime", "XTaskQueueCreate", proc_address(&pg6_gdk_xtask_queue_create)},
        {"xgameruntime", "XAsyncBegin", proc_address(&pg6_gdk_xasync_begin)},
        {"xgameruntime", "XUserGetGamertag", proc_address(&pg6_gdk_xuser_get_gamertag)},
        {"xgameruntime", "XStoreQueryGameLicenseAsync",
         proc_address(&pg6_gdk_xstore_query_game_license_async)},
        {"xgameruntime", "XNetworkingGetConnectivityHint",
         proc_address(&pg6_gdk_xnetworking_get_connectivity_hint)},
        {"Microsoft.Xbox.Services.141.GDK.C.Thunks.dll", "XblContextCreateHandle",
         proc_address(&pg6_gdk_xbl_context_create_handle)},
    };
    for (const Sample& s : samples) {
        const Pg6Handle mod = pg6_win32_load_library_a(s.module);
        CHECK(mod != 0);
        CHECK(by_name(mod, s.name) == s.expected);
        CHECK(s.expected != 0);
    }

    // Exports are case-sensitive like on Windows.
    CHECK(by_name(pg6_win32_get_module_handle_a("ws2_32"), "WSASTARTUP") == 0);
}

void test_exports_are_callable() {
    const Pg6Handle k32 = pg6_win32_get_module_handle_a("kernel32.dll");

    using TickFn = uint64_t (*)();
    using SetErrFn = void (*)(uint32_t);
    using GetErrFn = uint32_t (*)();
    const auto tick = reinterpret_cast<TickFn>(static_cast<uintptr_t>(by_name(k32, "GetTickCount64")));
    const auto set_err = reinterpret_cast<SetErrFn>(static_cast<uintptr_t>(by_name(k32, "SetLastError")));
    const auto get_err = reinterpret_cast<GetErrFn>(static_cast<uintptr_t>(by_name(k32, "GetLastError")));
    CHECK(tick != nullptr && set_err != nullptr && get_err != nullptr);
    if (tick == nullptr || set_err == nullptr || get_err == nullptr) {
        return;
    }
    const uint64_t t0 = tick();
    const uint64_t t1 = tick();
    CHECK(t1 >= t0);
    set_err(1234);
    CHECK(get_err() == 1234);
    set_err(0);

    // A shim from another family: XGameRuntime initialization state.
    const Pg6Handle gr = pg6_win32_get_module_handle_a("xgameruntime");
    using InitFn = int32_t (*)();
    const auto is_init =
        reinterpret_cast<InitFn>(static_cast<uintptr_t>(by_name(gr, "XGameRuntimeIsInitialized")));
    CHECK(is_init != nullptr);
    if (is_init != nullptr) {
        CHECK(is_init() == pg6_gdk_xgame_runtime_is_initialized());
    }
}

void test_unknown_names_fail() {
    const Pg6Handle k32 = pg6_win32_get_module_handle_a("kernel32");
    pg6_win32_set_last_error(0);
    CHECK(by_name(k32, "NoSuchExport") == 0);
    CHECK(pg6_win32_get_last_error() == ERROR_PROC_NOT_FOUND);

    // A kernel32 API must not leak into other modules.
    pg6_win32_set_last_error(0);
    CHECK(by_name(pg6_win32_get_module_handle_a("ws2_32"), "CreateFileW") == 0);
    CHECK(pg6_win32_get_last_error() == ERROR_PROC_NOT_FOUND);

    // Shim symbol spelling is not an export name.
    CHECK(by_name(k32, "pg6_win32_create_file_w") == 0);

    pg6_win32_set_last_error(0);
    CHECK(pg6_win32_get_module_handle_a("nosuchmodule.dll") == 0);
    CHECK(pg6_win32_get_last_error() == ERROR_MOD_NOT_FOUND);
}

}  // namespace

int main() {
    test_registration_idempotent();
    test_table_counts_and_lookup();
    test_module_lookup();
    test_samples_per_family();
    test_exports_are_callable();
    test_unknown_names_fail();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_shim_exports failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_shim_exports_tests passed.\n");
    return 0;
}
