#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/shim_exports.h"
#include "os/win32_loader.h"
#include "os/xasync_provider.h"
#include "os/xbox_middleware.h"
#include "os/xgameruntime.h"
#include "os/xgameruntime_provider.h"
#include "os/xgamesave_local.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xgameruntime_provider.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace cbk = playgta6::guest::callbacks;
namespace prov = playgta6::os::gdk::provider;
namespace win32 = playgta6::os::win32;
using com::Guid;
using playgta6::os::VirtualMemoryArena;

VirtualMemoryArena* g_arena = nullptr;

uint64_t load64(uint64_t va) {
    uint64_t value = 0;
    CHECK(g_arena->read(va, &value, sizeof(value)));
    return value;
}

void ret(GuestState* s) {
    s->gpr[GuestReg::Rsp] += 8;
}

// Guest callbacks (hand-written "lifted" functions).
constexpr uint64_t kRecord = 0x140020000ULL;     // void(void* ctx, bool canceled)
constexpr uint64_t kAsyncDone = 0x140020100ULL;  // void(XAsyncBlock* block)

std::atomic<int> g_record_calls{0};
std::atomic<uint64_t> g_record_ctx{0};
std::atomic<int> g_record_canceled{-1};
std::atomic<uint64_t> g_done_block{0};

void sub_record(GuestState* s) {
    g_record_ctx = s->gpr[GuestReg::Rcx];
    g_record_canceled = static_cast<int>(s->gpr[GuestReg::Rdx] & 0xFF);
    g_record_calls.fetch_add(1);
    ret(s);
}

void sub_async_done(GuestState* s) {
    g_done_block = s->gpr[GuestReg::Rcx];
    ret(s);
}

// A guest XAsync provider (HRESULT provider(XAsyncOp op, const XAsyncProviderData* data)) that
// drives the operation through the xgameruntime.dll XAsync interface like the game's statically
// linked libraries do: Begin schedules, DoWork returns E_PENDING (rescheduled) g_provider_pending
// times and then completes with an 8-byte payload, GetResult writes the payload.
constexpr uint64_t kProvider = 0x140020200ULL;
constexpr uint64_t kProviderRet = kProvider + 0x40;
constexpr uint64_t kWorkFn = 0x140020300ULL;  // HRESULT work(XAsyncBlock*): returns g_work_result
constexpr uint64_t kPayload = 0x0123456789ABCDEFULL;

uint64_t g_async_api = 0;
std::mutex g_provider_mutex;
std::string g_provider_ops;
int g_provider_pending = 0;
uint64_t g_seen_async = 0;
uint64_t g_seen_context = 0;
uint64_t g_seen_buffer_size = 0;
std::atomic<int> g_work_calls{0};
int32_t g_work_result = 0;

void store64(GuestState* /*s*/, uint64_t va, uint64_t value) {
    CHECK(g_arena->write(va, &value, sizeof(value)));
}

// `call [vtable + 8*slot]` on the XAsync interface from guest code (this + up to 3 arguments).
uint64_t guest_api_call(GuestState* s, uint32_t slot, std::initializer_list<uint64_t> args) {
    s->gpr[GuestReg::Rsp] -= 0x28;
    const uint64_t regs[3] = {GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    size_t i = 0;
    s->gpr[GuestReg::Rcx] = g_async_api;
    for (uint64_t a : args) {
        s->gpr[regs[i++]] = a;
    }
    s->gpr[GuestReg::Rsp] -= 8;
    store64(s, s->gpr[GuestReg::Rsp], kProviderRet);
    __playgta6_indirect_call(s, load64(load64(g_async_api) + 8ull * slot));
    s->gpr[GuestReg::Rsp] += 0x28;
    return s->gpr[GuestReg::Rax];
}

void provider_record(char c) {
    std::lock_guard<std::mutex> lock(g_provider_mutex);
    g_provider_ops.push_back(c);
}

std::string provider_ops() {
    std::lock_guard<std::mutex> lock(g_provider_mutex);
    return g_provider_ops;
}

void sub_provider(GuestState* s) {
    const uint32_t op = static_cast<uint32_t>(s->gpr[GuestReg::Rcx]);
    const uint64_t data = s->gpr[GuestReg::Rdx];
    const uint64_t async = load64(data);
    uint64_t rax = 0;
    switch (op) {
        case 0:  // Begin
            provider_record('B');
            g_seen_async = async;
            g_seen_context = load64(data + 24);
            rax = guest_api_call(s, 9, {async, 0});  // XAsyncSchedule(async, 0)
            break;
        case 1: {  // DoWork
            provider_record('W');
            bool again = false;
            {
                std::lock_guard<std::mutex> lock(g_provider_mutex);
                if (g_provider_pending > 0) {
                    --g_provider_pending;
                    again = true;
                }
            }
            if (again) {
                guest_api_call(s, 9, {async, 0});
                rax = static_cast<uint32_t>(playgta6::os::gdk::E_PENDING_HR);
            } else {
                guest_api_call(s, 10, {async, 0, 8});  // XAsyncComplete(async, S_OK, 8)
                rax = 0;
            }
            break;
        }
        case 2:  // GetResult
            provider_record('R');
            g_seen_buffer_size = load64(data + 8);
            store64(s, load64(data + 16), kPayload);
            break;
        case 3:  // Cancel
            provider_record('X');
            break;
        case 4:  // Cleanup
            provider_record('C');
            break;
        default:
            rax = static_cast<uint32_t>(com::E_INVALIDARG_HR);
    }
    s->gpr[GuestReg::Rax] = rax;
    ret(s);
}

void sub_work(GuestState* s) {
    g_work_calls.fetch_add(1);
    g_seen_async = s->gpr[GuestReg::Rcx];
    s->gpr[GuestReg::Rax] = static_cast<uint32_t>(g_work_result);
    ret(s);
}

// bool callback(const XGameSaveContainerInfo*, void* context): records the container name and
// blob count and stops after g_enum_stop_after calls.
constexpr uint64_t kEnumCb = 0x140020500ULL;
std::vector<std::string> g_enum_names;
std::vector<uint32_t> g_enum_blobs;
size_t g_enum_stop_after = 100;

void sub_enum_cb(GuestState* s) {
    const uint64_t info = s->gpr[GuestReg::Rcx];
    uint64_t name_ptr = 0;
    uint32_t blobs = 0;
    CHECK(g_arena->read(info, &name_ptr, sizeof(name_ptr)));
    CHECK(g_arena->read(info + 16, &blobs, sizeof(blobs)));
    char name[64] = {};
    CHECK(g_arena->read(name_ptr, name, sizeof(name) - 1));
    g_enum_names.push_back(name);
    g_enum_blobs.push_back(blobs);
    CHECK(s->gpr[GuestReg::Rdx] == 0xE11E);
    s->gpr[GuestReg::Rax] = g_enum_names.size() < g_enum_stop_after ? 1 : 0;
    ret(s);
}

// bool callback(const XGameSaveBlobInfo*, void* context): records name and size.
constexpr uint64_t kBlobCb = 0x140020600ULL;
std::vector<std::string> g_blob_names;
std::vector<uint32_t> g_blob_sizes;

void sub_blob_cb(GuestState* s) {
    const uint64_t info = s->gpr[GuestReg::Rcx];
    uint64_t name_ptr = 0;
    uint32_t size = 0;
    CHECK(g_arena->read(info, &name_ptr, sizeof(name_ptr)));
    CHECK(g_arena->read(info + 8, &size, sizeof(size)));
    char name[64] = {};
    CHECK(g_arena->read(name_ptr, name, sizeof(name) - 1));
    g_blob_names.push_back(name);
    g_blob_sizes.push_back(size);
    CHECK(s->gpr[GuestReg::Rdx] == 0xB10B);
    s->gpr[GuestReg::Rax] = 1;
    ret(s);
}

// void callback(void* context, const XNetworkingConnectivityHint* hint): records the hint.
constexpr uint64_t kHintCb = 0x140020700ULL;
std::atomic<int> g_hint_calls{0};
std::atomic<uint64_t> g_hint_ctx{0};
uint32_t g_hint_words[4] = {};

void sub_hint_cb(GuestState* s) {
    g_hint_ctx = s->gpr[GuestReg::Rcx];
    CHECK(g_arena->read(s->gpr[GuestReg::Rdx], g_hint_words, sizeof(g_hint_words)));
    g_hint_calls.fetch_add(1);
    ret(s);
}

const LiftedFunctionEntry kFunctions[] = {
    {kRecord, sub_record},
    {kAsyncDone, sub_async_done},
    {kProvider, sub_provider},
    {kWorkFn, sub_work},
    {kEnumCb, sub_enum_cb},
    {kBlobCb, sub_blob_cb},
    {kHintCb, sub_hint_cb},
};

struct GuestFault {
    FaultInfo info;
};

void throwing_handler(const FaultInfo* info) {
    throw GuestFault{*info};
}

bool diag_contains(const char* needle) {
    for (const std::string& d : com::diagnostics()) {
        if (d.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

template <typename Pred>
bool wait_for(Pred pred) {
    for (int i = 0; i < 3000; ++i) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}

struct Fixture;
uint64_t make_guest_block(Fixture& fx, uint64_t queue);

constexpr Guid kAsyncQueue = com::make_guid("073b7dcb-1fcf-4030-94be-e3c9eb623428");
constexpr Guid kUser = com::make_guid("01acd177-91f9-4763-a38e-ccbb55ce32e0");
constexpr Guid kUserExt = com::make_guid("cef4fac0-7676-4a94-a119-4c43f9eb5b74");
constexpr Guid kFeature = com::make_guid("8836fe87-edb9-4fe3-8dad-05f0d2cd5b40");
constexpr Guid kSystemProvider = com::make_guid("e349bd1a-fc20-4e40-b99c-4178cc6b409f");
constexpr Guid kSystemIid = com::make_guid("1861cf2e-e18b-4834-a9f5-b4a4e6efb4cf");
constexpr Guid kStoreProvider = com::make_guid("0dd112ac-7c24-448c-b92b-3960fb5bd30c");
constexpr Guid kStoreIid = com::make_guid("b09d803c-2414-4a05-82c6-66dfdc9e9a44");
constexpr Guid kStoreExtIid = com::make_guid("2d42fea5-e71d-4b76-97cd-c50afbb3ae5d");
constexpr Guid kPackageProvider = com::make_guid("af406016-e850-4aa8-a88d-2f3dcb9dac7e");
constexpr Guid kPackageIid = com::make_guid("0b9d556e-ab70-43f4-baa0-9999ac4cb040");
constexpr Guid kPlsProvider = com::make_guid("f4faf4d4-2d04-4fce-b3e0-474a713a3e84");
constexpr Guid kPlsIid = com::make_guid("41a4e10c-5a7e-41d9-8c37-37bde62a07d6");
constexpr Guid kHdrProvider = com::make_guid("03f0fe74-fdd9-4e5c-b630-f9339c47acc5");
constexpr Guid kHdrIid = com::make_guid("35f07670-706e-4bfb-9476-090798c5ebf3");
constexpr Guid kUnmappedProvider = com::make_guid("dfcd4649-4ff8-4043-ba07-35d607df98b0");
constexpr Guid kUnmappedIid = com::make_guid("eaf669df-5542-4590-99a3-8dc061f837cc");
constexpr Guid kGameProvider = com::make_guid("973a344e-24bf-4d0f-8457-56c534892b29");
constexpr Guid kGameIid = com::make_guid("2549f142-6419-4a06-97b5-931aab7c2f34");
constexpr Guid kNetworkingProvider = com::make_guid("37e56907-2f10-41e8-b72f-36edb185331a");
constexpr Guid kNetworkingIid = com::make_guid("bf2346b2-39af-4658-b5ea-44713c7e83b3");
constexpr Guid kBogus = com::make_guid("12345678-1234-1234-1234-123456789abc");

struct Fixture {
    VirtualMemoryArena arena;
    GuestThread thread{};
    uint64_t module = 0;
    uint64_t get_proc = 0;

    Fixture() {
        reset_guest_runtime();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        prov::reset_xgameruntime_provider();
        com::set_guest_memory(&arena);
        set_fault_handler(throwing_handler);
        g_arena = &arena;
        CHECK(register_lifted_functions(kFunctions));
        CHECK(create_guest_thread(arena, 256 * 1024, thread));
        module = prov::register_xgameruntime_provider();
        get_proc = register_host_callable(&pg6_gthunk_get_proc_address, "GetProcAddress");
    }

    ~Fixture() {
        prov::reset_xgameruntime_provider();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        reset_guest_runtime();
        g_arena = nullptr;
    }

    GuestState& state() { return thread.state; }
    void restore_stack() { thread.state.gpr[GuestReg::Rsp] = thread.stack_top; }

    uint64_t call(uint64_t target, std::initializer_list<uint64_t> args) {
        return call_guest(arena, thread.state, target, std::span<const uint64_t>(args.begin(), args.size()));
    }

    // `call [vtable + 8*slot]` with this = obj.
    uint64_t vcall(uint64_t obj, uint32_t slot, std::initializer_list<uint64_t> args = {}) {
        std::vector<uint64_t> all{obj};
        all.insert(all.end(), args.begin(), args.end());
        const uint64_t target = load64(load64(obj) + 8ull * slot);
        return call_guest(arena, thread.state, target, all);
    }

    uint64_t alloc(uint64_t size) { return com::guest_alloc(size); }

    uint64_t guid(const Guid& g) {
        const uint64_t va = alloc(16);
        CHECK(com::write_guest(va, g));
        return va;
    }

    uint64_t cstring(const char* text) {
        const uint64_t va = alloc(std::strlen(text) + 1);
        CHECK(com::write_guest(va, text, std::strlen(text) + 1));
        return va;
    }

    uint64_t proc(const char* name) { return call(get_proc, {module, cstring(name)}); }

    uint64_t query(const Guid& provider, const Guid& iid, int32_t* hr = nullptr) {
        const uint64_t out = alloc(8);
        const int32_t result =
            static_cast<int32_t>(call(prov::query_api_address(), {guid(provider), guid(iid), out}));
        if (hr != nullptr) {
            *hr = result;
        }
        return load64(out);
    }
};

int32_t hr32(uint64_t rax) {
    return static_cast<int32_t>(static_cast<uint32_t>(rax));
}

void test_module_and_exports() {
    Fixture fx;
    CHECK(fx.module != 0);
    CHECK(prov::register_xgameruntime_provider() == fx.module);  // idempotent
    CHECK(win32::LoadLibraryW(u"xgameruntime.dll") == fx.module);
    CHECK(win32::LoadLibraryExW(u"XGAMERUNTIME.DLL", 0, 0x800) == fx.module);
    for (const char* name : {"QueryApiImpl", "InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2",
                             "UninitializeApiImpl", "XErrorReport"}) {
        const uint64_t address = fx.proc(name);
        CHECK(is_synthetic_address(address));
    }
    CHECK(fx.proc("QueryApiImpl") == prov::query_api_address());
    CHECK(fx.proc("InitializeApiImpl") == prov::initialize_api_address());
    CHECK(fx.proc("UninitializeApiImpl") == prov::uninitialize_api_address());
    // Flat GDK shims registered by O11 are host-only: GetProcAddress does not hand them out.
    win32::register_all_shim_exports();
    com::clear_diagnostics();
    CHECK(fx.proc("XTaskQueueCreate") == 0);
    CHECK(diag_contains("host-only export"));

    // The client's initialization sequence: Ex2 first, then uninitialize at exit.
    CHECK(hr32(fx.call(fx.proc("InitializeApiImplEx2"), {0x0000633600002712ULL, 0x000065F400001EDDULL, 2, 0})) == 0);
    CHECK(pg6_gdk_xgame_runtime_is_initialized() == 1);
    CHECK(hr32(fx.call(fx.proc("InitializeApiImplEx"), {1, 2, 2})) == 0);
    CHECK(hr32(fx.call(fx.proc("InitializeApiImpl"), {0x6336, 0x2712})) == 0);
    for (int i = 0; i < 3; ++i) {
        fx.call(prov::uninitialize_api_address(), {});
    }
    CHECK(pg6_gdk_xgame_runtime_is_initialized() == 0);

    com::clear_diagnostics();
    fx.call(fx.proc("XErrorReport"), {0x89240100u, fx.cstring("runtime not initialized")});
    CHECK(diag_contains("XErrorReport hr=0x89240100 runtime not initialized"));
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

void test_query_api() {
    Fixture fx;
    int32_t hr = -1;
    const uint64_t q1 = fx.query(kAsyncQueue, kAsyncQueue, &hr);
    CHECK(hr == 0 && q1 != 0);
    const uint64_t q2 = fx.query(kAsyncQueue, kAsyncQueue, &hr);
    CHECK(q2 == q1);  // one singleton per API, AddRef'ed per query
    CHECK(fx.vcall(q1, 2) >= 1);  // Release, as every API wrapper does
    CHECK(fx.vcall(q2, 2) >= 1);
    // Same provider, different interface: distinct objects.
    const uint64_t user = fx.query(kUser, kUser, &hr);
    const uint64_t user_ext = fx.query(kUser, kUserExt, &hr);
    CHECK(hr == 0 && user != 0 && user_ext != 0 && user != user_ext);

    // Unknown pairs: REGDB_E_CLASSNOTREG, *out = 0, recorded once as a diagnostic.
    com::clear_diagnostics();
    CHECK(fx.query(kBogus, kBogus, &hr) == 0);
    CHECK(hr == prov::kUnknownApiResult);
    CHECK(fx.query(kUser, kBogus, &hr) == 0);
    CHECK(hr == prov::kUnknownApiResult);
    fx.query(kBogus, kBogus, &hr);
    CHECK(diag_contains("unknown API provider {12345678-1234-1234-1234-123456789ABC}"));
    bool found_unknown = false;
    for (const prov::ApiRequest& r : prov::api_requests()) {
        if (r.provider == com::format_guid(kBogus) && r.interface_id == com::format_guid(kBogus)) {
            found_unknown = !r.known && r.count == 2;
        }
    }
    CHECK(found_unknown);
    // Null out pointer / null GUID pointer.
    CHECK(hr32(fx.call(prov::query_api_address(), {fx.guid(kUser), fx.guid(kUser), 0})) == com::E_POINTER_HR);
    const uint64_t out = fx.alloc(8);
    CHECK(hr32(fx.call(prov::query_api_address(), {0, fx.guid(kUser), out})) == com::E_POINTER_HR);

    // Simple mapped slots.
    const uint64_t feature = fx.query(kFeature, kFeature);
    // Feature availability (public zero-based XGameRuntimeFeature numbering): the title asks only for
    // 10 (XNetworking, served) and 20 (Events, left false). Everything else stays false: store 14,
    // system 15, thread 17, user 18 and any unknown id.
    CHECK((fx.vcall(feature, 3, {10u}) & 0xFF) == 1);
    for (uint32_t id : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u, 20u,
                        21u, 22u, 23u, 1000u}) {
        CHECK((fx.vcall(feature, 3, {id}) & 0xFF) == 0);
    }
    const uint64_t package = fx.query(kPackageProvider, kPackageIid);
    CHECK((fx.vcall(package, 4) & 0xFF) == 1);
    {
        // XPackageGetCurrentProcessPackageIdentifier(33, buffer): 32 digits and a terminator.
        const uint64_t ident = fx.alloc(40);
        CHECK(hr32(fx.vcall(package, 3, {33, ident})) == 0);
        char text[33] = {};
        CHECK(com::read_guest(ident, text, sizeof(text)));
        CHECK(std::strlen(text) == 32);
        CHECK(hr32(fx.vcall(package, 3, {32, ident})) == playgta6::os::gdk::E_NOT_SUFFICIENT_BUFFER_HR);
        CHECK(hr32(fx.vcall(package, 3, {33, 0})) == com::E_INVALIDARG_HR);

        // Installation monitor: created without selectors, always complete and launchable.
        const uint64_t monitor_out = fx.alloc(8);
        CHECK(hr32(fx.vcall(package, 5, {ident, 0, 0, 5000, 0, monitor_out})) == 0);
        const uint64_t monitor = load64(monitor_out);
        CHECK(monitor != 0);
        const uint64_t progress = fx.alloc(24);
        fx.vcall(package, 7, {monitor, progress});
        CHECK(load64(progress) != 0 && load64(progress) == load64(progress + 8));
        uint8_t flags[2] = {};
        CHECK(com::read_guest(progress + 16, flags, sizeof(flags)));
        CHECK(flags[0] == 1 && flags[1] == 1);
        // Progress-changed registration hands out a token (the callback never fires).
        const uint64_t token = fx.alloc(8);
        CHECK(hr32(fx.vcall(package, 9, {monitor, 0x1234, kRecord, token})) == 0);
        CHECK(load64(token) != 0);
        CHECK(hr32(fx.vcall(package, 9, {monitor, 0x1234, 0, token})) == com::E_INVALIDARG_HR);
        CHECK(hr32(fx.vcall(package, 9, {0x5555, 0x1234, kRecord, token})) == com::E_INVALIDARG_HR);
        // Selectors without an array, and a missing out pointer, are rejected.
        CHECK(hr32(fx.vcall(package, 5, {ident, 2, 0, 5000, 0, monitor_out})) == com::E_INVALIDARG_HR);
        CHECK(load64(monitor_out) == 0);
        CHECK(hr32(fx.vcall(package, 5, {ident, 0, 0, 5000, 0, 0})) == com::E_POINTER_HR);
        fx.vcall(package, 6, {monitor});

        // XPackageEnumeratePackages: no content packages (the callback never runs); game
        // packages are not modelled.
        g_record_calls = 0;
        CHECK(hr32(fx.vcall(package, 39, {1, 0, 0xABC, kRecord})) == 0);
        CHECK(g_record_calls == 0);
        CHECK(hr32(fx.vcall(package, 39, {0, 0, 0xABC, kRecord})) == com::E_NOTIMPL_HR);
        CHECK(hr32(fx.vcall(package, 39, {1, 0, 0xABC, 0})) == com::E_INVALIDARG_HR);

        // Chunk slots: a fully installed package.
        const uint64_t sel = fx.alloc(16);
        const uint64_t out = fx.alloc(16);
        CHECK(hr32(fx.vcall(package, 11, {16, out})) == 0);
        char loc[8] = {};
        CHECK(com::read_guest(out, loc, sizeof(loc)) && std::strcmp(loc, "en-US") == 0);
        CHECK(hr32(fx.vcall(package, 11, {2, out})) == playgta6::os::gdk::E_NOT_SUFFICIENT_BUFFER_HR);
        CHECK(hr32(fx.vcall(package, 11, {16, 0})) == com::E_INVALIDARG_HR);
        CHECK(com::write_guest(out, uint32_t{9}));
        CHECK(hr32(fx.vcall(package, 12, {ident, 1, sel, out})) == 0);
        uint32_t avail = 9;
        CHECK(com::read_guest(out, avail) && avail == 0);  // Ready
        CHECK(hr32(fx.vcall(package, 12, {ident, 1, 0, out})) == com::E_INVALIDARG_HR);
        CHECK(hr32(fx.vcall(package, 12, {ident, 1, sel, 0})) == com::E_POINTER_HR);
        CHECK(hr32(fx.vcall(package, 14, {ident, 1, sel})) == 0);
        CHECK(hr32(fx.vcall(package, 14, {ident, 1, 0})) == com::E_INVALIDARG_HR);
        CHECK(com::write_guest(out, uint64_t{77}) && com::write_guest(out + 8, uint8_t{1}));
        CHECK(hr32(fx.vcall(package, 18, {ident, 1, sel, out, out + 8})) == 0);
        CHECK(load64(out) == 0);
        uint8_t confirm = 1;
        CHECK(com::read_guest(out + 8, confirm) && confirm == 0);
        CHECK(hr32(fx.vcall(package, 18, {ident, 1, sel, 0, 0})) == com::E_POINTER_HR);
        CHECK(hr32(fx.vcall(package, 19, {ident, 1, sel})) == 0);
        // s34: bool(name): installed query, true only for the title's own package identifier.
        CHECK((fx.vcall(package, 34, {ident}) & 0xFF) == 1);
        CHECK((fx.vcall(package, 34, {0}) & 0xFF) == 0);
        CHECK((fx.vcall(package, 34, {sel}) & 0xFF) == 0);
        {
            // s16 / s17 install chunks: completes, hands out a finished monitor.
            const uint64_t queue_api = fx.query(kAsyncQueue, kAsyncQueue);
            const uint64_t qout = fx.alloc(8);
            CHECK(hr32(fx.vcall(queue_api, 12, {0, 0, qout})) == 0);
            const uint64_t queue = load64(qout);
            const uint64_t block = make_guest_block(fx, queue);
            CHECK(hr32(fx.vcall(package, 16, {ident, 1, sel, 5000, 0, block})) == 0);
            g_done_block = 0;
            CHECK((fx.vcall(queue_api, 16, {queue, 0, 0}) & 0xFF) == 1);
            CHECK((fx.vcall(queue_api, 16, {queue, 1, 0}) & 0xFF) == 1);
            CHECK(g_done_block == block);
            const uint64_t mon = fx.alloc(8);
            CHECK(hr32(fx.vcall(package, 17, {block, mon})) == 0);
            CHECK(load64(mon) != 0);
            const uint64_t progress = fx.alloc(24);
            fx.vcall(package, 7, {load64(mon), progress});
            uint8_t flags[2] = {};
            CHECK(com::read_guest(progress + 16, flags, sizeof(flags)) && flags[0] == 1 && flags[1] == 1);
            fx.vcall(package, 6, {load64(mon)});
            CHECK(hr32(fx.vcall(package, 16, {ident, 1, sel, 5000, 0, 0})) == com::E_INVALIDARG_HR);

            // s37 / s38 / s25 / s26: mount the title's own package at the game-data root.
            const uint64_t mblock = make_guest_block(fx, queue);
            CHECK(hr32(fx.vcall(package, 37, {ident, mblock})) == 0);
            g_done_block = 0;
            CHECK((fx.vcall(queue_api, 16, {queue, 0, 0}) & 0xFF) == 1);
            CHECK((fx.vcall(queue_api, 16, {queue, 1, 0}) & 0xFF) == 1);
            CHECK(g_done_block == mblock);
            const uint64_t mount_out = fx.alloc(8);
            CHECK(hr32(fx.vcall(package, 38, {mblock, mount_out})) == 0);
            const uint64_t mount = load64(mount_out);
            CHECK(mount != 0);
            const uint64_t mpath = fx.alloc(256);
            CHECK(hr32(fx.vcall(package, 25, {mount, 256, mpath})) == 0);
            char mtext[8] = {};
            CHECK(com::read_guest(mpath, mtext, sizeof(mtext)) && std::strcmp(mtext, "G:\\") == 0);
            CHECK(hr32(fx.vcall(package, 25, {mount, 2, mpath})) == playgta6::os::gdk::E_NOT_SUFFICIENT_BUFFER_HR);
            CHECK(hr32(fx.vcall(package, 25, {mount, 256, 0})) == com::E_INVALIDARG_HR);
            CHECK(hr32(fx.vcall(package, 25, {0x5555, 256, mpath})) == com::E_INVALIDARG_HR);
            fx.vcall(package, 26, {mount});
            CHECK(hr32(fx.vcall(package, 25, {mount, 256, mpath})) == com::E_INVALIDARG_HR);  // closed
            CHECK(hr32(fx.vcall(package, 38, {mblock, 0})) == com::E_POINTER_HR);
            // Only the title's own identifier mounts; arguments are validated.
            const uint64_t other = fx.cstring("11111111111111111111111111111111");
            CHECK(hr32(fx.vcall(package, 37, {other, mblock})) == static_cast<int32_t>(0x80070490u));
            CHECK(hr32(fx.vcall(package, 37, {0, mblock})) == com::E_INVALIDARG_HR);
            CHECK(hr32(fx.vcall(package, 37, {ident, 0})) == com::E_INVALIDARG_HR);
        }
    }
    {
        // XPersistentLocalStorageGetPath(size, path, used): a /userdata overlay directory.
        const uint64_t pls = fx.query(kPlsProvider, kPlsIid);
        const uint64_t path = fx.alloc(256);
        const uint64_t used = fx.alloc(8);
        CHECK(hr32(fx.vcall(pls, 4, {256, path, 0})) == 0);
        char text[64] = {};
        CHECK(com::read_guest(path, text, sizeof(text)));
        CHECK(std::strcmp(text, "T:\\userdata\\pls\\") == 0);
        playgta6::os::win32::UserdataEntry entry;
        CHECK(playgta6::os::win32::userdata_stat("/userdata/pls", &entry) == 0 && entry.is_directory);
        CHECK(hr32(fx.vcall(pls, 4, {4, path, used})) == com::E_FAIL_HR);
        CHECK(load64(used) == std::strlen(text) + 1);
        // s3 path size matches.
        CHECK(hr32(fx.vcall(pls, 3, {used})) == 0 && load64(used) == std::strlen(text) + 1);
        CHECK(hr32(fx.vcall(pls, 3, {0})) == com::E_POINTER_HR);

        // s5 space info: used from the real contents, free = quota - used.
        const uint64_t info = fx.alloc(32);
        const uint64_t quota = 2048ull * 1024 * 1024;
        playgta6::os::win32::clear_userdata_overlay();
        CHECK(hr32(fx.vcall(pls, 5, {info})) == 0);
        CHECK(load64(info) == quota && load64(info + 8) == quota && load64(info + 16) == 0 &&
              load64(info + 24) == quota);
        playgta6::os::win32::userdata_create_directory("/userdata/pls");
        playgta6::os::win32::write_userdata_file("/userdata/pls/a.bin", std::vector<uint8_t>(1000, 7));
        CHECK(hr32(fx.vcall(pls, 5, {info})) == 0);
        CHECK(load64(info + 16) == 1000 && load64(info) == quota - 1000 && load64(info + 24) == quota);
        CHECK(hr32(fx.vcall(pls, 5, {0})) == com::E_POINTER_HR);

        // s6 / s7 make room: granted when the bytes fit, E_ABORT otherwise.
        const uint64_t queue_api = fx.query(kAsyncQueue, kAsyncQueue);
        const uint64_t qout = fx.alloc(8);
        CHECK(hr32(fx.vcall(queue_api, 12, {0, 0, qout})) == 0);
        const uint64_t queue = load64(qout);
        auto run = [&](uint64_t bytes) {
            const uint64_t block = make_guest_block(fx, queue);
            CHECK(hr32(fx.vcall(pls, 6, {bytes, block})) == 0);
            g_done_block = 0;
            CHECK((fx.vcall(queue_api, 16, {queue, 0, 0}) & 0xFF) == 1);
            CHECK((fx.vcall(queue_api, 16, {queue, 1, 0}) & 0xFF) == 1);
            CHECK(g_done_block == block);
            return hr32(fx.vcall(pls, 7, {block}));
        };
        CHECK(run(quota - 1000) == 0);
        CHECK(run(quota - 999) == playgta6::os::gdk::E_ABORT_HR);
        CHECK(hr32(fx.vcall(pls, 6, {1, 0})) == com::E_INVALIDARG_HR);
        playgta6::os::win32::clear_userdata_overlay();
    }
    {
        // {03f0fe74} slot 4 (XDisplayTryEnableHdrMode shape): Disabled, zeroed info.
        const uint64_t hdr = fx.query(kHdrProvider, kHdrIid);
        const uint64_t info = fx.alloc(12);
        const float ones[3] = {1.0f, 1.0f, 1.0f};
        CHECK(com::write_guest(info, ones, sizeof(ones)));
        CHECK(static_cast<uint32_t>(fx.vcall(hdr, 4, {0, info})) == 2);
        float back[3] = {};
        CHECK(com::read_guest(info, back, sizeof(back)));
        CHECK(back[0] == 0.0f && back[1] == 0.0f && back[2] == 0.0f);
        CHECK(static_cast<uint32_t>(fx.vcall(hdr, 4, {0, 0})) == 2);
    }
    {
        // XSpeechSynthesizer: silent synthesizer. Fixed handle, empty 16 kHz mono WAVE streams.
        // Pins the slot map read from the executable's wrappers: s8 EnumerateInstalledVoices,
        // s9 Create, s10 CloseHandle, s11 SetDefaultVoice, s12 SetCustomVoice, s13 FromText,
        // s14 CloseStream, s15 GetStreamDataSize, s16 GetStreamData, s21 FromSsml. (A guessed map
        // without s8 trapped the boot once Create started succeeding.)
        namespace gdk = playgta6::os::gdk;
        const uint64_t speech = fx.query(com::make_guid("3e241977-6237-41e9-8559-102c2d9983f1"),
                                         com::make_guid("d722b373-8c4d-4692-9e51-d6ad9b37aa7d"));
        // s8: no installed voices, so the callback never runs.
        g_record_calls = 0;
        CHECK(hr32(fx.vcall(speech, 8, {0x1234, kRecord})) == gdk::S_OK_HR);
        CHECK(g_record_calls == 0);
        CHECK(hr32(fx.vcall(speech, 8, {0, kRecord})) == gdk::S_OK_HR);
        CHECK(g_record_calls == 0);
        CHECK(hr32(fx.vcall(speech, 8, {0x1234, 0})) == com::E_INVALIDARG_HR);
        CHECK(g_record_calls == 0);
        const uint64_t handle = fx.alloc(8);
        CHECK(com::write_guest(handle, uint64_t{0x1234}));
        CHECK(hr32(fx.vcall(speech, 9, {handle})) == gdk::S_OK_HR);
        const uint64_t h = load64(handle);
        CHECK(h != 0 && h != 0x1234);
        CHECK(hr32(fx.vcall(speech, 9, {0})) == com::E_INVALIDARG_HR);
        CHECK(hr32(fx.vcall(speech, 11, {h})) == gdk::S_OK_HR);
        CHECK(hr32(fx.vcall(speech, 12, {h, fx.cstring("voice")})) == com::E_INVALIDARG_HR);
        const uint64_t stream_out = fx.alloc(8);
        CHECK(hr32(fx.vcall(speech, 21, {h, 0, stream_out})) == gdk::S_OK_HR);
        CHECK(load64(stream_out) != 0);
        CHECK(com::write_guest(stream_out, uint64_t{0}));
        CHECK(hr32(fx.vcall(speech, 13, {h, 0, stream_out})) == gdk::S_OK_HR);
        const uint64_t stream = load64(stream_out);
        CHECK(stream != 0);
        CHECK(hr32(fx.vcall(speech, 13, {h, 0, 0})) == com::E_INVALIDARG_HR);
        const uint64_t size = fx.alloc(8);
        CHECK(hr32(fx.vcall(speech, 15, {stream, size})) == gdk::S_OK_HR);
        CHECK(load64(size) == 44);
        const uint64_t buf = fx.alloc(64);
        const uint64_t used = fx.alloc(8);
        CHECK(hr32(fx.vcall(speech, 16, {stream, 10, buf, used})) == gdk::E_NOT_SUFFICIENT_BUFFER_HR);
        CHECK(hr32(fx.vcall(speech, 16, {stream, 64, buf, used})) == gdk::S_OK_HR);
        CHECK(load64(used) == 44);
        uint8_t wav[44] = {};
        CHECK(com::read_guest(buf, wav, sizeof(wav)));
        CHECK(std::memcmp(wav, "RIFF", 4) == 0 && std::memcmp(wav + 8, "WAVE", 4) == 0);
        CHECK(wav[22] == 1 && wav[24] == 0x80 && wav[25] == 0x3e && wav[34] == 16);
        CHECK(wav[40] == 0 && wav[41] == 0 && wav[42] == 0 && wav[43] == 0);
        CHECK(hr32(fx.vcall(speech, 14, {stream})) == gdk::S_OK_HR);
        CHECK(hr32(fx.vcall(speech, 10, {h})) == gdk::S_OK_HR);
        // s17 is not part of the map: it still traps.
        bool trapped = false;
        try {
            fx.vcall(speech, 17, {0});
        } catch (const GuestFault& f) {
            trapped = f.info.kind == static_cast<uint32_t>(FaultKind::UnimplementedSlot) && f.info.detail == 17;
        }
        fx.restore_stack();
        CHECK(trapped);
    }
    {
        // XGameSave (O34): a local provider. s3 hands out a handle, s4 completes through the
        // XAsync queue and s5 returns the handle, s6 closes it. A null block / user is invalid.
        namespace gs = playgta6::os::gdk::gamesave;
        const std::filesystem::path dir = std::filesystem::temp_directory_path() / "pg6_o34_provider_test";
        std::filesystem::remove_all(dir);
        gs::set_save_dir(dir.string());
        const uint64_t save = fx.query(com::make_guid("704c3f58-e629-4cc2-b197-30511b996fe2"),
                                       com::make_guid("1bfff3af-f14a-40a3-8e35-9ada906593f9"));
        const uint64_t provider = fx.alloc(8);
        const uint64_t cfg = fx.cstring("00000000-0000-0000-0000-000000000001");
        CHECK(com::write_guest(provider, uint64_t{0x1234}));
        CHECK(hr32(fx.vcall(save, 3, {0x10, cfg, 0, provider})) == 0);
        const uint64_t handle = load64(provider);
        CHECK(handle != 0 && handle != 0x1234);
        CHECK(hr32(fx.vcall(save, 3, {0x10, cfg, 0, 0})) == com::E_POINTER_HR);
        CHECK(hr32(fx.vcall(save, 3, {0, cfg, 0, provider})) == com::E_INVALIDARG_HR);
        CHECK(load64(provider) == 0);
        fx.vcall(save, 6, {handle});

        const uint64_t queue_api = fx.query(kAsyncQueue, kAsyncQueue);
        const uint64_t qout = fx.alloc(8);
        CHECK(hr32(fx.vcall(queue_api, 12, {0, 0, qout})) == 0);
        const uint64_t queue = load64(qout);
        const uint64_t block = make_guest_block(fx, queue);
        CHECK(hr32(fx.vcall(save, 4, {0x10, cfg, 0, block})) == 0);
        CHECK(hr32(fx.vcall(queue_api, 3, {block, 0})) == playgta6::os::gdk::E_PENDING_HR);
        CHECK(hr32(fx.vcall(save, 5, {block, provider})) == playgta6::os::gdk::E_PENDING_HR);
        g_done_block = 0;
        auto dispatch = [&](uint32_t port) { return (fx.vcall(queue_api, 16, {queue, port, 0}) & 0xFF) == 1; };
        CHECK(dispatch(0));  // DoWork
        CHECK(dispatch(1));  // completion callback
        CHECK(g_done_block == block);
        CHECK(hr32(fx.vcall(queue_api, 3, {block, 0})) == 0);
        CHECK(hr32(fx.vcall(save, 5, {block, provider})) == 0);
        const uint64_t async_handle = load64(provider);
        CHECK(async_handle != 0);
        fx.vcall(save, 6, {async_handle});
        CHECK(hr32(fx.vcall(save, 4, {0x10, cfg, 0, 0})) == com::E_INVALIDARG_HR);

        // s13 / s14: container enumeration through the guest callback (context passed through).
        {
            const uint64_t p2 = fx.alloc(8);
            CHECK(hr32(fx.vcall(save, 3, {0x10, cfg, 0, p2})) == 0);
            const uint64_t h2 = load64(p2);
            const uint64_t player = fx.cstring("Player");
            g_enum_names.clear();
            CHECK(hr32(fx.vcall(save, 13, {h2, player, 0xE11E, kEnumCb})) == 0);  // empty store
            CHECK(g_enum_names.empty());
            gs::Store store = gs::open_store("00000000-0000-0000-0000-000000000001");
            CHECK(store.create_container("Player") == 0);
            CHECK(store.create_container("Settings") == 0);
            CHECK(store.submit_update("Player", {{"a", {1, 2, 3}}}, {}) == 0);
            CHECK(hr32(fx.vcall(save, 13, {h2, player, 0xE11E, kEnumCb})) == 0);
            CHECK(g_enum_names.size() == 1 && g_enum_names[0] == "Player" && g_enum_blobs[0] == 1);
            g_enum_names.clear();
            CHECK(hr32(fx.vcall(save, 14, {h2, 0xE11E, kEnumCb})) == 0);
            CHECK(g_enum_names.size() == 2 && g_enum_names[1] == "Settings");
            g_enum_names.clear();
            g_enum_stop_after = 1;  // the callback returns false: enumeration stops
            CHECK(hr32(fx.vcall(save, 14, {h2, 0xE11E, kEnumCb})) == 0 && g_enum_names.size() == 1);
            g_enum_stop_after = 100;
            CHECK(hr32(fx.vcall(save, 14, {0x9999, 0xE11E, kEnumCb})) == com::E_INVALIDARG_HR);
            CHECK(hr32(fx.vcall(save, 14, {h2, 0xE11E, 0})) == com::E_INVALIDARG_HR);
            fx.vcall(save, 6, {h2});
            CHECK(hr32(fx.vcall(save, 14, {h2, 0xE11E, kEnumCb})) == com::E_INVALIDARG_HR);  // closed
        }

        // R40: containers, blobs and updates (s16..s29) as the title's save manager uses them.
        {
            namespace gsr = playgta6::os::gdk::gamesave;
            const int32_t kIllegalCall = static_cast<int32_t>(0x8000000Eu);
            const uint64_t p3 = fx.alloc(8);
            CHECK(hr32(fx.vcall(save, 3, {0x10, cfg, 0, p3})) == 0);
            const uint64_t prov3 = load64(p3);
            const uint64_t out = fx.alloc(8);
            auto u32_at = [&](uint64_t va) {
                uint32_t v = 0;
                CHECK(com::read_guest(va, v));
                return v;
            };
            auto run_queue = [&](uint64_t blk) {
                g_done_block = 0;
                CHECK((fx.vcall(queue_api, 16, {queue, 0, 0}) & 0xFF) == 1);  // DoWork
                CHECK((fx.vcall(queue_api, 16, {queue, 1, 0}) & 0xFF) == 1);  // completion
                CHECK(g_done_block == blk);
            };
            // s16: a handle for a new name; nothing is created until an update is submitted.
            const uint64_t slot_name = fx.cstring("Slot1");
            CHECK(hr32(fx.vcall(save, 16, {prov3, slot_name, out})) == 0);
            const uint64_t c1 = load64(out);
            CHECK(c1 != 0);
            g_enum_names.clear();
            CHECK(hr32(fx.vcall(save, 13, {prov3, slot_name, 0xE11E, kEnumCb})) == 0 && g_enum_names.empty());
            CHECK(hr32(fx.vcall(save, 16, {prov3, fx.cstring(""), out})) == gsr::E_GS_INVALID_CONTAINER_NAME);
            CHECK(load64(out) == 0);
            CHECK(hr32(fx.vcall(save, 16, {0x777, slot_name, out})) == com::E_INVALIDARG_HR);
            CHECK(hr32(fx.vcall(save, 16, {prov3, slot_name, 0})) == com::E_POINTER_HR);
            const uint64_t header_name = fx.cstring("Save Game Header");
            g_blob_names.clear();
            CHECK(hr32(fx.vcall(save, 19, {c1, header_name, 0xB10B, kBlobCb})) == 0 && g_blob_names.empty());

            // s23/s25/s26/s28/s29: header + payload through the async submit.
            CHECK(hr32(fx.vcall(save, 23, {c1, fx.cstring("Slot one"), out})) == 0);
            const uint64_t u1 = load64(out);
            std::vector<uint8_t> header(0x128);
            for (size_t i = 0; i < header.size(); ++i) {
                header[i] = static_cast<uint8_t>(i);
            }
            const uint64_t hdr_data = fx.alloc(header.size());
            CHECK(com::write_guest(hdr_data, header.data(), header.size()));
            const uint64_t payload = fx.cstring("payload-bytes");
            CHECK(hr32(fx.vcall(save, 25, {u1, header_name, hdr_data, header.size()})) == 0);
            CHECK(hr32(fx.vcall(save, 25, {u1, fx.cstring("Save Game Data"), payload, 13})) == 0);
            CHECK(hr32(fx.vcall(save, 25, {u1, fx.cstring(""), payload, 13})) == com::E_INVALIDARG_HR);
            CHECK(hr32(fx.vcall(save, 25, {u1, fx.cstring("Junk"), payload, 3})) == 0);
            CHECK(hr32(fx.vcall(save, 26, {u1, fx.cstring("Junk")})) == 0);  // the delete drops the write
            const uint64_t b2 = make_guest_block(fx, queue);
            CHECK(hr32(fx.vcall(save, 28, {u1, b2})) == 0);
            CHECK(hr32(fx.vcall(save, 29, {b2})) == playgta6::os::gdk::E_PENDING_HR);
            run_queue(b2);
            CHECK(hr32(fx.vcall(save, 29, {b2})) == 0);
            CHECK(hr32(fx.vcall(save, 25, {u1, header_name, hdr_data, 4})) == kIllegalCall);  // submitted
            fx.vcall(save, 24, {u1});
            CHECK(hr32(fx.vcall(save, 27, {u1})) == com::E_INVALIDARG_HR);  // closed

            // On disk: the container exists now, with its display name and two blobs.
            g_enum_names.clear();
            g_enum_blobs.clear();
            CHECK(hr32(fx.vcall(save, 13, {prov3, slot_name, 0xE11E, kEnumCb})) == 0);
            CHECK(g_enum_names.size() == 1 && g_enum_names[0] == "Slot1" && g_enum_blobs[0] == 2);
            std::vector<gsr::ContainerInfo> infos;
            gs::Store store = gs::open_store("00000000-0000-0000-0000-000000000001");
            CHECK(store.enumerate_containers("Slot1", &infos) == 0);
            CHECK(infos.size() == 1 && infos[0].display_name == "Slot one" && infos[0].total_size == 0x128 + 13);

            // s19: the title's header lookup (prefix match, size reported); s18 lists all.
            g_blob_names.clear();
            g_blob_sizes.clear();
            CHECK(hr32(fx.vcall(save, 19, {c1, header_name, 0xB10B, kBlobCb})) == 0);
            CHECK(g_blob_names.size() == 1 && g_blob_names[0] == "Save Game Header" && g_blob_sizes[0] == 0x128);
            g_blob_names.clear();
            CHECK(hr32(fx.vcall(save, 18, {c1, 0xB10B, kBlobCb})) == 0 && g_blob_names.size() == 2);

            // s20: read the header into an exactly sized buffer (24 + strlen + 1 + size).
            const uint64_t names = fx.alloc(16);
            CHECK(com::write_guest(names, header_name));
            const uint64_t count = fx.alloc(4);
            const uint64_t need = 24 + 16 + 1 + 0x128;
            const uint64_t buf = fx.alloc(need);
            CHECK(com::write_guest(count, uint32_t{1}));
            CHECK(hr32(fx.vcall(save, 20, {c1, names, count, need - 1, buf})) == gsr::E_GS_PROVIDED_BUFFER_TOO_SMALL);
            CHECK(com::write_guest(count, uint32_t{1}));
            CHECK(hr32(fx.vcall(save, 20, {c1, names, count, need, buf})) == 0);
            CHECK(u32_at(count) == 1);
            CHECK(u32_at(buf + 8) == 0x128);
            const uint64_t data_ptr = load64(buf + 16);
            CHECK(data_ptr >= buf + 24 && data_ptr + 0x128 <= buf + need);
            std::vector<uint8_t> back(0x128);
            CHECK(com::read_guest(data_ptr, back.data(), back.size()) && back == header);
            char got[32] = {};
            CHECK(com::read_guest(load64(buf), got, 17) && std::strcmp(got, "Save Game Header") == 0);
            CHECK(com::write_guest(names + 8, fx.cstring("Missing")));
            CHECK(com::write_guest(count, uint32_t{2}));
            CHECK(hr32(fx.vcall(save, 20, {c1, names, count, 4096, fx.alloc(4096)})) == gsr::E_GS_BLOB_NOT_FOUND);
            CHECK(u32_at(count) == 0);

            // s21/s22: async read of both blobs; the buffer size comes from XAsyncGetResultSize (s4).
            CHECK(com::write_guest(names + 8, fx.cstring("Save Game Data")));
            const uint64_t b3 = make_guest_block(fx, queue);
            CHECK(hr32(fx.vcall(save, 21, {c1, names, 2, b3})) == 0);
            run_queue(b3);
            const uint64_t size_out = fx.alloc(8);
            CHECK(hr32(fx.vcall(queue_api, 4, {b3, size_out})) == 0);
            const uint64_t need2 = need + 24 + 15 + 13;
            CHECK(load64(size_out) == need2);
            const uint64_t buf2 = fx.alloc(need2);
            CHECK(hr32(fx.vcall(save, 22, {b3, need2 - 1, buf2, count})) == gsr::E_GS_PROVIDED_BUFFER_TOO_SMALL);
            CHECK(hr32(fx.vcall(save, 22, {b3, need2, buf2, count})) == 0);
            CHECK(u32_at(count) == 2);
            CHECK(u32_at(buf2 + 24 + 8) == 13);
            char text[16] = {};
            CHECK(com::read_guest(load64(buf2 + 24 + 16), text, 13) && std::strcmp(text, "payload-bytes") == 0);

            // s17 closes; s8/s9 quota; s11/s12 delete; a closed handle is invalid.
            fx.vcall(save, 17, {c1});
            CHECK(hr32(fx.vcall(save, 19, {c1, header_name, 0xB10B, kBlobCb})) == com::E_INVALIDARG_HR);
            const uint64_t b4 = make_guest_block(fx, queue);
            CHECK(hr32(fx.vcall(save, 8, {prov3, b4})) == 0);
            run_queue(b4);
            const uint64_t quota = fx.alloc(8);
            CHECK(hr32(fx.vcall(save, 9, {b4, quota})) == 0);
            CHECK(load64(quota) == gsr::kDefaultQuotaBytes - (0x128 + 13 + 3));  // "Player" holds 3 bytes
            const uint64_t b5 = make_guest_block(fx, queue);
            CHECK(hr32(fx.vcall(save, 11, {prov3, slot_name, b5})) == 0);
            run_queue(b5);
            CHECK(hr32(fx.vcall(save, 12, {b5})) == 0);
            g_enum_names.clear();
            CHECK(hr32(fx.vcall(save, 13, {prov3, slot_name, 0xE11E, kEnumCb})) == 0 && g_enum_names.empty());

            // An oversized blob is refused up front; an empty update creates the container.
            CHECK(hr32(fx.vcall(save, 16, {prov3, fx.cstring("Fresh"), out})) == 0);
            const uint64_t c2 = load64(out);
            CHECK(hr32(fx.vcall(save, 23, {c2, 0, out})) == 0);
            const uint64_t u2 = load64(out);
            CHECK(hr32(fx.vcall(save, 25, {u2, header_name, hdr_data, gsr::kMaxBlobBytes + 1})) == com::E_INVALIDARG_HR);
            CHECK(hr32(fx.vcall(save, 27, {u2})) == 0);
            CHECK(store.enumerate_containers("Fresh", &infos) == 0 && infos.size() == 1 && infos[0].blob_count == 0);
            fx.vcall(save, 24, {u2});
            fx.vcall(save, 17, {c2});
            fx.vcall(save, 6, {prov3});
        }
        std::filesystem::remove_all(dir);
        gs::set_save_dir("");
    }
    {
        // {b884675d} slot 3: XSystemGetAnalyticsInfo through the hidden result pointer.
        const Guid analytics = com::make_guid("b884675d-b738-4a9c-815d-9a9a1e0c6c9b");
        const uint64_t api = fx.query(analytics, analytics);
        const uint64_t out = fx.alloc(0x90);
        CHECK(fx.vcall(api, 3, {out}) == out);
        char info[0x90] = {};
        CHECK(com::read_guest(out, info, sizeof(info)));
        uint16_t major = 0;
        std::memcpy(&major, info, 2);
        CHECK(major == 10);
        CHECK(std::strcmp(info + 16, "Windows.Xbox") == 0);
        CHECK(std::strcmp(info + 80, "Xbox Series X") == 0);
    }
    const uint64_t system = fx.query(kSystemProvider, kSystemIid);
    const uint64_t buffer = fx.alloc(64);
    const uint64_t used = fx.alloc(8);
    CHECK(hr32(fx.vcall(system, 3, {64, buffer, used})) == 0);
    CHECK(load64(used) == 39);
    char id[39] = {};
    CHECK(com::read_guest(buffer, id, sizeof(id)));
    CHECK(std::strlen(id) == 38);
    CHECK(hr32(fx.vcall(system, 3, {8, buffer, used})) == playgta6::os::gdk::E_NOT_SUFFICIENT_BUFFER_HR);
    // XSystemGetXboxLiveSandboxId (s4), no sandbox configured: E_NOTIMPL, empty string, used 0.
    CHECK(com::write_guest(used, uint64_t{77}));
    CHECK(hr32(fx.vcall(system, 4, {0x20, buffer, used})) == com::E_NOTIMPL_HR);
    CHECK(load64(used) == 0);
    char sandbox = 'x';
    CHECK(com::read_guest(buffer, &sandbox, 1) && sandbox == '\0');
    // A configured sandbox (a retail console: "RETAIL"): terminated string, used = length + 1;
    // a buffer that is too small fails with the size needed.
    prov::set_xbox_live_sandbox_id("RETAIL");
    CHECK(hr32(fx.vcall(system, 4, {0x20, buffer, used})) == 0);
    CHECK(load64(used) == 7);
    char text[8] = {};
    CHECK(com::read_guest(buffer, text, sizeof(text)));
    CHECK(std::strcmp(text, "RETAIL") == 0);
    CHECK(hr32(fx.vcall(system, 4, {6, buffer, used})) == playgta6::os::gdk::E_NOT_SUFFICIENT_BUFFER_HR);
    CHECK(load64(used) == 7);
    CHECK(hr32(fx.vcall(system, 4, {0x20, 0, used})) == com::E_INVALIDARG_HR);
    prov::set_xbox_live_sandbox_id("");
    CHECK(hr32(fx.vcall(system, 4, {0x20, buffer, used})) == com::E_NOTIMPL_HR);
}

void test_task_queue_slots() {
    Fixture fx;
    const uint64_t api = fx.query(kAsyncQueue, kAsyncQueue);
    fx.vcall(api, 29);  // XThreadAssertNotTimeSensitive: no-op
    const uint64_t out = fx.alloc(8);
    // XTaskQueueCreate(Manual, Manual, &queue) with the GDK enum values.
    CHECK(hr32(fx.vcall(api, 12, {0, 0, out})) == 0);
    const uint64_t queue = load64(out);
    CHECK(queue != 0);

    // SubmitDelayedCallback (no delay) + Dispatch: the guest callback runs via call_guest.
    g_record_calls = 0;
    CHECK(hr32(fx.vcall(api, 19, {queue, 0, 0, 0xABC, kRecord})) == 0);
    CHECK(g_record_calls == 0);
    CHECK((fx.vcall(api, 16, {queue, 0, 0}) & 0xFF) == 1);
    CHECK(g_record_calls == 1);
    CHECK(g_record_ctx == 0xABC);
    CHECK(g_record_canceled == 0);
    // With a delay: not dispatchable before it is due.
    CHECK(hr32(fx.vcall(api, 19, {queue, 0, 30, 0xDEF, kRecord})) == 0);
    CHECK((fx.vcall(api, 16, {queue, 0, 0}) & 0xFF) == 0);
    CHECK(wait_for([&] {
        fx.vcall(api, 16, {queue, 0, 0});
        return g_record_calls.load() == 2;
    }));
    CHECK(g_record_ctx == 0xDEF);
    CHECK(cbk::live_trampolines() == 0);

    // DuplicateHandle / CloseHandle reference counting.
    CHECK(hr32(fx.vcall(api, 15, {queue, out})) == 0);
    CHECK(load64(out) == queue);
    fx.vcall(api, 17, {queue});
    CHECK(hr32(fx.vcall(api, 19, {queue, 0, 0, 1, kRecord})) == 0);  // still alive
    CHECK((fx.vcall(api, 16, {queue, 0, 0}) & 0xFF) == 1);
    // Terminate with a guest callback, then close the last reference.
    g_record_calls = 0;
    CHECK(hr32(fx.vcall(api, 22, {queue, 1, 0x7E7, kRecord})) == 0);
    CHECK(wait_for([] { return g_record_calls.load() == 1; }));
    CHECK(g_record_ctx == 0x7E7);
    fx.vcall(api, 17, {queue});
    // A closed / made-up handle is rejected with a diagnostic, not dereferenced.
    com::clear_diagnostics();
    CHECK((fx.vcall(api, 16, {queue, 0, 0}) & 0xFF) == 0);
    CHECK(hr32(fx.vcall(api, 19, {0x1234, 0, 0, 1, kRecord})) == com::E_INVALIDARG_HR);
    CHECK(diag_contains("invalid XTaskQueue handle"));

    // Process task queue.
    CHECK((fx.vcall(api, 25, {out}) & 0xFF) == 1);
    const uint64_t process_queue = load64(out);
    CHECK(process_queue != 0);
    fx.vcall(api, 17, {process_queue});
    fx.vcall(api, 2);
    CHECK(cbk::live_trampolines() == 0);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

// XNetworking: the hint reports an initialized network with no connectivity, and each
// registration gets exactly one initial notification with that hint on its completion port.
void test_networking_slots() {
    Fixture fx;
    const uint64_t async = fx.query(kAsyncQueue, kAsyncQueue);
    const uint64_t net = fx.query(kNetworkingProvider, kNetworkingIid);
    const uint64_t hint = fx.alloc(16);
    CHECK(hr32(fx.vcall(net, 15, {hint})) == 0);  // XNetworkingGetConnectivityHint
    uint32_t words[4] = {};
    CHECK(com::read_guest(hint, words, sizeof(words)));
    CHECK(words[0] == 1u);            // connectivityLevel = None
    CHECK(words[1] == 0u);            // connectivityCost = Unknown
    CHECK((words[3] & 0xFFu) == 1u);  // networkInitialized
    CHECK((words[3] >> 8) == 0u);     // no data limit / roaming flags

    const uint64_t out = fx.alloc(8);
    CHECK(hr32(fx.vcall(async, 12, {0, 0, out})) == 0);  // Manual / Manual queue
    const uint64_t queue = load64(out);
    g_hint_calls = 0;
    std::memset(g_hint_words, 0xFF, sizeof(g_hint_words));
    CHECK(hr32(fx.vcall(net, 16, {queue, 0x4E7, kHintCb, out})) == 0);
    const uint64_t token = load64(out);
    CHECK(token != 0);
    CHECK(g_hint_calls == 0);                                // queued, not inline
    CHECK((fx.vcall(async, 16, {queue, 0, 0}) & 0xFF) == 0);  // nothing on the work port
    CHECK((fx.vcall(async, 16, {queue, 1, 0}) & 0xFF) == 1);  // initial notification
    CHECK(g_hint_calls == 1);
    CHECK(g_hint_ctx == 0x4E7);
    CHECK(g_hint_words[0] == 1u && g_hint_words[1] == 0u && (g_hint_words[3] & 0xFFu) == 1u);
    CHECK((fx.vcall(async, 16, {queue, 1, 0}) & 0xFF) == 0);  // only once
    CHECK((fx.vcall(net, 17, {token, 1}) & 0xFF) == 1);

    // Unregistered before dispatch: no callback.
    CHECK(hr32(fx.vcall(net, 16, {queue, 0, kHintCb, out})) == 0);
    CHECK((fx.vcall(net, 17, {load64(out), 0}) & 0xFF) == 1);
    fx.vcall(async, 16, {queue, 1, 0});
    CHECK(g_hint_calls == 1);

    // Null queue: delivered on the process task queue.
    CHECK(hr32(fx.vcall(net, 16, {0, 0x9, kHintCb, out})) == 0);
    CHECK(wait_for([] { return g_hint_calls.load() == 2; }));
    CHECK(g_hint_ctx == 0x9);
    CHECK((fx.vcall(net, 17, {load64(out), 1}) & 0xFF) == 1);
    CHECK(hr32(fx.vcall(net, 16, {queue, 0, 0, out})) == com::E_INVALIDARG_HR);
    fx.vcall(async, 17, {queue});
}

void test_user_slots() {
    Fixture fx;
    const uint64_t async = fx.query(kAsyncQueue, kAsyncQueue);
    const uint64_t user_api = fx.query(kUser, kUser);
    const uint64_t user_ext = fx.query(kUser, kUserExt);
    const uint64_t block = fx.alloc(sizeof(cbk::GuestAsyncBlock));
    cbk::GuestAsyncBlock guest{};
    guest.callback = kAsyncDone;
    CHECK(com::write_guest(block, guest));
    g_done_block = 0;
    // XUserAddAsync(AddDefaultUserSilently, block): completes on the process queue's worker
    // thread, whose guest callback runs on a lazily created guest thread.
    CHECK(hr32(fx.vcall(user_api, 7, {1, block})) == 0);
    CHECK(wait_for([] { return g_done_block.load() != 0; }));
    CHECK(g_done_block == block);
    CHECK(hr32(fx.vcall(async, 3, {block, 1})) == 0);  // XAsyncGetStatus(block, wait)
    const uint64_t size = fx.alloc(8);
    CHECK(hr32(fx.vcall(async, 4, {block, size})) == 0);
    CHECK(load64(size) == 8);
    const uint64_t out = fx.alloc(8);
    CHECK(hr32(fx.vcall(user_api, 8, {block, out})) == 0);  // XUserAddResult
    const uint64_t user = load64(out);
    CHECK(user != 0);
    CHECK(cbk::live_async_shadows() == 0);
    CHECK(hr32(fx.vcall(user_api, 11, {user, out})) == 0);  // XUserGetId
    CHECK(load64(out) == playgta6::os::gdk::kOfflineUserXuid);
    // XUserGetLocalId (s9): stable, non-zero, distinct from the XUID; invalid without an out pointer.
    CHECK(hr32(fx.vcall(user_api, 9, {user, out})) == 0);
    const uint64_t local_id = load64(out);
    CHECK(local_id != 0 && local_id != playgta6::os::gdk::kOfflineUserXuid);
    CHECK(hr32(fx.vcall(user_api, 9, {user, out})) == 0 && load64(out) == local_id);
    CHECK(hr32(fx.vcall(user_api, 9, {user, 0})) == com::E_INVALIDARG_HR);
    // XUserGetIsGuest (s13): the local user is not a guest; invalid without an out pointer.
    const uint64_t guest_flag = fx.alloc(8);
    CHECK(com::write_guest(guest_flag, uint64_t{0xff}));
    CHECK(hr32(fx.vcall(user_api, 13, {user, guest_flag})) == 0);
    CHECK((load64(guest_flag) & 0xff) == 0);
    CHECK(hr32(fx.vcall(user_api, 13, {user, 0})) == com::E_INVALIDARG_HR);
    // XUserGetAgeGroup (s19): the configured age group, default Unknown (0); every value is
    // reported as configured; a bad handle or a null out pointer is invalid.
    {
        namespace gp = playgta6::os::gdk::provider;
        CHECK(gp::user_age_group() == gp::UserAgeGroup::Unknown);
        const uint64_t age = fx.alloc(8);
        CHECK(com::write_guest(age, uint64_t{0xffffffffffffffffull}));
        CHECK(hr32(fx.vcall(user_api, 19, {user, age})) == 0);
        CHECK((load64(age) & 0xffffffffull) == 0);
        const std::pair<const char*, uint32_t> groups[] = {
            {"unknown", 0}, {"child", 1}, {"teen", 2}, {"adult", 3}};
        for (const auto& [name, value] : groups) {
            gp::UserAgeGroup group{};
            CHECK(gp::parse_user_age_group(name, &group));
            gp::set_user_age_group(group);
            CHECK(hr32(fx.vcall(user_api, 19, {user, age})) == 0);
            CHECK((load64(age) & 0xffffffffull) == value);
        }
        gp::UserAgeGroup ignored{};
        CHECK(!gp::parse_user_age_group("Adult", &ignored));
        CHECK(!gp::parse_user_age_group("", &ignored));
        gp::set_user_age_group(gp::UserAgeGroup::Unknown);
        CHECK(hr32(fx.vcall(user_api, 19, {user, 0})) == com::E_INVALIDARG_HR);
        CHECK(hr32(fx.vcall(user_api, 19, {0x1234, age})) == com::E_INVALIDARG_HR);
    }
    // XUserDuplicateHandle (s3): a second handle to the same local user, closed independently.
    const uint64_t dup_out = fx.alloc(8);
    CHECK(hr32(fx.vcall(user_api, 3, {user, dup_out})) == 0);
    const uint64_t dup = load64(dup_out);
    CHECK(dup != 0 && dup != user);
    CHECK(hr32(fx.vcall(user_api, 11, {dup, out})) == 0 && load64(out) == playgta6::os::gdk::kOfflineUserXuid);
    CHECK(hr32(fx.vcall(user_api, 9, {dup, out})) == 0 && load64(out) == local_id);
    fx.vcall(user_api, 4, {dup});  // XUserCloseHandle on the duplicate only
    CHECK(hr32(fx.vcall(user_api, 11, {user, out})) == 0);
    CHECK(hr32(fx.vcall(user_api, 3, {0x1234, dup_out})) == com::E_INVALIDARG_HR);
    const uint64_t tag = fx.alloc(32);
    CHECK(hr32(fx.vcall(user_ext, 3, {user, 0, 32, tag, out})) == 0);  // XUserGetGamertag
    char text[32] = {};
    CHECK(com::read_guest(tag, text, sizeof(text)));
    CHECK(std::string(text) == playgta6::os::gdk::kOfflineUserGamertag);
    CHECK(load64(out) == std::strlen(playgta6::os::gdk::kOfflineUserGamertag) + 1);
    // Change events: a token, no notifications for the local offline user.
    CHECK(hr32(fx.vcall(user_api, 33, {0, 0, kRecord, out})) == 0);
    CHECK(load64(out) != 0);
    CHECK((fx.vcall(user_api, 34, {load64(out), 1}) & 0xFF) == 1);
    fx.vcall(user_api, 4, {user});  // XUserCloseHandle
    com::clear_diagnostics();
    CHECK(hr32(fx.vcall(user_api, 11, {user, out})) == com::E_INVALIDARG_HR);
    CHECK(diag_contains("invalid XUserHandle"));
    CHECK(hr32(fx.vcall(user_api, 9, {user, out})) == com::E_INVALIDARG_HR);  // closed handle
    // Results on a block without an operation.
    const uint64_t other_block = fx.alloc(sizeof(cbk::GuestAsyncBlock));
    CHECK(hr32(fx.vcall(user_api, 8, {other_block, out})) == com::E_INVALIDARG_HR);
}

void test_unimplemented_slots() {
    Fixture fx;
    const uint64_t async = fx.query(kAsyncQueue, kAsyncQueue);
    GuestFault fault{};
    com::clear_diagnostics();
    bool trapped = false;
    try {
        fx.vcall(async, 8, {0, 0, 0});  // unmapped slot of the XAsync interface
    } catch (const GuestFault& f) {
        fault = f;
        trapped = true;
    }
    fx.restore_stack();
    CHECK(trapped);
    CHECK(fault.info.kind == static_cast<uint32_t>(FaultKind::UnimplementedSlot));
    CHECK(fault.info.detail == 8);
    CHECK(diag_contains("slot 8 called"));

    // R40: stub-only interfaces no longer trap; each answers one documented HRESULT and counts hits.
    const struct {
        Guid provider;
        Guid iid;
        uint32_t expected;
    } areas[] = {
        {kUnmappedProvider, kUnmappedIid, 0x80004004u},                                        // XGameUi: E_ABORT
        {com::make_guid("1b339674-328d-4283-a200-3171f18d3639"),
         com::make_guid("1b339674-328d-4283-a200-3171f18d3639"), 0x80070032u},                 // shell
        {com::make_guid("a4f1aee2-4bf1-4485-b008-a7c26d52ac27"),
         com::make_guid("22e672d7-b4e3-406c-bd50-8f0d25236f9e"), 0x80070032u},                 // capture
        {com::make_guid("bbfbdcc7-bfe7-409b-a5ca-edf054960b4d"),
         com::make_guid("bbfbdcc7-bfe7-409b-a5ca-edf054960b4d"), 0u},                          // events
    };
    for (const auto& area : areas) {
        int32_t hr = -1;
        const uint64_t obj = fx.query(area.provider, area.iid, &hr);
        CHECK(hr == 0 && obj != 0);
        const uint64_t r = fx.vcall(obj, 7, {1, 2, 3});
        fx.restore_stack();
        CHECK(static_cast<uint32_t>(r) == area.expected);
        bool reported = false;
        for (const com::StubHit& hit : com::stub_hit_report()) {
            if (hit.slot == 7 && hit.iid == area.iid && hit.hits == 1) {
                reported = true;
            }
        }
        CHECK(reported);
        CHECK(fx.vcall(obj, 2) >= 1);  // IUnknown still works
    }
}

uint64_t make_guest_block(Fixture& fx, uint64_t queue) {
    const uint64_t block = fx.alloc(sizeof(cbk::GuestAsyncBlock));
    cbk::GuestAsyncBlock guest{};
    guest.queue = queue;
    guest.context = 0xC0;
    guest.callback = kAsyncDone;
    CHECK(com::write_guest(block, guest));
    return block;
}

// The XAsync provider slots end to end: QueryApiImpl -> vtable slot -> framework -> guest provider.
void test_async_provider_slots() {
    namespace gdk = playgta6::os::gdk;
    Fixture fx;
    const uint64_t api = fx.query(kAsyncQueue, kAsyncQueue);
    g_async_api = api;
    const uint64_t out = fx.alloc(8);
    CHECK(hr32(fx.vcall(api, 12, {0, 0, out})) == 0);  // Manual / Manual queue
    const uint64_t queue = load64(out);
    const uint64_t identity = 0x1D1D;
    const uint64_t name = fx.cstring("TestOperation");
    auto dispatch = [&](uint64_t q, uint32_t port) { return (fx.vcall(api, 16, {q, port, 0}) & 0xFF) == 1; };

    // Begin -> DoWork (E_PENDING, rescheduled) -> DoWork (complete) -> callback -> GetResult -> Cleanup.
    {
        std::lock_guard<std::mutex> lock(g_provider_mutex);
        g_provider_ops.clear();
        g_provider_pending = 1;
    }
    const uint64_t block = make_guest_block(fx, queue);
    CHECK(hr32(fx.vcall(api, 7, {block, 0xC7C7, identity, name, kProvider})) == 0);  // XAsyncBegin
    CHECK(provider_ops() == "B");
    CHECK(g_seen_async == block && g_seen_context == 0xC7C7);  // the guest block and context
    CHECK(hr32(fx.vcall(api, 3, {block, 0})) == gdk::E_PENDING_HR);
    CHECK(dispatch(queue, 0));
    CHECK(hr32(fx.vcall(api, 3, {block, 0})) == gdk::E_PENDING_HR);
    CHECK(dispatch(queue, 0));
    CHECK(hr32(fx.vcall(api, 3, {block, 0})) == 0);
    g_done_block = 0;
    CHECK(dispatch(queue, 1));
    CHECK(g_done_block == block);  // guest completion callback with the guest block
    const uint64_t size = fx.alloc(8);
    CHECK(hr32(fx.vcall(api, 4, {block, size})) == 0 && load64(size) == 8);
    const uint64_t buffer = fx.alloc(16);
    const uint64_t used = fx.alloc(8);
    CHECK(hr32(fx.vcall(api, 11, {block, identity + 1, 16, buffer, used})) == com::E_INVALIDARG_HR);
    CHECK(hr32(fx.vcall(api, 11, {block, identity, 4, buffer, used})) == gdk::E_NOT_SUFFICIENT_BUFFER_HR);
    CHECK(hr32(fx.vcall(api, 11, {block, identity, 16, buffer, used})) == 0);  // XAsyncGetResult
    CHECK(load64(buffer) == kPayload && load64(used) == 8 && g_seen_buffer_size == 8);
    CHECK(provider_ops() == "BWWRC");
    CHECK(hr32(fx.vcall(api, 11, {block, identity, 16, buffer, 0})) == gdk::E_ILLEGAL_METHOD_CALL_HR);
    CHECK(hr32(fx.vcall(api, 3, {block, 1})) == 0);
    pg6_gdk_xasync_complete(cbk::async_shadow_find(block), gdk::E_FAIL_HR, 0);  // ignored
    fx.vcall(api, 10, {block, static_cast<uint32_t>(gdk::E_FAIL_HR), 0});    // ignored too
    CHECK(hr32(fx.vcall(api, 3, {block, 0})) == 0);

    // XAsyncCancel before the work ran: the provider is told, the queued DoWork aborts.
    {
        std::lock_guard<std::mutex> lock(g_provider_mutex);
        g_provider_ops.clear();
    }
    const uint64_t block2 = make_guest_block(fx, queue);
    CHECK(hr32(fx.vcall(api, 7, {block2, 0, identity, 0, kProvider})) == 0);
    fx.vcall(api, 5, {block2});  // XAsyncCancel
    CHECK(dispatch(queue, 0));
    CHECK(hr32(fx.vcall(api, 3, {block2, 0})) == gdk::E_ABORT_HR);
    g_done_block = 0;
    CHECK(dispatch(queue, 1));
    CHECK(g_done_block == block2);
    CHECK(provider_ops() == "BXC");

    // XAsyncRun: the guest work function runs as DoWork; its result is the status.
    g_work_result = static_cast<int32_t>(0x80070005u);
    const uint64_t block3 = make_guest_block(fx, queue);
    CHECK(hr32(fx.vcall(api, 6, {block3, kWorkFn})) == 0);
    CHECK(g_work_calls == 0);
    CHECK(dispatch(queue, 0));
    CHECK(g_work_calls == 1 && g_seen_async == block3);
    CHECK(hr32(fx.vcall(api, 3, {block3, 0})) == static_cast<int32_t>(0x80070005u));
    CHECK(dispatch(queue, 1));
    CHECK(hr32(fx.vcall(api, 6, {block3, 0})) == com::E_INVALIDARG_HR);
    CHECK(hr32(fx.vcall(api, 7, {block3, 0, 0, 0, 0})) == com::E_INVALIDARG_HR);

    // Null queue: the process task queue's threads run DoWork and the callback.
    {
        std::lock_guard<std::mutex> lock(g_provider_mutex);
        g_provider_ops.clear();
        g_provider_pending = 0;
    }
    const uint64_t block4 = make_guest_block(fx, 0);
    g_done_block = 0;
    CHECK(hr32(fx.vcall(api, 7, {block4, 0, identity, 0, kProvider})) == 0);
    CHECK(hr32(fx.vcall(api, 3, {block4, 1})) == 0);  // wait
    CHECK(g_done_block == block4);
    CHECK(hr32(fx.vcall(api, 11, {block4, identity, 8, buffer, used})) == 0);
    CHECK(provider_ops() == "BWRC");

    // XTaskQueueGetPort / XTaskQueueCreateComposite: work from `queue`, completions from `queue2`.
    CHECK(hr32(fx.vcall(api, 12, {0, 0, out})) == 0);
    const uint64_t queue2 = load64(out);
    CHECK(hr32(fx.vcall(api, 14, {queue, 0, out})) == 0);
    const uint64_t work_port = load64(out);
    CHECK(hr32(fx.vcall(api, 14, {queue2, 1, out})) == 0);
    const uint64_t completion_port = load64(out);
    CHECK(work_port != 0 && completion_port != 0 && work_port != completion_port);
    com::clear_diagnostics();
    CHECK(hr32(fx.vcall(api, 13, {work_port, 0x4242, out})) == com::E_INVALIDARG_HR);
    CHECK(diag_contains("invalid XTaskQueuePortHandle"));
    CHECK(hr32(fx.vcall(api, 13, {work_port, completion_port, out})) == 0);
    const uint64_t composite = load64(out);
    CHECK(composite != 0);
    {
        std::lock_guard<std::mutex> lock(g_provider_mutex);
        g_provider_ops.clear();
    }
    const uint64_t block5 = make_guest_block(fx, composite);
    CHECK(hr32(fx.vcall(api, 7, {block5, 0, identity, 0, kProvider})) == 0);
    CHECK(!dispatch(queue2, 0));
    CHECK(dispatch(queue, 0));  // the composite's work port is queue's
    g_done_block = 0;
    CHECK(!dispatch(queue, 1));
    CHECK(dispatch(composite, 1));  // dispatching the composite dispatches queue2's port
    CHECK(g_done_block == block5);
    CHECK(hr32(fx.vcall(api, 11, {block5, identity, 8, buffer, used})) == 0);
    for (uint64_t q : {composite, queue2, queue}) {
        fx.vcall(api, 17, {q});  // XTaskQueueCloseHandle
    }
    fx.vcall(api, 2);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

// libHttpClient flat imports taking a guest XAsyncBlock go through shadow blocks.
void test_http_client_thunks() {
    namespace gdk = playgta6::os::gdk;
    Fixture fx;
    const uint64_t api = fx.query(kAsyncQueue, kAsyncQueue);
    const uint64_t out = fx.alloc(8);
    CHECK(hr32(fx.vcall(api, 12, {0, 0, out})) == 0);
    const uint64_t queue = load64(out);
    const uint64_t perform = register_host_callable(&pg6_gthunk_hc_http_call_perform_async, "HCHttpCallPerformAsync");
    const uint64_t cleanup = register_host_callable(&pg6_gthunk_hc_cleanup_async, "HCCleanupAsync");
    CHECK(pg6_mw_hc_initialize(nullptr) == 0);
    Pg6HCCallHandle call = nullptr;
    CHECK(pg6_mw_hc_http_call_create(&call) == 0);
    const uint64_t block = make_guest_block(fx, queue);
    CHECK(hr32(fx.call(perform, {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(call)), block})) == 0);
    CHECK(hr32(fx.vcall(api, 3, {block, 0})) == 0);  // offline: completed at once
    g_done_block = 0;
    CHECK((fx.vcall(api, 16, {queue, 1, 0}) & 0xFF) == 1);
    CHECK(g_done_block == block);
    const uint64_t block2 = make_guest_block(fx, queue);
    CHECK(hr32(fx.call(cleanup, {block2})) == 0);
    CHECK((fx.vcall(api, 16, {queue, 1, 0}) & 0xFF) == 1);
    CHECK(g_done_block == block2);
    CHECK(hr32(fx.call(perform, {0x1234, block})) == gdk::E_INVALIDARG_HR);  // unknown call handle
    CHECK(hr32(fx.call(cleanup, {0})) == gdk::E_INVALIDARG_HR);
    pg6_mw_hc_http_call_close_handle(call);
    fx.vcall(api, 17, {queue});
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

// XStoreExt (O37): slot 71 AcquireLicenseForDurablesAsync and partner slot 72 Result return a
// truthful synchronous E_NOTIMPL offline failure (zero ownership/entitlement), while all other
// slots remain trapping stubs.
void test_store_ext_slots() {
    Fixture fx;
    int32_t hr = -1;
    const uint64_t store = fx.query(kStoreProvider, kStoreIid, &hr);
    CHECK(hr == 0 && store != 0);
    const uint64_t store_ext = fx.query(kStoreProvider, kStoreExtIid, &hr);
    CHECK(hr == 0 && store_ext != 0 && store_ext != store);

    // Create a valid XStoreContextHandle via XStore slot 3.
    const uint64_t ctx_out = fx.alloc(8);
    CHECK(hr32(fx.vcall(store, 3, {0, ctx_out})) == 0);
    const uint64_t ctx = load64(ctx_out);
    CHECK(ctx != 0);

    const uint64_t store_id = fx.cstring("9NBLGGH42RLT");
    const uint64_t block = make_guest_block(fx, 0);
    g_done_block = 0;

    // Slot 71: truthful synchronous offline failure (E_NOTIMPL), no async shadow or fake completion.
    const int32_t acquire_hr = hr32(fx.vcall(store_ext, 71, {ctx, store_id, block}));
    CHECK(acquire_hr == com::E_NOTIMPL_HR);
    CHECK(acquire_hr < 0);
    CHECK(cbk::live_async_shadows() == 0);
    CHECK(g_done_block == 0);

    // Slot 71 argument validation: bogus/null context, null store_id, null async_block.
    com::clear_diagnostics();
    CHECK(hr32(fx.vcall(store_ext, 71, {0x1234, store_id, block})) == com::E_INVALIDARG_HR);
    CHECK(diag_contains("invalid XStoreContextHandle"));
    CHECK(hr32(fx.vcall(store_ext, 71, {0, store_id, block})) == com::E_INVALIDARG_HR);
    CHECK(hr32(fx.vcall(store_ext, 71, {ctx, 0, block})) == com::E_INVALIDARG_HR);
    CHECK(hr32(fx.vcall(store_ext, 71, {ctx, store_id, 0})) == com::E_INVALIDARG_HR);

    // Slot 72: clears out license handle if non-null and returns E_NOTIMPL.
    const uint64_t license_out = fx.alloc(8);
    CHECK(com::write_guest(license_out, uint64_t{0xDEADBEEFULL}));
    CHECK(hr32(fx.vcall(store_ext, 72, {block, license_out})) == com::E_NOTIMPL_HR);
    CHECK(load64(license_out) == 0);
    CHECK(hr32(fx.vcall(store_ext, 72, {block, 0})) == com::E_NOTIMPL_HR);
    CHECK(hr32(fx.vcall(store_ext, 72, {0, license_out})) == com::E_NOTIMPL_HR);

    // Unimplemented slots on XStoreExt (both before 71 and after 72) still trap unchanged.
    for (uint32_t slot : {70u, 73u}) {
        GuestFault fault{};
        bool trapped = false;
        try {
            fx.vcall(store_ext, slot, {ctx, store_id, block});
        } catch (const GuestFault& f) {
            fault = f;
            trapped = true;
        }
        fx.restore_stack();
        CHECK(trapped);
        CHECK(fault.info.kind == static_cast<uint32_t>(FaultKind::UnimplementedSlot));
        CHECK(fault.info.detail == slot);
    }
    bool hit70 = false;
    bool hit73 = false;
    for (const com::StubHit& hit : com::stub_hit_report()) {
        if (hit.iid == kStoreExtIid && hit.slot == 70 && hit.hits == 1) {
            hit70 = true;
        }
        if (hit.iid == kStoreExtIid && hit.slot == 73 && hit.hits == 1) {
            hit73 = true;
        }
    }
    CHECK(hit70 && hit73);

    // Closing the context makes subsequent slot 71 calls reject the stale handle.
    fx.vcall(store, 4, {ctx});
    CHECK(hr32(fx.vcall(store_ext, 71, {ctx, store_id, block})) == com::E_INVALIDARG_HR);

    CHECK(fx.vcall(store_ext, 2) >= 1);
    CHECK(fx.vcall(store, 2) >= 1);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

}  // namespace

uint32_t g_exit_code = 0;
int g_exit_calls = 0;
void record_exit(uint32_t code, void*) {
    g_exit_code = code;
    ++g_exit_calls;
}

// XGame s4 XLaunchNewGame: the running title relaunching itself stops the guest with the relaunch
// exit code (the harness restarts it); another title stays E_NOTIMPL.
void test_launch_new_game() {
    CHECK(prov::is_self_launch_target("game.exe", "game.exe"));
    CHECK(prov::is_self_launch_target("D:\\Title\\GAME.EXE", "\"C:\\x y\\game.exe\" -arg 1"));
    CHECK(prov::is_self_launch_target("./game.exe", "game.exe -foo"));
    CHECK(!prov::is_self_launch_target("other.exe", "game.exe"));
    CHECK(!prov::is_self_launch_target("game.exe", ""));
    CHECK(!prov::is_self_launch_target("", "game.exe"));

    Fixture fx;
    int32_t hr = -1;
    const uint64_t game = fx.query(kGameProvider, kGameIid, &hr);
    CHECK(hr == 0 && game != 0);
    playgta6::os::win32::set_command_line("Title.exe -mode 2");
    playgta6::os::win32::reset_process_exit_state();
    playgta6::os::win32::set_process_exit_handler(&record_exit, nullptr);
    g_exit_calls = 0;

    CHECK(hr32(fx.vcall(game, 4, {fx.cstring("Other.exe"), fx.cstring(""), 0})) == com::E_NOTIMPL_HR);
    CHECK(g_exit_calls == 0);
    CHECK(hr32(fx.vcall(game, 4, {0, 0, 0})) == static_cast<int32_t>(0x80070057u));
    CHECK(hr32(fx.vcall(game, 4, {fx.cstring("title.exe"), fx.cstring("-mode 2"), 0})) == 0);
    CHECK(g_exit_calls == 1 && g_exit_code == playgta6::os::win32::kRelaunchSelfExitCode);
    uint32_t code = 0;
    CHECK(playgta6::os::win32::process_exit_requested(&code) && code == playgta6::os::win32::kRelaunchSelfExitCode);
    std::string relaunch_args;
    CHECK(playgta6::os::win32::relaunch_requested(&relaunch_args) && relaunch_args == "-mode 2");
    playgta6::os::win32::reset_relaunch_request();

    playgta6::os::win32::set_process_exit_handler(nullptr, nullptr);
    playgta6::os::win32::reset_process_exit_state();
    playgta6::os::win32::set_command_line("game.exe");
}

int main() {
    test_module_and_exports();
    test_query_api();
    test_task_queue_slots();
    test_user_slots();
    test_networking_slots();
    test_unimplemented_slots();
    test_async_provider_slots();
    test_http_client_thunks();
    test_store_ext_slots();
    test_launch_new_game();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_xgameruntime_provider: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xgameruntime_provider: all tests passed\n");
    return 0;
}
