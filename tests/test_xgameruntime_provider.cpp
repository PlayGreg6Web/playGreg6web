#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
#include "os/win32_kernel.h"
#include "os/shim_exports.h"
#include "os/win32_loader.h"
#include "os/xasync_provider.h"
#include "os/xbox_middleware.h"
#include "os/xgameruntime.h"
#include "os/xgameruntime_provider.h"

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

const LiftedFunctionEntry kFunctions[] = {
    {kRecord, sub_record},
    {kAsyncDone, sub_async_done},
    {kProvider, sub_provider},
    {kWorkFn, sub_work},
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

constexpr Guid kAsyncQueue = com::make_guid("073b7dcb-1fcf-4030-94be-e3c9eb623428");
constexpr Guid kUser = com::make_guid("01acd177-91f9-4763-a38e-ccbb55ce32e0");
constexpr Guid kUserExt = com::make_guid("cef4fac0-7676-4a94-a119-4c43f9eb5b74");
constexpr Guid kFeature = com::make_guid("8836fe87-edb9-4fe3-8dad-05f0d2cd5b40");
constexpr Guid kSystemProvider = com::make_guid("e349bd1a-fc20-4e40-b99c-4178cc6b409f");
constexpr Guid kSystemIid = com::make_guid("1861cf2e-e18b-4834-a9f5-b4a4e6efb4cf");
constexpr Guid kPackageProvider = com::make_guid("af406016-e850-4aa8-a88d-2f3dcb9dac7e");
constexpr Guid kPackageIid = com::make_guid("0b9d556e-ab70-43f4-baa0-9999ac4cb040");
constexpr Guid kPlsProvider = com::make_guid("f4faf4d4-2d04-4fce-b3e0-474a713a3e84");
constexpr Guid kPlsIid = com::make_guid("41a4e10c-5a7e-41d9-8c37-37bde62a07d6");
constexpr Guid kHdrProvider = com::make_guid("03f0fe74-fdd9-4e5c-b630-f9339c47acc5");
constexpr Guid kHdrIid = com::make_guid("35f07670-706e-4bfb-9476-090798c5ebf3");
constexpr Guid kUnmappedProvider = com::make_guid("dfcd4649-4ff8-4043-ba07-35d607df98b0");
constexpr Guid kUnmappedIid = com::make_guid("eaf669df-5542-4590-99a3-8dc061f837cc");
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
    CHECK((fx.vcall(feature, 3, {1}) & 0xFF) == 0);
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
        CHECK(hr32(fx.vcall(pls, 4, {4, path, used})) == playgta6::os::gdk::E_NOT_SUFFICIENT_BUFFER_HR);
        CHECK(load64(used) == std::strlen(text) + 1);
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
        // XSpeechSynthesizer slot 9 (XSpeechSynthesizerCreate): no synthesizer, so it fails and
        // leaves the handle null; a null out pointer is invalid.
        const uint64_t speech = fx.query(com::make_guid("3e241977-6237-41e9-8559-102c2d9983f1"),
                                         com::make_guid("d722b373-8c4d-4692-9e51-d6ad9b37aa7d"));
        const uint64_t handle = fx.alloc(8);
        CHECK(com::write_guest(handle, uint64_t{0x1234}));
        CHECK(hr32(fx.vcall(speech, 9, {handle})) == com::E_NOTIMPL_HR);
        CHECK(load64(handle) == 0);
        CHECK(hr32(fx.vcall(speech, 9, {0})) == com::E_INVALIDARG_HR);
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
    // XSystemGetXboxLiveSandboxId (s4): offline, no sandbox: E_NOTIMPL, empty string, used 0.
    CHECK(com::write_guest(used, uint64_t{77}));
    CHECK(hr32(fx.vcall(system, 4, {0x20, buffer, used})) == com::E_NOTIMPL_HR);
    CHECK(load64(used) == 0);
    char sandbox = 'x';
    CHECK(com::read_guest(buffer, &sandbox, 1) && sandbox == '\0');
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

    // A stub-only interface reports its slots too.
    int32_t hr = -1;
    const uint64_t unmapped = fx.query(kUnmappedProvider, kUnmappedIid, &hr);
    CHECK(hr == 0 && unmapped != 0);
    trapped = false;
    try {
        fx.vcall(unmapped, 7, {1, 2, 3});
    } catch (const GuestFault& f) {
        fault = f;
        trapped = true;
    }
    fx.restore_stack();
    CHECK(trapped && fault.info.detail == 7);
    bool reported = false;
    for (const com::StubHit& hit : com::stub_hit_report()) {
        if (hit.slot == 7 && hit.iid == kUnmappedIid && hit.hits == 1) {
            reported = true;
        }
    }
    CHECK(reported);
    CHECK(fx.vcall(unmapped, 2) >= 1);  // IUnknown still works
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

}  // namespace

int main() {
    test_module_and_exports();
    test_query_api();
    test_task_queue_slots();
    test_user_slots();
    test_unimplemented_slots();
    test_async_provider_slots();
    test_http_client_thunks();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_xgameruntime_provider: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xgameruntime_provider: all tests passed\n");
    return 0;
}
