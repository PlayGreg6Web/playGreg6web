#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
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

const LiftedFunctionEntry kFunctions[] = {
    {kRecord, sub_record},
    {kAsyncDone, sub_async_done},
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
    const uint64_t system = fx.query(kSystemProvider, kSystemIid);
    const uint64_t buffer = fx.alloc(64);
    const uint64_t used = fx.alloc(8);
    CHECK(hr32(fx.vcall(system, 3, {64, buffer, used})) == 0);
    CHECK(load64(used) == 39);
    char id[39] = {};
    CHECK(com::read_guest(buffer, id, sizeof(id)));
    CHECK(std::strlen(id) == 38);
    CHECK(hr32(fx.vcall(system, 3, {8, buffer, used})) == playgta6::os::gdk::E_NOT_SUFFICIENT_BUFFER_HR);
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
        fx.vcall(async, 13, {0, 0, 0});  // XTaskQueueCreateComposite: known name, not implemented
    } catch (const GuestFault& f) {
        fault = f;
        trapped = true;
    }
    fx.restore_stack();
    CHECK(trapped);
    CHECK(fault.info.kind == static_cast<uint32_t>(FaultKind::Unimplemented));
    CHECK(fault.info.detail == 13);
    CHECK(diag_contains("slot 13 (XTaskQueueCreateComposite)"));

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

}  // namespace

int main() {
    test_module_and_exports();
    test_query_api();
    test_task_queue_slots();
    test_user_slots();
    test_unimplemented_slots();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_xgameruntime_provider: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xgameruntime_provider: all tests passed\n");
    return 0;
}
