#include "os/xgameruntime_provider.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "guest/guest_callbacks.h"
#include "guest/guest_runtime.h"
#include "os/os_shim.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/xasync_provider.h"
#include "os/xbox_offline.h"
#include "os/xgameruntime.h"

// Evidence and mapping
// --------------------
// A static survey of the game executable (aggregate data only) found:
//   * one lazy loader: LoadLibraryExW("xgameruntime.dll") then GetProcAddress of QueryApiImpl,
//     InitializeApiImpl, InitializeApiImplEx, InitializeApiImplEx2, XErrorReport and
//     UninitializeApiImpl; a missing QueryApiImpl / XErrorReport / UninitializeApiImpl (or all
//     three Initialize* exports) fails the client with 0x89240101. Initialization tries
//     InitializeApiImplEx2(u64, u64, flags, options*) first, then Ex(u64, u64, flags), then
//     InitializeApiImpl(u32, u32); XErrorReport(HRESULT, const char*) reports client errors.
//   * QueryApiImpl(const GUID* provider, const GUID* iid, void** out) -> HRESULT, called by 161
//     one-per-API wrappers, each doing QueryApiImpl -> one vtable call -> Release (slot 2), with
//     no caching, no QueryInterface/AddRef: 19 (provider, iid) pairs, 17 providers, 322 vtable
//     call sites. REGDB_E_CLASSNOTREG (0x80040154) from QueryApiImpl is turned into
//     0x89240107 by the client; other failures are returned by the API wrapper as is.
// Slot -> GDK function mapping (argument counts exclude `this`; h/m/l = confidence):
//   {073b7dcb-...} XAsync/XTaskQueue: s3 XAsyncGetStatus(2) h, s4 XAsyncGetResultSize(2) m,
//     s5 XAsyncCancel(1) m, s6 XAsyncRun(2) h, s7 XAsyncBegin(5) h, s9 XAsyncSchedule(2) m,
//     s10 XAsyncComplete(3) h, s11 XAsyncGetResult(5) m, s12 XTaskQueueCreate(3) m,
//     s13 CreateComposite(3) h, s14 GetPort(3) h, s15 DuplicateHandle(2) h, s16 Dispatch(3) h,
//     s17 CloseHandle(1) h, s19 SubmitDelayedCallback(5) h, s22 Terminate(4) m,
//     s25 GetCurrentProcessTaskQueue(1) h, s29 ()->void called first by 58 wrappers
//     (XThreadAssertNotTimeSensitive) m. O22 implements s5..s7, s9..s11, s13 and s14 on the XAsync
//     provider framework (os/xasync_provider.h); the evidence's argument counts match the GDK
//     signatures: XAsyncCancel(block), XAsyncRun(block, work), XAsyncBegin(block, context,
//     identity, identityName, provider), XAsyncSchedule(block, delayMs), XAsyncComplete(block, hr,
//     requiredBufferSize), XAsyncGetResult(block, identity, bufferSize, buffer, bufferUsed),
//     XTaskQueueCreateComposite(workPort, completionPort, queue*), XTaskQueueGetPort(queue, port,
//     portHandle*). s8 has no call-site evidence (XAsyncBeginAlloc would take 7 arguments) and
//     stays a stub like the other unmapped slots.
//   {01acd177-...} XUser: s4 CloseHandle h, s7 AddAsync h, s8 AddResult h, s11 GetId m,
//     s20 CheckPrivilege m, s21/s22 ResolvePrivilegeWithUiAsync/Result m,
//     s23 GetTokenAndSignatureAsync(9) h, s24 ...ResultSize m, s33/s34 Register/Unregister
//     ForChangeEvent h; s3 Duplicate, s5 Compare, s9/s10 AddByIdWithUi, s12 FindUserById,
//     s13 GetLocalId, s14 FindUserByLocalId, s19 GetAgeGroup, s25 GetTokenAndSignatureResult
//     are left as stubs.
//   {01acd177-...}/{cef4fac0-...} XUser extension: s3 XUserGetGamertag(5) m.
//   {8836fe87-...} s3 XGameRuntimeIsFeatureAvailable(1)->bool h.
//   {e349bd1a-...}/{1861cf2e-...} XSystem: s3 GetConsoleId(3) h, s4 GetXboxLiveSandboxId(3) h
//     (offline: no sandbox, E_NOTIMPL), s5 GetAppSpecificDeviceId stub. XSystemGetDeviceType /
//     GetAnalyticsInfo are never called.
//   {973a344e-...}/{2549f142-...} XGame: s3 XGameGetXboxTitleId(1) h (s4 XLaunchNewGame stub).
//   {0dd112ac-...}/{b09d803c-...} XStore: s3 CreateContext h, s4 CloseContextHandle h; the
//     other 40 used slots stay stubs (O8 offline policy: nothing about licences is invented).
//   {37e56907-...}/{bf2346b2-...} XNetworking: s15 GetConnectivityHint h, s16/s17
//     Register/UnregisterConnectivityHintChanged h.
//   {af406016-...}/{0b9d556e-...} XPackage: s3 GetCurrentProcessPackageIdentifier(2) h (local
//     deterministic id), s4 IsPackagedProcess()->bool h, s5 CreateInstallationMonitor(6) h,
//     s6 CloseInstallationMonitorHandle(1) h, s7 GetInstallationProgress(2) h,
//     s9 RegisterInstallationProgressChanged(4) h (the local package is complete: progress is
//     always finished and launchable, and never changes), s39 EnumeratePackages(4) h (no local
//     content packages; game-package enumeration returns E_NOTIMPL).
//   {f4faf4d4-...}/{41a4e10c-...} XPersistentLocalStorage: s4 GetPath(3) h ("T:\userdata\pls\",
//     a directory of the /userdata overlay).
//   {03f0fe74-...}/{35f07670-...}: s4 has the call shape of XDisplayTryEnableHdrMode(2) h
//     (returns Disabled: the runtime presents SDR).
//   {3e241977-...}/{d722b373-...} XSpeechSynthesizer: s9 XSpeechSynthesizerCreate(1) h (one
//     pointer to a still-null handle field; fails with E_NOTIMPL: no speech synthesizer).
//   Stub-only (every slot logs and traps, so the first boot reports what it needs):
//     XGameSave {704c3f58-...}, XStore extension {0dd112ac-...}/{2d42fea5-...}, XGameEventWrite
//     {bbfbdcc7-...}, {1b339674-...} (XDisplay?), and unmapped {a4f1aee2-...}, {dfcd4649-...}.
//   {b884675d-...}: s3 is called through a wrapper that takes one pointer, passes a zeroed 0x90-byte
//     local as the hidden return slot and copies 0x90 bytes from the returned pointer: a
//     no-argument function returning a 144-byte struct by value, i.e. XSystemGetAnalyticsInfo()
//     -> XSystemAnalyticsInfo {XVersion osVersion, hostingOsVersion; char family[64], form[64]}.

namespace playgta6::os::gdk::provider {

namespace com = playgta6::guest::com;
namespace cbk = playgta6::guest::callbacks;
using com::CallContext;
using com::GuestPtr;
using com::Guid;
using com::OutPtr;

namespace {

// ---------------------------------------------------------------------------
// Guest-visible handles
// ---------------------------------------------------------------------------

// XTaskQueue / XUser / XStore handles given to guest code are the host shims' handle values
// (64-bit, opaque to the guest). Every one handed out is counted here, so a bogus handle coming
// back is rejected with a diagnostic instead of being dereferenced by a shim.
class HandleTable {
public:
    void add(const void* handle) {
        if (handle != nullptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++counts_[handle];
        }
    }
    void remove(const void* handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = counts_.find(handle);
        if (it != counts_.end() && --it->second == 0) {
            counts_.erase(it);
        }
    }
    bool contains(const void* handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return counts_.count(handle) != 0;
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        counts_.clear();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<const void*, uint64_t> counts_;
};

HandleTable& queues() {
    static HandleTable t;
    return t;
}
HandleTable& users() {
    static HandleTable t;
    return t;
}
HandleTable& stores() {
    static HandleTable t;
    return t;
}
HandleTable& ports() {
    static HandleTable t;
    return t;
}
HandleTable& installation_monitors() {
    static HandleTable t;
    return t;
}

template <typename T>
T* from_guest(uint64_t value) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(value));
}

uint64_t to_guest(const void* handle) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
}

bool check_handle(HandleTable& table, uint64_t value, const char* what) {
    if (value != 0 && table.contains(from_guest<void>(value))) {
        return true;
    }
    com::detail::log_bad_pointer(what, value);
    return false;
}

// Process task queue reference handed out by GetCurrentProcessTaskQueue (null queue argument).
bool optional_queue_ok(uint64_t value) {
    return value == 0 || check_handle(queues(), value, "XTaskQueue handle");
}

// Begins a shadowed async call for a guest XAsyncBlock (validates its queue first).
Pg6XAsyncBlock* begin_async(uint64_t guest_block) {
    cbk::GuestAsyncBlock guest{};
    if (guest_block == 0 || !com::read_guest(guest_block, guest)) {
        com::detail::log_bad_pointer("XAsyncBlock", guest_block);
        return nullptr;
    }
    if (!optional_queue_ok(guest.queue)) {
        return nullptr;
    }
    return cbk::async_shadow_begin(guest_block);
}

Pg6XAsyncBlock* find_async(uint64_t guest_block) {
    Pg6XAsyncBlock* shadow = cbk::async_shadow_find(guest_block);
    if (shadow == nullptr) {
        com::detail::log_bad_pointer("XAsyncBlock (no operation started on it)", guest_block);
    }
    return shadow;
}

// ---------------------------------------------------------------------------
// Guest async providers (O22)
// ---------------------------------------------------------------------------

// XAsyncProviderData in guest memory (Win64 layout).
struct GuestAsyncProviderData {
    uint64_t async;        // XAsyncBlock* (the guest block)
    uint64_t buffer_size;  // size_t
    uint64_t buffer;       // void*
    uint64_t context;      // void*
};
static_assert(sizeof(GuestAsyncProviderData) == 32, "XAsyncProviderData is 32 bytes on x64");
static_assert(offsetof(GuestAsyncProviderData, buffer_size) == 8);
static_assert(offsetof(GuestAsyncProviderData, buffer) == 16);
static_assert(offsetof(GuestAsyncProviderData, context) == 24);

// A guest XAsyncProvider function behind the framework. Its XAsyncProviderData lives in guest
// memory for the operation's lifetime; `async` is the guest block, `buffer` (GetResult) the guest
// buffer. DoWork is a queue callback and runs as a top-level callback; Begin, GetResult, Cancel
// and Cleanup run in the guest frames that caused them. A provider call left by an unwind (an
// exception caught by an outer guest frame) counts as failed with E_ABORT.
class GuestProvider final : public XAsyncProviderImpl {
public:
    GuestProvider(uint64_t function, uint64_t guest_block, uint64_t context) : function_(function) {
        data_ = com::guest_alloc(sizeof(GuestAsyncProviderData));
        const GuestAsyncProviderData data{guest_block, 0, 0, context};
        if (data_ != 0 && !com::write_guest(data_, data)) {
            com::guest_free(data_);
            data_ = 0;
        }
    }
    ~GuestProvider() override {
        if (data_ != 0) {
            com::guest_free(data_);
        }
    }
    bool ok() const { return data_ != 0; }

    HResult call(XAsyncOp op, XAsyncBlock* /*shadow*/, size_t buffer_size, uint64_t buffer) override {
        if (op == XAsyncOp::GetResult) {
            com::write_guest(data_ + offsetof(GuestAsyncProviderData, buffer_size), static_cast<uint64_t>(buffer_size));
            com::write_guest(data_ + offsetof(GuestAsyncProviderData, buffer), buffer);
        }
        playgta6::GuestState& state = com::current_guest_state();
        const com::GuestFunction<int32_t(uint32_t, uint64_t)> provider(function_);
        const auto code = static_cast<uint32_t>(op);
        const int32_t hr = op == XAsyncOp::DoWork ? provider.call_top_level(state, code, data_)
                                                  : provider.call(state, code, data_);
        return playgta6::guest::unwind_pending(state) ? E_ABORT_HR : hr;
    }

private:
    uint64_t function_ = 0;
    uint64_t data_ = 0;
};

std::string guest_string(uint64_t va, size_t max) {
    std::string text;
    for (uint64_t i = 0; va != 0 && i < max; ++i) {
        char c = 0;
        if (!com::read_guest(va + i, &c, 1) || c == 0) {
            break;
        }
        text.push_back(c);
    }
    return text;
}

// ---------------------------------------------------------------------------
// Delayed task queue callbacks
// ---------------------------------------------------------------------------

// XTaskQueueSubmitDelayedCallback: the O5 queue has no timers, so delayed submissions wait on
// one scheduler thread and are submitted when due (a zero delay submits immediately).
class DelayScheduler {
public:
    struct Item {
        std::chrono::steady_clock::time_point due;
        Pg6XTaskQueueHandle queue;
        uint32_t port;
        cbk::HostCallback<Pg6XTaskQueueCallback> callback;
    };

    void schedule(Item item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable()) {
            thread_ = std::thread([this] { run(); });
        }
        pg6_gdk_xtask_queue_duplicate_handle(item.queue, &item.queue);  // keep the queue alive
        items_.emplace(item.due, item);
        cv_.notify_all();
    }

    ~DelayScheduler() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            cv_.notify_all();
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_) {
            if (items_.empty()) {
                cv_.wait(lock);
                continue;
            }
            auto it = items_.begin();
            if (std::chrono::steady_clock::now() < it->first) {
                cv_.wait_until(lock, it->first);
                continue;
            }
            Item item = it->second;
            items_.erase(it);
            lock.unlock();
            if (pg6_gdk_xtask_queue_submit_callback(item.queue, item.port, item.callback.context,
                                                    item.callback.function) < 0) {
                cbk::destroy_trampoline(item.callback.trampoline);
            }
            pg6_gdk_xtask_queue_close_handle(item.queue);
            lock.lock();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::multimap<std::chrono::steady_clock::time_point, Item> items_;
    std::thread thread_;
    bool stop_ = false;
};

DelayScheduler& delay_scheduler() {
    static DelayScheduler instance;
    return instance;
}

// ---------------------------------------------------------------------------
// Interface implementations
// ---------------------------------------------------------------------------

// {073b7dcb-1fcf-4030-94be-e3c9eb623428}: XAsync + XTaskQueue (+ XThread).
class AsyncQueueApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XAsync/XTaskQueue";

    int32_t GetStatus(uint64_t block, bool wait) {
        Pg6XAsyncBlock* shadow = find_async(block);
        return shadow != nullptr ? pg6_gdk_xasync_get_status(shadow, wait ? 1 : 0) : E_INVALIDARG_HR;
    }

    int32_t GetResultSize(uint64_t block, GuestPtr<uint64_t> size) {
        Pg6XAsyncBlock* shadow = find_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        size_t host_size = 0;
        const int32_t hr = pg6_gdk_xasync_get_result_size(shadow, &host_size);
        if (!size.is_null()) {
            size.write(static_cast<uint64_t>(host_size));
        }
        return hr;
    }

    int32_t Create(uint32_t work_mode, uint32_t completion_mode, OutPtr queue) {
        if (queue.is_null()) {
            return E_POINTER_HR;
        }
        Pg6XTaskQueueHandle handle = nullptr;
        const int32_t hr = pg6_gdk_xtask_queue_create(work_mode, completion_mode, &handle);
        queues().add(handle);
        queue.write(to_guest(handle));
        return hr;
    }

    int32_t DuplicateHandle(uint64_t queue, OutPtr duplicated) {
        if (duplicated.is_null()) {
            return E_POINTER_HR;
        }
        duplicated.write(0);
        if (!check_handle(queues(), queue, "XTaskQueue handle")) {
            return E_INVALIDARG_HR;
        }
        Pg6XTaskQueueHandle handle = nullptr;
        const int32_t hr = pg6_gdk_xtask_queue_duplicate_handle(from_guest<Pg6XTaskQueue>(queue), &handle);
        queues().add(handle);
        duplicated.write(to_guest(handle));
        return hr;
    }

    bool Dispatch(uint64_t queue, uint32_t port, uint32_t timeout_ms) {
        if (!check_handle(queues(), queue, "XTaskQueue handle")) {
            return false;
        }
        return pg6_gdk_xtask_queue_dispatch(from_guest<Pg6XTaskQueue>(queue), port, timeout_ms) != 0;
    }

    void CloseHandle(uint64_t queue) {
        if (check_handle(queues(), queue, "XTaskQueue handle")) {
            queues().remove(from_guest<void>(queue));
            pg6_gdk_xtask_queue_close_handle(from_guest<Pg6XTaskQueue>(queue));
        }
    }

    int32_t SubmitDelayedCallback(uint64_t queue, uint32_t port, uint32_t delay_ms, uint64_t context,
                                  uint64_t callback) {
        if (callback == 0 || !check_handle(queues(), queue, "XTaskQueue handle")) {
            return E_INVALIDARG_HR;
        }
        cbk::HostCallback<Pg6XTaskQueueCallback> cb = cbk::adapt_task_queue_callback(callback, context);
        auto* host_queue = from_guest<Pg6XTaskQueue>(queue);
        if (delay_ms == 0) {
            const int32_t hr = pg6_gdk_xtask_queue_submit_callback(host_queue, port, cb.context, cb.function);
            if (hr < 0) {
                cbk::destroy_trampoline(cb.trampoline);
            }
            return hr;
        }
        delay_scheduler().schedule(DelayScheduler::Item{
            std::chrono::steady_clock::now() + std::chrono::milliseconds(delay_ms), host_queue, port, cb});
        return S_OK_HR;
    }

    int32_t Terminate(uint64_t queue, bool wait, uint64_t context, uint64_t callback) {
        if (!check_handle(queues(), queue, "XTaskQueue handle")) {
            return E_INVALIDARG_HR;
        }
        auto* host_queue = from_guest<Pg6XTaskQueue>(queue);
        if (callback == 0) {
            return pg6_gdk_xtask_queue_terminate(host_queue, wait ? 1 : 0, nullptr, nullptr);
        }
        cbk::HostCallback<Pg6XTaskQueueTerminatedCallback> cb =
            cbk::adapt_task_queue_terminated_callback(callback, context);
        const int32_t hr = pg6_gdk_xtask_queue_terminate(host_queue, wait ? 1 : 0, cb.context, cb.function);
        if (hr < 0) {
            cbk::destroy_trampoline(cb.trampoline);
        }
        return hr;
    }

    bool GetCurrentProcessTaskQueue(OutPtr queue) {
        if (queue.is_null()) {
            return false;
        }
        Pg6XTaskQueueHandle handle = nullptr;
        const bool ok = pg6_gdk_xtask_queue_get_current_process_task_queue(&handle) != 0;
        queues().add(handle);
        queue.write(to_guest(handle));
        return ok;
    }

    // XThreadAssertNotTimeSensitive (called at the top of 58 API wrappers): nothing to check.
    void AssertNotTimeSensitive() {}

    // XAsync provider API (O22, os/xasync_provider.h) on the guest block's shadow.
    void Cancel(uint64_t block) {
        if (Pg6XAsyncBlock* shadow = find_async(block)) {
            XAsyncCancel(shadow);
        }
    }

    int32_t Run(uint64_t block, uint64_t work) {
        if (work == 0) {
            return E_INVALIDARG_HR;
        }
        Pg6XAsyncBlock* shadow = begin_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        return XAsyncRunWith(shadow, [work, block](XAsyncBlock*) -> HResult {
            // DoWork of the queue: a top-level callback.
            return com::GuestFunction<int32_t(uint64_t)>(work).call_top_level(com::current_guest_state(), block);
        });
    }

    int32_t Begin(uint64_t block, uint64_t context, uint64_t identity, uint64_t identity_name, uint64_t provider) {
        if (provider == 0) {
            return E_INVALIDARG_HR;
        }
        Pg6XAsyncBlock* shadow = begin_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        auto adapter = std::make_unique<GuestProvider>(provider, block, context);
        if (!adapter->ok()) {
            return E_OUTOFMEMORY_HR;
        }
        const std::string name = guest_string(identity_name, 128);
        return XAsyncBeginWith(shadow, std::move(adapter), identity, name.c_str());
    }

    int32_t Schedule(uint64_t block, uint32_t delay_ms) {
        Pg6XAsyncBlock* shadow = find_async(block);
        return shadow != nullptr ? XAsyncSchedule(shadow, delay_ms) : E_INVALIDARG_HR;
    }

    void Complete(uint64_t block, int32_t result, uint64_t required_buffer_size) {
        if (Pg6XAsyncBlock* shadow = find_async(block)) {
            XAsyncComplete(shadow, result, static_cast<size_t>(required_buffer_size));
        }
    }

    int32_t GetResult(uint64_t block, uint64_t identity, uint64_t buffer_size, uint64_t buffer,
                      GuestPtr<uint64_t> buffer_used) {
        if (!buffer_used.is_null()) {
            buffer_used.write(0);
        }
        Pg6XAsyncBlock* shadow = find_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        size_t used = 0;
        const int32_t hr = XAsyncGetResultWith(shadow, identity, static_cast<size_t>(buffer_size), buffer, &used);
        if (!buffer_used.is_null()) {
            buffer_used.write(static_cast<uint64_t>(used));
        }
        return hr;
    }

    int32_t CreateComposite(uint64_t work_port, uint64_t completion_port, OutPtr queue) {
        if (queue.is_null()) {
            return E_POINTER_HR;
        }
        queue.write(0);
        if (!check_handle(ports(), work_port, "XTaskQueuePortHandle") ||
            !check_handle(ports(), completion_port, "XTaskQueuePortHandle")) {
            return E_INVALIDARG_HR;
        }
        Pg6XTaskQueueHandle handle = nullptr;
        const int32_t hr = pg6_gdk_xtask_queue_create_composite(from_guest<Pg6XTaskQueuePort>(work_port),
                                                                from_guest<Pg6XTaskQueuePort>(completion_port), &handle);
        queues().add(handle);
        queue.write(to_guest(handle));
        return hr;
    }

    int32_t GetPort(uint64_t queue, uint32_t port, OutPtr port_handle) {
        if (port_handle.is_null()) {
            return E_POINTER_HR;
        }
        port_handle.write(0);
        if (!check_handle(queues(), queue, "XTaskQueue handle")) {
            return E_INVALIDARG_HR;
        }
        Pg6XTaskQueuePortHandle handle = nullptr;
        const int32_t hr = pg6_gdk_xtask_queue_get_port(from_guest<Pg6XTaskQueue>(queue), port, &handle);
        if (handle != nullptr && !ports().contains(handle)) {
            ports().add(handle);
        }
        port_handle.write(to_guest(handle));
        return hr;
    }
};

// {01acd177-91f9-4763-a38e-ccbb55ce32e0}: XUser.
class UserApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XUser";

    void CloseHandle(uint64_t user) {
        if (check_handle(users(), user, "XUserHandle")) {
            users().remove(from_guest<void>(user));
            pg6_gdk_xuser_close_handle(from_guest<Pg6XUser>(user));
        }
    }

    int32_t AddAsync(uint32_t options, uint64_t block) {
        Pg6XAsyncBlock* shadow = begin_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        return pg6_gdk_xuser_add_async(options, shadow);
    }

    int32_t AddResult(uint64_t block, OutPtr user) {
        if (user.is_null()) {
            return E_POINTER_HR;
        }
        user.write(0);
        Pg6XAsyncBlock* shadow = find_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        Pg6XUserHandle handle = nullptr;
        const int32_t hr = pg6_gdk_xuser_add_result(shadow, &handle);
        if (hr != E_PENDING_HR) {
            cbk::async_shadow_end(block);
        }
        users().add(handle);
        user.write(to_guest(handle));
        return hr;
    }

    int32_t GetId(uint64_t user, GuestPtr<uint64_t> id) {
        if (!check_handle(users(), user, "XUserHandle")) {
            return E_INVALIDARG_HR;
        }
        uint64_t xuid = 0;
        const int32_t hr = pg6_gdk_xuser_get_id(from_guest<Pg6XUser>(user), &xuid);
        if (id.is_null()) {
            return E_INVALIDARG_HR;
        }
        id.write(xuid);
        return hr;
    }

    int32_t CheckPrivilege(uint64_t user, uint32_t options, uint32_t privilege, GuestPtr<uint8_t> has_privilege,
                           GuestPtr<uint32_t> deny_reasons) {
        if (!check_handle(users(), user, "XUserHandle")) {
            return E_INVALIDARG_HR;
        }
        bool has = false;
        uint32_t reason = 0;
        const int32_t hr =
            pg6_gdk_xuser_check_privilege(from_guest<Pg6XUser>(user), options, privilege, &has, &reason);
        has_privilege.write(has ? 1 : 0);
        deny_reasons.write(reason);
        return hr;
    }

    int32_t ResolvePrivilegeWithUiAsync(uint64_t user, uint32_t options, uint32_t privilege, uint64_t block) {
        if (!check_handle(users(), user, "XUserHandle")) {
            return E_INVALIDARG_HR;
        }
        Pg6XAsyncBlock* shadow = begin_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        return pg6_gdk_xuser_resolve_privilege_with_ui_async(from_guest<Pg6XUser>(user), options, privilege,
                                                             shadow);
    }

    int32_t ResolvePrivilegeWithUiResult(uint64_t block) {
        Pg6XAsyncBlock* shadow = find_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        const int32_t hr = pg6_gdk_xuser_resolve_privilege_with_ui_result(shadow);
        if (hr != E_PENDING_HR) {
            cbk::async_shadow_end(block);
        }
        return hr;
    }

    int32_t GetTokenAndSignatureAsync(uint64_t user, uint32_t options, const char* method, const char* url,
                                      uint64_t header_count, const void* headers, uint64_t body_size,
                                      const void* body, uint64_t block) {
        if (!check_handle(users(), user, "XUserHandle")) {
            return E_INVALIDARG_HR;
        }
        Pg6XAsyncBlock* shadow = begin_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        // Offline: the shim completes with the offline result without reading the request.
        return pg6_gdk_xuser_get_token_and_signature_async(from_guest<Pg6XUser>(user), options, method, url,
                                                           static_cast<size_t>(header_count), headers,
                                                           static_cast<size_t>(body_size), body, shadow);
    }

    int32_t GetTokenAndSignatureResultSize(uint64_t block, GuestPtr<uint64_t> size) {
        Pg6XAsyncBlock* shadow = find_async(block);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        size_t host_size = 0;
        const int32_t hr = pg6_gdk_xuser_get_token_and_signature_result_size(shadow, &host_size);
        size.write(static_cast<uint64_t>(host_size));
        return hr;
    }

    // Local offline user: there are no user changes to report, the registration only hands out a
    // token (as for XNetworking's connectivity hint).
    int32_t RegisterForChangeEvent(uint64_t queue, uint64_t /*context*/, uint64_t callback,
                                   GuestPtr<uint64_t> token) {
        if (callback == 0 || token.is_null() || !optional_queue_ok(queue)) {
            return E_INVALIDARG_HR;
        }
        token.write(next_token_++);
        return S_OK_HR;
    }

    bool UnregisterForChangeEvent(uint64_t /*token*/, bool /*wait*/) { return true; }

private:
    std::atomic<uint64_t> next_token_{1};
};

// {01acd177-...}/{cef4fac0-7676-4a94-a119-4c43f9eb5b74}: XUser extension.
class UserExtApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XUser extension";

    int32_t GetGamertag(uint64_t user, uint32_t component, uint64_t size, uint64_t buffer,
                        GuestPtr<uint64_t> used) {
        if (!check_handle(users(), user, "XUserHandle")) {
            return E_INVALIDARG_HR;
        }
        char* host_buffer = nullptr;
        if (buffer != 0) {
            host_buffer = static_cast<char*>(com::guest_to_host(buffer, static_cast<size_t>(size)));
            if (host_buffer == nullptr) {
                com::detail::log_bad_pointer("XUserGetGamertag buffer", buffer);
                return E_INVALIDARG_HR;
            }
        }
        size_t host_used = 0;
        const int32_t hr = pg6_gdk_xuser_get_gamertag(from_guest<Pg6XUser>(user), component,
                                                      static_cast<size_t>(size), host_buffer, &host_used);
        used.write(static_cast<uint64_t>(host_used));
        return hr;
    }
};

// {8836fe87-edb9-4fe3-8dad-05f0d2cd5b40}: XGameRuntimeIsFeatureAvailable. No optional runtime
// feature is advertised.
class FeatureApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XGameRuntimeFeature";
    bool IsFeatureAvailable(uint32_t /*feature*/) { return false; }
};

// {e349bd1a-...}/{1861cf2e-...}: XSystem.
class SystemApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XSystem";

    // Deterministic local console id: all zero digits (size includes the terminator).
    int32_t GetConsoleId(uint64_t size, uint64_t buffer, GuestPtr<uint64_t> used) {
        constexpr uint64_t kConsoleIdBytes = 39;
        used.write(kConsoleIdBytes);
        if (size < kConsoleIdBytes || buffer == 0) {
            return E_NOT_SUFFICIENT_BUFFER_HR;
        }
        char id[kConsoleIdBytes];
        std::memset(id, '0', sizeof(id) - 1);
        id[kConsoleIdBytes - 1] = '\0';
        return com::write_guest(buffer, id, sizeof(id)) ? S_OK_HR : E_INVALIDARG_HR;
    }

    // s4 XSystemGetXboxLiveSandboxId(size, buffer, used): the runtime is offline and has no Xbox
    // Live sandbox, so it reports none: E_NOTIMPL, an empty string and used = 0 (every caller
    // in the executable takes its failure path; no sandbox name is invented).
    int32_t GetXboxLiveSandboxId(uint64_t size, uint64_t buffer, GuestPtr<uint64_t> used) {
        if (!used.is_null()) {
            used.write(0);
        }
        if (buffer != 0 && size != 0) {
            const char nul = '\0';
            com::write_guest(buffer, &nul, 1);
        }
        if (!sandbox_logged_.exchange(true)) {
            os::log("xgameruntime: XSystemGetXboxLiveSandboxId: offline, no sandbox; returning E_NOTIMPL");
        }
        return com::E_NOTIMPL_HR;
    }

private:
    std::atomic<bool> sandbox_logged_{false};
};

std::atomic<uint32_t> g_title_id{0};

// {973a344e-...}/{2549f142-...}: XGame.
class GameApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XGame";

    int32_t GetXboxTitleId(GuestPtr<uint32_t> title_id) {
        if (title_id.is_null()) {
            return E_INVALIDARG_HR;
        }
        title_id.write(g_title_id.load(std::memory_order_relaxed));
        return S_OK_HR;
    }
};

// {0dd112ac-...}/{b09d803c-...}: XStore (contexts only; queries keep the O8 offline policy and are
// stubs until a boot shows which ones are needed).
class StoreApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XStore";

    int32_t CreateContext(uint64_t user, OutPtr context) {
        if (context.is_null()) {
            return E_POINTER_HR;
        }
        context.write(0);
        if (user != 0 && !check_handle(users(), user, "XUserHandle")) {
            return E_INVALIDARG_HR;
        }
        Pg6XStoreContextHandle handle = nullptr;
        const int32_t hr = pg6_gdk_xstore_create_context(from_guest<Pg6XUser>(user), &handle);
        stores().add(handle);
        context.write(to_guest(handle));
        return hr;
    }

    void CloseContextHandle(uint64_t context) {
        if (check_handle(stores(), context, "XStoreContextHandle")) {
            stores().remove(from_guest<void>(context));
            pg6_gdk_xstore_close_context_handle(from_guest<Pg6XStoreContext>(context));
        }
    }
};

void ignore_connectivity_change(void* /*context*/, const Pg6XNetworkingConnectivityHint* /*hint*/) {}

// {37e56907-...}/{bf2346b2-...}: XNetworking.
class NetworkingApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XNetworking";

    int32_t GetConnectivityHint(GuestPtr<Pg6XNetworkingConnectivityHint> hint) {
        Pg6XNetworkingConnectivityHint value{};
        const int32_t hr = pg6_gdk_xnetworking_get_connectivity_hint(&value);
        if (hint.is_null()) {
            return E_INVALIDARG_HR;
        }
        hint.write(value);
        return hr;
    }

    // The O8 shim never fires change notifications, so the guest callback is not kept.
    int32_t RegisterConnectivityHintChanged(uint64_t queue, uint64_t /*context*/, uint64_t callback,
                                            GuestPtr<uint64_t> token) {
        if (callback == 0 || token.is_null() || !optional_queue_ok(queue)) {
            return E_INVALIDARG_HR;
        }
        Pg6XTaskQueueRegistrationToken host_token{};
        const int32_t hr = pg6_gdk_xnetworking_register_connectivity_hint_changed(
            from_guest<Pg6XTaskQueue>(queue), nullptr, &ignore_connectivity_change, &host_token);
        token.write(host_token.token);
        return hr;
    }

    bool UnregisterConnectivityHintChanged(uint64_t token, bool wait) {
        Pg6XTaskQueueRegistrationToken host_token{};
        host_token.token = token;
        return pg6_gdk_xnetworking_unregister_connectivity_hint_changed(host_token, wait ? 1 : 0) != 0;
    }
};

// XPackageInstallationProgress (x64): totalBytes, installedBytes, launchable, completed.
struct GuestInstallationProgress {
    uint64_t total_bytes;
    uint64_t installed_bytes;
    uint8_t launchable;
    uint8_t completed;
    uint8_t padding[6];
};
static_assert(sizeof(GuestInstallationProgress) == 24, "XPackageInstallationProgress is 24 bytes on x64");

// {af406016-...}/{0b9d556e-...}: XPackage.
class PackageApi : public com::ComObject {
    struct InstallationMonitor {
        uint64_t unused = 0;
    };
    // Equal non-zero byte counts: their ratio is 1, never 0/0.
    static constexpr uint64_t kInstalledBytes = 1;
    std::atomic<uint64_t> next_progress_token_{1};

public:
    static constexpr const char* kComClassName = "xgameruntime XPackage";
    bool IsPackagedProcess() { return true; }

    // XPackageGetCurrentProcessPackageIdentifier(bufferSize, buffer): a deterministic local
    // identifier, 32 zero digits (XPACKAGE_IDENTIFIER_MAX_LENGTH, 33 bytes with the terminator),
    // like XSystemGetConsoleId. No package, licence or store state is derived from it here.
    int32_t GetCurrentProcessPackageIdentifier(uint64_t size, uint64_t buffer) {
        constexpr uint64_t kIdentifierBytes = 33;
        if (buffer == 0) {
            return E_INVALIDARG_HR;
        }
        if (size < kIdentifierBytes) {
            return E_NOT_SUFFICIENT_BUFFER_HR;
        }
        char id[kIdentifierBytes];
        std::memset(id, '0', sizeof(id) - 1);
        id[kIdentifierBytes - 1] = '\0';
        return com::write_guest(buffer, id, sizeof(id)) ? S_OK_HR : E_INVALIDARG_HR;
    }

    // XPackageCreateInstallationMonitor(packageIdentifier, selectorCount, selectors,
    // minimumUpdateIntervalMs, queue, installationMonitor*). The package is installed in full on
    // the local device, so a monitor only ever reports a finished installation and never fires.
    int32_t CreateInstallationMonitor(uint64_t /*identifier*/, uint64_t selector_count, uint64_t selectors,
                                      uint64_t /*interval_ms*/, uint64_t queue, OutPtr monitor) {
        if (monitor.is_null()) {
            return E_POINTER_HR;
        }
        monitor.write(0);
        if ((static_cast<uint32_t>(selector_count) != 0 && selectors == 0) || !optional_queue_ok(queue)) {
            return E_INVALIDARG_HR;
        }
        auto* handle = new InstallationMonitor{};
        installation_monitors().add(handle);
        monitor.write(to_guest(handle));
        return S_OK_HR;
    }

    void CloseInstallationMonitorHandle(uint64_t monitor) {
        if (check_handle(installation_monitors(), monitor, "XPackageInstallationMonitorHandle")) {
            installation_monitors().remove(from_guest<void>(monitor));
            delete from_guest<InstallationMonitor>(monitor);
        }
    }

    // XPackageRegisterInstallationProgressChanged(monitor, context, callback, token*): a finished
    // installation never changes, so the callback is not kept and never runs.
    int32_t RegisterInstallationProgressChanged(uint64_t monitor, uint64_t /*context*/, uint64_t callback,
                                                GuestPtr<uint64_t> token) {
        if (!check_handle(installation_monitors(), monitor, "XPackageInstallationMonitorHandle") ||
            callback == 0 || token.is_null()) {
            return E_INVALIDARG_HR;
        }
        token.write(next_progress_token_.fetch_add(1, std::memory_order_relaxed));
        return S_OK_HR;
    }

    // XPackageEnumeratePackages(kind, scope, context, callback). Content packages (kind 1): none
    // are installed locally, so the callback never runs. Game packages (kind 0) are not modelled
    // yet and fail with E_NOTIMPL, so a boot that needs them shows it.
    int32_t EnumeratePackages(uint64_t kind, uint64_t /*scope*/, uint64_t /*context*/, uint64_t callback) {
        constexpr uint32_t kXPackageKindContent = 1;
        if (callback == 0) {
            return E_INVALIDARG_HR;
        }
        if (static_cast<uint32_t>(kind) != kXPackageKindContent) {
            os::log("xgameruntime: XPackageEnumeratePackages(kind " + std::to_string(static_cast<uint32_t>(kind)) +
                    ") is not modelled; returning E_NOTIMPL");
            return com::E_NOTIMPL_HR;
        }
        return S_OK_HR;
    }

    // XPackageGetInstallationProgress(monitor, progress*): complete and launchable.
    void GetInstallationProgress(uint64_t monitor, GuestPtr<GuestInstallationProgress> progress) {
        if (!check_handle(installation_monitors(), monitor, "XPackageInstallationMonitorHandle") ||
            progress.is_null()) {
            return;
        }
        GuestInstallationProgress value{};
        value.total_bytes = kInstalledBytes;
        value.installed_bytes = kInstalledBytes;
        value.launchable = 1;
        value.completed = 1;
        progress.write(value);
    }
};

// {f4faf4d4-...}/{41a4e10c-...}: XPersistentLocalStorage. The storage is a directory of the
// runtime's writable /userdata overlay.
class PersistentLocalStorageApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XPersistentLocalStorage";
    static constexpr char kPath[] = "T:\\userdata\\pls\\";

    // XPersistentLocalStorageGetPath(pathSize, path, pathUsed*): size in bytes with the terminator.
    int32_t GetPath(uint64_t size, uint64_t path, GuestPtr<uint64_t> used) {
        constexpr uint64_t kBytes = sizeof(kPath);
        if (!used.is_null()) {
            used.write(kBytes);
        }
        if (path == 0) {
            return E_INVALIDARG_HR;
        }
        if (size < kBytes) {
            return E_NOT_SUFFICIENT_BUFFER_HR;
        }
        win32::userdata_create_directory("/userdata/pls");
        return com::write_guest(path, kPath, kBytes) ? S_OK_HR : E_INVALIDARG_HR;
    }
};

// {03f0fe74-...}/{35f07670-...}: slot 4 is called through a two-argument wrapper (a preference
// enum, a pointer to a 12-byte result) whose enum result the caller compares with 1 before taking
// an HDR set-up path: the shape of XDisplayTryEnableHdrMode(preference, XDisplayHdrModeInfo*) ->
// XDisplayHdrModeResult {Unknown, Enabled, Disabled}. The runtime presents SDR: Disabled.
class HdrModeApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XDisplay (HDR mode)";
    static constexpr uint32_t kHdrModeDisabled = 2;

    uint32_t TryEnableHdrMode(uint64_t /*preference*/, uint64_t info) {
        if (info != 0) {
            const float zero[3] = {0.0f, 0.0f, 0.0f};  // min / max / max full-frame tone-map nits
            com::write_guest(info, zero, sizeof(zero));
        }
        return kHdrModeDisabled;
    }
};

// XSpeechSynthesizer. The runtime has no speech synthesizer: creation fails, and the title's
// own failure path runs (no handle is invented).
class SpeechApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XSpeechSynthesizer";

    // s9: XSpeechSynthesizerCreate(XSpeechSynthesizerHandle*). The executable passes the address
    // of a handle field that is still null and runs its setup only on success.
    int32_t Create(GuestPtr<uint64_t> handle) {
        if (handle.is_null()) {
            return E_INVALIDARG_HR;
        }
        handle.write(0);
        if (!logged_.exchange(true)) {
            os::log("xgameruntime: XSpeechSynthesizerCreate: no speech synthesizer; returning E_NOTIMPL");
        }
        return com::E_NOTIMPL_HR;
    }

private:
    std::atomic<bool> logged_{false};
};

// XSystemAnalyticsInfo (x64): two XVersion (four uint16 each) and two 64-byte strings.
struct GuestAnalyticsInfo {
    uint16_t os_version[4];       // major, minor, build, revision
    uint16_t hosting_version[4];
    char family[64];
    char form[64];
};
static_assert(sizeof(GuestAnalyticsInfo) == 0x90, "XSystemAnalyticsInfo is 144 bytes");

// {b884675d-...}: XSystemGetAnalyticsInfo (returned by value through the hidden result pointer).
// A fixed, local description of the emulated console (the boot's default console type).
class AnalyticsInfoApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XSystemAnalyticsInfo";

    uint64_t GetAnalyticsInfo(uint64_t result) {
        GuestAnalyticsInfo info{};
        info.os_version[0] = 10;
        info.os_version[2] = 22621;
        info.hosting_version[0] = 10;
        info.hosting_version[2] = 22621;
        std::strncpy(info.family, "Windows.Xbox", sizeof(info.family) - 1);
        std::strncpy(info.form, "Xbox Series X", sizeof(info.form) - 1);
        if (result != 0) {
            com::write_guest(result, &info, sizeof(info));
        }
        return result;
    }
};

// Interfaces whose slots are all logging stubs.
class StubApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime (stub interface)";
};

// ---------------------------------------------------------------------------
// Interface declarations
// ---------------------------------------------------------------------------

template <typename Host>
using M = com::Methods<Host>;

// Builds a method table for slots first..first+N-1 from {slot, desc} pairs (gaps are stubs).
template <size_t N>
struct SlotTable {
    com::MethodDesc slots[N];
};

template <size_t N>
constexpr SlotTable<N> slot_table(std::initializer_list<std::pair<uint32_t, com::MethodDesc>> entries) {
    SlotTable<N> table{};
    for (const auto& [slot, desc] : entries) {
        table.slots[slot - 3] = desc;
    }
    return table;
}

constexpr Guid kAsyncQueueGuid = com::make_guid("073b7dcb-1fcf-4030-94be-e3c9eb623428");
constexpr Guid kUserGuid = com::make_guid("01acd177-91f9-4763-a38e-ccbb55ce32e0");
constexpr Guid kUserExtIid = com::make_guid("cef4fac0-7676-4a94-a119-4c43f9eb5b74");
constexpr Guid kFeatureGuid = com::make_guid("8836fe87-edb9-4fe3-8dad-05f0d2cd5b40");
constexpr Guid kSystemProvider = com::make_guid("e349bd1a-fc20-4e40-b99c-4178cc6b409f");
constexpr Guid kSystemIid = com::make_guid("1861cf2e-e18b-4834-a9f5-b4a4e6efb4cf");
constexpr Guid kGameProvider = com::make_guid("973a344e-24bf-4d0f-8457-56c534892b29");
constexpr Guid kGameIid = com::make_guid("2549f142-6419-4a06-97b5-931aab7c2f34");
constexpr Guid kStoreProvider = com::make_guid("0dd112ac-7c24-448c-b92b-3960fb5bd30c");
constexpr Guid kStoreIid = com::make_guid("b09d803c-2414-4a05-82c6-66dfdc9e9a44");
constexpr Guid kStoreExtIid = com::make_guid("2d42fea5-e71d-4b76-97cd-c50afbb3ae5d");
constexpr Guid kNetworkingProvider = com::make_guid("37e56907-2f10-41e8-b72f-36edb185331a");
constexpr Guid kNetworkingIid = com::make_guid("bf2346b2-39af-4658-b5ea-44713c7e83b3");
constexpr Guid kPackageProvider = com::make_guid("af406016-e850-4aa8-a88d-2f3dcb9dac7e");
constexpr Guid kPackageIid = com::make_guid("0b9d556e-ab70-43f4-baa0-9999ac4cb040");
constexpr Guid kGameSaveProvider = com::make_guid("704c3f58-e629-4cc2-b197-30511b996fe2");
constexpr Guid kGameSaveIid = com::make_guid("1bfff3af-f14a-40a3-8e35-9ada906593f9");
constexpr Guid kPlsProvider = com::make_guid("f4faf4d4-2d04-4fce-b3e0-474a713a3e84");
constexpr Guid kPlsIid = com::make_guid("41a4e10c-5a7e-41d9-8c37-37bde62a07d6");
constexpr Guid kSpeechProvider = com::make_guid("3e241977-6237-41e9-8559-102c2d9983f1");
constexpr Guid kSpeechIid = com::make_guid("d722b373-8c4d-4692-9e51-d6ad9b37aa7d");
constexpr Guid kEventGuid = com::make_guid("bbfbdcc7-bfe7-409b-a5ca-edf054960b4d");
constexpr Guid kDisplayGuid = com::make_guid("1b339674-328d-4283-a200-3171f18d3639");
constexpr Guid kUnmapped3Provider = com::make_guid("03f0fe74-fdd9-4e5c-b630-f9339c47acc5");
constexpr Guid kUnmapped3Iid = com::make_guid("35f07670-706e-4bfb-9476-090798c5ebf3");
constexpr Guid kUnmapped9Provider = com::make_guid("a4f1aee2-4bf1-4485-b008-a7c26d52ac27");
constexpr Guid kUnmapped9Iid = com::make_guid("22e672d7-b4e3-406c-bd50-8f0d25236f9e");
constexpr Guid kUnmapped10Guid = com::make_guid("b884675d-b738-4a9c-815d-9a9a1e0c6c9b");
constexpr Guid kUnmapped13Provider = com::make_guid("dfcd4649-4ff8-4043-ba07-35d607df98b0");
constexpr Guid kUnmapped13Iid = com::make_guid("eaf669df-5542-4590-99a3-8dc061f837cc");

using AQ = M<AsyncQueueApi>;
constexpr auto kAsyncQueueTable = slot_table<27>({
    {3, AQ::method<&AsyncQueueApi::GetStatus>("XAsyncGetStatus")},
    {4, AQ::method<&AsyncQueueApi::GetResultSize>("XAsyncGetResultSize")},
    {5, AQ::method<&AsyncQueueApi::Cancel>("XAsyncCancel")},
    {6, AQ::method<&AsyncQueueApi::Run>("XAsyncRun")},
    {7, AQ::method<&AsyncQueueApi::Begin>("XAsyncBegin")},
    {9, AQ::method<&AsyncQueueApi::Schedule>("XAsyncSchedule")},
    {10, AQ::method<&AsyncQueueApi::Complete>("XAsyncComplete")},
    {11, AQ::method<&AsyncQueueApi::GetResult>("XAsyncGetResult")},
    {12, AQ::method<&AsyncQueueApi::Create>("XTaskQueueCreate")},
    {13, AQ::method<&AsyncQueueApi::CreateComposite>("XTaskQueueCreateComposite")},
    {14, AQ::method<&AsyncQueueApi::GetPort>("XTaskQueueGetPort")},
    {15, AQ::method<&AsyncQueueApi::DuplicateHandle>("XTaskQueueDuplicateHandle")},
    {16, AQ::method<&AsyncQueueApi::Dispatch>("XTaskQueueDispatch")},
    {17, AQ::method<&AsyncQueueApi::CloseHandle>("XTaskQueueCloseHandle")},
    {19, AQ::method<&AsyncQueueApi::SubmitDelayedCallback>("XTaskQueueSubmitDelayedCallback")},
    {22, AQ::method<&AsyncQueueApi::Terminate>("XTaskQueueTerminate")},
    {25, AQ::method<&AsyncQueueApi::GetCurrentProcessTaskQueue>("XTaskQueueGetCurrentProcessTaskQueue")},
    {29, AQ::method<&AsyncQueueApi::AssertNotTimeSensitive>("XThreadAssertNotTimeSensitive")},
});
constexpr auto kIAsyncQueue =
    com::declare_interface<AsyncQueueApi>("xgameruntime.XAsyncTaskQueue", kAsyncQueueGuid, kAsyncQueueTable.slots,
                                          {.slot_count = 40});

using US = M<UserApi>;
constexpr auto kUserTable = slot_table<32>({
    {3, US::stub("XUserDuplicateHandle")},
    {4, US::method<&UserApi::CloseHandle>("XUserCloseHandle")},
    {5, US::stub("XUserCompare")},
    {7, US::method<&UserApi::AddAsync>("XUserAddAsync")},
    {8, US::method<&UserApi::AddResult>("XUserAddResult")},
    {9, US::stub("XUserAddByIdWithUiAsync")},
    {10, US::stub("XUserAddByIdWithUiResult")},
    {11, US::method<&UserApi::GetId>("XUserGetId")},
    {12, US::stub("XUserFindUserById")},
    {13, US::stub("XUserGetLocalId")},
    {14, US::stub("XUserFindUserByLocalId")},
    {19, US::stub("XUserGetAgeGroup")},
    {20, US::method<&UserApi::CheckPrivilege>("XUserCheckPrivilege")},
    {21, US::method<&UserApi::ResolvePrivilegeWithUiAsync>("XUserResolvePrivilegeWithUiAsync")},
    {22, US::method<&UserApi::ResolvePrivilegeWithUiResult>("XUserResolvePrivilegeWithUiResult")},
    {23, US::method<&UserApi::GetTokenAndSignatureAsync>("XUserGetTokenAndSignatureAsync")},
    {24, US::method<&UserApi::GetTokenAndSignatureResultSize>("XUserGetTokenAndSignatureResultSize")},
    {25, US::stub("XUserGetTokenAndSignatureResult")},
    {33, US::method<&UserApi::RegisterForChangeEvent>("XUserRegisterForChangeEvent")},
    {34, US::method<&UserApi::UnregisterForChangeEvent>("XUserUnregisterForChangeEvent")},
});
constexpr auto kIUser =
    com::declare_interface<UserApi>("xgameruntime.XUser", kUserGuid, kUserTable.slots, {.slot_count = 48});

using UE = M<UserExtApi>;
constexpr com::MethodDesc kUserExtMethods[] = {UE::method<&UserExtApi::GetGamertag>("XUserGetGamertag")};
constexpr auto kIUserExt = com::declare_interface<UserExtApi>("xgameruntime.XUserExt", kUserExtIid,
                                                              kUserExtMethods, {.slot_count = 16});

using FE = M<FeatureApi>;
constexpr com::MethodDesc kFeatureMethods[] = {
    FE::method<&FeatureApi::IsFeatureAvailable>("XGameRuntimeIsFeatureAvailable")};
constexpr auto kIFeature = com::declare_interface<FeatureApi>("xgameruntime.XGameRuntimeFeature", kFeatureGuid,
                                                              kFeatureMethods, {.slot_count = 16});

using SY = M<SystemApi>;
constexpr com::MethodDesc kSystemMethods[] = {
    SY::method<&SystemApi::GetConsoleId>("XSystemGetConsoleId"),
    SY::method<&SystemApi::GetXboxLiveSandboxId>("XSystemGetXboxLiveSandboxId"),
    SY::stub("XSystemGetAppSpecificDeviceId"),
};
constexpr auto kISystem =
    com::declare_interface<SystemApi>("xgameruntime.XSystem", kSystemIid, kSystemMethods, {.slot_count = 24});

using GA = M<GameApi>;
constexpr com::MethodDesc kGameMethods[] = {
    GA::method<&GameApi::GetXboxTitleId>("XGameGetXboxTitleId"),
    GA::stub("XLaunchNewGame"),
};
constexpr auto kIGame =
    com::declare_interface<GameApi>("xgameruntime.XGame", kGameIid, kGameMethods, {.slot_count = 16});

using ST = M<StoreApi>;
constexpr auto kStoreTable = slot_table<67>({
    {3, ST::method<&StoreApi::CreateContext>("XStoreCreateContext")},
    {4, ST::method<&StoreApi::CloseContextHandle>("XStoreCloseContextHandle")},
    {66, ST::stub("XStoreRegisterGameLicenseChanged")},
    {67, ST::stub("XStoreUnregisterGameLicenseChanged")},
    {68, ST::stub("XStoreRegisterPackageLicenseLost")},
    {69, ST::stub("XStoreUnregisterPackageLicenseLost")},
});
constexpr auto kIStore =
    com::declare_interface<StoreApi>("xgameruntime.XStore", kStoreIid, kStoreTable.slots, {.slot_count = 96});

using NW = M<NetworkingApi>;
constexpr auto kNetworkingTable = slot_table<15>({
    {3, NW::stub("XNetworkingQueryPreferredLocalUdpMultiplayerPort")},
    {4, NW::stub("XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync")},
    {5, NW::stub("XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult")},
    {8, NW::stub("XNetworkingQuerySecurityInformationForUrlAsync")},
    {9, NW::stub("XNetworkingQuerySecurityInformationForUrlAsyncResultSize")},
    {10, NW::stub("XNetworkingQuerySecurityInformationForUrlAsyncResult")},
    {14, NW::stub("XNetworkingVerifyServerCertificate")},
    {15, NW::method<&NetworkingApi::GetConnectivityHint>("XNetworkingGetConnectivityHint")},
    {16, NW::method<&NetworkingApi::RegisterConnectivityHintChanged>("XNetworkingRegisterConnectivityHintChanged")},
    {17, NW::method<&NetworkingApi::UnregisterConnectivityHintChanged>(
             "XNetworkingUnregisterConnectivityHintChanged")},
});
constexpr auto kINetworking = com::declare_interface<NetworkingApi>("xgameruntime.XNetworking", kNetworkingIid,
                                                                    kNetworkingTable.slots, {.slot_count = 32});

using PK = M<PackageApi>;
constexpr auto kPackageTable = slot_table<37>({
    {3, PK::method<&PackageApi::GetCurrentProcessPackageIdentifier>("XPackageGetCurrentProcessPackageIdentifier")},
    {4, PK::method<&PackageApi::IsPackagedProcess>("XPackageIsPackagedProcess")},
    {5, PK::method<&PackageApi::CreateInstallationMonitor>("XPackageCreateInstallationMonitor")},
    {6, PK::method<&PackageApi::CloseInstallationMonitorHandle>("XPackageCloseInstallationMonitorHandle")},
    {7, PK::method<&PackageApi::GetInstallationProgress>("XPackageGetInstallationProgress")},
    {9, PK::method<&PackageApi::RegisterInstallationProgressChanged>(
            "XPackageRegisterInstallationProgressChanged")},
    {39, PK::method<&PackageApi::EnumeratePackages>("XPackageEnumeratePackages")},
});
constexpr auto kIPackage = com::declare_interface<PackageApi>("xgameruntime.XPackage", kPackageIid,
                                                              kPackageTable.slots, {.slot_count = 48});

constexpr auto kIGameSave = com::declare_interface<StubApi>("xgameruntime.XGameSave", kGameSaveIid, {},
                                                            {.slot_count = 40});
constexpr auto kIStoreExt = com::declare_interface<StubApi>("xgameruntime.XStoreExt", kStoreExtIid, {},
                                                            {.slot_count = 96});
using PL = M<PersistentLocalStorageApi>;
constexpr auto kPlsTable = slot_table<2>({
    {4, PL::method<&PersistentLocalStorageApi::GetPath>("XPersistentLocalStorageGetPath")},
});
constexpr auto kIPersistentLocalStorage = com::declare_interface<PersistentLocalStorageApi>(
    "xgameruntime.XPersistentLocalStorage", kPlsIid, kPlsTable.slots, {.slot_count = 16});
using SP = M<SpeechApi>;
constexpr auto kSpeechTable = slot_table<7>({
    {9, SP::method<&SpeechApi::Create>("XSpeechSynthesizerCreate")},
});
constexpr auto kISpeech = com::declare_interface<SpeechApi>("xgameruntime.XSpeechSynthesizer", kSpeechIid,
                                                            kSpeechTable.slots, {.slot_count = 32});
constexpr auto kIEvent = com::declare_interface<StubApi>("xgameruntime.XGameEvent", kEventGuid, {},
                                                         {.slot_count = 16});
constexpr auto kIDisplay = com::declare_interface<StubApi>("xgameruntime.{1b339674} (XDisplay?)", kDisplayGuid,
                                                           {}, {.slot_count = 16});
using HD = M<HdrModeApi>;
constexpr auto kUnmapped3Table = slot_table<2>({
    {4, HD::method<&HdrModeApi::TryEnableHdrMode>("XDisplayTryEnableHdrMode")},
});
constexpr auto kIUnmapped3 = com::declare_interface<HdrModeApi>("xgameruntime.{03f0fe74}", kUnmapped3Iid,
                                                                kUnmapped3Table.slots, {.slot_count = 16});
constexpr auto kIUnmapped9 = com::declare_interface<StubApi>("xgameruntime.{a4f1aee2}", kUnmapped9Iid, {},
                                                             {.slot_count = 16});
using AI = M<AnalyticsInfoApi>;
constexpr auto kUnmapped10Table = slot_table<1>({
    {3, AI::method<&AnalyticsInfoApi::GetAnalyticsInfo>("XSystemGetAnalyticsInfo")},
});
constexpr auto kIUnmapped10 = com::declare_interface<AnalyticsInfoApi>(
    "xgameruntime.{b884675d}", kUnmapped10Guid, kUnmapped10Table.slots, {.slot_count = 16});
constexpr auto kIUnmapped13 = com::declare_interface<StubApi>("xgameruntime.{dfcd4649}", kUnmapped13Iid, {},
                                                              {.slot_count = 32});

// ---------------------------------------------------------------------------
// API table and singletons
// ---------------------------------------------------------------------------

struct ApiEntry {
    Guid provider;
    Guid iid;
    const com::InterfaceInfo* iface;
    com::ComObject* (*create)();  // returns an object holding one reference, nullptr on failure
};

template <typename T, typename Host>
com::ComObject* make(const com::Interface<Host>& iface) {
    return com::create_object<T>(com::implements(iface)).release_ownership();
}

const ApiEntry kApis[] = {
    {kAsyncQueueGuid, kAsyncQueueGuid, &kIAsyncQueue.info, [] { return make<AsyncQueueApi>(kIAsyncQueue); }},
    {kUserGuid, kUserGuid, &kIUser.info, [] { return make<UserApi>(kIUser); }},
    {kUserGuid, kUserExtIid, &kIUserExt.info, [] { return make<UserExtApi>(kIUserExt); }},
    {kFeatureGuid, kFeatureGuid, &kIFeature.info, [] { return make<FeatureApi>(kIFeature); }},
    {kSystemProvider, kSystemIid, &kISystem.info, [] { return make<SystemApi>(kISystem); }},
    {kGameProvider, kGameIid, &kIGame.info, [] { return make<GameApi>(kIGame); }},
    {kStoreProvider, kStoreIid, &kIStore.info, [] { return make<StoreApi>(kIStore); }},
    {kNetworkingProvider, kNetworkingIid, &kINetworking.info, [] { return make<NetworkingApi>(kINetworking); }},
    {kPackageProvider, kPackageIid, &kIPackage.info, [] { return make<PackageApi>(kIPackage); }},
    {kGameSaveProvider, kGameSaveIid, &kIGameSave.info, [] { return make<StubApi>(kIGameSave); }},
    {kStoreProvider, kStoreExtIid, &kIStoreExt.info, [] { return make<StubApi>(kIStoreExt); }},
    {kPlsProvider, kPlsIid, &kIPersistentLocalStorage.info,
     [] { return make<PersistentLocalStorageApi>(kIPersistentLocalStorage); }},
    {kSpeechProvider, kSpeechIid, &kISpeech.info, [] { return make<SpeechApi>(kISpeech); }},
    {kEventGuid, kEventGuid, &kIEvent.info, [] { return make<StubApi>(kIEvent); }},
    {kDisplayGuid, kDisplayGuid, &kIDisplay.info, [] { return make<StubApi>(kIDisplay); }},
    {kUnmapped3Provider, kUnmapped3Iid, &kIUnmapped3.info, [] { return make<HdrModeApi>(kIUnmapped3); }},
    {kUnmapped9Provider, kUnmapped9Iid, &kIUnmapped9.info, [] { return make<StubApi>(kIUnmapped9); }},
    {kUnmapped10Guid, kUnmapped10Guid, &kIUnmapped10.info, [] { return make<AnalyticsInfoApi>(kIUnmapped10); }},
    {kUnmapped13Provider, kUnmapped13Iid, &kIUnmapped13.info, [] { return make<StubApi>(kIUnmapped13); }},
};
constexpr size_t kApiCount = sizeof(kApis) / sizeof(kApis[0]);

struct ProviderState {
    std::mutex mutex;
    com::ComObject* singletons[kApiCount] = {};
    std::map<std::pair<std::string, std::string>, std::pair<uint64_t, bool>> requests;
    uint64_t query_api = 0;
    uint64_t initialize_api = 0;
    uint64_t uninitialize_api = 0;
};

ProviderState& state() {
    static ProviderState instance;
    return instance;
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

int32_t QueryApiImpl(GuestPtr<Guid> provider_ptr, GuestPtr<Guid> iid_ptr, OutPtr out) {
    if (out.is_null()) {
        return E_POINTER_HR;
    }
    if (!out.write(0)) {
        com::detail::log_bad_pointer("QueryApiImpl out pointer", out.va());
        return E_POINTER_HR;
    }
    Guid provider{};
    Guid iid{};
    if (!provider_ptr.read(provider) || !iid_ptr.read(iid)) {
        com::detail::log_bad_pointer("QueryApiImpl GUID pointer", provider_ptr.is_null() ? provider_ptr.va() : iid_ptr.va());
        return E_POINTER_HR;
    }
    ProviderState& st = state();
    size_t index = kApiCount;
    for (size_t i = 0; i < kApiCount; ++i) {
        if (kApis[i].provider == provider && kApis[i].iid == iid) {
            index = i;
            break;
        }
    }
    com::ComObject* object = nullptr;
    {
        std::lock_guard<std::mutex> lock(st.mutex);
        auto& request = st.requests[{com::format_guid(provider), com::format_guid(iid)}];
        ++request.first;
        request.second = index != kApiCount;
        if (index != kApiCount) {
            if (st.singletons[index] == nullptr) {
                st.singletons[index] = kApis[index].create();
            }
            object = st.singletons[index];
        }
        if (index == kApiCount && request.first == 1) {
            com::detail::diagnostic(("xgameruntime: QueryApiImpl for unknown API provider " +
                                     com::format_guid(provider) + " interface " + com::format_guid(iid))
                                        .c_str());
        }
    }
    if (index == kApiCount) {
        return kUnknownApiResult;
    }
    if (object == nullptr) {
        return E_OUTOFMEMORY_HR;
    }
    uint64_t pointer = 0;
    const int32_t hr = object->query_interface(iid, &pointer);
    if (hr >= 0 && !out.write(pointer)) {
        object->release();
        return E_POINTER_HR;
    }
    return hr;
}

int32_t InitializeApiImpl(uint32_t /*version_major*/, uint32_t /*version_minor*/) {
    return pg6_gdk_xgame_runtime_initialize();
}

int32_t InitializeApiImplEx(uint64_t /*version*/, uint64_t /*build*/, uint32_t /*flags*/) {
    return pg6_gdk_xgame_runtime_initialize();
}

int32_t InitializeApiImplEx2(uint64_t /*version*/, uint64_t /*build*/, uint32_t /*flags*/, uint64_t /*options*/) {
    return pg6_gdk_xgame_runtime_initialize();
}

void UninitializeApiImpl() {
    pg6_gdk_xgame_runtime_uninitialize();
}

void XErrorReport(int32_t hr, uint64_t message) {
    std::string text;
    for (uint64_t i = 0; message != 0 && i < 512; ++i) {
        char c = 0;
        if (!com::read_guest(message + i, &c, 1) || c == 0) {
            break;
        }
        text.push_back(c);
    }
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "xgameruntime: XErrorReport hr=0x%08X ", static_cast<uint32_t>(hr));
    com::detail::diagnostic((prefix + text).c_str());
}

struct ExportSpec {
    const char* name;
    playgta6::guest::HostThunkFn thunk;
    uint64_t ProviderState::*slot;
};

const ExportSpec kExports[] = {
    {"QueryApiImpl", com::function_thunk<&QueryApiImpl>(), &ProviderState::query_api},
    {"InitializeApiImpl", com::function_thunk<&InitializeApiImpl>(), &ProviderState::initialize_api},
    {"InitializeApiImplEx", com::function_thunk<&InitializeApiImplEx>(), nullptr},
    {"InitializeApiImplEx2", com::function_thunk<&InitializeApiImplEx2>(), nullptr},
    {"UninitializeApiImpl", com::function_thunk<&UninitializeApiImpl>(), &ProviderState::uninitialize_api},
    {"XErrorReport", com::function_thunk<&XErrorReport>(), nullptr},
};

}  // namespace

uint64_t register_xgameruntime_provider() {
    namespace win32 = playgta6::os::win32;
    ProviderState& st = state();
    std::lock_guard<std::mutex> lock(st.mutex);
    const uint64_t module = win32::register_virtual_module("xgameruntime.dll", /*pinned=*/true);
    if (module == 0) {
        return 0;
    }
    for (const ExportSpec& e : kExports) {
        const uint64_t proc = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(e.thunk));
        // Re-registers the guest callable after a reset_guest_runtime() (tests); otherwise no-op.
        uint64_t callable = playgta6::guest::guest_callable_for_proc(proc);
        if (callable == 0) {
            callable = playgta6::guest::register_host_callable(e.thunk, e.name, proc);
        }
        if (callable == 0 || !win32::register_export("xgameruntime.dll", e.name, 0, proc)) {
            return 0;
        }
        if (e.slot != nullptr) {
            st.*(e.slot) = callable;
        }
    }
    return module;
}

uint64_t query_api_address() {
    std::lock_guard<std::mutex> lock(state().mutex);
    return state().query_api;
}

uint64_t initialize_api_address() {
    std::lock_guard<std::mutex> lock(state().mutex);
    return state().initialize_api;
}

uint64_t uninitialize_api_address() {
    std::lock_guard<std::mutex> lock(state().mutex);
    return state().uninitialize_api;
}

std::vector<ApiRequest> api_requests() {
    ProviderState& st = state();
    std::lock_guard<std::mutex> lock(st.mutex);
    std::vector<ApiRequest> out;
    for (const auto& [key, value] : st.requests) {
        out.push_back(ApiRequest{key.first, key.second, value.first, value.second});
    }
    return out;
}

void set_xbox_title_id(uint32_t title_id) {
    g_title_id.store(title_id, std::memory_order_relaxed);
}

void reset_xgameruntime_provider() {
    ProviderState& st = state();
    std::lock_guard<std::mutex> lock(st.mutex);
    for (auto& object : st.singletons) {
        object = nullptr;  // owned by the COM bridge registry, dropped by reset_com_bridge()
    }
    st.requests.clear();
    st.query_api = st.initialize_api = st.uninitialize_api = 0;
    queues().clear();
    users().clear();
    stores().clear();
    ports().clear();
    g_title_id.store(0, std::memory_order_relaxed);
}

}  // namespace playgta6::os::gdk::provider
