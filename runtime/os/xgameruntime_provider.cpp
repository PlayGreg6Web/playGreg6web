#include "os/guest_window.h"
#include "os/xgameruntime_provider.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "guest/guest_callbacks.h"
#include "guest/guest_runtime.h"
#include "os/os_shim.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/xasync_provider.h"
#include "os/xbox_offline.h"
#include "os/xgameruntime.h"
#include "os/xgamesave_local.h"

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
//     ForChangeEvent h; s9 GetLocalId(2) h: the slots follow the public header order (Duplicate,
//     Close, Compare, GetMaxUsers, AddAsync, AddResult, GetLocalId, FindUserByLocalId, GetId,
//     FindUserById, ...); the executable keeps {handle, s11 XUID, s9 8-byte value} per user and
//     rebuilds that record from the value through s10 (value, &handle), i.e. FindUserByLocalId.
//     s3 DuplicateHandle(2) h. Stubs: s5 Compare, s10 FindUserByLocalId, s12 FindUserById,
//     s25 GetTokenAndSignatureResult. s13 GetIsGuest(2) h: the executable passes a one-byte
//     local and stores it as a flag (header order: FindUserById, GetIsGuest, ...); the local
//     offline user is not a guest. s19 GetAgeGroup(2) h (a 4-byte result the caller keeps only
//     when 1..3): the configured age group (--user-age-group, default Unknown); s14 is
//     unidentified.
//   {01acd177-...}/{cef4fac0-...} XUser extension: s3 XUserGetGamertag(5) m.
//   {8836fe87-...} s3 XGameRuntimeIsFeatureAvailable(1)->bool h.
//   {e349bd1a-...}/{1861cf2e-...} XSystem: s3 GetConsoleId(3) h, s4 GetXboxLiveSandboxId(3) h
//     (the configured console sandbox; none: E_NOTIMPL), s5 GetAppSpecificDeviceId stub. XSystemGetDeviceType /
//     GetAnalyticsInfo are never called.
//   {973a344e-...}/{2549f142-...} XGame: s3 XGameGetXboxTitleId(1) h, s4 XLaunchNewGame(3) h (self: relaunch
//     via kRelaunchSelfExitCode; another title: E_NOTIMPL).
//   {0dd112ac-...}/{b09d803c-...} XStore: s3 CreateContext h, s4 CloseContextHandle h; the
//     other 40 used slots stay stubs (O8 offline policy: nothing about licences is invented).
//   {0dd112ac-...}/{2d42fea5-...} XStore extension: s71 AcquireLicenseForDurablesAsync, s72
//     AcquireLicenseForDurablesResult (O37: truthful synchronous E_NOTIMPL, claiming zero
//     ownership/entitlement; all other slots stay trapping stubs).
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
//   {3e241977-...}/{d722b373-...} XSpeechSynthesizer (silent synthesizer: fixed handle, empty WAVE
//     streams). The executable's wrappers use s8 EnumerateInstalledVoices(2) h, s9 Create(1) h,
//     s10 CloseHandle(1), s11 SetDefaultVoice(1), s12 SetCustomVoice(2), s13 CreateStreamFromText(3),
//     s14 CloseStreamHandle(1), s15 GetStreamDataSize(2), s16 GetStreamData(4) (public header
//     order from s8), and s21 a second 3-argument stream creator: CreateStreamFromSsml.
//   {704c3f58-...}/{1bfff3af-...} XGameSave: s3 InitializeProvider(4) h, s4 InitializeProviderAsync(4) h,
//     s5 InitializeProviderResult, s6 CloseProvider (O34: a local provider on the host directory
//     --save-dir, never any cloud sync; s5/s6 by public header order), s13
//     EnumerateContainerInfoByName(provider, prefix, context, callback) and s14
//     EnumerateContainerInfo(provider, context, callback) (the title calls s13 with "Player" and
//     "SettingsProfileSettings", s14 once; the callback is a code address and compares the name).
//     R40, from the title's save manager (container handle at +0xc0, update handle at +0xc8,
//     provider at +0x88): s16 CreateContainer(provider, name, handle*) h (also called before every
//     read and write), s17 CloseContainer(handle) h (the manager's reset), s19
//     EnumerateBlobInfoByName(container, "Save Game Header", context, callback) h (the callback
//     sizes the read buffer as 24 + strlen(name) + 1 + size, i.e. XGameSaveBlob is {name, size,
//     data}), s20 ReadBlobData(container, names, &count, size, blobs) h, s21/s22
//     ReadBlobDataAsync(container, names, u32 count, block) / Result(block, size, blobs, &count) h,
//     s23 CreateUpdate(container, displayName, handle*) h, s24 CloseUpdate h, s25
//     SubmitBlobWrite(update, name, data, size) h (header 0x128 bytes + payload), s28/s29
//     SubmitUpdateAsync(update, block) / Result(block) h, s8/s9 GetRemainingQuotaAsync/Result h,
//     s11/s12 DeleteContainerAsync/Result h; s7, s10, s18, s26, s27 (no title wrappers) by the
//     same order m. s15 (no wrapper; GetContainerInfo; the public header has no such function, so
//     its signature is unknown) answers E_NOTIMPL without touching any argument.
//   R40 non-trapping stub-only interfaces (every slot logs, then answers one documented HRESULT):
//     events {bbfbdcc7-...} S_OK, shell {1b339674-...} and capture {a4f1aee2-...}
//     HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED), XGameUi {dfcd4649-...} E_ABORT.
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
// Configured age group of the local user (set_user_age_group).
std::atomic<uint32_t> g_user_age_group{static_cast<uint32_t>(UserAgeGroup::Unknown)};

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

    // s3 XUserDuplicateHandle(user, XUserHandle* duplicated): a new handle to the same local user
    // (closed independently with s4).
    int32_t DuplicateHandle(uint64_t user, OutPtr duplicated) {
        if (duplicated.is_null() || !duplicated.write(0) || !check_handle(users(), user, "XUserHandle")) {
            return E_INVALIDARG_HR;
        }
        Pg6XUserHandle copy = nullptr;
        const int32_t hr = gdk::XUserDuplicateHandle(from_guest<Pg6XUser>(user), &copy);
        if (hr < 0 || copy == nullptr) {
            return hr < 0 ? hr : E_INVALIDARG_HR;
        }
        users().add(copy);
        duplicated.write(to_guest(copy));
        return S_OK_HR;
    }

    // s9 XUserGetLocalId(user, XUserLocalId* localId): the local (per-console) identifier of the
    // runtime's one local user: stable for the user's lifetime, non-zero, distinct from the XUID.
    // It claims nothing about sign-in, privileges or ownership.
    int32_t GetLocalId(uint64_t user, GuestPtr<uint64_t> local_id) {
        if (!check_handle(users(), user, "XUserHandle") || local_id.is_null()) {
            return E_INVALIDARG_HR;
        }
        local_id.write(kOfflineUserLocalId);
        return S_OK_HR;
    }

    // s13 XUserGetIsGuest(user, bool* isGuest): the runtime's one local user is a real local
    // account, not a guest. Claims nothing about sign-in, privileges or ownership.
    int32_t GetIsGuest(uint64_t user, GuestPtr<uint8_t> is_guest) {
        if (!check_handle(users(), user, "XUserHandle") || is_guest.is_null()) {
            return E_INVALIDARG_HR;
        }
        is_guest.write(0);
        return S_OK_HR;
    }

    // s19 XUserGetAgeGroup(user, XUserAgeGroup* ageGroup): the configured age group of the local
    // user (set_user_age_group; default Unknown, the truthful answer without account data).
    int32_t GetAgeGroup(uint64_t user, GuestPtr<uint32_t> age_group) {
        if (!check_handle(users(), user, "XUserHandle") || age_group.is_null()) {
            return E_INVALIDARG_HR;
        }
        age_group.write(g_user_age_group.load(std::memory_order_relaxed));
        return S_OK_HR;
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
        os::guest_window::mark_written(buffer, static_cast<size_t>(size));  // O37
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
std::mutex& sandbox_mutex() {
    static std::mutex m;
    return m;
}
std::string& sandbox_id_storage() {
    static std::string s;
    return s;
}
std::string configured_sandbox_id() {
    std::lock_guard<std::mutex> lock(sandbox_mutex());
    return sandbox_id_storage();
}


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

    // s4 XSystemGetXboxLiveSandboxId(size, buffer, used): the emulated console's sandbox
    // (set_xbox_live_sandbox_id; console environment configuration, not a sign-in result). With
    // none configured: E_NOTIMPL, an empty string and used = 0. Otherwise used = length + 1 and
    // the terminated string, or E_NOT_SUFFICIENT_BUFFER when it does not fit.
    int32_t GetXboxLiveSandboxId(uint64_t size, uint64_t buffer, GuestPtr<uint64_t> used) {
        const std::string sandbox = configured_sandbox_id();
        if (sandbox.empty()) {
            if (!used.is_null()) {
                used.write(0);
            }
            if (buffer != 0 && size != 0) {
                const char nul = '\0';
                com::write_guest(buffer, &nul, 1);
            }
            if (!sandbox_logged_.exchange(true)) {
                os::log("xgameruntime: XSystemGetXboxLiveSandboxId: no sandbox configured; returning E_NOTIMPL");
            }
            return com::E_NOTIMPL_HR;
        }
        const uint64_t needed = sandbox.size() + 1;
        if (!used.is_null()) {
            used.write(needed);
        }
        if (buffer == 0) {
            return E_INVALIDARG_HR;
        }
        if (size < needed) {
            return E_NOT_SUFFICIENT_BUFFER_HR;
        }
        if (!sandbox_logged_.exchange(true)) {
            os::log("xgameruntime: XSystemGetXboxLiveSandboxId: configured sandbox \"" + sandbox + "\"");
        }
        return com::write_guest(buffer, sandbox.c_str(), static_cast<size_t>(needed)) ? S_OK_HR : E_INVALIDARG_HR;
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

    // XLaunchNewGame(exePath, args, defaultUser). The running title relaunching itself is a restart,
    // not a store or licence call: the guest stops with kRelaunchSelfExitCode through the
    // TerminateProcess path and the harness / browser boot start it again with the same arguments.
    // Like the real call it does not return on success (only without an exit handler, i.e. tests).
    // Launching another title stays E_NOTIMPL.
    int32_t LaunchNewGame(const char* exe_path, const char* args, uint64_t /*default_user*/) {
        if (exe_path == nullptr || exe_path[0] == '\0') {
            return E_INVALIDARG_HR;
        }
        const char* command_line = pg6_win32_get_command_line_a();
        if (!is_self_launch_target(exe_path, command_line != nullptr ? command_line : "")) {
            os::log(std::string("xgameruntime: XLaunchNewGame(\"") + exe_path +
                    "\"): another title; returning E_NOTIMPL");
            return com::E_NOTIMPL_HR;
        }
        os::log(std::string("xgameruntime: XLaunchNewGame(\"") + exe_path + "\", args \"" +
                (args != nullptr ? args : "") + "\"): relaunch of the running title requested");
        pg6_win32_terminate_process(win32::kCurrentProcessPseudoHandle, win32::kRelaunchSelfExitCode);
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

// {0dd112ac-...}/{2d42fea5-...}: XStore extension (O37: offline-truthful synchronous failure on
// durable license acquisition; claims zero ownership/entitlement and never invents a license).
class StoreExtApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XStore extension";

    // s71: XStoreAcquireLicenseForDurablesAsync(XStoreContextHandle, const char* storeId,
    // XAsyncBlock*): synchronous E_NOTIMPL offline failure (never allocates a license handle or
    // queues fake async completion).
    int32_t AcquireLicenseForDurablesAsync(uint64_t context, GuestPtr<const char> store_id,
                                           GuestPtr<Pg6XAsyncBlock> async_block) {
        if (!check_handle(stores(), context, "XStoreContextHandle") || store_id.is_null() ||
            async_block.is_null()) {
            return E_INVALIDARG_HR;
        }
        if (!acquire_logged_.exchange(true)) {
            os::log("xgameruntime: XStoreAcquireLicenseForDurablesAsync: offline mode (no store/entitlement); "
                    "returning E_NOTIMPL");
        }
        return com::E_NOTIMPL_HR;
    }

    // s72: XStoreAcquireLicenseForDurablesResult(XAsyncBlock*, XStoreLicenseHandle*): clears the
    // out handle if non-null and returns E_NOTIMPL.
    int32_t AcquireLicenseForDurablesResult(GuestPtr<Pg6XAsyncBlock> /*async_block*/, OutPtr license_handle) {
        if (!license_handle.is_null()) {
            license_handle.write(0);
        }
        if (!result_logged_.exchange(true)) {
            os::log("xgameruntime: XStoreAcquireLicenseForDurablesResult: offline mode (no store/entitlement); "
                    "returning E_NOTIMPL");
        }
        return com::E_NOTIMPL_HR;
    }

private:
    std::atomic<bool> acquire_logged_{false};
    std::atomic<bool> result_logged_{false};
};

// The guest callback of one connectivity registration. The O8 shim calls it at most once (the
// initial notification; the offline state never changes), so it frees itself when called. A
// registration removed before its notification was dispatched leaks this 16-byte record.
struct GuestConnectivityCallback {
    uint64_t function = 0;
    uint64_t context = 0;
};

// Guest-readable copy of the hint handed to guest callbacks (the offline hint is constant).
uint64_t guest_connectivity_hint() {
    static const uint64_t va = com::guest_alloc(sizeof(Pg6XNetworkingConnectivityHint));
    return va;
}

void forward_connectivity_change(void* context, const Pg6XNetworkingConnectivityHint* hint) {
    std::unique_ptr<GuestConnectivityCallback> cb(static_cast<GuestConnectivityCallback*>(context));
    const uint64_t va = guest_connectivity_hint();
    if (va == 0 || hint == nullptr || !com::write_guest(va, *hint)) {
        com::detail::log_bad_pointer("XNetworkingConnectivityHint (guest copy)", va);
        return;
    }
    // XNetworkingConnectivityHintChangedCallback(void* context, const XNetworkingConnectivityHint*).
    com::GuestFunction<void(uint64_t, uint64_t)>(cb->function).call_top_level(cb->context, va);
}

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

    // The O8 shim sends the documented initial notification (offline, network initialized) to
    // the queue's completion port; it is forwarded to the guest callback.
    int32_t RegisterConnectivityHintChanged(uint64_t queue, uint64_t context, uint64_t callback,
                                            GuestPtr<uint64_t> token) {
        if (callback == 0 || token.is_null() || !optional_queue_ok(queue)) {
            return E_INVALIDARG_HR;
        }
        auto cb = std::make_unique<GuestConnectivityCallback>(GuestConnectivityCallback{callback, context});
        Pg6XTaskQueueRegistrationToken host_token{};
        const int32_t hr = pg6_gdk_xnetworking_register_connectivity_hint_changed(
            from_guest<Pg6XTaskQueue>(queue), cb.get(), &forward_connectivity_change, &host_token);
        if (hr >= 0) {
            cb.release();  // owned by the queued notification now
        }
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

// XSpeechSynthesizer: a truthful silent synthesizer. Create returns a fixed handle, every stream
// (from text or SSML) is a valid 16 kHz mono 16-bit PCM WAVE with no samples, size / get calls
// report that stream, closing is free, and voice enumeration callbacks never run (no voices are
// installed). Slot map: see the table at kSpeechTable (s8..s16 in public header order, s21 SSML).
class SpeechApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XSpeechSynthesizer";
    static constexpr uint64_t kSynthesizerHandle = 0x5350454543480001ull;
    static constexpr uint64_t kStreamHandle = 0x5350454543530001ull;
    static constexpr uint64_t kWaveHeaderSize = 44;

    static void BuildSilentWave(uint8_t (&h)[44]) {
        auto put32 = [&](size_t o, uint32_t v) { for (int i = 0; i < 4; ++i) h[o + i] = uint8_t(v >> (8 * i)); };
        auto put16 = [&](size_t o, uint16_t v) { h[o] = uint8_t(v); h[o + 1] = uint8_t(v >> 8); };
        std::memcpy(h, "RIFF", 4);
        put32(4, 36);  // no samples
        std::memcpy(h + 8, "WAVEfmt ", 8);
        put32(16, 16);
        put16(20, 1);       // PCM
        put16(22, 1);       // mono
        put32(24, 16000);   // sample rate
        put32(28, 32000);   // byte rate
        put16(32, 2);       // block align
        put16(34, 16);      // bits per sample
        std::memcpy(h + 36, "data", 4);
        put32(40, 0);
    }

    // s8: XSpeechSynthesizerEnumerateInstalledVoices(void* context, callback*).
    // No voices are installed, so the callback never runs.
    int32_t EnumerateInstalledVoices(uint64_t /*context*/, uint64_t callback) {
        if (callback == 0) {
            return E_INVALIDARG_HR;
        }
        return S_OK_HR;
    }

    // s9: XSpeechSynthesizerCreate(XSpeechSynthesizerHandle*).
    int32_t Create(GuestPtr<uint64_t> handle) {
        if (handle.is_null()) {
            return E_INVALIDARG_HR;
        }
        handle.write(kSynthesizerHandle);
        return S_OK_HR;
    }

    // s10: close (free).
    int32_t Close(uint64_t /*handle*/) { return S_OK_HR; }

    // s11: SetDefaultVoice(handle): the default voice is the only one.
    int32_t SetDefaultVoice(uint64_t /*handle*/) { return S_OK_HR; }

    // s12: SetCustomVoice(handle, voiceId): no voices are installed, so no id names one.
    int32_t SetCustomVoice(uint64_t /*handle*/, uint64_t /*voice_id*/) { return E_INVALIDARG_HR; }

    // s13 / s21: CreateStreamFromText / FromSsml(handle, text, streamHandle*).
    int32_t CreateStream(uint64_t /*handle*/, uint64_t /*text*/, GuestPtr<uint64_t> stream) {
        if (stream.is_null()) {
            return E_INVALIDARG_HR;
        }
        stream.write(kStreamHandle);
        return S_OK_HR;
    }

    // s14: close stream (free).
    int32_t CloseStream(uint64_t /*stream*/) { return S_OK_HR; }

    // s15: GetStreamDataSize(stream, uint64_t* size).
    int32_t GetStreamSize(uint64_t /*stream*/, GuestPtr<uint64_t> size) {
        if (size.is_null()) {
            return E_INVALIDARG_HR;
        }
        size.write(kWaveHeaderSize);
        return S_OK_HR;
    }

    // s16: GetStreamData(stream, bufferSize, buffer, uint64_t* used).
    int32_t GetStreamData(uint64_t /*stream*/, uint64_t buffer_size, uint64_t buffer, GuestPtr<uint64_t> used) {
        if (buffer == 0) {
            return E_INVALIDARG_HR;
        }
        if (buffer_size < kWaveHeaderSize) {
            return E_NOT_SUFFICIENT_BUFFER_HR;
        }
        uint8_t header[44];
        BuildSilentWave(header);
        if (!com::write_guest(buffer, header, sizeof(header))) {
            return E_INVALIDARG_HR;
        }
        if (!used.is_null()) {
            used.write(kWaveHeaderSize);
        }
        return S_OK_HR;
    }
};

// XGameSave (public header order: s3 XGameSaveInitializeProvider, s4 XGameSaveInitializeProviderAsync,
// s5 XGameSaveInitializeProviderResult, s6 XGameSaveCloseProvider). The executable calls s4
// through a wrapper (user, configuration id string, bool syncOnDemand, 0x38-byte block =
// XAsyncBlock). O34: the provider is local, like a console's offline save store
// (os/xgamesave_local.h, host directory --save-dir). That is a truthful offline answer, not a
// cloud or entitlement claim: nothing is ever synchronised.
struct SaveProvider {
    gamesave::Store store;
    std::string configuration_id;
};

// XGameSaveContainerHandle: a name in a provider. Like the GDK, creating the handle does not create
// the container in storage; the first submitted update does.
struct SaveContainer {
    std::shared_ptr<SaveProvider> provider;
    std::string name;
};

// XGameSaveUpdateHandle: blob writes and deletes collected until one submit applies them all.
struct SaveUpdate {
    std::shared_ptr<SaveProvider> provider;
    std::string container;
    std::string display_name;
    std::mutex mutex;
    std::vector<gamesave::BlobWrite> writes;
    std::vector<std::string> deletes;
    bool submitted = false;
};

// Opaque guest handle values -> host objects (one table per handle type, distinct ranges).
template <typename T>
class SaveHandles {
public:
    explicit constexpr SaveHandles(uint64_t first) : first_(first) {}
    uint64_t add(std::shared_ptr<T> object) {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t id = first_ + next_++;
        objects_.emplace(id, std::move(object));
        return id;
    }
    std::shared_ptr<T> find(uint64_t id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = objects_.find(id);
        return it == objects_.end() ? nullptr : it->second;
    }
    bool remove(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return objects_.erase(id) != 0;
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        objects_.clear();
    }

private:
    uint64_t first_;
    mutable std::mutex mutex_;
    std::map<uint64_t, std::shared_ptr<T>> objects_;
    uint64_t next_ = 1;
};

SaveHandles<SaveProvider>& save_providers() {
    static SaveHandles<SaveProvider> h(0x5047530000000ULL);
    return h;
}
SaveHandles<SaveContainer>& save_containers() {
    static SaveHandles<SaveContainer> h(0x5047430000000ULL);
    return h;
}
SaveHandles<SaveUpdate>& save_updates() {
    static SaveHandles<SaveUpdate> h(0x5047550000000ULL);
    return h;
}

uint64_t add_save_provider(const std::string& configuration_id) {
    return save_providers().add(std::make_shared<SaveProvider>(
        SaveProvider{gamesave::open_store(configuration_id), configuration_id}));
}

// Blobs read for XGameSaveReadBlobData(Async).
struct ReadBlobs {
    std::vector<std::pair<std::string, std::vector<uint8_t>>> blobs;
    // Each XGameSaveBlob entry plus its name and data, packed (what the title sizes its buffer to:
    // sizeof(XGameSaveBlob) + strlen(name) + 1 + size).
    size_t required() const;
};

// XGameSaveBlobInfo / XGameSaveBlob (x64).
struct GuestBlobInfo {
    uint64_t name;  // const char*
    uint32_t size;
    uint32_t pad;
};
struct GuestBlob {
    GuestBlobInfo info;
    uint64_t data;  // uint8_t*
};
static_assert(sizeof(GuestBlobInfo) == 16 && sizeof(GuestBlob) == 24);

size_t ReadBlobs::required() const {
    size_t total = 0;
    for (const auto& blob : blobs) {
        total += sizeof(GuestBlob) + blob.first.size() + 1 + blob.second.size();
    }
    return total;
}

// Reads the named blobs (all blobs when `names` is empty) of a container. A container that was
// never submitted has no blobs.
HResult read_blobs(const SaveContainer& container, std::vector<std::string> names, ReadBlobs* out) {
    out->blobs.clear();
    if (names.empty()) {
        std::vector<gamesave::BlobInfo> infos;
        const HResult hr = container.provider->store.enumerate_blobs(container.name, &infos);
        if (hr < 0 && hr != gamesave::E_GS_NO_SUCH_CONTAINER) {
            return hr;
        }
        for (const gamesave::BlobInfo& info : infos) {
            names.push_back(info.name);
        }
    }
    for (const std::string& name : names) {
        std::vector<uint8_t> data;
        HResult hr = container.provider->store.read_blob(container.name, name, &data);
        if (hr == gamesave::E_GS_NO_SUCH_CONTAINER) {
            hr = gamesave::E_GS_BLOB_NOT_FOUND;
        }
        if (hr < 0) {
            out->blobs.clear();
            return hr;
        }
        out->blobs.emplace_back(name, std::move(data));
    }
    return S_OK_HR;
}

// Lays out the XGameSaveBlob array at `buffer`, followed by each blob's data and name.
bool write_blobs(uint64_t buffer, const ReadBlobs& read) {
    uint64_t cursor = buffer + read.blobs.size() * sizeof(GuestBlob);
    for (size_t i = 0; i < read.blobs.size(); ++i) {
        const auto& [name, data] = read.blobs[i];
        GuestBlob entry{};
        entry.data = cursor;
        entry.info.size = static_cast<uint32_t>(data.size());
        if (!data.empty() && !com::write_guest(cursor, data.data(), data.size())) {
            return false;
        }
        cursor += data.size();
        entry.info.name = cursor;
        if (!com::write_guest(cursor, name.c_str(), name.size() + 1)) {
            return false;
        }
        cursor += name.size() + 1;
        if (!com::write_guest(buffer + i * sizeof(GuestBlob), entry)) {
            return false;
        }
    }
    return true;
}

// Applies an update: the container is created by its first update; a failing update leaves no new
// container behind.
HResult submit_update(SaveUpdate& update) {
    std::vector<gamesave::BlobWrite> writes;
    std::vector<std::string> deletes;
    {
        std::lock_guard<std::mutex> lock(update.mutex);
        if (update.submitted) {
            return E_ILLEGAL_METHOD_CALL_HR;
        }
        update.submitted = true;
        writes = std::move(update.writes);
        deletes = std::move(update.deletes);
    }
    gamesave::Store& store = update.provider->store;
    std::vector<gamesave::BlobInfo> existing;
    const bool is_new = store.enumerate_blobs(update.container, &existing) == gamesave::E_GS_NO_SUCH_CONTAINER;
    HResult hr = store.create_container(update.container);
    if (hr >= 0) {
        hr = store.submit_update(update.container, writes, deletes);
    }
    if (hr >= 0 && !update.display_name.empty()) {
        hr = store.set_display_name(update.container, update.display_name);
    }
    if (hr < 0 && is_new) {
        store.delete_container(update.container);
    }
    return hr;
}

// Guest blob-name array (const char* names[count]).
bool guest_names(uint64_t names, uint32_t count, std::vector<std::string>* out) {
    out->clear();
    if (count > 4096 || (count != 0 && names == 0)) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t ptr = 0;
        if (!com::read_guest(names + 8ull * i, ptr) || ptr == 0) {
            return false;
        }
        out->push_back(guest_string(ptr, 256));
    }
    return true;
}

// XGameSaveInitializeProviderAsync: the provider is created in DoWork and handed over by Result.
class SaveInitOp final : public XAsyncProviderImpl {
public:
    explicit SaveInitOp(std::string configuration_id) : configuration_id_(std::move(configuration_id)) {}
    HResult call(XAsyncOp op, XAsyncBlock* block, size_t /*buffer_size*/, uint64_t buffer) override {
        switch (op) {
        case XAsyncOp::Begin:
            return XAsyncSchedule(block, 0);
        case XAsyncOp::DoWork:
            handle_ = add_save_provider(configuration_id_);
            XAsyncComplete(block, S_OK_HR, sizeof(uint64_t));
            return E_PENDING_HR;
        case XAsyncOp::GetResult:
            return com::write_guest(buffer, handle_) ? S_OK_HR : E_INVALIDARG_HR;
        default:
            return S_OK_HR;
        }
    }

private:
    std::string configuration_id_;
    uint64_t handle_ = 0;
};

// An XGameSave async operation whose work runs on the queue's work port: `work` returns the
// HRESULT and the result size, `result` writes the result into the guest buffer.
class SaveWorkOp final : public XAsyncProviderImpl {
public:
    using Work = std::function<HResult(size_t* required)>;
    using Result = std::function<HResult(uint64_t buffer)>;
    SaveWorkOp(Work work, Result result) : work_(std::move(work)), result_(std::move(result)) {}
    HResult call(XAsyncOp op, XAsyncBlock* block, size_t /*buffer_size*/, uint64_t buffer) override {
        switch (op) {
        case XAsyncOp::Begin:
            return XAsyncSchedule(block, 0);
        case XAsyncOp::DoWork: {
            size_t required = 0;
            const HResult hr = work_(&required);
            XAsyncComplete(block, hr, hr < 0 ? 0 : required);
            return E_PENDING_HR;
        }
        case XAsyncOp::GetResult:
            return result_ ? result_(buffer) : S_OK_HR;
        default:
            return S_OK_HR;
        }
    }

private:
    Work work_;
    Result result_;
};

// ReadBlobDataAsync results by guest XAsyncBlock until ReadBlobDataResult hands them over (the
// blob count is an output of Result, next to the buffer the framework fills).
struct PendingRead {
    std::mutex mutex;
    ReadBlobs read;
};
std::mutex g_pending_reads_mutex;
std::map<uint64_t, std::shared_ptr<PendingRead>> g_pending_reads;

class GameSaveApi : public com::ComObject {
public:
    static constexpr const char* kComClassName = "xgameruntime XGameSave";

    // s3: XGameSaveInitializeProvider(user, configurationId, syncOnDemand, XGameSaveProviderHandle*).
    int32_t InitializeProvider(uint64_t user, uint64_t configuration_id, uint64_t /*sync_on_demand*/,
                               GuestPtr<uint64_t> provider) {
        if (provider.is_null()) {
            return E_POINTER_HR;
        }
        provider.write(0);
        if (user == 0 || configuration_id == 0) {
            return E_INVALIDARG_HR;
        }
        log_once("XGameSaveInitializeProvider");
        provider.write(add_save_provider(guest_string(configuration_id, 256)));
        return S_OK_HR;
    }

    // s4: XGameSaveInitializeProviderAsync(user, configurationId, syncOnDemand, XAsyncBlock*).
    int32_t InitializeProviderAsync(uint64_t user, uint64_t configuration_id, uint64_t /*sync_on_demand*/,
                                    uint64_t async) {
        if (async == 0 || user == 0 || configuration_id == 0) {
            return E_INVALIDARG_HR;
        }
        log_once("XGameSaveInitializeProviderAsync");
        Pg6XAsyncBlock* shadow = begin_async(async);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        return XAsyncBeginWith(shadow, std::make_unique<SaveInitOp>(guest_string(configuration_id, 256)), 0,
                               "XGameSaveInitializeProviderAsync");
    }

    // s5: XGameSaveInitializeProviderResult(XAsyncBlock*, XGameSaveProviderHandle*).
    int32_t InitializeProviderResult(uint64_t async, uint64_t provider) {
        if (provider == 0) {
            return E_POINTER_HR;
        }
        com::write_guest(provider, uint64_t{0});
        return finish_async(async, sizeof(uint64_t), provider);
    }

    // s7: XGameSaveGetRemainingQuota(provider, int64_t* remainingQuota).
    int32_t GetRemainingQuota(uint64_t provider_handle, uint64_t remaining) {
        auto provider = save_providers().find(provider_handle);
        if (!provider || remaining == 0) {
            return E_INVALIDARG_HR;
        }
        int64_t value = 0;
        const int32_t hr = provider->store.remaining_quota(&value);
        com::write_guest(remaining, hr < 0 ? int64_t{0} : value);
        return hr;
    }

    // s8: XGameSaveGetRemainingQuotaAsync(provider, XAsyncBlock*).
    int32_t GetRemainingQuotaAsync(uint64_t provider_handle, uint64_t async) {
        auto provider = save_providers().find(provider_handle);
        if (!provider || async == 0) {
            return E_INVALIDARG_HR;
        }
        auto value = std::make_shared<int64_t>(0);
        return begin_work(async, "XGameSaveGetRemainingQuotaAsync",
                          [provider, value](size_t* required) {
                              *required = sizeof(int64_t);
                              return provider->store.remaining_quota(value.get());
                          },
                          [value](uint64_t buffer) {
                              return com::write_guest(buffer, *value) ? S_OK_HR : E_INVALIDARG_HR;
                          });
    }

    // s9: XGameSaveGetRemainingQuotaResult(XAsyncBlock*, int64_t* remainingQuota).
    int32_t GetRemainingQuotaResult(uint64_t async, uint64_t remaining) {
        if (remaining == 0) {
            return E_POINTER_HR;
        }
        com::write_guest(remaining, int64_t{0});
        return finish_async(async, sizeof(int64_t), remaining);
    }

    // s10: XGameSaveDeleteContainer(provider, containerName).
    int32_t DeleteContainer(uint64_t provider_handle, uint64_t name) {
        auto provider = save_providers().find(provider_handle);
        if (!provider || name == 0) {
            return E_INVALIDARG_HR;
        }
        return provider->store.delete_container(guest_string(name, 512));
    }

    // s11: XGameSaveDeleteContainerAsync(provider, containerName, XAsyncBlock*).
    int32_t DeleteContainerAsync(uint64_t provider_handle, uint64_t name, uint64_t async) {
        auto provider = save_providers().find(provider_handle);
        if (!provider || name == 0 || async == 0) {
            return E_INVALIDARG_HR;
        }
        const std::string container = guest_string(name, 512);
        if (!gamesave::valid_container_name(container)) {
            return gamesave::E_GS_INVALID_CONTAINER_NAME;
        }
        return begin_work(async, "XGameSaveDeleteContainerAsync",
                          [provider, container](size_t*) { return provider->store.delete_container(container); },
                          nullptr);
    }

    // s12: XGameSaveDeleteContainerResult(XAsyncBlock*).
    int32_t DeleteContainerResult(uint64_t async) { return finish_async(async, 0, 0); }

    // XGameSaveContainerInfo as handed to the enumeration callback (x64; GDK field order: name,
    // displayName, blobCount, totalSize, lastModifiedTime, needsSync).
    struct GuestContainerInfo {
        uint64_t name;
        uint64_t display_name;
        uint32_t blob_count;
        uint32_t pad;
        uint64_t total_size;
        uint64_t last_modified;  // time_t (seconds since 1970)
        uint8_t needs_sync;      // never: nothing is synchronised
        uint8_t pad2[7];
    };
    static_assert(sizeof(GuestContainerInfo) == 48);

    // Calls bool callback(const XGameSaveContainerInfo*, void* context) for every container whose
    // name starts with `prefix`, until the callback returns false.
    int32_t EnumerateContainers(uint64_t provider_handle, const std::string& prefix, uint64_t context,
                                uint64_t callback) {
        auto provider = save_providers().find(provider_handle);
        if (!provider || callback == 0) {
            return E_INVALIDARG_HR;
        }
        std::vector<gamesave::ContainerInfo> containers;
        const int32_t hr = provider->store.enumerate_containers(prefix, &containers);
        if (hr < 0) {
            return hr;
        }
        const com::GuestFunction<uint64_t(uint64_t, uint64_t)> cb(callback);
        for (const gamesave::ContainerInfo& info : containers) {
            const uint64_t size = sizeof(GuestContainerInfo) + info.name.size() + 1 + info.display_name.size() + 1;
            const uint64_t block = com::guest_alloc(size);
            if (block == 0) {
                return E_OUTOFMEMORY_HR;
            }
            GuestContainerInfo guest{};
            guest.name = block + sizeof(GuestContainerInfo);
            guest.display_name = guest.name + info.name.size() + 1;
            guest.total_size = info.total_size;
            guest.blob_count = info.blob_count;
            guest.last_modified = info.last_modified / 10000000ull - 11644473600ull;
            com::write_guest(block, guest);
            com::write_guest(guest.name, info.name.c_str(), info.name.size() + 1);
            com::write_guest(guest.display_name, info.display_name.c_str(), info.display_name.size() + 1);
            const bool keep_going = (cb.call(com::current_guest_state(), block, context) & 0xFF) != 0;
            com::guest_free(block);
            if (!keep_going) {
                break;
            }
        }
        return S_OK_HR;
    }

    // s13: XGameSaveEnumerateContainerInfoByName(provider, prefix, context, callback).
    int32_t EnumerateContainerInfoByName(uint64_t provider, uint64_t prefix, uint64_t context, uint64_t callback) {
        return prefix == 0 ? E_INVALIDARG_HR
                           : EnumerateContainers(provider, guest_string(prefix, 256), context, callback);
    }

    // s14: XGameSaveEnumerateContainerInfo(provider, context, callback).
    int32_t EnumerateContainerInfo(uint64_t provider, uint64_t context, uint64_t callback) {
        return EnumerateContainers(provider, std::string(), context, callback);
    }

    // s15: no public XGameSave function, signature unknown: documented E_NOTIMPL, arguments untouched.
    int32_t GetContainerInfoUnsupported() {
        log_once("XGameSave slot 15 (GetContainerInfo)");
        return E_NOTIMPL_HR;
    }

    // s16: XGameSaveCreateContainer(provider, containerName, XGameSaveContainerHandle*). The title
    // opens every container it reads or writes this way (handle kept in its save manager, closed
    // through s17 before the next one is opened).
    int32_t CreateContainer(uint64_t provider_handle, uint64_t name, uint64_t container) {
        if (container == 0) {
            return E_POINTER_HR;
        }
        com::write_guest(container, uint64_t{0});
        auto provider = save_providers().find(provider_handle);
        if (!provider || name == 0) {
            return E_INVALIDARG_HR;
        }
        std::string text = guest_string(name, 512);
        if (!gamesave::valid_container_name(text)) {
            return gamesave::E_GS_INVALID_CONTAINER_NAME;
        }
        const uint64_t handle =
            save_containers().add(std::make_shared<SaveContainer>(SaveContainer{provider, std::move(text)}));
        com::write_guest(container, handle);
        return S_OK_HR;
    }

    // s17: XGameSaveCloseContainer(XGameSaveContainerHandle).
    void CloseContainer(uint64_t container) {
        if (!save_containers().remove(container)) {
            com::detail::log_bad_pointer("XGameSaveContainerHandle", container);
        }
    }

    // Calls bool callback(const XGameSaveBlobInfo*, void* context) for every blob whose name starts
    // with `prefix`, until the callback returns false. A container never submitted has no blobs.
    int32_t EnumerateBlobs(uint64_t container_handle, const std::string& prefix, uint64_t context, uint64_t callback) {
        auto container = save_containers().find(container_handle);
        if (!container || callback == 0) {
            return E_INVALIDARG_HR;
        }
        std::vector<gamesave::BlobInfo> blobs;
        const int32_t hr = container->provider->store.enumerate_blobs(container->name, &blobs);
        if (hr < 0 && hr != gamesave::E_GS_NO_SUCH_CONTAINER) {
            return hr;
        }
        const com::GuestFunction<uint64_t(uint64_t, uint64_t)> cb(callback);
        for (const gamesave::BlobInfo& info : blobs) {
            if (info.name.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            const uint64_t block = com::guest_alloc(sizeof(GuestBlobInfo) + info.name.size() + 1);
            if (block == 0) {
                return E_OUTOFMEMORY_HR;
            }
            GuestBlobInfo guest{};
            guest.name = block + sizeof(GuestBlobInfo);
            guest.size = info.size;
            com::write_guest(block, guest);
            com::write_guest(guest.name, info.name.c_str(), info.name.size() + 1);
            const bool keep_going = (cb.call(com::current_guest_state(), block, context) & 0xFF) != 0;
            com::guest_free(block);
            if (!keep_going) {
                break;
            }
        }
        return S_OK_HR;
    }

    // s18: XGameSaveEnumerateBlobInfo(container, context, callback).
    int32_t EnumerateBlobInfo(uint64_t container, uint64_t context, uint64_t callback) {
        return EnumerateBlobs(container, std::string(), context, callback);
    }

    // s19: XGameSaveEnumerateBlobInfoByName(container, blobNamePrefix, context, callback). The title
    // looks up "Save Game Header" this way and sizes its read buffer from the reported size.
    int32_t EnumerateBlobInfoByName(uint64_t container, uint64_t prefix, uint64_t context, uint64_t callback) {
        return prefix == 0 ? E_INVALIDARG_HR : EnumerateBlobs(container, guest_string(prefix, 256), context, callback);
    }

    // s20: XGameSaveReadBlobData(container, const char** blobNames, uint32_t* countOfBlobs,
    // size_t blobsSize, XGameSaveBlob* blobData). No names (count 0) reads every blob.
    int32_t ReadBlobData(uint64_t container_handle, uint64_t names, uint64_t count_ptr, uint64_t size,
                         uint64_t buffer) {
        auto container = save_containers().find(container_handle);
        uint32_t count = 0;
        if (!container || count_ptr == 0 || !com::read_guest(count_ptr, count)) {
            return E_INVALIDARG_HR;
        }
        com::write_guest(count_ptr, uint32_t{0});
        std::vector<std::string> wanted;
        if (!guest_names(names, count, &wanted)) {
            return E_INVALIDARG_HR;
        }
        ReadBlobs read;
        const int32_t hr = read_blobs(*container, std::move(wanted), &read);
        if (hr < 0) {
            return hr;
        }
        if (size < read.required()) {
            return gamesave::E_GS_PROVIDED_BUFFER_TOO_SMALL;
        }
        if (!read.blobs.empty() && (buffer == 0 || !write_blobs(buffer, read))) {
            return E_INVALIDARG_HR;
        }
        com::write_guest(count_ptr, static_cast<uint32_t>(read.blobs.size()));
        return S_OK_HR;
    }

    // s21: XGameSaveReadBlobDataAsync(container, const char** blobNames, uint32_t countOfBlobs,
    // XAsyncBlock*). The result size (XAsyncGetResultSize) is what s22 needs as its buffer.
    int32_t ReadBlobDataAsync(uint64_t container_handle, uint64_t names, uint32_t count, uint64_t async) {
        auto container = save_containers().find(container_handle);
        std::vector<std::string> wanted;
        if (!container || async == 0 || !guest_names(names, count, &wanted)) {
            return E_INVALIDARG_HR;
        }
        auto pending = std::make_shared<PendingRead>();
        {
            std::lock_guard<std::mutex> lock(g_pending_reads_mutex);
            g_pending_reads[async] = pending;
        }
        const int32_t hr = begin_work(
            async, "XGameSaveReadBlobDataAsync",
            [container, wanted, pending](size_t* required) {
                std::lock_guard<std::mutex> lock(pending->mutex);
                const HResult r = read_blobs(*container, wanted, &pending->read);
                *required = pending->read.required();
                return r;
            },
            [pending](uint64_t buffer) {
                std::lock_guard<std::mutex> lock(pending->mutex);
                return write_blobs(buffer, pending->read) ? S_OK_HR : E_INVALIDARG_HR;
            });
        if (hr < 0) {
            take_pending_read(async);
        }
        return hr;
    }

    // s22: XGameSaveReadBlobDataResult(XAsyncBlock*, size_t blobsSize, XGameSaveBlob* blobData,
    // uint32_t* countOfBlobs).
    int32_t ReadBlobDataResult(uint64_t async, uint64_t size, uint64_t buffer, uint64_t count_ptr) {
        if (count_ptr == 0) {
            return E_POINTER_HR;
        }
        com::write_guest(count_ptr, uint32_t{0});
        int32_t hr = finish_async(async, size, buffer);
        if (hr == E_NOT_SUFFICIENT_BUFFER_HR) {
            return gamesave::E_GS_PROVIDED_BUFFER_TOO_SMALL;
        }
        if (hr == E_PENDING_HR) {
            return hr;
        }
        auto pending = take_pending_read(async);
        if (hr >= 0 && pending) {
            std::lock_guard<std::mutex> lock(pending->mutex);
            com::write_guest(count_ptr, static_cast<uint32_t>(pending->read.blobs.size()));
        }
        return hr;
    }

    // s23: XGameSaveCreateUpdate(container, containerDisplayName, XGameSaveUpdateHandle*).
    int32_t CreateUpdate(uint64_t container_handle, uint64_t display_name, uint64_t update) {
        if (update == 0) {
            return E_POINTER_HR;
        }
        com::write_guest(update, uint64_t{0});
        auto container = save_containers().find(container_handle);
        if (!container) {
            return E_INVALIDARG_HR;
        }
        auto state = std::make_shared<SaveUpdate>();
        state->provider = container->provider;
        state->container = container->name;
        state->display_name = guest_string(display_name, gamesave::kMaxDisplayNameChars);
        com::write_guest(update, save_updates().add(std::move(state)));
        return S_OK_HR;
    }

    // s24: XGameSaveCloseUpdate(XGameSaveUpdateHandle). An unsubmitted update is dropped.
    void CloseUpdate(uint64_t update) {
        if (!save_updates().remove(update)) {
            com::detail::log_bad_pointer("XGameSaveUpdateHandle", update);
        }
    }

    // s25: XGameSaveSubmitBlobWrite(update, blobName, const uint8_t* data, size_t size). The data is
    // copied now; a second write of the same blob replaces the first.
    int32_t SubmitBlobWrite(uint64_t update_handle, uint64_t name, uint64_t data, uint64_t size) {
        auto update = save_updates().find(update_handle);
        if (!update || name == 0 || (data == 0 && size != 0) || size > gamesave::kMaxBlobBytes) {
            return E_INVALIDARG_HR;
        }
        gamesave::BlobWrite write;
        write.name = guest_string(name, 256);
        if (!gamesave::valid_blob_name(write.name)) {
            return E_INVALIDARG_HR;
        }
        write.data.resize(static_cast<size_t>(size));
        if (size != 0 && !com::read_guest(data, write.data.data(), write.data.size())) {
            return E_INVALIDARG_HR;
        }
        std::lock_guard<std::mutex> lock(update->mutex);
        if (update->submitted) {
            return E_ILLEGAL_METHOD_CALL_HR;
        }
        std::erase(update->deletes, write.name);
        std::erase_if(update->writes, [&](const gamesave::BlobWrite& w) { return w.name == write.name; });
        update->writes.push_back(std::move(write));
        return S_OK_HR;
    }

    // s26: XGameSaveSubmitBlobDelete(update, blobName).
    int32_t SubmitBlobDelete(uint64_t update_handle, uint64_t name) {
        auto update = save_updates().find(update_handle);
        if (!update || name == 0) {
            return E_INVALIDARG_HR;
        }
        std::string blob = guest_string(name, 256);
        if (!gamesave::valid_blob_name(blob)) {
            return E_INVALIDARG_HR;
        }
        std::lock_guard<std::mutex> lock(update->mutex);
        if (update->submitted) {
            return E_ILLEGAL_METHOD_CALL_HR;
        }
        std::erase_if(update->writes, [&](const gamesave::BlobWrite& w) { return w.name == blob; });
        if (std::find(update->deletes.begin(), update->deletes.end(), blob) == update->deletes.end()) {
            update->deletes.push_back(std::move(blob));
        }
        return S_OK_HR;
    }

    // s27: XGameSaveSubmitUpdate(update): all writes and deletes, or nothing.
    int32_t SubmitUpdate(uint64_t update_handle) {
        auto update = save_updates().find(update_handle);
        return update ? submit_update(*update) : E_INVALIDARG_HR;
    }

    // s28: XGameSaveSubmitUpdateAsync(update, XAsyncBlock*). The title submits its header and
    // payload blobs this way and reads the outcome in the completion callback (s29).
    int32_t SubmitUpdateAsync(uint64_t update_handle, uint64_t async) {
        auto update = save_updates().find(update_handle);
        if (!update || async == 0) {
            return E_INVALIDARG_HR;
        }
        return begin_work(async, "XGameSaveSubmitUpdateAsync",
                          [update](size_t*) { return submit_update(*update); }, nullptr);
    }

    // s29: XGameSaveSubmitUpdateResult(XAsyncBlock*).
    int32_t SubmitUpdateResult(uint64_t async) { return finish_async(async, 0, 0); }

    // s6: XGameSaveCloseProvider(XGameSaveProviderHandle).
    void CloseProvider(uint64_t provider) {
        if (!save_providers().remove(provider)) {
            com::detail::log_bad_pointer("XGameSaveProviderHandle", provider);
        }
    }

private:
    int32_t begin_work(uint64_t async, const char* name, SaveWorkOp::Work work, SaveWorkOp::Result result) {
        Pg6XAsyncBlock* shadow = begin_async(async);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        return XAsyncBeginWith(shadow, std::make_unique<SaveWorkOp>(std::move(work), std::move(result)), 0, name);
    }

    // Result half of an async call: copies the result (size bytes) to `buffer` and ends the
    // shadow block once the operation is no longer pending.
    static int32_t finish_async(uint64_t async, uint64_t size, uint64_t buffer) {
        Pg6XAsyncBlock* shadow = find_async(async);
        if (shadow == nullptr) {
            return E_INVALIDARG_HR;
        }
        size_t used = 0;
        const int32_t hr = XAsyncGetResultWith(shadow, 0, static_cast<size_t>(size), buffer, &used);
        if (hr != E_PENDING_HR && hr != E_NOT_SUFFICIENT_BUFFER_HR) {  // a larger buffer may follow
            cbk::async_shadow_end(async);
        }
        return hr;
    }

    static std::shared_ptr<PendingRead> take_pending_read(uint64_t async) {
        std::lock_guard<std::mutex> lock(g_pending_reads_mutex);
        auto it = g_pending_reads.find(async);
        if (it == g_pending_reads.end()) {
            return nullptr;
        }
        auto pending = std::move(it->second);
        g_pending_reads.erase(it);
        return pending;
    }

    void log_once(const char* api) {
        if (!logged_.exchange(true)) {
            os::log(std::string("xgameruntime: ") + api + ": local game-save provider (dir " +
                    gamesave::save_dir() + "); no cloud sync");
        }
    }

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
    {3, US::method<&UserApi::DuplicateHandle>("XUserDuplicateHandle")},
    {4, US::method<&UserApi::CloseHandle>("XUserCloseHandle")},
    {5, US::stub("XUserCompare")},
    {7, US::method<&UserApi::AddAsync>("XUserAddAsync")},
    {8, US::method<&UserApi::AddResult>("XUserAddResult")},
    {9, US::method<&UserApi::GetLocalId>("XUserGetLocalId")},
    {10, US::stub("XUserFindUserByLocalId")},
    {11, US::method<&UserApi::GetId>("XUserGetId")},
    {12, US::stub("XUserFindUserById")},
    {13, US::method<&UserApi::GetIsGuest>("XUserGetIsGuest")},
    {14, US::stub("XUser slot 14 (unidentified)")},
    {19, US::method<&UserApi::GetAgeGroup>("XUserGetAgeGroup")},
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
    GA::method<&GameApi::LaunchNewGame>("XLaunchNewGame"),
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

using GS = M<GameSaveApi>;
constexpr auto kGameSaveTable = slot_table<27>({
    {3, GS::method<&GameSaveApi::InitializeProvider>("XGameSaveInitializeProvider")},
    {4, GS::method<&GameSaveApi::InitializeProviderAsync>("XGameSaveInitializeProviderAsync")},
    {5, GS::method<&GameSaveApi::InitializeProviderResult>("XGameSaveInitializeProviderResult")},
    {6, GS::method<&GameSaveApi::CloseProvider>("XGameSaveCloseProvider")},
    {7, GS::method<&GameSaveApi::GetRemainingQuota>("XGameSaveGetRemainingQuota")},
    {8, GS::method<&GameSaveApi::GetRemainingQuotaAsync>("XGameSaveGetRemainingQuotaAsync")},
    {9, GS::method<&GameSaveApi::GetRemainingQuotaResult>("XGameSaveGetRemainingQuotaResult")},
    {10, GS::method<&GameSaveApi::DeleteContainer>("XGameSaveDeleteContainer")},
    {11, GS::method<&GameSaveApi::DeleteContainerAsync>("XGameSaveDeleteContainerAsync")},
    {12, GS::method<&GameSaveApi::DeleteContainerResult>("XGameSaveDeleteContainerResult")},
    {13, GS::method<&GameSaveApi::EnumerateContainerInfoByName>("XGameSaveEnumerateContainerInfoByName")},
    {14, GS::method<&GameSaveApi::EnumerateContainerInfo>("XGameSaveEnumerateContainerInfo")},
    {15, GS::method<&GameSaveApi::GetContainerInfoUnsupported>("XGameSaveGetContainerInfo")},
    {16, GS::method<&GameSaveApi::CreateContainer>("XGameSaveCreateContainer")},
    {17, GS::method<&GameSaveApi::CloseContainer>("XGameSaveCloseContainer")},
    {18, GS::method<&GameSaveApi::EnumerateBlobInfo>("XGameSaveEnumerateBlobInfo")},
    {19, GS::method<&GameSaveApi::EnumerateBlobInfoByName>("XGameSaveEnumerateBlobInfoByName")},
    {20, GS::method<&GameSaveApi::ReadBlobData>("XGameSaveReadBlobData")},
    {21, GS::method<&GameSaveApi::ReadBlobDataAsync>("XGameSaveReadBlobDataAsync")},
    {22, GS::method<&GameSaveApi::ReadBlobDataResult>("XGameSaveReadBlobDataResult")},
    {23, GS::method<&GameSaveApi::CreateUpdate>("XGameSaveCreateUpdate")},
    {24, GS::method<&GameSaveApi::CloseUpdate>("XGameSaveCloseUpdate")},
    {25, GS::method<&GameSaveApi::SubmitBlobWrite>("XGameSaveSubmitBlobWrite")},
    {26, GS::method<&GameSaveApi::SubmitBlobDelete>("XGameSaveSubmitBlobDelete")},
    {27, GS::method<&GameSaveApi::SubmitUpdate>("XGameSaveSubmitUpdate")},
    {28, GS::method<&GameSaveApi::SubmitUpdateAsync>("XGameSaveSubmitUpdateAsync")},
    {29, GS::method<&GameSaveApi::SubmitUpdateResult>("XGameSaveSubmitUpdateResult")},
});
constexpr auto kIGameSave = com::declare_interface<GameSaveApi>("xgameruntime.XGameSave", kGameSaveIid,
                                                               kGameSaveTable.slots, {.slot_count = 40});
using SX = M<StoreExtApi>;
constexpr auto kStoreExtTable = slot_table<70>({
    {71, SX::method<&StoreExtApi::AcquireLicenseForDurablesAsync>("XStoreAcquireLicenseForDurablesAsync")},
    {72, SX::method<&StoreExtApi::AcquireLicenseForDurablesResult>("XStoreAcquireLicenseForDurablesResult")},
});
constexpr auto kIStoreExt = com::declare_interface<StoreExtApi>("xgameruntime.XStoreExt", kStoreExtIid,
                                                                kStoreExtTable.slots, {.slot_count = 96});
using PL = M<PersistentLocalStorageApi>;
constexpr auto kPlsTable = slot_table<2>({
    {4, PL::method<&PersistentLocalStorageApi::GetPath>("XPersistentLocalStorageGetPath")},
});
constexpr auto kIPersistentLocalStorage = com::declare_interface<PersistentLocalStorageApi>(
    "xgameruntime.XPersistentLocalStorage", kPlsIid, kPlsTable.slots, {.slot_count = 16});
using SP = M<SpeechApi>;
// Slot map from the executable's wrappers (argument counts in brackets): s8 EnumerateInstalledVoices
// (2), s9 Create (1), s10 CloseHandle (1), s11 SetDefaultVoice (1), s12 SetCustomVoice (2), s13
// CreateStreamFromText (3), s14 CloseStreamHandle (1), s15 GetStreamDataSize (2), s16 GetStreamData
// (4), s21 CreateStreamFromSsml (3). s17..s20 are not called.
constexpr auto kSpeechTable = slot_table<19>({
    {8, SP::method<&SpeechApi::EnumerateInstalledVoices>("XSpeechSynthesizerEnumerateInstalledVoices")},
    {9, SP::method<&SpeechApi::Create>("XSpeechSynthesizerCreate")},
    {10, SP::method<&SpeechApi::Close>("XSpeechSynthesizerCloseHandle")},
    {11, SP::method<&SpeechApi::SetDefaultVoice>("XSpeechSynthesizerSetDefaultVoice")},
    {12, SP::method<&SpeechApi::SetCustomVoice>("XSpeechSynthesizerSetCustomVoice")},
    {13, SP::method<&SpeechApi::CreateStream>("XSpeechSynthesizerCreateStreamFromText")},
    {14, SP::method<&SpeechApi::CloseStream>("XSpeechSynthesizerCloseStreamHandle")},
    {15, SP::method<&SpeechApi::GetStreamSize>("XSpeechSynthesizerGetStreamDataSize")},
    {16, SP::method<&SpeechApi::GetStreamData>("XSpeechSynthesizerGetStreamData")},
    {21, SP::method<&SpeechApi::CreateStream>("XSpeechSynthesizerCreateStreamFromSsml")},
});
constexpr auto kISpeech = com::declare_interface<SpeechApi>("xgameruntime.XSpeechSynthesizer", kSpeechIid,
                                                            kSpeechTable.slots, {.slot_count = 32});
// R40: the four areas the title reaches from its legal and privacy screens no longer trap. Slot
// numbers are not identified, so every slot of an interface answers one documented HRESULT:
//   events {bbfbdcc7}: S_OK (writes accepted and dropped, truthful no-op);
//   shell {1b339674}: HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED) (URI launch not supported);
//   capture {a4f1aee2}: ERROR_NOT_SUPPORTED (screenshots);
//   XGameUi {dfcd4649}: E_ABORT (dialogs behave as cancelled by the user).
constexpr int32_t kHrNotSupported = static_cast<int32_t>(0x80070032u);  // HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)
constexpr auto kIEvent = com::declare_interface<StubApi>(
    "xgameruntime.XGameEvent", kEventGuid, {},
    {.slot_count = 16, .policy = com::SlotPolicy::ReturnFixed, .stub_result = 0});
constexpr auto kIDisplay = com::declare_interface<StubApi>(
    "xgameruntime.{1b339674} (shell)", kDisplayGuid, {},
    {.slot_count = 16, .policy = com::SlotPolicy::ReturnFixed, .stub_result = kHrNotSupported});
using HD = M<HdrModeApi>;
constexpr auto kUnmapped3Table = slot_table<2>({
    {4, HD::method<&HdrModeApi::TryEnableHdrMode>("XDisplayTryEnableHdrMode")},
});
constexpr auto kIUnmapped3 = com::declare_interface<HdrModeApi>("xgameruntime.{03f0fe74}", kUnmapped3Iid,
                                                                kUnmapped3Table.slots, {.slot_count = 16});
constexpr auto kIUnmapped9 = com::declare_interface<StubApi>(
    "xgameruntime.{a4f1aee2} (capture)", kUnmapped9Iid, {},
    {.slot_count = 16, .policy = com::SlotPolicy::ReturnFixed, .stub_result = kHrNotSupported});
using AI = M<AnalyticsInfoApi>;
constexpr auto kUnmapped10Table = slot_table<1>({
    {3, AI::method<&AnalyticsInfoApi::GetAnalyticsInfo>("XSystemGetAnalyticsInfo")},
});
constexpr auto kIUnmapped10 = com::declare_interface<AnalyticsInfoApi>(
    "xgameruntime.{b884675d}", kUnmapped10Guid, kUnmapped10Table.slots, {.slot_count = 16});
constexpr auto kIUnmapped13 = com::declare_interface<StubApi>(
    "xgameruntime.{dfcd4649} (XGameUi)", kUnmapped13Iid, {},
    {.slot_count = 32, .policy = com::SlotPolicy::ReturnFixed, .stub_result = E_ABORT_HR});

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
    {kGameSaveProvider, kGameSaveIid, &kIGameSave.info, [] { return make<GameSaveApi>(kIGameSave); }},
    {kStoreProvider, kStoreExtIid, &kIStoreExt.info, [] { return make<StoreExtApi>(kIStoreExt); }},
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

namespace {
std::string_view exe_basename(std::string_view path) {
    while (!path.empty() && (path.back() == ' ' || path.back() == '\t')) {
        path.remove_suffix(1);
    }
    const size_t slash = path.find_last_of("\\/");
    return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

bool iequals_ascii(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c; };
        if (lower(a[i]) != lower(b[i])) {
            return false;
        }
    }
    return true;
}
}  // namespace

bool is_self_launch_target(std::string_view exe_path, std::string_view command_line) {
    size_t start = command_line.find_first_not_of(" \t");
    if (start == std::string_view::npos) {
        return false;
    }
    std::string_view first;
    if (command_line[start] == '"') {
        const size_t end = command_line.find('"', start + 1);
        first = command_line.substr(start + 1, end == std::string_view::npos ? std::string_view::npos : end - start - 1);
    } else {
        const size_t end = command_line.find_first_of(" \t", start);
        first = command_line.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);
    }
    const std::string_view self = exe_basename(first);
    const std::string_view target = exe_basename(exe_path);
    return !self.empty() && iequals_ascii(self, target);
}

void set_xbox_title_id(uint32_t title_id) {
    g_title_id.store(title_id, std::memory_order_relaxed);
}

void set_xbox_live_sandbox_id(const std::string& sandbox_id) {
    std::lock_guard<std::mutex> lock(sandbox_mutex());
    sandbox_id_storage() = sandbox_id.substr(0, 15);  // XSystemXboxLiveSandboxIdMaxBytes is 16
}

std::string xbox_live_sandbox_id() {
    return configured_sandbox_id();
}

void set_user_age_group(UserAgeGroup group) {
    g_user_age_group.store(static_cast<uint32_t>(group), std::memory_order_relaxed);
}

UserAgeGroup user_age_group() {
    return static_cast<UserAgeGroup>(g_user_age_group.load(std::memory_order_relaxed));
}

bool parse_user_age_group(const std::string& text, UserAgeGroup* group) {
    static constexpr std::pair<const char*, UserAgeGroup> kNames[] = {
        {"unknown", UserAgeGroup::Unknown},
        {"child", UserAgeGroup::Child},
        {"teen", UserAgeGroup::Teen},
        {"adult", UserAgeGroup::Adult},
    };
    for (const auto& [name, value] : kNames) {
        if (text == name) {
            *group = value;
            return true;
        }
    }
    return false;
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
    save_providers().clear();
    save_containers().clear();
    save_updates().clear();
    {
        std::lock_guard<std::mutex> reads(g_pending_reads_mutex);
        g_pending_reads.clear();
    }
    g_title_id.store(0, std::memory_order_relaxed);
    set_xbox_live_sandbox_id("");
    set_user_age_group(UserAgeGroup::Unknown);
}

}  // namespace playgta6::os::gdk::provider
