#include "os/xgameruntime.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "os/threading.h"
#include "os/xasync_provider.h"

struct Pg6XTaskQueuePort {
    Pg6XTaskQueue* owner = nullptr;
    uint32_t port = 0;  // XTaskQueuePort
};

struct Pg6XTaskQueue {
    std::atomic<uint32_t> ref_count{1};
    std::unique_ptr<playgta6::os::XTaskQueue> queue;  // null for a composite queue
    Pg6XTaskQueuePort ports[2];                         // this queue's own work / completion ports
    Pg6XTaskQueuePort* borrowed[2] = {nullptr, nullptr};  // composite: the ports it submits to
    std::atomic<bool> composite_terminated{false};

    Pg6XTaskQueue() {
        ports[0] = Pg6XTaskQueuePort{this, 0};
        ports[1] = Pg6XTaskQueuePort{this, 1};
    }
    ~Pg6XTaskQueue();  // a composite releases the queues owning its ports
};

struct Pg6XUser {
    std::atomic<uint32_t> ref_count{1};
    uint64_t xuid = playgta6::os::gdk::kOfflineUserXuid;
    std::string gamertag = playgta6::os::gdk::kOfflineUserGamertag;
};

namespace playgta6::os::gdk {

namespace {

void retain_task_queue(Pg6XTaskQueue* q) {
    if (q != nullptr) {
        q->ref_count.fetch_add(1, std::memory_order_relaxed);
    }
}

void release_task_queue(Pg6XTaskQueue* q) {
    if (q != nullptr && q->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete q;
    }
}

// The XTaskQueue and port index a submission to (queue, port) reaches: the queue's own port, or
// for a composite queue the borrowed port's owner. nullptr for an invalid port.
playgta6::os::XTaskQueue* port_target(XTaskQueueHandle queue, XTaskQueuePort port, uint32_t& index) {
    const auto i = static_cast<uint32_t>(port);
    if (i > 1) {
        return nullptr;
    }
    if (queue->queue != nullptr) {
        index = i;
        return queue->queue.get();
    }
    const Pg6XTaskQueuePort* p = queue->borrowed[i];
    if (p == nullptr || p->owner == nullptr || p->owner->queue == nullptr) {
        return nullptr;
    }
    index = p->port;
    return p->owner->queue.get();
}

void release_user(Pg6XUser* user) {
    if (user != nullptr && user->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete user;
    }
}

bool map_dispatch_mode(XTaskQueueDispatchMode mode,
                       playgta6::os::XTaskQueue::DispatchMode& out_mode,
                       uint32_t& out_threads) {
    switch (mode) {
        case XTaskQueueDispatchMode::ThreadPool:
            out_mode = playgta6::os::XTaskQueue::DispatchMode::ThreadPool;
            out_threads = 2;
            return true;
        case XTaskQueueDispatchMode::SerializedThreadPool:
            out_mode = playgta6::os::XTaskQueue::DispatchMode::ThreadPool;
            out_threads = 1;
            return true;
        case XTaskQueueDispatchMode::Manual:
            out_mode = playgta6::os::XTaskQueue::DispatchMode::Manual;
            out_threads = 0;
            return true;
        case XTaskQueueDispatchMode::Immediate:
            out_mode = playgta6::os::XTaskQueue::DispatchMode::Immediate;
            out_threads = 0;
            return true;
        default:
            return false;
    }
}

struct RuntimeState {
    std::mutex mutex;
    uint32_t init_count = 0;
    Pg6XTaskQueue* process_queue = nullptr;

    ~RuntimeState() {
        Pg6XTaskQueue* to_release = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex);
            to_release = process_queue;
            process_queue = nullptr;
        }
        if (to_release != nullptr) {
            release_task_queue(to_release);
        }
    }
};

RuntimeState& runtime_state() {
    static RuntimeState state;
    return state;
}

}  // namespace

// ---------------------------------------------------------------------------
// Game Runtime lifecycle
// ---------------------------------------------------------------------------

HResult XGameRuntimeInitialize() {
    auto& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.init_count;
    return S_OK_HR;
}

void XGameRuntimeUninitialize() {
    Pg6XTaskQueue* queue_to_release = nullptr;
    {
        auto& state = runtime_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.init_count == 0) {
            return;
        }
        --state.init_count;
        if (state.init_count == 0) {
            queue_to_release = state.process_queue;
            state.process_queue = nullptr;
        }
    }
    if (queue_to_release != nullptr) {
        queue_to_release->queue->terminate(true, true);
        release_task_queue(queue_to_release);
    }
}

bool XGameRuntimeIsInitialized() {
    auto& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.init_count > 0;
}

// ---------------------------------------------------------------------------
// XTaskQueue C-ABI handles
// ---------------------------------------------------------------------------

HResult XTaskQueueCreate(XTaskQueueDispatchMode work_mode,
                         XTaskQueueDispatchMode completion_mode,
                         XTaskQueueHandle* queue) {
    if (queue == nullptr) {
        return E_POINTER_HR;
    }
    *queue = nullptr;

    playgta6::os::XTaskQueue::DispatchMode mapped_work{};
    playgta6::os::XTaskQueue::DispatchMode mapped_completion{};
    uint32_t work_threads = 0;
    uint32_t completion_threads = 0;
    if (!map_dispatch_mode(work_mode, mapped_work, work_threads) ||
        !map_dispatch_mode(completion_mode, mapped_completion, completion_threads)) {
        return E_INVALIDARG_HR;
    }

    auto* handle = new Pg6XTaskQueue();
    handle->queue = std::make_unique<playgta6::os::XTaskQueue>(
        mapped_work, mapped_completion, work_threads);
    *queue = handle;
    return S_OK_HR;
}

void XTaskQueueCloseHandle(XTaskQueueHandle queue) {
    release_task_queue(queue);
}

HResult XTaskQueueDuplicateHandle(XTaskQueueHandle queue, XTaskQueueHandle* duplicated_handle) {
    if (duplicated_handle == nullptr) {
        return E_POINTER_HR;
    }
    *duplicated_handle = nullptr;
    if (queue == nullptr) {
        return E_INVALIDARG_HR;
    }
    retain_task_queue(queue);
    *duplicated_handle = queue;
    return S_OK_HR;
}

HResult XTaskQueueSubmitCallback(XTaskQueueHandle queue,
                                 XTaskQueuePort port,
                                 void* callback_context,
                                 XTaskQueueCallback callback) {
    if (queue == nullptr || callback == nullptr) {
        return E_INVALIDARG_HR;
    }
    uint32_t index = 0;
    playgta6::os::XTaskQueue* target = port_target(queue, port, index);
    if (target == nullptr) {
        return E_INVALIDARG_HR;
    }
    if (queue->queue == nullptr && queue->composite_terminated.load(std::memory_order_acquire)) {
        return E_ABORT_HR;
    }

    auto wrapper = [callback, callback_context](bool canceled) {
        callback(callback_context, canceled);
    };
    const bool submitted = index == 0 ? target->submit_work_cancellable(std::move(wrapper))
                                      : target->submit_completion_cancellable(std::move(wrapper));
    return submitted ? S_OK_HR : E_ABORT_HR;
}

bool XTaskQueueDispatch(XTaskQueueHandle queue,
                        XTaskQueuePort port,
                        uint32_t timeout_ms) {
    if (queue == nullptr) {
        return false;
    }
    uint32_t index = 0;
    playgta6::os::XTaskQueue* target = port_target(queue, port, index);
    if (target == nullptr) {
        return false;
    }
    return index == 0 ? target->dispatch_work(timeout_ms) : target->dispatch_completion(timeout_ms);
}

HResult XTaskQueueTerminate(XTaskQueueHandle queue,
                            bool wait,
                            void* callback_context,
                            XTaskQueueTerminatedCallback callback) {
    if (queue == nullptr) {
        return E_INVALIDARG_HR;
    }
    if (queue->queue == nullptr) {
        // Composite: stop submissions through it; the borrowed ports keep running.
        queue->composite_terminated.store(true, std::memory_order_release);
    } else {
        queue->queue->terminate(wait, true);
    }
    if (callback != nullptr) {
        callback(callback_context);
    }
    return S_OK_HR;
}

bool XTaskQueueGetCurrentProcessTaskQueue(XTaskQueueHandle* queue) {
    if (queue == nullptr) {
        return false;
    }
    *queue = nullptr;

    auto& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.process_queue == nullptr || state.process_queue->queue->is_terminated()) {
        if (state.process_queue != nullptr) {
            release_task_queue(state.process_queue);
            state.process_queue = nullptr;
        }
        auto* created = new Pg6XTaskQueue();
        created->queue = std::make_unique<playgta6::os::XTaskQueue>(
            playgta6::os::XTaskQueue::DispatchMode::ThreadPool,
            playgta6::os::XTaskQueue::DispatchMode::ThreadPool,
            2u);
        state.process_queue = created;
    }

    retain_task_queue(state.process_queue);
    *queue = state.process_queue;
    return true;
}

HResult XTaskQueueGetPort(XTaskQueueHandle queue, XTaskQueuePort port, XTaskQueuePortHandle* port_handle) {
    if (port_handle == nullptr) {
        return E_POINTER_HR;
    }
    *port_handle = nullptr;
    const auto i = static_cast<uint32_t>(port);
    if (queue == nullptr || i > 1) {
        return E_INVALIDARG_HR;
    }
    *port_handle = queue->queue != nullptr ? &queue->ports[i] : queue->borrowed[i];
    return *port_handle != nullptr ? S_OK_HR : E_INVALIDARG_HR;
}

HResult XTaskQueueCreateComposite(XTaskQueuePortHandle work_port,
                                  XTaskQueuePortHandle completion_port,
                                  XTaskQueueHandle* queue) {
    if (queue == nullptr) {
        return E_POINTER_HR;
    }
    *queue = nullptr;
    if (work_port == nullptr || completion_port == nullptr || work_port->owner == nullptr ||
        completion_port->owner == nullptr) {
        return E_INVALIDARG_HR;
    }
    auto* composite = new Pg6XTaskQueue();
    composite->borrowed[0] = work_port;
    composite->borrowed[1] = completion_port;
    retain_task_queue(work_port->owner);
    retain_task_queue(completion_port->owner);
    *queue = composite;
    return S_OK_HR;
}

// ---------------------------------------------------------------------------
// Offline XUser stub
// ---------------------------------------------------------------------------

namespace {

const char kXUserAddIdentity = 0;

// XUserAddAsync through the XAsync provider framework: DoWork on the block's queue creates the
// offline user, GetResult hands it out, Cleanup releases one nobody took.
class XUserAddProvider final : public XAsyncProviderImpl {
public:
    ~XUserAddProvider() override { release_user(user_); }

    HResult call(XAsyncOp op, XAsyncBlock* block, size_t buffer_size, uint64_t buffer) override {
        switch (op) {
            case XAsyncOp::Begin:
                return XAsyncSchedule(block, 0);
            case XAsyncOp::DoWork:
                user_ = new Pg6XUser();
                XAsyncComplete(block, S_OK_HR, sizeof(XUserHandle));
                return S_OK_HR;
            case XAsyncOp::GetResult:
                if (buffer == 0 || buffer_size < sizeof(XUserHandle)) {
                    return E_INVALIDARG_HR;
                }
                *reinterpret_cast<XUserHandle*>(static_cast<uintptr_t>(buffer)) = user_;
                user_ = nullptr;
                return S_OK_HR;
            case XAsyncOp::Cleanup:
                release_user(user_);
                user_ = nullptr;
                return S_OK_HR;
            default:
                return S_OK_HR;  // Cancel: a queued DoWork then completes with E_ABORT
        }
    }

private:
    Pg6XUser* user_ = nullptr;
};

}  // namespace

HResult XUserAddAsync(XUserAddOptions options, XAsyncBlock* async_block) {
    (void)options;
    if (async_block == nullptr) {
        return E_INVALIDARG_HR;
    }
    return XAsyncBeginWith(async_block, std::make_unique<XUserAddProvider>(),
                           static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&kXUserAddIdentity)), "XUserAddAsync");
}

HResult XUserAddResult(XAsyncBlock* async_block, XUserHandle* new_user) {
    if (async_block == nullptr || new_user == nullptr) {
        return E_INVALIDARG_HR;
    }
    *new_user = nullptr;
    return XAsyncGetResultWith(async_block, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&kXUserAddIdentity)),
                               sizeof(XUserHandle), static_cast<uint64_t>(reinterpret_cast<uintptr_t>(new_user)),
                               nullptr);
}

HResult XUserGetId(XUserHandle user, uint64_t* user_id) {
    if (user == nullptr || user_id == nullptr) {
        return E_INVALIDARG_HR;
    }
    *user_id = user->xuid;
    return S_OK_HR;
}

HResult XUserGetGamertag(XUserHandle user,
                         XUserGamertagComponent gamertag_component,
                         size_t gamertag_size,
                         char* gamertag,
                         size_t* gamertag_used) {
    if (user == nullptr) {
        return E_INVALIDARG_HR;
    }

    const char* src = nullptr;
    switch (gamertag_component) {
        case XUserGamertagComponent::Classic:
        case XUserGamertagComponent::Modern:
        case XUserGamertagComponent::UniqueModern:
            src = user->gamertag.c_str();
            break;
        case XUserGamertagComponent::ModernSuffix:
            src = "";
            break;
        default:
            return E_INVALIDARG_HR;
    }

    const size_t required = std::strlen(src) + 1;
    if (gamertag_used != nullptr) {
        *gamertag_used = required;
    }

    if (gamertag == nullptr) {
        return gamertag_size == 0 ? E_NOT_SUFFICIENT_BUFFER_HR : E_INVALIDARG_HR;
    }
    if (gamertag_size < required) {
        return E_NOT_SUFFICIENT_BUFFER_HR;
    }

    std::memcpy(gamertag, src, required);
    return S_OK_HR;
}

void XUserCloseHandle(XUserHandle user) {
    release_user(user);
}

// ---------------------------------------------------------------------------
// XSystem device & analytics info
// ---------------------------------------------------------------------------

XSystemDeviceType XSystemGetDeviceType() {
    return XSystemDeviceType::XboxScarlettAnaconda;
}

XSystemAnalyticsInfo XSystemGetAnalyticsInfo() {
    XSystemAnalyticsInfo info{};
    info.os_version = XVersion{10, 0, 26100, 1};
    info.hosting_os_version = XVersion{10, 0, 26100, 1};
    std::snprintf(info.family, sizeof(info.family), "%s", "Xbox");
    std::snprintf(info.form, sizeof(info.form), "%s", "Xbox Series X");
    return info;
}

}  // namespace playgta6::os::gdk

Pg6XTaskQueue::~Pg6XTaskQueue() {
    for (Pg6XTaskQueuePort* port : borrowed) {
        if (port != nullptr) {
            playgta6::os::gdk::release_task_queue(port->owner);
        }
    }
}

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints
// ---------------------------------------------------------------------------

extern "C" {

Pg6HResult pg6_gdk_xgame_runtime_initialize(void) {
    return playgta6::os::gdk::XGameRuntimeInitialize();
}

void pg6_gdk_xgame_runtime_uninitialize(void) {
    playgta6::os::gdk::XGameRuntimeUninitialize();
}

int32_t pg6_gdk_xgame_runtime_is_initialized(void) {
    return playgta6::os::gdk::XGameRuntimeIsInitialized() ? 1 : 0;
}

Pg6HResult pg6_gdk_xtask_queue_create(uint32_t work_mode,
                                      uint32_t completion_mode,
                                      Pg6XTaskQueueHandle* queue) {
    return playgta6::os::gdk::XTaskQueueCreate(
        static_cast<playgta6::os::gdk::XTaskQueueDispatchMode>(work_mode),
        static_cast<playgta6::os::gdk::XTaskQueueDispatchMode>(completion_mode),
        queue);
}

void pg6_gdk_xtask_queue_close_handle(Pg6XTaskQueueHandle queue) {
    playgta6::os::gdk::XTaskQueueCloseHandle(queue);
}

Pg6HResult pg6_gdk_xtask_queue_duplicate_handle(Pg6XTaskQueueHandle queue,
                                                Pg6XTaskQueueHandle* duplicated_handle) {
    return playgta6::os::gdk::XTaskQueueDuplicateHandle(queue, duplicated_handle);
}

Pg6HResult pg6_gdk_xtask_queue_submit_callback(Pg6XTaskQueueHandle queue,
                                               uint32_t port,
                                               void* callback_context,
                                               Pg6XTaskQueueCallback callback) {
    return playgta6::os::gdk::XTaskQueueSubmitCallback(
        queue,
        static_cast<playgta6::os::gdk::XTaskQueuePort>(port),
        callback_context,
        callback);
}

int32_t pg6_gdk_xtask_queue_dispatch(Pg6XTaskQueueHandle queue,
                                     uint32_t port,
                                     uint32_t timeout_ms) {
    return playgta6::os::gdk::XTaskQueueDispatch(
               queue,
               static_cast<playgta6::os::gdk::XTaskQueuePort>(port),
               timeout_ms)
               ? 1
               : 0;
}

Pg6HResult pg6_gdk_xtask_queue_terminate(Pg6XTaskQueueHandle queue,
                                         int32_t wait,
                                         void* callback_context,
                                         Pg6XTaskQueueTerminatedCallback callback) {
    return playgta6::os::gdk::XTaskQueueTerminate(
        queue, wait != 0, callback_context, callback);
}

int32_t pg6_gdk_xtask_queue_get_current_process_task_queue(Pg6XTaskQueueHandle* queue) {
    return playgta6::os::gdk::XTaskQueueGetCurrentProcessTaskQueue(queue) ? 1 : 0;
}

Pg6HResult pg6_gdk_xtask_queue_get_port(Pg6XTaskQueueHandle queue,
                                        uint32_t port,
                                        Pg6XTaskQueuePortHandle* port_handle) {
    return playgta6::os::gdk::XTaskQueueGetPort(queue, static_cast<playgta6::os::gdk::XTaskQueuePort>(port),
                                                port_handle);
}

Pg6HResult pg6_gdk_xtask_queue_create_composite(Pg6XTaskQueuePortHandle work_port,
                                                Pg6XTaskQueuePortHandle completion_port,
                                                Pg6XTaskQueueHandle* queue) {
    return playgta6::os::gdk::XTaskQueueCreateComposite(work_port, completion_port, queue);
}

Pg6HResult pg6_gdk_xasync_get_status(Pg6XAsyncBlock* async_block, int32_t wait) {
    return playgta6::os::gdk::XAsyncGetStatus(async_block, wait != 0);
}

Pg6HResult pg6_gdk_xasync_get_result_size(Pg6XAsyncBlock* async_block, size_t* buffer_size) {
    return playgta6::os::gdk::XAsyncGetResultSize(async_block, buffer_size);
}

Pg6HResult pg6_gdk_xuser_add_async(uint32_t options, Pg6XAsyncBlock* async_block) {
    return playgta6::os::gdk::XUserAddAsync(
        static_cast<playgta6::os::gdk::XUserAddOptions>(options), async_block);
}

Pg6HResult pg6_gdk_xuser_add_result(Pg6XAsyncBlock* async_block, Pg6XUserHandle* new_user) {
    return playgta6::os::gdk::XUserAddResult(async_block, new_user);
}

Pg6HResult pg6_gdk_xuser_get_id(Pg6XUserHandle user, uint64_t* user_id) {
    return playgta6::os::gdk::XUserGetId(user, user_id);
}

Pg6HResult pg6_gdk_xuser_get_gamertag(Pg6XUserHandle user,
                                      uint32_t gamertag_component,
                                      size_t gamertag_size,
                                      char* gamertag,
                                      size_t* gamertag_used) {
    return playgta6::os::gdk::XUserGetGamertag(
        user,
        static_cast<playgta6::os::gdk::XUserGamertagComponent>(gamertag_component),
        gamertag_size,
        gamertag,
        gamertag_used);
}

void pg6_gdk_xuser_close_handle(Pg6XUserHandle user) {
    playgta6::os::gdk::XUserCloseHandle(user);
}

uint32_t pg6_gdk_xsystem_get_device_type(void) {
    return static_cast<uint32_t>(playgta6::os::gdk::XSystemGetDeviceType());
}

Pg6XSystemAnalyticsInfo pg6_gdk_xsystem_get_analytics_info(void) {
    return playgta6::os::gdk::XSystemGetAnalyticsInfo();
}

}  // extern "C"
