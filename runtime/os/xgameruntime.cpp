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

struct Pg6XTaskQueue {
    std::atomic<uint32_t> ref_count{1};
    std::unique_ptr<playgta6::os::XTaskQueue> queue;
};

struct Pg6XUser {
    std::atomic<uint32_t> ref_count{1};
    uint64_t xuid = playgta6::os::gdk::kOfflineUserXuid;
    std::string gamertag = playgta6::os::gdk::kOfflineUserGamertag;
};

namespace playgta6::os::gdk {

namespace {

constexpr uint64_t kAsyncMagic = 0x5047364153594E43ULL;  // "PG6ASYNC"
constexpr uint32_t kAsyncFlagResultReady = 1u << 0;
constexpr uint32_t kAsyncFlagConsumed = 1u << 1;

struct AsyncInternalHeader {
    uint64_t magic = 0;
    uint64_t op_id = 0;
    int32_t cached_status = E_PENDING_HR;
    uint32_t flags = 0;
    uint64_t cached_result_size = 0;
};

static_assert(sizeof(AsyncInternalHeader) <= sizeof(Pg6XAsyncBlock::internal),
              "AsyncInternalHeader must fit inside Pg6XAsyncBlock::internal");

void write_async_header(XAsyncBlock* block, const AsyncInternalHeader& header) {
    std::memcpy(block->internal, &header, sizeof(header));
}

AsyncInternalHeader read_async_header(const XAsyncBlock* block) {
    AsyncInternalHeader header{};
    std::memcpy(&header, block->internal, sizeof(header));
    return header;
}

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

void release_user(Pg6XUser* user) {
    if (user != nullptr && user->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete user;
    }
}

struct ScopedTaskQueueRef {
    Pg6XTaskQueue* ptr = nullptr;

    explicit ScopedTaskQueueRef(Pg6XTaskQueue* q = nullptr, bool add_ref = true) : ptr(q) {
        if (add_ref && ptr != nullptr) {
            retain_task_queue(ptr);
        }
    }

    ~ScopedTaskQueueRef() { release_task_queue(ptr); }

    ScopedTaskQueueRef(const ScopedTaskQueueRef& other) : ptr(other.ptr) {
        retain_task_queue(ptr);
    }

    ScopedTaskQueueRef& operator=(const ScopedTaskQueueRef& other) {
        if (this != &other) {
            retain_task_queue(other.ptr);
            release_task_queue(ptr);
            ptr = other.ptr;
        }
        return *this;
    }

    ScopedTaskQueueRef(ScopedTaskQueueRef&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    ScopedTaskQueueRef& operator=(ScopedTaskQueueRef&& other) noexcept {
        if (this != &other) {
            release_task_queue(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
};

struct AsyncOperationState {
    std::mutex mutex;
    std::condition_variable cv;
    HResult status = E_PENDING_HR;
    size_t result_size = 0;
    bool result_ready = false;
    bool callback_completed = false;
    bool result_consumed = false;
    std::thread::id callback_thread{};
    Pg6XUser* user_result = nullptr;

    ~AsyncOperationState() {
        if (user_result != nullptr) {
            release_user(user_result);
            user_result = nullptr;
        }
    }
};

struct AsyncRegistry {
    std::mutex mutex;
    uint64_t next_id = 1;
    std::unordered_map<uint64_t, std::shared_ptr<AsyncOperationState>> ops;

    std::pair<uint64_t, std::shared_ptr<AsyncOperationState>> create() {
        auto state = std::make_shared<AsyncOperationState>();
        std::lock_guard<std::mutex> lock(mutex);
        const uint64_t id = next_id++;
        ops.emplace(id, state);
        return {id, std::move(state)};
    }

    std::shared_ptr<AsyncOperationState> find(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = ops.find(id);
        return it != ops.end() ? it->second : nullptr;
    }

    void erase(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        ops.erase(id);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        ops.clear();
    }
};

AsyncRegistry& async_registry() {
    static AsyncRegistry reg;
    return reg;
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

void complete_async_op(XAsyncBlock* async_block,
                       uint64_t op_id,
                       const std::shared_ptr<AsyncOperationState>& state,
                       HResult final_status,
                       size_t final_size) {
    XAsyncCompletionRoutine cb = nullptr;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->status = final_status;
        state->result_size = final_size;
        state->result_ready = true;
        state->callback_thread = std::this_thread::get_id();
        if (failed(final_status) && state->user_result != nullptr) {
            release_user(state->user_result);
            state->user_result = nullptr;
        }
        if (async_block != nullptr) {
            AsyncInternalHeader header{};
            header.magic = kAsyncMagic;
            header.op_id = op_id;
            header.cached_status = final_status;
            header.flags = kAsyncFlagResultReady;
            header.cached_result_size = static_cast<uint64_t>(final_size);
            write_async_header(async_block, header);
            cb = async_block->callback;
        }
    }

    if (cb != nullptr && async_block != nullptr) {
        cb(async_block);
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->callback_thread = std::thread::id{};
        state->callback_completed = true;
    }
    state->cv.notify_all();
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
    bool clear_ops = false;
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
            clear_ops = true;
        }
    }
    if (queue_to_release != nullptr) {
        queue_to_release->queue->terminate(true, true);
        release_task_queue(queue_to_release);
    }
    if (clear_ops) {
        async_registry().clear();
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
    if (queue == nullptr || queue->queue == nullptr || callback == nullptr) {
        return E_INVALIDARG_HR;
    }

    auto wrapper = [callback, callback_context](bool canceled) {
        callback(callback_context, canceled);
    };

    bool submitted = false;
    switch (port) {
        case XTaskQueuePort::Work:
            submitted = queue->queue->submit_work_cancellable(std::move(wrapper));
            break;
        case XTaskQueuePort::Completion:
            submitted = queue->queue->submit_completion_cancellable(std::move(wrapper));
            break;
        default:
            return E_INVALIDARG_HR;
    }

    return submitted ? S_OK_HR : E_ABORT_HR;
}

bool XTaskQueueDispatch(XTaskQueueHandle queue,
                        XTaskQueuePort port,
                        uint32_t timeout_ms) {
    if (queue == nullptr || queue->queue == nullptr) {
        return false;
    }
    switch (port) {
        case XTaskQueuePort::Work:
            return queue->queue->dispatch_work(timeout_ms);
        case XTaskQueuePort::Completion:
            return queue->queue->dispatch_completion(timeout_ms);
        default:
            return false;
    }
}

HResult XTaskQueueTerminate(XTaskQueueHandle queue,
                            bool wait,
                            void* callback_context,
                            XTaskQueueTerminatedCallback callback) {
    if (queue == nullptr || queue->queue == nullptr) {
        return E_INVALIDARG_HR;
    }
    queue->queue->terminate(wait, true);
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

// ---------------------------------------------------------------------------
// XAsyncBlock helpers
// ---------------------------------------------------------------------------

HResult XAsyncGetStatus(XAsyncBlock* async_block, bool wait) {
    if (async_block == nullptr) {
        return E_INVALIDARG_HR;
    }
    const AsyncInternalHeader header = read_async_header(async_block);
    if (header.magic != kAsyncMagic) {
        return E_INVALIDARG_HR;
    }

    auto state = async_registry().find(header.op_id);
    if (!state) {
        if ((header.flags & kAsyncFlagResultReady) != 0) {
            return header.cached_status;
        }
        return E_INVALIDARG_HR;
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    if (wait && !state->callback_completed) {
        if (state->callback_thread == std::this_thread::get_id() && state->result_ready) {
            return state->status;
        }
        state->cv.wait(lock, [&]() { return state->callback_completed; });
    }
    return state->result_ready ? state->status : E_PENDING_HR;
}

HResult XAsyncGetResultSize(XAsyncBlock* async_block, size_t* buffer_size) {
    if (async_block == nullptr || buffer_size == nullptr) {
        return E_INVALIDARG_HR;
    }
    *buffer_size = 0;

    const AsyncInternalHeader header = read_async_header(async_block);
    if (header.magic != kAsyncMagic) {
        return E_INVALIDARG_HR;
    }

    auto state = async_registry().find(header.op_id);
    if (!state) {
        if ((header.flags & kAsyncFlagResultReady) != 0) {
            if (failed(header.cached_status)) {
                return header.cached_status;
            }
            *buffer_size = static_cast<size_t>(header.cached_result_size);
            return S_OK_HR;
        }
        return E_INVALIDARG_HR;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->result_ready) {
        return E_PENDING_HR;
    }
    if (failed(state->status)) {
        return state->status;
    }
    *buffer_size = state->result_size;
    return S_OK_HR;
}

// ---------------------------------------------------------------------------
// Offline XUser stub
// ---------------------------------------------------------------------------

HResult XUserAddAsync(XUserAddOptions options, XAsyncBlock* async_block) {
    (void)options;
    if (async_block == nullptr) {
        return E_INVALIDARG_HR;
    }

    ScopedTaskQueueRef target_queue;
    if (async_block->queue != nullptr) {
        target_queue = ScopedTaskQueueRef(async_block->queue, /*add_ref=*/true);
    } else {
        XTaskQueueHandle proc_queue = nullptr;
        if (!XTaskQueueGetCurrentProcessTaskQueue(&proc_queue) || proc_queue == nullptr) {
            return E_FAIL_HR;
        }
        // XTaskQueueGetCurrentProcessTaskQueue already incremented ref_count.
        target_queue = ScopedTaskQueueRef(proc_queue, /*add_ref=*/false);
    }

    auto [op_id, state] = async_registry().create();

    AsyncInternalHeader init_header{};
    init_header.magic = kAsyncMagic;
    init_header.op_id = op_id;
    init_header.cached_status = E_PENDING_HR;
    init_header.flags = 0;
    init_header.cached_result_size = 0;
    write_async_header(async_block, init_header);

    const bool submitted = target_queue.ptr->queue->submit_work_cancellable(
        [async_block, op_id, state, target_queue](bool work_canceled) {
            if (work_canceled) {
                complete_async_op(async_block, op_id, state, E_ABORT_HR, 0);
                return;
            }

            auto* offline_user = new Pg6XUser();
            offline_user->xuid = kOfflineUserXuid;
            offline_user->gamertag = kOfflineUserGamertag;

            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->user_result = offline_user;
            }

            const bool comp_submitted = target_queue.ptr->queue->submit_completion_cancellable(
                [async_block, op_id, state, target_queue](bool comp_canceled) {
                    if (comp_canceled) {
                        complete_async_op(async_block, op_id, state, E_ABORT_HR, 0);
                    } else {
                        complete_async_op(
                            async_block, op_id, state, S_OK_HR, sizeof(XUserHandle));
                    }
                });

            if (!comp_submitted) {
                complete_async_op(async_block, op_id, state, E_ABORT_HR, 0);
            }
        });

    if (!submitted) {
        complete_async_op(async_block, op_id, state, E_ABORT_HR, 0);
        return E_ABORT_HR;
    }

    return S_OK_HR;
}

HResult XUserAddResult(XAsyncBlock* async_block, XUserHandle* new_user) {
    if (async_block == nullptr || new_user == nullptr) {
        return E_INVALIDARG_HR;
    }
    *new_user = nullptr;

    AsyncInternalHeader header = read_async_header(async_block);
    if (header.magic != kAsyncMagic) {
        return E_INVALIDARG_HR;
    }

    auto state = async_registry().find(header.op_id);
    if (!state) {
        if ((header.flags & kAsyncFlagConsumed) != 0) {
            return E_FAIL_HR;
        }
        return E_INVALIDARG_HR;
    }

    HResult result_hr = S_OK_HR;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->result_ready) {
            return E_PENDING_HR;
        }
        if (failed(state->status)) {
            result_hr = state->status;
        } else if (state->result_consumed || state->user_result == nullptr) {
            return E_FAIL_HR;
        } else {
            *new_user = state->user_result;
            state->user_result = nullptr;
            state->result_consumed = true;
            header.flags |= kAsyncFlagConsumed;
            write_async_header(async_block, header);
            result_hr = S_OK_HR;
        }
    }

    async_registry().erase(header.op_id);
    return result_hr;
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
