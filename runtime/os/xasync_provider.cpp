#include "os/xasync_provider.h"

#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "os/os_shim.h"

namespace playgta6::os::gdk {

namespace {

constexpr uint32_t kSignature = 0x58364750u;  // "PG6X"

// What XAsyncBegin writes into XAsyncBlock::internal.
struct BlockHeader {
    uint32_t signature;
    uint32_t reserved;
    uint64_t op_id;
    uint8_t unused[16];
};
static_assert(sizeof(BlockHeader) == sizeof(Pg6XAsyncBlock::internal), "header fills XAsyncBlock::internal");

struct Op {
    uint64_t id = 0;
    XAsyncBlock* block = nullptr;
    // Not a reference: like XTaskQueueSubmitCallback, an operation needs its queue to stay alive
    // until it completed (a reference released on the queue's own worker could destroy it there).
    XTaskQueueHandle queue = nullptr;
    std::unique_ptr<XAsyncProviderImpl> provider;  // null after Cleanup (or for immediate ops)
    uint64_t identity = 0;
    std::string identity_name;

    std::mutex mutex;
    std::condition_variable cv;
    HResult status = E_PENDING_HR;
    size_t required = 0;
    XAsyncCompletionRoutine callback = nullptr;
    bool completed = false;
    bool canceled = false;
    bool callback_done = false;
    bool cleaned_up = false;
    bool result_taken = false;
    bool abandoned = false;
    int inflight = 0;  // provider calls running
    std::thread::id callback_thread{};
};

using OpRef = std::shared_ptr<Op>;

struct Registry {
    std::mutex mutex;
    std::unordered_map<XAsyncBlock*, OpRef> ops;
    uint64_t next_id = 1;
};

Registry& registry() {
    static Registry instance;
    return instance;
}

std::string describe(const Op& op) {
    char text[96];
    std::snprintf(text, sizeof(text), "XAsyncBlock %p (operation %" PRIu64 "%s%s)", static_cast<void*>(op.block), op.id,
                  op.identity_name.empty() ? "" : ", ", op.identity_name.c_str());
    return text;
}

void diagnostic(const std::string& message) {
    os::log("xasync: " + message);
}

bool header_matches(const XAsyncBlock* block, const Op& op) {
    BlockHeader header{};
    std::memcpy(&header, block->internal, sizeof(header));
    return header.signature == kSignature && header.op_id == op.id;
}

void write_header(XAsyncBlock* block, uint64_t op_id) {
    BlockHeader header{};
    header.signature = kSignature;
    header.op_id = op_id;
    std::memcpy(block->internal, &header, sizeof(header));
}

OpRef find_op(XAsyncBlock* block) {
    if (block == nullptr) {
        return nullptr;
    }
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.ops.find(block);
    if (it == reg.ops.end() || !header_matches(block, *it->second)) {
        return nullptr;
    }
    return it->second;
}

// The queue an operation on `block` uses (the process queue for a null block->queue).
XTaskQueueHandle resolve_queue(const XAsyncBlock* block) {
    if (block->queue != nullptr) {
        return block->queue;
    }
    XTaskQueueHandle process = nullptr;
    if (!XTaskQueueGetCurrentProcessTaskQueue(&process) || process == nullptr) {
        return nullptr;
    }
    XTaskQueueCloseHandle(process);  // the runtime keeps the process queue alive
    return process;
}

// Runs provider(Cleanup) once the operation's provider is no longer needed.
void maybe_cleanup(const OpRef& op) {
    std::unique_ptr<XAsyncProviderImpl> provider;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (op->cleaned_up || !op->completed || op->inflight != 0) {
            return;
        }
        if (!(failed(op->status) || op->required == 0 || op->result_taken || op->abandoned)) {
            return;
        }
        op->cleaned_up = true;
        provider = std::move(op->provider);
    }
    if (provider) {
        provider->call(XAsyncOp::Cleanup, op->block, 0, 0);
    }
}

// Starts a provider call; nullptr if the operation's provider is gone.
XAsyncProviderImpl* begin_call_locked(Op& op) {
    if (!op.provider) {
        return nullptr;
    }
    ++op.inflight;
    return op.provider.get();
}

void end_call(const OpRef& op) {
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        --op->inflight;
    }
    maybe_cleanup(op);
}

struct Task {
    OpRef op;
};

HResult submit(const OpRef& op, XTaskQueuePort port, XTaskQueueCallback fn) {
    if (op->queue == nullptr) {
        return E_FAIL_HR;
    }
    auto* task = new Task{op};
    const HResult hr = XTaskQueueSubmitCallback(op->queue, port, task, fn);
    if (failed(hr)) {
        delete task;
    }
    return hr;
}

void finish_callback(const OpRef& op) {
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        op->callback_thread = std::thread::id{};
        op->callback_done = true;
    }
    op->cv.notify_all();
}

void completion_entry(void* context, bool canceled) {
    const std::unique_ptr<Task> task(static_cast<Task*>(context));
    const OpRef op = task->op;
    XAsyncCompletionRoutine callback = nullptr;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (canceled && !op->result_taken) {
            // Runtime convention (O5/O8): a completion the queue's termination cancelled reports
            // E_ABORT; the title is still notified.
            op->status = E_ABORT_HR;
            op->required = 0;
        }
        callback = op->callback;
        if (callback != nullptr) {
            op->callback_thread = std::this_thread::get_id();
        }
    }
    if (canceled) {
        maybe_cleanup(op);
    }
    if (callback != nullptr) {
        callback(op->block);
        finish_callback(op);
    }
}

// `explicit_call`: XAsyncComplete called by a provider (a second one is diagnosed).
void complete(const OpRef& op, HResult result, size_t required, bool explicit_call) {
    if (result == E_PENDING_HR) {
        diagnostic(describe(*op) + ": XAsyncComplete with E_PENDING ignored");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (op->completed) {
            if (explicit_call) {
                diagnostic(describe(*op) + ": completed twice, the second completion is ignored");
            }
            return;
        }
        op->completed = true;
        op->status = result;
        op->required = succeeded(result) ? required : 0;
        op->callback = op->block->callback;
        if (op->callback == nullptr) {
            op->callback_done = true;  // nothing to wait for
        }
    }
    op->cv.notify_all();
    maybe_cleanup(op);
    // Through the completion port even without a callback (as O5 did): the port's dispatcher
    // sees one completion per operation.
    const HResult hr = submit(op, XTaskQueuePort::Completion, &completion_entry);
    if (failed(hr)) {
        // The completion port refuses callbacks (terminated queue): no completion may be lost.
        completion_entry(new Task{op}, false);
    }
}

void work_entry(void* context, bool canceled) {
    const std::unique_ptr<Task> task(static_cast<Task*>(context));
    const OpRef op = task->op;
    if (canceled) {
        complete(op, E_ABORT_HR, 0, false);
        return;
    }
    XAsyncProviderImpl* provider = nullptr;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (op->completed) {
            return;
        }
        if (!op->canceled) {
            provider = begin_call_locked(*op);
        }
    }
    if (provider == nullptr) {
        complete(op, E_ABORT_HR, 0, false);
        return;
    }
    const HResult hr = provider->call(XAsyncOp::DoWork, op->block, 0, 0);
    end_call(op);
    bool canceled_meanwhile = false;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        canceled_meanwhile = op->canceled;
    }
    if (hr != E_PENDING_HR) {
        complete(op, hr, 0, false);
    } else if (canceled_meanwhile) {
        // Cancelled while this DoWork ran: XAsyncSchedule refuses more work, so "call me again"
        // can no longer happen.
        complete(op, E_ABORT_HR, 0, false);
    }
}

// Delayed XAsyncSchedule: one timer thread submits DoWork when due.
class DelayScheduler {
public:
    void schedule(const OpRef& op, uint32_t delay_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable()) {
            thread_ = std::thread([this] { run(); });
        }
        items_.emplace(std::chrono::steady_clock::now() + std::chrono::milliseconds(delay_ms), op);
        cv_.notify_all();
    }

    ~DelayScheduler() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
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
            OpRef op = std::move(it->second);
            items_.erase(it);
            lock.unlock();
            bool done = false;
            {
                std::lock_guard<std::mutex> op_lock(op->mutex);
                done = op->completed;
            }
            if (!done && failed(submit(op, XTaskQueuePort::Work, &work_entry))) {
                complete(op, E_ABORT_HR, 0, false);
            }
            lock.lock();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::multimap<std::chrono::steady_clock::time_point, OpRef> items_;
    std::thread thread_;
    bool stop_ = false;
};

DelayScheduler& delay_scheduler() {
    static DelayScheduler instance;
    return instance;
}

// Registers `op` for its block. Fails (nullptr old, false) if the block's operation is running.
bool register_op(const OpRef& op, OpRef* replaced) {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    op->id = reg.next_id++;
    auto it = reg.ops.find(op->block);
    if (it != reg.ops.end() && header_matches(op->block, *it->second)) {
        std::lock_guard<std::mutex> old_lock(it->second->mutex);
        if (!it->second->completed) {
            return false;
        }
        it->second->abandoned = true;
        *replaced = it->second;
    }
    reg.ops[op->block] = op;
    write_header(op->block, op->id);
    return true;
}

// Host providers ------------------------------------------------------------------------------

class CProvider final : public XAsyncProviderImpl {
public:
    CProvider(Pg6XAsyncProvider fn, XAsyncBlock* block, void* context) : fn_(fn) {
        data_.async = block;
        data_.context = context;
    }
    HResult call(XAsyncOp op, XAsyncBlock* /*block*/, size_t buffer_size, uint64_t buffer) override {
        if (op == XAsyncOp::GetResult) {
            data_.buffer_size = buffer_size;
            data_.buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(buffer));
        }
        return fn_(static_cast<uint32_t>(op), &data_);
    }

private:
    Pg6XAsyncProvider fn_;
    Pg6XAsyncProviderData data_{};
};

class RunProvider final : public XAsyncProviderImpl {
public:
    explicit RunProvider(std::function<HResult(XAsyncBlock*)> work) : work_(std::move(work)) {}
    HResult call(XAsyncOp op, XAsyncBlock* block, size_t, uint64_t) override {
        switch (op) {
            case XAsyncOp::Begin:
                return XAsyncSchedule(block, 0);
            case XAsyncOp::DoWork:
                return work_(block);
            case XAsyncOp::GetResult:
                return E_NOTIMPL_HR;  // XAsyncRun has no payload
            default:
                return S_OK_HR;
        }
    }

private:
    std::function<HResult(XAsyncBlock*)> work_;
};

const char kRunIdentity = 0;

}  // namespace

// ---------------------------------------------------------------------------
// Provider API
// ---------------------------------------------------------------------------

HResult XAsyncBeginWith(XAsyncBlock* async_block,
                        std::unique_ptr<XAsyncProviderImpl> provider,
                        uint64_t identity,
                        const char* identity_name) {
    if (async_block == nullptr || !provider) {
        return E_INVALIDARG_HR;
    }
    auto op = std::make_shared<Op>();
    op->block = async_block;
    op->queue = resolve_queue(async_block);
    if (op->queue == nullptr) {
        return E_FAIL_HR;
    }
    op->provider = std::move(provider);
    op->identity = identity;
    op->identity_name = identity_name != nullptr ? identity_name : "";
    OpRef replaced;
    if (!register_op(op, &replaced)) {
        diagnostic("XAsyncBegin on a block whose operation is still running");
        return E_INVALIDARG_HR;
    }
    if (replaced) {
        maybe_cleanup(replaced);  // its result was never fetched
    }
    XAsyncProviderImpl* p = nullptr;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        p = begin_call_locked(*op);
    }
    const HResult hr = p->call(XAsyncOp::Begin, async_block, 0, 0);
    if (failed(hr)) {
        {
            std::lock_guard<std::mutex> lock(op->mutex);
            if (!op->completed) {
                // A failed Begin ends the operation without a completion callback.
                op->completed = true;
                op->status = hr;
                op->required = 0;
                op->callback_done = true;
            }
        }
        op->cv.notify_all();
    }
    end_call(op);
    return failed(hr) ? hr : S_OK_HR;
}

HResult XAsyncBegin(XAsyncBlock* async_block,
                    void* context,
                    const void* identity,
                    const char* identity_name,
                    Pg6XAsyncProvider provider) {
    if (async_block == nullptr || provider == nullptr) {
        return E_INVALIDARG_HR;
    }
    return XAsyncBeginWith(async_block, std::make_unique<CProvider>(provider, async_block, context),
                           static_cast<uint64_t>(reinterpret_cast<uintptr_t>(identity)), identity_name);
}

HResult XAsyncSchedule(XAsyncBlock* async_block, uint32_t delay_ms) {
    const OpRef op = find_op(async_block);
    if (!op) {
        return E_INVALIDARG_HR;
    }
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (op->completed || op->canceled) {
            return E_ABORT_HR;
        }
    }
    if (delay_ms != 0) {
        delay_scheduler().schedule(op, delay_ms);
        return S_OK_HR;
    }
    return submit(op, XTaskQueuePort::Work, &work_entry);
}

void XAsyncComplete(XAsyncBlock* async_block, HResult result, size_t required_buffer_size) {
    const OpRef op = find_op(async_block);
    if (!op) {
        diagnostic("XAsyncComplete on a block without an operation");
        return;
    }
    complete(op, result, required_buffer_size, true);
}

HResult XAsyncGetResultWith(XAsyncBlock* async_block,
                            uint64_t identity,
                            size_t buffer_size,
                            uint64_t buffer,
                            size_t* buffer_used) {
    if (buffer_used != nullptr) {
        *buffer_used = 0;
    }
    const OpRef op = find_op(async_block);
    if (!op) {
        return E_INVALIDARG_HR;
    }
    XAsyncProviderImpl* provider = nullptr;
    size_t required = 0;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (!op->completed) {
            return E_PENDING_HR;
        }
        if (failed(op->status)) {
            return op->status;
        }
        if (identity != op->identity) {
            diagnostic(describe(*op) + ": XAsyncGetResult with a different identity than XAsyncBegin");
            return E_INVALIDARG_HR;
        }
        if (op->required == 0) {
            return S_OK_HR;
        }
        if (op->result_taken || op->cleaned_up) {
            return E_ILLEGAL_METHOD_CALL_HR;
        }
        if (buffer == 0) {
            return E_INVALIDARG_HR;
        }
        if (buffer_size < op->required) {
            return E_NOT_SUFFICIENT_BUFFER_HR;
        }
        provider = begin_call_locked(*op);
        if (provider == nullptr) {
            return E_ILLEGAL_METHOD_CALL_HR;
        }
        op->result_taken = true;
        required = op->required;
    }
    const HResult hr = provider->call(XAsyncOp::GetResult, async_block, required, buffer);
    end_call(op);
    if (succeeded(hr) && buffer_used != nullptr) {
        *buffer_used = required;
    }
    return hr;
}

HResult XAsyncGetResult(XAsyncBlock* async_block,
                        const void* identity,
                        size_t buffer_size,
                        void* buffer,
                        size_t* buffer_used) {
    return XAsyncGetResultWith(async_block, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(identity)),
                               buffer_size, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer)),
                               buffer_used);
}

void XAsyncCancel(XAsyncBlock* async_block) {
    const OpRef op = find_op(async_block);
    if (!op) {
        return;
    }
    XAsyncProviderImpl* provider = nullptr;
    {
        std::lock_guard<std::mutex> lock(op->mutex);
        if (op->completed || op->canceled) {
            return;
        }
        op->canceled = true;
        provider = begin_call_locked(*op);
    }
    HResult hr = E_NOTIMPL_HR;
    if (provider != nullptr) {
        hr = provider->call(XAsyncOp::Cancel, async_block, 0, 0);
        end_call(op);
    }
    if (failed(hr)) {
        complete(op, E_ABORT_HR, 0, false);
    }
}

HResult XAsyncRunWith(XAsyncBlock* async_block, std::function<HResult(XAsyncBlock*)> work) {
    if (async_block == nullptr || !work) {
        return E_INVALIDARG_HR;
    }
    return XAsyncBeginWith(async_block, std::make_unique<RunProvider>(std::move(work)),
                           static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&kRunIdentity)), "XAsyncRun");
}

HResult XAsyncRun(XAsyncBlock* async_block, Pg6XAsyncWork work) {
    if (work == nullptr) {
        return E_INVALIDARG_HR;
    }
    return XAsyncRunWith(async_block, [work](XAsyncBlock* block) { return work(block); });
}

HResult complete_async_immediately(XAsyncBlock* async_block, HResult status) {
    if (async_block == nullptr) {
        return E_INVALIDARG_HR;
    }
    auto op = std::make_shared<Op>();
    op->block = async_block;
    op->queue = resolve_queue(async_block);
    if (op->queue == nullptr) {
        return E_FAIL_HR;
    }
    op->completed = true;
    op->cleaned_up = true;  // no provider
    op->status = status;
    op->callback = async_block->callback;
    op->callback_done = op->callback == nullptr;
    uint8_t saved[sizeof(async_block->internal)];
    std::memcpy(saved, async_block->internal, sizeof(saved));
    OpRef replaced;
    if (!register_op(op, &replaced)) {
        return E_INVALIDARG_HR;
    }
    // Always through the completion port (even without a callback), so a terminated queue fails
    // the call up front as it did for O8's stubs.
    const HResult hr = submit(op, XTaskQueuePort::Completion, &completion_entry);
    if (failed(hr)) {
        Registry& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = reg.ops.find(async_block);
        if (it != reg.ops.end() && it->second == op) {
            if (replaced) {
                std::lock_guard<std::mutex> old_lock(replaced->mutex);
                replaced->abandoned = false;
                it->second = replaced;
            } else {
                reg.ops.erase(it);
            }
        }
        std::memcpy(async_block->internal, saved, sizeof(saved));
        return hr;
    }
    if (replaced) {
        maybe_cleanup(replaced);
    }
    return S_OK_HR;
}

void xasync_forget(XAsyncBlock* async_block) {
    OpRef op;
    {
        Registry& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = reg.ops.find(async_block);
        if (it == reg.ops.end()) {
            return;
        }
        {
            std::lock_guard<std::mutex> op_lock(it->second->mutex);
            if (!it->second->completed) {
                return;  // still running: the owner must keep the block
            }
            it->second->abandoned = true;
        }
        op = std::move(it->second);
        reg.ops.erase(it);
    }
    maybe_cleanup(op);
}

size_t xasync_record_count() {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return reg.ops.size();
}

bool xasync_in_flight(XAsyncBlock* async_block) {
    const OpRef op = find_op(async_block);
    if (!op) {
        return false;
    }
    std::lock_guard<std::mutex> lock(op->mutex);
    return !op->completed || !op->callback_done;
}

// ---------------------------------------------------------------------------
// XAsyncGetStatus / XAsyncGetResultSize (declared in os/xgameruntime.h)
// ---------------------------------------------------------------------------

HResult XAsyncGetStatus(XAsyncBlock* async_block, bool wait) {
    const OpRef op = find_op(async_block);
    if (!op) {
        return E_INVALIDARG_HR;
    }
    std::unique_lock<std::mutex> lock(op->mutex);
    if (wait && !op->callback_done) {
        if (op->completed && op->callback_thread == std::this_thread::get_id()) {
            return op->status;  // asked from inside the completion callback
        }
        op->cv.wait(lock, [&] { return op->callback_done; });
    }
    return op->completed ? op->status : E_PENDING_HR;
}

HResult XAsyncGetResultSize(XAsyncBlock* async_block, size_t* buffer_size) {
    if (buffer_size == nullptr) {
        return E_INVALIDARG_HR;
    }
    *buffer_size = 0;
    const OpRef op = find_op(async_block);
    if (!op) {
        return E_INVALIDARG_HR;
    }
    std::lock_guard<std::mutex> lock(op->mutex);
    if (!op->completed) {
        return E_PENDING_HR;
    }
    if (failed(op->status)) {
        return op->status;
    }
    *buffer_size = op->required;
    return S_OK_HR;
}

}  // namespace playgta6::os::gdk

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

namespace gdk = playgta6::os::gdk;

extern "C" {

Pg6HResult pg6_gdk_xasync_begin(Pg6XAsyncBlock* async_block,
                                void* context,
                                const void* identity,
                                const char* identity_name,
                                Pg6XAsyncProvider provider) {
    return gdk::XAsyncBegin(async_block, context, identity, identity_name, provider);
}

Pg6HResult pg6_gdk_xasync_schedule(Pg6XAsyncBlock* async_block, uint32_t delay_ms) {
    return gdk::XAsyncSchedule(async_block, delay_ms);
}

void pg6_gdk_xasync_complete(Pg6XAsyncBlock* async_block, Pg6HResult result, size_t required_buffer_size) {
    gdk::XAsyncComplete(async_block, result, required_buffer_size);
}

Pg6HResult pg6_gdk_xasync_get_result(Pg6XAsyncBlock* async_block,
                                     const void* identity,
                                     size_t buffer_size,
                                     void* buffer,
                                     size_t* buffer_used) {
    return gdk::XAsyncGetResult(async_block, identity, buffer_size, buffer, buffer_used);
}

void pg6_gdk_xasync_cancel(Pg6XAsyncBlock* async_block) {
    gdk::XAsyncCancel(async_block);
}

Pg6HResult pg6_gdk_xasync_run(Pg6XAsyncBlock* async_block, Pg6XAsyncWork work) {
    return gdk::XAsyncRun(async_block, work);
}

}  // extern "C"
