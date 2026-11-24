#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "os/xasync_provider.h"
#include "os/xgameruntime.h"

// Host-level tests of the XAsync provider framework and the XTaskQueue port / composite APIs
// (os/xasync_provider.h, os/xgameruntime.h). Guest providers on the xgameruntime.dll provider are
// covered by tests/test_xgameruntime_provider.cpp.

namespace {

int g_failures = 0;
std::mutex g_check_mutex;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::lock_guard<std::mutex> lock(g_check_mutex);
        std::fprintf(stderr, "FAIL [test_xasync_provider.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os::gdk;

constexpr uint32_t kWork = 0;
constexpr uint32_t kCompletion = 1;

template <typename Pred>
bool wait_for(Pred pred) {
    for (int i = 0; i < 5000; ++i) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}

XTaskQueueHandle make_queue(uint32_t work_mode, uint32_t completion_mode) {
    XTaskQueueHandle queue = nullptr;
    CHECK(pg6_gdk_xtask_queue_create(work_mode, completion_mode, &queue) == S_OK_HR);
    return queue;
}

// A provider driven by a script, recording what it was asked (C-ABI provider, context = Script).
struct Script {
    std::mutex mutex;
    std::string ops;              // B W R C X (Begin, DoWork, GetResult, Cancel, Cleanup)
    int pending_rounds = 0;       // DoWork returns E_PENDING (after rescheduling) this many times
    HResult begin_result = S_OK_HR;
    HResult cancel_result = S_OK_HR;
    HResult work_result = S_OK_HR;
    uint64_t payload = 0x1122334455667788ULL;
    size_t payload_size = 8;
    std::atomic<int> cleanups{0};
    std::atomic<int> callbacks{0};
    std::atomic<bool> in_callback_status_ok{false};
    std::thread::id work_thread{};
    std::thread::id callback_thread{};
    Pg6XAsyncBlock* data_async = nullptr;
    void* data_context = nullptr;
    size_t result_buffer_size = 0;

    void record(char c) {
        std::lock_guard<std::mutex> lock(mutex);
        ops.push_back(c);
    }
    std::string seen() {
        std::lock_guard<std::mutex> lock(mutex);
        return ops;
    }
};

const char kIdentity = 0;
const char kOtherIdentity = 0;

HResult script_provider(uint32_t op, const Pg6XAsyncProviderData* data) {
    auto* s = static_cast<Script*>(data->context);
    switch (static_cast<XAsyncOp>(op)) {
        case XAsyncOp::Begin:
            s->data_async = data->async;
            s->data_context = data->context;
            s->record('B');
            if (failed(s->begin_result)) {
                return s->begin_result;
            }
            return pg6_gdk_xasync_schedule(data->async, 0);
        case XAsyncOp::DoWork:
            s->record('W');
            s->work_thread = std::this_thread::get_id();
            if (s->pending_rounds > 0) {
                --s->pending_rounds;
                const HResult rescheduled = pg6_gdk_xasync_schedule(data->async, 0);  // E_ABORT once cancelled
                return failed(rescheduled) ? rescheduled : E_PENDING_HR;
            }
            if (failed(s->work_result)) {
                return s->work_result;
            }
            pg6_gdk_xasync_complete(data->async, S_OK_HR, s->payload_size);
            return S_OK_HR;
        case XAsyncOp::GetResult:
            s->record('R');
            s->result_buffer_size = data->buffer_size;
            std::memcpy(data->buffer, &s->payload, s->payload_size);
            return S_OK_HR;
        case XAsyncOp::Cancel:
            s->record('X');
            return s->cancel_result;
        case XAsyncOp::Cleanup:
            s->record('C');
            s->cleanups.fetch_add(1);
            return S_OK_HR;
    }
    return E_FAIL_HR;
}

void script_callback(Pg6XAsyncBlock* block) {
    auto* s = static_cast<Script*>(block->context);
    s->callback_thread = std::this_thread::get_id();
    s->in_callback_status_ok = pg6_gdk_xasync_get_status(block, 1) != E_PENDING_HR;  // wait from the callback
    s->callbacks.fetch_add(1);
}

Pg6XAsyncBlock make_block(XTaskQueueHandle queue, Script* script) {
    Pg6XAsyncBlock block{};
    block.queue = queue;
    block.context = script;
    block.callback = &script_callback;
    return block;
}

void test_lifecycle_on_manual_queue() {
    XTaskQueueHandle queue = make_queue(0, 0);
    Script s;
    s.pending_rounds = 2;
    Pg6XAsyncBlock block = make_block(queue, &s);
    CHECK(pg6_gdk_xasync_begin(&block, &s, &kIdentity, "Scripted", &script_provider) == S_OK_HR);
    CHECK(s.data_async == &block && s.data_context == &s);
    CHECK(pg6_gdk_xasync_get_status(&block, 0) == E_PENDING_HR);
    size_t size = 99;
    CHECK(pg6_gdk_xasync_get_result_size(&block, &size) == E_PENDING_HR && size == 0);
    uint64_t out = 0;
    CHECK(pg6_gdk_xasync_get_result(&block, &kIdentity, 8, &out, nullptr) == E_PENDING_HR);
    // E_PENDING twice (rescheduled), then complete.
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xasync_get_status(&block, 0) == E_PENDING_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 0);
    CHECK(pg6_gdk_xasync_get_status(&block, 0) == S_OK_HR);  // visible before the callback
    CHECK(s.callbacks == 0);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 1);
    CHECK(s.callbacks == 1 && s.in_callback_status_ok);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 0);  // exactly once
    CHECK(pg6_gdk_xasync_get_result_size(&block, &size) == S_OK_HR && size == 8);
    // GetResult checks: identity, buffer, size; then the payload exactly once.
    size_t used = 77;
    CHECK(pg6_gdk_xasync_get_result(&block, &kOtherIdentity, 8, &out, &used) == E_INVALIDARG_HR && used == 0);
    CHECK(pg6_gdk_xasync_get_result(&block, &kIdentity, 8, nullptr, &used) == E_INVALIDARG_HR);
    CHECK(pg6_gdk_xasync_get_result(&block, &kIdentity, 4, &out, &used) == E_NOT_SUFFICIENT_BUFFER_HR);
    CHECK(s.cleanups == 0);
    CHECK(pg6_gdk_xasync_get_result(&block, &kIdentity, 16, &out, &used) == S_OK_HR);
    CHECK(out == s.payload && used == 8 && s.result_buffer_size == 8);
    CHECK(s.cleanups == 1);
    CHECK(pg6_gdk_xasync_get_result(&block, &kIdentity, 16, &out, &used) == E_ILLEGAL_METHOD_CALL_HR);
    CHECK(pg6_gdk_xasync_get_status(&block, 1) == S_OK_HR);
    CHECK(pg6_gdk_xasync_get_result_size(&block, &size) == S_OK_HR && size == 8);
    CHECK(s.seen() == "BWWWRC");
    // A second XAsyncComplete is ignored.
    pg6_gdk_xasync_complete(&block, E_FAIL_HR, 0);
    CHECK(pg6_gdk_xasync_get_status(&block, 0) == S_OK_HR);
    // The block can be reused; a cleared block is not mistaken for the old operation.
    Pg6XAsyncBlock fresh = make_block(queue, &s);
    CHECK(pg6_gdk_xasync_get_status(&fresh, 0) == E_INVALIDARG_HR);
    std::memset(block.internal, 0, sizeof(block.internal));
    CHECK(pg6_gdk_xasync_get_status(&block, 0) == E_INVALIDARG_HR);
    pg6_gdk_xtask_queue_close_handle(queue);
}

void test_begin_failure_and_work_failure() {
    XTaskQueueHandle queue = make_queue(0, 0);
    Script s;
    s.begin_result = E_OUTOFMEMORY_HR;
    Pg6XAsyncBlock block = make_block(queue, &s);
    CHECK(pg6_gdk_xasync_begin(&block, &s, &kIdentity, nullptr, &script_provider) == E_OUTOFMEMORY_HR);
    CHECK(pg6_gdk_xasync_get_status(&block, 1) == E_OUTOFMEMORY_HR);  // no callback to wait for
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 0);
    CHECK(s.callbacks == 0 && s.cleanups == 1 && s.seen() == "BC");

    // A failing DoWork completes the operation with its result; Cleanup follows at completion.
    Script w;
    w.work_result = static_cast<HResult>(0x80070005u);
    Pg6XAsyncBlock b2 = make_block(queue, &w);
    CHECK(pg6_gdk_xasync_begin(&b2, &w, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xasync_get_status(&b2, 0) == w.work_result);
    CHECK(w.cleanups == 1);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 1);
    uint64_t out = 0;
    CHECK(pg6_gdk_xasync_get_result(&b2, &kIdentity, 8, &out, nullptr) == w.work_result);
    // Beginning a block whose operation runs is refused.
    Script r;
    Pg6XAsyncBlock b3 = make_block(queue, &r);
    CHECK(pg6_gdk_xasync_begin(&b3, &r, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    CHECK(pg6_gdk_xasync_begin(&b3, &r, &kIdentity, nullptr, &script_provider) == E_INVALIDARG_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 1);
    // Reusing it after completion abandons the unfetched result: Cleanup runs.
    CHECK(r.cleanups == 0);
    CHECK(pg6_gdk_xasync_begin(&b3, &r, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    CHECK(r.cleanups == 1);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 1);
    playgta6::os::gdk::xasync_forget(&b3);  // drops the second result too
    CHECK(r.cleanups == 2);
    CHECK(pg6_gdk_xasync_get_status(&b3, 0) == E_INVALIDARG_HR);
    pg6_gdk_xtask_queue_close_handle(queue);
}

void test_cancel() {
    XTaskQueueHandle queue = make_queue(0, 0);
    // Before the work ran: the provider accepts; the queued DoWork completes with E_ABORT.
    Script a;
    Pg6XAsyncBlock b1 = make_block(queue, &a);
    CHECK(pg6_gdk_xasync_begin(&b1, &a, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    pg6_gdk_xasync_cancel(&b1);
    CHECK(pg6_gdk_xasync_get_status(&b1, 0) == E_PENDING_HR);
    CHECK(pg6_gdk_xasync_schedule(&b1, 0) == E_ABORT_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xasync_get_status(&b1, 0) == E_ABORT_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 1);
    CHECK(a.callbacks == 1 && a.seen() == "BXC");
    // A provider that cannot cancel (E_NOTIMPL): completed with E_ABORT at once.
    Script b;
    b.cancel_result = E_NOTIMPL_HR;
    Pg6XAsyncBlock b2 = make_block(queue, &b);
    CHECK(pg6_gdk_xasync_begin(&b2, &b, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    pg6_gdk_xasync_cancel(&b2);
    CHECK(pg6_gdk_xasync_get_status(&b2, 0) == E_ABORT_HR);
    pg6_gdk_xasync_cancel(&b2);  // after completion: nothing
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);  // the stale DoWork does nothing
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 1);
    CHECK(b.seen() == "BXC" && b.callbacks == 1);
    // After the work completed: no-op, the result stays.
    Script c;
    Pg6XAsyncBlock b3 = make_block(queue, &c);
    CHECK(pg6_gdk_xasync_begin(&b3, &c, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    pg6_gdk_xasync_cancel(&b3);
    CHECK(pg6_gdk_xasync_get_status(&b3, 0) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kCompletion, 0) == 1);
    uint64_t out = 0;
    CHECK(pg6_gdk_xasync_get_result(&b3, &kIdentity, 8, &out, nullptr) == S_OK_HR && out == c.payload);
    CHECK(c.seen() == "BWRC");
    // Queue termination cancels pending work: E_ABORT, the callback still runs once.
    Script d;
    Pg6XAsyncBlock b4 = make_block(queue, &d);
    CHECK(pg6_gdk_xasync_begin(&b4, &d, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_terminate(queue, 1, nullptr, nullptr) == S_OK_HR);
    CHECK(pg6_gdk_xasync_get_status(&b4, 1) == E_ABORT_HR);
    CHECK(d.callbacks == 1 && d.cleanups == 1);
    pg6_gdk_xtask_queue_close_handle(queue);
}

void test_wait_from_another_thread_and_threads() {
    XTaskQueueHandle queue = make_queue(1, 1);  // thread pool ports
    Script s;
    Pg6XAsyncBlock block = make_block(queue, &s);
    CHECK(pg6_gdk_xasync_begin(&block, &s, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    CHECK(pg6_gdk_xasync_get_status(&block, 1) == S_OK_HR);
    CHECK(s.callbacks == 1);  // wait returns after the completion callback
    CHECK(s.work_thread != std::this_thread::get_id());
    CHECK(s.callback_thread != std::this_thread::get_id());
    uint64_t out = 0;
    CHECK(pg6_gdk_xasync_get_result(&block, &kIdentity, 8, &out, nullptr) == S_OK_HR);

    // Null queue: the process task queue.
    Script p;
    Pg6XAsyncBlock pblock = make_block(nullptr, &p);
    CHECK(pg6_gdk_xasync_begin(&pblock, &p, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    CHECK(pg6_gdk_xasync_get_status(&pblock, 1) == S_OK_HR);
    CHECK(p.callbacks == 1);
    CHECK(pg6_gdk_xasync_get_result(&pblock, &kIdentity, 8, &out, nullptr) == S_OK_HR);

    // Delayed schedule.
    XTaskQueueHandle manual = make_queue(0, 0);
    Script dl;
    Pg6XAsyncBlock dblock = make_block(manual, &dl);
    CHECK(pg6_gdk_xasync_begin(&dblock, &dl, &kIdentity, nullptr,
                               [](uint32_t op, const Pg6XAsyncProviderData* data) -> HResult {
                                   if (op == static_cast<uint32_t>(XAsyncOp::Begin)) {
                                       return pg6_gdk_xasync_schedule(data->async, 40);
                                   }
                                   if (op == static_cast<uint32_t>(XAsyncOp::DoWork)) {
                                       return S_OK_HR;  // completes with S_OK, no payload
                                   }
                                   return S_OK_HR;
                               }) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(manual, kWork, 0) == 0);  // not due yet
    CHECK(pg6_gdk_xtask_queue_dispatch(manual, kWork, 2000) == 1);
    CHECK(pg6_gdk_xasync_get_status(&dblock, 0) == S_OK_HR);
    size_t used = 5;
    CHECK(pg6_gdk_xasync_get_result(&dblock, &kIdentity, 0, nullptr, &used) == S_OK_HR && used == 0);
    CHECK(pg6_gdk_xtask_queue_dispatch(manual, kCompletion, 0) == 1);
    pg6_gdk_xtask_queue_close_handle(manual);
    pg6_gdk_xtask_queue_close_handle(queue);
}

std::atomic<int> g_run_calls{0};
HResult run_work(Pg6XAsyncBlock* block) {
    g_run_calls.fetch_add(1);
    return block->context == nullptr ? S_OK_HR : static_cast<HResult>(0x8007000Du);
}

void test_run() {
    XTaskQueueHandle queue = make_queue(0, 0);
    Pg6XAsyncBlock ok{};
    ok.queue = queue;
    CHECK(pg6_gdk_xasync_run(&ok, &run_work) == S_OK_HR);
    CHECK(pg6_gdk_xasync_get_status(&ok, 0) == E_PENDING_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xasync_get_status(&ok, 1) == S_OK_HR);
    Pg6XAsyncBlock bad{};
    bad.queue = queue;
    bad.context = &bad;
    CHECK(pg6_gdk_xasync_run(&bad, &run_work) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
    CHECK(pg6_gdk_xasync_get_status(&bad, 0) == static_cast<HResult>(0x8007000Du));
    CHECK(g_run_calls == 2);
    CHECK(pg6_gdk_xasync_run(&bad, nullptr) == E_INVALIDARG_HR);
    CHECK(pg6_gdk_xasync_run(nullptr, &run_work) == E_INVALIDARG_HR);
    CHECK(pg6_gdk_xasync_begin(&bad, nullptr, nullptr, nullptr, nullptr) == E_INVALIDARG_HR);
    CHECK(pg6_gdk_xasync_schedule(nullptr, 0) == E_INVALIDARG_HR);
    pg6_gdk_xtask_queue_close_handle(queue);
}

// Entries still queued when the last handle closes (no terminate call) are cancelled by the
// queue's destruction: every task record is released and each pending operation still completes
// (E_ABORT) exactly once. Regression for the AddressSanitizer leaks at teardown.
void test_pending_entries_released_at_queue_destruction() {
    const size_t tasks_before = xasync_task_count();
    {
        XTaskQueueHandle queue = make_queue(0, 0);
        // Pending DoWork never dispatched.
        Script a;
        Pg6XAsyncBlock b1 = make_block(queue, &a);
        CHECK(pg6_gdk_xasync_begin(&b1, &a, &kIdentity, nullptr, &script_provider) == S_OK_HR);
        // DoWork dispatched, completion left pending.
        Script b;
        Pg6XAsyncBlock b2 = make_block(queue, &b);
        CHECK(pg6_gdk_xasync_begin(&b2, &b, &kIdentity, nullptr, &script_provider) == S_OK_HR);
        CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
        CHECK(pg6_gdk_xtask_queue_dispatch(queue, kWork, 0) == 1);
        Script c;
        Pg6XAsyncBlock b3 = make_block(queue, &c);
        CHECK(pg6_gdk_xasync_begin(&b3, &c, &kIdentity, nullptr, &script_provider) == S_OK_HR);
        CHECK(xasync_task_count() > tasks_before);
        pg6_gdk_xtask_queue_close_handle(queue);
        CHECK(a.callbacks == 1 && b.callbacks == 1 && c.callbacks == 1);
        CHECK(pg6_gdk_xasync_get_status(&b3, 0) == E_ABORT_HR);
    }
    CHECK(xasync_task_count() == tasks_before);
}

void test_ports_and_composite() {
    XTaskQueueHandle a = make_queue(0, 0);
    XTaskQueueHandle b = make_queue(0, 0);
    Pg6XTaskQueuePortHandle a_work = nullptr;
    Pg6XTaskQueuePortHandle b_completion = nullptr;
    CHECK(pg6_gdk_xtask_queue_get_port(a, kWork, &a_work) == S_OK_HR && a_work != nullptr);
    CHECK(pg6_gdk_xtask_queue_get_port(b, kCompletion, &b_completion) == S_OK_HR && b_completion != nullptr);
    Pg6XTaskQueuePortHandle again = nullptr;
    CHECK(pg6_gdk_xtask_queue_get_port(a, kWork, &again) == S_OK_HR && again == a_work);
    CHECK(pg6_gdk_xtask_queue_get_port(a, 7, &again) == E_INVALIDARG_HR && again == nullptr);
    CHECK(pg6_gdk_xtask_queue_get_port(a, kWork, nullptr) == E_POINTER_HR);

    XTaskQueueHandle composite = nullptr;
    CHECK(pg6_gdk_xtask_queue_create_composite(a_work, nullptr, &composite) == E_INVALIDARG_HR);
    CHECK(pg6_gdk_xtask_queue_create_composite(a_work, b_completion, &composite) == S_OK_HR);
    // The composite keeps its ports' queues alive.
    pg6_gdk_xtask_queue_close_handle(a);
    pg6_gdk_xtask_queue_close_handle(b);
    Pg6XTaskQueuePortHandle cw = nullptr;
    CHECK(pg6_gdk_xtask_queue_get_port(composite, kWork, &cw) == S_OK_HR && cw == a_work);

    // An XAsync operation on the composite: DoWork on a's work port, completion on b's.
    Script s;
    Pg6XAsyncBlock block = make_block(composite, &s);
    CHECK(pg6_gdk_xasync_begin(&block, &s, &kIdentity, nullptr, &script_provider) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(composite, kCompletion, 0) == 0);
    CHECK(pg6_gdk_xtask_queue_dispatch(composite, kWork, 0) == 1);
    CHECK(s.callbacks == 0);
    CHECK(pg6_gdk_xtask_queue_dispatch(composite, kCompletion, 0) == 1);
    CHECK(s.callbacks == 1);
    uint64_t out = 0;
    CHECK(pg6_gdk_xasync_get_result(&block, &kIdentity, 8, &out, nullptr) == S_OK_HR);

    // A composite of a composite's ports, plain submissions, and termination of the composite.
    XTaskQueueHandle nested = nullptr;
    Pg6XTaskQueuePortHandle cc = nullptr;
    CHECK(pg6_gdk_xtask_queue_get_port(composite, kCompletion, &cc) == S_OK_HR && cc == b_completion);
    CHECK(pg6_gdk_xtask_queue_create_composite(cw, cc, &nested) == S_OK_HR);
    static std::atomic<int> hits{0};
    auto cb = [](void* ctx, bool canceled) {
        if (!canceled) {
            hits.fetch_add(*static_cast<int*>(ctx));
        }
    };
    int one = 1;
    int ten = 10;
    CHECK(pg6_gdk_xtask_queue_submit_callback(nested, kWork, &one, cb) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_submit_callback(composite, kCompletion, &ten, cb) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(composite, kWork, 0) == 1);       // the nested one's work
    CHECK(pg6_gdk_xtask_queue_dispatch(nested, kCompletion, 0) == 1);    // the composite's completion
    CHECK(hits == 11);
    std::atomic<int> terminated{0};
    CHECK(pg6_gdk_xtask_queue_terminate(nested, 1, &terminated,
                                        [](void* ctx) { static_cast<std::atomic<int>*>(ctx)->store(1); }) ==
          S_OK_HR);
    CHECK(terminated == 1);
    CHECK(pg6_gdk_xtask_queue_submit_callback(nested, kWork, &one, cb) == E_ABORT_HR);
    CHECK(pg6_gdk_xtask_queue_submit_callback(composite, kWork, &one, cb) == S_OK_HR);  // ports still run
    CHECK(pg6_gdk_xtask_queue_dispatch(composite, kWork, 0) == 1);
    CHECK(hits == 12);
    pg6_gdk_xtask_queue_close_handle(nested);
    pg6_gdk_xtask_queue_close_handle(composite);
}

// Many operations from several threads on thread-pool queues: every completion callback and
// every Cleanup exactly once, none lost; half of the operations race a cancel.
void test_stress() {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 150;
    XTaskQueueHandle q1 = make_queue(1, 1);
    XTaskQueueHandle q2 = make_queue(2, 1);
    struct Slot {
        Script script;
        Pg6XAsyncBlock block{};
    };
    std::vector<std::unique_ptr<Slot>> slots;
    for (int i = 0; i < kThreads * kPerThread; ++i) {
        slots.push_back(std::make_unique<Slot>());
    }
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int k = 0; k < kPerThread; ++k) {
                Slot& slot = *slots[static_cast<size_t>(t * kPerThread + k)];
                slot.script.pending_rounds = k % 3;
                slot.block = make_block(k % 2 ? q1 : q2, &slot.script);
                CHECK(pg6_gdk_xasync_begin(&slot.block, &slot.script, &kIdentity, nullptr, &script_provider) ==
                      S_OK_HR);
                if (k % 2 == 0) {
                    pg6_gdk_xasync_cancel(&slot.block);
                }
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
    int ok = 0;
    int aborted = 0;
    for (auto& slot : slots) {
        const HResult status = pg6_gdk_xasync_get_status(&slot->block, 1);
        CHECK(status == S_OK_HR || status == E_ABORT_HR);
        CHECK(slot->script.callbacks == 1);
        if (status == S_OK_HR) {
            ++ok;
            uint64_t out = 0;
            CHECK(pg6_gdk_xasync_get_result(&slot->block, &kIdentity, 8, &out, nullptr) == S_OK_HR);
            CHECK(out == slot->script.payload);
        } else {
            ++aborted;
        }
    }
    for (auto& slot : slots) {
        CHECK(wait_for([&] { return slot->script.cleanups.load() == 1; }));
        playgta6::os::gdk::xasync_forget(&slot->block);
    }
    CHECK(ok + aborted == kThreads * kPerThread);
    CHECK(ok >= kThreads * kPerThread / 2);  // every odd-numbered operation was not cancelled
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (auto& slot : slots) {
        CHECK(slot->script.callbacks == 1 && slot->script.cleanups == 1);  // nothing duplicated
    }
    pg6_gdk_xtask_queue_terminate(q1, 1, nullptr, nullptr);
    pg6_gdk_xtask_queue_terminate(q2, 1, nullptr, nullptr);
    pg6_gdk_xtask_queue_close_handle(q1);
    pg6_gdk_xtask_queue_close_handle(q2);
}

}  // namespace

int main() {
    const struct {
        const char* name;
        void (*fn)();
    } tests[] = {
        {"lifecycle_on_manual_queue", test_lifecycle_on_manual_queue},
        {"begin_failure_and_work_failure", test_begin_failure_and_work_failure},
        {"cancel", test_cancel},
        {"wait_from_another_thread_and_threads", test_wait_from_another_thread_and_threads},
        {"run", test_run},
        {"pending_entries_released_at_queue_destruction", test_pending_entries_released_at_queue_destruction},
        {"ports_and_composite", test_ports_and_composite},
        {"stress", test_stress},
    };
    for (const auto& t : tests) {
        std::fprintf(stderr, "test_xasync_provider: %s\n", t.name);
        t.fn();
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "test_xasync_provider: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_xasync_provider: all tests passed\n");
    return 0;
}
