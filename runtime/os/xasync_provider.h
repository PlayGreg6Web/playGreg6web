#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "os/xgameruntime.h"

// XAsync provider framework (O22): the GDK async-provider API (XAsyncBegin / XAsyncSchedule /
// XAsyncComplete / XAsyncGetResult / XAsyncCancel / XAsyncRun, with XAsyncGetStatus and
// XAsyncGetResultSize from os/xgameruntime.h) that libraries use to *implement* asynchronous
// operations, and the one completion path of every runtime shim that completes an XAsyncBlock.
//
// State machine (per operation)
// -----------------------------
//   XAsyncBegin        provider(Begin) on the calling thread. A failing Begin ends the
//                      operation with that status (no completion callback; Cleanup follows) and
//                      XAsyncBegin returns it. Providers usually call XAsyncSchedule here.
//   XAsyncSchedule     queues one DoWork on the work port of the block's queue (the process task
//                      queue when block->queue is null), after `delay_ms` if non-zero.
//   DoWork             provider(DoWork) on the thread that dispatches the work port. A result
//                      other than E_PENDING completes the operation with it (requiredBufferSize
//                      0); E_PENDING means "not done": the provider calls XAsyncSchedule again to
//                      be called again, or XAsyncComplete later from anywhere. A DoWork that runs
//                      after XAsyncCancel, or is cancelled by its queue's termination, completes
//                      the operation with E_ABORT without calling the provider.
//   XAsyncComplete     the first completion wins (later ones and E_PENDING are ignored, with a
//                      diagnostic). The status is visible at once; a successful operation keeps
//                      requiredBufferSize for XAsyncGetResultSize / GetResult. block->callback runs
//                      once on the completion port (a port that refuses the submission, e.g. a
//                      terminated queue, runs it on the completing thread; a completion
//                      callback cancelled by queue termination still runs, and the operation's
//                      status becomes E_ABORT first, as O5 and O8 did). XAsyncGetStatus(wait)
//                      returns once that callback has returned (or at completion without one).
//   XAsyncGetResult    E_PENDING while running; the failure status for a failed operation;
//                      E_INVALIDARG for a different identity than XAsyncBegin's; S_OK with
//                      *bufferUsed = 0 when there is no payload; E_INVALIDARG for a null buffer and
//                      E_NOT_SUFFICIENT_BUFFER for one smaller than requiredBufferSize (the result
//                      stays available); otherwise provider(GetResult) with data->bufferSize =
//                      requiredBufferSize, *bufferUsed = requiredBufferSize, and the payload is
//                      consumed: a second GetResult returns E_ILLEGAL_METHOD_CALL.
//   XAsyncCancel       no-op once completed; otherwise marks the operation cancelled and calls
//                      provider(Cancel) on the calling thread; a failing Cancel (e.g. E_NOTIMPL)
//                      completes the operation with E_ABORT.
//   Cleanup            provider(Cleanup) exactly once, after the operation completed and its
//                      payload is no longer needed (failure, no payload, consumed, or abandoned
//                      by reusing the block), and after every provider call in flight returned;
//                      it runs on the thread that made that last transition.
// Provider calls never run under a framework lock, so a provider may call back into any XAsync
// function. Blocks: a block whose operation has not completed cannot be begun again
// (E_INVALIDARG); beginning a completed block again abandons its old result.
//
// Block association
// -----------------
// Host-internal state is keyed by the block's address; XAsyncBlock::internal holds a signature
// and the operation id, written once by XAsyncBegin, so a block that was copied or cleared
// afterwards (titles may only do that before XAsyncBegin) is not mistaken for the original.
// The record stays after the operation finished (status and result size remain readable) until
// the block is begun again or xasync_forget() drops it. Guest XAsyncBlocks (guest memory, guest
// callbacks) reach the framework through O16's shadow blocks (guest/guest_callbacks.h): the
// shadow is a host XAsyncBlock whose callback calls the guest callback with the guest block
// address; the xgameruntime.dll provider (os/xgameruntime_provider.cpp) adapts guest provider
// functions to XAsyncProviderImpl.

struct Pg6XAsyncProviderData {
    Pg6XAsyncBlock* async;
    size_t buffer_size;
    void* buffer;
    void* context;
};

// GDK XAsyncProvider / XAsyncWork shapes for host callers (op = XAsyncOp).
using Pg6XAsyncProvider = Pg6HResult (*)(uint32_t op, const Pg6XAsyncProviderData* data);
using Pg6XAsyncWork = Pg6HResult (*)(Pg6XAsyncBlock* async_block);

namespace playgta6::os::gdk {

inline constexpr HResult E_ILLEGAL_METHOD_CALL_HR = static_cast<HResult>(0x8000000Eu);
inline constexpr HResult E_NOTIMPL_HR = static_cast<HResult>(0x80004001u);

enum class XAsyncOp : uint32_t {
    Begin = 0,
    DoWork = 1,
    GetResult = 2,
    Cancel = 3,
    Cleanup = 4,
};

// The provider of one operation. `buffer` is meaningful for GetResult only and is an address in
// the provider's own address space: a host pointer for host providers, a guest address for the
// guest adapter.
class XAsyncProviderImpl {
public:
    virtual ~XAsyncProviderImpl() = default;
    virtual HResult call(XAsyncOp op, XAsyncBlock* block, size_t buffer_size, uint64_t buffer) = 0;
};

HResult XAsyncBeginWith(XAsyncBlock* async_block,
                        std::unique_ptr<XAsyncProviderImpl> provider,
                        uint64_t identity,
                        const char* identity_name);
HResult XAsyncGetResultWith(XAsyncBlock* async_block,
                            uint64_t identity,
                            size_t buffer_size,
                            uint64_t buffer,
                            size_t* buffer_used);
HResult XAsyncRunWith(XAsyncBlock* async_block, std::function<HResult(XAsyncBlock*)> work);

HResult XAsyncBegin(XAsyncBlock* async_block,
                    void* context,
                    const void* identity,
                    const char* identity_name,
                    Pg6XAsyncProvider provider);
HResult XAsyncSchedule(XAsyncBlock* async_block, uint32_t delay_ms);
void XAsyncComplete(XAsyncBlock* async_block, HResult result, size_t required_buffer_size);
HResult XAsyncGetResult(XAsyncBlock* async_block,
                        const void* identity,
                        size_t buffer_size,
                        void* buffer,
                        size_t* buffer_used);
void XAsyncCancel(XAsyncBlock* async_block);
HResult XAsyncRun(XAsyncBlock* async_block, Pg6XAsyncWork work);

// For shims whose result is known when they are called (offline stubs): starts an operation on
// `block` that is already complete with `status` (no payload) and delivers the completion
// callback like XAsyncComplete. If the completion callback cannot be queued (terminated queue)
// the block is left untouched, the callback never runs and the submission error is returned.
HResult complete_async_immediately(XAsyncBlock* async_block, HResult status);

// Drops the record of `block` (abandoning a result nobody fetched; a running operation is kept).
// For owners of host blocks that are freed, e.g. O16's shadows.
void xasync_forget(XAsyncBlock* async_block);
// True while `block`'s operation runs or its completion callback has not returned yet: the
// block's owner must keep it.
bool xasync_in_flight(XAsyncBlock* async_block);
// Operation records held (diagnostics, leak tests).
size_t xasync_record_count();

}  // namespace playgta6::os::gdk

extern "C" {

Pg6HResult pg6_gdk_xasync_begin(Pg6XAsyncBlock* async_block,
                                void* context,
                                const void* identity,
                                const char* identity_name,
                                Pg6XAsyncProvider provider);
Pg6HResult pg6_gdk_xasync_schedule(Pg6XAsyncBlock* async_block, uint32_t delay_ms);
void pg6_gdk_xasync_complete(Pg6XAsyncBlock* async_block, Pg6HResult result, size_t required_buffer_size);
Pg6HResult pg6_gdk_xasync_get_result(Pg6XAsyncBlock* async_block,
                                     const void* identity,
                                     size_t buffer_size,
                                     void* buffer,
                                     size_t* buffer_used);
void pg6_gdk_xasync_cancel(Pg6XAsyncBlock* async_block);
Pg6HResult pg6_gdk_xasync_run(Pg6XAsyncBlock* async_block, Pg6XAsyncWork work);

}  // extern "C"
