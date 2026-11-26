#include "os/httpfs_mailbox.h"
#include "os/vmem.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

namespace playgta6::os {

static_assert(alignof(HttpfsMailbox) >= std::atomic_ref<int32_t>::required_alignment,
              "HttpfsMailbox alignment must satisfy atomic_ref<int32_t>");

void HttpfsFairMutex::lock() {
    std::unique_lock<std::mutex> lock(mutex_);
    const uint64_t ticket = next_ticket_++;
    cv_.wait(lock, [&] { return now_serving_ == ticket; });
}

void HttpfsFairMutex::unlock() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++now_serving_;
    }
    cv_.notify_all();
}

HttpfsMailboxClient::HttpfsMailboxClient(HttpfsMailbox* mailbox,
                                         ResponderCallback responder)
    : mailbox_(mailbox), responder_(std::move(responder)) {}

HttpfsMailboxClient::HttpfsMailboxClient(HttpfsDirectBackend backend)
    : direct_(std::move(backend)), direct_mode_(true) {}

void HttpfsMailboxClient::set_responder(ResponderCallback responder) {
    std::lock_guard<HttpfsFairMutex> lock(mutex_);
    responder_ = std::move(responder);
}

void HttpfsMailboxClient::set_retry_policy(RetryPolicy policy) {
    std::lock_guard<HttpfsFairMutex> lock(mutex_);
    retry_ = std::move(policy);
}

size_t HttpfsMailboxClient::effective_capacity_locked() const {
    if (direct_mode_) {
        return kHttpfsPayloadBytes;
    }
    if (mailbox_ == nullptr) {
        return 0;
    }
    if (mailbox_->payload_capacity == 0) {
        return kHttpfsPayloadBytes;
    }
    return std::min<size_t>(mailbox_->payload_capacity, kHttpfsPayloadBytes);
}

size_t HttpfsMailboxClient::payload_capacity() const {
    std::lock_guard<HttpfsFairMutex> lock(mutex_);
    return effective_capacity_locked();
}

void HttpfsMailboxClient::init_mailbox(HttpfsMailbox& mailbox, uint32_t payload_capacity) {
    const uint32_t clamped = payload_capacity == 0
                                 ? static_cast<uint32_t>(kHttpfsPayloadBytes)
                                 : std::min<uint32_t>(
                                       payload_capacity,
                                       static_cast<uint32_t>(kHttpfsPayloadBytes));
    mailbox.op = static_cast<uint32_t>(HttpfsOp::None);
    mailbox.handle = -1;
    mailbox.payload_capacity = clamped;
    mailbox.offset = 0;
    mailbox.length = 0;
    mailbox.result = 0;
    std::atomic_ref<int32_t>(mailbox.status)
        .store(static_cast<int32_t>(HttpfsStatus::Idle), std::memory_order_release);
}

bool HttpfsMailboxClient::wait_for_request(HttpfsMailbox& mailbox) {
    std::atomic_ref<int32_t> status_ref(mailbox.status);
    int32_t cur = status_ref.load(std::memory_order_acquire);
    while (cur == static_cast<int32_t>(HttpfsStatus::Idle) ||
           cur == static_cast<int32_t>(HttpfsStatus::ResponseReady) ||
           cur == static_cast<int32_t>(HttpfsStatus::Error)) {
        status_ref.wait(cur, std::memory_order_acquire);
        cur = status_ref.load(std::memory_order_acquire);
    }
    return cur == static_cast<int32_t>(HttpfsStatus::RequestReady);
}

void HttpfsMailboxClient::complete_response(HttpfsMailbox& mailbox,
                                            int64_t result,
                                            HttpfsStatus status) {
    mailbox.result = result;
    std::atomic_ref<int32_t> status_ref(mailbox.status);
    status_ref.store(static_cast<int32_t>(status), std::memory_order_release);
    status_ref.notify_one();
}

void HttpfsMailboxClient::shutdown_waiter(HttpfsMailbox& mailbox) {
    std::atomic_ref<int32_t> status_ref(mailbox.status);
    status_ref.store(static_cast<int32_t>(HttpfsStatus::Shutdown), std::memory_order_release);
    status_ref.notify_all();
}

HttpfsStatus HttpfsMailboxClient::execute_roundtrip_locked(int64_t& out_result) {
    std::atomic_ref<int32_t> status_ref(mailbox_->status);
    status_ref.store(static_cast<int32_t>(HttpfsStatus::RequestReady), std::memory_order_release);
    status_ref.notify_one();

    if (responder_) {
        responder_(*mailbox_);
    }

    int32_t cur = status_ref.load(std::memory_order_acquire);
    while (cur == static_cast<int32_t>(HttpfsStatus::RequestReady)) {
        status_ref.wait(static_cast<int32_t>(HttpfsStatus::RequestReady),
                        std::memory_order_acquire);
        cur = status_ref.load(std::memory_order_acquire);
    }

    out_result = mailbox_->result;
    return static_cast<HttpfsStatus>(cur);
}

int32_t HttpfsMailboxClient::open(std::string_view rel_path) {
    std::lock_guard<HttpfsFairMutex> lock(mutex_);
    if (direct_mode_) {
        if (rel_path.empty() || !direct_.open) {
            return -1;
        }
        const int32_t handle = direct_.open(rel_path);
        return handle < 0 ? -1 : handle;
    }
    if (mailbox_ == nullptr || rel_path.empty()) {
        return -1;
    }

    const size_t cap = effective_capacity_locked();
    if (rel_path.size() > cap) {
        return -1;
    }

    mailbox_->op = static_cast<uint32_t>(HttpfsOp::Open);
    mailbox_->handle = -1;
    mailbox_->offset = 0;
    mailbox_->length = static_cast<uint64_t>(rel_path.size());
    mailbox_->result = -1;
    std::memcpy(mailbox_->payload, rel_path.data(), rel_path.size());

    int64_t result = -1;
    const HttpfsStatus status = execute_roundtrip_locked(result);

    std::atomic_ref<int32_t> status_ref(mailbox_->status);
    if (status != HttpfsStatus::Shutdown) {
        status_ref.store(static_cast<int32_t>(HttpfsStatus::Idle), std::memory_order_release);
        status_ref.notify_one();
    }

    if (status != HttpfsStatus::ResponseReady || result < 0) {
        return -1;
    }
    return static_cast<int32_t>(result);
}

int64_t HttpfsMailboxClient::read(int32_t handle, uint64_t offset, void* dst, size_t len) {
    if (handle < 0) {
        return -1;
    }
    if (len == 0) {
        return 0;
    }
    if (dst == nullptr) {
        return -1;
    }

    // R40: the direct backend (and a host responder) may write `dst` from the kernel.
    const HostWriteScope host_write(dst, len);
    uint8_t* out = static_cast<uint8_t*>(dst);
    uint64_t cur_offset = offset;
    size_t remaining = len;
    int64_t total_read = 0;

    while (remaining > 0) {
        // The lock is taken per chunk (FIFO), so other threads' requests interleave with a bulk
        // read instead of waiting for all of it.
        size_t asked = 0;
        bool retryable = false;
        int64_t got_or_error = read_chunk(handle, cur_offset, out, remaining, &asked, &retryable);
        if (got_or_error < 0 && retryable) {
            RetryPolicy policy;
            {
                std::lock_guard<HttpfsFairMutex> lock(mutex_);
                policy = retry_;
            }
            uint32_t backoff = policy.initial_backoff_ms;
            for (uint32_t attempt = 0; attempt < policy.max_retries && got_or_error < 0 && retryable;
                 ++attempt) {
                // Backoff outside the lock so other threads' requests are not stalled by it.
                if (policy.sleep) {
                    policy.sleep(backoff);
                } else if (backoff > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(backoff));
                }
                backoff = std::min<uint32_t>(backoff * 2, 1000);
                retried_chunks_.fetch_add(1, std::memory_order_relaxed);
                retryable = false;
                got_or_error = read_chunk(handle, cur_offset, out, remaining, &asked, &retryable);
            }
        }
        if (got_or_error < 0) {
            return got_or_error;  // surfaced to the title as a failed read, never zero-filled
        }
        const size_t got = static_cast<size_t>(got_or_error);

        out += got;
        cur_offset += got;
        remaining -= got;
        total_read += static_cast<int64_t>(got);

        if (got < asked) {
            // Reached EOF.
            break;
        }
    }

    return total_read;
}

int64_t HttpfsMailboxClient::read_chunk(int32_t handle,
                                        uint64_t offset,
                                        uint8_t* dst,
                                        size_t remaining,
                                        size_t* asked,
                                        bool* retryable) {
    std::lock_guard<HttpfsFairMutex> lock(mutex_);
    *retryable = false;
    const size_t cap = effective_capacity_locked();
    if (cap == 0) {
        return -1;
    }
    const size_t chunk = std::min(remaining, cap);
    *asked = chunk;

    if (direct_mode_) {
        if (!direct_.read) {
            return -1;
        }
        const int64_t n = direct_.read(handle, offset, dst, chunk);
        if (n < 0) {
            *retryable = true;
            return n;
        }
        return static_cast<uint64_t>(n) > chunk ? -1 : n;
    }

    mailbox_->op = static_cast<uint32_t>(HttpfsOp::Read);
    mailbox_->handle = handle;
    mailbox_->offset = offset;
    mailbox_->length = static_cast<uint64_t>(chunk);
    mailbox_->result = -1;

    int64_t chunk_res = -1;
    const HttpfsStatus status = execute_roundtrip_locked(chunk_res);

    if (status != HttpfsStatus::ResponseReady || chunk_res < 0 ||
        static_cast<uint64_t>(chunk_res) > chunk) {
        std::atomic_ref<int32_t> status_ref(mailbox_->status);
        if (status != HttpfsStatus::Shutdown) {
            status_ref.store(static_cast<int32_t>(HttpfsStatus::Idle), std::memory_order_release);
            status_ref.notify_one();
        }
        *retryable = status != HttpfsStatus::Shutdown && chunk_res < 0;
        return chunk_res < 0 ? chunk_res : -1;
    }

    const size_t got = static_cast<size_t>(chunk_res);
    if (got > 0) {
        std::memcpy(dst, mailbox_->payload, got);
    }

    std::atomic_ref<int32_t> status_ref(mailbox_->status);
    status_ref.store(static_cast<int32_t>(HttpfsStatus::Idle), std::memory_order_release);
    status_ref.notify_one();
    return chunk_res;
}

bool HttpfsMailboxClient::close(int32_t handle) {
    if (handle < 0) {
        return false;
    }

    std::lock_guard<HttpfsFairMutex> lock(mutex_);
    if (direct_mode_) {
        return direct_.close ? direct_.close(handle) : false;
    }
    if (mailbox_ == nullptr) {
        return false;
    }

    mailbox_->op = static_cast<uint32_t>(HttpfsOp::Close);
    mailbox_->handle = handle;
    mailbox_->offset = 0;
    mailbox_->length = 0;
    mailbox_->result = -1;

    int64_t result = -1;
    const HttpfsStatus status = execute_roundtrip_locked(result);

    std::atomic_ref<int32_t> status_ref(mailbox_->status);
    if (status != HttpfsStatus::Shutdown) {
        status_ref.store(static_cast<int32_t>(HttpfsStatus::Idle), std::memory_order_release);
        status_ref.notify_one();
    }

    return status == HttpfsStatus::ResponseReady && result >= 0;
}

}  // namespace playgta6::os
