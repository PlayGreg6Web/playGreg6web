#include "os/httpfs_mailbox.h"

#include <algorithm>
#include <atomic>
#include <cstring>

namespace playgta6::os {

static_assert(alignof(HttpfsMailbox) >= std::atomic_ref<int32_t>::required_alignment,
              "HttpfsMailbox alignment must satisfy atomic_ref<int32_t>");

HttpfsMailboxClient::HttpfsMailboxClient(HttpfsMailbox* mailbox,
                                         ResponderCallback responder)
    : mailbox_(mailbox), responder_(std::move(responder)) {}

void HttpfsMailboxClient::set_responder(ResponderCallback responder) {
    std::lock_guard<std::mutex> lock(mutex_);
    responder_ = std::move(responder);
}

size_t HttpfsMailboxClient::effective_capacity_locked() const {
    if (mailbox_ == nullptr) {
        return 0;
    }
    if (mailbox_->payload_capacity == 0) {
        return kHttpfsPayloadBytes;
    }
    return std::min<size_t>(mailbox_->payload_capacity, kHttpfsPayloadBytes);
}

size_t HttpfsMailboxClient::payload_capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
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
    std::lock_guard<std::mutex> lock(mutex_);
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

    std::lock_guard<std::mutex> lock(mutex_);
    if (mailbox_ == nullptr) {
        return -1;
    }

    const size_t cap = effective_capacity_locked();
    if (cap == 0) {
        return -1;
    }

    uint8_t* out = static_cast<uint8_t*>(dst);
    uint64_t cur_offset = offset;
    size_t remaining = len;
    int64_t total_read = 0;

    while (remaining > 0) {
        const size_t chunk = std::min(remaining, cap);
        mailbox_->op = static_cast<uint32_t>(HttpfsOp::Read);
        mailbox_->handle = handle;
        mailbox_->offset = cur_offset;
        mailbox_->length = static_cast<uint64_t>(chunk);
        mailbox_->result = -1;

        int64_t chunk_res = -1;
        const HttpfsStatus status = execute_roundtrip_locked(chunk_res);

        if (status != HttpfsStatus::ResponseReady || chunk_res < 0 ||
            static_cast<uint64_t>(chunk_res) > chunk) {
            std::atomic_ref<int32_t> status_ref(mailbox_->status);
            if (status != HttpfsStatus::Shutdown) {
                status_ref.store(static_cast<int32_t>(HttpfsStatus::Idle),
                                 std::memory_order_release);
                status_ref.notify_one();
            }
            return chunk_res < 0 ? chunk_res : -1;
        }

        const size_t got = static_cast<size_t>(chunk_res);
        if (got > 0) {
            std::memcpy(out, mailbox_->payload, got);
        }

        std::atomic_ref<int32_t> status_ref(mailbox_->status);
        status_ref.store(static_cast<int32_t>(HttpfsStatus::Idle), std::memory_order_release);
        status_ref.notify_one();

        out += got;
        cur_offset += got;
        remaining -= got;
        total_read += static_cast<int64_t>(got);

        if (got < chunk) {
            // Reached EOF.
            break;
        }
    }

    return total_read;
}

bool HttpfsMailboxClient::close(int32_t handle) {
    if (handle < 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
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
