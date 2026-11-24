#include "os/os_shim.h"

#include <atomic>
#include <memory>
#include <mutex>

#include "os/httpfs_mailbox.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>

extern "C" {
EM_IMPORT(host_log) void host_log(const char* ptr, size_t len);
EM_IMPORT(host_httpfs_open) int32_t host_httpfs_open(const char* ptr, size_t len);
EM_IMPORT(host_httpfs_read) int64_t host_httpfs_read(int32_t handle, uint64_t offset, void* dst, size_t len);
EM_IMPORT(host_httpfs_close) void host_httpfs_close(int32_t handle);
EM_IMPORT(host_wgpu_submit) void host_wgpu_submit(const uint8_t* ptr, size_t len);

EMSCRIPTEN_KEEPALIVE void playgta6_attach_httpfs_mailbox(playgta6::os::HttpfsMailbox* mailbox) {
    playgta6::os::httpfs_attach_mailbox(mailbox);
}
}
#else
#include <cstdio>
#endif

namespace playgta6::os {

namespace {

struct AttachedHttpfsState {
    std::mutex mutex;
    std::shared_ptr<HttpfsMailboxClient> owned_client;
    HttpfsMailboxClient* external_client = nullptr;
};

AttachedHttpfsState& httpfs_state() {
    static AttachedHttpfsState state;
    return state;
}

struct ActiveHttpfsLease {
    std::shared_ptr<HttpfsMailboxClient> owned;
    HttpfsMailboxClient* client = nullptr;
};

ActiveHttpfsLease acquire_httpfs_client() {
    auto& st = httpfs_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    if (st.external_client != nullptr) {
        return ActiveHttpfsLease{nullptr, st.external_client};
    }
    if (st.owned_client != nullptr) {
        return ActiveHttpfsLease{st.owned_client, st.owned_client.get()};
    }
    return ActiveHttpfsLease{};
}

}  // namespace

void log(std::string_view message) {
#ifdef __EMSCRIPTEN__
    host_log(message.data(), message.size());
#else
    std::fwrite(message.data(), 1, message.size(), stderr);
    std::fputc('\n', stderr);
#endif
}

void httpfs_attach_mailbox(HttpfsMailbox* mailbox) {
    auto& st = httpfs_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.external_client = nullptr;
    if (mailbox != nullptr) {
        st.owned_client = std::make_shared<HttpfsMailboxClient>(mailbox);
    } else {
        st.owned_client.reset();
    }
}

void httpfs_attach_client(HttpfsMailboxClient* client) {
    auto& st = httpfs_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.owned_client.reset();
    st.external_client = client;
}

bool httpfs_has_attached_mailbox() {
    auto& st = httpfs_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    return st.external_client != nullptr || st.owned_client != nullptr;
}

int32_t httpfs_open(std::string_view rel_path) {
    if (auto lease = acquire_httpfs_client(); lease.client != nullptr) {
        return lease.client->open(rel_path);
    }
#ifdef __EMSCRIPTEN__
    return host_httpfs_open(rel_path.data(), rel_path.size());
#else
    (void)rel_path;
    return -1;
#endif
}

int64_t httpfs_read(int32_t handle, uint64_t offset, void* dst, size_t len) {
    if (auto lease = acquire_httpfs_client(); lease.client != nullptr) {
        return lease.client->read(handle, offset, dst, len);
    }
#ifdef __EMSCRIPTEN__
    return host_httpfs_read(handle, offset, dst, len);
#else
    (void)handle;
    (void)offset;
    (void)dst;
    (void)len;
    return -1;
#endif
}

void httpfs_close(int32_t handle) {
    if (auto lease = acquire_httpfs_client(); lease.client != nullptr) {
        (void)lease.client->close(handle);
        return;
    }
#ifdef __EMSCRIPTEN__
    host_httpfs_close(handle);
#else
    (void)handle;
#endif
}

namespace {
std::atomic<WgpuSubmitObserver> g_wgpu_observer{nullptr};
}  // namespace

void set_wgpu_submit_observer(WgpuSubmitObserver observer) {
    g_wgpu_observer.store(observer, std::memory_order_release);
}

void wgpu_submit(const uint8_t* data, size_t len) {
    if (WgpuSubmitObserver observer = g_wgpu_observer.load(std::memory_order_acquire)) {
        observer(data, len);
    }
#ifdef __EMSCRIPTEN__
    host_wgpu_submit(data, len);
#else
    (void)data;
    (void)len;
#endif
}

}  // namespace playgta6::os
