#include "os/xbox_offline.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <unordered_map>

#include "os/win32_kernel.h"
#include "os/xasync_provider.h"

namespace playgta6::os::offline {

namespace {

using gdk::E_ABORT_HR;
using gdk::E_FAIL_HR;
using gdk::E_INVALIDARG_HR;
using gdk::S_OK_HR;
using gdk::failed;
using gdk::succeeded;

// ---------------------------------------------------------------------------
// Per-import hit counters (import name -> call count)
// ---------------------------------------------------------------------------

struct HitCounters {
    std::mutex mutex;
    // Keys are string literals (static storage) passed by the stubs below.
    std::unordered_map<std::string_view, uint64_t> counts;
};

HitCounters& hit_counters() {
    static HitCounters counters;
    return counters;
}

std::atomic<uint64_t>& dropped_events() {
    static std::atomic<uint64_t> count{0};
    return count;
}

void hit(const char* import_name) {
    auto& counters = hit_counters();
    std::lock_guard<std::mutex> lock(counters.mutex);
    ++counters.counts[std::string_view(import_name)];
}

// ---------------------------------------------------------------------------
// Dummy handle table: distinct, ref-counted, never-dereferenced handle values
// ---------------------------------------------------------------------------

enum class HandleKind : uint32_t {
    WinHttpSession,
    WinHttpConnection,
    WinHttpRequest,
    XblContext,
    XStoreContext,
    EtwProvider,
    ConnectivityRegistration,
    LicenseRegistration,
};

class DummyHandleTable {
public:
    uint64_t create(HandleKind kind, uint64_t payload = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t value = next_;
        next_ += kStep;
        entries_.emplace(value, Entry{kind, 1u, payload});
        return value;
    }

    bool lookup(uint64_t value, HandleKind kind, uint64_t* payload = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(value);
        if (it == entries_.end() || it->second.kind != kind) {
            return false;
        }
        if (payload != nullptr) {
            *payload = it->second.payload;
        }
        return true;
    }

    bool lookup_any(uint64_t value, std::initializer_list<HandleKind> kinds) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(value);
        if (it == entries_.end()) {
            return false;
        }
        return std::find(kinds.begin(), kinds.end(), it->second.kind) != kinds.end();
    }

    bool retain(uint64_t value, HandleKind kind) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(value);
        if (it == entries_.end() || it->second.kind != kind) {
            return false;
        }
        ++it->second.refs;
        return true;
    }

    bool release(uint64_t value, std::initializer_list<HandleKind> kinds) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(value);
        if (it == entries_.end() ||
            std::find(kinds.begin(), kinds.end(), it->second.kind) == kinds.end()) {
            return false;
        }
        if (--it->second.refs == 0) {
            entries_.erase(it);
        }
        return true;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

private:
    // Small, aligned, non-null values that fit any pointer width; never NULL or INVALID_*.
    static constexpr uint64_t kBase = 0x0FF10010u;
    static constexpr uint64_t kStep = 0x10u;

    struct Entry {
        HandleKind kind;
        uint32_t refs;
        uint64_t payload;
    };

    std::mutex mutex_;
    uint64_t next_ = kBase;
    std::unordered_map<uint64_t, Entry> entries_;
};

DummyHandleTable& handles() {
    static DummyHandleTable table;
    return table;
}

template <typename T>
T* to_handle_ptr(uint64_t value) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(value));
}

template <typename T>
uint64_t from_handle_ptr(T* ptr) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

// ---------------------------------------------------------------------------
// XAsyncBlock completion through O5's XTaskQueue + XAsyncGetStatus machinery
// ---------------------------------------------------------------------------

// Completes `block` with `status` through the XAsync provider framework (os/xasync_provider.h):
// the status is visible immediately to XAsyncGetStatus and the title callback runs exactly once
// on the block's queue (or the process queue) completion port. On failure the block is left
// untouched and the callback never runs.
HResult complete_async(Pg6XAsyncBlock* block, HResult status) {
    return gdk::complete_async_immediately(block, status);
}

// Status for an XxxResult getter: the cached offline/abort status of a block completed above,
// E_PENDING, or E_INVALIDARG for a block that was not started by an offline network stub.
HResult async_result_status(Pg6XAsyncBlock* block) {
    if (block == nullptr) {
        return E_INVALIDARG_HR;
    }
    const HResult status = pg6_gdk_xasync_get_status(block, 0);
    return succeeded(status) ? E_INVALIDARG_HR : status;
}

// ---------------------------------------------------------------------------
// Stub category helpers (one per category so each new import is a one-line body)
// ---------------------------------------------------------------------------

// Live / Store service call: validate context + block, then complete with kOfflineHResult.
HResult offline_service_async(const char* import_name,
                              HandleKind context_kind,
                              const void* context,
                              Pg6XAsyncBlock* block) {
    hit(import_name);
    if (block == nullptr || !handles().lookup(from_handle_ptr(context), context_kind)) {
        return E_INVALIDARG_HR;
    }
    return complete_async(block, kOfflineHResult);
}

HResult xbl_async(const char* import_name, Pg6XblContextHandle context, Pg6XAsyncBlock* block) {
    return offline_service_async(import_name, HandleKind::XblContext, context, block);
}

HResult xstore_async(const char* import_name,
                     Pg6XStoreContextHandle context,
                     Pg6XAsyncBlock* block) {
    return offline_service_async(import_name, HandleKind::XStoreContext, context, block);
}

// XxxResult getter with a handle-typed out-parameter (always NULL offline).
HResult handle_result(const char* import_name,
                      Pg6XAsyncBlock* block,
                      Pg6OfflineResultHandle* out) {
    hit(import_name);
    if (out == nullptr) {
        return E_INVALIDARG_HR;
    }
    *out = nullptr;
    return async_result_status(block);
}

// XxxResultSize / XxxResultCount getter (always 0 offline).
template <typename T>
HResult count_result(const char* import_name, Pg6XAsyncBlock* block, T* out) {
    hit(import_name);
    if (out == nullptr) {
        return E_INVALIDARG_HR;
    }
    *out = 0;
    return async_result_status(block);
}

// XxxResult getter whose out-parameter is a GDK struct this shim does not model; the struct is
// left untouched because the result is always a failure.
HResult struct_result(const char* import_name, Pg6XAsyncBlock* block, void* out) {
    hit(import_name);
    if (out == nullptr) {
        return E_INVALIDARG_HR;
    }
    return async_result_status(block);
}

// WinSock: WSAStartup reference count and the shared Win32 last-error slot.
std::mutex& wsa_mutex() {
    static std::mutex mutex;
    return mutex;
}

uint32_t& wsa_startup_count() {
    static uint32_t count = 0;
    return count;
}

bool wsa_started() {
    std::lock_guard<std::mutex> lock(wsa_mutex());
    return wsa_startup_count() > 0;
}

void set_wsa_error(int32_t error) {
    win32::SetLastError(static_cast<uint32_t>(error));
}

// Any call on a SOCKET: no socket can exist offline.
int32_t socket_call(const char* import_name) {
    hit(import_name);
    set_wsa_error(wsa_started() ? WSAENOTSOCK : WSANOTINITIALISED);
    return SOCKET_ERROR_VALUE;
}

Pg6Socket socket_open(const char* import_name) {
    hit(import_name);
    set_wsa_error(wsa_started() ? WSAENETDOWN : WSANOTINITIALISED);
    return INVALID_SOCKET_VALUE;
}

int32_t resolve_name(const char* import_name, Pg6AddrInfo** result) {
    hit(import_name);
    if (result != nullptr) {
        *result = nullptr;
    }
    int32_t error = WSAHOST_NOT_FOUND;
    if (!wsa_started()) {
        error = WSANOTINITIALISED;
    } else if (result == nullptr) {
        error = WSAEINVAL;
    }
    set_wsa_error(error);
    return error;
}

// WinHTTP: BOOL-returning calls fail with `error`, or ERROR_INVALID_HANDLE for a bad handle.
int32_t winhttp_fail(const char* import_name,
                     Pg6HInternet handle,
                     std::initializer_list<HandleKind> kinds,
                     uint32_t error) {
    hit(import_name);
    win32::SetLastError(handles().lookup_any(handle, kinds) ? error : win32::ERROR_INVALID_HANDLE);
    return 0;
}

// WinHTTP: BOOL-returning local configuration calls succeed on any live WinHTTP handle.
int32_t winhttp_configure(const char* import_name, Pg6HInternet handle, bool allow_null_handle) {
    hit(import_name);
    if ((handle == 0 && allow_null_handle) ||
        handles().lookup_any(handle,
                             {HandleKind::WinHttpSession,
                              HandleKind::WinHttpConnection,
                              HandleKind::WinHttpRequest})) {
        win32::SetLastError(win32::ERROR_SUCCESS);
        return 1;
    }
    win32::SetLastError(win32::ERROR_INVALID_HANDLE);
    return 0;
}

Pg6HInternet winhttp_child(const char* import_name,
                           Pg6HInternet parent,
                           HandleKind parent_kind,
                           HandleKind child_kind,
                           bool arguments_valid) {
    hit(import_name);
    if (!handles().lookup(parent, parent_kind)) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    if (!arguments_valid) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return handles().create(child_kind);
}

// WinHttpCrackUrl helpers. A requested component either has a caller buffer (copied, NUL
// terminated) or a NULL pointer with a non-zero length (pointed into the URL string); a component
// with zero length and NULL pointer is not requested.
bool crack_component(const char16_t* src,
                     size_t count,
                     char16_t** pointer,
                     uint32_t* length,
                     bool* insufficient) {
    if (*length == 0) {
        return true;
    }
    if (*pointer == nullptr) {
        *pointer = const_cast<char16_t*>(src);
        *length = static_cast<uint32_t>(count);
        return true;
    }
    if (count + 1 > *length) {
        *length = static_cast<uint32_t>(count + 1);
        *insufficient = true;
        return false;
    }
    std::memcpy(*pointer, src, count * sizeof(char16_t));
    (*pointer)[count] = 0;
    *length = static_cast<uint32_t>(count);
    return true;
}

bool ascii_ieq(const char16_t* s, size_t n, const char* lit) {
    if (std::strlen(lit) != n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        char16_t c = s[i];
        if (c >= u'A' && c <= u'Z') {
            c = static_cast<char16_t>(c + 32);
        }
        if (c != static_cast<char16_t>(lit[i])) {
            return false;
        }
    }
    return true;
}

// Registration that never fires (connectivity / license change notifications).
HResult register_silent(const char* import_name,
                        HandleKind kind,
                        bool arguments_valid,
                        Pg6XTaskQueueRegistrationToken* token) {
    hit(import_name);
    if (token == nullptr) {
        return E_INVALIDARG_HR;
    }
    token->token = 0;
    if (!arguments_valid) {
        return E_INVALIDARG_HR;
    }
    token->token = handles().create(kind);
    return S_OK_HR;
}

int32_t unregister_silent(const char* import_name,
                          HandleKind kind,
                          Pg6XTaskQueueRegistrationToken token) {
    hit(import_name);
    return handles().release(token.token, {kind}) ? 1 : 0;
}

// Telemetry: accept and drop without reading the payload.
uint32_t drop_event(const char* import_name, const void* event_descriptor) {
    hit(import_name);
    if (event_descriptor == nullptr) {
        return ETW_ERROR_INVALID_PARAMETER;
    }
    dropped_events().fetch_add(1, std::memory_order_relaxed);
    return ETW_ERROR_SUCCESS;
}

}  // namespace

// ---------------------------------------------------------------------------
// Bring-up diagnostics
// ---------------------------------------------------------------------------

uint64_t stub_hit_count(std::string_view import_name) {
    auto& counters = hit_counters();
    std::lock_guard<std::mutex> lock(counters.mutex);
    auto it = counters.counts.find(import_name);
    return it != counters.counts.end() ? it->second : 0;
}

std::vector<std::pair<std::string, uint64_t>> stub_hit_snapshot() {
    std::vector<std::pair<std::string, uint64_t>> out;
    {
        auto& counters = hit_counters();
        std::lock_guard<std::mutex> lock(counters.mutex);
        out.reserve(counters.counts.size());
        for (const auto& [name, count] : counters.counts) {
            if (count != 0) {
                out.emplace_back(std::string(name), count);
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

uint64_t dropped_telemetry_event_count() {
    return dropped_events().load(std::memory_order_relaxed);
}

size_t open_dummy_handle_count() {
    return handles().size();
}

void reset_offline_diagnostics() {
    {
        auto& counters = hit_counters();
        std::lock_guard<std::mutex> lock(counters.mutex);
        for (auto& entry : counters.counts) {
            entry.second = 0;
        }
    }
    dropped_events().store(0, std::memory_order_relaxed);
}

}  // namespace playgta6::os::offline

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints
// ---------------------------------------------------------------------------

namespace off = playgta6::os::offline;
namespace gdk = playgta6::os::gdk;
namespace win32 = playgta6::os::win32;

extern "C" {

// ----- ws2_32 --------------------------------------------------------------

int32_t pg6_win32_wsa_startup(uint16_t version_requested, Pg6WsaData* wsa_data) {
    off::hit("WSAStartup");
    if (wsa_data == nullptr) {
        return off::WSAEFAULT;
    }
    *wsa_data = Pg6WsaData{};
    wsa_data->high_version = off::kWinsockHighVersion;
    if ((version_requested & 0xFFu) == 0) {
        wsa_data->version = off::kWinsockHighVersion;
        return off::WSAVERNOTSUPPORTED;
    }
    const bool newer = (version_requested & 0xFFu) > 2u ||
                       ((version_requested & 0xFFu) == 2u && (version_requested >> 8) > 2u);
    wsa_data->version = newer ? off::kWinsockHighVersion : version_requested;
    std::snprintf(wsa_data->description, sizeof(wsa_data->description), "%s",
                  "PlayGTA6Web offline WinSock");
    std::snprintf(wsa_data->system_status, sizeof(wsa_data->system_status), "%s",
                  "Offline (network unavailable)");
    {
        std::lock_guard<std::mutex> lock(off::wsa_mutex());
        ++off::wsa_startup_count();
    }
    return 0;
}

int32_t pg6_win32_wsa_cleanup(void) {
    off::hit("WSACleanup");
    std::lock_guard<std::mutex> lock(off::wsa_mutex());
    if (off::wsa_startup_count() == 0) {
        off::set_wsa_error(off::WSANOTINITIALISED);
        return off::SOCKET_ERROR_VALUE;
    }
    --off::wsa_startup_count();
    return 0;
}

int32_t pg6_win32_wsa_get_last_error(void) {
    off::hit("WSAGetLastError");
    return static_cast<int32_t>(win32::GetLastError());
}

void pg6_win32_wsa_set_last_error(int32_t error) {
    off::hit("WSASetLastError");
    off::set_wsa_error(error);
}

Pg6Socket pg6_win32_socket(int32_t, int32_t, int32_t) {
    return off::socket_open("socket");
}

Pg6Socket pg6_win32_wsa_socket_w(int32_t, int32_t, int32_t, void*, uint32_t, uint32_t) {
    return off::socket_open("WSASocketW");
}

int32_t pg6_win32_closesocket(Pg6Socket) {
    return off::socket_call("closesocket");
}

int32_t pg6_win32_connect(Pg6Socket, const void*, int32_t) {
    return off::socket_call("connect");
}

int32_t pg6_win32_bind(Pg6Socket, const void*, int32_t) {
    return off::socket_call("bind");
}

int32_t pg6_win32_listen(Pg6Socket, int32_t) {
    return off::socket_call("listen");
}

Pg6Socket pg6_win32_accept(Pg6Socket, void*, int32_t*) {
    off::socket_call("accept");
    return off::INVALID_SOCKET_VALUE;
}

int32_t pg6_win32_send(Pg6Socket, const char*, int32_t, int32_t) {
    return off::socket_call("send");
}

int32_t pg6_win32_recv(Pg6Socket, char*, int32_t, int32_t) {
    return off::socket_call("recv");
}

int32_t pg6_win32_sendto(Pg6Socket, const char*, int32_t, int32_t, const void*, int32_t) {
    return off::socket_call("sendto");
}

int32_t pg6_win32_recvfrom(Pg6Socket, char*, int32_t, int32_t, void*, int32_t*) {
    return off::socket_call("recvfrom");
}

int32_t pg6_win32_setsockopt(Pg6Socket, int32_t, int32_t, const char*, int32_t) {
    return off::socket_call("setsockopt");
}

int32_t pg6_win32_getsockopt(Pg6Socket, int32_t, int32_t, char*, int32_t*) {
    return off::socket_call("getsockopt");
}

int32_t pg6_win32_ioctlsocket(Pg6Socket, int32_t, uint32_t*) {
    return off::socket_call("ioctlsocket");
}

int32_t pg6_win32_shutdown(Pg6Socket, int32_t) {
    return off::socket_call("shutdown");
}

int32_t pg6_win32_select(int32_t, void* read_fds, void* write_fds, void* except_fds, const void*) {
    const int32_t result = off::socket_call("select");
    if (off::wsa_started() && read_fds == nullptr && write_fds == nullptr &&
        except_fds == nullptr) {
        off::set_wsa_error(off::WSAEINVAL);
    }
    return result;
}

int32_t pg6_win32_getaddrinfo(const char*, const char*, const void*, Pg6AddrInfo** result) {
    return off::resolve_name("getaddrinfo", result);
}

int32_t pg6_win32_get_addr_info_w(const char16_t*,
                                  const char16_t*,
                                  const void*,
                                  Pg6AddrInfo** result) {
    return off::resolve_name("GetAddrInfoW", result);
}

void pg6_win32_freeaddrinfo(Pg6AddrInfo*) {
    off::hit("freeaddrinfo");
}

void pg6_win32_free_addr_info_w(Pg6AddrInfo*) {
    off::hit("FreeAddrInfoW");
}

// ----- winhttp -------------------------------------------------------------

Pg6HInternet pg6_win32_winhttp_open(const char16_t*,
                                    uint32_t,
                                    const char16_t*,
                                    const char16_t*,
                                    uint32_t) {
    off::hit("WinHttpOpen");
    win32::SetLastError(win32::ERROR_SUCCESS);
    return off::handles().create(off::HandleKind::WinHttpSession);
}

Pg6HInternet pg6_win32_winhttp_connect(Pg6HInternet session,
                                       const char16_t* server_name,
                                       uint16_t,
                                       uint32_t) {
    return off::winhttp_child("WinHttpConnect",
                              session,
                              off::HandleKind::WinHttpSession,
                              off::HandleKind::WinHttpConnection,
                              server_name != nullptr);
}

Pg6HInternet pg6_win32_winhttp_open_request(Pg6HInternet connection,
                                            const char16_t*,
                                            const char16_t*,
                                            const char16_t*,
                                            const char16_t*,
                                            const char16_t* const*,
                                            uint32_t) {
    return off::winhttp_child("WinHttpOpenRequest",
                              connection,
                              off::HandleKind::WinHttpConnection,
                              off::HandleKind::WinHttpRequest,
                              true);
}

int32_t pg6_win32_winhttp_send_request(Pg6HInternet request,
                                       const char16_t*,
                                       uint32_t,
                                       const void*,
                                       uint32_t,
                                       uint32_t,
                                       uint64_t) {
    return off::winhttp_fail("WinHttpSendRequest",
                             request,
                             {off::HandleKind::WinHttpRequest},
                             off::ERROR_WINHTTP_NAME_NOT_RESOLVED);
}

int32_t pg6_win32_winhttp_receive_response(Pg6HInternet request, void*) {
    return off::winhttp_fail("WinHttpReceiveResponse",
                             request,
                             {off::HandleKind::WinHttpRequest},
                             off::ERROR_WINHTTP_INCORRECT_HANDLE_STATE);
}

int32_t pg6_win32_winhttp_query_data_available(Pg6HInternet request, uint32_t* bytes_available) {
    if (bytes_available != nullptr) {
        *bytes_available = 0;
    }
    return off::winhttp_fail("WinHttpQueryDataAvailable",
                             request,
                             {off::HandleKind::WinHttpRequest},
                             off::ERROR_WINHTTP_INCORRECT_HANDLE_STATE);
}

int32_t pg6_win32_winhttp_read_data(Pg6HInternet request,
                                    void*,
                                    uint32_t,
                                    uint32_t* bytes_read) {
    if (bytes_read != nullptr) {
        *bytes_read = 0;
    }
    return off::winhttp_fail("WinHttpReadData",
                             request,
                             {off::HandleKind::WinHttpRequest},
                             off::ERROR_WINHTTP_INCORRECT_HANDLE_STATE);
}

uint64_t pg6_win32_winhttp_set_status_callback(Pg6HInternet handle,
                                               uint64_t,
                                               uint32_t,
                                               uint64_t) {
    if (off::winhttp_configure("WinHttpSetStatusCallback", handle, false) == 0) {
        return ~uint64_t{0};  // WINHTTP_INVALID_STATUS_CALLBACK
    }
    // The callback is never stored: no request ever reaches a state that would notify it.
    return 0;
}

int32_t pg6_win32_winhttp_add_request_headers(Pg6HInternet request,
                                              const char16_t* headers,
                                              uint32_t headers_length,
                                              uint32_t) {
    off::hit("WinHttpAddRequestHeaders");
    if (!off::handles().lookup(request, off::HandleKind::WinHttpRequest)) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    if (headers == nullptr && headers_length != 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);  // headers of a request that is never sent
    return 1;
}

int32_t pg6_win32_winhttp_write_data(Pg6HInternet request,
                                     const void*,
                                     uint32_t,
                                     uint32_t* bytes_written) {
    if (bytes_written != nullptr) {
        *bytes_written = 0;
    }
    return off::winhttp_fail("WinHttpWriteData",
                             request,
                             {off::HandleKind::WinHttpRequest},
                             off::ERROR_WINHTTP_INCORRECT_HANDLE_STATE);
}

int32_t pg6_win32_winhttp_query_headers(Pg6HInternet request,
                                        uint32_t,
                                        const char16_t*,
                                        void*,
                                        uint32_t*,
                                        uint32_t*) {
    return off::winhttp_fail("WinHttpQueryHeaders",
                             request,
                             {off::HandleKind::WinHttpRequest},
                             off::ERROR_WINHTTP_INCORRECT_HANDLE_STATE);
}

int32_t pg6_win32_winhttp_crack_url(const char16_t* url,
                                    uint32_t url_length,
                                    uint32_t,
                                    Pg6UrlComponentsW* components) {
    off::hit("WinHttpCrackUrl");
    if (url == nullptr || components == nullptr ||
        components->struct_size != sizeof(Pg6UrlComponentsW)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    size_t total = url_length;
    if (total == 0) {
        while (url[total] != 0) {
            ++total;
        }
    }
    // scheme "://" authority [path] [?query|#fragment]
    size_t scheme_end = 0;
    while (scheme_end < total && url[scheme_end] != u':') {
        ++scheme_end;
    }
    if (scheme_end == 0 || scheme_end + 2 >= total || url[scheme_end + 1] != u'/' ||
        url[scheme_end + 2] != u'/') {
        win32::SetLastError(off::ERROR_WINHTTP_INVALID_URL);
        return 0;
    }
    int32_t scheme_id = 0;
    if (off::ascii_ieq(url, scheme_end, "http")) {
        scheme_id = off::INTERNET_SCHEME_HTTP;
    } else if (off::ascii_ieq(url, scheme_end, "https")) {
        scheme_id = off::INTERNET_SCHEME_HTTPS;
    } else {
        win32::SetLastError(off::ERROR_WINHTTP_UNRECOGNIZED_SCHEME);
        return 0;
    }
    const size_t auth_begin = scheme_end + 3;
    size_t auth_end = auth_begin;
    while (auth_end < total && url[auth_end] != u'/' && url[auth_end] != u'?' &&
           url[auth_end] != u'#') {
        ++auth_end;
    }
    size_t path_end = auth_end;
    while (path_end < total && url[path_end] != u'?' && url[path_end] != u'#') {
        ++path_end;
    }

    size_t host_begin = auth_begin;
    size_t user_end = auth_begin;
    size_t pass_begin = auth_begin;
    size_t pass_end = auth_begin;
    size_t at = auth_end;
    for (size_t i = auth_begin; i < auth_end; ++i) {
        if (url[i] == u'@') {
            at = i;  // the last '@' separates user info from the host
        }
    }
    if (at != auth_end) {
        size_t colon = at;
        for (size_t i = auth_begin; i < at; ++i) {
            if (url[i] == u':') {
                colon = i;
                break;
            }
        }
        user_end = colon;
        pass_begin = colon < at ? colon + 1 : at;
        pass_end = at;
        host_begin = at + 1;
    }
    size_t host_end = auth_end;
    uint32_t port = scheme_id == off::INTERNET_SCHEME_HTTPS ? 443u : 80u;
    {
        size_t colon = host_end;
        for (size_t i = host_begin; i < auth_end; ++i) {
            if (url[i] == u']') {
                colon = auth_end;  // ports follow the closing bracket of an IPv6 literal
                for (size_t j = i + 1; j < auth_end; ++j) {
                    if (url[j] == u':') {
                        colon = j;
                        break;
                    }
                }
                break;
            }
            if (url[i] == u':') {
                colon = i;
                break;
            }
        }
        if (colon < auth_end) {
            uint32_t value = 0;
            if (colon + 1 == auth_end) {
                win32::SetLastError(off::ERROR_WINHTTP_INVALID_URL);
                return 0;
            }
            for (size_t i = colon + 1; i < auth_end; ++i) {
                if (url[i] < u'0' || url[i] > u'9' || (value = value * 10 + (url[i] - u'0')) > 65535) {
                    win32::SetLastError(off::ERROR_WINHTTP_INVALID_URL);
                    return 0;
                }
            }
            port = value;
            host_end = colon;
        }
    }
    if (host_end == host_begin) {
        win32::SetLastError(off::ERROR_WINHTTP_INVALID_URL);
        return 0;
    }

    components->scheme_id = scheme_id;
    components->port = static_cast<uint16_t>(port);
    bool insufficient = false;
    off::crack_component(url, scheme_end, &components->scheme, &components->scheme_length,
                         &insufficient);
    off::crack_component(url + host_begin, host_end - host_begin, &components->host_name,
                         &components->host_name_length, &insufficient);
    off::crack_component(url + auth_begin, user_end - auth_begin, &components->user_name,
                         &components->user_name_length, &insufficient);
    off::crack_component(url + pass_begin, pass_end - pass_begin, &components->password,
                         &components->password_length, &insufficient);
    off::crack_component(url + auth_end, path_end - auth_end, &components->url_path,
                         &components->url_path_length, &insufficient);
    off::crack_component(url + path_end, total - path_end, &components->extra_info,
                         &components->extra_info_length, &insufficient);
    if (insufficient) {
        win32::SetLastError(122u);  // ERROR_INSUFFICIENT_BUFFER; required sizes are in the struct
        return 0;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_winhttp_set_option(Pg6HInternet handle, uint32_t, const void*, uint32_t) {
    return off::winhttp_configure("WinHttpSetOption", handle, true);
}

int32_t pg6_win32_winhttp_set_timeouts(Pg6HInternet handle, int32_t, int32_t, int32_t, int32_t) {
    return off::winhttp_configure("WinHttpSetTimeouts", handle, false);
}

int32_t pg6_win32_winhttp_close_handle(Pg6HInternet handle) {
    off::hit("WinHttpCloseHandle");
    if (off::handles().release(handle,
                               {off::HandleKind::WinHttpSession,
                                off::HandleKind::WinHttpConnection,
                                off::HandleKind::WinHttpRequest})) {
        win32::SetLastError(win32::ERROR_SUCCESS);
        return 1;
    }
    win32::SetLastError(win32::ERROR_INVALID_HANDLE);
    return 0;
}

// ----- XNetworking ---------------------------------------------------------

Pg6HResult pg6_gdk_xnetworking_get_connectivity_hint(Pg6XNetworkingConnectivityHint* hint) {
    off::hit("XNetworkingGetConnectivityHint");
    if (hint == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    *hint = Pg6XNetworkingConnectivityHint{};
    hint->connectivity_level = static_cast<uint32_t>(off::ConnectivityLevel::None);
    hint->connectivity_cost = static_cast<uint32_t>(off::ConnectivityCost::Unknown);
    // Initialized, so titles do not wait for a change notification that will never arrive.
    hint->network_initialized = true;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_gdk_xnetworking_register_connectivity_hint_changed(
    Pg6XTaskQueueHandle,
    void*,
    Pg6XNetworkingConnectivityHintChangedCallback callback,
    Pg6XTaskQueueRegistrationToken* token) {
    return off::register_silent("XNetworkingRegisterConnectivityHintChanged",
                                off::HandleKind::ConnectivityRegistration,
                                callback != nullptr,
                                token);
}

int32_t pg6_gdk_xnetworking_unregister_connectivity_hint_changed(
    Pg6XTaskQueueRegistrationToken token,
    int32_t) {
    return off::unregister_silent("XNetworkingUnregisterConnectivityHintChanged",
                                  off::HandleKind::ConnectivityRegistration,
                                  token);
}

// ----- Live-dependent XUser ------------------------------------------------

Pg6HResult pg6_gdk_xuser_get_token_and_signature_async(Pg6XUserHandle user,
                                                       uint32_t,
                                                       const char* method,
                                                       const char* url,
                                                       size_t,
                                                       const void*,
                                                       size_t,
                                                       const void*,
                                                       Pg6XAsyncBlock* async_block) {
    off::hit("XUserGetTokenAndSignatureAsync");
    if (user == nullptr || method == nullptr || url == nullptr || async_block == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    return off::complete_async(async_block, off::kOfflineHResult);
}

Pg6HResult pg6_gdk_xuser_get_token_and_signature_result_size(Pg6XAsyncBlock* async_block,
                                                             size_t* buffer_size) {
    return off::count_result("XUserGetTokenAndSignatureResultSize", async_block, buffer_size);
}

Pg6HResult pg6_gdk_xuser_get_token_and_signature_result(Pg6XAsyncBlock* async_block,
                                                        size_t,
                                                        void*,
                                                        void** ptr_to_buffer,
                                                        size_t* buffer_used) {
    off::hit("XUserGetTokenAndSignatureResult");
    if (ptr_to_buffer != nullptr) {
        *ptr_to_buffer = nullptr;
    }
    if (buffer_used != nullptr) {
        *buffer_used = 0;
    }
    if (ptr_to_buffer == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    return off::async_result_status(async_block);
}

Pg6HResult pg6_gdk_xuser_check_privilege(Pg6XUserHandle user,
                                         uint32_t,
                                         uint32_t,
                                         bool* has_privilege,
                                         uint32_t* deny_reason) {
    off::hit("XUserCheckPrivilege");
    if (has_privilege != nullptr) {
        *has_privilege = false;
    }
    if (deny_reason != nullptr) {
        *deny_reason = static_cast<uint32_t>(off::PrivilegeDenyReason::None);
    }
    if (user == nullptr || has_privilege == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    // Every Live privilege (multiplayer, communications, UGC, ...) is unavailable offline.
    if (deny_reason != nullptr) {
        *deny_reason = static_cast<uint32_t>(off::PrivilegeDenyReason::Unknown);
    }
    return gdk::S_OK_HR;
}

Pg6HResult pg6_gdk_xuser_resolve_privilege_with_ui_async(Pg6XUserHandle user,
                                                         uint32_t,
                                                         uint32_t,
                                                         Pg6XAsyncBlock* async_block) {
    off::hit("XUserResolvePrivilegeWithUiAsync");
    if (user == nullptr || async_block == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    return off::complete_async(async_block, off::kOfflineHResult);
}

Pg6HResult pg6_gdk_xuser_resolve_privilege_with_ui_result(Pg6XAsyncBlock* async_block) {
    off::hit("XUserResolvePrivilegeWithUiResult");
    return off::async_result_status(async_block);
}

// ----- Xbox Live (XSAPI) ---------------------------------------------------

Pg6HResult pg6_gdk_xbl_initialize(const void* init_args) {
    off::hit("XblInitialize");
    return init_args != nullptr ? gdk::S_OK_HR : gdk::E_INVALIDARG_HR;
}

Pg6HResult pg6_gdk_xbl_cleanup_async(Pg6XAsyncBlock* async_block) {
    off::hit("XblCleanupAsync");
    return off::complete_async(async_block, gdk::S_OK_HR);
}

Pg6HResult pg6_gdk_xbl_context_create_handle(Pg6XUserHandle user, Pg6XblContextHandle* context) {
    off::hit("XblContextCreateHandle");
    if (context == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    *context = nullptr;
    uint64_t xuid = 0;
    if (user == nullptr || gdk::failed(pg6_gdk_xuser_get_id(user, &xuid))) {
        return gdk::E_INVALIDARG_HR;
    }
    *context = off::to_handle_ptr<Pg6XblContext>(
        off::handles().create(off::HandleKind::XblContext, xuid));
    return gdk::S_OK_HR;
}

Pg6HResult pg6_gdk_xbl_context_duplicate_handle(Pg6XblContextHandle context,
                                                Pg6XblContextHandle* duplicated) {
    off::hit("XblContextDuplicateHandle");
    if (duplicated == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    *duplicated = nullptr;
    if (!off::handles().retain(off::from_handle_ptr(context), off::HandleKind::XblContext)) {
        return gdk::E_INVALIDARG_HR;
    }
    *duplicated = context;
    return gdk::S_OK_HR;
}

void pg6_gdk_xbl_context_close_handle(Pg6XblContextHandle context) {
    off::hit("XblContextCloseHandle");
    off::handles().release(off::from_handle_ptr(context), {off::HandleKind::XblContext});
}

Pg6HResult pg6_gdk_xbl_context_get_xbox_user_id(Pg6XblContextHandle context, uint64_t* xuid) {
    off::hit("XblContextGetXboxUserId");
    if (xuid == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    *xuid = 0;
    return off::handles().lookup(off::from_handle_ptr(context), off::HandleKind::XblContext, xuid)
               ? gdk::S_OK_HR
               : gdk::E_INVALIDARG_HR;
}

Pg6HResult pg6_gdk_xbl_achievements_get_achievements_for_title_id_async(
    Pg6XblContextHandle context,
    uint64_t,
    uint32_t,
    uint32_t,
    int32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    Pg6XAsyncBlock* async_block) {
    return off::xbl_async("XblAchievementsGetAchievementsForTitleIdAsync", context, async_block);
}

Pg6HResult pg6_gdk_xbl_achievements_get_achievements_for_title_id_result(
    Pg6XAsyncBlock* async_block,
    Pg6OfflineResultHandle* result) {
    return off::handle_result("XblAchievementsGetAchievementsForTitleIdResult", async_block, result);
}

Pg6HResult pg6_gdk_xbl_achievements_update_achievement_async(Pg6XblContextHandle context,
                                                             uint64_t,
                                                             const char*,
                                                             uint32_t,
                                                             Pg6XAsyncBlock* async_block) {
    return off::xbl_async("XblAchievementsUpdateAchievementAsync", context, async_block);
}

Pg6HResult pg6_gdk_xbl_profile_get_user_profile_async(Pg6XblContextHandle context,
                                                      uint64_t,
                                                      Pg6XAsyncBlock* async_block) {
    return off::xbl_async("XblProfileGetUserProfileAsync", context, async_block);
}

Pg6HResult pg6_gdk_xbl_profile_get_user_profile_result(Pg6XAsyncBlock* async_block,
                                                       void* profile) {
    return off::struct_result("XblProfileGetUserProfileResult", async_block, profile);
}

Pg6HResult pg6_gdk_xbl_presence_set_presence_async(Pg6XblContextHandle context,
                                                   int32_t,
                                                   const void*,
                                                   Pg6XAsyncBlock* async_block) {
    return off::xbl_async("XblPresenceSetPresenceAsync", context, async_block);
}

Pg6HResult pg6_gdk_xbl_user_statistics_get_single_user_statistic_async(
    Pg6XblContextHandle context,
    uint64_t,
    const char*,
    const char*,
    Pg6XAsyncBlock* async_block) {
    return off::xbl_async("XblUserStatisticsGetSingleUserStatisticAsync", context, async_block);
}

Pg6HResult pg6_gdk_xbl_user_statistics_get_single_user_statistic_result_size(
    Pg6XAsyncBlock* async_block,
    size_t* result_size) {
    return off::count_result(
        "XblUserStatisticsGetSingleUserStatisticResultSize", async_block, result_size);
}

Pg6HResult pg6_gdk_xbl_social_get_social_relationships_async(Pg6XblContextHandle context,
                                                             uint64_t,
                                                             uint32_t,
                                                             size_t,
                                                             size_t,
                                                             Pg6XAsyncBlock* async_block) {
    return off::xbl_async("XblSocialGetSocialRelationshipsAsync", context, async_block);
}

Pg6HResult pg6_gdk_xbl_social_get_social_relationships_result(Pg6XAsyncBlock* async_block,
                                                              Pg6OfflineResultHandle* result) {
    return off::handle_result("XblSocialGetSocialRelationshipsResult", async_block, result);
}

Pg6HResult pg6_gdk_xbl_title_storage_get_quota_async(Pg6XblContextHandle context,
                                                     const char*,
                                                     uint32_t,
                                                     Pg6XAsyncBlock* async_block) {
    return off::xbl_async("XblTitleStorageGetQuotaAsync", context, async_block);
}

Pg6HResult pg6_gdk_xbl_title_storage_get_quota_result(Pg6XAsyncBlock* async_block,
                                                      size_t* used_bytes,
                                                      size_t* quota_bytes) {
    off::hit("XblTitleStorageGetQuotaResult");
    if (used_bytes != nullptr) {
        *used_bytes = 0;
    }
    if (quota_bytes != nullptr) {
        *quota_bytes = 0;
    }
    if (used_bytes == nullptr || quota_bytes == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    return off::async_result_status(async_block);
}

Pg6HResult pg6_gdk_xbl_multiplayer_get_session_async(Pg6XblContextHandle context,
                                                     const void*,
                                                     Pg6XAsyncBlock* async_block) {
    return off::xbl_async("XblMultiplayerGetSessionAsync", context, async_block);
}

Pg6HResult pg6_gdk_xbl_multiplayer_get_session_result(Pg6XAsyncBlock* async_block,
                                                      Pg6OfflineResultHandle* session) {
    return off::handle_result("XblMultiplayerGetSessionResult", async_block, session);
}

Pg6HResult pg6_gdk_xbl_events_write_in_game_event(Pg6XblContextHandle context,
                                                  const char* event_name,
                                                  const char*,
                                                  const char*) {
    off::hit("XblEventsWriteInGameEvent");
    if (event_name == nullptr ||
        !off::handles().lookup(off::from_handle_ptr(context), off::HandleKind::XblContext)) {
        return gdk::E_INVALIDARG_HR;
    }
    off::dropped_events().fetch_add(1, std::memory_order_relaxed);
    return gdk::S_OK_HR;
}

// ----- XStore --------------------------------------------------------------

Pg6HResult pg6_gdk_xstore_create_context(Pg6XUserHandle, Pg6XStoreContextHandle* store_context) {
    off::hit("XStoreCreateContext");
    if (store_context == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    *store_context =
        off::to_handle_ptr<Pg6XStoreContext>(off::handles().create(off::HandleKind::XStoreContext));
    return gdk::S_OK_HR;
}

void pg6_gdk_xstore_close_context_handle(Pg6XStoreContextHandle store_context) {
    off::hit("XStoreCloseContextHandle");
    off::handles().release(off::from_handle_ptr(store_context), {off::HandleKind::XStoreContext});
}

Pg6HResult pg6_gdk_xstore_query_game_license_async(Pg6XStoreContextHandle store_context,
                                                   Pg6XAsyncBlock* async_block) {
    return off::xstore_async("XStoreQueryGameLicenseAsync", store_context, async_block);
}

Pg6HResult pg6_gdk_xstore_query_game_license_result(Pg6XAsyncBlock* async_block, void* license) {
    return off::struct_result("XStoreQueryGameLicenseResult", async_block, license);
}

Pg6HResult pg6_gdk_xstore_query_add_on_licenses_async(Pg6XStoreContextHandle store_context,
                                                      Pg6XAsyncBlock* async_block) {
    return off::xstore_async("XStoreQueryAddOnLicensesAsync", store_context, async_block);
}

Pg6HResult pg6_gdk_xstore_query_add_on_licenses_result_count(Pg6XAsyncBlock* async_block,
                                                             uint32_t* count) {
    return off::count_result("XStoreQueryAddOnLicensesResultCount", async_block, count);
}

Pg6HResult pg6_gdk_xstore_query_associated_products_async(Pg6XStoreContextHandle store_context,
                                                          uint32_t,
                                                          uint32_t,
                                                          Pg6XAsyncBlock* async_block) {
    return off::xstore_async("XStoreQueryAssociatedProductsAsync", store_context, async_block);
}

Pg6HResult pg6_gdk_xstore_query_associated_products_result(Pg6XAsyncBlock* async_block,
                                                           Pg6OfflineResultHandle* query) {
    return off::handle_result("XStoreQueryAssociatedProductsResult", async_block, query);
}

Pg6HResult pg6_gdk_xstore_query_product_for_current_game_async(
    Pg6XStoreContextHandle store_context,
    Pg6XAsyncBlock* async_block) {
    return off::xstore_async("XStoreQueryProductForCurrentGameAsync", store_context, async_block);
}

Pg6HResult pg6_gdk_xstore_query_product_for_current_game_result(Pg6XAsyncBlock* async_block,
                                                                Pg6OfflineResultHandle* query) {
    return off::handle_result("XStoreQueryProductForCurrentGameResult", async_block, query);
}

Pg6HResult pg6_gdk_xstore_register_game_license_changed(
    Pg6XStoreContextHandle store_context,
    Pg6XTaskQueueHandle,
    void*,
    Pg6XStoreGameLicenseChangedCallback callback,
    Pg6XTaskQueueRegistrationToken* token) {
    const bool context_valid = off::handles().lookup(off::from_handle_ptr(store_context),
                                                     off::HandleKind::XStoreContext);
    return off::register_silent("XStoreRegisterGameLicenseChanged",
                                off::HandleKind::LicenseRegistration,
                                context_valid && callback != nullptr,
                                token);
}

int32_t pg6_gdk_xstore_unregister_game_license_changed(Pg6XStoreContextHandle,
                                                       Pg6XTaskQueueRegistrationToken token,
                                                       int32_t) {
    return off::unregister_silent("XStoreUnregisterGameLicenseChanged",
                                  off::HandleKind::LicenseRegistration,
                                  token);
}

// ----- advapi32 ETW / TraceLogging -----------------------------------------

uint32_t pg6_win32_event_register(const void* provider_id,
                                  Pg6EtwEnableCallback,
                                  void*,
                                  Pg6RegHandle* reg_handle) {
    off::hit("EventRegister");
    if (reg_handle == nullptr) {
        return off::ETW_ERROR_INVALID_PARAMETER;
    }
    *reg_handle = 0;
    if (provider_id == nullptr) {
        return off::ETW_ERROR_INVALID_PARAMETER;
    }
    // The enable callback is never invoked: no trace session ever enables the provider.
    *reg_handle = off::handles().create(off::HandleKind::EtwProvider);
    return off::ETW_ERROR_SUCCESS;
}

uint32_t pg6_win32_event_unregister(Pg6RegHandle reg_handle) {
    off::hit("EventUnregister");
    return off::handles().release(reg_handle, {off::HandleKind::EtwProvider})
               ? off::ETW_ERROR_SUCCESS
               : off::ETW_ERROR_INVALID_HANDLE;
}

uint32_t pg6_win32_event_set_information(Pg6RegHandle reg_handle,
                                         uint32_t,
                                         const void*,
                                         uint32_t) {
    off::hit("EventSetInformation");
    return off::handles().lookup(reg_handle, off::HandleKind::EtwProvider)
               ? off::ETW_ERROR_SUCCESS
               : off::ETW_ERROR_INVALID_HANDLE;
}

uint8_t pg6_win32_event_enabled(Pg6RegHandle, const void*) {
    off::hit("EventEnabled");
    return 0;
}

uint8_t pg6_win32_event_provider_enabled(Pg6RegHandle, uint8_t, uint64_t) {
    off::hit("EventProviderEnabled");
    return 0;
}

uint32_t pg6_win32_event_write(Pg6RegHandle, const void* event_descriptor, uint32_t, const void*) {
    return off::drop_event("EventWrite", event_descriptor);
}

uint32_t pg6_win32_event_write_transfer(Pg6RegHandle,
                                        const void* event_descriptor,
                                        const void*,
                                        const void*,
                                        uint32_t,
                                        const void*) {
    return off::drop_event("EventWriteTransfer", event_descriptor);
}

}  // extern "C"
