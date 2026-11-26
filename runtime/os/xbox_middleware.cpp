#include "os/xbox_middleware.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cctype>
#include <chrono>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "os/guest_heap.h"
#include "os/os_shim.h"
#include "os/win32_loader.h"
#include "os/xasync_provider.h"

namespace playgta6::os::middleware {

namespace {

using gdk::E_ABORT_HR;
using gdk::E_FAIL_HR;
using gdk::E_INVALIDARG_HR;
using gdk::E_OUTOFMEMORY_HR;
using gdk::E_POINTER_HR;
using gdk::S_OK_HR;
using gdk::failed;

// ---------------------------------------------------------------------------
// XAsyncBlock completion helper (XAsync provider framework, os/xasync_provider.h)
// ---------------------------------------------------------------------------

Pg6HResult complete_async(Pg6XAsyncBlock* block, Pg6HResult status) {
    return gdk::complete_async_immediately(block, status);
}

// ---------------------------------------------------------------------------
// Minimal ref-counted COM stub object (used by AcpHalCreate and MFCreate*)
// ---------------------------------------------------------------------------

struct ComStubObject {
    const Pg6MwUnknownVtbl* vtbl = nullptr;
    std::atomic<uint32_t> ref_count{1};
    uint32_t kind = 0;
};

Pg6HResult com_stub_query_interface(void*, const void*, void** out_object) {
    if (out_object != nullptr) {
        *out_object = nullptr;
    }
    return kMwENoInterfaceHResult;
}

uint32_t com_stub_add_ref(void* self) {
    if (self == nullptr) {
        return 0;
    }
    auto* obj = static_cast<ComStubObject*>(self);
    return obj->ref_count.fetch_add(1, std::memory_order_relaxed) + 1u;
}

uint32_t com_stub_release(void* self) {
    if (self == nullptr) {
        return 0;
    }
    auto* obj = static_cast<ComStubObject*>(self);
    const uint32_t remaining = obj->ref_count.fetch_sub(1, std::memory_order_acq_rel) - 1u;
    if (remaining == 0) {
        delete obj;
    }
    return remaining;
}

const Pg6MwUnknownVtbl kComStubVtbl = {
    &com_stub_query_interface,
    &com_stub_add_ref,
    &com_stub_release,
};

void* create_com_stub(uint32_t kind) {
    auto* obj = new (std::nothrow) ComStubObject{};
    if (obj == nullptr) {
        return nullptr;
    }
    obj->vtbl = &kComStubVtbl;
    obj->kind = kind;
    return obj;
}

// ---------------------------------------------------------------------------
// 1. libHttpClient.GDK.dll stateful handle table
// ---------------------------------------------------------------------------

bool ascii_iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const unsigned char ca = static_cast<unsigned char>(a[i]);
        const unsigned char cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) {
            return false;
        }
    }
    return true;
}

struct HttpHeaderEntry {
    std::string name;
    std::string value;
    bool allow_tracing = true;
};

struct HttpCallState {
    uint32_t ref_count = 1;
    std::string method;
    std::string url;
    std::vector<uint8_t> request_body;
    std::vector<HttpHeaderEntry> request_headers;

    bool retry_allowed = true;
    uint32_t retry_cache_id = 0;
    uint32_t retry_delay_sec = 2;
    uint32_t timeout_sec = 30;
    uint32_t timeout_window_sec = 20;
    bool trace_call = true;

    bool performed = false;
    uint32_t status_code = 0;
    Pg6HResult network_error_code = S_OK_HR;
    uint32_t platform_network_error_code = 0;
    std::string platform_network_error_message;
    std::string response_string;
    std::vector<uint8_t> response_body;
    std::vector<HttpHeaderEntry> response_headers;
};

struct HttpClientGlobals {
    std::mutex mutex;
    uint32_t init_count = 0;
    uint32_t trace_level = 0;  // HCTraceLevel::Off
    bool etw_enabled = false;

    // Default request settings when a NULL call handle is passed to setters/getters.
    bool default_retry_allowed = true;
    uint32_t default_retry_cache_id = 0;
    uint32_t default_retry_delay_sec = 2;
    uint32_t default_timeout_sec = 30;
    uint32_t default_timeout_window_sec = 20;

    uint64_t next_handle = 0x0C110010ULL;
    std::unordered_map<uint64_t, std::shared_ptr<HttpCallState>> calls;
};

HttpClientGlobals& hc_globals() {
    static HttpClientGlobals g;
    return g;
}

Pg6HCCallHandle to_hc_handle(uint64_t id) {
    return reinterpret_cast<Pg6HCCallHandle>(static_cast<uintptr_t>(id));
}

uint64_t from_hc_handle(Pg6HCCallHandle handle) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
}

std::shared_ptr<HttpCallState> find_hc_call_locked(HttpClientGlobals& g, Pg6HCCallHandle handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    auto it = g.calls.find(from_hc_handle(handle));
    if (it == g.calls.end()) {
        return nullptr;
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// 2. Bink2Scarlett.dll state machine
// ---------------------------------------------------------------------------

constexpr uint32_t kBinkFromMemory = 0x04000000u;  // BINKFROMMEMORY: `name` is the file image

// The leading fields of a Bink 2 file ("KB2" + version letter), little-endian 32-bit words.
struct BinkHeader {
    uint32_t file_size = 0;
    uint32_t frames = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t frame_rate = 0;
    uint32_t frame_rate_div = 0;
};

bool parse_bink2_header(const char* image, BinkHeader* out) {
    uint32_t w[10] = {};
    std::memcpy(w, image, sizeof(w));
    if (std::memcmp(image, "KB2", 3) != 0) {
        return false;
    }
    BinkHeader h;
    h.file_size = w[1] + 8u;
    h.frames = w[2];
    h.width = w[5];
    h.height = w[6];
    h.frame_rate = w[7];
    h.frame_rate_div = w[8];
    if (h.frames == 0 || h.frames > 10000000u || h.width == 0 || h.width > 16384u || h.height == 0 ||
        h.height > 16384u || h.frame_rate == 0 || h.frame_rate_div == 0) {
        return false;
    }
    *out = h;
    return true;
}

// Real-time pacing of a movie that is not decoded: frame n is due (n - 1) frame periods after
// `start`, not counting paused time.
struct BinkClock {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point paused_at;
    bool paused = false;
};

struct BinkGlobals {
    std::mutex mutex;
    void* alloc_fn = nullptr;
    void* free_fn = nullptr;
    void* os_file_open_fn = nullptr;
    void* sound_open_fn = nullptr;
    uintptr_t sound_param = 0;
    std::unordered_set<int32_t> active_threads;
    std::unordered_set<Pg6HBink> live_handles;
    uint64_t refused_opens = 0;  // BinkOpenWithOptions calls answered with no movie
    std::unordered_map<Pg6HBink, BinkClock> clocks;  // playback clock per open movie
};

BinkGlobals& bink_globals() {
    static BinkGlobals g;
    return g;
}

// Frame `frame_num` of `bink` becomes due now (after a goto or a loop back to the first frame).
void restart_clock_locked(BinkGlobals& g, Pg6HBink bink) {
    const auto it = g.clocks.find(bink);
    if (it == g.clocks.end() || bink->frame_rate == 0) {
        return;
    }
    const double period = static_cast<double>(bink->frame_rate_div) / static_cast<double>(bink->frame_rate);
    const auto back = std::chrono::duration<double>(static_cast<double>(bink->frame_num - 1u) * period);
    it->second.start = std::chrono::steady_clock::now() -
                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(back);
    if (bink->paused != 0) {
        it->second.paused_at = std::chrono::steady_clock::now();
    }
}

bool is_live_bink_locked(const BinkGlobals& g, Pg6HBink bink) {
    return bink != nullptr && g.live_handles.find(bink) != g.live_handles.end();
}

// ---------------------------------------------------------------------------
// 3. AcpHal.dll (APU heap & address mapping)
// ---------------------------------------------------------------------------

struct ApuBlock {
    // Guest heap block (O24): guest memory the title reads and writes, accounted as guest memory.
    uint64_t va = 0;
    uint8_t* aligned_ptr = nullptr;  // host view of `va`
    uint32_t size = 0;
    uint32_t alignment = 16;
    uint32_t flags = 0;
    Pg6ApuAddress apu_addr = 0;
};

struct ApuGlobals {
    std::mutex mutex;
    bool heap_created = false;
    uint32_t cached_heap_size = 0;
    uint32_t non_cached_heap_size = 0;
    uint32_t shape_context_allocs = 0;
    std::vector<uint64_t> shape_context_arrays;  // guest heap blocks handed out by AllocateShapeContexts
    uint64_t xma_contexts = 0;                   // the kind-4 (XMA) array of the live allocation
    uint32_t xma_context_count = 0;
    Pg6ApuAddress next_apu_addr = 0x10000000u;
    std::vector<ApuBlock> blocks;
};

ApuGlobals& apu_globals() {
    static ApuGlobals g;
    return g;
}

uint32_t normalize_alignment(uint32_t align) {
    if (align < 16u) {
        return 16u;
    }
    // Round up to power of two (capped at 64 KiB).
    if (align > 65536u) {
        return 65536u;
    }
    uint32_t p = 16u;
    while (p < align) {
        p <<= 1u;
    }
    return p;
}

// ---------------------------------------------------------------------------
// 4. xapu.dll (stateful command/result session)
// ---------------------------------------------------------------------------

struct XApuSessionState {
    std::deque<std::vector<uint8_t>> results;
};

struct XApuGlobals {
    std::mutex mutex;
    uint64_t next_handle = 0x0A900010ULL;
    std::unordered_map<uint64_t, std::unique_ptr<XApuSessionState>> sessions;
};

XApuGlobals& xapu_globals() {
    static XApuGlobals g;
    return g;
}

Pg6XApuHandle to_xapu_handle(uint64_t id) {
    return reinterpret_cast<Pg6XApuHandle>(static_cast<uintptr_t>(id));
}

uint64_t from_xapu_handle(Pg6XApuHandle handle) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
}

// ---------------------------------------------------------------------------
// 5. chad.dll (XDsp session table)
// ---------------------------------------------------------------------------

struct XDspSessionState {
    uint32_t max_streams = 0;
    bool active = false;
    uint64_t submitted_commands = 0;
};

struct XDspGlobals {
    std::mutex mutex;
    uint64_t next_handle = 0x0D590010ULL;
    std::unordered_map<uint64_t, std::unique_ptr<XDspSessionState>> sessions;
};

XDspGlobals& xdsp_globals() {
    static XDspGlobals g;
    return g;
}

Pg6XDspHandle to_xdsp_handle(uint64_t id) {
    return reinterpret_cast<Pg6XDspHandle>(static_cast<uintptr_t>(id));
}

uint64_t from_xdsp_handle(Pg6XDspHandle handle) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
}

// ---------------------------------------------------------------------------
// 6. amd_fidelityfx_loader_xbox.dll (FFX context handle table)
// ---------------------------------------------------------------------------

struct FfxContextState {
    uint64_t desc_type = 0;
    uint64_t dispatch_count = 0;
};

struct FfxGlobals {
    std::mutex mutex;
    uint64_t next_handle = 0x0FF00010ULL;
    std::unordered_map<uint64_t, FfxContextState> contexts;
};

FfxGlobals& ffx_globals() {
    static FfxGlobals g;
    return g;
}

Pg6FfxContext to_ffx_handle(uint64_t id) {
    return reinterpret_cast<Pg6FfxContext>(static_cast<uintptr_t>(id));
}

uint64_t from_ffx_handle(Pg6FfxContext ctx) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ctx));
}

// ---------------------------------------------------------------------------
// 8 & 9. MFPlat.DLL & MFReadWrite.dll state
// ---------------------------------------------------------------------------

struct MfGlobals {
    std::mutex mutex;
    uint32_t startup_count = 0;
    std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();
};

MfGlobals& mf_globals() {
    static MfGlobals g;
    return g;
}

// ---------------------------------------------------------------------------
// Export registration table (exact DLL + symbol names from docs/imports_required.md)
// ---------------------------------------------------------------------------

struct MwExportSpec {
    const char* dll;
    const char* symbol;
    Pg6ProcAddress address;
};

template <typename FnPtr>
Pg6ProcAddress raw_proc_address(FnPtr fn) {
    return static_cast<Pg6ProcAddress>(reinterpret_cast<uintptr_t>(fn));
}

constexpr const char kDllHttpClient[] = "libHttpClient.GDK.dll";
constexpr const char kDllBink2[] = "Bink2Scarlett.dll";
constexpr const char kDllAcpHal[] = "AcpHal.dll";
constexpr const char kDllXApu[] = "xapu.dll";
constexpr const char kDllChad[] = "chad.dll";
constexpr const char kDllFfx[] = "amd_fidelityfx_loader_xbox.dll";
constexpr const char kDllBvh[] = "XboxBvhBuilder.dll";
constexpr const char kDllMfPlat[] = "MFPlat.DLL";
constexpr const char kDllMfReadWrite[] = "MFReadWrite.dll";

const MwExportSpec* middleware_export_table(size_t* out_count) {
    static const MwExportSpec kExports[] = {
        // libHttpClient.GDK.dll (38 symbols)
        {kDllHttpClient, "HCCleanupAsync", raw_proc_address(&pg6_mw_hc_cleanup_async)},
        {kDllHttpClient, "HCHttpCallCloseHandle", raw_proc_address(&pg6_mw_hc_http_call_close_handle)},
        {kDllHttpClient, "HCHttpCallCreate", raw_proc_address(&pg6_mw_hc_http_call_create)},
        {kDllHttpClient, "HCHttpCallDuplicateHandle", raw_proc_address(&pg6_mw_hc_http_call_duplicate_handle)},
        {kDllHttpClient, "HCHttpCallGetRequestUrl", raw_proc_address(&pg6_mw_hc_http_call_get_request_url)},
        {kDllHttpClient, "HCHttpCallPerformAsync", raw_proc_address(&pg6_mw_hc_http_call_perform_async)},
        {kDllHttpClient, "HCHttpCallRequestGetHeaderAtIndex", raw_proc_address(&pg6_mw_hc_http_call_request_get_header_at_index)},
        {kDllHttpClient, "HCHttpCallRequestGetNumHeaders", raw_proc_address(&pg6_mw_hc_http_call_request_get_num_headers)},
        {kDllHttpClient, "HCHttpCallRequestGetRequestBodyBytes", raw_proc_address(&pg6_mw_hc_http_call_request_get_request_body_bytes)},
        {kDllHttpClient, "HCHttpCallRequestGetRetryAllowed", raw_proc_address(&pg6_mw_hc_http_call_request_get_retry_allowed)},
        {kDllHttpClient, "HCHttpCallRequestGetRetryCacheId", raw_proc_address(&pg6_mw_hc_http_call_request_get_retry_cache_id)},
        {kDllHttpClient, "HCHttpCallRequestGetRetryDelay", raw_proc_address(&pg6_mw_hc_http_call_request_get_retry_delay)},
        {kDllHttpClient, "HCHttpCallRequestGetTimeout", raw_proc_address(&pg6_mw_hc_http_call_request_get_timeout)},
        {kDllHttpClient, "HCHttpCallRequestGetTimeoutWindow", raw_proc_address(&pg6_mw_hc_http_call_request_get_timeout_window)},
        {kDllHttpClient, "HCHttpCallRequestGetUrl", raw_proc_address(&pg6_mw_hc_http_call_request_get_url)},
        {kDllHttpClient, "HCHttpCallRequestSetHeader", raw_proc_address(&pg6_mw_hc_http_call_request_set_header)},
        {kDllHttpClient, "HCHttpCallRequestSetRequestBodyBytes", raw_proc_address(&pg6_mw_hc_http_call_request_set_request_body_bytes)},
        {kDllHttpClient, "HCHttpCallRequestSetRequestBodyString", raw_proc_address(&pg6_mw_hc_http_call_request_set_request_body_string)},
        {kDllHttpClient, "HCHttpCallRequestSetRetryAllowed", raw_proc_address(&pg6_mw_hc_http_call_request_set_retry_allowed)},
        {kDllHttpClient, "HCHttpCallRequestSetRetryCacheId", raw_proc_address(&pg6_mw_hc_http_call_request_set_retry_cache_id)},
        {kDllHttpClient, "HCHttpCallRequestSetRetryDelay", raw_proc_address(&pg6_mw_hc_http_call_request_set_retry_delay)},
        {kDllHttpClient, "HCHttpCallRequestSetTimeout", raw_proc_address(&pg6_mw_hc_http_call_request_set_timeout)},
        {kDllHttpClient, "HCHttpCallRequestSetTimeoutWindow", raw_proc_address(&pg6_mw_hc_http_call_request_set_timeout_window)},
        {kDllHttpClient, "HCHttpCallRequestSetUrl", raw_proc_address(&pg6_mw_hc_http_call_request_set_url)},
        {kDllHttpClient, "HCHttpCallResponseGetHeader", raw_proc_address(&pg6_mw_hc_http_call_response_get_header)},
        {kDllHttpClient, "HCHttpCallResponseGetHeaderAtIndex", raw_proc_address(&pg6_mw_hc_http_call_response_get_header_at_index)},
        {kDllHttpClient, "HCHttpCallResponseGetNetworkErrorCode", raw_proc_address(&pg6_mw_hc_http_call_response_get_network_error_code)},
        {kDllHttpClient, "HCHttpCallResponseGetNumHeaders", raw_proc_address(&pg6_mw_hc_http_call_response_get_num_headers)},
        {kDllHttpClient, "HCHttpCallResponseGetPlatformNetworkErrorMessage", raw_proc_address(&pg6_mw_hc_http_call_response_get_platform_network_error_message)},
        {kDllHttpClient, "HCHttpCallResponseGetResponseBodyBytes", raw_proc_address(&pg6_mw_hc_http_call_response_get_response_body_bytes)},
        {kDllHttpClient, "HCHttpCallResponseGetResponseBodyBytesSize", raw_proc_address(&pg6_mw_hc_http_call_response_get_response_body_bytes_size)},
        {kDllHttpClient, "HCHttpCallResponseGetResponseString", raw_proc_address(&pg6_mw_hc_http_call_response_get_response_string)},
        {kDllHttpClient, "HCHttpCallResponseGetStatusCode", raw_proc_address(&pg6_mw_hc_http_call_response_get_status_code)},
        {kDllHttpClient, "HCHttpCallSetTracing", raw_proc_address(&pg6_mw_hc_http_call_set_tracing)},
        {kDllHttpClient, "HCInitialize", raw_proc_address(&pg6_mw_hc_initialize)},
        {kDllHttpClient, "HCSettingsGetTraceLevel", raw_proc_address(&pg6_mw_hc_settings_get_trace_level)},
        {kDllHttpClient, "HCTraceImplMessage", raw_proc_address(&pg6_mw_hc_trace_impl_message)},
        {kDllHttpClient, "HCTraceSetEtwEnabled", raw_proc_address(&pg6_mw_hc_trace_set_etw_enabled)},

        // Bink2Scarlett.dll (23 symbols)
        {kDllBink2, "BinkClose", raw_proc_address(&pg6_mw_bink_close)},
        {kDllBink2, "BinkDoFrame", raw_proc_address(&pg6_mw_bink_do_frame)},
        {kDllBink2, "BinkDoFrameAsyncMulti", raw_proc_address(&pg6_mw_bink_do_frame_async_multi)},
        {kDllBink2, "BinkDoFrameAsyncWait", raw_proc_address(&pg6_mw_bink_do_frame_async_wait)},
        {kDllBink2, "BinkFreeGlobals", raw_proc_address(&pg6_mw_bink_free_globals)},
        {kDllBink2, "BinkGetFrameBuffersInfo", raw_proc_address(&pg6_mw_bink_get_frame_buffers_info)},
        {kDllBink2, "BinkGetKeyFrame", raw_proc_address(&pg6_mw_bink_get_key_frame)},
        {kDllBink2, "BinkGetPlatformInfo", raw_proc_address(&pg6_mw_bink_get_platform_info)},
        {kDllBink2, "BinkGoto", raw_proc_address(&pg6_mw_bink_goto)},
        {kDllBink2, "BinkNextFrame", raw_proc_address(&pg6_mw_bink_next_frame)},
        {kDllBink2, "BinkOpenWithOptions", raw_proc_address(&pg6_mw_bink_open_with_options)},
        {kDllBink2, "BinkPause", raw_proc_address(&pg6_mw_bink_pause)},
        {kDllBink2, "BinkRegisterFrameBuffers", raw_proc_address(&pg6_mw_bink_register_frame_buffers)},
        {kDllBink2, "BinkRequestStopAsyncThreadsMulti", raw_proc_address(&pg6_mw_bink_request_stop_async_threads_multi)},
        {kDllBink2, "BinkSetMemory", raw_proc_address(&pg6_mw_bink_set_memory)},
        {kDllBink2, "BinkSetOSFileCallbacks", raw_proc_address(&pg6_mw_bink_set_os_file_callbacks)},
        {kDllBink2, "BinkSetSoundSystem", raw_proc_address(&pg6_mw_bink_set_sound_system)},
        {kDllBink2, "BinkSetVolume", raw_proc_address(&pg6_mw_bink_set_volume)},
        {kDllBink2, "BinkSetWillLoop", raw_proc_address(&pg6_mw_bink_set_will_loop)},
        {kDllBink2, "BinkShouldSkip", raw_proc_address(&pg6_mw_bink_should_skip)},
        {kDllBink2, "BinkStartAsyncThread", raw_proc_address(&pg6_mw_bink_start_async_thread)},
        {kDllBink2, "BinkWait", raw_proc_address(&pg6_mw_bink_wait)},
        {kDllBink2, "BinkWaitStopAsyncThreadsMulti", raw_proc_address(&pg6_mw_bink_wait_stop_async_threads_multi)},

        // AcpHal.dll (9 symbols)
        {kDllAcpHal, "AcpHalAllocateShapeContexts", raw_proc_address(&pg6_mw_acp_hal_allocate_shape_contexts)},
        {kDllAcpHal, "AcpHalCreate", raw_proc_address(&pg6_mw_acp_hal_create)},
        {kDllAcpHal, "AcpHalReleaseShapeContexts", raw_proc_address(&pg6_mw_acp_hal_release_shape_contexts)},
        {kDllAcpHal, "ApuAlloc", raw_proc_address(&pg6_mw_apu_alloc)},
        {kDllAcpHal, "ApuCreateHeap", raw_proc_address(&pg6_mw_apu_create_heap)},
        {kDllAcpHal, "ApuFree", raw_proc_address(&pg6_mw_apu_free)},
        {kDllAcpHal, "ApuIsVirtualAddressValid", raw_proc_address(&pg6_mw_apu_is_virtual_address_valid)},
        {kDllAcpHal, "ApuMapApuAddress", raw_proc_address(&pg6_mw_apu_map_apu_address)},
        {kDllAcpHal, "ApuMapVirtualAddress", raw_proc_address(&pg6_mw_apu_map_virtual_address)},

        // xapu.dll (5 symbols)
        {kDllXApu, "XApuConnect", raw_proc_address(&pg6_mw_xapu_connect)},
        {kDllXApu, "XApuDequeueResult", raw_proc_address(&pg6_mw_xapu_dequeue_result)},
        {kDllXApu, "XApuDisconnect", raw_proc_address(&pg6_mw_xapu_disconnect)},
        {kDllXApu, "XApuEnqueueCommand", raw_proc_address(&pg6_mw_xapu_enqueue_command)},
        {kDllXApu, "XApuGetResultCount", raw_proc_address(&pg6_mw_xapu_get_result_count)},

        // chad.dll (5 symbols)
        {kDllChad, "XDspActivate", raw_proc_address(&pg6_mw_xdsp_activate)},
        {kDllChad, "XDspConnectWithMaximumStreamLimit", raw_proc_address(&pg6_mw_xdsp_connect_with_maximum_stream_limit)},
        {kDllChad, "XDspDeactivate", raw_proc_address(&pg6_mw_xdsp_deactivate)},
        {kDllChad, "XDspDisconnect", raw_proc_address(&pg6_mw_xdsp_disconnect)},
        {kDllChad, "XDspSubmitCommand", raw_proc_address(&pg6_mw_xdsp_submit_command)},

        // amd_fidelityfx_loader_xbox.dll (4 symbols)
        {kDllFfx, "ffxCreateContext", raw_proc_address(&pg6_mw_ffx_create_context)},
        {kDllFfx, "ffxDestroyContext", raw_proc_address(&pg6_mw_ffx_destroy_context)},
        {kDllFfx, "ffxDispatch", raw_proc_address(&pg6_mw_ffx_dispatch)},
        {kDllFfx, "ffxQuery", raw_proc_address(&pg6_mw_ffx_query)},

        // XboxBvhBuilder.dll (7 mangled symbols)
        {kDllBvh, "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z", raw_proc_address(&pg6_mw_bvh_builder_ctor)},
        {kDllBvh, "??1XboxBvhBuilder@@QEAA@XZ", raw_proc_address(&pg6_mw_bvh_builder_dtor)},
        {kDllBvh,
         "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z",
         raw_proc_address(&pg6_mw_bvh_builder_build_batched_2x)},
        {kDllBvh,
         "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z",
         raw_proc_address(&pg6_mw_bvh_builder_copy)},
        {kDllBvh,
         "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z",
         raw_proc_address(&pg6_mw_bvh_builder_get_prebuild_info)},
        {kDllBvh,
         "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z",
         raw_proc_address(&pg6_mw_bvh_builder_rehydrate_batched_2x)},
        {kDllBvh, "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z", raw_proc_address(&pg6_mw_bvh_builder_set_64bit_morton_codes)},

        // MFPlat.DLL (6 symbols)
        {kDllMfPlat, "MFCreateAttributes", raw_proc_address(&pg6_mw_mf_create_attributes)},
        {kDllMfPlat, "MFCreateMediaType", raw_proc_address(&pg6_mw_mf_create_media_type)},
        {kDllMfPlat, "MFCreateSourceResolver", raw_proc_address(&pg6_mw_mf_create_source_resolver)},
        {kDllMfPlat, "MFGetSystemTime", raw_proc_address(&pg6_mw_mf_get_system_time)},
        {kDllMfPlat, "MFShutdown", raw_proc_address(&pg6_mw_mf_shutdown)},
        {kDllMfPlat, "MFStartup", raw_proc_address(&pg6_mw_mf_startup)},

        // MFReadWrite.dll (1 symbol)
        {kDllMfReadWrite, "MFCreateSourceReaderFromMediaSource", raw_proc_address(&pg6_mw_mf_create_source_reader_from_media_source)},
    };

    static_assert(sizeof(kExports) / sizeof(kExports[0]) == kExpectedMiddlewareSymbolCount,
                  "Xbox middleware export table must contain exactly 98 symbols");

    if (out_count != nullptr) {
        *out_count = sizeof(kExports) / sizeof(kExports[0]);
    }
    return kExports;
}

}  // namespace
}  // namespace playgta6::os::middleware

namespace mw = playgta6::os::middleware;
namespace gdk = playgta6::os::gdk;

extern "C" {

// ---------------------------------------------------------------------------
// Registration entrypoint
// ---------------------------------------------------------------------------

uint32_t pg6_xbox_middleware_register_all(void) {
    static std::once_flag once;
    static uint32_t registered_count = 0;

    std::call_once(once, [] {
        size_t count = 0;
        const mw::MwExportSpec* exports = mw::middleware_export_table(&count);
        uint32_t ok = 0;
        for (size_t i = 0; i < count; ++i) {
            if (playgta6::os::win32::register_export(
                    exports[i].dll, exports[i].symbol, 0, exports[i].address)) {
                ++ok;
            }
        }
        registered_count = ok;
    });

    return registered_count;
}

// ---------------------------------------------------------------------------
// 1. libHttpClient.GDK.dll (38 symbols)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_hc_initialize(const void*) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    ++g.init_count;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_cleanup_async(Pg6XAsyncBlock* async_block) {
    if (async_block == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    {
        auto& g = mw::hc_globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        if (g.init_count > 0) {
            --g.init_count;
        }
    }
    return mw::complete_async(async_block, gdk::S_OK_HR);
}

Pg6HResult pg6_mw_hc_http_call_create(Pg6HCCallHandle* call) {
    if (call == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto state = std::make_shared<mw::HttpCallState>();
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    state->retry_allowed = g.default_retry_allowed;
    state->retry_cache_id = g.default_retry_cache_id;
    state->retry_delay_sec = g.default_retry_delay_sec;
    state->timeout_sec = g.default_timeout_sec;
    state->timeout_window_sec = g.default_timeout_window_sec;

    const uint64_t id = g.next_handle;
    g.next_handle += 0x10ULL;
    g.calls.emplace(id, std::move(state));
    *call = mw::to_hc_handle(id);
    return gdk::S_OK_HR;
}

Pg6HCCallHandle pg6_mw_hc_http_call_duplicate_handle(Pg6HCCallHandle call) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return nullptr;
    }
    ++state->ref_count;
    return call;
}

Pg6HResult pg6_mw_hc_http_call_close_handle(Pg6HCCallHandle call) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.calls.find(mw::from_hc_handle(call));
    if (it == g.calls.end()) {
        return gdk::E_INVALIDARG_HR;
    }
    if (--it->second->ref_count == 0) {
        g.calls.erase(it);
    }
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_perform_async(Pg6HCCallHandle call, Pg6XAsyncBlock* async_block) {
    if (async_block == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    {
        auto& g = mw::hc_globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        auto state = mw::find_hc_call_locked(g, call);
        if (!state) {
            return gdk::E_INVALIDARG_HR;
        }
        state->performed = true;
        state->status_code = 0u;
        state->network_error_code = mw::kMwOfflineHResult;
        state->platform_network_error_code = mw::kMwPlatformNetworkUnreachable;
        state->platform_network_error_message = "Offline mode (network unavailable)";
        state->response_string.clear();
        state->response_body.clear();
        state->response_headers.clear();
    }
    return mw::complete_async(async_block, gdk::S_OK_HR);
}

Pg6HResult pg6_mw_hc_http_call_request_set_url(Pg6HCCallHandle call,
                                               const char* method,
                                               const char* url) {
    if (method == nullptr || url == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    state->method = method;
    state->url = url;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_url(Pg6HCCallHandle call,
                                               const char** method,
                                               const char** url) {
    if (method == nullptr || url == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *method = nullptr;
        *url = nullptr;
        return gdk::E_INVALIDARG_HR;
    }
    *method = state->method.c_str();
    *url = state->url.c_str();
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_get_request_url(Pg6HCCallHandle call, const char** url) {
    if (url == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *url = nullptr;
        return gdk::E_INVALIDARG_HR;
    }
    *url = state->url.c_str();
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_set_request_body_bytes(Pg6HCCallHandle call,
                                                              const uint8_t* request_body_bytes,
                                                              uint32_t request_body_size) {
    if (request_body_size > 0 && request_body_bytes == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    if (request_body_size == 0) {
        state->request_body.clear();
    } else {
        state->request_body.assign(request_body_bytes, request_body_bytes + request_body_size);
    }
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_request_body_bytes(Pg6HCCallHandle call,
                                                              const uint8_t** request_body_bytes,
                                                              uint32_t* request_body_size) {
    if (request_body_bytes == nullptr || request_body_size == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *request_body_bytes = nullptr;
        *request_body_size = 0;
        return gdk::E_INVALIDARG_HR;
    }
    *request_body_size = static_cast<uint32_t>(state->request_body.size());
    *request_body_bytes = state->request_body.empty() ? nullptr : state->request_body.data();
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_set_request_body_string(Pg6HCCallHandle call,
                                                               const char* request_body_string) {
    if (request_body_string == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    const size_t len = std::strlen(request_body_string);
    return pg6_mw_hc_http_call_request_set_request_body_bytes(
        call,
        reinterpret_cast<const uint8_t*>(request_body_string),
        static_cast<uint32_t>(len));
}

Pg6HResult pg6_mw_hc_http_call_request_set_header(Pg6HCCallHandle call,
                                                  const char* header_name,
                                                  const char* header_value,
                                                  bool allow_tracing) {
    if (header_name == nullptr || header_value == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    for (auto& hdr : state->request_headers) {
        if (mw::ascii_iequals(hdr.name, header_name)) {
            hdr.value = header_value;
            hdr.allow_tracing = allow_tracing;
            return gdk::S_OK_HR;
        }
    }
    state->request_headers.push_back(mw::HttpHeaderEntry{header_name, header_value, allow_tracing});
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_num_headers(Pg6HCCallHandle call,
                                                       uint32_t* num_headers) {
    if (num_headers == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *num_headers = 0;
        return gdk::E_INVALIDARG_HR;
    }
    *num_headers = static_cast<uint32_t>(state->request_headers.size());
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_header_at_index(Pg6HCCallHandle call,
                                                           uint32_t header_index,
                                                           const char** header_name,
                                                           const char** header_value) {
    if (header_name == nullptr || header_value == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state || header_index >= state->request_headers.size()) {
        *header_name = nullptr;
        *header_value = nullptr;
        return gdk::E_INVALIDARG_HR;
    }
    *header_name = state->request_headers[header_index].name.c_str();
    *header_value = state->request_headers[header_index].value.c_str();
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_set_retry_allowed(Pg6HCCallHandle call, bool retry_allowed) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        g.default_retry_allowed = retry_allowed;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    state->retry_allowed = retry_allowed;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_retry_allowed(Pg6HCCallHandle call,
                                                         bool* retry_allowed) {
    if (retry_allowed == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        *retry_allowed = g.default_retry_allowed;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    *retry_allowed = state->retry_allowed;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_set_retry_cache_id(Pg6HCCallHandle call,
                                                          uint32_t retry_after_cache_id) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        g.default_retry_cache_id = retry_after_cache_id;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    state->retry_cache_id = retry_after_cache_id;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_retry_cache_id(Pg6HCCallHandle call,
                                                          uint32_t* retry_after_cache_id) {
    if (retry_after_cache_id == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        *retry_after_cache_id = g.default_retry_cache_id;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    *retry_after_cache_id = state->retry_cache_id;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_set_retry_delay(Pg6HCCallHandle call,
                                                       uint32_t retry_delay_in_seconds) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        g.default_retry_delay_sec = retry_delay_in_seconds;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    state->retry_delay_sec = retry_delay_in_seconds;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_retry_delay(Pg6HCCallHandle call,
                                                       uint32_t* retry_delay_in_seconds) {
    if (retry_delay_in_seconds == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        *retry_delay_in_seconds = g.default_retry_delay_sec;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    *retry_delay_in_seconds = state->retry_delay_sec;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_set_timeout(Pg6HCCallHandle call,
                                                   uint32_t timeout_in_seconds) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        g.default_timeout_sec = timeout_in_seconds;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    state->timeout_sec = timeout_in_seconds;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_timeout(Pg6HCCallHandle call,
                                                   uint32_t* timeout_in_seconds) {
    if (timeout_in_seconds == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        *timeout_in_seconds = g.default_timeout_sec;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    *timeout_in_seconds = state->timeout_sec;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_set_timeout_window(Pg6HCCallHandle call,
                                                          uint32_t timeout_window_in_seconds) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        g.default_timeout_window_sec = timeout_window_in_seconds;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    state->timeout_window_sec = timeout_window_in_seconds;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_request_get_timeout_window(Pg6HCCallHandle call,
                                                          uint32_t* timeout_window_in_seconds) {
    if (timeout_window_in_seconds == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (call == nullptr) {
        *timeout_window_in_seconds = g.default_timeout_window_sec;
        return gdk::S_OK_HR;
    }
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    *timeout_window_in_seconds = state->timeout_window_sec;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_header(Pg6HCCallHandle call,
                                                   const char* header_name,
                                                   const char** header_value) {
    if (header_name == nullptr || header_value == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *header_value = nullptr;
        return gdk::E_INVALIDARG_HR;
    }
    for (const auto& hdr : state->response_headers) {
        if (mw::ascii_iequals(hdr.name, header_name)) {
            *header_value = hdr.value.c_str();
            return gdk::S_OK_HR;
        }
    }
    *header_value = nullptr;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_header_at_index(Pg6HCCallHandle call,
                                                            uint32_t header_index,
                                                            const char** header_name,
                                                            const char** header_value) {
    if (header_name == nullptr || header_value == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state || header_index >= state->response_headers.size()) {
        *header_name = nullptr;
        *header_value = nullptr;
        return gdk::E_INVALIDARG_HR;
    }
    *header_name = state->response_headers[header_index].name.c_str();
    *header_value = state->response_headers[header_index].value.c_str();
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_num_headers(Pg6HCCallHandle call,
                                                        uint32_t* num_headers) {
    if (num_headers == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *num_headers = 0;
        return gdk::E_INVALIDARG_HR;
    }
    *num_headers = static_cast<uint32_t>(state->response_headers.size());
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_network_error_code(
    Pg6HCCallHandle call,
    Pg6HResult* network_error_code,
    uint32_t* platform_network_error_code) {
    if (network_error_code == nullptr || platform_network_error_code == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *network_error_code = gdk::E_INVALIDARG_HR;
        *platform_network_error_code = 0;
        return gdk::E_INVALIDARG_HR;
    }
    *network_error_code = state->network_error_code;
    *platform_network_error_code = state->platform_network_error_code;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_platform_network_error_message(
    Pg6HCCallHandle call,
    const char** platform_network_error_message) {
    if (platform_network_error_message == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *platform_network_error_message = nullptr;
        return gdk::E_INVALIDARG_HR;
    }
    *platform_network_error_message = state->platform_network_error_message.c_str();
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_response_body_bytes(Pg6HCCallHandle call,
                                                                size_t buffer_size,
                                                                uint8_t* buffer,
                                                                size_t* buffer_used) {
    if (buffer_used != nullptr) {
        *buffer_used = 0;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    const size_t avail = state->response_body.size();
    if (avail > 0) {
        if (buffer == nullptr) {
            return gdk::E_INVALIDARG_HR;
        }
        const size_t to_copy = std::min(buffer_size, avail);
        std::memcpy(buffer, state->response_body.data(), to_copy);
        if (buffer_used != nullptr) {
            *buffer_used = to_copy;
        }
    }
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_response_body_bytes_size(Pg6HCCallHandle call,
                                                                     size_t* buffer_size) {
    if (buffer_size == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *buffer_size = 0;
        return gdk::E_INVALIDARG_HR;
    }
    *buffer_size = state->response_body.size();
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_response_string(Pg6HCCallHandle call,
                                                            const char** response_string) {
    if (response_string == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *response_string = nullptr;
        return gdk::E_INVALIDARG_HR;
    }
    *response_string = state->response_string.c_str();
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_response_get_status_code(Pg6HCCallHandle call,
                                                        uint32_t* status_code) {
    if (status_code == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        *status_code = 0;
        return gdk::E_INVALIDARG_HR;
    }
    *status_code = state->status_code;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_http_call_set_tracing(Pg6HCCallHandle call, bool trace_call) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto state = mw::find_hc_call_locked(g, call);
    if (!state) {
        return gdk::E_INVALIDARG_HR;
    }
    state->trace_call = trace_call;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_hc_settings_get_trace_level(uint32_t* trace_level) {
    if (trace_level == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    *trace_level = g.trace_level;
    return gdk::S_OK_HR;
}

void pg6_mw_hc_trace_impl_message(const char*, uint32_t, const char*, ...) {
    // Offline no-op trace sink.
}

void pg6_mw_hc_trace_set_etw_enabled(bool enabled) {
    auto& g = mw::hc_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.etw_enabled = enabled;
}

// ---------------------------------------------------------------------------
// 2. Bink2Scarlett.dll (23 symbols)
// ---------------------------------------------------------------------------

Pg6HBink pg6_mw_bink_open_with_options(const char* name,
                                       const Pg6BinkOptions* options,
                                       uint32_t flags) {
    // There is no Bink decoder. A movie handed over in memory (BINKFROMMEMORY) is opened with the
    // dimensions, frame count and frame rate of its own header and plays out its real duration
    // (BinkWait paces it); its frame buffers stay blank: nothing is decoded into them. A movie by
    // file name, or memory without a valid Bink 2 header, is refused as Bink refuses a file it
    // cannot open: no handle (the title checks it).
    mw::BinkHeader h{};
    const bool from_memory = (flags & mw::kBinkFromMemory) != 0;
    if (name == nullptr || !from_memory || !mw::parse_bink2_header(name, &h)) {
        auto& g = mw::bink_globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        ++g.refused_opens;
        if (g.refused_opens <= 4) {
            std::fprintf(stderr, "Bink2: BinkOpenWithOptions(flags 0x%x): %s, movie not opened\n",
                         static_cast<unsigned>(flags),
                         from_memory ? "no Bink 2 header in the memory image" : "no decoder for movie files");
        }
        return nullptr;
    }
    auto* bink = new (std::nothrow) Pg6Bink{};
    if (bink == nullptr) {
        return nullptr;
    }
    bink->width = h.width;
    bink->height = h.height;
    bink->frames = h.frames;
    bink->frame_num = 1u;
    bink->last_frame_num = 0u;
    bink->frame_rate = h.frame_rate;
    bink->frame_rate_div = h.frame_rate_div;
    bink->read_error = 0u;
    bink->open_flags = flags | (options != nullptr ? options->flags : 0u);
    bink->size = h.file_size;
    bink->paused = 0;
    bink->will_loop = 1;
    bink->finished = 0;
    bink->async_in_progress = 0;
    bink->volume = 32768;
    bink->frame_buffers.total_frames = 2;
    bink->frame_buffers.ya_buffer_width = bink->width;
    bink->frame_buffers.ya_buffer_height = bink->height;
    bink->frame_buffers.crcb_buffer_width = bink->width / 2u;
    bink->frame_buffers.crcb_buffer_height = bink->height / 2u;
    bink->frame_buffers.frame_num = 0u;

    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.live_handles.insert(bink);
    g.clocks[bink] = mw::BinkClock{std::chrono::steady_clock::now(), {}, false};
    std::fprintf(stderr, "Bink2: opened an in-memory movie: %ux%u, %u frames at %u/%u fps (not decoded)\n",
                 h.width, h.height, h.frames, h.frame_rate, h.frame_rate_div);
    return bink;
}

void pg6_mw_bink_close(Pg6HBink bink) {
    if (bink == nullptr) {
        return;
    }
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.live_handles.find(bink);
    if (it != g.live_handles.end()) {
        g.live_handles.erase(it);
        g.clocks.erase(bink);
        delete bink;
    }
}

int32_t pg6_mw_bink_do_frame(Pg6HBink bink) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return 0;
    }
    if (bink->paused != 0) {
        return 0;
    }
    bink->last_frame_num = bink->frame_num;
    bink->frame_buffers.frame_num =
        (bink->frame_num > 0u) ? ((bink->frame_num - 1u) % 2u) : 0u;
    return 0;  // 0 = frame not skipped
}

int32_t pg6_mw_bink_do_frame_async_multi(Pg6HBink bink, uint32_t*, int32_t) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return 0;
    }
    if (bink->paused == 0) {
        bink->last_frame_num = bink->frame_num;
        bink->frame_buffers.frame_num =
            (bink->frame_num > 0u) ? ((bink->frame_num - 1u) % 2u) : 0u;
    }
    bink->async_in_progress = 1;
    return 1;
}

int32_t pg6_mw_bink_do_frame_async_wait(Pg6HBink bink, int64_t) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return 1;
    }
    bink->async_in_progress = 0;
    return 1;  // Complete immediately; never stalls playback loop.
}

void pg6_mw_bink_next_frame(Pg6HBink bink) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink) || bink->paused != 0) {
        return;
    }
    if (bink->frame_num < bink->frames) {
        ++bink->frame_num;
    } else if (bink->will_loop != 0) {
        bink->frame_num = 1u;
        mw::restart_clock_locked(g, bink);
    } else {
        bink->frame_num = bink->frames;
        bink->finished = 1;
    }
}

int32_t pg6_mw_bink_wait(Pg6HBink bink) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return 0;
    }
    if (bink->paused != 0) {
        return 1;
    }
    // The next frame is due once its time has come (real time at the movie's frame rate).
    const auto it = g.clocks.find(bink);
    if (it == g.clocks.end() || bink->frame_rate == 0) {
        return 0;
    }
    const double period = static_cast<double>(bink->frame_rate_div) / static_cast<double>(bink->frame_rate);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - it->second.start).count();
    const double due = static_cast<double>(bink->frame_num > 0 ? bink->frame_num - 1 : 0) * period;
    return elapsed >= due ? 0 : 1;
}

int32_t pg6_mw_bink_pause(Pg6HBink bink, int32_t pause) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return 0;
    }
    const int32_t now_paused = (pause != 0) ? 1 : 0;
    if (const auto it = g.clocks.find(bink); it != g.clocks.end() && now_paused != bink->paused) {
        const auto now = std::chrono::steady_clock::now();
        if (now_paused != 0) {
            it->second.paused_at = now;
        } else {
            it->second.start += now - it->second.paused_at;
        }
    }
    bink->paused = now_paused;
    return bink->paused;
}

void pg6_mw_bink_goto(Pg6HBink bink, uint32_t frame, int32_t) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return;
    }
    if (frame < 1u) {
        frame = 1u;
    } else if (frame > bink->frames) {
        frame = bink->frames;
    }
    bink->frame_num = frame;
    bink->finished = 0;
    mw::restart_clock_locked(g, bink);
}

uint32_t pg6_mw_bink_get_key_frame(Pg6HBink bink, uint32_t frame, int32_t) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return 1u;
    }
    if (frame < 1u) {
        return 1u;
    }
    if (frame > bink->frames) {
        return bink->frames;
    }
    return frame;
}

int32_t pg6_mw_bink_should_skip(Pg6HBink) {
    return 0;
}

void pg6_mw_bink_set_will_loop(Pg6HBink bink, int32_t onoff) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return;
    }
    bink->will_loop = (onoff != 0) ? 1 : 0;
}

void pg6_mw_bink_set_volume(Pg6HBink bink, uint32_t, int32_t volume) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return;
    }
    bink->volume = volume;
}

void pg6_mw_bink_get_frame_buffers_info(Pg6HBink bink, Pg6BinkFrameBuffers* set) {
    if (set == nullptr) {
        return;
    }
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        *set = Pg6BinkFrameBuffers{};
        return;
    }
    *set = bink->frame_buffers;
}

void pg6_mw_bink_register_frame_buffers(Pg6HBink bink, const Pg6BinkFrameBuffers* set) {
    if (set == nullptr) {
        return;
    }
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!mw::is_live_bink_locked(g, bink)) {
        return;
    }
    bink->frame_buffers = *set;
}

int32_t pg6_mw_bink_start_async_thread(int32_t thread_num, const void*) {
    if (thread_num < 0) {
        return 0;
    }
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.active_threads.insert(thread_num);
    return 1;
}

int32_t pg6_mw_bink_request_stop_async_threads_multi(uint32_t* threads, int32_t thread_count) {
    if (thread_count < 0) {
        return 0;
    }
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (threads != nullptr) {
        for (int32_t i = 0; i < thread_count; ++i) {
            g.active_threads.erase(static_cast<int32_t>(threads[i]));
        }
    }
    return 1;
}

int32_t pg6_mw_bink_wait_stop_async_threads_multi(uint32_t* threads, int32_t thread_count) {
    return pg6_mw_bink_request_stop_async_threads_multi(threads, thread_count);
}

void pg6_mw_bink_set_memory(void* alloc_fn, void* free_fn) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.alloc_fn = alloc_fn;
    g.free_fn = free_fn;
}

void pg6_mw_bink_set_os_file_callbacks(void* file_open_fn) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.os_file_open_fn = file_open_fn;
}

int32_t pg6_mw_bink_set_sound_system(void* open_fn, uintptr_t param) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.sound_open_fn = open_fn;
    g.sound_param = param;
    return 1;
}

int32_t pg6_mw_bink_get_platform_info(int32_t, void* output) {
    if (output == nullptr) {
        return 0;
    }
    std::memset(output, 0, sizeof(uint64_t));
    return 1;
}

void pg6_mw_bink_free_globals(void) {
    auto& g = mw::bink_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.alloc_fn = nullptr;
    g.free_fn = nullptr;
    g.os_file_open_fn = nullptr;
    g.sound_open_fn = nullptr;
    g.sound_param = 0;
    g.active_threads.clear();
}

// ---------------------------------------------------------------------------
// 3. AcpHal.dll (9 symbols)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_acp_hal_create(void** acp_hal) {
    if (acp_hal == nullptr) {
        return gdk::E_POINTER_HR;
    }
    void* obj = mw::create_com_stub(1u);
    if (obj == nullptr) {
        *acp_hal = nullptr;
        return gdk::E_OUTOFMEMORY_HR;
    }
    *acp_hal = obj;
    return gdk::S_OK_HR;
}

namespace {
// The request AcpHalAllocateShapeContexts takes: six 32-bit context counts, then one pointer per
// kind that the call fills with a zeroed array of that many contexts. Layout from the executable:
// it zeroes counts 0-3 and 5, sets count 4, and after the call clears count 4 x 64 bytes at the
// pointer at +0x38 (0x18 + 8 * 4).
struct ShapeContextRequest {
    uint32_t counts[6];
    uint64_t arrays[6];  // guest addresses, written by the call
};
static_assert(sizeof(ShapeContextRequest) == 0x48);
// Context size per kind; 0 = not established (no title has asked for that kind yet). Kind 4 is
// XMA (the only kind this title asks for: 350 contexts at 0x1443AEC40): 64-byte XMA contexts.
constexpr uint32_t kShapeContextKindXma = 4;
constexpr uint32_t kShapeContextBytes[6] = {0, 0, 0, 0, 64, 0};
constexpr uint32_t kShapeContextAlignment = 256;
}  // namespace

Pg6HResult pg6_mw_acp_hal_allocate_shape_contexts(void* contexts) {
    if (contexts == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    ShapeContextRequest request{};
    std::memcpy(&request, contexts, sizeof(request));
    for (uint32_t kind = 0; kind < 6; ++kind) {
        request.arrays[kind] = 0;
        if (request.counts[kind] != 0 && kShapeContextBytes[kind] == 0) {
            std::memcpy(contexts, &request, sizeof(request));
            playgta6::os::log("AcpHalAllocateShapeContexts: " + std::to_string(request.counts[kind]) +
                              " contexts of kind " + std::to_string(kind) +
                              " requested; the size of that kind is not established, returning E_NOTIMPL");
            return gdk::E_NOTIMPL_HR;
        }
    }
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    for (uint32_t kind = 0; kind < 6; ++kind) {
        if (request.counts[kind] == 0) {
            continue;
        }
        const uint64_t bytes = uint64_t{request.counts[kind]} * kShapeContextBytes[kind];
        const uint64_t va = heap.alloc(bytes, kShapeContextAlignment, playgta6::os::HeapFamily::Other,
                                       playgta6::os::kHeapZeroMemory);
        if (va == 0) {
            for (uint32_t k = 0; k < kind; ++k) {
                if (request.arrays[k] != 0) {
                    heap.free(request.arrays[k]);
                    g.shape_context_arrays.pop_back();
                }
            }
            return gdk::E_OUTOFMEMORY_HR;
        }
        request.arrays[kind] = va;
        g.shape_context_arrays.push_back(va);
        if (kind == kShapeContextKindXma) {
            g.xma_contexts = va;
            g.xma_context_count = request.counts[kind];
        }
    }
    std::memcpy(contexts, &request, sizeof(request));
    ++g.shape_context_allocs;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_acp_hal_release_shape_contexts(void) {
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    for (uint64_t va : g.shape_context_arrays) {
        playgta6::os::default_guest_heap().free(va);
    }
    g.shape_context_arrays.clear();
    g.xma_contexts = 0;
    g.xma_context_count = 0;
    if (g.shape_context_allocs > 0) {
        --g.shape_context_allocs;
    }
    return gdk::S_OK_HR;
}

uint64_t pg6_mw_xma_contexts(uint32_t* count) {
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (count != nullptr) {
        *count = g.xma_context_count;
    }
    return g.xma_contexts;
}

Pg6XmaContextCensus pg6_mw_xma_context_census(void) {
    Pg6XmaContextCensus c;
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (g.xma_contexts == 0) {
        return c;
    }
    const auto* base = static_cast<const uint8_t*>(
        playgta6::os::default_guest_heap().to_host(g.xma_contexts));
    if (base == nullptr) {
        return c;
    }
    c.contexts = g.xma_context_count;
    for (uint32_t i = 0; i < g.xma_context_count; ++i) {
        const uint8_t* ctx = base + size_t{i} * kShapeContextBytes[kShapeContextKindXma];
        uint32_t d0 = 0, d9 = 0;
        std::memcpy(&d0, ctx, 4);
        std::memcpy(&d9, ctx + 0x24, 4);
        if (std::any_of(ctx, ctx + kShapeContextBytes[kShapeContextKindXma], [](uint8_t v) { return v != 0; })) {
            ++c.touched;
        }
        if ((d0 & (3u << 20)) != 0) {
            ++c.input_valid;
        }
        if ((d0 >> 27) != (d9 & 0x1Fu)) {
            ++c.output_pending;
        }
    }
    return c;
}

Pg6HResult pg6_mw_apu_create_heap(uint32_t cached_size_in_bytes,
                                  uint32_t non_cached_size_in_bytes) {
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.heap_created = true;
    g.cached_heap_size = cached_size_in_bytes;
    g.non_cached_heap_size = non_cached_size_in_bytes;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_apu_alloc(void** virtual_address,
                            Pg6ApuAddress* physical_address,
                            uint32_t size_in_bytes,
                            uint32_t alignment_in_bytes,
                            uint32_t flags) {
    if (virtual_address == nullptr || size_in_bytes == 0u) {
        return gdk::E_INVALIDARG_HR;
    }
    *virtual_address = nullptr;
    if (physical_address != nullptr) {
        *physical_address = 0u;
    }

    const uint32_t align = mw::normalize_alignment(alignment_in_bytes);

    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);

    const uint64_t next_aligned =
        (static_cast<uint64_t>(g.next_apu_addr) + (static_cast<uint64_t>(align) - 1u)) &
        ~(static_cast<uint64_t>(align) - 1u);
    if (next_aligned + size_in_bytes > 0xFFFF0000ULL) {
        return gdk::E_OUTOFMEMORY_HR;
    }

    // Guest memory (the title reads and writes it), zero-filled like the console's APU heap.
    playgta6::os::GuestHeap& heap = playgta6::os::default_guest_heap();
    const uint64_t va = heap.alloc(size_in_bytes, align, playgta6::os::HeapFamily::Other,
                                   playgta6::os::kHeapZeroMemory);
    if (va == 0) {
        return gdk::E_OUTOFMEMORY_HR;
    }
    const Pg6ApuAddress apu_addr = static_cast<Pg6ApuAddress>(next_aligned);
    g.next_apu_addr = static_cast<Pg6ApuAddress>(next_aligned + size_in_bytes);

    mw::ApuBlock block{};
    block.va = va;
    block.aligned_ptr = heap.to_host(va);
    block.size = size_in_bytes;
    block.alignment = align;
    block.flags = flags;
    block.apu_addr = apu_addr;
    g.blocks.push_back(std::move(block));

    // Stored into guest memory: the guest address of the block.
    *virtual_address = reinterpret_cast<void*>(static_cast<uintptr_t>(va));
    if (physical_address != nullptr) {
        *physical_address = apu_addr;
    }
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_apu_free(void* virtual_address) {
    if (virtual_address == nullptr) {
        return gdk::S_OK_HR;
    }
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    for (auto it = g.blocks.begin(); it != g.blocks.end(); ++it) {
        if (it->aligned_ptr == virtual_address) {
            playgta6::os::default_guest_heap().free(it->va);
            g.blocks.erase(it);
            return gdk::S_OK_HR;
        }
    }
    return gdk::E_INVALIDARG_HR;
}

int32_t pg6_mw_apu_is_virtual_address_valid(const void* virtual_address,
                                            uint32_t physical_address_in_bytes) {
    if (virtual_address == nullptr) {
        return 0;
    }
    const uintptr_t query = reinterpret_cast<uintptr_t>(virtual_address);
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    for (const auto& block : g.blocks) {
        const uintptr_t start = reinterpret_cast<uintptr_t>(block.aligned_ptr);
        const uintptr_t end = start + static_cast<uintptr_t>(block.size);
        if (query >= start && query < end) {
            const uintptr_t span = static_cast<uintptr_t>(physical_address_in_bytes);
            if (span == 0u || (end - query) >= span) {
                return 1;
            }
        }
    }
    return 0;
}

void* pg6_mw_apu_map_apu_address(Pg6ApuAddress apu_address) {
    if (apu_address == 0u) {
        return nullptr;
    }
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    for (const auto& block : g.blocks) {
        const uint64_t start = block.apu_addr;
        const uint64_t end = start + block.size;
        if (apu_address >= start && apu_address < end) {
            const uint64_t offset = apu_address - start;
            return block.aligned_ptr + offset;
        }
    }
    return nullptr;
}

Pg6ApuAddress pg6_mw_apu_map_virtual_address(const void* virtual_address) {
    if (virtual_address == nullptr) {
        return 0u;
    }
    const uintptr_t query = reinterpret_cast<uintptr_t>(virtual_address);
    auto& g = mw::apu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    for (const auto& block : g.blocks) {
        const uintptr_t start = reinterpret_cast<uintptr_t>(block.aligned_ptr);
        const uintptr_t end = start + static_cast<uintptr_t>(block.size);
        if (query >= start && query < end) {
            const uint32_t offset = static_cast<uint32_t>(query - start);
            return block.apu_addr + offset;
        }
    }
    return 0u;
}

// ---------------------------------------------------------------------------
// 4. xapu.dll (5 symbols)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_xapu_connect(const void*, Pg6XApuHandle* handle) {
    if (handle == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::xapu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    const uint64_t id = g.next_handle;
    g.next_handle += 0x10ULL;
    g.sessions.emplace(id, std::make_unique<mw::XApuSessionState>());
    *handle = mw::to_xapu_handle(id);
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_xapu_disconnect(Pg6XApuHandle handle) {
    auto& g = mw::xapu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.sessions.find(mw::from_xapu_handle(handle));
    if (it == g.sessions.end()) {
        return gdk::E_INVALIDARG_HR;
    }
    g.sessions.erase(it);
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_xapu_enqueue_command(Pg6XApuHandle handle,
                                       const void* command,
                                       uint32_t command_size) {
    if (command_size > 0 && command == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::xapu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.sessions.find(mw::from_xapu_handle(handle));
    if (it == g.sessions.end()) {
        return gdk::E_INVALIDARG_HR;
    }
    std::vector<uint8_t> result;
    if (command_size > 0) {
        const auto* bytes = static_cast<const uint8_t*>(command);
        result.assign(bytes, bytes + command_size);
    }
    it->second->results.push_back(std::move(result));
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_xapu_dequeue_result(Pg6XApuHandle handle,
                                      void* result_buffer,
                                      uint32_t result_buffer_size,
                                      uint32_t* bytes_written) {
    if (bytes_written != nullptr) {
        *bytes_written = 0u;
    }
    auto& g = mw::xapu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.sessions.find(mw::from_xapu_handle(handle));
    if (it == g.sessions.end()) {
        return gdk::E_INVALIDARG_HR;
    }
    if (it->second->results.empty()) {
        return mw::kMwSFalseHResult;
    }
    const std::vector<uint8_t> front = std::move(it->second->results.front());
    it->second->results.pop_front();
    const uint32_t to_copy = static_cast<uint32_t>(
        std::min<size_t>(front.size(), static_cast<size_t>(result_buffer_size)));
    if (to_copy > 0 && result_buffer != nullptr) {
        std::memcpy(result_buffer, front.data(), to_copy);
    }
    if (bytes_written != nullptr) {
        *bytes_written = to_copy;
    }
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_xapu_get_result_count(Pg6XApuHandle handle, uint32_t* result_count) {
    if (result_count == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::xapu_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.sessions.find(mw::from_xapu_handle(handle));
    if (it == g.sessions.end()) {
        *result_count = 0u;
        return gdk::E_INVALIDARG_HR;
    }
    *result_count = static_cast<uint32_t>(it->second->results.size());
    return gdk::S_OK_HR;
}

// ---------------------------------------------------------------------------
// 5. chad.dll (5 symbols)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_xdsp_connect_with_maximum_stream_limit(uint32_t max_streams,
                                                         Pg6XDspHandle* handle) {
    if (handle == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::xdsp_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    const uint64_t id = g.next_handle;
    g.next_handle += 0x10ULL;
    auto state = std::make_unique<mw::XDspSessionState>();
    state->max_streams = max_streams;
    g.sessions.emplace(id, std::move(state));
    *handle = mw::to_xdsp_handle(id);
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_xdsp_disconnect(Pg6XDspHandle handle) {
    auto& g = mw::xdsp_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.sessions.find(mw::from_xdsp_handle(handle));
    if (it == g.sessions.end()) {
        return gdk::E_INVALIDARG_HR;
    }
    g.sessions.erase(it);
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_xdsp_activate(Pg6XDspHandle handle) {
    auto& g = mw::xdsp_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.sessions.find(mw::from_xdsp_handle(handle));
    if (it == g.sessions.end()) {
        return gdk::E_INVALIDARG_HR;
    }
    it->second->active = true;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_xdsp_deactivate(Pg6XDspHandle handle) {
    auto& g = mw::xdsp_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.sessions.find(mw::from_xdsp_handle(handle));
    if (it == g.sessions.end()) {
        return gdk::E_INVALIDARG_HR;
    }
    it->second->active = false;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_xdsp_submit_command(Pg6XDspHandle handle,
                                      const void* command,
                                      uint32_t command_size) {
    if (command_size > 0 && command == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    auto& g = mw::xdsp_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.sessions.find(mw::from_xdsp_handle(handle));
    if (it == g.sessions.end()) {
        return gdk::E_INVALIDARG_HR;
    }
    ++it->second->submitted_commands;
    return gdk::S_OK_HR;
}

// ---------------------------------------------------------------------------
// 6. amd_fidelityfx_loader_xbox.dll (4 symbols)
// ---------------------------------------------------------------------------

uint32_t pg6_mw_ffx_create_context(Pg6FfxContext* context,
                                   const Pg6FfxApiHeader* desc,
                                   const void*) {
    if (context == nullptr) {
        return mw::kFfxReturnErrorParameter;
    }
    auto& g = mw::ffx_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    const uint64_t id = g.next_handle;
    g.next_handle += 0x10ULL;
    mw::FfxContextState state{};
    if (desc != nullptr) {
        state.desc_type = desc->type;
    }
    g.contexts.emplace(id, state);
    *context = mw::to_ffx_handle(id);
    return mw::kFfxReturnOk;
}

uint32_t pg6_mw_ffx_destroy_context(Pg6FfxContext* context, const void*) {
    if (context == nullptr || *context == nullptr) {
        return mw::kFfxReturnErrorParameter;
    }
    auto& g = mw::ffx_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.contexts.find(mw::from_ffx_handle(*context));
    if (it == g.contexts.end()) {
        return mw::kFfxReturnErrorParameter;
    }
    g.contexts.erase(it);
    *context = nullptr;
    return mw::kFfxReturnOk;
}

uint32_t pg6_mw_ffx_dispatch(Pg6FfxContext* context, const Pg6FfxApiHeader* desc) {
    if (context == nullptr || *context == nullptr || desc == nullptr) {
        return mw::kFfxReturnErrorParameter;
    }
    auto& g = mw::ffx_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    auto it = g.contexts.find(mw::from_ffx_handle(*context));
    if (it == g.contexts.end()) {
        return mw::kFfxReturnErrorParameter;
    }
    ++it->second.dispatch_count;
    return mw::kFfxReturnOk;
}

// ffxQuery answers only the queries whose result is a pure function of the descriptor; every
// other query type fails (kFfxReturnError) and writes nothing, so no caller reads an output that
// was never produced.
//   0x00010004 upscale GetJitterPhaseCount {header; uint32 renderWidth; uint32 displayWidth;
//              int32* pOutPhaseCount}: the FSR rule int(8 * (displayWidth / renderWidth)^2).
uint32_t pg6_mw_ffx_query(Pg6FfxContext* context, Pg6FfxApiHeader* desc) {
    if (desc == nullptr) {
        return mw::kFfxReturnErrorParameter;
    }
    if (context != nullptr && *context != nullptr) {
        auto& g = mw::ffx_globals();
        std::lock_guard<std::mutex> lock(g.mutex);
        if (g.contexts.find(mw::from_ffx_handle(*context)) == g.contexts.end()) {
            return mw::kFfxReturnErrorParameter;
        }
    }
    if (desc->type == kFfxQueryUpscaleGetJitterPhaseCount) {
        auto* query = reinterpret_cast<Pg6FfxQueryJitterPhaseCount*>(desc);
        if (query->render_width == 0 || query->out_phase_count == nullptr) {
            return mw::kFfxReturnErrorParameter;
        }
        const float ratio = static_cast<float>(query->display_width) / static_cast<float>(query->render_width);
        *query->out_phase_count = static_cast<int32_t>(8.0f * ratio * ratio);
        return mw::kFfxReturnOk;
    }
    return mw::kFfxReturnError;
}

// ---------------------------------------------------------------------------
// 7. XboxBvhBuilder.dll (7 symbols)
// ---------------------------------------------------------------------------

void* pg6_mw_bvh_builder_ctor(void* self, void*, bool) {
    return self;
}

void pg6_mw_bvh_builder_dtor(void*) {}

void pg6_mw_bvh_builder_build_batched_2x(void*, void*, uint32_t, const void*) {}

void pg6_mw_bvh_builder_copy(void*, void*, uint64_t, uint64_t, uint32_t) {}

// The runtime builds no acceleration structures (the build, copy and rehydrate entry points
// write nothing), so any size is an upper bound for what its build writes. Zero is not usable:
// the title allocates the result buffer from the reported size and treats a null allocation as
// fatal (int3). The smallest valid D3D12 size is one alignment unit
// (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, 256 bytes) for each buffer.
void pg6_mw_bvh_builder_get_prebuild_info(void*,
                                          const void*,
                                          const void*,
                                          Pg6D3D12RaytracingPrebuildInfo* out_info) {
    constexpr uint64_t kMinimumSize = 256u;
    if (out_info != nullptr) {
        out_info->result_data_max_size_in_bytes = kMinimumSize;
        out_info->scratch_data_size_in_bytes = kMinimumSize;
        out_info->update_scratch_data_size_in_bytes = kMinimumSize;
    }
}

Pg6HResult pg6_mw_bvh_builder_rehydrate_batched_2x(void*,
                                                   void*,
                                                   uint32_t,
                                                   const void*,
                                                   const uint64_t*) {
    return gdk::S_OK_HR;
}

void pg6_mw_bvh_builder_set_64bit_morton_codes(void*, bool) {}

// ---------------------------------------------------------------------------
// 8. MFPlat.DLL (6 symbols) & 9. MFReadWrite.dll (1 symbol)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_mf_startup(uint32_t, uint32_t) {
    auto& g = mw::mf_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    ++g.startup_count;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_mf_shutdown(void) {
    auto& g = mw::mf_globals();
    std::lock_guard<std::mutex> lock(g.mutex);
    if (g.startup_count > 0) {
        --g.startup_count;
    }
    return gdk::S_OK_HR;
}

int64_t pg6_mw_mf_get_system_time(void) {
    auto& g = mw::mf_globals();
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - g.epoch).count();
    // MFTIME is measured in 100-nanosecond intervals (offset from 1 so callers never see 0).
    return (elapsed_ns / 100) + 1;
}

Pg6HResult pg6_mw_mf_create_attributes(void** out_attributes, uint32_t) {
    if (out_attributes == nullptr) {
        return gdk::E_POINTER_HR;
    }
    void* obj = mw::create_com_stub(2u);
    if (obj == nullptr) {
        *out_attributes = nullptr;
        return gdk::E_OUTOFMEMORY_HR;
    }
    *out_attributes = obj;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_mf_create_media_type(void** out_media_type) {
    if (out_media_type == nullptr) {
        return gdk::E_POINTER_HR;
    }
    void* obj = mw::create_com_stub(3u);
    if (obj == nullptr) {
        *out_media_type = nullptr;
        return gdk::E_OUTOFMEMORY_HR;
    }
    *out_media_type = obj;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_mf_create_source_resolver(void** out_source_resolver) {
    if (out_source_resolver == nullptr) {
        return gdk::E_POINTER_HR;
    }
    void* obj = mw::create_com_stub(4u);
    if (obj == nullptr) {
        *out_source_resolver = nullptr;
        return gdk::E_OUTOFMEMORY_HR;
    }
    *out_source_resolver = obj;
    return gdk::S_OK_HR;
}

Pg6HResult pg6_mw_mf_create_source_reader_from_media_source(void* media_source,
                                                            void*,
                                                            void** out_source_reader) {
    if (out_source_reader == nullptr) {
        return gdk::E_POINTER_HR;
    }
    *out_source_reader = nullptr;
    if (media_source == nullptr) {
        return gdk::E_INVALIDARG_HR;
    }
    void* obj = mw::create_com_stub(5u);
    if (obj == nullptr) {
        return gdk::E_OUTOFMEMORY_HR;
    }
    *out_source_reader = obj;
    return gdk::S_OK_HR;
}

}  // extern "C"
