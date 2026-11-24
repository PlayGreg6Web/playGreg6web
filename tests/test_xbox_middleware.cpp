#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "os/guest_heap.h"
#include "os/win32_loader.h"
#include "os/xasync_provider.h"
#include "os/xbox_middleware.h"
#include "os/xgameruntime.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xbox_middleware.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os;
using namespace playgta6::os::middleware;
using gdk::E_INVALIDARG_HR;
using gdk::E_POINTER_HR;
using gdk::S_OK_HR;

struct AsyncProbe {
    std::atomic<int> calls{0};
    std::atomic<Pg6HResult> status{0};
};

void async_probe_callback(Pg6XAsyncBlock* block) {
    auto* probe = static_cast<AsyncProbe*>(block->context);
    probe->status.store(pg6_gdk_xasync_get_status(block, 0));
    probe->calls.fetch_add(1);
}

Pg6XTaskQueueHandle make_manual_queue() {
    Pg6XTaskQueueHandle queue = nullptr;
    CHECK(gdk::XTaskQueueCreate(gdk::XTaskQueueDispatchMode::Manual,
                                gdk::XTaskQueueDispatchMode::Manual,
                                &queue) == S_OK_HR);
    return queue;
}

// ---------------------------------------------------------------------------
// 1. Registration & GetProcAddress across all 9 DLLs (98 symbols)
// ---------------------------------------------------------------------------

void test_registration_and_lookup() {
    const uint32_t count = pg6_xbox_middleware_register_all();
    CHECK(count == 98u);
    // Idempotent second call.
    CHECK(pg6_xbox_middleware_register_all() == 98u);

    struct DllSymbols {
        const char* dll;
        std::vector<const char*> symbols;
    };

    const std::vector<DllSymbols> expected = {
        {"libHttpClient.GDK.dll",
         {
             "HCCleanupAsync",
             "HCHttpCallCloseHandle",
             "HCHttpCallCreate",
             "HCHttpCallDuplicateHandle",
             "HCHttpCallGetRequestUrl",
             "HCHttpCallPerformAsync",
             "HCHttpCallRequestGetHeaderAtIndex",
             "HCHttpCallRequestGetNumHeaders",
             "HCHttpCallRequestGetRequestBodyBytes",
             "HCHttpCallRequestGetRetryAllowed",
             "HCHttpCallRequestGetRetryCacheId",
             "HCHttpCallRequestGetRetryDelay",
             "HCHttpCallRequestGetTimeout",
             "HCHttpCallRequestGetTimeoutWindow",
             "HCHttpCallRequestGetUrl",
             "HCHttpCallRequestSetHeader",
             "HCHttpCallRequestSetRequestBodyBytes",
             "HCHttpCallRequestSetRequestBodyString",
             "HCHttpCallRequestSetRetryAllowed",
             "HCHttpCallRequestSetRetryCacheId",
             "HCHttpCallRequestSetRetryDelay",
             "HCHttpCallRequestSetTimeout",
             "HCHttpCallRequestSetTimeoutWindow",
             "HCHttpCallRequestSetUrl",
             "HCHttpCallResponseGetHeader",
             "HCHttpCallResponseGetHeaderAtIndex",
             "HCHttpCallResponseGetNetworkErrorCode",
             "HCHttpCallResponseGetNumHeaders",
             "HCHttpCallResponseGetPlatformNetworkErrorMessage",
             "HCHttpCallResponseGetResponseBodyBytes",
             "HCHttpCallResponseGetResponseBodyBytesSize",
             "HCHttpCallResponseGetResponseString",
             "HCHttpCallResponseGetStatusCode",
             "HCHttpCallSetTracing",
             "HCInitialize",
             "HCSettingsGetTraceLevel",
             "HCTraceImplMessage",
             "HCTraceSetEtwEnabled",
         }},
        {"Bink2Scarlett.dll",
         {
             "BinkClose",
             "BinkDoFrame",
             "BinkDoFrameAsyncMulti",
             "BinkDoFrameAsyncWait",
             "BinkFreeGlobals",
             "BinkGetFrameBuffersInfo",
             "BinkGetKeyFrame",
             "BinkGetPlatformInfo",
             "BinkGoto",
             "BinkNextFrame",
             "BinkOpenWithOptions",
             "BinkPause",
             "BinkRegisterFrameBuffers",
             "BinkRequestStopAsyncThreadsMulti",
             "BinkSetMemory",
             "BinkSetOSFileCallbacks",
             "BinkSetSoundSystem",
             "BinkSetVolume",
             "BinkSetWillLoop",
             "BinkShouldSkip",
             "BinkStartAsyncThread",
             "BinkWait",
             "BinkWaitStopAsyncThreadsMulti",
         }},
        {"AcpHal.dll",
         {
             "AcpHalAllocateShapeContexts",
             "AcpHalCreate",
             "AcpHalReleaseShapeContexts",
             "ApuAlloc",
             "ApuCreateHeap",
             "ApuFree",
             "ApuIsVirtualAddressValid",
             "ApuMapApuAddress",
             "ApuMapVirtualAddress",
         }},
        {"xapu.dll",
         {
             "XApuConnect",
             "XApuDequeueResult",
             "XApuDisconnect",
             "XApuEnqueueCommand",
             "XApuGetResultCount",
         }},
        {"chad.dll",
         {
             "XDspActivate",
             "XDspConnectWithMaximumStreamLimit",
             "XDspDeactivate",
             "XDspDisconnect",
             "XDspSubmitCommand",
         }},
        {"amd_fidelityfx_loader_xbox.dll",
         {
             "ffxCreateContext",
             "ffxDestroyContext",
             "ffxDispatch",
             "ffxQuery",
         }},
        {"XboxBvhBuilder.dll",
         {
             "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z",
             "??1XboxBvhBuilder@@QEAA@XZ",
             "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z",
             "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z",
             "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z",
             "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z",
             "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z",
         }},
        {"MFPlat.DLL",
         {
             "MFCreateAttributes",
             "MFCreateMediaType",
             "MFCreateSourceResolver",
             "MFGetSystemTime",
             "MFShutdown",
             "MFStartup",
         }},
        {"MFReadWrite.dll",
         {
             "MFCreateSourceReaderFromMediaSource",
         }},
    };

    uint32_t total_checked = 0;
    for (const auto& entry : expected) {
        const win32::Handle hmod = win32::GetModuleHandleA(entry.dll);
        CHECK(hmod != 0);
        for (const char* sym : entry.symbols) {
            const win32::ProcAddress via_gpa = win32::GetProcAddressByName(hmod, sym);
            const win32::ProcAddress via_resolve = win32::resolve_import(entry.dll, sym);
            CHECK(via_gpa != 0);
            CHECK(via_gpa == via_resolve);
            ++total_checked;
        }
    }
    CHECK(total_checked == 98u);
}

// ---------------------------------------------------------------------------
// 2. libHttpClient.GDK.dll lifecycle
// ---------------------------------------------------------------------------

void test_libhttpclient() {
    CHECK(pg6_mw_hc_initialize(nullptr) == S_OK_HR);

    uint32_t trace_level = 99u;
    CHECK(pg6_mw_hc_settings_get_trace_level(&trace_level) == S_OK_HR);
    CHECK(trace_level == 0u);
    pg6_mw_hc_trace_set_etw_enabled(false);
    pg6_mw_hc_trace_impl_message("HTTP", 1u, "msg %d", 42);

    Pg6HCCallHandle call = nullptr;
    CHECK(pg6_mw_hc_http_call_create(&call) == S_OK_HR);
    CHECK(call != nullptr);

    // Duplicate handle reference counting.
    Pg6HCCallHandle dup = pg6_mw_hc_http_call_duplicate_handle(call);
    CHECK(dup == call);

    // URL + method.
    CHECK(pg6_mw_hc_http_call_request_set_url(call, "POST", "https://example.invalid/api/v1") ==
          S_OK_HR);
    const char* method = nullptr;
    const char* url = nullptr;
    CHECK(pg6_mw_hc_http_call_request_get_url(call, &method, &url) == S_OK_HR);
    CHECK(std::strcmp(method, "POST") == 0);
    CHECK(std::strcmp(url, "https://example.invalid/api/v1") == 0);

    const char* url_only = nullptr;
    CHECK(pg6_mw_hc_http_call_get_request_url(call, &url_only) == S_OK_HR);
    CHECK(std::strcmp(url_only, "https://example.invalid/api/v1") == 0);

    // Headers (including case-insensitive overwrite).
    CHECK(pg6_mw_hc_http_call_request_set_header(call, "Content-Type", "application/json", true) ==
          S_OK_HR);
    CHECK(pg6_mw_hc_http_call_request_set_header(call, "X-Test", "one", false) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_request_set_header(call, "x-test", "two", true) == S_OK_HR);

    uint32_t num_headers = 0;
    CHECK(pg6_mw_hc_http_call_request_get_num_headers(call, &num_headers) == S_OK_HR);
    CHECK(num_headers == 2u);

    const char* hdr_name = nullptr;
    const char* hdr_val = nullptr;
    CHECK(pg6_mw_hc_http_call_request_get_header_at_index(call, 1u, &hdr_name, &hdr_val) ==
          S_OK_HR);
    CHECK(std::strcmp(hdr_val, "two") == 0);
    CHECK(pg6_mw_hc_http_call_request_get_header_at_index(call, 2u, &hdr_name, &hdr_val) ==
          E_INVALIDARG_HR);

    // Request body string and bytes.
    CHECK(pg6_mw_hc_http_call_request_set_request_body_string(call, "{\"ok\":true}") == S_OK_HR);
    const uint8_t* body_ptr = nullptr;
    uint32_t body_len = 0;
    CHECK(pg6_mw_hc_http_call_request_get_request_body_bytes(call, &body_ptr, &body_len) ==
          S_OK_HR);
    CHECK(body_len == 11u);
    CHECK(body_ptr != nullptr && std::memcmp(body_ptr, "{\"ok\":true}", 11) == 0);

    // Retry and timeout configuration.
    CHECK(pg6_mw_hc_http_call_request_set_retry_allowed(call, false) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_request_set_retry_cache_id(call, 7u) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_request_set_retry_delay(call, 5u) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_request_set_timeout(call, 15u) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_request_set_timeout_window(call, 60u) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_set_tracing(call, false) == S_OK_HR);

    bool retry_allowed = true;
    uint32_t cache_id = 0, retry_delay = 0, timeout = 0, timeout_window = 0;
    CHECK(pg6_mw_hc_http_call_request_get_retry_allowed(call, &retry_allowed) == S_OK_HR);
    CHECK(!retry_allowed);
    CHECK(pg6_mw_hc_http_call_request_get_retry_cache_id(call, &cache_id) == S_OK_HR);
    CHECK(cache_id == 7u);
    CHECK(pg6_mw_hc_http_call_request_get_retry_delay(call, &retry_delay) == S_OK_HR);
    CHECK(retry_delay == 5u);
    CHECK(pg6_mw_hc_http_call_request_get_timeout(call, &timeout) == S_OK_HR);
    CHECK(timeout == 15u);
    CHECK(pg6_mw_hc_http_call_request_get_timeout_window(call, &timeout_window) == S_OK_HR);
    CHECK(timeout_window == 60u);

    // Offline PerformAsync completes cleanly through XTaskQueue without hanging.
    Pg6XTaskQueueHandle queue = make_manual_queue();
    AsyncProbe probe;
    Pg6XAsyncBlock async{};
    async.queue = queue;
    async.context = &probe;
    async.callback = &async_probe_callback;

    CHECK(pg6_mw_hc_http_call_perform_async(call, &async) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 1, 0) == 1);
    CHECK(probe.calls.load() == 1);
    CHECK(probe.status.load() == S_OK_HR);

    // Inspect offline response fields.
    uint32_t status_code = 200u;
    CHECK(pg6_mw_hc_http_call_response_get_status_code(call, &status_code) == S_OK_HR);
    CHECK(status_code == 0u);

    Pg6HResult net_err = S_OK_HR;
    uint32_t plat_err = 0u;
    CHECK(pg6_mw_hc_http_call_response_get_network_error_code(call, &net_err, &plat_err) ==
          S_OK_HR);
    CHECK(net_err == kMwOfflineHResult);
    CHECK(plat_err == kMwPlatformNetworkUnreachable);

    const char* err_msg = nullptr;
    CHECK(pg6_mw_hc_http_call_response_get_platform_network_error_message(call, &err_msg) ==
          S_OK_HR);
    CHECK(err_msg != nullptr && std::strlen(err_msg) > 0);

    const char* resp_str = nullptr;
    CHECK(pg6_mw_hc_http_call_response_get_response_string(call, &resp_str) == S_OK_HR);
    CHECK(resp_str != nullptr && resp_str[0] == '\0');

    size_t resp_size = 99u;
    CHECK(pg6_mw_hc_http_call_response_get_response_body_bytes_size(call, &resp_size) == S_OK_HR);
    CHECK(resp_size == 0u);

    uint8_t scratch[16]{};
    size_t resp_used = 99u;
    CHECK(pg6_mw_hc_http_call_response_get_response_body_bytes(
              call, sizeof(scratch), scratch, &resp_used) == S_OK_HR);
    CHECK(resp_used == 0u);

    uint32_t resp_hdrs = 99u;
    CHECK(pg6_mw_hc_http_call_response_get_num_headers(call, &resp_hdrs) == S_OK_HR);
    CHECK(resp_hdrs == 0u);

    const char* missing_hdr = reinterpret_cast<const char*>(1);
    CHECK(pg6_mw_hc_http_call_response_get_header(call, "Date", &missing_hdr) == S_OK_HR);
    CHECK(missing_hdr == nullptr);

    // Close both references; handle remains valid after first close and invalid after second.
    CHECK(pg6_mw_hc_http_call_close_handle(dup) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_response_get_status_code(call, &status_code) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_close_handle(call) == S_OK_HR);
    CHECK(pg6_mw_hc_http_call_close_handle(call) == E_INVALIDARG_HR);

    // CleanupAsync completes cleanly.
    AsyncProbe cleanup_probe;
    Pg6XAsyncBlock cleanup_async{};
    cleanup_async.queue = queue;
    cleanup_async.context = &cleanup_probe;
    cleanup_async.callback = &async_probe_callback;
    CHECK(pg6_mw_hc_cleanup_async(&cleanup_async) == S_OK_HR);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 1, 0) == 1);
    CHECK(cleanup_probe.calls.load() == 1);
    CHECK(cleanup_probe.status.load() == S_OK_HR);

    pg6_gdk_xtask_queue_close_handle(queue);
}

// ---------------------------------------------------------------------------
// 3. Bink2Scarlett.dll lifecycle
// ---------------------------------------------------------------------------

void test_bink2() {
    pg6_mw_bink_set_memory(nullptr, nullptr);
    pg6_mw_bink_set_os_file_callbacks(nullptr);
    CHECK(pg6_mw_bink_set_sound_system(nullptr, 0) == 1);

    uint32_t threads[2] = {0u, 1u};
    CHECK(pg6_mw_bink_start_async_thread(0, nullptr) == 1);
    CHECK(pg6_mw_bink_start_async_thread(1, nullptr) == 1);

    Pg6BinkOptions opts{};
    opts.total_size = 123456u;
    Pg6HBink bink = pg6_mw_bink_open_with_options("movies/intro.bk2", &opts, 0x10u);
    CHECK(bink != nullptr);
    CHECK(bink->width == kDefaultBinkWidth);
    CHECK(bink->height == kDefaultBinkHeight);
    CHECK(bink->frames == kDefaultBinkFrames);
    CHECK(bink->frame_num == 1u);
    CHECK(bink->size == 123456u);

    Pg6BinkFrameBuffers fb{};
    pg6_mw_bink_get_frame_buffers_info(bink, &fb);
    CHECK(fb.total_frames == 2);
    CHECK(fb.ya_buffer_width == bink->width);
    CHECK(fb.crcb_buffer_width == bink->width / 2u);
    pg6_mw_bink_register_frame_buffers(bink, &fb);

    // Non-blocking frame advance (sync + async).
    CHECK(pg6_mw_bink_wait(bink) == 0);
    CHECK(pg6_mw_bink_should_skip(bink) == 0);
    CHECK(pg6_mw_bink_do_frame(bink) == 0);
    CHECK(bink->last_frame_num == 1u);
    pg6_mw_bink_next_frame(bink);
    CHECK(bink->frame_num == 2u);

    CHECK(pg6_mw_bink_do_frame_async_multi(bink, threads, 2) == 1);
    CHECK(pg6_mw_bink_do_frame_async_wait(bink, 1000) == 1);
    CHECK(bink->last_frame_num == 2u);
    pg6_mw_bink_next_frame(bink);
    CHECK(bink->frame_num == 3u);

    // Pause blocks frame advancement and makes BinkWait return 1.
    CHECK(pg6_mw_bink_pause(bink, 1) == 1);
    CHECK(pg6_mw_bink_wait(bink) == 1);
    pg6_mw_bink_next_frame(bink);
    CHECK(bink->frame_num == 3u);
    CHECK(pg6_mw_bink_pause(bink, 0) == 0);
    CHECK(pg6_mw_bink_wait(bink) == 0);

    // Goto + looping vs non-looping end-of-movie behavior.
    pg6_mw_bink_set_volume(bink, 0u, 16384);
    CHECK(bink->volume == 16384);
    CHECK(pg6_mw_bink_get_key_frame(bink, 10u, 0) == 10u);
    pg6_mw_bink_goto(bink, bink->frames, 0);
    CHECK(bink->frame_num == bink->frames);

    // Looping enabled: NextFrame wraps back to 1.
    pg6_mw_bink_set_will_loop(bink, 1);
    pg6_mw_bink_next_frame(bink);
    CHECK(bink->frame_num == 1u);

    // Looping disabled: NextFrame clamps at final frame and sets finished.
    pg6_mw_bink_set_will_loop(bink, 0);
    pg6_mw_bink_goto(bink, bink->frames, 0);
    pg6_mw_bink_next_frame(bink);
    CHECK(bink->frame_num == bink->frames);
    CHECK(bink->finished == 1);

    uint64_t platform_info = 0xDEADBEEFULL;
    CHECK(pg6_mw_bink_get_platform_info(0, &platform_info) == 1);
    CHECK(platform_info == 0u);

    CHECK(pg6_mw_bink_request_stop_async_threads_multi(threads, 2) == 1);
    CHECK(pg6_mw_bink_wait_stop_async_threads_multi(threads, 2) == 1);

    pg6_mw_bink_close(bink);
    pg6_mw_bink_free_globals();
}

// ---------------------------------------------------------------------------
// 4. AcpHal / Apu / XApu / XDsp
// ---------------------------------------------------------------------------

void test_audio_apu_xapu_xdsp() {
    // AcpHal COM stub & shape contexts.
    void* acp_hal_raw = nullptr;
    CHECK(pg6_mw_acp_hal_create(&acp_hal_raw) == S_OK_HR);
    CHECK(acp_hal_raw != nullptr);
    auto* acp_com = static_cast<Pg6MwComObject*>(acp_hal_raw);
    CHECK(acp_com->vtbl->add_ref(acp_hal_raw) == 2u);
    CHECK(acp_com->vtbl->release(acp_hal_raw) == 1u);
    CHECK(acp_com->vtbl->release(acp_hal_raw) == 0u);

    // Shape contexts: six counts, then the call fills one pointer per kind with a zeroed array
    // (only kind 4, 64 bytes per context, is established). A kind of unknown size fails and
    // hands out nothing.
    struct ShapeContexts {
        uint32_t counts[6];
        uint64_t arrays[6];
    };
    ShapeContexts shape{};
    shape.counts[4] = 350;
    shape.arrays[0] = 0x1234;
    CHECK(pg6_mw_acp_hal_allocate_shape_contexts(&shape) == S_OK_HR);
    CHECK(shape.arrays[0] == 0 && shape.arrays[1] == 0 && shape.arrays[5] == 0);
    CHECK(shape.arrays[4] != 0 && shape.arrays[4] % 256 == 0);
    const auto* contexts = static_cast<const uint8_t*>(playgta6::os::default_guest_heap().to_host(shape.arrays[4]));
    CHECK(contexts != nullptr);
    CHECK(std::all_of(contexts, contexts + 350 * 64, [](uint8_t v) { return v == 0; }));
    CHECK(pg6_mw_acp_hal_release_shape_contexts() == S_OK_HR);
    ShapeContexts unknown{};
    unknown.counts[0] = 2;
    unknown.counts[4] = 1;
    CHECK(pg6_mw_acp_hal_allocate_shape_contexts(&unknown) == playgta6::os::gdk::E_NOTIMPL_HR);
    CHECK(unknown.arrays[0] == 0 && unknown.arrays[4] == 0);
    CHECK(pg6_mw_acp_hal_allocate_shape_contexts(nullptr) == playgta6::os::gdk::E_INVALIDARG_HR);

    // ApuCreateHeap -> ApuAlloc -> validity/mapping -> ApuFree round-trip.
    CHECK(pg6_mw_apu_create_heap(4u * 1024u * 1024u, 2u * 1024u * 1024u) == S_OK_HR);

    void* virt_va = nullptr;
    Pg6ApuAddress phys = 0u;
    CHECK(pg6_mw_apu_alloc(&virt_va, &phys, 4096u, 256u, 0u) == S_OK_HR);
    CHECK(virt_va != nullptr);
    CHECK(phys != 0u);
    CHECK((reinterpret_cast<uintptr_t>(virt_va) % 256u) == 0u);
    CHECK((phys % 256u) == 0u);
    // The out-parameter holds a guest address (a guest heap block, O24); the other entry points
    // take host pointers, as the import bridge passes them.
    playgta6::os::GuestHeap& guest_heap = playgta6::os::default_guest_heap();
    CHECK(guest_heap.owns(reinterpret_cast<uintptr_t>(virt_va)));
    void* virt = guest_heap.to_host(reinterpret_cast<uintptr_t>(virt_va));

    // Memory is readable/writable and valid within range.
    std::memset(virt, 0x5A, 4096u);
    CHECK(pg6_mw_apu_is_virtual_address_valid(virt, 4096u) == 1);
    CHECK(pg6_mw_apu_is_virtual_address_valid(static_cast<uint8_t*>(virt) + 128u, 64u) == 1);
    CHECK(pg6_mw_apu_is_virtual_address_valid(virt, 8192u) == 0);

    // Bidirectional VA <-> APU address mapping with interior offsets.
    CHECK(pg6_mw_apu_map_virtual_address(virt) == phys);
    CHECK(pg6_mw_apu_map_virtual_address(static_cast<uint8_t*>(virt) + 64u) == phys + 64u);
    CHECK(pg6_mw_apu_map_apu_address(phys) == virt);
    CHECK(pg6_mw_apu_map_apu_address(phys + 64u) == static_cast<uint8_t*>(virt) + 64u);

    CHECK(pg6_mw_apu_free(virt) == S_OK_HR);
    CHECK(pg6_mw_apu_is_virtual_address_valid(virt, 16u) == 0);
    CHECK(pg6_mw_apu_map_apu_address(phys) == nullptr);
    CHECK(pg6_mw_apu_free(virt) == E_INVALIDARG_HR);
    CHECK(!guest_heap.owns(reinterpret_cast<uintptr_t>(virt_va)));  // released to the guest heap

    // XApu connect / enqueue / result count / dequeue / disconnect.
    Pg6XApuHandle xapu = nullptr;
    CHECK(pg6_mw_xapu_connect(nullptr, &xapu) == S_OK_HR);
    CHECK(xapu != nullptr);

    uint32_t count = 99u;
    CHECK(pg6_mw_xapu_get_result_count(xapu, &count) == S_OK_HR);
    CHECK(count == 0u);

    const uint32_t cmd_payload = 0x12345678u;
    CHECK(pg6_mw_xapu_enqueue_command(xapu, &cmd_payload, sizeof(cmd_payload)) == S_OK_HR);
    CHECK(pg6_mw_xapu_get_result_count(xapu, &count) == S_OK_HR);
    CHECK(count == 1u);

    uint32_t out_payload = 0u;
    uint32_t bytes_written = 0u;
    CHECK(pg6_mw_xapu_dequeue_result(xapu, &out_payload, sizeof(out_payload), &bytes_written) ==
          S_OK_HR);
    CHECK(bytes_written == sizeof(cmd_payload));
    CHECK(out_payload == cmd_payload);
    CHECK(pg6_mw_xapu_get_result_count(xapu, &count) == S_OK_HR);
    CHECK(count == 0u);
    CHECK(pg6_mw_xapu_dequeue_result(xapu, &out_payload, sizeof(out_payload), &bytes_written) ==
          kMwSFalseHResult);
    CHECK(pg6_mw_xapu_disconnect(xapu) == S_OK_HR);
    CHECK(pg6_mw_xapu_disconnect(xapu) == E_INVALIDARG_HR);

    // XDsp (chad.dll) connect / activate / submit / deactivate / disconnect.
    Pg6XDspHandle xdsp = nullptr;
    CHECK(pg6_mw_xdsp_connect_with_maximum_stream_limit(64u, &xdsp) == S_OK_HR);
    CHECK(xdsp != nullptr);
    CHECK(pg6_mw_xdsp_activate(xdsp) == S_OK_HR);
    CHECK(pg6_mw_xdsp_submit_command(xdsp, &cmd_payload, sizeof(cmd_payload)) == S_OK_HR);
    CHECK(pg6_mw_xdsp_deactivate(xdsp) == S_OK_HR);
    CHECK(pg6_mw_xdsp_disconnect(xdsp) == S_OK_HR);
    CHECK(pg6_mw_xdsp_activate(xdsp) == E_INVALIDARG_HR);
}

// ---------------------------------------------------------------------------
// 5. AMD FidelityFX + XboxBvhBuilder + MFPlat/MFReadWrite
// ---------------------------------------------------------------------------

void test_ffx_bvh_mf() {
    // amd_fidelityfx_loader_xbox.dll
    Pg6FfxContext ffx = nullptr;
    Pg6FfxApiHeader desc{};
    desc.type = 0x00010001ULL;
    CHECK(pg6_mw_ffx_create_context(&ffx, &desc, nullptr) == kFfxReturnOk);
    CHECK(ffx != nullptr);
    CHECK(pg6_mw_ffx_query(&ffx, &desc) == kFfxReturnOk);
    CHECK(pg6_mw_ffx_query(nullptr, &desc) == kFfxReturnOk);
    CHECK(pg6_mw_ffx_dispatch(&ffx, &desc) == kFfxReturnOk);
    CHECK(pg6_mw_ffx_destroy_context(&ffx, nullptr) == kFfxReturnOk);
    CHECK(ffx == nullptr);
    CHECK(pg6_mw_ffx_dispatch(&ffx, &desc) == kFfxReturnErrorParameter);

    // XboxBvhBuilder.dll
    uint8_t builder_storage[64]{};
    CHECK(pg6_mw_bvh_builder_ctor(builder_storage, nullptr, true) == builder_storage);
    pg6_mw_bvh_builder_set_64bit_morton_codes(builder_storage, true);

    Pg6D3D12RaytracingPrebuildInfo prebuild{111u, 222u, 333u};
    pg6_mw_bvh_builder_get_prebuild_info(builder_storage, nullptr, nullptr, &prebuild);
    CHECK(prebuild.result_data_max_size_in_bytes == 0u);
    CHECK(prebuild.scratch_data_size_in_bytes == 0u);
    CHECK(prebuild.update_scratch_data_size_in_bytes == 0u);

    pg6_mw_bvh_builder_build_batched_2x(builder_storage, nullptr, 0u, nullptr);
    pg6_mw_bvh_builder_copy(builder_storage, nullptr, 0u, 0u, 0u);
    CHECK(pg6_mw_bvh_builder_rehydrate_batched_2x(builder_storage, nullptr, 0u, nullptr, nullptr) ==
          S_OK_HR);
    pg6_mw_bvh_builder_dtor(builder_storage);

    // MFPlat.DLL & MFReadWrite.dll
    CHECK(pg6_mw_mf_startup(0x00020070u, 0u) == S_OK_HR);
    const int64_t t0 = pg6_mw_mf_get_system_time();
    const int64_t t1 = pg6_mw_mf_get_system_time();
    CHECK(t0 > 0);
    CHECK(t1 >= t0);

    void* attrs = nullptr;
    void* media_type = nullptr;
    void* resolver = nullptr;
    void* reader = nullptr;
    CHECK(pg6_mw_mf_create_attributes(&attrs, 4u) == S_OK_HR);
    CHECK(attrs != nullptr);
    CHECK(pg6_mw_mf_create_media_type(&media_type) == S_OK_HR);
    CHECK(media_type != nullptr);
    CHECK(pg6_mw_mf_create_source_resolver(&resolver) == S_OK_HR);
    CHECK(resolver != nullptr);
    CHECK(pg6_mw_mf_create_source_reader_from_media_source(nullptr, attrs, &reader) ==
          E_INVALIDARG_HR);
    CHECK(pg6_mw_mf_create_source_reader_from_media_source(resolver, attrs, &reader) == S_OK_HR);
    CHECK(reader != nullptr);
    CHECK(pg6_mw_mf_create_attributes(nullptr, 0u) == E_POINTER_HR);

    static_cast<Pg6MwComObject*>(reader)->vtbl->release(reader);
    static_cast<Pg6MwComObject*>(resolver)->vtbl->release(resolver);
    static_cast<Pg6MwComObject*>(media_type)->vtbl->release(media_type);
    static_cast<Pg6MwComObject*>(attrs)->vtbl->release(attrs);

    CHECK(pg6_mw_mf_shutdown() == S_OK_HR);
}

}  // namespace

int main() {
    test_registration_and_lookup();
    test_libhttpclient();
    test_bink2();
    test_audio_apu_xapu_xdsp();
    test_ffx_bvh_mf();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_xbox_middleware FAILED with %d failure(s).\n", g_failures);
        return 1;
    }
    std::printf("test_xbox_middleware PASSED (all 98 symbols verified).\n");
    return 0;
}
