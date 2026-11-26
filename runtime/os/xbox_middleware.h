#pragma once

#include <cstddef>
#include <cstdint>

#include "os/xgameruntime.h"

// Xbox Series (Scarlett) middleware shims for Track 2 task O14:
//   - libHttpClient.GDK.dll (38 symbols): stateful offline HTTP call handle table
//   - Bink2Scarlett.dll     (23 symbols): non-blocking Bink2 playback state machine
//   - AcpHal.dll             (9 symbols): aligned APU heap allocator & address mapping + ACP HAL
//   - xapu.dll               (5 symbols): stateful XApu command/result loopback session
//   - chad.dll               (5 symbols): stateful XDsp stream session stub
//   - amd_fidelityfx_loader_xbox.dll (4 symbols): FidelityFX context/dispatch/query passthrough
//   - XboxBvhBuilder.dll     (7 symbols): zero-size prebuild info / safe no-op BVH builder
//   - MFPlat.DLL             (6 symbols): Media Foundation startup/time/COM object stubs
//   - MFReadWrite.dll        (1 symbol):  Media Foundation source reader stub
//
// Total: 98 symbols registered in Win32ModuleRegistry (O7) via pg6_xbox_middleware_register_all().

// ---------------------------------------------------------------------------
// Opaque C-ABI handle types (Pg6* prefix avoids Windows/GDK header collisions)
// ---------------------------------------------------------------------------

struct Pg6HCCall;
struct Pg6XApuSession;
struct Pg6XDspSession;
struct Pg6FfxContextObj;

using Pg6HCCallHandle = Pg6HCCall*;
using Pg6XApuHandle = Pg6XApuSession*;
using Pg6XDspHandle = Pg6XDspSession*;
using Pg6FfxContext = void*;
using Pg6ApuAddress = uint32_t;

// Minimal COM-compatible IUnknown vtable for AcpHalCreate and MFCreate* stub objects.
struct Pg6MwUnknownVtbl {
    Pg6HResult (*query_interface)(void* self, const void* riid, void** out_object);
    uint32_t (*add_ref)(void* self);
    uint32_t (*release)(void* self);
};

struct Pg6MwComObject {
    const Pg6MwUnknownVtbl* vtbl;
};

// ---------------------------------------------------------------------------
// Bink2 C-ABI structures (leading fields match RAD Bink2 BINK layout)
// ---------------------------------------------------------------------------

struct Pg6BinkPlane {
    int32_t allocate = 0;
    uint32_t pitch = 0;
    void* buffer = nullptr;
};

struct Pg6BinkFrameBuffer {
    Pg6BinkPlane y_plane{};
    Pg6BinkPlane cr_plane{};
    Pg6BinkPlane cb_plane{};
    Pg6BinkPlane a_plane{};
    Pg6BinkPlane h_plane{};
};

struct Pg6BinkFrameBuffers {
    int32_t total_frames = 0;
    uint32_t ya_buffer_width = 0;
    uint32_t ya_buffer_height = 0;
    uint32_t crcb_buffer_width = 0;
    uint32_t crcb_buffer_height = 0;
    uint32_t frame_num = 0;
    Pg6BinkFrameBuffer frames[2]{};
};

struct Pg6BinkOptions {
    uint64_t file_offset = 0;
    uint32_t total_size = 0;
    uint32_t flags = 0;
};

struct Pg6Bink {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t frames = 0;
    uint32_t frame_num = 0;
    uint32_t last_frame_num = 0;
    uint32_t frame_rate = 0;
    uint32_t frame_rate_div = 1;
    uint32_t read_error = 0;
    uint32_t open_flags = 0;
    uint32_t bink_type = 0;
    uint32_t size = 0;
    uint32_t frame_size = 0;
    uint32_t snd_size = 0;
    uint32_t frame_change_percent = 0;

    // Internal playback state (follows the public leading BINK header fields).
    int32_t paused = 0;
    int32_t will_loop = 1;
    int32_t finished = 0;
    int32_t async_in_progress = 0;
    int32_t volume = 32768;
    Pg6BinkFrameBuffers frame_buffers{};
};

using Pg6HBink = Pg6Bink*;

// ---------------------------------------------------------------------------
// FidelityFX API header & return codes (AMD FidelityFX SDK 1.1+)
// ---------------------------------------------------------------------------

struct Pg6FfxApiHeader {
    uint64_t type = 0;
    Pg6FfxApiHeader* p_next = nullptr;
};

// ffxQueryDescUpscaleGetJitterPhaseCount (FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT).
inline constexpr uint64_t kFfxQueryUpscaleGetJitterPhaseCount = 0x00010004ULL;
struct Pg6FfxQueryJitterPhaseCount {
    Pg6FfxApiHeader header;
    uint32_t render_width = 0;
    uint32_t display_width = 0;
    int32_t* out_phase_count = nullptr;
};
static_assert(sizeof(Pg6FfxQueryJitterPhaseCount) == 0x20);

// ---------------------------------------------------------------------------
// D3D12 raytracing prebuild info (matches D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO)
// ---------------------------------------------------------------------------

struct Pg6D3D12RaytracingPrebuildInfo {
    uint64_t result_data_max_size_in_bytes = 0;
    uint64_t scratch_data_size_in_bytes = 0;
    uint64_t update_scratch_data_size_in_bytes = 0;
};
static_assert(sizeof(Pg6D3D12RaytracingPrebuildInfo) == 24,
              "Pg6D3D12RaytracingPrebuildInfo must match D3D12 prebuild info layout");

namespace playgta6::os::middleware {

inline constexpr uint32_t kExpectedMiddlewareSymbolCount = 98u;

// Offline network error HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE).
inline constexpr Pg6HResult kMwOfflineHResult = static_cast<Pg6HResult>(0x800704CFu);
inline constexpr uint32_t kMwPlatformNetworkUnreachable = 1231u;
inline constexpr Pg6HResult kMwSFalseHResult = 1;
inline constexpr Pg6HResult kMwENoInterfaceHResult = static_cast<Pg6HResult>(0x80004002u);

// FidelityFX API return codes.
inline constexpr uint32_t kFfxReturnOk = 0u;
inline constexpr uint32_t kFfxReturnError = 1u;
inline constexpr uint32_t kFfxReturnErrorParameter = 2u;

// Default Bink2 synthetic video parameters.
inline constexpr uint32_t kDefaultBinkWidth = 640u;
inline constexpr uint32_t kDefaultBinkHeight = 360u;
inline constexpr uint32_t kDefaultBinkFrames = 60u;
inline constexpr uint32_t kDefaultBinkFrameRate = 30u;
inline constexpr uint32_t kDefaultBinkFrameRateDiv = 1u;

}  // namespace playgta6::os::middleware

extern "C" {

// ---------------------------------------------------------------------------
// Registration entrypoint
// ---------------------------------------------------------------------------

// Registers all 98 Xbox middleware symbols across 9 DLLs in Win32ModuleRegistry (O7).
// Idempotent; always returns 98.
uint32_t pg6_xbox_middleware_register_all(void);

// How often the Bink2Scarlett.dll fallback shim answered BinkOpenWithOptions (all calls, and the
// ones refused with no movie). Non-zero means the shim, not the lifted DLL, took the call.
void pg6_bink_shim_stats(uint64_t* open_calls, uint64_t* refused_opens);

// ---------------------------------------------------------------------------
// 1. libHttpClient.GDK.dll (38 symbols)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_hc_initialize(const void* args);
Pg6HResult pg6_mw_hc_cleanup_async(Pg6XAsyncBlock* async_block);

Pg6HResult pg6_mw_hc_http_call_create(Pg6HCCallHandle* call);
Pg6HCCallHandle pg6_mw_hc_http_call_duplicate_handle(Pg6HCCallHandle call);
Pg6HResult pg6_mw_hc_http_call_close_handle(Pg6HCCallHandle call);

Pg6HResult pg6_mw_hc_http_call_perform_async(Pg6HCCallHandle call, Pg6XAsyncBlock* async_block);

Pg6HResult pg6_mw_hc_http_call_request_set_url(Pg6HCCallHandle call,
                                               const char* method,
                                               const char* url);
Pg6HResult pg6_mw_hc_http_call_request_get_url(Pg6HCCallHandle call,
                                               const char** method,
                                               const char** url);
Pg6HResult pg6_mw_hc_http_call_get_request_url(Pg6HCCallHandle call, const char** url);

Pg6HResult pg6_mw_hc_http_call_request_set_request_body_bytes(Pg6HCCallHandle call,
                                                              const uint8_t* request_body_bytes,
                                                              uint32_t request_body_size);
Pg6HResult pg6_mw_hc_http_call_request_get_request_body_bytes(Pg6HCCallHandle call,
                                                              const uint8_t** request_body_bytes,
                                                              uint32_t* request_body_size);
Pg6HResult pg6_mw_hc_http_call_request_set_request_body_string(Pg6HCCallHandle call,
                                                               const char* request_body_string);

Pg6HResult pg6_mw_hc_http_call_request_set_header(Pg6HCCallHandle call,
                                                  const char* header_name,
                                                  const char* header_value,
                                                  bool allow_tracing);
Pg6HResult pg6_mw_hc_http_call_request_get_num_headers(Pg6HCCallHandle call, uint32_t* num_headers);
Pg6HResult pg6_mw_hc_http_call_request_get_header_at_index(Pg6HCCallHandle call,
                                                           uint32_t header_index,
                                                           const char** header_name,
                                                           const char** header_value);

Pg6HResult pg6_mw_hc_http_call_request_set_retry_allowed(Pg6HCCallHandle call, bool retry_allowed);
Pg6HResult pg6_mw_hc_http_call_request_get_retry_allowed(Pg6HCCallHandle call, bool* retry_allowed);
Pg6HResult pg6_mw_hc_http_call_request_set_retry_cache_id(Pg6HCCallHandle call,
                                                          uint32_t retry_after_cache_id);
Pg6HResult pg6_mw_hc_http_call_request_get_retry_cache_id(Pg6HCCallHandle call,
                                                          uint32_t* retry_after_cache_id);
Pg6HResult pg6_mw_hc_http_call_request_set_retry_delay(Pg6HCCallHandle call,
                                                       uint32_t retry_delay_in_seconds);
Pg6HResult pg6_mw_hc_http_call_request_get_retry_delay(Pg6HCCallHandle call,
                                                       uint32_t* retry_delay_in_seconds);
Pg6HResult pg6_mw_hc_http_call_request_set_timeout(Pg6HCCallHandle call,
                                                   uint32_t timeout_in_seconds);
Pg6HResult pg6_mw_hc_http_call_request_get_timeout(Pg6HCCallHandle call,
                                                   uint32_t* timeout_in_seconds);
Pg6HResult pg6_mw_hc_http_call_request_set_timeout_window(Pg6HCCallHandle call,
                                                          uint32_t timeout_window_in_seconds);
Pg6HResult pg6_mw_hc_http_call_request_get_timeout_window(Pg6HCCallHandle call,
                                                          uint32_t* timeout_window_in_seconds);

Pg6HResult pg6_mw_hc_http_call_response_get_header(Pg6HCCallHandle call,
                                                   const char* header_name,
                                                   const char** header_value);
Pg6HResult pg6_mw_hc_http_call_response_get_header_at_index(Pg6HCCallHandle call,
                                                            uint32_t header_index,
                                                            const char** header_name,
                                                            const char** header_value);
Pg6HResult pg6_mw_hc_http_call_response_get_num_headers(Pg6HCCallHandle call,
                                                        uint32_t* num_headers);
Pg6HResult pg6_mw_hc_http_call_response_get_network_error_code(
    Pg6HCCallHandle call,
    Pg6HResult* network_error_code,
    uint32_t* platform_network_error_code);
Pg6HResult pg6_mw_hc_http_call_response_get_platform_network_error_message(
    Pg6HCCallHandle call,
    const char** platform_network_error_message);
Pg6HResult pg6_mw_hc_http_call_response_get_response_body_bytes(Pg6HCCallHandle call,
                                                                size_t buffer_size,
                                                                uint8_t* buffer,
                                                                size_t* buffer_used);
Pg6HResult pg6_mw_hc_http_call_response_get_response_body_bytes_size(Pg6HCCallHandle call,
                                                                     size_t* buffer_size);
Pg6HResult pg6_mw_hc_http_call_response_get_response_string(Pg6HCCallHandle call,
                                                            const char** response_string);
Pg6HResult pg6_mw_hc_http_call_response_get_status_code(Pg6HCCallHandle call,
                                                        uint32_t* status_code);

Pg6HResult pg6_mw_hc_http_call_set_tracing(Pg6HCCallHandle call, bool trace_call);
Pg6HResult pg6_mw_hc_settings_get_trace_level(uint32_t* trace_level);
void pg6_mw_hc_trace_impl_message(const char* area_name, uint32_t level, const char* format, ...);
void pg6_mw_hc_trace_set_etw_enabled(bool enabled);

// ---------------------------------------------------------------------------
// 2. Bink2Scarlett.dll (23 symbols)
// ---------------------------------------------------------------------------

Pg6HBink pg6_mw_bink_open_with_options(const char* name,
                                       const Pg6BinkOptions* options,
                                       uint32_t flags);
void pg6_mw_bink_close(Pg6HBink bink);
int32_t pg6_mw_bink_do_frame(Pg6HBink bink);
int32_t pg6_mw_bink_do_frame_async_multi(Pg6HBink bink, uint32_t* threads, int32_t thread_count);
int32_t pg6_mw_bink_do_frame_async_wait(Pg6HBink bink, int64_t us);
void pg6_mw_bink_next_frame(Pg6HBink bink);
int32_t pg6_mw_bink_wait(Pg6HBink bink);
int32_t pg6_mw_bink_pause(Pg6HBink bink, int32_t pause);
void pg6_mw_bink_goto(Pg6HBink bink, uint32_t frame, int32_t flags);
uint32_t pg6_mw_bink_get_key_frame(Pg6HBink bink, uint32_t frame, int32_t flags);
int32_t pg6_mw_bink_should_skip(Pg6HBink bink);
void pg6_mw_bink_set_will_loop(Pg6HBink bink, int32_t onoff);
void pg6_mw_bink_set_volume(Pg6HBink bink, uint32_t track_id, int32_t volume);
void pg6_mw_bink_get_frame_buffers_info(Pg6HBink bink, Pg6BinkFrameBuffers* set);
void pg6_mw_bink_register_frame_buffers(Pg6HBink bink, const Pg6BinkFrameBuffers* set);
int32_t pg6_mw_bink_start_async_thread(int32_t thread_num, const void* thread_param);
int32_t pg6_mw_bink_request_stop_async_threads_multi(uint32_t* threads, int32_t thread_count);
int32_t pg6_mw_bink_wait_stop_async_threads_multi(uint32_t* threads, int32_t thread_count);
void pg6_mw_bink_set_memory(void* alloc_fn, void* free_fn);
void pg6_mw_bink_set_os_file_callbacks(void* file_open_fn);
int32_t pg6_mw_bink_set_sound_system(void* open_fn, uintptr_t param);
int32_t pg6_mw_bink_get_platform_info(int32_t bink_platform_enum, void* output);
void pg6_mw_bink_free_globals(void);

// ---------------------------------------------------------------------------
// 3. AcpHal.dll (9 symbols)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_acp_hal_create(void** acp_hal);
Pg6HResult pg6_mw_acp_hal_allocate_shape_contexts(void* contexts);  // counts in, context arrays out
Pg6HResult pg6_mw_acp_hal_release_shape_contexts(void);

// R40: census of the XMA decoder contexts (shape context kind 4) the title allocated. Each is 64
// bytes in the public XMA context layout; the executable's stream consumer (0x140078120) reads
// dword 0 bits 22-26 (output buffer block count) and 27-31 (output write offset, in 256-byte
// blocks) and dword 9 bits 0-4 (output read offset), and only copies decoded PCM out and sends
// AcpHal command 0x12 (tag 0xF1F1F1F1) when the write offset has moved. No XMA decoder exists
// here, so the write offset never moves and every XMA voice (music, ambience) stays silent.
struct Pg6XmaContextCensus {
    uint32_t contexts = 0;        // kind-4 contexts allocated
    uint32_t touched = 0;         // contexts with any non-zero byte (set up by the title)
    uint32_t input_valid = 0;     // dword 0 bit 20 or 21 (input buffer 0 / 1 valid): data waiting to decode
    uint32_t output_pending = 0;  // output write offset != read offset (decoded data not yet consumed)
};
Pg6XmaContextCensus pg6_mw_xma_context_census(void);
// The live kind-4 (XMA) context array: its guest address (0 when none) and context count.
uint64_t pg6_mw_xma_contexts(uint32_t* count);
Pg6HResult pg6_mw_apu_create_heap(uint32_t cached_size_in_bytes,
                                  uint32_t non_cached_size_in_bytes);
Pg6HResult pg6_mw_apu_alloc(void** virtual_address,
                            Pg6ApuAddress* physical_address,
                            uint32_t size_in_bytes,
                            uint32_t alignment_in_bytes,
                            uint32_t flags);
Pg6HResult pg6_mw_apu_free(void* virtual_address);
int32_t pg6_mw_apu_is_virtual_address_valid(const void* virtual_address,
                                            uint32_t physical_address_in_bytes);
void* pg6_mw_apu_map_apu_address(Pg6ApuAddress apu_address);
Pg6ApuAddress pg6_mw_apu_map_virtual_address(const void* virtual_address);

// ---------------------------------------------------------------------------
// 4. xapu.dll (5 symbols)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_xapu_connect(const void* params, Pg6XApuHandle* handle);
Pg6HResult pg6_mw_xapu_disconnect(Pg6XApuHandle handle);
Pg6HResult pg6_mw_xapu_enqueue_command(Pg6XApuHandle handle,
                                       const void* command,
                                       uint32_t command_size);
Pg6HResult pg6_mw_xapu_dequeue_result(Pg6XApuHandle handle,
                                      void* result_buffer,
                                      uint32_t result_buffer_size,
                                      uint32_t* bytes_written);
Pg6HResult pg6_mw_xapu_get_result_count(Pg6XApuHandle handle, uint32_t* result_count);

// ---------------------------------------------------------------------------
// 5. chad.dll (5 symbols)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_xdsp_connect_with_maximum_stream_limit(uint32_t max_streams,
                                                         Pg6XDspHandle* handle);
Pg6HResult pg6_mw_xdsp_disconnect(Pg6XDspHandle handle);
Pg6HResult pg6_mw_xdsp_activate(Pg6XDspHandle handle);
Pg6HResult pg6_mw_xdsp_deactivate(Pg6XDspHandle handle);
Pg6HResult pg6_mw_xdsp_submit_command(Pg6XDspHandle handle,
                                      const void* command,
                                      uint32_t command_size);

// ---------------------------------------------------------------------------
// 6. amd_fidelityfx_loader_xbox.dll (4 symbols)
// ---------------------------------------------------------------------------

uint32_t pg6_mw_ffx_create_context(Pg6FfxContext* context,
                                   const Pg6FfxApiHeader* desc,
                                   const void* mem_cb);
uint32_t pg6_mw_ffx_destroy_context(Pg6FfxContext* context, const void* mem_cb);
uint32_t pg6_mw_ffx_dispatch(Pg6FfxContext* context, const Pg6FfxApiHeader* desc);
uint32_t pg6_mw_ffx_query(Pg6FfxContext* context, Pg6FfxApiHeader* desc);

// ---------------------------------------------------------------------------
// 7. XboxBvhBuilder.dll (7 symbols)
// ---------------------------------------------------------------------------

void* pg6_mw_bvh_builder_ctor(void* self, void* d3d12_device, bool enable_flag);
void pg6_mw_bvh_builder_dtor(void* self);
void pg6_mw_bvh_builder_build_batched_2x(void* self,
                                         void* cmd_list,
                                         uint32_t num_descs,
                                         const void* descs);
void pg6_mw_bvh_builder_copy(void* self,
                             void* cmd_list,
                             uint64_t dest_va,
                             uint64_t src_va,
                             uint32_t copy_mode);
void pg6_mw_bvh_builder_get_prebuild_info(void* self,
                                          const void* inputs,
                                          const void* stream_desc,
                                          Pg6D3D12RaytracingPrebuildInfo* out_info);
Pg6HResult pg6_mw_bvh_builder_rehydrate_batched_2x(void* self,
                                                   void* cmd_list,
                                                   uint32_t num_descs,
                                                   const void* descs,
                                                   const uint64_t* dest_addresses);
void pg6_mw_bvh_builder_set_64bit_morton_codes(void* self, bool enabled);

// ---------------------------------------------------------------------------
// 8. MFPlat.DLL (6 symbols) & 9. MFReadWrite.dll (1 symbol)
// ---------------------------------------------------------------------------

Pg6HResult pg6_mw_mf_startup(uint32_t version, uint32_t flags);
Pg6HResult pg6_mw_mf_shutdown(void);
int64_t pg6_mw_mf_get_system_time(void);
Pg6HResult pg6_mw_mf_create_attributes(void** out_attributes, uint32_t initial_size);
Pg6HResult pg6_mw_mf_create_media_type(void** out_media_type);
Pg6HResult pg6_mw_mf_create_source_resolver(void** out_source_resolver);
Pg6HResult pg6_mw_mf_create_source_reader_from_media_source(void* media_source,
                                                            void* attributes,
                                                            void** out_source_reader);

}  // extern "C"
