#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/game_input.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/xgameruntime.h"

// Xbox Scarlett (Series X|S) system DLL shims (Track 2 / O12):
//   - xmem.dll       (12 imports: XMemAlloc*, XMem*PhysicalPages*, XMemVirtualAlloc,
//                     XMemTransferMemory, XMemGetWorkingSetStatistics, XMemIsToolsMemAvailable)
//   - XCRDAPI.dll    (1 import:  DStorageGetFactory + COM-like DirectStorage interfaces)
//   - xg_xs.dll      (2 imports: XGCreateTextureComputer, XGSuggestSwizzleMode)
//   - GameInput.dll  (1 import:  ordinal #1 / GameInputCreate -> live GameInputReader SAB bridge)
//
// All C-ABI types are prefixed with Pg6* and all C-ABI entrypoints are prefixed with pg6_xbox_*
// to avoid collisions with Windows SDK / Xbox GDK headers on MSVC hosts and Emscripten Wasm64.

// Pg6Guid is the one GUID type of os/win32_compat.h (a second definition here made the two headers
// impossible to include together, e.g. in the generated import bridge).
constexpr bool operator==(const Pg6Guid& a, const Pg6Guid& b) {
    if (a.data1 != b.data1 || a.data2 != b.data2 || a.data3 != b.data3) {
        return false;
    }
    for (size_t i = 0; i < 8; ++i) {
        if (a.data4[i] != b.data4[i]) {
            return false;
        }
    }
    return true;
}

constexpr bool operator!=(const Pg6Guid& a, const Pg6Guid& b) {
    return !(a == b);
}
static_assert(sizeof(Pg6Guid) == 16, "Pg6Guid must be 16 bytes (standard COM GUID layout)");

// Minimal COM-like base interface (QueryInterface / AddRef / Release).
struct Pg6IUnknown {
    virtual ~Pg6IUnknown() = default;
    virtual Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) = 0;
    virtual uint32_t AddRef() = 0;
    virtual uint32_t Release() = 0;
};

// ---------------------------------------------------------------------------
// 1. xmem.dll C-ABI structs & constants
// ---------------------------------------------------------------------------

struct Pg6XMemWorkingSetStats {
    uint64_t total_physical_bytes = 0;       // Total console RAM (16 GiB on Series X)
    uint64_t title_budget_bytes = 0;         // Title-accessible budget (13.5 GiB on Series X)
    uint64_t heap_allocated_bytes = 0;       // Live XMemAlloc / XMemAllocDefault bytes
    uint64_t physical_pages_bytes = 0;       // Live XMemAllocatePhysicalPages bytes
    uint64_t virtual_committed_bytes = 0;    // Live XMemVirtualAlloc committed bytes
    uint64_t virtual_reserved_bytes = 0;     // Live XMemVirtualAlloc reserved bytes
    uint64_t mapped_physical_bytes = 0;      // Currently mapped physical page bytes
    uint64_t peak_working_set_bytes = 0;     // High-water mark of total working set
    uint64_t available_title_bytes = 0;      // Remaining title budget
    uint64_t tools_mem_bytes = 0;            // Always 0 (retail web runtime, no devkit tools mem)
    uint32_t live_heap_allocations = 0;
    uint32_t live_physical_pages = 0;
    uint32_t live_virtual_mappings = 0;
    uint32_t reserved = 0;
};

// ---------------------------------------------------------------------------
// 2. XCRDAPI.dll (DirectStorage) COM-like interfaces & structs
// ---------------------------------------------------------------------------

struct Pg6DStorageFileInfo {
    uint64_t file_size = 0;
};

struct Pg6IDStorageFile : public Pg6IUnknown {
    virtual void Close() = 0;
    virtual Pg6HResult GetFileInformation(Pg6DStorageFileInfo* info) = 0;
};

struct Pg6IDStorageStatusArray : public Pg6IUnknown {
    virtual bool IsComplete(uint32_t index) = 0;
    virtual Pg6HResult GetHResult(uint32_t index) = 0;
};

struct Pg6DStorageQueueDesc {
    uint32_t source_type = 0;  // 0 = File, 1 = Memory
    uint16_t capacity = 256;   // Max pending requests
    int8_t priority = 0;       // -1 = Low, 0 = Normal, 1 = High, 2 = Realtime
    uint8_t reserved = 0;
    const char* name = nullptr;
};

struct Pg6DStorageRequest {
    Pg6IDStorageFile* file = nullptr;
    uint64_t offset = 0;
    uint32_t size = 0;
    uint32_t destination_size = 0;    // 0 -> defaults to `size`
    void* destination = nullptr;
    uint32_t compression_format = 0;  // 0 = None
    uint32_t reserved = 0;
    uint64_t cancellation_tag = 0;
    const char* name = nullptr;
};

struct Pg6DStorageErrorRecord {
    uint32_t failure_count = 0;
    Pg6HResult first_failure_hresult = 0;
    uint64_t first_failure_offset = 0;
    uint32_t first_failure_size = 0;
    uint32_t reserved = 0;
};

struct Pg6IDStorageQueue : public Pg6IUnknown {
    virtual void EnqueueRequest(const Pg6DStorageRequest* request) = 0;
    virtual void EnqueueRead(Pg6IDStorageFile* file,
                             uint64_t offset,
                             void* destination,
                             uint32_t size) = 0;
    virtual void EnqueueStatus(Pg6IDStorageStatusArray* status_array, uint32_t index) = 0;
    virtual void EnqueueSignal(Pg6Handle event_handle, uint64_t value) = 0;
    virtual void Submit() = 0;
    virtual void CancelRequestsWithTag(uint64_t mask, uint64_t value) = 0;
    virtual void Close() = 0;
    virtual uint32_t RetrieveErrorRecord(Pg6DStorageErrorRecord* record) = 0;
    virtual uint32_t QueryPendingCount() const = 0;
};

struct Pg6IDStorageFactory : public Pg6IUnknown {
    virtual Pg6HResult CreateQueue(const Pg6DStorageQueueDesc* desc,
                                   const Pg6Guid& riid,
                                   void** ppv) = 0;
    virtual Pg6HResult OpenFile(const char16_t* path,
                                const Pg6Guid& riid,
                                void** ppv) = 0;
    virtual Pg6HResult OpenFileA(const char* path,
                                 const Pg6Guid& riid,
                                 void** ppv) = 0;
    virtual Pg6HResult CreateStatusArray(uint32_t capacity,
                                         const char* name,
                                         const Pg6Guid& riid,
                                         void** ppv) = 0;
    virtual Pg6HResult SetStagingBufferSize(uint32_t size) = 0;
    virtual uint32_t GetStagingBufferSize() const = 0;
};

// ---------------------------------------------------------------------------
// 3. xg_xs.dll (Xbox Graphics Texture Layout Computer) interfaces & structs
// ---------------------------------------------------------------------------

struct Pg6XgTextureDesc {
    uint32_t width = 0;
    uint32_t height = 1;
    uint32_t depth_or_array_size = 1;
    uint32_t mip_levels = 1;          // 0 = full mip chain down to 1x1
    uint32_t format = 28;             // DXGI_FORMAT (28 = R8G8B8A8_UNORM)
    uint32_t dimension = 3;           // 2 = Texture1D, 3 = Texture2D, 4 = Texture3D
    uint32_t sample_count = 1;
    uint32_t tile_mode = 0;           // 0 = Linear, 1 = 2DThin, 2 = Standard64KB
    uint32_t pitch_alignment = 0;     // 0 = default (256 bytes, WebGPU/D3D12 aligned)
    uint32_t reserved = 0;
};

struct Pg6XgMipLevelLayout {
    uint64_t offset_bytes = 0;
    uint64_t size_bytes = 0;
    uint64_t slice_size_bytes = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t row_pitch_bytes = 0;
    uint32_t padded_rows = 0;
    uint32_t tile_mode = 0;
    uint32_t reserved = 0;
};

struct Pg6XgResourceLayout {
    uint64_t size_bytes = 0;
    uint64_t base_alignment_bytes = 0;
    uint32_t mip_levels = 0;
    uint32_t array_size = 0;
    uint32_t tile_mode = 0;
    uint32_t reserved = 0;
    Pg6XgMipLevelLayout mips[16]{};
};

struct Pg6IXGTextureAddressComputer : public Pg6IUnknown {
    virtual Pg6HResult GetResourceLayout(Pg6XgResourceLayout* out_layout) = 0;
    virtual uint64_t GetResourceSizeBytes() = 0;
    virtual uint64_t GetResourceBaseAlignmentBytes() = 0;
    virtual uint64_t GetMipLevelOffsetBytes(uint32_t plane, uint32_t mip_level) = 0;
    virtual uint64_t GetTexelElementOffsetBytes(uint32_t plane,
                                                uint32_t mip_level,
                                                uint64_t x,
                                                uint32_t y,
                                                uint32_t z_or_slice,
                                                uint32_t sample) = 0;
    virtual uint32_t GetMipLevelRowPitchBytes(uint32_t mip_level) = 0;
    virtual uint64_t GetMipLevelSliceSizeBytes(uint32_t mip_level) = 0;
    virtual Pg6HResult CopyIntoSubresource(void* dst_base,
                                           uint32_t plane,
                                           uint32_t subresource,
                                           const void* src_data,
                                           uint32_t src_row_pitch,
                                           uint32_t src_slice_pitch) = 0;
    virtual Pg6HResult CopyFromSubresource(void* dst_data,
                                           uint32_t dst_row_pitch,
                                           uint32_t dst_slice_pitch,
                                           const void* src_base,
                                           uint32_t plane,
                                           uint32_t subresource) = 0;
};

// ---------------------------------------------------------------------------
// 4. GameInput.dll COM-like interfaces & structs
// ---------------------------------------------------------------------------

struct Pg6GameInputGamepadState {
    uint32_t buttons = 0;             // Pg6GameInputGamepadButtons bitmask
    float left_trigger = 0.0f;        // [0.0, 1.0]
    float right_trigger = 0.0f;       // [0.0, 1.0]
    float left_thumbstick_x = 0.0f;   // [-1.0, 1.0]
    float left_thumbstick_y = 0.0f;   // [-1.0, 1.0]
    float right_thumbstick_x = 0.0f;  // [-1.0, 1.0]
    float right_thumbstick_y = 0.0f;  // [-1.0, 1.0]
};

struct Pg6GameInputMouseState {
    uint32_t buttons = 0;
    int64_t position_x = 0;
    int64_t position_y = 0;
    int64_t wheel_x = 0;
    int64_t wheel_y = 0;
};

struct Pg6IGameInputReading : public Pg6IUnknown {
    virtual uint32_t GetInputKind() = 0;
    virtual uint64_t GetSequenceNumber(uint32_t input_kind) = 0;
    virtual uint64_t GetTimestamp() = 0;
    virtual bool GetGamepadState(Pg6GameInputGamepadState* state) = 0;
    virtual bool GetMouseState(Pg6GameInputMouseState* state) = 0;
    virtual uint32_t GetRawButtons() = 0;
};

struct Pg6IGameInput : public Pg6IUnknown {
    virtual uint64_t GetCurrentTimestamp() = 0;
    virtual Pg6HResult GetCurrentReading(uint32_t input_kind,
                                         void* device,
                                         Pg6IGameInputReading** reading) = 0;
    virtual Pg6HResult GetNextReading(Pg6IGameInputReading* reference_reading,
                                      uint32_t input_kind,
                                      void* device,
                                      Pg6IGameInputReading** reading) = 0;
    virtual Pg6HResult GetPreviousReading(Pg6IGameInputReading* reference_reading,
                                          uint32_t input_kind,
                                          void* device,
                                          Pg6IGameInputReading** reading) = 0;
    virtual void SetFocusPolicy(uint32_t policy) = 0;
};

// ---------------------------------------------------------------------------
// C++ namespace constants, GUIDs, diagnostics & test/host integration helpers
// ---------------------------------------------------------------------------

namespace playgta6::os::scarlett {

using HResult = Pg6HResult;

// Additional COM / Win32 HRESULTs not already in xgameruntime.h.
inline constexpr HResult E_NOINTERFACE_HR = static_cast<HResult>(0x80004002u);
inline constexpr HResult E_FILE_NOT_FOUND_HR = static_cast<HResult>(0x80070002u);
inline constexpr HResult E_EOF_HR = static_cast<HResult>(0x80070026u);
inline constexpr HResult GAMEINPUT_E_READING_NOT_FOUND = static_cast<HResult>(0x838A0005u);

// Well-known interface IIDs for QueryInterface / DStorageGetFactory / CreateQueue / OpenFile.
inline constexpr Pg6Guid kIidIUnknown = {
    0x00000000u, 0x0000u, 0x0000u, {0xC0u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x46u}};
inline constexpr Pg6Guid kIidIDStorageFactory = {
    0x6924EA0Cu, 0xC3CDu, 0x4826u, {0xB1u, 0x0Au, 0xF6u, 0x4Fu, 0x4Eu, 0xD9u, 0x27u, 0xC1u}};
inline constexpr Pg6Guid kIidIDStorageFile = {
    0x5DE95E7Bu, 0x955Au, 0x4868u, {0xA7u, 0x3Cu, 0x24u, 0x3Bu, 0x29u, 0xF4u, 0xB8u, 0xDAu}};
inline constexpr Pg6Guid kIidIDStorageQueue = {
    0xCFDBD83Fu, 0x9E06u, 0x4FDAu, {0x8Eu, 0xA5u, 0x69u, 0x04u, 0x21u, 0x3Du, 0xE1u, 0x49u}};
inline constexpr Pg6Guid kIidIDStorageStatusArray = {
    0x82397587u, 0x7CD5u, 0x453Bu, {0xA0u, 0x2Eu, 0x31u, 0x37u, 0x9Bu, 0xD6u, 0x48u, 0xDBu}};
inline constexpr Pg6Guid kIidIXGTextureAddressComputer = {
    0x7B9F4D10u, 0x1A2Bu, 0x4C3Du, {0x8Eu, 0x9Fu, 0x01u, 0x23u, 0x45u, 0x67u, 0x89u, 0xABu}};
inline constexpr Pg6Guid kIidIGameInput = {
    0x11BE2A7Eu, 0x4254u, 0x445Au, {0x9Cu, 0x09u, 0xFFu, 0xC4u, 0x0Fu, 0x00u, 0x69u, 0x18u}};
inline constexpr Pg6Guid kIidIGameInputReading = {
    0x2156947Au, 0xE1FAu, 0x4DE0u, {0xA3u, 0x0Bu, 0xD8u, 0x12u, 0x93u, 0x1Du, 0xBDu, 0x8Du}};

// XMem console budget & page size constants (Xbox Series X / Scarlett Anaconda).
inline constexpr uint64_t kConsoleTotalRamBytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
inline constexpr uint64_t kConsoleTitleBudgetBytes =
    13ULL * 1024ULL * 1024ULL * 1024ULL + 512ULL * 1024ULL * 1024ULL;
inline constexpr uint64_t kXMemPageSize4K = 4096ULL;
inline constexpr uint64_t kXMemPageSize64K = 65536ULL;
inline constexpr uint64_t kXMemPageSize2M = 2ULL * 1024ULL * 1024ULL;

// XMem physical page flags.
inline constexpr uint32_t kXMemFlagPage4K = 0x00000001u;
inline constexpr uint32_t kXMemFlagPage64K = 0x00000000u;  // Default Xbox Scarlett physical page size
inline constexpr uint32_t kXMemFlagPage2M = 0x00000002u;
inline constexpr uint32_t kXMemFlagLargePages = 0x20000000u;

// XMEMALLOC_ATTRIBUTES bitfield helpers.
inline constexpr uint64_t kXMemAttrObjectTypeMask = 0xFFFFULL;
inline constexpr uint64_t kXMemAttrHeapTypeShift = 16;
inline constexpr uint64_t kXMemAttrAlignmentShift = 20;
inline constexpr uint64_t kXMemAttrAlignmentMask = 0x1FULL;
inline constexpr uint64_t kXMemAttrMemoryTypeShift = 25;
inline constexpr uint64_t kXMemAttrZeroInitBit = 1ULL << 29;

// Encodes alignment code (0 = default 16B; k in [1..20] -> 2^(k+1) bytes, e.g. 3=16, 7=256, 11=4K, 15=64K).
inline constexpr uint64_t make_xmem_attributes(uint16_t object_type = 0,
                                               uint8_t heap_type = 0,
                                               uint8_t alignment_code = 0,
                                               uint8_t memory_type = 0,
                                               bool zero_init = false) {
    return static_cast<uint64_t>(object_type) |
           ((static_cast<uint64_t>(heap_type) & 0xFULL) << kXMemAttrHeapTypeShift) |
           ((static_cast<uint64_t>(alignment_code) & kXMemAttrAlignmentMask)
            << kXMemAttrAlignmentShift) |
           ((static_cast<uint64_t>(memory_type) & 0xFULL) << kXMemAttrMemoryTypeShift) |
           (zero_init ? kXMemAttrZeroInitBit : 0ULL);
}

// XMem diagnostic & lifecycle helpers.
struct XMemAllocInfo {
    size_t size = 0;
    size_t alignment = 0;
    uint64_t attributes = 0;
};

using XMemAllocHookFn = void* (*)(size_t size, uint64_t attributes);
using XMemFreeHookFn = void (*)(void* address, uint64_t attributes);

void xmem_set_allocation_hooks(XMemAllocHookFn alloc_hook, XMemFreeHookFn free_hook);

// Guest allocation hooks (the console's XMem routes XMemAlloc / XMemFree through a title's own
// routines; XMemAllocDefault / XMemFreeDefault never do):
// * A module that exports the data symbol `__XMemAllocationHooks__` ({alloc routine, free
//   routine, ...}, guest function addresses) provides them: the boot passes that structure's
//   guest address here once the modules are loaded. Its entries are read at every call, so a
//   title that changes them later is followed. 0 removes it.
// * XMemSetAllocationHooks(alloc, free) (documented API) sets them explicitly and takes
//   precedence over the exported structure; null routines remove them.
// Routines are called on the calling thread's guest state as alloc(size, attributes) -> guest
// address and free(address, attributes). A host hook (xmem_set_allocation_hooks) takes
// precedence over both (tests, tools).
void xmem_set_guest_hook_table(uint64_t table_va);
uint64_t xmem_guest_hook_table();
bool xmem_get_allocation_info(const void* address, XMemAllocInfo* out_info);
bool xmem_virtual_free(void* address);
bool xmem_unmap_physical_pages(void* virtual_address);
void xmem_reset_for_testing();

// DirectStorage backing hooks (in-memory virtual file table + host read callback + HTTPFS fallback).
using DStorageHostOpenCallback = std::function<int64_t(std::string_view normalized_path)>;
using DStorageHostReadCallback =
    std::function<int64_t(std::string_view normalized_path, uint64_t offset, void* dst, size_t len)>;

void register_dstorage_virtual_file(std::string_view path, std::vector<uint8_t> data);
void unregister_dstorage_virtual_file(std::string_view path);
void clear_dstorage_virtual_files();
void set_dstorage_host_callbacks(DStorageHostOpenCallback open_cb,
                                 DStorageHostReadCallback read_cb);
void reset_dstorage_for_testing();

// xg_xs tile mode & resource dimension constants.
inline constexpr uint32_t kXgTileModeLinear = 0u;
inline constexpr uint32_t kXgTileMode2DThin = 1u;
inline constexpr uint32_t kXgTileModeStandard64KB = 2u;
inline constexpr uint32_t kXgTileModeDisplay = 3u;
inline constexpr uint32_t kXgTileModeInvalid = 0xFFFFFFFFu;

inline constexpr uint32_t kXgDimensionTexture1D = 2u;
inline constexpr uint32_t kXgDimensionTexture2D = 3u;
inline constexpr uint32_t kXgDimensionTexture3D = 4u;

// Subset of standard DXGI_FORMAT values used by xg_xs texture layout computation.
inline constexpr uint32_t kDxgiFormatR32G32B32A32Float = 2u;
inline constexpr uint32_t kDxgiFormatR16G16B16A16Float = 10u;
inline constexpr uint32_t kDxgiFormatR8G8B8A8Unorm = 28u;
inline constexpr uint32_t kDxgiFormatR8G8B8A8UnormSrgb = 29u;
inline constexpr uint32_t kDxgiFormatD32Float = 40u;
inline constexpr uint32_t kDxgiFormatR32Float = 41u;
inline constexpr uint32_t kDxgiFormatD24UnormS8Uint = 45u;
inline constexpr uint32_t kDxgiFormatR8G8Unorm = 49u;
inline constexpr uint32_t kDxgiFormatR16Float = 54u;
inline constexpr uint32_t kDxgiFormatR8Unorm = 61u;
inline constexpr uint32_t kDxgiFormatBc1Unorm = 71u;
inline constexpr uint32_t kDxgiFormatBc1UnormSrgb = 72u;
inline constexpr uint32_t kDxgiFormatBc2Unorm = 74u;
inline constexpr uint32_t kDxgiFormatBc3Unorm = 77u;
inline constexpr uint32_t kDxgiFormatBc3UnormSrgb = 78u;
inline constexpr uint32_t kDxgiFormatBc4Unorm = 80u;
inline constexpr uint32_t kDxgiFormatBc5Unorm = 83u;
inline constexpr uint32_t kDxgiFormatB8G8R8A8Unorm = 87u;
inline constexpr uint32_t kDxgiFormatBc6hUf16 = 95u;
inline constexpr uint32_t kDxgiFormatBc7Unorm = 98u;
inline constexpr uint32_t kDxgiFormatBc7UnormSrgb = 99u;

// GameInput kinds & button bitmask matching Xbox GDK GameInput.h.
inline constexpr uint32_t kGameInputKindUnknown = 0x00000000u;
inline constexpr uint32_t kGameInputKindGamepad = 0x00000008u;
inline constexpr uint32_t kGameInputKindKeyboard = 0x00000010u;
inline constexpr uint32_t kGameInputKindMouse = 0x00000020u;
inline constexpr uint32_t kGameInputKindAny = 0x0FFFFFFFu;

namespace gamepad_button {
inline constexpr uint32_t kNone = 0x00000000u;
inline constexpr uint32_t kMenu = 0x00000001u;
inline constexpr uint32_t kView = 0x00000002u;
inline constexpr uint32_t kA = 0x00000004u;
inline constexpr uint32_t kB = 0x00000008u;
inline constexpr uint32_t kX = 0x00000010u;
inline constexpr uint32_t kY = 0x00000020u;
inline constexpr uint32_t kDPadUp = 0x00000040u;
inline constexpr uint32_t kDPadDown = 0x00000080u;
inline constexpr uint32_t kDPadLeft = 0x00000100u;
inline constexpr uint32_t kDPadRight = 0x00000200u;
inline constexpr uint32_t kLeftShoulder = 0x00000400u;
inline constexpr uint32_t kRightShoulder = 0x00000800u;
inline constexpr uint32_t kLeftThumbstick = 0x00001000u;
inline constexpr uint32_t kRightThumbstick = 0x00002000u;
}  // namespace gamepad_button

// Attaches the live playgta6::InputState SharedArrayBuffer read by GameInput.dll.
void set_game_input_state(playgta6::InputState* shared_state,
                          playgta6::os::SeqlockMode mode = playgta6::os::SeqlockMode::MonotonicSeq);
playgta6::InputState* get_game_input_state();
void set_game_input_deadzone(const playgta6::os::DeadzoneConfig& config);

// Registers all 16 Xbox Scarlett exports in Win32ModuleRegistry (idempotent, thread-safe).
size_t register_all_scarlett_exports();

}  // namespace playgta6::os::scarlett

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints (all 16 imports + module registration helper)
// ---------------------------------------------------------------------------

extern "C" {

// xmem.dll (12 imports)
// XMemSetAllocationHooks(XMemAllocRoutine*, XMemFreeRoutine*): guest routine addresses.
void pg6_xbox_XMemSetAllocationHooks(uint64_t alloc_routine, uint64_t free_routine);
void* pg6_xbox_XMemAlloc(size_t size, uint64_t attributes);
void* pg6_xbox_XMemAllocDefault(size_t size, uint64_t attributes);
void pg6_xbox_XMemFree(void* address, uint64_t attributes);
void pg6_xbox_XMemFreeDefault(void* address, uint64_t attributes);
// Physical pages, in the shapes of the title's call sites (O27; the AWE-style
// AllocateUserPhysicalPages / MapUserPhysicalPages[Scatter] argument order):
//   XMemAllocatePhysicalPages(attributes, &count, pfns) -> 1 / 0
//   XMemMapPhysicalPages(address, count, pfns) -> address (pfns null: unmap the slots); a null
//     address places the pages in a fresh guest block (host callers; xmem_unmap_physical_pages)
//   XMemMapPhysicalPagesScatter(addresses, count, pfns) -> 1 / 0 (null array or pfn 0: unmap)
// `addresses` holds guest VAs (it is an array the import bridge does not translate).
int32_t pg6_xbox_XMemAllocatePhysicalPages(uint64_t attributes, size_t* num_pages, uint64_t* page_array);
int32_t pg6_xbox_XMemFreePhysicalPages(size_t num_pages, const uint64_t* page_array);
void* pg6_xbox_XMemMapPhysicalPages(void* virtual_address, size_t num_pages, const uint64_t* page_array);
int32_t pg6_xbox_XMemMapPhysicalPagesScatter(const uint64_t* virtual_addresses,
                                             size_t num_pages,
                                             const uint64_t* page_array);
// XMemVirtualAlloc(address, size, flAllocationType, XMem flags, flProtect): VirtualAlloc plus
// an XMem flags argument in the 4th position. The order is taken from the title's call sites
// (reserve with MEM_64K_PAGES | MEM_RESERVE, XMem flags 0x21 / 0x3, protection PAGE_READWRITE /
// PAGE_GRAPHICS_NOACCESS | PAGE_NOACCESS); the XMem flags are accepted and ignored.
void* pg6_xbox_XMemVirtualAlloc(void* address,
                                size_t size,
                                uint32_t allocation_type,
                                uint32_t xmem_flags,
                                uint32_t protect);
Pg6HResult pg6_xbox_XMemTransferMemory(void* dst, const void* src, size_t size, uint32_t flags);
Pg6HResult pg6_xbox_XMemGetWorkingSetStatistics(Pg6XMemWorkingSetStats* stats);
int32_t pg6_xbox_XMemIsToolsMemAvailable(void);

// XCRDAPI.dll (1 import)
Pg6HResult pg6_xbox_DStorageGetFactory(const Pg6Guid* riid, void** ppv);

// xg_xs.dll (2 imports)
Pg6HResult pg6_xbox_XGCreateTextureComputer(const Pg6XgTextureDesc* desc,
                                            Pg6IXGTextureAddressComputer** out_computer);
uint32_t pg6_xbox_XGSuggestSwizzleMode(const Pg6XgTextureDesc* desc);

// GameInput.dll (1 import, ordinal #1)
Pg6HResult pg6_xbox_GameInputCreate(Pg6IGameInput** game_input);

// Module registry helper (registers all 16 exports across xmem.dll, XCRDAPI.dll, xg_xs.dll, GameInput.dll)
size_t pg6_xbox_scarlett_register_all(void);

}  // extern "C"
