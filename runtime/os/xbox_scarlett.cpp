#include "os/xbox_scarlett.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "os/guest_heap.h"
#include "os/httpfs_mailbox.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"

namespace playgta6::os::scarlett {

namespace {

using gdk::E_FAIL_HR;
using gdk::E_INVALIDARG_HR;
using gdk::E_OUTOFMEMORY_HR;
using gdk::E_POINTER_HR;
using gdk::S_OK_HR;

constexpr size_t kMinAlignment = 16;
constexpr size_t kMaxHeapAlignment = 2 * 1024 * 1024;  // 2 MiB
constexpr uint32_t kDefaultStagingBufferSize = 32u * 1024u * 1024u;
constexpr uint32_t kMinStagingBufferSize = 64u * 1024u;
constexpr uint32_t kMaxStagingBufferSize = 256u * 1024u * 1024u;

uint64_t steady_microseconds() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

// All XMem memory comes from the guest address space (os/guest_heap.h, os/vmem.h): heap blocks and
// physical-page backing from the guest heap (HeapFamily::XMem), XMemVirtualAlloc regions straight
// from the arena. The shims below hand out host pointers (the import bridge translates top-level
// results to guest VAs) and convert back with guest_of() when looking blocks up.
VirtualMemoryArena& xmem_arena() {
    return *win32::get_vmem_arena();
}

uint64_t guest_of(const void* host) {
    return host == nullptr ? 0 : xmem_arena().guest_address(host);
}

void* host_of(uint64_t va) {
    return va == 0 ? nullptr : xmem_arena().translate(va);
}

void* aligned_alloc_bytes(size_t size, size_t alignment, bool zero = false) {
    GuestHeap& heap = default_guest_heap();
    const uint64_t va = heap.alloc(size,
                                   std::max(alignment, kMinAlignment),
                                   HeapFamily::XMem,
                                   zero ? kHeapZeroMemory : 0u);
    return heap.to_host(va);
}

void aligned_free_bytes(void* ptr, size_t alignment) {
    (void)alignment;
    if (ptr == nullptr) {
        return;
    }
    GuestHeap& heap = default_guest_heap();
    heap.free(heap.to_guest(ptr));
}

// Win32 PAGE_* value -> arena protection; console modifier bits (PAGE_GUARD, NOCACHE,
// WRITECOMBINE, GPU hints) in bits 8 and up are accepted and ignored.
bool xmem_protect(uint32_t protect, PageProtect& out) {
    switch (protect & 0xFFu) {
        case win32::PAGE_NOACCESS: out = PageProtect::NoAccess; return true;
        case win32::PAGE_READONLY: out = PageProtect::ReadOnly; return true;
        case win32::PAGE_READWRITE:
        case 0x08u: out = PageProtect::ReadWrite; return true;  // PAGE_WRITECOPY
        case win32::PAGE_EXECUTE_READ: out = PageProtect::ExecuteRead; return true;
        case win32::PAGE_EXECUTE_READWRITE:
        case 0x80u: out = PageProtect::ExecuteReadWrite; return true;  // PAGE_EXECUTE_WRITECOPY
        default: return false;
    }
}

size_t decode_xmem_alignment(uint64_t attributes) {
    const uint64_t code = (attributes >> kXMemAttrAlignmentShift) & kXMemAttrAlignmentMask;
    if (code == 0) {
        return kMinAlignment;
    }
    if (code <= 20) {
        const size_t requested = size_t{1} << (code + 1u);
        return std::clamp(requested, kMinAlignment, kMaxHeapAlignment);
    }
    return static_cast<size_t>(kXMemPageSize64K);
}

uint64_t decode_physical_page_size(uint32_t flags) {
    if ((flags & kXMemFlagPage2M) != 0 || (flags & kXMemFlagLargePages) != 0) {
        return kXMemPageSize2M;
    }
    if ((flags & kXMemFlagPage4K) != 0) {
        return kXMemPageSize4K;
    }
    return kXMemPageSize64K;
}

uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : (value + (alignment - rem));
}

// ===========================================================================
// 1. xmem.dll internal state
// ===========================================================================

struct HeapAllocation {
    size_t size = 0;
    size_t alignment = kMinAlignment;
    uint64_t attributes = 0;
};

struct PhysicalPage {
    uint64_t handle = 0;
    uint64_t page_size = kXMemPageSize64K;
    uint32_t protect = win32::PAGE_READWRITE;
    uint32_t flags = 0;
    // Guest heap block holding the page contents while the page is not mapped; 0 means "all
    // zero, or currently inside a mapping" (see the physical page model below).
    uint64_t store_va = 0;
};

struct VirtualRegion {
    void* base = nullptr;
    size_t size = 0;
    size_t alignment = static_cast<size_t>(kXMemPageSize64K);
    uint32_t allocation_type = 0;
    uint32_t protect = win32::PAGE_READWRITE;
    bool committed = false;
};

struct PhysicalMapping {
    void* base = nullptr;
    size_t size = 0;
    size_t alignment = static_cast<size_t>(kXMemPageSize64K);
    uint64_t page_size = kXMemPageSize64K;
    uint32_t protect = win32::PAGE_READWRITE;
    uint32_t flags = 0;
    bool owns_buffer = true;
    std::vector<uint64_t> page_handles;
    std::vector<uint64_t> virtual_offsets;
};

struct XMemState {
    std::mutex mutex;
    XMemAllocHookFn alloc_hook = nullptr;
    XMemFreeHookFn free_hook = nullptr;

    std::unordered_map<void*, HeapAllocation> heap_allocs;
    std::unordered_map<uint64_t, std::shared_ptr<PhysicalPage>> physical_pages;
    std::unordered_map<void*, VirtualRegion> virtual_regions;
    std::unordered_map<void*, PhysicalMapping> mappings;

    uint64_t next_page_handle = 0x1000ULL;
    uint64_t heap_allocated_bytes = 0;
    uint64_t physical_pages_bytes = 0;
    uint64_t virtual_committed_bytes = 0;
    uint64_t virtual_reserved_bytes = 0;
    uint64_t mapped_physical_bytes = 0;
    uint64_t peak_working_set_bytes = 0;

    void update_peak_locked() {
        const uint64_t current =
            heap_allocated_bytes + physical_pages_bytes + virtual_committed_bytes;
        if (current > peak_working_set_bytes) {
            peak_working_set_bytes = current;
        }
    }

    VirtualRegion* find_virtual_region_locked(void* ptr, size_t needed_bytes) {
        if (ptr == nullptr) {
            return nullptr;
        }
        auto it = virtual_regions.find(ptr);
        if (it == virtual_regions.end()) {
            return nullptr;
        }
        if (it->second.size < needed_bytes) {
            return nullptr;
        }
        return &it->second;
    }

    // Guest memory belongs to the arena; nothing to give back at process exit (the arena may
    // already be gone during static destruction).
};

XMemState& xmem_state() {
    static XMemState state;
    return state;
}

// Physical page model
// -------------------
// XMemAllocatePhysicalPages returns opaque page frame numbers for runtime-tracked pages; the
// pages own no memory until they hold data. Mapping a page set (XMemMapPhysicalPages[Scatter])
// materialises the pages in the guest address space - in a fresh guest heap block, or committed
// inside an XMemVirtualAlloc region - by MOVING each page's stored contents there (the store is
// freed). Unmapping, freeing the region, or xmem_reset_for_testing() moves the bytes back into a
// new store (all-zero pages need no store). A page's bytes therefore exist exactly once in guest
// memory and persist across unmap / map cycles; mapping the same page at two addresses at once
// is not coherent (the second mapping starts zeroed), as before.
void save_page_contents(PhysicalPage& page, const uint8_t* bytes) {
    GuestHeap& heap = default_guest_heap();
    if (page.store_va != 0) {
        heap.free(page.store_va);
        page.store_va = 0;
    }
    const size_t n = static_cast<size_t>(page.page_size);
    if (std::none_of(bytes, bytes + n, [](uint8_t b) { return b != 0; })) {
        return;
    }
    const uint64_t va = heap.alloc(n, kPageSize, HeapFamily::XMem);
    if (va == 0) {
        std::fprintf(stderr, "[xmem] out of guest memory saving a physical page; contents lost\n");
        return;
    }
    std::memcpy(heap.to_host(va), bytes, n);
    page.store_va = va;
}

void sync_mapping_to_pages_locked(XMemState& st, const PhysicalMapping& m) {
    const auto* src_bytes = static_cast<const uint8_t*>(m.base);
    for (size_t i = 0; i < m.page_handles.size(); ++i) {
        auto pit = st.physical_pages.find(m.page_handles[i]);
        if (pit != st.physical_pages.end()) {
            save_page_contents(*pit->second, src_bytes + m.virtual_offsets[i]);
        }
    }
    st.mapped_physical_bytes -= m.page_handles.size() * m.page_size;
}

void release_virtual_region(const VirtualRegion& region) {
    xmem_arena().release(guest_of(region.base));
}

void* map_physical_pages_impl(void* virtual_address,
                              size_t num_pages,
                              uint32_t protect,
                              const uint64_t* page_array,
                              const uint64_t* virtual_offsets,
                              uint32_t flags) {
    if (num_pages == 0 || page_array == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);

    std::vector<std::shared_ptr<PhysicalPage>> pages;
    pages.reserve(num_pages);
    uint64_t page_size = 0;
    for (size_t i = 0; i < num_pages; ++i) {
        auto it = st.physical_pages.find(page_array[i]);
        if (it == st.physical_pages.end()) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        if (i == 0) {
            page_size = it->second->page_size;
        } else if (it->second->page_size != page_size) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        pages.push_back(it->second);
    }

    std::vector<uint64_t> offsets(num_pages);
    uint64_t span_bytes = 0;
    for (size_t i = 0; i < num_pages; ++i) {
        const uint64_t off = virtual_offsets != nullptr ? virtual_offsets[i] : (i * page_size);
        if ((off % page_size) != 0) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        if (off > UINT64_MAX - page_size) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        offsets[i] = off;
        span_bytes = std::max(span_bytes, off + page_size);
    }

    const size_t align = static_cast<size_t>(std::min<uint64_t>(page_size, kXMemPageSize64K));
    void* target_base = nullptr;
    bool owns_buffer = true;

    if (virtual_address != nullptr) {
        if (st.mappings.find(virtual_address) != st.mappings.end()) {
            win32::SetLastError(win32::ERROR_ALREADY_EXISTS);
            return nullptr;
        }
        VirtualRegion* region = st.find_virtual_region_locked(
            virtual_address, static_cast<size_t>(span_bytes));
        if (region == nullptr) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        target_base = region->base;
        owns_buffer = false;
        // The mapping's protection applies to the mapped span (the region may have been
        // reserved or committed no-access).
        PageProtect pp = PageProtect::ReadWrite;
        xmem_protect(protect, pp);
        if (!region->committed) {
            if (!xmem_arena().commit(guest_of(region->base), region->size, pp)) {
                win32::SetLastError(win32::ERROR_NOT_ENOUGH_MEMORY);
                return nullptr;
            }
            region->committed = true;
            region->protect = protect;
            st.virtual_committed_bytes += region->size;
        } else {
            xmem_arena().protect(guest_of(region->base),
                                 static_cast<size_t>(align_up_u64(span_bytes, kPageSize)), pp);
        }
    } else {
        target_base = aligned_alloc_bytes(static_cast<size_t>(span_bytes), align, true);
        if (target_base == nullptr) {
            win32::SetLastError(win32::ERROR_NOT_ENOUGH_MEMORY);
            return nullptr;
        }
    }

    // Move the pages' stored contents into the mapping (gaps and unwritten pages stay zero).
    auto* dst_bytes = static_cast<uint8_t*>(target_base);
    for (size_t i = 0; i < num_pages; ++i) {
        if (pages[i]->store_va != 0) {
            std::memcpy(dst_bytes + offsets[i],
                        host_of(pages[i]->store_va),
                        static_cast<size_t>(page_size));
            default_guest_heap().free(pages[i]->store_va);
            pages[i]->store_va = 0;
        } else if (!owns_buffer) {
            std::memset(dst_bytes + offsets[i], 0, static_cast<size_t>(page_size));
        }
    }

    PhysicalMapping mapping;
    mapping.base = target_base;
    mapping.size = static_cast<size_t>(span_bytes);
    mapping.alignment = align;
    mapping.page_size = page_size;
    mapping.protect = protect;
    mapping.flags = flags;
    mapping.owns_buffer = owns_buffer;
    mapping.page_handles.assign(page_array, page_array + num_pages);
    mapping.virtual_offsets = std::move(offsets);

    st.mapped_physical_bytes += num_pages * page_size;
    st.mappings.emplace(target_base, std::move(mapping));
    st.update_peak_locked();
    return target_base;
}

// ===========================================================================
// 2. XCRDAPI.dll (DirectStorage) internal state & COM implementations
// ===========================================================================

struct DStorageGlobalState {
    std::mutex mutex;
    std::unordered_map<std::string, std::vector<uint8_t>> virtual_files;
    DStorageHostOpenCallback host_open_cb;
    DStorageHostReadCallback host_read_cb;
    uint32_t staging_buffer_size = kDefaultStagingBufferSize;
};

DStorageGlobalState& dstorage_state() {
    static DStorageGlobalState state;
    return state;
}

enum class DStorageFileBackend {
    VirtualTable,
    HostCallback,
    Httpfs,
    UserdataOverlay,
};

class DStorageFileImpl final : public Pg6IDStorageFile {
public:
    DStorageFileImpl(std::string normalized_path,
                     DStorageFileBackend backend,
                     uint64_t file_size,
                     int32_t httpfs_handle = -1)
        : path_(std::move(normalized_path)),
          backend_(backend),
          file_size_(file_size),
          httpfs_handle_(httpfs_handle) {}

    ~DStorageFileImpl() override { Close(); }

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIDStorageFile) {
            AddRef();
            *ppv_object = static_cast<Pg6IDStorageFile*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    void Close() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        closed_ = true;
        if (backend_ == DStorageFileBackend::Httpfs && httpfs_handle_ >= 0) {
            if (auto* client = win32::get_httpfs_client(); client != nullptr) {
                client->close(httpfs_handle_);
            }
            httpfs_handle_ = -1;
        }
    }

    Pg6HResult GetFileInformation(Pg6DStorageFileInfo* info) override {
        if (info == nullptr) {
            return E_POINTER_HR;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return gdk::E_HANDLE_HR;
        }
        info->file_size = file_size_;
        return S_OK_HR;
    }

    Pg6HResult read_at(uint64_t offset, void* dst, uint32_t size) {
        if (size == 0) {
            return S_OK_HR;
        }
        if (dst == nullptr) {
            return E_POINTER_HR;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return gdk::E_HANDLE_HR;
        }

        switch (backend_) {
            case DStorageFileBackend::VirtualTable: {
                auto& ds = dstorage_state();
                std::lock_guard<std::mutex> glock(ds.mutex);
                auto it = ds.virtual_files.find(path_);
                if (it == ds.virtual_files.end()) {
                    return E_FILE_NOT_FOUND_HR;
                }
                const auto& bytes = it->second;
                if (offset > bytes.size() || size > bytes.size() - offset) {
                    return E_EOF_HR;
                }
                std::memcpy(dst, bytes.data() + offset, size);
                return S_OK_HR;
            }
            case DStorageFileBackend::HostCallback: {
                DStorageHostReadCallback read_cb;
                {
                    auto& ds = dstorage_state();
                    std::lock_guard<std::mutex> glock(ds.mutex);
                    read_cb = ds.host_read_cb;
                }
                if (!read_cb) {
                    return E_FAIL_HR;
                }
                const int64_t n = read_cb(path_, offset, dst, size);
                if (n < 0) {
                    return E_FAIL_HR;
                }
                if (static_cast<uint64_t>(n) < size) {
                    return E_EOF_HR;
                }
                return S_OK_HR;
            }
            case DStorageFileBackend::Httpfs: {
                auto* client = win32::get_httpfs_client();
                if (client == nullptr || httpfs_handle_ < 0) {
                    return E_FAIL_HR;
                }
                const int64_t n = client->read(httpfs_handle_, offset, dst, size);
                if (n < 0) {
                    return E_FAIL_HR;
                }
                if (static_cast<uint64_t>(n) < size) {
                    return E_EOF_HR;
                }
                return S_OK_HR;
            }
            case DStorageFileBackend::UserdataOverlay: {
                std::vector<uint8_t> data;
                if (!win32::read_userdata_file(path_, data)) {
                    return E_FILE_NOT_FOUND_HR;
                }
                if (offset > data.size() || size > data.size() - offset) {
                    return E_EOF_HR;
                }
                std::memcpy(dst, data.data() + offset, size);
                return S_OK_HR;
            }
        }
        return E_FAIL_HR;
    }

private:
    std::atomic<uint32_t> ref_count_{1};
    std::mutex mutex_;
    std::string path_;
    DStorageFileBackend backend_;
    uint64_t file_size_ = 0;
    int32_t httpfs_handle_ = -1;
    bool closed_ = false;
};

class DStorageStatusArrayImpl final : public Pg6IDStorageStatusArray {
public:
    explicit DStorageStatusArrayImpl(uint32_t capacity)
        : complete_(capacity, 0u), results_(capacity, S_OK_HR) {}

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIDStorageStatusArray) {
            AddRef();
            *ppv_object = static_cast<Pg6IDStorageStatusArray*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    bool IsComplete(uint32_t index) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= complete_.size()) {
            return false;
        }
        return complete_[index] != 0u;
    }

    Pg6HResult GetHResult(uint32_t index) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= results_.size()) {
            return E_INVALIDARG_HR;
        }
        return results_[index];
    }

    void mark_pending(uint32_t index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < complete_.size()) {
            complete_[index] = 0u;
            results_[index] = gdk::E_PENDING_HR;
        }
    }

    void mark_complete(uint32_t index, Pg6HResult hr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < complete_.size()) {
            results_[index] = hr;
            complete_[index] = 1u;
        }
    }

private:
    std::atomic<uint32_t> ref_count_{1};
    std::mutex mutex_;
    std::vector<uint8_t> complete_;
    std::vector<Pg6HResult> results_;
};

class DStorageQueueImpl final : public Pg6IDStorageQueue {
public:
    explicit DStorageQueueImpl(const Pg6DStorageQueueDesc& desc)
        : capacity_(desc.capacity == 0 ? 256u : desc.capacity) {}

    ~DStorageQueueImpl() override { Close(); }

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIDStorageQueue) {
            AddRef();
            *ppv_object = static_cast<Pg6IDStorageQueue*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    void EnqueueRequest(const Pg6DStorageRequest* request) override {
        if (request == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        QueueItem item;
        item.kind = ItemKind::Read;
        item.file = static_cast<DStorageFileImpl*>(request->file);
        if (item.file != nullptr) {
            item.file->AddRef();
        }
        item.offset = request->offset;
        item.size = request->size;
        item.destination = request->destination;
        item.destination_size =
            request->destination_size == 0 ? request->size : request->destination_size;
        item.cancellation_tag = request->cancellation_tag;
        pending_.push_back(std::move(item));
    }

    void EnqueueRead(Pg6IDStorageFile* file,
                     uint64_t offset,
                     void* destination,
                     uint32_t size) override {
        Pg6DStorageRequest req{};
        req.file = file;
        req.offset = offset;
        req.size = size;
        req.destination_size = size;
        req.destination = destination;
        EnqueueRequest(&req);
    }

    void EnqueueStatus(Pg6IDStorageStatusArray* status_array, uint32_t index) override {
        if (status_array == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        auto* impl = static_cast<DStorageStatusArrayImpl*>(status_array);
        impl->AddRef();
        impl->mark_pending(index);
        QueueItem item;
        item.kind = ItemKind::Status;
        item.status_array = impl;
        item.status_index = index;
        pending_.push_back(std::move(item));
    }

    void EnqueueSignal(Pg6Handle event_handle, uint64_t value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        QueueItem item;
        item.kind = ItemKind::Signal;
        item.signal_handle = event_handle;
        item.signal_value = value;
        pending_.push_back(std::move(item));
    }

    void Submit() override {
        std::vector<QueueItem> batch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            batch.swap(pending_);
        }

        Pg6HResult batch_hr = S_OK_HR;
        for (auto& item : batch) {
            switch (item.kind) {
                case ItemKind::Read: {
                    if (item.cancelled) {
                        if (item.file != nullptr) {
                            item.file->Release();
                        }
                        break;
                    }
                    Pg6HResult hr = S_OK_HR;
                    if (item.file == nullptr || (item.size > 0 && item.destination == nullptr)) {
                        hr = E_INVALIDARG_HR;
                    } else if (item.destination_size < item.size) {
                        hr = gdk::E_NOT_SUFFICIENT_BUFFER_HR;
                    } else {
                        hr = item.file->read_at(item.offset, item.destination, item.size);
                    }
                    if (gdk::failed(hr)) {
                        record_error(hr, item.offset, item.size);
                        if (gdk::succeeded(batch_hr)) {
                            batch_hr = hr;
                        }
                    }
                    if (item.file != nullptr) {
                        item.file->Release();
                    }
                    break;
                }
                case ItemKind::Status: {
                    if (item.status_array != nullptr) {
                        item.status_array->mark_complete(item.status_index, batch_hr);
                        item.status_array->Release();
                    }
                    batch_hr = S_OK_HR;
                    break;
                }
                case ItemKind::Signal: {
                    if (item.signal_handle != win32::kNullHandle) {
                        win32::SetEvent(item.signal_handle);
                    }
                    break;
                }
            }
        }
    }

    void CancelRequestsWithTag(uint64_t mask, uint64_t value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& item : pending_) {
            if (item.kind == ItemKind::Read && (item.cancellation_tag & mask) == value) {
                item.cancelled = true;
            }
        }
    }

    void Close() override {
        std::vector<QueueItem> leftover;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
            leftover.swap(pending_);
        }
        for (auto& item : leftover) {
            if (item.file != nullptr) {
                item.file->Release();
            }
            if (item.status_array != nullptr) {
                item.status_array->mark_complete(item.status_index, gdk::E_ABORT_HR);
                item.status_array->Release();
            }
        }
    }

    uint32_t RetrieveErrorRecord(Pg6DStorageErrorRecord* record) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint32_t count = error_record_.failure_count;
        if (record != nullptr) {
            *record = error_record_;
        }
        error_record_ = Pg6DStorageErrorRecord{};
        return count;
    }

    uint32_t QueryPendingCount() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<uint32_t>(pending_.size());
    }

private:
    enum class ItemKind {
        Read,
        Status,
        Signal,
    };

    struct QueueItem {
        ItemKind kind = ItemKind::Read;
        DStorageFileImpl* file = nullptr;
        uint64_t offset = 0;
        uint32_t size = 0;
        uint32_t destination_size = 0;
        void* destination = nullptr;
        uint64_t cancellation_tag = 0;
        bool cancelled = false;
        DStorageStatusArrayImpl* status_array = nullptr;
        uint32_t status_index = 0;
        Pg6Handle signal_handle = win32::kNullHandle;
        uint64_t signal_value = 0;
    };

    void record_error(Pg6HResult hr, uint64_t offset, uint32_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_record_.failure_count == 0) {
            error_record_.first_failure_hresult = hr;
            error_record_.first_failure_offset = offset;
            error_record_.first_failure_size = size;
        }
        ++error_record_.failure_count;
    }

    std::atomic<uint32_t> ref_count_{1};
    mutable std::mutex mutex_;
    uint32_t capacity_ = 256;
    bool closed_ = false;
    std::vector<QueueItem> pending_;
    Pg6DStorageErrorRecord error_record_{};
};

class DStorageFactoryImpl final : public Pg6IDStorageFactory {
public:
    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIDStorageFactory) {
            AddRef();
            *ppv_object = static_cast<Pg6IDStorageFactory*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    Pg6HResult CreateQueue(const Pg6DStorageQueueDesc* desc,
                           const Pg6Guid& riid,
                           void** ppv) override {
        if (ppv == nullptr) {
            return E_POINTER_HR;
        }
        *ppv = nullptr;
        if (desc == nullptr) {
            return E_INVALIDARG_HR;
        }
        if (riid != kIidIUnknown && riid != kIidIDStorageQueue) {
            return E_NOINTERFACE_HR;
        }
        auto* queue = new (std::nothrow) DStorageQueueImpl(*desc);
        if (queue == nullptr) {
            return E_OUTOFMEMORY_HR;
        }
        *ppv = static_cast<Pg6IDStorageQueue*>(queue);
        return S_OK_HR;
    }

    Pg6HResult OpenFile(const char16_t* path, const Pg6Guid& riid, void** ppv) override {
        if (ppv == nullptr) {
            return E_POINTER_HR;
        }
        *ppv = nullptr;
        if (path == nullptr) {
            return E_INVALIDARG_HR;
        }
        const std::string utf8 = win32::utf16_to_utf8(path);
        return OpenFileA(utf8.c_str(), riid, ppv);
    }

    Pg6HResult OpenFileA(const char* path, const Pg6Guid& riid, void** ppv) override {
        if (ppv == nullptr) {
            return E_POINTER_HR;
        }
        *ppv = nullptr;
        if (path == nullptr || path[0] == '\0') {
            return E_INVALIDARG_HR;
        }
        if (riid != kIidIUnknown && riid != kIidIDStorageFile) {
            return E_NOINTERFACE_HR;
        }

        const std::string normalized = win32::normalize_path(path);
        if (normalized.empty()) {
            return E_INVALIDARG_HR;
        }

        // 1. In-memory virtual file table (highest priority).
        {
            auto& ds = dstorage_state();
            std::lock_guard<std::mutex> lock(ds.mutex);
            auto it = ds.virtual_files.find(normalized);
            if (it != ds.virtual_files.end()) {
                auto* file = new (std::nothrow) DStorageFileImpl(
                    normalized, DStorageFileBackend::VirtualTable, it->second.size());
                if (file == nullptr) {
                    return E_OUTOFMEMORY_HR;
                }
                *ppv = static_cast<Pg6IDStorageFile*>(file);
                return S_OK_HR;
            }
        }

        // 2. Host read callback hook.
        {
            DStorageHostOpenCallback open_cb;
            DStorageHostReadCallback read_cb;
            {
                auto& ds = dstorage_state();
                std::lock_guard<std::mutex> lock(ds.mutex);
                open_cb = ds.host_open_cb;
                read_cb = ds.host_read_cb;
            }
            if (read_cb) {
                const int64_t size = open_cb ? open_cb(normalized) : 0;
                if (size >= 0) {
                    auto* file = new (std::nothrow) DStorageFileImpl(
                        normalized, DStorageFileBackend::HostCallback, static_cast<uint64_t>(size));
                    if (file == nullptr) {
                        return E_OUTOFMEMORY_HR;
                    }
                    *ppv = static_cast<Pg6IDStorageFile*>(file);
                    return S_OK_HR;
                }
            }
        }

        // 3. HTTPFS mailbox client.
        if (auto* client = win32::get_httpfs_client(); client != nullptr) {
            const int32_t handle = client->open(normalized);
            if (handle >= 0) {
                auto* file = new (std::nothrow)
                    DStorageFileImpl(normalized, DStorageFileBackend::Httpfs, 0, handle);
                if (file == nullptr) {
                    client->close(handle);
                    return E_OUTOFMEMORY_HR;
                }
                *ppv = static_cast<Pg6IDStorageFile*>(file);
                return S_OK_HR;
            }
        }

        // 4. /userdata overlay fallback.
        if (win32::userdata_file_exists(normalized)) {
            std::vector<uint8_t> data;
            if (win32::read_userdata_file(normalized, data)) {
                auto* file = new (std::nothrow) DStorageFileImpl(
                    normalized, DStorageFileBackend::UserdataOverlay, data.size());
                if (file == nullptr) {
                    return E_OUTOFMEMORY_HR;
                }
                *ppv = static_cast<Pg6IDStorageFile*>(file);
                return S_OK_HR;
            }
        }

        return E_FILE_NOT_FOUND_HR;
    }

    Pg6HResult CreateStatusArray(uint32_t capacity,
                                 const char* name,
                                 const Pg6Guid& riid,
                                 void** ppv) override {
        (void)name;
        if (ppv == nullptr) {
            return E_POINTER_HR;
        }
        *ppv = nullptr;
        if (capacity == 0) {
            return E_INVALIDARG_HR;
        }
        if (riid != kIidIUnknown && riid != kIidIDStorageStatusArray) {
            return E_NOINTERFACE_HR;
        }
        auto* arr = new (std::nothrow) DStorageStatusArrayImpl(capacity);
        if (arr == nullptr) {
            return E_OUTOFMEMORY_HR;
        }
        *ppv = static_cast<Pg6IDStorageStatusArray*>(arr);
        return S_OK_HR;
    }

    Pg6HResult SetStagingBufferSize(uint32_t size) override {
        if (size < kMinStagingBufferSize || size > kMaxStagingBufferSize) {
            return E_INVALIDARG_HR;
        }
        auto& ds = dstorage_state();
        std::lock_guard<std::mutex> lock(ds.mutex);
        ds.staging_buffer_size = size;
        return S_OK_HR;
    }

    uint32_t GetStagingBufferSize() const override {
        auto& ds = dstorage_state();
        std::lock_guard<std::mutex> lock(ds.mutex);
        return ds.staging_buffer_size;
    }

private:
    std::atomic<uint32_t> ref_count_{1};
};

// ===========================================================================
// 3. xg_xs.dll (Texture Address Computer) implementation
// ===========================================================================

struct FormatMetrics {
    uint32_t block_width = 1;
    uint32_t block_height = 1;
    uint32_t bytes_per_block = 4;
};

FormatMetrics format_metrics(uint32_t dxgi_format) {
    switch (dxgi_format) {
        case kDxgiFormatR8Unorm:
            return {1, 1, 1};
        case kDxgiFormatR8G8Unorm:
        case kDxgiFormatR16Float:
            return {1, 1, 2};
        case kDxgiFormatR8G8B8A8Unorm:
        case kDxgiFormatR8G8B8A8UnormSrgb:
        case kDxgiFormatB8G8R8A8Unorm:
        case kDxgiFormatR32Float:
        case kDxgiFormatD32Float:
        case kDxgiFormatD24UnormS8Uint:
            return {1, 1, 4};
        case kDxgiFormatR16G16B16A16Float:
            return {1, 1, 8};
        case kDxgiFormatR32G32B32A32Float:
            return {1, 1, 16};
        case kDxgiFormatBc1Unorm:
        case kDxgiFormatBc1UnormSrgb:
        case kDxgiFormatBc4Unorm:
            return {4, 4, 8};
        case kDxgiFormatBc2Unorm:
        case kDxgiFormatBc3Unorm:
        case kDxgiFormatBc3UnormSrgb:
        case kDxgiFormatBc5Unorm:
        case kDxgiFormatBc6hUf16:
        case kDxgiFormatBc7Unorm:
        case kDxgiFormatBc7UnormSrgb:
            return {4, 4, 16};
        default:
            return {1, 1, 4};
    }
}

uint32_t compute_full_mip_count(uint32_t width, uint32_t height, uint32_t depth) {
    uint32_t dim = std::max({width, height, depth, 1u});
    uint32_t mips = 1;
    while (dim > 1) {
        dim >>= 1u;
        ++mips;
    }
    return std::min(mips, 16u);
}

class XGTextureAddressComputerImpl final : public Pg6IXGTextureAddressComputer {
public:
    explicit XGTextureAddressComputerImpl(const Pg6XgTextureDesc& desc) : desc_(desc) {
        metrics_ = format_metrics(desc.format);
        const bool is_3d = desc.dimension == kXgDimensionTexture3D;
        const bool is_1d = desc.dimension == kXgDimensionTexture1D;
        const uint32_t base_height = is_1d ? 1u : std::max(desc.height, 1u);
        const uint32_t base_depth = is_3d ? std::max(desc.depth_or_array_size, 1u) : 1u;
        const uint32_t array_size = is_3d ? 1u : std::max(desc.depth_or_array_size, 1u);

        const uint32_t max_mips = compute_full_mip_count(desc.width, base_height, base_depth);
        const uint32_t mips =
            desc.mip_levels == 0 ? max_mips : std::min({desc.mip_levels, max_mips, 16u});

        const uint32_t pitch_align = desc.pitch_alignment == 0 ? 256u : desc.pitch_alignment;
        const uint64_t mip_placement_align =
            desc.tile_mode == kXgTileModeStandard64KB ? kXMemPageSize64K : 512ULL;
        const uint64_t base_align =
            desc.tile_mode == kXgTileModeLinear ? 256ULL : kXMemPageSize64K;

        layout_.mip_levels = mips;
        layout_.array_size = array_size;
        layout_.tile_mode = desc.tile_mode;
        layout_.base_alignment_bytes = base_align;

        uint64_t cursor = 0;
        for (uint32_t m = 0; m < mips; ++m) {
            const uint32_t mip_w = std::max(1u, desc.width >> m);
            const uint32_t mip_h = is_1d ? 1u : std::max(1u, base_height >> m);
            const uint32_t mip_d = is_3d ? std::max(1u, base_depth >> m) : 1u;

            const uint32_t blocks_x =
                (mip_w + metrics_.block_width - 1u) / metrics_.block_width;
            const uint32_t blocks_y =
                (mip_h + metrics_.block_height - 1u) / metrics_.block_height;

            const uint32_t raw_row_bytes = blocks_x * metrics_.bytes_per_block;
            const uint32_t row_pitch =
                static_cast<uint32_t>(align_up_u64(raw_row_bytes, pitch_align));
            const uint64_t slice_size = static_cast<uint64_t>(row_pitch) * blocks_y;
            const uint64_t mip_total =
                slice_size * static_cast<uint64_t>(is_3d ? mip_d : array_size);

            cursor = align_up_u64(cursor, mip_placement_align);

            Pg6XgMipLevelLayout& ml = layout_.mips[m];
            ml.offset_bytes = cursor;
            ml.size_bytes = mip_total;
            ml.slice_size_bytes = slice_size;
            ml.width = mip_w;
            ml.height = mip_h;
            ml.depth = mip_d;
            ml.row_pitch_bytes = row_pitch;
            ml.padded_rows = blocks_y;
            ml.tile_mode = desc.tile_mode;

            cursor += mip_total;
        }

        layout_.size_bytes = align_up_u64(cursor, base_align);
    }

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIXGTextureAddressComputer) {
            AddRef();
            *ppv_object = static_cast<Pg6IXGTextureAddressComputer*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    Pg6HResult GetResourceLayout(Pg6XgResourceLayout* out_layout) override {
        if (out_layout == nullptr) {
            return E_POINTER_HR;
        }
        *out_layout = layout_;
        return S_OK_HR;
    }

    uint64_t GetResourceSizeBytes() override { return layout_.size_bytes; }

    uint64_t GetResourceBaseAlignmentBytes() override { return layout_.base_alignment_bytes; }

    uint64_t GetMipLevelOffsetBytes(uint32_t plane, uint32_t mip_level) override {
        if (plane != 0 || mip_level >= layout_.mip_levels) {
            return ~uint64_t{0};
        }
        return layout_.mips[mip_level].offset_bytes;
    }

    uint64_t GetTexelElementOffsetBytes(uint32_t plane,
                                        uint32_t mip_level,
                                        uint64_t x,
                                        uint32_t y,
                                        uint32_t z_or_slice,
                                        uint32_t sample) override {
        (void)sample;
        if (plane != 0 || mip_level >= layout_.mip_levels) {
            return ~uint64_t{0};
        }
        const Pg6XgMipLevelLayout& ml = layout_.mips[mip_level];
        const uint32_t max_slices =
            desc_.dimension == kXgDimensionTexture3D ? ml.depth : layout_.array_size;
        if (x >= ml.width || y >= ml.height || z_or_slice >= max_slices) {
            return ~uint64_t{0};
        }
        const uint64_t bx = x / metrics_.block_width;
        const uint64_t by = y / metrics_.block_height;
        return ml.offset_bytes + static_cast<uint64_t>(z_or_slice) * ml.slice_size_bytes +
               by * ml.row_pitch_bytes + bx * metrics_.bytes_per_block;
    }

    uint32_t GetMipLevelRowPitchBytes(uint32_t mip_level) override {
        if (mip_level >= layout_.mip_levels) {
            return 0;
        }
        return layout_.mips[mip_level].row_pitch_bytes;
    }

    uint64_t GetMipLevelSliceSizeBytes(uint32_t mip_level) override {
        if (mip_level >= layout_.mip_levels) {
            return 0;
        }
        return layout_.mips[mip_level].slice_size_bytes;
    }

    Pg6HResult CopyIntoSubresource(void* dst_base,
                                   uint32_t plane,
                                   uint32_t subresource,
                                   const void* src_data,
                                   uint32_t src_row_pitch,
                                   uint32_t src_slice_pitch) override {
        if (dst_base == nullptr || src_data == nullptr) {
            return E_POINTER_HR;
        }
        if (plane != 0 || layout_.mip_levels == 0) {
            return E_INVALIDARG_HR;
        }
        const uint32_t mip = subresource % layout_.mip_levels;
        const uint32_t slice = subresource / layout_.mip_levels;
        const bool is_3d = desc_.dimension == kXgDimensionTexture3D;
        if (!is_3d && slice >= layout_.array_size) {
            return E_INVALIDARG_HR;
        }
        if (is_3d && slice != 0) {
            return E_INVALIDARG_HR;
        }

        const Pg6XgMipLevelLayout& ml = layout_.mips[mip];
        const uint32_t blocks_x = (ml.width + metrics_.block_width - 1u) / metrics_.block_width;
        const uint32_t row_bytes = blocks_x * metrics_.bytes_per_block;
        const uint32_t depth_count = is_3d ? ml.depth : 1u;

        auto* dst_root = static_cast<uint8_t*>(dst_base) + ml.offset_bytes +
                         static_cast<uint64_t>(slice) * ml.slice_size_bytes;
        const auto* src_root = static_cast<const uint8_t*>(src_data);

        for (uint32_t z = 0; z < depth_count; ++z) {
            auto* dst_slice = dst_root + static_cast<uint64_t>(z) * ml.slice_size_bytes;
            const auto* src_slice = src_root + static_cast<uint64_t>(z) * src_slice_pitch;
            for (uint32_t r = 0; r < ml.padded_rows; ++r) {
                std::memcpy(dst_slice + static_cast<uint64_t>(r) * ml.row_pitch_bytes,
                            src_slice + static_cast<uint64_t>(r) * src_row_pitch,
                            row_bytes);
            }
        }
        return S_OK_HR;
    }

    Pg6HResult CopyFromSubresource(void* dst_data,
                                   uint32_t dst_row_pitch,
                                   uint32_t dst_slice_pitch,
                                   const void* src_base,
                                   uint32_t plane,
                                   uint32_t subresource) override {
        if (dst_data == nullptr || src_base == nullptr) {
            return E_POINTER_HR;
        }
        if (plane != 0 || layout_.mip_levels == 0) {
            return E_INVALIDARG_HR;
        }
        const uint32_t mip = subresource % layout_.mip_levels;
        const uint32_t slice = subresource / layout_.mip_levels;
        const bool is_3d = desc_.dimension == kXgDimensionTexture3D;
        if (!is_3d && slice >= layout_.array_size) {
            return E_INVALIDARG_HR;
        }
        if (is_3d && slice != 0) {
            return E_INVALIDARG_HR;
        }

        const Pg6XgMipLevelLayout& ml = layout_.mips[mip];
        const uint32_t blocks_x = (ml.width + metrics_.block_width - 1u) / metrics_.block_width;
        const uint32_t row_bytes = blocks_x * metrics_.bytes_per_block;
        const uint32_t depth_count = is_3d ? ml.depth : 1u;

        const auto* src_root = static_cast<const uint8_t*>(src_base) + ml.offset_bytes +
                               static_cast<uint64_t>(slice) * ml.slice_size_bytes;
        auto* dst_root = static_cast<uint8_t*>(dst_data);

        for (uint32_t z = 0; z < depth_count; ++z) {
            const auto* src_slice = src_root + static_cast<uint64_t>(z) * ml.slice_size_bytes;
            auto* dst_slice = dst_root + static_cast<uint64_t>(z) * dst_slice_pitch;
            for (uint32_t r = 0; r < ml.padded_rows; ++r) {
                std::memcpy(dst_slice + static_cast<uint64_t>(r) * dst_row_pitch,
                            src_slice + static_cast<uint64_t>(r) * ml.row_pitch_bytes,
                            row_bytes);
            }
        }
        return S_OK_HR;
    }

private:
    std::atomic<uint32_t> ref_count_{1};
    Pg6XgTextureDesc desc_{};
    FormatMetrics metrics_{};
    Pg6XgResourceLayout layout_{};
};

// ===========================================================================
// 4. GameInput.dll internal state & COM implementations
// ===========================================================================

struct GameInputGlobalState {
    std::mutex mutex;
    playgta6::InputState* shared_state = nullptr;
    GameInputReader reader;
    int64_t accum_mouse_x = 0;
    int64_t accum_mouse_y = 0;
    uint32_t focus_policy = 0;
};

GameInputGlobalState& game_input_state() {
    static GameInputGlobalState state;
    return state;
}

uint32_t map_sab_buttons_to_gamepad(uint32_t raw_buttons) {
    uint32_t out = gamepad_button::kNone;
    if ((raw_buttons & (input_button::kGamepadA | input_button::kJump)) != 0) {
        out |= gamepad_button::kA;
    }
    if ((raw_buttons & input_button::kGamepadB) != 0) {
        out |= gamepad_button::kB;
    }
    if ((raw_buttons & input_button::kGamepadX) != 0) {
        out |= gamepad_button::kX;
    }
    if ((raw_buttons & input_button::kGamepadY) != 0) {
        out |= gamepad_button::kY;
    }
    if ((raw_buttons & input_button::kLeftBumper) != 0) {
        out |= gamepad_button::kLeftShoulder;
    }
    if ((raw_buttons & input_button::kRightBumper) != 0) {
        out |= gamepad_button::kRightShoulder;
    }
    if ((raw_buttons & input_button::kMenu) != 0) {
        out |= gamepad_button::kMenu;
    }
    if ((raw_buttons & input_button::kView) != 0) {
        out |= gamepad_button::kView;
    }
    if ((raw_buttons & input_button::kSprint) != 0) {
        out |= gamepad_button::kLeftThumbstick;
    }
    return out;
}

class GameInputReadingImpl final : public Pg6IGameInputReading {
public:
    GameInputReadingImpl(uint32_t input_kind,
                         uint64_t sequence,
                         uint64_t timestamp,
                         uint32_t raw_buttons,
                         const Pg6GameInputGamepadState& gamepad,
                         const Pg6GameInputMouseState& mouse)
        : input_kind_(input_kind),
          sequence_(sequence),
          timestamp_(timestamp),
          raw_buttons_(raw_buttons),
          gamepad_(gamepad),
          mouse_(mouse) {}

    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIGameInputReading) {
            AddRef();
            *ppv_object = static_cast<Pg6IGameInputReading*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    uint32_t GetInputKind() override { return input_kind_; }

    uint64_t GetSequenceNumber(uint32_t input_kind) override {
        (void)input_kind;
        return sequence_;
    }

    uint64_t GetTimestamp() override { return timestamp_; }

    bool GetGamepadState(Pg6GameInputGamepadState* state) override {
        if (state == nullptr) {
            return false;
        }
        *state = gamepad_;
        return true;
    }

    bool GetMouseState(Pg6GameInputMouseState* state) override {
        if (state == nullptr) {
            return false;
        }
        *state = mouse_;
        return true;
    }

    uint32_t GetRawButtons() override { return raw_buttons_; }

private:
    std::atomic<uint32_t> ref_count_{1};
    uint32_t input_kind_ = kGameInputKindGamepad;
    uint64_t sequence_ = 0;
    uint64_t timestamp_ = 0;
    uint32_t raw_buttons_ = 0;
    Pg6GameInputGamepadState gamepad_{};
    Pg6GameInputMouseState mouse_{};
};

class GameInputImpl final : public Pg6IGameInput {
public:
    Pg6HResult QueryInterface(const Pg6Guid& riid, void** ppv_object) override {
        if (ppv_object == nullptr) {
            return E_POINTER_HR;
        }
        if (riid == kIidIUnknown || riid == kIidIGameInput) {
            AddRef();
            *ppv_object = static_cast<Pg6IGameInput*>(this);
            return S_OK_HR;
        }
        *ppv_object = nullptr;
        return E_NOINTERFACE_HR;
    }

    uint32_t AddRef() override { return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1; }

    uint32_t Release() override {
        const uint32_t next = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (next == 0) {
            delete this;
        }
        return next;
    }

    uint64_t GetCurrentTimestamp() override { return steady_microseconds(); }

    Pg6HResult GetCurrentReading(uint32_t input_kind,
                                 void* device,
                                 Pg6IGameInputReading** reading) override {
        (void)device;
        if (reading == nullptr) {
            return E_POINTER_HR;
        }
        *reading = nullptr;
        if (input_kind == kGameInputKindUnknown) {
            return E_INVALIDARG_HR;
        }

        Pg6GameInputGamepadState gp{};
        Pg6GameInputMouseState mouse{};
        uint64_t seq = 0;
        uint32_t raw_buttons = 0;

        {
            auto& gi = game_input_state();
            std::lock_guard<std::mutex> lock(gi.mutex);
            GameInputFrame frame{};
            if (gi.reader.is_attached() && gi.reader.poll(frame, true)) {
                seq = frame.seq;
                raw_buttons = frame.buttons;
                gp.buttons = map_sab_buttons_to_gamepad(frame.buttons);
                gp.left_trigger = frame.left_trigger;
                gp.right_trigger = frame.right_trigger;
                gp.left_thumbstick_x = frame.left_stick.x;
                gp.left_thumbstick_y = frame.left_stick.y;
                gp.right_thumbstick_x = frame.right_stick.x;
                gp.right_thumbstick_y = frame.right_stick.y;

                // Synthesize left stick from WASD when analog stick is neutral.
                if (gp.left_thumbstick_x == 0.0f && gp.left_thumbstick_y == 0.0f) {
                    float sx = 0.0f;
                    float sy = 0.0f;
                    if ((frame.buttons & input_button::kMoveRight) != 0) sx += 1.0f;
                    if ((frame.buttons & input_button::kMoveLeft) != 0) sx -= 1.0f;
                    if ((frame.buttons & input_button::kMoveForward) != 0) sy += 1.0f;
                    if ((frame.buttons & input_button::kMoveBackward) != 0) sy -= 1.0f;
                    const float mag = std::hypot(sx, sy);
                    if (mag > 1.0f) {
                        sx /= mag;
                        sy /= mag;
                    }
                    gp.left_thumbstick_x = sx;
                    gp.left_thumbstick_y = sy;
                }

                gi.accum_mouse_x += frame.mouse_delta.dx;
                gi.accum_mouse_y += frame.mouse_delta.dy;
                mouse.position_x = gi.accum_mouse_x;
                mouse.position_y = gi.accum_mouse_y;
            }
        }

        const uint32_t resolved_kind =
            (input_kind & kGameInputKindGamepad) != 0 ? kGameInputKindGamepad : input_kind;
        auto* impl = new (std::nothrow) GameInputReadingImpl(
            resolved_kind, seq, steady_microseconds(), raw_buttons, gp, mouse);
        if (impl == nullptr) {
            return E_OUTOFMEMORY_HR;
        }
        *reading = static_cast<Pg6IGameInputReading*>(impl);
        return S_OK_HR;
    }

    Pg6HResult GetNextReading(Pg6IGameInputReading* reference_reading,
                              uint32_t input_kind,
                              void* device,
                              Pg6IGameInputReading** reading) override {
        if (reading == nullptr) {
            return E_POINTER_HR;
        }
        *reading = nullptr;
        if (reference_reading == nullptr) {
            return GetCurrentReading(input_kind, device, reading);
        }
        Pg6IGameInputReading* latest = nullptr;
        const Pg6HResult hr = GetCurrentReading(input_kind, device, &latest);
        if (gdk::failed(hr)) {
            return hr;
        }
        if (latest->GetSequenceNumber(input_kind) <=
            reference_reading->GetSequenceNumber(input_kind)) {
            latest->Release();
            return GAMEINPUT_E_READING_NOT_FOUND;
        }
        *reading = latest;
        return S_OK_HR;
    }

    Pg6HResult GetPreviousReading(Pg6IGameInputReading* reference_reading,
                                  uint32_t input_kind,
                                  void* device,
                                  Pg6IGameInputReading** reading) override {
        (void)reference_reading;
        (void)input_kind;
        (void)device;
        if (reading == nullptr) {
            return E_POINTER_HR;
        }
        *reading = nullptr;
        return GAMEINPUT_E_READING_NOT_FOUND;
    }

    void SetFocusPolicy(uint32_t policy) override {
        auto& gi = game_input_state();
        std::lock_guard<std::mutex> lock(gi.mutex);
        gi.focus_policy = policy;
    }

private:
    std::atomic<uint32_t> ref_count_{1};
};

}  // namespace

// ---------------------------------------------------------------------------
// Public C++ helpers in playgta6::os::scarlett
// ---------------------------------------------------------------------------

void xmem_set_allocation_hooks(XMemAllocHookFn alloc_hook, XMemFreeHookFn free_hook) {
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.alloc_hook = alloc_hook;
    st.free_hook = free_hook;
}

bool xmem_get_allocation_info(const void* address, XMemAllocInfo* out_info) {
    if (address == nullptr) {
        return false;
    }
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    auto it = st.heap_allocs.find(const_cast<void*>(address));
    if (it == st.heap_allocs.end()) {
        return false;
    }
    if (out_info != nullptr) {
        out_info->size = it->second.size;
        out_info->alignment = it->second.alignment;
        out_info->attributes = it->second.attributes;
    }
    return true;
}

bool xmem_virtual_free(void* address) {
    if (address == nullptr) {
        return false;
    }
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    auto it = st.virtual_regions.find(address);
    if (it == st.virtual_regions.end()) {
        return false;
    }
    // Remove any physical mapping attached to this virtual reservation.
    auto mit = st.mappings.find(address);
    if (mit != st.mappings.end()) {
        sync_mapping_to_pages_locked(st, mit->second);
        st.mappings.erase(mit);
    }

    if (it->second.committed) {
        st.virtual_committed_bytes -= it->second.size;
    }
    st.virtual_reserved_bytes -= it->second.size;
    release_virtual_region(it->second);
    st.virtual_regions.erase(it);
    return true;
}

bool xmem_unmap_physical_pages(void* virtual_address) {
    if (virtual_address == nullptr) {
        return false;
    }
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    auto it = st.mappings.find(virtual_address);
    if (it == st.mappings.end()) {
        return false;
    }
    const PhysicalMapping& m = it->second;
    sync_mapping_to_pages_locked(st, m);
    if (m.owns_buffer) {
        aligned_free_bytes(m.base, m.alignment);
    }
    st.mappings.erase(it);
    return true;
}

void xmem_reset_for_testing() {
    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.alloc_hook = nullptr;
    st.free_hook = nullptr;
    for (auto& [ptr, rec] : st.heap_allocs) {
        aligned_free_bytes(ptr, rec.alignment);
    }
    st.heap_allocs.clear();
    for (auto& [ptr, mapping] : st.mappings) {
        if (mapping.owns_buffer) {
            aligned_free_bytes(ptr, mapping.alignment);
        }
    }
    st.mappings.clear();
    for (auto& [ptr, region] : st.virtual_regions) {
        release_virtual_region(region);
    }
    st.virtual_regions.clear();
    for (auto& [handle, page] : st.physical_pages) {
        if (page->store_va != 0) {
            default_guest_heap().free(page->store_va);
        }
    }
    st.physical_pages.clear();
    st.next_page_handle = 0x1000ULL;
    st.heap_allocated_bytes = 0;
    st.physical_pages_bytes = 0;
    st.virtual_committed_bytes = 0;
    st.virtual_reserved_bytes = 0;
    st.mapped_physical_bytes = 0;
    st.peak_working_set_bytes = 0;
}

void register_dstorage_virtual_file(std::string_view path, std::vector<uint8_t> data) {
    const std::string normalized = win32::normalize_path(path);
    if (normalized.empty()) {
        return;
    }
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.virtual_files[normalized] = std::move(data);
}

void unregister_dstorage_virtual_file(std::string_view path) {
    const std::string normalized = win32::normalize_path(path);
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.virtual_files.erase(normalized);
}

void clear_dstorage_virtual_files() {
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.virtual_files.clear();
}

void set_dstorage_host_callbacks(DStorageHostOpenCallback open_cb,
                                 DStorageHostReadCallback read_cb) {
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.host_open_cb = std::move(open_cb);
    ds.host_read_cb = std::move(read_cb);
}

void reset_dstorage_for_testing() {
    auto& ds = dstorage_state();
    std::lock_guard<std::mutex> lock(ds.mutex);
    ds.virtual_files.clear();
    ds.host_open_cb = nullptr;
    ds.host_read_cb = nullptr;
    ds.staging_buffer_size = kDefaultStagingBufferSize;
}

void set_game_input_state(playgta6::InputState* shared_state, SeqlockMode mode) {
    auto& gi = game_input_state();
    std::lock_guard<std::mutex> lock(gi.mutex);
    gi.shared_state = shared_state;
    gi.reader.attach(shared_state, mode);
    gi.accum_mouse_x = 0;
    gi.accum_mouse_y = 0;
}

playgta6::InputState* get_game_input_state() {
    auto& gi = game_input_state();
    std::lock_guard<std::mutex> lock(gi.mutex);
    return gi.shared_state;
}

void set_game_input_deadzone(const DeadzoneConfig& config) {
    auto& gi = game_input_state();
    std::lock_guard<std::mutex> lock(gi.mutex);
    gi.reader.set_deadzone(config);
}

size_t register_all_scarlett_exports() {
    using win32::export_entry;
    using win32::register_exports;

    static std::mutex reg_mutex;
    std::lock_guard<std::mutex> lock(reg_mutex);

    size_t total = 0;
    total += register_exports(
        "xmem.dll",
        {
            export_entry("XMemAlloc", &pg6_xbox_XMemAlloc),
            export_entry("XMemAllocDefault", &pg6_xbox_XMemAllocDefault),
            export_entry("XMemAllocatePhysicalPages", &pg6_xbox_XMemAllocatePhysicalPages),
            export_entry("XMemFree", &pg6_xbox_XMemFree),
            export_entry("XMemFreeDefault", &pg6_xbox_XMemFreeDefault),
            export_entry("XMemFreePhysicalPages", &pg6_xbox_XMemFreePhysicalPages),
            export_entry("XMemGetWorkingSetStatistics", &pg6_xbox_XMemGetWorkingSetStatistics),
            export_entry("XMemIsToolsMemAvailable", &pg6_xbox_XMemIsToolsMemAvailable),
            export_entry("XMemMapPhysicalPages", &pg6_xbox_XMemMapPhysicalPages),
            export_entry("XMemMapPhysicalPagesScatter", &pg6_xbox_XMemMapPhysicalPagesScatter),
            export_entry("XMemTransferMemory", &pg6_xbox_XMemTransferMemory),
            export_entry("XMemVirtualAlloc", &pg6_xbox_XMemVirtualAlloc),
        });

    total += register_exports(
        "XCRDAPI.dll",
        {
            export_entry("DStorageGetFactory", &pg6_xbox_DStorageGetFactory),
        });

    total += register_exports(
        "xg_xs.dll",
        {
            export_entry("XGCreateTextureComputer", &pg6_xbox_XGCreateTextureComputer),
            export_entry("XGSuggestSwizzleMode", &pg6_xbox_XGSuggestSwizzleMode),
        });

    total += register_exports(
        "GameInput.dll",
        {
            export_entry("GameInputCreate", &pg6_xbox_GameInputCreate, 1u),
        });

    return total;
}

}  // namespace playgta6::os::scarlett

// ---------------------------------------------------------------------------
// extern "C" entrypoints
// ---------------------------------------------------------------------------

extern "C" {

void* pg6_xbox_XMemAlloc(size_t size, uint64_t attributes) {
    playgta6::os::scarlett::XMemAllocHookFn hook = nullptr;
    {
        auto& st = playgta6::os::scarlett::xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        hook = st.alloc_hook;
    }
    if (hook != nullptr) {
        return hook(size, attributes);
    }
    return pg6_xbox_XMemAllocDefault(size, attributes);
}

void* pg6_xbox_XMemAllocDefault(size_t size, uint64_t attributes) {
    using namespace playgta6::os::scarlett;
    const size_t align = decode_xmem_alignment(attributes);
    void* ptr = aligned_alloc_bytes(size, align);
    if (ptr == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    if ((attributes & kXMemAttrZeroInitBit) != 0 && size > 0) {
        std::memset(ptr, 0, size);
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    st.heap_allocs.emplace(ptr, HeapAllocation{size, align, attributes});
    st.heap_allocated_bytes += size;
    st.update_peak_locked();
    return ptr;
}

void pg6_xbox_XMemFree(void* address, uint64_t attributes) {
    if (address == nullptr) {
        return;
    }
    playgta6::os::scarlett::XMemFreeHookFn hook = nullptr;
    {
        auto& st = playgta6::os::scarlett::xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        hook = st.free_hook;
    }
    if (hook != nullptr) {
        hook(address, attributes);
        return;
    }
    pg6_xbox_XMemFreeDefault(address, attributes);
}

void pg6_xbox_XMemFreeDefault(void* address, uint64_t attributes) {
    (void)attributes;
    if (address == nullptr) {
        return;
    }
    using namespace playgta6::os::scarlett;
    size_t align = kMinAlignment;
    {
        auto& st = xmem_state();
        std::lock_guard<std::mutex> lock(st.mutex);
        auto it = st.heap_allocs.find(address);
        if (it == st.heap_allocs.end()) {
            playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
            return;
        }
        align = it->second.alignment;
        st.heap_allocated_bytes -= it->second.size;
        st.heap_allocs.erase(it);
    }
    aligned_free_bytes(address, align);
}

int32_t pg6_xbox_XMemAllocatePhysicalPages(size_t* num_pages,
                                           uint64_t* page_array,
                                           uint32_t flags,
                                           uint32_t protect) {
    using namespace playgta6::os::scarlett;
    if (num_pages == nullptr || page_array == nullptr || *num_pages == 0) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return 0;
    }

    const uint64_t page_size = decode_physical_page_size(flags);
    const size_t count = *num_pages;

    std::vector<std::shared_ptr<PhysicalPage>> created;
    created.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto page = std::make_shared<PhysicalPage>();
        page->page_size = page_size;
        page->protect = protect;
        page->flags = flags;
        created.push_back(std::move(page));
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    for (size_t i = 0; i < count; ++i) {
        const uint64_t handle = st.next_page_handle++;
        created[i]->handle = handle;
        page_array[i] = handle;
        st.physical_pages.emplace(handle, std::move(created[i]));
        st.physical_pages_bytes += page_size;
    }
    st.update_peak_locked();
    return 1;
}

int32_t pg6_xbox_XMemFreePhysicalPages(size_t num_pages, const uint64_t* page_array) {
    using namespace playgta6::os::scarlett;
    if (num_pages == 0 || page_array == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return 0;
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    for (size_t i = 0; i < num_pages; ++i) {
        if (st.physical_pages.find(page_array[i]) == st.physical_pages.end()) {
            playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
            return 0;
        }
    }
    for (size_t i = 0; i < num_pages; ++i) {
        auto it = st.physical_pages.find(page_array[i]);
        if (it != st.physical_pages.end()) {
            st.physical_pages_bytes -= it->second->page_size;
            if (it->second->store_va != 0) {
                playgta6::os::default_guest_heap().free(it->second->store_va);
            }
            st.physical_pages.erase(it);
        }
    }
    return 1;
}

void* pg6_xbox_XMemMapPhysicalPages(void* virtual_address,
                                    size_t num_pages,
                                    uint32_t protect,
                                    const uint64_t* page_array,
                                    uint32_t flags) {
    return playgta6::os::scarlett::map_physical_pages_impl(
        virtual_address, num_pages, protect, page_array, nullptr, flags);
}

void* pg6_xbox_XMemMapPhysicalPagesScatter(void* virtual_address,
                                           size_t num_pages,
                                           uint32_t protect,
                                           const uint64_t* page_array,
                                           const uint64_t* virtual_offsets,
                                           uint32_t flags) {
    return playgta6::os::scarlett::map_physical_pages_impl(
        virtual_address, num_pages, protect, page_array, virtual_offsets, flags);
}

void* pg6_xbox_XMemVirtualAlloc(void* address,
                                size_t size,
                                uint32_t allocation_type,
                                uint32_t protect) {
    using namespace playgta6::os::scarlett;
    namespace os = playgta6::os;
    if (size == 0) {
        os::win32::SetLastError(os::win32::ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    // Console-only flags (MEM_LARGE_PAGES, MEM_TOP_DOWN, MEM_WRITE_WATCH, ...) are accepted; only
    // MEM_RESERVE / MEM_COMMIT select the operation, large pages are normal 64 KiB pages.
    const bool want_reserve = (allocation_type & os::kMemReserve) != 0;
    const bool want_commit = (allocation_type & os::kMemCommit) != 0;
    os::PageProtect pp = os::PageProtect::NoAccess;
    if ((!want_reserve && !want_commit) || !xmem_protect(protect, pp)) {
        os::win32::SetLastError(os::win32::ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const uint64_t page_align =
        (allocation_type & kXMemFlagLargePages) != 0 ? kXMemPageSize64K : os::kPageSize;
    const uint64_t aligned_size = align_up_u64(size, page_align);
    if (aligned_size < size) {
        os::win32::SetLastError(os::win32::ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    os::VirtualMemoryArena& arena = xmem_arena();

    // Committing an existing XMemVirtualAlloc reservation in place.
    if (address != nullptr && !want_reserve && want_commit) {
        VirtualRegion* existing =
            st.find_virtual_region_locked(address, static_cast<size_t>(aligned_size));
        if (existing != nullptr) {
            if (!existing->committed) {
                if (!arena.commit(guest_of(existing->base), existing->size, pp)) {
                    os::win32::SetLastError(os::win32::ERROR_NOT_ENOUGH_MEMORY);
                    return nullptr;
                }
                existing->committed = true;
                existing->protect = protect;
                st.virtual_committed_bytes += existing->size;
                st.update_peak_locked();
            }
            return existing->base;
        }
        // Not one of ours: plain VirtualAlloc semantics on whatever the arena has reserved there.
        const uint64_t committed =
            arena.virtual_alloc(guest_of(address), aligned_size, os::kMemCommit, pp);
        if (committed == 0) {
            os::win32::SetLastError(os::win32::ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        return host_of(committed);
    }

    // A non-null address is a placement request that must be honoured exactly.
    const uint32_t arena_type =
        (want_reserve ? os::kMemReserve : 0u) | (want_commit ? os::kMemCommit : 0u);
    uint64_t va = 0;
    try {
        va = arena.virtual_alloc(guest_of(address), aligned_size, arena_type, pp);
    } catch (const std::bad_alloc&) {
        va = 0;
    }
    if (va == 0) {
        os::win32::SetLastError(os::win32::ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }

    VirtualRegion region;
    region.base = host_of(va);
    region.size = static_cast<size_t>(aligned_size);
    region.alignment = static_cast<size_t>(kXMemPageSize64K);
    region.allocation_type = allocation_type;
    region.protect = protect;
    region.committed = want_commit;

    st.virtual_reserved_bytes += aligned_size;
    if (want_commit) {
        st.virtual_committed_bytes += aligned_size;
    }
    st.virtual_regions.emplace(region.base, region);
    st.update_peak_locked();
    return region.base;
}

Pg6HResult pg6_xbox_XMemTransferMemory(void* dst, const void* src, size_t size, uint32_t flags) {
    (void)flags;
    if (size == 0) {
        return playgta6::os::gdk::S_OK_HR;
    }
    if (dst == nullptr || src == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }
    std::memmove(dst, src, size);
    return playgta6::os::gdk::S_OK_HR;
}

Pg6HResult pg6_xbox_XMemGetWorkingSetStatistics(Pg6XMemWorkingSetStats* stats) {
    using namespace playgta6::os::scarlett;
    if (stats == nullptr) {
        playgta6::os::win32::SetLastError(playgta6::os::win32::ERROR_INVALID_PARAMETER);
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }

    auto& st = xmem_state();
    std::lock_guard<std::mutex> lock(st.mutex);
    const uint64_t used =
        st.heap_allocated_bytes + st.physical_pages_bytes + st.virtual_committed_bytes;

    stats->total_physical_bytes = kConsoleTotalRamBytes;
    stats->title_budget_bytes = kConsoleTitleBudgetBytes;
    stats->heap_allocated_bytes = st.heap_allocated_bytes;
    stats->physical_pages_bytes = st.physical_pages_bytes;
    stats->virtual_committed_bytes = st.virtual_committed_bytes;
    stats->virtual_reserved_bytes = st.virtual_reserved_bytes;
    stats->mapped_physical_bytes = st.mapped_physical_bytes;
    stats->peak_working_set_bytes = st.peak_working_set_bytes;
    stats->available_title_bytes =
        used >= kConsoleTitleBudgetBytes ? 0ULL : (kConsoleTitleBudgetBytes - used);
    stats->tools_mem_bytes = 0ULL;
    stats->live_heap_allocations = static_cast<uint32_t>(st.heap_allocs.size());
    stats->live_physical_pages = static_cast<uint32_t>(st.physical_pages.size());
    stats->live_virtual_mappings = static_cast<uint32_t>(st.mappings.size());
    stats->reserved = 0u;
    return playgta6::os::gdk::S_OK_HR;
}

int32_t pg6_xbox_XMemIsToolsMemAvailable(void) {
    return 0;
}

Pg6HResult pg6_xbox_DStorageGetFactory(const Pg6Guid* riid, void** ppv) {
    using namespace playgta6::os::scarlett;
    if (ppv == nullptr) {
        return playgta6::os::gdk::E_POINTER_HR;
    }
    *ppv = nullptr;
    if (riid != nullptr && *riid != kIidIUnknown && *riid != kIidIDStorageFactory) {
        return E_NOINTERFACE_HR;
    }
    auto* factory = new (std::nothrow) DStorageFactoryImpl();
    if (factory == nullptr) {
        return playgta6::os::gdk::E_OUTOFMEMORY_HR;
    }
    *ppv = static_cast<Pg6IDStorageFactory*>(factory);
    return playgta6::os::gdk::S_OK_HR;
}

Pg6HResult pg6_xbox_XGCreateTextureComputer(const Pg6XgTextureDesc* desc,
                                            Pg6IXGTextureAddressComputer** out_computer) {
    using namespace playgta6::os::scarlett;
    if (out_computer == nullptr) {
        return playgta6::os::gdk::E_POINTER_HR;
    }
    *out_computer = nullptr;
    if (desc == nullptr || desc->width == 0 || desc->depth_or_array_size == 0) {
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }
    if (desc->dimension != kXgDimensionTexture1D && desc->height == 0) {
        return playgta6::os::gdk::E_INVALIDARG_HR;
    }
    auto* comp = new (std::nothrow) XGTextureAddressComputerImpl(*desc);
    if (comp == nullptr) {
        return playgta6::os::gdk::E_OUTOFMEMORY_HR;
    }
    *out_computer = static_cast<Pg6IXGTextureAddressComputer*>(comp);
    return playgta6::os::gdk::S_OK_HR;
}

uint32_t pg6_xbox_XGSuggestSwizzleMode(const Pg6XgTextureDesc* desc) {
    using namespace playgta6::os::scarlett;
    if (desc == nullptr || desc->width == 0) {
        return kXgTileModeInvalid;
    }
    // WebGPU textures consume linear row-major upload layouts cleanly without GPU unswizzling.
    return kXgTileModeLinear;
}

Pg6HResult pg6_xbox_GameInputCreate(Pg6IGameInput** game_input) {
    using namespace playgta6::os::scarlett;
    if (game_input == nullptr) {
        return playgta6::os::gdk::E_POINTER_HR;
    }
    *game_input = nullptr;
    auto* impl = new (std::nothrow) GameInputImpl();
    if (impl == nullptr) {
        return playgta6::os::gdk::E_OUTOFMEMORY_HR;
    }
    *game_input = static_cast<Pg6IGameInput*>(impl);
    return playgta6::os::gdk::S_OK_HR;
}

size_t pg6_xbox_scarlett_register_all(void) {
    return playgta6::os::scarlett::register_all_scarlett_exports();
}

}  // extern "C"
