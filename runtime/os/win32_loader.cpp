#include "os/win32_loader.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace playgta6::os::win32 {

namespace {

// ---------------------------------------------------------------------------
// Heap implementation
// ---------------------------------------------------------------------------

constexpr uint64_t kHeapAlignment = 16;

struct HeapBlock {
    uint64_t size = 0;      // Requested size (HeapSize).
    uint64_t capacity = 0;  // 16-byte rounded footprint inside the segment.
};

struct Heap {
    std::mutex mutex;
    VirtualMemoryArena* arena = nullptr;
    PageProtect protect = PageProtect::ReadWrite;
    bool growable = true;
    bool destroyed = false;
    Handle handle = 0;  // Guest VA of the first segment; never released before HeapDestroy.

    std::map<uint64_t, uint64_t> segments;                 // base -> size
    std::map<uint64_t, uint64_t> free_by_addr;             // addr -> size (maximally coalesced)
    std::set<std::pair<uint64_t, uint64_t>> free_by_size;  // (size, addr) for best fit
    std::unordered_map<uint64_t, HeapBlock> live;          // addr -> block
};

uint64_t block_capacity(uint64_t bytes) {
    if (bytes == 0) {
        return kHeapAlignment;
    }
    return (bytes + (kHeapAlignment - 1u)) & ~(kHeapAlignment - 1u);
}

uint64_t arena_alloc(VirtualMemoryArena* arena, uint64_t size, PageProtect protect) {
    try {
        return arena->virtual_alloc(0, size, kMemReserve | kMemCommit, protect);
    } catch (const std::bad_alloc&) {
        return 0;
    }
}

void insert_free_locked(Heap& heap, uint64_t addr, uint64_t size) {
    heap.free_by_addr.emplace(addr, size);
    heap.free_by_size.emplace(size, addr);
}

void erase_free_locked(Heap& heap, std::map<uint64_t, uint64_t>::iterator it) {
    heap.free_by_size.erase({it->second, it->first});
    heap.free_by_addr.erase(it);
}

// Returns a free range to the heap, coalescing with free neighbours inside the same segment.
// Releases the whole segment back to the arena when it becomes empty (except the first one).
void add_free_locked(Heap& heap, uint64_t addr, uint64_t size) {
    const uint64_t end = addr + size;
    if (heap.segments.find(end) == heap.segments.end()) {
        auto next = heap.free_by_addr.find(end);
        if (next != heap.free_by_addr.end()) {
            size += next->second;
            erase_free_locked(heap, next);
        }
    }
    if (heap.segments.find(addr) == heap.segments.end()) {
        auto prev = heap.free_by_addr.lower_bound(addr);
        if (prev != heap.free_by_addr.begin()) {
            --prev;
            if (prev->first + prev->second == addr) {
                addr = prev->first;
                size += prev->second;
                erase_free_locked(heap, prev);
            }
        }
    }

    auto seg = heap.segments.find(addr);
    if (seg != heap.segments.end() && seg->second == size && addr != heap.handle) {
        heap.segments.erase(seg);
        heap.arena->release(addr);
        return;
    }
    insert_free_locked(heap, addr, size);
}

uint64_t take_free_locked(Heap& heap, uint64_t capacity) {
    auto it = heap.free_by_size.lower_bound({capacity, 0});
    if (it == heap.free_by_size.end()) {
        return 0;
    }
    const uint64_t size = it->first;
    const uint64_t addr = it->second;
    heap.free_by_size.erase(it);
    heap.free_by_addr.erase(addr);
    if (size > capacity) {
        // The neighbour after a maximal free range is live or a segment end: no coalescing needed.
        insert_free_locked(heap, addr + capacity, size - capacity);
    }
    return addr;
}

bool grow_locked(Heap& heap, uint64_t capacity) {
    uint64_t seg_size = 0;
    if (!align_up_checked(capacity, kAllocationGranularity, seg_size)) {
        return false;
    }
    seg_size = std::max(seg_size, kHeapSegmentSize);
    const uint64_t base = arena_alloc(heap.arena, seg_size, heap.protect);
    if (base == 0) {
        return false;
    }
    heap.segments.emplace(base, seg_size);
    insert_free_locked(heap, base, seg_size);
    return true;
}

void zero_range_locked(Heap& heap, uint64_t addr, uint64_t len) {
    if (len == 0) {
        return;
    }
    uint8_t* p = heap.arena->host_ptr(addr, static_cast<size_t>(len));
    if (p != nullptr) {
        std::memset(p, 0, static_cast<size_t>(len));
    }
}

uint64_t alloc_locked(Heap& heap, uint32_t flags, uint64_t bytes) {
    const uint64_t capacity = block_capacity(bytes);
    uint64_t addr = take_free_locked(heap, capacity);
    if (addr == 0 && heap.growable && grow_locked(heap, capacity)) {
        addr = take_free_locked(heap, capacity);
    }
    if (addr == 0) {
        return 0;
    }
    heap.live.emplace(addr, HeapBlock{bytes, capacity});
    if ((flags & HEAP_ZERO_MEMORY) != 0) {
        zero_range_locked(heap, addr, bytes);
    }
    return addr;
}

bool free_locked(Heap& heap, uint64_t addr) {
    auto it = heap.live.find(addr);
    if (it == heap.live.end()) {
        return false;
    }
    const uint64_t capacity = it->second.capacity;
    heap.live.erase(it);
    add_free_locked(heap, addr, capacity);
    return true;
}

void release_all_locked(Heap& heap) {
    for (const auto& [base, size] : heap.segments) {
        (void)size;
        heap.arena->release(base);
    }
    heap.segments.clear();
    heap.free_by_addr.clear();
    heap.free_by_size.clear();
    heap.live.clear();
    heap.destroyed = true;
}

std::shared_ptr<Heap> create_heap(uint32_t options, uint64_t initial_size, uint64_t maximum_size) {
    if (maximum_size != 0 && initial_size > maximum_size) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (initial_size > kMaxHeapRequest || maximum_size > kMaxHeapRequest) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }

    auto heap = std::make_shared<Heap>();
    heap->arena = get_vmem_arena();
    heap->protect = (options & HEAP_CREATE_ENABLE_EXECUTE) != 0 ? PageProtect::ExecuteReadWrite
                                                                 : PageProtect::ReadWrite;
    heap->growable = (maximum_size == 0);

    uint64_t seg_size = 0;
    if (heap->growable) {
        if (!align_up_checked(initial_size, kAllocationGranularity, seg_size)) {
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return nullptr;
        }
        seg_size = std::max(seg_size, kHeapSegmentSize);
    } else if (!align_up_checked(maximum_size, kPageSize, seg_size)) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }

    const uint64_t base = arena_alloc(heap->arena, seg_size, heap->protect);
    if (base == 0) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    heap->handle = base;
    heap->segments.emplace(base, seg_size);
    insert_free_locked(*heap, base, seg_size);
    return heap;
}

struct HeapRegistry {
    std::shared_mutex mutex;
    std::unordered_map<Handle, std::shared_ptr<Heap>> heaps;
    std::mutex process_heap_init;
    std::atomic<Handle> process_heap{0};
};

HeapRegistry& heap_registry() {
    static HeapRegistry registry;
    return registry;
}

std::shared_ptr<Heap> find_heap(Handle handle) {
    if (handle == kNullHandle) {
        return nullptr;
    }
    auto& reg = heap_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    auto it = reg.heaps.find(handle);
    return it != reg.heaps.end() ? it->second : nullptr;
}

// ---------------------------------------------------------------------------
// Module registry implementation
// ---------------------------------------------------------------------------

struct Module {
    std::string canonical;
    std::string display;
    std::string path;
    Handle handle = 0;
    bool pinned = true;
    bool guest = false;  // a recompiled guest DLL: exports are guest VAs
    uint64_t image_size = 0;  // guest modules / main image: SizeOfImage (0 = unknown)
    uint32_t load_count = 0;
    std::unordered_map<std::string, ProcAddress> by_name;
    std::unordered_map<uint32_t, ProcAddress> by_ordinal;
    std::unordered_map<std::string, std::string> forward_by_name;
    std::unordered_map<uint32_t, std::string> forward_by_ordinal;

    bool visible() const { return pinned || load_count > 0; }
};

struct ApiSetRule {
    std::string prefix;
    std::string host;
};

char ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

std::string ascii_lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) {
        c = ascii_lower(c);
    }
    return out;
}

// Basename with Windows default-extension rules applied, original case preserved.
std::string module_basename(std::string_view name) {
    const size_t sep = name.find_last_of("\\/");
    std::string base(sep == std::string_view::npos ? name : name.substr(sep + 1));
    if (base.empty()) {
        return {};
    }
    if (base.back() == '.') {
        base.pop_back();
    } else if (base.find('.') == std::string::npos) {
        base += ".dll";
    }
    return base;
}

bool is_api_set_name(std::string_view canonical) {
    return canonical.size() > 4 &&
           (canonical.substr(0, 4) == "api-" || canonical.substr(0, 4) == "ext-");
}

std::string utf8_from_utf16(const char16_t* s) {
    return utf16_to_utf8(s);
}

void append_utf16(std::u16string& out, uint32_t cp) {
    if (cp >= 0x10000u) {
        cp -= 0x10000u;
        out.push_back(static_cast<char16_t>(0xD800u + (cp >> 10)));
        out.push_back(static_cast<char16_t>(0xDC00u + (cp & 0x3FFu)));
    } else {
        out.push_back(static_cast<char16_t>(cp));
    }
}

std::u16string utf16_from_utf8(std::string_view s) {
    std::u16string out;
    size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<uint8_t>(s[i]);
        uint32_t cp = 0xFFFDu;
        size_t len = 1;
        if (b0 < 0x80u) {
            cp = b0;
        } else if ((b0 & 0xE0u) == 0xC0u) {
            len = 2;
            cp = b0 & 0x1Fu;
        } else if ((b0 & 0xF0u) == 0xE0u) {
            len = 3;
            cp = b0 & 0x0Fu;
        } else if ((b0 & 0xF8u) == 0xF0u) {
            len = 4;
            cp = b0 & 0x07u;
        } else {
            append_utf16(out, 0xFFFDu);
            ++i;
            continue;
        }
        bool valid = (i + len <= s.size());
        for (size_t k = 1; valid && k < len; ++k) {
            const auto b = static_cast<uint8_t>(s[i + k]);
            if ((b & 0xC0u) != 0x80u) {
                valid = false;
            } else {
                cp = (cp << 6) | (b & 0x3Fu);
            }
        }
        if (!valid || cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
            append_utf16(out, 0xFFFDu);
            ++i;
            continue;
        }
        append_utf16(out, cp);
        i += len;
    }
    return out;
}

struct ModuleRegistry {
    std::shared_mutex mutex;
    std::unordered_map<std::string, std::unique_ptr<Module>> by_name;
    std::unordered_map<Handle, Module*> by_handle;
    std::vector<ApiSetRule> api_set_rules;
    // Virtual modules whose name a guest module took over (still reachable by handle).
    std::vector<std::unique_ptr<Module>> retired;
    Module* main_module = nullptr;
    uint64_t next_virtual_slot = 0;

    ModuleRegistry();

    Module* find_locked(const std::string& canonical) const {
        auto it = by_name.find(canonical);
        if (it != by_name.end()) {
            return it->second.get();
        }
        if (!is_api_set_name(canonical)) {
            return nullptr;
        }
        const ApiSetRule* best = nullptr;
        for (const auto& rule : api_set_rules) {
            if (canonical.compare(0, rule.prefix.size(), rule.prefix) == 0 &&
                (best == nullptr || rule.prefix.size() > best->prefix.size())) {
                best = &rule;
            }
        }
        if (best == nullptr) {
            return nullptr;
        }
        auto host = by_name.find(best->host);
        return host != by_name.end() ? host->second.get() : nullptr;
    }

    Module* find_handle_locked(Handle handle) const {
        if (handle == kNullHandle) {
            return main_module;
        }
        auto it = by_handle.find(handle);
        return it != by_handle.end() ? it->second : nullptr;
    }

    Module* get_or_create_virtual_locked(std::string_view name, bool pinned) {
        const std::string display = module_basename(name);
        if (display.empty()) {
            return nullptr;
        }
        const std::string canonical = ascii_lower(display);
        auto it = by_name.find(canonical);
        if (it != by_name.end()) {
            it->second->pinned = it->second->pinned || pinned;
            return it->second.get();
        }

        Handle handle = 0;
        do {
            handle = kVirtualModuleBase + kVirtualModuleStride * next_virtual_slot++;
        } while (by_handle.find(handle) != by_handle.end());

        auto module = std::make_unique<Module>();
        module->canonical = canonical;
        module->display = display;
        module->path = "C:\\Windows\\System32\\" + display;
        module->handle = handle;
        module->pinned = pinned;
        Module* raw = module.get();
        by_handle.emplace(handle, raw);
        by_name.emplace(canonical, std::move(module));
        return raw;
    }

    bool add_export_locked(std::string_view module,
                           std::string_view name,
                           uint32_t ordinal,
                           ProcAddress address,
                           bool guest_export = false) {
        if (address == 0 || ordinal > 0xFFFFu || (name.empty() && ordinal == 0)) {
            return false;
        }
        Module* m = nullptr;
        auto existing = by_name.find(canonical_module_name(module));
        if (existing != by_name.end()) {
            m = existing->second.get();
        } else if (!guest_export) {
            m = get_or_create_virtual_locked(module, true);
        }
        // Guest exports go to guest modules and the main image, host shims everywhere else.
        if (m == nullptr || (guest_export ? !(m->guest || m == main_module) : m->guest)) {
            return false;
        }
        if (!name.empty()) {
            m->by_name[std::string(name)] = address;
        }
        if (ordinal != 0) {
            m->by_ordinal[ordinal] = address;
        }
        return true;
    }

    bool add_forwarder_locked(std::string_view module,
                              std::string_view name,
                              uint32_t ordinal,
                              std::string_view target) {
        if (target.empty() || ordinal > 0xFFFFu || (name.empty() && ordinal == 0)) {
            return false;
        }
        auto existing = by_name.find(canonical_module_name(module));
        if (existing == by_name.end()) {
            return false;
        }
        Module* m = existing->second.get();
        if (!name.empty()) {
            m->forward_by_name[std::string(name)] = std::string(target);
        }
        if (ordinal != 0) {
            m->forward_by_ordinal[ordinal] = std::string(target);
        }
        return true;
    }

    // Export lookup by name (ordinal == 0) or ordinal, following forwarders ("DLL.Name",
    // "DLL.#ordinal"; the module part may be an API set) for at most kMaxForwards hops.
    ProcAddress lookup_locked(const Module* m, std::string name, uint32_t ordinal) const {
        static constexpr int kMaxForwards = 16;
        for (int hop = 0; m != nullptr && hop <= kMaxForwards; ++hop) {
            const std::string* forward = nullptr;
            if (ordinal == 0) {
                if (auto it = m->by_name.find(name); it != m->by_name.end()) {
                    return it->second;
                }
                if (auto it = m->forward_by_name.find(name); it != m->forward_by_name.end()) {
                    forward = &it->second;
                }
            } else {
                if (auto it = m->by_ordinal.find(ordinal); it != m->by_ordinal.end()) {
                    return it->second;
                }
                if (auto it = m->forward_by_ordinal.find(ordinal);
                    it != m->forward_by_ordinal.end()) {
                    forward = &it->second;
                }
            }
            if (forward == nullptr) {
                return 0;
            }
            const size_t dot = forward->rfind('.');
            if (dot == std::string::npos || dot == 0 || dot + 1 == forward->size()) {
                return 0;
            }
            const std::string symbol = forward->substr(dot + 1);
            m = find_locked(canonical_module_name(std::string_view(*forward).substr(0, dot)));
            if (symbol[0] == '#') {
                const std::string digits = symbol.substr(1);
                if (digits.empty() || digits.size() > 5 ||
                    digits.find_first_not_of("0123456789") != std::string::npos) {
                    return 0;
                }
                ordinal = static_cast<uint32_t>(std::stoul(digits));
                name.clear();
                if (ordinal == 0 || ordinal > 0xFFFFu) {
                    return 0;
                }
            } else {
                name = symbol;
                ordinal = 0;
            }
        }
        return 0;
    }

    bool add_api_set_rule_locked(std::string_view contract_prefix, std::string_view host_module) {
        std::string prefix = ascii_lower(contract_prefix);
        std::string host = canonical_module_name(host_module);
        if (!is_api_set_name(prefix) || host.empty()) {
            return false;
        }
        for (auto& rule : api_set_rules) {
            if (rule.prefix == prefix) {
                rule.host = std::move(host);
                return true;
            }
        }
        api_set_rules.push_back(ApiSetRule{std::move(prefix), std::move(host)});
        return true;
    }
};

ModuleRegistry& module_registry() {
    static ModuleRegistry registry;
    return registry;
}

ModuleRegistry::ModuleRegistry() {
    // The runtime implements kernelbase-hosted contracts inside its KERNEL32 stand-in.
    add_api_set_rule_locked("api-ms-win-core-", "kernel32.dll");
    add_api_set_rule_locked("api-ms-win-core-com-", "combase.dll");
    add_api_set_rule_locked("api-ms-win-core-winrt-", "combase.dll");
    add_api_set_rule_locked("api-ms-win-crt-", "ucrtbase.dll");

    const ExportEntry kernel32_exports[] = {
        export_entry("GetProcessHeap", &pg6_win32_get_process_heap),
        export_entry("HeapCreate", &pg6_win32_heap_create),
        export_entry("HeapDestroy", &pg6_win32_heap_destroy),
        export_entry("HeapAlloc", &pg6_win32_heap_alloc),
        export_entry("HeapReAlloc", &pg6_win32_heap_realloc),
        export_entry("HeapFree", &pg6_win32_heap_free),
        export_entry("HeapSize", &pg6_win32_heap_size),
        export_entry("HeapValidate", &pg6_win32_heap_validate),
        export_entry("GetModuleHandleA", &pg6_win32_get_module_handle_a),
        export_entry("GetModuleHandleW", &pg6_win32_get_module_handle_w),
        export_entry("LoadLibraryA", &pg6_win32_load_library_a),
        export_entry("LoadLibraryW", &pg6_win32_load_library_w),
        export_entry("LoadLibraryExA", &pg6_win32_load_library_ex_a),
        export_entry("LoadLibraryExW", &pg6_win32_load_library_ex_w),
        export_entry("FreeLibrary", &pg6_win32_free_library),
        export_entry("GetModuleFileNameA", &pg6_win32_get_module_file_name_a),
        export_entry("GetModuleFileNameW", &pg6_win32_get_module_file_name_w),
        export_entry("GetProcAddress", &pg6_win32_get_proc_address),
    };
    get_or_create_virtual_locked("KERNEL32.dll", true);
    for (const auto& entry : kernel32_exports) {
        add_export_locked("KERNEL32.dll", entry.name, entry.ordinal, entry.address);
    }
}

Handle get_module_handle_impl(const char* utf8_name, bool is_null) {
    auto& reg = module_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    if (is_null) {
        if (reg.main_module == nullptr) {
            SetLastError(ERROR_MOD_NOT_FOUND);
            return kNullHandle;
        }
        return reg.main_module->handle;
    }
    const Module* m = reg.find_locked(canonical_module_name(utf8_name));
    if (m == nullptr || !m->visible()) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return kNullHandle;
    }
    return m->handle;
}

Handle load_library_impl(std::string_view utf8_name) {
    const std::string canonical = canonical_module_name(utf8_name);
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    Module* m = canonical.empty() ? nullptr : reg.find_locked(canonical);
    if (m == nullptr) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return kNullHandle;
    }
    if (m->load_count != UINT32_MAX) {
        ++m->load_count;
    }
    return m->handle;
}

template <typename CharT>
uint32_t copy_module_path(const std::basic_string<CharT>& path, CharT* out, uint32_t size) {
    if (size == 0) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    if (out == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }
    if (path.size() < size) {
        std::copy(path.begin(), path.end(), out);
        out[path.size()] = CharT{0};
        return static_cast<uint32_t>(path.size());
    }
    std::copy(path.begin(), path.begin() + (size - 1u), out);
    out[size - 1u] = CharT{0};
    SetLastError(ERROR_INSUFFICIENT_BUFFER);
    return size;
}

bool module_path(Handle module, std::string& out) {
    auto& reg = module_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    const Module* m = reg.find_handle_locked(module);
    if (m == nullptr || !m->visible()) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return false;
    }
    out = m->path;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Heap shims
// ---------------------------------------------------------------------------

Handle GetProcessHeap() {
    auto& reg = heap_registry();
    Handle handle = reg.process_heap.load(std::memory_order_acquire);
    if (handle != kNullHandle) {
        return handle;
    }
    std::lock_guard<std::mutex> init_lock(reg.process_heap_init);
    handle = reg.process_heap.load(std::memory_order_acquire);
    if (handle != kNullHandle) {
        return handle;
    }
    auto heap = create_heap(HEAP_GROWABLE, 0, 0);
    if (!heap) {
        return kNullHandle;
    }
    handle = heap->handle;
    {
        std::unique_lock<std::shared_mutex> lock(reg.mutex);
        reg.heaps.emplace(handle, std::move(heap));
    }
    reg.process_heap.store(handle, std::memory_order_release);
    return handle;
}

Handle HeapCreate(uint32_t options, uint64_t initial_size, uint64_t maximum_size) {
    auto heap = create_heap(options, initial_size, maximum_size);
    if (!heap) {
        return kNullHandle;
    }
    const Handle handle = heap->handle;
    auto& reg = heap_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    reg.heaps.emplace(handle, std::move(heap));
    return handle;
}

bool HeapDestroy(Handle heap_handle) {
    auto& reg = heap_registry();
    if (heap_handle == kNullHandle ||
        heap_handle == reg.process_heap.load(std::memory_order_acquire)) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    std::shared_ptr<Heap> heap;
    {
        std::unique_lock<std::shared_mutex> lock(reg.mutex);
        auto it = reg.heaps.find(heap_handle);
        if (it == reg.heaps.end()) {
            SetLastError(ERROR_INVALID_HANDLE);
            return false;
        }
        heap = std::move(it->second);
        reg.heaps.erase(it);
    }
    std::lock_guard<std::mutex> lock(heap->mutex);
    release_all_locked(*heap);
    return true;
}

uint64_t HeapAlloc(Handle heap_handle, uint32_t flags, uint64_t bytes) {
    auto heap = find_heap(heap_handle);
    if (!heap) {
        SetLastError(ERROR_INVALID_HANDLE);
        return 0;
    }
    if (bytes > kMaxHeapRequest) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    }
    std::lock_guard<std::mutex> lock(heap->mutex);
    if (heap->destroyed) {
        SetLastError(ERROR_INVALID_HANDLE);
        return 0;
    }
    const uint64_t addr = alloc_locked(*heap, flags, bytes);
    if (addr == 0) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    }
    return addr;
}

uint64_t HeapReAlloc(Handle heap_handle, uint32_t flags, uint64_t mem, uint64_t bytes) {
    auto heap = find_heap(heap_handle);
    if (!heap) {
        SetLastError(ERROR_INVALID_HANDLE);
        return 0;
    }
    if (bytes > kMaxHeapRequest) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    }
    std::lock_guard<std::mutex> lock(heap->mutex);
    if (heap->destroyed) {
        SetLastError(ERROR_INVALID_HANDLE);
        return 0;
    }
    auto it = heap->live.find(mem);
    if (mem == 0 || it == heap->live.end()) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }

    const HeapBlock old = it->second;
    const uint64_t new_capacity = block_capacity(bytes);
    const bool zero = (flags & HEAP_ZERO_MEMORY) != 0;

    auto finish_in_place = [&](uint64_t capacity) {
        it->second = HeapBlock{bytes, capacity};
        if (zero && bytes > old.size) {
            zero_range_locked(*heap, mem + old.size, bytes - old.size);
        }
        return mem;
    };

    if (new_capacity <= old.capacity) {
        if (new_capacity < old.capacity) {
            add_free_locked(*heap, mem + new_capacity, old.capacity - new_capacity);
        }
        return finish_in_place(new_capacity);
    }

    // Try to grow into the free range directly after the block (same segment only).
    const uint64_t next = mem + old.capacity;
    if (heap->segments.find(next) == heap->segments.end()) {
        auto nf = heap->free_by_addr.find(next);
        if (nf != heap->free_by_addr.end() && old.capacity + nf->second >= new_capacity) {
            const uint64_t remainder = old.capacity + nf->second - new_capacity;
            erase_free_locked(*heap, nf);
            if (remainder != 0) {
                insert_free_locked(*heap, mem + new_capacity, remainder);
            }
            return finish_in_place(new_capacity);
        }
    }

    if ((flags & HEAP_REALLOC_IN_PLACE_ONLY) != 0) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    }

    const uint64_t moved = alloc_locked(*heap, 0, bytes);  // May rehash `live`; `it` is stale.
    if (moved == 0) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return 0;
    }
    const uint64_t keep = std::min(old.size, bytes);
    if (keep != 0) {
        const uint8_t* src = heap->arena->host_ptr_const(mem, static_cast<size_t>(keep));
        uint8_t* dst = heap->arena->host_ptr(moved, static_cast<size_t>(keep));
        if (src != nullptr && dst != nullptr) {
            std::memcpy(dst, src, static_cast<size_t>(keep));
        }
    }
    if (zero && bytes > old.size) {
        zero_range_locked(*heap, moved + old.size, bytes - old.size);
    }
    free_locked(*heap, mem);
    return moved;
}

bool HeapFree(Handle heap_handle, uint32_t flags, uint64_t mem) {
    (void)flags;
    auto heap = find_heap(heap_handle);
    if (!heap) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    if (mem == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lock(heap->mutex);
    if (heap->destroyed) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    if (!free_locked(*heap, mem)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    return true;
}

uint64_t HeapSize(Handle heap_handle, uint32_t flags, uint64_t mem) {
    (void)flags;
    auto heap = find_heap(heap_handle);
    if (!heap) {
        SetLastError(ERROR_INVALID_HANDLE);
        return kHeapSizeError;
    }
    std::lock_guard<std::mutex> lock(heap->mutex);
    auto it = heap->live.find(mem);
    if (heap->destroyed || it == heap->live.end()) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return kHeapSizeError;
    }
    return it->second.size;
}

bool HeapValidate(Handle heap_handle, uint32_t flags, uint64_t mem) {
    (void)flags;
    auto heap = find_heap(heap_handle);
    if (!heap) {
        return false;
    }
    std::lock_guard<std::mutex> lock(heap->mutex);
    if (heap->destroyed) {
        return false;
    }
    if (mem != 0) {
        return heap->live.find(mem) != heap->live.end();
    }
    // Whole-heap accounting check: live + free footprints exactly tile the segments.
    if (heap->free_by_addr.size() != heap->free_by_size.size()) {
        return false;
    }
    uint64_t segment_bytes = 0;
    for (const auto& [base, size] : heap->segments) {
        (void)base;
        segment_bytes += size;
    }
    uint64_t used_bytes = 0;
    for (const auto& [addr, block] : heap->live) {
        (void)addr;
        used_bytes += block.capacity;
    }
    for (const auto& [addr, size] : heap->free_by_addr) {
        (void)addr;
        used_bytes += size;
    }
    return used_bytes == segment_bytes;
}

// ---------------------------------------------------------------------------
// Module shims
// ---------------------------------------------------------------------------

Handle GetModuleHandleA(const char* module_name) {
    return get_module_handle_impl(module_name, module_name == nullptr);
}

Handle GetModuleHandleW(const char16_t* module_name) {
    if (module_name == nullptr) {
        return get_module_handle_impl(nullptr, true);
    }
    const std::string utf8 = utf8_from_utf16(module_name);
    return get_module_handle_impl(utf8.c_str(), false);
}

Handle LoadLibraryA(const char* file_name) {
    return LoadLibraryExA(file_name, kNullHandle, 0);
}

Handle LoadLibraryW(const char16_t* file_name) {
    return LoadLibraryExW(file_name, kNullHandle, 0);
}

Handle LoadLibraryExA(const char* file_name, Handle file, uint32_t flags) {
    (void)flags;
    if (file_name == nullptr || file != kNullHandle) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return kNullHandle;
    }
    return load_library_impl(file_name);
}

Handle LoadLibraryExW(const char16_t* file_name, Handle file, uint32_t flags) {
    (void)flags;
    if (file_name == nullptr || file != kNullHandle) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return kNullHandle;
    }
    return load_library_impl(utf8_from_utf16(file_name));
}

bool FreeLibrary(Handle module) {
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    Module* m = module == kNullHandle ? nullptr : reg.find_handle_locked(module);
    if (m == nullptr || !m->visible()) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return false;
    }
    if (m->load_count > 0) {
        --m->load_count;
    }
    return true;
}

uint32_t GetModuleFileNameA(Handle module, char* file_name, uint32_t size) {
    std::string path;
    if (!module_path(module, path)) {
        return 0;
    }
    return copy_module_path(path, file_name, size);
}

uint32_t GetModuleFileNameW(Handle module, char16_t* file_name, uint32_t size) {
    std::string path;
    if (!module_path(module, path)) {
        return 0;
    }
    return copy_module_path(utf16_from_utf8(path), file_name, size);
}

ProcAddress GetProcAddress(Handle module, const char* proc_name) {
    const auto raw = reinterpret_cast<uintptr_t>(proc_name);
    if ((raw >> 16) == 0) {
        return GetProcAddressByOrdinal(module, static_cast<uint32_t>(raw));
    }
    return GetProcAddressByName(module, proc_name);
}

ProcAddress GetProcAddressByName(Handle module, std::string_view name) {
    auto& reg = module_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    const Module* m = reg.find_handle_locked(module);
    if (m == nullptr || !m->visible()) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return 0;
    }
    const ProcAddress address = name.empty() ? 0 : reg.lookup_locked(m, std::string(name), 0);
    if (address == 0) {
        SetLastError(ERROR_PROC_NOT_FOUND);
    }
    return address;
}

ProcAddress GetProcAddressByOrdinal(Handle module, uint32_t ordinal) {
    auto& reg = module_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    const Module* m = reg.find_handle_locked(module);
    if (m == nullptr || !m->visible()) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return 0;
    }
    const ProcAddress address = ordinal == 0 ? 0 : reg.lookup_locked(m, {}, ordinal);
    if (address == 0) {
        SetLastError(ERROR_PROC_NOT_FOUND);
    }
    return address;
}

// ---------------------------------------------------------------------------
// Registration API
// ---------------------------------------------------------------------------

std::string canonical_module_name(std::string_view name) {
    return ascii_lower(module_basename(name));
}

Handle register_main_module(std::string_view name,
                            std::string_view path,
                            uint64_t image_base,
                            uint64_t size_of_image) {
    const std::string display = module_basename(name);
    if (display.empty() || image_base == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return kNullHandle;
    }
    const std::string canonical = ascii_lower(display);
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    if (reg.main_module != nullptr) {
        if (reg.main_module->handle == image_base && reg.main_module->canonical == canonical) {
            reg.main_module->path = path.empty() ? display : std::string(path);
            if (size_of_image != 0) {
                reg.main_module->image_size = size_of_image;
            }
            return image_base;
        }
        SetLastError(ERROR_ALREADY_EXISTS);
        return kNullHandle;
    }
    if (reg.by_handle.find(image_base) != reg.by_handle.end() ||
        reg.by_name.find(canonical) != reg.by_name.end()) {
        SetLastError(ERROR_ALREADY_EXISTS);
        return kNullHandle;
    }
    auto module = std::make_unique<Module>();
    module->canonical = canonical;
    module->display = display;
    module->path = path.empty() ? display : std::string(path);
    module->handle = image_base;
    module->pinned = true;
    module->image_size = size_of_image;
    reg.main_module = module.get();
    reg.by_handle.emplace(image_base, module.get());
    reg.by_name.emplace(canonical, std::move(module));
    return image_base;
}

Handle register_guest_module(std::string_view name,
                             std::string_view path,
                             uint64_t image_base,
                             uint64_t size_of_image) {
    const std::string display = module_basename(name);
    if (display.empty() || image_base == 0 || size_of_image == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return kNullHandle;
    }
    const std::string canonical = ascii_lower(display);
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    auto by_handle = reg.by_handle.find(image_base);
    auto by_name = reg.by_name.find(canonical);
    if (by_handle != reg.by_handle.end() || by_name != reg.by_name.end()) {
        const Module* existing =
            by_handle != reg.by_handle.end() ? by_handle->second : by_name->second.get();
        if (existing->guest && existing->handle == image_base && existing->canonical == canonical) {
            return image_base;
        }
        // Only a virtual module's name can be taken over; its handle stays registered.
        if (by_handle != reg.by_handle.end() || by_name->second->guest ||
            by_name->second.get() == reg.main_module ||
            by_name->second->handle < kVirtualModuleBase) {
            SetLastError(ERROR_ALREADY_EXISTS);
            return kNullHandle;
        }
        reg.retired.push_back(std::move(by_name->second));
        reg.by_name.erase(by_name);
    }
    auto module = std::make_unique<Module>();
    module->canonical = canonical;
    module->display = display;
    module->path = path.empty() ? display : std::string(path);
    module->handle = image_base;
    module->pinned = true;
    module->guest = true;
    module->image_size = size_of_image;
    reg.by_handle.emplace(image_base, module.get());
    reg.by_name.emplace(canonical, std::move(module));
    return image_base;
}

Handle find_module_by_address(uint64_t address) {
    auto& reg = module_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    for (const auto& [handle, module] : reg.by_handle) {
        const uint64_t size = module->image_size != 0
                                  ? module->image_size
                                  : (handle >= kVirtualModuleBase ? kVirtualModuleStride : 0);
        if (size != 0 && address >= handle && address - handle < size && module->visible()) {
            return handle;
        }
    }
    SetLastError(ERROR_MOD_NOT_FOUND);
    return kNullHandle;
}

bool module_add_reference(Handle module, bool pin, bool add_reference) {
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    Module* m = reg.find_handle_locked(module);
    if (m == nullptr || !m->visible()) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return false;
    }
    if (pin) {
        m->pinned = true;
    } else if (add_reference && !m->pinned && m->load_count != UINT32_MAX) {
        ++m->load_count;
    }
    return true;
}

bool is_guest_module(Handle module) {
    auto& reg = module_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    const Module* m = module == kNullHandle ? nullptr : reg.find_handle_locked(module);
    return m != nullptr && (m->guest || m == reg.main_module);
}

bool register_guest_export(std::string_view module,
                           std::string_view name,
                           uint32_t ordinal,
                           uint64_t guest_va) {
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    if (!reg.add_export_locked(module, name, ordinal, guest_va, true)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    return true;
}

bool register_export_forwarder(std::string_view module,
                               std::string_view name,
                               uint32_t ordinal,
                               std::string_view target) {
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    if (!reg.add_forwarder_locked(module, name, ordinal, target)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    return true;
}

Handle register_virtual_module(std::string_view name, bool pinned) {
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    const Module* m = reg.get_or_create_virtual_locked(name, pinned);
    if (m == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return kNullHandle;
    }
    return m->handle;
}

bool register_export(std::string_view module,
                     std::string_view name,
                     uint32_t ordinal,
                     ProcAddress address) {
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    if (!reg.add_export_locked(module, name, ordinal, address)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    return true;
}

size_t register_exports(std::string_view module, const ExportEntry* entries, size_t count) {
    if (entries == nullptr) {
        return 0;
    }
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    size_t registered = 0;
    for (size_t i = 0; i < count; ++i) {
        const std::string_view name =
            entries[i].name != nullptr ? std::string_view(entries[i].name) : std::string_view{};
        if (reg.add_export_locked(module, name, entries[i].ordinal, entries[i].address)) {
            ++registered;
        }
    }
    if (registered != count) {
        SetLastError(ERROR_INVALID_PARAMETER);
    }
    return registered;
}

size_t register_exports(std::string_view module, std::initializer_list<ExportEntry> entries) {
    return register_exports(module, entries.begin(), entries.size());
}

bool register_api_set_rule(std::string_view contract_prefix, std::string_view host_module) {
    auto& reg = module_registry();
    std::unique_lock<std::shared_mutex> lock(reg.mutex);
    if (!reg.add_api_set_rule_locked(contract_prefix, host_module)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    return true;
}

ProcAddress resolve_import(std::string_view module, std::string_view name) {
    auto& reg = module_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    const Module* m = reg.find_locked(canonical_module_name(module));
    if (m == nullptr) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return 0;
    }
    const ProcAddress address = name.empty() ? 0 : reg.lookup_locked(m, std::string(name), 0);
    if (address == 0) {
        SetLastError(ERROR_PROC_NOT_FOUND);
    }
    return address;
}

ProcAddress resolve_import_ordinal(std::string_view module, uint32_t ordinal) {
    auto& reg = module_registry();
    std::shared_lock<std::shared_mutex> lock(reg.mutex);
    const Module* m = reg.find_locked(canonical_module_name(module));
    if (m == nullptr) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return 0;
    }
    const ProcAddress address = ordinal == 0 ? 0 : reg.lookup_locked(m, {}, ordinal);
    if (address == 0) {
        SetLastError(ERROR_PROC_NOT_FOUND);
    }
    return address;
}

}  // namespace playgta6::os::win32

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints
// ---------------------------------------------------------------------------

namespace {

std::string_view sv_or_empty(const char* s) {
    return s != nullptr ? std::string_view(s) : std::string_view{};
}

}  // namespace

extern "C" {

Pg6Handle pg6_win32_get_process_heap(void) {
    return playgta6::os::win32::GetProcessHeap();
}

Pg6Handle pg6_win32_heap_create(uint32_t options, uint64_t initial_size, uint64_t maximum_size) {
    return playgta6::os::win32::HeapCreate(options, initial_size, maximum_size);
}

int32_t pg6_win32_heap_destroy(Pg6Handle heap) {
    return playgta6::os::win32::HeapDestroy(heap) ? 1 : 0;
}

uint64_t pg6_win32_heap_alloc(Pg6Handle heap, uint32_t flags, uint64_t bytes) {
    return playgta6::os::win32::HeapAlloc(heap, flags, bytes);
}

uint64_t pg6_win32_heap_realloc(Pg6Handle heap, uint32_t flags, uint64_t mem, uint64_t bytes) {
    return playgta6::os::win32::HeapReAlloc(heap, flags, mem, bytes);
}

int32_t pg6_win32_heap_free(Pg6Handle heap, uint32_t flags, uint64_t mem) {
    return playgta6::os::win32::HeapFree(heap, flags, mem) ? 1 : 0;
}

uint64_t pg6_win32_heap_size(Pg6Handle heap, uint32_t flags, uint64_t mem) {
    return playgta6::os::win32::HeapSize(heap, flags, mem);
}

int32_t pg6_win32_heap_validate(Pg6Handle heap, uint32_t flags, uint64_t mem) {
    return playgta6::os::win32::HeapValidate(heap, flags, mem) ? 1 : 0;
}

Pg6Handle pg6_win32_get_module_handle_a(const char* module_name) {
    return playgta6::os::win32::GetModuleHandleA(module_name);
}

Pg6Handle pg6_win32_get_module_handle_w(const char16_t* module_name) {
    return playgta6::os::win32::GetModuleHandleW(module_name);
}

Pg6Handle pg6_win32_load_library_a(const char* file_name) {
    return playgta6::os::win32::LoadLibraryA(file_name);
}

Pg6Handle pg6_win32_load_library_w(const char16_t* file_name) {
    return playgta6::os::win32::LoadLibraryW(file_name);
}

Pg6Handle pg6_win32_load_library_ex_a(const char* file_name, Pg6Handle file, uint32_t flags) {
    return playgta6::os::win32::LoadLibraryExA(file_name, file, flags);
}

Pg6Handle pg6_win32_load_library_ex_w(const char16_t* file_name, Pg6Handle file, uint32_t flags) {
    return playgta6::os::win32::LoadLibraryExW(file_name, file, flags);
}

int32_t pg6_win32_free_library(Pg6Handle module) {
    return playgta6::os::win32::FreeLibrary(module) ? 1 : 0;
}

uint32_t pg6_win32_get_module_file_name_a(Pg6Handle module, char* file_name, uint32_t size) {
    return playgta6::os::win32::GetModuleFileNameA(module, file_name, size);
}

uint32_t pg6_win32_get_module_file_name_w(Pg6Handle module, char16_t* file_name, uint32_t size) {
    return playgta6::os::win32::GetModuleFileNameW(module, file_name, size);
}

Pg6ProcAddress pg6_win32_get_proc_address(Pg6Handle module, const char* proc_name) {
    return playgta6::os::win32::GetProcAddress(module, proc_name);
}

Pg6ProcAddress pg6_loader_get_proc_address_by_name(Pg6Handle module, const char* name) {
    return playgta6::os::win32::GetProcAddressByName(module, sv_or_empty(name));
}

Pg6ProcAddress pg6_loader_get_proc_address_by_ordinal(Pg6Handle module, uint32_t ordinal) {
    return playgta6::os::win32::GetProcAddressByOrdinal(module, ordinal);
}

Pg6Handle pg6_loader_register_main_module(const char* name, const char* path, uint64_t image_base) {
    return playgta6::os::win32::register_main_module(
        sv_or_empty(name), sv_or_empty(path), image_base);
}

Pg6Handle pg6_loader_register_virtual_module(const char* name, int32_t pinned) {
    return playgta6::os::win32::register_virtual_module(sv_or_empty(name), pinned != 0);
}

Pg6Handle pg6_loader_register_guest_module(const char* name,
                                           const char* path,
                                           uint64_t image_base,
                                           uint64_t size_of_image) {
    return playgta6::os::win32::register_guest_module(sv_or_empty(name), sv_or_empty(path),
                                                      image_base, size_of_image);
}

int32_t pg6_loader_register_guest_export(const char* module,
                                         const char* name,
                                         uint32_t ordinal,
                                         uint64_t guest_va) {
    return playgta6::os::win32::register_guest_export(sv_or_empty(module), sv_or_empty(name),
                                                      ordinal, guest_va)
               ? 1
               : 0;
}

int32_t pg6_loader_register_export_forwarder(const char* module,
                                             const char* name,
                                             uint32_t ordinal,
                                             const char* target) {
    return playgta6::os::win32::register_export_forwarder(sv_or_empty(module), sv_or_empty(name),
                                                          ordinal, sv_or_empty(target))
               ? 1
               : 0;
}

int32_t pg6_loader_register_export(const char* module,
                                   const char* name,
                                   uint32_t ordinal,
                                   Pg6ProcAddress address) {
    return playgta6::os::win32::register_export(
               sv_or_empty(module), sv_or_empty(name), ordinal, address)
               ? 1
               : 0;
}

size_t pg6_loader_register_exports(const char* module, const Pg6ExportEntry* entries, size_t count) {
    return playgta6::os::win32::register_exports(sv_or_empty(module), entries, count);
}

int32_t pg6_loader_register_api_set_rule(const char* contract_prefix, const char* host_module) {
    return playgta6::os::win32::register_api_set_rule(
               sv_or_empty(contract_prefix), sv_or_empty(host_module))
               ? 1
               : 0;
}

Pg6ProcAddress pg6_loader_resolve_import(const char* module, const char* name) {
    return playgta6::os::win32::resolve_import(sv_or_empty(module), sv_or_empty(name));
}

Pg6ProcAddress pg6_loader_resolve_import_ordinal(const char* module, uint32_t ordinal) {
    return playgta6::os::win32::resolve_import_ordinal(sv_or_empty(module), ordinal);
}

}  // extern "C"
