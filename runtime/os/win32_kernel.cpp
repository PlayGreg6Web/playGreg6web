#include "os/win32_kernel.h"

#include "os/xmem_windows.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>

namespace playgta6::os::win32 {

namespace {

// ---------------------------------------------------------------------------
// Thread-local last error & monotonic clock anchor
// ---------------------------------------------------------------------------

thread_local uint32_t g_last_error = ERROR_SUCCESS;
// When a guest thread runs on this host thread, its TEB LastErrorValue (see SetThreadLastErrorSlot).
thread_local uint32_t* g_last_error_slot = nullptr;

const std::chrono::steady_clock::time_point& process_start_time() {
    static const auto kStart = std::chrono::steady_clock::now();
    return kStart;
}

// ---------------------------------------------------------------------------
// Injectable backend pointers
// ---------------------------------------------------------------------------

std::atomic<VirtualMemoryArena*> g_vmem_arena{nullptr};
std::atomic<TlsTable*> g_tls_table{nullptr};
std::atomic<HttpfsMailboxClient*> g_httpfs_client{nullptr};

// ---------------------------------------------------------------------------
// In-memory /userdata overlay
// ---------------------------------------------------------------------------

// FILETIME ticks (100 ns since 1601) used for the overlay's per-file timestamps.
uint64_t now_filetime_ticks() {
    constexpr uint64_t kEpochDiff100Ns = 116444736000000000ULL;
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const int64_t unix_100ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() / 100;
    return kEpochDiff100Ns + static_cast<uint64_t>(unix_100ns > 0 ? unix_100ns : 0);
}

struct UserdataMeta {
    uint64_t created = 0;
    uint64_t accessed = 0;
    uint64_t written = 0;
    uint32_t attributes = 0;  // settable attribute bits only (READONLY, HIDDEN, ...)
};

struct UserdataOverlay {
    std::mutex mutex;
    std::unordered_map<std::string, std::vector<uint8_t>> files;
    // O15: explicit directories (a directory also exists implicitly while a file is below it)
    // and per-entry timestamps/attributes, keyed by the same normalized "/userdata/..." keys.
    std::set<std::string> dirs;
    std::unordered_map<std::string, UserdataMeta> meta;
};

UserdataOverlay& userdata_overlay() {
    static UserdataOverlay overlay;
    return overlay;
}

// ---------------------------------------------------------------------------
// Unified kernel handle table (dispatcher objects from os/win32_sync, and Files)
// ---------------------------------------------------------------------------

enum class KernelObjectKind : uint8_t {
    Dispatcher,
    File,
    Opaque,  // O15: find handles, registry keys, ... owned by os/win32_compat
};

struct KernelObject {
    explicit KernelObject(KernelObjectKind k) : kind(k) {}
    virtual ~KernelObject() = default;
    KernelObjectKind kind;
};

struct DispatcherEntry final : KernelObject {
    explicit DispatcherEntry(std::shared_ptr<DispatcherObject> obj)
        : KernelObject(KernelObjectKind::Dispatcher), object(std::move(obj)) {}

    std::shared_ptr<DispatcherObject> object;
};

struct OpaqueEntry final : KernelObject {
    OpaqueEntry(uint32_t t, std::shared_ptr<void> p)
        : KernelObject(KernelObjectKind::Opaque), tag(t), payload(std::move(p)) {}

    uint32_t tag;
    std::shared_ptr<void> payload;
};

struct FileObject final : KernelObject {
    FileObject() : KernelObject(KernelObjectKind::File) {}

    std::mutex mutex;
    bool is_userdata = false;
    uint32_t desired_access = 0;
    uint64_t cursor = 0;
    std::string userdata_key;
    HttpfsMailboxClient* httpfs_client = nullptr;
    int32_t httpfs_handle = -1;
    bool has_cached_size = false;
    uint64_t cached_size = 0;
};

struct KernelHandleTable {
    std::mutex mutex;
    Handle next_handle = 1;
    std::unordered_map<Handle, std::shared_ptr<KernelObject>> objects;

    Handle insert(std::shared_ptr<KernelObject> obj) {
        std::lock_guard<std::mutex> lock(mutex);
        Handle h = next_handle++;
        if (h == kNullHandle || h == INVALID_HANDLE_VALUE) {
            h = 1;
            next_handle = 2;
        }
        while (h == kNullHandle || h == INVALID_HANDLE_VALUE ||
               objects.find(h) != objects.end()) {
            h = next_handle++;
        }
        objects.emplace(h, std::move(obj));
        return h;
    }

    std::shared_ptr<KernelObject> lookup(Handle h) {
        if (h == kNullHandle || h == INVALID_HANDLE_VALUE) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex);
        auto it = objects.find(h);
        if (it == objects.end()) {
            return nullptr;
        }
        return it->second;
    }

    std::shared_ptr<KernelObject> remove(Handle h) {
        if (h == kNullHandle || h == INVALID_HANDLE_VALUE) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex);
        auto it = objects.find(h);
        if (it == objects.end()) {
            return nullptr;
        }
        auto obj = std::move(it->second);
        objects.erase(it);
        return obj;
    }
};

KernelHandleTable& handle_table() {
    static KernelHandleTable table;
    return table;
}

// Called with overlay.mutex held: stamps an entry as created (new) or written (modified).
void stamp_locked(UserdataOverlay& overlay, const std::string& key, bool created) {
    const uint64_t now = now_filetime_ticks();
    UserdataMeta& m = overlay.meta[key];
    if (created || m.created == 0) {
        m.created = now;
    }
    m.accessed = now;
    m.written = now;
}

constexpr char kUserdataRoot[] = "/userdata";

// Called with overlay.mutex held.
bool dir_exists_locked(const UserdataOverlay& overlay, const std::string& key) {
    if (key == kUserdataRoot || overlay.dirs.count(key) != 0) {
        return true;
    }
    const std::string prefix = key + "/";
    for (const auto& kv : overlay.files) {
        if (kv.first.compare(0, prefix.size(), prefix) == 0) {
            return true;
        }
    }
    for (const auto& d : overlay.dirs) {
        if (d.compare(0, prefix.size(), prefix) == 0) {
            return true;
        }
    }
    return false;
}

std::string parent_key(const std::string& key) {
    const size_t slash = key.rfind('/');
    return slash == std::string::npos ? std::string() : key.substr(0, slash);
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

bool to_page_protect(uint32_t protect, PageProtect& out) {
    switch (protect) {
        case PAGE_NOACCESS:
            out = PageProtect::NoAccess;
            return true;
        case PAGE_READONLY:
            out = PageProtect::ReadOnly;
            return true;
        case PAGE_READWRITE:
            out = PageProtect::ReadWrite;
            return true;
        case PAGE_EXECUTE_READ:
            out = PageProtect::ExecuteRead;
            return true;
        case PAGE_EXECUTE_READWRITE:
            out = PageProtect::ExecuteReadWrite;
            return true;
        default:
            return false;
    }
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | ((cp >> 6) & 0x1Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | ((cp >> 12) & 0x0Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | ((cp >> 18) & 0x07u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

bool equals_ascii_ci(std::string_view a, std::string_view b) {
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

bool resolve_httpfs_file_size_locked(FileObject& file, uint64_t& out_size) {
    if (file.has_cached_size) {
        out_size = file.cached_size;
        return true;
    }
    if (file.httpfs_client == nullptr || file.httpfs_handle < 0) {
        return false;
    }

    uint8_t probe = 0;
    const int64_t first = file.httpfs_client->read(file.httpfs_handle, 0, &probe, 1);
    if (first < 0) {
        return false;
    }
    if (first == 0) {
        file.cached_size = 0;
        file.has_cached_size = true;
        out_size = 0;
        return true;
    }

    // Exponential probing followed by binary search to find exact EOF in O(log N) probes.
    uint64_t lo = 1;
    uint64_t hi = 64;
    while (true) {
        const int64_t r = file.httpfs_client->read(file.httpfs_handle, hi, &probe, 1);
        if (r < 0) {
            return false;
        }
        if (r == 0) {
            break;
        }
        lo = hi + 1;
        if (hi > (std::numeric_limits<uint64_t>::max() / 2)) {
            return false;
        }
        hi *= 2;
    }

    while (lo < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        const int64_t r = file.httpfs_client->read(file.httpfs_handle, mid, &probe, 1);
        if (r < 0) {
            return false;
        }
        if (r > 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    file.cached_size = lo;
    file.has_cached_size = true;
    out_size = lo;
    return true;
}

Handle create_file_impl(std::string_view raw_path,
                        uint32_t desired_access,
                        uint32_t creation_disposition) {
    if (raw_path.empty()) {
        SetLastError(ERROR_PATH_NOT_FOUND);
        return INVALID_HANDLE_VALUE;
    }

    bool is_userdata = false;
    const std::string normalized = normalize_path(raw_path, &is_userdata);
    if (normalized.empty()) {
        SetLastError(ERROR_PATH_NOT_FOUND);
        return INVALID_HANDLE_VALUE;
    }

    if (is_userdata) {
        auto& overlay = userdata_overlay();
        std::lock_guard<std::mutex> lock(overlay.mutex);
        auto it = overlay.files.find(normalized);
        const bool exists = (it != overlay.files.end());

        switch (creation_disposition) {
            case CREATE_NEW:
                if (exists) {
                    SetLastError(ERROR_FILE_EXISTS);
                    return INVALID_HANDLE_VALUE;
                }
                overlay.files.emplace(normalized, std::vector<uint8_t>{});
                stamp_locked(overlay, normalized, true);
                SetLastError(ERROR_SUCCESS);
                break;

            case CREATE_ALWAYS:
                overlay.files[normalized].clear();
                stamp_locked(overlay, normalized, !exists);
                SetLastError(exists ? ERROR_ALREADY_EXISTS : ERROR_SUCCESS);
                break;

            case OPEN_EXISTING:
                if (!exists) {
                    SetLastError(ERROR_FILE_NOT_FOUND);
                    return INVALID_HANDLE_VALUE;
                }
                SetLastError(ERROR_SUCCESS);
                break;

            case OPEN_ALWAYS:
                if (!exists) {
                    overlay.files.emplace(normalized, std::vector<uint8_t>{});
                    stamp_locked(overlay, normalized, true);
                    SetLastError(ERROR_SUCCESS);
                } else {
                    SetLastError(ERROR_ALREADY_EXISTS);
                }
                break;

            case TRUNCATE_EXISTING:
                if (!exists) {
                    SetLastError(ERROR_FILE_NOT_FOUND);
                    return INVALID_HANDLE_VALUE;
                }
                if ((desired_access & GENERIC_WRITE) == 0) {
                    SetLastError(ERROR_ACCESS_DENIED);
                    return INVALID_HANDLE_VALUE;
                }
                it->second.clear();
                stamp_locked(overlay, normalized, false);
                SetLastError(ERROR_SUCCESS);
                break;

            default:
                SetLastError(ERROR_INVALID_PARAMETER);
                return INVALID_HANDLE_VALUE;
        }

        auto file = std::make_shared<FileObject>();
        file->is_userdata = true;
        file->desired_access = desired_access;
        file->userdata_key = normalized;
        return handle_table().insert(std::move(file));
    }

    // Read-only HTTPFS asset path.
    if ((desired_access & GENERIC_WRITE) != 0 ||
        creation_disposition == CREATE_NEW ||
        creation_disposition == CREATE_ALWAYS ||
        creation_disposition == TRUNCATE_EXISTING) {
        SetLastError(ERROR_ACCESS_DENIED);
        return INVALID_HANDLE_VALUE;
    }
    if (creation_disposition != OPEN_EXISTING && creation_disposition != OPEN_ALWAYS) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }

    HttpfsMailboxClient* client = get_httpfs_client();
    if (client == nullptr) {
        SetLastError(ERROR_FILE_NOT_FOUND);
        return INVALID_HANDLE_VALUE;
    }

    const int32_t hfs_handle = client->open(normalized);
    if (hfs_handle < 0) {
        SetLastError(ERROR_FILE_NOT_FOUND);
        return INVALID_HANDLE_VALUE;
    }

    auto file = std::make_shared<FileObject>();
    file->is_userdata = false;
    file->desired_access = desired_access;
    file->httpfs_client = client;
    file->httpfs_handle = hfs_handle;
    SetLastError(ERROR_SUCCESS);
    return handle_table().insert(std::move(file));
}

}  // namespace

// ---------------------------------------------------------------------------
// Injectable backends & overlay helpers
// ---------------------------------------------------------------------------

VirtualMemoryArena& default_vmem_arena() {
    // The process guest address space: identity-mapped where the platform allows it, and the
    // translation base lifted code uses unless set_vmem_arena() installs another arena.
    struct ProcessArena {
        VirtualMemoryArena arena{kDefaultArenaBase, kProcessGuestSpaceSize,
                                 ArenaPlacement::PreferIdentity};
        ProcessArena() {
            if (g_vmem_arena.load(std::memory_order_acquire) == nullptr) {
                publish_guest_address_space(arena);
            }
        }
    };
    static ProcessArena instance;
    return instance.arena;
}

void set_vmem_arena(VirtualMemoryArena* arena) {
    g_vmem_arena.store(arena, std::memory_order_release);
    publish_guest_address_space(arena != nullptr ? *arena : default_vmem_arena());
}

VirtualMemoryArena* get_vmem_arena() {
    VirtualMemoryArena* arena = g_vmem_arena.load(std::memory_order_acquire);
    return arena != nullptr ? arena : &default_vmem_arena();
}

TlsTable& default_tls_table() {
    static TlsTable table;
    return table;
}

void set_tls_table(TlsTable* table) {
    g_tls_table.store(table, std::memory_order_release);
}

TlsTable* get_tls_table() {
    TlsTable* table = g_tls_table.load(std::memory_order_acquire);
    return table != nullptr ? table : &default_tls_table();
}

void set_httpfs_client(HttpfsMailboxClient* client) {
    g_httpfs_client.store(client, std::memory_order_release);
}

HttpfsMailboxClient* get_httpfs_client() {
    return g_httpfs_client.load(std::memory_order_acquire);
}

Handle insert_dispatcher_object(std::shared_ptr<DispatcherObject> object) {
    if (!object) {
        return kNullHandle;
    }
    return handle_table().insert(std::make_shared<DispatcherEntry>(std::move(object)));
}

Handle insert_opaque_object(uint32_t tag, std::shared_ptr<void> payload) {
    return handle_table().insert(std::make_shared<OpaqueEntry>(tag, std::move(payload)));
}

std::shared_ptr<void> lookup_opaque_object(Handle handle, uint32_t tag) {
    auto obj = handle_table().lookup(handle);
    if (!obj || obj->kind != KernelObjectKind::Opaque) {
        return nullptr;
    }
    auto* entry = static_cast<OpaqueEntry*>(obj.get());
    return entry->tag == tag ? entry->payload : nullptr;
}

std::shared_ptr<void> remove_opaque_object(Handle handle, uint32_t tag) {
    if (!lookup_opaque_object(handle, tag)) {
        return nullptr;
    }
    auto obj = handle_table().remove(handle);
    if (!obj || obj->kind != KernelObjectKind::Opaque) {
        return nullptr;
    }
    return static_cast<OpaqueEntry*>(obj.get())->payload;
}

std::shared_ptr<DispatcherObject> lookup_dispatcher_object(Handle handle) {
    auto obj = handle_table().lookup(handle);
    if (!obj || obj->kind != KernelObjectKind::Dispatcher) {
        return nullptr;
    }
    return static_cast<DispatcherEntry*>(obj.get())->object;
}

void clear_userdata_overlay() {
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    overlay.files.clear();
    overlay.dirs.clear();
    overlay.meta.clear();
}

bool userdata_file_exists(std::string_view path) {
    bool is_ud = false;
    const std::string key = normalize_path(path, &is_ud);
    if (!is_ud || key.empty()) {
        return false;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    return overlay.files.find(key) != overlay.files.end();
}

bool read_userdata_file(std::string_view path, std::vector<uint8_t>& out) {
    bool is_ud = false;
    const std::string key = normalize_path(path, &is_ud);
    if (!is_ud || key.empty()) {
        return false;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    auto it = overlay.files.find(key);
    if (it == overlay.files.end()) {
        return false;
    }
    out = it->second;
    return true;
}

void write_userdata_file(std::string_view path, std::vector<uint8_t> data) {
    bool is_ud = false;
    const std::string key = normalize_path(path, &is_ud);
    if (!is_ud || key.empty()) {
        return;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    const bool existed = overlay.files.find(key) != overlay.files.end();
    overlay.files[key] = std::move(data);
    stamp_locked(overlay, key, !existed);
}

// ---------------------------------------------------------------------------
// Virtual filesystem seam (O15): directories, timestamps, attributes on the /userdata overlay
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kErrDirNotEmpty = 145u;
constexpr uint32_t kErrDirectory = 267u;
constexpr uint32_t kFileAttrReadOnly = 0x1u;
constexpr uint32_t kSettableAttrMask = 0x1u | 0x2u | 0x4u | 0x20u | 0x100u | 0x2000u;

bool is_userdata_key(std::string_view key) {
    constexpr std::string_view root = "/userdata";
    return key.size() >= root.size() && key.compare(0, root.size(), root) == 0 &&
           (key.size() == root.size() || key[root.size()] == '/');
}

// overlay.mutex held. ERROR_FILE_NOT_FOUND when the parent exists, else ERROR_PATH_NOT_FOUND.
uint32_t missing_error_locked(const UserdataOverlay& overlay, const std::string& key) {
    const std::string parent = parent_key(key);
    if (!parent.empty() && is_userdata_key(parent) && dir_exists_locked(overlay, parent) &&
        overlay.files.find(parent) == overlay.files.end()) {
        return ERROR_FILE_NOT_FOUND;
    }
    return ERROR_PATH_NOT_FOUND;
}

// overlay.mutex held; the entry must exist.
void fill_entry_locked(UserdataOverlay& overlay,
                       const std::string& key,
                       bool is_dir,
                       UserdataEntry* out) {
    UserdataMeta& m = overlay.meta[key];
    if (m.created == 0) {
        const uint64_t now = now_filetime_ticks();
        m.created = m.accessed = m.written = now;
    }
    out->is_directory = is_dir;
    out->size = 0;
    if (!is_dir) {
        out->size = overlay.files.find(key)->second.size();
    }
    out->created = m.created;
    out->accessed = m.accessed;
    out->written = m.written;
    out->attributes = m.attributes;
}

}  // namespace

uint32_t userdata_stat(std::string_view key_view, UserdataEntry* out) {
    const std::string key(key_view);
    if (out == nullptr) {
        return ERROR_INVALID_PARAMETER;
    }
    *out = UserdataEntry{};
    if (!is_userdata_key(key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    if (overlay.files.find(key) != overlay.files.end()) {
        fill_entry_locked(overlay, key, false, out);
        return ERROR_SUCCESS;
    }
    if (dir_exists_locked(overlay, key)) {
        fill_entry_locked(overlay, key, true, out);
        return ERROR_SUCCESS;
    }
    return missing_error_locked(overlay, key);
}

uint32_t userdata_create_directory(std::string_view key_view) {
    const std::string key(key_view);
    if (!is_userdata_key(key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    if (overlay.files.find(key) != overlay.files.end() || dir_exists_locked(overlay, key)) {
        return ERROR_ALREADY_EXISTS;
    }
    const std::string parent = parent_key(key);
    if (overlay.files.find(parent) != overlay.files.end() || !dir_exists_locked(overlay, parent)) {
        return ERROR_PATH_NOT_FOUND;
    }
    overlay.dirs.insert(key);
    stamp_locked(overlay, key, true);
    return ERROR_SUCCESS;
}

uint32_t userdata_remove_directory(std::string_view key_view) {
    const std::string key(key_view);
    if (!is_userdata_key(key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    if (overlay.files.find(key) != overlay.files.end()) {
        return kErrDirectory;
    }
    if (!dir_exists_locked(overlay, key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    if (key == kUserdataRoot) {
        return ERROR_ACCESS_DENIED;
    }
    const std::string prefix = key + "/";
    for (const auto& kv : overlay.files) {
        if (kv.first.compare(0, prefix.size(), prefix) == 0) {
            return kErrDirNotEmpty;
        }
    }
    for (const auto& d : overlay.dirs) {
        if (d.compare(0, prefix.size(), prefix) == 0) {
            return kErrDirNotEmpty;
        }
    }
    overlay.dirs.erase(key);
    overlay.meta.erase(key);
    return ERROR_SUCCESS;
}

uint32_t userdata_delete_file(std::string_view key_view) {
    const std::string key(key_view);
    if (!is_userdata_key(key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    auto it = overlay.files.find(key);
    if (it == overlay.files.end()) {
        return dir_exists_locked(overlay, key) ? ERROR_ACCESS_DENIED
                                               : missing_error_locked(overlay, key);
    }
    auto mit = overlay.meta.find(key);
    if (mit != overlay.meta.end() && (mit->second.attributes & kFileAttrReadOnly) != 0) {
        return ERROR_ACCESS_DENIED;
    }
    overlay.files.erase(it);
    overlay.meta.erase(key);
    return ERROR_SUCCESS;
}

uint32_t userdata_list_directory(std::string_view key_view, std::vector<UserdataEntry>* out) {
    const std::string key(key_view);
    if (out == nullptr) {
        return ERROR_INVALID_PARAMETER;
    }
    out->clear();
    if (!is_userdata_key(key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    if (overlay.files.find(key) != overlay.files.end() || !dir_exists_locked(overlay, key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    const std::string prefix = key + "/";
    std::map<std::string, bool> children;  // name -> is_directory (sorted by name)
    for (const auto& kv : overlay.files) {
        if (kv.first.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const std::string rest = kv.first.substr(prefix.size());
        const size_t slash = rest.find('/');
        if (slash == std::string::npos) {
            children[rest] = false;
        } else {
            children[rest.substr(0, slash)] = true;
        }
    }
    for (const auto& d : overlay.dirs) {
        if (d.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const std::string rest = d.substr(prefix.size());
        const size_t slash = rest.find('/');
        children[slash == std::string::npos ? rest : rest.substr(0, slash)] = true;
    }
    for (const auto& child : children) {
        UserdataEntry entry;
        fill_entry_locked(overlay, prefix + child.first, child.second, &entry);
        entry.name = child.first;
        out->push_back(std::move(entry));
    }
    return ERROR_SUCCESS;
}

uint32_t userdata_rename(std::string_view from_view,
                         std::string_view to_view,
                         bool replace_existing) {
    const std::string from(from_view);
    const std::string to(to_view);
    if (!is_userdata_key(from) || !is_userdata_key(to) || from == kUserdataRoot ||
        to == kUserdataRoot) {
        return ERROR_PATH_NOT_FOUND;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    const bool from_is_file = overlay.files.find(from) != overlay.files.end();
    if (!from_is_file && !dir_exists_locked(overlay, from)) {
        return missing_error_locked(overlay, from);
    }
    if (from == to) {
        return ERROR_SUCCESS;
    }
    const std::string to_parent = parent_key(to);
    if (overlay.files.find(to_parent) != overlay.files.end() ||
        !dir_exists_locked(overlay, to_parent)) {
        return ERROR_PATH_NOT_FOUND;
    }
    const bool to_is_file = overlay.files.find(to) != overlay.files.end();
    const bool to_is_dir = !to_is_file && dir_exists_locked(overlay, to);
    if (to_is_file || to_is_dir) {
        if (!replace_existing) {
            return ERROR_ALREADY_EXISTS;
        }
        if (!from_is_file || to_is_dir) {
            return ERROR_ACCESS_DENIED;  // replacing is only defined for file over file
        }
    }
    if (!from_is_file && to.compare(0, from.size() + 1, from + "/") == 0) {
        return ERROR_INVALID_PARAMETER;  // directory into itself
    }

    if (from_is_file) {
        auto node = overlay.files.extract(from);
        overlay.files.erase(to);
        node.key() = to;
        overlay.files.insert(std::move(node));
        auto mit = overlay.meta.find(from);
        if (mit != overlay.meta.end()) {
            const UserdataMeta m = mit->second;
            overlay.meta.erase(mit);
            overlay.meta[to] = m;
        }
        return ERROR_SUCCESS;
    }

    const std::string from_prefix = from + "/";
    auto rekey = [&](const std::string& old_key) { return to + old_key.substr(from.size()); };
    std::vector<std::string> moved_files;
    for (const auto& kv : overlay.files) {
        if (kv.first.compare(0, from_prefix.size(), from_prefix) == 0) {
            moved_files.push_back(kv.first);
        }
    }
    for (const auto& old_key : moved_files) {
        auto node = overlay.files.extract(old_key);
        node.key() = rekey(old_key);
        overlay.files.insert(std::move(node));
    }
    std::vector<std::string> moved_dirs;
    for (const auto& d : overlay.dirs) {
        if (d == from || d.compare(0, from_prefix.size(), from_prefix) == 0) {
            moved_dirs.push_back(d);
        }
    }
    for (const auto& d : moved_dirs) {
        overlay.dirs.erase(d);
        overlay.dirs.insert(rekey(d));
    }
    std::vector<std::string> moved_meta;
    for (const auto& kv : overlay.meta) {
        if (kv.first == from || kv.first.compare(0, from_prefix.size(), from_prefix) == 0) {
            moved_meta.push_back(kv.first);
        }
    }
    for (const auto& old_key : moved_meta) {
        const UserdataMeta m = overlay.meta[old_key];
        overlay.meta.erase(old_key);
        overlay.meta[rekey(old_key)] = m;
    }
    overlay.dirs.insert(to);  // the moved directory keeps existing even when it ends up empty
    return ERROR_SUCCESS;
}

uint32_t userdata_set_times(std::string_view key_view,
                            const uint64_t* created,
                            const uint64_t* accessed,
                            const uint64_t* written) {
    const std::string key(key_view);
    if (!is_userdata_key(key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    const bool is_file = overlay.files.find(key) != overlay.files.end();
    if (!is_file && !dir_exists_locked(overlay, key)) {
        return missing_error_locked(overlay, key);
    }
    UserdataEntry unused;
    fill_entry_locked(overlay, key, !is_file, &unused);  // materializes the meta record
    UserdataMeta& m = overlay.meta[key];
    if (created != nullptr) {
        m.created = *created;
    }
    if (accessed != nullptr) {
        m.accessed = *accessed;
    }
    if (written != nullptr) {
        m.written = *written;
    }
    return ERROR_SUCCESS;
}

uint32_t userdata_set_attributes(std::string_view key_view, uint32_t attributes) {
    const std::string key(key_view);
    if (!is_userdata_key(key)) {
        return ERROR_PATH_NOT_FOUND;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    const bool is_file = overlay.files.find(key) != overlay.files.end();
    if (!is_file && !dir_exists_locked(overlay, key)) {
        return missing_error_locked(overlay, key);
    }
    UserdataEntry unused;
    fill_entry_locked(overlay, key, !is_file, &unused);
    // Only explicit attribute bits are stored; FILE_ATTRIBUTE_NORMAL (0x80) is the "none set"
    // marker so a later query can tell it from a never-touched entry.
    uint32_t stored = attributes & kSettableAttrMask;
    if (stored == 0) {
        stored = 0x80u;
    }
    overlay.meta[key].attributes = stored;
    return ERROR_SUCCESS;
}

uint64_t userdata_total_bytes() {
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> lock(overlay.mutex);
    uint64_t total = 0;
    for (const auto& kv : overlay.files) {
        total += kv.second.size();
    }
    return total;
}

bool query_file_handle(Handle handle, FileHandleInfo* out) {
    if (out == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    *out = FileHandleInfo{};
    auto obj = handle_table().lookup(handle);
    if (!obj || obj->kind != KernelObjectKind::File) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    auto* file = static_cast<FileObject*>(obj.get());
    std::lock_guard<std::mutex> lock(file->mutex);
    out->is_userdata = file->is_userdata;
    out->desired_access = file->desired_access;
    out->cursor = file->cursor;
    out->userdata_key = file->userdata_key;
    return true;
}

bool set_end_of_file_at_cursor(Handle handle) {
    auto obj = handle_table().lookup(handle);
    if (!obj || obj->kind != KernelObjectKind::File) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    auto* file = static_cast<FileObject*>(obj.get());
    std::lock_guard<std::mutex> lock(file->mutex);
    if (!file->is_userdata || (file->desired_access & GENERIC_WRITE) == 0) {
        SetLastError(ERROR_ACCESS_DENIED);
        return false;
    }
    // Zero-extension is bounded so a hostile seek position cannot exhaust host memory.
    constexpr uint64_t kMaxUserdataFileBytes = 1ull << 32;
    if (file->cursor > kMaxUserdataFileBytes) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return false;
    }
    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> ud_lock(overlay.mutex);
    auto it = overlay.files.find(file->userdata_key);
    if (it == overlay.files.end()) {
        SetLastError(ERROR_FILE_NOT_FOUND);
        return false;
    }
    it->second.resize(static_cast<size_t>(file->cursor), 0);
    stamp_locked(overlay, file->userdata_key, false);
    return true;
}

std::string utf16_to_utf8(const char16_t* src) {
    if (src == nullptr) {
        return {};
    }
    std::string out;
    for (size_t i = 0; src[i] != 0; ++i) {
        const uint16_t unit = static_cast<uint16_t>(src[i]);
        if (unit >= 0xD800u && unit <= 0xDBFFu) {
            const uint16_t next = static_cast<uint16_t>(src[i + 1]);
            if (next >= 0xDC00u && next <= 0xDFFFu) {
                const uint32_t cp =
                    0x10000u + ((static_cast<uint32_t>(unit - 0xD800u) << 10) |
                                static_cast<uint32_t>(next - 0xDC00u));
                append_utf8(out, cp);
                ++i;
            } else {
                append_utf8(out, 0xFFFDu);
            }
        } else if (unit >= 0xDC00u && unit <= 0xDFFFu) {
            append_utf8(out, 0xFFFDu);
        } else {
            append_utf8(out, unit);
        }
    }
    return out;
}

std::string normalize_path(std::string_view raw_path, bool* is_userdata) {
    if (is_userdata != nullptr) {
        *is_userdata = false;
    }
    if (raw_path.empty()) {
        return {};
    }

    std::string s(raw_path);
    for (char& c : s) {
        if (c == '\\') {
            c = '/';
        }
    }

    size_t pos = 0;
    // Strip Win32 extended-length or device prefix (\\?\ or \\.\).
    if (s.size() >= 4 && s[0] == '/' && s[1] == '/' && (s[2] == '?' || s[2] == '.') &&
        s[3] == '/') {
        pos = 4;
    }

    // Strip Windows / Xbox drive prefixes such as G:\, D:\, T:\, C:\.
    if (pos + 2 <= s.size() &&
        std::isalpha(static_cast<unsigned char>(s[pos])) &&
        s[pos + 1] == ':') {
        pos += 2;
        if (pos < s.size() && s[pos] == '/') {
            ++pos;
        }
    }

    // Parse into normalized path segments (collapsing duplicate slashes, '.', and '..').
    std::vector<std::string> segments;
    while (pos < s.size()) {
        while (pos < s.size() && s[pos] == '/') {
            ++pos;
        }
        if (pos >= s.size()) {
            break;
        }
        size_t end = s.find('/', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        std::string_view seg(s.data() + pos, end - pos);
        pos = end;

        if (seg.empty() || seg == ".") {
            continue;
        }
        if (seg == "..") {
            if (!segments.empty()) {
                segments.pop_back();
            }
            continue;
        }
        segments.emplace_back(seg);
    }

    if (segments.empty()) {
        return {};
    }

    if (equals_ascii_ci(segments[0], "userdata")) {
        if (segments.size() < 2) {
            return {};
        }
        if (is_userdata != nullptr) {
            *is_userdata = true;
        }
        std::string result = "/userdata";
        for (size_t i = 1; i < segments.size(); ++i) {
            result.push_back('/');
            result.append(segments[i]);
        }
        return result;
    }

    std::string result;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i > 0) {
            result.push_back('/');
        }
        result.append(segments[i]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Virtual memory shims
// ---------------------------------------------------------------------------

uint64_t VirtualAlloc(uint64_t address,
                      uint64_t size,
                      uint32_t allocation_type,
                      uint32_t protect) {
    PageProtect pp = PageProtect::NoAccess;
    if (size == 0 || allocation_type == 0 ||
        (allocation_type & ~(MEM_RESERVE | MEM_COMMIT)) != 0 ||
        !to_page_protect(protect, pp)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }

    const uint64_t result = get_vmem_arena()->virtual_alloc(address, size, allocation_type, pp);
    if (result == 0) {
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    }
    return result;
}

bool VirtualProtect(uint64_t address,
                    uint64_t size,
                    uint32_t new_protect,
                    uint32_t* old_protect) {
    PageProtect pp = PageProtect::NoAccess;
    if (size == 0 || !to_page_protect(new_protect, pp)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }

    PageProtect prev = PageProtect::NoAccess;
    const bool ok = get_vmem_arena()->protect(address, size, pp, &prev);
    if (!ok) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    if (old_protect != nullptr) {
        *old_protect = static_cast<uint32_t>(prev);
    }
    return true;
}

bool VirtualFree(uint64_t address, uint64_t size, uint32_t free_type) {
    // MEM_RELEASE may carry MEM_PRESERVE_PLACEHOLDER (split / back to a placeholder) or
    // MEM_COALESCE_PLACEHOLDERS; the arena implements both.
    if (free_type != MEM_DECOMMIT && free_type != MEM_RELEASE &&
        free_type != (MEM_RELEASE | kMemPreservePlaceholder) &&
        free_type != (MEM_RELEASE | kMemCoalescePlaceholders)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    const bool ok = get_vmem_arena()->virtual_free(address, size, free_type);
    if (!ok) {
        SetLastError(ERROR_INVALID_PARAMETER);
    }
    return ok;
}

size_t VirtualQuery(uint64_t address, MemoryBasicInformation* buffer, size_t length) {
    if (buffer == nullptr || length < sizeof(MemoryBasicInformation)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return 0;
    }

    MemoryBasicInfo info = get_vmem_arena()->query(address);
    xmem_windows::Window phantom;
    if (info.state == MemState::Free && xmem_windows::is_phantom(address) &&
        xmem_windows::find_window(address, 1, &phantom)) {
        // O27: a GPU window placed outside the guest pool reads as a no-access reservation.
        info.base_address = address & ~(kPageSize - 1);
        info.allocation_base = phantom.base;
        info.region_size = phantom.base + phantom.size - info.base_address;
        info.state = MemState::Reserve;
        info.protect = PageProtect::NoAccess;
        info.allocation_protect = PageProtect::NoAccess;
    }
    buffer->base_address = info.base_address;
    buffer->allocation_base = info.allocation_base;
    buffer->allocation_protect = static_cast<uint32_t>(info.allocation_protect);
    buffer->partition_id = 0;
    buffer->region_size = info.region_size;
    buffer->state = static_cast<uint32_t>(info.state);
    buffer->protect = static_cast<uint32_t>(info.protect);
    buffer->type = (info.state == MemState::Free) ? 0u : MEM_PRIVATE;
    return sizeof(MemoryBasicInformation);
}

// ---------------------------------------------------------------------------
// Threading & TLS shims
// ---------------------------------------------------------------------------

uint32_t TlsAlloc() {
    const uint32_t idx = get_tls_table()->tls_alloc();
    if (idx == TLS_OUT_OF_INDEXES) {
        SetLastError(ERROR_NO_MORE_ITEMS);
    }
    return idx;
}

bool TlsFree(uint32_t tls_index) {
    const bool ok = get_tls_table()->tls_free(tls_index);
    if (!ok) {
        SetLastError(ERROR_INVALID_PARAMETER);
    }
    return ok;
}

void* TlsGetValue(uint32_t tls_index) {
    if (tls_index >= TlsTable::kMaxSlots) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    void* val = get_tls_table()->tls_get(tls_index);
    SetLastError(ERROR_SUCCESS);
    return val;
}

bool TlsSetValue(uint32_t tls_index, void* value) {
    const bool ok = get_tls_table()->tls_set(tls_index, value);
    if (!ok) {
        SetLastError(ERROR_INVALID_PARAMETER);
    }
    return ok;
}

void InitializeSRWLock(SrwLock* srw_lock) {
    (void)srw_lock;
}

void AcquireSRWLockExclusive(SrwLock* srw_lock) {
    if (srw_lock != nullptr) {
        srw_lock->lock.lock_exclusive();
    }
}

void ReleaseSRWLockExclusive(SrwLock* srw_lock) {
    if (srw_lock != nullptr) {
        srw_lock->lock.unlock_exclusive();
    }
}

void AcquireSRWLockShared(SrwLock* srw_lock) {
    if (srw_lock != nullptr) {
        srw_lock->lock.lock_shared();
    }
}

void ReleaseSRWLockShared(SrwLock* srw_lock) {
    if (srw_lock != nullptr) {
        srw_lock->lock.unlock_shared();
    }
}

bool TryAcquireSRWLockExclusive(SrwLock* srw_lock) {
    if (srw_lock == nullptr) {
        return false;
    }
    return srw_lock->lock.try_lock_exclusive();
}

bool TryAcquireSRWLockShared(SrwLock* srw_lock) {
    if (srw_lock == nullptr) {
        return false;
    }
    return srw_lock->lock.try_lock_shared();
}

// ---------------------------------------------------------------------------
// Synchronization shims (CriticalSection & Event)
// ---------------------------------------------------------------------------

namespace {

// Locks of critical sections, keyed by the section's address (see Pg6CriticalSection). Sharded so
// unrelated sections do not contend on one table lock.
struct CriticalSectionShard {
    std::mutex mutex;
    std::unordered_map<const void*, std::shared_ptr<std::recursive_mutex>> locks;
};

constexpr size_t kCriticalSectionShards = 64;

CriticalSectionShard& critical_section_shard(const void* address) {
    static CriticalSectionShard shards[kCriticalSectionShards];
    const auto key = reinterpret_cast<uintptr_t>(address);
    return shards[(key >> 4) % kCriticalSectionShards];
}

// The section's lock; created when missing and `create` (a zero-initialized section entered
// without an Initialize call behaves like an initialized one).
std::shared_ptr<std::recursive_mutex> critical_section_lock(const CriticalSection* cs, bool create) {
    CriticalSectionShard& shard = critical_section_shard(cs);
    std::lock_guard<std::mutex> guard(shard.mutex);
    auto it = shard.locks.find(cs);
    if (it != shard.locks.end()) {
        return it->second;
    }
    if (!create) {
        return nullptr;
    }
    auto lock = std::make_shared<std::recursive_mutex>();
    shard.locks.emplace(cs, lock);
    return lock;
}

uint64_t current_thread_owner_id() {
    // Any nonzero per-thread value works for inline OwningThread checks; prefer the shim's id.
    static std::atomic<uint64_t> next{1};
    thread_local const uint64_t id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}

}  // namespace

void InitializeCriticalSectionWithSpinCount(CriticalSection* critical_section, uint32_t spin_count) {
    if (critical_section == nullptr) {
        return;
    }
    // Construct in place: never read what the guest storage held before (uninitialized memory,
    // or a section that was deleted and is now initialized again).
    *critical_section = CriticalSection{~uint64_t{0}, -1, 0, 0, 0, spin_count};
    CriticalSectionShard& shard = critical_section_shard(critical_section);
    std::lock_guard<std::mutex> guard(shard.mutex);
    shard.locks[critical_section] = std::make_shared<std::recursive_mutex>();
}

void InitializeCriticalSection(CriticalSection* critical_section) {
    InitializeCriticalSectionWithSpinCount(critical_section, 0);
}

void DeleteCriticalSection(CriticalSection* critical_section) {
    if (critical_section == nullptr) {
        return;
    }
    CriticalSectionShard& shard = critical_section_shard(critical_section);
    std::lock_guard<std::mutex> guard(shard.mutex);
    shard.locks.erase(critical_section);
}

void EnterCriticalSection(CriticalSection* critical_section) {
    if (critical_section == nullptr) {
        SetLastError(ERROR_INVALID_HANDLE);
        return;
    }
    critical_section_lock(critical_section, true)->lock();
    ++critical_section->recursion_count;
    critical_section->owning_thread = current_thread_owner_id();
}

bool TryEnterCriticalSection(CriticalSection* critical_section) {
    if (critical_section == nullptr) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }
    if (!critical_section_lock(critical_section, true)->try_lock()) {
        return false;
    }
    ++critical_section->recursion_count;
    critical_section->owning_thread = current_thread_owner_id();
    return true;
}

void LeaveCriticalSection(CriticalSection* critical_section) {
    const auto lock = critical_section != nullptr ? critical_section_lock(critical_section, false) : nullptr;
    if (lock == nullptr) {
        SetLastError(ERROR_INVALID_HANDLE);
        return;
    }
    if (critical_section->recursion_count > 0 && --critical_section->recursion_count == 0) {
        critical_section->owning_thread = 0;
    }
    lock->unlock();
}

// CreateEventA/W, SetEvent, ResetEvent and WaitForSingleObject live in os/win32_sync.cpp.

// ---------------------------------------------------------------------------
// File I/O shims
// ---------------------------------------------------------------------------

Handle CreateFileA(const char* file_name,
                   uint32_t desired_access,
                   uint32_t share_mode,
                   void* security_attributes,
                   uint32_t creation_disposition,
                   uint32_t flags_and_attributes,
                   Handle template_file) {
    (void)share_mode;
    (void)security_attributes;
    (void)flags_and_attributes;
    (void)template_file;
    if (file_name == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    return create_file_impl(file_name, desired_access, creation_disposition);
}

Handle CreateFileW(const char16_t* file_name,
                   uint32_t desired_access,
                   uint32_t share_mode,
                   void* security_attributes,
                   uint32_t creation_disposition,
                   uint32_t flags_and_attributes,
                   Handle template_file) {
    (void)share_mode;
    (void)security_attributes;
    (void)flags_and_attributes;
    (void)template_file;
    if (file_name == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return INVALID_HANDLE_VALUE;
    }
    const std::string utf8_path = utf16_to_utf8(file_name);
    return create_file_impl(utf8_path, desired_access, creation_disposition);
}

namespace {

// x64 OVERLAPPED: Internal, InternalHigh, Offset, OffsetHigh, hEvent (32 bytes).
struct OverlappedLayout {
    uint64_t internal;
    uint64_t internal_high;
    uint32_t offset;
    uint32_t offset_high;
    uint64_t event;
};
static_assert(sizeof(OverlappedLayout) == 32, "OVERLAPPED is 32 bytes on x64");

uint64_t overlapped_offset(const void* overlapped) {
    OverlappedLayout o;
    std::memcpy(&o, overlapped, sizeof(o));
    return (static_cast<uint64_t>(o.offset_high) << 32) | o.offset;
}

// Every transfer completes synchronously: record the status and byte count in the OVERLAPPED and
// signal its event, as a completed asynchronous operation would.
void complete_overlapped(void* overlapped, uint32_t status, uint64_t transferred) {
    OverlappedLayout o;
    std::memcpy(&o, overlapped, sizeof(o));
    o.internal = status;
    o.internal_high = transferred;
    std::memcpy(overlapped, &o, sizeof(o));
    if (o.event != 0) {
        SetEvent(static_cast<Handle>(o.event));
    }
}

constexpr uint32_t kStatusSuccess = 0;
constexpr uint32_t kStatusEndOfFile = 0xC0000011u;

}  // namespace

namespace {

// Reads at `*start` (an OVERLAPPED offset) or at the file pointer; either way the file pointer
// ends after the bytes read, as for a synchronous handle.
bool read_file_at(Handle file_handle,
                  void* buffer,
                  uint32_t bytes_to_read,
                  uint32_t* bytes_read,
                  const uint64_t* start) {
    auto obj = handle_table().lookup(file_handle);
    if (!obj || obj->kind != KernelObjectKind::File) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }

    auto* file = static_cast<FileObject*>(obj.get());
    std::lock_guard<std::mutex> lock(file->mutex);
    if (start != nullptr) {
        file->cursor = *start;
    }

    if ((file->desired_access & GENERIC_READ) == 0) {
        SetLastError(ERROR_ACCESS_DENIED);
        return false;
    }
    if (bytes_to_read == 0) {
        return true;
    }
    if (buffer == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }

    if (file->is_userdata) {
        auto& overlay = userdata_overlay();
        std::lock_guard<std::mutex> ud_lock(overlay.mutex);
        auto it = overlay.files.find(file->userdata_key);
        if (it == overlay.files.end()) {
            SetLastError(ERROR_FILE_NOT_FOUND);
            return false;
        }
        const auto& data = it->second;
        if (file->cursor >= data.size()) {
            if (bytes_read != nullptr) {
                *bytes_read = 0;
            }
            return true;
        }
        const size_t avail = data.size() - static_cast<size_t>(file->cursor);
        const size_t to_copy = std::min<size_t>(avail, bytes_to_read);
        std::memcpy(buffer, data.data() + static_cast<size_t>(file->cursor), to_copy);
        file->cursor += to_copy;
        if (bytes_read != nullptr) {
            *bytes_read = static_cast<uint32_t>(to_copy);
        }
        return true;
    }

    if (file->httpfs_client == nullptr || file->httpfs_handle < 0) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }

    const int64_t got =
        file->httpfs_client->read(file->httpfs_handle, file->cursor, buffer, bytes_to_read);
    if (got < 0) {
        SetLastError(ERROR_READ_FAULT);
        return false;
    }
    file->cursor += static_cast<uint64_t>(got);
    if (bytes_read != nullptr) {
        *bytes_read = static_cast<uint32_t>(got);
    }
    return true;
}

bool write_file_at(Handle file_handle,
                   const void* buffer,
                   uint32_t bytes_to_write,
                   uint32_t* bytes_written,
                   const uint64_t* start) {
    auto obj = handle_table().lookup(file_handle);
    if (!obj || obj->kind != KernelObjectKind::File) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }

    auto* file = static_cast<FileObject*>(obj.get());
    std::lock_guard<std::mutex> lock(file->mutex);
    if (start != nullptr) {
        file->cursor = *start;
    }

    if (!file->is_userdata || (file->desired_access & GENERIC_WRITE) == 0) {
        SetLastError(ERROR_ACCESS_DENIED);
        return false;
    }
    if (bytes_to_write == 0) {
        return true;
    }
    if (buffer == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }

    auto& overlay = userdata_overlay();
    std::lock_guard<std::mutex> ud_lock(overlay.mutex);
    auto& data = overlay.files[file->userdata_key];
    const uint64_t end_pos = file->cursor + bytes_to_write;
    if (end_pos > data.size()) {
        data.resize(static_cast<size_t>(end_pos), 0);
    }
    std::memcpy(data.data() + static_cast<size_t>(file->cursor), buffer, bytes_to_write);
    file->cursor = end_pos;
    stamp_locked(overlay, file->userdata_key, false);
    if (bytes_written != nullptr) {
        *bytes_written = bytes_to_write;
    }
    return true;
}

}  // namespace

bool ReadFile(Handle file_handle,
              void* buffer,
              uint32_t bytes_to_read,
              uint32_t* bytes_read,
              void* overlapped) {
    if (bytes_read != nullptr) {
        *bytes_read = 0;
    }
    if (overlapped == nullptr) {
        return read_file_at(file_handle, buffer, bytes_to_read, bytes_read, nullptr);
    }
    // A read with an OVERLAPPED starts at its Offset/OffsetHigh, not at the file pointer, and
    // reading at or past the end fails with ERROR_HANDLE_EOF.
    const uint64_t offset = overlapped_offset(overlapped);
    uint32_t got = 0;
    if (!read_file_at(file_handle, buffer, bytes_to_read, &got, &offset)) {
        return false;
    }
    if (bytes_read != nullptr) {
        *bytes_read = got;
    }
    if (got == 0 && bytes_to_read != 0) {
        complete_overlapped(overlapped, kStatusEndOfFile, 0);
        SetLastError(ERROR_HANDLE_EOF);
        return false;
    }
    complete_overlapped(overlapped, kStatusSuccess, got);
    return true;
}

bool WriteFile(Handle file_handle,
               const void* buffer,
               uint32_t bytes_to_write,
               uint32_t* bytes_written,
               void* overlapped) {
    if (bytes_written != nullptr) {
        *bytes_written = 0;
    }
    if (overlapped == nullptr) {
        return write_file_at(file_handle, buffer, bytes_to_write, bytes_written, nullptr);
    }
    // A write with an OVERLAPPED starts at its Offset/OffsetHigh, not at the file pointer.
    const uint64_t offset = overlapped_offset(overlapped);
    uint32_t put = 0;
    if (!write_file_at(file_handle, buffer, bytes_to_write, &put, &offset)) {
        return false;
    }
    if (bytes_written != nullptr) {
        *bytes_written = put;
    }
    complete_overlapped(overlapped, kStatusSuccess, put);
    return true;
}

bool SetFilePointerEx(Handle file_handle,
                      int64_t distance_to_move,
                      int64_t* new_file_pointer,
                      uint32_t move_method) {
    auto obj = handle_table().lookup(file_handle);
    if (!obj || obj->kind != KernelObjectKind::File) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }

    auto* file = static_cast<FileObject*>(obj.get());
    std::lock_guard<std::mutex> lock(file->mutex);

    int64_t base = 0;
    switch (move_method) {
        case FILE_BEGIN:
            base = 0;
            break;

        case FILE_CURRENT:
            if (file->cursor > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                SetLastError(ERROR_INVALID_PARAMETER);
                return false;
            }
            base = static_cast<int64_t>(file->cursor);
            break;

        case FILE_END: {
            uint64_t size = 0;
            if (file->is_userdata) {
                auto& overlay = userdata_overlay();
                std::lock_guard<std::mutex> ud_lock(overlay.mutex);
                auto it = overlay.files.find(file->userdata_key);
                if (it == overlay.files.end()) {
                    SetLastError(ERROR_FILE_NOT_FOUND);
                    return false;
                }
                size = it->second.size();
            } else {
                if (!resolve_httpfs_file_size_locked(*file, size)) {
                    SetLastError(ERROR_READ_FAULT);
                    return false;
                }
            }
            if (size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                SetLastError(ERROR_INVALID_PARAMETER);
                return false;
            }
            base = static_cast<int64_t>(size);
            break;
        }

        default:
            SetLastError(ERROR_INVALID_PARAMETER);
            return false;
    }

    if ((distance_to_move > 0 &&
         base > std::numeric_limits<int64_t>::max() - distance_to_move) ||
        (distance_to_move < 0 &&
         base < std::numeric_limits<int64_t>::min() - distance_to_move)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }

    const int64_t target = base + distance_to_move;
    if (target < 0) {
        SetLastError(ERROR_NEGATIVE_SEEK);
        return false;
    }

    file->cursor = static_cast<uint64_t>(target);
    if (new_file_pointer != nullptr) {
        *new_file_pointer = target;
    }
    return true;
}

bool GetFileSizeEx(Handle file_handle, int64_t* file_size) {
    if (file_size == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }

    auto obj = handle_table().lookup(file_handle);
    if (!obj || obj->kind != KernelObjectKind::File) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }

    auto* file = static_cast<FileObject*>(obj.get());
    std::lock_guard<std::mutex> lock(file->mutex);

    uint64_t size = 0;
    if (file->is_userdata) {
        auto& overlay = userdata_overlay();
        std::lock_guard<std::mutex> ud_lock(overlay.mutex);
        auto it = overlay.files.find(file->userdata_key);
        if (it == overlay.files.end()) {
            SetLastError(ERROR_FILE_NOT_FOUND);
            return false;
        }
        size = it->second.size();
    } else {
        if (!resolve_httpfs_file_size_locked(*file, size)) {
            SetLastError(ERROR_READ_FAULT);
            return false;
        }
    }

    *file_size = static_cast<int64_t>(size);
    return true;
}

bool CloseHandle(Handle handle) {
    auto obj = handle_table().remove(handle);
    if (!obj) {
        SetLastError(ERROR_INVALID_HANDLE);
        return false;
    }

    if (obj->kind == KernelObjectKind::File) {
        auto* file = static_cast<FileObject*>(obj.get());
        std::lock_guard<std::mutex> lock(file->mutex);
        if (!file->is_userdata && file->httpfs_client != nullptr && file->httpfs_handle >= 0) {
            file->httpfs_client->close(file->httpfs_handle);
            file->httpfs_handle = -1;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Timing & thread-local error state
// ---------------------------------------------------------------------------

bool QueryPerformanceFrequency(int64_t* frequency) {
    if (frequency == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    *frequency = 1'000'000'000LL;  // 1 GHz (nanosecond ticks)
    return true;
}

bool QueryPerformanceCounter(int64_t* performance_count) {
    if (performance_count == nullptr) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    const auto elapsed = std::chrono::steady_clock::now() - process_start_time();
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    *performance_count = (ns >= 0 ? ns : 0) + 1;
    return true;
}

uint64_t GetTickCount64() {
    const auto elapsed = std::chrono::steady_clock::now() - process_start_time();
    const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    return ms >= 0 ? static_cast<uint64_t>(ms) : 0ULL;
}

void GetSystemTimeAsFileTime(FileTime* system_time_as_file_time) {
    if (system_time_as_file_time == nullptr) {
        return;
    }
    constexpr uint64_t kEpochDiff100Ns = 116444736000000000ULL;
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const int64_t unix_100ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() / 100;
    const uint64_t filetime_ticks =
        kEpochDiff100Ns + static_cast<uint64_t>(unix_100ns > 0 ? unix_100ns : 0);
    system_time_as_file_time->low_date_time =
        static_cast<uint32_t>(filetime_ticks & 0xFFFFFFFFULL);
    system_time_as_file_time->high_date_time =
        static_cast<uint32_t>((filetime_ticks >> 32) & 0xFFFFFFFFULL);
}

uint32_t GetLastError() {
    return g_last_error_slot != nullptr ? *g_last_error_slot : g_last_error;
}

void SetLastError(uint32_t error_code) {
    if (g_last_error_slot != nullptr) {
        *g_last_error_slot = error_code;
    } else {
        g_last_error = error_code;
    }
}

uint32_t* SetThreadLastErrorSlot(uint32_t* slot) {
    uint32_t* previous = g_last_error_slot;
    g_last_error_slot = slot;
    return previous;
}

}  // namespace playgta6::os::win32

// ---------------------------------------------------------------------------
// Prefixed C-ABI entrypoints
// ---------------------------------------------------------------------------

extern "C" {

uint64_t pg6_win32_virtual_alloc(uint64_t address,
                                 uint64_t size,
                                 uint32_t allocation_type,
                                 uint32_t protect) {
    return playgta6::os::win32::VirtualAlloc(address, size, allocation_type, protect);
}

int32_t pg6_win32_virtual_protect(uint64_t address,
                                  uint64_t size,
                                  uint32_t new_protect,
                                  uint32_t* old_protect) {
    return playgta6::os::win32::VirtualProtect(address, size, new_protect, old_protect) ? 1 : 0;
}

int32_t pg6_win32_virtual_free(uint64_t address, uint64_t size, uint32_t free_type) {
    return playgta6::os::win32::VirtualFree(address, size, free_type) ? 1 : 0;
}

size_t pg6_win32_virtual_query(uint64_t address,
                               Pg6MemoryBasicInformation* buffer,
                               size_t length) {
    return playgta6::os::win32::VirtualQuery(address, buffer, length);
}

uint32_t pg6_win32_tls_alloc(void) {
    return playgta6::os::win32::TlsAlloc();
}

int32_t pg6_win32_tls_free(uint32_t tls_index) {
    return playgta6::os::win32::TlsFree(tls_index) ? 1 : 0;
}

void* pg6_win32_tls_get_value(uint32_t tls_index) {
    return playgta6::os::win32::TlsGetValue(tls_index);
}

int32_t pg6_win32_tls_set_value(uint32_t tls_index, void* value) {
    return playgta6::os::win32::TlsSetValue(tls_index, value) ? 1 : 0;
}

void pg6_win32_initialize_srw_lock(Pg6SrwLock* srw_lock) {
    playgta6::os::win32::InitializeSRWLock(srw_lock);
}

void pg6_win32_acquire_srw_lock_exclusive(Pg6SrwLock* srw_lock) {
    playgta6::os::win32::AcquireSRWLockExclusive(srw_lock);
}

void pg6_win32_release_srw_lock_exclusive(Pg6SrwLock* srw_lock) {
    playgta6::os::win32::ReleaseSRWLockExclusive(srw_lock);
}

void pg6_win32_acquire_srw_lock_shared(Pg6SrwLock* srw_lock) {
    playgta6::os::win32::AcquireSRWLockShared(srw_lock);
}

void pg6_win32_release_srw_lock_shared(Pg6SrwLock* srw_lock) {
    playgta6::os::win32::ReleaseSRWLockShared(srw_lock);
}

int32_t pg6_win32_try_acquire_srw_lock_exclusive(Pg6SrwLock* srw_lock) {
    return playgta6::os::win32::TryAcquireSRWLockExclusive(srw_lock) ? 1 : 0;
}

int32_t pg6_win32_try_acquire_srw_lock_shared(Pg6SrwLock* srw_lock) {
    return playgta6::os::win32::TryAcquireSRWLockShared(srw_lock) ? 1 : 0;
}

void pg6_win32_initialize_critical_section(Pg6CriticalSection* critical_section) {
    playgta6::os::win32::InitializeCriticalSection(critical_section);
}

void pg6_win32_delete_critical_section(Pg6CriticalSection* critical_section) {
    playgta6::os::win32::DeleteCriticalSection(critical_section);
}

void pg6_win32_enter_critical_section(Pg6CriticalSection* critical_section) {
    playgta6::os::win32::EnterCriticalSection(critical_section);
}

int32_t pg6_win32_try_enter_critical_section(Pg6CriticalSection* critical_section) {
    return playgta6::os::win32::TryEnterCriticalSection(critical_section) ? 1 : 0;
}

void pg6_win32_leave_critical_section(Pg6CriticalSection* critical_section) {
    playgta6::os::win32::LeaveCriticalSection(critical_section);
}

Pg6Handle pg6_win32_create_event_a(void* security_attributes,
                                   int32_t manual_reset,
                                   int32_t initial_state,
                                   const char* name) {
    return playgta6::os::win32::CreateEventA(
        security_attributes, manual_reset != 0, initial_state != 0, name);
}

Pg6Handle pg6_win32_create_event_w(void* security_attributes,
                                   int32_t manual_reset,
                                   int32_t initial_state,
                                   const char16_t* name) {
    return playgta6::os::win32::CreateEventW(
        security_attributes, manual_reset != 0, initial_state != 0, name);
}

int32_t pg6_win32_set_event(Pg6Handle event_handle) {
    return playgta6::os::win32::SetEvent(event_handle) ? 1 : 0;
}

int32_t pg6_win32_reset_event(Pg6Handle event_handle) {
    return playgta6::os::win32::ResetEvent(event_handle) ? 1 : 0;
}

uint32_t pg6_win32_wait_for_single_object(Pg6Handle handle, uint32_t timeout_ms) {
    return playgta6::os::win32::WaitForSingleObject(handle, timeout_ms);
}

Pg6Handle pg6_win32_create_file_a(const char* file_name,
                                  uint32_t desired_access,
                                  uint32_t share_mode,
                                  void* security_attributes,
                                  uint32_t creation_disposition,
                                  uint32_t flags_and_attributes,
                                  Pg6Handle template_file) {
    return playgta6::os::win32::CreateFileA(
        file_name, desired_access, share_mode, security_attributes,
        creation_disposition, flags_and_attributes, template_file);
}

Pg6Handle pg6_win32_create_file_w(const char16_t* file_name,
                                  uint32_t desired_access,
                                  uint32_t share_mode,
                                  void* security_attributes,
                                  uint32_t creation_disposition,
                                  uint32_t flags_and_attributes,
                                  Pg6Handle template_file) {
    return playgta6::os::win32::CreateFileW(
        file_name, desired_access, share_mode, security_attributes,
        creation_disposition, flags_and_attributes, template_file);
}

int32_t pg6_win32_read_file(Pg6Handle file_handle,
                            void* buffer,
                            uint32_t bytes_to_read,
                            uint32_t* bytes_read,
                            void* overlapped) {
    return playgta6::os::win32::ReadFile(
               file_handle, buffer, bytes_to_read, bytes_read, overlapped)
               ? 1
               : 0;
}

int32_t pg6_win32_write_file(Pg6Handle file_handle,
                             const void* buffer,
                             uint32_t bytes_to_write,
                             uint32_t* bytes_written,
                             void* overlapped) {
    return playgta6::os::win32::WriteFile(
               file_handle, buffer, bytes_to_write, bytes_written, overlapped)
               ? 1
               : 0;
}

int32_t pg6_win32_set_file_pointer_ex(Pg6Handle file_handle,
                                      int64_t distance_to_move,
                                      int64_t* new_file_pointer,
                                      uint32_t move_method) {
    return playgta6::os::win32::SetFilePointerEx(
               file_handle, distance_to_move, new_file_pointer, move_method)
               ? 1
               : 0;
}

int32_t pg6_win32_get_file_size_ex(Pg6Handle file_handle, int64_t* file_size) {
    return playgta6::os::win32::GetFileSizeEx(file_handle, file_size) ? 1 : 0;
}

int32_t pg6_win32_close_handle(Pg6Handle handle) {
    return playgta6::os::win32::CloseHandle(handle) ? 1 : 0;
}

int32_t pg6_win32_query_performance_counter(int64_t* performance_count) {
    return playgta6::os::win32::QueryPerformanceCounter(performance_count) ? 1 : 0;
}

int32_t pg6_win32_query_performance_frequency(int64_t* frequency) {
    return playgta6::os::win32::QueryPerformanceFrequency(frequency) ? 1 : 0;
}

uint64_t pg6_win32_get_tick_count64(void) {
    return playgta6::os::win32::GetTickCount64();
}

void pg6_win32_get_system_time_as_file_time(Pg6FileTime* system_time_as_file_time) {
    playgta6::os::win32::GetSystemTimeAsFileTime(system_time_as_file_time);
}

uint32_t pg6_win32_get_last_error(void) {
    return playgta6::os::win32::GetLastError();
}

void pg6_win32_set_last_error(uint32_t error_code) {
    playgta6::os::win32::SetLastError(error_code);
}

}  // extern "C"
