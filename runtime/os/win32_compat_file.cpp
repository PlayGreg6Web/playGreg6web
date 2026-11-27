// File system part of the remaining Win32 core surface (task O15): directories, attributes, find
// handles, file times, disk space and moves on top of the /userdata overlay seam of
// os/win32_kernel.cpp. See os/win32_compat.h for the overview and limitations.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "os/win32_compat.h"
#include "os/win32_compat_internal.h"

namespace playgta6::os::win32 {

namespace {

using compat_detail::fold_char;
using compat_detail::kTagFind;
using compat_detail::ResolvedPath;

constexpr char kUserdataRoot[] = "/userdata";
// Timestamp reported for read-only game data (2020-01-01T00:00:00Z): files shipped with the game
// have no meaningful time and a fixed value keeps results deterministic.
constexpr uint64_t kGameDataTicks = 132223104000000000ULL;
// Capacity reported for the writable /userdata volume.
constexpr uint64_t kUserdataQuotaBytes = 64ULL << 30;
constexpr uint64_t kFileTimeMax = 0x7FFFFFFFFFFFFFFFULL;
constexpr uint32_t kAttrNormal = 0x80u;

struct Query {
    bool is_directory = false;
    uint64_t size = 0;
    uint64_t created = 0;
    uint64_t accessed = 0;
    uint64_t written = 0;
    uint32_t attributes = 0;
};

uint32_t userdata_attributes(const UserdataEntry& e) {
    if (e.is_directory) {
        return FILE_ATTRIBUTE_DIRECTORY | (e.attributes & ~kAttrNormal);
    }
    return e.attributes != 0 ? e.attributes : FILE_ATTRIBUTE_ARCHIVE;
}

Query query_from_entry(const UserdataEntry& e) {
    Query q;
    q.is_directory = e.is_directory;
    q.size = e.size;
    q.created = e.created;
    q.accessed = e.accessed;
    q.written = e.written;
    q.attributes = userdata_attributes(e);
    return q;
}

// Thread-safe read-only HTTPFS stat cache keyed by normalized game-data path and tagged with
// httpfs_client_generation(). Caches both positive existence (with optional size once queried) and
// negative probes (e.g. ERROR_FILE_NOT_FOUND) to avoid redundant CreateFileA + CloseHandle HTTPFS
// roundtrips on repeated attribute/find queries.
struct GameFileStatEntry {
    uint32_t err = ERROR_FILE_NOT_FOUND;
    bool has_size = false;
    uint64_t size = 0;
};

struct GameFileStatCache {
    std::shared_mutex mutex;
    uint64_t generation = 0;
    std::unordered_map<std::string, GameFileStatEntry> entries;
};

GameFileStatCache& game_file_stat_cache() {
    static GameFileStatCache cache;
    return cache;
}

// Probes a read-only game-data file through the file backend (CreateFileA, plus GetFileSizeEx when
// `size` is non-null), keeping the caller's last error intact. Returns ERROR_SUCCESS or the open
// error. Callers that only need existence/attributes pass null: on HTTPFS the size probe is a
// series of range reads, far more than the open.
uint32_t probe_game_file(const std::string& key, uint64_t* size) {
    const uint64_t gen = httpfs_client_generation();
    const bool stable_client = (gen & 1u) == 0u && get_httpfs_client() != nullptr;
    if (!stable_client) {
        auto& cache = game_file_stat_cache();
        if ((gen & 1u) == 0u) {
            std::shared_lock<std::shared_mutex> rlock(cache.mutex);
            if (cache.generation != gen && !cache.entries.empty()) {
                rlock.unlock();
                std::unique_lock<std::shared_mutex> wlock(cache.mutex);
                if (cache.generation != gen) {
                    cache.entries.clear();
                    cache.generation = gen;
                }
            }
        }
        const Handle h = CreateFileA(key.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, kNullHandle);
        if (h == INVALID_HANDLE_VALUE) {
            const uint32_t err = GetLastError();
            return err == ERROR_SUCCESS ? ERROR_FILE_NOT_FOUND : err;
        }
        if (size == nullptr) {
            CloseHandle(h);
            return ERROR_SUCCESS;
        }
        int64_t file_size = 0;
        const bool ok = GetFileSizeEx(h, &file_size);
        CloseHandle(h);
        *size = ok && file_size > 0 ? static_cast<uint64_t>(file_size) : 0;
        return ERROR_SUCCESS;
    }

    auto& cache = game_file_stat_cache();
    bool need_clear = false;
    {
        std::shared_lock<std::shared_mutex> rlock(cache.mutex);
        if (cache.generation != gen) {
            need_clear = true;
        } else {
            const auto it = cache.entries.find(key);
            if (it != cache.entries.end()) {
                if (it->second.err != ERROR_SUCCESS) {
                    return it->second.err;
                }
                if (size == nullptr) {
                    return ERROR_SUCCESS;
                }
                if (it->second.has_size) {
                    *size = it->second.size;
                    return ERROR_SUCCESS;
                }
            }
        }
    }
    if (need_clear) {
        std::unique_lock<std::shared_mutex> wlock(cache.mutex);
        if (cache.generation != gen) {
            cache.entries.clear();
            cache.generation = gen;
        }
    }

    // Fast path: 1-roundtrip HTTPFS Stat returns both existence and exact byte size from
    // io_worker's in-memory manifest without CreateFileA / GetFileSizeEx / CloseHandle probes.
    // Falls back cleanly when a mock responder or direct backend returns unsupported (-38 / -1).
    GameFileStatEntry entry;
    bool stat_resolved = false;
    if (HttpfsMailboxClient* client = get_httpfs_client();
        client != nullptr && httpfs_client_generation() == gen) {
        const int64_t st_size = client->stat(key);
        if (st_size >= 0) {
            entry.err = ERROR_SUCCESS;
            entry.has_size = true;
            entry.size = static_cast<uint64_t>(st_size);
            if (size != nullptr) {
                *size = entry.size;
            }
            stat_resolved = true;
        } else if (st_size == -2) {
            entry.err = ERROR_FILE_NOT_FOUND;
            stat_resolved = true;
        }
    }

    if (!stat_resolved) {
        const Handle h = CreateFileA(key.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, kNullHandle);
        if (h == INVALID_HANDLE_VALUE) {
            const uint32_t err = GetLastError();
            entry.err = err == ERROR_SUCCESS ? ERROR_FILE_NOT_FOUND : err;
        } else {
            entry.err = ERROR_SUCCESS;
            if (size != nullptr) {
                int64_t file_size = 0;
                const bool ok = GetFileSizeEx(h, &file_size);
                CloseHandle(h);
                if (ok) {
                    entry.has_size = true;
                    entry.size = file_size > 0 ? static_cast<uint64_t>(file_size) : 0;
                    *size = entry.size;
                } else {
                    *size = 0;
                }
            } else {
                CloseHandle(h);
            }
        }
    }

    if (httpfs_client_generation() == gen) {
        std::unique_lock<std::shared_mutex> wlock(cache.mutex);
        if (cache.generation != gen) {
            cache.entries.clear();
            cache.generation = gen;
        }
        auto it = cache.entries.find(key);
        if (it == cache.entries.end()) {
            cache.entries.emplace(key, entry);
        } else if (entry.err != ERROR_SUCCESS) {
            it->second = entry;
        } else {
            it->second.err = ERROR_SUCCESS;
            if (entry.has_size) {
                it->second.has_size = true;
                it->second.size = entry.size;
            }
        }
    }
    return entry.err;
}

// `need_size` false leaves q->size 0 for game data (attribute/existence queries).
uint32_t query_resolved(const ResolvedPath& rp, Query* q, bool need_size = true) {
    *q = Query{};
    if (rp.is_userdata) {
        UserdataEntry e;
        const uint32_t err = userdata_stat(rp.key, &e);
        if (err != ERROR_SUCCESS) {
            return err;
        }
        *q = query_from_entry(e);
        return ERROR_SUCCESS;
    }
    q->created = q->accessed = q->written = kGameDataTicks;
    if (rp.is_root) {
        q->is_directory = true;
        q->attributes = FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_READONLY;
        return ERROR_SUCCESS;
    }
    uint64_t size = 0;
    const uint32_t err = probe_game_file(rp.key, need_size ? &size : nullptr);
    if (err != ERROR_SUCCESS) {
        return err;
    }
    q->size = size;
    q->attributes = FILE_ATTRIBUTE_READONLY;
    return ERROR_SUCCESS;
}

// DOS-style wildcard match: '*' any run, '?' one character; case-insensitive; "*.*" matches all.
bool wildcard_match(std::u16string_view pattern, std::u16string_view name) {
    if (pattern == u"*.*") {
        pattern = u"*";
    }
    size_t p = 0, n = 0, star = std::u16string_view::npos, mark = 0;
    while (n < name.size()) {
        if (p < pattern.size() && (pattern[p] == u'?' || fold_char(pattern[p]) == fold_char(name[n]))) {
            ++p;
            ++n;
        } else if (p < pattern.size() && pattern[p] == u'*') {
            star = p++;
            mark = n;
        } else if (star != std::u16string_view::npos) {
            p = star + 1;
            n = ++mark;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == u'*') {
        ++p;
    }
    return p == pattern.size();
}

struct FindEntry {
    std::string name;  // UTF-8
    Query query;
};

struct FindState {
    std::mutex mutex;
    std::vector<FindEntry> entries;
    size_t next = 0;
};

void fill_find_data(const FindEntry& entry, Pg6Win32FindDataW* out) {
    *out = Pg6Win32FindDataW{};
    out->file_attributes = entry.query.attributes;
    out->creation_time = file_time_from_ticks(entry.query.created);
    out->last_access_time = file_time_from_ticks(entry.query.accessed);
    out->last_write_time = file_time_from_ticks(entry.query.written);
    out->file_size_high = static_cast<uint32_t>(entry.query.size >> 32);
    out->file_size_low = static_cast<uint32_t>(entry.query.size & 0xFFFFFFFFull);
    const std::u16string name = compat_detail::utf8_to_utf16(entry.name);
    const size_t n = std::min<size_t>(name.size(), 259);
    std::memcpy(out->file_name, name.data(), n * sizeof(char16_t));
    out->file_name[n] = 0;
}

bool has_wildcard(std::u16string_view s) {
    return s.find_first_of(u"*?") != std::u16string_view::npos;
}

std::string last_component(const std::string& key) {
    const size_t slash = key.rfind('/');
    return slash == std::string::npos ? key : key.substr(slash + 1);
}

bool is_null_or_empty(const char16_t* s) {
    return s == nullptr || s[0] == 0;
}

// Returns the number of whole 100 ns ticks as a FILETIME, validating the range.
bool valid_file_time(const Pg6FileTime* ft) {
    return ft != nullptr && file_time_ticks(*ft) <= kFileTimeMax;
}

}  // namespace

}  // namespace playgta6::os::win32

namespace win32 = playgta6::os::win32;
namespace cd = playgta6::os::win32::compat_detail;
using playgta6::os::win32::UserdataEntry;

extern "C" {

int32_t pg6_win32_create_directory_w(const char16_t* path_name, void*) {
    cd::ResolvedPath rp;
    uint32_t err = cd::resolve_path(path_name, &rp);
    if (err == win32::ERROR_SUCCESS) {
        if (!rp.is_userdata) {
            err = rp.is_root ? win32::ERROR_ALREADY_EXISTS : win32::ERROR_ACCESS_DENIED;
        } else {
            err = win32::userdata_create_directory(rp.key);
        }
    }
    win32::SetLastError(err);
    return err == win32::ERROR_SUCCESS ? 1 : 0;
}

int32_t pg6_win32_remove_directory_w(const char16_t* path_name) {
    cd::ResolvedPath rp;
    uint32_t err = cd::resolve_path(path_name, &rp);
    if (err == win32::ERROR_SUCCESS) {
        err = rp.is_userdata ? win32::userdata_remove_directory(rp.key)
                             : win32::ERROR_ACCESS_DENIED;
    }
    win32::SetLastError(err);
    return err == win32::ERROR_SUCCESS ? 1 : 0;
}

int32_t pg6_win32_delete_file_w(const char16_t* file_name) {
    cd::ResolvedPath rp;
    uint32_t err = cd::resolve_path(file_name, &rp);
    if (err == win32::ERROR_SUCCESS) {
        if (rp.is_userdata) {
            err = win32::userdata_delete_file(rp.key);
        } else {
            win32::Query q;
            const uint32_t qerr = win32::query_resolved(rp, &q, false);
            err = qerr == win32::ERROR_SUCCESS ? win32::ERROR_ACCESS_DENIED : qerr;
        }
    }
    win32::SetLastError(err);
    return err == win32::ERROR_SUCCESS ? 1 : 0;
}

uint32_t pg6_win32_get_file_attributes_w(const char16_t* file_name) {
    cd::ResolvedPath rp;
    uint32_t err = cd::resolve_path(file_name, &rp);
    win32::Query q;
    if (err == win32::ERROR_SUCCESS) {
        err = win32::query_resolved(rp, &q, false);
    }
    if (err != win32::ERROR_SUCCESS) {
        win32::SetLastError(err);
        return win32::INVALID_FILE_ATTRIBUTES;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return q.attributes;
}

uint32_t pg6_win32_get_file_attributes_a(const char* file_name) {
    if (file_name == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return win32::INVALID_FILE_ATTRIBUTES;
    }
    cd::ResolvedPath rp;
    uint32_t err = cd::resolve_path_utf8(file_name, &rp);
    win32::Query q;
    if (err == win32::ERROR_SUCCESS) {
        err = win32::query_resolved(rp, &q, false);
    }
    if (err != win32::ERROR_SUCCESS) {
        win32::SetLastError(err);
        return win32::INVALID_FILE_ATTRIBUTES;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return q.attributes;
}

int32_t pg6_win32_get_file_attributes_ex_w(const char16_t* file_name,
                                           int32_t info_level_id,
                                           void* file_information) {
    if (file_information == nullptr || info_level_id != win32::GetFileExInfoStandard) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    auto* out = static_cast<Pg6Win32FileAttributeData*>(file_information);
    *out = Pg6Win32FileAttributeData{};
    cd::ResolvedPath rp;
    uint32_t err = cd::resolve_path(file_name, &rp);
    win32::Query q;
    if (err == win32::ERROR_SUCCESS) {
        err = win32::query_resolved(rp, &q);
    }
    if (err != win32::ERROR_SUCCESS) {
        win32::SetLastError(err);
        return 0;
    }
    out->file_attributes = q.attributes;
    out->creation_time = win32::file_time_from_ticks(q.created);
    out->last_access_time = win32::file_time_from_ticks(q.accessed);
    out->last_write_time = win32::file_time_from_ticks(q.written);
    out->file_size_high = static_cast<uint32_t>(q.size >> 32);
    out->file_size_low = static_cast<uint32_t>(q.size & 0xFFFFFFFFull);
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_set_file_attributes_w(const char16_t* file_name, uint32_t file_attributes) {
    cd::ResolvedPath rp;
    uint32_t err = cd::resolve_path(file_name, &rp);
    if (err == win32::ERROR_SUCCESS) {
        if (rp.is_userdata) {
            err = win32::userdata_set_attributes(rp.key, file_attributes);
        } else {
            win32::Query q;
            const uint32_t qerr = win32::query_resolved(rp, &q, false);
            err = qerr == win32::ERROR_SUCCESS ? win32::ERROR_ACCESS_DENIED : qerr;
        }
    }
    win32::SetLastError(err);
    return err == win32::ERROR_SUCCESS ? 1 : 0;
}

Pg6Handle pg6_win32_find_first_file_w(const char16_t* file_name, Pg6Win32FindDataW* find_data) {
    if (find_data != nullptr) {
        *find_data = Pg6Win32FindDataW{};
    }
    if (find_data == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return win32::INVALID_HANDLE_VALUE;
    }
    if (win32::is_null_or_empty(file_name)) {
        win32::SetLastError(file_name == nullptr ? win32::ERROR_INVALID_PARAMETER
                                                 : win32::ERROR_PATH_NOT_FOUND);
        return win32::INVALID_HANDLE_VALUE;
    }

    // Split "dir\pattern" at the last separator.
    std::u16string_view full(file_name);
    const size_t sep = full.find_last_of(u"\\/");
    const std::u16string_view dir_part =
        sep == std::u16string_view::npos ? std::u16string_view() : full.substr(0, sep);
    const std::u16string_view pattern =
        sep == std::u16string_view::npos ? full : full.substr(sep + 1);
    if (pattern.empty()) {
        win32::SetLastError(win32::ERROR_FILE_NOT_FOUND);
        return win32::INVALID_HANDLE_VALUE;
    }

    auto state = std::make_shared<win32::FindState>();

    if (!win32::has_wildcard(pattern)) {
        // Exact name: one entry (file or directory) or not found.
        cd::ResolvedPath rp;
        uint32_t err = cd::resolve_path(file_name, &rp);
        win32::Query q;
        if (err == win32::ERROR_SUCCESS) {
            err = win32::query_resolved(rp, &q);
        }
        if (err != win32::ERROR_SUCCESS) {
            win32::SetLastError(err);
            return win32::INVALID_HANDLE_VALUE;
        }
        win32::FindEntry entry;
        entry.name = rp.is_root ? std::string() : win32::last_component(rp.key);
        entry.query = q;
        state->entries.push_back(std::move(entry));
    } else {
        // Wildcard: enumerate the directory (only /userdata can be enumerated).
        std::u16string dir_text(dir_part);
        if (dir_text.empty()) {
            dir_text = u".";
        }
        cd::ResolvedPath rp;
        uint32_t err = cd::resolve_path(dir_text.c_str(), &rp);
        if (err == win32::ERROR_SUCCESS && !rp.is_userdata) {
            err = win32::ERROR_PATH_NOT_FOUND;  // the HTTPFS protocol has no directory listing
        }
        std::vector<UserdataEntry> listing;
        if (err == win32::ERROR_SUCCESS) {
            err = win32::userdata_list_directory(rp.key, &listing);
        }
        if (err != win32::ERROR_SUCCESS) {
            win32::SetLastError(err == win32::ERROR_FILE_NOT_FOUND ? win32::ERROR_PATH_NOT_FOUND
                                                                    : err);
            return win32::INVALID_HANDLE_VALUE;
        }
        if (rp.key != win32::kUserdataRoot) {
            UserdataEntry self;
            if (win32::userdata_stat(rp.key, &self) == win32::ERROR_SUCCESS) {
                for (const char* dots : {".", ".."}) {
                    if (win32::wildcard_match(pattern, cd::utf8_to_utf16(dots))) {
                        win32::FindEntry entry;
                        entry.name = dots;
                        entry.query = win32::query_from_entry(self);
                        state->entries.push_back(std::move(entry));
                    }
                }
            }
        }
        for (const UserdataEntry& e : listing) {
            if (win32::wildcard_match(pattern, cd::utf8_to_utf16(e.name))) {
                win32::FindEntry entry;
                entry.name = e.name;
                entry.query = win32::query_from_entry(e);
                state->entries.push_back(std::move(entry));
            }
        }
        if (state->entries.empty()) {
            win32::SetLastError(win32::ERROR_FILE_NOT_FOUND);
            return win32::INVALID_HANDLE_VALUE;
        }
    }

    win32::fill_find_data(state->entries[0], find_data);
    state->next = 1;
    win32::SetLastError(win32::ERROR_SUCCESS);
    return win32::insert_opaque_object(win32::kTagFind, std::move(state));
}

int32_t pg6_win32_find_next_file_w(Pg6Handle find_file, Pg6Win32FindDataW* find_data) {
    if (find_data == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    auto state = std::static_pointer_cast<win32::FindState>(
        win32::lookup_opaque_object(find_file, win32::kTagFind));
    if (!state) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->next >= state->entries.size()) {
        win32::SetLastError(win32::ERROR_NO_MORE_FILES);
        return 0;
    }
    win32::fill_find_data(state->entries[state->next++], find_data);
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_find_close(Pg6Handle find_file) {
    if (!win32::remove_opaque_object(find_file, win32::kTagFind)) {
        win32::SetLastError(win32::ERROR_INVALID_HANDLE);
        return 0;
    }
    return 1;
}

int32_t pg6_win32_get_file_time(Pg6Handle file,
                                Pg6FileTime* creation_time,
                                Pg6FileTime* last_access_time,
                                Pg6FileTime* last_write_time) {
    for (Pg6FileTime* ft : {creation_time, last_access_time, last_write_time}) {
        if (ft != nullptr) {
            *ft = Pg6FileTime{};
        }
    }
    win32::FileHandleInfo info;
    if (!win32::query_file_handle(file, &info)) {
        return 0;  // last error set by the seam
    }
    uint64_t created = win32::kGameDataTicks, accessed = created, written = created;
    if (info.is_userdata) {
        UserdataEntry e;
        const uint32_t err = win32::userdata_stat(info.userdata_key, &e);
        if (err != win32::ERROR_SUCCESS) {
            win32::SetLastError(err);
            return 0;
        }
        created = e.created;
        accessed = e.accessed;
        written = e.written;
    }
    if (creation_time != nullptr) {
        *creation_time = win32::file_time_from_ticks(created);
    }
    if (last_access_time != nullptr) {
        *last_access_time = win32::file_time_from_ticks(accessed);
    }
    if (last_write_time != nullptr) {
        *last_write_time = win32::file_time_from_ticks(written);
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_set_file_time(Pg6Handle file,
                                const Pg6FileTime* creation_time,
                                const Pg6FileTime* last_access_time,
                                const Pg6FileTime* last_write_time) {
    win32::FileHandleInfo info;
    if (!win32::query_file_handle(file, &info)) {
        return 0;
    }
    if (!info.is_userdata || !win32::has_write_access(info.desired_access)) {
        win32::SetLastError(win32::ERROR_ACCESS_DENIED);
        return 0;
    }
    uint64_t values[3] = {0, 0, 0};
    const uint64_t* pointers[3] = {nullptr, nullptr, nullptr};
    const Pg6FileTime* inputs[3] = {creation_time, last_access_time, last_write_time};
    for (int i = 0; i < 3; ++i) {
        if (inputs[i] == nullptr) {
            continue;
        }
        const uint64_t ticks = win32::file_time_ticks(*inputs[i]);
        if (ticks > win32::kFileTimeMax && ticks != ~uint64_t{0}) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return 0;
        }
        // 0 means "no change"; -1 ("freeze updates") is accepted and ignored.
        if (ticks != 0 && ticks != ~uint64_t{0}) {
            values[i] = ticks;
            pointers[i] = &values[i];
        }
    }
    const uint32_t err =
        win32::userdata_set_times(info.userdata_key, pointers[0], pointers[1], pointers[2]);
    win32::SetLastError(err);
    return err == win32::ERROR_SUCCESS ? 1 : 0;
}

// The time zone is UTC, so local time and file time are the same instant.
int32_t pg6_win32_file_time_to_local_file_time(const Pg6FileTime* file_time,
                                               Pg6FileTime* local_file_time) {
    if (local_file_time != nullptr) {
        *local_file_time = Pg6FileTime{};
    }
    if (local_file_time == nullptr || !win32::valid_file_time(file_time)) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    *local_file_time = *file_time;
    return 1;
}

int32_t pg6_win32_local_file_time_to_file_time(const Pg6FileTime* local_file_time,
                                               Pg6FileTime* file_time) {
    return pg6_win32_file_time_to_local_file_time(local_file_time, file_time);
}

int32_t pg6_win32_flush_file_buffers(Pg6Handle file) {
    win32::FileHandleInfo info;
    if (!win32::query_file_handle(file, &info)) {
        return 0;
    }
    if (!win32::has_write_access(info.desired_access)) {
        win32::SetLastError(win32::ERROR_ACCESS_DENIED);
        return 0;
    }
    return 1;  // the overlay is memory; nothing to flush
}

int32_t pg6_win32_set_end_of_file(Pg6Handle file) {
    return win32::set_end_of_file_at_cursor(file) ? 1 : 0;
}

int32_t pg6_win32_get_disk_free_space_ex_w(const char16_t* directory_name,
                                           uint64_t* free_bytes_available_to_caller,
                                           uint64_t* total_number_of_bytes,
                                           uint64_t* total_number_of_free_bytes) {
    for (uint64_t* out : {free_bytes_available_to_caller, total_number_of_bytes,
                          total_number_of_free_bytes}) {
        if (out != nullptr) {
            *out = 0;
        }
    }
    bool userdata_volume = true;
    if (!win32::is_null_or_empty(directory_name)) {
        cd::ResolvedPath rp;
        uint32_t err = cd::resolve_path(directory_name, &rp);
        win32::Query q;
        if (err == win32::ERROR_SUCCESS) {
            err = win32::query_resolved(rp, &q, false);
        }
        if (err != win32::ERROR_SUCCESS) {
            win32::SetLastError(err == win32::ERROR_FILE_NOT_FOUND ? win32::ERROR_PATH_NOT_FOUND : err);
            return 0;
        }
        userdata_volume = rp.is_userdata;
    }
    const uint64_t used = win32::userdata_total_bytes();
    const uint64_t free_bytes =
        !userdata_volume ? 0
                         : (used >= win32::kUserdataQuotaBytes ? 0
                                                               : win32::kUserdataQuotaBytes - used);
    if (free_bytes_available_to_caller != nullptr) {
        *free_bytes_available_to_caller = free_bytes;
    }
    if (total_number_of_bytes != nullptr) {
        *total_number_of_bytes = win32::kUserdataQuotaBytes;
    }
    if (total_number_of_free_bytes != nullptr) {
        *total_number_of_free_bytes = free_bytes;
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

int32_t pg6_win32_move_file_ex_w(const char16_t* existing_file_name,
                                 const char16_t* new_file_name,
                                 uint32_t flags) {
    cd::ResolvedPath from, to;
    uint32_t err = cd::resolve_path(existing_file_name, &from);
    if (err == win32::ERROR_SUCCESS) {
        if (new_file_name == nullptr) {
            err = win32::ERROR_INVALID_PARAMETER;  // delete-at-reboot is not supported
        } else {
            err = cd::resolve_path(new_file_name, &to);
        }
    }
    if (err == win32::ERROR_SUCCESS) {
        if (!from.is_userdata) {
            win32::Query q;
            const uint32_t qerr = win32::query_resolved(from, &q, false);
            err = qerr == win32::ERROR_SUCCESS ? win32::ERROR_ACCESS_DENIED : qerr;
        } else if (!to.is_userdata) {
            err = win32::ERROR_ACCESS_DENIED;
        } else {
            err = win32::userdata_rename(from.key, to.key,
                                         (flags & win32::MOVEFILE_REPLACE_EXISTING) != 0);
        }
    }
    win32::SetLastError(err);
    return err == win32::ERROR_SUCCESS ? 1 : 0;
}

}  // extern "C"

namespace playgta6::os::win32::compat_detail {

uint64_t now_ticks() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const int64_t unix_100ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() / 100;
    return kFileTimeUnixEpochTicks + static_cast<uint64_t>(unix_100ns > 0 ? unix_100ns : 0);
}

uint32_t resolve_path_utf8(std::string_view path, ResolvedPath* out) {
    *out = ResolvedPath{};
    if (path.empty()) {
        return ERROR_PATH_NOT_FOUND;
    }
    bool is_userdata = false;
    std::string key = normalize_path(path, &is_userdata);
    if (key.empty()) {
        // normalize_path rejects the bare "userdata" directory (it only names files below it);
        // probe with a child to tell the /userdata root from the (empty) game-data root.
        std::string probe(path);
        probe.append("/x");
        bool child_userdata = false;
        if (normalize_path(probe, &child_userdata) == "/userdata/x" && child_userdata) {
            out->key = "/userdata";
            out->is_userdata = true;
        } else {
            out->is_root = true;
        }
        return ERROR_SUCCESS;
    }
    out->key = std::move(key);
    out->is_userdata = is_userdata;
    return ERROR_SUCCESS;
}

uint32_t resolve_path(const char16_t* path, ResolvedPath* out) {
    *out = ResolvedPath{};
    if (path == nullptr) {
        return ERROR_INVALID_PARAMETER;
    }
    return resolve_path_utf8(utf16_to_utf8(path), out);
}

}  // namespace playgta6::os::win32::compat_detail
