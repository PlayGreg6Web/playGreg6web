#include "os/xgamesave_local.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace playgta6::os::gdk::gamesave {

namespace fs = std::filesystem;

namespace {

std::mutex g_mutex;  // one operation at a time: updates are short and rare
std::string g_save_dir;  // "" = default_save_dir()
std::set<std::string> g_changed;  // paths changed since the host last took them (browser persistence)
std::atomic<uint64_t> g_dirty_generation{0};
std::atomic<uint64_t> g_temp_counter{0};
std::atomic<int> g_crash_point{0};

// Flushes a file's data to stable storage (best effort; false when it could not be opened).
bool sync_path(const fs::path& p, bool directory) {
#if defined(_WIN32)
    if (directory) {
        return true;  // NTFS journals renames; directories cannot be flushed through the Win32 file API
    }
    HANDLE h = CreateFileW(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    const bool ok = FlushFileBuffers(h) != 0;
    CloseHandle(h);
    return ok;
#else
    const int fd = ::open(p.c_str(), directory ? (O_RDONLY | O_DIRECTORY) : O_RDONLY);
    if (fd < 0) {
        return false;
    }
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
#endif
}

void mark_changed(const fs::path& p) {
    g_changed.insert(p.generic_string());
    g_dirty_generation.fetch_add(1);
}

constexpr size_t kMaxEncodedChars = 240;  // keeps "c-"/"b-" + name inside a path component

bool plain_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
}

// [a-z0-9_-] stay; an upper-case letter becomes '^' + lower-case; the rest %XX.
std::string encode(const std::string& name) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : name) {
        if (plain_char(c)) {
            out.push_back(static_cast<char>(c));
        } else if (c >= 'A' && c <= 'Z') {
            out.push_back('^');
            out.push_back(static_cast<char>(c - 'A' + 'a'));
        } else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 15]);
        }
    }
    return out;
}

bool decode(const std::string& text, std::string* out) {
    auto nib = [](char h) { return h >= '0' && h <= '9' ? h - '0' : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1; };
    out->clear();
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '^' && i + 1 < text.size() && text[i + 1] >= 'a' && text[i + 1] <= 'z') {
            out->push_back(static_cast<char>(text[++i] - 'a' + 'A'));
        } else if (c == '%' && i + 2 < text.size()) {
            const int hi = nib(text[i + 1]);
            const int lo = nib(text[i + 2]);
            if (hi < 0 || lo < 0) {
                return false;
            }
            out->push_back(static_cast<char>(hi * 16 + lo));
            i += 2;
        } else if (plain_char(static_cast<unsigned char>(c))) {
            out->push_back(c);
        } else {
            return false;
        }
    }
    return true;
}

bool valid_name(const std::string& name, size_t max_chars) {
    if (name.empty() || name.size() > max_chars) {
        return false;
    }
    for (unsigned char c : name) {
        if (c < 0x20 || c == 0x7f) {
            return false;
        }
    }
    return encode(name).size() <= kMaxEncodedChars;
}

constexpr const char* kDisplayNameFile = ".display-name";

std::string read_display_name(const fs::path& container) {
    std::ifstream in(container / kDisplayNameFile, std::ios::binary);
    std::string text;
    if (in) {
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    return text.size() > kMaxDisplayNameChars ? std::string() : text;
}

fs::path container_path(const std::string& root, const std::string& name) {
    return fs::path(root) / ("c-" + encode(name));
}
fs::path blob_path(const fs::path& container, const std::string& name) {
    return container / ("b-" + encode(name));
}

// Blob files of a container: name -> size. Temp files and foreign files are skipped.
std::map<std::string, uint64_t> list_blobs(const fs::path& container, uint64_t* newest_ticks = nullptr) {
    std::map<std::string, uint64_t> out;
    std::error_code ec;
    for (fs::directory_iterator it(container, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string file = it->path().filename().string();
        std::string name;
        if (file.compare(0, 2, "b-") != 0 || !decode(file.substr(2), &name) || !it->is_regular_file(ec)) {
            continue;
        }
        out[name] = static_cast<uint64_t>(it->file_size(ec));
        if (newest_ticks != nullptr) {
            const auto ft = it->last_write_time(ec);
            const auto sys = std::chrono::system_clock::now() +
                             std::chrono::duration_cast<std::chrono::system_clock::duration>(
                                 ft - fs::file_time_type::clock::now());
            const auto secs = std::chrono::duration_cast<std::chrono::seconds>(sys.time_since_epoch()).count();
            const uint64_t ticks = secs > 0 ? (static_cast<uint64_t>(secs) + 11644473600ull) * 10000000ull : 0;
            *newest_ticks = std::max(*newest_ticks, ticks);
        }
    }
    return out;
}

uint64_t used_bytes(const std::string& root) {
    uint64_t total = 0;
    std::error_code ec;
    for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_directory(ec) && it->path().filename().string().compare(0, 2, "c-") == 0) {
            for (const auto& entry : list_blobs(it->path())) {
                total += entry.second;
            }
        }
    }
    return total;
}

void remove_stale_temps(const fs::path& container) {
    std::error_code ec;
    for (fs::directory_iterator it(container, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().filename().string().compare(0, 5, ".tmp-") == 0) {
            fs::remove(it->path(), ec);
        }
    }
}

// Finishes or rolls back a swap that a crash interrupted: ".old-<c-dir>-N" is the previous
// container (restored when the container is missing), ".new-<c-dir>-N" an unfinished copy.
void recover_swaps(const std::string& root) {
    std::error_code ec;
    std::vector<fs::path> olds;
    std::vector<fs::path> news;
    for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string file = it->path().filename().string();
        if (file.compare(0, 5, ".old-") == 0) {
            olds.push_back(it->path());
        } else if (file.compare(0, 5, ".new-") == 0) {
            news.push_back(it->path());
        }
    }
    for (const fs::path& p : olds) {
        std::string target = p.filename().string().substr(5);
        target = target.substr(0, target.rfind('-'));
        const fs::path dest = fs::path(root) / target;
        if (fs::exists(dest, ec)) {
            fs::remove_all(p, ec);
        } else {
            fs::rename(p, dest, ec);
        }
    }
    for (const fs::path& p : news) {
        fs::remove_all(p, ec);
    }
}

}  // namespace

void Store::set_test_crash_point(int point) {
    g_crash_point.store(point);
}

bool valid_container_name(const std::string& name) {
    return valid_name(name, kMaxContainerNameChars);
}

bool valid_blob_name(const std::string& name) {
    return valid_name(name, kMaxBlobNameChars);
}

void set_save_dir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_save_dir = dir;
}

std::string default_save_dir() {
#if defined(__EMSCRIPTEN__)
    return "pg6-saves";
#else
    std::string base;
#if defined(_WIN32)
    if (const char* v = std::getenv("LOCALAPPDATA")) base = v;
#elif defined(__APPLE__)
    if (const char* v = std::getenv("HOME")) base = std::string(v) + "/Library/Application Support";
#else
    if (const char* v = std::getenv("XDG_DATA_HOME")) {
        base = v;
    } else if (const char* h = std::getenv("HOME")) {
        base = std::string(h) + "/.local/share";
    }
#endif
    if (base.empty()) {
        return "pg6-saves";
    }
    return (fs::path(base) / "playgta6web" / "saves").string();
#endif
}

std::string save_dir() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_save_dir.empty() ? default_save_dir() : g_save_dir;
}

void flush_all() {
    std::lock_guard<std::mutex> lock(g_mutex);
    const fs::path root = g_save_dir.empty() ? fs::path(default_save_dir()) : fs::path(g_save_dir);
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        const bool dir = it->is_directory(ec);
        if (dir || it->is_regular_file(ec)) {
            sync_path(it->path(), dir);
        }
    }
}

bool storage_dirty() { return g_dirty_generation.load() != 0; }
uint64_t storage_generation() { return g_dirty_generation.load(); }

std::vector<std::string> take_changed_paths() {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<std::string> out(g_changed.begin(), g_changed.end());
    g_changed.clear();
    return out;
}

Store::Store(const std::string& root, uint64_t quota_bytes) : root_(root), quota_(quota_bytes) {
    std::error_code ec;
    fs::create_directories(root_, ec);
    std::lock_guard<std::mutex> lock(g_mutex);
    recover_swaps(root_);
}

HResult Store::remaining_quota(int64_t* out) const {
    std::lock_guard<std::mutex> lock(g_mutex);
    const uint64_t used = used_bytes(root_);
    *out = used >= quota_ ? 0 : static_cast<int64_t>(quota_ - used);
    return S_OK_HR;
}

HResult Store::create_container(const std::string& name) {
    if (!valid_name(name, kMaxContainerNameChars)) {
        return E_GS_INVALID_CONTAINER_NAME;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const fs::path path = container_path(root_, name);
    std::error_code ec;
    fs::create_directories(path, ec);
    if (ec) {
        return E_FAIL_HR;
    }
    remove_stale_temps(path);
    mark_changed(path);
    return S_OK_HR;
}

HResult Store::delete_container(const std::string& name) {
    if (!valid_name(name, kMaxContainerNameChars)) {
        return E_GS_INVALID_CONTAINER_NAME;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const fs::path path = container_path(root_, name);
    std::error_code ec;
    if (!fs::is_directory(path, ec)) {
        return S_OK_HR;  // deleting a container that is not there succeeds
    }
    fs::remove_all(path, ec);
    if (!ec) {
        mark_changed(path);
    }
    return ec ? E_FAIL_HR : S_OK_HR;
}

HResult Store::enumerate_containers(const std::string& prefix, std::vector<ContainerInfo>* out) const {
    out->clear();
    std::lock_guard<std::mutex> lock(g_mutex);
    std::error_code ec;
    for (fs::directory_iterator it(root_, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string file = it->path().filename().string();
        ContainerInfo info;
        if (file.compare(0, 2, "c-") != 0 || !decode(file.substr(2), &info.name) || !it->is_directory(ec) ||
            info.name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        info.display_name = read_display_name(it->path());
        for (const auto& entry : list_blobs(it->path(), &info.last_modified)) {
            ++info.blob_count;
            info.total_size += entry.second;
        }
        out->push_back(std::move(info));
    }
    std::sort(out->begin(), out->end(), [](const ContainerInfo& a, const ContainerInfo& b) { return a.name < b.name; });
    return S_OK_HR;
}

HResult Store::get_container_info(const std::string& name, ContainerInfo* out) const {
    *out = ContainerInfo{};
    if (!valid_name(name, kMaxContainerNameChars)) {
        return E_GS_INVALID_CONTAINER_NAME;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const fs::path path = container_path(root_, name);
    std::error_code ec;
    if (!fs::is_directory(path, ec)) {
        return E_GS_NO_SUCH_CONTAINER;
    }
    out->name = name;
    out->display_name = read_display_name(path);
    for (const auto& entry : list_blobs(path, &out->last_modified)) {
        ++out->blob_count;
        out->total_size += entry.second;
    }
    return S_OK_HR;
}

HResult Store::enumerate_blobs(const std::string& container, std::vector<BlobInfo>* out) const {
    out->clear();
    if (!valid_name(container, kMaxContainerNameChars)) {
        return E_GS_INVALID_CONTAINER_NAME;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const fs::path path = container_path(root_, container);
    std::error_code ec;
    if (!fs::is_directory(path, ec)) {
        return E_GS_NO_SUCH_CONTAINER;
    }
    for (const auto& entry : list_blobs(path)) {
        out->push_back({entry.first, static_cast<uint32_t>(entry.second)});
    }
    return S_OK_HR;
}

HResult Store::read_blob(const std::string& container, const std::string& blob, std::vector<uint8_t>* out) const {
    out->clear();
    if (!valid_name(container, kMaxContainerNameChars)) {
        return E_GS_INVALID_CONTAINER_NAME;
    }
    if (!valid_name(blob, kMaxBlobNameChars)) {
        return E_INVALIDARG_HR;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const fs::path dir = container_path(root_, container);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return E_GS_NO_SUCH_CONTAINER;
    }
    std::ifstream in(blob_path(dir, blob), std::ios::binary);
    if (!in) {
        return E_GS_BLOB_NOT_FOUND;
    }
    out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return S_OK_HR;
}

HResult Store::submit_update(const std::string& container, const std::vector<BlobWrite>& writes,
                             const std::vector<std::string>& deletes) {
    if (!valid_name(container, kMaxContainerNameChars)) {
        return E_GS_INVALID_CONTAINER_NAME;
    }
    for (const BlobWrite& w : writes) {
        if (!valid_name(w.name, kMaxBlobNameChars)) {
            return E_INVALIDARG_HR;
        }
    }
    uint64_t update_bytes = 0;
    for (const BlobWrite& w : writes) {
        update_bytes += w.data.size();
    }
    if (update_bytes > kMaxUpdateBytes) {
        return E_GS_UPDATE_TOO_BIG;
    }
    for (const std::string& d : deletes) {
        if (!valid_name(d, kMaxBlobNameChars)) {
            return E_INVALIDARG_HR;
        }
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const fs::path dir = container_path(root_, container);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return E_GS_NO_SUCH_CONTAINER;
    }
    // Quota: bytes after the update (writes apply first, then deletes).
    std::map<std::string, uint64_t> sizes = list_blobs(dir);
    uint64_t total = used_bytes(root_);
    auto drop = [&](const std::string& name) {
        auto it = sizes.find(name);
        if (it != sizes.end()) {
            total -= it->second;
            sizes.erase(it);
        }
    };
    for (const BlobWrite& w : writes) {
        drop(w.name);
        sizes[w.name] = w.data.size();
        total += w.data.size();
    }
    for (const std::string& d : deletes) {
        drop(d);
    }
    if (total > quota_) {
        return E_GS_QUOTA_EXCEEDED;
    }
    // Build the complete new container next to the old one: existing files are hard-linked (or
    // copied), written blobs are fresh files, deleted blobs are left out.
    const std::string base = container_path(root_, container).filename().string();
    const std::string tag = "-" + std::to_string(g_temp_counter.fetch_add(1));
    const fs::path fresh = fs::path(root_) / (".new-" + base + tag);
    const fs::path aside = fs::path(root_) / (".old-" + base + tag);
    auto fail = [&]() {
        fs::remove_all(fresh, ec);
        return E_FAIL_HR;
    };
    fs::remove_all(fresh, ec);
    if (!fs::create_directory(fresh, ec) || ec) {
        return fail();
    }
    std::map<std::string, bool> replaced;  // file names of blobs written or deleted
    for (const BlobWrite& w : writes) {
        replaced[blob_path(dir, w.name).filename().string()] = true;
    }
    for (const std::string& d : deletes) {
        replaced[blob_path(dir, d).filename().string()] = true;
    }
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string file = it->path().filename().string();
        if (file.compare(0, 5, ".tmp-") == 0 || replaced.count(file) != 0 || !it->is_regular_file(ec)) {
            continue;
        }
        std::error_code link_ec;
        fs::create_hard_link(it->path(), fresh / file, link_ec);
        if (link_ec) {
            fs::copy_file(it->path(), fresh / file, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                return fail();
            }
        }
    }
    for (const BlobWrite& w : writes) {
        bool ok = false;
        {
            std::ofstream out(blob_path(fresh, w.name), std::ios::binary | std::ios::trunc);
            if (out) {
                out.write(reinterpret_cast<const char*>(w.data.data()), static_cast<std::streamsize>(w.data.size()));
                out.flush();
                ok = static_cast<bool>(out);
            }
        }
        ok = ok && sync_path(blob_path(fresh, w.name), false);
        if (!ok) {
            return fail();
        }
    }
    // Writes apply first, then deletes (a delete of a written name wins).
    for (const std::string& d : deletes) {
        fs::remove(blob_path(fresh, d), ec);
    }
    if (g_crash_point.load() == 1) {
        return E_FAIL_HR;
    }
    // Commit: old aside, new in, old removed.
    fs::rename(dir, aside, ec);
    if (ec) {
        return fail();
    }
    if (g_crash_point.load() == 2) {
        return E_FAIL_HR;
    }
    fs::rename(fresh, dir, ec);
    if (ec) {
        std::error_code undo;
        fs::rename(aside, dir, undo);
        return fail();
    }
    sync_path(dir, true);
    sync_path(fs::path(root_), true);
    fs::remove_all(aside, ec);
    mark_changed(dir);
    return S_OK_HR;
}

HResult Store::set_display_name(const std::string& container, const std::string& display_name) {
    if (!valid_name(container, kMaxContainerNameChars)) {
        return E_GS_INVALID_CONTAINER_NAME;
    }
    if (display_name.size() > kMaxDisplayNameChars) {
        return E_INVALIDARG_HR;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const fs::path dir = container_path(root_, container);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return E_GS_NO_SUCH_CONTAINER;
    }
    if (read_display_name(dir) == display_name) {
        return S_OK_HR;
    }
    const fs::path temp = dir / (".tmp-" + std::to_string(g_temp_counter.fetch_add(1)));
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        out.write(display_name.data(), static_cast<std::streamsize>(display_name.size()));
        out.flush();
        if (out) {
            out.close();
            sync_path(temp, false);
        }
        if (!out) {
            out.close();
            fs::remove(temp, ec);
            return E_FAIL_HR;
        }
    }
    fs::rename(temp, dir / kDisplayNameFile, ec);
    if (ec) {
        fs::remove(temp, ec);
        return E_FAIL_HR;
    }
    sync_path(dir / kDisplayNameFile, false);
    mark_changed(dir);
    return S_OK_HR;
}

Store open_store(const std::string& configuration_id) {
    const std::string root = save_dir();
    const std::string cfg = configuration_id.empty() ? "default" : encode(configuration_id);
    return Store((fs::path(root) / cfg / "local").string());
}

}  // namespace playgta6::os::gdk::gamesave
