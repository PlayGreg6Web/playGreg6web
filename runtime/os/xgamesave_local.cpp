#include "os/xgamesave_local.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>

namespace playgta6::os::gdk::gamesave {

namespace fs = std::filesystem;

namespace {

std::mutex g_mutex;  // one operation at a time: updates are short and rare
std::string g_save_dir = "pg6-saves";
std::atomic<uint64_t> g_temp_counter{0};

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

}  // namespace

void set_save_dir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_save_dir = dir.empty() ? "pg6-saves" : dir;
}

std::string save_dir() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_save_dir;
}

Store::Store(const std::string& root, uint64_t quota_bytes) : root_(root), quota_(quota_bytes) {
    std::error_code ec;
    fs::create_directories(root_, ec);
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
        for (const auto& entry : list_blobs(it->path(), &info.last_modified)) {
            ++info.blob_count;
            info.total_size += entry.second;
        }
        out->push_back(std::move(info));
    }
    std::sort(out->begin(), out->end(), [](const ContainerInfo& a, const ContainerInfo& b) { return a.name < b.name; });
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
        if (!valid_name(w.name, kMaxBlobNameChars) || w.data.size() > kMaxBlobBytes) {
            return E_INVALIDARG_HR;
        }
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
    // Stage every write; only when all are on disk do the renames start.
    std::vector<std::pair<fs::path, fs::path>> staged;  // temp -> blob
    auto abandon = [&]() {
        for (const auto& entry : staged) {
            fs::remove(entry.first, ec);
        }
    };
    for (const BlobWrite& w : writes) {
        const fs::path temp = dir / (".tmp-" + std::to_string(g_temp_counter.fetch_add(1)));
        staged.emplace_back(temp, blob_path(dir, w.name));
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (out) {
            out.write(reinterpret_cast<const char*>(w.data.data()), static_cast<std::streamsize>(w.data.size()));
            out.flush();
        }
        if (!out) {
            out.close();
            abandon();
            return E_FAIL_HR;
        }
    }
    for (size_t i = 0; i < staged.size(); ++i) {
        fs::rename(staged[i].first, staged[i].second, ec);
        if (ec) {
            for (size_t j = i; j < staged.size(); ++j) {
                fs::remove(staged[j].first, ec);
            }
            return E_FAIL_HR;
        }
    }
    for (const std::string& d : deletes) {
        fs::remove(blob_path(dir, d), ec);
    }
    return S_OK_HR;
}

Store open_store(const std::string& configuration_id) {
    const std::string root = save_dir();
    const std::string cfg = configuration_id.empty() ? "default" : encode(configuration_id);
    return Store((fs::path(root) / cfg / "local").string());
}

}  // namespace playgta6::os::gdk::gamesave
