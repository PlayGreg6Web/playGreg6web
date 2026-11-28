// Persistent on-disk cache behind Dawn's blob cache hooks (wgpu::DawnCacheDeviceDescriptor load /
// store callbacks). Dawn hands it the compiled shader and pipeline blobs it would otherwise rebuild
// with DXC and the driver on every run; a later run with the same title, Dawn build and adapter
// loads them instead, which removes the first-use stutter of lazily created pipelines.
//
// Layout: <root>/<title>/<dawn build>-<adapter key>/<hash>.blob, one file per Dawn key, written
// through a temporary file and a rename. A blob stores its key (a hash collision is a miss, never a
// wrong blob), its size and a checksum (a torn or damaged file is deleted and counts as a miss).
// The directory is size-capped: after a store the least recently used blobs (a load refreshes the
// file time, so recency survives restarts) are removed until the total fits.
//
// Header-only, no Dawn dependency (unit-tested without a GPU). Thread-safe: Dawn calls the hooks
// from its worker threads; the mutex guards the index only, never a file read or write. Holds no game bytes beyond what Dawn derives from the shaders the title
// creates, in the user's cache directory, never in a checkout.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace playgta6::gpu::native_exec {

struct BlobCacheStats {
    bool enabled = false;
    std::string dir;
    uint64_t hits = 0;           // full loads served from disk
    uint64_t misses = 0;         // lookups that found nothing usable
    uint64_t stores = 0;         // blobs written
    uint64_t store_rejected = 0; // empty or larger than the cap, or the write failed
    uint64_t bytes_loaded = 0;
    uint64_t bytes_stored = 0;
    uint64_t evictions = 0;
    uint64_t evicted_bytes = 0;
    uint64_t corrupt = 0;        // damaged files that were removed
    uint64_t entries = 0;
    uint64_t total_bytes = 0;    // blob bytes on disk (file sizes)
    uint64_t max_bytes = 0;
    uint64_t io_max_ns = 0;      // slowest single load / store file operation
    uint64_t io_slow = 0;        // load / store file operations that took over 1 s (a starved disk)
};

class BlobCache {
public:
    // "<LOCALAPPDATA>/playgta6/dawn-cache" (XDG cache / ~/.cache elsewhere); "" when no base exists.
    static std::string default_root() {
        std::string base;
#if defined(_WIN32)
        if (const char* v = std::getenv("LOCALAPPDATA")) base = v;
#else
        if (const char* v = std::getenv("XDG_CACHE_HOME")) {
            base = v;
        } else if (const char* h = std::getenv("HOME")) {
            base = std::string(h) + "/.cache";
        }
#endif
        if (base.empty()) return "";
        return (std::filesystem::path(base) / "playgta6" / "dawn-cache").string();
    }

    static std::string sanitize(const std::string& s) {
        std::string out;
        for (char c : s) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
            out += ok ? c : '_';
            if (out.size() >= 48) break;
        }
        return out.empty() ? "x" : out;
    }

    static uint64_t fnv1a(const void* data, size_t n, uint64_t seed = 1469598103934665603ull) {
        const auto* p = static_cast<const uint8_t*>(data);
        uint64_t h = seed;
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    // <root>/<title>/<dawn build>-<adapter key>. Dawn also mixes its own version and device
    // identity into every key; the directory split makes a stale generation removable as a whole.
    static std::string versioned_dir(const std::string& root, const std::string& title, const std::string& dawn_build,
                                     const std::string& adapter_key) {
        return (std::filesystem::path(root) / sanitize(title) / (sanitize(dawn_build) + "-" + sanitize(adapter_key))).string();
    }

    // Opens (creating) `dir` and indexes the blobs in it. False when the directory cannot be used;
    // the cache then stays disabled and every hook is a no-op miss. max_bytes 0 = unbounded.
    bool open(const std::string& dir, uint64_t max_bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        namespace fs = std::filesystem;
        index_.clear();
        total_ = 0;
        enabled_ = false;
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec || !fs::is_directory(dir, ec)) return false;
        dir_ = dir;
        max_bytes_ = max_bytes;
        struct Found {
            std::string name;
            uint64_t size;
            fs::file_time_type mtime;
        };
        std::vector<Found> found;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code e2;
            if (!it->is_regular_file(e2)) continue;
            const std::string name = it->path().filename().string();
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".blob") == 0) {
                found.push_back({name, static_cast<uint64_t>(it->file_size(e2)), it->last_write_time(e2)});
            } else if (name.find(".tmp") != std::string::npos) {
                fs::remove(it->path(), e2);  // a store that died before its rename
            }
        }
        std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.mtime < b.mtime; });
        for (const Found& f : found) {
            index_[f.name] = {f.size, ++seq_};
            total_ += f.size;
        }
        enabled_ = true;
        std::vector<fs::path> victims;
        evict_locked(std::string(), &victims);
        for (const fs::path& v : victims) fs::remove(v, ec);
        return true;
    }

    // Removes sibling generations (other Dawn builds / adapters) under `title_dir` that were not
    // touched for `max_age_days`: their blobs can no longer be hit. Best effort.
    static void prune_stale_siblings(const std::string& title_dir, const std::string& keep_dir, uint32_t max_age_days) {
        namespace fs = std::filesystem;
        std::error_code ec;
        const auto keep = fs::weakly_canonical(keep_dir, ec);
        const auto now = fs::file_time_type::clock::now();
        for (fs::directory_iterator it(title_dir, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code e2;
            if (!it->is_directory(e2)) continue;
            if (fs::weakly_canonical(it->path(), e2) == keep) continue;
            const auto age = now - it->last_write_time(e2);
            if (!e2 && age > std::chrono::hours(24) * max_age_days) fs::remove_all(it->path(), e2);
        }
    }

    bool enabled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return enabled_;
    }

    // Dawn's load hook. value == nullptr or value_size == 0 asks for the stored size only (0 = none,
    // not counted); otherwise copies the blob and returns its size (0 when it does not fit / is gone).
    // The file is read without the mutex: Dawn calls this from the shader workers, its pipeline
    // threads and the executor at once, and one read stalled on a busy disk must not hold up all the
    // others behind it.
    size_t load(const void* key, size_t key_size, void* value, size_t value_size) {
        namespace fs = std::filesystem;
        if (key == nullptr || key_size == 0) return 0;
        const std::string name = file_name(key, key_size);
        fs::path path;
        uint64_t gen = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!enabled_) return 0;
            auto it = index_.find(name);
            if (it == index_.end()) {
                ++misses_;
                return 0;
            }
            path = fs::path(dir_) / name;
            gen = it->second.gen;
        }
        const bool probe = value == nullptr || value_size == 0;
        const uint64_t t0 = io_now_ns();
        io_hook();
        Header h{};
        bool ok = false, damaged = false, fits = true;
        {
            std::ifstream in(path, std::ios::binary);
            if (in && in.read(reinterpret_cast<char*>(&h), sizeof h) && h.magic == kMagic && h.version == kVersion &&
                h.key_size == key_size) {
                std::vector<uint8_t> stored_key(key_size);
                ok = in.read(reinterpret_cast<char*>(stored_key.data()), static_cast<std::streamsize>(key_size)).good() &&
                     std::memcmp(stored_key.data(), key, key_size) == 0;
            } else {
                damaged = true;  // a hash collision between two keys (key mismatch) is a plain miss
            }
            if (ok && !probe) {
                if (value_size < h.value_size) {
                    fits = false;
                } else {
                    in.read(static_cast<char*>(value), static_cast<std::streamsize>(h.value_size));
                    if (!in || fnv1a(value, static_cast<size_t>(h.value_size)) != h.value_hash) {
                        ok = false;
                        damaged = true;
                    }
                }
            }
        }
        const bool hit = ok && fits && !probe;
        if (hit) {
            std::error_code ec;
            fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // recency survives a restart
        }
        const uint64_t dt = io_now_ns() - t0;
        bool remove_file = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            note_io_locked(dt);
            auto it = index_.find(name);
            const bool same = it != index_.end() && it->second.gen == gen;  // not replaced or dropped meanwhile
            if (!ok) {
                if (damaged && same) {
                    forget_locked(it, true);
                    remove_file = true;
                }
                ++misses_;
                return 0;
            }
            if (!fits) return 0;
            if (same) it->second.seq = ++seq_;
            if (hit) {
                ++hits_;
                bytes_loaded_ += h.value_size;
            }
        }
        if (remove_file) {
            std::error_code ec;
            fs::remove(path, ec);
        }
        return static_cast<size_t>(h.value_size);
    }

    // Dawn's store hook. Replaces an existing blob of the same key. Written through a uniquely named
    // temporary file and a rename, without the mutex (see load()).
    void store(const void* key, size_t key_size, const void* value, size_t value_size) {
        namespace fs = std::filesystem;
        if (key == nullptr || key_size == 0 || value == nullptr || value_size == 0) return;
        const uint64_t file_size = sizeof(Header) + key_size + value_size;
        const std::string name = file_name(key, key_size);
        fs::path path, tmp;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!enabled_) return;
            if (max_bytes_ != 0 && file_size > max_bytes_) {
                ++store_rejected_;
                return;
            }
            path = fs::path(dir_) / name;
            tmp = fs::path(dir_) / (name + ".tmp" + std::to_string(++tmp_counter_));
        }
        Header h{};
        h.magic = kMagic;
        h.version = kVersion;
        h.key_size = static_cast<uint32_t>(key_size);
        h.value_size = value_size;
        h.value_hash = fnv1a(value, value_size);
        const uint64_t t0 = io_now_ns();
        io_hook();
        bool written = false;
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            written = out && out.write(reinterpret_cast<const char*>(&h), sizeof h) &&
                      out.write(static_cast<const char*>(key), static_cast<std::streamsize>(key_size)) &&
                      out.write(static_cast<const char*>(value), static_cast<std::streamsize>(value_size));
        }
        std::error_code ec;
        if (written) {
            fs::rename(tmp, path, ec);
            written = !ec;
        }
        if (!written) fs::remove(tmp, ec);
        const uint64_t dt = io_now_ns() - t0;
        std::vector<fs::path> victims;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            note_io_locked(dt);
            if (!written) {
                ++store_rejected_;
                return;
            }
            auto it = index_.find(name);
            if (it != index_.end()) total_ -= it->second.size;
            index_[name] = {file_size, ++seq_, ++gen_};
            total_ += file_size;
            ++stores_;
            bytes_stored_ += value_size;
            evict_locked(name, &victims);
        }
        for (const fs::path& v : victims) fs::remove(v, ec);
    }

    // Tests only: runs on every load / store right where its file I/O starts (no lock held).
    void set_io_hook_for_testing(std::function<void()> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        io_hook_ = std::move(hook);
    }

    BlobCacheStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        BlobCacheStats s;
        s.enabled = enabled_;
        s.dir = dir_;
        s.hits = hits_;
        s.misses = misses_;
        s.stores = stores_;
        s.store_rejected = store_rejected_;
        s.bytes_loaded = bytes_loaded_;
        s.bytes_stored = bytes_stored_;
        s.evictions = evictions_;
        s.evicted_bytes = evicted_bytes_;
        s.corrupt = corrupt_;
        s.entries = index_.size();
        s.total_bytes = total_;
        s.max_bytes = max_bytes_;
        s.io_max_ns = io_max_ns_;
        s.io_slow = io_slow_;
        return s;
    }

private:
    static constexpr uint32_t kMagic = 0x42364750;  // "PG6B"
    static constexpr uint32_t kVersion = 1;
    struct Header {
        uint32_t magic;
        uint32_t version;
        uint32_t key_size;
        uint32_t reserved;
        uint64_t value_size;
        uint64_t value_hash;
    };
    struct Entry {
        uint64_t size = 0;
        uint64_t seq = 0;  // last use; larger = more recent
        uint64_t gen = 0;  // the store that wrote the file (0: found by open())
    };
    using Index = std::unordered_map<std::string, Entry>;

    static std::string file_name(const void* key, size_t n) {
        char buf[40];
        std::snprintf(buf, sizeof buf, "%016llx%016llx.blob", static_cast<unsigned long long>(fnv1a(key, n)),
                      static_cast<unsigned long long>(fnv1a(key, n, 0x9E3779B97F4A7C15ull)));
        return buf;
    }

    // Drops the entry from the index; the caller deletes its file after releasing the mutex.
    void forget_locked(Index::iterator it, bool corrupt) {
        total_ -= it->second.size;
        index_.erase(it);
        if (corrupt) ++corrupt_;
    }

    void note_io_locked(uint64_t ns) {
        io_max_ns_ = std::max(io_max_ns_, ns);
        if (ns >= 1000000000ull) ++io_slow_;
    }

    void io_hook() {
        std::function<void()> h;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            h = io_hook_;
        }
        if (h) h();
    }

    static uint64_t io_now_ns() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    // Drops least recently used blobs from the index until the total fits; `keep` (the blob just
    // stored) stays. Their files go to `victims_out` for the caller to delete without the mutex.
    void evict_locked(const std::string& keep, std::vector<std::filesystem::path>* victims_out) {
        if (max_bytes_ == 0 || total_ <= max_bytes_) return;
        std::vector<Index::iterator> order;
        order.reserve(index_.size());
        for (auto it = index_.begin(); it != index_.end(); ++it) order.push_back(it);
        // A store usually needs just one victim. Build a min-heap in linear time rather than
        // sorting every cached pipeline while Dawn's load/store callbacks share this mutex.
        const auto newer = [](const Index::iterator& a, const Index::iterator& b) {
            return a->second.seq > b->second.seq;
        };
        std::make_heap(order.begin(), order.end(), newer);
        std::vector<std::string> victims;
        uint64_t total = total_;
        while (total > max_bytes_ && !order.empty()) {
            std::pop_heap(order.begin(), order.end(), newer);
            const auto it = order.back();
            order.pop_back();
            if (it->first == keep) continue;
            total -= it->second.size;
            victims.push_back(it->first);
        }
        for (const std::string& v : victims) {
            auto it = index_.find(v);
            ++evictions_;
            evicted_bytes_ += it->second.size;
            forget_locked(it, false);
            victims_out->push_back(std::filesystem::path(dir_) / v);
        }
    }

    mutable std::mutex mutex_;
    bool enabled_ = false;
    std::string dir_;
    uint64_t max_bytes_ = 0;
    uint64_t total_ = 0;
    uint64_t seq_ = 0;
    uint64_t tmp_counter_ = 0;
    Index index_;
    uint64_t hits_ = 0, misses_ = 0, stores_ = 0, store_rejected_ = 0, bytes_loaded_ = 0, bytes_stored_ = 0;
    uint64_t evictions_ = 0, evicted_bytes_ = 0, corrupt_ = 0;
    uint64_t gen_ = 0, io_max_ns_ = 0, io_slow_ = 0;
    std::function<void()> io_hook_;
};

}  // namespace playgta6::gpu::native_exec
