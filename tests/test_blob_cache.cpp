// BlobCache (gpu/native_exec/blob_cache.h): the on-disk store behind Dawn's cache hooks.
// Store/load round trip, persistence across reopen, damaged files, LRU eviction under the cap,
// per-title/version directory split. No GPU, no Dawn.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "gpu/native_exec/blob_cache.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_blob_cache.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu::native_exec;
namespace fs = std::filesystem;

std::vector<uint8_t> bytes(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + i * 7);
    return v;
}
std::string key(int i) { return "key-" + std::to_string(i); }
void put(BlobCache& c, int i, const std::vector<uint8_t>& v) { c.store(key(i).data(), key(i).size(), v.data(), v.size()); }
bool get(BlobCache& c, int i, std::vector<uint8_t>* out) {
    const std::string k = key(i);
    const size_t n = c.load(k.data(), k.size(), nullptr, 0);
    if (n == 0) return false;
    out->assign(n, 0);
    return c.load(k.data(), k.size(), out->data(), out->size()) == n;
}

void test_round_trip_and_reopen(const fs::path& root) {
    const std::string dir = (root / "rt").string();
    {
        BlobCache c;
        CHECK(c.open(dir, 1 << 20));
        std::vector<uint8_t> got;
        CHECK(!get(c, 1, &got));  // empty cache: a miss
        CHECK(c.stats().misses == 1);
        put(c, 1, bytes(1000, 1));
        put(c, 2, bytes(5000, 2));
        CHECK(get(c, 1, &got) && got == bytes(1000, 1));
        CHECK(get(c, 2, &got) && got == bytes(5000, 2));
        const BlobCacheStats s = c.stats();
        CHECK(s.stores == 2 && s.hits == 2 && s.entries == 2);  // the size probes are not hits
        CHECK(s.bytes_loaded == 6000 && s.bytes_stored == 6000);
        // A buffer that is too small gets nothing; a replaced blob returns the new value.
        std::vector<uint8_t> small(10);
        const std::string k = key(1);
        CHECK(c.load(k.data(), k.size(), small.data(), small.size()) == 0);
        put(c, 1, bytes(300, 9));
        CHECK(get(c, 1, &got) && got == bytes(300, 9));
        CHECK(c.stats().entries == 2);
    }
    {
        BlobCache c;  // a new process: the blobs are still there
        CHECK(c.open(dir, 1 << 20));
        std::vector<uint8_t> got;
        CHECK(c.stats().entries == 2);
        CHECK(get(c, 1, &got) && got == bytes(300, 9));
        CHECK(get(c, 2, &got) && got == bytes(5000, 2));
        CHECK(c.stats().hits == 2 && c.stats().misses == 0);
    }
}

void test_corrupt_file(const fs::path& root) {
    const std::string dir = (root / "bad").string();
    BlobCache c;
    CHECK(c.open(dir, 1 << 20));
    put(c, 1, bytes(2000, 3));
    put(c, 2, bytes(2000, 4));
    // Flip a byte in the value of one blob (checksum mismatch) and truncate the other (torn write).
    int n = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".blob") continue;
        if (n++ == 0) {
            std::fstream f(e.path(), std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(-5, std::ios::end);
            f.put(static_cast<char>(0x55));
        } else {
            fs::resize_file(e.path(), 20);
        }
    }
    std::vector<uint8_t> got;
    const bool a = get(c, 1, &got);
    const bool b = get(c, 2, &got);
    CHECK(!a && !b);
    CHECK(c.stats().corrupt == 2 && c.stats().entries == 0 && c.stats().total_bytes == 0);
    put(c, 1, bytes(2000, 3));  // recovers
    CHECK(get(c, 1, &got) && got == bytes(2000, 3));
}

void test_eviction_lru(const fs::path& root) {
    const std::string dir = (root / "lru").string();
    BlobCache c;
    const size_t value = 1000;
    const uint64_t per_file = 32 + 5 + value;  // 32-byte header + the 5-char key + the value
    CHECK(c.open(dir, per_file * 3 + per_file / 2));  // room for three
    std::vector<uint8_t> got;
    put(c, 1, bytes(value, 1));
    put(c, 2, bytes(value, 2));
    put(c, 3, bytes(value, 3));
    CHECK(c.stats().entries == 3 && c.stats().evictions == 0);
    CHECK(get(c, 1, &got));      // 1 is now more recent than 2
    put(c, 4, bytes(value, 4));  // over the cap: the least recently used (2) goes
    BlobCacheStats s = c.stats();
    CHECK(s.entries == 3 && s.evictions == 1 && s.evicted_bytes == per_file);
    CHECK(s.total_bytes <= s.max_bytes);
    CHECK(!get(c, 2, &got));
    CHECK(get(c, 1, &got) && get(c, 3, &got) && get(c, 4, &got));
    // A blob that alone exceeds the cap is rejected and evicts nothing.
    put(c, 5, bytes(per_file * 4, 5));
    s = c.stats();
    CHECK(s.store_rejected == 1 && s.entries == 3 && s.evictions == 1);
    // Reopening with a smaller cap trims on open.
    BlobCache d;
    CHECK(d.open(dir, per_file * 2 + 10));
    CHECK(d.stats().entries == 2 && d.stats().total_bytes <= per_file * 2 + 10);
}

void test_recency_survives_reopen(const fs::path& root) {
    const std::string dir = (root / "recent").string();
    const size_t value = 500;
    const uint64_t per_file = 32 + 5 + value;
    {
        BlobCache c;
        CHECK(c.open(dir, 1 << 20));
        put(c, 1, bytes(value, 1));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        put(c, 2, bytes(value, 2));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::vector<uint8_t> got;
        CHECK(get(c, 1, &got));  // touches blob 1's file time: 2 is now the oldest
    }
    BlobCache c;
    CHECK(c.open(dir, per_file));  // keeps one
    std::vector<uint8_t> got;
    CHECK(c.stats().entries == 1);
    CHECK(get(c, 1, &got));
    CHECK(!get(c, 2, &got));
}

void test_disabled_and_dirs(const fs::path& root) {
    BlobCache c;  // never opened: every hook is a harmless miss
    std::vector<uint8_t> got;
    put(c, 1, bytes(10, 1));
    CHECK(!get(c, 1, &got) && !c.enabled() && c.stats().stores == 0);
    const std::string a = BlobCache::versioned_dir("R", "gta 6", "7435fd87", "nvidia/0x2684");
    const std::string b = BlobCache::versioned_dir("R", "gta 6", "7435fd87", "amd/0x744c");
    const std::string d = BlobCache::versioned_dir("R", "other", "7435fd87", "nvidia/0x2684");
    CHECK(a != b && a != d);
    CHECK(a.find("gta_6") != std::string::npos && a.find("7435fd87-nvidia_0x2684") != std::string::npos);
    // Old generations are pruned by age only: a fresh sibling stays.
    const fs::path title = root / "title";
    fs::create_directories(title / "gen-new");
    fs::create_directories(title / "gen-cur");
    BlobCache::prune_stale_siblings(title.string(), (title / "gen-cur").string(), 30);
    CHECK(fs::exists(title / "gen-new") && fs::exists(title / "gen-cur"));
    fs::last_write_time(title / "gen-new", fs::file_time_type::clock::now() - std::chrono::hours(24 * 40));
    BlobCache::prune_stale_siblings(title.string(), (title / "gen-cur").string(), 30);
    CHECK(!fs::exists(title / "gen-new") && fs::exists(title / "gen-cur"));
}

void test_threads(const fs::path& root) {
    BlobCache c;
    CHECK(c.open((root / "mt").string(), 1 << 22));
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t) {
        ts.emplace_back([&c, t] {
            for (int i = 0; i < 40; ++i) {
                std::vector<uint8_t> got;
                put(c, t * 100 + i, bytes(200 + i, static_cast<uint8_t>(t)));
                get(c, t * 100 + i, &got);
            }
        });
    }
    for (auto& t : ts) t.join();
    CHECK(c.stats().stores == 160 && c.stats().hits == 160 && c.stats().entries == 160);
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
                          ("pg6_blobcache_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    test_round_trip_and_reopen(root);
    test_corrupt_file(root);
    test_eviction_lru(root);
    test_recency_survives_reopen(root);
    test_disabled_and_dirs(root);
    test_threads(root);
    std::error_code ec;
    fs::remove_all(root, ec);
    if (g_failures == 0) std::puts("test_blob_cache: all passed");
    return g_failures == 0 ? 0 : 1;
}
