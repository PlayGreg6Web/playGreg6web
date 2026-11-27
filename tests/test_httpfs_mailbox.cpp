#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "os/httpfs_mailbox.h"

namespace {

std::atomic<int> g_failures{0};

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_httpfs_mailbox.cpp:%d]: %s\n", line, expr);
        g_failures.fetch_add(1, std::memory_order_relaxed);
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

uint8_t pattern_byte(int32_t handle, uint64_t offset) {
    const uint64_t mixed = (static_cast<uint64_t>(handle) * 0x9E3779B97F4A7C15ull) ^
                           (offset * 0xBF58476D1CE4E5B9ull);
    return static_cast<uint8_t>((mixed >> 56) ^ (mixed & 0xFFu));
}

void update_max(std::atomic<int>& target, int value) {
    int prev = target.load(std::memory_order_relaxed);
    while (value > prev &&
           !target.compare_exchange_weak(prev, value, std::memory_order_relaxed)) {
    }
}

// Reusable bounded-timeout rendezvous barrier so a broken serialization bug fails cleanly
// instead of hanging the test suite indefinitely.
class TimedRendezvous {
public:
    explicit TimedRendezvous(int expected) : expected_(expected) {}

    bool arrive_and_wait(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrived_;
        if (arrived_ >= expected_) {
            cv_.notify_all();
            return true;
        }
        return cv_.wait_for(lock, timeout, [&] { return arrived_ >= expected_; });
    }

private:
    const int expected_;
    std::mutex mutex_;
    std::condition_variable cv_;
    int arrived_ = 0;
};

// ---------------------------------------------------------------------------
// 1. concurrent = true executes direct open/read/close truly simultaneously
// ---------------------------------------------------------------------------

void test_concurrent_direct_parallel_io() {
    using namespace playgta6::os;

    constexpr int kThreads = 4;
    TimedRendezvous open_barrier(kThreads);
    TimedRendezvous read_barrier(kThreads);
    TimedRendezvous close_barrier(kThreads);

    std::atomic<int> in_flight_open{0};
    std::atomic<int> max_open{0};
    std::atomic<int> in_flight_read{0};
    std::atomic<int> max_read{0};
    std::atomic<int> in_flight_close{0};
    std::atomic<int> max_close{0};
    std::atomic<int> open_rendezvous_ok{0};
    std::atomic<int> read_rendezvous_ok{0};
    std::atomic<int> close_rendezvous_ok{0};

    HttpfsDirectBackend backend;
    backend.concurrent = true;
    backend.open = [&](std::string_view rel_path) -> int32_t {
        const int now = in_flight_open.fetch_add(1, std::memory_order_acq_rel) + 1;
        update_max(max_open, now);
        if (open_rendezvous_ok.load(std::memory_order_relaxed) < kThreads &&
            open_barrier.arrive_and_wait()) {
            open_rendezvous_ok.fetch_add(1, std::memory_order_relaxed);
        }
        in_flight_open.fetch_sub(1, std::memory_order_acq_rel);
        // Encode lane id from last char ("lane/0" .. "lane/3") into handle 10..13.
        if (rel_path.empty()) {
            return -1;
        }
        return 10 + (rel_path.back() - '0');
    };
    backend.read = [&](int32_t handle, uint64_t offset, void* dst, size_t len) -> int64_t {
        const int now = in_flight_read.fetch_add(1, std::memory_order_acq_rel) + 1;
        update_max(max_read, now);
        // First chunk on every thread must rendezvous simultaneously inside direct_.read.
        // If HttpfsFairMutex were held here, this barrier would time out and fail.
        if (offset == 0) {
            if (read_barrier.arrive_and_wait()) {
                read_rendezvous_ok.fetch_add(1, std::memory_order_relaxed);
            }
        }
        auto* out = static_cast<uint8_t*>(dst);
        for (size_t i = 0; i < len; ++i) {
            out[i] = pattern_byte(handle, offset + i);
        }
        in_flight_read.fetch_sub(1, std::memory_order_acq_rel);
        return static_cast<int64_t>(len);
    };
    backend.close = [&](int32_t handle) -> bool {
        const int now = in_flight_close.fetch_add(1, std::memory_order_acq_rel) + 1;
        update_max(max_close, now);
        if (close_barrier.arrive_and_wait()) {
            close_rendezvous_ok.fetch_add(1, std::memory_order_relaxed);
        }
        in_flight_close.fetch_sub(1, std::memory_order_acq_rel);
        return handle >= 10;
    };

    HttpfsMailboxClient client(std::move(backend));
    constexpr size_t kReadLen = kHttpfsPayloadBytes * 2 + 37;

    std::atomic<int> data_ok{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            const std::string path = "lane/" + std::to_string(t);
            const int32_t handle = client.open(path);
            if (handle != 10 + t) {
                return;
            }
            std::vector<uint8_t> buf(kReadLen, 0);
            const int64_t got = client.read(handle, 0, buf.data(), buf.size());
            if (got != static_cast<int64_t>(kReadLen)) {
                return;
            }
            bool match = true;
            for (size_t i = 0; i < kReadLen; ++i) {
                if (buf[i] != pattern_byte(handle, i)) {
                    match = false;
                    break;
                }
            }
            if (match && client.close(handle)) {
                data_ok.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    CHECK(open_rendezvous_ok.load() == kThreads);
    CHECK(read_rendezvous_ok.load() == kThreads);
    CHECK(close_rendezvous_ok.load() == kThreads);
    CHECK(max_open.load() == kThreads);
    CHECK(max_read.load() == kThreads);
    CHECK(max_close.load() == kThreads);
    CHECK(data_ok.load() == kThreads);
}

// ---------------------------------------------------------------------------
// 2. concurrent = false (default) preserves strict HttpfsFairMutex serialization
// ---------------------------------------------------------------------------

void test_default_direct_serializes_io() {
    using namespace playgta6::os;

    HttpfsDirectBackend backend;
    CHECK(!backend.concurrent);
    CHECK(backend.max_chunk_bytes == kHttpfsPayloadBytes);

    std::atomic<int> active_calls{0};
    std::atomic<int> max_active_calls{0};

    auto enter_and_hold = [&]() {
        const int now = active_calls.fetch_add(1, std::memory_order_acq_rel) + 1;
        update_max(max_active_calls, now);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        active_calls.fetch_sub(1, std::memory_order_acq_rel);
    };

    backend.open = [&](std::string_view rel_path) -> int32_t {
        enter_and_hold();
        return rel_path.empty() ? -1 : 100 + (rel_path.back() - '0');
    };
    backend.read = [&](int32_t handle, uint64_t offset, void* dst, size_t len) -> int64_t {
        enter_and_hold();
        auto* out = static_cast<uint8_t*>(dst);
        for (size_t i = 0; i < len; ++i) {
            out[i] = pattern_byte(handle, offset + i);
        }
        return static_cast<int64_t>(len);
    };
    backend.close = [&](int32_t handle) -> bool {
        enter_and_hold();
        return handle >= 100;
    };

    HttpfsMailboxClient client(std::move(backend));
    constexpr int kThreads = 4;
    constexpr size_t kReadLen = kHttpfsPayloadBytes * 3;
    std::atomic<int> completed{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            const int32_t handle = client.open("serial/" + std::to_string(t));
            if (handle != 100 + t) {
                return;
            }
            std::vector<uint8_t> buf(kReadLen, 0);
            const int64_t got = client.read(handle, 0, buf.data(), buf.size());
            if (got != static_cast<int64_t>(kReadLen)) {
                return;
            }
            bool match = true;
            for (size_t i = 0; i < kReadLen; ++i) {
                if (buf[i] != pattern_byte(handle, i)) {
                    match = false;
                    break;
                }
            }
            if (match && client.close(handle)) {
                completed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    CHECK(completed.load() == kThreads);
    CHECK(max_active_calls.load() == 1);
}

// ---------------------------------------------------------------------------
// 3. max_chunk_bytes overrides chunk size in direct mode & retries cleanly
// ---------------------------------------------------------------------------

void test_direct_max_chunk_bytes_and_retries() {
    using namespace playgta6::os;

    constexpr size_t kCustomChunk = 256 * 1024;  // 256 KiB

    // 3a. 256 KiB single read vs 64 KiB default, plus multi-chunk + EOF clamping.
    {
        std::vector<std::pair<uint64_t, size_t>> custom_calls;
        constexpr size_t kFileSize = 600 * 1024;  // 256 KiB + 256 KiB + 88 KiB

        HttpfsDirectBackend backend;
        backend.max_chunk_bytes = kCustomChunk;
        backend.open = [](std::string_view) { return 7; };
        backend.read = [&](int32_t handle, uint64_t offset, void* dst, size_t len) -> int64_t {
            if (handle != 7) {
                return -9;
            }
            custom_calls.emplace_back(offset, len);
            if (offset >= kFileSize) {
                return 0;
            }
            const size_t n = std::min<size_t>(len, kFileSize - static_cast<size_t>(offset));
            auto* out = static_cast<uint8_t*>(dst);
            for (size_t i = 0; i < n; ++i) {
                out[i] = pattern_byte(handle, offset + i);
            }
            return static_cast<int64_t>(n);
        };
        backend.close = [](int32_t) { return true; };

        HttpfsMailboxClient client(std::move(backend));
        CHECK(client.payload_capacity() == kCustomChunk);

        // Single 256 KiB read completes in exactly 1 direct_.read call.
        std::vector<uint8_t> single_buf(kCustomChunk, 0);
        CHECK(client.read(7, 0, single_buf.data(), single_buf.size()) ==
              static_cast<int64_t>(kCustomChunk));
        CHECK(custom_calls.size() == 1);
        CHECK(custom_calls[0].first == 0 && custom_calls[0].second == kCustomChunk);
        bool single_ok = true;
        for (size_t i = 0; i < kCustomChunk; ++i) {
            if (single_buf[i] != pattern_byte(7, i)) {
                single_ok = false;
                break;
            }
        }
        CHECK(single_ok);

        // Full 600 KiB read (+ extra requested beyond EOF) uses 256 KiB chunks then short tail.
        custom_calls.clear();
        std::vector<uint8_t> full_buf(kFileSize + 4096, 0);
        CHECK(client.read(7, 0, full_buf.data(), full_buf.size()) ==
              static_cast<int64_t>(kFileSize));
        CHECK(custom_calls.size() == 3);
        CHECK(custom_calls[0].first == 0 && custom_calls[0].second == kCustomChunk);
        CHECK(custom_calls[1].first == kCustomChunk && custom_calls[1].second == kCustomChunk);
        CHECK(custom_calls[2].first == 2 * kCustomChunk &&
              custom_calls[2].second == kFileSize - 2 * kCustomChunk + 4096);
        bool full_ok = true;
        for (size_t i = 0; i < kFileSize; ++i) {
            if (full_buf[i] != pattern_byte(7, i)) {
                full_ok = false;
                break;
            }
        }
        CHECK(full_ok);
    }

    // 3b. Default max_chunk_bytes (64 KiB) splits a 256 KiB read into 4 x 64 KiB calls.
    {
        std::vector<size_t> default_chunks;
        HttpfsDirectBackend backend;
        backend.open = [](std::string_view) { return 3; };
        backend.read = [&](int32_t, uint64_t, void* dst, size_t len) -> int64_t {
            default_chunks.push_back(len);
            std::memset(dst, 0x3C, len);
            return static_cast<int64_t>(len);
        };
        backend.close = [](int32_t) { return true; };

        HttpfsMailboxClient client(std::move(backend));
        CHECK(client.payload_capacity() == kHttpfsPayloadBytes);
        std::vector<uint8_t> buf(kCustomChunk, 0);
        CHECK(client.read(3, 0, buf.data(), buf.size()) == static_cast<int64_t>(kCustomChunk));
        CHECK(default_chunks.size() == 4);
        for (size_t sz : default_chunks) {
            CHECK(sz == kHttpfsPayloadBytes);
        }
    }

    // 3c. max_chunk_bytes == 0 falls back safely to kHttpfsPayloadBytes.
    {
        std::vector<size_t> fallback_chunks;
        HttpfsDirectBackend backend;
        backend.max_chunk_bytes = 0;
        backend.open = [](std::string_view) { return 4; };
        backend.read = [&](int32_t, uint64_t, void* dst, size_t len) -> int64_t {
            fallback_chunks.push_back(len);
            std::memset(dst, 0x7E, len);
            return static_cast<int64_t>(len);
        };
        backend.close = [](int32_t) { return true; };

        HttpfsMailboxClient client(std::move(backend));
        CHECK(client.payload_capacity() == kHttpfsPayloadBytes);
        std::vector<uint8_t> buf(kHttpfsPayloadBytes * 2, 0);
        CHECK(client.read(4, 0, buf.data(), buf.size()) ==
              static_cast<int64_t>(kHttpfsPayloadBytes * 2));
        CHECK(fallback_chunks.size() == 2);
        CHECK(fallback_chunks[0] == kHttpfsPayloadBytes);
        CHECK(fallback_chunks[1] == kHttpfsPayloadBytes);
    }

    // 3d. Transient errors & persistent failures with custom 256 KiB chunk size.
    {
        int fail_chunk1_remaining = 2;
        bool persistent_fail = false;
        bool return_oversized = false;
        std::vector<std::pair<uint64_t, size_t>> attempts;

        HttpfsDirectBackend backend;
        backend.max_chunk_bytes = kCustomChunk;
        backend.open = [](std::string_view) { return 9; };
        backend.read = [&](int32_t handle, uint64_t offset, void* dst, size_t len) -> int64_t {
            attempts.emplace_back(offset, len);
            if (return_oversized) {
                return static_cast<int64_t>(len) + 1;
            }
            if (persistent_fail) {
                return -5;
            }
            if (offset == kCustomChunk && fail_chunk1_remaining > 0) {
                --fail_chunk1_remaining;
                return -5;
            }
            auto* out = static_cast<uint8_t*>(dst);
            for (size_t i = 0; i < len; ++i) {
                out[i] = pattern_byte(handle, offset + i);
            }
            return static_cast<int64_t>(len);
        };
        backend.close = [](int32_t) { return true; };

        HttpfsMailboxClient client(std::move(backend));
        std::vector<uint32_t> sleeps;
        HttpfsMailboxClient::RetryPolicy policy;
        policy.max_retries = 3;
        policy.initial_backoff_ms = 10;
        policy.sleep = [&](uint32_t ms) { sleeps.push_back(ms); };
        client.set_retry_policy(policy);

        // 512 KiB read: chunk 0 succeeds immediately, chunk 1 fails twice then succeeds.
        std::vector<uint8_t> buf(2 * kCustomChunk, 0);
        CHECK(client.read(9, 0, buf.data(), buf.size()) ==
              static_cast<int64_t>(2 * kCustomChunk));
        CHECK(client.retried_chunks() == 2);
        CHECK(sleeps.size() == 2 && sleeps[0] == 10 && sleeps[1] == 20);
        CHECK(attempts.size() == 4);
        CHECK(attempts[0].second == kCustomChunk && attempts[1].second == kCustomChunk &&
              attempts[2].second == kCustomChunk && attempts[3].second == kCustomChunk);
        bool match = true;
        for (size_t i = 0; i < buf.size(); ++i) {
            if (buf[i] != pattern_byte(9, i)) {
                match = false;
                break;
            }
        }
        CHECK(match);

        // Persistent failure exhausts max_retries, leaves buffer sentinel intact.
        sleeps.clear();
        persistent_fail = true;
        std::vector<uint8_t> sentinel(kCustomChunk, 0xCC);
        CHECK(client.read(9, 0, sentinel.data(), sentinel.size()) == -5);
        CHECK(client.retried_chunks() == 5);
        CHECK(sleeps.size() == 3 && sleeps[0] == 10 && sleeps[1] == 20 && sleeps[2] == 40);
        CHECK(std::all_of(sentinel.begin(), sentinel.end(), [](uint8_t b) { return b == 0xCC; }));

        // Malformed response (n > chunk) is NOT retried.
        sleeps.clear();
        persistent_fail = false;
        return_oversized = true;
        CHECK(client.read(9, 0, sentinel.data(), sentinel.size()) == -1);
        CHECK(client.retried_chunks() == 5);
        CHECK(sleeps.empty());
    }
}

// ---------------------------------------------------------------------------
// 4. Concurrent retries across multiple threads with concurrent = true
// ---------------------------------------------------------------------------

void test_concurrent_direct_retries_multithreaded() {
    using namespace playgta6::os;

    constexpr int kThreads = 8;
    constexpr int kSucceedingThreads = 6;
    constexpr size_t kChunkBytes = 128 * 1024;
    constexpr int kChunksPerThread = 3;
    constexpr size_t kTotalPerThread = kChunkBytes * kChunksPerThread;
    constexpr uint32_t kMaxRetries = 3;

    // Each succeeding thread's chunk c fails `(t + c) % 2 + 1` times (1 or 2 transient failures)
    // before succeeding. Threads 6 and 7 always fail with -5 to test concurrent exhaustion.
    std::atomic<int> failures_remaining[kSucceedingThreads][kChunksPerThread];
    std::atomic<bool> first_attempt_seen[kThreads];
    uint64_t expected_retries = 0;
    for (int t = 0; t < kThreads; ++t) {
        first_attempt_seen[t].store(false, std::memory_order_relaxed);
        if (t < kSucceedingThreads) {
            for (int c = 0; c < kChunksPerThread; ++c) {
                const int fail_count = ((t + c) % 2) + 1;  // 1 or 2
                failures_remaining[t][c].store(fail_count, std::memory_order_relaxed);
                expected_retries += static_cast<uint64_t>(fail_count);
            }
        } else {
            // Stops on chunk 0 after kMaxRetries retries.
            expected_retries += kMaxRetries;
        }
    }

    TimedRendezvous initial_wave(kThreads);
    std::atomic<int> in_flight{0};
    std::atomic<int> max_in_flight{0};
    std::atomic<uint64_t> sleep_calls{0};

    HttpfsDirectBackend backend;
    backend.concurrent = true;
    backend.max_chunk_bytes = kChunkBytes;
    backend.open = [](std::string_view rel) -> int32_t {
        return rel.empty() ? -1 : (rel.back() - '0');
    };
    backend.read = [&](int32_t handle, uint64_t offset, void* dst, size_t len) -> int64_t {
        const int now = in_flight.fetch_add(1, std::memory_order_acq_rel) + 1;
        update_max(max_in_flight, now);

        const int c = static_cast<int>(offset / kChunkBytes);
        if (handle < 0 || handle >= kThreads || c < 0 || c >= kChunksPerThread ||
            len != kChunkBytes) {
            in_flight.fetch_sub(1, std::memory_order_acq_rel);
            return -22;
        }

        // Rendezvous all 8 threads on their very first chunk attempt to force overlapping retries.
        if (c == 0 && !first_attempt_seen[handle].exchange(true, std::memory_order_acq_rel)) {
            initial_wave.arrive_and_wait();
        }

        if (handle >= kSucceedingThreads) {
            in_flight.fetch_sub(1, std::memory_order_acq_rel);
            return -5;
        }

        const int prev = failures_remaining[handle][c].fetch_sub(1, std::memory_order_acq_rel);
        if (prev > 0) {
            in_flight.fetch_sub(1, std::memory_order_acq_rel);
            return -5;
        }

        auto* out = static_cast<uint8_t*>(dst);
        for (size_t i = 0; i < len; ++i) {
            out[i] = pattern_byte(handle + 20, offset + i);
        }
        in_flight.fetch_sub(1, std::memory_order_acq_rel);
        return static_cast<int64_t>(len);
    };
    backend.close = [](int32_t) { return true; };

    HttpfsMailboxClient client(std::move(backend));
    HttpfsMailboxClient::RetryPolicy policy;
    policy.max_retries = kMaxRetries;
    policy.initial_backoff_ms = 1;
    policy.sleep = [&](uint32_t) {
        sleep_calls.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
    };
    client.set_retry_policy(policy);

    std::atomic<int> ok_threads{0};
    std::atomic<int> expected_fail_threads{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            const int32_t handle = client.open("retry/" + std::to_string(t));
            if (handle != t) {
                return;
            }
            std::vector<uint8_t> buf(kTotalPerThread, 0xA5);
            const int64_t got = client.read(handle, 0, buf.data(), buf.size());
            if (t < kSucceedingThreads) {
                if (got != static_cast<int64_t>(kTotalPerThread)) {
                    return;
                }
                for (size_t i = 0; i < kTotalPerThread; ++i) {
                    if (buf[i] != pattern_byte(handle + 20, i)) {
                        return;
                    }
                }
                ok_threads.fetch_add(1, std::memory_order_relaxed);
            } else {
                const bool untouched =
                    std::all_of(buf.begin(), buf.end(), [](uint8_t b) { return b == 0xA5; });
                if (got == -5 && untouched) {
                    expected_fail_threads.fetch_add(1, std::memory_order_relaxed);
                }
            }
            client.close(handle);
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    CHECK(ok_threads.load() == kSucceedingThreads);
    CHECK(expected_fail_threads.load() == kThreads - kSucceedingThreads);
    CHECK(client.retried_chunks() == expected_retries);
    CHECK(sleep_calls.load() == expected_retries);
    CHECK(max_in_flight.load() >= kThreads);

    // Also exercise concurrent set_retry_policy() while concurrent readers fail & retry to verify
    // safe synchronization of retry_ under mutex_.
    std::atomic<bool> stop_mutator{false};
    std::thread mutator([&]() {
        uint32_t toggle = 0;
        while (!stop_mutator.load(std::memory_order_acquire)) {
            HttpfsMailboxClient::RetryPolicy p;
            p.max_retries = 1 + (toggle++ % 2);
            p.initial_backoff_ms = 1;
            p.sleep = [](uint32_t) { std::this_thread::yield(); };
            client.set_retry_policy(std::move(p));
        }
    });

    std::vector<std::thread> stress_readers;
    for (int t = 0; t < 4; ++t) {
        stress_readers.emplace_back([&]() {
            std::vector<uint8_t> buf(kChunkBytes, 0x5A);
            for (int iter = 0; iter < 8; ++iter) {
                const int64_t r = client.read(kSucceedingThreads, 0, buf.data(), buf.size());
                CHECK(r == -5);
            }
        });
    }
    for (auto& th : stress_readers) {
        th.join();
    }
    stop_mutator.store(true, std::memory_order_release);
    mutator.join();
}

// ---------------------------------------------------------------------------
// 5. HttpfsOp::Stat — 1-roundtrip file size query (mailbox and direct modes)
// ---------------------------------------------------------------------------

void test_stat_mailbox_and_direct() {
    using namespace playgta6::os;

    // 5a. Shared-memory mailbox mode.
    HttpfsMailbox mailbox{};
    HttpfsMailboxClient::init_mailbox(mailbox);
    std::atomic<int> stat_calls{0};
    std::atomic<int> read_calls{0};
    HttpfsMailboxClient mb_client(&mailbox, [&](HttpfsMailbox& mb) {
        const auto op = static_cast<HttpfsOp>(mb.op);
        if (op == HttpfsOp::Stat) {
            stat_calls.fetch_add(1, std::memory_order_relaxed);
            std::string path(reinterpret_cast<const char*>(mb.payload), static_cast<size_t>(mb.length));
            if (path == "game/common.rpf") {
                HttpfsMailboxClient::complete_response(mb, 5368709120LL, HttpfsStatus::ResponseReady);
            } else if (path == "empty.bin") {
                HttpfsMailboxClient::complete_response(mb, 0LL, HttpfsStatus::ResponseReady);
            } else {
                HttpfsMailboxClient::complete_response(mb, -2LL, HttpfsStatus::Error);
            }
            return;
        }
        if (op == HttpfsOp::Read) {
            read_calls.fetch_add(1, std::memory_order_relaxed);
        }
        HttpfsMailboxClient::complete_response(mb, -38LL, HttpfsStatus::Error);
    });

    CHECK(mb_client.stat("game/common.rpf") == 5368709120LL);
    CHECK(mb_client.stat("empty.bin") == 0LL);
    CHECK(mb_client.stat("missing.rpf") == -2LL);
    CHECK(mb_client.stat("") == -22LL);
    CHECK(stat_calls.load() == 3);
    CHECK(read_calls.load() == 0);

    // 5b. Direct backend with stat callback.
    HttpfsDirectBackend direct;
    direct.stat = [](std::string_view rel) -> int64_t {
        if (rel == "update/dlc.rpf") return 987654321LL;
        return -2;
    };
    direct.concurrent = true;
    HttpfsMailboxClient direct_client(std::move(direct));
    CHECK(direct_client.stat("update/dlc.rpf") == 987654321LL);
    CHECK(direct_client.stat("absent.rpf") == -2LL);
    CHECK(direct_client.stat("") == -22LL);

    // 5c. Direct backend without stat callback returns -38 (ENOSYS) cleanly.
    HttpfsDirectBackend legacy;
    HttpfsMailboxClient legacy_client(std::move(legacy));
    CHECK(legacy_client.stat("update/dlc.rpf") == -38LL);
}

}  // namespace

int main() {
    test_concurrent_direct_parallel_io();
    test_default_direct_serializes_io();
    test_direct_max_chunk_bytes_and_retries();
    test_concurrent_direct_retries_multithreaded();
    test_stat_mailbox_and_direct();

    const int failures = g_failures.load(std::memory_order_relaxed);
    if (failures != 0) {
        std::fprintf(stderr, "test_httpfs_mailbox failed with %d failure(s)\n", failures);
        return 1;
    }
    std::printf("All test_httpfs_mailbox tests passed.\n");
    return 0;
}
