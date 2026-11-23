#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "abi/playgta6_abi.h"
#include "os/game_input.h"
#include "os/httpfs_mailbox.h"
#include "os/threading.h"
#include "os/vmem.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_os_shims.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

// ---------------------------------------------------------------------------
// 1. VirtualMemoryArena tests
// ---------------------------------------------------------------------------

void test_virtual_memory_arena() {
    using namespace playgta6::os;

    VirtualMemoryArena arena;

    // Reserve 128 KiB (2 * 64 KiB granularity, 32 * 4 KiB pages).
    const uint64_t base = arena.reserve(0, 128 * 1024, PageProtect::ReadWrite);
    CHECK(base != 0);
    CHECK((base % kAllocationGranularity) == 0);
    CHECK(arena.reserved_bytes() == 128 * 1024);
    CHECK(arena.committed_bytes() == 0);
    CHECK(arena.reservation_count() == 1);

    // Query reserved (uncommitted) region.
    MemoryBasicInfo info = arena.query(base);
    CHECK(info.base_address == base);
    CHECK(info.allocation_base == base);
    CHECK(info.region_size == 128 * 1024);
    CHECK(info.state == MemState::Reserve);
    CHECK(info.protect == PageProtect::NoAccess);

    // Uncommitted read/write/protect must fail.
    uint8_t buf[16]{};
    CHECK(!arena.read(base, buf, sizeof(buf)));
    CHECK(!arena.write(base, buf, sizeof(buf)));
    CHECK(!arena.protect(base, kPageSize, PageProtect::ReadOnly));

    // Commit 3 pages (12 KiB) at page offset 1 (unaligned input rounds down/up to 4 KiB pages).
    CHECK(arena.commit(base + kPageSize + 123, 2 * kPageSize + 1, PageProtect::ReadWrite));
    CHECK(arena.committed_bytes() == 3 * kPageSize);

    // Newly committed pages must be zero-initialized.
    std::vector<uint8_t> zeros(3 * kPageSize, 0xAA);
    CHECK(arena.read(base + kPageSize, zeros.data(), zeros.size()));
    for (uint8_t b : zeros) {
        CHECK(b == 0);
        if (b != 0) break;
    }

    // Write distinct patterns into page 1, page 2, and page 3.
    std::vector<uint8_t> p1(kPageSize, 0x11);
    std::vector<uint8_t> p2(kPageSize, 0x22);
    std::vector<uint8_t> p3(kPageSize, 0x33);
    CHECK(arena.write(base + 1 * kPageSize, p1.data(), p1.size()));
    CHECK(arena.write(base + 2 * kPageSize, p2.data(), p2.size()));
    CHECK(arena.write(base + 3 * kPageSize, p3.data(), p3.size()));

    // Direct host_ptr works for writable single extent.
    uint8_t* raw = arena.host_ptr(base + 1 * kPageSize, 3 * kPageSize);
    CHECK(raw != nullptr);
    if (raw != nullptr) {
        CHECK(raw[0] == 0x11);
        CHECK(raw[kPageSize] == 0x22);
        CHECK(raw[2 * kPageSize] == 0x33);
    }

    // Re-committing already committed pages retains contents and updates protection.
    CHECK(arena.commit(base + 1 * kPageSize, kPageSize, PageProtect::ReadOnly));
    CHECK(arena.committed_bytes() == 3 * kPageSize);
    uint8_t verify_byte = 0;
    CHECK(arena.read(base + 1 * kPageSize + 7, &verify_byte, 1));
    CHECK(verify_byte == 0x11);
    CHECK(!arena.write(base + 1 * kPageSize, &verify_byte, 1));  // ReadOnly blocks writes
    CHECK(arena.host_ptr(base + 1 * kPageSize, 1) == nullptr);
    CHECK(arena.host_ptr_const(base + 1 * kPageSize, 1) != nullptr);

    // VirtualProtect transitions and old_protect reporting.
    PageProtect old_prot = PageProtect::NoAccess;
    CHECK(arena.protect(base + 1 * kPageSize, kPageSize, PageProtect::ExecuteRead, &old_prot));
    CHECK(old_prot == PageProtect::ReadOnly);
    CHECK(arena.read(base + 1 * kPageSize, &verify_byte, 1));
    CHECK(!arena.write(base + 1 * kPageSize, &verify_byte, 1));

    CHECK(arena.protect(base + 1 * kPageSize, kPageSize, PageProtect::NoAccess, &old_prot));
    CHECK(old_prot == PageProtect::ExecuteRead);
    CHECK(!arena.read(base + 1 * kPageSize, &verify_byte, 1));
    CHECK(arena.host_ptr_const(base + 1 * kPageSize, 1) == nullptr);

    CHECK(arena.protect(base + 1 * kPageSize, kPageSize, PageProtect::ReadWrite, &old_prot));
    CHECK(old_prot == PageProtect::NoAccess);

    // Partial middle decommit: decommit page 2, keeping page 1 and page 3 intact.
    CHECK(arena.decommit(base + 2 * kPageSize, kPageSize));
    CHECK(arena.committed_bytes() == 2 * kPageSize);
    CHECK(!arena.read(base + 2 * kPageSize, &verify_byte, 1));

    uint8_t b1 = 0, b3 = 0;
    CHECK(arena.read(base + 1 * kPageSize + 42, &b1, 1));
    CHECK(arena.read(base + 3 * kPageSize + 99, &b3, 1));
    CHECK(b1 == 0x11);
    CHECK(b3 == 0x33);

    // Re-committing page 2 yields zero-initialized memory and allows cross-extent reads.
    CHECK(arena.commit(base + 2 * kPageSize, kPageSize, PageProtect::ReadWrite));
    CHECK(arena.committed_bytes() == 3 * kPageSize);
    uint8_t cross[16]{};
    CHECK(arena.read(base + 2 * kPageSize - 4, cross, 12));
    CHECK(cross[0] == 0x11 && cross[3] == 0x11);
    CHECK(cross[4] == 0x00 && cross[11] == 0x00);

    // VirtualFree MEM_RELEASE rejects non-zero size or interior address.
    CHECK(!arena.virtual_free(base, kPageSize, kMemRelease));
    CHECK(!arena.virtual_free(base + kPageSize, 0, kMemRelease));

    // VirtualFree MEM_DECOMMIT with size == 0 at base decommits entire reservation.
    CHECK(arena.virtual_free(base, 0, kMemDecommit));
    CHECK(arena.committed_bytes() == 0);
    CHECK(arena.reserved_bytes() == 128 * 1024);

    // Release succeeds and frees VA range for reuse.
    CHECK(arena.virtual_free(base, 0, kMemRelease));
    CHECK(arena.reserved_bytes() == 0);
    CHECK(arena.reservation_count() == 0);
    CHECK(arena.query(base).state == MemState::Free);

    // Unified virtual_alloc: MEM_RESERVE | MEM_COMMIT and hint overlap rejection.
    const uint64_t va1 = arena.virtual_alloc(
        base, 16 * 1024, kMemReserve | kMemCommit, PageProtect::ReadWrite);
    CHECK(va1 == base);
    CHECK(arena.committed_bytes() == 16 * 1024);
    // Overlapping within the same 64 KiB granularity block must fail.
    CHECK(arena.reserve(base + 32 * 1024, 4096) == 0);
    // Next 64 KiB boundary succeeds.
    const uint64_t va2 = arena.virtual_alloc(
        base + kAllocationGranularity, 8192, kMemReserve, PageProtect::NoAccess);
    CHECK(va2 == base + kAllocationGranularity);
    // Commit-only on existing reservation via virtual_alloc.
    const uint64_t committed_va = arena.virtual_alloc(
        va2 + kPageSize, kPageSize, kMemCommit, PageProtect::ReadWrite);
    CHECK(committed_va == va2 + kPageSize);

    CHECK(arena.release(va1));
    CHECK(arena.release(va2));
    CHECK(arena.reservation_count() == 0);
    CHECK(arena.reserved_bytes() == 0);
    CHECK(arena.committed_bytes() == 0);
}

// ---------------------------------------------------------------------------
// 2. TlsTable tests
// ---------------------------------------------------------------------------

void test_tls_table() {
    using namespace playgta6::os;

    TlsTable table;
    TlsTable other_table;

    const uint32_t slot = table.tls_alloc();
    CHECK(slot != TlsTable::kInvalidIndex);
    CHECK(table.tls_get(slot) == nullptr);

    int main_val = 42;
    CHECK(table.tls_set(slot, &main_val));
    CHECK(table.tls_get(slot) == &main_val);

    // Separate TlsTable instance is isolated.
    const uint32_t other_slot = other_table.tls_alloc();
    CHECK(other_slot == slot);
    CHECK(other_table.tls_get(other_slot) == nullptr);

    // Multi-threaded per-thread isolation.
    constexpr int kNumThreads = 6;
    std::atomic<int> passed_threads{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&table, slot, i, &passed_threads]() {
            int local_val = 1000 + i;
            if (table.tls_get(slot) != nullptr) {
                return;
            }
            if (!table.tls_set(slot, &local_val)) {
                return;
            }
            std::this_thread::yield();
            if (table.tls_get(slot) == &local_val) {
                passed_threads.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    CHECK(passed_threads.load() == kNumThreads);
    CHECK(table.tls_get(slot) == &main_val);

    // Freeing and reallocating a slot resets its value on all threads via generation bump.
    CHECK(table.tls_free(slot));
    CHECK(!table.tls_free(slot));  // Double-free returns false.
    CHECK(table.tls_get(slot) == nullptr);
    CHECK(!table.tls_set(slot, &main_val));

    const uint32_t reused = table.tls_alloc();
    CHECK(reused == slot);
    CHECK(table.tls_get(reused) == nullptr);
    CHECK(table.tls_free(reused));

    // Exhaust all 1024 slots.
    std::vector<uint32_t> all_slots;
    all_slots.reserve(TlsTable::kMaxSlots);
    for (uint32_t i = 0; i < TlsTable::kMaxSlots; ++i) {
        const uint32_t idx = table.tls_alloc();
        CHECK(idx != TlsTable::kInvalidIndex);
        all_slots.push_back(idx);
    }
    CHECK(table.allocated_count() == TlsTable::kMaxSlots);
    CHECK(table.tls_alloc() == TlsTable::kInvalidIndex);
    CHECK(table.tls_get(TlsTable::kMaxSlots) == nullptr);
    CHECK(!table.tls_set(TlsTable::kMaxSlots, &main_val));
    CHECK(!table.tls_free(TlsTable::kMaxSlots));

    for (uint32_t idx : all_slots) {
        CHECK(table.tls_free(idx));
    }
    CHECK(table.allocated_count() == 0);
}

// ---------------------------------------------------------------------------
// 3. SrwLock tests
// ---------------------------------------------------------------------------

void test_srw_lock() {
    using namespace playgta6::os;

    SrwLock lock;

    // Multiple shared acquisitions succeed simultaneously; exclusive fails while shared held.
    lock.lock_shared();
    CHECK(lock.try_lock_shared());
    CHECK(!lock.try_lock_exclusive());
    lock.unlock_shared();
    lock.unlock_shared();

    // Exclusive acquisition blocks both shared and exclusive try-locks.
    lock.lock_exclusive();
    CHECK(!lock.try_lock_shared());
    CHECK(!lock.try_lock_exclusive());
    lock.unlock_exclusive();

    // Standard lock wrappers work directly.
    {
        std::unique_lock<SrwLock> ul(lock);
        CHECK(!lock.try_lock_shared());
    }
    {
        std::shared_lock<SrwLock> sl(lock);
        CHECK(lock.try_lock_shared());
        lock.unlock_shared();
    }

    // Concurrent readers + writers stress test (invariant: a == b * 2).
    uint64_t shared_a = 0;
    uint64_t shared_b = 0;
    std::atomic<bool> stop{false};
    std::atomic<bool> invariant_broken{false};

    std::vector<std::thread> readers;
    for (int r = 0; r < 3; ++r) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_acquire)) {
                std::shared_lock<SrwLock> sl(lock);
                if (shared_a != shared_b * 2) {
                    invariant_broken.store(true, std::memory_order_release);
                }
            }
        });
    }

    constexpr uint64_t kWritesPerWriter = 1000;
    std::vector<std::thread> writers;
    for (int w = 0; w < 2; ++w) {
        writers.emplace_back([&]() {
            for (uint64_t i = 0; i < kWritesPerWriter; ++i) {
                std::unique_lock<SrwLock> ul(lock);
                ++shared_b;
                shared_a = shared_b * 2;
            }
        });
    }

    for (auto& w : writers) {
        w.join();
    }
    stop.store(true, std::memory_order_release);
    for (auto& r : readers) {
        r.join();
    }

    CHECK(!invariant_broken.load());
    CHECK(shared_b == 2 * kWritesPerWriter);
    CHECK(shared_a == 4 * kWritesPerWriter);
}

// ---------------------------------------------------------------------------
// 4. XTaskQueue tests
// ---------------------------------------------------------------------------

void test_xtask_queue() {
    using namespace playgta6::os;

    // 4a. Manual dispatch FIFO ordering for work and completion ports.
    {
        XTaskQueue q(XTaskQueue::DispatchMode::Manual, XTaskQueue::DispatchMode::Manual);
        std::vector<int> order;

        CHECK(q.submit_work([&]() { order.push_back(1); }));
        CHECK(q.submit_work([&]() {
            order.push_back(2);
            q.submit_completion([&]() { order.push_back(20); });
        }));
        CHECK(q.submit_completion([&]() { order.push_back(10); }));

        CHECK(q.pending_work_count() == 2);
        CHECK(q.pending_completion_count() == 1);

        CHECK(q.dispatch_work());
        CHECK(q.dispatch_work());
        CHECK(!q.dispatch_work());  // Empty now.
        CHECK(q.drain_completion() == 2);
        CHECK(!q.dispatch_completion(1));  // Timed out cleanly.

        CHECK(order.size() == 4);
        CHECK(order[0] == 1 && order[1] == 2 && order[2] == 10 && order[3] == 20);
    }

    // 4b. ThreadPool worker drain + manual completion dispatch.
    {
        XTaskQueue q(XTaskQueue::DispatchMode::ThreadPool,
                     XTaskQueue::DispatchMode::Manual,
                     2);
        std::atomic<int> work_done{0};
        std::atomic<int> completions_done{0};
        constexpr int kTasks = 32;

        for (int i = 0; i < kTasks; ++i) {
            CHECK(q.submit_work([&q, &work_done, &completions_done]() {
                work_done.fetch_add(1, std::memory_order_relaxed);
                q.submit_completion([&completions_done]() {
                    completions_done.fetch_add(1, std::memory_order_relaxed);
                });
            }));
        }

        // Wait for completions via timed dispatch.
        int dispatched = 0;
        while (dispatched < kTasks && q.dispatch_completion(2000)) {
            ++dispatched;
        }
        CHECK(dispatched == kTasks);
        CHECK(work_done.load() == kTasks);
        CHECK(completions_done.load() == kTasks);

        q.terminate(true, false);
        CHECK(q.is_terminated());
        CHECK(!q.submit_work([]() {}));
        CHECK(!q.submit_completion([]() {}));
    }

    // 4c. Cancellation on terminate(wait=true, cancel_pending=true).
    {
        XTaskQueue q(XTaskQueue::DispatchMode::Manual, XTaskQueue::DispatchMode::Manual);
        int simple_ran = 0;
        int canceled_count = 0;
        int normal_count = 0;

        CHECK(q.submit_work([&]() { ++simple_ran; }));
        CHECK(q.submit_work_cancellable([&](bool canceled) {
            if (canceled) {
                ++canceled_count;
            } else {
                ++normal_count;
            }
        }));
        CHECK(q.submit_completion_cancellable([&](bool canceled) {
            if (canceled) {
                ++canceled_count;
            }
        }));

        q.terminate(true, true);
        CHECK(simple_ran == 0);
        CHECK(normal_count == 0);
        CHECK(canceled_count == 2);
        CHECK(q.pending_work_count() == 0);
        CHECK(q.pending_completion_count() == 0);
    }

    // 4d. Immediate dispatch mode.
    {
        XTaskQueue q(XTaskQueue::DispatchMode::Immediate, XTaskQueue::DispatchMode::Immediate);
        int ran = 0;
        CHECK(q.submit_work([&]() { ran += 3; }));
        CHECK(q.submit_completion([&]() { ran += 7; }));
        CHECK(ran == 10);
    }
}

// ---------------------------------------------------------------------------
// 5. GameInputReader tests
// ---------------------------------------------------------------------------

void test_game_input_reader() {
    using namespace playgta6::os;

    // 5a. Deadzone normalization.
    {
        // Inside inner deadzone -> zero.
        Vec2 z = normalize_stick(0.10f, -0.08f, 0.15f, 0.98f);
        CHECK(z.x == 0.0f && z.y == 0.0f);

        // Diagonal preserves 45 deg direction and scales linearly from inner..outer.
        Vec2 d = normalize_stick(0.5f, 0.5f, 0.15f, 0.98f);
        CHECK(approx_eq(d.x, d.y));
        const float expected_mag = (std::hypot(0.5f, 0.5f) - 0.15f) / (0.98f - 0.15f);
        CHECK(approx_eq(std::hypot(d.x, d.y), expected_mag));

        // Beyond outer deadzone clamps to unit length.
        Vec2 c = normalize_stick(2.0f, 0.0f, 0.15f, 0.98f);
        CHECK(approx_eq(c.x, 1.0f) && approx_eq(c.y, 0.0f));

        // Non-finite inputs return zero safely.
        Vec2 nan_stick = normalize_stick(std::numeric_limits<float>::quiet_NaN(), 0.5f);
        CHECK(nan_stick.x == 0.0f && nan_stick.y == 0.0f);

        // Trigger normalization.
        CHECK(normalize_trigger(0.03f, 0.05f, 0.95f) == 0.0f);
        CHECK(approx_eq(normalize_trigger(0.50f, 0.05f, 0.95f), 0.5f));
        CHECK(normalize_trigger(1.20f, 0.05f, 0.95f) == 1.0f);
        CHECK(normalize_trigger(std::numeric_limits<float>::quiet_NaN()) == 0.0f);
    }

    // 5b. Button edges + mouse delta atomic consumption (MonotonicSeq mode matching input_bridge.js).
    {
        playgta6::InputState shared{};
        GameInputReader reader(&shared, SeqlockMode::MonotonicSeq);
        CHECK(reader.is_attached());

        playgta6::InputState frame1{};
        frame1.buttons = input_button::kMoveForward | input_button::kJump;
        frame1.mouse_dx = 12;
        frame1.mouse_dy = -5;
        frame1.left_stick_x = 0.0f;
        frame1.left_stick_y = 0.98f;
        frame1.right_trigger = 0.98f;
        GameInputReader::write_monotonic(shared, frame1, true);

        // Simulate a second mousemove accumulation before polling.
        playgta6::InputState frame1_extra = frame1;
        frame1_extra.mouse_dx = 8;
        frame1_extra.mouse_dy = -3;
        GameInputReader::write_monotonic(shared, frame1_extra, true);

        GameInputFrame polled{};
        CHECK(reader.poll(polled, true));
        CHECK(polled.consistent);
        CHECK(polled.seq == 2);
        CHECK(is_button_down(polled.buttons, input_button::kMoveForward));
        CHECK(is_button_down(polled.buttons, input_button::kJump));
        CHECK(polled.buttons_pressed == (input_button::kMoveForward | input_button::kJump));
        CHECK(polled.buttons_released == 0);
        CHECK(polled.mouse_delta.dx == 20);
        CHECK(polled.mouse_delta.dy == -8);
        CHECK(approx_eq(polled.left_stick.y, 1.0f));
        CHECK(approx_eq(polled.right_trigger, 1.0f));

        // Mouse delta was atomically zeroed on consumption.
        MouseDelta empty_delta = reader.consume_mouse_delta();
        CHECK(empty_delta.dx == 0 && empty_delta.dy == 0);

        // Release Jump, press Sprint.
        playgta6::InputState frame2 = frame1;
        frame2.buttons = input_button::kMoveForward | input_button::kSprint;
        frame2.mouse_dx = 0;
        frame2.mouse_dy = 0;
        GameInputReader::write_monotonic(shared, frame2, false);

        CHECK(reader.poll(polled, true));
        CHECK(polled.buttons_pressed == input_button::kSprint);
        CHECK(polled.buttons_released == input_button::kJump);
    }

    // 5c. EvenOddSeqlock detects odd (in-progress) write and guarantees untorn snapshots under concurrency.
    {
        playgta6::InputState shared{};
        shared.seq = 3;  // Odd = writer currently mutating fields.
        GameInputReader reader(&shared, SeqlockMode::EvenOddSeqlock);

        playgta6::InputState snap{};
        uint32_t retries = 0;
        CHECK(!reader.read_snapshot(snap, 8, &retries));
        CHECK(retries == 8);

        // Reset and run concurrent writer + reader seqlock stress test.
        shared = playgta6::InputState{};
        std::atomic<bool> writer_done{false};
        std::atomic<bool> torn_detected{false};
        std::atomic<uint64_t> consistent_reads{0};

        std::thread writer([&]() {
            for (uint32_t i = 1; i <= 20000; ++i) {
                playgta6::InputState v{};
                v.buttons = i;
                v.mouse_dx = static_cast<int32_t>(i);
                v.mouse_dy = -static_cast<int32_t>(i);
                v.left_stick_x = static_cast<float>(i);
                v.left_stick_y = static_cast<float>(i);
                v.right_stick_x = static_cast<float>(i);
                v.right_stick_y = static_cast<float>(i);
                v.left_trigger = static_cast<float>(i);
                v.right_trigger = static_cast<float>(i);
                GameInputReader::write_seqlock(shared, v, false);
            }
            writer_done.store(true, std::memory_order_release);
        });

        while (!writer_done.load(std::memory_order_acquire) || consistent_reads.load() == 0) {
            playgta6::InputState s{};
            if (reader.read_snapshot(s, 128)) {
                consistent_reads.fetch_add(1, std::memory_order_relaxed);
                const float expected = static_cast<float>(s.buttons);
                if (s.mouse_dx != static_cast<int32_t>(s.buttons) ||
                    s.mouse_dy != -static_cast<int32_t>(s.buttons) ||
                    s.left_stick_x != expected ||
                    s.right_trigger != expected) {
                    torn_detected.store(true, std::memory_order_release);
                }
            }
        }
        writer.join();

        CHECK(!torn_detected.load());
        CHECK(consistent_reads.load() > 0);
    }
}

// ---------------------------------------------------------------------------
// 6. HttpfsMailbox & HttpfsMailboxClient tests
// ---------------------------------------------------------------------------

struct MockHttpfsServer {
    std::unordered_map<std::string, std::vector<uint8_t>> files;
    std::unordered_map<int32_t, std::string> open_handles;
    int32_t next_handle = 1;

    void handle_request(playgta6::os::HttpfsMailbox& mb) {
        using namespace playgta6::os;
        const auto op = static_cast<HttpfsOp>(mb.op);
        if (op == HttpfsOp::Open) {
            const std::string path(
                reinterpret_cast<const char*>(mb.payload),
                static_cast<size_t>(mb.length));
            if (files.find(path) == files.end()) {
                HttpfsMailboxClient::complete_response(mb, -2, HttpfsStatus::Error);
                return;
            }
            const int32_t handle = next_handle++;
            open_handles.emplace(handle, path);
            HttpfsMailboxClient::complete_response(mb, handle, HttpfsStatus::ResponseReady);
            return;
        }
        if (op == HttpfsOp::Read) {
            auto it = open_handles.find(mb.handle);
            if (it == open_handles.end()) {
                HttpfsMailboxClient::complete_response(mb, -9, HttpfsStatus::Error);
                return;
            }
            const auto& data = files.at(it->second);
            if (mb.offset >= data.size()) {
                HttpfsMailboxClient::complete_response(mb, 0, HttpfsStatus::ResponseReady);
                return;
            }
            const size_t avail = data.size() - static_cast<size_t>(mb.offset);
            const size_t cap = mb.payload_capacity == 0 ? kHttpfsPayloadBytes : mb.payload_capacity;
            const size_t to_copy = std::min<size_t>({avail, static_cast<size_t>(mb.length), cap});
            std::memcpy(mb.payload, data.data() + static_cast<size_t>(mb.offset), to_copy);
            HttpfsMailboxClient::complete_response(
                mb, static_cast<int64_t>(to_copy), HttpfsStatus::ResponseReady);
            return;
        }
        if (op == HttpfsOp::Close) {
            const size_t erased = open_handles.erase(mb.handle);
            if (erased == 0) {
                HttpfsMailboxClient::complete_response(mb, -9, HttpfsStatus::Error);
            } else {
                HttpfsMailboxClient::complete_response(mb, 0, HttpfsStatus::ResponseReady);
            }
            return;
        }
        HttpfsMailboxClient::complete_response(mb, -1, HttpfsStatus::Error);
    }
};

void test_httpfs_mailbox() {
    using namespace playgta6::os;

    // 6a. Synchronous injectable responder with forced multi-chunk transfers (64-byte capacity).
    {
        auto mb = std::make_unique<HttpfsMailbox>();
        HttpfsMailboxClient::init_mailbox(*mb, 64);

        MockHttpfsServer server;
        std::vector<uint8_t> blob(300);
        for (size_t i = 0; i < blob.size(); ++i) {
            blob[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xFFu);
        }
        server.files.emplace("data/archive.rpf", blob);

        HttpfsMailboxClient client(mb.get(), [&server](HttpfsMailbox& m) {
            server.handle_request(m);
        });
        CHECK(client.payload_capacity() == 64);

        // Missing file returns < 0.
        CHECK(client.open("data/does_not_exist.rpf") < 0);

        // Valid open.
        const int32_t handle = client.open("data/archive.rpf");
        CHECK(handle >= 0);

        // Full multi-chunk read (300 bytes across 5 x 64-byte chunks).
        std::vector<uint8_t> out(300, 0);
        const int64_t n = client.read(handle, 0, out.data(), out.size());
        CHECK(n == 300);
        CHECK(out == blob);

        // Offset range read + EOF clamping.
        std::vector<uint8_t> tail(100, 0);
        const int64_t tail_n = client.read(handle, 250, tail.data(), tail.size());
        CHECK(tail_n == 50);
        CHECK(std::memcmp(tail.data(), blob.data() + 250, 50) == 0);

        // Read strictly past EOF returns 0.
        uint8_t dummy = 0;
        CHECK(client.read(handle, 999, &dummy, 1) == 0);

        // Close and verify subsequent reads/closes fail.
        CHECK(client.close(handle));
        CHECK(!client.close(handle));
        CHECK(client.read(handle, 0, &dummy, 1) < 0);
    }

    // 6b. Asynchronous worker-to-worker Atomics.wait / Atomics.notify roundtrip.
    {
        auto mb = std::make_unique<HttpfsMailbox>();
        HttpfsMailboxClient::init_mailbox(*mb, 128);

        MockHttpfsServer server;
        const std::string payload_str = "playgta6web-httpfs-range-streaming-test-payload!";
        server.files.emplace(
            "boot/manifest.json",
            std::vector<uint8_t>(payload_str.begin(), payload_str.end()));

        std::thread io_worker([&]() {
            while (HttpfsMailboxClient::wait_for_request(*mb)) {
                server.handle_request(*mb);
            }
        });

        // No synchronous responder callback: client communicates purely via atomic wait/notify.
        HttpfsMailboxClient client(mb.get());

        CHECK(client.open("boot/missing.json") < 0);
        const int32_t handle = client.open("boot/manifest.json");
        CHECK(handle >= 0);

        std::string read_back(payload_str.size(), '\0');
        const int64_t got = client.read(handle, 0, read_back.data(), read_back.size());
        CHECK(got == static_cast<int64_t>(payload_str.size()));
        CHECK(read_back == payload_str);

        std::string sub(5, '\0');
        CHECK(client.read(handle, 12, sub.data(), sub.size()) == 5);
        CHECK(sub == "httpf");

        CHECK(client.close(handle));

        HttpfsMailboxClient::shutdown_waiter(*mb);
        io_worker.join();
    }
}

}  // namespace

int main() {
    test_virtual_memory_arena();
    test_tls_table();
    test_srw_lock();
    test_xtask_queue();
    test_game_input_reader();
    test_httpfs_mailbox();

    if (g_failures != 0) {
        std::fprintf(stderr, "os_shim_tests failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All os_shim_tests passed.\n");
    return 0;
}
