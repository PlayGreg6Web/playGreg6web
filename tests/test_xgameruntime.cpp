#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "os/httpfs_mailbox.h"
#include "os/threading.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xgameruntime.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_xgameruntime.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

// ---------------------------------------------------------------------------
// 1. Virtual memory Win32 shims
// ---------------------------------------------------------------------------

void test_win32_virtual_memory() {
    using namespace playgta6::os;
    using namespace playgta6::os::win32;

    VirtualMemoryArena custom_arena;
    set_vmem_arena(&custom_arena);

    // Reserve 64 KiB via C++ shim.
    const uint64_t base = VirtualAlloc(0, 64 * 1024, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(base != 0);
    CHECK((base % kAllocationGranularity) == 0);

    MemoryBasicInformation mbi{};
    CHECK(VirtualQuery(base, &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.base_address == base);
    CHECK(mbi.allocation_base == base);
    CHECK(mbi.region_size == 64 * 1024);
    CHECK(mbi.state == MEM_RESERVE);
    CHECK(mbi.protect == PAGE_NOACCESS);
    CHECK(mbi.type == MEM_PRIVATE);

    // Commit 2 pages via C-ABI entrypoint.
    const uint64_t committed = pg6_win32_virtual_alloc(
        base + kPageSize, 2 * kPageSize, MEM_COMMIT, PAGE_READWRITE);
    CHECK(committed == base + kPageSize);

    const uint32_t pattern = 0xDEADBEEFu;
    CHECK(custom_arena.write(committed, &pattern, sizeof(pattern)));
    uint32_t read_back = 0;
    CHECK(custom_arena.read(committed, &read_back, sizeof(read_back)));
    CHECK(read_back == pattern);

    CHECK(pg6_win32_virtual_query(committed, &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.state == MEM_COMMIT);
    CHECK(mbi.protect == PAGE_READWRITE);
    CHECK(mbi.region_size == 2 * kPageSize);

    // VirtualProtect transitions + old_protect reporting.
    uint32_t old_prot = 0;
    CHECK(VirtualProtect(committed, kPageSize, PAGE_READONLY, &old_prot));
    CHECK(old_prot == PAGE_READWRITE);
    CHECK(!custom_arena.write(committed, &pattern, sizeof(pattern)));
    CHECK(custom_arena.read(committed, &read_back, sizeof(read_back)));
    CHECK(read_back == pattern);

    CHECK(pg6_win32_virtual_protect(committed, kPageSize, PAGE_EXECUTE_READ, &old_prot) == 1);
    CHECK(old_prot == PAGE_READONLY);

    // Invalid parameters set ERROR_INVALID_PARAMETER.
    CHECK(VirtualAlloc(0, 0, MEM_COMMIT, PAGE_READWRITE) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(VirtualQuery(base, &mbi, 4) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    // Decommit and release.
    CHECK(VirtualFree(committed, 2 * kPageSize, MEM_DECOMMIT));
    CHECK(custom_arena.committed_bytes() == 0);
    CHECK(pg6_win32_virtual_free(base, 0, MEM_RELEASE) == 1);
    CHECK(custom_arena.reservation_count() == 0);

    CHECK(VirtualQuery(base, &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.state == MEM_FREE);
    CHECK(mbi.type == 0);

    set_vmem_arena(nullptr);
}

// ---------------------------------------------------------------------------
// 2. TLS & thread-local GetLastError / SetLastError
// ---------------------------------------------------------------------------

void test_win32_tls_and_last_error() {
    using namespace playgta6::os;
    using namespace playgta6::os::win32;

    TlsTable custom_tls;
    set_tls_table(&custom_tls);

    const uint32_t slot = pg6_win32_tls_alloc();
    CHECK(slot != TLS_OUT_OF_INDEXES);

    int main_cookie = 777;
    SetLastError(ERROR_ACCESS_DENIED);
    CHECK(GetLastError() == ERROR_ACCESS_DENIED);

    CHECK(pg6_win32_tls_set_value(slot, &main_cookie) == 1);
    // Win32 TlsGetValue clears LastError on valid slot.
    CHECK(pg6_win32_tls_get_value(slot) == &main_cookie);
    CHECK(GetLastError() == ERROR_SUCCESS);

    SetLastError(ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_get_last_error() == ERROR_FILE_NOT_FOUND);

    constexpr int kThreads = 6;
    std::atomic<int> passed{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        workers.emplace_back([slot, i, &passed]() {
            // Fresh thread starts with ERROR_SUCCESS and nullptr TLS slot.
            if (GetLastError() != ERROR_SUCCESS) {
                return;
            }
            if (TlsGetValue(slot) != nullptr) {
                return;
            }
            const uint32_t thread_err = 100u + static_cast<uint32_t>(i);
            SetLastError(thread_err);
            int local_cookie = 500 + i;
            if (!TlsSetValue(slot, &local_cookie)) {
                return;
            }
            std::this_thread::yield();
            if (GetLastError() != thread_err) {
                return;
            }
            if (TlsGetValue(slot) != &local_cookie) {
                return;
            }
            if (GetLastError() != ERROR_SUCCESS) {
                return;
            }
            passed.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& w : workers) {
        w.join();
    }

    CHECK(passed.load() == kThreads);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);
    CHECK(TlsGetValue(slot) == &main_cookie);

    CHECK(pg6_win32_tls_free(slot) == 1);
    CHECK(!TlsFree(slot));
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    set_tls_table(nullptr);
    SetLastError(ERROR_SUCCESS);
}

// ---------------------------------------------------------------------------
// 3. SRWLock, CriticalSection, and manual/auto-reset Events
// ---------------------------------------------------------------------------

void test_win32_sync_primitives() {
    using namespace playgta6::os::win32;

    // 3a. SRWLock (shared/exclusive + C-ABI).
    {
        SrwLock srw{};
        pg6_win32_initialize_srw_lock(&srw);

        AcquireSRWLockShared(&srw);
        CHECK(pg6_win32_try_acquire_srw_lock_shared(&srw) == 1);
        CHECK(!TryAcquireSRWLockExclusive(&srw));
        ReleaseSRWLockShared(&srw);
        pg6_win32_release_srw_lock_shared(&srw);

        pg6_win32_acquire_srw_lock_exclusive(&srw);
        CHECK(pg6_win32_try_acquire_srw_lock_shared(&srw) == 0);
        CHECK(pg6_win32_try_acquire_srw_lock_exclusive(&srw) == 0);
        pg6_win32_release_srw_lock_exclusive(&srw);
    }

    // 3b. CriticalSection (recursive re-entry + cross-thread mutual exclusion).
    {
        CriticalSection cs{};
        pg6_win32_initialize_critical_section(&cs);

        EnterCriticalSection(&cs);
        CHECK(pg6_win32_try_enter_critical_section(&cs) == 1);

        // Another thread cannot enter while held.
        std::atomic<bool> other_entered{false};
        std::thread probe([&]() {
            other_entered.store(TryEnterCriticalSection(&cs), std::memory_order_release);
        });
        probe.join();
        CHECK(!other_entered.load());

        // Still held after leaving once (recursive depth was 2).
        LeaveCriticalSection(&cs);
        std::thread probe2([&]() {
            other_entered.store(TryEnterCriticalSection(&cs), std::memory_order_release);
        });
        probe2.join();
        CHECK(!other_entered.load());

        // Released after second leave.
        pg6_win32_leave_critical_section(&cs);

        // Concurrent counter increment under CriticalSection.
        int shared_counter = 0;
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&]() {
                for (int i = 0; i < 500; ++i) {
                    EnterCriticalSection(&cs);
                    ++shared_counter;
                    LeaveCriticalSection(&cs);
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }
        CHECK(shared_counter == 2000);

        pg6_win32_delete_critical_section(&cs);
    }

    // 3c. Manual-reset Event: timeout, broadcast wake, persistent signaled state, ResetEvent, named alias.
    {
        const Handle ev = CreateEventW(nullptr, true, false, u"pg6_test_manual_event");
        CHECK(ev != kNullHandle && ev != INVALID_HANDLE_VALUE);
        CHECK(GetLastError() == ERROR_SUCCESS);

        // Named duplicate opens same underlying event.
        const Handle ev_dup = CreateEventA(nullptr, true, false, "pg6_test_manual_event");
        CHECK(ev_dup != kNullHandle && ev_dup != ev);
        CHECK(GetLastError() == ERROR_ALREADY_EXISTS);

        // Finite timeout before signal.
        CHECK(WaitForSingleObject(ev, 20) == WAIT_TIMEOUT);

        constexpr int kWaiters = 4;
        std::atomic<int> woken{0};
        std::vector<std::thread> waiters;
        for (int i = 0; i < kWaiters; ++i) {
            waiters.emplace_back([ev, &woken]() {
                if (WaitForSingleObject(ev, INFINITE_TIMEOUT) == WAIT_OBJECT_0) {
                    woken.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        CHECK(woken.load() == 0);

        // Signaling via duplicate handle wakes all waiters on original handle.
        CHECK(SetEvent(ev_dup));
        for (auto& w : waiters) {
            w.join();
        }
        CHECK(woken.load() == kWaiters);

        // Manual-reset stays signaled until ResetEvent.
        CHECK(WaitForSingleObject(ev, 0) == WAIT_OBJECT_0);
        CHECK(ResetEvent(ev));
        CHECK(WaitForSingleObject(ev, 0) == WAIT_TIMEOUT);

        CHECK(CloseHandle(ev_dup));
        CHECK(CloseHandle(ev));
        CHECK(!CloseHandle(ev));
        CHECK(GetLastError() == ERROR_INVALID_HANDLE);
        CHECK(WaitForSingleObject(ev, 0) == WAIT_FAILED);
    }

    // 3d. Auto-reset Event: wakes exactly one waiter and resets automatically.
    {
        const Handle ev = pg6_win32_create_event_a(nullptr, 0, 0, nullptr);
        CHECK(ev != kNullHandle);

        CHECK(pg6_win32_set_event(ev) == 1);
        CHECK(pg6_win32_wait_for_single_object(ev, 0) == WAIT_OBJECT_0);
        // Second immediate wait times out because event auto-reset.
        CHECK(pg6_win32_wait_for_single_object(ev, 0) == WAIT_TIMEOUT);

        // Delayed wake from worker thread.
        std::thread waker([ev]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            SetEvent(ev);
        });
        CHECK(WaitForSingleObject(ev, 2000) == WAIT_OBJECT_0);
        waker.join();

        CHECK(pg6_win32_close_handle(ev) == 1);
    }
}

// ---------------------------------------------------------------------------
// 4. File I/O shims (mock HttpfsMailbox worker thread + /userdata overlay)
// ---------------------------------------------------------------------------

struct MockHttpfsWorkerServer {
    std::unordered_map<std::string, std::vector<uint8_t>> files;
    std::unordered_map<int32_t, std::string> open_handles;
    int32_t next_handle = 1;
    std::atomic<int> close_count{0};

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
            const int32_t h = next_handle++;
            open_handles.emplace(h, path);
            HttpfsMailboxClient::complete_response(mb, h, HttpfsStatus::ResponseReady);
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
                close_count.fetch_add(1, std::memory_order_relaxed);
                HttpfsMailboxClient::complete_response(mb, 0, HttpfsStatus::ResponseReady);
            }
            return;
        }
        HttpfsMailboxClient::complete_response(mb, -1, HttpfsStatus::Error);
    }
};

void test_win32_file_io() {
    using namespace playgta6::os;
    using namespace playgta6::os::win32;

    // 4a. Path normalization checks.
    {
        bool is_ud = false;
        CHECK(normalize_path("\\\\?\\G:\\update\\x64\\dlc.rpf", &is_ud) == "update/x64/dlc.rpf");
        CHECK(!is_ud);
        CHECK(normalize_path("D:\\boot\\manifest.json", &is_ud) == "boot/manifest.json");
        CHECK(!is_ud);
        CHECK(normalize_path("T:\\userdata\\slot0\\save.dat", &is_ud) == "/userdata/slot0/save.dat");
        CHECK(is_ud);
        CHECK(normalize_path("/userdata/profile.bin", &is_ud) == "/userdata/profile.bin");
        CHECK(is_ud);
        CHECK(utf16_to_utf8(u"\\\\?\\G:\\data\\file.bin") == "\\\\?\\G:\\data\\file.bin");
    }

    // 4b. Mock HttpfsMailbox worker thread round-trip via CreateFileW/ReadFile/SetFilePointerEx/GetFileSizeEx.
    {
        auto mb = std::make_unique<HttpfsMailbox>();
        HttpfsMailboxClient::init_mailbox(*mb, 64);  // Force multi-chunk transfers.

        MockHttpfsWorkerServer server;
        std::vector<uint8_t> rpf_blob(300);
        for (size_t i = 0; i < rpf_blob.size(); ++i) {
            rpf_blob[i] = static_cast<uint8_t>((i * 29u + 7u) & 0xFFu);
        }
        server.files.emplace("update/x64/dlc.rpf", rpf_blob);
        server.files.emplace("empty.bin", std::vector<uint8_t>{});
        server.files.emplace("one.bin", std::vector<uint8_t>{0x5Au});
        server.files.emplace("exact64.bin", std::vector<uint8_t>(64, 0xC3u));

        std::thread worker([&]() {
            while (HttpfsMailboxClient::wait_for_request(*mb)) {
                server.handle_request(*mb);
            }
        });

        HttpfsMailboxClient client(mb.get());
        set_httpfs_client(&client);

        // Missing file fails with ERROR_FILE_NOT_FOUND.
        CHECK(CreateFileW(u"G:\\missing\\file.rpf", GENERIC_READ, FILE_SHARE_READ,
                          nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL) == INVALID_HANDLE_VALUE);
        CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);

        // Write attempt on read-only HTTPFS path fails with ERROR_ACCESS_DENIED.
        CHECK(CreateFileW(u"G:\\update\\x64\\dlc.rpf", GENERIC_WRITE, 0,
                          nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL) == INVALID_HANDLE_VALUE);
        CHECK(GetLastError() == ERROR_ACCESS_DENIED);

        // Open with \\?\G:\ prefix via UTF-16 CreateFileW.
        const Handle h = CreateFileW(u"\\\\?\\G:\\update\\x64\\dlc.rpf",
                                     GENERIC_READ,
                                     FILE_SHARE_READ,
                                     nullptr,
                                     OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL);
        CHECK(h != INVALID_HANDLE_VALUE);

        int64_t fsize = -1;
        CHECK(GetFileSizeEx(h, &fsize));
        CHECK(fsize == 300);

        // Full read.
        std::vector<uint8_t> out(300, 0);
        uint32_t bytes_read = 0;
        CHECK(ReadFile(h, out.data(), static_cast<uint32_t>(out.size()), &bytes_read));
        CHECK(bytes_read == 300);
        CHECK(out == rpf_blob);

        // Read at EOF returns true with 0 bytes read.
        uint8_t eof_byte = 0;
        CHECK(ReadFile(h, &eof_byte, 1, &bytes_read));
        CHECK(bytes_read == 0);

        // Seek relative to FILE_BEGIN, FILE_CURRENT, and FILE_END.
        int64_t new_pos = -1;
        CHECK(SetFilePointerEx(h, 100, &new_pos, FILE_BEGIN));
        CHECK(new_pos == 100);
        uint8_t slice[16]{};
        CHECK(ReadFile(h, slice, sizeof(slice), &bytes_read));
        CHECK(bytes_read == sizeof(slice));
        CHECK(std::memcmp(slice, rpf_blob.data() + 100, sizeof(slice)) == 0);

        CHECK(SetFilePointerEx(h, -8, &new_pos, FILE_CURRENT));
        CHECK(new_pos == 108);
        uint8_t b108 = 0;
        CHECK(ReadFile(h, &b108, 1, &bytes_read));
        CHECK(bytes_read == 1 && b108 == rpf_blob[108]);

        CHECK(SetFilePointerEx(h, -10, &new_pos, FILE_END));
        CHECK(new_pos == 290);
        uint8_t tail[20]{};
        CHECK(ReadFile(h, tail, sizeof(tail), &bytes_read));
        CHECK(bytes_read == 10);
        CHECK(std::memcmp(tail, rpf_blob.data() + 290, 10) == 0);

        // Negative seek fails cleanly.
        CHECK(!SetFilePointerEx(h, -1, &new_pos, FILE_BEGIN));
        CHECK(GetLastError() == ERROR_NEGATIVE_SEEK);

        // WriteFile on read-only HTTPFS handle fails.
        uint32_t written = 0;
        CHECK(!WriteFile(h, tail, 4, &written));
        CHECK(GetLastError() == ERROR_ACCESS_DENIED);

        CHECK(CloseHandle(h));
        CHECK(server.close_count.load() == 1);

        // Verify edge-case file sizes (0, 1, 64 bytes) via GetFileSizeEx.
        const Handle h_empty = CreateFileA("D:\\empty.bin", GENERIC_READ, FILE_SHARE_READ,
                                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
        CHECK(h_empty != INVALID_HANDLE_VALUE);
        CHECK(GetFileSizeEx(h_empty, &fsize) && fsize == 0);
        CHECK(CloseHandle(h_empty));

        const Handle h_one = CreateFileA("D:\\one.bin", GENERIC_READ, FILE_SHARE_READ,
                                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
        CHECK(h_one != INVALID_HANDLE_VALUE);
        CHECK(GetFileSizeEx(h_one, &fsize) && fsize == 1);
        CHECK(CloseHandle(h_one));

        const Handle h_64 = CreateFileA("D:\\exact64.bin", GENERIC_READ, FILE_SHARE_READ,
                                        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
        CHECK(h_64 != INVALID_HANDLE_VALUE);
        CHECK(GetFileSizeEx(h_64, &fsize) && fsize == 64);
        CHECK(CloseHandle(h_64));

        set_httpfs_client(nullptr);
        HttpfsMailboxClient::shutdown_waiter(*mb);
        worker.join();
    }

    // 4c. /userdata overlay WriteFile / ReadFile / Seek / Size round-trip (C++ & C-ABI).
    {
        clear_userdata_overlay();

        const Handle wf = pg6_win32_create_file_w(
            u"T:\\userdata\\saves\\slot0.sav",
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            kNullHandle);
        CHECK(wf != INVALID_HANDLE_VALUE);

        const std::string header = "GTA6SAVE_V1";
        uint32_t written = 0;
        CHECK(pg6_win32_write_file(
                  wf, header.data(), static_cast<uint32_t>(header.size()), &written, nullptr) == 1);
        CHECK(written == header.size());

        // Overwrite a slice at offset 4.
        int64_t pos = -1;
        CHECK(pg6_win32_set_file_pointer_ex(wf, 4, &pos, FILE_BEGIN) == 1);
        CHECK(pos == 4);
        const char patch[4] = {'W', 'E', 'B', '!'};
        CHECK(WriteFile(wf, patch, 4, &written));
        CHECK(written == 4);

        int64_t sz = -1;
        CHECK(pg6_win32_get_file_size_ex(wf, &sz) == 1);
        CHECK(sz == static_cast<int64_t>(header.size()));
        CHECK(pg6_win32_close_handle(wf) == 1);

        // Reopen with Unix-style /userdata path and verify round-trip.
        const Handle rf = CreateFileA("/userdata/saves/slot0.sav",
                                      GENERIC_READ,
                                      FILE_SHARE_READ,
                                      nullptr,
                                      OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL);
        CHECK(rf != INVALID_HANDLE_VALUE);
        std::string read_str(header.size(), '\0');
        uint32_t bytes_read = 0;
        CHECK(pg6_win32_read_file(
                  rf, read_str.data(), static_cast<uint32_t>(read_str.size()), &bytes_read, nullptr) == 1);
        CHECK(bytes_read == header.size());
        CHECK(read_str == "GTA6WEB!_V1");
        CHECK(CloseHandle(rf));

        // CREATE_NEW rejects existing save file.
        CHECK(CreateFileA("/userdata/saves/slot0.sav", GENERIC_WRITE, 0,
                          nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL) == INVALID_HANDLE_VALUE);
        CHECK(GetLastError() == ERROR_FILE_EXISTS);

        // TRUNCATE_EXISTING clears file to 0 bytes.
        const Handle tf = CreateFileA("/userdata/saves/slot0.sav",
                                      GENERIC_READ | GENERIC_WRITE,
                                      0,
                                      nullptr,
                                      TRUNCATE_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL);
        CHECK(tf != INVALID_HANDLE_VALUE);
        CHECK(GetFileSizeEx(tf, &sz) && sz == 0);
        CHECK(CloseHandle(tf));

        clear_userdata_overlay();
    }

    // 4d. ReadFile / WriteFile with an OVERLAPPED transfer at its Offset (not the file pointer),
    // fill Internal/InternalHigh and signal hEvent; a read at the end fails with ERROR_HANDLE_EOF.
    {
        clear_userdata_overlay();
        struct Overlapped {
            uint64_t internal;
            uint64_t internal_high;
            uint32_t offset;
            uint32_t offset_high;
            Handle event;
        };
        const Handle f = CreateFileA("/userdata/overlapped.bin", GENERIC_READ | GENERIC_WRITE, 0,
                                     nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL);
        CHECK(f != INVALID_HANDLE_VALUE);
        uint32_t n = 0;
        CHECK(WriteFile(f, "0123456789abcdef", 16, &n) && n == 16);  // file pointer now 16

        const Handle ev = CreateEventW(nullptr, true, false, nullptr);
        Overlapped ov{~0ull, ~0ull, 4, 0, ev};
        char buf[8] = {};
        CHECK(ReadFile(f, buf, 4, &n, &ov));
        CHECK(n == 4 && std::memcmp(buf, "4567", 4) == 0);
        CHECK(ov.internal == 0 && ov.internal_high == 4);
        CHECK(WaitForSingleObject(ev, 0) == WAIT_OBJECT_0);

        // Without an OVERLAPPED the read continues at the pointer the overlapped read left.
        CHECK(ReadFile(f, buf, 2, &n) && n == 2 && std::memcmp(buf, "89", 2) == 0);

        Overlapped wov{~0ull, ~0ull, 2, 0, kNullHandle};
        CHECK(WriteFile(f, "XY", 2, &n, &wov) && n == 2);
        CHECK(wov.internal == 0 && wov.internal_high == 2);
        Overlapped rov{0, 0, 0, 0, kNullHandle};
        CHECK(ReadFile(f, buf, 6, &n, &rov) && n == 6 && std::memcmp(buf, "01XY45", 6) == 0);

        Overlapped eov{~0ull, ~0ull, 16, 0, kNullHandle};
        CHECK(!ReadFile(f, buf, 4, &n, &eov));
        CHECK(GetLastError() == ERROR_HANDLE_EOF && n == 0);
        CHECK(eov.internal == 0xC0000011u && eov.internal_high == 0);

        CHECK(CloseHandle(ev));
        CHECK(CloseHandle(f));
        clear_userdata_overlay();
    }
}

// ---------------------------------------------------------------------------
// 5. Timing (QPC monotonicity, GetTickCount64, GetSystemTimeAsFileTime)
// ---------------------------------------------------------------------------

void test_win32_timing() {
    using namespace playgta6::os::win32;

    int64_t freq = 0;
    CHECK(pg6_win32_query_performance_frequency(&freq) == 1);
    CHECK(freq > 0);

    int64_t prev = 0;
    CHECK(QueryPerformanceCounter(&prev));
    CHECK(prev > 0);

    for (int i = 0; i < 64; ++i) {
        int64_t cur = 0;
        CHECK(pg6_win32_query_performance_counter(&cur) == 1);
        CHECK(cur >= prev);
        prev = cur;
    }

    const int64_t before_sleep = prev;
    const uint64_t tick0 = pg6_win32_get_tick_count64();
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    int64_t after_sleep = 0;
    CHECK(QueryPerformanceCounter(&after_sleep));
    CHECK(after_sleep > before_sleep);
    const uint64_t tick1 = GetTickCount64();
    CHECK(tick1 >= tick0);

    FileTime ft{};
    pg6_win32_get_system_time_as_file_time(&ft);
    const uint64_t ft64 =
        (static_cast<uint64_t>(ft.high_date_time) << 32) | static_cast<uint64_t>(ft.low_date_time);
    // 2020-01-01T00:00:00Z in Win32 FILETIME 100ns ticks = 132223104000000000.
    CHECK(ft64 > 132223104000000000ULL);
}

// ---------------------------------------------------------------------------
// 6. XGameRuntime lifecycle & XSystem info
// ---------------------------------------------------------------------------

void test_gdk_runtime_and_system() {
    using namespace playgta6::os::gdk;

    CHECK(!XGameRuntimeIsInitialized());
    CHECK(pg6_gdk_xgame_runtime_initialize() == S_OK_HR);
    CHECK(pg6_gdk_xgame_runtime_is_initialized() == 1);
    CHECK(XGameRuntimeIsInitialized());

    const XSystemDeviceType dev = XSystemGetDeviceType();
    CHECK(dev == XSystemDeviceType::XboxScarlettAnaconda ||
          dev == XSystemDeviceType::XboxScarlettLockhart);
    CHECK(pg6_gdk_xsystem_get_device_type() == static_cast<uint32_t>(dev));

    const XSystemAnalyticsInfo info = pg6_gdk_xsystem_get_analytics_info();
    CHECK(info.os_version.major == 10);
    CHECK(std::strcmp(info.family, "Xbox") == 0);
    CHECK(std::strcmp(info.form, "Xbox Series X") == 0 ||
          std::strcmp(info.form, "Xbox Series S") == 0);

    pg6_gdk_xgame_runtime_uninitialize();
    CHECK(!XGameRuntimeIsInitialized());
}

// ---------------------------------------------------------------------------
// 7. XTaskQueue C-ABI handle lifecycle
// ---------------------------------------------------------------------------

void test_gdk_xtask_queue() {
    using namespace playgta6::os::gdk;

    // 7a. Manual dispatch FIFO + terminate cancellation.
    {
        XTaskQueueHandle queue = nullptr;
        CHECK(XTaskQueueCreate(
                  XTaskQueueDispatchMode::Manual, XTaskQueueDispatchMode::Manual, &queue) ==
              S_OK_HR);
        CHECK(queue != nullptr);

        struct CallbackState {
            std::vector<int> log;
            int canceled_count = 0;
            bool terminated_called = false;
        } st;

        auto work_cb = [](void* ctx, bool canceled) {
            auto* s = static_cast<CallbackState*>(ctx);
            if (canceled) {
                ++s->canceled_count;
            } else {
                s->log.push_back(1);
            }
        };
        auto comp_cb = [](void* ctx, bool canceled) {
            auto* s = static_cast<CallbackState*>(ctx);
            if (canceled) {
                ++s->canceled_count;
            } else {
                s->log.push_back(2);
            }
        };
        auto term_cb = [](void* ctx) {
            static_cast<CallbackState*>(ctx)->terminated_called = true;
        };

        CHECK(XTaskQueueSubmitCallback(queue, XTaskQueuePort::Work, &st, work_cb) == S_OK_HR);
        CHECK(XTaskQueueSubmitCallback(queue, XTaskQueuePort::Completion, &st, comp_cb) == S_OK_HR);
        CHECK(XTaskQueueDispatch(queue, XTaskQueuePort::Work, 0));
        CHECK(XTaskQueueDispatch(queue, XTaskQueuePort::Completion, 0));
        CHECK(!XTaskQueueDispatch(queue, XTaskQueuePort::Work, 0));
        CHECK(st.log.size() == 2 && st.log[0] == 1 && st.log[1] == 2);

        // Enqueue two pending items and terminate -> both receive canceled=true.
        CHECK(XTaskQueueSubmitCallback(queue, XTaskQueuePort::Work, &st, work_cb) == S_OK_HR);
        CHECK(XTaskQueueSubmitCallback(queue, XTaskQueuePort::Completion, &st, comp_cb) == S_OK_HR);

        CHECK(XTaskQueueTerminate(queue, true, &st, term_cb) == S_OK_HR);
        CHECK(st.terminated_called);
        CHECK(st.canceled_count == 2);

        // Submissions after termination fail with E_ABORT_HR.
        CHECK(XTaskQueueSubmitCallback(queue, XTaskQueuePort::Work, &st, work_cb) == E_ABORT_HR);

        XTaskQueueCloseHandle(queue);
    }

    // 7b. ThreadPool work + Manual completion via C-ABI pg6_gdk_* functions.
    {
        Pg6XTaskQueueHandle queue = nullptr;
        CHECK(pg6_gdk_xtask_queue_create(
                  static_cast<uint32_t>(XTaskQueueDispatchMode::ThreadPool),
                  static_cast<uint32_t>(XTaskQueueDispatchMode::Manual),
                  &queue) == S_OK_HR);
        CHECK(queue != nullptr);

        struct PoolCtx {
            Pg6XTaskQueueHandle q = nullptr;
            std::atomic<int> work_ran{0};
            std::atomic<int> comp_ran{0};
        } ctx{queue, 0, 0};

        auto on_work = [](void* raw, bool canceled) {
            if (!canceled) {
                auto* c = static_cast<PoolCtx*>(raw);
                c->work_ran.fetch_add(1, std::memory_order_relaxed);
                pg6_gdk_xtask_queue_submit_callback(
                    c->q,
                    static_cast<uint32_t>(XTaskQueuePort::Completion),
                    c,
                    [](void* inner, bool comp_canceled) {
                        if (!comp_canceled) {
                            static_cast<PoolCtx*>(inner)->comp_ran.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    });
            }
        };

        constexpr int kCount = 8;
        for (int i = 0; i < kCount; ++i) {
            CHECK(pg6_gdk_xtask_queue_submit_callback(
                      queue, static_cast<uint32_t>(XTaskQueuePort::Work), &ctx, on_work) ==
                  S_OK_HR);
        }

        int dispatched = 0;
        while (dispatched < kCount &&
               pg6_gdk_xtask_queue_dispatch(
                   queue, static_cast<uint32_t>(XTaskQueuePort::Completion), 2000) == 1) {
            ++dispatched;
        }
        CHECK(dispatched == kCount);
        CHECK(ctx.work_ran.load() == kCount);
        CHECK(ctx.comp_ran.load() == kCount);

        CHECK(pg6_gdk_xtask_queue_terminate(queue, 1, nullptr, nullptr) == S_OK_HR);
        pg6_gdk_xtask_queue_close_handle(queue);
    }

    // 7c. Process task queue accessor.
    {
        XTaskQueueHandle proc_q = nullptr;
        CHECK(pg6_gdk_xtask_queue_get_current_process_task_queue(&proc_q) == 1);
        CHECK(proc_q != nullptr);
        XTaskQueueCloseHandle(proc_q);
    }
}

// ---------------------------------------------------------------------------
// 8. Offline XUser + XAsyncBlock flow
// ---------------------------------------------------------------------------

void test_gdk_xuser_offline() {
    using namespace playgta6::os::gdk;

    CHECK(XGameRuntimeInitialize() == S_OK_HR);

    // 8a. Manual XTaskQueue step-by-step async flow.
    {
        XTaskQueueHandle queue = nullptr;
        CHECK(XTaskQueueCreate(
                  XTaskQueueDispatchMode::Manual, XTaskQueueDispatchMode::Manual, &queue) ==
              S_OK_HR);

        bool callback_fired = false;
        XAsyncBlock async{};
        async.queue = queue;
        async.context = &callback_fired;
        async.callback = [](XAsyncBlock* blk) {
            *static_cast<bool*>(blk->context) = true;
        };

        CHECK(XUserAddAsync(XUserAddOptions::AddDefaultUserSilently, &async) == S_OK_HR);
        CHECK(XAsyncGetStatus(&async, false) == E_PENDING_HR);

        size_t res_size = 999;
        CHECK(XAsyncGetResultSize(&async, &res_size) == E_PENDING_HR);

        // Dispatch Work -> Completion.
        CHECK(XTaskQueueDispatch(queue, XTaskQueuePort::Work, 0));
        CHECK(!callback_fired);
        CHECK(XTaskQueueDispatch(queue, XTaskQueuePort::Completion, 0));
        CHECK(callback_fired);

        CHECK(XAsyncGetStatus(&async, false) == S_OK_HR);
        CHECK(XAsyncGetResultSize(&async, &res_size) == S_OK_HR);
        CHECK(res_size == sizeof(XUserHandle));

        XUserHandle user = nullptr;
        CHECK(XUserAddResult(&async, &user) == S_OK_HR);
        CHECK(user != nullptr);

        // Verify deterministic offline XUID and Gamertag.
        uint64_t xuid = 0;
        CHECK(XUserGetId(user, &xuid) == S_OK_HR);
        CHECK(xuid == 0x0009000000000001ULL);

        char small_buf[4]{};
        size_t used = 0;
        CHECK(XUserGetGamertag(
                  user, XUserGamertagComponent::Classic, sizeof(small_buf), small_buf, &used) ==
              E_NOT_SUFFICIENT_BUFFER_HR);
        CHECK(used == std::strlen("OfflinePlayer") + 1);

        char tag[64]{};
        CHECK(XUserGetGamertag(
                  user, XUserGamertagComponent::Classic, sizeof(tag), tag, &used) == S_OK_HR);
        CHECK(std::strcmp(tag, "OfflinePlayer") == 0);
        CHECK(used == std::strlen("OfflinePlayer") + 1);

        CHECK(XUserGetGamertag(
                  user, XUserGamertagComponent::Modern, sizeof(tag), tag, &used) == S_OK_HR);
        CHECK(std::strcmp(tag, "OfflinePlayer") == 0);

        CHECK(XUserGetGamertag(
                  user, XUserGamertagComponent::ModernSuffix, sizeof(tag), tag, &used) == S_OK_HR);
        CHECK(std::strcmp(tag, "") == 0);
        CHECK(used == 1);

        // Status/result size remain readable after result extraction.
        CHECK(XAsyncGetStatus(&async, false) == S_OK_HR);
        CHECK(XAsyncGetResultSize(&async, &res_size) == S_OK_HR);
        CHECK(res_size == sizeof(XUserHandle));

        XUserCloseHandle(user);
        XTaskQueueCloseHandle(queue);
    }

    // 8b. Default process task queue (async.queue == nullptr) via C-ABI pg6_gdk_* functions.
    {
        std::atomic<bool> cb_done{false};
        Pg6XAsyncBlock async{};
        async.queue = nullptr;
        async.context = &cb_done;
        async.callback = [](Pg6XAsyncBlock* blk) {
            static_cast<std::atomic<bool>*>(blk->context)->store(true, std::memory_order_release);
        };

        CHECK(pg6_gdk_xuser_add_async(
                  static_cast<uint32_t>(XUserAddOptions::AddDefaultUserSilently), &async) ==
              S_OK_HR);
        CHECK(pg6_gdk_xasync_get_status(&async, 1) == S_OK_HR);
        CHECK(cb_done.load(std::memory_order_acquire));

        size_t res_size = 0;
        CHECK(pg6_gdk_xasync_get_result_size(&async, &res_size) == S_OK_HR);
        CHECK(res_size == sizeof(Pg6XUserHandle));

        Pg6XUserHandle user = nullptr;
        CHECK(pg6_gdk_xuser_add_result(&async, &user) == S_OK_HR);
        CHECK(user != nullptr);

        uint64_t xuid = 0;
        CHECK(pg6_gdk_xuser_get_id(user, &xuid) == S_OK_HR);
        CHECK(xuid == kOfflineUserXuid);

        char tag[32]{};
        size_t used = 0;
        CHECK(pg6_gdk_xuser_get_gamertag(
                  user,
                  static_cast<uint32_t>(XUserGamertagComponent::UniqueModern),
                  sizeof(tag),
                  tag,
                  &used) == S_OK_HR);
        CHECK(std::strcmp(tag, kOfflineUserGamertag) == 0);

        pg6_gdk_xuser_close_handle(user);
    }

    // 8c. Canceled queue cleanly aborts pending XUserAddAsync.
    {
        XTaskQueueHandle queue = nullptr;
        CHECK(XTaskQueueCreate(
                  XTaskQueueDispatchMode::Manual, XTaskQueueDispatchMode::Manual, &queue) ==
              S_OK_HR);

        XAsyncBlock async{};
        async.queue = queue;
        CHECK(XUserAddAsync(XUserAddOptions::AddDefaultUserSilently, &async) == S_OK_HR);
        CHECK(XTaskQueueTerminate(queue, true) == S_OK_HR);
        CHECK(XAsyncGetStatus(&async, true) == E_ABORT_HR);

        XUserHandle user = nullptr;
        CHECK(XUserAddResult(&async, &user) == E_ABORT_HR);
        CHECK(user == nullptr);

        XTaskQueueCloseHandle(queue);
    }

    XGameRuntimeUninitialize();
    CHECK(!XGameRuntimeIsInitialized());
}

}  // namespace

int main() {
    test_win32_virtual_memory();
    test_win32_tls_and_last_error();
    test_win32_sync_primitives();
    test_win32_file_io();
    test_win32_timing();
    test_gdk_runtime_and_system();
    test_gdk_xtask_queue();
    test_gdk_xuser_offline();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_xgameruntime failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_xgameruntime_tests passed.\n");
    return 0;
}
