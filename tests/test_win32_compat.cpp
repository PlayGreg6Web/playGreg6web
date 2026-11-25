#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "os/guest_window.h"
#include "os/win32_compat.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/win32_sync.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_win32_compat.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::os::win32;

constexpr uint32_t kInvalidAttrs = 0xFFFFFFFFu;

bool write_file(const char16_t* path, const std::string& data) {
    const Handle h = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    uint32_t written = 0;
    const bool ok = WriteFile(h, data.data(), static_cast<uint32_t>(data.size()), &written) &&
                    written == data.size();
    CloseHandle(h);
    return ok;
}

std::u16string name_of(const Pg6Win32FindDataW& d) {
    return std::u16string(d.file_name);
}

std::vector<std::u16string> find_all(const char16_t* pattern, uint32_t* err = nullptr) {
    std::vector<std::u16string> names;
    Pg6Win32FindDataW d;
    const Handle h = pg6_win32_find_first_file_w(pattern, &d);
    if (h == INVALID_HANDLE_VALUE) {
        if (err != nullptr) {
            *err = GetLastError();
        }
        return names;
    }
    names.push_back(name_of(d));
    while (pg6_win32_find_next_file_w(h, &d)) {
        names.push_back(name_of(d));
    }
    if (err != nullptr) {
        *err = GetLastError();
    }
    CHECK(pg6_win32_find_close(h) == 1);
    return names;
}

// ---------------------------------------------------------------------------
// File system
// ---------------------------------------------------------------------------

void test_directories_and_attributes() {
    clear_userdata_overlay();
    // The root exists; missing things report the right error.
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata") == FILE_ATTRIBUTE_DIRECTORY);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\nope") == kInvalidAttrs);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\a\\b") == kInvalidAttrs);
    CHECK(GetLastError() == ERROR_PATH_NOT_FOUND);
    CHECK(pg6_win32_get_file_attributes_w(nullptr) == kInvalidAttrs);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_get_file_attributes_w(u"") == kInvalidAttrs);
    CHECK(GetLastError() == ERROR_PATH_NOT_FOUND);
    CHECK(pg6_win32_get_file_attributes_a("G:\\userdata\\nope") == kInvalidAttrs);
    CHECK(pg6_win32_get_file_attributes_a(nullptr) == kInvalidAttrs);

    // CreateDirectory: parent must exist, duplicates are ALREADY_EXISTS.
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\a\\b", nullptr) == 0);
    CHECK(GetLastError() == ERROR_PATH_NOT_FOUND);
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\a", nullptr) == 1);
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\a", nullptr) == 0);
    CHECK(GetLastError() == ERROR_ALREADY_EXISTS);
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\a\\b", nullptr) == 1);
    CHECK(pg6_win32_create_directory_w(nullptr, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_create_directory_w(u"G:\\gamedata\\x", nullptr) == 0);  // read-only data
    CHECK(GetLastError() == ERROR_ACCESS_DENIED);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\a\\b") == FILE_ATTRIBUTE_DIRECTORY);
    CHECK(pg6_win32_get_file_attributes_w(u"g:/UserData/a/b/") == FILE_ATTRIBUTE_DIRECTORY);

    // A file created by CreateFileW/WriteFile is visible to the compat calls.
    CHECK(write_file(u"G:\\userdata\\a\\b\\save.bin", "hello world"));
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\a\\b\\save.bin") ==
          FILE_ATTRIBUTE_ARCHIVE);
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\a\\b\\save.bin", nullptr) == 0);
    CHECK(GetLastError() == ERROR_ALREADY_EXISTS);
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\a\\b\\save.bin\\c", nullptr) == 0);
    CHECK(GetLastError() == ERROR_PATH_NOT_FOUND);

    Pg6Win32FileAttributeData data;
    std::memset(static_cast<void*>(&data), 0xAB, sizeof(data));
    CHECK(pg6_win32_get_file_attributes_ex_w(u"G:\\userdata\\a\\b\\save.bin", 0, &data) == 1);
    CHECK(data.file_size_low == 11 && data.file_size_high == 0);
    CHECK(data.file_attributes == FILE_ATTRIBUTE_ARCHIVE);
    CHECK(file_time_ticks(data.last_write_time) > kFileTimeUnixEpochTicks);
    CHECK(pg6_win32_get_file_attributes_ex_w(u"G:\\userdata\\a\\none", 0, &data) == 0);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);
    CHECK(data.file_size_low == 0 && data.file_attributes == 0);  // zeroed on failure
    CHECK(pg6_win32_get_file_attributes_ex_w(u"G:\\userdata\\a", 1, &data) == 0);  // bad level
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_get_file_attributes_ex_w(u"G:\\userdata\\a", 0, nullptr) == 0);

    // Attributes: read-only blocks deletion; NORMAL clears; directories keep DIRECTORY.
    CHECK(pg6_win32_set_file_attributes_w(u"G:\\userdata\\a\\b\\save.bin",
                                          FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN) == 1);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\a\\b\\save.bin") ==
          (FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN));
    CHECK(pg6_win32_delete_file_w(u"G:\\userdata\\a\\b\\save.bin") == 0);
    CHECK(GetLastError() == ERROR_ACCESS_DENIED);
    CHECK(pg6_win32_set_file_attributes_w(u"G:\\userdata\\a\\b\\save.bin", FILE_ATTRIBUTE_NORMAL) == 1);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\a\\b\\save.bin") == FILE_ATTRIBUTE_NORMAL);
    CHECK(pg6_win32_set_file_attributes_w(u"G:\\userdata\\a\\b", FILE_ATTRIBUTE_HIDDEN) == 1);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\a\\b") ==
          (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_HIDDEN));
    CHECK(pg6_win32_set_file_attributes_w(u"G:\\userdata\\zzz", FILE_ATTRIBUTE_HIDDEN) == 0);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);

    // RemoveDirectory / DeleteFile error codes.
    CHECK(pg6_win32_remove_directory_w(u"G:\\userdata\\a\\b") == 0);
    CHECK(GetLastError() == ERROR_DIR_NOT_EMPTY);
    CHECK(pg6_win32_remove_directory_w(u"G:\\userdata\\a\\b\\save.bin") == 0);
    CHECK(GetLastError() == ERROR_DIRECTORY);
    CHECK(pg6_win32_delete_file_w(u"G:\\userdata\\a\\b") == 0);
    CHECK(GetLastError() == ERROR_ACCESS_DENIED);
    CHECK(pg6_win32_delete_file_w(u"G:\\userdata\\a\\b\\save.bin") == 1);
    CHECK(pg6_win32_delete_file_w(u"G:\\userdata\\a\\b\\save.bin") == 0);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_delete_file_w(u"G:\\userdata\\q\\save.bin") == 0);
    CHECK(GetLastError() == ERROR_PATH_NOT_FOUND);
    CHECK(pg6_win32_remove_directory_w(u"G:\\userdata\\a\\b") == 1);
    CHECK(pg6_win32_remove_directory_w(u"G:\\userdata\\a\\b") == 0);
    CHECK(GetLastError() == ERROR_PATH_NOT_FOUND);
    CHECK(pg6_win32_remove_directory_w(u"G:\\userdata") == 0);
    CHECK(GetLastError() == ERROR_ACCESS_DENIED);
    CHECK(pg6_win32_remove_directory_w(u"G:\\userdata\\a") == 1);
    clear_userdata_overlay();
}

void test_find_files() {
    clear_userdata_overlay();
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\saves", nullptr) == 1);
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\saves\\sub", nullptr) == 1);
    CHECK(write_file(u"G:\\userdata\\saves\\Slot1.sav", "aaaa"));
    CHECK(write_file(u"G:\\userdata\\saves\\slot2.SAV", "bb"));
    CHECK(write_file(u"G:\\userdata\\saves\\notes.txt", ""));
    CHECK(write_file(u"G:\\userdata\\saves\\noext", "x"));

    uint32_t err = 0;
    auto all = find_all(u"G:\\userdata\\saves\\*", &err);
    CHECK(err == ERROR_NO_MORE_FILES);
    const std::vector<std::u16string> expect_all = {u".",       u"..",      u"Slot1.sav",
                                                    u"noext",   u"notes.txt", u"slot2.SAV",
                                                    u"sub"};
    CHECK(all == expect_all);

    // "*.*" matches names without an extension too; wildcards are case-insensitive.
    CHECK(find_all(u"G:\\userdata\\saves\\*.*").size() == expect_all.size());
    const std::vector<std::u16string> sav = {u"Slot1.sav", u"slot2.SAV"};
    CHECK(find_all(u"G:\\userdata\\saves\\*.sav") == sav);
    CHECK(find_all(u"G:\\userdata\\saves\\SLOT?.sav") == sav);
    CHECK(find_all(u"G:\\userdata\\saves\\slot1.*").size() == 1);
    CHECK(find_all(u"G:\\userdata\\saves\\n*").size() == 2);
    CHECK(find_all(u"G:\\userdata\\saves\\s?b").size() == 1);
    CHECK(find_all(u"G:\\userdata\\saves\\*zzz*", &err).empty());
    CHECK(err == ERROR_FILE_NOT_FOUND);
    CHECK(find_all(u"G:\\userdata\\missing\\*", &err).empty());
    CHECK(err == ERROR_PATH_NOT_FOUND);
    CHECK(find_all(u"G:\\gamedata\\*", &err).empty());  // the HTTPFS protocol cannot list
    CHECK(err == ERROR_PATH_NOT_FOUND);

    // The root has no "." / ".." entries.
    const auto root = find_all(u"G:\\userdata\\*");
    CHECK(root.size() == 1 && root[0] == u"saves");

    // Exact names: one entry, with size and attributes filled in.
    Pg6Win32FindDataW d;
    std::memset(static_cast<void*>(&d), 0xCD, sizeof(d));
    Handle h = pg6_win32_find_first_file_w(u"G:\\userdata\\saves\\Slot1.sav", &d);
    CHECK(h != INVALID_HANDLE_VALUE);
    CHECK(name_of(d) == u"Slot1.sav");
    CHECK(d.file_size_low == 4 && d.file_size_high == 0);
    CHECK(d.file_attributes == FILE_ATTRIBUTE_ARCHIVE);
    CHECK(d.alternate_file_name[0] == 0 && d.reserved0 == 0 && d.reserved1 == 0);
    CHECK(pg6_win32_find_next_file_w(h, &d) == 0);
    CHECK(GetLastError() == ERROR_NO_MORE_FILES);
    CHECK(pg6_win32_find_close(h) == 1);
    CHECK(pg6_win32_find_close(h) == 0);  // already closed
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_find_next_file_w(h, &d) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    h = pg6_win32_find_first_file_w(u"G:\\userdata\\saves\\sub", &d);
    CHECK(h != INVALID_HANDLE_VALUE && d.file_attributes == FILE_ATTRIBUTE_DIRECTORY);
    pg6_win32_find_close(h);
    CHECK(pg6_win32_find_first_file_w(u"G:\\userdata\\saves\\nofile.txt", &d) ==
          INVALID_HANDLE_VALUE);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_find_first_file_w(u"G:\\userdata\\saves\\", &d) == INVALID_HANDLE_VALUE);
    CHECK(pg6_win32_find_first_file_w(nullptr, &d) == INVALID_HANDLE_VALUE);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_find_first_file_w(u"G:\\userdata\\*", nullptr) == INVALID_HANDLE_VALUE);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_find_close(0) == 0);
    CHECK(pg6_win32_find_close(INVALID_HANDLE_VALUE) == 0);

    // A very long name is truncated to MAX_PATH-1 characters, never overflowing.
    const std::string long_name(300, 'x');
    std::u16string long_path = u"G:\\userdata\\saves\\" + std::u16string(300, u'x');
    CHECK(write_file(long_path.c_str(), "1"));
    h = pg6_win32_find_first_file_w((u"G:\\userdata\\saves\\" + std::u16string(300, u'x')).c_str(), &d);
    CHECK(h != INVALID_HANDLE_VALUE && name_of(d).size() == 259);
    pg6_win32_find_close(h);
    clear_userdata_overlay();
}

void test_file_times_and_handles() {
    clear_userdata_overlay();
    CHECK(write_file(u"G:\\userdata\\t.bin", "0123456789"));
    const Handle h = CreateFileW(u"G:\\userdata\\t.bin", GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
    CHECK(h != INVALID_HANDLE_VALUE);

    Pg6FileTime c, a, w;
    CHECK(pg6_win32_get_file_time(h, &c, &a, &w) == 1);
    CHECK(file_time_ticks(c) != 0 && file_time_ticks(w) >= file_time_ticks(c));
    const Pg6FileTime fixed = file_time_from_ticks(kFileTimeUnixEpochTicks + 1000 * kFileTimeTicksPerSecond);
    CHECK(pg6_win32_set_file_time(h, nullptr, nullptr, &fixed) == 1);
    CHECK(pg6_win32_get_file_time(h, nullptr, nullptr, &w) == 1);
    CHECK(file_time_ticks(w) == file_time_ticks(fixed));
    // 0 means "no change".
    const Pg6FileTime zero{};
    CHECK(pg6_win32_set_file_time(h, &zero, &zero, &zero) == 1);
    CHECK(pg6_win32_get_file_time(h, nullptr, nullptr, &w) == 1);
    CHECK(file_time_ticks(w) == file_time_ticks(fixed));
    Pg6Win32FileAttributeData data;
    CHECK(pg6_win32_get_file_attributes_ex_w(u"G:\\userdata\\t.bin", 0, &data) == 1);
    CHECK(file_time_ticks(data.last_write_time) == file_time_ticks(fixed));
    // Writing updates the last-write time.
    uint32_t n = 0;
    CHECK(WriteFile(h, "z", 1, &n) && n == 1);
    CHECK(pg6_win32_get_file_time(h, nullptr, nullptr, &w) == 1);
    CHECK(file_time_ticks(w) > file_time_ticks(fixed));

    // FlushFileBuffers needs write access.
    CHECK(pg6_win32_flush_file_buffers(h) == 1);
    const Handle ro = CreateFileW(u"G:\\userdata\\t.bin", GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL);
    CHECK(pg6_win32_flush_file_buffers(ro) == 0);
    CHECK(GetLastError() == ERROR_ACCESS_DENIED);
    CHECK(pg6_win32_set_file_time(ro, nullptr, nullptr, &fixed) == 0);
    CHECK(GetLastError() == ERROR_ACCESS_DENIED);
    CHECK(pg6_win32_set_end_of_file(ro) == 0);
    CloseHandle(ro);

    // SetEndOfFile truncates and zero-extends at the file pointer.
    int64_t pos = 0;
    CHECK(SetFilePointerEx(h, 4, &pos, FILE_BEGIN));
    CHECK(pg6_win32_set_end_of_file(h) == 1);
    int64_t size = 0;
    CHECK(GetFileSizeEx(h, &size) && size == 4);
    CHECK(SetFilePointerEx(h, 8, &pos, FILE_BEGIN));
    CHECK(pg6_win32_set_end_of_file(h) == 1);
    CHECK(GetFileSizeEx(h, &size) && size == 8);
    std::vector<uint8_t> bytes;
    CHECK(read_userdata_file("G:\\userdata\\t.bin", bytes));
    CHECK(bytes.size() == 8 && bytes[3] == '3' && bytes[4] == 0 && bytes[7] == 0);

    // Bad handles and null arguments.
    CHECK(pg6_win32_get_file_time(0x7777, &c, &a, &w) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(file_time_ticks(c) == 0);  // out parameters are initialised on failure
    CHECK(pg6_win32_set_file_time(INVALID_HANDLE_VALUE, nullptr, nullptr, nullptr) == 0);
    CHECK(pg6_win32_flush_file_buffers(kNullHandle) == 0);
    CHECK(pg6_win32_set_end_of_file(0x7777) == 0);
    const Handle ev = CreateEventW(nullptr, true, false, nullptr);
    CHECK(pg6_win32_get_file_time(ev, &c, &a, &w) == 0);  // not a file handle
    CloseHandle(ev);
    CloseHandle(h);
    clear_userdata_overlay();
}

void test_move_and_disk_space() {
    clear_userdata_overlay();
    CHECK(pg6_win32_create_directory_w(u"G:\\userdata\\d", nullptr) == 1);
    CHECK(write_file(u"G:\\userdata\\d\\a.txt", "A"));
    CHECK(write_file(u"G:\\userdata\\b.txt", "B"));
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\d\\a.txt", u"G:\\userdata\\c.txt", 0) == 1);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\d\\a.txt") == kInvalidAttrs);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\c.txt") != kInvalidAttrs);
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\c.txt", u"G:\\userdata\\b.txt", 0) == 0);
    CHECK(GetLastError() == ERROR_ALREADY_EXISTS);
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\c.txt", u"G:\\userdata\\b.txt",
                                   MOVEFILE_REPLACE_EXISTING) == 1);
    std::vector<uint8_t> bytes;
    CHECK(read_userdata_file("G:\\userdata\\b.txt", bytes) && bytes.size() == 1 && bytes[0] == 'A');
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\none", u"G:\\userdata\\x", 0) == 0);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\b.txt", u"G:\\userdata\\no\\x", 0) == 0);
    CHECK(GetLastError() == ERROR_PATH_NOT_FOUND);
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\b.txt", u"G:\\gamedata\\x", 0) == 0);
    CHECK(GetLastError() == ERROR_ACCESS_DENIED);
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\b.txt", nullptr, 0) == 0);
    CHECK(pg6_win32_move_file_ex_w(nullptr, u"G:\\userdata\\z", 0) == 0);
    // Directories move with their contents.
    CHECK(write_file(u"G:\\userdata\\d\\in.txt", "I"));
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\d", u"G:\\userdata\\e", 0) == 1);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\e\\in.txt") != kInvalidAttrs);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\userdata\\d") == kInvalidAttrs);
    CHECK(pg6_win32_move_file_ex_w(u"G:\\userdata\\e", u"G:\\userdata\\e\\f", 0) == 0);

    uint64_t avail = 1, total = 1, free_bytes = 1;
    CHECK(pg6_win32_get_disk_free_space_ex_w(u"G:\\userdata", &avail, &total, &free_bytes) == 1);
    CHECK(total > 0 && avail > 0 && avail <= total && free_bytes == avail);
    CHECK(pg6_win32_get_disk_free_space_ex_w(nullptr, &avail, nullptr, nullptr) == 1);
    CHECK(pg6_win32_get_disk_free_space_ex_w(u"G:\\userdata\\missing", &avail, &total, &free_bytes) == 0);
    CHECK(GetLastError() == ERROR_PATH_NOT_FOUND);
    CHECK(avail == 0 && total == 0 && free_bytes == 0);
    clear_userdata_overlay();
}

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

void test_time() {
    // 2021-03-04 05:06:07.089 UTC.
    Pg6SystemTime st{2021, 3, 0, 4, 5, 6, 7, 89};
    Pg6FileTime ft;
    CHECK(pg6_win32_system_time_to_file_time(&st, &ft) == 1);
    const uint64_t unix_secs = 1614834367ull;
    CHECK(file_time_ticks(ft) ==
          kFileTimeUnixEpochTicks + unix_secs * kFileTimeTicksPerSecond + 89 * 10000ull);
    Pg6SystemTime back;
    CHECK(pg6_win32_file_time_to_system_time(&ft, &back) == 1);
    CHECK(back.year == 2021 && back.month == 3 && back.day == 4 && back.hour == 5 &&
          back.minute == 6 && back.second == 7 && back.milliseconds == 89);
    CHECK(back.day_of_week == 4);  // Thursday

    // Epoch and boundaries.
    Pg6SystemTime epoch{1601, 1, 0, 1, 0, 0, 0, 0};
    CHECK(pg6_win32_system_time_to_file_time(&epoch, &ft) == 1 && file_time_ticks(ft) == 0);
    CHECK(pg6_win32_file_time_to_system_time(&ft, &back) == 1 && back.year == 1601 &&
          back.day_of_week == 1);  // Monday
    Pg6SystemTime unix_epoch{1970, 1, 0, 1, 0, 0, 0, 0};
    CHECK(pg6_win32_system_time_to_file_time(&unix_epoch, &ft) == 1 &&
          file_time_ticks(ft) == kFileTimeUnixEpochTicks);
    Pg6SystemTime leap{2000, 2, 0, 29, 23, 59, 59, 999};
    CHECK(pg6_win32_system_time_to_file_time(&leap, &ft) == 1);
    CHECK(pg6_win32_file_time_to_system_time(&ft, &back) == 1 && back.month == 2 && back.day == 29);
    Pg6SystemTime bad = leap;
    bad.year = 1900;  // not a leap year
    CHECK(pg6_win32_system_time_to_file_time(&bad, &ft) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(file_time_ticks(ft) == 0);
    for (int field = 0; field < 6; ++field) {
        bad = leap;
        switch (field) {
            case 0: bad.month = 13; break;
            case 1: bad.day = 0; break;
            case 2: bad.hour = 24; break;
            case 3: bad.minute = 60; break;
            case 4: bad.second = 60; break;
            default: bad.milliseconds = 1000; break;
        }
        CHECK(pg6_win32_system_time_to_file_time(&bad, &ft) == 0);
    }
    bad = leap;
    bad.year = 1600;
    CHECK(pg6_win32_system_time_to_file_time(&bad, &ft) == 0);
    CHECK(pg6_win32_system_time_to_file_time(nullptr, &ft) == 0);
    CHECK(pg6_win32_system_time_to_file_time(&leap, nullptr) == 0);
    const Pg6FileTime huge = file_time_from_ticks(0x8000000000000000ULL);
    CHECK(pg6_win32_file_time_to_system_time(&huge, &back) == 0);
    CHECK(back.year == 0);
    CHECK(pg6_win32_file_time_to_system_time(nullptr, &back) == 0);

    // Local time is UTC.
    Pg6FileTime local;
    CHECK(pg6_win32_file_time_to_local_file_time(&ft, &local) == 1 &&
          file_time_ticks(local) == file_time_ticks(ft));
    Pg6FileTime utc;
    CHECK(pg6_win32_local_file_time_to_file_time(&local, &utc) == 1 &&
          file_time_ticks(utc) == file_time_ticks(ft));
    CHECK(pg6_win32_file_time_to_local_file_time(&huge, &local) == 0);
    CHECK(pg6_win32_file_time_to_local_file_time(nullptr, &local) == 0);
    CHECK(pg6_win32_file_time_to_local_file_time(&ft, nullptr) == 0);

    // Current time functions agree with each other and with GetSystemTimeAsFileTime.
    Pg6SystemTime now_sys, now_local;
    pg6_win32_get_system_time(&now_sys);
    pg6_win32_get_local_time(&now_local);
    CHECK(now_sys.year >= 2024 && now_sys.month >= 1 && now_sys.month <= 12);
    CHECK(now_local.year == now_sys.year);
    Pg6FileTime a, b;
    GetSystemTimeAsFileTime(&a);
    pg6_win32_get_system_time_precise_as_file_time(&b);
    CHECK(file_time_ticks(b) >= file_time_ticks(a));
    pg6_win32_get_system_time(nullptr);  // no crash

    Pg6TimeZoneInformation tz;
    std::memset(&tz, 0xFF, sizeof(tz));
    CHECK(pg6_win32_get_time_zone_information(&tz) == 0);  // TIME_ZONE_ID_UNKNOWN
    CHECK(tz.bias == 0 && tz.standard_bias == 0 && tz.daylight_bias == 0);
    CHECK(tz.standard_date.month == 0 && tz.daylight_date.month == 0);
    CHECK(tz.standard_name[0] == u'C');
    CHECK(pg6_win32_get_time_zone_information(nullptr) == 0xFFFFFFFFu);

    // Date / time formatting (en-US).
    char16_t out[80];
    Pg6SystemTime when{2021, 3, 0, 4, 15, 6, 7, 0};
    CHECK(pg6_win32_get_date_format_ex(u"en-US", 1, &when, nullptr, out, 80, nullptr) == 9);
    CHECK(std::u16string(out) == u"3/4/2021");
    CHECK(pg6_win32_get_date_format_ex(nullptr, 2, &when, nullptr, out, 80, nullptr) > 0);
    CHECK(std::u16string(out) == u"Thursday, March 4, 2021");
    CHECK(pg6_win32_get_date_format_ex(u"", 0, &when, u"ddd dd-MMM-yy 'on' MMMM", out, 80, nullptr) > 0);
    CHECK(std::u16string(out) == u"Thu 04-Mar-21 on March");
    CHECK(pg6_win32_get_date_format_ex(u"en-US", 0, &when, u"'it''s' d", out, 80, nullptr) > 0);
    CHECK(std::u16string(out) == u"it's 4");
    CHECK(pg6_win32_get_date_format_ex(u"en-US", 1, &when, nullptr, nullptr, 0, nullptr) == 9);  // size query
    CHECK(pg6_win32_get_date_format_ex(u"en-US", 1, &when, nullptr, out, 4, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
    CHECK(pg6_win32_get_date_format_ex(u"en-US", 1, &when, u"d", out, 80, nullptr) == 0);  // flags+format
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);
    CHECK(pg6_win32_get_date_format_ex(u"de-DE", 0, &when, nullptr, out, 80, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_get_date_format_ex(u"en-US", 0, &when, nullptr, nullptr, 5, nullptr) == 0);
    Pg6SystemTime bad_date = when;
    bad_date.month = 0;
    CHECK(pg6_win32_get_date_format_ex(u"en-US", 0, &bad_date, nullptr, out, 80, nullptr) == 0);

    CHECK(pg6_win32_get_time_format_ex(u"en-US", 0, &when, nullptr, out, 80) > 0);
    CHECK(std::u16string(out) == u"3:06:07 PM");
    CHECK(pg6_win32_get_time_format_ex(nullptr, 2, &when, nullptr, out, 80) > 0);
    CHECK(std::u16string(out) == u"3:06 PM");
    CHECK(pg6_win32_get_time_format_ex(nullptr, 1, &when, nullptr, out, 80) > 0);
    CHECK(std::u16string(out) == u"3 PM");
    CHECK(pg6_win32_get_time_format_ex(nullptr, 8, &when, nullptr, out, 80) > 0);
    CHECK(std::u16string(out) == u"15:06:07");
    CHECK(pg6_win32_get_time_format_ex(nullptr, 4, &when, nullptr, out, 80) > 0);
    CHECK(std::u16string(out) == u"3:06:07");
    CHECK(pg6_win32_get_time_format_ex(nullptr, 0, &when, u"hh:mm:ss t", out, 80) > 0);
    CHECK(std::u16string(out) == u"03:06:07 P");
    Pg6SystemTime midnight{2021, 1, 0, 1, 0, 5, 0, 0};
    CHECK(pg6_win32_get_time_format_ex(nullptr, 0, &midnight, nullptr, out, 80) > 0);
    CHECK(std::u16string(out) == u"12:05:00 AM");
    CHECK(pg6_win32_get_time_format_ex(nullptr, 0x100, &when, nullptr, out, 80) == 0);
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);
    CHECK(pg6_win32_get_time_format_ex(nullptr, 0, nullptr, nullptr, out, 80) > 0);  // current time
    CHECK(pg6_win32_get_time_format_ex(nullptr, 0, &when, nullptr, out, 3) == 0);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
}

// ---------------------------------------------------------------------------
// Strings
// ---------------------------------------------------------------------------

int32_t mb2wc(uint32_t cp, uint32_t flags, const std::string& s, int32_t cb, std::u16string* out,
              int32_t cch = 256) {
    char16_t buf[256];
    std::memset(buf, 0x55, sizeof(buf));
    const int32_t n = pg6_win32_multi_byte_to_wide_char(cp, flags, s.c_str(), cb, cch == 0 ? nullptr : buf, cch);
    if (out != nullptr && n > 0 && cch != 0) {
        out->assign(buf, static_cast<size_t>(n));
    }
    return n;
}

int32_t wc2mb(uint32_t cp, uint32_t flags, const std::u16string& s, int32_t cch, std::string* out,
              int32_t cb = 256, const char* def = nullptr, int32_t* used = nullptr) {
    char buf[256];
    std::memset(buf, 0x55, sizeof(buf));
    const int32_t n = pg6_win32_wide_char_to_multi_byte(cp, flags, s.c_str(), cch, cb == 0 ? nullptr : buf,
                                                        cb, def, used);
    if (out != nullptr && n > 0 && cb != 0) {
        out->assign(buf, static_cast<size_t>(n));
    }
    return n;
}

void test_string_conversion() {
    std::u16string w;
    // Size query, NUL-terminated (-1) vs explicit length, CP_UTF8.
    CHECK(mb2wc(CP_UTF8, 0, "hello", -1, nullptr, 0) == 6);
    CHECK(mb2wc(CP_UTF8, 0, "hello", 5, nullptr, 0) == 5);
    CHECK(mb2wc(CP_UTF8, 0, "hello", -1, &w) == 6 && w.size() == 6 && w[5] == 0 && w.substr(0, 5) == u"hello");
    CHECK(mb2wc(CP_UTF8, 0, "hello", 5, &w) == 5 && w == u"hello");
    CHECK(mb2wc(CP_UTF8, 0, "h\xC3\xA9llo \xE2\x82\xAC \xF0\x9F\x98\x80", 15, &w) == 10);
    CHECK(w == u"h\u00E9llo \u20AC \xD83D\xDE00");
    // Exact fit and one too small.
    CHECK(mb2wc(CP_UTF8, 0, "abc", -1, &w, 4) == 4);
    CHECK(pg6_win32_multi_byte_to_wide_char(CP_UTF8, 0, "abc", -1, nullptr, 3) == 0);  // dst null, cch > 0
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(mb2wc(CP_UTF8, 0, "abc", -1, &w, 3) == 0);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
    // Invalid arguments.
    CHECK(pg6_win32_multi_byte_to_wide_char(CP_UTF8, 0, nullptr, -1, nullptr, 0) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(mb2wc(CP_UTF8, 0, "x", 0, &w) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(mb2wc(CP_UTF8, 0, "x", -2, &w) == 0);
    CHECK(mb2wc(CP_UTF8, 0, "x", -1, &w, -1) == 0);
    CHECK(mb2wc(12345, 0, "x", -1, &w) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(mb2wc(CP_OEMCP, 0, "x", -1, &w) == 0);
    // Flags.
    CHECK(mb2wc(CP_UTF8, MB_PRECOMPOSED, "x", -1, &w) == 0);
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);
    CHECK(mb2wc(CP_UTF8, MB_ERR_INVALID_CHARS, "ok", -1, &w) == 3);
    CHECK(mb2wc(CP_ACP, MB_PRECOMPOSED | MB_COMPOSITE, "x", -1, &w) == 0);
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);
    CHECK(mb2wc(CP_ACP, 0x100, "x", -1, &w) == 0);
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);
    CHECK(mb2wc(CP_ACP, MB_PRECOMPOSED, "x", -1, &w) == 2);
    // Invalid UTF-8: replacement characters, or failure with MB_ERR_INVALID_CHARS.
    CHECK(mb2wc(CP_UTF8, 0, "a\x80z", 3, &w) == 3 && w == u"a\uFFFDz");
    CHECK(mb2wc(CP_UTF8, 0, "a\xC3", 2, &w) == 2 && w == u"a\uFFFD");       // truncated
    CHECK(mb2wc(CP_UTF8, 0, "\xC0\xAF", 2, &w) == 2 && w == u"\uFFFD\uFFFD");  // overlong
    CHECK(mb2wc(CP_UTF8, 0, "\xED\xA0\x80", 3, &w) == 3);                      // surrogate
    CHECK(w == u"\uFFFD\uFFFD\uFFFD");
    CHECK(mb2wc(CP_UTF8, 0, "\xF4\x90\x80\x80", 4, &w) == 4);                 // > U+10FFFF
    CHECK(mb2wc(CP_UTF8, 0, "\xE2\x82z", 3, &w) == 2 && w == u"\uFFFDz");     // maximal subpart
    CHECK(mb2wc(CP_UTF8, MB_ERR_INVALID_CHARS, "a\x80z", 3, &w) == 0);
    CHECK(GetLastError() == ERROR_NO_UNICODE_TRANSLATION);
    CHECK(mb2wc(CP_UTF8, MB_ERR_INVALID_CHARS, "\xFF", 1, &w) == 0);
    CHECK(mb2wc(CP_UTF8, MB_ERR_INVALID_CHARS, "\xC3\xA9", 2, &w) == 1 && w == u"\u00E9");
    // CP_ACP is Windows-1252.
    CHECK(mb2wc(CP_ACP, 0, "\x80\xE9\x9F", 3, &w) == 3 && w == u"\u20AC\u00E9\u0178");
    CHECK(mb2wc(CP_LATIN1, 0, "\x80\xE9", 2, &w) == 2 && w == u"\u0080\u00E9");
    CHECK(mb2wc(CP_USASCII, 0, "a\xE9", 2, &w) == 2 && w == u"a\uFFFD");
    CHECK(mb2wc(CP_USASCII, MB_ERR_INVALID_CHARS, "a\xE9", 2, &w) == 0);
    // Embedded NULs are converted with an explicit length.
    CHECK(mb2wc(CP_UTF8, 0, std::string("a\0b", 3), 3, &w) == 3 && w[1] == 0);

    std::string s;
    // WideCharToMultiByte, CP_UTF8.
    CHECK(wc2mb(CP_UTF8, 0, u"h\u00E9llo \u20AC \xD83D\xDE00", -1, nullptr, 0) == 16);
    CHECK(wc2mb(CP_UTF8, 0, u"h\u00E9llo \u20AC \xD83D\xDE00", 10, &s) == 15);
    CHECK(s == "h\xC3\xA9llo \xE2\x82\xAC \xF0\x9F\x98\x80");
    CHECK(wc2mb(CP_UTF8, 0, u"abc", -1, &s, 4) == 4 && s.size() == 4 && s[3] == 0);
    CHECK(wc2mb(CP_UTF8, 0, u"abc", -1, &s, 3) == 0);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
    CHECK(wc2mb(CP_UTF8, 0, u"\u20AC", 1, &s, 2) == 0);  // 3 bytes need room for all of them
    CHECK(pg6_win32_wide_char_to_multi_byte(CP_UTF8, 0, nullptr, -1, nullptr, 0, nullptr, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(wc2mb(CP_UTF8, 0, u"x", 0, &s) == 0);
    CHECK(wc2mb(CP_UTF8, 0x1, u"x", -1, &s) == 0);
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);
    CHECK(wc2mb(CP_UTF8, 0, u"x", -1, &s, 256, "?") == 0);  // default char not allowed for UTF-8
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    // Lone surrogates.
    CHECK(wc2mb(CP_UTF8, 0, std::u16string(u"a\xD800z"), 3, &s) == 5);
    CHECK(s == "a\xEF\xBF\xBDz");
    CHECK(wc2mb(CP_UTF8, 0, std::u16string(u"\xDC00"), 1, &s) == 3);
    CHECK(wc2mb(CP_UTF8, WC_ERR_INVALID_CHARS, std::u16string(u"a\xD800z"), 3, &s) == 0);
    CHECK(GetLastError() == ERROR_NO_UNICODE_TRANSLATION);
    CHECK(wc2mb(CP_UTF8, WC_ERR_INVALID_CHARS, std::u16string(u"ab"), 2, &s) == 2);
    // Single-byte code pages and the default character.
    int32_t used = 7;
    CHECK(wc2mb(CP_ACP, 0, u"\u20AC\u00E9z", 3, &s, 256, nullptr, &used) == 3);
    CHECK(s == "\x80\xE9z" && used == 0);
    CHECK(wc2mb(CP_ACP, 0, u"a\u4E2Db", 3, &s, 256, nullptr, &used) == 3);
    CHECK(s == "a?b" && used == 1);
    CHECK(wc2mb(CP_ACP, 0, u"a\u4E2Db", 3, &s, 256, "#", &used) == 3 && s == "a#b" && used == 1);
    CHECK(wc2mb(CP_ACP, 0, std::u16string(u"a\xD83D\xDE00"), 3, &s, 256, nullptr, &used) == 2);
    CHECK(s == "a?");  // a surrogate pair is one unmappable character
    CHECK(wc2mb(CP_USASCII, 0, u"\u00E9", 1, &s) == 1 && s == "?");
    CHECK(wc2mb(CP_LATIN1, 0, u"\u00E9\u0100", 2, &s) == 2 && s == "\xE9?");
    CHECK(wc2mb(CP_ACP, WC_ERR_INVALID_CHARS, u"x", 1, &s) == 0);  // only valid for UTF-8
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);
    CHECK(wc2mb(CP_ACP, WC_NO_BEST_FIT_CHARS, u"x", 1, &s) == 1);
    CHECK(wc2mb(77777, 0, u"x", 1, &s) == 0);
    // used_default is cleared on failure.
    used = 9;
    CHECK(wc2mb(CP_ACP, 0x1, u"x", 1, &s, 256, nullptr, &used) == 0 && used == 0);

    // Round trip of a long string.
    std::u16string big(5000, u'\u00E9');
    std::string utf8;
    const int32_t need = pg6_win32_wide_char_to_multi_byte(CP_UTF8, 0, big.c_str(), 5000, nullptr, 0, nullptr, nullptr);
    CHECK(need == 10000);
    utf8.resize(static_cast<size_t>(need));
    CHECK(pg6_win32_wide_char_to_multi_byte(CP_UTF8, 0, big.c_str(), 5000, utf8.data(), need, nullptr, nullptr) == need);
    std::u16string wide(5000, u'?');
    CHECK(pg6_win32_multi_byte_to_wide_char(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), need, wide.data(), 5000) == 5000);
    CHECK(wide == big);
}

void test_string_compare_and_locale() {
    CHECK(pg6_win32_lstrcmp_w(u"abc", u"abc") == 0);
    CHECK(pg6_win32_lstrcmp_w(u"abc", u"abd") < 0);
    CHECK(pg6_win32_lstrcmp_w(u"b", u"a") > 0);
    CHECK(pg6_win32_lstrcmp_w(u"a", u"B") < 0);  // linguistic: case-insensitive first
    CHECK(pg6_win32_lstrcmp_w(u"a", u"A") < 0);  // lowercase sorts before uppercase
    CHECK(pg6_win32_lstrcmp_w(u"ab", u"abc") < 0);
    CHECK(pg6_win32_lstrcmp_w(nullptr, u"a") < 0 && pg6_win32_lstrcmp_w(u"a", nullptr) > 0);
    CHECK(pg6_win32_lstrcmp_w(nullptr, nullptr) == 0);
    CHECK(pg6_win32_lstrcmpi_w(u"Hello", u"hELLO") == 0);
    CHECK(pg6_win32_lstrcmpi_w(u"\u00C9", u"\u00E9") == 0);
    CHECK(pg6_win32_lstrlen_w(u"four") == 4 && pg6_win32_lstrlen_w(nullptr) == 0);
    CHECK(pg6_win32_compare_string_w(0x409, 0, u"a", -1, u"b", -1) == CSTR_LESS_THAN);
    CHECK(pg6_win32_compare_string_w(0x409, 0, u"b", 1, u"a", 1) == CSTR_GREATER_THAN);
    CHECK(pg6_win32_compare_string_w(0x409, 0, u"abc", 2, u"abd", 2) == CSTR_EQUAL);
    CHECK(pg6_win32_compare_string_w(0x409, NORM_IGNORECASE, u"ABC", -1, u"abc", -1) == CSTR_EQUAL);
    CHECK(pg6_win32_compare_string_w(0x409, 0, u"ABC", -1, u"abc", -1) != CSTR_EQUAL);
    CHECK(pg6_win32_compare_string_w(0x409, 0, nullptr, -1, u"abc", -1) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_compare_string_w(0x409, 0x40000000u, u"a", -1, u"a", -1) == 0);
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);
    CHECK(pg6_win32_compare_string_ordinal(u"a", -1, u"B", -1, 0) == CSTR_GREATER_THAN);  // 'a' > 'B'
    CHECK(pg6_win32_compare_string_ordinal(u"a", -1, u"B", -1, 1) == CSTR_LESS_THAN);
    CHECK(pg6_win32_compare_string_ordinal(u"abc", 3, u"abcd", 3, 0) == CSTR_EQUAL);
    CHECK(pg6_win32_compare_string_ordinal(nullptr, 0, u"a", 1, 0) == 0);

    char16_t buf[16];
    CHECK(pg6_win32_get_user_default_locale_name(buf, 16) == 6 && std::u16string(buf) == u"en-US");
    CHECK(pg6_win32_get_user_default_locale_name(buf, 5) == 0);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
    CHECK(pg6_win32_get_user_default_locale_name(nullptr, 5) == 0);
    CHECK(pg6_win32_get_user_default_geo_name(buf, 16) == 3 && std::u16string(buf) == u"US");
    CHECK(pg6_win32_get_user_default_geo_name(nullptr, 0) == 3);
    CHECK(pg6_win32_get_user_default_geo_name(buf, 2) == 0);
    CHECK(pg6_win32_locale_name_to_lcid(u"en-US", 0) == 0x0409);
    CHECK(pg6_win32_locale_name_to_lcid(u"EN-gb", 0) == 0x0809);
    CHECK(pg6_win32_locale_name_to_lcid(nullptr, 0) == 0x0409);
    CHECK(pg6_win32_locale_name_to_lcid(u"", 0) == 0x007F);
    CHECK(pg6_win32_locale_name_to_lcid(u"en", 0) == 0);  // neutral needs the flag
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_locale_name_to_lcid(u"en", 0x08000000u) == 0x0009);
    CHECK(pg6_win32_locale_name_to_lcid(u"xx-YY", 0) == 0);
    CHECK(pg6_win32_locale_name_to_lcid(u"en-US", 1) == 0);
    CHECK(GetLastError() == ERROR_INVALID_FLAGS);

    // FormatMessageA.
    char msg[128];
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, 2,
                                     0, msg, sizeof(msg), nullptr) == 44);
    CHECK(std::string(msg) == "The system cannot find the file specified.\r\n");
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_SYSTEM | 0xFF, nullptr, 5, 0, msg, sizeof(msg),
                                     nullptr) == 17);
    CHECK(std::string(msg) == "Access is denied.");
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, 0x80070057u, 0, msg, sizeof(msg),
                                     nullptr) > 0);  // HRESULT_FROM_WIN32(87)
    CHECK(std::string(msg).rfind("The parameter is incorrect.", 0) == 0);
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, 999999, 0, msg, sizeof(msg), nullptr) == 0);
    CHECK(GetLastError() == ERROR_MR_MID_NOT_FOUND);
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, 2, 0, msg, 10, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER);
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, 2, 0, nullptr, 10, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_format_message_a(0, nullptr, 2, 0, msg, sizeof(msg), nullptr) == 0);
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_STRING | FORMAT_MESSAGE_IGNORE_INSERTS, "a %1 b%%%n",
                                     0, 0, msg, sizeof(msg), nullptr) == 9);
    CHECK(std::string(msg) == "a %1 b%\r\n");
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_STRING, "x%1y%2!d!z", 0, 0, msg, sizeof(msg),
                                     nullptr) == 3);
    CHECK(std::string(msg) == "xyz");
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_STRING, "end%0 ignored", 0, 0, msg, sizeof(msg),
                                     nullptr) == 3);
    // ALLOCATE_BUFFER: the block comes from the process heap; LocalFree releases it.
    uint64_t block = 0xFFFF;
    CHECK(pg6_win32_format_message_a(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                         FORMAT_MESSAGE_IGNORE_INSERTS | 0xFF,
                                     nullptr, 5, 0, reinterpret_cast<char*>(&block), 0, nullptr) == 17);
    CHECK(block != 0 && block != 0xFFFF);
    char readback[32] = {};
    CHECK(get_vmem_arena()->read(block, readback, 18));
    CHECK(std::string(readback) == "Access is denied.");
    CHECK(HeapSize(GetProcessHeap(), 0, block) >= 18);
    CHECK(pg6_win32_local_free(block) == 0);
    CHECK(pg6_win32_local_free(block) == block);  // double free is rejected, handle returned
    CHECK(pg6_win32_local_free(0) == 0);
    CHECK(pg6_win32_local_free(0x1234) == 0x1234);
}

// ---------------------------------------------------------------------------
// Process / thread / system info
// ---------------------------------------------------------------------------

void test_process_and_threads() {
    CHECK(pg6_win32_get_current_process() == ~uint64_t{0});
    CHECK(pg6_win32_get_current_thread() == kCurrentThreadPseudoHandle);
    CHECK(pg6_win32_get_current_process_id() == kCurrentProcessId && kCurrentProcessId != 0);
    CHECK(pg6_win32_get_process_id(pg6_win32_get_current_process()) == kCurrentProcessId);
    CHECK(pg6_win32_get_process_id(0x1234) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);

    Pg6StartupInfoW si;
    std::memset(&si, 0xEE, sizeof(si));
    pg6_win32_get_startup_info_w(&si);
    CHECK(si.cb == 104 && si.flags == 0 && si.title == nullptr && si.std_input == 0);
    pg6_win32_get_startup_info_w(nullptr);

    // Priorities: current thread pseudo-handle, a real thread handle, invalid handles.
    CHECK(pg6_win32_get_thread_priority(kCurrentThreadPseudoHandle) == 0);
    CHECK(pg6_win32_set_thread_priority(kCurrentThreadPseudoHandle, 2) == 1);
    CHECK(pg6_win32_get_thread_priority(kCurrentThreadPseudoHandle) == 2);
    CHECK(pg6_win32_set_thread_priority(kCurrentThreadPseudoHandle, 0) == 1);
    CHECK(pg6_win32_set_thread_priority(kCurrentThreadPseudoHandle, 99) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_set_thread_priority(0x7777, 0) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(static_cast<uint32_t>(pg6_win32_get_thread_priority(0x7777)) == THREAD_PRIORITY_ERROR_RETURN);
    CHECK(pg6_win32_get_thread_priority(0) == static_cast<int32_t>(THREAD_PRIORITY_ERROR_RETURN));
    uint32_t tid = 0;
    // Thread start routines are host functions here; the thread only has to exist.
    const Handle th = CreateThread(nullptr, 0, [](void*) -> uint32_t { return 0; }, nullptr, CREATE_SUSPENDED, &tid);
    CHECK(th != kNullHandle);
    CHECK(pg6_win32_set_thread_priority(th, -2) == 1);
    CHECK(pg6_win32_get_thread_priority(th) == -2);
    CHECK(pg6_win32_suspend_thread(th) == 0xFFFFFFFFu);
    CHECK(GetLastError() == ERROR_NOT_SUPPORTED);
    CHECK(pg6_win32_suspend_thread(0x7777) == 0xFFFFFFFFu);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    Pg6GroupAffinity ga;
    CHECK(pg6_win32_get_thread_group_affinity(th, &ga) == 1);
    CHECK(ga.group == 0 && ga.mask != 0 && ga.reserved[0] == 0);
    Pg6GroupAffinity want{1, 0, {0, 0, 0}};
    Pg6GroupAffinity prev;
    CHECK(pg6_win32_set_thread_group_affinity(th, &want, &prev) == 1);
    CHECK(prev.mask != 0);
    want.group = 1;
    CHECK(pg6_win32_set_thread_group_affinity(th, &want, &prev) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    want.group = 0;
    want.mask = 0;
    CHECK(pg6_win32_set_thread_group_affinity(th, &want, nullptr) == 0);
    CHECK(pg6_win32_set_thread_group_affinity(th, nullptr, nullptr) == 0);
    CHECK(pg6_win32_get_thread_group_affinity(0x7777, &ga) == 0);
    CHECK(pg6_win32_get_thread_group_affinity(th, nullptr) == 0);
    // SetThreadIdealProcessorEx: a recorded hint; the previous value starts at processor 0.
    Pg6ProcessorNumber ideal{0, 1, 0};
    Pg6ProcessorNumber before{9, 9, 9};
    CHECK(pg6_win32_set_thread_ideal_processor_ex(th, &ideal, &before) == 1);
    CHECK(before.group == 0 && before.number == 0);
    ideal.number = 0;
    CHECK(pg6_win32_set_thread_ideal_processor_ex(th, &ideal, &before) == 1);
    CHECK(before.number == 1);
    ideal.group = 1;
    CHECK(pg6_win32_set_thread_ideal_processor_ex(th, &ideal, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    ideal = Pg6ProcessorNumber{0, 255, 0};
    CHECK(pg6_win32_set_thread_ideal_processor_ex(th, &ideal, nullptr) == 0);
    CHECK(pg6_win32_set_thread_ideal_processor_ex(th, nullptr, nullptr) == 0);
    CHECK(pg6_win32_set_thread_ideal_processor_ex(0x7777, &ideal, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(ResumeThread(th) != RESUME_THREAD_FAILED);
    CHECK(WaitForSingleObject(th, 5000) == WAIT_OBJECT_0);
    CloseHandle(th);

    CHECK(pg6_win32_open_thread(0x1FFFFF, 0, GetCurrentThreadId()) == kCurrentThreadPseudoHandle);
    CHECK(pg6_win32_open_thread(0x1FFFFF, 0, 0) == kNullHandle);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    CHECK(pg6_win32_set_process_priority_boost(pg6_win32_get_current_process(), 1) == 1);
    CHECK(pg6_win32_set_process_priority_boost(0x1234, 1) == 0);

    // Stack limits: wide default, overridable per thread.
    uint64_t lo = 0, hi = 0;
    pg6_win32_get_current_thread_stack_limits(&lo, &hi);
    CHECK(lo != 0 && hi > lo);
    set_current_thread_stack_limits(0x100000, 0x200000);
    pg6_win32_get_current_thread_stack_limits(&lo, &hi);
    CHECK(lo == 0x100000 && hi == 0x200000);
    set_current_thread_stack_limits(0, 0);
    pg6_win32_get_current_thread_stack_limits(nullptr, nullptr);

    // Processor features (Zen 2 class CPU).
    CHECK(pg6_win32_is_processor_feature_present(10) == 1);  // SSE2
    CHECK(pg6_win32_is_processor_feature_present(23) == 1);  // fastfail
    CHECK(pg6_win32_is_processor_feature_present(40) == 1);  // AVX2
    CHECK(pg6_win32_is_processor_feature_present(41) == 0);  // AVX-512
    CHECK(pg6_win32_is_processor_feature_present(29) == 0);  // ARM
    CHECK(pg6_win32_is_processor_feature_present(9999) == 0);

    // TerminateProcess records the request and calls the installed handler.
    static uint32_t handler_code = 0;
    static int handler_calls = 0;
    reset_process_exit_state();
    CHECK(!process_exit_requested(nullptr));
    CHECK(pg6_win32_terminate_process(0x1234, 1) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(!process_exit_requested(nullptr));
    set_process_exit_handler([](uint32_t code, void*) { handler_code = code; ++handler_calls; }, nullptr);
    CHECK(pg6_win32_terminate_process(pg6_win32_get_current_process(), 42) == 1);
    uint32_t code = 0;
    CHECK(process_exit_requested(&code) && code == 42);
    CHECK(handler_calls == 1 && handler_code == 42);
    set_process_exit_handler(nullptr, nullptr);
    CHECK(pg6_win32_terminate_process(pg6_win32_get_current_process(), 7) == 1);
    CHECK(process_exit_requested(&code) && code == 7 && handler_calls == 1);
    reset_process_exit_state();

    // Command line.
    CHECK(std::string(pg6_win32_get_command_line_a()) == "game.exe");
    set_command_line("game.exe -windowed");
    const char* first = pg6_win32_get_command_line_a();
    CHECK(std::string(first) == "game.exe -windowed");
    set_command_line("other.exe");
    CHECK(std::string(pg6_win32_get_command_line_a()) == "other.exe");
    CHECK(std::string(first) == "game.exe -windowed");  // earlier pointers stay valid
    set_command_line("game.exe");
}

void test_sysinfo() {
    Pg6SystemInfo info;
    std::memset(&info, 0xAA, sizeof(info));
    pg6_win32_get_system_info(&info);
    CHECK(info.processor_architecture == 9);
    CHECK(info.page_size == 4096 && info.allocation_granularity == 65536);
    CHECK(info.number_of_processors >= 1 && info.number_of_processors <= 64);
    CHECK(info.number_of_processors <= std::max(1u, std::thread::hardware_concurrency()));
    CHECK(info.active_processor_mask != 0);
    int mask_bits = 0;
    for (uint64_t m = info.active_processor_mask; m != 0; m >>= 1) {
        mask_bits += static_cast<int>(m & 1u);
    }
    CHECK(mask_bits == static_cast<int>(info.number_of_processors));
    CHECK(info.minimum_application_address == 0x10000 &&
          info.maximum_application_address > info.minimum_application_address);
    CHECK(info.processor_type == 8664 && info.reserved == 0);
    Pg6SystemInfo native;
    pg6_win32_get_native_system_info(&native);
    CHECK(std::memcmp(&info, &native, sizeof(info)) == 0);
    pg6_win32_get_system_info(nullptr);

    Pg6MemoryStatusEx mem;
    std::memset(&mem, 0, sizeof(mem));
    mem.length = sizeof(mem);
    CHECK(pg6_win32_global_memory_status_ex(&mem) == 1);
    CHECK(mem.total_phys > 0 && mem.avail_phys <= mem.total_phys && mem.memory_load <= 100);
    CHECK(mem.total_virtual > mem.total_phys && mem.avail_virtual <= mem.total_virtual);
    set_total_physical_memory(1ull << 30);
    CHECK(pg6_win32_global_memory_status_ex(&mem) == 1 && mem.total_phys == (1ull << 30));
    set_total_physical_memory(12ull << 30);
    mem.length = 12;
    CHECK(pg6_win32_global_memory_status_ex(&mem) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_global_memory_status_ex(nullptr) == 0);

    // GetLogicalProcessorInformation: size protocol, and consistent core / cache entries.
    uint32_t length = 0;
    CHECK(pg6_win32_get_logical_processor_information(nullptr, &length) == 0);
    CHECK(GetLastError() == ERROR_INSUFFICIENT_BUFFER && length >= 3 * sizeof(Pg6LogicalProcessorInformation));
    CHECK(length % sizeof(Pg6LogicalProcessorInformation) == 0);
    std::vector<Pg6LogicalProcessorInformation> entries(length / sizeof(Pg6LogicalProcessorInformation));
    uint32_t too_small = length - 1;
    CHECK(pg6_win32_get_logical_processor_information(entries.data(), &too_small) == 0 && too_small == length);
    CHECK(pg6_win32_get_logical_processor_information(entries.data(), &length) == 1);
    uint64_t core_union = 0;
    int numa = 0, packages = 0;
    for (const auto& e : entries) {
        if (e.relationship == 0) {
            CHECK((e.processor_mask & core_union) == 0);
            core_union |= e.processor_mask;
        } else if (e.relationship == 1) {
            ++numa;
        } else if (e.relationship == 3) {
            ++packages;
        }
    }
    CHECK(core_union == info.active_processor_mask && numa == 1 && packages == 1);
    CHECK(pg6_win32_get_logical_processor_information(entries.data(), nullptr) == 0);
}

// ---------------------------------------------------------------------------
// Debug, errors, lifecycle, misc
// ---------------------------------------------------------------------------

void test_debug_and_lifecycle() {
    CHECK(pg6_win32_is_debugger_present() == 0);
    pg6_win32_output_debug_string_a("test message\r\n");
    pg6_win32_output_debug_string_a(nullptr);

    CHECK(pg6_win32_set_unhandled_exception_filter(0x1000) == 0);
    CHECK(get_unhandled_exception_filter() == 0x1000);
    CHECK(pg6_win32_set_unhandled_exception_filter(0x2000) == 0x1000);
    CHECK(pg6_win32_unhandled_exception_filter(nullptr) == 1);  // EXCEPTION_EXECUTE_HANDLER
    CHECK(pg6_win32_set_unhandled_exception_filter(0) == 0x2000);

    // App lifecycle registrations.
    const size_t base = app_notification_registration_count();
    uint64_t reg1 = 0, reg2 = 0;
    CHECK(pg6_win32_register_app_state_change_notification(0x1000, 5, &reg1) == ERROR_SUCCESS);
    CHECK(pg6_win32_register_app_constrained_change_notification(0x2000, 6, &reg2) == ERROR_SUCCESS);
    CHECK(reg1 != 0 && reg2 != 0 && reg1 != reg2);
    CHECK(app_notification_registration_count() == base + 2);
    CHECK(pg6_win32_register_app_state_change_notification(0, 0, &reg2) == ERROR_INVALID_PARAMETER && reg2 == 0);
    CHECK(pg6_win32_register_app_state_change_notification(0x1000, 0, nullptr) == ERROR_INVALID_PARAMETER);
    pg6_win32_unregister_app_state_change_notification(reg1);
    pg6_win32_unregister_app_state_change_notification(reg1);  // idempotent
    CHECK(app_notification_registration_count() == base + 1);
    pg6_win32_unregister_app_constrained_change_notification(0x1);
    CHECK(app_notification_registration_count() == base + 1);

    // WER, named pipes.
    CHECK(pg6_win32_wer_register_file(u"C:\\x.log", 2, 0) == 0);
    CHECK(pg6_win32_wer_register_file(nullptr, 2, 0) == E_INVALIDARG);
    CHECK(pg6_win32_wer_register_file(u"x", 0, 0) == E_INVALIDARG);
    CHECK(pg6_win32_wer_register_file(u"x", 2, 0x80) == E_INVALIDARG);
    CHECK(pg6_win32_set_named_pipe_handle_state(0x1234, nullptr, nullptr, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_wait_named_pipe_w(u"\\\\.\\pipe\\x", 10) == 0);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_wait_named_pipe_w(nullptr, 10) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);

    // Misc synch / interlocked leftovers.
    Pg6CriticalSection cs;
    CHECK(pg6_win32_initialize_critical_section_and_spin_count(&cs, 4000) == 1);
    EnterCriticalSection(&cs);
    LeaveCriticalSection(&cs);
    DeleteCriticalSection(&cs);
    CHECK(pg6_win32_initialize_critical_section_and_spin_count(nullptr, 0) == 0);
    const Handle sem = pg6_win32_create_semaphore_ex_w(nullptr, 1, 2, nullptr, 0, 0x1F0003);
    CHECK(sem != kNullHandle && WaitForSingleObject(sem, 0) == WAIT_OBJECT_0);
    CloseHandle(sem);
    CHECK(pg6_win32_create_semaphore_ex_w(nullptr, 1, 2, nullptr, 1, 0) == kNullHandle);
    uint8_t slist[16];
    std::memset(slist, 0xFF, sizeof(slist));
    pg6_win32_initialize_s_list_head(slist);
    bool zero = true;
    for (uint8_t b : slist) {
        zero = zero && b == 0;
    }
    CHECK(zero);
    pg6_win32_initialize_s_list_head(nullptr);

    const auto t0 = GetTickCount64();
    pg6_win32_sleep(20);
    CHECK(GetTickCount64() - t0 >= 15);
    pg6_win32_sleep(0);

    // VirtualQueryEx routes to VirtualQuery for the current process only.
    const uint64_t va = VirtualAlloc(0, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    CHECK(va != 0);
    Pg6MemoryBasicInformation mbi;
    mbi = Pg6MemoryBasicInformation{};
    CHECK(pg6_win32_virtual_query_ex(pg6_win32_get_current_process(), va, &mbi, sizeof(mbi)) == sizeof(mbi));
    CHECK(mbi.state == MEM_COMMIT);
    CHECK(pg6_win32_virtual_query_ex(0x1234, va, &mbi, sizeof(mbi)) == 0);
    CHECK(GetLastError() == ERROR_INVALID_HANDLE);
    VirtualFree(va, 0, MEM_RELEASE);
}

// ---------------------------------------------------------------------------
// COM
// ---------------------------------------------------------------------------

void test_com() {
    CHECK(pg6_win32_co_initialize_ex(nullptr, COINIT_MULTITHREADED) == S_OK);
    CHECK(pg6_win32_co_initialize_ex(nullptr, COINIT_MULTITHREADED) == S_FALSE);
    CHECK(pg6_win32_co_initialize_ex(nullptr, COINIT_APARTMENTTHREADED) == RPC_E_CHANGED_MODE);
    CHECK(pg6_win32_co_initialize_ex(&g_failures, 0) == E_INVALIDARG);
    pg6_win32_co_uninitialize();
    pg6_win32_co_uninitialize();
    // Balanced: the apartment mode can change again.
    CHECK(pg6_win32_co_initialize_ex(nullptr, COINIT_APARTMENTTHREADED) == S_OK);
    pg6_win32_co_uninitialize();
    pg6_win32_co_uninitialize();  // extra call is harmless
    // Initialization is per thread.
    CHECK(pg6_win32_co_initialize_ex(nullptr, COINIT_MULTITHREADED) == S_OK);
    std::thread([] { CHECK(pg6_win32_co_initialize_ex(nullptr, COINIT_APARTMENTTHREADED) == S_OK); pg6_win32_co_uninitialize(); }).join();
    pg6_win32_co_uninitialize();

    Pg6Guid clsid{0x12345678, 0x9ABC, 0xDEF0, {1, 2, 3, 4, 5, 6, 7, 8}};
    void* obj = reinterpret_cast<void*>(0x1234);
    CHECK(pg6_win32_co_create_instance(&clsid, nullptr, 1, &clsid, &obj) == REGDB_E_CLASSNOTREG);
    CHECK(obj == nullptr);
    CHECK(pg6_win32_co_create_instance(&clsid, nullptr, 1, &clsid, nullptr) == E_POINTER);
    obj = reinterpret_cast<void*>(0x1234);
    CHECK(pg6_win32_co_create_instance(nullptr, nullptr, 1, &clsid, &obj) == E_INVALIDARG && obj == nullptr);

    char16_t text[64];
    CHECK(pg6_win32_string_from_guid2(&clsid, text, 64) == 39);
    CHECK(std::u16string(text) == u"{12345678-9ABC-DEF0-0102-030405060708}");
    CHECK(pg6_win32_string_from_guid2(&clsid, text, 38) == 0);
    CHECK(pg6_win32_string_from_guid2(&clsid, text, 39) == 39);
    CHECK(pg6_win32_string_from_guid2(nullptr, text, 64) == 0);
    CHECK(pg6_win32_string_from_guid2(&clsid, nullptr, 64) == 0);

    Pg6Guid a{}, b{};
    CHECK(pg6_win32_co_create_guid(&a) == S_OK && pg6_win32_co_create_guid(&b) == S_OK);
    CHECK(std::memcmp(&a, &b, sizeof(a)) != 0);
    CHECK((a.data3 >> 12) == 4 && (a.data4[0] & 0xC0) == 0x80);  // RFC 4122 version 4
    CHECK(pg6_win32_co_create_guid(nullptr) == E_INVALIDARG);

    // CoTaskMem* use the process heap; foreign pointers are ignored.
    const uint64_t mem = pg6_win32_co_task_mem_alloc(64);
    CHECK(mem != 0 && HeapSize(GetProcessHeap(), 0, mem) >= 64);
    pg6_win32_co_task_mem_free(mem);
    CHECK(HeapSize(GetProcessHeap(), 0, mem) == kHeapSizeError);
    pg6_win32_co_task_mem_free(mem);
    pg6_win32_co_task_mem_free(0);
    pg6_win32_co_task_mem_free(0x1234);
    CHECK(pg6_win32_co_task_mem_alloc(0) != 0);

    // PropVariantClear frees LPWSTR payloads and resets the variant.
    uint8_t pv[24];
    std::memset(pv, 0xCC, sizeof(pv));
    const uint64_t str = pg6_win32_co_task_mem_alloc(16);
    uint16_t vt = 31;
    std::memset(pv, 0, sizeof(pv));
    std::memcpy(pv, &vt, 2);
    std::memcpy(pv + 8, &str, 8);
    CHECK(pg6_win32_prop_variant_clear(pv) == S_OK);
    CHECK(HeapSize(GetProcessHeap(), 0, str) == kHeapSizeError);
    bool zero = true;
    for (uint8_t byte : pv) {
        zero = zero && byte == 0;
    }
    CHECK(zero);
    CHECK(pg6_win32_prop_variant_clear(nullptr) == E_INVALIDARG);
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

void test_registry() {
    registry_clear();
    Pg6Handle key = 0xDEAD;
    // Empty by default.
    CHECK(pg6_win32_reg_open_key_w(HKEY_LOCAL_MACHINE, u"SOFTWARE\\Vendor", &key) == ERROR_FILE_NOT_FOUND);
    CHECK(key == 0);
    CHECK(pg6_win32_reg_open_key_w(HKEY_CURRENT_USER, u"Software", &key) == ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_reg_open_key_w(0x123, u"x", &key) == ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_reg_open_key_w(HKEY_LOCAL_MACHINE, u"x", nullptr) == ERROR_INVALID_PARAMETER);
    // Opening a predefined key without a subkey succeeds and returns it.
    CHECK(pg6_win32_reg_open_key_w(HKEY_LOCAL_MACHINE, nullptr, &key) == 0 && key == HKEY_LOCAL_MACHINE);
    CHECK(pg6_win32_reg_open_key_w(0x80000002u, u"", &key) == 0 && key == 0x80000002u);  // 32-bit form
    CHECK(pg6_win32_reg_close_key(key) == 0);  // closing a predefined key is a no-op
    uint32_t type = 99, size = 4;
    uint8_t data[8];
    CHECK(pg6_win32_reg_query_value_ex_w(HKEY_LOCAL_MACHINE, u"Value", nullptr, &type, data, &size) ==
          ERROR_FILE_NOT_FOUND);

    // A populated registry (set up by the runtime).
    const uint32_t dword = 0x11223344;
    registry_set_value("HKEY_LOCAL_MACHINE\\Software\\Vendor\\App", u"Version", REG_DWORD, &dword, 4);
    registry_set_value("hklm/software/vendor/app", u"Name", REG_SZ, u"Game", 10);
    CHECK(pg6_win32_reg_open_key_w(HKEY_LOCAL_MACHINE, u"SOFTWARE\\VENDOR", &key) == 0);  // implicit parent
    CHECK(key != 0 && key != HKEY_LOCAL_MACHINE);
    Pg6Handle app = 0;
    CHECK(pg6_win32_reg_open_key_w(key, u"App", &app) == 0);
    CHECK(pg6_win32_reg_open_key_w(key, u"Missing", &app) == ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_reg_open_key_w(HKEY_LOCAL_MACHINE, u"Software\\Vendor\\App", &app) == 0);
    CHECK(pg6_win32_reg_close_key(key) == 0);
    CHECK(pg6_win32_reg_close_key(key) == ERROR_INVALID_HANDLE);

    size = 4;
    type = 0;
    std::memset(data, 0, sizeof(data));
    CHECK(pg6_win32_reg_query_value_ex_w(app, u"version", nullptr, &type, data, &size) == 0);  // case-insensitive
    CHECK(type == REG_DWORD && size == 4);
    uint32_t got = 0;
    std::memcpy(&got, data, 4);
    CHECK(got == dword);
    size = 2;  // too small
    CHECK(pg6_win32_reg_query_value_ex_w(app, u"Version", nullptr, &type, data, &size) == ERROR_MORE_DATA && size == 4);
    size = 0;
    CHECK(pg6_win32_reg_query_value_ex_w(app, u"Name", nullptr, &type, nullptr, &size) == 0);  // size query
    CHECK(size == 10 && type == REG_SZ);
    CHECK(pg6_win32_reg_query_value_ex_w(app, u"Name", nullptr, nullptr, nullptr, nullptr) == 0);
    CHECK(pg6_win32_reg_query_value_ex_w(app, u"Nope", nullptr, &type, data, &size) == ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_reg_query_value_ex_w(app, u"Version", &size, &type, data, &size) == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_reg_query_value_ex_w(app, u"Version", nullptr, &type, data, nullptr) == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_reg_query_value_ex_w(0x999, u"Version", nullptr, &type, data, &size) == ERROR_INVALID_HANDLE);
    CHECK(pg6_win32_reg_query_value_ex_w(HKEY_LOCAL_MACHINE, u"Version", nullptr, &type, data, &size) ==
          ERROR_FILE_NOT_FOUND);
    CHECK(pg6_win32_reg_close_key(app) == 0);
    CHECK(pg6_win32_reg_close_key(0) == ERROR_INVALID_HANDLE);
    registry_clear();
    CHECK(pg6_win32_reg_open_key_w(HKEY_LOCAL_MACHINE, u"Software\\Vendor\\App", &app) == ERROR_FILE_NOT_FOUND);
}

// ---------------------------------------------------------------------------
// bcrypt
// ---------------------------------------------------------------------------

void test_bcrypt() {
    Pg6Handle alg = 0xDEAD;
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(&alg, u"RNG", nullptr, 0) == STATUS_SUCCESS);
    CHECK(alg != 0 && alg != 0xDEAD);
    Pg6Handle alg2 = 0;
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(&alg2, u"rng", u"Microsoft Primitive Provider", 0) == STATUS_SUCCESS);
    CHECK(alg2 != alg);

    uint8_t a[64], b[64];
    std::memset(a, 0, sizeof(a));
    std::memset(b, 0, sizeof(b));
    CHECK(pg6_win32_bcrypt_gen_random(alg, a, sizeof(a), 0) == STATUS_SUCCESS);
    CHECK(pg6_win32_bcrypt_gen_random(alg, b, sizeof(b), 0) == STATUS_SUCCESS);
    CHECK(std::memcmp(a, b, sizeof(a)) != 0);
    int nonzero = 0;
    for (uint8_t v : a) {
        nonzero += v != 0;
    }
    CHECK(nonzero > 40);  // 64 random bytes: P(<=40 non-zero) is astronomically small
    // System-preferred RNG: NULL handle with the flag.
    CHECK(pg6_win32_bcrypt_gen_random(0, a, sizeof(a), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == STATUS_SUCCESS);
    CHECK(std::memcmp(a, b, sizeof(a)) != 0);
    // Odd sizes only touch the requested bytes.
    uint8_t guard[16];
    std::memset(guard, 0x7E, sizeof(guard));
    CHECK(pg6_win32_bcrypt_gen_random(alg, guard + 4, 5, 0) == STATUS_SUCCESS);
    CHECK(guard[3] == 0x7E && guard[9] == 0x7E);
    CHECK(pg6_win32_bcrypt_gen_random(alg, nullptr, 0, 0) == STATUS_SUCCESS);
    CHECK(pg6_win32_bcrypt_gen_random(alg, nullptr, 8, 0) == STATUS_INVALID_PARAMETER);
    CHECK(pg6_win32_bcrypt_gen_random(0, a, 8, 0) == STATUS_INVALID_HANDLE);
    CHECK(pg6_win32_bcrypt_gen_random(alg, a, 8, BCRYPT_USE_SYSTEM_PREFERRED_RNG) == STATUS_INVALID_PARAMETER);
    CHECK(pg6_win32_bcrypt_gen_random(alg, a, 8, 0x80) == STATUS_INVALID_PARAMETER);
    CHECK(pg6_win32_bcrypt_gen_random(0x5555, a, 8, 0) == STATUS_INVALID_HANDLE);
    // Large request.
    std::vector<uint8_t> big(1 << 16, 0);
    CHECK(pg6_win32_bcrypt_gen_random(alg, big.data(), static_cast<uint32_t>(big.size()), 0) == STATUS_SUCCESS);
    std::set<uint8_t> distinct(big.begin(), big.end());
    CHECK(distinct.size() > 200);

    // Open errors.
    Pg6Handle h = 7;
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(&h, u"AES", nullptr, 0) == STATUS_NOT_SUPPORTED && h == 0);
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(&h, u"SHA256", nullptr, 0) == STATUS_NOT_SUPPORTED);
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(&h, u"NOPE", nullptr, 0) == STATUS_NOT_FOUND);
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(&h, u"RNG", u"Other Provider", 0) == STATUS_NOT_FOUND);
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(&h, nullptr, nullptr, 0) == STATUS_INVALID_PARAMETER);
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(&h, u"RNG", nullptr, 8) == STATUS_INVALID_PARAMETER);
    CHECK(pg6_win32_bcrypt_open_algorithm_provider(nullptr, u"RNG", nullptr, 0) == STATUS_INVALID_PARAMETER);

    // Close.
    CHECK(pg6_win32_bcrypt_close_algorithm_provider(alg, 1) == STATUS_INVALID_PARAMETER);
    CHECK(pg6_win32_bcrypt_close_algorithm_provider(alg, 0) == STATUS_SUCCESS);
    CHECK(pg6_win32_bcrypt_close_algorithm_provider(alg, 0) == STATUS_INVALID_HANDLE);
    CHECK(pg6_win32_bcrypt_gen_random(alg, a, 8, 0) == STATUS_INVALID_HANDLE);
    CHECK(pg6_win32_bcrypt_gen_random(alg2, a, 8, 0) == STATUS_SUCCESS);
    CHECK(pg6_win32_bcrypt_close_algorithm_provider(alg2, 0) == STATUS_SUCCESS);
    CHECK(pg6_win32_bcrypt_close_algorithm_provider(0, 0) == STATUS_INVALID_HANDLE);

    // Concurrent generation.
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([] {
            uint8_t buf[32];
            for (int k = 0; k < 100; ++k) {
                CHECK(pg6_win32_bcrypt_gen_random(0, buf, sizeof(buf), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == STATUS_SUCCESS);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}

// ---------------------------------------------------------------------------
// Window fake
// ---------------------------------------------------------------------------

// Class names are MAKEINTATOM-style "name or atom" pointers: a value with HIWORD == 0 is an atom.
// Emscripten places string literals below 64 KiB on wasm64, so literals would be read as atoms;
// copy names onto the heap (guest pointers are never below 64 KiB either).
const char* class_name_ptr(const char* literal) {
    static std::deque<std::string> keep;
    keep.emplace_back(literal);
    return keep.back().c_str();
}

// The guest form of a host pointer, for fields of guest structures (class_name).
const char* guest_name(const char* literal) {
    return reinterpret_cast<const char*>(
        static_cast<uintptr_t>(playgta6::os::guest_window::guest_address(class_name_ptr(literal))));
}

void test_windows() {
    reset_window_state();
    Pg6WndClassExA wc{};
    wc.size = sizeof(wc);
    wc.window_proc = 0x140001000;
    wc.class_name = guest_name("TitleWindow");
    const uint16_t atom = pg6_win32_register_class_ex_a(&wc);
    CHECK(atom >= 0xC000);
    CHECK(pg6_win32_register_class_ex_a(&wc) == 0);
    CHECK(GetLastError() == ERROR_CLASS_ALREADY_EXISTS);
    wc.class_name = guest_name("TITLEWINDOW");  // class names are case-insensitive
    CHECK(pg6_win32_register_class_ex_a(&wc) == 0);
    wc.class_name = guest_name("Other");
    CHECK(pg6_win32_register_class_ex_a(&wc) == atom + 1);
    Pg6WndClassExA bad = wc;
    bad.size = 48;
    CHECK(pg6_win32_register_class_ex_a(&bad) == 0 && GetLastError() == ERROR_INVALID_PARAMETER);
    bad = wc;
    bad.class_name = nullptr;
    CHECK(pg6_win32_register_class_ex_a(&bad) == 0);
    bad = wc;
    bad.class_name = reinterpret_cast<const char*>(0x1234);  // an atom, not a name
    CHECK(pg6_win32_register_class_ex_a(&bad) == 0);
    bad = wc;
    bad.class_name = guest_name("NoProc");
    bad.window_proc = 0;
    CHECK(pg6_win32_register_class_ex_a(&bad) == 0);
    CHECK(pg6_win32_register_class_ex_a(nullptr) == 0);

    CHECK(window_count() == 0);
    CHECK(pg6_win32_create_window_ex_a(0, class_name_ptr("NoSuchClass"), "t", 0, 0, 0, 1, 1, 0, 0, 0, nullptr) == 0);
    CHECK(GetLastError() == ERROR_CANNOT_FIND_WND_CLASS);
    CHECK(pg6_win32_create_window_ex_a(0, nullptr, "t", 0, 0, 0, 1, 1, 0, 0, 0, nullptr) == 0);
    const Pg6Handle hwnd =
        pg6_win32_create_window_ex_a(0, class_name_ptr("titlewindow"), "Title", 0x10CF0000, 0, 0, 1920, 1080, 0, 0, 0x140000000, nullptr);
    CHECK(hwnd != 0 && window_count() == 1);
    const Pg6Handle by_atom = pg6_win32_create_window_ex_a(
        0, reinterpret_cast<const char*>(static_cast<uintptr_t>(atom)), nullptr, 0, 0, 0, 1, 1, hwnd, 0, 0, nullptr);
    CHECK(by_atom != 0 && by_atom != hwnd);
    CHECK(pg6_win32_create_window_ex_a(0, class_name_ptr("TitleWindow"), nullptr, 0, 0, 0, 1, 1, 0xBAD, 0, 0, nullptr) == 0);
    CHECK(GetLastError() == ERROR_INVALID_WINDOW_HANDLE);

    // DefWindowProc: 0, except WM_NCCREATE which must be TRUE for creation to proceed.
    CHECK(pg6_win32_def_window_proc_a(hwnd, 0x0010, 0, 0) == 0);
    CHECK(pg6_win32_def_window_proc_a(hwnd, 0x0081, 0, 0) == 1);

    // ShowWindow returns the previous visibility.
    CHECK(pg6_win32_show_window(hwnd, 5) == 0);
    CHECK(pg6_win32_show_window(hwnd, 5) == 1);
    CHECK(pg6_win32_show_window(hwnd, 0) == 1);
    CHECK(pg6_win32_show_window(hwnd, 0) == 0);
    CHECK(pg6_win32_show_window(0x9999, 5) == 0 && GetLastError() == ERROR_INVALID_WINDOW_HANDLE);

    // The queue is empty unless the runtime posts into it.
    Pg6Msg msg;
    std::memset(&msg, 0x11, sizeof(msg));
    CHECK(pg6_win32_peek_message_a(&msg, 0, 0, 0, PM_REMOVE) == 0);
    CHECK(msg.message == 0x11111111u);  // untouched when nothing is available
    CHECK(pg6_win32_peek_message_a(nullptr, 0, 0, 0, PM_REMOVE) == 0);
    CHECK(GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(pg6_win32_peek_message_a(&msg, 0x9999, 0, 0, PM_REMOVE) == 0);
    CHECK(GetLastError() == ERROR_INVALID_WINDOW_HANDLE);

    CHECK(post_window_message(hwnd, 0x0100, 'A', 0x1E0001));
    CHECK(post_window_message(by_atom, 0x0200, 0, 5));
    CHECK(post_window_message(0, 0x0400, 1, 2));  // thread message
    CHECK(!post_window_message(0x9999, 0x0100, 0, 0));
    CHECK(queued_message_count() == 3);
    // PM_NOREMOVE peeks without consuming; filters select by window and range.
    CHECK(pg6_win32_peek_message_a(&msg, by_atom, 0, 0, 0) == 1 && msg.message == 0x0200 && msg.lparam == 5);
    CHECK(queued_message_count() == 3);
    CHECK(pg6_win32_peek_message_a(&msg, 0, 0x0150, 0x0300, PM_REMOVE) == 1 && msg.message == 0x0200);
    CHECK(queued_message_count() == 2);
    CHECK(pg6_win32_peek_message_a(&msg, ~uint64_t{0}, 0, 0, PM_REMOVE) == 1 && msg.hwnd == 0 && msg.message == 0x0400);
    CHECK(pg6_win32_peek_message_a(&msg, by_atom, 0, 0, PM_REMOVE) == 0);
    CHECK(pg6_win32_peek_message_a(&msg, 0, 0, 0, PM_REMOVE) == 1);
    CHECK(msg.hwnd == hwnd && msg.message == 0x0100 && msg.wparam == 'A' && msg.lparam == 0x1E0001);
    CHECK(queued_message_count() == 0);

    // TranslateMessage generates no characters; DispatchMessage counts but cannot call the guest.
    CHECK(pg6_win32_translate_message(&msg) == 0);
    CHECK(pg6_win32_translate_message(nullptr) == 0 && GetLastError() == ERROR_INVALID_PARAMETER);
    CHECK(dispatched_message_count() == 0);
    CHECK(pg6_win32_dispatch_message_a(&msg) == 0);
    CHECK(dispatched_message_count() == 1);
    CHECK(pg6_win32_dispatch_message_a(nullptr) == 0);

    reset_window_state();
    CHECK(window_count() == 0 && queued_message_count() == 0);
    CHECK(pg6_win32_create_window_ex_a(0, class_name_ptr("TitleWindow"), nullptr, 0, 0, 0, 1, 1, 0, 0, 0, nullptr) == 0);
}

}  // namespace

// Game data on HTTPFS: attribute queries only open the file, and a file's size is probed once
// (the probe is ~2*log2(size) range reads) and reused by later handles to the same path.
void test_httpfs_size_cache() {
    using playgta6::os::HttpfsMailbox;
    using playgta6::os::HttpfsMailboxClient;
    using playgta6::os::HttpfsOp;
    using playgta6::os::HttpfsStatus;
    HttpfsMailbox mailbox{};
    HttpfsMailboxClient::init_mailbox(mailbox);
    const uint64_t kSize = 123457;
    int opens = 0;
    int reads = 0;
    HttpfsMailboxClient client(&mailbox, [&](HttpfsMailbox& mb) {
        if (mb.op == static_cast<uint32_t>(HttpfsOp::Open)) {
            const std::string path(reinterpret_cast<const char*>(mb.payload),
                                   static_cast<size_t>(mb.length));
            ++opens;
            if (path.size() >= 7 && path.compare(path.size() - 7, 7, "big.rpf") == 0) {
                HttpfsMailboxClient::complete_response(mb, 3);
            } else {
                HttpfsMailboxClient::complete_response(mb, -2, HttpfsStatus::Error);
            }
        } else if (mb.op == static_cast<uint32_t>(HttpfsOp::Read)) {
            ++reads;
            const uint64_t n = mb.offset >= kSize ? 0 : std::min<uint64_t>(mb.length, kSize - mb.offset);
            std::memset(mb.payload, 0x5A, static_cast<size_t>(n));
            HttpfsMailboxClient::complete_response(mb, static_cast<int64_t>(n));
        } else {
            HttpfsMailboxClient::complete_response(mb, 0);
        }
    });
    set_httpfs_client(&client);

    CHECK(pg6_win32_get_file_attributes_w(u"G:\\data\\big.rpf") == FILE_ATTRIBUTE_READONLY);
    CHECK(opens == 1 && reads == 0);
    CHECK(pg6_win32_get_file_attributes_w(u"G:\\data\\missing.rpf") == kInvalidAttrs);
    CHECK(GetLastError() == ERROR_FILE_NOT_FOUND);

    const Handle a = CreateFileW(u"G:\\data\\big.rpf", GENERIC_READ, FILE_SHARE_READ, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
    CHECK(a != INVALID_HANDLE_VALUE);
    int64_t size = 0;
    CHECK(GetFileSizeEx(a, &size) && static_cast<uint64_t>(size) == kSize);
    const int probe_reads = reads;
    CHECK(probe_reads > 2);
    CloseHandle(a);

    const Handle b = CreateFileW(u"G:\\data\\big.rpf", GENERIC_READ, FILE_SHARE_READ, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
    CHECK(b != INVALID_HANDLE_VALUE);
    size = 0;
    CHECK(GetFileSizeEx(b, &size) && static_cast<uint64_t>(size) == kSize);
    CHECK(reads == probe_reads);  // served from the per-path cache
    CloseHandle(b);

    // A new client starts with an empty cache.
    set_httpfs_client(nullptr);
    set_httpfs_client(&client);
    const Handle c = CreateFileW(u"G:\\data\\big.rpf", GENERIC_READ, FILE_SHARE_READ, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
    CHECK(GetFileSizeEx(c, &size) && static_cast<uint64_t>(size) == kSize);
    CHECK(reads == 2 * probe_reads);
    CloseHandle(c);
    set_httpfs_client(nullptr);
}

int main() {
    test_httpfs_size_cache();
    test_directories_and_attributes();
    test_find_files();
    test_file_times_and_handles();
    test_move_and_disk_space();
    test_time();
    test_string_conversion();
    test_string_compare_and_locale();
    test_process_and_threads();
    test_sysinfo();
    test_debug_and_lifecycle();
    test_com();
    test_registry();
    test_bcrypt();
    test_windows();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_win32_compat failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All playgta6_win32_compat_tests passed.\n");
    return 0;
}
