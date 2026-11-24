// File shims of task O26 on top of the O15 file system (os/win32_compat_file.cpp) and the
// /userdata overlay seam of os/win32_kernel.cpp: FindFirstFileExW, GetFileInformationByHandle,
// GetTempPathW, CopyFile2, CreateHardLinkW, CreateSymbolicLinkW.
//
// Limitations (all deliberate): FindExSearchLimitToDirectories is a hint that is ignored (the
// documented behaviour for file systems that cannot filter; callers check the attributes);
// FindExSearchLimitToDevices is ERROR_NOT_SUPPORTED. CopyFile2 does not call the progress routine
// and does not poll the cancel flag (both are guest pointers inside the parameter block); it
// copies file contents only (attributes and timestamps of the copy are the new file's).
// Hard links and symbolic links are not supported by the in-memory overlay: CreateHardLinkW fails
// with ERROR_NOT_SUPPORTED and CreateSymbolicLinkW with ERROR_PRIVILEGE_NOT_HELD (what a Windows
// process without the privilege gets), or ERROR_NOT_SUPPORTED when it asks for unprivileged
// creation.

#include <cstring>
#include <string>
#include <vector>

#include "os/win32_compat.h"
#include "os/win32_compat_internal.h"
#include "os/win32_rtl.h"

namespace win32 = playgta6::os::win32;
namespace cd = playgta6::os::win32::compat_detail;

namespace {

constexpr char16_t kTempPath[] = u"T:\\userdata\\temp\\";
constexpr char kTempKey[] = "/userdata/temp";
constexpr uint32_t kVolumeSerialUserdata = 0x50473655u;
constexpr uint32_t kVolumeSerialGameData = 0x50473654u;
constexpr uint32_t kErrorSharingViolation = 32u;
constexpr size_t kCopyChunk = 64 * 1024;

uint64_t fnv1a(const void* data, size_t size, uint64_t seed) {
    uint64_t h = seed;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h = (h ^ p[i]) * 0x100000001B3ull;
    }
    return h;
}

bool same_key(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (cd::fold_char(static_cast<unsigned char>(a[i])) !=
            cd::fold_char(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" {

Pg6Handle pg6_win32_find_first_file_ex_w(const char16_t* file_name,
                                         int32_t info_level_id,
                                         void* find_file_data,
                                         int32_t search_op,
                                         void* search_filter,
                                         uint32_t additional_flags) {
    constexpr uint32_t kKnownFlags = 0x1u | 0x2u | 0x4u;  // CASE_SENSITIVE, LARGE_FETCH, ONLY_ON_DISK
    if (find_file_data == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return win32::INVALID_HANDLE_VALUE;
    }
    std::memset(find_file_data, 0, sizeof(Pg6Win32FindDataW));
    if (info_level_id != win32::FindExInfoStandard && info_level_id != win32::FindExInfoBasic) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return win32::INVALID_HANDLE_VALUE;
    }
    if (search_op == win32::FindExSearchLimitToDevices) {
        win32::SetLastError(win32::ERROR_NOT_SUPPORTED);
        return win32::INVALID_HANDLE_VALUE;
    }
    if (search_op != win32::FindExSearchNameMatch && search_op != win32::FindExSearchLimitToDirectories) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return win32::INVALID_HANDLE_VALUE;
    }
    if (search_filter != nullptr || (additional_flags & ~kKnownFlags) != 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);  // no filter structure is defined
        return win32::INVALID_HANDLE_VALUE;
    }
    // Both info levels return WIN32_FIND_DATAW; the basic level leaves cAlternateFileName empty,
    // which is what the O15 implementation always does (there are no 8.3 names).
    return pg6_win32_find_first_file_w(file_name, static_cast<Pg6Win32FindDataW*>(find_file_data));
}

int32_t pg6_win32_get_file_information_by_handle(Pg6Handle file,
                                                 Pg6ByHandleFileInformation* file_information) {
    if (file_information == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    *file_information = Pg6ByHandleFileInformation{};
    win32::FileHandleInfo info;
    if (!win32::query_file_handle(file, &info)) {
        return 0;  // ERROR_INVALID_HANDLE set by the seam
    }
    uint64_t size = 0;
    uint64_t created = 0, accessed = 0, written = 0;
    uint32_t attributes = 0;
    uint64_t index = 0;
    if (info.is_userdata) {
        win32::UserdataEntry e;
        const uint32_t err = win32::userdata_stat(info.userdata_key, &e);
        if (err != win32::ERROR_SUCCESS) {
            win32::SetLastError(err);
            return 0;
        }
        size = e.size;
        created = e.created;
        accessed = e.accessed;
        written = e.written;
        attributes = e.is_directory ? (win32::FILE_ATTRIBUTE_DIRECTORY | (e.attributes & ~0x80u))
                                    : (e.attributes != 0 ? e.attributes : win32::FILE_ATTRIBUTE_ARCHIVE);
        index = fnv1a(info.userdata_key.data(), info.userdata_key.size(), 0xcbf29ce484222325ull);
        file_information->volume_serial_number = kVolumeSerialUserdata;
    } else {
        int64_t file_size = 0;
        if (!win32::GetFileSizeEx(file, &file_size)) {
            return 0;
        }
        size = file_size > 0 ? static_cast<uint64_t>(file_size) : 0;
        // Read-only game data: fixed timestamp (2020-01-01T00:00:00Z) like the O15 queries.
        created = accessed = written = 132223104000000000ULL;
        attributes = win32::FILE_ATTRIBUTE_READONLY;
        index = fnv1a(&file, sizeof(file), 0xcbf29ce484222325ull);
        file_information->volume_serial_number = kVolumeSerialGameData;
    }
    file_information->file_attributes = attributes;
    file_information->creation_time = win32::file_time_from_ticks(created);
    file_information->last_access_time = win32::file_time_from_ticks(accessed);
    file_information->last_write_time = win32::file_time_from_ticks(written);
    file_information->file_size_high = static_cast<uint32_t>(size >> 32);
    file_information->file_size_low = static_cast<uint32_t>(size & 0xFFFFFFFFull);
    file_information->number_of_links = 1;
    file_information->file_index_high = static_cast<uint32_t>(index >> 32);
    file_information->file_index_low = static_cast<uint32_t>(index & 0xFFFFFFFFull);
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 1;
}

uint32_t pg6_win32_get_temp_path_w(uint32_t buffer_length, char16_t* buffer) {
    constexpr uint32_t length = static_cast<uint32_t>(sizeof(kTempPath) / sizeof(char16_t)) - 1;
    // Make sure the directory exists in the overlay (idempotent).
    const uint32_t made = win32::userdata_create_directory(kTempKey);
    (void)made;
    if (buffer_length > length && buffer != nullptr) {
        std::memcpy(buffer, kTempPath, (length + 1) * sizeof(char16_t));
        win32::SetLastError(win32::ERROR_SUCCESS);
        return length;  // characters written, without the NUL
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return length + 1;  // required size including the NUL
}

int32_t pg6_win32_copy_file2(const char16_t* existing_file_name,
                             const char16_t* new_file_name,
                             void* extended_parameters) {
    constexpr uint32_t kParametersSize = 32;  // COPYFILE2_EXTENDED_PARAMETERS on x64
    constexpr uint32_t kKnownCopyFlags = 0x1u | 0x2u | 0x4u | 0x8u | 0x800u | 0x1000u | 0x2000u |
                                         0x4000u | 0x40000u | 0x400000u | 0x800000u;
    if (existing_file_name == nullptr || new_file_name == nullptr) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return win32::hresult_from_win32(win32::ERROR_INVALID_PARAMETER);
    }
    uint32_t copy_flags = 0;
    if (extended_parameters != nullptr) {
        uint32_t header[2];
        std::memcpy(header, extended_parameters, sizeof(header));  // dwSize, dwCopyFlags
        if (header[0] != kParametersSize || (header[1] & ~kKnownCopyFlags) != 0) {
            win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
            return win32::hresult_from_win32(win32::ERROR_INVALID_PARAMETER);
        }
        copy_flags = header[1];
    }

    cd::ResolvedPath from, to;
    uint32_t err = cd::resolve_path(existing_file_name, &from);
    if (err == win32::ERROR_SUCCESS) {
        err = cd::resolve_path(new_file_name, &to);
    }
    if (err != win32::ERROR_SUCCESS) {
        win32::SetLastError(err);
        return win32::hresult_from_win32(err);
    }
    if (same_key(from.key, to.key)) {
        win32::SetLastError(kErrorSharingViolation);
        return win32::hresult_from_win32(kErrorSharingViolation);
    }

    const win32::Handle src = win32::CreateFileW(existing_file_name, win32::GENERIC_READ,
                                                 win32::FILE_SHARE_READ, nullptr, win32::OPEN_EXISTING,
                                                 win32::FILE_ATTRIBUTE_NORMAL);
    if (src == win32::INVALID_HANDLE_VALUE) {
        const uint32_t e = win32::GetLastError();
        return win32::hresult_from_win32(e != 0 ? e : win32::ERROR_FILE_NOT_FOUND);
    }
    const win32::Handle dst = win32::CreateFileW(
        new_file_name, win32::GENERIC_WRITE, 0, nullptr,
        (copy_flags & win32::COPY_FILE_FAIL_IF_EXISTS) != 0 ? win32::CREATE_NEW : win32::CREATE_ALWAYS,
        win32::FILE_ATTRIBUTE_NORMAL);
    if (dst == win32::INVALID_HANDLE_VALUE) {
        const uint32_t e = win32::GetLastError();
        win32::CloseHandle(src);
        return win32::hresult_from_win32(e != 0 ? e : win32::ERROR_ACCESS_DENIED);
    }

    std::vector<uint8_t> chunk(kCopyChunk);
    uint32_t failure = win32::ERROR_SUCCESS;
    for (;;) {
        uint32_t got = 0;
        if (!win32::ReadFile(src, chunk.data(), static_cast<uint32_t>(chunk.size()), &got)) {
            failure = win32::GetLastError();
            if (failure == win32::ERROR_SUCCESS) {
                failure = win32::ERROR_READ_FAULT;
            }
            break;
        }
        if (got == 0) {
            break;
        }
        uint32_t put = 0;
        if (!win32::WriteFile(dst, chunk.data(), got, &put) || put != got) {
            failure = win32::GetLastError();
            if (failure == win32::ERROR_SUCCESS) {
                failure = win32::ERROR_WRITE_PROTECT;
            }
            break;
        }
    }
    win32::CloseHandle(src);
    win32::CloseHandle(dst);
    if (failure != win32::ERROR_SUCCESS) {
        pg6_win32_delete_file_w(new_file_name);  // do not leave a partial copy behind
        win32::SetLastError(failure);
        return win32::hresult_from_win32(failure);
    }
    win32::SetLastError(win32::ERROR_SUCCESS);
    return 0;  // S_OK
}

int32_t pg6_win32_create_hard_link_w(const char16_t* file_name,
                                     const char16_t* existing_file_name,
                                     void* security_attributes) {
    (void)security_attributes;
    if (file_name == nullptr || existing_file_name == nullptr || file_name[0] == 0 ||
        existing_file_name[0] == 0) {
        win32::SetLastError(win32::ERROR_PATH_NOT_FOUND);
        return 0;
    }
    if (pg6_win32_get_file_attributes_w(existing_file_name) == win32::INVALID_FILE_ATTRIBUTES) {
        return 0;  // last error describes why the existing file is not usable
    }
    win32::SetLastError(win32::ERROR_NOT_SUPPORTED);
    return 0;
}

uint8_t pg6_win32_create_symbolic_link_w(const char16_t* symlink_file_name,
                                         const char16_t* target_file_name,
                                         uint32_t flags) {
    constexpr uint32_t kKnown =
        win32::SYMBOLIC_LINK_FLAG_DIRECTORY | win32::SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE;
    if (symlink_file_name == nullptr || target_file_name == nullptr || symlink_file_name[0] == 0 ||
        target_file_name[0] == 0 || (flags & ~kKnown) != 0) {
        win32::SetLastError(win32::ERROR_INVALID_PARAMETER);
        return 0;
    }
    win32::SetLastError((flags & win32::SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE) != 0
                            ? win32::ERROR_NOT_SUPPORTED
                            : win32::ERROR_PRIVILEGE_NOT_HELD);
    return 0;
}

}  // extern "C"
