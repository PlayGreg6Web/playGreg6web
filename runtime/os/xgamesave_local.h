#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "os/xgameruntime.h"

// Local XGameSave store (O34): what a console does offline. Saves live in a host directory
// (boot option --save-dir) and are kept between runs; nothing is synchronised anywhere and the
// runtime never claims that it is (needsSync / cloud results stay "offline").
//
// Layout below <save-dir>: <configuration id>/<user>/c-<container>/b-<blob>. Names are
// percent-encoded so any name is a safe, case-insensitive-filesystem-safe path component.
// A blob write is a temp file in the container directory renamed over the blob; an update
// (writes + deletes) is staged completely (and checked against the quota) before the first
// rename, so a failing update changes nothing.
namespace playgta6::os::gdk::gamesave {

// XGameSave error codes (FACILITY 0x283).
inline constexpr HResult E_GS_INVALID_CONTAINER_NAME = static_cast<HResult>(0x80830001u);
inline constexpr HResult E_GS_NO_MORE_DATA = static_cast<HResult>(0x80830002u);
inline constexpr HResult E_GS_QUOTA_EXCEEDED = static_cast<HResult>(0x80830003u);
inline constexpr HResult E_GS_PROVIDED_BUFFER_TOO_SMALL = static_cast<HResult>(0x80830004u);
inline constexpr HResult E_GS_BLOB_NOT_FOUND = static_cast<HResult>(0x80830005u);
inline constexpr HResult E_GS_NO_SUCH_CONTAINER = static_cast<HResult>(0x80830006u);
inline constexpr HResult E_GS_CONTAINER_SYNC_FAILED = static_cast<HResult>(0x80830007u);

inline constexpr uint64_t kDefaultQuotaBytes = 256ull * 1024 * 1024;
inline constexpr size_t kMaxContainerNameChars = 255;
inline constexpr size_t kMaxBlobNameChars = 128;
inline constexpr uint64_t kMaxBlobBytes = 16ull * 1024 * 1024;

// Host directory that holds all saves (default "pg6-saves", relative to the working directory).
void set_save_dir(const std::string& dir);
std::string save_dir();

struct ContainerInfo {
    std::string name;
    uint32_t blob_count = 0;
    uint64_t total_size = 0;
    uint64_t last_modified = 0;  // FILETIME ticks (UTC)
};
struct BlobInfo {
    std::string name;
    uint32_t size = 0;
};
struct BlobWrite {
    std::string name;
    std::vector<uint8_t> data;
};

class Store {
public:
    // root: <save-dir>/<configuration>/<user>; created on first use.
    explicit Store(const std::string& root, uint64_t quota_bytes = kDefaultQuotaBytes);

    HResult remaining_quota(int64_t* out) const;
    HResult create_container(const std::string& name);  // opens an existing one as well
    HResult delete_container(const std::string& name);
    HResult enumerate_containers(const std::string& prefix, std::vector<ContainerInfo>* out) const;
    HResult enumerate_blobs(const std::string& container, std::vector<BlobInfo>* out) const;
    HResult read_blob(const std::string& container, const std::string& blob, std::vector<uint8_t>* out) const;
    HResult submit_update(const std::string& container, const std::vector<BlobWrite>& writes,
                          const std::vector<std::string>& deletes);

private:
    std::string root_;
    uint64_t quota_;
};

// Store of one (configuration id, local user) below save_dir().
Store open_store(const std::string& configuration_id);

}  // namespace playgta6::os::gdk::gamesave
