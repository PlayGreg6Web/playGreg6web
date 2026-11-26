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
// An update (writes + deletes) is checked (16 MiB per update, quota), built as a complete new
// copy of the container in a temp directory next to it, and committed by swapping directories
// (old aside, new in, old removed). A crash leaves the old or the new container; the next Store
// on the same root finishes or rolls back a half-done swap.
namespace playgta6::os::gdk::gamesave {

// XGameSave error codes (FACILITY 0x283), values per the public GDK XGameSave documentation
// ("XGameSave (API contents)", Error Codes).
inline constexpr HResult E_GS_INVALID_CONTAINER_NAME = static_cast<HResult>(0x80830001u);
inline constexpr HResult E_GS_NO_ACCESS = static_cast<HResult>(0x80830002u);
inline constexpr HResult E_GS_OUT_OF_LOCAL_STORAGE = static_cast<HResult>(0x80830003u);
inline constexpr HResult E_GS_UPDATE_TOO_BIG = static_cast<HResult>(0x80830005u);
inline constexpr HResult E_GS_QUOTA_EXCEEDED = static_cast<HResult>(0x80830006u);
inline constexpr HResult E_GS_PROVIDED_BUFFER_TOO_SMALL = static_cast<HResult>(0x80830007u);
inline constexpr HResult E_GS_BLOB_NOT_FOUND = static_cast<HResult>(0x80830008u);
inline constexpr HResult E_GS_CONTAINER_SYNC_FAILED = static_cast<HResult>(0x8083000Bu);
// Not in the public list; the value the platform runtime is observed to use for a missing container.
inline constexpr HResult E_GS_NO_SUCH_CONTAINER = static_cast<HResult>(0x80830010u);

inline constexpr uint64_t kDefaultQuotaBytes = 256ull * 1024 * 1024;
inline constexpr size_t kMaxContainerNameChars = 255;
inline constexpr size_t kMaxBlobNameChars = 128;
// GS_MAX_BLOB_SIZE: the total size of the writes of ONE update (so also of one blob).
inline constexpr uint64_t kMaxBlobBytes = 16ull * 1024 * 1024;
inline constexpr uint64_t kMaxUpdateBytes = kMaxBlobBytes;
inline constexpr size_t kMaxDisplayNameChars = 255;

// Host directory that holds all saves. Default: a stable directory outside the run dir, under the
// user's app data (%LOCALAPPDATA%\playgta6web\saves, ~/Library/Application Support/playgta6web/saves,
// $XDG_DATA_HOME or ~/.local/share/playgta6web/saves); "pg6-saves" in the browser build or when no
// app-data location exists. Each title has its own sub-directory (configuration id), so a second
// process start with the same configuration finds the first one's saves.
void set_save_dir(const std::string& dir);  // "" = default
std::string save_dir();
std::string default_save_dir();

// fsync of everything below the save root (suspend / exit / XLaunchNewGame). Every update is already
// flushed when it completes, so this only covers late writes.
void flush_all();

// Browser persistence contract: every completed write marks storage dirty and records the changed
// container directories (paths below save_dir(), '/' separated). The host takes the list and
// persists those trees to IndexedDB/OPFS.
bool storage_dirty();
uint64_t storage_generation();
std::vector<std::string> take_changed_paths();  // returns and clears the list

// Name checks of the store (non-empty, printable, short enough for a path component).
bool valid_container_name(const std::string& name);
bool valid_blob_name(const std::string& name);

struct ContainerInfo {
    std::string name;
    std::string display_name;  // as last given to an update ("" when none was given)
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
    // One container's info (what enumerate_containers reports for it); E_GS_NO_SUCH_CONTAINER when absent.
    HResult get_container_info(const std::string& name, ContainerInfo* out) const;
    HResult enumerate_blobs(const std::string& container, std::vector<BlobInfo>* out) const;
    HResult read_blob(const std::string& container, const std::string& blob, std::vector<uint8_t>* out) const;
    HResult submit_update(const std::string& container, const std::vector<BlobWrite>& writes,
                          const std::vector<std::string>& deletes);
    // Display name shown for an existing container (kept next to its blobs).
    HResult set_display_name(const std::string& container, const std::string& display_name);

    // Test hook: simulate a crash inside submit_update. 1 = after the new copy is built (nothing
    // swapped), 2 = after the old container was moved aside but before the new one is in place.
    // The update then returns E_FAIL without cleaning up. 0 = off.
    static void set_test_crash_point(int point);

private:
    std::string root_;
    uint64_t quota_;
};

// Store of one (configuration id, local user) below save_dir().
Store open_store(const std::string& configuration_id);

}  // namespace playgta6::os::gdk::gamesave
