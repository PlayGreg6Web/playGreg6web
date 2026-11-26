// Synthetic tests of the local XGameSave store (O34): containers, blobs, atomic updates, deletes,
// quota, persistence across provider re-init and error codes.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "os/xgamesave_local.h"

namespace gs = playgta6::os::gdk::gamesave;
namespace fs = std::filesystem;
using playgta6::os::gdk::E_INVALIDARG_HR;
using playgta6::os::gdk::S_OK_HR;

namespace {

int g_failures = 0;
#define CHECK(expr)                                                                     \
    do {                                                                                \
        if (!(expr)) {                                                                  \
            std::fprintf(stderr, "FAIL [test_xgamesave_local.cpp:%d]: %s\n", __LINE__, #expr); \
            ++g_failures;                                                               \
        }                                                                               \
    } while (0)

std::vector<uint8_t> bytes(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }

size_t count_temps(const fs::path& root) {
    size_t n = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().filename().string().compare(0, 5, ".tmp-") == 0) {
            ++n;
        }
    }
    return n;
}

void test_containers_and_blobs(const fs::path& base) {
    gs::Store store((base / "a").string());
    int64_t quota = 0;
    CHECK(store.remaining_quota(&quota) == S_OK_HR && quota == static_cast<int64_t>(gs::kDefaultQuotaBytes));

    CHECK(store.create_container("Slot1") == S_OK_HR);
    CHECK(store.create_container("Slot1") == S_OK_HR);  // opening again is fine
    CHECK(store.create_container("slot1") == S_OK_HR);  // names are case sensitive
    CHECK(store.create_container("Weird/Name:*?") == S_OK_HR);

    std::vector<gs::ContainerInfo> containers;
    CHECK(store.enumerate_containers("", &containers) == S_OK_HR && containers.size() == 3);
    CHECK(store.enumerate_containers("Slot", &containers) == S_OK_HR && containers.size() == 1 &&
          containers[0].name == "Slot1" && containers[0].blob_count == 0);

    // Round trip, overwrite, enumerate.
    CHECK(store.submit_update("Slot1", {{"header", bytes("hello")}, {"data", bytes("0123456789")}}, {}) == S_OK_HR);
    std::vector<uint8_t> out;
    CHECK(store.read_blob("Slot1", "header", &out) == S_OK_HR && out == bytes("hello"));
    CHECK(store.submit_update("Slot1", {{"header", bytes("hi")}}, {}) == S_OK_HR);
    CHECK(store.read_blob("Slot1", "header", &out) == S_OK_HR && out == bytes("hi"));
    std::vector<gs::BlobInfo> blobs;
    CHECK(store.enumerate_blobs("Slot1", &blobs) == S_OK_HR && blobs.size() == 2);
    CHECK(blobs[0].name == "data" && blobs[0].size == 10 && blobs[1].name == "header" && blobs[1].size == 2);
    CHECK(store.enumerate_blobs("slot1", &blobs) == S_OK_HR && blobs.empty());  // separate container
    CHECK(store.enumerate_containers("Slot1", &containers) == S_OK_HR && containers.size() == 1 &&
          containers[0].blob_count == 2 && containers[0].total_size == 12 && containers[0].last_modified != 0);

    // An empty blob is a blob.
    CHECK(store.submit_update("Slot1", {{"empty", {}}}, {}) == S_OK_HR);
    CHECK(store.read_blob("Slot1", "empty", &out) == S_OK_HR && out.empty());

    // Writes and deletes of one update apply together; a blob written and deleted ends deleted.
    CHECK(store.submit_update("Slot1", {{"new", bytes("n")}, {"tmp", bytes("t")}}, {"data", "tmp", "missing"}) == S_OK_HR);
    CHECK(store.read_blob("Slot1", "data", &out) == gs::E_GS_BLOB_NOT_FOUND);
    CHECK(store.read_blob("Slot1", "tmp", &out) == gs::E_GS_BLOB_NOT_FOUND);
    CHECK(store.read_blob("Slot1", "new", &out) == S_OK_HR && out == bytes("n"));
    CHECK(count_temps(base / "a") == 0);

    // Quota counts every blob of every container.
    CHECK(store.remaining_quota(&quota) == S_OK_HR);
    CHECK(quota == static_cast<int64_t>(gs::kDefaultQuotaBytes) - (2 + 0 + 1));

    // Delete a container (absent ones succeed); its blobs are gone and the quota is free again.
    CHECK(store.delete_container("Slot1") == S_OK_HR);
    CHECK(store.delete_container("Slot1") == S_OK_HR);
    CHECK(store.enumerate_blobs("Slot1", &blobs) == gs::E_GS_NO_SUCH_CONTAINER);
    CHECK(store.remaining_quota(&quota) == S_OK_HR && quota == static_cast<int64_t>(gs::kDefaultQuotaBytes));
}

void test_errors(const fs::path& base) {
    gs::Store store((base / "b").string());
    std::vector<uint8_t> out;
    std::vector<gs::BlobInfo> blobs;
    CHECK(store.create_container("") == gs::E_GS_INVALID_CONTAINER_NAME);
    CHECK(store.create_container(std::string(256, 'x')) == gs::E_GS_INVALID_CONTAINER_NAME);
    CHECK(store.create_container(std::string("bad\nname")) == gs::E_GS_INVALID_CONTAINER_NAME);
    CHECK(store.delete_container("") == gs::E_GS_INVALID_CONTAINER_NAME);
    CHECK(store.read_blob("nope", "b", &out) == gs::E_GS_NO_SUCH_CONTAINER);
    CHECK(store.enumerate_blobs("nope", &blobs) == gs::E_GS_NO_SUCH_CONTAINER);
    CHECK(store.submit_update("nope", {{"b", bytes("x")}}, {}) == gs::E_GS_NO_SUCH_CONTAINER);
    CHECK(store.create_container("c") == S_OK_HR);
    CHECK(store.read_blob("c", "nope", &out) == gs::E_GS_BLOB_NOT_FOUND);
    CHECK(store.read_blob("c", "", &out) == E_INVALIDARG_HR);
    CHECK(store.submit_update("c", {{"", bytes("x")}}, {}) == E_INVALIDARG_HR);
    CHECK(store.submit_update("c", {{std::string(129, 'b'), bytes("x")}}, {}) == E_INVALIDARG_HR);
    CHECK(store.submit_update("c", {}, {""}) == E_INVALIDARG_HR);
    CHECK(store.submit_update("c", {{"big", std::vector<uint8_t>(gs::kMaxBlobBytes + 1)}}, {}) == gs::E_GS_UPDATE_TOO_BIG);
    CHECK(store.submit_update("c", {{"a", std::vector<uint8_t>(gs::kMaxBlobBytes / 2 + 1)}, {"b", std::vector<uint8_t>(gs::kMaxBlobBytes / 2)}}, {}) == gs::E_GS_UPDATE_TOO_BIG);
    CHECK(store.submit_update("c", {{"a", std::vector<uint8_t>(gs::kMaxBlobBytes / 2)}, {"b", std::vector<uint8_t>(gs::kMaxBlobBytes / 2)}}, {}) == S_OK_HR);
    CHECK(gs::E_GS_UPDATE_TOO_BIG == static_cast<playgta6::os::gdk::HResult>(0x80830005u) && gs::E_GS_QUOTA_EXCEEDED == static_cast<playgta6::os::gdk::HResult>(0x80830006u) && gs::E_GS_PROVIDED_BUFFER_TOO_SMALL == static_cast<playgta6::os::gdk::HResult>(0x80830007u) && gs::E_GS_BLOB_NOT_FOUND == static_cast<playgta6::os::gdk::HResult>(0x80830008u));
    CHECK(store.enumerate_blobs("c", &blobs) == S_OK_HR && blobs.size() == 2);
}

void test_quota_and_atomicity(const fs::path& base) {
    gs::Store store((base / "q").string(), 100);
    CHECK(store.create_container("c") == S_OK_HR);
    CHECK(store.submit_update("c", {{"a", std::vector<uint8_t>(60, 1)}}, {}) == S_OK_HR);
    int64_t quota = 0;
    CHECK(store.remaining_quota(&quota) == S_OK_HR && quota == 40);

    // The second write does not fit: nothing of the update is applied (a is untouched, b absent).
    CHECK(store.submit_update("c", {{"b", std::vector<uint8_t>(30, 2)}, {"a", std::vector<uint8_t>(80, 3)}}, {}) ==
          gs::E_GS_QUOTA_EXCEEDED);
    std::vector<uint8_t> out;
    CHECK(store.read_blob("c", "a", &out) == S_OK_HR && out == std::vector<uint8_t>(60, 1));
    CHECK(store.read_blob("c", "b", &out) == gs::E_GS_BLOB_NOT_FOUND);
    CHECK(count_temps(base / "q") == 0);

    // Replacing a blob counts the difference only; a delete in the same update frees room.
    CHECK(store.submit_update("c", {{"a", std::vector<uint8_t>(100, 4)}}, {}) == S_OK_HR);
    CHECK(store.remaining_quota(&quota) == S_OK_HR && quota == 0);
    CHECK(store.submit_update("c", {{"b", std::vector<uint8_t>(50, 5)}}, {"a"}) == S_OK_HR);
    CHECK(store.remaining_quota(&quota) == S_OK_HR && quota == 50);

    // Quota spans containers.
    CHECK(store.create_container("d") == S_OK_HR);
    CHECK(store.submit_update("d", {{"x", std::vector<uint8_t>(51, 6)}}, {}) == gs::E_GS_QUOTA_EXCEEDED);
    CHECK(store.submit_update("d", {{"x", std::vector<uint8_t>(50, 6)}}, {}) == S_OK_HR);

    // A stale temp file (an interrupted update) is not a blob, not counted, and is cleaned on open.
    {
        std::ofstream stale(base / "q" / "local_stale", std::ios::binary);
    }
    std::vector<gs::ContainerInfo> containers;
    CHECK(store.enumerate_containers("", &containers) == S_OK_HR && containers.size() == 2);
}

void test_persistence_and_stale_temps(const fs::path& base) {
    gs::set_save_dir((base / "saves").string());
    {
        gs::Store store = gs::open_store("cfg-1");
        CHECK(store.create_container("Persist") == S_OK_HR);
        CHECK(store.submit_update("Persist", {{"b", bytes("kept")}}, {}) == S_OK_HR);
    }
    {
        // A new provider (re-init) on the same directory sees the data.
        gs::Store store = gs::open_store("cfg-1");
        std::vector<uint8_t> out;
        CHECK(store.read_blob("Persist", "b", &out) == S_OK_HR && out == bytes("kept"));
        std::vector<gs::ContainerInfo> containers;
        CHECK(store.enumerate_containers("", &containers) == S_OK_HR && containers.size() == 1);
        // Another configuration id is another store.
        gs::Store other = gs::open_store("cfg-2");
        CHECK(other.enumerate_containers("", &containers) == S_OK_HR && containers.empty());
        // A leftover temp file from a crashed update is removed when the container is opened.
        const fs::path container = fs::path(gs::save_dir()) / "cfg-1" / "local" / "c-^persist";
        CHECK(fs::is_directory(container));
        { std::ofstream tmp(container / ".tmp-999", std::ios::binary); tmp << "junk"; }
        std::vector<gs::BlobInfo> blobs;
        CHECK(store.enumerate_blobs("Persist", &blobs) == S_OK_HR && blobs.size() == 1);  // not a blob
        CHECK(store.create_container("Persist") == S_OK_HR);
        CHECK(count_temps(container) == 0);
        CHECK(store.read_blob("Persist", "b", &out) == S_OK_HR && out == bytes("kept"));
    }
    gs::set_save_dir("");
    CHECK(gs::save_dir() == gs::default_save_dir());
}

void test_display_name(const fs::path& base) {
    gs::Store store((base / "d").string());
    CHECK(store.set_display_name("Missing", "x") == gs::E_GS_NO_SUCH_CONTAINER);
    CHECK(store.set_display_name("", "x") == gs::E_GS_INVALID_CONTAINER_NAME);
    CHECK(store.create_container("Slot") == S_OK_HR);
    CHECK(store.set_display_name("Slot", std::string(gs::kMaxDisplayNameChars + 1, 'a')) == E_INVALIDARG_HR);
    CHECK(store.submit_update("Slot", {{"b", bytes("123")}}, {}) == S_OK_HR);
    CHECK(store.set_display_name("Slot", "Story save 1") == S_OK_HR);
    std::vector<gs::ContainerInfo> containers;
    CHECK(store.enumerate_containers("Slot", &containers) == S_OK_HR && containers.size() == 1 &&
          containers[0].display_name == "Story save 1" && containers[0].blob_count == 1 &&
          containers[0].total_size == 3);  // the display name is neither a blob nor quota
    int64_t quota = 0;
    CHECK(store.remaining_quota(&quota) == S_OK_HR && quota == static_cast<int64_t>(gs::kDefaultQuotaBytes - 3));
    gs::Store again((base / "d").string());  // kept across a provider re-init
    CHECK(again.enumerate_containers("Slot", &containers) == S_OK_HR && containers[0].display_name == "Story save 1");
    CHECK(count_temps(base / "d") == 0);
}

void test_container_info(const fs::path& base) {
    gs::Store store((base / "i").string());
    gs::ContainerInfo info;
    CHECK(store.get_container_info("", &info) == gs::E_GS_INVALID_CONTAINER_NAME);
    CHECK(store.get_container_info("Nope", &info) == gs::E_GS_NO_SUCH_CONTAINER);
    CHECK(store.create_container("Slot") == S_OK_HR);
    CHECK(store.get_container_info("Slot", &info) == S_OK_HR && info.name == "Slot" && info.blob_count == 0 &&
          info.total_size == 0 && info.display_name.empty());
    CHECK(store.submit_update("Slot", {{"a", bytes("12")}, {"b", bytes("345")}}, {}) == S_OK_HR);
    CHECK(store.set_display_name("Slot", "Save 1") == S_OK_HR);
    CHECK(store.get_container_info("Slot", &info) == S_OK_HR && info.blob_count == 2 && info.total_size == 5 &&
          info.display_name == "Save 1" && info.last_modified != 0);
    std::vector<gs::ContainerInfo> all;
    CHECK(store.enumerate_containers("Slot", &all) == S_OK_HR && all.size() == 1 &&
          all[0].total_size == info.total_size && all[0].blob_count == info.blob_count);
    CHECK(store.delete_container("Slot") == S_OK_HR);
    CHECK(store.get_container_info("Slot", &info) == gs::E_GS_NO_SUCH_CONTAINER);
}

}  // namespace

void test_crash_between_swap_steps(const fs::path& base) {
    const fs::path root = base / "crash";
    for (int point = 1; point <= 2; ++point) {
        gs::Store store((root / std::to_string(point)).string());
        CHECK(store.create_container("S") == S_OK_HR);
        CHECK(store.submit_update("S", {{"a", bytes("old-a")}, {"b", bytes("old-b")}}, {}) == S_OK_HR);
        gs::Store::set_test_crash_point(point);
        CHECK(store.submit_update("S", {{"a", bytes("new-a")}, {"c", bytes("new-c")}}, {"b"}) != S_OK_HR);
        gs::Store::set_test_crash_point(0);
        // A new Store on the same root recovers: the container is whole and is the old version.
        gs::Store again((root / std::to_string(point)).string());
        std::vector<gs::BlobInfo> blobs;
        CHECK(again.enumerate_blobs("S", &blobs) == S_OK_HR && blobs.size() == 2);
        std::vector<uint8_t> out;
        CHECK(again.read_blob("S", "a", &out) == S_OK_HR && out == bytes("old-a"));
        CHECK(again.read_blob("S", "b", &out) == S_OK_HR && out == bytes("old-b"));
        CHECK(again.read_blob("S", "c", &out) == gs::E_GS_BLOB_NOT_FOUND);
        // And a following update works.
        CHECK(again.submit_update("S", {{"a", bytes("new-a")}}, {"b"}) == S_OK_HR);
        CHECK(again.read_blob("S", "a", &out) == S_OK_HR && out == bytes("new-a"));
        CHECK(again.read_blob("S", "b", &out) == gs::E_GS_BLOB_NOT_FOUND);
    }
}

// Save dir outside the run dir, "simulated exit" (everything dropped, flush) and a second start that
// opens the same dir finds the first one's saves; the dirty signal lists the changed container.
void test_save_dir_survives_restart(const fs::path& base) {
    const std::string dir = (base / "persist").string();
    gs::set_save_dir(dir);
    CHECK(gs::save_dir() == dir);
    const uint64_t gen0 = gs::storage_generation();
    {
        gs::Store store = gs::open_store("title-1");
        CHECK(store.create_container("slot") == S_OK_HR);
        CHECK(store.submit_update("slot", {{"data", bytes("saved")}}, {}) == S_OK_HR);
    }
    CHECK(gs::storage_dirty() && gs::storage_generation() > gen0);
    const std::vector<std::string> changed = gs::take_changed_paths();
    CHECK(!changed.empty());
    CHECK(gs::take_changed_paths().empty());
    gs::flush_all();  // exit / XLaunchNewGame
    gs::set_save_dir("");  // process gone: the next start configures again
    CHECK(!gs::save_dir().empty() && gs::save_dir() == gs::default_save_dir());
    gs::set_save_dir(dir);
    gs::Store second = gs::open_store("title-1");
    std::vector<uint8_t> out;
    CHECK(second.read_blob("slot", "data", &out) == S_OK_HR && out == bytes("saved"));
    CHECK(count_temps(dir) == 0);
    gs::set_save_dir("");
}

int main() {
    const fs::path base = fs::temp_directory_path() / "pg6_o34_store_test";
    fs::remove_all(base);
    fs::create_directories(base);
    test_containers_and_blobs(base);
    test_errors(base);
    test_quota_and_atomicity(base);
    test_persistence_and_stale_temps(base);
    test_display_name(base);
    test_container_info(base);
    test_crash_between_swap_steps(base);
    test_save_dir_survives_restart(base);
    fs::remove_all(base);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_xgamesave_local: ok");
    return 0;
}
