#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "guest/boot.h"
#include "guest/guest_runtime.h"
#include "guest/program_loader.h"
#include "os/image_loader.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "wasm_boot.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_wasm_boot.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using playgta6::boot::WasmBootConfig;
using playgta6::boot::WasmBootResult;

struct FakeFile {
    std::vector<uint8_t> bytes;
};

struct FakeFs {
    std::map<std::string, FakeFile> files;
    std::map<int32_t, std::string> handles;
    int32_t next_handle = 1;
    int opens = 0;
    int reads = 0;
    int closes = 0;
    int fail_read_on = -1;  // 1-based read count to fail
    int64_t force_read_rc = 0;
    bool force_read = false;
};

FakeFs* g_fs = nullptr;
std::vector<std::string> g_logs;

int32_t fake_open(std::string_view rel_path) {
    ++g_fs->opens;
    const std::string key(rel_path);
    if (g_fs->files.find(key) == g_fs->files.end()) {
        return -2;
    }
    const int32_t h = g_fs->next_handle++;
    g_fs->handles[h] = key;
    return h;
}

int64_t fake_read(int32_t handle, uint64_t offset, void* dst, size_t len) {
    ++g_fs->reads;
    if (g_fs->fail_read_on > 0 && g_fs->reads == g_fs->fail_read_on) {
        return -9;
    }
    if (g_fs->force_read) {
        return g_fs->force_read_rc;
    }
    auto it = g_fs->handles.find(handle);
    if (it == g_fs->handles.end()) {
        return -9;
    }
    const FakeFile& file = g_fs->files[it->second];
    if (offset > file.bytes.size()) {
        return -9;
    }
    const size_t avail = file.bytes.size() - static_cast<size_t>(offset);
    const size_t n = avail < len ? avail : len;
    if (n > 0 && dst != nullptr) {
        std::memcpy(dst, file.bytes.data() + static_cast<size_t>(offset), n);
    }
    return static_cast<int64_t>(n);
}

void fake_close(int32_t handle) {
    ++g_fs->closes;
    g_fs->handles.erase(handle);
}

void fake_log(std::string_view message) {
    g_logs.emplace_back(message);
}

void put_file(FakeFs& fs, const std::string& path, std::string_view text) {
    FakeFile f;
    f.bytes.assign(text.begin(), text.end());
    fs.files[path] = std::move(f);
}

void put_bytes(FakeFs& fs, const std::string& path, std::vector<uint8_t> bytes) {
    FakeFile f;
    f.bytes = std::move(bytes);
    fs.files[path] = std::move(f);
}

std::vector<uint8_t> pgim_stub(size_t extra = 12) {
    std::vector<uint8_t> bytes(4 + extra, 0);
    bytes[0] = 'P';
    bytes[1] = 'G';
    bytes[2] = 'I';
    bytes[3] = 'M';
    for (size_t i = 0; i < extra; ++i) {
        bytes[4 + i] = static_cast<uint8_t>(i + 1);
    }
    return bytes;
}

WasmBootConfig make_cfg(FakeFs& fs) {
    g_fs = &fs;
    g_logs.clear();
    WasmBootConfig cfg;
    cfg.open_fn = fake_open;
    cfg.read_fn = fake_read;
    cfg.close_fn = fake_close;
    cfg.log_fn = fake_log;
    cfg.read_chunk_bytes = 4;
    return cfg;
}

void test_chunked_httpfs_reads() {
    FakeFs fs;
    const std::vector<uint8_t> payload = pgim_stub(12);  // 16 bytes, 4-byte chunks
    put_bytes(fs, "boot/guest.pgim", payload);
    WasmBootConfig cfg = make_cfg(fs);

    std::vector<uint8_t> out;
    std::string err;
    CHECK(playgta6::boot::load_file_via_httpfs(cfg, "boot/guest.pgim", out, err));
    CHECK(err.empty());
    CHECK(out == payload);
    CHECK(fs.opens == 1);
    CHECK(fs.closes == 1);
    CHECK(fs.handles.empty());
    CHECK(fs.reads >= 4);  // 16 bytes / 4 + possible short/eof
}

void test_close_on_read_error() {
    FakeFs fs;
    put_bytes(fs, "boot/guest.pgim", pgim_stub(32));
    WasmBootConfig cfg = make_cfg(fs);
    fs.fail_read_on = 2;

    std::vector<uint8_t> out;
    std::string err;
    CHECK(!playgta6::boot::load_file_via_httpfs(cfg, "boot/guest.pgim", out, err));
    CHECK(err.find("httpfs_read") != std::string::npos);
    CHECK(fs.opens == 1);
    CHECK(fs.closes == 1);
    CHECK(fs.handles.empty());
}

void test_max_size_guard() {
    FakeFs fs;
    put_bytes(fs, "boot/guest.pgim", pgim_stub(64));
    WasmBootConfig cfg = make_cfg(fs);
    cfg.max_file_bytes = 8;

    std::vector<uint8_t> out;
    std::string err;
    CHECK(!playgta6::boot::load_file_via_httpfs(cfg, "boot/guest.pgim", out, err));
    CHECK(err.find("max_file_bytes") != std::string::npos);
    CHECK(fs.closes == 1);
    CHECK(fs.handles.empty());
    CHECK(out.size() <= 8);
}

void test_negative_read_guard() {
    FakeFs fs;
    put_bytes(fs, "boot/guest.pgim", pgim_stub());
    WasmBootConfig cfg = make_cfg(fs);
    fs.force_read = true;
    fs.force_read_rc = -3;

    std::vector<uint8_t> out;
    std::string err;
    CHECK(!playgta6::boot::load_file_via_httpfs(cfg, "boot/guest.pgim", out, err));
    CHECK(fs.closes == 1);
}

void test_manifest_pgim_extraction() {
    const char* json = R"({
      "schema": "playgta6-recomp-program",
      "schema_version": 1,
      "modules": [
        {"module": "vcruntime140", "file": "vcruntime140.dll"},
        {"module": "game"}
      ],
      "files": [
        {"path": "vcruntime140\\image\\guest.pgim", "size": 10},
        {"path": "game/image/guest.pgim", "size": 20},
        {"path": "game/image/guest.pgim.json", "size": 4},
        {"path": "game/image/guest.pgim", "size": 20}
      ]
    })";
    const std::vector<std::string> paths = playgta6::boot::extract_pgim_paths_from_manifest(json);
    CHECK(paths.size() == 2);
    CHECK(paths[0] == "vcruntime140/image/guest.pgim");
    CHECK(paths[1] == "game/image/guest.pgim");
}

void test_escaped_pgim_path() {
    const char* json = R"({"files":[{"path":"mod\\image\\guest.pgim"}]})";
    const std::vector<std::string> paths = playgta6::boot::extract_pgim_paths_from_manifest(json);
    CHECK(paths.size() == 1);
    CHECK(paths[0] == "mod/image/guest.pgim");
}

void test_milestone_log_format() {
    FakeFs fs;
    WasmBootConfig cfg = make_cfg(fs);
    playgta6::boot::emit_boot_milestone(cfg, "PGIM_LOAD");
    playgta6::boot::emit_boot_milestone(cfg, "PGIM_LOAD", "loaded boot/guest.pgim (16 bytes)");
    playgta6::boot::emit_boot_milestone(cfg, "TLS_INIT", "smoke");
    playgta6::boot::emit_boot_milestone(cfg, "ENTRY_CALL", "");
    CHECK(g_logs.size() == 4);
    CHECK(g_logs[0] == "[boot] PGIM_LOAD");
    CHECK(g_logs[1] == "[boot] PGIM_LOAD: loaded boot/guest.pgim (16 bytes)");
    CHECK(g_logs[2] == "[boot] TLS_INIT: smoke");
    CHECK(g_logs[3] == "[boot] ENTRY_CALL");
}

void test_fallback_pgim_load() {
    FakeFs fs;
    put_bytes(fs, "boot/guest.pgim", pgim_stub(8));
    WasmBootConfig cfg = make_cfg(fs);
    cfg.manifest_path = "boot/program-manifest.json";
    cfg.fallback_pgim_path = "boot/guest.pgim";

    const WasmBootResult r = playgta6::boot::run_wasm_guest_boot(cfg);
    CHECK(r.ok);
    CHECK(!r.linked_guest_available);
    CHECK(r.modules_loaded == 1);
    CHECK(r.total_bytes_loaded == 12);
    CHECK(r.stage == "PGIM_LOAD");
    CHECK(r.exit_code == 0);
    bool saw_pgim = false;
    for (const std::string& line : g_logs) {
        if (line.find("[boot] PGIM_LOAD") == 0) {
            saw_pgim = true;
        }
    }
    CHECK(saw_pgim);
}

void test_manifest_driven_load() {
    FakeFs fs;
    put_file(fs, "boot/program-manifest.json",
             R"({"files":[{"path":"mod/image/guest.pgim"}]})");
    put_bytes(fs, "mod/image/guest.pgim", pgim_stub(4));
    WasmBootConfig cfg = make_cfg(fs);

    const WasmBootResult r = playgta6::boot::run_wasm_guest_boot(cfg);
    CHECK(r.ok);
    CHECK(r.modules_loaded == 1);
    CHECK(r.total_bytes_loaded == 8);
    CHECK(fs.files.count("boot/guest.pgim") == 0);
}

void test_invalid_pgim_magic() {
    FakeFs fs;
    std::vector<uint8_t> junk = {'N', 'O', 'P', 'E', 1, 2, 3, 4};
    put_bytes(fs, "boot/guest.pgim", junk);
    WasmBootConfig cfg = make_cfg(fs);

    const WasmBootResult r = playgta6::boot::run_wasm_guest_boot(cfg);
    CHECK(!r.ok);
    CHECK(r.stage == "pgim_magic");
    CHECK(r.error.find("invalid PGIM magic") != std::string::npos);
    CHECK(r.modules_loaded == 0);
}

void test_missing_everything() {
    FakeFs fs;
    WasmBootConfig cfg = make_cfg(fs);
    const WasmBootResult r = playgta6::boot::run_wasm_guest_boot(cfg);
    CHECK(!r.ok);
    CHECK(r.stage == "httpfs");
}

void test_has_pgim_magic() {
    const std::vector<uint8_t> ok = pgim_stub();
    CHECK(playgta6::boot::has_pgim_magic(ok));
    const uint8_t short_buf[] = {'P', 'G', 'I'};
    CHECK(!playgta6::boot::has_pgim_magic(short_buf));
    const uint8_t bad[] = {'P', 'G', 'I', 'X'};
    CHECK(!playgta6::boot::has_pgim_magic(bad));
}

void test_resolve_module_pgim_path() {
    const std::vector<std::string> manifest_paths = {
        "boot/vcruntime140_1/image/guest.pgim",
        "vcruntime140/image/guest.pgim",
        "boot/msvcp140/image/guest.pgim",
        "BOOT/Main/image/guest.pgim",
    };

    CHECK(playgta6::boot::resolve_module_pgim_path("main", manifest_paths) ==
          "BOOT/Main/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("MSVCP140", manifest_paths) ==
          "boot/msvcp140/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("vcruntime140", manifest_paths) ==
          "vcruntime140/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("vcruntime140_1", manifest_paths) ==
          "boot/vcruntime140_1/image/guest.pgim");

    // Prefix collision guard: "vcruntime140" must not match "vcruntime140_1".
    const std::vector<std::string> only_vcr1 = {"boot/vcruntime140_1/image/guest.pgim"};
    CHECK(playgta6::boot::resolve_module_pgim_path("vcruntime140", only_vcr1) ==
          "boot/vcruntime140/image/guest.pgim");

    // Fallback when manifest has no paths or no match.
    const std::vector<std::string> empty_paths;
    CHECK(playgta6::boot::resolve_module_pgim_path("msvcp140", empty_paths) ==
          "boot/msvcp140/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("ucrtbase", manifest_paths) ==
          "boot/ucrtbase/image/guest.pgim");
    CHECK(playgta6::boot::resolve_module_pgim_path("", manifest_paths).empty());
}

constexpr uint32_t kTestImageSize = 0x4000;
constexpr uint32_t kTestTextRva = 0x2000;
constexpr uint64_t kTestDllBase = 0x1'4000'0000ULL;
constexpr uint64_t kTestExeBase = 0x1'4001'0000ULL;

void put_bytes_at(std::vector<uint8_t>& buf, size_t at, const void* src, size_t n) {
    if (buf.size() < at + n) {
        buf.resize(at + n, 0);
    }
    std::memcpy(buf.data() + at, src, n);
}

void put_u16(std::vector<uint8_t>& buf, size_t at, uint16_t v) { put_bytes_at(buf, at, &v, 2); }
void put_u32(std::vector<uint8_t>& buf, size_t at, uint32_t v) { put_bytes_at(buf, at, &v, 4); }
void put_u64(std::vector<uint8_t>& buf, size_t at, uint64_t v) { put_bytes_at(buf, at, &v, 8); }

void append_le(std::vector<uint8_t>& out, uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        out.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
}

std::vector<uint8_t> make_valid_pgim(uint64_t base, uint32_t entry_rva) {
    namespace pgim = playgta6::os::pgim;

    std::vector<uint8_t> pe_hdr(0x1000, 0);
    constexpr size_t kOpt = 0x80 + 24;
    put_u16(pe_hdr, 0, 0x5A4D);
    put_u32(pe_hdr, 0x3C, 0x80);
    put_u32(pe_hdr, 0x80, 0x00004550);
    put_u16(pe_hdr, 0x84, 0x8664);
    put_u16(pe_hdr, kOpt, 0x020B);
    put_u64(pe_hdr, kOpt + 24, base);
    put_u32(pe_hdr, kOpt + 56, kTestImageSize);
    put_u32(pe_hdr, kOpt + 108, 16);

    struct Sec {
        const char* name;
        uint32_t rva;
        uint32_t prot;
        uint32_t flags;
        std::vector<uint8_t> bytes;
    };
    std::vector<Sec> secs = {
        {"<headers>", 0, pgim::kProtRead, pgim::kSectionFlagHeaders, std::move(pe_hdr)},
        {".text", kTestTextRva, pgim::kProtRead | pgim::kProtExecute, 0, {}},
    };

    const uint64_t tables_end = 64 + 64 * secs.size();
    std::vector<uint64_t> offsets;
    uint64_t cursor = tables_end;
    for (const Sec& s : secs) {
        if (s.bytes.empty()) {
            offsets.push_back(0);
            continue;
        }
        const uint64_t off = (cursor + 4095) / 4096 * 4096;
        offsets.push_back(off);
        cursor = off + s.bytes.size();
    }

    std::vector<uint8_t> out = {'P', 'G', 'I', 'M'};
    append_le(out, 1, 4);
    append_le(out, 64, 4);
    append_le(out, 64, 4);
    append_le(out, pgim::kFlagHeadersMapped, 4);
    append_le(out, 4096, 4);
    append_le(out, base, 8);
    append_le(out, kTestImageSize, 4);
    append_le(out, 0x1000, 4);
    append_le(out, entry_rva, 4);
    append_le(out, secs.size(), 4);
    append_le(out, 0, 4);
    append_le(out, 0, 4);
    append_le(out, cursor, 8);

    for (size_t i = 0; i < secs.size(); ++i) {
        std::string name = secs[i].name;
        name.resize(16, '\0');
        out.insert(out.end(), name.begin(), name.end());
        append_le(out, secs[i].rva, 4);
        append_le(out, 0x1000, 4);
        append_le(out, secs[i].prot, 4);
        append_le(out, secs[i].flags, 4);
        append_le(out, offsets[i], 8);
        append_le(out, secs[i].bytes.size(), 8);
        append_le(out, pgim::fnv1a_64(secs[i].bytes.data(), secs[i].bytes.size()), 8);
        append_le(out, 0, 4);
        append_le(out, 0, 4);
    }
    for (size_t i = 0; i < secs.size(); ++i) {
        if (!secs[i].bytes.empty()) {
            out.resize(static_cast<size_t>(offsets[i]), 0);
            out.insert(out.end(), secs[i].bytes.begin(), secs[i].bytes.end());
        }
    }
    return out;
}

bool g_test_dllmain_ran = false;
bool g_test_entry_ran = false;

void fake_dll_main(playgta6::GuestState* s) {
    g_test_dllmain_ran = true;
    s->gpr[playgta6::GuestReg::Rax] = 1;  // TRUE
    s->gpr[playgta6::GuestReg::Rsp] += 8;
}

void fake_exe_entry(playgta6::GuestState* s) {
    g_test_entry_ran = true;
    s->gpr[playgta6::GuestReg::Rax] = 42;
    s->gpr[playgta6::GuestReg::Rsp] += 8;
}

const Pg6LiftedFunctionEntry kTestDllTable[] = {{kTestDllBase + kTestTextRva, &fake_dll_main}};
const uint64_t kTestDllTableCount = 1;
const Pg6LiftedFunctionEntry kTestExeTable[] = {{kTestExeBase + kTestTextRva, &fake_exe_entry}};
const uint64_t kTestExeTableCount = 1;

void test_program_boot_invalid_descriptor() {
    FakeFs fs;
    WasmBootConfig cfg = make_cfg(fs);

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, 0, kPg6ProgramModuleDll, 0, nullptr,
         nullptr, nullptr, nullptr},
        {"game.exe", "main", kTestExeBase, kTestImageSize, kTestExeBase + kTestTextRva,
         kPg6ProgramModuleMain, 0, nullptr, nullptr, nullptr, nullptr},
    };

    // Wrong version.
    const Pg6Program bad_ver{99u, 2, mods, nullptr, nullptr};
    WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, bad_ver);
    CHECK(!r.ok);
    CHECK(r.linked_guest_available);
    CHECK(r.stage == "program_prepare");
    CHECK(!r.error.empty());
    CHECK(r.modules_loaded == 0);
    CHECK(fs.opens == 0);

    // No modules.
    const Pg6Program empty_prog{kPg6ProgramVersion, 0, nullptr, nullptr, nullptr};
    r = playgta6::boot::run_wasm_program_boot(cfg, empty_prog);
    CHECK(!r.ok);
    CHECK(r.stage == "program_prepare");
    CHECK(r.error.find("no modules") != std::string::npos);

    // Main module not last.
    const Pg6ProgramModule wrong_order[] = {mods[1], mods[0]};
    const Pg6Program bad_order{kPg6ProgramVersion, 2, wrong_order, nullptr, nullptr};
    r = playgta6::boot::run_wasm_program_boot(cfg, bad_order);
    CHECK(!r.ok);
    CHECK(r.stage == "program_prepare");

    // Empty build module name.
    Pg6ProgramModule empty_name[] = {mods[0], mods[1]};
    empty_name[0].module = "";
    const Pg6Program bad_name{kPg6ProgramVersion, 2, empty_name, nullptr, nullptr};
    r = playgta6::boot::run_wasm_program_boot(cfg, bad_name);
    CHECK(!r.ok);
    CHECK(r.stage == "program_prepare");
}

void test_program_boot_missing_module_pgim() {
    FakeFs fs;
    put_file(fs, "boot/program-manifest.json",
             R"({"files":[{"path":"helper/image/guest.pgim"},{"path":"main/image/guest.pgim"}]})");
    put_bytes(fs, "helper/image/guest.pgim", pgim_stub(8));
    WasmBootConfig cfg = make_cfg(fs);

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, 0, kPg6ProgramModuleDll, 0, nullptr,
         nullptr, nullptr, nullptr},
        {"game.exe", "main", kTestExeBase, kTestImageSize, kTestExeBase + kTestTextRva,
         kPg6ProgramModuleMain, 0, nullptr, nullptr, nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 2, mods, nullptr, nullptr};

    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    CHECK(!r.ok);
    CHECK(r.linked_guest_available);
    CHECK(r.stage == "httpfs");
    CHECK(r.modules_loaded == 1);
    CHECK(r.total_bytes_loaded == 12);
    CHECK(r.error.find("httpfs_open failed") != std::string::npos);
}

void test_program_boot_bad_magic() {
    FakeFs fs;
    put_bytes(fs, "boot/helper/image/guest.pgim", pgim_stub(8));
    put_bytes(fs, "boot/main/image/guest.pgim", std::vector<uint8_t>{'B', 'A', 'D', '!', 1, 2});
    WasmBootConfig cfg = make_cfg(fs);

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, 0, kPg6ProgramModuleDll, 0, nullptr,
         nullptr, nullptr, nullptr},
        {"game.exe", "main", kTestExeBase, kTestImageSize, kTestExeBase + kTestTextRva,
         kPg6ProgramModuleMain, 0, nullptr, nullptr, nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 2, mods, nullptr, nullptr};

    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    CHECK(!r.ok);
    CHECK(r.stage == "pgim_magic");
    CHECK(r.modules_loaded == 1);
    CHECK(r.error.find("invalid PGIM magic") != std::string::npos);
}

void test_program_boot_corrupt_pgim_error_propagation() {
    FakeFs fs;
    // One module without leading "boot/", one with leading "boot/", and no manifest:
    // exercises fallback path resolution and prefix-stripping retry seamlessly.
    put_bytes(fs, "helper/image/guest.pgim", pgim_stub(12));
    put_bytes(fs, "boot/main/image/guest.pgim", pgim_stub(16));
    WasmBootConfig cfg = make_cfg(fs);

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, 0, kPg6ProgramModuleDll, 0, nullptr,
         nullptr, nullptr, nullptr},
        {"game.exe", "main", kTestExeBase, kTestImageSize, kTestExeBase + kTestTextRva,
         kPg6ProgramModuleMain, 0, nullptr, nullptr, nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 2, mods, nullptr, nullptr};

    playgta6::os::VirtualMemoryArena arena;
    playgta6::os::win32::set_vmem_arena(&arena);
    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    playgta6::guest::reset_boot();
    playgta6::os::win32::set_vmem_arena(nullptr);

    CHECK(!r.ok);
    CHECK(r.linked_guest_available);
    CHECK(r.modules_loaded == 2);
    CHECK(r.total_bytes_loaded == 16 + 20);
    CHECK(r.stage == "load_failed");
    CHECK(!r.error.empty());
}

void test_program_boot_full_synthetic_execution() {
    FakeFs fs;
    const std::vector<uint8_t> dll_pgim = make_valid_pgim(kTestDllBase, kTestTextRva);
    const std::vector<uint8_t> exe_pgim = make_valid_pgim(kTestExeBase, kTestTextRva);
    put_file(fs, "boot/program-manifest.json",
             R"({"files":[{"path":"helper/image/guest.pgim"},{"path":"main/image/guest.pgim"}]})");
    // helper is stored under "boot/..." while manifest omits "boot/"; main matches manifest as-is.
    put_bytes(fs, "boot/helper/image/guest.pgim", dll_pgim);
    put_bytes(fs, "main/image/guest.pgim", exe_pgim);

    WasmBootConfig cfg = make_cfg(fs);
    cfg.read_chunk_bytes = 4096;

    const Pg6ProgramModule mods[] = {
        {"helper.dll", "helper", kTestDllBase, kTestImageSize, kTestDllBase + kTestTextRva,
         kPg6ProgramModuleDll, 0, kTestDllTable, &kTestDllTableCount, nullptr, nullptr},
        {"game_scarlett_master.exe", "main", kTestExeBase, kTestImageSize,
         kTestExeBase + kTestTextRva, kPg6ProgramModuleMain, 0, kTestExeTable, &kTestExeTableCount,
         nullptr, nullptr},
    };
    const Pg6Program prog{kPg6ProgramVersion, 2, mods, nullptr, nullptr};

    g_test_dllmain_ran = false;
    g_test_entry_ran = false;

    playgta6::os::VirtualMemoryArena arena;
    playgta6::os::win32::set_vmem_arena(&arena);
    const WasmBootResult r = playgta6::boot::run_wasm_program_boot(cfg, prog);
    const bool dll_registered = playgta6::os::win32::GetModuleHandleA("helper.dll") == kTestDllBase;
    playgta6::guest::reset_boot();
    playgta6::os::win32::set_vmem_arena(nullptr);

    CHECK(r.ok);
    CHECK(r.linked_guest_available);
    CHECK(r.stage == "ENTRY_CALL");
    CHECK(r.exit_code == 42);
    CHECK(r.error.empty());
    CHECK(r.modules_loaded == 2);
    CHECK(r.total_bytes_loaded == dll_pgim.size() + exe_pgim.size());
    CHECK(g_test_dllmain_ran);
    CHECK(g_test_entry_ran);
    CHECK(dll_registered);

    int pgim_milestones = 0;
    for (const std::string& line : g_logs) {
        if (line.find("[boot] PGIM_LOAD: ") == 0) {
            ++pgim_milestones;
        }
    }
    CHECK(pgim_milestones == 2);
}

}  // namespace

int main() {
    test_chunked_httpfs_reads();
    test_close_on_read_error();
    test_max_size_guard();
    test_negative_read_guard();
    test_manifest_pgim_extraction();
    test_escaped_pgim_path();
    test_milestone_log_format();
    test_fallback_pgim_load();
    test_manifest_driven_load();
    test_invalid_pgim_magic();
    test_missing_everything();
    test_has_pgim_magic();
    test_resolve_module_pgim_path();
    test_program_boot_invalid_descriptor();
    test_program_boot_missing_module_pgim();
    test_program_boot_bad_magic();
    test_program_boot_corrupt_pgim_error_propagation();
    test_program_boot_full_synthetic_execution();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d test(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_wasm_boot: ok\n");
    return 0;
}
