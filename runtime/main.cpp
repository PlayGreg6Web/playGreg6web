#include <array>
#include <cstdio>
#include <string>
#include <vector>

#if !defined(__EMSCRIPTEN__)
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif
#endif

#include "abi/playgta6_abi.h"
#include "gpu/cmd_encoder.h"
#include "os/os_shim.h"
#include "os/win32_compat.h"
#include "wasm_boot.h"

namespace {
constexpr const char kTitleArgsFlag[] = "--title-args=";

// argv of the restarted process: the same options, with the title-arguments option replaced by the
// title's new arguments (empty new arguments drop it).
[[maybe_unused]] std::vector<std::string> relaunch_argv(int argc, char** argv, const std::string& new_args) {
    std::vector<std::string> out;
    for (int i = 0; i < argc; ++i) {
        if (i > 0 && std::string(argv[i]).rfind(kTitleArgsFlag, 0) == 0) {
            continue;
        }
        out.emplace_back(argv[i]);
    }
    if (!new_args.empty()) {
        out.push_back(kTitleArgsFlag + new_args);
    }
    return out;
}

#if !defined(__EMSCRIPTEN__)
// XLaunchNewGame: replaces this process with a fresh run; returns only if the re-exec fails.
[[maybe_unused]] int reexec_self(int argc, char** argv, const std::string& new_args) {
    const std::vector<std::string> args = relaunch_argv(argc, argv, new_args);
    std::vector<const char*> raw;
    for (const std::string& a : args) {
        raw.push_back(a.c_str());
    }
    raw.push_back(nullptr);
    playgta6::os::log("playgta6web relaunch requested by the title (XLaunchNewGame): re-exec");
#if defined(_WIN32)
    _execv(argv[0], raw.data());
#else
    execv(argv[0], const_cast<char* const*>(raw.data()));
#endif
    playgta6::os::log("playgta6web relaunch: re-exec failed");
    return 1;
}
#endif
}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    char banner[96]{};
    std::snprintf(
        banner,
        sizeof(banner),
        "playgta6web Wasm64 runtime initialized (ABI v%u)",
        playgta6::kAbiVersion);
    playgta6::os::log(banner);

#if defined(PG6_BOOT_LINKED_GUEST)
    playgta6::boot::WasmBootConfig boot_config;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]).rfind(kTitleArgsFlag, 0) == 0) {
            boot_config.title_args = argv[i] + sizeof(kTitleArgsFlag) - 1;
        }
    }
    const playgta6::boot::WasmBootResult result = playgta6::boot::run_wasm_guest_boot(boot_config);
#if !defined(__EMSCRIPTEN__)
    if (result.relaunch_requested) {
        return reexec_self(argc, argv, result.relaunch_args);
    }
#endif
    if (!result.ok) {
        std::string msg = "playgta6web linked guest boot failed";
        if (!result.stage.empty()) {
            msg += " [";
            msg += result.stage;
            msg += "]";
        }
        if (!result.error.empty()) {
            msg += ": ";
            msg += result.error;
        }
        playgta6::os::log(msg);
        return result.exit_code != 0 ? static_cast<int>(result.exit_code) : 1;
    }
    return static_cast<int>(result.exit_code);
#else
    // Optional HTTPFS smoke probe when an HTTPFS mailbox or host bridge is active.
    const int32_t manifest_handle = playgta6::os::httpfs_open("data-manifest.json");
    if (manifest_handle >= 0) {
        std::array<char, 64> probe{};
        const int64_t bytes_read =
            playgta6::os::httpfs_read(manifest_handle, 0, probe.data(), probe.size());
        playgta6::os::httpfs_close(manifest_handle);

        char msg[128]{};
        std::snprintf(
            msg,
            sizeof(msg),
            "HTTPFS smoke probe ok: data-manifest.json (%lld bytes read)",
            static_cast<long long>(bytes_read));
        playgta6::os::log(msg);
    } else if (playgta6::os::httpfs_has_attached_mailbox()) {
        playgta6::os::log("HTTPFS mailbox attached (data-manifest.json not present)");
    }

    playgta6::gpu::CommandEncoder enc;
    enc.begin_frame();
    enc.clear_color(0.08f, 0.12f, 0.22f, 1.0f);
    enc.present();
    playgta6::os::wgpu_submit(enc.data(), enc.size());

    return 0;
#endif
}
