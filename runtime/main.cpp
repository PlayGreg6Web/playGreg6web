#include <array>
#include <cstdio>
#include <string>

#include "abi/playgta6_abi.h"
#include "gpu/cmd_encoder.h"
#include "os/os_shim.h"
#include "wasm_boot.h"

int main() {
    char banner[96]{};
    std::snprintf(
        banner,
        sizeof(banner),
        "playgta6web Wasm64 runtime initialized (ABI v%u)",
        playgta6::kAbiVersion);
    playgta6::os::log(banner);

#if defined(PG6_BOOT_LINKED_GUEST)
    const playgta6::boot::WasmBootResult result =
        playgta6::boot::run_wasm_guest_boot(playgta6::boot::WasmBootConfig{});
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
