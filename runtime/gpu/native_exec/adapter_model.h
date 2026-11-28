// Native executor: the adapter a device-loss recovery may come back on. A D3D12 device is one per
// adapter and process; while anything still holds the lost one, a new instance cannot open the adapter
// again (Dawn logs "D3D12CreateDevice failed with DXGI_ERROR_DEVICE_HUNG" while it initializes it) and
// RequestAdapter hands out the next one, on a one-GPU machine the CPU adapter (WARP). Every std-pack
// --live-executor boot that recovered from a TDR did that (device 5 ms instead of ~270 ms, 515 MB of GPU
// memory in use afterwards, d3d10warp.dll on a later crash stack): the frames after it ran in software,
// the GPU stall watch saw nothing complete for 8-20 s (no TDR on a CPU), the executor sat in one batch
// for 26-45 s and the title's 30 s main-loop watchdog fired (host_access_violation@0x1423cf620).
// Holders found: queued shader-worker jobs (each captures the device) and the min-size twin / attachment
// proxy caches, now released by the recovery. A recovery also accepts only the adapter the run started
// on; any other is a failed attempt, retried after a backoff on a new instance. Pure logic, no GPU.
#pragma once

#include <cstdint>
#include <string>

namespace playgta6::gpu::native_exec {

struct AdapterIdentity {
    std::string backend;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    bool cpu = false;
    std::string description;  // for the log only
};

inline bool same_adapter(const AdapterIdentity& first, const AdapterIdentity& now) {
    return now.backend == first.backend && now.cpu == first.cpu && now.vendor_id == first.vendor_id &&
           now.device_id == first.device_id;
}

}  // namespace playgta6::gpu::native_exec
