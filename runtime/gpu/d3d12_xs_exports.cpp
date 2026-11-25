#include "gpu/d3d12_xs_exports.h"

#include <cstring>
#include <mutex>
#include <string>

#include "guest/guest_runtime.h"
#include "os/os_shim.h"
#include "os/win32_loader.h"

namespace playgta6::gpu::d3d12_xs_exports {

namespace dg = playgta6::gpu::d3d12_guest;

namespace {

struct State {
    std::mutex mutex;
    dg::DeviceOptions options;
    uint64_t device_identity = 0;  // registered interface pointer of the live device (0 = none)
    VirtualMemoryConfig vm;
};

State& state() {
    static State s;
    return s;
}

// Pins the live device, if any.
com::Ref<dg::Device> live_device_locked(State& s) {
    if (s.device_identity == 0) {
        return {};
    }
    com::detail::Resolved r;
    if (!com::detail::lookup_interface(s.device_identity, &r) || r.iface != &dg::kID3D12Device.info ||
        r.object->ref_count() == 0) {
        s.device_identity = 0;
        return {};
    }
    return com::Ref<dg::Device>::share(static_cast<dg::Device*>(static_cast<dg::IDeviceHost*>(r.host)));
}

void blob_error(uint64_t* error_blob, const std::string& message) {
    os::log("d3d12_xs: D3D12SerializeVersionedRootSignature: " + message);
    if (error_blob == nullptr) {
        return;
    }
    auto blob = dg::Blob::create(message.c_str(), message.size() + 1);
    *error_blob = blob ? blob.detach(dg::kID3DBlob) : 0;
}

}  // namespace

void set_device_options(dg::DeviceOptions options) {
    std::lock_guard<std::mutex> lock(state().mutex);
    state().options = std::move(options);
}

dg::DeviceOptions device_options() {
    std::lock_guard<std::mutex> lock(state().mutex);
    return state().options;
}

com::Ref<dg::Device> current_device() {
    std::lock_guard<std::mutex> lock(state().mutex);
    return live_device_locked(state());
}

VirtualMemoryConfig virtual_memory_config() {
    std::lock_guard<std::mutex> lock(state().mutex);
    return state().vm;
}

void reset_for_tests() {
    std::lock_guard<std::mutex> lock(state().mutex);
    state().options = dg::DeviceOptions{};
    state().device_identity = 0;
    state().vm = VirtualMemoryConfig{};
}

namespace {

struct ExportSpec {
    const char* name;
    playgta6::guest::HostThunkFn thunk;
    Pg6ProcAddress proc;
};

const ExportSpec kExports[] = {
    {"D3D12XboxCreateDevice", com::function_thunk<&pg6_d3d12_d3d12_xbox_create_device>(),
     os::win32::proc_address(&pg6_d3d12_d3d12_xbox_create_device)},
    {"D3D12SerializeVersionedRootSignature", com::function_thunk<&pg6_d3d12_d3d12_serialize_versioned_root_signature>(),
     os::win32::proc_address(&pg6_d3d12_d3d12_serialize_versioned_root_signature)},
    {"D3D12ConfigureVirtualMemoryX", com::function_thunk<&pg6_d3d12_d3d12_configure_virtual_memory_x>(),
     os::win32::proc_address(&pg6_d3d12_d3d12_configure_virtual_memory_x)},
};

}  // namespace

size_t register_d3d12_xs_exports() {
    size_t n = 0;
    for (const ExportSpec& e : kExports) {
        // Re-registers after a reset_guest_runtime() (tests); otherwise finds the existing address.
        uint64_t callable = playgta6::guest::guest_callable_for_proc(e.proc);
        if (callable == 0) {
            callable = playgta6::guest::register_host_callable(e.thunk, e.name, e.proc);
        }
        if (callable != 0) {
            ++n;
        }
    }
    return n;
}

uint64_t export_address(const char* name) {
    for (const ExportSpec& e : kExports) {
        if (std::strcmp(e.name, name) == 0) {
            return playgta6::guest::guest_callable_for_proc(e.proc);
        }
    }
    return 0;
}

}  // namespace playgta6::gpu::d3d12_xs_exports

using namespace playgta6::gpu;
namespace dg = playgta6::gpu::d3d12_guest;
namespace dx = playgta6::gpu::d3d12_xs_exports;
namespace gcom = playgta6::guest::com;

extern "C" int32_t pg6_d3d12_d3d12_xbox_create_device(void* adapter, const void* parameters, const void* riid,
                                                      uint64_t* device) {
    if (riid == nullptr) {
        return dg::kEPointer;
    }
    gcom::Guid iid;
    std::memcpy(&iid, riid, sizeof(iid));
    if (adapter != nullptr) {
        playgta6::os::log("d3d12_xs: D3D12XboxCreateDevice: non-null adapter ignored");
    }
    uint32_t version = 0;
    if (parameters != nullptr) {
        std::memcpy(&version, parameters, sizeof(version));
    }
    auto& s = dx::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    gcom::Ref<dg::Device> dev = dx::live_device_locked(s);
    if (!dev) {
        dev = gcom::create_object<dg::Device>(gcom::implements(dg::kID3D12Device), s.options);
        if (!dev) {
            if (device != nullptr) {
                *device = 0;
            }
            return dg::kEOutOfMemory;
        }
        dev->initialize_guest_payload();
        s.device_identity = dev->identity();
        playgta6::os::log("d3d12_xs: D3D12XboxCreateDevice: device created (parameter version " +
                          std::to_string(version) + ", riid " + gcom::format_guid(iid) + ")");
    }
    if (device == nullptr) {
        return dg::kSFalse;
    }
    uint64_t pointer = 0;
    const gcom::HResult hr = dev->query_interface(iid, &pointer);
    *device = pointer;
    if (hr < 0) {
        playgta6::os::log("d3d12_xs: D3D12XboxCreateDevice: device does not answer " + gcom::format_guid(iid));
    }
    return hr;
}

extern "C" int32_t pg6_d3d12_d3d12_serialize_versioned_root_signature(const void* desc, uint64_t* blob,
                                                                      uint64_t* error_blob) {
    if (blob != nullptr) {
        *blob = 0;
    }
    if (error_blob != nullptr) {
        *error_blob = 0;
    }
    if (desc == nullptr || blob == nullptr) {
        dx::blob_error(error_blob, desc == nullptr ? "null root signature description" : "null blob pointer");
        return dg::kEInvalidArg;
    }
    dg::GuestVersionedRootSignatureDesc guest{};
    std::memcpy(&guest, desc, sizeof(guest));
    playgta6::gpu::d3d12_state::RootSignatureDesc parsed;
    std::string error;
    if (!dg::read_versioned_root_signature(guest, &parsed, &error)) {
        dx::blob_error(error_blob, error);
        return dg::kEInvalidArg;
    }
    playgta6::gpu::d3d12_state::RootSignature check;
    const auto err = playgta6::gpu::d3d12_state::RootSignature::build(parsed, &check);
    if (!err.ok()) {
        std::string message = std::string("invalid root signature: ") + playgta6::gpu::d3d12_state::status_name(err.status);
        if (err.parameter_index != playgta6::gpu::d3d12_state::kNoIndex) {
            message += " (parameter " + std::to_string(err.parameter_index) + ")";
        }
        dx::blob_error(error_blob, message);
        return dg::kEInvalidArg;
    }
    const std::vector<uint8_t> bytes = playgta6::gpu::d3d12_state::serialize_root_signature(parsed);
    auto out = dg::Blob::create(bytes.data(), bytes.size());
    if (!out) {
        return dg::kEOutOfMemory;
    }
    *blob = out.detach(dg::kID3DBlob);
    return dg::kSOk;
}

extern "C" int32_t pg6_d3d12_d3d12_configure_virtual_memory_x(const void* configuration) {
    auto& s = dx::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    ++s.vm.calls;
    if (configuration != nullptr) {
        std::memcpy(s.vm.bytes.data(), configuration, s.vm.bytes.size());
    } else {
        s.vm.bytes.fill(0);
    }
    return dg::kSOk;
}
