#pragma once

// d3d12_xs.dll flat exports (G24).
//
// The executable imports exactly three functions from d3d12_xs.dll; everything else it does with
// D3D12 is method calls on the objects D3D12XboxCreateDevice hands out (d3d12_guest_objects.h).
//
//   D3D12XboxCreateDevice(adapter, params, riid, ppDevice)   pg6_d3d12_d3d12_xbox_create_device
//   D3D12SerializeVersionedRootSignature(desc, ppBlob, ppErr) pg6_d3d12_d3d12_serialize_versioned_root_signature
//   D3D12ConfigureVirtualMemoryX(config)                      pg6_d3d12_d3d12_configure_virtual_memory_x
//
// The C-ABI shims are declared in gpu/d3d12_xs.h (the header the R5 import bridge scans) and take
// host-translated pointers like every R5-bound shim (pg6_seam_host_ptr); guest pointers *inside*
// guest structures are translated through the COM bridge seam. Interface pointers written to the
// out parameters are guest addresses.
//
// Evidence (static analysis of the executable, local only): each import has one call site
// (Serialize: two). D3D12XboxCreateDevice is called with adapter = null, a 0x50-byte parameter block
// and riid = IID_ID3D12Device11, HRESULT checked. D3D12SerializeVersionedRootSignature is called with
// a stack D3D12_VERSIONED_ROOT_SIGNATURE_DESC (Version at +0, NumParameters +8, pParameters +16,
// NumStaticSamplers +24, pStaticSamplers +32, Flags +40) and two zeroed out pointers.
// D3D12ConfigureVirtualMemoryX takes one pointer to a 32-byte block (zeroed, one u32 set at +4)
// before the device is created; failure is fatal for the caller.
//
// D3D12ConfigureVirtualMemoryX: on the console it reserves the GPU virtual address layout the
// title uses for its own placed resources. Here guest memory is the VirtualMemoryArena (one
// contiguous guest address space) and GPU virtual addresses of buffers are guest addresses or
// device-allocated address ranges, so there is nothing to configure: the block
// is recorded (virtual_memory_config()) and S_OK returned.
//
// Activation: register_d3d12_xs_exports() (called by os::win32::register_all_shim_exports())
// registers guest-callable thunks for the three exports so the guest GetProcAddress hands out
// dispatchable addresses; R5-generated import thunks bind the C-ABI shims by name.

#include <array>
#include <cstdint>

#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_xs.h"

namespace playgta6::gpu::d3d12_xs_exports {

namespace com = playgta6::guest::com;

// Options for the device the next D3D12XboxCreateDevice creates (tests: packet sink).
void set_device_options(d3d12_guest::DeviceOptions options);
d3d12_guest::DeviceOptions device_options();

// The live device (one per process, like D3D12 per adapter), or an empty Ref.
com::Ref<d3d12_guest::Device> current_device();

// Registers the guest-callable thunks (idempotent). Returns the number of exports registered.
size_t register_d3d12_xs_exports();
// Guest-callable synthetic address of an export by name (0 if not registered).
uint64_t export_address(const char* name);

struct VirtualMemoryConfig {
    uint32_t calls = 0;
    std::array<uint8_t, 32> bytes{};  // the last block passed
};
VirtualMemoryConfig virtual_memory_config();

// Tests: forgets the device singleton, options and recorded configuration (objects are dropped by
// com::reset_com_bridge()).
void reset_for_tests();

}  // namespace playgta6::gpu::d3d12_xs_exports
