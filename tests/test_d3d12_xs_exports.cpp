// d3d12_xs.dll exports and guest D3D12 objects (G24), driven the way recompiled code drives them:
// a hand-written "lifted" caller sets up Win64 arguments, loads `[this]` and calls a vtable slot
// (or a flat export) through __playgta6_indirect_call. Synthetic data only.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_guest_resources.h"
#include "gpu/d3d12_xs_exports.h"
#include "guest/com_bridge.h"
#include "guest/guest_runtime.h"
#include "os/shim_exports.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/xbox_scarlett.h"
#include "os/xmem_windows.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_d3d12_xs_exports.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace dg = playgta6::gpu::d3d12_guest;
namespace dx = playgta6::gpu::d3d12_xs_exports;
namespace d3d = playgta6::gpu::d3d12_state;
using com::Guid;
using playgta6::os::VirtualMemoryArena;

VirtualMemoryArena* g_arena = nullptr;

uint64_t load64(uint64_t va) {
    uint64_t v = 0;
    CHECK(g_arena->read(va, &v, sizeof(v)));
    return v;
}
uint32_t load32(uint64_t va) {
    uint32_t v = 0;
    CHECK(g_arena->read(va, &v, sizeof(v)));
    return v;
}
void store64(uint64_t va, uint64_t v) {
    CHECK(g_arena->write(va, &v, sizeof(v)));
}

template <typename T>
uint64_t put(const T& value) {
    const uint64_t va = com::guest_alloc(sizeof(T));
    CHECK(va != 0 && com::write_guest(va, value));
    return va;
}
uint64_t put_bytes(const void* data, size_t size) {
    const uint64_t va = com::guest_alloc(size);
    CHECK(va != 0 && com::write_guest(va, data, size));
    return va;
}
template <typename T>
T get(uint64_t va) {
    T v{};
    CHECK(com::read_guest(va, v));
    return v;
}

// ---------------------------------------------------------------------------
// Guest-convention caller (as lifted code does it)
// ---------------------------------------------------------------------------

constexpr uint64_t kVCall = 0x140020000ULL;

struct Arg {
    uint64_t bits;
    bool is_float;
};
Arg I(uint64_t v) {
    return Arg{v, false};
}
Arg F32(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return Arg{b, true};
}

struct VCallSpec {
    uint64_t target = 0;  // 0: virtual call through [args[0]] + 8 * slot
    uint32_t slot = 0;
    std::vector<Arg> args;
};
thread_local VCallSpec t_spec;
thread_local bool t_balanced = true;

void push(GuestState* s, uint64_t v) {
    s->gpr[GuestReg::Rsp] -= 8;
    store64(s->gpr[GuestReg::Rsp], v);
}

void sub_vcall(GuestState* s) {
    static constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    const size_t n = t_spec.args.size();
    const size_t stack_n = n > 4 ? n - 4 : 0;
    const uint64_t frame = ((0x20 + 8 * stack_n + 15) & ~uint64_t{15}) + 8;
    s->gpr[GuestReg::Rsp] -= frame;
    const uint64_t rsp = s->gpr[GuestReg::Rsp];
    for (size_t i = 0; i < n; ++i) {
        const Arg& a = t_spec.args[i];
        if (i < 4) {
            if (a.is_float) {
                s->xmm[i][0] = a.bits;
                s->gpr[kRegs[i]] = 0xBAD0BAD0BAD0BAD0ULL;
            } else {
                s->gpr[kRegs[i]] = a.bits;
                s->xmm[i][0] = 0xBAD1BAD1BAD1BAD1ULL;
            }
        } else {
            store64(rsp + 0x20 + 8 * (i - 4), a.bits);
        }
    }
    uint64_t target = t_spec.target;
    if (target == 0) {
        // `mov rax, [this]; call [rax + 8*slot]` with `this` in the register the caller chose.
        uint64_t self = s->gpr[GuestReg::Rcx];
        for (size_t i = 0; i < n && i < 4; ++i) {
            if (t_spec.args[i].bits == t_spec.args[0].bits) {
                self = t_spec.args[0].bits;
            }
        }
        target = load64(load64(self) + 8ull * t_spec.slot);
    }
    push(s, kVCall + 0x40);
    __playgta6_indirect_call(s, target);
    if (s->gpr[GuestReg::Rsp] != rsp) {
        t_balanced = false;
    }
    s->gpr[GuestReg::Rsp] += frame;
    s->gpr[GuestReg::Rsp] += 8;  // ret
}

// A title's XMem allocation hook (shape of the one the executable exports): allocator id 0x80 with
// page-size field 2 is carved, 64 KiB-rounded, out of an arena of kRtBudget bytes allocated once
// through the default allocator; everything else goes to the default allocator.
constexpr uint64_t kRtAllocHook = 0x140020400ULL;
constexpr uint64_t kRtFreeHook = 0x140020500ULL;
constexpr uint64_t kRtBudget = 0x5200000;
uint64_t g_rt_base = 0;
uint64_t g_rt_cursor = 0;
int g_rt_frees = 0;

uint64_t xmem_guest(void* host) {
    return host == nullptr ? 0 : playgta6::os::win32::get_vmem_arena()->guest_address(host);
}

void sub_rt_alloc_hook(GuestState* s) {
    const uint64_t size = s->gpr[GuestReg::Rcx];
    const uint64_t attributes = s->gpr[GuestReg::Rdx];
    uint64_t result = 0;
    if (((attributes >> 16) & 0xFF) == 0x80 && (attributes & 0xC000) == 0x8000) {
        if (g_rt_base == 0) {
            g_rt_base = g_rt_cursor = xmem_guest(pg6_xbox_XMemAllocDefault(kRtBudget, attributes));
        }
        const uint64_t next = g_rt_cursor + ((size + 0xFFFF) & ~uint64_t{0xFFFF});
        if (next <= g_rt_base + kRtBudget) {
            result = g_rt_cursor;
            g_rt_cursor = next;
        }
    } else {
        result = xmem_guest(pg6_xbox_XMemAllocDefault(static_cast<size_t>(size), attributes));
    }
    s->gpr[GuestReg::Rax] = result;
    s->gpr[GuestReg::Rsp] += 8;
}

void sub_rt_free_hook(GuestState* s) {
    const uint64_t va = s->gpr[GuestReg::Rcx];
    if (va >= g_rt_base && va < g_rt_base + kRtBudget) {
        ++g_rt_frees;
    } else {
        pg6_xbox_XMemFreeDefault(playgta6::os::win32::get_vmem_arena()->translate(va), s->gpr[GuestReg::Rdx]);
    }
    s->gpr[GuestReg::Rsp] += 8;
}

const LiftedFunctionEntry kFunctions[] = {{kVCall, sub_vcall}, {kRtAllocHook, sub_rt_alloc_hook},
                                          {kRtFreeHook, sub_rt_free_hook}};

struct GuestFault {
    FaultInfo info;
};
void throwing_handler(const FaultInfo* info) {
    throw GuestFault{*info};
}

struct Fixture {
    VirtualMemoryArena arena;
    GuestThread thread{};
    std::vector<std::vector<uint8_t>>* packets = nullptr;
    std::vector<std::vector<uint8_t>> captured;

    Fixture() {
        reset_guest_runtime();
        com::reset_com_bridge();
        dx::reset_for_tests();
        dg::reset_sret_stats();
        com::set_guest_memory(&arena);
        set_fault_handler(throwing_handler);
        g_arena = &arena;
        CHECK(register_lifted_functions(kFunctions));
        CHECK(create_guest_thread(arena, 512 * 1024, thread));
        CHECK(dx::register_d3d12_xs_exports() == 3);
        dg::DeviceOptions options;
        options.sink = [this](const uint8_t* data, size_t size) { captured.emplace_back(data, data + size); };
        dx::set_device_options(options);
        t_balanced = true;
    }
    ~Fixture() {
        com::reset_com_bridge();
        dx::reset_for_tests();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        reset_guest_runtime();
        g_arena = nullptr;
    }
    GuestState& state() { return thread.state; }

    uint64_t vcall(uint64_t obj, uint32_t slot, std::initializer_list<Arg> args = {}) {
        return vcall_on(thread, obj, slot, args);
    }
    // Same, on another guest thread (one per host thread).
    uint64_t vcall_on(GuestThread& t, uint64_t obj, uint32_t slot, std::initializer_list<Arg> args = {}) {
        t_spec = VCallSpec{};
        t_spec.slot = slot;
        t_spec.args.push_back(I(obj));
        t_spec.args.insert(t_spec.args.end(), args.begin(), args.end());
        return call_guest(arena, t.state, kVCall);
    }
    // Struct-returning call: rcx = result buffer, rdx = this (the order the executable uses), or
    // the MSVC member order when `this_first`.
    uint64_t sret_call(uint64_t obj, uint32_t slot, uint64_t result, std::initializer_list<Arg> args = {},
                       bool this_first = false) {
        t_spec = VCallSpec{};
        t_spec.target = load64(load64(obj) + 8ull * slot);
        if (this_first) {
            t_spec.args = {I(obj), I(result)};
        } else {
            t_spec.args = {I(result), I(obj)};
        }
        t_spec.args.insert(t_spec.args.end(), args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }
    uint64_t dcall(uint64_t target, std::initializer_list<Arg> args = {}) {
        t_spec = VCallSpec{};
        t_spec.target = target;
        t_spec.args.assign(args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }

    uint64_t guid(const Guid& g) { return put(g); }

    // D3D12XboxCreateDevice through the guest-callable export, as the executable calls it.
    uint64_t create_device(int32_t* hr_out = nullptr) {
        uint8_t params[0x50] = {};
        const uint32_t version = 4;
        std::memcpy(params, &version, 4);
        const uint64_t params_va = put_bytes(params, sizeof(params));
        const uint64_t out = put<uint64_t>(0);
        const int32_t hr = static_cast<int32_t>(
            dcall(dx::export_address("D3D12XboxCreateDevice"), {I(0), I(params_va), I(guid(dg::IID_ID3D12Device11)), I(out)}));
        if (hr_out != nullptr) {
            *hr_out = hr;
        }
        return load64(out);
    }

    // Generic `Create*(..., riid, ppv)` helper: returns the created interface pointer.
    uint64_t create(uint64_t obj, uint32_t slot, std::initializer_list<Arg> leading, const Guid& riid,
                    int32_t* hr_out = nullptr) {
        const uint64_t out = put<uint64_t>(0);
        std::vector<Arg> args(leading);
        args.push_back(I(guid(riid)));
        args.push_back(I(out));
        t_spec = VCallSpec{};
        t_spec.slot = slot;
        t_spec.args.push_back(I(obj));
        t_spec.args.insert(t_spec.args.end(), args.begin(), args.end());
        const int32_t hr = static_cast<int32_t>(call_guest(arena, thread.state, kVCall));
        if (hr_out != nullptr) {
            *hr_out = hr;
        }
        return load64(out);
    }
};

int32_t hr(uint64_t rax) {
    return static_cast<int32_t>(static_cast<uint32_t>(rax));
}

bool contains(const std::vector<std::string>& lines, const char* needle) {
    for (const std::string& l : lines) {
        if (l.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

dg::Device* host_device(uint64_t device_ptr) {
    com::detail::Resolved r;
    if (!com::detail::lookup_interface(device_ptr, &r)) {
        return nullptr;
    }
    return static_cast<dg::Device*>(static_cast<dg::IDeviceHost*>(r.host));
}

template <typename T, typename Host>
T* host_object(uint64_t ptr) {
    com::detail::Resolved r;
    if (!com::detail::lookup_interface(ptr, &r)) {
        return nullptr;
    }
    return static_cast<T*>(static_cast<Host*>(r.host));
}

// Slots (see d3d12_guest_objects.h).
namespace Slot {
constexpr uint32_t QueryInterface = 0, AddRef = 1, Release = 2, SetName = 6, GetDevice = 7;
// ID3D12Device
constexpr uint32_t GetNodeCount = 7, CreateCommandQueue = 8, CreateCommandAllocator = 9, CheckFeatureSupport = 13,
                   CreateDescriptorHeap = 14, GetDescriptorHandleIncrementSize = 15, CreateRootSignature = 16,
                   CreateConstantBufferView = 17, CreateSampler = 22, CopyDescriptorsSimple = 24,
                   GetResourceAllocationInfo = 25, GetCustomHeapProperties = 26, CreateFence = 36,
                   GetDeviceRemovedReason = 37, GetCopyableFootprints = 38, CreateQueryHeap = 39,
                   CreateCommandSignature = 41, CreateCommandQueueX = 48;
// ID3D12CommandQueue (Xbox order)
constexpr uint32_t QueueSignal = 16, QueueWait = 17, GetTimestampFrequency = 18, PresentX = 32;
// ID3D12CommandAllocator / ID3D12Fence / ID3D12DescriptorHeap / ID3DBlob
constexpr uint32_t AllocatorReset = 8, GetCompletedValue = 8, SetEventOnCompletion = 9, FenceSignal = 10,
                   HeapGetDesc = 8, GetCPUStart = 9, GetGPUStart = 10, GetBufferPointer = 3, GetBufferSize = 4;
}  // namespace Slot

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_export_registration() {
    using namespace playgta6::os::win32;
    register_all_shim_exports();
    const Pg6ProcAddress create = proc_address(&pg6_d3d12_d3d12_xbox_create_device);
    CHECK(pg6_loader_resolve_import("d3d12_xs.dll", "D3D12XboxCreateDevice") == create);
    CHECK(pg6_loader_resolve_import("D3D12_XS.DLL", "D3D12SerializeVersionedRootSignature") ==
          proc_address(&pg6_d3d12_d3d12_serialize_versioned_root_signature));
    CHECK(pg6_loader_resolve_import("d3d12_xs", "D3D12ConfigureVirtualMemoryX") ==
          proc_address(&pg6_d3d12_d3d12_configure_virtual_memory_x));

    Fixture fx;
    // GetProcAddress-style lookup: the export's proc address maps to a guest-callable thunk.
    CHECK(guest_callable_for_proc(create) == dx::export_address("D3D12XboxCreateDevice"));
    CHECK(is_synthetic_address(dx::export_address("D3D12ConfigureVirtualMemoryX")));
    CHECK(dx::export_address("D3D12CreateDevice") == 0);
}

void test_create_device() {
    Fixture fx;
    // ConfigureVirtualMemoryX runs first in the executable: a 32-byte block, S_OK expected.
    uint8_t block[32] = {};
    block[4] = 0x21;
    CHECK(hr(fx.dcall(dx::export_address("D3D12ConfigureVirtualMemoryX"), {I(put_bytes(block, 32))})) == 0);
    CHECK(dx::virtual_memory_config().calls == 1 && dx::virtual_memory_config().bytes[4] == 0x21);

    int32_t result = -1;
    const uint64_t device = fx.create_device(&result);
    CHECK(result == 0 && device != 0);
    dg::Device* host = host_device(device);
    CHECK(host != nullptr);
    CHECK(host->ref_count() == 1);
    // Inlined accessors read [device + 0x18]; it must be non-zero so descriptor writes use the API.
    uint8_t flag = 0;
    CHECK(com::read_guest(device + 0x18, &flag, 1) && flag == 1);
    CHECK(load64(device) == com::vtable_address(dg::kID3D12Device.info));

    // One device per process: a second create returns the same object with another reference.
    const uint64_t again = fx.create_device(&result);
    CHECK(result == 0 && again == device && host->ref_count() == 2);
    CHECK(dx::current_device().get() == host);

    // QueryInterface: ID3D12Device (base) and IUnknown answer, IDXGIDevice (queried by the executable)
    // does not.
    const uint64_t ppv = put<uint64_t>(0);
    CHECK(hr(fx.vcall(device, Slot::QueryInterface, {I(fx.guid(dg::IID_ID3D12Device)), I(ppv)})) == 0);
    CHECK(load64(ppv) == device);
    CHECK(hr(fx.vcall(device, Slot::QueryInterface,
                      {I(fx.guid(com::make_guid("54ec77fa-1377-44e6-8c32-88fd5f44c84c"))), I(ppv)})) ==
          dg::kENoInterface);
    CHECK(load64(ppv) == 0);
    CHECK(fx.vcall(device, Slot::GetNodeCount) == 1);
    CHECK(hr(fx.vcall(device, Slot::GetDeviceRemovedReason)) == 0);

    // A riid the device does not implement fails and hands out nothing.
    const uint64_t out = put<uint64_t>(0x55);
    CHECK(hr(fx.dcall(dx::export_address("D3D12XboxCreateDevice"),
                      {I(0), I(0), I(fx.guid(dg::IID_ID3D12Fence)), I(out)})) == dg::kENoInterface);
    CHECK(load64(out) == 0);

    // Release all three references (create, create, QI) -> destroyed, nothing leaks.
    CHECK(fx.vcall(device, Slot::Release) == 2);
    CHECK(fx.vcall(device, Slot::Release) == 1);
    CHECK(fx.vcall(device, Slot::Release) == 0);
    CHECK(!dx::current_device());
    CHECK(com::bridge_stats().live_objects == 0);
    // A new create after the final release makes a fresh device.
    const uint64_t fresh = fx.create_device(&result);
    CHECK(result == 0 && fresh != 0);
    CHECK(fx.vcall(fresh, Slot::Release) == 0);
    CHECK(t_balanced);
}

void test_queue_allocator_fence() {
    Fixture fx;
    const uint64_t device = fx.create_device();
    dg::Device* dev = host_device(device);

    // CreateCommandQueue (public slot 8) and the Xbox queue slot 48 take the same 16-byte descriptor.
    const uint64_t desc = put(dg::GuestCommandQueueDesc{0, 0, 0, 0});
    int32_t result = -1;
    const uint64_t queue = fx.create(device, Slot::CreateCommandQueue, {I(desc)}, dg::IID_ID3D12CommandQueue, &result);
    CHECK(result == 0 && queue != 0);
    const uint64_t compute_desc = put(dg::GuestCommandQueueDesc{2, 0, 0, 0});
    const uint64_t queue2 =
        fx.create(device, Slot::CreateCommandQueueX, {I(compute_desc)}, dg::IID_ID3D12CommandQueue, &result);
    CHECK(result == 0 && queue2 != 0 && queue2 != queue);
    CHECK(dev->stats().queues == 2);
    // A bundle queue is invalid.
    const uint64_t bad = put(dg::GuestCommandQueueDesc{1, 0, 0, 0});
    CHECK(fx.create(device, Slot::CreateCommandQueue, {I(bad)}, dg::IID_ID3D12CommandQueue, &result) == 0);
    CHECK(result == dg::kEInvalidArg);
    CHECK(dev->stats().bad_arguments == 1);
    // Compute queues with the Xbox queue flag bit 1 (flags 2 and 3) are accepted; unknown bits are not.
    for (uint32_t flags : {2u, 3u}) {
        const uint64_t flagged = put(dg::GuestCommandQueueDesc{2, 0, flags, 0});
        const uint64_t q = fx.create(device, Slot::CreateCommandQueueX, {I(flagged)}, dg::IID_ID3D12CommandQueue, &result);
        CHECK(q != 0 && result == 0);
        fx.vcall(q, Slot::Release);
    }
    const uint64_t unknown_flags = put(dg::GuestCommandQueueDesc{2, 0, 4, 0});
    CHECK(fx.create(device, Slot::CreateCommandQueueX, {I(unknown_flags)}, dg::IID_ID3D12CommandQueue, &result) == 0);
    CHECK(result == dg::kEInvalidArg);
    CHECK(dev->stats().bad_arguments == 2);

    const uint64_t freq = put<uint64_t>(0);
    CHECK(hr(fx.vcall(queue, Slot::GetTimestampFrequency, {I(freq)})) == 0);
    CHECK(load64(freq) == dev->timestamp_frequency());

    // Children answer GetDevice with the device and keep it alive.
    const uint64_t ppv = put<uint64_t>(0);
    CHECK(hr(fx.vcall(queue, Slot::GetDevice, {I(fx.guid(dg::IID_ID3D12Device11)), I(ppv)})) == 0);
    CHECK(load64(ppv) == device);
    CHECK(fx.vcall(device, Slot::Release) == 3);  // the GetDevice reference (own + 2 queues remain)

    const uint64_t allocator =
        fx.create(device, Slot::CreateCommandAllocator, {I(0)}, dg::IID_ID3D12CommandAllocator, &result);
    CHECK(result == 0 && allocator != 0);
    CHECK(hr(fx.vcall(allocator, Slot::AllocatorReset)) == 0);
    CHECK(fx.create(device, Slot::CreateCommandAllocator, {I(9)}, dg::IID_ID3D12CommandAllocator, &result) == 0 &&
          result == dg::kEInvalidArg);

    // Fence: CPU signal, queue signal and Win32 event completion.
    const uint64_t fence = fx.create(device, Slot::CreateFence, {I(5), I(0)}, dg::IID_ID3D12Fence, &result);
    CHECK(result == 0 && fence != 0);
    CHECK(fx.vcall(fence, Slot::GetCompletedValue) == 5);
    const Pg6Handle event = pg6_win32_create_event_w(nullptr, 1, 0, nullptr);
    CHECK(event != 0);
    CHECK(hr(fx.vcall(fence, Slot::SetEventOnCompletion, {I(7), I(event)})) == 0);
    CHECK(pg6_win32_wait_for_single_object(event, 0) != 0);  // not signalled yet
    CHECK(hr(fx.vcall(queue, Slot::QueueSignal, {I(fence), I(6)})) == 0);
    CHECK(fx.vcall(fence, Slot::GetCompletedValue) == 6);
    CHECK(pg6_win32_wait_for_single_object(event, 0) != 0);
    CHECK(hr(fx.vcall(queue, Slot::QueueSignal, {I(fence), I(7)})) == 0);
    CHECK(pg6_win32_wait_for_single_object(event, 0) == 0);  // WAIT_OBJECT_0
    // Already reached: the event is set immediately.
    CHECK(pg6_win32_reset_event(event) != 0);
    CHECK(hr(fx.vcall(fence, Slot::SetEventOnCompletion, {I(3), I(event)})) == 0);
    CHECK(pg6_win32_wait_for_single_object(event, 0) == 0);
    // CPU Signal through the fence; a null event blocks until reached (here from another thread).
    std::thread signaller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        host_object<dg::Fence, dg::IFenceHost>(fence)->Signal(20);
    });
    CHECK(hr(fx.vcall(fence, Slot::SetEventOnCompletion, {I(20), I(0)})) == 0);
    signaller.join();
    CHECK(fx.vcall(fence, Slot::GetCompletedValue) == 20);
    CHECK(hr(fx.vcall(fence, Slot::FenceSignal, {I(25)})) == 0);
    CHECK(fx.vcall(fence, Slot::GetCompletedValue) == 25);
    // Xbox slot 12: the address of a word holding the completed value, the same on every call and
    // kept current by CPU and queue signals.
    const uint64_t word = fx.vcall(fence, 12);
    CHECK(word != 0 && fx.vcall(fence, 12) == word);
    CHECK(load64(word) == 25);
    CHECK(hr(fx.vcall(queue, Slot::QueueSignal, {I(fence), I(30)})) == 0);
    CHECK(load64(word) == 30);
    CHECK(hr(fx.vcall(fence, Slot::FenceSignal, {I(~0ull)})) == 0);  // device-removed value
    CHECK(load64(word) == ~0ull);
    CHECK(hr(fx.vcall(fence, Slot::FenceSignal, {I(25)})) == 0);
    CHECK(load64(word) == 25);
    CHECK(com::stub_hits(dg::kID3D12Fence, 12) == 0);
    CHECK(hr(fx.vcall(queue, Slot::QueueWait, {I(fence), I(25)})) == 0);
    // A non-fence interface as fence argument is rejected.
    CHECK(hr(fx.vcall(queue, Slot::QueueSignal, {I(allocator), I(1)})) == dg::kEInvalidArg);
    pg6_win32_close_handle(event);

    // PresentX with planes ends the frame: BeginFrame (first packet of the frame) + Present (a
    // plane without a resource still presents; G27 tests cover the back buffer).
    fx.captured.clear();
    CHECK(hr(fx.vcall(queue, Slot::PresentX, {I(1), I(0), I(0)})) == dg::kEInvalidArg);  // no plane array
    CHECK(fx.captured.empty());
    const uint64_t plane = put(dg::xbox::GuestPresentPlaneX{});
    CHECK(hr(fx.vcall(queue, Slot::PresentX, {I(1), I(plane), I(0)})) == 0);
    CHECK(fx.captured.size() == 2);
    GpuPacketHeader h{};
    std::memcpy(&h, fx.captured[0].data(), sizeof(h));
    CHECK(h.opcode == static_cast<uint16_t>(GpuOpcode::BeginFrame));
    std::memcpy(&h, fx.captured[1].data(), sizeof(h));
    CHECK(h.opcode == static_cast<uint16_t>(GpuOpcode::Present));
    CHECK(hr(fx.vcall(queue, Slot::PresentX, {I(0), I(0), I(0)})) == 0);  // no planes: nothing
    CHECK(fx.captured.size() == 2);
    CHECK(dev->stats().presents == 1);

    // Lifetime: the device outlives its own release while children hold it.
    CHECK(fx.vcall(device, Slot::Release) == 4);  // the children (2 queues, allocator, fence) keep it alive
    for (uint64_t child : {queue, queue2, allocator, fence}) {
        CHECK(fx.vcall(child, Slot::Release) == 0);
    }
    CHECK(com::bridge_stats().live_objects == 0);
}

void test_descriptor_heaps() {
    Fixture fx;
    const uint64_t device = fx.create_device();
    dg::Device* dev = host_device(device);
    int32_t result = -1;

    CHECK(fx.vcall(device, Slot::GetDescriptorHandleIncrementSize, {I(0)}) == 32);
    CHECK(fx.vcall(device, Slot::GetDescriptorHandleIncrementSize, {I(1)}) == 16);
    CHECK(fx.vcall(device, Slot::GetDescriptorHandleIncrementSize, {I(7)}) == 0);

    const uint64_t srv_desc = put(dg::GuestDescriptorHeapDesc{0, 64, 1, 0});
    const uint64_t heap = fx.create(device, Slot::CreateDescriptorHeap, {I(srv_desc)}, dg::IID_ID3D12DescriptorHeap, &result);
    CHECK(result == 0 && heap != 0);
    auto* host_heap = host_object<dg::DescriptorHeap, dg::IDescriptorHeapHost>(heap);
    CHECK(host_heap != nullptr && host_heap->heap()->capacity() == 64);

    // Struct returns: the executable passes the result pointer in rcx and `this` in rdx. With
    // `this` in rcx an 8-byte handle comes back in rax (the executable's other call shape for
    // slot 9; rdx is whatever the caller left there and is not written).
    const uint64_t result_buf = put<uint64_t>(0);
    CHECK(fx.sret_call(heap, Slot::GetCPUStart, result_buf) == result_buf);
    const uint64_t cpu_start = load64(result_buf);
    CHECK(cpu_start == host_heap->heap()->cpu_start().ptr);
    // CPU handles point at guest memory of the heap's size (the executable writes through them).
    CHECK(com::guest_to_host(cpu_start, 64 * 32) != nullptr);
    store64(result_buf, 0);
    CHECK(fx.sret_call(heap, Slot::GetCPUStart, result_buf, {}, /*this_first=*/true) == cpu_start);
    CHECK(load64(result_buf) == 0);
    CHECK(fx.sret_call(heap, Slot::GetGPUStart, result_buf) == result_buf);
    CHECK(load64(result_buf) == host_heap->heap()->gpu_start().ptr && load64(result_buf) != 0);
    CHECK(dg::sret_stats().result_first == 2 && dg::sret_stats().this_first == 1);
    const uint64_t desc_buf = put(dg::GuestDescriptorHeapDesc{});
    fx.sret_call(heap, Slot::HeapGetDesc, desc_buf);
    CHECK(get<dg::GuestDescriptorHeapDesc>(desc_buf).NumDescriptors == 64);

    // CBV into slot 3 (handle arithmetic: start + 3 * increment), then copy it to slot 10.
    const uint64_t cbv = put(dg::GuestConstantBufferViewDesc{0x1000, 256});
    fx.vcall(device, Slot::CreateConstantBufferView, {I(cbv), I(cpu_start + 3 * 32)});
    const d3d::ViewDescriptor* view = host_heap->heap()->view(3);
    CHECK(view != nullptr && view->kind == d3d::DescriptorKind::Cbv && view->first_element_or_address == 0x1000 &&
          view->num_elements_or_size == 256);
    fx.vcall(device, Slot::CopyDescriptorsSimple, {I(1), I(cpu_start + 10 * 32), I(cpu_start + 3 * 32), I(0)});
    CHECK(host_heap->heap()->view(10)->kind == d3d::DescriptorKind::Cbv);

    // Xbox descriptor placed in title memory: {type, count, flags, node, address, size == count *
    // increment}. Its CPU handles are that address plus index * increment.
    {
        struct PlacedDesc {
            dg::GuestDescriptorHeapDesc base;
            uint64_t address;
            uint64_t size;
        };
        const uint64_t memory = 0x1'7000'0000ull;  // any address; descriptors live in the registry
        const uint64_t placed_desc = put(PlacedDesc{{0, 8, 1, 1}, memory, 8 * 32});
        const uint64_t placed = fx.create(device, Slot::CreateDescriptorHeap, {I(placed_desc)},
                                          dg::IID_ID3D12DescriptorHeap, &result);
        CHECK(result == 0 && placed != 0);
        CHECK(fx.sret_call(placed, Slot::GetCPUStart, result_buf, {}, /*this_first=*/true) == memory);
        auto* host_placed = host_object<dg::DescriptorHeap, dg::IDescriptorHeapHost>(placed);
        fx.vcall(device, Slot::CreateConstantBufferView, {I(cbv), I(memory + 5 * 32)});
        CHECK(host_placed->heap()->view(5) != nullptr && host_placed->heap()->view(5)->kind == d3d::DescriptorKind::Cbv);
        // A second heap overlapping the first is refused; a size that does not match is a public desc.
        CHECK(fx.create(device, Slot::CreateDescriptorHeap, {I(put(PlacedDesc{{0, 8, 1, 1}, memory + 64, 8 * 32}))},
                        dg::IID_ID3D12DescriptorHeap, &result) == 0);
        const uint64_t loose = fx.create(device, Slot::CreateDescriptorHeap,
                                         {I(put(PlacedDesc{{0, 8, 1, 1}, memory + 0x10000, 7}))},
                                         dg::IID_ID3D12DescriptorHeap, &result);
        CHECK(loose != 0 && fx.sret_call(loose, Slot::GetCPUStart, result_buf, {}, true) != memory + 0x10000);
        CHECK(dev->counter(dg::xbox::Counter::PlacedDescriptorHeaps) == 1);
        fx.vcall(loose, Slot::Release);
        fx.vcall(placed, Slot::Release);
    }

    // Sampler heap + CreateSampler.
    const uint64_t smp_desc = put(dg::GuestDescriptorHeapDesc{1, 16, 1, 0});
    const uint64_t smp_heap = fx.create(device, Slot::CreateDescriptorHeap, {I(smp_desc)}, dg::IID_ID3D12DescriptorHeap);
    auto* host_smp = host_object<dg::DescriptorHeap, dg::IDescriptorHeapHost>(smp_heap);
    dg::GuestSamplerDesc sd{};
    sd.Filter = 0x55;
    sd.AddressU = sd.AddressV = sd.AddressW = 3;
    sd.MaxAnisotropy = 8;
    sd.MaxLOD = 1000.0f;
    fx.vcall(device, Slot::CreateSampler, {I(put(sd)), I(host_smp->heap()->cpu_start().ptr + 16)});
    const d3d::SamplerDescriptor* s = host_smp->heap()->sampler(1);
    CHECK(s != nullptr && s->kind == d3d::DescriptorKind::Sampler && s->desc.filter == 0x55 &&
          s->desc.max_anisotropy == 8);

    // Malformed input is diagnosed, never crashes.
    CHECK(fx.create(device, Slot::CreateDescriptorHeap, {I(0)}, dg::IID_ID3D12DescriptorHeap, &result) == 0 &&
          result == dg::kEInvalidArg);
    const uint64_t rtv_visible = put(dg::GuestDescriptorHeapDesc{2, 4, 1, 0});  // RTV heaps cannot be visible
    CHECK(fx.create(device, Slot::CreateDescriptorHeap, {I(rtv_visible)}, dg::IID_ID3D12DescriptorHeap, &result) == 0 &&
          result == dg::kEInvalidArg);
    const uint64_t bad_before = dev->stats().bad_arguments;
    fx.vcall(device, Slot::CreateConstantBufferView, {I(cbv), I(0x1234)});  // not a descriptor handle
    fx.vcall(device, Slot::CreateSampler, {I(0x7000'0000'0000ULL), I(host_smp->heap()->cpu_start().ptr)});
    CHECK(dev->stats().bad_arguments == bad_before + 2);
    CHECK(contains(dev->diagnostics(), "CreateConstantBufferView: InvalidHandle"));

    CHECK(dev->stats().descriptor_heaps == 2);
    fx.vcall(heap, Slot::Release);
    CHECK(dev->stats().descriptor_heaps == 1 && dev->heaps().heap_count() == 1);
    fx.vcall(smp_heap, Slot::Release);
    CHECK(fx.vcall(device, Slot::Release) == 0);
    CHECK(com::bridge_stats().live_objects == 0);
}

// D3D12_VERSIONED_ROOT_SIGNATURE_DESC 1.1: { table(SRV t0..t3, CBV b1), constants b0 x4, root CBV b2 },
// one static sampler.
uint64_t build_root_signature_desc(uint32_t version = 2, uint32_t flags = 1) {
    dg::GuestDescriptorRange1 ranges[2] = {{0, 4, 0, 0, 0, 0xffffffffu}, {2, 1, 1, 0, 0, 0xffffffffu}};
    const uint64_t ranges_va = put_bytes(ranges, sizeof(ranges));
    dg::GuestRootParameter params[3] = {};
    params[0].ParameterType = 0;
    params[0].w0 = 2;
    params[0].q = ranges_va;
    params[0].ShaderVisibility = 5;  // pixel
    params[1].ParameterType = 1;
    params[1].w0 = 0;
    params[1].q = 4;  // Num32BitValues
    params[2].ParameterType = 2;
    params[2].w0 = 2;
    const uint64_t params_va = put_bytes(params, sizeof(params));
    dg::GuestStaticSamplerDesc sampler{};
    sampler.Filter = 0x15;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = 1;
    sampler.MaxAnisotropy = 1;
    sampler.MaxLOD = 1000.0f;
    sampler.ShaderVisibility = 5;
    dg::GuestVersionedRootSignatureDesc desc{};
    desc.Version = version;
    desc.NumParameters = 3;
    desc.pParameters = params_va;
    desc.NumStaticSamplers = 1;
    desc.pStaticSamplers = put(sampler);
    desc.Flags = flags;
    return put(desc);
}

void test_root_signatures_and_blobs() {
    Fixture fx;
    const uint64_t device = fx.create_device();
    dg::Device* dev = host_device(device);
    const uint64_t serialize = dx::export_address("D3D12SerializeVersionedRootSignature");
    const uint64_t blob_out = put<uint64_t>(0);
    const uint64_t err_out = put<uint64_t>(0x77);

    CHECK(hr(fx.dcall(serialize, {I(build_root_signature_desc()), I(blob_out), I(err_out)})) == 0);
    const uint64_t blob = load64(blob_out);
    CHECK(blob != 0 && load64(err_out) == 0);
    const uint64_t data = fx.vcall(blob, Slot::GetBufferPointer);
    const uint64_t size = fx.vcall(blob, Slot::GetBufferSize);
    CHECK(data != 0 && size > 16);
    // The blob round-trips through the G6 parser.
    std::vector<uint8_t> bytes(size);
    CHECK(com::read_guest(data, bytes.data(), bytes.size()));
    d3d::RootSignatureDesc parsed;
    CHECK(d3d::parse_root_signature_blob(bytes.data(), bytes.size(), &parsed).ok());
    CHECK(parsed.parameters.size() == 3 && parsed.parameters[0].ranges.size() == 2 &&
          parsed.parameters[1].num_32bit_values == 4 && parsed.static_samplers.size() == 1);

    int32_t result = -1;
    const uint64_t rs = fx.create(device, Slot::CreateRootSignature, {I(0), I(data), I(size)},
                                  dg::IID_ID3D12RootSignature, &result);
    CHECK(result == 0 && rs != 0);
    auto* host_rs = host_object<dg::RootSignature, dg::IRootSignatureHost>(rs);
    CHECK(host_rs != nullptr && host_rs->signature().parameter_count() == 3);
    CHECK(dev->stats().root_signatures == 1);
    CHECK((dev->root_signature_versions() == std::array<uint32_t, 3>{0, 1, 0}));  // one version 1.1

    // Version 1.0 layout (20-byte ranges, no flags) is read too; the empty local root signature the
    // executable serializes (version 1.1, no parameters, flags LOCAL_ROOT_SIGNATURE) works.
    dg::GuestVersionedRootSignatureDesc empty{};
    empty.Version = 2;
    empty.Flags = 0x80;
    CHECK(hr(fx.dcall(serialize, {I(put(empty)), I(blob_out), I(err_out)})) == 0);
    CHECK(fx.vcall(load64(blob_out), Slot::Release) == 0);

    // Malformed: bad version, unreadable parameter array, invalid range type -> error blob text.
    dg::GuestVersionedRootSignatureDesc bad{};
    bad.Version = 9;
    CHECK(hr(fx.dcall(serialize, {I(put(bad)), I(blob_out), I(err_out)})) == dg::kEInvalidArg);
    CHECK(load64(blob_out) == 0);
    const uint64_t err_blob = load64(err_out);
    CHECK(err_blob != 0);
    std::string text(fx.vcall(err_blob, Slot::GetBufferSize), '\0');
    com::read_guest(fx.vcall(err_blob, Slot::GetBufferPointer), text.data(), text.size());
    CHECK(text.find("unsupported root signature version 9") != std::string::npos);
    fx.vcall(err_blob, Slot::Release);
    bad.Version = 2;
    bad.NumParameters = 2;
    bad.pParameters = 0x7000'0000'0000ULL;
    CHECK(hr(fx.dcall(serialize, {I(put(bad)), I(blob_out), I(0)})) == dg::kEInvalidArg);

    // CreateRootSignature with garbage bytes is rejected and diagnosed.
    const uint8_t junk[32] = {1, 2, 3};
    CHECK(fx.create(device, Slot::CreateRootSignature, {I(0), I(put_bytes(junk, 32)), I(32)},
                    dg::IID_ID3D12RootSignature, &result) == 0);
    CHECK(result == dg::kEInvalidArg && contains(dev->diagnostics(), "blob parse failed"));

    fx.vcall(rs, Slot::Release);
    fx.vcall(blob, Slot::Release);
    fx.vcall(device, Slot::Release);
    CHECK(com::bridge_stats().live_objects == 0);
}

void test_queries_and_footprints() {
    Fixture fx;
    const uint64_t device = fx.create_device();
    int32_t result = -1;

    // CheckFeatureSupport: ROOT_SIGNATURE (in/out), ARCHITECTURE, FORMAT_SUPPORT, wrong size.
    const uint64_t rs_data = put<uint32_t>(0x3);
    CHECK(hr(fx.vcall(device, Slot::CheckFeatureSupport, {I(12), I(rs_data), I(4)})) == 0);
    CHECK(load32(rs_data) == 0x3);
    const uint32_t arch[4] = {0, 9, 9, 9};
    const uint64_t arch_va = put_bytes(arch, sizeof(arch));
    CHECK(hr(fx.vcall(device, Slot::CheckFeatureSupport, {I(1), I(arch_va), I(16)})) == 0);
    CHECK(load32(arch_va + 8) == 1);  // UMA
    const uint32_t fmt[3] = {28, 0, 0};  // R8G8B8A8_UNORM
    const uint64_t fmt_va = put_bytes(fmt, sizeof(fmt));
    CHECK(hr(fx.vcall(device, Slot::CheckFeatureSupport, {I(3), I(fmt_va), I(12)})) == 0);
    CHECK((load32(fmt_va + 4) & 0x10000) != 0 && (load32(fmt_va + 4) & 0x200) != 0);
    CHECK(hr(fx.vcall(device, Slot::CheckFeatureSupport, {I(3), I(fmt_va), I(8)})) == dg::kEInvalidArg);
    CHECK(hr(fx.vcall(device, Slot::CheckFeatureSupport, {I(99), I(fmt_va), I(12)})) == dg::kEInvalidArg);

    // GetCustomHeapProperties (struct return, result pointer first).
    const uint64_t props = put(dg::GuestHeapProperties{});
    CHECK(fx.sret_call(device, Slot::GetCustomHeapProperties, props, {I(1), I(2)}) == props);
    const auto p = get<dg::GuestHeapProperties>(props);
    CHECK(p.Type == 4 && p.CPUPageProperty == 2 && p.MemoryPoolPreference == 1);

    // GetResourceAllocationInfo for a buffer and a mipped 2D texture.
    dg::GuestResourceDesc descs[2] = {};
    descs[0].Dimension = 1;
    descs[0].Width = 1000;
    descs[0].Height = 1;
    descs[0].DepthOrArraySize = 1;
    descs[0].MipLevels = 1;
    descs[0].SampleCount = 1;
    descs[1].Dimension = 3;
    descs[1].Width = 256;
    descs[1].Height = 256;
    descs[1].DepthOrArraySize = 1;
    descs[1].MipLevels = 9;
    descs[1].Format = 28;
    descs[1].SampleCount = 1;
    const uint64_t info = put(dg::GuestResourceAllocationInfo{});
    fx.sret_call(device, Slot::GetResourceAllocationInfo, info, {I(0), I(2), I(put_bytes(descs, sizeof(descs)))});
    const auto ai = get<dg::GuestResourceAllocationInfo>(info);
    CHECK(ai.Alignment == 65536 && ai.SizeInBytes == 65536 + 393216);  // 256*256*4*4/3 rounded to 64K

    // GetCopyableFootprints: 2D texture, mips 0..2.
    const uint64_t layouts = com::guest_alloc(3 * sizeof(dg::GuestPlacedSubresourceFootprint));
    const uint64_t rows = com::guest_alloc(3 * 4);
    const uint64_t row_sizes = com::guest_alloc(3 * 8);
    const uint64_t total = put<uint64_t>(0);
    fx.vcall(device, Slot::GetCopyableFootprints,
             {I(put(descs[1])), I(0), I(3), I(0), I(layouts), I(rows), I(row_sizes), I(total)});
    const auto l0 = get<dg::GuestPlacedSubresourceFootprint>(layouts);
    const auto l2 = get<dg::GuestPlacedSubresourceFootprint>(layouts + 2 * sizeof(dg::GuestPlacedSubresourceFootprint));
    CHECK(l0.Offset == 0 && l0.Footprint.Width == 256 && l0.Footprint.RowPitch == 1024 && load32(rows) == 256);
    CHECK(l2.Footprint.Width == 64 && l2.Footprint.RowPitch == 256 && l2.Offset == 262144 + 65536);
    CHECK(load64(row_sizes + 16) == 256);
    CHECK(load64(total) == 262144 + 65536 + 64 * 256);
    // BC1: 4x4 blocks of 8 bytes.
    dg::GuestResourceDesc bc = descs[1];
    bc.Format = 71;
    bc.MipLevels = 1;
    fx.vcall(device, Slot::GetCopyableFootprints,
             {I(put(bc)), I(0), I(1), I(0), I(layouts), I(rows), I(row_sizes), I(total)});
    CHECK(load32(rows) == 64 && load64(row_sizes) == 64 * 8);
    // Out-of-range subresources: total = UINT64_MAX.
    fx.vcall(device, Slot::GetCopyableFootprints, {I(put(bc)), I(0), I(5), I(0), I(0), I(0), I(0), I(total)});
    CHECK(load64(total) == UINT64_MAX);

    // Query heap and command signature objects (created at startup by the executable).
    const uint64_t qh = fx.create(device, Slot::CreateQueryHeap, {I(put(dg::GuestQueryHeapDesc{1, 64, 0}))},
                                  dg::IID_ID3D12QueryHeap, &result);
    CHECK(result == 0 && qh != 0);
    const uint32_t arg_desc[4] = {1, 0, 0, 0};
    dg::GuestCommandSignatureDesc cs{};
    cs.ByteStride = 16;
    cs.NumArgumentDescs = 1;
    cs.pArgumentDescs = put_bytes(arg_desc, sizeof(arg_desc));
    const uint64_t sig = fx.create(device, Slot::CreateCommandSignature, {I(put(cs)), I(0)},
                                   dg::IID_ID3D12CommandSignature, &result);
    CHECK(result == 0 && sig != 0);
    fx.vcall(qh, Slot::Release);
    fx.vcall(sig, Slot::Release);
    fx.vcall(device, Slot::Release);
    CHECK(com::bridge_stats().live_objects == 0);
}

void test_unknown_slots_and_private_data() {
    Fixture fx;
    const uint64_t device = fx.create_device();
    com::clear_diagnostics();

    // An unestablished slot is a logging stub: E_NOTIMPL, counted, logged once with its index.
    CHECK(hr(fx.vcall(device, 99, {I(1), I(2)})) == dg::kENotImpl);
    CHECK(hr(fx.vcall(device, 99)) == dg::kENotImpl);
    CHECK(com::stub_hits(dg::kID3D12Device, 99) == 2);
    CHECK(com::stub_hits(dg::kID3D12Device, 31) == 0);
    CHECK(hr(fx.vcall(device, 31)) == dg::kENotImpl);  // named stub (CreateSharedHandle)
    bool found = false;
    for (const std::string& d : com::diagnostics()) {
        found = found || (d.find("ID3D12Device") != std::string::npos && d.find("slot 99") != std::string::npos);
    }
    CHECK(found);
    bool named = false;
    for (const com::StubHit& h : com::stub_hit_report()) {
        named = named || (h.slot == 31 && h.method_name == "CreateSharedHandle");
    }
    CHECK(named);

    // ID3D12Object: SetName / private data.
    const char16_t name[] = u"GraphicsDevice";
    CHECK(hr(fx.vcall(device, Slot::SetName, {I(put_bytes(name, sizeof(name)))})) == 0);
    CHECK(host_device(device)->debug_name() == u"GraphicsDevice");
    const Guid key = com::make_guid("01020304-0506-0708-090a-0b0c0d0e0f10");
    const uint32_t payload = 0xCAFE;
    CHECK(hr(fx.vcall(device, 4, {I(fx.guid(key)), I(4), I(put(payload))})) == 0);
    const uint64_t size = put<uint32_t>(4);
    const uint64_t out = put<uint32_t>(0);
    CHECK(hr(fx.vcall(device, 3, {I(fx.guid(key)), I(size), I(out)})) == 0);
    CHECK(load32(out) == 0xCAFE);
    CHECK(hr(fx.vcall(device, 3, {I(fx.guid(com::make_guid("ffffffff-0506-0708-090a-0b0c0d0e0f10"))), I(size), I(out)})) ==
          dg::kDxgiErrorNotFound);

    // A method called with a pointer that is not a device faults with a diagnostic (no host crash).
    GuestFault fault{};
    bool faulted = false;
    try {
        t_spec = VCallSpec{};
        t_spec.target = load64(load64(device) + 8 * Slot::GetNodeCount);
        t_spec.args = {I(0x1234560)};
        call_guest(fx.arena, fx.thread.state, kVCall);
    } catch (const GuestFault& f) {
        fault = f;
        faulted = true;
    }
    CHECK(faulted && fault.info.kind == static_cast<uint32_t>(FaultKind::Trap));
    fx.thread.state.gpr[GuestReg::Rsp] = fx.thread.stack_top;
    fx.vcall(device, Slot::Release);
}


// ---------------------------------------------------------------------------
// Resources, pipeline states, command lists (part 2)
// ---------------------------------------------------------------------------

namespace Slot2 {
// ID3D12Device
constexpr uint32_t CreateGraphicsPipelineState = 10, CreateComputePipelineState = 11, CreateCommandList = 12,
                   CreateShaderResourceView = 18, CreateRenderTargetView = 20, CreateCommittedResource = 27,
                   CreateHeap = 28, CreatePlacedResource = 29, CreateReservedResource = 30, CreateCommandListX = 73,
                   CreateCommittedResource3 = 127, CreatePlacedResourceX = 133;
// ID3D12Resource
constexpr uint32_t Map = 8, Unmap = 9, GetDesc = 10, GetGPUVirtualAddress = 11, GetHeapProperties = 14;
// ID3D12CommandQueue (Xbox)
constexpr uint32_t ExecuteCommandLists = 35;
// ID3D12GraphicsCommandList
constexpr uint32_t Close = 9, Reset = 10, DrawInstanced = 12, DrawIndexedInstanced = 13, CopyBufferRegion = 15,
                   CopyTextureRegion = 16, IASetPrimitiveTopology = 20, RSSetViewports = 21, RSSetScissorRects = 22,
                   SetPipelineState = 25, ResourceBarrier = 26, SetGraphicsRootSignature = 34, IASetIndexBuffer = 47,
                   IASetVertexBuffers = 48, OMSetRenderTargets = 50, ClearDepthStencilView = 51,
                   ClearRenderTargetView = 52;  // Xbox order (G27)
}  // namespace Slot2

struct Packet {
    GpuOpcode opcode{};
    std::vector<uint8_t> payload;
};

std::vector<Packet> packets_since(const Fixture& fx, size_t first_capture) {
    std::vector<uint8_t> all;
    for (size_t i = first_capture; i < fx.captured.size(); ++i) {
        all.insert(all.end(), fx.captured[i].begin(), fx.captured[i].end());
    }
    std::vector<Packet> out;
    size_t off = 0;
    while (off + sizeof(GpuPacketHeader) <= all.size()) {
        GpuPacketHeader h{};
        std::memcpy(&h, all.data() + off, sizeof(h));
        off += sizeof(h);
        Packet p;
        p.opcode = static_cast<GpuOpcode>(h.opcode);
        p.payload.assign(all.begin() + static_cast<std::ptrdiff_t>(off),
                         all.begin() + static_cast<std::ptrdiff_t>(off + h.payload_bytes));
        off += h.payload_bytes;
        out.push_back(std::move(p));
    }
    CHECK(off == all.size());
    return out;
}

template <typename T>
T payload_as(const Packet& p, size_t offset = 0) {
    T v{};
    CHECK(p.payload.size() >= offset + sizeof(T));
    if (p.payload.size() >= offset + sizeof(T)) {
        std::memcpy(&v, p.payload.data() + offset, sizeof(T));
    }
    return v;
}

size_t count_op(const std::vector<Packet>& ps, GpuOpcode op) {
    size_t n = 0;
    for (const Packet& p : ps) {
        n += p.opcode == op ? 1 : 0;
    }
    return n;
}

const Packet* find_op(const std::vector<Packet>& ps, GpuOpcode op, size_t nth = 0) {
    for (const Packet& p : ps) {
        if (p.opcode == op && nth-- == 0) {
            return &p;
        }
    }
    return nullptr;
}

size_t index_of(const std::vector<Packet>& ps, GpuOpcode op) {
    for (size_t i = 0; i < ps.size(); ++i) {
        if (ps[i].opcode == op) {
            return i;
        }
    }
    return SIZE_MAX;
}

dg::GuestResourceDesc buffer_desc(uint64_t size) {
    dg::GuestResourceDesc d{};
    d.Dimension = 1;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleCount = 1;
    d.Layout = 1;  // ROW_MAJOR
    return d;
}

dg::GuestResourceDesc texture_desc(uint32_t w, uint32_t h, uint32_t format, uint32_t flags, uint16_t mips = 1) {
    dg::GuestResourceDesc d{};
    d.Dimension = 3;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleCount = 1;
    d.Flags = flags;
    return d;
}

dg::GuestHeapProperties heap_props(uint32_t type) {
    dg::GuestHeapProperties p{};
    p.Type = type;
    p.CreationNodeMask = 1;
    p.VisibleNodeMask = 1;
    return p;
}

// CreateCommittedResource (public slot 27) through the guest vtable.
uint64_t committed(Fixture& fx, uint64_t device, uint32_t heap_type, const dg::GuestResourceDesc& desc, uint32_t state,
                   int32_t* result = nullptr) {
    return fx.create(device, Slot2::CreateCommittedResource,
                     {I(put(heap_props(heap_type))), I(0), I(put(desc)), I(state), I(0)}, dg::IID_ID3D12Resource, result);
}

template <typename T>
T* object_of(uint64_t ptr) {
    com::detail::Resolved r;
    if (!com::detail::lookup_interface(ptr, &r)) {
        return nullptr;
    }
    return static_cast<T*>(r.object.get());
}

// Committed render targets allocate their memory with XMemAlloc (allocator id 0x80, page-size
// field 2, 64 KiB-block sizes), so a title's hooked arena accounts for them: two 3840x2160 and two
// 1920x1080 RGBA8 targets take 84,541,440 of 86,507,520 bytes, less than one large page left.
void test_committed_render_targets_through_xmem_hooks() {
    Fixture fx;
    const uint64_t device = fx.create_device();
    auto& xmem_arena = *playgta6::os::win32::get_vmem_arena();
    const uint64_t table =
        xmem_arena.virtual_alloc(0, 0x1000, playgta6::os::kMemReserve | playgta6::os::kMemCommit,
                                 playgta6::os::PageProtect::ReadWrite);
    const uint64_t hooks[2] = {kRtAllocHook, kRtFreeHook};
    CHECK(table != 0 && xmem_arena.write(table, hooks, sizeof(hooks)));

    // Without hooks nothing changes for the title: the default allocator.
    const uint64_t plain = committed(fx, device, 1, texture_desc(1920, 1080, 28, 0x5), 0);
    CHECK(plain != 0 && g_rt_base == 0);
    fx.vcall(plain, Slot::Release);

    playgta6::os::scarlett::xmem_set_guest_hook_table(table);
    std::vector<uint64_t> targets;
    for (int i = 0; i < 2; ++i) {
        targets.push_back(committed(fx, device, 1, texture_desc(3840, 2160, 28, 0x5), 0));
        targets.push_back(committed(fx, device, 1, texture_desc(1920, 1080, 28, 0x5), 0));
    }
    for (uint64_t t : targets) {
        CHECK(t != 0);
    }
    CHECK(g_rt_base != 0);
    const uint64_t used = g_rt_cursor - g_rt_base;
    CHECK(used == 84541440ull);
    CHECK(kRtBudget - used <= 0x1FFFFF);  // the title's "less than one large page left" check
    // Non-render-target textures and buffers keep their previous allocation (not through XMem).
    const uint64_t sampled = committed(fx, device, 1, texture_desc(4096, 4096, 28, 0), 0);
    CHECK(sampled != 0 && g_rt_cursor - g_rt_base == used);
    fx.vcall(sampled, Slot::Release);
    for (uint64_t t : targets) {
        fx.vcall(t, Slot::Release);
    }
    CHECK(g_rt_frees == 4);
    playgta6::os::scarlett::xmem_set_guest_hook_table(0);
    fx.vcall(device, Slot::Release);
}

void test_resources() {
    Fixture fx;
    const uint64_t device = fx.create_device();
    dg::Device* dev = host_device(device);
    int32_t result = -1;
    const size_t mark = fx.captured.size();

    // UPLOAD buffer: guest memory, GPU VA == the mapped CPU address (UMA).
    const uint64_t upload = committed(fx, device, 2, buffer_desc(1000), 0xAC3, &result);  // GENERIC_READ
    CHECK(result == 0 && upload != 0);
    const uint64_t va = fx.vcall(upload, Slot2::GetGPUVirtualAddress);
    const uint64_t mapped = put<uint64_t>(0);
    CHECK(hr(fx.vcall(upload, Slot2::Map, {I(0), I(0), I(mapped)})) == 0);
    CHECK(load64(mapped) == va && va != 0 && va % 65536 == 0);
    CHECK(com::guest_to_host(va, 1000) != nullptr);
    fx.vcall(upload, Slot2::Unmap, {I(0), I(0)});

    // DEFAULT buffer: no CPU access, GPU VA from the device's range, Map fails.
    const uint64_t vb = committed(fx, device, 1, buffer_desc(4096), 0);
    const uint64_t vb_va = fx.vcall(vb, Slot2::GetGPUVirtualAddress);
    CHECK(vb_va >= dg::Device::kFakeGpuVaBase);
    CHECK(hr(fx.vcall(vb, Slot2::Map, {I(0), I(0), I(mapped)})) == dg::kEInvalidArg);
    CHECK(load64(mapped) == 0);

    // Render-target texture: CreateTexture2 with the v2 format code and RENDER_ATTACHMENT usage.
    const uint64_t rt = committed(fx, device, 1, texture_desc(64, 32, 28, 1), 4);  // R8G8B8A8_UNORM, RENDER_TARGET
    CHECK(rt != 0 && fx.vcall(rt, Slot2::GetGPUVirtualAddress) == 0);
    const uint64_t desc_buf = put(dg::GuestResourceDesc{});
    CHECK(fx.sret_call(rt, Slot2::GetDesc, desc_buf) == desc_buf);
    CHECK(get<dg::GuestResourceDesc>(desc_buf).Width == 64 && get<dg::GuestResourceDesc>(desc_buf).Format == 28);
    const uint64_t props = put(dg::GuestHeapProperties{});
    CHECK(hr(fx.vcall(rt, Slot2::GetHeapProperties, {I(props), I(0)})) == 0 && get<dg::GuestHeapProperties>(props).Type == 1);

    // Mipped BC texture (mip count 0 = full chain).
    const uint64_t bc = committed(fx, device, 1, texture_desc(256, 128, 71, 0, 0), 0);  // BC1_UNORM
    CHECK(bc != 0 && object_of<dg::Resource>(bc)->mip_levels() == 9);

    std::vector<Packet> ps = packets_since(fx, mark);
    CHECK(count_op(ps, GpuOpcode::CreateBuffer) == 2);
    const Packet* tex = find_op(ps, GpuOpcode::CreateTexture2);
    CHECK(tex != nullptr);
    if (tex != nullptr) {
        const auto p = payload_as<GpuCreateTexture2Payload>(*tex);
        CHECK(p.width == 64 && p.height == 32 && p.mip_levels == 1 && p.sample_count == 1);
        CHECK(p.format == 22 && (p.usage_flags & GpuTextureUsage::RenderAttachment) != 0);
        CHECK(p.dimension == static_cast<uint8_t>(GpuTextureDimension::Dim2D));
    }
    const Packet* tex2 = find_op(ps, GpuOpcode::CreateTexture2, 1);
    CHECK(tex2 != nullptr && payload_as<GpuCreateTexture2Payload>(*tex2).mip_levels == 9);

    // Heap + placed buffer: VA = heap base + offset; UPLOAD heap -> CPU-visible.
    dg::GuestHeapDesc hd{};
    hd.SizeInBytes = 1 << 20;
    hd.Properties = heap_props(2);
    const uint64_t heap = fx.create(device, Slot2::CreateHeap, {I(put(hd))}, dg::IID_ID3D12Heap, &result);
    CHECK(result == 0 && heap != 0);
    const uint64_t placed = fx.create(device, Slot2::CreatePlacedResource,
                                      {I(heap), I(65536), I(put(buffer_desc(256))), I(0), I(0)},
                                      dg::IID_ID3D12Resource, &result);
    CHECK(result == 0 && fx.vcall(placed, Slot2::GetGPUVirtualAddress) ==
                             object_of<dg::Heap>(heap)->base() + 65536);
    CHECK(hr(fx.vcall(placed, Slot2::Map, {I(0), I(0), I(mapped)})) == 0);

    // Xbox slot 133: a resource at memory the title allocated (GPU VA == that address).
    const uint64_t title_memory = com::guest_alloc(8192);
    const uint64_t at = fx.create(device, Slot2::CreatePlacedResourceX,
                                  {I(title_memory), I(put(buffer_desc(8192))), I(0), I(0)}, dg::IID_ID3D12Resource, &result);
    CHECK(result == 0 && fx.vcall(at, Slot2::GetGPUVirtualAddress) == title_memory);
    CHECK(object_of<dg::Resource>(at)->cpu_address() == title_memory);

    // Xbox textures may carry 0x100 | an XG swizzle mode in the layout field (here 0x102 on a
    // 640x360 R32_UINT UAV texture); other layouts above 3, and swizzle modes on buffers, stay invalid.
    const uint64_t tex_memory = com::guest_alloc(1 << 20);
    dg::GuestResourceDesc swizzled = texture_desc(640, 360, 42, 0x4);
    swizzled.Layout = 0x102;
    const uint64_t sw = fx.create(device, Slot2::CreatePlacedResourceX, {I(tex_memory), I(put(swizzled)), I(0), I(0)},
                                  dg::IID_ID3D12Resource, &result);
    CHECK(result == 0 && sw != 0);
    swizzled.Layout = 0x200;
    CHECK(fx.create(device, Slot2::CreatePlacedResourceX, {I(tex_memory), I(put(swizzled)), I(0), I(0)},
                    dg::IID_ID3D12Resource, &result) == 0);
    CHECK(result == dg::kEInvalidArg && contains(dev->diagnostics(), "layout 512"));
    dg::GuestResourceDesc swizzled_buffer = buffer_desc(4096);
    swizzled_buffer.Layout = 0x102;
    CHECK(fx.create(device, Slot2::CreatePlacedResourceX, {I(tex_memory), I(put(swizzled_buffer)), I(0), I(0)},
                    dg::IID_ID3D12Resource, &result) == 0);
    CHECK(result == dg::kEInvalidArg);
    fx.vcall(sw, Slot::Release);

    // Xbox slot 127 (CreateCommittedResource3 shape, DESC1 + layout + castable formats).
    uint8_t desc1[dg::kGuestResourceDesc1Size] = {};
    const dg::GuestResourceDesc base = buffer_desc(512);
    std::memcpy(desc1, &base, 52);
    const uint64_t c3 = fx.create(device, Slot2::CreateCommittedResource3,
                                  {I(put(heap_props(2))), I(0), I(put_bytes(desc1, sizeof(desc1))), I(0), I(0), I(0),
                                   I(0), I(0)},
                                  dg::IID_ID3D12Resource, &result);
    CHECK(result == 0 && c3 != 0);

    // Reserved resource: VA from the fake range, no heap properties.
    const uint64_t reserved = fx.create(device, Slot2::CreateReservedResource, {I(put(buffer_desc(1 << 16))), I(0), I(0)},
                                        dg::IID_ID3D12Resource, &result);
    CHECK(result == 0 && fx.vcall(reserved, Slot2::GetGPUVirtualAddress) >= dg::Device::kFakeGpuVaBase);
    CHECK(hr(fx.vcall(reserved, Slot2::GetHeapProperties, {I(props), I(0)})) == dg::kEInvalidArg);

    // Malformed descriptions are rejected: unknown dimension, zero width, buffer with a format.
    dg::GuestResourceDesc bad = buffer_desc(64);
    bad.Dimension = 7;
    CHECK(committed(fx, device, 1, bad, 0, &result) == 0 && result == dg::kEInvalidArg);
    bad = buffer_desc(0);
    CHECK(committed(fx, device, 1, bad, 0, &result) == 0 && result == dg::kEInvalidArg);
    bad = buffer_desc(64);
    bad.Format = 28;
    CHECK(committed(fx, device, 1, bad, 0, &result) == 0 && result == dg::kEInvalidArg);

    CHECK(dev->stats().resources == 8);
    const size_t before_release = fx.captured.size();
    for (uint64_t r : {upload, vb, rt, bc, placed, at, c3, reserved}) {
        CHECK(fx.vcall(r, Slot::Release) == 0);
    }
    CHECK(count_op(packets_since(fx, before_release), GpuOpcode::DestroyResource) == 8);
    CHECK(dev->stats().resources == 0);
    fx.vcall(heap, Slot::Release);
    CHECK(fx.vcall(device, Slot::Release) == 0);
    CHECK(com::bridge_stats().live_objects == 0);
}

// Shared scene: device, queue, allocator, root signature, render target + RTV, vertex buffer, PSO.
struct Scene {
    Fixture& fx;
    uint64_t device = 0, queue = 0, allocator = 0, rtv_heap = 0, rt = 0, rs = 0, vb = 0, pso = 0;
    uint64_t rtv = 0, vb_va = 0, vb_cpu = 0;
    std::vector<uint8_t> vs_bytes{'D', 'X', 'B', 'C', 1, 2, 3, 4, 5};
    std::vector<uint8_t> ps_bytes{'D', 'X', 'B', 'C', 9, 8, 7};
    static constexpr uint32_t kVsModule = 9001, kPsModule = 9002;

    explicit Scene(Fixture& f, bool register_shaders = true, bool name_back_buffer = true) : fx(f) {
        device = fx.create_device();
        dg::Device* dev = host_device(device);
        queue = fx.create(device, Slot::CreateCommandQueue, {I(put(dg::GuestCommandQueueDesc{}))}, dg::IID_ID3D12CommandQueue);
        allocator = fx.create(device, Slot::CreateCommandAllocator, {I(0)}, dg::IID_ID3D12CommandAllocator);
        rtv_heap = fx.create(device, Slot::CreateDescriptorHeap, {I(put(dg::GuestDescriptorHeapDesc{2, 4, 0, 0}))},
                             dg::IID_ID3D12DescriptorHeap);
        const uint64_t start = put<uint64_t>(0);
        fx.sret_call(rtv_heap, Slot::GetCPUStart, start);
        rtv = load64(start);
        rt = committed(fx, device, 1, texture_desc(64, 64, 87, 1), 4);  // B8G8R8A8_UNORM
        fx.vcall(device, Slot2::CreateRenderTargetView, {I(rt), I(0), I(rtv)});
        if (name_back_buffer) {
            CHECK(dev->set_back_buffer(rt));
        }

        // Empty root signature (ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT) through the serialize export.
        dg::GuestVersionedRootSignatureDesc rsd{};
        rsd.Version = 2;
        rsd.Flags = 1;
        const uint64_t blob_out = put<uint64_t>(0);
        CHECK(hr(fx.dcall(dx::export_address("D3D12SerializeVersionedRootSignature"), {I(put(rsd)), I(blob_out), I(0)})) == 0);
        const uint64_t blob = load64(blob_out);
        rs = fx.create(device, Slot::CreateRootSignature,
                       {I(0), I(fx.vcall(blob, Slot::GetBufferPointer)), I(fx.vcall(blob, Slot::GetBufferSize))},
                       dg::IID_ID3D12RootSignature);
        fx.vcall(blob, Slot::Release);

        // Upload-heap vertex buffer (3 x float3) written through the mapped pointer.
        vb = committed(fx, device, 2, buffer_desc(36), 0xAC3);
        vb_va = fx.vcall(vb, Slot2::GetGPUVirtualAddress);
        const uint64_t mapped = put<uint64_t>(0);
        fx.vcall(vb, Slot2::Map, {I(0), I(0), I(mapped)});
        vb_cpu = load64(mapped);
        const float tri[9] = {0, 0.5f, 0, -0.5f, -0.5f, 0, 0.5f, -0.5f, 0};
        CHECK(com::write_guest(vb_cpu, tri, sizeof(tri)));
        fx.vcall(vb, Slot2::Unmap, {I(0), I(0)});

        if (register_shaders) {
            dev->register_shader_module(dg::shader_identity(vs_bytes.data(), vs_bytes.size()), kVsModule);
            dev->register_shader_module(dg::shader_identity(ps_bytes.data(), ps_bytes.size()), kPsModule);
        }
        pso = create_pso();
    }

    uint64_t create_pso(int32_t* result = nullptr) {
        dg::GuestGraphicsPipelineStateDesc d{};
        d.pRootSignature = rs;
        d.VS = {put_bytes(vs_bytes.data(), vs_bytes.size()), vs_bytes.size()};
        d.PS = {put_bytes(ps_bytes.data(), ps_bytes.size()), ps_bytes.size()};
        for (auto& t : d.BlendState.RenderTarget) {
            t.SrcBlend = t.SrcBlendAlpha = 2;  // ONE
            t.DestBlend = t.DestBlendAlpha = 1;  // ZERO
            t.BlendOp = t.BlendOpAlpha = 1;  // ADD
            t.LogicOp = 4;  // NOOP
            t.RenderTargetWriteMask = 0xF;
        }
        d.SampleMask = 0xFFFFFFFFu;
        d.RasterizerState.FillMode = 3;
        d.RasterizerState.CullMode = 1;
        d.RasterizerState.DepthClipEnable = 1;
        d.DepthStencilState.DepthFunc = 2;
        d.DepthStencilState.FrontFace = d.DepthStencilState.BackFace = {1, 1, 1, 8};
        static const char kPosition[] = "POSITION";
        dg::GuestInputElementDesc e{};
        e.SemanticName = put_bytes(kPosition, sizeof(kPosition));
        e.Format = 6;  // R32G32B32_FLOAT
        d.InputLayout = {put(e), 1};
        d.PrimitiveTopologyType = 3;  // TRIANGLE
        d.NumRenderTargets = 1;
        d.RTVFormats[0] = 87;
        d.SampleCount = 1;
        return fx.create(device, Slot2::CreateGraphicsPipelineState, {I(put(d))}, dg::IID_ID3D12PipelineState, result);
    }

    // A list created through the Xbox slot 73 and closed right away, as the executable does.
    uint64_t new_list() {
        int32_t result = -1;
        const uint64_t list = fx.create(device, Slot2::CreateCommandListX,
                                        {I(put(dg::GuestCommandListDescX{0, {6, 1, 0}})), I(allocator), I(0)},
                                        dg::IID_ID3D12GraphicsCommandList, &result);
        CHECK(result == 0 && list != 0);
        // The executable asks slot 73 for the Xbox graphics-command-list IID; it is an alias.
        const uint64_t xbox_iid = put(com::make_guid("c3827890-4024-474c-98cf-645b51dc4874"));
        const uint64_t alias_out = put(uint64_t{0});
        CHECK(hr(fx.vcall(list, 0, {I(xbox_iid), I(alias_out)})) == 0);
        CHECK(load64(alias_out) == list);
        fx.vcall(list, Slot::Release);
        CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
        // Xbox slot 139 (allocator, null, 0) reopens the list like Reset; reopening an open list
        // fails like Reset does.
        CHECK(hr(fx.vcall(list, 139, {I(allocator), I(0), I(0)})) == 0);
        CHECK(hr(fx.vcall(list, 139, {I(allocator), I(0), I(0)})) < 0);
        CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
        return list;
    }

    // Records one frame on `list` (from Reset to Close) on guest thread `t`.
    void record(GuestThread& t, uint64_t list, uint32_t draws = 1) {
        CHECK(hr(fx.vcall_on(t, list, Slot2::Reset, {I(allocator), I(pso)})) == 0);
        fx.vcall_on(t, list, Slot2::SetGraphicsRootSignature, {I(rs)});
        fx.vcall_on(t, list, Slot2::OMSetRenderTargets, {I(1), I(put(rtv)), I(0), I(0)});
        const float color[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        fx.vcall_on(t, list, Slot2::ClearRenderTargetView, {I(rtv), I(put_bytes(color, sizeof(color))), I(0), I(0)});
        fx.vcall_on(t, list, Slot2::RSSetViewports, {I(1), I(put(dg::GuestViewport{0, 0, 32, 32, 0, 1}))});
        fx.vcall_on(t, list, Slot2::RSSetScissorRects, {I(1), I(put(dg::GuestRect{0, 0, 32, 32}))});
        fx.vcall_on(t, list, Slot2::IASetPrimitiveTopology, {I(4)});  // TRIANGLELIST
        fx.vcall_on(t, list, Slot2::IASetVertexBuffers, {I(0), I(1), I(put(dg::GuestVertexBufferView{vb_va, 36, 12}))});
        for (uint32_t i = 0; i < draws; ++i) {
            fx.vcall_on(t, list, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
        }
        CHECK(hr(fx.vcall_on(t, list, Slot2::Close)) == 0);
    }

    void execute(std::initializer_list<uint64_t> lists) {
        std::vector<uint64_t> v(lists);
        fx.vcall(queue, Slot2::ExecuteCommandLists, {I(v.size()), I(put_bytes(v.data(), 8 * v.size())), I(0), I(3)});
    }

    void release_all(std::initializer_list<uint64_t> extra = {}) {
        for (uint64_t o : extra) {
            fx.vcall(o, Slot::Release);
        }
        for (uint64_t o : {pso, vb, rs, rt, rtv_heap, allocator, queue}) {
            fx.vcall(o, Slot::Release);
        }
        CHECK(fx.vcall(device, Slot::Release) == 0);
        CHECK(com::bridge_stats().live_objects == 0);
    }
};

void test_frame_end_to_end() {
    Fixture fx;
    Scene scene(fx);
    dg::Device* dev = host_device(scene.device);
    const uint64_t list = scene.new_list();
    CHECK(object_of<dg::PipelineState>(scene.pso)->resolved());

    const size_t mark = fx.captured.size();
    scene.record(fx.thread, list);
    CHECK(fx.captured.size() == mark);  // recording submits nothing
    scene.execute({list});
    CHECK(hr(fx.vcall(scene.queue, Slot::PresentX, {I(1), I(put(dg::xbox::GuestPresentPlaneX{})), I(0)})) == 0);

    const std::vector<Packet> ps = packets_since(fx, mark);
    // The vertex data written through Map reaches the GPU before the list's packets.
    const size_t write = index_of(ps, GpuOpcode::WriteBuffer);
    const size_t pass = index_of(ps, GpuOpcode::BeginRenderPass);
    CHECK(write != SIZE_MAX && pass != SIZE_MAX && write < pass);
    if (write != SIZE_MAX) {
        const auto w = payload_as<GpuWriteBufferPayload>(ps[write]);
        CHECK(w.id == object_of<dg::Resource>(scene.vb)->id() && w.dst_offset == 0 && w.data_len == 36);
        float first = 1;
        std::memcpy(&first, ps[write].payload.data() + sizeof(GpuWriteBufferPayload) + 4, 4);
        CHECK(first == 0.5f);
    }
    const Packet* pipe = find_op(ps, GpuOpcode::CreateRenderPipeline2);
    CHECK(pipe != nullptr);
    if (pipe != nullptr) {
        const auto p = payload_as<GpuCreateRenderPipeline2Payload>(*pipe);
        CHECK(p.vs_module_id == Scene::kVsModule && p.fs_module_id == Scene::kPsModule);
    }
    const Packet* begin = find_op(ps, GpuOpcode::BeginRenderPass);
    if (begin != nullptr) {
        const auto b = payload_as<GpuBeginRenderPassPayload>(*begin);
        CHECK(b.color_count == 1 && b.color[0].texture_id == kGpuSwapchainTextureId);
        CHECK(b.color[0].load_op == static_cast<uint8_t>(playgta6::gpu::pso::WgpuLoadOp::Clear) && b.color[0].clear_value[2] == 0.75f);
    }
    CHECK(count_op(ps, GpuOpcode::SetViewport) == 1 && count_op(ps, GpuOpcode::SetScissorRect) == 1);
    const Packet* vbuf = find_op(ps, GpuOpcode::SetVertexBuffer);
    CHECK(vbuf != nullptr && payload_as<GpuSetVertexBufferPayload>(*vbuf).buffer_id == object_of<dg::Resource>(scene.vb)->id());
    const Packet* draw = find_op(ps, GpuOpcode::Draw);
    CHECK(draw != nullptr && payload_as<GpuDrawPayload>(*draw).vertex_count == 3);
    CHECK(count_op(ps, GpuOpcode::EndRenderPass) == 1);
    CHECK(ps.back().opcode == GpuOpcode::Present);
    CHECK(index_of(ps, GpuOpcode::Draw) < index_of(ps, GpuOpcode::EndRenderPass));

    // Second frame: unchanged memory is not uploaded again; one changed float uploads its page.
    const size_t mark2 = fx.captured.size();
    scene.record(fx.thread, list);
    scene.execute({list});
    CHECK(count_op(packets_since(fx, mark2), GpuOpcode::WriteBuffer) == 0);
    const float moved = 0.9f;
    CHECK(com::write_guest(scene.vb_cpu + 4, &moved, 4));
    const size_t mark3 = fx.captured.size();
    scene.execute({list});
    const auto ps3 = packets_since(fx, mark3);
    CHECK(count_op(ps3, GpuOpcode::WriteBuffer) == 1);
    CHECK(count_op(ps3, GpuOpcode::Draw) == 1);  // a closed list can be executed again
    CHECK(dev->stats().upload_bytes == 72);

    // Executing an open list is refused; calls on a closed list are ignored and diagnosed.
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(0)})) == 0);
    const size_t mark4 = fx.captured.size();
    scene.execute({list});
    CHECK(count_op(packets_since(fx, mark4), GpuOpcode::Draw) == 0);
    CHECK(contains(dev->diagnostics(), "is not closed"));
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    fx.vcall(list, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    CHECK(contains(dev->diagnostics(), "call on a closed command list"));
    CHECK(com::stub_hits(dg::kID3D12GraphicsCommandList, 130) == 0);
    CHECK(hr(fx.vcall(list, 130, {I(0x700000), I(0), I(0)})) == dg::kENotImpl);  // unestablished slot: logging stub
    CHECK(com::stub_hits(dg::kID3D12GraphicsCommandList, 130) == 1);

    // Xbox slots 71 / 74 write immediate values to guest memory when the list executes, in
    // recording order; slot 146 (enhanced barriers) records nothing and checks the resources.
    const uint64_t markers = com::guest_alloc(64);
    CHECK(com::write_guest(markers, uint64_t{0}) && com::write_guest(markers + 8, uint64_t{0}));
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(0)})) == 0);
    fx.vcall(list, 71, {I(markers + 4), I(0x1234), I(0), I(0)});
    fx.vcall(list, 74, {I(markers + 8), I(7), I(0x1122334455667788ull), I(1)});
    fx.vcall(list, 74, {I(markers + 8), I(7), I(42), I(1)});  // the later write wins
    fx.vcall(list, 71, {I(markers + 2), I(1), I(0), I(0)});  // misaligned: diagnosed, dropped
    CHECK(contains(dev->diagnostics(), "slot 71: invalid address"));
    struct BufferBarrier {
        uint32_t sync_before, sync_after, access_before, access_after;
        uint64_t resource, offset, size;
    };
    static_assert(sizeof(BufferBarrier) == 40);
    const BufferBarrier bb[2] = {{0, 0, 0, 0, scene.vb, 0, ~0ull}, {0, 0, 0, 0, 0x7000'0000ull, 0, ~0ull}};
    const uint64_t bb_va = put_bytes(bb, sizeof(bb));
    struct BarrierGroup {
        uint32_t type, count;
        uint64_t barriers;
    };
    const BarrierGroup group{2, 2, bb_va};
    const size_t diags_before = dev->diagnostics().size();
    fx.vcall(list, 146, {I(1), I(put_bytes(&group, sizeof(group)))});
    CHECK(dev->diagnostics().size() == diags_before + 1);  // only the unknown resource
    CHECK(contains(dev->diagnostics(), "names an unknown resource"));
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    CHECK(load64(markers) == 0 && load64(markers + 8) == 0);  // nothing before execution
    const size_t mark5 = fx.captured.size();
    scene.execute({list});
    CHECK(count_op(packets_since(fx, mark5), GpuOpcode::Draw) == 0);
    uint32_t marker = 0;
    CHECK(com::read_guest(markers + 4, marker) && marker == 0x1234);
    CHECK(load64(markers + 8) == 42);
    CHECK(com::stub_hits(dg::kID3D12GraphicsCommandList, 71) == 0 &&
          com::stub_hits(dg::kID3D12GraphicsCommandList, 74) == 0 &&
          com::stub_hits(dg::kID3D12GraphicsCommandList, 146) == 0);
    CHECK(t_balanced);
    scene.release_all({list});
}

void test_unresolved_shaders_and_indexed_draws() {
    Fixture fx;
    Scene scene(fx, /*register_shaders=*/false);
    dg::Device* dev = host_device(scene.device);
    CHECK(!object_of<dg::PipelineState>(scene.pso)->resolved());
    CHECK(dev->unresolved_shaders().size() == 2);
    CHECK((dev->unresolved_shader_stages() == std::array<uint32_t, 3>{1, 1, 0}));  // one vertex, one pixel
    const uint64_t list = scene.new_list();
    const size_t mark = fx.captured.size();
    scene.record(fx.thread, list, 2);
    scene.execute({list});
    const auto ps = packets_since(fx, mark);
    CHECK(count_op(ps, GpuOpcode::Draw) == 0);
    CHECK(count_op(ps, GpuOpcode::CreateRenderPipeline2) == 0);  // never a pipeline with module 0
    CHECK(dev->stats().skipped_draws == 2);
    // The first skipped draw of the pipeline state is described once: shader sizes, layout,
    // targets, root signature and what is bound (counts, formats, sizes).
    const std::vector<std::string> reports = dev->skipped_draw_reports();
    CHECK(reports.size() == 1);
    if (!reports.empty()) {
        const std::string& r = reports[0];
        CHECK(r.find("skipped DrawInstanced") != std::string::npos);
        CHECK(r.find("VS " + std::to_string(scene.vs_bytes.size()) + " B DXIL kind -1 (no module)") != std::string::npos);
        CHECK(r.find("PS " + std::to_string(scene.ps_bytes.size()) + " B DXIL kind -1 (no module)") != std::string::npos);
        CHECK(r.find("input layout") != std::string::npos);
        CHECK(r.find("render targets 1") != std::string::npos);
        CHECK(r.find("root signature 1.") != std::string::npos);
        CHECK(r.find("vertex buffers") != std::string::npos);
        CHECK(r.find("bound render targets") != std::string::npos);
    }
    // A draw the state bridge drops (no graphics pipeline state bound) is described too, once.
    const uint64_t bare = scene.new_list();
    CHECK(hr(fx.vcall(bare, Slot2::Reset, {I(scene.allocator), I(0)})) == 0);
    fx.vcall(bare, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    fx.vcall(bare, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    CHECK(hr(fx.vcall(bare, Slot2::Close)) == 0);
    const std::vector<std::string> reports2 = dev->skipped_draw_reports();
    CHECK(reports2.size() == 2);
    if (reports2.size() == 2) {
        CHECK(reports2[1].find("no graphics pipeline state bound") != std::string::npos);
    }
    fx.vcall(bare, Slot::Release);

    // Registering the modules later resolves PSOs created afterwards.
    dev->register_shader_module(dg::shader_identity(scene.vs_bytes.data(), scene.vs_bytes.size()), Scene::kVsModule);
    dev->register_shader_module(dg::shader_identity(scene.ps_bytes.data(), scene.ps_bytes.size()), Scene::kPsModule);
    CHECK(dev->unresolved_shaders().empty());
    const uint64_t pso2 = scene.create_pso();
    CHECK(object_of<dg::PipelineState>(pso2)->resolved());

    // Indexed draw with a 16-bit index buffer in the upload heap.
    const uint64_t ib = committed(fx, scene.device, 2, buffer_desc(6), 0xAC3);
    const uint16_t idx[3] = {0, 1, 2};
    CHECK(com::write_guest(object_of<dg::Resource>(ib)->cpu_address(), idx, sizeof(idx)));
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(pso2)})) == 0);
    fx.vcall(list, Slot2::OMSetRenderTargets, {I(1), I(put(scene.rtv)), I(1), I(0)});
    fx.vcall(list, Slot2::IASetPrimitiveTopology, {I(4)});
    fx.vcall(list, Slot2::IASetVertexBuffers, {I(0), I(1), I(put(dg::GuestVertexBufferView{scene.vb_va, 36, 12}))});
    fx.vcall(list, Slot2::IASetIndexBuffer,
             {I(put(dg::GuestIndexBufferView{fx.vcall(ib, Slot2::GetGPUVirtualAddress), 6, 57}))});
    fx.vcall(list, Slot2::DrawIndexedInstanced, {I(3), I(1), I(0), I(0), I(0)});
    // An unknown GPU VA is diagnosed (null binding), not dereferenced.
    fx.vcall(list, Slot2::IASetVertexBuffers, {I(1), I(1), I(put(dg::GuestVertexBufferView{0x123456000ULL, 36, 12}))});
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    CHECK(contains(dev->diagnostics(), "IASetVertexBuffers: unknown GPU virtual address"));
    const size_t mark2 = fx.captured.size();
    scene.execute({list});
    const auto ps2 = packets_since(fx, mark2);
    const Packet* di = find_op(ps2, GpuOpcode::DrawIndexed);
    CHECK(di != nullptr && payload_as<GpuDrawIndexedPayload>(*di).index_count == 3);
    const Packet* ibp = find_op(ps2, GpuOpcode::SetIndexBuffer);
    CHECK(ibp != nullptr && payload_as<GpuSetIndexBufferPayload>(*ibp).buffer_id == object_of<dg::Resource>(ib)->id());
    scene.release_all({list, pso2, ib});
}

void test_copies_and_barriers() {
    Fixture fx;
    Scene scene(fx);
    dg::Device* dev = host_device(scene.device);
    const uint64_t list = scene.new_list();
    const uint64_t upload = committed(fx, scene.device, 2, buffer_desc(256 * 4 + 512), 0xAC3);
    const uint64_t gpu_buf = committed(fx, scene.device, 1, buffer_desc(64), 0x400);  // COPY_DEST
    const uint64_t tex = committed(fx, scene.device, 1, texture_desc(16, 4, 28, 0), 0x400);
    const uint64_t up_cpu = object_of<dg::Resource>(upload)->cpu_address();

    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(0)})) == 0);
    // Buffer copy from the upload heap: the source bytes are read at ExecuteCommandLists.
    fx.vcall(list, Slot2::CopyBufferRegion, {I(gpu_buf), I(8), I(upload), I(16), I(32)});
    // Texture upload from a placed footprint (16x4 RGBA8, 256-byte row pitch, offset 512).
    dg::GuestTextureCopyLocation dst{};
    dst.pResource = tex;
    dst.Type = 0;  // SubresourceIndex 0
    dg::GuestTextureCopyLocation src{};
    src.pResource = upload;
    src.Type = 1;
    src.PlacedFootprint = {512, {28, 16, 4, 1, 256}};
    fx.vcall(list, Slot2::CopyTextureRegion, {I(put(dst)), I(0), I(0), I(0), I(put(src)), I(0)});
    // GPU -> GPU copy: no packet, diagnosed.
    fx.vcall(list, Slot2::CopyBufferRegion, {I(gpu_buf), I(0), I(gpu_buf), I(0), I(8)});
    // Barriers: COPY_DEST -> PIXEL_SHADER_RESOURCE; an unknown resource is diagnosed.
    dg::GuestResourceBarrier barriers[2] = {};
    barriers[0].Type = 0;
    barriers[0].pResource = tex;
    barriers[0].u0 = 0xFFFFFFFFu;
    barriers[0].u1 = 0x400;
    barriers[0].StateAfter = 0x80;
    barriers[1].Type = 0;
    barriers[1].pResource = 0x1234560;
    fx.vcall(list, Slot2::ResourceBarrier, {I(2), I(put_bytes(barriers, sizeof(barriers)))});
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    CHECK(dev->stats().unsupported_copies == 1);
    CHECK(contains(dev->diagnostics(), "transition of an unknown resource"));

    // Fill the source after recording: the GPU must see these bytes.
    std::vector<uint8_t> bytes(256 * 4 + 512);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(i * 7);
    }
    CHECK(com::write_guest(up_cpu, bytes.data(), bytes.size()));
    const size_t mark = fx.captured.size();
    scene.execute({list});
    const auto ps = packets_since(fx, mark);
    const Packet* wt = find_op(ps, GpuOpcode::WriteTexture);
    CHECK(wt != nullptr);
    if (wt != nullptr) {
        const auto t = payload_as<GpuWriteTexturePayload>(*wt);
        CHECK(t.texture_id == object_of<dg::Resource>(tex)->id() && t.width == 16 && t.height == 4 &&
              t.bytes_per_row == 256 && t.data_len == 256 * 3 + 64);
        CHECK(wt->payload[sizeof(GpuWriteTexturePayload)] == bytes[512]);
    }
    bool copied = false;
    for (const Packet& p : ps) {
        if (p.opcode == GpuOpcode::WriteBuffer) {
            const auto w = payload_as<GpuWriteBufferPayload>(p);
            if (w.id == object_of<dg::Resource>(gpu_buf)->id()) {
                copied = w.dst_offset == 8 && w.data_len == 32 && p.payload[sizeof(GpuWriteBufferPayload)] == bytes[16];
            }
        }
    }
    CHECK(copied);
    // The barrier reached the device-level state registry at execute.
    CHECK(dev->resource_states().state(object_of<dg::Resource>(tex)->id(), 0) == 0x80);
    scene.release_all({list, upload, gpu_buf, tex});
}

void test_concurrent_recording() {
    Fixture fx;
    Scene scene(fx);
    const uint64_t list_a = scene.new_list();
    const uint64_t list_b = scene.new_list();
    GuestThread other{};
    CHECK(create_guest_thread(fx.arena, 512 * 1024, other));
    std::thread worker([&] {
        t_balanced = true;
        scene.record(other, list_b, 20);
        CHECK(t_balanced);
    });
    scene.record(fx.thread, list_a, 10);
    worker.join();
    const size_t mark = fx.captured.size();
    scene.execute({list_a, list_b});
    const auto ps = packets_since(fx, mark);
    CHECK(count_op(ps, GpuOpcode::Draw) == 30);
    CHECK(count_op(ps, GpuOpcode::BeginRenderPass) == 2);
    destroy_guest_thread(fx.arena, other);
    scene.release_all({list_a, list_b});
}


void test_compute_views_and_public_list() {
    Fixture fx;
    Scene scene(fx);
    dg::Device* dev = host_device(scene.device);
    int32_t result = -1;

    // Public CreateCommandList (slot 12): created open, closes.
    const uint64_t list = fx.create(scene.device, Slot2::CreateCommandList, {I(0), I(0), I(scene.allocator), I(0)},
                                    dg::IID_ID3D12GraphicsCommandList, &result);
    CHECK(result == 0 && list != 0 && object_of<dg::CommandList>(list)->is_open());
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    CHECK(fx.create(scene.device, Slot2::CreateCommandList, {I(0), I(0), I(0), I(0)}, dg::IID_ID3D12GraphicsCommandList,
                    &result) == 0 &&
          result == dg::kEInvalidArg);  // no allocator

    // Compute PSO with a registered module: Dispatch reaches the stream.
    const std::vector<uint8_t> cs{'D', 'X', 'B', 'C', 0x42};
    dev->register_shader_module(dg::shader_identity(cs.data(), cs.size()), 9100);
    dg::GuestComputePipelineStateDesc cd{};
    cd.pRootSignature = scene.rs;
    cd.CS = {put_bytes(cs.data(), cs.size()), cs.size()};
    const uint64_t cpso = fx.create(scene.device, Slot2::CreateComputePipelineState, {I(put(cd))},
                                    dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && object_of<dg::PipelineState>(cpso)->is_compute() && object_of<dg::PipelineState>(cpso)->resolved());
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(cpso)})) == 0);
    fx.vcall(list, 33, {I(scene.rs)});  // SetComputeRootSignature (Xbox slot)
    fx.vcall(list, 14, {I(4), I(2), I(1)});  // Dispatch
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    const size_t mark = fx.captured.size();
    scene.execute({list});
    const auto ps = packets_since(fx, mark);
    const Packet* d = find_op(ps, GpuOpcode::DispatchCompute);
    CHECK(d != nullptr && payload_as<GpuDispatchComputePayload>(*d).workgroup_x == 4);
    CHECK(find_op(ps, GpuOpcode::CreateComputePipeline2) != nullptr);

    // The device records each shader's size and the kind its DXIL container declares (a
    // synthetic container: "DXBC", digest, version, size, one part "DXIL" whose program header
    // says vertex shader, kind 1).
    std::vector<uint8_t> dxbc(48, 0);
    auto put32 = [&](size_t at, uint32_t v) { std::memcpy(dxbc.data() + at, &v, 4); };
    put32(0, 0x43425844u);    // "DXBC"
    put32(24, 48);            // total size
    put32(28, 1);             // one part
    put32(32, 36);            // at offset 36
    put32(36, 0x4C495844u);   // "DXIL"
    put32(40, 8);             // part size
    put32(44, (1u << 16) | 0x60);  // program version: vertex shader 6.0
    dg::GuestComputePipelineStateDesc vd{};
    vd.pRootSignature = scene.rs;
    vd.CS = {put_bytes(dxbc.data(), dxbc.size()), dxbc.size()};
    const uint64_t vpso = fx.create(scene.device, Slot2::CreateComputePipelineState, {I(put(vd))},
                                    dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && vpso != 0);
    const uint64_t vid = dg::shader_identity(dxbc.data(), dxbc.size());
    CHECK(dev->shader_size(vid) == dxbc.size());
    CHECK(dev->shader_kind(vid) == 1);
    CHECK(dev->shader_kind(dg::shader_identity(cs.data(), cs.size())) == -1);  // not a container
    fx.vcall(vpso, Slot::Release);

    // A PSO without a vertex shader or with an invalid root signature is rejected.
    dg::GuestGraphicsPipelineStateDesc empty{};
    CHECK(fx.create(scene.device, Slot2::CreateGraphicsPipelineState, {I(put(empty))}, dg::IID_ID3D12PipelineState,
                    &result) == 0 &&
          result == dg::kEInvalidArg);
    cd.pRootSignature = scene.allocator;  // not a root signature
    CHECK(fx.create(scene.device, Slot2::CreateComputePipelineState, {I(put(cd))}, dg::IID_ID3D12PipelineState, &result) ==
              0 &&
          result == dg::kEInvalidArg);

    // Shader resource view of a texture into a shader-visible heap (G6 record).
    const uint64_t srv_heap = fx.create(scene.device, Slot::CreateDescriptorHeap,
                                        {I(put(dg::GuestDescriptorHeapDesc{0, 8, 1, 0}))}, dg::IID_ID3D12DescriptorHeap);
    const uint64_t start = put<uint64_t>(0);
    fx.sret_call(srv_heap, Slot::GetCPUStart, start);
    const uint64_t tex = committed(fx, scene.device, 1, texture_desc(32, 32, 28, 0, 6), 0x80);
    dg::GuestShaderResourceViewDesc srv{};
    srv.Format = 28;
    srv.ViewDimension = 4;  // TEXTURE2D
    srv.Shader4ComponentMapping = 0x1688;
    srv.u[0] = 1;            // MostDetailedMip
    srv.u[1] = 0xFFFFFFFFu;  // MipLevels: all
    fx.vcall(scene.device, Slot2::CreateShaderResourceView, {I(tex), I(put(srv)), I(load64(start) + 32)});
    const d3d::ViewDescriptor* v =
        host_object<dg::DescriptorHeap, dg::IDescriptorHeapHost>(srv_heap)->heap()->view(1);
    CHECK(v != nullptr && v->kind == d3d::DescriptorKind::Srv && v->resource_id == object_of<dg::Resource>(tex)->id() &&
          v->first_mip == 1 && v->mip_levels == 0xffff);
    srv.ViewDimension = 42;
    const uint64_t bad_before = dev->stats().bad_arguments;
    fx.vcall(scene.device, Slot2::CreateShaderResourceView, {I(tex), I(put(srv)), I(load64(start) + 64)});
    CHECK(dev->stats().bad_arguments == bad_before + 1);

    // Depth clear: the float depth arrives in xmm3 (Win64 position 3), the stencil byte on the stack.
    const uint64_t dsv_heap = fx.create(scene.device, Slot::CreateDescriptorHeap,
                                        {I(put(dg::GuestDescriptorHeapDesc{3, 1, 0, 0}))}, dg::IID_ID3D12DescriptorHeap);
    fx.sret_call(dsv_heap, Slot::GetCPUStart, start);
    const uint64_t dsv = load64(start);
    const uint64_t depth = committed(fx, scene.device, 1, texture_desc(64, 64, 40, 2), 0x10);  // D32_FLOAT
    fx.vcall(scene.device, 21, {I(depth), I(0), I(dsv)});  // CreateDepthStencilView
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(0)})) == 0);
    fx.vcall(list, Slot2::ClearDepthStencilView, {I(dsv), I(1), F32(0.25f), I(0x7F), I(0), I(0)});
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    const size_t mark2 = fx.captured.size();
    scene.execute({list});
    const auto ps2 = packets_since(fx, mark2);
    const Packet* pass = find_op(ps2, GpuOpcode::BeginRenderPass);
    CHECK(pass != nullptr);
    if (pass != nullptr) {
        const auto b = payload_as<GpuBeginRenderPassPayload>(*pass);        CHECK(b.depth_stencil.texture_id == object_of<dg::Resource>(depth)->id() &&
              b.depth_stencil.depth_clear_value == 0.25f);
    }

    scene.release_all({list, cpso, srv_heap, tex, dsv_heap, depth});
}

// ---------------------------------------------------------------------------
// G27: Xbox-specific device, queue and command-list slots (d3d12_xs_xbox.h)
// ---------------------------------------------------------------------------

namespace XSlot {
// ID3D12Device (Xbox)
constexpr uint32_t FeatureToggle = 49, RemovalCallbacks = 50, CreatePsoWithOverrides = 55, CreateNativePso = 57,
                   Slot61 = 61, PsoFromBlob = 62, PsoFromBlobCached = 67, SrvAtAddress = 70, UavAtAddress = 71,
                   PagePool = 74, SetFrameInterval = 78, ScheduleFrameEvent = 79, WaitFrameEvent = 80,
                   FrameStatistics = 81, CreatePsoStream = 86, CreateStateObject = 101, AllocationInfo1 = 108,
                   Slot119 = 119, Slot132 = 132;
// ID3D12CommandQueue (Xbox)
constexpr uint32_t Suspend = 23, Resume = 24, PageMappings = 31;
// ID3D12CommandAllocator (Xbox)
constexpr uint32_t AllocationCallback = 9;
// ID3D12GraphicsCommandList (Xbox)
constexpr uint32_t ExecuteIndirect = 61, OMSetDepthBounds = 70, RSSetShadingRate = 126, RSSetShadingRateImage = 127,
                   RegisterWrite = 131, SetDescriptorHeaps = 32, BeginQuery = 28;
}  // namespace XSlot

uint64_t counter(dg::Device* dev, dg::xbox::Counter c) {
    return dev->counter(c);
}

// Copies a PSO's 32-byte payload into the list and calls SetPipelineState with only `this` (rdx
// holds garbage), as the executable's inlined code does.
void staged_set_pipeline_state(Fixture& fx, GuestThread& t, uint64_t list, uint64_t pso) {
    uint8_t payload[32];
    CHECK(com::read_guest(pso + 0x10, payload, sizeof(payload)));
    CHECK(com::write_guest(list + dg::xbox::kListStagedPipelineOffset, payload, sizeof(payload)));
    fx.vcall_on(t, list, 25, {I(0x00000000DEAD0000ULL)});
}

void test_xbox_inlined_list_fields_and_staged_pso() {
    Fixture fx;
    Scene scene(fx);
    dg::Device* dev = host_device(scene.device);
    const uint64_t list = scene.new_list();

    // The list's guest payload is zero: the inlined fast paths' cursors never admit a write.
    for (uint64_t off : {0x18u, 0x20u, 0x28u, 0x30u, 0x544u, 0x548u, 0x554u}) {
        CHECK(load64(list + off) == 0);
    }
    // The PSO payload is the tag and the PSO's own interface pointer.
    CHECK(load64(scene.pso + 0x10) == dg::xbox::kPipelineStateTag && load64(scene.pso + 0x18) == scene.pso);

    // Record with the staged protocol only (no initial PSO on Reset).
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(0)})) == 0);
    fx.vcall(list, Slot2::SetGraphicsRootSignature, {I(scene.rs)});
    staged_set_pipeline_state(fx, fx.thread, list, scene.pso);
    CHECK(load64(list + dg::xbox::kListStagedPipelineOffset) == 0);  // consumed
    fx.vcall(list, Slot2::OMSetRenderTargets, {I(1), I(put(scene.rtv)), I(0), I(0)});
    fx.vcall(list, Slot2::RSSetViewports, {I(1), I(put(dg::GuestViewport{0, 0, 32, 32, 0, 1}))});
    fx.vcall(list, Slot2::RSSetScissorRects, {I(1), I(put(dg::GuestRect{0, 0, 32, 32}))});
    fx.vcall(list, Slot2::IASetPrimitiveTopology, {I(4)});
    fx.vcall(list, Slot2::IASetVertexBuffers, {I(0), I(1), I(put(dg::GuestVertexBufferView{scene.vb_va, 36, 12}))});
    fx.vcall(list, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    // Without a staged copy, slot 25 takes the PSO from rdx (public convention).
    fx.vcall(list, 25, {I(scene.pso)});
    fx.vcall(list, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    // Neither: diagnosed, nothing bound.
    const uint64_t bad_before = dev->stats().bad_arguments;
    fx.vcall(list, 25, {I(0x00000000BEEF0000ULL)});
    CHECK(dev->stats().bad_arguments > bad_before);
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    const size_t mark = fx.captured.size();
    scene.execute({list});
    const auto ps = packets_since(fx, mark);
    CHECK(count_op(ps, GpuOpcode::Draw) == 2);
    CHECK(find_op(ps, GpuOpcode::CreateRenderPipeline2) != nullptr);
    CHECK(dev->stats().skipped_draws == 0);
    CHECK(t_balanced);
    scene.release_all({list});
}

void test_xbox_frame_loop_present_and_pacing() {
    Fixture fx;
    Scene scene(fx, /*register_shaders=*/true, /*name_back_buffer=*/false);
    dg::Device* dev = host_device(scene.device);
    const uint64_t list = scene.new_list();
    const uint64_t rt_id = object_of<dg::Resource>(scene.rt)->id();

    // 5 ms frame events (output 0, interval, period, flags), as slot 78 is called.
    CHECK(hr(fx.vcall(scene.device, XSlot::SetFrameInterval, {I(0), I(5000), I(2), I(0)})) == 0);
    CHECK(dev->frame_pacer().interval_us() == 5000 && dev->frame_pacer().period() == 2);
    CHECK(hr(fx.vcall(scene.device, XSlot::ScheduleFrameEvent, {I(2), I(0), I(0), I(0)})) == 0);

    const uint64_t token_slot = put<uint64_t>(0);
    const uint64_t present_token = put<uint64_t>(0);
    const uint64_t rt_slot = put<uint64_t>(scene.rt);
    uint64_t previous = 0;
    std::vector<std::vector<Packet>> frames;
    const auto start = std::chrono::steady_clock::now();
    for (int frame = 0; frame < 4; ++frame) {
        const size_t mark = fx.captured.size();
        // Frame start: (type 0, timeout 0, 0, flags 2, &token) never fails and never blocks.
        store64(token_slot, 0);
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(hr(fx.vcall(scene.device, XSlot::WaitFrameEvent, {I(0), I(0), I(0), I(2), I(token_slot)})) == 0);
        CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(50));
        const uint64_t token = load64(token_slot);
        CHECK(token >= 1 && token >= previous);
        // Record and execute with the frame token (count, lists, token, flags 3).
        scene.record(fx.thread, list);
        std::vector<uint64_t> lists{list};
        fx.vcall(scene.queue, Slot2::ExecuteCommandLists, {I(1), I(put_bytes(lists.data(), 8)), I(token), I(3)});
        CHECK(dev->last_execute_token.load() == token);
        // Before presenting: wait until the previous frame's token is over (type 2, INFINITE).
        if (previous != 0) {
            store64(present_token, previous);
            CHECK(hr(fx.vcall(scene.device, XSlot::WaitFrameEvent,
                              {I(2), I(0xFFFFFFFFu), I(0), I(0), I(present_token)})) == 0);
            CHECK(load64(present_token) > previous);
        }
        // PresentX(plane count, planes, params): plane 0 names the back buffer.
        dg::xbox::GuestPresentPlaneX plane{};
        plane.Token = token;
        plane.ResourceCount = 1;
        plane.ppResources = rt_slot;
        uint8_t params[32] = {};
        CHECK(hr(fx.vcall(scene.queue, Slot::PresentX, {I(1), I(put(plane)), I(put_bytes(params, sizeof(params)))})) == 0);
        frames.push_back(packets_since(fx, mark));
        previous = token;
        // The next frame waits for a newer tick (bounded by one interval).
        store64(token_slot, 0);
        CHECK(hr(fx.vcall(scene.device, XSlot::WaitFrameEvent, {I(0), I(1000), I(0), I(0), I(token_slot)})) == 0);
        CHECK(load64(token_slot) > token);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    // Paced: four frames take at least ~3 intervals, and nothing spins for long.
    CHECK(elapsed >= std::chrono::milliseconds(12));
    CHECK(elapsed < std::chrono::seconds(5));
    // The first present names the back buffer: frame 0 rendered into the texture, later frames
    // into the canvas; every frame ends with Present.
    CHECK(counter(dev, dg::xbox::Counter::BackBufferMarks) == 1);
    CHECK(object_of<dg::Resource>(scene.rt)->is_back_buffer());
    for (size_t i = 0; i < frames.size(); ++i) {
        const Packet* pass = find_op(frames[i], GpuOpcode::BeginRenderPass);
        CHECK(pass != nullptr);
        if (pass != nullptr) {
            const uint32_t target = payload_as<GpuBeginRenderPassPayload>(*pass).color[0].texture_id;
            CHECK(target == (i == 0 ? rt_id : kGpuSwapchainTextureId));
        }
        CHECK(!frames[i].empty() && frames[i].back().opcode == GpuOpcode::Present);
    }
    CHECK(dev->stats().presents == 4 && counter(dev, dg::xbox::Counter::PresentsWithPlanes) == 4);
    CHECK(dev->last_present_token.load() == previous);
    // A present without planes (suspend flush) emits nothing.
    const size_t mark = fx.captured.size();
    CHECK(hr(fx.vcall(scene.queue, Slot::PresentX, {I(0), I(0), I(0)})) == 0);
    CHECK(fx.captured.size() == mark);

    // Shutdown releases a waiter blocked on another thread (longest interval) and later waits
    // return at once: no deadlock when the submission thread is still waiting.
    CHECK(hr(fx.vcall(scene.device, XSlot::SetFrameInterval, {I(0), I(100000000), I(1), I(0)})) == 0);
    CHECK(dev->frame_pacer().interval_us() == dg::xbox::FramePacer::kMaxIntervalUs);
    GuestThread other{};
    CHECK(create_guest_thread(fx.arena, 512 * 1024, other));
    std::atomic<int32_t> waiter_hr{-1};
    const uint64_t waiter_slot = put<uint64_t>(0);
    std::thread waiter([&] {
        waiter_hr = hr(fx.vcall_on(other, scene.device, XSlot::WaitFrameEvent,
                                   {I(0), I(0xFFFFFFFFu), I(0), I(0), I(waiter_slot)}));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    dev->shutdown_frame_events();
    waiter.join();
    CHECK(waiter_hr.load() == 0 && load64(waiter_slot) != 0);
    const auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) {
        CHECK(hr(fx.vcall(scene.device, XSlot::WaitFrameEvent, {I(0), I(0xFFFFFFFFu), I(0), I(0), I(token_slot)})) == 0);
    }
    CHECK(std::chrono::steady_clock::now() - s0 < std::chrono::milliseconds(100));
    destroy_guest_thread(fx.arena, other);
    CHECK(t_balanced);
    scene.release_all({list});
}

void test_frame_pacer_unit() {
    dg::xbox::FramePacer pacer;
    pacer.set_interval(2000, 1);
    uint64_t frame = 0;
    const uint64_t now = pacer.current_frame();
    // Waiting after the current tick blocks until the next one.
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(pacer.wait_after(now, 1000, &frame) == dg::xbox::FramePacer::WaitResult::Ok && frame > now);
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500));
    // Waiting after an older tick returns at once.
    CHECK(pacer.wait_after(0, 1000, &frame) == dg::xbox::FramePacer::WaitResult::Ok);
    // A timeout shorter than the interval times out.
    pacer.set_interval(100000, 1);
    CHECK(pacer.wait_after(pacer.current_frame(), 1, &frame) == dg::xbox::FramePacer::WaitResult::Timeout);
    // Clamping.
    pacer.set_interval(1, 1);
    CHECK(pacer.interval_us() == dg::xbox::FramePacer::kMinIntervalUs);
    pacer.set_interval(0, 1);
    CHECK(pacer.interval_us() == dg::xbox::FramePacer::kDefaultIntervalUs);
    pacer.shutdown();
    CHECK(pacer.is_shut_down());
    CHECK(pacer.wait_after(pacer.current_frame() + 100, 0xFFFFFFFFu, &frame) == dg::xbox::FramePacer::WaitResult::Shutdown);
}

// Pipeline-state subobject stream (public layout: pointer-aligned subobjects, u32 type first).
struct StreamBuilder {
    std::vector<uint8_t> bytes;
    template <typename T>
    void add(uint32_t type, size_t data_offset, const T& value) {
        const size_t off = (bytes.size() + 7) & ~size_t{7};
        bytes.resize(off + data_offset + sizeof(T), 0);
        std::memcpy(bytes.data() + off, &type, 4);
        std::memcpy(bytes.data() + off + data_offset, &value, sizeof(T));
    }
    void finish() { bytes.resize((bytes.size() + 7) & ~size_t{7}, 0); }
};

struct RtFormats {
    uint32_t formats[8];
    uint32_t count;
};

std::vector<uint8_t> graphics_stream(Scene& scene, uint32_t cull_mode) {
    StreamBuilder s;
    s.add<uint64_t>(0, 8, scene.rs);
    const dg::GuestShaderBytecode vs{put_bytes(scene.vs_bytes.data(), scene.vs_bytes.size()), scene.vs_bytes.size()};
    const dg::GuestShaderBytecode ps{put_bytes(scene.ps_bytes.data(), scene.ps_bytes.size()), scene.ps_bytes.size()};
    s.add(1, 8, vs);
    s.add(2, 8, ps);
    dg::GuestBlendDesc blend{};
    for (auto& t : blend.RenderTarget) {
        t.SrcBlend = t.SrcBlendAlpha = 2;
        t.DestBlend = t.DestBlendAlpha = 1;
        t.BlendOp = t.BlendOpAlpha = 1;
        t.LogicOp = 4;
        t.RenderTargetWriteMask = 0xF;
    }
    s.add(8, 4, blend);
    s.add<uint32_t>(9, 4, 0xFFFFFFFFu);
    dg::GuestRasterizerDesc r{};
    r.FillMode = 3;
    r.CullMode = cull_mode;
    r.DepthClipEnable = 1;
    s.add(10, 4, r);
    uint8_t ds1[56] = {};  // DEPTH_STENCIL1: depth off
    s.add(21, 4, ds1);
    const uint32_t sample[2] = {1, 0};
    s.add(17, 4, sample);
    static const char kPosition[] = "POSITION";
    dg::GuestInputElementDesc e{};
    e.SemanticName = put_bytes(kPosition, sizeof(kPosition));
    e.Format = 6;
    s.add(12, 8, dg::GuestInputLayoutDesc{put(e), 1});
    s.add<uint32_t>(14, 4, 3);
    RtFormats f{};
    f.formats[0] = 87;
    f.count = 1;
    s.add(15, 4, f);
    s.add<uint32_t>(16, 4, 0);
    s.finish();
    return s.bytes;
}

void test_xbox_pipeline_streams_and_overrides() {
    Fixture fx;
    Scene scene(fx);
    dg::Device* dev = host_device(scene.device);
    int32_t result = -1;

    // Unit: the parser follows the public layout and reports what it saw.
    const std::vector<uint8_t> bytes = graphics_stream(scene, 3);
    dg::xbox::StreamPipeline parsed;
    std::string error;
    CHECK(dg::xbox::parse_pipeline_stream(bytes.data(), bytes.size(), &parsed, &error));
    CHECK(parsed.kind == dg::xbox::StreamPipelineKind::Graphics && parsed.subobjects == 12);
    CHECK(parsed.graphics.pRootSignature == scene.rs && parsed.graphics.NumRenderTargets == 1 &&
          parsed.graphics.RTVFormats[0] == 87 && parsed.graphics.RasterizerState.CullMode == 3 &&
          parsed.graphics.SampleCount == 1 && parsed.graphics.PrimitiveTopologyType == 3 &&
          parsed.graphics.InputLayout.NumElements == 1 && parsed.graphics.DepthStencilState.DepthEnable == 0);
    std::vector<uint8_t> broken = bytes;
    broken[0] = 99;  // unknown subobject type
    CHECK(!dg::xbox::parse_pipeline_stream(broken.data(), broken.size(), &parsed, &error) &&
          error.find("unknown subobject type 99") != std::string::npos);
    CHECK(!dg::xbox::parse_pipeline_stream(bytes.data(), bytes.size() - 12, &parsed, &error));

    // Slot 86: ({SizeInBytes, pStream}, riid, out).
    auto stream_desc = [&](const std::vector<uint8_t>& b) {
        return put(dg::xbox::GuestPipelineStateStreamDesc{b.size(), put_bytes(b.data(), b.size())});
    };
    const uint64_t pso = fx.create(scene.device, XSlot::CreatePsoStream, {I(stream_desc(bytes))},
                                   dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && pso != 0);
    auto* state = object_of<dg::PipelineState>(pso);
    CHECK(state->resolved() && state->graphics_source() != nullptr);
    CHECK(state->graphics_source()->desc.RTVFormats[0] == playgta6::gpu::pso::DxgiFormat(87));
    CHECK(load64(pso + 0x10) == dg::xbox::kPipelineStateTag && load64(pso + 0x18) == pso);

    // Slot 55: (base, n, overrides of 0x14C bytes {u32 kind; desc}, riid, out).
    std::vector<uint8_t> overrides(2 * dg::xbox::kPsoOverrideStride, 0);
    const uint32_t raster_kind = 1, unknown_kind = 5;
    dg::GuestRasterizerDesc r{};
    r.FillMode = 2;  // WIREFRAME
    r.CullMode = 1;  // NONE
    std::memcpy(overrides.data(), &raster_kind, 4);
    std::memcpy(overrides.data() + 4, &r, sizeof(r));
    std::memcpy(overrides.data() + dg::xbox::kPsoOverrideStride, &unknown_kind, 4);
    const uint64_t derived = fx.create(scene.device, XSlot::CreatePsoWithOverrides,
                                       {I(pso), I(2), I(put_bytes(overrides.data(), overrides.size()))},
                                       dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && derived != 0 && derived != pso);
    const auto* dsrc = object_of<dg::PipelineState>(derived)->graphics_source();
    CHECK(dsrc != nullptr && dsrc->desc.RasterizerState.CullMode == playgta6::gpu::pso::D3D12CullMode(1) &&
          dsrc->desc.RasterizerState.FillMode == playgta6::gpu::pso::D3D12FillMode(2));
    CHECK(dsrc->desc.NumInputElements == 1 && std::string(dsrc->desc.pInputElementDescs[0].SemanticName) == "POSITION");
    CHECK(state->graphics_source()->desc.RasterizerState.CullMode == playgta6::gpu::pso::D3D12CullMode(3));
    CHECK(counter(dev, dg::xbox::Counter::UnknownOverrideKinds) == 1);
    CHECK(fx.create(scene.device, XSlot::CreatePsoWithOverrides, {I(scene.allocator), I(0), I(0)},
                    dg::IID_ID3D12PipelineState, &result) == 0 &&
          result == dg::kEInvalidArg);

    // A derived PSO draws: staged protocol on the list.
    const uint64_t list = scene.new_list();
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(0)})) == 0);
    fx.vcall(list, Slot2::SetGraphicsRootSignature, {I(scene.rs)});
    staged_set_pipeline_state(fx, fx.thread, list, derived);
    fx.vcall(list, Slot2::OMSetRenderTargets, {I(1), I(put(scene.rtv)), I(0), I(0)});
    fx.vcall(list, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    const size_t mark = fx.captured.size();
    scene.execute({list});
    CHECK(count_op(packets_since(fx, mark), GpuOpcode::Draw) == 1);

    // Compute stream (root signature + CS) and the public compute slot 11.
    const std::vector<uint8_t> cs{'D', 'X', 'B', 'C', 0x57};
    dev->register_shader_module(dg::shader_identity(cs.data(), cs.size()), 9300);
    StreamBuilder c;
    c.add<uint64_t>(0, 8, scene.rs);
    c.add(6, 8, dg::GuestShaderBytecode{put_bytes(cs.data(), cs.size()), cs.size()});
    c.finish();
    const uint64_t cpso = fx.create(scene.device, XSlot::CreatePsoStream, {I(stream_desc(c.bytes))},
                                    dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && object_of<dg::PipelineState>(cpso)->is_compute() && object_of<dg::PipelineState>(cpso)->resolved());
    dg::GuestComputePipelineStateDesc cd{};
    cd.pRootSignature = scene.rs;
    cd.CS = {put_bytes(cs.data(), cs.size()), cs.size()};
    const uint64_t cpso57 = fx.create(scene.device, Slot2::CreateComputePipelineState, {I(put(cd))},
                                      dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && object_of<dg::PipelineState>(cpso57)->is_compute());

    // Mesh stream: a placeholder that binds as unresolved (never a null pointer).
    StreamBuilder m;
    m.add<uint64_t>(0, 8, scene.rs);
    m.add(24, 8, dg::GuestShaderBytecode{put_bytes(cs.data(), cs.size()), cs.size()});
    m.finish();
    const uint64_t mpso = fx.create(scene.device, XSlot::CreatePsoStream, {I(stream_desc(m.bytes))},
                                    dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && mpso != 0 && !object_of<dg::PipelineState>(mpso)->resolved() &&
          !object_of<dg::PipelineState>(mpso)->unsupported_reason().empty());
    // Malformed stream: E_INVALIDARG and a cleared out pointer.
    CHECK(fx.create(scene.device, XSlot::CreatePsoStream, {I(stream_desc(broken))}, dg::IID_ID3D12PipelineState,
                    &result) == 0 &&
          result == dg::kEInvalidArg);

    // Title-blob PSOs (62, 67) and ray-tracing state objects (101) are refused with a null out.
    const uint64_t out62 = put<uint64_t>(0);
    CHECK(hr(fx.vcall(scene.device, XSlot::PsoFromBlob,
                      {I(put<uint64_t>(0)), I(0), I(0), I(fx.guid(dg::IID_ID3D12PipelineState)), I(out62)})) ==
          dg::kENotImpl);
    CHECK(load64(out62) == 0);
    const uint64_t out67 = put<uint64_t>(0);
    CHECK(hr(fx.vcall(scene.device, XSlot::PsoFromBlobCached,
                      {I(put<uint64_t>(0)), I(fx.guid(dg::IID_ID3D12PipelineState)), I(out67)})) == dg::kENotImpl);
    CHECK(load64(out67) == 0);
    const uint64_t so = fx.create(scene.device, XSlot::CreateStateObject, {I(put<uint64_t>(3))},
                                  com::make_guid("47016943-fca8-4594-93ea-af258b55346d"), &result);
    CHECK(so == 0 && result == dg::kENotImpl);
    CHECK(counter(dev, dg::xbox::Counter::StateObjectsRefused) == 1);
    // Ray tracing is reported as unsupported (OPTIONS5.RaytracingTier = 0).
    const uint64_t opts5 = put_bytes(std::vector<uint8_t>(12, 0xFF).data(), 12);
    CHECK(hr(fx.vcall(scene.device, Slot::CheckFeatureSupport, {I(27), I(opts5), I(12)})) == 0);
    CHECK(load32(opts5 + 8) == 0);

    CHECK(t_balanced);
    scene.release_all({list, pso, derived, cpso, cpso57, mpso});
}

// G32: console-native pipelines (Xbox slot 57). Synthetic records built from the range layout in
// d3d12_xs_xbox.h; the bytes are arbitrary patterns.
std::vector<uint8_t> byte_pattern(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<uint8_t>(seed * 31 + i * 7);
    }
    return v;
}

uint64_t put_aligned(const std::vector<uint8_t>& bytes, uint64_t alignment = 256) {
    const uint64_t va = com::guest_alloc(bytes.size() + alignment);
    CHECK(va != 0);
    const uint64_t aligned = (va + alignment - 1) / alignment * alignment;
    CHECK(com::write_guest(aligned, bytes.data(), bytes.size()));
    return aligned;
}

void test_xbox_native_pipelines() {
    Fixture fx;
    Scene scene(fx);
    dg::Device* dev = host_device(scene.device);
    int32_t result = -1;
    using R = dg::xbox::NativeRange;
    namespace xb = dg::xbox;

    const std::vector<uint8_t> header = byte_pattern(0x184, 1), meta = byte_pattern(0x50, 2);
    const std::vector<uint8_t> vs_code = byte_pattern(0x140, 3), vs_meta = byte_pattern(0x30, 4);
    const std::vector<uint8_t> ps_code = byte_pattern(0x180, 5), ps_meta = byte_pattern(0x70, 6);
    const uint64_t header_va = put_bytes(header.data(), header.size());
    const uint64_t meta_va = put_bytes(meta.data(), meta.size());
    const uint64_t vs_code_va = put_aligned(vs_code), vs_meta_va = put_bytes(vs_meta.data(), vs_meta.size());
    const uint64_t ps_code_va = put_aligned(ps_code), ps_meta_va = put_bytes(ps_meta.data(), ps_meta.size());
    auto set = [](xb::GuestNativePipelineDesc& d, R r, uint64_t address, uint64_t size) {
        d.ranges[static_cast<size_t>(r)] = xb::GuestNativeRange{address, size};
    };
    auto desc = [&](bool pixel, bool alt) {
        xb::GuestNativePipelineDesc d{};
        d.pRootSignature = scene.rs;
        set(d, R::Header, header_va, header.size());
        set(d, R::PipelineMetadata, meta_va, meta.size());
        if (pixel) {
            set(d, R::PixelCode, ps_code_va, ps_code.size());
            set(d, R::PixelMetadata, ps_meta_va, ps_meta.size());
        }
        set(d, alt ? R::VertexCodeAlt : R::VertexCode, vs_code_va, vs_code.size());
        set(d, alt ? R::VertexMetadataAlt : R::VertexMetadata, vs_meta_va, vs_meta.size());
        return d;
    };

    // Unit: the range layout rules.
    xb::NativePipeline p;
    std::string error;
    CHECK(xb::parse_native_pipeline(desc(true, false), &p, &error));
    CHECK(p.root_signature == scene.rs && p.header.address == header_va && p.header.size == header.size() &&
          p.metadata.size == meta.size() && p.pixel.present() && p.pixel.code.address == ps_code_va &&
          p.vertex.code.address == vs_code_va && p.vertex.metadata.size == vs_meta.size() && !p.vertex_alt);
    CHECK(xb::parse_native_pipeline(desc(false, true), &p, &error));
    CHECK(!p.pixel.present() && p.vertex_alt && p.vertex.code.address == vs_code_va);
    auto rejects = [&](xb::GuestNativePipelineDesc d, const char* message) {
        const bool ok = xb::parse_native_pipeline(d, &p, &error);
        CHECK(!ok && error.find(message) != std::string::npos);
    };
    xb::GuestNativePipelineDesc bad = desc(true, false);
    bad.ranges[6] = bad.ranges[4];
    bad.ranges[7] = bad.ranges[5];
    rejects(bad, "two vertex stages");
    bad = desc(true, false);
    bad.ranges[4] = bad.ranges[5] = {};
    rejects(bad, "no vertex stage");
    bad = desc(true, false);
    bad.ranges[4].address += 4;
    rejects(bad, "256-byte aligned");
    bad = desc(true, false);
    bad.ranges[3] = {};
    rejects(bad, "pixel stage has code without metadata");
    bad = desc(true, false);
    bad.ranges[0] = {};
    rejects(bad, "no header record");
    bad = desc(true, false);
    bad.ranges[1].size = 0;
    rejects(bad, "address or a size but not both");
    bad = desc(true, false);
    bad.ranges[2].size = xb::kMaxNativeRangeBytes + 256;
    rejects(bad, "too large");

    // Slot 57: an unresolved graphics pipeline state with one identity per stage (code, then metadata).
    const uint64_t vs_id = dg::native_stage_identity(vs_code.data(), vs_code.size(), vs_meta.data(), vs_meta.size());
    const uint64_t ps_id = dg::native_stage_identity(ps_code.data(), ps_code.size(), ps_meta.data(), ps_meta.size());
    std::vector<uint8_t> concat = vs_code;
    concat.insert(concat.end(), vs_meta.begin(), vs_meta.end());
    CHECK(vs_id == dg::shader_identity(concat.data(), concat.size()) && vs_id != ps_id);
    const uint64_t pso = fx.create(scene.device, XSlot::CreateNativePso, {I(put(desc(true, false)))},
                                   dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && pso != 0);
    auto* state = object_of<dg::PipelineState>(pso);
    CHECK(!state->is_compute() && !state->resolved() && state->graphics_source() == nullptr &&
          !state->unsupported_reason().empty());
    CHECK(state->native() != nullptr && state->native()->vertex_identity == vs_id &&
          state->native()->pixel_identity == ps_id && state->native()->header == header);
    CHECK((state->shader_identities() == std::vector<uint64_t>{vs_id, ps_id}));
    CHECK(dev->shader_size(vs_id) == vs_code.size() && dev->shader_size(ps_id) == ps_code.size());
    CHECK(load64(pso + 0x10) == xb::kPipelineStateTag && load64(pso + 0x18) == pso);
    // The same stages elsewhere in memory keep their identities; other metadata gives another one.
    const uint64_t vs_code_copy = put_aligned(vs_code);
    xb::GuestNativePipelineDesc moved = desc(true, false);
    moved.ranges[4].address = vs_code_copy;
    const uint64_t pso_moved = fx.create(scene.device, XSlot::CreateNativePso, {I(put(moved))},
                                         dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && object_of<dg::PipelineState>(pso_moved)->native()->vertex_identity == vs_id);
    const std::vector<uint8_t> vs_meta2 = byte_pattern(0x30, 9);
    xb::GuestNativePipelineDesc depth_only = desc(false, true);
    depth_only.ranges[7].address = put_bytes(vs_meta2.data(), vs_meta2.size());
    const uint64_t pso_depth = fx.create(scene.device, XSlot::CreateNativePso, {I(put(depth_only))},
                                         dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && pso_depth != 0);
    const dg::PipelineState::Native* dn = object_of<dg::PipelineState>(pso_depth)->native();
    CHECK(dn->vertex_identity != vs_id && dn->pixel_identity == 0 && dn->layout.vertex_alt);
    CHECK((dev->native_stage_counts() == std::array<uint32_t, 2>{2, 1}));
    CHECK(dev->unresolved_shaders().empty());  // the DXIL list stays for the shader track
    CHECK(counter(dev, xb::Counter::NativePipelineStates) == 3);
    // Malformed descriptor: E_INVALIDARG and a cleared out pointer.
    bad = desc(true, false);
    bad.ranges[0] = {};
    CHECK(fx.create(scene.device, XSlot::CreateNativePso, {I(put(bad))}, dg::IID_ID3D12PipelineState, &result) == 0 &&
          result == dg::kEInvalidArg);
    CHECK(counter(dev, xb::Counter::NativePipelineStatesRejected) == 1);

    // Slot 55 on a native base: the base is handed out, the override entries are counted.
    std::vector<uint8_t> overrides(2 * xb::kPsoOverrideStride, 0);
    const uint32_t blend_kind = 2;
    std::memcpy(overrides.data() + xb::kPsoOverrideStride, &blend_kind, 4);
    const uint64_t derived = fx.create(scene.device, XSlot::CreatePsoWithOverrides,
                                       {I(pso), I(2), I(put_bytes(overrides.data(), overrides.size()))},
                                       dg::IID_ID3D12PipelineState, &result);
    CHECK(result == 0 && derived == pso);
    CHECK(counter(dev, xb::Counter::NativeOverridesIgnored) == 2);

    // The draw binds it as a graphics pipeline and is skipped (described once, with the stages).
    const uint64_t skipped_before = dev->stats().skipped_draws;
    const uint64_t list = scene.new_list();
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(0)})) == 0);
    fx.vcall(list, Slot2::SetGraphicsRootSignature, {I(scene.rs)});
    staged_set_pipeline_state(fx, fx.thread, list, pso);
    fx.vcall(list, Slot2::OMSetRenderTargets, {I(1), I(put(scene.rtv)), I(0), I(0)});
    fx.vcall(list, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    fx.vcall(list, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    const size_t mark = fx.captured.size();
    scene.execute({list});
    CHECK(count_op(packets_since(fx, mark), GpuOpcode::Draw) == 0);
    CHECK(dev->stats().skipped_draws == skipped_before + 2);
    const std::vector<std::string> reports = dev->skipped_draw_reports();
    CHECK(contains(reports, "native stages: vertex 320 B code + 48 B metadata (ranges 4/5), pixel 384 B code + "
                            "112 B metadata; header record 388 B, pipeline metadata 80 B;"));
    CHECK(contains(reports, "console-native pipeline"));

    CHECK(t_balanced);
    scene.release_all({list, derived, pso, pso_moved, pso_depth});
}

void test_xbox_device_queue_and_list_slots() {
    Fixture fx;
    Scene scene(fx);
    dg::Device* dev = host_device(scene.device);
    int32_t result = -1;

    // Recorded no-ops with S_OK (results ignored or only checked for success by the caller).
    CHECK(hr(fx.vcall(scene.device, XSlot::FeatureToggle, {I(0x11F), I(1)})) == 0);
    CHECK(hr(fx.vcall(scene.device, XSlot::RemovalCallbacks, {I(0x140001000), I(0), I(0x140001000)})) == 0);
    CHECK(hr(fx.vcall(scene.device, XSlot::Slot61, {I(1), I(2)})) == 0);
    CHECK(hr(fx.vcall(scene.device, XSlot::Slot119, {I(put<uint64_t>(0)), I(0x80), I(0)})) == 0);
    CHECK(hr(fx.vcall(scene.device, XSlot::Slot132, {I(3)})) == 0);
    CHECK(counter(dev, dg::xbox::Counter::FeatureToggles) == 1 && counter(dev, dg::xbox::Counter::XboxNoOps) == 4);

    // Page pool (address, pages, &handle) and frame statistics (token, flags, &count, entries).
    const uint64_t handle = put<uint64_t>(0);
    CHECK(hr(fx.vcall(scene.device, XSlot::PagePool, {I(com::guest_alloc(65536)), I(1), I(handle)})) == 0);
    CHECK(load64(handle) != 0);
    const uint64_t stat_count = put<uint32_t>(6);
    CHECK(hr(fx.vcall(scene.device, XSlot::FrameStatistics, {I(1), I(0x11100), I(stat_count), I(put_bytes(
                                                                std::vector<uint8_t>(6 * 0x38).data(), 6 * 0x38))})) == 0);
    CHECK(load32(stat_count) == 0);

    // GetResourceAllocationInfo1 (slot 108): struct return, result in rcx and this in rdx.
    dg::GuestResourceDesc descs[2] = {buffer_desc(1000), buffer_desc(70000)};
    const uint64_t descs_va = put_bytes(descs, sizeof(descs));
    const uint64_t infos = put_bytes(std::vector<uint8_t>(48, 0xEE).data(), 48);
    const uint64_t result_buf = put(dg::GuestResourceAllocationInfo{});
    CHECK(fx.sret_call(scene.device, XSlot::AllocationInfo1, result_buf, {I(1), I(2), I(descs_va), I(infos)}) == result_buf);
    const auto total = get<dg::GuestResourceAllocationInfo>(result_buf);
    const auto i0 = get<dg::xbox::GuestResourceAllocationInfo1>(infos);
    const auto i1 = get<dg::xbox::GuestResourceAllocationInfo1>(infos + 24);
    CHECK(total.SizeInBytes == 65536 + 131072 && total.Alignment == 65536);
    CHECK(i0.Offset == 0 && i0.SizeInBytes == 65536 && i1.Offset == 65536 && i1.SizeInBytes == 131072);

    // Buffer views at GPU addresses (slots 70/71): view desc + u64 address at +0x28.
    const uint64_t buf = committed(fx, scene.device, 2, buffer_desc(4096), 0xAC3);
    const uint64_t buf_va = fx.vcall(buf, Slot2::GetGPUVirtualAddress);
    const uint32_t buf_id = object_of<dg::Resource>(buf)->id();
    const uint64_t heap = fx.create(scene.device, Slot::CreateDescriptorHeap,
                                    {I(put(dg::GuestDescriptorHeapDesc{0, 8, 1, 0}))}, dg::IID_ID3D12DescriptorHeap);
    const uint64_t start = put<uint64_t>(0);
    fx.sret_call(heap, Slot::GetCPUStart, start);
    const uint64_t cpu0 = load64(start);
    auto* host_heap = host_object<dg::DescriptorHeap, dg::IDescriptorHeapHost>(heap)->heap();
    uint8_t srv[48] = {};
    dg::GuestShaderResourceViewDesc sd{};
    sd.ViewDimension = 1;  // BUFFER
    sd.Shader4ComponentMapping = 0x1688;
    sd.u[2] = 4;   // NumElements
    sd.u[3] = 16;  // StructureByteStride
    std::memcpy(srv, &sd, sizeof(sd));
    const uint64_t at = buf_va + 64;
    std::memcpy(srv + 0x28, &at, 8);
    const uint64_t rdesc = put(buffer_desc(64));
    fx.vcall(scene.device, XSlot::SrvAtAddress, {I(rdesc), I(put_bytes(srv, sizeof(srv))), I(cpu0)});
    const d3d::ViewDescriptor* v0 = host_heap->view(0);
    CHECK(v0 != nullptr && v0->kind == d3d::DescriptorKind::Srv && v0->resource_id == buf_id && v0->first_element_or_address == 4);
    uint8_t uav[48] = {};
    dg::GuestUnorderedAccessViewDesc ud{};
    ud.Format = 39;  // R32_TYPELESS
    ud.ViewDimension = 1;
    ud.u[2] = 8;
    ud.u[6] = 1;  // RAW
    std::memcpy(uav, &ud, sizeof(ud));
    const uint64_t at2 = buf_va + 16;
    std::memcpy(uav + 0x28, &at2, 8);
    fx.vcall(scene.device, XSlot::UavAtAddress, {I(rdesc), I(put_bytes(uav, sizeof(uav))), I(cpu0 + 32)});
    const d3d::ViewDescriptor* v1 = host_heap->view(1);
    CHECK(v1 != nullptr && v1->kind == d3d::DescriptorKind::Uav && v1->resource_id == buf_id && v1->first_element_or_address == 4);
    const uint64_t unknown = 0x00000BAD00000000ULL;
    std::memcpy(srv + 0x28, &unknown, 8);
    const uint64_t bad_before = dev->stats().bad_arguments;
    fx.vcall(scene.device, XSlot::SrvAtAddress, {I(rdesc), I(put_bytes(srv, sizeof(srv))), I(cpu0 + 64)});
    CHECK(dev->stats().bad_arguments == bad_before + 1);  // unknown address: null descriptor, diagnosed
    CHECK(counter(dev, dg::xbox::Counter::ViewsAtAddress) == 3);

    // GPU alias window: a buffer at a GPU-window address reads its bytes from the CPU window.
    const uint64_t cpu_window = com::guest_alloc(1 << 16);
    const uint64_t gpu_window = 0x00007D0000000000ULL;
    dev->register_gpu_alias(gpu_window, cpu_window, 1 << 16);
    CHECK(dev->cpu_address_for_gpu(gpu_window + 0x1000) == cpu_window + 0x1000);
    CHECK(dev->cpu_address_for_gpu(0x1234) == 0x1234);
    const uint64_t aliased = fx.create(scene.device, Slot2::CreatePlacedResourceX,
                                       {I(gpu_window + 0x1000), I(put(buffer_desc(4096))), I(0), I(0)},
                                       dg::IID_ID3D12Resource, &result);
    CHECK(result == 0 && fx.vcall(aliased, Slot2::GetGPUVirtualAddress) == gpu_window + 0x1000);
    CHECK(object_of<dg::Resource>(aliased)->cpu_address() == cpu_window + 0x1000);

    // Queue: suspend / resume and page mappings.
    auto* queue = host_object<dg::CommandQueue, dg::ICommandQueueHost>(scene.queue);
    CHECK(hr(fx.vcall(scene.queue, XSlot::Suspend, {I(0)})) == 0 && queue->suspended());
    CHECK(hr(fx.vcall(scene.queue, XSlot::Resume)) == 0 && !queue->suspended());
    CHECK(hr(fx.vcall(scene.queue, XSlot::PageMappings, {I(0), I(0), I(0), I(0), I(0)})) == 0);
    CHECK(counter(dev, dg::xbox::Counter::PageMappingsApplied) == 1);

    // Allocator slot 9: the command-buffer allocation callback is recorded, never called.
    CHECK(hr(fx.vcall(scene.allocator, XSlot::AllocationCallback, {I(0x140005000), I(0x1234)})) == 0);
    auto* host_allocator = host_object<dg::CommandAllocator, dg::ICommandAllocatorHost>(scene.allocator);
    CHECK(host_allocator->allocation_callback() == 0x140005000);

    // Command list: Xbox slots in a recording (draw, unsupported state, recorded-only slots).
    const uint64_t list = scene.new_list();
    scene.record(fx.thread, list);  // closes
    CHECK(hr(fx.vcall(list, Slot2::Reset, {I(scene.allocator), I(scene.pso)})) == 0);
    fx.vcall(list, Slot2::SetGraphicsRootSignature, {I(scene.rs)});
    fx.vcall(list, Slot2::OMSetRenderTargets, {I(1), I(put(scene.rtv)), I(0), I(0)});
    fx.vcall(list, XSlot::OMSetDepthBounds, {F32(0.0f), F32(1.0f)});
    fx.vcall(list, XSlot::RSSetShadingRate, {I(0), I(0)});
    fx.vcall(list, XSlot::RSSetShadingRateImage, {I(0)});
    fx.vcall(list, XSlot::RegisterWrite, {I(0xC280), I(0x20000), I(0), I(0xFFFFFFFFFF00ULL)});
    fx.vcall(list, 132, {I(0x700000), I(0), I(0), I(0xFFFFFFFFFF00ULL)});
    fx.vcall(list, XSlot::BeginQuery, {I(0), I(0), I(0)});
    fx.vcall(list, XSlot::ExecuteIndirect, {I(0), I(1), I(buf), I(0), I(0), I(0)});
    fx.vcall(list, Slot2::DrawInstanced, {I(3), I(1), I(0), I(0)});
    CHECK(hr(fx.vcall(list, Slot2::Close)) == 0);
    const auto calls = object_of<dg::CommandList>(list)->xbox_slot_calls();
    CHECK(calls.size() == 2 && calls[0].first == 131 && calls[1].first == 132);
    CHECK(counter(dev, dg::xbox::Counter::DepthBoundsIgnored) == 1 &&
          counter(dev, dg::xbox::Counter::ShadingRateIgnored) == 2 &&
          counter(dev, dg::xbox::Counter::IndirectSkipped) == 1);
    const size_t mark = fx.captured.size();
    scene.execute({list});
    CHECK(count_op(packets_since(fx, mark), GpuOpcode::Draw) == 1);  // the indirect draw has no packet
    CHECK(com::stub_hits(dg::kID3D12GraphicsCommandList, 131) == 0);

    CHECK(t_balanced);
    scene.release_all({list, buf, heap, aliased});
}

// O27: the XMem CPU/GPU windows reach the device's alias table whether they are reserved before
// or after the device, and queue slot 31 applies GPU-window page mappings.
void test_xmem_window_alias_wiring() {
    namespace xw = playgta6::os::xmem_windows;
    namespace sc = playgta6::os::scarlett;
    constexpr uint64_t S = 1ull << 30, k64K = 64ull << 10;
    constexpr uint32_t kType = 0x20402000u;
    sc::xmem_reset_for_testing();
    xw::set_policy(xw::PlacementPolicy::PhantomGpuWindows);
    void* cpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kType, 0x21u, 0x4u);
    void* gpu = pg6_xbox_XMemVirtualAlloc(nullptr, S, kType, 0x3u, 0x801u);
    CHECK(cpu != nullptr && gpu != nullptr);
    {
        Fixture fx;
        const uint64_t device = fx.create_device();
        dg::Device* dev = host_device(device);
        const uint64_t C = playgta6::os::win32::get_vmem_arena()->guest_address(cpu);
        const uint64_t G = playgta6::os::win32::get_vmem_arena()->guest_address(gpu);
        CHECK(G >= xw::kPhantomBase && dev->cpu_address_for_gpu(G + 3 * k64K + 8) == C + 3 * k64K + 8);
        // A buffer created at a GPU-window address reads its bytes from the CPU window, page by page
        // as pages get mapped.
        int32_t result = -1;
        const uint64_t buf = fx.create(device, Slot2::CreatePlacedResourceX,
                                       {I(G + 4 * k64K), I(put(buffer_desc(2 * k64K))), I(0), I(0)},
                                       dg::IID_ID3D12Resource, &result);
        CHECK(result == 0 && object_of<dg::Resource>(buf)->cpu_address() == C + 4 * k64K);
        // Queue slot 31: one range, map pool pages 2 and 3 at GPU pages 4 and 5.
        const uint64_t queue = fx.create(device, Slot::CreateCommandQueue, {I(put(dg::GuestCommandQueueDesc{}))},
                                         dg::IID_ID3D12CommandQueue);
        const uint32_t entries[6] = {1, 2, 1, 1, 3, 1};
        struct {
            uint64_t gpu_va;
            uint32_t n, pad;
            uint64_t entries;
        } range{G + 4 * k64K, 2, 0, put_bytes(entries, sizeof(entries))};
        CHECK(hr(fx.vcall(queue, 31, {I(1), I(put(range)), I(0x1000), I(16), I(0)})) == 0);
        CHECK(xw::gpu_page_pool_index(G + 4 * k64K) == 2 && xw::gpu_page_pool_index(G + 5 * k64K) == 3);
        CHECK(hr(fx.vcall(queue, 31, {I(1), I(0), I(0), I(0), I(0)})) == dg::kEInvalidArg);
        fx.vcall(buf, Slot::Release);
        fx.vcall(queue, Slot::Release);
        CHECK(fx.vcall(device, Slot::Release) == 0);
    }
    {
        // Windows reserved after the device exists.
        sc::xmem_reset_for_testing();
        xw::set_policy(xw::PlacementPolicy::PhantomGpuWindows);
        Fixture fx;
        const uint64_t device = fx.create_device();
        dg::Device* dev = host_device(device);
        void* cpu2 = pg6_xbox_XMemVirtualAlloc(nullptr, S, kType, 0x21u, 0x4u);
        void* gpu2 = pg6_xbox_XMemVirtualAlloc(nullptr, S, kType, 0x3u, 0x801u);
        const uint64_t C = playgta6::os::win32::get_vmem_arena()->guest_address(cpu2);
        const uint64_t G = playgta6::os::win32::get_vmem_arena()->guest_address(gpu2);
        CHECK(dev->cpu_address_for_gpu(G + 8) == C + 8);
        CHECK(fx.vcall(device, Slot::Release) == 0);
    }
    sc::xmem_reset_for_testing();
}

}  // namespace

int main() {
    test_export_registration();
    test_create_device();
    test_queue_allocator_fence();
    test_descriptor_heaps();
    test_root_signatures_and_blobs();
    test_queries_and_footprints();
    test_unknown_slots_and_private_data();
    test_resources();
    test_committed_render_targets_through_xmem_hooks();
    test_frame_end_to_end();
    test_unresolved_shaders_and_indexed_draws();
    test_copies_and_barriers();
    test_concurrent_recording();
    test_compute_views_and_public_list();
    test_xbox_inlined_list_fields_and_staged_pso();
    test_xbox_frame_loop_present_and_pacing();
    test_frame_pacer_unit();
    test_xbox_pipeline_streams_and_overrides();
    test_xbox_native_pipelines();
    test_xbox_device_queue_and_list_slots();
    test_xmem_window_alias_wiring();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_d3d12_xs_exports: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_d3d12_xs_exports: all tests passed\n");
    return 0;
}
