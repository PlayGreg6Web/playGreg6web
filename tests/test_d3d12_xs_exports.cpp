// d3d12_xs.dll exports and guest D3D12 objects (G24), driven the way recompiled code drives them:
// a hand-written "lifted" caller sets up Win64 arguments, loads `[this]` and calls a vtable slot
// (or a flat export) through __playgta6_indirect_call. Synthetic data only.

#include <atomic>
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
#include "gpu/d3d12_xs_exports.h"
#include "guest/com_bridge.h"
#include "guest/guest_runtime.h"
#include "os/shim_exports.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"

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

const LiftedFunctionEntry kFunctions[] = {{kVCall, sub_vcall}};

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
        t_spec = VCallSpec{};
        t_spec.slot = slot;
        t_spec.args.push_back(I(obj));
        t_spec.args.insert(t_spec.args.end(), args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
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
    CHECK(hr(fx.vcall(queue, Slot::QueueWait, {I(fence), I(25)})) == 0);
    // A non-fence interface as fence argument is rejected.
    CHECK(hr(fx.vcall(queue, Slot::QueueSignal, {I(allocator), I(1)})) == dg::kEInvalidArg);
    pg6_win32_close_handle(event);

    // PresentX with planes ends the frame: BeginFrame (first packet of the frame) + Present.
    fx.captured.clear();
    CHECK(hr(fx.vcall(queue, Slot::PresentX, {I(1), I(0), I(0)})) == 0);
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

    // Struct returns: the executable passes the result pointer in rcx and `this` in rdx; the MSVC
    // member order (this, result) works too.
    const uint64_t result_buf = put<uint64_t>(0);
    CHECK(fx.sret_call(heap, Slot::GetCPUStart, result_buf) == result_buf);
    const uint64_t cpu_start = load64(result_buf);
    CHECK(cpu_start == host_heap->heap()->cpu_start().ptr);
    store64(result_buf, 0);
    CHECK(fx.sret_call(heap, Slot::GetCPUStart, result_buf, {}, /*this_first=*/true) == result_buf);
    CHECK(load64(result_buf) == cpu_start);
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
    CHECK(com::stub_hits(dg::kID3D12Device, 12) == 0);
    CHECK(hr(fx.vcall(device, 12)) == dg::kENotImpl);  // named stub (CreateCommandList, not in this layer yet)
    bool found = false;
    for (const std::string& d : com::diagnostics()) {
        found = found || (d.find("ID3D12Device") != std::string::npos && d.find("slot 99") != std::string::npos);
    }
    CHECK(found);
    bool named = false;
    for (const com::StubHit& h : com::stub_hit_report()) {
        named = named || (h.slot == 12 && h.method_name == "CreateCommandList");
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

}  // namespace

int main() {
    test_export_registration();
    test_create_device();
    test_queue_allocator_fence();
    test_descriptor_heaps();
    test_root_signatures_and_blobs();
    test_queries_and_footprints();
    test_unknown_slots_and_private_data();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_d3d12_xs_exports: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_d3d12_xs_exports: all tests passed\n");
    return 0;
}
