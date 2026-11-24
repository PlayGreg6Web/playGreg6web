#pragma once

// Guest-facing D3D12 objects for d3d12_xs.dll (G24), built on the guest COM bridge (O16,
// guest/com_bridge.h) and driving the G6 descriptor heaps / root signatures, the G7 PSO cache and
// the G20 state bridge.
//
// Guest code reaches every D3D12 method as `mov rax,[rcx]; call [rax+8*slot]` on interface pointers
// that d3d12_xs_exports.h hands out. Each interface below is a declarative com::declare_interface
// table in the slot order established from the real executable (static analysis, local only; only
// slot indices, argument counts, call-site counts and IIDs are recorded here):
//
// Evidence summary (game_scarlett_master.exe)
// -------------------------------------------
// * D3D12XboxCreateDevice is called once, with adapter = null, a 0x50-byte parameter block on the
//   stack and riid = IID_ID3D12Device11; the device lands in one global and is copied into 11 more.
//   Interface pointers were followed from there (globals, call arguments, return values, queue
//   wrapper objects): 42 distinct device slots in 54 functions (~93 call sites).
// * ID3D12Device: the public ID3D12Device order holds for slots 0..43. Confirmed by IID arguments
//   at the public creation slots (8 queue, 9 allocator, 14 descriptor heap, 16 root signature,
//   27 committed resource, 36 fence, 39 query heap, 41 command signature), by argument counts
//   (15 GetDescriptorHandleIncrementSize 2, 17 CreateConstantBufferView 3, 18 CreateShaderResourceView
//   4, 19 CreateUnorderedAccessView 5, 24 CopyDescriptorsSimple 5) and by the struct-returning shape
//   of 25 GetResourceAllocationInfo / 26 GetCustomHeapProperties. From slot 44 on the public order
//   does NOT hold: slot 48 returns an ID3D12CommandQueue from a 16-byte queue descriptor (4 calls),
//   slot 73 an ID3D12GraphicsCommandList from (16-byte descriptor, allocator, PSO, riid, out)
//   (3 calls), slots 55/57/67/86 return ID3D12PipelineState objects, 101 an ID3D12StateObject,
//   127 an ID3D12Resource with the CreateCommittedResource3 argument shape (heap properties, heap
//   flags, DESC1, initial layout, clear value, protected session, castable formats, riid, out) and
//   133 an ID3D12Resource from (address, desc, initial state, clear value, riid, out) (5 calls).
//   Slots 49, 50, 61, 62, 70, 71, 74, 78, 79, 80, 81, 108, 119, 132 are Xbox-specific with no public
//   counterpart; G27 established them (d3d12_xs_xbox.h). Unestablished slots stay logging stubs.
// * Struct returns: at slots 25/26 (and Xbox slot 108) the caller passes the hidden result pointer
//   in rcx and `this` in rdx (C-style), not `this` first as MSVC member functions do. Struct-
//   returning methods here accept both orders (SretMethods below) and count which one they saw.
//   An 8-byte struct (a descriptor handle) called with `this` in rcx is returned in rax: the
//   executable calls descriptor-heap slot 9 that way (rcx = this only, rax used).
// * Inlined accessors: 11 functions compare the byte at [device + 0x18] against 0 before writing
//   descriptors straight into descriptor memory; a non-zero byte makes them call
//   CopyDescriptorsSimple / CreateConstantBufferView instead. The device payload sets it to 1.
// * ID3D12CommandQueue: GetTimestampFrequency is slot 18 (out u64, result divided into a float),
//   Signal is slot 16 (fence, value), i.e. the public slots shifted by +2: two extra methods sit
//   between slots 3 and 15, so ExecuteCommandLists (public 10) is NOT established and stays a stub.
//   The submission thread executes command lists through slot 35 (this, count, lists, frame token,
//   flags) and presents through slot 32 (plane count, plane array, parameters); no other execute
//   slot is called. Slots 23/24/31: d3d12_xs_xbox.h.
// * ID3D12GraphicsCommandList: created by device slot 73 and immediately closed through slot 9
//   (public Close). G27 followed the lists through the context objects that own them: the Xbox
//   order (queries at 28..31, public 28..51 shifted by +4) and the inlined fast paths that write
//   command buffers directly are documented in d3d12_xs_xbox.h.
// * Allocators, descriptor heaps, fences, root signatures: only Release (slot 2) was observed on
//   tracked pointers; their method slots follow the public order (low confidence).
//
// Lifetime: bridge refcounting; children hold a com::Ref to their device. Unimplemented slots of
// every interface are logging stubs with SlotPolicy::ReturnNotImpl (first call logged with the slot
// index, every call counted in com::stub_hit_report()).

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_guest_structs.h"
#include "gpu/d3d12_xs_dump.h"
#include "gpu/d3d12_xs_xbox.h"
#include "gpu/descriptor_heap.h"
#include "gpu/pso_cache.h"
#include "gpu/shader_pack_index.h"
#include "gpu/stage_reflection.h"
#include "gpu/state_bridge.h"
#include "guest/com_bridge.h"

namespace playgta6::gpu::d3d12_guest {

namespace com = playgta6::guest::com;
namespace d3d = playgta6::gpu::d3d12_state;
using com::GuestPtr;
using com::Guid;
using com::HResult;
using com::InterfaceArg;
using com::OutPtr;

// ---------------------------------------------------------------------------------------------
// HRESULTs and IIDs (public values)
// ---------------------------------------------------------------------------------------------

inline constexpr HResult kSOk = 0;
inline constexpr HResult kSFalse = 1;
inline constexpr HResult kEInvalidArg = static_cast<HResult>(0x80070057u);
inline constexpr HResult kEOutOfMemory = static_cast<HResult>(0x8007000Eu);
inline constexpr HResult kEPointer = static_cast<HResult>(0x80004003u);
inline constexpr HResult kENoInterface = static_cast<HResult>(0x80004002u);
inline constexpr HResult kENotImpl = static_cast<HResult>(0x80004001u);
inline constexpr HResult kEFail = static_cast<HResult>(0x80004005u);
inline constexpr HResult kDxgiErrorNotFound = static_cast<HResult>(0x887A0002u);
inline constexpr HResult kDxgiErrorMoreData = static_cast<HResult>(0x887A0003u);

inline constexpr Guid IID_ID3D12Object = com::make_guid("c4fec28f-7966-4e95-9f94-f431cb56c3b8");
inline constexpr Guid IID_ID3D12DeviceChild = com::make_guid("905db94b-a00c-4140-9df5-2b64ca9ea357");
inline constexpr Guid IID_ID3D12Pageable = com::make_guid("63ee58fb-1268-4835-86da-f008ce62f0d6");
inline constexpr Guid IID_ID3D12Device = com::make_guid("189819f1-1db6-4b57-be54-1821339b85f7");
inline constexpr Guid IID_ID3D12Device11 = com::make_guid("5405c344-d457-444e-b4dd-2366e45aee39");
inline constexpr Guid IID_ID3D12CommandQueue = com::make_guid("0ec870a6-5d7e-4c22-8cfc-5baae07616ed");
inline constexpr Guid IID_ID3D12CommandAllocator = com::make_guid("6102dee4-af59-4b09-b999-b44d73f09b24");
inline constexpr Guid IID_ID3D12CommandList = com::make_guid("7116d91c-e7e4-47ce-b8c6-ec8168f437e5");
inline constexpr Guid IID_ID3D12GraphicsCommandList = com::make_guid("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455");
inline constexpr Guid IID_ID3D12DescriptorHeap = com::make_guid("8efb471d-616c-4f49-90f7-127bb763fa51");
inline constexpr Guid IID_ID3D12RootSignature = com::make_guid("c54a6b66-72df-4ee8-8be5-a946a1429214");
inline constexpr Guid IID_ID3D12Fence = com::make_guid("0a753dcf-c4d8-4b91-adf6-be5a60d95a76");
inline constexpr Guid IID_ID3D12Fence1 = com::make_guid("433685fe-e22b-4ca0-a8db-b5b4f4dd0e4a");
inline constexpr Guid IID_ID3D12QueryHeap = com::make_guid("0d9658ae-ed45-469e-a61d-970ec583cab4");
inline constexpr Guid IID_ID3D12CommandSignature = com::make_guid("c36a797c-ec80-4f0a-8985-a7b2475082d1");
inline constexpr Guid IID_ID3D12Resource = com::make_guid("696442be-a72e-4059-bc79-5b5c98040fad");
inline constexpr Guid IID_ID3D12Heap = com::make_guid("6b3b2502-6e51-45b3-90ee-9884265e8df3");
inline constexpr Guid IID_ID3D12PipelineState = com::make_guid("765a30f3-f624-4c6f-a828-ace948622445");
inline constexpr Guid IID_ID3DBlob = com::make_guid("8ba5fb08-5195-40e2-ac58-0d989c3a0102");

// ---------------------------------------------------------------------------------------------
// Struct-returning methods (hidden result pointer in either rcx or rdx, see the header comment)
// ---------------------------------------------------------------------------------------------

struct SretStats {
    uint64_t this_first = 0;    // rcx = this, rdx = result (MSVC member convention)
    uint64_t result_first = 0;  // rcx = result, rdx = this (observed in the executable)
};
SretStats sret_stats();
void reset_sret_stats();

namespace detail {
void note_sret(bool this_first);
bool resolve_sret_this(GuestState& state, const void* host_tag, com::detail::Resolved* out, bool* this_first);

template <typename Host, auto M, typename R, typename... A, size_t... I>
void invoke_sret(com::CallContext& ctx, Host* self, uint64_t hidden, std::tuple<A...>*, std::index_sequence<I...>) {
    R result{};
    com::detail::marshal_call<void, A...>(
        ctx, 2u, [&](auto&&... args) { result = (self->*M)(std::forward<decltype(args)>(args)...); },
        std::index_sequence<I...>{});
    if (!com::write_guest(hidden, result)) {
        com::detail::log_bad_pointer("hidden result pointer", hidden);
    }
    ctx.set_rax(hidden);
}

template <typename Host, auto M>
void sret_thunk(void* raw_state) {
    using Traits = com::detail::MemberFn<decltype(M)>;
    using R = typename Traits::Ret;
    using Params = typename Traits::Params;
    static_assert(std::is_class_v<R> && std::is_trivially_copyable_v<R>, "struct-returning methods only");
    GuestState& state = *static_cast<GuestState*>(raw_state);
    com::detail::Resolved r;
    bool this_first = true;
    resolve_sret_this(state, com::detail::type_tag<Host>(), &r, &this_first);
    com::GuestStateScope scope(state);
    const uint64_t self = this_first ? state.gpr[GuestReg::Rcx] : state.gpr[GuestReg::Rdx];
    const uint64_t hidden = this_first ? state.gpr[GuestReg::Rdx] : state.gpr[GuestReg::Rcx];
    com::CallContext ctx{state, self, r.object.get(), r.iface, state.gpr[GuestReg::Rsp]};
    if constexpr (sizeof(R) <= 8 && std::tuple_size_v<Params> == 0) {
        // An 8-byte struct (a descriptor handle) with `this` in rcx comes back in rax: the
        // executable calls GetCPUDescriptorHandleForHeapStart with rcx = this only and uses rax
        // (rdx holds whatever the caller left there). With the result pointer in rcx and `this`
        // in rdx it is a hidden result, as below.
        if (this_first) {
            R result = (static_cast<Host*>(r.host)->*M)();
            uint64_t bits = 0;
            std::memcpy(&bits, &result, sizeof(R));
            ctx.set_rax(bits);
            com::detail::pop_return(state);
            return;
        }
    }
    invoke_sret<Host, M, R>(ctx, static_cast<Host*>(r.host), hidden, static_cast<Params*>(nullptr),
                            std::make_index_sequence<std::tuple_size_v<Params>>{});
    com::detail::pop_return(state);
}
}  // namespace detail

template <typename Host>
struct Methods : com::Methods<Host> {
    template <auto M>
    static constexpr com::MethodDesc sret(const char* name) {
        return com::MethodDesc{name, &detail::sret_thunk<Host, M>};
    }
};

// ---------------------------------------------------------------------------------------------
// Host interfaces
// ---------------------------------------------------------------------------------------------

// ID3D12Object (slots 3..6).
struct IObjectHost {
    virtual HResult GetPrivateData(const Guid& guid, GuestPtr<uint32_t> data_size, uint64_t data) = 0;
    virtual HResult SetPrivateData(const Guid& guid, uint32_t data_size, uint64_t data) = 0;
    virtual HResult SetPrivateDataInterface(const Guid& guid, uint64_t unknown) = 0;
    virtual HResult SetName(uint64_t name) = 0;
};

// ID3D12DeviceChild (slot 7).
struct IDeviceChildHost : IObjectHost {
    virtual HResult GetDevice(const Guid& riid, OutPtr device) = 0;
};

class Device;

// Private data / name storage shared by every object (ID3D12Object semantics).
class PrivateData {
public:
    HResult get(const Guid& guid, GuestPtr<uint32_t> data_size, uint64_t data);
    HResult set(const Guid& guid, uint32_t data_size, uint64_t data);
    HResult set_interface(const Guid& guid, uint64_t unknown);
    HResult set_name(uint64_t name);
    std::u16string name() const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, std::vector<uint8_t>> entries_;  // key: raw GUID bytes
    std::u16string name_;
};

template <typename Iface>
class ObjectBase : public com::ComObject, public Iface {
public:
    HResult GetPrivateData(const Guid& guid, GuestPtr<uint32_t> size, uint64_t data) override {
        return data_.get(guid, size, data);
    }
    HResult SetPrivateData(const Guid& guid, uint32_t size, uint64_t data) override {
        return data_.set(guid, size, data);
    }
    HResult SetPrivateDataInterface(const Guid& guid, uint64_t unknown) override {
        return data_.set_interface(guid, unknown);
    }
    HResult SetName(uint64_t name) override { return data_.set_name(name); }
    std::u16string debug_name() const { return data_.name(); }

private:
    PrivateData data_;
};

template <typename Iface>
class ChildBase : public ObjectBase<Iface> {
public:
    explicit ChildBase(com::Ref<Device> device) : device_(std::move(device)) {}
    HResult GetDevice(const Guid& riid, OutPtr out) override;
    Device& device() const { return *device_; }

protected:
    com::Ref<Device> device_;
};

// ID3DBlob: 3 GetBufferPointer, 4 GetBufferSize.
struct IBlobHost {
    virtual uint64_t GetBufferPointer() = 0;
    virtual uint64_t GetBufferSize() = 0;
};

// ID3D12CommandAllocator: 8 Reset; Xbox 9 (callback, context) (see d3d12_xs_xbox.h).
struct ICommandAllocatorHost : IDeviceChildHost {
    virtual HResult Reset() = 0;
    virtual HResult SetAllocationCallback_unverified(uint64_t callback, uint64_t context) = 0;
};

// ID3D12Fence: 8 GetCompletedValue, 9 SetEventOnCompletion, 10 Signal; ID3D12Fence1: 11 GetCreationFlags.
struct IFenceHost : IDeviceChildHost {
    virtual uint64_t GetCompletedValue() = 0;
    virtual HResult SetEventOnCompletion(uint64_t value, uint64_t event_handle) = 0;
    virtual HResult Signal(uint64_t value) = 0;
    virtual uint32_t GetCreationFlags() = 0;
    // Xbox slot 12: () -> the guest address of a 64-bit word holding the completed value. The
    // executable's fence wait (0x144FB0AD0) reads the value through it, compares it with
    // UINT64_MAX (device removed) and with the target value, and waits while it is below.
    virtual uint64_t CompletedValueAddress() = 0;
};

// ID3D12DescriptorHeap: 8 GetDesc, 9 GetCPUDescriptorHandleForHeapStart, 10 GetGPUDescriptorHandleForHeapStart.
struct CpuHandleResult {
    uint64_t ptr;
};
struct GpuHandleResult {
    uint64_t ptr;
};
struct IDescriptorHeapHost : IDeviceChildHost {
    virtual GuestDescriptorHeapDesc GetDesc() = 0;
    virtual CpuHandleResult GetCPUDescriptorHandleForHeapStart() = 0;
    virtual GpuHandleResult GetGPUDescriptorHandleForHeapStart() = 0;
};

// ID3D12RootSignature, ID3D12QueryHeap, ID3D12CommandSignature: no methods beyond ID3D12DeviceChild.
struct IRootSignatureHost : IDeviceChildHost {};
struct IQueryHeapHost : IDeviceChildHost {};
struct ICommandSignatureHost : IDeviceChildHost {};

// ID3D12CommandQueue in the Xbox slot order (see the header comment).
struct ICommandQueueHost : IDeviceChildHost {
    virtual HResult Signal(InterfaceArg<IFenceHost> fence, uint64_t value) = 0;                // 16
    virtual HResult Wait(InterfaceArg<IFenceHost> fence, uint64_t value) = 0;                  // 17
    virtual HResult GetTimestampFrequency(GuestPtr<uint64_t> frequency) = 0;                   // 18
    virtual HResult PresentX(uint32_t plane_count, uint64_t planes, uint64_t parameters) = 0;  // 32
    virtual void ExecuteCommandListsX(com::CallContext& ctx) = 0;                              // 35
    // d3d12_xs_xbox.cpp
    virtual HResult SuspendX(uint32_t flags) = 0;                                              // 23
    virtual HResult ResumeX() = 0;                                                             // 24
    virtual HResult Slot31_UpdatePageMappings_unverified(com::CallContext& ctx) = 0;           // 31
};

// D3D12_RESOURCE_DESC1 as a struct-return value (72 bytes).
struct ResourceDesc1Result {
    uint8_t bytes[kGuestResourceDesc1Size];
};

// ID3D12Resource: 8 Map, 9 Unmap, 10 GetDesc, 11 GetGPUVirtualAddress, 12 WriteToSubresource,
// 13 ReadFromSubresource, 14 GetHeapProperties; ID3D12Resource2: 16 GetDesc1.
struct IResourceHost : IDeviceChildHost {
    virtual HResult Map(uint32_t subresource, uint64_t read_range, OutPtr data) = 0;
    virtual void Unmap(uint32_t subresource, uint64_t written_range) = 0;
    virtual GuestResourceDesc GetDesc() = 0;
    virtual uint64_t GetGPUVirtualAddress() = 0;
    virtual HResult WriteToSubresource(com::CallContext& ctx) = 0;
    virtual HResult ReadFromSubresource(com::CallContext& ctx) = 0;
    virtual HResult GetHeapProperties(GuestPtr<GuestHeapProperties> properties, GuestPtr<uint32_t> flags) = 0;
    virtual ResourceDesc1Result GetDesc1() = 0;
};

// ID3D12Heap: 8 GetDesc.
struct IHeapHost : IDeviceChildHost {
    virtual GuestHeapDesc GetDesc() = 0;
};

// ID3D12PipelineState: 8 GetCachedBlob.
struct IPipelineStateHost : IDeviceChildHost {
    virtual HResult GetCachedBlob(OutPtr blob) = 0;
};

// ID3D12GraphicsCommandList. The slot comments give the public order; the Xbox vtable
// (d3d12_guest_command_list.cpp, evidence in d3d12_xs_xbox.h) keeps 0..27, puts the query and
// predication methods at 28..31, shifts public 28..51 by +4 and has Xbox-only slots further up.
struct IGraphicsCommandListHost : IDeviceChildHost {
    virtual uint32_t GetType() = 0;                                                                  // 8
    virtual HResult Close() = 0;                                                                     // 9
    virtual HResult Reset(InterfaceArg<ICommandAllocatorHost> allocator,
                          InterfaceArg<IPipelineStateHost> initial_state) = 0;                       // 10
    // Xbox slot 139: (allocator, initial state, u64) -> HRESULT, a Reset with a third argument.
    virtual HResult ResetX(InterfaceArg<ICommandAllocatorHost> allocator, InterfaceArg<IPipelineStateHost> initial_state,
                           uint64_t extra) = 0;
    virtual void ClearState(InterfaceArg<IPipelineStateHost> state) = 0;                             // 11
    virtual void DrawInstanced(uint32_t vertices, uint32_t instances, uint32_t start_vertex,
                               uint32_t start_instance) = 0;                                         // 12
    virtual void DrawIndexedInstanced(uint32_t indices, uint32_t instances, uint32_t start_index,
                                      int32_t base_vertex, uint32_t start_instance) = 0;             // 13
    virtual void Dispatch(uint32_t x, uint32_t y, uint32_t z) = 0;                                   // 14
    virtual void CopyBufferRegion(InterfaceArg<IResourceHost> dst, uint64_t dst_offset,
                                  InterfaceArg<IResourceHost> src, uint64_t src_offset,
                                  uint64_t bytes) = 0;                                               // 15
    virtual void CopyTextureRegion(com::CallContext& ctx) = 0;                                       // 16
    // Xbox slot 85: (pDst, DstX, DstY, DstZ, pSrc, pSrcBox, flags), CopyTextureRegion + trailing flags.
    virtual void CopyTextureRegionX(com::CallContext& ctx) = 0;
    virtual void CopyResource(InterfaceArg<IResourceHost> dst, InterfaceArg<IResourceHost> src) = 0;  // 17
    virtual void ResolveSubresource(com::CallContext& ctx) = 0;                                      // 19
    virtual void IASetPrimitiveTopology(uint32_t topology) = 0;                                      // 20
    virtual void RSSetViewports(uint32_t count, uint64_t viewports) = 0;                             // 21
    virtual void RSSetScissorRects(uint32_t count, uint64_t rects) = 0;                              // 22
    virtual void OMSetBlendFactor(uint64_t factor) = 0;                                              // 23
    virtual void OMSetStencilRef(uint32_t ref) = 0;                                                  // 24
    virtual void SetPipelineState(InterfaceArg<IPipelineStateHost> state) = 0;                       // 25
    virtual void ResourceBarrier(uint32_t count, uint64_t barriers) = 0;                             // 26
    virtual void SetDescriptorHeaps(uint32_t count, uint64_t heaps) = 0;                             // 28
    virtual void SetComputeRootSignature(InterfaceArg<IRootSignatureHost> signature) = 0;            // 29
    virtual void SetGraphicsRootSignature(InterfaceArg<IRootSignatureHost> signature) = 0;           // 30
    virtual void SetComputeRootDescriptorTable(uint32_t index, uint64_t base) = 0;                   // 31
    virtual void SetGraphicsRootDescriptorTable(uint32_t index, uint64_t base) = 0;                  // 32
    virtual void SetComputeRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) = 0;   // 33
    virtual void SetGraphicsRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) = 0;  // 34
    virtual void SetComputeRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values,
                                              uint32_t offset) = 0;                                  // 35
    virtual void SetGraphicsRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values,
                                               uint32_t offset) = 0;                                 // 36
    virtual void SetComputeRootConstantBufferView(uint32_t index, uint64_t va) = 0;                  // 37
    virtual void SetGraphicsRootConstantBufferView(uint32_t index, uint64_t va) = 0;                 // 38
    virtual void SetComputeRootShaderResourceView(uint32_t index, uint64_t va) = 0;                  // 39
    virtual void SetGraphicsRootShaderResourceView(uint32_t index, uint64_t va) = 0;                 // 40
    virtual void SetComputeRootUnorderedAccessView(uint32_t index, uint64_t va) = 0;                 // 41
    virtual void SetGraphicsRootUnorderedAccessView(uint32_t index, uint64_t va) = 0;                // 42
    virtual void IASetIndexBuffer(uint64_t view) = 0;                                                // 43
    virtual void IASetVertexBuffers(uint32_t start, uint32_t count, uint64_t views) = 0;             // 44
    virtual void SOSetTargets(uint32_t start, uint32_t count, uint64_t views) = 0;                   // 45
    virtual void OMSetRenderTargets(uint32_t count, uint64_t rtvs, bool single_range, uint64_t dsv) = 0;  // 46
    virtual void ClearDepthStencilView(uint64_t dsv, uint32_t flags, float depth, uint8_t stencil,
                                       uint32_t rect_count, uint64_t rects) = 0;                     // 47
    virtual void ClearRenderTargetView(uint64_t rtv, uint64_t color, uint32_t rect_count,
                                       uint64_t rects) = 0;                                          // 48
    virtual void ClearUnorderedAccessView(com::CallContext& ctx) = 0;                                // 49, 50
    virtual void DiscardResource(InterfaceArg<IResourceHost> resource, uint64_t region) = 0;         // 51
    virtual void QueryOperation(com::CallContext& ctx) = 0;                                          // 52..54
    virtual void SetPredication(com::CallContext& ctx) = 0;                                          // 55
    virtual void DebugMarker(com::CallContext& ctx) = 0;                                             // 56..58
    // Xbox slots (see d3d12_xs_xbox.h).
    virtual void SetPipelineStateX(com::CallContext& ctx) = 0;  // Xbox 25: staged PSO, rdx fallback
    virtual void ExecuteIndirect(com::CallContext& ctx) = 0;    // Xbox 61
    virtual void OMSetDepthBounds(float min_depth, float max_depth) = 0;               // Xbox 70
    virtual void RSSetShadingRate(uint32_t rate, uint64_t combiners) = 0;              // Xbox 126
    virtual void RSSetShadingRateImage(InterfaceArg<IResourceHost> image) = 0;         // Xbox 127
    // Xbox 71: (address, u32 value, 0, 0): writes a 32-bit value to guest memory when the GPU
    // reaches this point (the title's per-frame marker array, 3 frames x 18 u32).
    virtual void WriteValue32(uint64_t address, uint32_t value, uint32_t a3, uint32_t a4) = 0;
    // Xbox 74: (address, 7, value, 1): writes a 64-bit value (a frame counter, a word the CPU
    // also writes with a 64-bit store) to guest memory when the GPU reaches this point.
    virtual void WriteValue64(uint64_t address, uint32_t a2, uint64_t value, uint32_t a4) = 0;
    // Xbox 146: enhanced barriers, (UINT32 NumBarrierGroups, const D3D12_BARRIER_GROUP*).
    virtual void Barrier(uint32_t group_count, uint64_t groups) = 0;
    // Xbox-only slots whose meaning is not established: recorded, no effect.
    virtual void note_xbox_slot(uint32_t slot) = 0;
    template <uint32_t Slot>
    void XboxSlot(com::CallContext&) {
        note_xbox_slot(Slot);
    }
};

struct HeapPropertiesResult {
    GuestHeapProperties props;
};

// ID3D12Device (public slots 0..43 plus the evidenced Xbox slots).
struct IDeviceHost : IObjectHost {
    virtual uint32_t GetNodeCount() = 0;                                                                  // 7
    virtual HResult CreateCommandQueue(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) = 0;  // 8
    virtual HResult CreateCommandAllocator(uint32_t type, const Guid& riid, OutPtr out) = 0;               // 9
    virtual HResult CheckFeatureSupport(uint32_t feature, uint64_t data, uint32_t size) = 0;               // 13
    virtual HResult CreateDescriptorHeap(const GuestDescriptorHeapDesc* desc, const Guid& riid, OutPtr out) = 0;  // 14
    virtual uint32_t GetDescriptorHandleIncrementSize(uint32_t type) = 0;                                 // 15
    virtual HResult CreateRootSignature(uint32_t node_mask, uint64_t blob, uint64_t length, const Guid& riid,
                                        OutPtr out) = 0;                                                    // 16
    virtual void CreateConstantBufferView(uint64_t desc, uint64_t dest) = 0;                              // 17
    virtual void CreateSampler(uint64_t desc, uint64_t dest) = 0;                                         // 22
    virtual void CopyDescriptors(com::CallContext& ctx) = 0;                                              // 23
    virtual void CopyDescriptorsSimple(uint32_t count, uint64_t dest, uint64_t src, uint32_t type) = 0;   // 24
    virtual GuestResourceAllocationInfo GetResourceAllocationInfo(uint32_t visible_mask, uint32_t count,
                                                                  uint64_t descs) = 0;                     // 25
    virtual HeapPropertiesResult GetCustomHeapProperties(uint32_t node_mask, uint32_t heap_type) = 0;     // 26
    virtual HResult MakeResident(uint32_t count, uint64_t objects) = 0;                                   // 34
    virtual HResult Evict(uint32_t count, uint64_t objects) = 0;                                          // 35
    virtual HResult CreateFence(uint64_t initial_value, uint32_t flags, const Guid& riid, OutPtr out) = 0;  // 36
    virtual HResult GetDeviceRemovedReason() = 0;                                                         // 37
    virtual void GetCopyableFootprints(com::CallContext& ctx) = 0;                                        // 38
    virtual HResult CreateQueryHeap(const GuestQueryHeapDesc* desc, const Guid& riid, OutPtr out) = 0;    // 39
    virtual HResult SetStablePowerState(bool enable) = 0;                                                 // 40
    virtual HResult CreateCommandSignature(const GuestCommandSignatureDesc* desc, uint64_t root_signature,
                                           const Guid& riid, OutPtr out) = 0;                               // 41
    virtual HResult CreateCommandQueueX(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) = 0;  // 48
    // Resources, PSOs, command lists, views (d3d12_guest_resources.cpp, d3d12_guest_command_list.cpp).
    virtual HResult CreateGraphicsPipelineState(uint64_t desc, const Guid& riid, OutPtr out) = 0;          // 10
    virtual HResult CreateComputePipelineState(uint64_t desc, const Guid& riid, OutPtr out) = 0;           // 11
    virtual HResult CreateCommandList(uint32_t node_mask, uint32_t type, InterfaceArg<ICommandAllocatorHost> allocator,
                                      InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid,
                                      OutPtr out) = 0;                                                     // 12
    virtual void CreateShaderResourceView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) = 0;  // 18
    virtual void CreateUnorderedAccessView(InterfaceArg<IResourceHost> resource, InterfaceArg<IResourceHost> counter,
                                           uint64_t desc, uint64_t dest) = 0;                             // 19
    virtual void CreateRenderTargetView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) = 0;  // 20
    virtual void CreateDepthStencilView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) = 0;  // 21
    virtual HResult CreateCommittedResource(com::CallContext& ctx) = 0;                                   // 27
    virtual HResult CreateHeap(const GuestHeapDesc* desc, const Guid& riid, OutPtr out) = 0;              // 28
    virtual HResult CreatePlacedResource(com::CallContext& ctx) = 0;                                      // 29
    virtual HResult CreateReservedResource(uint64_t desc, uint32_t state, uint64_t clear, const Guid& riid,
                                           OutPtr out) = 0;                                                 // 30
    virtual HResult CreateCommandListX(const GuestCommandListDescX* desc, InterfaceArg<ICommandAllocatorHost> allocator,
                                       InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid,
                                       OutPtr out) = 0;                                                    // 73
    virtual HResult CreateCommittedResource3(com::CallContext& ctx) = 0;                                  // 127
    virtual HResult CreatePlacedResourceX(com::CallContext& ctx) = 0;                                     // 133
    // Xbox slots (d3d12_xs_xbox.cpp; evidence and confidence in d3d12_xs_xbox.h).
    virtual HResult Slot49_SetFeatureToggle_unverified(uint32_t feature, uint32_t value) = 0;           // 49
    virtual HResult Slot50_RegisterRemovalCallbacks_unverified(uint64_t a, uint64_t b, uint64_t c) = 0;  // 50
    virtual HResult CreatePipelineStateWithOverrides(InterfaceArg<IPipelineStateHost> base, uint32_t count,
                                                     uint64_t overrides, const Guid& riid, OutPtr out) = 0;  // 55
    virtual HResult CreateNativePipelineState(uint64_t desc, const Guid& riid, OutPtr out) = 0;          // 57
    virtual HResult Slot61_unverified(uint64_t a, uint64_t b) = 0;                                       // 61
    virtual HResult CreateComputePipelineStateX(com::CallContext& ctx) = 0;                              // 62
    virtual HResult CreatePipelineStateFromBlob_unverified(com::CallContext& ctx) = 0;                   // 67
    virtual void CreateShaderResourceViewAtAddress(uint64_t resource_desc, uint64_t view_desc, uint64_t dest) = 0;  // 70
    virtual void CreateUnorderedAccessViewAtAddress(uint64_t resource_desc, uint64_t view_desc,
                                                    uint64_t dest) = 0;                                  // 71
    virtual HResult RegisterPagePool_unverified(uint64_t address, uint32_t pages, GuestPtr<uint64_t> handle) = 0;  // 74
    virtual HResult SetFrameIntervalX(uint64_t output, uint32_t interval_us, uint32_t period, uint32_t flags) = 0;  // 78
    virtual HResult ScheduleFrameEventX(uint32_t type, uint32_t offset_us, uint64_t reserved, uint32_t flags) = 0;  // 79
    virtual HResult WaitFrameEventX(uint32_t type, uint32_t timeout_ms, uint64_t reserved, uint32_t flags,
                                    GuestPtr<uint64_t> token) = 0;                                       // 80
    virtual HResult QueryFrameStatistics_unverified(uint64_t token, uint32_t flags, GuestPtr<uint32_t> count,
                                                    uint64_t entries) = 0;                               // 81
    virtual HResult CreatePipelineStateFromStream(uint64_t stream_desc, const Guid& riid, OutPtr out) = 0;  // 86
    virtual HResult CreateStateObject(uint64_t desc, const Guid& riid, OutPtr out) = 0;                  // 101
    virtual GuestResourceAllocationInfo GetResourceAllocationInfo1(uint32_t visible_mask, uint32_t count, uint64_t descs,
                                                                   uint64_t infos) = 0;                  // 108
    virtual HResult Slot119_unverified(uint64_t buffer, uint32_t size, uint32_t flags) = 0;              // 119
    virtual HResult Slot132_unverified(uint32_t value) = 0;                                              // 132
};

// ---------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------

// Where packets go (default: playgta6::os::wgpu_submit). Called with the device's submit mutex held.
using PacketSink = std::function<void(const uint8_t* data, size_t size)>;

// Pluggable detiling callback for console-tiled guest textures (G49b / G49a hook).
// Given raw guest bytes (`src`, `src_bytes`) in console layout `req.layout` (e.g. 0x100 | swizzle),
// writes the linear rows of subresource (`req.mip`, `req.array_slice`) with `req.dst_row_pitch`
// bytes per row into `dst` (`dst_bytes`). `src` is the whole surface (every mip and slice) from
// the resource's base address. Returns true on success, false if unsupported.
struct TextureDetileRequest {
    uint32_t dxgi_format = 0;
    uint32_t width = 0;   // mip 0
    uint32_t height = 0;
    uint32_t depth = 1;   // 3D depth (1 for 1D/2D)
    uint32_t mip_levels = 1;
    uint32_t layout = 0;
    uint32_t block_bytes = 0;
    uint32_t block_width = 1;
    uint32_t block_height = 1;
    uint32_t dst_row_pitch = 0;
    uint32_t array_size = 1;   // 1D/2D array slices
    uint32_t mip = 0;          // subresource to write
    uint32_t array_slice = 0;
};
using DetileFn = std::function<bool(const TextureDetileRequest& req,
                                    const uint8_t* src,
                                    size_t src_bytes,
                                    uint8_t* dst,
                                    size_t dst_bytes)>;

// Dirty-tracking state and layout metadata for a guest-memory-backed GPU texture (G49b).
struct TrackedTexture {
    uint32_t resource_id = 0;
    uint64_t gpu_va = 0;
    uint32_t dxgi_format = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t layout = 0;
    uint32_t dimension = 0;     // ResourceDimension (2=1D, 3=2D, 4=3D)
    uint32_t sample_count = 1;
    uint64_t last_upload_hash = 0;  // whole source surface
    bool ever_uploaded = false;
    // Per subresource (mip + array_slice * mip_levels): hash of the last uploaded linear texels,
    // 0 = never uploaded (the GPU texture is still zero there).
    std::vector<uint64_t> subresource_hashes;
};

struct DeviceOptions {
    PacketSink sink;                    // empty: os::wgpu_submit
    bridge::BridgeOptions bridge;       // id_allocator is replaced by the device's allocator
    uint64_t timestamp_frequency = 1000000000ull;
    // Xbox slot 57: index the console-native pipeline's stages, then fail the call (E_NOTIMPL) so
    // the title creates the pipeline from DXIL through slot 86 when it is requested (d3d12_xs_xbox.h,
    // "Native pipelines and their DXIL"). On by default. This deliberately differs from a console,
    // which creates these pipelines: the fallback is the title's own code path (the same one it
    // takes for compute when slot 67 is refused), and only it carries the public state, input
    // layouts and shader signatures the WebGPU side needs. Off: a native pipeline state is
    // created (unresolved; its draws are skipped). Boot harness: --native-pipelines turns it off.
    bool refuse_native_pipelines = true;
    // Pipeline dump directory (d3d12_xs_dump.h); empty: no dump.
    std::string dump_dir;
    // Keys of the WGSL pack the WebGPU side loads (shader_pack_index.h; boot harness:
    // --shader-pack). Graphics pipelines whose stages all have a key are resolved: see
    // Device::resolve_with_pack. Null: no pack, every pipeline stays unresolved.
    std::shared_ptr<const ShaderPackIndex> shader_pack;
    // Reflection of the pack's stages (gpu/stage_reflection.h; boot harness: <pack>.reflect.json).
    // Pack-resolved pipelines whose stages all have an entry get their vertex attributes and
    // pipeline layout from it (join_vertex_inputs / join_pipeline_layout); a pipeline that cannot be
    // joined stays unresolved (its draws are skipped), with one report per pipeline. Stages without
    // an entry keep layout_from_root_signature and element i -> @location(i).
    std::shared_ptr<const StageReflectionIndex> stage_reflection;
    // Optional detiling hook for tiled console layouts (layout >= 0x101).
    DetileFn detile_texture;
};

struct DeviceStats {
    uint64_t resources = 0;
    uint64_t heaps = 0;
    uint64_t pipeline_states = 0;
    uint64_t command_lists = 0;
    uint64_t upload_bytes = 0;        // WriteBuffer / WriteTexture payload bytes emitted from guest memory
    uint64_t textures_uploaded = 0;   // WriteTexture packets emitted from guest texture memory
    uint64_t skipped_draws = 0;       // draws skipped because the bound PSO has no shader module
    uint64_t unsupported_copies = 0;  // GPU-to-GPU copies, resolves, ... (no packet; diagnosed)
    uint64_t queues = 0;
    uint64_t command_allocators = 0;
    uint64_t descriptor_heaps = 0;
    uint64_t root_signatures = 0;
    uint64_t fences = 0;
    uint64_t presents = 0;
    uint64_t submissions = 0;
    uint64_t bad_arguments = 0;  // malformed guest structs / pointers (diagnosed, E_INVALIDARG)
};

class Resource;
class Heap;
class CommandList;
struct GraphicsPsoSource;

class Device final : public ObjectBase<IDeviceHost> {
public:
    static constexpr const char* kComClassName = "ID3D12Device";
    // Guest payload (after the vtable pointer, at device + 0x10): the byte at device + 0x18 is read
    // by inlined accessors in the executable (see the header comment).
    static constexpr uint64_t kComPayloadSize = 16;
    static constexpr uint64_t kInlineApiFlagOffset = 0x18;

    explicit Device(DeviceOptions options = {});
    ~Device() override;

    // Must run once right after create_object (writes the guest payload).
    void initialize_guest_payload();

    // ---- IDeviceHost ----
    uint32_t GetNodeCount() override { return 1; }
    HResult CreateCommandQueue(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) override;
    HResult CreateCommandAllocator(uint32_t type, const Guid& riid, OutPtr out) override;
    HResult CheckFeatureSupport(uint32_t feature, uint64_t data, uint32_t size) override;
    HResult CreateDescriptorHeap(const GuestDescriptorHeapDesc* desc, const Guid& riid, OutPtr out) override;
    uint32_t GetDescriptorHandleIncrementSize(uint32_t type) override;
    HResult CreateRootSignature(uint32_t node_mask, uint64_t blob, uint64_t length, const Guid& riid,
                                OutPtr out) override;
    void CreateConstantBufferView(uint64_t desc, uint64_t dest) override;
    void CreateSampler(uint64_t desc, uint64_t dest) override;
    void CopyDescriptors(com::CallContext& ctx) override;
    void CopyDescriptorsSimple(uint32_t count, uint64_t dest, uint64_t src, uint32_t type) override;
    GuestResourceAllocationInfo GetResourceAllocationInfo(uint32_t visible_mask, uint32_t count,
                                                          uint64_t descs) override;
    HeapPropertiesResult GetCustomHeapProperties(uint32_t node_mask, uint32_t heap_type) override;
    HResult MakeResident(uint32_t, uint64_t) override { return kSOk; }
    HResult Evict(uint32_t, uint64_t) override { return kSOk; }
    HResult CreateFence(uint64_t initial_value, uint32_t flags, const Guid& riid, OutPtr out) override;
    HResult GetDeviceRemovedReason() override { return kSOk; }
    void GetCopyableFootprints(com::CallContext& ctx) override;
    HResult CreateQueryHeap(const GuestQueryHeapDesc* desc, const Guid& riid, OutPtr out) override;
    HResult SetStablePowerState(bool) override { return kSOk; }
    HResult CreateCommandSignature(const GuestCommandSignatureDesc* desc, uint64_t root_signature,
                                   const Guid& riid, OutPtr out) override;
    HResult CreateCommandQueueX(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) override;
    HResult CreateGraphicsPipelineState(uint64_t desc, const Guid& riid, OutPtr out) override;
    HResult CreateComputePipelineState(uint64_t desc, const Guid& riid, OutPtr out) override;
    HResult CreateCommandList(uint32_t node_mask, uint32_t type, InterfaceArg<ICommandAllocatorHost> allocator,
                              InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid, OutPtr out) override;
    void CreateShaderResourceView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) override;
    void CreateUnorderedAccessView(InterfaceArg<IResourceHost> resource, InterfaceArg<IResourceHost> counter,
                                   uint64_t desc, uint64_t dest) override;
    void CreateRenderTargetView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) override;
    void CreateDepthStencilView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) override;
    HResult CreateCommittedResource(com::CallContext& ctx) override;
    HResult CreateHeap(const GuestHeapDesc* desc, const Guid& riid, OutPtr out) override;
    HResult CreatePlacedResource(com::CallContext& ctx) override;
    HResult CreateReservedResource(uint64_t desc, uint32_t state, uint64_t clear, const Guid& riid,
                                   OutPtr out) override;
    HResult CreateCommandListX(const GuestCommandListDescX* desc, InterfaceArg<ICommandAllocatorHost> allocator,
                               InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid, OutPtr out) override;
    HResult CreateCommittedResource3(com::CallContext& ctx) override;
    HResult CreatePlacedResourceX(com::CallContext& ctx) override;
    // Xbox slots (d3d12_xs_xbox.cpp).
    HResult Slot49_SetFeatureToggle_unverified(uint32_t feature, uint32_t value) override;
    HResult Slot50_RegisterRemovalCallbacks_unverified(uint64_t a, uint64_t b, uint64_t c) override;
    HResult CreatePipelineStateWithOverrides(InterfaceArg<IPipelineStateHost> base, uint32_t count, uint64_t overrides,
                                             const Guid& riid, OutPtr out) override;
    HResult CreateNativePipelineState(uint64_t desc, const Guid& riid, OutPtr out) override;
    HResult Slot61_unverified(uint64_t a, uint64_t b) override;
    HResult CreateComputePipelineStateX(com::CallContext& ctx) override;
    HResult CreatePipelineStateFromBlob_unverified(com::CallContext& ctx) override;
    void CreateShaderResourceViewAtAddress(uint64_t resource_desc, uint64_t view_desc, uint64_t dest) override;
    void CreateUnorderedAccessViewAtAddress(uint64_t resource_desc, uint64_t view_desc, uint64_t dest) override;
    HResult RegisterPagePool_unverified(uint64_t address, uint32_t pages, GuestPtr<uint64_t> handle) override;
    HResult SetFrameIntervalX(uint64_t output, uint32_t interval_us, uint32_t period, uint32_t flags) override;
    HResult ScheduleFrameEventX(uint32_t type, uint32_t offset_us, uint64_t reserved, uint32_t flags) override;
    HResult WaitFrameEventX(uint32_t type, uint32_t timeout_ms, uint64_t reserved, uint32_t flags,
                            GuestPtr<uint64_t> token) override;
    HResult QueryFrameStatistics_unverified(uint64_t token, uint32_t flags, GuestPtr<uint32_t> count,
                                            uint64_t entries) override;
    HResult CreatePipelineStateFromStream(uint64_t stream_desc, const Guid& riid, OutPtr out) override;
    HResult CreateStateObject(uint64_t desc, const Guid& riid, OutPtr out) override;
    GuestResourceAllocationInfo GetResourceAllocationInfo1(uint32_t visible_mask, uint32_t count, uint64_t descs,
                                                           uint64_t infos) override;
    HResult Slot119_unverified(uint64_t buffer, uint32_t size, uint32_t flags) override;
    HResult Slot132_unverified(uint32_t value) override;

    // ---- Xbox frame model (d3d12_xs_xbox.h) ----
    xbox::FramePacer& frame_pacer() { return pacer_; }
    // Wakes every WaitFrameEventX and makes later ones return at once (shutdown, tests).
    void shutdown_frame_events() { pacer_.shutdown(); }
    void count(xbox::Counter c, uint64_t n = 1) { xbox_counters_[static_cast<size_t>(c)].fetch_add(n); }
    uint64_t counter(xbox::Counter c) const { return xbox_counters_[static_cast<size_t>(c)].load(); }
    // Last frame token passed to ExecuteCommandLists / a present plane (0 = none).
    std::atomic<uint64_t> last_execute_token{0};
    std::atomic<uint64_t> last_present_token{0};

    // ---- Pipeline-state creation from host-side descriptions (also used by the Xbox slots) ----
    HResult create_graphics_pso(const GuestGraphicsPipelineStateDesc& g, const Guid& riid, OutPtr out);
    HResult create_graphics_pso_from(GraphicsPsoSource source, const Guid& riid, OutPtr out);
    HResult create_compute_pso(const GuestComputePipelineStateDesc& g, const Guid& riid, OutPtr out);
    // A pipeline state that binds as "unresolved" (draws skipped) for pipelines the layer cannot
    // build (mesh shaders, ...): handed out instead of a null pointer the caller would dereference.
    HResult create_placeholder_pso(const std::string& reason, const Guid& riid, OutPtr out);
    // A console-native graphics pipeline (Xbox slot 57, d3d12_xs_xbox.h): unresolved, with one
    // identity per stage (native_stage_identity).
    HResult create_native_pso(const xbox::NativePipeline& pipeline, const Guid& riid, OutPtr out);

    // ---- Shaders ----
    // Shader identity (shader_identity(): FNV-1a 64 over the DXIL container bytes the PSO desc passes)
    // -> CreateShaderModule id. PSOs created before their modules are registered stay unresolved.
    void register_shader_module(uint64_t identity, uint32_t module_id);
    enum class ShaderStage : uint8_t { Vertex, Pixel, Compute };
    void note_unresolved_shader(uint64_t identity, ShaderStage stage);
    std::vector<uint64_t> unresolved_shaders() const;
    // Distinct unresolved identities by the stage they were first used at (vertex, pixel, compute).
    std::array<uint32_t, 3> unresolved_shader_stages() const;
    // Root signatures created, by blob version (1.0, 1.1, 1.2).
    std::array<uint32_t, 3> root_signature_versions() const;
    // Console-native stages (Xbox slot 57): distinct identities are counted per stage (vertex,
    // pixel) instead of being logged one by one; the first native pipeline is logged once. They are
    // not in unresolved_shaders() (that list is the DXIL the shader track can translate).
    void note_native_stages(uint64_t vertex_identity, uint64_t pixel_identity);
    std::array<uint32_t, 2> native_stage_counts() const;
    // A console-native stage as the guest layer sees it (Xbox slot 57).
    struct NativeStageRecord {
        uint64_t identity = 0;       // native_stage_identity: code then metadata
        uint64_t code_identity = 0;  // FNV-1a 64 over the code range only
        uint32_t code_bytes = 0;
        uint32_t metadata_bytes = 0;
    };
    using DxilDigest = std::array<uint8_t, 16>;
    // Reads a stage's code and metadata (false: unreadable, diagnosed). With `index`, the stage is
    // also indexed by the 16-byte values at its metadata's digest positions (d3d12_xs_xbox.h,
    // "Native pipelines and their DXIL") so find_native_stage can join a DXIL container to it.
    // `pixel` / `layout_b` only label the stage in the pipeline dump.
    bool read_native_stage(const xbox::NativeStage& stage, bool index, NativeStageRecord* out, bool pixel = false,
                           bool layout_b = false);
    // The indexed native stage whose metadata carries `digest` (a DXIL container's HASH digest).
    bool find_native_stage(const DxilDigest& digest, NativeStageRecord* out) const;
    size_t native_index_size() const;
    // Pipeline dump (DeviceOptions::dump_dir), null when off. flush_pipeline_dump rewrites the
    // manifest (also done every 60 presents and when the device is destroyed).
    PipelineDump* pipeline_dump() const { return dump_.get(); }
    // A GPU-only buffer owned by the runtime (rectangle-list expansion); empty on failure.
    com::Ref<Resource> create_internal_buffer(uint64_t size);
    // Uniform bindings at offsets WebGPU cannot bind (BridgeOptions::relocate_uniform): the bytes
    // at [va, va + size) (CPU-visible memory only) get a 256-byte aligned home in runtime-owned
    // buffers, refreshed from guest memory at every submission (flush_cpu_buffers) when changed.
    bool relocate_uniform(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset);
    size_t relocated_uniform_count() const;
    // Storage buffer views at unaligned offsets or address-only GPU VAs (G51): the bytes at
    // [va, va + size) (readable guest/alias memory only) get a 256-byte aligned home (or a
    // dedicated offset-0 buffer when `require_zero_offset` is true, e.g. element stride does not
    // divide 256) in runtime-owned Storage|CopyDst buffers registered with the state bridge,
    // uploaded immediately and refreshed on flush_cpu_buffers when guest bytes change.
    bool relocate_storage_buffer(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset,
                                 bool require_zero_offset = false);
    size_t relocated_storage_count() const;
    // Index / vertex buffer views at address-only GPU VAs (G58: dynamic guest-RAM ring buffers
    // with no ID3D12Resource): the bytes at [va, va + size) (readable guest/alias memory only, at
    // most 16 MiB) get a dedicated Vertex|Index|CopyDst buffer at offset 0, uploaded immediately
    // and refreshed on flush_cpu_buffers when guest bytes change (FNV-1a 64). A request inside an
    // existing relocation's range (a ring buffer advancing its write pointer) reuses that buffer
    // at offset va - base when the offset is 4-byte aligned. Bounded LRU (entries and bytes).
    bool relocate_ia_buffer(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset);
    size_t relocated_ia_count() const;
    // Shader pack resolution (DeviceOptions::shader_pack). Each stage of a graphics pipeline is
    // looked up by the code-only identity of the console-native stage its DXIL joins to, else by
    // its DXIL identity. When every stage has a key, the stages get modules (one CreateShaderModule
    // per key, code = pack_key_code(key), resolved by the WebGPU side from its pack) and, unless
    // one is registered, a pipeline layout derived from the root signature (one group; bindings
    // numbered from 0 in root-parameter order: table ranges descriptor by descriptor, root
    // descriptors and root constants one each, then static samplers). Returns the keys used
    // (0 = stage absent); false when a stage has no key.
    bool resolve_with_pack(GraphicsPsoSource& source, uint64_t* vs_key, uint64_t* ps_key);
    // Reflection join failures reported so far (once per pipeline: stage keys + root signature).
    size_t reflection_join_failures() const;
    void flush_pipeline_dump();
    bool shader_known(uint64_t identity) const;
    // Bytecode length of each shader identity seen in a pipeline state (0 if unknown).
    void note_shader_size(uint64_t identity, uint64_t bytes, int kind = -1);
    uint64_t shader_size(uint64_t identity) const;
    // The shader kind the DXIL container declares (program header; 0 pixel, 1 vertex, 5 compute,
    // ...), -1 if unknown.
    int shader_kind(uint64_t identity) const;
    // The first skipped draw of each pipeline state is described once (counts, formats, sizes):
    // claim_skipped_draw_report is true exactly once per pipeline state.
    bool claim_skipped_draw_report(const void* pipeline_state);
    void add_skipped_draw_report(const std::string& text);
    std::vector<std::string> skipped_draw_reports() const;

    // ---- Resources ----
    // GPU virtual address range for memory without guest backing (default heaps, reserved
    // resources): kFakeGpuVaBase upwards, 64 KiB aligned, never reused.
    static constexpr uint64_t kFakeGpuVaBase = 0x00007E0000000000ull;
    uint64_t allocate_gpu_va(uint64_t size);
    // CPU-visible buffers whose guest memory is compared with a shadow copy at every
    // ExecuteCommandLists (changed 4 KiB pages become WriteBuffer packets).
    void track_cpu_buffer(Resource* resource);
    void untrack_cpu_buffer(Resource* resource);
    void flush_cpu_buffers(CommandEncoder& encoder);
    // Dirty-tracked guest texture memory upload (G49b, G64): tracks FNV-1a 64 hashes of the whole
    // source surface and of each subresource, plus layout metadata, for guest-memory-backed GPU
    // textures, and emits one WriteTexture per changed subresource (every mip and array slice of a
    // console-tiled surface) when bound SRV memory changes.
    void track_guest_texture(const TrackedTexture& tex);
    void untrack_guest_texture(uint32_t resource_id);
    bool tracked_texture_info(uint32_t resource_id, TrackedTexture* out) const;
    void set_detile_fn(DetileFn fn);
    bool sync_bound_texture_memory(uint32_t resource_id, CommandEncoder& encoder);
    // Marks a texture as the destination of a recorded copy (CopyTextureRegion). The copy writes
    // texels in API coordinates and defines the contents from then on; the guest memory at the
    // texture's address is no longer uploaded (or de-tiled) over it.
    void note_texture_copy_destination(uint32_t resource_id);
    // Creates (or returns a cached) WebGPU texture for a GPU VA that has no ID3D12Resource
    // (Xbox Slot-70/71 address views and raw GFX10 T# image descriptors in descriptor memory).
    // Registers TextureInfo in the state bridge and tracks guest memory for detiled WriteTexture
    // uploads on bind. Returns 0 when dimensions or format are unsupported.
    uint32_t ensure_address_texture(uint64_t gpu_va, uint32_t dxgi_format, uint32_t width, uint32_t height,
                                    uint32_t depth, uint32_t mip_levels, uint32_t d3d12_layout,
                                    GpuTextureDimension dim, bool uav);
    // Designates a texture as the presented back buffer: render passes that target it write the
    // canvas texture (kGpuSwapchainTextureId). PresentX calls it for plane 0's resource the first
    // time that resource is presented (d3d12_xs_xbox.h). False if `resource` is not a 2D texture.
    bool set_back_buffer(uint64_t resource_interface_pointer);
    // GPU address windows aliasing CPU memory (the executable reserves a CPU window and a GPU window
    // of equal size and maps the same pages into both; d3d12_xs_xbox.h). A buffer created at a GPU
    // address inside a registered window reads its contents from the matching CPU address.
    void register_gpu_alias(uint64_t gpu_base, uint64_t cpu_base, uint64_t size);
    uint64_t cpu_address_for_gpu(uint64_t gpu_va) const;  // gpu_va itself outside every window
    void note_upload(uint64_t bytes) { upload_bytes_.fetch_add(bytes, std::memory_order_relaxed); }
    void note_texture_upload(uint64_t count = 1) { textures_uploaded_.fetch_add(count, std::memory_order_relaxed); }
    void note_skipped_draw() { skipped_draws_.fetch_add(1, std::memory_order_relaxed); }
    void note_unsupported_copy(const std::string& message);
    // True when a texture of `dxgi_format` gets a substitute WebGPU format whose texels differ from
    // the guest's (BC without texture-compression-bc, packed 16-bit RGBA): the guest's bytes cannot
    // be uploaded as they are, and no CPU conversion exists yet, so uploads to it are skipped.
    bool upload_needs_conversion(uint32_t dxgi_format) const;

    // ---- Descriptor words in descriptor memory (bridge/descriptor_words.h, G48) ----
    // After a registry write at `cpu_handle`: writes the words the console's D3D12 leaves in the
    // slot (heaps without guest memory are skipped). The title copies whole slots itself when it
    // fills shader-visible tables, so the words must be in its memory.
    void store_descriptor_words(uint64_t cpu_handle);
    // Writes `words` (one slot of the heap's increment) at `cpu_handle` (Xbox slots 70 / 71).
    void store_descriptor_words(uint64_t cpu_handle, const uint32_t* words, size_t bytes);
    // CopyDescriptors*: the console copies the slots' memory along with the descriptors.
    void copy_descriptor_memory(const std::vector<std::pair<uint64_t, uint32_t>>& dst_ranges,
                                const std::vector<std::pair<uint64_t, uint32_t>>& src_ranges, uint32_t type);
    // Slots written per outcome (index = bridge::WordsStatus).
    std::array<uint64_t, 7> descriptor_word_counts() const;
    // Root-table slots read back from descriptor memory, per binding kind (0 CBV, 1 SRV, 2 UAV,
    // 3 sampler) and outcome (index = bridge::DecodeStatus). Ok slots bind; every other outcome binds
    // null and its first occurrence per kind is logged with the reason.
    using SlotReadCounts = std::array<std::array<uint64_t, 7>, 4>;
    SlotReadCounts descriptor_slot_reads() const;
    // Xbox slots 70 / 71 after parsing the view description (d3d12_xs_xbox.cpp).
    void view_at_address(bool uav, uint64_t resource_desc, const d3d::ViewDesc& desc, uint64_t va, uint64_t dest);
    uint64_t count_texture_view_at_address();

    // ---- Runtime side ----
    d3d::DescriptorHeapRegistry& heaps() { return heaps_; }
    d3d::ResourceStateRegistry& resource_states() { return resource_states_; }
    bridge::StateBridgeDevice& state_bridge() { return *bridge_; }
    uint32_t allocate_id() { return next_id_.fetch_add(1, std::memory_order_relaxed); }
    uint64_t timestamp_frequency() const { return options_.timestamp_frequency; }

    // Writes `bytes` to the sink as one submission (thread-safe, serialized).
    void submit(const uint8_t* data, size_t size);
    void submit(const CommandEncoder& encoder) { submit(encoder.data(), encoder.size()); }
    // Several byte ranges as one submission (no other submission in between).
    void submit_parts(const std::vector<std::pair<const uint8_t*, size_t>>& parts);
    // Present packet (and BeginFrame before the next frame's first submission).
    void present();
    void set_sink(PacketSink sink);

    DeviceStats stats() const;
    void note_bad_argument(const std::string& message);
    std::vector<std::string> diagnostics() const;

    // Child counters.
    std::atomic<uint64_t> queues{0}, allocators{0}, descriptor_heaps{0}, root_signatures{0}, fences{0};
    std::atomic<uint64_t> resources{0}, memory_heaps{0}, pipeline_states{0}, command_lists{0};

private:
    HResult create_queue(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out);

    DeviceOptions options_;
    d3d::DescriptorHeapRegistry heaps_;
    d3d::ResourceStateRegistry resource_states_;
    std::atomic<uint32_t> next_id_{1};
    std::unique_ptr<bridge::StateBridgeDevice> bridge_;

    mutable std::mutex submit_mutex_;
    PacketSink sink_;
    bool frame_open_ = false;
    uint64_t presents_ = 0;
    uint64_t submissions_ = 0;

    mutable std::mutex diag_mutex_;
    std::vector<std::string> diags_;
    std::atomic<uint64_t> bad_arguments_{0};

    HResult create_resource(const GuestResourceDesc& desc, const GuestHeapProperties& heap, uint32_t heap_flags,
                            uint32_t initial_state, com::Ref<Heap> placed_heap, uint64_t heap_offset,
                            uint64_t address, bool reserved, const Guid& riid, OutPtr out);
    HResult create_command_list(uint32_t type, InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid,
                                OutPtr out);

    mutable std::mutex shader_mutex_;
    std::vector<uint64_t> unresolved_shaders_;
    std::array<uint32_t, 3> unresolved_by_stage_{};
    std::array<uint32_t, 3> root_signature_versions_{};  // under shader_mutex_
    std::unordered_set<uint64_t> known_shaders_;
    std::unordered_set<uint64_t> native_stages_[2];  // under shader_mutex_ (vertex, pixel)
    struct DigestHash {
        size_t operator()(const DxilDigest& d) const {
            uint64_t a = 0, b = 0;
            std::memcpy(&a, d.data(), 8);
            std::memcpy(&b, d.data() + 8, 8);
            return static_cast<size_t>(a ^ (b * 0x9E3779B97F4A7C15ull));
        }
    };
    std::unordered_map<DxilDigest, NativeStageRecord, DigestHash> native_by_digest_;  // under shader_mutex_
    std::unordered_set<uint64_t> indexed_native_;                                     // under shader_mutex_
    std::unique_ptr<PipelineDump> dump_;
    struct RelocatedUniform {
        uint64_t va = 0;
        uint64_t size = 0;
        uint32_t buffer_id = 0;
        uint64_t offset = 0;
        std::vector<uint8_t> shadow;  // bytes last written (empty: never written)
    };
    mutable std::mutex relocation_mutex_;
    std::map<std::pair<uint64_t, uint64_t>, RelocatedUniform> relocations_;  // (va, size)
    struct RelocatedStorage {
        uint64_t va = 0;
        uint64_t size = 0;
        uint32_t buffer_id = 0;
        uint64_t offset = 0;
        uint64_t slot_capacity = 0;
        uint64_t last_hash = 0;
        uint64_t last_used_seq = 0;
        bool ever_uploaded = false;
        bool dedicated = false;
    };
    struct RelocatedStorageKey {
        uint64_t va = 0;
        uint64_t size = 0;
        bool dedicated = false;
        bool operator<(const RelocatedStorageKey& o) const noexcept {
            if (va != o.va) return va < o.va;
            if (size != o.size) return size < o.size;
            return dedicated < o.dedicated;
        }
    };
    mutable std::mutex storage_relocation_mutex_;
    std::map<RelocatedStorageKey, RelocatedStorage> storage_relocations_;
    std::vector<uint32_t> storage_relocation_buffers_;
    uint32_t storage_relocation_buffer_ = 0;
    uint64_t storage_relocation_used_ = 0;
    uint64_t storage_relocation_seq_ = 0;
    struct RelocatedIa {
        uint64_t buffer_bytes = 0;   // WebGPU buffer size (guest size rounded up to 4)
        uint32_t buffer_id = 0;
        uint64_t last_hash = 0;
        uint64_t last_used_seq = 0;
    };
    mutable std::mutex ia_relocation_mutex_;
    std::map<std::pair<uint64_t, uint64_t>, RelocatedIa> ia_relocations_;  // (base GPU VA, size)
    uint64_t ia_relocation_bytes_ = 0;
    uint64_t ia_relocation_seq_ = 0;
    mutable std::mutex pack_mutex_;
    std::unordered_map<uint64_t, uint32_t> pack_modules_;                 // pack key -> module id
    std::set<std::pair<uint64_t, uint64_t>> pack_layouts_;                // (vs, ps) identities
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> reflection_failures_;  // (vs key, ps key, root signature)
    uint32_t relocation_buffer_ = 0;  // current GPU buffer id (CreateBuffer packet only, no COM object)
    uint64_t relocation_used_ = 0;    // bytes used in relocation_buffer_
    std::atomic<uint64_t> dump_presents_{0};
    struct ShaderFacts {
        uint64_t bytes = 0;
        int kind = -1;
    };
    std::unordered_map<uint64_t, ShaderFacts> shader_sizes_;  // under shader_mutex_
    std::unordered_set<const void*> described_pipelines_;   // under shader_mutex_
    std::vector<std::string> skipped_draw_reports_;          // under shader_mutex_ (first 64)
    std::mutex va_mutex_;
    uint64_t next_fake_va_ = kFakeGpuVaBase;
    std::mutex tracked_mutex_;
    std::vector<Resource*> tracked_;
    mutable std::mutex tracked_textures_mutex_;
    std::unordered_map<uint32_t, TrackedTexture> tracked_textures_;
    std::unordered_set<uint32_t> copy_destinations_;  // under tracked_textures_mutex_
    std::unordered_set<uint32_t> conversion_skipped_;  // under tracked_textures_mutex_: reported once
    struct AddressTextureKey {
        uint64_t gpu_va = 0;
        uint32_t dxgi_format = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 1;
        uint32_t mip_levels = 1;
        uint32_t d3d12_layout = 0;
        uint8_t dim = 0;

        bool operator==(const AddressTextureKey& o) const noexcept {
            return gpu_va == o.gpu_va && dxgi_format == o.dxgi_format && width == o.width && height == o.height &&
                   depth == o.depth && mip_levels == o.mip_levels && d3d12_layout == o.d3d12_layout && dim == o.dim;
        }
    };
    struct AddressTextureKeyHash {
        size_t operator()(const AddressTextureKey& k) const noexcept {
            uint64_t h = uint64_t{0xcbf29ce484222325ull};
            auto mix = [&](uint64_t v) {
                h ^= v;
                h *= uint64_t{0x00000100000001b3ull};
            };
            mix(k.gpu_va);
            mix(k.dxgi_format);
            mix(k.width);
            mix(k.height);
            mix(k.depth);
            mix(k.mip_levels);
            mix(k.d3d12_layout);
            mix(k.dim);
            return static_cast<size_t>(h);
        }
    };
    mutable std::mutex address_textures_mutex_;
    std::unordered_map<AddressTextureKey, uint32_t, AddressTextureKeyHash> address_textures_;
    mutable std::mutex detile_mutex_;
    DetileFn detile_texture_;
    std::atomic<uint64_t> upload_bytes_{0};
    std::atomic<uint64_t> textures_uploaded_{0};
    std::array<std::atomic<uint64_t>, 7> descriptor_words_{};
    std::atomic<uint32_t> descriptor_words_reported_{0};  // bit per WordsStatus already logged
    // Root tables read from descriptor memory (BridgeOptions::slot_source).
    class MemorySlotSource final : public d3d::DescriptorSlotSource {
    public:
        explicit MemorySlotSource(Device& device) : device_(device) {}
        bool reads(const d3d::DescriptorHeap& heap) const override;
        bool snapshot(const d3d::DescriptorHeap& heap, uint32_t first, uint32_t count,
                      std::vector<uint8_t>* out) const override;
        d3d::ViewDescriptor view(const d3d::DescriptorHeap& heap, uint32_t index,
                                 d3d::DescriptorRangeType range) const override;
        bool sampler(const d3d::DescriptorHeap& heap, uint32_t index, d3d::SamplerDesc* out) const override;

    private:
        void note(uint32_t kind, uint32_t status, uint64_t address) const;
        Device& device_;
    };
    std::unique_ptr<MemorySlotSource> slot_source_;
    mutable std::array<std::array<std::atomic<uint64_t>, 7>, 4> slot_reads_{};
    mutable std::atomic<uint32_t> slot_reads_reported_{0};  // bit per (kind, status) already logged
    std::atomic<uint64_t> skipped_draws_{0};
    std::atomic<uint64_t> unsupported_copies_{0};
    xbox::FramePacer pacer_;
    std::atomic<uint64_t> xbox_counters_[static_cast<size_t>(xbox::Counter::Count)] = {};
    std::atomic<uint64_t> last_frame_token_{0};
    struct AliasWindow {
        uint64_t gpu_base, cpu_base, size;
    };
    mutable std::mutex alias_mutex_;
    std::vector<AliasWindow> aliases_;
    uint64_t alias_listener_ = 0;  // os::xmem_windows subscription (O27)
};

// ---------------------------------------------------------------------------------------------
// Children
// ---------------------------------------------------------------------------------------------

class Blob final : public com::ComObject, public IBlobHost {
public:
    static constexpr const char* kComClassName = "ID3DBlob";
    explicit Blob(uint64_t size);
    ~Blob() override;
    bool valid() const { return buffer_ != 0 || size_ == 0; }
    uint64_t GetBufferPointer() override { return buffer_; }
    uint64_t GetBufferSize() override { return size_; }
    // Creates a blob holding `bytes` (guest memory); empty Ref on allocation failure.
    static com::Ref<Blob> create(const void* bytes, size_t size);

private:
    uint64_t buffer_ = 0;
    uint64_t size_ = 0;
};

class CommandAllocator final : public ChildBase<ICommandAllocatorHost> {
public:
    static constexpr const char* kComClassName = "ID3D12CommandAllocator";
    CommandAllocator(com::Ref<Device> device, uint32_t type);
    ~CommandAllocator() override;
    HResult Reset() override;
    // Xbox slot 9: the executable registers a callback that hands out memory for the allocator's
    // command buffers. This layer never writes command buffers into guest memory, so the callback
    // is only recorded (never called).
    HResult SetAllocationCallback_unverified(uint64_t callback, uint64_t context) override;
    uint32_t type() const { return type_; }
    uint64_t resets() const { return resets_.load(); }
    uint64_t allocation_callback() const { return callback_.load(); }

private:
    uint32_t type_;
    std::atomic<uint64_t> resets_{0};
    std::atomic<uint64_t> callback_{0};
};

class Fence final : public ChildBase<IFenceHost> {
public:
    static constexpr const char* kComClassName = "ID3D12Fence";
    Fence(com::Ref<Device> device, uint64_t initial_value, uint32_t flags);
    ~Fence() override;
    uint64_t GetCompletedValue() override { return value_.load(std::memory_order_acquire); }
    HResult SetEventOnCompletion(uint64_t value, uint64_t event_handle) override;
    HResult Signal(uint64_t value) override;
    uint32_t GetCreationFlags() override { return flags_; }
    uint64_t CompletedValueAddress() override;
    // GPU-side signal (queue Signal): monotonically advances completed value when the work
    // submitted before it completed, which in this runtime is at submission time (see Queue::Signal).
    void signal_from_queue(uint64_t value);

private:
    void fire_events_locked(std::vector<uint64_t>* to_set);
    void publish_value_locked();

    std::atomic<uint64_t> value_;
    uint32_t flags_;
    uint64_t value_word_ = 0;  // guest word behind CompletedValueAddress (allocated on first use)
    bool destroyed_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::multimap<uint64_t, uint64_t> waits_;  // value -> event handle
};

class DescriptorHeap final : public ChildBase<IDescriptorHeapHost> {
public:
    static constexpr const char* kComClassName = "ID3D12DescriptorHeap";
    DescriptorHeap(com::Ref<Device> device, d3d::DescriptorHeap* heap, const GuestDescriptorHeapDesc& desc);
    ~DescriptorHeap() override;
    GuestDescriptorHeapDesc GetDesc() override { return desc_; }
    CpuHandleResult GetCPUDescriptorHandleForHeapStart() override { return {heap_->cpu_start().ptr}; }
    GpuHandleResult GetGPUDescriptorHandleForHeapStart() override { return {heap_->gpu_start().ptr}; }
    d3d::DescriptorHeap* heap() const { return heap_; }
    // Guest memory behind the CPU handles that the device allocated for this heap (freed with it).
    void set_owned_memory(uint64_t va) { owned_memory_ = va; }

private:
    d3d::DescriptorHeap* heap_;
    GuestDescriptorHeapDesc desc_;
    uint64_t owned_memory_ = 0;
};

class RootSignature final : public ChildBase<IRootSignatureHost> {
public:
    static constexpr const char* kComClassName = "ID3D12RootSignature";
    RootSignature(com::Ref<Device> device, std::unique_ptr<d3d::RootSignature> signature);
    ~RootSignature() override;
    const d3d::RootSignature& signature() const { return *signature_; }

private:
    std::unique_ptr<d3d::RootSignature> signature_;
};

class QueryHeap final : public ChildBase<IQueryHeapHost> {
public:
    static constexpr const char* kComClassName = "ID3D12QueryHeap";
    QueryHeap(com::Ref<Device> device, const GuestQueryHeapDesc& desc) : ChildBase(std::move(device)), desc_(desc) {}
    const GuestQueryHeapDesc& desc() const { return desc_; }

private:
    GuestQueryHeapDesc desc_;
};

class CommandSignature final : public ChildBase<ICommandSignatureHost> {
public:
    static constexpr const char* kComClassName = "ID3D12CommandSignature";
    CommandSignature(com::Ref<Device> device, uint32_t byte_stride, uint32_t argument_count)
        : ChildBase(std::move(device)), byte_stride_(byte_stride), argument_count_(argument_count) {}
    uint32_t byte_stride() const { return byte_stride_; }
    uint32_t argument_count() const { return argument_count_; }

private:
    uint32_t byte_stride_;
    uint32_t argument_count_;
};

class CommandQueue final : public ChildBase<ICommandQueueHost> {
public:
    static constexpr const char* kComClassName = "ID3D12CommandQueue";
    CommandQueue(com::Ref<Device> device, const GuestCommandQueueDesc& desc);
    ~CommandQueue() override;
    HResult Signal(InterfaceArg<IFenceHost> fence, uint64_t value) override;
    HResult Wait(InterfaceArg<IFenceHost> fence, uint64_t value) override;
    HResult GetTimestampFrequency(GuestPtr<uint64_t> frequency) override;
    // Xbox slot 32 (d3d12_xs_xbox.cpp): plane 0's first resource is the back buffer.
    HResult PresentX(uint32_t plane_count, uint64_t planes, uint64_t parameters) override;
    // Xbox slot 35: (count, command lists, frame token, flags); the token is recorded, the flags are
    // not interpreted.
    void ExecuteCommandListsX(com::CallContext& ctx) override;
    HResult SuspendX(uint32_t flags) override;
    HResult ResumeX() override;
    HResult Slot31_UpdatePageMappings_unverified(com::CallContext& ctx) override;
    bool suspended() const { return suspended_.load(); }
    // ExecuteCommandLists semantics for host callers and tests: list interface pointers.
    void execute(const std::vector<uint64_t>& lists);
    const GuestCommandQueueDesc& desc() const { return desc_; }

private:
    GuestCommandQueueDesc desc_;
    std::atomic<bool> suspended_{false};
};

// ---------------------------------------------------------------------------------------------
// Interface declarations (vtables)
// ---------------------------------------------------------------------------------------------

extern const com::Interface<IDeviceHost> kID3D12Device;
extern const com::Interface<ICommandQueueHost> kID3D12CommandQueue;
extern const com::Interface<ICommandAllocatorHost> kID3D12CommandAllocator;
extern const com::Interface<IFenceHost> kID3D12Fence;
extern const com::Interface<IDescriptorHeapHost> kID3D12DescriptorHeap;
extern const com::Interface<IRootSignatureHost> kID3D12RootSignature;
extern const com::Interface<IQueryHeapHost> kID3D12QueryHeap;
extern const com::Interface<ICommandSignatureHost> kID3D12CommandSignature;
extern const com::Interface<IBlobHost> kID3DBlob;

// DXGI format block layout (bytes per element or compressed block, block size, plane count).
struct DxgiFormatLayout {
    uint32_t block_bytes = 0;
    uint32_t block_width = 1;
    uint32_t block_height = 1;
    uint32_t planes = 1;
    bool known = false;
};
DxgiFormatLayout dxgi_format_layout(uint32_t format);
// Bytes per element of `plane` (depth-stencil formats: 4-byte depth plane, 1-byte stencil plane).
uint32_t plane_block_bytes(uint32_t format, uint32_t plane);
// Linear size of a resource's subresources (no tiling or alignment padding).
uint64_t resource_size_estimate(const GuestResourceDesc& desc);

// Process-wide totals of the work the title records, by outcome (monotonic; sampled per Present).
// A draw is "issued" when the title records it and ends in exactly one of: emitted (a packet),
// no shader module, mesh/amplification pipeline (placeholder state), rectangle list, or dropped by
// the state bridge (no pipeline state, missing bindings, ...). Dispatches likewise. Indirect
// executions have no packet and are counted separately. Uploads count WriteTexture / WriteBuffer
// packets the layer builds from guest memory; copies it cannot express are "copies skipped".
enum class Work : size_t {
    DrawsIssued,
    DrawsEmitted,
    DrawsNoShaderModule,
    DrawsMeshPipeline,
    DrawsRectList,
    DrawsDroppedByBridge,
    DispatchesIssued,
    DispatchesEmitted,
    DispatchesNoShaderModule,
    DispatchesDroppedByBridge,
    IndirectExecutionsSkipped,
    TextureUploads,
    BufferUploads,
    CopiesSkipped,
    ResourcesCreated,
    ResourcesDestroyed,
    Count
};
void count_work(Work w, uint64_t n = 1);
uint64_t work_total(Work w);
const char* work_name(Work w);
// Bytes the console's D3D12 allocates for a committed resource: the 64 KiB-block surface size
// for the 2D surfaces os/surface_blocks.h covers, otherwise resource_size_estimate.
uint64_t committed_memory_size(const GuestResourceDesc& desc);

// XMemAlloc attributes for committed-resource memory (XALLOC_ATTRIBUTES fields):
//   allocator id (bits 16..23) = 0x80 - inferred from a title's allocator hook, which registers
//     its D3D resource arena handler for allocator id 0x80 (the public layout reserves ids from
//     128 for the system; the first is the D3D allocator);
//   page size (bits 14..15) = 2 - inferred from the same hook, which accepts only page-size field
//     2 (large pages) for that arena;
//   object type (bits 0..13), alignment, memory type and the rest = 0 (not documented values).
inline constexpr uint64_t kD3D12ResourceXMemAttributes = (uint64_t{0x80} << 16) | (uint64_t{2} << 14);
com::Ref<Device> device_ref(Device* device);

// Root signature description from a D3D12_VERSIONED_ROOT_SIGNATURE_DESC in guest memory (all nested
// arrays read through the seam). False + message on malformed input.
bool read_versioned_root_signature(uint64_t desc_va, d3d::RootSignatureDesc* out, std::string* error);
bool read_versioned_root_signature(const GuestVersionedRootSignatureDesc& desc, d3d::RootSignatureDesc* out,
                                   std::string* error);

// Hands `object` to guest code as the interface answering `riid` (QueryInterface semantics): writes
// the pointer to *out and returns S_OK, or E_NOINTERFACE (and *out = 0). Consumes the reference.
template <typename T>
HResult hand_out(com::Ref<T> object, const Guid& riid, OutPtr out) {
    if (!object) {
        if (out) {
            out.write(0);
        }
        return kEOutOfMemory;
    }
    if (out.is_null()) {
        return kSFalse;  // D3D12: a null out pointer only tests whether creation would succeed
    }
    uint64_t pointer = 0;
    const HResult hr = object->query_interface(riid, &pointer);
    if (hr < 0 || !out.write(pointer)) {
        if (pointer != 0) {
            object->release();
        }
        out.write(0);
        return hr < 0 ? hr : kEPointer;
    }
    return kSOk;
}

template <typename Iface>
HResult ChildBase<Iface>::GetDevice(const Guid& riid, OutPtr out) {
    if (out.is_null()) {
        return kEPointer;
    }
    uint64_t pointer = 0;
    const HResult hr = device_->query_interface(riid, &pointer);
    out.write(pointer);
    return hr;
}

}  // namespace playgta6::gpu::d3d12_guest
