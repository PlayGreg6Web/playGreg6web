#pragma once

// Win64 layouts of the D3D12 structures guest code passes to the d3d12_xs layer (G24), written
// from the public D3D12 documentation. Every struct is read from guest memory as raw bytes and
// converted at the boundary into the G6 (descriptor_heap.h), G7 (pso_cache.h) and state bridge
// (state_bridge.h) description types; pointers inside these structs are guest addresses (u64) and
// are translated through the COM bridge seam (com::read_guest / com::guest_to_host).
//
// Xbox-specific evidence (game_scarlett_master.exe, static analysis only, see d3d12_guest_objects.h):
//  * The parameter block passed to D3D12XboxCreateDevice is zero-initialised over 0x50 bytes by the
//    caller; only its first u32 (a version word) is read here.
//  * Command queue descriptors passed to the queue-creating device slots are 16 bytes (one xmm
//    store), matching D3D12_COMMAND_QUEUE_DESC.
//  * The command-list-creating Xbox device slot takes a 16-byte descriptor whose first u32 is the
//    command list type (GuestCommandListDescX below); the remaining three u32 are not interpreted.
//  * The committed-resource Xbox slot is called with a 20-byte heap-properties block (xmm + u32),
//    matching D3D12_HEAP_PROPERTIES.
// Every other layout is the public desktop one; readers validate enum ranges and sizes and report
// a diagnostic instead of interpreting data they cannot check.

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace playgta6::gpu::d3d12_guest {

struct GuestGuid {
    uint32_t data1;
    uint16_t data2;
    uint16_t data3;
    uint8_t data4[8];
};
static_assert(sizeof(GuestGuid) == 16);

// ---------------------------------------------------------------------------------------------
// Device / queue / heap descriptors
// ---------------------------------------------------------------------------------------------

struct GuestXboxCreateDeviceParams {
    uint32_t version;  // first word of the caller's 0x50-byte block; the rest is not interpreted
};

struct GuestCommandQueueDesc {  // D3D12_COMMAND_QUEUE_DESC
    uint32_t Type;
    int32_t Priority;
    uint32_t Flags;
    uint32_t NodeMask;
};
static_assert(sizeof(GuestCommandQueueDesc) == 16);

struct GuestCommandListDescX {  // Xbox command-list creation descriptor (16 bytes, see header)
    uint32_t Type;
    uint32_t reserved[3];
};
static_assert(sizeof(GuestCommandListDescX) == 16);

struct GuestDescriptorHeapDesc {  // D3D12_DESCRIPTOR_HEAP_DESC
    uint32_t Type;
    uint32_t NumDescriptors;
    uint32_t Flags;
    uint32_t NodeMask;
};
static_assert(sizeof(GuestDescriptorHeapDesc) == 16);

struct GuestHeapProperties {  // D3D12_HEAP_PROPERTIES
    uint32_t Type;  // 1 DEFAULT, 2 UPLOAD, 3 READBACK, 4 CUSTOM
    uint32_t CPUPageProperty;
    uint32_t MemoryPoolPreference;
    uint32_t CreationNodeMask;
    uint32_t VisibleNodeMask;
};
static_assert(sizeof(GuestHeapProperties) == 20);

struct GuestHeapDesc {  // D3D12_HEAP_DESC
    uint64_t SizeInBytes;
    GuestHeapProperties Properties;
    uint64_t Alignment;
    uint32_t Flags;
};
static_assert(sizeof(GuestHeapDesc) == 48 && offsetof(GuestHeapDesc, Alignment) == 32 &&
              offsetof(GuestHeapDesc, Flags) == 40);

struct GuestResourceDesc {  // D3D12_RESOURCE_DESC
    uint32_t Dimension;  // 0 UNKNOWN, 1 BUFFER, 2 TEXTURE1D, 3 TEXTURE2D, 4 TEXTURE3D
    uint64_t Alignment;
    uint64_t Width;
    uint32_t Height;
    uint16_t DepthOrArraySize;
    uint16_t MipLevels;
    uint32_t Format;
    uint32_t SampleCount;
    uint32_t SampleQuality;
    uint32_t Layout;
    uint32_t Flags;
};
static_assert(sizeof(GuestResourceDesc) == 56 && offsetof(GuestResourceDesc, Width) == 16 &&
              offsetof(GuestResourceDesc, Format) == 32 && offsetof(GuestResourceDesc, Flags) == 48);

// D3D12_RESOURCE_DESC1 = the first 52 bytes of D3D12_RESOURCE_DESC followed by D3D12_MIP_REGION
// (3 x u32 at offset 52), 72 bytes in all. Only the shared prefix is interpreted.
inline constexpr size_t kGuestResourceDesc1Size = 72;

namespace ResourceDimension {
inline constexpr uint32_t Unknown = 0;
inline constexpr uint32_t Buffer = 1;
inline constexpr uint32_t Texture1D = 2;
inline constexpr uint32_t Texture2D = 3;
inline constexpr uint32_t Texture3D = 4;
}  // namespace ResourceDimension

namespace ResourceFlags {
inline constexpr uint32_t AllowRenderTarget = 0x1;
inline constexpr uint32_t AllowDepthStencil = 0x2;
inline constexpr uint32_t AllowUnorderedAccess = 0x4;
inline constexpr uint32_t DenyShaderResource = 0x8;
inline constexpr uint32_t AllowSimultaneousAccess = 0x20;
}  // namespace ResourceFlags

namespace HeapType {
inline constexpr uint32_t Default = 1;
inline constexpr uint32_t Upload = 2;
inline constexpr uint32_t Readback = 3;
inline constexpr uint32_t Custom = 4;
}  // namespace HeapType

struct GuestClearValue {  // D3D12_CLEAR_VALUE
    uint32_t Format;
    float Color[4];  // or { float Depth; uint8_t Stencil; }
};
static_assert(sizeof(GuestClearValue) == 20);

struct GuestRange {  // D3D12_RANGE
    uint64_t Begin;
    uint64_t End;
};
static_assert(sizeof(GuestRange) == 16);

struct GuestResourceAllocationInfo {  // D3D12_RESOURCE_ALLOCATION_INFO
    uint64_t SizeInBytes;
    uint64_t Alignment;
};
static_assert(sizeof(GuestResourceAllocationInfo) == 16);

struct GuestQueryHeapDesc {  // D3D12_QUERY_HEAP_DESC
    uint32_t Type;
    uint32_t Count;
    uint32_t NodeMask;
};
static_assert(sizeof(GuestQueryHeapDesc) == 12);

struct GuestCommandSignatureDesc {  // D3D12_COMMAND_SIGNATURE_DESC
    uint32_t ByteStride;
    uint32_t NumArgumentDescs;
    uint64_t pArgumentDescs;
    uint32_t NodeMask;
};
static_assert(sizeof(GuestCommandSignatureDesc) == 24 && offsetof(GuestCommandSignatureDesc, NodeMask) == 16);

// ---------------------------------------------------------------------------------------------
// Root signatures
// ---------------------------------------------------------------------------------------------

struct GuestVersionedRootSignatureDesc {  // D3D12_VERSIONED_ROOT_SIGNATURE_DESC
    uint32_t Version;  // 1 = 1.0, 2 = 1.1, 3 = 1.2
    uint32_t pad0;     // the Desc_1_x union is 8-byte aligned
    uint32_t NumParameters;
    uint64_t pParameters;
    uint32_t NumStaticSamplers;
    uint64_t pStaticSamplers;
    uint32_t Flags;
};
static_assert(sizeof(GuestVersionedRootSignatureDesc) == 48 &&
              offsetof(GuestVersionedRootSignatureDesc, NumParameters) == 8 &&
              offsetof(GuestVersionedRootSignatureDesc, pParameters) == 16 &&
              offsetof(GuestVersionedRootSignatureDesc, NumStaticSamplers) == 24 &&
              offsetof(GuestVersionedRootSignatureDesc, pStaticSamplers) == 32 &&
              offsetof(GuestVersionedRootSignatureDesc, Flags) == 40);

// D3D12_ROOT_PARAMETER / D3D12_ROOT_PARAMETER1: { ParameterType, union @8 (16 bytes), ShaderVisibility }.
// Union: table { u32 NumDescriptorRanges; u64 pDescriptorRanges @+8 } | constants { u32 ShaderRegister,
// RegisterSpace, Num32BitValues } | descriptor { u32 ShaderRegister, RegisterSpace (, Flags in 1.1) }.
struct GuestRootParameter {
    uint32_t ParameterType;
    uint32_t pad0;
    uint32_t w0;         // NumDescriptorRanges / ShaderRegister
    uint32_t w1;         // RegisterSpace
    uint64_t q;          // pDescriptorRanges; low half = Num32BitValues / Flags (1.1)
    uint32_t ShaderVisibility;
    uint32_t pad1;
};
static_assert(sizeof(GuestRootParameter) == 32 && offsetof(GuestRootParameter, q) == 16 &&
              offsetof(GuestRootParameter, ShaderVisibility) == 24);

struct GuestDescriptorRange {  // D3D12_DESCRIPTOR_RANGE (1.0)
    uint32_t RangeType;
    uint32_t NumDescriptors;
    uint32_t BaseShaderRegister;
    uint32_t RegisterSpace;
    uint32_t OffsetInDescriptorsFromTableStart;
};
static_assert(sizeof(GuestDescriptorRange) == 20);

struct GuestDescriptorRange1 {  // D3D12_DESCRIPTOR_RANGE1 (1.1)
    uint32_t RangeType;
    uint32_t NumDescriptors;
    uint32_t BaseShaderRegister;
    uint32_t RegisterSpace;
    uint32_t Flags;
    uint32_t OffsetInDescriptorsFromTableStart;
};
static_assert(sizeof(GuestDescriptorRange1) == 24);

struct GuestStaticSamplerDesc {  // D3D12_STATIC_SAMPLER_DESC (1.0/1.1); DESC1 (1.2) appends Flags
    uint32_t Filter;
    uint32_t AddressU;
    uint32_t AddressV;
    uint32_t AddressW;
    float MipLODBias;
    uint32_t MaxAnisotropy;
    uint32_t ComparisonFunc;
    uint32_t BorderColor;
    float MinLOD;
    float MaxLOD;
    uint32_t ShaderRegister;
    uint32_t RegisterSpace;
    uint32_t ShaderVisibility;
};
static_assert(sizeof(GuestStaticSamplerDesc) == 52);
inline constexpr size_t kGuestStaticSamplerDesc1Size = 56;

// ---------------------------------------------------------------------------------------------
// Views and samplers
// ---------------------------------------------------------------------------------------------

struct GuestConstantBufferViewDesc {  // D3D12_CONSTANT_BUFFER_VIEW_DESC
    uint64_t BufferLocation;
    uint32_t SizeInBytes;
};
static_assert(sizeof(GuestConstantBufferViewDesc) == 16);

// D3D12_SHADER_RESOURCE_VIEW_DESC: { Format, ViewDimension, Shader4ComponentMapping, union @16 }.
struct GuestShaderResourceViewDesc {
    uint32_t Format;
    uint32_t ViewDimension;
    uint32_t Shader4ComponentMapping;
    uint32_t pad;
    uint32_t u[6];  // the union (24 bytes, 8-byte aligned)
};
static_assert(sizeof(GuestShaderResourceViewDesc) == 40);

// D3D12_UNORDERED_ACCESS_VIEW_DESC: { Format, ViewDimension, union @8 (32 bytes) }.
struct GuestUnorderedAccessViewDesc {
    uint32_t Format;
    uint32_t ViewDimension;
    uint32_t u[8];
};
static_assert(sizeof(GuestUnorderedAccessViewDesc) == 40);

// D3D12_RENDER_TARGET_VIEW_DESC: { Format, ViewDimension, union @8 (16 bytes) }.
struct GuestRenderTargetViewDesc {
    uint32_t Format;
    uint32_t ViewDimension;
    uint32_t u[4];
};
static_assert(sizeof(GuestRenderTargetViewDesc) == 24);

// D3D12_DEPTH_STENCIL_VIEW_DESC: { Format, ViewDimension, Flags, union @12 (12 bytes) }.
struct GuestDepthStencilViewDesc {
    uint32_t Format;
    uint32_t ViewDimension;
    uint32_t Flags;
    uint32_t u[3];
};
static_assert(sizeof(GuestDepthStencilViewDesc) == 24);

struct GuestSamplerDesc {  // D3D12_SAMPLER_DESC
    uint32_t Filter;
    uint32_t AddressU;
    uint32_t AddressV;
    uint32_t AddressW;
    float MipLODBias;
    uint32_t MaxAnisotropy;
    uint32_t ComparisonFunc;
    float BorderColor[4];
    float MinLOD;
    float MaxLOD;
};
static_assert(sizeof(GuestSamplerDesc) == 52);

// ---------------------------------------------------------------------------------------------
// Command list arguments
// ---------------------------------------------------------------------------------------------

struct GuestViewport {  // D3D12_VIEWPORT
    float TopLeftX;
    float TopLeftY;
    float Width;
    float Height;
    float MinDepth;
    float MaxDepth;
};
static_assert(sizeof(GuestViewport) == 24);

struct GuestRect {  // D3D12_RECT
    int32_t left;
    int32_t top;
    int32_t right;
    int32_t bottom;
};
static_assert(sizeof(GuestRect) == 16);

struct GuestVertexBufferView {  // D3D12_VERTEX_BUFFER_VIEW
    uint64_t BufferLocation;
    uint32_t SizeInBytes;
    uint32_t StrideInBytes;
};
static_assert(sizeof(GuestVertexBufferView) == 16);

struct GuestIndexBufferView {  // D3D12_INDEX_BUFFER_VIEW
    uint64_t BufferLocation;
    uint32_t SizeInBytes;
    uint32_t Format;  // DXGI_FORMAT_R16_UINT (57) / R32_UINT (42)
};
static_assert(sizeof(GuestIndexBufferView) == 16);

// D3D12_RESOURCE_BARRIER: { Type, Flags, union @8 (24 bytes) }.
struct GuestResourceBarrier {
    uint32_t Type;  // 0 TRANSITION, 1 ALIASING, 2 UAV
    uint32_t Flags;
    uint64_t pResource;        // Transition.pResource / Aliasing.pResourceBefore / UAV.pResource
    uint32_t u0;               // Transition.Subresource / low half of Aliasing.pResourceAfter
    uint32_t u1;               // Transition.StateBefore / high half of Aliasing.pResourceAfter
    uint32_t StateAfter;       // Transition.StateAfter
    uint32_t pad;
};
static_assert(sizeof(GuestResourceBarrier) == 32 && offsetof(GuestResourceBarrier, pResource) == 8 &&
              offsetof(GuestResourceBarrier, StateAfter) == 24);

struct GuestSubresourceFootprint {  // D3D12_SUBRESOURCE_FOOTPRINT
    uint32_t Format;
    uint32_t Width;
    uint32_t Height;
    uint32_t Depth;
    uint32_t RowPitch;
};
static_assert(sizeof(GuestSubresourceFootprint) == 20);

struct GuestPlacedSubresourceFootprint {  // D3D12_PLACED_SUBRESOURCE_FOOTPRINT
    uint64_t Offset;
    GuestSubresourceFootprint Footprint;
};
static_assert(sizeof(GuestPlacedSubresourceFootprint) == 32);

struct GuestTextureCopyLocation {  // D3D12_TEXTURE_COPY_LOCATION
    uint64_t pResource;
    uint32_t Type;  // 0 SUBRESOURCE_INDEX, 1 PLACED_FOOTPRINT
    uint32_t pad;
    GuestPlacedSubresourceFootprint PlacedFootprint;  // or u32 SubresourceIndex at offset 16
};
static_assert(sizeof(GuestTextureCopyLocation) == 48 && offsetof(GuestTextureCopyLocation, PlacedFootprint) == 16);

struct GuestBox {  // D3D12_BOX
    uint32_t left;
    uint32_t top;
    uint32_t front;
    uint32_t right;
    uint32_t bottom;
    uint32_t back;
};
static_assert(sizeof(GuestBox) == 24);

// ---------------------------------------------------------------------------------------------
// Pipeline state
// ---------------------------------------------------------------------------------------------

struct GuestShaderBytecode {  // D3D12_SHADER_BYTECODE
    uint64_t pShaderBytecode;
    uint64_t BytecodeLength;
};
static_assert(sizeof(GuestShaderBytecode) == 16);

struct GuestStreamOutputDesc {  // D3D12_STREAM_OUTPUT_DESC
    uint64_t pSODeclaration;
    uint32_t NumEntries;
    uint64_t pBufferStrides;
    uint32_t NumStrides;
    uint32_t RasterizedStream;
};
static_assert(sizeof(GuestStreamOutputDesc) == 32);

struct GuestRenderTargetBlendDesc {  // D3D12_RENDER_TARGET_BLEND_DESC
    int32_t BlendEnable;
    int32_t LogicOpEnable;
    uint32_t SrcBlend;
    uint32_t DestBlend;
    uint32_t BlendOp;
    uint32_t SrcBlendAlpha;
    uint32_t DestBlendAlpha;
    uint32_t BlendOpAlpha;
    uint32_t LogicOp;
    uint8_t RenderTargetWriteMask;
};
static_assert(sizeof(GuestRenderTargetBlendDesc) == 40);

struct GuestBlendDesc {  // D3D12_BLEND_DESC
    int32_t AlphaToCoverageEnable;
    int32_t IndependentBlendEnable;
    GuestRenderTargetBlendDesc RenderTarget[8];
};
static_assert(sizeof(GuestBlendDesc) == 328);

struct GuestRasterizerDesc {  // D3D12_RASTERIZER_DESC
    uint32_t FillMode;
    uint32_t CullMode;
    int32_t FrontCounterClockwise;
    int32_t DepthBias;
    float DepthBiasClamp;
    float SlopeScaledDepthBias;
    int32_t DepthClipEnable;
    int32_t MultisampleEnable;
    int32_t AntialiasedLineEnable;
    uint32_t ForcedSampleCount;
    uint32_t ConservativeRaster;
};
static_assert(sizeof(GuestRasterizerDesc) == 44);

struct GuestDepthStencilOpDesc {  // D3D12_DEPTH_STENCILOP_DESC
    uint32_t StencilFailOp;
    uint32_t StencilDepthFailOp;
    uint32_t StencilPassOp;
    uint32_t StencilFunc;
};
static_assert(sizeof(GuestDepthStencilOpDesc) == 16);

struct GuestDepthStencilDesc {  // D3D12_DEPTH_STENCIL_DESC
    int32_t DepthEnable;
    uint32_t DepthWriteMask;
    uint32_t DepthFunc;
    int32_t StencilEnable;
    uint8_t StencilReadMask;
    uint8_t StencilWriteMask;
    GuestDepthStencilOpDesc FrontFace;
    GuestDepthStencilOpDesc BackFace;
};
static_assert(sizeof(GuestDepthStencilDesc) == 52 && offsetof(GuestDepthStencilDesc, FrontFace) == 20);

struct GuestInputElementDesc {  // D3D12_INPUT_ELEMENT_DESC
    uint64_t SemanticName;  // guest char*
    uint32_t SemanticIndex;
    uint32_t Format;
    uint32_t InputSlot;
    uint32_t AlignedByteOffset;
    uint32_t InputSlotClass;
    uint32_t InstanceDataStepRate;
};
static_assert(sizeof(GuestInputElementDesc) == 32);

struct GuestInputLayoutDesc {  // D3D12_INPUT_LAYOUT_DESC
    uint64_t pInputElementDescs;
    uint32_t NumElements;
};
static_assert(sizeof(GuestInputLayoutDesc) == 16);

struct GuestCachedPipelineState {  // D3D12_CACHED_PIPELINE_STATE
    uint64_t pCachedBlob;
    uint64_t CachedBlobSizeInBytes;
};
static_assert(sizeof(GuestCachedPipelineState) == 16);

struct GuestGraphicsPipelineStateDesc {  // D3D12_GRAPHICS_PIPELINE_STATE_DESC
    uint64_t pRootSignature;
    GuestShaderBytecode VS;
    GuestShaderBytecode PS;
    GuestShaderBytecode DS;
    GuestShaderBytecode HS;
    GuestShaderBytecode GS;
    GuestStreamOutputDesc StreamOutput;
    GuestBlendDesc BlendState;
    uint32_t SampleMask;
    GuestRasterizerDesc RasterizerState;
    GuestDepthStencilDesc DepthStencilState;
    GuestInputLayoutDesc InputLayout;
    uint32_t IBStripCutValue;
    uint32_t PrimitiveTopologyType;
    uint32_t NumRenderTargets;
    uint32_t RTVFormats[8];
    uint32_t DSVFormat;
    uint32_t SampleCount;
    uint32_t SampleQuality;
    uint32_t NodeMask;
    GuestCachedPipelineState CachedPSO;
    uint32_t Flags;
};
static_assert(sizeof(GuestGraphicsPipelineStateDesc) == 656);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, StreamOutput) == 88);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, BlendState) == 120);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, SampleMask) == 448);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, RasterizerState) == 452);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, DepthStencilState) == 496);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, InputLayout) == 552);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, PrimitiveTopologyType) == 572);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, RTVFormats) == 580);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, DSVFormat) == 612);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, NodeMask) == 624);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, CachedPSO) == 632);
static_assert(offsetof(GuestGraphicsPipelineStateDesc, Flags) == 648);

struct GuestComputePipelineStateDesc {  // D3D12_COMPUTE_PIPELINE_STATE_DESC
    uint64_t pRootSignature;
    GuestShaderBytecode CS;
    uint32_t NodeMask;
    GuestCachedPipelineState CachedPSO;
    uint32_t Flags;
};
static_assert(sizeof(GuestComputePipelineStateDesc) == 56 && offsetof(GuestComputePipelineStateDesc, CachedPSO) == 32);

static_assert(std::is_trivially_copyable_v<GuestGraphicsPipelineStateDesc>);

}  // namespace playgta6::gpu::d3d12_guest
