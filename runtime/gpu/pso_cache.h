#pragma once

// D3D12 pipeline-state-object (PSO) cache, output-merger/rasterizer dynamic state tracking and
// render pass inference for the d3d12_xs -> WebGPU bridge.
//
// Layering
// --------
//   d3d12_xs shim  --(D3D12GraphicsPsoDesc)-->  GraphicsPso  --(PipelineVariantState)-->  GraphicsPipelineKey
//                                                                    |
//                                                     PipelineCache: key -> stable pipeline id
//   d3d12_xs command list --(OM/RS/IA calls)--> CommandListStateTracker --> StateEvent stream
//
// The layer above emits a pipeline-creation packet only when PipelineResolution::created is true and
// translates the StateEvent stream into render-pass / dynamic-state packets. This file never emits
// packets and does not depend on the packet ABI beyond reusing `playgta6::GpuPrimitiveTopology`
// numbering for WgpuPrimitiveTopology.
//
// Coordinate conventions (D3D12 vs. WebGPU)
// -----------------------------------------
// * NDC: both APIs use x,y in [-1, 1] with +y up and z in [0, 1]. No depth-range remapping is applied.
// * Framebuffer: both APIs put the origin at the top-left with +y down, and the viewport transform maps
//   NDC +y to framebuffer -y in both. Therefore no Y flip is applied anywhere in this layer.
// * Winding: both APIs evaluate polygon orientation in framebuffer space with the same orientation, so
//   FrontCounterClockwise=FALSE maps to frontFace "cw" and TRUE maps to "ccw" without inversion. If a
//   future layer ever introduces a Y flip (e.g. rendering into a flipped intermediate), it must invert the
//   front face as well.
// * Viewport depth: D3D12 MinDepth/MaxDepth map directly to setViewport(minDepth, maxDepth). D3D12 allows
//   MinDepth > MaxDepth; WebGPU does not (diagnosed, see StateDiag::ViewportDepthInverted).
//
// Threading
// ---------
// * PipelineCache and GraphicsPso are safe for concurrent use from many recording threads.
// * CommandListStateTracker belongs to exactly one command list and is not thread-safe (D3D12 command
//   lists are single-threaded too).

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu::pso {

static_assert(std::endian::native == std::endian::little,
              "pipeline keys are hashed as raw little-endian bytes");

// =====================================================================================================
// D3D12 / DXGI declarations (numeric values match the public D3D12/DXGI API). These live in this
// namespace so they cannot clash with d3d12_xs.h or the descriptor-heap layer; they should be unified
// into one shared header once those layers settle.
// =====================================================================================================

enum class DxgiFormat : uint32_t {
    UNKNOWN = 0,
    R32G32B32A32_TYPELESS = 1,
    R32G32B32A32_FLOAT = 2,
    R32G32B32A32_UINT = 3,
    R32G32B32A32_SINT = 4,
    R32G32B32_TYPELESS = 5,
    R32G32B32_FLOAT = 6,
    R32G32B32_UINT = 7,
    R32G32B32_SINT = 8,
    R16G16B16A16_TYPELESS = 9,
    R16G16B16A16_FLOAT = 10,
    R16G16B16A16_UNORM = 11,
    R16G16B16A16_UINT = 12,
    R16G16B16A16_SNORM = 13,
    R16G16B16A16_SINT = 14,
    R32G32_TYPELESS = 15,
    R32G32_FLOAT = 16,
    R32G32_UINT = 17,
    R32G32_SINT = 18,
    R32G8X24_TYPELESS = 19,
    D32_FLOAT_S8X24_UINT = 20,
    R32_FLOAT_X8X24_TYPELESS = 21,
    X32_TYPELESS_G8X24_UINT = 22,
    R10G10B10A2_TYPELESS = 23,
    R10G10B10A2_UNORM = 24,
    R10G10B10A2_UINT = 25,
    R11G11B10_FLOAT = 26,
    R8G8B8A8_TYPELESS = 27,
    R8G8B8A8_UNORM = 28,
    R8G8B8A8_UNORM_SRGB = 29,
    R8G8B8A8_UINT = 30,
    R8G8B8A8_SNORM = 31,
    R8G8B8A8_SINT = 32,
    R16G16_TYPELESS = 33,
    R16G16_FLOAT = 34,
    R16G16_UNORM = 35,
    R16G16_UINT = 36,
    R16G16_SNORM = 37,
    R16G16_SINT = 38,
    R32_TYPELESS = 39,
    D32_FLOAT = 40,
    R32_FLOAT = 41,
    R32_UINT = 42,
    R32_SINT = 43,
    R24G8_TYPELESS = 44,
    D24_UNORM_S8_UINT = 45,
    R24_UNORM_X8_TYPELESS = 46,
    X24_TYPELESS_G8_UINT = 47,
    R8G8_TYPELESS = 48,
    R8G8_UNORM = 49,
    R8G8_UINT = 50,
    R8G8_SNORM = 51,
    R8G8_SINT = 52,
    R16_TYPELESS = 53,
    R16_FLOAT = 54,
    D16_UNORM = 55,
    R16_UNORM = 56,
    R16_UINT = 57,
    R16_SNORM = 58,
    R16_SINT = 59,
    R8_TYPELESS = 60,
    R8_UNORM = 61,
    R8_UINT = 62,
    R8_SNORM = 63,
    R8_SINT = 64,
    A8_UNORM = 65,
    R1_UNORM = 66,
    R9G9B9E5_SHAREDEXP = 67,
    R8G8_B8G8_UNORM = 68,
    G8R8_G8B8_UNORM = 69,
    BC1_TYPELESS = 70,
    BC1_UNORM = 71,
    BC1_UNORM_SRGB = 72,
    BC2_TYPELESS = 73,
    BC2_UNORM = 74,
    BC2_UNORM_SRGB = 75,
    BC3_TYPELESS = 76,
    BC3_UNORM = 77,
    BC3_UNORM_SRGB = 78,
    BC4_TYPELESS = 79,
    BC4_UNORM = 80,
    BC4_SNORM = 81,
    BC5_TYPELESS = 82,
    BC5_UNORM = 83,
    BC5_SNORM = 84,
    B5G6R5_UNORM = 85,
    B5G5R5A1_UNORM = 86,
    B8G8R8A8_UNORM = 87,
    B8G8R8X8_UNORM = 88,
    R10G10B10_XR_BIAS_A2_UNORM = 89,
    B8G8R8A8_TYPELESS = 90,
    B8G8R8A8_UNORM_SRGB = 91,
    B8G8R8X8_TYPELESS = 92,
    B8G8R8X8_UNORM_SRGB = 93,
    BC6H_TYPELESS = 94,
    BC6H_UF16 = 95,
    BC6H_SF16 = 96,
    BC7_TYPELESS = 97,
    BC7_UNORM = 98,
    BC7_UNORM_SRGB = 99,
    // 100..114 are video/palettized formats (AYUV .. P8/A8P8); none has a WebGPU equivalent.
    B4G4R4A4_UNORM = 115,
    // Xbox-only (d3d12_x / d3d12_xs) 16-bit depth + 8-bit stencil family.
    D16_UNORM_S8_UINT = 118,
    R16_UNORM_X8_TYPELESS = 119,
    X16_TYPELESS_G8_UINT = 120,
};

enum class D3D12Blend : uint32_t {
    Zero = 1,
    One = 2,
    SrcColor = 3,
    InvSrcColor = 4,
    SrcAlpha = 5,
    InvSrcAlpha = 6,
    DestAlpha = 7,
    InvDestAlpha = 8,
    DestColor = 9,
    InvDestColor = 10,
    SrcAlphaSat = 11,
    BlendFactor = 14,
    InvBlendFactor = 15,
    Src1Color = 16,
    InvSrc1Color = 17,
    Src1Alpha = 18,
    InvSrc1Alpha = 19,
    AlphaFactor = 20,
    InvAlphaFactor = 21,
};

enum class D3D12BlendOp : uint32_t { Add = 1, Subtract = 2, RevSubtract = 3, Min = 4, Max = 5 };

enum class D3D12LogicOp : uint32_t {
    Clear = 0,
    Set = 1,
    Copy = 2,
    CopyInverted = 3,
    Noop = 4,
    Invert = 5,
    And = 6,
    Nand = 7,
    Or = 8,
    Nor = 9,
    Xor = 10,
    Equiv = 11,
    AndReverse = 12,
    AndInverted = 13,
    OrReverse = 14,
    OrInverted = 15,
};

namespace D3D12ColorWriteEnable {
inline constexpr uint8_t Red = 1;
inline constexpr uint8_t Green = 2;
inline constexpr uint8_t Blue = 4;
inline constexpr uint8_t Alpha = 8;
inline constexpr uint8_t All = 15;
}  // namespace D3D12ColorWriteEnable

enum class D3D12FillMode : uint32_t { Wireframe = 2, Solid = 3 };
enum class D3D12CullMode : uint32_t { None = 1, Front = 2, Back = 3 };

enum class D3D12ComparisonFunc : uint32_t {
    None = 0,  // only meaningful for samplers; treated as invalid for depth/stencil
    Never = 1,
    Less = 2,
    Equal = 3,
    LessEqual = 4,
    Greater = 5,
    NotEqual = 6,
    GreaterEqual = 7,
    Always = 8,
};

enum class D3D12StencilOp : uint32_t {
    Keep = 1,
    Zero = 2,
    Replace = 3,
    IncrSat = 4,
    DecrSat = 5,
    Invert = 6,
    Incr = 7,
    Decr = 8,
};

enum class D3D12DepthWriteMask : uint32_t { Zero = 0, All = 1 };
enum class D3D12ConservativeRasterizationMode : uint32_t { Off = 0, On = 1 };
enum class D3D12InputClassification : uint32_t { PerVertexData = 0, PerInstanceData = 1 };

enum class D3D12PrimitiveTopologyType : uint32_t {
    Undefined = 0,
    Point = 1,
    Line = 2,
    Triangle = 3,
    Patch = 4,
};

// D3D_PRIMITIVE_TOPOLOGY (set with IASetPrimitiveTopology).
enum class D3DPrimitiveTopology : uint32_t {
    Undefined = 0,
    PointList = 1,
    LineList = 2,
    LineStrip = 3,
    TriangleList = 4,
    TriangleStrip = 5,
    TriangleFan = 6,
    LineListAdj = 10,
    LineStripAdj = 11,
    TriangleListAdj = 12,
    TriangleStripAdj = 13,
    ControlPointPatchList1 = 33,
    ControlPointPatchList32 = 64,
};

enum class D3D12IndexBufferStripCutValue : uint32_t {
    Disabled = 0,
    Value0xFFFF = 1,
    Value0xFFFFFFFF = 2,
};

inline constexpr uint32_t kD3D12AppendAlignedElement = 0xFFFFFFFFu;
inline constexpr uint32_t kD3D12DefaultSampleMask = 0xFFFFFFFFu;
inline constexpr uint32_t kD3D12SimultaneousRenderTargetCount = 8;
inline constexpr uint32_t kD3D12IaVertexInputSlotCount = 32;

namespace D3D12ClearFlags {
inline constexpr uint32_t Depth = 0x1;
inline constexpr uint32_t Stencil = 0x2;
}  // namespace D3D12ClearFlags

namespace D3D12DsvFlags {
inline constexpr uint32_t ReadOnlyDepth = 0x1;
inline constexpr uint32_t ReadOnlyStencil = 0x2;
}  // namespace D3D12DsvFlags

struct D3D12RenderTargetBlendDesc {
    bool BlendEnable = false;
    bool LogicOpEnable = false;
    D3D12Blend SrcBlend = D3D12Blend::One;
    D3D12Blend DestBlend = D3D12Blend::Zero;
    D3D12BlendOp BlendOp = D3D12BlendOp::Add;
    D3D12Blend SrcBlendAlpha = D3D12Blend::One;
    D3D12Blend DestBlendAlpha = D3D12Blend::Zero;
    D3D12BlendOp BlendOpAlpha = D3D12BlendOp::Add;
    D3D12LogicOp LogicOp = D3D12LogicOp::Noop;
    uint8_t RenderTargetWriteMask = D3D12ColorWriteEnable::All;
};

struct D3D12BlendDesc {
    bool AlphaToCoverageEnable = false;
    bool IndependentBlendEnable = false;
    D3D12RenderTargetBlendDesc RenderTarget[kD3D12SimultaneousRenderTargetCount] = {};
};

struct D3D12RasterizerDesc {
    D3D12FillMode FillMode = D3D12FillMode::Solid;
    D3D12CullMode CullMode = D3D12CullMode::Back;
    bool FrontCounterClockwise = false;
    int32_t DepthBias = 0;
    float DepthBiasClamp = 0.0f;
    float SlopeScaledDepthBias = 0.0f;
    bool DepthClipEnable = true;
    bool MultisampleEnable = false;
    bool AntialiasedLineEnable = false;
    uint32_t ForcedSampleCount = 0;
    D3D12ConservativeRasterizationMode ConservativeRaster = D3D12ConservativeRasterizationMode::Off;
};

struct D3D12DepthStencilOpDesc {
    D3D12StencilOp StencilFailOp = D3D12StencilOp::Keep;
    D3D12StencilOp StencilDepthFailOp = D3D12StencilOp::Keep;
    D3D12StencilOp StencilPassOp = D3D12StencilOp::Keep;
    D3D12ComparisonFunc StencilFunc = D3D12ComparisonFunc::Always;
};

// D3D12_DEPTH_STENCIL_DESC1 (DepthBoundsTestEnable is the DESC1 extension).
struct D3D12DepthStencilDesc {
    bool DepthEnable = true;
    D3D12DepthWriteMask DepthWriteMask = D3D12DepthWriteMask::All;
    D3D12ComparisonFunc DepthFunc = D3D12ComparisonFunc::Less;
    bool StencilEnable = false;
    uint8_t StencilReadMask = 0xFF;
    uint8_t StencilWriteMask = 0xFF;
    D3D12DepthStencilOpDesc FrontFace = {};
    D3D12DepthStencilOpDesc BackFace = {};
    bool DepthBoundsTestEnable = false;
};

struct D3D12InputElementDesc {
    const char* SemanticName = nullptr;
    uint32_t SemanticIndex = 0;
    DxgiFormat Format = DxgiFormat::UNKNOWN;
    uint32_t InputSlot = 0;
    uint32_t AlignedByteOffset = kD3D12AppendAlignedElement;
    D3D12InputClassification InputSlotClass = D3D12InputClassification::PerVertexData;
    uint32_t InstanceDataStepRate = 0;
};

struct DxgiSampleDesc {
    uint32_t Count = 1;
    uint32_t Quality = 0;
};

// One entry of the vertex shader's input signature as produced by the shader pipeline (semantic ->
// WGSL @location). Semantic names are matched case-insensitively, like D3D12 does.
struct VertexInputSignatureElement {
    const char* semantic_name = nullptr;
    uint32_t semantic_index = 0;
    uint32_t location = 0;
};

enum class WgpuVertexFormat : uint8_t;

// A vertex attribute fixed by the vertex stage's reflection (G62, gpu/stage_reflection.h): the
// attribute reads `format` at `offset` in vertex buffer `slot` into @location(location).
struct VertexInputFetch {
    uint32_t slot = 0;
    uint32_t offset = 0;
    uint32_t location = 0;
    WgpuVertexFormat format{};  // WgpuVertexFormat::Undefined
};

// D3D12_GRAPHICS_PIPELINE_STATE_DESC (or the equivalent pipeline-state stream), with shader bytecode
// replaced by opaque 64-bit shader identities (WGSL module id or content hash; 0 = no shader) and the
// root signature replaced by an opaque id from the descriptor-heap/root-signature layer.
// Pointers only need to stay valid for the duration of the call that consumes the desc.
struct D3D12GraphicsPsoDesc {
    uint32_t root_signature_id = 0;
    uint64_t vs = 0;
    uint64_t ps = 0;
    uint64_t ds = 0;
    uint64_t hs = 0;
    uint64_t gs = 0;
    uint64_t as = 0;  // amplification shader (mesh pipeline)
    uint64_t ms = 0;  // mesh shader (mesh pipeline)
    // Optional VS input signature. When absent, input element i is bound to @location(i).
    const VertexInputSignatureElement* vs_input_signature = nullptr;
    uint32_t vs_input_signature_count = 0;
    // Optional attributes from the vertex stage's reflection (G62). When present they replace the
    // element -> location mapping: the input elements still give each slot's classification and
    // tight stride, and elements no fetch reads are dropped.
    const VertexInputFetch* vs_input_fetches = nullptr;
    uint32_t vs_input_fetch_count = 0;
    // Bit i set = the pixel shader writes SV_Target i. Targets the PS does not write get writeMask 0,
    // which WebGPU requires. Default assumes every declared target is written.
    uint32_t ps_output_mask = 0xFFu;
    bool stream_output_enabled = false;  // StreamOutput.NumEntries > 0
    D3D12BlendDesc BlendState = {};
    uint32_t SampleMask = kD3D12DefaultSampleMask;
    D3D12RasterizerDesc RasterizerState = {};
    D3D12DepthStencilDesc DepthStencilState = {};
    const D3D12InputElementDesc* pInputElementDescs = nullptr;
    uint32_t NumInputElements = 0;
    D3D12IndexBufferStripCutValue IBStripCutValue = D3D12IndexBufferStripCutValue::Disabled;
    D3D12PrimitiveTopologyType PrimitiveTopologyType = D3D12PrimitiveTopologyType::Triangle;
    uint32_t NumRenderTargets = 0;
    DxgiFormat RTVFormats[kD3D12SimultaneousRenderTargetCount] = {};
    DxgiFormat DSVFormat = DxgiFormat::UNKNOWN;
    DxgiSampleDesc SampleDesc = {};
};

struct D3D12ComputePsoDesc {
    uint32_t root_signature_id = 0;
    uint64_t cs = 0;
};

struct D3D12Viewport {
    float TopLeftX = 0.0f;
    float TopLeftY = 0.0f;
    float Width = 0.0f;
    float Height = 0.0f;
    float MinDepth = 0.0f;
    float MaxDepth = 1.0f;
};

struct D3D12Rect {
    int32_t left = 0;
    int32_t top = 0;
    int32_t right = 0;
    int32_t bottom = 0;
};

// =====================================================================================================
// WebGPU-side enums. Values are this layer's own compact encoding (the JS worker maps them to the
// GPU* strings returned by the *_name() functions below), except WgpuPrimitiveTopology which reuses the
// frozen packet numbering of playgta6::GpuPrimitiveTopology.
// =====================================================================================================

enum class WgpuTextureFormat : uint8_t {
    Undefined = 0,
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    R16Unorm,
    R16Snorm,
    R16Uint,
    R16Sint,
    R16Float,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    R32Uint,
    R32Sint,
    R32Float,
    Rg16Unorm,
    Rg16Snorm,
    Rg16Uint,
    Rg16Sint,
    Rg16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Rgba8Snorm,
    Rgba8Uint,
    Rgba8Sint,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgb9e5Ufloat,
    Rgb10a2Uint,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    Rg32Uint,
    Rg32Sint,
    Rg32Float,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Stencil8,
    Depth16Unorm,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
    Bc1RgbaUnorm,
    Bc1RgbaUnormSrgb,
    Bc2RgbaUnorm,
    Bc2RgbaUnormSrgb,
    Bc3RgbaUnorm,
    Bc3RgbaUnormSrgb,
    Bc4RUnorm,
    Bc4RSnorm,
    Bc5RgUnorm,
    Bc5RgSnorm,
    Bc6hRgbUfloat,
    Bc6hRgbFloat,
    Bc7RgbaUnorm,
    Bc7RgbaUnormSrgb,
    Count,
};

enum class WgpuVertexFormat : uint8_t {
    Undefined = 0,
    Uint8,
    Uint8x2,
    Uint8x4,
    Sint8,
    Sint8x2,
    Sint8x4,
    Unorm8,
    Unorm8x2,
    Unorm8x4,
    Snorm8,
    Snorm8x2,
    Snorm8x4,
    Uint16,
    Uint16x2,
    Uint16x4,
    Sint16,
    Sint16x2,
    Sint16x4,
    Unorm16,
    Unorm16x2,
    Unorm16x4,
    Snorm16,
    Snorm16x2,
    Snorm16x4,
    Float16,
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Unorm10_10_10_2,
    Unorm8x4Bgra,
    Count,
};

enum class WgpuBlendFactor : uint8_t {
    Zero = 0,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
    Src1,
    OneMinusSrc1,
    Src1Alpha,
    OneMinusSrc1Alpha,
    Count,
};

enum class WgpuBlendOperation : uint8_t { Add = 0, Subtract, ReverseSubtract, Min, Max, Count };

enum class WgpuCompareFunction : uint8_t {
    Undefined = 0,
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
    Count,
};

enum class WgpuStencilOperation : uint8_t {
    Keep = 0,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
    Count,
};

enum class WgpuCullMode : uint8_t { None = 0, Front, Back, Count };
enum class WgpuFrontFace : uint8_t { Ccw = 0, Cw, Count };

// Same numbering as playgta6::GpuPrimitiveTopology (packet ABI), stored in one byte.
enum class WgpuPrimitiveTopology : uint8_t {
    TriangleList = static_cast<uint8_t>(GpuPrimitiveTopology::TriangleList),
    TriangleStrip = static_cast<uint8_t>(GpuPrimitiveTopology::TriangleStrip),
    LineList = static_cast<uint8_t>(GpuPrimitiveTopology::LineList),
    LineStrip = static_cast<uint8_t>(GpuPrimitiveTopology::LineStrip),
    PointList = static_cast<uint8_t>(GpuPrimitiveTopology::PointList),
    Count,
};

enum class WgpuIndexFormat : uint8_t { Undefined = 0, Uint16, Uint32, Count };

// `Unused` marks a hole in GPUVertexState.buffers (emit `null`).
enum class WgpuVertexStepMode : uint8_t { Unused = 0, Vertex, Instance, Count };

enum class WgpuLoadOp : uint8_t { Undefined = 0, Load, Clear, Count };
enum class WgpuStoreOp : uint8_t { Undefined = 0, Store, Discard, Count };

enum class WgpuTextureAspect : uint8_t { All = 0, DepthOnly, StencilOnly, Count };

// Optional WebGPU device features (GPUFeatureName) this layer cares about, as a bitmask.
namespace WgpuFeature {
inline constexpr uint32_t DepthClipControl = 1u << 0;        // "depth-clip-control"
inline constexpr uint32_t Depth32FloatStencil8 = 1u << 1;    // "depth32float-stencil8"
inline constexpr uint32_t TextureCompressionBc = 1u << 2;    // "texture-compression-bc"
inline constexpr uint32_t Float32Blendable = 1u << 3;        // "float32-blendable"
inline constexpr uint32_t Rg11b10UfloatRenderable = 1u << 4; // "rg11b10ufloat-renderable"
inline constexpr uint32_t DualSourceBlending = 1u << 5;      // "dual-source-blending"
inline constexpr uint32_t TextureFormatsTier1 = 1u << 6;     // "texture-formats-tier1"
inline constexpr uint32_t Bgra8UnormStorage = 1u << 7;       // "bgra8unorm-storage"
inline constexpr uint32_t All = (1u << 8) - 1u;
}  // namespace WgpuFeature

const char* texture_format_name(WgpuTextureFormat format) noexcept;
const char* vertex_format_name(WgpuVertexFormat format) noexcept;
const char* blend_factor_name(WgpuBlendFactor factor) noexcept;
const char* blend_operation_name(WgpuBlendOperation op) noexcept;
const char* compare_function_name(WgpuCompareFunction func) noexcept;
const char* stencil_operation_name(WgpuStencilOperation op) noexcept;
const char* cull_mode_name(WgpuCullMode mode) noexcept;
const char* front_face_name(WgpuFrontFace face) noexcept;
const char* primitive_topology_name(WgpuPrimitiveTopology topology) noexcept;
const char* index_format_name(WgpuIndexFormat format) noexcept;
const char* load_op_name(WgpuLoadOp op) noexcept;
const char* store_op_name(WgpuStoreOp op) noexcept;
const char* feature_name(uint32_t single_feature_bit) noexcept;

// =====================================================================================================
// Format tables
// =====================================================================================================

namespace TextureFormatCaps {
inline constexpr uint16_t ColorRenderable = 1u << 0;  // RENDER_ATTACHMENT without any feature
inline constexpr uint16_t Blendable = 1u << 1;        // blendable without any feature
inline constexpr uint16_t Multisample = 1u << 2;
inline constexpr uint16_t Depth = 1u << 3;
inline constexpr uint16_t Stencil = 1u << 4;
inline constexpr uint16_t HasAlpha = 1u << 5;
inline constexpr uint16_t Srgb = 1u << 6;
inline constexpr uint16_t Integer = 1u << 7;
inline constexpr uint16_t Compressed = 1u << 8;
}  // namespace TextureFormatCaps

// Capabilities of a WebGPU texture format per the WebGPU "Texture Format Capabilities" appendix.
struct WgpuTextureFormatInfo {
    const char* name;
    uint16_t caps;                  // TextureFormatCaps bits available on a core device
    uint32_t required_feature;      // feature needed to use the format at all (0 = core)
    uint32_t renderable_feature;    // feature that makes a non-core-renderable format renderable
    uint32_t blendable_feature;     // feature that makes a renderable format blendable
    uint8_t rt_byte_cost;           // render target pixel byte cost
    uint8_t rt_alignment;           // render target component alignment
    WgpuTextureFormat fallback;     // replacement when the needed feature is missing (Undefined = none)
};

const WgpuTextureFormatInfo& texture_format_info(WgpuTextureFormat format) noexcept;
bool is_color_renderable(WgpuTextureFormat format, uint32_t features) noexcept;
bool is_blendable(WgpuTextureFormat format, uint32_t features) noexcept;
bool has_depth_aspect(WgpuTextureFormat format) noexcept;
// The format that differs from `format` only in sRGB-ness (rgba8unorm <-> rgba8unorm-srgb, ...),
// or Undefined. WebGPU lets a texture list it in viewFormats without a feature; D3D12 views of
// the same resource switch freely between the two.
WgpuTextureFormat srgb_sibling(WgpuTextureFormat format) noexcept;
bool has_stencil_aspect(WgpuTextureFormat format) noexcept;
bool has_alpha_channel(WgpuTextureFormat format) noexcept;

// WebGPU "Calculating color attachment bytes per sample" over a list of formats (Undefined = hole).
uint32_t color_attachment_bytes_per_sample(std::span<const WgpuTextureFormat> formats) noexcept;

uint32_t vertex_format_byte_size(WgpuVertexFormat format) noexcept;

enum class FormatUsage : uint8_t {
    RenderTarget,
    DepthStencil,
    Sampled,
};

namespace FormatDiag {
inline constexpr uint32_t Unsupported = 1u << 0;        // no WebGPU representation for this usage
inline constexpr uint32_t TypelessResolved = 1u << 1;   // typeless format resolved by usage
inline constexpr uint32_t FeatureFallback = 1u << 2;    // feature missing: promoted to `format`
inline constexpr uint32_t NeedsCpuConversion = 1u << 3; // uploads must be converted (BC decode, widening)
inline constexpr uint32_t AlphaIgnored = 1u << 4;       // X8 channel formats: alpha must read as 1
inline constexpr uint32_t NeedsViewReinterpretation = 1u << 5;  // typeless resource viewed as other types
}  // namespace FormatDiag

struct TextureFormatMapping {
    WgpuTextureFormat format = WgpuTextureFormat::Undefined;
    WgpuTextureAspect aspect = WgpuTextureAspect::All;
    uint32_t required_features = 0;  // features the device must have enabled for `format`
    uint32_t diagnostics = 0;        // FormatDiag bits
    bool supported() const noexcept { return format != WgpuTextureFormat::Undefined; }
};

// DXGI -> GPUTextureFormat for a given usage on a device with `features` enabled.
// * sRGB variants map to the "-srgb" formats; depth formats map to depth16unorm / depth24plus-stencil8 /
//   depth32float / depth32float-stencil8 (falls back to depth24plus-stencil8 without the feature).
// * Typeless formats resolve by usage: depth-capable typeless formats become the depth format for
//   DepthStencil (and for Sampled the matching depth format + aspect), otherwise the UNORM (or FLOAT if no
//   UNORM exists in core WebGPU) sibling; flagged TypelessResolved | NeedsViewReinterpretation.
// * BC formats require "texture-compression-bc"; without it they fall back to an uncompressed format
//   (NeedsCpuConversion). BC formats are never renderable.
// * Formats whose renderability or existence depends on a feature fall back to a wider format.
// * Packed 16-bit B5G5R5A1_UNORM / B4G4R4A4_UNORM have no WebGPU format: Sampled / RenderTarget promote to
//   rgba8unorm (FeatureFallback | NeedsCpuConversion: uploads widen 16bpp -> 32bpp, channels swizzled).
// * Xbox D16_UNORM_S8_UINT (and its R16_UNORM_X8 / X16_G8 views) has no 16-bit depth + stencil WebGPU format:
//   it promotes to depth24plus-stencil8 (FeatureFallback), with the depth/stencil aspect for the views.
TextureFormatMapping map_texture_format(DxgiFormat format, FormatUsage usage, uint32_t features) noexcept;

struct VertexFormatMapping {
    WgpuVertexFormat format = WgpuVertexFormat::Undefined;
    uint32_t byte_size = 0;
    bool supported() const noexcept { return format != WgpuVertexFormat::Undefined; }
};

// DXGI -> GPUVertexFormat for input-assembler elements. 3-component 8/16-bit, R11G11B10_FLOAT,
// R10G10B10A2_UINT, sRGB and packed 16-bit formats have no WebGPU vertex format and report unsupported.
VertexFormatMapping map_vertex_format(DxgiFormat format) noexcept;

// Enum lowering helpers (exposed for tests and for the layer above). Invalid inputs return nullopt.
std::optional<WgpuBlendFactor> map_blend_factor(D3D12Blend blend) noexcept;
std::optional<WgpuBlendOperation> map_blend_op(D3D12BlendOp op) noexcept;
std::optional<WgpuCompareFunction> map_compare_function(D3D12ComparisonFunc func) noexcept;
std::optional<WgpuStencilOperation> map_stencil_op(D3D12StencilOp op) noexcept;
std::optional<WgpuCullMode> map_cull_mode(D3D12CullMode mode) noexcept;
WgpuFrontFace map_front_face(bool front_counter_clockwise) noexcept;

// D3D12 sample count -> WebGPU (only 1 and 4 exist). Returns 1 for 0/1, 4 otherwise.
uint32_t map_sample_count(uint32_t d3d12_count) noexcept;

struct TopologyMapping {
    WgpuPrimitiveTopology topology = WgpuPrimitiveTopology::TriangleList;
    D3D12PrimitiveTopologyType type = D3D12PrimitiveTopologyType::Undefined;
    uint64_t diagnostics = 0;  // PipelineDiag bits
};

// IASetPrimitiveTopology value -> WebGPU topology (+ the D3D12 topology type it belongs to).
TopologyMapping map_primitive_topology(D3DPrimitiveTopology topology) noexcept;

// =====================================================================================================
// Pipeline diagnostics
// =====================================================================================================

namespace PipelineDiag {
inline constexpr uint64_t GeometryShaderUnsupported = 1ull << 0;      // fatal: GS has no WebGPU stage
inline constexpr uint64_t TessellationUnsupported = 1ull << 1;        // fatal: HS/DS or patch topology
inline constexpr uint64_t MeshShaderUnsupported = 1ull << 2;          // fatal: AS/MS need compute lowering
inline constexpr uint64_t StreamOutputUnsupported = 1ull << 3;        // SO dropped
inline constexpr uint64_t MissingVertexShader = 1ull << 4;            // fatal
inline constexpr uint64_t WireframeFillUnsupported = 1ull << 5;       // fallback: solid fill
inline constexpr uint64_t LogicOpUnsupported = 1ull << 6;             // fallback: COPY (no blending)
inline constexpr uint64_t DualSourceBlendUnavailable = 1ull << 7;     // fallback: SRC1 -> SRC factors
inline constexpr uint64_t DualSourceMultipleTargets = 1ull << 8;      // fatal: WebGPU allows 1 target
inline constexpr uint64_t BlendOnNonBlendableFormat = 1ull << 9;      // fallback: blending disabled
inline constexpr uint64_t AlphaFactorApproximated = 1ull << 10;       // ALPHA_FACTOR -> "constant"
inline constexpr uint64_t AlphaToCoverageDropped = 1ull << 11;        // needs count 4 + blendable RT0 w/ alpha
inline constexpr uint64_t ConservativeRasterUnsupported = 1ull << 12; // fallback: standard raster
inline constexpr uint64_t DepthBoundsUnsupported = 1ull << 13;        // fallback: test disabled
inline constexpr uint64_t DepthClipControlUnavailable = 1ull << 14;   // fallback: depth clipping on
inline constexpr uint64_t DepthBiasDroppedForNonTriangle = 1ull << 15;// WebGPU: bias must be 0 for lines/points
inline constexpr uint64_t SampleCountRounded = 1ull << 16;            // 2/8/16 samples -> 4
inline constexpr uint64_t SampleQualityIgnored = 1ull << 17;
inline constexpr uint64_t ForcedSampleCountUnsupported = 1ull << 18;  // fallback: ignored
inline constexpr uint64_t LineRasterModeIgnored = 1ull << 19;         // AA / quadrilateral lines
inline constexpr uint64_t TopologyTypeMismatch = 1ull << 20;          // IA topology vs PSO topology type
inline constexpr uint64_t TriangleFanNeedsConversion = 1ull << 21;    // fallback: triangle-list key
inline constexpr uint64_t AdjacencyTopologyUnsupported = 1ull << 22;  // fallback: non-adjacency topology
inline constexpr uint64_t StripCutValueMismatch = 1ull << 23;         // WebGPU always restarts strips
inline constexpr uint64_t RenderTargetFormatMismatch = 1ull << 24;    // bound RTV formats != PSO RTVFormats
inline constexpr uint64_t DepthFormatMismatch = 1ull << 25;           // bound DSV format != PSO DSVFormat
inline constexpr uint64_t UnsupportedTargetFormat = 1ull << 26;       // fatal: RT/DS format has no mapping
inline constexpr uint64_t TargetFormatFallback = 1ull << 27;          // RT/DS format promoted (feature)
inline constexpr uint64_t ColorAttachmentBytesExceeded = 1ull << 28;  // fatal: > maxColorAttachmentBytesPerSample
inline constexpr uint64_t TooManyVertexAttributes = 1ull << 29;       // extras dropped (limit 16)
inline constexpr uint64_t VertexSlotOutOfRange = 1ull << 30;          // slot >= 8 dropped
inline constexpr uint64_t VertexFormatUnsupported = 1ull << 31;       // element dropped
inline constexpr uint64_t VertexLayoutInvalid = 1ull << 32;           // stride/offset breaks WebGPU rules
inline constexpr uint64_t MissingVertexInput = 1ull << 33;            // VS input not in the input layout
inline constexpr uint64_t InstanceStepRateUnsupported = 1ull << 34;   // step rate > 1 -> 1
inline constexpr uint64_t NullPixelShaderWithColorTargets = 1ull << 35; // layer must bind empty FS
inline constexpr uint64_t NoAttachments = 1ull << 36;                 // fatal: WebGPU needs >= 1 attachment
inline constexpr uint64_t InvalidEnumValue = 1ull << 37;              // invalid D3D12 enum, default used
inline constexpr uint64_t SampleCountMismatch = 1ull << 38;           // bound sample count != PSO

inline constexpr uint64_t kFatalMask = GeometryShaderUnsupported | TessellationUnsupported |
                                       MeshShaderUnsupported | MissingVertexShader |
                                       DualSourceMultipleTargets | UnsupportedTargetFormat |
                                       ColorAttachmentBytesExceeded | NoAttachments;
}  // namespace PipelineDiag

inline constexpr bool is_fatal(uint64_t pipeline_diagnostics) noexcept {
    return (pipeline_diagnostics & PipelineDiag::kFatalMask) != 0;
}

// =====================================================================================================
// Pipeline keys. Canonical, padding-free, fixed-size; every byte is meaningful or an explicitly zeroed
// reserved byte, so keys compare with memcmp and hash as raw bytes. A key is exactly what WebGPU's
// createRenderPipeline/createComputePipeline needs (besides shader modules looked up by id and the
// pipeline layout derived from root_signature_id), so it can travel verbatim in a creation packet.
// =====================================================================================================

inline constexpr uint32_t kMaxColorTargets = 8;     // maxColorAttachments (default limit)
inline constexpr uint32_t kMaxVertexBuffers = 8;    // maxVertexBuffers (default limit)
inline constexpr uint32_t kMaxVertexAttributes = 16;// maxVertexAttributes (default limit)
inline constexpr uint32_t kMaxVertexBufferArrayStride = 2048;

struct BlendComponentKey {
    WgpuBlendOperation operation;
    WgpuBlendFactor src_factor;
    WgpuBlendFactor dst_factor;
    uint8_t reserved;
};

struct ColorTargetKey {
    WgpuTextureFormat format;  // Undefined = null target (hole); every other field is then zero
    uint8_t blend_enabled;     // 0 = GPUColorTargetState.blend omitted
    uint8_t write_mask;        // GPUColorWrite bits (same bit values as D3D12)
    uint8_t reserved;
    BlendComponentKey color;
    BlendComponentKey alpha;
};

struct StencilFaceKey {
    WgpuCompareFunction compare;
    WgpuStencilOperation fail_op;
    WgpuStencilOperation depth_fail_op;
    WgpuStencilOperation pass_op;
};

struct DepthStencilKey {
    WgpuTextureFormat format;  // Undefined = no depthStencil state; every other field is then zero
    uint8_t depth_write_enabled;
    WgpuCompareFunction depth_compare;
    uint8_t reserved;
    StencilFaceKey stencil_front;
    StencilFaceKey stencil_back;
    uint32_t stencil_read_mask;
    uint32_t stencil_write_mask;
    int32_t depth_bias;
    uint32_t depth_bias_slope_scale_bits;  // IEEE-754 bits, -0/NaN canonicalised to +0
    uint32_t depth_bias_clamp_bits;
};

struct PrimitiveKey {
    WgpuPrimitiveTopology topology;
    WgpuIndexFormat strip_index_format;  // Undefined unless topology is a strip
    WgpuFrontFace front_face;
    WgpuCullMode cull_mode;
    uint8_t unclipped_depth;
    uint8_t reserved[3];
};

struct MultisampleKey {
    uint32_t count;  // 1 or 4
    uint32_t mask;   // already masked to `count` bits
    uint8_t alpha_to_coverage_enabled;
    uint8_t reserved[3];
};

struct VertexBufferKey {
    uint32_t array_stride;
    WgpuVertexStepMode step_mode;  // Unused = hole (emit null)
    uint8_t attribute_count;
    uint8_t reserved[2];
};

struct VertexAttributeKey {
    uint32_t offset;
    WgpuVertexFormat format;
    uint8_t shader_location;
    uint8_t buffer_index;  // == D3D12 input slot
    uint8_t reserved;
};

struct GraphicsPipelineKey {
    uint64_t vs_shader;
    uint64_t ps_shader;  // 0 = no fragment stage (or an empty FS when color targets exist)
    uint32_t root_signature_id;
    uint32_t color_target_count;  // highest non-null target + 1
    ColorTargetKey color_targets[kMaxColorTargets];
    DepthStencilKey depth_stencil;
    PrimitiveKey primitive;
    MultisampleKey multisample;
    uint32_t vertex_buffer_count;     // highest used slot + 1
    uint32_t vertex_attribute_count;  // attributes sorted by (buffer_index, shader_location)
    VertexBufferKey vertex_buffers[kMaxVertexBuffers];
    VertexAttributeKey vertex_attributes[kMaxVertexAttributes];
    uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<GraphicsPipelineKey>);
static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>,
              "GraphicsPipelineKey must not contain padding");
static_assert(sizeof(GraphicsPipelineKey) == 376, "GraphicsPipelineKey layout changed");

struct ComputePipelineKey {
    uint64_t cs_shader;
    uint32_t root_signature_id;
    uint32_t reserved;
};
static_assert(std::has_unique_object_representations_v<ComputePipelineKey>,
              "ComputePipelineKey must not contain padding");
static_assert(sizeof(ComputePipelineKey) == 16, "ComputePipelineKey layout changed");

bool operator==(const GraphicsPipelineKey& a, const GraphicsPipelineKey& b) noexcept;
bool operator==(const ComputePipelineKey& a, const ComputePipelineKey& b) noexcept;

// Stable (platform- and run-independent) 64-bit hash of raw bytes.
uint64_t stable_hash64(const void* data, size_t size) noexcept;
uint64_t hash_key(const GraphicsPipelineKey& key) noexcept;
uint64_t hash_key(const ComputePipelineKey& key) noexcept;

// =====================================================================================================
// PSO translation and variants
// =====================================================================================================

struct PipelineKeyOptions {
    uint32_t features = 0;                           // WgpuFeature bits enabled on the device
    uint32_t max_color_attachment_bytes_per_sample = 32;  // device limit (default limit is 32)
};

// Dynamic command-list state that WebGPU bakes into the pipeline. D3D12 sets these outside the PSO,
// so one PSO yields several WebGPU pipeline *variants*. Padding-free so it can be compared bytewise.
namespace VariantFlags {
inline constexpr uint32_t BoundTargets = 1u << 0;  // rtv/dsv formats + sample_count come from bound views
}  // namespace VariantFlags

struct PipelineVariantState {
    D3DPrimitiveTopology topology = D3DPrimitiveTopology::Undefined;  // Undefined = PSO type default
    uint32_t flags = 0;
    uint32_t num_render_targets = 0;
    DxgiFormat rtv_formats[kMaxColorTargets] = {};
    DxgiFormat dsv_format = DxgiFormat::UNKNOWN;
    uint32_t sample_count = 1;
    WgpuIndexFormat index_format = WgpuIndexFormat::Undefined;  // bound index buffer format
    uint8_t reserved[3] = {};
    uint32_t bound_vertex_slots = 0;  // bit i set = vertex_strides[i] comes from IASetVertexBuffers
    uint32_t vertex_strides[kMaxVertexBuffers] = {};
};
static_assert(std::has_unique_object_representations_v<PipelineVariantState>,
              "PipelineVariantState must not contain padding");

// PSO-level lowering, done once at CreateGraphicsPipelineState. `base` holds everything that does not
// depend on dynamic state (format-independent canonicalisation only); specialisation applies the variant.
struct GraphicsPsoTranslation {
    GraphicsPipelineKey base{};
    uint64_t diagnostics = 0;  // PSO-level PipelineDiag bits
    D3D12PrimitiveTopologyType topology_type = D3D12PrimitiveTopologyType::Triangle;
    D3D12IndexBufferStripCutValue strip_cut_value = D3D12IndexBufferStripCutValue::Disabled;
    uint32_t num_render_targets = 0;
    DxgiFormat rtv_formats[kMaxColorTargets] = {};
    DxgiFormat dsv_format = DxgiFormat::UNKNOWN;
    uint32_t sample_count = 1;   // raw D3D12 count
    uint32_t sample_mask = kD3D12DefaultSampleMask;
    uint32_t ps_output_mask = 0;
    uint32_t used_vertex_slots = 0;            // slots that carry at least one attribute
    uint32_t instance_stride_zero_slots = 0;   // per-instance slots with InstanceDataStepRate 0
    uint32_t tight_strides[kMaxVertexBuffers] = {};
};

GraphicsPsoTranslation translate_graphics_pso(const D3D12GraphicsPsoDesc& desc,
                                              const PipelineKeyOptions& options = {});

// Combine a translated PSO with dynamic state into a canonical key. Returns variant-level diagnostics
// (the PSO-level ones are in translation.diagnostics).
uint64_t specialize_graphics_pipeline_key(const GraphicsPsoTranslation& translation,
                                          const PipelineVariantState& variant,
                                          const PipelineKeyOptions& options,
                                          GraphicsPipelineKey* out_key);

struct GraphicsKeyResult {
    GraphicsPipelineKey key{};
    uint64_t diagnostics = 0;  // PSO-level | variant-level
};

// translate + specialise in one call. A default-constructed variant uses the PSO's own formats.
GraphicsKeyResult make_graphics_pipeline_key(const D3D12GraphicsPsoDesc& desc,
                                             const PipelineVariantState& variant = {},
                                             const PipelineKeyOptions& options = {});

ComputePipelineKey make_compute_pipeline_key(const D3D12ComputePsoDesc& desc) noexcept;

// =====================================================================================================
// Pipeline cache
// =====================================================================================================

enum class EvictionPolicy : uint8_t {
    Retain,             // keep pipelines after their last PSO is released (D3D12 titles re-create PSOs)
    EvictUnreferenced,  // drop a pipeline when its reference count reaches zero
};

struct PipelineCacheOptions {
    EvictionPolicy eviction = EvictionPolicy::Retain;
    // Allocates pipeline ids (must be thread-safe and never return 0). Default: internal counter from 1.
    std::function<uint32_t()> id_allocator;
    // Test hook: replaces stable_hash64 (e.g. to force collisions).
    uint64_t (*hash_override)(const void* data, size_t size) = nullptr;
};

struct PipelineCacheStats {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;           // == pipelines created
    uint64_t variants = 0;         // pipelines a PSO needed beyond its first one
    uint64_t hash_collisions = 0;  // distinct keys sharing a 64-bit hash
    uint64_t evictions = 0;
    uint64_t live_graphics = 0;
    uint64_t live_compute = 0;
};

struct CacheLookup {
    uint32_t pipeline_id = 0;
    bool created = false;  // true = miss; the caller must emit the pipeline-creation packet
};

class PipelineCache {
public:
    explicit PipelineCache(PipelineCacheOptions options = {});
    ~PipelineCache();
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    // Find-or-create plus one reference. Cost: one hash over the key + one shard lookup.
    CacheLookup acquire(const GraphicsPipelineKey& key);
    CacheLookup acquire(const ComputePipelineKey& key);

    // Drop one reference. Returns the pipeline id when the entry was evicted (the caller then emits
    // DestroyResource for it).
    std::optional<uint32_t> release(const GraphicsPipelineKey& key);
    std::optional<uint32_t> release(const ComputePipelineKey& key);

    std::optional<uint32_t> find(const GraphicsPipelineKey& key) const;
    std::optional<uint32_t> find(const ComputePipelineKey& key) const;
    uint32_t ref_count(const GraphicsPipelineKey& key) const;

    PipelineCacheStats stats() const;
    void note_variant() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct PipelineResolution {
    uint32_t pipeline_id = 0;
    bool created = false;                       // emit the creation packet using *key
    const GraphicsPipelineKey* key = nullptr;   // valid while the GraphicsPso lives
    uint64_t diagnostics = 0;                   // PSO-level | variant-level
    bool usable() const noexcept { return pipeline_id != 0 && !is_fatal(diagnostics); }
};

// A created graphics PSO: its translation plus the variants it has resolved so far. Thread-safe.
class GraphicsPso {
public:
    explicit GraphicsPso(const D3D12GraphicsPsoDesc& desc, const PipelineKeyOptions& options = {});
    ~GraphicsPso();
    GraphicsPso(const GraphicsPso&) = delete;
    GraphicsPso& operator=(const GraphicsPso&) = delete;

    const GraphicsPsoTranslation& translation() const noexcept { return translation_; }
    uint64_t diagnostics() const noexcept { return translation_.diagnostics; }

    // Resolve the pipeline for the current dynamic state. Repeated states hit a per-PSO table (the most
    // recent variant is checked first) without touching the shared cache. Fatal keys are not cached
    // and return pipeline_id 0.
    PipelineResolution resolve(PipelineCache& cache, const PipelineVariantState& variant);

    // Release every reference this PSO holds (call on final Release of the PSO). Returns the pipeline ids
    // the cache evicted.
    std::vector<uint32_t> release(PipelineCache& cache);

    size_t variant_count() const;

private:
    struct Variant {
        PipelineVariantState state;
        GraphicsPipelineKey key;
        uint32_t pipeline_id;
        uint64_t diagnostics;
    };

    PipelineVariantState canonical_variant(const PipelineVariantState& variant) const noexcept;

    GraphicsPsoTranslation translation_;
    PipelineKeyOptions options_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Variant>> variants_;
    size_t last_hit_ = 0;
};

// =====================================================================================================
// Output-merger / rasterizer state tracking and render pass inference
// =====================================================================================================

// Resolved RTV/DSV descriptor contents. view_id is an opaque id from the descriptor-heap layer that
// identifies the view (0 = null descriptor); width/height are the mip-level extent.
struct RenderTargetViewInfo {
    uint32_t view_id = 0;
    DxgiFormat format = DxgiFormat::UNKNOWN;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sample_count = 1;
};

struct DepthStencilViewInfo {
    uint32_t view_id = 0;
    DxgiFormat format = DxgiFormat::UNKNOWN;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sample_count = 1;
    uint32_t flags = 0;  // D3D12DsvFlags
};

// Implemented by the descriptor-heap layer: turns CPU descriptor handles into view contents.
class DescriptorViewResolver {
public:
    virtual ~DescriptorViewResolver() = default;
    virtual uint64_t rtv_descriptor_increment() const = 0;
    virtual bool resolve_rtv(uint64_t cpu_handle, RenderTargetViewInfo* out) const = 0;
    virtual bool resolve_dsv(uint64_t cpu_handle, DepthStencilViewInfo* out) const = 0;
};

struct WgpuViewport {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float min_depth = 0.0f;
    float max_depth = 1.0f;
    bool operator==(const WgpuViewport&) const = default;
};

struct WgpuScissorRect {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    bool operator==(const WgpuScissorRect&) const = default;
};

struct ColorAttachmentDesc {
    uint32_t view_id = 0;  // 0 = null attachment (hole)
    WgpuTextureFormat format = WgpuTextureFormat::Undefined;
    WgpuLoadOp load_op = WgpuLoadOp::Undefined;
    WgpuStoreOp store_op = WgpuStoreOp::Undefined;
    std::array<float, 4> clear_value = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct DepthStencilAttachmentDesc {
    uint32_t view_id = 0;
    WgpuTextureFormat format = WgpuTextureFormat::Undefined;
    WgpuLoadOp depth_load_op = WgpuLoadOp::Undefined;    // Undefined when read-only or no depth aspect
    WgpuStoreOp depth_store_op = WgpuStoreOp::Undefined;
    float depth_clear_value = 1.0f;
    WgpuLoadOp stencil_load_op = WgpuLoadOp::Undefined;  // Undefined when read-only or no stencil aspect
    WgpuStoreOp stencil_store_op = WgpuStoreOp::Undefined;
    uint32_t stencil_clear_value = 0;
    bool depth_read_only = false;
    bool stencil_read_only = false;
};

struct RenderPassDesc {
    uint32_t color_count = 0;  // highest non-null attachment + 1
    std::array<ColorAttachmentDesc, kMaxColorTargets> color{};
    bool has_depth_stencil = false;
    DepthStencilAttachmentDesc depth_stencil{};
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sample_count = 1;
};

// Partial-rect (or, with MidPassClearPolicy::DrawFallback, mid-pass) clear that WebGPU cannot express
// as a load op. The encoder layer must draw it inside the current pass, e.g. one scissored full-screen
// triangle per rect with a clear pipeline (depth: compare always + depth write; stencil: replace).
struct ClearFallback {
    uint32_t view_id = 0;
    bool depth_stencil = false;
    uint32_t attachment_index = 0;   // color slot (unused for depth-stencil)
    uint32_t clear_flags = 0;        // D3D12ClearFlags for depth-stencil
    std::array<float, 4> color = {0.0f, 0.0f, 0.0f, 0.0f};
    float depth = 1.0f;
    uint8_t stencil = 0;
    std::vector<WgpuScissorRect> rects;  // already clipped to the view
};

enum class StateEventType : uint8_t {
    BeginRenderPass,     // index -> passes()
    EndRenderPass,
    SetPipeline,         // pipeline_id; pipeline_key != nullptr when it must be created first
    SetViewport,         // viewport
    SetScissorRect,      // scissor
    SetBlendConstant,    // blend_constant
    SetStencilReference, // stencil_reference
    ClearFallback,       // index -> clear_fallbacks(); leaves viewport/scissor/pipeline undefined
};

struct StateEvent {
    StateEventType type = StateEventType::EndRenderPass;
    uint32_t index = 0;
    uint32_t pipeline_id = 0;
    const GraphicsPipelineKey* pipeline_key = nullptr;
    WgpuViewport viewport{};
    WgpuScissorRect scissor{};
    std::array<float, 4> blend_constant = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t stencil_reference = 0;
};

enum class MidPassClearPolicy : uint8_t {
    SplitPass,     // end the pass; the next pass folds the clear into loadOp "clear" (exact)
    DrawFallback,  // keep the pass; emit a ClearFallback covering the whole view
};

namespace StateDiag {
inline constexpr uint64_t MultipleViewportsIgnored = 1ull << 0;   // only viewport/scissor 0 is used
inline constexpr uint64_t ViewportClamped = 1ull << 1;            // outside WebGPU viewport limits
inline constexpr uint64_t ViewportDepthInverted = 1ull << 2;      // MinDepth > MaxDepth: swapped
inline constexpr uint64_t ViewportNotSet = 1ull << 3;             // draw without RSSetViewports
inline constexpr uint64_t ScissorNotSet = 1ull << 4;              // draw without RSSetScissorRects
inline constexpr uint64_t AttachmentSizeMismatch = 1ull << 5;     // WebGPU needs equal sizes
inline constexpr uint64_t AttachmentSampleCountMismatch = 1ull << 6;
inline constexpr uint64_t UnresolvedDescriptor = 1ull << 7;       // resolver failed for a handle
inline constexpr uint64_t DrawWithoutAttachments = 1ull << 8;     // draw skipped
inline constexpr uint64_t DrawWithoutPipeline = 1ull << 9;        // draw skipped
inline constexpr uint64_t PipelineUnusable = 1ull << 10;          // fatal pipeline diagnostics: skipped
inline constexpr uint64_t ClearOnReadOnlyAspect = 1ull << 11;     // clear dropped
inline constexpr uint64_t UnsupportedAttachmentFormat = 1ull << 12;
}  // namespace StateDiag

struct TrackerOptions {
    MidPassClearPolicy mid_pass_clear = MidPassClearPolicy::SplitPass;
    uint32_t features = 0;                      // WgpuFeature bits (attachment format mapping)
    uint32_t max_texture_dimension_2d = 8192;   // device limit used for viewport validation
};

struct TrackerStats {
    uint64_t passes_begun = 0;
    uint64_t clears_folded = 0;          // full clears turned into loadOp "clear"
    uint64_t clear_fallbacks = 0;
    uint64_t clear_only_passes = 0;      // pending clears flushed as their own pass
    uint64_t redundant_calls_filtered = 0;
    uint64_t draws = 0;
    uint64_t draws_skipped = 0;
};

// Per-command-list tracker. Mirrors the D3D12 calls that touch output-merger, rasterizer and
// input-assembler dynamic state, filters redundant state, and infers WebGPU render passes:
//
// * A pass begins lazily at the first draw after the bound attachment set (RTVs, DSV, DSV read-only
//   flags) changed; OMSetRenderTargets alone never begins or ends a pass, so A->B->A switches without a
//   draw in between cost nothing.
// * A full clear of a view that is not attached to the open pass is recorded as pending and folded into
//   the loadOp/clearValue of the next pass that attaches the view (last clear wins; depth and stencil
//   clears merge).
// * A full clear of a view attached to the open pass ends the pass (SplitPass, default) or emits a
//   ClearFallback (DrawFallback).
// * A clear with rects not covering the whole view emits a ClearFallback inside a pass that attaches the
//   view: the open pass, else a pass for the bound attachments (when they include the view), else a
//   temporary pass with just that view.
// * Pending clears are flushed as clear-only passes by flush_pending_clears() / close(). The layer above
//   must call flush_pending_clears() before anything that reads or copies a render target (resource
//   barriers, copies, resolves) and end_render_pass() before compute/copy work.
// * WebGPU resets viewport, scissor, blend constant, stencil reference and pipeline at every pass begin,
//   so prepare_draw() re-emits whichever differ from the WebGPU defaults after a pass begins, and only
//   changed values within a pass.
class CommandListStateTracker {
public:
    explicit CommandListStateTracker(TrackerOptions options = {});

    // ID3D12GraphicsCommandList::Reset: all state back to defaults, events cleared.
    void reset();

    void OMSetRenderTargets(uint32_t num_render_target_descriptors,
                            const uint64_t* render_target_descriptors,
                            bool rts_single_handle_to_descriptor_range,
                            const uint64_t* depth_stencil_descriptor,
                            const DescriptorViewResolver& resolver);
    // Same, with already-resolved descriptors (null view_id = unbound slot).
    void set_render_targets(std::span<const RenderTargetViewInfo> rtvs, const DepthStencilViewInfo* dsv);

    void ClearRenderTargetView(const RenderTargetViewInfo& rtv,
                               const float color_rgba[4],
                               uint32_t num_rects = 0,
                               const D3D12Rect* rects = nullptr);
    void ClearDepthStencilView(const DepthStencilViewInfo& dsv,
                               uint32_t clear_flags,
                               float depth,
                               uint8_t stencil,
                               uint32_t num_rects = 0,
                               const D3D12Rect* rects = nullptr);

    void RSSetViewports(uint32_t num_viewports, const D3D12Viewport* viewports);
    void RSSetScissorRects(uint32_t num_rects, const D3D12Rect* rects);
    void OMSetBlendFactor(const float blend_factor[4]);  // nullptr = {1,1,1,1}
    void OMSetStencilRef(uint32_t stencil_ref);
    void IASetPrimitiveTopology(D3DPrimitiveTopology topology);
    // Only the parts of IASetIndexBuffer / IASetVertexBuffers that WebGPU bakes into pipelines.
    void set_index_format(WgpuIndexFormat format);
    void set_vertex_strides(uint32_t start_slot, uint32_t num_views, const uint32_t* strides);
    void SetPipelineState(GraphicsPso* pso);

    // Call before every draw. Begins/ends passes and emits state; returns false when the draw must be
    // skipped (see diagnostics()). `cache` may be null to track state without resolving pipelines.
    bool prepare_draw(PipelineCache* cache);

    void flush_pending_clears();
    void end_render_pass();
    void close();  // flush_pending_clears() + end_render_pass()

    const std::vector<StateEvent>& events() const noexcept { return events_; }
    const std::vector<RenderPassDesc>& passes() const noexcept { return passes_; }
    const std::vector<ClearFallback>& clear_fallbacks() const noexcept { return clear_fallbacks_; }
    void clear_events();  // after the encoder consumed them

    bool in_render_pass() const noexcept { return pass_open_; }
    const PipelineVariantState& variant_state() const noexcept { return variant_; }
    uint64_t diagnostics() const noexcept { return diagnostics_; }
    uint64_t pipeline_diagnostics() const noexcept { return pipeline_diagnostics_; }
    const TrackerStats& stats() const noexcept { return stats_; }

    // Pure helpers (exposed for tests): D3D12 -> WebGPU dynamic state for a pass of the given size.
    static WgpuViewport convert_viewport(const D3D12Viewport& viewport,
                                         uint32_t max_texture_dimension_2d,
                                         uint64_t* diagnostics);
    static WgpuScissorRect convert_scissor(const D3D12Rect& rect, uint32_t width, uint32_t height);

private:
    struct Attachments {
        std::array<RenderTargetViewInfo, kMaxColorTargets> rtvs{};
        uint32_t num_rtvs = 0;
        DepthStencilViewInfo dsv{};
        bool operator==(const Attachments& other) const noexcept;
        bool empty() const noexcept;
        bool contains(uint32_t view_id) const noexcept;
    };

    struct PendingClear {
        uint32_t view_id = 0;
        bool depth_stencil = false;
        RenderTargetViewInfo rtv{};
        DepthStencilViewInfo dsv{};
        std::array<float, 4> color = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t clear_flags = 0;
        float depth = 1.0f;
        uint8_t stencil = 0;
    };

    void begin_pass(const Attachments& attachments);
    void record_pending_clear(const PendingClear& clear);
    PendingClear* find_pending(uint32_t view_id);
    void emit_clear_fallback(ClearFallback fallback, const Attachments& temp_pass);
    void update_variant_targets();

    TrackerOptions options_;
    std::vector<StateEvent> events_;
    std::vector<RenderPassDesc> passes_;
    std::vector<ClearFallback> clear_fallbacks_;
    std::vector<PendingClear> pending_clears_;

    Attachments bound_{};
    Attachments pass_attachments_{};
    bool pass_open_ = false;
    bool pass_state_known_ = false;  // false right after a ClearFallback clobbered encoder state
    uint32_t pass_width_ = 0;
    uint32_t pass_height_ = 0;

    // D3D12 command-list state.
    bool viewport_set_ = false;
    bool scissor_set_ = false;
    D3D12Viewport viewport_{};
    D3D12Rect scissor_{};
    std::array<float, 4> blend_factor_ = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t stencil_ref_ = 0;
    GraphicsPso* pso_ = nullptr;
    PipelineVariantState variant_{};

    // WebGPU state currently applied in the open pass.
    WgpuViewport pass_viewport_{};
    WgpuScissorRect pass_scissor_{};
    std::array<float, 4> pass_blend_constant_ = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t pass_stencil_ref_ = 0;
    uint32_t pass_pipeline_id_ = 0;

    uint64_t diagnostics_ = 0;
    uint64_t pipeline_diagnostics_ = 0;
    TrackerStats stats_{};
};

}  // namespace playgta6::gpu::pso
