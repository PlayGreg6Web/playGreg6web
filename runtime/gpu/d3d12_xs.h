#pragma once

#include <cstdint>
#include <string_view>

#include "gpu/cmd_encoder.h"

namespace playgta6::gpu::d3d12_xs {

using HRESULT = int32_t;

inline constexpr HRESULT S_OK = 0;
inline constexpr HRESULT E_INVALIDARG = static_cast<HRESULT>(0x80070057u);
inline constexpr HRESULT E_FAIL = static_cast<HRESULT>(0x80004005u);

inline constexpr bool Succeeded(HRESULT hr) noexcept {
    return hr >= 0;
}

inline constexpr bool Failed(HRESULT hr) noexcept {
    return hr < 0;
}

// Minimal COM-like intrusive reference-counting base for d3d12_xs shims.
class D3D12Object {
public:
    virtual ~D3D12Object() = default;

    uint32_t AddRef() noexcept { return ++ref_count_; }

    uint32_t Release() noexcept {
        if (ref_count_ > 0) {
            --ref_count_;
        }
        return ref_count_;
    }

    uint32_t ref_count() const noexcept { return ref_count_; }

private:
    uint32_t ref_count_ = 1;
};

struct D3D12ResourceHandle {
    uint32_t id = 0;
    uint64_t byte_size = 0;
    uint32_t usage_flags = 0;
};

struct D3D12ShaderHandle {
    uint32_t id = 0;
    GpuShaderStage stage = GpuShaderStage::Vertex;
};

struct D3D12PipelineStateHandle {
    uint32_t id = 0;
};

struct D3D12BufferDesc {
    uint64_t width = 0;
    uint32_t usage_flags = GpuBufferUsage::Vertex | GpuBufferUsage::CopyDst;
};

struct D3D12GraphicsPipelineStateDesc {
    uint32_t vs_shader_id = 0;
    uint32_t fs_shader_id = 0;
    uint32_t vertex_stride = 0;
    GpuPrimitiveTopology topology = GpuPrimitiveTopology::TriangleList;
    uint32_t flags = 0;
};

struct D3D12VertexBufferView {
    uint32_t buffer_id = 0;
    uint64_t offset = 0;
    uint32_t size_in_bytes = 0;
    uint32_t stride_in_bytes = 0;
};

// Minimal d3d12_xs graphics command list shim recording directly into CommandEncoder.
class D3D12GraphicsCommandList : public D3D12Object {
public:
    explicit D3D12GraphicsCommandList(CommandEncoder* encoder = nullptr) noexcept
        : encoder_(encoder) {}

    HRESULT Reset() {
        if (encoder_ == nullptr) {
            return E_FAIL;
        }
        is_open_ = true;
        encoder_->begin_frame();
        return S_OK;
    }

    HRESULT Close() noexcept {
        if (!is_open_) {
            return E_FAIL;
        }
        is_open_ = false;
        return S_OK;
    }

    bool is_open() const noexcept { return is_open_; }

    void ClearRenderTargetView(const float color_rgba[4]) {
        if (!is_open_ || encoder_ == nullptr || color_rgba == nullptr) {
            return;
        }
        encoder_->clear_color(color_rgba[0], color_rgba[1], color_rgba[2], color_rgba[3]);
    }

    void CopyBufferRegion(
        const D3D12ResourceHandle& dst_resource,
        uint32_t dst_offset,
        const void* src_data,
        uint32_t num_bytes) {
        if (!is_open_ || encoder_ == nullptr || dst_resource.id == 0) {
            return;
        }
        encoder_->write_buffer(dst_resource.id, dst_offset, src_data, num_bytes);
    }

    void SetPipelineState(const D3D12PipelineStateHandle& pso) {
        if (!is_open_ || encoder_ == nullptr || pso.id == 0) {
            return;
        }
        encoder_->set_pipeline(pso.id);
    }

    void IASetVertexBuffers(uint32_t start_slot, const D3D12VertexBufferView& view) {
        if (!is_open_ || encoder_ == nullptr || view.buffer_id == 0) {
            return;
        }
        encoder_->set_vertex_buffer(start_slot, view.buffer_id, view.offset);
    }

    void DrawInstanced(
        uint32_t vertex_count_per_instance,
        uint32_t instance_count,
        uint32_t start_vertex_location,
        uint32_t start_instance_location) {
        if (!is_open_ || encoder_ == nullptr) {
            return;
        }
        encoder_->draw(
            vertex_count_per_instance,
            instance_count,
            start_vertex_location,
            start_instance_location);
    }

private:
    CommandEncoder* encoder_ = nullptr;
    bool is_open_ = false;
};

// Minimal Xbox Series PresentX shim emitting Present packets into CommandEncoder.
class PresentX : public D3D12Object {
public:
    explicit PresentX(CommandEncoder* encoder = nullptr) noexcept : encoder_(encoder) {}

    HRESULT Present(uint32_t sync_interval = 1, uint32_t flags = 0) {
        if (encoder_ == nullptr) {
            return E_FAIL;
        }
        last_sync_interval_ = sync_interval;
        last_flags_ = flags;
        encoder_->present();
        ++present_count_;
        return S_OK;
    }

    HRESULT operator()(uint32_t sync_interval = 1, uint32_t flags = 0) {
        return Present(sync_interval, flags);
    }

    uint64_t present_count() const noexcept { return present_count_; }
    uint32_t last_sync_interval() const noexcept { return last_sync_interval_; }
    uint32_t last_flags() const noexcept { return last_flags_; }

private:
    CommandEncoder* encoder_ = nullptr;
    uint64_t present_count_ = 0;
    uint32_t last_sync_interval_ = 0;
    uint32_t last_flags_ = 0;
};

// Minimal d3d12_xs device shim allocating monotonic resource IDs and emitting resource packets.
class D3D12Device : public D3D12Object {
public:
    explicit D3D12Device(CommandEncoder* encoder = nullptr) noexcept : encoder_(encoder) {}

    uint32_t allocate_resource_id() noexcept { return next_resource_id_++; }

    HRESULT CreateCommittedResource(
        const D3D12BufferDesc& desc,
        D3D12ResourceHandle* out_resource) {
        if (encoder_ == nullptr || out_resource == nullptr || desc.width == 0) {
            return E_INVALIDARG;
        }
        const uint32_t id = allocate_resource_id();
        encoder_->create_buffer(id, desc.usage_flags, desc.width);
        *out_resource = D3D12ResourceHandle{id, desc.width, desc.usage_flags};
        return S_OK;
    }

    HRESULT UploadBufferData(
        const D3D12ResourceHandle& resource,
        uint32_t dst_offset,
        const void* data,
        uint32_t data_len) {
        if (encoder_ == nullptr || resource.id == 0 || (data == nullptr && data_len > 0)) {
            return E_INVALIDARG;
        }
        encoder_->write_buffer(resource.id, dst_offset, data, data_len);
        return S_OK;
    }

    HRESULT CreateShaderModule(
        GpuShaderStage stage,
        std::string_view wgsl_code,
        D3D12ShaderHandle* out_shader) {
        if (encoder_ == nullptr || out_shader == nullptr || wgsl_code.empty()) {
            return E_INVALIDARG;
        }
        const uint32_t id = allocate_resource_id();
        encoder_->create_shader_module(id, stage, wgsl_code);
        *out_shader = D3D12ShaderHandle{id, stage};
        return S_OK;
    }

    HRESULT CreateGraphicsPipelineState(
        const D3D12GraphicsPipelineStateDesc& desc,
        D3D12PipelineStateHandle* out_pso) {
        if (encoder_ == nullptr || out_pso == nullptr || desc.vs_shader_id == 0) {
            return E_INVALIDARG;
        }
        const uint32_t id = allocate_resource_id();
        encoder_->create_render_pipeline(
            id,
            desc.vs_shader_id,
            desc.fs_shader_id,
            desc.vertex_stride,
            desc.topology,
            desc.flags);
        *out_pso = D3D12PipelineStateHandle{id};
        return S_OK;
    }

    HRESULT CreateCommandList(D3D12GraphicsCommandList* out_list) {
        if (encoder_ == nullptr || out_list == nullptr) {
            return E_INVALIDARG;
        }
        *out_list = D3D12GraphicsCommandList(encoder_);
        return S_OK;
    }

    HRESULT CreatePresentX(PresentX* out_present) {
        if (encoder_ == nullptr || out_present == nullptr) {
            return E_INVALIDARG;
        }
        *out_present = PresentX(encoder_);
        return S_OK;
    }

    HRESULT DestroyResource(uint32_t resource_id) {
        if (encoder_ == nullptr || resource_id == 0) {
            return E_INVALIDARG;
        }
        encoder_->destroy_resource(resource_id);
        return S_OK;
    }

private:
    CommandEncoder* encoder_ = nullptr;
    uint32_t next_resource_id_ = 1;
};

}  // namespace playgta6::gpu::d3d12_xs
