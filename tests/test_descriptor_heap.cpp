#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

#include "gpu/descriptor_heap.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_descriptor_heap.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu::d3d12_state;

DescriptorHeap* make_heap(DescriptorHeapRegistry& reg, DescriptorHeapType type, uint32_t n, bool shader_visible) {
    DescriptorHeapDesc desc;
    desc.type = type;
    desc.num_descriptors = n;
    desc.flags = shader_visible ? DescriptorHeapFlags::ShaderVisible : DescriptorHeapFlags::None;
    DescriptorHeap* heap = nullptr;
    CHECK(reg.create_heap(desc, &heap) == Status::Ok);
    CHECK(heap != nullptr);
    return heap;
}

CpuDescriptorHandle cpu_at(const DescriptorHeap* heap, uint32_t index) {
    return CpuDescriptorHandle{heap->cpu_start().ptr + static_cast<uint64_t>(index) * heap->increment()};
}

GpuDescriptorHandle gpu_at(const DescriptorHeap* heap, uint32_t index) {
    return GpuDescriptorHandle{heap->gpu_start().ptr + static_cast<uint64_t>(index) * heap->increment()};
}

ViewDesc texture2d_srv(uint32_t first_mip = 0, uint32_t mip_levels = 1) {
    ViewDesc d;
    d.format = 28;  // DXGI_FORMAT_R8G8B8A8_UNORM
    d.dimension = ViewDimension::Texture2D;
    d.first_mip = first_mip;
    d.mip_levels = mip_levels;
    return d;
}

// Fake GPU VA space for the bind-group tests: buffer 70 at 0x10000 (64 KiB),
// buffer 80 at 0x100000 (128 KiB).
bool test_resolver(void* user, uint64_t va, GpuVaResolution* out) {
    auto* calls = static_cast<int*>(user);
    if (calls != nullptr) {
        ++*calls;
    }
    if (va >= 0x10000 && va < 0x20000) {
        *out = GpuVaResolution{70, va - 0x10000, 0x10000};
        return true;
    }
    if (va >= 0x100000 && va < 0x120000) {
        *out = GpuVaResolution{80, va - 0x100000, 0x20000};
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// 1. Heap creation and handle arithmetic
// ---------------------------------------------------------------------------

// Backlog 28: live heap / slot gauges follow create, destroy and registry teardown.
void test_live_heap_gauges() {
    const DescriptorLiveCounts base = descriptor_live_counts();
    {
        DescriptorHeapRegistry reg;
        DescriptorHeapDesc d;
        d.type = DescriptorHeapType::CbvSrvUav;
        d.num_descriptors = 16;
        DescriptorHeap* a = nullptr;
        DescriptorHeap* b = nullptr;
        CHECK(reg.create_heap(d, &a) == Status::Ok);
        d.num_descriptors = 8;
        CHECK(reg.create_heap(d, &b) == Status::Ok);
        DescriptorLiveCounts c = descriptor_live_counts();
        CHECK(c.heaps == base.heaps + 2 && c.heap_slots == base.heap_slots + 24);
        reg.destroy_heap(a);
        reg.destroy_heap(a);  // double destroy is ignored
        c = descriptor_live_counts();
        CHECK(c.heaps == base.heaps + 1 && c.heap_slots == base.heap_slots + 8);
    }
    const DescriptorLiveCounts end = descriptor_live_counts();
    CHECK(end.heaps == base.heaps && end.heap_slots == base.heap_slots);
}

void test_heap_creation_and_handles() {
    DescriptorHeapRegistry reg;
    CHECK(descriptor_handle_increment_size(DescriptorHeapType::CbvSrvUav) == 32);
    CHECK(descriptor_handle_increment_size(DescriptorHeapType::Sampler) == 16);
    CHECK(descriptor_handle_increment_size(DescriptorHeapType::Rtv) == 32);
    CHECK(descriptor_handle_increment_size(DescriptorHeapType::Dsv) == 32);

    DescriptorHeap* bad = reinterpret_cast<DescriptorHeap*>(1);
    DescriptorHeapDesc d;
    d.num_descriptors = 0;
    CHECK(reg.create_heap(d, &bad) == Status::InvalidArgument);
    CHECK(bad == nullptr);
    d.num_descriptors = 4;
    d.type = DescriptorHeapType::Rtv;
    d.flags = DescriptorHeapFlags::ShaderVisible;
    CHECK(reg.create_heap(d, &bad) == Status::InvalidArgument);
    d.type = DescriptorHeapType::Dsv;
    CHECK(reg.create_heap(d, &bad) == Status::InvalidArgument);
    d.type = DescriptorHeapType::Sampler;
    d.num_descriptors = kMaxShaderVisibleSamplers + 1;
    CHECK(reg.create_heap(d, &bad) == Status::InvalidArgument);
    d.type = static_cast<DescriptorHeapType>(7);
    d.num_descriptors = 4;
    CHECK(reg.create_heap(d, &bad) == Status::InvalidArgument);
    d.type = DescriptorHeapType::CbvSrvUav;
    d.flags = 0x10;
    CHECK(reg.create_heap(d, &bad) == Status::InvalidArgument);
    d.flags = 0;
    d.num_descriptors = kMaxDescriptorsPerHeap + 1;
    CHECK(reg.create_heap(d, &bad) == Status::InvalidArgument);
    CHECK(reg.create_heap(d, nullptr) == Status::InvalidArgument);
    CHECK(reg.heap_count() == 0);

    DescriptorHeap* cbv = make_heap(reg, DescriptorHeapType::CbvSrvUav, 100, true);
    DescriptorHeap* staging = make_heap(reg, DescriptorHeapType::CbvSrvUav, 100, false);
    DescriptorHeap* rtv = make_heap(reg, DescriptorHeapType::Rtv, 8, false);
    CHECK(reg.heap_count() == 3);
    CHECK(cbv->capacity() == 100);
    CHECK(cbv->shader_visible());
    CHECK(!staging->shader_visible());
    CHECK(cbv->cpu_start().ptr != 0);
    CHECK(cbv->gpu_start().ptr != 0);
    CHECK(staging->gpu_start().ptr == 0);  // non-shader-visible: no GPU handle
    CHECK(cbv->cpu_start().ptr != staging->cpu_start().ptr);
    CHECK(cbv->cpu_start().ptr != cbv->gpu_start().ptr);

    // Raw pointer arithmetic round-trips to (heap, index).
    for (uint32_t i : {0u, 1u, 50u, 99u}) {
        ResolvedDescriptor r;
        CHECK(reg.resolve_cpu(cpu_at(cbv, i), &r) == Status::Ok);
        CHECK(r.heap == cbv && r.index == i);
        CHECK(reg.resolve_gpu(gpu_at(cbv, i), &r) == Status::Ok);
        CHECK(r.heap == cbv && r.index == i);
        CHECK(cbv->cpu_handle(i).ptr == cpu_at(cbv, i).ptr);
        CHECK(cbv->gpu_handle(i).ptr == gpu_at(cbv, i).ptr);
    }
    ResolvedDescriptor r;
    CHECK(reg.resolve_cpu(cpu_at(rtv, 7), &r) == Status::Ok && r.heap == rtv && r.index == 7);

    // Validation of bad handles.
    CHECK(reg.resolve_cpu(cpu_at(cbv, 100), &r) == Status::HandleOutOfRange);
    CHECK(reg.resolve_cpu(CpuDescriptorHandle{cbv->cpu_start().ptr + 1}, &r) == Status::MisalignedHandle);
    CHECK(reg.resolve_cpu(CpuDescriptorHandle{cbv->cpu_start().ptr + 16}, &r) == Status::MisalignedHandle);
    CHECK(reg.resolve_cpu(CpuDescriptorHandle{cbv->gpu_start().ptr}, &r) == Status::InvalidHandle);
    CHECK(reg.resolve_gpu(GpuDescriptorHandle{cbv->cpu_start().ptr}, &r) == Status::InvalidHandle);
    CHECK(reg.resolve_cpu(CpuDescriptorHandle{0}, &r) == Status::InvalidHandle);
    CHECK(reg.resolve_cpu(CpuDescriptorHandle{0x00007ff612345678ull}, &r) == Status::InvalidHandle);
    CHECK(reg.resolve_cpu(CpuDescriptorHandle{0x4000ffff00000000ull}, &r) == Status::StaleHandle);
    // Underflow below the heap start wraps into another heap id: never resolves to this heap.
    const Status under = reg.resolve_cpu(CpuDescriptorHandle{cbv->cpu_start().ptr - 32}, &r);
    CHECK(under != Status::Ok || r.heap != cbv);

    // Destroyed heap handles become stale; slot reuse does not revive them.
    const CpuDescriptorHandle old_cpu = staging->cpu_start();
    const uint32_t old_serial = staging->serial();
    reg.destroy_heap(staging);
    CHECK(reg.heap_count() == 2);
    CHECK(reg.resolve_cpu(old_cpu, &r) == Status::StaleHandle);
    DescriptorHeap* reused = make_heap(reg, DescriptorHeapType::CbvSrvUav, 100, false);
    CHECK((reused->serial() & 0xffffu) == (old_serial & 0xffffu));
    CHECK(reused->serial() != old_serial);
    CHECK(reg.resolve_cpu(old_cpu, &r) == Status::StaleHandle);
    CHECK(reg.resolve_cpu(reused->cpu_start(), &r) == Status::Ok && r.heap == reused);
    reg.destroy_heap(reused);
    reg.destroy_heap(reused);  // double destroy is ignored
    reg.destroy_heap(nullptr);
    CHECK(reg.heap_count() == 2);
}

// ---------------------------------------------------------------------------
// 2. Descriptor writes
// ---------------------------------------------------------------------------

void test_descriptor_writes() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 16, false);
    DescriptorHeap* samplers = make_heap(reg, DescriptorHeapType::Sampler, 4, false);
    DescriptorHeap* rtvs = make_heap(reg, DescriptorHeapType::Rtv, 4, false);
    DescriptorHeap* dsvs = make_heap(reg, DescriptorHeapType::Dsv, 4, false);

    CHECK(heap->view(0)->kind == DescriptorKind::Empty);
    CHECK(heap->generation() == 0);

    ConstantBufferViewDesc cbv{0x10100, 512};
    CHECK(reg.create_constant_buffer_view(&cbv, cpu_at(heap, 0)) == Status::Ok);
    CHECK(heap->view(0)->kind == DescriptorKind::Cbv);
    CHECK(heap->view(0)->first_element_or_address == 0x10100);
    CHECK(heap->view(0)->num_elements_or_size == 512);
    ConstantBufferViewDesc bad_cbv{0x10100, 100};
    CHECK(reg.create_constant_buffer_view(&bad_cbv, cpu_at(heap, 1)) == Status::InvalidArgument);
    bad_cbv = ConstantBufferViewDesc{0x10001, 256};
    CHECK(reg.create_constant_buffer_view(&bad_cbv, cpu_at(heap, 1)) == Status::InvalidArgument);
    CHECK(heap->view(1)->kind == DescriptorKind::Empty);
    CHECK(reg.create_constant_buffer_view(nullptr, cpu_at(heap, 1)) == Status::Ok);  // null CBV
    CHECK(heap->view(1)->kind == DescriptorKind::Cbv && heap->view(1)->first_element_or_address == 0);

    ViewDesc srv = texture2d_srv(2, 3);
    CHECK(reg.create_shader_resource_view(42, &srv, cpu_at(heap, 2)) == Status::Ok);
    const ViewDescriptor* v = heap->view(2);
    CHECK(v->kind == DescriptorKind::Srv && v->resource_id == 42 && v->format == 28);
    CHECK(v->dimension == ViewDimension::Texture2D && v->first_mip == 2 && v->mip_levels == 3);
    CHECK(v->component_mapping == kDefaultShader4ComponentMapping);

    CHECK(reg.create_shader_resource_view(43, nullptr, cpu_at(heap, 3)) == Status::Ok);  // default view
    CHECK(heap->view(3)->dimension == ViewDimension::Unknown && heap->view(3)->mip_levels == 0xffff);
    CHECK(reg.create_shader_resource_view(0, nullptr, cpu_at(heap, 4)) == Status::InvalidArgument);
    CHECK(reg.create_shader_resource_view(0, &srv, cpu_at(heap, 4)) == Status::Ok);  // null SRV
    CHECK(heap->view(4)->kind == DescriptorKind::Srv && heap->view(4)->resource_id == 0);
    ViewDesc bad_dim = srv;
    bad_dim.dimension = 99;
    CHECK(reg.create_shader_resource_view(42, &bad_dim, cpu_at(heap, 5)) == Status::InvalidArgument);

    ViewDesc uav;
    uav.dimension = ViewDimension::Buffer;
    uav.first_element = 16;
    uav.num_elements = 64;
    uav.structure_byte_stride = 16;
    uav.counter_offset_in_bytes = 4096;
    CHECK(reg.create_unordered_access_view(50, 51, &uav, cpu_at(heap, 5)) == Status::Ok);
    v = heap->view(5);
    CHECK(v->kind == DescriptorKind::Uav && v->resource_id == 50 && v->counter_resource_id == 51);
    CHECK(v->first_element_or_address == 16 && v->num_elements_or_size == 64 && v->structure_byte_stride == 16);
    CHECK(v->counter_offset_in_bytes == 4096 && v->component_mapping == 0);

    SamplerDesc s;
    s.filter = 0x55;  // anisotropic
    s.max_anisotropy = 8;
    s.address_u = 3;
    CHECK(reg.create_sampler(s, cpu_at(samplers, 1)) == Status::Ok);
    CHECK(samplers->sampler(1)->kind == DescriptorKind::Sampler);
    CHECK(samplers->sampler(1)->desc.max_anisotropy == 8 && samplers->sampler(1)->desc.address_u == 3);
    CHECK(samplers->view(0) == nullptr);
    CHECK(heap->sampler(0) == nullptr);

    ViewDesc rtv_desc;
    rtv_desc.dimension = ViewDimension::Texture2DArray;
    rtv_desc.first_mip = 1;
    rtv_desc.first_array_slice = 2;
    rtv_desc.array_size = 3;
    CHECK(reg.create_render_target_view(60, &rtv_desc, cpu_at(rtvs, 0)) == Status::Ok);
    CHECK(rtvs->view(0)->kind == DescriptorKind::Rtv && rtvs->view(0)->first_array_slice == 2);
    CHECK(rtvs->view(0)->mip_levels == 1);
    ViewDesc dsv_desc;
    dsv_desc.dimension = DsvDimension::Texture2D;
    dsv_desc.flags = DsvFlags::ReadOnlyDepth;
    CHECK(reg.create_depth_stencil_view(61, &dsv_desc, cpu_at(dsvs, 3)) == Status::Ok);
    CHECK(dsvs->view(3)->kind == DescriptorKind::Dsv && dsvs->view(3)->flags == DsvFlags::ReadOnlyDepth);

    // Heap type checks.
    CHECK(reg.create_shader_resource_view(42, &srv, cpu_at(samplers, 0)) == Status::WrongHeapType);
    CHECK(reg.create_sampler(s, cpu_at(heap, 0)) == Status::WrongHeapType);
    CHECK(reg.create_render_target_view(60, &rtv_desc, cpu_at(heap, 0)) == Status::WrongHeapType);
    CHECK(reg.create_depth_stencil_view(61, &dsv_desc, cpu_at(rtvs, 0)) == Status::WrongHeapType);
    CHECK(reg.create_shader_resource_view(42, &srv, cpu_at(heap, 16)) == Status::HandleOutOfRange);
    CHECK(reg.create_shader_resource_view(42, &srv, CpuDescriptorHandle{0}) == Status::InvalidHandle);

    // Generation bumps only when content changes.
    const uint64_t g0 = heap->generation();
    const uint64_t slot_g = heap->slot_generation(2);
    CHECK(slot_g > 0 && slot_g <= g0);
    CHECK(reg.create_shader_resource_view(42, &srv, cpu_at(heap, 2)) == Status::Ok);
    CHECK(heap->generation() == g0);
    CHECK(heap->slot_generation(2) == slot_g);
    srv.mip_levels = 4;
    CHECK(reg.create_shader_resource_view(42, &srv, cpu_at(heap, 2)) == Status::Ok);
    CHECK(heap->generation() == g0 + 1);
    CHECK(heap->slot_generation(2) == g0 + 1);
}

// ---------------------------------------------------------------------------
// 3. CopyDescriptorsSimple / CopyDescriptors
// ---------------------------------------------------------------------------

void fill_srvs(DescriptorHeapRegistry& reg, DescriptorHeap* heap, uint32_t first, uint32_t count, uint32_t id_base) {
    ViewDesc srv = texture2d_srv();
    for (uint32_t i = 0; i < count; ++i) {
        CHECK(reg.create_shader_resource_view(id_base + i, &srv, cpu_at(heap, first + i)) == Status::Ok);
    }
}

void test_copy_descriptors_simple() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* src = make_heap(reg, DescriptorHeapType::CbvSrvUav, 32, false);
    DescriptorHeap* dst = make_heap(reg, DescriptorHeapType::CbvSrvUav, 32, true);
    DescriptorHeap* samp = make_heap(reg, DescriptorHeapType::Sampler, 8, false);
    fill_srvs(reg, src, 0, 10, 100);

    CHECK(reg.copy_descriptors_simple(4, cpu_at(dst, 20), cpu_at(src, 3), DescriptorHeapType::CbvSrvUav) == Status::Ok);
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(dst->view(20 + i)->resource_id == 103 + i);
    }
    CHECK(dst->view(24)->kind == DescriptorKind::Empty);
    CHECK(reg.copy_descriptors_simple(0, CpuDescriptorHandle{0}, CpuDescriptorHandle{0}, DescriptorHeapType::CbvSrvUav) ==
          Status::Ok);
    CHECK(reg.copy_descriptors_simple(4, cpu_at(dst, 30), cpu_at(src, 0), DescriptorHeapType::CbvSrvUav) ==
          Status::HandleOutOfRange);
    CHECK(reg.copy_descriptors_simple(1, cpu_at(dst, 0), cpu_at(src, 0), DescriptorHeapType::Sampler) ==
          Status::WrongHeapType);
    CHECK(reg.copy_descriptors_simple(1, cpu_at(samp, 0), cpu_at(src, 0), DescriptorHeapType::Sampler) ==
          Status::WrongHeapType);
    CHECK(reg.copy_descriptors_simple(1, cpu_at(dst, 0), CpuDescriptorHandle{dst->gpu_start().ptr},
                                      DescriptorHeapType::CbvSrvUav) == Status::InvalidHandle);

    // Overlapping copies inside one heap behave like memmove (forward and backward).
    CHECK(reg.copy_descriptors_simple(6, cpu_at(src, 2), cpu_at(src, 0), DescriptorHeapType::CbvSrvUav) == Status::Ok);
    const uint32_t expected_fwd[10] = {100, 101, 100, 101, 102, 103, 104, 105, 108, 109};
    for (uint32_t i = 0; i < 10; ++i) {
        CHECK(src->view(i)->resource_id == expected_fwd[i]);
    }
    fill_srvs(reg, src, 0, 10, 100);
    CHECK(reg.copy_descriptors_simple(6, cpu_at(src, 0), cpu_at(src, 2), DescriptorHeapType::CbvSrvUav) == Status::Ok);
    const uint32_t expected_back[10] = {102, 103, 104, 105, 106, 107, 106, 107, 108, 109};
    for (uint32_t i = 0; i < 10; ++i) {
        CHECK(src->view(i)->resource_id == expected_back[i]);
    }
    // Self copy is a no-op.
    const uint64_t gen = src->generation();
    CHECK(reg.copy_descriptors_simple(5, cpu_at(src, 0), cpu_at(src, 0), DescriptorHeapType::CbvSrvUav) == Status::Ok);
    CHECK(src->generation() == gen);

    // Sampler copies.
    SamplerDesc s;
    s.max_anisotropy = 4;
    CHECK(reg.create_sampler(s, cpu_at(samp, 0)) == Status::Ok);
    CHECK(reg.copy_descriptors_simple(1, cpu_at(samp, 5), cpu_at(samp, 0), DescriptorHeapType::Sampler) == Status::Ok);
    CHECK(samp->sampler(5)->kind == DescriptorKind::Sampler && samp->sampler(5)->desc.max_anisotropy == 4);
}

void test_copy_descriptors_multi_range() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* a = make_heap(reg, DescriptorHeapType::CbvSrvUav, 64, false);
    DescriptorHeap* b = make_heap(reg, DescriptorHeapType::CbvSrvUav, 64, false);
    DescriptorHeap* gpu = make_heap(reg, DescriptorHeapType::CbvSrvUav, 64, true);
    fill_srvs(reg, a, 0, 8, 200);
    fill_srvs(reg, b, 0, 8, 300);

    // dst: [gpu 0..3), [gpu 10..12)   src: [a 1], [b 4..8)
    const CpuDescriptorHandle dsts[2] = {cpu_at(gpu, 0), cpu_at(gpu, 10)};
    const uint32_t dst_sizes[2] = {3, 2};
    const CpuDescriptorHandle srcs[2] = {cpu_at(a, 1), cpu_at(b, 4)};
    const uint32_t src_sizes[2] = {1, 4};
    CHECK(reg.copy_descriptors(2, dsts, dst_sizes, 2, srcs, src_sizes, DescriptorHeapType::CbvSrvUav) == Status::Ok);
    CHECK(gpu->view(0)->resource_id == 201);
    CHECK(gpu->view(1)->resource_id == 304);
    CHECK(gpu->view(2)->resource_id == 305);
    CHECK(gpu->view(10)->resource_id == 306);
    CHECK(gpu->view(11)->resource_id == 307);
    CHECK(gpu->view(3)->kind == DescriptorKind::Empty);

    // Null size arrays mean one descriptor per range.
    const CpuDescriptorHandle singles_dst[3] = {cpu_at(gpu, 40), cpu_at(gpu, 45), cpu_at(gpu, 50)};
    const CpuDescriptorHandle singles_src[3] = {cpu_at(a, 7), cpu_at(b, 0), cpu_at(a, 0)};
    CHECK(reg.copy_descriptors(3, singles_dst, nullptr, 3, singles_src, nullptr, DescriptorHeapType::CbvSrvUav) ==
          Status::Ok);
    CHECK(gpu->view(40)->resource_id == 207 && gpu->view(45)->resource_id == 300 && gpu->view(50)->resource_id == 200);

    // Zero-sized ranges are skipped (their handles are not validated).
    const CpuDescriptorHandle with_empty_dst[2] = {CpuDescriptorHandle{0}, cpu_at(gpu, 60)};
    const uint32_t with_empty_sizes[2] = {0, 1};
    CHECK(reg.copy_descriptors(2, with_empty_dst, with_empty_sizes, 1, srcs, nullptr, DescriptorHeapType::CbvSrvUav) ==
          Status::Ok);
    CHECK(gpu->view(60)->resource_id == 201);

    // Count mismatch / validation is atomic: nothing written on failure.
    const uint64_t gen = gpu->generation();
    const uint32_t too_many[2] = {3, 3};
    CHECK(reg.copy_descriptors(2, dsts, too_many, 2, srcs, src_sizes, DescriptorHeapType::CbvSrvUav) ==
          Status::CountMismatch);
    const CpuDescriptorHandle bad_srcs[2] = {cpu_at(a, 1), cpu_at(b, 62)};
    CHECK(reg.copy_descriptors(2, dsts, dst_sizes, 2, bad_srcs, src_sizes, DescriptorHeapType::CbvSrvUav) ==
          Status::HandleOutOfRange);
    CHECK(gpu->generation() == gen);
    CHECK(reg.copy_descriptors(1, nullptr, nullptr, 1, srcs, nullptr, DescriptorHeapType::CbvSrvUav) ==
          Status::InvalidArgument);

    // Overlapping multi-range copy: rotate a[0..4) left by one using the same heap.
    // Result must equal "read all sources first, then write".
    fill_srvs(reg, a, 0, 8, 200);
    const CpuDescriptorHandle rot_dst[1] = {cpu_at(a, 0)};
    const uint32_t rot_dst_size[1] = {4};
    const CpuDescriptorHandle rot_src[2] = {cpu_at(a, 1), cpu_at(a, 0)};
    const uint32_t rot_src_size[2] = {3, 1};
    CHECK(reg.copy_descriptors(1, rot_dst, rot_dst_size, 2, rot_src, rot_src_size, DescriptorHeapType::CbvSrvUav) ==
          Status::Ok);
    CHECK(a->view(0)->resource_id == 201);
    CHECK(a->view(1)->resource_id == 202);
    CHECK(a->view(2)->resource_id == 203);
    CHECK(a->view(3)->resource_id == 200);
    CHECK(a->view(4)->resource_id == 204);

    // Sampler multi-range.
    DescriptorHeap* s0 = make_heap(reg, DescriptorHeapType::Sampler, 8, false);
    DescriptorHeap* s1 = make_heap(reg, DescriptorHeapType::Sampler, 8, true);
    for (uint32_t i = 0; i < 4; ++i) {
        SamplerDesc sd;
        sd.max_anisotropy = i + 1;
        CHECK(reg.create_sampler(sd, cpu_at(s0, i)) == Status::Ok);
    }
    const CpuDescriptorHandle sdst[1] = {cpu_at(s1, 2)};
    const uint32_t sdst_size[1] = {4};
    const CpuDescriptorHandle ssrc[2] = {cpu_at(s0, 2), cpu_at(s0, 0)};
    const uint32_t ssrc_size[2] = {2, 2};
    CHECK(reg.copy_descriptors(1, sdst, sdst_size, 2, ssrc, ssrc_size, DescriptorHeapType::Sampler) == Status::Ok);
    CHECK(s1->sampler(2)->desc.max_anisotropy == 3 && s1->sampler(3)->desc.max_anisotropy == 4);
    CHECK(s1->sampler(4)->desc.max_anisotropy == 1 && s1->sampler(5)->desc.max_anisotropy == 2);
}

// ---------------------------------------------------------------------------
// 4. Generation / dirty tracking on a large heap
// ---------------------------------------------------------------------------

void test_generation_tracking() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 1u << 20, true);
    CHECK(!heap->range_changed_since(0, heap->capacity(), 0));

    const uint64_t g0 = heap->generation();
    fill_srvs(reg, heap, 130, 1, 1);
    CHECK(heap->generation() == g0 + 1);
    CHECK(heap->range_changed_since(130, 1, g0));
    CHECK(heap->range_changed_since(129, 2, g0));
    CHECK(heap->range_changed_since(128, 64, g0));  // whole block
    CHECK(heap->range_changed_since(0, 1u << 20, g0));
    CHECK(!heap->range_changed_since(131, 10, g0));  // same block, after the slot
    CHECK(!heap->range_changed_since(64, 66, g0));   // same block, before the slot
    CHECK(!heap->range_changed_since(0, 128, g0));
    CHECK(!heap->range_changed_since(130, 1, heap->generation()));
    CHECK(!heap->range_changed_since(130, 0, g0));
    CHECK(!heap->range_changed_since(1u << 20, 5, g0));
    CHECK(heap->range_changed_since(100, 0xffffffffu, g0));  // count clamped to capacity

    // Writes elsewhere do not invalidate an unrelated range.
    const uint64_t g1 = heap->generation();
    fill_srvs(reg, heap, 900000, 3, 7);
    CHECK(heap->range_changed_since(899990, 20, g1));
    CHECK(!heap->range_changed_since(0, 899999, g1));
    CHECK(!heap->range_changed_since(900003, 1000, g1));

    // Rewriting identical content is not a change.
    const uint64_t g2 = heap->generation();
    fill_srvs(reg, heap, 900000, 3, 7);
    CHECK(heap->generation() == g2);
    CHECK(!heap->range_changed_since(900000, 3, g2));

    // Copies stamp destinations only.
    CHECK(reg.copy_descriptors_simple(2, cpu_at(heap, 5000), cpu_at(heap, 900000), DescriptorHeapType::CbvSrvUav) ==
          Status::Ok);
    CHECK(heap->range_changed_since(5000, 2, g2));
    CHECK(!heap->range_changed_since(900000, 3, g2));
}

// ---------------------------------------------------------------------------
// 5. Root signatures
// ---------------------------------------------------------------------------

DescriptorRangeDesc range(DescriptorRangeType type, uint32_t count, uint32_t base, uint32_t space = 0,
                          uint32_t offset = kDescriptorRangeOffsetAppend) {
    DescriptorRangeDesc r;
    r.range_type = type;
    r.num_descriptors = count;
    r.base_shader_register = base;
    r.register_space = space;
    r.offset_in_descriptors_from_table_start = offset;
    return r;
}

RootParameterDesc table(std::vector<DescriptorRangeDesc> ranges, ShaderVisibility vis = ShaderVisibility::All) {
    RootParameterDesc p;
    p.parameter_type = RootParameterType::DescriptorTable;
    p.shader_visibility = vis;
    p.ranges = std::move(ranges);
    return p;
}

RootParameterDesc root_view(RootParameterType type, uint32_t reg, uint32_t space = 0,
                            ShaderVisibility vis = ShaderVisibility::All) {
    RootParameterDesc p;
    p.parameter_type = type;
    p.shader_register = reg;
    p.register_space = space;
    p.shader_visibility = vis;
    return p;
}

RootParameterDesc constants(uint32_t reg, uint32_t count, uint32_t space = 0) {
    RootParameterDesc p;
    p.parameter_type = RootParameterType::Constants32Bit;
    p.shader_register = reg;
    p.register_space = space;
    p.num_32bit_values = count;
    return p;
}

StaticSamplerDesc static_sampler(uint32_t reg, uint32_t border = 0) {
    StaticSamplerDesc s;
    s.shader_register = reg;
    s.border_color = border;
    s.address_u = 4;
    return s;
}

// Representative signature used by several tests:
//  p0: table { CBV b0-b1, SRV t0-t3 (append), UAV u0 @ offset 10 }
//  p1: sampler table { s0-s1 } (pixel)
//  p2: root constants b2 (4 dwords)
//  p3: root CBV b3
//  p4: root SRV t10 (pixel)
//  p5: table { SRV t0.. unbounded, space 1 }
//  static samplers s4 (opaque white border), s5
RootSignatureDesc representative_desc() {
    RootSignatureDesc d;
    d.version = RootSignatureVersion::V1_1;
    d.flags = RootSignatureFlags::AllowInputAssemblerInputLayout;
    d.parameters.push_back(table({range(DescriptorRangeType::Cbv, 2, 0), range(DescriptorRangeType::Srv, 4, 0),
                                  range(DescriptorRangeType::Uav, 1, 0, 0, 10)}));
    d.parameters.push_back(table({range(DescriptorRangeType::Sampler, 2, 0)}, ShaderVisibility::Pixel));
    d.parameters.push_back(constants(2, 4));
    d.parameters.push_back(root_view(RootParameterType::Cbv, 3));
    d.parameters.push_back(root_view(RootParameterType::Srv, 10, 0, ShaderVisibility::Pixel));
    d.parameters.push_back(table({range(DescriptorRangeType::Srv, kUnboundedDescriptorCount, 0, 1)}));
    d.static_samplers.push_back(static_sampler(4, 2));
    d.static_samplers.push_back(static_sampler(5));
    return d;
}

void test_root_signature_model() {
    RootSignature rs;
    const RootSignatureError err = RootSignature::build(representative_desc(), &rs);
    CHECK(err.ok());
    CHECK(rs.parameter_count() == 6);
    CHECK(rs.cost_dwords() == 1 + 1 + 4 + 2 + 2 + 1);
    CHECK(rs.total_constant_dwords() == 4);

    const RootSignature::ParameterInfo& p0 = rs.parameter(0);
    CHECK(p0.type == RootParameterType::DescriptorTable);
    CHECK(p0.table_heap_type == DescriptorHeapType::CbvSrvUav);
    CHECK(p0.table_extent == 11);  // UAV at offset 10
    CHECK(!p0.table_unbounded);
    CHECK(p0.first_slot == 0 && p0.slot_count == 3);
    CHECK(rs.parameter(1).table_heap_type == DescriptorHeapType::Sampler);
    CHECK(rs.parameter(1).table_extent == 2);
    CHECK(rs.parameter(2).constants_offset == 0 && rs.parameter(2).num_32bit_values == 4);
    CHECK(rs.parameter(5).table_unbounded && rs.parameter(5).table_extent == 0);

    // Slot order: parameters in declaration order (one slot per range), then static samplers.
    const BindingLayout& L = rs.layout();
    CHECK(L.slots.size() == 3 + 1 + 1 + 1 + 1 + 1 + 2);
    CHECK(L.slots[0].kind == DescriptorRangeType::Cbv && L.slots[0].table_offset == 0);
    CHECK(L.slots[0].descriptor_count == 2 && L.slots[0].source == SlotSource::DescriptorTable);
    CHECK(L.slots[1].kind == DescriptorRangeType::Srv && L.slots[1].table_offset == 2);  // OFFSET_APPEND
    CHECK(L.slots[2].kind == DescriptorRangeType::Uav && L.slots[2].table_offset == 10);
    CHECK(L.slots[2].range_index == 2 && L.slots[2].root_parameter_index == 0);
    CHECK(L.slots[3].kind == DescriptorRangeType::Sampler && L.slots[3].visibility == ShaderVisibility::Pixel);
    CHECK(L.slots[4].source == SlotSource::RootConstants && L.slots[4].kind == DescriptorRangeType::Cbv);
    CHECK(L.slots[4].num_32bit_values == 4 && L.slots[4].base_register == 2);
    CHECK(L.slots[5].source == SlotSource::RootDescriptor && L.slots[5].kind == DescriptorRangeType::Cbv);
    CHECK(L.slots[6].source == SlotSource::RootDescriptor && L.slots[6].kind == DescriptorRangeType::Srv);
    CHECK(L.slots[7].unbounded && L.slots[7].register_space == 1 && L.slots[7].table_offset == 0);
    CHECK(L.slots[8].source == SlotSource::StaticSampler && L.slots[8].root_parameter_index == 0);
    CHECK(L.slots[9].source == SlotSource::StaticSampler && L.slots[9].base_register == 5);
    CHECK(L.has_unbounded_ranges);

    // Register lookups (the S6 join key) -> slot + descriptor offset in the table.
    uint32_t offset = 77;
    const uint32_t vf = WebGpuStage::Vertex | WebGpuStage::Fragment;
    CHECK(L.find(DescriptorRangeType::Srv, 0, 3, vf, &offset) == &L.slots[1] && offset == 5);
    CHECK(L.find(DescriptorRangeType::Cbv, 0, 1, vf, &offset) == &L.slots[0] && offset == 1);
    CHECK(L.find(DescriptorRangeType::Cbv, 0, 2, vf, &offset) == &L.slots[4] && offset == 0);
    CHECK(L.find(DescriptorRangeType::Sampler, 0, 1, WebGpuStage::Fragment, &offset) == &L.slots[3] && offset == 1);
    CHECK(L.find(DescriptorRangeType::Sampler, 0, 1, WebGpuStage::Vertex, &offset) == nullptr);  // pixel only
    CHECK(L.find(DescriptorRangeType::Sampler, 0, 1, WebGpuStage::Compute, &offset) == &L.slots[3]);
    CHECK(L.find(DescriptorRangeType::Sampler, 0, 5, vf, &offset) == &L.slots[9] && offset == 0);
    CHECK(L.find(DescriptorRangeType::Srv, 1, 5000, vf, &offset) == &L.slots[7] && offset == 5000);  // unbounded
    CHECK(L.find(DescriptorRangeType::Srv, 0, 4, vf, &offset) == nullptr);
    CHECK(L.find(DescriptorRangeType::Uav, 0, 0, vf, &offset) == &L.slots[2] && offset == 10);
    CHECK(L.find(DescriptorRangeType::Srv, 0, 10, WebGpuStage::Fragment, nullptr) == &L.slots[6]);
    CHECK(L.find(DescriptorRangeType::Srv, 0, 10, WebGpuStage::Vertex, nullptr) == nullptr);

    CHECK(webgpu_stage_mask(ShaderVisibility::All) == 7u);
    CHECK(webgpu_stage_mask(ShaderVisibility::Pixel) == WebGpuStage::Fragment);
    CHECK(webgpu_stage_mask(ShaderVisibility::Mesh) == WebGpuStage::Compute);
    CHECK(webgpu_stage_mask(ShaderVisibility::Geometry) == 0u);

    // Version 1.0 normalizes flags to "everything volatile".
    RootSignatureDesc v10 = representative_desc();
    v10.version = RootSignatureVersion::V1_0;
    RootSignature rs10;
    CHECK(RootSignature::build(v10, &rs10).ok());
    CHECK(rs10.desc().parameters[0].ranges[0].flags ==
          (DescriptorRangeFlags::DescriptorsVolatile | DescriptorRangeFlags::DataVolatile));
    CHECK(rs10.desc().parameters[1].ranges[0].flags == DescriptorRangeFlags::DescriptorsVolatile);
    CHECK(rs10.desc().parameters[3].flags == RootDescriptorFlags::DataVolatile);
    CHECK(rs10.hash() != rs.hash());

    // Empty signature.
    RootSignature empty;
    CHECK(RootSignature::build(RootSignatureDesc{}, &empty).ok());
    CHECK(empty.parameter_count() == 0 && empty.layout().slots.empty());
}

void expect_build_error(const RootSignatureDesc& d, Status status, uint32_t param, uint32_t range_index, int line) {
    RootSignature rs;
    const RootSignatureError err = RootSignature::build(d, &rs);
    check_impl(err.status == status, status_name(status), line);
    check_impl(err.parameter_index == param, "parameter_index", line);
    check_impl(err.range_index == range_index, "range_index", line);
}

void test_root_signature_validation() {
    RootSignatureDesc d;
    d.parameters.push_back(table({}));
    expect_build_error(d, Status::EmptyDescriptorTable, 0, kNoIndex, __LINE__);

    d = RootSignatureDesc{};
    d.parameters.push_back(table({range(DescriptorRangeType::Srv, 0, 0)}));
    expect_build_error(d, Status::ZeroDescriptorRange, 0, 0, __LINE__);

    d = RootSignatureDesc{};
    d.parameters.push_back(constants(0, 1));
    d.parameters.push_back(table({range(DescriptorRangeType::Srv, 1, 0), range(DescriptorRangeType::Sampler, 1, 0)}));
    expect_build_error(d, Status::MixedSamplerTable, 1, 1, __LINE__);

    d = RootSignatureDesc{};
    d.parameters.push_back(table({range(DescriptorRangeType::Srv, kUnboundedDescriptorCount, 0),
                                  range(DescriptorRangeType::Srv, 1, 0, 1)}));
    expect_build_error(d, Status::AppendAfterUnboundedRange, 0, 1, __LINE__);
    // ...but an explicit offset after an unbounded range is fine.
    d.parameters[0].ranges[1].offset_in_descriptors_from_table_start = 0;
    expect_build_error(d, Status::Ok, kNoIndex, kNoIndex, __LINE__);

    d = RootSignatureDesc{};
    d.parameters.push_back(table({range(static_cast<DescriptorRangeType>(9), 1, 0)}));
    expect_build_error(d, Status::InvalidRangeType, 0, 0, __LINE__);

    d = RootSignatureDesc{};
    d.parameters.push_back(table({range(DescriptorRangeType::Srv, 4, 0, 0, 0xfffffffe)}));
    expect_build_error(d, Status::DescriptorOffsetOverflow, 0, 0, __LINE__);

    // Register overlap with overlapping visibility.
    d = RootSignatureDesc{};
    d.parameters.push_back(root_view(RootParameterType::Cbv, 0));
    d.parameters.push_back(table({range(DescriptorRangeType::Cbv, 4, 0)}, ShaderVisibility::Pixel));
    expect_build_error(d, Status::RegisterOverlap, 1, 0, __LINE__);
    // Disjoint visibility is allowed.
    d.parameters[0].shader_visibility = ShaderVisibility::Vertex;
    expect_build_error(d, Status::Ok, kNoIndex, kNoIndex, __LINE__);
    // Different register class or space is allowed.
    d = RootSignatureDesc{};
    d.parameters.push_back(root_view(RootParameterType::Cbv, 0));
    d.parameters.push_back(root_view(RootParameterType::Srv, 0));
    d.parameters.push_back(root_view(RootParameterType::Cbv, 0, 1));
    expect_build_error(d, Status::Ok, kNoIndex, kNoIndex, __LINE__);
    // Unbounded range overlaps everything above its base.
    d = RootSignatureDesc{};
    d.parameters.push_back(table({range(DescriptorRangeType::Srv, kUnboundedDescriptorCount, 10)}));
    d.parameters.push_back(root_view(RootParameterType::Srv, 1000));
    expect_build_error(d, Status::RegisterOverlap, 1, kNoIndex, __LINE__);
    // Static sampler overlapping a sampler table.
    d = RootSignatureDesc{};
    d.parameters.push_back(table({range(DescriptorRangeType::Sampler, 2, 0)}));
    d.static_samplers.push_back(static_sampler(1));
    expect_build_error(d, Status::RegisterOverlap, kNoIndex, 0, __LINE__);

    // Size limit: 64 DWORDs.
    d = RootSignatureDesc{};
    d.parameters.push_back(constants(0, 63));
    d.parameters.push_back(root_view(RootParameterType::Cbv, 1));
    expect_build_error(d, Status::RootSignatureTooLarge, 1, kNoIndex, __LINE__);
    d.parameters.pop_back();
    d.parameters.push_back(table({range(DescriptorRangeType::Srv, 1, 0)}));
    expect_build_error(d, Status::Ok, kNoIndex, kNoIndex, __LINE__);

    d = RootSignatureDesc{};
    d.parameters.push_back(constants(0, 0));
    expect_build_error(d, Status::InvalidArgument, 0, kNoIndex, __LINE__);

    d = RootSignatureDesc{};
    d.parameters.push_back(root_view(static_cast<RootParameterType>(7), 0));
    expect_build_error(d, Status::InvalidParameterType, 0, kNoIndex, __LINE__);

    d = RootSignatureDesc{};
    d.parameters.push_back(root_view(RootParameterType::Cbv, 0, 0, static_cast<ShaderVisibility>(8)));
    expect_build_error(d, Status::InvalidShaderVisibility, 0, kNoIndex, __LINE__);

    d = RootSignatureDesc{};
    d.version = static_cast<RootSignatureVersion>(4);
    expect_build_error(d, Status::UnsupportedVersion, kNoIndex, kNoIndex, __LINE__);

    d = RootSignatureDesc{};
    d.static_samplers.resize(kMaxStaticSamplers + 1);
    expect_build_error(d, Status::TooManyStaticSamplers, kNoIndex, kNoIndex, __LINE__);

    // Large tables are fine (WebGPU limits are checked per bind group layout, by S6).
    d = RootSignatureDesc{};
    d.parameters.push_back(table({range(DescriptorRangeType::Srv, 2000, 0)}));
    RootSignature big;
    CHECK(RootSignature::build(d, &big).ok());
    CHECK(big.parameter(0).table_extent == 2000);
    CHECK(RootSignature::build(d, nullptr).status == Status::InvalidArgument);
}

void test_root_signature_hash() {
    RootSignature a;
    RootSignature b;
    CHECK(RootSignature::build(representative_desc(), &a).ok());
    CHECK(RootSignature::build(representative_desc(), &b).ok());
    CHECK(a.hash() == b.hash());
    CHECK(a.hash() != 0);

    // OFFSET_APPEND and the equivalent explicit offsets hash identically.
    RootSignatureDesc explicit_offsets = representative_desc();
    explicit_offsets.parameters[0].ranges[0].offset_in_descriptors_from_table_start = 0;
    explicit_offsets.parameters[0].ranges[1].offset_in_descriptors_from_table_start = 2;
    RootSignature c;
    CHECK(RootSignature::build(explicit_offsets, &c).ok());
    CHECK(c.hash() == a.hash());

    // Any semantic change changes the hash.
    RootSignatureDesc changed = representative_desc();
    changed.parameters[3].shader_register = 4;
    RootSignature e;
    CHECK(RootSignature::build(changed, &e).ok());
    CHECK(e.hash() != a.hash());
    changed = representative_desc();
    changed.static_samplers[1].max_lod = 4.0f;
    CHECK(RootSignature::build(changed, &e).ok());
    CHECK(e.hash() != a.hash());
    changed = representative_desc();
    changed.parameters[1].shader_visibility = ShaderVisibility::All;
    CHECK(RootSignature::build(changed, &e).ok());
    CHECK(e.hash() != a.hash());

    // Golden value: the hash is a persistent cache key, it must not drift
    // across compilers, platforms or refactors.
    RootSignatureDesc tiny;
    tiny.parameters.push_back(root_view(RootParameterType::Cbv, 0));
    RootSignature t;
    CHECK(RootSignature::build(tiny, &t).ok());
    if (t.hash() != 0xb6ef5345f09444c3ull) {
        std::fprintf(stderr, "golden root signature hash = 0x%016llx\n", static_cast<unsigned long long>(t.hash()));
    }
    CHECK(t.hash() == 0xb6ef5345f09444c3ull);
}

// ---------------------------------------------------------------------------
// 6. Root signature blob parsing
// ---------------------------------------------------------------------------

void put_u32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) {
        v.push_back(static_cast<uint8_t>(x >> (8 * i)));
    }
}

std::vector<uint8_t> wrap_dxbc(const std::vector<uint8_t>& rts0, bool include_other_part) {
    std::vector<uint8_t> out;
    put_u32(out, 0x43425844u);  // "DXBC"
    for (int i = 0; i < 4; ++i) {
        put_u32(out, 0);  // digest
    }
    put_u32(out, 1);  // u16 major = 1, u16 minor = 0
    const uint32_t part_count = include_other_part ? 2 : 1;
    const size_t total_at = out.size();
    put_u32(out, 0);
    put_u32(out, part_count);
    const size_t offsets_at = out.size();
    for (uint32_t i = 0; i < part_count; ++i) {
        put_u32(out, 0);
    }
    auto patch = [&](size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            out[at + i] = static_cast<uint8_t>(v >> (8 * i));
        }
    };
    uint32_t idx = 0;
    if (include_other_part) {
        patch(offsets_at, static_cast<uint32_t>(out.size()));
        put_u32(out, 0x4e475349u);  // "ISGN"
        put_u32(out, 8);
        put_u32(out, 0);
        put_u32(out, 8);
        idx = 1;
    }
    patch(offsets_at + 4 * idx, static_cast<uint32_t>(out.size()));
    put_u32(out, 0x30535452u);  // "RTS0"
    put_u32(out, static_cast<uint32_t>(rts0.size()));
    out.insert(out.end(), rts0.begin(), rts0.end());
    patch(total_at, static_cast<uint32_t>(out.size()));
    return out;
}

void test_root_signature_blob_roundtrip() {
    for (RootSignatureVersion version :
         {RootSignatureVersion::V1_0, RootSignatureVersion::V1_1, RootSignatureVersion::V1_2}) {
        RootSignatureDesc d = representative_desc();
        d.version = version;
        d.parameters[3].flags = RootDescriptorFlags::DataStatic;
        d.static_samplers[0].flags = 0x1;  // only kept for 1.2
        RootSignature direct;
        CHECK(RootSignature::build(d, &direct).ok());

        const std::vector<uint8_t> blob = serialize_root_signature(d);
        const uint32_t sampler_bytes = version == RootSignatureVersion::V1_2 ? 56u : 52u;
        CHECK(blob.size() > 24 + 6 * 12 + 2 * sampler_bytes);
        RootSignatureDesc parsed;
        CHECK(parse_root_signature_blob(blob.data(), blob.size(), &parsed).ok());
        CHECK(parsed.version == version);
        CHECK(parsed.parameters.size() == 6 && parsed.static_samplers.size() == 2);
        CHECK(parsed.flags == RootSignatureFlags::AllowInputAssemblerInputLayout);
        CHECK(parsed.parameters[0].ranges.size() == 3);
        CHECK(parsed.parameters[0].ranges[2].offset_in_descriptors_from_table_start == 10);
        CHECK(parsed.parameters[5].ranges[0].num_descriptors == kUnboundedDescriptorCount);
        CHECK(parsed.parameters[2].num_32bit_values == 4);
        CHECK(parsed.static_samplers[0].border_color == 2 && parsed.static_samplers[0].address_u == 4);
        CHECK(parsed.static_samplers[0].flags == (version == RootSignatureVersion::V1_2 ? 0x1u : 0u));
        CHECK(parsed.parameters[3].flags == (version == RootSignatureVersion::V1_0 ? 0u : RootDescriptorFlags::DataStatic));
        RootSignature from_blob;
        CHECK(RootSignature::build(parsed, &from_blob).ok());
        CHECK(from_blob.hash() == direct.hash());

        // Same blob inside a DXBC container (as passed to CreateRootSignature).
        RootSignatureDesc from_container;
        const std::vector<uint8_t> dxbc = wrap_dxbc(blob, true);
        CHECK(parse_root_signature_blob(dxbc.data(), dxbc.size(), &from_container).ok());
        RootSignature rs_container;
        CHECK(RootSignature::build(from_container, &rs_container).ok());
        CHECK(rs_container.hash() == direct.hash());
    }
}

void test_root_signature_blob_known_layout() {
    // Hand-assembled version 1.0 blob following the public RTS0 layout:
    // header {version, numParams, paramsOffset, numStaticSamplers, samplersOffset, flags}
    // param {type, visibility, payloadOffset}; table {numRanges, rangesOffset};
    // 1.0 range {type, count, baseRegister, space, offset}; root constants {reg, space, count}.
    std::vector<uint8_t> b;
    put_u32(b, 1);     // version 1.0
    put_u32(b, 2);     // two parameters
    put_u32(b, 24);    // parameters at 24
    put_u32(b, 0);     // no static samplers
    put_u32(b, 88);    // (empty) static sampler array at the end
    put_u32(b, 0x1);   // ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    put_u32(b, 0);     // p0: descriptor table
    put_u32(b, 5);     // pixel
    put_u32(b, 48);    // payload
    put_u32(b, 1);     // p1: 32-bit constants
    put_u32(b, 0);     // all
    put_u32(b, 76);    // payload
    put_u32(b, 1);     // @48: one range
    put_u32(b, 56);    // ranges at 56
    put_u32(b, 0);     // @56: SRV
    put_u32(b, 3);     // three descriptors
    put_u32(b, 4);     // t4
    put_u32(b, 2);     // space2
    put_u32(b, 0xffffffffu);  // append
    put_u32(b, 7);     // @76: b7
    put_u32(b, 0);
    put_u32(b, 2);     // two dwords
    CHECK(b.size() == 88);
    RootSignatureDesc d;
    CHECK(parse_root_signature_blob(b.data(), b.size(), &d).ok());
    CHECK(d.version == RootSignatureVersion::V1_0 && d.flags == 1);
    CHECK(d.parameters.size() == 2);
    CHECK(d.parameters[0].shader_visibility == ShaderVisibility::Pixel);
    CHECK(d.parameters[0].ranges.size() == 1);
    CHECK(d.parameters[0].ranges[0].range_type == DescriptorRangeType::Srv);
    CHECK(d.parameters[0].ranges[0].num_descriptors == 3);
    CHECK(d.parameters[0].ranges[0].base_shader_register == 4);
    CHECK(d.parameters[0].ranges[0].register_space == 2);
    CHECK(d.parameters[1].parameter_type == RootParameterType::Constants32Bit);
    CHECK(d.parameters[1].shader_register == 7 && d.parameters[1].num_32bit_values == 2);
    RootSignature rs;
    CHECK(RootSignature::build(d, &rs).ok());
    CHECK(rs.cost_dwords() == 3);
    // The serializer reproduces exactly this layout.
    CHECK(serialize_root_signature(d) == b);
}

void test_root_signature_blob_malformed() {
    RootSignatureDesc out;
    const std::vector<uint8_t> good = serialize_root_signature(representative_desc());

    CHECK(parse_root_signature_blob(nullptr, 0, &out).status == Status::BlobTruncated);
    CHECK(parse_root_signature_blob(nullptr, 10, &out).status == Status::InvalidArgument);
    CHECK(parse_root_signature_blob(good.data(), good.size(), nullptr).status == Status::InvalidArgument);
    CHECK(parse_root_signature_blob(good.data(), 3, &out).status == Status::BlobTruncated);
    CHECK(parse_root_signature_blob(good.data(), 20, &out).status == Status::BlobTruncated);

    std::vector<uint8_t> bad = good;
    bad[0] = 9;  // version 9 / not DXBC
    CHECK(parse_root_signature_blob(bad.data(), bad.size(), &out).status == Status::BlobBadMagic);

    bad = good;
    bad[10] = 0xf0;  // parameters offset far out of bounds
    RootSignatureError e = parse_root_signature_blob(bad.data(), bad.size(), &out);
    CHECK(e.status == Status::BlobOffsetOutOfBounds && e.byte_offset == 8);

    bad = good;
    bad[4] = 0xff;  // 255 parameters: array would run past the end
    bad[5] = 0xff;
    CHECK(parse_root_signature_blob(bad.data(), bad.size(), &out).status == Status::BlobOffsetOutOfBounds);

    bad = good;
    bad[24] = 9;  // parameter 0 type
    e = parse_root_signature_blob(bad.data(), bad.size(), &out);
    CHECK(e.status == Status::InvalidParameterType && e.parameter_index == 0 && e.byte_offset == 24);

    bad = good;
    bad[24 + 8] = 0xf0;  // parameter 0 payload offset
    bad[24 + 9] = 0xff;
    e = parse_root_signature_blob(bad.data(), bad.size(), &out);
    CHECK(e.status == Status::BlobOffsetOutOfBounds && e.parameter_index == 0);

    bad = good;
    bad[12] = 0xff;  // static sampler count
    CHECK(parse_root_signature_blob(bad.data(), bad.size(), &out).status == Status::BlobOffsetOutOfBounds);

    // Every strict prefix fails (all bytes are referenced) and never crashes.
    for (size_t n = 0; n < good.size(); ++n) {
        CHECK(!parse_root_signature_blob(good.data(), n, &out).ok());
    }

    // DXBC containers.
    std::vector<uint8_t> dxbc_no_rts0 = wrap_dxbc(good, true);
    dxbc_no_rts0[dxbc_no_rts0.size() - good.size() - 8] = 'X';  // rename RTS0 part
    CHECK(parse_root_signature_blob(dxbc_no_rts0.data(), dxbc_no_rts0.size(), &out).status ==
          Status::BlobMissingRootSignaturePart);
    std::vector<uint8_t> dxbc = wrap_dxbc(good, false);
    dxbc[32] = 0xff;  // part offset out of bounds
    dxbc[33] = 0xff;
    CHECK(parse_root_signature_blob(dxbc.data(), dxbc.size(), &out).status == Status::BlobOffsetOutOfBounds);
    dxbc = wrap_dxbc(good, false);
    CHECK(parse_root_signature_blob(dxbc.data(), dxbc.size() - 1, &out).status == Status::BlobOffsetOutOfBounds);
    dxbc = wrap_dxbc(good, false);
    dxbc[28] = 0xff;  // part count
    CHECK(parse_root_signature_blob(dxbc.data(), dxbc.size(), &out).status == Status::BlobOffsetOutOfBounds);
    CHECK(parse_root_signature_blob(dxbc.data(), 30, &out).status == Status::BlobTruncated);

    // Deterministic byte-flip fuzzing: must never crash; parsed results must
    // still go through RootSignature::build validation without crashing.
    uint32_t seed = 0x12345678u;
    for (int iter = 0; iter < 4000; ++iter) {
        std::vector<uint8_t> f = (iter & 1) ? good : wrap_dxbc(good, true);
        for (int k = 0; k < 3; ++k) {
            seed = seed * 1664525u + 1013904223u;
            f[(seed >> 8) % f.size()] = static_cast<uint8_t>(seed >> 24);
        }
        RootSignatureDesc fd;
        if (parse_root_signature_blob(f.data(), f.size(), &fd).ok()) {
            RootSignature rs;
            (void)RootSignature::build(fd, &rs);
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Root argument state
// ---------------------------------------------------------------------------

void test_root_arguments() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* cbv = make_heap(reg, DescriptorHeapType::CbvSrvUav, 64, true);
    DescriptorHeap* cbv2 = make_heap(reg, DescriptorHeapType::CbvSrvUav, 64, true);
    DescriptorHeap* smp = make_heap(reg, DescriptorHeapType::Sampler, 16, true);
    DescriptorHeap* cpu_only = make_heap(reg, DescriptorHeapType::CbvSrvUav, 64, false);
    DescriptorHeap* rtv = make_heap(reg, DescriptorHeapType::Rtv, 4, false);
    RootSignature rs;
    CHECK(RootSignature::build(representative_desc(), &rs).ok());
    RootSignature other;
    RootSignatureDesc od;
    od.parameters.push_back(root_view(RootParameterType::Uav, 0));
    CHECK(RootSignature::build(od, &other).ok());

    RootArgumentState st;
    const BindPoint G = BindPoint::Graphics;
    const BindPoint C = BindPoint::Compute;

    // SetDescriptorHeaps validation.
    DescriptorHeap* heaps[2] = {cbv, smp};
    DescriptorHeap* two_cbv[2] = {cbv, cbv2};
    DescriptorHeap* non_sv[1] = {cpu_only};
    DescriptorHeap* rtv_list[1] = {rtv};
    CHECK(st.set_descriptor_heaps(2, two_cbv) == Status::InvalidArgument);
    CHECK(st.set_descriptor_heaps(1, non_sv) == Status::HeapNotShaderVisible);
    CHECK(st.set_descriptor_heaps(1, rtv_list) == Status::WrongHeapType);
    CHECK(st.set_descriptor_heaps(3, heaps) == Status::InvalidArgument);
    CHECK(st.set_descriptor_heaps(2, heaps) == Status::Ok);
    CHECK(st.bound_heap(DescriptorHeapType::CbvSrvUav) == cbv);
    CHECK(st.bound_heap(DescriptorHeapType::Sampler) == smp);

    // Arguments before a root signature.
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv, 0)) == Status::RootSignatureNotSet);
    CHECK(st.set_root_signature(G, nullptr) == Status::InvalidArgument);
    CHECK(st.set_root_signature(G, &rs) == Status::Ok);
    CHECK(st.root_signature(G) == &rs);
    CHECK(st.root_signature(C) == nullptr);
    CHECK(st.dirty_parameters(G) == 0x3f);
    CHECK(st.unset_parameters(G) == 0x3f);
    st.clear_dirty(G);
    CHECK(st.dirty_parameters(G) == 0);

    // Descriptor tables.
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv, 8)) == Status::Ok);
    CHECK(st.argument(G, 0).heap == cbv && st.argument(G, 0).base_index == 8 && st.argument(G, 0).is_set);
    CHECK(st.dirty_parameters(G) == 0x1);
    st.clear_dirty(G);
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv, 8)) == Status::Ok);  // same value: not dirty
    CHECK(st.dirty_parameters(G) == 0);
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv, 53)) == Status::Ok);  // 53 + 11 == 64 fits
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv, 54)) == Status::HandleOutOfRange);
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv, 64)) == Status::HandleOutOfRange);
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv2, 0)) == Status::HeapNotBound);
    CHECK(st.set_descriptor_table(G, 0, gpu_at(smp, 0)) == Status::WrongHeapType);
    CHECK(st.set_descriptor_table(G, 1, gpu_at(cbv, 0)) == Status::WrongHeapType);
    CHECK(st.set_descriptor_table(G, 0, GpuDescriptorHandle{cbv->cpu_start().ptr}) == Status::InvalidHandle);
    CHECK(st.set_descriptor_table(G, 0, GpuDescriptorHandle{cbv->gpu_start().ptr + 8}) == Status::MisalignedHandle);
    CHECK(st.set_descriptor_table(G, 0, GpuDescriptorHandle{0}) == Status::InvalidHandle);
    CHECK(st.set_descriptor_table(G, 2, gpu_at(cbv, 0)) == Status::RootParameterTypeMismatch);
    CHECK(st.set_descriptor_table(G, 6, gpu_at(cbv, 0)) == Status::InvalidRootParameterIndex);
    CHECK(st.set_descriptor_table(G, 1, gpu_at(smp, 14)) == Status::Ok);
    CHECK(st.set_descriptor_table(G, 1, gpu_at(smp, 15)) == Status::HandleOutOfRange);
    // Unbounded table: only the base must be in range.
    CHECK(st.set_descriptor_table(G, 5, gpu_at(cbv, 63)) == Status::Ok);

    // Root views.
    st.clear_dirty(G);
    CHECK(st.set_root_view(G, 3, RootParameterType::Cbv, 0x10000) == Status::Ok);
    CHECK(st.argument(G, 3).gpu_address == 0x10000);
    CHECK(st.dirty_parameters(G) == (1ull << 3));
    CHECK(st.set_root_view(G, 3, RootParameterType::Srv, 0x10000) == Status::RootParameterTypeMismatch);
    CHECK(st.set_root_view(G, 0, RootParameterType::Cbv, 0x10000) == Status::RootParameterTypeMismatch);
    CHECK(st.set_root_view(G, 3, RootParameterType::DescriptorTable, 0) == Status::InvalidArgument);
    CHECK(st.set_root_view(G, 4, RootParameterType::Srv, 0x100000) == Status::Ok);

    // Root constants.
    st.clear_dirty(G);
    const uint32_t vals[4] = {1, 2, 3, 4};
    CHECK(st.set_root_constants(G, 2, 4, vals, 0) == Status::Ok);
    CHECK(st.dirty_parameters(G) == (1ull << 2));
    std::span<const uint32_t> c = st.root_constants(G, 2);
    CHECK(c.size() == 4 && c[0] == 1 && c[3] == 4);
    st.clear_dirty(G);
    CHECK(st.set_root_constant(G, 2, 3, 2) == Status::Ok);  // unchanged value
    CHECK(st.dirty_parameters(G) == 0);
    CHECK(st.set_root_constant(G, 2, 9, 3) == Status::Ok);
    CHECK(st.dirty_parameters(G) == (1ull << 2));
    CHECK(st.root_constants(G, 2)[3] == 9);
    CHECK(st.set_root_constants(G, 2, 2, vals, 3) == Status::ConstantsOutOfRange);
    CHECK(st.set_root_constants(G, 2, 1, nullptr, 0) == Status::InvalidArgument);
    CHECK(st.set_root_constant(G, 3, 1, 0) == Status::RootParameterTypeMismatch);
    CHECK(st.root_constants(G, 3).empty());
    CHECK(st.unset_parameters(G) == 0);

    // Compute bind point is independent.
    CHECK(st.set_root_signature(C, &other) == Status::Ok);
    CHECK(st.unset_parameters(C) == 0x1);
    CHECK(st.set_root_view(C, 0, RootParameterType::Uav, 0x100000) == Status::Ok);
    CHECK(st.unset_parameters(C) == 0);
    CHECK(st.unset_parameters(G) == 0);

    // Re-setting the same signature keeps arguments; a different one clears them.
    st.clear_dirty(G);
    CHECK(st.set_root_signature(G, &rs) == Status::Ok);
    CHECK(st.dirty_parameters(G) == 0 && st.argument(G, 3).is_set);
    CHECK(st.set_root_signature(G, &other) == Status::Ok);
    CHECK(st.dirty_parameters(G) == 0x1 && st.unset_parameters(G) == 0x1);
    CHECK(st.set_root_signature(G, &rs) == Status::Ok);
    CHECK(!st.argument(G, 3).is_set && st.unset_parameters(G) == 0x3f);

    // Changing a descriptor heap clears tables that pointed into the old one.
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv, 0)) == Status::Ok);
    CHECK(st.set_descriptor_table(G, 1, gpu_at(smp, 0)) == Status::Ok);
    CHECK(st.set_root_view(G, 3, RootParameterType::Cbv, 0x10000) == Status::Ok);
    st.clear_dirty(G);
    DescriptorHeap* swap[2] = {smp, cbv2};
    CHECK(st.set_descriptor_heaps(2, swap) == Status::Ok);
    CHECK(!st.argument(G, 0).is_set);
    CHECK(st.argument(G, 1).is_set);  // sampler heap unchanged
    CHECK(st.argument(G, 3).is_set);
    CHECK(st.dirty_parameters(G) == 0x1);
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv, 0)) == Status::HeapNotBound);
    CHECK(st.set_descriptor_table(G, 0, gpu_at(cbv2, 0)) == Status::Ok);

    st.reset();
    CHECK(st.root_signature(G) == nullptr && st.bound_heap(DescriptorHeapType::CbvSrvUav) == nullptr);
    CHECK(!st.argument(G, 0).is_set);
}

// ---------------------------------------------------------------------------
// 8. Bind group compilation & caching
// ---------------------------------------------------------------------------

ShaderBinding sb(uint32_t binding, DescriptorRangeType cls, uint32_t reg, uint32_t space = 0) {
    ShaderBinding b;
    b.binding = binding;
    b.register_class = cls;
    b.shader_register = reg;
    b.register_space = space;
    return b;
}

void test_bind_group_plan() {
    RootSignature rs;
    CHECK(RootSignature::build(representative_desc(), &rs).ok());
    const uint32_t vf = WebGpuStage::Vertex | WebGpuStage::Fragment;

    // Shader-owned binding numbers in arbitrary order, as S6 reflects them.
    const ShaderBinding group0[] = {
        sb(7, DescriptorRangeType::Uav, 0),      // table p0, offset 10
        sb(0, DescriptorRangeType::Cbv, 0),      // table p0, offset 0
        sb(3, DescriptorRangeType::Srv, 2),      // table p0, offset 4
        sb(9, DescriptorRangeType::Srv, 42, 1),  // unbounded table p5, offset 42
        sb(12, DescriptorRangeType::Srv, 9),     // no slot -> unresolved
    };
    const BindGroupPlan plan = BindGroupPlan::build(rs, group0, vf);
    CHECK(plan.signature_hash() == rs.hash());
    CHECK(plan.entries().size() == 5);
    CHECK(plan.unresolved_count() == 1);
    CHECK(plan.parameter_mask() == ((1ull << 0) | (1ull << 5)));
    const std::vector<PlanEntry>& e = plan.entries();
    CHECK(e[0].binding == 0 && e[0].source == PlanSource::DescriptorTable && e[0].table_offset == 0);
    CHECK(e[1].binding == 3 && e[1].table_offset == 4 && e[1].root_parameter_index == 0);
    CHECK(e[2].binding == 7 && e[2].table_offset == 10 && e[2].register_class == DescriptorRangeType::Uav);
    CHECK(e[3].binding == 9 && e[3].root_parameter_index == 5 && e[3].table_offset == 42);
    CHECK(e[4].binding == 12 && e[4].source == PlanSource::Unresolved);
    CHECK(plan.table_windows().size() == 2);
    CHECK(plan.table_windows()[0].parameter_index == 0 && plan.table_windows()[0].first_offset == 0);
    CHECK(plan.table_windows()[0].count == 11);
    CHECK(plan.table_windows()[1].parameter_index == 5 && plan.table_windows()[1].first_offset == 42);
    CHECK(plan.table_windows()[1].count == 1);

    // Same join -> same hash; different bindings -> different hash.
    CHECK(BindGroupPlan::build(rs, group0, vf).hash() == plan.hash());
    const ShaderBinding other[] = {sb(1, DescriptorRangeType::Cbv, 0)};
    CHECK(BindGroupPlan::build(rs, other, vf).hash() != plan.hash());

    // Root constants (S6 root_constants: true), root views, static samplers, visibility.
    ShaderBinding pc;
    pc.binding = 4;
    pc.root_constants = true;
    const ShaderBinding group1[] = {pc, sb(0, DescriptorRangeType::Cbv, 3), sb(1, DescriptorRangeType::Srv, 10),
                                    sb(2, DescriptorRangeType::Sampler, 4), sb(3, DescriptorRangeType::Sampler, 1)};
    const BindGroupPlan frag = BindGroupPlan::build(rs, group1, WebGpuStage::Fragment);
    CHECK(frag.unresolved_count() == 0);
    CHECK(frag.entries()[0].source == PlanSource::RootDescriptor && frag.entries()[0].root_parameter_index == 3);
    CHECK(frag.entries()[1].source == PlanSource::RootDescriptor && frag.entries()[1].root_parameter_index == 4);
    CHECK(frag.entries()[2].source == PlanSource::StaticSampler && frag.entries()[2].root_parameter_index == 0);
    CHECK(frag.entries()[3].source == PlanSource::DescriptorTable && frag.entries()[3].root_parameter_index == 1);
    CHECK(frag.entries()[4].source == PlanSource::RootConstants && frag.entries()[4].root_parameter_index == 2);
    CHECK(frag.parameter_mask() == ((1ull << 1) | (1ull << 2) | (1ull << 3) | (1ull << 4)));
    // Vertex stage cannot see the pixel-only root SRV and sampler table.
    const BindGroupPlan vert = BindGroupPlan::build(rs, group1, WebGpuStage::Vertex);
    CHECK(vert.unresolved_count() == 2);
    // Compute ignores visibility.
    CHECK(BindGroupPlan::build(rs, group1, WebGpuStage::Compute).unresolved_count() == 0);

    // Root constants by register (b2) resolve to the same parameter.
    const ShaderBinding by_reg[] = {sb(0, DescriptorRangeType::Cbv, 2)};
    const BindGroupPlan rc = BindGroupPlan::build(rs, by_reg, vf);
    CHECK(rc.entries()[0].source == PlanSource::RootConstants && rc.entries()[0].root_parameter_index == 2);

    // Empty layout.
    const BindGroupPlan empty = BindGroupPlan::build(rs, std::span<const ShaderBinding>{}, vf);
    CHECK(empty.entries().empty() && empty.parameter_mask() == 0);
}

// SM 6.6 directly indexed heaps (S34 heap sources): the descriptor comes from the bound heap at
// byte offset (root constant << shift), chosen at draw time.
void test_heap_indexed_bind_groups() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 64, true);
    DescriptorHeap* smp = make_heap(reg, DescriptorHeapType::Sampler, 16, true);
    RootSignatureDesc d;
    d.flags = 0x400 | 0x800;  // CBV_SRV_UAV / SAMPLER heap directly indexed
    d.parameters.push_back(constants(0, 2));  // s8..s9
    RootSignature rs;
    CHECK(RootSignature::build(d, &rs).ok());

    SamplerDesc a;
    a.max_anisotropy = 4;
    SamplerDesc b;
    b.max_anisotropy = 8;
    CHECK(reg.create_sampler(a, cpu_at(smp, 3)) == Status::Ok);
    CHECK(reg.create_sampler(b, cpu_at(smp, 5)) == Status::Ok);
    fill_srvs(reg, heap, 7, 1, 700);
    fill_srvs(reg, heap, 9, 1, 900);

    // Sampler index in constant 0 (byte offset = value << 4), texture index in constant 1 (<< 5).
    ShaderBinding s;
    s.binding = 0;
    s.register_class = DescriptorRangeType::Sampler;
    s.heap = HeapAccess::Sampler;
    s.heap_index_parameter = 0;
    s.heap_index_dword = 0;
    s.heap_shift = 4;
    ShaderBinding t;
    t.binding = 1;
    t.register_class = DescriptorRangeType::Srv;
    t.heap = HeapAccess::Resource;
    t.heap_index_parameter = 0;
    t.heap_index_dword = 1;
    t.heap_shift = 5;
    const ShaderBinding bindings[] = {s, t};
    const BindGroupPlan plan = BindGroupPlan::build(rs, bindings, WebGpuStage::Fragment);
    CHECK(plan.unresolved_count() == 0 && plan.parameter_mask() == 1);
    CHECK(plan.entries()[0].source == PlanSource::HeapIndexed && plan.entries()[1].source == PlanSource::HeapIndexed);

    // The index must be a value of a constants parameter, and the heap must match the class.
    ShaderBinding off_end = s;
    off_end.heap_index_dword = 2;
    ShaderBinding wrong_heap = t;
    wrong_heap.heap = HeapAccess::Sampler;
    const ShaderBinding bad[] = {off_end, wrong_heap};
    CHECK(BindGroupPlan::build(rs, bad, WebGpuStage::Fragment).unresolved_count() == 2);

    RootArgumentState st;
    BindGroupCache cache;
    const BindPoint G = BindPoint::Graphics;
    DescriptorHeap* heaps[2] = {heap, smp};
    CHECK(st.set_descriptor_heaps(2, heaps) == Status::Ok);
    CHECK(st.set_root_signature(G, &rs) == Status::Ok);
    const uint32_t k0[2] = {3, 7};
    CHECK(st.set_root_constants(G, 0, 2, k0, 0) == Status::Ok);
    BindGroupLookup g = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(g.status == Status::Ok && g.rebuilt && g.bind_group != nullptr);
    CHECK(g.bind_group->entries[0].kind == BindResourceKind::Sampler &&
          g.bind_group->entries[0].sampler.max_anisotropy == 4);
    CHECK(g.bind_group->entries[1].kind == BindResourceKind::TextureView && g.bind_group->entries[1].resource_id == 700);
    CHECK(g.bind_group->null_entries == 0 && cache.heap_index_misses() == 0);
    g.bind_group->gpu_id = 50;
    CHECK(!cache.get_or_build(st, G, plan, 1, test_resolver, nullptr).rebuilt);

    // Changing the root constants selects other descriptors.
    const uint32_t k1[2] = {5, 9};
    CHECK(st.set_root_constants(G, 0, 2, k1, 0) == Status::Ok);
    BindGroupLookup g2 = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(g2.rebuilt && g2.bind_group != g.bind_group);
    CHECK(g2.bind_group->entries[0].sampler.max_anisotropy == 8 && g2.bind_group->entries[1].resource_id == 900);
    g2.bind_group->gpu_id = 51;
    // Back to the first values: the first group again, unchanged.
    CHECK(st.set_root_constants(G, 0, 2, k0, 0) == Status::Ok);
    BindGroupLookup back = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(!back.rebuilt && back.bind_group->gpu_id == 50);
    // Rewriting the slot the group reads rebuilds it in place.
    fill_srvs(reg, heap, 7, 1, 701);
    BindGroupLookup rewritten = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(rewritten.rebuilt && rewritten.retired_gpu_id == 50 && rewritten.bind_group->entries[1].resource_id == 701);

    // 32-bit soffset wrap (G66): shifted indices overflowing 32 bits wrap like the hardware and land in the heap.
    const uint32_t wrapk[2] = {0x10000003u, 0x08000007u};  // << 4 -> 0x30 (slot 3), << 5 -> 0xE0 (slot 7)
    CHECK(st.set_root_constants(G, 0, 2, wrapk, 0) == Status::Ok);
    BindGroupLookup wrapped = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(wrapped.status == Status::Ok && wrapped.bind_group != nullptr);
    CHECK(wrapped.bind_group->entries[0].kind == BindResourceKind::Sampler &&
          wrapped.bind_group->entries[0].sampler.max_anisotropy == 4);
    CHECK(wrapped.bind_group->entries[1].kind == BindResourceKind::TextureView &&
          wrapped.bind_group->entries[1].resource_id == 701);

    // Out of range (and not a whole descriptor): null entries, counted, no crash.
    const uint32_t far[2] = {16, 0xffffffffu};
    CHECK(st.set_root_constant(G, 0, far[0], 0) == Status::Ok);
    CHECK(st.set_root_constant(G, 0, far[1], 1) == Status::Ok);
    BindGroupLookup oob = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(oob.status == Status::Ok && oob.bind_group != nullptr);
    CHECK(oob.bind_group->entries[0].kind == BindResourceKind::Null && oob.bind_group->entries[1].kind == BindResourceKind::Null);
    CHECK(oob.bind_group->null_entries == 2 && cache.heap_index_misses() == 2);
    CHECK(!cache.get_or_build(st, G, plan, 1, test_resolver, nullptr).rebuilt);  // cached, not re-counted
    CHECK(cache.heap_index_misses() == 2);

    // No sampler heap bound: null, counted.
    DescriptorHeap* only_views[1] = {heap};
    CHECK(st.set_descriptor_heaps(1, only_views) == Status::Ok);
    CHECK(st.set_root_constants(G, 0, 2, k0, 0) == Status::Ok);
    BindGroupLookup no_heap = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(no_heap.status == Status::Ok && no_heap.bind_group->entries[0].kind == BindResourceKind::Null);
    CHECK(no_heap.bind_group->entries[1].resource_id == 701 && cache.heap_index_misses() == 3);

    // O36 end_epoch: under the bound nothing is dropped; above it, entries this epoch did not use
    // are dropped (their gpu ids retired), and a full epoch above the bound empties the cache.
    {
        std::vector<uint32_t> retired;
        const size_t before = cache.size();
        CHECK(before >= 3);
        CHECK(cache.end_epoch(before, &retired) == 0 && retired.empty() && cache.size() == before);
        CHECK(st.set_root_constants(G, 0, 2, k1, 0) == Status::Ok);
        BindGroupLookup used = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);  // used this epoch
        CHECK(used.bind_group != nullptr);
        used.bind_group->gpu_id = 99;
        const size_t cached = cache.size();
        CHECK(cache.end_epoch(1, &retired) == cached - 1 && cache.size() == 1);
        CHECK(std::find(retired.begin(), retired.end(), 99u) == retired.end());
        CHECK(cache.end_epoch(0, &retired) == 1 && cache.size() == 0);
        CHECK(std::find(retired.begin(), retired.end(), 99u) != retired.end());
    }
    // #2203 pressure trim: with trim_over below the size, the unused entries are dropped even though
    // the cache is under max_entries, and the entries the epoch used survive.
    {
        std::vector<uint32_t> retired;
        cache.end_epoch(1000, &retired);  // new epoch, nothing used in it yet
        CHECK(st.set_root_constants(G, 0, 2, k1, 0) == Status::Ok);
        BindGroupLookup used = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
        CHECK(used.bind_group != nullptr);
        const size_t cached = cache.size();
        CHECK(cached >= 1);
        cache.end_epoch(1000, &retired, 0);
        CHECK(cache.size() == 1 && used.bind_group != nullptr);
        cache.end_epoch(0, &retired);
        CHECK(cache.size() == 0);
    }
    // Backlog 28: the live-entry gauge follows the cache through insert, eviction and clear.
    {
        const uint64_t others = descriptor_live_counts().cached_bind_groups - cache.size();
        CHECK(st.set_root_constants(G, 0, 2, k1, 0) == Status::Ok);
        BindGroupLookup l = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
        CHECK(l.bind_group != nullptr && cache.size() >= 1);
        CHECK(descriptor_live_counts().cached_bind_groups == others + cache.size());
        cache.clear(nullptr);
        CHECK(descriptor_live_counts().cached_bind_groups == others);
    }
    // Backlog 34: a tiny slot budget never drops a lookup (the cache grows), it counts the overflow;
    // the per-epoch peak tracks the most entries one recording needed.
    {
        reset_bind_group_slot_counters();
        cache.clear(nullptr);
        set_bind_group_slot_budget(2);
        const uint32_t consts[4][2] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
        for (const auto& c : consts) {
            CHECK(st.set_root_constants(G, 0, 2, c, 0) == Status::Ok);
            CHECK(cache.get_or_build(st, G, plan, 1, test_resolver, nullptr).bind_group != nullptr);
        }
        DescriptorLiveCounts c = descriptor_live_counts();
        CHECK(cache.size() == 4);
        CHECK(c.bind_group_slot_budget == 2 && c.bind_group_budget_overflows == 2);
        CHECK(c.peak_bind_groups_per_epoch >= 4 && c.draws_dropped_no_slot == 0);
        cache.end_epoch(0, nullptr);
        set_bind_group_slot_budget(0);
        CHECK(cache.get_or_build(st, G, plan, 1, test_resolver, nullptr).bind_group != nullptr);
        CHECK(descriptor_live_counts().bind_group_budget_overflows == 2);
        cache.clear(nullptr);
    }
}

// Binding array elements (G62, lane-indexed heap windows): element k reads the heap index plus k;
// a window without a base (HeapIndexSource::Constant) starts at heap index 0.
void test_heap_indexed_binding_array() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 16, true);
    RootSignatureDesc d;
    d.flags = 0x400;
    d.parameters.push_back(constants(0, 1));
    RootSignature rs;
    CHECK(RootSignature::build(d, &rs).ok());
    fill_srvs(reg, heap, 0, 8, 100);  // slot i -> resource 100 + i

    std::vector<ShaderBinding> bindings;
    for (uint32_t k = 0; k < 3; ++k) {
        ShaderBinding based;  // window base in root constant 0
        based.binding = 0;
        based.register_class = DescriptorRangeType::Srv;
        based.heap = HeapAccess::Resource;
        based.heap_index_parameter = 0;
        based.heap_shift = 5;
        based.array_size = 3;
        based.array_index = k;
        bindings.push_back(based);
    }
    for (uint32_t k = 0; k < 2; ++k) {
        ShaderBinding fixed;  // no window base
        fixed.binding = 1;
        fixed.register_class = DescriptorRangeType::Srv;
        fixed.heap = HeapAccess::Resource;
        fixed.heap_index_source = HeapIndexSource::Constant;
        fixed.heap_shift = 5;
        fixed.array_size = 2;
        fixed.array_index = k;
        bindings.push_back(fixed);
    }
    CHECK(!same_join(bindings[0], bindings[1]));
    const BindGroupPlan plan = BindGroupPlan::build(rs, bindings, WebGpuStage::Fragment);
    CHECK(plan.unresolved_count() == 0 && plan.parameter_mask() == 1);

    RootArgumentState st;
    BindGroupCache cache;
    const BindPoint G = BindPoint::Graphics;
    DescriptorHeap* heaps[1] = {heap};
    CHECK(st.set_descriptor_heaps(1, heaps) == Status::Ok);
    CHECK(st.set_root_signature(G, &rs) == Status::Ok);
    CHECK(st.set_root_constant(G, 0, 4, 0) == Status::Ok);  // window base 4
    BindGroupLookup g = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(g.status == Status::Ok && g.bind_group != nullptr && g.bind_group->entries.size() == 5);
    if (g.bind_group != nullptr && g.bind_group->entries.size() == 5) {
        CHECK(g.bind_group->entries[0].resource_id == 104 && g.bind_group->entries[1].resource_id == 105 &&
              g.bind_group->entries[2].resource_id == 106);
        CHECK(g.bind_group->entries[3].resource_id == 100 && g.bind_group->entries[4].resource_id == 101);
        CHECK(g.bind_group->null_entries == 0);
    }
    // Moving the window base selects other descriptors.
    CHECK(st.set_root_constant(G, 0, 1, 0) == Status::Ok);
    BindGroupLookup moved = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(moved.rebuilt && moved.bind_group != nullptr && moved.bind_group->entries[0].resource_id == 101 &&
          moved.bind_group->entries[2].resource_id == 103);
}

// Non-power-of-two heap stride (G62, S63 lane-indexed windows scaled with s_mul_i32): the byte
// offset is index * heap_stride, not index << heap_shift; the stride is part of the join identity.
void test_heap_indexed_stride() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 16, true);
    RootSignatureDesc d;
    d.flags = 0x400;
    d.parameters.push_back(constants(0, 1));
    RootSignature rs;
    CHECK(RootSignature::build(d, &rs).ok());
    fill_srvs(reg, heap, 0, 16, 100);  // slot i -> resource 100 + i

    std::vector<ShaderBinding> bindings;
    for (uint32_t k = 0; k < 3; ++k) {
        ShaderBinding b;  // window base in root constant 0, 96-byte step (3 descriptors of 32 bytes)
        b.binding = 0;
        b.register_class = DescriptorRangeType::Srv;
        b.heap = HeapAccess::Resource;
        b.heap_index_parameter = 0;
        b.heap_shift = 0;
        b.heap_stride = 96;
        b.array_size = 3;
        b.array_index = k;
        bindings.push_back(b);
    }
    ShaderBinding shifted = bindings[0];
    shifted.heap_stride = 0;
    CHECK(!same_join(bindings[0], shifted));
    const BindGroupPlan plan = BindGroupPlan::build(rs, bindings, WebGpuStage::Fragment);
    CHECK(plan.unresolved_count() == 0);
    const std::vector<ShaderBinding> shifted_bindings = {shifted};
    const BindGroupPlan shift_plan = BindGroupPlan::build(rs, shifted_bindings, WebGpuStage::Fragment);
    CHECK(plan.hash() != shift_plan.hash());

    RootArgumentState st;
    BindGroupCache cache;
    const BindPoint G = BindPoint::Graphics;
    DescriptorHeap* heaps[1] = {heap};
    CHECK(st.set_descriptor_heaps(1, heaps) == Status::Ok);
    CHECK(st.set_root_signature(G, &rs) == Status::Ok);
    CHECK(st.set_root_constant(G, 0, 1, 0) == Status::Ok);  // window base 1
    BindGroupLookup g = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(g.status == Status::Ok && g.bind_group != nullptr && g.bind_group->entries.size() == 3);
    if (g.bind_group != nullptr && g.bind_group->entries.size() == 3) {
        // (1 + k) * 96 / 32 = slots 3, 6, 9.
        CHECK(g.bind_group->entries[0].resource_id == 103 && g.bind_group->entries[1].resource_id == 106 &&
              g.bind_group->entries[2].resource_id == 109);
        CHECK(g.bind_group->null_entries == 0);
    }
}

// CB-indexed heaps (G62, #482): SM 6.6 ResourceDescriptorHeap[cb.index], the index read from a
// constant buffer (root CBV, a table slot's CBV or root constants) when the draw is recorded.
// Guest memory here: a dword map the slot source reads (it decodes no heap itself).
class CbMemory final : public DescriptorSlotSource {
public:
    std::map<uint64_t, uint32_t> dwords;
    mutable uint32_t reads_count = 0;
    bool reads(const DescriptorHeap&) const override { return false; }
    bool snapshot(const DescriptorHeap&, uint32_t, uint32_t, std::vector<uint8_t>* out) const override {
        out->clear();
        return false;
    }
    ViewDescriptor view(const DescriptorHeap&, uint32_t, DescriptorRangeType) const override { return {}; }
    bool sampler(const DescriptorHeap&, uint32_t, SamplerDesc*) const override { return false; }
    bool read_gpu_memory(uint64_t va, void* out, size_t bytes) const override {
        ++reads_count;
        auto it = dwords.find(va);
        if (bytes != 4 || it == dwords.end()) {
            return false;
        }
        std::memcpy(out, &it->second, 4);
        return true;
    }
};

void test_cb_indexed_heap_bind_groups() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 64, true);
    DescriptorHeap* smp = make_heap(reg, DescriptorHeapType::Sampler, 16, true);
    RootSignatureDesc d;
    d.flags = 0x400 | 0x800;  // CBV_SRV_UAV / SAMPLER heap directly indexed
    d.parameters.push_back(root_view(RootParameterType::Cbv, 0));                    // p0: root CBV
    d.parameters.push_back(table({range(DescriptorRangeType::Cbv, 1, 1)}));         // p1: table { CBV b1 }
    d.parameters.push_back(constants(2, 4));                                         // p2: 4 constants
    RootSignature rs;
    CHECK(RootSignature::build(d, &rs).ok());

    fill_srvs(reg, heap, 7, 1, 700);
    fill_srvs(reg, heap, 9, 1, 900);
    fill_srvs(reg, heap, 11, 1, 1100);
    SamplerDesc a;
    a.max_anisotropy = 4;
    CHECK(reg.create_sampler(a, cpu_at(smp, 3)) == Status::Ok);
    // Table at slot 40: a CBV at 0x5000 (256 bytes).
    ConstantBufferViewDesc cbv{0x5000, 256};
    CHECK(reg.create_constant_buffer_view(&cbv, cpu_at(heap, 40)) == Status::Ok);

    CbMemory mem;
    mem.dwords[0x8010] = 7;           // root CBV at 0x8000, index at byte 16
    mem.dwords[0x5020] = 0x00090000;  // table CBV, index in bits 16.. at byte 32
    mem.dwords[0x8004] = 3;           // sampler index (root CBV, byte 4)

    // Texture 0 <- heap[rootCBV[16] << 5]; texture 1 <- heap[((tableCBV[32] >> 16) & 0xff) << 5];
    // texture 2 <- heap[constants[8] << 5]; sampler 3 <- sampler heap[rootCBV[4] << 4].
    ShaderBinding t0;
    t0.binding = 0;
    t0.register_class = DescriptorRangeType::Srv;
    t0.heap = HeapAccess::Resource;
    t0.heap_index_source = HeapIndexSource::CbRootDescriptor;
    t0.heap_index_parameter = 0;
    t0.cb_byte_offset = 16;
    t0.heap_shift = 5;
    ShaderBinding t1 = t0;
    t1.binding = 1;
    t1.heap_index_source = HeapIndexSource::CbTable;
    t1.heap_index_parameter = 1;
    t1.cb_table_slot = 0;
    t1.cb_byte_offset = 32;
    t1.cb_bit_offset = 16;
    t1.cb_mask = 0xff;
    ShaderBinding t2 = t0;
    t2.binding = 2;
    t2.heap_index_source = HeapIndexSource::CbRootConstants;
    t2.heap_index_parameter = 2;
    t2.cb_byte_offset = 8;
    ShaderBinding s3 = t0;
    s3.binding = 3;
    s3.register_class = DescriptorRangeType::Sampler;
    s3.heap = HeapAccess::Sampler;
    s3.cb_byte_offset = 4;
    s3.heap_shift = 4;
    const ShaderBinding bindings[] = {t0, t1, t2, s3};
    const BindGroupPlan plan = BindGroupPlan::build(rs, bindings, WebGpuStage::Fragment);
    CHECK(plan.unresolved_count() == 0 && plan.parameter_mask() == 7);
    for (const PlanEntry& e : plan.entries()) {
        CHECK(e.source == PlanSource::HeapIndexed);
    }
    CHECK(plan.table_windows().empty());  // the CBV slot is read for the index, not bound

    // The CB must be the parameter type the source names, the offset dword aligned and (root
    // constants) inside the parameter; the plan hash tells the sources apart.
    ShaderBinding not_cbv = t0;
    not_cbv.heap_index_parameter = 2;
    ShaderBinding not_table = t1;
    not_table.heap_index_parameter = 0;
    ShaderBinding past_constants = t2;
    past_constants.cb_byte_offset = 16;
    ShaderBinding unaligned = t0;
    unaligned.cb_byte_offset = 6;
    ShaderBinding bad_bits = t1;
    bad_bits.cb_bit_offset = 32;
    const ShaderBinding bad[] = {not_cbv, not_table, past_constants, unaligned, bad_bits};
    CHECK(BindGroupPlan::build(rs, bad, WebGpuStage::Fragment).unresolved_count() == 5);
    ShaderBinding other_offset = t0;
    other_offset.cb_byte_offset = 20;
    const ShaderBinding one[] = {t0};
    const ShaderBinding one_other[] = {other_offset};
    CHECK(BindGroupPlan::build(rs, one, WebGpuStage::Fragment).hash() !=
          BindGroupPlan::build(rs, one_other, WebGpuStage::Fragment).hash());
    ShaderBinding same = t0;
    CHECK(same_join(same, t0) && !same_join(t0, other_offset) && !same_join(t0, t1));

    RootArgumentState st;
    BindGroupCache cache;
    cache.set_slot_source(&mem);
    const BindPoint G = BindPoint::Graphics;
    DescriptorHeap* heaps[2] = {heap, smp};
    CHECK(st.set_descriptor_heaps(2, heaps) == Status::Ok);
    CHECK(st.set_root_signature(G, &rs) == Status::Ok);
    CHECK(st.set_root_view(G, 0, RootParameterType::Cbv, 0x8000) == Status::Ok);
    CHECK(st.set_descriptor_table(G, 1, gpu_at(heap, 40)) == Status::Ok);
    const uint32_t k[4] = {0, 0, 11, 0};
    CHECK(st.set_root_constants(G, 2, 4, k, 0) == Status::Ok);

    // Hand-made CB contents -> the descriptors they index.
    BindGroupLookup g = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(g.status == Status::Ok && g.rebuilt && g.bind_group != nullptr);
    CHECK(g.bind_group->entries[0].kind == BindResourceKind::TextureView && g.bind_group->entries[0].resource_id == 700);
    CHECK(g.bind_group->entries[1].kind == BindResourceKind::TextureView && g.bind_group->entries[1].resource_id == 900);
    CHECK(g.bind_group->entries[2].kind == BindResourceKind::TextureView && g.bind_group->entries[2].resource_id == 1100);
    CHECK(g.bind_group->entries[3].kind == BindResourceKind::Sampler && g.bind_group->entries[3].sampler.max_anisotropy == 4);
    CHECK(g.bind_group->null_entries == 0 && cache.heap_index_misses() == 0 && cache.cb_index_misses() == 0);
    g.bind_group->gpu_id = 60;
    CHECK(!cache.get_or_build(st, G, plan, 1, test_resolver, nullptr).rebuilt);  // same contents: cached

    // Changing the CB contents selects another descriptor: a new bind group.
    mem.dwords[0x8010] = 9;
    BindGroupLookup g2 = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(g2.rebuilt && g2.bind_group != g.bind_group && g2.bind_group->entries[0].resource_id == 900);
    g2.bind_group->gpu_id = 61;
    // Back to the first contents: the first group again.
    mem.dwords[0x8010] = 7;
    BindGroupLookup back = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(!back.rebuilt && back.bind_group->gpu_id == 60);
    // Another CB at the root CBV (same contents elsewhere) picks the index from there.
    mem.dwords[0x9010] = 11;
    mem.dwords[0x9004] = 3;
    CHECK(st.set_root_view(G, 0, RootParameterType::Cbv, 0x9000) == Status::Ok);
    BindGroupLookup moved = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(moved.rebuilt && moved.bind_group->entries[0].resource_id == 1100);
    moved.bind_group->gpu_id = 62;
    // Rewriting the heap slot the group reads rebuilds it in place.
    fill_srvs(reg, heap, 11, 1, 1101);
    BindGroupLookup rewritten = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(rewritten.rebuilt && rewritten.retired_gpu_id == 62 && rewritten.bind_group->entries[0].resource_id == 1101);
    CHECK(cache.heap_index_misses() == 0);

    // Out of range index: null, counted as a heap miss (not a CB miss), no crash.
    mem.dwords[0x9010] = 64;  // slot 64 of a 64-slot heap
    BindGroupLookup oob = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(oob.status == Status::Ok && oob.bind_group != nullptr);
    CHECK(oob.bind_group->entries[0].kind == BindResourceKind::Null && oob.bind_group->entries[1].resource_id == 900);
    CHECK(oob.bind_group->null_entries == 1 && cache.heap_index_misses() == 1 && cache.cb_index_misses() == 0);
    CHECK(!cache.get_or_build(st, G, plan, 1, test_resolver, nullptr).rebuilt);  // cached, not re-counted
    CHECK(cache.heap_index_misses() == 1);
    mem.dwords[0x9010] = 0xffffffffu;  // << 5 wraps (32-bit soffset) to 0xffffffe0: past the heap
    BindGroupLookup huge = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(huge.bind_group->entries[0].kind == BindResourceKind::Null && cache.heap_index_misses() == 2);

    // Unreadable CB (dword missing) and a root CBV at 0: null, counted as CB misses.
    mem.dwords.erase(0x9010);
    BindGroupLookup unreadable = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(unreadable.status == Status::Ok && unreadable.bind_group->entries[0].kind == BindResourceKind::Null);
    CHECK(cache.heap_index_misses() == 3 && cache.cb_index_misses() == 1);
    CHECK(st.set_root_view(G, 0, RootParameterType::Cbv, 0) == Status::Ok);
    BindGroupLookup no_cb = cache.get_or_build(st, G, plan, 1, test_resolver, nullptr);
    CHECK(no_cb.status == Status::Ok && no_cb.bind_group->entries[0].kind == BindResourceKind::Null &&
          no_cb.bind_group->entries[3].kind == BindResourceKind::Null);
    CHECK(cache.heap_index_misses() == 5 && cache.cb_index_misses() == 3);

    // Table CBV: index past the CBV's size, and a slot that holds no CBV: null, CB misses.
    CHECK(st.set_root_view(G, 0, RootParameterType::Cbv, 0x8000) == Status::Ok);
    ShaderBinding past_cbv = t1;
    past_cbv.binding = 0;
    past_cbv.cb_byte_offset = 256;
    ShaderBinding srv_slot = t1;
    srv_slot.binding = 1;
    srv_slot.cb_table_slot = 1;  // slot 41: empty
    const ShaderBinding table_bad[] = {past_cbv, srv_slot};
    const BindGroupPlan table_plan = BindGroupPlan::build(rs, table_bad, WebGpuStage::Fragment);
    CHECK(table_plan.unresolved_count() == 0);
    const uint64_t cb_before = cache.cb_index_misses();
    BindGroupLookup tb = cache.get_or_build(st, G, table_plan, 2, test_resolver, nullptr);
    CHECK(tb.status == Status::Ok && tb.bind_group->null_entries == 2 && cache.cb_index_misses() == cb_before + 2);

    // No slot source: a root-CBV index cannot be read (null, counted); root constants still work.
    BindGroupCache plain;
    const ShaderBinding rc_only[] = {t0, t2};
    const BindGroupPlan rc_plan = BindGroupPlan::build(rs, rc_only, WebGpuStage::Fragment);
    BindGroupLookup pl = plain.get_or_build(st, G, rc_plan, 3, test_resolver, nullptr);
    CHECK(pl.status == Status::Ok && pl.bind_group->entries[0].kind == BindResourceKind::Null &&
          pl.bind_group->entries[1].resource_id == 1101 && plain.cb_index_misses() == 1);
}

// Inline samplers (#482): the plan entry carries the decoded sampler; the bind group binds it with no
// root-signature sampler, table or heap involved, and the plan hash tells two samplers apart.
void test_inline_sampler_plan() {
    RootSignatureDesc d;
    d.parameters.push_back(constants(0, 1));
    RootSignature rs;
    CHECK(RootSignature::build(d, &rs).ok());
    ShaderBinding s;
    s.binding = 34;
    s.register_class = DescriptorRangeType::Sampler;
    s.inline_sampler = true;
    s.sampler.filter = 0x15;
    s.sampler.address_u = s.sampler.address_v = s.sampler.address_w = 3;
    s.sampler.max_lod = 15.0f;
    const ShaderBinding one[] = {s};
    const BindGroupPlan plan = BindGroupPlan::build(rs, one, WebGpuStage::Fragment);
    CHECK(plan.unresolved_count() == 0 && plan.parameter_mask() == 0);
    CHECK(plan.entries().size() == 1 && plan.entries()[0].source == PlanSource::InlineSampler);
    ShaderBinding other = s;
    other.sampler.filter = 0x80;
    other.sampler.comparison_func = 4;
    const ShaderBinding two[] = {other};
    CHECK(BindGroupPlan::build(rs, two, WebGpuStage::Fragment).hash() != plan.hash());

    RootArgumentState st;
    BindGroupCache cache;
    CHECK(st.set_root_signature(BindPoint::Graphics, &rs) == Status::Ok);
    CHECK(st.set_root_constant(BindPoint::Graphics, 0, 0, 0) == Status::Ok);
    BindGroupLookup g = cache.get_or_build(st, BindPoint::Graphics, plan, 1, test_resolver, nullptr);
    CHECK(g.status == Status::Ok && g.bind_group != nullptr && g.bind_group->entries.size() == 1);
    if (g.bind_group != nullptr && g.bind_group->entries.size() == 1) {
        const BindGroupEntry& e = g.bind_group->entries[0];
        CHECK(e.binding == 34 && e.kind == BindResourceKind::Sampler && e.sampler.filter == 0x15 &&
              e.sampler.address_u == 3 && e.sampler.max_lod == 15.0f && g.bind_group->null_entries == 0);
    }
    CHECK(!cache.get_or_build(st, BindPoint::Graphics, plan, 1, test_resolver, nullptr).rebuilt);
}

void test_bind_group_cache() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 256, true);
    DescriptorHeap* smp = make_heap(reg, DescriptorHeapType::Sampler, 16, true);
    RootSignature rs;
    CHECK(RootSignature::build(representative_desc(), &rs).ok());
    const uint32_t vf = WebGpuStage::Vertex | WebGpuStage::Fragment;

    // Table p0 at index 100: CBV b0,b1 (100,101), SRV t0-t3 (102..105), UAV u0 (110).
    ConstantBufferViewDesc cb{0x10200, 256};
    CHECK(reg.create_constant_buffer_view(&cb, cpu_at(heap, 100)) == Status::Ok);
    ConstantBufferViewDesc cb_unknown{0x900000, 256};
    CHECK(reg.create_constant_buffer_view(&cb_unknown, cpu_at(heap, 101)) == Status::Ok);
    fill_srvs(reg, heap, 102, 3, 500);  // 105 stays empty
    ViewDesc uav;
    uav.dimension = ViewDimension::Buffer;
    uav.first_element = 4;
    uav.num_elements = 10;
    uav.structure_byte_stride = 16;
    CHECK(reg.create_unordered_access_view(600, 0, &uav, cpu_at(heap, 110)) == Status::Ok);
    SamplerDesc sd;
    sd.max_anisotropy = 16;
    CHECK(reg.create_sampler(sd, cpu_at(smp, 3)) == Status::Ok);
    CHECK(reg.create_sampler(sd, cpu_at(smp, 4)) == Status::Ok);
    // Bindless table p5 at index 200 (space1, unbounded): t7 -> 207.
    fill_srvs(reg, heap, 207, 1, 800);

    // S6-style layouts: group 0 = table resources, group 1 = samplers, group 2 = root data.
    const ShaderBinding g0_bindings[] = {
        sb(0, DescriptorRangeType::Cbv, 0), sb(1, DescriptorRangeType::Cbv, 1),
        sb(2, DescriptorRangeType::Srv, 0), sb(3, DescriptorRangeType::Srv, 1),
        sb(4, DescriptorRangeType::Srv, 2), sb(5, DescriptorRangeType::Srv, 3),
        sb(6, DescriptorRangeType::Uav, 0), sb(7, DescriptorRangeType::Srv, 7, 1),
    };
    const ShaderBinding g1_bindings[] = {
        sb(0, DescriptorRangeType::Sampler, 0), sb(1, DescriptorRangeType::Sampler, 1),
        sb(2, DescriptorRangeType::Sampler, 4), sb(3, DescriptorRangeType::Sampler, 5),
    };
    ShaderBinding pc;
    pc.binding = 0;
    pc.root_constants = true;
    const ShaderBinding g2_bindings[] = {pc, sb(1, DescriptorRangeType::Cbv, 3), sb(2, DescriptorRangeType::Srv, 10)};
    const BindGroupPlan plan0 = BindGroupPlan::build(rs, g0_bindings, vf);
    const BindGroupPlan plan1 = BindGroupPlan::build(rs, g1_bindings, WebGpuStage::Fragment);
    const BindGroupPlan plan2 = BindGroupPlan::build(rs, g2_bindings, WebGpuStage::Fragment);
    CHECK(plan0.unresolved_count() == 0 && plan1.unresolved_count() == 0 && plan2.unresolved_count() == 0);

    RootArgumentState st;
    BindGroupCache cache;
    const BindPoint G = BindPoint::Graphics;
    int resolver_calls = 0;

    CHECK(cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls).status == Status::RootSignatureNotSet);
    DescriptorHeap* heaps[2] = {heap, smp};
    CHECK(st.set_descriptor_heaps(2, heaps) == Status::Ok);
    CHECK(st.set_root_signature(G, &rs) == Status::Ok);
    CHECK(cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls).status == Status::RootArgumentNotSet);

    CHECK(st.set_descriptor_table(G, 0, gpu_at(heap, 100)) == Status::Ok);
    CHECK(st.set_descriptor_table(G, 1, gpu_at(smp, 3)) == Status::Ok);
    const uint32_t k[4] = {0, 0, 0, 0};
    CHECK(st.set_root_constants(G, 2, 4, k, 0) == Status::Ok);
    CHECK(st.set_root_view(G, 3, RootParameterType::Cbv, 0x10100) == Status::Ok);
    CHECK(st.set_root_view(G, 4, RootParameterType::Srv, 0x100040) == Status::Ok);
    CHECK(st.set_descriptor_table(G, 5, gpu_at(heap, 200)) == Status::Ok);
    CHECK((st.dirty_parameters(G) & plan0.parameter_mask()) != 0);

    // Group 0.
    BindGroupLookup g0 = cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls);
    CHECK(g0.status == Status::Ok && g0.rebuilt && g0.bind_group != nullptr && g0.retired_gpu_id == 0);
    const std::vector<BindGroupEntry>& e = g0.bind_group->entries;
    CHECK(e.size() == 8);
    CHECK(e[0].binding == 0 && e[0].kind == BindResourceKind::Buffer && e[0].resource_id == 70);
    CHECK(e[0].offset == 0x200 && e[0].size == 256);
    CHECK(e[1].kind == BindResourceKind::Null);  // unresolvable CBV address
    CHECK(e[2].binding == 2 && e[2].kind == BindResourceKind::TextureView && e[2].resource_id == 500);
    CHECK(e[4].resource_id == 502);
    CHECK(e[5].binding == 5 && e[5].kind == BindResourceKind::Null);  // never written
    CHECK(e[6].binding == 6 && e[6].kind == BindResourceKind::Buffer && e[6].resource_id == 600);
    CHECK(e[6].offset == 64 && e[6].size == 160);
    CHECK(e[6].view.kind == DescriptorKind::Uav);
    CHECK(e[7].binding == 7 && e[7].kind == BindResourceKind::TextureView && e[7].resource_id == 800);
    CHECK(g0.bind_group->null_entries == 2);
    g0.bind_group->gpu_id = 1000;

    // Second lookup hits without touching the resolver.
    const int calls_after_build = resolver_calls;
    BindGroupLookup hit = cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls);
    CHECK(hit.status == Status::Ok && !hit.rebuilt && hit.bind_group == g0.bind_group);
    CHECK(hit.bind_group->gpu_id == 1000);
    CHECK(resolver_calls == calls_after_build);
    CHECK(cache.hits() == 1 && cache.misses() == 1);

    // Writes outside the descriptors the plan reads do not invalidate; identical rewrites neither.
    fill_srvs(reg, heap, 150, 4, 900);
    fill_srvs(reg, heap, 102, 3, 500);
    fill_srvs(reg, heap, 206, 1, 950);  // bindless neighbour of t7
    fill_srvs(reg, heap, 208, 1, 951);
    CHECK(!cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls).rebuilt);

    // A changed descriptor the plan reads forces a rebuild in place.
    fill_srvs(reg, heap, 105, 1, 777);
    BindGroupLookup rebuilt = cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls);
    CHECK(rebuilt.rebuilt && rebuilt.retired_gpu_id == 1000 && rebuilt.bind_group == g0.bind_group);
    CHECK(rebuilt.bind_group->gpu_id == 0);
    CHECK(rebuilt.bind_group->entries[5].kind == BindResourceKind::TextureView);
    CHECK(rebuilt.bind_group->entries[5].resource_id == 777);
    CHECK(rebuilt.bind_group->null_entries == 1);
    rebuilt.bind_group->gpu_id = 1001;
    fill_srvs(reg, heap, 207, 1, 801);  // bindless slot itself
    BindGroupLookup bindless = cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls);
    CHECK(bindless.rebuilt && bindless.retired_gpu_id == 1001);
    CHECK(bindless.bind_group->entries[7].resource_id == 801);
    bindless.bind_group->gpu_id = 1002;

    // Different table base / layout key -> different cache entries.
    CHECK(st.set_descriptor_table(G, 0, gpu_at(heap, 0)) == Status::Ok);
    BindGroupLookup moved = cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls);
    CHECK(moved.rebuilt && moved.bind_group != g0.bind_group && moved.retired_gpu_id == 0);
    CHECK(moved.bind_group->null_entries == 7);
    CHECK(st.set_descriptor_table(G, 0, gpu_at(heap, 100)) == Status::Ok);
    BindGroupLookup back = cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls);
    CHECK(!back.rebuilt && back.bind_group->gpu_id == 1002);
    CHECK(cache.get_or_build(st, G, plan0, 2, test_resolver, &resolver_calls).rebuilt);

    // Group 1: sampler table + static samplers.
    BindGroupLookup g1 = cache.get_or_build(st, G, plan1, 1, test_resolver, &resolver_calls);
    CHECK(g1.status == Status::Ok && g1.bind_group->entries.size() == 4);
    CHECK(g1.bind_group->entries[0].kind == BindResourceKind::Sampler);
    CHECK(g1.bind_group->entries[0].sampler.max_anisotropy == 16);
    CHECK(g1.bind_group->entries[2].binding == 2 && g1.bind_group->entries[2].kind == BindResourceKind::Sampler);
    CHECK(g1.bind_group->entries[2].sampler.address_u == 4);
    CHECK(g1.bind_group->entries[2].sampler.border_color[0] == 1.0f);  // opaque white
    CHECK(g1.bind_group->entries[2].sampler.border_color[3] == 1.0f);
    CHECK(g1.bind_group->entries[3].sampler.border_color[3] == 0.0f);  // transparent black
    SamplerDesc sd2;
    sd2.max_anisotropy = 2;
    CHECK(reg.create_sampler(sd2, cpu_at(smp, 4)) == Status::Ok);
    BindGroupLookup g1b = cache.get_or_build(st, G, plan1, 1, test_resolver, &resolver_calls);
    CHECK(g1b.rebuilt && g1b.bind_group->entries[1].sampler.max_anisotropy == 2);

    // Group 2: root constants, root CBV, root SRV.
    BindGroupLookup g2 = cache.get_or_build(st, G, plan2, 1, test_resolver, &resolver_calls);
    CHECK(g2.status == Status::Ok && g2.bind_group->entries.size() == 3);
    const std::vector<BindGroupEntry>& r = g2.bind_group->entries;
    CHECK(r[0].kind == BindResourceKind::RootConstants && r[0].size == 16 && r[0].root_parameter_index == 2);
    CHECK(r[1].kind == BindResourceKind::Buffer && r[1].resource_id == 70 && r[1].offset == 0x100);
    CHECK(r[1].size == 0x10000 - 0x100);  // clamped to the end of the buffer (< 64 KiB)
    CHECK(r[2].kind == BindResourceKind::Buffer && r[2].resource_id == 80 && r[2].offset == 0x40 && r[2].size == 0);
    // Root constant values are not part of the key.
    const uint32_t k2[4] = {5, 6, 7, 8};
    CHECK(st.set_root_constants(G, 2, 4, k2, 0) == Status::Ok);
    CHECK(!cache.get_or_build(st, G, plan2, 1, test_resolver, &resolver_calls).rebuilt);
    // Root view address changes are.
    CHECK(st.set_root_view(G, 3, RootParameterType::Cbv, 0x100000) == Status::Ok);
    BindGroupLookup g2b = cache.get_or_build(st, G, plan2, 1, test_resolver, &resolver_calls);
    CHECK(g2b.rebuilt && g2b.bind_group->entries[1].resource_id == 80);
    CHECK(g2b.bind_group->entries[1].size == kMaxUniformBindingSize);

    // Unresolved bindings compile to Null entries.
    const ShaderBinding missing[] = {sb(0, DescriptorRangeType::Srv, 0), sb(1, DescriptorRangeType::Uav, 5)};
    const BindGroupPlan partial = BindGroupPlan::build(rs, missing, vf);
    BindGroupLookup part = cache.get_or_build(st, G, partial, 1, test_resolver, &resolver_calls);
    CHECK(part.status == Status::Ok && part.bind_group->entries.size() == 2);
    CHECK(part.bind_group->entries[1].kind == BindResourceKind::Null && part.bind_group->null_entries == 1);

    // A plan built for another signature is rejected; an empty plan yields no bind group.
    RootSignature only_root;
    RootSignatureDesc od;
    od.parameters.push_back(root_view(RootParameterType::Cbv, 0));
    CHECK(RootSignature::build(od, &only_root).ok());
    CHECK(st.set_root_signature(BindPoint::Compute, &only_root) == Status::Ok);
    CHECK(cache.get_or_build(st, BindPoint::Compute, plan0, 1, test_resolver, nullptr).status ==
          Status::InvalidArgument);
    const BindGroupPlan none_plan =
        BindGroupPlan::build(only_root, std::span<const ShaderBinding>{}, WebGpuStage::Compute);
    BindGroupLookup none = cache.get_or_build(st, BindPoint::Compute, none_plan, 1, test_resolver, nullptr);
    CHECK(none.status == Status::Ok && none.bind_group == nullptr);

    // A CBV descriptor in an SRV range is a mismatch -> Null.
    CHECK(reg.create_constant_buffer_view(&cb, cpu_at(heap, 102)) == Status::Ok);
    BindGroupLookup mismatch = cache.get_or_build(st, G, plan0, 1, test_resolver, &resolver_calls);
    CHECK(mismatch.rebuilt && mismatch.bind_group->entries[2].kind == BindResourceKind::Null);

    g2b.bind_group->gpu_id = 2000;
    std::vector<uint32_t> retired;
    const size_t entries = cache.size();
    CHECK(entries >= 6);
    cache.clear(&retired);
    CHECK(cache.size() == 0);
    CHECK(retired.size() == 1 && retired[0] == 2000);
}

// ---------------------------------------------------------------------------
// 9. Resource state tracking
// ---------------------------------------------------------------------------

bool has_diag(const std::vector<StateDiagnostic>& diags, StateDiagnosticCode code, uint32_t id, uint32_t sub) {
    for (const StateDiagnostic& d : diags) {
        if (d.code == code && d.resource_id == id && d.subresource == sub) {
            return true;
        }
    }
    return false;
}

void test_state_tracker_basic() {
    ResourceStateRegistry reg;
    ResourceStateDesc tex;
    tex.subresource_count = 6;  // 3 mips x 2 slices
    tex.initial_state = ResourceStates::Common;
    CHECK(reg.register_resource(1, tex) == Status::Ok);
    CHECK(reg.register_resource(1, tex) == Status::ResourceAlreadyRegistered);
    CHECK(reg.register_resource(0, tex) == Status::InvalidArgument);
    ResourceStateDesc zero;
    zero.subresource_count = 0;
    CHECK(reg.register_resource(9, zero) == Status::InvalidArgument);
    CHECK(reg.contains(1) && !reg.contains(9));
    CHECK(reg.state(1, kAllSubresources) == ResourceStates::Common);
    CHECK(reg.state(1, 6) == kStateUnknown);
    CHECK(reg.state(77, 0) == kStateUnknown);

    CommandListStateTracker list(&reg);
    const ResourceBarrier to_rt = ResourceBarrier::transition(1, ResourceStates::Common, ResourceStates::RenderTarget);
    list.resource_barrier(1, &to_rt);
    CHECK(list.state(1, kAllSubresources) == ResourceStates::RenderTarget);
    CHECK(list.state(1, 3) == ResourceStates::RenderTarget);
    CHECK(list.expected_initial_state(1, 0) == ResourceStates::Common);
    CHECK(list.diagnostics().empty());
    CHECK(list.is_uniform(1));

    const ResourceBarrier to_srv =
        ResourceBarrier::transition(1, ResourceStates::RenderTarget, ResourceStates::PixelShaderResource);
    list.resource_barrier(1, &to_srv);
    CHECK(list.state(1, kAllSubresources) == ResourceStates::PixelShaderResource);

    // Before-state mismatch: reported, never fatal, state still follows the barrier.
    const ResourceBarrier wrong =
        ResourceBarrier::transition(1, ResourceStates::RenderTarget, ResourceStates::CopySource);
    list.resource_barrier(1, &wrong);
    CHECK(list.diagnostics().size() == 1);
    const StateDiagnostic& d = list.diagnostics()[0];
    CHECK(d.code == StateDiagnosticCode::BeforeStateMismatch && d.resource_id == 1);
    CHECK(d.expected_state == ResourceStates::PixelShaderResource && d.actual_state == ResourceStates::RenderTarget);
    CHECK(d.barrier_index == 0);
    CHECK(list.state(1, 0) == ResourceStates::CopySource);

    // Other diagnostics.
    const ResourceBarrier batch[6] = {
        ResourceBarrier::transition(1, ResourceStates::CopySource, ResourceStates::CopySource),
        ResourceBarrier::transition(1, ResourceStates::CopySource, ResourceStates::RenderTarget | ResourceStates::CopyDest),
        ResourceBarrier::transition(1, ResourceStates::Common, ResourceStates::CopyDest, 6),
        ResourceBarrier::transition(55, ResourceStates::Common, ResourceStates::CopyDest),
        ResourceBarrier::transition(1, 0, 0, kAllSubresources, 3),
        ResourceBarrier{static_cast<ResourceBarrierType>(9)},
    };
    list.resource_barrier(6, batch);
    const std::vector<StateDiagnostic>& diags = list.diagnostics();
    CHECK(has_diag(diags, StateDiagnosticCode::RedundantTransition, 1, kAllSubresources));
    CHECK(has_diag(diags, StateDiagnosticCode::InvalidStateCombination, 1, kAllSubresources));
    CHECK(has_diag(diags, StateDiagnosticCode::SubresourceOutOfRange, 1, 6));
    CHECK(has_diag(diags, StateDiagnosticCode::UnknownResource, 55, kAllSubresources));
    CHECK(has_diag(diags, StateDiagnosticCode::InvalidBarrierFlags, 1, kAllSubresources));
    bool saw_type = false;
    for (const StateDiagnostic& x : diags) {
        saw_type = saw_type || (x.code == StateDiagnosticCode::InvalidBarrierType && x.barrier_index == 5);
    }
    CHECK(saw_type);
    CHECK(std::strcmp(state_diagnostic_name(StateDiagnosticCode::BeforeStateMismatch), "BeforeStateMismatch") == 0);

    // UAV and aliasing barriers.
    const ResourceBarrier misc[3] = {ResourceBarrier::uav(1), ResourceBarrier::uav(0), ResourceBarrier::aliasing(0, 1)};
    list.resource_barrier(3, misc);
    CHECK(list.uav_barrier_count() == 2 && list.aliasing_barrier_count() == 1);

    list.reset();
    CHECK(list.diagnostics().empty() && list.state(1, 0) == kStateUnknown && list.uav_barrier_count() == 0);
    list.resource_barrier(1, nullptr);
}

void test_state_tracker_subresources_and_split() {
    ResourceStateRegistry reg;
    ResourceStateDesc tex;
    tex.subresource_count = 4;
    CHECK(reg.register_resource(2, tex) == Status::Ok);
    CommandListStateTracker list(&reg);

    // Per-subresource divergence.
    const ResourceBarrier all_srv =
        ResourceBarrier::transition(2, ResourceStates::Common, ResourceStates::PixelShaderResource);
    list.resource_barrier(1, &all_srv);
    const ResourceBarrier mip1_rt =
        ResourceBarrier::transition(2, ResourceStates::PixelShaderResource, ResourceStates::RenderTarget, 1);
    list.resource_barrier(1, &mip1_rt);
    CHECK(!list.is_uniform(2));
    CHECK(list.state(2, 0) == ResourceStates::PixelShaderResource);
    CHECK(list.state(2, 1) == ResourceStates::RenderTarget);
    CHECK(list.state(2, kAllSubresources) == kStateUnknown);
    CHECK(list.expected_initial_state(2, 1) == ResourceStates::Common);

    // An ALL transition validates each subresource individually.
    const ResourceBarrier all_copy =
        ResourceBarrier::transition(2, ResourceStates::PixelShaderResource, ResourceStates::CopySource);
    list.resource_barrier(1, &all_copy);
    CHECK(list.diagnostics().size() == 1);
    CHECK(has_diag(list.diagnostics(), StateDiagnosticCode::BeforeStateMismatch, 2, 1));
    // ...and collapses back to the uniform fast path once states agree.
    CHECK(list.is_uniform(2));
    CHECK(list.state(2, kAllSubresources) == ResourceStates::CopySource);

    // Per-subresource transitions that converge also collapse.
    const ResourceBarrier per_sub[4] = {
        ResourceBarrier::transition(2, ResourceStates::CopySource, ResourceStates::CopyDest, 0),
        ResourceBarrier::transition(2, ResourceStates::CopySource, ResourceStates::CopyDest, 1),
        ResourceBarrier::transition(2, ResourceStates::CopySource, ResourceStates::CopyDest, 2),
        ResourceBarrier::transition(2, ResourceStates::CopySource, ResourceStates::CopyDest, 3),
    };
    list.resource_barrier(3, per_sub);
    CHECK(!list.is_uniform(2));
    list.resource_barrier(1, &per_sub[3]);
    CHECK(list.is_uniform(2));
    CHECK(list.state(2, kAllSubresources) == ResourceStates::CopyDest);
    CHECK(list.diagnostics().size() == 1);

    // Split barriers.
    list.reset();
    const ResourceBarrier begin = ResourceBarrier::transition(
        2, ResourceStates::RenderTarget, ResourceStates::PixelShaderResource, 0, ResourceBarrierFlags::BeginOnly);
    const ResourceBarrier end = ResourceBarrier::transition(
        2, ResourceStates::RenderTarget, ResourceStates::PixelShaderResource, 0, ResourceBarrierFlags::EndOnly);
    list.resource_barrier(1, &begin);
    CHECK(list.in_split_barrier(2, 0));
    CHECK(!list.in_split_barrier(2, 1));
    CHECK(list.state(2, 0) == ResourceStates::RenderTarget);
    CHECK(list.expected_initial_state(2, 0) == ResourceStates::RenderTarget);
    list.note_access(2, 0, ResourceStates::PixelShaderResource);  // use during split
    CHECK(has_diag(list.diagnostics(), StateDiagnosticCode::UseDuringSplitBarrier, 2, 0));
    list.resource_barrier(1, &end);
    CHECK(!list.in_split_barrier(2, 0));
    CHECK(list.state(2, 0) == ResourceStates::PixelShaderResource);
    CHECK(list.diagnostics().size() == 1);

    // END without BEGIN on a known state, mismatched END, double BEGIN.
    list.resource_barrier(1, &end);
    CHECK(has_diag(list.diagnostics(), StateDiagnosticCode::SplitBarrierWithoutBegin, 2, 0));
    const ResourceBarrier begin2 = ResourceBarrier::transition(
        2, ResourceStates::PixelShaderResource, ResourceStates::CopySource, 0, ResourceBarrierFlags::BeginOnly);
    list.resource_barrier(1, &begin2);
    list.resource_barrier(1, &begin2);
    CHECK(has_diag(list.diagnostics(), StateDiagnosticCode::SplitBarrierAlreadyBegun, 2, 0));
    const ResourceBarrier wrong_end = ResourceBarrier::transition(
        2, ResourceStates::CopySource, ResourceStates::CopyDest, 0, ResourceBarrierFlags::EndOnly);
    list.resource_barrier(1, &wrong_end);
    CHECK(has_diag(list.diagnostics(), StateDiagnosticCode::SplitBarrierMismatch, 2, 0));
    CHECK(list.state(2, 0) == ResourceStates::CopyDest);

    // A regular transition while a split is in flight.
    list.reset();
    list.resource_barrier(1, &begin);
    const ResourceBarrier during = ResourceBarrier::transition(
        2, ResourceStates::PixelShaderResource, ResourceStates::CopySource, 0);
    list.resource_barrier(1, &during);
    CHECK(has_diag(list.diagnostics(), StateDiagnosticCode::UseDuringSplitBarrier, 2, 0));
    CHECK(list.state(2, 0) == ResourceStates::CopySource);
}

void test_state_tracker_execute() {
    ResourceStateRegistry reg;
    ResourceStateDesc tex;
    tex.subresource_count = 4;
    CHECK(reg.register_resource(3, tex) == Status::Ok);
    ResourceStateDesc buf;
    buf.implicit_promotion = true;
    CHECK(reg.register_resource(4, buf) == Status::Ok);
    ResourceStateDesc tex2;
    CHECK(reg.register_resource(5, tex2) == Status::Ok);

    // List A: texture 3 COMMON -> RT on mip 2 only, buffer used as COPY_DEST via promotion.
    CommandListStateTracker a(&reg);
    const ResourceBarrier rt2 = ResourceBarrier::transition(3, ResourceStates::Common, ResourceStates::RenderTarget, 2);
    a.resource_barrier(1, &rt2);
    a.note_access(4, kAllSubresources, ResourceStates::CopyDest);
    a.note_access(4, kAllSubresources, ResourceStates::CopyDest);
    CHECK(a.diagnostics().empty());
    std::vector<StateDiagnostic> diags;
    reg.execute(a, &diags);
    CHECK(diags.empty());
    CHECK(reg.state(3, 2) == ResourceStates::RenderTarget);
    CHECK(reg.state(3, 0) == ResourceStates::Common);
    CHECK(reg.state(3, kAllSubresources) == kStateUnknown);  // diverged
    CHECK(reg.state(4, 0) == ResourceStates::Common);         // buffer decayed

    // List B recorded independently expects mip 2 in PIXEL_SHADER_RESOURCE: mismatch at execute.
    CommandListStateTracker b(&reg);
    const ResourceBarrier wrong = ResourceBarrier::transition(
        3, ResourceStates::PixelShaderResource, ResourceStates::CopySource, 2);
    b.resource_barrier(1, &wrong);
    CHECK(b.diagnostics().empty());  // unknown locally: nothing to compare against yet
    CHECK(b.expected_initial_state(3, 2) == ResourceStates::PixelShaderResource);
    reg.execute(b, &diags);
    CHECK(diags.size() == 1);
    CHECK(diags[0].code == StateDiagnosticCode::ExecuteStateMismatch && diags[0].resource_id == 3);
    CHECK(diags[0].subresource == 2 && diags[0].expected_state == ResourceStates::PixelShaderResource);
    CHECK(diags[0].actual_state == ResourceStates::RenderTarget);
    CHECK(reg.state(3, 2) == ResourceStates::CopySource);  // committed anyway

    // List C brings everything back to COMMON: the global state collapses to uniform.
    CommandListStateTracker c(&reg);
    const ResourceBarrier back = ResourceBarrier::transition(3, ResourceStates::CopySource, ResourceStates::Common, 2);
    c.resource_barrier(1, &back);
    diags.clear();
    reg.execute(c, &diags);
    CHECK(diags.empty());
    CHECK(reg.state(3, kAllSubresources) == ResourceStates::Common);

    // Texture read access from COMMON is an implicit promotion (no diagnostic) and decays.
    CommandListStateTracker d(&reg);
    d.note_access(5, 0, ResourceStates::PixelShaderResource);
    reg.execute(d, &diags);
    CHECK(diags.empty());
    CHECK(reg.state(5, 0) == ResourceStates::Common);
    // ...but a render-target access from COMMON is not promotable.
    CommandListStateTracker e(&reg);
    e.note_access(5, 0, ResourceStates::RenderTarget);
    reg.execute(e, &diags);
    CHECK(diags.size() == 1 && diags[0].code == StateDiagnosticCode::ExecuteStateMismatch);
    diags.clear();
    // Access checks against known local state.
    CommandListStateTracker f(&reg);
    const ResourceBarrier to_rt = ResourceBarrier::transition(5, ResourceStates::RenderTarget, ResourceStates::RenderTarget);
    f.resource_barrier(1, &to_rt);
    f.note_access(5, 0, ResourceStates::PixelShaderResource);
    CHECK(has_diag(f.diagnostics(), StateDiagnosticCode::AccessStateMismatch, 5, 0));
    f.note_access(5, 3, ResourceStates::RenderTarget);
    CHECK(has_diag(f.diagnostics(), StateDiagnosticCode::SubresourceOutOfRange, 5, 3));

    // Split barrier begun in one list and ended in the next.
    CommandListStateTracker g(&reg);
    const ResourceBarrier begin = ResourceBarrier::transition(
        3, ResourceStates::Common, ResourceStates::CopyDest, kAllSubresources, ResourceBarrierFlags::BeginOnly);
    g.resource_barrier(1, &begin);
    reg.execute(g, &diags);
    CHECK(diags.empty());
    CHECK(reg.state(3, kAllSubresources) == ResourceStates::CopyDest);
    CommandListStateTracker h(&reg);
    const ResourceBarrier end = ResourceBarrier::transition(
        3, ResourceStates::Common, ResourceStates::CopyDest, kAllSubresources, ResourceBarrierFlags::EndOnly);
    h.resource_barrier(1, &end);
    CHECK(h.diagnostics().empty());
    reg.execute(h, &diags);
    CHECK(diags.empty());

    // Unregistered between record and execute.
    CommandListStateTracker i(&reg);
    i.note_access(5, 0, ResourceStates::Common);
    reg.unregister_resource(5);
    reg.execute(i, &diags);
    CHECK(diags.size() == 1 && diags[0].code == StateDiagnosticCode::UnknownResource);
    CHECK(!reg.contains(5));

    // Null registry: everything is unknown, nothing crashes.
    CommandListStateTracker orphan(nullptr);
    orphan.note_access(1, 0, ResourceStates::Common);
    CHECK(orphan.diagnostics().size() == 1);
}

void test_state_tracker_diagnostic_cap() {
    ResourceStateRegistry reg;
    ResourceStateDesc tex;
    CHECK(reg.register_resource(1, tex) == Status::Ok);
    CommandListStateTracker list(&reg);
    const ResourceBarrier bad = ResourceBarrier::transition(1, ResourceStates::CopyDest, ResourceStates::CopyDest);
    for (size_t i = 0; i < CommandListStateTracker::kMaxDiagnostics + 10; ++i) {
        list.resource_barrier(1, &bad);
    }
    CHECK(list.diagnostics().size() == CommandListStateTracker::kMaxDiagnostics);
    CHECK(list.dropped_diagnostics() == 10);
}

// ---------------------------------------------------------------------------
// 10. WebGPU read/write conflict query
// ---------------------------------------------------------------------------

void test_usage_scope_conflicts() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 8, false);
    DescriptorHeap* rtvs = make_heap(reg, DescriptorHeapType::Rtv, 4, false);
    DescriptorHeap* dsvs = make_heap(reg, DescriptorHeapType::Dsv, 4, false);

    ViewDesc srv_mip0 = texture2d_srv(0, 1);
    ViewDesc srv_mip1 = texture2d_srv(1, 1);
    ViewDesc srv_all = texture2d_srv(0, 0xffff);
    CHECK(reg.create_shader_resource_view(10, &srv_mip0, cpu_at(heap, 0)) == Status::Ok);
    CHECK(reg.create_shader_resource_view(10, &srv_mip1, cpu_at(heap, 1)) == Status::Ok);
    CHECK(reg.create_shader_resource_view(10, &srv_all, cpu_at(heap, 2)) == Status::Ok);
    ViewDesc rtv;
    rtv.dimension = ViewDimension::Texture2D;
    CHECK(reg.create_render_target_view(10, &rtv, cpu_at(rtvs, 0)) == Status::Ok);
    ViewDesc dsv;
    dsv.dimension = DsvDimension::Texture2D;
    dsv.flags = DsvFlags::ReadOnlyDepth;
    CHECK(reg.create_depth_stencil_view(20, &dsv, cpu_at(dsvs, 0)) == Status::Ok);
    dsv.flags = 0;
    CHECK(reg.create_depth_stencil_view(20, &dsv, cpu_at(dsvs, 1)) == Status::Ok);
    ViewDesc depth_srv = texture2d_srv();
    CHECK(reg.create_shader_resource_view(20, &depth_srv, cpu_at(heap, 3)) == Status::Ok);

    CHECK(view_usage(*heap->view(0)) == ResourceUsage::Sampled);
    CHECK(view_usage(*rtvs->view(0)) == ResourceUsage::RenderTarget);
    CHECK(view_usage(*dsvs->view(0)) == ResourceUsage::DepthStencilRead);
    CHECK(view_usage(*dsvs->view(1)) == ResourceUsage::DepthStencilWrite);
    const SubresourceRange r0 = view_subresource_range(*heap->view(1));
    CHECK(r0.first_mip == 1 && r0.mip_count == 1 && r0.first_slice == 0 && r0.slice_count == 1);
    CHECK(view_subresource_range(*heap->view(2)).mip_count == kAllMips);

    auto add_view = [](UsageScope& scope, const ViewDescriptor* v) {
        scope.add(v->resource_id, view_subresource_range(*v), view_usage(*v));
    };
    UsageScope scope;
    std::vector<UsageConflict> conflicts;

    // Sampling mip 1 while rendering to mip 0: allowed.
    add_view(scope, heap->view(1));
    add_view(scope, rtvs->view(0));
    CHECK(scope.find_conflicts(&conflicts) == 0);
    // Sampling the whole chain while rendering to mip 0: conflict.
    add_view(scope, heap->view(2));
    CHECK(scope.find_conflicts(&conflicts) == 1);
    CHECK(conflicts.size() == 1 && conflicts[0].resource_id == 10);
    CHECK(conflicts[0].first == ResourceUsage::RenderTarget && conflicts[0].second == ResourceUsage::Sampled);
    CHECK(conflicts[0].first_index == 1 && conflicts[0].second_index == 2);

    // Read-only depth + sampling the depth texture: allowed; writable depth: conflict.
    scope.clear();
    conflicts.clear();
    add_view(scope, dsvs->view(0));
    add_view(scope, heap->view(3));
    CHECK(scope.find_conflicts(&conflicts) == 0);
    add_view(scope, dsvs->view(1));
    CHECK(scope.find_conflicts(&conflicts) == 2);  // write vs read-only DSV, write vs SRV

    // Storage writes may alias each other; storage write + sampled may not.
    scope.clear();
    SubresourceRange all;
    scope.add(30, all, ResourceUsage::StorageWrite);
    scope.add(30, all, ResourceUsage::StorageWrite);
    scope.add(31, all, ResourceUsage::Sampled);
    CHECK(scope.find_conflicts(nullptr) == 0);
    scope.add(30, all, ResourceUsage::StorageRead);
    CHECK(scope.find_conflicts(nullptr) == 2);

    // Disjoint array slices never conflict.
    scope.clear();
    SubresourceRange slice0;
    slice0.first_slice = 0;
    slice0.slice_count = 1;
    SubresourceRange slice1 = slice0;
    slice1.first_slice = 1;
    scope.add(40, slice0, ResourceUsage::RenderTarget);
    scope.add(40, slice1, ResourceUsage::Sampled);
    CHECK(scope.find_conflicts(nullptr) == 0);
    scope.add(40, slice0, ResourceUsage::RenderTarget);  // same attachment twice
    CHECK(scope.find_conflicts(nullptr) == 1);
}

// ---------------------------------------------------------------------------
// 11. Threading contract smoke test
// ---------------------------------------------------------------------------

void test_concurrent_disjoint_writes() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 4096, true);
    constexpr uint32_t kThreads = 4;
    constexpr uint32_t kPerThread = 1024;
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&reg, heap, t] {
            ViewDesc srv = texture2d_srv();
            for (uint32_t i = 0; i < kPerThread; ++i) {
                const uint32_t index = t * kPerThread + i;
                reg.create_shader_resource_view(index + 1, &srv, cpu_at(heap, index));
            }
            // Create and destroy scratch heaps concurrently with resolution.
            for (int k = 0; k < 16; ++k) {
                DescriptorHeapDesc d;
                d.num_descriptors = 8;
                DescriptorHeap* scratch = nullptr;
                if (reg.create_heap(d, &scratch) == Status::Ok) {
                    reg.destroy_heap(scratch);
                }
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
    CHECK(heap->generation() == kThreads * kPerThread);
    bool all_ok = true;
    for (uint32_t i = 0; i < kThreads * kPerThread; ++i) {
        all_ok = all_ok && heap->view(i)->resource_id == i + 1;
    }
    CHECK(all_ok);
    CHECK(heap->range_changed_since(0, kThreads * kPerThread, 0));
    CHECK(!heap->range_changed_since(0, kThreads * kPerThread, heap->generation()));
    CHECK(reg.heap_count() == 1);
}

// ---------------------------------------------------------------------------
// 12. G77 hot-path stress tests: placed heaps, flat BindGroupCache, flat tracker index
// ---------------------------------------------------------------------------

void test_placed_heaps_sorted_lookup() {
    DescriptorHeapRegistry reg;
    // Create 12 placed heaps out-of-order (covers both linear scan <= 8 and binary search > 8),
    // mixing CbvSrvUav (stride 32) and Sampler (stride 16) heaps.
    constexpr uint32_t kPlacedCount = 12;
    constexpr uint64_t kBaseAddr = 0x10000000ull;
    constexpr uint64_t kSlotSpacing = 0x10000ull;
    std::vector<DescriptorHeap*> heaps(kPlacedCount, nullptr);
    const uint32_t order[kPlacedCount] = {7, 2, 11, 0, 5, 9, 3, 1, 10, 6, 4, 8};

    for (uint32_t idx : order) {
        DescriptorHeapDesc d;
        d.type = (idx % 2 == 0) ? DescriptorHeapType::CbvSrvUav : DescriptorHeapType::Sampler;
        d.num_descriptors = 64;
        d.flags = DescriptorHeapFlags::ShaderVisible;
        d.cpu_address = kBaseAddr + static_cast<uint64_t>(idx) * kSlotSpacing;
        CHECK(reg.create_heap(d, &heaps[idx]) == Status::Ok);
        CHECK(heaps[idx] != nullptr);
    }

    // Overlap checks: exact duplicate, partial tail overlap, partial head overlap, enclosing.
    {
        DescriptorHeap* bad = nullptr;
        DescriptorHeapDesc ov;
        ov.type = DescriptorHeapType::CbvSrvUav;
        ov.num_descriptors = 16;
        ov.cpu_address = kBaseAddr + 4 * kSlotSpacing;
        CHECK(reg.create_heap(ov, &bad) == Status::InvalidArgument && bad == nullptr);
        ov.cpu_address = kBaseAddr + 4 * kSlotSpacing + 32;
        CHECK(reg.create_heap(ov, &bad) == Status::InvalidArgument && bad == nullptr);
        ov.cpu_address = kBaseAddr + 4 * kSlotSpacing - 32;
        CHECK(reg.create_heap(ov, &bad) == Status::InvalidArgument && bad == nullptr);
        // Unaligned cpu_address (< 8 bytes).
        ov.cpu_address = kBaseAddr + 20 * kSlotSpacing + 4;
        CHECK(reg.create_heap(ov, &bad) == Status::InvalidArgument && bad == nullptr);
    }

    // Resolve valid slots, misaligned slots, and gap/out-of-range handles across all 12 heaps.
    for (uint32_t i = 0; i < kPlacedCount; ++i) {
        const uint32_t inc = heaps[i]->increment();
        for (uint32_t s : {0u, 1u, 31u, 63u}) {
            ResolvedDescriptor r;
            const CpuDescriptorHandle h = heaps[i]->cpu_handle(s);
            CHECK(h.ptr == kBaseAddr + static_cast<uint64_t>(i) * kSlotSpacing + static_cast<uint64_t>(s) * inc);
            CHECK(reg.resolve_cpu(h, &r) == Status::Ok);
            CHECK(r.heap == heaps[i] && r.index == s);
        }
        ResolvedDescriptor r;
        // Misaligned inside heap.
        CHECK(reg.resolve_cpu(CpuDescriptorHandle{heaps[i]->cpu_start().ptr + 8}, &r) == Status::MisalignedHandle);
        // Past end (in gap between heaps).
        CHECK(reg.resolve_cpu(heaps[i]->cpu_handle(64), &r) == Status::InvalidHandle);
        CHECK(reg.resolve_cpu(CpuDescriptorHandle{heaps[i]->cpu_handle(64).ptr + 8}, &r) == Status::InvalidHandle);
    }
    // Below lowest placed heap.
    {
        ResolvedDescriptor r;
        CHECK(reg.resolve_cpu(CpuDescriptorHandle{kBaseAddr - 32}, &r) == Status::InvalidHandle);
    }

    // Destroy first, middle, and last heaps; verify they no longer resolve while neighbors still do.
    for (uint32_t kill : {0u, 5u, 11u}) {
        const CpuDescriptorHandle old_h = heaps[kill]->cpu_handle(3);
        reg.destroy_heap(heaps[kill]);
        heaps[kill] = nullptr;
        ResolvedDescriptor r;
        CHECK(reg.resolve_cpu(old_h, &r) == Status::InvalidHandle);
    }
    // Now 9 heaps remain (>8); destroy 2 more to drop to 7 (<=8) and check linear path too.
    for (uint32_t kill : {2u, 8u}) {
        reg.destroy_heap(heaps[kill]);
        heaps[kill] = nullptr;
    }
    for (uint32_t i = 0; i < kPlacedCount; ++i) {
        if (heaps[i] == nullptr) {
            continue;
        }
        ResolvedDescriptor r;
        CHECK(reg.resolve_cpu(heaps[i]->cpu_handle(10), &r) == Status::Ok);
        CHECK(r.heap == heaps[i] && r.index == 10);
    }

    // Re-insert into a previously destroyed range (index 5).
    {
        DescriptorHeapDesc d;
        d.type = DescriptorHeapType::CbvSrvUav;
        d.num_descriptors = 32;
        d.cpu_address = kBaseAddr + 5 * kSlotSpacing;
        CHECK(reg.create_heap(d, &heaps[5]) == Status::Ok && heaps[5] != nullptr);
        ResolvedDescriptor r;
        CHECK(reg.resolve_cpu(heaps[5]->cpu_handle(31), &r) == Status::Ok);
        CHECK(r.heap == heaps[5] && r.index == 31);
    }
}

void test_bind_group_cache_flat_mru_erase_and_scale() {
    const DescriptorLiveCounts base_counts = descriptor_live_counts();

    DescriptorHeapRegistry reg;
    constexpr uint32_t kSlots = 1500;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, kSlots, true);
    for (uint32_t i = 0; i < kSlots; ++i) {
        ViewDesc srv = texture2d_srv();
        CHECK(reg.create_shader_resource_view(1000u + i, &srv, cpu_at(heap, i)) == Status::Ok);
    }

    RootSignatureDesc rs_desc;
    RootParameterDesc p0;
    p0.parameter_type = RootParameterType::DescriptorTable;
    DescriptorRangeDesc rng;
    rng.range_type = DescriptorRangeType::Srv;
    rng.num_descriptors = 1;
    rng.base_shader_register = 0;
    p0.ranges.push_back(rng);
    rs_desc.parameters.push_back(p0);
    RootSignature rs;
    CHECK(RootSignature::build(rs_desc, &rs).ok());

    const ShaderBinding bindings[1] = {{0, DescriptorRangeType::Srv, 0, 0, false}};
    const BindGroupPlan plan = BindGroupPlan::build(rs, bindings, WebGpuStage::Fragment);
    CHECK(!plan.has_heap_indexed());

    RootArgumentState st;
    DescriptorHeap* bound_heaps[1] = {heap};
    CHECK(st.set_descriptor_heaps(1, bound_heaps) == Status::Ok);
    CHECK(st.set_root_signature(BindPoint::Graphics, &rs) == Status::Ok);

    BindGroupCache cache;
    constexpr uint32_t kCount = 1200;
    std::vector<CachedBindGroup*> ptrs(kCount, nullptr);

    // Insert > 1,000 bind groups to force multiple flat-table growth/rehash passes,
    // verifying pointer stability of the first entry across all rehashes.
    for (uint32_t i = 0; i < kCount; ++i) {
        CHECK(st.set_descriptor_table(BindPoint::Graphics, 0, gpu_at(heap, i)) == Status::Ok);
        BindGroupLookup lk = cache.get_or_build(st, BindPoint::Graphics, plan, 77, test_resolver, nullptr);
        CHECK(lk.status == Status::Ok && lk.rebuilt && lk.bind_group != nullptr);
        lk.bind_group->gpu_id = 10000u + i;
        lk.bind_group->view_ids.push_back(50000u + i);
        cache.note_group_ids(*lk.bind_group);  // ids added after the build are noted (as the bridge does)
        ptrs[i] = lk.bind_group;

        // Immediate repeat lookup must hit the 1-entry MRU cache with identical pointer.
        const uint64_t hits_before = cache.hits();
        BindGroupLookup mru = cache.get_or_build(st, BindPoint::Graphics, plan, 77, test_resolver, nullptr);
        CHECK(mru.status == Status::Ok && !mru.rebuilt && mru.bind_group == ptrs[i]);
        CHECK(cache.hits() == hits_before + 1);
    }
    CHECK(cache.size() == kCount);
    CHECK(descriptor_live_counts().cached_bind_groups == base_counts.cached_bind_groups + kCount);

    // Every earlier entry must still be found at the exact same pointer after all rehashes.
    for (uint32_t i = 0; i < kCount; i += 37) {
        CHECK(st.set_descriptor_table(BindPoint::Graphics, 0, gpu_at(heap, i)) == Status::Ok);
        BindGroupLookup lk = cache.get_or_build(st, BindPoint::Graphics, plan, 77, test_resolver, nullptr);
        CHECK(lk.status == Status::Ok && !lk.rebuilt && lk.bind_group == ptrs[i]);
        CHECK(lk.bind_group->gpu_id == 10000u + i);
    }

    // O(1) erase: null pointer, foreign/missing pointer, and valid pointer (including MRU entry).
    CHECK(cache.erase(nullptr) == 0);
    CachedBindGroup foreign;
    foreign.key_hash = ptrs[0]->key_hash;  // same hash as a cached entry, different pointer
    CHECK(cache.erase(&foreign) == 0);
    foreign.key_hash = 0xdeadbeefcafebabeull;
    CHECK(cache.erase(&foreign) == 0);

    // Make ptrs[10] the MRU entry, then erase it; next lookup must miss and rebuild.
    CHECK(st.set_descriptor_table(BindPoint::Graphics, 0, gpu_at(heap, 10)) == Status::Ok);
    CHECK(cache.get_or_build(st, BindPoint::Graphics, plan, 77, test_resolver, nullptr).bind_group == ptrs[10]);
    CHECK(cache.erase(ptrs[10]) == 10010u);
    CHECK(cache.size() == kCount - 1);
    BindGroupLookup rebuilt10 = cache.get_or_build(st, BindPoint::Graphics, plan, 77, test_resolver, nullptr);
    CHECK(rebuilt10.status == Status::Ok && rebuilt10.rebuilt && rebuilt10.bind_group != nullptr);
    rebuilt10.bind_group->gpu_id = 20010u;
    ptrs[10] = rebuilt10.bind_group;
    CHECK(cache.size() == kCount);

    // Evict a batch by resource_id and view_id.
    std::vector<uint32_t> evict_res;
    std::vector<uint32_t> evict_views;
    for (uint32_t i = 0; i < 300; ++i) {
        evict_res.push_back(1000u + i);
    }
    for (uint32_t i = 300; i < 600; ++i) {
        evict_views.push_back(50000u + i);
    }
    std::vector<uint32_t> retired;
    std::vector<uint64_t> evicted_hashes;
    const size_t evicted = cache.evict_resources_and_views(evict_res, evict_views, &retired, &evicted_hashes);
    CHECK(evicted == 600);
    CHECK(retired.size() == 600 && evicted_hashes.size() == 600);
    CHECK(cache.size() == kCount - 600);

    // Start new epoch, touch 200 surviving entries, then end_epoch(250) to evict untouched entries
    // and trigger tombstone compaction.
    retired.clear();
    CHECK(cache.end_epoch(kCount, &retired) == 0 && retired.empty());  // advances epoch without dropping
    for (uint32_t i = 600; i < 800; ++i) {
        CHECK(st.set_descriptor_table(BindPoint::Graphics, 0, gpu_at(heap, i)) == Status::Ok);
        BindGroupLookup lk = cache.get_or_build(st, BindPoint::Graphics, plan, 77, test_resolver, nullptr);
        CHECK(lk.status == Status::Ok && !lk.rebuilt && lk.bind_group == ptrs[i]);
    }
    const size_t epoch_dropped = cache.end_epoch(250, &retired);
    CHECK(epoch_dropped == 400);
    CHECK(retired.size() == 400);
    CHECK(cache.size() == 200);

    // Surviving 200 entries must still hit cleanly after compaction/rehash.
    for (uint32_t i = 600; i < 800; ++i) {
        CHECK(st.set_descriptor_table(BindPoint::Graphics, 0, gpu_at(heap, i)) == Status::Ok);
        BindGroupLookup lk = cache.get_or_build(st, BindPoint::Graphics, plan, 77, test_resolver, nullptr);
        CHECK(lk.status == Status::Ok && !lk.rebuilt && lk.bind_group == ptrs[i]);
    }

    // Clear and verify live gauge returns to baseline.
    retired.clear();
    cache.clear(&retired);
    CHECK(retired.size() == 200 && cache.size() == 0);
    CHECK(descriptor_live_counts().cached_bind_groups == base_counts.cached_bind_groups);
}

void test_state_tracker_flat_index_and_reset() {
    ResourceStateRegistry reg;
    constexpr uint32_t kResourceCount = 600;
    for (uint32_t i = 1; i <= kResourceCount; ++i) {
        ResourceStateDesc d;
        d.subresource_count = (i % 10 == 0) ? 4u : 1u;
        d.initial_state = ResourceStates::Common;
        d.implicit_promotion = (i % 3 == 0);
        CHECK(reg.register_resource(i, d) == Status::Ok);
    }

    CommandListStateTracker tracker(&reg);
    for (int frame = 0; frame < 4; ++frame) {
        const uint32_t from = (frame % 2 == 0) ? ResourceStates::Common : ResourceStates::RenderTarget;
        const uint32_t to = (frame % 2 == 0) ? ResourceStates::RenderTarget : ResourceStates::Common;
        for (uint32_t i = 1; i <= kResourceCount; ++i) {
            const uint32_t expected_from = (i % 3 == 0) ? ResourceStates::Common : from;
            if (expected_from != to) {
                const ResourceBarrier b = ResourceBarrier::transition(i, expected_from, to);
                tracker.resource_barrier(1, &b);
            } else {
                tracker.note_access(i, 0, ResourceStates::PixelShaderResource);
            }
        }
        CHECK(tracker.diagnostics().empty());
        for (uint32_t i = 1; i <= kResourceCount; ++i) {
            const uint32_t expected_from = (i % 3 == 0) ? ResourceStates::Common : from;
            if (expected_from != to) {
                CHECK(tracker.state(i, 0) == to);
                CHECK(tracker.expected_initial_state(i, 0) == expected_from);
            } else {
                CHECK(tracker.state(i, 0) == ResourceStates::PixelShaderResource);
            }
        }
        std::vector<StateDiagnostic> exec_diags;
        reg.execute(tracker, &exec_diags);
        CHECK(exec_diags.empty());

        tracker.reset();
        CHECK(tracker.diagnostics().empty());
        CHECK(tracker.state(1, 0) == kStateUnknown);
        CHECK(tracker.state(kResourceCount, 0) == kStateUnknown);
        CHECK(tracker.expected_initial_state(kResourceCount / 2, 0) == kStateUnknown);
    }
}

// ---------------------------------------------------------------------------
// 29. Hardening regression tests (heap0 serial 0, TableWindow saturation,
//     handle overflow, CBV alignment, slot source switch, batched copy stamps)
// ---------------------------------------------------------------------------

void test_heap0_serial_zero_vs_unbound_heap_indexed() {
    DescriptorHeapRegistry reg;
    // First heap created in a fresh registry has serial() == 0.
    DescriptorHeap* heap0 = make_heap(reg, DescriptorHeapType::CbvSrvUav, 16, true);
    CHECK(heap0->serial() == 0u);
    const ViewDesc srv = texture2d_srv(0, 1);
    CHECK(reg.create_shader_resource_view(777, &srv, cpu_at(heap0, 3)) == Status::Ok);

    RootSignatureDesc rsd;
    RootParameterDesc rc;
    rc.parameter_type = RootParameterType::Constants32Bit;
    rc.shader_register = 0;
    rc.num_32bit_values = 2;
    rsd.parameters.push_back(rc);
    rsd.flags = RootSignatureFlags::CbvSrvUavHeapDirectlyIndexed;
    RootSignature rs;
    CHECK(RootSignature::build(rsd, &rs).ok());

    ShaderBinding b;
    b.binding = 0;
    b.register_class = DescriptorRangeType::Srv;
    b.heap = HeapAccess::Resource;
    b.heap_index_parameter = 0;
    b.heap_index_dword = 0;
    b.heap_shift = 5;  // * 32 bytes
    const BindGroupPlan plan = BindGroupPlan::build(rs, std::span<const ShaderBinding>(&b, 1), WebGpuStage::Fragment);

    RootArgumentState state;
    CHECK(state.set_root_signature(BindPoint::Graphics, &rs) == Status::Ok);
    CHECK(state.set_root_constant(BindPoint::Graphics, 0, 3, 0) == Status::Ok);

    BindGroupCache cache;
    // 1) Draw before binding heap0: caches Null entry.
    BindGroupLookup l0 = cache.get_or_build(state, BindPoint::Graphics, plan, 1, test_resolver, nullptr);
    CHECK(l0.status == Status::Ok && l0.rebuilt && l0.bind_group != nullptr);
    CHECK(l0.bind_group->entries.size() == 1 && l0.bind_group->entries[0].kind == BindResourceKind::Null);
    l0.bind_group->gpu_id = 101;

    // 2) Bind heap0 (serial == 0): MUST NOT collide with unbound key and MUST resolve resource 777.
    DescriptorHeap* heaps[1] = {heap0};
    CHECK(state.set_descriptor_heaps(1, heaps) == Status::Ok);
    BindGroupLookup l1 = cache.get_or_build(state, BindPoint::Graphics, plan, 1, test_resolver, nullptr);
    CHECK(l1.status == Status::Ok && l1.rebuilt && l1.bind_group != nullptr);
    CHECK(l1.bind_group != l0.bind_group);
    CHECK(l1.bind_group->entries[0].kind == BindResourceKind::TextureView);
    CHECK(l1.bind_group->entries[0].resource_id == 777);
    l1.bind_group->gpu_id = 102;

    // 3) Repeat lookup hits cache.
    BindGroupLookup l2 = cache.get_or_build(state, BindPoint::Graphics, plan, 1, test_resolver, nullptr);
    CHECK(l2.status == Status::Ok && !l2.rebuilt && l2.bind_group == l1.bind_group);

    // 4) Unbind heap0: MUST NOT return stale heap0 bind group.
    CHECK(state.set_descriptor_heaps(0, nullptr) == Status::Ok);
    BindGroupLookup l3 = cache.get_or_build(state, BindPoint::Graphics, plan, 1, test_resolver, nullptr);
    CHECK(l3.status == Status::Ok && !l3.rebuilt && l3.bind_group == l0.bind_group);
    CHECK(l3.bind_group->entries[0].kind == BindResourceKind::Null);
}

void test_table_window_u32_span_saturation() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 8, true);
    const ViewDesc srv = texture2d_srv(0, 1);
    CHECK(reg.create_shader_resource_view(10, &srv, cpu_at(heap, 0)) == Status::Ok);

    RootSignatureDesc rsd;
    RootParameterDesc tbl;
    tbl.parameter_type = RootParameterType::DescriptorTable;
    tbl.ranges.push_back(DescriptorRangeDesc{DescriptorRangeType::Srv, kUnboundedDescriptorCount, 0, 0});
    rsd.parameters.push_back(tbl);
    RootSignature rs;
    CHECK(RootSignature::build(rsd, &rs).ok());

    // Direct table slots at 0 and 0xffffffffu would wrap (hi - lo + 1) to 0 without saturation.
    ShaderBinding bindings[2];
    bindings[0].binding = 0;
    bindings[0].register_class = DescriptorRangeType::Srv;
    bindings[0].table_parameter = 0;
    bindings[0].table_slot = 0;
    bindings[1].binding = 1;
    bindings[1].register_class = DescriptorRangeType::Srv;
    bindings[1].table_parameter = 0;
    bindings[1].table_slot = 0xffffffffu;
    const BindGroupPlan plan = BindGroupPlan::build(rs, bindings, WebGpuStage::Fragment);
    CHECK(plan.table_windows().size() == 1);
    CHECK(plan.table_windows()[0].first_offset == 0);
    CHECK(plan.table_windows()[0].count == 0xffffffffu);

    RootArgumentState state;
    DescriptorHeap* heaps[1] = {heap};
    CHECK(state.set_descriptor_heaps(1, heaps) == Status::Ok);
    CHECK(state.set_root_signature(BindPoint::Graphics, &rs) == Status::Ok);
    CHECK(state.set_descriptor_table(BindPoint::Graphics, 0, heap->gpu_start()) == Status::Ok);

    BindGroupCache cache;
    BindGroupLookup a = cache.get_or_build(state, BindPoint::Graphics, plan, 7, test_resolver, nullptr);
    CHECK(a.status == Status::Ok && a.rebuilt && a.bind_group->entries[0].resource_id == 10);
    a.bind_group->gpu_id = 55;

    // Mutate slot 0: saturated window MUST detect invalidation instead of skipping with count == 0.
    CHECK(reg.create_shader_resource_view(11, &srv, cpu_at(heap, 0)) == Status::Ok);
    BindGroupLookup b = cache.get_or_build(state, BindPoint::Graphics, plan, 7, test_resolver, nullptr);
    CHECK(b.status == Status::Ok && b.rebuilt && b.retired_gpu_id == 55);
    CHECK(b.bind_group->entries[0].resource_id == 11);
}

void test_handle_overflow_guard_and_cbv_alignment_guard() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* heap0 = make_heap(reg, DescriptorHeapType::CbvSrvUav, 16, true);
    DescriptorHeap* heap1 = make_heap(reg, DescriptorHeapType::CbvSrvUav, 16, true);
    DescriptorHeap* samp0 = make_heap(reg, DescriptorHeapType::Sampler, 16, true);
    (void)heap1;

    // 32-byte increment: index (1 << 27) overflows 32-bit offset and must NOT alias heap1!
    const CpuDescriptorHandle bad_cpu = heap0->cpu_handle(1u << 27);
    const GpuDescriptorHandle bad_gpu = heap0->gpu_handle(1u << 27);
    CHECK(bad_cpu.ptr == 0);
    CHECK(bad_gpu.ptr == 0);
    ResolvedDescriptor r;
    CHECK(reg.resolve_cpu(bad_cpu, &r) == Status::InvalidHandle);
    CHECK(reg.resolve_gpu(bad_gpu, &r) == Status::InvalidHandle);

    // 16-byte sampler increment: index (1 << 28) overflows 32-bit offset.
    CHECK(samp0->cpu_handle(1u << 28).ptr == 0);
    CHECK(samp0->gpu_handle(1u << 28).ptr == 0);
    // Out-of-capacity but non-overflowing index still yields HandleOutOfRange.
    CHECK(reg.resolve_cpu(samp0->cpu_handle(16), &r) == Status::HandleOutOfRange);

    // Invalid CBV alignments (0 or non-power-of-two) are ignored instead of causing div-by-zero.
    reg.set_cbv_alignment(0);
    reg.set_cbv_alignment(48);
    ConstantBufferViewDesc cbv{0x10100, 256};
    CHECK(reg.create_constant_buffer_view(&cbv, cpu_at(heap0, 0)) == Status::Ok);
    reg.set_cbv_alignment(16);
    ConstantBufferViewDesc cbv16{0x10010, 32};
    CHECK(reg.create_constant_buffer_view(&cbv16, cpu_at(heap0, 1)) == Status::Ok);
    ConstantBufferViewDesc cbv_bad{0x10008, 32};
    CHECK(reg.create_constant_buffer_view(&cbv_bad, cpu_at(heap0, 2)) == Status::InvalidArgument);
}

void test_slot_source_switch_and_rebuild_clears_metadata() {
    class EmptySlotSource final : public DescriptorSlotSource {
    public:
        bool reads(const DescriptorHeap&) const override { return true; }
        bool snapshot(const DescriptorHeap&, uint32_t, uint32_t count, std::vector<uint8_t>* out) const override {
            out->assign(static_cast<size_t>(count) * 4u, 0x5Au);
            return true;
        }
        ViewDescriptor view(const DescriptorHeap&, uint32_t, DescriptorRangeType) const override {
            ViewDescriptor v{};
            v.kind = DescriptorKind::Srv;
            v.dimension = ViewDimension::Texture2D;
            v.resource_id = 9001;
            return v;
        }
        bool sampler(const DescriptorHeap&, uint32_t, SamplerDesc*) const override { return false; }
    };

    DescriptorHeapRegistry reg;
    DescriptorHeap* heap = make_heap(reg, DescriptorHeapType::CbvSrvUav, 8, true);
    const ViewDesc srv = texture2d_srv(0, 1);
    CHECK(reg.create_shader_resource_view(42, &srv, cpu_at(heap, 0)) == Status::Ok);

    RootSignatureDesc rsd;
    RootParameterDesc rc;
    rc.parameter_type = RootParameterType::Constants32Bit;
    rc.shader_register = 0;
    rc.num_32bit_values = 1;
    rsd.parameters.push_back(rc);
    RootSignature rs;
    CHECK(RootSignature::build(rsd, &rs).ok());

    ShaderBinding b;
    b.binding = 0;
    b.register_class = DescriptorRangeType::Srv;
    b.heap = HeapAccess::Resource;
    b.heap_index_parameter = 0;
    b.heap_index_dword = 0;
    b.heap_shift = 5;
    const BindGroupPlan plan = BindGroupPlan::build(rs, std::span<const ShaderBinding>(&b, 1), WebGpuStage::Fragment);

    RootArgumentState state;
    DescriptorHeap* heaps[1] = {heap};
    CHECK(state.set_descriptor_heaps(1, heaps) == Status::Ok);
    CHECK(state.set_root_signature(BindPoint::Graphics, &rs) == Status::Ok);
    CHECK(state.set_root_constant(BindPoint::Graphics, 0, 0, 0) == Status::Ok);

    BindGroupCache cache;
    BindGroupLookup l0 = cache.get_or_build(state, BindPoint::Graphics, plan, 1, test_resolver, nullptr);
    CHECK(l0.status == Status::Ok && l0.rebuilt && l0.bind_group->entries[0].resource_id == 42);
    l0.bind_group->gpu_id = 301;
    l0.bind_group->null_write_entries = 2;
    l0.bind_group->view_ids.push_back(888);
    l0.bind_group->relocations.push_back(CachedBindGroup::Relocation{0, false, 0x1000, 64, 64, 77, 0});
    l0.bind_group->relocation_epoch = 5;
    l0.bind_group->retire_epoch = 6;
    l0.bind_group->residency_epoch = 7;
    l0.bind_group->residency_key = 8;
    cache.note_group_ids(*l0.bind_group);

    // Switching slot_source_ MUST invalidate cached entry, retire gpu_id, and clear stale metadata.
    EmptySlotSource src;
    cache.set_slot_source(&src);
    BindGroupLookup l1 = cache.get_or_build(state, BindPoint::Graphics, plan, 1, test_resolver, nullptr);
    CHECK(l1.status == Status::Ok && l1.rebuilt && l1.retired_gpu_id == 301);
    CHECK(l1.bind_group->entries[0].resource_id == 9001);
    CHECK(l1.bind_group->null_write_entries == 0);
    CHECK(l1.bind_group->view_ids.empty());
    CHECK(l1.bind_group->relocations.empty());
    CHECK(l1.bind_group->relocation_epoch == 0 && l1.bind_group->retire_epoch == 0);
    CHECK(l1.bind_group->residency_epoch == 0 && l1.bind_group->residency_key == 0);

    // Destroying the heap and calling refresh_table_entries MUST reset HeapIndexed entry to Null.
    reg.destroy_heap(heap);
    CHECK(refresh_table_entries(reg, src, plan, test_resolver, nullptr, l1.bind_group, true));
    CHECK(l1.bind_group->entries[0].kind == BindResourceKind::Null);
    CHECK(l1.bind_group->null_entries == 1);
}

void test_copy_descriptors_batched_gen_and_overlap_fastpath() {
    DescriptorHeapRegistry reg;
    DescriptorHeap* src_cbv = make_heap(reg, DescriptorHeapType::CbvSrvUav, 256, false);
    DescriptorHeap* dst_cbv = make_heap(reg, DescriptorHeapType::CbvSrvUav, 256, true);
    DescriptorHeap* src_smp = make_heap(reg, DescriptorHeapType::Sampler, 64, false);
    DescriptorHeap* dst_smp = make_heap(reg, DescriptorHeapType::Sampler, 64, true);

    fill_srvs(reg, src_cbv, 0, 200, 1000);
    for (uint32_t i = 0; i < 32; ++i) {
        SamplerDesc s;
        s.max_anisotropy = 1 + (i % 16);
        s.mip_lod_bias = static_cast<float>(i) * 0.25f;
        CHECK(reg.create_sampler(s, cpu_at(src_smp, i)) == Status::Ok);
    }

    // Batched copy_descriptors_simple bumps destination generation at most ONCE across 150 slots.
    const uint64_t gen0 = dst_cbv->generation();
    CHECK(reg.copy_descriptors_simple(150, cpu_at(dst_cbv, 10), cpu_at(src_cbv, 20), DescriptorHeapType::CbvSrvUav) ==
          Status::Ok);
    CHECK(dst_cbv->generation() == gen0 + 1);
    for (uint32_t i = 0; i < 150; ++i) {
        CHECK(dst_cbv->view(10 + i)->resource_id == 1020 + i);
    }
    CHECK(dst_cbv->range_changed_since(10, 150, gen0));
    CHECK(!dst_cbv->range_changed_since(0, 10, gen0));
    CHECK(!dst_cbv->range_changed_since(160, 96, gen0));
    // Re-copying identical content must NOT bump generation.
    CHECK(reg.copy_descriptors_simple(150, cpu_at(dst_cbv, 10), cpu_at(src_cbv, 20), DescriptorHeapType::CbvSrvUav) ==
          Status::Ok);
    CHECK(dst_cbv->generation() == gen0 + 1);

    // 16-byte sampler heap copy_descriptors with zero-count ranges and disjoint heaps fast path.
    const uint64_t sgen0 = dst_smp->generation();
    CpuDescriptorHandle dst_starts[3] = {cpu_at(dst_smp, 4), CpuDescriptorHandle{0}, cpu_at(dst_smp, 20)};
    uint32_t dst_sizes[3] = {8, 0, 8};
    CpuDescriptorHandle src_starts[2] = {cpu_at(src_smp, 2), cpu_at(src_smp, 10)};
    uint32_t src_sizes[2] = {10, 6};
    CHECK(reg.copy_descriptors(3, dst_starts, dst_sizes, 2, src_starts, src_sizes, DescriptorHeapType::Sampler) ==
          Status::Ok);
    CHECK(dst_smp->range_changed_since(4, 8, sgen0));
    CHECK(dst_smp->range_changed_since(20, 8, sgen0));
    CHECK(dst_smp->sampler(4)->desc.max_anisotropy == src_smp->sampler(2)->desc.max_anisotropy);
    CHECK(dst_smp->sampler(27)->desc.max_anisotropy == src_smp->sampler(15)->desc.max_anisotropy);

    // Same-heap overlapping multi-range copy on 16-byte sampler heap stages cleanly.
    CpuDescriptorHandle ov_dst[2] = {cpu_at(dst_smp, 6), cpu_at(dst_smp, 10)};
    uint32_t ov_sizes[2] = {4, 4};
    CpuDescriptorHandle ov_src[2] = {cpu_at(dst_smp, 4), cpu_at(dst_smp, 8)};
    const float expected_bias_6 = dst_smp->sampler(4)->desc.mip_lod_bias;
    const float expected_bias_13 = dst_smp->sampler(11)->desc.mip_lod_bias;
    CHECK(reg.copy_descriptors(2, ov_dst, ov_sizes, 2, ov_src, ov_sizes, DescriptorHeapType::Sampler) == Status::Ok);
    CHECK(dst_smp->sampler(6)->desc.mip_lod_bias == expected_bias_6);
    CHECK(dst_smp->sampler(13)->desc.mip_lod_bias == expected_bias_13);

    // Large-batch eviction (>8 resource IDs and >8 view IDs) exercises binary-search path.
    RootSignatureDesc rsd;
    RootParameterDesc tbl;
    tbl.parameter_type = RootParameterType::DescriptorTable;
    tbl.ranges.push_back(DescriptorRangeDesc{DescriptorRangeType::Srv, 1, 0, 0});
    rsd.parameters.push_back(tbl);
    RootSignature rs;
    CHECK(RootSignature::build(rsd, &rs).ok());
    ShaderBinding sb;
    sb.binding = 0;
    sb.register_class = DescriptorRangeType::Srv;
    sb.shader_register = 0;
    const BindGroupPlan plan = BindGroupPlan::build(rs, std::span<const ShaderBinding>(&sb, 1), WebGpuStage::Fragment);

    RootArgumentState state;
    DescriptorHeap* hs[1] = {dst_cbv};
    CHECK(state.set_descriptor_heaps(1, hs) == Status::Ok);
    CHECK(state.set_root_signature(BindPoint::Graphics, &rs) == Status::Ok);
    BindGroupCache cache;
    for (uint32_t i = 0; i < 20; ++i) {
        CHECK(state.set_descriptor_table(BindPoint::Graphics, 0, gpu_at(dst_cbv, 10 + i)) == Status::Ok);
        BindGroupLookup lk = cache.get_or_build(state, BindPoint::Graphics, plan, 1, test_resolver, nullptr);
        CHECK(lk.status == Status::Ok && lk.rebuilt);
        lk.bind_group->gpu_id = 500 + i;
        lk.bind_group->view_ids.push_back(9000 + i);
        cache.note_group_ids(*lk.bind_group);
    }
    std::vector<uint32_t> evict_res;
    std::vector<uint32_t> evict_views;
    for (uint32_t i = 0; i < 10; ++i) {
        evict_res.push_back(1020 + i);  // matches first 10
    }
    for (uint32_t i = 10; i < 20; ++i) {
        evict_views.push_back(9000 + i);  // matches next 10 via view_ids
    }
    std::vector<uint32_t> retired;
    const size_t dropped = cache.evict_resources_and_views(evict_res, evict_views, &retired);
    CHECK(dropped == 20);
    CHECK(retired.size() == 20);
    CHECK(cache.size() == 0);

    // Exact eviction index: groups sharing ids, stale index entries (an evicted id evicted again,
    // groups rebuilt under the same key, an id re-noted after its groups were evicted) evict exactly
    // the groups that reference the ids, in table order.
    for (uint32_t i = 0; i < 40; ++i) {
        CHECK(state.set_descriptor_table(BindPoint::Graphics, 0, gpu_at(dst_cbv, 10 + i)) == Status::Ok);
        BindGroupLookup lk = cache.get_or_build(state, BindPoint::Graphics, plan, 2, test_resolver, nullptr);
        CHECK(lk.status == Status::Ok && lk.rebuilt);
        lk.bind_group->gpu_id = 700 + i;
        lk.bind_group->view_ids = {7000 + i % 4, 7100 + i};  // view 7000+k shared by every 4th group
        cache.note_group_ids(*lk.bind_group);
    }
    CHECK(cache.size() == 40);
    retired.clear();
    const uint32_t shared_view[1] = {7001};
    CHECK(cache.evict_resources_and_views({}, shared_view, &retired) == 10);
    std::vector<uint32_t> expect;
    for (uint32_t i = 1; i < 40; i += 4) expect.push_back(700 + i);
    std::sort(retired.begin(), retired.end());
    CHECK(retired == expect);
    CHECK(cache.evict_resources_and_views({}, shared_view, &retired) == 0);  // nothing references it any more
    const uint32_t gone_view[1] = {7105};  // its group (i = 5) was evicted above
    CHECK(cache.evict_resources_and_views({}, gone_view, &retired) == 0);
    // Rebuild group 5 under the same key: it references 7001 again once noted.
    CHECK(state.set_descriptor_table(BindPoint::Graphics, 0, gpu_at(dst_cbv, 15)) == Status::Ok);
    BindGroupLookup again = cache.get_or_build(state, BindPoint::Graphics, plan, 2, test_resolver, nullptr);
    CHECK(again.status == Status::Ok && again.rebuilt);
    again.bind_group->gpu_id = 799;
    again.bind_group->view_ids = {7001};
    cache.note_group_ids(*again.bind_group);
    retired.clear();
    CHECK(cache.evict_resources_and_views({}, shared_view, &retired) == 1 && retired == std::vector<uint32_t>{799});
    // Resource ids the groups bind (heap slot 10 + i -> resource 1020 + i) plus views.
    retired.clear();
    const uint32_t res[2] = {1022, 1023};      // groups 2 and 3
    const uint32_t views[2] = {7000, 7120};    // groups 0, 4, 8, ... and 20 (already in the 7000 set)
    CHECK(cache.evict_resources_and_views(res, views, &retired) == 12);
    CHECK(cache.size() == 40 - 10 - 12);
}

void test_large_descriptor_range_intersections() {
    // Reference: snapshot every source, then scatter in caller order. Exercise
    // zero ranges, nested/duplicate sources, late overlap, and mixed heaps.
    constexpr uint32_t ranges = 256, slots = 272;
    for (uint32_t seed = 0; seed < 3; ++seed) for (uint32_t scenario = 0; scenario < 6; ++scenario) {
        DescriptorHeapRegistry reg;
        std::array<DescriptorHeap*, 3> heaps{};
        std::array<std::vector<ViewDescriptor>, 3> expected;
        for (uint32_t h = 0; h < heaps.size(); ++h) {
            DescriptorHeapDesc desc;
            desc.type = DescriptorHeapType::CbvSrvUav;
            desc.num_descriptors = slots;
            // Repeat the snapshot oracle for address-based placed heap handles.
            if (seed == 2) desc.cpu_address = 0x10000000ull + h * 0x10000ull;
            CHECK(reg.create_heap(desc, &heaps[h]) == Status::Ok);
            fill_srvs(reg, heaps[h], 0, slots, 10000 * h + 1);
            for (uint32_t i = 0; i < slots; ++i) expected[h].push_back(*heaps[h]->view(i));
        }
        std::vector<CpuDescriptorHandle> src, dst;
        std::vector<uint32_t> counts, src_heap, dst_heap, src_index, dst_index;
        for (uint32_t i = 0; i < ranges; ++i) {
            const uint32_t sh = scenario == 2 ? 1 + i % 2 : (scenario == 3 ? i % 3 : 0);
            const uint32_t dh = scenario == 4 ? i % 2 : 0;
            src_heap.push_back(sh); dst_heap.push_back(dh);
            src_index.push_back(8 * ((i * 7 + seed) % 32) + i % 2);
            dst_index.push_back(8 * ((i * 13 + seed) % 32) + 4);
            counts.push_back(i % 4);
        }
        // This late alias corrupts a
        // source read later unless the whole source stream is snapshotted first.
        if (scenario == 1) dst_index[65] = src_index[3];
        for (uint32_t i = 0; i < ranges; ++i) {
            src.push_back(counts[i] ? cpu_at(heaps[src_heap[i]], src_index[i]) : CpuDescriptorHandle{0});
            dst.push_back(counts[i] ? cpu_at(heaps[dst_heap[i]], dst_index[i]) : CpuDescriptorHandle{0});
        }
        std::array<uint64_t, 3> updates{}, generations{};
        for (uint32_t h = 0; h < heaps.size(); ++h) generations[h] = heaps[h]->generation();
        if (scenario == 5) {
            src.back() = cpu_at(heaps[0], slots);  // late failure must not write any destination
        } else {
            std::vector<ViewDescriptor> stream;
            for (uint32_t i = 0; i < ranges; ++i) for (uint32_t j = 0; j < counts[i]; ++j)
                stream.push_back(expected[src_heap[i]][src_index[i] + j]);
            size_t cursor = 0;
            for (uint32_t i = 0; i < ranges; ++i) {
                bool changed = false;
                for (uint32_t j = 0; j < counts[i]; ++j) {
                    auto& target = expected[dst_heap[i]][dst_index[i] + j];
                    changed |= std::memcmp(&target, &stream[cursor], sizeof(target)) != 0;
                    target = stream[cursor++];
                }
                updates[dst_heap[i]] += changed;
            }
        }
        const auto status = reg.copy_descriptors(ranges, dst.data(), counts.data(), ranges,
                                                 src.data(), counts.data(), DescriptorHeapType::CbvSrvUav);
        CHECK(scenario == 5 ? status != Status::Ok : status == Status::Ok);
        for (uint32_t h = 0; h < heaps.size(); ++h) {
            CHECK(heaps[h]->generation() == generations[h] + updates[h]);
            for (uint32_t i = 0; i < slots; ++i)
                CHECK(std::memcmp(heaps[h]->view(i), &expected[h][i], sizeof(ViewDescriptor)) == 0);
        }
    }
}

void test_large_sampler_range_intersections() {
    DescriptorHeapRegistry reg;
    auto* heap = make_heap(reg, DescriptorHeapType::Sampler, 256, true);
    std::array<DescriptorHeap*, 2> sources{
        make_heap(reg, DescriptorHeapType::Sampler, 256, false),
        make_heap(reg, DescriptorHeapType::Sampler, 256, false)};
    std::array<CpuDescriptorHandle, 128> src{}, dst{};
    for (uint32_t i = 0; i < src.size(); ++i) {
        SamplerDesc sampler;
        sampler.mip_lod_bias = static_cast<float>(i) * 0.001f;
        CHECK(reg.create_sampler(sampler, cpu_at(sources[i % 2], 2 * i)) == Status::Ok);
        src[i] = cpu_at(sources[i % 2], 2 * ((i * 67) % 128));
        dst[i] = cpu_at(heap, 2 * i + 1);
    }
    const auto generation = heap->generation();
    CHECK(reg.copy_descriptors(src.size(), dst.data(), nullptr, src.size(), src.data(), nullptr,
                               DescriptorHeapType::Sampler) == Status::Ok);
    CHECK(heap->generation() == generation + src.size());
    for (uint32_t i = 0; i < src.size(); ++i)
        CHECK(heap->sampler(2 * i + 1)->desc.mip_lod_bias == static_cast<float>((i * 67) % 128) * 0.001f);
}

}  // namespace

int main() {
    test_heap_creation_and_handles();
    test_live_heap_gauges();
    test_descriptor_writes();
    test_copy_descriptors_simple();
    test_copy_descriptors_multi_range();
    test_generation_tracking();
    test_root_signature_model();
    test_root_signature_validation();
    test_root_signature_hash();
    test_root_signature_blob_roundtrip();
    test_root_signature_blob_known_layout();
    test_root_signature_blob_malformed();
    test_root_arguments();
    test_bind_group_plan();
    test_bind_group_cache();
    test_heap_indexed_bind_groups();
    test_heap_indexed_binding_array();
    test_heap_indexed_stride();
    test_cb_indexed_heap_bind_groups();
    test_inline_sampler_plan();
    test_state_tracker_basic();
    test_state_tracker_subresources_and_split();
    test_state_tracker_execute();
    test_state_tracker_diagnostic_cap();
    test_usage_scope_conflicts();
    test_concurrent_disjoint_writes();
    test_placed_heaps_sorted_lookup();
    test_bind_group_cache_flat_mru_erase_and_scale();
    test_state_tracker_flat_index_and_reset();
    test_heap0_serial_zero_vs_unbound_heap_indexed();
    test_table_window_u32_span_saturation();
    test_handle_overflow_guard_and_cbv_alignment_guard();
    test_slot_source_switch_and_rebuild_clears_metadata();
    test_copy_descriptors_batched_gen_and_overlap_fastpath();
    test_large_descriptor_range_intersections();
    test_large_sampler_range_intersections();

    if (g_failures != 0) {
        std::fprintf(stderr, "test_descriptor_heap failed with %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("All descriptor_heap_tests passed.\n");
    return 0;
}
