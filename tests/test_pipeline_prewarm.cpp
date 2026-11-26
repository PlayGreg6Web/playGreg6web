#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <utility>
#include <thread>
#include <atomic>
#include "gpu/shader_pack_index.h"
#include <memory>
#include <string>
#include <vector>

#include "gpu/pipeline_prewarm.h"
#include "gpu/pipeline_prewarm_host.h"
#include "gpu/prewarm_pipeline_desc.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_pipeline_prewarm.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu;

struct E {
    uint32_t kind;
    uint64_t k0;
    uint64_t k1;
};

std::vector<uint8_t> make_list(const std::vector<E>& es) {
    PipelinePrewarm src;
    for (const E& e : es) src.note_title_pipeline(e.kind, e.k0, e.k1);
    return src.save_warm_list();
}

void test_roundtrip_and_order() {
    auto bytes = make_list({{0, 10, 11}, {1, 20, 0}, {0, 12, 13}});
    CHECK(bytes.size() == kPrewarmListHeaderBytes + 3 * kPrewarmListEntryBytes);
    PipelinePrewarm q;
    std::string err;
    uint32_t added = 0;
    CHECK(q.load_warm_list(bytes.data(), bytes.size(), &err, &added));
    CHECK(added == 3);
    PrewarmTask t[8];
    CHECK(q.take(t, 8) == 3);
    CHECK(t[0].key0 == 10 && t[1].key0 == 20 && t[1].kind == kPrewarmKindCompute && t[2].key0 == 12);
}

void test_in_flight_cap_and_progress() {
    auto bytes = make_list({{0, 1, 0}, {0, 2, 0}, {0, 3, 0}});
    PipelinePrewarm q;
    q.set_max_in_flight(2);
    CHECK(q.load_warm_list(bytes.data(), bytes.size(), nullptr));
    PrewarmTask t[8];
    CHECK(q.take(t, 8) == 2);
    CHECK(q.take(t, 8) == 0);
    PrewarmProgress p = q.progress();
    CHECK(p.total == 3 && p.queued == 1 && p.in_flight == 2);
    CHECK(q.complete(1, true, 5000));
    CHECK(!q.complete(1, true, 1));
    CHECK(q.complete(2, false, 9000));
    CHECK(!q.complete(99, true, 1));
    p = q.progress();
    CHECK(p.done == 1 && p.failed == 1 && p.longest_ns == 9000 && p.total_ns == 14000);
    CHECK(q.take(t, 8) == 1 && t[0].key0 == 3);
}

void test_title_beats_queue() {
    auto bytes = make_list({{0, 1, 0}, {0, 2, 0}});
    PipelinePrewarm q;
    CHECK(q.load_warm_list(bytes.data(), bytes.size(), nullptr));
    CHECK(q.note_title_pipeline(kPrewarmKindGraphics, 1, 0));
    CHECK(!q.note_title_pipeline(kPrewarmKindGraphics, 1, 0));
    PrewarmTask t[4];
    CHECK(q.take(t, 4) == 1 && t[0].key0 == 2);
    PrewarmProgress p = q.progress();
    CHECK(p.skipped == 1 && p.observed == 1);
    CHECK(q.save_warm_list().size() == kPrewarmListHeaderBytes + kPrewarmListEntryBytes);
}

void test_pack_filter() {
    auto pack = std::make_shared<ShaderPackIndex>();
    (*pack)[1] = PackStage::Vertex;
    (*pack)[2] = PackStage::Fragment;
    auto bytes = make_list({{0, 1, 2}, {0, 1, 99}, {1, 77, 0}});
    PipelinePrewarm q;
    q.set_pack_index(pack);
    uint32_t added = 0;
    CHECK(q.load_warm_list(bytes.data(), bytes.size(), nullptr, &added));
    CHECK(added == 1);
    CHECK(q.progress().skipped == 2);
    CHECK(q.load_warm_list(bytes.data(), bytes.size(), nullptr, &added) && added == 0);
}

ShaderPackIndex make_pack() {
    ShaderPackIndex p;
    for (uint64_t k : {900ull, 5ull, 77ull, 31ull, 1000000007ull}) p[k] = PackStage::Vertex;
    return p;
}

// Ids depend only on the key, not on whether the prewarm thread or the title reaches a module first.
void test_module_ids_order_independent() {
    const ShaderPackIndex pack = make_pack();
    PackModuleTable on_demand(pack), warmed(pack);
    std::vector<uint32_t> a, b;
    std::vector<uint32_t> created;
    const auto mk = [&](uint32_t id) { created.push_back(id); };
    const uint64_t title_order[] = {77, 5, 900, 31};
    for (uint64_t k : title_order) a.push_back(on_demand.get_or_create(k, mk));
    // prewarm builds in a different order first, the title then reaches the same keys
    for (uint64_t k : {31ull, 900ull, 1000000007ull, 5ull}) warmed.get_or_create(k, mk);
    for (uint64_t k : title_order) b.push_back(warmed.get_or_create(k, mk));
    CHECK(a == b);
    CHECK(a[0] == kPackModuleIdBase + 2 && a[1] == kPackModuleIdBase + 0);  // rank in ascending key order
    CHECK(on_demand.get_or_create(12345, mk) == 0);                         // not a pack key
    CHECK(warmed.created() == 5);
}

void test_module_concurrent_single_build() {
    const ShaderPackIndex pack = make_pack();
    PackModuleTable t(pack);
    std::atomic<int> builds{0};
    std::vector<uint32_t> got(8);
    std::vector<std::thread> th;
    for (int i = 0; i < 8; ++i) {
        th.emplace_back([&, i] {
            got[i] = t.get_or_create(77, [&](uint32_t) { ++builds; std::this_thread::yield(); });
        });
    }
    for (auto& x : th) x.join();
    CHECK(builds.load() == 1);
    for (uint32_t g : got) CHECK(g == got[0] && g != 0);
}

void test_malformed() {
    PipelinePrewarm q;
    std::string err;
    uint8_t tiny[4] = {};
    CHECK(!q.load_warm_list(tiny, sizeof(tiny), &err));
    auto bytes = make_list({{0, 1, 0}});
    auto bad = bytes;
    bad[0] = 'X';
    CHECK(!q.load_warm_list(bad.data(), bad.size(), &err));
    CHECK(!q.load_warm_list(bytes.data(), bytes.size() - 1, &err));
    bad = bytes;
    bad[kPrewarmListHeaderBytes] = 7;  // kind
    CHECK(!q.load_warm_list(bad.data(), bad.size(), &err));
    CHECK(q.progress().total == 0);
    CHECK(!q.note_title_pipeline(0, 0, 0));
}
void test_file_roundtrip_and_ordering() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "pg6_prewarm_host_test";
    fs::remove_all(dir);
    const std::string path = (dir / "sub" / "title.pg6w").string();

    // Nothing observed: nothing written (a short run never erases a list).
    CHECK(!write_warm_list_file(path, PipelinePrewarm().save_warm_list()));
    CHECK(!fs::exists(path));

    auto bytes = make_list({{0, 30, 31}, {1, 10, 0}, {0, 20, 0}});
    CHECK(write_warm_list_file(path, bytes));
    CHECK(!fs::exists(path + ".tmp"));
    std::vector<uint8_t> back;
    CHECK(read_warm_list_file(path, &back));
    CHECK(back == bytes);

    // Load through the host: first-use order survives the file.
    pipeline_prewarm().reset();
    std::string msg;
    CHECK(prewarm_host_configure(path, &msg) == 3);
    PrewarmTask t[3];
    CHECK(pipeline_prewarm().take(t, 3) == 3);
    CHECK(t[0].key0 == 30 && t[0].key1 == 31 && t[0].kind == 0);
    CHECK(t[1].key0 == 10 && t[1].kind == 1);
    CHECK(t[2].key0 == 20);

    // Counters: two built, the title uses one of them; the third is still in flight.
    CHECK(pipeline_prewarm().complete(t[0].id, true, 5));
    CHECK(pipeline_prewarm().complete(t[1].id, true, 5));
    pipeline_prewarm().note_title_pipeline(0, 30, 31);
    PrewarmUse u = pipeline_prewarm().use_stats();
    CHECK(u.prewarmed == 2 && u.hit == 1 && u.wasted == 1);
    // The title uses a pipeline while it is still being built: counts as a hit.
    pipeline_prewarm().note_title_pipeline(0, 20, 0);
    CHECK(pipeline_prewarm().complete(t[2].id, true, 5));
    u = pipeline_prewarm().use_stats();
    CHECK(u.prewarmed == 3 && u.hit == 2 && u.wasted == 1);
    CHECK(prewarm_host_save());
    CHECK(prewarm_host_report_json().find("\"hit_before_use\": 2") != std::string::npos);

    // Corrupt file: ignored, queue stays empty.
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "PG6Wgarbage";
    }
    pipeline_prewarm().reset();
    CHECK(prewarm_host_configure(path, &msg) == 0);
    CHECK(msg.find("ignored") != std::string::npos);
    PrewarmTask none[1];
    CHECK(pipeline_prewarm().take(none, 1) == 0);
    // Missing file and "off" are quiet no-ops.
    CHECK(prewarm_host_configure((dir / "absent.pg6w").string(), &msg) == 0);
    CHECK(prewarm_host_configure("off", &msg) == 0);
    CHECK(!prewarm_host_save());

    // Background builder: drains the queue through the callback in list order.
    pipeline_prewarm().reset();
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(prewarm_host_configure(path, &msg) == 3);
    std::vector<uint64_t> built;
    prewarm_host_start({[] { return true; }, [&built](uint32_t, uint64_t k0, uint64_t) {
                            built.push_back(k0);
                            return true;
                        }});
    prewarm_host_wait();
    CHECK((built == std::vector<uint64_t>{30, 10, 20}));
    CHECK(pipeline_prewarm().progress().done == 3);
    CHECK(pipeline_prewarm().use_stats().wasted == 3);

    prewarm_host_configure("off", &msg);
    pipeline_prewarm().reset();
    fs::remove_all(dir);
}

std::vector<uint8_t> blob(uint8_t seed, size_t n = 40) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + i);
    return v;
}

// Version 2 carries full pipeline descriptions; a list without any is still written as version 1, and
// old module-only lists load unchanged.
void test_full_list_roundtrip_and_v1_compat() {
    PipelinePrewarm src;
    src.note_title_pipeline(kPrewarmKindGraphics, 10, 11);       // module-only: redundant with the full ones
    src.note_title_pipeline_full(kPrewarmKindGraphics, 10, 11, blob(1));
    src.note_title_pipeline_full(kPrewarmKindGraphics, 10, 11, blob(2));  // another state of the same shaders
    CHECK(!src.note_title_pipeline_full(kPrewarmKindGraphics, 10, 11, blob(2)));  // same pipeline again
    src.note_title_pipeline(kPrewarmKindCompute, 20, 0);          // no full entry: stays module-only
    src.note_title_pipeline_full(kPrewarmKindCompute, 30, 0, blob(3, 7));
    CHECK(src.progress().observed == 5);
    const std::vector<uint8_t> bytes = src.save_warm_list();
    CHECK(bytes[4] == kPrewarmListVersionPipelines);
    CHECK(bytes.size() == kPrewarmListHeaderBytes + 4 * kPrewarmListEntryBytes + 40 + 40 + 7);  // entry 10/11 module-only elided

    PipelinePrewarm q;
    std::string err;
    uint32_t added = 0;
    CHECK(q.load_warm_list(bytes.data(), bytes.size(), &err, &added));
    CHECK(added == 4);
    PrewarmTask t[8];
    CHECK(q.take(t, 8) == 4);
    CHECK(t[0].key0 == 10 && t[1].key0 == 10 && t[2].key0 == 20 && t[3].key0 == 30);
    CHECK(q.detail_of(t[0].id) != nullptr && *q.detail_of(t[0].id) == blob(1));
    CHECK(*q.detail_of(t[1].id) == blob(2));
    CHECK(q.detail_of(t[2].id) == nullptr);  // module-only entry
    CHECK(*q.detail_of(t[3].id) == blob(3, 7));
    CHECK(q.detail_of(99) == nullptr);
    // Loading again adds nothing; the order and details survive a second save only through observation.
    CHECK(q.load_warm_list(bytes.data(), bytes.size(), nullptr, &added) && added == 0);

    // A list with no detail stays version 1 (readable by older builds).
    PipelinePrewarm plain;
    plain.note_title_pipeline(0, 1, 2);
    const std::vector<uint8_t> v1 = plain.save_warm_list();
    CHECK(v1[4] == kPrewarmListVersionModules && v1.size() == kPrewarmListHeaderBytes + kPrewarmListEntryBytes);
    PipelinePrewarm r;
    CHECK(r.load_warm_list(v1.data(), v1.size(), nullptr, &added) && added == 1);
    CHECK(r.take(t, 8) == 1 && r.detail_of(t[0].id) == nullptr);

    // Truncated or oversized detail is refused and leaves the queue unchanged.
    PipelinePrewarm bad;
    CHECK(!bad.load_warm_list(bytes.data(), bytes.size() - 1, &err));
    std::vector<uint8_t> huge = bytes;
    huge[kPrewarmListHeaderBytes + 7] = 0x7f;  // detail_len of the first entry: far past the end
    CHECK(!bad.load_warm_list(huge.data(), huge.size(), &err));
    CHECK(bad.progress().total == 0);
    std::vector<uint8_t> v3 = bytes;
    v3[4] = 3;
    CHECK(!bad.load_warm_list(v3.data(), v3.size(), &err));
}

// pipelines_prewarmed / hit_before_use / wasted count full pipelines apart from modules.
void test_full_pipeline_counters() {
    PipelinePrewarm src;
    for (uint8_t i = 1; i <= 3; ++i) src.note_title_pipeline_full(kPrewarmKindGraphics, 5, 6, blob(i));
    src.note_title_pipeline(kPrewarmKindCompute, 7, 0);
    const std::vector<uint8_t> bytes = src.save_warm_list();

    PipelinePrewarm q;
    CHECK(q.load_warm_list(bytes.data(), bytes.size(), nullptr));
    PrewarmTask t[8];
    CHECK(q.take(t, 8) == 4);
    CHECK(q.complete(t[0].id, true, 10));  // pipeline 1: built, then reached by the title (hit)
    CHECK(q.complete(t[1].id, true, 10));  // pipeline 2: built, never reached (wasted)
    CHECK(q.complete_lost(t[2].id));       // pipeline 3: the title created it first
    CHECK(!q.complete_lost(t[2].id));
    CHECK(q.complete(t[3].id, true, 10));  // module-only compute entry: built
    CHECK(q.note_task_used(t[0].id));      // the title reached the prewarmed pipeline
    CHECK(!q.note_task_used(t[0].id));
    const PrewarmUse u = q.use_stats();
    CHECK(u.pipelines_prewarmed == 2 && u.pipelines_hit == 1 && u.pipelines_wasted == 1);
    CHECK(u.prewarmed == 1 && u.hit == 0 && u.wasted == 1);
    CHECK(q.progress().skipped == 1);
    // The reached pipeline enters the next run's list even though the title did not create it.
    CHECK(q.progress().observed == 1);
    const std::vector<uint8_t> next = q.save_warm_list();
    PipelinePrewarm r;
    uint32_t added = 0;
    CHECK(r.load_warm_list(next.data(), next.size(), nullptr, &added) && added == 1);
}

void test_desc_codec() {
    PrewarmPipelineDesc d;
    d.kind = kPrewarmKindGraphics;
    d.pack_key0 = 0x1111;
    d.pack_key1 = 0x2222;
    std::memset(&d.graphics, 0, sizeof(d.graphics));
    d.graphics.vs_shader = 0xA1;
    d.graphics.ps_shader = 0xB2;
    d.graphics.color_target_count = 1;
    d.graphics.vertex_buffer_count = 1;
    d.graphics.vertex_attribute_count = 1;
    d.graphics.vertex_attributes[0].offset = 12;
    d.has_layout = true;
    d.layout.content_id = 0x99;
    bridge::GroupLayoutDesc g;
    g.content_id = 0x77;
    bridge::LayoutBinding b;
    b.entry.binding = 3;
    b.entry.visibility = 2;
    b.entry.binding_array_size = 4;
    b.entry.min_binding_size = 64;
    b.join.binding = 3;
    b.join.shader_register = 5;
    b.join.register_class = d3d12_state::DescriptorRangeType::Uav;
    b.join.root_constants = true;
    b.join.cb_mask = 0xff;
    b.join.inline_sampler = true;
    b.join.sampler.max_anisotropy = 8;
    g.bindings.push_back(b);
    g.bindings.push_back(bridge::LayoutBinding{});
    d.layout.groups.push_back(g);
    d.layout.groups.push_back(bridge::GroupLayoutDesc{});

    const std::vector<uint8_t> bytes = encode_prewarm_pipeline_desc(d);
    PrewarmPipelineDesc back;
    CHECK(decode_prewarm_pipeline_desc(bytes.data(), bytes.size(), &back));
    CHECK(back.kind == d.kind && back.pack_key0 == 0x1111 && back.pack_key1 == 0x2222);
    CHECK(std::memcmp(&back.graphics, &d.graphics, sizeof(d.graphics)) == 0);
    CHECK(back.has_layout && back.layout.content_id == 0x99 && back.layout.groups.size() == 2);
    CHECK(back.layout.groups[0].content_id == 0x77 && back.layout.groups[0].bindings.size() == 2);
    const bridge::LayoutBinding& bb = back.layout.groups[0].bindings[0];
    CHECK(std::memcmp(&bb.entry, &b.entry, sizeof(b.entry)) == 0);
    CHECK(d3d12_state::same_join(bb.join, b.join));
    CHECK(back.layout.groups[1].bindings.empty());
    CHECK(encode_prewarm_pipeline_desc(back) == bytes);  // stable: identical descriptions hash identically

    // Compute, with layout "auto".
    PrewarmPipelineDesc c;
    c.kind = kPrewarmKindCompute;
    c.pack_key0 = 0x3333;
    c.compute.cs_shader = 0xC5;
    c.compute.root_signature_id = 9;
    const std::vector<uint8_t> cb = encode_prewarm_pipeline_desc(c);
    PrewarmPipelineDesc cback;
    CHECK(decode_prewarm_pipeline_desc(cb.data(), cb.size(), &cback));
    CHECK(cback.kind == kPrewarmKindCompute && cback.compute.cs_shader == 0xC5 && !cback.has_layout);

    // Truncation at any point, trailing bytes and a foreign version are refused.
    for (size_t n = 0; n < bytes.size(); n += 7) {
        PrewarmPipelineDesc x;
        CHECK(!decode_prewarm_pipeline_desc(bytes.data(), n, &x));
    }
    std::vector<uint8_t> extra = bytes;
    extra.push_back(0);
    PrewarmPipelineDesc x;
    CHECK(!decode_prewarm_pipeline_desc(extra.data(), extra.size(), &x));
    std::vector<uint8_t> v = bytes;
    v[0] = 9;
    CHECK(!decode_prewarm_pipeline_desc(v.data(), v.size(), &x));
    CHECK(!decode_prewarm_pipeline_desc(nullptr, 0, &x));

    // Pack module ids map back to their key.
    const ShaderPackIndex pack = make_pack();
    PackModuleTable t(pack);
    CHECK(t.key_of(t.id_of(77)) == 77 && t.key_of(t.id_of(1000000007ull)) == 1000000007ull);
    CHECK(t.key_of(5) == 0 && t.key_of(kPackModuleIdBase + 5) == 0);
}

// The host thread hands detailed entries to build_pipeline (with their detail and task id) and module-only
// ones to build; a lost race is counted as skipped, not as built.
void test_host_builds_pipelines() {
    PipelinePrewarm src;
    src.note_title_pipeline_full(kPrewarmKindGraphics, 40, 41, blob(1));
    src.note_title_pipeline_full(kPrewarmKindGraphics, 42, 43, blob(2));
    src.note_title_pipeline_full(kPrewarmKindGraphics, 44, 45, blob(3));
    src.note_title_pipeline(kPrewarmKindCompute, 46, 0);
    const std::vector<uint8_t> bytes = src.save_warm_list();
    pipeline_prewarm().reset();
    CHECK(pipeline_prewarm().load_warm_list(bytes.data(), bytes.size(), nullptr));

    std::vector<uint64_t> modules;
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> pipelines;
    prewarm_host_start(
        {[] { return true; },
         [&modules](uint32_t, uint64_t k0, uint64_t) {
             modules.push_back(k0);
             return true;
         },
         [&pipelines](uint32_t task, uint32_t, uint64_t k0, uint64_t, const std::vector<uint8_t>& detail) {
             CHECK(task != 0);
             pipelines.emplace_back(k0, detail);
             if (k0 == 42) return PrewarmBuild::Lost;
             if (k0 == 44) return PrewarmBuild::Failed;
             return PrewarmBuild::Built;
         }});
    prewarm_host_wait();
    CHECK(modules == std::vector<uint64_t>{46});
    CHECK(pipelines.size() == 3 && pipelines[0].first == 40 && pipelines[0].second == blob(1));
    CHECK(pipelines[1].second == blob(2) && pipelines[2].second == blob(3));
    const PrewarmProgress p = pipeline_prewarm().progress();
    CHECK(p.done == 2 && p.skipped == 1 && p.failed == 1);
    const PrewarmUse u = pipeline_prewarm().use_stats();
    CHECK(u.pipelines_prewarmed == 1 && u.pipelines_wasted == 1 && u.wasted == 1);
    const std::string json = prewarm_host_report_json();
    CHECK(json.find("\"pipelines_prewarmed\": 1") != std::string::npos);
    CHECK(json.find("\"pipelines_wasted\": 1") != std::string::npos);
    CHECK(json.find("\"pipelines_hit_before_use\": 0") != std::string::npos);
    pipeline_prewarm().reset();
}

}  // namespace

int main() {
    test_roundtrip_and_order();
    test_in_flight_cap_and_progress();
    test_title_beats_queue();
    test_pack_filter();
    test_malformed();
    test_module_ids_order_independent();
    test_module_concurrent_single_build();
    test_file_roundtrip_and_ordering();
    test_full_list_roundtrip_and_v1_compat();
    test_full_pipeline_counters();
    test_desc_codec();
    test_host_builds_pipelines();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_pipeline_prewarm: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_pipeline_prewarm: ok");
    return 0;
}
