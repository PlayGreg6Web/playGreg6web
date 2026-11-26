#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <thread>
#include <atomic>
#include "gpu/shader_pack_index.h"
#include <memory>
#include <string>
#include <vector>

#include "gpu/pipeline_prewarm.h"
#include "gpu/pipeline_prewarm_host.h"

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
    if (g_failures != 0) {
        std::fprintf(stderr, "test_pipeline_prewarm: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_pipeline_prewarm: ok");
    return 0;
}
