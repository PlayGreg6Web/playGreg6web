#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "gpu/pipeline_prewarm.h"

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
}  // namespace

int main() {
    test_roundtrip_and_order();
    test_in_flight_cap_and_progress();
    test_title_beats_queue();
    test_pack_filter();
    test_malformed();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_pipeline_prewarm: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_pipeline_prewarm: ok");
    return 0;
}
