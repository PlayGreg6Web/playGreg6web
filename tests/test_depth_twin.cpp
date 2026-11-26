#include <cstdint>
#include <cstdio>

#include "gpu/depth_twin.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_depth_twin.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using playgta6::gpu::bridge::DepthTwinTracker;
using Side = DepthTwinTracker::Side;

// A fresh texture has no twin contents: copy in once, then not again until the depth side changes.
void test_copy_in_once() {
    DepthTwinTracker t;
    CHECK(t.side(7) == Side::Stale);
    CHECK(t.needs_copy_in(7));
    t.note_copied_in(7);
    CHECK(t.side(7) == Side::Synced);
    CHECK(!t.needs_copy_in(7));
    CHECK(!t.needs_copy_out(7));
    CHECK(t.needs_copy_in(8));  // other textures are independent
}

// A write dispatch leaves the depth plane behind; the copy back makes both equal again (no copy in).
void test_write_then_copy_out() {
    DepthTwinTracker t;
    t.note_copied_in(7);
    t.note_twin_written(7);
    CHECK(t.needs_copy_out(7));
    CHECK(!t.needs_copy_in(7));
    t.note_copied_out(7);
    CHECK(t.side(7) == Side::Synced);
    CHECK(!t.needs_copy_out(7));
    CHECK(!t.needs_copy_in(7));
}

// A render pass writing the depth plane makes the twin stale; an untracked texture stays untracked.
void test_depth_write_invalidates() {
    DepthTwinTracker t;
    t.note_depth_written(3);  // never twinned: nothing to track
    CHECK(t.tracked() == 0);
    t.note_copied_in(3);
    t.note_depth_written(3);
    CHECK(t.needs_copy_in(3));
    CHECK(!t.needs_copy_out(3));
    t.note_copied_in(3);
    CHECK(!t.needs_copy_in(3));
}

// Pending twin writes are not forgotten by a depth write (the caller flushes them first).
void test_pending_write_kept() {
    DepthTwinTracker t;
    t.note_copied_in(5);
    t.note_twin_written(5);
    t.note_depth_written(5);
    CHECK(t.needs_copy_out(5));
}

void test_forget_and_reset() {
    DepthTwinTracker t;
    t.note_copied_in(1);
    t.note_copied_in(2);
    t.forget(1);
    CHECK(t.needs_copy_in(1) && !t.needs_copy_in(2));
    t.reset();
    CHECK(t.needs_copy_in(2) && t.tracked() == 0);
}
}  // namespace

int main() {
    test_copy_in_once();
    test_write_then_copy_out();
    test_depth_write_invalidates();
    test_pending_write_kept();
    test_forget_and_reset();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("test_depth_twin: ok");
    return 0;
}
