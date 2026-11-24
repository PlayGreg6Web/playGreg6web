// Micro-benchmark for guest indirect-call dispatch (__playgta6_indirect_call, O33).
//
// Builds a synthetic registry the size of the real program (~440k lifted functions in four
// modules, like exe + vcruntime140 + vcruntime140_1 + msvcp140), in either dispatch mode
// (per-function tables or R37 shard dispatch tables), plus host callables, and times
// __playgta6_indirect_call for several target patterns with the per-thread dispatch cache off
// and on:
//   hot-N      skewed (cubic) picks from N distinct functions, the shape of a frame loop
//   uniform    uniform picks from all functions (worst case for any cache)
//   mixed      hot-1024 with every 8th call to a host callable (import-like)
// Every dispatch is checked (the callee counts calls), so the benchmark doubles as a smoke test.
//
//   bench_indirect_call --full [--threads N] [--calls N]   (best of 5 interleaved runs each)
// Without --full (CTest, also as wasm64) it is a smoke run: a small registry and few calls,
// whose numbers mean little.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/guest_runtime.h"

namespace {

using namespace playgta6;
using namespace playgta6::guest;

thread_local uint64_t t_calls = 0;
thread_local uint64_t t_check = 0;

void lifted_target(GuestState* s) {
    ++t_calls;
    t_check += s->rip;
}

void shard_target(uint32_t local_idx, Pg6GuestState* s) {
    ++t_calls;
    t_check += s->rip ^ local_idx;
}

void host_target(void* state) {
    ++t_calls;
    t_check += static_cast<GuestState*>(state)->rip;
}

struct Rng {
    uint64_t x;
    uint64_t next() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return x;
    }
    double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

struct Registry {
    std::vector<uint64_t> vas;  // every lifted function, ascending
    std::vector<std::vector<LiftedFunctionEntry>> tables;
    std::vector<std::vector<Pg6ShardFunctionEntry>> shard_entries;
    std::vector<std::vector<Pg6ShardDispatchFn>> shard_fns;
    std::vector<uint64_t> host;
};

// Module sizes of the real program's minimal boot set (counts, not contents).
Registry build_registry(bool shard, size_t scale_div) {
    static const struct {
        uint64_t base;
        size_t count;
    } kModules[] = {{0x140001000ULL, 437000}, {0x7FF800001000ULL, 700}, {0x7FF810001000ULL, 300},
                    {0x7FF820001000ULL, 2000}};
    constexpr size_t kShardSize = 1155;
    Registry r;
    Rng rng{0x243F6A8885A308D3ULL};
    for (const auto& m : kModules) {
        const size_t count = std::max<size_t>(1, m.count / scale_div);
        std::vector<uint64_t> module_vas;
        uint64_t va = m.base;
        for (size_t i = 0; i < count; ++i) {
            module_vas.push_back(va);
            va += 16 * (1 + rng.next() % 40);  // 16..640 bytes apart, 16-aligned like MSVC entries
        }
        r.vas.insert(r.vas.end(), module_vas.begin(), module_vas.end());
        if (!shard) {
            std::vector<LiftedFunctionEntry> t;
            for (uint64_t v : module_vas) {
                t.push_back({v, lifted_target});
            }
            r.tables.push_back(std::move(t));
        } else {
            std::vector<Pg6ShardFunctionEntry> e;
            for (size_t i = 0; i < module_vas.size(); ++i) {
                e.push_back({module_vas[i], static_cast<uint32_t>(i / kShardSize),
                             static_cast<uint32_t>(i % kShardSize)});
            }
            r.shard_fns.emplace_back((module_vas.size() + kShardSize - 1) / kShardSize, shard_target);
            r.shard_entries.push_back(std::move(e));
        }
    }
    return r;
}

bool register_all(Registry& r) {
    bool ok = true;
    for (const auto& t : r.tables) {
        ok = ok && register_lifted_functions(t);
    }
    for (size_t i = 0; i < r.shard_entries.size(); ++i) {
        Pg6ShardDispatchTable table{};
        table.version = kPg6ShardDispatchVersion;
        table.shard_count = static_cast<uint32_t>(r.shard_fns[i].size());
        table.entry_count = r.shard_entries[i].size();
        table.shards = r.shard_fns[i].data();
        table.entries = r.shard_entries[i].data();
        ok = ok && register_shard_dispatch_table(table);
    }
    for (int i = 0; i < 64; ++i) {
        r.host.push_back(register_host_callable(host_target, "bench"));
        ok = ok && r.host.back() != 0;
    }
    return ok;
}

struct Pattern {
    std::string name;
    std::vector<uint64_t> targets;
};

std::vector<uint64_t> hot_targets(const Registry& r, size_t n, size_t length, Rng& rng, bool mixed) {
    std::vector<uint64_t> hot;
    for (size_t i = 0; i < n; ++i) {
        hot.push_back(r.vas[rng.next() % r.vas.size()]);
    }
    std::vector<uint64_t> out(length);
    for (size_t i = 0; i < length; ++i) {
        const double u = rng.unit();
        out[i] = (mixed && i % 8 == 7) ? r.host[rng.next() % r.host.size()]
                                       : hot[static_cast<size_t>(u * u * u * static_cast<double>(n))];
    }
    return out;
}

std::vector<Pattern> make_patterns(const Registry& r, size_t length) {
    Rng rng{0x13198A2E03707344ULL};
    std::vector<Pattern> p;
    for (size_t n : {256, 4096, 32768}) {
        p.push_back({"hot-" + std::to_string(n), hot_targets(r, std::min(n, r.vas.size()), length, rng, false)});
    }
    std::vector<uint64_t> uniform(length);
    for (auto& t : uniform) {
        t = r.vas[rng.next() % r.vas.size()];
    }
    p.push_back({"uniform", std::move(uniform)});
    p.push_back({"mixed", hot_targets(r, std::min<size_t>(1024, r.vas.size()), length, rng, true)});
    return p;
}

// Runs `calls` dispatches over the pattern on `threads` threads; returns ns per dispatch
// (wall time / dispatches per thread), or -1 if a dispatch went missing.
double run(const Pattern& p, size_t calls, int threads) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<bool> bad{false};
    std::vector<double> ns(static_cast<size_t>(threads));
    auto body = [&](int w) {
        GuestState state{};
        const size_t n = p.targets.size();
        // Warm up (fills the cache when on) outside the timed loop.
        for (size_t i = 0; i < std::min(n, calls); ++i) {
            __playgta6_indirect_call(&state, p.targets[i]);
        }
        ready.fetch_add(1);
        while (!go.load()) {
            std::this_thread::yield();
        }
        t_calls = 0;
        const auto t0 = std::chrono::steady_clock::now();
        size_t i = static_cast<size_t>(w) * 7919 % n;
        for (size_t c = 0; c < calls; ++c) {
            __playgta6_indirect_call(&state, p.targets[i]);
            if (++i == n) {
                i = 0;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (t_calls != calls) {
            bad.store(true);
        }
        ns[static_cast<size_t>(w)] = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                                     static_cast<double>(calls);
    };
    if (threads == 1) {
        go.store(true);  // on the calling thread (a wasm pthread per run starts slowly)
        body(0);
    } else {
        std::vector<std::thread> workers;
        for (int w = 0; w < threads; ++w) {
            workers.emplace_back(body, w);
        }
        while (ready.load() != threads) {
            std::this_thread::yield();
        }
        go.store(true);
        for (auto& t : workers) {
            t.join();
        }
    }
    if (bad.load()) {
        return -1.0;
    }
    double sum = 0;
    for (double v : ns) {
        sum += v;
    }
    return sum / threads;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = true;
    int threads = 1;
    size_t calls = 4'000'000;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0) {
            quick = false;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--calls") == 0 && i + 1 < argc) {
            calls = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else {
            std::fprintf(stderr, "usage: bench_indirect_call [--full] [--threads N] [--calls N]\n");
            return 64;
        }
    }
    if (quick) {
        reps = 1;
        calls = 20000;
    }
    const size_t pattern_length = quick ? 4096 : size_t{1} << 20;
    int failures = 0;
    std::printf("%-6s %-10s %12s %12s %8s  (ns per dispatch, best of %d x %zu calls, %d thread(s))\n", "mode",
                "pattern", "cache off", "cache on", "speedup", reps, calls, threads);
    for (bool shard : {false, true}) {
        reset_guest_runtime();
        Registry r = build_registry(shard, quick ? 100 : 1);
        if (!register_all(r)) {
            std::fprintf(stderr, "FAIL: registration (%s)\n", shard ? "shard" : "table");
            return 1;
        }
        for (const Pattern& p : make_patterns(r, pattern_length)) {
            // Interleaved repetitions, best of each: robust against a busy machine.
            double off = 1e300;
            double on = 1e300;
            for (int rep = 0; rep < reps; ++rep) {
                set_dispatch_cache_enabled(false);
                const double a = run(p, calls, threads);
                set_dispatch_cache_enabled(true);
                const double b = run(p, calls, threads);
                off = (a < 0 || off < 0) ? -1.0 : std::min(off, a);
                on = (b < 0 || on < 0) ? -1.0 : std::min(on, b);
            }
            if (off < 0 || on < 0) {
                std::fprintf(stderr, "FAIL: dispatches lost (%s %s)\n", shard ? "shard" : "table", p.name.c_str());
                ++failures;
            }
            std::printf("%-6s %-10s %12.2f %12.2f %7.2fx\n", shard ? "shard" : "table", p.name.c_str(), off, on,
                        on > 0 ? off / on : 0.0);
        }
        reset_guest_runtime();
    }
    return failures == 0 ? 0 : 1;
}
