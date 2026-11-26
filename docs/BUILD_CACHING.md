# Build caching: incremental lifting

How a native or browser build avoids redoing work after a recompiler change. The stages and
their caches, outermost first:

| Stage | Cache | Key | Hit means |
|---|---|---|---|
| `recomp` (cargo release build) | `<object cache>-recomp` | every workspace crate's sources + `rustc -vV` | no cargo build |
| `program` (whole) | `<object cache>-lift` (#1475) | recompiler sources, exe, DLLs, shim headers, lift options | no discovery, no lift |
| `program` (discovery, per module) | `<shard lift cache>/discovery` (this page) | discovery sources + `rustc -vV`, module bytes, guest image bytes, seeds | no discovery |
| `program` (per shard) | `<object cache>-lift-shards` (this page) | per shard: its functions' CFGs + callee flag summaries + lifter version + target/memory/dispatch/probes | that shard is not lifted |
| `harness` (per shard object) | `<object cache>` (`build_native_boot.py`) | per shard: IR hash, triple, CPU, `-O`, PIC, LLVM version, retarget code | that shard is not compiled |

The whole-program entry misses on **any** recompiler source change, so discovery always runs again
after one (2 to 6 minutes on the game executable, depending on machine load). After it, every
function used to be lifted again, and any change that altered a function's size (discovery, jump
tables, CFG rules) shifted every later shard boundary, so most shard objects were compiled again.
Two pieces in `recomp/src/pipeline/shard_lift.rs` fix that.

## What a rebuild costs now

Game executable, 24 threads, warm caches unless noted. Numbers are from the PR descriptions (rounded;
discovery varies with load). The recompiler cargo build itself is not included (not measured).

| You changed | Recompiler cache | Discovery | Lift | Shards compiled | Harness (runtime libs + link) | Expected wall time |
|---|---|---|---|---|---|---|
| Runtime / host C++ only (no recompiler or shim input) | hit | skipped | skipped (whole-program hit) | 0 | ~18 s link (core 30-60 s if it changed) | ~20-60 s |
| Shim header / anything outside the recompiler that feeds the whole-program key | hit | 1 s (hit) | ~21 s, 0 functions re-lifted | 0 (compile ~2 s) | ~18 s | ~1 min |
| Lifter (`recomp/src/lift/**`, `cfg.rs`) | cargo rebuild | 1 s (hit) | ~36-40 s, all functions re-lifted | few (7 of 845 for one instruction, ~8 s) | ~18 s | ~1.5-2 min + cargo |
| Discovery rule (`jump_table.rs`, `discover.rs`, `reach.rs`) | cargo rebuild | ~155 s (miss) | ~24 s, only affected functions | 1 of 845 for one jump-table rule (~8 s) | ~18 s | ~3.5-4 min + cargo |
| The game executable / DLLs (or a cold cache) | cargo rebuild | ~147 s | ~48 s | all 845, ~281 s with `-O t` (was 348 s) | ~18 s (runtime libs hidden behind the compile) | ~8-9 min |

Players never compile: `tools/update_prebuilt.ps1` (and `-Web`) builds once into a folder and
`play_native.ps1 -Prebuilt` / `play_local.ps1 -Prebuilt` start it without any build (#1854, #1862).

What produced these numbers:

| PR | Effect |
|---|---|
| #1639 | prebuilt recompiler cache + per-build-dir lock; cold 9m29s -> 1m36s with a lift cache hit |
| #1853 | runtime libs build during the shard compile, PCH for `playgta6_core`: harness step ~72 s -> ~18 s |
| #1878 | `-O t` default pass list: cold full compile 348 s -> 281 s (`-O f`: faster, ~15% larger objects) |
| #1885 | per-shard lift cache + stable shard plan: a jump-table rule change recompiles 1 shard instead of 64; no-op 0 shards |
| #1908 | discovery cache: 147 s -> ~1 s on no-op / lifter change; a discovery rule change still pays ~155 s |
| #1898 / #1900 | `trim_caches.py` and auto-trim on low disk; one cache root (`PLAYGTA6_CACHE_ROOT`) |

## One cache root

Every tool (`build_native_boot.py`, `run_native_boot.py`, `run_browser_boot.py`, `update_prebuilt.ps1`,
`play_native`/`play_local`, `trim_caches.py`) derives its cache dirs from one root, `tools/cache_root.py`:
`PLAYGTA6_CACHE_ROOT` (a directory, or `auto` = `<drive with most free space>:\playgta6-cache`), else
`%LOCALAPPDATA%\playgta6\cache`. Below it: `objcache`, `objcache-lift`, `objcache-lift-shards`,
`objcache-recomp`, `objcache-targets` and `wasm_obj_cache` (`PLAYGTA6_WASM_CACHE`, `off` = per build dir).
Explicit `--cache-dir` / `PLAYGTA6_OBJ_CACHE` / per-cache env vars still win. Earlier locations
(`%LOCALAPPDATA%\playgta6\objcache*`, `<drive>:\pg6nb-objcache*`) are never moved or deleted automatically:
a one-time hint names them; `python tools/trim_caches.py` sizes them (it still scans them), and the
`%LOCALAPPDATA%` ones can be moved into the root by name. `PLAYGTA6_NO_CACHE_HINT=1` silences the hint.

## Stable shard plan

The old plan (`plan_shards`) filled shards greedily in entry order up to 65536 instructions. A
function that grew or shrank by one instruction moved every later boundary, so every later shard
got different functions and different IR.

`plan_shards_stable` keeps the two budgets (`--shard-size` functions, `--shard-instructions`
instructions) and also cuts before every *anchor*: a function whose SplitMix64 hash of its entry
VA falls under `weight / spacing` (spacing = half the instruction budget), i.e. each instruction is
an anchor with probability 1/32768. Whether a function is an anchor depends on that function
alone, and a cut at an anchor does not depend on where the shard started, so two plans that differ
somewhere agree again from the first anchor after the difference. Changing, adding or removing a
function moves only the boundaries between the anchors around it: one shard usually, at most a few
when the gap between two anchors is longer than the budget (then it is split by the budget).

Cost: shards average about half the budget, so the game executable has 837 shards instead of 379.
Each compiles faster (the `opt`/`llc` cost grows faster than linearly with shard size, see
`recomp/src/pipeline.rs` "Sharding"), and a change recompiles less.

## Shard lift cache (`build-program --lift-cache <dir>`)

Each shard's IR is stored under a 128-bit key (FNV-1a 128):

- per function, `function_key`: the `Debug` text of its whole CFG (every block and decoded
  instruction with address, length and operands, every edge including resolved jump-table cases
  and tail calls, call sites, resume targets) plus the flag/state summaries (`FlagSummaries`) of the
  function and of every address it calls or jumps to. These are the only inputs of
  `lift::lift_function_in` apart from the memory model; a discovery or jump-table change therefore
  changes exactly the keys of the functions whose CFG it changed, plus their direct callers when a
  callee's summary changed;
- per shard, `shard_key`: the function keys in order, which functions are probed
  (`--probe-functions`) and which keep external linkage (`--indirect-dispatch shard`), the target,
  the memory model, the shard dispatcher symbol, and `LIFT_CODEGEN`.

`LIFT_CODEGEN` is computed by `recomp/build.rs` from the sources that decide how a CFG becomes IR:
`recomp/src/lift/**` (without `tests/`), `src/cfg.rs`, `src/pipeline/shard_lift.rs`, `build.rs`,
the crate manifest and `Cargo.lock` (iced-x86). A change to any of them re-lifts every shard
(correctness first: no attempt to guess which instructions a lifter edit affects). Lifting is the
cheap part (about 50 s for all 435k functions on 16 threads); what matters is that the shards
whose functions do not use the changed code come out byte-identical and hit the object cache.

Entries are `<key>.ll` plus `<key>.meta` (size, FNV-1a 64, unsupported instructions), written
through temporaries and renamed, `.meta` last; a hit needs both and matching sizes, so a damaged or
half-written entry is a miss. Hits are hard-linked into `lift/` (copied across volumes) and
touched. The outputs (`build-manifest.json`, `report.txt`, every `.ll`) are byte-identical with
and without the cache (tested in `pipeline::tests::shard_lift_cache_*`).

`tools/run_native_boot.py` and `tools/run_browser_boot.py` pass `--lift-cache` whenever the
whole-program entry misses: `<lift cache>-shards`, or `PLAYGTA6_SHARD_LIFT_CACHE=<dir>|off`. After
the lift the cache is trimmed least recently used first to `PLAYGTA6_SHARD_LIFT_CACHE_GB` (default
24, about three full sets; entries used in the last hour are kept). Files shared by hard link with
a program directory or a whole-program entry take no extra space.

`build-program` prints `lift cache: R/N shards reused, L of F functions lifted` per module.

## Trimming every cache (`tools/trim_caches.py`)

`python tools/trim_caches.py` lists the object, lift, shard, recompiler, indirect-target, wasm object and Dawn blob caches
with sizes and what each would free to its budget (LRU by atime/mtime); it is a dry run until `--apply`. It only looks at
the known cache roots, and keeps the newest entry of each cache, anything used in the last 6 h, and anything holding a
fresh lock or temporary. The builders call it on their own (log line `disk low: ...`) when free disk is below
`PLAYGTA6_MIN_FREE_GB` (default 20, 0 disables).

## Discovery cache (`build-program --discovery-cache <dir>|off`)

`recomp/src/pipeline/discovery_cache.rs` stores the whole discovery stage of a module (the
`refine::discover_refined` fixpoint: `discover::Analysis`, the code-pointer outcome, the final
entry points and no-return slots) as `<module>-<key>.disc`. The default directory is
`<lift cache>/discovery` whenever `--lift-cache` is given, so `tools/run_native_boot.py` and
`tools/run_browser_boot.py` use it without new options (the shard trim and `tools/trim_caches.py` leave the subdirectory alone; the cache keeps
its own 3 newest entries per module).

Key (FNV-1a 128):

- `DISCOVERY_CODE` from `recomp/build.rs`: every non-test source of the crate except
  `src/lift/**` and `src/pipeline/shard_lift.rs` (they run after discovery), plus `build.rs`,
  the manifest, `Cargo.lock` (iced-x86) and `rustc -vV`;
- the module bytes as `seed` reads them (rebased), the guest image bytes after the IAT pre-fill
  (the refinement reads base relocations from them), the image options, and the seeded entry
  points and no-return slots (`.pdata`, EH, exports, `--entry`);
- the trace-seed re-run (#1654) has its own entry: the first stage's key plus the accepted seed
  entries, the only new input of that run. A seed file that accepts the same entries hits; a
  different one re-runs only the second stage.

It is one entry per module, not per function, on purpose. Discovery is a single global fixpoint:
jump-table slices cross into predecessor blocks of other functions, a resolved table adds leaders
that split blocks elsewhere, tail calls depend on which addresses are entries, and the pointer-slot,
vtable and code-pointer rounds add entries from the whole block map. A per-function fact is only
valid together with every cross-function fact that produced it, which is the whole analysis, and
a change to any discovery rule (`jump_table.rs`, `discover.rs`, `reach.rs`, ...) can change any
function, so no fact computed by the old code can be reused without deriving it again. Reusing
parts of the fixpoint after a rule change would need the fixpoint itself to be incremental (inside
`discover.rs`), not a cache around it.

Instructions are stored as lengths and decoded again from the image bytes on load, on all
threads (a decode that does not reproduce a stored length is a miss). The header carries the key,
the payload length and an FNV-1a 64 of the payload; a truncated or damaged entry is a miss.
Writes go through a temporary and a rename; the newest 3 entries per module are kept (a hit
touches its entry). The game executable's entry is 126 MB.

`--verify-discovery-cache` runs discovery anyway, compares it with the stored entry (every field,
every instruction with `eq_all_bits`, since `Instruction`'s `PartialEq` ignores IP and length) and
with its own encode/decode round trip, and fails the build on a difference. Verified on the game
executable (435,719 functions, 23.5M instructions): identical. The pipeline tests check that the
outputs are byte-identical cold, warm, verified and after damaged entries
(`pipeline::tests::discovery_cache_*`).

Measured on the game executable's `main` module (24 threads; discovery alone, no lift):

| Change | Discovery before | Discovery after |
|---|---|---|
| none (rebuild after a change outside the recompiler, e.g. a shim header) | 147 s | 1.1 s (hit) |
| one lifter instruction (`popcnt` ZF `icmp eq x, 0` -> `icmp ult x, 1`) | 147 s | 0.9 s (hit) |
| one jump-table rule (`MAX_INFERRED_ENTRIES` 4096 -> 256) | 147 s | 155 s (miss, +~10 s to store) |

A miss costs the uncached time plus the encode and write (about 10 to 20 s, within the run to run
noise of discovery itself).

## Measurements

Game executable `main` module: 435,719 functions, 23.5M instructions. 24 hardware threads,
`build-program --target native --jobs 16`, shard objects compiled by `build_native_boot.py` at
16 jobs. Discovery times vary between runs with machine load (120 to 380 s for the same input),
so the lift stage is listed separately. "Shards to compile" counts shard IR files whose content
was not in the previous build, i.e. the object cache misses; the compile time uses the measured
mean per-shard `opt`+`llc` time (`build_native_boot.py --sample`: 16.1 s for the greedy plan's
~62k-instruction shards, 8.0 s for the stable plan's ~28k), divided over 16 jobs.

| Change | Plan | Functions lifted | Lift stage | Shards to compile | IR to compile | Compile (est.) |
|---|---|---|---|---|---|---|
| none (repeat after a recompiler change elsewhere) | before | 435,719 | 38 s | 0 / 384 | 0 | 0 |
| | after | 0 | 21 s | 0 / 845 | 0 | 0 |
| jump-table rule (`MAX_INFERRED_ENTRIES` 4096 -> 256) | before | 435,719 | 40 s | **64** / 384 | 1299 MiB | ~65 s |
| | after | 226 | 24 s | **1** / 845 | 4 MiB | ~8 s |
| one lifter instruction (`popcnt` ZF: `icmp eq x, 0` -> `icmp ult x, 1`) | before | 435,719 | 40 s | 7 / 384 | 148 MiB | ~16 s (one wave) |
| | after | 435,719 | 36 s | 7 / 845 | 59 MiB | ~8 s (one wave) |
| full build from scratch | before | 435,719 | 38 s | 384 | 7.4 GiB | ~381 s |
| | after | 435,719 | 48 s | 845 | 7.4 GiB | ~404 s |

What this shows:

- Discovery and CFG changes are where the old plan hurt: one function whose size changed moved
  every later boundary (64 shards, 1.3 GiB of IR recompiled for one rule). With anchors it is one
  shard, and only that shard's 226 functions are lifted.
- A lifter edit already recompiled only the shards whose IR changed (the object cache is keyed by
  IR content, and a lifter edit does not change function sizes); here the gain is smaller shards
  (less IR per changed shard). Every function is lifted again (the codegen version is crate-wide),
  which costs under a minute.
- The stable plan costs about 6% more compile time on a from-scratch build (more, smaller shards);
  it pays for itself with the first discovery change.

## Limits and next steps

- Discovery is cached per module, but any change to the discovery sources (the jump-table,
  reach, CFG or refinement rules, and conservatively every other non-lifter source such as the
  reports) re-runs it in full. Narrower keys (per source file actually reached) or an incremental
  fixpoint would be the next step.
- Lifter edits re-lift everything (cheap) because the codegen version is crate-wide; per-module
  versions (for example SIMD lowering only for functions with vector instructions) would need the
  lifter to report which modules a function used.
- In `--indirect-dispatch shard` (browser) mode the dispatcher symbol carries the shard index, and
  a shard's external-linkage set depends on its callers elsewhere, so a shard inserted early renames
  the dispatchers after it; native (table dispatch) shards do not carry their index.
