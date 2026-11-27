# Incremental discovery: design

Status: Phase 0 implemented (0a #1962, 0b and 0c below); Phases 1-3 are proposals. Owner of the files involved (`recomp/src/discover.rs`,
`reach.rs`, `jump_table.rs`, `cfg.rs`, `eh.rs`): Marc. This page only describes the scheme and the
numbers behind it; see "Rollout" for which step touches which file.

## Problem

After #1885 (per-shard lift cache) and #1908 (per-module discovery cache), a recompiler change
that cannot affect discovery costs about one second of discovery. A change to a discovery rule
(`jump_table.rs`, `reach.rs`, `discover.rs`) misses the discovery cache and pays the whole global
fixpoint again: ~147-155 s on the game executable (24 threads, load dependent), while the lift
and compile after it already touch only the affected shards (one jump-table rule change: 226
functions re-lifted, 1 of 845 shards compiled). Discovery is now most of the wall time of a
discovery-rule iteration (`docs/BUILD_CACHING.md`, "What a rebuild costs now").

The goal: a discovery-rule change re-derives only what the change can affect, gives a result that
is bit-identical to a full run, and falls back to the full run whenever that cannot be shown.

## Where the time goes (measured)

Game executable, `main` at 867ffc27, one `discover()` call (the first refinement round, 396,195
functions) with temporary timers around each phase (scratch build, not committed). Absolute times
depend on machine load (the same round took 46.5 s on one run and 58.5 s on another); the shares
are what matters.

| Phase | Calls | Time | Share |
|---|---:|---:|---:|
| `predecessors()` (whole-map rebuild, inside the phases below) | 30 | 36.5 s | 62 % |
| `run()` (block decoding worklist) | 17 | 4.1 s | 7 % |
| `resolve_jump_tables` | 17 | 18.3 s | |
| `revise_jump_tables` | 7 | 10.2 s | |
| `requeue_changed_jumps` | 5 | 7.3 s | |
| late resume targets + pointer slots + vtables | 9 | 10.7 s | |
| `verify_jump_tables` | 1+ | 2.5 s | |
| `classify_unresolved_jumps` | 1 | 1.5 s | |
| relocations | 1 | 1.1 s | |
| `into_functions` (membership walks) | 1 | 2.8 s | |
| **total** | | **58.5 s** | |

Jump-table matching itself (the resolve/revise/requeue/seed/verify/classify rows minus their
predecessor rebuilds) is about 14 s, roughly 24 %.

`discover_refined` then runs `discover()` from scratch once per code-pointer refinement round:
**4 rounds** on the game executable (`build-manifest.json`: `discovery.rounds = 4`), each on a
slightly larger seed set. A full stage therefore pays about four times the table above.

Two things follow:

1. About 60 % of every round is rebuilding the same `BTreeMap` of predecessors over ~4M blocks 30
   times, and three of the four rounds redo the work of the round before. Both can be removed
   without any incremental bookkeeping (Phase 0 below), and that also speeds up cold runs and
   changes to the executable.
2. The part a jump-table rule actually decides is small (next section). That makes rule-level
   incrementality worth doing after Phase 0.

## After Phase 0 (measured)

Game executable, cold refined discovery (no cache, 4 rounds), per-phase timers (what
`build-program --timings` prints; same machine, one run each, absolute times move by 5-10 s
with load):

| Phase | after 0a | after 0b |
|---|---:|---:|
| decode (`run`) | 24.7 s | 11.2 s |
| jump tables (resolve, revise, requeue, verify, classify) | 2.1 s | 1.4 s |
| late resume targets + pointer slots + vtables | 17.1 s | 11.0 s |
| relocations | 3.5 s | 3.8 s |
| functions (membership walks) | 11.1 s | 10.1 s |
| refinement (code-pointer classification) | 2.4 s | 2.4 s |
| rounds (s) | 13.2, 14.9, 15.3, 15.3 | 12.8, 15.1, 4.9, 4.7 |
| **total** | **67.3 s** | **46.2 s** |

(Before 0a: 312 s, measured on a more loaded run in #1962; 0a gave 99.5 s on that run.)

The whole encoded result (`discovery_cache::encode`, 126 MB) is byte-identical to `main` after
each step.

Phase 0b keeps the `Discovery` state between refinement rounds (`discover::Session`): a round
moves its blocks into the analysis, the refinement classifies the code pointers on it, and
`Session::reclaim` takes the blocks back (undoing the membership walk's tail-call rewrites and
restoring the tables verification demoted, so the state is the pre-verification fixpoint again)
before the accepted entries are added. A round starts over instead when the previous round found a
no-return thunk that decoded code calls directly (`no_return` decides where a block ends at a
direct call, so those blocks would have to be decoded again), or when the continued round hit an
event a fresh run may order differently (`Session::diverged`: a resolved table revised to one
that drops targets or no longer matching, a retry or revision cap, a pointer-slot seed on a jump
that later resolved as a table). On the game executable round 1 finds called no-return thunks, so
round 2 starts over and rounds 3-4 continue (none diverged). Unit tests check every `discover()`
call against a continued session (the entries split in two, and all added to an empty session)
and `discover_refined_rounds` continued against fresh rounds.

What is left is per-round full scans: decoding is serial, and the late resume targets, vtable /
pointer-slot seeds, relocations and membership walks each visit all ~4M blocks once or more per
round. Making those visit only blocks changed since their last pass, and continuing round 2
through a no-return addition (retracting the decoded tails of the affected calls), are the
remaining Phase 0-style wins; neither needs the match records of Phase 1.

## What a discovery rule can reach (measured)

Same executable, refined discovery (435,719 functions, 4,001,806 blocks, 23,481,703
instructions), computed from the final `Analysis` by a scratch example:

| Quantity | Value | Share of total |
|---|---:|---:|
| Indirect-`jmp` sites (resolved 3,432 + unresolved 8,318) | 11,823 | |
| Functions containing at least one such site | 10,892 | 2.5 % of functions |
| Their blocks / instructions | 212,476 / 1,052,289 | 5.3 % / 4.5 % |
| Functions containing a resolved table | 2,855 | 0.66 % |
| Their blocks / instructions | 155,076 / 751,025 | 3.9 % / 3.2 % |
| Blocks of those functions reachable only through a table edge | 97,364 | 2.4 % |
| Functions containing an inferred table (216 tables) | 185 | 0.04 % |
| Blocks shared by more than one function | 2,296 | 0.06 % |
| Sites per function | median 1, p99 2, max 158 | |
| Backward read-set of a site, all sites (blocks in its owner that reach it) | median 3, p90 16, p99 92, max 1,434; sum 111,374 | 2.8 % of blocks |
| Backward read-set, resolved tables only | median 4, p90 35, p99 277, max 1,434 | |

The read-set column approximates what `reach::reaching_facts` / `reaching_constant_sets` see (the
backward subgraph from the slice's first block, stopping at entries, capped at 4,096 blocks): even
the largest is far below the cap, and all 11,823 read-sets together are 2.8 % of the block map.

Footprint of recent discovery-rule changes on the same executable (from the PR descriptions,
measured on the real exe):

| Change | What it changed in discovery | Functions whose CFG changed |
|---|---|---|
| #1658 inferred extent ends at the first entry inside an instruction | unresolved jumps 8330 -> 8320 | the dispatching function and a neighbour (1-2) |
| #1662 inferred extent clipped at the next known table base | inferred tables only | at most the 185 functions with an inferred table; in practice a handful |
| #1670 retry an unresolved switch when its slice changes; sibling copies | unresolved 8322 -> 8319 | 1 (plus the 3 retried sites) |
| #1689 table base is one of a few constants | tables 3376 -> 3377 | 1 (7 -> 295 blocks) |
| `MAX_INFERRED_ENTRIES` 4096 -> 256 (the #1885/#1908 benchmark) | inferred tables shrink | 226 functions re-lifted (CFG or callee summary changed) |

Every one of them changed fewer than 0.1 % of the functions, and yet each paid the full ~150 s.

## The fixpoint today: what each rule reads and writes

State of `Discovery` (see `discover.rs`): `blocks` (global block map), `leaders`, `entries`,
`jump_tables`, `failed_jumps` + `jump_retries`, `table_revisions`, `resume_roots` /
`return_sites` / `deferred_resume_targets`, `pointer_slots` / `slot_checked`,
`scanned_data_refs` / `vtable_scanned_blocks`, `no_return`. Inputs fixed for a module: the image
bytes, `.pdata` (`function_table`), EH metadata (`image.eh`), `data_relocs`, seeds.

| Rule (function) | Reads | Writes | Locality |
|---|---|---|---|
| R1 decode (`run`, `decode_block`, `classify`, `branch`, `split`) | image bytes at the leader; `leaders` (block end at the next leader); `no_return`; `entries`; `.pdata` (cross-function `jmp`/`jcc` tail-call seeding) | `blocks`, `leaders`, `entries`, `pending_jumps`, diagnostics | local to the leader, except splits (rewrite the containing block) and new entries |
| R2 match (`resolve_jump_tables` -> `match_jump_table_raw` -> `jump_table::resolve_in`, `reach::*`) | the jump's sole-predecessor chain (<= 4 blocks); if it stops at a merge, the backward subgraph to the next entry / resume target / return site (`reaching_facts`, `reaching_constant_sets`); `leaf_extent` = neighbouring `entries`; table bytes; `.pdata` owner of each target | one `JumpTable` per site; its block's successors; targets become leaders | read-set = trace + backward subgraph + two neighbouring entries |
| R2' clip (`clip_inferred`) | `table_bases()` of **every** resolved table | truncates inferred tables | global, but only the bases inside `[table, table + n*stride)` matter |
| R3 requeue (`requeue_changed_jumps`, `jump_shape`) | `(first block, trace length, incoming edges)` of each failed jump; retry counters | `pending_jumps`, `jump_retries` | local; already a read-set fingerprint |
| R4 revise (`revise_jump_tables`) | as R2 for every resolved table; `table_revisions` | replaces tables, adds leaders | local per table, capped at 8 revisions |
| R5 late resume targets (`add_late_resume_targets`) | every block's calls (setjmp thunks) and RIP-relative `lea`s; `entries`; `.pdata` | `return_sites`, `resume_roots`, leaders | local per block |
| R6 pointer slots (`seed_pointer_slots`) | slices of unresolved jumps (as R2), `data_relocs`, pointer bytes | `pointer_slots`, `entries` | local per site |
| R7 vtables / callbacks (`seed_vtable_callbacks`) | RIP-relative operands of new blocks, `data_relocs`, `.rdata` pointers | `entries` | local per block; `vtable_scanned_blocks` already memoizes by `(start, end)` |
| R8 verify (`verify_jump_tables`) | as R2 for every table, on the final CFG | **removes** tables (demotes to `IndirectJump`) | local, but non-monotone |
| R9 classify, relocations | final CFG | `unresolved_jumps`, `relocations` | local per site / block |
| R10 membership (`into_functions`) | final `blocks`, `entries`, `resume_roots`, `return_sites` | `functions` (blocks copied per function, tail calls) | per function; reads the global entry set |
| R11 refinement (`pipeline::refine::discover_refined`) | the whole `Analysis` (lea pointers, base-reloc pointers, jump-table data ranges, no-return thunks) | `image.entry_points`, `image.no_return`, then **reruns R1-R10 from scratch** | global; 4 rounds |
| EH (`eh.rs`) | PE bytes only | `image.eh` seeds before discovery | input-only, independent of every other rule |

Why #1908 had to cache per module: R2 reads blocks that R1 decoded from leaders that other
functions' R2 results created, R2' reads all table bases, R10 reads the global entry set, and R11
reruns everything. A per-function cache key would have to contain all of that.

The way out is that these cross-function reads are **narrow**: each R2 evaluation reads a known,
small set of blocks (median 3), and only R2' and R10 read global sets, of which they use a
small window.

## Design

### 1. The state that matters is small; the block map is derived

Given the image, `no_return` and the final sets `leaders`, `entries`, `resume_roots`,
`return_sites` and `jump_tables`, the final block map is fixed: each block runs from its leader to
the first terminator or the next leader, its successors come from its last instruction, or from
its table when one is resolved. Decoding is 7 % of a round. So the incremental scheme does not
need to diff blocks: it diffs the *decisions* (tables, entries, roots) and re-decodes only around
the leaders they add or remove.

Every leader and entry gets a **provenance**: the set of reasons it exists (`Seed`, `Branch(from
block)`, `Table(site)`, `Resume(owner)`, `ReturnSite(call)`, `PointerSlot(site)`, `Vtable(slot)`,
`CodePointer(round)`). This is the "support" of Delete-and-Rederive (DRed) incremental view
maintenance: a leader is retracted only when its last reason goes.

### 2. Rule families with their own version hashes

`DISCOVERY_CODE` (one hash of every non-lifter source, `recomp/build.rs`) becomes a few family
hashes, each over the files that implement it:

| Family | Rules | Files (target layout) | On change |
|---|---|---|---|
| `EH` | EH metadata | `eh.rs`, `pdata.rs` | recompute the seeds (input-only, ~0 s); a seed diff is replayed as additions/retractions |
| `DECODE` | R1, R3, R5-R7, R10, R11 and the fixpoint driver | `discover.rs` (driver), `cfg.rs`, `code_image.rs`, `pointer_slot.rs`, `reloc.rs`, `pipeline/refine.rs` | full run (fallback) |
| `MATCH` | R2, R2', R4, R8, R9 | `jump_table.rs`, `reach.rs`, and the match helpers now in `discover.rs` (`jump_trace`, `reaching_prefix`, `match_*`, `clip_inferred`, `classify_unresolved_jumps`) | incremental (section 4) |

This needs the match helpers moved out of `discover.rs` into their own file (for example
`recomp/src/discover/matcher.rs`) so a matcher edit does not change the `DECODE` hash. Of the
recent rule PRs, #1658, #1689, #1781 (jump_table part) and the `MAX_INFERRED_ENTRIES` change are
pure `MATCH` changes; #1662 (clip) and #1635/#1670 reach parts are `MATCH`; the #1670 requeue
change and #1785's vtable/tail-call seeding are `DECODE` changes and would still pay the full run
(Phase 0 makes that run ~3-4x cheaper).

### 3. Per-site match memo with read-sets

For every R2/R4/R8/R9 evaluation the cache stores a **match record**:

- `site`, the matcher's output (`Option<JumpTable>`, or the `UnresolvedJump` class), and the
  `MATCH` hash it was computed with;
- its **read-set**: the start VAs of the trace blocks and of the backward subgraph that
  `reaching_*` walked, with a 64-bit content hash per block (instruction bytes, successors,
  predecessor list, entry / resume / return-site flags), plus `leaf_extent` and the table-base
  window that R2' looked at (`bases ∩ [table, end)`);
- a fingerprint `H(read-set contents)`.

The read-sets are already computed during matching (`collect_backward_subgraph_bounded` builds
the block set; `jump_shape` is a coarse version of the fingerprint), so recording them costs the
size of the set: 111k block ids for all sites, a few MB in the cache entry.

An inverted index `block -> sites whose read-set contains it` makes dirty propagation `O(changed
blocks)`.

### 4. A `MATCH` change: re-match, diff, repair

Input: the cached final state `S0` (computed with `MATCH` = old, `DECODE` = current) and its match
records. New matcher `M1`.

1. **Re-match on the old CFG.** Evaluate `M1` on every site of `S0` (11,823 sites; their
   read-sets together are 2.8 % of the blocks, so this is the ~14 s of matching of one round
   spread over far less input: about 1-3 s on 24 threads), using `S0`'s block map and a
   predecessor map built once.
2. **Diff.** `D` = sites where `M1`'s output differs from `S0`'s. If `D` is empty, `S0` is the
   result (see soundness, case A). This is the common case for a rule that does not fire on the
   real exe, and turns a ~150 s miss into ~3 s.
3. **Repair.** Otherwise apply the differences as a DRed step, limited to the functions that own
   a site in `D`:
   - *delete*: drop the old tables of `D`; drop the `Table(site)` reason of each old target and
     retract leaders/entries whose support is now empty, transitively through `Branch(from)`
     reasons of blocks that are no longer decoded; un-split blocks that only existed because of
     a retracted leader;
   - *insert*: install the new tables of `D`, add their targets as leaders;
   - *rederive*: run the normal fixpoint loop (`run` / resolve / revise / requeue / seeds) with
     `pending_jumps` = `D` ∪ every site whose read-set contains a block that was created, removed,
     split or whose successors or predecessors changed (inverted index). Every site outside that
     set keeps its memoized output without being evaluated, because its inputs are bit-identical.
   - repeat until no site is dirty, then R8 (verify) and R9 (classify) on dirty sites only, then
     R10 for the functions whose block set changed (membership walk is per function).
4. **Refinement (R11).** Re-run the code-pointer classification on the repaired analysis. If it
   accepts the same candidate set as `S0`'s outcome (it reads lea pointers, base-reloc pointers
   and jump-table data ranges; only the last can move, and only by the tables in `D`), done.
   Otherwise fall back (Phase 3 makes this incremental too).
5. Store the new state and match records under the new key.

### 5. Soundness

"Sound" here means: the incremental result equals what a full run with the new code computes,
bit for bit (the same standard as `--verify-discovery-cache`). Argument:

- **Case A (`D` empty).** `S0` is a fixpoint of `DECODE` (unchanged) and of the old matcher. The
  new matcher gives the same output on every site evaluated on `S0`'s final CFG, so `S0` is also
  a fixpoint of the new rules: no table changes, no new leaders, verify keeps every table. That it
  is *the* fixpoint a full run reaches needs one more condition, because the full run passes
  through intermediate CFGs where `M1` might fire differently than `M0` did (for example a table
  resolved early and later revised). So case A also requires that no site hit a cap
  (`MAX_TABLE_REVISIONS` revisions or retries) and that no table was demoted by R8 in `S0`:
  under those conditions the rounds only add leaders and tables (monotone), the result is the
  least fixpoint over the seeds, and the least fixpoint is unique. Sites that did hit a cap or a
  demotion are recorded in the cache; if one of them is in a changed function or its read-set
  contains one, fall back.
- **Case B (repair).** Every rule outside `MATCH` is a function of its read-set (R1 of the bytes
  and leaders around a leader, R5-R7 of a block, R10 of a function's blocks and the entry set);
  every `MATCH` evaluation is a function of its recorded read-set. The repair re-evaluates every
  evaluation whose read-set changed, transitively, so every surviving memoized output is the
  output the new rules give on the final state. The final state is then a fixpoint of the new
  rules. Uniqueness again needs the monotone condition of case A for the dirty region (no cap hit,
  no demotion, no retraction that a later insertion re-adds in a different order); otherwise
  fall back.
- **Global reads.** R2' reads only the table bases inside a table's extent; recording that window
  in the read-set makes it local. `leaf_extent` reads the two neighbouring entries: recorded. R10
  reads the entry set to classify tail calls: a changed entry set re-runs R10 for every function
  with an edge to the changed entries (blocks have their successor lists; the inverted index
  covers them).
- **Determinism.** The full run is already independent of thread count and visiting order
  (`par_map`, sorted worklists, per-round predecessor snapshots); the repair uses the same
  functions on the same snapshots, so it inherits that.
- **What is assumed and checked, not proven.** The match helpers are pure functions of their
  read-set (no hidden global reads). That is true today by inspection; a debug mode asserts it by
  recomputing every memoized record from scratch (`--verify-discovery-cache` already compares
  every field against a full run and becomes the gate for this).

### 6. Fallback to the full run

Any of the following runs the full fixpoint (today's behaviour) and stores fresh records:

- the `DECODE` hash changed, the module bytes / guest image / options / seeds changed (except an
  `EH`-only seed diff, Phase 3), the cache format changed, or no records exist;
- the dirty set exceeds a budget (for example 5 % of the sites or 2 % of the blocks: repair
  bookkeeping would cost more than the run);
- a cap or demotion in the dirty region (soundness, case A/B);
- the refinement accepts a different candidate set (until Phase 3);
- `--verify-discovery-cache` mode (always computes the full run and compares; CI and a weekly
  local check run it so the incremental path is continuously checked against the real thing).

### 7. Expected savings

Discovery stage on the game executable, rough, 24 threads:

| Change | Today | Phase 0 (no incrementality) | Phase 2 (`MATCH` incremental) |
|---|---:|---:|---:|
| no discovery source changed | ~1 s (hit) | ~1 s | ~1 s |
| `MATCH` rule that changes nothing on the exe | ~150 s | ~35-50 s | ~3 s (load + re-match) |
| `MATCH` rule changing 1-10 functions (#1658, #1670, #1689) | ~150 s | ~35-50 s | ~4-6 s (re-match + repair of a few functions + R10/R11 on them) |
| `MATCH` rule changing ~200 functions (`MAX_INFERRED_ENTRIES`) | ~155 s | ~35-50 s | ~6-10 s |
| `DECODE` rule (driver, seeding, refine) | ~150 s | ~35-50 s | ~35-50 s (fallback) |
| executable / cold | ~147 s | ~35-50 s | ~35-50 s |

Phase 0 estimate: removing the 30 predecessor rebuilds saves ~60 % of a round; warm-started
refinement rounds 2-4 cost only the new entries' share instead of a full round, so the stage goes
from ~4 rounds to ~1.2 rounds of ~40 % of today's round. The range covers the load variance
measured above. The end-to-end iteration for a jump-table rule then becomes cargo + ~5 s discovery
+ ~24 s lift + ~8 s compile + ~18 s link instead of cargo + ~155 s + the same.

## Rollout

Each phase is a separate PR, ships behind the existing `--discovery-cache` path, and is verified on
the real executable with `--verify-discovery-cache` (identical result) before it is the default.

- **Phase 0 - cheaper full run, no incremental state** (`discover.rs`, `pipeline/refine.rs`):
  a. maintain the predecessor map incrementally (update it in `decode_block`, `split` and wherever
     a block's successors are rewritten) instead of `predecessors()` per phase, or at least build
     it once per fixpoint iteration and share it between resolve / revise / requeue / seeds;
  b. keep the `Discovery` state between refinement rounds and add the accepted entries to it
     instead of re-running `discover()` from scratch (needs the pre-verify state kept, and the
     equality to a fresh run checked on the exe, because R8 demotion is not monotone);
  c. per-phase timers: `build-program --timings` prints them per module, and the discovery cache
     stores them next to each entry (`<module>-<key>.disc-meta.json`; a hit prints the stored
     ones). Not in `build-manifest.json`, which stays deterministic.
  Benefits every discovery miss including cold runs. Expected ~150 s -> ~35-50 s.
- **Phase 1 - records only** (`discover.rs`, `reach.rs` return the subgraph it walked; cache
  format 2): record provenance, match records and read-sets; a `build-program
  --discovery-dry-rematch` diagnostic that re-matches every site of the cached state with the
  current matcher and prints `D` (how many sites / functions a rule change touches) without using
  it. Useful on its own when developing a rule, and measures the real dirty sets.
- **Phase 2 - `MATCH` incremental** (`recomp/build.rs` family hashes, the matcher moved to its own
  file, `pipeline/discovery_cache.rs` repair driver): case A first (`D` empty -> reuse), then the
  repair for small `D`, fallback otherwise. Gate: `--verify-discovery-cache` identical on the exe
  for the last ten `MATCH` PRs replayed (#1635, #1658, #1662, #1670, #1689, #1781, the
  `MAX_INFERRED_ENTRIES` change, ...).
- **Phase 3 - seeds and refinement**: `EH` seed diffs and code-pointer rounds replayed as
  additions (monotone) with the same repair; `DECODE` stays a full run.

## Open questions

- Is moving the match helpers out of `discover.rs` acceptable, or should the `MATCH` hash cover
  `discover.rs` too (then only `jump_table.rs` / `reach.rs` edits are incremental, which still
  covers most recent rule PRs)?
- Phase 0b: does any test or report rely on each refinement round starting from a fresh state?
- Who implements: Marc in his files, or us in agreed files (`pipeline/discovery_cache.rs`,
  `pipeline/refine.rs`, `build.rs`, a new `discover/matcher.rs`) with small hooks in
  `discover.rs` / `reach.rs` for him to review.
