# Ray tracing on WebGPU without hardware ray tracing

Status: design, 2026-11-26. Covers backlog items 31 (c) and 36. Nothing here is implemented yet.
Owners: `runtime/gpu/` and `web/` (this track); `shaders/` (Marc) gets a contract only, section 6.

Sources: our own decoder and runtime, plus what two public VIctory forks *do* (the avatarmanz
7900XTX fork, which emulates the ray test, and the Belobug AMD fork, which has no rays and paints
a constant light). Only their behaviour and measurements are used here; no code is taken from them.

## 1. What the title actually does

The title does **not** use DXR. `CreateStateObject` already fails with `E_NOTIMPL` and
`OPTIONS5.RaytracingTier` is 0 (`runtime/gpu/d3d12_xs_xbox.h`, "Raytracing"). The title copes, and
that does not change under this design. Its ray tracing is built from three parts:

1. **Traversal is the title's own compute code.** About 64 trace programs (shadows, screen
   probes, irradiance cache, reflections, the "V-buffer" trace) walk the trees themselves. Each
   thread keeps a small stack in workgroup memory, and the programs load nodes through raw 40-bit
   GPU addresses, not through descriptors.
2. **Only the per-node test is a hardware instruction**, on the MIMG encoding:
   - `image_bvh_intersect_ray` (GFX10.3 opcode **0xE6**). It takes a 32-bit node pointer and
     11 address VGPRs (8 with A16).
   - `image_bvh64_intersect_ray` (opcode **0xE7**). It takes a 64-bit node pointer and 12 VGPRs
     (9 with A16).
   - The VGPRs hold, in order: the node pointer, the ray extent (t_max), the origin xyz, the
     direction xyz and the inverse direction xyz. With A16, direction and inverse direction are
     packed halves.
   - Both return 4 dwords.
   - The resource operand is a 128-bit "BVH descriptor": base address in 256-byte units in
     dword 0 and the low byte of dword 1, "sort boxes" in bit 31 of dword 1, "return barycentrics"
     in bit 24 of dword 3, and type 8 in bits 31:28 of dword 3.
   - For the 64-bit form, the programs build that descriptor in SGPRs with base 0 and add
     `address >> 3` to the pointer themselves.
3. **The structures sit in the console's native node format in title GPU memory**:
   - The static world's structures ship prebuilt but packed ("dehydrated"). The `XboxBvhBuilder.dll`
     rehydrate entry point unpacks them in place once they have streamed in (about 3,000 in the
     opening).
   - The dynamic structures (characters, vehicles, the top level) are built and refit every frame
     by the same DLL's compute programs (52 of them).
   - For a dynamic structure, the title reads the compacted size back from a small buffer, places
     the structure, and copies it there.

Without any of this, every ray misses: whatever the sun does not reach directly is black, and
glossy surfaces reflect the open sky. Where a pass reads data that was never translated, it shows
bright magenta, which is a useful translator signal (item 36). **We do not paint a constant or
sky-derived light over the result**, as the Belobug fork does. A pass either produces real
results or is skipped explicitly and counted (section 5).

### Where we are today

| Piece | Status |
|---|---|
| Decoder (`shaders/src/rdna2.rs:1420`) | Labels **0xE0/0xE1** as the BVH ops. GFX10.3 puts them at **0xE6/0xE7**; 0xE0 is not a BVH op. Fix first (contract item C1). |
| Translator | No lowering for the BVH ops. FLAT/GLOBAL loads and stores (the raw-address path) are decoded but not lowered (`rdna2_to_wgsl.rs:7037`, `:8631`); they count as unsupported. |
| `XboxBvhBuilder.dll` | Host-only shims (`recomp/.../policy.rs` `DEFAULT_HOST_ONLY`). Build, copy and rehydrate do nothing; prebuild answers 256 bytes (`runtime/os/xbox_middleware.cpp:2071+`). |
| Compute root SRV/UAV | Ignored by the state bridge (`d3d12_guest_command_list.cpp:1660`, `:1673`). The builder and trace programs pass addresses this way, and through spilled user-data tables. |
| Feature queries | Item 31 (a) and (b) (E_NOTIMPL for unknown queries, plus a census) are separate and land first. |

## 2. Node layout (the behaviour our builder must reproduce)

A node pointer's low 3 bits give the kind; the node is at `base + ((ptr >> 3) << 6)` (64-byte
granularity).

| Kind | Node | Bytes | Layout |
|---|---|---|---|
| 0-3 | triangle *k* of a triangle node | 64 | 5 vertices × 3 f32 (bytes 0-59), dword 15 = one byte per triangle: bits 1:0 the vertex weighted by I, bits 3:2 the one weighted by J. Triangle *k* uses vertices {0,1,2}, {1,3,2}, {2,3,4}, {2,4,0}. Unused vertices may carry program data. |
| 4 | box node, f16 bounds | 64 | 4 child pointers (dw 0-3, `0xFFFFFFFF` = none), 4 × (min xyz, max xyz) as f16 from byte 16. |
| 5 | box node, f32 bounds | 128 | 4 child pointers, 4 × 6 f32 from byte 16; bytes 112-127 are not read. |
| 6 | instance | 128 (title format) | Read by the title's code, not by the instruction (it answers a triangle miss). |
| 7 | procedural | n/a | Same as kind 6. |

**What the instruction returns.**
- *Box node:* the child pointers of the boxes the ray enters within `[0, extent]` (slab test on
  all 4 boxes). With sort set they come nearest entry first; equal distances keep child order;
  a miss or missing child is `0xFFFFFFFF`. A ray that starts inside a box enters it at t = 0. An
  empty slot's inverted bounds never count as entered.
- *Triangle:* watertight test (shear to the dominant axis, edge-function signs). It returns
  `t_num, t_den, i_num, j_num` as f32, so t = num/den and the program divides. The denominator is
  positive for front faces (clockwise seen from the origin, left-handed). A miss returns
  `(+inf, 1, 0, 0)`. The extent is **not** tested for triangles: the program compares t itself.
  Any NaN misses; a triangle seen edge-on misses; a ray through a shared edge hits both triangles.
- *Unknown on the console, assumed:* exact rounding, tie order, and the exact-edge case. Our
  reference picks one answer for each and the tests pin it.

**Structure header (64 bytes, observed in dumps, behaviour only).**
- dw0: version 6 in the low half; marks in the high half (bottom level `0x8000`, compacted `0x4`,
  rehydrated `0x2`, still packed `0x100`).
- dw1: size in 64-byte nodes. dw3: element count. dw4-9: bounds.
- dw10: offset of the box nodes. dw12: the root node pointer.
- dw13: offset of the elements. dw14: offset of the parent table. For static structures this
  offset is negative, because the index counts leaves too.

**Instances** (top level, at the header's element offset): 128 bytes each.
- dw 0-11: a 3x4 matrix.
- Then id and mask, flags, the root pointer (bit 30 set means static), and the bottom level's
  address in 256-byte units.

Unknowns we have to settle against a real dump before building:
- the parent-table format (does any trace program use it for restarts?);
- whether the trace programs accept f32 box nodes (kind 5) where the shipped data uses f16
  (kind 4). Section 4.3 leans on this.

## 3. Raw 40-bit addressing on WebGPU: the address arena

WebGPU has no buffer device addresses, no 64-bit integers or atomics, at most 4 bind groups, and
we cap `maxStorageBufferBindingSize` at 512 MiB (`web/workers/wgpu_worker.js:1508`). The design:

- **Arena.** The trace and builder programs reach title GPU memory through up to **4 storage
  buffers of 512 MiB** (2 GiB in all; adjustable, with the size and use shown in the boot report).
  They are filled in **64 KiB pages**.
- **Directory.** A two-level table, built at a single 64 KiB page size:
  - The top level has 4,096 entries, one per 256 MiB of the 40-bit space.
  - Each second-level table has 4,096 entries, one per 64 KiB page, and is allocated only when
    used.
  - A second-level entry packs the arena index (2 bits) and the page slot (rest); 0 means "not
    resident".
  - Two dependent u32 loads per access, which cache well because rays touch few pages. A flat
    table would need 64 MiB, so the two levels are worth it.
- **Access semantics** (the same for the trace programs, the builder and our own kernels):
  - An address is a `vec2<u32>` (lo, hi); the page is `addr >> 16`.
  - A load from a page that is not resident returns 0, a store to one is dropped, and each such
    access bumps `rt_fault` (an atomic counter). Nothing traps.
  - Atomics available on arena words: u32 add, sub, min, max, and, or, xor, exchange and
    compare-exchange.
  - Float min/max (the builder's refit) maps onto u32 min/max through the order-preserving bit
    flip. Its result bits are the same as a real float min/max, except for -0/+0 and NaN; the
    tests pin those cases.
  - A program that needs 64-bit atomics makes its pass a skip, with reason `atomic64`.
- **Residency (runtime/gpu).** Before an RT-class dispatch (section 5), the runtime makes resident
  the pages named by that dispatch's addresses:
  - root CBV/SRV/UAV addresses, root constants the translator reflected as addresses, and
    descriptor ranges;
  - for every structure such a dispatch reaches through a top level, the whole structure. The
    header's dw1 gives its size, and instances give the bottom-level addresses. The CPU can walk
    a header and instance list at record time only when the GPU has not written it; otherwise the
    structure's pages stay resident from when it was built.

  The upload source is guest memory. We reuse the O37 window write marks and the existing
  texture-memory change detection to upload only changed pages.
- **Ownership.** A page that a builder or trace program has written becomes GPU-authoritative.
  CPU uploads skip it until the title frees or remaps that memory. Arena contents come back to
  guest memory only for small buffers the title reads on the CPU (size answers). The title never
  reads tree nodes on the CPU.
- **Eviction.** LRU by frame across non-structure pages. Structure pages are pinned while their
  structure is live; if pinned pages do not fit, the arena is full, and later RT passes skip
  with reason `arena_full`.

This is also the general fix for "loads through raw addresses", which is not unique to RT. Any
program with FLAT/GLOBAL accesses can use the arena once the translator lowers them (C3).

## 4. Structures: where they come from

### 4.1 Static world: rehydrate

The packed format is the DLL's private format, and reverse-engineering it is the risky part. Plan:
**run the DLL's own rehydrate path.**

- The policy already supports per-symbol rules. Mark only
  `RehydrateRaytracingAccelerationStructureBatched2X` as guest (lifted). This follows the Bink2
  precedent (R44): the lifted copy runs when the DLL is supplied, and the shim serves when it is
  absent.
- The lifted code records its rehydrate dispatches (DecompressHeader, Box32/Box16Rehydration,
  LeafRehydration) through our D3D12 layer. Their programs translate like any title program.
- This needs the spilled user-data table and compute root UAVs (runtime work item G-RT2).
- Rehydration is in place and asks for no size readback, so latency is not an issue.

### 4.2 Dynamic structures: our own builder, sizes answered on the CPU

The avatarmanz fork runs the DLL's builder and hits **overlapping structures**: 8 of 65 good in
one run, 48 of 275 in another. The cause: the title reads the compacted size one or two frames
after asking, without checking that it is there. Readbacks that lag by one or two frames return
another structure's stale size, and the following copy overwrites neighbours. Waiting on the
submission did not fix it and cost about 20 ms a frame. Browser readback is asynchronous, so we
would be worse off.

So **build, prebuild and copy stay host-side, and we write our own builder** whose sizes are known
on the CPU:
- `GetRaytracingAccelerationStructurePrebuildInfo`: answers a deterministic upper bound from the
  primitive count. Example: one triangle per triangle node, so leaves ≤ N, box nodes
  ≤ ceil((N − 1)/3) + 1, plus header, parent table and instances, rounded to 256. Scratch is the
  sort buffers.
- Compacted-size postbuild info: written into guest memory **on the CPU at record time**, equal
  to that same bound. We never compact, so the "compacted" size is always ≥ what was written,
  and neighbours can never overlap. This costs memory, not correctness.
- `CopyRaytracingAccelerationStructure`: clone and compact are both an arena-to-arena copy of
  the bound size, plus a rebase of the header's self-relative fields if any turn out to be
  absolute (to confirm against a dump).
- `BuildRaytracingAccelerationStructureBatched2X`: for each desc, the steps below run as
  hand-written WGSL kernels of ours in the web worker. They are not game-derived, so they can
  live in the repo.
  1. Read the vertices and indices through the arena (skinned vertices are GPU-written, so the
     build must happen on the GPU).
  2. Assign 30-bit Morton codes; `Set64BitMortonCodes` may later select 60-bit codes.
  3. Radix sort.
  4. Karras LBVH, collapsed to 4-wide.
  5. Bottom-up bounds with arena atomics.
  6. Write the native format: header, kind-5 box nodes (see 4.3), triangle nodes and the parent
     table.
  - The top level writes instance records.
  - `PERFORM_UPDATE` refits in place, keeping the topology.

### 4.3 Box precision

For dynamic structures, start with **f32 box nodes (kind 5)**. They are exact, so there is no
outward rounding to get wrong. Switch to f16 (kind 4, rounded outward) only if a dump shows the
title's traversal assumes 64-byte box strides. Static structures keep whatever rehydration writes.

### 4.4 Safety rule (taken from the forks' experience)

A partly working builder writes garbage bounds into title memory and the title crashes. So
**all-or-none**:
- the RT gate opens only when every rehydrate pipeline translated with 0 unsupported ops, and our
  builder kernels are ready;
- otherwise no rehydrate runs, no build writes, and every RT-class pass is skipped with reason
  `structures_unavailable`.

A quarantine mode keeps builder stores out of the arena pages the title also uploads to, for a
first test.

## 5. Running trace passes: the gate and the counters

A pipeline is **RT-class** when its reflection reports `bvh_ops > 0` or `raw_address = true`
(C2). For each RT-class dispatch, the runtime decides:

| Check | On failure: skip with reason |
|---|---|
| RT enabled (`--rt=on`, off by default until M2) | `rt_off` |
| Program translated with 0 unsupported ops | `untranslated` |
| No 64-bit atomics, no unsupported wave width | `atomic64`, `wave` |
| Structures available (4.4) | `structures_unavailable` |
| Every named address resident (section 3) | `not_resident`, `arena_full` |

- A **skipped dispatch is not recorded.** Its outputs keep what the title last left in them,
  which is black where unlit. That is honest and visible, and it is never a fake constant.
- The boot report gets an `rt` block:
  - dispatches run, and skipped per reason × pipeline name (top 10);
  - structures rehydrated, built, refit and copied;
  - arena pages resident, uploaded and GPU-owned, faults (`rt_fault`), and arena size and use;
  - size answers given;
  - and, when timestamps are available, GPU ms per pass family.
- The magenta signal stays: a pass that ran but read untranslated data still shows it.

**Our own reference traversal** is hand-written WGSL plus a C++ CPU twin. It is a closest-hit
loop with a 32-entry stack: pop a pointer, call the node test, then either push the returned
children far to near or, for a triangle, keep the hit if `0 < t < t_best` (t = num/den). It
handles kind-6 instances by transforming the ray. It is **not** used for shading. It exists for:
- builder tests: build synthetic meshes, trace, and compare with brute force;
- a debug view (`--rt-debug=depth`) that shows our structures independently of the title's
  programs;
- checking a dump's static structures against expectations.

## 6. Contract for `shaders/` (proposal, Marc decides)

- **C1. Decoder.** `image_bvh_intersect_ray` = MIMG 0xE6, `image_bvh64_intersect_ray` = 0xE7:
  - address VGPR count 11 / 12 (A16: 8 / 9); result is 4 VGPRs; the resource is a 4-dword
    descriptor (no sampler);
  - the descriptor comes either from a table (a host buffer: pointers count from its start), or
    is built in SGPRs (the raw-address form, in which dw1 and dw3 carry the flags).
- **C2. Reflection.** Per program:
  - `bvh_ops: u32`, `raw_address: bool`, `atomic64: bool`;
  - `address_sources`: the root params, root constants and spilled user-data dwords that feed
    FLAT/GLOBAL or BVH addresses, so the runtime can make them resident.
  - The runtime keys the gate on these fields.
- **C3. Arena bindings.** Proposal: group 3, bindings 16-21. Group 3 binding 0 is already the
  push-constant block. Names:
  - `pg6_arena0..3: array<u32>` (read_write storage);
  - `pg6_arena_dir: array<u32>` (read; top level of 4,096 entries, then the second-level tables);
  - `pg6_rt_stats: array<atomic<u32>>` (index 0 = faults).
  - Lowering: FLAT/GLOBAL loads, stores and atomics, and raw-address node fetches, go through one
    `pg6_addr_load/store/atomic(addr: vec2<u32>, …)` helper set with the semantics of section 3.
  - **The runtime owns the helper source.** It ships as a WGSL prelude string, versioned, so the
    paging details can change without a translator release.
- **C4. Node test.** The translator emits a call to `pg6_bvh_intersect(desc: vec4<u32>,
  node: vec2<u32>, extent: f32, origin: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>) -> vec4<u32>`.
  - For A16, the translator unpacks the halves first.
  - The function lives in the same runtime-owned prelude, so there is one implementation and one
    set of tests.
- **C5. Test vectors.** A synthetic set in `tests/data/bvh_vectors.json`: nodes of every kind, and
  rays covering inside-box, axis-aligned, on-face, empty-slot, ties, shared-edge, edge-on and NaN
  cases. It is generated by a script from our own reference. The prelude has to match it bit for
  bit on the GPU, and our CPU twin has to match too.
- **C6. Waves.** Trace programs are written for 64-lane waves and use lane masks as numbers. They
  use the existing `wave_emul` path; anything that cannot be emulated reports `wave` in C2.

## 7. Cost estimates

The anchor is avatarmanz's measurement on a 7900 XTX (D3D12, node test already emulated in
software), dorm scene:
- About +40 ms of GPU time a frame with the builder on (75 of 90 ms busy, against 34 of 64 without).
- Traces 24 ms: V-buffer 7.9, screen probes 6.4, irradiance cache 3.5, primary probes 2.3, shadows 1.6.
- The DLL's copy pass 13.3 ms before they bounded it.

What changes for us:

| Factor | Effect |
|---|---|
| Directory lookup per load (2 cached u32 loads) | +20-40% on the traces (they are memory-bound) |
| u64 math emulated as `vec2<u32>` in address arithmetic | +5-10% |
| Wave64 on 32-wide or subgroup-less hosts | +0-50% depending on the program |
| Our builder: no compaction, LBVH quality below the DLL's SAH-style trees | traces +10-30% (more nodes visited); builder about 2-6 ms for the dynamic set |
| No 13 ms copy problem (sizes are bounded by construction) | -10 ms against their first numbers |

Estimate:
- **High-end desktop (7900 XTX / 4080 class): 35-70 ms extra a frame.**
- **Mid-range (about 1/3 of that throughput): 100-250 ms.**
- Arena memory: 2 GiB default. Directory tables: under 1 MiB. Upper-bound sizing wastes roughly
  1.5-2× on dynamic structures, which is small next to the static set.

This is acceptable for M1 ("correct but slow"). M3 brings it down.

## 8. Staged plan

| Stage | Track | Deliverable | Done when |
|---|---|---|---|
| **R0** | runtime | Item 31 (a) and (b); RT-class detection off reflection; the `rt` boot-report block with skip counters (everything skips `rt_off`) | Unit test; boot report lists RT-class pipelines and skip counts |
| **R1** | shaders (C1, C2) + runtime | Decoder fix; reflection fields; runtime prelude with `pg6_bvh_intersect` + CPU twin; C5 vectors | Vectors pass on CPU and on GPU (headless WebGPU test) |
| **R2** | runtime + web; shaders (C3) | Address arena, directory, residency and upload; compute root SRV/UAV and spilled user-data in the state bridge; FLAT/GLOBAL lowering | Synthetic program reads and writes by address; fault counter works; arena shown in the boot report |
| **R3** | runtime | Rehydrate via the lifted DLL symbol; our builder, prebuild, copy and size answers; reference traversal and its tests | Synthetic meshes: reference traversal equals brute force; the gate opens only all-or-none |
| **M1** | all | `--rt=on`: real trace passes on real structures, slow; every unsatisfied pass is skipped and counted; **no constant light** | Locally, by the owner: an indoor frame lit by lamps instead of black; `rt_fault` ≈ 0 in steady state; skip reasons explained |
| **M2** | runtime | RT on by default where the adapter is fast enough (measured GPU ms per family) | Boot report timing; no regression with RT off |
| **M3** | runtime + web | Speed: f16 boxes, better builder (binned SAH for static-size items), the title's own quality and resolution knobs, refit instead of rebuild, cache-friendly node order | Traces within about 1.5× of the anchor on the high-end class |

Explicitly out of scope:
- DXR or a WebGPU ray-tracing extension, since the title does not use DXR;
- a constant or sky-driven stand-in light (item 36's Belobug approach). If the owner later wants
  a stand-in, it has to be a separate, default-off debug option, and it does not count toward M1.
