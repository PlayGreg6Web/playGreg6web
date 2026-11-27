# Unresolved indirect `jmp` survey

Static census of indirect `jmp`s (no import slot) that discovery leaves on run-time dispatch,
produced by `playgta6-recomp survey-jumps <exe> [--top N] [--examples N] [--out file]`
(`recomp/src/jump_survey.rs`). It reuses the refined discovery analysis (`Analysis::unresolved_jumps`)
and re-reads each jump's local instruction window (its block plus sole-predecessor blocks) to group
by pattern: lifter reason, role (switch or tail call), jump form, index scaling, guard shape and
table/pointer location. "Reachable" walks direct calls and tail calls from the PE entry point and TLS
callbacks only; indirect calls are not followed, so it is a lower bound. The report holds addresses,
section names and counts only.

## Result for the target executable (refined discovery, 435,719 functions; refreshed after #1781)

- 8,318 unresolved indirect jumps; 751 sit in the 32,180 functions reachable from boot.
- By lifter reason: loaded_pointer 5,252, memory_operand 1,795, register_live_in 1,229,
  table_base_unknown 33 (7 reachable), table_unbounded 5 (0 reachable), computed 4.
  Before the #1781 table work these were 42 / 6 / 9 (and 8,333 total, 753 reachable).
- By role: tail_call 8,280 (744 reachable), fn_table 18, ptr_array 11, data_table 5, switch 4 (2 reachable).
  About 8,280 of them are tail calls through a pointer, not switches; only about 38 have a scaled
  index into a table the matcher rejects.
- Pointer-slot tail calls with a statically seeded target: 21 (was 4).

## Top patterns (total / reachable from boot)

| # | Pattern | Total | Reachable | Example sites |
|---|---------|------:|----------:|---------------|
| 1 | `jmp reg` after load of a pointer from memory, preceded by a null/equality check (base is itself a loaded pointer) | 3,027 | 346 | 0x14000646b 0x14001b525 0x1400266ac |
| 2 | `jmp [reg+disp]`, no index, preceded by a null check | 1,036 | 102 | 0x140018348 0x140051f2c 0x1400712c1 |
| 3 | `jmp reg` after load of a pointer from memory, no guard | 1,459 | 93 | 0x140008cda 0x14002682c 0x140026952 |
| 4 | as #1 but base register is live-in (argument / this) | 528 | 91 | 0x14000b049 0x140027b9e 0x140027bc3 |
| 5 | `jmp [reg+disp]`, no index, no guard | 647 | 58 | 0x140048387 0x14005aa98 0x14006acf9 |
| 6 | `jmp reg`, register live-in (pointer passed in) | 1,172 | 10 | 0x14016d002 0x14016d01c 0x140243482 |
| 7 | as #1 where the pointer slot is in `.data` (writable) | 42 | 7 | 0x14017e042 0x1404b2156 0x1404b215c |
| 15/21/25/28 (now #16, #23 in the refreshed ranking) | byte-scaled (`scale=1`) index into a loaded-pointer table | 8 | 3 | 0x14469f9f9 0x144f6683d 0x140338024 0x1406c90a0 |
| 16/18/26/27 (now #17, #19, #26, #30 in the refreshed ranking) | `jmp [base+idx*8]` with base from `.rdata`/`.data` anchor or loaded pointer (`table_base_unknown`, roles fn_table/data_table/ptr_array) | 33 | 7 | 0x142847e4d 0x14030a57a 0x14013ef7c 0x141672a85 |
| 24 (now #31, #34) | switch-role, `jmp [mem]`/`jmp reg` scale 8, base register live-in (`table_base_unknown`) | 3 | 3 | 0x1411be4ed 0x1411c0ab4 0x14198b644 |

Note: the row ids above are from the earlier survey; the pattern ranking is regenerated on each run, so ids shift. Remaining table shapes (refreshed): `table_base_unknown` fn_table with `.rdata` anchor and no guard, 13 sites, 1 reachable (e.g. 0x14030a57a 0x1405dd3dd 0x14073841a 0x140a95646); `.data` anchor with null check, 2 sites, both reachable (0x142847e4d 0x142847e6d); live-in-base switches (patterns #31, #34), 3 reachable sites.

Equality guards are null/sentinel checks of lazily initialised function pointers, not bounds checks.

Shallowest boot-reachable functions holding such jumps (entry, call depth): 0x144eacc70 (4),
0x1428cb1e0 (5), 0x140093420 (7, 247 callers), 0x140104070 (7), 0x144fc2180 (8, 650 callers),
0x14321dc10 (8), 0x141148da0 (9), 0x1402e5b80 (11, 1,986 callers, 3 sites).

## Pointer-slot tail calls (backlog 43)

The lifter already lowers every unresolved `jmp` as a tail call through the run-time dispatcher
(lifted-function lookup by the runtime target, existing unknown-target path). Discovery additionally
looks for `mov reg, [rip+slot]` ... `jmp reg` where `slot` is a base-relocated `DIR64` pointer in a
data section holding a code address, and seeds that address as a function entry
(`Analysis::pointer_slots`, `recomp/src/pointer_slot.rs`). Slots without a base relocation or with a
non-code value are ignored; the jump stays on run-time dispatch because the slot can be rewritten.
On the target executable 4 jumps have such a slot, and their targets were already entries through
the refined relocated-pointer seeding (function count unchanged at 435,718); the survey prints the
count as `pointer-slot tail calls with a statically seeded target`. The other ~8,200 go through
pointers loaded via another loaded pointer or a register, which no static slot describes.
