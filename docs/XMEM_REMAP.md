# XMem remap of a mirrored window slot

Follow-up to #1934 / #1942 (host readers re-check the window slot under the arena lock). This note
covers the guest's own loads: lifted code reads `host = va + __playgta6_window_table[slot]` with
no lock and no re-check.

## The problem

Under the window memory models a MEM_PHYSICAL reservation is handed to the title at a window
range (a mirror, O37). Its slots show the reservation's own arena memory: the page mapped there
lives at that arena VA (its home), and the window delta of the slot never changes while the slot
is mapped.

`XMemMapPhysicalPages(va, n, pfns)` over a slot that already shows a page ran
`map_slot_locked`, which first did a full unmap and then a fresh map:

1. copy the old page's bytes to a store,
2. set the window slots to `kTrapDelta`,
3. decommit the arena page,
4. commit it again (zero),
5. copy the new page's bytes in,
6. map the window slots again.

A guest load during steps 2 to 6 reads the trap delta and faults. A load that read the delta
before step 2 and loads after step 3 hits decommitted memory (host fault); after step 4 it sees
zeros. On the console a remap swaps the page: a load sees the old page or the new one.

## Measured exposure (before the fix)

`tests/test_vmem_window_stress.cpp` phase 4: one thread re-points a mirrored 2 MiB page between
two physical pages (500 remaps per seed, never unmapped), readers (2 x hardware threads, 8 to 32)
do raw guest-style byte loads at random offsets. 3 seeds on a 24-thread desktop:

| | loads | saw `kTrapDelta` | saw zero |
|---|---|---|---|
| seed 1 | 2.66e9 | 1.64e9 (62%) | 36 |
| seed 2 | 3.14e9 | 1.98e9 (63%) | 59 |
| seed 3 | 2.69e9 | 1.68e9 (63%) | 40 |

The share is high because the mutator does nothing but remap. Every remap opened a window of a
decommit, a commit and a 2 MiB copy. Without an exception guard around the loads the test
exits with an access violation in the first seed: some loads go through a stale delta into the
decommitted page.

## The fix: remap in place

`remap_in_place_locked` picks the case where the slot can stay valid the whole time:

- the slot is the primary slot of a mirrored, uncommitted MEM_PHYSICAL reservation (not phantom,
  not the translated CPU window, not a shared-zone reservation),
- the page mapped there now is homed at that slot and is not shown there as an alias,
- the new page has the same size, lives nowhere else (no home, no shared view): its bytes come
  from its store, or it is zero.

Then `unmap_slot_locked(..., keep_committed = true)` saves the old bytes to a store as before.
It leaves the arena page committed and the window slots mapped, and does not touch
`virtual_committed_bytes`. `map_slot_locked` skips the commit and copies the new bytes over the
old ones with `VirtualMemoryArena::fill_committed`. That helper copies under the arena lock, so
`GuestRangeView` / `read_guest` (which re-check under that lock) still see one whole page. Then
the slots are mapped again with the same delta, which sets their write marks.

Mapping the page that is already there is now a no-op. Before, it took the full unmap and map.

After the fix, phase 4 sees 0 trap deltas and 0 zero bytes over ~1.9e9 raw loads (3 seeds).

### Lock order and earlier races

- Still XMem lock, then arena lock (as commit / decommit already did). The window lock is taken
  only after the arena lock is released (`map_window_page`). Readers take only the arena lock
  (#1913).
- The delta stays the same, so no reader is left holding a pointer to memory that went away. The
  stale-pointer families of #1817 / #1818 / #1855 do not come back. No page is decommitted, so
  there is no decommit window.
- R40 in-flight work: `map_slot_locked` still runs after `wait_idle_locked`, as before.

### What is left

- A guest load racing the copy can see some bytes of the old page and some of the new one. Each
  byte is one or the other, never zero. Locked host readers see whole pages. The arena lock is
  held for the copy (about 0.1 ms for 2 MiB).
- Unmap (null pfn) followed by a later map still goes through `kTrapDelta`. That matches the
  console: a load from an unmapped page faults.
- Other remap shapes still unmap and then map: an alias slot (page homed elsewhere) replaced by
  another page, a primary page replaced by an alias, a page with a shared view, and pages of
  different sizes. They could move to the same scheme by first filling the new memory and then
  swapping the delta (staging), when a title is seen doing it.
- Placeholder APIs (`VirtualAlloc2` + `MapViewOfFile3` with `MEM_REPLACE_PLACEHOLDER`) would make
  the swap atomic for the page as a whole. But mirrored reservations are plain arena memory, not
  views, so that needs the arena to back them with sections (as the shared zone does). It is not
  needed for the guarantee above.
