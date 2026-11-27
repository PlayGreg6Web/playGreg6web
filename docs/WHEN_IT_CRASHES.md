# When it crashes

Short guide for players and developers: where the report is, how to read it, what the usual endings mean, and
what to send. Reports hold hashes, addresses and counts only, never game bytes.

## Where the report is

Every run writes a directory `<runs-dir>/<name>/` with `boot-report.json` (the main file), `boot.log`,
`summary.txt` and `command.json` (see the run layout in `tools/run_native_boot.py`).

| How you played | Run directory |
|---|---|
| Native from the repo (`play_native.ps1`) | `work\native-boot\runs\<name>\` |
| Native prebuilt (`play_native.ps1 -Prebuilt`) | `<prebuilt dir>\runs\<name>\` (default `%LOCALAPPDATA%\playgta6\prebuilt`, or `PLAYGTA6_PREBUILT_DIR`, e.g. `H:\pg6-play`) |
| Browser (`play_local.ps1`) | `work\browser-boot\runs\<name>\` |
| Browser prebuilt (`play_local.ps1 -Prebuilt`) | `<prebuilt-web>\build\runs\play-local-<timestamp>\` (`PLAYGTA6_PREBUILT_WEB_DIR`) |

Browser runs also hold `telemetry.json` and `console.log`. A guest fault in either build is recorded as a
`fault_record` in `boot-report.json`; the native window stays open on the last frame titled `GUEST FAULT ...`
and the console prints the fault report right away. Close the window to finish the report.

The window also shows a box on the last frame, even with the F3 overlay off, and the launcher prints the same
line at the end (`python tools\stop_message.py <run dir>` prints it again):

- `Stopped: <slot> - this needs an owner decision (see docs/HARD_STOPS.md section 7). Not a crash.` An
  unimplemented slot that [HARD_STOPS.md](HARD_STOPS.md) lists; nothing to report.
- `Crashed: <kind> at <site> - report: <run dir>`: anything else. Follow this guide and send that report.

## Read it

```
python tools\boot_summary.py <run dir>          # how it ended, fault symbols, screens, fps, GPU errors
python tools\fault_report.py <run dir>\boot-report.json     # rip, lifted function, call stack, last entries
python tools\triage_fault.py <run dir>\boot-report.json --pe <game.exe>   # why the faulting indirect site failed
python tools\minidump_stack.py <run dir>\boot-report.json   # host stack of a minidump
python tools\run_history.py --root H:/ --since 3d           # table of past runs, to see when it started
```

`boot_summary.py` is printed after every native run anyway. `--pe <game.exe>` (read-only) names addresses.
Minidumps: a host fault (access violation in the harness itself) writes one automatically; for a guest fault
add `--dump-on-fault` to the boot options (pass it to `run_native_boot.py` after `--`). The dump path is
recorded in `fault_record.minidump`; `minidump_stack.py` takes the dump or the report. Dumps stay local, do
not commit or upload them.

Dumps are written by a helper process (the harness executable started with `--pg6-minidump-helper`), never
inside the faulting process, so a dump cannot freeze the run (#663). A run that hangs is never silent: at
`--watchdog-ms` the harness prints every thread's stack (`[hang-stack]` lines) and writes
`host-fault-host_watchdog_timeout-<pid>.dmp` before it builds the report, and if the fault or stop report
path itself does not finish within 30 s, it prints the stacks again and terminates with the fault/watchdog
exit code. `python tools\minidump_stack.py <dump> --all-threads --exe-dir <bin dir>` names every thread's frames.

The `bink` block of the report lists Bink export lookups that returned null (`lookup_null`, with the first distinct names) and the `BinkOpenWithOptions` stream creates (`open_log`: path, flags, result); read it when a video or the loading screen stops. `guest_sleep` shows how long guest `Sleep` calls really took.

## Common endings today

- **Unimplemented slot, e.g. `XSystemGetAppSpecificDeviceId`**: the title called a platform interface slot
  that is deliberately not implemented. This one waits on an owner decision, see section 7 of
  [HARD_STOPS.md](HARD_STOPS.md). It is not a bug to fix in your setup; `boot_summary.py` shows the hard-stop status.
- **Guest `int3` / assert in the title**: the game's own assertion fired (a breakpoint at a guest address). The
  fault record gives the address and call stack; usually a missing or wrong runtime behaviour earlier on.
- **Host access violation**: a bug in the harness or runtime, not the game. Gives a minidump; use
  `minidump_stack.py`.
- **Host access violation at address 0 in `sub_142215A60`** (host frames `sub_142215A60` <- `sub_142215EA0`,
  the faulting thread only calls `Sleep(1000)` and `QueryPerformanceCounter`): the title's own hang watchdog. It
  writes to address 0 on purpose after the main loop has not ticked for about 30 s, so the cause is whatever
  the main thread was stuck on. Look at the report's `blocked_waits` (every thread parked over 1 s in a
  `WaitFor*`, with the object's state, signals since it parked and the last signaler) and `streaming_drain`.
  An object `signaled_for_waiter: true` with the waiter still parked is a lost wakeup in the runtime; an
  object never signaled points at the producer (a job that never finishes, a streaming request never done).
- **Host access violation in `sub_140F2C470`** (GPU page-table command drain, around 60-105 s into
  menu-to-ingame, nondeterministic): two owners of one 64 KiB GPU page. Probes on the title's buddy allocator
  (alloc `sub_145238780`, free-list pop `sub_145238830`, frees `sub_14221B7D0` / `sub_145238960`), shadowing
  every page's allocated/free state, saw no block handed out twice and none freed twice, and the allocator
  lock (`0x1471BFAD8`, a recursive lock with semaphore handoff) was held at every entry. Every per-page
  refcount read-modify-write runs under that lock. The bad pages come one level up: a tile-map call
  (vtable method `sub_1428C9910` through `sub_1428CB9B0`, from `sub_1443C4850`, op 2) on thread 84
  covers a page that thread 80 allocated and mapped for another resource, or that thread 80's queue drain
  unmapped and freed 45 ms earlier (refcount reached zero after the 4 KiB sub-allocator released the page's
  last slice). The title's CPU view (`[0x149E1DAD0]`) and GPU view (`[0x149E1DAC0]`) are separate
  reservations here (e.g. 0x180110000 vs 0xA00000000), and its conversions between them were consistent in
  the paths read. Still open: whether the mapped resource's lifetime ends early on our side (a completion
  signalled too soon) or the range rounding in the tile-map caller reaches a neighbouring page.
- **Unresolved indirect call**: rerun, then use `--seed-targets` (see `tools/run_native_boot.py`), or `triage_fault.py`.
- **Watchdog / time limit**: the run hit `--max-ms` (not a crash) or `--watchdog-ms` (it did not stop in time:
  read the `[hang-stack]` lines and the watchdog dump).

## What to send to developers

- The run directory **name** (for example `play-local-20261127-101500`) and the build commit (`prebuilt.json`
  or the `commit` in the report), plus the `boot_summary.py` output.
- Optionally `boot-report.json` itself (and `console.log` / `telemetry.json` for the browser).
- Never send game files, the executable, the package, saves or minidumps unless a developer asks for a dump.
