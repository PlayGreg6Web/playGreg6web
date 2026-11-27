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

## Common endings today

- **Unimplemented slot, e.g. `XSystemGetAppSpecificDeviceId`**: the title called a platform interface slot
  that is deliberately not implemented. This one waits on an owner decision, see section 7 of
  [HARD_STOPS.md](HARD_STOPS.md). It is not a bug to fix in your setup; `boot_summary.py` shows the hard-stop status.
- **Guest `int3` / assert in the title**: the game's own assertion fired (a breakpoint at a guest address). The
  fault record gives the address and call stack; usually a missing or wrong runtime behaviour earlier on.
- **Host access violation**: a bug in the harness or runtime, not the game. Gives a minidump; use
  `minidump_stack.py`.
- **Unresolved indirect call**: rerun, then use `--seed-targets` (see `tools/run_native_boot.py`), or `triage_fault.py`.
- **Watchdog / time limit**: not a crash; the run hit `--max-ms` or `--watchdog-ms`.

## What to send to developers

- The run directory **name** (for example `play-local-20261127-101500`) and the build commit (`prebuilt.json`
  or the `commit` in the report), plus the `boot_summary.py` output.
- Optionally `boot-report.json` itself (and `console.log` / `telemetry.json` for the browser).
- Never send game files, the executable, the package, saves or minidumps unless a developer asks for a dump.
