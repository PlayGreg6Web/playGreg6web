#!/usr/bin/env python3
"""Print the compact fault record of a playgta6_boot report and suggest how to inspect the address.

The boot (runtime/guest/boot.cpp, BootConfig::fault_record, on by default; `--no-fault-record` turns
it off) writes a `fault_record` object into the boot report when the guest faults or runs into a
recompiler gap (unknown_target, unknown_resume_target, trap): guest rip, the containing lifted
function (entry + offset, nearest export), the indirect call source, a register snapshot, the guest
call stack, the last 64 dispatched function entries of the faulting thread, the run's elapsed time
and the newest present's draws per frame. The same facts appear as one `[fault] ...` log line.

Usage:
  tools/fault_report.py <boot-report.json> [--entries N] [--json]
  tools/fault_report.py -            (read the report from stdin)

Exit status: 0 = a fault record was printed, 1 = the report has none (no fault, or fault_record
was off), 2 = unreadable input. Output contains guest addresses and symbol names only.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guest_symbols as gs  # noqa: E402

HARD_STOPS = Path(__file__).resolve().parent.parent / "docs" / "HARD_STOPS.md"
RECOMP = "cargo run -p playgta6-recomp --release --"


def _hex(value: Any) -> int | None:
    if isinstance(value, str):
        try:
            return int(value, 16)
        except ValueError:
            return None
    return None


def _loc(loc: Any) -> str:
    """`sub_<entry>+0xOFF [module!export+0xOFF]` for a code_location object."""
    if not isinstance(loc, dict):
        return "(no lifted function)"
    parts = []
    entry = _hex(loc.get("entry"))
    offset = _hex(loc.get("offset"))
    if entry is not None:
        parts.append(f"sub_{entry:X}+0x{offset or 0:x}")
    sym = loc.get("symbol")
    if isinstance(sym, dict):
        sym_off = _hex(sym.get("offset")) or 0
        parts.append(f"[{sym.get('module')}!{sym.get('name')}+0x{sym_off:x}]")
    return " ".join(parts) if parts else "(no lifted function)"


_SYMS: "gs.PeSymbols | None" = None  # set by render(); names come from the PE tables only


def _addr(value: Any) -> str:
    v = _hex(value)
    return "-" if v is None else f"0x{v:016x}"


def _named(value: Any) -> str:
    """`0x... (module fn 0x..+0x.. export Name+0x..)` when the address lies in the supplied PE."""
    v = _hex(value)
    text = _addr(value)
    if v is not None and _SYMS is not None:
        d = _SYMS.describe(v)
        if d:
            text += f" ({d})"
    return text


def next_step(report: dict[str, Any], rec: dict[str, Any], slot: dict[str, Any] | None) -> str:
    kind = rec.get("kind")
    if kind == "unimplemented_slot" or slot is not None:
        return ("next step: unimplemented COM slot -> owner decision (docs/HARD_STOPS.md) or implement the slot "
                "in runtime/os/xgameruntime_provider.cpp")
    if rec.get("code_miss") or kind in ("unknown_target", "unknown_resume_target", "trap"):
        return "next step: recompiler gap -> tools/triage_fault.py <report> --pe <game.exe>"
    return "next step: not a recompiler gap or COM slot; inspect the call stack and recent entries above"


def suggestions(rec: dict[str, Any]) -> list[str]:
    """Commands (static analysis, no game run) that inspect the faulting address."""
    out: list[str] = []
    kind = rec.get("kind")
    target = _hex(rec.get("target"))
    fn = rec.get("function") if isinstance(rec.get("function"), dict) else {}
    entry = _hex(fn.get("entry"))
    src = rec.get("source") if isinstance(rec.get("source"), dict) else {}
    src_loc = src.get("location") if isinstance(src.get("location"), dict) else {}
    src_entry = _hex(src_loc.get("entry"))
    if kind == "unknown_target" and target is not None:
        out.append(
            "seed the target as a function entry and rebuild: "
            f"{RECOMP} build --from-env --out work/<run> --entry 0x{target:X}"
        )
        if src_entry is not None:
            out.append(
                "inspect the indirect call site (vtable / jump table / callback): "
                f"{RECOMP} lift <pe-path> --out work/fault_lift --entry 0x{src_entry:X}"
                f"   # read sub_{src_entry:X} in work/fault_lift/lifted.ll"
            )
    elif kind == "unknown_resume_target":
        resume = _hex(src.get("resume_rip"))
        base = resume if resume is not None else target
        if base is not None:
            out.append(
                "an exception/longjmp resumed at an address with no lifted block; seed it: "
                f"{RECOMP} build --from-env --out work/<run> --entry 0x{base:X}"
            )
    elif kind in ("trap", "unimplemented") and entry is not None:
        out.append(
            "inspect the lifted function containing the trap (unresolved switch / undecoded opcode): "
            f"{RECOMP} lift <pe-path> --out work/fault_lift --entry 0x{entry:X}"
            f"   # find the trap near sub_{entry:X}+0x{_hex(fn.get('offset')) or 0:x} in work/fault_lift/lifted.ll"
        )
    if target is not None:
        out.append(
            "list recompiler-discovered functions around the address: "
            f"{RECOMP} discover <pe-path> --out work/discover.json   # then search 0x{target:X}"
        )
    return out


def render(report: dict[str, Any], entries: int = 16, syms: "gs.PeSymbols | None" = None,
           hard_stops: str | None = None, log_lines: Any = ()) -> str:
    global _SYMS
    _SYMS = syms
    rec = report.get("fault_record")
    if not isinstance(rec, dict):
        return ""
    lines: list[str] = []
    kind = rec.get("kind", "?")
    flag = "  [recompiler gap]" if rec.get("code_miss") else ""
    lines.append(f"FAULT {kind}{flag}  status={report.get('status')}  phase={report.get('phase')}")
    if report.get("message"):
        lines.append(f"  message : {report['message']}")
    lines.append(f"  rip     : {_named(rec.get('rip'))}")
    lines.append(f"  target  : {_named(rec.get('target'))}  detail={_addr(rec.get('detail'))}")
    lines.append(f"  function: {_loc(rec.get('function'))}")
    src = rec.get("source")
    if isinstance(src, dict):
        if "return_address" in src:
            lines.append(
                f"  source  : indirect call, return address {_named(src['return_address'])} in {_loc(src.get('location'))}"
            )
        elif "resume_rip" in src:
            lines.append(f"  source  : resume rip {_named(src['resume_rip'])} in {_loc(src.get('location'))}")
    lines.append(f"  thread  : {rec.get('thread')}   elapsed: {rec.get('elapsed_ms')} ms")
    present = rec.get("present")
    if isinstance(present, dict):
        lines.append(f"  screen  : presents={present.get('presents')} draws/frame={present.get('draws_per_frame')}")
    else:
        lines.append("  screen  : no present yet")
    regs = rec.get("registers")
    if isinstance(regs, dict):
        lines.append("  registers:")
        names = list(regs)
        for i in range(0, len(names), 4):
            lines.append("    " + "  ".join(f"{n:<3}={_addr(regs[n])[2:]}" for n in names[i : i + 4]))
    stack = rec.get("call_stack") if isinstance(rec.get("call_stack"), dict) else {}
    frames = stack.get("frames") or []
    lines.append(f"  call stack ({stack.get('method', 'none')}), innermost caller first:")
    for i, fr in enumerate(frames):
        addr = _named(fr.get("address")) if isinstance(fr, dict) else "-"
        lines.append(f"    #{i:<2} {addr} {_loc(fr)}")
    if not frames:
        lines.append("    (none)")
    recent = rec.get("recent_entries") or []
    shown = recent[-entries:] if entries > 0 else []
    lines.append(f"  last {len(shown)} of {len(recent)} dispatched entries on the faulting thread (oldest first):")
    for e in shown:
        name = f" {e['name']}" if e.get("name") else ""
        lines.append(f"    -> {_named(e.get('target'))}{name}  from {_named(e.get('from'))}")
    slot = gs.find_unimplemented(report, log_lines)
    if slot is not None:
        lines.append(f"  com slot: {slot['iface']} {slot['guid']} slot {slot['slot']}"
                     f"{' ' + slot['name'] if slot.get('name') else ''}")
        status = gs.hard_stop_status(hard_stops, slot["iface"], slot["slot"], slot.get("name")) if hard_stops else None
        lines.append(f"  hard stop: {status}" if status else "  hard stop: not listed in docs/HARD_STOPS.md")
    lines.append("  " + next_step(report, rec, slot))
    sug = suggestions(rec)
    if sug:
        lines.append("  next steps:")
        lines.extend(f"    $ {s}" for s in sug)
    misses = report.get("code_misses")
    if isinstance(misses, dict):
        total = {k: (v or {}).get("count", 0) for k, v in misses.items() if isinstance(v, dict)}
        if any(total.values()):
            lines.append("  code_misses in this run: " + ", ".join(f"{k}={v}" for k, v in total.items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print the fault record of a playgta6_boot report.")
    ap.add_argument("report", help="boot report JSON ('-' for stdin)")
    ap.add_argument("--entries", type=int, default=16, help="dispatched entries to show (default 16)")
    ap.add_argument("--pe", help="game PE (read-only): annotate addresses with export/import/.pdata names")
    ap.add_argument("--log", help="boot log file: also search it for the COM-bridge unimplemented-slot line")
    ap.add_argument("--json", action="store_true", help="print the raw fault_record object instead")
    args = ap.parse_args(argv)
    try:
        text = sys.stdin.read() if args.report == "-" else open(args.report, encoding="utf-8").read()
        report = json.loads(text)
    except (OSError, ValueError) as err:
        print(f"fault_report: cannot read report: {err}", file=sys.stderr)
        return 2
    if not isinstance(report, dict) or not isinstance(report.get("fault_record"), dict):
        print("fault_report: no fault_record in this report (no fault, or the boot ran with --no-fault-record)")
        return 1
    if args.json:
        print(json.dumps(report["fault_record"], indent=2))
    else:
        syms = None
        if args.pe:
            try:
                syms = gs.PeSymbols.from_path(args.pe)
            except (OSError, ValueError, struct.error) as err:
                print(f"fault_report: cannot read PE symbols: {err}", file=sys.stderr)
        try:
            hs = HARD_STOPS.read_text(encoding="utf-8")
            log = open(args.log, encoding="utf-8", errors="replace").read().splitlines() if args.log else ()
        except OSError:
            hs, log = None, ()
        sys.stdout.write(render(report, args.entries, syms, hs, log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
