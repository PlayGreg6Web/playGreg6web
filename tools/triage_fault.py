#!/usr/bin/env python3
"""One-shot triage of a boot fault: fault_record -> recompiler analysis of the faulting indirect site.

Reads the `fault_record` of a playgta6_boot report (see tools/fault_report.py), picks the indirect
site that produced the fault (the call whose return address is `source.return_address`, or the
faulting function offset for traps), and runs `playgta6-recomp explain-site` on the game PE. The
explanation lists the instruction window (mnemonics and operands only), the definitions reaching
the index/base registers, the jump_table rule that rejected the site and why, the survey pattern
class and a suggested fix category.

Usage:
  tools/triage_fault.py <boot-report.json> --pe <game.exe> [--recomp "<cmd>"] [--no-refine] [--plan]

  --recomp   command prefix for the recompiler (default: cargo run -q -p playgta6-recomp --release --)
  --plan     only print the explain-site command instead of running it

Output is local only: it contains guest addresses, mnemonics and operands decoded from the game
image. Do not commit it, and never write raw bytes of the image to files in git.
Exit status: 0 = explained, 1 = no fault record / no site, 2 = unreadable input or recompiler error.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from typing import Any

import fault_report as fr

DEFAULT_RECOMP = "cargo run -q -p playgta6-recomp --release --"


def pick_site(rec: dict[str, Any]) -> tuple[int, bool, str] | None:
    """(address, is_return_address, why) of the site to explain, or None."""
    src = rec.get("source") if isinstance(rec.get("source"), dict) else {}
    ret = fr._hex(src.get("return_address"))
    if ret is not None:
        return ret, True, "indirect call that reached the unknown target (return address)"
    fn = rec.get("function") if isinstance(rec.get("function"), dict) else {}
    entry, off = fr._hex(fn.get("entry")), fr._hex(fn.get("offset"))
    if entry is not None and off is not None:
        return entry + off, False, "faulting offset inside the lifted function"
    return None


def command(recomp: str, pe: str, addr: int, ret: bool, no_refine: bool) -> list[str]:
    parts = [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in shlex.split(recomp, posix=False)]
    cmd = parts + ["explain-site", pe, f"0x{addr:X}"]
    if ret:
        cmd.append("--return-address")
    if no_refine:
        cmd.append("--no-refine")
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Triage a boot fault with the recompiler's own analysis.")
    ap.add_argument("report", help="boot report JSON")
    ap.add_argument("--pe", required=True, help="game PE path (read-only)")
    ap.add_argument("--recomp", default=DEFAULT_RECOMP, help="recompiler command prefix")
    ap.add_argument("--no-refine", action="store_true", help="skip the refined discovery pass")
    ap.add_argument("--plan", action="store_true", help="print the command instead of running it")
    args = ap.parse_args(argv)
    try:
        with open(args.report, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError) as err:
        print(f"triage_fault: cannot read report: {err}", file=sys.stderr)
        return 2
    rec = report.get("fault_record") if isinstance(report, dict) else None
    if not isinstance(rec, dict):
        print("triage_fault: no fault_record in this report")
        return 1
    syms = None
    try:
        syms = fr.gs.PeSymbols.from_path(args.pe)
    except (OSError, ValueError, fr.struct.error):
        pass  # names are best effort; explain-site below reports an unreadable PE
    try:
        hs = fr.HARD_STOPS.read_text(encoding="utf-8")
    except OSError:
        hs = None
    sys.stdout.write(fr.render(report, 4, syms, hs))
    site = pick_site(rec)
    if site is None:
        print("\ntriage_fault: the record names no indirect site or lifted function to explain")
        return 1
    addr, ret, why = site
    cmd = command(args.recomp, args.pe, addr, ret, args.no_refine)
    print(f"\n== site analysis: 0x{addr:X} ({why}) ==")
    if args.plan:
        print("$ " + " ".join(cmd))
        return 0
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as err:
        print(f"triage_fault: cannot run the recompiler: {err}", file=sys.stderr)
        return 2
    if proc.returncode != 0:
        print(f"triage_fault: explain-site failed:\n{proc.stderr}", file=sys.stderr)
        return 2
    sys.stdout.write(proc.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
