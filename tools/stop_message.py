#!/usr/bin/env python3
"""The one line a person sees when the guest stops (same texts as runtime/guest/stop_message.h).

  Stopped: <slot> - this needs an owner decision (see docs/HARD_STOPS.md §7). Not a crash.
      an unimplemented COM slot that docs/HARD_STOPS.md lists (fault_record + HARD_STOPS status lookup)
  Crashed: <kind> at <site> - report: <run dir> (see docs/WHEN_IT_CRASHES.md)
      any other fault

The native window shows the same text as a banner (the harness gets the HARD_STOPS.md slot list in
$PLAYGTA6_HARD_STOP_SLOTS, see pending_arg); run_native_boot.py / play_native.ps1 print it at the end.

    python tools/stop_message.py <run dir | boot-report.json>   # prints the line, exit 1 when no fault
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guest_symbols as gs  # noqa: E402

DASH = "—"
SECTION = "§"


def _short(iface: str) -> str:
    return iface.split(".")[-1].lower()


def _sections(md: str) -> list[tuple[int, str, str]]:
    """(section number, section heading, section body) for every `## N.` section."""
    out: list[tuple[int, str, str]] = []
    parts = re.split(r"(?m)^(?=## )", md)
    for part in parts:
        m = re.match(r"## (\d+)\.", part)
        if m:
            heading = part.splitlines()[0]
            out.append((int(m.group(1)), heading, part))
    return out


def _detail_section(sections: list[tuple[int, str, str]], iface: str, slot: int, names: list[str]) -> str | None:
    """The section whose `###` sub-heading is the detail text of this slot (by name, else by interface + slot)."""
    for num, _heading, body in sections:
        for line in body.splitlines():
            if not line.startswith("### "):
                continue
            low = line.lower()
            if any(n and n.lower() in low for n in names):
                return str(num)
            if iface in low and "slot" in low and slot in {int(x) for x in re.findall(r"\d+", low.split("(", 1)[-1])}:
                return str(num)
    return None


def pending_slots(md: str | None) -> list[tuple[str, int, str | None]]:
    """(interface, slot, detail section) for every row of docs/HARD_STOPS.md that is not marked Implemented."""
    if not md:
        return []
    sections = _sections(md)
    out: list[tuple[str, int, str | None]] = []
    sect = ""
    for line in md.splitlines():
        if line.startswith("## "):
            sect = line
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[2].lower().startswith("implemented"):
            continue
        m = re.match(r"^(?:(\w+)\s+)?(\d+(?:\s*/\s*\d+)*)\s*$", cells[0])
        if not m or (m.group(1) or "").lower() in ("slot", "slots"):  # option tables, not slot rows
            continue
        slots = [int(x) for x in re.findall(r"\d+", m.group(2))]
        ifaces = [m.group(1).lower()] if m.group(1) else [
            t.lower() for t in re.findall(r"\bX[A-Za-z]+\b", sect.split("(")[0] + " " + " ".join(re.findall(r"xgameruntime\.(\w+)", sect)))]
        names = [n.strip() for n in cells[1].split("/")]
        for iface in dict.fromkeys(ifaces):
            for slot in slots:
                out.append((iface, slot, _detail_section(sections, iface, slot, names)))
    return out


def pending_arg(md: str | None) -> str:
    """The value for $PLAYGTA6_HARD_STOP_SLOTS: `iface/slot/section;...` (section `-` when there is none)."""
    return ";".join(f"{i}/{s}/{sec or '-'}" for i, s, sec in dict.fromkeys(pending_slots(md)))


def select(report: dict[str, Any] | None, md: str | None, run_dir: str | Path | None = None) -> str | None:
    """The stop line for a report, None when the run did not fault."""
    rec = report.get("fault_record") if isinstance(report, dict) else None
    if not isinstance(rec, dict):
        # Browser run (tools/run_browser_boot.py): the wasm `[fault-record]` line may be gone from the HUD tail,
        # but the tracer's fatal trap is still in the report's `fault` block (kind e.g. INDIRECT_CALL_MISS, va).
        rec = report.get("fault") if isinstance(report, dict) else None
        if not isinstance(rec, dict) or not (rec.get("kind") or rec.get("trap_kind")):
            return None
        rec = {**rec, "kind": rec.get("kind") or rec.get("trap_kind")}
        report = {**report, "fault_record": rec}
    slot = gs.find_unimplemented(report) if rec.get("kind") == "unimplemented_slot" or rec.get("unimplemented_slot") else None
    if slot:
        short = _short(slot["iface"])
        for iface, num, sec in pending_slots(md):
            if iface == short and num == slot["slot"]:
                name = slot.get("name") or f"{short} slot {slot['slot']}"
                where = "docs/HARD_STOPS.md" + (f" {SECTION}{sec}" if sec else "")
                return f"Stopped: {name} {DASH} this needs an owner decision (see {where}). Not a crash."
    site = rec.get("va") or rec.get("rip") or "unknown site"
    return (f"Crashed: {rec.get('kind') or 'fault'} at {site} {DASH} report: {run_dir or '(see the console)'} "
            "(see docs/WHEN_IT_CRASHES.md)")


def emit(text: str, out=None) -> None:
    """Print the line; a console that cannot encode the dash gets '-' and 'section'."""
    out = out or sys.stdout
    try:
        print(text, file=out, flush=True)
    except UnicodeEncodeError:
        print(text.replace(DASH, "-").replace(SECTION, "section "), file=out, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(args[0])
    report_path = target / "boot-report.json" if target.is_dir() else target
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {report_path}: {exc}", file=sys.stderr)
        return 2
    hs = Path(__file__).resolve().parent.parent / "docs" / "HARD_STOPS.md"
    md = hs.read_text(encoding="utf-8") if hs.is_file() else None
    text = select(report, md, report_path.parent)
    if text is None:
        return 1
    emit(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
