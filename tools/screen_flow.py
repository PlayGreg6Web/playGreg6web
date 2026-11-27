"""Screen flow detector for the native boot's input scripts (#2106).

The menu presets of tools/run_native_boot.py used to synchronise on exact draw counts of the fresh-setup
flow (a 37-draw "legal notices" screen). A normal boot of the current build, with or without a shader
pack, goes instead (draws per present, recorded `present_samples`, see the reports under the run dirs):

    startup (0-1) -> intro video (13-15) -> loading artwork slideshow (2 / 4 / 6) -> UI (7-12, ~9-10)
    -> the 3D scene (38 up to ~520, from ~55-95 s)

so detection uses draw-count RANGES that must be sustained, plus the present milestones the report
already has (FIRST_PRESENT, BINK_OPEN) and `music_controller.init_seen`; never an exact count. The legacy
fresh-setup screens (legal notices 37 / 44, legal viewer 33 / 50 / 51) are classified only on request
(`legacy=True`), because their counts collide with the scene's range.

Also: per-step reporting of an input script's wait / repeat steps (named by `# step <name>` comments), so a
timeout says which step missed and what screen the run was on.

CLI: python tools/screen_flow.py <boot-report.json or run dir> [--json]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCENE_MIN_DRAWS = 38
SCENE_HOLD_MS = 1500.0
MIN_PRESENTS = 4

SCREEN_STARTUP = "startup"
SCREEN_INTRO_VIDEO = "intro-video"
SCREEN_LOADING_ART = "loading-artwork"
SCREEN_UI = "ui"
SCREEN_UI_BUSY = "ui-busy"
SCREEN_SCENE = "scene"
SCREEN_LEGAL_NOTICES = "legal-notices"
SCREEN_LEGAL_VIEWER = "legal-viewer"


def classify_draws(draws: int, legacy: bool = False) -> str:
    """Screen kind of one present by its draw count (ranges, not exact counts)."""
    if legacy:
        if draws in (37, 44):
            return SCREEN_LEGAL_NOTICES
        if draws in (33, 50, 51):
            return SCREEN_LEGAL_VIEWER
    if draws >= SCENE_MIN_DRAWS:
        return SCREEN_SCENE
    if draws <= 1:
        return SCREEN_STARTUP
    if draws <= 6:
        return SCREEN_LOADING_ART
    if draws <= 12:
        return SCREEN_UI
    if draws <= 15:
        return SCREEN_INTRO_VIDEO
    return SCREEN_UI_BUSY


def _runs(samples: Sequence[Mapping[str, Any]], legacy: bool) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for s in samples:
        d = int(s.get("draws") or 0)
        t = float(s.get("wall_ms") or 0)
        kind = classify_draws(d, legacy)
        if runs and runs[-1]["screen"] == kind:
            r = runs[-1]
            r["end_ms"] = t
            r["presents"] += 1
            r["draws_min"] = min(r["draws_min"], d)
            r["draws_max"] = max(r["draws_max"], d)
        else:
            runs.append({"screen": kind, "start_ms": t, "end_ms": t, "presents": 1,
                         "draws_min": d, "draws_max": d})
    return runs


def _sustained(run: Mapping[str, Any]) -> bool:
    if run["screen"] == SCREEN_SCENE:
        return run["end_ms"] - run["start_ms"] >= SCENE_HOLD_MS and run["presents"] >= 3
    return run["presents"] >= MIN_PRESENTS


def detect_flow(samples: Sequence[Mapping[str, Any]], milestones: Sequence[Mapping[str, Any]] = (),
                music_init_seen: bool | None = None, legacy: bool = False) -> dict[str, Any]:
    """Screens a run went through, from `present_samples` ({wall_ms, draws, ...}).

    Returns {flow, screens: [{screen, start_ms, end_ms, presents, draws_min, draws_max}, ...] (sustained runs
    only, adjacent same-kind runs merged), first_ms: {screen: first sustained start}, scene_ms, last_screen,
    milestones: {FIRST_PRESENT / BINK_OPEN: elapsed_ms}, music_controller_init}. `flow` is "scene" once a
    sustained 3D scene was seen, "setup" if only legacy setup screens (legacy=True) were, else "incomplete".
    """
    screens: list[dict[str, Any]] = []
    for r in _runs(samples, legacy):
        if not _sustained(r):
            continue
        if screens and screens[-1]["screen"] == r["screen"]:
            p = screens[-1]
            p["end_ms"] = r["end_ms"]
            p["presents"] += r["presents"]
            p["draws_min"] = min(p["draws_min"], r["draws_min"])
            p["draws_max"] = max(p["draws_max"], r["draws_max"])
        else:
            screens.append(dict(r))
    first: dict[str, float] = {}
    for s in screens:
        first.setdefault(s["screen"], s["start_ms"])
    ms: dict[str, float] = {}
    for m in milestones:
        name = m.get("milestone")
        if name in ("FIRST_PRESENT", "BINK_OPEN", "FIRST_VISIBLE_FRAME") and name not in ms:
            ms[name] = float(m.get("elapsed_ms") or 0)
    flow = "scene" if SCREEN_SCENE in first else "setup" if (
        SCREEN_LEGAL_NOTICES in first or SCREEN_LEGAL_VIEWER in first) else "incomplete"
    last = classify_draws(int(samples[-1].get("draws") or 0), legacy) if samples else None
    return {"flow": flow, "screens": screens, "first_ms": first, "scene_ms": first.get(SCREEN_SCENE),
            "last_screen": last, "milestones": ms, "music_controller_init": music_init_seen}


def detect_from_report(report: Mapping[str, Any], legacy: bool = False) -> dict[str, Any]:
    mc = report.get("music_controller")
    seen = mc.get("init_seen") if isinstance(mc, Mapping) else None
    return detect_flow(report.get("present_samples") or [], report.get("milestones") or [], seen, legacy)


# ---------------------------------------------------------------------------
# Input script steps: which wait / repeat step missed
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"^\s*#\s*step\s*:?\s*(.+?)\s*$", re.IGNORECASE)
_WAIT_RE = re.compile(r"^\s*(wait|repeat)\b(.*)$")
_LOG_RE = re.compile(r"^\[input\] (\d+) ms: (wait|repeat)\b.*? (met|timed out) \(presents (\d+), "
                     r"draws_per_frame (\d+)\)\s*$")


def script_steps(script_text: str) -> list[dict[str, str]]:
    """wait / repeat steps of an input script in order: {name, line}. The name is the `# step <name>` comment
    right before the step (else `step N`)."""
    steps: list[dict[str, str]] = []
    pending: str | None = None
    for raw in script_text.splitlines():
        m = _STEP_RE.match(raw)
        if m:
            pending = m.group(1)
            continue
        m = _WAIT_RE.match(raw)
        if m and not raw.startswith((" ", "\t")):
            steps.append({"name": pending or f"step {len(steps) + 1}", "line": raw.strip()})
            pending = None
    return steps


def step_outcomes(log_text: str, script_text: str) -> list[dict[str, Any]]:
    """One entry per finished wait / repeat step in the harness log ('[input] T ms: wait <cond> met|timed out
    (presents P, draws_per_frame D)'), matched in order to the script's steps."""
    steps = script_steps(script_text)
    out: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        m = _LOG_RE.match(line.strip())
        if not m:
            continue
        i = len(out)
        s = steps[i] if i < len(steps) else {"name": f"step {i + 1}", "line": "?"}
        out.append({"index": i + 1, "name": s["name"], "line": s["line"], "ms": int(m.group(1)),
                    "met": m.group(3) == "met", "presents": int(m.group(4)), "draws": int(m.group(5)),
                    "screen": classify_draws(int(m.group(5)), True)})
    return out


def timeout_lines(log_text: str, script_text: str, last_draws: int | None = None,
                  end_ms: float | None = None) -> list[str]:
    """Digest lines naming each step that timed out, and the step the run ended in (never finished)."""
    outcomes = step_outcomes(log_text, script_text)
    steps = script_steps(script_text)
    lines = [f"input step {o['index']}/{len(steps)} '{o['name']}' TIMED OUT at {o['ms'] / 1000:.1f} s "
             f"({o['line']}; presents {o['presents']}, draws {o['draws']} = {o['screen']})"
             for o in outcomes if not o["met"] and not o["name"].endswith("*")]  # `*`: runs out by design
    if len(outcomes) < len(steps) and "[input]" in log_text:
        nxt = steps[len(outcomes)]
        nxt = {**nxt, "name": nxt["name"].rstrip("*")}
        where = ""
        if end_ms is not None:
            where = f" at {end_ms / 1000:.1f} s"
        if last_draws is not None:
            where += f" on {classify_draws(last_draws, True)} ({last_draws} draws)"
        lines.append(f"input step {len(outcomes) + 1}/{len(steps)} '{nxt['name']}' never finished: the run "
                     f"ended{where} ({nxt['line']})")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def render(flow: Mapping[str, Any]) -> str:
    lines = [f"flow: {flow['flow']}; scene from {flow['scene_ms'] / 1000:.1f} s" if flow["scene_ms"] is not None
             else f"flow: {flow['flow']}; no sustained scene (last screen {flow['last_screen']})"]
    if flow["milestones"]:
        lines.append("milestones: " + ", ".join(f"{k} {v / 1000:.1f} s" for k, v in flow["milestones"].items()))
    if flow["music_controller_init"] is not None:
        lines.append(f"music_controller_init: {flow['music_controller_init']}")
    for s in flow["screens"]:
        lines.append(f"  {s['start_ms'] / 1000:7.1f}-{s['end_ms'] / 1000:7.1f} s  {s['screen']:<16} "
                     f"draws {s['draws_min']}-{s['draws_max']}  ({s['presents']} presents)")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    p = Path(args[0])
    if p.is_dir():
        p = p / "boot-report.json"
    flow = detect_from_report(json.loads(p.read_text(encoding="utf-8")), legacy="--legacy" in argv)
    print(json.dumps(flow, indent=2) if "--json" in argv else render(flow))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
