"""tools/screen_flow.py: screen detection on recorded present_samples (run-length fixtures of real boots,
tests/fixtures/present_flow_runs.json: draws per present only, no game data) and input step reporting."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import run_native_boot as rnb  # noqa: E402
import screen_flow as sf  # noqa: E402

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "present_flow_runs.json").read_text())


def expand(runs):
    out = []
    for draws, start, end, n in runs:
        for i in range(n):
            out.append({"draws": draws, "wall_ms": start + (end - start) * i / max(1, n - 1)})
    return out


def flow_of(name, legacy=False):
    fx = FIXTURES[name]
    return sf.detect_flow(expand(fx["runs"]), fx["milestones"], fx["music_init_seen"], legacy)


class ClassifyTests(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual([sf.classify_draws(d) for d in (0, 1, 2, 4, 6, 9, 10, 14, 28, 37, 38, 520)],
                         ["startup", "startup", "loading-artwork", "loading-artwork", "loading-artwork", "ui",
                          "ui", "intro-video", "ui-busy", "ui-busy", "scene", "scene"])

    def test_legacy_only_on_request(self):
        self.assertEqual(sf.classify_draws(37), "ui-busy")
        self.assertEqual(sf.classify_draws(37, legacy=True), "legal-notices")
        self.assertEqual(sf.classify_draws(44, legacy=True), "legal-notices")
        self.assertEqual(sf.classify_draws(50, legacy=True), "legal-viewer")
        self.assertEqual(sf.classify_draws(212, legacy=True), "scene")


class RecordedFlowTests(unittest.TestCase):
    def check_flow(self, name, scene_between):
        f = flow_of(name)
        self.assertEqual(f["flow"], "scene")
        lo, hi = scene_between
        self.assertTrue(lo <= f["scene_ms"] / 1000 <= hi, f["scene_ms"])
        order = [s["screen"] for s in f["screens"]]
        # intro video, then the loading artwork slideshow, then the UI screens, then the scene
        long_art = max((s for s in f["screens"] if s["screen"] == "loading-artwork"), key=lambda s: s["presents"])
        self.assertLess(f["first_ms"]["intro-video"], long_art["start_ms"])
        self.assertLess(order.index("ui"), order.index("scene"))
        self.assertEqual(order[-1], "scene")
        self.assertIn("BINK_OPEN", f["milestones"])
        self.assertLess(f["milestones"]["FIRST_PRESENT"], f["milestones"]["BINK_OPEN"])
        self.assertLess(f["milestones"]["BINK_OPEN"], f["scene_ms"])
        self.assertTrue(f["music_controller_init"])
        self.assertEqual(f["last_screen"], "scene")

    def test_screen2_shot1(self):
        self.check_flow("shot1", (50, 110))

    def test_screen2_shot2(self):
        self.check_flow("shot2", (50, 110))

    def test_std_shader_pack_run(self):
        f = flow_of("std_pack_attfmt")
        self.assertIn(f["flow"], ("scene", "incomplete"))
        self.assertIn("intro-video", f["first_ms"])
        self.assertIn("loading-artwork", f["first_ms"])
        self.assertNotIn("legal-notices", f["first_ms"])

    def test_no_exact_count_needed(self):
        # shot1 goes through 9 / 10 / 8 draw UI screens: all one `ui` kind, not separate exact-count screens.
        f = flow_of("shot1")
        ui = [s for s in f["screens"] if s["screen"] == "ui"]
        self.assertTrue(ui and all(7 <= s["draws_min"] and s["draws_max"] <= 12 for s in ui))

    def test_blip_is_not_a_scene(self):
        samples = [{"draws": 3, "wall_ms": i * 20} for i in range(50)]
        samples += [{"draws": 54, "wall_ms": 1000 + i * 20} for i in range(5)]  # ~80 ms
        samples += [{"draws": 3, "wall_ms": 1200 + i * 20} for i in range(50)]
        f = sf.detect_flow(samples)
        self.assertEqual(f["flow"], "incomplete")
        self.assertIsNone(f["scene_ms"])

    def test_legacy_setup_flow(self):
        samples = [{"draws": 4, "wall_ms": i * 20} for i in range(30)]
        samples += [{"draws": 37, "wall_ms": 600 + i * 40} for i in range(60)]
        f = sf.detect_flow(samples, legacy=True)
        self.assertEqual(f["flow"], "setup")
        self.assertIn("legal-notices", f["first_ms"])

    def test_empty(self):
        f = sf.detect_flow([])
        self.assertEqual((f["flow"], f["screens"], f["last_screen"]), ("incomplete", [], None))
        self.assertIn("no sustained scene", sf.render(f))


class StepReportTests(unittest.TestCase):
    SCRIPT = rnb.input_script_text(900_000, "menu-to-ingame")

    def test_steps_are_named(self):
        names = [s["name"] for s in sf.script_steps(self.SCRIPT)]
        self.assertEqual(names, ["intro-to-scene", "scene-settle", "movement-rounds*"])
        setup = [s["name"] for s in sf.script_steps(rnb.input_script_text(900_000, "menu-to-ingame-setup"))]
        self.assertEqual(setup[:4], ["legal-notices", "legal-notices-steady", "close-viewer", "legal-terms-screen"])
        self.assertEqual(setup[-1], "movement-rounds*")

    def test_current_flow_script_has_no_exact_count_gate(self):
        waits = [l for l in self.SCRIPT.splitlines() if l.startswith(("wait", "repeat"))]
        self.assertIn("repeat every 2500ms until draws_per_frame>=38 for 2s timeout 180s", waits)
        self.assertFalse([l for l in waits if "==37" in l or "==33" in l])

    def test_timeout_names_the_step(self):
        log = ("[input] 20000 ms: press 0x100 100\n"
               "[input] 200020 ms: repeat (73 round(s)) draws_per_frame>=38 timed out (presents 5000, "
               "draws_per_frame 9)\n"
               "[input] 200020 ms: wait draws_per_frame>=38 timed out (presents 5000, draws_per_frame 9)\n")
        lines = sf.timeout_lines(log, self.SCRIPT)
        self.assertEqual(len(lines), 3)  # + the unfinished movement step
        self.assertIn("step 1/3 'intro-to-scene' TIMED OUT at 200.0 s", lines[0])
        self.assertIn("draws 9 = ui", lines[0])
        self.assertIn("step 2/3 'scene-settle' TIMED OUT", lines[1])

    def test_met_steps_are_silent_and_unfinished_step_is_named(self):
        log = ("[input] 20000 ms: press 0x100 100\n"
               "[input] 70000 ms: repeat (20 round(s)) draws_per_frame>=38 met (presents 3000, "
               "draws_per_frame 120)\n")
        lines = sf.timeout_lines(log, self.SCRIPT, last_draws=9, end_ms=95000)
        self.assertEqual(len(lines), 1)
        self.assertIn("step 2/3 'scene-settle' never finished", lines[0])
        self.assertIn("95.0 s on ui (9 draws)", lines[0])

    def test_expected_runout_step_is_not_reported(self):
        log = ("[input] 70000 ms: repeat (20 round(s)) draws_per_frame>=38 met (presents 3000, draws_per_frame 120)\n"
               "[input] 75000 ms: wait draws_per_frame>=38 met (presents 3100, draws_per_frame 120)\n"
               "[input] 875000 ms: repeat (66 round(s)) presents>=999999999 timed out (presents 9, "
               "draws_per_frame 120)\n")
        self.assertEqual(sf.timeout_lines(log, self.SCRIPT), [])

    def test_fresh_setup_picks_the_setup_script(self):
        self.assertIn("legal-notices", [s["name"] for s in sf.script_steps(
            rnb.input_script_text(900_000, "menu-to-ingame-setup"))])
        self.assertIn("menu-to-ingame-setup", rnb.PRESETS)


if __name__ == "__main__":
    unittest.main()
