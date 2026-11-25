import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import screen_timeline as st  # noqa: E402


def present(t, draws, res):
    return {"wall_ms": t, "counters": {"draws_issued": draws, "resources_created": res}}


class ScreenTimeline(unittest.TestCase):
    def test_collapse_and_split(self):
        ps = [present(0, 10, 5), present(10, 20, 5), present(20, 31, 5),  # 10, 10, 11 per frame
              present(30, 41, 5), present(40, 91, 5), present(50, 141, 9)]
        segs = st.build_timeline(ps, [5, 35, 36, 45])
        self.assertEqual([s["first_present"] for s in segs], [0, 4, 5])
        self.assertEqual(segs[0]["presents"], 4)
        self.assertEqual(segs[0]["draws_per_frame"], 10)
        self.assertEqual(segs[0]["end_time"], 30)
        self.assertEqual(segs[2]["new_resources_at_start"], 4)
        self.assertEqual([s["input_events_in_segment"] for s in segs], [3, 1, 0])

    def test_cap(self):
        ps = [present(i, i * (i + 1) * 5, 0) for i in range(1, 50)]
        segs = st.build_timeline(ps, max_segments=5)
        self.assertEqual(len(segs), 5)
        self.assertEqual(sum(s["presents"] for s in segs), 49)

    def test_empty_and_parse(self):
        self.assertEqual(st.build_timeline([]), [])
        self.assertEqual(st.parse_input_times("# c\n100 A 50\n250 START\n"), [100.0, 250.0])
        self.assertTrue(st.format_table(st.build_timeline([present(0, 1, 0)])))


if __name__ == "__main__":
    unittest.main()
