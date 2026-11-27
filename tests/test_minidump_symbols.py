import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import minidump_stack as ms  # noqa: E402


class FakeBackend:
    def __init__(self, ok=True, names=None):
        self.ok, self.names, self.loads = ok, names or {}, []

    def load(self, image, base, size):
        self.loads.append((image, base, size))
        return self.ok

    def resolve(self, addr):
        return self.names.get(addr)


class FakeGuest:
    def contains(self, a):
        return 0x140000000 <= a < 0x141000000

    def describe(self, a):
        return "game.exe fn 0x140001000+0x4"


MODS = [(0x7E0000000, 0x1000000, r"C:\x\playgta6_boot.exe"), (0x7FFF0000, 0x10000, r"C:\Windows\ntdll.dll")]


class SymbolizerTests(unittest.TestCase):
    def test_resolves_function_and_line(self):
        be = FakeBackend(names={0x7E0000100: "sub_14344FFD0+0x10 (a.cpp:7)"})
        img = os.path.join("D:bin", "playgta6_boot.exe")
        s = ms.Symbolizer(MODS, be, ["D:bin"], exists=lambda p: p == img)
        self.assertEqual(be.loads, [(img, 0x7E0000000, 0x1000000)])
        self.assertEqual(s.describe(0x7E0000100), "playgta6_boot.exe+0x100  sub_14344FFD0+0x10 (a.cpp:7)")

    def test_system_dll_is_offset_only(self):
        s = ms.Symbolizer(MODS, FakeBackend(), [], exists=lambda p: True)
        self.assertEqual(s.describe(0x7FFF0010), "ntdll.dll+0x10")

    def test_missing_image_degrades(self):
        be = FakeBackend()
        s = ms.Symbolizer(MODS, be, [], exists=lambda p: False)
        self.assertEqual(be.loads, [])
        self.assertTrue(s.notes)
        self.assertEqual(s.describe(0x7E0000100), "playgta6_boot.exe+0x100")

    def test_missing_pdb_symbol(self):
        s = ms.Symbolizer(MODS, FakeBackend(), [], exists=lambda p: True)
        self.assertIn("no symbol", s.describe(0x7E0000100))

    def test_load_failure_degrades(self):
        s = ms.Symbolizer(MODS, FakeBackend(ok=False), [], exists=lambda p: True)
        self.assertEqual(s.describe(0x7E0000100), "playgta6_boot.exe+0x100")
        self.assertTrue(s.notes)

    def test_guest_range_and_unknown(self):
        s = ms.Symbolizer(MODS, None, [], FakeGuest())
        self.assertEqual(s.describe(0x140001004), "guest game.exe fn 0x140001000+0x4")
        self.assertIsNone(s.describe(0x1234))


if __name__ == "__main__":
    unittest.main()
