import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import boot_lock  # noqa: E402

HOLDER = ("import sys,time;sys.path.insert(0,sys.argv[1]);import boot_lock;"
          "l=boot_lock.BootLock(sys.argv[2],'holder');l.acquire();print('held',flush=True);"
          "time.sleep(float(sys.argv[3]));l.release()")


class BootLockTests(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.d = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_two_processes_contend(self):
        p = subprocess.Popen([sys.executable, "-c", HOLDER, str(TOOLS), str(self.d), "1.5"],
                             stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(), "held")
            with open(os.devnull, "w") as out:
                with self.assertRaises(TimeoutError):
                    boot_lock.BootLock(self.d, "b", timeout_s=0.3, poll_s=0.05, out=out).acquire()
                lk = boot_lock.BootLock(self.d, "b", timeout_s=20, poll_s=0.05, out=out)
                lk.acquire()
            self.assertGreater(lk.waited_s, 0.3)
            lk.release()
            self.assertFalse((self.d / "boot.lock").exists())
        finally:
            p.wait(10)

    def test_stale_lock_broken(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        (self.d / "boot.lock").write_text('{"pid": %d, "started": 0, "name": "x"}' % dead.pid)
        with open(os.devnull, "w") as out:
            lk = boot_lock.BootLock(self.d, "n", timeout_s=5, poll_s=0.05, out=out)
            lk.acquire()
        self.assertTrue(lk.broke_stale)
        lk.release()

    def test_disabled(self):
        self.assertTrue(boot_lock.disabled({"PG6_NO_BOOT_LOCK": "1"}))
        self.assertFalse(boot_lock.disabled({"PG6_NO_BOOT_LOCK": "0"}))
        self.assertTrue(boot_lock.disabled({}, True))


if __name__ == "__main__":
    unittest.main()
