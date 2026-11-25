import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_stale_revert as csr  # noqa: E402


def g(d, *a):
    subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a],
                   check=True, capture_output=True)


def write(d, name, text):
    with open(os.path.join(d, name), "w") as f:
        f.write(text)


class StaleRevert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = self.tmp.name
        g(d, "init", "-q", "-b", "main")
        write(d, "a.txt", "one\n")
        write(d, "b.txt", "b\n")
        g(d, "add", ".")
        g(d, "commit", "-qm", "base")
        g(d, "checkout", "-qb", "pr")

    def tearDown(self):
        self.tmp.cleanup()

    def main_change(self):
        g(self.d, "checkout", "-q", "main")
        write(self.d, "a.txt", "one\ntwo\n")
        g(self.d, "commit", "-qam", "add two")
        g(self.d, "checkout", "-q", "pr")

    def run_cli(self):
        return csr.main(["--repo", self.d, "--base", "main", "--head", "pr"])

    def test_stale_branch_reverting_later_change_fails(self):
        write(self.d, "a.txt", "changed\n")
        g(self.d, "commit", "-qam", "edit")
        write(self.d, "a.txt", "one\n")
        g(self.d, "commit", "-qam", "restore old a")
        self.main_change()
        found = csr.find_reverts(self.d, "main", "pr")
        self.assertEqual([p for p, _ in found], ["a.txt"])
        self.assertIn("add two", found[0][1][0])
        self.assertEqual(self.run_cli(), 1)

    def test_normal_branch_passes(self):
        write(self.d, "b.txt", "b2\n")
        g(self.d, "commit", "-qam", "edit b")
        self.main_change()
        self.assertEqual(self.run_cli(), 0)

    def test_branch_editing_same_file_passes(self):
        write(self.d, "a.txt", "zero\none\n")
        g(self.d, "commit", "-qam", "edit a")
        self.main_change()
        self.assertEqual(csr.find_reverts(self.d, "main", "pr"), [])


if __name__ == "__main__":
    unittest.main()
