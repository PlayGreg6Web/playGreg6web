import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import repo_update as ru  # noqa: E402


def git(cwd, *a):
    subprocess.run(["git", "-C", cwd, "-c", "user.name=t", "-c", "user.email=t@t", *a],
                   check=True, capture_output=True)


def commit(cwd, name, text="x"):
    with open(os.path.join(cwd, name), "w") as f:
        f.write(text)
    git(cwd, "add", name)
    git(cwd, "commit", "-m", name)


class RepoUpdateTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        r = self.td.name
        self.up, self.work, self.clone = (os.path.join(r, n) for n in ("up.git", "work", "clone"))
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", self.up], check=True)
        git(r, "clone", "-q", self.up, self.work)
        git(self.work, "checkout", "-q", "-B", "main")
        commit(self.work, "a")
        git(self.work, "push", "-q", "origin", "main")
        git(r, "clone", "-q", self.up, self.clone)

    def tearDown(self):
        self.td.cleanup()

    def advance(self):
        commit(self.work, "b")
        git(self.work, "push", "-q", "origin", "main")

    def head(self):
        return ru._short(self.clone)

    def test_check_warns_when_behind(self):
        self.assertEqual(ru.check(self.clone), "")
        self.advance()
        self.assertIn("1 commit behind", ru.check(self.clone))

    def test_update_fast_forwards(self):
        self.advance()
        ok, msg = ru.update(self.clone)
        self.assertTrue(ok)
        self.assertIn("->", msg)
        self.assertEqual(ru.behind_count(self.clone), 0)

    def test_update_from_detached_head(self):
        self.advance()
        git(self.clone, "checkout", "-q", "--detach", "HEAD")
        ok, _ = ru.update(self.clone)
        self.assertTrue(ok)
        self.assertEqual(ru.current_branch(self.clone), "main")
        self.assertEqual(ru.behind_count(self.clone), 0)

    def test_uncommitted_changes_untouched(self):
        self.advance()
        p = os.path.join(self.clone, "a")
        with open(p, "w") as f:
            f.write("dirty")
        before = self.head()
        ok, msg = ru.update(self.clone)
        self.assertFalse(ok)
        self.assertIn("uncommitted", msg)
        self.assertEqual(self.head(), before)
        with open(p) as f:
            self.assertEqual(f.read(), "dirty")

    def test_untracked_files_do_not_block(self):
        self.advance()
        with open(os.path.join(self.clone, "junk"), "w") as f:
            f.write("u")
        self.assertTrue(ru.update(self.clone)[0])

    def test_diverged_is_left_alone(self):
        commit(self.clone, "local")
        self.advance()
        before = self.head()
        ok, msg = ru.update(self.clone)
        self.assertFalse(ok)
        self.assertIn("cannot fast-forward", msg)
        self.assertEqual(self.head(), before)

    def test_other_branch_left_alone(self):
        self.advance()
        git(self.clone, "checkout", "-q", "-b", "feature")
        ok, _ = ru.update(self.clone)
        self.assertFalse(ok)
        self.assertEqual(ru.current_branch(self.clone), "feature")

    def test_up_to_date(self):
        ok, msg = ru.update(self.clone)
        self.assertFalse(ok)
        self.assertIn("up to date", msg)


if __name__ == "__main__":
    unittest.main()
