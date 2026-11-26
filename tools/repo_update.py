#!/usr/bin/env python3
"""Keep a play checkout current without ever discarding anything.

  repo_update.py check   [--repo DIR] [--no-fetch]  one-line warning when behind origin/main
  repo_update.py update  [--repo DIR]               fetch + fast-forward main

update: only when there are no uncommitted tracked changes: switch to main if HEAD is detached (a checkout
on another branch is left alone), `git fetch origin`, `git merge --ff-only origin/main`, print old -> new.
Anything else prints a clear message and returns 0 so the caller continues with the current code. Never
resets, stashes, force-checks-out or deletes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def _git(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def has_tracked_changes(repo) -> bool:
    rc, out, _ = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return rc != 0 or bool(out)


def current_branch(repo):
    """Branch name, or None when HEAD is detached."""
    rc, out, _ = _git(repo, "symbolic-ref", "--short", "-q", "HEAD")
    return out if rc == 0 and out else None


def behind_count(repo, ref="origin/main"):
    """Commits in ref that HEAD lacks, or None when ref is unknown."""
    rc, out, _ = _git(repo, "rev-list", "--count", f"HEAD..{ref}")
    return int(out) if rc == 0 and out.isdigit() else None


def _short(repo, rev="HEAD"):
    rc, out, _ = _git(repo, "rev-parse", "--short", rev)
    return out if rc == 0 else "?"


def check(repo, fetch=True) -> str:
    """Return the warning line ('' when up to date or unknown)."""
    if fetch:
        _git(repo, "fetch", "-q", "origin")
    n = behind_count(repo)
    if not n:
        return ""
    return (f"WARNING: this checkout is {n} commit{'s' if n != 1 else ''} behind origin/main; "
            "run with -Update to fast-forward (nothing is changed without it).")


def update(repo) -> tuple[bool, str]:
    """Returns (updated, message). Never raises on git-state problems."""
    if has_tracked_changes(repo):
        return False, "update skipped: uncommitted tracked changes; continuing with the current code."
    rc, _, err = _git(repo, "fetch", "origin")
    if rc != 0:
        return False, f"update skipped: git fetch origin failed ({err}); continuing with the current code."
    if current_branch(repo) is None:
        old = _short(repo)
        rc, _, err = _git(repo, "switch", "main")
        if rc != 0:
            return False, (f"update skipped: HEAD is detached at {old} and 'git switch main' failed ({err}); "
                           "continuing with the current code.")
    elif current_branch(repo) != "main":
        return False, (f"update skipped: on branch '{current_branch(repo)}', not main; "
                       "continuing with the current code.")
    old = _short(repo)
    rc, _, err = _git(repo, "merge", "--ff-only", "origin/main")
    if rc != 0:
        why = err.splitlines()[0] if err else "diverged"
        return False, f"update skipped: main cannot fast-forward to origin/main ({why}); continuing with the current code."
    new = _short(repo)
    if new == old:
        return False, f"already up to date at {new}."
    return True, f"updated main: {old} -> {new}."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fast-forward a play checkout without discarding anything.")
    ap.add_argument("mode", choices=["check", "update"])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "check":
        msg = check(a.repo, fetch=not a.no_fetch)
        if msg:
            print(msg)
    else:
        print(update(a.repo)[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
