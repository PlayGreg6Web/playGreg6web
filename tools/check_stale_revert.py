#!/usr/bin/env python3
"""Fail a PR that would undo changes merged to the base branch after its merge-base.

A file is flagged when all of these hold:
  * commits on base that are not in the PR changed it since the merge-base,
  * commits of the PR touched it,
  * the PR's final version is byte-identical to the merge-base version
    (so merging/squashing it puts the pre-change content back).

Usage: check_stale_revert.py [--base origin/main] [--head HEAD] [--repo DIR]
Exit 1 when something would be reverted, 0 otherwise.
"""
import argparse
import subprocess
import sys


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("git %s failed: %s" % (" ".join(args), r.stderr.strip()))
    return r.stdout


def names(repo, *args):
    return {n for n in git(repo, *args).split("\0") if n}


def blob(repo, rev, path):
    r = subprocess.run(["git", "-C", repo, "rev-parse", "-q", "--verify", "%s:%s" % (rev, path)],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def find_reverts(repo, base, head):
    mb = git(repo, "merge-base", base, head).strip()
    base_changed = names(repo, "diff", "--name-only", "-z", "--no-renames", mb, base)
    pr_touched = names(repo, "log", "--name-only", "--format=", "-z", "--no-renames",
                       "%s..%s" % (mb, head))
    out = []
    for path in sorted(base_changed & pr_touched):
        mb_blob = blob(repo, mb, path)
        if blob(repo, head, path) != mb_blob or blob(repo, base, path) == mb_blob:
            continue
        commits = git(repo, "log", "--format=%h %s", "%s..%s" % (mb, base), "--", path)
        out.append((path, [c for c in commits.splitlines() if c]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    found = find_reverts(a.repo, a.base, a.head)
    if not found:
        print("check_stale_revert: ok, nothing merged on %s is undone" % a.base)
        return 0
    print("check_stale_revert: this PR would revert changes already on %s:" % a.base)
    for path, commits in found:
        print("  %s" % path)
        for c in commits:
            print("    reverts %s" % c)
    print("Rebase onto %s and restore those files." % a.base)
    return 1


if __name__ == "__main__":
    sys.exit(main())
