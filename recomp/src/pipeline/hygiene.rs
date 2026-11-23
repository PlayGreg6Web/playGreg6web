//! Output-directory hygiene: build outputs are derived from game code and data, so they may only
//! be written outside every git work tree, or inside one only below a gitignored directory (by
//! convention the repo's `work/`).
//!
//! The check is done in-process, without running `git`:
//! 1. `--out` is made absolute (relative to the current directory) and normalised lexically
//!    (`.` dropped, `..` pops); the longest existing prefix is then canonicalised so symlinks and
//!    (on Windows) drive-letter spellings resolve to the real location.
//! 2. The enclosing work tree is the nearest ancestor (or the directory itself) containing a
//!    `.git` entry: a directory for a normal checkout, a file for linked worktrees and
//!    submodules. No `.git` anywhere above means the directory is outside every repository.
//! 3. Inside a work tree, the directory is accepted only if it or one of its ancestors below the
//!    root is ignored. Ignore rules come from `.git/info/exclude` (normal checkouts only) and from
//!    every `.gitignore` between the root and the directory, applied with git's precedence
//!    (deeper files override shallower ones, the last matching line wins, `!` re-includes). The
//!    pattern syntax follows gitignore(5): `#` comments, `\` escapes, `!` negation, trailing `/`
//!    (directory only; every path tested here is a directory), leading or inner `/` anchors the
//!    pattern to the `.gitignore`'s directory, `*`/`?`/`[...]` globs that never cross `/`, and
//!    `**` segments that match any number of directories. Matching is case-sensitive.
//!
//! Global excludes (`core.excludesFile`) are deliberately not consulted: they live outside the
//! repository, so a directory ignored only there is still visible to other clones. Anything the
//! parser cannot prove ignored is refused, so every approximation errs on the side of refusing.

use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Checks that `out` may receive build outputs (see the module docs). `Err` carries a message
/// for the user.
pub fn check_output_dir(out: &Path) -> Result<(), String> {
    let absolute = if out.is_absolute() {
        out.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|e| format!("cannot determine the current directory: {e}"))?
            .join(out)
    };
    let path = resolve(&normalize(&absolute));
    if path.exists() && !path.is_dir() {
        return Err(format!(
            "output path '{}' exists and is not a directory",
            out.display()
        ));
    }
    let Some(root) = path.ancestors().find(|dir| dir.join(".git").exists()) else {
        return Ok(());
    };
    let relative: Vec<String> = path
        .strip_prefix(root)
        .map_err(|_| format!("cannot relate '{}' to its work tree", out.display()))?
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if relative.first().is_some_and(|c| c == ".git") {
        return Err(format!(
            "refusing to write build outputs into git metadata ('{}')",
            out.display()
        ));
    }
    let rules = IgnoreRules::load(root, &relative);
    if (1..=relative.len()).any(|len| rules.is_ignored(&relative[..len])) {
        return Ok(());
    }
    Err(format!(
        "refusing to write build outputs to '{}': it is inside the git work tree '{}' and not in a \
         gitignored directory. Build outputs are derived from game code and data; write them to \
         the gitignored work/ directory instead (e.g. --out work/recomp) or outside the repository",
        out.display(),
        root.display()
    ))
}

/// Lexical normalisation: drops `.`, resolves `..` against the preceding component.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push(component);
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// Canonicalises the longest existing prefix of `path` and appends the rest.
fn resolve(path: &Path) -> PathBuf {
    let mut existing = path;
    let mut rest = Vec::new();
    loop {
        if let Ok(canonical) = fs::canonicalize(existing) {
            let mut out = canonical;
            out.extend(rest.iter().rev());
            return out;
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                existing = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// One parsed ignore line.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Pattern {
    negated: bool,
    /// Matched against the path relative to `base` (otherwise against the last component only).
    anchored: bool,
    segments: Vec<String>,
    /// Number of leading path components of the directory holding the ignore file.
    base: usize,
}

impl Pattern {
    fn parse(line: &str, base: usize) -> Option<Self> {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let line = trim_trailing_spaces(line);
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (negated, body) = match line.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, line),
        };
        let body = body.strip_suffix('/').unwrap_or(body);
        if body.is_empty() {
            return None;
        }
        let anchored = body.contains('/');
        let segments = body
            .trim_start_matches('/')
            .split('/')
            .map(str::to_string)
            .collect();
        Some(Self {
            negated,
            anchored,
            segments,
            base,
        })
    }

    /// Whether the pattern matches the directory `path` (components relative to the work tree).
    fn matches(&self, path: &[String]) -> bool {
        if path.len() <= self.base {
            return false;
        }
        let relative = &path[self.base..];
        if self.anchored {
            match_segments(&self.segments, relative)
        } else {
            relative
                .last()
                .is_some_and(|name| glob(self.segments[0].as_bytes(), name.as_bytes()))
        }
    }
}

/// Trailing spaces are ignored unless escaped with a backslash.
fn trim_trailing_spaces(line: &str) -> &str {
    let mut end = line.len();
    while end > 0 && line.as_bytes()[end - 1] == b' ' {
        if end >= 2 && line.as_bytes()[end - 2] == b'\\' {
            break;
        }
        end -= 1;
    }
    &line[..end]
}

/// Matches pattern segments against path components; a `**` segment spans any number of them.
fn match_segments(pattern: &[String], path: &[String]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((first, rest)) if first == "**" => {
            (0..=path.len()).any(|skip| match_segments(rest, &path[skip..]))
        }
        Some((first, rest)) => path.split_first().is_some_and(|(name, tail)| {
            glob(first.as_bytes(), name.as_bytes()) && match_segments(rest, tail)
        }),
    }
}

/// Glob match of one segment: `*`, `?`, `[...]` (with `!`/`^` negation and ranges), `\` escapes.
fn glob(pattern: &[u8], name: &[u8]) -> bool {
    match pattern.split_first() {
        None => name.is_empty(),
        Some((b'*', rest)) => (0..=name.len()).any(|skip| glob(rest, &name[skip..])),
        Some((b'?', rest)) => !name.is_empty() && glob(rest, &name[1..]),
        Some((b'[', rest)) => match (name.split_first(), class(rest)) {
            (Some((&c, tail)), Some((set, after))) => set.matches(c) && glob(after, tail),
            // An unterminated class is a literal `[`.
            (Some((&c, tail)), None) => c == b'[' && glob(rest, tail),
            (None, _) => false,
        },
        Some((b'\\', rest)) if !rest.is_empty() => {
            name.first() == Some(&rest[0]) && glob(&rest[1..], &name[1..])
        }
        Some((&p, rest)) => name.first() == Some(&p) && glob(rest, &name[1..]),
    }
}

struct Class {
    negated: bool,
    ranges: Vec<(u8, u8)>,
}

impl Class {
    fn matches(&self, c: u8) -> bool {
        self.ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&c)) != self.negated
    }
}

/// Parses the body of a `[...]` class (after the `[`). Returns the class and the pattern after `]`.
fn class(pattern: &[u8]) -> Option<(Class, &[u8])> {
    let (negated, mut rest) = match pattern.first() {
        Some(b'!' | b'^') => (true, &pattern[1..]),
        _ => (false, pattern),
    };
    let mut ranges = Vec::new();
    let mut first = true;
    loop {
        let (&c, tail) = rest.split_first()?;
        if c == b']' && !first {
            return Some((Class { negated, ranges }, tail));
        }
        first = false;
        if tail.len() >= 2 && tail[0] == b'-' && tail[1] != b']' {
            ranges.push((c, tail[1]));
            rest = &tail[2..];
        } else {
            ranges.push((c, c));
            rest = tail;
        }
    }
}

/// Every ignore rule that can apply to a path below the work tree, lowest precedence first.
struct IgnoreRules {
    patterns: Vec<Pattern>,
}

impl IgnoreRules {
    /// Loads `.git/info/exclude` and the `.gitignore` of the root and of every directory on the
    /// way to `relative` (exclusive of the last component: a directory's own `.gitignore` only
    /// applies to its contents).
    fn load(root: &Path, relative: &[String]) -> Self {
        let mut patterns = Vec::new();
        let mut add = |file: PathBuf, base: usize| {
            if let Ok(text) = fs::read_to_string(file) {
                patterns.extend(text.lines().filter_map(|line| Pattern::parse(line, base)));
            }
        };
        let git = root.join(".git");
        if git.is_dir() {
            add(git.join("info").join("exclude"), 0);
        }
        let mut dir = root.to_path_buf();
        for (depth, component) in relative.iter().enumerate() {
            add(dir.join(".gitignore"), depth);
            dir.push(component);
        }
        Self { patterns }
    }

    fn is_ignored(&self, path: &[String]) -> bool {
        self.patterns
            .iter()
            .rev()
            .find(|p| p.matches(path))
            .is_some_and(|p| !p.negated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comps(path: &str) -> Vec<String> {
        path.split('/').map(str::to_string).collect()
    }

    fn rules(lines: &[(&str, usize)]) -> IgnoreRules {
        IgnoreRules {
            patterns: lines
                .iter()
                .filter_map(|&(line, base)| Pattern::parse(line, base))
                .collect(),
        }
    }

    #[test]
    fn gitignore_pattern_semantics() {
        let r = rules(&[
            ("# comment", 0),
            ("work/", 0),
            ("/build-*", 0),
            ("out*/", 0),
            ("!out-keep/", 0),
            ("docs/**/gen", 0),
            ("cache[0-9]", 0),
            ("local", 1),
            ("\\#hash", 0),
            ("trailing\\ ", 0),
        ]);
        let ignored = |p: &str| r.is_ignored(&comps(p));
        assert!(ignored("work"));
        assert!(
            ignored("a/b/work"),
            "unanchored patterns match at any depth"
        );
        assert!(ignored("build-x"));
        assert!(!ignored("sub/build-x"), "leading / anchors to the root");
        assert!(ignored("out-a"));
        assert!(!ignored("out-keep"), "! re-includes");
        assert!(ignored("docs/gen"));
        assert!(ignored("docs/a/b/gen"), "** spans directories");
        assert!(!ignored("docs/a/b/gen2"));
        assert!(ignored("cache7"));
        assert!(!ignored("cachex"));
        assert!(
            ignored("sub/local"),
            "base-relative pattern from sub/.gitignore"
        );
        assert!(
            !ignored("local"),
            "pattern of sub/.gitignore does not apply above it"
        );
        assert!(ignored("#hash"));
        assert!(ignored("trailing "));
        assert!(!ignored("src"));
    }

    #[test]
    fn glob_classes_and_escapes() {
        assert!(glob(b"a?c", b"abc"));
        assert!(!glob(b"a?c", b"ac"));
        assert!(glob(b"[!a]x", b"bx"));
        assert!(!glob(b"[!a]x", b"ax"));
        assert!(glob(b"[]]", b"]"));
        assert!(glob(b"[a", b"[a"), "unterminated class is literal");
        assert!(glob(b"\\*", b"*"));
        assert!(!glob(b"\\*", b"x"));
        assert!(glob(b"*.pgim", b"game.pgim"));
    }

    #[test]
    fn normalizes_dot_segments() {
        let p = normalize(Path::new("/a/./b/../c"));
        assert_eq!(p, PathBuf::from("/a/c"));
    }
}
