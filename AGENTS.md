# Agent Instructions

## Hard Rules
- **No copyrighted game files or game source code in this repo.** Never commit, paste, or add game binaries, keys, `.rpf`/`.xvd`/`.xvc`/`.ttpf`/`.awc` archives, extracted assets, decrypted dumps, or generated game shaders. All user-supplied game data and generated outputs belong strictly in gitignored local folders (`local-data/`, `work/`) referenced through `.env`.
- **Keep `README.md` small and clean.** Do not bloat it with long checklists, redundant repo trees, or unrequested sections. Only update `README.md` when explicitly asked or when a core architectural fact changes.
- **Don't just add stuff.** Only create files, dependencies, or abstractions that are directly needed for the current task.

## Start-Here Playbook (When a User Says "Help Work On This")
Follow these steps whenever starting a session with a new or returning contributor:

1. **Read the project context:**
   - [docs/END_GOAL.md](./docs/END_GOAL.md) — what the finished product is (and non-goals).
   - [docs/WORKSTREAMS.md](./docs/WORKSTREAMS.md) — the 5 parallel tracks and open tasks.
   - ABI contracts when touching cross-track boundaries: [docs/io_manifest.md](./docs/io_manifest.md), [docs/wasm_abi.md](./docs/wasm_abi.md), [docs/gpu_commands.md](./docs/gpu_commands.md).
2. **Check the user's local machine & game path:**
   - Run `python tools/local_env.py --check`.
   - If `GTA6_GAME_DIR` is **NOT SET**, ask the user where their local Xbox Series game folder is (or whether they want to work on a task that does not require game files).
   - When the user provides a path, save it into gitignored `.env` via `python tools/local_env.py --set-game "<path>"`. Never copy game files into the repo.
3. **Claim a task & branch cleanly (avoid duplicate work — and NEVER stall):**
   - Check open GitHub Issues (`gh issue list --state open`), open PRs (`gh pr list --state open`), and [docs/WORKSTREAMS.md](./docs/WORKSTREAMS.md) for any task already marked `IN PROGRESS` or assigned to someone else — **never start a claimed task**.
   - Match the user's installed toolchains (`tools/local_env.py --check`) and whether they have local game files to an unclaimed task, then **claim it first** (create/assign a GitHub Issue via `gh issue create --assignee "@me"` and/or mark it `IN PROGRESS` in `docs/WORKSTREAMS.md`).
   - **If every listed task in `docs/WORKSTREAMS.md` is already `DONE` or `IN PROGRESS`, DO NOT stop and say "there is nothing left to take"!** Instead:
     1. Read [docs/END_GOAL.md](./docs/END_GOAL.md) (M1 → M2 → M3 → M4) and check open PR file lists (`gh pr view <n> --json files`) to see which files are currently being edited.
     2. Define the next concrete, non-overlapping milestone task needed to boot and play the real game (strictly disjoint files from open PRs), claim it with a new GitHub Issue (`gh issue create --assignee "@me"`), append it to `docs/WORKSTREAMS.md` as `IN PROGRESS`, and immediately start building it.
     3. After completing and opening a PR for one task, **automatically continue** by claiming and starting the next unclaimed task toward M4 unless the user asked you to stop.
   - Work on a dedicated branch named `<track>/<task-slug>` (e.g., `recomp/pe-imports`, `shaders/naga-pass`, `web/opfs-cache`) so multiple contributors don't collide on `main`.
   - If a shared ABI in `docs/` or `runtime/abi/playgta6_abi.h` must change, keep that in a small dedicated PR first.
4. **Verify before finishing:**
   - Run `cargo test --workspace`, `python -m unittest discover -s tests -v`, `node --check` on modified JS files, `node --test tests/*.mjs`, and `ctest` on `runtime/` if CMake is installed.
   - Run `git status` and confirm zero game files, dumps, or `.env` are staged.

## Cross-Contributor Communication (GitHub Issues & PRs)
Multiple humans and AI agents work on this repository in parallel across tracks (`recomp/`, `runtime/`, `shaders/`, `web/`, `tools/`). Communicate proactively through GitHub Issues and PRs so everyone can move faster without colliding:
- **Post findings early and often:** When you uncover a blocker, root cause, live boot/replay metric, shader/reflection mismatch, or cross-track contract gap, post a concise comment on the relevant open issue (or file a new issue and cross-reference it with `#<num>`).
- **Read and reply to active threads:** Check recent comments on related issues before proposing or changing cross-track behavior, answer open questions from other contributors, and clearly note which folders/files you are (and are not) touching.
- **Coordinate cross-track contracts:** When a fix spans tracks (e.g. `shaders/` reflection ↔ `runtime/gpu/` binding, or `runtime/` ↔ `web/`), propose the minimal JSON/ABI shape on the tracking issue before making breaking changes.
- **Always stay sanitized:** In all GitHub issues, comments, and PRs, share only counts, hashes, instruction mnemonics, ABI/register offsets, and error summaries — **never** paste copyrighted game files, binary dumps, generated game shaders/IR, or local host paths.

## Folder Responsibilities
- `web/` — Browser frontend (`index.html`, `loader.js`), dedicated workers (`web/workers/`), and input bridge (`web/input/`). Keep the main thread non-blocking; require Cross-Origin Isolation (`COOP`/`COEP`) for `SharedArrayBuffer`.
- `recomp/` — **Rust** static `x86-64 -> LLVM IR -> Wasm64` recompilation pipeline.
- `runtime/` — **C++20 + Emscripten (`-sMEMORY64 -pthread`)** runtime: shared ABI (`runtime/abi/`), Xbox/Win32 OS shims (`runtime/os/`), and `d3d12_xs -> WebGPU` command bridge (`runtime/gpu/`).
- `shaders/` — **Rust** offline `RDNA2 -> SPIR-V -> WGSL` (`naga`) translation and shader packing.
- `server/` — Lightweight local HTTP server with `COOP`/`COEP` headers and HTTP `Range` (`206 Partial Content`) support.
- `tools/` — **Rust + Python** local setup (`tools/local_env.py`), manifest/bootset builders, and diagnostics.
- `docs/` — End goal, parallel workstreams, ABI contracts, and WebGPU/Wasm64 findings. Store/licence/user hard stops (owner decisions, do not implement): [docs/HARD_STOPS.md](./docs/HARD_STOPS.md).
- `tests/` — Automated unit and integration tests.
- `third_party/` — Open-source submodules/dependencies only (never game files).
