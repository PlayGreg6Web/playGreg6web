# LLVM toolchain for the recompiler

The recompiler shells out to `opt` + `llc` (or `clang`). Which LLVM it finds changes codegen, so
discovery is explicit and version-checked (`recomp/src/lift/compile.rs`).

Search order:
1. `PLAYGTA6_LLC` - path to `llc` or `clang`.
2. `PLAYGTA6_LLVM_BIN_DIR` - directory holding matching `llc` + `opt` (or `clang`). Recommended.
3. `llc` on `PATH`, then `clang` on `PATH`.
4. The rustup `llvm-tools` component (its LLVM follows the Rust release, e.g. 23, not CI's).

The found tool's major version is compared with the expected one (default **18**, the LLVM of
the `ubuntu-latest` runner the CI uses; `ci.yml` does not install LLVM itself). On a mismatch a
`warning: ... is LLVM N but CI/production expects LLVM M` line is printed to stderr. Set
`PLAYGTA6_LLVM_MAJOR=<n>` to expect another major (e.g. when deliberately testing a newer LLVM).

Example (Linux/WSL, matching CI): `export PLAYGTA6_LLVM_BIN_DIR=/usr/lib/llvm-18/bin`.
Windows: point it at the `bin` folder of an installed LLVM 18 release.

Note: the Emscripten `emcc` path (`tools/build_guest_wasm.py`) uses the LLVM bundled with
Emscripten (see R22, LLVM 24) and is independent of this setting.
