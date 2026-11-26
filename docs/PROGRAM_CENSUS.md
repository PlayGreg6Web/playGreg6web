# Program census (programs that cannot run)

`tools/program_census.py` explains why draws are dropped by classifying every shader program that cannot fully run, by reason, with counts. It prints hashes and counts only, never shader bytes. Pure stdlib, no game files needed beyond the pack artifacts.

## Run

```
python tools/program_census.py \
  --pack-report <pack>/mesh.json \                 # translator summary (has per-stage `entries`)
  --pack <pack>/mesh.wgsl.pack \                   # optional: only the header and entry table are read
  --reflect <pack>/mesh.wgsl.pack.reflect.json \   # optional: sidecar
  --boot-report runs/<run>/boot-report.json \      # optional: aggregate drop counters
  --boot-log runs/<run>/boot.log \                 # optional: join failures, no-shader-module lines
  --drop-counts drops.json \                       # optional: {"0x<hash>": draws_lost}
  --json-out census.json --md-out census.md
```

With no output flag the markdown goes to stdout. `--mesh-limit-bytes` (default 32768) sets the per-group mesh output limit.

## What it reports

- Never reached the pack: the translator's `skipped_oversize` / failed / budget counts. Raw-address and BVH programs and big geometry shaders are not packed, so they cannot be named per program from the pack; the counts come from the translator summary. If its `failures` list carries reasons, they are split into raw-address/BVH versus other.
- Packed but not fully runnable, each program attributed to one primary reason (and also counted as a tag): `mesh_oversize`, `not_naga_validated`, `unsupported_ops`, `writable_binding_without_source` (UAV with no source), `binding_without_source`, `no_reflection_entry`. Unresolved bindings are also split by kind.
- Runtime: the boot report's aggregate counters (`draws_no_shader_module`, `draws_mesh_pipeline`, `draws_dropped_by_bridge`, dispatch equivalents, reflection counters), the boot log's reflection and mesh join failures with normalised reasons (for example "no root parameter covers sN") and the programs with no shader module.
- Weighting by draws lost: the boot report has only aggregate counters today, so per-program weighting needs `--drop-counts` (or a future `gpu_work` table of `{hash: count}`, which is picked up automatically). Draws lost are then summed per reason and the top programs listed; programs absent from the pack show as `not_in_pack`.

Tests: `python -m unittest tests.test_program_census`.
