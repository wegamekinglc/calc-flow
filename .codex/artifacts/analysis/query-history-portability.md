# Query history and portability implementation evidence

DAL-511 / S0a-2, 2026-11-11. Base `2114e374441fa5099902f511bf23914c286d0c87`;
runtime implementation and native build commit
`93f06e2784aa8eeccaf6c74aff7cbcceddbe0e9f`. The final evidence commit adds
this record, the provenance JSON, and strengthens the nullable input fixture;
runtime and Python package sources remain identical to the build commit.

The controlling [API note](../api-notes/query-history-portability.md) is preserved
unchanged. The existing dedicated checkout and branch were reused; the branch
is now `feature/dal-511-query-history-portability`. No contract deviation.

## Delivered behavior and files

- `crates/calc-flow/src/query_analysis.rs`, `query_analysis/sql.rs`, and
  `pipeline/analysis.rs`: owned facts, revisions, stable paths/codes, conservative
  AST proofs, consumption-edge propagation, and asynchronous checked compilers.
  Crate exports, the typed error, and shared validator visibility connect these
  to the existing compilers.
- `crates/calc-flow-python/src/config.rs`, `error.rs`, and
  `python/calc_flow/_native.pyi`: the data-only analysis seam and CompileError
  report projection.
- `python/calc_flow/symbolic/query_analysis.py`, `analyzer.py`, `program.py`,
  `errors.py`, package exports, `compute.py`, and `stream.py`: immutable public
  models, logical addresses, original issue projection, and explicit gates under
  the registration lock before preparation-cache access, input wrapping,
  source binding, and job start. Async planning uses the compilation worker.
- Focused Rust/PyO3/Python regressions, `docs/query-history-portability.md`,
  documentation links, and the existing Python/Rust example 29.

The implementation distinguishes lifecycle admission from comparison proof.
No request means `portability=None`. Both unknown and unsupported comparisons
reject strict execution. No execution fingerprint, project format, preparation
key, SQL default, or checkpoint layout/version is changed. Rust's typed
DataFrame/SQL frontend remains the separate S0b work.

## RED evidence

Each production behavior started with an observed focused failure. Accepted
failures below exclude fixture/import setup errors.

- Initial Python report test: missing `AnalysisOptions`; Rust integration
  target: missing public types and checked methods.
- `cargo test -p calc-flow --test query_history_portability`: row-local case
  green, aggregate scope and SQL shape cases red (`Unknown` history/admission).
  A later mode case exposed missing stream-only classification; the snapshot
  edge case observed `Supported` where the original compiler rejects the edge.
- `pytest ... -k strict_execution`: three unexpected `portability` keyword
  failures. The sequence-source test separately observed `__getitem__(0)`
  before the gate.
- `pytest ... -k existing_schema_errors`: two incorrect `supported` reports
  despite original field/SQL errors.
- Composition cases: an unknown upstream was incorrectly labeled append;
  DISTINCT consuming an aggregate snapshot lost the complete-snapshot update.
- Selection cases: unproved per-batch LIMIT was accepted, and LIMIT 0 received
  an invented full-history counterexample. Both now remain unknown.
- Snapshot facts incorrectly changed latest-relation history to current-batch
  in rejected batch mode. Portable time schemas emitted Debug spellings instead
  of `time32[s]` / `time64[us]`.
- PyO3 error test: portability failure fell through to ExecutionError.
  Request sequence ownership and public code vocabulary tests also began red.

## GREEN verification

All final checks use this checkout's sources. The installed development
interpreter supplies CPython 3.13.9, PyArrow 24.0.0, and JAX 0.10.2. Package and
native load paths are asserted in [provenance](query-history-portability.json).

Commands ran from the repository root with:

```bash
export CARGO_TARGET_DIR=target/query-history-portability/cargo
export CARGO_BUILD_JOBS=4
export PYO3_PYTHON=/home/wegamekinglc/anaconda3/bin/python
DEV_PY=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python
FLAGS=(--release
  --config 'profile.release.package.calc-flow.opt-level=0'
  --config 'profile.release.package.calc-flow.codegen-units=256'
  --config 'profile.release.package.calc-flow-python.opt-level=0'
  --config 'profile.release.package.calc-flow-python.codegen-units=256')
```

The package-specific optimization overrides bound local build time; dependency
optimization stays at normal release settings. No Cargo manifest/profile was
changed. These are behavior checks, with no performance claim.

```bash
cargo clippy -p calc-flow -p calc-flow-python --lib --tests "${FLAGS[@]}" -- -D warnings
cargo test -p calc-flow --test query_history_portability "${FLAGS[@]}"
PYTHONPATH="$PWD/target/query-history-portability/python" \
  LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib \
  cargo test -p calc-flow-python --lib "${FLAGS[@]}" \
  query_portability_errors_keep_the_immutable_report
```

Clippy passed; core **7 passed**; focused PyO3 test **1 passed**, 133 deselected.
The PyO3 test requires absolute PYTHONPATH because Cargo runs it in the crate
directory. The original typed compilation error for a missing UDF is retained.

```bash
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/maturin \
  build "${FLAGS[@]}" --interpreter "$PYO3_PYTHON" \
  --out target/query-history-portability/wheels
# Extract only this wheel's native module to the target overlay.
PYTHONPATH="$PWD/target/query-history-portability/python" JAX_PLATFORMS=cpu \
  "$DEV_PY" -m pytest python/tests/test_query_history_portability.py \
  python/tests/test_symbolic_analysis.py python/tests/test_symbolic_compile_cache.py \
  python/tests/test_program_engine.py -q \
  --basetemp=target/query-history-portability/pytest-final
```

Native build passed; Python **122 passed in 3.83s**: 27 focused cases and 95
affected analysis/cache/engine regressions. The overlay's Python files resolve
to this checkout and its extension is extracted from this build, not an
installed package from another checkout. No extension remains in source.

```bash
RUSTDOCFLAGS='-D warnings' cargo doc -p calc-flow --no-deps "${FLAGS[@]}"
RUSTDOCFLAGS='-D warnings' cargo test -p calc-flow --doc "${FLAGS[@]}" \
  pipeline::analysis::PipelineBuilder::compile_batch_checked
cargo run -p calc-flow --example 29_sql_stream_switch "${FLAGS[@]}" -- --engine both
PYTHONPATH="$PWD/target/query-history-portability/python" \
  "$DEV_PY" examples/29_sql_stream_switch.py --engine both
```

Crate rustdoc with warnings denied passed; the selected new doctest passed.
Both existing examples passed: batch 6, stream 1/6. Changed-file Ruff lint and
format, `cargo fmt --all --check`, Python 3.9 syntax parsing of changed modules
and stubs, documentation whitespace/links, and `git diff --check` passed.

## Acceptance evidence and limits

- C01: deterministic row-local reports, SQL filtering/ordering, nullable
  predicates, strict collection/streaming, schemas, immutable facts and escaped
  shared-node addresses. Unsupported requests remain distinct from unknowns.
- C02: both batch and stream preparation caches are warmed before rejected
  accepted-history/latest requests. The diagnostic locates the actual snapshot
  to append edge, including an intervening projection. Ordinary stream SUM to
  SUM still yields 1/7.
- C03: ordinary top-1 still yields `[1]` / `[2]`; DISTINCT and ROW_NUMBER remain
  per batch. Full-history counterexamples reject; unproved selections, CTEs,
  aggregate/HAVING shapes and ordering stay unknown.
- C04: one stream job yields 1/6; the same cached batch plan independently
  executes to 1/5. Simple aggregate scope/update/finality are explicit. Strict
  final relation stays unknown when zero input events leave no last snapshot.
- C13: original type/schema and lifecycle errors, missing UDF registration,
  immutable engine rejection and snapshot sink restrictions remain effective.
  Rejection probes assert **0 input wraps, 0 source iterations, 0 bindings,
  0 job starts**; async analysis runs off the event-loop thread.

The fixed baseline program fingerprint is
`20a49fa51a1c423a8bd4cadbb6b95bc142292f21f3294e4970efe9b45eddb7de`;
native batch and stream fingerprints remain respectively
`0044b2322eff73664b78a51f577af6293c86bed2e4d2cd6ba93b1fc214f733f6` and
`3bb9b534577cd7034059bcb870d041aff073e87bdc4888e49731ea06e78a39d5`.
Project bytes before/after analysis and warm cache ownership are also checked.
These were recorded before replacing the baseline native extension and are
asserted against the final matching build.

Existing DAL-510 checkpoint inventory/capture/recovery evidence is reused.
SQL checkpoint writers/readers, layouts 3/4/5, state version 2, recovery
configuration and fingerprint encoding have no diff. No new checkpoint-byte
comparison is claimed. Full regression, platform/feature/container matrices,
optimized release artifacts and the combined 90% Rust coverage gate remain CI
responsibilities. No benchmark was run. No independent review or merge is
claimed; DAL-511 remains in_progress for the coordinator's reviewer handoff.
