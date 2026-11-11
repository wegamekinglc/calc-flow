# DAL-512 relational frontend verification

Baseline: `775ea9738265597de492b848f8e0763962c7041b`.
Branch: `feature/dal-512-relational-frontends`. The task's existing isolated
checkout was reused. The frozen [API note](../api-notes/first-class-relational-frontends.md)
is preserved unchanged. Parent plan v2 and the main review were read through the
Multica attachment CLI, alongside the existing query-history API and evidence.
The runtime brief's three `.codex/references/` files do not exist in this repo;
`AGENTS.md`, the existing implementer contract and the frozen note govern this work.

## Change and design

- Rust `dataframe.rs`, `sql.rs`, `program.rs` and `program/lower.rs` add immutable
  typed relation/scalar declarations, named Programs, schema-only planning,
  QueryAnalysis, fresh compilation and data-only project export. Arithmetic is
  rendered into the existing ExpressionOperator; SQL uses the existing parser,
  schema planner, compiler and runners. There is no separate calculation engine.
- `pipeline/mod.rs` and `pipeline/compile.rs` carry private optional named binding
  maps into the shared graph compiler before runtime routing and delivery proof.
  They must cover the exact graph boundaries. Ordinary low-level builders keep
  their existing naming and fingerprints. Named output identity leaves preserve
  a single shared upstream owner, including repeated aliases.
- Python Program adds exactly-one mode/legacy-engine selection, immutable
  `for_mode`, table schemas, overloads and CompileMode exports. Existing lowering,
  declaration fingerprint bytes, caches, async entry capture, strict gate and
  owned stream implementation remain in use.
- The PyO3 ExecutionPlan getter, native stub and Python BatchExecutionPlan expose
  immutable physical input/output name tuples for public project reload. This
  small binding completion makes the note's physical binding discovery contract
  usable in Python; stream plans retain their existing binding ID properties.
- Tests, the separate public consumer, both existing example 32 programs, the
  crate quickstart and relation guide cover the owning-language interfaces.

The public consumer declares only calc-flow as its calculation dependency, plus
Tokio/async-trait for lifecycle implementation. It has no Python, PyO3,
DataFusion dependency, repository helper, handwritten ports/node IDs or Project
JSON construction. Its Arrow patches preserve the repository's dependency
selection. All outputs are checked against small independent expectations;
unordered consumer results compare multisets.

## Test-first observations

Each production increment followed an observed focused failure. Local Rust
commands used the settings below; package-specific optimization overrides bound
build cost and are not performance evidence. Build/test outputs stay under
`target/relational/`.

```bash
export CARGO_TARGET_DIR=target/relational/cargo
export CARGO_BUILD_JOBS=4
# Append these to the Rust commands shown below:
# --release
# --config 'profile.release.package.calc-flow.opt-level=0'
# --config 'profile.release.package.calc-flow.codegen-units=256'
```

- `cargo test -p calc-flow --test relational_frontend typed_dataframe_plans_without_rows`:
  RED missing public Arrow/input/literal/mode APIs; GREEN 1 test.
- `cargo test -p calc-flow --test relational_program`:
  RED missing Program/CompileOptions/SQL relation; GREEN 1 mixed DAG test,
  including shared outputs, strict proof, warm batch executions and export roundtrip.
- `cargo test -p calc-flow --test relational_frontend`:
  RED missing filter/cast/Boolean operators and immutable input/output builders;
  GREEN 4 tests, including NULL/empty inputs and logical errors.
- `cargo test -p calc-flow --test relational_validation`:
  RED explicit input inference, normalized alias collision, derived field overwrite,
  and unlocated schema errors; GREEN 6 tests. Further focused RED cases found
  special-name paths, unresolved NULL/nonportable schema success, lost temporal
  identity, duplicate SQL projection error projection, read-only SQL error paths,
  and known-empty schema compilation. Final results are recorded below.
- `PYTHONPATH=target/relational/python python -m pytest python/tests/test_relational_frontend.py -q`:
  initial mode selection RED (4 failures/1 existing pass), then GREEN 5;
  schema-only test RED missing collect_schema, then GREEN 1.
- The public Python export/reload test observed the missing physical binding
  getters before PyO3/wrapper/stub implementation. Its final GREEN uses the
  rebuilt native artifact recorded in provenance, not the baseline binary.

Preliminary Python checks used this checkout's source adapters and a copied
baseline native only to establish adapter RED/GREEN while core compilation was
underway. Those are superseded by final matched-source verification below.
Existing declaration comparison uses `identical`, exported data sources carry
empty row placeholders, arrays use the native backend, and Batch uses
`from_pyarrow`/`to_pyarrow`; the new tests follow those existing contracts.
The old engine signature-reflection assertion was updated exactly as the frozen
note requires, retaining missing/conflict behavior assertions. No result oracle,
historical fixture, coverage floor, lint rule or other gate was relaxed.

## Final edge contract

The independent consumer observed a focused RED for unsigned negation: native
SQL rejected it with a generic schema error after the typed layer admitted the
operand. The typed layer now uses the same signed-integer/floating restriction
as Python and returns a located `unsupported_type` before native planning. That
regression also checks JSON-subscript paths for a feature containing a dot.
After the 26-test pass, `dataframe.rs` changed only the unary check and
named-feature path formatting. The independent export/reload compilation then
exposed generated identifiers exceeding the existing 64-character project-v3
limit. `program/lower.rs` now shortens internal physical IDs while keeping full
declaration fingerprints; the project validator is unchanged. The frontend,
Program and validation targets and consumer cover these final corrections; the
unaffected low-level query-history target result is reused.

## Verification boundary

This is S0b row-local delivery: no DataFrame aggregate, replacement consumer,
finite relational operator expansion, comprehensive Polars compatibility,
external-provider isolation or batch query checkpoint continuation was added.
Old SQL snapshot accumulation, native mode/cache/checkpoint identities and
fresh-job ownership remain the existing contracts.

Full workspace regression, connector services, real coverage and cross-platform
execution remain CI gates. No benchmark or performance suite was run. Independent
dal-reviewer review is required before merge; opening this PR does not complete
the issue's merge acceptance criterion.

## Acceptance evidence map

| Acceptance      | Executed evidence                                                                      |
| --------------- | -------------------------------------------------------------------------------------- |
| A01 / A02       | Independent public consumer: four graph forms, schema/analyze/explain, batch/stream.   |
| A03             | Python exports/stub, async mapping capture, native bridge cancellation, owned close.   |
| A04             | Both languages: both mixed directions, shared outputs, distinct compiled modes.        |
| A05 row-local   | NULL-only, empty and duplicate rows, casts/predicates, schema/error and lineage tests. |
| A07             | Mode/engine matrix, declaration copies/inference, legacy identities and warm state.    |
| A08             | Both languages: data-only export, reload, compile and execute via public bindings.     |
| A09 / C01 / C13 | Existing history/strict-gate regressions plus new logical schema and mode errors.      |

## Core commands and results

Commands below use the `CARGO_TARGET_DIR`, jobs and core release overrides from
the test-first section (release options precede Clippy's `--`). All exited zero.
The final three Rust targets contain
18 tests (4 frontend, 1 Program, 13 validation). The unchanged low-level
`query_history_portability` target's 9-test result is reused from the earlier
26-test run; only the two new typed frontend files changed afterward.

```bash
cargo test -p calc-flow --test relational_frontend --test relational_program --test relational_validation
cargo clippy -p calc-flow --lib --test relational_frontend --test relational_program --test relational_validation --example 32_sql_expression_pipeline -- -D warnings
cargo run -p calc-flow --example 32_sql_expression_pipeline
cargo test -p calc-flow --doc dataframe
RUSTDOCFLAGS='-D warnings' cargo doc -p calc-flow --no-deps
cargo fmt --all --check
rustfmt --edition 2024 --check crates/calc-flow/tests/relational-consumer/src/main.rs
python scripts/verify_complexity_gates.py
git diff --check
git diff 775ea9738265597de492b848f8e0763962c7041b --exit-code -- schemas/project-v3.schema.json .codex/agents .claude/agents
```

The example prints `net = [Some(18.0), Some(27.0)]`; the focused rustdoc example
passes (1 test). Core/test/example Clippy and rustdoc use warnings as errors.
The complexity ratchet holds. No new complexity/lint suppression was added.
The standalone consumer's exact final command is:

```bash
CARGO_TARGET_DIR=target/relational/cargo CARGO_BUILD_JOBS=4 cargo run \
  --manifest-path crates/calc-flow/tests/relational-consumer/Cargo.toml \
  --locked --offline --release \
  --config 'profile.release.package.calc-flow.opt-level=0' \
  --config 'profile.release.package.calc-flow.codegen-units=256' \
  --config 'profile.release.package.relational-public-consumer.opt-level=0' \
  --config 'profile.release.package.relational-public-consumer.codegen-units=256'
```

It prints four `schema, analysis, batch, owned stream, export/reload passed`
lines, then `same batch plan 1/5; same stream job 1/6; new stream job 5`.
The fixture binds Disabled watermarks because its source declares NeverEmits;
the runner's source capability preflight is preserved. Its source opens and
closes exactly once per job, and completed jobs report no errors.

## Build provenance

Product source commit: `e5ef248b9853e095be651b5bf06c2286dbadbf86`.
The subsequent evidence commit adds a preservation test and does not change
product source. Machine-readable
artifact hashes, source trees, final Python counts and build commands accompany
this report in `first-class-relational-frontends.json`.

Toolchain: Rust/Cargo 1.94.0, CPython 3.13.9, PyArrow 24.0.0 on Linux x86_64.
The frozen note SHA256 is
`f3185b9217233a10f2fc584ecceb383cafbef8345bc92a3a8ea3e979b7084e71`,
byte-identical to the frozen platform attachment. All local build artifacts and
logs remain under `target/relational/`; no native module is left in source.

## Matched Python/native verification

The following build used this checkout and the product commit above. Maturin's
frontend interpreter is a pre-existing environment; the explicit target and
`PYO3_PYTHON` select CPython 3.13.9. No package was installed globally.

```bash
export PYO3_PYTHON=/home/wegamekinglc/anaconda3/bin/python
export CARGO_TARGET_DIR=target/relational/cargo
export CARGO_BUILD_JOBS=4
/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python -m maturin build \
  --release --interpreter /home/wegamekinglc/anaconda3/bin/python \
  --out target/relational/wheels \
  --config 'profile.release.package.calc-flow.opt-level=0' \
  --config 'profile.release.package.calc-flow.codegen-units=256' \
  --config 'profile.release.package.calc-flow-python.opt-level=0' \
  --config 'profile.release.package.calc-flow-python.codegen-units=256'
```

Build passed in 7m45s. The resulting wheel has the local `linux_x86_64` tag;
Maturin warned it is not a PyPI-uploadable compatible-platform wheel. This is
local verification, not release artifact certification. All 55 tracked Python
package files in the unpacked wheel match the checkout byte for byte. Tests
imported both Python and native from `target/relational/wheel-python/`.

Wheel SHA256:
`c3f6dc948dadc5380147a96bd33b16ef89faeb4f00be565e7501dfe31f8c2fe8`.
Native SHA256:
`8faf47d39df917ead9aa837b57890fef89597894323eaee673cc111b9721a543`.

```bash
export PYTHONPATH=target/relational/wheel-python
export JAX_PLATFORMS=cpu
python -m pytest python/tests/test_relational_frontend.py python/tests/test_program_engine.py python/tests/test_symbolic_program.py python/tests/test_symbolic_compile_cache.py python/tests/test_query_history_portability.py python/tests/test_sql_pipeline.py -q --junitxml=target/relational/python-final.xml
python -m pytest python/tests/test_stream_results.py -q -k 'close_waits_for_compile_worker or releases_source_at_eof or cancels_idle_source or concurrent_consumers or named_branches_consume_source_once or start_failure_closes_only_acquired or cancellation_at_native_start_result or close_before_entry or multi_alias_sql_before_iterator_acquisition' --junitxml=target/relational/python-owned-final.xml
python -m pytest python/tests/test_compute.py -q -k 'async_collection_snapshots_mapping_and_forwards_options or async_compute_cancellation_waits_for_compile_worker or async_cancellation_before_start_does_not_leak_coroutines or async_cancellation_drains_native_bridge' --junitxml=target/relational/python-async-final.xml
python examples/32_sql_expression_pipeline.py
```

Results: **135 + 11 + 5 = 151 passed**, no failures. Python example output:
`{'order_id': [1, 3], 'net': [18.0, 27.0]}`.
The additional preservation regression verifies Python same-plan 1/5,
same-job 1/6 and a new job's 5 with the mode wrappers; it passed without a
production change. Existing regression assertions also retain exact Python
declaration/native fingerprints and the legacy SUM-of-SUM stream result 1/7.
Strict rejection is tested before input wrapping, source acquisition and job
creation. Cancellation tests await compile workers/native bridges and owned
source/job cleanup, including failed start and early close.

Changed Python files pass Ruff lint/format. The API and native stub parse under
Python 3.9 grammar; this does not claim an executed Python 3.9 runtime matrix.

PyO3 library Clippy passes with `-D warnings` (3m04s), using `PYO3_PYTHON`,
target/jobs and the same core/PyO3 release overrides as the wheel build:

```bash
cargo clippy -p calc-flow-python --release --lib \
  --config 'profile.release.package.calc-flow.opt-level=0' \
  --config 'profile.release.package.calc-flow.codegen-units=256' \
  --config 'profile.release.package.calc-flow-python.opt-level=0' \
  --config 'profile.release.package.calc-flow-python.codegen-units=256' \
  -- -D warnings
```

Standalone default core build also passes (4m51s), without selecting the Python
binding package. The independent consumer's separate dependency graph likewise
contains no Python/PyO3 crate:

```bash
CARGO_TARGET_DIR=target/relational/cargo CARGO_BUILD_JOBS=4 \
  cargo build -p calc-flow --lib --release \
  --config 'profile.release.package.calc-flow.opt-level=0' \
  --config 'profile.release.package.calc-flow.codegen-units=256'
```

Relative documentation paths in the six changed guides/READMEs resolve. Final
product source trees remain identical to the native build commit; the JSON
records their Git object IDs. Remaining working-tree changes are evidence and
the additional Python preservation test only.
