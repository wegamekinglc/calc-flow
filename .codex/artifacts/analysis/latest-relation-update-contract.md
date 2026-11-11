# Latest-relation update contract — implementation evidence

DAL-513 / S1a, 2026-11-11. The frozen [API note](../api-notes/latest-relation-update-contract.md) is preserved unchanged. Baseline: `f524493cb80546f35b65db0ecb9d2c86f1a2ea2e`. The existing checkout was reused and its branch renamed to `feature/dal-513-latest-relation`.

Validation source: `8f76053f` (full SHA in the companion [evidence JSON](latest-relation-update-contract.json)). Native production build: `7ef99c99e3931379ec45f43515a8717fe84dfc88`. Later changes are explicit default literals in Rust fixtures, a documentation snippet, and the binding test's expected analysis revision. The JSON verifies unchanged core/Python trees and the byte-identical production prefix of the binding error module; its only later binding-source change is inside `cfg(test)`.

## Delivered behavior and files

- Rust `program.rs`, `program/lower.rs`, `pipeline/{mod,analysis,compile}.rs`, `query_analysis{,/sql,/updates}.rs`: immutable Program/builder selection, shared AST proofs, mode-specific mandatory admission, stable logical and physical consumption addresses, and report schema/revision 2.
- `config.rs`, `project_store.rs`, crate exports and `schemas/project-v3.schema.json`: strict `graph.update_semantics`, omitted legacy defaults, non-legacy native identity, and `graph.update_semantics: invalid_literal` validation. Full Rust `PipelineSpec` literals need the new field; repository fixtures and the standalone public consumer demonstrate the migration.
- Python Program, analysis/projection, lowering, compute/stream preparation, public exports, error vocabulary and native stub: one literal selector shared by DataFrame and SQL, registration-scoped admission before caches, row wrapping, bindings and job start, and preservation across internal declaration rewrites.
- PyO3 error projection retains immutable v2 reports on `CompileError`. User guides and the existing standalone consumer cover the public surface. The JSON lists every changed file.

S1a admits known cumulative producers and deterministic row-local snapshot transforms. It rejects ordinary replacement aggregation and unproved consumers. DataFrame aggregation and ordinary SQL replacement consumption remain DAL-514 work.

## Acceptance observations

- Default and explicit legacy SUM→SUM over `[1]` / `[2,3]` emit 1 / 7. A terminal SUM emits 1 / 6. Latest rejects the second append consumer, including after a projection/filter, without a portability request.
- Whole-program rejection precedes source acquisition. Tests warm legacy preparation first, retry latest, and forbid input wrapping, source iteration, binding creation and job start. Ordinary and strict sync/async entries retain zero forbidden calls. Additional portability obligations cannot replace the current mode's update failure.
- SUM→filter emits a named nonempty event followed by an exact-schema empty event. Row-local SQL/CTE retains snapshot/replacement facts and those actual outputs. One Batch containing multiple RecordBatches remains one event. An empty Data event emits an update; an async source with no events emits none.
- DISTINCT, ORDER and LIMIT after snapshots remain unknown; known append aggregation is unsupported. Shared producers retain every offending consumer edge, including escaped output names. Repeated reports are deterministic. A stale v1 report cannot license latest.
- Real managed checkpoint capture/restart tests reject both legacy→latest and latest→legacy identity mismatches with source open/read count 0. Identical terminal recovery succeeds. These are actual runner/storage recoveries, not only fingerprint comparisons.
- Legacy project bytes, Program identity and native batch/stream identities match captured pre-change golden values. Explicit legacy JSON normalizes to the omitted-field bytes. Latest survives export/reload and native rejection, and has a distinct identity. The generated schema equals the schema returned by this native build.
- The same warm latest batch plan produces independent 1 / 5 executions. The standalone consumer also proves one stream job's 1 / 6 and a fresh job's 5.
- Existing snapshot SQL checks retain initialization, empty replacement, whole-event edge rejection, candidate failure/cancellation, durable cuts and terminal restore. The latest terminal snapshot SQL consumer also executes successfully.

## RED and corrections

`pytest python/tests/test_update_semantics.py -q -k update_choice` first failed with the expected missing `Program.with_update_semantics` AttributeError. The focused positive case then passed.

`pytest python/tests/test_update_semantics.py -q -k lowering_rewrite` reproduced loss of `graph.update_semantics` after an internal declaration copy. Preserving the choice at the rewrite boundary made the same assertion pass; the final wheel passes it too.

Affected regression runs exposed changed legacy CTE/ORDER analysis. New proof rules were restricted to explicit latest; existing legacy assertions were retained. The original materialized-sequence empty-input rejection is retained: the no-event probe uses a supported empty async source. A host-bound exported graph uses the existing graph compiler, while the connector compiler retains its source-binding requirement.

Lint fixes use an ordinary generic serde default predicate, borrowed error-report arguments, and explicit enum literals in fixtures. No lint suppression, baseline, CI gate, state layout or dependency policy was relaxed. The binding immutability test now expects the intentionally versioned analysis revision 2 after its rejected mutation.

## Verification and source boundaries

| Check                                     | Observed result                                                             |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| Rust initial affected modules, `f16f7314` | 65 passed; config 30, latest 10, store 15, history 9, relation program 1    |
| Rust follow-up after callback fixes       | 25 passed; latest 10 and project store 15                                   |
| Snapshot SQL compatibility                | 16 passed                                                                   |
| Final wheel: six affected Python modules  | 164 passed, no failure/error/skip                                           |
| Frozen Python exports                     | 1 passed                                                                    |
| PyO3 error mapping, final test source     | 5 passed, including immutable v2 report                                     |
| New Rust selector doctests                | 2 passed                                                                    |
| Standalone Rust core build                | Passed with no PyO3 interpreter configuration                               |
| Independent public Rust consumer          | DF, SQL, DF→SQL→DF→SQL, SQL→DF→SQL, schema, stream, export and state passed |
| Library/affected-target clippy, rustdoc   | Passed with warnings denied                                                 |

The 65-test result is reused at its recorded SHA; it is not presented as a second final-head run. Follow-up execution covers the changed serde/default boundary and latest admission/recovery. Fixture-only lint changes preserve the same LegacyAppend value and pass affected-target compilation.

All commands and result/log hashes are in the JSON. Commands ran from the repository root with these local build-cost settings:

```bash
export CARGO_TARGET_DIR=target/latest/cargo
export CARGO_BUILD_JOBS=4
CORE_FLAGS=(--release
  --config 'profile.release.package.calc-flow.opt-level=0'
  --config 'profile.release.package.calc-flow.codegen-units=256')
FLAGS=("${CORE_FLAGS[@]}"
  --config 'profile.release.package.calc-flow-python.opt-level=0'
  --config 'profile.release.package.calc-flow-python.codegen-units=256')
```

The Cargo artifact cache is reused through a local target symlink; source compilation and package contents belong to this checkout. Standalone core/consumer commands remove PYO3_PYTHON. No repository Cargo profile was changed and these settings make no performance claim.

```bash
cargo test -p calc-flow --test latest_relation --test query_history_portability \
  --test relational_program --test config --test project_store "${CORE_FLAGS[@]}"
cargo test -p calc-flow --test latest_relation --test project_store "${CORE_FLAGS[@]}"
cargo test -p calc-flow --test snapshot_sql_checkpoint_contract "${CORE_FLAGS[@]}"
cargo build -p calc-flow --lib "${CORE_FLAGS[@]}"
cargo run --manifest-path crates/calc-flow/tests/relational-consumer/Cargo.toml \
  --locked "${CORE_FLAGS[@]}"
cargo clippy -p calc-flow -p calc-flow-python --lib "${FLAGS[@]}" -- -D warnings
cargo clippy -p calc-flow --test latest_relation --test query_history_portability \
  --test relational_program --test config --test project_store --test properties \
  --test stream_join_validation "${CORE_FLAGS[@]}" -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo test -p calc-flow --doc "${CORE_FLAGS[@]}" with_update_semantics
RUSTDOCFLAGS='-D warnings' cargo doc -p calc-flow --no-deps "${CORE_FLAGS[@]}"
cargo run -p calc-flow --example gen_v3_schema "${CORE_FLAGS[@]}"
```

The JSON records the exact Maturin command, interpreter and loader configuration. CPython 3.13.9 and PyArrow 24.0.0 come from the shared development environment; the package and extension come from this checkout's newly built wheel. Every packaged `.py`/`.pyi` is checked byte for byte against source. The wheel is a local Linux test artifact.

Final native SHA256: `40f584cb052b5b8e45c4788bb7057a86f40e5c0dfbaf222ba04915c55eda3d83`. Final wheel SHA256: `625c36fc71b1ba64d8cbd3bc23fa0268e58550e960bf4d93a5ae07d13b9887e8`.

```bash
export PYTHONPATH=target/latest/final-python
export JAX_PLATFORMS=cpu
python -m pytest python/tests/test_update_semantics.py \
  python/tests/test_query_history_portability.py python/tests/test_relational_frontend.py \
  python/tests/test_symbolic_compile_cache.py python/tests/test_program_engine.py \
  python/tests/test_symbolic_program.py -q
python -m pytest python/tests/test_symbolic_expr.py -q -k public_surface
cargo test -p calc-flow-python --lib "${FLAGS[@]}" error::tests -- --test-threads=1
```

Changed-file Ruff/format, cargo fmt, Python 3.9 grammar for changed modules/stub, generated schema/roundtrip, public annotations, complexity ratchet, Lizard 1.26 changed-function checks, spec references, relative document links and whitespace checks also pass. Python 3.9 grammar is not an executed interpreter matrix.

## Delivery limits

No unresolved local blocker or public-contract deviation remains. S1a continues to refuse latest replacement aggregation and unproved consumers; the Rust struct-literal migration is explicit. Full workspace/Python suites, service containers, coverage and performance measurements were not run locally. CI and the independent reviewer remain merge gates; DAL-513 stays in_progress until actual merge. No DAL-514 work or specialist dispatch was started.
