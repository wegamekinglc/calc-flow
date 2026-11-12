# DAL-517 finite inner join implementation evidence

Base: `3c4b919eb389544d8b8080d3571e49479da68b2c` (merged PR #600).
Branch: `feature/dal-517-finite-inner-join`. The assigned checkout was reused
and renamed; the frozen `../api-notes/finite-inner-join.md` is included unchanged.
No second checkout, team definition, dependency, project schema, checkpoint
codec, state version, or layout was introduced.

## Design and changed files

The frozen API note is the design checkpoint. `operator/join/finite.rs` adapts
the existing native operator with run-local rows. It shares native pure
admission masks, versioned key encoding and logical charge, DataFusion equality
query, i128 closed bounds and ordering, match-limit enforcement, and Arrow gather.
Both complete ingress envelopes validate before work. Admission is left then
right, with no stream context, watermark, EOF injection, checkpoint or persistent
owner. A compiled batch plan passes its run-scoped DataFusion runtime to the
adapter. Direct BatchOperator calls own a temporary runtime. Errors/cancellation
return no partial result; every invocation drops its scratch state. Empty output
contains an exact-schema RecordBatch and finite metadata uses node/sequence zero.

- `operator/join.rs`, `admitted.rs`, `admission_masks.rs`, new `finite.rs`:
  finite lifecycle and existing row/admission interfaces, with stream handlers
  and checkpoint encoding retained.
- `pipeline/batch.rs`, `config.rs`: both direct and project batch admission,
  null snapshots, non-null restore rejection, no stateful reset requirement.
- `dataframe.rs`, `lib.rs`, `program/lower.rs`, new `program/lower/join.rs`:
  Rust `inner_join`, `JoinOutputOrdering`, full-spec identity, native schema and
  dual-mode lowering, nested ordering and join-local SQL passthrough proof.
- `sql.rs`, new `sql/inner_join.rs`: shared native AST recognition of exactly
  the explicit three-string-argument CF_INNER_JOIN dialect, strict configuration,
  explicit bounds/limits and useful subfield errors. Ordinary SQL stays separate.
- Binding `config.rs`, `_native.pyi`: private data-only SQL descriptor. The
  dialect exports actual native join nodes rather than executing SQL text joins.
- Python symbolic `analyzer.py`, `inner_join_sql.py`, `program.py`, lower
  `program.py`/`sql.py`, `compute.py`: both lifecycles, mixed DAG memoization,
  shared owners, pure identity passthrough and a macro-specific cache strategy.
  Batch joins use the general materialized graph; existing stream @1/@2 paths
  retain their original identities, names, topology and recovery contracts.
- `query_analysis.rs`, Python `query_analysis.py`, `capabilities.py`: revision 6,
  conditional unordered accepted-history EOF proof, append semantics and latest
  ingress rejection; explicit update mismatch takes priority in the Python
  latest gate. Older reports still parse but cannot satisfy the r6 latest gate.
  Current-producer revision/capability assertions were updated, including the
  PyO3 error test. Existing stale-report fixtures remain and now include r5.
- Focused Rust/Python tests, `docs/relational-frontends.md`, both independently
  runnable `finite_inner_join` examples and the existing method docstring.

The identity brief references three absent `.codex/references` files. Actual
AGENTS.md, repository code-style skill, cf-implementer methodology, symbolic
spec/approved critique, frozen note, relevant source and nearby tests were read.
There is no dal-public component. Existing core examples 12/13 were retained.

## Test-first observations

All commands ran in the assigned checkout. `cargo` commands below inherit
`CARGO_BUILD_JOBS=2` and the existing shared Cargo artifact cache named
`target/pr-repair-20261112/cargo-debug` in the development checkout; this is an
artifact cache, not a different source tree. No other checkout supplied candidate
GREEN product code. Python RED used the predecessor wheel solely to establish
missing behavior; all candidate GREEN checks use the native built here and the
58 Python/stub files copied from this checkout.

- `cargo test --locked -p calc-flow --test finite_inner_join`: RED, the join was
  stream-only. The same complete-input oracle then passed. Edge checks now cover
  duplicate pairs, inclusive/exclusive bounds, negative and i64 endpoint times,
  native NULL time/key drop, typed empty, full Batch match budget across multiple
  RecordBatches, side budget even opposite empty, schema/cancellation failure,
  same-plan retry, and snapshot/restore semantics.
- `cargo test --locked -p calc-flow --test finite_inner_frontend`: RED, no
  `inner_join` method. An initial incorrect compile_stream argument count was
  corrected before the meaningful missing-symbol RED run. Subsequent RED checks
  caught accepted partial output ordering and incomplete input sequence identity;
  the final tests pass canonical nesting and reject incomplete/millisecond keys.
- `python -m pytest python/tests/test_finite_inner_join.py -q`: baseline RED,
  batch DF rejected by the engine gate and SQL function absent. The independent
  r6 proof test was RED at revision 5. Candidate tests then exposed and fixed
  missing data-source declarations in single-join batch compilation, time
  arithmetic retaining proof, SQL schema diagnostics using stream_join rather
  than sql.inner_join, unsupported key types claiming analysis support, and
  latest admission choosing an incidental schema error before the ingress
  update_interpretation_mismatch.
- Adjacent fixed-window tests exposed ordinary invalid SQL probe paths changing
  from sql.query. Six cases were RED, then GREEN after the native probe retained
  that original contract. No old SQL test expectation was weakened.

Fixture mistakes (ArrowError conversion, watermarks requiring a per-input map,
an empty materialized stream list, and an AnalysisResult attribute spelling)
were corrected without changing product behavior. An EOF-only async generator
tests the existing empty-source contract. The old stream-only test assertion
was replaced with batch admission; its opposite-engine guard and missing stream
ordering assertions remain intact.

## Initial verification and provenance at 5d01defd

The interpreter is the existing managed CPython 3.13 development environment.
`PYTHONPATH=target/finite-inner-join/site` loads the native and Python package
only from this checkout. All logs, wheel, private site, and a full source hash
manifest are under `target/finite-inner-join/`. These runtime files are local
working evidence; this committed report and the PR are the review artifacts.

Native build:

```sh
python -m maturin build --locked --profile dev --interpreter <managed-python> \
  --out target/finite-inner-join/wheels
```

Passed. Native SHA-256:
`267ebcec042b0d681d0c24af55bceceb4c488aaae53b9325aac4d23ac55ff293`.
The private site overlays the exact current Python sources onto that wheel.
Every one of its 58 Python/stub files matches this checkout byte-for-byte;
`provenance.json` records the interpreter, loaded paths, native hash, and hashes
of Rust product sources, Cargo manifests/lockfile and Python sources. Final
tests use this matching combination; no source `_native*.so` is generated.

```sh
cargo test --locked -p calc-flow --test finite_inner_join \
  --test finite_inner_frontend --test stream_join_spec --test stream_join_state \
  --test stream_join_validation --test query_history_portability
cargo test --locked -p calc-flow --test finite_inner_frontend \
  --test finite_inner_join --test finite_union_frontend --test latest_relation
```

The initial six-target run passed 47 tests. The final four-target run passes 21,
including three public-frontend tests and three finite-native tests. Counts
overlap: there are **64 distinct passing Rust tests**, comprising the final 21
and 43 unchanged stream/history tests. The public Rust test executes both DF
and SQL in batch and in a real StreamingRunner job against the literal six-pair
oracle; both outputs share the same two source owners. Another test verifies
three-source nesting, SQL passthrough and one first-join owner.

```sh
python -m pytest python/tests/test_finite_inner_join.py \
  python/tests/test_symbolic_stream_join.py \
  python/tests/test_symbolic_stream_join_acceptance.py \
  python/tests/test_symbolic_relational_dag.py \
  python/tests/test_symbolic_relational_dag_acceptance.py \
  python/tests/test_capabilities.py python/tests/test_update_semantics.py \
  python/tests/test_query_history_portability.py \
  python/tests/test_finite_event_window.py python/tests/test_finite_union.py \
  python/tests/test_relational_frontend.py -q -W error
```

**303 passed** in `python-final.log`. Includes four-surface shared schemas and
native export, literal six pairs, same finite plan/cache without reset, nested
SQL/DF joins, shared/independent branches and project reload, multi-batch stream
terminal reduction with disabled watermarks, a fresh EOF-only job, layered NULL
time admission, SQL limits/shape diagnostics, per-ingress latest rejection before
iteration, old @1/@2 identity and source collision fixtures, real managed restart
and next-input equivalence, and old 7/latest 6 regression. Stream comparisons
collect every append pair; they make no cross-ingress total-order claim.

```sh
cargo clippy --locked -p calc-flow --lib --test finite_inner_join \
  --test finite_inner_frontend --example finite_inner_join \
  --example 12_symbolic_stream_join --example 13_symbolic_relational_dag \
  -- -D warnings
cargo run --locked -p calc-flow --example finite_inner_join
python examples/finite_inner_join.py
cargo fmt --all --check
python scripts/verify_complexity_gates.py
git diff --check
```

All pass, along with changed-file Ruff check/format (18 Python/stub files).
Clippy initially caught ordinary import/default/literal/doc issues; they were
fixed without lint allowances or baseline changes. Both examples execute public
DF/SQL finite oracles and compile stream declarations. Existing core examples
12/13 compile independently under Clippy. No team files, historical v1 fixtures,
lockfile or generated project schema changed.

## Handoff limits

No material API or scope deviation was needed. Examples retain explicit required
configuration and are longer than the note illustrative target after formatters.
Finite matching uses the shared serial equality/gather path, not the stream
worker scheduler; no performance measurements were requested or run.
Public nullable time, broader SQL ON/CTE/subqueries, join-path union/window/
aggregate combinations, outer/semi/anti/cross/replacement joins, global ordering,
and batch mid-query recovery remain outside the frozen supported domain.

Full workspace suites, coverage, Windows and hosted CI remain CI responsibilities.
The native/Python paths and hashes above identify this implementation environment;
test and reviewer acceptance are still independent. One main PR is authorized;
its exact commit/URL and one nonblocking CI snapshot are returned in the issue
comment. Pending does not authorize merge. No delegation or merge is performed;
DAL-517 remains in_progress until the coordinator completes acceptance/merge.


## Codacy and cfg(test) repair after 5d01defd

This increment stays on the same branch, checkout and PR #601. The earlier
64 distinct Rust / 303 Python results and native hash above remain evidence
for the initial implementation; they are not relabelled as a fresh full run.
This repair changes private decomposition and explicit example/test data, with
no API, spec, schema, state version, identity or checkpoint-format change.

The triggering check run `131029310005` reported 20 annotations: 16 complexity
findings (12 Rust, 4 Python), one example SQL construction warning, one example
assert warning, and two test execute-input warnings. The annotation list was
read once through the GitHub API and saved in `codacy-annotations.json`.
Lizard reproduced all 16 complexity failures before refactoring. Rust `?`
paths contribute to CCN. Local versions 1.23.0 and pinned 1.26.0 count some Rust
constructs differently; both now verify every reported function and every
extracted helper at CCN <= 8. Unchanged pre-existing analyzer warnings remain
unchanged. No threshold, baseline, ignore, lint allowance or tool configuration
was changed.

- `operator/join/finite.rs`: separate full-envelope validation/admission,
  equality probe, pair ordering, output collection and record/block admission.
  One run-owned FiniteAdmission retains rows, row ID and total logical bytes
  across all RecordBatches. Match limits remain per complete Batch; NULL/error
  classification, counter advancement, cancellation checks and gather order stay
  in their original order. Moving owned admitted payloads avoids an unnecessary
  clone and adds no persistent state.
- `program/lower/join.rs`: separate side/time/key validation, SQL identity
  mapping, and project/SQL time-source proofs; preserve diagnostic paths and
  nested ordering checks.
- `sql/inner_join.rs`: separate AST FROM/shape checks, argument decoding,
  alias validation, strict JSON configuration and output-ordering validation.
  Alias errors still precede third-argument/configuration errors.
- Rust example and the two focused Rust integration files: extract declaration,
  input, execution and assertion helpers. Every original pair/count/schema,
  failure/retry and restore assertion remains. The original stricter budgets
  on nested/invalid fixtures are retained.
- Python analyzer and inner_join_sql: extract path/time and passthrough mapping
  helpers while preserving issue order and existing declaration bytes.
- Python example/tests: replace formatted SQL with fixed complete SQL/JSON
  literals, and use explicit a.input/b.input Batch mappings with physical-name
  assertions. All SQL configurations, negative cases, multiset and cache checks
  remain. The example uses a RuntimeError oracle, which also runs under python -O.

One bounded failure snapshot of the original head additionally found Linux
`Rust core parity` job `131028901868` and Windows `Clippy (Windows cfg parity)`
job `131028884352` failing the same items_after_test_module lint. Their completed
job logs were read directly, without waiting for the remaining jobs. Local RED:

```sh
cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
```

It failed at `operator/join/admitted.rs` with the same lint. The Vec adapter and
its import were moved before the cfg(test) fixtures; its implementation body
is byte-for-byte unchanged. The same command is GREEN. This expands compilation
only to reproduce the observed CI error; it does not run the full Rust suite or
claim Windows execution. The other targeted Clippy command from the initial
verification, including the new example, also passes after the repair.

Final affected-scope verification, with the original shared artifact cache and
managed interpreter described above:

```sh
cargo test --locked -p calc-flow --test finite_inner_join --test finite_inner_frontend
python -m pytest python/tests/test_finite_inner_join.py \
  python/tests/test_symbolic_stream_join.py \
  python/tests/test_symbolic_stream_join_acceptance.py \
  python/tests/test_symbolic_relational_dag.py \
  python/tests/test_symbolic_relational_dag_acceptance.py -q -W error
python -m pytest python/tests/test_finite_event_window.py \
  -k ordinary_sql_query_error -q -W error
cargo run --locked -p calc-flow --example finite_inner_join
python -O examples/finite_inner_join.py
```

Results: **6 Rust tests**, **75 Python tests**, plus **6 ordinary-SQL diagnostic
cases** pass. The two examples run successfully. This includes actual stream
jobs, original @1/@2 identity fixtures, nested owner sharing and real managed
restart/next-input equivalence. No assertions were weakened and no full suite,
coverage, performance or Windows run was added.

Changed-file Rustfmt/Ruff checks, the unchanged complexity ratchet, and whitespace
checks pass. Pinned Lizard 1.26.0 and installed 1.23.0 both verify the repair;
`complexity-green.json` records function comparisons against 5d01defd. The
existing Bandit 1.9.4 reports no example findings. This additional local scan
passes with **0 findings, 0 errors, 200 applicable rules on 2 files**:

```sh
semgrep scan --config=p/python --config=p/security-audit --metrics=off \
  --disable-version-check --error python/tests/test_finite_inner_join.py \
  examples/finite_inner_join.py
```

Semgrep is 1.180.0; this local rule set is not represented as a hosted Codacy
verdict. The new head still requires the normal hosted gate.

Native was rebuilt after the final Rust declaration-order change using the
recorded maturin command, then the exact current Python tree was overlaid.
All 58 loaded Python/stub files match the checkout byte-for-byte. The new native
SHA-256 is `2a6d537354830e3d1dc993ac16f73a7e5359cf10bd1352b4f82c9659b186e4a7`;
the original hash `267ebcec042b0d681d0c24af55bceceb4c488aaae53b9325aac4d23ac55ff293`
is preserved above and in the original source manifest. The new independent
`provenance-codacy.json` records both hashes, loaded paths and the current Rust,
Cargo and Python source hashes. All final Python checks use this new native.

The final repair commit and one nonblocking new-head CI snapshot are delivered
in the triggering issue thread. Hosted pending checks are not green evidence;
there is no poll, merge or delegation. DAL-517 remains in_progress for the
coordinator's independent tester then reviewer handoff.
