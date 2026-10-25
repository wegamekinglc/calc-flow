# DAL-431 ordered LAST_VALUE: T1 admission and semantics

This stage provides an executable DF55 conformance proof, without enabling
ordered compact execution or changing public APIs. The proof is registered in
`crates/calc-flow/src/datafusion.rs` and exercised by
`datafusion_ordered_last_proof_tests.rs`. Every proof helper is exercised by the
tests. T2 must connect these conditions to actual admission before discarding
any input history.

Baseline: `214794368413ec697657f2d76d65afed21c262a3`. On 2026-10-25,
`origin/main` advanced to `536ae37c0fa8e4f9eb95a289525c234fb3f88293` through
documentation-only PR #495. No Rust, dependency, or build changes affect this
proof. Keep the existing task branch and single draft PR against `main`.

## Admission contract

- One declared input alias, one plain table scan, append input and nonempty
  direct-column GROUP BY. Group types: signed/unsigned integers, Boolean,
  Utf8/LargeUtf8; composite and null keys are valid.
- Exactly one built-in ordered LAST_VALUE. Its value must be a direct integer,
  Float32 or Float64 column. Every ordering term must be a direct integer or
  Timestamp column (s/ms/us/ns, preserving timezone). Direction and null
  placement may differ per term. Default, RESPECT and IGNORE NULLS are valid.
- Project only group keys and aggregate results, including column aliases.
  A final full-result sort is allowed. Reject filters, HAVING, LIMIT, DISTINCT,
  mixed/global/window aggregates, expression operands/orderings, unsupported
  key/value/order types, joins and subqueries at this boundary.
- Require the default DataFusion configuration, standard analyzer/optimizer
  rules and session options, the unbounded DataFusion memory pool, and a
  one-partition FIFO MemTable with no advertised ordering or fetch. Inspect the
  complete physical tree: exactly one Single/Linear grouped AggregateExec,
  one source, native ordered GroupsAccumulator support, no filters, no reverse
  to FIRST_VALUE, no partial/final split, no aggregate limit, and only direct
  projections before aggregation. A result sort cannot appear below aggregation.
- Empty input defers the decision. An unsupported shape/model is a retained
  candidate before history is discarded; this does not prove that every
  pre-existing retained query succeeds. Once T2 installs compact state, errors
  must not switch to retained using incomplete history.

## Why the model is stable as rows grow

The checked-in lockfile fixes DataFusion 55.0.0 and Arrow's 59.2.0 execution
crates. The proof combines physical inspection with the following fixed-version
source conditions, rather than extrapolating one small first batch:

- `datafusion-physical-optimizer/src/combine_partial_final_agg.rs::optimize`
  combines adjacent compatible Partial/Final expressions independently of row
  statistics. One input partition requires no repartition/exchange. The raw
  shape has no filter, limit, scalar rewrite or input sort to introduce a
  different order-dependent computation.
- `datafusion-physical-plan/src/aggregates/mod.rs::create_stream` and
  `should_use_single_hash_stream` select the single grouped stream from mode,
  ordering, limit and grouping shape, not row/group cardinality.
  `aggregate_hash_table/common.rs` uses `create_group_accumulator`; its native
  versus adapter choice is based on the aggregate's supported types.
- The Single stream has no adaptive partial-aggregation skip. Spill/remerge
  would require a failed memory reservation; the admitted default unbounded
  pool does not cause that. Other pools are deliberately not certified.
- Tests cross Arrow batch boundaries (8,191/8,193 rows), 100,003 rows and
  low/high cardinality. These are semantic/plan probes, not performance or
  state-budget measurements. No finite probe claims correctness for arbitrary
  alternate optimizer settings or future engine versions.
- Negative physical-plan probes construct a Partial aggregate, reverse the
  aggregate to FIRST_VALUE, and sort the input; each must fail certification.

## Semantics and T2 integration

The reference query is the unchanged quotes SQL from example 31. Compare every
prefix by group content against cumulative DataFusion SQL, without promising
output group order. The native model evaluates physical input projections and
feeds value plus the entire ordering tuple to the real GroupsAccumulator.
It exports `[value, ordering..., is_set]`, creates a fresh grouped accumulator,
merges old state first and then updates raw new rows for the next prefix.

DF55 `functions-aggregate/src/first_last.rs::get_filtered_extreme_of_each_group`
and `update_batch` use strict lexicographical comparisons. Equal full tuples
keep the first encountered maximum, both within a batch and across updates.
Scalar `Accumulator::update_batch` is not a substitute. The oracle tests assert
the exact first price (10 rather than 20/30), including legal batch splits.

Read resolved SortOptions rather than inventing defaults: default ASC puts
nulls last, default DESC puts nulls first. Null ordering keys participate.
RESPECT/default can install a null value; IGNORE excludes only null values and
keeps all-null groups with `is_set = false`. NaN is a value, not SQL null.
Timestamp unit/timezone and all order terms must survive projection and later
identity encoding. The tests cover all numeric value widths, integral ordering,
Timestamp units with/without UTC, disorder, negative timestamps, composite and
null keys, explicit/default sort/null options, empty batches and composite ties.

T2 must reuse/promote this certificate into the production ordered-last module,
check both raw shape and analyzed resolved expressions, require proof before
installing compact state, preserve input sort-column dependencies, and defer
empty-first admission. Async planning and synchronous restore planning must
agree. This stage does not implement quotas, layout 5 or restore/emit atomicity.

The baseline retained-path test includes the original example query as a
measurement of the pre-T2 gap. When T2 enables compact execution, replace that
case with the compact/zero-retained-history assertion; keep the unsupported
expression and nondefault-model cases as retained regressions.

## Preserved earlier WIP

The stopped run's 21 modified tracked files and four new Rust files are saved
in stash `dd50616be5168d013fe5ade52afd16fb4f75efb4` and an independent patch/tar
snapshot in DAL-438's workdir. They span state updates, budgeting/bindings and
layout/export/restore work; none is accepted as T1 implementation.

The initial `cargo test -p calc-flow --lib ordered_last -- --nocapture` on that
WIP failed compilation with six diagnostics: missing `owned_reservation` in a
checkpoint constructor, an iterator borrowing a temporary expressions vector,
two missing `sql_compact_state_budget` initializers, incompatible AggregateUDF
comparison, and two new fields missing from a PartialAccumulator fixture.
The dedicated proof is tested on the clean baseline; later stages must fix the
relevant WIP before claiming their own validation.

Keep these existing WIP areas separate when resuming:

- T2: `operator/sql/{incremental,ordered_last,grouped_sum,native_expression}.rs`
  and `ordered_last_tests.rs`; the existing physical proof also needs the
  stricter pool/shape checks exercised here.
- T3: `datafusion_compact_budget.rs`, DataFusion reservation changes,
  `pipeline/stream.rs`, `runtime/streaming/runner.rs`, PyO3 `continuous.rs`,
  Python `runtime.py`, and compact state accounting.
- T4: compact identity/control/capture/storage/restore, metadata, IPC and
  Timestamp state export changes. Do not include these in the T2 commit merely
  to make all preserved WIP compile.

## Validation on 2026-10-25

All commands below ran with earlier WIP shelved, against the baseline plus the
T1 test module and its registration. The following commit contains that exact
tested Rust content. WIP is restored afterward and is not covered by these
passing results.

- `cargo test -p calc-flow --lib ordered_last_proof_tests -- --nocapture`:
  12 passed, 0 failed, 0 ignored; 2,418 unrelated tests filtered out. Final test
  execution took 5.82 seconds (builds are separate).
- `cargo clippy -p calc-flow --lib --tests -- -D warnings`: passed.
- `cargo fmt --all --check`: passed.
- `git diff --check` and
  `git diff --exit-code -- schemas/project-v3.schema.json`: passed.

The baseline original example SQL and schema return price 10 for complete ties,
but still capture layout 4 with `input-retained` and retained rows growing from
2 to 4 after two batches. The expression-ordering and nondefault-model cases
also remain retained. The reproduced example gap is absence of compact
execution; this evidence does not establish an existing wrong-result defect.

The first larger oracle run found two fixture failures because inferred Arrow
nullability varied between cuts. Explicit nullable schemas fixed the fixtures;
the final run above includes empty batches and null-containing legal cuts.
No full regression, coverage or performance benchmark was run locally.
