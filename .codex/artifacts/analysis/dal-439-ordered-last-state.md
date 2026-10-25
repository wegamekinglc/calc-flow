# DAL-439 / T2: compact ordered LAST_VALUE state

Baseline: `8deb6c75c20f3eb5f1b9cd82ae5ea4347dbd98dc`, the merged T1
PR #496. This stage connects the admission and semantics described in
`dal-431-ordered-last.md` to the production SQL operator.

## Execution and ownership

`datafusion_ordered_last.rs` certifies the default unbounded-pool,
single-partition FIFO, Single/Linear native grouped execution model before
history is discarded. Raw logical shape and supported types are checked in
`operator/sql/ordered_last.rs`. Empty first batches defer certification;
unsupported shapes retain their existing execution path.

Each group owns its encoded key, key scalars, nullable winner, entire ordering
tuple and non-null `is_set`. New transactions seed the native grouped
accumulator with old winners before updating raw rows. Scalar accumulators do
not process ordered input. Equal complete tuples preserve the first winner;
IGNORE NULLS still retains groups that have no eligible winner. Ordering
dependencies participate in argument evaluation, column dependencies and plan
work accounting. Complete cumulative snapshots remain the output contract,
without an additional group-order guarantee.

No input Arrow arrays remain owned by compact state. The eight-key regression
compares 100 through 10,000 quotes, asserting exactly eight groups, unchanged
group/index capacity and unchanged live incremental-pool reservations after
each batch. This is a state-size assertion, not a timing benchmark or RSS cap.
The existing cumulative input quota remains in force.

## Recovery seam required for an independently mergeable T2

Ordered state writes layout 5 with native semantics 2 and the independent
`ordered_last_grouped_v1` policy. Its identity includes resolved ordering
columns/options, null treatment, tie/grouped model, engine versions and trusted
schemas (including Timestamp unit/timezone). Timestamp state export supports
all four units. Nullable winner/order fields represent unset IGNORE NULLS
groups; flags and inconsistent winner states fail closed.

The existing bounded full/delta capture and prepare/install restore mechanisms
are reused. Tests cover continuation across captures/restores, null groups,
ties, incompatible/unknown layouts, failed restores, emit errors and cancellation
without committing winners or the input ledger. Layout-3 encoding stays intact;
nonempty layout-4 ordered checkpoints continue retained execution after restore.

T3 still owns the compact budget API, bindings and complete budget accounting.
T4 still owns comprehensive corruption, storage-failure, delta-rewrite and
budget/restore acceptance. Its starting point includes this required layout-5
seam. T5 owns application documentation and cross-layer example validation.

The previous 21 tracked changes and four new files remain byte-for-byte intact
in the original DAL-431 checkout. Stash
`32edcee9716fcd7a4d0cb595d69b86967fdb7c37` and the task-local patch/archive backup
remain available. Unaccepted budget/binding WIP is excluded from this stage.

## Validation on 2026-10-25

The following commands passed on the Rust content in this stage's commit.
Builds reused the original checkout's `target/` with
`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0`.

- `cargo test --locked -p calc-flow --lib ordered_last -- --nocapture`:
  22 passed; test execution 4.26 seconds, compilation reported separately.
- `cargo test --locked -p calc-flow --lib test_compact_storage_two_roundtrips_continue_and_recapture_every_prefix -- --nocapture`:
  one existing SUM/COUNT control passed, including layout-3 continuation.
- `cargo test --locked -p calc-flow --lib incremental::grouped_sum::tests -- --nocapture`:
  both existing dense/sparse grouped seeding regressions passed.
- `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings` and
  `cargo fmt --all --check`: passed.
- Lizard threshold 8 on the new modules and changed oracle tests: passed.
  Comparing changed tracked functions against the baseline introduced no new
  complexity above 8 and increased no existing function above that threshold.
- `git diff --check` and the unchanged Cargo.lock/project-schema checks passed.

T1's dependency audit/deny evidence remains applicable: no dependency or build
configuration changed. Full regression, workspace coverage and platform gates
remain with CI; this stage does not claim their result before it arrives.
