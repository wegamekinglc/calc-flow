# Issue 406: Join output projection review

**Branch:** `feature/issue406-join-output-projection`
**Base:** `90c323e9`, ordinary product `995b1b9a`
**Reviewed state:** final implementation working diff; 22 changed/new Rust files
**Verified final source manifest SHA-256:**
`da12a185e77804d59d896d4f0d6ad73432aedf8aa17d326416e4ac7cf799af86`

The reviewer owns only this report in the owner's existing worktree. No
product/test edits, builds, tests, measurements, CI queries or remote mutations
were performed. This is a branch review; no PR/check-run approval is claimed.

## Summary

The compiler derives Join output dependencies from all eligible expression
consumers, unions them in original logical-column order and installs a private
physical mapping. Serial gather, owned column workers and flat/nested output
preflight use the selected columns while retained input/state schemas remain
full. ASOF keeps its existing exact-projection qualification.

The physical mapping has no identified ordering or funding blocker. The two
SQL qualification findings are resolved with passing regressions, and emission
extraction preserves the original sequence, error and funding boundaries.
Focused execution/recovery/lifecycle, lint and source-manifest evidence pass.
The strengthened cross-node UDF regression independently proves that the graph
guard protects known builtins from selected native overrides.

## Build and test results

- Rust: **Passed** focused affected verification; full CI/coverage uncertified.
- Python application and bindings: **Not touched**; no suite run by reviewer.
- `projection-red.log` records the four existing projection registrations:
  **two expected failures and two passes** after a **6m 57s** compile. Join exact
  projection expects three columns but gets eight; arithmetic dependency
  projection expects two but gets eight. The two existing ASOF controls pass.
- `projection-green.log` records **compilation failure**, before execution,
  on two `unused_qualifications` errors in materialization signatures. The owner
  corrected those paths. `projection-green-fixed.log` then records the two
  initial exact/arithmetic schema controls passing in **0.01 seconds**, after
  a **4m 42s** compile. This precedes the later SQL qualification corrections.
- `materialization-green.log` records the existing fallback registration and
  its new serial-column/budget helper passing in **0.02 seconds**.
  `owned-green.log` records the owned output registration and its new helper
  passing in **0.07 seconds**. These initial controls are passing evidence,
  rather than the final expanded projected-lifecycle run.
- The qualification regressions start with a test compilation failure:
  `qualification-red.log` preserves an incorrectly boxed Expression variant
  and attempted equality on Port values. After test-only corrections,
  `qualification-red-fixed.log` records **two expected behavior failures**
  in **0.02 seconds**, after a **3m 28s** compile: `now()` incorrectly qualifies
  in dependency analysis, and a cross-node-selected UDF narrows eight Join
  columns to one. Product qualification is corrected only afterward.
- Final affected verification passes **19 unique registrations**:
  `qualification-and-recovery-green.log` records **four** pipeline tests in
  **0.13 seconds** after a **3m 10s** compile; `materialization-final.log`
  records **five** tests in **0.06 seconds**; `owned-lifecycle-final.log`
  records **seven** tests in **0.22 seconds**; and
  `dictionary-and-gather-final.log` records **three** tests in **0.01 seconds**.
  The reviewer independently reconciles the nineteen distinct test names.
- `clippy-final.log` preserves the first scoped Clippy **failure** on
  `manual_let_else` and `filter_map_bool_then`. Both are corrected without
  allowances. The split filter/map preserves lazy right-side index subtraction.
  Additional assertions cover mixed full-input consumers, unconsumed Join,
  full-reference width and a predicate removing one of two restored matches.
  Only the **three changed registrations** are rerun: all pass in
  **0.14 seconds** after a **1m 15s** compile
  (`final-corrections-green.log`). They are a subset of the nineteen tests,
  not three additional registrations; unchanged passing checks are not repeated.
- Scoped core lib/tests all-feature Clippy passes with `-D warnings`
  in **1m 31s** (`clippy-corrected.log`). Owner formatting, generated-schema
  no-drift and whitespace checks pass; reviewer schema/diff checks pass.
  Existing registration count remains **2,033**.
- `complexity-before-builtin-override.json` records **89 changed/new functions**, maximum CCN
  **eight**, with no parser omissions. The reviewer independently verifies the
  complete 22-file changed/new Rust inventory, every file SHA-256 and the
  aggregate digest
  `971739c96648f428ae1ea8a11f61ec316229b5a473cfa767ce27a405b4362b43`.
  The digest encodes sorted relative paths,
  one ASCII space, lowercase file SHA-256 and LF per entry. The earlier
  88-function snapshot is retained as
  `complexity-before-lint-correction.json`.
- Before final handoff, the owner strengthens the cross-node selected UDF
  regression from unknown name `shared_value` to known Immutable builtin
  `abs`. The former test can also pass because unknown functions decline;
  overriding `abs` makes the graph-selected guard necessary independently.
  The test first proves ordinary `abs(left__seq)` narrows to `left__seq`, then
  proves an `abs` override selected by another node retains all eight fields.
  Product code is unchanged. The single affected registration passes in
  **0.03 seconds** after a **1m 37s** compile
  (`builtin-override-regression.log`). Final scoped Clippy passes in
  **40.29 seconds** (`clippy-builtin-override-final.log`). The other eighteen
  unique passing registrations are not repeated.
- Refreshed `complexity-final.json` retains **22 Rust files, 89 functions,
  maximum CCN eight and no parser omissions**. Relative to the previously
  verified seal, only the qualification test file changes; the reviewer
  independently verifies that file's current hash, the twenty-one unchanged
  manifest entries and the new aggregate digest in this report's header.
- Required CI, cross-platform gates and combined 90% workspace coverage are
  **uncertified**. No CI snapshot was queried under the assigned scope.

Owner commands ran from this isolated worktree with
`CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/issue406-join-hotpaths-build`,
`CARGO_BUILD_JOBS=2` and `RUST_TEST_THREADS=1`:

```bash
cargo test -p calc-flow --lib pipeline::stream::projection::tests:: -- --test-threads=1 --nocapture
cargo test -p calc-flow --lib operator::join::materialization::tests:: -- --test-threads=1 --nocapture
cargo test -p calc-flow --lib operator::join::tests::column_gather_tests:: -- --test-threads=1 --nocapture
cargo test -p calc-flow --lib -- test_nested_dictionary_output_discards_unused_values_under_same_budget test_multi_parent_nested_dictionary_fallback_discards_unused_values test_flat_nullable_gather_matches_old_concat_for_each_supported_shape --test-threads=1 --nocapture
cargo test -p calc-flow --lib -- asof_materializes_only_columns_required_by_every_projection_consumer physical_asof_projection_preserves_aliases_nulls_and_projected_state_restore test_charge_fast_path_falls_back_when_the_bound_exceeds_the_budget --test-threads=1 --nocapture
cargo test -p calc-flow --lib asof_materializes_only_columns_required_by_every_projection_consumer -- --test-threads=1 --nocapture
cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings
cargo fmt --all --check
git diff --exit-code -- schemas/project-v3.schema.json
git diff --check
```

The fourth and fifth test commands use three libtest OR substring filters;
none of these commands uses `--exact`. Counts above are taken from execution
logs, rather than inferred from filter names.

## Blocking issues

None remain. The source corrections below, affected verification and
independently discriminating builtin-shadowing regression resolve the initial
review findings.

1. **Resolved in source — graph-selected UDFs:** initial qualification checks
   the consumer's own UDF list and already installed stream resources, but
   output projection runs before
   `CompiledStreamOperator::convert` installs the table resources. Those
   resources register the union of UDF references selected anywhere in the
   graph in every expression runtime. A consumer with an empty local UDF list
   can therefore call a scalar UDF selected by another node and pass the new
   arbitrary-function dependency analysis. Such a UDF may observe input batch
   boundaries, which physical byte-budget pruning can change. The correction
   passes validated `CompiledGraph.table` resources into
   `prepare_stream_operators` and the physical pass; any selected graph UDF
   conservatively disables Join projection. ASOF keeps its existing branch.
   The added cross-node UDF qualification control produces the expected RED
   before this fix; corrected GREEN passes, including the final augmented
   exact-projection registration and known-builtin override control.
2. **Resolved in source — statement-sensitive functions:** absence of `OVER`,
   filters and special arguments proves syntax shape, not row-local semantics.
   The initial dependency helper accepts `now()`/
   `current_timestamp`, current-date/time and volatile functions such as
   `random()`. Pinned DataFusion 54's `NowFunc` chooses its value during each
   statement's planning; changing a wide Join's several output chunks into one
   narrowed chunk can change the statement grouping visible to rows. Existing
   `compile_stream` determinism validation already rejects volatile and
   wall-clock builtin calls, so the `now()` RED establishes a dependency-helper
   proof gap, rather than a demonstrated reachable wall-clock runtime
   regression. The correction independently resolves a single-part builtin
   name in the shared default FunctionRegistry and requires
   `Volatility::Immutable`. Unknown, stable and volatile functions decline,
   and non-function expressions require an explicit row-local AST shape.
   Added qualification controls pass and require no timing-dependent assertions.

The agreed producer/consumer, recovery and projected lifecycle coverage also
passes. Initial RED, compilation and lint failures remain recorded above.

## Style issues and nonblocking caveats

No unsafe code, dependency change, public control injection or caller mutation
is introduced. Mutation remains at the compiler's owned builder and owned
Arrow gather boundaries.

The initial nested fallback rebuilt projection indices/schema for each distinct
matched row. This advisory is addressed: `estimated_rows` now prepares one
`ProjectedRow` per side, reusing immutable indices and schema across its rows.
Per-row Arrow views remain necessary for the existing exact measurement path.
No timing or allocation speedup is inferred from this source refactor.

## Correctness, budgets and ownership

- Logical graph validation/fingerprint construction precedes physical
  projection. Every consumer must qualify; a wildcard, unresolved reference,
  non-expression consumer or unconsumed external producer keeps full output.
  A `BTreeSet` union retains original producer-column order.
- The stored parsed query is inspected at compilation, not reparsed per batch.
  Canonical AST equality restricts the outer query to one `SELECT` from `input`
  with an optional predicate. Nested queries, aggregates, window syntax and
  wildcard arguments decline. The corrected function registry/volatility proof
  and explicit expression-shape allowlist close the initial qualification gaps.
- The mapping selects source side/index separately from the output ordinal.
  Serial gather iterates only selected output fields. The owned worker uses
  the mapped source index and the ordinal's canonical output type. Both retain
  matched-pair order and the existing null/dictionary compaction path.
- Flat preflight visits selected columns and tracks null bitmaps by output
  ordinal; nested preflight measures selected row views and recursively
  validates materialized selected ranges before emission, preparing each
  side's projected schema/indices once. Existing full-row
  charges remain a conservative single-chunk fast path for the compiler's
  unique dependency union; declining that bound reaches selected-field
  preflight. Duplicate/reordered private-map tests also exercise exact paths.
- Empty output retains the exact physical schema. Nonempty zero-column output
  uses `RecordBatchOptions::with_row_count`, preserving matched row count while
  still respecting the edge's row limit.
- Owned capture keeps the existing per-side row coordinates and source owners.
  Projection metadata is an `Arc<[usize]>` clone, with no new heap copy in the
  worker handoff. `funding::input_bytes` includes `size_of::<Columns>()`, so the
  added Arc field is covered. Output credit remains conservative full-row
  charges; results retain credit through emission and cleanup.
- Input ports, retained rows/charges, matching, checkpoint encoders/decoders
  and source row IDs are not projected. Existing cancellation, declined-capture
  serial fallback and cleanup mechanisms remain in place. Final tests must
  establish these claims for an actually projected operator, rather than only
  for unchanged unprojected controls.
- `materialization::emission::{emit_ranges, gather, message}` extracts the
  original delivery loop. `emit_prepared` still obtains every validated range
  and checks the whole sequence range before delivery. Each range checks
  cancellation before gather; the returned funding remains live through
  message validation and awaited emit. Only successful emission drops funding
  and increments sequence. Refused gather retains serial fallback, errors keep
  their original operator/message, and failure unwinds owned credit normally.

## Test coverage

New source controls include union/exact/arithmetic schema assertions, a column-
kernel counter, both incoming sides, duplicate/reordered private mappings,
nullable/string/list values, selected byte budgets, empty output, zero-column
row counts, independent buffers and owned credit refund. These reuse existing
registrations. Initial helpers and final expanded controls pass as recorded
above; no extra test registrations are added.

The new execution helper checks real producer-to-expression arithmetic,
predicates and constants, full/projected V1 and V2 snapshot metadata/segments,
cross-restore in both directions and continued output/status equality. It also
checks the logical fingerprint before physical installation, serialized Join
configuration, retained input schemas and caller data. The final full reference
explicitly retains eight columns; two restored matches enter the consumer and
its actual predicate removes one, while the other consumer and constant-only
case retain both rows.

Existing lifecycle registrations now run full and projected variants for
capture-budget refusal, attempt/generation refusal, cancellation/abandonment
and blocked/failed emission. The output control adds both incoming sides,
sliced multiple-parent inputs, reordering/duplication/nulls and owned buffer
independence; nested dictionary controls include projection under the same
budget. These expanded scenarios pass. Evidence retains expected RED,
compilation and style failures, with exact final commands above; unchanged
passing checks are not repeated.

## Documentation consistency

The specification correctly limits this increment to phase 2.1 physical Join
output materialization. Phase 2.2 retained-state projection and the complete
eight-target objective remain required work. ASOF's previous sole performance
attempt is **STOPPED before observations**, with no retry; neither that record
nor the earlier source approval establishes performance for this change.

No public API, project-v3 format or verification command changes require
normative API documentation changes. Final analysis should record the SQL
qualification correction, actual focused results and remaining acceptance
scope. No performance acquisition or broad suite was launched by this reviewer.

## Verdict

**Approve — local source implementation and focused verification.** The SQL
qualification findings and nested projection advisory are resolved; recorded
execution/recovery/lifecycle, lint, format, builtin-shadowing regression and
final source-manifest evidence pass. No correctness, ownership, style or
documentation blocker remains for this diff. Phase 2.2 retained-state pruning
remains required subsequent work.
Performance, phase-wide resource acceptance, required CI/coverage and complete
issue-wide acceptance remain uncertified. No push, PR update or merge is
authorized or performed by this review.
