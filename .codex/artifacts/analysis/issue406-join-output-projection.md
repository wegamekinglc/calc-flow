# Issue 406: Join output projection evidence

This output-only increment starts at `90c323e9`. The complete issue remains
active, including all eight original targets, retained-state projection and
remaining acceleration phases. This increment includes no performance
comparison or release build. The preceding ASOF comparison remains a
stopped attempt with zero observations; it is not retried here.

## Recorded behavior checks

All command logs below are under the repository root
`target/issue406-join-output-projection-20261014/`. Rust builds reuse
`target/issue406-join-hotpaths-build/`, two Cargo jobs and serial test execution.
Test registrations are reused to preserve the existing 2,033 default libtests.

- `projection-red.log`: two expected schema failures, full eight columns
  versus required three/two; two existing ASOF checks pass. Compilation took
  6m57s and execution 0.05s.
- `projection-green.log`: compilation failed on two redundant qualified
  `RowPayload` paths. This is a compiler diagnostic, not a behavior result.
- `projection-green-fixed.log`: the two changed schema cases pass after the
  initial implementation and redundant `RowPayload` path corrections;
  compilation 4m42s, tests
  0.01s. Unchanged ASOF recovery/unconsumed cases were excluded from this run.
- `materialization-green.log`: one existing registration executes the new
  selected serial-column and budget controls, then its original fallback
  check; passes in 0.02s. Both incoming sides, duplicate/reordered mapping,
  selected nullable flat/nested budgets, zero-column row counts and independent
  recursive buffers are exercised.
- `owned-green.log`: one existing registration executes the new three-column
  owned-worker control and its existing nullable/chunk controls; passes in
  0.07s. Worker kernels equal selected fields; output remains funded and final
  cleanup refunds the pool.

## Review corrections in progress

The initial specialist review withheld approval for two eligibility gaps:
node-local UDF checks run before whole-graph selected resources are installed;
and syntactically scalar functions need an immutable builtin identity proof.
Regression cases were added before either product correction. The first
qualification RED build (`qualification-red.log`) failed to compile two new
test-helper types: a boxed Expression enum variant and equality on Port values.
Those test helpers were corrected without changing product eligibility.
`qualification-red-fixed.log` then compiles in 3m28s and records two expected
failures in 0.02s: `now()` incorrectly qualifies, and a graph-selected UDF
incorrectly narrows the Join from eight columns to one.
The `now()` case proves the helper's eligibility gap; the existing stream
compiler already rejects wall-clock/volatile queries, so this is not evidence
of a reachable wall-clock execution regression. The graph-selected UDF case
exercises actual stream compilation.

The correction carries the validated graph's selected resources into the
physical pass and conservatively disables Join pruning whenever any UDF is
selected. ASOF retains its original exact-column conditions. Scalar functions
need a known Immutable implementation in the shared default function registry;
other expressions must belong to the explicit row-local AST set. Unknown,
Stable and Volatile implementations decline. This analysis happens once at
compile time.

Nested preflight now prepares each side's indices/schema once, instead of
rebuilding projection metadata per distinct matched row. The existing chunk
delivery code is split into `emit_ranges`, `gather` and `message`, preserving
preflight before emission, credit through awaited emit, cancellation checks and
sequence advancement only after acceptance. This keeps changed functions at
CCN 8 or below; the final scan covers 89 changed/new functions.

The corrected implementation passes **19 focused registrations**:

- `qualification-and-recovery-green.log`: four pipeline tests, compilation
  3m10s and execution 0.13s. The two qualification regressions pass. The existing
  async registration also executes new Join controls for both V1 and V2:
  projected/full snapshots have identical metadata and segments, both can cold
  restore the other, state status/charges agree, and arithmetic/immutable
  scalar/filter/alias and constant-only consumers match full-input execution.
  Logical fingerprints, full input schemas and serialized Join configuration
  remain exact. Constant-only compilation produces a zero-column intermediate
  with a preserved matched row count.
- `materialization-final.log`: five tests, 0.06s, after nested metadata and
  delivery refactoring. Selected serial kernels/budgets, sparse flat/nested
  costs, conservative charge fallback and scratch-allocation checks pass.
- `owned-lifecycle-final.log`: seven tests, 0.22s. Projected and full controls
  cover attempt/generation admission refusal without emission or commit;
  capture-credit refusal with refunded serial fallback; cancellation and future
  abandonment with source owners alive until actual drain; output credit across
  blocked/failed emit; and nullable sliced multiple-parent output with duplicate
  and reordered fields in both incoming directions. The new owned helper
  proves three selected kernels, independent buffers and eventual pool refund.
  Existing concurrent-worker/Tokio responsiveness control also passes.
- `dictionary-and-gather-final.log`: three tests, 0.01s. Projected nested
  dictionary output discards unused values under the same byte budget. Full
  multiple-parent nested dictionary fallback and supported nullable flat shapes
  continue to match the former concat oracle.

`clippy-final.log` records the first scoped lib/tests all-features Clippy
failure: `manual_let_else` and `filter_map_bool_then`. Both were corrected
without lint allowances. The split filter/map keeps right-side index
subtraction lazy. Additional assertions explicitly cover unconsumed Join,
mixed wildcard/aggregate/unresolved consumers, full-reference width, and a
predicate actually removing one of two matched rows after cold restore.
Only the three affected registrations were rerun in
`final-corrections-green.log`: all pass, compilation 1m15s, execution 0.14s.
The other unchanged passing registrations were not repeated.

`clippy-corrected.log` passes
`cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings` in
1m31s. `complexity-before-builtin-override.json` records 89 changed/new functions with maximum
CCN 8, hashes every changed Rust file and verifies no declared function is
omitted by the parser. Its sorted file/hash manifest digest is
`971739c96648f428ae1ea8a11f61ec316229b5a473cfa767ce27a405b4362b43`.
Before final approval, the cross-node UDF regression was strengthened to
override `abs`, a known Immutable builtin. A control with no selected UDF
narrows to `left__seq`; selecting the override only on another node must keep
all eight Join fields. This distinguishes the graph-level guard from the
unknown-function guard. Product code is unchanged. Only its single changed
registration is rerun in `builtin-override-regression.log`: compilation 1m37s,
execution 0.03s, PASS. Clippy for that final test source also passes in
`clippy-builtin-override-final.log`, 40.29s, using the same scoped command.

The final sorted file/hash manifest digest is now
`da12a185e77804d59d896d4f0d6ad73432aedf8aa17d326416e4ac7cf799af86`;
89 changed/new functions still have maximum CCN 8 and no parser omissions.
Its exact encoding is UTF-8 of sorted relative paths, one space, lowercase
file SHA-256 and one LF per entry. The earlier snapshots are preserved as
`complexity-before-lint-correction.json` and
`complexity-before-builtin-override.json`. `cargo fmt --all --check`, schema
no-drift and whitespace checks pass. Final specialist review is **Approve**
for the sealed local diff and focused evidence; it independently verifies the
final 22-file manifest and aggregate. There are no remaining source or focused
verification blockers for this increment.

## Scope and uncertainty

Physical Join mapping changes actual column kernels and selected output-byte
preflight. Retained schemas, row charges, checkpoint formats and the existing
ASOF projection eligibility remain outside this increment's behavioral change.
No performance acceptance, complete eight-target verdict, 90% coverage gate or
remote CI result follows from these local checks. There has been no push, PR
update or merge for this increment.
