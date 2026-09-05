# Warm streaming performance implementation

Status: in progress. Base: `7f305dad9bfc2e2c9e4dda68565cbc564711d1a3`.

## Objective

Complete both optimization rounds in one pull request. Preserve the actual
StreamingRunner benchmark boundary: prepared Data enqueue, source and channel
dispatch, rolling ingestion and watermark finalization, sink, and to_pyarrow.
Compilation, startup, history preparation, validation, checkpoint, and shutdown
remain outside the measured interval. Report forced-GC and continuous execution
separately; changing the benchmark is not an engine speedup.

## Deliverables

- [ ] P0: measure data and watermark handling, source/sink boundaries, queueing,
  and rolling phases; add reproducible warm-state benchmark and correctness gates.
- [ ] P1: eliminate unnecessary SQL planning for identity, column selection,
  and rename projections, preserving schemas, metadata, errors, and ordering.
- [ ] P2: use columnar ordered rolling ingestion/finalization, with a validated
  fast path and an equivalent general path for unordered inputs.
- [ ] P3: bound updates by touched entities and retained window suffixes; avoid
  full row materialization and unnecessary whole-state copying.
- [ ] P4: reduce Python/Rust scheduling and callback allocation without weakening
  event ordering, backpressure, cancellation, or connector lifecycle ownership.
- [ ] Compare baseline and candidate release builds on the same machine across
  10,240 / 102,400 / 1,024,000 / 10,240,000 history rows and
  64 / 640 / 6,400 / 64,000 appended rows, for SMA and dual SMA with 64 entities.
- [ ] Document measured P50/P95, uncertainty, provenance, limits, and regressions.
- [ ] Pass relevant repository validation and publish one reviewable pull request.

## Semantic constraints

Input envelopes remain immutable. Final rows cannot be emitted before their
watermark. Preserve canonical ordering, atomic envelope validation, duplicate
and lateness behavior, schema/null/numerical policies, checkpoint compatibility,
recovery, output budgeting, and cancellation. An optimization must not silently
drop an unsupported shape; retain an equivalent fallback.

## Performance budgets

The previously discussed ranges are hypotheses, not acceptance evidence.
First-round provisional P50 goals under the original benchmark conditions are
1.3 ms for 64-row SMA, 60 ms for 64,000-row SMA, and 65 ms for 64,000-row dual SMA.
Full-round exploration targets are 0.42–0.73 ms for 64-row SMA and
16.4–32.7 ms for 64,000-row SMA. Reconcile these with phase measurements.
Complete both rounds even if a target needs an evidence-backed revision.

## Verification record

Observed RED on the base implementation:

- `cargo test -p calc-flow --lib operator::expression::tests -- --nocapture`:
  two projection tests fail because a SQL runtime is created for column-only
  projection; computed/filter fallback test passes.
- `watermark_processing_is_measured_without_a_data_message`: deterministic
  injected-clock regression reports 0 ns instead of the expected 17 ns.
- `ordered_stream_finalization_reuses_input_columns_and_keeps_incremental_state`:
  input and output Arrow buffer pointers differ before the columnar fast path.
- `warm_stream_finalization_allocations_depend_on_touched_entities`: updating
  one entity allocates 85 times with four resident entities, but 12,361 times
  with 4,096 resident entities before touched-entity staging.
- `runtime::tests`: a completed Future reports an empty completion channel and
  a ready coroutine runs one loop turn late before the eager/inline bridge.
- `ready_dispatches_share_one_wakeup_and_yield_between_bounded_groups`:
  129 ready calls produce 129 thread-safe wakeups before shared dispatch.
- Extending the ordered finalization test to compare retained-row allocation
  pointers exposes another copy: unchanged history rows are cloned on every
  append. The follow-up stages append deltas and trims owned tails at commit.
- Comparing fast projections directly against the SQL reference reveals lost
  schema-level metadata. Preserve schema and field metadata, and return the
  original immutable envelope for exact identity projections.

Passing intermediate checks (not final-head acceptance):

- Core lib: 762 passed, four ignored; rolling ordered-stream integration: five
  passed; runner: two; state: 28; stream: 18. This run covers P0–P3 before P4.
- Python bridge: eager coroutine, inline Future completion, and custom task
  factory tests pass after the first P4 change. Shared dispatch and broader
  lifecycle validation remain in progress.
- P4 intermediate: all 104 PyO3 unit tests pass. The release wheel also passes
  35 continuous-runtime and warm-harness Python tests. Both focused core and
  binding clippy checks pass before the retained-tail follow-up.
- An exploratory five-pair forced-GC release smoke run (dirty candidate, not
  final evidence) at H=1,024,000 reports SMA N=64 at 1.784 -> 1.117 ms and
  N=64,000 at 135.351 -> 9.559 ms. Dual SMA reports 1.709 -> 1.353 ms and
  135.785 -> 10.439 ms. Small-batch targets remain unmet; investigate retained
  history copying and callback boundaries before the full matrix.
- A full-core rerun was invalidated by a concurrent focused-test build replacing
  the running test executable; the checkpoint smoke re-exec received a deleted
  executable path. Rerun core tests after compilation finishes, without another
  build targeting the same executable during the run. This is not evidence of
  a checkpoint assertion regression or a passing full-core gate.
- After the retained-tail and projection-metadata fixes, core clippy passes;
  the uncontended core run passes 764 unit tests (four ignored) and all 53
  rolling integration tests (ordered: five, runner: two, state: 28, stream: 18).
  The projection reference checks and unchanged-history pointer check are green.

## Reproduction entrypoints

`scripts/profile_warm_stream.py build` builds a release wheel and binds its
native-module and wheel hashes to the checked source identity. Use separate
Cargo target directories for baseline and candidate worktrees. `compare`
installs each wheel into an isolated target-tree directory and uses two
persistent workers with alternating AB/BA samples; IPC and validation remain
outside timing. `benchmarks/warm_stream.py` owns each advancing runner and
checks every warm and measured batch against the numerical/identity oracle.
The full default matrix has seven distinct (history, append) points, two
indicators, and forced/normal-GC modes. JSON retains individual callback phase
intervals, raw samples, before/after operator metrics, P50/P95, and a seeded
20,000-resample paired median-speedup interval. Final-head/full-matrix evidence,
metric phase refinement, lifecycle edge cases, and the PR are still pending.

Every behavior change starts with an observed focused failing test. Record the
commands and expected failures, then their passing reruns. Run Rust/Python,
documentation, formatting, coverage, generated-contract, and affected Studio
checks before final handoff. Bind benchmark evidence and remote checks to the
published commit; do not use a stale baseline wheel as current-main evidence.
