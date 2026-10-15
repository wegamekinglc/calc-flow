## Branch Review: issue 406 compiled stream expressions

**Branch:** `feature/issue406-compiled-expressions`

**Reviewed candidate:** `33e9a184c39a05b14f07770f51de337c5ac313e6`

**Base:** `d83e68ff303032c0cdeb8a934d4149626d6f08c1`

**Scope:** ten changed files, including six Rust files. The initial implementation
at `fe5ca7b2bf0b33eb1c503d17e945d0c8e2420cd1` was read in full; the final
follow-up was reviewed against that implementation. The base contains the prior
phase 2.2 review, whose approval is unchanged.

### Summary

Eligible stream ExpressionOperator queries use SQL for their first batch, then
compile the existing bounded empty-scan template into immutable DataFusion
projection/filter steps. Later calls own their coalescers and result collection.
Batch execution, public runtime SQL methods, column-only projection,
configuration, fingerprints and checkpoint formats retain their contracts.

The final follow-up corrects two stale module comments and normalizes three raw
Arrow-error branches through DataFusionError. It adds a defensive error-envelope
comparison and an actual SQL filter/projection control. No unresolved execution,
schema, chunk-order, cache-ownership or cancellation blocker was found.

### Build and Test Results

These are owner executions inspected by the reviewer, without repeating passing
checks. Evidence is under repository-root
`target/issue406-compiled-expressions-20261014/`.

- Rust focused controls: **Passed**. The final
  `compiled-arrow-envelope-component-green.log` reports the augmented existing
  registration passing in 0.18s after 1m47s compilation, with 107.643s runner
  elapsed. It includes the prior runtime controls, the defensive Arrow-envelope
  comparison and the filter-only-output control. Default libtest registration
  count remains 2,033.
- Adjacent expression controls: **Passed**. The unchanged
  `compiled-expressions-adjacent.log` records eight passes in 0.05s, excluding
  the augmented registration. They were not repeated.
- Scoped core Clippy: **Passed**. The final command was
  `cargo clippy -p calc-flow --lib --tests -- -D warnings`, using **default
  features**. `compiled-expressions-clippy-final.log` records 29.44s tool elapsed
  and 29.577s runner elapsed. This does not establish workspace/all-feature lint
  coverage. The earlier corrected Clippy evidence used the same limited scope.
- Formatting, project-schema no-drift and whitespace/newline checks: recorded
  **Passed** for the final source. Documentation verification records 27 local
  links/anchors and preserved table structure.
- Python/new-native integration, release verification, full regression CI,
  cross-platform checks and the combined 90% Rust line gate: **Not run/unverified**
  for this increment.
- New unresolved failures: **None found in the scoped evidence**. Full-suite
  regression status remains unverified.

The original expected RED remains in `compiled-expressions-red.log`: value,
schema and metadata parity passed before the private SQL metric count was two
rather than one. It failed after 3m11s compilation and 0.02s execution, with
191.454s runner elapsed. Fixture-only Cancelled-pattern and ignored-unit lint
errors were corrected before the recorded passing checks.

The follow-up preserves its separate fixture compile failure, which needed an
explicit `Arc<dyn ExecutionPlan>`, and the corrected expected RED in
`compiled-arrow-envelope-component-corrected-red.log`. That test failed in
0.33s after 1m32s compilation, with 93.571s runner elapsed: the raw Arrow error
lacked the real MemoryStream error's `Arrow error:` prefix. The final GREEN
establishes the normalization. This is an invariant/component comparison that
deliberately bypasses cache schema matching; no legal SQL production trigger
for that defensive failure has been established.

The reviewer independently matched every final Rust file's disk and committed
bytes to `complexity-final.json` and recomputed the sorted
`file + ASCII space + sha256 + LF` aggregate:

```text
94c6c1919367193c933556cc0830a5d58901ac62e964239913290873c15d713e
```

The manifest contains six Rust files and 40 changed/new functions, maximum CCN 8,
with no changed/new parser omissions. Its two parser omissions,
`physical_plan_statistics` and `parallelism_decision`, are exact unchanged
baseline excerpts. The archived initial and comment-only seals identify their
earlier snapshots; neither substitutes for this final source identity.

### Blocking Issues

None for this local increment. The agreed raw Arrow-error normalization is
present in `compiled::push` and `filter_record`, and the component helper has its
test-only re-export and passing call. The final source and evidence resolve the
earlier pending review items.

### Style Issues

None outstanding. The datafusion_template module comment now distinguishes
public SQL plan rebinding from stream compiled evaluation; the projection module
comment correctly describes its column-only path. No dependency addition,
unsafe-code exception, executable configuration or public API addition was
found.

### Test Coverage

The controls cover nullable arithmetic/filtering, exact record chunks,
multiple/empty records, zero results, Float64/Float32 and metadata schema cache
changes, a genuine zero-column three-row input with constant projection,
metadata/source-cursor parity, fresh cloned runtimes, released raw input-buffer
owners, exact cast errors and cached retry. Aggregate/window/stable functions
and selected UDFs, both declared and resource-selected, retain SQL execution.
Mid-call cancellation and future abandonment after a yield emit no output,
release call-owned buffers and permit a cached retry.

The added SQL control selects `upper(symbol)` while filtering on omitted
`Price`, compares distinct batches with SQL, and checks the first SQL plan's
FilterExec projection removes the predicate column. A private SQL metric count
of one proves later batches use compiled evaluation and correctly remap the
following Projector. The separate defensive component control compares genuine
RecordBatch projection failure with real MemoryStream failure; its deliberately
invalid schema path is accurately distinguished from SQL coverage.

Source review confirms templates hold empty scans and immutable expressions;
filter buffering is per call. Physical-node, fetch and ordering eligibility
remain conservative. Against pinned DataFusion 54 source, predicate evaluation,
filter projection, Arrow filtering and LimitedBatchCoalescer behavior match
FilterExecStream. Flushes preserve dependencies between successive filter steps.
Output collection reuses the existing SQL visible-slice limit helper.

Evaluation checks cancellation/deadlines between records and steps and before
output, yielding between records. Synchronous DataFusion/Arrow kernels retain
their existing interruption limits. Evaluation errors drop call-owned state
before external emission; collector failure during the subsequent emit call
remains a distinct boundary.

### Documentation Consistency

The specification, runtime-envelope execution-path section, development plan and
analysis describe eligibility, ownership, cancellation and metrics scope without
a speedup claim. Private stream runtime SQL-query metrics now measure actual SQL
fallback calls; runner callback/status measurements still cover every call.
Public DataFusionRuntime SQL and batch metrics are unchanged. The two stale
source comments are corrected in the reviewed head. No qualifying public-contract
changelog change was identified.

The analysis attributes initial and final evidence separately, labels the
defensive component test accurately, and states the default-feature Clippy scope.
Arithmetic-tail fusion, release integration, performance and the original
eight-target issue acceptance remain outside this increment's demonstrated
result.

### Verdict

**Approve — final local specialist review of
`33e9a184c39a05b14f07770f51de337c5ac313e6`.**

This approves the reviewed increment and its scoped implementation evidence.
It does not certify CI, 90% coverage, release/new-native Python integration,
performance, merge readiness or completion of issue 406. No PR/CI status was
queried, no remote review was submitted, and the reviewer performed no product
edits, builds, test executions or measurements.
