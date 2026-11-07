## PR #556 Review: SQL and streaming boundary repairs

**Branch:** `fix/weekly-review-20261107` → `main`.
Reviewed source head: `7a2463a89a2f4340a64cbdec90c97d7db05097ba`, after merging
main `2193f8ee7fbdc75c2bb6c1da35b21fcb491b8b86`.
The parent supplied the isolated worktree; this reviewer did not alter product
source, commit, publish, remove that shared worktree, or run a benchmark.

### Summary

The merge preserves the weekly SQL correctness and accounting repairs while
retaining main's expanded immutable query-template implementation. Two Python
test refactors address the prior Codacy complexity and pickle annotations
without changing product behavior or dropping test assertions.

This review covers the merge resolution, its affected query-template and native
join callers, and the two test refactors. The original weekly review remains
the scoped evidence for the unchanged ASOF, Python input, exception and cleanup
repairs; this is not a fresh audit of every change merged into main.

### Build and Test Results

- Native checks pass from one freshly rebuilt `calc-flow` library test binary:
  seven input-partition, five finite-template and 28 native-join tests (40 total).
- The command is `cargo test --locked -p calc-flow --features benchmark-internals
  --lib --no-run --message-format=json`, using the existing dedicated debug
  target, one Cargo job, incremental compilation disabled and development
  debug information disabled. Only compatible dependency artifacts are reused.
  Compilation takes 312.54 seconds, separate from the performance budget.
- The selected filters are the seven experimental input-partition tests, the
  five finite-template tests and the native-join module. They directly cover
  the merge's changed planning and rebinding paths. Unchanged ASOF checks are
  not repeated solely to attach a new binary hash.
- The author's five selected Python refactor cases pass, with Ruff and format
  checks also green; their receipt and actual pytest output were inspected.
  Those checks use current Python source over the prior exact `7db` native,
  explicitly qualified as such. This is not a test of a newly rebuilt merge
  wheel. Scanner acceptance, exact final-head full CI and the performance
  comparison belong to the parent. The old head's successful product checks
  do not certify this merge.
- Subsequent exact candidate-wheel validation supersedes that early native
  qualification: the complete `test_arrow_stream_inputs.py`, `test_config.py`
  and `test_batch_query_templates.py` modules pass, 102 tests in 2.15 seconds.
  The receipt and actual pytest log verify installed package/native paths and
  native SHA-256
  `e8193ec5f1ee931e5c86a2e92eb2bc7f6c2f0b6df02b4be20dca1bf715a8bb5a`.
  No additional test was run by this reviewer to repeat that passing evidence.
- The executable SHA-256 is
  `866d3f2bc0f81113ed4bf7bbbb99fec9f1646052ec62a863c747902601674354`.
  Source hashes match before and after the run, and every owned process has
  exited. Commands, exit codes, logs, elapsed times and six source hashes live
  under `target/weekly-review-20261107/pr556-repair/` as `native-build`,
  `partition-green`, `template-green`, `join-green` and `native-final-source`.
  The corresponding test-reported runtimes are 0.07, 0.19 and 7.80 seconds.
- No new native failure or regression was observed in these selected tests.
- The clean baseline and candidate release manifests independently pass
  wheel/native digest verification and source-identity checks. They use
  identical Cargo.lock, Rust/Python versions and release options. The baseline
  and accepted candidate builds take 2,418.07 and 2,545.64 seconds respectively;
  their receipts show overlapping build intervals, all before acquisition.
  The withdrawn first candidate preparation is retained separately and has
  no scientific observation.
- The saved GraphQL snapshot `ci/20261107T071534.877480Z.json` matches source
  head `7a2463a8`: all 33 contexts are `SUCCESS`, including Codacy, combined
  coverage, platform and package checks. All contexts fit in the returned
  page, and GitHub reports merge state `CLEAN`. This is the prepared source
  head's CI result; a later delivery commit still needs its own final checks.

### Blocking Issues

No unresolved source blocker was found in the reviewed repair. Performance
acceptance remains unmet: both environment preflights rejected acquisition
before any worker started, and no further automatic attempt is permitted by
the declared resource-resumption plan. The prepared source head's green CI
cannot waive that gate. A later delivery commit also requires final green CI,
so this report does not grant merge readiness.

### Style Issues

No additional style violation was found in the reviewed source. Codacy passes
on the prepared source head; a later delivery commit remains part of the
parent's final-head handoff.

### Test Coverage

`prepare_query`, `with_batch_templates` and the template capture fields match
current main. The `input_partitions == 1` guards already exist in current main;
they restrict only the benchmark-only input scan partition override. Production
fixed target partition counts above one keep `input_partitions = 1`, so the
new effective-partition cache key and table-map binding remain eligible.

The finite-template module checks multi-input aggregation, same-schema alias
identity, immutable input ownership, fresh run metrics, and planning fallback
for joins and execution-stateful limited queries. These five tests use their
declared default configuration; they are not described as a separate four-thread
performance or runtime acceptance set. The partition module checks both changing
the requested input partitions and changing input batch count under fixed P4.

`plan_query` obtains the same effective partition count as main and passes it
both to native planning and template capture. Its remaining product delta from
main is the native query's local empty rolling audit. Native plans do not pass
through the planner that produces the shared audit; the regular DataFusion path
still reads that planner's audit. The existing differential regressions exercise
CTE shadowing, column alias rebinding and AVG followed by native ASOF/interval.

The materialized-record test extracts preparation into `_materialized_records`;
its output row counts and values, original input rows and metadata, and terminal
task-count assertions remain present. `_pickle_round_trip` serializes the test's
own constructed exception and immediately reads those bytes. The one-line
Semgrep suppression documents that trusted origin and creates no application
decoder or repository-wide exclusion. Real pickle, shallow-copy and deep-copy
round trips retain the exception-state assertions.

### Documentation Consistency

The changelog resolution retains both main's template/snapshot-key entries and
the weekly repair entry. No new API, schema or checkpoint representation is
introduced by this conflict resolution or the test refactors. The original
weekly reports preserve their historical identities; final performance and CI
claims must be updated by the parent only after observing those results.

### Performance environment episodes

Both frozen protocols compare main `2193f8ee` with prepared candidate
`7a2463a8`. Four workloads, two rounds of ten alternating pairs, one warmup and
three samples per fresh worker, CPU placement, full oracles, paired statistics
and source/wheel/native checks stay fixed. No sample count, input size,
regression threshold or guest quiet threshold was relaxed. The worker is
byte-identical across the two episodes.

The first protocol has SHA-256
`8529c3d271d9d7a7c739ad81106716225f9fa29725be0534258007fbcab03183`.
Its preflight rejects mean guest CPU use of 20.309375%, above 10%, despite
CPU 0 measuring 6.2%. The full supervisor episode ends in 8.952325191 seconds
with exit code one and `accepted_acquisition = false`. Its summary contains
no case, and no
worker, warmup or timed sample is produced. The failed records remain intact.

The user explicitly requests waiting for other tasks to finish naturally.
The resource-resumption plan therefore records a separate episode after a
resource-state change, preserves the first failure, extends the build guard
to the observed C++ build/test-driver chain and permits no attempt after this
resumption. It retains the same quiet thresholds and all workload/statistical
conditions. The new supervisor uses a 590-second budget, interrupting at 575
and reserving the final cleanup window; the first consumed time plus the
resumption's budget stay below 600 seconds. The guard only observes boundaries
and does not terminate external processes or claim continuous host isolation.

The second protocol has SHA-256
`382de1ab169249609bdcdcb3da069abec914a08576f3f7b485516aac660ddf6d`.
It also rejects in preflight: CPU 0 is 33%, above 10%, while guest mean use is
5.4875%. Its full supervisor episode ends in 4.536642391 seconds, again with
exit code one, no accepted acquisition and no case, worker, warmup or timed
sample. Neither episode sends a signal or leaves an owned survivor. Both
performance directories contain only their controller summary, not a worker
observation. All 102 file seals in each protocol match current files.

The independently recomputed combined active time is 13.488967582 seconds.
The gap from the first episode's end to the second episode's start is
1,417.932697439 seconds, reported as resource waiting. These are two episodes,
not one continuous 600-second acquisition. The separate native build costs
remain preparation evidence. No third automatic attempt was made; none of the
four cases entered scientific acquisition, so there is no paired timing
verdict, speedup, equivalence or confirmed runtime regression.

The failed summaries, protocol identities, process-settlement receipts and
combined ledger were independently inspected without rerunning a quiet guard,
test or workload. Six source files still match the reviewed candidate and
targeted native test receipt. This reviewer ran no performance workload.

The [evidence archive](../analysis/pr556-repair-evidence.tar.gz) and its
[digest index](../analysis/pr556-repair-evidence-index.json) were independently
verified: 51 members, 60,615 bytes, every member's size and SHA-256 matching the
index. Archive SHA-256 is
`860decb046b8dad2c08d515bbbe665bf16e84ac5af8c568f22f4e3f1e7894ea6`.
The archived state records two completed preflights and no additional attempt;
both failed summaries contain zero cases. This evidence does not establish
benchmark non-regression or authorize merging the draft.

### Verdict

**Comment Only.** Source review, all 40 targeted native tests, 102 exact-wheel
Python tests and the prepared head's 33 CI checks pass. No unresolved source
finding remains in this repair. The declared performance comparison remains
unmeasured; its completion and any later delivery commit's final CI are required
before an overall approval is possible.
