# Branch Review: join CPU pipeline

**Branch:** `feature/join-cpu-pipeline` → `main`

**Feature base:** `80d98ae089c8a317c2a884967b83728c94cb87ff`

**Reviewed and measured source:**
`61aab0e571800df8e8a18ad2244993d0929f5247`.

**Files:** 44 implementation, test, specification and documentation files, plus
this review.

The source review used the implementation owner's isolated worktree, retained
for subsequent build and measurement work. The reviewer changed only this
report and ran no native compilation, product tests or measurements. No PR,
check-run snapshot or prior GitHub review applied at that handoff. No remote
review or merge action was taken. The follow-up below records the completed
installed-wheel checks and measurements without changing the approved source.

## Summary

ASOF preparation now builds local output plans from contiguous immutable prefix
fragments in CPU workers. Segmented plans preserve local source identities and
shared Arrow owners, while eligible row gathering also works for single-column
output. Interval joins parallelize retained selection, logical charging and
chunk filling, preserving source-order admission and the existing output
acceptance and state commit boundary.

Job-owned diagnostics remain behind `benchmark-internals`. The review checks
these changes against the [specification](../specs/join-cpu-pipeline.md), domain
contracts, allocation accounting, cancellation ownership and checkpoint paths.

## Build and Test Results

Evidence is recorded by the implementation owner under
`target/performance/join-cpu-pipeline/` in the parent checkout. The reviewer read
the logs and execution records. Earlier results are retained evidence from
before the final cold-pool repair, rather than relabeled final-binary results.

- Rust, final default-feature focused suite: **Passed**, 28 tests, zero failures
  or ignored tests (`final-default-green.log`). This includes all `cpu_pipeline`
  cases and the three `test_reuse_only_` service cases. The execution record
  identifies the candidate worktree and a fresh Cargo compilation.
- Rust, earlier feature-enabled focused suite: **Passed**, 93 tests and one
  explicitly ignored empty-task diagnostic (`final-feature-green.log`). The
  filters cover CPU pipeline behavior, shared gather lifecycle, diagnostic
  ownership, output ranges, columnar charging and admission. The ignored
  `test_empty_parallel_roundtrip` is an explicit D7 diagnostic, not a failed
  correctness test.
- Rust, earlier owner and recovery regressions: **Passed**, 24 tests
  (`final-owner-recovery.log`), including output ownership, prefix handling,
  restored backing budgets and expiration behavior.
- Scoped core and binding Clippy with `--lib --tests`,
  `--features benchmark-internals` and `-D warnings`: **Passed**
  (`final-clippy-repaired.log`). Earlier lint findings were repaired.
- PyO3 diagnostic feature-gate test: **Passed**, one test, zero failures
  (`final-binding-feature.log`).
  `benchmark_dispatch_trace_methods_follow_feature_gate` covers diagnostic
  method availability and errors on cleared native instances.
- Python application source: **Not touched**. The full Python product suite was
  **not run** for this review. A focused Rust PyO3 test does not establish that
  suite's status.
- `cargo fmt --all --check`, generated project-schema drift and whitespace:
  **Passed**. No dependency manifest, public project contract or workspace
  unsafe-code policy changed.
- New unresolved failures: **None observed**. The earlier cold-pool regression
  reproduced as one failed and one passing boundary test
  (`cold-boundary-red.log`); the final 28-test run passes both boundary cases and
  the custom-pool case.
- Full workspace, connector, cross-platform and combined 90% Rust line-coverage
  gates: **Not run locally** and not certified by this review. No CI results
  were obtained for this unsubmitted branch.

The core checks compile the selected library test binary with one of these
commands, then execute the Cargo-reported binary with focused filters and
`--test-threads=1`:

```bash
cargo test -p calc-flow --lib --no-run --message-format=json
cargo test -p calc-flow --lib --features benchmark-internals --no-run --message-format=json
```

The final default binary uses `cpu_pipeline test_reuse_only_`. The earlier
feature binary uses `cpu_pipeline runtime::streaming::gather_work::tests
benchmark_internals::join_cpu_trace output_ranges columnar_charge
columnar_admission admission_controls`. Build helpers set the parent checkout's
`target/` as `CARGO_TARGET_DIR` and verify the source manifest and non-fresh
compilation, avoiding cross-worktree artifact substitution.

The 24 owner/recovery tests reuse the earlier feature binary with filters
`operator::asof::finalize::owner_runs`,
`operator::asof::finalize::prefix::tests`,
`operator::asof::tests::restored_workspace`,
`operator::asof::tests::gather_admission` and
`operator::asof::tests::expiration_output_integration`, with
`--test-threads=1`. The final scoped lint and binding commands are:

```bash
cargo clippy -p calc-flow -p calc-flow-python --lib --tests --features benchmark-internals -- -D warnings
cargo test -p calc-flow-python --lib --features benchmark-internals benchmark_dispatch_trace_methods_follow_feature_gate -- --test-threads=1
```

These commands use `CARGO_BUILD_JOBS=2`, the parent checkout's `target/` and its
managed `.venv/bin/python` as `PYO3_PYTHON`. The binding test also receives that
interpreter's shared-library directory through `LD_LIBRARY_PATH`.

## Blocking Issues

No source finding remains. The following review findings are closed in the
frozen implementation:

- **ASOF fragment and prefix funding:** worker summaries and the owner summary
  retain independent credit while simultaneously live. Snapshot metadata and
  the owned operator name are prepaid; optional control refusal falls back
  cleanly. Allocation-backed tests cover multiple source and key cardinalities.
- **Cooperative snapshot and fallback work:** ASOF capture yields during large
  snapshots. Interval serial-layout compaction obeys the existing visit and
  byte quantum while preserving the already computed logical charges.
- **Interval optional metadata:** settled selection credit shrinks to the
  remaining metadata. Refused chunk preparation drops prepaid units,
  references and optional controls before compacting into the serial layout.
  Charges computed before payload projection are preserved.
- **Cold interval service admission:** both retained stages use the actual
  incremental pool's exact type. Only the standard `UnboundedMemoryPool` may
  start a generation; bounded and custom pools request atomic reuse. The
  readiness check follows pressure retirement under the same Home lock as
  attempt installation. An installed unsettled attempt prevents retirement,
  so later dispatch cannot silently fund a new generation. Cold finite and
  custom pools retain the serial admission guarantee.
- **Diagnostic identity and completion:** ordinary and observed submissions
  receive the correct operator identity at their shared admission point.
  Trace retrieval requires completed, naturally ended work with settled spans;
  thread CPU attribution is confined to synchronous worker spans.

All requested focused verification conditions have passed. Required CI and
coverage remain uncertified. The measurement follow-up below records the
per-configuration performance conclusions separately from source approval.

## Style Issues

None remain. Scoped Clippy and formatting pass without new lint exemptions.
Mutation is confined to owned preparation, service and state boundaries; worker
snapshots retain immutable input owners. No new dependency, executable project
configuration or checkpoint format was introduced.

## Test Coverage

ASOF coverage includes hot-key parallel preparation, equal-time matching,
source-local output identities, nulls, shared left owners, CPU capacities,
actual single-column worker overlap, allocation bounds, optional budget
refusal, snapshot yielding and cancellation with delayed ownership release.
Existing output-rejection, restoration and prefix tests cover the retained
commit boundary.

Interval coverage includes exact row identities and charges, ordered chunks,
probe-only input, delayed worker retirement, mandatory refusal before commit
and budget fallback after projection. The warm refusal fixtures bootstrap the
same Home with an independent standard unbounded pool and assert actual
selection-worker counts: four before chunk refusal and two at the exact serial
capacity boundary. Cold finite and custom pools receive no warmup. These tests
therefore exercise both parallel preparation and the serial fallback rather
than passing solely through an unused parallel path.

Service tests cover cold refusal without retained credit, warm reuse of the
same generation and generation retirement after scope creation. Final pool
cleanup is checked explicitly. The focused checks do not replace the full
regression and coverage gates.

## Documentation Consistency

The ASOF guide, streaming guide, changelog and specification describe the final
scope. Owner assembly is described in terms of actual fragment, span, source
and key-owner metadata; fragmented input can still have row-scale metadata.
Interval append coordinates, expiration preparation and index updates remain
on the owner. The bounded/custom cold-service fallback is documented.

At the original source-review handoff, candidate acquisition had not begun and
the new guides made no measured-speedup claim. The fixed five-configuration
contract and separate diagnostic provenance remain in the specification. The
completed conclusions are now recorded in the follow-up below and the
[performance report](../analysis/join-cpu-pipeline.md). Changed Markdown
structure, links, whitespace and final newlines were checked.

## Verdict

**Approve** source commit `61aab0e571800df8e8a18ad2244993d0929f5247` and its focused
verification. No blocking source, style or documentation finding remains.
Required CI and coverage remain uncertified. This report does not declare the
branch merge-ready or authorize a merge.

## Installed-wheel and measurement follow-up

This narrow follow-up reviews the completed evidence and
[final performance report](../analysis/join-cpu-pipeline.md). It leaves source
approval bound to `61aab0e571800df8e8a18ad2244993d0929f5247`. The subsequent
analysis/review documentation commit is not a new build or measurement source.
No product source, benchmark, native artifact or measurement was changed by
this review.

The final installed default-release wheel passed **seven Python tests** from
`test_symbolic_stream_join_acceptance.py` and `test_stream_budget_defaults.py`
(`installed-candidate.log`). `installed-candidate-proof.json` records successful
execution, the sealed import path and absence of diagnostic methods. Its native
SHA-256 matches the formal candidate build:
`1c1ad9934bdbb229976ff4d0d283e751d5ce65dd80fdf2e6437014a54dfe2fca`.
This is focused installed-wheel evidence; the full Python suite, required CI
and combined coverage gate remain unverified.

The reviewer checked `formal-offline-audit.json`,
`diagnostic-offline-audit.json`, `diagnostic-analysis.json`, the installed-wheel
proof and log, and the report's interpretation of the maintained comparison
function. All five declared configurations completed two rounds of ten pairs:
**200 formal observations and 20 excluded warmups**. The formal baseline is
`80d98ae089c8a317c2a884967b83728c94cb87ff`; default and diagnostic candidate builds
both identify the approved source, with separate native hashes and features.

| Configuration             | P50 time change | Maintained verdict |
|---------------------------|-----------------|--------------------|
| ASOF 1,000,000 / 1 CPU    | +6.25%          | Inconclusive       |
| ASOF 1,000,000 / 32 CPU   | -38.53%         | Improved           |
| Interval 262,144 / 1 CPU  | +2.01%          | Inconclusive       |
| Interval 262,144 / 8 CPU  | -19.49%         | Improved           |
| Interval 262,144 / 32 CPU | -8.59%          | Inconclusive       |

Negative changes mean shorter time relative to the baseline. The two improved
cases satisfy the maintained rule in both rounds. The other three remain
inconclusive: neither their pooled P50 changes nor the absence of a confirmed
regression establishes equivalence or no regression. Nearest-rank descriptive
P95 and the maintained paired intervals retain their separate definitions.
Cross-CPU ratios are descriptive because those configurations were not paired.

The single 32-CPU profile per version and scenario supports local parallel
execution, not sustained use of 26 or 28 executing cores. The report correctly
distinguishes peak instrumented worker threads from average process CPU use,
and overlapping worker sums from parent-stage wall time. It does not attribute
the ASOF improvement to copied row gathering, which this workload did not
exercise, or infer the cause of interval's 8-to-32-CPU difference from a single
32-CPU profile. The interval conclusion is limited to the observed remaining
wall time rather than a proved scaling cause.

The diagnostic audit preserves four accepted warmups, four observations and
one failed warmup: **nine actual diagnostic calls**. The failed interval warmup
retains its error report and process output; its complete raw trace was not
saved before validation failed. The technical repair and its distinct baseline
instrumentation build remain explicit and do not alter formal statistics.
Total acquisition was **126.373 seconds**, including that failed attempt,
within the declared 600-second budget; builds are accounted for separately.
The raw evidence is retained at the local paths linked by the analysis report.

**Documentation verdict: Approve.** The final analysis agrees with the audited
evidence and states its statistical, diagnostic and coverage limits. Source
approval remains unchanged; there is no blanket performance-pass or
merge-ready claim.
