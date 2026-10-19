# Branch Review: join CPU pipeline

**Branch:** `feature/join-cpu-pipeline` → `main`

**Feature base:** `80d98ae089c8a317c2a884967b83728c94cb87ff`

**Reviewed source:** frozen working tree over the feature base; the implementation
owner will bind the final verdict to the resulting commit.

**Files:** 44 implementation, test, specification and documentation files, plus
this review.

This review uses the implementation owner's isolated worktree, which remains
available for the pending build and measurement work. The reviewer changed only
this report and ran no native compilation, product tests or measurements. No PR,
check-run snapshot or prior GitHub review applies to this working diff. No remote
review or merge action was taken.

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

All requested focused verification conditions have passed. Required CI,
coverage and performance acceptance remain separate from this source review.

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

The new documentation makes no measured-speedup claim. Candidate acquisition
has not begun at this handoff. The fixed five-configuration measurement
contract and separate diagnostic provenance remain in the specification;
neither default-release speedup nor CPU-scaling acceptance is established here.
Changed Markdown structure, links, whitespace and final newlines were checked.

## Verdict

**Approve** the frozen source and its focused verification. No blocking source,
style or documentation finding remains. Candidate performance acceptance,
required CI and coverage remain uncertified. This report does not declare the
branch merge-ready or authorize a merge.
