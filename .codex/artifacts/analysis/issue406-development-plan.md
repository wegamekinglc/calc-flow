# Issue 406: development and acceptance status

The active user objective is to complete the development plan against the
eight-workload target table supplied on 2026-10-11. That table supersedes the
earlier example comparison table and the incomplete seven-workload final
comparison protocol. All changes remain in PR #407, without a dependency on
PR #405. The objective is not complete.

## Required performance coverage

Historical main figures below are supplied reference values, not fresh results.
Every final report must retain all eight rows, identify the measured main and
candidate commits, and distinguish measurements from targets.

| Workload (1M rows unless specified)        | Historical main          | Single-thread target                | Parallel target     |
|--------------------------------------------|--------------------------|-------------------------------------|---------------------|
| Stream Join, sealed dimension              | 701 ms                   | <= 60 ms                            | <= 15 ms            |
| Stream Join, 10M rows                      | 8,061 ms                 | <= 600 ms                           | <= 150 ms           |
| Retained Stream Join, 1M to 4M scaling     | 5.7x (previous main)     | <= 4.4x                             | —                   |
| Join capture / restore per row             | 1.8 / 2.3 us             | <= 0.1 / 0.2 us                     | —                   |
| Join handler latency during compaction     | 40x steady               | <= 2x steady                        | —                   |
| Stream ASOF                                | 391 ms                   | <= 100 ms                           | <= 30 ms            |
| ASOF suite overhead, 1,024-row batches     | 167 ms over operator     | <= 2x measured projection floor     | —                   |
| Join operator cost per 1,024-row batch     | 0.2–0.9 ms               | <= 0.05 ms                          | —                   |

The absolute stream targets use checkpoint-disabled execution. Separately
measure enabled and low-frequency checkpoints on applicable streaming paths;
report actual periodic coverage and terminal-checkpoint cost. Capture/restore
and checkpoint compaction require their intrinsic checkpoint lifecycle, rather
than pretending they execute with checkpointing disabled. Define their timing
boundaries separately. The approximate historical 70 ms overhead target
(twice the projection floor) must not replace a current, matched measurement.
Record both that reference and twice the measured floor. As in the maintained
suite underlying the user's table, ASOF's 1M size means 1M rows per side,
2M aggregate input rows and 1M output rows. Use that size with 1,024-row batches;
do not silently substitute the older 100k-per-side diagnostic. Join's size means
quote rows plus its separate 64-row dimension. Capture/restore denominators
count actual retained rows across both sides. Total ASOF/projection ratios alone
do not isolate harness overhead.

No row may disappear because it is expensive or needs a missing harness. A
missing metric requires implementation and validation of the measurement path.
An unsupported exact-main mode stays explicitly unsupported; patching main or
using a long interval cannot manufacture a true-off baseline. The previously
observed main EOF checkpoint failure remains lifecycle-invalid when reproduced.

## Source audit and remaining proposals

This source audit describes runtime capabilities at `b9a12255`, before the
pending staging increment. It does not certify the new absolute targets.
Earlier paired evidence remains valid only for its recorded revisions and scope.

- **Phase 0:** probe-key reuse (0.1) and flat multi-source gathering (0.2)
  exist. Owned probe cancellation still checks each input position (0.3).
  Contiguous output slicing (0.4) is absent and must preserve existing
  independent-output-buffer guarantees. Release-profile changes (0.5) are
  unimplemented proposals. CPU and checkpoint evidence exist, but thread,
  event-layout, unordered-input and wide-key coverage (0.6) remains partial.
- **Phase 1:** admission masks, typed timestamps, persistent key IDs,
  immutable payload chunks, per-key ordered runs, count/limit/fill and shared
  checkpoint payloads exist. Row admission descriptors (1.1), row-oriented
  key framing (1.2), dense row metadata and ordered expiration maps (1.3),
  gather rather than incoming identity selection (1.4), and per-row checkpoint
  journals/base compaction (1.5) remain. These are partial implementations of
  the proposed architecture, not measured proof that every replacement pays.
- **Phase 2:** ASOF has exact-projection and retained-payload pruning; Join
  lacks equivalent pushdown (2.1–2.2). Expression templates are cached but
  still rebound per batch (2.3). Exact projection chains can fuse after ASOF;
  Join/arithmetic tail fusion is absent (2.4). Dictionaries are local to
  operators; shared graph-wide key domains are absent (2.5).
- **Phase 3:** ASOF has typed/interned admission with a restricted columnar
  left-side path (3.1), monotonic cursors and parallel probes (3.2), and sorted
  permutations/ordered merges with remaining row-wise installation (3.3).
  Latest applicable right payload plus a suffix is implemented (3.4). Earlier
  identity history is required for duplicate detection and is not unretired
  payload.
- **Phase 4:** independent ingress admission (4.1) is absent. Persistent
  range-partitioned probe work and ASOF gather exist, but Join gathers columns
  synchronously and combines separate worker fragments (4.2). Key-partitioned
  operator instances (4.3) are absent; their saturation premise is unproved.
- **Phase 5:** batched Python bridging (5.1), a materialized native Arrow stream
  source (5.2), and latency-bounded ingress coalescing (5.4) are absent. Shared
  byte-estimate caching exists, while endpoint label precomputation and status
  throttling are remaining fixed-cost proposals (5.3).

Relevant implementation boundaries are `operator/join/`, `operator/asof/`,
`pipeline/stream/projection.rs`, `runtime/streaming/gather_work/`,
`runtime/streaming/runner/operator_fusion/`, and the Python continuous bridge.
Future changes must preserve ordering, caller-owned buffers, limit accounting,
checkpoint cuts, duplicate detection, delivery and owned cancellation.

## Current increment and next work

The completed initial diagnosis measured checkpoint-off separately from 1 s
and 100 ms checkpoint modes. It found repeated staging-directory fsyncs in a
four-segment owner snapshot. The selected batch-staging implementation retains
each file sync and batches only the owner-directory confirmation. It also fixes
retry after visible staging bytes outlive a failed directory sync.

Its added operator cancellation test exposed an existing outer cancellation
gap: `dispatch_or_cancel` could drop admitted checkpoint staging before
settlement. The fix moves the admitted checkpoint boundary outside that select;
eight staging/publication cases now pass after the recorded RED. Callback
cancellation controls and the ASOF post-capture status regression also pass.
Final source review approved the implementation after 85 distinct focused test
entries, the new API doctest, scoped Clippy and formatting/contract checks.
Performance acceptance remains outstanding. The new eight-row acceptance
protocol replaces the old unexecuted final comparison inventory; preserve the
already collected evidence as historical evidence, without relabeling it.

The complete initial inventory has now been attempted against pinned main
`588726c8` and runtime `e8a28d99`, with execution harness `427426b2`. All eight
rows are represented: 52 metric arms have 20 samples each; 12 are blocked by
eight physical checkpoint preparation failures. The absolute OFF goals remain
unmet. Main's enabled runtime timings are processing diagnostics because EOF
checkpoint completion fails. See the [complete diagnosis](issue406-eight-workload-diagnosis.md)
for all eight rows, separate OFF/LOW/ON observations, failure attribution and
provenance. Four fixed CPU profiles also completed within their 180-second
budget; they do not replace acceptance measurements.

Three source corrections are reviewed and committed: `a9fe388c` bounds checkpoint
construction by contiguous row runs (22 focused tests), and `9be7bd3f` separates
probe scratch from retained key credit (15 focused tests). `988a5972` preallocates
base/delta index and payload-frame wire buffers exactly (19 focused tests).
These preserve prepaid allocation and alias lifetimes; each has recorded
RED/GREEN evidence and passing
library Clippy. Test-target Clippy retains the known pinned-toolchain generated
libtest failure. These corrections have no new performance measurements yet.

The fixed 4M diagnostic at `9063f2fe` has now run once. Full admission passes,
but fresh preparation fails its IPC write reservation: 997,664,473 live bytes
plus 134,217,216 requested bytes exceeds the unchanged 1 GiB pool. The collector
stopped before all timing samples; cleanup settled and reservation returned to
zero. Build/audit time was 614.567 seconds; the stopped attempt took 8.065 seconds.
The eight-row timing table still describes `e8a28d99`.

The IPC lifetime correction in `8a740b1e` reproduces that failure at 8,192 rows
in a 2,400,000-byte pool, then succeeds by retiring the compacted input after
owned Arrow messages are generated and before output growth. All 22 focused
checks pass; library Clippy passes. No new performance samples were collected.

Further source analysis shows that two independent 4M left serialized bases
plus the observed admitted state exceed 1 GiB by at least 27,248,808 bytes,
before scratch/right-side/control costs. Restore's existing fold reservations
independently exceed 6.576 GB for 4,000,064 upserts. Constructor chunking or the
IPC phase split alone cannot make the complete lifecycle pass. Keep the next
release build/full diagnostic held until the relevant ownership and restore
bounds are resolved.

Commit `f9adf9e2` now reuses fresh preparation only when all four Cut fields
match. Three new tests and seven existing writer regressions pass, including
full-pool reuse, stale fields, complete byte/recovery equivalence, cancellation
and retirement. Library Clippy passes; the known test-target generated-libtest
diagnostic remains. This avoids a redundant writer for an unchanged cut; it does
not solve held-history replacement or restore capacity.

Commit `640d8c84` now separates each segment's byte-capacity reservation from
Base inventory controls and temporary encoding credit. Held aliases retain their
own backing and registration funding; completing an encoder retires only scratch
that is no longer live. Four recorded ownership/admission failures now pass,
along with the related writer controls (34 unique focused tests). Library Clippy
passes. These are correctness and ownership checks, with no new timing samples.

Commit `0d2b23c7` implements the selected single-string local interner. It
constructs canonical frames only for distinct values, preserving full canonical
bytes, collision equality, admission arithmetic and aliases. Nine focused tests
and library Clippy pass. Work-count tests reduce 14 frames to 6 and 28 to 12;
owned-key encoding counts were already distinct-value counts. No timing gain is
claimed from these tests.

Commit `0dbbfea5` now borrows pending rows for synchronous checkpoint encoding.
Each side owns an exact prepaid pointer selection; no full pending-log or
materialized StoredRow copy is created, and scratch retires with its allocations.
Three runtime REDs resolve, and 30 unique focused tests pass, including complete
capture fault/retry and distinct Base/Delta worker ownership through cancellation.
Library Clippy passes; the known generated-libtest diagnostic remains. Async
preparation still owns its original pending copy through worker drain and is not
certified to fit the hypothetical 4M dirty batch.

The next implementation steps are:

1. For fully checkpointed append-only history, finalize the eligibility and
   funding guards, then implement a bounded sorted-index merge sharing existing
   whole payload segments. Pending operations,
   tombstones and partially dead payloads retain the existing encoder. Reconcile
   all held old/new backing, capture controls and retirement ownership.
2. Implement bounded restore folding and scoped per-row scratch while preserving
   malformed-history rejection and atomic recovery. Explicitly select the
   expiration representation and its source-pool funding boundary: charging the
   proposed append-grown arrays to ordinary source state also requires reducing
   fresh writer scratch and held-rebase overlap.
3. Recompute complete simultaneous-reservation bounds before another release
   build/full 4M diagnostic. The worksheet's 941 MB rebase, 928 MB restore and
   borrowed E1's 979 MB leading terms exclude unresolved controls and are not
   capacity certificates.
4. Continue the remaining Join, ASOF, parallelism and harness improvements using
   the completed CPU profiles. Measure the interner and pending-writer increments
   with the final runtime; retain all eight targets and checkpoint policies.

The observed Python complexity CI failure is corrected in `25b6eb78` without
changing the baseline or acceptance semantics. All 102 unique focused tests pass,
including failure-evidence preservation, and Ruff/complexity checks pass. Commit
`fe40a96c` corrects checkpoint-aware soak fixtures and canonical publication hook
paths; the five previously failing cases pass locally. Windows awaits CI. The
pinned-toolchain generated-libtest `large_stack_arrays` failure remains unresolved;
no lint suppression, toolchain change or test removal is included.

Exact wire allocation preserves byte-for-byte builder output and cancellation;
constructor coalescing preserves the v2 format and deterministic equivalent
recovery without claiming the old IPC physical grouping. Each increment needs
reviewed implementation and focused evidence. The final comparison still covers
every row and checkpoint policy against the then-current runtime. An unchanged
known failure is retained, not retried to fill a table or disguised as noise.

Completion requires reviewed implementations, focused correctness checks,
complete and reproducible eight-row evidence with separate checkpoint modes,
and a PR #407 performance table comparing pinned main and current code. Targets
that are missed, comparisons that are invalid, and unresolved CI checks must
remain visible. Neither a local speedup nor finishing one increment completes
the full objective or authorizes merging.
