# Issue 406: development and acceptance status

The user has removed the implicit 1 GiB runtime-pool constraint. The current
candidate uses `UnboundedMemoryPool` in ordinary execution and primary checkpoint
diagnostics, while retaining per-runtime reservation accounting and release tracking.
Explicit state, edge, and SQL result collection budgets remain separate. Refusal
controls now install their finite pools explicitly; the earlier 10 MiB and 1 GiB
operator diagnostics are historical bounded controls, not performance requirements.
Checkpoint-history v2 records `pool_policy=unbounded` and null limits, while v1
continues to validate historical bounded evidence. No performance numbers below
have been reattributed to the new policy. Fresh reports must keep all eight targets,
pinned main/current provenance, each pool policy, and distinct OFF/LOW/ON results.

Latest full-inventory performance attempt: the [fresh a681e7da attempt](issue406-fresh-acceptance.md)
stopped on the candidate's 4M retained-ON warmup at epoch 11. It contains all
eight target rows, fresh pinned-main/current mode comparisons, and complete
cff/current retained-OFF controls. Rows 4–8 remain unmeasured in that attempt;
earlier measurements below keep their original revisions. OFF scaling observes
4.011×, while sealed Join 1M/10M still miss their absolute targets. The candidate
failure and all final acceptance verdicts remain unresolved. No retry or fill-in
was performed, and all owned processes settled.

The pending-aware checkpoint rebase correction now preserves the trusted, checkpoint-
enabled, append-only fast path when the current batch contains new rows. It serializes
only pending Upserts, shares the existing payload owners, and retires temporary input
ownership before allocating merged indexes. Both phases remain memory-funded. Exact
checkpoint-cut validation and atomic installation remain unchanged; tombstones,
migration, disabled checkpointing, and untrusted history retain their existing fallback
boundaries. Malformed selected history remains an error.

The unchanged 32,768-left-row plus 64-right-row diagnostic now passes within its
original 10 MiB pool: E6 preparation peaks at 8,009,855 bytes, introduces no new writer-
admission refusal, and preserves the held checkpoint payload. All owned memory returns
to zero. A separate focused test restores the rebased snapshot and verifies continued
outputs. Ownership, cancellation, malformed-input, and fallback controls also pass. The
pending-input fault sweep covers 59 allocation refusals, and the malformed history
control covers 16 variants with pending Upserts. The 16 focused invocations cover 15
distinct tests. Scoped formatting and library Clippy pass, and final specialist review
approves this correction. These are correctness diagnostics, not performance samples.
They do not establish the cause or successful recovery of the managed 4M/E11 failure;
that original workload and the eight performance targets still require fresh evidence.

The active user objective is to complete the development plan against the
eight-workload target table supplied on 2026-10-11. That table supersedes the
earlier example comparison table and the incomplete seven-workload final
comparison protocol. All changes remain in PR #407, without a dependency on
PR #405. The objective is not complete.

## Required performance coverage

Historical main figures below are supplied reference values, not fresh results.
Every final report must retain all eight rows, identify the measured main and
candidate commits, and distinguish measurements from targets.

| Workload (1M rows unless specified)    | Historical main      | Single-thread target            | Parallel target |
|----------------------------------------|----------------------|---------------------------------|-----------------|
| Stream Join, sealed dimension          | 701 ms               | <= 60 ms                        | <= 15 ms        |
| Stream Join, 10M rows                  | 8,061 ms             | <= 600 ms                       | <= 150 ms       |
| Retained Stream Join, 1M to 4M scaling | 5.7x (previous main) | <= 4.4x                         | —               |
| Join capture / restore per row         | 1.8 / 2.3 us         | <= 0.1 / 0.2 us                 | —               |
| Join handler latency during compaction | 40x steady           | <= 2x steady                    | —               |
| Stream ASOF                            | 391 ms               | <= 100 ms                       | <= 30 ms        |
| ASOF suite overhead, 1,024-row batches | 167 ms over operator | <= 2x measured projection floor | —               |
| Join operator cost per 1,024-row batch | 0.2–0.9 ms           | <= 0.05 ms                      | —               |

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

Commit `70941d5f` adds a restricted shared-payload rebase for trusted,
fully checkpointed, append-only V2 history. It merges ordered index records into
exact replacement indexes while sharing immutable payload segments. Pending
operations, tombstones, migration and untrusted OFF-mutated history retain the
full encoder. Cut validation, retained metrics, source owners, cancellation and
independent old-index retirement are preserved.

Expiration storage now uses an indexed heap with exact restore arrays. A
1,024-row allocation test drops construction peak from 101,696 to 32,768 bytes;
this is an allocation measurement, not a throughput result. Ordinary source/V1
logical accounting remains unchanged. V2-restored arrays keep separate funding
through empty reuse and append; replacement capacity is admitted before output.
Sparse chunk hints now carry dense positions and verify exact chunk/offset
ownership, preserving moved-slot repair and independent left/right restore.

The combined increment has 41 distinct passing focused tests and library Clippy
passes. Tests cover tree/heap behavioral equivalence, ordering, cancellation,
allocation refusal before emission, retry, sparse chunk repair, restored
continuation, rebase fallback and retained segment owners. Fixture errors are
retained separately from expected behavior REDs. No new release build or
performance samples are included.

The bounded-restore increment replaces historical row maps/sets with independently
validated upsert/tombstone cursors, merged by row identity and epoch. One pass
checks the complete history, locator bijection and live charge totals before
admitting the exact live-row vector. A second pass validates every historical
payload row, including deleted rows. Reader and per-row child workspaces now
retire after their allocations, while resident owners retain their funding.
Prepared-state installation remains the sole publication boundary.

The 21 selected tests pass, including a recorded 4,096-row capacity RED/GREEN,
interleaved and gapped histories, malformed locators, rich Arrow payloads, exact
alias lifetimes, all 56 restore admission refusals and all 611 cancellation
points. Failed restore preserves the old destination and refunds temporary
credit; retry succeeds. Library Clippy passes. Test-target Clippy retains only the known generated-libtest diagnostic;
its code, message and spans match the prior baseline. The frozen validation
manifest records all source hashes and exact commands.

The release build/audit at `cffcf57b` took 667.792 seconds. The unchanged 4M
checkpoint lifecycle passed under the original 1 GiB pools: all four pools had
zero failed grows and zero final reservations, and both restores validated every
row. Measured peaks were fresh 997,663,225 bytes, history 979,293,354, old restore
932,489,048 and new restore 932,453,624.

The source worksheets had omitted Arrow's all-valid IPC bitmaps from serialized
and restored backing. The correction preserves the original worksheets and
explains the measured restore peaks exactly. Conservative rebase/capture bounds
become 943,251,521 / 943,244,391 bytes; restore bounds become
932,493,947 / 932,456,163. Fresh/E1 constructor bounds are unchanged. These remain
pool-reservation bounds for the specified fixture; empty-destination restore has
separate source funding. They do not bound RSS or same-pool/live-state restore.

The follow-up collected complete retained OFF 1M/4M and capture/restore groups.
Fresh e8-to-cff processing p50 improved from 383.842 to 258.226 ms at 1M and
1512.430 to 900.200 ms at 4M; current scaling is 3.486x. However, both EOF sizes
and the 4M full-cycle comparison are regressions under the maintained paired
rule in this acquisition. Full-cycle 4M p50 rose from 1876.961 to 2381.441 ms
(+26.88%). Host isolation remains unverified; the regression requires diagnosis
and must not be hidden by the processing improvement.

Snapshot/restore cost is 0.221539 / 0.494471 us per total resident row, above
both targets. The compaction group exhausted its fixed 240-second allowance at
18/20 samples per arm. Its descriptive handler ratio is 0.985x, with no acceptance
verdict. The collector stopped at 566.758 / 900 seconds, all 52 child processes
settled, and no forced kill or retry occurred. All eight rows remain in the
[diagnosis](issue406-eight-workload-diagnosis.md); final-runtime acceptance is
still incomplete.

Two subsequent implementation increments are reviewed and committed. They have
no new release-performance measurements. `3b4ecafb` reuses owned ASOF descriptor
storage; `5ab1ac09` retires completely expired Join state in bulk when checkpoints
are disabled. Their focused correctness and ownership evidence is recorded in
the diagnosis. The cff terminal regression remains the measured result until a
fresh comparison demonstrates the effect of the correction.

Three later source increments remain unmeasured in release performance:
`16af86e4` prepares strictly ordered ASOF left input directly as columnar chunks;
`d4e66037` gathers eligible Join output columns on owned parallel workers;
`a681e7da` constructs each checkpoint payload segment once and reuses its
checksum for its inventory name and index reference. The ASOF path is restricted
to one inline Utf8 key and one UInt64 sequence, with an actual resident-state
ordering proof. Other inputs retain the validated path. Join parallel gathering
requires at least 64,000 output rows and six eligible columns; small or unsupported
outputs keep serial gathering. Neither restriction is a claim of speedup.

Ten ASOF and nine Join controls pass. They cover exact dictionary/order and
error behavior, projected nullable slices, checkpoint bytes and continuation,
funded worker cancellation/abandonment, actual concurrent gathering, early
reservation fallback, failed service admission and output credit across awaited
delivery. The direct ASOF task is eight bytes smaller on this build, charged
once by its wrapper and once by work admission. The intermediate credit differs
by 16 bytes; the later overall peak and successful state-install cutoff remain
unchanged. This does not claim a general 16-byte reduction or lower row funding.

Eleven checkpoint controls pass after a genuine base-path RED observed 3,264
bytes hashed for 1,632 payload bytes; cleanup completed before the assertion.
The final test exercises both base and pending encoders, checks the independent
segment SHA against metadata and decoded index references, and requires one
hash pass. Exact wire, funded allocations, alias retirement, refusal, cancellation
and restore controls pass. The implementation keeps `StateSegment::new` as the
checksum constructor and prepays segment/control ownership before constructing
it; it does not extend the trust contract of `from_validated`.

Library Clippy, scoped formatting, whitespace and generated-contract checks
pass. Test-target Clippy still fails with exactly the prior
`large_stack_arrays` compiler message; no lint is suppressed. Final independent
source and evidence review approves these scoped implementations, not merge
readiness or the performance targets.

The checkpoint CPU profile is complete. The duplicated payload digest represents
208 / 2,194 attributed preparation samples (9.48%), and is now removed at source.
Cancellation checking and row-level restore validation remain distinct proposals;
CPU percentages cannot substitute for latency measurements.

The next steps are:

1. Build and measure the complete eight-workload inventory on one common runtime,
   with separate OFF/LOW/ON policies, explicit pinned-main limitations and complete
   preregistered samples. Compare cff versus the new runtime for retained OFF 1M/4M,
   including the final watermark, terminal time and full lifecycle.
2. Use the resulting evidence to select remaining checkpoint work. Candidate
   changes are bounded writer cancellation checks and reused restore decoding
   setup/scratch. Preserve cancellation bounds, prepaid ownership, every obsolete
   historical row check, and both validation scans.
3. Reconcile all eight rows against the final runtime after further changes.
   Keep every missed target, invalid main comparison and incomplete attempt
   visible; do not extend the sealed cff compaction run or retry for a favorable
   classification.

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
