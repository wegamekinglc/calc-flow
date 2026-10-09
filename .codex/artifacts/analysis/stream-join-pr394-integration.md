# PR #394 review and selective integration into #393

Review anchor: [PR #394](https://github.com/wegamekinglc/calc-flow/pull/394)
at 8b9154103614bd3608d13dd30a45b2657af7ea73,
against main 25435770d41d3ab8e5cbc1367ed8c791400ed6fa. This is an
independent static source review; PR #394's tests and performance samples were
not rerun. Integration and its verification use the separate PR #393 worktree.
The frozen #394 head requires changes for the findings below. The selective
#393 integration passed independent source review.

## Sealed integration results

The cumulative #393 candidate improves all three selected Join cases against
current main under the maintained paired rule. Projection is inconclusive;
there is no confirmed regression in this selected set. This comparison does
not isolate Rowed's individual contribution, measure #394's frozen head,
establish the six-case acceptance aggregate or prove the original absolute
J2 planning targets. Required CI and coverage remain separate merge gates.

| Case                        | Main P50 ms | Candidate P50 ms | Change  | Maintained verdict |
|-----------------------------|-------------|------------------|---------|--------------------|
| Lookup 1M / batch 64000     | 1534.214    | 421.751          | -72.51% | improved           |
| Lookup 100k / batch 1024    | 219.907     | 148.045          | -32.68% | improved           |
| Retained interval Join 100k | 1131.119    | 418.814          | -62.97% | improved           |
| Projection 1M               | 9.583       | 9.163            | -4.38%  | inconclusive       |

Each case has two rounds of ten adjacent alternating AB/BA pairs, twenty timed
observations per side. Change above compares pooled P50 values; decisions use
the per-round paired median intervals below. Regression requires both lower
bounds above +5%; all three Join cases have both upper bounds below -5%.
Projection crosses +5% and remains inconclusive; it neither proves equivalence
nor excludes a regression.
The exact nominal 95% order-statistic interval has 97.85% coverage at ten
pairs under the iid assumption. This assumption was not independently verified
on the shared host, and interference is not excluded.

| Case                        | Round 1 paired change [CI] | Round 2 paired change [CI] |
|-----------------------------|----------------------------|----------------------------|
| Lookup 1M / batch 64000     | -72.86% [-75.52%, -71.12%] | -70.90% [-74.79%, -70.18%] |
| Lookup 100k / batch 1024    | -33.98% [-37.71%, -24.83%] | -35.88% [-39.10%, -30.75%] |
| Retained interval Join 100k | -64.08% [-65.96%, -62.83%] | -64.52% [-66.44%, -56.18%] |
| Projection 1M               | -7.15% [-13.59%, +5.24%]   | +1.93% [-16.18%, +9.30%]   |

All 160 timed observations passed maintained reference checks with maximum
absolute error zero. Lookup/projection reference results contain the declared
input row count; the interval case emits 1,098,080 matched rows for its
maintained N=100,000 parameter. Existing stream evidence validates replay,
limits and retained-state dimensions where the catalog requires them. Timing
covers ready-runner enqueue through Arrow output; runner startup/shutdown and
reference checking are outside each sample timer but inside the total budget.

Supervised measurement, including setup, installs, fixtures, warmup, checks,
statistics and owned-process cleanup, took 143.774 seconds
of 600. All four cases completed once; no retries, resampling or case
expansion occurred. Supervisor exit was zero, timeout false, and the final
owned process group was empty. No new CPU sampling was collected.

### Source and binary seals

- Main baseline: `25435770d41d3ab8e5cbc1367ed8c791400ed6fa`.
- Local candidate: `eff2fcad17b8922de6aa83955cc14ef602611a0b`.
- Published source-equivalent candidate: `bfbe6fbf68e8715e5b0e5c800396fc80e24dfd6e`.
- Both candidate commits have tree `ba005b744e10c3bf183aa64f646390f6b07fc1da`.
- Baseline native SHA-256: `bacfd960a97ce829bf5ab0b09d6b717ea3c6c1993db7c4b2aaec589d9d7ab612`.
- Candidate native SHA-256: `3c4b6d31a2c41d74abe155e98e99e02d5b1de0c28be5d4036986d7c108141428`.
- Candidate runtime source SHA-256: `840064b5c55aebc29eb4b5eafb5a3ec49668e78accb3a62f923e815c7d10f520`.
- Maintained harness SHA-256: `7dfd17584183b3ad7e2ae3aa8eab102551588f215faa84209be6d7c3c9ad3d5e`.

Both builds use clean exact-SHA sources, Rust 1.88.0, CPython 3.13.9 and four
Cargo jobs. Baseline build elapsed 1,066.155 seconds; the valid candidate build
elapsed 340.887 seconds. Including the rejected cached candidate and aborted
preflight, logged build attempts total 1,501.371 seconds, separate from the
measurement budget. The original 10–20 minute estimate was exceeded.

Workers verify their native hashes and matching machine/dependency/thread
fingerprints: WSL2 x86_64, 32 available logical CPUs, 32 Tokio/Polars threads,
BLAS/OpenMP one thread, NumPy 2.5.2, PyArrow 24.0.0, DataFusion 54.0.0 and
Polars 1.44.2. The user authorized running on the shared host without waiting
for unrelated work. Cross-session absolute timings are not compared.

### Resource observations

These are process lifetime RSS high-water marks through atexit, including
fixtures, warmup and native threads. There are only two receipts per side per
case. They are not twenty independent memory samples, timed-window live bytes
or allocation counts; no memory-growth or equivalence verdict is inferred.

| Case                        | Main HWM MiB range | Candidate HWM MiB range |
|-----------------------------|--------------------|-------------------------|
| Lookup 1M / batch 64000     | 765.7–779.6        | 574.7–596.8             |
| Lookup 100k / batch 1024    | 366.6–372.1        | 372.1–372.9             |
| Retained interval Join 100k | 1379.7–1450.2      | 787.1–808.1             |
| Projection 1M               | 505.7–506.6        | 503.8–511.2             |

The four cases do not time generic Rowed capture/restore. Earlier lifecycle
numbers concern the old candidate and baseline and cannot establish checkpoint
performance for this integration. Existing funded-native instrumentation tests
cover their focused cases; these release timings do not prove zero SQL calls
or absence of denied-credit fallback for every workload.

## Why the representation matters

The strongest idea is parent-backed row storage combined with a column gather.
The default multi-partition runtime does not satisfy serial_owned_sql, so the
funded owned-copy admission path does not run for the ordinary suite workload.
Previously, generic admission created a one-row RecordBatch wrapper for every
accepted row. Each wrapper also created per-column slice objects. A native key
dictionary alone leaves this work in place.

The earlier #393 profile also identified Float64 payloads outside the owned
copy whitelist. Payload type and serial runtime are separate eligibility gates:
widening that type whitelist alone would still leave the default runtime on
generic admission. Optimizing the generic path selected by these suite workloads is the
central lesson here.

RowPayload::Rowed stores one Arc<RecordBatch> parent per source record and a
physical offset per accepted row. Retention and dirty tracking can share those
references; one-row views are created only where the existing codec or fallback
actually needs them. This keeps Legacy's input-buffer sharing semantics rather
than enabling the separately proven funded-copy path for more runtimes.

When all output pairs on one side address the same parent, their physical
offsets can be collected once and applied to each column with Arrow take.
This avoids constructing one slice object per pair per column and then
concatenating them. Repeated offsets preserve fanout, and the existing matched
pair order defines output order. Mixed parents and types whose backing contract
has not been proved continue through the existing concatenation path.

PR #394 reports approximately 88% lower Lookup duration after this combination.
That is useful investigation evidence, not a measurement of the integrated
#393 candidate. Its shared-host results, shorter retained-state rounds,
best-of microbenchmarks and pooled confidence intervals do not replace the
maintained paired acceptance rule. A historical machine-speed ratio cannot
establish an absolute 60 ms result on another host; duration reduction and
throughput increase are different quantities.

## Review findings for the frozen #394 head

1. **P1: partial hot-key eviction underfunds retained capacity.**
   [NativeIndex::remove](https://github.com/wegamekinglc/calc-flow/blob/8b9154103614bd3608d13dd30a45b2657af7ea73/crates/calc-flow/src/operator/join/native_lookup.rs#L223) refunds 192 bytes per row,
   but a nonempty run retains its Vec capacity. A 4096-entry run has at least
   98,304 bytes of entry backing after deletion down to one row, against only
   1,216 bytes of index credit. Unique-key eviction tests do not cover this.
   Keep #393's actual-capacity funding and hot-run allocation controls.
2. **P1: nested Dictionary bypasses output protection.**
   [materialize_output_column](https://github.com/wegamekinglc/calc-flow/blob/8b9154103614bd3608d13dd30a45b2657af7ea73/crates/calc-flow/src/operator/join.rs#L5459) excludes only a top-level
   Dictionary. A Struct/List containing Dictionary can still use take, whose
   dictionary child retains all values. A singleton match with a large unused
   dictionary value can fail an edge byte budget that the original
   concat-with-empty path satisfied. Use a proven flat-type allowlist and
   retain the original nested fallback.
3. **P2: whole-record row-ID reservation changes error precedence.**
   [admit_legacy_record](https://github.com/wegamekinglc/calc-flow/blob/8b9154103614bd3608d13dd30a45b2657af7ea73/crates/calc-flow/src/operator/join.rs#L3937) reserves every physical row before
   classifying the first. With next_row_id = u64::MAX - 1, two physical rows,
   and an overflowing non-null timestamp in the first row, the old path
   returns the conversion error; the new path returns row-ID exhaustion.
   #393 must keep reservation at each physical row's consumption point.
4. **P2: canonical key owners pin an expired batch's reservation.**
   [KeyDictionary::intern](https://github.com/wegamekinglc/calc-flow/blob/8b9154103614bd3608d13dd30a45b2657af7ea73/crates/calc-flow/src/operator/join/native_lookup.rs#L84) retains the first batch's
   canonical Arc until the entire key disappears. If a later batch keeps the
   key alive after all first-batch row and dirty owners expire, the dictionary
   still pins that first batch's complete probe reservation. Keep #393's
   surviving-row owner refresh and both expiration-order regressions.
5. **P2: hot-run prefix eviction is quadratic.**
   native_lookup.rs:223 repeatedly removes the first Vec entry during ordered
   watermark expiry, moving N(N-1)/2 entries overall. The unique-key eviction
   benchmark misses this shape. Keep #393's head cursor and geometric
   compaction rather than importing this dictionary implementation.
6. **P2: unchecked u32 narrowing can panic at valid public sizes.**
   NativeIndex::append in native_lookup.rs:199 and output gather in
   join.rs:5432 use expect when narrowing dense retained-row positions and parent-local
   physical offsets, respectively, to u32. Public row limits
   are wider. #393 keeps dense usize indices, checks dictionary-ID capacity
   before native allocation/fallback, and uses UInt64 gather offsets.

The dictionary-deletion fix in #394 is useful, but #393 already has collision,
slot-reuse and relocation controls. #394 also still encodes every probe row
into a reused byte buffer; it allocates canonical owners per distinct key.
These are separate costs. Keep #393's borrowed-key interner rather than
describing every-row encoding as distinct-only encoding.

Do not directly replace #393's hasher with #394's DataFusion RandomState:
DataFusion 54 aliases FoldHash FixedState, whose write-call protocol is not
interchangeable with #393's segmented borrowed-key hashing and whole-frame
stored-key hashing. Both sides would need one proved protocol, including
compound framing, all key types and collision tests.

The earlier concern about Legacy-only metadata tests was withdrawn after
tracing the actual fallback: owned_payload returning None still reaches
append_copy_rows(None), which deliberately remains Legacy. Those exact
metadata/IPC/funding assertions must remain intact.

## Integration decisions

Main's V2 checkpoint reader is merged first. Keep #393's native dictionary,
borrowed interner, capacity/peak funding, key-owner refresh, 64-row admission
masks, per-row error order and Quantum cancellation/fairness work.

Only generic admit_legacy_record introduces parent-backed rows; an owned-copy
construction refusal still uses its original Legacy fallback. Use a separate
parent RowView so Shared and RestoredV2 retain their existing typed owners.
Plan each output side once by shared columns identity, using UInt64 offsets.
Take applies only to independently compact flat types. Dictionary, nested,
view, run-end and mixed-parent outputs keep the established concatenation and
canonical timestamp handling. This is an internal representation change with
no public schema, checkpoint-version or runtime-limit change.

## Verification

Observed focused RED before implementation: generic admission produced a
one-row column instead of the three-row parent; output gather produced 36
column views instead of zero. The final serial Join module passed 203 tests
in 4.40 seconds, and the property suite passed four in 1.69 seconds. These
include parent/physical-offset identity, exact row IPC, input immutability,
nulls and sliced inputs, repeated offsets, incoming-side order, mixed parents,
nested Dictionary byte budgets, offsets beyond u32, V1/V2 restore,
credit/refund, cancellation and error precedence.

An initial NullArray test used physical null_count instead of the logical
null-count API; correcting that test left production logic unchanged. Both
initial and final logs are preserved. Scoped Clippy with warnings denied,
formatting, whitespace and unchanged generated-contract checks passed. Actual
Lizard analysis of changed functions found maximum CCN 8 and no violations.
Independent specialist source review approved the sealed candidate. Full
workspace CI and coverage remain separate gates.

The performance comparison measures cumulative #393 against current main; it
does not isolate each optimization or compare against the frozen #394 head.
It uses clean current-main and final-candidate release wheels, the maintained workload/oracle/worker harness, two rounds of ten
adjacent AB/BA pairs, and four fixed cases: Lookup 1M/batch 64000, Lookup
100k/batch 1024, retained interval Join 100k, and projection 1M. Measurement
budget is 600 seconds, with builds separate. It includes fixture construction,
install/startup, warmup, correctness, resource collection and cleanup. RSS is
worker lifetime through atexit, with only two resource receipts per side per
case, not twenty independent memory samples, timed live bytes or allocation
counts.
Shared-host interference is not excluded; no favorable resampling or case
expansion is authorized by this plan. The four selected cases do not measure
generic Rowed capture/restore; earlier lifecycle results do not establish
checkpoint performance for this integration.

## Build-cache rejection before timing

The first attempt used one owned Cargo target directory for both frozen
worktrees. The candidate build reused baseline workspace artifacts: its core
dep-info lacked the candidate-only modules and both wheels had the same native
SHA-256. No measurement was started with those wheels. The invalid candidate
wheel, manifests, logs and fingerprints are preserved separately.

The owned release artifacts and fingerprints for calc-flow, calc-flow-python
and calc-flow-connectors were removed. A preflight rejected incidental
Cargo.lock ordering drift caused by clean; the exact frozen lockfile was
restored before rebuilding. The replacement build compiled all three workspace crates from the candidate,
included the new modules in core dep-info, passed source-before/after identity
checks and produced a distinct native hash. Independent source-chain review
approved these artifacts before measurement began.
Relative dep-info names alone are not source-location proof; the actual
compilation paths and build cwd bind them to this candidate. The maintained
build recipe and measurement harness remain unchanged.

## Similar opportunities elsewhere

Existing evidence identifies a separate Join checkpoint cost: the earlier
[9db4cd25 profiles](stream-join-j2a-dictionary-measurements.md) attributed about 59-61% inclusive CPU samples in
capture/compaction to row IPC encoding, and restore to IPC parsing/validation.
These overlapping inclusive CPU shares come from 9db4's V1 workloads; they
are not wall-time shares or measurements of current main. Main's new V2 reader
does not change V1 capture. Batch-oriented encoding may
help, but replacing frozen V1 row bytes requires its own format/migration
design and compatibility evidence; this integration does not make that claim.

The following are source-grounded hypotheses, not measured improvements:

1. **Rolling checkpoint:** [rolling.rs:1530](../../../crates/calc-flow/src/operator/rolling.rs#L1530) expands columnar state to scalar
   rows, then [rolling/state_v3.rs:341](../../../crates/calc-flow/src/operator/rolling/state_v3.rs#L341) reconstructs Arrow arrays. Gather directly
   from parent batches and physical positions. This also requires retaining
   parent/offset representation and ownership for history and open buffers;
   changing the codec alone cannot remove the earlier scalar expansion.
   Keep v3 schema, ordering,
   projections, EWMA seeds and transition counters. Fund state, gather and IPC
   overlap. Start with 640 rows/16 entities, retained history plus open buffer,
   and verify restore/resume.
2. **Cross-section generic payloads:** [cross_section.rs:2082](../../../crates/calc-flow/src/operator/cross_section.rs#L2082) scalarizes every
   cell and build_grouped_record at :2125 reconstructs all input columns.
   Keep typed identity plus parent/offset payloads and read only calculation
   operands. Preserve group/tie/null/NaN ordering, duplicates, failure rollback
   and real parent backing charges. Start with mean/residual/top/bottom and a
   Utf8 payload column.
3. **Rolling generic fallback:** [rolling/state_codec.rs:1007](../../../crates/calc-flow/src/operator/rolling/state_codec.rs#L1007) reads all payload
   cells and :1084 rebuilds input arrays. Target out-of-order input or other input/state combinations ineligible for
   try_buffer_ordered; the eligible ordered path already avoids this. Retain projected history and gather
   buffered parents, preserving lateness/duplicate precedence, numerical
   profiles and failed-emit retry state. Start with 640 shuffled rows and wide
   strings, including watermark and checkpoint continuation.
4. **ASOF mixed late input:** [asof/admission.rs:232](../../../crates/calc-flow/src/operator/asof/admission.rs#L232) slices and rebuilds identity
   encodings for each accepted run, which can become one row when late and
   accepted arrivals alternate. A selection mask and packed key/sequence
   gather can amortize this while retaining physical-to-packed mapping.
   Preserve time-equals-watermark acceptance, null-identity errors before
   late drops, and overlapping allocation funding. Start with the 129-row
   differential boundary and one 1024-row, 50%-late admission case.
5. **Late-output preflight:** [late_output/plan.rs:158](../../../crates/calc-flow/src/operator/late_output/plan.rs#L158) wraps each row for charging;
   :167 constructs diagnostic data discarded before emit constructs it again.
   First consider budgeted charging views and constant diagnostic costs.
   Preserve the exact charge formula, including the existing 177/93-byte
   fixtures, all-row preflight before first emit, diagnostic order and
   bounded scratch; do not batch outputs without reviewing the byte
   contract. Start with 512 late rows, null/Utf8 and multiple records.

ASOF's normal output already uses parent offsets and GatherPlan::column
([asof/output.rs:422](../../../crates/calc-flow/src/operator/asof/output.rs#L422)) with spans and backing-size guards. Reuse that design and
GatherScope::submit ([runtime/streaming/gather_work.rs:661](../../../crates/calc-flow/src/runtime/streaming/gather_work.rs#L661)) for paid, cancellable
work with retirement ownership. Runtime/sink is not a blanket candidate:
sinks already write batches, and TableBatch::estimated_bytes is cached.

## Portable evidence

The [lossless evidence package](stream-join-pr394-integration-evidence.json.gz)
contains the complete measurement plan/results and raw observations, per-worker
RSS receipts and stderr, release manifests/build logs, rejected cache artifacts'
metadata, dependency/fingerprint diagnosis, exact scripts, RED/GREEN/lint logs,
actual complexity audit and independent review receipts. Wheel binaries remain
local; their wheel/native hashes and source seals are preserved.

Compressed bytes: 60,911; SHA-256:
`a7be8dd0978b496772c5848be4b8940786c74eda9353b0f18185e0d2d38d7c07`.
Decompressed JSON bytes: 815,527; SHA-256:
`40de4ef68ba5fb1cd41ba2c7ab2519d988952230ebf02c6735ba7434db71a384`. The package contains 80 exact text files
with per-file sizes and SHA-256 hashes. The earlier nine-case package remains
unchanged and describes its own frozen candidate.
