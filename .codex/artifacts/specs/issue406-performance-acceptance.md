# Issue 406: complete performance acceptance protocol

Status: all five sealed release builds passed independent provenance review.
The first real-job correctness preparation exposed a Python harness ownership
error and stopped before acceptance sampling. The focused correction and one
corrected fixed preparation are tracked below. No performance measurements under
this protocol have started.
This replaces the never-executed seven-workload final protocol. Earlier sustained
measurements remain historical evidence in
`../analysis/issue406-sustained-performance.md`.

## Required target rows

The complete table supplied for this task is authoritative. Every final report
must contain all eight rows, fresh exact-main observations for every supported
cell, current-candidate observations, validity, and an explicit target result.
An unsupported mode is recorded as unsupported rather than silently dropping
the corresponding revision or row.

| ID | Workload                                   | Historical reference | Single-CPU target             | Parallel target |
|----|--------------------------------------------|----------------------|-------------------------------|-----------------|
| 1  | Sealed-dimension StreamJoin, 1M quotes     | 701 ms               | ≤60 ms                        | ≤15 ms          |
| 2  | Sealed-dimension StreamJoin, 10M quotes    | 8,061 ms             | ≤600 ms                       | ≤150 ms         |
| 3  | Retained StreamJoin, 1M→4M resident quotes | 5.7×                 | ≤4.4×                         | —               |
| 4  | Join capture / restore per row             | 1.8 / 2.3 µs         | ≤0.1 / 0.2 µs                 | —               |
| 5  | Join handler latency with compaction       | 40× steady           | ≤2× steady                    | —               |
| 6  | StreamASOF, 1M rows per side               | 391 ms               | ≤100 ms                       | ≤30 ms          |
| 7  | ASOF harness overhead, 1024-row batches    | 167 ms over operator | ≤2× measured projection floor | —               |
| 8  | Join operator cost, 1024-row batch         | 0.2–0.9 ms           | ≤0.05 ms                      | —               |

The historical references are from issue #363, not fresh main measurements.
Its 5.7× row retained every quote; neither bounded-window traffic scaling nor a
no-op watermark kernel is an equivalent replacement. Its row-7 historical
measurement used N=100k (100k per ASOF side): 203 ms suite minus 35.8 ms native, with a
34.3 ms projection floor. The parenthetical approximately 70 ms was twice that
floor. The new primary row-7 measurement uses 1M rows per side, as required by
the current table's default. Report both the newly measured `2 × floor` and the
historical approximately 70 ms reference; do not call 70 ms the floor or
silently substitute a 100k primary workload. No extra historical-size run is
selected.

## Revisions, builds and compatibility

- Main is `588726c8bce0f1c57b90077064834c4a472ee5f2`, confirmed by a fresh fetch.
- Candidate production commit is `e8a28d9979ff3132a571523918a12ffec0bffdae`.
  The benchmark helper may be a later commit with explicitly audited default
  production equivalence. Preserve the actual release build revision and fill
  its release-manifest digest before execution.
- Exact production release wheels remain separate from the common native
  measurement harness. Row 7 uses a separately sealed feature-enabled wheel for
  its ASOF and projection controls in every mode; the other rows use the audited
  production wheel. Record product revision, harness revision, original and
  effective workload digests, Cargo/Python dependency locks, build flags,
  executable/wheel hashes, and environment fingerprints.
- Reuse a sealed release only after auditing those settings and installed native
  identity. Do not import a potentially stale source-tree native module.
- A common harness may add a narrowly scoped benchmark entry point and helper;
  it must not replace or copy operator implementations, patch exact-main product
  behavior, or present a `cfg(test)` runtime as the production runtime. Explicitly
  record all helper/module/manifest overlays and their reviewed source digests.
- Classify a pair only if its machine, dependencies, effective workload,
  checkpoint semantics, thread configuration, and timing boundary match. Main
  checkpoint-off is unsupported; candidate-off absolute observations therefore
  have no invented exact-main off comparator.

The original build harness is `a2f31da023dccb00518d2e043aa628d663aa93fb`.
A later Python-only ownership correction captures the plan fingerprint before
the runner consumes that plan. Reuse retains the original build SHA and a
separate execution-helper SHA. It requires identical historical overlay-producing
source, identical copied native files, and reproduction of every sealed effective
source hash; historical Python is never executed. The frozen plan carries this
explicit reuse witness.

Builds are outside the measurement budget, with duration and artifact reuse
reported separately. The three native harness variants and two instrumented
row-7 wheels have a combined estimate of 25–35 minutes, with compatible build
artifacts reused. This estimate does not extend the 75-minute measurement budget.

## Modes and CPU configuration

Runtime cases 1, 2, 3, 6, and 7 use three separately identified candidate modes:

- `off`: checkpointing is actually disabled; no periodic or terminal durable
  checkpoint and no state files are permitted.
- `low`: checkpointing enabled, 24-hour periodic interval. Bookkeeping and the
  terminal checkpoint remain present; this is not checkpoint-off.
- `on`: checkpointing enabled, 100 ms interval. Record actual nonterminal epochs,
  completion/retirement, and bytes. A finite run that completes before its first
  periodic checkpoint remains an on-configured observation with zero periodic
  coverage. Do not extend, sleep, or force an epoch to make it qualify.

Exact main is measured in both supported enabled modes. A main EOF-checkpoint
failure makes its full lifecycle invalid; preserve processing timings as
diagnostics with the error, cleanup outcome, and mode, never as a valid lifecycle
gate. Candidate off remains independently measurable against its absolute target.
The parent froze these new finite-workload planning goals before execution:
off uses the table's absolute targets; low throughput must be at least 90% of
matching candidate off (`T_low <= T_off / 0.90`); on throughput must be at least
75% of matching candidate off (`T_on <= T_off / 0.75`). These are not transplanted
sustained-job acceptance results. Enabled modes also require valid terminal
durability. On requires completed advancing nonterminal checkpoint evidence
inside processing before its periodic-performance goal can pass. Zero-epoch
runs report configuration overhead and unmeasured periodic impact. Report EOF
and full-cycle cost separately. Row 7 off/low additionally uses the matched
residual target below; on uses its 75% throughput goal and checkpoint costs,
with no on-minus-native residual. Rows 4/5 retain their intrinsic enabled
component targets, and row 8 distinguishes off/enabled direct contexts.

Capture/restore and the compaction trigger are intrinsically checkpoint-enabled
native component cases. Their direct callback contexts have no periodic interval;
report `enabled component`, not fictional separate low/on samples. Off compaction
and off capture/restore are not applicable.

Rows 1, 2, and 6 have single-CPU and parallel configurations. Other target rows
use single-CPU. Before any native/library import, fix child CPU affinity to one
declared CPU for single-CPU and the same declared 32-CPU set for parallel; set
Tokio workers to 1 or 32 respectively and OpenMP/BLAS/MKL/Polars threads to 1.
Record effective affinity, limits, environment, and observed thread CPU. The
gather pool derives its size from available parallelism; thread creation alone
does not prove parallel work. Preserve the production parallel dispatch threshold
and describe any actual serial fallback. Compare revisions only within the same
configuration; single/parallel workload identities differ.

## Row-to-harness mapping and exact clocks

### Rows 1 and 2: sealed static lookup

Reuse `benchmarks.engine_comparison.workload`, `engine_stream.stream_plan`,
`stream_dimension`, and the maintained static Join ready timer. Inputs have 64
string keys, required Arrow fields, deterministic exact-binary-fraction prices,
and 64,000-row quote batches. Preload the 64-row dimension and causally observe
its sealing watermark before starting the timer. The timed scope starts with
prepared immutable input at a ready job, includes quote enqueue, real operator
tasks, sink delivery, and Arrow concatenation, and ends with all N output rows.
Plan construction, preload, dedicated retention proofs, oracle, EOF, and cleanup
remain outside processing; startup, full-cycle, terminal and cleanup clocks are
recorded separately. The common 10 ms checkpoint observer runs inside every
mode's elapsed processing interval. The post-timer proof requires zero retained/evicted left rows and exactly
N emitted matches. Validate every sequence/value, not only row count.

Use N=1,000,000 and N=10,000,000 respectively. Every source/sink/worker is owned;
terminal and cleanup results remain mandatory even though excluded from this
processing clock. The ordinary scheduled catalog is unchanged.

### Row 3: whole retained-stream scaling

Use the same 64-key static lookup, output, and 64,000-row batches, but deliberately
leave the dimension watermark unsealed until after all quote data has completed.
Raise only this explicit retained case's row/byte limits to admit N quote rows
plus the dimension. The source must not become idle or ended during the timed
phase. Preload dimension data and prove its admission without sealing it.

For N=1,000,000 and N=4,000,000, measure the entire ready prepared-input through
N-row Arrow output boundary. Outside that clock, causally prove all N quotes
remain resident, zero quote eviction, observed positive retained byte charges, and
unchanged dimension cardinality; then advance progress, drain, and settle the
job. Validate every output row and the post-drain state. Use bounded oracle
storage; one active large-state process at a time. The primary ratio is candidate
p50 at 4M divided by candidate p50 at 1M, separately for each runtime mode. Also
report exact-main ratios and paired per-size revision deltas when valid.

Byte charges are recorded from native status; no independent byte recount is claimed.

The earlier proposed ±5-second interval fixture is expressly withdrawn: it
does not retain 1M or 4M quote rows. The existing retained no-op/sparse progress
kernel is supporting coverage, not this target's measurement.

### Row 4: capture and restore

Extend `crates/calc-flow/benches/stream_join_perf.rs` with a declared 1,000,000-row
resident Join snapshot fixture, split deterministically across both inputs. The
capture arm starts after admission with all those rows dirty and times all
required dirty-snapshot materialization: `prepare_checkpoint_async` followed by
`checkpoint` returning the completed immutable snapshot. There may be no earlier
preparation of the same cut. The ≤0.1 µs target uses this complete total divided
by actual captured dirty rows. Report preparation and capture subclocks; timing
only the final call would hide serialization moved into preparation. Fixture
creation and durable file staging/publication have separate clocks.

The restore arm starts from an already created, validated snapshot and a fresh
operator, and times restore to usable state; normalize by actual restored live
rows. After timing, verify status, all snapshot segment/checksum references, and
output equivalence using deterministic matching probes. Record state bytes, RSS,
dirty/live counts, snapshot format, and the checkpoint-enabled context. Preserve
the maintained smaller cases; the new task selects the explicit 1M cases only.

### Row 5: actual handler latency around compaction

Extend the maintained armed-versus-steady Join fixture to 1,000,000 resident rows
and a 500-row incoming batch. Prepare identical retained state and checkpoint
history except the compaction trigger. Prove the armed branch actually performs
compaction and the control does not. Measure `prepare_checkpoint_async` separately
from the subsequent `process_data` handler. The target is the actual 500-row data
handler's compacted/steady p50 ratio; a prepare-plus-handler duration cannot be
renamed handler latency. Preserve per-sample clocks and p95 for both phases.

Report the combined unavoidable service boundary as additional context when
available, clearly separate from the target. Validate identical outputs, state,
checkpoint lineage, and restore correctness. The historical 60k/500-row 40×
number remains a historical reference, not a fresh comparator. This controlled
component experiment does not establish live-job queue latency or a sustained
checkpoint-throughput acceptance result.

### Row 6: ASOF end to end

Use the maintained `asof_join` EngineCase at 1,000,000 rows per side and 64,000-row
batches. Both inputs use `QUOTE_FIELDS`: required UTC microsecond time, uint64
sequence, string symbol, and float64 price; tolerance is zero. Output is
`sequence=left__sequence, value=right__price`. Preserve the production compiled
projection and ASOF limits. Use the maintained per-chunk input/watermark schedule
and sink-event rendezvous, without per-chunk status polling. The common 10 ms
checkpoint observer remains inside every mode's processing interval.
Validate all values and post-timer retention/progress; record terminal separately.

### Row 7: ASOF instrumented harness overhead

Use row 6's exact fixture at 1M rows per side, 1024-row batches, and the corresponding
ASOF state limits. Measure the same-size/batch `projection` case as the runtime
floor, in each runtime mode. The projection is an explicitly different necessary
control; its fingerprint cannot be substituted for ASOF's workload fingerprint.

The native control consumes the exact exported project, sealed Arrow inputs,
and actual ASOF ingress dispatch trace from its corresponding end-to-end
observation. Submission order does not prove actual dispatch order. Seal the
shared partial-order identity and actual dispatch-trace envelope separately.
A narrow benchmark-only
crate-internal helper compiles through the ordinary runtime and obtains its actual
compiled operators, preserving compile-time ASOF output projection and downstream
select. Drive data, ingress progress, resulting watermark/idle transitions, and
the relevant finalization through those operators; measure the complete native
operator chain in one non-overlapping wall interval. Match each output and
post-run state to the Python case and settle owned gather work. An opt-in
recorder allocates a fixed 8192-record buffer after ready/empty admission and
before processing. Both clocks retain the recorder's per-dispatch hook cost;
allocation and trace serialization have separate untimed phase costs. Record
actual data, ingress-progress and watermark callback counts, per-node aggregate
input/output batch and row counts, ingress batch ordinals, rows, sequence bounds,
exact ingress/final-output record-batch geometry, completed frontiers, and the
full ASOF post-processing state. Reject overflow, unsupported events, malformed
geometry and terminal events in the processing trace. Off/low forbid barriers;
on may record them but supplies no native residual.

The primary target is candidate-off end-to-end minus matched native-off work.
The suite retains ready-to-Arrow time and separately measures Arrow-to-completed
final output projection watermark. Require both the ASOF published frontier and
the terminal projection's completed watermark to reach the minimum input
watermark minus one microsecond. Poll status once immediately and then at a fixed
1 ms interval, only for this final rendezvous. The projection marker advances
after its full watermark dispatch returns. The primary end-to-end time is the
sum of both clocks, including the final polling overhead. Native processing ends
after the corresponding last non-EOF callback/progress transition and chunk
assembly return. Processing, EOF/on-end, and cleanup have separate corresponding
clocks in both arms. No gather-owner teardown belongs in processing.
A distinct terminal clock cannot be subtracted from a processing-only clock.
After that cutoff and before EOF, take and serialize the trace, then seal an
envelope containing its source observation ID, raw observation SHA-256 and
matching workload identity. Native replay verifies the exact source trace,
actual callbacks, data counts, ingress/final-output geometry, outputs and state.
The Python report independently compares these returned native witness fields
with the sealed source trace. It validates all 22
source/replay links per arm, including the two warmups, before computing timing
medians from the 20 measured observations. A missing or mismatched replay keeps
the target inconclusive.

The accepted claim is an instrumented matched-component residual with the same
recorded ASOF ingress/callback chain, exact ingress/final-output batch geometry,
per-node aggregate data counts, and ASOF state. It does
not establish identical global task scheduling, identical CPU instructions or
same-job causal CPU attribution, and instrumentation cost is not subtracted.
Pure input expressions and FIFO downstream projection preserve the relevant
transform ordering. A separate correctness-only preflight covers all 36
combinations of six legal two-source FIFO interleavings across two rendezvoused
chunks and a final partial chunk. Its observable output/progress/state evidence
is auxiliary and cannot independently establish row-7 acceptance.

Final correctness preparation also runs one real instrumented PyO3 job and its
native trace replay in each of baseline/low, candidate/off, and candidate/low.
Each fixed fixture has 3073 rows per side in three full 1024-row batches and one
partial row. Reuse the worker preparation warmup as that arm's sole real job,
close its owned worker, then replay the sealed fixture and actual trace using
native `--check`. Require the same complete oracle, state, geometry, trace and
cleanup witnesses as the measured path. Preserve any baseline terminal failure
separately. These three jobs and three native replays are correctness evidence
only and never enter performance statistics. Retain the three 36-schedule
auxiliary proofs too; all preparation time, startup, validation and cleanup is
precharged against the same 4500-second budget. The fixed correctness execution
timeout remains 60 seconds with no unchanged retries; artifact validation and
final sealing are also charged to the total budget.

The first preparation failed after its baseline/low job because the Python
driver read the compiled plan fingerprint after transferring plan ownership.
It yielded no accepted observations or verified replay. All owned processes
settled without forced kills. Preserve that attempt and charge its full
7.572431079 seconds plus a conservative 5-second failure-sealing allowance to
the same 4500-second budget. After the focused regression test and ownership
correction pass, run one corrected preparation with the original three real
3073-row jobs/replays and three auxiliary proofs. The performance case inventory,
sample counts and total budget are unchanged; this correction cannot discard or
replace an unfavorable performance sample.

A fresh main low-mode residual is permitted as a separately labeled diagnostic
when both arms genuinely enable checkpoint bookkeeping and prove zero
nonterminal epochs within the common processing interval. Keep the 24-hour mode
labeled low and preserve its separately timed terminal checkpoint failure if
present. It supplies neither a main-off observation nor valid lifecycle
acceptance. Apply the same enabled-context control to candidate low. For on mode,
report end-to-end duration, actual checkpoint coverage/cost, and its descriptive
difference from the corresponding off end-to-end case. Do not subtract a
checkpoint-free direct operator run and call durability work harness overhead.

Compute the independently collected suite p50 minus native operator-chain p50,
retaining the signed difference and both component median intervals. Matching
list positions do not define timing pairs, and no paired residual interval is
claimed. Compare
the observed overhead with twice the currently measured projection floor and
report both quantities. The projection floor retains its maintained
ready-to-Arrow scope because it lacks an equivalent exposed callback-completion
witness; identify this smaller scope as a conservative floor. Negative or
incompatible residuals cannot become a
zero-clamped pass. Total ASOF/projection ratio is not this acceptance metric.

Existing cumulative Python operator timers omit `on_ingress_progress` and can
overlap fused work; they cannot supply this subtraction. The old 100k Int64-key
`stream_asof_e2e` timer also differs in schema and physical output, so it cannot
supply the native control. Preserve its existing four-case acceptance contract;
add a separately identified opt-in matched case rather than silently changing it.

### Row 8: small-batch Join operator cost

Add explicit 1024-row callback sampling to the maintained native Join harness,
using the row-1 lookup shape and physical output contract. Preload and seal the
dimension outside the timer, then drive a fixed 1M quote stream. Record actual
`process_data` durations for every full 1024-row batch, retaining the final partial
batch separately. Fixture creation, checkpoint preparation, oracle, and cleanup
are outside each handler timer and still count toward wall budget. No division
of a 10k-batch duration or subtraction of a projection proxy is acceptable.

Each independent observation reports its median/p95 full-batch callback duration;
the target uses the median of those observation medians, not pooled batches as
independent statistical repetitions. Verify exact outputs and zero left retention.
Use actual off versus enabled callback contexts where the native helper can set
them; exact-main off remains unsupported. Direct contexts have no periodic low/on
interval and must be labeled accordingly.

## Fixed samples, ordering, statistics and budget

The proposed broad-task budget is 75 minutes across all selected runtime and
native workloads, including fixture construction, source export, preflight,
warmup, native component startup, restore, oracle, lifecycle, analysis, and
settlement. Builds are separate. Reserve the final five minutes for cleanup and
sealing; stop starting measurements at minute 70. This replaces the withdrawn
240-second stage budget and requires explicit parent/reviewer freeze before GO.
The estimate is driven by 10M main observations, two retained sizes, all runtime
modes, and fixed paired sampling; there is no all-scale sweep.

Every compatible revision comparison uses two rounds of ten alternating
baseline/candidate pairs, with one excluded warmup per arm per round. Native
controls and candidate-only absolute arms also have exactly 20 independent
observations plus those two warmups. Reuse a sample across multiple reported
metrics only when its row/round/index and shared-observation relationship are
explicit. Report every sample, including failed attempts; no favorable retries,
calibration-driven sample expansion, minimum reduction, or optional extra size.

The executable inventory contains 38 semantic workloads and 64 supported metric
arms: 10 arms each for rows 1, 2, 3, and 6; four each for rows 4 and 5; 13 for row
7; and three for row 8. That produces 1,280 measured metric observations and 128
excluded metric warmups. Capture/restore share one physical invocation, and
armed/steady compaction handlers share one physical invocation. Their records
must retain identical `observation_id` and both `group_member_ids`; they are not
independent samples. Thus there are 36 physical workload groups, 60 physical
revision arms, **1,200 measured invocations plus 120 excluded warmups**. The emitted
plan derives both semantic and physical counts from the executable inventory.
It separately records three real tiny PyO3 correctness jobs, three native actual
trace replays and three auxiliary ordering-proof processes; none is a measured
or warmup performance observation.

The following are wall-budget envelopes, not performance predictions. They
already include each family's fixtures, source export, process startup, warmups,
preflight, full oracle, terminal checkpoint, manifest inspection and state
cleanup. The historical timing source is issue #363 (recorded locally in
`target/issue406-acceptance-issue363.md`), supplemented by the historical common
v5 harness table in `stream-join-asof-performance.md`; none of those observations
is a compatible new baseline or a substitute for the fresh measurements.

| Family                                       | Physical invocations, including warmups | Total envelope | Historical basis and uncertainty                                                                                                                                                                                                            |
|----------------------------------------------|-----------------------------------------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Rows 1/6, 1M sealed Join and ASOF            | 440                                     | 10 min         | Historical processing about 0.70 s/0.39 s per observation; the envelope includes full runtime and fixture costs. Strict one-CPU and checkpoint policies are new scopes.                                                                     |
| Row 2, 10M sealed Join                       | 220                                     | 18 min         | Historical main processing 8.061 s; its 88 baseline invocations alone imply about 11.8 min. Remaining time assumes candidate improvement and covers preparation/EOF/oracle; this is a material budget risk.                                 |
| Row 3, 1M/4M fully retained Join             | 220                                     | 18 min         | Historical retained processing 1.311 s/7.409 s. 44 baseline invocations at each size imply about 6.4 min before checkpoint and cleanup. 4M EOF, snapshot, and memory costs remain uncertain.                                                |
| Rows 4/5, paired native state components     | 88                                      | 11.5 min       | Historical capture/restore 1.8/2.3 µs per row suggests about 4.1 s for 1M combined timed work; admission, compaction arms, oracle, restore probes and state disposal are additional. Linear size extrapolation is only a planning estimate. |
| Rows 7/8, small batches and matched controls | 352                                     | 7.5 min        | Historical 100k/1024 ASOF/projection about 0.085/0.031 s suggests order-of-seconds 1M work; exact matched native setup and 977 callbacks add unmeasured costs.                                                                              |
| Evidence analysis and sealing                | —                                       | 5 min          | Fixed inventory, provenance, paired summaries, target table and artifact hashing.                                                                                                                                                           |
| Final owned cleanup reserve                  | —                                       | 5 min          | Stop starting work at minute 70 and settle owned processes.                                                                                                                                                                                 |

The envelopes total 75 minutes. No large performance pilot was used to choose
them. The fixed 3073-row real-job/replay checks and 2049-row ordering fixtures are
preparation evidence only, with no performance result; allow one minute of the
fixture/preflight envelopes for this final sealed-release preparation.
Native builds are estimated separately. The historical data lack complete
comparable EOF/setup costs, so completion within the envelope is uncertain.
All actual work charges the same hard deadline; no family allowance is an
independent extension. Stop and report incomplete rows if measured work exceeds
it. Cases and samples remain fixed. No measurement starts while a target lacks
its executable reviewed harness, including row 7.

Only baseline/candidate observations from the same configured workload are
true alternating revision pairs. Cross-mode groups, the two row-3 sizes, and
row-7 E2E/native/projection groups are separate collections: matching list
indices do not make them temporal pairs. Derive mode throughput and scaling as
ratios of group p50 values and row-7 residual as the signed difference of group
p50 values; retain each component's p50/p95 and median uncertainty separately.
Do not claim a paired confidence interval for these cross-group quantities.
Row-5 armed/steady values share one invocation; label any within-invocation
ratio separately from baseline/candidate revision pairing. Short ON observations
with no advancing processing epoch remain complete configuration-overhead
observations with periodic impact unmeasured; never extend them to add an epoch.

Use the maintained engine paired-median confidence calculation and two-round
+5% rule for compatible revision gates. Do not substitute the standalone
contract-v2 minimum diagnostic or require every confidence interval to fall below
an absolute target. Report observed absolute p50 target attainment, p95, and
uncertainty separately from revision-regression verdicts. Lifecycle-invalid main
runs remain diagnostic. Noisy WSL2 observations cannot establish a reliable
regression, speedup, or equivalence merely by crossing a threshold.

## Resource, correctness and completion gates

Preflight must identify WSL2/host limitations, known competing processes, available
memory, child affinity, power information if available, and disk capacity. Run
one measured workload at a time. Stop if another controlled build/measurement is
active, or if the environment becomes unsuitable. Default resource stops are
aggregate owned-process RSS at 24 GiB, available memory below 4 GiB, state/output
disk at 20 GiB, or less than 10 GiB free disk. Record resource sampling cadence
and conservative peaks. Never start a 4M retained arm with another retained state
resident; settle the previous worker first.

Each successful processing result requires an exact oracle and a separately
recorded terminal/cleanup outcome. On exception, budget expiry, or cancellation,
stop feeding, cancel the job, await its owned tasks/gather work, settle child
processes, preserve raw output, and seal the partial report. Do not continue
dependent target calculations with missing controls. Record checkpoint counts,
state bytes, snapshot/restore evidence, process CPU, peak RSS, and per-phase wall
clocks where applicable.

Completion requires all eight target rows to be represented honestly with fresh
supported main/current observations, mode/thread applicability, raw artifacts,
and target status. Unmeasured, unsupported, invalid, or inconclusive cells remain
explicit; they cannot be described as a completed acceptance pass. The report
does not claim the separate maintained 20-round lifecycle/recovery suite passed
unless that suite was actually selected and completed.

## Implementation ownership and focused red tests

The Python chunk owns `benchmarks/engine_stream.py`, its focused tests, and a narrow
acceptance driver/validator under `scripts/benchmark_suite/` with tests under
`scripts/`. Reuse release/provenance and paired-statistics helpers. Required red
tests cover true-off versus enabled-low, timing exclusions, retained-state proof,
complete eight-row inventory, invalid terminal handling, per-mode main coverage,
matching component fingerprints, fixed samples, and budget/cleanup stop behavior.

The native chunk owns the maintained Join/ASOF benchmark extensions, the small
crate-internal measurement entry point, and common-harness provenance/loader tests.
Required red tests cover exact fixture/project/output matching, full progress
work, actual dirty/restored denominators, compaction proof with split clocks,
exact 1024-row callbacks, wrong/missing/duplicate output rejection, and rejected
unreviewed source overlays. Production behavior and API changes require separate
review. Both chunks preserve other agents' work and coordinate shared files.

All runtime modes use the same 10 ms checkpoint status observer. Its work remains
inside the measured elapsed clocks and is part of the workload identity. Row 7
uses the settled processing clock for subtraction and tests checkpoint activity
through final projection completion; the original Arrow clock is retained.
