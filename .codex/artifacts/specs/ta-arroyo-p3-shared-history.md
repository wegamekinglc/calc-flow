# P3 same-input numeric Rows shared history — specification

Date: 2026-11-08. Source baseline: main
`8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`.
Status: implementation and focused local verification complete; review, CI and performance gates remain.
Rebase on then-current main and inspect merged P1/P2 changes before editing code.

## Source, objective and scope

The user authorized the TA-Lib/Arroyo plan, one PR per implementation step,
autonomous CI/Codacy remediation, and merge only after real performance
improvement, required checks and final specialist review. Plan P3 reduces
duplicated **typed numeric-cache** history and transactional copying when two
different Rows windows read the same Float64 column. It does not change the
mean algorithm or reduce each window's arithmetic work.

The first implementation covers the existing dual-mean/mean-difference path,
with two distinct Rows widths on the same input. It supports both existing
numerical profiles, batch and stream, and existing NaN-as-value compatibility
where that path is admitted. Same-input/same-frame groups already share one
numeric accumulator; they must not be counted as a new benefit. Extrema, Pair,
Duration, exact integer, EWMA/cumulative and the independently developed P2
Decay cache are outside the first sharing path. Other existing plans retain
their execution paths and results.

Use the [introduction](../../../docs/introduction.md),
[Rust rolling contract](../../../docs/rust-api.md),
[symbolic design](../../../docs/symbolic-design.md),
[D13 contract](symbolic-computation-contract.md#15-d13--floating-point-equivalence),
[AGENTS verification](../../../AGENTS.md#verification), and `code-style`.

## Verified implementation facts

- `rolling/compile.rs:901` already groups numeric readouts by input and frame;
  different windows remain different groups, irrespective of `min_periods`.
- `rolling/kernel.rs:2543–2756` gives each Float64NumericState its own timed
  SlidingSampleQueue. Cloning a Rows queue preserves spare slide capacity.
  Update order is add, append original row position, expire, then repair.
- `rolling/kernel/mean.rs:173–221` increments the entity transition count,
  updates groups in declaration order and only then reads outputs. It is a
  specialized path, not merely a wrapper around generic fill.
- `rolling/kernel/entity_parallel.rs:915–975` also calls numeric updates
  directly; a second path cannot silently keep using an emptied owned queue.
- `rolling/kernel.rs:1432,2302` prepares touched entities privately and keeps
  resident states under Arc. Warm copies currently clone every group queue.
- `rolling/kernel.rs:2710,875` uses each group's actual row-frame length for
  repair/rebase and forces stable_v2 rebase while installing restored state.
- `rolling/state_codec.rs:1249` rebuilds typed caches from persisted retained
  history, then restores recurrence seeds/counts. The tape is not currently
  part of a durable checkpoint representation.
- `rolling/budget.rs:201,258` charges logical retained history/reorder state;
  `kernel.rs:1255,2630,4611` reports historical typed byte estimates. These
  estimates are not a MemoryReservation or an allocation-lifetime budget.

The general retained history/columnar rows used for checkpoint recovery are
already shared across outputs and stay unchanged. P3's payload reduction is
from two typed queues' sum of window widths to the maximum width plus a slide
slot. It is not a claim about total state, process RSS, or total arithmetic.

## Functional requirements

- **FR1 — Identity and shape.** Keep public API/project JSON, primitive and
  numerical-profile versions, schema/metadata, row order/identity, output
  count, `min_periods`, late-row and validation/error semantics. Preserve
  group/readout order and same-frame sharing.
- **FR2 — Numerical identity.** For an identical input and segmentation,
  shared versus original independent queues produce bit-identical Float64
  readouts and accumulator fields, including mean, M2, compensated state,
  infinity/NaN counts and signed-zero behavior. Do not substitute sum/count,
  remove M2, add FMA, change add/remove order or weaken a comparison tolerance.
- **FR3 — Windows.** A null/NaN row still occupies an original row position.
  Expiry is by each frame's original-row count, not valid-count or tape size.
  Each group has its own valid count, summary, row-view bounds and repair.
- **FR4 — Owned transitions.** Installed state and old snapshots remain
  immutable during preparation, output construction, rejection or cancellation.
  Successful emission commits the same touched entities at the existing
  boundary; failed/retried transitions do not leak new rows, cursors or credit.
- **FR5 — Useful sharing.** Append one input value per entity row into one
  shared history, and copy that history at most once per touched entity per
  candidate batch. Do not replace duplicate storage with duplicate scratch
  windows or per-group full-history copies during ordinary updates.
- **FR6 — Single-window control.** If there is no eligible different-width
  same-input pair, retain the original queue/update path: no shared tape/map,
  shared-cache credit or per-row sharing/cursor work. Existing one-window
  clone's spare-slot allocation property remains. Report unavoidable fixed
  header/dispatch changes rather than claiming literally zero global overhead.

## Ownership and transition proposal

This is one coherent proposal to critique, not an additional public abstraction.
The implementer may simplify concrete types while preserving the invariants.

### Plan facts and entity owner

Derive private sharing facts from the unchanged logical numeric group list:
input column, participating group ordinals, their Rows widths, and maximum
width. Do not replace/reorder TypedGroupPlan descriptors merely to select a
storage policy. Compute these facts once per plan/batch entry, not per row.

An eligible entity owns one tape handle for that input and the existing
independent numeric summaries. A tape allocation has one actual owner carrying
its storage and cache credit; an Arc may keep that immutable owner alive across
resident/candidate/snapshot views. Per-group views identify the owner and frame
bounds through the entity; they do not each hold a separate mutable Arc or
history allocation. Rows tapes may store Option<f64> without redundant event
times because time is not their expiry coordinate. Preserve value IEEE bits.

On touched-entity preparation, copy group summaries and obtain a private tape
owner once, reserving its new allocation before copying. Arc cloning alone
does not authorize mutation. Avoid repeated `Arc::make_mut` at each group;
the owner must already be exclusively writable before the row loop. Untouched
entities and retained old snapshots keep their prior owners and credits.

Shared groups must not also keep populated independent sample queues. Existing
owned-queue representation remains available for nonsharing/fallback plans.
The representation must permit all refold/rebase reads through a borrowed
group-specific ordered view without materializing a Vec for that view.

### Exact per-row sequence

For each entity row, using logical ordinal t and width Wg for group g:

1. Perform the existing checked entity-transition increment and preserve its
   original error site. Read/validate the input according to the existing NaN
   policy. Append that Option value to the private tape once.
2. Iterate groups in their original order. For each group, add the current
   valid value using its existing accumulator transition. If its prior row
   count was at least Wg, remove the value at ordinal `t-Wg`, if valid, using
   the existing removal transition. Do not remove before add.
3. Repair that group immediately at its original point. Its post-expiry view
   is original positions `max(0,t-Wg+1)…t`; its row count is `min(t+1,Wg)`.
   Iterate values oldest first for refold/rebase. The shared maximum width is
   never a substitute for this group's sample length or rebase cadence.
4. After all groups have completed their updates/repairs, trim the tape to
   the maximum-width suffix and perform the existing output readouts in their
   original order. The transient append requires at most max-width+1 live row
   positions. In particular, do not discard the largest group's outgoing value
   before that group's removal callback can read it.

The tape coordinate is private; rebasing a local coordinate must not change
the durable entity transition count. Avoid introducing a new lifetime-counter
overflow error that an unchanged query could encounter sooner. Row views must
work across VecDeque wraparound and after restore with only a retained suffix.

For Wsmall=2 and Wlarge=4, after appending the fifth original row, the small
group removes row 3 while the large group removes row 1. Both outgoing values
must remain readable until their respective group update completes; each
repair sees only its own last 2 or 4 rows.

### Execution-path and failure coverage

The mean-specialized fill, generic typed fill when used by restore, forced
stable_v2 restore rebase and any admitted entity-parallel lane must understand
the same owner/views. Reuse one transition/repair definition where practical;
do not silently disable an existing dual-SMA parallel path to obtain smaller
serial state. A necessary lane adapter must preserve lane cancellation points,
original row/group error tags and deterministic selected-error arbitration.
No tape is shared between entities or concurrent entity lanes.

If the new optimization cannot prepare a private credited owner, select the
old owned-queue path **before changing installed state**. When converting a
resident shared cache for that candidate, seed each old queue from the proper
window view and copy its existing summary directly; replaying add operations
would change West/M2 rounding. Preserve the old installed shared owner until
normal commit. Failure or cancellation drops the candidate and leaves the
prior summaries, tape and visible counters intact.

## Actual allocation credit versus logical StateBudget

Keep the existing user-visible logical StateBudget definition, success/failure
thresholds, snapshot charge and JSON counter range. It does not become a limit
on physical RSS or a promise to account for every allocator allocation.

The **new shared cache** must use a private bounded resource pool and RAII
reservations, following the merged P2 cache facility when available rather
than inventing a second public budget. Ownership/credits remain separate from
P2's Decay histories; do not mix their semantics or offsets. The design must
name the existing/private pool source and its actual limit before code review.

Credit requirements are testable:

- Reserve conservatively before allocating/growing the tape or added cache
  metadata, then reconcile to actual allocated capacity and the facility's
  documented owner/header charge. Capacity, not current length or a reporting
  getter, determines storage charge.
- A deep candidate copy has a distinct credit while the old owner is live.
  An Arc alias of the same allocation does not double-charge it or free its
  credit prematurely. Credit stays with the actual allocation until its last
  owner drops, including a held old snapshot or abandoned candidate.
- Account for growth/reallocation peak while old and new buffers coexist;
  reserve spare slide capacity so the first append after a warm copy does not
  allocate again. Checked arithmetic guards requested capacity/byte products.
- Cache-credit refusal chooses the old implementation without a new public
  budget error or silent dropping of a required old logical charge. That
  fallback retains existing allocation/budget behavior; it does not claim
  physical coverage for the old caches. Real allocator failure has no stronger
  guarantee than the existing path.
- Observe cache capacity, live owner count, held credit, copied payload and
  released credit independently of the historical `state_bytes` estimate.
  Preserve the old estimate basis for untouched paths; any shared-path metric
  adjustment is documented as that metric, not misrepresented as enforcement.

If the proposed cache policy changes whether an otherwise valid query passes
its logical StateBudget, the implementation fails this specification.

## Checkpoint compatibility

Retain the existing writer layout and history projection, state schema,
configuration hash, numerical-profile identity and checkpoint fingerprint.
Sharing facts are private storage facts, not new logical group descriptors;
keeping the group/readout descriptors unchanged should make a new legacy-hash
mapping unnecessary. If the final implementation changes fingerprint inputs,
derive and test exact legacy identities rather than bypass validation.

On restore, use the same retained rows and recurrence/count seeding already
used by the original implementation, then build shared owners/views. Each
group's reconstructed summary must match the original reconstruction at that
same point; stable_v2's forced rebase remains per-window and oldest first.
Do not add unpersisted recurrence history whose rounding cannot be recovered.
The uninterrupted candidate and restored candidate obey the existing cross-
recovery numerical contract; additionally, candidate versus baseline is
bit-identical for each identical uninterrupted or restored execution route.

An old-baseline checkpoint with live history/pending rows must restore and
continue into the candidate. Candidate checkpoints must preserve the old
reader's expected contract, not just be readable by themselves. Failed restore
does not replace installed state or leak prepared cache credit.

## Focused acceptance criteria and RED tests

Begin each new behavior with a focused observed failure, then green. Reuse
neighboring kernel/mean/entity-parallel/restore tests and allocation evidence.

- [ ] Same-input Rows 2/4 and 5/20 dual means/difference use one tape payload;
  same-frame outputs still use their existing single group. Different inputs,
  single-window, unsupported types and excluded group families use old paths.
- [ ] Per-row sample and every accumulator field/readout match independent
  queues bit-for-bit under both profiles. Include warm-up, different
  min_periods, null/NaN gaps, NaN-as-value, infinities entering/leaving,
  alternating signs/zeros, strong cancellation, the M2 repair case
  `[1e155,1e155,1e155,-1e155]`, and near-overflow finite West mean.
- [ ] Narrow/wide windows expire their distinct outgoing positions correctly;
  refold/rebase uses each group's own post-expiry view and length. Cover
  wrapped storage, single-row/multi-row appends, empty batches, mixed entities
  and multiple Arrow RecordBatches.
- [ ] Touched-entity copies duplicate shared payload once, preserve spare slide
  capacity and leave held old snapshots unchanged. Each owner stores at most max(W) steady logical rows plus one transient row;
  actual capacity, candidate copies and held old owners are credited separately.
  There are no hidden per-group queues or copied scratch windows.
  Single-window allocation count and old estimated-byte basis remain covered.
- [ ] Credit tests force refusal during initialization, cloning and growth;
  old-path fallback preserves result bits and logical budget decisions.
  Held aliases, failed prepare, cancellation, output refusal and retry release
  precisely the candidate credits; dropping the last old view releases its
  own credit. No uncredited growth or prematurely released credit remains.
- [ ] Generic bootstrap/forced restore rebase, mean-specialized execution and
  admitted parallel lanes match independent queues. Lane errors/cancellation
  retain original row/group ordering and settle all candidate owners.
- [ ] A legacy golden snapshot made by fixed baseline restores with original
  fingerprints and yields matching subsequent batches across expiry/rebase.
  Compare candidate writer schema/metadata/history and old-reader acceptance.
  Wrong-profile/schema/configuration and oversized restore remain rejected
  before state installation; no new schema-version fields appear.
- [ ] Focused Rust kernel/mean/parallel/codec tests and corresponding Python
  dual-SMA finance tests pass; format/lint/module checks cover changed source.
  Full regression, combined coverage, cross-platform and Codacy gates stay
  in CI and must pass before merge; final specialist review is required.

## Performance acceptance

### Fixed acquisition protocol — planned, not run

This replaces the initial three-case sketch; no benchmark/probe/build has run
for this protocol. The measured target is **warm append end-to-end latency**,
from prepared Data enqueue through native tasks/channels, rolling finality,
projection, sink and `to_pyarrow` to received Arrow output. As maintained in
`profile_warm_stream.py:667–695`, IPC, generation, startup/history, validation,
checkpoint and shutdown are outside that per-append timer; they all count
against the inclusive acquisition budget below. Callback/kernel timers alone
cannot satisfy the user's real end-to-end speedup requirement.

**Revision gate.** Final baseline/candidate SHAs and native/build identities
are pending; this is a case/resource freeze, not a declaration that an old
wheel is usable. Baseline must be then-current main, including the final
merged P2 fixes if P2 lands first. Rebase P3 onto that exact main, run the
narrow numerical/legacy/resource/sorted regressions after resolving changes,
obtain delta review for any new source delta, then freeze candidate source and
both compatible release wheels. Candidate differs from that baseline only by
P3; do not copy unmerged P2 or use P2's earlier failed/fix-pending/P1 wheel as a
baseline. Keep the old 8d legacy writer fixture as compatibility evidence; it
is not the future performance baseline. Record actual SHAs, effective diffs,
source digests and build hashes before acquisition; any rebase/source/native
change invalidates the old freeze and requires explicit re-freezing before
measurement, not mixing samples.

**Exactly three workloads, one normal-GC mode, Float64, default stable_v1.**
The generic helper accepts these windows; its current defaults are 5/20, so
20/60 must be passed explicitly. All histories are fully warmed before the
first measured append. `append_entities` is explicit even in the one-entity
cases, avoiding `matrix_points`' implicit extra 1,024,000-history workloads.

| ID / role     | Indicator and Rows windows            | Entities / active per append | History rows / physical segment | Append rows | Samples        |
|---------------|---------------------------------------|------------------------------|---------------------------------|-------------|----------------|
| P3-T1 target  | dual_sma_spread, fast20/slow60        | 1 / 1                        | 1000 / 1000                     | 1024        | 30 AB/BA pairs |
| P3-C1 control | rolling_mean, window60; fast20 unused | 1 / 1                        | 1000 / 1000                     | 1024        | 30 AB/BA pairs |
| P3-T2 target  | dual_sma_spread, fast20/slow60        | 10000 / 1000                 | 600000 / 1000                   | 1000        | 30 AB/BA pairs |

T1 targets duplicate input history and transactional queue copying. C1 is the
same resource/input/append configuration without a sharing pair. T2 has 60
samples for every one of 10,000 resident entities, the maintained holdings
scale, then touches only the first 1000 entities per append. The 600,000 warm
rows are the minimum full slow-window history for this fixture, not arbitrary
inflation. Never reduce its entity count, silently use partial warm windows,
change window widths, add cores/types/profiles or add a fourth SQL workload to
seek a favorable result. The 20/60 and sub-64,000-row append shapes do not
enter entity-parallel lanes (`kernel/entity_parallel.rs:310–333`); generic,
lane/rebase and sorted SQL two-AVG behavior retain their direct correctness
coverage, not a claim of measured performance here.

**Frozen resources and ownership.** Use CPU affinity `[0,2]`, maintained
four Tokio workers on both sides, no `--worker-threads` experiment, one fixed
DataFusion partition and its default batch size 8192/rolling rewrite setting.
Use the same interpreter, dependency lock, release profile/native features,
Arrow/NumPy versions and normal GC (`collect_gc=false`, GC stays enabled;
no `gc.disable`, forced collection or GC tuning). Fix inherited `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `RAYON_NUM_THREADS` and
`POLARS_MAX_THREADS` to 1 on both sides and record them with observed
affinity/Tokio count. If those CPUs or the quiet machine are unavailable, stop; do not substitute another configuration in this run.
Retain `Runtime()` and `_native_program(..., engine="streaming")`, no selected
UDF, original partial-window/min_periods/default stable_v1 plan, SourceProvided
watermarks and the ordinary sink. Assert the lowered plan/profile/config in
preflight rather than assuming Python defaults. `WarmScenario.start` uses a
fresh managed directory, checkpoint interval 24 hours and edge budget
`max_rows=max(history_segment_rows,append_rows)`, bytes=64 MiB: 1024 rows for
T1/C1 and 1000 for T2. No nonterminal checkpoint/restore is injected; original
EOF/terminal settlement stays outside append timing and inside acquisition.
Logical StateBudget defaults remain unchanged. The candidate's existing
process-local 64 MiB shared-tape quota remains active, with no test-pool override.

There is no extra held-old-snapshot case: the maintained worker exposes no
cache-alias holder. The resident old owner during ordinary prepare/emit/commit
already causes real transactional COW. That is distinct from deliberately
holding an additional snapshot across batches; unit tests cover such aliases,
but this performance protocol does not manufacture or claim that exposure.

### Existing entry points and minimal harness gaps

Reuse `benchmarks/warm_stream.py` and `scripts/profile_warm_stream.py`, including
`_fresh_case`, `_measure_case`, `_compatible_builds`, worker cleanup and
`paired_summary`. Current compare CLI exposes history/append/active/indicator/GC,
**not** entities/window/fast_window/history_segment_rows. Defaults alone would
measure 64 entities and 5/20, and its ordinary matrix adds unrelated cases
(`profile_warm_stream.py:58–88,842–865`). Add only four positive scalar flags
`--entities`, `--window`, `--fast-window`, `--history-segment-rows`; carry them
into every explicit case config/start response and identity. Existing CLI
behavior without these flags stays unchanged; use the existing explicit
append_entities product for these three cases, not a new framework.

T2 requires two additional **benchmark fixture** corrections before it can run:

- `ScenarioConfig` currently requires history and history_segment_rows to be
  complete entity ticks (`warm_stream.py:328–369`): segment1000/e10000 is
  rejected. Do not merely delete the check. Keep history600000 divisible by
  entities10000, and permit this segment1000 only with bounded warm feeding:
  enqueue ten original 1000-row Data envelopes for one complete 10,000-row
  history tick, then one strictly advancing Watermark closing that tick; drain
  and validate all 10,000 output rows/chunks before feeding the next tick.
  Preserve the 600 physical Data boundaries and 60 complete-tick watermarks.
  The runtime's ordered buffer concatenates closed records and chunks output
  to the existing edge budget (`ordered_stream.rs:444–538`); the harness must
  receive all chunks rather than waiting for one 1000-row result per partial
  tick. Keep at most one warm tick pending. Do not emit equal/early watermarks:
  rolling rejects nonadvancing watermarks and treats later rows at a closed
  timestamp as late (`rolling.rs:1557–1603,1889–1904`).
- `_segment`/`_input_table` currently name symbols with fixed `:03d`; at 1000
  the lexical order diverges from numeric sequence. Use the same minimal
  dynamic width `max(3,len(str(entities-1)))` in both functions: four digits
  for T2, unchanged three digits for T1/C1. Seal this changed fixture identity.

Distinguish warm from measured sparse timestamp layout explicitly. Warm T2 has
60 complete 10,000-row ticks with `positions=sequence//entities`. Its measured
appends already use the maintained `_input_table` sparse branch
(`warm_stream.py:136–161`): `positions=sequence` gives every appended row a
strictly increasing new timestamp, and symbol index is
`(sequence-history_rows)%append_entities`. Thus each measured 1000-row batch
has 1000 active entities; the other 9000 do not arrive at that new time. Its
single closing watermark safely produces that sample's 1000 outputs without
waiting for the other entities. Preserve that sparse layout and its existing
`_sparse_expected`/`_entity_context` formulas; never merge ten measured append
batches into a 10,000-entity update or describe sparse timestamps as full-tick
ones. Baseline and candidate receive identical sealed inputs and watermarks.
Add focused helper checks for this fixed layout, chunked warm drain, correct
oracle context, strict watermark advance and zero late rows. A gap in those
checks or old-baseline compatibility blocks acquisition, not permission to
change this case after seeing timings.

The existing build command is:

```bash
uv run python scripts/profile_warm_stream.py build \
  --source <sealed-source> --target <separate-target> --output <fresh-wheel-dir>
```

It uses release/locked Maturin with `pyo3/abi3-py313` and emits `build.json`;
use the same suitable managed interpreter and flags for both sources. The
following compare commands are **future commands requiring the four scalar
flags and fixture/oracle fixes above**; they are not runnable as written on
today's unextended CLI. Existing compare options/statistics remain in use.
Run T1/C1 first (their fixed seeded order), then T2, once each:

```bash
taskset -c 0,2 uv run python scripts/profile_warm_stream.py compare \
  --baseline-build <baseline-build.json> --candidate-build <candidate-build.json> \
  --output target/p3-perf/t1-c1.json --samples 30 --seed 20260905 \
  --history-rows 1000 --append-rows 1024 --append-entities 1 \
  --indicators dual_sma_spread rolling_mean --gc-modes normal \
  --entities 1 --window 60 --fast-window 20 --history-segment-rows 1000

taskset -c 0,2 uv run python scripts/profile_warm_stream.py compare \
  --baseline-build <baseline-build.json> --candidate-build <candidate-build.json> \
  --output target/p3-perf/t2.json --samples 30 --seed 20260905 \
  --history-rows 600000 --append-rows 1000 --append-entities 1000 \
  --indicators dual_sma_spread --gc-modes normal \
  --entities 10000 --window 60 --fast-window 20 --history-segment-rows 1000
```

The controller and worker harness are common to both installed wheels. Seal
all four existing reported harness hashes, the exact shared helper patch,
fixture formulas/warm event boundaries, expanded ScenarioConfig, project/plan
fingerprints, build/source/Cargo.lock/wheel/native hashes, rustc/Python and
worker machine/dependency/affinity identities. `_load_build` rechecks source
and native hashes; `_worker_environments` permits only native SHA differences
for an engine comparison and verifies default Tokio4. Do not use
`--worker-threads 4 4`: that selects a same-wheel scheduler experiment and
rejects different engine wheels. Recheck seals at finish and retain all raw
start rows/samples, phase intervals and terminal outcomes. Reuse the maintained
quiet guard/owned RSS process helpers, without checkpoint-off SUM validators
or changing warm worker ownership. No profiling callbacks are enabled in the
three timing cases.

### Bitwise oracle, resource evidence and fixed decisions

The current independent warm oracle uses `np.isclose(rtol=1e-10,atol=1e-10,
equal_nan=True)` and the controller only checks `passed`/start_row
(`warm_stream.py:253–290`, `profile_warm_stream.py:477–479`). Keep that independent
value oracle, and add baseline-vs-candidate **exact** digest validation outside
timing. Keep exact logical Arrow row order; canonicalize only chunk boundaries
and the schema/field names/types/nullability/metadata representation. Use a
versioned SHA-256 encoding with length-delimited schema/column/row counts,
then hash each Float64 field's logical validity bits and only its raw 64-bit
value bits for **valid** slots. Respect array offsets and
chunk boundaries, use fixed little-endian encoding, preserve signed zero and
all valid NaN payload bits; exclude undefined null payload/padding bytes.
Do not sort rows or normalize values. Other identity columns retain
exact schema/sequence/input-table comparisons. Return the digest for all warm
outputs as an ordered running digest in the start response, and for each
measured sample after the existing timer stops. Match both start responses
and each corresponding AB/BA pair before accepting its timing. Any one-bit
mismatch fails the case even when `isclose` passes. Focused helper checks must
distinguish ±0 and valid NaN payloads, reject validity changes and ignore only
null payload/chunk-layout differences; do not add these as timing workloads.

Keep three different evidence levels separate:

- **Direct unit evidence, already recorded by author/reviewer:**
  `numeric_rows.rs:624–674` measures exactly two actual COW allocations
  (payload+owner), a distinct warm-capacity tape owner, unchanged old data,
  independent live credit and a zero-allocation warm slide. Summary-bit/refold and two-profile old-writer
  tests verify arithmetic/codec; `tests/shared_rows.rs:355–439` verifies real
  serial emit refusal/cancellation and release/replay with a held old alias.
  These are correctness/ownership observations, not P3 timing results.
- **Existing native observations:** before/after rolling metrics report
  touched_entities, copied_entities, numeric_rows, history_rows_materialized,
  output rows/chunks and exclusive callback stages (`rolling_metrics.rs:45–111`).
  Native copied_entities counts state copies, not tape/queue bytes, and may
  remain unchanged between versions. These lifetime deltas include post-warm
  terminal work; never attribute them to an individual sample without exact
  boundaries. No current wheel-visible counter measures shared-tape copied
  rows/bytes, actual tape reservations or fallback count. If those quantities
  are claimed as measured, the minimal additional opt-in diagnostic is private
  allocation-site counters for successful tape init/COW, actual rows/bytes
  copied, reserve refusal/fallback and actual credit acquire/release/peak;
  old queues need equivalent copy-site instrumentation. A sampled getter or
  estimated_bytes does not establish allocation ownership. Such a diagnostic
  must be separately sealed and untimed; it is not another favorable timing
  mode or a prerequisite for reporting the narrower native counters honestly.
- **Whole-process resources:** current worker finish reports only terminal
  `rss_bytes`, not peak. Use the existing owned-process RSS sampler for the
  two worker process trees, label its sampled peak/20 ms poll interval
  separately, and record process lifetime/cleanup. Missing/failed resource
  samples are unmeasured, not zero. Never infer whole RSS, allocation counts
  or bytes copied from ΣW→maxW, state_bytes estimates or end RSS. In particular
  live resident state, temporary old/new COW owners, Arrow retained history,
  keys, fixed headers and runtime infrastructure all remain real costs.

Use `paired-warm-stream-v1` unchanged: fresh process pair per case; 30 advancing
append pairs within those two persistent runners, alternating AB/BA by pair
index. This is not 30 fresh-process pairs. Report baseline/candidate P50/P95,
raw latencies, throughput, CV and the maintained ratio median. With r=B/C,
`paired_summary` computes median of the 30 paired ratios, 20,000 median
bootstrap resamples with seed771031 and percentile CI95
(`profile_warm_stream.py:101–130`), not a ratio of separate medians and not the
release_pairs two-round contract. Freeze these decisions before any sample:

- At least one pre-registered target T1 or T2 must have
  `paired_speedup_median > 1.05` **and** `paired_speedup_ci95[0] > 1`.
- A confirmed >5% regression in **any of all three cases** is
  `paired_speedup_ci95[1] < 1/1.05` and blocks merge. Do not invent a stricter
  requirement that every lower bound exceed the no-regression threshold.
- Missing/invalid/bit-mismatched/incomparable/incomplete samples cannot pass.
  A nonsignificant result is inconclusive, not equivalence or improvement.
  P95/resources and all unmeasured routes must be reported; an apparent
  resource-only gain does not satisfy this user's merge authority.

**Inclusive wall budget: 600 seconds total across both commands**, comprising
50 seconds artifact/fixture/helper preflight and provenance, 70 worker wheel
installation/startup, 260 history feeding/oracles across all six runners
(mainly T2), 120 for 90 AB/BA pairs with all generation/IPC/validation gaps,
60 EOF/terminal checkpoint/owned cleanup and 40 bootstrap/report/seal checks.
These are planning allowances, not measured costs. Build estimates are
10–25 minutes per compatible release wheel, separately logged; colder builds
can exceed that and are not acquisition time. One shared monotonic deadline
starts before preflight, spans both commands and reserves owned cleanup;
current standalone CLI has no global deadline, so the targeted controller
wrapper must enforce remaining time and settle both workers/processes, not
merely kill the shell. Stop on timeout, incompatible baseline/helper, source
drift, resource noise or unsettled ownership; preserve partial evidence.
Do not increase rows, samples, retry count, cases or budget, adjust thresholds,
disable GC or rerun until favorable. Real target improvement, no confirmed
selected regression, final-head CI/Codacy and final specialist review remain
merge gates; source review alone does not authorize merge.

## Handoff gates

`cf-critic` must first confirm exact numerical sequencing across all selected
execution/restore paths, fallible COW/cache-credit fallback, unchanged logical
budget semantics, and the unchanged durable identity. Then `cf-implementer`
can implement the scoped plan and focused RED tests. If a version/identity or
public budget change is unavoidable, return for a specific contract decision;
do not solve it by weakening the mean algorithm or altering the default profile.

The initial specification was produced by static source research. Implementation
and focused verification evidence is recorded below; no P3 performance
measurement has been run.

## Implementation decisions (2026-11-08)

The first implementation uses a private process-local 64 MiB
`GreedyMemoryPool` exclusively for optional shared numeric Rows tapes. Each
actual Arc-owned tape carries a `MemoryReservation` covering its allocated
capacity, the tape/Arc header, and a conservative 256-byte registration charge.
This pool does not change logical `StateBudget` accounting.

Admission reserves the complete maximum Rows width plus one slide slot before
allocation. There is no growing allocation in the row loop. A shallow alias
keeps its owner's credit; preparing a writable candidate reserves and copies
one tape. Refusal materializes each existing window suffix and preserves its
accumulator fields. Restore replay keeps the original logical groups and
reconstruction order, then enables sharing before recurrence/count seeding.

### Test-first execution log

- Initial storage test: `dual_mean_rows_do_not_retain_duplicate_input_queues`
  uses the existing 2/4 mean path, checks values 7.5/6.5, and requires independent
  queues to retain at most the widest window's four samples. Baseline retains
  six. The first attempted run had a test-only missing enum import; that was
  corrected before any production change. It is not counted as a behavior RED.
- Corrected focused RED is running with `cargo test --locked -p calc-flow --lib
  --features benchmark-internals dual_mean_rows_do_not_retain_duplicate_input_queues`.
  Only the core library test target is rebuilt, with no full suite or timings.

The immutable kernel plan now records an optional private sharing shape after
checking the real input schema is Float64 and the output shape is an exclusive
mean/difference plan. This field is excluded from `KernelFingerprintInput` and
the state codec; cloning it for output-cleared reconstruction preserves the
original admission fact. One optional Arc handle increases the entity's fixed
typed-cache header; generic `WindowAccumulator` and `WindowState` layouts are
unchanged. That fixed header is included in the single-window control's
observed cost and is not represented as a logical history-budget change.

- Corrected storage RED completed: 1 failed, 0 passed, 2586 filtered; expected
  assertion `duplicate standalone samples: 6`. Core build took 6m24s; execution
  took 0.00s. No production source had been changed at that point.
- Initial implementation and additional focused numerical/ownership tests are
  written and formatted. GREEN compilation is deliberately pending the root's
  release of the P1b measurement quiet window. No P3 timing has been acquired.

### Integration evidence (2026-11-08)

- First MeanRows/backend GREEN: storage test passed after a 4m18s core build.
  All three initial numerical/ownership tests passed (0.02s). They check every
  Float64 summary bit, raw suffixes, warm-copy spare capacity, old aliases and
  credit refusal without replay.
- Existing parallel complete-state/bit test then failed on the unadapted lane;
  existing compiled-vs-generic property failed on the unadapted generic queue.
  Both REDs were observed using the already-built test binary.
- A new forced-restore rebase test observed valid-count zero from the emptied
  owned queue instead of the narrow window's five rows. That RED build took
  2m48s. The adapted borrowed-view rebase passed after a 4m32s core build,
  without copying the shared tape. The generic property also passed.
- Parallel focused execution: 48/49 tests passed, including output/state bits,
  cancellation/error arbitration and unchanged lane allocation bounds. The
  remaining allocator probe exposed the intentional process-wide pool's first
  Arc allocation. A cold isolated reproduction failed with one allocation;
  explicitly initializing the same pool before the identical zero-owned-byte
  probe made both selected tests pass. The fixture now performs that one-time
  infrastructure initialization outside the owned-merge probe; zero assertions
  are retained.
- The fixed two-profile old-source fixture was generated by untouched main
  runtime at `8d8be7d9` plus the adjacent test-only generator patch. An independent
  cache was copied only after the candidate cache became quiescent; the old core
  was explicitly cleaned and rebuilt. Capture took 6m45s build plus 0.03s test.
  The candidate's complete writer bytes/metadata, logical budget charge, and
  both 72-row continuation routes match the pinned old fixture exactly
  (1 pass, 0.03s). Provenance and binary/patch hashes are stored beside it.
- New optional-cache code has no growth path: initialization reserves max-width
  plus the slide slot. Refusal is tested at initialization and COW, while a
  warm first slide is allocation-free. The process-wide pool object is fixed
  infrastructure; allocation reservations cover each tape/Arc/header and its
  conservative registration charge. Shallow aliases do not duplicate credit.
- The generic fill cleanup keeps row/group/readout ordering and observed-work
  accounting, with lizard CC8 for fill and CC4 for its entity-COW helper. No lint
  suppression was added. Final focused checks and scoped clippy remain pending.


### Final verification handoff

The consumer-registration cleanup uses DataFusion 55's
`MemoryReservation::new_empty`: both leases share their registration/consumer,
while each actual tape owner has an independent reservation size and drop. The
old owner's bytes are never split or transferred into the new owner. Initial
owners still register normally; COW keeps the same conservative fee basis and
reserves the new allocation before copying. The resource test now also requires
the same consumer address and exactly two actual COW allocations (payload and
owner). Its existing held-alias, quota-refusal, and independent-drop assertions
are unchanged.

Final scoped `cargo clippy -p calc-flow --lib --tests --features
benchmark-internals -- -D warnings` passed in 2m18s after two test-only style
fixes, without suppressions. `cargo fmt --all --check`, whitespace, and schema
no-drift checks passed. The final COW/credit resource execution passed all four tests after a 2m45s
build (0.02s execution). The independent positive-credit budget/rejection,
cold zero-owned allocator, and generic property checks then passed all three
selected tests (0.40s). This confirms the same consumer with independent lease
sizes/drops and two actual COW allocations. Earlier numerical, rollback and
legacy compatibility results remain recorded above. No full-workspace tests, coverage, Python native rebuild,
remote CI, commit, push, or performance acquisition were performed here.


The frozen source/fixture manifest is stored under
`target/ta-arroyo-p3-debug-cargo/p3-source-freeze.json`. Its
`code_manifest_sha256` identifier is
`97cd2b97fd269ecea161ed91ce6bd78271624b36bdad2f2ee772a058e96fcc6d`;
the JSON file's own SHA-256 is
`ed2e0fa7bc340a682216c791174de2ac799437442fbd8467f1436a9c783d31d1`.
The generator's original context-form patch hash is retained in provenance;
the committed audit patch uses zero context to avoid a whitespace-only context
line, and passes reverse-application validation against the actual legacy tree.
This presentation change does not change any Rust source, the pinned binary,
or the frozen fixture bytes. There are no owned build/test processes left at
this handoff. Source is uncommitted; no P3 speedup has been established.


## Source-review follow-up: SQL sorted execution and serial callback credit

The parent confirmed that the existing batch/stream dual-mean scope includes
physically sorted SQL AVG Rows5/20. DataFusion 55 can put both frames in one
bounded window node; it must not be treated as a single-group plan. The first
review's owned-queue SQL boundary is now implemented within the original
Float64 Numeric Rows scope. COUNT and other unsupported shapes are unchanged.

The focused RED instantiated the actual DataFusionRollingKernel with two AVG
frames, proved the compiled sharing fact was present, and failed because its
sorted entity still owned separate queues. Build took 3m30s; the independent
new serial-credit test already passed and required no production change.

Sorted preparation now uses the same owner/views and ordered group transitions.
The real DataFusion stream consumes its uniquely owned state instead of cloning
it every batch. A continuing partition keeps its tape allocation; an explicit
old alias still causes a separately credited COW. The borrowed test/reference
entry clones before entering that owned transition, preserving isolation.
Partitions that finish release their owner before admitting the next tape.

DataFusion's existing reservation.try_resize(state_bytes) remains in the same
execution boundary. Shared sorted state charges its group headers and one
owner reservation, plus the prior key estimate; it does not charge a tape per
readout. The private pool separately bounds actual tape owners, including held
old aliases. Refusal materializes the original per-window queues and summaries;
the old queue estimate and DataFusion reservation check remain active. Neither
budget is advertised as process RSS, and fixed headers/dispatch cost remain
part of the controls rather than a claim of zero nonsharing overhead.

The new serial direct-operator test installs an independent pool before creating
cache, holds a real old typed-state alias, and requires positive resident credit.
For ordinary first-chunk refusal and the existing Cancelled result it compares
encoded state, typed state, tape identity, logical charge, watermark and output
sequence; candidate credit must return exactly to the resident amount. Explicit
caller replay then succeeds with exact mean outputs. Dropping snapshot/aliases
and target returns credit to zero. This direct emit_ordered boundary does not
promise retry of an already-cancelled managed runtime callback.

Follow-up evidence:

- GREEN build 4m09s; five new tests passed in 0.04s. They cover a real SQL query
  with one windows=2 sorted node over small batches/two partitions, bitwise
  original DataFusion finite reference, the unchanged zero-byte DataFusion
  rejection, exact original-queue NaN/partition/refusal behavior, owner reuse,
  held-alias COW/fallback, and the serial rejection/cancellation combination.
- Six directly affected old controls passed: original dictionary/single-window/
  empty-batch continuation, four numerical/credit/rebase tests, and the pinned
  old-writer/two-profile continuation. The separate COUNT SQL kernel metadata
  fingerprint control passed as well.
- Scoped core lib/tests Clippy passed in 3m20s with -D warnings; the actual
  scripts/verify_complexity_gates.py ratchet passed without baseline or allow
  changes. Changed sorted/owned-transition functions have lizard CC at most 7.
- The old 97cd manifest and fixture are retained unchanged. The follow-up seal
  is target/ta-arroyo-p3-debug-cargo/p3-source-freeze-review-followup.json with
  code_manifest_sha256 52807475a37fa96781260c578dc089f5a3abe22c8109bfb2caf02786259570bd (15 runtime/test/fixture files).

No performance acquisition, commit, push, remote publication, Python rebuild or
full-workspace verification was performed for this follow-up. Independent delta
review, final-head CI/Codacy and real performance improvement remain merge gates.


## Draft source handoff

The independently reviewed 15 runtime/test/fixture files still match the
52807475 code manifest exactly; its JSON file digest is
5907a767e4c6251f8e07096581b0ec074a4a7cdbf93b989189158d04d40e6731.
The original 97cd record, pinned old fixture and both generator identities are
retained. The source-stage reviewer cleared the sorted and serial-credit
follow-up; this does not replace committed-head CI or performance acceptance.

The final evidence remains archived under target/ta-arroyo-p3-debug-cargo:
p3-review-followup-green.log records 5 passes, p3-review-controls.log records
6 passes, p3-review-sql-route-control.log records 1 pass, and
p3-review-scoped-clippy.log records successful core lib/tests Clippy with
benchmark-internals and -D warnings. The actual complexity ratchet, formatter,
tracked/untracked whitespace and project-schema no-drift checks also passed
during the reviewed implementation. This handoff verifies those receipts and
source seals rather than repeating unchanged successful tests.

The performance protocol above is unchanged. No native rebuild, adapter,
acquisition, Python matrix, remote CI result or measured speedup is added by
this source handoff. A local source commit prepares a draft PR for real
CI/Codacy; the final baseline and candidate builds still require then-current
main and any final merged P2 changes as specified by the protocol.

## Integration onto main 7ac64990

P3 is rebased onto exact main
`7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75`, which includes the merged P2
implementation. The runtime conflicts only combine P2's existing decay pool
with P3's shared-row plan initialization and test constructors. Both caches
retain their admission and owner accounting. P3 still admits only exclusive
Float64 Rows mean/difference plans; a stage containing decay keeps P2's
original typed cache/reference path. No numeric recurrence, M2 transition,
codec, fingerprint input, quota or logical-budget rule changed in this merge.
The original shared-row allocator, entity-parallel implementation and old
8d fixtures/generator remain byte-identical to the previously reviewed P3.

Two Python integration defects had focused RED/GREEN evidence. The existing
P2 small-stream test first failed with `KeyError: rolling_decay`: the new plan
preflight had bypassed P2's `_warm_program`. Plan preparation now uses that
unchanged declaration helper and checks the exact existing decay or mixed
decay/mean outputs as well as the mean shapes. Both P2 indicators pass the
same small-stream test, including their independent value oracles. A second
controller test showed that requiring bitwise equality for certified decay
would reject results allowed by P2's existing D13 tolerance. All output
digests remain recorded; paired exact equality applies to the original
mean/dual-mean indicators, including every frozen P3 case. P2 decay indicators
retain their original value-oracle acceptance. Expanded configuration and
project/program/plan identity checks remain required for both families.

The integrated core library test build took 3m20s. Its eight shared-history
tests pass, including both-profile old-writer recovery, positive-credit
restore/serial rejection and actual SQL sorted AVG. Seventeen further exact
test selections pass in 0.90s: P2 typed/mixed/buffered recovery, refused and
cancelled work, budgets and twelve work counters; P3 MeanRows/generic parity,
all summary bits, sorted continuation/alias/refusal and parallel paths.
Scoped core `--lib --tests --features benchmark-internals` Clippy with
`-D warnings` passes in 2m27s. The directly affected Python/controller checks
pass 62 tests plus two subtests; twenty P2 helper controls and the existing
small-stream test also pass. Formatter, Ruff, C901 at most eight, the actual
complexity ratchet, whitespace and project-schema no-drift checks pass.

Receipts are archived under `target/ta-arroyo-p3-debug-cargo` with the
`p3-main7ac-` prefix; `p3-main7ac-preservation.json` records twenty-five
byte-preservation checks. Earlier manifests remain historical evidence;
the integrated source receives a separate freeze. No release build or
performance acquisition ran here. The three workloads, sample counts and
inclusive budget above are unchanged. The dedicated deadline/quiet/RSS/seal
adapter, compatible sealed release identities, delta review and final-head
CI/Codacy remain required. An existing cc6 release artifact is not relabeled
as main 7ac by this integration.
