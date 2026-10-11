# Issue 406: interval-Join eviction follow-up

## Profile evidence and selected direction

The new checkpoint-off profile identifies index eviction and output materialization
as comparable major CPU consumers. This supports investigating NativeIndex eviction
without weakening checkpoint durability. The paired comparison below finds no
confirmed regression in either mode, with lower observed medians but insufficient
evidence for the maintained rule's confirmed-improvement classification.

Baseline is `311b75d781f6585d2d6400b69068afbefb59c24c`. The fixed scenario uses
`interval_join`, 200,000 rows per side, and 8,192-row source batches. Checkpoint-off
sets `checkpointing=False` with no managed backend; checkpoint-on uses 100 ms.
The maintained `EngineCase` fixture/oracle and `checkpoint_cycles.measure_cycle`
provide the workload and complete finite-job timing boundary.

The predeclared profile consists of one warmup and ten CPU samples per mode,
plus one warmup and one strace sample for on: 24 finite jobs, executed once.
All jobs, process startup, fixtures, oracle checks, evidence extraction and cleanup
finished in **26.063 seconds**, within the 120-second profile allowance. No workload
was retried. The corrected final offline audit took another 0.652 seconds,
separately recorded (26.715 seconds combined). The unprofiled comparison retained
its separate 480-second allowance; the combined declared measurement limit is 600 seconds.

All 24 outputs matched the Arrow oracle at 2,198,080 rows. Every job completed with
`natural_end`, zero remaining tasks and task errors, ended sources/operators/sinks,
no checkpoint failure or in-flight epoch, and successful owned cleanup. All 11 off
jobs produced no checkpoint epochs or state files and reported BestEffort delivery.
All 13 on jobs retained evidence of nonempty nonterminal state and a terminal
manifest. The CPU-profile on jobs each ended at epoch 7; the traced warmup and sample
ended at epochs 9 and 8. Retention limits the surviving manifest inventory; these
counts do not certify a continuous 20-epoch workload.

### CPU attribution

Samply sampled at 1,000 Hz. The following percentages use only the ten measured
ready-through-EOF/owned-cleanup windows and all threads of the matching worker.
Fixture construction, plan/startup, warmup, post-job oracle and manifest inspection
are outside those explicitly recorded windows. Inclusive percentages overlap and
must not be added. Values are diagnostic CPU sample shares, not wall-time phases.

| Function or path                  | Off self | On self | Off inclusive | On inclusive |
| --------------------------------- | -------- | ------- | ------------- | ------------ |
| `NativeIndex::remove`             | 7.22%    | 4.42%   | —             | —            |
| `SideGather::new`                 | 6.20%    | 4.72%   | —             | —            |
| Arrow `take_bytes`                | 6.16%    | 5.26%   | —             | —            |
| `BTreeMap::insert`                | 5.63%    | 4.61%   | —             | —            |
| `NativeIndex::window_by_id`       | 4.72%    | 3.60%   | —             | —            |
| `evict_opposite`                  | 4.19%    | 4.33%   | 18.77%        | 21.24%       |
| `materialize_output_record`       | —        | —       | 18.87%        | 14.67%       |
| `prepare_batch`                   | —        | —       | 17.08%        | 13.36%       |

There are 2,840 off and 3,668 on in-window weighted samples. Unresolved native leaf
frames are 0 of 2,194 off native leaves and 0 of 2,861 on native leaves. This does
not imply source-line/inlining attribution: the release binary has function
symbols, and some work is aggregated under generic or large functions.

On-mode checkpoint-related inclusive samples include `PendingLog::push` 3.00%,
`run_live_checkpoint_task` 2.10%, `encode::pending_owned` and `Encoder::payload`
1.72% each, `compact`/`compact_column` 1.55% each, and manifest validation 1.47%.
These overlapping CPU stacks show encoding and bookkeeping work but cannot
partition asynchronous barrier alignment, publication or commit waits. Public
Python status does not expose cumulative checkpoint phase durations.

The profiled full-cycle wall medians were 257.78 ms off and 899.08 ms on; process
CPU medians were 327.44 ms and 440.86 ms respectively, including Tokio worker
threads. Profiling perturbs execution, these modes ran sequentially, and CPU can
exceed wall time. These figures are not an off/on causal estimate or an unprofiled
performance verdict.

### Window validation and analysis repair

The first postprocessor incorrectly treated samply timestamps as relative to
`meta.startTime`; the original failed analysis and controller result remain in
the raw evidence. Samply actually recorded absolute Linux monotonic milliseconds.
The worker recorded Unix nanosecond phase boundaries using a paired clock offset,
but did not retain the original monotonic offset. Offline analysis therefore uses
a preserved post-capture clock calibration and assumes no realtime/monotonic
clock step between capture and calibration. This is a limitation of the diagnosis.

All mapped windows are ordered, disjoint and inside the matching Python worker
lifetime: PID 2245148 for off and 2245818 for on. A second reduction discards 10 ms
from both ends of each window. The off `remove` share remains 7.58%, gather 6.48%
and Arrow take 6.44%; on shares remain 4.52%, 4.87% and 5.38%. This supports the
hotspot selection despite small boundary uncertainty. It does not turn the
reconstructed timestamps into exact checkpoint phase measurements. No additional
workload was run to repair the analysis.

### Checkpoint-on I/O diagnosis

The single traced measured job has a 1.33949-second window. It contains 133 `fsync`
calls with summed overlapping call durations of 641.71 ms (largest call 8.46 ms).
Path decoding separates temporary state/manifest files from their directories:

| Fsync target                | Calls | Sum of call duration |
| --------------------------- | ----- | -------------------- |
| State directories           | 81    | 390.15 ms            |
| Temporary state segments    | 30    | 147.68 ms            |
| Temporary manifest files    | 8     | 36.35 ms             |
| Manifest directory          | 14    | 67.52 ms             |

The same window contains 38 writes totaling 2.09 ms, 38 `renameat2` calls totaling
2.47 ms and 30 renames totaling 1.69 ms. Concurrent syscall durations overlap;
their sums are not exclusive wall-time fractions. Strace also perturbs timing.
This identifies durability I/O as a material on-mode cost but does not prove a
safe fsync-removal strategy or provide separate barrier/encode/commit wall phases.

## Build, environment and raw evidence

The source was exported from the exact baseline commit. The existing release
dependency cache was reused after cleaning only the three local workspace crates
to avoid stale source-path reuse. Maturin used `--release --locked --offline
--strip false`; optimization, LTO and Rust flags were unchanged. Build plus clean
and packaging took 388.636 seconds, recorded separately from measurement. The
result has a native symbol table and native SHA-256
`0fc5a595b1bc65891b9a648d4ea9501d532577401fdbc365207bb2abbd25d2ec`.
The first symbol checker used the wrong checkpoint namespace; the corrected check
validated the same artifact without rebuilding. Cargo reordered two lockfile
entries; normalized package versions, checksums and dependency edges are equal.

The new binary's `.text` differs from the preceding stage's stripped wheel, so
historical timings and that old wheel are not used as this comparison's baseline.
The new baseline wheel and isolated site are retained for the candidate comparison
with the same build configuration.

The environment is WSL2, 32 logical CPUs and 32 Tokio workers, CPython 3.13.9,
NumPy 2.5.2 and PyArrow 24.0.0. Owned build/test processes were settled before
capture, but WSL2 host noise is uncontrolled. Execution was outside the sandbox
because an independently reproduced sandbox asyncio/thread delivery hang would
invalidate runs inside it. This is a diagnostic profile, not a contract-v2 timing
classification.

Raw profiles, symbol sidecars, all warmup/sample/status/oracle records, strace,
explicit timing windows, clock calibration, failed first analysis, corrected
summary, scripts, build provenance and SHA-256 seals live under
`target/issue406-next/` in the main checkout. Entry points are
`performance-plan.json`, `build-profile.json`, `profile-execution.json`,
`profiles-summary.json`, `profile-evidence-audit.json`, and
`profile-evidence-sha256.json`. Raw target artifacts are local rather than Git
payloads; this tracked report records the decision and its limits.

## Paired results and acceptance boundary

The declared comparison used only this new baseline and the frozen candidate,
checkpoint-off and 100 ms on, two rounds of ten alternating AB/BA pairs per mode,
and one warmup per worker per round: 80 samples plus eight warmups. It reused the
complete finite-lifecycle helper, Arrow oracle, fingerprint checks and maintained
paired-median confidence rule. No profiler, tracing or phase polling participated
in those timings. The single attempt took 80.598 seconds including all fixture,
startup, warmup, oracle, manifest inspection and cleanup work. Final postprocessing
took 0.069 seconds; profile capture plus the final profile audit and this paired
stage total **107.382 seconds** of the 600-second measurement allowance.

The candidate is the frozen baseline plus `candidate-source.patch` and the
untracked files preserved in `candidate-source.tar`. Build/export/clean took
375.413 seconds, with the same release settings as the new baseline; candidate
native SHA-256 is
`86ba1e3b82a3e1e34985c8db66cec9820a9ce6fcb52531c21a856c28f4bce976`.
An initial post-build checker mistakenly treated dependency-edge order as a graph
change. The original failure is retained; normalized edges were verified against
the existing wheel in 0.321 seconds without rebuilding. Both built Cargo locks
have the same SHA-256. Release configuration, Python adapters and maintained
harness sources match; all machine, dependency and workload fingerprints match
within each mode. Candidate source files still matched the export after execution.

| Mode   | Version   | Wall p50  | Wall p95  | Process CPU p50 |
| ------ | --------- | --------- | --------- | --------------- |
| Off    | Baseline  | 286.89 ms | 306.86 ms | 369.65 ms       |
| Off    | Candidate | 275.85 ms | 284.98 ms | 358.15 ms       |
| 100 ms | Baseline  | 888.11 ms | 938.65 ms | 481.84 ms       |
| 100 ms | Candidate | 876.55 ms | 921.52 ms | 477.69 ms       |

| Mode   | Round | Paired median change | Paired-median confidence interval |
| ------ | ----- | -------------------- | --------------------------------- |
| Off    | 1     | -3.40%               | [-6.03%, +0.04%]                  |
| Off    | 2     | -3.34%               | [-8.29%, -1.42%]                  |
| 100 ms | 1     | -1.75%               | [-14.27%, +2.02%]                 |
| 100 ms | 2     | -1.99%               | [-6.86%, +2.76%]                  |

Each round has ten aligned pairs; the maintained order-statistic interval has
97.85% coverage under its independence/common-distribution assumptions. Alternation
does not establish those assumptions on WSL2. The maintained two-round +5% rule
returns `no-confirmed-regression` for both modes. Neither meets its confirmed
improvement threshold; these observations do not prove equivalence or a general
3–4% speedup. Ratios of pooled medians are not substituted for paired estimates.

All 88 jobs, including warmups, passed the 2,198,080-row oracle, natural completion,
zero task/resource errors, ended resources, checkpoint settlement and cleanup.
All 44 off jobs had zero checkpoint files/bytes/epochs and BestEffort delivery.
All 44 on jobs had nonempty nonterminal and terminal manifest evidence. All eight
workers exited successfully. No samples were discarded and no cases or runs were
added. Detailed phase quantiles, retained checkpoint bytes, worker-lifetime RSS
and every sample are retained in `summary.json`, the eight worker JSON files,
`paired-evidence-audit.json`, `provenance.json`, and `evidence-sha256.json` under the
same raw artifact root. These finite-job diagnostics preserve the scope below.

The off and on modes require separate timing verdicts. This finite scenario does
not certify Polars-relative targets, a 1-second checkpoint configuration, durable
restore performance, or 20 continuous nonterminal epochs with nonempty state.
Such acceptance needs an explicitly scoped sustained workload and retained phase,
quantile, checkpoint-byte, RSS, restore and provenance evidence. Repetition of
fresh finite jobs does not substitute for that contract.

## Implementation and focused verification

`KeyRun::remove` compares the requested identity with the first active entry
before using the existing binary search. Prefix expiration advances the start
offset directly; arbitrary-order removal retains the original search and shift.
`NativeIndex::remove` still updates the moved dense-row position, refreshes the
canonical key owner and decrements the row count on every removal. When the key
remains live, its backing capacities are unchanged, so it skips the redundant
resident-byte publication and refund. Removing the last row of a key or the whole
index retains the original capacity release and exact refund. Append-credit
commit/abort, checkpoint bytes, state charges and delivery semantics are unchanged.

The existing hot-run test gained explicit work-count assertions before the
production edit. On the old implementation, 512 prefix removals performed 5,632
identity comparisons and 512 refund calls; the focused test failed against the
expected 512 comparisons and zero redundant refunds. The candidate passes those
assertions. Its full 1,536-row expire/refill cycle performs one refund, on final
key removal. Added reserve/abort assertions verify exact funding after abort.

Two focused Join tests pass: the dictionary test covers collisions, reused key
IDs, sorted dense-row mappings, owner migration and the prefix work bound; the
allocation test checks 4,096 rows with one, 17 and 4,096 keys, ordered/unordered
inserts, ordered/random removals, empty-run reuse, allocation peaks, exact funding
and final pool release. These are controls within two existing tests, not new
performance cases. Core Clippy with `--lib --tests --all-features --locked
--offline -- -D warnings` and focused formatting/whitespace checks pass.

The same increment resolves the earlier generated libtest stack-array lint by
moving 31 unchanged public RollingSpec declaration tests to an integration target.
All 31 and three directly affected unit controls pass; the existing test and
coverage runners include that target automatically. The unit harness now contains
2,035 tests. See the checkpoint-mode report's follow-up for migration evidence.

## Checkpoint-on follow-up assessment

Source inspection supports evaluating batched publication of the segments from
one operator snapshot in one epoch. Managed execution stages, validates and
publishes these segments in `state/transaction.rs::stage_operator_state_locked`
before the operator acknowledges the barrier. The later checkpoint manifest
publication receives an empty `staged_segments` collection; optimizing only that
manifest-stage loop would miss this path.

The narrow candidate is to retain every segment's file sync, staging-directory
sync and validation, then rename all new segments and sync each affected committed
directory once before returning working pins or acknowledging the barrier.
Single-segment and other-backend behavior must retain their current completion
guarantees. This proposal has not been implemented or timed.

The durability sequence must remain: segment contents durable, validated segment
renames and target directories durable, operator acknowledgement, manifest contents
durable, manifest installation and directory durable, then sink commit and source
checkpoint acknowledgement. Retention separately makes old-manifest deletion
durable before garbage-collecting unreferenced segments. Initial directory
creation also syncs ancestors. These distinct operations explain why the 81
state-directory and 14
manifest-directory calls are not all redundant publication syncs.

A batch implementation needs focused failure tests for partial renames, directory
sync failure and retry, cancellation settlement, newly created paths and multiple
target directories. In particular, a retry must not treat an already-visible
matching file as proof that its directory was synced. Existing controls in
`state/local.rs`, `state/transaction.rs` and
`runtime/streaming/runner/tests.rs` cover segment validation, manifest
installed/unknown outcomes, cancellation and retention ordering, but do not by
themselves prove a new batched publication window safe. No fsync or recovery
contract is changed by this eviction increment.

## Final specialist review

The final specialist review approved the incremental code, test migration,
documentation and evidence with no outstanding findings. It independently checked
all 88 paired jobs, both native seals, 982 candidate source/config/harness hashes,
all 77 final evidence entries and all 44 profile evidence entries. Required CI,
coverage and cross-platform checks remain separate merge gates; this review is
not a merge-ready claim.
