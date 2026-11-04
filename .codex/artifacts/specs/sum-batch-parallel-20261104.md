# Single-Batch Parallel Integer SUM - Specification

## Outcome

The prototype at public revision
`fe0e86ad220bd88ae3aeb1ae6c384e3a72e0a792` passed its focused functional
review but failed the fixed performance acceptance below: dense four- and
eight-CPU cases confirmed regressions. The archive delivery therefore preserves
the target branch's serial SUM implementation; its final tree is checked
against merged `d5e05c31110ebb1e6bfb8fa3d63359f50010f16e`. The
requirements below describe the rejected prototype, not a shipped parallel SUM
contract. See the
[experiment decision and sealed results](../../../benchmarks/sum-state/sum-batch-parallel-20261104.md).

## Source

- User request on 2026-11-04: continue performance work under the previous
  profiling, measurement, independent review, CI repair, and merge requirements.
- Base: `c9afa829ff025280ab346667582c62dc26a34785` (merged PR #534).
- Contracts: [introduction](../../../docs/introduction.md),
  [SQL streaming boundaries](../../../docs/streaming-guide.md#stream-ownership-and-sql-boundaries),
  [SQL recovery](../../../docs/streaming-guide.md#sql-aggregate-recovery-compatibility),
  [runtime ownership](../../../docs/runtime-envelope.md), and
  [delivery policy](../../agents/README.md#shared-delivery-policy).
- Existing implementation: [incremental SQL](../../../crates/calc-flow/src/operator/sql/incremental.rs),
  [integer SUM selection](../../../crates/calc-flow/src/operator/sql/grouped_sum.rs),
  [native input predicates](../../../crates/calc-flow/src/operator/sql/predicate.rs),
  [owned CPU work](../../../crates/calc-flow/src/runtime/streaming/gather_work.rs), and
  [parallel CPU work](../../../crates/calc-flow/src/runtime/streaming/gather_work/parallel.rs).
- Prior evidence: [PR #534 measurements](../../../benchmarks/sum-state/sum-followup-columnar-20261104/README.md).
- Accepted implementation restrictions: [preimplementation critique](../critiques/sum-batch-parallel-20261104.md),
  commit `f1d9ce645e2a7119c9da300d8892cd8b7d971221`.

## Problem Statement

The current compact grouped integer SUM processes each original input `Batch`
serially, even when the streaming job has multiple available CPUs. The latest
candidate profile attributes 45.33% of 1,306 selected user-CPU samples to the
update path; finishing and committing remain material costs. Inclusive shares
overlap, 29.48% of leaf samples are unresolved, and these observations do not
establish wall-time limits or a parallel speedup. This work must demonstrate
useful multicore improvement while preserving transactional stream semantics
and the prior sparse single-CPU improvement.

## Goals and Non-Goals

- Admit bounded parallel local partial aggregation within one original `Batch`
  for the narrow built-in grouped integer SUM shape below.
- Preserve each accepted input prefix's complete snapshot, ordered group slots,
  state, metadata, budgets, and recovery identity.
- Prove bounded simultaneous CPU work, complete cleanup, and meaningful
  elapsed-time improvement using the fixed measurement contract below.
- Exclude cross-batch concurrency, persistent group sharding, retained-input
  SQL optimization, finite DataFusion execution, and new executors.
- Exclude new public APIs, execution knobs, project/schema fields, Python
  binding changes, checkpoint layouts, and changes to floating-point,
  decimal, ordered, distinct, custom aggregate, or UDF behavior.

## Functional Requirements

- **FR1 — Eligibility.** Only compact append SQL with nonempty grouping and
  exclusively DataFusion's concrete built-in non-distinct, unordered SUM
  aggregates returning `Int64` or `UInt64` may use this route. Every SUM
  argument and grouping expression must be an actual bound input `Column`,
  without an implicit computed physical expression, and `input_checks` must
  be empty. Normalization may retain fallible eager checks even after removing
  their projected expressions; simple final column syntax is insufficient.
  Multiple eligible SUM columns are allowed. Input key types must already be
  admitted by the serial compact implementation, including the measured two
  UTF-8 key shape. Mixed eligible/ineligible aggregate lists remain serial.
  Shapes already using the single Boolean/integer-key `NativeKeys` fast path
  retain that serial path instead of switching to generic parallel grouping.
  Initial parallel admission additionally requires the exact concrete
  `UnboundedMemoryPool`; every bounded or custom scratch pool remains serial.
- **FR2 — Expression boundary.** Absent predicates/filters are allowed.
  `WHERE` and aggregate `FILTER` may run in workers only when their complete
  expression is accepted by the existing corresponding native allowlist and
  a stricter infallible parallel subset: input columns/literals, supported
  comparisons, `NOT`, and null tests. Physical `AND`/`OR` remain serial because
  DataFusion 55 short-circuit evaluation can copy complete input records,
  including unrelated variable-width columns. Reject CAST/TRY_CAST, CASE,
  division, UDFs/callbacks, and any unproven expression. The current WHERE
  allowlist alone is insufficient because it admits casts. Inspect the complete
  bound physical predicates, including inserted implicit casts, rather than
  relying only on logical syntax. Eligibility must fail closed before launching
  work; there is no new expression admission or Python execution. Existing
  projection, HAVING, ORDER BY, and LIMIT handling remains on its current
  coordinator/output path.
- **FR3 — Scheduling.** All original input Batches remain sequential
  transactions. Within one Batch, use the existing job-owned
  `gather_work`/`ParallelCpuWork` lifecycle and effective CPU allowance. Require
  at least 32,768 original rows and four original chunks. For chunk count `C`,
  select `U = min(8, effective gather parallelism, floor(C / 2))` and require
  `U >= 2`. Unit `i` owns chunk ordinals in the half-open range
  `[floor(C*i/U), floor(C*(i+1)/U))`, giving at least two complete chunks per
  unit and at most eight queued results. Empty/tiny and one-CPU batches remain
  serial. Test these internal admission boundaries; they are not new public
  settings. Do not add unbounded spawning, a second CPU executor, or long
  blocking CPU work on Tokio threads.
- **FR4 — Deterministic reduction.** Worker-local grouping and partial SUM
  state do not own committed/global group slots. The coordinator alone
  resolves global keys, assigns new slots in first-appearance order over the
  original record-batch/chunk/row order, incorporates saved history exactly
  once, and finalizes candidates. Retain original `CHUNK_ROWS` boundaries;
  units own contiguous ranges of complete original chunks. Units may execute
  concurrently; within each unit preserve chunk order and each chunk's legacy
  reserve/selection/input-check/key/argument/filter stage order. Do not run
  whole-Batch eager stage prechecks ahead of their chunks' key/argument stages.
  The coordinator visits units and their first-appearance keys in original
  ordinal order and alone applies global group/state/output limits. Completion
  order must not change results, group order, checkpoint identity, or serial
  ordinary-error precedence. Unsupported fallible expressions remain serial;
  optional prelaunch proof failures decline to serial rather than overtaking
  an ordinary error. Unexpected worker internal errors/panics retain the gather
  service's deterministic ordinal precedence, non-cancellation-error priority,
  and secondary-error reporting.
- **FR5 — SQL semantics.** Preserve signed/unsigned wrapping addition,
  null-only groups, duplicate keys, null keys, per-aggregate FILTER selection,
  SQL three-valued Boolean selection, empty input, and unchanged groups.
  WHERE-rejected rows do not introduce groups; a WHERE-selected group exists
  even if all its SUM filters reject every row. Each original accepted Batch
  still emits its existing complete cumulative snapshot, rather than worker
  partials. Caller Arrow buffers and previously emitted snapshots remain
  immutable. Preserve schema, field metadata, Batch metadata, and current
  HAVING/order/limit behavior.
- **FR6 — Admission and ownership.** Before allocating or launching any
  parallel-only structures, obtain checked conservative capacity credit for
  all simultaneously owned input backing buffers, references, control
  structures, encoded keys, local group maps/ranks, partial accumulator buffers,
  filter workspace, result buffers, and coordinator merge overlap. Include variable-width key
  data, capacity growth, and checked arithmetic. Credit remains owned until
  the covered data is destroyed or transferred to another credited owner;
  releasing a work ticket must not leave uncharged live results. Retain the
  original input backing reservation through the last worker/result reference
  to its buffers; charging only Arrow/Arc handles is insufficient. Without an
  input owner, conservatively fund the retained backing buffers or decline to
  serial before launching. Worker-local keys, maps, partials, results, and
  controls are engine scratch. Coordinator candidates, global partials, dirty
  state, captures, and containers retain their existing classifications and
  charges. Unbounded pool admission still requires checked real reservations;
  these reservations are not an RSS cap.
- **FR7 — Fallback and global limits.** If extra parallel credit is
  unavailable, choose fewer units or the existing serial path before workers
  launch. Do not turn an otherwise serial-admissible input into a memory
  failure solely because optional parallel scratch cannot fit. Never retry
  already-started work serially after a worker failure. For this initial route,
  every bounded/custom scratch pool must choose serial before any optional
  parallel allocation or submission, for both cold and warm gather owners.
  Do not alter generic gather funding or retain extra home/generation credit
  on that fallback path. Count compact
  `max_rows` against global distinct live groups, including null-only groups,
  rather than summing worker-local duplicates. Existing cumulative input,
  compact byte, engine memory, and edge output limits still apply without
  multiplying or bypassing them.
- **FR8 — Transaction and cleanup.** Workers may mutate only their owned
  local data. No committed group, key index, history, ledger, dirty state,
  sequence, or checkpoint capture changes before successful output acceptance
  at the existing commit boundary. Cancellation, timeout, worker error/panic,
  launch/admission failure, merge failure, and output refusal must discard the
  entire candidate and preserve committed state. Await running workers and
  settle their result/input owners before returning failure from the owned
  lifecycle; dropping an interrupted future must transfer cleanup to the
  existing job owner. Settle individual attempts through the existing ticket,
  cleanup, and retirement lifecycle; do not close the reusable job owner after
  each Batch. Job close must drain that owner and release all credit. Workers
  check the existing stop token at bounded work intervals.
- **FR9 — Control and recovery.** A Barrier, EOF, next Batch, or capture may
  not overtake outstanding work for the current Batch. Full/delta captures,
  restore/continue, and reset retain existing semantics and compatibility.
  Recovery must not require worker-local state or a particular CPU count.

## Performance and Measurement Contract

Before execution, the performance owner must freeze a protocol identifying
baseline/candidate source, binary and harness hashes, fixture hashes, exact
case dimensions, admission rule, affinity/topology, sample counts, profiling
scope, and budget. Builds are estimated and recorded separately. A prior
`af400b38511bc7f6255fd3549518095b51b20e0c` baseline artifact may be reused only
with its actual identity and a verified receipt proving compatible compiled
source, compiler, lockfile, features, flags, and common harness against the
merged base; it must not be relabeled as a fresh base build. Freeze and publish
the candidate product revision before its performance build. Prior evidence
directories remain immutable.

Select exactly these five cases; do not add a Cartesian matrix:

| Case                   | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| Dense stream, 1 CPU    | Serial admission and overhead control against the merged base. |
| Dense stream, 4 CPUs   | Intermediate useful parallelism and reduction overhead.        |
| Dense stream, 8 CPUs   | Target multicore improvement under the effective CPU limit.    |
| Sparse stream, 1 CPU   | Preserve the prior sparse retained-state gain.                 |
| Hot-key stream, 8 CPUs | Detect skew sensitivity of local work and reduction.           |

Reuse maintained dense/sparse fixtures and snapshot retention/oracles where
compatible. Freeze the hot-key distribution and independent arithmetic oracle
before measurement; match the dense row/batch schedule and preserve a retained
group tail so full-snapshot costs remain visible. Fixture adaptation must use
the same audited measurement instrumentation on both products.

Each case uses two rounds of ten alternating baseline/candidate pairs, with
one warmup per arm per round. One 600-second wall-clock budget covers all five
cases, fixtures, preflight, startup, warmups, correctness, profile capture and
decoding, analysis, publication, and cleanup. Stop and preserve partial evidence
on deadline, source drift, invalid environment, or oracle failure. Do not add
cases, samples, or timing-driven retries. Uncompleted cases remain unmeasured.

Use [the maintained paired-median decision rule](../../../scripts/benchmark_suite/report.py):
`regression` requires both round interval lower bounds above +5%; any round
upper bound above +5% otherwise yields `inconclusive`; both upper bounds below
−5% yield `improved`; the remaining result is `no-confirmed-regression`, with
the maintained endpoint tolerance. Preserve all round intervals and observations.
Do not require every interval to exclude the +5% boundary to claim absence of
a confirmed regression, and do not describe inconclusive results as equivalence.

Performance acceptance requires `improved` on at least one of the dense
4/8-CPU target cases and no `regression` verdict on any completed selected
case. All five cases must complete for full acceptance. Higher CPU occupancy
alone is insufficient. Report wall time, total process CPU, average cores,
whole-process peak RSS, and unchanged per-snapshot correctness for every case.
Obtain accepted on-CPU profile evidence for representative multicore processing,
using verified processing windows/timestamps, source/binary identities,
per-thread attribution, unresolved samples, and overlapping-share caveats.
Attribute local update, coordinator merge, finish/output, and commit costs as
the available evidence permits; estimates remain hypotheses. If the criterion
is unmet, report the prototype's limits without claiming a speedup or expanding
the experiment to seek a favorable result.

## Acceptance Criteria and RED/GREEN Evidence

Record each new behavior test's expected failure on the base before its
implementation, followed by the smallest focused GREEN checks. Existing
passing controls need not be rerun unchanged.

- [ ] An eligible large single Batch with at least two effective CPUs executes
  at least two local work units simultaneously; a deterministic stop-aware
  probe also proves the eight-unit maximum is never exceeded. The 32,768-row,
  four-chunk and two-CPU admission boundaries match FR3's exact unit formula
  and contiguous ranges. Empty/tiny and one-CPU cases take the serial path
  without waiting on a multi-worker-only test barrier.
- [ ] Signed and unsigned multi-chunk/multi-record-batch input matches an
  independent modular prefix oracle after every original Batch, including
  multiple SUMs, wrapping boundaries, nulls, WHERE/FILTER distinctions,
  Unicode/NUL keys, repeated keys across units, and held earlier snapshots.
  Deliberately reversed unit completion preserves first-appearance group order.
- [ ] Mixed SUM/COUNT, float, decimal, distinct/ordered/custom/UDF and computed
  SUM/key expressions, retained eager input checks, and explicit/implicit
  cast/CASE predicates remain serial. Multiple
  deliberately failing chunks retain the serial first ordinary error and
  chunk-stage precedence. Existing float prefix bit-pattern and
  ordered/checkpoint controls remain valid where touched.
- [ ] Checked capacity overflow declines before covered allocations/work
  launch. A bounded one-byte-short parallel-credit control still succeeds
  when the original serial path fits. Cold and warm bounded/custom pools enter
  no worker and retain no optional gather funding. Missing input backing
  ownership either obtains complete buffer credit or declines to serial
  before launch.
  Delayed results and coordinator overlap remain charged until destruction or
  ownership transfer. A duplicate-key-across-units fixture hits the same global
  group-limit boundary as serial processing.
- [ ] Injected cancellation, deadline, worker panic/error, launch failure,
  merge failure, and output refusal leave the prior encoded state/capture and
  input ledger unchanged. A permitted retry applies the Batch once. Held input
  owners, pending result owners, job workers, and reservations all settle after
  close; tests have finite deadlines and cannot deadlock on serial admission.
- [ ] A managed checkpoint Barrier behind active work cannot complete early;
  restore/continue at a different CPU allowance matches uninterrupted prefix
  results and ordering. Existing checkpoint identity/layout remains accepted,
  and reset returns to the existing empty-state semantics.
- [ ] The five-case protocol, actual raw paired observations, oracle receipts,
  CPU/RSS data, source/build seals, budget debit, accepted profile analysis, and
  honest per-case verdicts are published under this work item's shared slug.
  Full performance acceptance satisfies the criterion above.
- [ ] Relevant focused SQL/gather lifecycle tests and necessary module
  formatting/lint checks pass. `git diff --check` passes and
  `git diff --exit-code -- schemas/project-v3.schema.json` shows no contract
  drift. Update affected runtime/performance documentation without new knobs.
- [ ] An independent pre-implementation concurrency review and final specialist
  review have no unresolved blockers. Required CI, Codacy, coverage (including
  the existing 90% workspace line floor), and cross-platform checks finish
  green before the already-authorized merge; pending CI is not merge readiness.

## Final Disposition

Independent source review and 70 focused/affected Rust tests verified the
prototype's functional and ownership boundaries. The frozen five-case
measurement then rejected its performance. The runtime prototype is preserved
at the public revision above, while the delivery restores the merged serial
runtime and publishes the tools and sealed evidence. The
[canonical review](../reviews/sum-batch-parallel-20261104.md) records the
separate source, measurement, and archive-delivery decisions. Required remote
CI remains a condition for merging that archive delivery.
