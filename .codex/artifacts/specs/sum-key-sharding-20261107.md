# Key-sharded compact grouped SUM updates - Specification

Status: draft for critique. D = shard the per-batch incremental update of one
`IncrementalSql` by full group key on the job-scoped native CPU service.
This spec unseals the parallel-aggregate question that
[sum-batch-parallel-20261104](../../benchmarks/sum-state/sum-batch-parallel-20261104.md)
closed for the row-range lane design, and follows the sharding direction its
own review and the example-30 multicore investigation both recommended.

## Source

- Code (under `crates/calc-flow/src/operator/sql/` unless noted):
  `incremental.rs` (`IncrementalSql`, `update_records`, `update_chunk`,
  `prepare_transaction`, `commit`, `Transaction`, `output_count`),
  `group_aggregates.rs` (`InlineSum`, `GroupAggregates`),
  `grouped_sum.rs`, `dirty.rs`, `compact/capture.rs`,
  `compact_state/export.rs`.
- Rejected precedent: `benchmarks/sum-state/sum-batch-parallel-20261104.md`
  (lane variant, dense 4CPU +15.75% / 8CPU +18.60%, total CPU +37%,
  "does not prove that all batch-parallel SUM designs regress") and its
  review `.codex/artifacts/reviews/sum-batch-parallel-20261104.md`.
- Investigation: `benchmarks/stock-positions/multicore-investigation.md`
  (single compact state owner; recommendation 3 is exactly this design).
- Work evidence: `benchmarks/stock-positions/six-engines-dc8393d2.md` -
  streaming reaches CPU-work parity with partitioned SQL at large batches
  (1M-row batches: streaming 0.68 CPU-s at 1.37 average cores vs calc SQL
  0.60 CPU-s at 2.71 cores), so the remaining 2.2x wall gap at large batches
  is parallelism, not total work. Post-template and key-column-cache
  changes, small batches are already at parity or better.
- Native CPU service: `runtime/streaming/gather_work/parallel.rs`
  (`ParallelCpuWork`, `submit_parallel_work`, ordinal-ordered outputs);
  admission-threshold precedent: join keys >= 64,000 rows and 16,000 rows
  per unit (`operator/join/key_dictionary/parallel.rs`).
- Rust internals only: no public API, Python, `_native.pyi`, project-v3, or
  checkpoint-format change.

## Motivation

`IncrementalSql::update_records` applies every input chunk to one
`index + groups` state on the operator task. After the inline-state,
template-reuse, and key-column-cache work, this single-threaded update and
its serial routing are the last large-batch cost that scales with input
rows while the host has idle capacity the job already owns. The rejected
lane design parallelized by contiguous row ranges, so every lane rebuilt
partial state for every touched group and a serial coordinator merged
per-group fragments; that is where its +37% total CPU came from. Sharding by
the complete group key has no cross-group merge at all: each group lives in
exactly one shard, candidate counts do not multiply, and the only new work
is one routing pass and an ordered concat.

## Non-Goals

- No change to the output contract: every original input batch still emits
  one complete cumulative snapshot; snapshot frequency, row order, schemas
  and metadata are unchanged.
- No change to `LiveProgressCoordinator`, edge SPSC/FIFO contracts, barrier
  alignment, sink delivery, or checkpoint wire formats (`DirtyGroups`,
  `GroupLog` base/delta frames, `QuotaLedger` all keep their shapes).
- No sharding of order-sensitive shapes: grouped-float sequential proofs,
  ordered-LAST, and any aggregate outside the inline integer-SUM admission
  keep the serial path.
- No new user knobs; admission follows worker capacity and batch size, as
  the multicore investigation recommended.
- No join/ASOF/rolling changes; no repartitioning of the input graph.

## Design

### D1 - Routing pass (operator task, serial, light)

For an admitted batch, `update_records` first evaluates the group key of
every input row as it does today, hashes the full key bytes with a
**fixed-seed** hasher, and gathers row indices into `S` sub-batches.
The routing pass assigns global group slots in first-appearance order, so
new groups join the shards in exactly the order the serial path would
produce; the committed `index` and `groups` vec are unchanged in shape and
order. Fixed seeding keeps shard assignment stable across restarts and
deterministic in tests.

### D2 - Sharded update (native CPU service)

The per-shard update is one `ParallelCpuWork` with `unit_count = S`:
unit `i` runs the existing per-chunk candidate/accumulator logic
(`update_chunk` internals) over shard `i`'s rows against a borrowed view of
its shard's groups. Because sharding uses the complete key, no group is
visited by two units and no per-group merge exists; unit outputs are
candidate maps, collected by ordinal.

### D3 - Admission

Sharding is admitted only when all hold: the aggregate shape is the inline
integer SUM admission (single trusted built-in, non-distinct, unordered,
integer SUM); `rows >= 64_000`; `active groups >= 4_096`;
`S = min(cpus, rows / 16_000, active groups) >= 2`; the native CPU service
has capacity; no refusal path was taken last batch (refusals fall back to
serial permanently for the job, mirroring `SerialReason` semantics). The
memory pool must cover the routing index copies and per-shard candidate
scratch before dispatch; a refused reservation runs the serial path.

### D4 - Deterministic finish and output

`finish_candidates`, `prepare_transaction`, and output construction stay on
the operator task with today's semantics, consuming per-shard candidates in
shard-ordinal order; emitted records and the snapshot key-column cache are
byte-identical to the serial path.

## Verified baseline contract (normative)

- Group slots are append-only and group keys never change
  (`incremental.rs` candidate/commit paths; `install_container` moves the
  same `Arc` keys in order); `import_native_state` requires an empty plan.
- `Transaction.records` are the complete cumulative snapshot per original
  input batch; refused output must refund exactly
  (`test_*_refunds*` suites).
- The native service collects `ParallelCpuWork` outputs by ordinal and
  supports `GatherStop` cooperative cancellation and per-work memory
  reservation (`gather_work/parallel.rs`, `gather_work.rs` funding).

## Functional Requirements

- FR1 - Byte-identical outputs. For any admitted schedule, every emitted
  snapshot equals the serial path's bytes, including chunk layout, key
  column sharing (`Arc::ptr_eq` observable), order, and metadata; proven by
  a property test running both paths on the same fixtures.
- FR2 - No candidate multiplication. Shard units never construct candidates
  for groups outside their shard; total candidate constructions per batch
  equal the serial path's (asserted by test counters).
- FR3 - Cancellation and deadlines. Every unit checks its `GatherStop` at
  chunk boundaries; a cancelled or failed unit surfaces the existing typed
  error, and no partial state is committed.
- FR4 - Backpressure and cooperation. Dispatch awaits the service ticket as
  today; the operator task yields between routing and finish; no Tokio
  worker thread blocks on the shards.
- FR5 - Failure atomicity. A unit panic reaches the existing panic
  location reporting (`NumericSite` pattern); state before the batch is
  untouched; the job fails without emitting a snapshot for that batch.
- FR6 - Memory. Routing copies and shard scratch are reservation-checked
  before dispatch; a refusal runs serial; the pool returns exactly to the
  pre-batch baseline after commit (extends the refund suites).
- FR7 - Checkpoint compatibility. Capture/export paths are untouched; a
  checkpoint taken between batches is byte-identical to the serial path's.
- FR8 - Diagnostics. Metrics record admitted sharding, `S`, rows, active
  groups, and the serial fallback reason; no public metric fields change
  shape (additive counters only).

## Non-Functional Requirements

- Performance, pre-registered acceptance: paired AB/BA on the streaming
  stock-positions workload (100x100 keys, checkpoints disabled, pinned
  four CPUs, two rounds of five pairs): 100,000-row batches improve wall
  time with verdict `improved`; 1,000,000-row batches reach within 10% of
  the calc-SQL state-feedback wall time at no more than 5% additional child
  CPU versus the serial baseline; 1,000-row batches (below admission) show
  no confirmed regression; dense single-CPU, sparse, and hotkey cases show
  no confirmed regression.
- Budget: the five-case acceptance set runs within the default ten-minute
  measurement budget on one machine; CI engine jobs show no confirmed
  regression.

## Acceptance criteria

- [ ] AC1 (FR1, property): serial-vs-sharded byte equality across dense,
      sparse, hotkey, empty, null-key, and new-group-late schedules,
      including two consecutive checkpoints.
- [ ] AC2 (FR2, unit): candidate-construction counters equal the serial
      path's on a dense fixture.
- [ ] AC3 (FR3/FR5, fault): injected unit panic and cancellation preserve
      pre-batch state, fail with the typed error, and allow retry.
- [ ] AC4 (FR6, unit): refused routing/scratch reservation runs serial and
      leaves the pool at its baseline after commit.
- [ ] AC5 (FR7, integration): checkpoint-restore cycles round-trip sharded
      jobs identically to serial jobs.
- [ ] AC6 (NFR): the pre-registered paired measurement, with per-case
      verdicts, recorded in `benchmarks/sum-state/`.

## Rollback and compatibility

Crate-internal; admission gating makes the serial path the default and the
fallback. Revert restores today's behavior with no data or format
migration.

## Questions for the critic

- Is the admission floor (64k rows / 4,096 groups / 16k rows per shard)
  right for the maintained dense case, or should the first measurement
  sweep floors before freezing them?
- Should the routing hash reuse the existing key bytes only, or also admit
  dictionary-encoded keys without full materialization?
- Does per-job permanent serial fallback after one refusal match the
  entity-work precedent, or should it retry admission per batch?
