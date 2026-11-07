# Key-sharded grouped SUM updates - measured and withdrawn

Status: withdrawn after measurement. The runtime prototype lives on branch
`feature/sum-key-sharding` for reproduction; no runtime change is delivered.
This closes the second parallel-aggregate route after the rejected row-range
lane study [sum-batch-parallel-20261104](sum-batch-parallel-20261104.md)
(dense 4CPU +15.75%, 8CPU +18.60%) and follows
[sum-key-sharding-20261107 spec](../../.codex/artifacts/specs/sum-key-sharding-20261107.md).

## What was built

The design the spec pre-registered: route every input row by a fixed-seed
hash of its complete group key into shard sub-records (`arrow` take), run
each shard's chunk update as one `ParallelCpuWork` unit on the job-scoped
native CPU service (owned local `PartialGroups` + local key index), and fold
fragments back on the operator task by resolving global slots in original
first-appearance order (byte-identical snapshots, no per-group merge because
shards own disjoint groups) before merging each fragment's partial states
once. Admission: inline integer SUM, plain-column keys and arguments, no
predicates or filters, unbounded incremental pool, rows >= 64,000,
`min(cpus, rows/16,000) >= 2` units. Unit tests proved dispatch, serial
fallback below admission, and row-identical snapshots against the serial
path across two batches (3,000 groups, 70,000 rows).

## Measurement

Paired AB/BA, two rounds of five alternating pairs, `taskset -c 0-3`,
baseline = `main@2193f8ee` binary, candidate = the branch binary, streaming
stock-positions example 30 (checkpoints disabled, arithmetic oracle in every
run), verdicts by the maintained two-round paired-median rule (+5%):

| Case                          | Baseline P50 | Candidate P50 | Change  | Rounds            | Verdict      |
|-------------------------------|--------------|---------------|---------|-------------------|--------------|
| 100,000-row batches, 10M rows | 1227.0 ms    | 1351.5 ms     | +10.15% | (2.49, 12.81) / (5.96, 32.87) | inconclusive (leaning regression) |
| 1,000,000-row batches, 10M    | 1245.5 ms    | 1350.5 ms     | +8.43%  | (0.67, 12.25) / (6.86, 12.82) | inconclusive (leaning regression) |
| 1,000-row batches, 2M rows    | 1876.5 ms    | 1647.0 ms     | -12.23% | (-5.56, 5.74) / (-30.11, 5.61) | inconclusive (below admission) |

Neither large-batch case improved; both leaned slower with a second-round
lower bound above +5%. The pre-registered acceptance (improved at 100k-row
batches) failed, so the change is withdrawn.

## Why the serial overhead ate the parallel win

The zero-cross-group-merge design removed the lane study's +37% total-CPU
failure mode, but three serial additions remain per batch: the routing pass
(hash plus a five-column `take` copy of every row), the per-fragment Arrow
state round-trip (`emit_states` plus `merge_batch` for every distinct group),
and slot resolution for every touched group. With the inline-SUM state,
snapshot key-column cache, and template work already landed, the update that
sharding can parallelize is a minority of each batch's serial budget, so
4-thread parallelism cannot pay for the routing and state round-trips. Both
parallel routes for this operator are now measured rejections; the serial
path with its landed optimizations remains the delivered behavior.

## Reproduction

```bash
git checkout feature/sum-key-sharding
cargo build -p calc-flow --release --example 30_streaming_stock_positions
# baseline: main@2193f8ee built the same way
taskset -c 0-3 <binary> --rows 10000000 --batch-size 100000
cargo test -p calc-flow --release --lib grouped_key_sharding
```
