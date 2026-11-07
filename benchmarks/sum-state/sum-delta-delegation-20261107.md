# Delegated delta aggregation for compact SUM - measured and withdrawn

Status: withdrawn after measurement. The runtime prototype lives on branch
`feature/sum-delta-delegation` for reproduction; no runtime change is
delivered. This closes the third parallel-aggregate route, after the
rejected row-range lane study
[sum-batch-parallel-20261104](sum-batch-parallel-20261104.md) and the
withdrawn key-shard study
[sum-key-sharding-20261107](sum-key-sharding-20261107.md).

## What was built

Exactly the route the core-count sweep bounded: when a compact-SQL batch is
large (>= 100,000 rows), its shape is the inline integer SUM over plain
columns, and the fixed partition policy yields at least two partitions, the
per-batch delta aggregation runs as one partitioned `DataFusion` `GROUP BY`
on a dedicated delegated session - kept separate because plan-time schema
analysis freezes the node's main session at a zero-row decision - with the
physical-plan template reuse of #557 holding it to a single plan per node.
The grouped result (one row per touched group, ~10,000 rows on the stock
workload) is rebuilt as a mini-batch carrying the exact input schema and
fed through the ordinary incremental kernel, so state, checkpoints,
snapshots, and refund contracts are untouched; every failure path falls
back to the serial kernel on the original batch. Unit tests proved
dispatch, fallback, and row-for-row content equality against the serial
kernel; the red state was recorded by raising the admission floor.

## The mechanism works - and is fast

Instrumented runs showed the delegated aggregation itself is not the
problem: 800,000-row batches aggregated to 10,000 groups in **6.7-8.9 ms**
at four pinned partitions, versus roughly ten milliseconds for the serial
inline kernel on the same rows. The route had none of the earlier
prototypes' failure modes.

## Measurement

Paired AB/BA, two rounds of five alternating pairs, `taskset -c 0-3`,
fixed `p4` configuration, baseline = `main@9a75a71f` binary with the same
probe (`streaming_delegation_probe`, example-30 pipeline, checkpoints
disabled, arithmetic oracle every run), candidate = the branch binary:

| Case                          | Baseline P50 | Candidate P50 | Change  | Verdict                   |
|-------------------------------|--------------|---------------|---------|---------------------------|
| 1,000,000-row batches, 10M    | 1255.5 ms    | 1230.0 ms     | -2.03%  | no-confirmed-regression   |
| 100,000-row batches, 10M      | 1135.0 ms    | 1135.0 ms     | +0.00%  | no-confirmed-regression   |
| 1,000-row batches, 2M (below floor) | 1213.0 ms | 1212.0 ms    | -0.08%  | no-confirmed-regression   |

No case reached the pre-registered `improved` verdict, so the change is
withdrawn.

## Why: the aggregation is no longer the bottleneck

The two sharding prototypes failed because their coordination overhead
exceeded the parallel win. This route failed for the opposite reason: the
inline-state work of #542/#548 already drove the serial kernel's per-batch
aggregation cost down to roughly ten milliseconds per million-row batch,
about what the partitioned delegation itself costs. With nothing material
left to save on the aggregate, the remaining large-batch streaming/SQL gap
lives in the pipeline's other serial stages - the upstream expression
node, snapshot emission, and progress coordination - not in aggregation.
All three parallel-aggregate routes are now measured; the serial inline
kernel remains the delivered behavior, and any future large-batch work
should target those serial stages instead.

## Reproduction

```bash
git checkout feature/sum-delta-delegation
cargo build -p calc-flow --release --example streaming_delegation_probe
DELEGATION_DEBUG=1 DELEGATION_PROBE_PARTITIONS=4 \
  taskset -c 0-3 <probe> --rows 3000000 --batch-size 1000000
cargo test -p calc-flow --release --lib grouped_delegation
```
