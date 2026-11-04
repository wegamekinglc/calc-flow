# Input-partitioned `MemTable` experiment (negative result)

> Status: **negative result** for roadmap item `W-SQL-1` (Phase A mechanism
> probe). Recorded for reproducibility; no product default changed and no
> speedup claimed. Maintained-harness integration (Phase B of the original
> plan) is not pursued because the hypothesis was falsified.

## Hypothesis

Each batch input alias was registered as one `MemTable` partition, so row-local
operators (projection/filter/scan) might inherit one partition even when the
DataFusion session targets many partitions. Splitting input record batches into
N partitions was expected to parallelize the scan stage.

## Result: falsified for DataFusion 55 defaults

Two independent observations:

1. **Plan diagnostic** (200,000 rows / 25 record batches, `target_partitions=16`
   fixed; fixed-mode work cap reduces the effective count to 4):
   the same physical plan appears for one and four input partitions:
   `ProjectionExec -> FilterExec -> DataSourceExec: partitions=4`.
   DataFusion's round-robin repartition (`enable_round_robin_repartition`,
   default true) increases scan parallelism to the effective target-partition
   count; the registered `MemTable` partition count is not a ceiling.
2. **Timing probe** (2,000,000 rows / 244 batches, `target_partitions=16`
   fixed, WSL2 release build, 16-core affinity, `TOKIO_WORKER_THREADS=16`,
   seven samples per configuration; descriptive only, not paired AB/BA):
   projection medians 3.30/2.96/3.15/3.32 ms for 1/4/8/16 input partitions;
   grouped `SUM` medians 2.94/3.03/2.86/2.86 ms; every configuration reported
   16 output partitions. Splitting the input did not change latency.

## Interpretation

- The effective target-partition decision (`DataFusionConfig` fixed target or
  `auto`/`auto_rows`) is the SQL-engine parallelism lever, not input-table
  partitioning.
- The roadmap's `W-SQL-1` (MemTable input partitioning) is dropped; its
  expected benefit belongs to making `target_partitions > 1` reachable by
  default/opt-in (`W-SQL-2` `auto_rows`, `W-SQL-3` `active_entities`,
  `W-SQL-4` fast-path decoupling), which remains the SQL-engine priority.
- Keep the probe example so the conclusion can be rechecked on engine upgrades;
  it is benchmark-internal and unreachable from product APIs.

## Reproduction

```bash
cargo run -p calc-flow --release --features benchmark-internals \
  --example input_partitions_probe
```

Environment: WSL2 Ubuntu-26.04, 32 logical CPUs (i9-13900HX), cargo 1.94.0,
DataFusion 55.0.0, Arrow 59.2. Affinity `0-15`, `TOKIO_WORKER_THREADS=16`.
